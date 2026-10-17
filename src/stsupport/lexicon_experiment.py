"""Streaming lexicon construction and tagger-free corpus evaluation commands."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from itertools import islice
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from tqdm import tqdm

from stsupport.config import dump_config, load_config
from stsupport.corpus import iter_corpus
from stsupport.engine import leaf_categories, render_auto
from stsupport.lexicon import GlobalLexicon

logger = logging.getLogger(__name__)


def process_resources(pids=()):
    """Dependency-free Linux RSS/swap diagnostics; CPU time also works elsewhere."""
    result = {"cpu_seconds": time.process_time()}
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as source:
            for line in source:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmSwap"}:
                    result["rss_gib" if key == "VmRSS" else "swap_gib"] = int(value.split()[0]) / (1024 ** 2)
    except (OSError, ValueError):
        pass
    for pid in pids:
        try:
            # /proc/PID/stat fields 14/15, after the parenthesized comm field.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            result["cpu_seconds"] += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmSwap"}:
                    name = "rss_gib" if key == "VmRSS" else "swap_gib"
                    result[name] = result.get(name, 0) + int(value.split()[0]) / (1024 ** 2)
        except (OSError, ValueError, IndexError, AttributeError):
            pass
    return result


def _protect_inputs(inputs, outputs):
    protected = {Path(p).resolve() for p in inputs}
    for path in outputs:
        if Path(path).resolve() in protected:
            raise ValueError(f"output would overwrite an input file: {path}")


def _batches(stream, size):
    if size < 1:
        raise ValueError("batch size must be positive")
    stream = iter(stream)
    while chunk := list(islice(stream, size)):
        yield chunk


def _setup(args):
    from stsupport.cli import _apply_overrides

    cfg = load_config(args.config)
    _apply_overrides(cfg, args.set or [])
    if args.corpus is not None:
        cfg.corpus.path = args.corpus
    if args.limit is not None:
        cfg.corpus.limit = args.limit
    if args.format is not None:
        cfg.corpus.format = args.format
    return cfg


def _stream(cfg):
    return iter_corpus(cfg.corpus.path, fmt=cfg.corpus.format, limit=cfg.corpus.limit,
                       min_tokens=cfg.corpus.min_tokens, drop_quotes=cfg.corpus.drop_quotes)


def cmd_build_lexicon(args):
    from stsupport.supertagger import HolCCGSupertagger

    cfg = _setup(args)
    map_path = Path(args.lexicon).with_suffix(".map.json")
    _protect_inputs([cfg.corpus.path] + ([args.config] if args.config else []),
                    [args.lexicon, map_path, Path(args.lexicon).with_suffix(".summary.json")])
    if args.device:
        cfg.tagger.device = args.device
    if args.k < 1 or not 0 < args.beta <= 1 or not 0 <= args.min_prob <= 1:
        raise ValueError("require k >= 1, 0 < beta <= 1, 0 <= min-prob <= 1")
    tagger = HolCCGSupertagger(
        cfg.tagger.model, device=cfg.tagger.device, grammar_mask=cfg.tagger.grammar_mask,
        hol_ccg_src=cfg.tagger.hol_ccg_src, autocast_bf16=cfg.tagger.autocast_bf16,
    )
    lex = GlobalLexicon(tagger.categories, lowercase=args.lowercase, metadata={
        "source": asdict(cfg.corpus), "tagger": asdict(cfg.tagger),
        "support_selection": {"k": args.k, "beta": args.beta, "min_prob": args.min_prob},
        "aggregation": "arithmetic mean of full licensed per-occurrence distributions",
    })
    t0 = time.perf_counter()
    sentences = 0
    logger.info("building lexicon from %s", cfg.corpus.path)
    for chunk in _batches(_stream(cfg), cfg.tagger.batch_size):
        distributions = tagger.tag(chunk, batch_size=cfg.tagger.batch_size)
        if len(distributions) != len(chunk):
            raise ValueError("tagger returned the wrong number of sentences")
        for tokens, probs in zip(chunk, distributions):
            lex.add(tokens, probs, k=args.k, beta=args.beta, min_prob=args.min_prob, exclude=tagger.excluded_ids)
        sentences += len(chunk)
        if sentences % (cfg.tagger.batch_size * 32) == 0:
            logger.info("aggregated %d sentences, %d word types", sentences, len(lex.counts))
    lex.metadata["sentences"] = sentences
    lex.save(args.lexicon)
    map_path.write_text(json.dumps(lex.category_map(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"sentences": sentences, "tokens": sum(lex.counts.values()), "word_types": len(lex.counts),
               "category_pairs": sum(len(s) for s in lex.support.values()),
               "seconds": round(time.perf_counter() - t0, 3), "lexicon": str(args.lexicon), "map": str(map_path)}
    Path(args.lexicon).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _prepare_record(lex, tokens, max_tokens, min_support, weighting):
    lattice = lex.lattice(tokens, min_support=min_support, weighting=weighting)
    oov = [i for i, w in enumerate(tokens) if lex.key(w) not in lex.counts]
    empty = [i for i, cell in enumerate(lattice) if not cell and i not in oov]
    record = {"tokens": tokens, "status": "fail", "reason": None,
              "oov_positions": oov, "filtered_positions": empty,
              "lattice_sizes": [len(c) for c in lattice], "root_cat": None,
              "final_tags": None, "logprob": None, "auto": None, "root_cell_cats": []}
    if len(tokens) > max_tokens:
        record.update(status="skipped", reason="max_tokens")
    elif oov:
        record["reason"] = "oov"
    elif empty:
        record["reason"] = "empty_support"
    return record, lattice


def _fill_record(record, out):
    if out["ok"]:
        record.update(status="parsed", root_cat=out["root_cat"], final_tags=leaf_categories(out),
                      logprob=float(out["score"]), auto=render_auto(out))
    else:
        record.update(reason="no_derivation", root_cell_cats=list(out["root_cell_cats"]))


def evaluate_processes(grammar, lex, sentences, *, processes=4, max_in_flight=8, prepare_batch_size=64,
                       max_tokens=80, min_support=1, weighting="mean", heartbeat_seconds=10,
                       on_complete=None, on_wait=None, spool_dir=None):
    """Map sentences to private engines; reduce in input order via a disk spool.

    The spool bounds Python result memory without holding up scheduling behind
    a slow early sentence. It is temporary storage, NOT a resume checkpoint.
    """
    from stsupport.engine import load_seen_rules
    from stsupport.process_parser import IndependentParsers

    if processes < 1 or max_in_flight < processes or prepare_batch_size < 1:
        raise ValueError("require processes >= 1, max_in_flight >= processes, prepare_batch_size >= 1")
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be finite and positive")
    seen_rules = load_seen_rules(grammar.seen_rules_path) if grammar.use_seen_rules else None
    source = iter(sentences)
    pool = None
    exhausted = False
    submitted = completed = next_output = eligible = total_sizes = token_count = max_size = 0
    started = time.perf_counter()
    with TemporaryDirectory(prefix="ccg-results-", dir=spool_dir) as temporary:
        with closing(sqlite3.connect(Path(temporary) / "ordered.sqlite", isolation_level=None)) as spool:
            spool.execute("PRAGMA journal_mode=OFF")
            spool.execute("PRAGMA synchronous=OFF")
            spool.execute("CREATE TABLE ready (id INTEGER PRIMARY KEY, record TEXT NOT NULL)")

            def finished(index, record):
                nonlocal completed
                completed += 1
                if on_complete is not None:
                    on_complete(index, record)
                spool.execute("INSERT INTO ready VALUES (?, ?)", (index, json.dumps(record, ensure_ascii=False)))

            try:
                while not exhausted or (pool is not None and pool.pending):
                    # The preparation batch only limits coordinator work per
                    # turn; it never imposes a barrier on running sentences.
                    for _ in range(prepare_batch_size):
                        if exhausted or (pool is not None and len(pool.pending) >= max_in_flight):
                            break
                        try:
                            tokens = next(source)
                        except StopIteration:
                            exhausted = True
                            break
                        index = submitted
                        submitted += 1
                        record, lattice = _prepare_record(lex, tokens, max_tokens, min_support, weighting)
                        sizes = record["lattice_sizes"]
                        total_sizes += sum(sizes)
                        token_count += len(sizes)
                        max_size = max(max_size, max(sizes, default=0))
                        if record["reason"] is not None:
                            finished(index, record)
                        else:
                            eligible += 1
                            if pool is None:
                                pool = IndependentParsers(grammar, seen_rules, processes)
                            pool.submit(index, tokens, lattice, record)
                    if pool is not None:
                        should_wait = pool.pending and (exhausted or len(pool.pending) >= max_in_flight)
                        timeout = min(heartbeat_seconds, 0.2) if should_wait else 0
                        for index, record, out in pool.poll(timeout):
                            _fill_record(record, out)
                            finished(index, record)
                    if on_wait is not None:
                        running = list(pool.busy.values()) if pool is not None else []
                        pending = pool.pending if pool is not None else {}
                        examples = sorted(running, key=lambda i: sum(pending[i]["lattice_sizes"]), reverse=True)[:3]
                        on_wait({"backend": "processes", "elapsed_seconds": time.perf_counter() - started,
                                 "running": len(running), "queued": len(pool.queued) if pool is not None else 0,
                                 "eligible": eligible, "submitted": submitted, "completed": completed,
                                 "buffered_results": completed - next_output,
                                 "avg_candidates": total_sizes / max(1, token_count), "max_candidates": max_size,
                                 "pids": pool.pids if pool is not None else [],
                                 "examples": [{"index": i, "tokens": len(pending[i]["tokens"]),
                                               "candidates": sum(pending[i]["lattice_sizes"])} for i in examples]})
                    while row := spool.execute("SELECT record FROM ready WHERE id=?", (next_output,)).fetchone():
                        spool.execute("DELETE FROM ready WHERE id=?", (next_output,))
                        next_output += 1
                        yield json.loads(row[0])
            finally:
                if pool is not None:
                    pool.close(abort=not exhausted or bool(pool.pending))
                if hasattr(source, "close"):
                    source.close()


def evaluate_batch(engine, lex, sentences, *, max_tokens=80, n_threads=0, min_support=1, weighting="mean",
                   on_complete=None, on_wait=None, heartbeat_seconds=10):
    """Parse full dictionary lattices; never prune or consult a supertagger."""
    records, active, lattices = [], [], []
    for tokens in sentences:
        record, lattice = _prepare_record(lex, tokens, max_tokens, min_support, weighting)
        if record["reason"] is None:
            active.append(len(records))
            lattices.append(lattice)
        records.append(record)
        if record["reason"] is not None and on_complete is not None:
            on_complete(len(records) - 1, record)

    def fill(index, out):
        _fill_record(records[index], out)
        if on_complete is not None:
            on_complete(index, records[index])

    if active:
        if on_complete is None and on_wait is None:
            parsed = engine.parse_batch([sentences[i] for i in active], lattices, n_threads)
            if len(parsed) != len(active):
                raise ValueError("parser returned the wrong number of results")
            for i, out in zip(active, parsed):
                fill(i, out)
        else:
            if heartbeat_seconds <= 0:
                raise ValueError("heartbeat_seconds must be positive")
            threads = min(len(active), n_threads if n_threads > 0 else (os.cpu_count() or 1))
            # parse_raw releases the GIL in C++; each call has its own Parser.
            # The shared engine keeps the same synchronized rule/category caches.
            t0 = time.perf_counter()
            cancellation = engine.new_cancellation_token() if hasattr(engine, "new_cancellation_token") else None
            pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="ccg-parse")
            pending = {}
            try:
                for i, lattice in zip(active, lattices):
                    kwargs = {"cancellation": cancellation} if cancellation is not None else {}
                    pending[pool.submit(engine.parse_raw, sentences[i], lattice, **kwargs)] = i
                sizes = [size for record in records for size in record["lattice_sizes"]]
                while pending:
                    if on_wait is not None:
                        running = [i for future, i in pending.items() if future.running()]
                        examples = sorted(running, key=lambda i: sum(records[i]["lattice_sizes"]), reverse=True)[:3]
                        on_wait({"elapsed_seconds": time.perf_counter() - t0,
                                 "running": len(running), "queued": sum(not f.running() and not f.done() for f in pending),
                                 "eligible": len(active), "batch_sentences": len(sentences),
                                 "max_tokens": max(map(len, sentences)),
                                 "avg_candidates": sum(sizes) / max(1, len(sizes)),
                                 "max_candidates": max(sizes, default=0),
                                 "examples": [{"index": i, "tokens": len(sentences[i]),
                                               "candidates": sum(records[i]["lattice_sizes"])} for i in examples]})
                    # Short waits also let Windows/Python handle Ctrl+C promptly;
                    # CLI heartbeat output has its own interval throttle.
                    done, _ = wait(pending, timeout=min(heartbeat_seconds, 0.2), return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        fill(index, future.result())
            except BaseException:
                # Cancel native loops BEFORE joining the executor. Cancelling
                # Future objects alone cannot stop an already running C++ call.
                if cancellation is not None:
                    cancellation.cancel()
                for future in pending:
                    future.cancel()
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    return records


def cmd_eval_lexicon(args):
    from stsupport.cli import _build_engine

    cfg = _setup(args)
    if cfg.grammar.use_seen_rules and cfg.grammar.seen_rules_path in (None, "model"):
        raise ValueError("tagger-free evaluation needs an explicit grammar.seen_rules_path when use_seen_rules=true")
    if cfg.search.max_tokens < 1 or args.min_support < 1 or args.batch_size < 1:
        raise ValueError("max_tokens, min_support and batch_size must be positive")
    if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be finite and positive")
    max_in_flight = args.max_in_flight if args.max_in_flight is not None else args.processes * 2
    if args.processes < 1 or max_in_flight < args.processes:
        raise ValueError("require processes >= 1 and max-in-flight >= processes")
    lex = GlobalLexicon.load(args.lexicon)
    source = lex.metadata.get("source", {})
    if "drop_quotes" in source and source["drop_quotes"] != cfg.corpus.drop_quotes:
        raise ValueError("corpus.drop_quotes must match the source lexicon preprocessing")
    engine = _build_engine(cfg, None) if args.parser_backend == "threads" else None
    logger.info("[CPU] backend=%s processes=%s threads=%s max_in_flight=%s", args.parser_backend,
                args.processes if engine is None else 0, cfg.search.n_threads if engine is not None else 1,
                max_in_flight if engine is None else args.batch_size)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / cfg.output.name
    _protect_inputs([cfg.corpus.path, args.lexicon] + ([args.config] if args.config else []),
                    [f"{prefix}.{suffix}" for suffix in ("config.yaml", "jsonl", "auto", "summary.json", "oov.json")])
    Path(f"{prefix}.config.yaml").write_text(dump_config(cfg), encoding="utf-8")
    counts, reasons, oov_counts = Counter(), Counter(), Counter()
    types = set()
    candidate_pairs = 0
    live_counts = Counter()
    logger.info("[CPU] Counting target sentences for parsing progress")
    target_total = sum(1 for _ in _stream(cfg))
    t0 = time.perf_counter()
    with tqdm(total=target_total, desc="[CPU] Parse sentences", unit="sent", dynamic_ncols=True,
              mininterval=1, disable=args.no_progress) as bar, \
            Path(f"{prefix}.jsonl").open("w", encoding="utf-8") as jsonl, \
            Path(f"{prefix}.auto").open("w", encoding="utf-8") as auto:
        last_heartbeat = float("-inf")
        last_cpu_sample = (time.monotonic(), time.process_time())
        batch_start = batch_length = 0

        def completed(index, record):
            live_counts[record["status"]] += 1
            bar.set_postfix(parsed=live_counts["parsed"], failed=live_counts["fail"],
                            skipped=live_counts["skipped"], refresh=False)
            bar.update(1)

        def heartbeat(info):
            nonlocal last_heartbeat, last_cpu_sample
            bar.set_postfix(parsed=live_counts["parsed"], failed=live_counts["fail"], skipped=live_counts["skipped"],
                            running=info["running"], queued=info["queued"], max_cands=info["max_candidates"], refresh=False)
            now = time.monotonic()
            if now - last_heartbeat >= args.heartbeat_seconds:
                resources = process_resources(info.get("pids", ()))
                interval = now - last_cpu_sample[0]
                sampled = not info.get("pids") or sys.platform.startswith("linux")
                cores = (resources["cpu_seconds"] - last_cpu_sample[1]) / interval if interval >= 0.1 and sampled else None
                resource_text = f" cpu_cores={cores:.2f}" if cores is not None else ""
                resource_text += "".join(f" {name}={resources[name]:.2f}" for name in ("rss_gib", "swap_gib") if name in resources)
                is_process = info.get("backend") == "processes"
                offset = 0 if is_process else batch_start
                examples = "; ".join(f"sentence.{offset + item['index'] + 1} len={item['tokens']} candidates={item['candidates']}"
                                     for item in info["examples"])
                label = (f"Processes submitted={info['submitted']} completed={info['completed']} "
                         f"buffered_results={info['buffered_results']}" if is_process
                         else f"Batch {batch_start + 1}-{batch_start + batch_length}")
                tqdm.write(f"[CPU] {label}: elapsed={info['elapsed_seconds']:.1f}s "
                           f"running={info['running']} queued={info['queued']} eligible={info['eligible']} "
                           f"avg_cands/token={info['avg_candidates']:.1f} max_cands/token={info['max_candidates']}"
                           f"{resource_text}; {examples}", file=sys.stderr)
                bar.refresh()
                last_heartbeat = now
                last_cpu_sample = (now, resources["cpu_seconds"])

        def threaded_records():
            nonlocal batch_start, batch_length
            for chunk in _batches(_stream(cfg), args.batch_size):
                batch_start, batch_length = counts["sentences"], len(chunk)
                yield from evaluate_batch(engine, lex, chunk, max_tokens=cfg.search.max_tokens,
                                          n_threads=cfg.search.n_threads, min_support=args.min_support, weighting=args.weighting,
                                          on_complete=completed, on_wait=heartbeat, heartbeat_seconds=args.heartbeat_seconds)

        records = (evaluate_processes(cfg.grammar, lex, _stream(cfg), processes=args.processes,
                                     max_in_flight=max_in_flight, prepare_batch_size=args.batch_size,
                                     max_tokens=cfg.search.max_tokens, min_support=args.min_support, weighting=args.weighting,
                                     on_complete=completed, on_wait=heartbeat, heartbeat_seconds=args.heartbeat_seconds,
                                     spool_dir=out_dir)
                   if args.parser_backend == "processes" else threaded_records())
        with closing(records):
            for record in records:
                tokens = record["tokens"]
                counts["sentences"] += 1
                counts["tokens"] += len(tokens)
                counts[record["status"]] += 1
                counts["oov_tokens"] += len(record["oov_positions"])
                counts["available_tokens"] += sum(n > 0 for n in record["lattice_sizes"])
                candidate_pairs += sum(record["lattice_sizes"])
                types.update(lex.key(w) for w in tokens)
                oov_counts.update(lex.key(tokens[i]) for i in record["oov_positions"])
                covered = not record["oov_positions"] and not record["filtered_positions"]
                counts["covered_sentences"] += covered
                counts["covered_eligible"] += covered and record["status"] != "skipped"
                if record["reason"]:
                    reasons[record["reason"]] += 1
                record["id"] = f"sentence.{counts['sentences']}"
                if cfg.output.write_jsonl:
                    jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                if cfg.output.write_auto:
                    auto.write(f"ID={record['id']} PARSER=GLOBALLEXICON STATUS={record['status']} "
                               f"NUMPARSE={int(record['status'] == 'parsed')}\n{record['auto'] or ''}\n")
                if args.no_progress and (counts["sentences"] % args.batch_size == 0 or counts["sentences"] == target_total):
                    logger.info("[CPU] evaluated %d/%d sentences: %d parsed, %d failed, %d skipped", counts["sentences"],
                                target_total, counts["parsed"], counts["fail"], counts["skipped"])
    def ratio(n, d):
        return n / d if d else None

    total, eligible = counts["sentences"], counts["sentences"] - counts["skipped"]
    summary = {
        "sentences": total, "eligible_sentences": eligible, "parsed": counts["parsed"],
        "failed": counts["fail"], "skipped": counts["skipped"], "reasons": dict(reasons),
        "tokens": counts["tokens"], "word_types": len(types), "oov_tokens": counts["oov_tokens"],
        "oov_types": len(oov_counts), "oov_token_rate": ratio(counts["oov_tokens"], counts["tokens"]),
        "word_type_coverage": ratio(len(types) - len(oov_counts), len(types)),
        "token_candidate_coverage": ratio(counts["available_tokens"], counts["tokens"]),
        "covered_sentences": counts["covered_sentences"],
        "sentence_candidate_coverage": ratio(counts["covered_sentences"], total),
        "parse_rate_all": ratio(counts["parsed"], total),
        "parse_rate_eligible": ratio(counts["parsed"], eligible),
        "parse_rate_covered_eligible": ratio(counts["parsed"], counts["covered_eligible"]),
        "avg_candidates_per_token": ratio(candidate_pairs, counts["tokens"]),
        "avg_candidates_per_available_token": ratio(candidate_pairs, counts["available_tokens"]),
        "seconds": round(time.perf_counter() - t0, 3),
        "lexicon": str(Path(args.lexicon).resolve()), "source_metadata": lex.metadata,
        "target_corpus": asdict(cfg.corpus), "grammar": asdict(cfg.grammar),
        "min_support": args.min_support, "weighting": args.weighting, "oov_policy": "strict",
        "all_dictionary_candidates_used": True, "exact_chart": cfg.grammar.cell_beam == 0,
        "max_tokens": cfg.search.max_tokens,
        "n_threads": cfg.search.n_threads, "batch_size": args.batch_size,
        "parser_backend": args.parser_backend,
        "processes": args.processes if args.parser_backend == "processes" else 0,
        "max_in_flight": max_in_flight if args.parser_backend == "processes" else args.batch_size,
        "threads_per_process": 1 if args.parser_backend == "processes" else None,
    }
    Path(f"{prefix}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(f"{prefix}.oov.json").write_text(json.dumps(dict(oov_counts.most_common()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"source_metadata", "grammar", "target_corpus"}}, indent=2))
    return 0


def register_commands(sub):
    for name, func, help_text in (
        ("build-lexicon", cmd_build_lexicon, "tag a source corpus and aggregate a global lexicon"),
        ("eval-lexicon", cmd_eval_lexicon, "evaluate a fixed lexicon on any corpus, without a tagger"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", default=None)
        command.add_argument("--corpus", required=True, help="source corpus for build; target corpus for eval")
        command.add_argument("--format", choices=["auto", "txt", "stories", "jsonl"], default=None)
        command.add_argument("--limit", type=int, default=None)
        command.add_argument("--lexicon", required=True, help="output path for build; input path for eval")
        command.add_argument("--set", action="append", metavar="section.key=value")
        command.set_defaults(func=func)
        if name == "build-lexicon":
            command.add_argument("--k", type=int, default=8, help="per-occurrence candidate cap before taking the union")
            command.add_argument("--beta", type=float, default=0.99, help="per-occurrence cumulative mass threshold")
            command.add_argument("--min-prob", type=float, default=0.0)
            command.add_argument("--lowercase", action="store_true", help="merge case variants in word lookup")
            command.add_argument("--device", default=None)
        else:
            command.add_argument("--no-progress", action="store_true", help="disable CPU parsing tqdm bar")
            command.add_argument("--heartbeat-seconds", type=float, default=10, help="CPU running/queued status interval (default: 10s)")
            command.add_argument("--min-support", type=int, default=1, help="minimum selected occurrences of each word-category pair")
            command.add_argument("--weighting", choices=["mean", "uniform"], default="mean")
            command.add_argument("--batch-size", type=int, default=16, help="thread batch size; process mode: preparation chunk size (no parse barrier)")
            command.add_argument("--parser-backend", choices=["threads", "processes"], default="threads",
                                 help="shared-engine threads (default) or independent-engine processes")
            command.add_argument("--processes", type=int, default=4, help="independent parser processes (default: 4; one native thread each)")
            command.add_argument("--max-in-flight", type=int, default=None, help="maximum queued + running eligible sentences (default: 2 * processes)")
