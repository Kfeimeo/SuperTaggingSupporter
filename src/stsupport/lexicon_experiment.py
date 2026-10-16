"""Streaming lexicon construction and tagger-free corpus evaluation commands."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from tqdm import tqdm

from stsupport.config import dump_config, load_config
from stsupport.corpus import iter_corpus
from stsupport.engine import leaf_categories, render_auto
from stsupport.lexicon import GlobalLexicon

logger = logging.getLogger(__name__)


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


def evaluate_batch(engine, lex, sentences, *, max_tokens=80, n_threads=0, min_support=1, weighting="mean"):
    """Parse full dictionary lattices; never prune or consult a supertagger."""
    records, active, lattices = [], [], []
    for tokens in sentences:
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
        else:
            active.append(len(records))
            lattices.append(lattice)
        records.append(record)
    if active:
        parsed = engine.parse_batch([sentences[i] for i in active], lattices, n_threads)
        if len(parsed) != len(active):
            raise ValueError("parser returned the wrong number of results")
        for i, out in zip(active, parsed):
            if out["ok"]:
                records[i].update(status="parsed", root_cat=out["root_cat"], final_tags=leaf_categories(out),
                                  logprob=float(out["score"]), auto=render_auto(out))
            else:
                records[i].update(reason="no_derivation", root_cell_cats=list(out["root_cell_cats"]))
    return records


def cmd_eval_lexicon(args):
    from stsupport.cli import _build_engine

    cfg = _setup(args)
    if cfg.grammar.use_seen_rules and cfg.grammar.seen_rules_path in (None, "model"):
        raise ValueError("tagger-free evaluation needs an explicit grammar.seen_rules_path when use_seen_rules=true")
    if cfg.search.max_tokens < 1 or args.min_support < 1:
        raise ValueError("max_tokens and min_support must be positive")
    lex = GlobalLexicon.load(args.lexicon)
    source = lex.metadata.get("source", {})
    if "drop_quotes" in source and source["drop_quotes"] != cfg.corpus.drop_quotes:
        raise ValueError("corpus.drop_quotes must match the source lexicon preprocessing")
    engine = _build_engine(cfg, None)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / cfg.output.name
    _protect_inputs([cfg.corpus.path, args.lexicon] + ([args.config] if args.config else []),
                    [f"{prefix}.{suffix}" for suffix in ("config.yaml", "jsonl", "auto", "summary.json", "oov.json")])
    Path(f"{prefix}.config.yaml").write_text(dump_config(cfg), encoding="utf-8")
    counts, reasons, oov_counts = Counter(), Counter(), Counter()
    types = set()
    candidate_pairs = 0
    logger.info("[CPU] Counting target sentences for parsing progress")
    target_total = sum(1 for _ in _stream(cfg))
    t0 = time.perf_counter()
    with tqdm(total=target_total, desc="[CPU] Parse sentences", unit="sent", dynamic_ncols=True,
              mininterval=1, disable=args.no_progress) as bar, \
            Path(f"{prefix}.jsonl").open("w", encoding="utf-8") as jsonl, \
            Path(f"{prefix}.auto").open("w", encoding="utf-8") as auto:
        for chunk in _batches(_stream(cfg), args.batch_size):
            records = evaluate_batch(engine, lex, chunk, max_tokens=cfg.search.max_tokens,
                                     n_threads=cfg.search.n_threads, min_support=args.min_support, weighting=args.weighting)
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
            bar.set_postfix(parsed=counts["parsed"], failed=counts["fail"], skipped=counts["skipped"], refresh=False)
            bar.update(len(records))
            if args.no_progress:
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
            command.add_argument("--min-support", type=int, default=1, help="minimum selected occurrences of each word-category pair")
            command.add_argument("--weighting", choices=["mean", "uniform"], default="mean")
            command.add_argument("--batch-size", type=int, default=16, help="sentences per parser batch")
