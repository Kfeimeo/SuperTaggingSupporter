"""CCG supertag validation and global-lexicon coverage experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from stsupport.config import Config, dump_config, load_config

logger = logging.getLogger("stsupport")


def _parse_override(text: str) -> tuple[list[str], Any]:
    key, _, raw = text.partition("=")
    if not _:
        raise argparse.ArgumentTypeError(f"override must look like section.key=value, got {text!r}")
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.split("."), value


def _apply_overrides(cfg: Config, overrides: list[str]) -> None:
    for text in overrides:
        path, value = _parse_override(text)
        target: Any = cfg
        for part in path[:-1]:
            target = getattr(target, part)
        if not hasattr(target, path[-1]):
            raise SystemExit(f"unknown config key: {'.'.join(path)}")
        setattr(target, path[-1], value)


def _build_engine(cfg: Config, bundle_dir: Path | None):
    from stsupport.engine import CKYEngine, load_seen_rules

    seen = None
    if cfg.grammar.use_seen_rules:
        spec = cfg.grammar.seen_rules_path or "model"
        path = (bundle_dir / "grammar_rule_counter.json") if spec == "model" and bundle_dir else Path(spec)
        seen = load_seen_rules(path)
        logger.info("seen-rules filter: %d child pairs from %s", len(seen), path)
    return CKYEngine(cfg.grammar, seen)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    _apply_overrides(cfg, args.set or [])
    if args.corpus:
        cfg.corpus.path = args.corpus
    if args.limit is not None:
        cfg.corpus.limit = args.limit
    if args.k is not None:
        cfg.search.k = args.k
    if args.beta is not None:
        cfg.search.beta = args.beta
    if args.device:
        cfg.tagger.device = args.device

    from tqdm import tqdm

    from stsupport.pipeline import Validator, summarize

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.output.name}.config.yaml").write_text(dump_config(cfg), encoding="utf-8")

    sentences, tagger = _load_tagger_and_corpus(cfg)
    engine = _build_engine(cfg, tagger.bundle_dir)
    validator = Validator(engine, tagger.categories, cfg.search, exclude_ids=tagger.excluded_ids)

    jsonl_path = out_dir / f"{cfg.output.name}.jsonl"
    auto_path = out_dir / f"{cfg.output.name}.auto"
    results = []
    t0 = time.perf_counter()
    tag_seconds = 0.0
    parse_seconds = 0.0
    with jsonl_path.open("w", encoding="utf-8") as jsonl, auto_path.open("w", encoding="utf-8", newline="\n") as auto:
        bs = cfg.tagger.batch_size
        for start in tqdm(range(0, len(sentences), bs), desc="batches", disable=not sys.stderr.isatty()):
            chunk = sentences[start : start + bs]
            t1 = time.perf_counter()
            probs = tagger.tag(chunk, batch_size=bs)
            tag_seconds += time.perf_counter() - t1
            t2 = time.perf_counter()
            batch_results = validator.validate_batch(chunk, probs)
            parse_seconds += time.perf_counter() - t2
            for offset, res in enumerate(batch_results):
                idx = start + offset + 1
                record = {"id": f"sentence.{idx}", **res.to_dict()}
                if cfg.output.write_jsonl:
                    jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                if cfg.output.write_auto:
                    auto.write(f"ID=sentence.{idx} PARSER=STSUPPORT STATUS={res.status} NUMPARSE=1\n")
                    auto.write((res.auto or "") + "\n")
                results.append(res)

    summary = summarize(results)
    summary.update(
        {
            "k": cfg.search.k,
            "beta": cfg.search.beta,
            "strategy": cfg.search.strategy,
            "tagging_seconds": round(tag_seconds, 2),
            "parsing_seconds": round(parse_seconds, 2),
            "total_seconds": round(time.perf_counter() - t0, 2),
        }
    )
    (out_dir / f"{cfg.output.name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _load_tagger_and_corpus(cfg: Config):
    from stsupport.corpus import load_corpus
    from stsupport.supertagger import HolCCGSupertagger

    logger.info("loading corpus %s", cfg.corpus.path)
    sentences = load_corpus(
        cfg.corpus.path,
        fmt=cfg.corpus.format,
        limit=cfg.corpus.limit,
        min_tokens=cfg.corpus.min_tokens,
        drop_quotes=cfg.corpus.drop_quotes,
    )
    logger.info("%d sentences", len(sentences))
    logger.info("loading supertagger %s on %s", cfg.tagger.model, cfg.tagger.device)
    tagger = HolCCGSupertagger(
        cfg.tagger.model,
        device=cfg.tagger.device,
        grammar_mask=cfg.tagger.grammar_mask,
        hol_ccg_src=cfg.tagger.hol_ccg_src,
        autocast_bf16=cfg.tagger.autocast_bf16,
    )
    return sentences, tagger


def cmd_sweep(args: argparse.Namespace) -> int:
    """Tag the corpus once, then evaluate every (k, beta) combination."""

    import copy
    import itertools

    cfg = load_config(args.config)
    _apply_overrides(cfg, args.set or [])
    if args.corpus:
        cfg.corpus.path = args.corpus
    if args.limit is not None:
        cfg.corpus.limit = args.limit

    from stsupport.pipeline import Validator, summarize

    sentences, tagger = _load_tagger_and_corpus(cfg)
    engine = _build_engine(cfg, tagger.bundle_dir)

    t0 = time.perf_counter()
    probs = []
    for start in range(0, len(sentences), cfg.tagger.batch_size):
        probs.extend(tagger.tag(sentences[start : start + cfg.tagger.batch_size], batch_size=cfg.tagger.batch_size))
    logger.info("tagging: %.1fs", time.perf_counter() - t0)

    ks = [int(x) for x in args.k.split(",")]
    betas = [float(x) for x in args.beta.split(",")]
    rows = []
    for k, beta in itertools.product(ks, betas):
        search = copy.deepcopy(cfg.search)
        search.k, search.beta = k, beta
        validator = Validator(engine, tagger.categories, search, exclude_ids=tagger.excluded_ids)
        t1 = time.perf_counter()
        results = []
        for start in range(0, len(sentences), 256):
            results.extend(validator.validate_batch(sentences[start : start + 256], probs[start : start + 256]))
        seconds = time.perf_counter() - t1
        s = summarize(results)
        avg_cands = sum(sum(r.lattice_sizes) for r in results if r.lattice_sizes) / max(
            1, sum(len(r.lattice_sizes) for r in results if r.lattice_sizes)
        )
        rows.append({"k": k, "beta": beta, **s, "avg_cands_per_token": round(avg_cands, 2), "parse_seconds": round(seconds, 2)})

    header = f"{'k':>4} {'beta':>6} {'onebest':>8} {'recov':>6} {'fail':>5} {'parse_rate':>10} {'cands/tok':>9} {'sec':>6}"
    print(header)
    for r in rows:
        print(
            f"{r['k']:>4} {r['beta']:>6.3f} {r['onebest']:>8} {r['recovered']:>6} {r['fail']:>5} "
            f"{r['parse_rate']:>10.3f} {r['avg_cands_per_token']:>9} {r['parse_seconds']:>6}"
        )
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.output.name}.sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from stsupport.corpus import prepare_sentences

    count = prepare_sentences(args.src, args.dst, limit=args.limit)
    print(f"wrote {count} sentences to {args.dst}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Parse a hand-written lattice: ``word|CAT:p|CAT:p ...`` tokens."""

    import math

    cfg = load_config(args.config)
    _apply_overrides(cfg, args.set or [])
    from stsupport.engine import CKYEngine, render_auto, render_bracketed

    engine = CKYEngine(cfg.grammar)
    tokens: list[str] = []
    lattice = []
    for item in args.items:
        word, *specs = item.split("|")
        if not specs:
            raise SystemExit(f"token {item!r} has no category; use word|CAT:p|CAT:p")
        tokens.append(word)
        cands = []
        for spec in specs:
            cat, _, p = spec.rpartition(":") if ":" in spec else (spec, "", "")
            cands.append((cat, math.log(float(p)) if p else 0.0))
        lattice.append(cands)
    out = engine.parse_raw(tokens, lattice)
    if not out["ok"]:
        print("FAIL; full-span categories:", out["root_cell_cats"])
        return 1
    print("root:", out["root_cat"], "logprob:", round(out["score"], 4))
    print(render_bracketed(out))
    print(render_auto(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stsupport", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="tag + validate a corpus")
    run.add_argument("--config", default=None, help="YAML config (defaults when omitted)")
    run.add_argument("--corpus", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--k", type=int, default=None)
    run.add_argument("--beta", type=float, default=None)
    run.add_argument("--device", default=None)
    run.add_argument("--set", action="append", metavar="section.key=value", help="override any config value (JSON literal)")
    run.set_defaults(func=cmd_run)

    sweep = sub.add_parser("sweep", help="tag once, then evaluate several k / beta settings")
    sweep.add_argument("--config", default=None)
    sweep.add_argument("--corpus", default=None)
    sweep.add_argument("--limit", type=int, default=None)
    sweep.add_argument("--k", default="1,2,4,8,16", help="comma-separated k values")
    sweep.add_argument("--beta", default="0.95", help="comma-separated beta values")
    sweep.add_argument("--set", action="append", metavar="section.key=value")
    sweep.set_defaults(func=cmd_sweep)

    prep = sub.add_parser("prepare-corpus", help="split raw TinyStories text into one sentence per line")
    prep.add_argument("src")
    prep.add_argument("dst")
    prep.add_argument("--limit", type=int, default=None)
    prep.set_defaults(func=cmd_prepare)

    check = sub.add_parser("check", help="parse a hand-written lattice, e.g. 'the|NP[nb]/N:0.9 dog|N:0.8|NP:0.1 barks|S[dcl]\\NP:0.7'")
    check.add_argument("--config", default=None)
    check.add_argument("--set", action="append", metavar="section.key=value")
    check.add_argument("items", nargs="+")
    check.set_defaults(func=cmd_check)

    from stsupport.lexicon_experiment import register_commands

    register_commands(sub)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
