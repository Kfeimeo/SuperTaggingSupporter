"""Resume full original TinyStories training-set tagging, then evaluate Brown news.

Every complete group of stories is committed atomically with its byte offset.
Interrupted groups are replayed without duplicating committed soft counts.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm

from download_tinystories import SHA256, download
from stsupport.async_pipeline import LexiconAccumulator, prefetch_one
from stsupport.corpus import sentence_splitter_name, split_sentences, tokenize
from stsupport.disk_lexicon import LexiconStore
from stsupport.lexicon import GlobalLexicon
from stsupport.supertagger import HolCCGSupertagger

MARKER = b"<|endoftext|>"


def run_logged(command, log_path):
    """Keep child progress visible, preserving carriage returns in its log."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with Path(log_path).open("w", encoding="utf-8", newline="") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env) as process:
            with io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace", newline="") as stream:
                while char := stream.read(1):
                    log.write(char)
                    sys.stderr.write(char)
                    if char in "\r\n":
                        log.flush()
                        sys.stderr.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)


def stories(path, offset=0):
    """Yield decoded stories and exact restart offsets, including inline markers."""
    with Path(path).open("rb") as source:
        source.seek(offset)
        pending = []
        while True:
            start = source.tell()
            line = source.readline()
            if not line:
                text = b"".join(pending).decode("utf-8-sig").strip()
                if text:
                    yield text, source.tell()
                return
            while MARKER in line:
                before, _, after = line.partition(MARKER)
                pending.append(before)
                text = b"".join(pending).decode("utf-8-sig").strip()
                start += len(before) + len(MARKER)
                if text:
                    yield text, start
                pending = []
                line = after
            pending.append(line)


def write_status(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepared_groups(path, offset, group_size):
    stream = stories(path, offset)
    try:
        while group := list(islice(stream, group_size)):
            sentences, filtered = [], 0
            for text, _ in group:
                for sentence in split_sentences(text):
                    tokens = tokenize(sentence)
                    if len(tokens) >= 2:
                        sentences.append(tokens)
                    else:
                        filtered += 1
            sentences.sort(key=len)
            yield {"sentences": sentences, "stories": len(group), "offset": group[-1][1], "filtered": filtered}
    finally:
        stream.close()


def tag_safe(tagger, sentences, stats):
    try:
        return tagger.tag(sentences, batch_size=len(sentences), vectorized_pooling=True)
    except Exception as error:
        capacity = "exceeds model subword capacity" in str(error)
        oom = isinstance(error, tagger.torch.cuda.OutOfMemoryError)
        if not capacity and not oom:
            raise
        if oom:
            tagger.torch.cuda.empty_cache()
        reason = "capacity" if capacity else "oom"
    # Retry outside the exception handler so CUDA tensors in traceback are freed.
    if len(sentences) > 1:
        middle = len(sentences) // 2
        stats[f"batch_splits_{reason}"] = stats.get(f"batch_splits_{reason}", 0) + 1
        return tag_safe(tagger, sentences[:middle], stats) + tag_safe(tagger, sentences[middle:], stats)
    if reason == "oom" or len(sentences[0]) < 2:
        raise RuntimeError(f"cannot tag a single sentence: {reason}")
    # All words are retained; exceptional over-capacity sentences use subcontexts.
    stats["long_sentence_chunk_operations"] = stats.get("long_sentence_chunk_operations", 0) + 1
    words = sentences[0]
    middle = len(words) // 2
    return [np.vstack(tag_safe(tagger, [words[:middle]], stats) + tag_safe(tagger, [words[middle:]], stats))]


def benchmark(tagger, path, output):
    sentences = []
    for text, _ in islice(stories(path), 32):
        sentences.extend(tokens for sentence in split_sentences(text) if len(tokens := tokenize(sentence)) >= 2)
    sentences.sort(key=len)
    sentences = sentences[:256]
    # Compare the opt-in vectorized encoder against the unmodified inference path.
    check = sentences[::max(1, len(sentences) // 32)][:32]
    original = tagger.tag(check, batch_size=32)
    fast = tagger.tag(check, batch_size=32, vectorized_pooling=True)
    max_error = max(float(np.max(np.abs(a - b))) for a, b in zip(original, fast))
    top1_agreement = all(np.array_equal(a.argmax(-1), b.argmax(-1)) for a, b in zip(original, fast))
    if max_error > 2e-5 or not top1_agreement:
        raise ValueError(f"vectorized inference validation failed: error={max_error}, top1={top1_agreement}")
    rows = []
    for size in (32, 64, 128):
        t0 = time.perf_counter()
        results = tagger.tag(sentences, batch_size=size, vectorized_pooling=True)
        seconds = time.perf_counter() - t0
        assert len(results) == len(sentences)
        rows.append({"batch_size": size, "sentences": len(sentences), "tokens": sum(map(len, sentences)),
                     "seconds": seconds, "tokens_per_second": sum(map(len, sentences)) / seconds})
    result = {"precision": "float32", "vectorized_max_abs_probability_error": max_error,
              "vectorized_top1_agreement": top1_agreement, "benchmarks": rows}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def cpu_command(corpus, threads=4, batch_size=4, no_progress=False):
    return [sys.executable, "-u", "-m", "stsupport", "eval-lexicon",
            "--lexicon", "output/tinystories_full.lexicon.sqlite", "--corpus", str(corpus),
            "--format", "jsonl", "--set", "output.name=brown_news_full_global_lexicon",
            "--set", f"search.n_threads={threads}", "--batch-size", str(batch_size)] + (["--no-progress"] if no_progress else [])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--parse-threads", type=int, default=4, help="CPU parser threads (default: 4; 0: hardware thread count)")
    parser.add_argument("--parse-batch-size", type=int, default=4, help="target sentences per CPU parser batch (default: 4)")
    parser.add_argument("--pipeline-depth", type=int, default=2, help="maximum CPU aggregation batches in flight (default: 2)")
    parser.add_argument("--async-workers", type=int, default=1, help="CPU aggregation threads during GPU stage (default: 1; plus one prefetch thread)")
    parser.add_argument("--no-async", action="store_true", help="disable CPU prefetch/aggregation overlap for comparison")
    parser.add_argument("--stories-per-checkpoint", type=int, default=128)
    parser.add_argument("--data-dir", type=Path, default=None, help="contains raw/TinyStories-train.txt and brown_news.jsonl")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm bars; retain stage messages and checkpoint logs")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.stories_per_checkpoint < 1 or args.parse_batch_size < 1 or args.pipeline_depth < 1 or args.async_workers < 1:
        parser.error("batch sizes, checkpoint size, pipeline-depth and async-workers must be positive")
    if args.parse_threads < 0:
        parser.error("parse-threads must be >= 0")
    return args


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = (args.data_dir or root / "data").expanduser().resolve()
    os.chdir(root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = root / "output"
    output.mkdir(exist_ok=True)
    from filelock import FileLock
    lock = FileLock(output / "tinystories_full.run.lock")
    lock.acquire(timeout=0)
    state_path = output / "tinystories_full.status.json"
    gpu_bar = None
    groups = None
    try:
        print("[1/4 PREPARE] Verify source data and load GPU model", flush=True)
        if not args.benchmark_only:
            write_status(state_path, {"stage": "verifying_source", "pid": os.getpid()})
        path = download(root, output_dir=data_dir / "raw")
        brown_corpus = data_dir / "brown_news.jsonl"
        if not args.benchmark_only and not brown_corpus.is_file():
            raise FileNotFoundError(f"prepare the target corpus first: python scripts/prepare_brown_news.py --data-dir {data_dir}")
        source_name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        tagger = HolCCGSupertagger()
        tagger.torch.set_num_threads(4)
        if args.benchmark_only:
            benchmark(tagger, path, output / "tinystories_full.benchmark.json")
            return
        metadata = {
            "source": {"path": source_name, "format": "stories", "limit": None,
                       "min_tokens": 2, "drop_quotes": True, "sha256": SHA256,
                       "split": "train", "version": "original"},
            "tagger": {"model": "ryosuke-yamaki/hol-ccg-roberta-large", "device": "cuda",
                       "grammar_mask": True, "autocast_bf16": False, "vectorized_pooling": True,
                       "bundle": str(tagger.bundle_dir.resolve())},
            "lowercase": False, "support_selection": {"k": 8, "beta": 0.99, "min_prob": 0},
            "aggregation": "arithmetic mean of full licensed per-occurrence distributions",
            "long_sentence_policy": "recursively split over-capacity sentences; retain every token; report operations",
            "sentence_splitter": sentence_splitter_name(),
        }
        store = LexiconStore(output / "tinystories_full.lexicon.sqlite", tagger.categories, metadata)
        progress = store.progress
        progress.setdefault("stories", 0)
        progress.setdefault("sentences", 0)
        progress.setdefault("offset", 0)
        progress.setdefault("elapsed_seconds", 0)
        progress.setdefault("filtered_short_sentences", 0)
        groups = prefetch_one(prepared_groups(path, progress["offset"], args.stories_per_checkpoint), enabled=not args.no_async)
        elapsed_base = progress["elapsed_seconds"]
        run_start = time.perf_counter()
        print(f"[2/4 GPU] Supertag + lexicon (CPU overlap={not args.no_async}, async_workers={args.async_workers if not args.no_async else 0}, pipeline_depth={args.pipeline_depth})", flush=True)
        gpu_bar = tqdm(total=path.stat().st_size, initial=progress["offset"],
                       desc="[2/4 GPU] Source processed", unit="B", unit_scale=True,
                       dynamic_ncols=True, mininterval=1, disable=args.no_progress)
        last_gpu_refresh = time.monotonic()
        for group in groups:
            sentences = group["sentences"]
            delta = GlobalLexicon(tagger.categories)
            with LexiconAccumulator(delta, exclude=tagger.excluded_ids, enabled=not args.no_async,
                                    max_pending=args.pipeline_depth, workers=args.async_workers) as accumulator:
                for start in range(0, len(sentences), args.batch_size):
                    batch = sentences[start:start + args.batch_size]
                    distributions = tag_safe(tagger, batch, progress)
                    accumulator.add_batch(batch, distributions)
                    gpu_bar.set_postfix(stories=progress["stories"], sentences=progress["sentences"],
                                        batch=f"{min(start + len(batch), len(sentences))}/{len(sentences)}", refresh=False)
                    if time.monotonic() - last_gpu_refresh >= 1:
                        gpu_bar.refresh()
                        last_gpu_refresh = time.monotonic()
            # Context exit joins CPU aggregation; only then advance the durable offset.
            progress["stories"] += group["stories"]
            progress["sentences"] += len(sentences)
            progress["filtered_short_sentences"] += group["filtered"]
            progress["offset"] = group["offset"]
            progress["elapsed_seconds"] = elapsed_base + time.perf_counter() - run_start
            store.commit(delta, progress)
            current = {**progress, **store.summary(), "stage": "building_lexicon", "pid": os.getpid(),
                       "source_bytes": path.stat().st_size,
                       "source_byte_fraction": progress["offset"] / path.stat().st_size}
            current.update(cpu_gpu_overlap=not args.no_async, pipeline_depth=args.pipeline_depth,
                           async_workers=args.async_workers if not args.no_async else 0)
            if progress["offset"]:
                current["estimated_remaining_hours"] = progress["elapsed_seconds"] / progress["offset"] * (path.stat().st_size - progress["offset"]) / 3600
            write_status(state_path, current)
            gpu_bar.update(progress["offset"] - gpu_bar.n)
            gpu_bar.set_postfix(stories=progress["stories"], sentences=progress["sentences"],
                                words=current["word_types"], refresh=False)
            tqdm.write(json.dumps(current), file=sys.stdout)
        progress["complete"] = True
        progress["offset"] = path.stat().st_size
        store.commit(GlobalLexicon(tagger.categories), progress)
        gpu_bar.update(path.stat().st_size - gpu_bar.n)
        gpu_bar.close()
        gpu_bar = None
        print("[2/4 LEXICON] Save complete lexicon and export category map", flush=True)
        summary = {**progress, **store.summary(), "metadata": metadata}
        (output / "tinystories_full.lexicon.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        store.export_map(output / "tinystories_full.lexicon.map.json")
        store.close()
        del tagger
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        write_status(state_path, {**summary, "stage": "evaluating_brown_news", "pid": os.getpid()})
        print(f"[3/4 CPU] Parse Brown news (threads={args.parse_threads}, batch_size={args.parse_batch_size})", flush=True)
        run_logged(cpu_command(brown_corpus, args.parse_threads, args.parse_batch_size, args.no_progress),
                   output / "brown_news_full_global_lexicon.log")
        # A separate report step can also be rerun independently after completion.
        print("[4/4 REPORT] Generate experiment report", flush=True)
        subprocess.run([sys.executable, "scripts/report_full_tinystories_experiment.py"], check=True)
        write_status(state_path, {**summary, "stage": "complete", "pid": os.getpid()})
        print("[COMPLETE] All experiment stages finished", flush=True)
    except BaseException as error:
        if not args.benchmark_only:
            write_status(state_path, {"stage": "failed", "pid": os.getpid(), "error": repr(error),
                                      "resume": "python -u scripts/run_full_tinystories_experiment.py"})
        raise
    finally:
        if groups is not None:
            groups.close()
        if gpu_bar is not None:
            gpu_bar.close()
        lock.release()


if __name__ == "__main__":
    main()
