"""Portable TinyStories downloader: standard library only, resume, pinned hashes.

Examples:
    python scripts/download_tinystories.py --split train
    python scripts/download_tinystories.py --split all --output-dir /mnt/data/raw
    python scripts/download_tinystories.py --verify-only
"""
from __future__ import annotations

import hashlib
import argparse
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/TinyStories-train.txt?download=true"
SHA256 = "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f"
FILES = {
    "train": {"filename": "TinyStories-train.txt", "bytes": 1924281556, "sha256": SHA256},
    "validation": {"filename": "TinyStories-valid.txt", "bytes": 19447282,
                   "sha256": "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"},
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as src:
        while block := src.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def _verify(path, expected_sha256, expected_size):
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    if path.stat().st_size != expected_size:
        raise ValueError(f"file size mismatch: {path}; expected {expected_size:,} bytes")
    actual = digest(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}; got {actual}; file retained")


def download_file(url, target, expected_sha256, expected_size, *, retries=5, timeout=60,
                  progress_interval=10, verify_only=False):
    """Use Range safely; publish the final filename only after verification."""
    target = Path(target).expanduser().resolve()
    if retries < 0 or timeout <= 0 or progress_interval < 0:
        raise ValueError("require retries >= 0, timeout > 0 and progress_interval >= 0")
    if target.exists() or verify_only:
        _verify(target, expected_sha256, expected_size)
        print(f"verified: {target}", flush=True)
        return target
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(f"partial file exceeds expected size: {partial}; file retained")
    if offset < expected_size:
        for attempt in range(retries + 1):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == expected_size:
                    break
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    append = offset > 0 and response.status == 206
                    if response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                            raise ValueError(f"invalid Content-Range {content_range!r}; partial file retained")
                    elif response.status == 200:
                        if offset:
                            print("server ignored Range; restarting partial download", flush=True)
                        offset = 0
                    else:
                        raise ValueError(f"unexpected HTTP status: {response.status}")
                    last = time.monotonic()
                    with partial.open("ab" if append else "wb") as out:
                        while block := response.read(4 * 1024 * 1024):
                            out.write(block)
                            offset += len(block)
                            if offset > expected_size:
                                raise ValueError("response exceeds expected file size; partial file retained")
                            if time.monotonic() - last >= progress_interval:
                                print(f"{target.name}: {offset:,} / {expected_size:,} bytes ({offset / expected_size:.1%})", flush=True)
                                last = time.monotonic()
                if offset != expected_size:
                    raise OSError(f"incomplete response: {offset:,} / {expected_size:,} bytes")
                break
            except (OSError, http.client.IncompleteRead) as error:
                if attempt == retries:
                    raise
                print(f"download interrupted ({error}); retry {attempt + 1}/{retries}", flush=True)
                time.sleep(min(2 ** attempt, 30))
    _verify(partial, expected_sha256, expected_size)
    partial.replace(target)
    print(f"downloaded and verified: {target}", flush=True)
    return target


def download(root, *, output_dir=None, split="train", retries=5, timeout=60, progress_interval=10, verify_only=False):
    """The original download(root) API continues to return the train file path."""
    if split not in FILES:
        raise ValueError(f"unknown split: {split}")
    spec = FILES[split]
    directory = Path(output_dir).expanduser() if output_dir is not None else Path(root) / "data/raw"
    url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/{spec['filename']}?download=true"
    target = download_file(url, directory / spec["filename"], spec["sha256"], spec["bytes"],
                           retries=retries, timeout=timeout, progress_interval=progress_interval, verify_only=verify_only)
    manifest = {"dataset": "roneneldan/TinyStories", "version": "original", "split": split,
                "revision": REVISION, "url": url, "sha256": spec["sha256"], "bytes": spec["bytes"],
                "path": target.name, "validation_used": False}
    manifest_path = target.with_suffix(".manifest.json")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "validation", "all"], default="train")
    parser.add_argument("--output-dir", type=Path, default=None, help="raw file directory (default: PROJECT/data/raw)")
    parser.add_argument("--verify-only", action="store_true", help="check existing final files without network access")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--progress-interval", type=float, default=10)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        for split in (FILES if args.split == "all" else [args.split]):
            download(root, output_dir=args.output_dir, split=split, retries=args.retries, timeout=args.timeout,
                     progress_interval=args.progress_interval, verify_only=args.verify_only)
    except (OSError, ValueError, http.client.IncompleteRead) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    main()
