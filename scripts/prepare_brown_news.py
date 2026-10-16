"""Prepare all Brown news sentences with the project's PTB preprocessing.

Downloads the official NLTK archive into the workspace, verifies its SHA-256,
and writes tokenized JSONL plus a provenance manifest. No model is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from download_tinystories import download_file


URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"
SHA256 = "9b275f9b3b95d7bd66ccfb7cd259f445a13bbe5d1f4107aba09fd3e8364bafa6"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="data directory (default: PROJECT/data)")
    parser.add_argument("--download-only", action="store_true", help="download verified archive without NLTK or export")
    parser.add_argument("--verify-only", action="store_true", help="verify existing archive offline without export")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    directory = (args.data_dir or root / "data").expanduser().resolve()
    archive = download_file(URL, directory / "raw/brown.zip", SHA256, 3314357,
                            retries=args.retries, timeout=args.timeout, verify_only=args.verify_only)
    if args.download_only or args.verify_only:
        return 0
    import nltk
    from nltk.corpus.reader import CategorizedTaggedCorpusReader
    from nltk.data import ZipFilePathPointer
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    from stsupport.corpus import tokenize

    # Register only this corpus directory with NLTK's data-path checks.
    nltk.data.path.insert(0, str(archive.parent))
    brown = CategorizedTaggedCorpusReader(
        ZipFilePathPointer(str(archive), "brown/"), r"c[a-z]\d\d", cat_file="cats.txt",
        tagset="brown", encoding="ascii",
    )
    path = directory / "brown_news.jsonl"
    detokenizer = TreebankWordDetokenizer()
    documents, raw_sentences, sentences, tokens, dropped = [], 0, 0, 0, 0
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for fileid in sorted(brown.fileids(categories="news")):
            documents.append(fileid)
            for index, words in enumerate(brown.sents(fileids=[fileid]), 1):
                raw_sentences += 1
                text = detokenizer.detokenize(list(words))
                ptb_tokens = tokenize(text, drop_quotes=True)
                if len(ptb_tokens) < 2:
                    dropped += 1
                    continue
                record = {"id": f"brown.news.{fileid}.{index}", "document": fileid,
                          "sentence_in_document": index, "tokens": ptb_tokens}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                sentences += 1
                tokens += len(ptb_tokens)
    manifest = {
        "corpus": "Brown", "category": "news", "selection": "all news documents; no sampling",
        "archive_url": URL, "archive_sha256": SHA256,
        "metadata_url": "https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/brown.xml",
        "license_in_nltk_metadata": "May be used for non-commercial purposes.",
        "documents": documents, "document_count": len(documents),
        "raw_sentences": raw_sentences, "sentences": sentences, "tokens": tokens,
        "dropped_under_two_tokens": dropped,
        "preprocessing": "keep Brown sentence boundaries; TreebankWordDetokenizer then stsupport.corpus.tokenize(drop_quotes=True); preserve case",
        "output": path.name, "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    main()
