"""Corpus loading: TinyStories (raw stories or one sentence per line) -> PTB tokens."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

_END_OF_TEXT = "<|endoftext|>"
# Sentence end: terminal punctuation, optional closing quotes/brackets, then
# whitespace followed by an (optionally quoted) capital or digit.
_SENT_END = re.compile(r"[.!?][\"')]*(?=\s+[\"'(]?[A-Z0-9])")
_PTB_MAP = {"(": "-LRB-", ")": "-RRB-", "{": "-LCB-", "}": "-RCB-", "[": "-LSB-", "]": "-RSB-"}


def _regex_split(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _SENT_END.finditer(text):
        end = match.end()
        sentences.append(text[start:end])
        start = end
    sentences.append(text[start:])
    return [s.strip() for s in sentences if s.strip()]


@lru_cache(maxsize=1)
def _sentence_tokenizer():
    """Resolve once per process; repeated failed resource searches are expensive."""
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
        return nltk.sent_tokenize
    except Exception:
        return None


def sentence_splitter_name() -> str:
    return "nltk_punkt" if _sentence_tokenizer() is not None else "regex"


def split_sentences(text: str) -> list[str]:
    """Sentence-split a paragraph. Uses NLTK punkt when its data is present,
    otherwise a conservative regex splitter."""

    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = " ".join(text.split())
    try:
        tokenizer = _sentence_tokenizer()
        if tokenizer is not None:
            return [s.strip() for s in tokenizer(text) if s.strip()]
    except Exception:
        return _regex_split(text)
    return _regex_split(text)


_QUOTE_TOKENS = {"``", "''", '"', "“", "”"}


def tokenize(sentence: str, *, drop_quotes: bool = True) -> list[str]:
    """PTB-style tokenization (CCGbank conventions: -LRB-/-RRB-).

    CCGbank contains no quotation marks (they were removed from the WSJ text),
    so a CCGbank-trained supertagger has never seen `` or ''. By default they
    are dropped here as well.
    """

    from nltk.tokenize import TreebankWordTokenizer

    sentence = sentence.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    tokens = TreebankWordTokenizer().tokenize(sentence)
    tokens = [_PTB_MAP.get(t, t) for t in tokens]
    if drop_quotes:
        tokens = [t for t in tokens if t not in _QUOTE_TOKENS]
    return tokens


def iter_story_sentences(path: str | Path) -> Iterator[str]:
    """Raw TinyStories dump (stories separated by <|endoftext|>) -> sentences."""

    buffer: list[str] = []
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if _END_OF_TEXT in line:
                before, _, after = line.partition(_END_OF_TEXT)
                buffer.append(before)
                yield from split_sentences(" ".join(buffer))
                buffer = [after]
            else:
                buffer.append(line)
    if buffer:
        yield from split_sentences(" ".join(buffer))


def iter_line_sentences(path: str | Path) -> Iterator[str]:
    """One sentence per line; lines that still contain several sentences
    (e.g. ``... play ! '' And so , ...``) are split again."""

    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line and _END_OF_TEXT not in line:
                yield from split_sentences(line)


def iter_jsonl_tokens(path: str | Path, *, drop_quotes: bool = True) -> Iterator[list[str]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "tokens" in record:
                tokens = [str(t) for t in record["tokens"]]
                yield [t for t in tokens if t not in _QUOTE_TOKENS] if drop_quotes else tokens
            elif "text" in record:
                yield tokenize(record["text"], drop_quotes=drop_quotes)


def iter_corpus(
    path: str | Path,
    *,
    fmt: str = "auto",
    limit: int | None = None,
    min_tokens: int = 2,
    drop_quotes: bool = True,
) -> Iterator[list[str]]:
    """Stream tokenized sentences, with the same preprocessing for both corpora."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if limit == 0:
        return

    path = Path(path)
    if fmt == "auto":
        if path.suffix == ".jsonl":
            fmt = "jsonl"
        else:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                head = handle.read(200_000)
            fmt = "stories" if _END_OF_TEXT in head or "\n\n" in head else "txt"

    if fmt == "jsonl":
        stream: Iterable[list[str]] = iter_jsonl_tokens(path, drop_quotes=drop_quotes)
    elif fmt == "stories":
        stream = (tokenize(s, drop_quotes=drop_quotes) for s in iter_story_sentences(path))
    elif fmt == "txt":
        stream = (tokenize(s, drop_quotes=drop_quotes) for s in iter_line_sentences(path))
    else:
        raise ValueError(f"unknown corpus format {fmt!r}")

    count = 0
    for tokens in stream:
        if len(tokens) < min_tokens:
            continue
        yield tokens
        count += 1
        if limit is not None and count >= limit:
            break


def load_corpus(
    path: str | Path,
    *,
    fmt: str = "auto",
    limit: int | None = None,
    min_tokens: int = 2,
    drop_quotes: bool = True,
) -> list[list[str]]:
    """Return tokenized sentences from a TinyStories-style file."""
    return list(iter_corpus(path, fmt=fmt, limit=limit, min_tokens=min_tokens, drop_quotes=drop_quotes))


def prepare_sentences(src: str | Path, dst: str | Path, *, limit: int | None = None) -> int:
    """Split a raw TinyStories dump into one sentence per line."""

    count = 0
    with Path(dst).open("w", encoding="utf-8", newline="\n") as out:
        for sentence in iter_story_sentences(src):
            out.write(sentence + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count
