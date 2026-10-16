"""Transactional, resumable lexical soft counts for large source corpora."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from stsupport.lexicon import GlobalLexicon


class LexiconStore:
    def __init__(self, path, categories, metadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS entries (
            word TEXT PRIMARY KEY, count INTEGER NOT NULL, soft BLOB NOT NULL,
            support TEXT NOT NULL, pairs INTEGER NOT NULL)""")
        expected = {"version": 1, "categories": list(categories), "metadata": metadata}
        existing = self.db.execute("SELECT value FROM meta WHERE key='spec'").fetchone()
        if existing:
            if json.loads(existing[0]) != expected:
                self.db.close()
                raise ValueError("checkpoint settings differ from this run; use a separate database")
        else:
            with self.db:
                self.db.execute("INSERT INTO meta VALUES ('spec', ?)", (json.dumps(expected),))
                self.db.execute("INSERT INTO meta VALUES ('progress', '{}')")

    @property
    def progress(self):
        return json.loads(self.db.execute("SELECT value FROM meta WHERE key='progress'").fetchone()[0])

    def commit(self, delta: GlobalLexicon, progress):
        """Commit statistics AND corpus offset together, preventing duplicate replay."""
        with self.db:
            for word, count in delta.counts.items():
                previous = self.db.execute("SELECT count, soft, support FROM entries WHERE word=?", (word,)).fetchone()
                soft = delta.soft_counts[word]
                support = delta.support[word].copy()
                if previous:
                    count += previous[0]
                    soft = soft + np.frombuffer(previous[1], dtype="<f8")
                    support.update({int(i): n for i, n in json.loads(previous[2]).items()})
                self.db.execute("INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?)",
                                (word, count, soft.astype("<f8", copy=False).tobytes(),
                                 json.dumps(dict(support), separators=(",", ":")), len(support)))
            self.db.execute("UPDATE meta SET value=? WHERE key='progress'", (json.dumps(progress),))

    def summary(self):
        words, tokens, pairs = self.db.execute("SELECT count(*), coalesce(sum(count),0), coalesce(sum(pairs),0) FROM entries").fetchone()
        return {"word_types": words, "tokens": tokens, "category_pairs": pairs}

    def export_map(self, path):
        categories = json.loads(self.db.execute("SELECT value FROM meta WHERE key='spec'").fetchone()[0])["categories"]
        with Path(path).open("w", encoding="utf-8") as out:
            out.write("{")
            for index, (word, support) in enumerate(self.db.execute("SELECT word,support FROM entries ORDER BY word")):
                if index:
                    out.write(",")
                cats = [categories[i] for i in sorted(map(int, json.loads(support)))]
                out.write("\n" + json.dumps(word, ensure_ascii=False) + ":" + json.dumps(cats, ensure_ascii=False))
            out.write("\n}\n")

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()


class _SoftCounts:
    def __init__(self, db):
        self.db = db

    @lru_cache(maxsize=4096)
    def __getitem__(self, word):
        row = self.db.execute("SELECT soft FROM entries WHERE word=?", (word,)).fetchone()
        if row is None:
            raise KeyError(word)
        return np.frombuffer(row[0], dtype="<f8")


class DiskLexicon(GlobalLexicon):
    """Read-only adapter: vocabulary/support in RAM, dense vectors loaded on demand."""
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        spec = json.loads(self.db.execute("SELECT value FROM meta WHERE key='spec'").fetchone()[0])
        if spec["version"] != 1:
            raise ValueError("unsupported disk lexicon version")
        super().__init__(spec["categories"], lowercase=spec["metadata"].get("lowercase", False), metadata=spec["metadata"])
        self.metadata["progress"] = json.loads(self.db.execute("SELECT value FROM meta WHERE key='progress'").fetchone()[0])
        for word, count, support in self.db.execute("SELECT word,count,support FROM entries"):
            self.counts[word] = count
            self.support[word] = Counter({int(i): n for i, n in json.loads(support).items()})
        self.soft_counts = _SoftCounts(self.db)

    def add(self, *args, **kwargs):
        raise RuntimeError("loaded SQLite lexicons are read-only; use LexiconStore to commit a delta")

    def close(self):
        self.soft_counts.__getitem__.cache_clear()
        self.db.close()
