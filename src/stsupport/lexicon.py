"""Context-free lexical distributions distilled from contextual supertags.

Full soft counts and candidate support are deliberately separate: nonzero
softmax tails do not constitute evidence that a category belongs in the map.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from stsupport.lattice import Lattice, prune_distribution


class GlobalLexicon:
    def __init__(self, categories: Sequence[str], *, lowercase: bool = False, metadata: dict[str, Any] | None = None):
        self.categories = list(categories)
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must be nonempty and unique")
        self.lowercase = lowercase
        self.metadata = dict(metadata or {})
        self.counts: Counter[str] = Counter()
        self.soft_counts: dict[str, np.ndarray] = {}
        self.support: dict[str, Counter[int]] = {}

    def key(self, word: str) -> str:
        return word.lower() if self.lowercase else word

    def add(
        self, tokens: Sequence[str], probs: np.ndarray, *, k: int = 8, beta: float = 0.99,
        min_prob: float = 0.0, exclude: Sequence[int] = (),
    ) -> None:
        if k < 1 or not 0 < beta <= 1 or not 0 <= min_prob <= 1:
            raise ValueError("require k >= 1, 0 < beta <= 1, 0 <= min_prob <= 1")
        p = np.array(probs, dtype=np.float64, copy=True)
        if p.shape != (len(tokens), len(self.categories)):
            raise ValueError("probability matrix must have shape (tokens, categories)")
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("probabilities must be finite and nonnegative")
        if len(exclude):
            p[:, list(exclude)] = 0
        totals = p.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise ValueError("each token must have positive licensed probability mass")
        p /= totals
        for token, row in zip(tokens, p):
            word = self.key(token)
            if word not in self.soft_counts:
                self.soft_counts[word] = np.zeros(len(self.categories), dtype=np.float64)
                self.support[word] = Counter()
            self.counts[word] += 1
            self.soft_counts[word] += row
            self.support[word].update(i for i, _ in prune_distribution(row, k, beta, min_prob=min_prob))

    def category_map(self, *, min_support: int = 1) -> dict[str, list[str]]:
        if min_support < 1:
            raise ValueError("min_support must be >= 1")
        return {
            word: [self.categories[i] for i in sorted(support) if support[i] >= min_support]
            for word, support in sorted(self.support.items())
        }

    def distribution(self, word: str) -> np.ndarray | None:
        """Mean FULL distribution, before support filtering or renormalization."""
        word = self.key(word)
        if word not in self.counts:
            return None
        return self.soft_counts[word] / self.counts[word]

    def candidates(self, word: str, *, min_support: int = 1, weighting: str = "mean") -> list[tuple[str, float]]:
        if min_support < 1:
            raise ValueError("min_support must be >= 1")
        if weighting not in {"mean", "uniform"}:
            raise ValueError("weighting must be mean or uniform")
        word = self.key(word)
        if word not in self.counts:
            return []
        ids = [i for i, n in self.support[word].items() if n >= min_support]
        ids.sort(key=lambda i: (-self.soft_counts[word][i], i))
        if not ids:
            return []
        # Condition the mean distribution on the allowed set. This changes each
        # token's scores by a constant, preserving the best complete derivation.
        mass = float(self.soft_counts[word][ids].sum())
        return [
            (self.categories[i], -math.log(len(ids)) if weighting == "uniform"
             else math.log(float(self.soft_counts[word][i]) / mass))
            for i in ids
        ]

    def lattice(self, tokens: Sequence[str], *, min_support: int = 1, weighting: str = "mean") -> Lattice:
        return [self.candidates(w, min_support=min_support, weighting=weighting) for w in tokens]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write entries incrementally instead of making a second dense copy.
        with path.open("w", encoding="utf-8") as out:
            header = {"version": 1, "categories": self.categories, "lowercase": self.lowercase, "metadata": self.metadata}
            out.write(json.dumps(header, ensure_ascii=False)[:-1] + ', "entries": {')
            for index, word in enumerate(sorted(self.counts)):
                if index:
                    out.write(",")
                entry = {"count": self.counts[word], "soft_counts": self.soft_counts[word].tolist(),
                         "support": dict(sorted(self.support[word].items()))}
                out.write("\n" + json.dumps(word, ensure_ascii=False) + ":" + json.dumps(entry, ensure_ascii=False))
            out.write("\n}}\n")

    @classmethod
    def load(cls, path: str | Path) -> GlobalLexicon:
        if Path(path).suffix in {".sqlite", ".db"}:
            from stsupport.disk_lexicon import DiskLexicon

            return DiskLexicon(path)
        with Path(path).open(encoding="utf-8") as source:
            data = json.load(source)
        if data.get("version") != 1:
            raise ValueError("unsupported lexicon version")
        lex = cls(data["categories"], lowercase=data["lowercase"], metadata=data["metadata"])
        for word, entry in data["entries"].items():
            count = entry["count"]
            sums = np.asarray(entry["soft_counts"], dtype=np.float64)
            support = Counter({int(i): int(n) for i, n in entry["support"].items()})
            if (not isinstance(count, int) or count < 1 or sums.shape != (len(lex.categories),)
                    or not np.isfinite(sums).all() or (sums < 0).any()
                    or not np.isclose(sums.sum(), count)
                    or any(i < 0 or i >= len(lex.categories) or n < 1 or n > count or sums[i] <= 0
                           for i, n in support.items())):
                raise ValueError(f"invalid lexicon entry: {word!r}")
            lex.counts[word], lex.soft_counts[word], lex.support[word] = count, sums, support
        return lex
