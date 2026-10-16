"""1-best check, then top-k / beta lattice Viterbi search."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from stsupport.config import SearchConfig
from stsupport.engine import CKYEngine, leaf_categories, render_auto
from stsupport.lattice import build_lattice, lattice_sizes


@dataclass
class SearchResult:
    status: str                       # "onebest" | "recovered" | "fail" | "skipped"
    tokens: list[str]
    onebest_tags: list[str]
    final_tags: list[str] | None = None
    changed_positions: list[int] = field(default_factory=list)
    root_cat: str | None = None
    logprob: float | None = None       # sum of log supertag probabilities of the derivation
    onebest_logprob: float | None = None
    auto: str | None = None
    lattice_sizes: list[int] = field(default_factory=list)
    k_used: int | None = None
    beta_used: float | None = None
    stages: int = 0                    # number of parser calls
    seconds: float = 0.0
    root_cell_cats: list[str] = field(default_factory=list)  # labels in the full span when failing

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Validator:
    """Runs the search for one sentence given its per-token distribution."""

    def __init__(self, engine: CKYEngine, categories: Sequence[str], search: SearchConfig, *, exclude_ids: Sequence[int] = ()):
        self.engine = engine
        self.categories = list(categories)
        self.search = search
        self.exclude_ids = list(exclude_ids)

    # ------------------------------------------------------------------
    def _schedule(self) -> list[tuple[int, float]]:
        s = self.search
        if s.strategy == "direct":
            return [(s.k, s.beta)]
        if s.strategy == "progressive":
            ks: list[int] = []
            k = 2
            while k < s.k:
                ks.append(k)
                k *= 2
            ks.append(s.k)
            return [(k, s.beta) for k in ks]
        raise ValueError(f"unknown search strategy {s.strategy!r}")

    def _lattice(self, probs: np.ndarray, k: int, beta: float):
        return build_lattice(probs, self.categories, k, beta, min_prob=self.search.min_prob, exclude=self.exclude_ids)

    def validate(self, tokens: Sequence[str], probs: np.ndarray) -> SearchResult:
        t0 = time.perf_counter()
        tokens = [str(t) for t in tokens]
        onebest_lattice = self._lattice(probs, 1, 1.0)
        onebest_tags = [c[0][0] for c in onebest_lattice]
        onebest_lp = float(sum(c[0][1] for c in onebest_lattice))
        result = SearchResult(status="fail", tokens=tokens, onebest_tags=onebest_tags, onebest_logprob=onebest_lp)
        if len(tokens) > self.search.max_tokens:
            result.status = "skipped"
            result.seconds = time.perf_counter() - t0
            return result

        # Stage 1: 1-best (the lattice with k = 1).
        parsed = self.engine.parse_raw(tokens, onebest_lattice)
        result.stages = 1
        if parsed["ok"]:
            self._fill(result, parsed, onebest_lattice, 1, 1.0, "onebest")
            result.seconds = time.perf_counter() - t0
            return result
        result.root_cell_cats = list(parsed["root_cell_cats"])

        # Stage 2+: widen the lattice.
        for k, beta in self._schedule():
            lattice = self._lattice(probs, k, beta)
            parsed = self.engine.parse_raw(tokens, lattice)
            result.stages += 1
            result.lattice_sizes = lattice_sizes(lattice)
            result.k_used, result.beta_used = k, beta
            if parsed["ok"]:
                self._fill(result, parsed, lattice, k, beta, "recovered")
                break
            result.root_cell_cats = list(parsed["root_cell_cats"])
        result.seconds = time.perf_counter() - t0
        return result

    def _fill(self, result: SearchResult, parsed: dict[str, Any], lattice, k: int, beta: float, status: str) -> None:
        result.status = status
        result.final_tags = leaf_categories(parsed)
        result.changed_positions = [i for i, (a, b) in enumerate(zip(result.onebest_tags, result.final_tags)) if a != b]
        result.root_cat = parsed["root_cat"]
        result.logprob = float(parsed["score"])
        result.auto = render_auto(parsed)
        result.lattice_sizes = lattice_sizes(lattice)
        result.k_used, result.beta_used = k, beta

    # ------------------------------------------------------------------
    def validate_batch(self, sentences: Sequence[Sequence[str]], probs_list: Sequence[np.ndarray]) -> list[SearchResult]:
        """Batched variant: all 1-best lattices in one threaded C++ call, then
        every failing sentence is widened together, stage by stage."""

        t0 = time.perf_counter()
        n = len(sentences)
        tokens_list = [[str(t) for t in s] for s in sentences]
        results: list[SearchResult] = []
        onebest_lattices = [self._lattice(p, 1, 1.0) for p in probs_list]
        for tokens, lat in zip(tokens_list, onebest_lattices):
            results.append(
                SearchResult(
                    status="fail",
                    tokens=tokens,
                    onebest_tags=[c[0][0] for c in lat],
                    onebest_logprob=float(sum(c[0][1] for c in lat)),
                )
            )
        active = [i for i in range(n) if len(tokens_list[i]) <= self.search.max_tokens]
        for i in set(range(n)) - set(active):
            results[i].status = "skipped"

        parsed = self.engine.parse_batch([tokens_list[i] for i in active], [onebest_lattices[i] for i in active], self.search.n_threads)
        pending: list[int] = []
        for i, out in zip(active, parsed):
            results[i].stages = 1
            if out["ok"]:
                self._fill(results[i], out, onebest_lattices[i], 1, 1.0, "onebest")
            else:
                results[i].root_cell_cats = list(out["root_cell_cats"])
                pending.append(i)

        for k, beta in self._schedule():
            if not pending:
                break
            lattices = [self._lattice(probs_list[i], k, beta) for i in pending]
            parsed = self.engine.parse_batch([tokens_list[i] for i in pending], lattices, self.search.n_threads)
            still: list[int] = []
            for i, lat, out in zip(pending, lattices, parsed):
                results[i].stages += 1
                results[i].lattice_sizes = lattice_sizes(lat)
                results[i].k_used, results[i].beta_used = k, beta
                if out["ok"]:
                    self._fill(results[i], out, lat, k, beta, "recovered")
                else:
                    results[i].root_cell_cats = list(out["root_cell_cats"])
                    still.append(i)
            pending = still

        elapsed = time.perf_counter() - t0
        for r in results:
            r.seconds = elapsed / max(n, 1)
        return results


def summarize(results: Sequence[SearchResult]) -> dict[str, Any]:
    counts = {"onebest": 0, "recovered": 0, "fail": 0, "skipped": 0}
    changed = 0
    tokens_changed = 0
    tokens_total = 0
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.status == "recovered":
            changed += 1
            tokens_changed += len(r.changed_positions)
            tokens_total += len(r.tokens)
    total = len(results)
    parsed = counts["onebest"] + counts["recovered"]
    return {
        "sentences": total,
        **counts,
        "onebest_rate": counts["onebest"] / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
        "recovered_sentences": changed,
        "avg_changed_tokens_in_recovered": tokens_changed / changed if changed else 0.0,
        "avg_tokens_in_recovered": tokens_total / changed if changed else 0.0,
    }
