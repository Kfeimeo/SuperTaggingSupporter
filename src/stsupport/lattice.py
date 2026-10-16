"""Top-k / cumulative-beta pruning of per-token category distributions."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Lattice = list[list[tuple[str, float]]]


def prune_distribution(
    probs: np.ndarray,
    k: int,
    beta: float,
    *,
    min_prob: float = 0.0,
    exclude: Sequence[int] | None = None,
) -> list[tuple[int, float]]:
    """Keep categories in descending probability until the cumulative mass
    reaches ``beta`` or ``k`` categories are kept, whichever comes first.

    ``k`` and ``beta`` are independent limits: ``k=1`` is the 1-best tag,
    ``beta>=1`` with large ``k`` keeps up to ``k`` categories regardless of mass.
    At least one category is always returned. Returns ``(index, prob)`` pairs.
    """

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    p = np.asarray(probs, dtype=np.float64)
    if exclude:
        p = p.copy()
        p[list(exclude)] = 0.0
    order = np.argsort(-p, kind="stable")
    kept: list[tuple[int, float]] = []
    cum = 0.0
    for idx in order[: max(k, 1)]:
        prob = float(p[idx])
        if kept and (cum >= beta or prob < min_prob or prob <= 0.0):
            break
        kept.append((int(idx), prob))
        cum += prob
        if cum >= beta:
            break
    return kept


def build_lattice(
    probs: np.ndarray,
    categories: Sequence[str],
    k: int,
    beta: float,
    *,
    min_prob: float = 0.0,
    exclude: Sequence[int] | None = None,
) -> Lattice:
    """Per-token ``[(category_string, log_prob), ...]`` lists for the C++ engine.

    ``probs`` has shape ``(n_tokens, n_categories)``.
    """

    lattice: Lattice = []
    for row in probs:
        kept = prune_distribution(row, k, beta, min_prob=min_prob, exclude=exclude)
        lattice.append([(categories[idx], math.log(max(prob, 1e-300))) for idx, prob in kept])
    return lattice


def lattice_sizes(lattice: Lattice) -> list[int]:
    return [len(c) for c in lattice]
