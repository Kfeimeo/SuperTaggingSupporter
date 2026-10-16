from __future__ import annotations

import math

import numpy as np

from stsupport.lattice import build_lattice, prune_distribution


def test_k_and_beta_are_independent():
    p = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    assert [i for i, _ in prune_distribution(p, k=1, beta=0.99)] == [0]
    assert [i for i, _ in prune_distribution(p, k=10, beta=0.75)] == [0, 1]       # beta stops first
    assert [i for i, _ in prune_distribution(p, k=3, beta=0.999)] == [0, 1, 2]    # k stops first
    assert [i for i, _ in prune_distribution(p, k=10, beta=1.0)] == [0, 1, 2, 3, 4]


def test_exclude_and_min_prob():
    p = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    assert [i for i, _ in prune_distribution(p, k=10, beta=1.0, exclude=[0])] == [1, 2, 3, 4]
    assert [i for i, _ in prune_distribution(p, k=10, beta=1.0, min_prob=0.08)] == [0, 1, 2]


def test_always_at_least_one():
    p = np.array([0.0, 1.0])
    assert prune_distribution(p, k=1, beta=0.0) == [(1, 1.0)]


def test_build_lattice_logs():
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    lattice = build_lattice(probs, ["A", "B"], k=2, beta=0.75)
    assert lattice[0][0][0] == "A" and math.isclose(lattice[0][0][1], math.log(0.7))
    assert [c for c, _ in lattice[1]] == ["B"]
