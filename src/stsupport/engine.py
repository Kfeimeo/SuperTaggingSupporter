"""Python wrapper around the C++ lattice CKY (``stsupport._ccgcky``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from stsupport.config import GrammarConfig

Lattice = list[list[tuple[str, float]]]


def load_seen_rules(path: str | Path) -> list[tuple[str, str]]:
    """Read (left, right) child pairs from hol-ccg's ``grammar_rule_counter.json``."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pairs = {(r["left"], r["right"]) for r in payload.get("rules", [])}
    return sorted(pairs)


class CKYEngine:
    """Viterbi CKY over supertag lattices with CCG combinators and Eisner NF."""

    def __init__(self, grammar: GrammarConfig | None = None, seen_rules: Sequence[tuple[str, str]] | None = None):
        from stsupport import _ccgcky  # compiled extension

        self.grammar = grammar or GrammarConfig()
        self._engine = _ccgcky.Engine(self.grammar.to_engine_dict(list(seen_rules or [])))

    # -- parsing ---------------------------------------------------------
    def parse(self, tokens: Sequence[str], lattice: Lattice) -> dict[str, Any] | None:
        """Return the best S-rooted derivation or ``None`` when the lattice admits none."""

        out = self._engine.parse(list(tokens), lattice)
        return out if out["ok"] else None

    def parse_raw(self, tokens: Sequence[str], lattice: Lattice) -> dict[str, Any]:
        return self._engine.parse(list(tokens), lattice)

    def parse_batch(self, tokens: Sequence[Sequence[str]], lattices: Sequence[Lattice], n_threads: int = 0) -> list[dict[str, Any]]:
        return self._engine.parse_batch([list(t) for t in tokens], list(lattices), n_threads)

    # -- grammar introspection -------------------------------------------
    def normalize(self, category: str) -> str:
        return self._engine.normalize(category)

    def is_valid_supertag(self, supertag: str) -> bool:
        return self._engine.is_valid(supertag)

    def combine(self, left: str, right: str) -> list[tuple[str, str, int]]:
        return self._engine.combine(left, right)

    def unary(self, category: str) -> list[tuple[str, str]]:
        return self._engine.unary(category)

    def is_root(self, category: str) -> bool:
        return self._engine.is_root(category)


# ---------------------------------------------------------------------------
# Derivation rendering
# ---------------------------------------------------------------------------

def render_auto(result: dict[str, Any], pos: str = "POS") -> str:
    """Render a parse result as a CCGbank-style ``.auto`` derivation string."""

    nodes = result["nodes"]

    def rec(idx: int) -> str:
        node = nodes[idx]
        children = node["children"]
        if not children:
            cat = node["cat"]
            return f"(<L {cat} {pos} {pos} {node['word']} {cat}>)"
        if len(children) == 1:
            return f"(<T {node['cat']} 0 1> {rec(children[0])} )"
        return f"(<T {node['cat']} {node['head']} 2> {rec(children[0])} {rec(children[1])} )"

    return rec(result["root"])


def render_bracketed(result: dict[str, Any]) -> str:
    """Compact human-readable bracketing: ``[S[dcl] [NP the dog] [S[dcl]\\NP barks]]``."""

    nodes = result["nodes"]

    def rec(idx: int) -> str:
        node = nodes[idx]
        if not node["children"]:
            return f"{node['word']}|{node['cat']}"
        inner = " ".join(rec(c) for c in node["children"])
        return f"[{node['cat']}:{node['rule']} {inner}]"

    return rec(result["root"])


def leaf_categories(result: dict[str, Any]) -> list[str]:
    """Per-token supertag strings used by the derivation (in token order)."""

    leaves = [n for n in result["nodes"] if not n["children"]]
    leaves.sort(key=lambda n: n["token"])
    return [n["supertag"] for n in leaves]
