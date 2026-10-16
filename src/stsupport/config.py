"""Typed configuration (YAML-backed) for the supertagging support pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# C&C type-changing table (UnaryRules.java) and CCGbank type-raising targets
# (trNP / trAP / trPP / trVP_to). A bare ``S`` is the feature variable.
DEFAULT_TYPE_CHANGE_RULES: list[tuple[str, str]] = [
    ("N", "NP"),
    ("NP", "NP/(NP\\NP)"),
    ("S[dcl]/NP", "NP\\NP"),
    ("(S[to]\\NP)/NP", "NP\\NP"),
    ("S[dcl]", "NP\\NP"),
    ("S[dcl]", "S\\S"),
    ("S[dcl]\\NP", "NP\\NP"),
    ("S[pss]\\NP", "NP\\NP"),
    ("S[pss]\\NP", "(S\\NP)\\(S\\NP)"),
    ("S[pss]\\NP", "S/S"),
    ("S[ng]\\NP", "NP\\NP"),
    ("S[ng]\\NP", "(S\\NP)\\(S\\NP)"),
    ("S[ng]\\NP", "S/S"),
    ("S[ng]\\NP", "(S\\NP)/(S\\NP)"),
    ("S[ng]\\NP", "S\\S"),
    ("S[ng]\\NP", "NP"),
    ("S[adj]\\NP", "NP\\NP"),
    ("S[adj]\\NP", "(S\\NP)\\(S\\NP)"),
    ("S[adj]\\NP", "S/S"),
    ("S[to]\\NP", "NP\\NP"),
    ("S[to]\\NP", "(S\\NP)\\(S\\NP)"),
    ("S[to]\\NP", "N\\N"),
    ("S[to]\\NP", "S/S"),
]

DEFAULT_TYPE_RAISE_RULES: list[tuple[str, str]] = [
    ("NP", "S/(S\\NP)"),
    ("NP", "(S\\NP)\\((S\\NP)/NP)"),
    ("NP", "((S\\NP)/NP)\\(((S\\NP)/NP)/NP)"),
    ("NP", "((S\\NP)/(S[to]\\NP))\\(((S\\NP)/(S[to]\\NP))/NP)"),
    ("NP", "((S\\NP)/PP)\\(((S\\NP)/PP)/NP)"),
    ("NP", "((S\\NP)/(S[adj]\\NP))\\(((S\\NP)/(S[adj]\\NP))/NP)"),
    ("NP", "S/(S/NP)"),
    ("S[adj]\\NP", "(S\\NP)\\((S\\NP)/(S[adj]\\NP))"),
    ("PP", "(S\\NP)\\((S\\NP)/PP)"),
    ("S[to]\\NP", "(S\\NP)\\((S\\NP)/(S[to]\\NP))"),
]


@dataclass
class GrammarConfig:
    """Symbolic grammar used by the C++ Viterbi CKY."""

    max_composition_degree: int = 2
    harmonic_backward_composition: bool = True
    crossed_composition: bool = True
    bx_require_vp: bool = True
    comma_as_conj: bool = True
    type_raising: bool = True
    type_changing: bool = True
    max_unary_depth: int = 2
    unary_penalty: float = 0.0
    type_raise_penalty: float = 0.0
    cell_beam: int = 0
    strip_features: list[str] = field(default_factory=lambda: ["nb"])
    punct_atoms: list[str] = field(default_factory=lambda: [",", ".", ":", ";", "LRB", "RRB", "LQU", "RQU"])
    left_punct: list[str] = field(default_factory=lambda: [",", ":", ";", "LRB", "LQU"])
    right_punct: list[str] = field(default_factory=lambda: [",", ".", ":", ";", "RRB", "RQU"])
    root_categories: list[str] = field(default_factory=lambda: ["S"])
    type_change_rules: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_TYPE_CHANGE_RULES))
    type_raise_rules: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_TYPE_RAISE_RULES))
    # Optional treebank "seen rules" filter: path to hol-ccg's
    # grammar_rule_counter.json (or "model" to take it from the model bundle).
    use_seen_rules: bool = False
    seen_rules_path: str | None = "model"

    def to_engine_dict(self, seen_rules: list[tuple[str, str]] | None = None) -> dict[str, Any]:
        d = asdict(self)
        d.pop("seen_rules_path", None)
        d["type_change_rules"] = [tuple(x) for x in self.type_change_rules]
        d["type_raise_rules"] = [tuple(x) for x in self.type_raise_rules]
        d["seen_rules"] = [tuple(x) for x in (seen_rules or [])]
        return d


@dataclass
class SearchConfig:
    """Lattice construction and search strategy. ``k`` and ``beta`` are independent."""

    k: int = 8                    # at most k categories per token
    beta: float = 0.95            # ...or stop once cumulative probability reaches beta
    min_prob: float = 0.0         # drop categories below this probability (0 = off)
    # "direct": 1-best, then one lattice with (k, beta).
    # "progressive": 1-best, then k' = 2, 4, ... up to k (beta fixed), stop at first success.
    strategy: str = "direct"
    n_threads: int = 0            # C++ parser threads for batched lattices (0 = hardware)
    max_tokens: int = 80          # skip longer sentences


@dataclass
class TaggerConfig:
    model: str = "ryosuke-yamaki/hol-ccg-roberta-large"
    device: str = "cuda"
    batch_size: int = 32
    # Apply hol-ccg's candc word mask (drops supertags that are not licensed
    # unary chains). Recommended.
    grammar_mask: bool = True
    # Source checkout of hol-ccg, used when the holccg package is not installed.
    hol_ccg_src: str | None = "../hol-ccg"
    autocast_bf16: bool = False


@dataclass
class CorpusConfig:
    path: str = "data/tinystories_sample.txt"
    limit: int | None = None
    # txt: one raw sentence per line; stories: raw TinyStories text (split into
    # sentences); jsonl: records with a "tokens" (or "text") field; auto: by suffix.
    format: str = "auto"
    min_tokens: int = 2
    # CCGbank has no quotation marks; drop `` '' so the tagger sees familiar input.
    drop_quotes: bool = True


@dataclass
class OutputConfig:
    dir: str = "output"
    name: str = "run"
    write_auto: bool = True
    write_jsonl: bool = True


@dataclass
class Config:
    grammar: GrammarConfig = field(default_factory=GrammarConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    tagger: TaggerConfig = field(default_factory=TaggerConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge(dc: Any, data: dict[str, Any] | None) -> Any:
    if not data:
        return dc
    for key, value in data.items():
        if not hasattr(dc, key):
            raise KeyError(f"Unknown config key {key!r} for {type(dc).__name__}")
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__"):
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config (defaults when ``path`` is None) and apply nested overrides."""

    cfg = Config()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        _merge(cfg, data)
    if overrides:
        _merge(cfg, overrides)
    return cfg


def dump_config(cfg: Config) -> str:
    return yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True)
