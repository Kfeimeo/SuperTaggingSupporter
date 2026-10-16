"""Supertagging support: lattice-constrained validation of CCG supertagger output.

Pipeline
--------
1. A supertagger (hol-ccg's RoBERTa + word classifier) yields, per token, a
   probability distribution over lexical categories.
2. The 1-best tag sequence is checked with a symbolic CKY parser (C++): can it
   derive a sentence category ``S``?
3. If not, each token keeps its categories up to cumulative probability ``beta``
   or at most ``k`` of them, and a Viterbi CKY in Eisner normal form over that
   lattice returns the derivation rooted in ``S`` with the highest product of
   supertag probabilities.
"""

from stsupport.config import Config, GrammarConfig, SearchConfig, load_config
from stsupport.engine import CKYEngine, render_auto
from stsupport.lattice import build_lattice, prune_distribution
from stsupport.lexicon import GlobalLexicon
from stsupport.pipeline import SearchResult, Validator

__all__ = [
    "CKYEngine",
    "Config",
    "GrammarConfig",
    "GlobalLexicon",
    "SearchConfig",
    "SearchResult",
    "Validator",
    "build_lattice",
    "load_config",
    "prune_distribution",
    "render_auto",
]
