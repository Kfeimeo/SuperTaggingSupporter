"""Grammar/engine tests. No model required."""

from __future__ import annotations

import math

import pytest

from stsupport.config import GrammarConfig
from stsupport.engine import CKYEngine, leaf_categories, render_auto


@pytest.fixture(scope="module")
def engine() -> CKYEngine:
    return CKYEngine(GrammarConfig())


def lat(*cands: list[tuple[str, float]]):
    return [[(c, math.log(p)) for c, p in cell] for cell in cands]


# --- categories -------------------------------------------------------------

def test_normalize_strips_nb_and_renders_canonically(engine):
    assert engine.normalize("NP[nb]/N") == "NP/N"
    assert engine.normalize("(S[dcl]\\NP)/NP") == "(S[dcl]\\NP)/NP"
    assert engine.normalize("S[X]/(S[X]\\NP)") == "S/(S\\NP)"
    assert engine.normalize("S[dcl]\\NP[conj]") == "S[dcl]\\NP[conj]"


def test_invalid_supertags_are_rejected(engine):
    assert not engine.is_valid_supertag("<unk>")
    assert not engine.is_valid_supertag("((S[b]\\NP)/NP)/-->(S[b]\\NP)/NP")
    assert engine.is_valid_supertag("N-->NP")


# --- combinators ------------------------------------------------------------

def test_application_and_feature_flow(engine):
    assert ("NP", "fa", 0) in engine.combine("NP[nb]/N", "N")
    assert ("S[dcl]", "ba", 1) in engine.combine("NP", "S[dcl]\\NP")
    # bare S is the variable: S/S + S[dcl] -> S[dcl]
    assert ("S[dcl]", "fa", 0) in engine.combine("S/S", "S[dcl]")
    # modifier returns the modified category with its feature
    assert ("S[ng]\\NP", "ba", 1) in engine.combine("S[ng]\\NP", "(S\\NP)\\(S\\NP)")
    # feature-less NP argument matches NP[expl]
    assert ("S[dcl]\\NP", "fa", 0) in engine.combine("(S[dcl]\\NP)/NP", "NP[expl]")
    # incompatible S features do not unify
    assert not any(r == "fa" for _, r, _ in engine.combine("(S[dcl]\\NP)/(S[b]\\NP)", "S[ng]\\NP"))


def test_composition_rules(engine):
    assert ("(S[dcl]\\NP)/NP", "fc", 0) in engine.combine("(S[dcl]\\NP)/(S[b]\\NP)", "(S[b]\\NP)/NP")
    # modifier composed into a verb inherits the feature
    assert ("(S[dcl]\\NP)/NP", "fc", 0) in engine.combine("(S\\NP)/(S\\NP)", "(S[dcl]\\NP)/NP")
    # variable binding flows into the secondary functor: (S\NP)/(S\NP) gets S[b]
    assert ("(S[dcl]\\NP)/(S[b]\\NP)", "fc", 0) in engine.combine("(S[dcl]\\NP)/(S[b]\\NP)", "(S\\NP)/(S\\NP)")
    # >B2
    assert ("(S[dcl]/PP)/NP", "gfc", 0) in engine.combine("S/(S\\NP)", "((S[dcl]\\NP)/PP)/NP")
    # <Bx (VP-restricted)
    assert ("(S[dcl]\\NP)/NP", "bx", 1) in engine.combine("(S[dcl]\\NP)/NP", "(S\\NP)\\(S\\NP)")
    assert not any(r == "bx" for _, r, _ in engine.combine("NP/N", "NP\\NP"))


def test_conjunction_and_punctuation(engine):
    assert ("N[conj]", "conj", 1) in engine.combine("conj", "N")
    assert ("N", "coord", 0) in engine.combine("N", "N[conj]")
    assert ("S[dcl]", "rp", 0) in engine.combine("S[dcl]", ".")
    assert ("NP", "lp", 1) in engine.combine(",", "NP")
    assert ("NP[conj]", "conj", 1) in engine.combine(",", "NP")
    # conj-marked categories are not functors/arguments
    assert not any(r in ("fa", "ba") for _, r, _ in engine.combine("NP/N", "N[conj]"))


def test_unary_rules(engine):
    assert ("NP", "tc") in engine.unary("N")
    assert ("NP", "tc") in engine.unary("N[num]")
    assert ("S/(S\\NP)", "tr") in engine.unary("NP")
    assert ("NP\\NP", "tc") in engine.unary("S[dcl]\\NP")
    assert engine.unary(".") == []


def test_root_categories(engine):
    assert engine.is_root("S[dcl]")
    assert engine.is_root("S[wq]")
    assert not engine.is_root("NP")
    assert not engine.is_root("S[dcl][conj]")


# --- parsing ----------------------------------------------------------------

def test_simple_sentence(engine):
    tokens = ["The", "dog", "barks", "."]
    out = engine.parse(tokens, lat([("NP[nb]/N", 0.9)], [("N", 0.8)], [("S[dcl]\\NP", 0.7)], [(".", 0.99)]))
    assert out is not None
    assert out["root_cat"] == "S[dcl]"
    assert math.isclose(out["score"], math.log(0.9) + math.log(0.8) + math.log(0.7) + math.log(0.99))
    auto = render_auto(out)
    assert auto == (
        "(<T S[dcl] 0 2> (<T S[dcl] 1 2> (<T NP 0 2> (<L NP[nb]/N POS POS The NP[nb]/N>) "
        "(<L N POS POS dog N>) ) (<L S[dcl]\\NP POS POS barks S[dcl]\\NP>) ) (<L . POS POS . .>) )"
    )


def test_onebest_fails_and_lattice_recovers(engine):
    tokens = ["Mia", "found", "the", "ball"]
    # 1-best gives the verb an intransitive category: no S.
    onebest = lat([("N", 0.9)], [("S[dcl]\\NP", 0.6)], [("NP[nb]/N", 0.9)], [("N", 0.9)])
    assert engine.parse(tokens, onebest) is None
    widened = lat(
        [("N", 0.9), ("NP", 0.05)],
        [("S[dcl]\\NP", 0.6), ("(S[dcl]\\NP)/NP", 0.3)],
        [("NP[nb]/N", 0.9)],
        [("N", 0.9), ("NP", 0.05)],
    )
    out = engine.parse(tokens, widened)
    assert out is not None
    assert out["root_cat"] == "S[dcl]"
    assert leaf_categories(out) == ["N", "(S[dcl]\\NP)/NP", "NP[nb]/N", "N"]
    # N -> NP type change on both NPs, highest-probability consistent choice
    assert math.isclose(out["score"], math.log(0.9) + math.log(0.3) + math.log(0.9) + math.log(0.9))


def test_viterbi_prefers_higher_probability_derivation(engine):
    tokens = ["He", "ran"]
    out = engine.parse(tokens, lat([("NP", 0.9), ("N", 0.1)], [("S[dcl]\\NP", 0.5), ("S[b]\\NP", 0.4)]))
    assert out is not None
    assert leaf_categories(out) == ["NP", "S[dcl]\\NP"]


def test_unary_chain_supertag(engine):
    tokens = ["Dogs", "bark"]
    out = engine.parse(tokens, lat([("N-->NP", 0.7)], [("S[dcl]\\NP", 0.8)]))
    assert out is not None
    assert render_auto(out) == "(<T S[dcl] 1 2> (<T NP 0 1> (<L N POS POS Dogs N>) ) (<L S[dcl]\\NP POS POS bark S[dcl]\\NP>) )"


def test_coordination_and_modifier(engine):
    tokens = ["Then", "cats", "and", "dogs", "ran", "fast", "."]
    out = engine.parse(
        tokens,
        lat(
            [("S/S", 0.8)], [("N", 0.9)], [("conj", 0.99)], [("N", 0.9)],
            [("S[dcl]\\NP", 0.9)], [("(S\\NP)\\(S\\NP)", 0.8)], [(".", 0.99)],
        ),
    )
    assert out is not None
    assert out["root_cat"] == "S[dcl]"


def test_type_raising_with_composition(engine):
    # object extraction: "what he saw" -> needs NP type raising + >B
    tokens = ["what", "he", "saw"]
    out = engine.parse(tokens, lat([("S[wq]/(S[dcl]/NP)", 0.8)], [("NP", 0.9)], [("(S[dcl]\\NP)/NP", 0.9)]))
    assert out is not None
    assert out["root_cat"] == "S[wq]"


def test_eisner_normal_form_blocks_spurious_composition(engine):
    # "I will go": NF derivation is application-first, not (I will) composition.
    tokens = ["I", "will", "go"]
    out = engine.parse(tokens, lat([("NP", 0.9)], [("(S[dcl]\\NP)/(S[b]\\NP)", 0.9)], [("S[b]\\NP", 0.9)]))
    assert out is not None
    nodes = out["nodes"]
    root = nodes[out["root"]]
    assert root["rule"] == "ba"
    assert all(n["rule"] not in ("fc", "gfc") for n in nodes)


def test_crossed_composition_output_feeds_application(engine):
    # "It was not scary anymore ." : (was not) by <Bx must then take `scary` by >.
    tokens = ["It", "was", "not", "scary", "anymore", "."]
    out = engine.parse(
        tokens,
        lat(
            [("NP", 0.9)], [("(S[dcl]\\NP)/(S[adj]\\NP)", 0.9)], [("(S\\NP)\\(S\\NP)", 0.9)],
            [("S[adj]\\NP", 0.9)], [("(S\\NP)\\(S\\NP)", 0.9)], [(".", 0.99)],
        ),
    )
    assert out is not None and out["root_cat"] == "S[dcl]"
    assert any(n["rule"] == "bx" for n in out["nodes"])
    # and <Bx output as primary functor of >B: "do n't want to play"
    tokens = ["I", "do", "n't", "want", "to", "play"]
    out = engine.parse(
        tokens,
        lat(
            [("NP", 0.9)], [("(S[dcl]\\NP)/(S[b]\\NP)", 0.9)], [("(S\\NP)\\(S\\NP)", 0.9)],
            [("(S[b]\\NP)/(S[to]\\NP)", 0.9)], [("(S[to]\\NP)/(S[b]\\NP)", 0.9)], [("S[b]\\NP", 0.9)],
        ),
    )
    assert out is not None and out["root_cat"] == "S[dcl]"


def test_parse_batch_matches_single(engine):
    tokens = ["The", "dog", "barks", "."]
    lattice = lat([("NP[nb]/N", 0.9)], [("N", 0.8)], [("S[dcl]\\NP", 0.7)], [(".", 0.99)])
    single = engine.parse(tokens, lattice)
    batch = engine.parse_batch([tokens, tokens, ["x"]], [lattice, lattice, lat([("N", 1.0)])], n_threads=2)
    assert batch[0]["ok"] and batch[1]["ok"] and not batch[2]["ok"]
    assert render_auto(batch[0]) == render_auto(single)


def test_seen_rules_filter():
    seen = [("NP[nb]/N", "N"), ("NP", "S[dcl]\\NP")]
    engine = CKYEngine(GrammarConfig(use_seen_rules=True), seen_rules=seen)
    assert engine.combine("NP/N", "N")  # features stripped for the key
    assert engine.combine("S/S", "S[dcl]") == []
