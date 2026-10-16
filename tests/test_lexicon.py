from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest

from stsupport.cli import main
from stsupport.config import GrammarConfig
from stsupport.corpus import iter_corpus
from stsupport.engine import CKYEngine
from stsupport.lexicon import GlobalLexicon
from stsupport.lexicon_experiment import evaluate_batch


def test_full_soft_mean_is_separate_from_selected_union(tmp_path):
    lex = GlobalLexicon(["N", "NP", "S"], lowercase=True)
    lex.add(["Duck", "duck"], np.array([[0.9, 0.09, 0.01], [0.2, 0.79, 0.01]]), k=1)
    assert lex.counts == {"duck": 2}
    assert lex.category_map() == {"duck": ["N", "NP"]}
    np.testing.assert_allclose(lex.distribution("DUCK"), [0.55, 0.44, 0.01])
    candidates = dict(lex.candidates("Duck"))
    assert math.exp(candidates["N"]) == pytest.approx(0.55 / 0.99)
    assert dict(lex.candidates("duck", weighting="uniform")) == {"N": -math.log(2), "NP": -math.log(2)}
    assert lex.candidates("duck", min_support=2) == []
    assert lex.candidates("missing") == []
    path = tmp_path / "lex.json"
    lex.save(path)
    restored = GlobalLexicon.load(path)
    assert restored.category_map() == lex.category_map()
    np.testing.assert_allclose(restored.distribution("DUCK"), lex.distribution("duck"))


def test_mask_and_support_threshold():
    lex = GlobalLexicon(["<unk>", "N", "NP"])
    lex.add(["duck", "duck"], np.array([[0.9, 0.06, 0.04], [0.9, 0.09, 0.01]]), k=3, beta=0.95, exclude=[0])
    np.testing.assert_allclose(lex.distribution("duck"), [0, 0.75, 0.25])
    assert lex.category_map(min_support=2) == {"duck": ["N", "NP"]}


@pytest.mark.parametrize("probs", [np.array([[0, 0]]), np.array([[1, -1]]), np.array([[np.nan, 1]]), np.ones((2, 2))])
def test_invalid_probabilities(probs):
    with pytest.raises(ValueError):
        GlobalLexicon(["N", "NP"]).add(["duck"], probs)


def make_lexicon():
    lex = GlobalLexicon(["NP", "S[dcl]\\NP", "."], metadata={"source": {"drop_quotes": True}})
    lex.add(["Lily", "sleeps", "."], np.eye(3))
    return lex


def test_direct_dictionary_parsing_and_failure_attribution():
    engine = CKYEngine(GrammarConfig())
    lex = make_lexicon()
    sentences = [["Lily", "sleeps", "."], ["Mia", "sleeps", "."], ["sleeps", "sleeps"], ["Lily"] * 5]
    records = evaluate_batch(engine, lex, sentences, max_tokens=4)
    assert [r["status"] for r in records] == ["parsed", "fail", "fail", "skipped"]
    assert [r["reason"] for r in records] == [None, "oov", "no_derivation", "max_tokens"]
    assert records[0]["final_tags"] == ["NP", "S[dcl]\\NP", "."]
    assert records[1]["oov_positions"] == [0]
    filtered = evaluate_batch(engine, lex, sentences[:1], min_support=2)[0]
    assert filtered["reason"] == "empty_support"


def test_eval_cli_custom_jsonl_does_not_import_tagger(tmp_path, monkeypatch):
    lexpath = tmp_path / "lex.json"
    make_lexicon().save(lexpath)
    corpus = tmp_path / "custom.jsonl"
    corpus.write_text('\n'.join(json.dumps({"tokens": words}) for words in
                                [["Lily", "sleeps", "."], ["Mia", "sleeps", "."], ["sleeps", "sleeps"]]), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "stsupport.supertagger", None)
    assert main(["eval-lexicon", "--lexicon", str(lexpath), "--corpus", str(corpus),
                 "--set", f"output.dir={tmp_path.as_posix()}", "--set", "output.name=custom_eval"]) == 0
    summary = json.loads((tmp_path / "custom_eval.summary.json").read_text())
    assert summary["parse_rate_all"] == pytest.approx(1 / 3)
    assert summary["parse_rate_covered_eligible"] == 0.5
    assert summary["oov_tokens"] == 1
    assert summary["reasons"] == {"oov": 1, "no_derivation": 1}
    assert json.loads((tmp_path / "custom_eval.oov.json").read_text()) == {"Mia": 1}
    original = corpus.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        main(["eval-lexicon", "--lexicon", str(lexpath), "--corpus", str(corpus),
              "--set", f"output.dir={tmp_path.as_posix()}", "--set", "output.name=custom"])
    assert corpus.read_bytes() == original


def test_streaming_corpus_limits(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"tokens":["a","b"]}\n{"tokens":["c","d"]}\n', encoding="utf-8")
    assert list(iter_corpus(path, limit=1)) == [["a", "b"]]
    assert list(iter_corpus(path, limit=0)) == []
    with pytest.raises(ValueError):
        list(iter_corpus(path, limit=-1))
