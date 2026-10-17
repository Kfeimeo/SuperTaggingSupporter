from __future__ import annotations

from threading import Event

import numpy as np

from stsupport.config import GrammarConfig
from stsupport.engine import CKYEngine
from stsupport.lexicon import GlobalLexicon
from stsupport.lexicon_experiment import evaluate_batch, process_resources


def lexicon():
    lex = GlobalLexicon(["N"])
    lex.add(["slow", "fast"], np.ones((2, 1)))
    return lex


def test_sentence_completion_does_not_wait_for_slow_batch_member():
    slow_started, release = Event(), Event()
    class Engine:
        def parse_raw(self, tokens, lattice):
            if tokens == ["slow"]:
                slow_started.set()
                if not release.wait(2):
                    raise RuntimeError("batch completion barrier blocked fast result")
            elif not slow_started.wait(2):
                raise RuntimeError("slow worker did not start")
            return {"ok": False, "root_cell_cats": []}
    completed = []
    def on_complete(index, record):
        completed.append(index)
        if index == 1:
            release.set()
    try:
        records = evaluate_batch(Engine(), lexicon(), [["slow"], ["fast"]], n_threads=2, on_complete=on_complete)
    finally:
        release.set()
    assert completed == [1, 0]
    assert [record["tokens"] for record in records] == [["slow"], ["fast"]]


def test_heartbeat_continues_without_completed_sentences():
    release = Event()
    class Engine:
        def parse_raw(self, tokens, lattice):
            if not release.wait(2):
                raise RuntimeError("heartbeat never released worker")
            return {"ok": False, "root_cell_cats": []}
    snapshots = []
    def heartbeat(info):
        snapshots.append(info)
        if info["elapsed_seconds"] >= 0.02:
            release.set()
    try:
        evaluate_batch(Engine(), lexicon(), [["slow"]], n_threads=1, on_wait=heartbeat, heartbeat_seconds=0.01)
    finally:
        release.set()
    assert len(snapshots) >= 2
    assert snapshots[-1]["running"] == 1
    assert snapshots[-1]["max_candidates"] == 1
    assert snapshots[-1]["examples"][0]["tokens"] == 1


def test_incremental_and_native_batch_results_match():
    lex = GlobalLexicon(["NP", "S[dcl]\\NP", "."])
    lex.add(["Lily", "sleeps", "."], np.eye(3))
    engine = CKYEngine(GrammarConfig())
    sentences = [["Lily", "sleeps", "."], ["unknown", "sleeps"], ["sleeps", "sleeps"], ["Lily"] * 6]
    expected = evaluate_batch(engine, lex, sentences, max_tokens=5, n_threads=2)
    seen = []
    actual = evaluate_batch(engine, lex, sentences, max_tokens=5, n_threads=2,
                            on_complete=lambda index, record: seen.append(index))
    assert actual == expected
    assert sorted(seen) == [0, 1, 2, 3]
    assert process_resources()["cpu_seconds"] >= 0
