from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from stsupport.config import GrammarConfig
from stsupport.engine import CKYEngine
from stsupport.lexicon import GlobalLexicon
from stsupport.lexicon_experiment import evaluate_batch, evaluate_processes


def _delayed_worker(connection, grammar, seen_rules):
    """Deterministic slow first sentence without expensive grammar search."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    connection.send(("ready", None))
    while task := connection.recv():
        index, tokens, lattice = task
        if index == 0:
            deadline = time.monotonic() + 8
            while not Path(grammar.seen_rules_path).exists():
                if time.monotonic() >= deadline:
                    connection.send(("error", "slow sentence prevented later scheduling"))
                    return
                time.sleep(0.01)
        connection.send(("result", (index, {"ok": False, "root_cell_cats": []})))


def _crashing_worker(connection, grammar, seen_rules):
    os._exit(7)


def _lex():
    lex = GlobalLexicon(["NP", "S[dcl]\\NP", "."])
    lex.add(["Lily", "sleeps", "."], np.eye(3))
    return lex


def _children():
    return {process.pid for process in mp.active_children()}


def test_private_process_results_equal_native_results(tmp_path):
    lex = _lex()
    sentences = [["Lily", "sleeps", "."], ["unknown", "sleeps"], ["sleeps", "sleeps"], ["Lily"] * 6] * 3
    expected = evaluate_batch(CKYEngine(), lex, sentences, max_tokens=5, n_threads=1)
    initial = _children()
    seen, snapshots = [], []
    actual = list(evaluate_processes(GrammarConfig(), lex, iter(sentences), processes=2, max_in_flight=3,
                                   prepare_batch_size=2, max_tokens=5, spool_dir=tmp_path,
                                   on_complete=lambda i, r: seen.append(i), on_wait=snapshots.append))
    assert actual == expected
    assert sorted(seen) == list(range(len(sentences)))
    assert any(len(info["pids"]) == 2 and len(set(info["pids"])) == 2 for info in snapshots)
    assert all(info["running"] + info["queued"] <= 3 for info in snapshots)
    assert _children() == initial
    assert not list(tmp_path.iterdir())


def test_later_sentences_are_scheduled_before_first_completes(tmp_path, monkeypatch):
    import stsupport.process_parser as module
    monkeypatch.setattr(module, "_worker", _delayed_worker)
    lex = _lex()
    release = tmp_path / "release"
    seen, snapshots = [], []
    def finished(index, record):
        seen.append(index)
        if index == 7:
            release.touch()
    actual = list(evaluate_processes(GrammarConfig(seen_rules_path=str(release)), lex,
                                   [["Lily", "sleeps"]] * 10, processes=2, max_in_flight=3,
                                   prepare_batch_size=2, spool_dir=tmp_path, on_complete=finished,
                                   on_wait=snapshots.append))
    assert seen.index(7) < seen.index(0)
    assert len(actual) == 10
    assert any(info["buffered_results"] >= 7 for info in snapshots)
    assert all(info["running"] + info["queued"] <= 3 for info in snapshots)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["release"]


def test_sigint_terminates_workers_and_removes_spool(tmp_path, monkeypatch):
    import stsupport.process_parser as module
    monkeypatch.setattr(module, "_worker", _delayed_worker)
    initial = _children()
    def interrupt(info):
        if info["running"]:
            signal.raise_signal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        list(evaluate_processes(GrammarConfig(seen_rules_path=str(tmp_path / "never")), _lex(),
                                [["Lily", "sleeps"]] * 10, processes=2, max_in_flight=3,
                                on_wait=interrupt, spool_dir=tmp_path))
    assert _children() == initial
    assert not list(tmp_path.iterdir())


def test_worker_crash_fails_and_cleans_up(tmp_path, monkeypatch):
    import stsupport.process_parser as module
    monkeypatch.setattr(module, "_worker", _crashing_worker)
    initial = _children()
    with pytest.raises(RuntimeError, match="worker"):
        list(evaluate_processes(GrammarConfig(), _lex(), [["Lily", "sleeps"]],
                                processes=2, max_in_flight=2, spool_dir=tmp_path))
    assert _children() == initial
    assert not list(tmp_path.iterdir())


def test_all_oov_does_not_start_workers(tmp_path):
    snapshots = []
    records = list(evaluate_processes(GrammarConfig(), _lex(), [["unknown"]] * 20,
                                      prepare_batch_size=2, spool_dir=tmp_path, on_wait=snapshots.append))
    assert len(records) == 20
    assert all(record["reason"] == "oov" for record in records)
    assert all(not info["pids"] for info in snapshots)


def test_explicit_seen_rules_are_preserved_in_workers(tmp_path):
    from stsupport.engine import load_seen_rules
    path = tmp_path / "seen.json"
    path.write_text(json.dumps({"rules": [{"left": "NP", "right": "S[dcl]\\NP"}]}))
    grammar = GrammarConfig(use_seen_rules=True, seen_rules_path=str(path))
    sentences = [["Lily", "sleeps"], ["sleeps", "sleeps"]]
    expected = evaluate_batch(CKYEngine(grammar, load_seen_rules(path)), _lex(), sentences, n_threads=1)
    actual = list(evaluate_processes(grammar, _lex(), sentences, processes=2, max_in_flight=2, spool_dir=tmp_path))
    assert actual == expected


@pytest.mark.parametrize("storage", ["json", "sqlite"])
def test_cli_process_mode_jsonl_order_and_summary(tmp_path, storage):
    lex = tmp_path / f"lexicon.{storage}"
    if storage == "json":
        _lex().save(lex)
    else:
        from stsupport.disk_lexicon import LexiconStore
        dictionary = _lex()
        store = LexiconStore(lex, dictionary.categories, {})
        store.commit(dictionary, {})
        store.close()
    corpus = tmp_path / "custom_input.jsonl"
    sentences = [["Lily", "sleeps", "."], ["unknown", "sleeps"], ["sleeps", "sleeps"]] * 3
    corpus.write_text("".join(json.dumps({"tokens": tokens}) + "\n" for tokens in sentences), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "stsupport", "eval-lexicon", "--lexicon", str(lex),
                             "--corpus", str(corpus), "--parser-backend", "processes", "--processes", "2",
                             "--max-in-flight", "3", "--batch-size", "2", "--no-progress",
                             "--set", f"output.dir={tmp_path.as_posix()}", "--set", "output.name=custom"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in (tmp_path / "custom.jsonl").read_text().splitlines()]
    assert [record["tokens"] for record in records] == sentences
    assert [record["id"] for record in records] == [f"sentence.{i + 1}" for i in range(9)]
    summary = json.loads((tmp_path / "custom.summary.json").read_text())
    assert summary["parser_backend"] == "processes"
    assert summary["processes"] == 2 and summary["max_in_flight"] == 3
    assert summary["sentences"] == 9 and summary["parsed"] == 3
    assert summary["reasons"] == {"oov": 3, "no_derivation": 3}


def test_worker_initialization_error_is_reported(tmp_path):
    initial = _children()
    with pytest.raises(RuntimeError, match="worker"):
        list(evaluate_processes(GrammarConfig(root_categories=["("]), _lex(), [["Lily", "sleeps"]],
                                processes=2, max_in_flight=2, spool_dir=tmp_path))
    assert _children() == initial
    assert not list(tmp_path.iterdir())


def test_full_runner_passes_process_arguments(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_full_tinystories_experiment as runner
    args = runner.parse_args(["--parse-backend", "processes", "--parse-processes", "3", "--parse-max-in-flight", "6"])
    command = runner.cpu_command("target.jsonl", backend=args.parse_backend, processes=args.parse_processes,
                                 max_in_flight=args.parse_max_in_flight)
    assert command[command.index("--parser-backend") + 1] == "processes"
    assert command[command.index("--processes") + 1] == "3"
    assert command[command.index("--max-in-flight") + 1] == "6"
    for invalid in (["--parse-processes", "0"], ["--parse-processes", "4", "--parse-max-in-flight", "2"]):
        with pytest.raises(SystemExit):
            runner.parse_args(invalid)
