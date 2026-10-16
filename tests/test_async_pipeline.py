from __future__ import annotations

from threading import Event

import numpy as np
import pytest

from stsupport.async_pipeline import LexiconAccumulator, prefetch_one
from stsupport.disk_lexicon import LexiconStore
from stsupport.lexicon import GlobalLexicon


def test_prefetch_prepares_next_group_while_consumer_is_busy():
    started, release, closed = Event(), Event(), Event()
    consumed = []
    def source():
        try:
            consumed.append(1)
            yield 1
            consumed.append(2)
            started.set()
            if not release.wait(2):
                raise RuntimeError("consumer did not release producer")
            yield 2
            consumed.append(3)
            yield 3
        finally:
            closed.set()
    stream = prefetch_one(source())
    try:
        assert next(stream) == 1
        assert started.wait(2)  # CPU already works ahead while main owns the first group
        assert consumed == [1, 2]  # one group ahead, never the whole corpus
    finally:
        release.set()
        stream.close()
    assert closed.is_set()


def test_prefetch_propagates_producer_failure():
    def broken():
        yield 1
        raise ValueError("bad source")
    with pytest.raises(ValueError, match="bad source"):
        list(prefetch_one(broken()))
    assert list(prefetch_one([1, 2], enabled=False)) == [1, 2]


def test_aggregation_runs_without_blocking_inference_caller(monkeypatch):
    started, release = Event(), Event()
    lex = GlobalLexicon(["N", "NP"])
    with LexiconAccumulator(lex, max_pending=1) as accumulator:
        original = accumulator._accumulate
        def blocked(*args):
            started.set()
            if not release.wait(2):
                raise RuntimeError("caller did not release aggregator")
            original(*args)
        monkeypatch.setattr(accumulator, "_accumulate", blocked)
        try:
            accumulator.add_batch([["dog"]], [np.array([[0.9, 0.1]])])
            assert started.wait(2)
            assert not lex.counts  # main can start the next GPU inference here
        finally:
            release.set()
    assert lex.counts["dog"] == 1


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_async_and_serial_checkpoints_are_identical(tmp_path, workers):
    categories = ["N", "NP", "S"]
    groups = [
        [([["duck", "duck"]], [np.array([[0.9, 0.09, 0.01], [0.2, 0.79, 0.01]])]),
         ([["other"]], [np.array([[0.1, 0.2, 0.7]])])],
        [([["duck"]], [np.array([[0.4, 0.5, 0.1]])])],
    ]
    paths = []
    for enabled in (False, True):
        path = tmp_path / f"{enabled}.sqlite"
        paths.append(path)
        store = LexiconStore(path, categories, {})
        for index, batches in enumerate(groups):
            delta = GlobalLexicon(categories)
            with LexiconAccumulator(delta, enabled=enabled, max_pending=1, workers=workers) as accumulator:
                for sentences, probs in batches:
                    accumulator.add_batch(sentences, probs)
            store.commit(delta, {"offset": index + 1})
        store.close()
    serial, parallel = map(GlobalLexicon.load, paths)
    try:
        assert serial.counts == parallel.counts
        assert serial.category_map() == parallel.category_map()
        assert serial.support == parallel.support
        assert serial.metadata["progress"] == parallel.metadata["progress"]
        for word in serial.counts:
            np.testing.assert_array_equal(serial.distribution(word), parallel.distribution(word))
    finally:
        serial.close()
        parallel.close()


@pytest.mark.parametrize("workers", [1, 4])
def test_worker_failure_does_not_advance_checkpoint(tmp_path, workers):
    store = LexiconStore(tmp_path / "checkpoint.sqlite", ["N", "NP"], {})
    store.commit(GlobalLexicon(["N", "NP"]), {"offset": 100})
    delta = GlobalLexicon(["N", "NP"])
    with pytest.raises(ValueError, match="finite"):
        with LexiconAccumulator(delta, workers=workers) as accumulator:
            accumulator.add_batch([["dog"]], [np.array([[0.9, 0.1]])])
            accumulator.add_batch([["two", "words"]], [np.array([[np.nan, 0.1], [0.2, 0.8]])])
        store.commit(delta, {"offset": 200})
    assert store.progress == {"offset": 100}
    assert store.summary()["tokens"] == 0
    store.close()


def test_multiple_workers_actually_run_concurrently(monkeypatch):
    release = Event()
    started = [Event(), Event()]
    lex = GlobalLexicon(["N", "NP"])
    with LexiconAccumulator(lex, workers=2) as accumulator:
        words = {}
        for i in range(1000):
            word = f"word{i}"
            words.setdefault(accumulator._owner(word), word)
            if len(words) == 2:
                break
        assert len(words) == 2
        original = accumulator._accumulate_shard
        def blocked(shard, items):
            started[accumulator.shards.index(shard)].set()
            if not release.wait(2):
                raise RuntimeError("concurrent workers did not start")
            original(shard, items)
        monkeypatch.setattr(accumulator, "_accumulate_shard", blocked)
        try:
            accumulator.add_batch([[words[0], words[1]]], [np.array([[0.9, 0.1], [0.2, 0.8]])])
            assert all(event.wait(2) for event in started)
        finally:
            release.set()
    assert sum(lex.counts.values()) == 2


def test_word_shards_preserve_existing_counts_and_case_variants():
    serial, parallel = [GlobalLexicon(["N", "NP"], lowercase=True) for _ in range(2)]
    for lex in (serial, parallel):
        lex.add(["Duck"], np.array([[0.9, 0.1]]))
    batches = [([["DUCK", "dog"]], [np.array([[0.2, 0.8], [0.3, 0.7]])]),
               ([["duck", "Dog"]], [np.array([[0.7, 0.3], [0.8, 0.2]])])]
    for lex, enabled in ((serial, False), (parallel, True)):
        with LexiconAccumulator(lex, enabled=enabled, workers=4) as accumulator:
            for sentences, probs in batches:
                accumulator.add_batch(sentences, probs)
    assert serial.counts == parallel.counts == {"duck": 3, "dog": 2}
    assert serial.support == parallel.support
    for word in serial.counts:
        np.testing.assert_array_equal(serial.soft_counts[word], parallel.soft_counts[word])
