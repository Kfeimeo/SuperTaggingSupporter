"""Bounded CPU work around a single GPU inference owner.

No worker touches CUDA or SQLite. Aggregation remains in input order so a
checkpoint can only advance after all of its soft counts have finished.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from stsupport.lexicon import GlobalLexicon


def prefetch_one(iterable, *, enabled=True):
    """Prepare at most one following group while the consumer handles this one."""
    if not enabled:
        yield from iterable
        return
    iterator = iter(iterable)
    sentinel = object()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="corpus-prefetch") as pool:
        future = pool.submit(next, iterator, sentinel)
        try:
            while True:
                item = future.result()
                if item is sentinel:
                    return
                future = pool.submit(next, iterator, sentinel)
                yield item
        finally:
            future.cancel()
            # Join any active next() before closing its underlying generator.
            pool.shutdown(wait=True, cancel_futures=True)
            if hasattr(iterator, "close"):
                iterator.close()


class LexiconAccumulator:
    """Ordered CPU aggregation with bounded probability batches in flight."""
    def __init__(self, lexicon, *, exclude=(), enabled=True, max_pending=2, workers=1):
        if max_pending < 1 or workers < 1:
            raise ValueError("max_pending and workers must be positive")
        self.lexicon = lexicon
        self.exclude = tuple(exclude)
        self.enabled = enabled
        self.max_pending = max_pending
        self.pending = deque()
        self.workers = workers
        self.pools = []
        self.shards = []

    def __enter__(self):
        if self.enabled:
            self.pools = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"lexicon-aggregate-{i}")
                          for i in range(self.workers)]
            if self.workers > 1:
                self.shards = [GlobalLexicon(self.lexicon.categories, lowercase=self.lexicon.lowercase)
                               for _ in range(self.workers)]
                for word, count in self.lexicon.counts.items():
                    shard = self.shards[self._owner(word)]
                    shard.counts[word] = count
                    shard.soft_counts[word] = self.lexicon.soft_counts[word].copy()
                    shard.support[word] = self.lexicon.support[word].copy()
        return self

    def _owner(self, word):
        return hash(self.lexicon.key(word)) % self.workers

    def _accumulate(self, sentences, distributions):
        if len(sentences) != len(distributions):
            raise ValueError("tagger sentence count mismatch")
        for tokens, probs in zip(sentences, distributions):
            self.lexicon.add(tokens, probs, k=8, beta=0.99, exclude=self.exclude)

    def add_batch(self, sentences, distributions):
        if not self.enabled:
            self._accumulate(sentences, distributions)
            return
        if len(self.pending) >= self.max_pending:
            self._wait_oldest()
        if len(sentences) != len(distributions):
            raise ValueError("tagger sentence count mismatch")
        if self.workers == 1:
            self.pending.append([self.pools[0].submit(self._accumulate, sentences, distributions)])
            return
        # Route all occurrences of a word to one ordered queue. Workers never
        # mutate the same arrays; per-word addition order stays identical.
        routed = [[] for _ in range(self.workers)]
        for tokens, probs in zip(sentences, distributions):
            if probs.shape != (len(tokens), len(self.lexicon.categories)):
                raise ValueError("probability matrix must have shape (tokens, categories)")
            indices = [[] for _ in range(self.workers)]
            for index, word in enumerate(tokens):
                indices[self._owner(word)].append(index)
            for owner, selected in enumerate(indices):
                if selected:
                    routed[owner].append((tokens, probs, selected))
        self.pending.append([pool.submit(self._accumulate_shard, shard, items)
                             for pool, shard, items in zip(self.pools, self.shards, routed) if items])

    def _accumulate_shard(self, shard, items):
        for tokens, probs, indices in items:
            shard.add([tokens[i] for i in indices], probs[indices], k=8, beta=0.99, exclude=self.exclude)

    def _wait_oldest(self):
        for future in self.pending.popleft():
            future.result()  # backpressure, and propagate worker errors

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                while self.pending:
                    self._wait_oldest()
                for shard in self.shards:
                    # Replace seeded entries with their fully updated word shard.
                    for word, count in shard.counts.items():
                        self.lexicon.counts[word] = count
                    self.lexicon.soft_counts.update(shard.soft_counts)
                    self.lexicon.support.update(shard.support)
        finally:
            for batch in self.pending:
                for future in batch:
                    future.cancel()
            for pool in self.pools:
                pool.shutdown(wait=True, cancel_futures=True)
        return False
