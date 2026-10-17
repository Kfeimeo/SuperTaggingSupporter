"""Bounded sentence scheduling with one private native engine per process."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import signal
import traceback


def _worker(connection, grammar, seen_rules):
    # Only the coordinator handles terminal interrupts and owns output files.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        from stsupport.engine import CKYEngine

        engine = CKYEngine(grammar, seen_rules)
        connection.send(("ready", None))
        while True:
            task = connection.recv()
            if task is None:
                break
            index, tokens, lattice = task
            connection.send(("result", (index, engine.parse_raw(tokens, lattice))))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


@contextmanager
def _single_library_thread():
    # Spawned interpreters may import NumPy via the CLI. Native parsing itself
    # is single-threaded; avoid allocating a BLAS thread pool in every worker.
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(dict.fromkeys(names, "1"))
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class IndependentParsers:
    """Small spawn pool with public multiprocessing APIs and explicit cleanup.

    Each busy worker holds exactly one sentence. Remaining bounded tasks stay
    in the parent's deque. Workers never load the dictionary or write outputs.
    """

    def __init__(self, grammar, seen_rules, workers):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.children = {}
        self.busy = {}
        self.idle = []
        self.queued = deque()
        self.pending = {}
        context = mp.get_context("spawn")
        try:
            with _single_library_thread():
                for _ in range(workers):
                    parent, child = context.Pipe()
                    process = context.Process(target=_worker, args=(child, grammar, seen_rules),
                                              name="ccg-independent-parser")
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        child.close()
                        raise
                    child.close()
                    self.children[parent] = process
        except BaseException:
            self.close(abort=True)
            raise

    @property
    def pids(self):
        return [process.pid for process in self.children.values()]

    def submit(self, index, tokens, lattice, record):
        self.pending[index] = record
        self.queued.append((index, tokens, lattice))
        self._dispatch()

    def _dispatch(self):
        while self.idle and self.queued:
            connection = self.idle.pop()
            task = self.queued.popleft()
            self.busy[connection] = task[0]
            try:
                connection.send(task)
            except (EOFError, OSError) as error:
                raise self._failure(connection, "disconnected while receiving a task") from error

    def _failure(self, connection, reason):
        process = self.children[connection]
        # EOF can arrive just before multiprocessing observes the exit status.
        process.join(timeout=0.1)
        code = process.exitcode
        index = self.busy.get(connection)
        task = (f"sentence.{index + 1} len={len(self.pending[index]['tokens'])} "
                f"candidates={sum(self.pending[index]['lattice_sizes'])}" if index is not None
                else "during initialization or while idle")
        details = ""
        if os.name == "posix" and code is not None and code < 0:
            try:
                details = f" signal={signal.Signals(-code).name}"
            except ValueError:
                pass
            if code == -signal.SIGKILL:
                details += "; SIGKILL may indicate an OOM kill or an external kill; check cgroup/kernel records"
        return RuntimeError(f"parser worker pid={process.pid} {reason}; exitcode={code}{details}; {task}")

    def poll(self, timeout):
        results = []
        # Check even idle children: a crashed process must never silently leave
        # the coordinator waiting for a missing result forever.
        for connection, process in self.children.items():
            if process.exitcode is not None and not connection.poll():
                raise self._failure(connection, "exited unexpectedly")
        for connection in wait(list(self.children), timeout=timeout):
            try:
                kind, value = connection.recv()
            except (EOFError, OSError) as error:
                raise self._failure(connection, "disconnected before completing its task") from error
            if kind == "error":
                raise self._failure(connection, f"failed:\n{value}")
            if kind == "result":
                index, out = value
                if self.busy.pop(connection) != index:
                    raise RuntimeError("parser worker returned the wrong sentence ID")
                results.append((index, self.pending.pop(index), out))
            elif kind != "ready":
                raise RuntimeError(f"unknown worker message: {kind}")
            self.idle.append(connection)
        self._dispatch()
        return results

    def close(self, *, abort=False):
        try:
            for connection, process in self.children.items():
                if process.is_alive():
                    if abort:
                        process.terminate()
                    else:
                        try:
                            connection.send(None)
                        except (OSError, EOFError):
                            process.terminate()
            for process in self.children.values():
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
        finally:
            for connection, process in self.children.items():
                connection.close()
                if not process.is_alive():
                    process.close()
            self.children.clear()
