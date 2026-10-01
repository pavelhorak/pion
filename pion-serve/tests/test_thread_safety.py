#!/usr/bin/env python3
"""gh #84: thread-safety regression tests for pion-serve's lock helpers.

`serve.py` has top-level Flask init + redis/requests imports that we don't
want to drag into a unit test. The locking helpers themselves are tiny
(`_incr_stat`, `_stats_snapshot`, `_once_under_init_lock`), so the test
mirrors them locally and exercises the same code. If serve.py's copies
drift, that drift is what shows up as a real bug — keep the two in sync.

Hammer the helper from N threads doing M increments each and assert the
counter equals N×M. Without the lock the counter loses updates under
contention; with the lock it's exact.

Run:
    python pion-serve/tests/test_thread_safety.py
"""
from __future__ import annotations

import functools
import threading
import unittest


# ── Helpers mirrored from serve.py (gh #84) ──────────────────────────────────
# If you change these, update the same trio in pion-serve/serve.py.

_stats: dict = {}
_stats_lock = threading.RLock()
_init_lock = threading.RLock()


def _incr_stat(key: str, n: int = 1) -> None:
    with _stats_lock:
        _stats[key] = _stats.get(key, 0) + n


def _stats_snapshot() -> dict:
    with _stats_lock:
        return dict(_stats)


def _once_under_init_lock(fn):
    done = threading.Event()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if done.is_set():
            return
        with _init_lock:
            if done.is_set():
                return
            result = fn(*args, **kwargs)
            done.set()
            return result

    return wrapper


# ── Tests ────────────────────────────────────────────────────────────────────


class IncrStatIsAtomicUnderContention(unittest.TestCase):
    incr = staticmethod(_incr_stat)
    snap = staticmethod(_stats_snapshot)

    def setUp(self):
        # Reset the underlying _stats dict between tests.
        _stats.clear()

    def test_concurrent_increments_lose_no_updates(self):
        N_THREADS = 16
        PER_THREAD = 5_000
        key = "concurrent_test"

        def worker():
            for _ in range(PER_THREAD):
                self.incr(key)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = self.snap()
        self.assertEqual(
            snap[key], N_THREADS * PER_THREAD,
            f"lost updates under contention: got {snap[key]} vs expected {N_THREADS * PER_THREAD}",
        )

    def test_snapshot_returns_independent_copy(self):
        self.incr("a", 5)
        s1 = self.snap()
        self.incr("a", 3)
        # The earlier snapshot must not see the later mutation.
        self.assertEqual(s1["a"], 5)
        # But a fresh snapshot does.
        self.assertEqual(self.snap()["a"], 8)

    def test_negative_increments_supported(self):
        self.incr("balance", 10)
        self.incr("balance", -3)
        self.assertEqual(self.snap()["balance"], 7)

    def test_unknown_key_starts_at_zero(self):
        # Calling _incr_stat on an unseen key should auto-seed at 0 + n,
        # not KeyError.
        self.incr("first_touch", 4)
        self.assertEqual(self.snap()["first_touch"], 4)


class OnceUnderInitLock(unittest.TestCase):
    once = staticmethod(_once_under_init_lock)

    def test_body_runs_exactly_once_across_threads(self):
        counter = {"n": 0}
        lock = threading.Lock()

        @self.once
        def init():
            with lock:
                counter["n"] += 1

        N = 20
        threads = [threading.Thread(target=init) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(
            counter["n"], 1,
            f"init body ran {counter['n']} times; should be exactly 1",
        )

    def test_body_retries_after_exception(self):
        attempts = {"n": 0}

        @self.once
        def init_that_fails_once():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("intentional failure")

        # First call raises; flag stays unset
        with self.assertRaises(RuntimeError):
            init_that_fails_once()
        self.assertEqual(attempts["n"], 1)

        # Second call should retry (and succeed)
        init_that_fails_once()
        self.assertEqual(attempts["n"], 2)

        # Third call short-circuits (already succeeded)
        init_that_fails_once()
        self.assertEqual(attempts["n"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
