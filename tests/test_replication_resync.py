#!/usr/bin/env python3
"""A replica must converge to the primary however it joins — and WAIT must
say so honestly (gh #390).

WHY
A replica used to follow the primary's WAL only from the moment it connected:
its FULLRESYNC carried an empty snapshot. So a replica added to a running
primary, or one that missed writes while it was down, silently lacked data,
and nothing reported it. The primary never read the replicas' ACKs, so WAIT
counted nobody, and its timeout was a count of `usleep(1000)` calls that each
cost ~4.7 ms. A SAVE truncated the log under the streaming thread.

CASES
  1. join during writes — a replica started while a writer is mid-stream
     (strings AND aggregates) ends up with every key, the ones before it
     joined included, with the primary's exact values.
  2. SAVE mid-stream — the primary's checkpoint resets every WAL offset; the
     replica must resync, not read offsets the log no longer has.
  3. away and back — keys the primary DELETES while a replica is down must be
     gone after it returns (the snapshot is preceded by a flush), and keys
     written meanwhile must arrive.
  4. WAIT — counts a caught-up replica fast, and when it cannot be satisfied
     returns after its timeout measured on the clock, not 4.7x it.
  5. WAIT and a FULLRESYNC — the replica ACKs a snapshot only once it has
     APPLIED it, so whatever WAIT counted is readable on the replica.

    python3 tests/test_replication_resync.py [--port 2451]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready, wait_port_free  # noqa: E402

BIN = os.path.abspath(os.environ.get("PION_BIN", "./pion-server"))
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


class Node:
    def __init__(self, work, port, sub, extra):
        self.port, self.dir, self.extra = port, os.path.join(work, sub), extra
        os.makedirs(self.dir, exist_ok=True)
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen([BIN, "-p", str(self.port), "-w", "1", "--no-crash-log",
                                      "--no-auto-detect", "--no-auto-embed", "--cluster",
                                      "--cluster-host", "127.0.0.1"] + self.extra, cwd=self.dir,
                                     stdout=open(os.path.join(self.dir, "log"), "a"), stderr=subprocess.STDOUT)
        wait_ready(self.port, 30, proc=self.proc)
        c = Conn(self.port, timeout=30)
        return c

    def stop(self):
        if self.proc:
            self.proc.kill(); self.proc.wait(); self.proc = None
            wait_port_free(self.port)


def converge(a, b, keys, timeout=30.0):
    """Wait until the replica holds exactly the primary's value for every key.
    Returns the keys that still differ."""
    deadline = time.time() + timeout
    bad = keys
    while time.time() < deadline:
        bad = [k for k in keys if snapshot_of(a, k) != snapshot_of(b, k)]
        if not bad:
            return []
        time.sleep(0.3)
    return bad


def snapshot_of(c, k):
    t = c.cmd("TYPE", k)
    if t == "string":
        return ("string", c.cmd("GET", k))
    if t == "list":
        return ("list", tuple(c.cmd("LRANGE", k, "0", "-1")))
    if t == "hash":
        return ("hash", tuple(sorted(c.cmd("HGETALL", k))))
    if t == "set":
        return ("set", tuple(sorted(c.cmd("SMEMBERS", k))))
    if t == "zset":
        return ("zset", tuple(c.cmd("ZRANGE", k, "0", "-1", "WITHSCORES")))
    return (t,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2451)
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion-repl-resync-")
    pa, pb = args.port, args.port + 20
    primary = Node(work, pa, "primary", [])
    replica = Node(work, pb, "replica", ["--cluster-replica", "--cluster-primary-host", "127.0.0.1",
                                         "--cluster-primary-port", str(pa)])
    try:
        a = primary.start()
        # ── 1. a replica joins while a writer is mid-stream ──────────────────
        print("=== 1. join during writes ===")
        written = []
        stop = threading.Event()

        def writer():
            w = Conn(pa, timeout=30)
            i = 0
            while not stop.is_set():
                w.pipeline([("SET", f"{{r}}s:{i}", f"v{i}"),
                            ("RPUSH", f"{{r}}l:{i % 7}", f"e{i}"),
                            ("HSET", f"{{r}}h:{i % 5}", f"f{i}", f"x{i}"),
                            ("ZADD", f"{{r}}z", str(i), f"m{i}")])
                written.append(i)
                i += 1
                time.sleep(0.001)             # paced, so the stream spans the join
        t = threading.Thread(target=writer)
        t.start()
        time.sleep(0.5)                       # history exists before the replica
        at_join = len(written)
        b = replica.start()
        b.cmd("READONLY")
        time.sleep(1.0)                       # ...and keeps growing after it
        stop.set()
        t.join()
        n = len(written)
        check(f"writes landed before ({at_join}) and after ({n - at_join}) the replica joined",
              at_join > 50 and n - at_join > 50, f"{at_join} / {n - at_join}")
        keys = ([f"{{r}}s:{i}" for i in range(n)] + [f"{{r}}l:{j}" for j in range(7)]
                + [f"{{r}}h:{j}" for j in range(5)] + ["{r}z"])
        acked = a.cmd("WAIT", "1", "10000")
        check("WAIT counts the replica once it has applied everything", acked == 1, repr(acked))
        bad = converge(a, b, keys)
        check(f"replica converged on all {len(keys)} keys ({n} writer rounds, joined mid-stream)",
              not bad, f"{len(bad)} differ, e.g. {bad[:3]}")

        # ── 2. SAVE mid-stream resets every WAL offset ───────────────────────
        print("=== 2. SAVE mid-stream ===")
        a.pipeline([("SET", f"{{r}}pre-save:{i}", str(i)) for i in range(200)])
        check("SAVE", a.cmd("SAVE") == "OK")
        a.pipeline([("SET", f"{{r}}post-save:{i}", str(i)) for i in range(200)])
        keys2 = [f"{{r}}pre-save:{i}" for i in range(200)] + [f"{{r}}post-save:{i}" for i in range(200)]
        bad = converge(a, b, keys2)
        check("replica holds every key written before and after the primary's SAVE",
              not bad, f"{len(bad)} differ, e.g. {bad[:3]}")
        a.cmd("SET", "{r}after-save-probe", "1")
        bad = converge(a, b, ["{r}after-save-probe"])
        check("the stream is live again after the SAVE", not bad)

        # ── 3. away and back: deletes must not survive the resync ────────────
        print("=== 3. away and back ===")
        replica.stop()
        a.cmd("DEL", "{r}s:0", "{r}s:1", "{r}l:0", "{r}h:0")
        a.pipeline([("SET", f"{{r}}while-away:{i}", str(i)) for i in range(100)])
        b = replica.start()
        b.cmd("READONLY")
        keys3 = ["{r}s:0", "{r}s:1", "{r}l:0", "{r}h:0"] + [f"{{r}}while-away:{i}" for i in range(100)]
        bad = converge(a, b, keys3)
        check("keys deleted and written while the replica was down are deleted / present",
              not bad, f"{len(bad)} differ, e.g. {bad[:3]}")
        check("every earlier key is still right after the resync", not converge(a, b, keys[:500], 10))

        # ── 4. WAIT honours its count and its clock ──────────────────────────
        print("=== 4. WAIT ===")
        a.cmd("SET", "{r}w", "1")
        t0 = time.time(); r = a.cmd("WAIT", "1", "2000"); dt = time.time() - t0
        check("WAIT 1 2000 with a live replica answers 1", r == 1, repr(r))
        check("... and well inside its timeout", dt < 1.0, f"{dt:.2f} s")
        t0 = time.time(); r = a.cmd("WAIT", "2", "300"); dt = time.time() - t0
        check("WAIT 2 300 with one replica answers 1", r == 1, repr(r))
        check("... after ~300 ms, not ~1.4 s", 0.25 <= dt < 0.8, f"{dt:.2f} s")

        # ── 5. WAIT must not count a snapshot the replica has not applied ────
        print("=== 5. WAIT during a FULLRESYNC ===")
        replica.stop()
        N = 300_000      # ~12 MB of snapshot: several 4 MB drains on the replica
        for base in range(0, N, 5000):
            a.pipeline([("SET", f"{{r}}big:{i}", f"value-{i}") for i in range(base, base + 5000)])
        b = replica.start()
        b.cmd("READONLY")
        deadline, r = time.time() + 60, 0
        while time.time() < deadline:
            r = a.cmd("WAIT", "1", "20")
            if r == 1:
                break
        check("WAIT counts the replica once its FULLRESYNC is applied", r == 1, repr(r))
        # Read the replica at once: whatever WAIT counted must be readable now.
        probe = [f"{{r}}big:{i}" for i in (0, N // 2, N - 2, N - 1)]
        got = [b.cmd("GET", k) for k in probe]
        want = [f"value-{i}".encode() for i in (0, N // 2, N - 2, N - 1)]
        check("every key is on the replica the moment WAIT counts it", got == want,
              f"{sum(g is None for g in got)} of {len(probe)} sampled keys missing")
    finally:
        replica.stop()
        primary.stop()
        shutil.rmtree(work, ignore_errors=True)
    print(f"{len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
