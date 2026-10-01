#!/usr/bin/env python3
"""gh #145 — a cross-index FT.OPTIMIZE must not fail silently.

Pion holds ONE index per server: `FT.CREATE`/`FT.OPTIMIZE` on index B replace
the HNSW graph and the BM25 postings that were serving index A. That is the
contract, but until this fix a query on A answered `*0` — indistinguishable
from "your query matched nothing", which is exactly the failure mode gh #139
removed for the not-built case. Now both the BM25 and the KNN paths report
which index actually owns the current state.

Multi-index support is NOT what this tests. It pins the single-index contract
and its diagnostics:
  * a query naming the displaced index errors, and names both sides
  * the live index keeps working
  * re-ingesting and re-optimizing the displaced index restores it
  * the normal single-index flow never trips the guard (no false positives)

Usage:
    python3 tests/test_gh145_index_scope.py [--binary ./pion-server] [--port 7991]
"""

import argparse
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


class Client:
    def __init__(self, port, timeout=30):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        # makefile('rb'), never manual slicing — see python_resp_array_slicing_trap
        self.f = self.sock.makefile("rb")

    def cmd(self, *args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.sock.sendall(out)
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        tag, body = line[:1], line[1:-2]
        if tag == b"*":
            n = int(body)
            return [] if n < 0 else [self._read() for _ in range(n)]
        if tag == b"$":
            n = int(body)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if tag == b"-":
            return Exception(body.decode(errors="replace"))
        if tag == b":":
            return int(body)
        return body

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def wait_ready(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = Client(port, timeout=2)
            r = c.cmd("PING")
            c.close()
            if r == b"PONG":
                return True
        except (OSError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


SCHEMA = ["SCHEMA", "body", "TEXT", "vec", "VECTOR", "HNSW", "6",
          "TYPE", "FLOAT32", "DIM", "4", "DISTANCE_METRIC", "L2"]
VEC_A = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
VEC_B = struct.pack("<4f", 0.0, 0.0, 1.0, 0.0)


def knn(c, index, blob, k=2):
    """Canonical KNN form — the bare-blob shorthand never sets blob_set, so it
    returns an empty array for reasons unrelated to index identity."""
    return c.cmd("FT.SEARCH", index, f"*=>[KNN {k} @vec $q]", "PARAMS", "2", "q", blob)


def bm25(c, index, query, k=5):
    r = c.cmd("FT.SEARCH", index, "BM25", query, "K", str(k))
    if isinstance(r, Exception):
        return r
    return [int(r[i]) for i in range(1, len(r), 2)]


def ingest_a(c):
    c.cmd("HSET", "9000", "vec", VEC_A)
    c.cmd("HSET", "9000", "body", "alpha bravo charlie delta")
    c.cmd("HSET", "9001", "vec", struct.pack("<4f", 0.0, 1.0, 0.0, 0.0))
    c.cmd("HSET", "9001", "body", "echo foxtrot golf")


# ── Test 1: the issue's repro ────────────────────────────────────────────────
def test_cross_index_clobber(port):
    print("\n[1] cross-index FT.OPTIMIZE (the reported repro)")
    c = Client(port)
    try:
        check("FT.CREATE clobA", c.cmd("FT.CREATE", "clobA", *SCHEMA) == b"OK")
        ingest_a(c)
        check("FT.OPTIMIZE clobA", c.cmd("FT.OPTIMIZE", "clobA") == b"OK")
        before = bm25(c, "clobA", "alpha bravo")
        check("clobA BM25 serves results before the clobber",
              not isinstance(before, Exception) and len(before) == 1, before)

        check("FT.CREATE clobB", c.cmd("FT.CREATE", "clobB", *SCHEMA) == b"OK")
        c.cmd("HSET", "9500", "vec", VEC_B)
        c.cmd("HSET", "9500", "body", "zulu yankee xray")
        check("FT.OPTIMIZE clobB", c.cmd("FT.OPTIMIZE", "clobB") == b"OK")

        # The bug: this used to be a silent [0].
        after = bm25(c, "clobA", "alpha bravo")
        ok = isinstance(after, Exception)
        check("displaced index BM25 errors instead of returning empty", ok,
              str(after)[:100])
        if ok:
            msg = str(after)
            check("error names the queried index", "clobA" in msg)
            check("error names the index that owns the postings", "clobB" in msg)
            check("error explains the one-index-per-server contract",
                  "one index at a time" in msg)

        live = bm25(c, "clobB", "zulu yankee")
        check("the live index still serves BM25",
              not isinstance(live, Exception) and len(live) == 1, live)
    finally:
        c.close()


# ── Test 2: the KNN path had the same silent failure ─────────────────────────
def test_knn_path(port):
    print("\n[2] KNN path reports the displaced index too")
    c = Client(port)
    try:
        r = knn(c, "clobA", VEC_A)
        ok = isinstance(r, Exception)
        check("displaced index KNN errors instead of returning empty", ok,
              str(r)[:100])
        if ok:
            check("KNN error names both indexes",
                  "clobA" in str(r) and "clobB" in str(r))

        r = knn(c, "clobB", VEC_B)
        check("the live index still serves KNN",
              not isinstance(r, Exception) and r not in ([], [0]), str(r)[:80])
    finally:
        c.close()


# ── Test 3: FT.HYBRID fuses BM25, so it must refuse too ──────────────────────
def test_hybrid_path(port):
    print("\n[3] FT.HYBRID refuses rather than degrading to vector-only")
    c = Client(port)
    try:
        r = c.cmd("FT.HYBRID", "clobA", "alpha bravo", VEC_A, "K", "2")
        check("FT.HYBRID on the displaced index errors",
              isinstance(r, Exception), str(r)[:100])
        r = c.cmd("FT.HYBRID", "clobB", "zulu yankee", VEC_B, "K", "2")
        check("FT.HYBRID on the live index still works",
              not isinstance(r, Exception), str(r)[:80])
    finally:
        c.close()


# ── Test 4: the documented recovery path actually recovers ───────────────────
def test_recovery(port):
    print("\n[4] re-ingest + FT.OPTIMIZE restores the displaced index")
    c = Client(port)
    try:
        check("FT.CREATE clobA again", c.cmd("FT.CREATE", "clobA", *SCHEMA) == b"OK")
        ingest_a(c)
        check("FT.OPTIMIZE clobA", c.cmd("FT.OPTIMIZE", "clobA") == b"OK")
        r = bm25(c, "clobA", "alpha bravo")
        check("clobA serves BM25 again",
              not isinstance(r, Exception) and len(r) == 1, r)
        # ...and now B is the displaced one — the guard is symmetric.
        r = bm25(c, "clobB", "zulu yankee")
        check("clobB is now the one that errors", isinstance(r, Exception),
              str(r)[:80])
    finally:
        c.close()


# ── Test 5: no false positives on the normal single-index flow ───────────────
def test_no_false_positive(port2):
    print("\n[5] single-index flow never trips the guard")
    c = Client(port2)
    try:
        check("FT.CREATE solo", c.cmd("FT.CREATE", "solo", *SCHEMA) == b"OK")
        for i, text in ((1, "alpha bravo charlie"), (2, "delta echo foxtrot")):
            c.cmd("FT.ADDTEXT", "solo", str(i), text)
        check("FT.OPTIMIZE solo", c.cmd("FT.OPTIMIZE", "solo") == b"OK")

        # Repeated optimizes of the SAME index must stay serviceable — this is
        # the ordinary re-ingest loop and it must not look like a clobber.
        for round_i in range(3):
            c.cmd("FT.ADDTEXT", "solo", str(10 + round_i), f"golf hotel india{round_i}")
            c.cmd("FT.OPTIMIZE", "solo")
            r = bm25(c, "solo", "alpha bravo")
            if not check(f"re-optimize round {round_i} keeps solo serving",
                         not isinstance(r, Exception) and r == [1], r):
                return
        # solo is text-only (FT.ADDTEXT, no vectors), so an empty KNN result is
        # correct. What matters here is that it does not raise the index-name
        # error — that would be the false positive this test exists to catch.
        r = knn(c, "solo", VEC_A)
        check("KNN on the single index does not raise an index-name error",
              not isinstance(r, Exception), str(r)[:80])

        # Byte-exact comparison: index names are case-sensitive, same as the
        # pre-existing KNN check. Pin it so it cannot drift silently.
        r = bm25(c, "SOLO", "alpha bravo")
        check("index names are case-sensitive", isinstance(r, Exception),
              str(r)[:80])

        # A dropped index reverts to the gh #139 'not built' error, not a
        # stale-owner mismatch — the postings are gone, not reassigned.
        check("FT.DROPINDEX solo", c.cmd("FT.DROPINDEX", "solo") == b"OK")
        c.cmd("FT.CREATE", "solo", *SCHEMA)
        c.cmd("FT.OPTIMIZE", "solo")
        r = bm25(c, "solo", "alpha bravo")
        check("after DROPINDEX the error is 'not built', not a mismatch",
              isinstance(r, Exception) and "not built" in str(r), str(r)[:90])
    finally:
        c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7991)
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}")
        return 2

    workdir = tempfile.mkdtemp(prefix="pion-gh145-")
    print(f"gh #145 index-scope tests — binary={args.binary} port={args.port}")

    def serve(port, subdir):
        d = os.path.join(workdir, subdir)
        os.makedirs(d, exist_ok=True)
        return subprocess.Popen(
            [args.binary, "--profile", "vector", "-w", "1", "-p", str(port),
             "--no-auto-detect", "--no-auto-embed"],
            cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )

    def stop(p):
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    proc = serve(args.port, "clobber")
    try:
        if not check("server becomes ready", wait_ready(args.port)):
            return 1
        test_cross_index_clobber(args.port)
        test_knn_path(args.port)
        test_hybrid_path(args.port)
        test_recovery(args.port)
    finally:
        stop(proc)

    # Fresh process: the false-positive check must not inherit clobbered state.
    port2 = args.port + 10
    proc = serve(port2, "solo")
    try:
        if not check("second server becomes ready", wait_ready(port2)):
            return 1
        test_no_false_positive(port2)
    finally:
        stop(proc)

    print(f"\n{'='*60}")
    print(f"PASS {len(PASS)}  FAIL {len(FAIL)}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
