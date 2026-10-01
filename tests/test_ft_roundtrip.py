#!/usr/bin/env python3
"""FT.* round-trip properties — no oracle needed, only consistency.

Written 2026-09-26 after three FT bugs surfaced in one day, each of which a
property check states in one line and nothing tested:
  #360  a multi-field HSET drops the indexed vector from the hash
  #361  an unrecognised FT.SEARCH query form answers [] instead of an error
  #357  a worker without the doc substituted the key for the `id` field
Properties, each on data whose right answer is known by construction:

  P1 write/read   every field HSET writes, HGET returns — including the vector
  P2 shape        every FT.SEARCH reply is [count, key, fields, ...] with
                  count == number of pairs and fields a name/value list
  P3 self-first   every stored vector, queried exactly, comes back first
                  — on a 4-worker server, from connections spread across
                  workers (serially opened connections all land on one)
  P4 refuse       a query the server does not understand is an ERROR
  P5 counts       FT.INFO num_docs equals the number of vectors inserted
  P6 filter       every result of a NUMERIC-filtered query satisfies it

A known bug is XFAIL with its issue; if it starts passing that is XPASS and
FAILS the run, so the marker is removed on purpose.

Usage: python3 tests/test_ft_roundtrip.py [--binary ./pion-server] [--port 6430]
"""
import argparse
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM, N = 1536, 64
FAIL, XFAIL = [], []


def check(name, ok, detail=""):
    print(f"  {'PASS ' if ok else 'FAIL '} {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def xfail(name, ok, issue, detail=""):
    if ok:
        print(f"  XPASS {name} — {issue} looks fixed: make this a check()")
        FAIL.append(f"XPASS {name}")
    else:
        print(f"  XFAIL {name} ({issue})" + (f" — {detail}" if detail else ""))
        XFAIL.append(name)


class Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.f = self.s.makefile("rb")

    def cmd(self, *args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            a = a if isinstance(a, bytes) else str(a).encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.s.sendall(out)
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        t, body = line[:1], line[1:-2]
        if t == b"*":
            n = int(body)
            return None if n < 0 else [self._read() for _ in range(n)]
        if t == b"$":
            n = int(body)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if t == b"-":
            return RuntimeError(body.decode(errors="replace"))
        if t == b":":
            return int(body)
        return body


def vec(seed):
    rnd = random.Random(seed)
    return struct.pack(f"<{DIM}f", *[rnd.gauss(0, 1) for _ in range(DIM)])


def knn(c, q, k=3, extra=()):
    return c.cmd("FT.SEARCH", "idx", f"*=>[KNN {k} @embedding $v]", *extra, "PARAMS", "2", "v", q, "DIALECT", "2")


def well_formed(reply):
    if not isinstance(reply, list) or not reply or not isinstance(reply[0], int):
        return False
    body = reply[1:]
    if len(body) % 2 or reply[0] != len(body) // 2:
        return False
    return all(isinstance(k, bytes) and isinstance(f, list) and len(f) % 2 == 0 for k, f in zip(body[0::2], body[1::2]))


class Server:
    def __init__(self, binary, port, workers):
        self.port, self.dir = port, tempfile.mkdtemp(prefix="pion_ftrt_")
        self.log = os.path.join(self.dir, "server.log")
        cmd = [binary, "-p", str(port), "-w", str(workers), "--no-auto-detect", "--no-auto-embed"]
        if workers > 1:
            cmd.append("--independent-workers")
        self.p = subprocess.Popen(cmd, cwd=self.dir, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        for _ in range(150):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                return
            except OSError:
                time.sleep(0.1)
        raise SystemExit("server did not start")

    def healthy(self):
        text = open(self.log, errors="replace").read()
        return "SIGSEGV" not in text and self.p.poll() is None

    def stop(self):
        self.p.kill()
        self.p.wait()
        shutil.rmtree(self.dir, ignore_errors=True)


def build(c):
    c.cmd("FT.CREATE", "idx", "ON", "HASH", "PREFIX", "1", "d:", "SCHEMA", "embedding", "VECTOR", "HNSW", "6",
          "TYPE", "FLOAT32", "DIM", DIM, "DISTANCE_METRIC", "L2", "price", "NUMERIC")
    for i in range(N):
        c.cmd("HSET", f"d:{i}", "embedding", vec(i), "text", f"doc {i}", "price", str(i))
    return c.cmd("FT.OPTIMIZE", "idx")


def single_worker(binary, port):
    print("[single worker]")
    s = Server(binary, port, 1)
    try:
        c = Client(port)
        check("FT.OPTIMIZE succeeds", build(c) == b"OK")
        # P1
        got = c.cmd("HGETALL", "d:5")
        fields = dict(zip(got[0::2], got[1::2])) if isinstance(got, list) else {}
        check("P1 non-vector fields read back", fields.get(b"text") == b"doc 5" and fields.get(b"price") == b"5", str(list(fields)))
        check("P1 the vector field reads back (gh #360)", c.cmd("HGET", "d:5", "embedding") == vec(5))
        # P2 + P3
        shapes, firsts = [], []
        for i in range(N):
            r = knn(c, vec(i))
            shapes.append(well_formed(r))
            firsts.append(isinstance(r, list) and len(r) > 1 and r[1] == f"d:{i}".encode())
        check("P2 every KNN reply is well formed", all(shapes), f"{shapes.count(False)} malformed")
        check("P3 every stored vector retrieves itself first", all(firsts), f"{firsts.count(False)}/{N} missed")
        # P4
        for label, args in [("blob K k", ("FT.SEARCH", "idx", vec(1), "K", "3")),
                            ("blob KNN k", ("FT.SEARCH", "idx", vec(1), "KNN", "3"))]:
            r = c.cmd(*args)
            check(f"P4 unrecognised form '{label}' is an error (gh #361)", isinstance(r, RuntimeError), repr(r)[:60])
        r = c.cmd("FT.SEARCH", "idx", "*=>[KNN 3 @nonexistent $v]", "PARAMS", "2", "v", vec(1), "DIALECT", "2")
        check("P4 KNN on an unknown field is an error (gh #361)", isinstance(r, RuntimeError), repr(r)[:60])
        r = c.cmd("FT.SEARCH", "no_such_index", "*=>[KNN 3 @embedding $v]", "PARAMS", "2", "v", vec(1), "DIALECT", "2")
        check("P4 an unknown index is an error", isinstance(r, RuntimeError), repr(r)[:80])
        # P5
        info = c.cmd("FT.INFO", "idx")
        kv = dict(zip(info[0::2], info[1::2])) if isinstance(info, list) else {}
        check("P5 FT.INFO num_docs == vectors inserted", str(kv.get(b"num_docs", b"")).strip("b'") == str(N), repr(kv.get(b"num_docs")))
        # P6
        def prices_of(r):
            keys = r[1::2] if isinstance(r, list) else []
            return [int(c.cmd("HGET", k, "price")) for k in keys]
        # Pion's documented form (doc/vector_engine.md "Metadata Filters")
        got = prices_of(knn(c, vec(40), k=10, extra=("FILTER", "@price:[10 20]")))
        check("P6 NUMERIC FILTER (documented @f:[a b] form) returns results", len(got) > 0, str(got))
        check("P6 every result satisfies the documented filter (gh #367)", all(10 <= p <= 20 for p in got), str(got))
        # RediSearch's standard form: silently ignored today — answers unfiltered
        got = prices_of(knn(c, vec(40), k=10, extra=("FILTER", "price", "10", "20")))
        check("P6 RediSearch 'FILTER price 10 20' filters (gh #361)",
              len(got) > 0 and all(10 <= p <= 20 for p in got), str(got))
        # P6 — the other filter spellings, and refusals (gh #361 / #367)
        def knn_q(q, k=10, extra=()):
            return c.cmd("FT.SEARCH", "idx", q, *extra, "PARAMS", "2", "v", vec(40), "DIALECT", "2")
        got = prices_of(knn_q("(@price:[10 20])=>[KNN 10 @embedding $v]"))
        check("P6 RediSearch prefilter (@price:[10 20])=>[KNN …] filters",
              len(got) == 10 and all(10 <= p <= 20 for p in got), str(got))
        got = prices_of(knn(c, vec(40), k=10, extra=("FILTER", "price", "[10 20]")))
        check("P6 'FILTER price [10 20]' filters", len(got) == 10 and all(10 <= p <= 20 for p in got), str(got))
        got = prices_of(knn(c, vec(40), k=5, extra=("FILTER", "@price:[-inf 3]")))
        check("P6 -inf bound", sorted(got) == [0, 1, 2, 3], str(got))
        got = prices_of(knn(c, vec(40), k=3, extra=("FILTER", "@price:[10 20]", "FILTER", "@price:[15 +inf]")))
        check("P6 two FILTER clauses AND together", len(got) == 3 and all(15 <= p <= 20 for p in got), str(got))
        for label, extra in [("garbage FILTER", ("FILTER", "price>10")),
                             ("exclusive bound", ("FILTER", "@price:[(10 20]")),
                             ("non-numeric bound", ("FILTER", "@price:[ten 20]")),
                             ("OR of clauses", ("FILTER", "@price:[1 2] | @price:[5 6]"))]:
            r = knn(c, vec(40), k=3, extra=extra)
            check(f"P6 {label} is refused, not ignored", isinstance(r, RuntimeError), repr(r)[:80])
        for label, args in [
                ("missing $param", ("FT.SEARCH", "idx", "*=>[KNN 3 @embedding $nope]", "PARAMS", "2", "v", vec(1))),
                ("blob of the wrong size", ("FT.SEARCH", "idx", "*=>[KNN 3 @embedding $v]", "PARAMS", "2", "v", b"\0" * 12)),
                ("no query at all", ("FT.SEARCH", "idx")),
                ("plain text query", ("FT.SEARCH", "idx", "hello world"))]:
            r = c.cmd(*args)
            check(f"P4 {label} is an error (gh #361)", isinstance(r, RuntimeError), repr(r)[:80])
        r = c.cmd("FT.SEARCH", "idx", "*=>[KNN 3 @embedding $v EF_RUNTIME 50 AS dist]",
                  "PARAMS", "2", "v", vec(1), "SORTBY", "dist", "LIMIT", "0", "3", "DIALECT", "2")
        check("P4 EF_RUNTIME / AS / SORTBY / LIMIT are accepted", well_formed(r) and r[1] == b"d:1", repr(r)[:80])
        # P7 score order (gh #365) and scale (gh #376). This index holds N(0,1)
        # vectors; the HSET-ingest build used to quantize them with the
        # uncalibrated ±0.2 range, clipping nearly every component, so the
        # self-match scored ~1,134. Calibrated, it is a small fraction of the
        # distance to any real neighbour (~2·dim for unit-variance data).
        r = knn(c, vec(7), k=3)
        scores = [float(dict(zip(f[0::2], f[1::2]))[b"score"]) for f in r[2::2]]
        check("P7 scores ascend and the self-match is first",
              scores == sorted(scores) and r[1] == b"d:7", str(scores))
        check("P7 the self-match scores ~0, not the clipping error (gh #376)",
              len(scores) == 3 and scores[0] < 5e-3 * scores[1], str(scores))
        check("server healthy after the single-worker suite", s.healthy())
    finally:
        s.stop()


def multi_worker(binary, port):
    print("[4 workers, connections opened concurrently]")
    s = Server(binary, port, 4)
    try:
        check("FT.OPTIMIZE succeeds", build(Client(port)) == b"OK")
        conns = [None] * 16

        def open_one(i):
            conns[i] = Client(port)
        ts = [threading.Thread(target=open_one, args=(i,)) for i in range(16)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        misses, malformed, fabricated = 0, 0, 0
        for j, c in enumerate(conns):
            for i in range(j, N, 16):
                r = knn(c, vec(i))
                if not well_formed(r):
                    malformed += 1
                    continue
                if r[1] != f"d:{i}".encode():
                    misses += 1
                named = dict(zip(r[2][0::2], r[2][1::2]))
                if b"id" in named and named[b"id"] != str(i).encode() and named[b"id"] != f"d:{i}".encode():
                    fabricated += 1
        check("P2 replies well formed from every worker", malformed == 0, f"{malformed} malformed")
        check("P3 self-first from every worker", misses == 0, f"{misses}/{N} missed")
        check("no substituted id fields (gh #357)", fabricated == 0, f"{fabricated}")
        check("server healthy after the multi-worker suite", s.healthy())
    finally:
        s.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=6430)
    a = ap.parse_args()
    binary = os.path.abspath(a.binary)
    single_worker(binary, a.port)
    multi_worker(binary, a.port)
    print(f"\n{'FAIL: ' + '; '.join(FAIL) if FAIL else 'ALL PASS'}  ({len(XFAIL)} known-bug XFAIL)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
