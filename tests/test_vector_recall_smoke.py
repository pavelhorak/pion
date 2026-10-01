#!/usr/bin/env python3
"""Vector search must find the right neighbours — release smoke test (gh #341).

Standard library only, like the other release-gate tests: the CI step runs it
on every platform before packaging. Until this existed the release gate was
test_raw + test_parity, which issue no vector query, so an x86-64-v2 build
whose INT8 dot product was wrong (recall@10 0.001, PR #341) would have been
packaged green.

Ground truth is PLANTED rather than brute-forced (a pure-Python brute force
over 1536-dim vectors is minutes): CLUSTERS well-separated random centres,
PER cluster points = centre + small noise, and each query is a centre plus
smaller noise, so its true top-K is exactly its own cluster. The margin is
large — within-cluster distance ~0.07, between-cluster ~1.4 on unit-scale
data — so a correct index gets ~1.0 and any real distance bug collapses it.
DIM is 1536 on purpose: only that width takes the tuned INT8 beam
(_beam_search_1536), which is where #341 lived.

    python3 tests/test_vector_recall_smoke.py --port 1974
"""
import argparse
import math
import random
import socket
import struct
import sys
import time

DIM = 1536
CLUSTERS = 150
PER = 10
QUERIES = 60
K = 10
MIN_RECALL = 0.90
INDEX = "rsmoke"


class Resp:
    def __init__(self, port: int):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=120)
        self.buf = b""

    @staticmethod
    def encode(*args) -> bytes:
        out = [b"*%d\r\n" % len(args)]
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            elif isinstance(a, int):
                a = str(a).encode()
            out.append(b"$%d\r\n%s\r\n" % (len(a), a))
        return b"".join(out)

    def _line(self) -> bytes:
        while b"\r\n" not in self.buf:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("server closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def _exact(self, n: int) -> bytes:
        while len(self.buf) < n + 2:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("server closed the connection")
            self.buf += chunk
        data, self.buf = self.buf[:n], self.buf[n + 2:]
        return data

    def read(self):
        line = self._line()
        t, rest = line[:1], line[1:]
        if t in (b"+",):
            return rest.decode()
        if t == b"-":
            raise RuntimeError(rest.decode())
        if t == b":":
            return int(rest)
        if t == b"$":
            n = int(rest)
            return None if n < 0 else self._exact(n)
        if t == b"*":
            n = int(rest)
            return None if n < 0 else [self.read() for _ in range(n)]
        raise RuntimeError(f"unexpected RESP line {line!r}")

    def call(self, *args):
        self.s.sendall(self.encode(*args))
        return self.read()

    def pipeline(self, cmds):
        self.s.sendall(b"".join(self.encode(*c) for c in cmds))
        return [self.read() for _ in cmds]


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def pack(v) -> bytes:
    return struct.pack(f"<{DIM}f", *v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()
    rng = random.Random(0x341)

    centres = [unit([rng.gauss(0, 1) for _ in range(DIM)]) for _ in range(CLUSTERS)]
    r = Resp(args.port)
    r.call("FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", "rs:", "SCHEMA", "vec",
           "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", DIM, "DISTANCE_METRIC", "L2")
    cmds = []
    for c in range(CLUSTERS):
        for p in range(PER):
            v = [x + rng.gauss(0, 0.05 / math.sqrt(DIM)) for x in centres[c]]
            cmds.append(("HSET", f"rs:{c}:{p}", "vec", pack(v)))
            if len(cmds) == 250:
                r.pipeline(cmds)
                cmds = []
    if cmds:
        r.pipeline(cmds)
    t0 = time.time()
    r.call("FT.OPTIMIZE", INDEX)
    built = time.time() - t0

    hits = 0
    for qi in range(QUERIES):
        c = rng.randrange(CLUSTERS)
        q = [x + rng.gauss(0, 0.02 / math.sqrt(DIM)) for x in centres[c]]
        res = r.call("FT.SEARCH", INDEX, f"*=>[KNN {K} @vec $q EF_RUNTIME 150 as score]",
                     "PARAMS", "2", "q", pack(q), "SORTBY", "score",
                     "LIMIT", "0", str(K), "DIALECT", "2")
        keys = [res[i].decode() for i in range(1, len(res), 2)]
        hits += sum(1 for k in keys if k.startswith(f"rs:{c}:"))
    r.call("FT.DROPINDEX", INDEX)
    recall = hits / (QUERIES * K)
    print(f"vector recall smoke: recall@{K} {recall:.3f} over {QUERIES} queries "
          f"({CLUSTERS * PER} vectors, dim {DIM}, FT.OPTIMIZE {built:.1f}s)")
    if recall < MIN_RECALL:
        print(f"FAIL: recall@{K} {recall:.3f} < {MIN_RECALL} — vector distances are wrong on this build")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
