#!/usr/bin/env python3
"""gh #365 — FT.SEARCH `score` is the index metric's distance, in its units.

Before the fix the score was the beam's distance in QUANTIZED code space: a
vector matched against itself scored ~750-1500 and a neighbour ~780,000, so a
client reading it as RediSearch documents it (squared L2 under L2, 1 - cos
under COSINE) got nonsense — pion-autogen's score_threshold kept everything,
pion-mcp's semantic cache never hit.

The server now recomputes the distance for the k rows it returns, from the FP32
re-rank copy when one exists, else from the INT8 codes dequantized with their
calibration. Checked here against exact values computed in Python:

  COSINE  unit vectors               score ≈ 1 - cos(q, v)   (±0.01)
  L2      OpenAI-scale vectors       score ≈ |q - v|^2       (±2%, self ≈ 0)
  both    rows ascend by score; the self-match is first
  L2      raw N(0,1) vectors         same, self ≈ 0 (≤ 0.5% of a neighbour)
  -w 2    a worker that BORROWED the index answers the same (connections
          opened concurrently, so both workers serve)

The raw-scale suite is gh #376: the HSET-ingest build did not calibrate the
quantizer, so every component outside ±0.2 was clipped at ingest and a vector
scored ~1,134 against itself.

Usage: python3 tests/test_gh365_scores.py [./pion-server]
"""
import math
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

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6474
DIM, N = 1536, 200
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


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


def start(workers):
    d = tempfile.mkdtemp(prefix="pion_gh365_")
    cmd = [BINARY, "-p", str(PORT), "-w", str(workers), "--no-auto-detect", "--no-auto-embed"]
    if workers > 1:
        cmd.append("--independent-workers")
    p = subprocess.Popen(cmd, cwd=d, stdout=open(os.path.join(d, "log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(200):
        try:
            c = Client(PORT)
            if c.cmd("PING") == b"PONG":
                return p, d, c
        except OSError:
            time.sleep(0.1)
    raise SystemExit("server did not start")


def stop(p, d):
    p.kill()
    p.wait()
    shutil.rmtree(d, ignore_errors=True)


def vectors(metric, scale=0.04):
    rnd = random.Random(365)
    out = []
    for _ in range(N):
        v = [rnd.gauss(0, 1) for _ in range(DIM)]
        if metric == "COSINE":
            n = math.sqrt(sum(x * x for x in v))
            v = [x / n for x in v]
        else:
            v = [x * scale for x in v]   # 0.04 = OpenAI-embedding scale, |x| < 0.2
        out.append(v)
    return out


def exact(metric, q, v):
    if metric == "COSINE":
        nq = math.sqrt(sum(x * x for x in q))
        nv = math.sqrt(sum(x * x for x in v))
        return 1.0 - sum(a * b for a, b in zip(q, v)) / (nq * nv)
    return sum((a - b) ** 2 for a, b in zip(q, v))


def load(c, metric, vs):
    c.cmd("FT.CREATE", "sx", "ON", "HASH", "PREFIX", "1", "s:", "SCHEMA", "vec", "VECTOR", "HNSW", "6",
          "TYPE", "FLOAT32", "DIM", DIM, "DISTANCE_METRIC", metric)
    for i, v in enumerate(vs):
        c.cmd("HSET", f"s:{i}", "vec", struct.pack(f"<{DIM}f", *v), "n", str(i))
    return c.cmd("FT.OPTIMIZE", "sx")


def query(c, v, k=5):
    r = c.cmd("FT.SEARCH", "sx", f"*=>[KNN {k} @vec $q]", "PARAMS", "2", "q",
              struct.pack(f"<{DIM}f", *v), "DIALECT", "2")
    if not isinstance(r, list):
        return None
    rows = []
    for key, fields in zip(r[1::2], r[2::2]):
        f = dict(zip(fields[0::2], fields[1::2]))
        rows.append((int(key.split(b":")[1]), float(f[b"score"])))
    return rows


def suite(metric, workers, scale=0.04):
    print(f"[{metric}, -w {workers}" + (f", scale {scale}]" if metric == "L2" else "]"))
    p, d, c = start(workers)
    try:
        vs = vectors(metric, scale)
        check("FT.OPTIMIZE", load(c, metric, vs) == b"OK")
        conns = [c]
        if workers > 1:
            extra: list = [None] * 7

            def open_one(i):
                extra[i] = Client(PORT)
            ts = [threading.Thread(target=open_one, args=(i,)) for i in range(7)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            conns += extra
        tol_abs = 0.01 if metric == "COSINE" else 0.0
        bad, order_bad, self_bad, worst = 0, 0, 0, 0.0
        for qi in range(0, N, 10):
            cc = conns[qi % len(conns)]
            rows = query(cc, vs[qi])
            if not rows:
                bad += 1
                continue
            if rows[0][0] != qi:
                self_bad += 1
            scores = [s for _, s in rows]
            if scores != sorted(scores):
                order_bad += 1
            others = [exact(metric, vs[qi], vs[doc]) for doc, _ in rows if doc != qi]
            for doc, s in rows:
                e = exact(metric, vs[qi], vs[doc])
                err = abs(s - e)
                tol = tol_abs if metric == "COSINE" else max(0.02 * e, 0.002)
                if metric == "L2" and doc == qi and others:
                    # INT8 with 3-sigma calibration leaves a self-distance of
                    # ~0.1% of a neighbour's on Gaussian data (the clipped
                    # tails); uncalibrated it was ~40%
                    tol = max(tol, 5e-3 * min(others))
                worst = max(worst, err)
                if err > tol:
                    bad += 1
        check(f"scores are {'1 - cos' if metric == 'COSINE' else 'squared L2'} within tolerance",
              bad == 0, f"{bad} rows off, worst |err| {worst:.4f}")
        check("rows ascend by score", order_bad == 0, f"{order_bad} queries out of order")
        check("the self-match is first", self_bad == 0, f"{self_bad} queries")
        check("server alive", p.poll() is None)
    finally:
        stop(p, d)


def main():
    print(f"[gh #365] {BINARY}")
    suite("COSINE", 1)
    suite("L2", 1)
    suite("L2", 1, scale=1.0)
    suite("COSINE", 2)
    print(f"\n{'FAILED' if FAIL else 'PASSED'}: {len(FAIL)} failure(s)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
