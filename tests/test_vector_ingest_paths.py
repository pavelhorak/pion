#!/usr/bin/env python3
"""A vector written to a hash is indexed however it is written (#43).

Only the fast path's HSET used to send the index's vector field to the index.
The slow path's HSET, HMSET and HSETNX stored the field and indexed nothing,
so each of these wrote a document FT.SEARCH could not find:

  - HSET inside MULTI/EXEC (EXEC replays on the slow path);
  - HSET from a script;
  - HSET pipelined behind a slow-path command in the same write;
  - HMSET, and HSETNX, always.

Each document below is written one of those ways (the plain HSET is the
control), the index is built, and each document's own vector must find it
first. HGET must return the stored vector too (gh #360).

    python3 tests/test_vector_ingest_paths.py [--port 6500]
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import signal
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, encode, wait_ready_pid, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.abspath(os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
DIM = 16
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def vec(seed: int) -> bytes:
    rnd = random.Random(seed)
    v = [rnd.gauss(0, 1) for _ in range(DIM)]
    n = sum(x * x for x in v) ** 0.5
    return struct.pack(f"<{DIM}f", *[x / n for x in v])


def search(c: Conn, q: bytes) -> list:
    r = c.cmd("FT.SEARCH", "idx", "*=>[KNN 1 @vec $B]", "PARAMS", "2", "B", q, "DIALECT", "2")
    return [r[k] for k in range(1, len(r), 2)] if isinstance(r, list) else [r]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6500)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion_vecpaths_")
    proc = subprocess.Popen([BIN, "-p", str(a.port), "-w", "1", "--no-crash-log", "--no-auto-detect",
                             "--no-auto-embed"], cwd=work, stdout=open(os.path.join(work, "log"), "a"),
                            stderr=subprocess.STDOUT)
    try:
        wait_ready_pid(a.port, proc, 60)
        c = Conn(a.port, timeout=60)
        c.cmd("FT.CREATE", "idx", "SCHEMA", "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM),
              "DISTANCE_METRIC", "L2")
        # background documents, written the plain way
        for k in range(200):
            c.cmd("HSET", f"bg:{k}", "vec", vec(1000 + k))
        docs = {}
        docs["plain"] = vec(1)
        c.cmd("HSET", "plain", "vec", docs["plain"], "title", "x")
        docs["multi"] = vec(2)
        r = c.pipeline([("MULTI",), ("HSET", "multi", "vec", docs["multi"], "title", "x"), ("EXEC",)])
        check("HSET inside MULTI/EXEC runs", r[-1] == [2], repr(r))
        docs["script"] = vec(3)
        r = c.cmd("EVAL", "return redis.call('HSET', KEYS[1], 'vec', ARGV[1], 'title', 'x')", "1", "script",
                  docs["script"])
        check("HSET from a script runs", r == 2, repr(r))
        docs["behind"] = vec(4)
        c.sock.sendall(encode(("TIME",)) + encode(("HSET", "behind", "vec", docs["behind"], "title", "x")))
        c.read()
        check("HSET pipelined behind a slow command runs", c.read() == 2)
        docs["hmset"] = vec(5)
        check("HMSET runs", c.cmd("HMSET", "hmset", "vec", docs["hmset"], "title", "x") == "OK")
        docs["hsetnx"] = vec(6)
        check("HSETNX runs", c.cmd("HSETNX", "hsetnx", "vec", docs["hsetnx"]) == 1)
        docs["hsetnx2"] = vec(7)
        c.cmd("HSET", "hsetnx2", "title", "x")
        check("HSETNX on an existing hash runs", c.cmd("HSETNX", "hsetnx2", "vec", docs["hsetnx2"]) == 1)
        c.cmd("FT.OPTIMIZE", "idx")
        for name, v in docs.items():
            got = search(c, v)
            check(f"{name}: its own vector finds it first", got[:1] == [name.encode()], repr(got))
            check(f"{name}: HGET returns the stored vector", c.cmd("HGET", name, "vec") == v)
        c.close()
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        wait_port_free(a.port)
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
