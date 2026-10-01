#!/usr/bin/env python3
"""D11: every vector build must answer FT.SEARCH identically on the same index.

Builds one HNSW index with the first binary, then warm-loads that exact
index file into each binary in turn (`pixi run build`, which links
libpion_vector, and `pixi run build-open`, the open reference) and requires
identical ordered keys AND identical score strings for
every query. HNSW construction is nondeterministic, so comparing two fresh
builds would compare two graphs; loading one saved graph isolates the search
routine, which is the only thing that differs between the builds.

    python3 tests/test_vector_build_equivalence.py \
        --binaries pion-server-lib pion-server-open [--server-arg=--turboquant]

The whole-server counterpart of tests/test_vector_differential.mojo: that one
feeds each closed routine synthetic state, this one runs the real index.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import redis

DIM = 1536   # warm restore exists only for the server's startup dim
INDEX = "eqidx"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATE = ("pion.hnsw.", "pion.wal.", "pion.snapshot.", "pion.blob.")


EXTRA_ARGS: list = []


def start(binary: str, port: int, workdir: str, log: str):
    cmd = [os.path.join(ROOT, binary), "-p", str(port), "-w", "1",
           "--no-auto-detect", "--no-auto-embed", "--no-crash-log"] + EXTRA_ARGS
    proc = subprocess.Popen(cmd, cwd=workdir, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            r = redis.Redis(port=port, socket_connect_timeout=1)
            r.ping()
            r.close()
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(f"{binary} exited: see {log}")
            time.sleep(0.3)
    raise RuntimeError(f"{binary} did not come up")


def stop(proc) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def knn(r, q: np.ndarray, k: int, ef: int) -> list:
    resp = r.execute_command(
        "FT.SEARCH", INDEX, f"*=>[KNN {k} @vector $vec EF_RUNTIME {ef} as score]",
        "PARAMS", "2", "vec", q.tobytes(), "SORTBY", "score",
        "LIMIT", "0", str(k), "DIALECT", "2")
    out = []
    for i in range(1, len(resp), 2):
        fields = resp[i + 1] if i + 1 < len(resp) else []
        score = None
        for j in range(0, len(fields) - 1, 2):
            if fields[j] == b"score":
                score = fields[j + 1]
        out.append((resp[i], score))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binaries", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--ef", type=int, default=150)
    ap.add_argument("--port", type=int, default=6412)
    ap.add_argument("--server-arg", action="append", default=[],
                    help="extra server flag, repeatable (e.g. --server-arg=--turboquant)")
    args = ap.parse_args()
    EXTRA_ARGS.extend(args.server_arg)

    rng = np.random.default_rng(0xD11)
    # Clustered data, so neighbourhoods are meaningful and pruning bites.
    centers = rng.standard_normal((64, DIM)).astype(np.float32)
    vecs = centers[rng.integers(0, 64, args.n)] + 0.35 * rng.standard_normal((args.n, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    queries = (vecs[rng.choice(args.n, args.queries, replace=False)]
               + 0.05 * rng.standard_normal((args.queries, DIM)).astype(np.float32)).astype(np.float32)

    work = tempfile.mkdtemp(prefix="pion_d11_eq_")
    saved = tempfile.mkdtemp(prefix="pion_d11_saved_")
    try:
        proc = start(args.binaries[0], args.port, work, os.path.join(work, "build.log"))
        try:
            r = redis.Redis(port=args.port)
            r.execute_command("FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", "doc:",
                              "SCHEMA", "vector", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32",
                              "DIM", str(DIM), "DISTANCE_METRIC", "L2")
            pipe = r.pipeline(transaction=False)
            for i in range(args.n):
                pipe.hset(f"doc:{i}", mapping={"vector": vecs[i].tobytes()})
                if i % 500 == 499:
                    pipe.execute()
            pipe.execute()
            r.execute_command("FT.OPTIMIZE", INDEX)
            r.close()
        finally:
            stop(proc)
        for name in os.listdir(work):
            if name.startswith(STATE):
                shutil.copy2(os.path.join(work, name), saved)

        results = {}
        for b in args.binaries:
            for name in os.listdir(work):
                if name.startswith(STATE):
                    os.remove(os.path.join(work, name))
            for name in os.listdir(saved):
                shutil.copy2(os.path.join(saved, name), work)
            log = os.path.join(work, f"warm_{os.path.basename(b)}.log")
            proc = start(b, args.port, work, log)
            try:
                r = redis.Redis(port=args.port)
                knn(r, queries[0], args.k, args.ef)  # warm the connection
                t0 = time.perf_counter()
                results[b] = [knn(r, q, args.k, args.ef) for q in queries]
                elapsed = time.perf_counter() - t0
                r.close()
            finally:
                stop(proc)
            if "HNSW loaded from disk" not in open(log).read():
                print(f"FAIL: {b} did not warm-load the saved index ({log})")
                return 1
            got = sum(len(x) for x in results[b])
            print(f"{b}: {got} results over {args.queries} queries, "
                  f"{elapsed * 1e3 / args.queries:.3f} ms/query (1 client, informational)")
            if got == 0:
                print(f"FAIL: {b} returned nothing")
                return 1

        base = args.binaries[0]
        bad = 0
        for b in args.binaries[1:]:
            diff = sum(1 for x, y in zip(results[base], results[b]) if x != y)
            print(f"{b} vs {base}: {diff}/{args.queries} queries differ")
            bad += diff
        if bad:
            return 1
        print("PASS: identical keys and scores across builds")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(saved, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
