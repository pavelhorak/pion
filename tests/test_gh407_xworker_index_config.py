#!/usr/bin/env python3
"""gh #407 — FT.OPTIMIZE builds with FT.CREATE's config, on whichever worker runs it.

FT.CREATE wrote DIM, the vector field, DISTANCE_METRIC and EF_CONSTRUCTION into
the CREATING worker's graph, and the accept race usually hands FT.OPTIMIZE to a
different worker, which built with its own defaults:

  * `DIM 16` was built as 1536-d from the 16-d ingest buffer; every query then
    answered `ERR ... this index expects 6144 (DIM 1536 x FLOAT32)`;
  * `DISTANCE_METRIC COSINE` was built as L2, and publish_to_shared wrote that
    back over the shared metric: queries went un-normalized against a
    normalized graph (an exact match scored ~1493 instead of ~0);
  * `EF_CONSTRUCTION` was ignored.

And, independently of the race, a warm-loaded index forgot its vector field, so
`@<field>` was refused on every worker for any field not named `vector`.

The builder is chosen deterministically through the affinity ports
(`port + 2 + worker_id`): FT.CREATE on worker 0, ingest on worker 2, FT.OPTIMIZE
on worker 1. The saved index header is read back to pin what was built.

Usage:
    python3 tests/test_gh407_xworker_index_config.py [--binary ./pion-server] [--port 7407]
"""

import argparse
import os
import shutil
import struct
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gh403_ft_info_name import (  # noqa: E402
    Client, serve, stop, wait_ready, worker_of,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = 4
PASS, FAIL = [], []


def check(name, ok, detail: object = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def per_worker(port, fn):
    out = []
    for w in range(WORKERS):
        c = Client(port + 2 + w)
        try:
            out.append(fn(c))
        finally:
            c.close()
    return out


def knn(c, field, blob, k=3):
    return c.cmd("FT.SEARCH", "idx", f"*=>[KNN {k} @{field} $q]", "PARAMS", "2", "q", blob)


def top(r):
    """(first key, its score) of an FT.SEARCH reply, or the reply itself."""
    if not isinstance(r, list) or len(r) < 3:
        return r
    fields = r[2]
    kv = {fields[i]: fields[i + 1] for i in range(0, len(fields) - 1, 2)}
    return r[1], float(kv.get(b"score", b"nan"))


def header(path):
    with open(path, "rb") as f:
        h = f.read(256)
    w = struct.unpack("<32Q", h)
    vf_len = w[26]
    return {"dim": w[3], "efc": w[5], "metric": w[24],
            "field": h[216:216 + vf_len] if vf_len <= 32 else b"?"}


def build_cross_worker(port, dim, field, metric, efc, vecs):
    c0 = Client(port + 2)
    r = c0.cmd("FT.CREATE", "idx", "ON", "HASH", "PREFIX", "1", "doc:", "SCHEMA",
               field, "VECTOR", "HNSW", "8", "TYPE", "FLOAT32", "DIM", str(dim),
               "DISTANCE_METRIC", metric, "EF_CONSTRUCTION", str(efc))
    c0.close()
    c2 = Client(port + 4)
    buf = b"".join(Client.encode("HSET", f"doc:{i}", field, v.tobytes()) for i, v in enumerate(vecs))
    c2.sock.sendall(buf)
    acks = [c2.read() for _ in range(len(vecs))]
    c2.close()
    c1 = Client(port + 3)
    o = c1.cmd("FT.OPTIMIZE", "idx")
    c1.close()
    return r, acks, o


def main():
    workdir = tempfile.mkdtemp(prefix="pion-gh407-")
    try:
        return run(workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)   # 256 MB of WAL per worker


def run(workdir):
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7407)
    args = ap.parse_args()
    args.binary = os.path.abspath(args.binary)  # the server runs in a temp cwd
    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}")
        return 2
    print(f"gh #407 cross-worker FT config — binary={args.binary} dir={workdir}")
    rng = np.random.default_rng(407)

    # ── [1] DIM 16, COSINE, EF_CONSTRUCTION 64, built on a non-creating worker ─
    print("\n[1] FT.CREATE on worker 0, FT.OPTIMIZE on worker 1: DIM 16 COSINE efc 64")
    port = args.port
    d1 = os.path.join(workdir, "dim16")
    os.makedirs(d1)
    proc = serve(args.binary, port, WORKERS, d1, "server.log")
    try:
        if not check("server ready", wait_ready(port)):
            return 1
        ids = per_worker(port, worker_of)
        if not check("affinity port port+2+w reaches worker w", ids == list(range(WORKERS)), ids):
            return 1
        # Slabs and query scratch are sized for the startup --dim (1536) and
        # never grow: a larger DIM overwrote the next node's vector on the
        # builder. Refused up front, touching nothing.
        c0 = Client(port + 2)
        r = c0.cmd("FT.CREATE", "wide", "ON", "HASH", "SCHEMA", "vec", "VECTOR", "HNSW", "6",
                   "TYPE", "FLOAT32", "DIM", "3072", "DISTANCE_METRIC", "L2")
        check("FT.CREATE DIM 3072 on a --dim 1536 server is refused",
              isinstance(r, Exception) and "exceeds" in str(r), repr(r)[:100])
        check("... and registered nothing", isinstance(c0.cmd("FT.INFO", "wide"), Exception))
        c0.close()
        vecs = (3.0 * rng.standard_normal((300, 16))).astype(np.float32)   # NOT unit norm
        r, acks, o = build_cross_worker(port, 16, "vec", "COSINE", 64, vecs)
        check("FT.CREATE / ingest / FT.OPTIMIZE", r == b"OK" and o == b"OK"
              and all(a in (1, 2) for a in acks), f"{r!r} {o!r}")
        res = per_worker(port, lambda c: top(knn(c, "vec", vecs[11].tobytes())))
        check("DIM 16 queries are accepted on every worker and find doc:11 first "
              "(was: 'expects 6144 (DIM 1536)')",
              all(isinstance(x, tuple) and x[0] == b"doc:11" for x in res), res)
        check("scores are cosine distances: exact match ~0 on every worker "
              "(was raw L2 of an un-normalized query)",
              all(isinstance(x, tuple) and x[1] < 0.01 for x in res), res)
        # A scaled copy must be the same direction: same answer, same ~0 score.
        res = per_worker(port, lambda c: top(knn(c, "vec", (7.5 * vecs[23]).tobytes())))
        check("a scaled query finds the same direction (COSINE really normalizes)",
              all(isinstance(x, tuple) and x[0] == b"doc:23" and x[1] < 0.01 for x in res), res)
        h = header(os.path.join(d1, "pion.hnsw.0"))
        check("saved index: dim 16, metric COSINE, EF_CONSTRUCTION 64, field 'vec'",
              h == {"dim": 16, "efc": 64, "metric": 1, "field": b"vec"}, h)
    finally:
        stop(proc)

    # ── [2] 1536-d, field 'emb', cross-worker build, then a warm restart ────────
    print("\n[2] field 'emb', cross-worker build, warm restart")
    port = args.port + 10
    d2 = os.path.join(workdir, "warm")
    os.makedirs(d2)
    vecs = (2.0 * rng.standard_normal((300, 1536))).astype(np.float32)
    proc = serve(args.binary, port, WORKERS, d2, "server.log")
    try:
        if not check("server ready", wait_ready(port)):
            return 1
        r, acks, o = build_cross_worker(port, 1536, "emb", "COSINE", 100, vecs)
        check("FT.CREATE / ingest / FT.OPTIMIZE", r == b"OK" and o == b"OK", f"{r!r} {o!r}")
        res = per_worker(port, lambda c: top(knn(c, "emb", vecs[5].tobytes())))
        check("live: @emb finds doc:5 first on every worker, cosine score",
              all(isinstance(x, tuple) and x[0] == b"doc:5" and x[1] < 0.01 for x in res), res)
        h = header(os.path.join(d2, "pion.hnsw.0"))
        check("saved index carries field 'emb' and metric COSINE",
              h["field"] == b"emb" and h["metric"] == 1, h)
    finally:
        stop(proc)
    proc = serve(args.binary, port, WORKERS, d2, "server-warm.log")
    try:
        if not check("warm server ready", wait_ready(port)):
            return 1
        deadline = time.time() + 30
        res = None
        while time.time() < deadline:
            res = per_worker(port, lambda c: top(knn(c, "emb", vecs[42].tobytes())))
            if all(isinstance(x, tuple) for x in res):
                break
            time.sleep(0.5)
        check("warm: @emb finds doc:42 first on every worker "
              "(was: \"KNN field '@emb' is not this index's vector field\")",
              res is not None and all(isinstance(x, tuple) and x[0] == b"doc:42" for x in res), res)
        check("warm: cosine scores survive the restart",
              res is not None and all(isinstance(x, tuple) and x[1] < 0.01 for x in res), res)
        res = per_worker(port, lambda c: knn(c, "vector", vecs[42].tobytes()))
        check("warm: the default field name is now refused (the index is @emb)",
              all(isinstance(x, Exception) for x in res), [repr(x)[:60] for x in res])
    finally:
        stop(proc)

    print(f"\n{'=' * 60}")
    print(f"PASS {len(PASS)}  FAIL {len(FAIL)}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
