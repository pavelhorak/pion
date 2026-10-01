#!/usr/bin/env python3
"""Multi-worker FT.SEARCH recall regression gate.

Pins the bug fixed in commit ff51823 (cross-worker slot→hash-key map):

  All bench HSETs pipelined on one Redis connection land on a single worker.
  Pre-fix, only that worker had the __hk__<slot>→hash_key entries in its
  local keyspace; FT.SEARCH on a different worker missed every lookup and
  fell back to writing the slot number, which doesn't match the bench's
  hash key — recall collapsed to ~0.0019 (~25-50% of multi-worker runs).

Failure signature pre-fix:
  - non-empty results, but all 'id' fields are slot integers (0..N-1) that
    don't match the bench's hash keys (doc:N format).
  - the query (doc:17) — though stored — never appears in results because
    its hash key isn't recoverable from the cross-worker view.

Failure signature post-fix:
  - non-empty results contain the bench's actual hash keys, including doc:17.

gh #357 — the "empty" probes were never empty. This test used to parse the
`id` FIELD and call a probe that yielded none "EMPTY (borrow-race flake)".
The server was answering 10 results every time; on a worker that had not
ingested the docs it substituted the KEY for the missing field (id="doc:17"
where the stored value is "17"), `int()` rejected it, and the probe was
waved through as noise in ~1/3 of runs. Serially opened connections all land
on one worker, so a run was all-OK (probes on the ingest worker) or
all-"EMPTY" (probes elsewhere) — which read as a per-server flake.

Now: correctness is judged from the result KEYS (resolved cross-worker via
the shared slot→key map); an `id` field, when present, must equal the doc's
stored id (a substituted value is BAD); an empty result is BAD; and probes
open CONCURRENTLY so they spread across workers.

Requires: pion-server -w 4 --independent-workers (or higher) running on PORT.
"""
from __future__ import annotations

import argparse
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from typing import List

try:
    import redis
except ImportError:
    print("redis-py not installed; pip install 'redis<5.0'")
    sys.exit(2)

import numpy as np

DEFAULT_PORT = 6396
# 1536 matches Pion's default config — avoids a latent bug where FT.CREATE's DIM
# override is only set on the receiving worker, so FT.OPTIMIZE on any other
# worker uses the wrong dim. Using the default sidesteps it; this test is about
# cross-worker hash-key resolution, not dim-routing.
DIM = 1536
N = 2000         # vectors — large enough for HNSW to build a robust graph
NUM_PROBES = 32  # concurrent connections — enough to land on every worker
INDEX = "midx"
PIPE_BATCH = 200


def make_dataset(seed: int = 0xC0DE) -> np.ndarray:
    """N×DIM unit-norm vectors. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((N, DIM)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


def insert_all_on_one_connection(host: str, port: int, vecs: np.ndarray) -> None:
    """Pipeline FT.CREATE + N HSETs + FT.OPTIMIZE on a single Redis connection.

    This mirrors VectorDBBench's load phase: all writes share one connection,
    so they all hit one Pion worker. That single-worker concentration is the
    failure mode the cross-worker hk_keys_buf fix addresses.
    """
    r = redis.Redis(host=host, port=port, decode_responses=False)
    # Drop any prior incarnation; ignore "Unknown index" errors.
    try:
        r.execute_command("FT.DROPINDEX", INDEX)
    except redis.ResponseError:
        pass
    # NOTE: no DIM override — uses Pion's default (1536). Setting DIM on
    # FT.CREATE only affects the worker that receives it; other workers' local
    # hnsw.dim stays default and FT.OPTIMIZE/HSET routing breaks. Mirrors how
    # VectorDBBench operates (always default-dim).
    r.execute_command(
        "FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", "doc:",
        "SCHEMA", "vector", "VECTOR", "HNSW", "6",
        "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "COSINE",
    )
    pipe = r.pipeline(transaction=False)
    for i, v in enumerate(vecs):
        pipe.hset(
            f"doc:{i}",
            mapping={
                "id": str(i),
                "vector": v.astype(np.float32).tobytes(),
            },
        )
        if i % PIPE_BATCH == PIPE_BATCH - 1:
            pipe.execute()
    pipe.execute()
    r.execute_command("FT.OPTIMIZE", INDEX)
    # Brief settle so ready_atomic propagates and downstream connections can
    # ACQUIRE-load it. Without this, race-accepted fresh connections sometimes
    # see ready_atomic=0 (FT.OPTIMIZE has returned +OK but the cache line
    # hasn't reached the other CPUs' L1 yet on weak-memory ARM).
    time.sleep(0.3)
    r.close()


def search_on_connection(r, query: np.ndarray, k: int):
    """Issue one FT.SEARCH on an open connection.

    Returns (ids, fabricated): ids parsed from the result KEYS (b"doc:N" -> N),
    and a list of (key, id-field) pairs whose id field disagrees with the key's
    doc number — a substituted value, the gh #357 bug."""
    qbytes = query.astype(np.float32).tobytes()
    res = r.execute_command(
        "FT.SEARCH", INDEX, f"*=>[KNN {k} @vector $vec EF_RUNTIME 64 AS score]",
        "PARAMS", "2", "vec", qbytes,
        "RETURN", "1", "id",
        "DIALECT", "2",
    )
    # RESP2 layout: [count, key1, [field, val, ...], key2, [...], ...]
    ids: List[int] = []
    fabricated = []
    if not isinstance(res, list) or len(res) < 1:
        return ids, fabricated
    for i in range(1, len(res) - 1, 2):
        key, fields = res[i], res[i + 1]
        if not (isinstance(key, (bytes, bytearray)) and key.startswith(b"doc:")):
            fabricated.append((key, "key is not a doc: hash key"))
            continue
        n = int(key[4:])
        ids.append(n)
        named = dict(zip(fields[0::2], fields[1::2])) if isinstance(fields, list) else {}
        if b"id" in named and named[b"id"] != str(n).encode():
            fabricated.append((key, named[b"id"]))
    return ids, fabricated


def ground_truth_nearest(vecs: np.ndarray, query: np.ndarray, k: int) -> List[int]:
    # cosine sim — vectors are unit-norm so dot product is cosine.
    sims = vecs @ query
    order = np.argsort(-sims)[:k]
    return [int(i) for i in order]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--workers", type=int, default=4,
                   help="If --start, launch pion-server with this many workers.")
    p.add_argument("--start", action="store_true",
                   help="Launch pion-server for the duration of the test.")
    p.add_argument("--probes", type=int, default=NUM_PROBES,
                   help="Number of fresh search connections to verify.")
    p.add_argument("--retries", type=int, default=2,
                   help="Unused since gh #357 (there is no INCONCLUSIVE outcome any more); kept so existing invocations still parse.")
    p.add_argument("--lifetimes", type=int, default=3,
                   help="With --start: run this many fresh server lifetimes; EVERY one must pass. "
                        "gh #357 failed in ~1/3 of lifetimes, and the loop below used to break after one.")
    p.add_argument("--exact-match-required", action="store_true", default=True,
                   help="Require that the query (which is itself a stored vector) appears in top-3 results. Without the cross-worker map fix, the response carries slot numbers instead of hash keys, so the query's hash key can't appear at all.")
    p.add_argument("--min-overlap", type=int, default=3,
                   help="Min number of ground-truth top-10 ids each probe must return. Pre-fix this falls to ~0 because the response carries slot numbers, not hash keys.")
    return p.parse_args()


def start_pion(args) -> "subprocess.Popen":
    binary = os.environ.get("PION_BIN") or os.path.join(os.path.dirname(__file__), "..", "pion-server")
    binary = os.path.abspath(binary)
    cmd = [binary, "-p", str(args.port), "-w", str(args.workers),
           "--no-auto-detect", "--no-auto-embed"]
    if args.workers > 1:
        cmd.append("--independent-workers")   # gh #253
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for pat in ("pion.wal.", "pion.hnsw.", "pion.snapshot."):
        for w in range(args.workers):
            f = os.path.join(cwd, f"{pat}{w}")
            if os.path.exists(f):
                os.remove(f)
    log_path = "/tmp/pion_mw_test.log"
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=log_fp, stderr=log_fp,
        preexec_fn=os.setsid,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = redis.Redis(host=args.host, port=args.port, socket_connect_timeout=1)
            r.ping()
            r.close()
            return proc
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"pion-server did not come up on port {args.port}")


def stop_pion(proc) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def main() -> int:
    args = parse_args()

    vecs = make_dataset()
    query_idx = 17
    query = vecs[query_idx]
    gt = ground_truth_nearest(vecs, query, k=10)
    print(f"Ground truth top-10 for doc:{query_idx}: {gt[:5]}...")

    rc = 0
    proc = None
    attempt = 0
    max_attempts = args.lifetimes if args.start else 1
    try:
      while attempt < max_attempts:
        attempt += 1
        if args.start:
            stop_pion(proc); proc = None
            print(f"\n--- attempt {attempt}/{max_attempts}: starting fresh pion-server ---")
            proc = start_pion(args)

        insert_all_on_one_connection(args.host, args.port, vecs)
        print(f"Inserted {N} docs on one connection (mirrors bench load phase).")

        # Open every probe connection CONCURRENTLY, then query on each.
        # Serially opened connections all land on one worker (gh #253),
        # which is how gh #357 hid: a run's probes were either all
        # on the ingest worker or all elsewhere.
        conns = [None] * args.probes
        def _open(k):
            c = redis.Redis(host=args.host, port=args.port, decode_responses=False,
                            single_connection_client=True)
            c.connection_pool.get_connection("_")  # force the accept now
            conns[k] = c
        threads = [threading.Thread(target=_open, args=(k,)) for k in range(args.probes)]
        for t in threads: t.start()
        for t in threads: t.join()

        # MEASURE the spread instead of assuming it (gh #253): keyspaces are
        # per worker, so two connections share a worker iff each sees the
        # other's marker. One group means every probe hit the same worker and
        # an agreeing result proves nothing.
        for k, c in enumerate(conns):
            c.set(f"wmark:{attempt}:{k}", b"1")
        groups = set()
        for k, c in enumerate(conns):
            seen = frozenset(j for j in range(len(conns)) if c.get(f"wmark:{attempt}:{j}") == b"1")
            groups.add(seen)
        print(f"  probe connections landed on {len(groups)} distinct worker(s)")
        if args.start and args.workers > 1 and len(groups) < 2:
            print("FAIL: all probes landed on one worker — the lifetime cannot test cross-worker search")
            rc = 1

        good = 0
        bad: List[str] = []
        for probe, c in enumerate(conns):
            got, fabricated = search_on_connection(c, query, k=10)
            c.close()
            overlap = len(set(got) & set(gt))
            ok_exact = query_idx in got[:3]
            ok = bool(got) and ok_exact and overlap >= args.min_overlap and not fabricated
            print(f"  probe {probe + 1:2d}: {'OK ' if ok else 'BAD'}  n={len(got)} exact_in_top3={ok_exact} "
                  f"overlap10={overlap}/10  ids={got[:5]}{' FABRICATED id: ' + repr(fabricated[:2]) if fabricated else ''}")
            if ok:
                good += 1
            else:
                bad.append(f"probe {probe + 1}: n={len(got)} ids={got[:5]} fabricated={fabricated[:2]}")

        print(f"\nSummary: {good} OK, {len(bad)} BAD.")
        if bad:
            print(f"FAIL (lifetime {attempt}): a probe returned no results, wrong keys, or an id field that is not the doc's stored id (gh #357).")
            for b in bad[:5]:
                print(f"  {b}")
            rc = 1
        else:
            print(f"lifetime {attempt}: {good}/{args.probes} probes returned correct keys; no substituted id fields.")
        # No break: every lifetime must pass (an intermittent bug is caught
        # by repetition, not by one lucky run).
      if rc == 0:
          print(f"PASS: {max_attempts} lifetime(s), every probe correct")


    finally:
        stop_pion(proc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
