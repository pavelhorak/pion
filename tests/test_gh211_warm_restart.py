#!/usr/bin/env python3
"""Regression gate for gh #211: warm-restarted HNSW must serve the SAME keys.

Pre-fix: the slot→key reverse map lived only in process memory (shared
hk_keys_buf) and in un-WAL-logged __hk__ keyspace entries, so after a
restart every key-resolution tier missed and FT.SEARCH fell back to raw
slot numbers — the server logged "HNSW loaded from disk: N nodes" and then
silently served recall ≈ 0.002. Also: only the loading worker could serve
at all (the loaded graph was never published to the shared view), and the
per-group INT8 calibration was not persisted (queries re-quantized with the
global scale against per-group codes).

Post-fix (HNSW file v3): the slot→key map + group calibration persist with
the index, load scatters them back, and the loading worker publishes to the
shared view.

This test ingests docs in SHUFFLED key order (so slot numbers != doc ids —
the pre-fix slot-number fallback is guaranteed to return wrong keys), runs
identical KNN queries before and after a SIGTERM restart, and requires the
result sets to match.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

try:
    import redis
except ImportError:
    print("redis-py not installed; pip install 'redis<5.0'")
    sys.exit(2)

import numpy as np

DEFAULT_PORT = 6411
# DIM must match the server's startup default (1536): load_from_disk refuses a
# config-mismatched file, so warm restore only exists for the built-in dim.
DIM = 1536
N = 2000
K = 10
N_QUERIES = 50
INDEX = "wridx"
PIPE_BATCH = 200
# 2,000 random unit vectors at EF_RUNTIME 100 is an easy set: every mode,
# INT2 included, should clear this. It exists to catch a search running the
# wrong kernel over the compact bytes (recall ≈ 0), not to rank quantizers.
MIN_RECALL = {"int8": 0.80, "polarquant": 0.80, "turboquant": 0.80,
              # INT2 navigates coarsely on isotropic random data: measured
              # 2026-09-25 at 0.61 / 0.73 / 0.85 / 0.94 / 0.99 for EF_RUNTIME
              # 50 / 100 / 200 / 400 / 800 — monotone, so not a kernel fault.
              "nanoquant": 0.60}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cleanup_state() -> None:
    for name in os.listdir(ROOT):
        if name.startswith(("pion.hnsw.", "pion.wal.", "pion.snapshot.", "pion.blob.")):
            try:
                os.remove(os.path.join(ROOT, name))
            except OSError:
                pass


QUANT_FLAGS = {"int8": [], "polarquant": ["--polarquant"],
               "turboquant": ["--turboquant"], "nanoquant": ["--nanoquant"]}
# What FT.OPTIMIZE must print when the quantized compaction really ran. Without
# this check a quant server that silently built INT8 passes every other assert
# here — which is exactly how gh #350 shipped.
QUANT_BUILD_MARKER = {"polarquant": "[PolarQuant]", "turboquant": "[TurboQuant] Block-INT3",
                      "nanoquant": "[NanoQuant] Block-INT2"}


def start_pion(port: int, binary: str, workers: int, log_path: str,
               quant: str = "int8") -> "subprocess.Popen":
    cmd = [os.path.join(ROOT, binary), "-p", str(port), "-w", str(workers),
           "--no-auto-detect", "--no-auto-embed"] + QUANT_FLAGS[quant]
    # gh #253 fences -w N>1 behind an explicit acknowledgement that the N
    # keyspaces are independent. This test WANTS multiple workers — it spreads
    # connections across them to prove a warm-loaded index is published to the
    # shared view — so it is one of the in-repo callers that must pass the flag.
    # Without it the server exits 1 at startup and this gate silently stopped
    # being runnable at all.
    if workers > 1:
        cmd.append("--independent-workers")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = redis.Redis(port=port, socket_connect_timeout=1)
            r.ping()
            r.close()
            return proc
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"{binary} did not come up on port {port}")


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


def knn_keys(r, qvec: np.ndarray) -> list:
    """Run one KNN query, return the ordered list of doc keys."""
    resp = r.execute_command(
        "FT.SEARCH", INDEX,
        f"*=>[KNN {K} @vector $vec EF_RUNTIME 100 as score]",
        "PARAMS", "2", "vec", qvec.tobytes(),
        "SORTBY", "score", "LIMIT", "0", str(K), "DIALECT", "2",
    )
    # RESP shape: [count, key1, [fields...], key2, [fields...], ...]
    keys = []
    i = 1
    while i < len(resp):
        keys.append(resp[i].decode() if isinstance(resp[i], bytes) else str(resp[i]))
        i += 2
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", "pion-server"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-state", action="store_true")
    ap.add_argument("--quant", choices=sorted(QUANT_FLAGS), default="int8",
                    help="quantization mode to build AND warm-load with (gh #350)")
    args = ap.parse_args()

    cleanup_state()
    rng = np.random.default_rng(0x211)
    vecs = rng.standard_normal((N, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    queries = vecs[rng.choice(N, N_QUERIES, replace=False)] + \
        0.01 * rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    queries = queries.astype(np.float32)

    # ── Phase 1: fresh build (shuffled ingest order → slot != doc id) ───────
    fresh_log = f"/tmp/pion_gh211_fresh_{args.port}.log"
    proc = start_pion(args.port, args.binary, args.workers, fresh_log, args.quant)
    try:
        r = redis.Redis(port=args.port, decode_responses=False)
        r.execute_command(
            "FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", "doc:",
            "SCHEMA", "vector", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "COSINE",
        )
        order = rng.permutation(N)
        pipe = r.pipeline(transaction=False)
        for j, i in enumerate(order):
            pipe.hset(f"doc:{i}", mapping={"id": str(i), "vector": vecs[i].tobytes()})
            if j % PIPE_BATCH == PIPE_BATCH - 1:
                pipe.execute()
        pipe.execute()
        r.execute_command("FT.OPTIMIZE", INDEX)
        fresh = [knn_keys(r, q) for q in queries]
        r.close()
    finally:
        stop_pion(proc)

    non_doc = sum(1 for res in fresh for k in res if not k.startswith("doc:"))
    if non_doc:
        print(f"FAIL: fresh server returned {non_doc} non-doc keys (slot fallback?)")
        return 1
    # Exact top-K by brute force: overlap alone would pass a warm server that
    # reproduces a fresh build's wrong answers.
    truth = [{f"doc:{j}" for j in np.argsort(-(vecs @ q))[:K]} for q in queries]

    def recall(results) -> float:
        return sum(len(set(r) & t) for r, t in zip(results, truth)) / (K * len(truth))

    fresh_recall = recall(fresh)
    print(f"fresh recall@{K}: {fresh_recall:.3f}")
    if fresh_recall < MIN_RECALL[args.quant]:
        print(f"FAIL: fresh {args.quant} recall {fresh_recall:.3f} < {MIN_RECALL[args.quant]}")
        return 1
    marker = QUANT_BUILD_MARKER.get(args.quant)
    if marker and marker not in open(fresh_log).read():
        print(f"FAIL: --{args.quant} FT.OPTIMIZE never logged '{marker}' — the "
              "quantized compaction did not run (gh #350: fell back to INT8)")
        return 1
    print(f"PASS phase 1: fresh {args.quant} build, {N_QUERIES} queries, all keys doc-shaped")

    # ── Phase 2: warm restart must serve the same keys ──────────────────────
    log_path = f"/tmp/pion_gh211_warm_{args.port}.log"
    proc = start_pion(args.port, args.binary, args.workers, log_path, args.quant)
    try:
        # A worker != 0 must also answer (shared-view publish): open several
        # connections so the accept race spreads across workers.
        conns = [redis.Redis(port=args.port, decode_responses=False)
                 for _ in range(max(4, args.workers))]
        warm = [knn_keys(conns[qi % len(conns)], q)
                for qi, q in enumerate(queries)]
        for c in conns:
            c.close()
    finally:
        stop_pion(proc)

    log_text = open(log_path).read()
    if "HNSW loaded from disk" not in log_text:
        print("FAIL: warm restart did not log 'HNSW loaded from disk'")
        print("\n".join(log_text.splitlines()[-30:]))
        return 1

    non_doc = sum(1 for res in warm for k in res if not k.startswith("doc:"))
    if non_doc:
        print(f"FAIL: warm server returned {non_doc} non-doc keys "
              "(gh #211 slot-number fallback)")
        return 1

    overlaps = []
    for f_res, w_res in zip(fresh, warm):
        inter = len(set(f_res) & set(w_res))
        overlaps.append(inter / max(len(f_res), 1))
    mean_overlap = sum(overlaps) / len(overlaps)
    print(f"warm/fresh key overlap: mean={mean_overlap:.3f} min={min(overlaps):.3f}")
    warm_recall = recall(warm)
    print(f"warm recall@{K}: {warm_recall:.3f}")
    if warm_recall < fresh_recall - 0.02:
        print(f"FAIL: warm recall {warm_recall:.3f} below fresh {fresh_recall:.3f}")
        return 1
    if mean_overlap < 0.90:
        print(f"FAIL: warm-restart results diverge from fresh (mean overlap "
              f"{mean_overlap:.3f} < 0.90) — key map or calibration lost")
        return 1
    print("PASS phase 2: warm restart serves the same keys")

    if not args.keep_state:
        cleanup_state()
    print("\nALL CHECKS PASSED — gh #211 fix verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
