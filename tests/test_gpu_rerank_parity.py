#!/usr/bin/env python3
"""GPU rerank dispatch + parity gate (Mac, --gpu).

Closes Task A from the Kernel 3 delivery:
proves the FP32 gather-rerank Metal kernel actually fires on FT.SEARCH and
that its top-K results are consistent with a CPU-only run on the same data.

The doc's verification gap was: "the default --gpu gate passes, but the
recall delta could conceivably come from CPU fallback + run-to-run noise."
This test removes that ambiguity by snapshotting the `gpu_rerank_count`
atomic counter in XGPU INFO around a query batch — if the counter increments,
the kernel ran. End of debate.

Two-stage check:

  Stage 1 — dispatch proof:
    Start pion with --gpu, ingest N vectors, FT.OPTIMIZE, then snapshot
    XGPU INFO → run M searches with k ≥ GPU_RERANK_THRESHOLD (64) → snapshot
    again. Assert gpu_rerank_count delta ≥ M.

  Stage 2 — CPU/GPU parity (best-effort):
    Restart on a fresh process *without* --gpu (CPU-only). Repeat the same
    queries and assert top-K id sets agree with the GPU run within a small
    tie-tolerance (FP32 reduction order on GPU produces a slightly different
    tiebreak on near-equal distances — that's the documented +0.6pp recall
    bump and is expected). Default tolerance: ≥80% top-10 overlap per query.

Requires: pion-server built on macOS with Metal toolchain installed
(see memory/mac_metal_toolchain_required.md). Skipped on Linux (no Metal).
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import signal
import subprocess
import sys
import time
from typing import Dict, List, Tuple

try:
    import redis
except ImportError:
    print("redis-py not installed; pip install 'redis<5.0'")
    sys.exit(2)

import numpy as np


DIM = 1536                # MUST match Pion's Metal context init dim; pion_metal_register_rerank_buffer
                          # rejects dim mismatch (rc=-2), gpu_rerank_registered stays False, _try_gpu_rerank
                          # bails at gate 2 → counter never increments. See metal_wrap.m:566.
N_DOCS = 500              # > GPU_RERANK_THRESHOLD (64), large enough for ef=100 to gather > 64 cands
N_QUERIES = 16            # at least this many rerank dispatches expected
KNN_K = 100               # ≥ GPU_RERANK_THRESHOLD = 64 → rerank fires
INDEX = "rerank_idx"
PIPE_BATCH = 100


def make_dataset(seed: int = 0xBEEF) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((N_DOCS, DIM)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


def make_queries(n: int, seed: int = 0xCAFE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    qs = rng.standard_normal((n, DIM)).astype(np.float32)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True) + 1e-9
    return qs


def parse_xgpu_info(blob: bytes) -> Dict[str, str]:
    """XGPU INFO returns a RESP bulk string with `key:value\\r\\n` lines."""
    out: Dict[str, str] = {}
    for line in blob.decode("utf-8", errors="replace").splitlines():
        if line.startswith("#") or not line.strip() or ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def xgpu_info(host: str, port: int) -> Dict[str, str]:
    r = redis.Redis(host=host, port=port, decode_responses=False)
    try:
        blob = r.execute_command("XGPU", "INFO")
    finally:
        r.close()
    if not isinstance(blob, (bytes, bytearray)):
        raise RuntimeError(f"XGPU INFO unexpected type: {type(blob)}: {blob!r}")
    return parse_xgpu_info(bytes(blob))


def ingest_and_optimize(host: str, port: int, vecs: np.ndarray) -> None:
    r = redis.Redis(host=host, port=port, decode_responses=False)
    try:
        try:
            r.execute_command("FT.DROPINDEX", INDEX)
        except redis.ResponseError:
            pass
        r.execute_command(
            "FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", "doc:",
            "SCHEMA", "vector", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "COSINE",
        )
        pipe = r.pipeline(transaction=False)
        for i, v in enumerate(vecs):
            pipe.hset(
                f"doc:{i}",
                mapping={"id": str(i), "vector": v.astype(np.float32).tobytes()},
            )
            if i % PIPE_BATCH == PIPE_BATCH - 1:
                pipe.execute()
        pipe.execute()
        r.execute_command("FT.OPTIMIZE", INDEX)
        time.sleep(0.3)
    finally:
        r.close()


def search_topk(host: str, port: int, query: np.ndarray, k: int) -> List[int]:
    r = redis.Redis(host=host, port=port, decode_responses=False)
    try:
        qbytes = query.astype(np.float32).tobytes()
        res = r.execute_command(
            "FT.SEARCH", INDEX, f"*=>[KNN {k} @vector $vec EF_RUNTIME 150 AS score]",
            "PARAMS", "2", "vec", qbytes,
            "RETURN", "1", "id",
            "DIALECT", "2",
        )
    finally:
        r.close()
    out: List[int] = []
    if not isinstance(res, list) or len(res) < 1:
        return out
    i = 1
    while i < len(res):
        i += 1  # skip key
        if i >= len(res):
            break
        fields = res[i]
        i += 1
        if not isinstance(fields, list):
            continue
        for j in range(0, len(fields) - 1, 2):
            if fields[j] == b"id":
                try:
                    out.append(int(fields[j + 1]))
                except (TypeError, ValueError):
                    pass
                break
    return out


def start_pion(args, gpu: bool, log_suffix: str) -> "subprocess.Popen":
    binary = os.environ.get("PION_BIN") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pion-server"))
    cmd = [binary, "-p", str(args.port), "-w", "1",
           "--no-auto-detect", "--no-auto-embed"]
    if gpu:
        cmd.append("--gpu")
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for pat in ("pion.wal.", "pion.hnsw.", "pion.snapshot."):
        for w in range(8):
            f = os.path.join(cwd, f"{pat}{w}")
            if os.path.exists(f):
                os.remove(f)
    log_path = f"/tmp/pion_rerank_{log_suffix}.log"
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=log_fp, stderr=log_fp, preexec_fn=os.setsid,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = redis.Redis(host=args.host, port=args.port, socket_connect_timeout=1)
            r.ping(); r.close()
            return proc
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"pion-server did not come up on port {args.port} (log: {log_path})")


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6398)
    p.add_argument("--start", action="store_true",
                   help="Launch pion-server for the duration of the test.")
    p.add_argument("--skip-parity", action="store_true",
                   help="Skip the CPU/GPU parity stage (only verify dispatch).")
    p.add_argument("--min-overlap", type=int, default=8,
                   help="Min top-10 id overlap between CPU and GPU runs per query (default 8/10).")
    return p.parse_args()


def run_query_batch(host: str, port: int, queries: np.ndarray) -> List[List[int]]:
    return [search_topk(host, port, q, KNN_K) for q in queries]


def main() -> int:
    if platform.system() != "Darwin":
        print("SKIP: GPU rerank kernel is macOS/Metal-only.")
        return 0

    args = parse_args()
    vecs = make_dataset()
    queries = make_queries(N_QUERIES)

    proc = None
    try:
        # ── Stage 1: dispatch proof under --gpu ──────────────────────────
        if args.start:
            proc = start_pion(args, gpu=True, log_suffix="gpu")

        info0 = xgpu_info(args.host, args.port)
        if info0.get("gpu_available") != "1":
            print(f"SKIP: gpu_available={info0.get('gpu_available')!r}; "
                  f"Metal toolchain not active. See memory/mac_metal_toolchain_required.md.")
            return 0

        before = int(info0.get("gpu_rerank_count", "0"))
        before_disp = int(info0.get("gpu_dispatch_count", "0"))

        ingest_and_optimize(args.host, args.port, vecs)
        gpu_results = run_query_batch(args.host, args.port, queries)

        info1 = xgpu_info(args.host, args.port)
        after = int(info1.get("gpu_rerank_count", "0"))
        after_disp = int(info1.get("gpu_dispatch_count", "0"))
        rerank_delta = after - before
        disp_delta = after_disp - before_disp

        print(f"Stage 1 (--gpu): gpu_rerank_count {before} → {after} (delta={rerank_delta})")
        print(f"                 gpu_dispatch_count {before_disp} → {after_disp} (delta={disp_delta})")
        print(f"                 ran {N_QUERIES} queries, k={KNN_K}, threshold=64")

        if rerank_delta < N_QUERIES:
            print(f"FAIL: expected gpu_rerank_count to grow by ≥ {N_QUERIES} (one dispatch per query); "
                  f"saw {rerank_delta}. Either rerank kernel is silently falling back to CPU "
                  f"(gpu_rerank_registered=False?) or the threshold gate kept firing. "
                  f"Check /tmp/pion_rerank_gpu.log.")
            return 1
        print(f"PASS: rerank kernel fired ≥ once per query.")

        # ── Stage 2: CPU/GPU top-K parity ────────────────────────────────
        if args.skip_parity:
            print("Stage 2 skipped (--skip-parity).")
            return 0
        if not args.start:
            print("Stage 2 skipped: needs --start to launch a CPU-only instance.")
            return 0

        stop_pion(proc); proc = None
        proc = start_pion(args, gpu=False, log_suffix="cpu")
        ingest_and_optimize(args.host, args.port, vecs)
        cpu_results = run_query_batch(args.host, args.port, queries)

        bad: List[Tuple[int, int]] = []
        for qi, (g, c) in enumerate(zip(gpu_results, cpu_results)):
            top10_g = set(g[:10])
            top10_c = set(c[:10])
            overlap = len(top10_g & top10_c)
            if overlap < args.min_overlap:
                bad.append((qi, overlap))

        if bad:
            print(f"FAIL: {len(bad)}/{N_QUERIES} queries had top-10 overlap < {args.min_overlap}/10.")
            for qi, ov in bad[:5]:
                print(f"  q{qi}: overlap={ov}/10  gpu_top5={gpu_results[qi][:5]}  cpu_top5={cpu_results[qi][:5]}")
            return 1

        print(f"Stage 2 (CPU vs GPU): all {N_QUERIES} queries top-10 overlap ≥ {args.min_overlap}/10.")
        print("PASS: GPU rerank produces same neighborhood as CPU rerank within tie-tolerance.")
        return 0

    finally:
        stop_pion(proc)


if __name__ == "__main__":
    sys.exit(main())
