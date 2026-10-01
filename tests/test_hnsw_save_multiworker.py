#!/usr/bin/env python3
"""Regression gate for gh #8: HNSW persistence must work with -w > 1.

Pre-fix: `vector.mojo` only called `save_to_disk` when `worker_id == 0`.
Pion's accept-race delivers FT.OPTIMIZE to a random worker, so on `-w N`
the save fired with probability 1/N. With -w 16 the file was missing
~94% of the time; warm-restart skip-FT.OPTIMIZE was effectively dead.

Post-fix: whichever worker handled FT.OPTIMIZE persists `pion.hnsw.0`.
This test launches `-w 4` (so 3/4 of the runs would fail pre-fix) and
asserts the file appears, is non-empty, and that a warm restart loads it.
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

DEFAULT_PORT = 6397
DIM = 1536
N = 1000
INDEX = "shidx"
PIPE_BATCH = 200
WORKERS = 4
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HNSW_FILE = os.path.join(ROOT, "pion.hnsw.0")


def make_dataset(seed: int = 0xBEEF) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((N, DIM)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


def cleanup_state() -> None:
    for name in os.listdir(ROOT):
        if name.startswith(("pion.hnsw.", "pion.wal.", "pion.snapshot.")):
            try:
                os.remove(os.path.join(ROOT, name))
            except OSError:
                pass


def start_pion(port: int) -> "subprocess.Popen":
    binary = os.environ.get("PION_BIN") or os.path.join(ROOT, "pion-server")
    cmd = [binary, "-p", str(port), "-w", str(WORKERS),
           "--no-auto-detect", "--no-auto-embed", "--independent-workers"]
    log_path = f"/tmp/pion_gh8_{port}.log"
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_fp, stderr=log_fp,
                            preexec_fn=os.setsid)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = redis.Redis(port=port, socket_connect_timeout=1)
            r.ping()
            r.close()
            return proc
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"pion-server did not come up on port {port}")


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


def ingest_and_optimize(port: int, vecs: np.ndarray) -> None:
    r = redis.Redis(port=port, decode_responses=False)
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
        pipe.hset(f"doc:{i}",
                  mapping={"id": str(i), "vector": v.tobytes()})
        if i % PIPE_BATCH == PIPE_BATCH - 1:
            pipe.execute()
    pipe.execute()
    r.execute_command("FT.OPTIMIZE", INDEX)
    r.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--keep-state", action="store_true",
                   help="Don't delete pion.hnsw.* on exit (for inspection).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cleanup_state()
    vecs = make_dataset()

    # ── Phase 1: prove FT.OPTIMIZE on -w 4 reliably writes pion.hnsw.0 ──────
    proc = start_pion(args.port)
    try:
        ingest_and_optimize(args.port, vecs)
    finally:
        stop_pion(proc)

    if not os.path.exists(HNSW_FILE):
        print(f"FAIL: {HNSW_FILE} missing after FT.OPTIMIZE on -w {WORKERS} "
              "(this is the gh #8 regression)")
        return 1
    size_mb = os.path.getsize(HNSW_FILE) / (1024 * 1024)
    if size_mb < 1.0:
        print(f"FAIL: {HNSW_FILE} only {size_mb:.2f} MB — body looks empty")
        return 1
    print(f"PASS phase 1: {HNSW_FILE} = {size_mb:.1f} MB after -w {WORKERS} FT.OPTIMIZE")

    # ── Phase 2: warm restart loads it (worker 0 picks up the snapshot) ─────
    log_path = f"/tmp/pion_gh8_warm_{args.port}.log"
    proc = subprocess.Popen(
        [os.environ.get("PION_BIN") or os.path.join(ROOT, "pion-server"), "-p", str(args.port),
         "-w", str(WORKERS), "--no-auto-detect", "--no-auto-embed",
         "--independent-workers"],
        cwd=ROOT, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                r = redis.Redis(port=args.port, socket_connect_timeout=1)
                r.ping()
                r.close()
                break
            except Exception:
                time.sleep(0.3)
        else:
            print("FAIL: pion-server did not come up after warm restart")
            return 1
    finally:
        stop_pion(proc)

    log_text = open(log_path).read()
    if "HNSW loaded from disk" not in log_text:
        print("FAIL: warm restart did not log 'HNSW loaded from disk'")
        print("--- last 30 log lines ---")
        print("\n".join(log_text.splitlines()[-30:]))
        return 1
    print("PASS phase 2: warm restart loaded pion.hnsw.0")

    if not args.keep_state:
        cleanup_state()
    print("\nALL CHECKS PASSED — gh #8 fix verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
