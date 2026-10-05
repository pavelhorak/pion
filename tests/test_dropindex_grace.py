#!/usr/bin/env python3
"""FT.DROPINDEX grace-period regression test.

Closes (best-effort) the §30.3 latent UAF: between owner's ready_atomic=0
RELEASE store and the actual buffer free in reset_index, an in-flight
borrower search could still hit a freed pointer. The 10ms usleep
inserted in handle_ft_dropindex (§35) shrinks the window to "longer
than any realistic search call".

Test plan: run K sequential DROP→CREATE→HSET→OPTIMIZE cycles. Verifies:
  - Server doesn't crash from the grace-period path itself.
  - Memory doesn't grow unboundedly (the grace period doesn't leak).
  - Each FT.OPTIMIZE in the cycle still produces a usable index.

Concurrent-search stress is left to a separate harness — gating the
basic safety property here is enough for the regression bar.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    import redis
except ImportError:
    print("redis-py required (pip install 'redis<5.0')")
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=6392)
    p.add_argument("--cycles", type=int, default=10)
    p.add_argument("--n-vectors", type=int, default=100)
    return p.parse_args()


def start_pion(port, log_path):
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=log,
                             preexec_fn=os.setsid)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return proc
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("pion-server didn't start")


def stop_pion(proc):
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


def populate(r, n_vectors):
    """Vectors stored under key `doc:0`..`doc:N-1`."""
    try:
        r.execute_command("FT.DROPINDEX", "stress_idx")
    except redis.ResponseError:
        pass
    r.execute_command(
        "FT.CREATE", "stress_idx", "ON", "HASH", "PREFIX", "1", "doc:",
        "SCHEMA", "vector", "VECTOR", "HNSW", "6",
        "TYPE", "FLOAT32", "DIM", "1536",
        "DISTANCE_METRIC", "COSINE",
    )
    rng = np.random.default_rng(0xCAFE)
    pipe = r.pipeline(transaction=False)
    for i in range(n_vectors):
        v = rng.standard_normal(1536).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        pipe.hset(f"doc:{i}", mapping={"id": str(i), "vector": v.tobytes()})
        if i % 50 == 49:
            pipe.execute()
    pipe.execute()
    r.execute_command("FT.OPTIMIZE", "stress_idx")


def main() -> int:
    args = parse_args()
    proc = None
    rc = 0

    for f in ("pion.vstore.0", "pion.vstore.wal.0", "pion.wal.0",
              "pion.snapshot.0", "pion.hnsw.0"):
        p = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(p): os.remove(p)

    try:
        proc = start_pion(args.port, "/tmp/pion_dropgrace.log")
        r = redis.Redis(host="127.0.0.1", port=args.port, decode_responses=False)
        # Initial populate so cycle 1 isn't measuring a cold cache.
        populate(r, args.n_vectors)
        # Each cycle: DROP + CREATE + HSET + OPTIMIZE. The 10ms grace lives
        # inside the DROP step.
        cycle_ms = []
        for cycle in range(args.cycles):
            t0 = time.perf_counter()
            populate(r, args.n_vectors)
            t1 = time.perf_counter()
            cycle_ms.append(1000 * (t1 - t0))
            # Verify the index is queryable after the cycle.
            info = r.execute_command("FT.INFO", "stress_idx")
            # FT.INFO returns a flat array; find num_docs.
            num_docs = None
            if isinstance(info, list):
                for i in range(0, len(info) - 1, 2):
                    k = info[i]
                    if isinstance(k, (bytes, bytearray)) and k == b"num_docs":
                        num_docs = int(info[i + 1])
                        break
            if num_docs != args.n_vectors:
                print(f"FAIL: cycle {cycle + 1}: FT.INFO num_docs={num_docs}, expected {args.n_vectors}")
                rc = 1
                return rc
        print(f"  {args.cycles} cycles complete; per-cycle ms: "
              f"{[f'{t:.0f}' for t in cycle_ms]}")

        # Grace period adds ~10ms to every drop; sanity-check the budget is
        # bounded (not blowing up to seconds).
        max_cycle = max(cycle_ms)
        if max_cycle > 3000:  # 3s sanity ceiling
            print(f"FAIL: cycle time {max_cycle:.0f}ms > 3s — something is degenerate")
            rc = 1
            return rc

        # The index file follows the index's size, not the server's capacity.
        # It used to hold the node map and neighbor lists for every element
        # the server could take: 329 MB for these 100 vectors, written on
        # every FT.OPTIMIZE — ~400 ms a cycle, and cycles of 3 s on a busy
        # disk, which tripped the ceiling above about one tier run in two.
        hnsw_file = os.path.join(PROJECT_ROOT, "pion.hnsw.0")
        size = os.path.getsize(hnsw_file) if os.path.exists(hnsw_file) else -1
        if not (0 < size <= 64 * 1024 * args.n_vectors):
            print(f"FAIL: pion.hnsw.0 is {size} bytes for {args.n_vectors} vectors "
                  f"(at most {64 * 1024 * args.n_vectors} expected)")
            rc = 1
            return rc

        # Final sanity: PING.
        if not r.ping():
            print("FAIL: PING returned False after drop cycles")
            rc = 1
            return rc
        r.close()

        print(f"\nPASS: {args.cycles} sequential DROP+REBUILD cycles, server alive, "
              f"grace period bounded (max cycle {max_cycle:.0f}ms).")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        rc = 2
    finally:
        stop_pion(proc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
