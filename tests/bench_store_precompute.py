#!/usr/bin/env python3
"""W11 Phase 2 follow-up — STORE-time selector precompute latency check.

Validates that doing K_mean + K_min/K_max precompute at ATTEND.PREFIX.STORE
time (default since the precompute commit) moves the lazy populate off the
first-query critical path:

  PION_SDPA_NO_STORE_PRECOMPUTE=1  → STORE fast, first QUERY slow (lazy)
  default (precompute at STORE)    → STORE slow, first QUERY fast (warm)

Total wall is the same; this just moves where the latency sits. The
first-query-latency win matters because that's user-facing TTFT on the
sparse path.

Pre-req:
  PION_SDPA_NO_STORE_PRECOMPUTE=... ./pion-server --kvcache --metal-attention \
      --no-auto-embed -w 1

(Run this script TWICE — once with the env var set on the server, once
without — and compare the printed timings.)
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pion-vllm-mlx"))

from pion_vllm_mlx.prompt_cache import PionPromptCache


def store_via_resp(host: str, port: int, sid: str, layer: int,
                   K: np.ndarray, V: np.ndarray) -> float:
    """ATTEND.PREFIX.STORE wire frame. Returns wall ms."""
    H, N, D = K.shape
    Kb = K.astype(np.float32).tobytes()
    Vb = V.astype(np.float32).tobytes()
    sock = socket.socket()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024 * 1024)
    sock.settimeout(60.0)
    sock.connect((host, port))
    parts = [b"ATTEND.PREFIX.STORE", sid.encode(), str(layer).encode(),
             str(H).encode(), str(N).encode(), str(D).encode(), Kb, Vb]
    frame = f"*{len(parts)}\r\n".encode()
    for p in parts:
        frame += f"${len(p)}\r\n".encode() + p + b"\r\n"
    t0 = time.perf_counter()
    sock.sendall(frame)
    line = b""
    while not line.endswith(b"\r\n"):
        c = sock.recv(1)
        if not c:
            break
        line += c
    dt = (time.perf_counter() - t0) * 1000
    sock.close()
    if not line.startswith(b"+OK"):
        raise RuntimeError(f"STORE failed: {line!r}")
    return dt


def main():
    HOST, PORT = "127.0.0.1", 1974
    H = 4
    N = 16384
    D = 128
    B = 64
    K_top = 8
    LAYER = 0
    NS = "store_precompute_bench"
    SID = f"{NS}_attn"

    print(f"shape: H={H} N={N} D={D} B={B} K_top={K_top}")
    print(f"  bytes per blob: {H*N*D*4/1e6:.1f} MB K + same V = {2*H*N*D*4/1e6:.1f} MB total")
    print()

    rng = np.random.RandomState(42)
    K = rng.randn(H, N, D).astype(np.float32) * 0.1
    V = rng.randn(H, N, D).astype(np.float32)
    Q = rng.randn(H, D).astype(np.float32)

    print(f"[1] STORE K/V to Pion...")
    store_ms = store_via_resp(HOST, PORT, SID, LAYER, K, V)
    print(f"    STORE: {store_ms:7.1f} ms")

    pc = PionPromptCache(model=None, host=HOST, port=PORT)
    pc._local[NS] = N

    print(f"\n[2] first QUERY_SPARSE_AUTO selector=block_mean (cache cold OR warm)")
    t0 = time.perf_counter()
    _ = pc.attend_query_sparse_auto(NS, LAYER, Q, B, K_top, H_kv=H,
                                     selector="block_mean")
    bm_first_ms = (time.perf_counter() - t0) * 1000
    print(f"    block_mean[0]: {bm_first_ms:7.1f} ms")

    print(f"\n[3] second + third block_mean (cache warm)")
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        _ = pc.attend_query_sparse_auto(NS, LAYER, Q, B, K_top, H_kv=H,
                                         selector="block_mean")
        times.append((time.perf_counter() - t0) * 1000)
    print(f"    block_mean[1..3]: " + " ".join(f"{t:6.1f}" for t in times) + " ms")
    bm_warm_med = sorted(times)[len(times)//2]

    print(f"\n[4] first QUERY_SPARSE_AUTO selector=quest")
    t0 = time.perf_counter()
    _ = pc.attend_query_sparse_auto(NS, LAYER, Q, B, K_top, H_kv=H,
                                     selector="quest")
    qs_first_ms = (time.perf_counter() - t0) * 1000
    print(f"    quest[0]: {qs_first_ms:7.1f} ms")

    print(f"\n[5] second + third quest (cache warm)")
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        _ = pc.attend_query_sparse_auto(NS, LAYER, Q, B, K_top, H_kv=H,
                                         selector="quest")
        times.append((time.perf_counter() - t0) * 1000)
    print(f"    quest[1..3]: " + " ".join(f"{t:6.1f}" for t in times) + " ms")
    qs_warm_med = sorted(times)[len(times)//2]

    print("\n=== Summary ===")
    print(f"  STORE wall:            {store_ms:7.1f} ms (includes precompute when enabled)")
    print(f"  block_mean first:      {bm_first_ms:7.1f} ms")
    print(f"  block_mean warm (med): {bm_warm_med:7.1f} ms")
    print(f"  Δ (first - warm) BM:   {bm_first_ms - bm_warm_med:7.1f} ms  ← lazy populate spike if STORE precompute disabled")
    print(f"  quest first:           {qs_first_ms:7.1f} ms")
    print(f"  quest warm (med):      {qs_warm_med:7.1f} ms")
    print(f"  Δ (first - warm) QS:   {qs_first_ms - qs_warm_med:7.1f} ms  ← lazy populate spike if STORE precompute disabled")
    print()
    print(f"  Expected with STORE precompute enabled (default):")
    print(f"    Δ ≈ 0 — first call is cache-warm, no lazy spike")
    print(f"  Expected with PION_SDPA_NO_STORE_PRECOMPUTE=1:")
    print(f"    Δ ≈ several ms (block_mean) / tens of ms (Quest) at this N")


if __name__ == "__main__":
    main()
