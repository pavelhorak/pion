#!/usr/bin/env python3
"""§25 feasibility check — wire cost of 28 sequential attend_query calls
at Llama-3.2-3B's actual prefix size (1160 tokens).

If this dominates over what mlx-lm currently spends on prefix-attention,
the §25 monkey-patch can't pay for itself. If it doesn't, the next step
(implementing the online-softmax merge in mlx-lm's Attention.__call__) is
worth the engineering.

Scenario: §22.2 workload, 3B model, ~1160-token prefix, suffix prefill of
~30 tokens. For each of the 28 layers, mlx-lm currently runs full attention
over (1160 + 30) tokens. To offload prefix-attention to Pion via
ATTEND.PREFIX.QUERY, we'd issue one call per layer per token of suffix
prefill, OR (better) one call per layer for the whole suffix Q at once.

We test BOTH wire patterns since the protocol's batchability for multi-Q
isn't shipped yet — single-Q-per-call gives the upper bound today.

Requires `pion-server --kvcache -w 1`.
"""
from __future__ import annotations

import socket
import sys
import time

import numpy as np


HOST = "127.0.0.1"
PORT = 1974

# Match Llama-3.2-3B-Instruct-4bit GQA shape
N_LAYERS = 28
N_KV_HEADS = 8
HEAD_DIM = 128
PREFIX_LEN = 1160
SUFFIX_LEN = 30  # rough suffix prefill length per §22.2


class _RESP:
    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.sock.settimeout(60)
        self.sock.connect((HOST, PORT))
        self.buf = b""

    @staticmethod
    def _encode(parts) -> bytes:
        out = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            if isinstance(p, bytes):
                out.append(f"${len(p)}\r\n".encode())
                out.append(p)
                out.append(b"\r\n")
            else:
                s = str(p)
                out.append(f"${len(s)}\r\n{s}\r\n".encode())
        return b"".join(out)

    def _read(self) -> bytes:
        while True:
            done = self._first_complete(self.buf)
            if done is not None:
                msg = self.buf[:done]
                self.buf = self.buf[done:]
                return msg
            chunk = self.sock.recv(64 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("pion closed")
            self.buf += chunk

    @staticmethod
    def _first_complete(d: bytes):
        if len(d) < 3:
            return None
        p = d[0:1]
        nl = d.find(b"\r\n")
        if nl < 0:
            return None
        if p in (b"+", b"-", b":"):
            return nl + 2
        if p == b"$":
            ls = d[1:nl].decode()
            if ls == "-1":
                return nl + 2
            n = int(ls)
            need = nl + 2 + n + 2
            return need if len(d) >= need else None
        return nl + 2

    def call(self, *parts):
        self.sock.sendall(self._encode(parts))
        return self._read()


def main() -> int:
    print(f"§25 feasibility — Llama-3.2-3B-shaped wire cost test")
    print(f"  N_LAYERS={N_LAYERS} H={N_KV_HEADS} D={HEAD_DIM} prefix={PREFIX_LEN} suffix={SUFFIX_LEN}")

    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable. Start with: ./pion-server --kvcache -w 1")
        return 2

    rng = np.random.default_rng(0)
    H, N, D = N_KV_HEADS, PREFIX_LEN, HEAD_DIM
    K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    Q_single = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)

    pion = _RESP()
    sid = f"feas_{int(time.time())}"

    # Push K/V to MLX sidecar — once per layer
    print("[setup] storing K/V for 28 layers...")
    t0 = time.perf_counter()
    for layer_id in range(N_LAYERS):
        r = pion.call(
            "ATTEND.PREFIX.STORE", sid, str(layer_id), str(H), str(N), str(D),
            np.ascontiguousarray(K, dtype=np.float32).tobytes(),
            np.ascontiguousarray(V, dtype=np.float32).tobytes(),
        )
        if not r.startswith(b"+OK"):
            print(f"FAIL: STORE failed at layer {layer_id}: {r[:80]!r}")
            return 2
    setup_ms = (time.perf_counter() - t0) * 1000
    print(f"  STORE × {N_LAYERS} layers: {setup_ms:.0f} ms total (~{setup_ms/N_LAYERS:.1f} ms/layer)")

    # ── Pattern 1: one ATTEND.PREFIX.QUERY per layer (suffix collapsed to 1 Q) ─
    # This is the realistic shape for greedy decode TTFT — the Q is shape (H, D)
    # representing the last suffix token's query that needs attention over the
    # cached prefix. Mlx-lm would call this once per layer per generated token.
    print("\n[pattern 1] 28 sequential attend_query calls (one per layer)")
    print("            simulates: greedy decode of one new token, prefix attention only")
    times = []
    for _ in range(20):  # 20 trials
        t0 = time.perf_counter()
        for layer_id in range(N_LAYERS):
            r = pion.call(
                "ATTEND.PREFIX.QUERY", sid, str(layer_id), str(H), str(D), str(N),
                np.ascontiguousarray(Q_single, dtype=np.float32).tobytes(),
            )
            if not r.startswith(b"$"):
                print(f"FAIL: QUERY at layer {layer_id}: {r[:80]!r}")
                return 2
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
    p1_med = float(np.median(times))
    p1_p99 = float(np.percentile(times, 99))
    print(f"  median: {p1_med:.1f} ms  p99: {p1_p99:.1f} ms  (per call: {p1_med/N_LAYERS:.2f} ms)")

    # ── Pattern 2: 28 layers × 30 suffix tokens = 840 calls per request ─
    # This is the worst case: prefill the suffix one token at a time per layer.
    # Real mlx-lm prefill batches the suffix across len in one Attention call,
    # so this is a strict upper bound the protocol would have to beat.
    print("\n[pattern 2] 28 layers × 30 suffix-Q calls = 840 total calls")
    print("            simulates: per-token-per-layer if Q can't be batched")
    t0 = time.perf_counter()
    for _ in range(SUFFIX_LEN):
        for layer_id in range(N_LAYERS):
            r = pion.call(
                "ATTEND.PREFIX.QUERY", sid, str(layer_id), str(H), str(D), str(N),
                np.ascontiguousarray(Q_single, dtype=np.float32).tobytes(),
            )
    p2_total = (time.perf_counter() - t0) * 1000
    print(f"  total: {p2_total:.0f} ms  (per call: {p2_total/(SUFFIX_LEN * N_LAYERS):.2f} ms)")

    # ── Pattern 3: 28 layers, batched Q across the whole suffix in one call ─
    # This is the production-path Stage-2 monkey-patch shape: M=SUFFIX_LEN tokens
    # of suffix-Q sent in a single ATTEND.PREFIX.QUERY per layer (M derived from
    # the Q blob length, dense softmax). One round trip per layer instead of
    # SUFFIX_LEN round trips per layer.
    print("\n[pattern 3] 28 layers × 1 batched-Q call (M=30 tokens per call)")
    print("            simulates: TTFT prefix-attention with batched-Q protocol")
    Q_batched = (rng.standard_normal((H, SUFFIX_LEN, D)) * 0.5).astype(np.float32)
    times3 = []
    for _ in range(20):  # 20 trials
        t0 = time.perf_counter()
        for layer_id in range(N_LAYERS):
            r = pion.call(
                "ATTEND.PREFIX.QUERY", sid, str(layer_id), str(H), str(D), str(N),
                np.ascontiguousarray(Q_batched, dtype=np.float32).tobytes(),
            )
            if not r.startswith(b"$"):
                print(f"FAIL: batched QUERY at layer {layer_id}: {r[:80]!r}")
                return 2
        dt = (time.perf_counter() - t0) * 1000
        times3.append(dt)
    p3_med = float(np.median(times3))
    p3_p99 = float(np.percentile(times3, 99))
    print(f"  median: {p3_med:.1f} ms  p99: {p3_p99:.1f} ms  (per layer: {p3_med/N_LAYERS:.2f} ms)")
    speedup_vs_p2 = p2_total / p3_med if p3_med > 0 else 0
    print(f"  vs Pattern 2 (per-suffix-token):  {speedup_vs_p2:.1f}× faster (single batched call replaces {SUFFIX_LEN} sequential)")

    # ── Verdict against baseline ─────────────────────────────────────────────
    # Current §22.2 warm 3B mean TTFT is ~278 ms total. Of that, mlx-lm spends
    # roughly 28 layers × ~5 ms = ~140 ms on prefix-attention.
    print("\n──────── Feasibility verdict ────────")
    print(f"  Baseline: mlx-lm prefix-attention  ~140 ms (28 layers × ~5 ms)")
    print(f"  Pattern 1 (per-layer single-Q):    {p1_med:.1f} ms  — decode-step path")
    print(f"  Pattern 2 (per-layer per-suffix):  {p2_total:.0f} ms — old TTFT path (no batched-Q)")
    print(f"  Pattern 3 (per-layer batched-Q):   {p3_med:.1f} ms  — TTFT path with new protocol")
    print(f"")
    if p3_med < 70:
        print(f"  Pattern 3 wire cost ({p3_med:.0f} ms) << local attention ({140} ms)")
        print(f"  → Stage 2 monkey-patch viable for TTFT itself, not just decode.")
    elif p3_med < 140:
        print(f"  Pattern 3 wire cost ({p3_med:.0f} ms) is comparable to local attention ({140} ms)")
        print(f"  → Marginal for TTFT; clean win for decode (Pattern 1).")
    else:
        print(f"  Pattern 3 wire cost ({p3_med:.0f} ms) >= local attention ({140} ms)")
        print(f"  → TTFT-side monkey-patch still loses on this Mac; decode-step still wins.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
