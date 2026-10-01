"""End-to-end bench: ATTEND.PREFIX.QUERY through the wire (Mojo→C→Metal→C→Mojo→TCP)
versus MLX raw compute (no wire, no Pion).

Goal: confirm the inline-MSL kernel that beats MLX SDPA by 1.34-1.55× standalone
also beats it (or at least matches) end-to-end with TCP framing on top.

Setup:
  Term 1: ./pion-server --metal-attention -w 1
  Term 2: python3.11 tests/bench_pion_metal_attention.py

Shape: H=8 N=2048 d_head=128 (matches tests/bench_msl_sdpa_q1.m / tests/bench_mlx_vs_mojo.py).

Run order:
  1. Connect, ATTEND.PREFIX.STORE H*N*D K/V (one-time cost, not timed).
  2. Loop ATTEND.PREFIX.QUERY 100 times with the SAME random Q (warm cache).
  3. Loop ATTEND.PREFIX.QUERY 100 times with FRESH Q each iter (worst case).
  4. Report median/p95/min for both. Compare with MLX raw from bench_mlx_vs_mojo.py.
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

import numpy as np

HOST = "127.0.0.1"
PORT = 1974
H, N, D = 8, 2048, 128
WARMUP = 20
ITERS = 100

# Override defaults via env so the same harness can sweep N for W3.1.
import os as _os
_NENV = _os.environ.get("PION_BENCH_N")
if _NENV:
    N = int(_NENV)
_HENV = _os.environ.get("PION_BENCH_H")
if _HENV:
    H = int(_HENV)
_DENV = _os.environ.get("PION_BENCH_D")
if _DENV:
    D = int(_DENV)


def encode(parts):
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


class Conn:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.s.settimeout(60)
        self.s.connect((HOST, PORT))
        self.buf = b""

    def first_complete(self, d):
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
        self.s.sendall(encode(parts))
        while True:
            done = self.first_complete(self.buf)
            if done is not None:
                msg = self.buf[:done]
                self.buf = self.buf[done:]
                return msg
            chunk = self.s.recv(64 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("pion closed")
            self.buf += chunk

    def store_kv(self, sid, layer, K, V):
        Kb = np.ascontiguousarray(K, dtype=np.float32).tobytes()
        Vb = np.ascontiguousarray(V, dtype=np.float32).tobytes()
        Hh, Nn, Dd = K.shape
        r = self.call("ATTEND.PREFIX.STORE", sid, str(layer), str(Hh), str(Nn), str(Dd), Kb, Vb)
        return r.startswith(b"+OK"), r

    def query(self, sid, layer, Q, top_k=2048):
        Qb = np.ascontiguousarray(Q, dtype=np.float32).tobytes()
        Hh, Dd = Q.shape
        r = self.call("ATTEND.PREFIX.QUERY", sid, str(layer), str(Hh), str(Dd), str(top_k), Qb)
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            return None, r
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        return np.frombuffer(r[nl + 2:nl + 2 + blen], dtype=np.float32).copy().reshape(Hh, Dd), r


def cpu_attention(Q, K, V):
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q[:, None, :], K.transpose(0, 2, 1)) * scale
    sm = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(sm)
    a = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(a, V).reshape(H, D)


def cosine(a, b):
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def percentile(times_ms, p):
    s = sorted(times_ms)
    return s[min(int(len(s) * p), len(s) - 1)]


def bench(label, fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)
    median = percentile(times, 0.5)
    p95 = percentile(times, 0.95)
    print(f"  {label:<42s} median={median:.3f} ms  p95={p95:.3f} ms  min={min(times):.3f} ms")
    return median


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mlx", action="store_true", help="skip MLX comparison")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero on regression (cosine < 0.999 or median > 0.6 ms). "
                         "Skips MLX side; just validates Pion's --metal-attention path.")
    args = ap.parse_args()

    print(f"shape: H={H} N={N} d_head={D}")
    print(f"warmup={WARMUP} iters={ITERS}")
    print()

    rng = np.random.default_rng(0)
    K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    cpu_ref = cpu_attention(Q, K, V)

    try:
        c = Conn()
    except OSError:
        print("FAIL: pion not reachable. Start with: ./pion-server --metal-attention -w 1")
        return 2

    sid = f"bench_{int(time.time())}"
    ok, _ = c.store_kv(sid, 0, K, V)
    if not ok:
        print("FAIL: ATTEND.PREFIX.STORE rejected")
        return 3

    out, raw = c.query(sid, 0, Q)
    if out is None:
        print(f"FAIL: ATTEND.PREFIX.QUERY rejected: {raw[:200]!r}")
        return 4

    cos_pion = cosine(out, cpu_ref)
    print(f"correctness: cosine(cpu, pion-msl) = {cos_pion:.7f}")
    if cos_pion < 0.999:
        print("FAIL: cosine < 0.999")
        return 5

    print("\nPion ATTEND.PREFIX.QUERY (Mojo→C→Metal→C→Mojo→TCP→Python):")
    median_warm = bench("warm-cache, same Q", lambda: c.query(sid, 0, Q))

    Q_pool = [(rng.standard_normal((H, D)) * 0.5).astype(np.float32) for _ in range(ITERS + WARMUP)]
    counter = [0]
    def fresh_q():
        Qi = Q_pool[counter[0] % len(Q_pool)]
        counter[0] += 1
        c.query(sid, 0, Qi)
    median_fresh = bench("fresh Q each iter", fresh_q)

    if args.gate:
        # Use the fresh-Q median — warm-cache shows the same code path but is
        # subject to OS cache warmth artifacts (saw 0.624 ms median on a noisy
        # run vs 0.426 ms fresh-Q in the same invocation). Fresh-Q at H=8/N=2048/
        # d=128 has measured ~0.42-0.45 ms across multiple runs; 0.8 ms catches
        # any regression worse than 2× without flaking.
        gate_threshold_ms = 0.8
        if cos_pion < 0.999:
            print(f"\nGATE FAILED — cosine {cos_pion:.7f} < 0.999")
            return 6
        if median_fresh > gate_threshold_ms:
            print(f"\nGATE FAILED — fresh-Q median {median_fresh:.3f} ms > {gate_threshold_ms} ms (perf regression)")
            return 7
        print(f"\nGATE PASSED — cosine={cos_pion:.7f} fresh-Q median={median_fresh:.3f} ms (≤ {gate_threshold_ms} ms)")
        return 0

    if not args.no_mlx:
        print("\nMLX raw compute (no wire, no Pion — same shape):")
        try:
            import mlx.core as mx
            Qm = mx.array(Q[:, None, :])
            Km = mx.array(K)
            Vm = mx.array(V)
            mx.eval(Qm, Km, Vm)
            scale = 1.0 / np.sqrt(D)

            def mlx_naive():
                scores = mx.matmul(Qm, mx.transpose(Km, (0, 2, 1))) * scale
                sm = scores - mx.max(scores, axis=-1, keepdims=True)
                w = mx.exp(sm)
                w = w / mx.sum(w, axis=-1, keepdims=True)
                o = mx.matmul(w, Vm)
                mx.eval(o)
                return o

            def mlx_sdpa():
                Q4 = mx.expand_dims(Qm, 0)
                K4 = mx.expand_dims(Km, 0)
                V4 = mx.expand_dims(Vm, 0)
                o = mx.fast.scaled_dot_product_attention(Q4, K4, V4, scale=scale)
                mx.eval(o)
                return o

            bench("MLX naive matmul + softmax", mlx_naive)
            bench("MLX fast.scaled_dot_product_attention", mlx_sdpa)
        except ImportError:
            print("  (mlx.core not importable — skipping)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
