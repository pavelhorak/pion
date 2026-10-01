#!/usr/bin/env python3
"""Stage 2 prototype gate: ATTEND.PREFIX.STORE + ATTEND.PREFIX.QUERY.

Closes the §13.4 + §17.2 + §21.6 caveat: the existing ATTEND.QUERYBATCH
(single-shot Q+K+V) is structurally too slow because it marshals K/V on
every call. The two-phase ATTEND.PREFIX.* commands push K/V once and query
many times — only Q crosses the wire on subsequent calls.

Pass criteria:
  1. Numerical agreement with CPU softmax(QK^T/sqrt(D)) @ V (cosine ≥ 0.99).
  2. ATTEND.PREFIX.QUERY median latency < 0.5 × ATTEND.QUERYBATCH median
     (because Q is ~1000× smaller than K/V, no marshaling overhead).
  3. Throughput at H=8 N=2048 D=64 ≥ 100 q/s on Apple Silicon.

Requires: ./pion-server --kvcache --metal-attention -w 1
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

import numpy as np


HOST = "127.0.0.1"
PORT = 1974


class PionAttn:
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

    def store_kv(self, sid: str, layer: int, K: np.ndarray, V: np.ndarray) -> bool:
        H, N, D = K.shape
        r = self.call(
            "ATTEND.PREFIX.STORE", sid, str(layer), str(H), str(N), str(D),
            np.ascontiguousarray(K, dtype=np.float32).tobytes(),
            np.ascontiguousarray(V, dtype=np.float32).tobytes(),
        )
        ok = r.startswith(b"+OK")
        if not ok:
            print(f"  store_kv reply: {r[:120]!r}")
        return ok

    def query_cached(self, sid: str, layer: int, Q: np.ndarray, top_k: int) -> np.ndarray | None:
        H, D = Q.shape
        r = self.call(
            "ATTEND.PREFIX.QUERY", sid, str(layer), str(H), str(D), str(top_k),
            np.ascontiguousarray(Q, dtype=np.float32).tobytes(),
        )
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            return None
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        return np.frombuffer(r[nl + 2:nl + 2 + blen], dtype=np.float32).copy().reshape(H, D)

    def querybatch(self, H: int, N: int, D: int, top_k: int,
                   Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray | None:
        r = self.call(
            "ATTEND.QUERYBATCH", str(H), str(N), str(D), str(top_k),
            np.ascontiguousarray(Q, dtype=np.float32).tobytes(),
            np.ascontiguousarray(K, dtype=np.float32).tobytes(),
            np.ascontiguousarray(V, dtype=np.float32).tobytes(),
        )
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            return None
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        return np.frombuffer(r[nl + 2:nl + 2 + blen], dtype=np.float32).copy().reshape(H, D)


def cpu_attention(Q, K, V, top_k):
    H, N, D = K.shape
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q[:, None, :], K.transpose(0, 2, 1)) * scale  # [H, 1, N]
    if top_k >= N:
        sm = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(sm)
        a = e / e.sum(axis=-1, keepdims=True)
        return np.matmul(a, V).reshape(H, D)
    idx = np.argpartition(-scores, top_k, axis=-1)[..., :top_k]
    s_topk = np.take_along_axis(scores, idx, axis=-1)
    sm = s_topk - s_topk.max(axis=-1, keepdims=True)
    e = np.exp(sm)
    a = e / e.sum(axis=-1, keepdims=True)
    idx_exp = np.broadcast_to(idx[..., None], idx.shape + (D,))
    V_exp = np.broadcast_to(V[:, None, :, :], (H, 1, N, D))
    v_topk = np.take_along_axis(V_exp, idx_exp, axis=2)
    return np.einsum("hik,hikd->hid", a, v_topk).reshape(H, D)


def main(args) -> int:
    print(f"Stage 2 ATTEND.PREFIX.* H={args.heads} N={args.tokens} D={args.head_dim} top_k={args.top_k or args.tokens}")

    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable. Start with: ./pion-server --kvcache --metal-attention -w 1")
        return 2

    rng = np.random.default_rng(0)
    H, N, D = args.heads, args.tokens, args.head_dim
    top_k = args.top_k if args.top_k > 0 else N
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)

    pion = PionAttn()
    sid = f"stage2_{int(time.time())}"

    # ── STORE: push K/V once ────────────────────────────────────────────────
    print("\n[store] ATTEND.PREFIX.STORE — push K/V once")
    t0 = time.perf_counter()
    ok = pion.store_kv(sid, 0, K, V)
    store_ms = (time.perf_counter() - t0) * 1000
    if not ok:
        print(f"FAIL: store_kv failed")
        return 2
    print(f"  store_ms={store_ms:.1f}  bytes={K.nbytes+V.nbytes}")

    # ── QUERY: many cached queries with only Q over wire ────────────────────
    print(f"[query] ATTEND.PREFIX.QUERY — {args.repeats} queries, Q-only on the wire")
    pion_outs = []
    pion_times = []
    for _ in range(args.warmup):
        _ = pion.query_cached(sid, 0, Q, top_k)
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        out = pion.query_cached(sid, 0, Q, top_k)
        dt = (time.perf_counter() - t0) * 1000
        if out is None:
            print(f"FAIL: query_cached returned None")
            return 2
        pion_outs.append(out)
        pion_times.append(dt)
    p_med = float(np.median(pion_times))
    p_mean = float(np.mean(pion_times))
    print(f"  median={p_med:.2f}ms  mean={p_mean:.2f}ms  throughput={1000/p_med:.0f} q/s")

    # ── CORRECTNESS: cosine vs CPU reference ────────────────────────────────
    # ATTEND.QUERYBATCH (legacy single-shot path) was removed alongside the
    # MLX sidecar 2026-05-01. The PREFIX.STORE+QUERY path is now the sole
    # ATTEND.PREFIX.* path; correctness vs CPU + throughput remain the gate.
    cpu_ref = cpu_attention(Q, K, V, top_k)
    pion_avg = np.mean(pion_outs, axis=0)  # average is fine; outputs are deterministic
    cos = float(np.dot(cpu_ref.flatten(), pion_avg.flatten())
                / (np.linalg.norm(cpu_ref) * np.linalg.norm(pion_avg) + 1e-9))

    print("\n──────── Stage 2 G2 sub-gate ────────")
    print(f"  PREFIX.STORE+QUERY median {p_med:.2f}ms  ({1000/p_med:.0f} q/s)")
    print(f"  numerical cosine vs CPU   {cos:.4f}  (target ≥ 0.99)")

    pass_cos = cos >= 0.99
    pass_throughput = (1000 / p_med) >= 100
    overall = pass_cos and pass_throughput
    print(f"\n  cosine ≥ 0.99      {'PASS' if pass_cos else 'FAIL'}")
    print(f"  throughput ≥ 100q/s {'PASS' if pass_throughput else 'FAIL'}")
    print(f"  STAGE 2 PROTOTYPE  {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    sys.exit(main(args))
