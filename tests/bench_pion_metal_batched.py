"""Phase C — M>1 batched-Q correctness + perf gate.

Validates the new sdpa_batched_q_fp32 kernel: output cosine ≥ 0.999 vs CPU
reference, LSE ≈ CPU log-sum-exp (max+log(sum_exp)), and prints latency
across a sweep of M values.

Setup:
  Term 1: ./pion-server --metal-attention -w 1
  Term 2: python3.11 tests/bench_pion_metal_batched.py
"""
from __future__ import annotations

import socket
import sys
import time

import numpy as np

HOST = "127.0.0.1"
PORT = 1974
H = 8
N = 2048
D = 128
M_VALUES = [2, 4, 8, 16, 32, 64, 128]
WARMUP = 5
ITERS = 30


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
        Hh, Nn, Dd = K.shape
        r = self.call("ATTEND.PREFIX.STORE", sid, str(layer), str(Hh), str(Nn), str(Dd),
                      np.ascontiguousarray(K, dtype=np.float32).tobytes(),
                      np.ascontiguousarray(V, dtype=np.float32).tobytes())
        return r.startswith(b"+OK"), r

    def query_batched(self, sid, layer, Q, top_k=2048):
        Hh, Mm, Dd = Q.shape
        r = self.call("ATTEND.PREFIX.QUERY", sid, str(layer), str(Hh), str(Dd), str(top_k),
                      np.ascontiguousarray(Q, dtype=np.float32).tobytes())
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            return None, None, r
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        body = r[nl + 2:nl + 2 + blen]
        out_size = Hh * Mm * Dd * 4
        lse_size = Hh * Mm * 4
        if blen == out_size:
            # M=1 wire format (no LSE)
            out = np.frombuffer(body, dtype=np.float32).copy().reshape(Hh, Mm, Dd) if Mm > 1 \
                  else np.frombuffer(body, dtype=np.float32).copy().reshape(Hh, Dd)
            return out, None, r
        if blen != out_size + lse_size:
            return None, None, r
        out = np.frombuffer(body[:out_size], dtype=np.float32).copy().reshape(Hh, Mm, Dd)
        lse = np.frombuffer(body[out_size:], dtype=np.float32).copy().reshape(Hh, Mm)
        return out, lse, r


def cpu_attention_batched(Q, K, V, D):
    # Q [H, M, D]; K, V [H, N, D]
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale  # [H, M, N]
    m = scores.max(axis=-1, keepdims=True)
    exp_s = np.exp(scores - m)
    sum_exp = exp_s.sum(axis=-1, keepdims=True)
    weights = exp_s / sum_exp
    out = np.matmul(weights, V)               # [H, M, D]
    lse = (m + np.log(sum_exp)).squeeze(-1)   # [H, M]
    return out, lse


def cosine(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def percentile(times, p):
    s = sorted(times)
    return s[min(int(len(s) * p), len(s) - 1)]


def main():
    print(f"Phase C M>1 sweep at H={H} N={N} D={D}")
    print()
    try:
        c = Conn()
    except OSError:
        print("FAIL: pion not reachable. Start with: ./pion-server --metal-attention -w 1")
        return 2

    rng = np.random.default_rng(0)
    K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    sid = f"phaseC_{int(time.time() * 1000)}"
    ok, _ = c.store_kv(sid, 0, K, V)
    if not ok:
        print("FAIL: ATTEND.PREFIX.STORE rejected")
        return 3

    print(f"  {'M':>4s}  {'cos(out)':>10s}  {'maxΔlse':>9s}  {'median':>9s}  {'p95':>9s}  result")
    print(f"  {'-' * 4}  {'-' * 10}  {'-' * 9}  {'-' * 9}  {'-' * 9}  ------")
    fail = 0
    for M in M_VALUES:
        Q = (rng.standard_normal((H, M, D)) * 0.5).astype(np.float32)
        out_cpu, lse_cpu = cpu_attention_batched(Q, K, V, D)
        out, lse, raw = c.query_batched(sid, 0, Q)
        if out is None:
            print(f"  {M:>4d}  QUERY rejected: {raw[:80]!r}")
            fail += 1
            continue
        if lse is None:
            print(f"  {M:>4d}  no LSE returned (wire mismatch?)")
            fail += 1
            continue
        cos = cosine(out, out_cpu)
        max_lse_delta = float(np.abs(lse - lse_cpu).max())

        # Fresh-Q timing
        Q_pool = [(rng.standard_normal((H, M, D)) * 0.5).astype(np.float32)
                  for _ in range(ITERS + WARMUP)]
        for i in range(WARMUP):
            c.query_batched(sid, 0, Q_pool[i])
        times = []
        for i in range(ITERS):
            t0 = time.perf_counter_ns()
            c.query_batched(sid, 0, Q_pool[WARMUP + i])
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)
        median = percentile(times, 0.5)
        p95 = percentile(times, 0.95)
        result = "OK" if cos >= 0.999 and max_lse_delta < 1e-3 else "FAIL"
        if cos < 0.999 or max_lse_delta >= 1e-3:
            fail += 1
        print(f"  {M:>4d}  {cos:>10.7f}  {max_lse_delta:>9.6f}  {median:>7.3f}ms  {p95:>7.3f}ms  {result}")

    print()
    if fail:
        print(f"FAIL: {fail} M-values had problems")
        return 1
    print(f"PASS: all {len(M_VALUES)} M-values correct (cosine ≥ 0.999, |Δlse| < 1e-3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
