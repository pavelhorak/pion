"""D-sweep bench for Pion's --metal-attention path.

Validates that the templated kernel (Phase B, function-constant D_HEAD)
produces correct output and reasonable latency at every supported D.

Setup:
  Term 1: ./pion-server --metal-attention -w 1
  Term 2: python3.11 tests/bench_pion_metal_attention_dsweep.py
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
D_VALUES = [64, 96, 128, 160, 192, 256]
WARMUP = 10
ITERS = 50


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

    def query(self, sid, layer, Q, top_k=2048):
        Hh, Dd = Q.shape
        r = self.call("ATTEND.PREFIX.QUERY", sid, str(layer), str(Hh), str(Dd), str(top_k),
                      np.ascontiguousarray(Q, dtype=np.float32).tobytes())
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            return None, r
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        return np.frombuffer(r[nl + 2:nl + 2 + blen], dtype=np.float32).copy().reshape(Hh, Dd), r


def cpu_attention(Q, K, V, D):
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q[:, None, :], K.transpose(0, 2, 1)) * scale
    sm = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(sm)
    a = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(a, V).reshape(H, D)


def cosine(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def percentile(times, p):
    s = sorted(times)
    return s[min(int(len(s) * p), len(s) - 1)]


def main():
    print(f"D-sweep at H={H} N={N} (warmup={WARMUP} iters={ITERS}, fresh-Q each iter)")
    print()
    try:
        c = Conn()
    except OSError:
        print("FAIL: pion not reachable. Start with: ./pion-server --metal-attention -w 1")
        return 2

    rng = np.random.default_rng(0)
    print(f"  {'D':>3s}  {'cosine':>10s}  {'median':>9s}  {'p95':>9s}  {'min':>9s}  result")
    print(f"  {'---':>3s}  {'-' * 10}  {'-' * 9}  {'-' * 9}  {'-' * 9}  ------")
    fail = 0
    for D in D_VALUES:
        K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
        V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
        Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
        cpu_ref = cpu_attention(Q, K, V, D)
        sid = f"dsweep_{D}_{int(time.time() * 1000)}"
        ok, _ = c.store_kv(sid, 0, K, V)
        if not ok:
            print(f"  {D:>3d}  STORE failed")
            fail += 1
            continue
        out, raw = c.query(sid, 0, Q)
        if out is None:
            print(f"  {D:>3d}  QUERY failed: {raw[:80]!r}")
            fail += 1
            continue
        cos = cosine(out, cpu_ref)

        # Fresh-Q timing: pre-generate Q pool, sweep through it.
        Q_pool = [(rng.standard_normal((H, D)) * 0.5).astype(np.float32)
                  for _ in range(ITERS + WARMUP)]
        for i in range(WARMUP):
            c.query(sid, 0, Q_pool[i])
        times = []
        for i in range(ITERS):
            t0 = time.perf_counter_ns()
            c.query(sid, 0, Q_pool[WARMUP + i])
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)
        median = percentile(times, 0.5)
        p95 = percentile(times, 0.95)
        result = "OK" if cos >= 0.999 else "FAIL"
        if cos < 0.999:
            fail += 1
        print(f"  {D:>3d}  {cos:>10.7f}  {median:>7.3f}ms  {p95:>7.3f}ms  {min(times):>7.3f}ms  {result}")

    print()
    if fail:
        print(f"FAIL: {fail} D-values had problems")
        return 1
    print(f"PASS: all {len(D_VALUES)} D-values correct (cosine ≥ 0.999)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
