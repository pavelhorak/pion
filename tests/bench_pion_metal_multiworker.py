"""Phase 2 — Multi-worker SDPA test.

Spins up pion-server with `-w 4 --independent-workers --metal-attention` and validates that:
  - Each worker independently serves ATTEND.PREFIX.STORE/QUERY (correctness).
  - Concurrent stores/queries from multiple connections don't corrupt state.
  - The per-worker session caches are isolated (Pion's shared-nothing accept
    routing means a connection's traffic always hits the same worker, so
    sessions stored on one connection are findable on the same connection).

Setup:
  Term 1: ./pion-server --metal-attention -w 4 --independent-workers
  Term 2: python3.11 tests/bench_pion_metal_multiworker.py
"""
from __future__ import annotations

import socket
import sys
import threading
import time

import numpy as np

HOST = "127.0.0.1"
PORT = 1974
H, N, D = 8, 2048, 128
NUM_THREADS = 8
QUERIES_PER_THREAD = 50


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

    def query(self, sid, layer, Q):
        Hh, Dd = Q.shape
        r = self.call("ATTEND.PREFIX.QUERY", sid, str(layer), str(Hh), str(Dd), str(Hh * Dd),
                      np.ascontiguousarray(Q, dtype=np.float32).tobytes())
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
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def worker_thread(thread_id, results, errors):
    """Each thread holds one connection (so it lands on one worker via accept()),
    stores its own K/V, queries it many times, validates output."""
    try:
        c = Conn()
        rng = np.random.default_rng(0xC0FFEE + thread_id)
        K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
        V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
        Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
        cpu_ref = cpu_attention(Q, K, V)
        sid = f"mw_t{thread_id}_{int(time.time() * 1000)}"
        ok, raw = c.store_kv(sid, 0, K, V)
        if not ok:
            errors.append(f"thread {thread_id}: STORE failed: {raw[:80]!r}")
            return

        # First sanity query
        out, raw = c.query(sid, 0, Q)
        if out is None:
            errors.append(f"thread {thread_id}: QUERY failed: {raw[:80]!r}")
            return
        cos = cosine(out, cpu_ref)
        if cos < 0.999:
            errors.append(f"thread {thread_id}: cosine {cos:.7f} < 0.999")
            return

        # Hammer queries with fresh Q each iter
        Q_pool = [(rng.standard_normal((H, D)) * 0.5).astype(np.float32)
                  for _ in range(QUERIES_PER_THREAD)]
        ref_pool = [cpu_attention(Qi, K, V) for Qi in Q_pool]
        cos_min = 1.0
        t0 = time.perf_counter()
        for i in range(QUERIES_PER_THREAD):
            out, _ = c.query(sid, 0, Q_pool[i])
            if out is None:
                errors.append(f"thread {thread_id}: query #{i} failed")
                return
            cos_i = cosine(out, ref_pool[i])
            if cos_i < cos_min:
                cos_min = cos_i
        elapsed = time.perf_counter() - t0
        results.append((thread_id, cos_min, QUERIES_PER_THREAD, elapsed))
    except Exception as e:
        errors.append(f"thread {thread_id}: exception: {e!r}")


def main():
    print(f"Phase 2 multi-worker test: {NUM_THREADS} concurrent threads × {QUERIES_PER_THREAD} queries each")
    print(f"  shape: H={H} N={N} D={D}")
    print()
    try:
        c = Conn()
        c.call("PING")
        c.s.close()
    except OSError:
        print("FAIL: pion not reachable. Start with: "
              "./pion-server --metal-attention -w 4 --independent-workers")
        return 2

    results = []
    errors = []
    threads = [threading.Thread(target=worker_thread, args=(i, results, errors))
               for i in range(NUM_THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_elapsed = time.perf_counter() - t0

    if errors:
        print("FAIL: one or more threads errored:")
        for e in errors:
            print(f"  {e}")
        return 1

    print(f"  thread  min(cosine)  queries  elapsed   q/s")
    print(f"  ------  -----------  -------  -------   -----")
    total_qps = 0
    for tid, cos_min, n, elapsed in sorted(results):
        qps = n / elapsed
        total_qps += qps
        print(f"  {tid:>6d}  {cos_min:>11.7f}  {n:>7d}  {elapsed:>6.3f}s  {qps:>5.0f}")
    print()
    print(f"  total wall: {total_elapsed:.3f}s, aggregate {total_qps:.0f} q/s across {NUM_THREADS} threads")

    all_correct = all(cos_min >= 0.999 for _, cos_min, _, _ in results)
    if not all_correct:
        print("FAIL: some thread saw cosine < 0.999 — multi-worker correctness regressed")
        return 1
    print(f"\nPASS: {NUM_THREADS} threads × {QUERIES_PER_THREAD} queries, all cosine ≥ 0.999")
    return 0


if __name__ == "__main__":
    sys.exit(main())
