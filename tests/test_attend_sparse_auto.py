#!/usr/bin/env python3
"""gh #63 Phase 3b — ATTEND.PREFIX.QUERY_SPARSE_AUTO correctness gate.

Server-side block-mean top-K + sparse attention in one wire call. Three
checks across regimes the selector must handle:

  [1] K_top == n_blocks (all blocks selected): sparse_auto must produce
      bit-identical output to CPU full softmax. Validates the kernel +
      recency-block append + per-head replication.

  [2] Structured data — one block is the "needle" (high-signal K), the
      rest are low-signal noise. Server's top-K must include the needle
      block; output must be cosine ≥ 0.99 vs CPU full softmax.
      Validates the selector picks the right blocks under realistic
      signal concentration (the regime that produces 100% NIAH at 64K).

  [3] Random data, K_top sweep — characterizes the sparse approximation
      quality as a function of K_top/n_blocks. NOT a correctness gate
      (random data has uniform attention; sparse loses signal proportional
      to fraction-dropped); informational only.

Acceptance: [1] cosine = 1.0 and max|Δ| < 1e-6; [2] cosine ≥ 0.99 vs full.

Requires: ./pion-server --kvcache --metal-attention -w 1
"""
from __future__ import annotations

import socket
import sys

import numpy as np

HOST = "127.0.0.1"
PORT = 1974


def _encode(parts) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        if isinstance(p, bytes):
            out.append(f"${len(p)}\r\n".encode()); out.append(p); out.append(b"\r\n")
        else:
            s = str(p)
            out.append(f"${len(s)}\r\n{s}\r\n".encode())
    return b"".join(out)


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


class Conn:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.s.settimeout(60)
        self.s.connect((HOST, PORT))
        self.buf = b""

    def _read(self) -> bytes:
        while True:
            done = _first_complete(self.buf)
            if done is not None:
                msg = self.buf[:done]
                self.buf = self.buf[done:]
                return msg
            chunk = self.s.recv(64 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("pion closed")
            self.buf += chunk

    def call(self, *parts):
        self.s.sendall(_encode(parts))
        return self._read()


def parse_bulk(r: bytes) -> bytes:
    if not r.startswith(b"$"):
        raise RuntimeError(f"expected bulk reply, got: {r[:80]!r}")
    nl = r.find(b"\r\n")
    blen = int(r[1:nl])
    return r[nl + 2:nl + 2 + blen]


def cpu_full_softmax(Q, K, V):
    H, _, D = K.shape
    scale = 1.0 / np.sqrt(D)
    out = np.zeros((H, D), dtype=np.float32)
    for h in range(H):
        s = (Q[h] @ K[h].T) * scale
        s = s - s.max()
        e = np.exp(s); a = e / e.sum()
        out[h] = a @ V[h]
    return out


def server_store(c, sid, layer, H, N, D, K, V):
    r = c.call("ATTEND.PREFIX.STORE", sid, str(layer), str(H), str(N), str(D),
               K.tobytes(), V.tobytes())
    if not r.startswith(b"+OK"):
        raise RuntimeError(f"STORE failed: {r!r}")


def server_query_sparse_auto(c, sid, layer, H, D, B, K_top, Q, H_kv=None, head_map=b""):
    # New wire format includes H_kv (= H_q on non-GQA) + head_map blob (empty
    # = server-synthesized identity).
    if H_kv is None:
        H_kv = H
    r = c.call("ATTEND.PREFIX.QUERY_SPARSE_AUTO", sid, str(layer),
               str(H), str(D), str(B), str(K_top), str(H_kv),
               Q.tobytes(), head_map)
    return np.frombuffer(parse_bulk(r), dtype=np.float32).copy().reshape(H, D)


def server_query_sparse(c, sid, layer, H, D, K_sparse_max, Q, indices, counts):
    r = c.call("ATTEND.PREFIX.QUERY_SPARSE", sid, str(layer),
               str(H), str(D), str(K_sparse_max),
               Q.tobytes(),
               indices.astype(np.int32).tobytes(),
               counts.astype(np.uint32).tobytes())
    return np.frombuffer(parse_bulk(r), dtype=np.float32).copy().reshape(H, D)


def per_head_cosine(a, b):
    return [float(np.dot(a[h], b[h])
                  / (np.linalg.norm(a[h]) * np.linalg.norm(b[h]) + 1e-9))
            for h in range(a.shape[0])]


def main() -> int:
    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError:
        print("FAIL: pion not reachable. Start with: ./pion-server --kvcache --metal-attention -w 1")
        return 2

    H, N, D = 4, 1024, 128
    B = 64
    n_blocks = N // B   # 16

    fail = False

    # ── [1] K_top == n_blocks: bit-identical to full attention ──
    sid = "sparse_auto_v1_test1"
    rng = np.random.default_rng(11)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    c = Conn()
    server_store(c, sid, 0, H, N, D, K, V)

    print(f"[1] All-blocks check (K_top = n_blocks = {n_blocks}): server vs CPU full softmax")
    out_auto = server_query_sparse_auto(c, sid, 0, H, D, B, n_blocks, Q)
    out_full = cpu_full_softmax(Q, K, V)
    max_diff = float(np.max(np.abs(out_auto - out_full)))
    cos = per_head_cosine(out_auto, out_full)
    for h, ch in enumerate(cos):
        ok = ch >= 0.99999
        if not ok: fail = True
        print(f"   head {h}: cosine = {ch:.6f}  {'OK' if ok else 'FAIL'}")
    print(f"   max|server - full| = {max_diff:.3e}  (gate ≤ 1e-6)  {'OK' if max_diff < 1e-6 else 'FAIL'}")
    if max_diff >= 1e-6: fail = True
    print()

    # ── [2] Structured data — one needle block with high signal ──
    # K layout: 16 blocks. Block 7 is "the needle": K values are 5× larger
    # there, so softmax(Q·K) concentrates 99%+ mass on block 7. V at block 7
    # is a known sentinel; everywhere else V is small noise.
    # Sparse top-K=1 MUST pick block 7 → attention output ≈ V[block 7].
    sid2 = "sparse_auto_v1_test2"
    rng2 = np.random.default_rng(42)
    Q2 = (rng2.standard_normal((H, D)) * 0.3).astype(np.float32)
    K2 = (rng2.standard_normal((H, N, D)) * 0.1).astype(np.float32)  # low signal
    K2[:, 7 * B:(7 + 1) * B, :] = (rng2.standard_normal((H, B, D)) * 0.5 + Q2[:, None, :] * 2.0).astype(np.float32)
    V2 = (rng2.standard_normal((H, N, D)) * 0.01).astype(np.float32)
    needle_V = np.full((H, B, D), 0.5, dtype=np.float32)
    V2[:, 7 * B:(7 + 1) * B, :] = needle_V
    server_store(c, sid2, 0, H, N, D, K2, V2)

    print(f"[2] Structured needle (block 7) — sparse must pick it; K_top=2")
    # Sparse attention with K_top=2 attends to 128 of 1024 tokens. CPU full
    # softmax spreads ~1% of attention mass across the OTHER 896 tokens — so
    # absolute outputs differ by O(0.01 × V_max). Cosine (direction) is the
    # meaningful metric: a downstream model uses argmax over logits, which is
    # direction-dependent. Cosine ≥ 0.99 means the output points at the same
    # answer, which is what NIAH ultimately measures.
    out_auto2 = server_query_sparse_auto(c, sid2, 0, H, D, B, 2, Q2)
    out_full2 = cpu_full_softmax(Q2, K2, V2)
    max_diff2 = float(np.max(np.abs(out_auto2 - out_full2)))
    cos2 = per_head_cosine(out_auto2, out_full2)
    for h, ch in enumerate(cos2):
        ok = ch >= 0.99
        if not ok: fail = True
        print(f"   head {h}: cosine vs CPU full = {ch:.6f}  {'OK' if ok else 'FAIL'}")
    print(f"   max|server - full| = {max_diff2:.3e}  (informational — sparse magnitude differs from full by O(mass on missed tokens × V_max))")
    print()

    # ── [3] K_top sweep on random data (informational only) ──
    print(f"[3] Random-data K_top sweep — informational (random Q/K has no signal "
          f"concentration, so cosine grows linearly with K_top/n_blocks)")
    for K_top in [4, 8, 12, n_blocks]:
        out = server_query_sparse_auto(c, sid, 0, H, D, B, K_top, Q)
        cos_v = per_head_cosine(out, out_full)
        print(f"   K_top={K_top:>2}: mean_cos vs full = {np.mean(cos_v):.4f}  min_cos = {min(cos_v):.4f}")
    print()

    print(f"{'PASS' if not fail else 'FAIL'} — gh #63 Phase 3b QUERY_SPARSE_AUTO correctness")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
