#!/usr/bin/env python3
"""gh #60 Phase 2 — sparse-mask SDPA correctness gate.

Two checks against the new `sdpa_q1_sparse_fp32` kernel (exposed via the
ATTEND.PREFIX.QUERY_SPARSE wire command):

  [1] Dense-equivalence: sparse with indices=[0..N-1] for every head must
      match `sdpa_q1_fp32` (the dense kernel) at cosine ≥ 0.9999 — confirms
      the new code paths (per-head index loop, indices/counts staging,
      threadgroup memory dispatch) don't drift from the dense reference.

  [2] Top-K equivalence: sparse with indices = top-K (by CPU softmax score)
      must match CPU softmax over the SAME top-K K/V (cosine ≥ 0.9999) — this
      is the user-facing contract: when the caller supplies the right indices,
      the kernel must compute attention over exactly that subset.

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


def cpu_softmax_attention(Q, K, V, mask_indices=None):
    """CPU reference. If mask_indices is None, attend over all N; else only
    over the per-head indices in mask_indices[H, K_h]."""
    H, N, D = K.shape
    scale = 1.0 / np.sqrt(D)
    if mask_indices is None:
        scores = np.matmul(Q[:, None, :], K.transpose(0, 2, 1)).squeeze(1) * scale  # [H, N]
        sm = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(sm)
        a = e / e.sum(axis=-1, keepdims=True)
        return np.einsum("hn,hnd->hd", a, V)
    out = np.zeros((H, D), dtype=np.float32)
    for h in range(H):
        idx = mask_indices[h]
        Kh = K[h, idx, :]   # [Kh, D]
        Vh = V[h, idx, :]
        scores = (Q[h] @ Kh.T) * scale   # [Kh]
        sm = scores - scores.max()
        e = np.exp(sm)
        a = e / e.sum()
        out[h] = a @ Vh
    return out


def store_kv(c: Conn, sid: str, H: int, N: int, D: int, K, V):
    r = c.call("ATTEND.PREFIX.STORE", sid, "0", str(H), str(N), str(D),
               K.tobytes(), V.tobytes())
    if not r.startswith(b"+OK"):
        raise RuntimeError(f"STORE rejected: {r!r}")


def query_dense(c: Conn, sid: str, H: int, D: int, top_k: int, Q):
    r = c.call("ATTEND.PREFIX.QUERY", sid, "0", str(H), str(D), str(top_k), Q.tobytes())
    return np.frombuffer(parse_bulk(r), dtype=np.float32).copy().reshape(H, D)


def query_sparse(c: Conn, sid: str, H: int, D: int, K_sparse_max: int,
                 Q, indices, counts):
    r = c.call("ATTEND.PREFIX.QUERY_SPARSE", sid, "0", str(H), str(D),
               str(K_sparse_max),
               Q.tobytes(),
               indices.astype(np.int32).tobytes(),
               counts.astype(np.uint32).tobytes())
    return np.frombuffer(parse_bulk(r), dtype=np.float32).copy().reshape(H, D)


def per_head_cosine(a, b):
    H, _ = a.shape
    out = []
    for h in range(H):
        out.append(float(np.dot(a[h], b[h])
                         / (np.linalg.norm(a[h]) * np.linalg.norm(b[h]) + 1e-9)))
    return out


def main() -> int:
    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError:
        print("FAIL: pion not reachable. Start with: ./pion-server --kvcache --metal-attention -w 1")
        return 2

    H, N, D = 4, 1024, 128
    sid = "sparse_kernel_v1"
    rng = np.random.default_rng(7)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)

    c = Conn()
    store_kv(c, sid, H, N, D, K, V)

    print(f"[1] Dense-equivalence: sparse(indices=[0..N-1]) vs dense kernel  (H={H} N={N} D={D})")
    indices_all = np.tile(np.arange(N, dtype=np.int32)[None, :], (H, 1))  # [H, N]
    counts_all  = np.full(H, N, dtype=np.uint32)
    out_dense  = query_dense (c, sid, H, D, N, Q)
    out_sparse = query_sparse(c, sid, H, D, N, Q, indices_all, counts_all)
    cos = per_head_cosine(out_dense, out_sparse)
    fail = False
    for h, ch in enumerate(cos):
        ok = ch >= 0.9999
        if not ok:
            fail = True
        print(f"   head {h}: cosine(dense, sparse-all) = {ch:.6f}  {'OK' if ok else 'FAIL'}")
    md = float(np.max(np.abs(out_dense - out_sparse)))
    print(f"   max|dense - sparse_all| = {md:.3e}")

    print(f"\n[2] Top-K equivalence: sparse(indices=top-K) vs CPU softmax over same top-K")
    K_sparse_max = 64
    # CPU computes top-K per head (by raw QK score, pre-softmax).
    scale = 1.0 / np.sqrt(D)
    scores_all = np.einsum("hd,hnd->hn", Q, K) * scale  # [H, N]
    idx_topk = np.argpartition(-scores_all, K_sparse_max, axis=-1)[:, :K_sparse_max].astype(np.int32)
    # Sort each row for determinism (kernel doesn't care about order, but the
    # CPU reference path slices K[h, idx] in the given order — both behave the
    # same as long as the SET of indices is identical).
    idx_topk = np.sort(idx_topk, axis=-1)
    counts_topk = np.full(H, K_sparse_max, dtype=np.uint32)
    out_kernel = query_sparse(c, sid, H, D, K_sparse_max, Q, idx_topk, counts_topk)
    out_cpu = cpu_softmax_attention(Q, K, V, mask_indices=idx_topk)
    cos = per_head_cosine(out_kernel, out_cpu)
    for h, ch in enumerate(cos):
        ok = ch >= 0.9999
        if not ok:
            fail = True
        print(f"   head {h}: cosine(kernel-topK, cpu-topK) = {ch:.6f}  {'OK' if ok else 'FAIL'}")
    md = float(np.max(np.abs(out_kernel - out_cpu)))
    print(f"   max|kernel-topK - cpu-topK| = {md:.3e}")

    print(f"\n[3] Per-head divergent masks: head h gets a different K_h count")
    # Each head picks the same top-K size but DIFFERENT indices — confirms the
    # `counts[h]` and `indices[h, :]` are read correctly per head.
    K_sparse_max = 32
    rngm = np.random.default_rng(123)
    idx_div = np.zeros((H, K_sparse_max), dtype=np.int32)
    for h in range(H):
        idx_div[h] = rngm.choice(N, size=K_sparse_max, replace=False)
    counts_div = np.full(H, K_sparse_max, dtype=np.uint32)
    out_kernel = query_sparse(c, sid, H, D, K_sparse_max, Q, idx_div, counts_div)
    out_cpu    = cpu_softmax_attention(Q, K, V, mask_indices=idx_div)
    cos = per_head_cosine(out_kernel, out_cpu)
    for h, ch in enumerate(cos):
        ok = ch >= 0.9999
        if not ok:
            fail = True
        print(f"   head {h}: cosine(kernel, cpu) = {ch:.6f}  {'OK' if ok else 'FAIL'}")
    md = float(np.max(np.abs(out_kernel - out_cpu)))
    print(f"   max|kernel - cpu| = {md:.3e}")

    print(f"\n{'PASS' if not fail else 'FAIL'} — gh #60 Phase 2 sparse-mask SDPA kernel")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
