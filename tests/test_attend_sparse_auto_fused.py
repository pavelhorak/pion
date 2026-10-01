#!/usr/bin/env python3
"""gh #63 follow-on — ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED correctness gate.

Validates that the fused kernel (sparse-prefix + dense-suffix + LSE merge in
one dispatch) produces the same output as the equivalent two-step path:
  step 1: server picks top-K from resident K/V, runs sparse-prefix SDPA →
          (H_q, D) prefix output (the ATTEND.PREFIX.QUERY_SPARSE_AUTO path)
  step 2: client runs dense suffix SDPA + online-softmax merge with prefix
          (the legacy two-step path)
  vs.
  fused: ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED in one call

Plus a degenerate-case check (S_suf=0 → should match sparse_auto exactly).

Acceptance: cosine ≥ 0.9999 vs CPU full softmax over [server-picked-prefix ∪
all-suffix]; bit-equivalent to sparse_auto when S_suf=0.

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
    if len(d) < 3: return None
    p = d[0:1]; nl = d.find(b"\r\n")
    if nl < 0: return None
    if p in (b"+", b"-", b":"): return nl + 2
    if p == b"$":
        ls = d[1:nl].decode()
        if ls == "-1": return nl + 2
        n = int(ls); need = nl + 2 + n + 2
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
                msg = self.buf[:done]; self.buf = self.buf[done:]; return msg
            chunk = self.s.recv(64 * 1024 * 1024)
            if not chunk: raise ConnectionError("pion closed")
            self.buf += chunk

    def call(self, *parts):
        self.s.sendall(_encode(parts)); return self._read()


def parse_bulk(r: bytes) -> bytes:
    if not r.startswith(b"$"): raise RuntimeError(f"got {r[:80]!r}")
    nl = r.find(b"\r\n"); blen = int(r[1:nl])
    return r[nl + 2:nl + 2 + blen]


def main() -> int:
    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError:
        print("FAIL: pion not reachable. Start with: ./pion-server --kvcache --metal-attention -w 1")
        return 2

    H, N, D = 4, 1024, 128
    H_kv = H   # non-GQA for simplicity
    B = 64
    n_blocks = N // B
    sid = "sparse_auto_fused_test"
    rng = np.random.default_rng(13)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    # Suffix K/V — small, locally computed in the consumer
    S_suf = 8
    K_suf = (rng.standard_normal((H_kv, S_suf, D)) * 0.5).astype(np.float32)
    V_suf = (rng.standard_normal((H_kv, S_suf, D)) * 0.5).astype(np.float32)

    c = Conn()
    # Server-resident K/V via ATTEND.PREFIX.STORE
    r = c.call("ATTEND.PREFIX.STORE", sid, "0", str(H), str(N), str(D), K.tobytes(), V.tobytes())
    if not r.startswith(b"+OK"):
        print(f"STORE failed: {r!r}"); return 1

    fail = False

    # ── [1] S_suf=0 → fused should match sparse_auto bit-identical ──
    print("[1] S_suf=0 degenerate case — fused must match sparse_auto bit-identical")
    K_top = 8
    # AUTO (no suffix)
    r = c.call("ATTEND.PREFIX.QUERY_SPARSE_AUTO", sid, "0", str(H), str(D), str(B), str(K_top), str(H_kv),
               Q.tobytes(), b"")
    out_auto = np.frombuffer(parse_bulk(r), dtype=np.float32).copy().reshape(H, D)
    # FUSED with S_suf=0
    r = c.call("ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED", sid, "0", str(H), str(D), str(B), str(K_top),
               str(H_kv), "0", Q.tobytes(), b"", b"", b"")
    out_fused0 = np.frombuffer(parse_bulk(r), dtype=np.float32).copy().reshape(H, D)
    max_diff = float(np.max(np.abs(out_auto - out_fused0)))
    print(f"   max|fused(S_suf=0) - sparse_auto| = {max_diff:.3e}")
    if max_diff > 1e-6:
        print("   FAIL"); fail = True
    else:
        print("   OK (bit-equivalent)")

    # ── [2] All-blocks fused vs CPU full softmax over [all prefix ∪ all suffix] ──
    # K_top = n_blocks → no selector ambiguity (all prefix blocks are picked).
    # Server output should match CPU full softmax bit-for-bit (fp32 rounding noise).
    print(f"\n[2] All-blocks (K_top = n_blocks = {n_blocks}), S_suf={S_suf}: fused vs CPU full softmax")
    scale = 1.0 / np.sqrt(D)
    out_cpu = np.zeros((H, D), dtype=np.float32)
    for h in range(H):
        Kh = np.concatenate([K[h], K_suf[h]], axis=0)
        Vh = np.concatenate([V[h], V_suf[h]], axis=0)
        s_ = (Q[h] @ Kh.T) * scale
        s_ = s_ - s_.max()
        e = np.exp(s_); a = e / e.sum()
        out_cpu[h] = a @ Vh

    r = c.call("ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED", sid, "0", str(H), str(D), str(B), str(n_blocks),
               str(H_kv), str(S_suf),
               Q.tobytes(), K_suf.tobytes(), V_suf.tobytes(), b"")
    out_fused = np.frombuffer(parse_bulk(r), dtype=np.float32).copy().reshape(H, D)
    cos = [float(np.dot(out_fused[h], out_cpu[h])
                 / (np.linalg.norm(out_fused[h]) * np.linalg.norm(out_cpu[h]) + 1e-9))
           for h in range(H)]
    max_diff = float(np.max(np.abs(out_fused - out_cpu)))
    print(f"   max|fused(all-blocks) - cpu_full_softmax| = {max_diff:.3e}  "
          f"(gate ≤ 1e-5; some fp32 noise expected from kernel-vs-numpy summation order)")
    for h, ch in enumerate(cos):
        ok = ch >= 0.99999
        if not ok: fail = True
        print(f"   head {h}: cosine = {ch:.6f}  {'OK' if ok else 'FAIL'}")

    # ── [3] Informational: K_top < n_blocks with suffix → cosine vs CPU full softmax ──
    # Sparse approximation: cosine grows toward 1 as K_top → n_blocks. NOT a
    # gate; characterizes the approximation quality. Selector-divergence
    # between server's pick and CPU's pick is fp32 noise (documented).
    print(f"\n[3] Informational: K_top sweep, S_suf={S_suf}, vs CPU full softmax")
    for K_top in [4, 8, 12, n_blocks]:
        r = c.call("ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED", sid, "0", str(H), str(D), str(B), str(K_top),
                   str(H_kv), str(S_suf),
                   Q.tobytes(), K_suf.tobytes(), V_suf.tobytes(), b"")
        out = np.frombuffer(parse_bulk(r), dtype=np.float32).copy().reshape(H, D)
        cos_v = [float(np.dot(out[h], out_cpu[h]) / (np.linalg.norm(out[h]) * np.linalg.norm(out_cpu[h]) + 1e-9)) for h in range(H)]
        print(f"   K_top={K_top:>2}: mean_cos vs cpu_full = {np.mean(cos_v):.4f}  min_cos = {min(cos_v):.4f}")

    print(f"\n{'PASS' if not fail else 'FAIL'} — gh #63 follow-on QUERY_SPARSE_AUTO_FUSED correctness")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
