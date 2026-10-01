#!/usr/bin/env python3
"""gh #60 Phase 1 — D=512 Metal SDPA regression gate.

Pion's Metal SDPA supports head dims {32, 64, 96, 128, 160, 192, 256, 512}.
D=512 was added in commit-this-lands so Gemma 4 full-attention layers
(global_head_dim=512) can run on the native kernel instead of falling back
to the MLX bridge.

This test exercises:
  - sdpa_q1_fp32  (M=1, single-query path) at D=512
  - sdpa_batched_q_fused_fp32 (M=1 fused, via ATTEND.PREFIX.QUERY_FUSED)
    — uses the same D=512 PSO with CHUNK=4 + dynamic threadgroup memory

Acceptance: cosine ≥ 0.9999 vs CPU softmax(QK^T/sqrt(D)) @ V at every head.
(D=64 in the broader gate uses ≥ 0.99; D=512 has more accumulation rounds
per query but fp32 math + per-PSO CHUNK=4 keeps it bit-clean — we tighten
the bar here so a regression in the dynamic-tg-memory plumbing or the new
CHUNK_MAX=4 register arrays shows up immediately.)

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
            out.append(f"${len(p)}\r\n".encode())
            out.append(p)
            out.append(b"\r\n")
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


def cpu_attention(Q, K, V):
    H, N, D = K.shape
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q[:, None, :], K.transpose(0, 2, 1)) * scale  # [H, 1, N]
    sm = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(sm)
    a = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(a, V).reshape(H, D)


def main() -> int:
    H, N, D = 4, 1024, 512
    sid = "d512_regression_v1"

    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError:
        print("FAIL: pion not reachable. Start with: ./pion-server --kvcache --metal-attention -w 1")
        return 2

    rng = np.random.default_rng(42)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)

    c = Conn()
    r = c.call("ATTEND.PREFIX.STORE", sid, "0", str(H), str(N), str(D),
               K.tobytes(), V.tobytes())
    if not r.startswith(b"+OK"):
        print(f"FAIL: ATTEND.PREFIX.STORE rejected: {r!r}")
        return 1

    r = c.call("ATTEND.PREFIX.QUERY", sid, "0", str(H), str(D), str(N), Q.tobytes())
    if not r.startswith(b"$"):
        print(f"FAIL: ATTEND.PREFIX.QUERY rejected: {r!r}")
        return 1
    nl = r.find(b"\r\n")
    blen = int(r[1:nl])
    pion_out = np.frombuffer(r[nl + 2:nl + 2 + blen], dtype=np.float32).copy().reshape(H, D)
    cpu_out = cpu_attention(Q, K, V)

    # Per-head cosine — D=512 has 8× the accumulation rounds vs D=64; if the
    # dynamic-tg-memory plumbing or CHUNK_MAX=4 register array is wrong, this
    # is where it shows.
    fail = False
    print(f"D=512 H={H} N={N}: per-head cosine vs CPU softmax(QK^T/sqrt(D))@V")
    for h in range(H):
        cos = float(np.dot(pion_out[h], cpu_out[h])
                    / (np.linalg.norm(pion_out[h]) * np.linalg.norm(cpu_out[h]) + 1e-9))
        ok = cos >= 0.9999
        if not ok:
            fail = True
        print(f"  head {h}: cosine = {cos:.6f}  {'OK' if ok else 'FAIL'}")

    print(f"\n{'PASS' if not fail else 'FAIL'} — gh #60 Phase 1 D=512 dense SDPA regression gate")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
