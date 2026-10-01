"""gh #49: parity test for ATTEND.PREFIX.QUERY_FUSED.

Two checks:

  1. S_suf=0 + identity head_map (H_q==H_kv) → fused kernel must produce
     the same output as legacy QUERY (prefix-only attention).

  2. Reference attention over the concatenation [prefix_K | suffix_K]
     with manual softmax + GQA expansion must match the fused output to
     cosine ≥ 0.99999 (fp32) on a non-trivial (H_q≠H_kv, M>1, S_suf>1) case.

Server: ./pion-server-dev --kvcache --metal-attention -w 1 --no-auto-detect --no-auto-embed
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import time

import numpy as np


PION_HOST = os.environ.get("PION_HOST", "127.0.0.1")
PION_PORT = int(os.environ.get("PION_PORT", "1974"))


class RESPClient:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port))
        self.f = self.sock.makefile("rb")

    def call(self, *args) -> bytes:
        parts = [b"*" + str(len(args)).encode() + b"\r\n"]
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            elif not isinstance(a, (bytes, bytearray)):
                raise TypeError(f"unsupported arg type: {type(a)}")
            parts.append(b"$" + str(len(a)).encode() + b"\r\n")
            parts.append(bytes(a))
            parts.append(b"\r\n")
        self.sock.sendall(b"".join(parts))
        return self._read_reply()

    def _read_reply(self) -> bytes:
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed")
        first = line[:1]
        if first == b"+" or first == b"-" or first == b":":
            return line.rstrip(b"\r\n")
        if first == b"$":
            n = int(line[1:].rstrip(b"\r\n"))
            if n < 0:
                return b"$-1"
            body = self.f.read(n + 2)  # body + CRLF
            return b"$" + str(n).encode() + b"\r\n" + body[:-2]
        raise NotImplementedError(f"unsupported reply prefix: {first!r}")

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def parse_bulk(reply: bytes) -> bytes:
    if not reply.startswith(b"$"):
        raise RuntimeError(f"not a bulk reply: {reply[:160]!r}")
    nl = reply.find(b"\r\n")
    blen = int(reply[1:nl])
    if blen < 0:
        raise RuntimeError(f"nil reply: {reply[:80]!r}")
    body = reply[nl + 2:nl + 2 + blen]
    return body


def reference_sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   head_map: np.ndarray, S_suf: int, M: int) -> np.ndarray:
    """Reference fp32 attention over (prefix ∪ suffix) for parity check.
    K/V are pre-concatenated [prefix_K | suffix_K] of shape (H_kv, N+S_suf, D).
    Causal-within-suffix mask (suffix queries cannot peek at later suffix tokens)."""
    H_q, M_, D = Q.shape
    assert M_ == M
    H_kv = K.shape[0]
    NS = K.shape[1]
    N_pref = NS - S_suf
    scale = 1.0 / np.sqrt(D)
    out = np.zeros((H_q, M, D), dtype=np.float32)
    for hq in range(H_q):
        hkv = head_map[hq]
        Kh = K[hkv]                # (NS, D)
        Vh = V[hkv]                # (NS, D)
        for mq in range(M):
            scores = (Q[hq, mq] @ Kh.T) * scale  # (NS,)
            # Apply causal-within-suffix: suffix index t > suf_off + mq is masked.
            if S_suf > 0:
                suf_off = max(0, S_suf - M)
                causal_lo = suf_off + mq
                # Mask suffix positions beyond causal_lo (suffix indices: N_pref..NS-1)
                for t in range(N_pref + causal_lo + 1, NS):
                    scores[t] = -1e30
            m = scores.max()
            ex = np.exp(scores - m)
            w = ex / ex.sum()
            out[hq, mq] = w @ Vh
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    rng = np.random.default_rng(0xC0FFEE)
    # Llama-3.2-1B-style head config but tiny for fast test.
    H_kv, H_q, D = 4, 16, 64
    N_pref = 64
    S_suf = 8
    M = 4

    sid = "gh49_parity_test"
    layer_id = 0

    K_pref = rng.standard_normal((H_kv, N_pref, D)).astype(np.float32) * 0.1
    V_pref = rng.standard_normal((H_kv, N_pref, D)).astype(np.float32) * 0.1
    K_suf  = rng.standard_normal((H_kv, S_suf,  D)).astype(np.float32) * 0.1
    V_suf  = rng.standard_normal((H_kv, S_suf,  D)).astype(np.float32) * 0.1
    Q      = rng.standard_normal((H_q,  M,      D)).astype(np.float32) * 0.1

    head_map = np.repeat(np.arange(H_kv, dtype=np.uint8), H_q // H_kv)  # (H_q,)

    c = RESPClient(PION_HOST, PION_PORT)
    try:
        # ── 1. Store prefix ──
        rep = c.call(
            "ATTEND.PREFIX.STORE", sid, str(layer_id),
            str(H_kv), str(N_pref), str(D),
            K_pref.tobytes(), V_pref.tobytes(),
        )
        if not rep.startswith(b"+OK"):
            print(f"FAIL: STORE returned {rep[:160]!r}")
            return 1

        # ── 2. S_suf=0 parity: fused with no suffix == legacy QUERY ──
        # Legacy QUERY needs Q in shape (H_kv, rep*M, D) per the existing GQA repack.
        rep_factor = H_q // H_kv
        Q_legacy = Q.reshape(H_kv, rep_factor, M, D).reshape(H_kv, rep_factor * M, D)
        legacy_rep = c.call(
            "ATTEND.PREFIX.QUERY", sid, str(layer_id),
            str(H_kv), str(D), "256",
            Q_legacy.tobytes(),
        )
        body = parse_bulk(legacy_rep)
        out_bytes_legacy = H_kv * rep_factor * M * D * 4
        legacy_out = np.frombuffer(body[:out_bytes_legacy], dtype=np.float32).reshape(
            H_kv, rep_factor * M, D
        ).reshape(H_kv, rep_factor, M, D).reshape(H_q, M, D)

        # Fused with empty suffix should match legacy after GQA reshape.
        fused_rep = c.call(
            "ATTEND.PREFIX.QUERY_FUSED", sid, str(layer_id),
            str(H_q), str(D), "0", str(H_kv),
            Q.tobytes(), b"", b"", head_map.tobytes(),
        )
        body = parse_bulk(fused_rep)
        if len(body) != H_q * M * D * 4:
            print(f"FAIL: fused S_suf=0 reply size {len(body)}, expected {H_q*M*D*4}")
            return 1
        fused_out_0 = np.frombuffer(body, dtype=np.float32).reshape(H_q, M, D)

        cos1 = cosine(legacy_out, fused_out_0)
        max_abs1 = float(np.abs(legacy_out - fused_out_0).max())
        print(f"S_suf=0  cosine(legacy, fused) = {cos1:.10f}  max|Δ| = {max_abs1:.4e}")
        if cos1 < 0.99999:
            print("FAIL: S_suf=0 parity below 0.99999")
            return 1

        # ── 3. Full fused vs reference (host-side fp64-ish reference) ──
        fused_full_rep = c.call(
            "ATTEND.PREFIX.QUERY_FUSED", sid, str(layer_id),
            str(H_q), str(D), str(S_suf), str(H_kv),
            Q.tobytes(), K_suf.tobytes(), V_suf.tobytes(), head_map.tobytes(),
        )
        body = parse_bulk(fused_full_rep)
        fused_full = np.frombuffer(body, dtype=np.float32).reshape(H_q, M, D)

        K_concat = np.concatenate([K_pref, K_suf], axis=1)  # (H_kv, N+S_suf, D)
        V_concat = np.concatenate([V_pref, V_suf], axis=1)
        ref_out = reference_sdpa(Q, K_concat, V_concat, head_map, S_suf, M)

        cos2 = cosine(ref_out, fused_full)
        max_abs2 = float(np.abs(ref_out - fused_full).max())
        print(f"S_suf={S_suf}  cosine(ref, fused) = {cos2:.10f}  max|Δ| = {max_abs2:.4e}")
        if cos2 < 0.99999:
            print("FAIL: full fused parity below 0.99999")
            return 1

        print("\nPASS — gh #49 fused kernel matches reference.")
        return 0
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(main())
