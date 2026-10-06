#!/usr/bin/env python3
"""Sliding-window flash attention (--fa-window N) correctness gate.

Closes gh #35: when the server is started with `--fa-window W`, the Metal
SDPA kernels (sdpa_q1_fp32 + sdpa_q1_fp16, plus the batched variants) must
scan only the last W tokens of K/V — equivalent to attention computed over
K[N-W:] and V[N-W:] only.

The test launches its own server (so the window is unambiguous) and
compares:
  - Pion ATTEND.PREFIX.QUERY with --fa-window W
  - CPU reference over the same K/V truncated to the last W rows
on a (H=8, N=2048, D=128) workload.

For hybrid-attention models (Qwen3.5: every 4th layer full softmax) the
windowed layers don't need to match the full-attention output bit-for-bit;
the unrelated full-attention layer catches anything missed. We assert the
weaker, more useful invariant: the kernel exactly implements "softmax over
the last W tokens" with cosine ≥ 0.999 vs the CPU reference. On dense
transformers this is a lossy approximation of full attention; the caller
opts in by passing --fa-window > 0.

Usage:
  python3 tests/test_metal_attention_window.py            # default H=8 N=2048 D=128 W=512
  python3 tests/test_metal_attention_window.py -w 1024
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_ready_pid, wait_port_free  # noqa: E402

HOST = "127.0.0.1"
PORT = 7799  # avoid collision with the iOS PionMesh app on 1974


def _spawn_server(window: int, fp16: bool, port: int) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or ("./pion-server-dev" if os.path.exists("./pion-server-dev") else "./pion-server")
    if not os.path.exists(binary):
        print(f"FAIL: neither ./pion-server-dev nor ./pion-server exists. Run `pixi run build-dev` first.")
        sys.exit(2)
    flag = "--metal-attention-fp16" if fp16 else "--metal-attention"
    cmd = [binary, flag, "--kvcache", "-p", str(port), "-w", "1", "--no-auto-detect", "--no-auto-embed"]
    if window > 0:
        cmd += ["--fa-window", str(window)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready_pid(port, proc, 15)   # this process, not a lingering listener (#27)
    except RuntimeError:
        proc.terminate()
        print(f"FAIL: server failed to bind {HOST}:{port} within 15s")
        sys.exit(2)
    return proc


class PionAttn:
    def __init__(self, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.sock.settimeout(60)
        self.sock.connect((HOST, port))
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
        return r.startswith(b"+OK")

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


def cpu_softmax_attn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    H, N, D = K.shape
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q[:, None, :], K.transpose(0, 2, 1)) * scale  # [H,1,N]
    sm = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(sm)
    a = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(a, V).reshape(H, D)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def run_one(window: int, H: int, N: int, D: int, fp16: bool, threshold: float) -> bool:
    print(f"\n=== --fa-window {window}  H={H} N={N} D={D} fp16={fp16} ===")
    server = _spawn_server(window, fp16, PORT)
    ok = False
    try:
        rng = np.random.default_rng(0)
        Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
        K = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)
        V = (rng.standard_normal((H, N, D)) * 0.5).astype(np.float32)

        pion = PionAttn(PORT)
        sid = f"win_{int(time.time() * 1000)}"
        if not pion.store_kv(sid, 0, K, V):
            print("FAIL: store_kv failed")
            return False

        out = pion.query_cached(sid, 0, Q, N)
        if out is None:
            print("FAIL: query_cached returned None")
            return False

        eff = window if 0 < window < N else N
        K_eff = K[:, N - eff:, :]
        V_eff = V[:, N - eff:, :]
        ref = cpu_softmax_attn(Q, K_eff, V_eff)
        cos_window = cosine(out, ref)

        # Also report cosine vs full attention so the user can see how lossy
        # the window is on this dense workload — that's the "lossy-on-dense"
        # caveat called out in the issue.
        ref_full = cpu_softmax_attn(Q, K, V)
        cos_full = cosine(out, ref_full)

        print(f"  cosine vs windowed-CPU (last {eff} tokens) = {cos_window:.6f}  (gate ≥ {threshold})")
        print(f"  cosine vs full-CPU       (all   {N} tokens) = {cos_full:.6f}  (informational; lossy on dense)")

        ok = cos_window >= threshold
        print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        wait_port_free(PORT)   # the next sub-gate's server reuses the port
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("-w", "--window", type=int, default=512)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.999)
    args = ap.parse_args()

    if args.tokens <= args.window:
        print(f"NOTE: --tokens {args.tokens} ≤ --window {args.window}; raising tokens to {args.window * 2} so the window is exercised")
        args.tokens = args.window * 2

    # Three sub-gates:
    #   1. window=W (smaller than N) — kernel must agree with windowed-CPU.
    #   2. window=0 → full attention (regression check on the unwindowed path).
    #   3. window=N → equivalent to window=0 (no clamp triggered).
    results = []
    results.append(("windowed",      run_one(args.window, args.heads, args.tokens, args.head_dim, args.fp16, args.threshold)))
    results.append(("full (W=0)",    run_one(0,           args.heads, args.tokens, args.head_dim, args.fp16, args.threshold)))
    results.append(("W>=N no-clamp", run_one(args.tokens * 2, args.heads, args.tokens, args.head_dim, args.fp16, args.threshold)))

    print("\n──── --fa-window correctness gate ────")
    for name, ok in results:
        print(f"  {name:<18} {'PASS' if ok else 'FAIL'}")
    overall = all(ok for _, ok in results)
    print(f"  OVERALL            {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
