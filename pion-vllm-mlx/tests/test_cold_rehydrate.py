#!/usr/bin/env python3
"""gh #67 — PionPromptCache transparent cold-tier rehydrate gate.

Validates that when a session evicted from Pion's Metal SDPA cache is
queried again via `PionPromptCache.attend_query(...)`, the client
transparently:
  1. Receives `-COLDMISS` from the server.
  2. Issues `KV.PREFIX.WARM <ns> H D <ns>_attn` (server-side rehydrate).
  3. Retries the original ATTEND.PREFIX.QUERY.
  4. Returns the attention output to the caller — same shape, no surfaced error.

The test populates one "victim" namespace fully (V-store + Metal SDPA),
then floods the Metal SDPA cache with 256 unrelated sessions to force
the victim into the cold tier, then runs `attend_query` against the victim
and checks (a) cosine similarity vs a host-side reference is high, and
(b) `pcache.cold_rehydrates_observed == 1`.

Wire-only — `model=None` constructs a PionPromptCache that talks RESP
without needing MLX. Runs in <5 s on M-series.

Requires: ./pion-server-dev --kvcache --metal-attention -p <PORT> -w 1
The script manages the server lifecycle itself.
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

# Path massaging — pion-vllm-mlx is in a sibling dir of the test runner cwd.
HERE = Path(__file__).resolve().parent
PKG = HERE.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from pion_vllm_mlx.prompt_cache import PionPromptCache  # noqa: E402

PORT = 1981
HOST = "127.0.0.1"


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
    p = d[0:1]
    nl = d.find(b"\r\n")
    if nl < 0: return None
    if p in (b"+", b"-", b":"): return nl + 2
    if p == b"$":
        ls = d[1:nl].decode()
        if ls == "-1": return nl + 2
        n = int(ls); need = nl + 2 + n + 2
        return need if len(d) >= need else None
    return nl + 2


class Conn:
    def __init__(self, port: int = PORT):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.settimeout(30)
        self.s.connect((HOST, port))
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


def _spawn_server() -> subprocess.Popen:
    repo_root = Path(__file__).resolve().parents[2]
    binary = repo_root / "pion-server-dev"
    if not binary.exists():
        binary = repo_root / "pion-server"
    if not binary.exists():
        raise RuntimeError("no pion-server binary in repo root")
    proc = subprocess.Popen(
        [str(binary), "--kvcache", "--metal-attention", "-p", str(PORT), "-w", "1"],
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            s = socket.create_connection((HOST, PORT), timeout=1); s.close()
            return proc
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError(f"pion-server failed to start on port {PORT}")


def main() -> int:
    # Wipe any leftover state.
    repo_root = Path(__file__).resolve().parents[2]
    for fn in ("pion.vstore.0", "pion.wal.0", "pion.hnsw.0"):
        try: (repo_root / fn).unlink()
        except FileNotFoundError: pass

    proc = _spawn_server()
    try:
        c = Conn()
        fail = False

        H, D = 4, 64
        N = 16
        kv_dim = H * D
        victim_ns = f"victim_{uuid.uuid4().hex[:8]}"

        # ── 1. Populate the victim: V-store side (KV.PREFIX.REGISTER + V.STOREBATCH)
        #       + Metal SDPA side (ATTEND.PREFIX.STORE with sid = <ns>_attn).
        K = (np.random.RandomState(11).randn(H, N, D) * 0.5).astype(np.float32)
        V = (np.random.RandomState(12).randn(H, N, D) * 0.5).astype(np.float32)

        print(f"[1] Populate victim namespace '{victim_ns}' in V-store + Metal SDPA")
        r = c.call("KV.PREFIX.REGISTER", victim_ns, str(kv_dim), "fp16")
        if not r.startswith(b"+OK"):
            print(f"   REGISTER failed: {r[:60]!r}"); return 1
        # V-store layout: [N, H*D] token-major.
        K_flat = K.transpose(1, 0, 2).reshape(N, kv_dim).astype(np.float32)
        V_flat = V.transpose(1, 0, 2).reshape(N, kv_dim).astype(np.float32)
        r = c.call("V.STOREBATCH", f"{victim_ns}_pk", "0", "0", str(N), K_flat.tobytes())
        if not r.startswith(b"+OK"):
            print(f"   STOREBATCH K failed: {r[:60]!r}"); return 1
        r = c.call("V.STOREBATCH", f"{victim_ns}_pv", "0", "0", str(N), V_flat.tobytes())
        if not r.startswith(b"+OK"):
            print(f"   STOREBATCH V failed: {r[:60]!r}"); return 1
        # Metal SDPA side — production convention is `<ns>_attn` for ATTEND sids.
        attend_sid = f"{victim_ns}_attn"
        r = c.call("ATTEND.PREFIX.STORE", attend_sid, "0", str(H), str(N), str(D),
                   K.tobytes(), V.tobytes())
        if not r.startswith(b"+OK"):
            print(f"   ATTEND.PREFIX.STORE failed: {r[:60]!r}"); return 1
        print(f"   victim populated OK (attend_sid={attend_sid})")

        # ── 2. Flood the Metal SDPA cache with 256 unrelated sessions, forcing
        #       the victim to be the LRU and getting demoted.
        print(f"\n[2] Flood the SDPA cache with 256 unrelated sessions")
        for i in range(256):
            sid = f"flood_{i:03d}_{uuid.uuid4().hex[:4]}"
            rs = np.random.RandomState(2000 + i)
            Kf = (rs.randn(H, N, D) * 0.5).astype(np.float32)
            Vf = (rs.randn(H, N, D) * 0.5).astype(np.float32)
            r = c.call("ATTEND.PREFIX.STORE", sid, "0", str(H), str(N), str(D),
                       Kf.tobytes(), Vf.tobytes())
            if not r.startswith(b"+OK"):
                print(f"   flood {i} failed: {r[:40]!r}"); return 1
        # Victim should be COLD now.
        r = c.call("ATTEND.PREFIX.LOOKUP", attend_sid, "0")
        if not r.startswith(b"+COLD"):
            print(f"   FAIL: expected +COLD, got {r[:40]!r}"); return 1
        print(f"   victim is now COLD ({r[:20]!r})")

        # ── 3. Construct PionPromptCache (wire-only, no MLX model) + call attend_query.
        #       The retry wrapper should: detect -COLDMISS, issue KV.PREFIX.WARM,
        #       retry the query, return the output transparently.
        print(f"\n[3] PionPromptCache.attend_query(...) should transparently rehydrate")
        pcache = PionPromptCache(model=None, host=HOST, port=PORT, stage2=True)
        Q = (np.random.RandomState(13).randn(H, D) * 0.5).astype(np.float32)
        try:
            out = pcache.attend_query(victim_ns, layer_id=0, Q=Q, top_k=N)
        except RuntimeError as e:
            print(f"   FAIL: attend_query raised: {e}"); return 1
        if out.shape != (H, D):
            print(f"   FAIL: expected shape ({H}, {D}), got {out.shape}"); return 1

        # ── 4. Verify cold_rehydrates_observed bumped exactly once.
        observed = getattr(pcache, "cold_rehydrates_observed", 0)
        print(f"   pcache.cold_rehydrates_observed = {observed}")
        if observed != 1:
            print(f"   FAIL: expected 1 rehydrate observed, got {observed}"); fail = True

        # ── 5. Compare attention output to a host-side fp16-rounded reference.
        scale = 1.0 / np.sqrt(D)
        ref = np.zeros((H, D), dtype=np.float32)
        for h in range(H):
            Kh = K[h].astype(np.float16).astype(np.float32)
            Vh = V[h].astype(np.float16).astype(np.float32)
            scores = (Q[h] @ Kh.T) * scale
            w = np.exp(scores - scores.max()); w /= w.sum()
            ref[h] = w @ Vh
        cos = (out * ref).sum() / max(1e-12, np.linalg.norm(out) * np.linalg.norm(ref))
        print(f"   cosine(out, fp16_ref) = {cos:.4f}")
        if cos < 0.99:
            print(f"   FAIL: attention output diverged after rehydrate"); fail = True

        # ── 6. Verify second attend_query is a direct WARM hit (no further rehydrate).
        print(f"\n[4] Second attend_query — should hit WARM directly (no extra rehydrate)")
        _ = pcache.attend_query(victim_ns, layer_id=0, Q=Q, top_k=N)
        observed2 = getattr(pcache, "cold_rehydrates_observed", 0)
        print(f"   pcache.cold_rehydrates_observed after 2nd query = {observed2}")
        if observed2 != 1:
            print(f"   FAIL: expected counter unchanged (1), got {observed2}"); fail = True

        print(f"\n{'PASS' if not fail else 'FAIL'} — gh #67 client transparent rehydrate gate")
        return 1 if fail else 0
    finally:
        proc.kill()
        proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
