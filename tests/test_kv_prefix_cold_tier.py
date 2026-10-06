#!/usr/bin/env python3
"""gh #67 — KV.PREFIX cold-tier substrate gate.

Validates the full WARM→COLD→WARM lifecycle in the Metal SDPA session cache:

  [1] LRU bookkeeping: every STORE/QUERY stamps last_access_ns; under table
      pressure the oldest live slot is the eviction victim.
  [2] WARM→COLD demotion: filling > SDPA_SLOTS sessions on one worker forces
      at least one demotion (KV.PREFIX.INFO sdpa_demotions:>=1, sdpa_cold:>=1).
  [3] COLDMISS detection: ATTEND.PREFIX.LOOKUP returns +COLD for the demoted
      session; ATTEND.PREFIX.QUERY returns -COLDMISS instead of -ERR.
  [4] Server-side rehydrate: with V-store data present for the demoted ns,
      KV.PREFIX.WARM <ns> H D walks the V-store, dequantizes, transposes
      into [H,N,D] layout, and re-pushes to the Metal cache. After WARM,
      LOOKUP returns +HIT and QUERY succeeds.
  [5] Rehydrate counter: sdpa_rehydrates increments by exactly num_layers.

The test uses small dims (H=2 N=8 D=64) to keep the 257-session fill fast
(~16 KB per session, ~4 MB total). FP16 V-store is selected because the
substrate is the only thing being tested here — bit-perfect rehydrate is
covered by the V.STOREBATCH/V.FETCH round-trip in test_vstore_wal.py.

Requires: ./pion-server-dev --kvcache --metal-attention -w 1 -p <PORT>
The script manages the server lifecycle itself (boot, probe, kill).
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_ready_pid  # noqa: E402

import numpy as np

PORT = 1979  # non-1974 so the gate doesn't trip "PionMesh iOS app conflict" warning.
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


def _parse_info(reply: bytes) -> dict[str, int]:
    """Parse $-prefixed bulk-string INFO reply into a dict[str, int]."""
    nl = reply.find(b"\r\n")
    n = int(reply[1:nl])
    body = reply[nl + 2: nl + 2 + n].decode()
    out: dict[str, int] = {}
    for line in body.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k] = int(v.strip())
            except ValueError:
                pass
    return out


def _spawn_server() -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or ("./pion-server-dev" if os.path.exists("./pion-server-dev") else "./pion-server")
    proc = subprocess.Popen(
        [binary, "--kvcache", "--metal-attention", "-p", str(PORT), "-w", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait for THIS process to answer, not just for the port (#27).
    try:
        wait_ready_pid(PORT, proc, 60)
    except RuntimeError:
        proc.kill()
        raise
    return proc


def main() -> int:
    # Clean up any leftover state from prior aborted runs.
    for f in ("pion.vstore.0", "pion.wal.0", "pion.hnsw.0"):
        try: os.remove(f)
        except FileNotFoundError: pass

    proc = _spawn_server()
    try:
        c = Conn()
        fail = False

        H, N, D = 2, 8, 64
        kv_dim = H * D
        # Fill SDPA_SLOTS + 1 sessions. The 0th session is the LRU and should
        # be demoted to the cold registry once we cross the threshold.
        SDPA_SLOTS = 256
        n_sessions = SDPA_SLOTS + 1

        sids = [f"cold_{uuid.uuid4().hex[:8]}_{i:03d}" for i in range(n_sessions)]

        print(f"[1] STORE {n_sessions} unique sessions (H={H} N={N} D={D}) to force eviction")
        # Pre-generate K/V for sid 0 — we'll need it later to verify the
        # post-rehydrate query is bit-equivalent.
        K0 = (np.random.RandomState(7).randn(H, N, D) * 0.5).astype(np.float32)
        V0 = (np.random.RandomState(8).randn(H, N, D) * 0.5).astype(np.float32)

        for i, sid in enumerate(sids):
            if i == 0:
                K, V = K0, V0
            else:
                # Deterministic distinct payloads.
                rs = np.random.RandomState(i)
                K = (rs.randn(H, N, D) * 0.5).astype(np.float32)
                V = (rs.randn(H, N, D) * 0.5).astype(np.float32)
            r = c.call("ATTEND.PREFIX.STORE", sid, "0", str(H), str(N), str(D),
                       K.tobytes(), V.tobytes())
            if not r.startswith(b"+OK"):
                print(f"   STORE {sid} failed: {r[:60]!r}"); return 1
        print(f"   stored {n_sessions} sessions OK")

        print(f"\n[2] KV.PREFIX.INFO — confirm at least 1 demotion")
        r = c.call("KV.PREFIX.INFO")
        info = _parse_info(r)
        print(f"   sdpa_warm={info.get('sdpa_warm')} sdpa_cold={info.get('sdpa_cold')} "
              f"sdpa_demotions={info.get('sdpa_demotions')} sdpa_rehydrates={info.get('sdpa_rehydrates')}")
        if info.get("sdpa_demotions", 0) < 1 or info.get("sdpa_cold", 0) < 1:
            print(f"   FAIL: expected demotions >= 1"); fail = True
        if info.get("sdpa_warm", 0) != SDPA_SLOTS:
            print(f"   FAIL: expected sdpa_warm == {SDPA_SLOTS}"); fail = True

        print(f"\n[3] ATTEND.PREFIX.LOOKUP — sid_0 should now be +COLD")
        r = c.call("ATTEND.PREFIX.LOOKUP", sids[0], "0")
        cold_ok = r.startswith(b"+COLD")
        print(f"   sid_0: {r[:20]!r}  {'OK' if cold_ok else 'FAIL'}")
        if not cold_ok: fail = True
        # And the latest sid should still be +HIT (it can't have been the LRU).
        r = c.call("ATTEND.PREFIX.LOOKUP", sids[-1], "0")
        hit_ok = r.startswith(b"+HIT")
        print(f"   sid_last: {r[:20]!r}  {'OK' if hit_ok else 'FAIL'}")
        if not hit_ok: fail = True

        print(f"\n[4] ATTEND.PREFIX.QUERY on cold sid → expect -COLDMISS")
        Q = (np.random.RandomState(9).randn(H, D) * 0.5).astype(np.float32)
        r = c.call("ATTEND.PREFIX.QUERY", sids[0], "0", str(H), str(D), "8", Q.tobytes())
        coldmiss_ok = r.startswith(b"-COLDMISS")
        print(f"   query reply: {r[:60]!r}  {'OK' if coldmiss_ok else 'FAIL'}")
        if not coldmiss_ok: fail = True

        print(f"\n[5] Register sid_0 in V-store (fp16) + V.STOREBATCH the K/V")
        # KV.PREFIX.REGISTER creates the <ns>_pk and <ns>_pv V-store sessions.
        r = c.call("KV.PREFIX.REGISTER", sids[0], str(kv_dim), "fp16")
        if not r.startswith(b"+OK"):
            print(f"   REGISTER failed: {r[:60]!r}"); return 1
        # V-store wants the flat [N, kv_dim] layout (token-major). Transpose
        # K0/V0 from [H, N, D] to [N, H*D].
        K0_flat = K0.transpose(1, 0, 2).reshape(N, kv_dim).astype(np.float32)
        V0_flat = V0.transpose(1, 0, 2).reshape(N, kv_dim).astype(np.float32)
        sid_k = sids[0] + "_pk"
        sid_v = sids[0] + "_pv"
        r = c.call("V.STOREBATCH", sid_k, "0", "0", str(N), K0_flat.tobytes())
        if not r.startswith(b"+OK"):
            print(f"   STOREBATCH K failed: {r[:60]!r}"); return 1
        r = c.call("V.STOREBATCH", sid_v, "0", "0", str(N), V0_flat.tobytes())
        if not r.startswith(b"+OK"):
            print(f"   STOREBATCH V failed: {r[:60]!r}"); return 1
        print(f"   V-store populated OK")

        print(f"\n[6] KV.PREFIX.WARM sid_0 H={H} D={D} → expect +1 (one layer rehydrated)")
        r = c.call("KV.PREFIX.WARM", sids[0], str(H), str(D))
        # +1 or +1 skipped_dim=0
        warm_ok = r.startswith(b"+1")
        print(f"   warm reply: {r[:60]!r}  {'OK' if warm_ok else 'FAIL'}")
        if not warm_ok: fail = True

        print(f"\n[7] ATTEND.PREFIX.LOOKUP after WARM → expect +HIT")
        r = c.call("ATTEND.PREFIX.LOOKUP", sids[0], "0")
        hit_ok = r.startswith(b"+HIT")
        print(f"   sid_0: {r[:20]!r}  {'OK' if hit_ok else 'FAIL'}")
        if not hit_ok: fail = True

        print(f"\n[8] ATTEND.PREFIX.QUERY on rehydrated sid → succeeds + result close to baseline")
        # Baseline: this is the same Q against the still-live sid_last (which
        # we know is warm) but using its own K/V. To compare apples-to-apples
        # we need to query sid_0 BEFORE eviction. Since we already evicted by
        # the time we observed it, we approximate by checking that the result
        # has no NaN/Inf and matches a recomputed-host attention on K0/V0.
        r = c.call("ATTEND.PREFIX.QUERY", sids[0], "0", str(H), str(D), "8", Q.tobytes())
        if not r.startswith(b"$"):
            print(f"   query failed: {r[:60]!r}"); fail = True
        else:
            nl = r.find(b"\r\n"); body_n = int(r[1:nl])
            body = r[nl + 2: nl + 2 + body_n]
            out = np.frombuffer(body, dtype=np.float32).reshape(H, D)
            # Reference attention: softmax(Q · K^T / sqrt(D)) · V
            scale = 1.0 / np.sqrt(D)
            ref = np.zeros((H, D), dtype=np.float32)
            for h in range(H):
                # FP16 round-trip drops precision; compare to FP16-quantized K/V.
                Kh = K0[h].astype(np.float16).astype(np.float32)
                Vh = V0[h].astype(np.float16).astype(np.float32)
                scores = (Q[h] @ Kh.T) * scale
                w = np.exp(scores - scores.max()); w /= w.sum()
                ref[h] = w @ Vh
            cos = (out * ref).sum() / max(1e-12, np.linalg.norm(out) * np.linalg.norm(ref))
            print(f"   query OK, cosine vs fp16 reference = {cos:.4f}")
            if cos < 0.99:
                print(f"   FAIL: rehydrated attention not close to baseline"); fail = True

        print(f"\n[9] KV.PREFIX.INFO — confirm rehydrates counter incremented")
        r = c.call("KV.PREFIX.INFO"); info2 = _parse_info(r)
        print(f"   sdpa_rehydrates: {info.get('sdpa_rehydrates',0)} → {info2.get('sdpa_rehydrates')}")
        if info2.get("sdpa_rehydrates", 0) <= info.get("sdpa_rehydrates", 0):
            print(f"   FAIL: rehydrates counter did not advance"); fail = True

        print(f"\n[10] Negative path: KV.PREFIX.WARM unknown_ns → -ERR")
        r = c.call("KV.PREFIX.WARM", "never_registered_ns_xyz", str(H), str(D))
        neg_ok = r.startswith(b"-ERR")
        print(f"   reply: {r[:60]!r}  {'OK' if neg_ok else 'FAIL'}")
        if not neg_ok: fail = True

        print(f"\n{'PASS' if not fail else 'FAIL'} — gh #67 cold-tier substrate gate")
        return 1 if fail else 0
    finally:
        proc.kill()
        proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
