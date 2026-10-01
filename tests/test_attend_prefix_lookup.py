#!/usr/bin/env python3
"""gh #65 follow-on — ATTEND.PREFIX.LOOKUP correctness gate.

The wire command `ATTEND.PREFIX.LOOKUP <session_id> <layer_id>` returns
`+HIT` when the Metal SDPA session cache has K/V for that (sid, layer),
`+MISS` otherwise. Symmetric to KV.PREFIX.LOOKUP for V-store state.

Validates:
  [1] After ATTEND.PREFIX.STORE, LOOKUP returns +HIT for the stored layer.
  [2] LOOKUP returns +MISS for a layer that wasn't stored.
  [3] After ATTEND.PREFIX.STORE on a different sid, the original sid stays HIT.
  [4] After server restart (manual — can't simulate in this test), LOOKUP
      would return +MISS. (Tested implicitly by always-fresh sid in tests.)

This is the substrate that makes `PionPromptCache.lookup` stage2-aware
without false hits on stale V-store registrations.

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
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.settimeout(30)
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


def main() -> int:
    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError:
        print("FAIL: pion not reachable. Start with: ./pion-server --kvcache --metal-attention -w 1")
        return 2

    c = Conn()
    fail = False

    # Use a unique sid for this test run so we can verify the HIT-before-STORE
    # state ourselves (the server may have stale state from prior tests).
    import uuid
    sid_a = f"attn_lookup_test_a_{uuid.uuid4().hex[:8]}"
    sid_b = f"attn_lookup_test_b_{uuid.uuid4().hex[:8]}"

    print(f"[1] Pre-STORE: LOOKUP for never-touched sid should return MISS")
    r = c.call("ATTEND.PREFIX.LOOKUP", sid_a, "0")
    ok = r.startswith(b"+MISS")
    print(f"   sid_a layer 0: {r[:20]!r}  {'OK' if ok else 'FAIL'}")
    if not ok: fail = True

    print(f"\n[2] STORE one layer, LOOKUP should return HIT for that layer only")
    H, N, D = 4, 64, 32
    K = (np.random.randn(H, N, D) * 0.5).astype(np.float32)
    V = (np.random.randn(H, N, D) * 0.5).astype(np.float32)
    r = c.call("ATTEND.PREFIX.STORE", sid_a, "0", str(H), str(N), str(D), K.tobytes(), V.tobytes())
    if not r.startswith(b"+OK"):
        print(f"   STORE failed: {r[:40]!r}"); return 1

    r = c.call("ATTEND.PREFIX.LOOKUP", sid_a, "0")
    ok0 = r.startswith(b"+HIT")
    print(f"   sid_a layer 0 (stored):  {r[:20]!r}  {'OK' if ok0 else 'FAIL'}")
    if not ok0: fail = True

    r = c.call("ATTEND.PREFIX.LOOKUP", sid_a, "1")
    ok1 = r.startswith(b"+MISS")
    print(f"   sid_a layer 1 (unstored): {r[:20]!r}  {'OK' if ok1 else 'FAIL'}")
    if not ok1: fail = True

    print(f"\n[3] STORE different sid, original sid stays HIT")
    r = c.call("ATTEND.PREFIX.STORE", sid_b, "0", str(H), str(N), str(D), K.tobytes(), V.tobytes())
    if not r.startswith(b"+OK"):
        print(f"   STORE sid_b failed: {r[:40]!r}"); return 1
    r = c.call("ATTEND.PREFIX.LOOKUP", sid_a, "0")
    okA = r.startswith(b"+HIT")
    r = c.call("ATTEND.PREFIX.LOOKUP", sid_b, "0")
    okB = r.startswith(b"+HIT")
    print(f"   sid_a layer 0 (still HIT?): {okA}  {'OK' if okA else 'FAIL'}")
    print(f"   sid_b layer 0 (new HIT?):   {okB}  {'OK' if okB else 'FAIL'}")
    if not (okA and okB): fail = True

    print(f"\n{'PASS' if not fail else 'FAIL'} — ATTEND.PREFIX.LOOKUP correctness gate")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
