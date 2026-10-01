#!/usr/bin/env python3
"""KV.PREFIX LRU eviction — closes §13.3 Day 5 / §27.3 of the shared-KV-cache design.

V-store holds at most MAX_VS_SESSIONS (256) sessions. Two `KV.PREFIX.REGISTER`
calls (`<ns>_pk` + `<ns>_pv`) consume two slots, so the cache supports
128 distinct prefixes before pressure starts.

Before this fix the 17th prefix would fail with
`-ERR KV.PREFIX.REGISTER failed (no free V-store slots)`. With LRU eviction
the 17th register succeeds by evicting the least-recently-touched prefix.

What this test verifies:
  1. Filling the cache (16 prefixes) → all REGISTER + LOOKUP succeed.
  2. A fresh REGISTER under pressure succeeds (would have failed pre-fix).
  3. A prefix touched by LOOKUP, or read only through V.FETCH (the way
     serve's prefix lineages are read), survives subsequent REGISTER pressure.
  4. The least-recently-touched prefix is the one evicted.
  5. INFO reports a non-zero `vstore_evictions` counter.

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import os
import socket
import sys

MAX_VS_SESSIONS = int(os.environ.get("PION_VS_SESSIONS", "256"))   # src/network/v_store.mojo


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


class RESP:
    def __init__(self, host: str = "127.0.0.1", port: int = 1974) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(10)
        self.sock.connect((host, port))
        self.buf = b""

    def call(self, *parts) -> bytes:
        self.sock.sendall(_encode(parts))
        while True:
            done = _first_complete(self.buf)
            if done is not None:
                msg = self.buf[:done]
                self.buf = self.buf[done:]
                return msg
            chunk = self.sock.recv(64 * 1024)
            if not chunk:
                raise ConnectionError("pion closed")
            self.buf += chunk


def info_field(info_blob: bytes, key: str) -> int | None:
    body = info_blob.split(b"\r\n", 1)[1] if info_blob.startswith(b"$") else info_blob
    for line in body.splitlines():
        if line.startswith(key.encode() + b":"):
            return int(line.split(b":", 1)[1])
    return None


def main() -> int:
    r = RESP(port=int(os.environ.get("PION_PORT", "1974")))
    # MAX_VS_SESSIONS slots -> CAP distinct prefixes (each takes 2: _pk + _pv).
    # Fill CAP, touch the first half, then add 4 more under pressure.
    CAP = MAX_VS_SESSIONS // 2
    Q = CAP // 4
    KV_DIM = 512
    BASE = "lru_test_prefix_"
    row = b"\x00" * (KV_DIM * 4)       # one fp32 row, so V.FETCH has something to read

    # Phase 1 — fill to capacity; every prefix holds one row.
    for i in range(CAP):
        ns = f"{BASE}{i:03d}"
        rep = r.call("KV.PREFIX.REGISTER", ns, str(KV_DIM), "fp16")
        assert rep.startswith(b"+OK"), f"register {i} failed: {rep!r}"
        for side in ("_pk", "_pv"):
            rep = r.call("V.STOREBATCH", ns + side, "0", "0", "1", row)
            assert rep.startswith(b"+OK"), f"storebatch {i}{side} failed: {rep!r}"

    for i in range(CAP):
        rep = r.call("KV.PREFIX.LOOKUP", f"{BASE}{i:03d}")
        assert rep.startswith(b"+HIT"), f"lookup {i} miss before pressure: {rep!r}"

    # Phase 2 — touch the first quarter through LOOKUP and the second quarter
    # through V.FETCH only, so the second half is least recently used. A read
    # through V.FETCH must count as an access: serve's prefix lineages are read
    # that way alone (fetch_tokens / fetch_range_fp16_raw touch internally).
    for i in range(Q):
        rep = r.call("KV.PREFIX.LOOKUP", f"{BASE}{i:03d}")
        assert rep.startswith(b"+HIT"), f"touch {i} failed: {rep!r}"
    for i in range(Q, 2 * Q):
        for side in ("_pk", "_pv"):
            rep = r.call("V.FETCH", f"{BASE}{i:03d}{side}", "0", "RANGE", "0", "1")
            assert rep.startswith(b"$"), f"fetch-touch {i}{side} failed: {rep[:60]!r}"

    info_before = r.call("KV.PREFIX.INFO")
    evictions_before = info_field(info_before, "vstore_evictions") or 0

    # Phase 3 — register 4 NEW prefixes under pressure. Each should evict an LRU.
    for i in range(CAP, CAP + 4):
        ns = f"{BASE}{i:03d}"
        rep = r.call("KV.PREFIX.REGISTER", ns, str(KV_DIM), "fp16")
        assert rep.startswith(b"+OK"), f"register-under-pressure {i} failed: {rep!r}"
        rep = r.call("KV.PREFIX.LOOKUP", ns)
        assert rep.startswith(b"+HIT"), f"new prefix {i} not present after register: {rep!r}"

    info_after = r.call("KV.PREFIX.INFO")
    evictions_after = info_field(info_after, "vstore_evictions") or 0
    assert evictions_after - evictions_before == 8, (
        f"expected 8 evictions (4 new prefixes x 2 sids), got {evictions_after - evictions_before}"
    )

    def alive(i):
        return r.call("KV.PREFIX.LOOKUP", f"{BASE}{i:03d}").startswith(b"+HIT")

    # Phase 4 — both touched quarters survive; exactly the 4 oldest of the LRU
    # half are gone and the rest of it survives. (LOOKUP touches, so check the
    # expected-dead ones first.)
    dead = [i for i in range(2 * Q, 2 * Q + 4) if not alive(i)]
    lookup_touched = sum(alive(i) for i in range(Q))
    fetch_touched = sum(alive(i) for i in range(Q, 2 * Q))
    lru_rest = sum(alive(i) for i in range(2 * Q + 4, CAP))
    print(f"[lru] capacity {CAP} prefixes; evictions {evictions_before} -> {evictions_after} (expect +8)")
    print(f"[lru] LOOKUP-touched survivors {lookup_touched}/{Q}, V.FETCH-touched survivors {fetch_touched}/{Q}")
    print(f"[lru] oldest LRU evicted {len(dead)}/4, rest of LRU half alive {lru_rest}/{CAP - 2 * Q - 4}")
    assert lookup_touched == Q, "LRU bug — LOOKUP-touched prefixes were evicted"
    assert fetch_touched == Q, "LRU bug — a prefix read through V.FETCH was treated as unused"
    assert len(dead) == 4, f"LRU ordering bug — expected the 4 oldest evicted, got {dead}"
    assert lru_rest == CAP - 2 * Q - 4, "LRU ordering bug — a newer prefix was evicted"
    print("[lru] PASS — LRU eviction correct, reads and writes count as access, ordering preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
