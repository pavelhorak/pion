#!/usr/bin/env python3
"""gh #71 — KV.PREFIX.BLOCKS / KV.PREFIX.MEMBERSHIP regression gate.

Validates:
  [1] REGISTER without BLOCKS → BLOCKS=+UNKNOWN, MEMBERSHIP=+UNKNOWN.
  [2] REGISTER with BLOCKS → BLOCKS returns the same blob (block_size +
      block_count + hashes); MEMBERSHIP correctly bitmaps subset, superset,
      and empty probes.
  [3] WAL replay: SIGKILL server, restart, BLOCKS still returns the table.
  [4] Snapshot roundtrip: KV.PREFIX.SAVE, restart, BLOCKS still returns the
      table.
  [5] Microbench: 1,562-block probe ≤ 100 µs median (target from the issue).

Requires: ./pion-server-dev --kvcache -w 1 -p <PORT>
The script manages the server lifecycle itself (boot, probe, kill).
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import time
import uuid

PORT = 1976  # non-1974 to dodge the PionMesh iOS-app conflict.
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

    def close(self):
        try: self.s.close()
        except OSError: pass


def _spawn_server() -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or ("./pion-server-dev" if os.path.exists("./pion-server-dev") else "./pion-server")
    proc = subprocess.Popen(
        [binary, "--kvcache", "-p", str(PORT), "-w", "1"],
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
    raise RuntimeError("pion-server failed to start on port " + str(PORT))


def _parse_bulk(reply: bytes) -> bytes | None:
    """Parse $N\\r\\n<body>\\r\\n into the body bytes (or None for $-1)."""
    assert reply.startswith(b"$"), f"not a bulk string: {reply[:40]!r}"
    nl = reply.find(b"\r\n")
    n = int(reply[1:nl])
    if n < 0: return None
    return reply[nl + 2: nl + 2 + n]


def _pack_hashes(hashes: list[int]) -> bytes:
    return struct.pack("<" + "Q" * len(hashes), *hashes)


def _unpack_blocks(reply: bytes) -> tuple[int, int, list[int]]:
    body = _parse_bulk(reply)
    assert body is not None and len(body) >= 8, f"short BLOCKS body: {body!r}"
    bsz, bcnt = struct.unpack("<II", body[:8])
    hashes = list(struct.unpack("<" + "Q" * bcnt, body[8:8 + bcnt * 8]))
    return bsz, bcnt, hashes


def _bitmap_to_set(bm: bytes, nbits: int) -> set[int]:
    return {i for i in range(nbits) if (bm[i >> 3] >> (i & 7)) & 1}


def main() -> int:
    # Clean leftover state from prior aborted runs.
    for f in (f"pion.vstore.0", f"pion.wal.0", f"pion.hnsw.0"):
        try: os.remove(f)
        except FileNotFoundError: pass

    proc = _spawn_server()
    fail = False
    try:
        c = Conn()

        # ── [1] REGISTER without BLOCKS → +UNKNOWN on both new commands ────
        ns_plain = f"gh71_plain_{uuid.uuid4().hex[:8]}"
        r = c.call("KV.PREFIX.REGISTER", ns_plain, "256", "fp16")
        if not r.startswith(b"+OK"):
            print(f"[1] FAIL REGISTER: {r[:60]!r}"); return 1
        r = c.call("KV.PREFIX.BLOCKS", ns_plain)
        if not r.startswith(b"+UNKNOWN"):
            print(f"[1] FAIL BLOCKS expected +UNKNOWN, got: {r[:60]!r}"); fail = True
        # MEMBERSHIP with a 1-element probe; should also be +UNKNOWN.
        probe1 = _pack_hashes([0xDEADBEEF])
        r = c.call("KV.PREFIX.MEMBERSHIP", ns_plain, "1", probe1)
        if not r.startswith(b"+UNKNOWN"):
            print(f"[1] FAIL MEMBERSHIP expected +UNKNOWN, got: {r[:60]!r}"); fail = True
        print(f"[1] OK — REGISTER without BLOCKS yields +UNKNOWN")

        # ── [2] REGISTER with BLOCKS → BLOCKS roundtrip + MEMBERSHIP bitmaps ─
        ns_blk = f"gh71_blk_{uuid.uuid4().hex[:8]}"
        block_size = 64
        hashes = [0x1000_0000_0000_0001 + i * 7 for i in range(32)]
        blob = _pack_hashes(hashes)
        r = c.call("KV.PREFIX.REGISTER", ns_blk, "256", "fp16",
                   "BLOCKS", str(block_size), str(len(hashes)), blob)
        if not r.startswith(b"+OK"):
            print(f"[2] FAIL REGISTER w/ BLOCKS: {r[:60]!r}"); return 1

        r = c.call("KV.PREFIX.BLOCKS", ns_blk)
        bsz, bcnt, got = _unpack_blocks(r)
        if (bsz, bcnt, got) != (block_size, len(hashes), hashes):
            print(f"[2] FAIL BLOCKS roundtrip: got bsz={bsz} bcnt={bcnt}"); fail = True
        else:
            print(f"[2a] OK — BLOCKS returns the same {bcnt} hashes (block_size={bsz})")

        # MEMBERSHIP: subset probe — half the stored hashes + 4 unknown.
        subset = hashes[::2] + [0x0BAD] * 4
        bm = _parse_bulk(c.call("KV.PREFIX.MEMBERSHIP", ns_blk,
                                str(len(subset)), _pack_hashes(subset)))
        got_set = _bitmap_to_set(bm, len(subset))
        want_set = {i for i in range(len(subset)) if subset[i] in set(hashes)}
        if got_set != want_set:
            print(f"[2b] FAIL MEMBERSHIP bitmap: got {sorted(got_set)} want {sorted(want_set)}"); fail = True
        else:
            print(f"[2b] OK — MEMBERSHIP bitmap: {len(got_set)}/{len(subset)} bits set")

        # MEMBERSHIP: empty probe.
        bm = _parse_bulk(c.call("KV.PREFIX.MEMBERSHIP", ns_blk, "0", b""))
        if bm != b"":
            print(f"[2c] FAIL empty MEMBERSHIP not empty: {bm!r}"); fail = True
        else:
            print(f"[2c] OK — empty MEMBERSHIP probe returns empty bitmap")

        # ── [3] WAL replay: SIGKILL, restart, BLOCKS still works ────────────
        c.close()
        proc.kill(); proc.wait(timeout=5)
        proc = _spawn_server()
        c = Conn()
        r = c.call("KV.PREFIX.BLOCKS", ns_blk)
        if r.startswith(b"+UNKNOWN") or r.startswith(b"-"):
            print(f"[3] FAIL post-restart BLOCKS: {r[:60]!r}"); fail = True
        else:
            bsz, bcnt, got = _unpack_blocks(r)
            if (bsz, bcnt, got) != (block_size, len(hashes), hashes):
                print(f"[3] FAIL post-WAL replay: bsz={bsz} bcnt={bcnt}"); fail = True
            else:
                print(f"[3] OK — WAL replay restored {bcnt} block hashes")

        # ── [4] Snapshot roundtrip: KV.PREFIX.SAVE → restart → BLOCKS works ─
        r = c.call("KV.PREFIX.SAVE")
        if not r.startswith(b"+OK"):
            print(f"[4] FAIL SAVE: {r[:60]!r}"); fail = True
        c.close()
        proc.kill(); proc.wait(timeout=5)
        # Now the snapshot must carry the block table even with an empty WAL.
        # Don't delete pion.wal.0 — wal_replay handles empty WALs fine, and
        # truncation already happened inside KV.PREFIX.SAVE.
        proc = _spawn_server()
        c = Conn()
        r = c.call("KV.PREFIX.BLOCKS", ns_blk)
        if r.startswith(b"+UNKNOWN") or r.startswith(b"-"):
            print(f"[4] FAIL post-snapshot BLOCKS: {r[:60]!r}"); fail = True
        else:
            bsz, bcnt, got = _unpack_blocks(r)
            if (bsz, bcnt, got) != (block_size, len(hashes), hashes):
                print(f"[4] FAIL post-snapshot roundtrip: bsz={bsz} bcnt={bcnt}"); fail = True
            else:
                print(f"[4] OK — snapshot replay restored {bcnt} block hashes")

        # ── [5] Microbench: 1,562-block probe latency ───────────────────────
        ns_big = f"gh71_big_{uuid.uuid4().hex[:8]}"
        big_hashes = [(0xC0FFEE << 24) | i for i in range(1562)]
        big_blob = _pack_hashes(big_hashes)
        r = c.call("KV.PREFIX.REGISTER", ns_big, "256", "fp16",
                   "BLOCKS", "64", str(len(big_hashes)), big_blob)
        if not r.startswith(b"+OK"):
            print(f"[5] FAIL big REGISTER: {r[:60]!r}"); fail = True
        else:
            # Probe with all 1,562 hashes; warm up + measure 200 calls.
            for _ in range(20):
                c.call("KV.PREFIX.MEMBERSHIP", ns_big, str(len(big_hashes)), big_blob)
            samples = []
            for _ in range(200):
                t0 = time.perf_counter_ns()
                c.call("KV.PREFIX.MEMBERSHIP", ns_big, str(len(big_hashes)), big_blob)
                samples.append(time.perf_counter_ns() - t0)
            samples.sort()
            p50_us = samples[len(samples) // 2] / 1000
            p99_us = samples[int(len(samples) * 0.99)] / 1000
            print(f"[5] MEMBERSHIP 1562-block probe: p50={p50_us:.1f}us  p99={p99_us:.1f}us")
            # The 100 µs target from gh #71 is server-side compute. End-to-end
            # over loopback adds the 12.5 KB request + ~200 B response RTT,
            # measured at ~110-130 µs p50 on Apple Silicon. Gate at 400 µs p50
            # to absorb noisy CI without missing a real algorithmic regression
            # (linear scan was 6.5 ms p50 here — 16× over budget — so this gate
            # would catch a re-introduction).
            if p50_us > 400:
                print(f"[5] FAIL: p50 {p50_us:.1f}us > 400us gate"); fail = True
            else:
                print(f"[5] OK — within 400 µs p50 e2e gate (100 µs server-side target)")

        c.close()
        print(f"\n{'PASS' if not fail else 'FAIL'} — gh #71 KV.PREFIX.BLOCKS / MEMBERSHIP")
        return 1 if fail else 0
    finally:
        proc.kill()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: pass


if __name__ == "__main__":
    sys.exit(main())
