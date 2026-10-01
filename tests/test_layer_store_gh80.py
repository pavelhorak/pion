#!/usr/bin/env python3
"""gh #80 follow-up — layer_store session LRU eviction gate.

Validates the wiring of the previously-orphaned delete_session free path:
at MAX_SESSIONS the layer store evicts the least-recently-accessed session
(reusing the same free logic) instead of rejecting new sessions.

  [1] No rejection at capacity: storing session #(MAX_SESSIONS) succeeds
      (binary STATUS_OK, not STATUS_MISS) once all 256 slots are full.
  [2] LRU ordering: after filling, FETCH-touching the oldest session
      protects it; the next new session evicts the least-recently-accessed
      untouched session instead. The touched session and the new session
      both remain fetchable; the untouched LRU victim misses.

LayerStore is only reachable over the 0xCA5E binary lane on port+1
(CMD_LAYER_STORE / CMD_LAYER_FETCH). The script manages the server
lifecycle itself (boot, probe, kill).

Requires: ./pion-server-dev (or ./pion-server) built with the gh #80 fix.
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import time

PORT = 1979          # RESP port (non-1974 to dodge the PionMesh iOS conflict)
BINARY_PORT = PORT + 1
HOST = "127.0.0.1"

BINARY_MAGIC = 0xCA5E
CMD_PING = 0xFF
CMD_LAYER_STORE = 0x10
CMD_LAYER_FETCH = 0x11
STATUS_OK = 0x00
STATUS_MISS = 0x01

MAX_SESSIONS = 256   # comptime MAX_SESSIONS in src/network/layer_store.mojo


def build_request(cmd: int, body: bytes) -> bytes:
    return struct.pack("<H", BINARY_MAGIC) + struct.pack("B", cmd) + struct.pack("<I", len(body)) + body


def layer_store_body(session_id: str, layer_id: int, tensor: bytes) -> bytes:
    sid = session_id.encode()
    return struct.pack("<H", len(sid)) + sid + struct.pack("<H", layer_id) + tensor


def layer_fetch_body(session_id: str, layer_id: int) -> bytes:
    sid = session_id.encode()
    return struct.pack("<H", len(sid)) + sid + struct.pack("<H", layer_id)


class BinaryClient:
    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, port))
        self.sock.settimeout(10)
        self.rf = self.sock.makefile("rb")

    def request(self, cmd: int, body: bytes):
        self.sock.sendall(build_request(cmd, body))
        header = self.rf.read(7)
        if len(header) < 7:
            raise ConnectionError("short binary response header")
        magic, status, body_len = struct.unpack("<HBI", header)
        assert magic == BINARY_MAGIC, f"bad magic {magic:#x}"
        payload = self.rf.read(body_len) if body_len else b""
        return status, payload

    def store(self, session_id: str, layer_id: int, tensor: bytes) -> int:
        status, _ = self.request(CMD_LAYER_STORE, layer_store_body(session_id, layer_id, tensor))
        return status

    def fetch(self, session_id: str, layer_id: int):
        return self.request(CMD_LAYER_FETCH, layer_fetch_body(session_id, layer_id))

    def close(self):
        self.rf.close()
        self.sock.close()


def boot_server() -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or ("./pion-server-dev" if os.path.exists("./pion-server-dev") else "./pion-server")
    # CI injects --epoll --no-auto-embed (hosted runners seccomp-block io_uring
    # and lack the embed sidecar's `transformers`); local Mac runs leave it unset.
    extra = os.environ.get("PION_SERVER_EXTRA_ARGS", "").split()
    proc = subprocess.Popen(
        [binary, "--kvcache", "-w", "1", "-p", str(PORT), *extra],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            c = BinaryClient(BINARY_PORT)
            status, _ = c.request(CMD_PING, b"")
            c.close()
            if status == STATUS_OK:
                return proc
        except (ConnectionRefusedError, OSError):
            if proc.poll() is not None:
                raise RuntimeError(f"{binary} exited during boot (rc={proc.returncode})")
            time.sleep(0.3)
    proc.kill()
    raise RuntimeError("binary lane did not come up in 30s")


def test_eviction_not_rejection(c: BinaryClient):
    print(f"[1] No rejection at capacity: filling {MAX_SESSIONS} sessions + 1 ...")
    tensor = b"GH80_LAYER_" * 4  # small — only the first page of each 64MB arena is touched
    for i in range(MAX_SESSIONS):
        st = c.store(f"ls80_{i}", 0, tensor)
        assert st == STATUS_OK, f"store #{i} failed: status={st}"
    # All 256 slots are now active. One more must evict, not reject.
    st = c.store("ls80_overflow", 0, tensor)
    assert st == STATUS_OK, f"overflow store rejected (status={st}) — eviction not wired"
    ov_status, ov_body = c.fetch("ls80_overflow", 0)
    assert ov_status == STATUS_OK and tensor in ov_body, "overflow session not fetchable after store"
    print("    OK — capacity store evicted instead of rejecting; new session fetchable")


def test_lru_ordering(c: BinaryClient):
    print("[2] LRU ordering: touched session survives the next eviction ...")
    tensor = b"GH80_LAYER_" * 4
    # State from [1]: one slot was already recycled for ls80_overflow (victim was
    # the LRU at that point, ls80_0). Re-seat a clean, fully-known LRU order by
    # re-storing every session, oldest first, then probe.
    for i in range(MAX_SESSIONS):
        assert c.store(f"lru_{i}", 0, tensor) == STATUS_OK
    # lru_0 is the oldest. Touch it so it becomes most-recently-used.
    touched_status, _ = c.fetch("lru_0", 0)
    assert touched_status == STATUS_OK, "lru_0 unexpectedly already evicted"
    # Force one eviction. The victim must be the oldest UNTOUCHED session: lru_1.
    assert c.store("lru_new", 0, tensor) == STATUS_OK
    survived_status, survived_body = c.fetch("lru_0", 0)
    assert survived_status == STATUS_OK and tensor in survived_body, \
        "recently-touched session was evicted — not LRU"
    victim_status, _ = c.fetch("lru_1", 0)
    assert victim_status == STATUS_MISS, "untouched LRU victim still present — eviction order wrong"
    new_status, _ = c.fetch("lru_new", 0)
    assert new_status == STATUS_OK, "newly stored session not fetchable"
    print("    OK — touched session survived, untouched LRU victim evicted, slot reused")


def test_blob_grow_past_64mb(c: BinaryClient):
    print("[3] Blob grow past 64MB initial cap: small marker + 17x4MB fillers + small marker ...")
    # Each session arena starts at a 64MB initial_cap (src/network/layer_store.mojo
    # store_layer). Accumulating >64MB of layer bytes crosses the cap and forces
    # the doubling copy-realloc (alloc new arena, memcpy blob_used bytes, free old).
    #
    # Design: a tiny marker at layer 0 lives at offset 0 (written BEFORE any grow);
    # 17x4MB fillers (68MB) drive the cap past 64MB mid-run; a tiny marker at the
    # last layer lives PAST the grown boundary. We fetch only the two small markers
    # to prove (a) pre-grow data survived the realloc memcpy and (b) post-grow data
    # landed at the right relative offset. The 4MB fillers are store-only — the
    # realloc we're validating is entirely on the store path, so small fetches
    # suffice. (Fetching a >=4MB layer over the binary lane is now safe too — the
    # old fixed-4MB-buffer overflow was fixed in gh #91; see test_layer_fetch_gh91.py.)
    FILLER_SZ = 4 * 1024 * 1024   # 4 MB
    N_FILLERS = 17                # 68 MB > 64 MB initial cap -> forces doubling realloc
    MARKER_LO = b"GH80_GROW_LO_" * 4
    MARKER_HI = b"GH80_GROW_HI_" * 4
    sid = "grow80"
    last_layer = 1 + N_FILLERS    # layers: 0 = lo marker, 1..17 = fillers, 18 = hi marker

    assert c.store(sid, 0, MARKER_LO) == STATUS_OK, "lo marker store failed"
    for k in range(N_FILLERS):
        filler = bytes([(k + 1) & 0xFF]) * FILLER_SZ
        assert c.store(sid, 1 + k, filler) == STATUS_OK, f"filler {k} store failed"
    assert c.store(sid, last_layer, MARKER_HI) == STATUS_OK, "hi marker store failed"

    lo_st, lo_body = c.fetch(sid, 0)
    assert lo_st == STATUS_OK and lo_body == MARKER_LO, \
        "pre-grow marker corrupted/lost across realloc memcpy"
    hi_st, hi_body = c.fetch(sid, last_layer)
    assert hi_st == STATUS_OK and hi_body == MARKER_HI, \
        "post-grow marker missing or at wrong offset after realloc"
    total_mb = N_FILLERS * FILLER_SZ // (1024 * 1024)
    print(f"    OK — {total_mb}MB drove the arena past 64MB; both markers intact across the doubling realloc")


def main() -> int:
    print(f"gh #80 layer_store gate — booting server on :{PORT} (binary :{BINARY_PORT})")
    proc = boot_server()
    try:
        c = BinaryClient(BINARY_PORT)
        test_eviction_not_rejection(c)
        test_lru_ordering(c)
        test_blob_grow_past_64mb(c)
        c.close()
        print("\n=== ALL gh #80 layer_store TESTS PASSED ===")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
