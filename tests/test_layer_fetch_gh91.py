#!/usr/bin/env python3
"""gh #91 — binary LAYER_FETCH large-blob response guard.

Before the fix, the binary-lane CMD_LAYER_FETCH handler memcpy'd the fetched
layer into a fixed 4 MB `binary_resp_buf` with no size check. A layer whose
tensor is >= ~4 MB overflowed the buffer (heap OOB write) and the subsequent
send() over-read it, so the client received a corrupt/short frame and blocked
forever on `rf.read(body_len)`.

The fix streams the 7-byte header + blob via scatter-gather (server.send loop)
for blobs that don't fit alongside the header, lifting the 4 MB cap.

Coverage:
  [1] 6 MB layer round-trips byte-for-byte over the binary lane (the core bug:
      pre-fix this hung / corrupted).
  [2] Boundary: a layer at exactly (4 MB - 7) still takes the buffered path and
      round-trips, and a layer at (4 MB - 6) — the smallest writev case — does too.
  [3] A small layer stored AFTER a large fetch still works (server/connection
      not left in a bad state by the large send).

LayerStore is only reachable over the 0xCA5E binary lane on port+1
(CMD_LAYER_STORE / CMD_LAYER_FETCH). The script manages the server lifecycle.

Requires: ./pion-server-dev (or ./pion-server) built with the gh #91 fix.
"""
from __future__ import annotations

import hashlib
import os
import socket
import struct
import subprocess
import sys
import time

PORT = 1981          # RESP port (non-1974 to dodge the PionMesh iOS conflict)
BINARY_PORT = PORT + 1
HOST = "127.0.0.1"

BINARY_MAGIC = 0xCA5E
CMD_PING = 0xFF
CMD_LAYER_STORE = 0x10
CMD_LAYER_FETCH = 0x11
STATUS_OK = 0x00
STATUS_MISS = 0x01

RESP_BUF_CAP = 4 * 1024 * 1024
BINARY_RESP_HEADER_SIZE = 7


def build_request(cmd: int, body: bytes) -> bytes:
    return struct.pack("<H", BINARY_MAGIC) + struct.pack("B", cmd) + struct.pack("<I", len(body)) + body


def layer_store_body(session_id: str, layer_id: int, tensor: bytes) -> bytes:
    sid = session_id.encode()
    return struct.pack("<H", len(sid)) + sid + struct.pack("<H", layer_id) + tensor


def layer_fetch_body(session_id: str, layer_id: int) -> bytes:
    sid = session_id.encode()
    return struct.pack("<H", len(sid)) + sid + struct.pack("<H", layer_id)


def make_tensor(seed: int, size: int) -> bytes:
    # Deterministic, non-uniform content so a truncated/corrupt frame is caught
    # (not just a length check). Cheap to generate for multi-MB sizes.
    block = hashlib.sha256(struct.pack("<I", seed)).digest()  # 32 bytes
    reps = size // len(block) + 1
    return (block * reps)[:size]


class BinaryClient:
    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, port))
        self.sock.settimeout(15)
        self.rf = self.sock.makefile("rb")

    def request(self, cmd: int, body: bytes):
        self.sock.sendall(build_request(cmd, body))
        header = self.rf.read(7)
        if len(header) < 7:
            raise ConnectionError("short binary response header")
        magic, status, body_len = struct.unpack("<HBI", header)
        assert magic == BINARY_MAGIC, f"bad magic {magic:#x}"
        payload = self.rf.read(body_len) if body_len else b""
        if len(payload) != body_len:
            raise ConnectionError(f"short body: got {len(payload)} of {body_len}")
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


def _roundtrip(c: BinaryClient, sid: str, layer: int, tensor: bytes, label: str):
    assert c.store(sid, layer, tensor) == STATUS_OK, f"{label}: store failed"
    st, body = c.fetch(sid, layer)
    assert st == STATUS_OK, f"{label}: fetch status={st}"
    assert len(body) == len(tensor), f"{label}: length {len(body)} != {len(tensor)}"
    assert body == tensor, f"{label}: body corrupted (hash mismatch)"


def test_large_layer_roundtrip(c: BinaryClient):
    print("[1] 6 MB layer round-trips byte-for-byte over the binary lane ...")
    tensor = make_tensor(0xBEEF, 6 * 1024 * 1024)
    _roundtrip(c, "gh91_big", 0, tensor, "6MB")
    print("    OK — 6 MB blob fetched intact (no overflow, no hang)")


def test_boundary(c: BinaryClient):
    print("[2] Boundary at the 4 MB buffer cap ...")
    # Largest blob that still fits buffered: blob_len + 7 == 4MB exactly.
    buffered = RESP_BUF_CAP - BINARY_RESP_HEADER_SIZE          # 4194297
    _roundtrip(c, "gh91_edge", 0, make_tensor(1, buffered), "4MB-7 (buffered)")
    # Smallest blob that must take the writev path: blob_len + 7 == 4MB + 1.
    writev_min = RESP_BUF_CAP - BINARY_RESP_HEADER_SIZE + 1    # 4194298
    _roundtrip(c, "gh91_edge", 1, make_tensor(2, writev_min), "4MB-6 (writev)")
    print("    OK — both sides of the buffered/writev threshold round-trip")


def test_connection_survives(c: BinaryClient):
    print("[3] Small layer after a large fetch still works ...")
    big = make_tensor(0xF00D, 5 * 1024 * 1024)
    _roundtrip(c, "gh91_seq", 0, big, "5MB")
    marker = b"GH91_SMALL_AFTER_BIG"
    _roundtrip(c, "gh91_seq", 1, marker, "small-after-big")
    miss_status, _ = c.fetch("gh91_seq", 99)
    assert miss_status == STATUS_MISS, "absent layer should MISS, not corrupt"
    print("    OK — connection healthy after large send; small fetch + MISS correct")


def main() -> int:
    print(f"gh #91 layer-fetch large-blob gate — booting server on :{PORT} (binary :{BINARY_PORT})")
    proc = boot_server()
    try:
        c = BinaryClient(BINARY_PORT)
        test_large_layer_roundtrip(c)
        test_boundary(c)
        test_connection_survives(c)
        c.close()
        print("\n=== ALL gh #91 layer-fetch TESTS PASSED ===")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
