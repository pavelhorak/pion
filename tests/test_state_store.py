#!/usr/bin/env python3
"""Test STATE.ALLOC / STATE.WRITE / STATE.READ / STATE.FREE / STATE.INFO commands.

Issue #31 (A4) + A12 fold-in (ring-buffer mode).

Requires: ./pion-server --kvcache -w 1
"""

import socket
import sys
import time

HOST = "127.0.0.1"
PORT = 1974


class StateClient:
    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.settimeout(10)
        self.sock.connect((host, port))
        self.f = self.sock.makefile("rb")

    def _encode(self, parts):
        out = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            if isinstance(p, bytes):
                out.append(f"${len(p)}\r\n".encode() + p + b"\r\n")
            else:
                s = str(p)
                out.append(f"${len(s)}\r\n{s}\r\n".encode())
        return b"".join(out)

    def _read_line(self):
        return self.f.readline().rstrip(b"\r\n")

    def _read_reply(self):
        line = self._read_line()
        t, body = line[:1], line[1:]
        if t in (b"+", b"-", b":"):
            return line  # keep prefix
        if t == b"$":
            n = int(body)
            if n < 0:
                return None
            data = self.f.read(n)
            assert self.f.read(2) == b"\r\n"
            return data
        raise ValueError(f"unexpected reply: {line!r}")

    def call(self, *parts):
        self.sock.sendall(self._encode(parts))
        return self._read_reply()

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


# ── Test cases ──────────────────────────────────────────────────────────


def test_alloc_free_roundtrip():
    c = StateClient()
    sid = "rt_basic"
    # Defensive cleanup in case prior run left state
    c.call("STATE.FREE", sid)

    r = c.call("STATE.ALLOC", sid, 1024)
    assert r == b"+OK", f"ALLOC: {r!r}"

    r = c.call("STATE.WRITE", sid, 0, b"hello world")
    assert r == b":11", f"WRITE: {r!r}"

    r = c.call("STATE.READ", sid, 0, 11)
    assert r == b"hello world", f"READ: {r!r}"

    # Sparse write
    r = c.call("STATE.WRITE", sid, 100, b"AAA")
    assert r == b":3"
    r = c.call("STATE.READ", sid, 100, 3)
    assert r == b"AAA"
    # Region untouched in between is still zero
    r = c.call("STATE.READ", sid, 50, 4)
    assert r == b"\x00\x00\x00\x00"

    r = c.call("STATE.FREE", sid)
    assert r == b":1"

    # Re-alloc same sid succeeds
    r = c.call("STATE.ALLOC", sid, 64)
    assert r == b"+OK"
    r = c.call("STATE.READ", sid, 0, 4)
    assert r == b"\x00\x00\x00\x00", "re-alloc not zeroed"
    c.call("STATE.FREE", sid)
    c.close()
    print("PASS: test_alloc_free_roundtrip")


def test_fixed_mode_overflow():
    c = StateClient()
    sid = "fixed_oob"
    c.call("STATE.FREE", sid)
    assert c.call("STATE.ALLOC", sid, 16) == b"+OK"

    # Write within bounds
    assert c.call("STATE.WRITE", sid, 0, b"X" * 16) == b":16"
    # Boundary read
    assert c.call("STATE.READ", sid, 0, 16) == b"X" * 16

    # Overflow write → -ERR
    r = c.call("STATE.WRITE", sid, 10, b"Y" * 10)
    assert r.startswith(b"-ERR"), f"expected -ERR, got: {r!r}"
    assert b"out of range" in r

    # Overflow read → -ERR
    r = c.call("STATE.READ", sid, 8, 16)
    assert r.startswith(b"-ERR")

    c.call("STATE.FREE", sid)
    c.close()
    print("PASS: test_fixed_mode_overflow")


def test_ring_mode_wraparound():
    c = StateClient()
    sid = "ring_wrap"
    c.call("STATE.FREE", sid)
    assert c.call("STATE.ALLOC", sid, 16, "MODE", "ring") == b"+OK"

    # Pre-fill with a known pattern
    assert c.call("STATE.WRITE", sid, 0, b"0123456789abcdef") == b":16"
    assert c.call("STATE.READ", sid, 0, 16) == b"0123456789abcdef"

    # Wrapping write: 8 bytes starting at offset 12 → physical[12..15] + physical[0..3]
    assert c.call("STATE.WRITE", sid, 12, b"WXYZ!@#$") == b":8"
    # Physical buffer is now: !@#$456789abWXYZ
    assert c.call("STATE.READ", sid, 0, 16) == b"!@#$456789abWXYZ"

    # Logical read across wrap also stitches correctly
    assert c.call("STATE.READ", sid, 12, 8) == b"WXYZ!@#$"

    # Offset beyond size: ring mod-maps it
    assert c.call("STATE.WRITE", sid, 16 + 4, b"ZZZZ") == b":4"
    # offset 20 % 16 = 4; bytes at physical[4..7] become ZZZZ
    assert c.call("STATE.READ", sid, 4, 4) == b"ZZZZ"

    c.call("STATE.FREE", sid)
    c.close()
    print("PASS: test_ring_mode_wraparound")


def test_concurrent_sessions_dont_collide():
    c = StateClient()
    for sid in ("multi_a", "multi_b", "multi_c"):
        c.call("STATE.FREE", sid)
        assert c.call("STATE.ALLOC", sid, 32) == b"+OK"

    assert c.call("STATE.WRITE", "multi_a", 0, b"AAAA") == b":4"
    assert c.call("STATE.WRITE", "multi_b", 0, b"BBBB") == b":4"
    assert c.call("STATE.WRITE", "multi_c", 0, b"CCCC") == b":4"

    assert c.call("STATE.READ", "multi_a", 0, 4) == b"AAAA"
    assert c.call("STATE.READ", "multi_b", 0, 4) == b"BBBB"
    assert c.call("STATE.READ", "multi_c", 0, 4) == b"CCCC"

    # Free middle, re-alloc, others unchanged
    assert c.call("STATE.FREE", "multi_b") == b":1"
    assert c.call("STATE.ALLOC", "multi_b", 32) == b"+OK"
    assert c.call("STATE.READ", "multi_a", 0, 4) == b"AAAA"
    assert c.call("STATE.READ", "multi_c", 0, 4) == b"CCCC"

    for sid in ("multi_a", "multi_b", "multi_c"):
        c.call("STATE.FREE", sid)
    c.close()
    print("PASS: test_concurrent_sessions_dont_collide")


def test_double_alloc_rejected():
    c = StateClient()
    sid = "double_alloc"
    c.call("STATE.FREE", sid)
    assert c.call("STATE.ALLOC", sid, 64) == b"+OK"
    r = c.call("STATE.ALLOC", sid, 128)
    assert r.startswith(b"-ERR") and b"already allocated" in r, f"got {r!r}"
    c.call("STATE.FREE", sid)
    c.close()
    print("PASS: test_double_alloc_rejected")


def test_write_to_unallocated_sid():
    c = StateClient()
    sid = "no_such_sid_xyz_unique"
    c.call("STATE.FREE", sid)
    r = c.call("STATE.WRITE", sid, 0, b"x")
    assert r.startswith(b"-ERR") and b"not allocated" in r, f"got {r!r}"
    r = c.call("STATE.READ", sid, 0, 1)
    assert r.startswith(b"-ERR") and b"not allocated" in r
    r = c.call("STATE.FREE", sid)
    assert r == b":0"  # idempotent
    c.close()
    print("PASS: test_write_to_unallocated_sid")


def test_invalid_args():
    c = StateClient()
    r = c.call("STATE.ALLOC")
    assert r.startswith(b"-ERR")
    r = c.call("STATE.ALLOC", "x", 0)  # zero size
    assert r.startswith(b"-ERR")
    r = c.call("STATE.ALLOC", "x", -1)  # negative size
    assert r.startswith(b"-ERR")

    sid = "bad_args"
    c.call("STATE.FREE", sid)
    assert c.call("STATE.ALLOC", sid, 32) == b"+OK"
    r = c.call("STATE.WRITE", sid, -5, b"x")
    assert r.startswith(b"-ERR") and b"non-negative" in r
    r = c.call("STATE.READ", sid, 0, 0)
    assert r.startswith(b"-ERR")
    r = c.call("STATE.READ", sid, 0, -1)
    assert r.startswith(b"-ERR")

    # Bad MODE keyword
    c.call("STATE.FREE", "bad_mode")
    r = c.call("STATE.ALLOC", "bad_mode", 32, "MODE", "swirl")
    assert r.startswith(b"-ERR") and b"MODE" in r
    c.call("STATE.FREE", sid)
    c.close()
    print("PASS: test_invalid_args")


def test_info_surface():
    c = StateClient()
    sid = "info_check"
    c.call("STATE.FREE", sid)
    assert c.call("STATE.ALLOC", sid, 256, "MODE", "ring") == b"+OK"
    c.call("STATE.WRITE", sid, 0, b"hello")
    c.call("STATE.READ", sid, 0, 5)

    # Per-session
    info = c.call("STATE.INFO", sid).decode()
    assert "size:256" in info, info
    assert "mode:ring" in info, info
    assert "bytes_written:5" in info, info
    assert "bytes_read:5" in info, info

    # Global
    info = c.call("STATE.INFO").decode()
    assert "enabled:1" in info
    assert "max_sessions:64" in info
    assert "sessions:" in info

    c.call("STATE.FREE", sid)
    c.close()
    print("PASS: test_info_surface")


# ── Runner ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((HOST, PORT))
        s.close()
    except Exception:
        print(f"Pion not running at {HOST}:{PORT}")
        print("Start with: ./pion-server --kvcache -w 1")
        sys.exit(1)

    print("=" * 60)
    print("STATE.* tests (issue #31, A4 + A12 fold-in)")
    print("=" * 60)

    tests = [
        test_alloc_free_roundtrip,
        test_fixed_mode_overflow,
        test_ring_mode_wraparound,
        test_concurrent_sessions_dont_collide,
        test_double_alloc_rejected,
        test_write_to_unallocated_sid,
        test_invalid_args,
        test_info_surface,
    ]
    passed = 0
    failed = []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed.append((t.__name__, repr(e)))
            print(f"FAIL: {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if failed:
        for name, err in failed:
            print(f"  - {name}: {err}")
    sys.exit(0 if passed == len(tests) else 1)
