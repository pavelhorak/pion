#!/usr/bin/env python3
"""gh #76 — SSM.PREFIX.FETCH must not crash on bulk-string responses >= 4 MB.

Pre-fix: server's response writer memcpy'd into the 4 MB RESP_BUF_SIZE without
a bounds check; an 8 MB blob overflowed and SIGSEGV'd `process_slow_path`.

Post-fix: handler routes through `append_bulk_bytes_writev`, which sends large
payloads with a blocking send loop instead of memcpying into the response
buffer. Small payloads still take the buffered fast path.

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import os
import socket
import sys

HOST = "127.0.0.1"
PORT = int(os.environ.get("PION_PORT", "1974"))


def _encode(parts) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        if isinstance(p, bytes):
            out.append(f"${len(p)}\r\n".encode()); out.append(p); out.append(b"\r\n")
        else:
            s = str(p)
            out.append(f"${len(s)}\r\n{s}\r\n".encode())
    return b"".join(out)


class Conn:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.s.settimeout(60)
        self.s.connect((HOST, PORT))
        self.f = self.s.makefile("rb")

    def call_status(self, *parts) -> bytes:
        self.s.sendall(_encode(parts))
        return self.f.readline()

    def call_bulk(self, *parts) -> bytes | None:
        """Send command and read one bulk-string (or null) reply via makefile.

        makefile('rb') avoids the O(N²) bytes-slicing trap on multi-MB replies
        (memory: python_resp_array_slicing_trap.md, 2026-05-02).
        """
        self.s.sendall(_encode(parts))
        head = self.f.readline()
        if head.startswith(b"$-1"):
            return None
        if not head.startswith(b"$"):
            raise RuntimeError(f"expected bulk reply, got: {head[:80]!r}")
        n = int(head[1:-2])
        body = self.f.read(n)
        trailer = self.f.read(2)  # \r\n
        assert trailer == b"\r\n", f"missing CRLF after {n}-byte bulk"
        return body


def _try_connect() -> bool:
    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
        return True
    except OSError:
        return False


def _check(label: str, size: int, c: Conn) -> bool:
    """STORE a blob of `size` bytes with a known pattern, FETCH it, byte-compare."""
    sid = f"ssm_gh76_{size}"
    # Pattern: byte i % 251 — non-trivial, catches off-by-one + boundary issues.
    blob = bytes((i % 251) for i in range(size))

    r = c.call_status("SSM.PREFIX.STORE", sid, "0", blob)
    if not r.startswith(b"+OK"):
        print(f"   [{label}] FAIL: STORE rejected: {r[:80]!r}")
        return False

    try:
        got = c.call_bulk("SSM.PREFIX.FETCH", sid, "0")
    except (ConnectionError, BrokenPipeError, OSError) as e:
        print(f"   [{label}] FAIL: FETCH connection closed — server likely crashed ({e})")
        return False

    if got is None:
        print(f"   [{label}] FAIL: FETCH returned null")
        return False
    if len(got) != size:
        print(f"   [{label}] FAIL: length mismatch in={size} out={len(got)}")
        return False
    if got != blob:
        diffs = sum(1 for a, b in zip(blob, got) if a != b)
        print(f"   [{label}] FAIL: byte mismatch ({diffs} of {size} bytes differ)")
        return False
    c.call_status("SSM.PREFIX.DROP", sid, "0")
    print(f"   [{label}] OK: {size} bytes round-tripped exactly")
    return True


def main() -> int:
    if not _try_connect():
        print(f"FAIL: pion not reachable at {HOST}:{PORT}. Start: ./pion-server --kvcache -w 1")
        return 2

    print("gh #76 — SSM.PREFIX.FETCH large-blob regression")
    c = Conn()
    fail = False

    # Boundary table — the issue triangulated the crash between 2 MB and 8 MB.
    # Cover under-, at-, and well-past the 4 MB RESP_BUF_SIZE.
    sizes = [
        ("< 1 MB",  512 * 1024),
        ("~= 2 MB",  2 * 1024 * 1024),         # known-good from gh #61 spike
        ("~= 4 MB",  4 * 1024 * 1024),         # exact buffer size
        ("4 MB + 1", 4 * 1024 * 1024 + 1),    # one byte over — header alone overflows
        ("~= 8 MB",  8 * 1024 * 1024),         # the original repro from the issue
        ("16 MB",   16 * 1024 * 1024),         # LMCache-class blob
    ]
    for label, n in sizes:
        if not _check(label, n, c):
            fail = True
            break  # stop after the first failure — connection likely dead

    print(f"\n{'PASS' if not fail else 'FAIL'} — gh #76 SSM.PREFIX.FETCH large-blob")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
