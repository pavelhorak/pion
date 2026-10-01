#!/usr/bin/env python3
"""Value-size boundary sweep — every representation switch in the engine.

Pion changes how a value is stored at several exact sizes, and each switch is a
place where a length can be computed with the wrong representation in mind:

  * 23 bytes  — GenericValue STRING_SSO (inline, zero heap) -> heap STRING.
                The SSO form packs length+chars into three UInt64 words; the
                heap form puts a pointer in _data0. Off-by-one here means
                reading characters as a pointer or vice versa.
  * 64 bytes  — SlabList ziplist entry threshold (values over it force the
                segmented quicklist conversion).
  * 1024 entries — ziplist -> quicklist conversion by count.
  * 1 MiB     — blob tier (gh #163): values at or above --blob-threshold go to
                an mmap'd arena and the WAL logs a 24-byte pointer record
                instead of the payload. Values carry BLOB_TAG; free_str_payload
                must never free() them.
  * 3 MB      — bulk replies above this use writev (gh #76).

Each case asserts a byte-exact round-trip, so a truncation, an off-by-one, or a
representation mix-up shows up as a diff rather than a crash. Payloads are
patterned (not zeros) so truncation cannot masquerade as success.

Usage: python3 tests/test_size_boundaries.py [--port 1974] [--big]
       --big adds the 1 MiB / 3 MB / 4 MB cases (slower, needs disk headroom).
"""

import argparse
import hashlib
import socket
import sys

HOST = "127.0.0.1"
PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} {detail}")


class Conn:
    def __init__(self, port, timeout=30):
        self.s = socket.create_connection((HOST, port), timeout=timeout)
        self.s.settimeout(timeout)
        self.f = self.s.makefile("rb")

    def cmd(self, *args):
        buf = f"*{len(args)}\r\n".encode()
        for a in args:
            a = a.encode() if isinstance(a, str) else a
            buf += b"$%d\r\n%s\r\n" % (len(a), a)
        self.s.sendall(buf)
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise EOFError("server closed")
        t, body = line[:1], line[1:-2]
        if t == b"+":
            return body.decode()
        if t == b"-":
            return "ERR:" + body.decode()
        if t == b":":
            return int(body)
        if t == b"$":
            n = int(body)
            return None if n == -1 else self.f.read(n + 2)[:-2]   # raw bytes
        if t == b"*":
            n = int(body)
            return [] if n <= 0 else [self._read() for _ in range(n)]
        raise ValueError(f"bad RESP {line!r}")

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def payload(n):
    """Patterned bytes — a truncation or a zero-fill cannot look correct."""
    return bytes((i * 31 + (i >> 8) * 7) & 0xFF for i in range(n))


# The exact switch points, each bracketed by -1 / +1.
SIZES = [0, 1, 7, 8, 15, 16, 22, 23, 24, 31, 32, 63, 64, 65,
         127, 128, 255, 256, 511, 512, 1023, 1024, 1025,
         4095, 4096, 8192, 65535, 65536]
BIG_SIZES = [1024 * 1024 - 1, 1024 * 1024, 1024 * 1024 + 1,
             3 * 1024 * 1024, 4 * 1024 * 1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--big", action="store_true")
    args = ap.parse_args()

    sizes = SIZES + (BIG_SIZES if args.big else [])
    try:
        c = Conn(args.port)
    except OSError as e:
        print(f"FATAL: no server on {HOST}:{args.port} ({e})")
        return 2

    print(f"size-boundary sweep: {len(sizes)} sizes "
          f"(SSO at 23B, ziplist at 64B, blob tier at 1MiB, writev at 3MB)\n")

    print("[1] SET/GET byte-exact round-trip")
    for n in sizes:
        key = f"sb:str:{n}"
        data = payload(n)
        try:
            c.cmd("SET", key, data)
            got = c.cmd("GET", key)
        except (EOFError, socket.timeout) as e:
            check(f"SET/GET {n}B", False, f"{type(e).__name__} — server died?")
            return 1
        if got is None:
            got = b""
        ok = got == data
        check(f"SET/GET {n}B", ok,
              f"sha {hashlib.sha256(got).hexdigest()[:12]}" if ok
              else f"got {len(got)}B of {n}B")

    print("\n[2] STRLEN agrees with the stored length")
    for n in sizes:
        got = c.cmd("STRLEN", f"sb:str:{n}")
        check(f"STRLEN {n}B", got == n, f"got {got}")

    print("\n[3] APPEND across the SSO boundary (grow 1 byte at a time)")
    c.cmd("DEL", "sb:app")
    acc = b""
    for step in range(30):          # walks 0 -> 30 bytes, crossing 23
        c.cmd("APPEND", "sb:app", b"x")
        acc += b"x"
        got = c.cmd("GET", "sb:app")
        if got != acc:
            check(f"APPEND to {len(acc)}B", False,
                  f"got {len(got) if got else 0}B, expected {len(acc)}B")
            break
    else:
        check("APPEND across SSO boundary (1..30B)", True, "every step exact")

    print("\n[4] GETRANGE at and around the boundary")
    key = "sb:gr"
    data = payload(64)
    c.cmd("SET", key, data)
    for start, end in [(0, 22), (0, 23), (0, 24), (22, 23), (23, 24), (0, -1), (-5, -1)]:
        got = c.cmd("GETRANGE", key, str(start), str(end))
        exp = data[start:end + 1] if end >= 0 else data[start:len(data) + end + 1]
        check(f"GETRANGE [{start},{end}]", got == exp,
              f"got {len(got) if got else 0}B want {len(exp)}B")

    print("\n[5] List elements across the ziplist threshold")
    for n in [22, 23, 24, 63, 64, 65, 200]:
        key = f"sb:list:{n}"
        c.cmd("DEL", key)
        elems = [payload(n), payload(n)[::-1], bytes([0xAB]) * n]
        for e in elems:
            c.cmd("RPUSH", key, e)
        got = c.cmd("LRANGE", key, "0", "-1")
        ok = isinstance(got, list) and got == elems
        check(f"list of {n}B elements", ok,
              "3 elements byte-exact" if ok else f"got {got if not isinstance(got, list) else [len(x) for x in got]}")

    print("\n[6] Ziplist -> quicklist conversion by COUNT (1024 entries)")
    key = "sb:many"
    c.cmd("DEL", key)
    for i in range(1100):
        c.cmd("RPUSH", key, f"e{i}".encode())
    llen = c.cmd("LLEN", key)
    check("1100 entries: LLEN", llen == 1100, f"got {llen}")
    first = c.cmd("LRANGE", key, "0", "0")
    last = c.cmd("LRANGE", key, "-1", "-1")
    check("1100 entries: first survives conversion", first == [b"e0"], f"got {first}")
    check("1100 entries: last survives conversion", last == [b"e1099"], f"got {last}")

    print("\n[7] Server still healthy")
    check("PING after all sizes", c.cmd("PING") == "PONG")
    c.close()

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
