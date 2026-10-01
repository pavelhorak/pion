#!/usr/bin/env python3
"""gh #221 — ZREVRANGEBYSCORE / ZREVRANGEBYLEX.

Two stacked bugs meant neither command had ever worked:

  1. The dispatch arms in slow_path.mojo carried each other's lengths
     (ZREVRANGEBYSCORE is 16 bytes, ZREVRANGEBYLEX is 14), and length is the
     ONLY thing separating them — both arms match on tp[3]=='v'. So each
     command ran the other's handler: SCORE answered [] *silently*, LEX
     answered -ERR internal error.

  2. With dispatch crossed, handle_zrevrangebyscore never saw its own input, so
     its argument indexing was untested: the low bound read the MAX token and
     the high bound read the KEY, making `ZREVRANGEBYSCORE zf 3 1` call
     atof("zf").

The empty-array reply is why this needs a test rather than a spot check: a
client cannot distinguish "no members in range" from "command is broken", and
neither command appears in test_parity.py or the gate rows.

Usage: python3 tests/test_gh221_zrev.py [--port 1974]
Assumes a server is already listening.
"""

import argparse
import socket
import sys

HOST = "127.0.0.1"
PASSED, FAILED = [], []


def check(name, got, want):
    ok = got == want
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} -> {got}"
          + ("" if ok else f"   want {want}"))


class Conn:
    def __init__(self, port):
        self.s = socket.create_connection((HOST, port), timeout=5)
        self.s.settimeout(5)
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
            raise EOFError("server closed — a raise in the handler aborts a -O0 build")
        t, body = line[:1], line[1:-2]
        if t in b"+:":
            return body.decode()
        if t == b"-":
            return "ERR:" + body.decode()
        if t == b"$":
            n = int(body)
            return None if n == -1 else self.f.read(n + 2)[:-2].decode()
        if t == b"*":
            n = int(body)
            return None if n == -1 else [self._read() for _ in range(n)]
        raise ValueError(f"bad RESP {line!r}")

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()

    try:
        c = Conn(args.port)
    except OSError as e:
        print(f"FATAL: no server on {HOST}:{args.port} ({e})")
        return 2

    print(f"gh #221 ZREVRANGEBY* against {HOST}:{args.port}\n")
    c.cmd("DEL", "gh221:z")
    c.cmd("ZADD", "gh221:z", "1", "a", "2", "b", "3", "c", "4", "d")

    # ── ZREVRANGEBYSCORE: key max min, results high -> low ──
    check("ZREVRANGEBYSCORE 4 1",
          c.cmd("ZREVRANGEBYSCORE", "gh221:z", "4", "1"), ["d", "c", "b", "a"])
    check("ZREVRANGEBYSCORE 3 2",
          c.cmd("ZREVRANGEBYSCORE", "gh221:z", "3", "2"), ["c", "b"])
    check("ZREVRANGEBYSCORE +inf -inf",
          c.cmd("ZREVRANGEBYSCORE", "gh221:z", "+inf", "-inf"), ["d", "c", "b", "a"])
    # Exclusive bounds — the '(' path parses its own token and was NOT affected
    # by the off-by-one, so this pins both paths against each other.
    check("ZREVRANGEBYSCORE (4 1",
          c.cmd("ZREVRANGEBYSCORE", "gh221:z", "(4", "1"), ["c", "b", "a"])
    check("ZREVRANGEBYSCORE 4 (1",
          c.cmd("ZREVRANGEBYSCORE", "gh221:z", "4", "(1"), ["d", "c", "b"])
    # An empty range must be empty for the RIGHT reason, not because the
    # command is broken — the exact ambiguity that hid this bug.
    check("ZREVRANGEBYSCORE 99 90 (genuinely empty)",
          c.cmd("ZREVRANGEBYSCORE", "gh221:z", "99", "90"), [])
    check("ZREVRANGEBYSCORE WITHSCORES",
          c.cmd("ZREVRANGEBYSCORE", "gh221:z", "2", "1", "WITHSCORES"),
          ["b", "2", "a", "1"])

    # ── ZREVRANGEBYLEX: key max min, reverse lexical ──
    check("ZREVRANGEBYLEX [d [a",
          c.cmd("ZREVRANGEBYLEX", "gh221:z", "[d", "[a"), ["d", "c", "b", "a"])
    check("ZREVRANGEBYLEX + -",
          c.cmd("ZREVRANGEBYLEX", "gh221:z", "+", "-"), ["d", "c", "b", "a"])
    check("ZREVRANGEBYLEX (d [a",
          c.cmd("ZREVRANGEBYLEX", "gh221:z", "(d", "[a"), ["c", "b", "a"])

    # ── Controls: the neighbouring arms must be untouched by the length swap.
    # ZREM* share the tl values and differ only at tp[3] ('m' vs 'v').
    check("ctl ZRANGEBYSCORE 1 4",
          c.cmd("ZRANGEBYSCORE", "gh221:z", "1", "4"), ["a", "b", "c", "d"])
    check("ctl ZRANGEBYLEX - +",
          c.cmd("ZRANGEBYLEX", "gh221:z", "-", "+"), ["a", "b", "c", "d"])
    check("ctl ZREVRANGE 0 -1",
          c.cmd("ZREVRANGE", "gh221:z", "0", "-1"), ["d", "c", "b", "a"])
    check("ctl ZREMRANGEBYSCORE 1 1",
          c.cmd("ZREMRANGEBYSCORE", "gh221:z", "1", "1"), "1")
    check("ctl ZREMRANGEBYLEX [b [b",
          c.cmd("ZREMRANGEBYLEX", "gh221:z", "[b", "[b"), "1")
    check("ctl survivors after ZREM*",
          c.cmd("ZRANGE", "gh221:z", "0", "-1"), ["c", "d"])

    # The connection must still be usable: a handler raise used to abort the
    # whole process on a -O0 build rather than erroring one command.
    check("server still serving", c.cmd("PING"), "PONG")

    c.close()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
