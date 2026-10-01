#!/usr/bin/env python3
"""DIAGNOSTIC (not a gate): which commands swallow the next pipelined command?

Pipelines `<cmd>` and `PING` in a single write and reports every command whose
`PING` never comes back. A missing PONG means the handler consumed tokens it did
not answer for — the gh #156 / #162 / FT.*-2026-08-05 class, where a handler
returns `num_tokens - i - 1` (a skip to the end of the whole recv buffer) from a
dispatch site that does `i += handler(...)`. One command in, one reply out, and
everything batched behind it silently disappears.

This is a reporting tool, deliberately NOT wired into /gate: as of 0.899 it
reports 19 affected commands, all pre-existing, and fixing them is a dispatch-loop
change that needs its own gate on both platforms. Run it before and after that
work; when the list is empty, promote the sweep to a hard assertion and flip the
three KNOWN entries in tests/test_gh202_gh203.py to real checks.

    ./pion-server -p 2031 -w 1 --no-auto-detect --no-auto-embed &
    python3 tests/diag_pipeline_desync_sweep.py 2031
"""
import socket
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 1974

# Fixtures so every probed command hits a populated key rather than an
# early-out error path — an error reply can return a different skip count than
# the success path (that WAS gh #156: the error paths desynced while the
# success paths were correct), so probing empty keys can hide the bug.
FIXTURES = [
    ("SET", "k", "v"),
    ("XADD", "st", "1-1", "f", "v"),
    ("LPUSH", "l", "a"),
    ("HSET", "h", "f", "v"),
    ("SADD", "st2", "m"),
    ("ZADD", "z", "1", "m"),
    ("PFADD", "hll", "a"),
    ("GEOADD", "g", "13.0", "38.0", "p"),
]

COMMANDS = [
    ("PING",), ("ECHO", "x"), ("GET", "k"), ("SET", "k", "v"), ("DEL", "nope"),
    ("EXISTS", "k"), ("TTL", "k"), ("TYPE", "k"), ("STRLEN", "k"),
    ("INCRBY", "n", "1"), ("APPEND", "k", "z"),
    ("XLEN", "st"), ("XRANGE", "st", "-", "+"), ("XREVRANGE", "st", "+", "-"),
    ("XINFO", "STREAM", "st"), ("XDEL", "st", "9-9"), ("XTRIM", "st", "MAXLEN", "5"),
    ("LLEN", "l"), ("LRANGE", "l", "0", "-1"),
    ("HGET", "h", "f"), ("HGETALL", "h"),
    ("SMEMBERS", "st2"), ("SCARD", "st2"), ("SINTERCARD", "1", "st2"),
    ("ZCARD", "z"), ("ZRANGE", "z", "0", "-1"), ("ZSCORE", "z", "m"),
    ("ZREM", "z", "nope"),
    ("PFCOUNT", "hll"),
    ("GEOPOS", "g", "p"), ("GEODIST", "g", "p", "p"),
    ("GEOSEARCH", "g", "FROMMEMBER", "p", "BYRADIUS", "1", "km", "ASC"),
    ("ACL", "WHOAMI"), ("ACL", "LIST"), ("ACL", "USERS"), ("ACL", "CAT"), ("ACL", "LOG"),
    ("CONFIG", "GET", "maxmemory"), ("COMMAND", "DOCS"), ("COMMAND", "COUNT"),
    ("CLIENT", "ID"), ("DEBUG", "JMAP"), ("MEMORY", "USAGE", "k"),
    ("SLOWLOG", "GET"), ("LATENCY", "RESET"), ("MODULE", "LIST"),
    ("DBSIZE",), ("INFO",), ("OBJECT", "ENCODING", "k"), ("RANDOMKEY",),
    ("SCAN", "0"), ("KEYS", "*"), ("LPOS", "l", "a"),
    ("FUNCTION", "LIST"), ("SCRIPT", "EXISTS", "abc"),
    ("CLUSTER", "INFO"), ("RESET",),
]


def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)


def connect(timeout=2.0):
    s = socket.create_connection(("127.0.0.1", PORT), timeout=timeout)
    return s


def main():
    s = connect()
    for f in FIXTURES:
        s.sendall(encode(f))
        try:
            s.recv(65536)
        except socket.timeout:
            pass
    s.close()

    swallowed = []
    for cmd in COMMANDS:
        # Fresh connection per probe: a swallow leaves the connection with an
        # unanswered request, which would corrupt every later probe.
        s = connect()
        buf = b""
        try:
            s.sendall(encode(cmd) + encode(("PING",)))
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline and b"+PONG" not in buf:
                try:
                    chunk = s.recv(65536)
                except (socket.timeout, OSError):
                    break
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()
        if b"+PONG" not in buf:
            swallowed.append((" ".join(map(str, cmd)), buf[:60]))

    print(f"{len(COMMANDS)} commands probed on port {PORT}; "
          f"{len(swallowed)} swallow the pipelined PING")
    for name, reply in swallowed:
        print(f"  {name:38} reply={reply!r}")
    if not swallowed:
        print("  (none — promote this sweep to a hard assertion)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
