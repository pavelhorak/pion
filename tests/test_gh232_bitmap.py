#!/usr/bin/env python3
"""gh #232 — bitmap bit ORDER and BITFIELD numerics, against a real Redis.

Redis numbers bits from the MOST significant end of each byte: `SETBIT k 7 1`
yields the string `\\x01` and `SETBIT k 0 1` yields `\\x80`. Pion used
`offset % 8` directly, i.e. bit 0 = LSB, producing exactly the reverse.

That was invisible from inside the bitmap commands, which is why it survived:
SETBIT/GETBIT agreed with each other, and BITCOUNT is a popcount so it cannot
see the order at all. It only shows up when the bytes LEAVE the bitmap
commands — a GET on the key, a snapshot read by another tool, or the standard
Bloom-filter idiom of building a bitmap with SETBIT and parsing it as a string.

Four separate defects are pinned here, each of which returned a plausible
number rather than an error:

  bit order    SETBIT/GETBIT/BITPOS/BITFIELD were all LSB-first
  sign         BITFIELD GET i16 of -1234 returned 64302
  wrap         BITFIELD INCRBY u16 past 65535 returned 65536, a value the
               field cannot hold and that a following GET does not return
  allocation   a missing key became a fixed 8-byte bitmap, so a 1-byte SET
               produced an 8-byte string and a read-only GET CREATED the key

Run:  python3 tests/test_gh232_bitmap.py [--port 1974] [--redis-port 6396]
"""
import argparse
import socket
import subprocess
import sys
import time


def client(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    f = s.makefile("rb")

    def read():
        line = f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        t, body = line[:1], line[1:-2]
        if t == b"$":
            n = int(body)
            return None if n == -1 else f.read(n + 2)[:-2]
        if t == b"*":
            n = int(body)
            return None if n == -1 else [read() for _ in range(n)]
        return (t + body).decode()

    def cmd(*args):
        enc = [str(a).encode() for a in args]
        s.sendall(b"*%d\r\n" % len(enc) +
                  b"".join(b"$%d\r\n%s\r\n" % (len(e), e) for e in enc))
        return read()

    return cmd


def probes():
    out = []
    # Bit order across byte boundaries and into a sparse high offset.
    for off in (0, 1, 2, 6, 7, 8, 9, 15, 16, 63, 100):
        out += [("DEL", "z"), ("SETBIT", "z", off, 1), ("GET", "z"),
                ("BITCOUNT", "z"), ("GETBIT", "z", off),
                ("GETBIT", "z", off + 1), ("BITPOS", "z", 1), ("BITPOS", "z", 0)]
    # Two bits in one byte: order is visible in the byte value itself.
    out += [("DEL", "z"), ("SETBIT", "z", 1, 1), ("SETBIT", "z", 6, 1),
            ("GET", "z"), ("BITPOS", "z", 1), ("BITCOUNT", "z")]
    # BITFIELD: signedness, widths, unaligned offsets, wrap-on-overflow.
    for ty, val in [("u8", 200), ("i8", -5), ("i8", 127), ("u4", 9),
                    ("i16", -1234), ("i32", -70000), ("u16", 65535),
                    ("i4", -3), ("u1", 1), ("i64", -9007199254740993)]:
        for off in (0, 3, 8, 13):
            out += [("DEL", "f"),
                    ("BITFIELD", "f", "SET", ty, off, val),
                    ("BITFIELD", "f", "GET", ty, off),
                    ("GET", "f"), ("STRLEN", "f"),
                    ("BITFIELD_RO", "f", "GET", ty, off),
                    ("BITFIELD", "f", "INCRBY", ty, off, 1),
                    ("BITFIELD", "f", "GET", ty, off),
                    ("BITFIELD", "f", "INCRBY", ty, off, -2),
                    ("BITFIELD", "f", "GET", ty, off)]
    # A read-only BITFIELD must not create the key.
    out += [("DEL", "ro"), ("BITFIELD", "ro", "GET", "u8", 0), ("EXISTS", "ro")]
    # Multiple subcommands in one BITFIELD.
    out += [("DEL", "m"),
            ("BITFIELD", "m", "SET", "u8", 0, 1, "SET", "u8", 8, 2,
             "GET", "u8", 0, "GET", "u8", 8),
            ("GET", "m"), ("STRLEN", "m")]
    # ── §4: a bitmap IS a string, in both directions ────────────────────
    # Redis has no separate bitmap type. Pion modelled one, so the textbook
    # idioms answered WRONGTYPE: `SET k "hello"; BITCOUNT k` and the
    # build-with-SETBIT-then-read-as-a-string flow. Read-only ops only —
    # SETBIT on a string still refuses, deliberately: setbit() frees and
    # reallocs when it grows, and a STRING payload may live in the gh #163
    # blob arena, which the heap allocator must never free.
    #
    # Both GenericValue string shapes are exercised. STRING_SSO (<=23 bytes)
    # keeps its bytes INSIDE the value, so its _data0 is a length-and-chars
    # word, not an address — reading it as a bitmap pointer would dereference
    # packed characters. The 23/24-byte pair straddles that boundary.
    for text in ("hello", "a", "x" * 23, "y" * 24, "hello world, a longer heap string"):
        out += [("DEL", "sb"), ("SET", "sb", text),
                ("BITCOUNT", "sb"), ("GETBIT", "sb", 0), ("GETBIT", "sb", 6),
                ("GETBIT", "sb", 8 * len(text) - 1),
                ("GETBIT", "sb", 8 * len(text)),      # past the end -> 0
                ("BITPOS", "sb", 1), ("STRLEN", "sb")]
    # The other direction: a key built by SETBIT must answer string commands.
    for off in (0, 7, 10, 100):
        out += [("DEL", "bs"), ("SETBIT", "bs", off, 1),
                ("STRLEN", "bs"), ("GETRANGE", "bs", 0, -1),
                ("GETRANGE", "bs", 0, 0), ("SUBSTR", "bs", 0, -1),
                ("GET", "bs"), ("TYPE", "bs")]
    # Negative: broadening the type check must NOT let a real aggregate
    # through. Over-permissiveness is the worse bug — it reports "empty"
    # where the caller has a type error.
    out += [("DEL", "agg"), ("RPUSH", "agg", "a"),
            ("BITCOUNT", "agg"), ("GETBIT", "agg", 0), ("STRLEN", "agg"),
            ("GETRANGE", "agg", 0, -1)]
    out += [("DEL", "agg"), ("HSET", "agg", "f", "v"),
            ("BITCOUNT", "agg"), ("GETBIT", "agg", 0), ("STRLEN", "agg")]
    out += [("DEL", "agg"), ("SADD", "agg", "m"),
            ("BITCOUNT", "agg"), ("GETBIT", "agg", 0), ("STRLEN", "agg")]
    out += [("DEL", "agg"), ("ZADD", "agg", 1, "m"),
            ("BITCOUNT", "agg"), ("GETBIT", "agg", 0), ("STRLEN", "agg")]

    out.append(("PING",))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--redis-port", type=int, default=6396)
    args = ap.parse_args()

    redis = subprocess.Popen(
        ["redis-server", "--port", str(args.redis_port), "--save", "",
         "--appendonly", "no"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    oracle = None
    for _ in range(50):
        try:
            oracle = client(args.redis_port)
            break
        except OSError:
            time.sleep(0.1)
    if oracle is None:
        print("could not start redis-server on port %d" % args.redis_port)
        return 2
    try:
        pion = client(args.port)
    except OSError as e:
        print("cannot reach Pion on port %d: %s" % (args.port, e))
        redis.terminate()
        return 2

    # STRLEN on a bitmap is a KNOWN, deliberate divergence: Redis models
    # bitmaps as strings, Pion as a distinct type. Documented on gh #232 and
    # not in scope here — this test is about bit order and BITFIELD numerics.
    def known_divergence(probe, a, b):
        return (probe[0] in ("STRLEN",)
                and isinstance(a, str) and a.startswith("-WRONGTYPE"))

    # Start from the same state on both. Pion is typically a long-lived server
    # that other tests have already written to, while the oracle is fresh, so
    # the FIRST `DEL` of each key reports 1 on one side and 0 on the other —
    # a difference that is about the fixture, not the code.
    for k in ("z", "f", "ro", "m"):
        pion("DEL", k)
        oracle("DEL", k)

    diffs = []
    checked = 0
    for probe in probes():
        got, want = pion(*probe), oracle(*probe)
        checked += 1
        if got != want and not known_divergence(probe, got, want):
            diffs.append((probe, got, want))

    redis.terminate()
    redis.wait(timeout=5)

    if diffs:
        print("gh #232 bitmap: %d/%d agree with Redis, %d differ"
              % (checked - len(diffs), checked, len(diffs)))
        for probe, got, want in diffs[:25]:
            print("\n  %s" % " ".join(map(str, probe)))
            print("    redis: %r" % (want,))
            print("    pion : %r" % (got,))
        if len(diffs) > 25:
            print("\n  ... and %d more" % (len(diffs) - 25))
        return 1

    print("gh #232 bitmap: %d/%d agree with Redis" % (checked, checked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
