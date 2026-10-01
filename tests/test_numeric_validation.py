#!/usr/bin/env python3
"""Numeric argument validation, Int64-MIN formatting, and absolute-expiry bounds.

Three defects that all live in "the server turned a number into something else":

  1. Integer/float ARGUMENTS were parsed by loops that accumulated digits and
     SKIPPED everything else, with no overflow check. `INCRBY k 1abc2` added 12;
     `INCRBY k abc` added 0 and replied with an integer, so a client whose delta
     stringified to garbage saw a successful-looking reply and an unmoved
     counter. The stored-value parse in the same handlers was already strict —
     only the argument parse was not.

  2. `format_int_to_buf` negated with `n = -n`, which is a NO-OP for Int64 MIN.
     A value that reached exactly -9223372036854775808 was reported as 0 and
     PERSISTED as the string "-0".

  3. `EXPIREAT`/`PEXPIREAT` multiplied an unbounded epoch by 1e9/1e6, wrapping
     past ~2262, and never honoured the "epoch in the past deletes the key"
     rule.

Every assertion is a value comparison, not a substring check: the failure mode
being guarded against is a *plausible* wrong answer, so "did it error" is not
enough — the resulting stored value has to be right too.

Usage: python3 tests/test_numeric_validation.py [--port 1974]
"""

import argparse
import socket
import sys
import time

HOST = "127.0.0.1"
INT64_MIN = "-9223372036854775808"
INT64_MAX = "9223372036854775807"
PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append((name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}")


class Conn:
    def __init__(self, port, timeout=10):
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
            return None if n == -1 else self.f.read(n + 2)[:-2].decode()
        if t == b"*":
            n = int(body)
            return [] if n <= 0 else [self._read() for _ in range(n)]
        if t == b"_":
            return None
        return body.decode(errors="replace")


def is_err(r):
    return isinstance(r, str) and r.startswith("ERR:")


# Arguments Redis rejects. The first three are the dangerous ones: they used to
# parse to 0, so the reply looked like a success.
BAD_INTS = ["abc", "1abc2", "", "-", "1e3", "0x10", "  5", "5 ", "+5", "1,000",
            "1.5", "99999999999999999999", "18446744073709551616",
            "-99999999999999999999", "9223372036854775808"]
# NOT "0x1": Redis parses float arguments with strtold, which reads hex —
# redis-server 8.10 answers `SET k 7; INCRBYFLOAT k 0x1` with 8 (checked
# 2026-09-29, gh #393). Rejecting it was stricter than the oracle.
BAD_FLOATS = ["abc", "", "1abc2", "1.2.3", "e5", ".", "--1", "1e"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()
    try:
        c = Conn(args.port)
    except OSError as e:
        print(f"FATAL: no server on {HOST}:{args.port} ({e})")
        return 2

    print("[1] Integer arguments: garbage must ERROR and leave the value alone")
    for bad in BAD_INTS:
        for cmd_name in ("INCRBY", "DECRBY"):
            c.cmd("DEL", "nv:i")
            c.cmd("SET", "nv:i", "7")
            r = c.cmd(cmd_name, "nv:i", bad)
            after = c.cmd("GET", "nv:i")
            check(f"{cmd_name} {bad!r}", is_err(r) and after == "7",
                  "" if is_err(r) else f"accepted -> {r}, value now {after!r}")
        c.cmd("DEL", "nv:h")
        c.cmd("HSET", "nv:h", "f", "7")
        r = c.cmd("HINCRBY", "nv:h", "f", bad)
        after = c.cmd("HGET", "nv:h", "f")
        check(f"HINCRBY {bad!r}", is_err(r) and after == "7",
              "" if is_err(r) else f"accepted -> {r}, value now {after!r}")

    print("\n[2] Float arguments: same rule")
    for bad in BAD_FLOATS:
        c.cmd("DEL", "nv:f")
        c.cmd("SET", "nv:f", "7")
        r = c.cmd("INCRBYFLOAT", "nv:f", bad)
        after = c.cmd("GET", "nv:f")
        check(f"INCRBYFLOAT {bad!r}", is_err(r) and after == "7",
              "" if is_err(r) else f"accepted -> {r}, value now {after!r}")
        c.cmd("DEL", "nv:hf")
        c.cmd("HSET", "nv:hf", "f", "7")
        r = c.cmd("HINCRBYFLOAT", "nv:hf", "f", bad)
        check(f"HINCRBYFLOAT {bad!r}", is_err(r),
              "" if is_err(r) else f"accepted -> {r}")

    print("\n[3] Legitimate arithmetic is unchanged")
    c.cmd("DEL", "nv:ok")
    c.cmd("SET", "nv:ok", "10")
    for cmd_args, want in [(("INCRBY", "nv:ok", "5"), 15),
                           (("DECRBY", "nv:ok", "3"), 12),
                           (("INCRBY", "nv:ok", "-4"), 8),
                           (("DECRBY", "nv:ok", "-2"), 10)]:
        got = c.cmd(*cmd_args)
        check(f"{' '.join(cmd_args[::2] if False else cmd_args)}", got == want, f"got {got}")
    c.cmd("DEL", "nv:new")
    check("INCRBY on a missing key", c.cmd("INCRBY", "nv:new", "7") == 7)
    c.cmd("DEL", "nv:max")
    c.cmd("SET", "nv:max", "0")
    check("INCRBY Int64 MAX", c.cmd("INCRBY", "nv:max", INT64_MAX) == int(INT64_MAX))
    c.cmd("DEL", "nv:flt")
    c.cmd("SET", "nv:flt", "10.5")
    # gh #232 widened INCRBYFLOAT from Float32 to Float64, so this no longer
    # rounds to a tidy "10.6" — it returns 10.59999999999999964, which is what
    # real redis-server 8.10 returns for the same input, verified byte-for-byte.
    # The old "10.6" was the Float32 artifact, not the correct answer, so assert
    # the VALUE and let the representation be whatever IEEE754 doubles give.
    _f = c.cmd("INCRBYFLOAT", "nv:flt", "0.1")
    check("INCRBYFLOAT 0.1", abs(float(_f) - 10.6) < 1e-12, f"got {_f}")
    c.cmd("DEL", "nv:exp")
    c.cmd("SET", "nv:exp", "1")
    check("INCRBYFLOAT exponent 3e2", c.cmd("INCRBYFLOAT", "nv:exp", "3e2") == "301")

    print("\n[4] Overflow of the RESULT stays guarded (this always worked)")
    c.cmd("DEL", "nv:o")
    c.cmd("SET", "nv:o", INT64_MAX)
    check("INCR at Int64 MAX errors", is_err(c.cmd("INCR", "nv:o")))
    check("INCRBY 1 at Int64 MAX errors", is_err(c.cmd("INCRBY", "nv:o", "1")))

    print("\n[5] A non-numeric STORED value is refused")
    c.cmd("DEL", "nv:s")
    c.cmd("SET", "nv:s", "hello")
    check("INCRBY on a non-numeric string", is_err(c.cmd("INCRBY", "nv:s", "1")))
    c.cmd("DEL", "nv:hs")
    c.cmd("HSET", "nv:hs", "f", "abc")
    r = c.cmd("HINCRBY", "nv:hs", "f", "1")
    check("HINCRBY on a non-numeric field", is_err(r),
          "" if is_err(r) else f"accepted -> {r}")

    print("\n[6] Int64 MIN survives the formatter (reached by ARITHMETIC)")
    c.cmd("DEL", "nv:m")
    c.cmd("SET", "nv:m", "-9223372036854775807")
    check("DECRBY 1 -> Int64 MIN", c.cmd("DECRBY", "nv:m", "1") == int(INT64_MIN))
    check("GET returns Int64 MIN, not '-0'", c.cmd("GET", "nv:m") == INT64_MIN)
    check("STRLEN of Int64 MIN is 20", c.cmd("STRLEN", "nv:m") == 20)
    c.cmd("DEL", "nv:m2")
    c.cmd("SET", "nv:m2", "0")
    check("INCRBY Int64 MIN onto 0", c.cmd("INCRBY", "nv:m2", INT64_MIN) == int(INT64_MIN))
    c.cmd("DEL", "nv:m3")
    c.cmd("HSET", "nv:m3", "f", "-9223372036854775807")
    check("HINCRBY -1 -> Int64 MIN", c.cmd("HINCRBY", "nv:m3", "f", "-1") == int(INT64_MIN))
    check("HGET returns Int64 MIN", c.cmd("HGET", "nv:m3", "f") == INT64_MIN)

    print("\n[7] Ordinary integers round-trip (format_int_to_buf is on every reply)")
    for v in [0, 1, -1, 9, -9, 10, -10, 99, -99, 100, -100, 12345, -12345,
              10**18, -(10**18), int(INT64_MAX), int(INT64_MAX) * -1]:
        c.cmd("DEL", "nv:r")
        c.cmd("SET", "nv:r", str(v))
        check(f"round-trip {v}", c.cmd("GET", "nv:r") == str(v))

    print("\n[8] Relative expiry: non-positive deletes, Redis's overflow rule")
    for secs in ["-1", "0"]:
        c.cmd("DEL", "nv:t")
        c.cmd("SET", "nv:t", "v")
        r = c.cmd("EXPIRE", "nv:t", secs)
        check(f"EXPIRE {secs} deletes the key",
              r == 1 and c.cmd("EXISTS", "nv:t") == 0, f"reply {r}")
    # Redis's own overflow rule (gh #393): the seconds must scale to ms and the
    # sum with now must fit in ms. Past that it is an error and the key stays.
    for secs in [str(10**18), "-9223372036854775808"]:
        c.cmd("DEL", "nv:t")
        c.cmd("SET", "nv:t", "v")
        r = c.cmd("EXPIRE", "nv:t", secs)
        check(f"EXPIRE {secs} errors, key untouched",
              is_err(r) and c.cmd("EXISTS", "nv:t") == 1, f"reply {r}")
    # Inside that rule Redis sets the TTL. Pion's ns deadline ends in 2262, so
    # it stores the latest one it can — never a WRAPPED one (0.915 answered
    # TTL -2 on a present key here, and a TTL of ~246 y for ~3171 y).
    for secs in ["9300000000", "99999999999"]:
        c.cmd("DEL", "nv:t")
        c.cmd("SET", "nv:t", "v")
        r = c.cmd("EXPIRE", "nv:t", secs)
        ttl = c.cmd("TTL", "nv:t")
        check(f"EXPIRE {secs} accepted, TTL saturated past 200 y, key kept",
              r == 1 and ttl > 200 * 365 * 86400 and c.cmd("EXISTS", "nv:t") == 1,
              f"reply {r}, TTL {ttl}")
    c.cmd("DEL", "nv:t")
    c.cmd("SET", "nv:t", "v")
    c.cmd("EXPIRE", "nv:t", "100")
    check("EXPIRE 100 still sets a TTL", c.cmd("TTL", "nv:t") in (99, 100))
    c.cmd("DEL", "nv:t2")
    c.cmd("SET", "nv:t2", "v")
    check("EXPIRE 9000000000 (285y) still allowed",
          c.cmd("EXPIRE", "nv:t2", "9000000000") == 1)

    print("\n[9] Absolute expiry: past deletes, Redis's overflow rule")
    now = int(time.time())
    for ep in [now - 100, 1, -5, -99999999999]:
        c.cmd("DEL", "nv:a")
        c.cmd("SET", "nv:a", "v")
        r = c.cmd("EXPIREAT", "nv:a", str(ep))
        check(f"EXPIREAT past epoch {ep} deletes",
              r == 1 and c.cmd("EXISTS", "nv:a") == 0, f"reply {r}")
    for ep in [str(10**18), "-9223372036854775808"]:
        c.cmd("DEL", "nv:a")
        c.cmd("SET", "nv:a", "v")
        r = c.cmd("EXPIREAT", "nv:a", ep)
        check(f"EXPIREAT {ep} errors, key untouched",
              is_err(r) and c.cmd("EXISTS", "nv:a") == 1, f"reply {r}")
    for ep in ["9300000000", "99999999999"]:
        c.cmd("DEL", "nv:a")
        c.cmd("SET", "nv:a", "v")
        r = c.cmd("EXPIREAT", "nv:a", ep)
        ttl = c.cmd("TTL", "nv:a")
        check(f"EXPIREAT {ep} accepted, TTL saturated past 200 y",
              r == 1 and ttl > 200 * 365 * 86400, f"reply {r}, TTL {ttl}")
    c.cmd("DEL", "nv:a2")
    c.cmd("SET", "nv:a2", "v")
    r = c.cmd("EXPIREAT", "nv:a2", str(now + 100))
    ttl = c.cmd("TTL", "nv:a2")
    check("EXPIREAT future epoch sets the right TTL",
          r == 1 and 90 <= ttl <= 100, f"reply {r}, TTL {ttl}")
    c.cmd("DEL", "nv:p")
    c.cmd("SET", "nv:p", "v")
    r = c.cmd("PEXPIREAT", "nv:p", str((now - 100) * 1000))
    check("PEXPIREAT past epoch deletes",
          r == 1 and c.cmd("EXISTS", "nv:p") == 0, f"reply {r}")
    c.cmd("DEL", "nv:p2")
    c.cmd("SET", "nv:p2", "v")
    r = c.cmd("PEXPIREAT", "nv:p2", "9223372036854775807")
    ttl = c.cmd("TTL", "nv:p2")
    check("PEXPIREAT at Int64 max is a legal deadline (Redis: 1)",
          r == 1 and ttl > 200 * 365 * 86400, f"reply {r}, TTL {ttl}")
    c.cmd("DEL", "nv:p3")
    c.cmd("SET", "nv:p3", "v")
    r = c.cmd("PEXPIREAT", "nv:p3", str((now + 100) * 1000))
    ttl = c.cmd("TTL", "nv:p3")
    check("PEXPIREAT future epoch sets the right TTL",
          r == 1 and 90 <= ttl <= 100, f"reply {r}, TTL {ttl}")

    print("\n[10] Server still framed correctly")
    check("PING after everything", c.cmd("PING") == "PONG")
    for k in ["nv:i", "nv:h", "nv:f", "nv:hf", "nv:ok", "nv:new", "nv:max",
              "nv:flt", "nv:exp", "nv:o", "nv:s", "nv:hs", "nv:m", "nv:m2",
              "nv:m3", "nv:r", "nv:t", "nv:t2", "nv:a", "nv:a2", "nv:p",
              "nv:p2", "nv:p3"]:
        c.cmd("DEL", k)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name, detail in FAILED:
        print(f"  FAILED: {name} {detail}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
