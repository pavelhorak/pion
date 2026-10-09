#!/usr/bin/env python3
"""gh #232 — XADD destroying keys, WRONGTYPE conflation, and INCR-family codes.

Three defect families, all of which returned something that LOOKED like success:

1. **XADD replaced any key with a new stream.** `get_or_create_stream` `set()`
   a fresh stream whenever the key held anything that was not one, so
   `SET k hello; XADD k * f v` answered with an id and the string was GONE —
   `GET k` then said WRONGTYPE and the heap payload was leaked, not freed.
   XADD was the ONLY create-if-missing command with this hole; RPUSH, LPUSH,
   SADD, HSET, ZADD, PFADD, GEOADD, SETBIT, APPEND and SETRANGE were probed
   against every other type and already refused. Both directions are pinned
   below, because "every creator now errors" would be a worse bug.

   `NOMKSTREAM` came with it: the arm tested `tl == 11` and the flag is 10
   bytes, so it never matched and fell out of the loop as the ID token. The
   literal text was parsed into an id (33420896299-0), every field/value pair
   shifted by one, and the last value was dropped.

2. **Missing key and wrong-type key gave the same answer** in ten commands
   (HGET LPOP RPOP LRANGE SCARD SPOP XLEN XRANGE ZCOUNT ZPOPMIN). The
   conflation runs in the dangerous direction: a caller who stored the wrong
   kind of value is told the container is empty, so the bug presents as
   missing data instead of a type error at the call site. The missing-key
   replies are asserted too — over-correcting to WRONGTYPE everywhere would
   break every "read a key that isn't there yet" caller.

3. **INCR/INCRBY/DECRBY answered ERR where Redis answers WRONGTYPE**, and
   clients branch on the code (redis-py raises a different exception class
   for each). Redis validates the ARGUMENT first — `INCRBY <list> abc` is
   ERR, `INCRBY <list> 1` is WRONGTYPE — and that order is observable, so it
   is pinned. BITMAP and HLL must stay ERR: Redis stores both as strings, so
   `SETBIT k 7 1; INCR k` really is a not-an-integer error there.

   `INCRBYFLOAT` had the same shape one level down: it validated its ARGUMENT
   strictly but parsed the STORED value leniently, so "hello" read as 0.0 and
   the key was overwritten with the delta. It also did its arithmetic in
   Float32, whose 24-bit mantissa cannot represent 100000001 — so a counter
   past ~16.7M stopped incrementing while still replying success.

Run:  python3 tests/test_gh232_wrongtype.py [--port 1974]

With a real redis-server on PATH the numeric/format expectations below are
additionally confirmed against it (`--redis-port`, default: spawn one).
"""
import argparse
import ctypes
import socket
import subprocess
import sys
import time

WRONGTYPE = "-WRONGTYPE Operation against a key holding the wrong kind of value"

PASS = 0
FAIL = 0


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
        enc = [str(a).encode() if not isinstance(a, bytes) else a for a in args]
        s.sendall(b"*%d\r\n" % len(enc) +
                  b"".join(b"$%d\r\n%s\r\n" % (len(e), e) for e in enc))
        return read()

    return cmd


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got!r}\n          want {want!r}")
    return ok


def is_wrongtype(label, got):
    global PASS, FAIL
    ok = isinstance(got, str) and got.startswith("-WRONGTYPE")
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got {got!r}, wanted a -WRONGTYPE error")
    return ok


def is_err(label, got):
    """Plain -ERR, and specifically NOT -WRONGTYPE — the whole point is the code."""
    global PASS, FAIL
    ok = isinstance(got, str) and got.startswith("-ERR")
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got {got!r}, wanted a plain -ERR")
    return ok


# Seeds a key of each type, and the read-back that proves it SURVIVED.
SEEDS = {
    "string": (lambda c, k: c("SET", k, "hello"), lambda c, k: c("GET", k), b"hello"),
    "list":   (lambda c, k: c("RPUSH", k, "a", "b", "c"),
               lambda c, k: c("LRANGE", k, 0, -1), [b"a", b"b", b"c"]),
    "hash":   (lambda c, k: c("HSET", k, "f", "v"), lambda c, k: c("HGET", k, "f"), b"v"),
    "set":    (lambda c, k: c("SADD", k, "m1"), lambda c, k: c("SMEMBERS", k), [b"m1"]),
    "zset":   (lambda c, k: c("ZADD", k, 1, "m"), lambda c, k: c("ZRANGE", k, 0, -1), [b"m"]),
    "stream": (lambda c, k: c("XADD", k, "*", "f", "v"), lambda c, k: c("XLEN", k), ":1"),
}


def section_xadd_does_not_destroy(c):
    print("\n=== 1. XADD must not replace a key of another type ===")
    for tname, (seed, readback, expect) in SEEDS.items():
        if tname == "stream":
            continue
        k = f"gh232:xadd:{tname}"
        c("DEL", k)
        seed(c, k)
        is_wrongtype(f"XADD onto a {tname} is refused", c("XADD", k, "*", "f", "v"))
        check(f"  ...and the {tname} survived", readback(c, k), expect)
        check(f"  ...and TYPE is still {tname}", c("TYPE", k), f"+{tname}")

    print("\n  -- the other direction: XADD on a real stream still works --")
    k = "gh232:xadd:ok"
    c("DEL", k)
    rid = c("XADD", k, "*", "f", "v")
    check("XADD on a missing key creates the stream", isinstance(rid, bytes), True)
    check("  ...XLEN is 1", c("XLEN", k), ":1")
    c("XADD", k, "*", "f2", "v2")
    check("  ...XLEN is 2 after a second add", c("XLEN", k), ":2")
    c("DEL", k)
    for i in range(6):
        c("XADD", k, "MAXLEN", 3, "*", "i", i)
    check("MAXLEN still trims", c("XLEN", k), ":3")


def section_nomkstream(c):
    print("\n=== 2. NOMKSTREAM ===")
    k = "gh232:nomk:missing"
    c("DEL", k)
    check("missing key replies nil", c("XADD", k, "NOMKSTREAM", "*", "f", "v"), None)
    check("  ...and the key was NOT created", c("EXISTS", k), ":0")

    k = "gh232:nomk:exists"
    c("DEL", k)
    c("XADD", k, "*", "a", "1")
    c("XADD", k, "NOMKSTREAM", "*", "f", "v")
    entries = c("XRANGE", k, "-", "+") or []
    check("existing stream appends", len(entries), 2)
    # The flag used to be parsed as the ID, which shifted the pairs and dropped
    # the last value: the entry came back as field '*' = 'f', with 'v' gone.
    # Indexed defensively — on the buggy code this list is short, and a test
    # that raises there hides every result after it.
    check("  ...field/value pairs are NOT shifted",
          entries[1][1] if len(entries) > 1 else None, [b"f", b"v"])

    k = "gh232:nomk:wrongtype"
    c("DEL", k)
    c("SET", k, "x")
    is_wrongtype("wrong-type key is refused", c("XADD", k, "NOMKSTREAM", "*", "f", "v"))
    check("  ...and the string survived", c("GET", k), b"x")


# (command, args) -> the reply a MISSING key must still produce.
PERMISSIVE = [
    (("HGET", "%K", "f"), None),
    (("LPOP", "%K"), None),
    (("RPOP", "%K"), None),
    (("LRANGE", "%K", 0, -1), []),
    (("SCARD", "%K"), ":0"),
    (("SPOP", "%K"), None),
    (("XLEN", "%K"), ":0"),
    (("XRANGE", "%K", "-", "+"), []),
    (("ZCOUNT", "%K", "-inf", "+inf"), ":0"),
    (("ZPOPMIN", "%K"), []),
]

# Which seeded type each command legitimately operates on — skip that pairing.
NATIVE = {"HGET": "hash", "LPOP": "list", "RPOP": "list", "LRANGE": "list",
          "SCARD": "set", "SPOP": "set", "XLEN": "stream", "XRANGE": "stream",
          "ZCOUNT": "zset", "ZPOPMIN": "zset"}


def section_wrongtype_split(c):
    print("\n=== 3. missing key vs wrong-type key are different answers ===")
    for probe, _missing in PERMISSIVE:
        name = probe[0]
        for tname, (seed, readback, expect) in SEEDS.items():
            if NATIVE[name] == tname:
                continue
            k = f"gh232:wt:{name}:{tname}"
            c("DEL", k)
            seed(c, k)
            args = [k if a == "%K" else a for a in probe]
            is_wrongtype(f"{name} on a {tname}", c(*args))
            check(f"  ...{tname} survived {name}", readback(c, k), expect)

    print("\n  -- the other direction: a MISSING key keeps its old reply --")
    for probe, missing in PERMISSIVE:
        name = probe[0]
        k = f"gh232:miss:{name}"
        c("DEL", k)
        args = [k if a == "%K" else a for a in probe]
        check(f"{name} on a missing key", c(*args), missing)


def section_incr_family(c):
    print("\n=== 4. INCR/INCRBY/DECRBY error CODE ===")
    for tname, (seed, readback, expect) in SEEDS.items():
        if tname == "string":
            continue                      # a string is a value error, not a type error
        k = f"gh232:incr:{tname}"
        c("DEL", k)
        seed(c, k)
        is_wrongtype(f"INCR on a {tname}", c("INCR", k))
        is_wrongtype(f"INCRBY on a {tname}", c("INCRBY", k, 1))
        is_wrongtype(f"DECRBY on a {tname}", c("DECRBY", k, 1))
        check(f"  ...{tname} survived", readback(c, k), expect)

    print("\n  -- the ARGUMENT is validated before the type (Redis order) --")
    k = "gh232:incr:order"
    c("DEL", k)
    c("RPUSH", k, "a")
    is_err("INCRBY <list> abc is ERR, not WRONGTYPE", c("INCRBY", k, "abc"))
    is_wrongtype("INCRBY <list> 1 is WRONGTYPE", c("INCRBY", k, 1))

    print("\n  -- bitmap and HLL are STRINGS in Redis, so they stay ERR --")
    k = "gh232:incr:bitmap"
    c("DEL", k); c("SETBIT", k, 7, 1)
    is_err("INCR on a bitmap is ERR", c("INCR", k))
    k = "gh232:incr:hll"
    c("DEL", k); c("PFADD", k, "a")
    is_err("INCR on an HLL is ERR", c("INCR", k))

    print("\n  -- a float-valued string is a value error, not a type error --")
    k = "gh232:incr:float"
    c("DEL", k); c("SET", k, "1.5")
    is_err("INCR on '1.5' is ERR", c("INCR", k))

    print("\n  -- the other direction: counters still count --")
    k = "gh232:incr:ok"
    c("DEL", k)
    check("INCR from missing", c("INCR", k), ":1")
    check("INCRBY 5", c("INCRBY", k, 5), ":6")
    check("DECRBY 2", c("DECRBY", k, 2), ":4")
    c("DEL", k); c("SET", k, "10")
    check("INCR on a numeric string", c("INCR", k), ":11")


def section_incrbyfloat(c):
    print("\n=== 5. INCRBYFLOAT: stored-value validation and Float64 range ===")
    k = "gh232:ibf:str"
    c("DEL", k); c("SET", k, "hello")
    is_err("INCRBYFLOAT on a non-numeric string errors", c("INCRBYFLOAT", k, "1.5"))
    check("  ...and the string was NOT overwritten", c("GET", k), b"hello")

    for tname, (seed, readback, expect) in SEEDS.items():
        if tname == "string":
            continue
        k = f"gh232:ibf:{tname}"
        c("DEL", k)
        seed(c, k)
        is_wrongtype(f"INCRBYFLOAT on a {tname}", c("INCRBYFLOAT", k, "1.5"))
        check(f"  ...{tname} survived", readback(c, k), expect)

    print("\n  -- Float32 could not hold 100000001, so the increment vanished --")
    k = "gh232:ibf:big"
    c("DEL", k); c("SET", k, "100000000")
    check("1e8 + 1 actually increments", c("INCRBYFLOAT", k, "1"), b"100000001")
    c("DEL", k); c("SET", k, "1000000000000")
    check("1e12 + 1 actually increments", c("INCRBYFLOAT", k, "1"), b"1000000000001")

    print("\n  -- and the stored value round-trips instead of being truncated --")
    k = "gh232:ibf:pi"
    c("DEL", k); c("SET", k, "3.14159265358979")
    # Redis computes in long double and prints %.17Lg, so a value a Float32
    # would truncate comes back with its digits. Pion matches Redis byte for
    # byte here (§6 asks a real Redis the same probes). The point is the digits
    # survive, not that they are clean, and what %.17Lg prints depends on the
    # platform's long double: where it is just a double (macOS arm64) the
    # double's own trailing artifact shows, 3.14159265358979001; where it is
    # wider (x86-64's 80-bit, aarch64 Linux's 128-bit) the value prints clean.
    # Either way, not the Float32 truncation 3.141592.
    pi0 = b"3.14159265358979001" if ctypes.sizeof(ctypes.c_longdouble) == 8 else b"3.14159265358979"
    check("pi + 0 keeps its digits", c("INCRBYFLOAT", k, "0"), pi0)

    print("\n  -- the other direction: ordinary arithmetic still works --")
    k = "gh232:ibf:ok"
    c("DEL", k)
    check("missing + 3.0", c("INCRBYFLOAT", k, "3.0"), b"3")
    check("3 + 2.5", c("INCRBYFLOAT", k, "2.5"), b"5.5")
    check("5.5 - 5.5", c("INCRBYFLOAT", k, "-5.5"), b"0")
    is_err("a non-numeric DELTA still errors", c("INCRBYFLOAT", k, "abc"))


def section_lmove(c):
    """LMOVE / RPOPLPUSH must be a NO-OP when they refuse.

    Found by extending the wrong-type sweep on 2026-08-25. Both popped from the
    source BEFORE checking the destination's type, so with a wrong-type
    destination the element was removed from the source and never added
    anywhere — destroyed. LMOVE additionally wrote the popped element as a
    reply and THEN appended the error, so one command produced TWO replies and
    desynced the connection for everything after it.

    RPOPLPUSH is the reliable-queue primitive (`RPOPLPUSH work processing`), so
    there the failure is a job vanishing behind a reply that says it moved.
    """
    print("\n=== 5b. LMOVE / RPOPLPUSH refuse without destroying (gh #232) ===")
    for name, mv in (("LMOVE", ("LMOVE", "gh232:mv:src", "gh232:mv:dst", "LEFT", "RIGHT")),
                     ("RPOPLPUSH", ("RPOPLPUSH", "gh232:mv:src", "gh232:mv:dst"))):
        for tname, seed in (("string", ("SET", "gh232:mv:dst", "occupied")),
                            ("hash", ("HSET", "gh232:mv:dst", "f", "v")),
                            ("set", ("SADD", "gh232:mv:dst", "m")),
                            ("zset", ("ZADD", "gh232:mv:dst", "1", "m"))):
            c("DEL", "gh232:mv:src"); c("DEL", "gh232:mv:dst")
            c("RPUSH", "gh232:mv:src", "a", "b", "c")
            c(*seed)
            is_wrongtype(f"{name} onto a {tname} destination", c(*mv))
            check(f"  ...source untouched", c("LRANGE", "gh232:mv:src", 0, -1),
                  [b"a", b"b", b"c"])

    print("\n  -- a MISSING source is nil, and the destination is never examined --")
    for name, mv in (("LMOVE", ("LMOVE", "gh232:mv:none", "gh232:mv:dst", "LEFT", "RIGHT")),
                     ("RPOPLPUSH", ("RPOPLPUSH", "gh232:mv:none", "gh232:mv:dst"))):
        c("DEL", "gh232:mv:none"); c("DEL", "gh232:mv:dst")
        c("SET", "gh232:mv:dst", "occupied")
        check(f"{name} with a missing source", c(*mv), None)

    print("\n  -- the other direction: a legal move still moves --")
    c("DEL", "gh232:mv:src"); c("DEL", "gh232:mv:dst")
    c("RPUSH", "gh232:mv:src", "a", "b", "c")
    check("LMOVE LEFT RIGHT returns the head", c("LMOVE", "gh232:mv:src", "gh232:mv:dst",
                                                 "LEFT", "RIGHT"), b"a")
    check("  ...source shrank", c("LRANGE", "gh232:mv:src", 0, -1), [b"b", b"c"])
    check("  ...destination got it", c("LRANGE", "gh232:mv:dst", 0, -1), [b"a"])
    check("RPOPLPUSH returns the tail", c("RPOPLPUSH", "gh232:mv:src", "gh232:mv:dst"), b"c")
    check("  ...destination head-pushed", c("LRANGE", "gh232:mv:dst", 0, -1), [b"c", b"a"])
    c("DEL", "gh232:mv:rot"); c("RPUSH", "gh232:mv:rot", "a", "b", "c")
    check("LMOVE k k rotates", c("LMOVE", "gh232:mv:rot", "gh232:mv:rot", "LEFT", "RIGHT"), b"a")
    check("  ...list rotated", c("LRANGE", "gh232:mv:rot", 0, -1), [b"b", b"c", b"a"])


def section_value_bugs(c):
    """Right type, right operation, WRONG ANSWER — the class no type sweep sees."""
    print("\n=== 5c. BITFIELD reads a string; HPERSIST's no-TTL code (gh #232) ===")
    # gh #232 §4 established that a bitmap IS a string for read-only ops, but
    # BITFIELD was missed: a STRING fell into the "no key yet" branch and was
    # read as an empty buffer, so `SET k hello; BITFIELD k GET u8 0` returned 0
    # instead of 104 ('h'). GETRANGE on the same key already returned 'h'.
    c("DEL", "gh232:bf"); c("SET", "gh232:bf", "hello")
    check("BITFIELD GET u8 0 reads 'h'", c("BITFIELD", "gh232:bf", "GET", "u8", 0), [":104"])
    check("BITFIELD GET u8 8 reads 'e'", c("BITFIELD", "gh232:bf", "GET", "u8", 8), [":101"])
    check("  ...and the string is untouched", c("GET", "gh232:bf"), b"hello")

    # The SSO/heap boundary is the trap: STRING_SSO keeps its bytes INSIDE the
    # value, so its _data0 is a length-and-characters word, not an address.
    for n in (22, 23, 24, 64):
        v = "A" * (n - 1) + "Z"
        c("DEL", "gh232:bf2"); c("SET", "gh232:bf2", v)
        check(f"len {n}: first byte", c("BITFIELD", "gh232:bf2", "GET", "u8", 0), [":65"])
        check(f"len {n}: last byte", c("BITFIELD", "gh232:bf2", "GET", "u8", (n - 1) * 8), [":90"])
        check(f"len {n}: value intact", c("GET", "gh232:bf2"), v.encode())

    # Redis has one type for strings and bitmaps, for writes too: BITFIELD
    # SET / INCRBY on a plain string mutate it, they are not refused. Pion
    # matches, including growing a value past the blob-tier threshold (probed
    # against redis-server 8.10, and a 2 MiB value grown with BITFIELD SET
    # does not crash). "hello" is h=104.
    c("DEL", "gh232:bf3"); c("SET", "gh232:bf3", "hello")
    check("BITFIELD SET on a string returns the old byte", c("BITFIELD", "gh232:bf3", "SET", "u8", 0, 255), [":104"])
    check("BITFIELD INCRBY on a string wraps the byte", c("BITFIELD", "gh232:bf3", "INCRBY", "u8", 0, 1), [":0"])
    check("  ...the string was mutated in place", c("GET", "gh232:bf3"), b"\x00ello")

    # ...but BITFIELD on a real bitmap, and on a missing key, still writes.
    c("DEL", "gh232:bf4")
    check("BITFIELD SET creates the key", c("BITFIELD", "gh232:bf4", "SET", "u8", 0, 255), [":0"])
    check("  ...and reads back", c("BITFIELD", "gh232:bf4", "GET", "u8", 0), [":255"])

    print("\n  -- HPERSIST: -2 no field, -1 no TTL, 1 removed --")
    c("DEL", "gh232:hp"); c("HSET", "gh232:hp", "f", "v")
    check("no such field is -2", c("HPERSIST", "gh232:hp", "FIELDS", 1, "nope"), [":-2"])
    check("field with no TTL is -1", c("HPERSIST", "gh232:hp", "FIELDS", 1, "f"), [":-1"])
    c("HEXPIRE", "gh232:hp", 100, "FIELDS", 1, "f")
    check("removing a TTL is 1", c("HPERSIST", "gh232:hp", "FIELDS", 1, "f"), [":1"])
    check("...and now it is -1 again", c("HPERSIST", "gh232:hp", "FIELDS", 1, "f"), [":-1"])


def section_oracle(c, rc):
    """Every expectation above, re-asked of a real redis-server."""
    print("\n=== 6. same probes against real Redis ===")
    global PASS, FAIL
    probes = [
        ("SET", "o:a", "hello"), ("XADD", "o:a", "*", "f", "v"), ("GET", "o:a"),
        ("RPUSH", "o:b", "x"), ("XADD", "o:b", "*", "f", "v"), ("LRANGE", "o:b", 0, -1),
        ("XADD", "o:c", "NOMKSTREAM", "*", "f", "v"), ("EXISTS", "o:c"),
        ("RPUSH", "o:d", "x"), ("INCRBY", "o:d", "abc"), ("INCRBY", "o:d", 1),
        ("INCR", "o:d"), ("DECRBY", "o:d", 1), ("HGET", "o:d", "f"), ("SCARD", "o:d"),
        ("ZCOUNT", "o:d", "-inf", "+inf"), ("XLEN", "o:d"), ("SPOP", "o:d"),
        ("SETBIT", "o:e", 7, 1), ("INCR", "o:e"),
        ("SET", "o:f", "hello"), ("INCRBYFLOAT", "o:f", "1.5"), ("GET", "o:f"),
        ("SET", "o:g", "100000000"), ("INCRBYFLOAT", "o:g", "1"),
        ("SET", "o:h", "10.5"), ("INCRBYFLOAT", "o:h", "0.1"),
        ("HGET", "o:miss", "f"), ("LPOP", "o:miss"), ("LRANGE", "o:miss", 0, -1),
        ("SCARD", "o:miss"), ("XLEN", "o:miss"), ("ZPOPMIN", "o:miss"),
    ]
    for k in ("o:a", "o:b", "o:c", "o:d", "o:e", "o:f", "o:g", "o:h", "o:miss"):
        c("DEL", k)
        rc("DEL", k)
    diffs = 0
    for p in probes:
        got, want = c(*p), rc(*p)
        # Error MESSAGE wording is not a compatibility contract; the CODE is.
        if isinstance(got, str) and isinstance(want, str):
            got, want = got.split(" ", 1)[0], want.split(" ", 1)[0]
        if got != want:
            diffs += 1
            print(f"  DIFF  {' '.join(str(x) for x in p):40s} pion={got!r} redis={want!r}")
    if diffs == 0:
        PASS += 1
        print(f"  PASS  all {len(probes)} probes agree with real Redis")
    else:
        FAIL += 1
        print(f"  FAIL  {diffs} of {len(probes)} probes disagree with real Redis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--redis-port", type=int, default=6397)
    ap.add_argument("--no-redis", action="store_true")
    a = ap.parse_args()

    c = client(a.port)
    section_xadd_does_not_destroy(c)
    section_nomkstream(c)
    section_wrongtype_split(c)
    section_incr_family(c)
    section_incrbyfloat(c)
    section_lmove(c)
    section_value_bugs(c)

    proc = None
    if not a.no_redis:
        try:
            proc = subprocess.Popen(
                ["redis-server", "--port", str(a.redis_port), "--save", "",
                 "--appendonly", "no"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0)
            section_oracle(c, client(a.redis_port))
        except (OSError, ConnectionError) as e:
            print(f"\n=== 6. real Redis unavailable ({e}) — ENV_SKIP, not a failure ===")
        finally:
            if proc:
                proc.terminate()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
