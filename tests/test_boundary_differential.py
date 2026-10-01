#!/usr/bin/env python3
"""Both sides of every representation switch, against a live Redis 8.

WHY
Pion changes representation at thresholds that Redis does not have, and each
switch is a place where one half of an implementation can be wrong while the
other half is tested:

  * lists: contiguous ziplist up to 1024 entries AND 64-byte values, segmented
    quicklist above (SEG_SIZE 256). RPOP/LPOP broke the segmented invariant for
    years — tests checked "drains fully" (a count), never the popped VALUES.
  * strings: SSO at <= 23 bytes, heap above; writev above 512-byte replies.
  * keys: SSO vs heap hashing (a half-converted derivation passes SSO-only tests).
  * hash / set / zset members at 23/24 bytes.

This file drives SEEDED, SCRAMBLED operation sequences through both servers in
lockstep and compares every reply by VALUE (unordered replies are sorted; error
replies compare their code). Expected values come from Redis, never from Pion.
Values carry non-ASCII bytes (0x80-0xFF, NUL, CR/LF), so a byte-mangling path
(#334: `else: s += "?"`) is a divergence too. After every command both
connections must still be in sync (PING is the very next reply).

    python3 tests/test_boundary_differential.py --pion-port 1974 --start-redis
Exit 0 = no divergence.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, wait_ready  # noqa: E402

REDIS = os.environ.get("REDIS_SERVER", "redis-server")

# Replies whose ORDER is not part of the contract.
UNORDERED = {"SMEMBERS", "HKEYS", "HVALS", "SINTER", "SUNION", "SDIFF", "KEYS"}
PAIRED_UNORDERED = {"HGETALL"}


def norm(cmd, r):
    name = cmd[0].upper()
    if isinstance(r, RespError):
        return ("ERR", r.split(" ", 1)[0])
    if isinstance(r, list) and name in UNORDERED:
        return sorted(r, key=repr)
    if isinstance(r, list) and name in PAIRED_UNORDERED:
        return sorted(zip(r[0::2], r[1::2]), key=repr)
    if isinstance(r, dict):
        return sorted(r.items(), key=repr)
    return r


class Diff:
    def __init__(self, pion, redis, verbose=False):
        self.p, self.r = pion, redis
        self.n = 0
        self.fails: list[str] = []
        self.verbose = verbose
        self.section = ""

    def __call__(self, *cmd):
        self.n += 1
        # Framing: one command in, one reply out — each command travels with
        # a PING and the reply after it must be PONG.
        rp = self.p.cmd_synced(*cmd)
        rr = self.r.cmd_synced(*cmd)
        if norm(cmd, rp) != norm(cmd, rr):
            shown = [c if len(c) <= 40 else c[:20] + b"...<%dB>" % len(c) for c in
                     (x.encode() if isinstance(x, str) else x for x in cmd)]
            self.fails.append(f"[{self.section}] {shown}\n      pion : {short(rp)}\n      redis: {short(rr)}")
        return rr

    def done(self, name):
        bad = [f for f in self.fails if f.startswith(f"[{name}]")]
        print(f"  {'FAIL' if bad else 'ok  '}  {name}" + (f"  ({len(bad)} divergences)" if bad else ""))


def short(v):
    s = repr(v)
    return s if len(s) <= 200 else s[:200] + f"...<{len(s)} chars>"


# Values with bytes a text-mangling path would destroy.
SPICE = [b"", b"\x00", b"\r\n", b"\xc3\xa9", b"\xff\xfe", b"\xe2\x80\x94", b"\x80"]


HIGH = 256   # --ascii lowers this to 127 (isolates structure bugs from byte bugs)


def value(rng: random.Random, n: int, tag: int) -> bytes:
    """A unique value of EXACTLY n bytes (so size classes are what they say)."""
    head = b"%d:" % tag + (rng.choice(SPICE) if HIGH == 256 else b"")
    if len(head) >= n:
        return (b"%d" % tag)[-n:].rjust(n, b"#") if n else b""
    return head + bytes(rng.randrange(33, HIGH) for _ in range(n - len(head)))


# ── lists ────────────────────────────────────────────────────────────────────
def list_section(d: Diff, rng: random.Random, n_entries: int, elem: int, key: bytes):
    d("DEL", key, key + b":dst")
    tag = 0
    # Build with a SCRAMBLED mix of LPUSH/RPUSH, several per command.
    while tag < n_entries:
        batch = min(rng.randint(1, 7), n_entries - tag)
        vals = [value(rng, elem, tag + i) for i in range(batch)]
        tag += batch
        d(rng.choice(["LPUSH", "RPUSH"]), key, *vals)
    d("LLEN", key)
    for _ in range(260):
        op = rng.random()
        llen = d("LLEN", key)
        if op < 0.22:
            d("RPOP", key)
        elif op < 0.44:
            d("LPOP", key)
        elif op < 0.50:
            d(rng.choice(["LPOP", "RPOP"]), key, str(rng.randint(1, 5)))
        elif op < 0.60:
            i = rng.randint(-llen - 2, llen + 2) if llen else 0
            d("LINDEX", key, str(i))
        elif op < 0.70:
            a = rng.randint(-llen - 3, llen + 3) if llen else 0
            b = rng.randint(a, a + rng.randint(0, 300))
            d("LRANGE", key, str(a), str(b))
        elif op < 0.74 and llen:
            d("LSET", key, str(rng.randint(-llen, llen - 1)), value(rng, elem, tag)); tag += 1
        elif op < 0.78:
            d(rng.choice(["LPUSH", "RPUSH"]), key, value(rng, elem, tag)); tag += 1
        elif op < 0.81 and llen:
            pivot = d("LINDEX", key, str(rng.randint(0, llen - 1)))
            if pivot is not None:
                d("LINSERT", key, rng.choice(["BEFORE", "AFTER"]), pivot, value(rng, elem, tag)); tag += 1
        elif op < 0.84 and llen:
            v = d("LINDEX", key, str(rng.randint(0, llen - 1)))
            if v is not None:
                d("LPOS", key, v)
                d("LREM", key, str(rng.choice([0, 1, -1])), v)
        elif op < 0.90:
            d("LMOVE", key, key + b":dst", rng.choice(["LEFT", "RIGHT"]), rng.choice(["LEFT", "RIGHT"]))
        elif op < 0.93:
            d("RPOPLPUSH", key + b":dst", key)
        elif op < 0.95 and llen > 40:
            # Trim a little off each end: keeps the list large.
            d("LTRIM", key, str(rng.randint(0, 3)), str(-rng.randint(1, 4)))
        else:
            d("LLEN", key + b":dst")
    d("LRANGE", key, "0", "-1")
    d("LRANGE", key + b":dst", "0", "-1")
    # Drain both ends to empty, checking every value (the #369-era bug).
    while True:
        r = d(rng.choice(["LPOP", "RPOP"]), key)
        if r is None:
            break
    d("EXISTS", key)


# ── strings / keys ───────────────────────────────────────────────────────────
KEY_LENS = [1, 22, 23, 24, 25, 64]
VAL_LENS = [0, 1, 22, 23, 24, 25, 63, 64, 65, 511, 512, 513, 4096]


def keyname(rng, n, tag):
    base = b"bd:" + b"%d:" % tag + (rng.choice([b"", b"\xc3\xa9", b"\xff"]) if HIGH == 256 else b"")
    return (base + b"k" * n)[:n] if n >= len(base) else (b"%d" % tag).rjust(n, b"k")[-n:]


def string_section(d: Diff, rng: random.Random):
    tag = 0
    keys = []
    for kl in KEY_LENS:
        for vl in VAL_LENS:
            k = keyname(rng, kl, tag); tag += 1
            v = value(rng, vl, tag)
            d("SET", k, v)
            d("GET", k)
            d("STRLEN", k)
            keys.append((k, vl))
    rng.shuffle(keys)
    for k, vl in keys[:60]:
        d("APPEND", k, value(rng, rng.choice([1, 2, 23, 24]), tag)); tag += 1
        d("GET", k)
        d("GETRANGE", k, str(rng.randint(-30, 30)), str(rng.randint(-30, 600)))
        d("SETRANGE", k, str(rng.randint(0, 30)), value(rng, rng.choice([1, 5, 24]), tag)); tag += 1
        d("GET", k)
    # MSET/MGET across SSO and heap keys in ONE command, in scrambled order.
    for _ in range(6):
        pick = rng.sample(keys, 10)
        args = []
        for k, _vl in pick:
            args += [k, value(rng, rng.choice(VAL_LENS), tag)]; tag += 1
        d("MSET", *args)
        d("MGET", *[k for k, _ in rng.sample(pick, len(pick))] + [b"bd:missing"])
    # Integers stored as strings, at the digit counts where SSO flips.
    for digits in (1, 18, 19, 20, 22, 23, 24):
        k = keyname(rng, rng.choice(KEY_LENS), tag); tag += 1
        d("SET", k, "9" * digits)
        d("INCR", k)
        d("INCRBY", k, "-7")
        d("GET", k)
    for k, _ in keys[:25]:
        d("GETSET", k, value(rng, rng.choice(VAL_LENS), tag)); tag += 1
        d("EXISTS", k)
    for k, _ in keys[25:45]:
        nk = keyname(rng, rng.choice(KEY_LENS), tag); tag += 1
        d("RENAME", k, nk)
        d("GET", nk)
        d("EXISTS", k)
    for k, _ in keys[45:70]:
        d("GETDEL", k)
        d("GET", k)
    d("DEL", *[k for k, _ in keys])


# ── hash / set / zset members at the 23/24 boundary ──────────────────────────
MEMBER_LENS = [1, 22, 23, 24, 25, 64, 65, 200]


def hash_section(d: Diff, rng: random.Random, n: int):
    key = b"bd:hash:" + b"%d" % n
    d("DEL", key)
    fields = [value(rng, rng.choice(MEMBER_LENS), i) for i in range(n)]
    rng.shuffle(fields)
    for i in range(0, n, 5):
        args = []
        for f in fields[i:i + 5]:
            args += [f, value(rng, rng.choice(VAL_LENS[:10]), i)]
        d("HSET", key, *args)
    d("HLEN", key)
    for f in rng.sample(fields, min(n, 60)):
        d("HGET", key, f)
        d("HSTRLEN", key, f)
        d("HEXISTS", key, f)
    d("HMGET", key, *rng.sample(fields, min(n, 8)), b"bd:nofield")
    for f in rng.sample(fields, n // 3):
        d("HDEL", key, f)
    d("HGETALL", key)
    d("HKEYS", key)
    d("HVALS", key)


def set_section(d: Diff, rng: random.Random, n: int):
    a, b = b"bd:set:a:%d" % n, b"bd:set:b:%d" % n
    d("DEL", a, b)
    members = [value(rng, rng.choice(MEMBER_LENS), i) for i in range(n)]
    for i in range(0, n, 6):
        d("SADD", a, *members[i:i + 6])
    d("SADD", b, *rng.sample(members, n // 2), value(rng, 24, 10**6))
    d("SCARD", a)
    for m in rng.sample(members, min(n, 50)):
        d("SISMEMBER", a, m)
    d("SMISMEMBER", a, *rng.sample(members, min(n, 6)), b"nope")
    for m in rng.sample(members, n // 4):
        d("SREM", a, m)
    d("SMEMBERS", a)
    d("SINTER", a, b)
    d("SUNION", a, b)
    d("SDIFF", a, b)
    d("SINTERCARD", "2", a, b)


def zset_section(d: Diff, rng: random.Random, n: int):
    key = b"bd:zset:%d" % n
    d("DEL", key)
    members = [value(rng, rng.choice(MEMBER_LENS), i) for i in range(n)]
    # Scores with many TIES, in SCRAMBLED insert order: equal-score members
    # must come out lexicographically (gh #251 hid behind reverse insertion).
    scores = [rng.choice([0, 0, 0, 1, 2.5, -3, 1e6, 0.25]) for _ in range(n)]
    order = list(range(n)); rng.shuffle(order)
    for i in range(0, n, 4):
        args = []
        for j in order[i:i + 4]:
            args += [repr(scores[j]), members[j]]
        d("ZADD", key, *args)
    d("ZCARD", key)
    d("ZRANGE", key, "0", "-1", "WITHSCORES")
    for m in rng.sample(members, min(n, 40)):
        d("ZSCORE", key, m)
        d("ZRANK", key, m)
        d("ZREVRANK", key, m)
    d("ZRANGEBYSCORE", key, "0", "2.5")
    d("ZRANGE", key, "(0", "+inf", "BYSCORE", "LIMIT", "1", "7")
    d("ZCOUNT", key, "-inf", "0")
    for m in rng.sample(members, n // 5):
        d("ZREM", key, m)
    d("ZINCRBY", key, "0.5", members[0])
    d("ZPOPMIN", key, "3")
    d("ZPOPMAX", key, "2")
    d("ZRANGE", key, "0", "-1", "WITHSCORES")
    # Score-0 lex index — the autocomplete idiom.
    lex = b"bd:lex:%d" % n
    d("DEL", lex)
    words = [value(rng, rng.choice([3, 23, 24, 30]), i) for i in range(min(n, 300))]
    rng.shuffle(words)
    for i in range(0, len(words), 8):
        args = []
        for w in words[i:i + 8]:
            args += ["0", w]
        d("ZADD", lex, *args)
    d("ZRANGEBYLEX", lex, "-", "+")
    d("ZRANGEBYLEX", lex, "[1", "(5")


# ── SORT, and hash-field TTLs, over non-ASCII data ───────────────────────────
def sort_section(d: Diff, rng: random.Random):
    for n in (10, 1023, 1025, 1300):
        key = b"bd:sort:%d" % n
        d("DEL", key)
        nums = [repr(rng.choice([rng.randint(-10**6, 10**6), rng.randint(-50, 50) / 4]))
                for _ in range(n)]
        for i in range(0, n, 16):
            d("RPUSH", key, *nums[i:i + 16])
        d("SORT", key)
        d("SORT", key, "DESC", "LIMIT", "3", "25")
        # ALPHA over text, valid UTF-8 and binary (no NUL: Redis compares
        # ALPHA elements with strcoll, which stops at NUL).
        akey = b"bd:sorta:%d" % n
        d("DEL", akey)
        words = [value(rng, rng.choice([3, 23, 24, 65]), i).replace(b"\x00", b"0")
                 for i in range(n)]
        for i in range(0, n, 16):
            d("RPUSH", akey, *words[i:i + 16])
        d("SORT", akey, "ALPHA")
        d("SORT", akey, "ALPHA", "DESC", "LIMIT", "0", "40")
        d("SORT", akey)          # non-numeric elements: must be an error
        d("SORT", key, "LIMIT", "5", "7", "STORE", b"bd:sortdst")
        d("LRANGE", b"bd:sortdst", "0", "-1")
        d("SORT", akey, "ALPHA", "STORE", b"bd:sortdst")
        d("LLEN", b"bd:sortdst")
        d("SORT", b"bd:nokey", "STORE", b"bd:sortdst")    # empty result deletes dest
        d("EXISTS", b"bd:sortdst")
    # Sets and sorted sets sort too (they used to answer []), and a string is
    # WRONGTYPE.
    d("DEL", b"bd:sorts", b"bd:sortz", b"bd:sortstr")
    d("SADD", b"bd:sorts", *[repr(x) for x in (5, -1, 3.5, 100, 0.25)])
    d("SORT", b"bd:sorts")
    d("SORT", b"bd:sorts", "DESC", "LIMIT", "1", "2")
    d("ZADD", b"bd:sortz", "1", "banana", "2", "apple", "3", "cherry")
    d("SORT", b"bd:sortz", "ALPHA")
    d("SORT", b"bd:sortz")                       # non-numeric members -> error
    d("SET", b"bd:sortstr", "x")
    d("SORT", b"bd:sortstr")
    d("SORT", b"bd:sorts", "BY", "nosort", "LIMIT", "0", "0")


def geo_units_section(d: Diff):
    """Units are exactly m/km/ft/mi (any case). Five parsers matched on the
    first byte or two ("kilograms" was km, "f..." was ft) and four used
    1609.344 for a mile where Redis uses 1609.34."""
    d("DEL", b"bd:geo")
    d("GEOADD", b"bd:geo", "13.361389", "38.115556", "Palermo", "15.087269", "37.502669", "Catania")
    for unit in ("m", "KM", "Mi", "ft", "kilograms", "fathoms", "kmx", "mile", ""):
        d("GEODIST", b"bd:geo", "Palermo", "Catania", unit)
        d("GEOSEARCH", b"bd:geo", "FROMLONLAT", "15", "37", "BYRADIUS", "124", unit, "ASC")
        d("GEORADIUS", b"bd:geo", "15", "37", "124", unit, "ASC")
        d("GEORADIUSBYMEMBER", b"bd:geo", "Palermo", "167", unit, "ASC")
    # Order and shape: ASC/DESC were parsed and never applied (GEORADIUS), or
    # not parsed at all (GEORADIUSBYMEMBER, which also dropped WITHDIST).
    d("GEOADD", b"bd:geo", "13.583333", "37.316667", "Agrigento", "12.4964", "41.9028", "Roma")
    for order in ("ASC", "DESC"):
        d("GEORADIUS", b"bd:geo", "15", "37", "700", "km", order)
        d("GEORADIUSBYMEMBER", b"bd:geo", "Palermo", "700", "km", order)
        d("GEORADIUSBYMEMBER", b"bd:geo", "Palermo", "700", "km", order, "WITHDIST")
        d("GEORADIUS", b"bd:geo", "15", "37", "700", "km", order, "WITHDIST")
    d("GEORADIUS", b"bd:geo", "15", "37", "700", "km", "COUNT", "2")      # COUNT alone: nearest first
    d("GEORADIUSBYMEMBER", b"bd:geo", "Palermo", "700", "km", "COUNT", "2")


def field_ttl_section(d: Diff):
    for label, key, field in (("ascii", b"bd:ft:a", b"f"),
                              ("utf8", "bd:ft:\u00e9".encode(), "\u00e9t\u00e9".encode()),
                              ("binary", b"bd:ft:\xff\xfe", b"f\x80\xff")):
        d("DEL", key)
        d("HSET", key, field, "v", b"keep", "w")
        d("HPEXPIRE", key, "150", "FIELDS", "1", field)
        d("HGET", key, field)
        time.sleep(0.3)
        d("HGET", key, field)          # expired: nil on both
        d("HGET", key, b"keep")
        d("HLEN", key)
        d("HEXISTS", key, field)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pion-port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    ap.add_argument("--redis-port", type=int, default=6412)
    ap.add_argument("--start-redis", action="store_true")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--quick", action="store_true", help="fewer list sizes")
    ap.add_argument("--ascii", action="store_true",
                    help="printable ASCII only — separates structure bugs from byte-mangling bugs")
    args = ap.parse_args()
    global HIGH
    if args.ascii:
        HIGH = 127

    rproc, rdir = None, None
    if args.start_redis:
        rdir = tempfile.mkdtemp(prefix="bd-redis-")
        # LC_ALL=C: SORT ALPHA collates with strcoll(); pin it to byte order.
        rproc = subprocess.Popen([REDIS, "--port", str(args.redis_port), "--save", "",
                                  "--appendonly", "no", "--dir", rdir],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 env=dict(os.environ, LC_ALL="C"))
    try:
        wait_ready(args.redis_port, 15, proc=rproc)
        wait_ready(args.pion_port, 30)
        pion, redis = Conn(args.pion_port, timeout=30), Conn(args.redis_port, timeout=30)
        for c in (pion, redis):
            if c.cmd("FLUSHALL") != "OK":
                print("FATAL: FLUSHALL refused"); return 2
        d = Diff(pion, redis)
        rng = random.Random(args.seed)

        print("lists (ziplist <= 1024 entries & <= 64 B values; segmented above):")
        sizes = [1023, 1024, 1025, 1300] if args.quick else [10, 1023, 1024, 1025, 1280, 1537, 2100]
        for n in sizes:
            for elem in (8, 23, 24, 64, 65):
                d.section = f"list n={n} elem={elem}B"
                list_section(d, rng, n, elem, b"bd:list:%d:%d" % (n, elem))
                d.done(d.section)

        d.section = "strings/keys at SSO + writev boundaries"
        string_section(d, rng)
        d.done(d.section)

        d.section = "SORT (numeric, ALPHA, error) across the list switch"
        sort_section(d, rng)
        d.done(d.section)

        d.section = "GEO units: exact keywords, Redis constants"
        geo_units_section(d)
        d.done(d.section)

        d.section = "hash field TTL with ASCII / UTF-8 / binary names"
        field_ttl_section(d)
        d.done(d.section)

        for n in (5, 200, 2000):
            d.section = f"hash n={n}"
            hash_section(d, rng, n); d.done(d.section)
            d.section = f"set n={n}"
            set_section(d, rng, n); d.done(d.section)
            d.section = f"zset n={n}"
            zset_section(d, rng, n); d.done(d.section)

        print(f"\n{d.n} commands compared, {len(d.fails)} divergences")
        # Up to 4 per section, so one noisy section cannot hide another.
        shown = {}
        for f in d.fails:
            sec = f[1:f.index("]")]
            shown[sec] = shown.get(sec, 0) + 1
            if shown[sec] <= 4:
                print("  " + f)
        for sec, k in shown.items():
            if k > 4:
                print(f"  ... [{sec}] {k - 4} more")
        return 1 if d.fails else 0
    finally:
        if rproc:
            rproc.kill(); rproc.wait()
        if rdir:
            import shutil
            shutil.rmtree(rdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
