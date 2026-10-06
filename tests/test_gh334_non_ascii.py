#!/usr/bin/env python3
"""gh #334: non-ASCII keys and values survive every write and read command.

RESP3Token.value() spelled every byte >= 0x80 as '?', valid UTF-8 included.
The fast path stores raw bytes, the slow path took value(), so:
  - SET ... EX/PX/NX/KEEPTTL/GET, SETEX, PSETEX, SETNX, GETSET, MSETNX,
    HSETNX, RPUSHX stored "café" as "caf??" (and a binary value mangled);
  - 20 of 27 read commands (STRLEN, TTL, EXPIRE, HGETALL, LRANGE, SMEMBERS,
    ZRANGE, RENAME, ...) missed a non-ASCII key the fast path had stored.

Now value() is exact for valid UTF-8 and the writers take raw_value() for
the bytes they store. Keys that are NOT valid UTF-8 still reach slow-path
handlers spelled '?' (a Mojo String must be valid UTF-8); that remainder is
checked as a known limitation, not asserted away.

    python3 tests/test_gh334_non_ascii.py [--port 1974]
"""
import argparse
import sys

import redis

TEXT = "café — ü € 😀 日本"
BINARY = bytes(range(256)) * 2          # every byte value, including invalid UTF-8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    r = redis.Redis(port=ap.parse_args().port)
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + name)
        fails += 0 if ok else 1

    for label, v in (("text", TEXT.encode()), ("binary", BINARY)):
        def fresh(k):
            k = f"gh334:{label}:{k}"
            r.delete(k)
            return k
        writes = {
            "SET": lambda k: r.set(k, v),
            "SET EX": lambda k: r.set(k, v, ex=100),
            "SET PX": lambda k: r.set(k, v, px=100000),
            "SET NX": lambda k: r.set(k, v, nx=True),
            "SET XX": lambda k: (r.set(k, "x"), r.set(k, v, xx=True)),
            "SET KEEPTTL": lambda k: (r.set(k, "x", ex=100), r.set(k, v, keepttl=True)),
            "SET GET": lambda k: (r.set(k, "x"), r.set(k, v, get=True)),
            "SETEX": lambda k: r.setex(k, 100, v),
            "PSETEX": lambda k: r.psetex(k, 100000, v),
            "SETNX": lambda k: r.setnx(k, v),
            "GETSET": lambda k: (r.set(k, "x"), r.getset(k, v)),
            "MSETNX": lambda k: r.msetnx({k: v}),
        }
        for name, w in writes.items():
            k = fresh(name.replace(" ", "_"))
            w(k)
            check(f"{label} value round-trips through {name}", r.get(k) == v)
        k = fresh("hsetnx")
        r.hsetnx(k, "f", v)
        check(f"{label} value round-trips through HSETNX", r.hget(k, "f") == v)
        k = fresh("rpushx")
        r.rpush(k, "x")
        r.rpushx(k, v)
        check(f"{label} value round-trips through RPUSHX", r.lindex(k, 1) == v)
        k = fresh("lpushx")
        r.rpush(k, "x")
        r.lpushx(k, v)
        check(f"{label} value round-trips through LPUSHX", r.lindex(k, 0) == v)
        k = fresh("set_ifeq")
        r.set(k, v)
        r.execute_command("SET", k, "new", "IFEQ", v)
        check(f"{label} SET IFEQ matches the stored {label} value", r.get(k) == b"new")

    # Non-ASCII (valid UTF-8) KEYS through the read commands, stored by the
    # fast path.
    K, H, L, S, Z, F = "gh334:clé:ü", "gh334:hé:ü", "gh334:lé:ü", "gh334:sé:ü", "gh334:zé:ü", "fé"

    def setup():
        r.delete(K, H, L, S, Z, K + "2")
        r.set(K, "value")
        r.hset(H, F, "v")
        r.rpush(L, "a", "b")
        r.sadd(S, "m")
        r.zadd(Z, {"m": 1})

    reads = {
        "GET": lambda: r.get(K) == b"value", "STRLEN": lambda: r.strlen(K) == 5,
        "GETRANGE": lambda: r.getrange(K, 0, 1) == b"va", "GETEX": lambda: r.getex(K) == b"value",
        "TTL": lambda: r.ttl(K) == -1, "EXPIRE": lambda: r.expire(K, 1000) is True,
        "SET EX on a non-ASCII key, read by name": lambda: (r.set(K, "w", ex=100), r.get(K))[1] == b"w",
        "HGETALL": lambda: r.hgetall(H) == {F.encode(): b"v"}, "HMGET": lambda: r.hmget(H, [F]) == [b"v"],
        "HEXISTS": lambda: r.hexists(H, F) is True, "HLEN": lambda: r.hlen(H) == 1,
        "HKEYS": lambda: r.hkeys(H) == [F.encode()],
        "LRANGE": lambda: r.lrange(L, 0, -1) == [b"a", b"b"], "LLEN": lambda: r.llen(L) == 2,
        "SMEMBERS": lambda: r.smembers(S) == {b"m"}, "SISMEMBER": lambda: r.sismember(S, "m") in (True, 1),   # RESP3 clients get the integer
        "SCARD": lambda: r.scard(S) == 1, "ZRANGE": lambda: r.zrange(Z, 0, -1) == [b"m"],
        "ZSCORE": lambda: r.zscore(Z, "m") == 1.0, "ZCARD": lambda: r.zcard(Z) == 1,
        "RENAME": lambda: r.rename(K, K + "2") and r.get(K + "2") == b"value",
        "DEL": lambda: r.delete(K) == 1,
    }
    for name, f in reads.items():
        setup()
        try:
            ok = f()
        except redis.ResponseError as e:
            ok, name = False, f"{name} ({e})"
        check(f"non-ASCII key: {name}", bool(ok))

    # Known limitation: a key that is not valid UTF-8 cannot become a Mojo
    # String, so slow-path handlers still see it with '?' in place of the
    # invalid bytes. Report, do not assert.
    bk = b"gh334:bin\xff\xfe"
    r.delete(bk)
    r.set(bk, "v")
    lim = r.strlen(bk)
    print(f"INFO binary (invalid UTF-8) key through a slow-path read: STRLEN -> {lim} "
          f"({'works' if lim == 1 else 'still spelled ? — known limitation'})")
    r.delete(bk, K, H, L, S, Z, K + "2")
    for k in r.scan_iter("gh334:*"):
        r.delete(k)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
