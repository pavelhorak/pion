#!/usr/bin/env python3
"""#50: SCAN, HSCAN, SSCAN and ZSCAN walk incrementally with a working cursor.

Before, each returned its whole table in one reply with cursor 0, so COUNT
and the cursor did nothing and a large keyspace ran into the 4 MB reply cap.
The contract here is Redis's:
  - a full iteration (until the cursor returns 0) yields every key present
    from the first call to the last, at least once, even across inserts,
    deletes and the table rebuilds those cause;
  - COUNT bounds a dense table into many calls;
  - a small keyspace still comes back in one call (Pion's table is millions
    of slots, Redis's dict is tiny, but both return cursor 0 at once here);
  - MATCH and TYPE filter; HSCAN/SSCAN/ZSCAN walk a large collection too.

    python3 tests/test_scan_cursor.py [--port 1974]
"""
from __future__ import annotations

import argparse
import fnmatch
import sys

import redis

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def full_scan(fn, cap=500000):
    cur, seen, calls = 0, set(), 0
    while True:
        cur, items = fn(cur)
        for it in items:
            seen.add(it)
        calls += 1
        if cur == 0 or calls > cap:
            break
    return seen, calls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    r = redis.Redis(port=ap.parse_args().port, socket_timeout=30)
    r.flushall()

    print("[1] small keyspace: one call, cursor 0")
    for k in ("a", "b", "c"):
        r.set(k, "1")
    cur, keys = r.scan(0, count=100)
    check("SCAN 0 COUNT 100 on 3 keys returns all 3", cur == 0 and len(keys) == 3,
          f"cursor={cur}, {len(keys)} keys")
    r.flushall()

    print("[2] dense keyspace: paginates, full iteration is complete")
    N = 50000
    p = r.pipeline(transaction=False)
    for i in range(N):
        p.set(f"k:{i}", 1)
    p.execute()
    first_cur, first = r.scan(0, count=100)
    check("first SCAN COUNT 100 returns a page, not everything", first_cur != 0 and 0 < len(first) <= 400,
          f"cursor={first_cur}, {len(first)} keys")
    seen, calls = full_scan(lambda c: r.scan(c, count=100))
    want = {f"k:{i}".encode() for i in range(N)}
    check("full SCAN returns every key", seen == want and calls > 1, f"{len(seen)}/{N} in {calls} calls")

    print("[3] MATCH and TYPE filter")
    seen_m, _ = full_scan(lambda c: r.scan(c, match="k:1*", count=500))
    want_m = {f"k:{i}".encode() for i in range(N) if fnmatch.fnmatch(f"k:{i}", "k:1*")}
    check("SCAN MATCH k:1* returns exactly the matches", seen_m == want_m, f"{len(seen_m)} vs {len(want_m)}")
    r.lpush("aList", "x")
    r.sadd("aSet", "y")
    seen_t, _ = full_scan(lambda c: r.scan(c, _type="list", count=1000))
    check("SCAN TYPE list returns only the list", seen_t == {b"aList"}, str(seen_t))
    r.delete("aList", "aSet")

    print("[4] every key present throughout a mutating iteration is returned")
    cur, seen2, calls2, mutated = 0, set(), 0, False
    while True:
        cur, keys = r.scan(cur, count=200)
        seen2.update(keys)
        calls2 += 1
        if not mutated and calls2 == 10:
            pp = r.pipeline(transaction=False)
            for i in range(N // 2, N):          # delete the upper half (tombstones, rehash)
                pp.delete(f"k:{i}")
            for i in range(N, N + 5000):        # and add new keys
                pp.set(f"k:{i}", 1)
            pp.execute()
            mutated = True
        if cur == 0 or calls2 > 500000:
            break
    lower = {f"k:{i}".encode() for i in range(N // 2)}   # present from start to end
    check("no key present throughout the scan is missed", lower <= seen2,
          f"{len(lower - seen2)} missed")
    r.flushall()

    print("[5] HSCAN / SSCAN / ZSCAN walk a large collection")
    M = 5000
    p = r.pipeline(transaction=False)
    for i in range(M):
        p.hset("H", f"f{i}", i)
        p.sadd("S", f"m{i}")
        p.zadd("Z", {f"z{i}": i})
    p.execute()
    hseen, hcalls = full_scan(lambda c: r.hscan("H", c, count=100))   # redis-py returns (cur, dict)
    check("HSCAN returns every field in many calls", len(hseen) == M and hcalls > 1, f"{len(hseen)} in {hcalls}")
    sseen, scalls = full_scan(lambda c: r.sscan("S", c, count=100))
    check("SSCAN returns every member in many calls", len(sseen) == M and scalls > 1, f"{len(sseen)} in {scalls}")
    zseen, zcalls = full_scan(lambda c: r.zscan("Z", c, count=100))   # (cur, list of (member, score))
    zmembers = {m for (m, _s) in zseen}
    check("ZSCAN returns every member in many calls", len(zmembers) == M and zcalls > 1, f"{len(zmembers)} in {zcalls}")
    zd = dict(r.zscan_iter("Z"))
    check("ZSCAN pairs member with its score", zd.get(b"z4242") == 4242.0, str(zd.get(b"z4242")))
    r.flushall()

    check("server alive", r.ping() is True)
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
