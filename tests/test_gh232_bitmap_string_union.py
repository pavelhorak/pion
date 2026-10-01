#!/usr/bin/env python3
"""gh #232 remainder — the MUTATING half of "a bitmap IS a string".

gh #232 §4 made the read-only ops (STRLEN/GETRANGE/GETBIT/BITCOUNT) work across
STRING and BITMAP, and deliberately stopped there:

    "Read-only only. `setbit()` frees-and-reallocs when it grows, and a STRING
     payload may live in the gh #163 blob arena, which the heap allocator must
     never free — so SETBIT/APPEND/SETRANGE still refuse the other shape."

That left Pion answering WRONGTYPE to `SET k "hello"; SETBIT k 10 1`, which real
Redis accepts — the rarer and more confusing direction: refusing textbook usage rather than accepting something it shouldn't.

The blocker is resolved by COPYING rather than mutating in place
(`GenericValue.owned_bitmap_copy`), which always returns plain heap whatever the
source shape — SSO, heap, or blob-arena-backed — so the arena is never handed to
the allocator. APPEND and SETRANGE already copied out, so those only needed the
type gate widened.

Divergence count against real redis-server 8.10: **20 -> 15**, and every one of
the 15 that remain is HLL, which stays deliberately distinct (Pion stores dense
registers, Redis a sparse HYLL encoding; exposing Pion's bytes as a string would
swap a type divergence for a VALUE divergence, which is worse).

One trap this test exists to catch, because it bit during the fix: widening the
OUTER type gate on INCRBYFLOAT let a BITMAP reach the handler, skip the strict
stored-value validation (which still tested `is_string()`), and keep cur_f = 0.0
— so `SETBIT k 0 1; INCRBYFLOAT k 1.5` replied 1.5 and DESTROYED the bitmap.
A type-gate widening must widen every gate the value then meets.

Every assertion is differential against a live redis-server.

Usage: python3 tests/test_gh232_bitmap_string_union.py [--port 1974] [--redis-port 6399]
"""
import argparse, sys

try:
    import redis
except ImportError as e:
    print(f"SKIP: needs redis-py ({e})")
    sys.exit(0)

failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def both(p, r, fn, label):
    """Run fn against Pion and Redis; compare replies (errors compared by code)."""
    out = []
    for c in (p, r):
        try:
            out.append(("ok", fn(c)))
        except redis.ResponseError as e:
            out.append(("err", str(e).split()[0]))
        except Exception as e:                      # noqa: BLE001
            out.append(("exc", type(e).__name__))
    check(label, out[0] == out[1], f"pion={out[0]} redis={out[1]}")
    return out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--redis-port", type=int, default=6399)
    a = ap.parse_args()
    try:
        p = redis.Redis(port=a.port); p.ping()
    except Exception as e:
        print(f"FATAL: no Pion on {a.port} ({e})"); return 2
    try:
        r = redis.Redis(port=a.redis_port); r.ping()
    except Exception as e:
        print(f"FATAL: no redis-server on {a.redis_port} ({e}) — this test is "
              f"differential and proves nothing without the oracle"); return 2

    def reset(seed_fn):
        for c in (p, r):
            c.delete("u")
            seed_fn(c)

    # Every string shape the union has to survive. SSO vs heap matters: an SSO
    # value keeps its bytes INSIDE the GenericValue, so `_data0` is a
    # length-and-characters word and not an address — reading it as a pointer
    # dereferences packed characters. 23/24 straddles that boundary.
    SHAPES = [
        ("empty",     lambda c: c.set("u", "")),
        ("sso-2B",    lambda c: c.set("u", "hi")),
        ("sso-23B",   lambda c: c.set("u", "x" * 23)),
        ("heap-24B",  lambda c: c.set("u", "y" * 24)),
        ("heap-100B", lambda c: c.set("u", "z" * 100)),
        ("bitmap",    lambda c: c.execute_command("SETBIT", "u", 5, 1)),
    ]

    print("\n[1] SETBIT across every string shape — the textbook idiom")
    for name, seed in SHAPES:
        for off in (0, 7, 10, 100, 900):
            reset(seed)
            both(p, r, lambda c, o=off: c.execute_command("SETBIT", "u", o, 1),
                 f"SETBIT {name} @{off} reply")
            check(f"SETBIT {name} @{off} bytes match",
                  p.get("u") == r.get("u"), f"pion={p.get('u')!r} redis={r.get('u')!r}")

    print("\n[2] Clearing a bit, and STRLEN/BITCOUNT after the mutation")
    for name, seed in SHAPES:
        reset(seed)
        for c in (p, r): c.execute_command("SETBIT", "u", 12, 1)
        both(p, r, lambda c: c.execute_command("SETBIT", "u", 12, 0),
             f"SETBIT {name} clear reply")
        check(f"SETBIT {name} clear bytes", p.get("u") == r.get("u"))
        check(f"STRLEN after {name}", p.strlen("u") == r.strlen("u"))
        check(f"BITCOUNT after {name}",
              p.execute_command("BITCOUNT", "u") == r.execute_command("BITCOUNT", "u"))

    print("\n[3] APPEND / SETRANGE / GETDEL on a BITMAP")
    for label, fn in (
        ("APPEND on a bitmap",   lambda c: c.execute_command("APPEND", "u", "x")),
        ("SETRANGE on a bitmap", lambda c: c.execute_command("SETRANGE", "u", 0, "AB")),
        ("GETDEL on a bitmap",   lambda c: c.execute_command("GETDEL", "u")),
        ("GETSET on a bitmap",   lambda c: c.execute_command("GETSET", "u", "z")),
    ):
        reset(lambda c: c.execute_command("SETBIT", "u", 5, 1))
        both(p, r, fn, label)
        check(f"{label}: resulting bytes match", p.get("u") == r.get("u"),
              f"pion={p.get('u')!r} redis={r.get('u')!r}")

    print("\n[4] INCRBYFLOAT must still REFUSE non-numeric bytes")
    print("    (widening the outer gate alone made this reply 1.5 and destroy")
    print("     the bitmap — the stored-value check had to widen too)")
    reset(lambda c: c.execute_command("SETBIT", "u", 5, 1))
    both(p, r, lambda c: c.execute_command("INCRBYFLOAT", "u", "1.5"),
         "INCRBYFLOAT on a bitmap errors")
    check("INCRBYFLOAT left the bitmap intact", p.get("u") == r.get("u"),
          f"pion={p.get('u')!r} redis={r.get('u')!r}")
    # And the numeric case still works, so the widening did not break it.
    reset(lambda c: c.set("u", "10.5"))
    both(p, r, lambda c: c.execute_command("INCRBYFLOAT", "u", "1.5"),
         "INCRBYFLOAT on a numeric string still works")

    print("\n[5] Containers must STILL be refused (the widening must not leak)")
    for label, seed in (
        ("list",   lambda c: c.rpush("u", "a")),
        ("hash",   lambda c: c.hset("u", "f", "v")),
        ("set",    lambda c: c.sadd("u", "m")),
        ("zset",   lambda c: c.zadd("u", {"m": 1})),
    ):
        for cmd in ("SETBIT", "APPEND", "SETRANGE"):
            reset(seed)
            args = {"SETBIT": ("u", 0, 1), "APPEND": ("u", "x"),
                    "SETRANGE": ("u", 0, "x")}[cmd]
            both(p, r, lambda c, k=cmd, ar=args: c.execute_command(k, *ar),
                 f"{cmd} on a {label} is refused")

    for c in (p, r): c.delete("u")
    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
