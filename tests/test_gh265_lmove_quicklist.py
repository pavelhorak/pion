#!/usr/bin/env python3
"""gh #265 regression test — LMOVE/RPOPLPUSH must work past the ziplist boundary.

`handle_lmove` hand-rolled its pop against `zip_buf` and answered NIL for
anything else, so LMOVE on a >1024-entry list (quicklist mode) reported
"source is empty" while the list was full. Callers branch on nil, so the
reliable-queue idiom `RPOPLPUSH src backup` silently stopped moving items at
exactly the size where a queue starts to matter — the gh #241 rule again: a
plausible value the caller branches on is worse than an error.

`SlabList.lpop`/`rpop` already implemented both representations, so the fix was
to delete the hand-rolled path rather than write a new one.

REPRESENTATION-SWITCH RULE (gh #241): test BOTH sides. `SlabList` converts to a
segmented quicklist above 1024 entries, so every case here runs at a small size
(ziplist) AND above the threshold (quicklist). A test that only exercised one
side is exactly how LTRIM shipped deleting whole lists.

Every assertion is differential against a real redis-server when one is
reachable, so "what should this return" is a diff and not an opinion.

Usage: python3 tests/test_gh265_lmove_quicklist.py [--port 1974] [--redis-port 6399]
"""
import argparse, socket, sys

SMALL = 10        # ziplist side
BIG = 1500        # quicklist side (threshold is 1024)

failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


class Client:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=20)
        self.f = self.sock.makefile("rb")

    def __call__(self, *args):
        parts = [f"*{len(args)}\r\n".encode()]
        for a in args:
            a = str(a).encode()
            parts.append(b"$%d\r\n%s\r\n" % (len(a), a))
        self.sock.sendall(b"".join(parts))
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise RuntimeError("server closed")
        t, body = line[:1], line[1:-2]
        if t == b"+":
            return body.decode()
        if t == b"-":
            return "ERR:" + body.decode().split()[0]
        if t == b":":
            return int(body)
        if t == b"$":
            n = int(body); return None if n == -1 else self.f.read(n + 2)[:-2].decode()
        if t == b"*":
            n = int(body); return [] if n <= 0 else [self._read() for _ in range(n)]
        return body.decode(errors="replace")


def seed(c, key, n, prefix="e"):
    c("DEL", key)
    # One RPUSH per element: the conversion to segmented mode happens on
    # overflow, so pushing in bulk would not exercise the same path.
    for i in range(n):
        c("RPUSH", key, f"{prefix}{i}")


def scenario(c, size, src_dir, dst_dir):
    """Run one LMOVE and report (reply, src_len, dst_len, dst_contents_head)."""
    seed(c, "q:src", size)
    c("DEL", "q:dst")
    r = c("LMOVE", "q:src", "q:dst", src_dir, dst_dir)
    return (r, c("LLEN", "q:src"), c("LLEN", "q:dst"), c("LRANGE", "q:dst", 0, 0))


def rpoplpush_scenario(c, size):
    seed(c, "q:src", size)
    c("DEL", "q:dst")
    r = c("RPOPLPUSH", "q:src", "q:dst")
    return (r, c("LLEN", "q:src"), c("LLEN", "q:dst"), c("LRANGE", "q:dst", 0, 0))


def drain_scenario(c, size):
    """The reliable-queue idiom: move EVERY element across, one at a time."""
    seed(c, "q:src", size)
    c("DEL", "q:dst")
    moved = 0
    for _ in range(size + 5):
        r = c("RPOPLPUSH", "q:src", "q:dst")
        if r is None:
            break
        moved += 1
    return (moved, c("LLEN", "q:src"), c("LLEN", "q:dst"), c("EXISTS", "q:src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--redis-port", type=int, default=6399)
    args = ap.parse_args()

    try:
        pion = Client(args.port)
    except OSError as e:
        print(f"FATAL: no Pion on {args.port} ({e})")
        return 2
    try:
        redis = Client(args.redis_port)
        redis("PING")
    except (OSError, RuntimeError):
        redis = None
        print(f"NOTE: no redis-server on {args.redis_port} — asserting against "
              f"expected values instead of a live oracle")

    for size, label in ((SMALL, "ziplist"), (BIG, "quicklist")):
        print(f"\n[{label}, {size} entries] LMOVE all four direction pairs")
        for sd in ("LEFT", "RIGHT"):
            for dd in ("LEFT", "RIGHT"):
                got = scenario(pion, size, sd, dd)
                name = f"{label}: LMOVE {sd} {dd}"
                if redis:
                    want = scenario(redis, size, sd, dd)
                    check(name, got == want, f"pion={got} redis={want}")
                else:
                    expect_elem = "e0" if sd == "LEFT" else f"e{size-1}"
                    check(name, got == (expect_elem, size - 1, 1, [expect_elem]),
                          f"got {got}")

        print(f"\n[{label}, {size} entries] RPOPLPUSH")
        got = rpoplpush_scenario(pion, size)
        if redis:
            want = rpoplpush_scenario(redis, size)
            check(f"{label}: RPOPLPUSH", got == want, f"pion={got} redis={want}")
        else:
            check(f"{label}: RPOPLPUSH",
                  got == (f"e{size-1}", size - 1, 1, [f"e{size-1}"]), f"got {got}")

        print(f"\n[{label}, {size} entries] the reliable-queue idiom drains fully")
        got = drain_scenario(pion, size)
        if redis:
            want = drain_scenario(redis, size)
            check(f"{label}: drain moves every element", got == want,
                  f"pion={got} redis={want}")
        else:
            # gh #234: an emptied list must be REMOVED, so EXISTS is 0.
            check(f"{label}: drain moves every element", got == (size, 0, size, 0),
                  f"got {got} (moved {got[0]} of {size})")

    print("\n[cross-representation] a move from a quicklist INTO a ziplist and back")
    seed(pion, "q:src", BIG)
    seed(pion, "q:dst", SMALL, prefix="d")
    r1 = pion("LMOVE", "q:src", "q:dst", "RIGHT", "LEFT")
    got = (r1, pion("LLEN", "q:src"), pion("LLEN", "q:dst"), pion("LRANGE", "q:dst", 0, 0))
    if redis:
        seed(redis, "q:src", BIG); seed(redis, "q:dst", SMALL, prefix="d")
        r2 = redis("LMOVE", "q:src", "q:dst", "RIGHT", "LEFT")
        want = (r2, redis("LLEN", "q:src"), redis("LLEN", "q:dst"),
                redis("LRANGE", "q:dst", 0, 0))
        check("quicklist -> ziplist move", got == want, f"pion={got} redis={want}")
    else:
        check("quicklist -> ziplist move",
              got == (f"e{BIG-1}", BIG - 1, SMALL + 1, [f"e{BIG-1}"]), f"got {got}")

    print("\n[framing] the connection is still synced")
    check("PING after everything", pion("PING") == "PONG")

    for k in ("q:src", "q:dst"):
        pion("DEL", k)
        if redis: redis("DEL", k)

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
