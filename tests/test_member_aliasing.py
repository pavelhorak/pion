#!/usr/bin/env python3
"""Commands that copy members between containers must not SHARE them.

WHY
A GenericValue holding a string > 23 bytes is a pointer to a heap payload, and
SlabHashMap.set() stores the GenericValue it is given — a shallow copy. A
command that inserts another container's member into a new container
therefore aliases the payload unless it deep-copies, and whichever owner is
freed first leaves the other serving freed memory:

  * SUNION built its reply in a temporary set of shallow copies, then
    destroyed the temporary — which frees every key — so a READ-ONLY command
    freed the source set's members. The next read returned allocator
    free-list pointers as the first 8 bytes of each member.
  * SUNIONSTORE stored the shallow copies, so the destination and each source
    owned the same payloads: DEL of either is a use-after-free for the other.

Tests with short (SSO, inline) members cannot see this, and a read straight
after the command usually still finds the freed bytes intact. So each
scenario uses members well past 23 bytes, deletes one owner, CHURNS the
allocator with same-sized values (so freed chunks are handed out again), and
only then reads the survivor — compared byte-for-byte with Redis 8.

    python3 tests/test_member_aliasing.py --pion-port 1974 --start-redis
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, RespProtocolError, wait_ready  # noqa: E402

M = [b"member-%02d-" % i + b"x" * 40 for i in range(8)]         # heap members
S = [b"s%d" % i for i in range(4)]                                 # SSO members


def norm(r, ordered):
    if isinstance(r, RespError):
        return ("ERR", r.split(" ", 1)[0])
    if isinstance(r, list) and not ordered:
        return sorted(r, key=repr)
    return r


def churn(c: Conn, tag: str):
    """Hand freed chunks back out: many values of the members' size class."""
    c.pipeline([("SET", f"churn:{tag}:{i}", b"Z" * len(M[0])) for i in range(400)])
    c.pipeline([("SADD", f"churnset:{tag}", b"Y%03d" % i + b"y" * 44) for i in range(200)])


# Each scenario: setup commands, the command under test, then the READERS run
# after (a) deleting the named owner and (b) churning.
def scenarios():
    set_setup = [("SADD", "a", *M[:6], *S), ("SADD", "b", *M[3:], S[0])]
    z_setup = [("ZADD", "za", *[x for i, m in enumerate(M[:6]) for x in (str(i), m)]),
               ("ZADD", "zb", *[x for i, m in enumerate(M[3:]) for x in (str(i * 10), m)])]
    h_setup = [("HSET", "ha", *[x for m in M[:5] for x in (m, m + b"-val")])]
    l_setup = [("RPUSH", "la", *M[:5])]
    rd_sets = [("SMEMBERS", "a"), ("SMEMBERS", "b"), ("SMEMBERS", "dst")]
    rd_z = [("ZRANGE", "za", "0", "-1", "WITHSCORES"), ("ZRANGE", "zb", "0", "-1", "WITHSCORES"),
            ("ZRANGE", "zdst", "0", "-1", "WITHSCORES")]
    out = []
    for cmd in (("SUNION", "a", "b"), ("SINTER", "a", "b"), ("SDIFF", "a", "b"),
                ("SINTERCARD", "2", "a", "b"), ("SRANDMEMBER", "a", "-20"),
                ("SMISMEMBER", "a", *M[:3])):
        out.append((f"read-only {cmd[0]}", set_setup, cmd, None, rd_sets[:2]))
    for cmd in (("SUNIONSTORE", "dst", "a", "b"), ("SINTERSTORE", "dst", "a", "b"),
                ("SDIFFSTORE", "dst", "a", "b"), ("SUNIONSTORE", "a", "a", "b"),
                ("COPY", "a", "dst")):
        for victim in ("a", "b", "dst"):
            out.append((f"{' '.join(cmd[:2])} then DEL {victim}", set_setup, cmd, victim, rd_sets))
    out.append(("SMOVE then DEL a", set_setup, ("SMOVE", "a", "b", M[1]), "a", rd_sets[:2]))
    out.append(("SMOVE then DEL b", set_setup, ("SMOVE", "a", "b", M[1]), "b", rd_sets[:2]))
    for cmd in (("ZUNION", "2", "za", "zb", "WITHSCORES"), ("ZINTER", "2", "za", "zb"),
                ("ZDIFF", "2", "za", "zb"), ("ZRANDMEMBER", "za", "-10")):
        out.append((f"read-only {cmd[0]}", z_setup, cmd, None, rd_z[:2]))
    for cmd in (("ZUNIONSTORE", "zdst", "2", "za", "zb"), ("ZINTERSTORE", "zdst", "2", "za", "zb"),
                ("ZDIFFSTORE", "zdst", "2", "za", "zb"), ("ZRANGESTORE", "zdst", "za", "0", "-1"),
                ("ZUNIONSTORE", "za", "2", "za", "zb"), ("COPY", "za", "zdst")):
        for victim in ("za", "zb", "zdst"):
            out.append((f"{' '.join(cmd[:2])} then DEL {victim}", z_setup, cmd, victim, rd_z))
    rd_h = [("HGETALL", "ha"), ("HGETALL", "hdst")]
    for cmd in (("HRANDFIELD", "ha", "-10", "WITHVALUES"), ("HGETALL", "ha"), ("HKEYS", "ha")):
        out.append((f"read-only {cmd[0]}", h_setup, cmd, None, rd_h[:1]))
    for victim in ("ha", "hdst"):
        out.append((f"COPY hash then DEL {victim}", h_setup, ("COPY", "ha", "hdst"), victim, rd_h))
    rd_l = [("LRANGE", "la", "0", "-1"), ("LRANGE", "ldst", "0", "-1")]
    for cmd in (("LMOVE", "la", "ldst", "LEFT", "RIGHT"), ("RPOPLPUSH", "la", "ldst"),
                ("COPY", "la", "ldst"), ("SORT", "la", "ALPHA", "STORE", "ldst")):
        for victim in ("la", "ldst"):
            out.append((f"{cmd[0]} then DEL {victim}", l_setup, cmd, victim, rd_l))
    return out


UNORDERED = {"SMEMBERS", "SUNION", "SINTER", "SDIFF", "HKEYS", "SRANDMEMBER", "HRANDFIELD",
             "ZRANDMEMBER", "HGETALL"}


def run(pion_port: int, redis_port: int):
    fails, n = [], 0
    pion, redis = Conn(pion_port, timeout=10), Conn(redis_port, timeout=10)
    for name, setup, cmd, victim, readers in scenarios():
        try:
            n += 1
            got = one(pion, redis, setup, cmd, victim, readers)
        except (TimeoutError, ConnectionError, RespProtocolError) as e:
            # A truncated or missing reply: that IS the failure. Reconnect so
            # the remaining scenarios still run.
            fails.append((name, [f"TRANSPORT {type(e).__name__}: {str(e)[:160]}"], ["(a complete reply)"]))
            pion.close(); redis.close()
            pion, redis = Conn(pion_port, timeout=10), Conn(redis_port, timeout=10)
            continue
        if got["pion"] != got["redis"]:
            fails.append((name, got["pion"], got["redis"]))
    return n, fails


def one(pion, redis, setup, cmd, victim, readers):
    if True:
        for c in (pion, redis):
            c.cmd("FLUSHALL")
            for s in setup:
                c.cmd(*s)
        got = {}
        for side, c in (("pion", pion), ("redis", redis)):
            steps = []
            r = c.cmd(*cmd)
            if cmd[0] in ("SRANDMEMBER", "ZRANDMEMBER", "HRANDFIELD"):
                r = ("random", isinstance(r, list) and len(r))
            steps.append(norm(r, cmd[0] not in UNORDERED))
            if victim:
                steps.append(c.cmd("DEL", victim))
            churn(c, side)
            for rd in readers:
                rr = c.cmd(*rd)
                if rd[0] == "HGETALL" and isinstance(rr, list):
                    rr = sorted(zip(rr[0::2], rr[1::2]))
                steps.append(norm(rr, rd[0] not in UNORDERED))
            got[side] = steps
            c.assert_in_sync()
        return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pion-port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    ap.add_argument("--redis-port", type=int, default=6416)
    ap.add_argument("--start-redis", action="store_true")
    args = ap.parse_args()
    rproc = rdir = None
    if args.start_redis:
        rdir = tempfile.mkdtemp(prefix="alias-redis-")
        rproc = subprocess.Popen(["redis-server", "--port", str(args.redis_port), "--save", "",
                                  "--appendonly", "no", "--dir", rdir],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 env=dict(os.environ, LC_ALL="C"))
    try:
        wait_ready(args.redis_port, 15, proc=rproc)
        wait_ready(args.pion_port, 30)
        n, fails = run(args.pion_port, args.redis_port)
        print(f"{n} ownership scenarios, {len(fails)} diverge from Redis")
        for name, p, r in fails:
            print(f"  FAIL {name}")
            for i, (a, b) in enumerate(zip(p, r)):
                if a != b:
                    print(f"     step {i}: pion  {str(a)[:180]}")
                    print(f"             redis {str(b)[:180]}")
                    break
        return 1 if fails else 0
    finally:
        if rproc:
            rproc.kill(); rproc.wait()
        if rdir:
            shutil.rmtree(rdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
