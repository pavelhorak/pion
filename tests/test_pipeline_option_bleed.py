#!/usr/bin/env python3
"""A command's option scan must stop at the end of ITS OWN frame.

WHY
SET's option loop ran to the end of the whole pipelined batch (num_tokens)
instead of its own frame (cmd_end_tok). So the most ordinary pipeline there
is —

    SET k v EX 100
    GET k

— had SET take the next command's name as its own `GET` option (it answered
the OLD value) and the stray `k` came back as "ERR unknown command 'k'".
`SET k v XX` + `NX` and `SET k v KEEPTTL` + `EX 5` hung the connection. PFADD
took the next command's tokens as HLL elements. test_dispatch_sweep.py sends
each command followed by PING, and PING is nobody's option name, so it could
not see any of it.

This pipelines FIRST + NEXT + PING for every pair below, where each NEXT is a
real command whose NAME is also an option keyword of some command (GET, SET,
PERSIST, TYPE, INCRBY, COPY, KEYS, EXISTS …), and compares every reply with a
live Redis 8. A bleed shows up as a wrong reply, an extra reply, or a hang.

    python3 tests/test_pipeline_option_bleed.py --pion-port 1974 --start-redis
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

FIXTURES = [("SET", "k", "v0"), ("SET", "c", "5"), ("RPUSH", "l", "3", "1", "2"), ("SADD", "s", "a", "b"),
            ("ZADD", "z", "1", "a", "2", "b"), ("HSET", "h", "f", "1"), ("PFADD", "hl", "x"),
            ("XADD", "x", "1-1", "f", "v"), ("SET", "bf", "ab")]

FIRST = [
    ("SET", "k", "v1"), ("SET", "k", "v1", "EX", "100"), ("SET", "k", "v1", "NX"),
    ("SET", "k", "v1", "XX"), ("SET", "k", "v1", "KEEPTTL"), ("SET", "k", "v1", "GET"),
    ("GETEX", "k"), ("GETEX", "k", "EX", "100"), ("PFADD", "hl", "y"), ("SORT", "l"),
    ("SORT", "l", "LIMIT", "0", "2"), ("BITFIELD", "bf", "GET", "u8", "0"), ("ZADD", "z", "3", "c"),
    ("ZRANGE", "z", "0", "-1"), ("LPOS", "l", "1"), ("SINTERCARD", "1", "s"), ("ZINTERCARD", "1", "z"),
    ("LMPOP", "1", "l", "LEFT"), ("ZMPOP", "1", "z", "MIN"), ("EXPIRE", "k", "100"),
    ("HGETALL", "h"), ("XRANGE", "x", "-", "+"), ("SSCAN", "s", "0"), ("ZSCORE", "z", "a"),
    ("GETRANGE", "k", "0", "1"), ("SETRANGE", "k", "1", "Z"), ("INCRBY", "c", "2"),
    ("LRANGE", "l", "0", "-1"), ("OBJECT", "ENCODING", "k"), ("COPY", "k", "k2"),
]

NEXT = [("GET", "k"), ("SET", "k", "v2"), ("PERSIST", "k"), ("TYPE", "k"), ("INCRBY", "c", "1"),
        ("EXISTS", "k"), ("COPY", "k", "k3"), ("KEYS", "k"), ("PING",)]

RANDOM = {"SPOP", "SRANDMEMBER"}


def norm(cmd, r):
    if isinstance(r, RespError):
        return ("ERR", r.split(" ", 1)[0])
    if cmd[0] == "OBJECT":
        return "encoding"                     # implementation-private
    if cmd[0] in ("SSCAN",) and isinstance(r, list):
        return ("scan", sorted(r[1]))
    if cmd[0] in ("KEYS", "HGETALL") and isinstance(r, list):
        return sorted(r, key=repr)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pion-port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    ap.add_argument("--redis-port", type=int, default=6421)
    ap.add_argument("--start-redis", action="store_true")
    args = ap.parse_args()
    rproc = rdir = None
    if args.start_redis:
        rdir = tempfile.mkdtemp(prefix="bleed-redis-")
        rproc = subprocess.Popen(["redis-server", "--port", str(args.redis_port), "--save", "",
                                  "--appendonly", "no", "--dir", rdir],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fails, n = [], 0
    try:
        wait_ready(args.redis_port, 15, proc=rproc)
        wait_ready(args.pion_port, 30)
        p, r = Conn(args.pion_port, timeout=5), Conn(args.redis_port, timeout=5)
        for first in FIRST:
            for nxt in NEXT:
                n += 1
                got = {}
                for name, c, port in (("pion", p, args.pion_port), ("redis", r, args.redis_port)):
                    c.cmd("FLUSHALL")
                    for fx in FIXTURES:
                        c.cmd(*fx)
                    try:
                        replies = c.pipeline([first, nxt, ("PING",)])
                        got[name] = [norm(cmd, x) for cmd, x in zip((first, nxt, ("PING",)), replies)]
                        c.assert_in_sync()
                    except (TimeoutError, ConnectionError, RespProtocolError) as e:
                        got[name] = f"FRAMING {type(e).__name__}: {str(e)[:100]}"
                        c.close()
                        c = Conn(port, timeout=5)
                        if name == "pion":
                            p = c
                        else:
                            r = c
                if got["pion"] != got["redis"]:
                    fails.append((first, nxt, got["pion"], got["redis"]))
        print(f"{n} pipelined pairs, {len(fails)} diverge from Redis")
        for first, nxt, gp, gr in fails[:40]:
            print(f"  FAIL {' '.join(first)}  +  {' '.join(nxt)}")
            print(f"      pion : {str(gp)[:160]}")
            print(f"      redis: {str(gr)[:160]}")
        return 1 if fails else 0
    finally:
        if rproc:
            rproc.kill(); rproc.wait()
        if rdir:
            shutil.rmtree(rdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
