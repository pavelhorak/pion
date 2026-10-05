#!/usr/bin/env python3
"""Every numeric argument, fed malformed and edge values, against Redis 8.

WHY
gh #229 found INCRBY's argument parsed by a loop that SKIPPED non-digits —
`INCRBY k abc` added 0 and replied with an integer, i.e. a silent success —
and fixed the handlers it looked at. The same loop survived elsewhere:
SETEX / PSETEX / MSETEX skip non-digits in the TTL (`SETEX k -5 v` expires in
5 s, `SETEX k abc v` expires at once, both replying +OK), and the stored-value
parse of INCR/DECR wrapped past Int64. A per-handler fix converts the sites
someone looked at; this sweeps the argument POSITIONS.

For each (command, numeric position) it sends each value in BAD (and a few
legal edge values), on a freshly built fixture, and compares with Redis:
  * the reply (error replies compare their code — the first word);
  * the key's state afterwards (a refused command must change nothing, and
    an accepted one must change it the same way).
TTLs are compared as "none / has one / gone", never as a number.

    python3 tests/test_numeric_args_differential.py --pion-port 1974 --start-redis
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, wait_ready  # noqa: E402

INT_BAD = ["abc", "", "1abc", "1.5", "-1", "0", "99999999999999999999", " 1", "+1", "0x10",
           "1e3", "-0", "01", "9223372036854775807", "-9223372036854775808", "1 ", "\x001"]
FLOAT_BAD = ["abc", "", "1abc", "nan", "inf", "-inf", "1e400", "0x10", " 1.5", "1.5 ", "1,5",
             "--1", "1e", ".", "+.5", "5.", "-0"]

FIX = {
    "str": [("SET", "k", "hello")],
    "num": [("SET", "k", "10")],
    "list": [("RPUSH", "k", "a", "b", "c", "d", "e")],
    "hash": [("HSET", "k", "f", "5", "g", "x")],
    "set": [("SADD", "k", "a", "b", "c")],
    "zset": [("ZADD", "k", "1", "a", "2", "b", "3", "c")],
    "bitmap": [("SETBIT", "k", "7", "1")],
    "none": [],
}

# (fixture, command template with "#" at the numeric position, kind)
PROBES = [
    ("str", ("EXPIRE", "k", "#"), "int"), ("str", ("PEXPIRE", "k", "#"), "int"),
    ("str", ("EXPIREAT", "k", "#"), "int"), ("str", ("PEXPIREAT", "k", "#"), "int"),
    ("none", ("SETEX", "k", "#", "v"), "int"), ("none", ("PSETEX", "k", "#", "v"), "int"),
    ("none", ("SET", "k", "v", "EX", "#"), "int"), ("none", ("SET", "k", "v", "PX", "#"), "int"),
    ("str", ("GETEX", "k", "EX", "#"), "int"),
    ("num", ("INCRBY", "k", "#"), "int"), ("num", ("DECRBY", "k", "#"), "int"),
    ("num", ("INCRBYFLOAT", "k", "#"), "float"),
    ("hash", ("HINCRBY", "k", "f", "#"), "int"), ("hash", ("HINCRBYFLOAT", "k", "f", "#"), "float"),
    ("list", ("LRANGE", "k", "#", "2"), "int"), ("list", ("LRANGE", "k", "0", "#"), "int"),
    ("list", ("LINDEX", "k", "#"), "int"), ("list", ("LSET", "k", "#", "v"), "int"),
    ("list", ("LTRIM", "k", "#", "-1"), "int"), ("list", ("LPOP", "k", "#"), "int"),
    ("list", ("RPOP", "k", "#"), "int"), ("list", ("LPOS", "k", "a", "RANK", "#"), "int"),
    ("list", ("LPOS", "k", "a", "COUNT", "#"), "int"), ("list", ("LREM", "k", "#", "a"), "int"),
    ("set", ("SPOP", "k", "#"), "int"), ("set", ("SRANDMEMBER", "k", "#"), "int"),
    ("zset", ("ZADD", "k", "#", "m"), "float"), ("zset", ("ZINCRBY", "k", "#", "a"), "float"),
    ("zset", ("ZRANGE", "k", "#", "-1"), "int"), ("zset", ("ZCOUNT", "k", "#", "+inf"), "float"),
    ("zset", ("ZRANGEBYSCORE", "k", "-inf", "#"), "float"),
    ("zset", ("ZRANGEBYSCORE", "k", "-inf", "+inf", "LIMIT", "#", "1"), "int"),
    ("zset", ("ZPOPMIN", "k", "#"), "int"), ("zset", ("ZREMRANGEBYRANK", "k", "#", "0"), "int"),
    ("zset", ("ZPOPMAX", "k", "#"), "int"),
    ("zset", ("ZRANGEBYSCORE", "k", "-inf", "+inf", "LIMIT", "0", "#"), "int"),
    ("zset", ("ZRANGEBYLEX", "k", "-", "+", "LIMIT", "#", "1"), "int"),
    ("zset", ("ZREVRANGEBYLEX", "k", "+", "-", "LIMIT", "#", "1"), "int"),
    ("zset", ("ZREVRANGEBYSCORE", "k", "+inf", "-inf", "LIMIT", "#", "1"), "int"),
    ("none", ("SET", "k", "v", "EXAT", "#"), "int"), ("none", ("SET", "k", "v", "PXAT", "#"), "int"),
    ("str", ("GETRANGE", "k", "#", "2"), "int"), ("str", ("SETRANGE", "k", "#", "x"), "int"),
    ("str", ("GETBIT", "k", "#"), "int"), ("str", ("SETBIT", "k", "#", "1"), "int"),
    ("str", ("SETBIT", "k", "1", "#"), "int"), ("str", ("BITCOUNT", "k", "#", "-1"), "int"),
    ("str", ("BITPOS", "k", "1", "#"), "int"),
    ("hash", ("HRANDFIELD", "k", "#"), "int"), ("zset", ("ZRANDMEMBER", "k", "#"), "int"),
    ("none", ("SCAN", "#"), "int"), ("set", ("SSCAN", "k", "0", "COUNT", "#"), "int"),
    ("set", ("SINTERCARD", "#", "k"), "int"), ("zset", ("ZUNION", "#", "k"), "int"),
    ("hash", ("HEXPIRE", "k", "#", "FIELDS", "1", "f"), "int"),
    ("hash", ("HEXPIRE", "k", "100", "FIELDS", "#", "f"), "int"),
    ("bitmap", ("BITFIELD", "k", "GET", "u8", "#"), "int"),
    ("bitmap", ("BITFIELD", "k", "SET", "u8", "#", "1"), "int"),
    ("bitmap", ("BITFIELD", "k", "INCRBY", "u8", "#", "1"), "int"),
]


# Known divergences, gh #393: (template, value). A divergence NOT listed here
# fails the test; a listed one that now AGREES with Redis also fails — remove
# it on purpose, so a fix cannot be lost silently (the XPASS rule).
KNOWN: set = set()   # filled from KNOWN_FILE when present
KNOWN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numeric_args_known.tsv")


def load_known():
    if os.path.exists(KNOWN_FILE):
        for line in open(KNOWN_FILE, encoding="utf-8"):
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                tmpl, _, val = line.partition("\t")
                KNOWN.add((tmpl, val.encode("latin-1").decode("unicode_escape")))


def state(c: Conn):
    t = c.cmd("TYPE", "k")
    ttl = c.cmd("PTTL", "k")
    ttl = ttl if ttl in (-1, -2) else "ttl"
    body = {"string": lambda: c.cmd("GET", "k"),
            "list": lambda: c.cmd("LRANGE", "k", "0", "-1"),
            "hash": lambda: sorted(zip(*[iter(c.cmd("HGETALL", "k"))] * 2)),
            "set": lambda: sorted(c.cmd("SMEMBERS", "k")),
            "zset": lambda: c.cmd("ZRANGE", "k", "0", "-1", "WITHSCORES"),
            }.get(t, lambda: None)()
    return (t, ttl, body)


def norm(cmd, r):
    if isinstance(r, RespError):
        return ("ERR", r.split(" ", 1)[0])
    if cmd[0] in ("SPOP", "SRANDMEMBER", "HRANDFIELD", "ZRANDMEMBER"):
        # random members: compare the SHAPE (count, or single vs nil)
        return ("random", len(r) if isinstance(r, list) else r is not None)
    if cmd[0] == "SCAN" and isinstance(r, list):
        return ("scan", len(r))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pion-port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    ap.add_argument("--redis-port", type=int, default=6418)
    ap.add_argument("--start-redis", action="store_true")
    ap.add_argument("--write-known", action="store_true",
                    help="record the current divergences as the known list (gh #393) and exit 0")
    args = ap.parse_args()
    rproc = rdir = None
    if args.start_redis:
        rdir = tempfile.mkdtemp(prefix="num-redis-")
        rproc = subprocess.Popen(["redis-server", "--port", str(args.redis_port), "--save", "",
                                  "--appendonly", "no", "--dir", rdir,
                                  "--databases", "1"],  # Pion has one database
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(args.redis_port, 15, proc=rproc)
        wait_ready(args.pion_port, 30)
        p, r = Conn(args.pion_port), Conn(args.redis_port)
        load_known()
        n, fails, agreed_known = 0, [], []
        # TWO passes on the same server, then allocator churn: an out-of-bounds
        # write does not fault where it happens. SETRANGE k 9223372036854775807 x
        # corrupted the heap silently and the server died in a LATER pass, in
        # an unrelated RPUSH's free().
        for pass_no in (1, 2):
          for fx, tmpl, kind in PROBES:
              for v in (INT_BAD if kind == "int" else FLOAT_BAD):
                  cmd = tuple(v if a == "#" else a for a in tmpl)
                  got = []
                  for c in (p, r):
                      c.cmd("DEL", "k")
                      for s in FIX[fx]:
                          c.cmd(*s)
                      try:
                          rep = norm(cmd, c.cmd_synced(*cmd))
                      except (ConnectionError, TimeoutError) as e:
                          print(f"FAIL {'pion' if c is p else 'redis'} lost the connection on "
                                f"{cmd!r} ({type(e).__name__}) — a crash or a hang; stopping")
                          return 1
                      got.append((rep, state(c)))
                  if pass_no == 2:
                      continue          # pass 2 only has to RUN; pass 1 compared
                  n += 1
                  key = (" ".join(tmpl), v)
                  if got[0] != got[1] and key not in KNOWN:
                      fails.append((tmpl, cmd, got[0], got[1]))
                  elif got[0] == got[1] and key in KNOWN:
                      agreed_known.append(key)
        survived = True
        try:
            for i in range(3000):
                p.pipeline([("RPUSH", f"churn{i % 7}", "a", b"x" * 40), ("DEL", f"churn{(i + 3) % 7}"),
                            ("SET", f"cs{i % 9}", b"y" * 60)])
            survived = p.cmd("PING") == "PONG"
        except (ConnectionError, TimeoutError):
            survived = False
        if not survived:
            print("FAIL the server died or stopped answering during the churn after two passes — "
                  "a probe corrupted memory without faulting on the spot")
            return 1
        if args.write_known:
            with open(KNOWN_FILE, "w", encoding="utf-8") as fh:
                fh.write("# gh #393: known numeric-argument divergences (template<TAB>value).\n")
                for tmpl, cmd, _, _ in fails:
                    val = cmd[tmpl.index("#")]
                    fh.write(" ".join(tmpl) + "\t" + val.encode("unicode_escape").decode("latin-1") + "\n")
            print(f"wrote {len(fails)} entries to {KNOWN_FILE}")
            return 0
        print(f"{n} numeric-argument probes, {len(fails)} new divergences from Redis "
              f"({len(KNOWN)} known, gh #393), {len(agreed_known)} known ones now agree")
        for key in agreed_known:
            print(f"  XPASS {key[0]!r} value {key[1]!r} now agrees with Redis — remove it from {os.path.basename(KNOWN_FILE)}")
        by = {}
        for tmpl, cmd, gp, gr in fails:
            by.setdefault(" ".join(tmpl), []).append((cmd, gp, gr))
        for key, rows in by.items():
            cmd, gp, gr = rows[0]
            vals = ", ".join(repr(c[key.split().index("#")]) for c, _, _ in rows)
            print(f"  {key:36} x{len(rows):<3} values: {vals[:110]}")
            print(f"      pion : {str(gp)[:170]}")
            print(f"      redis: {str(gr)[:170]}")
        return 1 if fails or agreed_known else 0
    finally:
        if rproc:
            rproc.kill(); rproc.wait()
        if rdir:
            shutil.rmtree(rdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
