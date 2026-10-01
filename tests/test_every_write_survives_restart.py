#!/usr/bin/env python3
"""Every write command's effect must survive a restart — in every crash mode.

WHY
Durability is per-handler: each mutating command must append an effect
record to the WAL (or route through a dispatcher that does). A handler that
forgets is invisible to every test that reads its reply, and to durability
tests that only exercise the commands someone thought of: RENAME and the
*STORE family (SUNIONSTORE, ZUNIONSTORE, …) logged nothing, so their results
vanished on restart — and, because a replica decodes the same WAL, never
reached a replica either.

This runs one invocation of (nearly) every command Redis flags `write`, each
on its own key, then dumps the WHOLE keyspace (every key: TYPE, a full
type-specific read, and whether it has a TTL), restarts, and requires the
identical dump, in four modes:

  kill          SIGKILL, restart            (WAL replay only)
  save+kill     SAVE, SIGKILL, restart      (snapshot load only)
  save+more     SAVE, the writes again on new keys, SIGKILL
                                            (snapshot + WAL tail)
  rotate        --wal-size 1, SIGKILL       (replay across sealed segments;
                                             the test ASSERTS a rotation
                                             happened, or it proves nothing)
  rewrite       BGREWRITEAOF, SIGKILL       (the rewritten log alone — the
                                             old rewrite dropped every stream
                                             and every TTL)

Readiness after restart is a PING reply, never TCP accept: the server
listens before it has replayed.

    python3 tests/test_every_write_survives_restart.py [--port 2601] [--mode all]
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

BIN = os.environ.get("PION_BIN", "./pion-server")
L = b"L" * 40      # heap-sized payloads (SSO ends at 23 bytes)


def workload(p: str):
    """(command, …) tuples. {p} keeps a run's keys in one hash slot and apart
    from the next run's (save+more replays the workload under a new prefix)."""
    k = lambda n: f"{{{p}}}{n}"
    return [
        # strings
        ("SET", k("set"), "v"), ("SET", k("setl"), L), ("SETNX", k("setnx"), "v"),
        ("SETEX", k("setex"), "5000", "v"), ("PSETEX", k("psetex"), "5000000", "v"),
        ("SET", k("setex2"), "v", "EX", "5000"), ("GETSET", k("set"), "v2"),
        ("MSET", k("m1"), "a", k("m2"), L), ("MSETNX", k("mnx1"), "a", k("mnx2"), "b"),
        ("APPEND", k("app"), "x"), ("APPEND", k("app"), L), ("SETRANGE", k("rng"), "3", "abc"),
        ("INCR", k("ctr")), ("INCRBY", k("ctr"), "10"), ("DECR", k("ctr")), ("DECRBY", k("ctr"), "2"),
        ("INCRBYFLOAT", k("flt"), "2.5"),
        ("SET", k("gd"), "x"), ("GETDEL", k("gd")),
        ("SET", k("gex"), "x"), ("GETEX", k("gex"), "EX", "5000"),
        ("SET", k("del"), "x"), ("DEL", k("del")), ("SET", k("unl"), "x"), ("UNLINK", k("unl")),
        # keyspace
        ("SET", k("ren"), L), ("RENAME", k("ren"), k("ren2")),
        ("SET", k("rnx"), "x"), ("RENAMENX", k("rnx"), k("rnx2")),
        ("SET", k("cpy"), L), ("COPY", k("cpy"), k("cpy2")),
        ("SET", k("exp"), "x"), ("EXPIRE", k("exp"), "5000"),
        ("SET", k("pexp"), "x"), ("PEXPIRE", k("pexp"), "5000000"),
        ("SET", k("expat"), "x"), ("EXPIREAT", k("expat"), "4102444800"),
        ("SET", k("pexpat"), "x"), ("PEXPIREAT", k("pexpat"), "4102444800000"),
        ("SET", k("pers"), "x"), ("EXPIRE", k("pers"), "5000"), ("PERSIST", k("pers")),
        # hashes
        ("HSET", k("h"), "f", "v", "g", L), ("HSETNX", k("h"), "n", "1"), ("HMSET", k("h"), "o", "p"),
        ("HDEL", k("h"), "o"), ("HINCRBY", k("h"), "i", "7"), ("HINCRBYFLOAT", k("h"), "fl", "1.5"),
        ("HSET", k("hx"), "f", "v", "g", "w"), ("HEXPIRE", k("hx"), "5000", "FIELDS", "1", "f"),
        ("HSET", k("hp"), "f", "v"), ("HPEXPIRE", k("hp"), "5000000", "FIELDS", "1", "f"),
        ("HPERSIST", k("hp"), "FIELDS", "1", "f"),
        ("HSET", k("hxa"), "f", "v", "g", "w"), ("HEXPIREAT", k("hxa"), "4102444800", "FIELDS", "1", "g"),
        ("HSET", k("hpa"), "f", "v"), ("HPEXPIREAT", k("hpa"), "4102444800000", "FIELDS", "1", "f"),
        ("HSET", k("hxr"), "f", L, "g", "w"), ("HEXPIRE", k("hxr"), "5000", "FIELDS", "1", "f"),
        ("RENAME", k("hxr"), k("hxr2")),
        # lists
        ("RPUSH", k("l"), "a", "b", "c", L), ("LPUSH", k("l"), "z"), ("LPUSHX", k("l"), "y"),
        ("RPUSHX", k("l"), "w"), ("LPOP", k("l")), ("RPOP", k("l")), ("LSET", k("l"), "0", "A"),
        ("LINSERT", k("l"), "BEFORE", "b", "bb"), ("LREM", k("l"), "1", "c"),
        ("RPUSH", k("lt"), "1", "2", "3", "4"), ("LTRIM", k("lt"), "1", "2"),
        ("RPUSH", k("lm"), "a", "b"), ("LMOVE", k("lm"), k("lm2"), "LEFT", "RIGHT"),
        ("RPOPLPUSH", k("lm"), k("lm2")),
        ("RPUSH", k("lmp"), "a", "b", "c"), ("LMPOP", "1", k("lmp"), "LEFT", "COUNT", "2"),
        ("RPUSH", k("big"), *[b"e%04d" % i for i in range(1100)]),     # segmented
        ("RPOP", k("big")), ("LPOP", k("big")),
        # sets
        ("SADD", k("s"), "a", "b", "c", L), ("SREM", k("s"), "c"),
        ("SADD", k("s2"), "b", "x"), ("SMOVE", k("s2"), k("s"), "x"),
        ("SADD", k("sp"), "only"), ("SPOP", k("sp")),
        ("SUNIONSTORE", k("su"), k("s"), k("s2")), ("SINTERSTORE", k("si"), k("s"), k("s2")),
        ("SDIFFSTORE", k("sd"), k("s"), k("s2")),
        # sorted sets
        ("ZADD", k("z"), "1", "a", "2", "b", "3", L, "4", "d", "5", "e"), ("ZINCRBY", k("z"), "0.5", "a"),
        ("ZREM", k("z"), "e"), ("ZPOPMIN", k("z")), ("ZPOPMAX", k("z")),
        ("ZADD", k("z2"), "10", "b", "20", "q"),
        ("ZUNIONSTORE", k("zu"), "2", k("z"), k("z2")), ("ZINTERSTORE", k("zi"), "2", k("z"), k("z2")),
        ("ZDIFFSTORE", k("zd"), "2", k("z"), k("z2")), ("ZRANGESTORE", k("zr"), k("z"), "0", "-1"),
        ("ZADD", k("zl"), "0", "a", "0", "b", "0", "c", "0", "d"), ("ZREMRANGEBYLEX", k("zl"), "[a", "[b"),
        ("ZADD", k("zk"), "1", "a", "2", "b", "3", "c"), ("ZREMRANGEBYRANK", k("zk"), "0", "0"),
        ("ZREMRANGEBYSCORE", k("zk"), "3", "3"),
        ("ZADD", k("zm"), "1", "a", "2", "b"), ("ZMPOP", "1", k("zm"), "MIN"),
        # streams
        ("XADD", k("x"), "1-1", "f", "v"), ("XADD", k("x"), "2-1", "g", L), ("XADD", k("x"), "3-1", "h", "w"),
        ("XDEL", k("x"), "1-1"), ("XTRIM", k("x"), "MAXLEN", "1"),
        # hll / bitmaps / geo
        ("PFADD", k("hll"), "a", "b", "c"), ("PFADD", k("hll2"), "c", "d"),
        ("PFMERGE", k("hllm"), k("hll"), k("hll2")),
        ("SETBIT", k("bm"), "7", "1"), ("SETBIT", k("bm"), "300", "1"),
        ("SET", k("bs1"), "abc"), ("SET", k("bs2"), "abd"), ("BITOP", "XOR", k("bx"), k("bs1"), k("bs2")),
        ("BITFIELD", k("bf"), "SET", "u8", "0", "200"),
        ("GEOADD", k("g"), "13.361389", "38.115556", "Palermo", "15.087269", "37.502669", "Catania"),
        ("GEOSEARCHSTORE", k("gs"), k("g"), "FROMLONLAT", "15", "37", "BYRADIUS", "200", "km", "ASC"),
        # vector sets
        ("VADD", k("v"), "VALUES", "3", "1", "0", "0", "e1"), ("VADD", k("v"), "VALUES", "3", "0", "1", "0", "e2"),
        ("VSETATTR", k("v"), "e1", '{"a":1}'), ("VREM", k("v"), "e2"),
        # sort store, scripts
        ("RPUSH", k("so"), "3", "1", "2"), ("SORT", k("so"), "STORE", k("so2")),
        ("EVAL", "return redis.call('SET', KEYS[1], ARGV[1])", "1", k("lua"), "fromlua"),
    ]


def _field_ttl(c: Conn, key, field):
    t = c.cmd("HPTTL", key, "FIELDS", "1", field)[0]
    return "ttl" if t > 0 else t


def dump(c: Conn):
    """Every key: (TYPE, TTL presence, full contents)."""
    keys, cur = set(), "0"
    while True:
        cur, batch = c.cmd("SCAN", cur, "COUNT", "1000")
        keys.update(batch)
        if cur in (b"0", "0"):
            break
    out = {}
    for key in sorted(keys):
        t = c.cmd("TYPE", key)
        ttl = c.cmd("PTTL", key)
        body = {"string": lambda: c.cmd("GET", key),
                # gh #392: with each field's TTL presence — HEXPIRE was in the
                # workload all along, but only HGETALL was compared, so field
                # TTLs vanishing on every restart went unnoticed.
                "hash": lambda: sorted((f, v, _field_ttl(c, key, f))
                                       for f, v in zip(*[iter(c.cmd("HGETALL", key))] * 2)),
                "list": lambda: c.cmd("LRANGE", key, "0", "-1"),
                "set": lambda: sorted(c.cmd("SMEMBERS", key)),
                "zset": lambda: c.cmd("ZRANGE", key, "0", "-1", "WITHSCORES"),
                "stream": lambda: c.cmd("XRANGE", key, "-", "+"),
                "vectorset": lambda: (c.cmd("VCARD", key), c.cmd("VEMB", key, "e1"),
                                      c.cmd("VGETATTR", key, "e1")),
                }.get(t, lambda: ("?", t))()
        if key.endswith(b"hll") or key.endswith(b"hll2") or key.endswith(b"hllm"):
            body = ("pfcount", c.cmd("PFCOUNT", key))
        out[key] = (t, "ttl" if isinstance(ttl, int) and ttl > 0 else ttl, body)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2601)
    ap.add_argument("--mode", default="all",
                    choices=["all", "kill", "save+kill", "save+more", "rotate", "rewrite"])
    args = ap.parse_args()
    modes = ["kill", "save+kill", "save+more", "rotate", "rewrite"] if args.mode == "all" else [args.mode]
    fails = []
    for mode in modes:
        work = tempfile.mkdtemp(prefix=f"pion-dur-{mode.replace('+', '-')}-")
        extra = ["--wal-size", "1"] if mode == "rotate" else []
        cmd = [os.path.abspath(BIN), "-p", str(args.port), "-w", "1", "--no-crash-log",
               "--no-auto-detect", "--no-auto-embed"] + extra
        log = open(os.path.join(work, "server.log"), "a")
        proc = subprocess.Popen(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(args.port, 30, proc=proc)
            c = Conn(args.port, timeout=30)
            refused = []
            reps = 12 if mode == "rotate" else 1   # enough bytes to seal segments at --wal-size 1
            for r in range(reps):
                for w in workload(f"a{r}"):
                    res = c.cmd(*w)
                    if isinstance(res, RespError):
                        refused.append((w[0], str(res)[:60]))
            if mode == "rotate":
                # ~2 MiB of plain SETs so at least one 1 MiB segment must seal.
                c.pipeline([("SET", f"{{fill}}{i}", b"F" * 32768) for i in range(64)])
            if mode in ("save+kill", "save+more"):
                if c.cmd("SAVE") != "OK":
                    fails.append(f"[{mode}] SAVE did not return OK")
            if mode == "save+more":
                for w in workload("b0"):
                    c.cmd(*w)
            if mode == "rewrite":
                if c.cmd("BGREWRITEAOF") not in ("OK", "Background append only file rewriting started"):
                    fails.append("[rewrite] BGREWRITEAOF was refused")
            c.assert_in_sync()
            before = dump(c)
            if mode == "rotate":
                info = c.cmd("INFO", "persistence")
                sealed = [ln for ln in info.split(b"\r\n") if ln.startswith(b"wal_segments_sealed:")]
                if not sealed or int(sealed[0].split(b":")[1]) < 1:
                    fails.append(f"[rotate] no WAL segment was sealed ({sealed}) — the mode proved nothing")
            proc.kill(); proc.wait()
            proc = subprocess.Popen(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT)
            wait_ready(args.port, 60, proc=proc)
            after = dump(Conn(args.port, timeout=30))
            lost = [k for k in before if k not in after]
            extra_keys = [k for k in after if k not in before]
            changed = [k for k in before if k in after and before[k] != after[k]]
            for k in lost[:12]:
                fails.append(f"[{mode}] LOST {k!r}: {str(before[k])[:110]}")
            for k in extra_keys[:6]:
                fails.append(f"[{mode}] RESURRECTED {k!r}: {str(after[k])[:110]}")
            for k in changed[:12]:
                fails.append(f"[{mode}] CHANGED {k!r}:\n        before {str(before[k])[:110]}\n"
                             f"        after  {str(after[k])[:110]}")
            if len(lost) > 12 or len(changed) > 12:
                fails.append(f"[{mode}] … {len(lost)} lost, {len(changed)} changed in total")
            print(f"[{mode}] {len(before)} keys before restart: {len(lost)} lost, "
                  f"{len(extra_keys)} resurrected, {len(changed)} changed"
                  + (f"; refused at write time: {sorted(set(x[0] for x in refused))}" if refused else ""))
        finally:
            proc.kill(); proc.wait()
            shutil.rmtree(work, ignore_errors=True)
    for f in fails:
        print("  FAIL " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
