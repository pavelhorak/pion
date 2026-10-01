#!/usr/bin/env python3
"""A replica must hold what its primary holds — every type, not just strings.

WHY
The primary streams its raw WAL and the replica decodes it with
`apply_wal_entries`, the THIRD WAL decoder (after replay and the snapshot
loader). It understood 1 SET / 2 DEL / 31 MSET and silently skipped every
other record: hashes, lists, sets, zsets, streams, HLL, bitmaps, TTLs and
vector sets never reached a replica, and CLUSTER FAILOVER promoted a node
holding only the string keys. test_replication.py checked "found > 0" of 50
string keys, so it could not see any of this.

Second, the replica's read/write classifier was a hand-written allowlist of
~25 read commands: every other read (SMEMBERS, ZRANGE, HGETALL, XLEN, …) was
refused with -READONLY. A guard needs a probe of what it must ALLOW.

This test, on a fresh primary + replica pair:
  1. writes one of every record kind on the primary, including the
     non-deterministic ones that log their resolved effect (SPOP, ZPOPMIN,
     LPOP, XADD *, EXPIRE) and drains that must DELETE a key;
  2. waits for the replica, then dumps every key on both (TYPE + a full,
     type-specific read + TTL presence) and requires them identical;
  3. on the replica (READONLY): every read in READS must be answered exactly
     as the primary answers it; every write in WRITES must be refused AND
     change nothing.

    python3 tests/test_replication_all_types.py [--port 2201]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, wait_ready  # noqa: E402

BIN = os.environ.get("PION_BIN", "./pion-server")
L = b"x" * 40   # heap-sized payload

WORKLOAD = [
    ("SET", "{t}str", "v" + "s" * 30), ("APPEND", "{t}str", "!"),
    ("MSET", "{t}m1", "one", "{t}m2", L),
    ("INCR", "{t}ctr"), ("INCRBY", "{t}ctr", "41"),
    ("SET", "{t}gone", "x"), ("DEL", "{t}gone"),
    ("HSET", "{t}h", "f1", "v1", "f2", L, "f3", "v3"), ("HDEL", "{t}h", "f3"),
    ("HINCRBY", "{t}h", "n", "7"),
    ("RPUSH", "{t}l", "a", "b", "c", L), ("LPUSH", "{t}l", "z"), ("LPOP", "{t}l"),
    ("LSET", "{t}l", "0", "A"), ("LINSERT", "{t}l", "BEFORE", "c", "bb"),
    ("RPUSH", "{t}ldrain", "only"), ("RPOP", "{t}ldrain"),          # drain deletes the key
    ("SADD", "{t}s", "m1", "m2", "m3", L), ("SREM", "{t}s", "m3"),
    ("SADD", "{t}sp", "only"), ("SPOP", "{t}sp"),                     # resolved effect + delete
    ("ZADD", "{t}z", "1", "a", "2", "b", "3", L, "0", "zz"), ("ZINCRBY", "{t}z", "1.5", "a"),
    ("ZPOPMIN", "{t}z"),
    ("XADD", "{t}x", "1-1", "f", "v"), ("XADD", "{t}x", "*", "g", "w"), ("XDEL", "{t}x", "1-1"),
    ("PFADD", "{t}hll", "a", "b", "c"),
    ("SETBIT", "{t}bm", "7", "1"), ("SETBIT", "{t}bm", "100", "1"),
    ("SET", "{t}ttl", "v"), ("EXPIRE", "{t}ttl", "5000"),
    ("SET", "{t}pers", "v"), ("EXPIRE", "{t}pers", "5000"), ("PERSIST", "{t}pers"),
    ("VADD", "{t}vs", "VALUES", "3", "1", "0", "0", "e1"),
    ("VADD", "{t}vs", "VALUES", "3", "0", "1", "0", "e2"),
    ("VSETATTR", "{t}vs", "e1", '{"k":1}'), ("VREM", "{t}vs", "e2"),
    ("SUNIONSTORE", "{t}su", "{t}s", "{t}sp"),
    ("RENAME", "{t}m1", "{t}m1r"),
    # gh #392: field TTLs ride the stream as cmd 32/33 records
    ("HSET", "{t}hf", "a", "1", "b", "2", "c", "3"), ("HEXPIRE", "{t}hf", "5000", "FIELDS", "2", "a", "b"),
    ("HPERSIST", "{t}hf", "FIELDS", "1", "b"),
]

KEYS = ["{t}str", "{t}m1", "{t}m1r", "{t}m2", "{t}ctr", "{t}gone", "{t}h", "{t}l", "{t}ldrain",
        "{t}s", "{t}sp", "{t}z", "{t}x", "{t}hll", "{t}bm", "{t}ttl", "{t}pers", "{t}vs", "{t}su",
        "{t}hf"]


def dump(c: Conn, key: str):
    t = c.cmd("TYPE", key)
    has_ttl = c.cmd("TTL", key)
    has_ttl = has_ttl if has_ttl in (-1, -2) else "ttl>0"
    body = {
        "string": lambda: c.cmd("GET", key),
        "hash": lambda: sorted((f, v, "ttl" if c.cmd("HTTL", key, "FIELDS", "1", f)[0] > 0 else -1)
                               for f, v in zip(*[iter(c.cmd("HGETALL", key))] * 2)),
        "list": lambda: c.cmd("LRANGE", key, "0", "-1"),
        "set": lambda: sorted(c.cmd("SMEMBERS", key)),
        "zset": lambda: c.cmd("ZRANGE", key, "0", "-1", "WITHSCORES"),
        "stream": lambda: [(e[1]) for e in c.cmd("XRANGE", key, "-", "+")],   # ids may differ for `*`
        "vectorset": lambda: (c.cmd("VCARD", key), c.cmd("VGETATTR", key, "e1"),
                              c.cmd("VEMB", key, "e1")),
        "none": lambda: None,
    }.get(t if isinstance(t, str) else "?", lambda: ("unknown type", t))()
    if t == "string" and key == "{t}hll":
        body = c.cmd("PFCOUNT", key)
    return (t, has_ttl, body)


# Reads Redis answers on a replica. Each must be SERVED, matching the primary.
READS = [
    ("GET", "{t}str"), ("STRLEN", "{t}str"), ("GETRANGE", "{t}str", "0", "3"), ("MGET", "{t}m2", "{t}str"),
    ("HGET", "{t}h", "f1"), ("HGETALL", "{t}h"), ("HKEYS", "{t}h"), ("HLEN", "{t}h"), ("HEXISTS", "{t}h", "f1"),
    ("LRANGE", "{t}l", "0", "-1"), ("LLEN", "{t}l"), ("LINDEX", "{t}l", "1"), ("LPOS", "{t}l", "c"),
    ("SMEMBERS", "{t}s"), ("SCARD", "{t}s"), ("SISMEMBER", "{t}s", "m1"), ("SINTER", "{t}s", "{t}su"),
    ("ZRANGE", "{t}z", "0", "-1"), ("ZSCORE", "{t}z", "a"), ("ZRANK", "{t}z", "a"), ("ZCARD", "{t}z"),
    ("ZRANGEBYSCORE", "{t}z", "-inf", "+inf"),
    ("XLEN", "{t}x"), ("PFCOUNT", "{t}hll"), ("GETBIT", "{t}bm", "7"), ("BITCOUNT", "{t}bm"),
    ("EXISTS", "{t}str", "{t}h"), ("TYPE", "{t}z"), ("DBSIZE",),
    ("VCARD", "{t}vs"), ("VDIM", "{t}vs"), ("VISMEMBER", "{t}vs", "e1"),
]

# Writes: each must be REFUSED on the replica and leave the dump unchanged.
WRITES = [
    ("SET", "{t}str", "overwritten"), ("DEL", "{t}h"), ("HSET", "{t}h", "f1", "X"),
    ("RPUSH", "{t}l", "X"), ("SADD", "{t}s", "X"), ("ZADD", "{t}z", "9", "X"),
    ("XADD", "{t}x", "*", "X", "Y"), ("EXPIRE", "{t}str", "1"), ("INCR", "{t}ctr"),
    ("VADD", "{t}vs", "VALUES", "3", "1", "1", "1", "X"), ("FLUSHALL",),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2201)
    args = ap.parse_args()
    pa, pb = args.port, args.port + 20
    work = tempfile.mkdtemp(prefix="pion-repl-types-")
    procs, fails = [], []

    def start(port, extra, sub):
        d = os.path.join(work, sub); os.makedirs(d)
        log = open(os.path.join(d, "server.log"), "w")
        p = subprocess.Popen([os.path.abspath(BIN), "-p", str(port), "-w", "1", "--no-crash-log",
                              "--no-auto-detect", "--no-auto-embed", "--cluster",
                              "--cluster-host", "127.0.0.1"] + extra,
                             cwd=d, stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        wait_ready(port, 30, proc=p)
        return p

    try:
        start(pa, [], "primary")
        start(pb, ["--cluster-replica", "--cluster-primary-host", "127.0.0.1",
                   "--cluster-primary-port", str(pa)], "replica")
        a = Conn(pa)
        b = Conn(pb)
        if b.cmd("READONLY") != "OK":
            fails.append("replica refused READONLY")
        # The stream must be LIVE before the workload, so what follows tests
        # record decoding, not initial sync (which Pion does not do: a write
        # made before the replica connects never reaches it — see
        # test_replication_initial_sync.py).
        deadline = time.time() + 20
        while True:
            a.cmd("SET", "{t}sentinel", "up")
            if b.cmd("GET", "{t}sentinel") == b"up":
                break
            if time.time() > deadline:
                print("FAIL replication stream never delivered a sentinel SET within 20 s")
                return 1
            time.sleep(0.2)
        for w in WORKLOAD:
            r = a.cmd(*w)
            if isinstance(r, RespError):
                fails.append(f"primary refused workload step {w}: {r}")
        a.assert_in_sync()

        want = {k: dump(a, k) for k in KEYS}
        deadline = time.time() + 15
        while True:
            got = {k: dump(b, k) for k in KEYS}
            if got == want or time.time() > deadline:
                break
            time.sleep(0.3)
        for k in KEYS:
            if got[k] != want[k]:
                fails.append(f"replica differs on {k}:\n      primary {str(want[k])[:150]}\n"
                             f"      replica {str(got[k])[:150]}")

        # The ALLOW direction of the replica's write guard.
        for rd in READS:
            ra, rb = a.cmd(*rd), b.cmd(*rd)
            if isinstance(rb, RespError) or (rd[0] not in ("SINTER", "HKEYS", "SMEMBERS", "HGETALL")
                                             and ra != rb):
                fails.append(f"replica read {rd}: replica {str(rb)[:100]!r} vs primary {str(ra)[:100]!r}")

        # The REFUSE direction — and a refusal must be a no-op.
        before = {k: dump(b, k) for k in KEYS}
        for w in WRITES:
            r = b.cmd(*w)
            if not isinstance(r, RespError):
                fails.append(f"replica ACCEPTED write {w}: {r!r}")
        after = {k: dump(b, k) for k in KEYS}
        for k in KEYS:
            if before[k] != after[k]:
                fails.append(f"a refused write changed {k} on the replica: {before[k]} -> {after[k]}")
        b.assert_in_sync()
    finally:
        for p in procs:
            p.kill(); p.wait()
        shutil.rmtree(work, ignore_errors=True)

    print(f"{len(WORKLOAD)} writes, {len(KEYS)} keys compared, {len(READS)} replica reads, "
          f"{len(WRITES)} replica writes: {len(fails)} failure(s)")
    for f in fails:
        print("  FAIL " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
