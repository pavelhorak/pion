#!/usr/bin/env python3
"""Differential parity: run identical commands against Pion and real Redis, diff.

The type-confusion sweep (`test_type_confusion.py`) can prove a command does not
CRASH on the wrong type, but it cannot say what the reply SHOULD be — so it
files anything non-error as "accepted, review". That left ~168 cases resting on
someone's memory of Redis semantics.

This harness removes the judgement call. Every probe runs against both servers
and only DISAGREEMENTS are reported. Real Redis is the oracle, so "Pion returns
'' where Redis returns WRONGTYPE" becomes a fact rather than an opinion.

Deliberately narrow: it compares reply SHAPE and VALUE, not timing, and it skips
commands whose output is legitimately server-specific (INFO, CONFIG, versions,
anything with an address, pid, or clock in it).

Usage:
  python3 tests/test_redis_differential.py --pion-port 1974 --redis-port 6399
  python3 tests/test_redis_differential.py --start-redis        # spawn our own
"""

import argparse
import shutil
import socket
import subprocess
import sys
import time

HOST = "127.0.0.1"


RESP3 = False   # set from --resp3: every connection, reconnects too, says HELLO 3


class Conn:
    def __init__(self, port, timeout=6):
        self.s = socket.create_connection((HOST, port), timeout=timeout)
        self.s.settimeout(timeout)
        self.f = self.s.makefile("rb")
        if RESP3:
            kind, _ = self.cmd("HELLO", "3")
            if kind != "map":
                raise OSError(f"HELLO 3 on port {port} did not answer a map ({kind})")

    def cmd(self, *args):
        buf = f"*{len(args)}\r\n".encode()
        for a in args:
            a = a.encode() if isinstance(a, str) else a
            buf += b"$%d\r\n%s\r\n" % (len(a), a)
        self.s.sendall(buf)
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise EOFError("closed")
        t, body = line[:1], line[1:-2]
        if t == b"+":
            return ("status", body.decode())
        if t == b"-":
            # Compare only the error CODE (first word). Message wording differs
            # between implementations and is not a compatibility contract.
            return ("error", body.decode().split(" ", 1)[0])
        if t == b":":
            return ("int", int(body))
        if t == b"$":
            n = int(body)
            return ("nil", None) if n == -1 else ("bulk", self.f.read(n + 2)[:-2])
        if t == b"*":
            n = int(body)
            if n == -1:
                return ("nil", None)
            return ("array", [self._read() for _ in range(n)])
        if t == b"_":
            return ("nil", None)
        # RESP3 (`--resp3`). Each type keeps its own kind, so a reply sent as
        # an array where Redis sends a map, a set or a double is a divergence:
        # the TYPE is what a RESP3 client decodes it into (#23).
        if t == b"%":
            n = int(body)
            items = [self._read() for _ in range(2 * n)]
            return ("map", list(zip(items[0::2], items[1::2])))
        if t == b"~":
            return ("set", [self._read() for _ in range(int(body))])
        if t == b">":
            return ("push", [self._read() for _ in range(int(body))])
        if t == b"=":
            n = int(body)
            return ("verbatim", self.f.read(n + 2)[:-2])
        if t == b",":
            return ("double", body.decode())
        if t == b"#":
            return ("bool", body.decode())
        if t == b"(":
            return ("bignum", body.decode())
        return ("raw", body.decode(errors="replace"))

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


FIXTURES = [
    (["SET", "%P:string", "hello"], "string"),
    (["RPUSH", "%P:list", "a", "b", "c"], "list"),
    (["HSET", "%P:hash", "f", "v"], "hash"),
    (["SADD", "%P:set", "m1", "m2"], "set"),
    (["ZADD", "%P:zset", "1", "m1"], "zset"),
    (["PFADD", "%P:hll", "e1", "e2"], "hll"),
    (["SETBIT", "%P:bitmap", "7", "1"], "bitmap"),
    (["XADD", "%P:stream", "*", "f", "v"], "stream"),
]

# Commands probed against every fixture type (and a missing key).
PROBES = [
    ["GET", "%K"], ["STRLEN", "%K"], ["APPEND", "%K", "x"],
    ["GETRANGE", "%K", "0", "1"], ["SETRANGE", "%K", "0", "x"],
    ["INCR", "%K"], ["INCRBY", "%K", "1"], ["DECRBY", "%K", "1"],
    ["INCRBYFLOAT", "%K", "1.5"], ["GETDEL", "%K"], ["GETSET", "%K", "z"],
    ["LPUSH", "%K", "x"], ["RPUSH", "%K", "x"], ["LPOP", "%K"], ["RPOP", "%K"],
    ["LRANGE", "%K", "0", "-1"], ["LLEN", "%K"], ["LSET", "%K", "0", "x"],
    ["LINSERT", "%K", "BEFORE", "a", "x"], ["LTRIM", "%K", "0", "0"],
    ["HGET", "%K", "f"], ["HSET", "%K", "f", "v"], ["HDEL", "%K", "f"],
    ["HLEN", "%K"], ["HGETALL", "%K"], ["HKEYS", "%K"], ["HVALS", "%K"],
    ["HEXISTS", "%K", "f"], ["HINCRBY", "%K", "f", "1"],
    ["SADD", "%K", "m"], ["SREM", "%K", "m"], ["SCARD", "%K"],
    ["SMEMBERS", "%K"], ["SISMEMBER", "%K", "m"], ["SPOP", "%K"],
    ["ZADD", "%K", "1", "m"], ["ZREM", "%K", "m"], ["ZCARD", "%K"],
    ["ZSCORE", "%K", "m"], ["ZRANGE", "%K", "0", "-1"], ["ZINCRBY", "%K", "1", "m"],
    ["ZPOPMIN", "%K"], ["ZCOUNT", "%K", "-inf", "+inf"],
    ["PFADD", "%K", "e"], ["PFCOUNT", "%K"],
    ["GETBIT", "%K", "0"], ["SETBIT", "%K", "0", "1"], ["BITCOUNT", "%K"],
    ["XADD", "%K", "*", "f", "v"], ["XLEN", "%K"], ["XRANGE", "%K", "-", "+"],
    ["TYPE", "%K"], ["TTL", "%K"], ["PTTL", "%K"], ["PERSIST", "%K"],
    ["EXPIRE", "%K", "100"], ["EXISTS", "%K"],

    # --- 2026-08-25 extension -------------------------------------------------
    # The list above covered 57 commands, and the ten missing-vs-wrong-type
    # conflations closed in gh #232 were all found by it. ~100 keyspace commands
    # were never sent at all, which by the usual rule means their behaviour on a
    # wrong-type key was simply unknown — not verified. Same mechanism, so the
    # same class of bug is what this is looking for.
    #
    # Deliberately NOT added, because each would bake a permanent non-zero into
    # a count whose value is that it reads zero, without indicating a defect:
    #   OBJECT ENCODING  Pion does not use Redis's encoding names (listpack/…)
    #   DUMP / RESTORE   the serialization format is Redis-private
    #   XINFO STREAM     Redis returns many more fields than Pion implements
    ["SETNX", "%K", "z"], ["SETEX", "%K", "100", "z"], ["PSETEX", "%K", "100000", "z"],
    ["GETEX", "%K"], ["SUBSTR", "%K", "0", "1"], ["TOUCH", "%K"],
    ["BITPOS", "%K", "1"], ["BITFIELD", "%K", "GET", "u8", "0"],

    ["LINDEX", "%K", "0"], ["LPOS", "%K", "a"], ["LPUSHX", "%K", "x"],
    ["RPUSHX", "%K", "x"], ["LREM", "%K", "0", "a"],
    ["RPOPLPUSH", "%K", "df:dest"], ["LMOVE", "%K", "df:dest", "LEFT", "RIGHT"],
    ["LMPOP", "1", "%K", "LEFT"],

    ["HMGET", "%K", "f"], ["HSETNX", "%K", "f", "v"], ["HSTRLEN", "%K", "f"],
    ["HRANDFIELD", "%K"], ["HINCRBYFLOAT", "%K", "f", "1.5"],
    ["HTTL", "%K", "FIELDS", "1", "f"], ["HPERSIST", "%K", "FIELDS", "1", "f"],

    ["SMISMEMBER", "%K", "m"], ["SMOVE", "%K", "df:dest", "m"],
    ["SRANDMEMBER", "%K"], ["SDIFF", "%K"], ["SINTER", "%K"], ["SUNION", "%K"],
    ["SDIFFSTORE", "df:dest", "%K"],

    ["ZPOPMAX", "%K"], ["ZRANK", "%K", "m"], ["ZREVRANK", "%K", "m"],
    ["ZMSCORE", "%K", "m"], ["ZRANDMEMBER", "%K"], ["ZLEXCOUNT", "%K", "-", "+"],
    ["ZRANGEBYSCORE", "%K", "-inf", "+inf"], ["ZRANGEBYLEX", "%K", "-", "+"],
    ["ZREVRANGE", "%K", "0", "-1"], ["ZREMRANGEBYRANK", "%K", "0", "0"],
    ["ZREMRANGEBYSCORE", "%K", "-inf", "+inf"], ["ZREMRANGEBYLEX", "%K", "-", "+"],
    ["ZDIFF", "1", "%K"], ["ZINTER", "1", "%K"], ["ZUNION", "1", "%K"],
    ["ZINTERCARD", "1", "%K"], ["ZMPOP", "1", "%K", "MIN"],
    ["ZRANGESTORE", "df:dest", "%K", "0", "-1"],

    ["XDEL", "%K", "1-1"], ["XTRIM", "%K", "MAXLEN", "1"], ["XREVRANGE", "%K", "+", "-"],

    # 4102444800 = 2100-01-01. NOT a larger epoch: Pion stores deadlines in
    # NANOSECONDS, which end in ~2262, so a later deadline is stored saturated
    # (gh #393) and EXPIRETIME would report 2262, not the year asked for.
    ["EXPIREAT", "%K", "4102444800"], ["PEXPIRE", "%K", "100000"],
    ["EXPIRETIME", "%K"], ["PEXPIRETIME", "%K"],
    ["RENAME", "%K", "df:dest"], ["RENAMENX", "%K", "df:dest"],
    ["COPY", "%K", "df:dest"],

    ["GEOPOS", "%K", "m"], ["GEODIST", "%K", "a", "b"], ["GEOHASH", "%K", "m"],
    ["PFMERGE", "df:dest", "%K"],
]


def normalize(kind_val, cmd):
    """Collapse differences that are not compatibility contracts."""
    kind, val = kind_val
    if cmd[0] in ("PTTL", "HPTTL"):
        # A remaining-TTL in milliseconds is read at a slightly different
        # instant on each server (HPTTL: Pion 100000 vs Redis 99999 — one
        # "divergence" in a 757-step run). Round to 100 ms; the sentinels
        # -1 (no TTL) and -2 (missing) stay exact, and a wrong deadline is
        # still off by far more than 100 ms.
        def ms(v):
            return v if v < 0 else round(v, -2)
        if kind == "int":
            return (kind, ms(val))
        if kind == "array":
            return (kind, [(k, ms(x)) if k == "int" else (k, x) for k, x in val])
    if kind == "array" and cmd[0] in ("SCAN", "HSCAN", "SSCAN", "ZSCAN"):
        # Must precede the generic array branch below, which returns
        # unconditionally. A SCAN cursor is implementation-private — Redis and
        # Pion number their buckets differently and comparing cursors would
        # report a divergence on every probe. What IS a contract: whether the
        # iteration COMPLETED (cursor "0") and which elements it yielded.
        # Order within the batch is unspecified, so the elements are sorted.
        cur, items = (list(val) + [("array", [])])[:2]
        elems = [repr(x) for x in (items[1] if items[0] == "array" else [])]
        return ("scan", cur[1] in (b"0", "0"), sorted(elems))
    if kind == "bulk" and cmd[0] == "GEOPOS":
        # GEOPOS coordinates come back as printed doubles and the two servers
        # format the last digit differently (38.1155563954963 vs
        # ...49629) — same value, C's %.17g against Mojo's shortest repr.
        # Compared at 10 decimals: far tighter than geohash resolution (~0.6 m
        # of error at 26 bits), so a genuinely wrong coordinate still fails.
        try:
            return ("geo-coord", round(float(val), 10))
        except (TypeError, ValueError):
            return (kind, val)
    if kind == "array" and cmd[0] in ("XRANGE", "XREVRANGE"):
        # Also before the generic array branch. The wrong-type fixture is built
        # with `XADD * `, so every entry carries a wall-clock id and the two
        # servers differ by a millisecond — the ids leak into the reply BODY,
        # not just XADD's own reply, which is why normalizing XADD alone left
        # the count flapping between 27 and 28. Entry COUNT and the field/value
        # payload are still compared; only the id is reduced to its shape.
        out = []
        for e in val:
            if e[0] == "array" and len(e[1]) == 2:
                idv, fields = e[1]
                parts = idv[1].split(b"-") if isinstance(idv[1], bytes) else []
                shape = len(parts) == 2 and all(p.isdigit() for p in parts)
                out.append(("entry", shape, normalize(fields, cmd)))
            else:
                out.append(normalize(e, cmd))
        return ("xrange", out)
    if kind in ("map", "set"):
        # RESP3 map / set: order is unspecified, the type is not.
        return (kind, sorted(repr(normalize(x, cmd)) for x in val))
    if kind == "array":
        # Order is unspecified for set-like replies. The RANDMEMBER family is
        # here for the same reason and is only ever probed in shapes whose
        # CONTENTS are determined (single-element container, count 0, count
        # larger than the container) — order is the only free variable, so
        # sorting makes them comparable without weakening the check.
        if cmd[0] in ("SMEMBERS", "HKEYS", "HVALS", "HGETALL", "SPOP",
                      "KEYS", "SINTER", "SUNION", "SDIFF",
                      "HRANDFIELD", "SRANDMEMBER", "ZRANDMEMBER"):
            return ("array-sorted", sorted(repr(x) for x in val))
        return (kind, [normalize(x, cmd) for x in val])
    if kind == "bulk" and cmd[0] in ("SPOP", "SRANDMEMBER", "ZRANDMEMBER", "HRANDFIELD"):
        # WHICH member comes back is unspecified. The array forms are sorted
        # above; these are the no-count forms, whose reply is a single bulk
        # string, so there is nothing to sort and the raw value differs at
        # random. Left unnormalized, SRANDMEMBER alone made the wrong-type
        # count flap between 27 and 28 across identical runs.
        return ("bulk-any", None)
    if kind == "bulk" and cmd[0] == "XADD" and "*" in cmd:
        # An auto-generated stream id is a wall-clock read, so the two servers
        # answer with ids a millisecond apart at random. Left raw this probe
        # differed on roughly one run in three, which makes the wrong-type
        # baseline flap and turns a deterministic number into a coin toss.
        # The SHAPE is still checked: <ms>-<seq>, both integers.
        parts = val.split(b"-")
        ok = len(parts) == 2 and all(p.isdigit() for p in parts)
        return ("stream-id", ok)
    return (kind, val)


class _Desync(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pion-port", type=int, default=1974)
    ap.add_argument("--redis-port", type=int, default=6399)
    ap.add_argument("--start-redis", action="store_true")
    ap.add_argument("--show-agreements", action="store_true")
    ap.add_argument("--resp3", action="store_true",
                    help="speak RESP3 (HELLO 3) to both servers and compare reply TYPES too")
    args = ap.parse_args()
    global RESP3
    RESP3 = args.resp3

    rproc = None
    if args.start_redis:
        # #27: was /opt/homebrew/bin/redis-server — a Homebrew path, so the
        # oracle never started on Linux.
        exe = shutil.which("redis-server")
        if exe is None:
            print("FATAL: --start-redis needs redis-server on PATH")
            return 2
        rproc = subprocess.Popen(
            [exe, "--port", str(args.redis_port), "--save", "", "--appendonly", "no"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)

    try:
        pion = Conn(args.pion_port)
        redis = Conn(args.redis_port)
    except OSError as e:
        print(f"FATAL: need both servers up ({e})")
        if rproc:
            rproc.kill()
        return 2

    # Start both sides from an empty keyspace.
    #
    # The per-probe DEL below is not sufficient, and the reason is worth
    # stating: `--start-redis` spawns a redis on a fixed port and kills it at
    # exit, so back-to-back invocations race — run N+1 can connect to run N's
    # instance if teardown lags, inheriting its accumulated fixtures. That
    # showed up as the differ count moving 119 -> 120 with no code change,
    # which is indistinguishable from the regression signal this test exists
    # to provide. Pion accumulates the same way when it is long-lived.
    #
    # FLUSHALL on both makes the count deterministic regardless of either
    # server's history, which is a property of the harness rather than
    # something a human has to remember before each run.
    for _side, _c in (("Pion", pion), ("Redis", redis)):
        try:
            _c.cmd("FLUSHALL")
        except Exception:                                   # noqa: BLE001
            print(f"WARNING: FLUSHALL failed on {_side}; the differ count may "
                  f"drift by one or two from accumulated fixtures")

    # Each side uses its own key prefix so a stale key cannot cross-contaminate.
    diffs, same, n = [], 0, 0
    desyncs = []
    targets = [(f, kind) for f, kind in FIXTURES] + [(None, "missing")]

    for fixture, kind in targets:
        for probe in PROBES:
            n += 1
            key_p = f"df:{kind}"
            for conn in (pion, redis):
                conn.cmd("DEL", key_p)
                # The store/move probes share one destination key, and it
                # accumulates. Once the two servers disagree about whether a
                # store is allowed, `df:dest` ends up holding DIFFERENT TYPES
                # on each — after which the next probe against it reports a
                # divergence that is pure leftover state. Clear it per probe so
                # a destination-typed reply means what it says.
                conn.cmd("DEL", "df:dest")
                if fixture:
                    conn.cmd(*[p.replace("%P", "df") if isinstance(p, str) else p
                               for p in fixture])
            cmd = [key_p if p == "%K" else p for p in probe]
            try:
                rp = normalize(pion.cmd(*cmd), cmd)
                rr = normalize(redis.cmd(*cmd), cmd)
                # A wrong reply COUNT desyncs the stream and silently corrupts
                # every later comparison, so verify framing after each probe.
                for name, conn in (("Pion", pion), ("Redis", redis)):
                    if conn.cmd("PING") != ("status", "PONG"):
                        desyncs.append((kind, cmd, name))
                        conn.close()
                        newc = Conn(args.pion_port if name == "Pion" else args.redis_port)
                        if name == "Pion":
                            pion = newc
                        else:
                            redis = newc
                        raise _Desync()
            except _Desync:
                continue
            except (EOFError, socket.timeout) as e:
                diffs.append((kind, cmd, f"TRANSPORT {type(e).__name__}", "-"))
                pion = Conn(args.pion_port); redis = Conn(args.redis_port)
                continue
            if rp == rr:
                same += 1
                if args.show_agreements:
                    print(f"  ok   {kind:8} {' '.join(cmd[:3]):34} {rp}")
            else:
                diffs.append((kind, cmd, rp, rr))

    print(f"\n{n} probes: {same} agree, {len(diffs)} differ, "
          f"{len(desyncs)} DESYNCED\n")
    if desyncs:
        print("REPLY-COUNT DESYNCS (wrong number of replies — corrupts the "
              "connection for every later command):")
        for kind, cmd, who in desyncs:
            print(f"  {who:6} {kind:9} {' '.join(str(c) for c in cmd)}")
        print()
    if diffs:
        print(f"{'type':9} {'command':34} {'Pion':28} Redis 8.10")
        print("-" * 104)
        for kind, cmd, rp, rr in diffs:
            print(f"{kind:9} {' '.join(str(c) for c in cmd[:3]):34} "
                  f"{str(rp)[:27]:28} {str(rr)[:34]}")

    sd, ssame, sn = run_semantics(pion, redis)
    print(f"\nCORRECT-TYPE semantics: {sn} steps, {ssame} agree, {len(sd)} differ\n")
    if sd:
        for name, cmd, rp, rr in sd:
            print(f"  {name[:34]:36} {' '.join(str(x) for x in cmd)[:36]:38} "
                  f"Pion {str(rp)[:24]:26} Redis {str(rr)[:28]}")
    diffs.extend(sd)

    pion.close()
    redis.close()
    if rproc:
        rproc.kill()
    # The documented fence (gh #232): Pion stores HLL as dense
    # registers where Redis stores a sparse string, so string/bitmap reads of
    # an HLL key are WRONGTYPE on Pion and succeed on Redis. Exactly that shape
    # is fenced — an HLL probe where Pion answers anything OTHER than
    # WRONGTYPE, and every non-HLL divergence, still fails. The run used to
    # exit 1 on the fence itself, so it could never gate.
    fenced = [d for d in diffs if d[0] == "hll" and d[2] == ("error", "WRONGTYPE")]
    real = [d for d in diffs if d not in fenced]
    print(f"fenced (documented HLL divergence): {len(fenced)} · unfenced: {len(real)} · desyncs: {len(desyncs)}")
    return 1 if (real or desyncs) else 0




# ── Correct-type semantics ────────────────────────────────────────────────
# The probes above use WRONG types. But the HSET return-value bug (gh #232 §3)
# was a CORRECT-type bug: right type, right operation, wrong answer. Those are
# the ones that break working code rather than error paths, so they deserve a
# matrix of their own. Each entry is a script: a list of commands run in order
# on both servers, with every reply compared.
SEMANTIC_SCRIPTS = [
    ("string: SET/APPEND/STRLEN/GETRANGE", [
        ["DEL", "%K"], ["SET", "%K", "hello"], ["APPEND", "%K", " world"],
        ["STRLEN", "%K"], ["GETRANGE", "%K", "0", "4"],
        ["GETRANGE", "%K", "-5", "-1"], ["GETRANGE", "%K", "0", "-1"],
        ["GETRANGE", "%K", "100", "200"], ["GETRANGE", "%K", "-100", "-99"],
        ["SETRANGE", "%K", "6", "WORLD"], ["GET", "%K"],
        ["SETRANGE", "%K", "20", "x"], ["STRLEN", "%K"], ["GET", "%K"]]),
    ("string: SETNX / MSETNX / GETDEL / GETEX", [
        ["DEL", "%K"], ["SETNX", "%K", "a"], ["SETNX", "%K", "b"], ["GET", "%K"],
        ["GETDEL", "%K"], ["EXISTS", "%K"], ["GETDEL", "%K"]]),
    ("incr family", [
        ["DEL", "%K"], ["INCR", "%K"], ["INCRBY", "%K", "5"],
        ["DECR", "%K"], ["DECRBY", "%K", "3"], ["GET", "%K"],
        ["INCRBYFLOAT", "%K", "1.5"], ["GET", "%K"],
        ["INCRBYFLOAT", "%K", "-0.5"], ["GET", "%K"]]),
    ("list: push/pop/range/index/set", [
        ["DEL", "%K"], ["RPUSH", "%K", "a", "b", "c"], ["LPUSH", "%K", "z"],
        ["LRANGE", "%K", "0", "-1"], ["LLEN", "%K"], ["LINDEX", "%K", "0"],
        ["LINDEX", "%K", "-1"], ["LINDEX", "%K", "99"],
        ["LSET", "%K", "1", "B"], ["LRANGE", "%K", "0", "-1"],
        ["LPOP", "%K"], ["RPOP", "%K"], ["LRANGE", "%K", "0", "-1"],
        ["LRANGE", "%K", "5", "10"], ["LRANGE", "%K", "-100", "100"]]),
    ("list: LINSERT / LREM / LTRIM", [
        ["DEL", "%K"], ["RPUSH", "%K", "a", "b", "a", "c", "a"],
        ["LINSERT", "%K", "BEFORE", "b", "X"], ["LRANGE", "%K", "0", "-1"],
        ["LINSERT", "%K", "AFTER", "nope", "Y"],
        ["LREM", "%K", "2", "a"], ["LRANGE", "%K", "0", "-1"],
        ["LTRIM", "%K", "0", "1"], ["LRANGE", "%K", "0", "-1"], ["LLEN", "%K"]]),
    ("hash: HSET/HDEL/HLEN/HEXISTS/HSETNX", [
        ["DEL", "%K"], ["HSET", "%K", "a", "1"], ["HSET", "%K", "a", "2"],
        ["HSET", "%K", "b", "1", "c", "2"], ["HLEN", "%K"],
        ["HEXISTS", "%K", "a"], ["HEXISTS", "%K", "zz"],
        ["HSETNX", "%K", "a", "9"], ["HSETNX", "%K", "d", "9"],
        ["HGET", "%K", "a"], ["HDEL", "%K", "a"], ["HDEL", "%K", "a"],
        ["HDEL", "%K", "b", "c"], ["HLEN", "%K"],
        ["HINCRBY", "%K", "n", "5"], ["HINCRBY", "%K", "n", "-2"]]),
    ("set: SADD/SREM/SCARD/SISMEMBER", [
        ["DEL", "%K"], ["SADD", "%K", "a", "b", "c"], ["SADD", "%K", "a"],
        ["SADD", "%K", "a", "d"], ["SCARD", "%K"],
        ["SISMEMBER", "%K", "a"], ["SISMEMBER", "%K", "zz"],
        ["SREM", "%K", "a"], ["SREM", "%K", "a"], ["SCARD", "%K"]]),
    ("zset: ZADD/ZSCORE/ZINCRBY/ZCARD/ZCOUNT", [
        ["DEL", "%K"], ["ZADD", "%K", "1", "a", "2", "b"], ["ZADD", "%K", "3", "a"],
        ["ZCARD", "%K"], ["ZSCORE", "%K", "a"], ["ZSCORE", "%K", "zz"],
        ["ZINCRBY", "%K", "1.5", "a"], ["ZSCORE", "%K", "a"],
        ["ZCOUNT", "%K", "-inf", "+inf"], ["ZCOUNT", "%K", "2", "3"],
        ["ZRANGE", "%K", "0", "-1"], ["ZREM", "%K", "a"], ["ZREM", "%K", "a"],
        ["ZCARD", "%K"]]),
    ("key mgmt: TYPE / RENAME / COPY / EXISTS", [
        ["DEL", "%K", "%K2"], ["SET", "%K", "v"], ["TYPE", "%K"],
        ["EXISTS", "%K", "%K", "%K2"], ["RENAME", "%K", "%K2"],
        ["EXISTS", "%K"], ["GET", "%K2"], ["RENAMENX", "%K2", "%K"],
        ["GET", "%K"]]),
    ("bitmap: SETBIT/GETBIT/BITCOUNT", [
        ["DEL", "%K"], ["SETBIT", "%K", "0", "1"], ["SETBIT", "%K", "0", "1"],
        ["GETBIT", "%K", "0"], ["GETBIT", "%K", "1"], ["GETBIT", "%K", "100"],
        ["SETBIT", "%K", "10", "1"], ["BITCOUNT", "%K"], ["STRLEN", "%K"]]),
    ("ttl: SET clears, EXPIRE/PERSIST", [
        ["DEL", "%K"], ["SET", "%K", "v"], ["TTL", "%K"],
        ["EXPIRE", "%K", "100"], ["PERSIST", "%K"], ["TTL", "%K"],
        ["EXPIRE", "%K", "100"], ["SET", "%K", "v2"], ["TTL", "%K"],
        ["EXPIRE", "%K", "0"], ["EXISTS", "%K"]]),

    ("string: GETRANGE / SETRANGE boundaries", [
        ["DEL", "%K"], ["SET", "%K", "hello world"],
        ["GETRANGE", "%K", "-100", "-99"], ["GETRANGE", "%K", "-100", "-1"],
        ["GETRANGE", "%K", "-100", "0"], ["GETRANGE", "%K", "5", "3"],
        ["GETRANGE", "%K", "0", "1000"], ["GETRANGE", "%K", "-1", "-1"],
        ["GETRANGE", "%K", "11", "20"], ["GETRANGE", "%K", "-1", "-5"]]),
    ("expire: NX/XX/GT/LT flags", [
        ["DEL", "%K"], ["SET", "%K", "v"],
        ["EXPIRE", "%K", "100", "NX"], ["TTL", "%K"],
        ["EXPIRE", "%K", "200", "NX"], ["TTL", "%K"],
        ["EXPIRE", "%K", "300", "XX"], ["TTL", "%K"],
        ["EXPIRE", "%K", "100", "GT"], ["TTL", "%K"],
        ["EXPIRE", "%K", "400", "GT"], ["TTL", "%K"],
        ["EXPIRE", "%K", "100", "LT"], ["TTL", "%K"]]),
    ("list: LPOS", [
        ["DEL", "%K"], ["RPUSH", "%K", "a", "b", "c", "a", "b", "a"],
        ["LPOS", "%K", "a"], ["LPOS", "%K", "a", "RANK", "2"],
        ["LPOS", "%K", "a", "RANK", "-1"], ["LPOS", "%K", "a", "COUNT", "2"],
        ["LPOS", "%K", "zz"]]),
    ("set: SMOVE / SINTERCARD / store ops", [
        ["DEL", "%K", "%K2"], ["SADD", "%K", "a", "b", "c"], ["SADD", "%K2", "b", "c", "d"],
        ["SMOVE", "%K", "%K2", "a"], ["SCARD", "%K"], ["SCARD", "%K2"],
        ["SMOVE", "%K", "%K2", "zz"], ["SISMEMBER", "%K2", "a"]]),
    ("zset: range by score and lex", [
        ["DEL", "%K"], ["ZADD", "%K", "1", "a", "2", "b", "3", "c"],
        ["ZRANGEBYSCORE", "%K", "-inf", "+inf"], ["ZRANGEBYSCORE", "%K", "2", "3"],
        ["ZRANGEBYSCORE", "%K", "(1", "3"], ["ZREVRANGE", "%K", "0", "-1"],
        ["ZRANK", "%K", "b"], ["ZRANK", "%K", "zz"], ["ZREVRANK", "%K", "b"],
        ["ZRANGE", "%K", "0", "-1", "WITHSCORES"], ["ZCOUNT", "%K", "(1", "+inf"]]),
    ("hash: HRANDFIELD / HSTRLEN / HMGET", [
        ["DEL", "%K"], ["HSET", "%K", "a", "1", "b", "22"],
        ["HSTRLEN", "%K", "a"], ["HSTRLEN", "%K", "b"], ["HSTRLEN", "%K", "zz"],
        ["HMGET", "%K", "a", "zz", "b"], ["HLEN", "%K"]]),
    ("key: COPY / RENAME edge cases", [
        ["DEL", "%K", "%K2"], ["SET", "%K", "v1"], ["SET", "%K2", "v2"],
        ["COPY", "%K", "%K2"], ["GET", "%K2"],
        ["COPY", "%K", "%K2", "REPLACE"], ["GET", "%K2"],
        ["RENAMENX", "%K", "%K2"], ["EXISTS", "%K"]]),
    ("string: SETEX / GETEX / SETRANGE past end", [
        ["DEL", "%K"], ["SETEX", "%K", "100", "v"], ["TTL", "%K"],
        ["GETEX", "%K", "PERSIST"], ["TTL", "%K"],
        ["DEL", "%K"], ["SETRANGE", "%K", "5", "abc"], ["STRLEN", "%K"], ["GET", "%K"]]),

    # ── Added 2026-08-23 ───────────────────────────────────────────────────
    # Three of the scripts above name a command in their title and never send
    # it — SINTERCARD, ZRANGEBYLEX and HRANDFIELD were all "covered" by a
    # title only. That is the same shape as the dispatch sweep's lesson: the
    # probes you actually send are the coverage, and a plausible name is not a
    # probe. These fill those gaps and add the option/flag surfaces, which is
    # where a correct-type wrong-answer bug (gh #232 §3, HSET) hides: the
    # command is right, the type is right, only the reply is wrong.

    # SET's options are its own little command set, and each one is a place to
    # answer plausibly-but-wrongly: NX/XX gate the write, GET changes the
    # reply TYPE, KEEPTTL is defined by what it does NOT do.
    ("string: SET NX/XX/GET/KEEPTTL/EXAT", [
        ["DEL", "%K"],
        ["SET", "%K", "a", "NX"], ["GET", "%K"],
        ["SET", "%K", "b", "NX"], ["GET", "%K"],
        ["SET", "%K", "b", "XX"], ["GET", "%K"],
        ["DEL", "%K"], ["SET", "%K", "c", "XX"], ["EXISTS", "%K"],
        ["SET", "%K", "d"], ["SET", "%K", "e", "GET"], ["GET", "%K"],
        ["DEL", "%K"], ["SET", "%K", "f", "GET"],
        ["SET", "%K", "g", "EX", "100"], ["TTL", "%K"],
        ["SET", "%K", "h", "KEEPTTL"], ["TTL", "%K"],
        ["SET", "%K", "i"], ["TTL", "%K"],
        ["SET", "%K", "j", "EXAT", "4102444800"], ["EXPIRETIME", "%K"]]),

    # ZADD's flags interact: GT/LT only move a score one way, CH changes what
    # the integer reply COUNTS, and INCR changes the reply type to a double.
    ("zset: ZADD NX/XX/GT/LT/CH/INCR", [
        ["DEL", "%K"],
        ["ZADD", "%K", "5", "m"], ["ZSCORE", "%K", "m"],
        ["ZADD", "%K", "NX", "9", "m"], ["ZSCORE", "%K", "m"],
        ["ZADD", "%K", "XX", "7", "m"], ["ZSCORE", "%K", "m"],
        ["ZADD", "%K", "XX", "3", "new"], ["ZSCORE", "%K", "new"],
        ["ZADD", "%K", "GT", "4", "m"], ["ZSCORE", "%K", "m"],
        ["ZADD", "%K", "GT", "10", "m"], ["ZSCORE", "%K", "m"],
        ["ZADD", "%K", "LT", "20", "m"], ["ZSCORE", "%K", "m"],
        ["ZADD", "%K", "LT", "2", "m"], ["ZSCORE", "%K", "m"],
        ["ZADD", "%K", "CH", "2", "m"], ["ZADD", "%K", "CH", "8", "m"],
        ["ZADD", "%K", "INCR", "1.5", "m"], ["ZSCORE", "%K", "m"],
        ["ZADD", "%K", "NX", "INCR", "1", "m"],
        ["ZCARD", "%K"]]),

    # Every other zset probe here updates INTEGER scores, and that is why a
    # truncation in the member dict survived: GenericValue.from_float stored
    # Int(score), so moving a member off a fractional score unlinked nothing
    # and left a duplicate (`ZADD 1.5 a; ZADD 2.5 a` -> ZCARD 2, ZSCORE 1.5),
    # and GT/LT compared against the truncated score.
    ("zset: fractional-score updates", [
        ["DEL", "%K"],
        ["ZADD", "%K", "1.5", "a"], ["ZADD", "%K", "2.5", "a"],
        ["ZCARD", "%K"], ["ZSCORE", "%K", "a"],
        ["ZADD", "%K", "GT", "2.25", "a"], ["ZSCORE", "%K", "a"],
        ["ZADD", "%K", "LT", "2.25", "a"], ["ZSCORE", "%K", "a"],
        ["ZADD", "%K", "CH", "2.25", "a"], ["ZADD", "%K", "CH", "2.125", "a"],
        ["ZADD", "%K", "0.5", "b", "0.5", "c"], ["ZADD", "%K", "0.75", "b"],
        ["ZRANGE", "%K", "0", "-1", "WITHSCORES"],
        ["ZINCRBY", "%K", "0.5", "b"], ["ZINCRBY", "%K", "0.25", "b"],
        ["ZRANK", "%K", "b"], ["ZCARD", "%K"],
        ["ZADD", "%K", "1790257349.703803", "t"], ["ZADD", "%K", "1790257349.71166", "t"],
        ["ZCARD", "%K"], ["ZSCORE", "%K", "t"],
        ["ZREM", "%K", "a"], ["ZRANGE", "%K", "0", "-1", "WITHSCORES"]]),

    # gh #243 shipped a real _glob_match where both KEYS and SCAN MATCH had
    # been ignoring the pattern entirely. Redis is the specification for glob
    # semantics, so diff against it rather than against a reading of the docs.
    # Literal key names, not %K: the point is matching across a keyspace.
    ("key: KEYS glob semantics (gh #243)", [
        ["DEL", "glob:aa", "glob:ab", "glob:b", "glob:cc", "glob:a-b", "other:x"],
        ["MSET", "glob:aa", "1", "glob:ab", "1", "glob:b", "1",
                 "glob:cc", "1", "glob:a-b", "1", "other:x", "1"],
        ["KEYS", "glob:*"], ["KEYS", "glob:a*"], ["KEYS", "glob:?"],
        ["KEYS", "glob:??"], ["KEYS", "glob:[ac]*"], ["KEYS", "glob:[a-c]*"],
        ["KEYS", "glob:[^a]*"], ["KEYS", "glob:a[ab]"], ["KEYS", "nomatch*"],
        ["KEYS", "glob:aa"], ["KEYS", "*:x"],
        ["DEL", "glob:aa", "glob:ab", "glob:b", "glob:cc", "glob:a-b", "other:x"]]),

    # Lexicographic ranges: promised by a title above, never sent. The bracket
    # syntax ([, (, -, +) is easy to get subtly wrong in a way that returns a
    # plausible subset rather than an error.
    ("zset: lex ranges", [
        ["DEL", "%K"],
        ["ZADD", "%K", "0", "a", "0", "b", "0", "c", "0", "d"],
        ["ZRANGEBYLEX", "%K", "-", "+"], ["ZRANGEBYLEX", "%K", "[b", "[c"],
        ["ZRANGEBYLEX", "%K", "(b", "+"], ["ZRANGEBYLEX", "%K", "-", "(c"],
        ["ZREVRANGEBYLEX", "%K", "+", "-"], ["ZREVRANGEBYLEX", "%K", "[c", "[b"],
        ["ZLEXCOUNT", "%K", "-", "+"], ["ZLEXCOUNT", "%K", "[b", "[c"],
        ["ZRANGEBYLEX", "%K", "[z", "+"],
        ["ZREMRANGEBYLEX", "%K", "[a", "[a"], ["ZRANGE", "%K", "0", "-1"]]),

    # The 6.2 unified ZRANGE: same command, four different meanings depending
    # on the modifier, plus REV flipping how LIMIT is applied.
    ("zset: unified ZRANGE REV/BYSCORE/BYLEX/LIMIT", [
        ["DEL", "%K"],
        ["ZADD", "%K", "1", "a", "2", "b", "3", "c", "4", "d"],
        ["ZRANGE", "%K", "0", "-1", "REV"],
        ["ZRANGE", "%K", "1", "3", "BYSCORE"],
        ["ZRANGE", "%K", "(1", "+inf", "BYSCORE"],
        ["ZRANGE", "%K", "+inf", "-inf", "BYSCORE", "REV"],
        ["ZRANGE", "%K", "-", "+", "BYLEX"],
        ["ZRANGE", "%K", "1", "4", "BYSCORE", "LIMIT", "1", "2"],
        ["ZRANGE", "%K", "1", "4", "BYSCORE", "LIMIT", "0", "-1"],
        ["ZRANGE", "%K", "0", "1"]]),

    # Fills the "SINTERCARD / store ops" the set script promised.
    ("set: SINTERCARD / SDIFF / store ops", [
        ["DEL", "%K", "%K2", "%K3"],
        ["SADD", "%K", "a", "b", "c"], ["SADD", "%K2", "b", "c", "d"],
        ["SINTER", "%K", "%K2"], ["SUNION", "%K", "%K2"], ["SDIFF", "%K", "%K2"],
        ["SINTERCARD", "2", "%K", "%K2"],
        ["SINTERCARD", "2", "%K", "%K2", "LIMIT", "1"],
        ["SINTERCARD", "2", "%K", "%K2", "LIMIT", "0"],
        ["SINTERSTORE", "%K3", "%K", "%K2"], ["SCARD", "%K3"], ["SMEMBERS", "%K3"],
        ["SUNIONSTORE", "%K3", "%K", "%K2"], ["SCARD", "%K3"],
        ["SDIFFSTORE", "%K3", "%K", "%K2"], ["SMEMBERS", "%K3"],
        ["SMISMEMBER", "%K", "a", "zz", "c"],
        ["SDIFFSTORE", "%K3", "%K", "%K"], ["EXISTS", "%K3"]]),

    # RANDMEMBER-family, kept DETERMINISTIC on purpose: a single-element
    # container, count 0, and a count larger than the container (which must
    # return every element exactly once, no padding). Order still varies, so
    # these rely on the array-sorted normalisation.
    ("random-family: deterministic shapes only", [
        ["DEL", "%K"], ["HSET", "%K", "f", "v"],
        ["HRANDFIELD", "%K"], ["HRANDFIELD", "%K", "0"],
        ["HRANDFIELD", "%K", "3"], ["HRANDFIELD", "%K", "-2"],
        ["HRANDFIELD", "%K", "1", "WITHVALUES"],
        ["DEL", "%K2"], ["SADD", "%K2", "only"],
        ["SRANDMEMBER", "%K2"], ["SRANDMEMBER", "%K2", "0"],
        ["SRANDMEMBER", "%K2", "3"], ["SRANDMEMBER", "%K2", "-2"],
        ["DEL", "%K3"], ["ZADD", "%K3", "1", "z"],
        ["ZRANDMEMBER", "%K3"], ["ZRANDMEMBER", "%K3", "0"],
        ["ZRANDMEMBER", "%K3", "3"],
        ["HRANDFIELD", "nosuch:key", "3"], ["SRANDMEMBER", "nosuch:key", "3"]]),

    ("list: LMOVE / RPOPLPUSH / PUSHX / LMPOP", [
        ["DEL", "%K", "%K2"],
        ["RPUSH", "%K", "a", "b", "c"],
        ["RPOPLPUSH", "%K", "%K2"], ["LRANGE", "%K", "0", "-1"], ["LRANGE", "%K2", "0", "-1"],
        ["LMOVE", "%K", "%K2", "LEFT", "RIGHT"], ["LRANGE", "%K2", "0", "-1"],
        ["LMOVE", "%K", "%K", "LEFT", "RIGHT"], ["LRANGE", "%K", "0", "-1"],
        ["LPUSHX", "%K", "x"], ["RPUSHX", "%K", "y"], ["LRANGE", "%K", "0", "-1"],
        ["LPUSHX", "nosuch:list", "x"], ["EXISTS", "nosuch:list"],
        ["LMPOP", "2", "nosuch:list", "%K", "LEFT"],
        ["LMPOP", "2", "nosuch:list", "%K", "LEFT", "COUNT", "2"]]),

    # gh #229 territory: a bad numeric ARGUMENT must be an error, not a
    # plausible integer. The failure shape that shipped was replying with a
    # number that looks like success, so compare against the oracle.
    ("numeric arguments must reject, not guess (gh #229)", [
        ["DEL", "%K"],
        ["SET", "%K", "10"],
        ["INCRBY", "%K", "abc"], ["GET", "%K"],
        ["INCRBY", "%K", "1abc2"], ["GET", "%K"],
        ["INCRBY", "%K", "1,000"], ["GET", "%K"],
        ["INCRBY", "%K", ""], ["GET", "%K"],
        ["INCRBY", "%K", " 5"], ["GET", "%K"],
        ["INCRBYFLOAT", "%K", "x"], ["GET", "%K"],
        ["DEL", "%K"], ["SET", "%K", "abc"], ["INCR", "%K"],
        ["DEL", "%K"], ["SET", "%K", "9223372036854775807"], ["INCR", "%K"],
        ["DEL", "%K"], ["SET", "%K", "-9223372036854775808"], ["DECR", "%K"], ["GET", "%K"],
        ["DEL", "%K"], ["INCRBYFLOAT", "%K", "1.0e2"], ["GET", "%K"]]),

    ("expire: absolute forms and EXPIRETIME", [
        ["DEL", "%K"], ["SET", "%K", "v"],
        ["TTL", "%K"], ["PTTL", "%K"], ["EXPIRETIME", "%K"], ["PEXPIRETIME", "%K"],
        ["EXPIREAT", "%K", "4102444800"], ["EXPIRETIME", "%K"],
        ["PERSIST", "%K"], ["EXPIRETIME", "%K"],
        ["PEXPIREAT", "%K", "4102444800000"], ["PEXPIRETIME", "%K"],
        ["PERSIST", "%K"], ["PERSIST", "%K"],
        ["TTL", "nosuch:key"], ["PTTL", "nosuch:key"],
        ["EXPIRETIME", "nosuch:key"], ["PEXPIRE", "nosuch:key", "100"],
        ["EXPIREAT", "%K", "1"], ["EXISTS", "%K"]]),

    ("zset: POP with count, ZMPOP, ZMSCORE", [
        ["DEL", "%K"],
        ["ZADD", "%K", "1", "a", "2", "b", "3", "c", "4", "d"],
        ["ZPOPMIN", "%K", "2"], ["ZPOPMAX", "%K", "1"], ["ZRANGE", "%K", "0", "-1"],
        ["ZMSCORE", "%K", "d", "zz"],
        ["ZPOPMIN", "%K", "0"], ["ZPOPMIN", "%K", "99"], ["EXISTS", "%K"],
        ["ZPOPMIN", "nosuch:key"], ["ZPOPMIN", "nosuch:key", "2"],
        ["DEL", "%K"], ["ZADD", "%K", "1", "a", "2", "b"],
        ["ZMPOP", "2", "nosuch:key", "%K", "MIN"],
        ["ZMPOP", "2", "nosuch:key", "%K", "MAX", "COUNT", "5"], ["EXISTS", "%K"]]),

    # Redis orders EQUAL-SCORE members lexicographically ascending; that
    # ordering is the whole premise of the score-0 lex-index idiom. The insert
    # order here is SCRAMBLED on purpose: Pion returns reverse-insertion order
    # among ties, so a script that inserts in reverse-alphabetical order gets
    # the right answer from the wrong mechanism and passes on buggy code.
    ("zset: equal-score members order lexicographically", [
        ["DEL", "%K"],
        ["ZADD", "%K", "0", "b", "0", "d", "0", "a", "0", "c"],
        ["ZRANGE", "%K", "0", "-1"], ["ZREVRANGE", "%K", "0", "-1"],
        ["ZRANGEBYLEX", "%K", "-", "+"], ["ZRANGEBYSCORE", "%K", "-inf", "+inf"],
        ["ZRANK", "%K", "a"], ["ZRANK", "%K", "d"], ["ZREVRANK", "%K", "a"],
        ["ZRANGE", "%K", "0", "1"], ["ZRANGEBYLEX", "%K", "[b", "[c"],
        # A partial tie: only b and c share a score, so the tie block has to
        # sort internally while the distinct scores stay put.
        ["DEL", "%K2"],
        ["ZADD", "%K2", "2", "c", "2", "b", "1", "a", "3", "d"],
        ["ZRANGE", "%K2", "0", "-1"], ["ZRANK", "%K2", "b"], ["ZRANK", "%K2", "c"]]),

    ("string: APPEND / GETEX variants / empty values", [
        ["DEL", "%K"],
        ["APPEND", "%K", "abc"], ["GET", "%K"], ["APPEND", "%K", ""], ["STRLEN", "%K"],
        ["SET", "%K", ""], ["STRLEN", "%K"], ["GET", "%K"], ["EXISTS", "%K"],
        ["GETRANGE", "%K", "0", "-1"],
        ["SET", "%K", "v", "EX", "100"],
        ["GETEX", "%K"], ["TTL", "%K"],
        ["GETEX", "%K", "EX", "200"], ["TTL", "%K"],
        ["GETEX", "%K", "EXAT", "4102444800"], ["EXPIRETIME", "%K"],
        ["GETEX", "%K", "PERSIST"], ["TTL", "%K"],
        ["GETEX", "nosuch:key"], ["GETDEL", "nosuch:key"]]),

    # Streams had ZERO correct-type coverage until 2026-08-25, and hand-probing
    # the family that day turned up three real bugs (gh #232): XADD replacing a
    # key of another type, NOMKSTREAM matched at the wrong length so it parsed
    # as the ID, and the pairs shifting as a result. An unprobed family is not
    # a working family.
    #
    # Every id here is EXPLICIT. `XADD k * ...` reads the wall clock, so the two
    # servers answer with ids a millisecond apart and the probe reports a
    # divergence at random — the same nondeterminism `normalize()` had to
    # defuse for the wrong-type sweep.
    ("stream: XADD explicit ids / XLEN / XRANGE / XREVRANGE", [
        ["DEL", "%K"],
        ["XADD", "%K", "1-1", "a", "1"], ["XADD", "%K", "1-2", "b", "2"],
        ["XADD", "%K", "2-1", "c", "3"], ["XLEN", "%K"],
        ["XRANGE", "%K", "-", "+"], ["XREVRANGE", "%K", "+", "-"],
        ["XRANGE", "%K", "1-2", "+"], ["XRANGE", "%K", "-", "1-2"],
        ["XRANGE", "%K", "-", "+", "COUNT", "2"],
        ["XRANGE", "%K", "5-1", "+"], ["XRANGE", "%K", "-", "0-1"],
        ["XLEN", "nosuch:stream"], ["XRANGE", "nosuch:stream", "-", "+"]]),

    ("stream: id monotonicity and the 0-0 floor", [
        ["DEL", "%K"],
        ["XADD", "%K", "5-5", "a", "1"],
        # Each of these must be REFUSED: a duplicate id, a smaller sequence, a
        # smaller ms, and 0-0. gh #242 found the explicit-id path doing no
        # ordering check at all, which leaves a stream whose ids go backwards —
        # and every consumer cursor is built on ids increasing.
        ["XADD", "%K", "5-5", "b", "2"], ["XADD", "%K", "5-4", "c", "3"],
        ["XADD", "%K", "4-9", "d", "4"], ["XADD", "%K", "0-0", "e", "5"],
        ["XLEN", "%K"], ["XRANGE", "%K", "-", "+"],
        ["XADD", "%K", "5-6", "f", "6"], ["XADD", "%K", "6-0", "g", "7"],
        ["XLEN", "%K"]]),

    ("stream: XDEL / XTRIM / NOMKSTREAM", [
        ["DEL", "%K"],
        ["XADD", "%K", "1-1", "a", "1"], ["XADD", "%K", "2-1", "b", "2"],
        ["XADD", "%K", "3-1", "c", "3"], ["XADD", "%K", "4-1", "d", "4"],
        ["XDEL", "%K", "2-1"], ["XLEN", "%K"], ["XRANGE", "%K", "-", "+"],
        ["XDEL", "%K", "2-1"], ["XDEL", "%K", "99-1"],
        ["XTRIM", "%K", "MAXLEN", "2"], ["XLEN", "%K"], ["XRANGE", "%K", "-", "+"],
        ["DEL", "%K2"],
        ["XADD", "%K2", "NOMKSTREAM", "1-1", "a", "1"], ["EXISTS", "%K2"],
        ["XADD", "%K2", "1-1", "a", "1"],
        ["XADD", "%K2", "NOMKSTREAM", "2-1", "b", "2"], ["XLEN", "%K2"],
        # The flag must not be swallowed into the field/value pairs.
        ["XRANGE", "%K2", "-", "+"]]),

    # SCAN was NOT probed, while KEYS was — and gh #243 was a single bug in
    # both: the pattern was ignored, so "SCAN with MATCH, DEL each result", the
    # recommended prefix-delete idiom, deleted the whole keyspace. The half that
    # got the test was the half that could not destroy data.
    ("scan: MATCH must actually match (gh #243)", [
        ["FLUSHALL"],
        ["MSET", "sc:a", "1", "sc:b", "2", "other:c", "3", "zzz", "4"],
        ["SCAN", "0"], ["SCAN", "0", "MATCH", "sc:*"],
        ["SCAN", "0", "MATCH", "other:*"], ["SCAN", "0", "MATCH", "zzz"],
        ["SCAN", "0", "MATCH", "nomatch:*"], ["SCAN", "0", "MATCH", "*"],
        ["SCAN", "0", "MATCH", "sc:?"], ["SCAN", "0", "MATCH", "[sz]*"],
        ["SCAN", "0", "COUNT", "100"],
        ["SCAN", "0", "MATCH", "sc:*", "COUNT", "100"],
        ["KEYS", "sc:*"], ["KEYS", "*"], ["KEYS", "nomatch:*"]]),

    # The matrix ran 513 clean steps while ZUNION/ZINTER aggregated NOTHING —
    # it never sent two keys sharing a member, so SUM was never exercised.
    # `ZADD z1 2 b; ZADD z2 10 b; ZUNION 2 z1 z2` answered b=2 where Redis
    # answers b=12, and ZUNIONSTORE persisted that. Overlap is the whole point
    # of these commands, so probe it.
    ("zset multi-key: aggregation, WEIGHTS, AGGREGATE", [
        ["DEL", "%K"], ["DEL", "%K2"], ["DEL", "%K3"],
        ["ZADD", "%K", "1", "a", "2", "b"], ["ZADD", "%K2", "10", "b", "20", "c"],
        ["ZUNION", "2", "%K", "%K2"], ["ZUNION", "2", "%K", "%K2", "WITHSCORES"],
        ["ZINTER", "2", "%K", "%K2", "WITHSCORES"], ["ZINTERCARD", "2", "%K", "%K2"],
        ["ZDIFF", "2", "%K", "%K2", "WITHSCORES"],
        # WITHSCORES must still be found when it FOLLOWS other options — a scan
        # that only looked one token past the keys dropped it silently.
        ["ZUNION", "2", "%K", "%K2", "WEIGHTS", "2", "3", "WITHSCORES"],
        ["ZUNION", "2", "%K", "%K2", "AGGREGATE", "MIN", "WITHSCORES"],
        ["ZUNION", "2", "%K", "%K2", "AGGREGATE", "MAX", "WITHSCORES"],
        ["ZINTER", "2", "%K", "%K2", "WEIGHTS", "2", "3", "WITHSCORES"],
        ["ZUNIONSTORE", "%K3", "2", "%K", "%K2"], ["ZRANGE", "%K3", "0", "-1", "WITHSCORES"],
        ["ZINTERSTORE", "%K3", "2", "%K", "%K2"], ["ZRANGE", "%K3", "0", "-1", "WITHSCORES"],
        ["ZUNIONSTORE", "%K3", "2", "%K", "%K2", "WEIGHTS", "2", "3"],
        ["ZRANGE", "%K3", "0", "-1", "WITHSCORES"],
        ["ZDIFFSTORE", "%K3", "2", "%K", "%K2"], ["ZRANGE", "%K3", "0", "-1", "WITHSCORES"],
        ["ZUNIONSTORE", "%K3", "2", "nosuch:a", "nosuch:b"], ["EXISTS", "%K3"]]),

    # A SET is a zset whose members all score 1, and Redis accepts one anywhere
    # a zset is taken — including as the exclusion side of ZDIFF.
    ("zset multi-key: SETs participate", [
        ["DEL", "%K"], ["DEL", "%K2"], ["DEL", "%K3"],
        ["SADD", "%K", "a", "b"], ["SADD", "%K2", "b", "c"],
        ["ZUNION", "2", "%K", "%K2", "WITHSCORES"],
        ["ZINTER", "2", "%K", "%K2", "WITHSCORES"],
        ["ZINTERCARD", "2", "%K", "%K2"],
        ["ZDIFF", "2", "%K", "%K2", "WITHSCORES"],
        ["ZUNION", "1", "%K"], ["ZUNION", "1", "%K", "WITHSCORES"],
        ["ZUNIONSTORE", "%K3", "2", "%K", "%K2"], ["ZRANGE", "%K3", "0", "-1", "WITHSCORES"],
        ["ZDIFFSTORE", "%K3", "2", "%K", "%K2"], ["ZRANGE", "%K3", "0", "-1", "WITHSCORES"],
        # mixed: the same member in a set and a zset still aggregates
        ["DEL", "%K2"], ["ZADD", "%K2", "5", "b"],
        ["ZUNION", "2", "%K", "%K2", "WITHSCORES"],
        ["ZINTER", "2", "%K", "%K2", "WITHSCORES"],
        ["ZDIFF", "2", "%K2", "%K", "WITHSCORES"]]),

    # BITCOUNT/BITPOS take a BYTE|BIT unit and negative indices, and none of
    # that was ever sent — only the bare forms. Ranges are where an off-by-one
    # lives, and BITPOS additionally has the "no 0 bit found" / "no 1 bit
    # found" edge whose answer differs depending on whether an end index was
    # given at all.
    ("bitmap: BITCOUNT / BITPOS ranges and BYTE|BIT units", [
        ["DEL", "%K"], ["SET", "%K", "foobar"],
        ["BITCOUNT", "%K"], ["BITCOUNT", "%K", "0", "0"], ["BITCOUNT", "%K", "1", "1"],
        ["BITCOUNT", "%K", "0", "-1"], ["BITCOUNT", "%K", "-2", "-1"],
        ["BITCOUNT", "%K", "0", "5", "BYTE"], ["BITCOUNT", "%K", "5", "30", "BIT"],
        ["BITCOUNT", "%K", "0", "0", "BIT"], ["BITCOUNT", "%K", "100", "200"],
        ["BITCOUNT", "%K", "-100", "-99"],
        ["BITPOS", "%K", "1"], ["BITPOS", "%K", "0"],
        ["BITPOS", "%K", "1", "2"], ["BITPOS", "%K", "1", "0", "-1"],
        ["BITPOS", "%K", "1", "2", "-1", "BYTE"], ["BITPOS", "%K", "1", "0", "5", "BIT"],
        ["DEL", "%K2"], ["SET", "%K2", "\xff\xff\xff"],
        ["BITPOS", "%K2", "0"], ["BITPOS", "%K2", "0", "0", "-1"],
        ["BITCOUNT", "%K2"],
        ["DEL", "%K3"], ["SET", "%K3", "\x00\x00\x00"],
        ["BITPOS", "%K3", "1"], ["BITPOS", "%K3", "0"]]),

    # LPOS has RANK (including NEGATIVE rank, which searches from the tail),
    # COUNT (including COUNT 0 = all) and MAXLEN. Only the bare form was sent.
    ("list: LPOS RANK / COUNT / MAXLEN", [
        ["DEL", "%K"], ["RPUSH", "%K", "a", "b", "c", "a", "b", "c", "a"],
        ["LPOS", "%K", "a"], ["LPOS", "%K", "nope"],
        ["LPOS", "%K", "a", "RANK", "1"], ["LPOS", "%K", "a", "RANK", "2"],
        ["LPOS", "%K", "a", "RANK", "3"], ["LPOS", "%K", "a", "RANK", "4"],
        ["LPOS", "%K", "a", "RANK", "-1"], ["LPOS", "%K", "a", "RANK", "-2"],
        ["LPOS", "%K", "a", "COUNT", "0"], ["LPOS", "%K", "a", "COUNT", "2"],
        ["LPOS", "%K", "a", "COUNT", "99"],
        ["LPOS", "%K", "a", "RANK", "-1", "COUNT", "0"],
        ["LPOS", "%K", "a", "RANK", "2", "COUNT", "0"],
        ["LPOS", "%K", "a", "MAXLEN", "2"], ["LPOS", "%K", "c", "MAXLEN", "3"],
        ["LPOS", "%K", "a", "COUNT", "0", "MAXLEN", "4"]]),

    # The LIMIT clause on the range families, and SINTERCARD's — an ignored
    # LIMIT returns MORE than asked for, which reads as success.
    ("zset/set: LIMIT clauses", [
        ["DEL", "%K"], ["ZADD", "%K", "1", "a", "2", "b", "3", "c", "4", "d"],
        ["ZRANGEBYSCORE", "%K", "-inf", "+inf", "LIMIT", "0", "2"],
        ["ZRANGEBYSCORE", "%K", "-inf", "+inf", "LIMIT", "1", "2"],
        ["ZRANGEBYSCORE", "%K", "-inf", "+inf", "LIMIT", "2", "-1"],
        ["ZRANGEBYSCORE", "%K", "-inf", "+inf", "LIMIT", "99", "2"],
        ["ZREVRANGEBYSCORE", "%K", "+inf", "-inf", "LIMIT", "0", "2"],
        ["ZRANGEBYLEX", "%K", "-", "+", "LIMIT", "1", "2"],
        ["ZRANGE", "%K", "(1", "+inf", "BYSCORE", "LIMIT", "0", "2"],
        ["DEL", "%K2"], ["SADD", "%K2", "a", "b", "c"],
        ["DEL", "%K3"], ["SADD", "%K3", "b", "c", "d"],
        ["SINTERCARD", "2", "%K2", "%K3"],
        ["SINTERCARD", "2", "%K2", "%K3", "LIMIT", "1"],
        ["SINTERCARD", "2", "%K2", "%K3", "LIMIT", "0"],
        ["SINTERCARD", "2", "%K2", "%K3", "LIMIT", "99"]]),

    # The GEO family had NO semantic coverage, and the wrong-type sweep has no
    # GEO fixture — so nothing probed the legitimate case at all. That is how a
    # wrong-type guard added to GEOPOS/GEODIST/GEOHASH could check for ZSET
    # while GEOADD stores ValueType.GEO, making Pion's own GEO commands reject
    # their own keys, with every existing test still green.
    ("geo: GEOADD / GEODIST / GEOHASH / GEOPOS / GEOSEARCH", [
        ["DEL", "%K"],
        ["GEOADD", "%K", "13.361389", "38.115556", "Palermo"],
        ["GEOADD", "%K", "15.087269", "37.502669", "Catania"],
        ["GEOADD", "%K", "13.361389", "38.115556", "Palermo"],   # duplicate -> 0
        ["TYPE", "%K"],
        # NOT probed here: ZCARD/ZSCORE/ZRANGE on a geo key. In Redis a geo key
        # IS a zset so they all work; Pion models GEO as its own ValueType and
        # answers WRONGTYPE. That is a type-model gap of the same kind as the
        # bitmap/HLL fence, tracked with them — not a GEO-command defect, and
        # parking it here would leave this matrix permanently non-zero, which
        # is how a check stops being read.
        ["GEODIST", "%K", "Palermo", "Catania"],
        ["GEODIST", "%K", "Palermo", "Catania", "km"],
        ["GEODIST", "%K", "Palermo", "Catania", "mi"],
        ["GEODIST", "%K", "Palermo", "Catania", "ft"],
        ["GEODIST", "%K", "Palermo", "Palermo"],                 # self -> 0.0000
        ["GEODIST", "%K", "Palermo", "NoSuch"],                  # missing -> nil
        ["GEODIST", "nosuch:geo", "a", "b"],
        ["GEOHASH", "%K", "Palermo"], ["GEOHASH", "%K", "Palermo", "Catania"],
        ["GEOHASH", "%K", "NoSuch"],
        ["GEOPOS", "%K", "Palermo"], ["GEOPOS", "%K", "NoSuch"],
        ["GEOPOS", "nosuch:geo", "a"],
        ["GEOSEARCH", "%K", "FROMLONLAT", "15", "37", "BYRADIUS", "200", "km", "ASC"],
        ["GEOSEARCH", "%K", "FROMLONLAT", "15", "37", "BYRADIUS", "1", "km", "ASC"],
        ["GEOSEARCH", "%K", "FROMMEMBER", "Palermo", "BYRADIUS", "200", "km", "ASC"],
        # a western-hemisphere point, so the sign handling is exercised too
        ["GEOADD", "%K", "-122.4", "37.8", "SF"], ["GEOHASH", "%K", "SF"],
        ["GEODIST", "%K", "Palermo", "SF", "km"]]),

    # SETRANGE past the end ZERO-PADS, and APPEND/SETRANGE are how a string
    # crosses Pion's 23-byte SSO boundary into heap storage. Growth is where a
    # representation switch goes wrong, and gh #241 showed those are cliffs.
    ("string: SETRANGE / APPEND growth and the SSO boundary", [
        ["DEL", "%K"], ["SETRANGE", "%K", "5", "hello"], ["GET", "%K"], ["STRLEN", "%K"],
        ["DEL", "%K"], ["SET", "%K", "hi"], ["SETRANGE", "%K", "10", "x"],
        ["GET", "%K"], ["STRLEN", "%K"],
        ["DEL", "%K"], ["SETRANGE", "%K", "0", ""], ["EXISTS", "%K"],
        ["DEL", "%K"], ["SET", "%K", "abc"], ["SETRANGE", "%K", "1", "ZZ"], ["GET", "%K"],
        # walk across 23 bytes one APPEND at a time
        ["DEL", "%K"], ["APPEND", "%K", "0123456789"], ["STRLEN", "%K"],
        ["APPEND", "%K", "0123456789"], ["STRLEN", "%K"], ["GET", "%K"],
        ["APPEND", "%K", "012"], ["STRLEN", "%K"], ["GET", "%K"],
        ["APPEND", "%K", "3"], ["STRLEN", "%K"], ["GET", "%K"],
        ["APPEND", "%K", "456789"], ["STRLEN", "%K"], ["GET", "%K"],
        ["GETRANGE", "%K", "0", "22"], ["GETRANGE", "%K", "23", "-1"],
        ["SETRANGE", "%K", "22", "!!"], ["GET", "%K"], ["STRLEN", "%K"]]),

    # Binary safety: embedded NULs and high bytes, on both sides of the SSO
    # boundary. `value()` spells bytes >= 128 as '?', so any handler keying off
    # it mangles them — the stream handlers carry that warning explicitly.
    ("string: binary-safe values across SSO/heap", [
        ["DEL", "%K"], ["SET", "%K", b"a\x00b"], ["GET", "%K"], ["STRLEN", "%K"],
        ["APPEND", "%K", b"\x00\xff"], ["GET", "%K"], ["STRLEN", "%K"],
        ["DEL", "%K"], ["SET", "%K", b"\xff\xfe\xfd"], ["GET", "%K"],
        ["GETRANGE", "%K", "0", "0"], ["GETRANGE", "%K", "-1", "-1"],
        ["DEL", "%K"], ["SET", "%K", b"\x00" * 30], ["STRLEN", "%K"], ["GET", "%K"],
        ["DEL", "%K"], ["SET", "%K", b"k\x00ey"], ["EXISTS", "%K"], ["TYPE", "%K"],
        ["DEL", "%K2"], ["HSET", "%K2", b"f\x001", b"v\x001"], ["HGET", "%K2", b"f\x001"],
        ["HGET", "%K2", "f"], ["HLEN", "%K2"],
        ["DEL", "%K3"], ["SADD", "%K3", b"m\x001", b"m\xff2"], ["SCARD", "%K3"],
        ["SISMEMBER", "%K3", b"m\x001"], ["SISMEMBER", "%K3", "m"]]),

    # The hash-field TTL family (Redis 7.4) has a full option surface —
    # NX/XX/GT/LT — and nine commands sharing it. Only HTTL/HPERSIST were
    # touched, and only in the wrong-type direction.
    ("hash: HEXPIRE family and its NX/XX/GT/LT flags", [
        ["DEL", "%K"], ["HSET", "%K", "f1", "v1", "f2", "v2"],
        ["HTTL", "%K", "FIELDS", "1", "f1"],
        ["HPERSIST", "%K", "FIELDS", "1", "f1"],
        ["HEXPIRE", "%K", "100", "FIELDS", "1", "f1"],
        ["HTTL", "%K", "FIELDS", "1", "f1"],
        ["HTTL", "%K", "FIELDS", "2", "f1", "f2"],
        ["HTTL", "%K", "FIELDS", "1", "nosuch"],
        ["HEXPIRE", "%K", "200", "NX", "FIELDS", "1", "f1"],
        ["HEXPIRE", "%K", "200", "XX", "FIELDS", "1", "f1"],
        ["HEXPIRE", "%K", "100", "GT", "FIELDS", "1", "f1"],
        ["HEXPIRE", "%K", "300", "GT", "FIELDS", "1", "f1"],
        ["HEXPIRE", "%K", "100", "LT", "FIELDS", "1", "f1"],
        ["HPERSIST", "%K", "FIELDS", "1", "f1"],
        ["HPERSIST", "%K", "FIELDS", "1", "f1"],
        ["HPEXPIRE", "%K", "100000", "FIELDS", "1", "f2"],
        ["HPTTL", "%K", "FIELDS", "1", "f2"],
        ["HEXPIRETIME", "%K", "FIELDS", "1", "f2"],
        ["HEXPIRE", "%K", "100", "FIELDS", "1", "nosuch"],
        ["HEXPIRE", "nosuch:hash", "100", "FIELDS", "1", "f"],
        ["HTTL", "nosuch:hash", "FIELDS", "1", "f"],
        # a 0 TTL deletes the field outright
        ["HEXPIRE", "%K", "0", "FIELDS", "1", "f1"], ["HGET", "%K", "f1"],
        ["HLEN", "%K"]]),

    ("scan: HSCAN / SSCAN / ZSCAN", [
        ["DEL", "%K"],
        ["HSET", "%K", "f1", "v1", "f2", "v2", "g1", "w1"],
        ["HSCAN", "%K", "0"], ["HSCAN", "%K", "0", "MATCH", "f*"],
        ["HSCAN", "%K", "0", "MATCH", "nomatch*"],
        ["DEL", "%K2"], ["SADD", "%K2", "m1", "m2", "x1"],
        ["SSCAN", "%K2", "0"], ["SSCAN", "%K2", "0", "MATCH", "m*"],
        ["DEL", "%K3"], ["ZADD", "%K3", "1", "a1", "2", "a2", "3", "b1"],
        ["ZSCAN", "%K3", "0"], ["ZSCAN", "%K3", "0", "MATCH", "a*"],
        ["SSCAN", "nosuch:key", "0"], ["HSCAN", "nosuch:key", "0"]]),
]


def run_semantics(pion, redis):
    diffs, same, n = [], 0, 0
    for name, script in SEMANTIC_SCRIPTS:
        for step, cmd in enumerate(script):
            c = [{"%K": "dfs:k", "%K2": "dfs:k2", "%K3": "dfs:k3"}.get(p, p)
                 for p in cmd]
            n += 1
            try:
                rp = normalize(pion.cmd(*c), c)
                rr = normalize(redis.cmd(*c), c)
            except (EOFError, socket.timeout) as e:
                diffs.append((name, c, f"TRANSPORT {type(e).__name__}", "-"))
                break
            # A setup DEL's reply counts keys that happened to exist, which
            # depends on prior state — and Pion is long-lived here while Redis
            # is spawned fresh per run. Comparing it produces false positives
            # that look exactly like real divergences. The DEL still RUNS on
            # both; only its reply is uncompared.
            if c[0] == "DEL":
                same += 1
            elif rp == rr:
                same += 1
            else:
                diffs.append((name, c, rp, rr))
    return diffs, same, n


if __name__ == "__main__":
    sys.exit(main())
