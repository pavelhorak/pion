#!/usr/bin/env python3
"""A key whose TTL has passed is gone for every command (#45).

Only GET used to check a key's deadline when it read the key. Every other
command saw an expired key as live until the background sweep reached it:
EXISTS answered 1, HGETALL returned the old fields, and a write (HSET, RPUSH,
INCR, APPEND) added to the old value and kept its passed deadline, so the
sweep then deleted the new data too. Expiry was not logged either, so a key
created again after it expired came back from a restart with the old value
under the old deadline.

  [1] each command on a key just past its deadline, against Redis 8: the
      reply and what is left afterwards. Both servers run with active expiry
      off (DEBUG SET-ACTIVE-EXPIRE 0), so only lazy expiry can remove the key.
  [2] durability: a key created again after it expired (lazily, or by the
      sweep) survives a SIGKILL as the new key.
  [3] RANDOMKEY is random and never returns an expired key; KEYS and SCAN
      skip expired keys. DBSIZE counts a key until it is removed, as Redis's
      does.
  [4] DEBUG: refused unless --enable-debug-command allows it, as Redis.

    python3 tests/test_lazy_expiry.py [--port 6506]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, wait_ready_pid, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.abspath(os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


class Server:
    def __init__(self, work, port, extra=(), sub="data"):
        self.port, self.dir, self.extra = port, os.path.join(work, sub), list(extra)
        os.makedirs(self.dir, exist_ok=True)
        self.proc = None

    def start(self) -> Conn:
        self.proc = subprocess.Popen([BIN, "-p", str(self.port), "-w", "1", "--no-crash-log", "--no-auto-detect",
                                      "--no-auto-embed", *self.extra], cwd=self.dir,
                                     stdout=open(os.path.join(self.dir, "log"), "a"), stderr=subprocess.STDOUT)
        wait_ready_pid(self.port, self.proc, 60)
        return Conn(self.port, timeout=30)

    def stop(self, sig=signal.SIGTERM):
        if self.proc:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            wait_port_free(self.port)


# Each type's setup, then commands run on the key once its TTL has passed.
SETUP = {
    "string": [["SET", "k", "12"]],
    "hash": [["HSET", "k", "f", "1", "g", "2"]],
    "list": [["RPUSH", "k", "a", "b"]],
    "set": [["SADD", "k", "a", "b"]],
    "zset": [["ZADD", "k", "1", "a", "2", "b"]],
    "stream": [["XADD", "k", "1-1", "f", "v"]],
}
PROBES = {
    "string": [["GET", "k"], ["EXISTS", "k"], ["EXISTS", "k", "k", "k2"], ["TYPE", "k"], ["STRLEN", "k"],
               ["INCR", "k"], ["APPEND", "k", "x"], ["GETRANGE", "k", "0", "-1"], ["SETRANGE", "k", "1", "x"],
               ["MGET", "k", "k2"], ["GETDEL", "k"], ["GETEX", "k"], ["GETEX", "k", "PERSIST"], ["SETNX", "k", "v"],
               ["SET", "k", "v", "NX"], ["SET", "k", "v", "XX"], ["SET", "k", "v", "GET"],
               ["SET", "k", "v", "KEEPTTL"], ["GETSET", "k", "v"], ["INCRBYFLOAT", "k", "1.5"], ["DECR", "k"],
               ["INCRBY", "k", "5"], ["TTL", "k"], ["PTTL", "k"], ["EXPIRETIME", "k"], ["PERSIST", "k"],
               ["EXPIRE", "k", "100"], ["PEXPIRE", "k", "100", "XX"], ["DUMP", "k"], ["OBJECT", "ENCODING", "k"],
               ["RENAME", "k", "k2"], ["RENAMENX", "k", "k2"], ["COPY", "k", "k2"], ["MSETNX", "k", "v"],
               ["LCS", "k", "k"], ["BITCOUNT", "k"], ["GETBIT", "k", "1"], ["SETBIT", "k", "1", "1"],
               ["BITPOS", "k", "1"], ["DEL", "k"], ["UNLINK", "k"], ["TOUCH", "k"], ["DBSIZE"], ["KEYS", "*"],
               ["RANDOMKEY"], ["SCAN", "0"], ["SORT", "k"],
               ["EVAL", "return redis.call('EXISTS', KEYS[1])", "1", "k"],
               ["EVAL", "return redis.call('GET', KEYS[1])", "1", "k"]],
    "hash": [["HGET", "k", "f"], ["HLEN", "k"], ["HGETALL", "k"], ["HSET", "k", "h", "3"], ["HINCRBY", "k", "f", "1"],
             ["HEXISTS", "k", "f"], ["HSETNX", "k", "f", "9"], ["HDEL", "k", "f"], ["HKEYS", "k"], ["HVALS", "k"],
             ["HMGET", "k", "f"], ["HSTRLEN", "k", "f"], ["HRANDFIELD", "k"], ["HSCAN", "k", "0"], ["TYPE", "k"],
             ["HMSET", "k", "z", "1"], ["HINCRBYFLOAT", "k", "f", "1"]],
    "list": [["LLEN", "k"], ["LRANGE", "k", "0", "-1"], ["LPUSH", "k", "z"], ["RPUSH", "k", "z"], ["LPOP", "k"],
             ["RPOP", "k"], ["LINDEX", "k", "0"], ["LPUSHX", "k", "z"], ["RPUSHX", "k", "z"], ["LSET", "k", "0", "z"],
             ["LINSERT", "k", "BEFORE", "a", "z"], ["LREM", "k", "0", "a"], ["LPOS", "k", "a"],
             ["LTRIM", "k", "0", "0"], ["LMOVE", "k", "k2", "LEFT", "LEFT"], ["RPOPLPUSH", "k", "k2"],
             ["LMPOP", "1", "k", "LEFT"], ["SORT", "k", "ALPHA"]],
    "set": [["SCARD", "k"], ["SMEMBERS", "k"], ["SADD", "k", "c"], ["SISMEMBER", "k", "a"], ["SREM", "k", "a"],
            ["SPOP", "k"], ["SRANDMEMBER", "k"], ["SINTER", "k"], ["SUNION", "k"], ["SDIFF", "k"],
            ["SMOVE", "k", "k2", "a"], ["SINTERSTORE", "k2", "k"], ["SMISMEMBER", "k", "a"], ["SINTERCARD", "1", "k"]],
    "zset": [["ZCARD", "k"], ["ZRANGE", "k", "0", "-1"], ["ZADD", "k", "3", "c"], ["ZSCORE", "k", "a"],
             ["ZINCRBY", "k", "1", "a"], ["ZRANK", "k", "a"], ["ZPOPMIN", "k"], ["ZPOPMAX", "k"], ["ZREM", "k", "a"],
             ["ZCOUNT", "k", "-inf", "+inf"], ["ZRANGEBYSCORE", "k", "-inf", "+inf"],
             ["ZUNIONSTORE", "k2", "1", "k"], ["ZMSCORE", "k", "a"], ["ZRANDMEMBER", "k"], ["ZMPOP", "1", "k", "MIN"]],
    "stream": [["XLEN", "k"], ["XRANGE", "k", "-", "+"], ["XADD", "k", "*", "f", "v"], ["XDEL", "k", "1-1"],
               ["XTRIM", "k", "MAXLEN", "0"], ["XREAD", "STREAMS", "k", "0"], ["XINFO", "STREAM", "k"], ["TYPE", "k"]],
}
READ = {"string": ["GET", "k"], "hash": ["HGETALL", "k"], "list": ["LRANGE", "k", "0", "-1"],
        "set": ["SMEMBERS", "k"], "zset": ["ZRANGE", "k", "0", "-1", "WITHSCORES"], "stream": ["XLEN", "k"]}
STREAM_ID = re.compile(rb"^\d+-\d+$")


def norm(v):
    """XADD * answers a clock-made ID: compare its shape."""
    if isinstance(v, bytes) and STREAM_ID.match(v) and v != b"1-1":
        return b"<id>"
    if isinstance(v, list):
        return [norm(x) for x in v]
    return v


def state(c: Conn):
    """What is left: k's existence, type, TTL (as a sign), contents, k2, and
    DBSIZE (which counts expired keys not yet removed, in Redis as here)."""
    dbsize = c.cmd("DBSIZE")
    t = c.cmd("TYPE", "k")
    read = READ.get(t if isinstance(t, str) else "", None)
    body = norm(c.cmd(*read)) if read else None
    if isinstance(body, list) and t == "set":
        body = sorted(body)
    elif isinstance(body, list) and t == "hash":
        body = sorted(zip(body[0::2], body[1::2]))
    pttl = c.cmd("PTTL", "k")
    return [dbsize, t, "ttl" if pttl > 0 else pttl, body, c.cmd("EXISTS", "k2")]


def run_against_redis(p: Conn, r: Conn, pport: int, rport: int):
    print("[1] commands on an expired key, as Redis answers them (active expiry off on both)")
    total = differ = 0
    for typ, probes in PROBES.items():
        for cmd in probes:
            total += 1
            for c in (p, r):
                c.cmd("FLUSHALL")
                for s in SETUP[typ]:
                    c.cmd(*s)
                c.cmd("PEXPIRE", "k", "1")
            time.sleep(0.01)
            a, b = norm(p.cmd(*cmd)), norm(r.cmd(*cmd))
            sa, sb = state(p), state(r)
            if isinstance(a, list) and cmd[0] in ("SMEMBERS", "SINTER", "SUNION", "SDIFF", "HKEYS", "HVALS"):
                a, b = sorted(a), sorted(b) if isinstance(b, list) else b
            if a != b or sa != sb:
                differ += 1
                check(f"{typ}: {' '.join(cmd)}", False, f"pion {a!r:.70} {sa}  redis {b!r:.70} {sb}")
    check(f"{total} commands on an expired key answer and leave the keyspace as Redis does", differ == 0,
          f"{differ} differ")
    # several keys, a mix of live and expired
    for c in (p, r):
        c.cmd("FLUSHALL")
        c.cmd("MSET", "a", "1", "b", "2", "c", "3")
        c.cmd("PEXPIRE", "b", "1")
    time.sleep(0.01)
    for cmd in (["MGET", "a", "b", "c"], ["EXISTS", "a", "b", "c"], ["DEL", "a", "b"], ["KEYS", "*"], ["DBSIZE"]):
        a, b = p.cmd(*cmd), r.cmd(*cmd)
        if cmd[0] == "KEYS":
            a, b = sorted(a), sorted(b)
        check(f"live and expired keys: {' '.join(cmd)}", a == b, f"pion {a!r} redis {b!r}")
    # HINCRBY creating a hash stores the number (it stored a pointer's text)
    for c in (p, r):
        c.cmd("FLUSHALL")
        c.cmd("HINCRBY", "hc", "f", "5")
        c.cmd("HINCRBY", "hc", "f", "1")
    a, b = p.cmd("HGET", "hc", "f"), r.cmd("HGET", "hc", "f")
    check("HINCRBY on a missing key, then again: the field reads 6", a == b == b"6", f"pion {a!r} redis {b!r}")
    # a transaction and WATCH
    for c in (p, r):
        c.cmd("FLUSHALL")
        c.cmd("SET", "w", "1", "PX", "30")
    wp, wr = Conn(pport), Conn(rport)
    for c in (wp, wr):
        c.cmd("WATCH", "w")
    time.sleep(0.06)
    ea = wp.pipeline([("MULTI",), ("SET", "x", "1"), ("EXEC",)])[-1]
    eb = wr.pipeline([("MULTI",), ("SET", "x", "1"), ("EXEC",)])[-1]
    ea = None if ea in (None, [], b"") else ea
    eb = None if eb in (None, [], b"") else eb
    check("WATCH on a key that expires before EXEC: EXEC as Redis answers it", ea == eb, f"pion {ea!r} redis {eb!r}")
    wp.close()
    wr.close()


def run_durability(work: str, port: int):
    print("[2] a key created again after it expired survives a restart as the new key")
    s = Server(work, port, ["--enable-debug-command", "yes"], "durable")
    c = s.start()
    c.cmd("DEBUG", "SET-ACTIVE-EXPIRE", "0")
    c.cmd("HSET", "h", "a", "1")
    c.cmd("PEXPIRE", "h", "50")
    c.cmd("RPUSH", "l", "old")
    c.cmd("PEXPIRE", "l", "50")
    c.cmd("SET", "s", "old", "PX", "50")
    c.cmd("SET", "r", "x", "PX", "50")
    time.sleep(0.12)
    c.cmd("HSET", "h", "b", "2")                    # lazily expired, then created again
    c.cmd("RPUSH", "l", "new")
    c.cmd("APPEND", "s", "new")
    check("live: the new values only", c.cmd("HGETALL", "h") == [b"b", b"2"] and c.cmd("LRANGE", "l", "0", "-1") == [b"new"]
          and c.cmd("GET", "s") == b"new" and c.cmd("PTTL", "h") == -1)
    check("a read expires a key too", c.cmd("EXISTS", "r") == 0)
    c.cmd("DEBUG", "SET-ACTIVE-EXPIRE", "1")
    c.cmd("SET", "sw", "old", "PX", "50")
    deadline = time.time() + 20
    while time.time() < deadline and c.cmd("DBSIZE") > 3:   # the sweep takes sw (h, l, s stay)
        time.sleep(0.05)
    c.cmd("SET", "sw2", "x")
    time.sleep(0.3)                                  # housekeeping logs the swept key's DEL
    c.close()
    s.stop(signal.SIGKILL)
    c = s.start()
    check("after a SIGKILL: the hash is the new one, with no TTL", c.cmd("HGETALL", "h") == [b"b", b"2"]
          and c.cmd("PTTL", "h") == -1, repr(c.cmd("HGETALL", "h")))
    check("...the list and the string too", c.cmd("LRANGE", "l", "0", "-1") == [b"new"] and c.cmd("GET", "s") == b"new")
    check("...a key expired by a read stays gone", c.cmd("EXISTS", "r") == 0)
    check("...a key the sweep expired stays gone", c.cmd("EXISTS", "sw") == 0)
    time.sleep(0.2)
    check("...and nothing vanishes later", c.cmd("HGETALL", "h") == [b"b", b"2"] and c.cmd("GET", "s") == b"new")
    c.close()
    s.stop()


def run_scan(work: str, port: int):
    print("[3] RANDOMKEY, KEYS, SCAN and DBSIZE")
    s = Server(work, port, ["--enable-debug-command", "local"], "scan")
    c = s.start()
    c.cmd("DEBUG", "SET-ACTIVE-EXPIRE", "0")
    c.cmd("MSET", *[x for k in range(50) for x in (f"live:{k}", "v")])
    for k in range(50):
        c.cmd("SET", f"dead:{k}", "v", "PX", "1")
    time.sleep(0.02)
    picks = {c.cmd("RANDOMKEY") for _ in range(40)}
    check("RANDOMKEY is random", len(picks) > 5, repr(sorted(picks)[:5]))
    check("...and never an expired key", all(p.startswith(b"live:") for p in picks), repr(picks))
    check("KEYS skips expired keys", sorted(c.cmd("KEYS", "*")) == sorted(f"live:{k}".encode() for k in range(50)))
    check("SCAN skips them", len(c.cmd("SCAN", "0", "COUNT", "1000")[1]) == 50)
    c.close()
    s.stop()


def run_debug(work: str, port: int):
    print("[4] DEBUG")
    s = Server(work, port, (), "debug-off")
    c = s.start()
    r = c.cmd("DEBUG", "SET-ACTIVE-EXPIRE", "0")
    check("refused by default, as Redis 7 and later", isinstance(r, RespError) and "DEBUG command not allowed" in r
          and "enable-debug-command" in r, repr(r))
    check("CONFIG GET enable-debug-command: no", c.cmd("CONFIG", "GET", "enable-debug-command")
          == [b"enable-debug-command", b"no"])
    c.close()
    s.stop()
    s = Server(work, port, ["--enable-debug-command", "yes"], "debug-on")
    c = s.start()
    h = c.cmd("DEBUG", "HELP")
    check("HELP lists what Pion has", isinstance(h, list) and "SET-ACTIVE-EXPIRE <0|1>" in h and "SLEEP <seconds>" in h, repr(h))
    t0 = time.time()
    check("SLEEP sleeps", c.cmd("DEBUG", "SLEEP", "0.2") == "OK" and time.time() - t0 >= 0.18)
    r = c.cmd("DEBUG", "JMAP")
    check("a subcommand Pion does not have is refused, not acknowledged",
          r == "ERR unknown subcommand or wrong number of arguments for 'JMAP'. Try DEBUG HELP.", repr(r))
    c.close()
    s.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6506)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion_lazyexp_")
    redis = None
    try:
        rs = shutil.which("redis-server")
        if rs:
            sk = socket.socket()
            sk.bind(("127.0.0.1", 0))
            rport = sk.getsockname()[1]
            sk.close()
            redis = subprocess.Popen([rs, "--port", str(rport), "--save", "", "--appendonly", "no", "--dir", work,
                                      "--enable-debug-command", "yes"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait_ready_pid(rport, redis, 30)
            s = Server(work, a.port, ["--enable-debug-command", "yes"], "vs-redis")
            p = s.start()
            r = Conn(rport)
            for c in (p, r):
                check("DEBUG SET-ACTIVE-EXPIRE 0", c.cmd("DEBUG", "SET-ACTIVE-EXPIRE", "0") == "OK")
            run_against_redis(p, r, a.port, rport)
            p.close()
            r.close()
            s.stop()
        else:
            print("  SKIP  [1]: no redis-server on PATH")
        run_durability(work, a.port)
        run_scan(work, a.port)
        run_debug(work, a.port)
    finally:
        if redis:
            redis.send_signal(signal.SIGTERM)
            redis.wait()
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
