#!/usr/bin/env python3
"""DUMP, RESTORE and MIGRATE keep every value whole, as Redis's do (#41).

  1. Every type survives DUMP → DEL → RESTORE: strings (short, binary, 3 MB),
     integers, lists (short and past 1,024 entries), hashes with a field TTL,
     sets, sorted sets, geo keys, bitmaps, HyperLogLogs, streams, vector sets.
     (Sorted sets, geo keys, streams, HyperLogLogs and long lists came back
     empty, and a value over 1 MB overflowed DUMP's buffer.)
  2. RESTORE's TTL argument is the TTL, ABSTTL takes a unix time in ms, an
     expired TTL restores nothing, REPLACE drops the old key's TTL.
  3. A restored key, and a migrated key's deletion, survive a SIGKILL.
  4. MIGRATE between two servers: moves, COPY, REPLACE, KEYS, NOKEY, AUTH,
     the target's own error, a refused connection.
  5. With a redis-server on PATH: RESTORE's errors and their order match
     Redis's, a Redis payload is refused here and a Pion payload there, and
     MIGRATE into Redis reports the target's refusal.

    python3 tests/test_dump_restore.py [--port 6502]
"""
from __future__ import annotations

import argparse
import os
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
    def __init__(self, work, port, sub, extra=()):
        self.port, self.dir, self.extra = port, os.path.join(work, sub), list(extra)
        os.makedirs(self.dir, exist_ok=True)
        self.proc = None

    def start(self) -> Conn:
        self.proc = subprocess.Popen([BIN, "-p", str(self.port), "-w", "1", "--no-crash-log", "--no-auto-detect",
                                      "--no-auto-embed", *self.extra], cwd=self.dir,
                                     stdout=open(os.path.join(self.dir, "log"), "a"), stderr=subprocess.STDOUT)
        pw = self.extra[self.extra.index("--requirepass") + 1] if "--requirepass" in self.extra else None
        wait_ready_pid(self.port, self.proc, 60, password=pw)
        return Conn(self.port, timeout=30)

    def kill(self):
        if self.proc:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait()
            self.proc = None
            wait_port_free(self.port)

    def stop(self):
        if self.proc:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            wait_port_free(self.port)


def snapshot(c: Conn, k: str):
    """A comparable picture of a key's value, whatever its type."""
    t = c.cmd("TYPE", k)
    if t == "string":
        return (t, c.cmd("GET", k))
    if t == "list":
        return (t, c.cmd("LRANGE", k, "0", "-1"))
    if t == "hash":
        return (t, sorted(c.cmd("HGETALL", k)))
    if t == "set":
        return (t, sorted(c.cmd("SMEMBERS", k)))
    if t == "zset":
        return (t, c.cmd("ZRANGE", k, "0", "-1", "WITHSCORES"))
    if t == "stream":
        return (t, c.cmd("XRANGE", k, "-", "+"))
    if t == "vectorset":
        return (t, sorted(c.cmd("VRANGE", k, "-", "+", "1000") or []), c.cmd("VCARD", k))
    return (t, None)


def fill(c: Conn):
    c.cmd("SET", "s:short", "hello")
    c.cmd("SET", "s:bin", bytes(range(256)))
    c.cmd("SET", "s:big", b"x" * (3 << 20))
    c.cmd("SET", "s:int", "12345")
    c.cmd("INCR", "s:int")
    c.cmd("RPUSH", "l:short", "a", "b", "c")
    c.cmd("RPUSH", "l:long", *[f"e{k}" for k in range(1500)])
    c.cmd("HSET", "h", "f1", "v1", "f2", b"\x00\xff")
    c.cmd("HEXPIRE", "h", "1000", "FIELDS", "1", "f1")
    c.cmd("SADD", "set", "a", "b", "c")
    c.cmd("ZADD", "z", "1.5", "a", "-2", "b", "inf", "c")
    c.cmd("GEOADD", "g", "13.361389", "38.115556", "Palermo", "15.087269", "37.502669", "Catania")
    c.cmd("SETBIT", "bm", "100", "1")
    c.cmd("PFADD", "hll", "a", "b", "c", "d")
    c.cmd("XADD", "st", "1-1", "f", "v")
    c.cmd("XADD", "st", "2-5", "a", "b", "c", "d")
    c.cmd("VADD", "vs", "VALUES", "3", "1", "0", "0", "e1")
    c.cmd("VADD", "vs", "VALUES", "3", "0", "1", "0", "e2", "SETATTR", '{"x":1}')
    return ["s:short", "s:bin", "s:big", "s:int", "l:short", "l:long", "h", "set", "z", "g", "bm", "hll", "st", "vs"]


def run_roundtrip(c: Conn):
    print("[1] every type round-trips")
    keys = fill(c)
    for k in keys:
        before = snapshot(c, k)
        payload = c.cmd("DUMP", k)
        ok = isinstance(payload, bytes) and len(payload) > 10
        c.cmd("DEL", k)
        r = c.cmd("RESTORE", k, "0", payload) if ok else None
        after = snapshot(c, k)
        check(f"{k} ({before[0]}): DUMP, RESTORE, same value", ok and r == "OK" and after == before,
              f"restore={r!r} before={str(before)[:60]} after={str(after)[:60]}")
    check("the hash field keeps its TTL", c.cmd("HTTL", "h", "FIELDS", "1", "f1")[0] > 0)
    check("DUMP of a missing key is nil", c.cmd("DUMP", "nosuch") is None)


def run_ttl(c: Conn):
    print("[2] TTLs")
    c.cmd("SET", "t", "v")
    p = c.cmd("DUMP", "t")
    check("RESTORE ttl 5000: PTTL ~5000", c.cmd("RESTORE", "t2", "5000", p) == "OK"
          and 4000 < c.cmd("PTTL", "t2") <= 5000)
    check("RESTORE ttl 0: no TTL", c.cmd("RESTORE", "t3", "0", p) == "OK" and c.cmd("PTTL", "t3") == -1)
    future = int(time.time() * 1000) + 60000
    c.cmd("RESTORE", "t4", str(future), p, "ABSTTL")
    check("ABSTTL: the given unix time", 50000 < c.cmd("PTTL", "t4") <= 60000)
    check("an ABSTTL in the past restores nothing", c.cmd("RESTORE", "t5", "1000", p, "ABSTTL") == "OK"
          and c.cmd("EXISTS", "t5") == 0)
    c.cmd("SET", "t6", "old", "EX", "1000")
    check("REPLACE drops the old key's TTL", c.cmd("RESTORE", "t6", "0", p, "REPLACE") == "OK"
          and c.cmd("PTTL", "t6") == -1 and c.cmd("GET", "t6") == b"v")
    check("IDLETIME and FREQ are accepted", c.cmd("RESTORE", "t7", "0", p, "IDLETIME", "10") == "OK"
          and c.cmd("RESTORE", "t8", "0", p, "FREQ", "5") == "OK")


def run_durable(s: Server, c: Conn) -> Conn:
    print("[3] durability")
    c.cmd("SET", "d", "dv")
    p = c.cmd("DUMP", "d")
    c.cmd("RESTORE", "d:copy", "100000", p)
    c.cmd("ZADD", "dz", "1", "a")
    pz = c.cmd("DUMP", "dz")
    c.cmd("DEL", "dz")
    c.cmd("RESTORE", "dz", "0", pz)
    c.close()
    s.kill()
    c = s.start()
    check("a restored key survives a SIGKILL, with its TTL",
          c.cmd("GET", "d:copy") == b"dv" and c.cmd("PTTL", "d:copy") > 0)
    check("...a restored sorted set too", c.cmd("ZRANGE", "dz", "0", "-1", "WITHSCORES") == [b"a", b"1"])
    return c


def run_migrate(work: str, base: int, c: Conn) -> Conn:
    print("[4] MIGRATE between two servers")
    t = Server(work, base + 10, "target", ["--requirepass", "pw"])
    try:
        _migrate_to(t, base, c)
    finally:
        t.stop()
    # a target that refuses AUTH: Redis keeps every key, even one the target
    # stored (its RESTORE needed no password)
    c.cmd("SET", "m5", "five")
    c.cmd("SET", "m6", "six")
    t2 = Server(work, base + 10, "target2")
    try:
        t2c = t2.start()
        port = str(base + 10)
        r = c.cmd("MIGRATE", "127.0.0.1", port, "m6", "0", "1000", "AUTH", "pw")
        check("a refused AUTH comes back as the target's error", isinstance(r, RespError)
              and r.startswith("ERR Target instance replied with error: ") and "AUTH" in r, repr(r))
        check("...and no key is deleted, as in Redis", c.cmd("GET", "m6") == b"six")
        check("moved", c.cmd("MIGRATE", "127.0.0.1", port, "m5", "0", "1000") == "OK")
        t2c.close()
    finally:
        t2.stop()
    return c


def _migrate_to(t: Server, base: int, c: Conn):
    tc = t.start()
    tc.cmd("AUTH", "pw")
    c.cmd("SET", "m1", "one")
    c.cmd("RPUSH", "m2", *[f"x{k}" for k in range(2000)])
    c.cmd("SET", "m3", "three", "EX", "1000")
    port = str(base + 10)
    check("NOKEY when no key exists", c.cmd("MIGRATE", "127.0.0.1", port, "nosuch", "0", "1000") == "NOKEY")
    r = c.cmd("MIGRATE", "127.0.0.1", port, "m1", "0", "1000")
    check("a target that needs a password replies NOAUTH", isinstance(r, RespError) and "NOAUTH" in r, repr(r))
    check("...and the key stays", c.cmd("GET", "m1") == b"one")
    r = c.cmd("MIGRATE", "localhost", port, "", "0", "2000", "AUTH", "pw", "KEYS", "m1", "m2", "m3", "absent")
    check("AUTH + KEYS moves every key that exists (a host name resolves)", r == "OK", repr(r))
    check("...gone here", c.cmd("EXISTS", "m1", "m2", "m3") == 0)
    check("...there, whole", tc.cmd("GET", "m1") == b"one" and tc.cmd("LLEN", "m2") == 2000
          and tc.cmd("LINDEX", "m2", "1999") == b"x1999")
    check("...with the TTL", 900000 < tc.cmd("PTTL", "m3") <= 1000000)
    c.cmd("SET", "m4", "four")
    tc.cmd("SET", "m4", "theirs")
    r = c.cmd("MIGRATE", "127.0.0.1", port, "m4", "0", "1000", "AUTH", "pw")
    check("the target's error comes back", isinstance(r, RespError) and "BUSYKEY" in r, repr(r))
    check("...and the key stays", c.cmd("GET", "m4") == b"four")
    check("COPY REPLACE", c.cmd("MIGRATE", "127.0.0.1", port, "m4", "0", "1000", "COPY", "REPLACE", "AUTH", "pw") == "OK"
          and c.cmd("GET", "m4") == b"four" and tc.cmd("GET", "m4") == b"four")
    r = c.cmd("MIGRATE", "127.0.0.1", port, "m4", "0", "1000", "KEYS", "m4")
    check("KEYS needs the empty key argument", isinstance(r, RespError) and "empty string" in r, repr(r))
    r = c.cmd("MIGRATE", "127.0.0.1", str(base + 33), "m4", "0", "300")
    check("a refused connection is IOERR", r == "IOERR error or timeout connecting to the client", repr(r))
    r = c.cmd("MIGRATE", "127.0.0.1", "x", "m4", "0", "300")
    check("a port that is not a number fails to connect, as Redis's atoi",
          r == "IOERR error or timeout connecting to the client", repr(r))
    r = c.cmd("MIGRATE", "127.0.0.1", port, "m4", "0", "x")
    check("a timeout that is not a number is refused", r == "ERR value is not an integer or out of range", repr(r))
    tc.close()


def run_redis(c: Conn, redis_port: int):
    print("[5] against redis-server")
    r = Conn(redis_port)
    for conn in (c, r):
        conn.cmd("SET", "rk", "v")
    pp = c.cmd("DUMP", "rk")
    rp = r.cmd("DUMP", "rk")
    cases = [
        ("RESTORE", "rk", "0", pp), ("RESTORE", "nk", "x", pp), ("RESTORE", "nk", "-1", pp),
        ("RESTORE", "nk", "0", b"garbage"), ("RESTORE", "nk", "0", pp, "BADOPT"),
        ("RESTORE", "nk", "0", pp, "IDLETIME", "-1"), ("RESTORE", "nk", "0", pp, "FREQ", "300"),
        ("RESTORE", "nk", "0", pp, "IDLETIME", "5", "FREQ", "5"), ("RESTORE", "nk", "0", pp, "ABSTTL", "IDLETIME"),
        ("RESTORE", "rk", "x", b"garbage"), ("RESTORE", "rk", "0", b"garbage", "REPLACE"), ("RESTORE", "nk"),
        ("RESTORE-ASKING", "nk"), ("DUMP",), ("DUMP", "a", "b"),
    ]
    for cmd in cases:
        a = c.cmd(*cmd)
        b = r.cmd(*[pp if x is pp else x for x in cmd])
        if cmd[0] == "RESTORE" and cmd[-1] is pp and cmd[1] == "nk" and cmd[2] == "0":
            continue
        check(f"{cmd[0]} {' '.join(str(x)[:10] for x in cmd[1:3])}...: as Redis", a == b, f"pion {a!r} redis {b!r}")
    check("a Redis payload is refused here", c.cmd("RESTORE", "x1", "0", rp) == "ERR DUMP payload version or checksum are wrong")
    check("a Pion payload is refused by Redis", r.cmd("RESTORE", "x1", "0", pp) == "ERR DUMP payload version or checksum are wrong")
    m = c.cmd("MIGRATE", "127.0.0.1", str(redis_port), "rk", "0", "1000", "REPLACE")
    check("MIGRATE into Redis reports its refusal", isinstance(m, RespError) and "Target instance replied with error" in m,
          repr(m))
    r.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6502)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion_dump_")
    s = Server(work, a.port, "src")
    redis = None
    try:
        c = s.start()
        run_roundtrip(c)
        run_ttl(c)
        c = run_durable(s, c)
        c = run_migrate(work, a.port, c)
        c.close()
        s.kill()
        c = s.start()
        check("a migrated key's deletion survives a SIGKILL of the source", c.cmd("EXISTS", "m1", "m5") == 0)
        rs = shutil.which("redis-server")
        if rs:
            sk = socket.socket()
            sk.bind(("127.0.0.1", 0))
            rport = sk.getsockname()[1]
            sk.close()
            redis = subprocess.Popen([rs, "--port", str(rport), "--save", "", "--appendonly", "no", "--dir", work],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait_ready_pid(rport, redis, 30)
            run_redis(c, rport)
        c.close()
    finally:
        s.stop()
        if redis:
            redis.send_signal(signal.SIGTERM)
            redis.wait()
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
