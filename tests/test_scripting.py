#!/usr/bin/env python3
"""Scripts run the server's own commands, and FUNCTION libraries are durable (#36).

tests/test_redis_differential.py compares every script reply with Redis. This
covers what a differential cannot see:

  1. A script's writes survive a SIGKILL restart, a key it did not declare in
     KEYS[] included: each redis.call() logs its own WAL record.
  2. FUNCTION LOAD, DELETE and FLUSH survive a restart through the WAL and
     through a snapshot (SAVE), and reach a replica.
  3. --lua-time-limit stops a script that has not written; one that has
     written runs on. --lua-memory-limit caps the Lua heap.
  4. Commands pipelined behind an EVAL are answered after it, in order (the
     nested dispatch parses into its own token tables). EVAL inside
     MULTI/EXEC runs its calls instead of queueing them. WAIT and XREAD BLOCK
     answer at once inside a script.
  5. FUNCTION DUMP / RESTORE round-trip, with the APPEND / REPLACE / FLUSH
     policies, all or nothing.

    python3 tests/test_scripting.py [--port 6480]
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready_pid, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.abspath(os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
failures: list = []

LIB = ("#!lua name=durable\n"
       "redis.register_function('d_get', function(keys, args) return redis.call('GET', keys[1]) end)\n"
       "redis.register_function{function_name='d_count', callback=function(keys, args) return #args end,"
       " flags={'no-writes'}, description='counts its arguments'}\n")
LIB2 = ("#!lua name=second\n"
        "redis.register_function('s_one', function() return 1 end)\n")


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


class Server:
    def __init__(self, work, port, sub="node", extra=()):
        self.port, self.dir, self.extra, self.proc = port, os.path.join(work, sub), list(extra), None
        os.makedirs(self.dir, exist_ok=True)

    def start(self, extra=()):
        self.proc = subprocess.Popen([BIN, "-p", str(self.port), "-w", "1", "--no-crash-log",
                                      "--no-auto-detect", "--no-auto-embed"] + self.extra + list(extra),
                                     cwd=self.dir, stdout=open(os.path.join(self.dir, "log"), "a"),
                                     stderr=subprocess.STDOUT)
        wait_ready_pid(self.port, self.proc, 60)
        return Conn(self.port, timeout=30)

    def kill(self):
        if self.proc:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait()
            self.proc = None
            wait_port_free(self.port)


def function_names(c):
    out = []
    for lib in c.cmd("FUNCTION", "LIST"):
        fields = dict(zip(lib[0::2], lib[1::2]))
        for f in fields[b"functions"]:
            out.append(dict(zip(f[0::2], f[1::2]))[b"name"].decode())
    return sorted(out)


def durability(work, port):
    print("[1] a script's writes survive a restart, undeclared keys included")
    srv = Server(work, port, "dur")
    c = srv.start()
    c.cmd("FLUSHALL")
    c.cmd("FUNCTION", "FLUSH")
    check("EVAL writes a declared and an undeclared key",
          c.cmd("EVAL", "redis.call('SET', KEYS[1], 'a'); redis.call('RPUSH', 'undeclared:list', 'x', 'y');"
                        " redis.call('HSET', 'undeclared:hash', 'f', 'v'); return redis.call('EXPIRE', KEYS[1], 1000)",
                "1", "declared") == 1)
    check("FCALL writes through a function", c.cmd("FUNCTION", "LOAD", LIB) == b"durable")
    c.cmd("EVAL", "return redis.call('INCRBY', 'counter', 5)", "0")

    print("[2] FUNCTION LOAD / DELETE / FLUSH survive a restart (WAL, then snapshot)")
    c.cmd("FUNCTION", "LOAD", LIB2)
    c.cmd("FUNCTION", "DELETE", "second")
    c.close()
    srv.kill()
    c = srv.start()
    check("declared key back", c.cmd("GET", "declared") == b"a")
    check("its TTL back", 0 < c.cmd("TTL", "declared") <= 1000, repr(c.cmd("TTL", "declared")))
    check("undeclared list back", c.cmd("LRANGE", "undeclared:list", "0", "-1") == [b"x", b"y"])
    check("undeclared hash back", c.cmd("HGET", "undeclared:hash", "f") == b"v")
    check("counter back", c.cmd("GET", "counter") == b"5")
    check("library back after a WAL restart, the deleted one gone",
          function_names(c) == ["d_count", "d_get"], repr(function_names(c)))
    check("its flags and description too",
          b"counts its arguments" in repr(c.cmd("FUNCTION", "LIST")).encode()
          and b"no-writes" in repr(c.cmd("FUNCTION", "LIST")).encode())
    check("FCALL works after the restart", c.cmd("FCALL", "d_get", "1", "declared") == b"a")
    # snapshot: SAVE, then a change only the WAL holds, then restart
    check("SAVE", c.cmd("SAVE") == "OK")
    c.cmd("FUNCTION", "LOAD", LIB2)
    c.close()
    srv.kill()
    c = srv.start()
    check("libraries back from snapshot + WAL", function_names(c) == ["d_count", "d_get", "s_one"],
          repr(function_names(c)))
    c.cmd("FUNCTION", "FLUSH")
    c.cmd("SAVE")
    c.cmd("FUNCTION", "LOAD", LIB2)
    c.cmd("FUNCTION", "FLUSH", "ASYNC")
    c.close()
    srv.kill()
    c = srv.start()
    check("a FUNCTION FLUSH survives", function_names(c) == [], repr(function_names(c)))
    c.close()
    srv.kill()


def replica(work, port):
    print("[2b] FUNCTION changes and script writes reach a replica")
    pa, pb = port + 1, port + 21
    prim = Server(work, pa, "primary", ["--cluster", "--cluster-host", "127.0.0.1"])
    rep = Server(work, pb, "replica", ["--cluster", "--cluster-host", "127.0.0.1", "--cluster-replica",
                                       "--cluster-primary-host", "127.0.0.1", "--cluster-primary-port", str(pa)])
    try:
        a = prim.start()
        a.cmd("FUNCTION", "LOAD", LIB)              # before the replica: via the FULLRESYNC image
        b = rep.start()
        b.cmd("READONLY")
        a.cmd("FUNCTION", "LOAD", LIB2)             # after: via the stream
        a.cmd("EVAL", "redis.call('SET', 'r:k', 'v'); return redis.call('SADD', 'r:s', 'm')", "0")
        a.cmd("FUNCTION", "DELETE", "durable")
        deadline = time.time() + 20
        got = None
        while time.time() < deadline:
            got = (function_names(b), b.cmd("GET", "r:k"), b.cmd("SMEMBERS", "r:s"))
            if got == (["s_one"], b"v", [b"m"]):
                break
            time.sleep(0.3)
        check("the replica holds the primary's libraries and the script's writes",
              got == (["s_one"], b"v", [b"m"]), repr(got))
    finally:
        rep.kill()
        prim.kill()


def limits(work, port):
    print("[3] --lua-time-limit and --lua-memory-limit")
    srv = Server(work, port + 2, "limits")
    c = srv.start(["--lua-time-limit", "300", "--lua-memory-limit", "4mb"])
    t0 = time.time()
    r = c.cmd("EVAL", "while true do end", "0")
    dt = time.time() - t0
    check("a non-writing loop is stopped", isinstance(r, str) and "lua-time-limit (300 ms)" in r, repr(r))
    check("...at the limit", 0.25 < dt < 3.0, f"{dt:.2f}s")
    check("the server answers after it", c.cmd("PING") == "PONG")
    t0 = time.time()
    r = c.cmd("EVAL", "redis.call('SET', 'w', '1'); local t = os.clock(); while os.clock() - t < 0.6 do end;"
                      " return redis.call('GET', 'w')", "0")
    check("a script that has written runs past the limit", r == b"1" and time.time() - t0 >= 0.55, repr(r))
    r = c.cmd("EVAL", "local s = string.rep('x', 8 * 1024 * 1024); return #s", "0")
    check("the memory cap stops a large allocation", isinstance(r, str) and "memory" in r, repr(r))
    check("and the next script runs", c.cmd("EVAL", "return 42", "0") == 42)
    c.close()
    srv.kill()


def dispatch(work, port):
    print("[4] pipelining around EVAL, EVAL inside MULTI, WAIT and XREAD BLOCK inside a script")
    srv = Server(work, port + 3, "dispatch")
    c = srv.start()
    c.cmd("FLUSHALL")
    replies = c.pipeline([("SET", "p:a", "1"),
                          ("EVAL", "redis.call('INCR', 'p:a'); redis.call('SET', 'p:b', 'x');"
                                   " return redis.call('MGET', 'p:a', 'p:b')", "0"),
                          ("GET", "p:a"), ("GET", "p:b"),
                          ("EVAL", "return redis.call('LPUSH', 'p:l', 'a', 'b', 'c')", "0"),
                          ("LRANGE", "p:l", "0", "-1"), ("PING",)])
    check("pipelined replies in order", replies == ["OK", [b"2", b"x"], b"2", b"x", 3, [b"c", b"b", b"a"], "PONG"],
          repr(replies))
    replies = c.pipeline([("MULTI",), ("SET", "m:a", "1"),
                          ("EVAL", "return redis.call('INCR', 'm:a')", "0"),
                          ("EVAL", "return redis.call('GET', 'm:a')", "0"), ("EXEC",)])
    check("EVAL in MULTI runs its calls at EXEC",
          replies == ["OK", "QUEUED", "QUEUED", "QUEUED", ["OK", 2, b"2"]], repr(replies))
    t0 = time.time()
    r = c.cmd("EVAL", "return {redis.call('WAIT', '1', '2000'),"
                      " redis.call('XREAD', 'BLOCK', '2000', 'STREAMS', 'nostream', '$')}", "0")
    check("WAIT and XREAD BLOCK answer at once", time.time() - t0 < 1.0, f"{time.time() - t0:.2f}s {r!r}")
    big = "x" * (300 * 1024)
    c.cmd("SET", "big", big)
    check("a large value through redis.call", c.cmd("EVAL", "return #redis.call('GET', 'big')", "0") == len(big))
    check("and as the script's reply", c.cmd("EVAL", "return redis.call('GET', 'big')", "0") == big.encode())
    check("a script that returns a big table",
          len(c.cmd("EVAL", "local t = {} for i = 1, 20000 do t[i] = i end return t", "0")) == 20000)
    c.close()
    srv.kill()


def dump_restore(work, port):
    print("[5] FUNCTION DUMP / RESTORE")
    srv = Server(work, port + 4, "dump")
    c = srv.start()
    c.cmd("FUNCTION", "FLUSH")
    c.cmd("FUNCTION", "LOAD", LIB)
    c.cmd("FUNCTION", "LOAD", LIB2)
    payload = c.cmd("FUNCTION", "DUMP")
    check("DUMP is a bulk payload", isinstance(payload, bytes) and len(payload) > 0)
    check("RESTORE APPEND onto the same libraries refuses",
          isinstance(c.cmd("FUNCTION", "RESTORE", payload), str)
          and "already exists" in c.cmd("FUNCTION", "RESTORE", payload))
    check("...and changed nothing", function_names(c) == ["d_count", "d_get", "s_one"])
    check("RESTORE REPLACE", c.cmd("FUNCTION", "RESTORE", payload, "REPLACE") == "OK")
    c.cmd("FUNCTION", "FLUSH")
    check("RESTORE onto an empty server", c.cmd("FUNCTION", "RESTORE", payload) == "OK"
          and function_names(c) == ["d_count", "d_get", "s_one"])
    c.cmd("FUNCTION", "DELETE", "second")
    c.cmd("FUNCTION", "LOAD", "#!lua name=other\nredis.register_function('o_one', function() return 1 end)")
    check("RESTORE FLUSH drops what was there", c.cmd("FUNCTION", "RESTORE", payload, "FLUSH") == "OK"
          and function_names(c) == ["d_count", "d_get", "s_one"], repr(function_names(c)))
    check("a bad payload is refused", c.cmd("FUNCTION", "RESTORE", b"nonsense")
          == "ERR DUMP payload version or checksum are wrong")
    clash = c.cmd("FUNCTION", "DUMP")
    c.cmd("FUNCTION", "FLUSH")
    c.cmd("FUNCTION", "LOAD", "#!lua name=thief\nredis.register_function('s_one', function() return 9 end)")
    r = c.cmd("FUNCTION", "RESTORE", clash)
    check("a restore that collides on a function name fails whole",
          isinstance(r, str) and function_names(c) == ["s_one"] and c.cmd("FCALL", "s_one", "0") == 9,
          f"{r!r} {function_names(c)!r}")
    c.cmd("FUNCTION", "FLUSH")
    c.cmd("FUNCTION", "RESTORE", clash)
    c.close()
    srv.kill()
    c = srv.start()
    check("a RESTORE survives a restart", function_names(c) == ["d_count", "d_get", "s_one"],
          repr(function_names(c)))
    c.close()
    srv.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6480)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion_scripting_")
    try:
        durability(work, a.port)
        replica(work, a.port)
        limits(work, a.port)
        dispatch(work, a.port)
        dump_restore(work, a.port)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
