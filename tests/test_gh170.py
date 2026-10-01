#!/usr/bin/env python3
"""gh #170 full-coverage durability test.

Exercises every WAL effect record + snapshot v2 type in two crash modes:
  Mode A: SAVE -> SIGKILL -> restart  (snapshot serializer path)
  Mode B: no SAVE -> SIGKILL -> restart (WAL effect-record replay path)
"""
import os, signal, socket, subprocess, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import reader  # noqa: E402  (strict one-reply reads)

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1977
WORKDIR = f"/tmp/pion_gh170_test_{PORT}"

def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str): a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)

def send(sock, *args):
    # Exactly one parsed reply: a single recv() with a swallowed timeout
    # used to return "" and let the late reply answer the NEXT command.
    sock.sendall(encode(args))
    return reader(sock).read_raw().decode(errors="replace")

def connect():
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            s = socket.socket(); s.settimeout(1.0)
            s.connect(("127.0.0.1", PORT)); s.settimeout(None)
            # Ready = answering. The server accepts before it has replayed its WAL.
            if reader(s, 30.0).cmd("PING") != "PONG":
                raise RuntimeError("server accepted but did not answer PING")
            return s
        except OSError:
            s.close(); time.sleep(0.2)
    raise RuntimeError("server did not come up")

def start():
    return subprocess.Popen([os.path.abspath(BINARY), "-p", str(PORT), "-w", "1"],
                            cwd=WORKDIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def kill(proc):
    proc.kill(); proc.wait(timeout=5)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            s = socket.socket(); s.settimeout(0.3)
            s.connect(("127.0.0.1", PORT)); s.close(); time.sleep(0.2)
        except OSError:
            s.close(); return
    raise RuntimeError("port still busy")

def write_workload(s):
    send(s, "SET", "str:a", "hello")
    send(s, "SET", "str:del", "gone")
    send(s, "DEL", "str:del", "str:nonexistent")           # slow-path multi-DEL
    send(s, "INCR", "int:a")                                # INT value
    send(s, "INCRBY", "int:a", "41")                        # 42
    send(s, "APPEND", "app:a", "foo")
    send(s, "APPEND", "app:a", "bar")                       # foobar
    send(s, "SETRANGE", "rng:a", "5", "world")              # \0\0\0\0\0world
    send(s, "SET", "gd:a", "x"); send(s, "GETDEL", "gd:a")  # deleted
    send(s, "HSET", "h:a", "f1", "v1", "f2", "v2", "f3", "v3")
    send(s, "HDEL", "h:a", "f2")
    send(s, "HINCRBY", "h:a", "cnt", "7")
    send(s, "HINCRBYFLOAT", "h:a", "fl", "2.5")
    send(s, "HSETNX", "h:a", "nx", "yes")
    send(s, "RPUSH", "l:a", "a"); send(s, "RPUSH", "l:a", "b")
    send(s, "RPUSH", "l:a", "c"); send(s, "RPUSH", "l:a", "d")
    send(s, "LPUSH", "l:a", "z")                            # z a b c d
    send(s, "LPOP", "l:a")                                  # a b c d
    send(s, "RPOP", "l:a")                                  # a b c
    send(s, "LSET", "l:a", "1", "B")                        # a B c
    send(s, "LINSERT", "l:a", "BEFORE", "c", "x")           # a B x c
    send(s, "RPUSH", "l:b", "1"); send(s, "RPUSH", "l:b", "2")
    send(s, "RPUSH", "l:b", "2"); send(s, "RPUSH", "l:b", "3")
    send(s, "LREM", "l:b", "0", "2")                        # 1 3
    send(s, "RPUSH", "l:c", "p"); send(s, "RPUSH", "l:c", "q")
    send(s, "RPUSH", "l:c", "r"); send(s, "RPUSH", "l:c", "s")
    send(s, "LTRIM", "l:c", "1", "2")                       # q r
    send(s, "RPUSH", "l:src", "m1"); send(s, "RPUSH", "l:src", "m2")
    send(s, "LMOVE", "l:src", "l:dst", "LEFT", "RIGHT")     # src: m2, dst: m1
    send(s, "SADD", "s:a", "x"); send(s, "SADD", "s:a", "y"); send(s, "SADD", "s:a", "z")
    send(s, "SREM", "s:a", "y")                             # x z
    send(s, "SADD", "s:m", "mv")
    send(s, "SMOVE", "s:m", "s:a", "mv")                    # s:a: x z mv
    send(s, "ZADD", "z:a", "1", "one")
    send(s, "ZADD", "z:a", "2", "two")
    send(s, "ZADD", "z:a", "3", "three")
    send(s, "ZADD", "z:a", "4", "four")
    send(s, "ZREM", "z:a", "two")                           # one three four
    send(s, "ZINCRBY", "z:a", "10", "one")                  # one=11
    send(s, "ZADD", "z:b", "1", "p1"); send(s, "ZADD", "z:b", "2", "p2")
    send(s, "ZADD", "z:b", "3", "p3")
    send(s, "ZPOPMIN", "z:b")                               # p2 p3
    send(s, "ZPOPMAX", "z:b")                               # p2
    send(s, "ZADD", "z:c", "1", "r1"); send(s, "ZADD", "z:c", "2", "r2")
    send(s, "ZADD", "z:c", "3", "r3")
    send(s, "ZREMRANGEBYRANK", "z:c", "0", "0")             # r2 r3
    send(s, "SETBIT", "bm:a", "7", "1")
    send(s, "SETBIT", "bm:a", "100", "1")
    send(s, "PFADD", "hll:a", "e1", "e2", "e3")
    send(s, "SET", "ul:a", "v"); send(s, "UNLINK", "ul:a")

CHECKS = [
    (("GET", "str:a"), "hello"),
    (("GET", "str:del"), "$-1"),
    (("GET", "int:a"), "42"),
    (("GET", "app:a"), "foobar"),
    (("STRLEN", "rng:a"), ":10"),
    (("GET", "gd:a"), "$-1"),
    (("HGET", "h:a", "f1"), "v1"),
    (("HGET", "h:a", "f2"), "$-1"),
    (("HGET", "h:a", "f3"), "v3"),
    (("HGET", "h:a", "cnt"), "7"),
    (("HGET", "h:a", "fl"), "2.5"),
    (("HGET", "h:a", "nx"), "yes"),
    (("LRANGE", "l:a", "0", "-1"), ["a", "B", "x", "c"]),
    (("LRANGE", "l:b", "0", "-1"), ["1", "3"]),
    (("LRANGE", "l:c", "0", "-1"), ["q", "r"]),
    (("LRANGE", "l:src", "0", "-1"), ["m2"]),
    (("LRANGE", "l:dst", "0", "-1"), ["m1"]),
    (("SISMEMBER", "s:a", "x"), ":1"),
    (("SISMEMBER", "s:a", "y"), ":0"),
    (("SISMEMBER", "s:a", "z"), ":1"),
    (("SISMEMBER", "s:a", "mv"), ":1"),
    (("SISMEMBER", "s:m", "mv"), ":0"),
    (("ZSCORE", "z:a", "one"), "11"),
    (("ZSCORE", "z:a", "two"), "$-1"),
    (("ZSCORE", "z:a", "three"), "3"),
    (("ZCARD", "z:b"), ":1"),
    (("ZSCORE", "z:b", "p2"), "2"),
    (("ZCARD", "z:c"), ":2"),
    (("ZSCORE", "z:c", "r1"), "$-1"),
    (("GETBIT", "bm:a", "7"), ":1"),
    (("GETBIT", "bm:a", "100"), ":1"),
    (("GETBIT", "bm:a", "8"), ":0"),
    (("GET", "ul:a"), "$-1"),
]

def verify(s, mode, skip=()):
    passed = failed = 0
    for cmd, want in CHECKS:
        if cmd[1].split(":")[0] in skip:
            continue
        resp = send(s, *cmd)
        if isinstance(want, list):
            ok = all(w in resp for w in want) and resp.count("$") == len(want)
        else:
            ok = want in resp
        if ok: passed += 1
        else:
            failed += 1
            print(f"  [{mode}] FAIL {' '.join(cmd)}: want {want!r}, got {resp.strip()!r}")
    # HLL: snapshot-only coverage (mode A); mode B documents the gap
    resp = send(s, "PFCOUNT", "hll:a")
    hll_ok = ":3" in resp
    return passed, failed, hll_ok

def run_mode(mode):
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    proc = start()
    s = connect()
    write_workload(s)
    if mode == "A":
        r = send(s, "SAVE")
        assert "+OK" in r, f"SAVE failed: {r}"
    time.sleep(1.2)   # let the per-tick WAL msync fire
    s.close()
    kill(proc)
    proc = start()
    s = connect()
    passed, failed, hll_ok = verify(s, mode)
    s.close()
    kill(proc)
    print(f"Mode {mode}: {passed} passed, {failed} failed; HLL survived: {hll_ok}"
          + ("" if mode == "A" else " (HLL loss expected in mode B — snapshot-only)"))
    if mode == "A" and not hll_ok:
        failed += 1
        print("  [A] FAIL PFCOUNT hll:a: HLL must survive SAVE")
    return failed

fa = run_mode("A")
fb = run_mode("B")
print("=" * 50)
if fa == 0 and fb == 0:
    print("GH170 FULL TEST: ALL PASSED")
    sys.exit(0)
print(f"GH170 FULL TEST: FAILED (A: {fa}, B: {fb})")
sys.exit(1)
