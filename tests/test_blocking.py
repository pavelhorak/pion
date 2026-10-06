#!/usr/bin/env python3
"""Blocking list and sorted-set commands block, and wake as in Redis (#38).

BLPOP and BRPOP used to answer nil at once whatever the timeout, so a worker
loop written for Redis spun; BLMOVE, BLMPOP, BRPOPLPUSH, BZPOPMIN, BZPOPMAX
and BZMPOP did not exist. tests/test_redis_differential.py compares their
immediate replies and errors with Redis. This covers what needs two
connections and a clock:

  1. Each command blocks on empty keys and is served when another connection
     pushes (a ZADD for the BZ forms), with its own reply shape.
  2. The timeout: nil after it (a null array, or nil for BLMOVE/BRPOPLPUSH),
     not before; 0 waits until served.
  3. Clients blocked on one key are served in the order they blocked.
  4. What a client pipelined behind its blocking command is answered after it.
  5. A client that disconnects while blocked takes nothing with it.
  6. Inside MULTI/EXEC the commands answer nil at once.
  7. A key that changes type while a client waits does not wake it.
  8. A served pop survives a restart (it is WAL-logged like any pop).

    python3 tests/test_blocking.py [--port 6490]
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready_pid, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.abspath(os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def blocked(port, cmd, out, key):
    """Run cmd on its own connection; record (reply, seconds it took)."""
    c = Conn(port, timeout=15)
    t0 = time.time()
    out[key] = (c.cmd(*cmd), time.time() - t0)
    c.close()


def wake(port, base, block_cmd, push_cmds, label, expect):
    out = {}
    t = threading.Thread(target=blocked, args=(port, block_cmd, out, "r"))
    t.start()
    time.sleep(0.3)
    check(f"{label}: still blocked before the push", t.is_alive())
    t_push = time.time()
    for pc in push_cmds:
        base.cmd(*pc)
    t.join(10)
    r = out.get("r", (None, 0))[0]
    check(f"{label}: served by the push", r == expect, repr(r))
    check(f"{label}: promptly", time.time() - t_push < 1.0, f"{time.time() - t_push:.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6490)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion_blocking_")
    port = a.port
    proc = None

    def start():
        nonlocal proc
        proc = subprocess.Popen([BIN, "-p", str(port), "-w", "1", "--no-crash-log", "--no-auto-detect",
                                 "--no-auto-embed"], cwd=work, stdout=open(os.path.join(work, "log"), "a"),
                                stderr=subprocess.STDOUT)
        wait_ready_pid(port, proc, 60)
        return Conn(port, timeout=15)

    def kill():
        nonlocal proc
        if proc:
            proc.send_signal(signal.SIGKILL)
            proc.wait()
            proc = None
            wait_port_free(port)

    try:
        base = start()
        base.cmd("FLUSHALL")

        print("[1] each command blocks, then is served by a push")
        wake(port, base, ("BLPOP", "b:l1", "b:l2", "5"), [("RPUSH", "b:l2", "x")], "BLPOP", [b"b:l2", b"x"])
        wake(port, base, ("BRPOP", "b:l1", "5"), [("RPUSH", "b:l1", "a", "b")], "BRPOP", [b"b:l1", b"b"])
        base.cmd("DEL", "b:l1")
        wake(port, base, ("BRPOPLPUSH", "b:src", "b:dst", "5"), [("LPUSH", "b:src", "m")], "BRPOPLPUSH", b"m")
        check("BRPOPLPUSH moved it", base.cmd("LRANGE", "b:dst", "0", "-1") == [b"m"])
        wake(port, base, ("BLMOVE", "b:src", "b:dst", "LEFT", "LEFT", "5"), [("RPUSH", "b:src", "n")], "BLMOVE", b"n")
        check("BLMOVE moved it", base.cmd("LRANGE", "b:dst", "0", "-1") == [b"n", b"m"])
        wake(port, base, ("BLMPOP", "5", "2", "b:m1", "b:m2", "RIGHT", "COUNT", "2"),
             [("RPUSH", "b:m2", "1", "2", "3")], "BLMPOP", [b"b:m2", [b"3", b"2"]])
        wake(port, base, ("BZPOPMIN", "b:z1", "b:z2", "5"), [("ZADD", "b:z2", "2", "two", "1", "one")],
             "BZPOPMIN", [b"b:z2", b"one", b"1"])
        base.cmd("DEL", "b:z2")                     # BZPOPMIN left "two" behind
        wake(port, base, ("BZPOPMAX", "b:z2", "5"), [("ZADD", "b:z2", "9", "nine", "8", "eight")], "BZPOPMAX",
             [b"b:z2", b"nine", b"9"])
        base.cmd("DEL", "b:z2")
        wake(port, base, ("BZMPOP", "5", "1", "b:z3", "MIN", "COUNT", "5"), [("ZADD", "b:z3", "1", "a", "2", "b")],
             "BZMPOP", [b"b:z3", [[b"a", b"1"], [b"b", b"2"]]])
        wake(port, base, ("BLPOP", "b:lp", "5"), [("LPUSHX", "b:lp", "no"), ("RPUSH", "b:lp", "yes")],
             "BLPOP woken by the RPUSH after an LPUSHX that did nothing", [b"b:lp", b"yes"])

        print("[2] timeouts")
        c = Conn(port, timeout=15)
        t0 = time.time()
        r = c.cmd("BLPOP", "b:none", "0.4")
        dt = time.time() - t0
        check("BLPOP times out with a null array", r is None, repr(r))
        check("...after the timeout, not before", 0.35 <= dt < 1.5, f"{dt:.2f}s")
        t0 = time.time()
        check("BLMOVE times out with nil", c.cmd("BLMOVE", "b:none", "b:d", "LEFT", "LEFT", "0.2") is None)
        check("BZMPOP times out with nil", c.cmd("BZMPOP", "0.2", "1", "b:none", "MIN") is None)
        check("timeouts are honoured in sequence", 0.35 <= time.time() - t0 < 2.0)
        c.cmd("HELLO", "3")
        check("RESP3: nil", c.cmd("BLPOP", "b:none", "0.1") is None and c.cmd("BRPOPLPUSH", "b:none", "b:d", "0.1") is None)
        c.close()
        out = {}
        t = threading.Thread(target=blocked, args=(port, ("BLPOP", "b:forever", "0"), out, "r"))
        t.start()
        time.sleep(1.2)
        check("timeout 0 still waiting after 1.2 s", t.is_alive())
        base.cmd("RPUSH", "b:forever", "done")
        t.join(5)
        check("...and served", out.get("r", (None,))[0] == [b"b:forever", b"done"])

        print("[3] FIFO, [4] pipelining behind a blocked command")
        out = {}
        ths = []
        for name in ("first", "second", "third"):
            th = threading.Thread(target=blocked, args=(port, ("BLPOP", "b:q", "5"), out, name))
            th.start()
            ths.append(th)
            time.sleep(0.15)
        base.cmd("RPUSH", "b:q", "1", "2", "3")
        for th in ths:
            th.join(5)
        check("served in the order they blocked",
              [out[n][0][1] for n in ("first", "second", "third")] == [b"1", b"2", b"3"], repr(out))
        c = Conn(port, timeout=15)
        c.sock.sendall(b"*3\r\n$5\r\nBLPOP\r\n$5\r\nb:pip\r\n$1\r\n5\r\n*3\r\n$3\r\nSET\r\n$5\r\nb:aft\r\n$1\r\n1\r\n"
                       b"*2\r\n$3\r\nGET\r\n$5\r\nb:aft\r\n")
        time.sleep(0.3)
        check("the command pipelined behind it has not run", base.cmd("EXISTS", "b:aft") == 0)
        base.cmd("RPUSH", "b:pip", "v")
        check("replies in order after the wake", [c.read(), c.read(), c.read()] == [[b"b:pip", b"v"], "OK", b"1"])
        c.close()

        print("[5] a disconnect while blocked")
        c = Conn(port, timeout=15)
        c.sock.sendall(b"*3\r\n$5\r\nBLPOP\r\n$6\r\nb:gone\r\n$1\r\n0\r\n")
        time.sleep(0.2)
        c.close()
        time.sleep(0.2)
        base.cmd("RPUSH", "b:gone", "kept")
        time.sleep(0.2)
        check("the pushed element is still there", base.cmd("LRANGE", "b:gone", "0", "-1") == [b"kept"])

        print("[6] MULTI/EXEC answers at once")
        r = base.pipeline([("MULTI",), ("BLPOP", "b:none", "0"), ("BZPOPMIN", "b:none", "0"),
                           ("BLMOVE", "b:none", "b:d", "LEFT", "LEFT", "0"), ("EXEC",)])
        check("all nil", r == ["OK", "QUEUED", "QUEUED", "QUEUED", [None, None, None]], repr(r))

        print("[7] a key that changes type does not wake a waiter")
        out = {}
        t = threading.Thread(target=blocked, args=(port, ("BLPOP", "b:tc", "5"), out, "r"))
        t.start()
        time.sleep(0.2)
        base.cmd("SET", "b:tc", "a string")
        time.sleep(0.3)
        check("still blocked over a string", t.is_alive())
        base.cmd("DEL", "b:tc")
        base.cmd("RPUSH", "b:tc", "v")
        t.join(5)
        check("served once it is a list again", out.get("r", (None,))[0] == [b"b:tc", b"v"])

        print("[8] a served pop is durable")
        base.cmd("RPUSH", "b:dur", "a", "b")
        out = {}
        t = threading.Thread(target=blocked, args=(port, ("BLMOVE", "b:dsrc", "b:ddst", "RIGHT", "LEFT", "5"), out, "r"))
        t.start()
        time.sleep(0.2)
        base.cmd("RPUSH", "b:dsrc", "1", "2")
        t.join(5)
        base.close()
        kill()
        base = start()
        check("after a restart the move is there",
              base.cmd("LRANGE", "b:dsrc", "0", "-1") == [b"1"] and base.cmd("LRANGE", "b:ddst", "0", "-1") == [b"2"],
              f"{base.cmd('LRANGE', 'b:dsrc', '0', '-1')!r} {base.cmd('LRANGE', 'b:ddst', '0', '-1')!r}")
        base.close()
    finally:
        kill()
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
