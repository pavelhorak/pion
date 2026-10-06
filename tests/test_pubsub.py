#!/usr/bin/env python3
"""Pub/sub delivers every message whole, to every subscriber (#42).

  1. Large messages and long patterns: a 1 MB PUBLISH to a subscriber, and a
     5 KB pattern, arrive intact. (Delivery built each message in a fixed
     4 KB buffer with no bound: a 1 MB message crashed the worker.)
  2. No caps: 300 subscribers on one channel, 300 channels, 300 patterns all
     receive what they subscribed to. (Past 64 / 256 / 256 the subscription
     was acknowledged and nothing was delivered.)
  3. A subscriber that is slow to read loses nothing and never sees half a
     frame: messages wait in its output buffer; one that falls past the
     buffer's limit is disconnected, as Redis disconnects past its
     output-buffer limit. (send() was tried once, and the rest dropped.)
  4. RESP3: messages are pushes; a publisher subscribed to the channel gets
     its own message before PUBLISH's reply. A script's PUBLISH reaches
     subscribers.
  5. Shard channels: SSUBSCRIBE, SPUBLISH and SUNSUBSCRIBE deliver `smessage`
     and keep their own counts; PUBSUB SHARDCHANNELS / SHARDNUMSUB.
  6. Between workers (`-w 2 --independent-workers`): a message over 2 KB
     reaches a subscriber on the other worker. (Messages over 2 KB were
     dropped between workers.)
  7. The confirmations and PUBSUB replies match a real redis-server's, when
     one is on PATH.

    python3 tests/test_pubsub.py [--port 6494]
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
from resp_strict import Conn, RespError, encode, wait_ready_pid, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.abspath(os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def drain(c: Conn, quiet: float = 0.3) -> list:
    out = []
    while True:
        c.timeout = quiet
        try:
            out.append(c.read())
        except TimeoutError:
            c.timeout = 10.0
            return out


def start(port: int, work: str, extra=()) -> subprocess.Popen:
    p = subprocess.Popen([BIN, "-p", str(port), "-w", "1", "--no-crash-log", "--no-auto-detect",
                          "--no-auto-embed", *extra], cwd=work,
                         stdout=open(os.path.join(work, f"log{port}"), "a"), stderr=subprocess.STDOUT)
    wait_ready_pid(port, p, 60)
    return p


def stop(p: subprocess.Popen, ports):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    for q in ports:
        wait_port_free(q)


def sub_conn(port: int, *channels, cmd="SUBSCRIBE") -> Conn:
    c = Conn(port, timeout=20)
    c.sock.sendall(encode((cmd, *channels)))
    for _ in channels:
        c.read()
    return c


def run_large(port: int):
    print("[1] large messages, long patterns")
    s = sub_conn(port, "big")
    p = Conn(port)
    msg = os.urandom(1 << 20)
    check("PUBLISH of 1 MB answers :1", p.cmd("PUBLISH", "big", msg) == 1)
    got = s.read()
    check("...and the subscriber gets all of it", got == [b"message", b"big", msg],
          repr(got)[:120] if not isinstance(got, list) else f"len {len(got[2]) if len(got) > 2 else '?'}")
    pat = b"lp:" + b"?" * 5000 + b"*"
    ps = sub_conn(port, pat, cmd="PSUBSCRIBE")
    ch = b"lp:" + b"x" * 5000 + b"tail"
    check("a 5 KB pattern matches", p.cmd("PUBLISH", ch, b"m") == 1)
    got = ps.read()
    check("...and pmessage carries it", got == [b"pmessage", pat, ch, b"m"], repr(got)[:120])
    check("server alive", p.cmd("PING") == "PONG")
    for c in (s, ps, p):
        c.close()


def run_caps(port: int):
    print("[2] no caps")
    subs = [sub_conn(port, "many") for _ in range(300)]
    p = Conn(port)
    check("300 subscribers on one channel", p.cmd("PUBLISH", "many", "hi") == 300)
    ok = sum(1 for s in subs if s.read() == [b"message", b"many", b"hi"])
    check("...all 300 receive it", ok == 300, str(ok))
    for s in subs:
        s.close()
    chans = [f"ch{k}" for k in range(300)]
    one = sub_conn(port, *chans)
    check("300 channels on one connection", all(p.cmd("PUBLISH", c, c) == 1 for c in chans))
    got = [one.read() for _ in chans]
    check("...each delivered", got == [[b"message", c.encode(), c.encode()] for c in chans])
    one.close()
    pats = [f"pt{k}:*" for k in range(300)]
    pc = sub_conn(port, *pats, cmd="PSUBSCRIBE")
    check("300 patterns", p.cmd("PUBLISH", "pt299:x", "v") == 1)
    check("...the 300th matches", pc.read() == [b"pmessage", b"pt299:*", b"pt299:x", b"v"])
    pc.close()
    p.close()


def run_slow(port: int):
    print("[3] a slow subscriber")
    s = socket.create_connection(("127.0.0.1", port))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    s.sendall(encode(("SUBSCRIBE", "slow")))
    time.sleep(0.2)
    p = Conn(port)
    body = b"y" * 20000
    n = 100                                   # 2 MB: more than the socket holds, less than the limit
    for k in range(n):
        p.cmd("PUBLISH", "slow", body + str(k).encode())
    sc = Conn.wrap(s, timeout=20)
    first = sc.read()
    msgs = [sc.read() for _ in range(n)]
    check("subscribe confirmation, then every message, whole and in order",
          first[:2] == [b"subscribe", b"slow"] and msgs == [[b"message", b"slow", body + str(k).encode()] for k in range(n)],
          f"{len(msgs)} read")
    s.close()
    s2 = socket.create_connection(("127.0.0.1", port))
    s2.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    s2.sendall(encode(("SUBSCRIBE", "slow2")))
    time.sleep(0.2)
    big = b"z" * 100000
    n_big = 400                               # 40 MB: past the 32 MB a subscriber may be owed (#49)
    for k in range(n_big):
        p.cmd("PUBLISH", "slow2", big)
    sc2 = Conn.wrap(s2, timeout=20)
    frames, closed, bad = 0, False, False
    try:
        sc2.read()
        while True:
            m = sc2.read()
            if m != [b"message", b"slow2", big]:
                bad = True
                break
            frames += 1
    except ConnectionError:
        closed = True
    except Exception:                         # noqa: BLE001
        bad = True
    check("past the limit: disconnected after whole frames, never a cut one", closed and not bad and frames < n_big,
          f"closed={closed} bad={bad} frames={frames}")
    check("the server is fine", p.cmd("PING") == "PONG")
    s2.close()
    p.close()


def run_resp3(port: int):
    print("[4] RESP3, self-delivery, scripts")
    c = Conn(port)
    c.cmd("HELLO", "3")
    c.cmd("SUBSCRIBE", "r3")
    c.sock.sendall(encode(("PUBLISH", "r3", "own")))
    got = [c.read(), c.read()]
    check("own message before PUBLISH's reply", got == [[b"message", b"r3", b"own"], 1], repr(got))
    raw = Conn(port)
    raw.cmd("HELLO", "3")
    raw.sock.sendall(encode(("SUBSCRIBE", "r3b")))
    raw.read_raw()
    other = Conn(port)
    other.cmd("PUBLISH", "r3b", "x")
    check("a RESP3 subscriber gets a push", raw.read_raw() == b">3\r\n$7\r\nmessage\r\n$3\r\nr3b\r\n$1\r\nx\r\n")
    s = sub_conn(port, "fromlua")
    r = other.cmd("EVAL", "return redis.call('PUBLISH', 'fromlua', ARGV[1])", "0", "hello")
    check("a script's PUBLISH counts the subscriber", r == 1, repr(r))
    check("...and reaches it", s.read() == [b"message", b"fromlua", b"hello"])
    for x in (c, raw, other, s):
        x.close()


def run_shard(port: int):
    print("[5] shard channels")
    s = Conn(port)
    r = s.cmd("SSUBSCRIBE", "sh1", "sh2")
    check("SSUBSCRIBE confirms each, counting shard channels",
          r == [b"ssubscribe", b"sh1", 1] and s.read() == [b"ssubscribe", b"sh2", 2], repr(r))
    p = Conn(port)
    check("SPUBLISH counts the subscriber", p.cmd("SPUBLISH", "sh1", "m") == 1)
    check("...and delivers smessage", s.read() == [b"smessage", b"sh1", b"m"])
    check("PUBLISH does not reach a shard channel", p.cmd("PUBLISH", "sh1", "x") == 0)
    check("PUBSUB SHARDCHANNELS", sorted(p.cmd("PUBSUB", "SHARDCHANNELS")) == [b"sh1", b"sh2"])
    check("PUBSUB SHARDNUMSUB", p.cmd("PUBSUB", "SHARDNUMSUB", "sh1", "zz") == [b"sh1", 1, b"zz", 0])
    s.sock.sendall(encode(("SUNSUBSCRIBE",)))
    got = sorted(map(repr, [s.read(), s.read()]))
    check("SUNSUBSCRIBE leaves both", got == sorted(map(repr, [[b"sunsubscribe", b"sh1", 1], [b"sunsubscribe", b"sh2", 0]]))
          or got == sorted(map(repr, [[b"sunsubscribe", b"sh2", 1], [b"sunsubscribe", b"sh1", 0]])), repr(got))
    check("...after which SPUBLISH reaches nobody", p.cmd("SPUBLISH", "sh1", "m") == 0)
    s.close()
    p.close()


def run_workers(work: str, port: int):
    print("[6] between workers")
    proc = subprocess.Popen([BIN, "-p", str(port), "-w", "2", "--independent-workers", "--no-crash-log",
                             "--no-auto-detect", "--no-auto-embed"], cwd=work,
                            stdout=open(os.path.join(work, "logw"), "a"), stderr=subprocess.STDOUT)
    try:
        wait_ready_pid(port, proc, 60)
        time.sleep(0.5)
        s = sub_conn(port + 2, "xw")          # worker 0's affinity port
        p = Conn(port + 3)                    # worker 1's
        msg = b"w" * 50000
        p.cmd("PUBLISH", "xw", msg)
        got = s.read()
        check("a 50 KB message reaches a subscriber on the other worker", got == [b"message", b"xw", msg],
              repr(got)[:100])
        p.cmd("SPUBLISH", "xw", b"s")
        s.cmd("SSUBSCRIBE", "xs")
        p.cmd("SPUBLISH", "xs", b"shard")
        check("...and a shard message", s.read() == [b"smessage", b"xs", b"shard"])
        s.close()
        p.close()
    finally:
        stop(proc, (port, port + 2, port + 3))


SEQUENCE = [
    ("SUBSCRIBE", "a", "b"), ("PSUBSCRIBE", "p*"), ("PING",), ("UNSUBSCRIBE", "zz"),
    ("UNSUBSCRIBE",), ("PUNSUBSCRIBE",), ("PUNSUBSCRIBE",), ("UNSUBSCRIBE",),
    ("SSUBSCRIBE", "s1"), ("SUNSUBSCRIBE", "s1"), ("SUNSUBSCRIBE",),
    ("SUBSCRIBE",), ("PUBSUB", "NUMPAT"), ("PUBSUB", "CHANNELS"), ("PUBSUB", "NUMSUB", "a", "nope"),
    ("PUBSUB", "NUMSUB"), ("PUBSUB", "CHANNELS", "a*", "x"), ("PUBSUB", "NOPE"), ("PUBSUB",),
    ("PUBLISH", "a"), ("SPUBLISH", "a", "b", "c"), ("PUBSUB", "HELP"),
]


def replies(port: int, resp3: bool) -> list:
    c = Conn(port)
    if resp3:
        c.cmd("HELLO", "3")
    out = []
    for cmd in SEQUENCE:
        c.sock.sendall(encode(cmd))
        out.append((cmd, drain(c, 0.15)))
    c.close()
    return out


def unordered(rs):
    """UNSUBSCRIBE without arguments leaves the channels in hash order, which
    is not a contract: compare the names as a set and the counts in order."""
    if rs and all(isinstance(r, list) and len(r) == 3 for r in rs):
        return (sorted(repr(r[1]) for r in rs), [(r[0], r[2]) for r in rs])
    return rs


def run_compare(port: int, redis_port: int):
    print("[7] the same replies as redis-server")
    for resp3 in (False, True):
        a = [(cmd, unordered(r)) for cmd, r in replies(port, resp3)]
        b = [(cmd, unordered(r)) for cmd, r in replies(redis_port, resp3)]
        diffs = [(cmd, x, y) for (cmd, x), (_, y) in zip(a, b) if x != y]
        check(f"{'RESP3' if resp3 else 'RESP2'}: every reply", not diffs,
              "; ".join(f"{cmd}: pion {x!r} redis {y!r}" for cmd, x, y in diffs[:4]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6494)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion_pubsub_")
    proc = None
    redis = None
    try:
        proc = start(a.port, work)
        run_large(a.port)
        run_caps(a.port)
        run_slow(a.port)
        run_resp3(a.port)
        run_shard(a.port)
        rs = shutil.which("redis-server")
        if rs:
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            rp = s.getsockname()[1]
            s.close()
            redis = subprocess.Popen([rs, "--port", str(rp), "--save", "", "--appendonly", "no", "--dir", work],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait_ready_pid(rp, redis, 30)
            run_compare(a.port, rp)
        else:
            print("  (no redis-server on PATH: [7] skipped)")
        stop(proc, (a.port, a.port + 2))
        proc = None
        run_workers(work, a.port + 10)
    finally:
        if proc:
            stop(proc, (a.port, a.port + 2))
        if redis:
            redis.send_signal(signal.SIGTERM)
            redis.wait()
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
