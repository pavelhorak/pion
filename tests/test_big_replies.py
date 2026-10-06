#!/usr/bin/env python3
"""#49: replies of any size arrive whole and in order, and a slow reader
holds up nobody else.

Until #49 the server built every reply in one 4 MB buffer per worker, and a
connection's unsent bytes waited in a second 4 MB block. A reply past either
was cut off: an array header, as many elements as fit, then
`-ERR response exceeds buffer` inside the array, and the client waited
forever for the rest. A redis-py pipeline of 20,000 GETs of 400-byte values
got 2,050 answers. A GET of a value over 3 MB took a blocking send loop
instead, which stalled every connection on the worker while one client read,
and wrote ahead of the replies already queued for that client.

Every reply here is parsed by resp_strict, which fails on a cut frame, a
surplus reply or a timeout, and each section ends with an in-sync PING.

    python3 tests/test_big_replies.py [--port 1974]
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, encode  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


def recv_exactly(sock: socket.socket, n: int, timeout: float) -> bytes:
    sock.settimeout(timeout)
    out = bytearray()
    while len(out) < n:
        try:
            d = sock.recv(min(1 << 20, n - len(out)))
        except socket.timeout:
            break
        if not d:
            break
        out += d
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    port = ap.parse_args().port
    c = Conn(port, timeout=120)
    keys = ["br:list", "br:hash", "br:blob", "br:cnt"] + [f"br:v{i}" for i in range(100)] \
        + [f"br:m{i}" for i in range(40)]
    c.cmd("DEL", *keys)

    print("[1] one reply larger than the buffer")
    n_list = 300_000                               # about 6 MB of reply
    for i in range(0, n_list, 2_000):       # within the old 2,048-argument cap (#52)
        c.cmd("RPUSH", "br:list", *[f"e-{j:09d}" for j in range(i, i + 2_000)])
    v = c.cmd("LRANGE", "br:list", 0, -1)
    check("LRANGE of 300K elements (about 6 MB)",
          isinstance(v, list) and len(v) == n_list and v[0] == b"e-000000000"
          and v[-1] == f"e-{n_list - 1:09d}".encode(), f"{len(v) if isinstance(v, list) else v!r}")
    n_hash = 150_000
    for i in range(0, n_hash, 1_000):
        args = []
        for j in range(i, i + 1_000):
            args += [f"f-{j:09d}", f"v-{j:09d}"]
        c.cmd("HSET", "br:hash", *args)
    h = c.cmd("HGETALL", "br:hash")
    hd = dict(zip(h[0::2], h[1::2])) if isinstance(h, list) else {}
    check("HGETALL of 150K fields, RESP2", isinstance(h, list) and len(h) == 2 * n_hash
          and len(hd) == n_hash and hd.get(b"f-000123456") == b"v-000123456",
          f"{len(h) if isinstance(h, list) else h!r}")
    c.assert_in_sync()

    print("[2] the same under RESP3")
    r3 = Conn(port, timeout=120)
    r3.cmd("HELLO", "3")
    m = r3.cmd("HGETALL", "br:hash")
    check("HGETALL of 150K fields, a RESP3 map", isinstance(m, dict) and len(m) == n_hash
          and m.get(b"f-000123456") == b"v-000123456")
    s3 = r3.cmd("LRANGE", "br:list", 0, -1)
    check("LRANGE under RESP3", isinstance(s3, list) and len(s3) == n_list)
    r3.assert_in_sync()
    r3.close()

    print("[3] a pipeline whose replies outgrow the connection's queue")
    val = b"v" * 400
    c.pipeline([("SET", f"br:v{i}", val) for i in range(100)])
    t0 = time.time()
    res = c.pipeline([("GET", f"br:v{i % 100}") for i in range(20_000)])   # one write, then read
    check("20,000 pipelined GETs of 400 B: every one answered",
          len(res) == 20_000 and all(x == val for x in res),
          f"{sum(1 for x in res if x == val)} right of {len(res)}")
    print(f"      ({time.time() - t0:.2f} s)")
    c.assert_in_sync()

    print("[4] a value larger than the buffer, in order with small replies")
    blob = bytes(range(256)) * (20 * 1024 * 1024 // 256)          # 20 MB
    check("SET 20 MB", c.cmd("SET", "br:blob", blob) == "OK")
    out = c.pipeline([("INCR", "br:cnt"), ("GET", "br:blob"), ("INCR", "br:cnt"), ("GET", "br:blob"), ("PING",)])
    check("INCR, GET 20 MB, INCR, GET 20 MB, PING: in order, whole",
          out[0] == 1 and out[1] == blob and out[2] == 2 and out[3] == blob and out[4] == "PONG",
          f"{[type(x).__name__ for x in out]}")
    mv = b"m" * 200_000
    c.pipeline([("SET", f"br:m{i}", mv) for i in range(40)])
    mg = c.cmd("MGET", *[f"br:m{i}" for i in range(40)])
    check("MGET of 40 values of 200 KB (8 MB)", isinstance(mg, list) and len(mg) == 40 and all(x == mv for x in mg))
    c.assert_in_sync()

    print("[5] scripts and transactions")
    e = c.cmd("EVAL", "return redis.call('LRANGE', KEYS[1], 0, -1)", 1, "br:list")
    check("EVAL returning the 300K-element list", isinstance(e, list) and len(e) == n_list
          and e[-1] == f"e-{n_list - 1:09d}".encode())
    eb = c.cmd("EVAL", "return redis.call('GET', KEYS[1])", 1, "br:blob")
    check("EVAL returning the 20 MB value", eb == blob)
    x = c.pipeline([("MULTI",), ("LRANGE", "br:list", 0, -1), ("GET", "br:blob"), ("PING",), ("EXEC",)])
    ex = x[-1]
    check("MULTI/EXEC with 26 MB of replies", isinstance(ex, list) and len(ex) == 3
          and len(ex[0]) == n_list and ex[1] == blob and ex[2] == "PONG")
    c.assert_in_sync()

    print("[6] a slow reader holds up nobody else")
    a = socket.create_connection(("127.0.0.1", port))
    a.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    a.sendall(encode(("GET", "br:blob")) * 5)                     # 100 MB owed, not read yet
    time.sleep(0.5)
    lat = []
    for _ in range(50):
        t1 = time.time()
        c.cmd("PING")
        lat.append(time.time() - t1)
    check("PING on another connection while 100 MB wait for a reader", max(lat) < 0.5,
          f"slowest {max(lat) * 1000:.0f} ms")
    one = b"$%d\r\n" % len(blob) + blob + b"\r\n"
    got = recv_exactly(a, 5 * len(one), 60)
    check("the slow reader then gets all 100 MB, byte for byte", got == one * 5, f"{len(got)} of {5 * len(one)}")
    a.close()
    b = socket.create_connection(("127.0.0.1", port))
    b.sendall(encode(("GET", "br:blob")) * 5)
    time.sleep(0.5)
    b.close()                                                     # leaves with output queued
    time.sleep(0.3)
    c.assert_in_sync()
    check("server serves after a client left with 100 MB queued", True)

    print("[7] a subscriber that does not read is cut off, the publisher is not")
    s = socket.create_connection(("127.0.0.1", port))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    s.sendall(encode(("SUBSCRIBE", "br:ch")))
    time.sleep(0.3)
    msg = b"x" * 1_000_000
    t2 = time.time()
    for _ in range(60):                                           # 60 MB, past the 32 MB limit
        c.cmd("PUBLISH", "br:ch", msg)
    pub_s = time.time() - t2
    s.settimeout(10)
    closed = False
    while True:
        try:
            d = s.recv(1 << 20)
        except socket.timeout:
            break
        if not d:
            closed = True
            break
    s.close()
    check("the subscriber was disconnected", closed)
    check("PUBLISH stayed fast", pub_s < 5.0, f"{pub_s:.1f} s for 60")
    c.assert_in_sync()

    c.cmd("DEL", *keys)
    c.close()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
