#!/usr/bin/env python3
"""A request larger than the client buffer closes ITS connection, and only it.

WHY
Each connection buffers up to 256 MB of an unfinished request. What happened
when a client kept sending past that differed per event loop: kqueue closed
the connection; epoll stopped reading and `continue`d, and since
level-triggered epoll reports the fd again at once, the worker spun at 100%
CPU for good; io_uring's multishot path did not close. All three now close
it. This sends a bulk header that promises 300 MB and streams past 256 MB;
the server must drop that client and keep serving.

    python3 tests/test_client_buffer_overflow.py [--port 1974]
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn  # noqa: E402

failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    args = ap.parse_args()

    bystander = Conn(args.port, timeout=10)
    check("bystander connection works", bystander.cmd("SET", "bystander", "ok") == "OK")

    print("[1] stream an unfinished 300 MB request")
    s = socket.create_connection(("127.0.0.1", args.port), timeout=30)
    s.sendall(b"*3\r\n$3\r\nSET\r\n$3\r\nbig\r\n$300000000\r\n")
    chunk = b"x" * (1 << 20)
    sent, closed, t0 = 0, False, time.monotonic()
    while sent < 300_000_000 and time.monotonic() - t0 < 120:
        try:
            s.sendall(chunk)
            sent += len(chunk)
        except OSError:
            closed = True
            break
    if not closed:
        # The kernel may have taken everything into socket buffers before the
        # server reacted; the close then shows up on read.
        s.settimeout(30)
        try:
            closed = s.recv(1) == b""
        except OSError:
            closed = True
    s.close()
    check("the server closed the oversized request's connection", closed,
          f"sent {sent >> 20} MB without the connection closing")
    check("...after it passed 256 MB, not before", sent >= 200 << 20, f"closed after {sent >> 20} MB")

    print("[2] everyone else is still served")
    try:
        ok = bystander.cmd("GET", "bystander") == b"ok"
    except (OSError, ConnectionError, TimeoutError) as e:
        ok = False
        print(f"    {e!r}")
    check("the bystander connection still answers", ok)
    with Conn(args.port, timeout=10) as c:
        check("a new connection is served", c.cmd("PING") == "PONG")
        check("the oversized request was not applied", c.cmd("EXISTS", "big") == 0)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
