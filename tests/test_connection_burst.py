#!/usr/bin/env python3
"""Hundreds of connections sending at the same instant all get their replies.

WHY
The io_uring loop queues its submissions (one SEND per answered connection,
plus a buffer hand-back per received chunk, plus re-armed RECVs) while it
drains completions, and hands them to the kernel once per pass. The ring has
1024 slots, and a pass that answers ~500 connections at once needs more. A
full ring is flushed to the kernel before another slot is taken, and a slot is
published to the kernel only once it is filled.

HOW
Open N connections, release them through one barrier so their pipelines land
in the same instant, and require every connection to get every reply, its own,
in time.

    python3 tests/test_connection_burst.py [--port 1974] [--conns 600]
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, encode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    ap.add_argument("--conns", type=int, default=600)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = args.conns + 64
    if soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(want, hard), hard))
        except (ValueError, OSError):
            pass
    n = min(args.conns, resource.getrlimit(resource.RLIMIT_NOFILE)[0] - 64)
    if n < 300:
        print(f"SKIP: open-file limit allows only {n} connections")
        return 0
    conns = [Conn(args.port, timeout=30) for _ in range(n)]
    problems: list = []
    lock = threading.Lock()
    for rnd in range(args.rounds):
        barrier = threading.Barrier(n)

        def go(i):
            c = conns[i]
            cmds = [("SET", f"burst:{rnd}:{i}:{j}", f"v{rnd}-{i}-{j}") for j in range(4)]
            cmds += [("GET", f"burst:{rnd}:{i}:{j}") for j in range(4)]
            payload = b"".join(encode(x) for x in cmds)
            try:
                barrier.wait(timeout=60)
                c.sock.sendall(payload)
                got = [c.read() for _ in cmds]
            except Exception as e:  # noqa: BLE001
                with lock:
                    problems.append(f"conn {i}: {e!r}"[:160])
                return
            want = ["OK"] * 4 + [f"v{rnd}-{i}-{j}".encode() for j in range(4)]
            if got != want:
                with lock:
                    problems.append(f"conn {i}: got {got!r}"[:160])
        ts = [threading.Thread(target=go, args=(i,)) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        print(f"  round {rnd + 1}: {n} connections, {len(problems)} problems so far")
    for c in conns:
        c.close()
    with Conn(args.port, timeout=10) as c:
        alive = c.cmd("PING") == "PONG"
    for p in problems[:8]:
        print(f"    {p}")
    ok = alive and not problems
    print("ALL PASS" if ok else f"FAILED ({len(problems)} connections without their replies"
          + ("" if alive else ", server not answering") + ")")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
