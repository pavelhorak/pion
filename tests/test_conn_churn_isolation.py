#!/usr/bin/env python3
"""Connections that die mid-flight never affect the ones that replace them.

WHY
On io_uring a connection's RECV and SEND are in flight in the kernel while the
loop runs, and the kernel hands a new client the lowest free fd, so the next
connection usually gets the number of the one that just closed. The loop
closes in two phases (nothing is released while the kernel still owns it) and
stamps every completion with the connection's generation, so nothing in flight
for an old connection can touch a new one.

HOW
Many threads, each looping: open a connection, pipeline SET/GET pairs on keys
only it uses, and either read every reply (and check each is its OWN value)
or reset the connection mid-pipeline (SO_LINGER 0), leaving requests and
replies in flight. Under churn, fd numbers are reused constantly. Any reply
that is not the one its own command asked for is a leak.

    python3 tests/test_conn_churn_isolation.py [--port 1974] [--seconds 15]
"""
from __future__ import annotations

import argparse
import os
import random
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, encode  # noqa: E402

lock = threading.Lock()
stats = {"conns": 0, "aborted": 0, "checked": 0, "bad": 0, "errors": 0}
bad_examples: list = []


def rst_close(c):
    """Close with RST (SO_LINGER 0): no TIME_WAIT left on the client side, so
    the run does not exhaust the ephemeral ports (macOS has ~16k)."""
    try:
        c.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    except OSError:
        pass
    c.close()


def worker(port, tid, deadline, seed, budget):
    rnd = random.Random(seed)
    n = 0
    while time.monotonic() < deadline and n < budget:
        n += 1
        try:
            c = Conn(port, timeout=15)
        except OSError:
            with lock:
                stats["errors"] += 1
            continue
        with lock:
            stats["conns"] += 1
        pairs = rnd.randint(5, 60)
        cmds = []
        for i in range(pairs):
            key = f"churn:{tid}:{n}:{i}"
            val = f"{tid}-{n}-{i}-" + "x" * rnd.randint(0, 300)
            cmds.append(("SET", key, val))
            cmds.append(("GET", key))
        payload = b"".join(encode(cmd) for cmd in cmds)
        if rnd.random() < 0.4:
            # Die mid-flight: send part of the pipeline, then reset.
            try:
                c.sock.sendall(payload[: rnd.randint(1, len(payload))])
            except OSError:
                pass
            rst_close(c)
            with lock:
                stats["aborted"] += 1
            continue
        try:
            c.sock.sendall(payload)
            for j, cmd in enumerate(cmds):
                r = c.read()
                want = "OK" if cmd[0] == "SET" else cmds[j - 1][2].encode()
                with lock:
                    stats["checked"] += 1
                    if r != want:
                        stats["bad"] += 1
                        if len(bad_examples) < 5:
                            bad_examples.append((cmd[:2], repr(r)[:80], repr(want)[:40]))
        except (OSError, ConnectionError, TimeoutError) as e:
            with lock:
                stats["errors"] += 1
                if len(bad_examples) < 5:
                    bad_examples.append(("transport", repr(e)[:120], ""))
        # Half the clean ones close with FIN, half with RST: both close paths.
        if rnd.random() < 0.5:
            c.close()
        else:
            rst_close(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--connections", type=int, default=4000,
                    help="total budget; the FIN-closed ones sit in TIME_WAIT for ~30 s")
    args = ap.parse_args()
    deadline = time.monotonic() + args.seconds
    budget = max(1, args.connections // args.threads)
    ts = [threading.Thread(target=worker, args=(args.port, t, deadline, 1000 + t, budget))
          for t in range(args.threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"  {stats['conns']} connections, {stats['aborted']} reset mid-flight, "
          f"{stats['checked']} replies checked, {stats['bad']} wrong, {stats['errors']} transport errors")
    for ex in bad_examples:
        print(f"    e.g. {ex}")
    ok = True
    with Conn(args.port, timeout=10) as c:
        alive = c.cmd("PING") == "PONG"
    if not alive:
        print("  FAIL  the server stopped answering")
        ok = False
    if stats["bad"]:
        print("  FAIL  a reply that was not its own command's")
        ok = False
    if stats["errors"]:
        print("  FAIL  a connection that was not reset by its client broke")
        ok = False
    if stats["checked"] < 1000 or stats["aborted"] < 50:
        print("  FAIL  too little churn to mean anything")
        ok = False
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
