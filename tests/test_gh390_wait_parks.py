#!/usr/bin/env python3
"""WAIT parks its client; it does not block the worker (gh #390).

WHY
WAIT used to poll for replica ACKs inside the event loop — `usleep` in a loop
until the count was met or the timeout passed. The worker serves every
connection it accepted, so one `WAIT 2 1000` froze every other client of that
worker for a second, and a timeout of 0 (Redis: wait for as long as it takes)
had to be answered at once to avoid hanging the worker forever.

Now the connection is parked: WAIT writes no reply, the worker keeps serving,
and the event loop answers the WAIT once enough replicas ACK or its timeout
passes. Nothing the client pipelined behind WAIT runs before that answer.

CASES (primary + one replica, one worker each)
  1. another client's PINGs are answered promptly while a WAIT is parked,
     and the parked WAIT still answers 1 after its timeout.
  2. order: `SET k before | WAIT 2 800 | SET k after | GET k` in ONE write —
     the SET behind WAIT has not run 300 ms in (another client reads
     "before"), and the four replies arrive in order.
  3. a command sent SEPARATELY while parked is answered after the WAIT.
  4. `WAIT 1 0` with a caught-up replica answers 1; `WAIT 2 0` does not
     answer at all (Redis semantics), and closing that connection leaves the
     server healthy — the next connection on the same fd number works.
  5. inside MULTI/EXEC WAIT does not park (Redis's deny-blocking rule).

    python3 tests/test_gh390_wait_parks.py [--port 2471]
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, encode, wait_ready, wait_port_free  # noqa: E402

BIN = os.path.abspath(os.environ.get("PION_BIN", "./pion-server"))
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


class Node:
    def __init__(self, work, port, sub, extra):
        self.port, self.dir, self.extra = port, os.path.join(work, sub), extra
        os.makedirs(self.dir, exist_ok=True)
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen([BIN, "-p", str(self.port), "-w", "1", "--no-crash-log",
                                      "--no-auto-detect", "--no-auto-embed", "--cluster",
                                      "--cluster-host", "127.0.0.1"] + self.extra, cwd=self.dir,
                                     stdout=open(os.path.join(self.dir, "log"), "a"), stderr=subprocess.STDOUT)
        wait_ready(self.port, 30, proc=self.proc)
        return Conn(self.port, timeout=30)

    def stop(self):
        if self.proc:
            self.proc.kill(); self.proc.wait(); self.proc = None
            wait_port_free(self.port)


def timed_replies(conn, n, t0):
    """Read n replies; return [(reply, seconds since t0), ...]."""
    out = []
    for _ in range(n):
        r = conn.read()
        out.append((r, time.time() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2471)
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion-gh390-wait-")
    pa, pb = args.port, args.port + 20
    primary = Node(work, pa, "primary", [])
    replica = Node(work, pb, "replica", ["--cluster-replica", "--cluster-primary-host", "127.0.0.1",
                                         "--cluster-primary-port", str(pa)])
    try:
        a = primary.start()
        b = replica.start()
        b.cmd("READONLY")
        a.cmd("SET", "{w}warm", "1")
        check("the replica is caught up (WAIT 1 5000 answers 1)", a.cmd("WAIT", "1", "5000") == 1)

        # ── 1. the worker keeps serving while a WAIT is parked ───────────────
        print("=== 1. other clients are served while a WAIT is parked ===")
        w = Conn(pa, timeout=10)
        other = Conn(pa, timeout=10)
        t_send = time.time()
        w.sock.sendall(encode(["WAIT", "2", "1500"]))
        time.sleep(0.1)
        lat = []
        for _ in range(20):
            t = time.time()
            r = other.cmd("PING")
            lat.append(time.time() - t)
            if r != "PONG":
                break
        check("20 PINGs from another client answered while the WAIT is parked",
              len(lat) == 20 and max(lat) < 0.1, f"max {max(lat):.3f} s over {len(lat)}")
        check("... all of them before the WAIT's timeout", sum(lat) < 1.0, f"{sum(lat):.2f} s")
        r = w.read()
        dt = time.time() - t_send
        check("the parked WAIT 2 1500 answers 1 (one replica)", r == 1, repr(r))
        check("... after ~1.5 s", 1.3 <= dt < 2.3, f"{dt:.2f} s")

        # ── 2. nothing pipelined behind WAIT runs before its answer ──────────
        print("=== 2. order: SET | WAIT | SET | GET in one write ===")
        w.sock.sendall(encode(["SET", "{w}k", "before"]) + encode(["WAIT", "2", "800"])
                       + encode(["SET", "{w}k", "after"]) + encode(["GET", "{w}k"]))
        t0 = time.time()
        time.sleep(0.3)
        seen = other.cmd("GET", "{w}k")
        check("300 ms in, the SET behind the parked WAIT has not run", seen == b"before", repr(seen))
        reps = timed_replies(w, 4, t0)
        check("replies in order: OK, 1, OK, 'after'",
              [x[0] for x in reps] == ["OK", 1, "OK", b"after"], repr([x[0] for x in reps]))
        check("the WAIT's reply came after ~800 ms", 0.7 <= reps[1][1] < 1.5, f"{reps[1][1]:.2f} s")

        # ── 3. a command sent separately while parked is answered after ──────
        print("=== 3. a command sent while parked waits its turn ===")
        t0 = time.time()
        w.sock.sendall(encode(["WAIT", "2", "600"]))
        time.sleep(0.2)
        w.sock.sendall(encode(["PING"]))
        reps = timed_replies(w, 2, t0)
        check("replies: 1 then PONG", [x[0] for x in reps] == [1, "PONG"], repr([x[0] for x in reps]))
        check("PONG did not overtake the WAIT (both at ~600 ms)",
              reps[0][1] >= 0.5 and reps[1][1] >= reps[0][1], f"{reps[0][1]:.2f} / {reps[1][1]:.2f} s")

        # ── 4. timeout 0 ─────────────────────────────────────────────────────
        print("=== 4. WAIT n 0 ===")
        a.cmd("SET", "{w}z", "1")
        t0 = time.time(); r = a.cmd("WAIT", "1", "0"); dt = time.time() - t0
        check("WAIT 1 0 with a caught-up replica answers 1, promptly", r == 1 and dt < 1.0, f"{r!r} in {dt:.2f} s")
        forever = socket.create_connection(("127.0.0.1", pa), timeout=5)
        forever.sendall(encode(["WAIT", "2", "0"]))
        forever.settimeout(0.8)
        try:
            got = forever.recv(64)
        except socket.timeout:
            got = None
        check("WAIT 2 0 with one replica does not answer (it waits, as in Redis)", got is None, repr(got))
        forever.close()
        time.sleep(0.2)
        check("the server serves after a parked client disconnects", other.cmd("PING") == "PONG")
        fresh = Conn(pa, timeout=5)
        check("a new connection works (the closed fd's WAIT is gone)",
              fresh.cmd("SET", "{w}fresh", "1") == "OK" and fresh.cmd("GET", "{w}fresh") == b"1")
        t0 = time.time(); r = fresh.cmd("WAIT", "1", "2000"); dt = time.time() - t0
        check("... and its WAIT is not confused with the dropped one", r == 1 and dt < 1.0, f"{r!r} in {dt:.2f} s")

        # ── 5. MULTI/EXEC never parks ────────────────────────────────────────
        print("=== 5. WAIT inside MULTI/EXEC ===")
        t0 = time.time()
        reps = w.pipeline([("MULTI",), ("WAIT", "2", "2000"), ("EXEC",)])
        dt = time.time() - t0
        check("MULTI / WAIT 2 2000 / EXEC answers at once with the current count",
              reps == ["OK", "QUEUED", [1]] and dt < 0.5, f"{reps!r} in {dt:.2f} s")
        w.assert_in_sync()
        other.assert_in_sync()
    finally:
        replica.stop()
        primary.stop()
        shutil.rmtree(work, ignore_errors=True)
    print(f"{len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
