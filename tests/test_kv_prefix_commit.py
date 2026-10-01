#!/usr/bin/env python3
"""KV.PREFIX.COMMIT — a durability barrier over the keyspace WAL and the
V-store WAL (msync MS_SYNC + F_FULLFSYNC on macOS).

What a test on a shared machine can check (a power cut cannot be staged
here; that needs a physical power-cut or kernel-panic run on real hardware):
  1. +OK after keyspace and V-store writes; its latency (the event-loop stall
     it costs) is reported.
  2. -ERR under --no-wal: nothing is promised, so nothing is acknowledged.
  3. Pipelined COMMIT + PING -> exactly two replies, in order.
  4. Data written before a COMMIT is there after SIGKILL + restart.

Servers run in private temp dirs that are removed afterwards.

    python3 tests/test_kv_prefix_commit.py [--port 1994]
"""
import argparse
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class RESP:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.f = self.s.makefile("rb")

    def send(self, *parts):
        out = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            b = p if isinstance(p, bytes) else str(p).encode()
            out += [f"${len(b)}\r\n".encode(), b, b"\r\n"]
        self.s.sendall(b"".join(out))

    def read(self):
        line = self.f.readline()
        if line[:1] == b"$":
            n = int(line[1:-2])
            return None if n < 0 else self.f.read(n + 2)[:-2]
        return line[:-2]

    def call(self, *parts):
        self.send(*parts)
        return self.read()


def start(port, cwd, log, *extra):
    proc = subprocess.Popen([os.environ.get("PION_BIN") or os.path.join(ROOT, "pion-server"), "--kvcache", "-w", "1", "-p", str(port),
                             "--no-auto-detect", "--no-auto-embed", *extra],
                            cwd=cwd, stdout=log, stderr=log, preexec_fn=os.setsid)
    for _ in range(100):
        try:
            r = RESP(port)
            if r.call("PING") == b"+PONG":
                return proc, r
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("server did not start")


def kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1994)
    port = ap.parse_args().port
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + name)
        fails += 0 if ok else 1

    d = tempfile.mkdtemp(prefix="pion_commit_")
    log = open(os.path.join(d, "server.log"), "w")
    proc = None
    try:
        proc, r = start(port, d, log)
        rows = np.random.default_rng(2).standard_normal((64, 128)).astype(np.float16)
        check("register", r.call("KV.PREFIX.REGISTER", "cm", 128, "fp16") == b"+OK")
        check("storebatch", r.call("V.STOREBATCH", "cm_pk", 0, 0, 64, rows.tobytes(), "FMT", "F16") == b"+OK")
        check("keyspace SET", r.call("SET", "cm:journal", b"\x01\x02\xff") == b"+OK")
        check("KV.PREFIX.COMMIT -> +OK", r.call("KV.PREFIX.COMMIT") == b"+OK")

        lat = []
        for i in range(20):
            r.call("V.STOREBATCH", "cm_pk", 0, 64 + i * 16, 16, rows[:16].tobytes(), "FMT", "F16")
            r.call("SET", f"cm:k{i}", "v" * 100)
            t0 = time.perf_counter()
            r.call("KV.PREFIX.COMMIT")
            lat.append((time.perf_counter() - t0) * 1000)
        print(f"INFO barrier latency after a 16-row store + a SET: median {statistics.median(lat):.2f} ms, "
              f"max {max(lat):.2f} ms (the event-loop stall per COMMIT)")

        r.send("KV.PREFIX.COMMIT")
        r.send("PING")
        replies = [r.read(), r.read()]
        check(f"pipelined COMMIT + PING -> 2 replies {replies}", replies == [b"+OK", b"+PONG"])

        before = r.call("V.FETCH", "cm_pk", 0, "RANGE", 0, 64, "FMT", "NATIVE")
        check("COMMIT then SIGKILL", r.call("KV.PREFIX.COMMIT") == b"+OK")
        kill(proc)
        proc, r = start(port, d, log)
        check("V-store rows survive", r.call("V.FETCH", "cm_pk", 0, "RANGE", 0, 64, "FMT", "NATIVE") == before)
        check("keyspace value survives", r.call("GET", "cm:journal") == b"\x01\x02\xff")
        kill(proc)
        proc = None

        proc, r = start(port, d, log, "--no-wal")
        r.call("KV.PREFIX.REGISTER", "nw", 128, "fp16")
        check("--no-wal: KV.PREFIX.COMMIT refuses (-ERR)", (r.call("KV.PREFIX.COMMIT") or b"").startswith(b"-ERR"))
    finally:
        if proc is not None:
            kill(proc)
        log.close()
        if fails:
            print(f"server log kept in {d}")
        else:
            shutil.rmtree(d, ignore_errors=True)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
