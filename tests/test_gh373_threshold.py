#!/usr/bin/env python3
"""gh #373 — an explicit THRESHOLD must be honoured, including 0 and negatives.

The similarity lookups resolved `override if override > 0 else default`, and
their hand-rolled parsers skipped every byte that was not a digit or a dot. So:

    THRESHOLD 0      -> "unset" -> the default (0.95 here) -> a miss
    THRESHOLD -0.5   -> parsed as 0.5 (the '-' skipped)    -> a miss
    THRESHOLD abc    -> parsed as 0.0 -> "unset" -> the default, no error

KV.FETCH takes the embedding itself, so the cosine between query and entry is
chosen by construction here — no model, no sidecar, deterministic. The same
resolver (`resolve_threshold`) and the same strict parse now serve
AI.SEMANTIC_CACHE GET and AI.COMPLETE (their end-to-end checks live in
tests/test_ai_gateway.py, which needs an embedding backend).

Usage: python3 tests/test_gh373_threshold.py [./pion-server]
"""
import math
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6473
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


class Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=20)
        self.f = self.s.makefile("rb")

    def cmd(self, *args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            a = a if isinstance(a, bytes) else str(a).encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.s.sendall(out)
        return self._read()

    def _read(self):
        line = self.f.readline()
        t, body = line[:1], line[1:-2]
        if t == b"*":
            n = int(body)
            return None if n < 0 else [self._read() for _ in range(n)]
        if t == b"$":
            n = int(body)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if t == b"-":
            return RuntimeError(body.decode(errors="replace"))
        if t == b":":
            return int(body)
        return body


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def pack(v):
    return struct.pack(f"<{len(v)}f", *v)


def main():
    d = tempfile.mkdtemp(prefix="pion_gh373_")
    p = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--kvcache",
                          "--no-auto-detect", "--no-auto-embed"],
                         cwd=d, stdout=open(os.path.join(d, "log"), "w"), stderr=subprocess.STDOUT)
    try:
        c = None
        for _ in range(200):
            try:
                c = Client(PORT)
                if c.cmd("PING") == b"PONG":
                    break
            except OSError:
                time.sleep(0.1)
        if c is None:
            raise SystemExit("server did not start")
        info = c.cmd("KV.INFO")
        text = info.decode() if isinstance(info, bytes) else str(info)
        dim = int([l.split(":")[1] for l in text.splitlines() if l.startswith("dimensions:")][0])
        print(f"[gh #373] {BINARY}  kvcache dim={dim}")

        rnd = random.Random(373)
        v = unit([rnd.gauss(0, 1) for _ in range(dim)])
        w = [rnd.gauss(0, 1) for _ in range(dim)]
        dot = sum(a * b for a, b in zip(v, w))
        w = unit([b - dot * a for a, b in zip(v, w)])          # w ⟂ v
        u = unit([0.3 * a + math.sqrt(1 - 0.09) * b for a, b in zip(v, w)])  # cos(u, v) = 0.3
        opp = [-a for a in v]                                   # cos = -1

        check("KV.STORE", c.cmd("KV.STORE", "e1", pack(v), b"payload-1") == b"OK")
        check("exact query hits with the default threshold",
              c.cmd("KV.FETCH", pack(v)) == b"payload-1")
        check("cos 0.3 misses with the default threshold",
              c.cmd("KV.FETCH", pack(u)) is None)
        r = c.cmd("KV.FETCH", pack(u), "THRESHOLD", "0")
        check("THRESHOLD 0 is honoured (cos 0.3 hits)", r == b"payload-1", repr(r)[:80])
        r = c.cmd("KV.FETCH", pack(u), "THRESHOLD", "0.2")
        check("THRESHOLD 0.2 hits cos 0.3", r == b"payload-1", repr(r)[:80])
        r = c.cmd("KV.FETCH", pack(u), "THRESHOLD", "0.5")
        check("THRESHOLD 0.5 misses cos 0.3", r is None, repr(r)[:80])
        r = c.cmd("KV.FETCH", pack(w), "THRESHOLD", "-0.5")
        check("THRESHOLD -0.5 hits an orthogonal query (the '-' is not dropped)",
              r == b"payload-1", repr(r)[:80])
        r = c.cmd("KV.FETCH", pack(opp), "THRESHOLD", "-0.5")
        check("THRESHOLD -0.5 misses the opposite vector", r is None, repr(r)[:80])
        for bad in ("abc", "", "0.5x", "--1"):
            r = c.cmd("KV.FETCH", pack(v), "THRESHOLD", bad)
            check(f"THRESHOLD {bad!r} is an error", isinstance(r, RuntimeError), repr(r)[:80])
        # The error must not desync the connection (the reply count is one).
        check("connection still in step after the errors", c.cmd("PING") == b"PONG")
        # Pipelined: the optional-argument scan is bounded by its own command.
        c.s.sendall(b"".join(
            b"*%d\r\n" % len(a) + b"".join(b"$%d\r\n%s\r\n" % (len(x), x) for x in a)
            for a in [[b"KV.FETCH", pack(u)], [b"ECHO", b"THRESHOLD"], [b"ECHO", b"0"]]))
        replies = [c._read(), c._read(), c._read()]
        check("a pipelined command after KV.FETCH is not read as its THRESHOLD",
              replies == [None, b"THRESHOLD", b"0"], repr(replies)[:120])
        check("server alive", p.poll() is None)
    finally:
        p.kill()
        p.wait()
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{'FAILED' if FAIL else 'PASSED'}: {len(FAIL)} failure(s)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
