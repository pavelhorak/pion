#!/usr/bin/env python3
"""gh #177 — RESP3 double type for score replies (gh #172 remainder).

Raw wire-level, same rationale as test_gh172.py: this is a wire-shape change
and a client library would hide the exact type byte we need to assert.

Scope is Redis-faithful, not uniform: probed against Redis 8.10 (2026-08-03),
only ZSCORE and ZINCRBY answer with the RESP3 double type `,`. INCRBYFLOAT,
HINCRBYFLOAT, and GEODIST stay bulk strings even under RESP3 in real Redis,
so they must stay bulk strings here too — those are asserted as negative
cases, not converted.

Covers:
  1. RESP3 ZSCORE is `,<score>\r\n` — fractional and integer-valued (the
     latter must emit bare digits, `,3\r\n` not `,3.0\r\n`, matching Redis).
  2. RESP3 ZINCRBY is a double for both branches (existing key and
     key-created-by-ZINCRBY).
  3. RESP3 ZSCORE miss is still the null type `_\r\n`.
  4. RESP2 replies are byte-identical to the pre-#177 wire (regression guard).
  5. INCRBYFLOAT / HINCRBYFLOAT stay bulk strings under RESP3 (Redis parity).

Seeding uses multi-member ZADD so scores take the slow path: the fast-path
single-member ZADD arm truncates fractional scores (pre-existing, gh #179)
and would poison these fixtures.
"""
import os
import socket
import subprocess
import sys
import time

PORT = int(os.environ.get("PION_TEST_PORT", "7377"))
BIN = os.environ.get("PION_BIN", "./pion-server")

failures = []
passes = []


def check(name, cond, detail=""):
    if cond:
        passes.append(name)
        print(f"  PASS  {name}")
    else:
        failures.append((name, detail))
        print(f"  FAIL  {name}   {detail}")


class Conn:
    def __init__(self, port=PORT):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.f = self.s.makefile("rb")

    def cmd(self, *args):
        out = f"*{len(args)}\r\n".encode()
        for a in args:
            b = a.encode() if isinstance(a, str) else a
            out += b"$%d\r\n%s\r\n" % (len(b), b)
        self.s.sendall(out)

    def read_reply(self):
        ln = self.f.readline()
        if not ln:
            raise EOFError("connection closed")
        t = ln[:1]
        raw = ln
        if t in (b"+", b"-", b":", b",", b"#", b"_"):
            return t, raw
        if t == b"$":
            n = int(ln[1:].strip())
            if n == -1:
                return t, raw
            return t, raw + self.f.read(n + 2)
        if t in (b"*", b"%", b">", b"~"):
            n = int(ln[1:].strip())
            if n == -1:
                return t, raw
            for _ in range(n * 2 if t == b"%" else n):
                _, sub = self.read_reply()
                raw += sub
            return t, raw
        raise AssertionError(f"unknown RESP type byte {t!r} in {ln!r}")

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def start_server():
    p = subprocess.Popen(
        [BIN, "-p", str(PORT), "--no-auto-detect", "--no-auto-embed", "-w", "1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        try:
            c = socket.create_connection(("127.0.0.1", PORT), timeout=1)
            c.close()
            time.sleep(0.5)
            return p
        except OSError:
            time.sleep(0.5)
    p.kill()
    raise RuntimeError(f"server did not come up on {PORT}")


def main():
    proc = start_server()
    try:
        c3 = Conn()
        c3.cmd("HELLO", "3")
        c3.read_reply()
        c2 = Conn()  # never sends HELLO — plain RESP2

        # Seed via multi-member ZADD (slow path — see module docstring).
        c2.cmd("ZADD", "gh177:z", "1.5", "frac", "3", "whole")
        _, raw = c2.read_reply()
        check("fixture ZADD accepted", raw == b":2\r\n", repr(raw))

        # ---- 1. RESP3 ZSCORE ---------------------------------------------
        c3.cmd("ZSCORE", "gh177:z", "frac")
        _, raw = c3.read_reply()
        check("RESP3 ZSCORE fractional is `,1.5`", raw == b",1.5\r\n", repr(raw))

        c3.cmd("ZSCORE", "gh177:z", "whole")
        _, raw = c3.read_reply()
        check("RESP3 ZSCORE integer-valued is bare `,3` (no .0)",
              raw == b",3\r\n", repr(raw))

        # ---- 3. RESP3 ZSCORE miss stays the null type ----------------------
        c3.cmd("ZSCORE", "gh177:z", "absent")
        _, raw = c3.read_reply()
        check("RESP3 ZSCORE miss is `_`", raw == b"_\r\n", repr(raw))

        # ---- 2. RESP3 ZINCRBY, both branches -------------------------------
        c3.cmd("ZINCRBY", "gh177:z", "0.5", "frac")   # 1.5 + 0.5 = 2
        _, raw = c3.read_reply()
        check("RESP3 ZINCRBY to integer value is bare `,2`",
              raw == b",2\r\n", repr(raw))

        c3.cmd("ZINCRBY", "gh177:z", "0.25", "frac")  # 2 + 0.25 = 2.25
        _, raw = c3.read_reply()
        check("RESP3 ZINCRBY fractional is `,2.25`", raw == b",2.25\r\n", repr(raw))

        c3.cmd("ZINCRBY", "gh177:new", "7.5", "m")    # key created by ZINCRBY
        _, raw = c3.read_reply()
        check("RESP3 ZINCRBY on fresh key is a double", raw == b",7.5\r\n", repr(raw))

        # ---- 4. RESP2 byte-identical regression guard ----------------------
        c2.cmd("ZSCORE", "gh177:z", "frac")
        _, raw = c2.read_reply()
        check("RESP2 ZSCORE fractional unchanged", raw == b"$4\r\n2.25\r\n", repr(raw))

        c2.cmd("ZSCORE", "gh177:z", "whole")
        _, raw = c2.read_reply()
        check("RESP2 ZSCORE integer-valued unchanged", raw == b"$1\r\n3\r\n", repr(raw))

        c2.cmd("ZINCRBY", "gh177:z", "0.75", "frac")  # 2.25 + 0.75 = 3
        _, raw = c2.read_reply()
        check("RESP2 ZINCRBY unchanged", raw == b"$1\r\n3\r\n", repr(raw))

        c2.cmd("ZSCORE", "gh177:z", "absent")
        _, raw = c2.read_reply()
        check("RESP2 ZSCORE miss is still `$-1`", raw == b"$-1\r\n", repr(raw))

        # ---- 5. Redis-parity negative cases: these stay bulk under RESP3 ---
        c3.cmd("INCRBYFLOAT", "gh177:k", "3.5")
        t, raw = c3.read_reply()
        check("RESP3 INCRBYFLOAT stays a bulk string (Redis parity)",
              t == b"$", repr(raw))

        c3.cmd("HSET", "gh177:h", "f", "1.0")
        c3.read_reply()
        c3.cmd("HINCRBYFLOAT", "gh177:h", "f", "4.5")
        t, raw = c3.read_reply()
        check("RESP3 HINCRBYFLOAT stays a bulk string (Redis parity)",
              t == b"$", repr(raw))

        # Frame-sync guard: a PING after the new-type replies must line up.
        for c in (c3, c2):
            c.cmd("PING")
            _, raw = c.read_reply()
            check("connection frame-synced after score replies",
                  raw == b"+PONG\r\n", repr(raw))

        c3.close(); c2.close()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        for f in ("pion.wal.0",):
            if os.path.exists(f):
                os.remove(f)

    print()
    print(f"gh #177: {len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  - {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
