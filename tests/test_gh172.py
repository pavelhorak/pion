#!/usr/bin/env python3
"""gh #172 — RESP3 response mode.

Verified at the raw wire level rather than through redis-py, for two reasons:
the Mac gate box runs redis-py 4.6 (no `protocol` kwarg, RESP2 only), and the
bug this closes was a *wire-shape* bug — a client library would paper over the
exact bytes we need to assert.

Covers:
  1. HELLO 3 negotiates and replies with a RESP3 map (`%`), not an array.
  2. HELLO 2 / no-HELLO still get the byte-identical RESP2 replies (regression
     guard: RESP2 is the wire every existing client speaks).
  3. Nulls become `_\\r\\n` on RESP3 and stay `$-1\\r\\n` on RESP2.
  4. Pub/sub confirmations and message delivery use the push type `>` on RESP3.
  5. CONFIG GET is a map on RESP3, flat array on RESP2.
  6. The protocol is per-connection, not per-server.
  7. Frame-sync regression guards for the three hand-counted pub/sub lengths
     that were wrong before this change (SUBSCRIBE over-read by one byte;
     UNSUBSCRIBE/PUNSUBSCRIBE/SUNSUBSCRIBE-all each dropped their final byte).
"""
import socket
import subprocess
import sys
import time
import os

PORT = int(os.environ.get("PION_TEST_PORT", "7372"))
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
    """Minimal RESP client. Uses makefile('rb') so framing comes from the
    stream, never from slicing a fixed-size recv (see the RESP-slicing trap
    that has bitten these tests before)."""

    def __init__(self, port=PORT):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.f = self.s.makefile("rb")

    def cmd(self, *args):
        out = f"*{len(args)}\r\n".encode()
        for a in args:
            b = a.encode() if isinstance(a, str) else a
            out += b"$%d\r\n%s\r\n" % (len(b), b)
        self.s.sendall(out)

    def line(self):
        ln = self.f.readline()
        if not ln:
            raise EOFError("connection closed")
        return ln

    def read_reply(self):
        """Return (type_byte, raw_bytes_of_whole_reply) with children consumed."""
        ln = self.line()
        t = ln[:1]
        raw = ln
        if t in (b"+", b"-", b":", b",", b"#", b"_"):
            return t, raw
        if t == b"$":
            n = int(ln[1:].strip())
            if n == -1:
                return t, raw
            body = self.f.read(n + 2)
            return t, raw + body
        if t in (b"*", b"%", b">", b"~"):
            n = int(ln[1:].strip())
            if n == -1:
                return t, raw
            count = n * 2 if t == b"%" else n
            for _ in range(count):
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
        # ---- 1/2. HELLO negotiation -------------------------------------
        c3 = Conn()
        c3.cmd("HELLO", "3")
        t, raw = c3.read_reply()
        check("HELLO 3 replies with a RESP3 map", t == b"%", f"got type {t!r}")
        check("HELLO 3 map declares proto 3", b"proto" in raw and b":3\r\n" in raw,
              raw[:90])

        c2 = Conn()
        c2.cmd("HELLO", "2")
        t2, raw2 = c2.read_reply()
        check("HELLO 2 still replies with a RESP2 array", t2 == b"*",
              f"got type {t2!r}")
        check("HELLO 2 array declares proto 2", b":2\r\n" in raw2, raw2[:90])

        cbad = Conn()
        cbad.cmd("HELLO", "4")
        tb, rawb = cbad.read_reply()
        check("HELLO 4 still -NOPROTO", tb == b"-" and b"NOPROTO" in rawb, rawb)
        cbad.close()

        # ---- 3. nulls ---------------------------------------------------
        c3.cmd("GET", "gh172:absent")
        t, raw = c3.read_reply()
        check("RESP3 null GET is `_\\r\\n`", raw == b"_\r\n", repr(raw))

        c2.cmd("GET", "gh172:absent")
        t, raw = c2.read_reply()
        check("RESP2 null GET is still `$-1\\r\\n`", raw == b"$-1\r\n", repr(raw))

        # ---- 6. per-connection isolation --------------------------------
        # c2 must be unaffected by c3's HELLO 3 — same server, same worker.
        c2.cmd("SET", "gh172:k", "v")
        _, raw = c2.read_reply()
        check("RESP2 connection unaffected by another fd's HELLO 3",
              raw == b"+OK\r\n", repr(raw))

        # ---- 5. CONFIG GET ----------------------------------------------
        c3.cmd("CONFIG", "GET", "maxmemory")
        t, raw = c3.read_reply()
        check("RESP3 CONFIG GET is a map", t == b"%", f"got type {t!r}")
        c2.cmd("CONFIG", "GET", "maxmemory")
        t, raw = c2.read_reply()
        check("RESP2 CONFIG GET is still a flat array", t == b"*" and raw.startswith(b"*2\r\n"),
              repr(raw[:20]))

        # ---- 4 + 7. pub/sub push type and frame sync ---------------------
        sub3 = Conn()
        sub3.cmd("HELLO", "3")
        sub3.read_reply()
        sub3.cmd("SUBSCRIBE", "gh172chan")
        t, raw = sub3.read_reply()
        check("RESP3 SUBSCRIBE confirmation uses push type `>`", t == b">",
              f"got type {t!r} raw={raw[:40]!r}")

        sub2 = Conn()
        sub2.cmd("SUBSCRIBE", "gh172chan")
        t, raw = sub2.read_reply()
        check("RESP2 SUBSCRIBE confirmation stays an array", t == b"*",
              f"got type {t!r}")
        check("SUBSCRIBE confirmation is exactly `subscribe`",
              raw == b"*3\r\n$9\r\nsubscribe\r\n$9\r\ngh172chan\r\n:1\r\n", repr(raw))

        pub = Conn()
        pub.cmd("PUBLISH", "gh172chan", "hi")
        pub.read_reply()

        t, raw = sub3.read_reply()
        check("RESP3 message delivery uses push type `>`", t == b">",
              f"got type {t!r} raw={raw[:40]!r}")
        check("RESP3 delivered message body intact", b"hi" in raw, repr(raw))
        t, raw = sub2.read_reply()
        check("RESP2 message delivery stays an array", t == b"*", f"got type {t!r}")
        check("RESP2 delivered message body intact", b"hi" in raw, repr(raw))

        # Frame-sync guard: the three *-all unsubscribes used to emit one byte
        # short, which desyncs everything after them on the connection. Issue an
        # unsubscribe-all then a PING and require the PING reply to line up.
        for cmdname in ("UNSUBSCRIBE", "PUNSUBSCRIBE", "SUNSUBSCRIBE"):
            cc = Conn()
            cc.cmd(cmdname)
            t, raw = cc.read_reply()
            cc.cmd("PING")
            t2, raw2 = cc.read_reply()
            check(f"{cmdname}-all keeps the connection frame-synced",
                  raw2 == b"+PONG\r\n",
                  f"after {cmdname}: reply={raw!r} then ping={raw2!r}")
            cc.close()

        sub3.close(); sub2.close(); pub.close(); c3.close(); c2.close()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        for f in (f"pion.wal.0",):
            if os.path.exists(f):
                os.remove(f)

    print()
    print(f"gh #172: {len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  - {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
