#!/usr/bin/env python3
"""gh #219 — a MULTI...EXEC transaction sent in ONE write must execute.

Before the fix, the MULTI-mode interception in slow_path.mojo enqueued the
whole recv buffer (not the one command frame) and then set `i = num_tokens`,
so a pipelined transaction got a single +QUEUED, never reached its EXEC, and
wrote nothing — while the client waited on N missing replies.

This is redis-py's DEFAULT pipeline shape (`r.pipeline()` is transaction=True,
which emits MULTI + commands + EXEC in a single sendall).

Usage: python3 tests/test_gh219.py [--port 1974]
Assumes a server is already listening (start it with -w 1).
"""

import argparse
import socket
import sys
import time

HOST = "127.0.0.1"
PASSED = []
FAILED = []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))


class Conn:
    def __init__(self, port, timeout=5.0):
        self.s = socket.create_connection((HOST, port), timeout=timeout)
        self.s.settimeout(timeout)
        self.f = self.s.makefile("rb")

    def send(self, *cmds):
        """Send every command in ONE write (the pipelined shape #219 broke)."""
        buf = b""
        for c in cmds:
            parts = c if isinstance(c, (list, tuple)) else c.split()
            buf += f"*{len(parts)}\r\n".encode()
            for p in parts:
                p = p.encode() if isinstance(p, str) else p
                buf += b"$%d\r\n%s\r\n" % (len(p), p)
        self.s.sendall(buf)

    def read_reply(self):
        line = self.f.readline()
        if not line:
            raise EOFError("connection closed")
        t, body = line[:1], line[1:-2]
        if t in b"+-:":
            return body.decode()
        if t == b"$":
            n = int(body)
            if n == -1:
                return None
            data = self.f.read(n + 2)[:-2]
            return data.decode()
        if t == b"*":
            n = int(body)
            if n == -1:
                return None
            return [self.read_reply() for _ in range(n)]
        raise ValueError(f"bad RESP type {t!r} in {line!r}")

    def read_n(self, n):
        return [self.read_reply() for _ in range(n)]

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def has_pending(conn, wait=0.3):
    """True if the server sent bytes we did not account for."""
    conn.s.settimeout(wait)
    try:
        extra = conn.s.recv(4096)
    except socket.timeout:
        return False, b""
    finally:
        conn.s.settimeout(5.0)
    return len(extra) > 0, extra


def test_pipelined_transaction(port):
    """The headline case: MULTI + 3 commands + EXEC in one write."""
    c = Conn(port)
    c.send("MULTI", "SET gh219:a 1", "SET gh219:b 2", "INCR gh219:c", "EXEC")
    replies = c.read_n(5)
    check(
        "pipelined MULTI: 5 replies for 5 commands",
        len(replies) == 5,
        f"got {replies}",
    )
    check("  reply 1 is +OK (MULTI)", replies[0] == "OK", f"got {replies[0]!r}")
    check(
        "  replies 2-4 are +QUEUED",
        replies[1:4] == ["QUEUED"] * 3,
        f"got {replies[1:4]}",
    )
    check(
        "  reply 5 is the EXEC array [OK, OK, 1]",
        replies[4] == ["OK", "OK", "1"],
        f"got {replies[4]!r}",
    )
    c.close()

    # The writes must actually be visible afterwards.
    v = Conn(port)
    v.send("GET gh219:a", "GET gh219:b", "GET gh219:c")
    vals = v.read_n(3)
    check(
        "pipelined MULTI: keys are written",
        vals == ["1", "2", "1"],
        f"got {vals}",
    )
    v.close()


def test_connection_not_desynced(port):
    """After a pipelined transaction the connection must be reply-aligned."""
    c = Conn(port)
    c.send("MULTI", "SET gh219:d 9", "EXEC")
    c.read_n(3)
    leftover, extra = has_pending(c)
    check("no stray replies after EXEC", not leftover, f"extra={extra!r}")

    # A subsequent command must get its OWN reply, not a stale one.
    c.send("PING")
    r = c.read_reply()
    check("connection still aligned after EXEC", r == "PONG", f"got {r!r}")

    # And the connection must no longer be in MULTI mode.
    c.send("SET gh219:e 5")
    r = c.read_reply()
    check("MULTI mode exited after EXEC", r == "OK", f"got {r!r}")
    c.close()


def test_pipelined_discard(port):
    """DISCARD in the same write must drop the queued commands, not apply them."""
    c = Conn(port)
    c.send("MULTI", "SET gh219:discarded 1", "DISCARD")
    replies = c.read_n(3)
    check(
        "pipelined DISCARD: 3 replies",
        len(replies) == 3 and replies[0] == "OK" and replies[2] == "OK",
        f"got {replies}",
    )
    c.send("GET gh219:discarded")
    r = c.read_reply()
    check("pipelined DISCARD: key NOT written", r is None, f"got {r!r}")
    c.close()


def test_commands_after_exec_in_same_write(port):
    """Commands trailing EXEC in the same write still run (batch continues)."""
    c = Conn(port)
    c.send("MULTI", "SET gh219:f 1", "EXEC", "GET gh219:f", "PING")
    replies = c.read_n(5)
    check(
        "commands after EXEC in same write execute",
        len(replies) == 5 and replies[3] == "1" and replies[4] == "PONG",
        f"got {replies}",
    )
    c.close()


def test_multi_only_then_separate_exec(port):
    """The single-write-per-command path must keep working (no regression)."""
    c = Conn(port)
    for cmd, want in (("MULTI", "OK"), ("SET gh219:g 7", "QUEUED"), ("INCR gh219:h", "QUEUED")):
        c.send(cmd)
        r = c.read_reply()
        if r != want:
            check(f"one-write-per-command: {cmd}", False, f"got {r!r} want {want!r}")
            c.close()
            return
    c.send("EXEC")
    r = c.read_reply()
    check("one-write-per-command MULTI still works", r == ["OK", "1"], f"got {r!r}")
    c.close()


def test_redis_py_default_pipeline(port):
    """The real client shape, if redis-py is importable."""
    try:
        import redis
    except ImportError:
        print("  SKIP  redis-py not installed")
        return
    # redis-py < 5 has no `protocol` kwarg; ≥5 defaults to RESP2 anyway unless
    # HELLO 3 is requested, so pass it only where it exists.
    kw = {"host": HOST, "port": port, "socket_timeout": 5}
    if redis.VERSION[0] >= 5:
        kw["protocol"] = 2
    r = redis.Redis(**kw)
    with r.pipeline() as p:  # transaction=True by default
        p.set("gh219:rp:a", "1")
        p.set("gh219:rp:b", "2")
        p.incr("gh219:rp:c")
        res = p.execute()
    check("redis-py default pipeline() executes", res == [True, True, 1], f"got {res}")
    vals = [r.get("gh219:rp:a"), r.get("gh219:rp:b"), r.get("gh219:rp:c")]
    check(
        "redis-py default pipeline() keys written",
        vals == [b"1", b"2", b"1"],
        f"got {vals}",
    )
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()

    try:
        Conn(args.port).close()
    except OSError as e:
        print(f"FATAL: no server on {HOST}:{args.port} ({e})")
        return 2

    print(f"gh #219 pipelined-transaction tests against {HOST}:{args.port}\n")

    # Self-isolate. A server that loaded a snapshot from an earlier run still
    # holds these keys, and the INCR assertions then read an accumulated
    # counter (3 instead of 1) and false-fail — reporting a regression where
    # there is only leftover state.
    cleanup = Conn(args.port)
    keys = ["gh219:a", "gh219:b", "gh219:c", "gh219:d", "gh219:e", "gh219:f",
            "gh219:g", "gh219:h", "gh219:discarded",
            "gh219:rp:a", "gh219:rp:b", "gh219:rp:c"]
    cleanup.send(*[f"DEL {k}" for k in keys])
    cleanup.read_n(len(keys))
    cleanup.close()
    for t in (
        test_pipelined_transaction,
        test_connection_not_desynced,
        test_pipelined_discard,
        test_commands_after_exec_in_same_write,
        test_multi_only_then_separate_exec,
        test_redis_py_default_pipeline,
    ):
        print(t.__doc__.splitlines()[0])
        try:
            t(args.port)
        except Exception as e:  # a desync shows up as a timeout/parse error
            check(t.__name__, False, f"raised {type(e).__name__}: {e}")
        print()
        time.sleep(0.05)

    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
