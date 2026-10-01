#!/usr/bin/env python3
"""gh #147: a pipelined batch that ends on a partial frame must still reply to
the complete frames it contains.

Also probes the adjacent exit (gh #147 comment): an *unknown* command mid-batch
makes the fast path break with `fast_path_ok=False` and fall to `return 0`,
discarding a non-zero `consumed`. If the caller then re-runs the slow path over
the whole buffer, the already-executed prefix commands run a second time.

Usage: python3 tests/test_gh147_pipeline_flush.py [--port 1974]
"""

import argparse
import socket
import sys

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


def cmd(*args):
    out = b"*%d\r\n" % len(args)
    for a in args:
        b = a.encode() if isinstance(a, str) else a
        out += b"$%d\r\n%s\r\n" % (len(b), b)
    return out


def connect(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.settimeout(2.0)
    return s


def drain(s, timeout=1.5):
    """Read whatever is available until a short idle gap."""
    s.settimeout(timeout)
    buf = b""
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            s.settimeout(0.25)
    except socket.timeout:
        pass
    finally:
        s.settimeout(2.0)
    return buf


def test_partial_tail_still_replies(port):
    """The headline #147 repro: complete frame + partial frame in one write()."""
    s = connect(port)
    s.sendall(cmd("DEL", "gh147:a", "gh147:b"))
    drain(s)

    a = cmd("SET", "gh147:a", "one")
    b = cmd("SET", "gh147:b", "two")
    s.sendall(a + b[:7])  # complete frame + 7-byte partial tail

    early = drain(s, timeout=1.5)
    check(
        "partial tail: complete frame is answered before the tail arrives",
        early == b"+OK\r\n",
        f"got {early!r} (want b'+OK\\r\\n')",
    )

    s.sendall(b[7:])
    rest = drain(s)
    total = early + rest
    check(
        "partial tail: both replies arrive exactly once overall",
        total == b"+OK\r\n+OK\r\n",
        f"got {total!r}",
    )
    s.close()


def test_partial_tail_exactly_once(port):
    """Execution must stay exactly-once across the partial-frame boundary."""
    s = connect(port)
    s.sendall(cmd("DEL", "gh147:ctr"))
    drain(s)

    a = cmd("INCR", "gh147:ctr")
    b = cmd("INCR", "gh147:ctr")
    s.sendall(a + b[:7])
    drain(s, timeout=1.0)
    s.sendall(b[7:])
    drain(s)

    s.sendall(cmd("GET", "gh147:ctr"))
    got = drain(s)
    check(
        "partial tail: counter is exactly 2 (no double execution)",
        got == b"$1\r\n2\r\n",
        f"GET returned {got!r} (want b'$1\\r\\n2\\r\\n')",
    )
    s.close()


def test_unknown_cmd_midbatch_exactly_once(port):
    """A fast-path miss after a handled command must not re-execute the prefix."""
    s = connect(port)
    s.sendall(cmd("DEL", "gh147:mix"))
    drain(s)

    # INCR is fast-path; ECHO is slow-path only -> fast path breaks mid-batch.
    s.sendall(cmd("INCR", "gh147:mix") + cmd("ECHO", "hello"))
    replies = drain(s)

    s.sendall(cmd("GET", "gh147:mix"))
    got = drain(s)
    check(
        "fast-path miss mid-batch: prefix command executed exactly once",
        got == b"$1\r\n1\r\n",
        f"GET returned {got!r} (want b'$1\\r\\n1\\r\\n' — b'2' means double execution)",
    )
    check(
        "fast-path miss mid-batch: exactly two replies, in order",
        replies == b":1\r\n$5\r\nhello\r\n",
        f"got {replies!r} (want b':1\\r\\n$5\\r\\nhello\\r\\n')",
    )
    s.close()


def test_many_frames_then_partial(port):
    """A deeper batch: 8 complete frames + a partial tail."""
    s = connect(port)
    s.sendall(cmd("DEL", "gh147:deep"))
    drain(s)

    batch = b"".join(cmd("INCR", "gh147:deep") for _ in range(8))
    tail = cmd("INCR", "gh147:deep")
    s.sendall(batch + tail[:5])

    early = drain(s, timeout=1.5)
    check(
        "deep batch: all 8 complete frames answered before the tail completes",
        early == b":1\r\n:2\r\n:3\r\n:4\r\n:5\r\n:6\r\n:7\r\n:8\r\n",
        f"got {early!r}",
    )

    s.sendall(tail[5:])
    drain(s)
    s.sendall(cmd("GET", "gh147:deep"))
    got = drain(s)
    check(
        "deep batch: counter is exactly 9",
        got == b"$1\r\n9\r\n",
        f"GET returned {got!r}",
    )
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()

    print(f"gh #147 pipelined partial-frame flush — port {args.port}")
    for fn in (
        test_partial_tail_still_replies,
        test_partial_tail_exactly_once,
        test_unknown_cmd_midbatch_exactly_once,
        test_many_frames_then_partial,
    ):
        try:
            fn(args.port)
        except Exception as e:  # noqa: BLE001
            check(fn.__name__, False, f"exception: {e!r}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
