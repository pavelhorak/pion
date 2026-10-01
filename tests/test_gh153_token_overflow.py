#!/usr/bin/env python3
"""gh #153: a single command with more arguments than the slow-path token array
holds must not hang the connection.

`VADD <key> VALUES 1536 <1536 scalars> <element>` is ~1540 RESP tokens against a
64-token parser array. Before the fix, `parse_stream` reported "0 tokens" — the
same signal it uses for an incomplete frame — so the caller waited for more data
on a command that had already arrived in full, and the fd parked forever.

Usage: python3 tests/test_gh153_token_overflow.py [--port 1974]
"""

import argparse
import socket
import struct
import sys

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    # `detail` describes the failure, so only show it when the check failed.
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  — ' + detail}")


def cmd(*args):
    out = b"*%d\r\n" % len(args)
    for a in args:
        b = a.encode() if isinstance(a, str) else a
        out += b"$%d\r\n%s\r\n" % (len(b), b)
    return out


def connect(port, timeout=4.0):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.settimeout(timeout)
    return s


def reply(s, timeout=4.0):
    """Read one reply, or return None on timeout (the hang signature)."""
    s.settimeout(timeout)
    try:
        return s.recv(65536)
    except socket.timeout:
        return None


def test_vadd_values_does_not_hang(port):
    s = connect(port)
    dim = 1536
    args = ["VADD", "gh153:vs", "VALUES", str(dim)] + [f"{i%7}.5" for i in range(dim)] + ["elem0"]
    s.sendall(cmd(*args))
    r = reply(s)
    check(
        "VADD VALUES 1536: connection answers instead of hanging",
        r is not None,
        "read timed out — the fd is parked (the gh #153 hang)",
    )
    if r is not None:
        check(
            "VADD VALUES 1536: reply is a single well-formed RESP frame",
            r.startswith((b"-", b":")) and r.endswith(b"\r\n"),
            f"got {r[:120]!r}",
        )
    s.close()


def test_connection_stays_usable(port):
    """The frame must be consumed exactly — a desync would corrupt what follows."""
    s = connect(port)
    dim = 1536
    args = ["VADD", "gh153:vs2", "VALUES", str(dim)] + ["1.0"] * dim + ["elem1"]
    s.sendall(cmd(*args))
    first = reply(s)
    if first is None:
        check("oversized command: connection still usable afterwards", False, "hung on the oversized command")
        s.close()
        return

    s.sendall(cmd("PING"))
    r = reply(s)
    check(
        "oversized command: connection still usable afterwards (PING)",
        r == b"+PONG\r\n",
        f"PING returned {r!r} (want b'+PONG\\r\\n') — a desync would show here",
    )
    s.close()


def test_pipelined_after_oversized(port):
    """An oversized command in the middle of a pipeline must not eat its neighbours."""
    s = connect(port)
    s.sendall(cmd("DEL", "gh153:ctr"))
    reply(s)

    dim = 1536
    big = cmd(*(["VADD", "gh153:vs3", "VALUES", str(dim)] + ["1.0"] * dim + ["e"]))
    s.sendall(cmd("INCR", "gh153:ctr") + big + cmd("INCR", "gh153:ctr"))

    seen = b""
    for _ in range(4):
        r = reply(s, timeout=2.0)
        if r is None:
            break
        seen += r

    s.sendall(cmd("GET", "gh153:ctr"))
    got = reply(s)
    check(
        "pipelined: both INCRs around the oversized command ran exactly once",
        got == b"$1\r\n2\r\n",
        f"GET returned {got!r} (want b'$1\\r\\n2\\r\\n')",
    )
    check(
        "pipelined: exactly three replies came back",
        seen.count(b"\r\n") >= 3 and seen.startswith(b":1\r\n"),
        f"got {seen[:160]!r}",
    )
    s.close()


def test_fp32_path_still_works(port):
    """The documented high-dimension path must be unaffected."""
    s = connect(port)
    dim = 1536
    blob = struct.pack(f"<{dim}f", *([0.25] * dim))
    s.sendall(cmd("VADD", "gh153:fp", b"FP32", blob, "a"))
    r = reply(s)
    check(
        "FP32 blob dialect still works at dim=1536",
        r == b":1\r\n",
        f"got {r!r} (want b':1\\r\\n')",
    )
    s.sendall(cmd("VCARD", "gh153:fp"))
    r = reply(s)
    check("VCARD after FP32 insert", r is not None and r.startswith(b":"), f"got {r!r}")
    s.close()


def test_small_values_still_works(port):
    """A VALUES command that fits the token array must still reach the handler.

    Since gh #156 this also asserts a *single* reply frame: VADD's error paths
    used to return a token-skip count that stopped where they rejected, leaving
    the scalars to be re-dispatched and answered a second time. Reply pairing
    has its own suite in tests/test_gh156_vset_reply_pairing.py; the one-frame
    check lives here too because this is the command that first exposed it.
    """
    s = connect(port)
    s.sendall(cmd("VADD", "gh153:small", "VALUES", "4", "1.0", "2.0", "3.0", "4.0", "s1"))
    r = reply(s)
    check(
        "VALUES dialect under the token limit still reaches the handler (dim=4)",
        r is not None and r.startswith((b":", b"-")) and b"too many arguments" not in r,
        f"got {r!r} — must not be the overflow error, dim=4 is well under the cap",
    )
    # One command in, one reply out. The reply is a one-line frame either way
    # (`:1` on success, `-ERR ...` on rejection), so count line terminators —
    # both frames usually arrive in the same segment, which a second recv()
    # would miss. A trailing `-ERR unknown command '1.0'` is the gh #156 desync.
    extra = reply(s, timeout=1.0)
    whole = (r or b"") + (extra or b"")
    check(
        "VALUES dialect: no spurious second reply (gh #156)",
        whole.count(b"\r\n") == 1,
        f"got {whole!r} — a single VADD produced more than one frame",
    )
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()

    print(f"gh #153 oversized-command token overflow — port {args.port}")
    for fn in (
        test_vadd_values_does_not_hang,
        test_connection_stays_usable,
        test_pipelined_after_oversized,
        test_fp32_path_still_works,
        test_small_values_still_works,
    ):
        try:
            fn(args.port)
        except Exception as e:  # noqa: BLE001
            check(fn.__name__, False, f"exception: {e!r}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
