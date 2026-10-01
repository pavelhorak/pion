#!/usr/bin/env python3
"""Malformed-frame robustness: a bad frame must never crash or desync.

gh #102 (bulk-length overflow wrapping Int negative), gh #153 (a command with
more args than the token table, which used to hang the connection) and gh #166
(token table on the stack) are all the same family: a frame the parser did not
expect, reaching code that assumed it was well-formed.

The invariant under test is deliberately weak, because a server may reasonably
answer OR wait OR close on garbage — what it may NOT do is:

  * crash the worker (every later client dies too),
  * accept the frame and then answer the NEXT command with the wrong reply,
  * hang forever on a frame that can never complete.

Each case therefore ends with a liveness probe on a FRESH connection: the
server must still be serving. Cases that leave the probing connection itself
in an undefined state are fine — that connection is discarded.

Usage: python3 tests/test_resp_malformed.py [--port 1974]
"""

import argparse
import socket
import sys

HOST = "127.0.0.1"
PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {detail}")


def server_alive(port):
    """Fresh connection, PING, expect PONG. The real invariant."""
    try:
        s = socket.create_connection((HOST, port), timeout=4)
        s.settimeout(4)
        s.sendall(b"*1\r\n$4\r\nPING\r\n")
        data = s.recv(64)
        s.close()
        return data == b"+PONG\r\n"
    except OSError:
        return False


def send_frame(port, payload, wait=0.4):
    """Send raw bytes, return whatever comes back (or b'' on timeout)."""
    try:
        s = socket.create_connection((HOST, port), timeout=4)
        s.settimeout(wait)
        s.sendall(payload)
        try:
            out = s.recv(4096)
        except socket.timeout:
            out = b""
        s.close()
        return out
    except OSError as e:
        return f"CONN-ERR:{e}".encode()


# (label, payload) — each is a frame a hostile or buggy client can send.
CASES = [
    ("negative bulk length",        b"*2\r\n$3\r\nGET\r\n$-5\r\nxx\r\n"),
    ("negative arg count",          b"*-3\r\n$3\r\nGET\r\n"),
    ("huge bulk length (19 digits)", b"*2\r\n$3\r\nGET\r\n$9999999999999999999\r\nx\r\n"),
    ("huge arg count (19 digits)",  b"*9999999999999999999\r\n$3\r\nGET\r\n"),
    ("bulk length overflows Int64", b"*2\r\n$3\r\nGET\r\n$99999999999999999999999\r\nx\r\n"),
    ("length/payload mismatch",     b"*2\r\n$3\r\nGET\r\n$100\r\nshort\r\n"),
    ("missing CRLF after bulk",     b"*2\r\n$3\r\nGET\r\n$3\r\nabc"),
    ("truncated mid-header",        b"*2\r\n$3\r\nGE"),
    ("bare CR",                     b"*1\r\r$4\r\nPING\r\n"),
    ("bare LF",                     b"*1\n$4\nPING\n"),
    ("NUL inside command name",     b"*1\r\n$4\r\nPI\x00G\r\n"),
    ("NUL inside key",              b"*2\r\n$3\r\nGET\r\n$5\r\na\x00b\x00c\r\n"),
    ("empty command name",          b"*1\r\n$0\r\n\r\n"),
    ("zero args",                   b"*0\r\n"),
    ("non-numeric bulk length",     b"*2\r\n$3\r\nGET\r\n$abc\r\nx\r\n"),
    ("non-numeric arg count",       b"*abc\r\n$3\r\nGET\r\n"),
    ("type byte garbage",           b"\x01\x02\x03\x04\r\n"),
    ("inline garbage",              b"this is not resp at all\r\n"),
    ("only CRLF",                   b"\r\n"),
    ("nested array as command",     b"*1\r\n*1\r\n$4\r\nPING\r\n"),
    ("bulk claims 512MB+1",         b"*2\r\n$3\r\nGET\r\n$536870913\r\nx\r\n"),
    ("many small args (4096)",      b"*4096\r\n" + b"$1\r\nx\r\n" * 4096),
    ("deep pipeline of garbage",    b"*1\r\n$0\r\n\r\n" * 200),
    ("valid cmd then garbage tail", b"*1\r\n$4\r\nPING\r\n\xff\xfe\xfd"),
    ("CRLF flood",                  b"\r\n" * 500),
    # gh #410: these are WELL-FORMED frames carrying a valid command with a
    # hostile argument. EVAL/FCALL sized a numkeys-count array before checking
    # numkeys against the frame, so a huge numkeys made the alloc fail and
    # aborted the whole process — one unauthenticated packet, default config.
    ("EVAL numkeys 1e6",            b"*3\r\n$4\r\nEVAL\r\n$8\r\nreturn 1\r\n$7\r\n1000000\r\n"),
    ("EVAL numkeys int64 max",      b"*3\r\n$4\r\nEVAL\r\n$8\r\nreturn 1\r\n$19\r\n9223372036854775807\r\n"),
    ("FCALL numkeys 1e6",           b"*3\r\n$5\r\nFCALL\r\n$4\r\nnofn\r\n$7\r\n1000000\r\n"),
    # A returned table with a metamethod that errors: result serialization runs
    # outside any protected call, so the error used to panic the process.
    ("EVAL erroring __index return",
     b"*3\r\n$4\r\nEVAL\r\n$57\r\nreturn setmetatable({},{__index=function() error(1) end})\r\n$1\r\n0\r\n"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()

    if not server_alive(args.port):
        print(f"FATAL: no server on {HOST}:{args.port}")
        return 2

    print(f"malformed-frame robustness against {HOST}:{args.port}")
    print(f"{len(CASES)} frames; after each, a FRESH connection must still PING\n")

    for label, payload in CASES:
        reply = send_frame(args.port, payload)
        alive = server_alive(args.port)
        if not alive:
            check(label, False, f"SERVER DEAD after this frame (reply={reply[:40]!r})")
            print("\n  Stopping: the server is gone, later results would be meaningless.")
            print("  Check the stderr log for a Mojo ABORT and pion-<port>.crash.log")
            print("  for a signal backtrace (0.914+ captures frames).")
            break
        # A reply is optional (waiting for more bytes is legitimate for a
        # truncated frame); staying alive is not.
        shape = "no reply (waiting)" if reply == b"" else f"replied {reply[:34]!r}"
        check(label, True, shape)

    print(f"\n{len(PASSED)}/{len(CASES)} frames left the server serving")
    if FAILED:
        for f in FAILED:
            print(f"  FAILED: {f}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
