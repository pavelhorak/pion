#!/usr/bin/env python3
"""ATTEND.* must refuse malformed arguments — and must answer what it stored.

WHY
Every ATTEND numeric argument was parsed as `n = n * 10 + (byte - 48)` over
every byte (gh #229's bug, in the substrate), and no blob length was checked
against the session's dimensions:

  ATTEND.CREATE s abc 128          -> a session with a garbage key_dim
  ATTEND.CREATE s -5 128           -> a negative dimension
  ATTEND.STORE s 0 99999 <128 B>…  -> +OK, having read ~6 MB PAST the request
                                      (other clients' bytes) into the store
  ATTEND.QUERY s 0 100000000 <q>   -> a ~13 GB scratch allocation
  ATTEND.QUERY s 0 1 <1 byte>      -> a 64-byte over-read of the query
  ATTEND.QUERY nosuch 0 1 <q>      -> [] — indistinguishable from "no match"

and all of them answered success. This checks each is refused, that a refusal
stores nothing, that framing holds (each command is followed by a PING), and
that a query with a stored key returns exactly that key's stored value.

Needs a server started with --kvcache.
    python3 tests/test_attend_args.py [--port 1974]
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, wait_ready  # noqa: E402

KD, VD = 16, 8


def f32(vals):
    return struct.pack(f"<{len(vals)}f", *vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    args = ap.parse_args()
    wait_ready(args.port, 30)
    c = Conn(args.port, timeout=10)
    fails = []

    def expect_error(label, *cmd):
        r = c.cmd_synced(*cmd)
        if not isinstance(r, RespError):
            fails.append(f"{label}: accepted, replied {r!r:.80}")

    info0 = c.cmd_synced("ATTEND.INFO")
    for label, cmd in [
        ("key_dim 'abc'", ("ATTEND.CREATE", "bad1", "abc", "8")),
        ("key_dim -5", ("ATTEND.CREATE", "bad2", "-5", "8")),
        ("value_dim 0", ("ATTEND.CREATE", "bad3", "16", "0")),
        ("key_dim 1e3", ("ATTEND.CREATE", "bad4", "1e3", "8")),
        ("key_dim huge", ("ATTEND.CREATE", "bad5", "99999999999999999999", "8")),
    ]:
        expect_error(f"CREATE {label}", *cmd)
    if c.cmd_synced("ATTEND.INFO") != info0:
        fails.append("a refused ATTEND.CREATE changed ATTEND.INFO (it created a session)")

    sid = c.cmd_synced("ATTEND.CREATE", "good", str(KD), str(VD))
    if not isinstance(sid, int) or sid < 0:
        fails.append(f"CREATE good: {sid!r}")

    # Two tokens, distinct keys; value rows are unique markers.
    keys = [[1.0 if d == 0 else 0.0 for d in range(KD)], [1.0 if d == 5 else 0.0 for d in range(KD)]]
    # 0/1 rows: values are INT8-quantized with one scale per layer, and only
    # the range endpoints come back exact.
    vals = [[float(d % 2) for d in range(VD)], [float((d + 1) % 2) for d in range(VD)]]
    kb = f32(keys[0] + keys[1]); vb = f32(vals[0] + vals[1])
    for label, cmd in [
        ("count 99999 vs 2-row blobs", ("ATTEND.STORE", "good", "0", "99999", kb, vb)),
        ("count 3 vs 2-row blobs", ("ATTEND.STORE", "good", "0", "3", kb, vb)),
        ("count 'abc'", ("ATTEND.STORE", "good", "0", "abc", kb, vb)),
        ("layer -1", ("ATTEND.STORE", "good", "-1", "2", kb, vb)),
        ("layer 'x'", ("ATTEND.STORE", "good", "x", "2", kb, vb)),
        ("short values blob", ("ATTEND.STORE", "good", "0", "2", kb, vb[:-4])),
        ("unknown session", ("ATTEND.STORE", "nosuch", "0", "2", kb, vb)),
    ]:
        expect_error(f"STORE {label}", *cmd)

    r = c.cmd_synced("ATTEND.STORE", "good", "0", "2", kb, vb)
    if r != "OK":
        fails.append(f"STORE good: {r!r}")
    r = c.cmd_synced("ATTEND.FINALIZE", "good", "0")
    if r != "OK":
        fails.append(f"FINALIZE good: {r!r}")
    expect_error("FINALIZE layer 'x'", "ATTEND.FINALIZE", "good", "x")

    q0 = f32(keys[0])
    for label, cmd in [
        ("k 'abc'", ("ATTEND.QUERY", "good", "0", "abc", q0)),
        ("k 100000000", ("ATTEND.QUERY", "good", "0", "100000000", q0)),
        ("k 0", ("ATTEND.QUERY", "good", "0", "0", q0)),
        ("1-byte query", ("ATTEND.QUERY", "good", "0", "1", b"\x00")),
        ("query of key_dim+1 floats", ("ATTEND.QUERY", "good", "0", "1", f32(keys[0] + [0.0]))),
        ("layer 9999999999", ("ATTEND.QUERY", "good", "9999999999", "1", q0)),
        ("unknown session", ("ATTEND.QUERY", "nosuch", "0", "1", q0)),
    ]:
        expect_error(f"QUERY {label}", *cmd)

    # Round trip: querying with a stored key must return THAT key's value row.
    for t in (0, 1):
        r = c.cmd_synced("ATTEND.QUERY", "good", "0", "1", f32(keys[t]))
        got = list(struct.unpack(f"<{VD}f", r)) if isinstance(r, bytes) and len(r) == VD * 4 else r
        if got != vals[t]:
            fails.append(f"QUERY with stored key {t} returned {got!r:.100}, expected {vals[t]}")

    c.assert_in_sync()
    print(f"ATTEND argument contract: {len(fails)} failure(s)")
    for f in fails:
        print("  FAIL " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
