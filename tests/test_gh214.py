#!/usr/bin/env python3
"""gh #214 + #218 — no command may swallow the next pipelined command.

This is `tests/diag_pipeline_desync_sweep.py` promoted from a diagnostic to a
hard assertion, which is what that file asked for once its list reached zero.

## The bug

19 of 57 probed commands consumed the command pipelined behind them without
answering it. Their handlers return `num_tokens - i - 1` — a skip to the end of
the *entire recv buffer's* token array — from a dispatch site that advances with
`i += handler(...)`:

    i += handle_xinfo(tokens, i, num_tokens, writer, self.keyspace)
    i += handle_acl(tokens, i, num_tokens, writer)

`num_tokens` is the batch bound, not the command bound. One command in, one reply
out, and everything batched behind it disappears. A pipelining client
(redis-py `pipeline()`, memtier `-P>1`, any batching driver) then pairs every
later reply with the wrong request — silent wrong answers, not an error.

Affected: XRANGE, XREVRANGE, XINFO, XDEL, XTRIM, SINTERCARD, ZREM, ACL
(WHOAMI/LIST/USERS/CAT/LOG), DEBUG, MEMORY, SLOWLOG, LATENCY, MODULE,
FUNCTION LIST, SCRIPT EXISTS.

Fixed by converting all 15 dispatch sites to the canonical form already used by
the VADD/VSIM/GEO/FT.* families:

    _ = handle_xinfo(tokens, i, cmd_end_tok, writer, self.keyspace)
    i = cmd_end_tok - 1

The parser's command boundary is authoritative, so the handler's return value is
discarded. Handlers also take `cmd_end_tok` as their token bound, not
`num_tokens`, so an optional-argument scan cannot reach into the next pipelined
command's tokens.

This is the fourth recurrence of the same class (gh #156 VADD/VSIM error paths,
gh #162 GEO, FT.* 2026-08-05), which is why it is a gate test and not a
diagnostic: the probe is cheap and the failure mode is invisible to any client
that does not strictly pair pipelined replies.

## gh #218 — the same bug, two paths #214's probe could not reach

#214 converted only the 15 dispatch sites its 57-command success-path sweep
reached. A wider probe found the class still open at 171 unconverted sites:

  * **4 commands swallowed on VALID input** — TOUCH, SORT, UNLINK, XACK. All
    variadic, so their trailing scan ran to `num_tokens` and ate the next
    command. No malformed input required.
  * **50 commands swallowed on their arity-error path**, and read the next
    command's tokens as their own arguments. Bare `ECHO` replied
    `$4\r\nPING\r\n` — the following command's name, echoed back. Bare `DUMP`
    used `PING` as its key; bare `GETDEL` returned 8 NUL bytes.

Fixed by converting ALL 186 dispatch sites, so `cmd_end_tok` bounds every
handler. That makes the handler's return value structurally irrelevant rather
than incidentally harmless, which is the point — this was the fifth instance of
one bug. This file now probes all three surfaces: success paths (#214), valid
variadic forms and argumentless forms (#218), plus a direct assertion that bare
ECHO cannot see the next command.

Usage: python3 tests/test_gh214.py [./pion-server-dev]
"""
import os
import shutil
import socket
import subprocess
import sys
import time

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 2041
WORKDIR = f"/tmp/pion_gh214_test_{PORT}"

# Fixtures so every probed command hits a populated key rather than an early-out
# error path. An error reply can return a different skip count than the success
# path — that WAS gh #156, where the error paths desynced while the success paths
# were correct — so probing empty keys can hide the bug.
FIXTURES = [
    ("SET", "k", "v"),
    ("XADD", "st", "1-1", "f", "v"),
    ("LPUSH", "l", "a"),
    ("HSET", "h", "f", "v"),
    ("SADD", "st2", "m"),
    ("ZADD", "z", "1", "m"),
    ("PFADD", "hll", "a"),
    ("GEOADD", "g", "13.0", "38.0", "p"),
]

# The 19 confirmed-broken probes, plus the families they live in so a regression
# in a sibling subcommand is caught too.
PROBES = [
    ("XRANGE", "st", "-", "+"),
    ("XREVRANGE", "st", "+", "-"),
    ("XINFO", "STREAM", "st"),
    ("XDEL", "st", "9-9"),
    ("XTRIM", "st", "MAXLEN", "5"),
    ("XLEN", "st"),
    ("SINTERCARD", "1", "st2"),
    ("ZREM", "z", "nope"),
    ("ZCARD", "z"),
    ("ACL", "WHOAMI"),
    ("ACL", "LIST"),
    ("ACL", "USERS"),
    ("ACL", "CAT"),
    ("ACL", "LOG"),
    ("DEBUG", "JMAP"),
    ("MEMORY", "USAGE", "k"),
    ("SLOWLOG", "GET"),
    ("LATENCY", "RESET"),
    ("MODULE", "LIST"),
    ("FUNCTION", "LIST"),
    ("SCRIPT", "EXISTS", "abc"),
]

# gh #218 tier 1: VALID minimal forms of variadic commands. Their trailing scan
# (`TOUCH key [key...]`, `UNLINK key [key...]`, `SORT key [options...]`,
# `XACK key group id [id...]`) was bounded by num_tokens — the batch bound — so
# it ate whatever followed in the pipeline. No client bug needed for this one.
VALID_VARIADIC = [
    ("TOUCH", "k"),
    ("SORT", "l"),
    ("UNLINK", "k2"),
    ("XACK", "st", "g", "1-1"),
    ("SDIFF", "st2"),
    ("SINTER", "st2"),
    ("SUNION", "st2"),
    ("ZMSCORE", "z", "m"),
    ("SMISMEMBER", "st2", "m"),
    ("PFCOUNT", "hll"),
]

# gh #218 tier 2: argumentless probes. A wrong-arity command must error, not
# consume its successor. 50 commands used to swallow here, and worse, read the
# next command's tokens as their own arguments.
ARGLESS = [
    "BITFIELD", "CLIENT", "CLUSTER", "CONFIG", "DUMP", "ECHO", "EVAL",
    "EVALSHA", "EXPIRETIME", "FCALL", "GETDEL", "GETEX", "HGETALL", "HKEYS",
    "HLEN", "HRANDFIELD", "HVALS", "KEYS", "PERSIST", "PEXPIRETIME", "PTTL",
    "PUBLISH", "PUBSUB", "SCARD", "SDIFF", "SINTER", "SMEMBERS", "SORT",
    "SPUBLISH", "SRANDMEMBER", "STRLEN", "SUNION", "TOUCH", "TTL", "TYPE",
    "UNLINK", "WATCH", "XACK", "XAUTOCLAIM", "XCLAIM", "XGROUP", "XLEN",
    "XPENDING", "XREAD", "XREADGROUP", "ZCARD", "ZPOPMAX", "ZRANDMEMBER",
]


def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)


def connect(timeout=2.0):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        s = socket.socket()
        try:
            s.settimeout(timeout)
            s.connect(("127.0.0.1", PORT))
            return s
        except OSError:
            s.close()
            time.sleep(0.2)
    raise RuntimeError("server did not come up")


def probe(args):
    """Pipeline `args` + PING in ONE write; return True if the PING is answered.

    Own connection per probe: a swallow leaves the reader waiting, and a
    timed-out socket stays unusable, so a shared client would poison every
    later probe with a false failure.
    """
    s = connect()
    try:
        s.sendall(encode(args) + encode(("PING",)))
        deadline = time.monotonic() + 2.0
        buf = b""
        while time.monotonic() < deadline and b"+PONG" not in buf:
            try:
                chunk = s.recv(65536)
            except (socket.timeout, OSError):
                break
            if not chunk:
                break
            buf += chunk
        return b"+PONG" in buf, buf
    finally:
        s.close()


def main():
    if not os.path.exists(BINARY):
        print(f"binary not found: {BINARY}")
        return 1

    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    proc = subprocess.Popen(
        [os.path.abspath(BINARY), "-p", str(PORT), "-w", "1",
         "--no-auto-detect", "--no-auto-embed"],
        cwd=WORKDIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        s = connect()
        for f in FIXTURES:
            s.sendall(encode(f))
            try:
                s.recv(65536)
            except (socket.timeout, OSError):
                pass
        s.close()

        swallowed = []
        total = 0

        def run(label, args):
            nonlocal total
            total += 1
            ok, reply = probe(args)
            name = " ".join(args)
            if ok:
                print(f"  PASS  [{label}] {name}")
            else:
                swallowed.append((name, reply[:60]))
                print(f"  FAIL  [{label}] {name} swallowed the pipelined PING  "
                      f"reply={reply[:60]!r}")
            return reply

        for args in PROBES:
            run("gh214", args)
        for args in VALID_VARIADIC:
            run("gh218-valid", args)
        for name in ARGLESS:
            run("gh218-argless", (name,))

        # The sharpest gh #218 assertion: a bare ECHO must not read the next
        # command as its argument. It used to reply `$4\r\nPING\r\n` — the
        # pipelined command's own name echoed back.
        bleed = probe(("ECHO",))[1]
        if b"PING" in bleed:
            swallowed.append(("ECHO argument bleed",
                              b"bare ECHO returned the next command: " + bleed[:40]))
            print(f"  FAIL  [gh218-bleed] bare ECHO read the next command: {bleed[:40]!r}")
        else:
            print("  PASS  [gh218-bleed] bare ECHO does not read the next command")

        print(f"\n{total - len(swallowed)}/{total} probes leave the pipeline intact")
        if swallowed:
            print(f"FAILED — {len(swallowed)} probe(s) swallow the next pipelined "
                  f"command (gh #214 / #218 regression)")
            return 1
        print("PASSED")
        return 0
    finally:
        proc.kill()
        proc.wait(timeout=5)
        shutil.rmtree(WORKDIR, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
