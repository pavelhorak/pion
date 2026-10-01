#!/usr/bin/env python3
"""gh #156: one VSET command in must produce exactly one reply out.

`handle_vadd`/`handle_vsim`'s error paths returned a token-skip count that
stopped at the token they rejected on, not at the end of the command. The
leftover tokens — the VALUES scalars, the element name — were re-dispatched by
the slow-path token loop and answered again, so a single rejected command
emitted two frames:

    -ERR dimension mismatch: expected 1536
    -ERR unknown command '1.0'

For a pipelined client that shifts request/reply pairing by one from that point
on. The fix makes the parser's command boundary authoritative at the VSET
dispatch sites (`i = cmd_end_tok - 1`), and hands each handler `cmd_end_tok` as
its bound so optional-argument scans can't read into the next command either.

Making the boundary authoritative surfaced a second desync of the same family:
the parser's command-boundary table held 16 entries while the token array holds
64, so past the 16th command in a batch the dispatch loop fell back to
`cmd_end_tok = num_tokens` and the 17th command swallowed everything behind it
— 20 pipelined commands in, 17 replies out. Covered below too.

Every check counts *frames*, not bytes: a reply-per-command assertion is the
only thing that catches a desync.

Run against a single-worker server — VSET state is per-worker, so the
success-path checks want all connections landing on the same one:

    ./pion-server -p 6390 -w 1
    python3 tests/test_gh156_vset_reply_pairing.py --port 6390
"""

import argparse
import socket
import struct
import sys

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
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


def read_frame(f):
    """Read exactly one RESP frame off a buffered file object.

    Buffered reads, never slicing a growing bytes object: slicing
    a growing buffer is O(N^2) in the reply size.
    """
    line = f.readline()
    if not line:
        return None
    t = line[:1]
    if t in (b"+", b"-", b":", b","):
        return line
    if t == b"$":
        n = int(line[1:-2])
        if n == -1:
            return line
        return line + f.read(n + 2)
    if t in (b"*", b"~", b">"):
        n = int(line[1:-2])
        out = line
        for _ in range(max(n, 0)):
            sub = read_frame(f)
            if sub is None:
                return out
            out += sub
        return out
    return line


def frames(f, count):
    """Read up to `count` frames; stop early on timeout (a missing reply).

    Callers ask for one more frame than the command count, so a spurious extra
    reply shows up as a longer list and a missing one as a shorter list.
    """
    out = []
    for _ in range(count):
        try:
            fr = read_frame(f)
        except (socket.timeout, TimeoutError, OSError):
            break
        if fr is None:
            break
        out.append(fr)
    return out


def session(port, timeout=2.0):
    s = connect(port, timeout)
    return s, s.makefile("rb")


# ── VADD ──────────────────────────────────────────────────────────────────────


def test_vadd_dim_mismatch_single_reply(port):
    """The repro from the issue: a wrong-dim VALUES command, one reply."""
    s, f = session(port)
    # Establish the index dim with a well-formed insert first, so the mismatch
    # branch is the one that fires (hnsw.dim > 0 and dim != hnsw.dim).
    dim = 1536
    s.sendall(cmd("VADD", "gh156:vs", b"FP32", struct.pack(f"<{dim}f", *([0.25] * dim)), "seed"))
    read_frame(f)

    s.sendall(cmd("VADD", "gh156:vs", "VALUES", "4", "1.0", "2.0", "3.0", "4.0", "s1"))
    got = frames(f, 2)
    check(
        "VADD VALUES dim mismatch: exactly one reply frame",
        len(got) == 1,
        f"got {len(got)} frames: {got!r} — the extra frame is the gh #156 desync",
    )
    if got:
        check(
            "VADD VALUES dim mismatch: the reply is the dimension error",
            got[0].startswith(b"-ERR") and b"dimension mismatch" in got[0],   # Redis's wording since gh #366
            f"got {got[0]!r}",
        )
    s.close()


def test_vadd_error_keeps_pipeline_paired(port):
    """A rejected VADD in mid-pipeline must not shift the replies that follow."""
    s, f = session(port)
    s.sendall(
        cmd("SET", "gh156:a", "one")
        + cmd("VADD", "gh156:vs", "VALUES", "4", "1.0", "2.0", "3.0", "4.0", "s2")
        + cmd("GET", "gh156:a")
    )
    got = frames(f, 4)
    check(
        "pipelined VADD error: three commands produce three replies",
        len(got) == 3,
        f"got {len(got)} frames: {got!r}",
    )
    check(
        "pipelined VADD error: the reply after it is still GET's",
        len(got) == 3 and got[0] == b"+OK\r\n" and got[2] == b"$3\r\none\r\n",
        f"got {got!r} — a shift here is the reply-pairing bug",
    )
    s.close()


def test_vadd_bad_format_single_reply(port):
    """The `VADD requires FP32 or VALUES` branch has the same shape."""
    s, f = session(port)
    s.sendall(cmd("VADD", "gh156:vs", "BOGUS", "4", "1.0", "x"))
    got = frames(f, 2)
    check(
        "VADD unknown format: exactly one reply frame",
        len(got) == 1,
        f"got {len(got)} frames: {got!r}",
    )
    s.close()


def test_vadd_not_enough_values_single_reply(port):
    """`not enough values`: the declared dim overruns the tokens present."""
    s, f = session(port)
    s.sendall(cmd("VADD", "gh156:vs", "VALUES", "8", "1.0", "2.0", "3.0", "elem"))
    got = frames(f, 2)
    check(
        "VADD short VALUES list: exactly one reply frame",
        len(got) == 1,
        f"got {len(got)} frames: {got!r}",
    )
    s.close()


# ── VSIM ──────────────────────────────────────────────────────────────────────


def test_vsim_values_single_reply(port):
    """VSIM's VALUES branch: one reply whatever the dim."""
    s, f = session(port)
    s.sendall(cmd("VSIM", "gh156:vs", "VALUES", "4", "1.0", "2.0", "3.0", "4.0", "COUNT", "3"))
    got = frames(f, 2)
    check(
        "VSIM VALUES: exactly one reply frame",
        len(got) == 1,
        f"got {len(got)} frames: {got!r}",
    )
    s.close()


def test_vsim_bad_format_single_reply(port):
    s, f = session(port)
    s.sendall(cmd("VSIM", "gh156:vs", "BOGUS", "1.0", "COUNT", "3"))
    got = frames(f, 2)
    check(
        "VSIM unknown format: exactly one reply frame",
        len(got) == 1,
        f"got {len(got)} frames: {got!r}",
    )
    s.close()


def test_vsim_options_dont_eat_next_command(port):
    """VSIM's TOPN/WITHSCORES scan used to run to the end of the *buffer*.

    Pipelined behind a VSIM, the following command's tokens were swallowed by
    the option loop and never answered.
    """
    s, f = session(port)
    dim = 1536
    s.sendall(cmd("VADD", "gh156:vs", b"FP32", struct.pack(f"<{dim}f", *([0.5] * dim)), "seed2"))
    read_frame(f)

    blob = struct.pack(f"<{dim}f", *([0.5] * dim))
    s.sendall(cmd("VSIM", "gh156:vs", b"FP32", blob, "COUNT", "2") + cmd("PING"))
    got = frames(f, 3)
    check(
        "VSIM + PING pipelined: two replies, PING answered",
        len(got) == 2 and got[1] == b"+PONG\r\n",
        f"got {got!r} — a swallowed PING is the option-scan overrun",
    )
    s.close()


# ── Neighbouring VSET commands with optional args ─────────────────────────────


def test_vrandmember_no_count_doesnt_eat_next(port):
    """VRANDMEMBER's arity probe was `i + 2 < num_tokens` (buffer-wide)."""
    s, f = session(port)
    s.sendall(cmd("VRANDMEMBER", "gh156:vs") + cmd("PING"))
    got = frames(f, 3)
    check(
        "VRANDMEMBER without count + PING: two replies",
        len(got) == 2 and got[1] == b"+PONG\r\n",
        f"got {got!r}",
    )
    s.close()


def test_vemb_without_raw_doesnt_eat_next(port):
    """VEMB's RAW probe had the same buffer-wide bound."""
    s, f = session(port)
    s.sendall(cmd("VEMB", "gh156:vs", "seed") + cmd("PING"))
    got = frames(f, 3)
    check(
        "VEMB without RAW + PING: two replies",
        len(got) == 2 and got[1] == b"+PONG\r\n",
        f"got {got!r}",
    )
    s.close()


# ── Success paths must be untouched ───────────────────────────────────────────


def test_success_paths_unchanged(port):
    """Correct-dim traffic was never affected; prove it stayed that way."""
    s, f = session(port)
    dim = 1536
    blob = struct.pack(f"<{dim}f", *([0.75] * dim))
    s.sendall(cmd("VADD", "gh156:ok", b"FP32", blob, "e1"))
    r = read_frame(f)
    check("VADD FP32 success still returns :1", r == b":1\r\n", f"got {r!r}")

    s.sendall(cmd("VCARD", "gh156:ok") + cmd("VDIM", "gh156:ok") + cmd("PING"))
    got = frames(f, 4)
    check(
        "VCARD/VDIM/PING pipelined: three replies, PING last",
        len(got) == 3 and got[0].startswith(b":") and got[2] == b"+PONG\r\n",
        f"got {got!r}",
    )
    s.close()


# ── Deep pipelines (the cmd_ends cap) ────────────────────────────────────────


def test_deep_pipeline_every_command_answered(port):
    """The command-boundary table used to hold 16 entries.

    Past that the dispatch loop fell back to `cmd_end_tok = num_tokens`, so the
    17th command in a batch consumed every command behind it — 20 unknown
    commands in, 17 replies out. Every handler that skips with
    `i = cmd_end_tok - 1` was exposed, which after this fix includes all of
    VSET. Uses an unknown command because it is the shortest slow-path route
    that takes the cmd_end_tok skip.
    """
    for n in (17, 20, 40):
        s, f = session(port)
        s.sendall(b"".join(cmd("GH156X", "k%d" % i) for i in range(n)))
        got = frames(f, n + 2)
        check(
            f"{n} pipelined slow-path commands: {n} replies",
            len(got) == n,
            f"got {len(got)} replies for {n} commands — the batch tail was swallowed",
        )
        s.close()


def test_inline_batch_every_command_answered(port):
    """Inline (telnet-style) commands never registered a command boundary.

    `num_cmds` stayed 0 for an all-inline batch, so every cmd_end_tok handler
    saw the end of the whole batch as its own end.
    """
    s, f = session(port)
    n = 5
    s.sendall(b"".join(b"GH156X k%d\r\n" % i for i in range(n)))
    got = frames(f, n + 2)
    check(
        f"{n} inline slow-path commands: {n} replies",
        len(got) == n,
        f"got {len(got)} replies for {n} inline commands",
    )
    s.close()

    s, f = session(port)
    s.sendall(b"VCARD gh156:vs\r\nVDIM gh156:vs\r\nPING\r\n")
    got = frames(f, 4)
    check(
        "inline VCARD/VDIM/PING: three replies, PING last",
        len(got) == 3 and got[2] == b"+PONG\r\n",
        f"got {got!r}",
    )
    s.close()


def test_deep_pipeline_vadd(port):
    """The VSET ingest shape: VADD FP32 pipelined deeper than the old cap."""
    s, f = session(port)
    dim = 1536
    blob = struct.pack(f"<{dim}f", *([0.125] * dim))
    n = 20
    s.sendall(b"".join(cmd("VADD", "gh156:deep", b"FP32", blob, "d%d" % i) for i in range(n)))
    got = frames(f, n + 2)
    check(
        f"{n} pipelined VADD FP32: {n} replies",
        len(got) == n,
        f"got {len(got)} replies for {n} VADDs",
    )
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()

    print(f"gh #156 VSET reply pairing / token skip — port {args.port}")
    for fn in (
        test_vadd_dim_mismatch_single_reply,
        test_vadd_error_keeps_pipeline_paired,
        test_vadd_bad_format_single_reply,
        test_vadd_not_enough_values_single_reply,
        test_vsim_values_single_reply,
        test_vsim_bad_format_single_reply,
        test_vsim_options_dont_eat_next_command,
        test_vrandmember_no_count_doesnt_eat_next,
        test_vemb_without_raw_doesnt_eat_next,
        test_deep_pipeline_every_command_answered,
        test_inline_batch_every_command_answered,
        test_deep_pipeline_vadd,
        test_success_paths_unchanged,
    ):
        try:
            fn(args.port)
        except Exception as e:  # noqa: BLE001
            check(fn.__name__, False, f"exception: {e!r}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for x in FAIL:
        print(f"  FAILED: {x}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
