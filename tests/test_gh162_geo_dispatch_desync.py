#!/usr/bin/env python3
"""gh #162: GEO command dispatch — reply pairing and batch survival.

Two defects, both of the gh #156 token-skip family:

  1. GEOSEARCHSTORE and GEORADIUSBYMEMBER matched on length + first byte only
     (`tl == N and (tp[0]|0x20) == 'g'`), so any 14- or 17-byte token starting
     with 'g' was routed into a GEO handler. An unknown command like
     `GXXXXXXXXXXXXX` came back as a GEO arity error instead of
     `-ERR unknown command`.

  2. Every GEO handler's error paths returned a token-skip count that stopped
     at the token they rejected on, not the end of the command. A rejected
     `GEOADD k` left the key to be re-dispatched as its own command — one
     command in, two replies out. Pipelined, the desync shifts request/reply
     pairing by one, and deep enough it mis-frames a bulk length and trips the
     slow-path catch-all, which used to drop the whole recv buffer.

The fix makes the parser's command boundary authoritative at every GEO dispatch
site (`i = cmd_end_tok - 1`) and matches the full command name; the catch-all
now recovers at command granularity instead of dropping the batch.

Every check counts *frames*, not bytes — a reply-per-command assertion is the
only thing that catches a desync.

Run against a single-worker server:

    ./pion-server -p 6390 -w 1
    python3 tests/test_gh162_geo_dispatch_desync.py --port 6390
"""

import argparse
import socket
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
    """Read exactly one RESP frame off a buffered file object."""
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
    """Read up to `count` frames; stop early on timeout (a missing reply)."""
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


# ── Unknown g* no longer routes into a GEO handler ────────────────────────────


def test_unknown_14byte_g_is_unknown_command(port):
    """A 14-byte 'g*' token used to become a GEOSEARCHSTORE arity error."""
    s, f = session(port)
    s.sendall(cmd("GXXXXXXXXXXXXX", "k"))  # 14 bytes, starts with 'g'
    got = frames(f, 2)
    check(
        "unknown 14-byte g* command: exactly one reply",
        len(got) == 1,
        f"got {len(got)} frames: {got!r}",
    )
    if got:
        check(
            "unknown 14-byte g* command: it's -ERR unknown command",
            got[0].startswith(b"-ERR unknown command"),
            f"got {got[0]!r} — a GEO arity error means the loose match is back",
        )
    s.close()


def test_unknown_17byte_g_is_unknown_command(port):
    """A 17-byte 'g*' token used to become a GEORADIUSBYMEMBER arity error."""
    s, f = session(port)
    s.sendall(cmd("GXXXXXXXXXXXXXXXX", "k"))  # 17 bytes
    got = frames(f, 2)
    check(
        "unknown 17-byte g* command: exactly one reply",
        len(got) == 1,
        f"got {len(got)} frames: {got!r}",
    )
    if got:
        check(
            "unknown 17-byte g* command: it's -ERR unknown command",
            got[0].startswith(b"-ERR unknown command"),
            f"got {got[0]!r}",
        )
    s.close()


def test_unknown_g_deep_pipeline_all_answered(port):
    """The batch-loss repro: N unknown g* commands, N replies (not 1)."""
    for n in (17, 30):
        s, f = session(port)
        s.sendall(b"".join(cmd("GXXXXXXXXXXXXX", "k%d" % j) for j in range(n)))
        got = frames(f, n + 2)
        check(
            f"{n} unknown g* commands pipelined: {n} replies",
            len(got) == n,
            f"got {len(got)} — the batch tail was swallowed",
        )
        s.close()


# ── GEO arity errors: one command, one reply ──────────────────────────────────


def test_geo_arity_errors_single_reply(port):
    """Each GEO command's arity-error path must consume its whole command."""
    cases = [
        ("GEOADD", ("GEOADD", "k")),
        ("GEOPOS", ("GEOPOS",)),
        ("GEODIST", ("GEODIST", "k")),
        ("GEOHASH", ("GEOHASH",)),
        ("GEORADIUS", ("GEORADIUS", "k")),
        ("GEOSEARCH", ("GEOSEARCH", "k")),
        ("GEOSEARCHSTORE", ("GEOSEARCHSTORE", "dst")),
        ("GEORADIUSBYMEMBER", ("GEORADIUSBYMEMBER", "k")),
    ]
    for label, args in cases:
        s, f = session(port)
        s.sendall(cmd(*args))
        got = frames(f, 2)
        check(
            f"{label} arity error: exactly one reply frame",
            len(got) == 1,
            f"got {len(got)} frames: {got!r} — the extra frame is the gh #162 desync",
        )
        s.close()


def test_geo_error_keeps_pipeline_paired(port):
    """A rejected GEO command mid-pipeline must not shift what follows."""
    for label, args in [
        ("GEOADD", ("GEOADD", "k")),
        ("GEOSEARCHSTORE", ("GEOSEARCHSTORE", "dst")),
        ("GEORADIUSBYMEMBER", ("GEORADIUSBYMEMBER", "k")),
    ]:
        s, f = session(port)
        s.sendall(cmd("SET", "gh162:a", "one") + cmd(*args) + cmd("GET", "gh162:a"))
        got = frames(f, 4)
        check(
            f"pipelined {label} error: three commands, three replies",
            len(got) == 3,
            f"got {len(got)}: {got!r}",
        )
        check(
            f"pipelined {label} error: GET still paired to its reply",
            len(got) == 3 and got[0] == b"+OK\r\n" and got[2] == b"$3\r\none\r\n",
            f"got {got!r} — a shift is the reply-pairing bug",
        )
        s.close()


def test_deep_pipeline_mixed_geo_errors(port):
    """Many rejected GEO commands interleaved with pings: all answered, paired."""
    s, f = session(port)
    payload = b""
    n = 25
    for j in range(n):
        payload += cmd("GEOSEARCHSTORE", "dst%d" % j)  # arity error each
        payload += cmd("PING")
    s.sendall(payload)
    got = frames(f, 2 * n + 2)
    check(
        "25×(GEOSEARCHSTORE-error + PING): 50 replies",
        len(got) == 2 * n,
        f"got {len(got)} replies for {2 * n} commands",
    )
    pongs = [g for g in got if g == b"+PONG\r\n"]
    check(
        "25×(GEOSEARCHSTORE-error + PING): every PING answered, in order",
        len(pongs) == n and all(got[2 * j + 1] == b"+PONG\r\n" for j in range(n)),
        f"got {len(pongs)} PONGs; odd slots must all be PONG",
    )
    s.close()


# ── GEO happy paths unchanged ─────────────────────────────────────────────────


def test_geo_happy_path(port):
    """GEOADD success + a GEO read command followed by PING still frame-sync.

    A fresh connection per assertion: `frames()` deliberately over-reads by one
    to catch a spurious extra reply, and that trailing read times out on a
    makefile — reusing the same reader afterwards would surface the timeout, not
    a server desync. GEO state is server-side and per-worker, so it persists
    across the reconnects. The GEO read replies are nil here (member lookup is a
    separate known SSO issue); what this asserts is that the trailing PING is
    still paired, i.e. the authoritative skip consumed exactly one command.
    """
    s, f = session(port)
    s.sendall(cmd("DEL", "gh162:geo"))
    read_frame(f)
    s.sendall(
        cmd("GEOADD", "gh162:geo", "13.361389", "38.115556", "Palermo")
        + cmd("GEOADD", "gh162:geo", "15.087269", "37.502669", "Catania")
    )
    got = frames(f, 3)
    check(
        "two GEOADDs: two integer replies",
        len(got) == 2 and got[0].startswith(b":") and got[1].startswith(b":"),
        f"got {got!r}",
    )
    s.close()

    for label, geo in [
        ("GEODIST", cmd("GEODIST", "gh162:geo", "Palermo", "Catania")),
        ("GEOPOS", cmd("GEOPOS", "gh162:geo", "Palermo")),
        ("GEOSEARCH", cmd("GEOSEARCH", "gh162:geo", "FROMMEMBER", "Palermo", "BYRADIUS", "200", "km", "ASC")),
    ]:
        s, f = session(port)
        s.sendall(geo + cmd("PING"))
        got = frames(f, 3)
        check(
            f"{label} + PING: two replies, PING answered",
            len(got) == 2 and got[1] == b"+PONG\r\n",
            f"got {got!r}",
        )
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()

    print(f"gh #162 GEO dispatch desync — port {args.port}")
    for fn in (
        test_unknown_14byte_g_is_unknown_command,
        test_unknown_17byte_g_is_unknown_command,
        test_unknown_g_deep_pipeline_all_answered,
        test_geo_arity_errors_single_reply,
        test_geo_error_keeps_pipeline_paired,
        test_deep_pipeline_mixed_geo_errors,
        test_geo_happy_path,
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
