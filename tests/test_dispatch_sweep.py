#!/usr/bin/env python3
"""Dispatch integrity across the ENTIRE command surface.

gh #156, #162, FT.* (2026-08-05), #214, #218, #219 and #221 are all one bug
class: a dispatch site that miscounts what it consumed, or matches the wrong
name/length, so a pipelined client's replies stop lining up with its requests.
Every previous sweep covered a hand-picked subset — gh #214's probe reached 57
commands across 186 sites, "under a third of the surface", and gh #218 then
found two more tiers in the part it had not reached.

This sweep takes its command list from `src/commands/command_table.mojo`, which
is generated from the dispatch chains themselves (gh #220), so coverage is the
whole surface by construction and grows automatically when a command is added.

For each command it sends `<CMD>` with no arguments followed by `PING` in ONE
write, and requires exactly two replies with `+PONG` second. That is the
minimal property every dispatch site must hold: answer your own command, and
consume exactly your own command. An argumentless command should produce an
arity error — what it must never do is stay silent (the client then pairs
PING's reply with it) or swallow the PING.

Usage: python3 tests/test_dispatch_sweep.py [--port 1974] [--verbose]
"""

import argparse
import re
import socket
import sys
from pathlib import Path

HOST = "127.0.0.1"
TABLE = Path(__file__).resolve().parents[1] / "src" / "commands" / "command_table.mojo"

# Commands whose no-arg form legitimately changes connection or server state,
# so probing them would corrupt the sweep rather than test it. Each is excluded
# for a reason, not for being inconvenient.
SKIP = {
    # destructive / server lifecycle
    "flushall", "flushdb", "shutdown", "save", "bgsave", "bgrewriteaof",
    "debug", "failover", "migrate", "reset",
    # connection lifecycle or protocol mode
    "quit", "hello", "auth", "select", "swapdb", "subscribe", "psubscribe",
    "ssubscribe", "unsubscribe", "punsubscribe", "sunsubscribe", "monitor",
    "asking", "readonly", "readwrite", "client",
    # transaction state (covered by test_tx_probe.py)
    "multi", "exec", "discard", "watch", "unwatch",
    # replication stream: these hijack the connection
    "sync", "psync", "replconf", "replicaof", "slaveof",
    # blocking commands — a no-arg form is an arity error, but a partially
    # accepted one would park the connection for the timeout
    "blpop", "brpop", "blmove", "blmpop", "bzmpop", "bzpopmin", "bzpopmax",
    "brpoplpush", "wait", "waitaof",
}


def load_commands():
    if not TABLE.exists():
        print(f"FATAL: {TABLE} missing — run `python3 tools/gen_command_table.py`")
        sys.exit(2)
    names = re.findall(r'_cmd_eq_ci\(tp, tl, "([^"]+)"\)', TABLE.read_text())
    return sorted(set(names))


class Conn:
    def __init__(self, port):
        self.s = socket.create_connection((HOST, port), timeout=5)
        self.s.settimeout(5)
        self.f = self.s.makefile("rb")

    def hello3(self):
        """Bind this connection to RESP3. Returns True if the server agreed."""
        self.s.sendall(b"*2\r\n$5\r\nHELLO\r\n$1\r\n3\r\n")
        try:
            self.read()          # map reply; the reader handles % natively
            return True
        except Exception:
            return False

    def send_raw(self, *cmds):
        buf = b""
        for parts in cmds:
            buf += f"*{len(parts)}\r\n".encode()
            for p in parts:
                buf += b"$%d\r\n%s\r\n" % (len(p), p.encode())
        self.s.sendall(buf)

    def read(self):
        line = self.f.readline()
        if not line:
            raise EOFError("connection closed")
        t, body = line[:1], line[1:-2]
        if t in b"+-:":
            return body.decode()
        if t == b"$":
            n = int(body)
            return None if n == -1 else self.f.read(n + 2)[:-2].decode()
        if t == b"*":
            n = int(body)
            if n == -1:
                return None
            return [self.read() for _ in range(n)]
        if t in b"%~>":  # RESP3 map / set / push
            n = int(body)
            mult = 2 if t == b"%" else 1
            return [self.read() for _ in range(n * mult)]
        # Remaining RESP3 scalars. `_` (null) is the one that matters here:
        # under HELLO 3 every appender that emits a RESP2 `$-1` emits `_`
        # instead, so a reader that does not know it reports a protocol error
        # for a perfectly correct reply — which is a bug in the test, not the
        # server, and would have looked like 14 command failures.
        if t == b"_":            # null
            return None
        if t == b",":            # double
            return body.decode()
        if t == b"#":            # boolean
            return body.decode()
        if t == b"(":            # big number
            return body.decode()
        if t == b"!":            # blob error
            n = int(body)
            return "ERR:" + self.f.read(n + 2)[:-2].decode()
        if t == b"=":            # verbatim string
            n = int(body)
            return self.f.read(n + 2)[:-2].decode()
        raise ValueError(f"unparseable RESP {line!r}")

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


# Commands the main sweep SKIPs for state reasons, but whose ARGUMENT HANDLING
# is still safe to probe — they take no arguments, so sending one must produce
# a reply and consume the whole frame. This exists because the skip list means
# "untested": six of these arms were modified for gh #223 and nothing covered
# them until a hand probe caught the gap.
SKIPPED_BUT_PROBEABLE = ["RESET", "READONLY", "READWRITE", "ASKING", "DBSIZE"]


def probe_skipped_arms(port):
    print("\n[skip-list arms] no-arg commands given an argument must still "
          "reply once and consume the frame")
    bad = []
    for cmd in SKIPPED_BUT_PROBEABLE:
        for extra, shape in (([], "0 args"), (["extra"], "1 arg")):
            try:
                c = Conn(port)
                c.send_raw([cmd] + extra, ["PING"])
                r1 = c.read()
                r2 = c.read()
                c.close()
                if r2 != "PONG":
                    bad.append(f"{cmd} [{shape}]: second reply {r2!r} (first {r1!r})")
                    print(f"  FAIL {cmd:12s} [{shape:6s}] second reply {r2!r}", flush=True)
                else:
                    print(f"  ok   {cmd:12s} [{shape:6s}] -> {str(r1)[:28]}", flush=True)
            except Exception as e:
                bad.append(f"{cmd} [{shape}]: {type(e).__name__}: {e}")
                print(f"  FAIL {cmd:12s} [{shape:6s}] {type(e).__name__}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--resp3", action="store_true",
                    help="bind each probe connection to RESP3 via HELLO 3 first. "
                         "gh #172/#177 added a RESP3 branch to every divergent "
                         "appender; this runs the same 576 probes through them.")
    args = ap.parse_args()

    cmds = load_commands()
    probed = [c for c in cmds if c not in SKIP]
    print(f"dispatch sweep: {len(cmds)} commands in the table, "
          f"{len(probed)} probed, {len(cmds) - len(probed)} skipped by policy")

    # Two shapes, because they reach different code.
    #
    #   0 args — the command's own arity guard fires (or fails to).
    #   1 arg  — a command needing >= 2 passes its first guard and then reads a
    #            SECOND token, which in a pipelined batch is the next command's
    #            name. gh #218 called this the arity-error tier and found it
    #            reading the neighbouring command; a zero-arg probe cannot reach
    #            it, because the first guard stops the arm before the second read.
    #   2 args — gh #238: a command whose OPTIONAL argument is the second one.
    #            `ZPOPMIN key` is correct and `ZPOPMIN` errors correctly, so both
    #            shapes above pass — but `ZPOPMIN key 2` neither honoured nor
    #            CONSUMED the count, leaving the "2" to be dispatched as its own
    #            command: one command in, two replies out. Any arm that stops
    #            consuming at its required arity is invisible until a probe
    #            supplies an optional argument, so the third shape is not
    #            redundant with the second.
    #
    # The probe arguments are keys nobody else uses, so a command that does
    # execute cannot disturb another probe. The second argument is "1" because
    # optional second args are overwhelmingly counts/indices, and a numeric token
    # reaches the arm's parse rather than being rejected as a bad type.
    #   3 args — gh #239 showed the 2-arg shape found 7 desyncs that 0/1 could
    #            not reach. The same argument applies one level further: any arm
    #            whose optional argument is the THIRD (SET k v EX, LPOS k e RANK,
    #            GETRANGE k 0 -1) stops consuming at its required arity and is
    #            invisible to all three shapes above.
    shapes = [([], "0 args"), (["gh223:probe:key"], "1 arg"),
              (["gh223:probe:key", "1"], "2 args"),
              (["gh223:probe:key", "1", "1"], "3 args"),
              #   4 args — gh #240 made under-consumption structurally impossible,
              #            so a finding here is a WRONG REPLY rather than a desync.
              #            Still worth probing: each added shape so far has found
              #            real bugs (2 args -> 7, 3 args -> 6).
              (["gh223:probe:key", "1", "1", "1"], "4 args")]
    print(f"shapes: {', '.join(s[1] for s in shapes)}"
          + ("   protocol: RESP3 (HELLO 3)" if args.resp3 else "   protocol: RESP2") + "\n")

    failures = []
    last_ok = None
    for cmd in probed:
        # Fresh connection per command: a desync must not cascade into the next
        # command's verdict, or one bug hides every result after it.
        # Report each failure AS IT HAPPENS. Collecting them for a summary at
        # the end loses everything if the server dies mid-sweep — which is
        # exactly when the output matters most.
        def fail(shape, why):
            failures.append((f"{cmd} [{shape}]", why))
            print(f"  FAIL {cmd:24s} [{shape:6s}] {why}", flush=True)

        for extra, shape in shapes:
            try:
                c = Conn(args.port)
                if args.resp3:
                    c.hello3()
            except OSError as e:
                print(f"\nFATAL: cannot connect ({e}) while probing {cmd!r} [{shape}].")
                print(f"       Last command answered normally: {last_ok!r}")
                print(f"       {len(failures)} failure(s) recorded before this point.")
                print("       Check the server's stderr log for a Mojo ABORT trace and\n"
                      "       pion-<port>.crash.log for a signal backtrace.")
                return 2
            try:
                c.send_raw([cmd.upper()] + extra, ["PING"])
                r1 = c.read()
                r2 = c.read()
                if r2 != "PONG":
                    fail(shape, f"second reply was {r2!r}, not PONG "
                                f"(first was {r1!r}) — command consumed the PING")
                else:
                    last_ok = f"{cmd} [{shape}]"
                    if args.verbose:
                        print(f"  ok   {cmd:24s} [{shape:6s}] -> {str(r1)[:44]}", flush=True)
            except (EOFError, ConnectionResetError) as e:
                fail(shape, f"NO REPLY / server closed ({type(e).__name__}) — crash or silent handler")
            except socket.timeout:
                fail(shape, "TIMEOUT — no reply emitted, connection desynced")
            except Exception as e:
                fail(shape, f"{type(e).__name__}: {e}")
            finally:
                c.close()

    failures += [(c, w) for c, w in
                 ((f.split(':')[0], f) for f in probe_skipped_arms(args.port))]

    total = len(probed) * len(shapes) + len(SKIPPED_BUT_PROBEABLE) * len(shapes)
    print(f"\n{total - len(failures)}/{total} probes kept the pipeline intact "
          f"({len(probed)} commands x {len(shapes)} shapes)")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for cmd, why in failures:
            print(f"  {cmd:24s} {why}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
