#!/usr/bin/env python3
"""Transaction-shape probe — the surface gh #219 did not cover.

gh #156, #162, FT.* (2026-08-05), #214, #218 and #219 are six instances of one
bug class: a dispatch site that miscounts what it consumed, so a pipelined
client's replies stop lining up with its requests. #219 fixed the MULTI queueing
and the EXEC reply-array ordering. This probe walks the transaction shapes that
fix did NOT exercise.

Every check asserts BOTH the reply value and that the connection is still
reply-aligned afterwards (a desync is the failure mode this class produces, and
it hides behind a lenient client).

Reference semantics are Redis 7/8:
  - a queued command that is unknown or has bad arity errors AT QUEUE TIME and
    poisons the transaction; EXEC then replies -EXECABORT and applies nothing
  - a queued command that fails AT RUNTIME (e.g. INCR on a string) yields an
    error ELEMENT inside the array; the other commands still apply
  - WATCH on a key modified before EXEC makes EXEC reply nil (*-1)
  - MULTI inside MULTI errors but leaves the transaction open
  - EXEC/DISCARD outside MULTI error

Usage: python3 tests/test_tx_probe.py [--port 1974] [--binary ./pion-server]
With --binary the probe starts and stops its own server.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

HOST = "127.0.0.1"
PASSED, FAILED, DIVERGE = [], [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))


def skip(name, why):
    """Not applicable on this server configuration — reported, not counted as
    a pass or a failure."""
    print(f"  SKIP  {name}   {why}")


def diverge(name, detail):
    """A Redis-semantics divergence that is not a desync — reported, not fatal."""
    DIVERGE.append(f"{name}: {detail}")
    print(f"  DIFF  {name}   {detail}")


class Conn:
    def __init__(self, port, timeout=5.0):
        self.s = socket.create_connection((HOST, port), timeout=timeout)
        self.s.settimeout(timeout)
        self.f = self.s.makefile("rb")

    def send(self, *cmds):
        """All commands in ONE write — the shape this bug class breaks."""
        buf = b""
        for c in cmds:
            parts = c if isinstance(c, (list, tuple)) else c.split()
            buf += f"*{len(parts)}\r\n".encode()
            for p in parts:
                p = p.encode() if isinstance(p, str) else p
                buf += b"$%d\r\n%s\r\n" % (len(p), p)
        self.s.sendall(buf)

    def send_raw(self, data):
        self.s.sendall(data)

    def read_reply(self):
        line = self.f.readline()
        if not line:
            raise EOFError("connection closed")
        t, body = line[:1], line[1:-2]
        if t == b"+":
            return body.decode()
        if t == b"-":
            return Err(body.decode())
        if t == b":":
            return body.decode()
        if t == b"$":
            n = int(body)
            if n == -1:
                return None
            return self.f.read(n + 2)[:-2].decode()
        if t == b"*":
            n = int(body)
            if n == -1:
                return None
            return [self.read_reply() for _ in range(n)]
        raise ValueError(f"bad RESP type {t!r} in {line!r}")

    def read_n(self, n):
        return [self.read_reply() for _ in range(n)]

    def roundtrip(self, cmd="PING", want="PONG"):
        """Prove the connection is still aligned: this reply must be OUR reply."""
        self.send(cmd)
        return self.read_reply() == want

    def pending(self, wait=0.3):
        self.s.settimeout(wait)
        try:
            return self.s.recv(4096)
        except socket.timeout:
            return b""
        finally:
            self.s.settimeout(5.0)

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


class Err(str):
    """An -ERR reply, so error vs string is never ambiguous in a comparison."""

    __slots__ = ()


def is_err(v, prefix=None):
    return isinstance(v, Err) and (prefix is None or v.startswith(prefix))


# ─────────────────────────── probes ───────────────────────────


def p_watch_pipelined(port):
    """WATCH + MULTI + EXEC in one write, key untouched -> transaction applies."""
    c = Conn(port)
    c.send("SET txp:w1 0")
    c.read_reply()
    c.send("WATCH txp:w1", "MULTI", "SET txp:w1 100", "EXEC")
    r = c.read_n(4)
    check(
        "WATCH+MULTI+EXEC one write: 4 replies, EXEC applies",
        r[0] == "OK" and r[1] == "OK" and r[2] == "QUEUED" and r[3] == ["OK"],
        f"got {r}",
    )
    check("  connection aligned", c.roundtrip())
    c.send("GET txp:w1")
    v = c.read_reply()
    check("  WATCH transaction wrote the key", v == "100", f"got {v!r}")
    c.close()


def p_watch_violated(port):
    """WATCH key, another connection writes it, EXEC must abort with nil.

    REQUIRES `-w 1`. Pion is shared-nothing: with more than one worker each
    owns a private keyspace, so connection B's write may land on a different
    worker and never touch the key A is watching — EXEC then correctly does
    NOT abort, and this probe false-fails. Verified at -w 4: A writes a key, B
    reads nil for it. The probe checks that below rather than reporting a bug
    that isn't one.
    """
    a, b = Conn(port), Conn(port)
    # Do these two connections even share a keyspace? If not, the premise of
    # this probe does not hold on this server.
    a.send("SET txp:w2:probe 1")
    a.read_reply()
    b.send("GET txp:w2:probe")
    if b.read_reply() != "1":
        skip("WATCH violated -> EXEC replies nil",
             "connections are on different workers (shared-nothing); needs -w 1")
        a.close()
        b.close()
        return
    a.send("SET txp:w2 0")
    a.read_reply()
    a.send("WATCH txp:w2")
    a.read_reply()
    b.send("SET txp:w2 999")  # violate the watch from another fd
    b.read_reply()
    a.send("MULTI", "SET txp:w2 111", "EXEC")
    r = a.read_n(3)
    if r[2] is None:
        check("WATCH violated -> EXEC replies nil", True, "got nil")
    else:
        check("WATCH violated -> EXEC replies nil", False, f"got {r[2]!r}")
    check("  connection aligned", a.roundtrip())
    a.send("GET txp:w2")
    v = a.read_reply()
    check("  aborted transaction wrote nothing", v == "999", f"got {v!r}")
    a.close()
    b.close()


def p_unwatch(port):
    """UNWATCH clears the watch, so a later write does not abort EXEC."""
    a, b = Conn(port), Conn(port)
    a.send("SET txp:w3 0")
    a.read_reply()
    a.send("WATCH txp:w3", "UNWATCH")
    a.read_n(2)
    b.send("SET txp:w3 5")
    b.read_reply()
    a.send("MULTI", "SET txp:w3 7", "EXEC")
    r = a.read_n(3)
    check("UNWATCH clears the watch -> EXEC applies", r[2] == ["OK"], f"got {r[2]!r}")
    check("  connection aligned", a.roundtrip())
    a.close()
    b.close()


def p_unknown_command_queued(port):
    """An unknown command inside MULTI: error at queue time, EXEC aborts."""
    c = Conn(port)
    # Clear first: a server that loaded a snapshot from an earlier run still
    # holds these keys, and "was it applied?" then false-fails on stale state.
    c.send("DEL txp:u1")
    c.read_reply()
    c.send("MULTI", "NOSUCHCOMMAND x", "SET txp:u1 1", "EXEC")
    r = c.read_n(4)
    queued_err = is_err(r[1])
    if not queued_err:
        diverge(
            "unknown command inside MULTI",
            f"queue-time reply was {r[1]!r}, Redis errors here",
        )
    if is_err(r[3], "EXECABORT"):
        check("unknown queued command -> EXEC aborts", True, "got EXECABORT")
    else:
        diverge("unknown queued command", f"EXEC replied {r[3]!r}, Redis says EXECABORT")
    check("  connection aligned after abort", c.roundtrip())
    c.send("GET txp:u1")
    v = c.read_reply()
    check(
        "  poisoned transaction applied nothing",
        v is None,
        f"txp:u1={v!r} (Redis: nil — the transaction must not run)",
    )
    c.close()


def p_bad_arity_queued(port):
    """Wrong arity inside MULTI: error at queue time, EXEC aborts."""
    c = Conn(port)
    c.send("DEL txp:u2")  # see the note in the unknown-command probe
    c.read_reply()
    c.send("MULTI", "GET", "SET txp:u2 1", "EXEC")
    r = c.read_n(4)
    if not is_err(r[1]):
        diverge("bad-arity command inside MULTI", f"queue-time reply was {r[1]!r}")
    if not is_err(r[3], "EXECABORT"):
        diverge("bad-arity queued command", f"EXEC replied {r[3]!r}, Redis says EXECABORT")
    check("  connection aligned after arity abort", c.roundtrip())
    c.send("GET txp:u2")
    v = c.read_reply()
    check(
        "  arity-poisoned transaction applied nothing",
        v is None,
        f"txp:u2={v!r} (Redis: nil)",
    )
    c.close()


def p_runtime_error_element(port):
    """A command that fails at RUNTIME yields an error element; others apply."""
    c = Conn(port)
    c.send("SET txp:str hello")
    c.read_reply()
    c.send("MULTI", "INCR txp:str", "SET txp:r1 ok", "EXEC")
    r = c.read_n(4)
    arr = r[3]
    ok_shape = isinstance(arr, list) and len(arr) == 2
    check(
        "runtime-error transaction: array has both elements",
        ok_shape,
        f"got {arr!r}",
    )
    if ok_shape:
        check("  element 0 is an error", is_err(arr[0]), f"got {arr[0]!r}")
        check("  element 1 applied (+OK)", arr[1] == "OK", f"got {arr[1]!r}")
    check("  connection aligned", c.roundtrip())
    c.send("GET txp:r1")
    v = c.read_reply()
    check("  the non-failing command persisted", v == "ok", f"got {v!r}")
    c.close()


def p_nested_multi(port):
    """MULTI inside MULTI errors but leaves the transaction usable."""
    c = Conn(port)
    c.send("MULTI", "MULTI", "SET txp:n1 1", "EXEC")
    r = c.read_n(4)
    check("nested MULTI errors", is_err(r[1]), f"got {r[1]!r}")
    check(
        "nested MULTI leaves the transaction open (EXEC still runs it)",
        r[3] == ["OK"],
        f"got {r[3]!r}",
    )
    check("  connection aligned", c.roundtrip())
    c.close()


def p_exec_discard_without_multi(port):
    """EXEC and DISCARD outside a transaction error, one reply each."""
    c = Conn(port)
    c.send("EXEC", "DISCARD", "PING")
    r = c.read_n(3)
    check("EXEC without MULTI errors", is_err(r[0]), f"got {r[0]!r}")
    check("DISCARD without MULTI errors", is_err(r[1]), f"got {r[1]!r}")
    check("  batch continues past both", r[2] == "PONG", f"got {r[2]!r}")
    check("  connection aligned", c.roundtrip())
    c.close()


def p_empty_transaction(port):
    """MULTI + EXEC with nothing queued -> empty array."""
    c = Conn(port)
    c.send("MULTI", "EXEC", "PING")
    r = c.read_n(3)
    check("empty transaction -> *0", r[1] == [], f"got {r[1]!r}")
    check("  batch continues after empty EXEC", r[2] == "PONG", f"got {r[2]!r}")
    check("  connection aligned", c.roundtrip())
    c.close()


def p_back_to_back_transactions(port):
    """Two complete transactions in ONE write."""
    c = Conn(port)
    c.send(
        "MULTI", "SET txp:b1 1", "EXEC",
        "MULTI", "SET txp:b2 2", "EXEC",
    )
    r = c.read_n(6)
    check(
        "two transactions in one write: 6 replies",
        len(r) == 6 and r[2] == ["OK"] and r[5] == ["OK"],
        f"got {r}",
    )
    check("  connection aligned", c.roundtrip())
    c.send("GET txp:b1", "GET txp:b2")
    v = c.read_n(2)
    check("  both transactions wrote", v == ["1", "2"], f"got {v}")
    c.close()


def p_mixed_fast_and_slow(port):
    """Queue fast-path-eligible AND slow-path-only commands in one transaction."""
    c = Conn(port)
    c.send(
        "MULTI",
        "SET txp:m1 1",        # fast path when not in MULTI
        "INCR txp:m1",         # fast path
        "LPUSH txp:ml a",      # fast path
        "ECHO hi",             # slow path only
        "SETRANGE txp:m1 0 9", # slow path only
        "EXEC",
    )
    r = c.read_n(7)
    arr = r[6]
    check(
        "mixed fast/slow transaction: 5 elements",
        isinstance(arr, list) and len(arr) == 5,
        f"got {arr!r}",
    )
    check("  connection aligned", c.roundtrip())
    c.close()


def p_split_across_writes(port):
    """A transaction split MID-FRAME across two writes must still work."""
    c = Conn(port)
    payload = b""
    for cmd in (["MULTI"], ["SET", "txp:s1", "1"], ["EXEC"]):
        payload += f"*{len(cmd)}\r\n".encode()
        for p in cmd:
            payload += b"$%d\r\n%s\r\n" % (len(p), p.encode())
    cut = len(payload) // 2  # deliberately mid-frame
    c.send_raw(payload[:cut])
    time.sleep(0.15)
    c.send_raw(payload[cut:])
    r = c.read_n(3)
    check(
        "transaction split mid-frame across writes",
        r[0] == "OK" and r[1] == "QUEUED" and r[2] == ["OK"],
        f"got {r}",
    )
    check("  connection aligned", c.roundtrip())
    c.close()


def p_queue_capacity(port):
    """MAX_QUEUED_CMDS: past the cap, does the client still get one reply each?

    slow_path.mojo discards enqueue()'s bool (`_ = self.tx_state.enqueue(...)`)
    and emits +QUEUED regardless, so a 129th command is expected to be accepted
    on the wire but dropped from the array — N+1 requests, N replies, desync.
    """
    n = 200  # > MAX_QUEUED_CMDS (128)
    c = Conn(port)
    cmds = ["MULTI"] + [f"SET txp:cap:{i} {i}" for i in range(n)] + ["EXEC"]
    c.send(*cmds)
    try:
        head = c.read_n(1 + n)  # MULTI + n queue replies
    except (socket.timeout, EOFError) as e:
        check(f"queue cap: {n} queued commands each get a reply", False, f"{type(e).__name__}")
        c.close()
        return
    queued_ok = all(x == "QUEUED" for x in head[1:])
    errs = sum(1 for x in head[1:] if is_err(x))
    check(
        f"queue cap: all {n} queue-time replies present",
        len(head) == n + 1,
        f"got {len(head)}",
    )
    if queued_ok:
        pass
    else:
        print(f"        ({errs} of {n} queue replies were errors — a loud cap)")
    try:
        arr = c.read_reply()  # EXEC
    except (socket.timeout, EOFError) as e:
        check(f"queue cap: EXEC replies at all", False, f"{type(e).__name__}")
        c.close()
        return
    if isinstance(arr, list):
        check(
            f"queue cap: EXEC array has one element per accepted command",
            len(arr) == n,
            f"queued {n} (all +QUEUED), EXEC array has {len(arr)} "
            f"-> {n - len(arr)} commands silently dropped"
            if queued_ok
            else f"array={len(arr)}, {errs} rejected at queue time",
        )
    else:
        check("queue cap: EXEC replies with an array", False, f"got {arr!r}")
    aligned = c.roundtrip()
    check("  connection aligned after cap", aligned)
    if not aligned:
        print("        ^ a client that sent N and got N-k replies is now permanently skewed")
    # Did the dropped commands actually run?
    v = Conn(port)
    v.send(f"GET txp:cap:{n-1}")
    last = v.read_reply()
    check(
        f"  last queued command (#{n}) actually executed",
        last == str(n - 1),
        f"txp:cap:{n-1}={last!r}",
    )
    v.close()
    c.close()


def p_watch_many_keys(port):
    """WATCH beyond MAX_WATCHED_KEYS (16): is the 20th key really watched?

    transaction.mojo's watch_key() returns silently past the cap while WATCH
    still replies +OK, so a key the client believes is watched would not abort
    EXEC — the exact lost update WATCH exists to prevent.
    """
    n = 20  # > MAX_WATCHED_KEYS (16)
    keys = [f"txp:mw:{i}" for i in range(n)]
    a, b = Conn(port), Conn(port)
    a.send(*[f"SET {k} 0" for k in keys])
    a.read_n(n)
    a.send("WATCH " + " ".join(keys))
    w = a.read_reply()
    check(f"WATCH {n} keys replies", w == "OK" or is_err(w), f"got {w!r}")
    # Violate the LAST key — past the cap.
    b.send(f"SET {keys[-1]} 999")
    b.read_reply()
    b.close()
    a.send("MULTI", f"SET {keys[-1]} 111", "EXEC")
    r = a.read_n(3)
    if w == "OK":
        check(
            f"  WATCH honoured for key #{n} (EXEC must abort)",
            r[2] is None,
            f"EXEC returned {r[2]!r} — key #{n} was NOT watched, the update is lost",
        )
    else:
        check(f"  WATCH rejected loudly past the cap", True, f"{w!r}")
    check("  connection aligned", a.roundtrip())
    a.close()


def p_per_fd_isolation(port):
    """Two connections in MULTI at once must not share a queue."""
    a, b = Conn(port), Conn(port)
    a.send("MULTI")
    a.read_reply()
    b.send("MULTI")
    b.read_reply()
    a.send("SET txp:iso:a 1")
    a.read_reply()
    b.send("SET txp:iso:b 2")
    b.read_reply()
    a.send("EXEC")
    ra = a.read_reply()
    b.send("EXEC")
    rb = b.read_reply()
    check(
        "per-fd queues are isolated",
        ra == ["OK"] and rb == ["OK"],
        f"a={ra!r} b={rb!r}",
    )
    v = Conn(port)
    v.send("GET txp:iso:a", "GET txp:iso:b")
    vals = v.read_n(2)
    check("  both fds' transactions applied", vals == ["1", "2"], f"got {vals}")
    v.close()
    a.close()
    b.close()


def p_disconnect_mid_multi(port):
    """A connection dropped mid-MULTI must not leak state onto the next fd."""
    c = Conn(port)
    c.send("MULTI", "SET txp:leak 1")
    c.read_n(2)
    c.close()  # drop without EXEC/DISCARD
    time.sleep(0.2)
    d = Conn(port)
    d.send("PING")
    r = d.read_reply()
    check("fresh connection after mid-MULTI drop is not in MULTI", r == "PONG", f"got {r!r}")
    d.send("SET txp:leak2 1")
    r = d.read_reply()
    check("  fresh connection executes, not queues", r == "OK", f"got {r!r}")
    d.send("GET txp:leak")
    r = d.read_reply()
    check("  dropped transaction did not apply", r is None, f"got {r!r}")
    d.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--binary", default=None, help="start/stop our own server")
    args = ap.parse_args()

    proc = None
    if args.binary:
        proc = subprocess.Popen(
            [args.binary, "-p", str(args.port), "-w", "1",
             "--no-auto-detect", "--no-auto-embed"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        for _ in range(60):
            time.sleep(0.5)
            try:
                Conn(args.port).close()
                break
            except OSError:
                continue

    try:
        Conn(args.port).close()
    except OSError as e:
        print(f"FATAL: no server on {HOST}:{args.port} ({e})")
        return 2

    print(f"transaction-shape probe against {HOST}:{args.port}\n")
    probes = [
        p_watch_pipelined,
        p_watch_violated,
        p_unwatch,
        p_unknown_command_queued,
        p_bad_arity_queued,
        p_runtime_error_element,
        p_nested_multi,
        p_exec_discard_without_multi,
        p_empty_transaction,
        p_back_to_back_transactions,
        p_mixed_fast_and_slow,
        p_split_across_writes,
        p_queue_capacity,
        p_watch_many_keys,
        p_per_fd_isolation,
        p_disconnect_mid_multi,
    ]
    for p in probes:
        print(p.__doc__.splitlines()[0])
        try:
            p(args.port)
        except Exception as e:
            check(p.__name__, False, f"raised {type(e).__name__}: {e}")
        print()

    print(f"{len(PASSED)} passed, {len(FAILED)} failed, {len(DIVERGE)} Redis divergences")
    for f in FAILED:
        print(f"  FAILED:  {f}")
    for d in DIVERGE:
        print(f"  DIVERGE: {d}")

    if proc:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
