#!/usr/bin/env python3
"""Stream entries keep every byte, and a pipelined XREAD BLOCK is answered in
order.

CHECKS
  1. Fields and values of 65,535, 65,536, 70,000 bytes and 1 MiB come back
     whole from XRANGE, XREVRANGE and XREAD, and so does a small field that
     follows a large one in the same entry.
  2. They come back whole after a restart that replays the WAL, and after one
     that loads a snapshot (SAVE), and the stream keeps its last id.
  3. A replica holds them whole: entries written before it joined (sent in
     its full resync) and entries written after (streamed from the WAL).
  4. XREAD BLOCK woken by large entries returns them all, whole: ten 20 KB
     entries from one MULTI/EXEC arrive in one reply, and so does a 1 MiB one.
  5. Commands pipelined behind XREAD BLOCK are answered after it: on a timeout
     (a null array, then PONG), on a wake, and under RESP3 (`_`, then PONG).
     Nothing arrives while the reader is blocked.
  6. A reader that disconnects while blocked is forgotten: the server keeps
     serving, and a new connection that may reuse its fd gets no reply it did
     not ask for.

    python3 tests/test_stream_entries.py [--port 6441]
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import (Conn, RespProtocolError, encode, wait_ready_pid,  # noqa: E402
                         wait_port_free)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.abspath(os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
SIZES = [65535, 65536, 70000, 1 << 20]
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def blob(n, seed):
    return random.Random(seed).randbytes(n)


class Server:
    def __init__(self, work, port, sub, extra=()):
        self.port, self.dir, self.extra = port, os.path.join(work, sub), list(extra)
        os.makedirs(self.dir, exist_ok=True)
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen([BIN, "-p", str(self.port), "-w", "1", "--no-crash-log",
                                      "--no-auto-detect", "--no-auto-embed"] + self.extra,
                                     cwd=self.dir, stdout=open(os.path.join(self.dir, "log"), "a"),
                                     stderr=subprocess.STDOUT)
        wait_ready_pid(self.port, self.proc, 60)
        return Conn(self.port, timeout=30)

    def stop(self, sig=signal.SIGKILL):
        if self.proc:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            wait_port_free(self.port)


# ── the data every section checks ────────────────────────────────────────────
def expected_entries():
    """{key: [(field, value), ...]} — one stream per size, plus one entry whose
    small fields sit around a large one."""
    want = {}
    for n in SIZES:
        want[f"{{s}}v{n}"] = [(b"f", blob(n, n))]
        want[f"{{s}}f{n}"] = [(blob(n, n + 1), b"v")]
    want["{s}mixed"] = [(b"a", b"1"), (b"big", blob(70000, 7)), (b"z", b"2")]
    return want


def write_entries(c, want, suffix=""):
    ids = {}
    for key, pairs in want.items():
        args = ["XADD", key + suffix, "*"]
        for f, v in pairs:
            args += [f, v]
        r = c.cmd(*args)
        ids[key + suffix] = r
    return ids


def flat(pairs):
    out = []
    for f, v in pairs:
        out += [f, v]
    return out


def entries_match(c, want, ids, suffix=""):
    """Every stream holds exactly its one entry, by XRANGE and XREVRANGE.
    Returns the keys that do not."""
    bad = []
    for key, pairs in want.items():
        k = key + suffix
        exp = [[ids[k], flat(pairs)]]
        if c.cmd("XRANGE", k, "-", "+") != exp or c.cmd("XREVRANGE", k, "+", "-") != exp:
            bad.append(k)
    return bad


def describe(c, key):
    r = c.cmd("XRANGE", key, "-", "+")
    if not isinstance(r, list) or not r:
        return repr(r)[:80]
    return ", ".join(f"{len(x)} B" for x in r[0][1])


# ── sections ─────────────────────────────────────────────────────────────────
def section_roundtrip_and_restart(work, port):
    print("[1] large fields and values round-trip")
    srv = Server(work, port, "restart")
    try:
        _roundtrip_and_restart(srv)
    finally:
        srv.stop()


def _roundtrip_and_restart(srv):
    c = srv.start()
    want = expected_entries()
    ids = write_entries(c, want)
    check("XADD answered an id for every entry",
          all(isinstance(v, bytes) and b"-" in v for v in ids.values()), repr(ids)[:200])
    bad = entries_match(c, want, ids)
    check("XRANGE and XREVRANGE return every byte", not bad,
          "; ".join(f"{k}: {describe(c, k)}" for k in bad[:4]))
    keys = list(want)
    r = c.cmd("XREAD", "STREAMS", *keys, *(["0"] * len(keys)))
    got = {k.decode(): e for k, e in r} if isinstance(r, list) else {}
    exp = {k: [[ids[k], flat(p)]] for k, p in want.items()}
    check("XREAD returns every byte", got == exp,
          f"{len(got)} streams; differing: {[k for k in exp if got.get(k) != exp[k]][:4]}")

    print("[2] ... after a restart")
    c.close()
    srv.stop(signal.SIGKILL)                 # the WAL is all there is
    c = srv.start()
    bad = entries_match(c, want, ids)
    check("whole after WAL replay", not bad,
          "; ".join(f"{k}: {describe(c, k)}" for k in bad[:4]))
    check("SAVE", c.cmd("SAVE") == "OK")
    c.close()
    srv.stop(signal.SIGKILL)
    c = srv.start()
    bad = entries_match(c, want, ids)
    check("whole after loading the snapshot", not bad,
          "; ".join(f"{k}: {describe(c, k)}" for k in bad[:4]))
    r = c.cmd("XADD", "{s}mixed", "*", "after", "restart")
    check("the stream keeps its last id", isinstance(r, bytes) and
          tuple(map(int, r.split(b"-"))) > tuple(map(int, ids["{s}mixed"].split(b"-"))),
          f"{r!r} after {ids['{s}mixed']!r}")
    c.close()
    srv.stop(signal.SIGTERM)


def section_replica(work, port):
    print("[3] a replica holds them whole")
    rport = port + 20
    primary = Server(work, port, "primary", ["--cluster", "--cluster-host", "127.0.0.1"])
    replica = Server(work, rport, "replica", ["--cluster", "--cluster-host", "127.0.0.1",
                                              "--cluster-replica", "--cluster-primary-host",
                                              "127.0.0.1", "--cluster-primary-port", str(port)])
    try:
        a = primary.start()
        want = expected_entries()
        ids = write_entries(a, want, ":before")       # in the replica's full resync
        b = replica.start()
        ids.update(write_entries(a, want, ":after"))  # streamed from the WAL
        deadline = time.time() + 60
        bad_before = bad_after = list(want)
        while time.time() < deadline:
            bad_before = entries_match(b, want, ids, ":before")
            bad_after = entries_match(b, want, ids, ":after")
            if not bad_before and not bad_after:
                break
            time.sleep(0.5)
        check("entries from the full resync are whole", not bad_before,
              "; ".join(f"{k}: {describe(b, k + ':before')}" for k in bad_before[:4]))
        check("entries streamed after joining are whole", not bad_after,
              "; ".join(f"{k}: {describe(b, k + ':after')}" for k in bad_after[:4]))
        a.close()
        b.close()
    finally:
        replica.stop()
        primary.stop()


def read_within(sock, seconds):
    """Bytes the server sends within `seconds` (b"" if none)."""
    sock.settimeout(seconds)
    try:
        return sock.recv(1 << 20)
    except (socket.timeout, TimeoutError):
        return b""
    finally:
        sock.settimeout(30)


def section_blocking(work, port):
    srv = Server(work, port, "blocking", ["--no-wal"])
    c = srv.start()
    try:
        print("[4] a woken XREAD BLOCK returns large entries whole")
        r = Conn(port, timeout=30)
        r.sock.sendall(encode(("XREAD", "BLOCK", "0", "STREAMS", "w", "$")))
        time.sleep(0.2)
        values = [blob(20000, 100 + i) for i in range(10)]
        c.cmd("MULTI")
        for v in values:
            c.cmd("XADD", "w", "*", "f", v)
        ids = c.cmd("EXEC")
        got = r.read()
        exp = [[b"w", [[i, [b"f", v]] for i, v in zip(ids, values)]]]
        check("ten 20 KB entries from one EXEC arrive in one reply, whole", got == exp,
              f"{len(got[0][1]) if isinstance(got, list) and got else got!r} entries")
        check("the connection is in sync afterwards", r.cmd("PING") == "PONG")

        r.sock.sendall(encode(("XREAD", "BLOCK", "0", "STREAMS", "w", "$")))
        time.sleep(0.2)
        big = blob(1 << 20, 99)
        i = c.cmd("XADD", "w", "*", "f", big)
        got = r.read()
        check("a 1 MiB entry arrives whole", got == [[b"w", [[i, [b"f", big]]]]],
              f"{len(got[0][1][0][1][1]) if isinstance(got, list) else got!r} bytes")

        print("[5] commands pipelined behind XREAD BLOCK are answered after it")
        t0 = time.monotonic()
        r.sock.sendall(encode(("XREAD", "BLOCK", "300", "STREAMS", "t", "$")) + encode(("PING",)))
        first = r.read_raw()
        waited = time.monotonic() - t0
        second = r.read_raw()
        check("timeout: a null array, after the timeout", first == b"*-1\r\n" and waited >= 0.25,
              f"{first!r} after {waited:.3f} s")
        check("then PONG", second == b"+PONG\r\n", repr(second))

        c.cmd("SET", "k", "v")
        r.sock.sendall(encode(("XREAD", "BLOCK", "0", "STREAMS", "t", "$")) + encode(("PING",))
                       + encode(("GET", "k")))
        early = read_within(r.sock, 0.3)
        check("nothing arrives while the reader is blocked", early == b"", repr(early[:60]))
        r.buf += early
        i = c.cmd("XADD", "t", "*", "f", "v")
        got = [r.read(), r.read(), r.read()]
        check("wake: the entry, then PONG, then GET's reply",
              got == [[[b"t", [[i, [b"f", b"v"]]]]], "PONG", b"v"], repr(got)[:200])

        r3 = Conn(port, timeout=30)
        r3.cmd("HELLO", "3")
        r3.sock.sendall(encode(("XREAD", "BLOCK", "200", "STREAMS", "t", "$")) + encode(("PING",)))
        replies = [r3.read_raw(), r3.read_raw()]
        check("RESP3 timeout: `_`, then PONG", replies == [b"_\r\n", b"+PONG\r\n"], repr(replies))
        r3.close()

        print("[6] a reader that disconnects while blocked is forgotten")
        gone = Conn(port, timeout=30)
        gone.sock.sendall(encode(("XREAD", "BLOCK", "0", "STREAMS", "d", "$")))
        time.sleep(0.2)
        gone.close()
        time.sleep(0.2)
        fresh = [Conn(port, timeout=30) for _ in range(4)]   # one likely reuses its fd
        check("new connections answer", all(f.cmd("PING") == "PONG" for f in fresh))
        c.cmd("XADD", "d", "*", "f", "v")
        time.sleep(0.2)
        stray = [read_within(f.sock, 0.05) for f in fresh]
        check("no connection receives the departed reader's reply", stray == [b""] * 4,
              repr([s[:40] for s in stray]))
        proc = srv.proc
        check("the server keeps serving", c.cmd("PING") == "PONG"
              and proc is not None and proc.poll() is None)
        for f in fresh:
            f.close()
        r.close()
    finally:
        c.close()
        srv.stop(signal.SIGTERM)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6441)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="stream_entries_")
    try:
        for section in (section_roundtrip_and_restart, section_replica, section_blocking):
            try:
                section(work, a.port)
            except (OSError, TimeoutError, RespProtocolError, RuntimeError) as e:
                # A reply that does not parse is a failure of this section;
                # the next one starts a fresh server.
                check(f"{section.__name__} ran to the end", False, f"{type(e).__name__}: {e}"[:300])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
