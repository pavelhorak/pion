#!/usr/bin/env python3
"""Commands Redis 7 has that Pion lacked (#39), where a reply cannot be compared
with Redis step by step. tests/test_redis_differential.py compares the rest
(LCS, ROLE, the errors of PFDEBUG and the replication commands).

  1. MONITOR: the lines, compared with a real redis-server's when one is on
     PATH (timestamps and client addresses masked): what is shown, in which
     order (a command after its own reply; EXEC after what it ran; a script
     before what it calls), what is hidden (admin commands, unknown commands,
     wrong argument counts, commands queued by MULTI), what is redacted (AUTH,
     HELLO AUTH). A monitor may not touch the keyspace, RESET ends monitoring,
     MONITOR inside MULTI is refused at EXEC, a blocked command is shown once.
  2. LOLWUT: Pion's version by default, Schotter (VERSION 5) and the skyline
     (VERSION 6) with the shapes their arguments ask for.
  3. PFDEBUG on a HyperLogLog (Pion keeps every one dense) and PFSELFTEST.
  4. REPLCONF ACK and GETACK answer nothing; SYNC, PSYNC and REPLICAOF to a
     host are refused with an error that says what Pion does instead.

    python3 tests/test_missing_commands.py [--port 6492]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, encode, wait_ready_pid, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.abspath(os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def drain(c: Conn, quiet: float = 0.3) -> list:
    """Every reply that arrives until `quiet` seconds pass without one."""
    out = []
    while True:
        c.timeout = quiet
        try:
            out.append(c.read())
        except TimeoutError:
            c.timeout = 10.0
            return out


LINE = re.compile(r"^(\d+)\.(\d{6}) \[0 ([^\]]+)\] (.*)$")


def masked(lines: list) -> list:
    """MONITOR lines without the timestamp, and with the client address
    replaced by its role: `lua` for a script, `client` otherwise."""
    out = []
    for ln in lines:
        m = LINE.match(ln) if isinstance(ln, str) else None
        if not m:
            out.append(("?", repr(ln)))
            continue
        who = "lua" if m.group(3) == "lua" else "client"
        out.append((who, m.group(4)))
    return out


# The commands one client runs while another monitors. Each runs and is read
# back on its own, so the order of the lines is the order of the commands.
SCRIPT = [
    ("SET", "mon:k", 'hello world "q"\n\x01\x7f\\'),
    ("GET", "mon:k"),
    ("PING",),
    ("AUTH", "pw"),
    ("AUTH", "user", "pw"),
    ("CONFIG", "GET", "maxmemory"),          # admin: hidden
    ("NOSUCHCOMMAND", "x"),                   # unknown: hidden
    ("GET",),                                 # wrong argument count: hidden
    ("DEBUG", "SLEEP", "0"),                  # admin: hidden
    ("MULTI",), ("INCR", "mon:ctr"), ("SET", "mon:k2", "v"), ("EXEC",),
    ("EVAL", "redis.call('set', KEYS[1], 'ev'); return redis.call('get', KEYS[1])", "1", "mon:ek"),
    ("EVAL", "return redis.pcall('nosuch')", "0"),
    ("FUNCTION", "LOAD", "REPLACE",
     "#!lua name=monlib\nredis.register_function('monf', function(k) return redis.call('incr', k[1]) end)"),
    ("FCALL", "monf", "1", "mon:fk"),
    ("CLIENT", "SETNAME", "mon-client"),
    ("CLIENT", "KILL", "ID", "999999"),      # admin: hidden
    ("BLPOP", "mon:empty", "0.05"),           # blocks, times out: shown once
    ("LPUSH", "mon:l", "a"), ("RPOPLPUSH", "mon:l", "mon:l2"),
    ("MONITOR",),                             # admin: hidden (this client starts monitoring too)
]


def monitor_lines(port: int) -> tuple:
    """Run SCRIPT on one connection while another monitors. Returns the
    monitor's masked lines, and what the scripted client got back."""
    m = Conn(port)
    first = m.cmd("MONITOR")
    c = Conn(port)
    replies = []
    for cmd in SCRIPT:
        try:
            replies.append(c.cmd(*cmd))
        except (TimeoutError, ConnectionError) as e:
            replies.append(f"<{type(e).__name__}>")
    hello = Conn(port)
    hello.cmd("HELLO", "3", "AUTH", "default", "secret")   # WRONGPASS: still shown, redacted
    hello.close()
    lines = drain(m)
    m.close()
    c.close()
    return first, masked(lines), replies


def run_monitor(port: int, redis_port: int | None):
    print("[1] MONITOR")
    first, lines, replies = monitor_lines(port)
    check("MONITOR answers +OK", first == "OK", repr(first))
    shown = [l for _, l in lines]
    check("a line per shown command, none for the hidden ones",
          not any(x.startswith(('"CONFIG"', '"NOSUCHCOMMAND"', '"DEBUG"', '"MONITOR"')) for x in shown)
          and '"GET"' not in shown, repr(shown))
    check("arguments quoted and escaped as Redis does",
          '"SET" "mon:k" "hello world \\"q\\"\\n\\x01\\x7f\\\\"' in shown, repr(shown[:2]))
    check("AUTH's arguments redacted", '"AUTH" "(redacted)"' in shown and '"AUTH" "(redacted)" "(redacted)"' in shown)
    check("HELLO's credentials redacted", '"HELLO" "3" "AUTH" "(redacted)" "(redacted)"' in shown, repr(shown[-3:]))
    try:
        ix = [shown.index(x) for x in ('"MULTI"', '"INCR" "mon:ctr"', '"SET" "mon:k2" "v"', '"EXEC"')]
    except ValueError:
        ix = []
    check("MULTI, then the queued commands as they run, then EXEC", ix == sorted(ix) and len(ix) == 4, repr(ix))
    ev = [i for i, (who, x) in enumerate(lines) if x.startswith('"EVAL" "redis.call')]
    lua = [(who, x) for who, x in lines[ev[0] + 1:ev[0] + 3]] if ev else []
    check("a script's line, then what it calls as [0 lua]",
          lua == [("lua", '"set" "mon:ek" "ev"'), ("lua", '"get" "mon:ek"')], repr(lua))
    fc = [i for i, (who, x) in enumerate(lines) if x.startswith('"FCALL" "monf"')]
    check("FCALL, then its call", bool(fc) and lines[fc[0] + 1] == ("lua", '"incr" "mon:fk"'),
          repr(lines[fc[0]:fc[0] + 2]) if fc else "no FCALL line")
    check("a blocked command shown once", shown.count('"BLPOP" "mon:empty" "0.05"') == 1)
    if redis_port:
        _, rlines, _ = monitor_lines(redis_port)
        # Redis 8 shows its own lines for the commands it has that Pion's
        # 7.0 surface does not; there are none in SCRIPT.
        check("the same lines as redis-server", lines == rlines,
              "\n      pion:  " + "\n             ".join(repr(x) for x in lines)
              + "\n      redis: " + "\n             ".join(repr(x) for x in rlines))

    # A monitoring connection's own commands.
    m = Conn(port)
    m.cmd("MONITOR")
    m.sock.sendall(encode(("PING",)))
    got = drain(m)
    check("a monitor's own command: its reply, then its line",
          len(got) == 2 and got[0] == "PONG" and masked(got[1:]) == [("client", '"PING"')], repr(got))
    m.sock.sendall(encode(("GET", "mon:k")))
    got = drain(m)
    check("a monitor may not touch the keyspace",
          got == [RespError("ERR Replica can't interact with the keyspace")], repr(got))
    m.sock.sendall(encode(("MONITOR",)) + encode(("ECHO", "x")))
    got = drain(m)
    check("MONITOR again answers nothing", len(got) == 2 and got[0] == b"x", repr(got))
    m.sock.sendall(encode(("RESET",)))
    got = drain(m)
    check("RESET ends monitoring", got == ["RESET"], repr(got))
    c = Conn(port)
    c.cmd("SET", "mon:after", "1")
    check("...no more lines", drain(m) == [])
    check("...and the keyspace is reachable again", m.cmd("GET", "mon:after") == b"1")
    m.close()
    c.close()

    c = Conn(port)
    r = c.pipeline([("MULTI",), ("MONITOR",), ("EXEC",)])
    check("MONITOR inside MULTI is refused at EXEC",
          r[:2] == ["OK", "QUEUED"] and isinstance(r[2], list) and isinstance(r[2][0], RespError)
          and "MONITOR isn't allowed" in r[2][0], repr(r))
    c.close()

    # The fast path is off while a client monitors and back on after: the
    # commands keep answering either way.
    m = Conn(port)
    m.cmd("MONITOR")
    c = Conn(port)
    vals = [c.cmd("SET", f"mon:f{j}", str(j)) for j in range(20)]
    gets = c.pipeline([("GET", f"mon:f{j}") for j in range(20)])
    check("pipelined commands are answered while monitored", vals == ["OK"] * 20
          and gets == [str(j).encode() for j in range(20)], repr(gets[:3]))
    lines = drain(m)
    check("...and each one is shown", len(lines) == 40, str(len(lines)))
    m.close()
    time.sleep(0.2)
    check("after the monitor leaves", c.pipeline([("GET", "mon:f3"), ("PING",)]) == [b"3", "PONG"])
    c.close()


def run_lolwut(port: int):
    print("[2] LOLWUT")
    c = Conn(port)
    ver = c.cmd("INFO", "server")
    m = re.search(rb"pion_version:([^+\r\n]+)", ver)     # the version, without the build SHA
    plain = c.cmd("LOLWUT")
    check("LOLWUT prints Pion's version", plain == b"Pion ver. " + (m.group(1) if m else b"?") + b"\n", repr(plain))
    check("...whatever the arguments", c.cmd("LOLWUT", "VERSION", "7") == plain and c.cmd("LOLWUT", "1", "2") == plain)
    # the keyword folds case, as Redis's strcasecmp: only a matched VERSION
    # reads its number (the differential spells keywords upper-case)
    r = c.cmd("LOLWUT", "version", "x")
    check("VERSION matches in any case", isinstance(r, RespError) and r == "ERR value is not an integer or out of range",
          repr(r))
    s5 = c.cmd("LOLWUT", "VERSION", "5", "1", "2", "3")
    check("VERSION 5 1 2 3: one braille cell, as Redis draws it",
          s5.startswith(b"\xe2\xa0") and s5.endswith(b"\n\nGeorg Nees - schotter, plotter on paper, 1968. " + plain[:-1] + b"\n"),
          repr(s5))
    s5d = c.cmd("LOLWUT", "VERSION", "5").decode()
    rows = [r for r in s5d.split("\n") if r and "\u2800" <= r[0] <= "\u28ff"]
    check("VERSION 5: rows of 66 braille cells", len(rows) > 10 and all(len(r) == 66 for r in rows),
          str({len(r) for r in rows}))
    s6 = c.cmd("LOLWUT", "VERSION", "6", "10", "4").decode()
    art = s6.split("\nDedicated")[0].split("\n")
    check("VERSION 6 10 4: 4 rows of 10 cells", len(art) == 4 and all(r.count("\x1b[0m") == 10 for r in art),
          repr(art[:1]))
    c.cmd("HELLO", "3")
    v3 = c.raw("LOLWUT")
    check("RESP3: a verbatim string", v3.startswith(b"=") and b"\r\ntxt:Pion ver. " in v3, repr(v3))
    c.close()


def run_hll(port: int):
    print("[3] PFDEBUG, PFSELFTEST")
    c = Conn(port)
    c.cmd("DEL", "hll:a")
    c.cmd("PFADD", "hll:a", *[f"e{j}" for j in range(1000)])
    regs = c.cmd("PFDEBUG", "GETREG", "hll:a")
    check("GETREG: 16384 registers", isinstance(regs, list) and len(regs) == 16384, str(len(regs) if isinstance(regs, list) else regs))
    check("...each 0..51, some set", all(0 <= r <= 51 for r in regs) and 900 < sum(1 for r in regs if r) <= 1000)
    check("ENCODING: dense", c.cmd("PFDEBUG", "ENCODING", "hll:a") == "dense")
    check("TODENSE: already dense", c.cmd("PFDEBUG", "TODENSE", "hll:a") == 0)
    d = c.cmd("PFDEBUG", "DECODE", "hll:a")
    check("DECODE: Redis's error for a dense one", isinstance(d, RespError) and d == "ERR HLL encoding is not sparse", repr(d))
    t0 = time.time()
    st = c.cmd("PFSELFTEST")
    check("PFSELFTEST passes", st == "OK", repr(st))
    print(f"        ({time.time() - t0:.2f} s)")
    c.close()


def run_replication(port: int):
    print("[4] REPLCONF, SYNC, PSYNC, REPLICAOF, ROLE")
    c = Conn(port)
    c.sock.sendall(encode(("REPLCONF", "ack", "5")) + encode(("REPLCONF", "getack", "*")) + encode(("PING",)))
    r = [c.read_raw()] + drain(c)
    check("REPLCONF ACK and GETACK answer nothing", r == [b"+PONG\r\n"], repr(r))
    s = c.cmd("SYNC")
    check("SYNC is refused, naming Pion's replication",
          isinstance(s, RespError) and "SYNC is not supported" in s and "--cluster-replica" in s, repr(s))
    p = c.cmd("PSYNC", "?", "-1")
    check("PSYNC too (it answered +OK)", isinstance(p, RespError) and "PSYNC is not supported" in p, repr(p))
    r = c.cmd("REPLICAOF", "127.0.0.1", "6379")
    check("REPLICAOF host port is refused outside cluster mode, naming the flags",
          isinstance(r, RespError) and "--cluster-primary-host" in r, repr(r))
    check("ROLE: a standalone primary", c.cmd("ROLE") == [b"master", 0, []])
    check("still in sync", c.cmd("PING") == "PONG")
    c.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6492)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion_missing_cmds_")
    proc = subprocess.Popen([BIN, "-p", str(a.port), "-w", "1", "--no-crash-log", "--no-auto-detect",
                             "--no-auto-embed"], cwd=work, stdout=open(os.path.join(work, "log"), "a"),
                            stderr=subprocess.STDOUT)
    redis = None
    redis_port = None
    try:
        wait_ready_pid(a.port, proc, 60)
        rs = shutil.which("redis-server")
        if rs:
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            redis_port = s.getsockname()[1]
            s.close()
            redis = subprocess.Popen([rs, "--port", str(redis_port), "--save", "", "--appendonly", "no",
                                      "--dir", work], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait_ready_pid(redis_port, redis, 30)
        else:
            print("  (no redis-server on PATH: MONITOR lines are checked on their own)")
        run_monitor(a.port, redis_port)
        run_lolwut(a.port)
        run_hll(a.port)
        run_replication(a.port)
    finally:
        for p in (proc, redis):
            if p:
                p.send_signal(signal.SIGTERM)
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        wait_port_free(a.port)
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
