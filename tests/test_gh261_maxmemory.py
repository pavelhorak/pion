#!/usr/bin/env python3
"""gh #261 — --maxmemory refuses memory-growing writes the way Redis does.

Before this there was no limit at all: a cache that grew past RAM was SIGKILLed
by jetsam / the OOM killer, with no error a client could see. Now RSS above the
limit makes Pion answer Redis's own error to the commands Redis flags
`denyoom` (plus the substrate ingest commands), and keeps serving everything
else, so a client can still read and free memory.

Every expectation below was read off a live redis-server 8.10 first:

  1. over the limit: SET / INCR / HSET answer
     -OOM command not allowed when used memory > 'maxmemory'.
  2. still served: GET, DEL, LPOP, EXPIRE, PING
  3. pipelined: one reply per command, nothing swallowed
  4. MULTI: a denyoom command is refused at QUEUE time, EXEC -> EXECABORT
  5. crossing the limit AFTER queueing: EXEC -> "EXECABORT Transaction
     discarded because of: OOM ..." and NOTHING is applied
  6. EVAL: a read-only script runs; a redis.call('SET') inside one errors
     with the OOM text (and any redis.call error now keeps its message)
  7. INFO reports maxmemory / maxmemory_policy:noeviction
  8. CONFIG SET maxmemory 0 lifts the limit at once
  9. CLI: --maxmemory accepts Redis units and N%, refuses garbage (exit 1)
 10. the port+1 binary lane (PionPromptCache's K/V path) refuses its store
     opcodes too, and still answers PING
 11. with the WAL on (the default), the log's resident pages do not count:
     overwrites that leave the data the same size are never refused, however
     far the log grows, and new keys are still refused once the DATA outgrows
     the headroom. Before, the log's pages counted under noeviction: every
     write was refused once the log outgrew the headroom, and since refused
     writes do not grow the log, the refusal never lifted. A SAVE gives the
     log's pages back, so overwrites after it are not refused either.
 12. the same under an eviction policy: a SAVE used to leave the log's pages
     resident while their count went to 0, so the controller took them for
     data and evicted keys to pay for them: 50,000 keys fell to 33,400 and
     stayed there, every new write evicting one.

Usage: python3 tests/test_gh261_maxmemory.py [./pion-server]
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_ready_pid, wait_port_free  # noqa: E402

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6473
OOM = "OOM command not allowed when used memory > 'maxmemory'."
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


class Conn:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.f = self.s.makefile("rb")

    def send(self, *cmds):
        out = b""
        for c in cmds:
            parts = [a if isinstance(a, bytes) else str(a).encode() for a in c]
            out += b"*%d\r\n" % len(parts) + b"".join(
                b"$%d\r\n%s\r\n" % (len(p), p) for p in parts)
        self.s.sendall(out)

    def read(self):
        line = self.f.readline()
        t, b = line[:1], line[1:-2]
        if t == b"+":
            return b.decode()
        if t == b"-":
            return ("ERR", b.decode())
        if t == b":":
            return int(b)
        if t == b"$":
            n = int(b)
            return None if n < 0 else self.f.read(n + 2)[:-2].decode("latin-1")
        if t == b"*":
            n = int(b)
            return None if n < 0 else [self.read() for _ in range(n)]
        raise RuntimeError(f"bad reply {line!r}")

    def __call__(self, *a):
        self.send(a)
        return self.read()


def is_oom(r):
    return isinstance(r, tuple) and r[1] == OOM


def start(args, port=PORT, wal=False):
    d = tempfile.mkdtemp(prefix="pion_gh261_")
    log = os.path.join(d, "out.log")
    p = subprocess.Popen([BINARY, "-p", str(port), "--no-auto-detect", "--no-auto-embed",
                          *([] if wal else ["--no-wal"]), *args], cwd=d, stdout=open(log, "w"),
                         stderr=subprocess.STDOUT)
    # #27: ready means THIS process answers. A connect that succeeded used to
    # count, and right after the previous server was killed it could reach
    # that server's lingering listener (section [9] failed that way on Linux).
    try:
        wait_ready_pid(port, p, 30)
    except RuntimeError:
        pass        # a server expected to refuse to start; callers check p.poll()
    return p, d, log


def stop(p, d, port=PORT):
    if p.poll() is None:
        p.kill()
        p.wait()
    try:
        wait_port_free(port)
    except RuntimeError:
        pass
    shutil.rmtree(d, ignore_errors=True)


def used_memory(c):
    info = c("INFO")
    for line in info.split("\r\n"):
        if line.startswith("used_memory:"):
            return int(line.split(":")[1])
    return 0


def info_field(c, name):
    for line in c("INFO").split("\r\n"):
        if line.startswith(name + ":"):
            return line.split(":", 1)[1]
    return None


def fill_until_oom(c, prefix, limit=2000, size=256 * 1024):
    """SET 256 KB values until the server refuses. Returns (ok_count, first_err)."""
    val = b"x" * size
    for k in range(limit):
        r = c("SET", f"{prefix}{k}", val)
        if r != "OK":
            return k, r
    return limit, None


def main():
    if not os.path.exists(BINARY):
        print(f"no binary at {BINARY}")
        return 2

    p, d, log = start([])
    try:
        c = Conn(PORT)
        admin = Conn(PORT)
        base = used_memory(c)
        check("baseline RSS reported", base > 0, f"used_memory={base}")
        check("CONFIG GET maxmemory is 0 by default",
              c("CONFIG", "GET", "maxmemory") == ["maxmemory", "0"])
        c("SET", "keep", "v")
        c("RPUSH", "lst", "a", "b")

        # Found building [6]: every redis.call() error used to reach the client
        # as "ERR unknown Lua error" — the bridge never copied the message.
        r = c("EVAL", "return redis.call('HGET','keep','f')", "0")
        check("a redis.call() error inside a script keeps its message (WRONGTYPE)",
              isinstance(r, tuple) and r[1].startswith("WRONGTYPE"), f"got {r!r}")
        check("the connection is fine afterwards", c("PING") == "PONG")

        print("\n[1] over the limit, denyoom writes answer Redis's -OOM")
        check("CONFIG SET maxmemory <rss + 32 MB>",
              admin("CONFIG", "SET", "maxmemory", str(base + 32 * 1024 * 1024)) == "OK")
        n_ok, err = fill_until_oom(c, "big")
        check("some writes succeed below the limit", n_ok > 0, f"n_ok={n_ok}")
        check("then SET is refused with the exact Redis error", is_oom(err), f"got {err!r}")
        time.sleep(0.3)
        check("INCR refused", is_oom(c("INCR", "ctr")))
        check("HSET refused", is_oom(c("HSET", "h", "f", "v")))
        check("RPUSH refused", is_oom(c("RPUSH", "lst", "c")))
        check("the refused INCR created nothing", c("EXISTS", "ctr") == 0)

        print("\n[2] reads and memory-freeing commands still run")
        check("GET served", c("GET", "keep") == "v")
        check("PING served", c("PING") == "PONG")
        check("LPOP served", c("LPOP", "lst") == "a")
        check("EXPIRE served", c("EXPIRE", "keep", "1000") == 1)
        check("DEL served", c("DEL", "big0") == 1)

        print("\n[3] pipelining keeps one reply per command")
        c.send(("SET", "p1", "v"), ("GET", "keep"), ("INCR", "ctr"), ("PING",))
        r = [c.read() for _ in range(4)]
        check("SET/GET/INCR/PING -> OOM, value, OOM, PONG",
              is_oom(r[0]) and r[1] == "v" and is_oom(r[2]) and r[3] == "PONG", f"got {r!r}")

        print("\n[4] MULTI: refused at queue time, EXEC aborts")
        check("MULTI", c("MULTI") == "OK")
        check("queued SET -> OOM", is_oom(c("SET", "tx", "1")))
        check("queued GET -> QUEUED", c("GET", "keep") == "QUEUED")
        r = c("EXEC")
        check("EXEC -> EXECABORT", isinstance(r, tuple) and r[1].startswith("EXECABORT"), f"got {r!r}")
        check("nothing from it applied", c("EXISTS", "tx") == 0)

        print("\n[6] EVAL: read-only scripts run, writes inside are refused")
        check("EVAL 'return 1' runs", c("EVAL", "return 1", "0") == 1)
        r = c("EVAL", "return redis.call('SET','ev','1')", "0")
        check("redis.call('SET') inside a script errors with OOM",
              isinstance(r, tuple) and "OOM" in r[1], f"got {r!r}")
        check("...and wrote nothing", c("EXISTS", "ev") == 0)
        r = c("EVAL", "return redis.call('GET','keep')", "0")
        check("redis.call('GET') inside a script works", r == "v", f"got {r!r}")

        print("\n[7] INFO")
        check("maxmemory in INFO", info_field(c, "maxmemory") == str(base + 32 * 1024 * 1024))
        check("maxmemory_policy:noeviction", info_field(c, "maxmemory_policy") == "noeviction")
        check("maxmemory_refusing_writes:1", info_field(c, "maxmemory_refusing_writes") == "1")

        print("\n[8] CONFIG SET maxmemory 0 lifts the limit immediately")
        check("CONFIG SET maxmemory 0", admin("CONFIG", "SET", "maxmemory", "0") == "OK")
        check("SET works on the very next command", c("SET", "after", "1") == "OK")
        check("INCR works", c("INCR", "ctr") == 1)
        check("refusing_writes back to 0", info_field(c, "maxmemory_refusing_writes") == "0")

        print("\n[5] limit crossed AFTER queueing: EXEC aborts, applies nothing")
        check("MULTI", c("MULTI") == "OK")
        check("SET queued", c("SET", "late", "1") == "QUEUED")
        check("INCR queued", c("INCR", "ctr") == "QUEUED")
        check("another client sets a limit below current RSS",
              admin("CONFIG", "SET", "maxmemory", "1mb") == "OK")
        r = c("EXEC")
        check("EXEC -> EXECABORT ... because of: OOM",
              isinstance(r, tuple) and r[1] == "EXECABORT Transaction discarded because of: " + OOM,
              f"got {r!r}")
        admin("CONFIG", "SET", "maxmemory", "0")
        check("the queued SET was not applied", c("EXISTS", "late") == 0)
        check("the queued INCR was not applied", c("GET", "ctr") == "1")
        check("a read-only transaction under the limit still runs",
              c("MULTI") == "OK" and c("GET", "keep") == "QUEUED" and c("EXEC") == ["v"])
    finally:
        stop(p, d)

    print("\n[9] command line")
    p, d, log = start(["--maxmemory", "1mb"])
    try:
        out = open(log).read()
        check("--maxmemory 1mb starts", p.poll() is None, out[-300:])
        check("startup names the limit", "Maxmemory: 1 MB" in out, out[-500:])
        c = Conn(PORT)
        time.sleep(0.3)
        r = None
        for _ in range(50):            # the first housekeeping ticks flip the flag
            r = c("SET", "a", "b")
            if r != "OK":
                break
            time.sleep(0.02)
        check("a server started over its limit refuses writes", is_oom(r), f"got {r!r}")
        check("...and serves reads", c("PING") == "PONG")
        check("CONFIG GET reports the flag's bytes",
              c("CONFIG", "GET", "maxmemory") == ["maxmemory", str(1024 * 1024)])
        time.sleep(0.2)
        check("the crossing was logged with the measured RSS",
              "refusing memory-growing writes" in open(log).read())
    finally:
        stop(p, d)
    p, d, log = start(["--maxmemory", "50%"])
    try:
        c = Conn(PORT)
        mm = int(c("CONFIG", "GET", "maxmemory")[1])
        check("--maxmemory 50% resolves against physical RAM", mm > 1024 ** 3, f"got {mm}")
        check("...and writes are accepted", c("SET", "a", "b") == "OK")
    finally:
        stop(p, d)
    for bad in ("lots", "10xb", "0%", "101%", "-5"):
        p, d, log = start(["--maxmemory", bad])
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        out = open(log).read()
        check(f"--maxmemory {bad} is refused at startup (exit 1)",
              p.poll() == 1 and "FATAL" in out, f"rc={p.poll()} {out[-200:]!r}")
        stop(p, d)

    print("\n[10] binary lane (port+1)")
    p, d, log = start(["--kvcache", "--maxmemory", "1mb"])
    try:
        import struct

        def frame(cmd, body):
            return struct.pack("<HBI", 0xCA5E, cmd, len(body)) + body

        def reply(sock):
            hdr = b""
            while len(hdr) < 7:
                hdr += sock.recv(7 - len(hdr))
            _, status, n = struct.unpack("<HBI", hdr)
            body = b""
            while len(body) < n:
                body += sock.recv(n - len(body))
            return status, body

        c = Conn(PORT)
        time.sleep(0.3)
        for _ in range(50):
            if is_oom(c("SET", "warm", "1")):
                break
            time.sleep(0.02)
        b = socket.create_connection(("127.0.0.1", PORT + 1), timeout=10)
        create = frame(0x20, struct.pack("<H", 3) + b"s01" + struct.pack("<HH", 64, 64))
        b.sendall(create)
        st, body = reply(b)
        check("ATTEND_CREATE over the limit -> STATUS_ERROR with the OOM text",
              st == 2 and OOM.encode() in body, f"status={st} body={body!r}")
        b.sendall(frame(0xFF, b""))
        st, _ = reply(b)
        check("binary PING still OK", st == 0, f"status={st}")
        c("CONFIG", "SET", "maxmemory", "0")
        b.sendall(create)
        st, body = reply(b)
        check("ATTEND_CREATE accepted once the limit is lifted", st == 0, f"status={st} body={body!r}")
        b.close()
    finally:
        stop(p, d)


    print("\n[11] WAL on: the log's pages do not count against the limit")
    p, d, log = start([], wal=True)
    try:
        c = Conn(PORT)
        val = "v" * 200
        n, batch = 50_000, 1000

        def write_all(prefix, start_at=0):
            """SET n keys, pipelined; returns how many were refused."""
            refused = 0
            for i in range(0, n, batch):
                c.send(*[("SET", f"{prefix}{start_at + j}", val) for j in range(i, i + batch)])
                for _ in range(batch):
                    r = c.read()
                    if is_oom(r):
                        refused += 1
                    elif r != "OK":
                        raise RuntimeError(f"SET -> {r!r}")
            return refused

        write_all("w")
        time.sleep(0.3)
        head = 8 * 1024 * 1024
        rss0 = used_memory(c)
        tail0 = int(info_field(c, "wal_tail_offset") or 0)
        check("the log already holds more than the headroom", tail0 > head, f"wal_tail_offset={tail0}")
        check("CONFIG SET maxmemory <rss + 8 MB>",
              c("CONFIG", "SET", "maxmemory", str(rss0 + head)) == "OK")
        refused = sum(write_all("w") for _ in range(4))
        tail1 = int(info_field(c, "wal_tail_offset") or 0)
        sealed = int(info_field(c, "wal_segments_sealed") or 0)
        check("the overwrites grew the log well past the headroom",
              tail1 - tail0 > 4 * head or sealed > 0, f"tail {tail0} -> {tail1}, sealed {sealed}")
        check("no overwrite was refused (the data did not grow)", refused == 0,
              f"{refused:,} of {4 * n:,} refused; rss {rss0 >> 20} -> {used_memory(c) >> 20} MB")
        # A SAVE empties the log: its pages must leave RSS with it. Before, the
        # offset went to 0 while the pages stayed mapped, so the limit counted
        # them as data from then on.
        check("SAVE", c("SAVE") == "OK")
        time.sleep(0.3)
        rss_saved = used_memory(c)
        check("SAVE gives the log's pages back", rss_saved < rss0,
              f"rss {rss0 >> 20} MB before the overwrites, {rss_saved >> 20} MB after SAVE")
        refused = write_all("w")
        check("overwrites after SAVE are not refused", refused == 0, f"{refused:,} of {n:,} refused")
        # The limit is sampled every 100 ms, so one fast burst can overshoot it
        # by megabytes before it is seen: write new keys in rounds until one is
        # refused, up to 8 x 50K x 200 B, far past the 8 MB of headroom.
        refused = rounds = 0
        while refused == 0 and rounds < 8:
            refused = write_all("new", rounds * n)
            rounds += 1
            time.sleep(0.2)
        check("new keys are still refused once the data outgrows the headroom",
              refused > 0, f"{rounds * n:,} new 200 B keys; rss {used_memory(c) >> 20} MB, wal_tail "
              f"{int(info_field(c, 'wal_tail_offset') or 0) >> 20} MB, limit {(rss0 + head) >> 20} MB")
        out = open(log).read()
        check("the crossing names the log's share",
              "wal_mb=" in out and "refusing memory-growing writes" in out, out[-400:])
    finally:
        stop(p, d)

    print("\n[12] WAL on, eviction policy: a SAVE does not shrink the keyspace")
    p, d, log = start([], wal=True)
    try:
        c = Conn(PORT)
        val = "v" * 200
        n, batch = 50_000, 1000

        def write_all(prefix):
            """SET n keys, pipelined; returns how many were refused."""
            refused = 0
            for i in range(0, n, batch):
                c.send(*[("SET", f"{prefix}{j}", val) for j in range(i, i + batch)])
                for _ in range(batch):
                    r = c.read()
                    if is_oom(r):
                        refused += 1
                    elif r != "OK":
                        raise RuntimeError(f"SET -> {r!r}")
            return refused

        write_all("w")
        time.sleep(0.3)
        c("CONFIG", "SET", "maxmemory-policy", "allkeys-lru")
        c("CONFIG", "SET", "maxmemory", str(used_memory(c) + 8 * 1024 * 1024))
        for _ in range(4):
            write_all("w")
        time.sleep(0.3)
        before = c("DBSIZE")
        check("the overwrites evicted nothing", before == n, f"DBSIZE {before}")
        check("SAVE", c("SAVE") == "OK")
        time.sleep(0.5)
        refused = write_all("w")
        time.sleep(0.3)
        after = c("DBSIZE")
        check("no write refused after SAVE", refused == 0, f"{refused:,} of {n:,} refused")
        check("the keyspace survives SAVE", after >= n * 9 // 10,
              f"DBSIZE {before} -> {after}, evicted_keys {info_field(c, 'evicted_keys')}")
    finally:
        stop(p, d)

    print(f"\n{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILED'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
