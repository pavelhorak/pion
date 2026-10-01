#!/usr/bin/env python3
"""gh #369 — a dropped aggregate's container is freed, and COPY does not alias.

Before: nothing ever freed a list/hash/set/zset container. Keys removed by DEL,
or dropped because their last element was popped (gh #234), left the container
allocated, so a create/drain loop on ONE key leaked a container per cycle —
60K ZADD/ZPOPMIN cycles grew RSS by ~1 GB (a 16 KB node slab + a member dict
each; ~24 KB for a list). And COPY copied the 32-byte handle of an aggregate,
so `COPY a b` made both keys share one container — a write to b showed up in a.

Asserted:
  1. RSS stays bounded over create/drain cycles for every aggregate type, and
     over create/DEL cycles (drain via pop, and DEL/UNLINK of non-empty keys)
  2. COPY makes an independent container: mutating the copy leaves the source
     alone, and DEL of either leaves the other intact and readable
  3. the drained key is gone (EXISTS 0) and reusable as another type

Usage: python3 tests/test_gh369_container_free.py [./pion-server]
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6475
FAIL = []
CYCLES = 20000
BOUND_KB = 8 * 1024   # a leak of even 1 KB/cycle would be 20 MB


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def rss_kb(pid):
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip())


class Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.f = self.s.makefile("rb")

    @staticmethod
    def enc(args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            a = a if isinstance(a, bytes) else str(a).encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        return out

    def cmd(self, *args):
        self.s.sendall(self.enc(args))
        return self._read()

    def pipeline(self, cmds):
        """Send in chunks and read every reply — fast enough for 20K cycles."""
        out = []
        for k in range(0, len(cmds), 500):
            chunk = cmds[k:k + 500]
            self.s.sendall(b"".join(self.enc(c) for c in chunk))
            out += [self._read() for _ in chunk]
        return out

    def _read(self):
        line = self.f.readline()
        t, body = line[:1], line[1:-2]
        if t == b"*":
            n = int(body)
            return None if n < 0 else [self._read() for _ in range(n)]
        if t == b"$":
            n = int(body)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if t == b"-":
            return RuntimeError(body.decode(errors="replace"))
        if t == b":":
            return int(body)
        return body


def growth(proc, c, make_cycle):
    """RSS growth over CYCLES after a warm-up of the same workload."""
    c.pipeline([cmd for i in range(2000) for cmd in make_cycle(i)])
    base = rss_kb(proc.pid)
    replies = c.pipeline([cmd for i in range(CYCLES) for cmd in make_cycle(i)])
    errors = [r for r in replies if isinstance(r, RuntimeError)]
    return rss_kb(proc.pid) - base, errors


def main():
    d = tempfile.mkdtemp(prefix="pion_gh369_")
    # --no-wal: the WAL grows with every command and would swamp the RSS signal.
    proc = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-wal",
                             "--no-auto-detect", "--no-auto-embed"],
                            cwd=d, stdout=open(os.path.join(d, "log"), "w"), stderr=subprocess.STDOUT)
    try:
        c = None
        for _ in range(200):
            try:
                c = Client(PORT)
                if c.cmd("PING") == b"PONG":
                    break
            except OSError:
                time.sleep(0.1)
        if c is None:
            raise SystemExit("server did not start")
        print(f"[gh #369] {BINARY}")

        workloads = {
            "ZADD/ZPOPMIN (the issue's loop)": lambda i: [("ZADD", "z", "1", "n"), ("ZPOPMIN", "z")],
            "RPUSH/LPOP": lambda i: [("RPUSH", "l", "a"), ("LPOP", "l")],
            "HSET/HDEL": lambda i: [("HSET", "h", "f", "v"), ("HDEL", "h", "f")],
            "SADD/SPOP": lambda i: [("SADD", "s", "m"), ("SPOP", "s")],
            "ZADD/ZREM": lambda i: [("ZADD", "z2", "1", "m"), ("ZREM", "z2", "m")],
            "DEL of a 3-member zset": lambda i: [("ZADD", "zd", "1", "a", "2", "b", "3", "c"), ("DEL", "zd")],
            "DEL of a 3-element list": lambda i: [("RPUSH", "ld", "a", "b", "c"), ("DEL", "ld")],
            "UNLINK of a 3-field hash": lambda i: [("HSET", "hd", "a", "1", "b", "2", "c", "3"), ("UNLINK", "hd")],
            "DEL of a 3-member set": lambda i: [("SADD", "sd", "a", "b", "c"), ("DEL", "sd")],
        }
        for name, mk in workloads.items():
            grown, errors = growth(proc, c, mk)
            check(f"{name}: RSS bounded over {CYCLES} cycles", grown < BOUND_KB, f"grew {grown} KB")
            check(f"{name}: no errors", not errors, str(errors[:2]))

        # 3. the drained key is gone and reusable
        c.cmd("ZADD", "zz", "1", "a")
        c.cmd("ZPOPMIN", "zz")
        check("drained zset key does not exist", c.cmd("EXISTS", "zz") == 0)
        check("drained key is reusable as a string", c.cmd("SET", "zz", "x") == b"OK" and c.cmd("GET", "zz") == b"x")

        # 2. COPY independence, one per aggregate type
        c.cmd("ZADD", "cz", "1", "a", "2", "b")
        c.cmd("COPY", "cz", "cz2")
        c.cmd("ZADD", "cz2", "3", "c")
        check("COPY zset: writing the copy leaves the source alone",
              c.cmd("ZRANGE", "cz", "0", "-1") == [b"a", b"b"], repr(c.cmd("ZRANGE", "cz", "0", "-1")))
        c.cmd("DEL", "cz")
        check("COPY zset: the copy survives DEL of the source",
              c.cmd("ZRANGE", "cz2", "0", "-1") == [b"a", b"b", b"c"])
        c.cmd("RPUSH", "cl", "a", "b")
        c.cmd("COPY", "cl", "cl2")
        c.cmd("LPOP", "cl2")
        check("COPY list: popping the copy leaves the source alone", c.cmd("LRANGE", "cl", "0", "-1") == [b"a", b"b"])
        c.cmd("DEL", "cl2")
        check("COPY list: the source survives DEL of the copy", c.cmd("LRANGE", "cl", "0", "-1") == [b"a", b"b"])
        c.cmd("HSET", "ch", "f", "1")
        c.cmd("COPY", "ch", "ch2")
        c.cmd("HSET", "ch2", "g", "2")
        check("COPY hash: independent", c.cmd("HLEN", "ch") == 1 and c.cmd("HLEN", "ch2") == 2)
        c.cmd("SADD", "cs", "m")
        c.cmd("COPY", "cs", "cs2")
        c.cmd("SREM", "cs2", "m")
        check("COPY set: draining the copy leaves the source", c.cmd("SMEMBERS", "cs") == [b"m"]
              and c.cmd("EXISTS", "cs2") == 0)
        c.cmd("ZADD", "rz", "5", "old")
        c.cmd("COPY", "cz2", "rz", "REPLACE")
        check("COPY REPLACE over a zset serves the new members",
              c.cmd("ZRANGE", "rz", "0", "-1") == [b"a", b"b", b"c"])
        long_member = "m" * 40   # a heap-STRING member (SSO ends at 23 bytes)
        c.cmd("ZADD", "lz", "1", long_member)
        c.cmd("COPY", "lz", "lz2")
        c.cmd("DEL", "lz")
        check("COPY with a heap-string member survives DEL of the source",
              c.cmd("ZRANGE", "lz2", "0", "-1") == [long_member.encode()])
        check("server alive", proc.poll() is None and c.cmd("PING") == b"PONG")
    finally:
        proc.kill()
        proc.wait()
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{'FAILED' if FAIL else 'PASSED'}: {len(FAIL)} failure(s)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
