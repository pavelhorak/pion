#!/usr/bin/env python3
"""Small lists, sorted sets, geo keys and streams: what they cost, and the
ziplist's byte limit (2026-11-18).

    python3 tests/test_small_containers.py [./pion-server]

1. LSET and LINSERT on a ziplist they would push past its 8,192 bytes. Both
   used to skip the write and answer as if it had happened: LSET +OK with the
   old value still there, LINSERT the unchanged length. Checked live and after
   a restart from the WAL.
2. RSS per key for 5,000 keys of one type on a fresh server: a 3-element list
   of 100-byte values, a 2-member sorted set, a 1-member geo key, a 1-entry
   stream. Before, each cost 16-18 KB on macOS (two 8 KiB list segments a
   small list never used, a whole 16 KiB page for a sorted set's first node
   slab) and a stream ~3.7 KB (64 entry slots).
3. MEMORY USAGE against that RSS: it is what eviction steers by, and it
   counted a list's elements but not its buffers (1/67 of the RSS), a stream's
   header only (1/42).
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
PORT = 6476
N = 5000
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


class Conn:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=60)
        self.f = self.s.makefile("rb")

    def send(self, cmds):
        out = []
        for c in cmds:
            parts = [a if isinstance(a, bytes) else str(a).encode() for a in c]
            out.append(b"*%d\r\n" % len(parts))
            for p in parts:
                out.append(b"$%d\r\n%s\r\n" % (len(p), p))
        self.s.sendall(b"".join(out))

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
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(b)
            return None if n < 0 else [self.read() for _ in range(n)]
        raise RuntimeError(f"bad reply {line!r}")

    def __call__(self, *a):
        self.send([a])
        return self.read()

    def many(self, cmds):
        self.send(cmds)
        return [self.read() for _ in cmds]


class Server:
    def __init__(self, wal=False, work=None):
        self.own = work is None
        self.d = work or tempfile.mkdtemp(prefix="pion_small_")
        cmd = [BINARY, "-p", str(PORT), "--no-auto-detect", "--no-auto-embed"]
        if not wal:
            cmd.append("--no-wal")
        self.p = subprocess.Popen(cmd, cwd=self.d, stdout=open(os.path.join(self.d, "out.log"), "a"),
                                  stderr=subprocess.STDOUT)
        wait_ready_pid(PORT, self.p, 60)

    def stop(self, keep=False):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()
        wait_port_free(PORT)
        if self.own and not keep:
            shutil.rmtree(self.d, ignore_errors=True)


def rss(c):
    for line in c("INFO").decode().split("\r\n"):
        if line.startswith("used_memory:"):
            return int(line.split(":", 1)[1])
    raise RuntimeError("no used_memory in INFO")


def ziplist_limit():
    print("\n[1] LSET / LINSERT past the ziplist's 8,192 bytes")
    work = tempfile.mkdtemp(prefix="pion_small_wal_")
    try:
        s = Server(wal=True, work=work)
        c = Conn(PORT)
        # 2 + 40 bytes an entry: 194 entries are 8,148 bytes, 195 are 8,190.
        e = [b"%039d" % i + b"e" for i in range(195)]
        c.many([("RPUSH", "zl:ins", x) for x in e[:194]] + [("RPUSH", "zl:set", x) for x in e])
        big = b"B" * 64
        n = c("LINSERT", "zl:ins", "BEFORE", e[100], big)
        check("LINSERT that passes the limit inserts", n == 195, f"reply {n!r}")
        want_ins = e[:100] + [big] + e[100:194]
        got = c("LRANGE", "zl:ins", 0, -1)
        check("LINSERT: list holds the new element in place", got == want_ins)
        r = c("LSET", "zl:set", 5, big)
        got = c("LRANGE", "zl:set", 0, -1)
        want_set = e[:5] + [big] + e[6:]
        check("LSET that passes the limit writes", r == "OK" and got == want_set, f"reply {r!r}")
        # Pushes keep working on both after the switch, at both ends.
        c.many([("LPUSH", "zl:ins", "h"), ("RPUSH", "zl:ins", "t")])
        want_ins = [b"h"] + want_ins + [b"t"]
        check("pushes after the switch", c("LRANGE", "zl:ins", 0, -1) == want_ins)
        s.stop(keep=True)
        s = Server(wal=True, work=work)
        c = Conn(PORT)
        check("after a restart (WAL replay): LINSERT list", c("LRANGE", "zl:ins", 0, -1) == want_ins)
        check("after a restart (WAL replay): LSET list", c("LRANGE", "zl:set", 0, -1) == want_set)
        s.stop(keep=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def payload(k, n):
    return (b"%08d" % k) * (n // 8 + 1)


SHAPES = {
    # type: (command for key k, RSS/key ceiling in bytes)
    # Measured on macOS arm64 (release build, 5,000 keys): list 393, sorted set
    # 2,218, geo 2,199, stream 505. 1.1.4 took 16,980 / 18,163 / 17,908 / 3,703.
    # The ceilings leave room for Linux's allocator and still fail either way.
    "list (3 x 100 B)": (lambda k: ("RPUSH", f"l:{k}", "a", "b", payload(k, 100)[:100]), 2000),
    "sorted set (2 members)": (lambda k: ("ZADD", f"z:{k}", 1, f"m{k}", 2, f"n{k}"), 4000),
    "geo (1 member)": (lambda k: ("GEOADD", f"g:{k}", "13.36", "38.11", f"p{k}"), 4000),
    "stream (1 entry)": (lambda k: ("XADD", f"x:{k}", "*", "f", payload(k, 40)[:40]), 2000),
}


def per_key():
    print(f"\n[2] RSS per key, {N} keys of one type, and [3] MEMORY USAGE against it")
    for name, (cmd, ceiling) in SHAPES.items():
        s = Server()
        c = Conn(PORT)
        time.sleep(0.3)
        r0 = rss(c)
        for i in range(0, N, 500):
            c.many([cmd(j) for j in range(i, i + 500)])
        time.sleep(0.3)
        per = (rss(c) - r0) / N
        mu = [c("MEMORY", "USAGE", cmd(j)[1]) for j in range(0, N, N // 50)]
        mu = sum(mu) / len(mu)
        check(f"{name}: RSS/key under {ceiling} B", per < ceiling, f"{per:.0f} B")
        # Within 2x either way of what the key costs; it was 1/13 to 1/67.
        check(f"{name}: MEMORY USAGE within 2x of RSS/key", per / 2 <= mu <= per * 2,
              f"MEMORY USAGE {mu:.0f} B, RSS/key {per:.0f} B")
        s.stop()


def main():
    print(f"binary: {BINARY}")
    ziplist_limit()
    per_key()
    print(f"\n{'FAIL' if FAIL else 'PASS'}: {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
