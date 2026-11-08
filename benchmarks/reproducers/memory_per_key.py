#!/usr/bin/env python3
"""What a key costs in memory: Pion against Redis, same keys, same values.

    python3 benchmarks/reproducers/memory_per_key.py ./pion-server [nkeys] [value_bytes] [pion args...]

Starts Pion (`-w 1 --no-wal`), then `redis-server` from the PATH with
persistence off, and loads nkeys keys `k:<n>` with value_bytes-byte values,
1,000 per pipeline (default 200,000 x 200 B). For each it prints RSS idle and
loaded, the delta per key, INFO memory, and on macOS the `vmmap` region summary.
`NO_REDIS=1` skips Redis, `NO_PION=1` Pion. Standard RESP over a socket, no client library;
nothing is written outside a temporary directory, which is removed.

The CHANGELOG entry for the hash table's 7/8 load factor has the numbers
before and after it.
"""
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

BIN = os.path.abspath(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
VB = int(sys.argv[3]) if len(sys.argv) > 3 else 200
EXTRA = sys.argv[4:]
VALUE = b"v" * VB


class Conn:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=60)
        self.f = self.s.makefile("rb", buffering=1 << 20)

    def _read(self):
        line = self.f.readline()
        t, b = line[:1], line[1:-2]
        if t == b"+":
            return b
        if t == b"-":
            raise RuntimeError(b.decode())
        if t == b":":
            return int(b)
        if t == b"$":
            n = int(b)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if t == b"*":
            return [self._read() for _ in range(int(b))]
        raise RuntimeError(repr(line))

    def many(self, cmds):
        out = []
        for c in cmds:
            out.append(b"*%d\r\n" % len(c))
            for a in c:
                a = a if isinstance(a, bytes) else str(a).encode()
                out.append(b"$%d\r\n%s\r\n" % (len(a), a))
        self.s.sendall(b"".join(out))
        return [self._read() for _ in cmds]

    def __call__(self, *a):
        return self.many([a])[0]


def wait(port):
    for _ in range(300):
        try:
            Conn(port)("PING")
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("no server")


def rss(pid):
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip()) * 1024


def vmmap(pid):
    if sys.platform != "darwin":
        return "(vmmap is macOS only)"
    try:
        out = subprocess.check_output(["vmmap", "-summary", str(pid)], stderr=subprocess.DEVNULL, text=True)
    except Exception as e:
        return f"vmmap failed: {e}"
    keep = []
    on = False
    for l in out.split("\n"):
        if l.startswith("REGION TYPE"):
            on = True
        if on:
            keep.append(l)
        if on and l.startswith("TOTAL"):
            break
    return "\n".join(keep)


def info_mem(c):
    txt = c("INFO", "memory")
    txt = txt.decode() if isinstance(txt, bytes) else str(txt)
    d = {}
    for l in txt.split("\r\n"):
        if ":" in l:
            k, v = l.split(":", 1)
            d[k] = v
    return d


def load(c):
    B = 1000
    for i in range(0, N, B):
        c.many([("SET", b"k:%d" % j, VALUE) for j in range(i, min(N, i + B))])


def run(name, argv, port, cwd):
    p = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(port)
        c = Conn(port)
        time.sleep(1.0)
        r0 = rss(p.pid)
        m0 = info_mem(c)
        v0 = vmmap(p.pid)
        t = time.time()
        load(c)
        dt = time.time() - t
        time.sleep(1.5)
        r1 = rss(p.pid)
        m1 = info_mem(c)
        v1 = vmmap(p.pid)
        print(f"===== {name}: {N:,} keys x {VB} B, load {dt:.1f}s")
        print(f"RSS idle {r0/2**20:.1f} MB  loaded {r1/2**20:.1f} MB  delta {(r1-r0)/2**20:.1f} MB = {(r1-r0)/N:.0f} B/key")
        for k in ("used_memory", "used_memory_rss", "used_memory_dataset", "used_memory_overhead", "mem_allocator"):
            if k in m1:
                print(f"  {k}: idle {m0.get(k)} loaded {m1.get(k)}")
        print("--- vmmap idle\n" + v0)
        print("--- vmmap loaded\n" + v1)
        print(flush=True)
    finally:
        p.kill()
        p.wait()


d = tempfile.mkdtemp(prefix="memprobe_")
try:
    if not os.environ.get("NO_PION"):
        run("pion " + " ".join(EXTRA), [BIN, "-p", "6530", "-w", "1", "--no-wal", "--no-auto-detect",
                                        "--no-auto-embed", "--no-crash-log"] + EXTRA, 6530, d)
    if shutil.which("redis-server") and not os.environ.get("NO_REDIS"):
        run("redis", ["redis-server", "--port", "6540", "--save", "", "--appendonly", "no"], 6540, d)
finally:
    shutil.rmtree(d, ignore_errors=True)
