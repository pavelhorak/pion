#!/usr/bin/env python3
"""The kv profile has no vector search, and says so.

`--profile kv` sizes the HNSW graph for ONE vector (config.apply_profile).
FT.CREATE used to succeed anyway: ingest kept one document and dropped the
rest without a word, FT.OPTIMIZE built it, and FT.SEARCH answered from it
(recall@10 0.002 on tests/test_vector_recall_smoke.py), a plausible reply over
missing data. Now FT.CREATE refuses under the kv profile, so there is never an
index to answer from, and hashes are stored as plain hashes.

Checks both directions: the kv profile refuses, the default profile still
creates and finds. Starts its own servers ($PION_BIN, default ./pion-server).

    python3 tests/test_kv_profile_ft.py
"""
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

BIN = os.environ.get("PION_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pion-server"))
PORT = 16781
fails = []


def check(name, ok, detail: object = ""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("" if ok else f"   [{detail}]"))
    if not ok:
        fails.append(name)


class Resp:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.buf = b""

    def cmd(self, *args):
        out = [b"*%d\r\n" % len(args)]
        for a in args:
            a = a if isinstance(a, bytes) else str(a).encode()
            out.append(b"$%d\r\n%s\r\n" % (len(a), a))
        self.s.sendall(b"".join(out))
        return self._read()

    def _line(self):
        while b"\r\n" not in self.buf:
            d = self.s.recv(65536)
            if not d:
                raise ConnectionError("closed")
            self.buf += d
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def _read(self):
        line = self._line()
        t, rest = line[:1], line[1:]
        if t == b"+":
            return rest.decode()
        if t == b"-":
            return RuntimeError(rest.decode())
        if t == b":":
            return int(rest)
        if t == b"$":
            n = int(rest)
            if n < 0:
                return None
            while len(self.buf) < n + 2:
                self.buf += self.s.recv(65536)
            v, self.buf = self.buf[:n], self.buf[n + 2:]
            return v
        if t == b"*":
            n = int(rest)
            return None if n < 0 else [self._read() for _ in range(n)]
        raise ValueError(line)


def start(extra):
    d = tempfile.mkdtemp(prefix="kvprof_")
    p = subprocess.Popen([BIN, "-p", str(PORT), "-w", "1", "--no-auto-detect", "--no-auto-embed",
                          "--no-crash-log", "--dim", "4"] + extra, cwd=d,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try:
            r = Resp(PORT)
            if r.cmd("PING") == "PONG":
                return p, r
        except OSError:
            time.sleep(0.1)
    p.kill()
    raise SystemExit("server did not start")


def stop(p, r):
    try:
        r.cmd("SHUTDOWN", "NOSAVE")
    except (OSError, ConnectionError):
        pass
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def vec(i):
    return struct.pack("<4f", float(i), 1.0, 0.0, 0.0)


def scenario(r):
    """Create, ingest 20 docs, build, search for doc 7's own vector."""
    created = r.cmd("FT.CREATE", "idx", "ON", "HASH", "PREFIX", "1", "d:", "SCHEMA",
                    "v", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", "4", "DISTANCE_METRIC", "L2")
    for i in range(20):
        r.cmd("HSET", f"d:{i}", "v", vec(i), "n", str(i))
    built = r.cmd("FT.OPTIMIZE", "idx")
    found = r.cmd("FT.SEARCH", "idx", "*=>[KNN 3 @v $q]", "PARAMS", "2", "q", vec(7), "DIALECT", "2")
    return created, built, found


p, r = start(["--profile", "kv"])
print("[1] --profile kv")
created, built, found = scenario(r)
check("FT.CREATE is refused", isinstance(created, RuntimeError) and "kv profile" in str(created), created)
check("FT.SEARCH has no index to answer from", isinstance(found, RuntimeError), found)
check("the hashes are stored as hashes", r.cmd("HGET", "d:7", "n") == b"7" and r.cmd("DBSIZE") == 20,
      (r.cmd("HGET", "d:7", "n"), r.cmd("DBSIZE")))
stop(p, r)

p, r = start([])
print("[2] default profile")
created, built, found = scenario(r)
check("FT.CREATE succeeds", created == "OK", created)
check("FT.SEARCH finds doc 7 first",
      isinstance(found, list) and len(found) > 1 and isinstance(found[0], int) and found[0] >= 1 and found[1] == b"d:7", found)
stop(p, r)

print("ALL PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
