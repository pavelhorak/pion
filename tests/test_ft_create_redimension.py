#!/usr/bin/env python3
"""FT.CREATE of a new index while another is built must not corrupt the heap
or answer from the other index.

Found 2026-09-26 by running tests/test_framework_integrations.py end to end
(it is not in the gate): the full run SIGSEGV'd the server in
HNSWGraph.build_index_from_shared, deterministically. Delta-debugging the
recorded 153-command stream gave this 10-command core:

  FT.CREATE a ... DIM 4;  HSET a:7 <4-dim>;  FT.OPTIMIZE a;  FT.SEARCH a
  FT.CREATE b ... DIM 384                     # no DROPINDEX in between
  FT.SEARCH b                                 # answered with a:7 — the OTHER index
  HSET x3 <384-dim>;  FT.OPTIMIZE             # SIGSEGV (tcmalloc free list)

FT.CREATE overwrote the live graph's index name and `dim` while the graph was
still the DIM-4 build, so gh #145's name check passed and the search ran a
384-float query over 4-float slots — out of bounds, heap corrupted, and the
next allocation faulted.

Each check runs on its own fresh server. The crash check reads the server log
for SIGSEGV and PINGs afterwards; a check that "passes" on a dead server is
the failure mode this suite exists to stop.

Usage: python3 tests/test_ft_create_redimension.py [--port 6399] [--binary ./pion-server]
"""
import argparse
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_ready_pid, wait_port_free  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


class Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.f = self.s.makefile("rb")

    def cmd(self, *args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            a = a if isinstance(a, bytes) else str(a).encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.s.sendall(out)
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        t, body = line[:1], line[1:-2]
        if t == b"*":
            return [self._read() for _ in range(max(int(body), 0))]
        if t == b"$":
            n = int(body)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if t == b"-":
            return RuntimeError(body.decode(errors="replace"))
        if t == b":":
            return int(body)
        return body


def vec(dim, seed):
    rnd = random.Random(seed)
    return struct.pack(f"<{dim}f", *[rnd.gauss(0, 1) for _ in range(dim)])


def keys_of(reply):
    """Result keys of an FT.SEARCH reply, [] for an error or empty answer."""
    if not isinstance(reply, list) or not reply:
        return []
    return [k for k in reply[1::2] if isinstance(k, bytes)]


def create(c, name, dim, prefix):
    return c.cmd("FT.CREATE", name, "ON", "HASH", "PREFIX", "1", prefix, "SCHEMA", "embedding", "VECTOR",
                 "HNSW", "6", "TYPE", "FLOAT32", "DIM", dim, "DISTANCE_METRIC", "COSINE")


def knn(c, name, q, k=3):
    return c.cmd("FT.SEARCH", name, f"*=>[KNN {k} @embedding $v]", "PARAMS", "2", "v", q, "DIALECT", "2")


class Server:
    def __init__(self, binary, port):
        self.port, self.dir = port, tempfile.mkdtemp(prefix="pion_redim_")
        self.log = os.path.join(self.dir, "server.log")
        self.p = subprocess.Popen([binary, "-p", str(port), "-w", "1", "--no-auto-detect", "--no-auto-embed"],
                                  cwd=self.dir, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        try:
            wait_ready_pid(port, self.p, 15)   # this process, not a lingering listener (#27)
        except RuntimeError:
            raise SystemExit("server did not start")

    def alive(self):
        try:
            return Client(self.port).cmd("PING") == b"PONG"
        except OSError:
            return False

    def segv(self):
        """True if the server faulted — and then print why, so a failing run
        carries its own evidence instead of a bare FAIL."""
        time.sleep(0.3)
        text = open(self.log, errors="replace").read()
        died = "SIGSEGV" in text or "PION EXIT" in text or self.p.poll() is not None
        if died:
            print("    ---- server log tail ----")
            for line in text.splitlines()[-14:]:
                print("    " + line[:160])
        return died

    def stop(self):
        # Each server maps a 256 MB WAL in its own directory; leaving them
        # behind leaked 6.4 GB in one afternoon and degraded the KV gate.
        self.p.kill()
        self.p.wait()
        shutil.rmtree(self.dir, ignore_errors=True)
        wait_port_free(self.port)   # the next scenario's server reuses the port


def scenario_minimal(s):
    """The delta-debugged sequence: nothing may crash, nothing may cross indexes."""
    c = Client(s.port)
    create(c, "a", 4, "a:")
    c.cmd("HSET", "a:7", "embedding", vec(4, 7), "text", "doc a7")
    c.cmd("FT.OPTIMIZE", "a")
    check("index a answers its own document", keys_of(knn(c, "a", vec(4, 7)))[:1] == [b"a:7"])
    create(c, "b", 384, "b:")
    got = knn(c, "b", vec(384, 1))
    check("unbuilt index b never answers with index a's documents",
          not any(k.startswith(b"a:") for k in keys_of(got)), repr(got)[:120])
    try:
        for i in range(3):
            c.cmd("HSET", f"b:{i}", "embedding", vec(384, 100 + i), "text", f"doc b{i}")
        opt = c.cmd("FT.OPTIMIZE", "b")
        after = keys_of(knn(c, "b", vec(384, 101)))
    except (ConnectionError, OSError) as e:
        opt, after = e, []
    check("server survives FT.OPTIMIZE of the re-dimensioned index", not s.segv() and s.alive(), repr(opt)[:80])
    check("index b answers its own documents, nearest first", after[:1] == [b"b:1"], repr(after))


def scenario_pending_ingest(s):
    """Vectors queued for a DIM-4 index must not leak into a DIM-384 build."""
    c = Client(s.port)
    create(c, "a", 4, "a:")
    for i in range(3):
        c.cmd("HSET", f"a:{i}", "embedding", vec(4, i))
    create(c, "b", 384, "b:")
    for i in range(3):
        c.cmd("HSET", f"b:{i}", "embedding", vec(384, 100 + i))
    try:
        c.cmd("FT.OPTIMIZE", "b")
        got = keys_of(knn(c, "b", vec(384, 102), k=6))
    except (ConnectionError, OSError):
        got = []
    check("server survives a build after a dimension change with ingest pending",
          not s.segv() and s.alive())
    check("the build holds only index b's vectors", sorted(got) == [b"b:0", b"b:1", b"b:2"], repr(got))


def scenario_same_index_recreate(s):
    """Re-issuing FT.CREATE for the index being served (clients do this
    defensively) must leave it serving."""
    c = Client(s.port)
    create(c, "docs", 8, "d:")
    for i in range(5):
        c.cmd("HSET", f"d:{i}", "embedding", vec(8, i))
    c.cmd("FT.OPTIMIZE", "docs")
    create(c, "docs", 8, "d:")
    check("same-name, same-dim FT.CREATE keeps the index serving",
          keys_of(knn(c, "docs", vec(8, 3)))[:1] == [b"d:3"])
    check("server alive", not s.segv() and s.alive())


def scenario_dim_above_startup(s):
    """An index wider than the server's startup dim (default 1536) — e.g.
    text-embedding-3-large at 3072. Several per-query buffers were sized
    max(startup dim, 1536) once, at graph construction."""
    c = Client(s.port)
    r = create(c, "wide", 3072, "w:")
    if isinstance(r, RuntimeError):
        check("DIM above the startup dim is refused with an error, not accepted", True, str(r)[:100])
        return
    try:
        for i in range(4):
            c.cmd("HSET", f"w:{i}", "embedding", vec(3072, 300 + i))
        c.cmd("FT.OPTIMIZE", "wide")
        got = keys_of(knn(c, "wide", vec(3072, 302)))
    except (ConnectionError, OSError) as e:
        got = [repr(e).encode()]
    check("server survives a DIM-3072 index", not s.segv() and s.alive())
    check("DIM-3072 index answers its own document first", got[:1] == [b"w:2"], repr(got)[:100])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6399)
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    args = ap.parse_args()
    for title, fn in [("[1] minimal crash sequence", scenario_minimal),
                      ("[2] dimension change with ingest pending", scenario_pending_ingest),
                      ("[3] same-index re-create is harmless", scenario_same_index_recreate),
                      ("[4] index wider than the startup dim", scenario_dim_above_startup)]:
        print(title)
        s = Server(os.path.abspath(args.binary), args.port)
        try:
            fn(s)
        except Exception as e:  # a scenario that throws is a failure, never a skip
            check(f"{title} ran to completion", False, repr(e)[:120])
        finally:
            s.stop()
    print(f"\n{'FAIL: ' + '; '.join(FAIL) if FAIL else 'ALL PASS'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
