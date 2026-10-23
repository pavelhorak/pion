#!/usr/bin/env python3
"""An index's schema survives a restart: its fields, their kinds and options.

WHY
The index file kept the graph, the vector field and the PREFIX list, and not
the schema's other fields. A restart therefore lost them, and with them every
check that reads them: from 1.0.4 a filter must name a schema field of the
right kind (a range on a NUMERIC field, a tag set on a TAG field), and a TAG
field splits its value at its SEPARATOR and folds case unless CASESENSITIVE.
After a restart those checks would have had no schema to read.

The schema is now saved with the index (header word 25, bit 10) and put back
by the warm load. An index file written before 1.0.4 has no schema section:
it loads as before, and its filters read any hash field, as they used to.

  1. 1.0.4 file: FT.CREATE with a NUMERIC field, a CASESENSITIVE TAG field with
     SEPARATOR ';', a SORTABLE NOINDEX field; documents; the automatic build
     writes the file; restart; every rule still holds, FT._LIST names the
     index, SORTBY reads the schema.
  2. A file without the schema section (bit 10 cleared, the section cut off,
     as a pre-1.0.4 file is): it loads, and filters work, permissively.

    python3 tests/test_ft_schema_restart.py [--binary ./pion-server] [--port 7517]
"""
import argparse
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 8
N = 60
fails = []


def check(name, ok, detail: object = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  — {detail}"))
    if not ok:
        fails.append(name)
    return ok


def serve(binary, port, cwd):
    cmd = [binary, "-p", str(port), "-w", "1", "--dim", str(DIM),
           "--no-auto-detect", "--no-auto-embed", "--no-crash-log"]
    p = subprocess.Popen(cmd, cwd=cwd, stdout=open(os.path.join(cwd, "log"), "a"),
                         stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with Conn(port, timeout=2) as c:
                if c.cmd("PING") == "PONG":
                    return p
        except OSError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"server on {port} did not start")


def stop(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass


def info(c):
    out = {}
    for line in c.cmd("INFO").decode().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k] = v.strip()
    return out


def vec(i):
    return struct.pack(f"<{DIM}f", *[((i * 7 + d * 3) % 11) / 11.0 for d in range(DIM)])


def knn(c, pre="*", extra=(), k=N):
    r = c.cmd("FT.SEARCH", "idx", f"{pre}=>[KNN {k} @vec $v]", "PARAMS", "2", "v", vec(1),
              "NOCONTENT", "LIMIT", "0", str(k), *extra, "DIALECT", "2")
    if isinstance(r, RespError):
        return r
    return sorted(x.decode() for x in r[1:]) if not extra or "SORTBY" not in extra else [x.decode() for x in r[1:]]


def build(c):
    r = c.cmd("FT.CREATE", "idx", "ON", "HASH", "PREFIX", "1", "doc:", "SCHEMA",
              "price", "NUMERIC",
              "tags", "TAG", "SEPARATOR", ";", "CASESENSITIVE",
              "rank", "NUMERIC", "SORTABLE", "NOINDEX",
              "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2")
    check("FT.CREATE", r == "OK", r)
    for i in range(N):
        tags = "Red;blue" if i % 3 == 0 else ("green" if i % 3 == 1 else "red")
        c.cmd("HSET", f"doc:{i}", "vec", vec(i), "price", str(i), "tags", tags, "rank", str(N - i))
    deadline = time.time() + 60
    while time.time() < deadline and info(c).get("vector_index") != "graph":
        time.sleep(0.05)


def rules(c, label):
    """The schema's rules, as RediSearch applies them."""
    red = sorted(f"doc:{i}" for i in range(N) if i % 3 == 2)        # exactly "red"
    big_red = sorted(f"doc:{i}" for i in range(N) if i % 3 == 0)    # "Red" in "Red;blue"
    blue = big_red
    check(f"{label}: CASESENSITIVE tag {{red}}", knn(c, "(@tags:{red})") == red, knn(c, "(@tags:{red})"))
    check(f"{label}: CASESENSITIVE tag {{Red}}", knn(c, "(@tags:{Red})") == big_red, knn(c, "(@tags:{Red})"))
    check(f"{label}: SEPARATOR ';' splits", knn(c, "(@tags:{blue})") == blue, knn(c, "(@tags:{blue})"))
    r = knn(c, "(@nosuch:{a})")
    check(f"{label}: a field not in the schema is an error", isinstance(r, RespError), r)
    r = knn(c, "(@tags:[0 1])")
    check(f"{label}: a range on a TAG field is an error", isinstance(r, RespError), r)
    r = knn(c, "(@price:{a})")
    check(f"{label}: a tag set on a NUMERIC field is an error", isinstance(r, RespError), r)
    check(f"{label}: a NOINDEX field matches nothing", knn(c, "(@rank:[0 1000])") == [], knn(c, "(@rank:[0 1000])"))
    want = sorted(f"doc:{i}" for i in range(10, 21))
    check(f"{label}: NUMERIC range", knn(c, "(@price:[10 20])") == want, knn(c, "(@price:[10 20])"))
    top = knn(c, extra=("SORTBY", "price", "DESC"), k=N)
    check(f"{label}: SORTBY a schema field", top[:3] == [f"doc:{N - 1}", f"doc:{N - 2}", f"doc:{N - 3}"], top[:3])
    lst = c.cmd("FT._LIST")
    check(f"{label}: FT._LIST names the index", lst == ["idx"], lst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7517)
    a = ap.parse_args()
    a.binary = os.path.abspath(a.binary)   # the server runs in a scratch directory
    d = tempfile.mkdtemp(prefix="ft-schema-restart-")
    try:
        print("=== 1. the schema survives a restart ===")
        p = serve(a.binary, a.port, d)
        try:
            with Conn(a.port, timeout=60) as c:
                build(c)
                rules(c, "before")
            deadline = time.time() + 30
            while time.time() < deadline and not os.path.exists(os.path.join(d, "pion.hnsw.0")):
                time.sleep(0.1)
        finally:
            stop(p)
        path = os.path.join(d, "pion.hnsw.0")
        if not check("the build wrote the index file", os.path.exists(path)):
            return 1
        with open(path, "rb") as f:
            f.seek(25 * 8)
            w25 = struct.unpack("<Q", f.read(8))[0]
        check("the file flags its schema section (word 25 bit 10)", (w25 & 1024) != 0, hex(w25))
        p = serve(a.binary, a.port, d)
        try:
            with Conn(a.port, timeout=60) as c:
                rules(c, "after a restart")
        finally:
            stop(p)

        print("=== 2. a file without the schema section (as before 1.0.4) ===")
        # bit 10 cleared and the section cut off: [count u64][count * 36 bytes]
        size = os.path.getsize(path)
        cut = 8 + 3 * 36
        with open(path, "r+b") as f:
            f.seek(size - cut)
            nsc = struct.unpack("<Q", f.read(8))[0]
            check("the section holds 3 fields", nsc == 3, nsc)
            f.seek(25 * 8)
            f.write(struct.pack("<Q", w25 & ~1024))
            f.truncate(size - cut)
        p = serve(a.binary, a.port, d)
        try:
            with Conn(a.port, timeout=60) as c:
                r = c.cmd("FT.INFO", "idx")
                check("it loads", isinstance(r, list), r)
                want = sorted(f"doc:{i}" for i in range(10, 21))
                check("a NUMERIC range still filters", knn(c, "(@price:[10 20])") == want, knn(c, "(@price:[10 20])"))
                r = knn(c, "(@nosuch:{a})")
                check("its schema is not known: a field outside it reads the hash (no hits, no error)",
                      r == [], r)
        finally:
            stop(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(fails)} failure(s)" if fails else "\nall passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
