#!/usr/bin/env python3
"""FT.CREATE ... PREFIX: an index holds only the hashes whose key starts with
one of its prefixes, as in RediSearch.

WHY
PREFIX was accepted and never applied. An index on `PREFIX 1 doc:` took every
hash that had its vector field and DIM, whatever its key, and since 1.0.1
FT.CREATE's backfill took all of them at once. A search could answer with
documents that were never part of the index, and nothing said so.

What RediSearch does, and this checks (`--oracle PORT` runs the shared cases
against a live RediSearch to confirm the expectations are its answers):
  * the match is byte for byte and case-sensitive; a key equal to a prefix
    matches; several prefixes are ORed, and overlapping ones list a key once;
  * `PREFIX 0`, or an empty prefix, is every key;
  * a count that is not a number, or is negative, is an error;
  * a hash renamed out of the prefix leaves the index; one renamed in joins.

Pion's own parts: more than 16 prefixes, or one over 63 bytes, is refused
(never cut short) and leaves no index behind; a write outside the prefix is not
counted in vector_unindexed_writes; the list survives a restart (in the index
file); an FT.CREATE with another list replaces the index; under -w 4 every
worker applies it.

    python3 tests/test_ft_create_prefix.py [--binary ./pion-server] [--port 7501] [--oracle 6391]
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 16
LONG = "doc:" + "k" * 40          # over 31 bytes: resolved through __hk__<slot>
KEYS = ["doc:1", "Doc:2", "docs:3", "a:4", "b:5", "doc:", "x", LONG]
fails = []


def check(name, ok, detail: object = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  — {detail}"))
    if not ok:
        fails.append(name)
    return ok


def serve(binary, port, cwd, workers=1):
    cmd = [binary, "-p", str(port), "-w", str(workers), "--dim", str(DIM),
           "--no-auto-detect", "--no-auto-embed", "--no-crash-log"]
    if workers > 1:
        cmd.append("--independent-workers")
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


def stop(p, sig=signal.SIGTERM):
    try:
        os.killpg(os.getpgid(p.pid), sig)
        p.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait(timeout=10)
        except Exception:
            pass


def info(c):
    out = {}
    for line in c.cmd("INFO").decode(errors="replace").split("\r\n"):
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            out[k] = v
    return out


def unindexed(c):
    return int(info(c).get("vector_unindexed_writes", 0))


def vec(key):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return rng.standard_normal(DIM).astype(np.float32).tobytes()


def write(c, keys):
    c.pipeline([("HSET", k, "vec", vec(k)) for k in keys])


def create(c, *prefix, oracle=False):
    algo = "FLAT" if oracle else "HNSW"
    return c.cmd("FT.CREATE", "idx", "ON", "HASH", *prefix, "SCHEMA", "vec", "VECTOR", algo, "6",
                 "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2")


def members(c, oracle=False):
    """Every key the index answers with (k covers them all)."""
    if oracle:
        time.sleep(0.3)        # RediSearch indexes in the background
    r = c.cmd("FT.SEARCH", "idx", "*=>[KNN 50 @vec $q]", "PARAMS", "2", "q", vec("query"),
              "NOCONTENT", "LIMIT", "0", "50", "DIALECT", "2")
    if isinstance(r, RespError):
        return r
    keys = [k.decode() for k in r[1:]]
    return sorted(keys) if len(keys) == len(set(keys)) else ("DUPLICATES", keys)


def drop(c):
    c.cmd("FT.DROPINDEX", "idx")


# ── cases that are RediSearch's behaviour: run against Pion and the oracle ──
def shared_cases(c, oracle):
    tag = "RediSearch" if oracle else "Pion"
    c.cmd("FLUSHALL")
    write(c, KEYS)
    check(f"[{tag}] PREFIX 1 doc: takes doc:1, doc: and the long doc: key, not Doc:2 or docs:3",
          create(c, "PREFIX", "1", "doc:", oracle=oracle) == "OK"
          and members(c, oracle) == sorted(["doc:1", "doc:", LONG]), members(c, oracle))
    write(c, ["doc:9", "e:9"])
    check(f"[{tag}] an HSET afterwards: doc:9 joins, e:9 does not",
          members(c, oracle) == sorted(["doc:1", "doc:", LONG, "doc:9"]), members(c, oracle))
    c.cmd("RENAME", "doc:1", "zz:1")
    c.cmd("RENAME", "a:4", "doc:4")
    check(f"[{tag}] renamed out of the prefix leaves, renamed in joins",
          members(c, oracle) == sorted(["doc:", LONG, "doc:9", "doc:4"]), members(c, oracle))
    drop(c)
    c.cmd("FLUSHALL")
    write(c, KEYS)
    check(f"[{tag}] PREFIX 2 a: b: takes both",
          create(c, "PREFIX", "2", "a:", "b:", oracle=oracle) == "OK"
          and members(c, oracle) == ["a:4", "b:5"], members(c, oracle))
    drop(c)
    check(f"[{tag}] PREFIX 0 is every key",
          create(c, "PREFIX", "0", oracle=oracle) == "OK" and members(c, oracle) == sorted(KEYS),
          members(c, oracle))
    drop(c)
    check(f"[{tag}] an empty prefix is every key",
          create(c, "PREFIX", "1", "", oracle=oracle) == "OK" and members(c, oracle) == sorted(KEYS),
          members(c, oracle))
    drop(c)
    check(f"[{tag}] overlapping prefixes list a key once",
          create(c, "PREFIX", "2", "doc:", "doc", oracle=oracle) == "OK"
          and members(c, oracle) == sorted(["doc:1", "doc:", "docs:3", LONG]), members(c, oracle))
    drop(c)
    check(f"[{tag}] the keyword is case-insensitive",
          create(c, "prefix", "1", "a:", oracle=oracle) == "OK" and members(c, oracle) == ["a:4"],
          members(c, oracle))
    drop(c)
    for bad in (["PREFIX", "x", "a:"], ["PREFIX", "-1"]):
        r = create(c, *bad, oracle=oracle)
        check(f"[{tag}] {' '.join(bad)} is an error", isinstance(r, RespError), r)
        drop(c)


# ── Pion's own ──────────────────────────────────────────────────────────────
def phase_limits(binary, port, d):
    print("=== a refused list leaves no index behind ===")
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            write(c, KEYS)
            for name, pre in (("17 prefixes", ["PREFIX", "17"] + [f"p{i}:" for i in range(17)]),
                              ("a 64-byte prefix", ["PREFIX", "1", "p" * 64]),
                              ("a count past the arguments", ["PREFIX", "40", "a:"]),
                              ("a count that runs into SCHEMA", ["PREFIX", "3", "a:"])):
                r = create(c, *pre)
                check(f"{name} is refused", isinstance(r, RespError), r)
                check(f"  and no index exists after it",
                      info(c).get("vector_index") == "none" and isinstance(c.cmd("FT.INFO", "idx"), RespError),
                      info(c).get("vector_index"))
            check("16 prefixes of 63 bytes are taken",
                  create(c, "PREFIX", "16", *[f"{i:02d}" + "p" * 61 for i in range(15)], "doc:") == "OK"
                  and members(c) == sorted(["doc:1", "doc:", LONG]), members(c))
    finally:
        stop(p)


def phase_unindexed(binary, port, d):
    print("=== after FT.OPTIMIZE, only writes inside the prefix count as unindexed ===")
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            write(c, KEYS)
            create(c, "PREFIX", "1", "doc:")
            check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
            write(c, ["e:1", "e:2"])
            check("two writes outside the prefix: not counted", unindexed(c) == 0, unindexed(c))
            write(c, ["doc:new"])
            check("one inside: counted", unindexed(c) == 1, unindexed(c))
    finally:
        stop(p)


def phase_restart(binary, port, d, optimize):
    kind = "FT.OPTIMIZE" if optimize else "an automatic build"
    print(f"=== the list survives a restart ({kind}) ===")
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            write(c, KEYS + [f"doc:{i}" for i in range(10, 40)])
            create(c, "PREFIX", "2", "doc:", "b:")
            if optimize:
                c.cmd("FT.OPTIMIZE", "idx")
            else:
                deadline = time.time() + 30
                while info(c).get("vector_index") != "graph" and time.time() < deadline:
                    time.sleep(0.05)
            before = members(c)
            check("built", info(c).get("vector_index") == "graph" and "b:5" in before and "a:4" not in before,
                  (info(c).get("vector_index"), before))
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            check("the same members after the restart", members(c) == before, (members(c), before))
            write(c, ["e:1", "doc:new", "b:new"])
            if optimize:
                check("a write outside is not counted, two inside are", unindexed(c) == 2, unindexed(c))
            else:
                check("b:new and doc:new join, e:1 does not",
                      members(c) == sorted(before + ["doc:new", "b:new"]), members(c))
    finally:
        stop(p)


def phase_recreate(binary, port, d):
    print("=== FT.CREATE with another list replaces the index; the same list changes nothing ===")
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            write(c, KEYS)
            create(c, "PREFIX", "1", "doc:")
            deadline = time.time() + 30
            while info(c).get("vector_index") != "graph" and time.time() < deadline:
                time.sleep(0.05)
            m = members(c)
            check("the same list again: unchanged", create(c, "PREFIX", "1", "doc:") == "OK" and members(c) == m,
                  members(c))
            check("another list: the index holds what it takes",
                  create(c, "PREFIX", "1", "a:") == "OK" and members(c) == ["a:4"], members(c))
            check("no prefix: every key", create(c) == "OK" and members(c) == sorted(KEYS), members(c))
    finally:
        stop(p)


def phase_multi(binary, port, d):
    print("=== -w 4: every worker applies the list ===")
    p = serve(binary, port, d, workers=4)
    conns = []
    try:
        conns = [Conn(port + 2 + w, timeout=60) for w in range(4)]
        write(conns[1], ["doc:w1", "e:w1"])
        check("FT.CREATE on worker 0", create(conns[0], "PREFIX", "1", "doc:") == "OK")
        deadline = time.time() + 5
        while members(conns[3]) != ["doc:w1"] and time.time() < deadline:
            time.sleep(0.05)
        check("worker 1's backfill takes doc:w1, not e:w1", members(conns[3]) == ["doc:w1"], members(conns[3]))
        write(conns[2], ["doc:w2", "e:w2"])
        check("an HSET on worker 2 routes by the list", members(conns[0]) == ["doc:w1", "doc:w2"], members(conns[0]))
    finally:
        for c in conns:
            c.close()
        stop(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7501)
    ap.add_argument("--oracle", type=int, help="also run the shared cases against RediSearch on this port")
    a = ap.parse_args()
    binary = os.path.abspath(a.binary)
    if a.oracle:
        print("=== RediSearch (the oracle) ===")
        with Conn(a.oracle, timeout=30) as c:
            shared_cases(c, True)
            c.cmd("FLUSHALL")
    root = tempfile.mkdtemp(prefix="pion_prefix_")
    try:
        d = os.path.join(root, "shared")
        os.makedirs(d)
        print("=== Pion: the cases RediSearch answers ===")
        p = serve(binary, a.port, d)
        try:
            with Conn(a.port, timeout=60) as c:
                shared_cases(c, False)
        finally:
            stop(p)
        for n, phase in enumerate([phase_limits, phase_unindexed,
                                   lambda b, pt, dd: phase_restart(b, pt, dd, False),
                                   lambda b, pt, dd: phase_restart(b, pt, dd, True),
                                   phase_recreate, phase_multi]):
            d = os.path.join(root, str(n))
            os.makedirs(d)
            phase(binary, a.port, d)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
