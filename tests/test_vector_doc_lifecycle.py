#!/usr/bin/env python3
"""A document that leaves the keyspace leaves the search results (#46).

The vector index recorded a document's key when the vector was ingested and
never updated the record. FT.SEARCH kept answering with documents that were
deleted, expired or flushed, with a renamed document's old name, and with a
document for a vector it no longer had (HSET of a new vector, HDEL of the
field). Here each of those happens to a document of a built index, and the
document's own vector must no longer find it — while the search still
returns k live documents. Then the same holds across a restart, after SAVE,
for a slot that died before FT.OPTIMIZE, and across workers.

    python3 tests/test_vector_doc_lifecycle.py [--port 6510]
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready_pid, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.abspath(os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
DIM = 16
K = 3
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def vec(seed: int) -> bytes:
    r = random.Random(seed)
    v = [r.gauss(0, 1) for _ in range(DIM)]
    n = sum(x * x for x in v) ** 0.5
    return struct.pack(f"<{DIM}f", *[x / n for x in v])


def knn(c: Conn, q: bytes, k: int = K) -> list:
    r = c.cmd("FT.SEARCH", "idx", f"*=>[KNN {k} @vec $B]", "PARAMS", "2", "B", q, "DIALECT", "2")
    return [r[i] for i in range(1, len(r), 2)] if isinstance(r, list) else [r]


class Server:
    def __init__(self, work, port, sub, workers=1):
        self.port, self.dir, self.workers = port, os.path.join(work, sub), workers
        os.makedirs(self.dir, exist_ok=True)
        self.proc = None

    def start(self) -> Conn:
        extra = ["--independent-workers"] if self.workers > 1 else []
        self.proc = subprocess.Popen([BIN, "-p", str(self.port), "-w", str(self.workers), "--no-crash-log",
                                      "--no-auto-detect", "--no-auto-embed", "--dim", str(DIM), *extra],
                                     cwd=self.dir, stdout=open(os.path.join(self.dir, "log"), "a"),
                                     stderr=subprocess.STDOUT)
        wait_ready_pid(self.port, self.proc, 60)
        return Conn(self.port, timeout=60)

    def stop(self, sig=signal.SIGTERM):
        if self.proc:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            wait_port_free(self.port)
            for p in range(self.port + 1, self.port + 2 + self.workers + 1):
                wait_port_free(p)


def create(c: Conn, n: int, prefix: str = "doc"):
    c.cmd("FT.CREATE", "idx", "SCHEMA", "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM),
          "DISTANCE_METRIC", "L2")
    for k in range(n):
        c.cmd("HSET", f"{prefix}:{k}", "vec", vec(k), "t", "x")


def gone(c: Conn, name: str, seed: int, label: str):
    got = knn(c, vec(seed))
    check(f"{label}: its vector no longer finds {name}, and still {K} results",
          name.encode() not in got and len(got) == K, repr(got))


def run_built(work: str, port: int):
    print("[1] a built index follows the keyspace")
    s = Server(work, port, "built")
    c = s.start()
    create(c, 200)
    check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
    check("control: doc:3's vector finds doc:3 first", knn(c, vec(3))[:1] == [b"doc:3"])
    c.cmd("DEL", "doc:5")
    gone(c, "doc:5", 5, "DEL")
    c.cmd("HSET", "doc:6", "vec", vec(9006))
    gone(c, "doc:6", 6, "HSET of a new vector")
    c.cmd("RENAME", "doc:7", "doc:7b")
    got = knn(c, vec(7))
    check("RENAME: neither name answers (an HSET after FT.OPTIMIZE is not indexed)",
          b"doc:7" not in got and b"doc:7b" not in got and len(got) == K, repr(got))
    c.cmd("HDEL", "doc:8", "vec")
    gone(c, "doc:8", 8, "HDEL of the vector field")
    c.cmd("HSET", "doc:9", "vec", b"not a vector")
    gone(c, "doc:9", 9, "the vector field overwritten with text")
    c.cmd("UNLINK", "doc:12")
    gone(c, "doc:12", 12, "UNLINK")
    c.cmd("SET", "doc:13", "a string now")
    gone(c, "doc:13", 13, "SET over the hash")
    c.cmd("PEXPIRE", "doc:10", "30")
    deadline = time.time() + 30
    while time.time() < deadline and b"doc:10" in knn(c, vec(10)):
        time.sleep(0.1)
    gone(c, "doc:10", 10, "expired (removed by the sweep)")
    c.cmd("HSET", "doc:14", "t", "y")
    check("a write to another field keeps the document", knn(c, vec(14))[:1] == [b"doc:14"])
    c.cmd("FT.CREATE", "idx", "SCHEMA", "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM),
          "DISTANCE_METRIC", "L2")
    c.cmd("HSET", "late:1", "vec", vec(7001))
    check("FT.CREATE again for the served index restarts no slot numbering: doc:0 keeps its name",
          knn(c, vec(0))[:1] == [b"doc:0"], repr(knn(c, vec(0))))
    # a restart keeps every one of them gone, and relinks the live hashes
    c.close()
    s.stop(signal.SIGKILL)
    c = s.start()
    for name, seed in (("doc:5", 5), ("doc:6", 6), ("doc:8", 8), ("doc:9", 9), ("doc:10", 10),
                       ("doc:12", 12), ("doc:13", 13)):
        gone(c, name, seed, f"after a SIGKILL restart, {name}")
    c.cmd("DEL", "doc:20")
    gone(c, "doc:20", 20, "DEL after the restart (the hash was relinked to its slot)")
    c.cmd("SAVE")
    c.close()
    s.stop()
    c = s.start()
    gone(c, "doc:20", 20, "after SAVE and a restart")
    gone(c, "doc:5", 5, "after SAVE and a restart, doc:5")
    check("a live document is still found", knn(c, vec(30))[:1] == [b"doc:30"])
    c.cmd("FLUSHALL")
    got = knn(c, vec(30))
    check("FLUSHALL: nothing is found", got == [] or got == [0], repr(got))
    c.close()
    s.stop()


def run_prebuild(work: str, port: int):
    print("[2] before FT.OPTIMIZE: deleted, renamed, copied and restored documents")
    s = Server(work, port, "prebuild")
    c = s.start()
    create(c, 100)
    c.cmd("DEL", "doc:1")
    c.cmd("RENAME", "doc:2", "doc:2b")
    c.cmd("COPY", "doc:3", "doc:3c")
    payload = c.cmd("DUMP", "doc:4")
    c.cmd("RESTORE", "doc:4r", "0", payload)
    c.cmd("HSET", "doc:5", "vec", vec(5005))
    check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
    gone(c, "doc:1", 1, "deleted before the build")
    got = knn(c, vec(2))
    check("renamed before the build: found under the new name only",
          got[:1] == [b"doc:2b"] and b"doc:2" not in got, repr(got))
    got = knn(c, vec(3), 2)
    check("copied before the build: both found", sorted(got) == [b"doc:3", b"doc:3c"], repr(got))
    got = knn(c, vec(4), 2)
    check("restored before the build: both found", sorted(got) == [b"doc:4", b"doc:4r"], repr(got))
    check("a new vector before the build: the new one finds it", knn(c, vec(5005))[:1] == [b"doc:5"])
    gone(c, "doc:5", 5, "...and the old one does not")
    c.close()
    s.stop(signal.SIGKILL)
    c = s.start()
    gone(c, "doc:1", 1, "after a restart, the document deleted before the build")
    gone(c, "doc:5", 5, "after a restart, the vector replaced before the build")
    c.close()
    s.stop()


def run_workers(work: str, port: int):
    print("[3] two workers: a deletion on one hides the document on the other")
    s = Server(work, port, "workers", workers=2)
    c = s.start()
    w0, w1 = Conn(port + 2), Conn(port + 3)          # each worker's affinity port
    c.cmd("FT.CREATE", "idx", "SCHEMA", "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM),
          "DISTANCE_METRIC", "L2")
    for k in range(60):
        (w0 if k % 2 == 0 else w1).cmd("HSET", f"doc:{k}", "vec", vec(k))
    check("FT.OPTIMIZE", w0.cmd("FT.OPTIMIZE", "idx") == "OK")
    check("worker 1 finds a document worker 0 holds", knn(w1, vec(10))[:1] == [b"doc:10"])
    w0.cmd("DEL", "doc:10")
    gone(w1, "doc:10", 10, "deleted on worker 0, searched on worker 1")
    w1.cmd("DEL", "doc:11")
    gone(w0, "doc:11", 11, "deleted on worker 1, searched on worker 0")
    for x in (w0, w1, c):
        x.close()
    s.stop()


def run_generations(work: str, port: int):
    print("[4] the documents of a dropped index, deleted later, do not touch the next index")
    s = Server(work, port, "generations")
    c = s.start()
    create(c, 10, "old")
    c.cmd("FT.OPTIMIZE", "idx")
    c.cmd("FT.DROPINDEX", "idx")
    for k in range(10):
        c.cmd("DEL", f"old:{k}")          # their slots were 0..9, the next index's are too
    create(c, 10, "new")
    check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
    got = knn(c, vec(4))
    check("the new index finds its own documents", got[:1] == [b"new:4"] and len(got) == K, repr(got))
    c.close()
    s.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6510)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="pion_doclife_")
    try:
        run_built(work, a.port)
        run_prebuild(work, a.port)
        run_workers(work, a.port)
        run_generations(work, a.port)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
