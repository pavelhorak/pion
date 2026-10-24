#!/usr/bin/env python3
"""The vector index's slot map stays out of the keyspace.

WHY
Each document the vector index ingests gets a slot, and the worker remembers
which key the slot holds. Until 1.0.5 it remembered it in the keyspace: the
key `__hk__<slot>` held the document's key, and the set `__hk_dead__` held the
slots that died. So:

  * DBSIZE, KEYS, SCAN, RANDOMKEY and INFO's key count saw twice the keys a
    client wrote (20 documents read DBSIZE 40);
  * a client's own key named `__hk__7` was overwritten when slot 7 was
    ingested, and FT.DROPINDEX DD deleted it;
  * deleting those keys (SCAN MATCH + DEL, any cleanup script) broke the index
    at the next restart: nothing linked a hash to its slot any more, so a
    document deleted after the restart stayed in the results.

Now the map and the dead slots are tables beside the keyspace, logged as WAL
records 46-48 and written by SAVE and BGREWRITEAOF. A data directory an older
server wrote holds the old keys, and loading it reads them into the tables.
This test makes such a directory by rewriting the files this server wrote
into the older format (--legacy-binary uses a real older build instead).

    python3 tests/test_vector_slot_map.py [--binary ./pion-server] [--port 7531]
        [--legacy-binary /path/to/pion-1.0.4/pion-server]
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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 16
LONG = "doc:" + "k" * 40      # over 31 bytes: the shared 31-byte map cannot hold it
fails = []


def check(name, ok, detail: object = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  — {detail}"))
    if not ok:
        fails.append(name)
    return ok


def serve(binary, port, cwd, workers=1, extra=()):
    cmd = [binary, "-p", str(port), "-w", str(workers), "--dim", str(DIM), "--wal-size", "16",
           "--no-auto-detect", "--no-auto-embed", "--no-crash-log"] + list(extra)
    if workers > 1:
        cmd.append("--independent-workers")
    p = subprocess.Popen(cmd, cwd=cwd, stdout=open(os.path.join(cwd, "log"), "a"),
                         stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    deadline = time.time() + 120
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"server on {port} exited {p.returncode}")
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


def create(c, text=False):
    schema = ["body", "TEXT"] if text else []
    return c.cmd("FT.CREATE", "idx", "ON", "HASH", "SCHEMA", *schema,
                 "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2")


def write(c, docs):
    c.pipeline([("HSET", k, "vec", v.astype(np.float32).tobytes()) for k, v in docs.items()])


def knn(c, q, k=1):
    r = c.cmd("FT.SEARCH", "idx", f"*=>[KNN {k} @vec $q]", "PARAMS", "2", "q",
              q.astype(np.float32).tobytes(), "NOCONTENT", "DIALECT", "2")
    if not isinstance(r, list):
        return r
    return [x.decode() for x in r[1:]]


def self_misses(c, docs):
    return [k for k, v in docs.items() if knn(c, v) != [k]]


def found(c, docs, gone):
    """The keys of `gone` some query still returns."""
    hit = set()
    for k in gone:
        r = knn(c, docs[k], 5)
        if isinstance(r, list):
            hit |= set(r) & set(gone)
    return sorted(hit)


def all_keys(c):
    return sorted(k.decode() for k in c.cmd("KEYS", "*"))


def scan_keys(c):
    out, cur = set(), b"0"
    while True:
        cur, ks = c.cmd("SCAN", cur, "COUNT", "100")
        out |= {k.decode() for k in ks}
        if cur == b"0":
            return sorted(out)


def info_keys(c):
    for line in c.cmd("INFO", "keyspace").decode().split("\r\n"):
        if line.startswith("db0:keys="):
            return int(line.split("=")[1].split(",")[0])
    return 0


def internal(keys):
    return [k for k in keys if k.startswith("__hk") and k not in ("__hk__7", "__hk__9")]


def vecs(rng, n):
    return rng.standard_normal((n, DIM)).astype(np.float32)


def docs_for(rng, n, prefix="doc", longs=3):
    d = {f"{prefix}:{i}": v for i, v in enumerate(vecs(rng, n))}
    for j in range(longs):
        d[LONG + f"{prefix}{j}"] = vecs(rng, 1)[0]
    return d


def keyspace_is(c, want, label):
    """DBSIZE, KEYS, SCAN, INFO and RANDOMKEY all see exactly `want`."""
    want = sorted(want)
    check(f"{label}: DBSIZE = {len(want)}", c.cmd("DBSIZE") == len(want), c.cmd("DBSIZE"))
    ks = all_keys(c)
    check(f"{label}: KEYS * lists only the client's keys", ks == want, internal(ks) or set(ks) ^ set(want))
    sk = scan_keys(c)
    check(f"{label}: SCAN walks only the client's keys", sk == want, internal(sk) or set(sk) ^ set(want))
    check(f"{label}: INFO keyspace counts them", info_keys(c) == len(want), info_keys(c))
    rk = {c.cmd("RANDOMKEY") for _ in range(100)}
    check(f"{label}: RANDOMKEY draws only them", all(k is None or k.decode() in want for k in rk),
          [k for k in rk if k is not None and k.decode() not in want][:3])


# ── the format an older server (1.0.4 and earlier) wrote ────────────────────

def read_records(buf):
    out, off = [], 0
    while off + 13 <= len(buf):
        elen, cmd, kl = struct.unpack_from("<IBI", buf, off)
        if elen < 13 or off + elen > len(buf):
            break
        k = bytes(buf[off + 9:off + 9 + kl])
        vl = struct.unpack_from("<I", buf, off + 9 + kl)[0]
        v = bytes(buf[off + 13 + kl:off + 13 + kl + vl])
        out.append((cmd, k, v))
        off += elen
    return out


def pack_records(recs):
    return b"".join(struct.pack("<IBI", 13 + len(k) + len(v), cmd, len(k)) + k + struct.pack("<I", len(v)) + v
                    for cmd, k, v in recs)


def to_legacy(recs, snapshot=False):
    """Records 46-48 as the keys an older server wrote. Returns (records, n).
    An older snapshot held the set itself, never a DEL of it."""
    out, n = [], 0
    for cmd, k, v in recs:
        if cmd == 46:
            out.append((1, b"__hk__" + k, v) if v else (2, b"__hk__" + k, b""))
            n += 1
        elif cmd == 47:
            out.append((8, b"__hk_dead__", k))
            n += 1
        elif cmd == 48:
            if k[0] & 1 and not snapshot:
                out.append((2, b"__hk_dead__", b""))
            n += 1
        else:
            out.append((cmd, k, v))
    return out, n


def wal_header(path):
    with open(path, "rb") as f:
        return struct.unpack("<8Q", f.read(64))


def legacy_wal(path):
    """Rewrite one WAL segment as an older server wrote it. Returns how many
    records changed."""
    with open(path, "r+b") as f:
        hdr = list(struct.unpack("<8Q", f.read(64)))
        data = f.read(hdr[1])
        recs, n = to_legacy(read_records(data))
        new = pack_records(recs)
        f.seek(64)
        f.write(new + b"\0" * max(0, len(data) - len(new)))
        hdr[1], hdr[4], hdr[5] = len(new), 0, 0
        f.seek(0)
        f.write(struct.pack("<8Q", *hdr))
    return n


def legacy_snapshot(path):
    with open(path, "rb") as f:
        b = f.read()
    hdr = bytearray(b[:64])
    recs, n = to_legacy(read_records(b[64:]), snapshot=True)
    struct.pack_into("<Q", hdr, 16, len(recs))
    struct.pack_into("<I", hdr, 32, 0)
    with open(path, "wb") as f:
        f.write(bytes(hdr) + pack_records(recs))
    return n


# ── phases ───────────────────────────────────────────────────────────────────

def phase_keyspace(binary, port, d, rng):
    print("=== the keyspace holds the client's keys and nothing else ===")
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            c.cmd("SET", "__hk__7", "mine")
            c.cmd("SADD", "__hk_dead__", "a", "b")
            create(c, text=True)
            docs = docs_for(rng, 30)
            write(c, docs)
            c.cmd("HSET", LONG + "doc0", "body", "photon polarization")
            client = ["__hk__7", "__hk_dead__"]
            keyspace_is(c, list(docs) + client, "after ingest")
            check("a client key named __hk__7 keeps its value (slot 7 was ingested)",
                  c.cmd("GET", "__hk__7") == b"mine", c.cmd("GET", "__hk__7"))
            m = self_misses(c, docs)
            check(f"all {len(docs)} documents find themselves, long keys too", not m, m[:5])
            check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
            r = c.cmd("FT.SEARCH", "idx", "BM25", "photon", "K", "5")
            check("BM25 reads a long key's text through the slot map",
                  isinstance(r, list) and r[0] == 1 and r[1].decode() == LONG + "doc0", r)
            c.cmd("DEL", "doc:3", LONG + "doc1")
            del docs["doc:3"], docs[LONG + "doc1"]
            keyspace_is(c, list(docs) + client, "after a build and two DELs")
            check("the client's set __hk_dead__ is untouched by dead slots",
                  sorted(c.cmd("SMEMBERS", "__hk_dead__")) == [b"a", b"b"], c.cmd("SMEMBERS", "__hk_dead__"))
    finally:
        stop(p)


def phase_restart(binary, port, d, rng, how):
    print(f"=== a restart ({how}) keeps the map: DEL afterwards still removes a document ===")
    docs = docs_for(rng, 40)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            c.cmd("SET", "__hk__7", "mine")
            create(c)
            write(c, docs)
            check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
            c.cmd("DEL", "doc:3", LONG + "doc0")
            gone = {"doc:3": docs.pop("doc:3"), LONG + "doc0": docs.pop(LONG + "doc0")}
            if how == "SAVE":
                check("SAVE", c.cmd("SAVE") == "OK")
            elif how == "BGREWRITEAOF":
                c.cmd("BGREWRITEAOF")
            # one more after the snapshot or the rewrite: a WAL record over it
            c.cmd("DEL", "doc:4")
            gone["doc:4"] = docs.pop("doc:4")
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            keyspace_is(c, list(docs) + ["__hk__7"], "after the restart")
            check("__hk__7 is still the client's", c.cmd("GET", "__hk__7") == b"mine", c.cmd("GET", "__hk__7"))
            m = self_misses(c, docs)
            check(f"all {len(docs)} documents find themselves, long keys too", not m, m[:5])
            alld = dict(docs, **gone)
            check("documents deleted before the restart stay out", not found(c, alld, list(gone)),
                  found(c, alld, list(gone)))
            c.cmd("DEL", "doc:5", LONG + "doc1")
            after = ["doc:5", LONG + "doc1"]
            check("documents deleted after the restart leave the results", not found(c, alld, after),
                  found(c, alld, after))
    finally:
        stop(p)


def phase_auto(binary, port, d, rng):
    print("=== an automatic build's index: restart, then DEL ===")
    docs = docs_for(rng, 300)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            create(c)
            write(c, docs)
            deadline = time.time() + 60
            while time.time() < deadline and "vector_index:graph\r" not in c.cmd("INFO").decode():
                time.sleep(0.05)
            check("built by itself", "vector_index:graph\r" in c.cmd("INFO").decode())
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            keyspace_is(c, list(docs), "after the restart")
            m = self_misses(c, docs)
            check(f"all {len(docs)} documents find themselves", not m, m[:5])
            c.cmd("DEL", "doc:9", LONG + "doc2")
            check("DEL after the restart removes them", not found(c, docs, ["doc:9", LONG + "doc2"]),
                  found(c, docs, ["doc:9", LONG + "doc2"]))
    finally:
        stop(p)


def phase_dropindex(binary, port, d, rng):
    print("=== FT.DROPINDEX DD deletes the documents and nothing else ===")
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            c.cmd("SET", "__hk__7", "mine")
            create(c)
            write(c, docs_for(rng, 20))
            check("FT.DROPINDEX idx DD", c.cmd("FT.DROPINDEX", "idx", "DD") == "OK")
            keyspace_is(c, ["__hk__7"], "after DD")
            check("__hk__7 survives DD", c.cmd("GET", "__hk__7") == b"mine", c.cmd("GET", "__hk__7"))
            create(c)
            docs = docs_for(rng, 12, prefix="new", longs=2)
            write(c, docs)
            m = self_misses(c, docs)
            check("a new index's documents find themselves", not m, m[:5])
            check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            keyspace_is(c, list(docs) + ["__hk__7"], "after a restart")
            m = self_misses(c, docs)
            check("they still do after a restart", not m, m[:5])
    finally:
        stop(p)


def make_legacy(binary, legacy_binary, port, d, rng, snapshot):
    """A data directory in the older format: an index built by FT.OPTIMIZE,
    two documents deleted after it. Returns the live documents and the
    deleted ones."""
    docs = docs_for(rng, 40)
    p = serve(legacy_binary or binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            create(c)
            write(c, docs)
            check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
            c.cmd("DEL", "doc:3", LONG + "doc0")
            gone = {"doc:3": docs.pop("doc:3"), LONG + "doc0": docs.pop(LONG + "doc0")}
            if snapshot:
                check("SAVE", c.cmd("SAVE") == "OK")
            if legacy_binary:
                n = c.cmd("DBSIZE")
                check(f"the older server holds __hk__ keys (DBSIZE {n} for {len(docs)} documents)", n > len(docs))
    finally:
        stop(p, signal.SIGKILL)
    if not legacy_binary:
        if snapshot:
            n = legacy_snapshot(os.path.join(d, "pion.snapshot.0"))
        else:
            n = legacy_wal(os.path.join(d, "pion.wal.0"))
        check(f"rewritten into the older format ({n} records)", n >= len(docs) + 2, n)
    return docs, gone


def phase_upgrade(binary, legacy_binary, port, d, rng, snapshot):
    print(f"=== an older server's data directory ({'snapshot' if snapshot else 'WAL'}) loads into the map ===")
    docs, gone = make_legacy(binary, legacy_binary, port, d, rng, snapshot)
    alld = dict(docs, **gone)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            keyspace_is(c, list(docs), "after the upgrade")
            m = self_misses(c, docs)
            check(f"all {len(docs)} documents find themselves, long keys too", not m, m[:5])
            check("documents deleted before the upgrade stay out", not found(c, alld, list(gone)),
                  found(c, alld, list(gone)))
            c.cmd("DEL", "doc:5")
            check("DEL after the upgrade removes a document", not found(c, alld, ["doc:5"]))
            docs.pop("doc:5")
            if not snapshot:
                h = wal_header(os.path.join(d, "pion.wal.0"))
                check("the WAL now says where its newer records start", h[4] & 1 == 1 and h[5] > 0, h[4:6])
            # appended to the older server's log: a client key of the old name
            c.cmd("SET", "__hk__9", "mine")
            more = docs_for(rng, 5, prefix="more", longs=1)
            write(c, more)
            docs.update(more)
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            keyspace_is(c, list(docs) + ["__hk__9"], "after a second restart")
            check("__hk__9, written after the upgrade, is the client's key",
                  c.cmd("GET", "__hk__9") == b"mine", c.cmd("GET", "__hk__9"))
            m = [k for k in self_misses(c, docs) if not k.startswith("more:") and not k.startswith(LONG + "more")]
            check("the older documents still find themselves", not m, m[:5])
            c.cmd("DEL", "doc:6", LONG + "doc1")
            check("and DEL still removes them", not found(c, alld, ["doc:6", LONG + "doc1"]))
    finally:
        stop(p)


def append_wal(path, recs):
    """Append records to a WAL segment as a server that knows nothing of its
    header's newer words would (an older server)."""
    with open(path, "r+b") as f:
        hdr = list(struct.unpack("<8Q", f.read(64)))
        new = pack_records(recs)
        f.seek(64 + hdr[1])
        f.write(new)
        hdr[1] += len(new)
        f.seek(0)
        f.write(struct.pack("<8Q", *hdr))


def index_version(d):
    with open(os.path.join(d, "pion.hnsw.0"), "rb") as f:
        return struct.unpack("<2Q", f.read(16))[1]


def phase_adopt(binary, port, d, rng):
    print("=== slot-map keys written after this server's records are adopted, a client's key is not ===")
    docs = docs_for(rng, 30)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            create(c)
            write(c, docs)
            check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
    finally:
        stop(p, signal.SIGKILL)
    check("the index file is version 5 (a server before 1.0.5 refuses it)", index_version(d) == 5, index_version(d))
    # what an older server appends: its slot keys, and a client key that
    # names no hash
    keys = list(docs)
    append_wal(os.path.join(d, "pion.wal.0"),
               [(1, b"__hk__" + str(s).encode(), keys[s].encode()) for s in range(len(keys))]
               + [(1, b"__hk__99", b"not-a-hash")])
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            keyspace_is(c, keys + ["__hk__99"], "after the restart")
            check("the log says how many it moved", "33 slot-map key(s) an older server wrote" in
                  open(os.path.join(d, "log")).read())
            m = self_misses(c, docs)
            check("all documents find themselves", not m, m[:5])
            c.cmd("DEL", "doc:2", LONG + "doc0")
            check("DEL removes them", not found(c, docs, ["doc:2", LONG + "doc0"]))
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            keyspace_is(c, [k for k in keys if k not in ("doc:2", LONG + "doc0")] + ["__hk__99"],
                        "after a second restart (the move was logged)")
            check("__hk__99 is still the client's", c.cmd("GET", "__hk__99") == b"not-a-hash")
    finally:
        stop(p)


def phase_roundtrip(binary, legacy_binary, port, d, rng):
    print("=== to an older server and back: it refuses the index, indexes again, and this one reads that ===")
    docs = docs_for(rng, 30)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            create(c)
            write(c, docs)
            check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
    finally:
        stop(p)
    p = serve(legacy_binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            r = c.cmd("FT.SEARCH", "idx", "*=>[KNN 1 @vec $q]", "PARAMS", "2", "q",
                      docs["doc:1"].astype(np.float32).tobytes(), "DIALECT", "2")
            check("the older server does not serve the index", not isinstance(r, list), r)
            check("and its log says why", "unsupported (want 3 or 4)" in open(os.path.join(d, "log")).read())
            create(c)
            # another numbering: new documents first, then the old ones again
            more = docs_for(rng, 10, prefix="more", longs=1)
            write(c, more)
            for k in docs:
                c.cmd("HSET", k, "vec", docs[k].astype(np.float32).tobytes())
            docs.update(more)
            check("FT.OPTIMIZE on the older server", c.cmd("FT.OPTIMIZE", "idx") == "OK")
            c.cmd("DEL", "doc:7")
            gone = {"doc:7": docs.pop("doc:7")}
    finally:
        stop(p)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            keyspace_is(c, list(docs), "back on this server")
            m = self_misses(c, docs)
            check(f"all {len(docs)} documents find themselves", not m, m[:5])
            alld = dict(docs, **gone)
            check("the one the older server deleted stays out", not found(c, alld, ["doc:7"]))
            c.cmd("DEL", "doc:8", LONG + "doc1")
            check("DEL removes documents", not found(c, alld, ["doc:8", LONG + "doc1"]))
    finally:
        stop(p)


def phase_multi(binary, port, d, rng):
    print("=== -w 4: each worker's keyspace holds its own documents only ===")
    p = serve(binary, port, d, workers=4)
    try:
        conns = [Conn(port + 2 + w, timeout=60) for w in range(4)]
        create(conns[0])
        per = []
        for w, c in enumerate(conns):
            ds = docs_for(rng, 8, prefix=f"w{w}", longs=1)
            write(c, ds)
            per.append(ds)
        for w, c in enumerate(conns):
            keyspace_is(c, list(per[w]), f"worker {w}")
        for c in conns:
            c.close()
    finally:
        stop(p)


def phase_replica(binary, port, d, rng):
    print("=== a replica's keyspace matches the primary's ===")
    pa, pb = port, port + 20
    da, db = os.path.join(d, "a"), os.path.join(d, "b")
    os.makedirs(da)
    os.makedirs(db)
    cluster = ["--cluster", "--cluster-host", "127.0.0.1"]
    a = serve(binary, pa, da, extra=cluster)
    b = None
    try:
        with Conn(pa, timeout=60) as c:
            create(c)
            docs = docs_for(rng, 20)
            write(c, docs)
            b = serve(binary, pb, db, extra=cluster + ["--cluster-replica", "--cluster-primary-host", "127.0.0.1",
                                                      "--cluster-primary-port", str(pa)])
            more = docs_for(rng, 5, prefix="more", longs=1)
            write(c, more)
            want = sorted(list(docs) + list(more))
            with Conn(pb, timeout=60) as r:
                r.cmd("READONLY")
                deadline = time.time() + 30
                while time.time() < deadline and r.cmd("DBSIZE") != len(want):
                    time.sleep(0.1)
                check(f"replica DBSIZE = {len(want)}", r.cmd("DBSIZE") == len(want), r.cmd("DBSIZE"))
                check("replica KEYS * lists only the documents", all_keys(r) == want, internal(all_keys(r)))
    finally:
        if b is not None:
            stop(b)
        stop(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--legacy-binary", default="", help="an older build (1.0.4) to write the upgrade phases' data")
    ap.add_argument("--port", type=int, default=7531)
    a = ap.parse_args()
    binary = os.path.abspath(a.binary)
    legacy = os.path.abspath(a.legacy_binary) if a.legacy_binary else ""
    rng = np.random.default_rng(5)
    root = tempfile.mkdtemp(prefix="pion_slot_map_")
    phases = [
        lambda d: phase_keyspace(binary, a.port, d, rng),
        lambda d: phase_restart(binary, a.port, d, rng, "WAL"),
        lambda d: phase_restart(binary, a.port, d, rng, "SAVE"),
        lambda d: phase_restart(binary, a.port, d, rng, "BGREWRITEAOF"),
        lambda d: phase_auto(binary, a.port, d, rng),
        lambda d: phase_dropindex(binary, a.port, d, rng),
        lambda d: phase_upgrade(binary, legacy, a.port, d, rng, snapshot=False),
        lambda d: phase_upgrade(binary, legacy, a.port, d, rng, snapshot=True),
        lambda d: phase_adopt(binary, a.port, d, rng),
        lambda d: phase_multi(binary, a.port, d, rng),
        lambda d: phase_replica(binary, a.port, d, rng),
    ]
    if legacy:
        phases.append(lambda d: phase_roundtrip(binary, legacy, a.port, d, rng))
    try:
        for n, phase in enumerate(phases):
            d = os.path.join(root, str(n))
            os.makedirs(d)
            try:
                phase(d)
            except Exception as e:  # a phase that cannot run is a failure, not a skip
                check(f"phase {n} ran", False, repr(e))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
