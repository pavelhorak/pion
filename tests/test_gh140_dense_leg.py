#!/usr/bin/env python3
"""gh #140 — FT.SEARCHTEXT result depth and FT.ADDTEXT corpus size.

Two defects sat under the "the dense leg adds no recall" report, and both are
silent: neither produced an error, so a caller measuring retrieval quality saw
only bad numbers.

Part A (ef clamp): FT.SEARCHTEXT hardcoded the HNSW beam width to ef=32 while
passing the caller's K through as the result count. Any `K > 32` therefore
returned exactly 32 hits — indistinguishable from a corpus that only had 32
matching docs. The evaluation that motivated this issue queried at K=120.

Part B (pool exhaustion): FT.ADDTEXT allocated a keyspace HASH per document via
ObjectPool.acquire(), which past its 1000-object capacity returns raw
alloc[T](1) — memory never constructed as a SlabHashMap. The 1001st distinct
document SIGSEGV'd the worker. The HSET fast path already guarded this; the
FT.ADDTEXT path did not.

Usage:
    python3 tests/test_gh140_dense_leg.py [--binary ./pion-server] [--port 7877]
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


class Client:
    def __init__(self, port, timeout=300):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.f = self.sock.makefile("rb")

    def cmd(self, *args):
        out = [b"*%d\r\n" % len(args)]
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            out.append(b"$%d\r\n%s\r\n" % (len(a), a))
        self.sock.sendall(b"".join(out))
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise IOError("connection closed mid-reply (server died?)")
        t, body = line[:1], line[1:-2]
        if t == b"+":
            return body.decode()
        if t == b"-":
            return RespError(body.decode())
        if t == b":":
            return int(body)
        if t == b"$":
            n = int(body)
            return None if n == -1 else self.f.read(n + 2)[:-2].decode("utf-8", "replace")
        if t == b"*":
            n = int(body)
            return None if n < 0 else [self._read() for _ in range(n)]
        raise IOError("unknown RESP type %r" % t)


class RespError(str):
    pass


def start(binary, port, flags, workdir):
    """Run the server in `workdir`.

    pion.wal.0 / pion.hnsw.0 are written relative to cwd, so two servers sharing
    a working directory silently corrupt each other's state. Anything but a
    private directory per server makes this test flaky when run alongside
    another Pion instance.
    """
    logpath = os.path.join(tempfile.gettempdir(), f"gh140_test_{port}.log")
    log = open(logpath, "w")
    proc = subprocess.Popen(
        [os.path.join(REPO, binary), "-p", str(port), "-w", "1"] + flags,
        cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError("server exited early; see " + logpath)
        try:
            c = Client(port)
            if c.cmd("PING") == "PONG":
                return proc, log, c, logpath
        except Exception:
            continue
    raise RuntimeError("server never answered PING; see " + logpath)


_TOPICS = [
    "asynchronous scheduling of background work",
    "memory reclamation without a garbage collector",
    "keeping tail latency predictable under load",
    "reciprocal rank fusion of two result orderings",
    "tombstone bitsets and reconnecting orphaned graph nodes",
    "quantising vectors to four bits to shrink the index",
    "a memory mapped ring buffer for the write ahead log",
    "binding the listening socket before the hash table is built",
    "neural engine acceleration for sentence embeddings",
    "sliding window attention over recent tokens only",
    "group commit that flushes once per tick",
    "swiss table probing with SIMD metadata fingerprints",
]
_MODES = ["design notes on", "a benchmark of", "the rationale behind",
          "failure modes in", "an implementation of", "tuning guidance for"]


def doc_text(i):
    """Distinct text per doc.

    Near-duplicate passages (same sentence, only an index number changing) embed
    to near-identical vectors; the HNSW beam then ties on distance and stops
    expanding after roughly one neighbour list, capping results near 2*M
    regardless of ef. That is a property of the graph on degenerate data, not of
    the K plumbing under test here, so the corpus has to be genuinely varied.
    """
    return (f"{_MODES[i % len(_MODES)].capitalize()} {_TOPICS[i % len(_TOPICS)]}, "
            f"case study {i}: section {i % 7} revisits {_TOPICS[(i * 5 + 3) % len(_TOPICS)]} "
            f"and contrasts it with {_TOPICS[(i * 11 + 7) % len(_TOPICS)]}.")


# Grammatical natural-language sentences from four category vocabularies. Apple
# NLEmbedding is trained on real English and embeds word-salad into a cramped
# region (self-retrieval then ties); natural sentences separate cleanly across
# every backend. 41*43*37*41 ≈ 2.7M combinations -> no collision in a few-k ids.
_SUBJ = ("the biologist the engineer the historian the sailor the gardener the pilot "
         "the chemist the novelist the architect the farmer the surgeon the linguist "
         "the astronomer the drummer the diplomat the geologist the baker the weaver "
         "the archer the potter the cartographer the beekeeper the welder the florist "
         "the mathematician the ranger the violinist the jeweller the shepherd the miner "
         "the cellist the falconer the vintner the cobbler the glassblower the mason "
         "the tailor the cooper the blacksmith the herbalist the navigator").split()
_VERB = ("studied measured sketched repaired planted flew analysed described designed "
         "harvested examined translated charted rehearsed negotiated mapped baked wove "
         "aimed shaped surveyed tended welded arranged proved patrolled tuned polished "
         "herded excavated bowed hunted pressed mended blew laid stitched assembled "
         "forged gathered steered").split()
_OBJ = ("the migration the bridge the manuscript the harbour the orchard the storm "
        "the reaction the plot the atrium the field the incision the dialect "
        "the eclipse the rhythm the treaty the fault the loaf the tapestry "
        "the target the vessel the coastline the hive the girder the bouquet "
        "the theorem the trail the sonata the emerald the flock the seam "
        "the concerto the falcon the vintage the boot the pane the archway").split()
_PLACE = ("in Lisbon in Nairobi in Kyoto in Oslo in Cairo in Lima in Prague in Perth "
          "in Quito in Accra in Riga in Bern in Hanoi in Cusco in Tunis in Malmo "
          "in Dakar in Bruges in Muscat in Sucre in Vaduz in Ghent in Kobe in Bonn "
          "in Tromso in Ponce in Leon in Graz in Faro in Nice in Turku in Split "
          "in Delft in Bilbao in Aarhus in Cork in Ostend in Trieste in Bergen "
          "in Nantes in Girona").split()


def unique_doc(i):
    """Deterministic, distinct, grammatical sentence per id (no RNG dependency).

    Coprime-ish strides across four category vocabularies give a unique readable
    English sentence per id, so a doc's own text embeds nearest to itself by a
    clear margin on NLEmbedding, MiniLM, or any other backend.
    """
    s = _SUBJ[(i * 1) % len(_SUBJ)].capitalize()
    v = _VERB[(i * 3 + 1) % len(_VERB)]
    o = _OBJ[(i * 7 + 2) % len(_OBJ)]
    p = _PLACE[(i * 11 + 5) % len(_PLACE)]
    return f"{s} {v} {o} {p}, in a study catalogued as entry number {i}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", "./pion-server"))
    ap.add_argument("--port", type=int, default=7877)
    a = ap.parse_args()

    # --nle-embed keeps the test hermetic on macOS: in-process embedding, no
    # sidecar download and no Ollama dependency. The PyTorch sidecar is spawned
    # via paths relative to the repo root, so that path has to run there.
    # GH140_EMBED=sidecar runs the MiniLM sidecar on macOS too, which is
    # what every Linux box runs.
    nle = sys.platform == "darwin" and os.environ.get("GH140_EMBED") != "sidecar"
    flags = ["--nle-embed"] if nle else ["--auto-embed", "--no-auto-detect"]
    tmpdir = tempfile.mkdtemp(prefix="gh140_")
    workdir = tmpdir if nle else REPO
    proc, log, c, logpath = start(a.binary, a.port, flags, workdir)
    try:
        print("Part A — FT.SEARCHTEXT honours K beyond the old ef=32 beam")
        # unique_doc, not doc_text (#27): doc_text's 72 templates differ only
        # in an index number, which MiniLM embeds to the same INT8 codes. HNSW
        # cannot keep exact duplicates connected (the neighbour heuristic keeps
        # one of each), so on Linux K=200 found 150 of the 300 docs. That is
        # the data, not the K plumbing this part checks; NLEmbedding happened
        # to separate the templates. Offset ids, so no sentence here repeats
        # one Part C stores.
        c.cmd("FT.CREATE", "gh140a", "SCHEMA", "body", "TEXT", "vec", "VECTOR")
        for i in range(300):
            c.cmd("FT.ADDTEXT", "gh140a", str(i), unique_doc(100_000 + i))
        c.cmd("FT.OPTIMIZE", "gh140a")

        q = "how is tail latency kept predictable"
        depths = {}
        for k in (5, 32, 40, 120, 200):
            r = c.cmd("FT.SEARCHTEXT", "gh140a", q, "K", str(k))
            depths[k] = len(r) if isinstance(r, list) else -1
        check("K=5 returns 5", depths[5] == 5, f"got {depths[5]}")
        check("K=32 returns 32", depths[32] == 32, f"got {depths[32]}")
        check("K=40 returns 40 (was clamped to 32)", depths[40] == 40, f"got {depths[40]}")
        check("K=120 returns 120 (was clamped to 32)", depths[120] == 120, f"got {depths[120]}")
        check("K=200 returns 200 (was clamped to 32)", depths[200] == 200, f"got {depths[200]}")

        print("\nPart B — FT.ADDTEXT survives past the 1000-object hash pool")
        c.cmd("FT.CREATE", "gh140b", "SCHEMA", "body", "TEXT", "vec", "VECTOR")
        survived = 0
        try:
            for i in range(1500):
                r = c.cmd("FT.ADDTEXT", "gh140b", str(i), doc_text(i))
                if isinstance(r, RespError):
                    break
                survived = i + 1
        except IOError as e:
            check("1500 FT.ADDTEXT docs without a crash", False,
                  f"died after {survived} docs ({e})")
        else:
            check("1500 FT.ADDTEXT docs without a crash", survived == 1500,
                  f"accepted {survived}")

        alive = False
        try:
            alive = c.cmd("PING") == "PONG"
        except Exception:
            alive = False
        check("worker still serving after >1000 docs", alive)

        if alive:
            opt = c.cmd("FT.OPTIMIZE", "gh140b")
            check("FT.OPTIMIZE succeeds on a >1000-doc corpus", opt == "OK", f"got {opt!r}")
            hits = c.cmd("FT.SEARCH", "gh140b", "BM25", "tail latency predictable", "K", "5")
            check("BM25 still answers on a >1000-doc corpus",
                  isinstance(hits, list) and len(hits) > 1, f"got {hits!r}")

        print("\nPart C — dense self-retrieval holds at scale (compact_buffer UAF)")
        # Query each of a spread of documents with its OWN exact text; a correct
        # dense index returns that document at rank 1. The compact_buffer
        # use-after-free left this at ~2/5 by 200 docs and 0/5 by 600 — the
        # embeddings were bit-perfect, the graph they were compacted into was not.
        # The corpus must be lexically DISTINCT (unique_doc, not doc_text): with
        # near-duplicate bodies two docs can quantize alike and tie out self at
        # rank 1, which would be an embedding-discrimination artefact, not the
        # graph bug under test.
        c.cmd("FT.CREATE", "gh140c", "SCHEMA", "body", "TEXT", "vec", "VECTOR")
        N = 800
        stored = [unique_doc(i) for i in range(N)]
        for i in range(N):
            c.cmd("FT.ADDTEXT", "gh140c", str(i), stored[i])
        c.cmd("FT.OPTIMIZE", "gh140c")
        probes = [0, N // 4, N // 2, (3 * N) // 4, N - 1]
        r1 = 0
        for pi in probes:
            res = c.cmd("FT.SEARCHTEXT", "gh140c", stored[pi], "K", "5")
            ids = [x for x in res if not isinstance(x, list)] if isinstance(res, list) else []
            if ids and ids[0] == str(pi):
                r1 += 1
        check(f"self-retrieval rank-1 at {N} docs", r1 == len(probes),
              f"{r1}/{len(probes)} — dense index degraded at scale")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        log.close()
        crashed = [l for l in open(logpath) if "SIGSEGV" in l or "SIGBUS" in l]
        if crashed:
            check("no fatal signal in server log", False, crashed[0].strip())
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
