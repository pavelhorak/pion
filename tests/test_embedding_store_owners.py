#!/usr/bin/env python3
"""Each command that embeds text sees only its own entries (#29).

WHY
AI.SEMANTIC_CACHE, FT.ADDTEXT and AI.MEMORY all write into one per-worker
embedding store, and none of their lookups filtered by owner:
  * FT.SEARCHTEXT <index> searched every index's documents, plus cached
    responses and memories, and returned them as doc ids;
  * AI.SEMANTIC_CACHE GET could answer with a doc id or a memory;
  * AI.MEMORY RECALL <session> ignored the session;
  * FT.ADDTEXT dropped every non-ASCII byte of a doc id.

CHECKS
  1. FT.SEARCHTEXT returns only its index's ids, even for a query that is
     word for word another index's document; an index with no documents
     returns nothing.
  2. K is honoured (300 documents, K=200 -> 200).
  3. AI.SEMANTIC_CACHE GET never answers with a document or a memory.
  4. AI.MEMORY RECALL finds a session's memory and only that session's.
  5. A non-ASCII doc id comes back byte for byte.

    python3 tests/test_embedding_store_owners.py [--binary pion-server] [--port 7879]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_port_free  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


SUBJ = ("biologist engineer historian sailor gardener pilot chemist novelist architect farmer "
        "surgeon linguist astronomer drummer diplomat geologist baker weaver archer potter").split()
VERB = ("studied measured sketched repaired planted analysed described designed harvested "
        "examined translated charted rehearsed negotiated mapped baked surveyed tended").split()
OBJ = ("migration bridge manuscript harbour orchard storm reaction atrium field dialect eclipse "
       "rhythm treaty fault tapestry vessel coastline hive girder theorem").split()
PLACE = ("Lisbon Nairobi Kyoto Oslo Cairo Lima Prague Perth Quito Accra Riga Bern Hanoi Cusco "
         "Tunis Dakar Bruges Muscat Sucre Vaduz Ghent Kobe Bonn").split()


def sentence(i):
    return (f"The {SUBJ[i % len(SUBJ)]} {VERB[(i * 3 + 1) % len(VERB)]} the "
            f"{OBJ[(i * 7 + 2) % len(OBJ)]} in {PLACE[(i * 11 + 5) % len(PLACE)]}, entry {i}.")


def start(binary, port, work):
    import subprocess
    import time
    nle = sys.platform == "darwin"
    flags = ["--nle-embed"] if nle else ["--auto-embed", "--no-auto-detect"]
    # The PyTorch sidecar is found relative to the repo, so it runs there.
    cwd = work if nle else REPO
    log_path = os.path.join(work, "server.log")
    proc = subprocess.Popen([os.path.join(REPO, binary), "-p", str(port), "-w", "1",
                             "--no-crash-log", "--no-wal"] + flags,
                            cwd=cwd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited:\n" + open(log_path).read()[-2000:])
        try:
            c = Conn(port, timeout=120)
            if c.cmd("PING") == "PONG":
                return proc, c, log_path
            c.close()
        except OSError:
            pass
        time.sleep(0.3)
    raise RuntimeError("server never answered:\n" + open(log_path).read()[-2000:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", "pion-server"))
    ap.add_argument("--port", type=int, default=7879)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="emb_owners_")
    proc, c, log_path = start(a.binary, a.port, work)
    try:
        print("[1] FT.SEARCHTEXT sees only its index")
        c.cmd("FT.CREATE", "ia", "SCHEMA", "body", "TEXT", "vec", "VECTOR")
        for i in range(300):
            c.cmd("FT.ADDTEXT", "ia", f"a{i}", sentence(i))
        for i in range(300, 340):
            c.cmd("FT.ADDTEXT", "ib", f"b{i}", sentence(i))
        r = c.cmd("FT.SEARCHTEXT", "ia", sentence(310), "K", "50")
        ids = [x.decode() for x in r]
        check("50 results, all from index ia", len(ids) == 50 and all(x.startswith("a") for x in ids),
              f"{len(ids)} results, foreign: {[x for x in ids if not x.startswith('a')][:5]}")
        r = c.cmd("FT.SEARCHTEXT", "ib", sentence(310), "K", "5")
        check("index ib finds its own document first", bool(r) and r[0] == b"b310", repr(r[:3]))
        check("an index with no documents returns nothing",
              c.cmd("FT.SEARCHTEXT", "nosuch", sentence(1), "K", "5") == [])

        print("[2] K is honoured")
        r = c.cmd("FT.SEARCHTEXT", "ia", "a study of storms", "K", "200")
        check("K=200 over 300 documents returns 200", len(r) == 200, f"{len(r)}")

        print("[3] the semantic cache answers only with cached responses")
        check("SET", c.cmd("AI.SEMANTIC_CACHE", "SET", "what is the capital of France",
                           "Paris is the capital of France.") == "OK")
        got = c.cmd("AI.SEMANTIC_CACHE", "GET", sentence(5))
        check("GET on a document's exact text is not answered with the doc id",
              got not in (b"a5",) and not (isinstance(got, bytes) and got.startswith(b"a")),
              repr(got))
        got = c.cmd("AI.SEMANTIC_CACHE", "GET", "what is the capital of France")
        check("GET of the cached query still hits", got == b"Paris is the capital of France.", repr(got))

        print("[4] AI.MEMORY RECALL is per session")
        c.cmd("AI.MEMORY", "ADD", "s1", "user", "my favourite colour is ultramarine blue")
        got = c.cmd("AI.MEMORY", "RECALL", "s2", "my favourite colour is ultramarine blue")
        check("another session recalls nothing", got == [], repr(got))
        got = c.cmd("AI.MEMORY", "RECALL", "s1", "my favourite colour is ultramarine blue")
        check("the session recalls its memory", got == b"user: my favourite colour is ultramarine blue",
              repr(got))
        got = c.cmd("AI.SEMANTIC_CACHE", "GET", "my favourite colour is ultramarine blue")
        check("the semantic cache does not answer with a memory", got is None or not
              (isinstance(got, bytes) and got.startswith(b"user:")), repr(got))

        print("[5] a non-ASCII doc id survives")
        doc_id = "doc-é-π"
        c.cmd("FT.ADDTEXT", "ic", doc_id, "A lighthouse keeper logged the fog over Reykjavik.")
        r = c.cmd("FT.SEARCHTEXT", "ic", "A lighthouse keeper logged the fog over Reykjavik.", "K", "1")
        check("the id comes back byte for byte", r == [doc_id.encode()], repr(r))
    finally:
        c.close()
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            proc.kill()
        try:
            wait_port_free(a.port)
        except RuntimeError:
            pass
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
