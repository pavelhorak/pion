#!/usr/bin/env python3
"""gh #139 — BM25 ingest paths, ranking fidelity, and k1/b tuning.

Part A (API footgun): `FT.ADDTEXT` used to write the text into the keyspace and
the semantic-cache HNSW but never into the BM25 doc set, because `build_bm25`
walked the *main* HNSW node array and an ADDTEXT doc has no vector, hence no
node. `FT.SEARCH <idx> BM25 <q>` therefore answered `[0]` — indistinguishable
from a genuine miss. Now: ADDTEXT docs are indexed, `HSET doc:N ...` docs are
reachable through the `__hk__` reverse map (no integer-key requirement), and an
index with no inverted index at all answers with an error, not silence.

Part B (ranking): `k1` / `b` are per-query parameters, and tokenization keeps
flag-like and hyphenated terms (`--fa-window`) intact instead of shattering them
on interior punctuation. Two silent truncations that skewed df/idf are gone —
the 64-unique-terms-per-doc cap and the 32K-term vocabulary cap.

Usage:
    python3 tests/test_gh139_bm25.py [--binary ./pion-server] [--port 7871]
"""

import argparse
import os
import socket
import struct
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


# ── tiny RESP client ─────────────────────────────────────────────────────────
class Client:
    def __init__(self, port, timeout=20):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        # makefile('rb'), never manual slicing — see python_resp_array_slicing_trap
        self.f = self.sock.makefile("rb")

    def cmd(self, *args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.sock.sendall(out)
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        tag, body = line[:1], line[1:-2]
        if tag == b"*":
            n = int(body)
            return [] if n < 0 else [self._read() for _ in range(n)]
        if tag == b"$":
            n = int(body)
            if n < 0:
                return None
            return self.f.read(n + 2)[:-2]
        if tag == b"-":
            return Exception(body.decode(errors="replace"))
        if tag == b":":
            return int(body)
        return body

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def wait_ready(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = Client(port, timeout=2)
            r = c.cmd("PING")
            c.close()
            if r == b"PONG":
                return True
        except (OSError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


def bm25(client, index, query, k=10, k1=None, b=None):
    """Run FT.SEARCH … BM25 and return (ids, scores) or the Exception.

    Response shape: a flat array whose head is the hit count, followed by
    `<key>, <fields>` pairs. `fields` is `[b"id", <id>, b"score", <score>]`
    when the doc has an "id" hash field, else `[b"score", <score>]` (gh #357:
    a missing field is omitted, never substituted) — so read score by name.
    """
    args = ["FT.SEARCH", index, "BM25", query, "K", str(k)]
    if k1 is not None:
        args += ["K1", str(k1)]
    if b is not None:
        args += ["B", str(b)]
    r = client.cmd(*args)
    if isinstance(r, Exception):
        return r
    ids, scores = [], []
    for i in range(1, len(r), 2):
        # FT.ADDTEXT docs come back as their numeric id; HSET docs as their
        # real hash key (b"doc:alpha") since gh #357 gave BM25 the key map.
        ids.append(int(r[i]) if r[i].isdigit() else r[i])
        fields = r[i + 1]
        named = dict(zip(fields[0::2], fields[1::2]))
        scores.append(float(named[b"score"]) if b"score" in named else 0.0)
    return ids, scores


SCHEMA = ["SCHEMA", "body", "TEXT", "vec", "VECTOR", "HNSW", "6",
          "TYPE", "FLOAT32", "DIM", "4", "DISTANCE_METRIC", "L2"]

# doc 10 is long enough to blow past the old 64-unique-terms-per-doc cap; the
# query term sits at position 150 so a truncating indexer cannot find it.
LONG_DOC = " ".join(
    ["sesquipedalian" if i == 150 else f"w10_{i}" for i in range(200)]
)
# docs 30/31: same term, short-and-few vs long-and-many. Which one wins is
# entirely a function of `b`, so they pin down that the parameter is live.
SHORT_DOC = "alpha alpha alpha"
LONG_TF_DOC = ("alpha " * 6) + " ".join(f"f31_{i}" for i in range(60))

DOCS = {
    1:  "quantum entanglement in superconducting qubits",
    2:  "classical thermodynamics of steam engines",
    3:  "quantum error correction with surface codes",
    10: LONG_DOC,
    20: "the --fa-window flag caps the sliding attention span",
    21: "a window seat beside the flag ship",
    30: SHORT_DOC,
    31: LONG_TF_DOC,
}


# ── Test 1: FT.ADDTEXT feeds BM25; unbuilt index errors ──────────────────────
def test_addtext_feeds_bm25(port):
    print("\n[1] FT.ADDTEXT populates the BM25 index (Part A)")
    c = Client(port)
    try:
        check("FT.CREATE accepted", c.cmd("FT.CREATE", "idx", *SCHEMA) == b"OK")

        # Before any ingest+optimize: an error, not an empty array. This is the
        # whole point of the issue — a caller could not tell the two apart.
        r = bm25(c, "idx", "quantum")
        check("BM25 on an unbuilt index errors", isinstance(r, Exception),
              str(r)[:90])
        if isinstance(r, Exception):
            msg = str(r)
            check("error names FT.OPTIMIZE as the fix", "FT.OPTIMIZE" in msg)
            check("error names FT.ADDTEXT as an ingest path", "FT.ADDTEXT" in msg)

        for doc_id, text in DOCS.items():
            if c.cmd("FT.ADDTEXT", "idx", str(doc_id), text) != b"OK":
                check(f"FT.ADDTEXT {doc_id}", False)
                return
        check("FT.ADDTEXT accepted for all docs", True, f"{len(DOCS)} docs")
        check("FT.OPTIMIZE accepted", c.cmd("FT.OPTIMIZE", "idx") == b"OK")

        # The regression the issue reported: this used to be [].
        r = bm25(c, "idx", "quantum")
        ok = not isinstance(r, Exception)
        if check("BM25 over ADDTEXT-only docs returns hits", ok, str(r)[:90]) and ok:
            ids, scores = r
            check("both 'quantum' docs found, and only those",
                  sorted(ids) == [1, 3], ids)
            check("scores are positive", all(s > 0 for s in scores), scores)

        r = bm25(c, "idx", "thermodynamics")
        check("single-term query isolates its doc",
              not isinstance(r, Exception) and r[0] == [2], r)

        r = bm25(c, "idx", "quantum", k=1)
        check("K bounds the result count",
              not isinstance(r, Exception) and len(r[0]) == 1, r)

        r = bm25(c, "idx", "nonexistentterm")
        check("a genuine miss is an empty result, not an error",
              not isinstance(r, Exception) and r[0] == [], r)
    finally:
        c.close()


# ── Test 2: long documents index in full ─────────────────────────────────────
def test_long_document(port):
    print("\n[2] no per-doc term cap (was MAX_TERMS=64)")
    c = Client(port)
    try:
        r = bm25(c, "idx", "sesquipedalian")
        ok = not isinstance(r, Exception)
        check("term at position 150 of a 200-term doc is indexed",
              ok and r[0] == [10], r if not ok else r[0])
        # A term beyond the old cap must also carry sane length normalisation:
        # the long doc should score lower than a short doc for a shared term.
        r = bm25(c, "idx", "w10_199")
        check("last term of the long doc is indexed",
              not isinstance(r, Exception) and r[0] == [10], r)
    finally:
        c.close()


# ── Test 3: tokenization of flag-like / hyphenated terms ─────────────────────
def test_tokenization(port):
    print("\n[3] hyphenated and flag-like tokens (Part B)")
    c = Client(port)
    try:
        r = bm25(c, "idx", "--fa-window")
        ok = not isinstance(r, Exception)
        check("`--fa-window` matches the doc that contains it",
              ok and r[0] == [20], r if not ok else r[0])
        # Kept as one token, so the doc that merely says "window" is not a hit —
        # that is the difference from splitting on interior hyphens.
        if ok:
            check("`--fa-window` does not leak into a plain 'window' doc",
                  21 not in r[0], r[0])

        r = bm25(c, "idx", "window")
        check("plain 'window' still matches its own doc",
              not isinstance(r, Exception) and r[0] == [21], r)

        # Leading/trailing punctuation is trimmed, so these all hash alike.
        base = bm25(c, "idx", "thermodynamics")
        for variant in ("thermodynamics.", "Thermodynamics", "(thermodynamics)"):
            r = bm25(c, "idx", variant)
            check(f"edge punctuation/case ignored: {variant!r}",
                  not isinstance(r, Exception) and r[0] == base[0], r)
    finally:
        c.close()


# ── Test 4: k1 / b are tunable per query ─────────────────────────────────────
def test_k1_b_tuning(port):
    print("\n[4] k1 / b exposed per query (Part B)")
    c = Client(port)
    try:
        # doc30 = 3 tokens, tf=3. doc31 = 66 tokens, tf=6.
        # b=0 disables length normalisation → raw tf wins → doc31 first.
        # b=1 applies it in full → the short dense doc wins → doc30 first.
        r0 = bm25(c, "idx", "alpha", b=0.0)
        r1 = bm25(c, "idx", "alpha", b=1.0)
        ok = not isinstance(r0, Exception) and not isinstance(r1, Exception)
        if check("B accepted on both ends of its range", ok, f"{r0} / {r1}") and ok:
            check("b=0 ranks the high-tf long doc first", r0[0][0] == 31, r0[0])
            check("b=1 ranks the short dense doc first", r1[0][0] == 30, r1[0])
            check("b actually changed the ranking", r0[0][0] != r1[0][0])

        # k1 controls tf saturation: a large k1 widens the gap between tf=3
        # and tf=6, a near-zero k1 collapses it.
        lo = bm25(c, "idx", "alpha", k1=0.01, b=0.0)
        hi = bm25(c, "idx", "alpha", k1=10.0, b=0.0)
        ok = not isinstance(lo, Exception) and not isinstance(hi, Exception)
        if check("K1 accepted", ok, f"{lo} / {hi}") and ok:
            lo_gap = abs(lo[1][0] - lo[1][1])
            hi_gap = abs(hi[1][0] - hi[1][1])
            check("k1 changes the scores", lo[1] != hi[1], f"{lo[1]} vs {hi[1]}")
            check("larger k1 widens the tf gap", hi_gap > lo_gap,
                  f"{hi_gap:.4f} vs {lo_gap:.4f}")

        # Defaults must be unchanged when the knobs are absent.
        default = bm25(c, "idx", "alpha")
        explicit = bm25(c, "idx", "alpha", k1=1.2, b=0.75)
        check("omitting K1/B matches the documented defaults (1.2 / 0.75)",
              not isinstance(default, Exception) and default == explicit,
              f"{default} vs {explicit}")

        for bad in (("K1", "-1"), ("K1", "1000"), ("B", "2.5"), ("B", "-0.1")):
            r = c.cmd("FT.SEARCH", "idx", "BM25", "alpha", bad[0], bad[1])
            check(f"out-of-range {bad[0]}={bad[1]} is rejected",
                  isinstance(r, Exception) and "out of range" in str(r), str(r)[:70])
    finally:
        c.close()


# ── Test 5: vocabulary and forward-index growth ──────────────────────────────
def test_growth_paths(port):
    """Exercise the doubling paths: >4096 unique terms forces vocab-array
    reallocation plus a full hash-table rehash, and >65536 postings forces the
    forward index to grow. Both used to be fixed caps that silently dropped
    everything past the limit."""
    print("\n[5] vocabulary / forward-index growth (was capped at 32K terms)")
    c = Client(port, timeout=120)
    try:
        n_docs, terms_per = 400, 200
        check("FT.CREATE accepted", c.cmd("FT.CREATE", "big", *SCHEMA) == b"OK")
        for d in range(n_docs):
            text = " ".join(f"t{d}x{i}" for i in range(terms_per))
            if c.cmd("FT.ADDTEXT", "big", str(d), text) != b"OK":
                check(f"FT.ADDTEXT big doc {d}", False)
                return
        check(f"ingested {n_docs}×{terms_per} = {n_docs*terms_per} unique terms", True)
        check("FT.OPTIMIZE accepted", c.cmd("FT.OPTIMIZE", "big") == b"OK")

        # One probe per region of the vocabulary, and one per region of each
        # doc — a truncating indexer fails the later ones.
        for d, i in ((0, 0), (0, terms_per - 1), (n_docs // 2, terms_per // 2),
                     (n_docs - 1, 0), (n_docs - 1, terms_per - 1)):
            r = bm25(c, "big", f"t{d}x{i}")
            check(f"term t{d}x{i} resolves to doc {d}",
                  not isinstance(r, Exception) and r[0] == [d], r)

        # A query spanning many docs must return exactly K.
        r = bm25(c, "big", " ".join(f"t{d}x0" for d in range(20)), k=20)
        check("20-term cross-doc query returns 20 docs",
              not isinstance(r, Exception) and len(r[0]) == 20, r)
    finally:
        c.close()


# ── Test 6: FT.DROPINDEX clears the text doc set ─────────────────────────────
def test_dropindex_clears_text_docs(port):
    print("\n[6] FT.DROPINDEX drops the registered text docs")
    c = Client(port)
    try:
        # [5] replaced 'idx' with 'big' — this server holds one index. Dropping
        # 'idx' used to "work" because FT.DROPINDEX ignored its argument and
        # dropped whatever was served; since gh #403 an unknown name is refused
        # and nothing is touched, so drop the index the server actually holds.
        r = c.cmd("FT.DROPINDEX", "idx")
        check("FT.DROPINDEX of the displaced 'idx' is refused (gh #403)",
              isinstance(r, Exception) and "Unknown index name" in str(r), str(r)[:90])
        check("FT.DROPINDEX accepted", c.cmd("FT.DROPINDEX", "big") == b"OK")
        check("FT.CREATE after drop accepted", c.cmd("FT.CREATE", "idx", *SCHEMA) == b"OK")
        check("FT.OPTIMIZE on the empty index accepted",
              c.cmd("FT.OPTIMIZE", "idx") == b"OK")
        # The doc HASHes still exist in the keyspace; only the registrations went
        # away. Resurrecting them here would mean a dropped index still answers.
        r = bm25(c, "idx", "quantum")
        check("BM25 after drop reports an unbuilt index",
              isinstance(r, Exception), str(r)[:90])
    finally:
        c.close()


# ── Test 7: HSET-ingested docs reachable via the __hk__ reverse map ──────────
def test_hset_hk_indirection(port):
    print("\n[7] HSET <non-integer key> docs are BM25-indexable")
    c = Client(port)
    try:
        check("FT.CREATE accepted", c.cmd("FT.CREATE", "hk", *SCHEMA) == b"OK")
        vecs = {
            "doc:alpha": ("photon polarization interference experiments", [1.0, 0.0, 0.0, 0.0]),
            "doc:beta":  ("steam turbine maintenance schedules", [0.0, 1.0, 0.0, 0.0]),
        }
        for key, (text, vec) in vecs.items():
            blob = struct.pack("<4f", *vec)
            # Single-field HSETs: the vector field is what assigns the slot and
            # writes __hk__<slot> → <key>, which is how build_bm25 finds the text
            # for a doc whose key is not the decimal ext_id.
            c.cmd("HSET", key, "vec", blob)
            c.cmd("HSET", key, "body", text)
        check("FT.OPTIMIZE accepted", c.cmd("FT.OPTIMIZE", "hk") == b"OK")

        r = bm25(c, "hk", "photon")
        ok = not isinstance(r, Exception)
        check("BM25 finds the text behind a 'doc:alpha'-style key",
              ok and len(r[0]) == 1, r if not ok else r[0])
        r2 = bm25(c, "hk", "turbine")
        ok2 = not isinstance(r2, Exception)
        check("the other doc resolves to a different slot",
              ok and ok2 and len(r2[0]) == 1 and r2[0] != r[0],
              f"{r[0] if ok else r} vs {r2[0] if ok2 else r2}")

        r3 = bm25(c, "hk", "photon turbine")
        check("a two-term query scores both docs",
              not isinstance(r3, Exception) and len(r3[0]) == 2, r3)
    finally:
        c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7871)
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}")
        return 2

    workdir = tempfile.mkdtemp(prefix="pion-gh139-")
    print(f"gh #139 BM25 tests — binary={args.binary} port={args.port}")
    print(f"workdir={workdir}")

    def serve(port, subdir):
        d = os.path.join(workdir, subdir)
        os.makedirs(d, exist_ok=True)
        return subprocess.Popen(
            [args.binary, "--profile", "vector", "-w", "1", "-p", str(port),
             "--no-auto-detect", "--no-auto-embed"],
            cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )

    def stop(proc):
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    # A server holds one index at a time, and FT.OPTIMIZE frees the shared
    # ingest buffer — so the vector-ingest test gets its own process rather than
    # sharing state with the ADDTEXT-only run.
    proc = serve(args.port, "text")
    try:
        if not check("server becomes ready", wait_ready(args.port)):
            return 1
        test_addtext_feeds_bm25(args.port)
        test_long_document(args.port)
        test_tokenization(args.port)
        test_k1_b_tuning(args.port)
        test_growth_paths(args.port)
        test_dropindex_clears_text_docs(args.port)
    finally:
        stop(proc)

    hk_port = args.port + 10
    proc = serve(hk_port, "hset")
    try:
        if not check("second server becomes ready", wait_ready(hk_port)):
            return 1
        test_hset_hk_indirection(hk_port)
    finally:
        stop(proc)

    print(f"\n{'='*60}")
    print(f"PASS {len(PASS)}  FAIL {len(FAIL)}")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
