#!/usr/bin/env python3
"""FT.SEARCH pipelined dispatch — reply pairing and batch survival.

gh #156/#162 token-skip family, FT.* edition. `handle_ft_search`'s optional-arg
scan (FILTER/PARAMS/LIMIT detection) advanced over every unrecognized token
bounded by `num_tokens` — the token count of the WHOLE recv buffer — instead of
`cmd_end_tok`. Consequences, both reproduced against 0.866 before the fix:

  1. Every command pipelined behind an FT.SEARCH KNN query was silently
     swallowed: the scan consumed its tokens, `ci` landed at num_tokens - 1,
     and the dispatch loop exited. 20/20 raw-socket FT+PING trials lost the
     PONG; `pipelined_bench.py` (Gate 4b) deadlocked on all connections.
  2. Two pipelined FT.SEARCH frames in one buffer: the first query's PARAMS
     scan walked into the second command and overwrote `blob_ptr` with the
     SECOND query's blob — request 1 answered with query 2's results,
     request 2 never answered.
  3. FT.HYBRID's -1 return made the dispatch site `return consumed_bytes`,
     eating the rest of the batch even when its own reply was flushed.

The fix passes `cmd_end_tok` as every FT.* handler's token bound and makes the
parser's command boundary authoritative at every FT.* dispatch site
(`i = cmd_end_tok - 1`), same as the GEO/NEURON.PKM/VSET sites.

Every check counts *frames* — a reply-per-command assertion is the only thing
that catches a desync.

Run against a single-worker server:

    ./pion-server-dev -p 6391 -w 1
    python3 tests/test_ft_search_pipeline_desync.py --port 6391
"""

import argparse
import socket
import struct
import sys
import random

PASS, FAIL = [], []
DIM = 1536


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  — ' + detail}")


def cmd(*args):
    out = b"*%d\r\n" % len(args)
    for a in args:
        b = a.encode() if isinstance(a, str) else a
        out += b"$%d\r\n%s\r\n" % (len(b), b)
    return out


def connect(port, timeout=6.0):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.settimeout(timeout)
    return s


def read_frame(f):
    """Read exactly one RESP frame off a buffered file object."""
    line = f.readline()
    if not line:
        return None
    t = line[:1]
    if t in (b"+", b"-", b":", b","):
        return line
    if t == b"$":
        n = int(line[1:-2])
        if n == -1:
            return line
        return line + f.read(n + 2)
    if t in (b"*", b"~", b">"):
        n = int(line[1:-2])
        out = line
        for _ in range(max(n, 0)):
            sub = read_frame(f)
            if sub is None:
                return out
            out += sub
        return out
    return line


def vec_blob(seed):
    rnd = random.Random(seed)
    return struct.pack(f"<{DIM}f", *(rnd.gauss(0, 1) for _ in range(DIM)))


def ft_search(blob, k=5):
    return cmd(
        "FT.SEARCH", "pidx",
        f"*=>[KNN {k} @vector $vec EF_RUNTIME 64 as score]",
        "PARAMS", "2", "vec", blob,
        "SORTBY", "score", "LIMIT", "0", str(k), "DIALECT", "2",
    )


def search_ids(frame):
    """Doc ids from an FT.SEARCH reply frame (RESP array: count, id, fields...)."""
    ids = []
    for ln in frame.split(b"\r\n"):
        if ln.startswith(b"v:"):
            ids.append(ln)
    return ids


def setup_index(port, n=64):
    """FT.CREATE + HSET n vectors + FT.OPTIMIZE, one frame per command."""
    s = connect(port, timeout=30.0)
    f = s.makefile("rb")
    s.sendall(cmd("FT.CREATE", "pidx", "ON", "HASH", "PREFIX", "1", "v:",
                  "SCHEMA", "vector", "VECTOR", "HNSW", "6",
                  "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2"))
    read_frame(f)
    for i in range(n):
        s.sendall(cmd("HSET", f"v:{i}", "vector", vec_blob(i)))
        read_frame(f)
    s.sendall(cmd("FT.OPTIMIZE", "pidx"))
    read_frame(f)
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6391)
    ap.add_argument("--skip-setup", action="store_true",
                    help="index already built (e.g. by a prior bench run)")
    args = ap.parse_args()

    if not args.skip_setup:
        print("[setup] building 64-vector index ...")
        setup_index(args.port)

    # ── 1. FT.SEARCH + PING: the PONG must arrive ──────────────────────────
    s = connect(args.port)
    f = s.makefile("rb")
    s.sendall(ft_search(vec_blob(1000)) + cmd("PING"))
    r1 = read_frame(f)
    r2 = read_frame(f)
    check("FT.SEARCH+PING: search reply is an array",
          r1 is not None and r1[:1] == b"*", repr(r1[:20] if r1 else r1))
    check("FT.SEARCH+PING: PONG survives", r2 == b"+PONG\r\n", repr(r2))
    s.close()

    # ── 2. FT.SEARCH + FT.SEARCH with different blobs: two replies, ────────
    #      and reply 1 must match query 1 (not query 2's blob).
    sa = connect(args.port)
    fa = sa.makefile("rb")
    sa.sendall(ft_search(vec_blob(7)))
    solo_a = search_ids(read_frame(fa))
    sa.sendall(ft_search(vec_blob(8)))
    solo_b = search_ids(read_frame(fa))
    sa.close()
    check("distinct queries return distinct top-k (test is meaningful)",
          solo_a != solo_b, f"{solo_a} == {solo_b}")

    sp = connect(args.port)
    fp = sp.makefile("rb")
    sp.sendall(ft_search(vec_blob(7)) + ft_search(vec_blob(8)))
    p1 = read_frame(fp)
    p2 = read_frame(fp)
    check("FT+FT pipelined: both replies arrive",
          p1 is not None and p2 is not None and p2[:1] == b"*",
          f"r1={None if p1 is None else p1[:16]!r} r2={None if p2 is None else p2[:16]!r}")
    if p1 is not None and p2 is not None:
        check("FT+FT pipelined: reply 1 answers query 1 (no blob poisoning)",
              search_ids(p1) == solo_a,
              f"pipelined={search_ids(p1)} solo={solo_a}")
        check("FT+FT pipelined: reply 2 answers query 2",
              search_ids(p2) == solo_b,
              f"pipelined={search_ids(p2)} solo={solo_b}")
    sp.close()

    # ── 3. Depth-8 pipeline: exactly 8 replies, correctly paired ───────────
    sd = connect(args.port)
    fd_ = sd.makefile("rb")
    seeds = [200 + i for i in range(8)]
    solos = []
    for sd_seed in seeds:
        sd.sendall(ft_search(vec_blob(sd_seed)))
        solos.append(search_ids(read_frame(fd_)))
    payload = b"".join(ft_search(vec_blob(x)) for x in seeds)
    sd.sendall(payload)
    got = []
    for _ in range(8):
        fr = read_frame(fd_)
        if fr is None:
            break
        got.append(search_ids(fr))
    check("depth-8 pipeline: 8 replies", len(got) == 8, f"got {len(got)}")
    check("depth-8 pipeline: replies pair with requests", got == solos,
          "reply order/content mismatch")
    sd.close()

    # ── 4. FT.INFO + PING and FT.SEARCH + ECHO: other FT.* sites ───────────
    s4 = connect(args.port)
    f4 = s4.makefile("rb")
    s4.sendall(cmd("FT.INFO", "pidx") + cmd("PING"))
    _ = read_frame(f4)
    r = read_frame(f4)
    check("FT.INFO+PING: PONG survives", r == b"+PONG\r\n", repr(r))
    s4.sendall(ft_search(vec_blob(9)) + cmd("ECHO", "tail"))
    _ = read_frame(f4)
    r = read_frame(f4)
    check("FT.SEARCH+ECHO: ECHO survives", r == b"$4\r\ntail\r\n", repr(r))
    s4.close()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
