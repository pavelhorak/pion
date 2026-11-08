#!/usr/bin/env python3
"""A build must not depend on what the server built before it.

HNSW gives each node a level. Those levels used to come from the process's
one random stream (`random_float64` in `_random_level`), which nothing
reseeded: every build, FT.OPTIMIZE or automatic, advanced it. So the graph
FT.OPTIMIZE built from the same vectors depended on how many builds the
server had run before, including the automatic ones that start after 200 ms
of quiet ingest. On the tight clusters below, recall@10 read anywhere from
0.80 to 1.00 for the same data (x86 CI, 22 timings, any binary).

Now a node's level is a fixed function of its position in the build, so the
same vectors in the same order get the same graph however the server got
there. This test builds the same data three times on one server and requires
identical replies:

  A  ingest in one go, FT.OPTIMIZE;
  B  ingest half, wait for the automatic build, ingest the rest, wait again,
     FT.OPTIMIZE (and A's build is history for B);
  C  A again, after A's and B's builds.

1,500 vectors build on one lane (the build goes parallel from 4,096), so
nothing but the levels could make the graphs differ. Standard library only.

    python3 tests/test_hnsw_build_history.py --port 1974
"""
import argparse
import math
import random
import sys
import time

from test_vector_recall_smoke import DIM, CLUSTERS, PER, K, Resp, pack, unit

QUERIES = 60


def info_auto_builds(r) -> int:
    text = r.call("INFO").decode()
    for line in text.splitlines():
        if line.startswith("vector_auto_builds:"):
            return int(line.split(":", 1)[1])
    return -1


def build_and_query(r, index, prefix, vectors, queries, pause: bool):
    r.call("FT.CREATE", index, "ON", "HASH", "PREFIX", "1", prefix, "SCHEMA", "vec",
           "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", DIM, "DISTANCE_METRIC", "L2")
    cmds = [("HSET", f"{prefix}{name}", "vec", blob) for name, blob in vectors]
    half = len(cmds) // 2
    batches = [cmds[:half], cmds[half:]] if pause else [cmds]
    for batch in batches:
        for i in range(0, len(batch), 250):
            r.pipeline(batch[i:i + 250])
        if pause:
            time.sleep(1.5)   # 200 ms of quiet starts a build; 1,500 rows build in well under 1 s
    r.call("FT.OPTIMIZE", index)
    replies = []
    for q in queries:
        res = r.call("FT.SEARCH", index, f"*=>[KNN {K} @vec $q EF_RUNTIME 150 as score]",
                     "PARAMS", "2", "q", q, "SORTBY", "score",
                     "LIMIT", "0", str(K), "DIALECT", "2")
        rows = []
        for i in range(1, len(res), 2):
            key = res[i].decode()
            fields = res[i + 1]
            score = fields[fields.index(b"score") + 1].decode() if b"score" in fields else None
            rows.append((key[len(prefix):], score))
        replies.append(rows)
    r.call("FT.DROPINDEX", index)
    r.pipeline([("DEL", c[1]) for c in cmds])
    return replies


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()
    rng = random.Random(0x5EED)

    centres = [unit([rng.gauss(0, 1) for _ in range(DIM)]) for _ in range(CLUSTERS)]
    vectors = []
    for c in range(CLUSTERS):
        for p in range(PER):
            v = [x + rng.gauss(0, 0.05 / math.sqrt(DIM)) for x in centres[c]]
            vectors.append((f"{c}:{p}", pack(v)))
    queries = []
    for _ in range(QUERIES):
        c = rng.randrange(CLUSTERS)
        queries.append(pack([x + rng.gauss(0, 0.02 / math.sqrt(DIM)) for x in centres[c]]))

    r = Resp(args.port)
    auto0 = info_auto_builds(r)
    a = build_and_query(r, "bh_a", "bha:", vectors, queries, pause=False)
    b = build_and_query(r, "bh_b", "bhb:", vectors, queries, pause=True)
    auto1 = info_auto_builds(r)
    c = build_and_query(r, "bh_c", "bhc:", vectors, queries, pause=False)

    fails = 0
    # B must have had automatic builds in its history, or it tests less
    # than it claims (A's build is history for it either way).
    if auto0 >= 0 and auto1 - auto0 < 1:
        print(f"FAIL: no automatic build ran during B (vector_auto_builds {auto0} -> {auto1})")
        fails += 1
    for name, other in (("B", b), ("C", c)):
        differ = [i for i in range(QUERIES) if other[i] != a[i]]
        if differ:
            i = differ[0]
            print(f"FAIL: build {name} answered {len(differ)}/{QUERIES} queries differently from A; "
                  f"query {i}: A {a[i][:3]} vs {name} {other[i][:3]}")
            fails += 1
        else:
            print(f"ok: build {name} answered all {QUERIES} queries as A did")
    if any(len(rows) != K for rows in a):
        print(f"FAIL: a query returned fewer than {K} rows")
        fails += 1
    print(f"automatic builds during B: {auto1 - auto0 if auto0 >= 0 else 'n/a'}")
    print("PASS" if fails == 0 else f"{fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
