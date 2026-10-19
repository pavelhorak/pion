#!/usr/bin/env python3
"""delta_cost.py: what the exact delta costs a query (SPIKE.md, B's first kill criterion).

Against a running Pion (the drop-in branch): FT.CREATE, HSET the 50K OpenAI train vectors,
wait for the automatic build (INFO vector_index: graph), then time KNN 100 / EF_RUNTIME 150
queries over one connection (c=1, so the server time is what moves), first with no delta,
then after DELTA more vectors (the test set, below the 1/10 rebuild threshold), then the
same queries against the exact scan alone (FT.DROPINDEX, re-ingest, query before the build
would start is not reliable, so the exact-only figure is taken from the delta slope).

Usage: <venv with redis, numpy, pandas>/bin/python delta_cost.py <port> <dataset dir> [DELTA]
Prints p50/p90 latency in ms for each phase and the added ms per 1,000 delta rows.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import redis

port = int(sys.argv[1])
ds = sys.argv[2]
DELTA = int(sys.argv[3]) if len(sys.argv) > 3 else 4900
r = redis.Redis(port=port)


def vecs(path):
    df = pd.read_parquet(path)
    col = "emb" if "emb" in df.columns else df.columns[-1]
    return np.stack(df[col].to_numpy()).astype(np.float32)


def state():
    i = r.info()
    return i.get("vector_index"), i.get("vector_graph_rows"), i.get("vector_exact_rows")


def wait(want, timeout=300):
    t = time.time()
    while time.time() - t < timeout:
        if state()[0] == want:
            return True
        time.sleep(0.2)
    return False


def ingest(v, start):
    p = r.pipeline(transaction=False)
    for i, x in enumerate(v):
        p.hset(str(start + i), mapping={"id": str(start + i), "vector": x.tobytes()})
        if i % 1000 == 999:
            p.execute()
    p.execute()


def timed(qs, n=1000):
    lat = []
    for q in qs[:n]:
        t = time.perf_counter()
        r.execute_command("FT.SEARCH", "idx", "*=>[KNN 100 @vector $v EF_RUNTIME 150 AS score]",
                          "PARAMS", 2, "v", q.tobytes(), "RETURN", 1, "id", "LIMIT", 0, 100,
                          "DIALECT", 2)
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat[50:])          # drop the warm-up
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 90))


train = vecs(os.path.join(ds, "shuffle_train.parquet") if os.path.exists(os.path.join(ds, "shuffle_train.parquet"))
             else os.path.join(ds, "train.parquet"))
test = vecs(os.path.join(ds, "test.parquet"))
print(f"train {train.shape}, test {test.shape}")
r.execute_command("FT.CREATE", "idx", "ON", "HASH", "SCHEMA", "id", "TAG", "vector", "VECTOR", "HNSW", "10",
                  "TYPE", "FLOAT32", "DIM", str(train.shape[1]), "DISTANCE_METRIC", "COSINE",
                  "M", "16", "EF_CONSTRUCTION", "128")
ingest(train, 0)
print("ingested; waiting for the build:", wait("graph"), state())
base = timed(test)
print(f"graph only:         p50 {base[0]:.3f} ms  p90 {base[1]:.3f} ms")
extra = train[:DELTA] * 1.0001          # new keys, train rows scaled (cost does not depend on content)
ingest(extra, len(train))
time.sleep(0.5)
st = state()
print("after delta:", st)
rows = int(st[2])
withd = timed(test)
print(f"graph + {rows} delta: p50 {withd[0]:.3f} ms  p90 {withd[1]:.3f} ms")
print(f"added: {withd[0] - base[0]:.3f} ms at p50 = {(withd[0] - base[0]) / rows * 1000:.3f} ms per 1,000 rows; "
      f"{(withd[0] - base[0]) / base[0] * 100:.0f}% of the graph-only p50 (client RTT included in it)")
