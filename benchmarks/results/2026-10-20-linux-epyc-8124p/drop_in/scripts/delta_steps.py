#!/usr/bin/env python3
"""delta_steps.py: phase B's 500K block (phase_b_2026-10-19/BOX_B.md block 4, as corrected on
2026-10-20). What the delta costs a query at 500K, at several sizes, and how long a rebuild of
the whole index takes, against Pion B started with PION_VECTOR_SETTLE_MS=30000 and a delta cap
that does not bind (d_block.sh). The settle is 30 s, not 0 as on the Mac: at 500K the load is ten
times longer, and a 200 ms gap in it would start the first build early; with the settle off, the
rows after that build's snapshot (under 1/10 of it) would then never be folded in. 30 s is longer
than any timing phase here (1,000 queries), so it cannot fire during a step; the script stops if
the graph moves anyway.

A copy of spike_2026-10-19/delta_cost.py (same FT.CREATE, same queries: KNN 100, EF_RUNTIME 150,
one connection, the first 50 dropped as warm-up) with three changes:
  - the delta is grown in steps (STEPS rows, cumulative) and timed at each, so the slope comes
    from the box, not from one point;
  - every step stays under 1/10 of the index: at 500K a rebuild starts at 50,000 delta rows
    whatever PION_VECTOR_DELTA_MAX says (vector_auto.mojo: cap = max(256, min(built/10,
    delta_max))), and a 10,000-row delta with the default cap of 8,192 would have started one;
  - after the last step, rows are added past 1/10 so a rebuild starts, and queries are timed in
    batches WHILE it runs (the state just before a publish: the largest delta, plus the build
    competing for the CPU), then once more after it.
Builds are followed through INFO vector_auto_builds / vector_build_running, not row counts: a
rebuild can start while the last rows still arrive (searches in the last second let it start
without a pause), and the rows its snapshot missed stay in the delta when they are under the cap
and the settle's 1% bar (Mac test, 2026-10-20: 97 of 56,000).
The slope is fitted over the delta > 0 points only: a delta costs a fixed amount the moment it
exists (~0.09 ms on the Mac), and a fit through the zero point would fold that into the slope.

Usage: <venv with redis, numpy, pandas>/bin/python delta_steps.py <port> <dataset dir> [STEPS]
STEPS: comma-separated cumulative delta sizes, default 10000,25000,45000.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import redis

port = int(sys.argv[1])
ds = sys.argv[2]
STEPS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "10000,25000,45000").split(",")]
r = redis.Redis(port=port)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def vecs(path):
    df = pd.read_parquet(path)
    col = "emb" if "emb" in df.columns else df.columns[-1]
    return np.stack(df[col].to_numpy()).astype(np.float32)


def state():
    i = r.info()
    return (i.get("vector_index"), int(i.get("vector_graph_rows") or 0), int(i.get("vector_exact_rows") or 0),
            int(i.get("vector_auto_builds") or 0), int(i.get("vector_build_running") or 0))


def wait_built(timeout, held=35):
    """Until a graph serves, no build runs, and either nothing is exact or the state has held for
    `held` s (longer than the 30 s settle, so a delta the settle could fold would have been).
    Returns (seconds waited, state), or (None, state)."""
    t = time.time(); last = None; since = t
    while time.time() - t < timeout:
        st = state()
        if st != last:
            last, since = st, time.time()
        if st[0] == "graph" and st[4] == 0 and (st[2] == 0 or time.time() - since >= held):
            return time.time() - t, st
        time.sleep(0.5)
    return None, last


def ingest(v, start):
    p = r.pipeline(transaction=False)
    for i, x in enumerate(v):
        p.hset(str(start + i), mapping={"id": str(start + i), "vector": x.tobytes()})
        if i % 1000 == 999:
            p.execute()
    p.execute()


def query(q):
    r.execute_command("FT.SEARCH", "idx", "*=>[KNN 100 @vector $v EF_RUNTIME 150 AS score]",
                      "PARAMS", 2, "v", q.tobytes(), "RETURN", 1, "id", "LIMIT", 0, 100,
                      "DIALECT", 2)


def timed(qs, n=1000, warm=50):
    lat = []
    for q in qs[:n]:
        t = time.perf_counter()
        query(q)
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat[warm:])
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 90))


train = vecs(os.path.join(ds, "shuffle_train.parquet") if os.path.exists(os.path.join(ds, "shuffle_train.parquet"))
             else os.path.join(ds, "train.parquet"))
test = vecs(os.path.join(ds, "test.parquet"))
N = len(train)
log(f"train {train.shape}, test {test.shape}, steps {STEPS}")
assert max(STEPS) < N // 10, f"the last step must stay under 1/10 of the index ({N // 10})"
r.execute_command("FT.CREATE", "idx", "ON", "HASH", "SCHEMA", "id", "TAG", "vector", "VECTOR", "HNSW", "10",
                  "TYPE", "FLOAT32", "DIM", str(train.shape[1]), "DISTANCE_METRIC", "COSINE",
                  "M", "16", "EF_CONSTRUCTION", "128")
t = time.time()
ingest(train, 0)
log(f"ingested {N} rows in {time.time() - t:.1f} s; state (index, graph, exact, builds, running) {state()}")
w, st = wait_built(1800)
if w is None:
    log(f"first build NOT done after 1800 s; state {st}")
    sys.exit(2)
log(f"first build: settled {w:.1f} s after the ingest ended; state {st}")

e0 = st[2]
base = timed(test)
log(f"RESULT delta {e0}: p50 {base[0]:.3f} ms  p90 {base[1]:.3f} ms (graph{' only' if e0 == 0 else ', rows the build missed'})")
pts = []
prev = 0
for size in STEPS:
    b0 = state()[3]
    ingest(train[prev:size] * 1.0001, N + prev)   # new keys; content does not change the cost
    prev = size
    time.sleep(0.5)
    st = state()
    if st[3] != b0 or st[4]:
        log(f"STOP: a rebuild started during the step; state {st}")
        sys.exit(3)
    e = st[2]
    p50, p90 = timed(test)
    pts.append((e, p50))
    log(f"RESULT delta {e}: p50 {p50:.3f} ms  p90 {p90:.3f} ms  added {p50 - base[0]:.3f} ms "
        f"= {(p50 - base[0]) / base[0] * 100:.1f}% of the delta-{e0} p50 (client RTT included in it)")
    st = state()
    if st[3] != b0 or st[4]:
        log(f"STOP: a rebuild started during the timing; state {st}")
        sys.exit(3)

x = np.array([p[0] for p in pts], dtype=float); y = np.array([p[1] for p in pts])
slope, icept = np.polyfit(x, y, 1)
fixed = icept - base[0]
log(f"RESULT fit over the {len(pts)} delta points: added = {fixed:.3f} ms fixed + {slope * 1000:.4f} ms per 1,000 delta rows")
for d in (34_000, 61_000):
    a = fixed + slope * d
    log(f"RESULT at a {d}-row delta the fit gives +{a:.3f} ms = {a / base[0] * 100:.1f}% of the delta-{e0} p50")

# A rebuild: past 1/10 of the index, then time queries while it runs
b0 = state()[3]
extra = N // 10 - prev + 1000
ingest(train[prev:prev + extra] * 1.0001, N + prev)
total = N + prev + extra
t_trig = time.time()
log(f"added {extra} rows ({total} in all): the delta is past 1/10 of the index, so a rebuild starts "
    f"(during the ingest or after a 200 ms pause); state {state()}")
batches = []
while True:
    st = state()
    if st[3] > b0 and st[4] == 0:
        break
    if time.time() - t_trig > 1800:
        log(f"rebuild NOT done after 1800 s; state {st}")
        sys.exit(4)
    p50, p90 = timed(test[(len(batches) * 200) % 800:], n=200, warm=10)
    batches.append((time.time() - t_trig, st[2], p50, p90, st[4]))
    log(f"{'during the rebuild' if st[4] else 'before it starts'}: t+{batches[-1][0]:.1f} s "
        f"delta {st[2]} p50 {p50:.3f} ms p90 {p90:.3f} ms")
log(f"RESULT rebuild published {time.time() - t_trig:.1f} s after the last row was added (client clock; "
    f"the server's 'indexed in' line is the build's own time); state {st}")
run = [b for b in batches if b[4]]
if run:
    log(f"RESULT during the rebuild: {len(run)} batches of 200 queries, delta {run[0][1]}-{run[-1][1]}, "
        f"p50 median {np.median([b[2] for b in run]):.3f} ms, max {max(b[2] for b in run):.3f} ms")
after = timed(test)
log(f"RESULT after the rebuild: p50 {after[0]:.3f} ms  p90 {after[1]:.3f} ms; state {state()}")
