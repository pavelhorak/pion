#!/usr/bin/env python3
"""M14 Phase 3: 128K token scale test with persistent connection.

Tests ATTEND.STORE + ATTEND.QUERY at 4K and 128K token scales using a single
persistent TCP connection.

It used to print its numbers and exit 0 whatever happened — "STORE FAILED",
a partial store and empty query replies all passed the gate. Now it FAILS
unless every token is stored, the layer finalizes, every query answers, and a
query with a STORED key returns that key's own value row (values are unique
per token, so a wrong row is visible). Timings are reported, never asserted.
"""

import sys
import time

sys.path.insert(0, "vllm-pion")

import numpy as np
from vllm_pion.attention_client import PionAttentionClient

HOST = "127.0.0.1"
PORT = 1974
KEY_DIM = 128
VAL_DIM = 128


def run_scale(client: PionAttentionClient, n_tokens: int, batch_size: int = 500):
    """Store n_tokens and measure query latency."""
    session_id = f"scale128k_{n_tokens}"
    layer_id = 0

    sid = client.create_session(session_id, KEY_DIM, VAL_DIM)
    if sid < 0:
        print(f"  {n_tokens:>7,}: CREATE FAILED")
        return None

    # Store. Value row i encodes i in bits: a unique, checkable marker.
    np.random.seed(42)
    t0 = time.perf_counter()
    stored = 0
    probe_keys = {}
    for batch_start in range(0, n_tokens, batch_size):
        n = min(batch_size, n_tokens - batch_start)
        keys = np.random.randn(n, KEY_DIM).astype(np.float32)
        # Token id in BITS: INT8 value quantization (one scale per layer) keeps
        # 0.0/1.0 exact, where an integer id would round to a neighbour.
        vals = ((np.arange(batch_start, batch_start + n)[:, None] >> np.arange(VAL_DIM)) & 1).astype(np.float32)
        if batch_start % (batch_size * 16) == 0:
            probe_keys[batch_start] = keys[0].copy()
        ok = client.store_tokens(session_id, layer_id, keys, vals)
        if not ok:
            print(f"  {n_tokens:>7,}: STORE FAILED at {stored}")
            break
        stored += n
        if stored % 10000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {stored:>7,} tokens ({stored / elapsed:,.0f} tok/s, {elapsed:.1f}s)")
    t_store = time.perf_counter() - t0
    store_rate = stored / t_store if t_store > 0 else 0

    if stored < n_tokens:
        print(f"  {n_tokens:>7,}: Only stored {stored:,} of {n_tokens:,}")

    # Finalize HNSW (compact once after all inserts)
    print(f"    Finalizing HNSW index ({stored:,} tokens)...")
    t_fin = time.perf_counter()
    fin_ok = client.finalize_layer(session_id, layer_id)
    t_fin_done = time.perf_counter() - t_fin
    print(f"    Finalized in {t_fin_done:.1f}s")

    # Query latency (20 iterations) — every query must answer.
    latencies = []
    empty = 0
    for _ in range(20):
        q = np.random.randn(KEY_DIM).astype(np.float32)
        t0 = time.perf_counter()
        result = client.query_topk(session_id, layer_id, q, k=5)
        latencies.append((time.perf_counter() - t0) * 1000)
        empty += result is None
    # Self-queries: a stored key must come back with its own value row.
    wrong = []
    for tok, key in probe_keys.items():
        r = client.query_topk(session_id, layer_id, key, k=1)
        got = None if r is None else int(sum(int(round(float(b))) << bi
                                          for bi, b in enumerate(np.frombuffer(r, dtype=np.float32)[:20])))
        if got != tok:
            wrong.append((tok, got))

    avg_lat = sum(latencies) / len(latencies)
    qps = 1000 / avg_lat if avg_lat > 0 else 0
    mem_mb = stored * (KEY_DIM + VAL_DIM) * 4 / (1024 * 1024)

    print(f"  {stored:>7,} tokens: store={t_store:.1f}s ({store_rate:,.0f} tok/s)  "
          f"query={avg_lat:.2f}ms ({qps:,.0f} QPS)  mem≈{mem_mb:.0f}MB  "
          f"40L≈{mem_mb * 40 / 1024:.1f}GB")

    return {
        "requested": n_tokens,
        "finalized": bool(fin_ok),
        "empty_queries": empty,
        "self_query_wrong": wrong,
        "self_queries": len(probe_keys),
        "tokens": stored,
        "store_s": t_store,
        "store_rate": store_rate,
        "query_ms": avg_lat,
        "qps": qps,
        "mem_mb": mem_mb,
    }


def main():
    print("M14 Phase 3: 128K Token Scale Test (persistent connection)")
    print(f"Server: {HOST}:{PORT}, dim={KEY_DIM}")
    print()

    client = PionAttentionClient(HOST, PORT, timeout=60)

    results = {}
    # Test 128K directly (skip smaller scales to save memory — each session allocs ~110MB per layer)
    for n in [4000, 128000]:
        print(f"--- {n:,} tokens ---")
        r = run_scale(client, n, batch_size=500)
        if r:
            results[n] = r
        print()

    client.close()

    # Summary
    print("=" * 75)
    print(f"{'Tokens':>10} {'Store(tok/s)':>14} {'Query(ms)':>12} {'QPS':>8} {'Mem(MB)':>10} {'40L(GB)':>10}")
    print("-" * 75)
    for n, r in sorted(results.items()):
        print(f"{r['tokens']:>10,} {r['store_rate']:>14,.0f} {r['query_ms']:>12.2f} "
              f"{r['qps']:>8,.0f} {r['mem_mb']:>8.0f}MB {r['mem_mb']*40/1024:>9.1f}")

    # 128K plan validation
    if 128000 in results:
        r = results[128000]
        print(f"\n--- 128K Plan Validation ---")
        # The target is server-side; this is the Python client's round trip
        # (~1.4 ms of it is the client and socket, gh #391).
        print(f"  Query latency: {r['query_ms']:.2f}ms client round trip (server-side target: <0.30ms)")
        print(f"  40-layer total: {r['query_ms'] * 40:.1f}ms (target: <12ms)")
        print(f"  Store throughput: {r['store_rate']:,.0f} tok/s (target: >100K)")
        print(f"  Memory (40 layers): {r['mem_mb'] * 40 / 1024:.1f}GB (target: <3GB)")
    elif 32000 in results:
        r = results[32000]
        import math
        est_lat = r["query_ms"] * math.log(128000) / math.log(r["tokens"])
        print(f"\n--- 128K Extrapolation (from {r['tokens']:,}) ---")
        print(f"  Query latency: ~{est_lat:.2f}ms (target: <0.30ms)")
        print(f"  40-layer total: ~{est_lat * 40:.1f}ms (target: <12ms)")


    # Correctness — the part that can fail.
    fails = []
    for n in (4000, 128000):
        r = results.get(n)
        if r is None:
            fails.append(f"{n}: session could not be created")
            continue
        if r["tokens"] != r["requested"]:
            fails.append(f"{n}: stored {r['tokens']} of {r['requested']} tokens")
        if not r["finalized"]:
            fails.append(f"{n}: ATTEND.FINALIZE failed")
        if r["empty_queries"]:
            fails.append(f"{n}: {r['empty_queries']}/20 queries returned nothing")
        if r["self_query_wrong"]:
            fails.append(f"{n}: {len(r['self_query_wrong'])}/{r['self_queries']} stored keys did not "
                         f"retrieve their own value: {r['self_query_wrong'][:4]}")
    for f in fails:
        print("FAIL " + f)
    print("PASS" if not fails else f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
