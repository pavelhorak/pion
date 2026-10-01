#!/usr/bin/env python3
"""Pipelined FT.SEARCH bench against a pre-loaded Pion (or Redis) instance.

Companion to vectordbbench's mp_runner — that runner is synchronous-per-process
(one query in flight per connection, blocked on socket round-trip ~80% of wall
time at C=10 on Apple Silicon). This harness uses redis-py pipelines to keep
M × K queries in flight, where M is the process count and K is the pipeline
depth. Lets workers actually saturate so server-side CPU optimizations show up.

Reads the same test queries vectordbbench uses (`test.parquet`) and the same
ground-truth (`neighbors.parquet`), so recall is comparable to the standard bench.

Usage:
    # Pre-load Pion via the standard bench (or another mechanism), then:
    python3 benchmarks/VectorDBBench/pipelined_bench.py \\
        --port 6395 --processes 10 --pipeline-depth 8 --duration 30

Output:
    PIPELINED_BENCH_RESULT processes=N pipeline=K duration=S \\
        qps=X recall=R p99_ms=L
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import struct
import sys
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import redis


DATASET_DIR = "/private/tmp/vectordb_bench/dataset/openai/openai_small_50k"
INDEX_NAME = "index"  # vectordbbench's INDEX_NAME constant (redis.py line 14)


def load_queries(case_dir: str = DATASET_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Returns (queries[N, dim] float32, ground_truth_neighbors[N, k] int)."""
    test_path = os.path.join(case_dir, "test.parquet")
    neighbors_path = os.path.join(case_dir, "neighbors.parquet")
    test_table = pq.read_table(test_path)
    neighbors_table = pq.read_table(neighbors_path)
    queries = np.stack([np.array(row, dtype=np.float32) for row in test_table.column("emb").to_pylist()])
    truth = np.stack([np.array(row, dtype=np.int64) for row in neighbors_table.column("neighbors_id").to_pylist()])
    return queries, truth


def build_query_args(query_blob: bytes, k: int, ef: int) -> tuple:
    """Build raw FT.SEARCH arg tuple for execute_command."""
    return (
        "FT.SEARCH",
        INDEX_NAME,
        f"*=>[KNN {k} @vector $vec EF_RUNTIME {ef} as score]",
        "PARAMS", "2", "vec", query_blob,
        "SORTBY", "score",
        "LIMIT", "0", str(k),
        "DIALECT", "2",
    )


def parse_search_response(resp: Any) -> list[int]:
    """FT.SEARCH RESP: [count, doc_id, [field_name, field_val, ...], doc_id, ...]"""
    if not isinstance(resp, list) or len(resp) < 1:
        return []
    ids: list[int] = []
    for i in range(1, len(resp), 2):
        if i >= len(resp):
            break
        doc_id = resp[i]
        if isinstance(doc_id, bytes):
            doc_id = doc_id.decode()
        try:
            ids.append(int(doc_id))
        except (ValueError, TypeError):
            pass
    return ids


def worker_loop(
    proc_id: int,
    port: int,
    queries: np.ndarray,
    truth: np.ndarray,
    pipeline_depth: int,
    k: int,
    ef: int,
    duration_s: float,
    sync_barrier: Any,
    stats_q: Any,
):
    """One worker process: pipelines K queries at a time, measures throughput + recall."""
    r = redis.Redis(host="localhost", port=port, decode_responses=False, socket_timeout=30.0)

    # Pre-pack all query blobs once
    blobs = [q.tobytes() for q in queries]
    n_queries = len(queries)

    # Sync all workers to start at the same instant
    sync_barrier.wait()
    start = time.perf_counter()
    deadline = start + duration_s

    completed = 0
    correct_at_k = 0  # sum of |returned ∩ truth_top_k| across all queries
    total_truth = 0   # sum of k across all queries (for recall denominator)
    latencies: list[float] = []
    cursor = (proc_id * 7919) % n_queries  # offset starting query per process

    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        # Queue K queries (fresh pipe each iteration — avoids redis-py state quirks)
        pipe = r.pipeline(transaction=False)
        batch_indices = []
        for _ in range(pipeline_depth):
            batch_indices.append(cursor)
            args = build_query_args(blobs[cursor], k, ef)
            pipe.execute_command(*args)
            cursor = (cursor + 1) % n_queries

        t_send = time.perf_counter()
        try:
            results = pipe.execute(raise_on_error=False)
        except Exception as e:
            print(f"[proc {proc_id}] pipe.execute error: {e}", file=sys.stderr)
            continue
        t_recv = time.perf_counter()

        per_query_latency = (t_recv - t_send) / pipeline_depth
        for i, raw in enumerate(results):
            if isinstance(raw, Exception):
                continue
            returned_ids = parse_search_response(raw)
            qi = batch_indices[i]
            truth_top_k = set(int(x) for x in truth[qi][:k])
            correct_at_k += sum(1 for rid in returned_ids if rid in truth_top_k)
            total_truth += k
            latencies.append(per_query_latency * 1000.0)  # ms
            completed += 1

    elapsed = time.perf_counter() - start
    qps = completed / elapsed if elapsed > 0 else 0.0
    recall = correct_at_k / total_truth if total_truth > 0 else 0.0
    p99_ms = float(np.percentile(latencies, 99)) if latencies else 0.0
    p95_ms = float(np.percentile(latencies, 95)) if latencies else 0.0
    p50_ms = float(np.percentile(latencies, 50)) if latencies else 0.0
    stats_q.put({
        "proc_id": proc_id,
        "completed": completed,
        "elapsed": elapsed,
        "qps": qps,
        "recall": recall,
        "p99_ms": p99_ms,
        "p95_ms": p95_ms,
        "p50_ms": p50_ms,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=6395)
    p.add_argument("--processes", type=int, default=10, help="parallel client processes (M)")
    p.add_argument("--pipeline-depth", type=int, default=8, help="queries in flight per process (K)")
    p.add_argument("--duration", type=float, default=30.0, help="seconds to run")
    p.add_argument("--k", type=int, default=100, help="KNN k")
    p.add_argument("--ef", type=int, default=150, help="HNSW ef_runtime")
    p.add_argument("--warmup", type=float, default=2.0, help="warmup seconds before timing")
    p.add_argument("--dataset-dir", default=DATASET_DIR)
    args = p.parse_args()

    # Sanity check: server reachable + index ready
    r = redis.Redis(host=args.host, port=args.port)
    try:
        r.ping()
    except Exception as e:
        print(f"FATAL: cannot connect to {args.host}:{args.port}: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        info = r.execute_command("FT.INFO", INDEX_NAME)
    except Exception as e:
        print(f"FATAL: FT.INFO {INDEX_NAME} failed: {e}", file=sys.stderr)
        print("Pre-load the dataset with the standard bench first.", file=sys.stderr)
        sys.exit(1)

    print(f"  Connected to {args.host}:{args.port}, FT.INFO ok ({len(info)} fields)")
    print(f"  Loading test queries from {args.dataset_dir}...")
    queries, truth = load_queries(args.dataset_dir)
    print(f"  Loaded {len(queries)} queries (dim={queries.shape[1]}), truth k_max={truth.shape[1]}")

    # Optional warmup (single-process, throws away stats)
    if args.warmup > 0:
        print(f"  Warming up for {args.warmup}s...")
        warm_q = mp.Queue()
        warm_b = mp.Barrier(1)
        warm_p = mp.Process(target=worker_loop, args=(
            0, args.port, queries, truth, args.pipeline_depth, args.k, args.ef,
            args.warmup, warm_b, warm_q,
        ))
        warm_p.start()
        warm_p.join()
        # Discard warmup stats

    print(f"  Running M={args.processes} processes × K={args.pipeline_depth} pipeline depth = "
          f"{args.processes * args.pipeline_depth} queries in flight, duration={args.duration}s")

    barrier = mp.Barrier(args.processes)
    stats_q: Any = mp.Queue()
    procs = []
    for i in range(args.processes):
        proc = mp.Process(target=worker_loop, args=(
            i, args.port, queries, truth, args.pipeline_depth, args.k, args.ef,
            args.duration, barrier, stats_q,
        ))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    # Aggregate
    all_stats = []
    while not stats_q.empty():
        all_stats.append(stats_q.get())

    if not all_stats:
        print("FATAL: no worker stats collected", file=sys.stderr)
        sys.exit(1)

    total_completed = sum(s["completed"] for s in all_stats)
    avg_elapsed = sum(s["elapsed"] for s in all_stats) / len(all_stats)
    aggregate_qps = total_completed / avg_elapsed if avg_elapsed > 0 else 0.0

    # Recall: sum across processes
    total_recall_num = sum(s["recall"] * s["completed"] for s in all_stats)
    total_recall_den = sum(s["completed"] for s in all_stats)
    aggregate_recall = total_recall_num / total_recall_den if total_recall_den > 0 else 0.0

    p99_ms = max(s["p99_ms"] for s in all_stats)
    p95_ms = max(s["p95_ms"] for s in all_stats)
    p50_ms = sum(s["p50_ms"] for s in all_stats) / len(all_stats)

    print()
    print(f"PIPELINED_BENCH_RESULT processes={args.processes} pipeline={args.pipeline_depth} "
          f"duration={args.duration}s qps={aggregate_qps:.0f} recall={aggregate_recall:.4f} "
          f"p99_ms={p99_ms:.2f} p95_ms={p95_ms:.2f} p50_ms={p50_ms:.2f} "
          f"completed={total_completed}")


if __name__ == "__main__":
    main()
