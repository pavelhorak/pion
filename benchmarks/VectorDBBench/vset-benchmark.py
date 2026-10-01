#!/usr/bin/env python3
"""Redis 8.0 VSET (VADD/VSIM) benchmark — same dataset & metrics as VectorDBBench.

Downloads the OpenAI-SMALL-50K dataset via vectordb-bench, then benchmarks Redis 8.0
VADD/VSIM against the same recall/QPS methodology.

Usage:
    python3 benchmarks/VectorDBBench/vset-benchmark.py [--redis-server /path/to/redis-server]
"""

import argparse
import json
import math
import multiprocessing
import os
import signal
import socket
import struct
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_MD = os.path.join(SCRIPT_DIR, "benchmark_results.md")
PORT = 6399
DIM = 1536
K = 100
CONCURRENCY_LEVELS = [1, 5, 10]
CONCURRENCY_DURATION = 5  # seconds per level


def load_dataset():
    """Load OpenAI-SMALL-50K from vectordb-bench cache or download."""
    import pyarrow.parquet as pq

    # Check vectordb-bench cache first
    cache_dir = "/tmp/vectordb_bench/dataset/openai/openai_small_50k"
    train_path = os.path.join(cache_dir, "shuffle_train.parquet")
    test_path = os.path.join(cache_dir, "test.parquet")
    neighbors_path = os.path.join(cache_dir, "neighbors.parquet")

    if not all(os.path.exists(p) for p in [train_path, test_path, neighbors_path]):
        print("  Dataset not in cache. Running vectordbbench to download...")
        # Use a dummy run that will download the dataset
        venv_bench = os.path.join(PROJECT_ROOT, "venv_zvec/bin/vectordbbench")
        if os.path.exists(venv_bench):
            subprocess.run([venv_bench, "redis", "--help"], capture_output=True, timeout=30)
        if not os.path.exists(train_path):
            print("  ERROR: Could not find or download dataset.")
            print(f"  Run the Pion vector benchmark first to cache the dataset:")
            print(f"    python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 10")
            sys.exit(1)

    train_table = pq.read_table(train_path)
    test_table = pq.read_table(test_path)
    neighbors_table = pq.read_table(neighbors_path)

    train = np.array(train_table.column("emb").to_pylist(), dtype=np.float32)
    train_ids = np.array(train_table.column("id").to_pylist(), dtype=np.int64)
    test = np.array(test_table.column("emb").to_pylist(), dtype=np.float32)
    ncol = "neighbors_id" if "neighbors_id" in neighbors_table.column_names else "neighbors"
    neighbors = np.array(neighbors_table.column(ncol).to_pylist(), dtype=np.int64)
    return train, train_ids, test, neighbors


def redis_cmd(sock, *args):
    """Send a RESP command and read the response."""
    cmd = f"*{len(args)}\r\n"
    for a in args:
        if isinstance(a, bytes):
            cmd = cmd.encode() + f"${len(a)}\r\n".encode() + a + b"\r\n"
        else:
            s = str(a)
            part = f"${len(s)}\r\n{s}\r\n"
            if isinstance(cmd, str):
                cmd += part
            else:
                cmd += part.encode()
    if isinstance(cmd, str):
        cmd = cmd.encode()
    sock.sendall(cmd)
    return read_resp(sock)


def read_resp(sock):
    """Read one RESP response (simple, bulk, array, integer, error)."""
    buf = b""
    while b"\r\n" not in buf:
        buf += sock.recv(4096)
    line, rest = buf.split(b"\r\n", 1)
    prefix = chr(line[0])

    if prefix == "+":
        return line[1:].decode()
    elif prefix == "-":
        return f"ERR:{line[1:].decode()}"
    elif prefix == ":":
        return int(line[1:])
    elif prefix == "$":
        length = int(line[1:])
        if length == -1:
            return None
        while len(rest) < length + 2:
            rest += sock.recv(4096)
        return rest[:length]
    elif prefix == "*":
        count = int(line[1:])
        if count == -1:
            return None
        # Re-inject remaining bytes
        items = []
        # Create a buffered reader for the remaining data
        remaining = rest
        for _ in range(count):
            while b"\r\n" not in remaining:
                remaining += sock.recv(4096)
            el_line, remaining = remaining.split(b"\r\n", 1)
            el_prefix = chr(el_line[0])
            if el_prefix == "$":
                el_len = int(el_line[1:])
                if el_len == -1:
                    items.append(None)
                else:
                    while len(remaining) < el_len + 2:
                        remaining += sock.recv(4096)
                    items.append(remaining[:el_len])
                    remaining = remaining[el_len + 2:]
            elif el_prefix == ":":
                items.append(int(el_line[1:]))
            elif el_prefix == "+":
                items.append(el_line[1:].decode())
            elif el_prefix == "-":
                items.append(f"ERR:{el_line[1:].decode()}")
            else:
                items.append(el_line)
        return items
    return line


def start_redis(redis_server, port):
    """Start Redis 8.0 server."""
    proc = subprocess.Popen(
        [redis_server, "--port", str(port), "--save", "", "--appendonly", "no",
         "--loglevel", "warning", "--dir", "/tmp",
         "--maxmemory", "8gb", "--maxmemory-policy", "noeviction"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    time.sleep(2)
    if proc.poll() is not None:
        print(f"Redis failed to start (exit {proc.returncode})")
        return None
    # Verify
    try:
        s = socket.socket()
        s.settimeout(3)
        s.connect(("127.0.0.1", port))
        resp = redis_cmd(s, "PING")
        s.close()
        assert resp == "PONG"
    except Exception as e:
        print(f"Redis not responding: {e}")
        proc.kill()
        return None
    return proc


def ingest_vset(port, train, train_ids, dim):
    """Ingest vectors using VADD via redis-py pipeline."""
    import redis as _redis
    n = len(train)
    print(f"  Ingesting {n} vectors (VADD)...")
    t0 = time.perf_counter()

    r = _redis.Redis(host="127.0.0.1", port=port, socket_timeout=60)
    BATCH = 500
    for batch_start in range(0, n, BATCH):
        batch_end = min(batch_start + BATCH, n)
        pipe = r.pipeline(transaction=False)
        for i in range(batch_start, batch_end):
            blob = train[i].astype(np.float32).tobytes()
            elem = str(int(train_ids[i]))
            pipe.execute_command("VADD", "vset", "FP32", blob, elem)
        pipe.execute()
        if batch_start % 10000 == 0 and batch_start > 0:
            elapsed = time.perf_counter() - t0
            print(f"    {batch_start}/{n} ({batch_start/elapsed:.0f} vec/s)")

    dur = time.perf_counter() - t0
    r.close()
    print(f"  Ingest: {dur:.2f}s ({n/dur:.0f} vectors/s)")
    return dur


def search_vset_serial(port, test, neighbors, k):
    """Serial search for recall measurement."""
    import redis as _redis
    n_queries = len(test)
    print(f"  Serial search ({n_queries} queries, K={k})...")

    r = _redis.Redis(host="127.0.0.1", port=port, socket_timeout=10)

    recalls = []
    latencies = []

    for qi in range(n_queries):
        blob = test[qi].astype(np.float32).tobytes()
        t0 = time.perf_counter()
        result = r.execute_command("VSIM", "vset", "FP32", blob, "COUNT", str(k))
        lat = time.perf_counter() - t0
        latencies.append(lat)

        if isinstance(result, list):
            found_ids = set()
            for val in result:
                try:
                    if isinstance(val, bytes):
                        found_ids.add(int(val))
                    elif isinstance(val, int):
                        found_ids.add(val)
                except (ValueError, TypeError):
                    pass
            true_ids = set(neighbors[qi][:k].tolist())
            recall = len(found_ids & true_ids) / k if k > 0 else 0
            recalls.append(recall)
        else:
            recalls.append(0.0)

    r.close()

    avg_recall = np.mean(recalls)
    p99 = np.percentile(latencies, 99)
    p95 = np.percentile(latencies, 95)
    avg_lat = np.mean(latencies)
    print(f"  Recall@{k}: {avg_recall:.4f}, Avg latency: {avg_lat*1000:.2f}ms, P99: {p99*1000:.2f}ms")
    return avg_recall, p99, p95, avg_lat


def _search_worker(port, test, k, duration, worker_id, result_queue):
    """Worker process for concurrent search."""
    import redis as _redis
    r = _redis.Redis(host="127.0.0.1", port=port, socket_timeout=10)

    n_queries = len(test)
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        qi = count % n_queries
        blob = test[qi].astype(np.float32).tobytes()
        r.execute_command("VSIM", "vset", "FP32", blob, "COUNT", str(k))
        count += 1

    elapsed = time.perf_counter() - start
    r.close()
    result_queue.put((worker_id, count, elapsed))


def search_vset_concurrent(port, test, k, concurrency_levels, duration):
    """Concurrent search for QPS measurement."""
    results = {}
    for c in concurrency_levels:
        print(f"  Concurrent search (c={c}, {duration}s)...")
        q = multiprocessing.Queue()
        procs = []
        for i in range(c):
            p = multiprocessing.Process(target=_search_worker, args=(port, test, k, duration, i, q))
            procs.append(p)
            p.start()
        for p in procs:
            p.join(timeout=duration + 10)

        total_count = 0
        max_dur = 0
        while not q.empty():
            wid, cnt, dur = q.get()
            total_count += cnt
            max_dur = max(max_dur, dur)

        qps = total_count / max_dur if max_dur > 0 else 0
        results[c] = qps
        print(f"    c={c}: {total_count} queries in {max_dur:.1f}s = {qps:.1f} QPS")

    return results


def main():
    parser = argparse.ArgumentParser(description="Redis 8.0 VSET benchmark")
    parser.add_argument("--redis-server", default="/tmp/redis-8.0/src/redis-server",
                        help="Path to Redis 8.0 server binary")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--duration", type=int, default=CONCURRENCY_DURATION)
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingest (use existing data)")
    args = parser.parse_args()

    if not os.path.exists(args.redis_server):
        print(f"Redis server not found at {args.redis_server}")
        print("Build Redis 8.0: cd /tmp && git clone --depth 1 --branch 8.0.2 https://github.com/redis/redis.git redis-8.0 && cd redis-8.0 && make -j$(nproc)")
        sys.exit(1)

    # Load dataset
    print("Loading dataset (OpenAI-SMALL-50K)...")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "venv_zvec/lib/python3.12/site-packages"))
    train, train_ids, test, neighbors = load_dataset()
    print(f"  Train: {train.shape}, Test: {test.shape}, Neighbors: {neighbors.shape}")

    # Start Redis 8.0
    print(f"Starting Redis 8.0 on port {args.port}...")
    proc = start_redis(args.redis_server, args.port)
    if proc is None:
        sys.exit(1)

    try:
        # Flush
        import redis as _redis
        _r = _redis.Redis(host="127.0.0.1", port=args.port, socket_timeout=5)
        _r.flushall()
        _r.close()

        # Ingest
        if not args.skip_ingest:
            insert_dur = ingest_vset(args.port, train, train_ids, DIM)
        else:
            insert_dur = 0

        # Serial search (recall)
        recall, p99, p95, avg_lat = search_vset_serial(args.port, test, neighbors, args.k)

        # Concurrent search (QPS)
        qps_results = search_vset_concurrent(args.port, test, args.k, CONCURRENCY_LEVELS, args.duration)
        peak_qps = max(qps_results.values())

        # Report
        from datetime import datetime
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*60}")
        print(f"Redis 8.0 VSET Benchmark Results ({run_timestamp})")
        print(f"{'='*60}")
        print(f"  Insert:     {insert_dur:.2f}s")
        print(f"  Recall@{args.k}:  {recall:.4f}")
        print(f"  P99 latency: {p99*1000:.2f}ms")
        print(f"  P95 latency: {p95*1000:.2f}ms")
        for c, qps in sorted(qps_results.items()):
            print(f"  QPS (c={c:2d}):  {qps:.1f}")
        print(f"  Peak QPS:   {peak_qps:.1f}")
        print(f"{'='*60}")

        # Append to results file
        new_content = f"### Redis 8.0 VSET Run: {run_timestamp}\n"
        new_content += f"**Case:** Performance1536D50K, **K:** {args.k}\n\n"
        new_content += "| Engine | Load Time (s) | Peak QPS | P99 Latency (s) | Recall |\n"
        new_content += "| :--- | :---: | :---: | :---: | :---: |\n"
        new_content += f"| Redis 8.0 VSET (VADD/VSIM) | {insert_dur:.1f} | {peak_qps:.1f} | {p99:.4f} | {recall:.4f} |\n\n"

        old_content = ""
        if os.path.exists(RESULTS_MD):
            with open(RESULTS_MD, "r") as f:
                old_content = f.read()
        with open(RESULTS_MD, "w") as f:
            f.write(new_content + old_content)
        print(f"Results appended to {RESULTS_MD}")

    finally:
        print("Stopping Redis 8.0...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
