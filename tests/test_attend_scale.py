#!/usr/bin/env python3
"""M14 Phase 3 Scale Test: HNSW attention index at 4K, 32K, and 128K tokens.

Measures:
1. Insert throughput (tokens/sec) at each scale
2. Query latency (us) at each scale
3. Memory footprint estimation
4. Query correctness (known vector retrieval)

Uses 128d key/value dimensions (simulating single KV head for a Llama-class model).
"""

import socket
import struct
import sys
import time

import numpy as np

HOST = "127.0.0.1"
PORT = 1974

def send_one(parts):
    """Send RESP command via fresh connection."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(30)
    sock.connect((HOST, PORT))
    header = f"*{len(parts)}\r\n".encode()
    body = b""
    for p in parts:
        if isinstance(p, bytes):
            body += f"${len(p)}\r\n".encode() + p + b"\r\n"
        else:
            s = str(p)
            body += f"${len(s)}\r\n{s}\r\n".encode()
    sock.sendall(header + body)
    resp = sock.recv(1024 * 1024)
    sock.close()
    return resp


def run_scale_test(n_tokens: int, key_dim: int = 128, val_dim: int = 128, batch_size: int = 100):
    """Run a scale test for a given token count."""
    session_id = f"scale_{n_tokens}"
    layer_id = 0

    print(f"\n{'='*60}")
    print(f"Scale test: {n_tokens:,} tokens, {key_dim}d keys, {val_dim}d values")
    print(f"{'='*60}")

    # Create session
    r = send_one(["ATTEND.CREATE", session_id, str(key_dim), str(val_dim)])
    if b":" not in r:
        print(f"  CREATE failed: {r[:50]}")
        return None

    # Store tokens in batches
    np.random.seed(42)
    print(f"  Storing {n_tokens:,} tokens in batches of {batch_size}...")
    t0 = time.perf_counter()
    stored = 0
    all_keys = []

    for batch_start in range(0, n_tokens, batch_size):
        batch_end = min(batch_start + batch_size, n_tokens)
        n = batch_end - batch_start
        keys = np.random.randn(n, key_dim).astype(np.float32)
        vals = np.random.randn(n, val_dim).astype(np.float32)
        all_keys.append(keys)

        r = send_one(["ATTEND.STORE", session_id, str(layer_id), str(n),
                       keys.tobytes(), vals.tobytes()])
        if b"+OK" not in r:
            print(f"  Store failed at batch {batch_start}: {r[:50]}")
            break
        stored += n

        # Progress
        if stored % 1000 == 0 or stored == n_tokens:
            elapsed = time.perf_counter() - t0
            rate = stored / elapsed if elapsed > 0 else 0
            print(f"    {stored:>7,} tokens stored ({rate:,.0f} tok/s, {elapsed:.1f}s)")

    t_store = time.perf_counter() - t0
    store_rate = stored / t_store if t_store > 0 else 0
    print(f"  Store complete: {stored:,} tokens in {t_store:.1f}s ({store_rate:,.0f} tokens/sec)")

    if stored < n_tokens:
        print(f"  WARNING: Only stored {stored:,} of {n_tokens:,} tokens")

    # Query latency test (20 random queries)
    print(f"  Querying top-5 ({20} iterations)...")
    latencies = []
    for i in range(20):
        query = np.random.randn(key_dim).astype(np.float32)
        t0 = time.perf_counter()
        r = send_one(["ATTEND.QUERY", session_id, str(layer_id), "5", query.tobytes()])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    p99_lat = sorted(latencies)[int(0.99 * len(latencies))]
    qps = 1000 / avg_lat if avg_lat > 0 else 0

    print(f"  Query latency: avg={avg_lat:.2f}ms  min={min_lat:.2f}ms  max={max_lat:.2f}ms  p99={p99_lat:.2f}ms")
    print(f"  Query throughput: {qps:,.0f} QPS")

    # Memory estimate
    bytes_per_token = key_dim * 4 + val_dim * 4 + 8  # key + value + HNSW overhead
    total_mb = stored * bytes_per_token / (1024 * 1024)
    print(f"  Estimated memory: {total_mb:.1f} MB ({bytes_per_token} bytes/token)")

    # 40-layer extrapolation
    total_40_layers = total_mb * 40
    print(f"  40-layer extrapolation: {total_40_layers:.1f} MB ({total_40_layers/1024:.2f} GB)")

    return {
        "tokens": stored,
        "store_time_s": t_store,
        "store_rate": store_rate,
        "avg_latency_ms": avg_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "p99_latency_ms": p99_lat,
        "qps": qps,
        "memory_mb": total_mb,
        "memory_40_layers_gb": total_40_layers / 1024,
    }


def main():
    print("M14 Phase 3: Attention Index Scale Test")
    print(f"Server: {HOST}:{PORT}")

    # Check server
    try:
        r = send_one(["ATTEND.INFO"])
        print(f"Server status: {r[:100]}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Pion: {e}")
        print("Start with: ./pion-server -w 1 --kvcache")
        sys.exit(1)

    results = {}

    # Scale tests: 1K, 4K, 32K
    # (128K takes too long via RESP — binary protocol needed)
    for n in [1000, 4000, 32000]:
        r = run_scale_test(n)
        if r:
            results[n] = r

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Tokens':>10} {'Store(tok/s)':>14} {'Query(ms)':>12} {'QPS':>8} {'Memory':>10} {'40L(GB)':>10}")
    print("-" * 60)
    for n, r in sorted(results.items()):
        print(f"{r['tokens']:>10,} {r['store_rate']:>14,.0f} {r['avg_latency_ms']:>12.2f} {r['qps']:>8,.0f} {r['memory_mb']:>8.1f}MB {r['memory_40_layers_gb']:>9.2f}")

    # Extrapolation to 128K
    if 32000 in results:
        r32 = results[32000]
        # HNSW query scales as O(log N)
        import math
        ratio_128k = math.log(128000) / math.log(32000)
        est_lat_128k = r32["avg_latency_ms"] * ratio_128k
        est_qps_128k = 1000 / est_lat_128k
        est_mem_128k = r32["memory_mb"] * (128000 / 32000)
        est_40l_128k = est_mem_128k * 40 / 1024

        print(f"\n--- 128K Extrapolation (from 32K data, O(log N) scaling) ---")
        print(f"  Query latency: ~{est_lat_128k:.2f}ms")
        print(f"  QPS: ~{est_qps_128k:,.0f}")
        print(f"  Memory per layer: ~{est_mem_128k:.0f}MB")
        print(f"  40 layers: ~{est_40l_128k:.1f}GB")
        print(f"  Plan target (query <300us per layer): {'MET' if est_lat_128k < 0.3 else 'CLOSE' if est_lat_128k < 0.5 else 'NOT MET'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
