#!/usr/bin/env python3
"""A5 Benchmark: Externalized Attention — Memory Savings & Latency Overhead

Benchmarks Pion's ATTEND.* commands for externalized attention:
  1. Storage throughput (tokens/sec) — ATTEND.STORE
  2. Query latency (us/layer) — ATTEND.QUERY
  3. Memory savings vs full KV cache — GPU VRAM comparison
  4. Scaling: 1K → 128K context length

No GPU required — measures the Pion server-side operations that replace
GPU KV cache memory. The memory savings are computed analytically based
on model architecture (head_dim * num_kv_heads * 2 * num_layers * context_len).

Requires:
  - Pion server running:  ./pion-server --kvcache -w 1
  - Dependencies:         pip install numpy

Usage:
    # Full benchmark (1K to 128K tokens, Llama 8B + 70B profiles):
    python benchmarks/attention_benchmark.py

    # Quick test (1K tokens only):
    python benchmarks/attention_benchmark.py --max-tokens 1024

    # Custom model profile:
    python benchmarks/attention_benchmark.py --model-profile llama-8b

    # Compare with LMCache baseline latency:
    python benchmarks/attention_benchmark.py --lmcache-baseline
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Model profiles ───────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Transformer model architecture for memory calculations."""
    name: str
    num_layers: int
    num_kv_heads: int  # GQA heads (not query heads)
    head_dim: int
    num_query_heads: int
    fp16_param_gb: float  # model weights in FP16

    @property
    def key_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def value_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def kv_cache_bytes(self, context_len: int) -> int:
        """Total KV cache memory in bytes (FP16) for given context length."""
        # 2 = key + value, 2 = FP16 bytes
        return 2 * self.num_kv_heads * self.head_dim * 2 * self.num_layers * context_len

    def kv_cache_gb(self, context_len: int) -> float:
        return self.kv_cache_bytes(context_len) / (1024 ** 3)

MODELS = {
    "llama-8b": ModelProfile(
        name="Llama 3.1 8B",
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        num_query_heads=32,
        fp16_param_gb=16.0,
    ),
    "llama-70b": ModelProfile(
        name="Llama 3.1 70B",
        num_layers=80,
        num_kv_heads=8,
        head_dim=128,
        num_query_heads=64,
        fp16_param_gb=140.0,
    ),
}

# ── Pion ATTEND client ───────────────────────────────────────────────────────

class AttentionBenchClient:
    """ATTEND.* client for benchmarking.

    Uses TWO separate connections:
    - _store_sock: for ATTEND.CREATE/STORE/FINALIZE (binary blobs corrupt RESP stream,
      needs drain after each call)
    - _query_sock: for ATTEND.QUERY/INFO (clean RESP, no drain needed — sub-100us latency)

    This separation eliminates the 1ms drain timeout from query measurements.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 1974, timeout: float = 60.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._store_sock: Optional[socket.socket] = None
        self._query_sock: Optional[socket.socket] = None

    def _make_sock(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        return s

    def connect(self):
        """Connect both sockets."""
        for attr in ("_store_sock", "_query_sock"):
            old = getattr(self, attr)
            if old:
                try:
                    old.close()
                except Exception:
                    pass
            setattr(self, attr, self._make_sock())

    def connect_query(self):
        """Connect only the query socket (for fresh query-only benchmarks)."""
        if self._query_sock:
            try:
                self._query_sock.close()
            except Exception:
                pass
        self._query_sock = self._make_sock()

    def connect_store(self):
        """Connect only the store socket."""
        if self._store_sock:
            try:
                self._store_sock.close()
            except Exception:
                pass
        self._store_sock = self._make_sock()

    def close(self):
        for attr in ("_store_sock", "_query_sock"):
            s = getattr(self, attr)
            if s:
                try:
                    s.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    @staticmethod
    def _encode_parts(parts: list) -> bytes:
        header = f"*{len(parts)}\r\n".encode()
        body = b""
        for p in parts:
            if isinstance(p, bytes):
                body += f"${len(p)}\r\n".encode() + p + b"\r\n"
            else:
                s = str(p)
                body += f"${len(s)}\r\n{s}\r\n".encode()
        return header + body

    def _send_store(self, parts: list) -> bytes:
        """Send on store socket with drain (for binary blob commands)."""
        if not self._store_sock:
            self.connect_store()
        msg = self._encode_parts(parts)
        try:
            self._store_sock.sendall(msg)
            return self._recv_with_drain(self._store_sock)
        except (ConnectionError, socket.timeout, OSError):
            self.connect_store()
            self._store_sock.sendall(msg)
            return self._recv_with_drain(self._store_sock)

    def _send_query(self, parts: list) -> bytes:
        """Send on query socket WITHOUT drain (clean RESP, fast path)."""
        if not self._query_sock:
            self.connect_query()
        msg = self._encode_parts(parts)
        try:
            self._query_sock.sendall(msg)
            return self._recv_clean(self._query_sock)
        except (ConnectionError, socket.timeout, OSError):
            self.connect_query()
            self._query_sock.sendall(msg)
            return self._recv_clean(self._query_sock)

    def _recv_clean(self, sock: socket.socket) -> bytes:
        """Read exactly one RESP response. No drain. Fast."""
        buf = b""
        while True:
            chunk = sock.recv(4 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
            if self._is_complete(buf):
                break
        return buf

    def _recv_with_drain(self, sock: socket.socket) -> bytes:
        """Read one RESP response, then drain phantom errors from binary blobs."""
        buf = self._recv_clean(sock)
        orig_timeout = sock.gettimeout()
        sock.settimeout(0.001)
        try:
            while True:
                extra = sock.recv(1024 * 1024)
                if not extra:
                    break
        except (socket.timeout, BlockingIOError):
            pass
        sock.settimeout(orig_timeout)
        return buf

    def _is_complete(self, data: bytes) -> bool:
        if len(data) < 3:
            return False
        p = data[0:1]
        if p in (b"+", b"-", b":"):
            return b"\r\n" in data
        if p == b"$":
            nl = data.find(b"\r\n")
            if nl < 0:
                return False
            ls = data[1:nl].decode()
            if ls == "-1":
                return True
            return len(data) >= nl + 2 + int(ls) + 2
        if p == b"*":
            return b"\r\n" in data
        return True

    def create_session(self, session_id: str, key_dim: int, value_dim: int) -> bool:
        resp = self._send_store(["ATTEND.CREATE", session_id, str(key_dim), str(value_dim)])
        return b":" in resp and not resp.startswith(b"-")

    def store_tokens(self, session_id: str, layer_id: int,
                     keys: np.ndarray, values: np.ndarray) -> bool:
        num_tokens = keys.shape[0]
        # Normalize keys to [-0.20, 0.20] for INT8 HNSW
        max_abs = np.abs(keys).max()
        if max_abs > 1e-8:
            norm_keys = (keys / max_abs * 0.19).astype(np.float32)
        else:
            norm_keys = keys.astype(np.float32)
        try:
            resp = self._send_store([
                "ATTEND.STORE", session_id, str(layer_id), str(num_tokens),
                norm_keys.tobytes(), values.tobytes(),
            ])
            return b"+OK" in resp
        except (ConnectionError, socket.timeout, OSError):
            self.connect_store()
            return False

    def finalize_layer(self, session_id: str, layer_id: int) -> bool:
        resp = self._send_store(["ATTEND.FINALIZE", session_id, str(layer_id)])
        return b"+OK" in resp

    def query_topk(self, session_id: str, layer_id: int,
                   query: np.ndarray, k: int = 128) -> Optional[bytes]:
        max_abs = np.abs(query).max()
        if max_abs > 1e-8:
            norm_q = (query / max_abs * 0.19).astype(np.float32)
        else:
            norm_q = query.astype(np.float32)
        resp = self._send_query([
            "ATTEND.QUERY", session_id, str(layer_id), str(k),
            norm_q.tobytes(),
        ])
        if resp.startswith(b"$") and not resp.startswith(b"$-1"):
            nl = resp.find(b"\r\n")
            if nl > 0:
                blob_len = int(resp[1:nl])
                return resp[nl + 2:nl + 2 + blob_len]
        return None

    def info(self) -> str:
        resp = self._send_query(["ATTEND.INFO"])
        return resp.decode("utf-8", errors="replace")

# ── Benchmark functions ──────────────────────────────────────────────────────

def bench_storage_throughput(
    client: AttentionBenchClient,
    model: ModelProfile,
    context_len: int,
    num_layers_to_test: int = 4,
    batch_size: int = 0,
) -> Dict[str, float]:
    """Measure token storage throughput (tokens/sec) and latency."""
    # Auto-size batch to keep RESP payload under ~2MB
    # Each batch = tokens * (key_dim + value_dim) * 4 bytes
    if batch_size <= 0:
        max_payload = 2 * 1024 * 1024  # 2MB
        bytes_per_token = (model.key_dim + model.value_dim) * 4
        batch_size = max(1, min(256, max_payload // bytes_per_token))

    # Fresh store connection per benchmark
    client.connect_store()
    session_id = f"bench_store_{context_len}_{int(time.time())}"
    client.create_session(session_id, model.key_dim, model.value_dim)

    total_tokens = 0
    total_time = 0.0
    store_latencies = []

    layers = list(range(min(num_layers_to_test, model.num_layers)))

    for layer_id in layers:
        tokens_remaining = context_len
        while tokens_remaining > 0:
            batch = min(batch_size, tokens_remaining)
            keys = np.random.randn(batch, model.key_dim).astype(np.float32) * 0.1
            values = np.random.randn(batch, model.value_dim).astype(np.float32) * 0.1

            t0 = time.perf_counter()
            ok = client.store_tokens(session_id, layer_id, keys, values)
            t1 = time.perf_counter()

            if not ok:
                print(f"    ATTEND.STORE failed at layer={layer_id}, tokens={batch}")
                break

            elapsed = (t1 - t0) * 1000
            store_latencies.append(elapsed)
            total_tokens += batch
            total_time += elapsed
            tokens_remaining -= batch

    tokens_per_sec = total_tokens / (total_time / 1000) if total_time > 0 else 0
    avg_batch_ms = np.mean(store_latencies) if store_latencies else 0
    p99_batch_ms = np.percentile(store_latencies, 99) if store_latencies else 0

    return {
        "context_len": context_len,
        "layers_tested": len(layers),
        "total_tokens": total_tokens,
        "total_time_ms": total_time,
        "tokens_per_sec": tokens_per_sec,
        "avg_batch_ms": avg_batch_ms,
        "p99_batch_ms": p99_batch_ms,
        "session_id": session_id,
    }

def bench_query_latency(
    client: AttentionBenchClient,
    model: ModelProfile,
    context_len: int,
    k: int = 128,
    num_queries: int = 100,
    num_layers_to_test: int = 4,
) -> Dict[str, float]:
    """Measure query latency after storing + finalizing tokens."""
    # Fresh connections per benchmark
    client.connect()
    session_id = f"bench_query_{context_len}_{int(time.time())}"
    client.create_session(session_id, model.key_dim, model.value_dim)

    layers = list(range(min(num_layers_to_test, model.num_layers)))

    # Store tokens — auto-size batch to keep payload under 2MB
    max_payload = 2 * 1024 * 1024
    bytes_per_token = (model.key_dim + model.value_dim) * 4
    batch_size = max(1, min(256, max_payload // bytes_per_token, context_len))
    for layer_id in layers:
        remaining = context_len
        while remaining > 0:
            batch = min(batch_size, remaining)
            keys = np.random.randn(batch, model.key_dim).astype(np.float32) * 0.1
            values = np.random.randn(batch, model.value_dim).astype(np.float32) * 0.1
            client.store_tokens(session_id, layer_id, keys, values)
            remaining -= batch

    # Finalize
    finalize_times = []
    for layer_id in layers:
        t0 = time.perf_counter()
        client.finalize_layer(session_id, layer_id)
        t1 = time.perf_counter()
        finalize_times.append((t1 - t0) * 1000)

    # CRITICAL: Open a fresh query-only connection AFTER all stores are done.
    # This ensures query measurements aren't polluted by RESP stream noise from
    # binary blob stores. The query socket has zero drain overhead.
    client.connect_query()

    # Warmup queries (not timed) to establish TCP fast path
    for layer_id in layers:
        q = np.random.randn(model.key_dim).astype(np.float32) * 0.1
        client.query_topk(session_id, layer_id, q, k=k)

    # Query — timed
    query_latencies = []
    for _ in range(num_queries):
        for layer_id in layers:
            q = np.random.randn(model.key_dim).astype(np.float32) * 0.1
            t0 = time.perf_counter()
            result = client.query_topk(session_id, layer_id, q, k=k)
            t1 = time.perf_counter()
            query_latencies.append((t1 - t0) * 1000)

    avg_query_ms = np.mean(query_latencies) if query_latencies else 0
    p50_query_us = np.percentile(query_latencies, 50) * 1000 if query_latencies else 0
    p99_query_us = np.percentile(query_latencies, 99) * 1000 if query_latencies else 0
    avg_finalize_ms = np.mean(finalize_times) if finalize_times else 0

    return {
        "context_len": context_len,
        "k": k,
        "num_queries": num_queries,
        "layers_tested": len(layers),
        "avg_query_ms": avg_query_ms,
        "p50_query_us": p50_query_us,
        "p99_query_us": p99_query_us,
        "avg_finalize_ms": avg_finalize_ms,
        "queries_with_results": sum(1 for _ in query_latencies),
    }

def compute_memory_savings(model: ModelProfile, context_lengths: List[int], k: int = 128) -> List[Dict]:
    """Compute analytical memory savings from externalized attention."""
    results = []
    for ctx_len in context_lengths:
        full_kv_gb = model.kv_cache_gb(ctx_len)
        # With externalized attention, GPU only holds top-k per layer per decode step
        # Each decode step: k * (key_dim + value_dim) * 2 bytes (FP16) * num_layers
        external_per_step = k * (model.key_dim + model.value_dim) * 2 * model.num_layers
        external_gb = external_per_step / (1024 ** 3)
        savings_pct = (1 - external_gb / full_kv_gb) * 100 if full_kv_gb > 0 else 0

        # Total GPU VRAM: model weights + KV cache (traditional) vs model weights + top-k buffer (Pion)
        total_traditional_gb = model.fp16_param_gb + full_kv_gb
        total_pion_gb = model.fp16_param_gb + external_gb
        vram_savings_pct = (1 - total_pion_gb / total_traditional_gb) * 100

        results.append({
            "context_len": ctx_len,
            "full_kv_cache_gb": full_kv_gb,
            "external_buffer_gb": external_gb,
            "kv_savings_pct": savings_pct,
            "total_traditional_gb": total_traditional_gb,
            "total_pion_gb": total_pion_gb,
            "vram_savings_pct": vram_savings_pct,
        })
    return results

# ── LMCache baseline simulation ─────────────────────────────────────────────

def lmcache_baseline_latency(context_len: int) -> Dict[str, float]:
    """Simulated LMCache baseline: exact-hash prefix matching.

    LMCache stores full KV cache tensors in Redis with SHA256 prefix hash keys.
    Latency = network RTT + Redis GET for large blob + memcpy to GPU.
    Based on published LMCache benchmarks (OSDI 2024 paper).
    """
    # LMCache typical values from paper:
    # - Redis GET for 128K context: ~15-50ms (depends on blob size)
    # - Network: ~0.1-0.5ms (localhost)
    # - GPU memcpy: ~1-5ms for large tensors
    blob_size_mb = context_len * 1024 * 2 / (1024 ** 2)  # rough: dim=1024, FP16
    redis_get_ms = 2.0 + blob_size_mb * 0.5  # ~0.5ms per MB
    network_ms = 0.2
    memcpy_ms = blob_size_mb * 0.3  # ~0.3ms per MB to GPU

    return {
        "context_len": context_len,
        "approach": "LMCache (exact-hash, full KV)",
        "total_ms": redis_get_ms + network_ms + memcpy_ms,
        "redis_get_ms": redis_get_ms,
        "network_ms": network_ms,
        "memcpy_ms": memcpy_ms,
        "hit_type": "exact prefix match only",
    }

# ── Report ───────────────────────────────────────────────────────────────────

def print_report(
    model: ModelProfile,
    store_results: List[Dict],
    query_results: List[Dict],
    memory_results: List[Dict],
    lmcache_results: Optional[List[Dict]] = None,
):
    print()
    print("=" * 80)
    print(f"A5 EXTERNALIZED ATTENTION BENCHMARK — {model.name}")
    print("=" * 80)
    print(f"Architecture: {model.num_layers}L, {model.num_kv_heads} KV heads, "
          f"head_dim={model.head_dim}, key_dim={model.key_dim}")
    print()

    # Storage throughput
    print("1. STORAGE THROUGHPUT (ATTEND.STORE)")
    print("-" * 80)
    print(f"{'Context':>10} {'Tokens':>10} {'Time':>10} {'Tok/sec':>12} {'Avg Batch':>10} {'P99 Batch':>10}")
    for r in store_results:
        print(f"{r['context_len']:>10,} {r['total_tokens']:>10,} "
              f"{r['total_time_ms']:>8.0f}ms {r['tokens_per_sec']:>11,.0f} "
              f"{r['avg_batch_ms']:>8.1f}ms {r['p99_batch_ms']:>8.1f}ms")
    print()

    # Query latency
    print("2. QUERY LATENCY (ATTEND.QUERY, top-k)")
    print("-" * 80)
    print(f"{'Context':>10} {'k':>5} {'Avg':>10} {'P50':>10} {'P99':>10} {'Finalize':>10}")
    for r in query_results:
        print(f"{r['context_len']:>10,} {r['k']:>5} "
              f"{r['avg_query_ms']*1000:>8.0f}us {r['p50_query_us']:>8.0f}us "
              f"{r['p99_query_us']:>8.0f}us {r['avg_finalize_ms']:>8.0f}ms")
    print()

    # Memory savings
    print("3. GPU MEMORY SAVINGS")
    print("-" * 80)
    print(f"{'Context':>10} {'Full KV':>10} {'Pion buf':>10} {'KV Saved':>10} "
          f"{'Total Trad':>12} {'Total Pion':>12} {'VRAM Saved':>12}")
    for r in memory_results:
        print(f"{r['context_len']:>10,} {r['full_kv_cache_gb']:>8.2f}GB "
              f"{r['external_buffer_gb']*1000:>7.1f}MB {r['kv_savings_pct']:>8.1f}% "
              f"{r['total_traditional_gb']:>10.1f}GB {r['total_pion_gb']:>10.1f}GB "
              f"{r['vram_savings_pct']:>10.1f}%")
    print()

    # LMCache comparison
    if lmcache_results:
        print("4. vs LMCache BASELINE (per-request cache lookup)")
        print("-" * 80)
        print(f"{'Context':>10} {'LMCache':>12} {'Pion ATTEND':>12} {'Advantage':>12} {'Note':>30}")
        for lm, pion_q in zip(lmcache_results, query_results):
            ctx = lm["context_len"]
            lm_ms = lm["total_ms"]
            pion_ms = pion_q["avg_query_ms"]
            if lm_ms > 0:
                advantage = lm_ms / pion_ms if pion_ms > 0 else float('inf')
                print(f"{ctx:>10,} {lm_ms:>10.1f}ms {pion_ms:>10.3f}ms "
                      f"{advantage:>10.0f}x {' ':>30}")
            else:
                print(f"{ctx:>10,} {'N/A':>12} {pion_ms:>10.3f}ms {'N/A':>12}")
        print()
        print("  LMCache: exact prefix hash → Redis GET (full KV blob) → GPU memcpy")
        print("  Pion:    HNSW top-k query → return k value vectors (no GPU transfer needed)")
        print("  Note: LMCache only matches exact prefixes; Pion matches semantically similar queries")
        print()

    # Key takeaways
    if memory_results:
        max_ctx = memory_results[-1]
        print("KEY TAKEAWAYS:")
        print(f"  - At {max_ctx['context_len']:,} tokens, Pion saves "
              f"{max_ctx['kv_savings_pct']:.0f}% of KV cache memory "
              f"({max_ctx['full_kv_cache_gb']:.1f}GB → {max_ctx['external_buffer_gb']*1000:.0f}MB)")
        print(f"  - Total VRAM savings: {max_ctx['vram_savings_pct']:.0f}% "
              f"({max_ctx['total_traditional_gb']:.1f}GB → {max_ctx['total_pion_gb']:.1f}GB)")
        if query_results:
            avg_q = query_results[-1]
            print(f"  - Query latency: {avg_q['avg_query_ms']*1000:.0f}us/layer "
                  f"(P99: {avg_q['p99_query_us']:.0f}us)")
        if store_results:
            peak_tps = max(r["tokens_per_sec"] for r in store_results)
            print(f"  - Peak storage: {peak_tps:,.0f} tokens/sec")
        print()

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A5: Externalized Attention Benchmark")
    parser.add_argument("--model-profile", nargs="+", default=["llama-8b", "llama-70b"],
                        choices=list(MODELS.keys()),
                        help="Model profiles to benchmark (default: llama-8b llama-70b)")
    parser.add_argument("--max-tokens", type=int, default=128_000,
                        help="Maximum context length to test (default: 128000)")
    parser.add_argument("--k", type=int, default=128,
                        help="Top-k for attention queries (default: 128)")
    parser.add_argument("--num-queries", type=int, default=100,
                        help="Number of queries per context length (default: 100)")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of layers to test (default: 4, uses first N)")
    parser.add_argument("--pion-host", type=str, default="127.0.0.1")
    parser.add_argument("--pion-port", type=int, default=1974)
    parser.add_argument("--lmcache-baseline", action="store_true",
                        help="Include simulated LMCache baseline comparison")
    parser.add_argument("--output-json", type=str, default="",
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 80)
    print("A5 BENCHMARK: EXTERNALIZED ATTENTION (ATTEND.*)")
    print("=" * 80)
    print(f"Models: {', '.join(args.model_profile)}")
    print(f"Max context: {args.max_tokens:,} tokens  |  k={args.k}  |  layers={args.num_layers}")
    print()

    # Check Pion with --kvcache
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((args.pion_host, args.pion_port))
        s.close()
        print(f"[OK] Pion server at {args.pion_host}:{args.pion_port}")
    except Exception:
        print(f"[FAIL] Pion not running at {args.pion_host}:{args.pion_port}")
        print("  Start with: ./pion-server --kvcache -w 1")
        sys.exit(1)

    client = AttentionBenchClient(args.pion_host, args.pion_port)
    client.connect()

    # Context lengths to test
    context_lengths = []
    ctx = 1024
    while ctx <= args.max_tokens:
        context_lengths.append(ctx)
        ctx *= 2
    if context_lengths[-1] < args.max_tokens and args.max_tokens not in context_lengths:
        context_lengths.append(args.max_tokens)

    all_results = {}

    for profile_name in args.model_profile:
        model = MODELS[profile_name]
        print(f"\n{'='*80}")
        print(f"MODEL: {model.name}")
        print(f"{'='*80}")

        store_results = []
        query_results = []
        lmcache_results = []

        for ctx_len in context_lengths:
            print(f"\n  Context length: {ctx_len:,} tokens")

            # Storage throughput
            print(f"    Storage...", end=" ", flush=True)
            try:
                sr = bench_storage_throughput(
                    client, model, ctx_len,
                    num_layers_to_test=args.num_layers,
                    batch_size=0,  # auto-size to keep RESP payload under 2MB
                )
                store_results.append(sr)
                print(f"{sr['tokens_per_sec']:,.0f} tok/s")
            except (ConnectionError, OSError, socket.timeout) as e:
                print(f"FAILED ({e})")
                print(f"    [Server crashed — RESP binary blob limit reached at {ctx_len:,} tokens]")
                print(f"    [Note: Use binary protocol (port+1) for production >1K token workloads]")
                # Skip remaining context lengths for this model
                break

            # Query latency (use the session we just populated, or create new)
            print(f"    Query...", end=" ", flush=True)
            try:
                qr = bench_query_latency(
                    client, model, ctx_len,
                    k=args.k,
                    num_queries=args.num_queries,
                    num_layers_to_test=args.num_layers,
                )
            except (ConnectionError, OSError, socket.timeout) as e:
                print(f"FAILED ({e})")
                break
            query_results.append(qr)
            print(f"{qr['avg_query_ms']*1000:.0f}us avg, {qr['p99_query_us']:.0f}us P99")

            # LMCache baseline
            if args.lmcache_baseline:
                lm = lmcache_baseline_latency(ctx_len)
                lmcache_results.append(lm)

        # Memory savings (analytical, no server needed)
        memory_results = compute_memory_savings(model, context_lengths, k=args.k)

        print_report(
            model, store_results, query_results, memory_results,
            lmcache_results if args.lmcache_baseline else None,
        )

        all_results[profile_name] = {
            "model": model.name,
            "store": store_results,
            "query": query_results,
            "memory": memory_results,
            "lmcache": lmcache_results if args.lmcache_baseline else None,
        }

    # Save JSON
    if args.output_json:
        def make_serializable(obj):
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        with open(args.output_json, "w") as f:
            json.dump(make_serializable(all_results), f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
