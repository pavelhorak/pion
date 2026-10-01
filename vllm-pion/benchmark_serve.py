#!/usr/bin/env python3
"""Pion Serve — Phase 2: End-to-End Latency Benchmark.

Measures TTFT, cache hit rate, and throughput using synthetic coding workloads
from Phase 1. Runs in dry-run mode (no GPU/model required) or with a real
HuggingFace model.

Usage:
    # Dry-run (cache layer only, no model):
    python benchmark_serve.py

    # With model:
    python benchmark_serve.py --model meta-llama/Llama-3.2-1B-Instruct

    # With Pion server:
    python benchmark_serve.py --pion-host 127.0.0.1 --pion-port 1974
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_pion.serve_engine import PionServeEngine, ServeConfig
from vllm_pion.semantic_cache import CacheConfig


# ── Workload generation (subset of Phase 1) ─────────────────────────────────

MULTI_TURN_PAIRS = [
    # (cache_prompt, query_prompt) — same code, different question
    (
        "System: You are a coding assistant.\n\nContext:\n```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\n\nUser: Explain what this code does.",
        "System: You are a coding assistant.\n\nContext:\n```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```\n\nUser: How would I add error handling to this?",
    ),
    (
        "System: You are a coding assistant.\n\nContext:\n```python\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = {}\n        self.order = []\n    def get(self, key):\n        if key in self.cache:\n            self.order.remove(key)\n            self.order.append(key)\n            return self.cache[key]\n        return -1\n    def put(self, key, value):\n        if key in self.cache:\n            self.order.remove(key)\n        elif len(self.cache) >= self.capacity:\n            oldest = self.order.pop(0)\n            del self.cache[oldest]\n        self.cache[key] = value\n        self.order.append(key)\n```\n\nUser: What is the time complexity of each operation?",
        "System: You are a coding assistant.\n\nContext:\n```python\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = {}\n        self.order = []\n    def get(self, key):\n        if key in self.cache:\n            self.order.remove(key)\n            self.order.append(key)\n            return self.cache[key]\n        return -1\n    def put(self, key, value):\n        if key in self.cache:\n            self.order.remove(key)\n        elif len(self.cache) >= self.capacity:\n            oldest = self.order.pop(0)\n            del self.cache[oldest]\n        self.cache[key] = value\n        self.order.append(key)\n```\n\nUser: Can you optimize this using OrderedDict?",
    ),
    (
        "System: You are a coding assistant.\n\nContext:\n```mojo\nstruct SlabAllocator[T: AnyType]:\n    var base_ptr: UnsafePointer[T]\n    var capacity: Int\n    var next_free: Int\n    var free_list: UnsafePointer[Int]\n    var free_count: Int\n\n    fn allocate(mut self) -> UnsafePointer[T]:\n        if self.free_count > 0:\n            self.free_count -= 1\n            var idx = self.free_list[self.free_count]\n            return self.base_ptr + idx\n        if self.next_free >= self.capacity:\n            return UnsafePointer[T]()\n        var ptr = self.base_ptr + self.next_free\n        self.next_free += 1\n        return ptr\n```\n\nUser: Explain the allocation strategy.",
        "System: You are a coding assistant.\n\nContext:\n```mojo\nstruct SlabAllocator[T: AnyType]:\n    var base_ptr: UnsafePointer[T]\n    var capacity: Int\n    var next_free: Int\n    var free_list: UnsafePointer[Int]\n    var free_count: Int\n\n    fn allocate(mut self) -> UnsafePointer[T]:\n        if self.free_count > 0:\n            self.free_count -= 1\n            var idx = self.free_list[self.free_count]\n            return self.base_ptr + idx\n        if self.next_free >= self.capacity:\n            return UnsafePointer[T]()\n        var ptr = self.base_ptr + self.next_free\n        self.next_free += 1\n        return ptr\n```\n\nUser: What happens when the allocator is full?",
    ),
]

INLINE_PAIRS = [
    # (original, minor_edit) — same code with small changes
    (
        "Complete the following code:\n```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n```",
        "Complete the following code:\n```python\ndef merge_sort(data):\n    if len(data) <= 1:\n        return data\n    mid = len(data) // 2\n    left = merge_sort(data[:mid])\n    right = merge_sort(data[mid:])\n    return merge(left, right)\n```",
    ),
    (
        "Complete the following code:\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```",
        "Complete the following code:\n```python\ndef fibonacci(n):\n    # Calculate nth fibonacci number\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```",
    ),
]

CROSS_USER_PAIRS = [
    (
        "I'm working on this codebase. Here's the hash map:\n```mojo\nfn probe(self, hash: UInt64) -> Int:\n    var h1 = hash >> 7\n    var h2 = hash & 0x7F\n    var group = h1 % self.capacity\n    # SIMD probe\n```\nHow does the probing work?",
        "Looking at this Swiss table implementation:\n```mojo\nfn probe(self, hash: UInt64) -> Int:\n    var h1 = hash >> 7\n    var h2 = hash & 0x7F\n    var group = h1 % self.capacity\n    # SIMD probe\n```\nExplain the probe mechanism.",
    ),
]

UNRELATED = [
    "What is the capital of France?",
    "Write a recipe for pancakes.",
    "Explain quantum entanglement.",
    "How do I train a neural network?",
    "What are the causes of climate change?",
]


def build_benchmark_workload() -> list[dict]:
    """Build the benchmark workload.

    Returns list of dicts: {phase, prompt, expected_hit, cache_prompt}
    """
    workload = []

    # Phase 1: Populate cache with initial prompts
    for cache_prompt, query_prompt in MULTI_TURN_PAIRS:
        workload.append({"phase": "populate", "prompt": cache_prompt, "expected_hit": False, "cache_prompt": ""})
    for cache_prompt, query_prompt in INLINE_PAIRS:
        workload.append({"phase": "populate", "prompt": cache_prompt, "expected_hit": False, "cache_prompt": ""})
    for cache_prompt, query_prompt in CROSS_USER_PAIRS:
        workload.append({"phase": "populate", "prompt": cache_prompt, "expected_hit": False, "cache_prompt": ""})

    # Phase 2: Query with similar prompts (should hit)
    for cache_prompt, query_prompt in MULTI_TURN_PAIRS:
        workload.append({"phase": "query", "prompt": query_prompt, "expected_hit": True, "cache_prompt": cache_prompt})
    for cache_prompt, query_prompt in INLINE_PAIRS:
        workload.append({"phase": "query", "prompt": query_prompt, "expected_hit": True, "cache_prompt": cache_prompt})
    for cache_prompt, query_prompt in CROSS_USER_PAIRS:
        workload.append({"phase": "query", "prompt": query_prompt, "expected_hit": True, "cache_prompt": cache_prompt})

    # Phase 3: Unrelated queries (should miss)
    for prompt in UNRELATED:
        workload.append({"phase": "negative", "prompt": prompt, "expected_hit": False, "cache_prompt": ""})

    return workload


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(engine: PionServeEngine, workload: list[dict]) -> list[dict]:
    """Run the benchmark and collect results."""
    results = []

    for i, item in enumerate(workload):
        result = engine.generate(
            prompt=item["prompt"],
            max_tokens=1,  # minimal generation for benchmarking cache layer
            temperature=0.0,
        )

        results.append({
            "idx": i,
            "phase": item["phase"],
            "expected_hit": item["expected_hit"],
            "actual_hit": result.cache_hit,
            "similarity": result.cache_similarity,
            "ttft_ms": result.ttft_ms,
            "total_ms": result.total_ms,
            "prompt_tokens": result.prompt_tokens,
            "correct": result.cache_hit == item["expected_hit"],
        })

    return results


def print_report(results: list[dict], engine: PionServeEngine):
    """Print benchmark report."""
    stats = engine.stats

    print()
    print("=" * 70)
    print("PION SERVE — PHASE 2 BENCHMARK RESULTS")
    print("=" * 70)

    # Overall stats
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    populate = [r for r in results if r["phase"] == "populate"]
    query = [r for r in results if r["phase"] == "query"]
    negative = [r for r in results if r["phase"] == "negative"]

    print(f"\nWorkload: {total} requests ({len(populate)} populate, {len(query)} query, {len(negative)} negative)")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print()

    # Cache performance
    query_hits = sum(1 for r in query if r["actual_hit"])
    query_misses = sum(1 for r in query if not r["actual_hit"])
    neg_hits = sum(1 for r in negative if r["actual_hit"])

    print("Cache Performance:")
    print(f"  Query hit rate:     {query_hits}/{len(query)} ({query_hits/len(query)*100:.0f}%)" if query else "  No queries")
    print(f"  False positives:    {neg_hits}/{len(negative)} ({neg_hits/len(negative)*100:.0f}%)" if negative else "  No negatives")
    print(f"  Overall hit rate:   {stats.cache_hit_rate*100:.1f}%")
    print()

    # Latency
    populate_ttft = [r["ttft_ms"] for r in populate]
    query_ttft = [r["ttft_ms"] for r in query]
    hit_ttft = [r["ttft_ms"] for r in query if r["actual_hit"]]
    miss_ttft = [r["ttft_ms"] for r in query if not r["actual_hit"]]

    print("Latency (TTFT):")
    if populate_ttft:
        print(f"  Populate (cold):    p50={np.median(populate_ttft):.2f}ms  p99={np.percentile(populate_ttft, 99):.2f}ms")
    if hit_ttft:
        print(f"  Cache hit:          p50={np.median(hit_ttft):.2f}ms  p99={np.percentile(hit_ttft, 99):.2f}ms")
    if miss_ttft:
        print(f"  Cache miss:         p50={np.median(miss_ttft):.2f}ms  p99={np.percentile(miss_ttft, 99):.2f}ms")
    print(f"  Overall avg:        {stats.avg_ttft_ms:.2f}ms")
    print()

    # Detailed results
    print(f"{'#':>3} {'Phase':>10} {'Expected':>9} {'Actual':>7} {'Sim':>6} {'TTFT':>8} {'OK':>4}")
    print("-" * 55)
    for r in results:
        phase = r["phase"][:8]
        expected = "HIT" if r["expected_hit"] else "MISS"
        actual = "HIT" if r["actual_hit"] else "MISS"
        sim = f"{r['similarity']:.3f}" if r["actual_hit"] else "-"
        ok = "Y" if r["correct"] else "N"
        print(f"{r['idx']:>3} {phase:>10} {expected:>9} {actual:>7} {sim:>6} {r['ttft_ms']:>6.2f}ms {ok:>4}")

    print()

    # Cache manager stats
    cm_stats = engine._cache_manager.stats
    print("Cache Manager Stats:")
    print(f"  Lookups:            {cm_stats.lookups}")
    print(f"  Hits:               {cm_stats.hits}")
    print(f"  Misses:             {cm_stats.misses}")
    print(f"  Stores:             {cm_stats.stores}")
    print(f"  Pion errors:        {cm_stats.pion_errors}")
    print(f"  Local fallbacks:    {cm_stats.local_fallback_hits}")
    print(f"  Avg lookup:         {cm_stats.avg_lookup_ms:.2f}ms")
    print()

    # Validation gate
    print("VALIDATION GATE:")
    hit_pass = query_hits / len(query) > 0.50 if query else False
    fp_pass = neg_hits / len(negative) < 0.10 if negative else True
    print(f"  Hit rate >50%:      {query_hits}/{len(query)} — {'PASS' if hit_pass else 'FAIL'}")
    print(f"  FP rate <10%:       {neg_hits}/{len(negative)} — {'PASS' if fp_pass else 'FAIL'}")
    if hit_pass and fp_pass:
        print(f"\n  >>> PHASE 2 VALIDATED — cache integration works <<<")
    else:
        print(f"\n  >>> NEEDS INVESTIGATION <<<")


def main():
    parser = argparse.ArgumentParser(description="Pion Serve — Phase 2 Benchmark")
    parser.add_argument("--model", type=str, default="",
                        help="HuggingFace model ID (empty = dry-run mode)")
    parser.add_argument("--pion-host", type=str, default="127.0.0.1")
    parser.add_argument("--pion-port", type=int, default=1974)
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Cosine similarity threshold for cache hits")
    parser.add_argument("--embed-provider", type=str, default="ngram",
                        help="Embedding provider: ngram, openai, ollama, auto")
    args = parser.parse_args()

    config = ServeConfig(
        model_id=args.model,
        cache=CacheConfig(
            pion_host=args.pion_host,
            pion_port=args.pion_port,
            cosine_threshold=args.threshold,
            embed_provider=args.embed_provider,
            fallback_to_local=True,
        ),
    )

    print("=" * 70)
    print("Pion Serve — Phase 2: End-to-End Benchmark")
    print("=" * 70)
    mode = f"model={args.model}" if args.model else "dry-run (cache layer only)"
    print(f"Mode: {mode}")
    print(f"Embedding: {args.embed_provider}")
    print(f"Threshold: {args.threshold}")
    print()

    workload = build_benchmark_workload()
    print(f"Workload: {len(workload)} requests")

    with PionServeEngine(config) as engine:
        print("\n[1/2] Running benchmark...")
        results = run_benchmark(engine, workload)

        print("\n[2/2] Results:")
        print_report(results, engine)

    return 0


if __name__ == "__main__":
    sys.exit(main())
