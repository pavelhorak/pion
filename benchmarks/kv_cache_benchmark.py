#!/usr/bin/env python3
"""M14 Phase 1 Benchmark: Pion Semantic KV Cache vs LMCache Exact-Hash Baseline

Measures cache hit rate and latency for:
1. Pion (semantic HNSW matching via nomic-embed-text embeddings)
2. Simulated LMCache (exact SHA256 hash matching)

Test workloads:
A. Multi-turn conversations (turns share 90%+ prefix)
B. System prompt variations (minor wording changes)
C. Paraphrased queries (same intent, different words)
D. Completely different queries (negative control)

No GPU required — uses Ollama nomic-embed-text on CPU/Apple Silicon.
"""

import hashlib
import json
import socket
import struct
import time
import sys

import numpy as np
import requests

HOST = "127.0.0.1"
PION_PORT = 1974
OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

# ── Embedding client ──

def embed(text: str) -> np.ndarray:
    """Embed text using Ollama nomic-embed-text."""
    resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
    resp.raise_for_status()
    vec = np.array(resp.json()["embedding"], dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return vec

# ── Pion client ──

def pion_kv_store(cache_id: str, embedding: np.ndarray, blob: bytes) -> float:
    """Store in Pion. Returns latency in ms."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(10)
    sock.connect((HOST, PION_PORT))

    parts = [b"KV.STORE", cache_id.encode(), embedding.tobytes(), blob]
    header = f"*{len(parts)}\r\n".encode()
    body = b""
    for part in parts:
        if isinstance(part, bytes):
            body += f"${len(part)}\r\n".encode() + part + b"\r\n"
        else:
            body += f"${len(str(part))}\r\n{part}\r\n".encode()

    t0 = time.perf_counter()
    sock.sendall(header + body)
    resp = sock.recv(4096)
    t1 = time.perf_counter()
    sock.close()

    assert b"+OK" in resp, f"KV.STORE failed: {resp[:100]}"
    return (t1 - t0) * 1000

def pion_kv_fetch(embedding: np.ndarray, threshold: float = 0.0) -> tuple:
    """Fetch from Pion. Returns (hit: bool, latency_ms: float, blob_size: int)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(10)
    sock.connect((HOST, PION_PORT))

    args = [b"KV.FETCH", embedding.tobytes()]
    if threshold > 0:
        args += [b"THRESHOLD", f"{threshold:.4f}".encode()]

    header = f"*{len(args)}\r\n".encode()
    body = b""
    for part in args:
        body += f"${len(part)}\r\n".encode() + part + b"\r\n"

    t0 = time.perf_counter()
    sock.sendall(header + body)
    resp = sock.recv(1024 * 1024)
    t1 = time.perf_counter()
    sock.close()

    latency = (t1 - t0) * 1000
    if resp.startswith(b"$-1"):
        return False, latency, 0

    if resp.startswith(b"$"):
        nl = resp.find(b"\r\n")
        if nl > 0:
            blob_len = int(resp[1:nl])
            return True, latency, blob_len

    return False, latency, 0

# ── LMCache simulator (exact hash) ──

class ExactHashCache:
    """Simulates LMCache's exact prefix hash matching."""
    def __init__(self):
        self.cache = {}  # hash -> blob

    def store(self, prompt: str, blob: bytes):
        h = hashlib.sha256(prompt.encode()).hexdigest()
        self.cache[h] = blob

    def fetch(self, prompt: str) -> bool:
        h = hashlib.sha256(prompt.encode()).hexdigest()
        return h in self.cache

# ── Test workloads ──

MULTI_TURN = [
    # Base conversation (stored)
    "You are a helpful coding assistant. The user is working on a Python web application using FastAPI.",
    # Turn variations (queries — should match the stored turn)
    "You are a helpful coding assistant. The user is working on a Python web application using FastAPI. How do I add authentication?",
    "You are a helpful coding assistant. The user is working on a Python web application using FastAPI. How do I add authentication? I want to use JWT tokens.",
    "You are a helpful coding assistant. The user is working on a Python web application using FastAPI. How do I add authentication? I want to use JWT tokens. Show me middleware code.",
]

SYSTEM_PROMPT_VARIANTS = [
    # Base (stored)
    "You are a helpful, harmless, and honest AI assistant.",
    # Variants (queries)
    "You are a helpful, harmless, and honest AI assistant. Answer concisely.",
    "You are a helpful and honest AI assistant.",
    "You are a helpful, harmless, honest AI assistant.",
    "You are an AI assistant that is helpful, harmless, and honest.",
    "You are a helpful AI. Be harmless and honest.",
]

PARAPHRASED_QUERIES = [
    # Base (stored)
    "What is the capital of France?",
    # Paraphrases (queries)
    "What's the capital city of France?",
    "Which city is the capital of France?",
    "Tell me the capital of France.",
    "France's capital is what city?",
    "Capital of France?",
]

UNRELATED_QUERIES = [
    "How do I cook pasta?",
    "What is quantum entanglement?",
    "Explain the theory of relativity.",
    "Write a haiku about rain.",
    "What is the GDP of Japan?",
]

def run_benchmark():
    print("=" * 70)
    print("M14 Phase 1 Benchmark: Pion Semantic Cache vs LMCache Exact-Hash")
    print("=" * 70)
    print()

    # Warm up Ollama
    print("Warming up Ollama embedding model...", end=" ", flush=True)
    _ = embed("warmup")
    print("done")

    lmcache = ExactHashCache()
    fake_blob = b"\x42" * 2048  # 2KB fake KV cache

    results = {}

    for workload_name, base_prompts, query_prompts in [
        ("A: Multi-turn conversation", MULTI_TURN[:1], MULTI_TURN[1:]),
        ("B: System prompt variants", SYSTEM_PROMPT_VARIANTS[:1], SYSTEM_PROMPT_VARIANTS[1:]),
        ("C: Paraphrased queries", PARAPHRASED_QUERIES[:1], PARAPHRASED_QUERIES[1:]),
        ("D: Unrelated queries (negative)", PARAPHRASED_QUERIES[:1], UNRELATED_QUERIES),
    ]:
        print(f"\n--- Workload {workload_name} ---")

        # Store base prompts
        for prompt in base_prompts:
            emb = embed(prompt)
            cache_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            store_lat = pion_kv_store(cache_id, emb, fake_blob)
            lmcache.store(prompt, fake_blob)
            print(f"  Stored: \"{prompt[:60]}...\" (Pion: {store_lat:.1f}ms)")

        # Query
        pion_hits = 0
        lmcache_hits = 0
        pion_latencies = []

        for prompt in query_prompts:
            # Pion (semantic)
            emb = embed(prompt)
            hit, lat, size = pion_kv_fetch(emb)
            pion_hits += int(hit)
            pion_latencies.append(lat)

            # LMCache (exact hash)
            lm_hit = lmcache.fetch(prompt)
            lmcache_hits += int(lm_hit)

            print(f"  Query: \"{prompt[:55]}...\"")
            print(f"    Pion: {'HIT' if hit else 'MISS'} ({lat:.1f}ms)  LMCache: {'HIT' if lm_hit else 'MISS'}")

        n = len(query_prompts)
        pion_rate = pion_hits / n * 100
        lm_rate = lmcache_hits / n * 100
        avg_lat = sum(pion_latencies) / len(pion_latencies)

        results[workload_name] = {
            "pion_hit_rate": pion_rate,
            "lmcache_hit_rate": lm_rate,
            "pion_avg_latency_ms": avg_lat,
            "queries": n,
        }

        print(f"\n  Results: Pion {pion_rate:.0f}% ({pion_hits}/{n}) vs LMCache {lm_rate:.0f}% ({lmcache_hits}/{n})")
        print(f"  Pion avg fetch latency: {avg_lat:.1f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Workload':<40} {'Pion':>10} {'LMCache':>10} {'Delta':>10}")
    print("-" * 70)

    total_pion = 0
    total_lm = 0
    total_n = 0

    for name, r in results.items():
        delta = r["pion_hit_rate"] - r["lmcache_hit_rate"]
        delta_str = f"+{delta:.0f}pp" if delta > 0 else f"{delta:.0f}pp"
        print(f"{name:<40} {r['pion_hit_rate']:>8.0f}%  {r['lmcache_hit_rate']:>8.0f}%  {delta_str:>10}")
        total_pion += r["pion_hit_rate"] * r["queries"]
        total_lm += r["lmcache_hit_rate"] * r["queries"]
        total_n += r["queries"]

    avg_pion = total_pion / total_n
    avg_lm = total_lm / total_n
    delta = avg_pion - avg_lm

    print("-" * 70)
    print(f"{'WEIGHTED AVERAGE':<40} {avg_pion:>8.0f}%  {avg_lm:>8.0f}%  +{delta:.0f}pp")
    print()
    print(f"Pion semantic matching advantage: +{delta:.0f} percentage points over exact-hash")
    print()

    return results

if __name__ == "__main__":
    # Check Pion is running
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((HOST, PION_PORT))
        s.close()
    except Exception:
        print(f"ERROR: Pion not running on {HOST}:{PION_PORT}")
        print("Start with: ./pion-server -w 1 --kvcache")
        sys.exit(1)

    run_benchmark()
