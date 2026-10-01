#!/usr/bin/env python3
"""
Demo: Pion as AI Memory Layer — Semantic Cache + AI.COMPLETE in front of Ollama.

Shows Pion's unique advantage over plain Ollama/LM Studio:
  1. Semantic cache: identical/similar queries answered instantly (<1ms vs 500ms+)
  2. Single command: AI.COMPLETE handles cache check + Ollama call + cache store
  3. Redis-protocol: any Redis client in any language can use it

Architecture:
    Client → AI.COMPLETE → Pion semantic cache (HNSW, sub-ms)
                               ↓ cache miss only
                           Ollama /api/generate

Pion handles the full flow: embed query → check HNSW → if hit return cached;
else call Ollama, embed response, store, return.

Usage:
    # 1. Start Ollama
    ollama pull nomic-embed-text
    ollama pull llama3.2:1b
    ollama serve

    # 2. Start Pion (auto-detects Ollama, enables semantic cache automatically)
    ./pion-server -w 1

    # 3. Run demo
    python3 examples/pion_vs_ollama_demo.py

Requirements:
    pip install redis requests
"""
from __future__ import annotations

import sys
import time

import redis
import requests

PION_PORT    = 1974
OLLAMA_URL   = "http://127.0.0.1:11434"
CACHE_THRESHOLD = 0.92

r = redis.Redis(port=PION_PORT, decode_responses=True)


def ai_complete(prompt: str, max_tokens: int = 200, threshold: float = CACHE_THRESHOLD) -> tuple[str, float]:
    """Call Pion AI.COMPLETE — handles cache + Ollama internally.
    Returns (response_text, latency_ms)."""
    t0 = time.perf_counter()
    result = r.execute_command(
        "AI.COMPLETE", prompt,
        "TOKENS", str(max_tokens),
        "THRESHOLD", str(threshold),
    )
    latency = (time.perf_counter() - t0) * 1000
    return (result or ""), latency


def check_prerequisites() -> tuple[bool, bool, bool]:
    """Returns (pion_ok, ollama_ok, cache_enabled)."""
    pion_ok = False
    try:
        r.ping()
        pion_ok = True
    except Exception as e:
        print(f"❌ Pion not available on port {PION_PORT}: {e}")
        print("   Start with: ./pion-server -w 1")

    ollama_ok = False
    models: list[str] = []
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        ollama_ok = True
    except Exception as e:
        print(f"⚠  Ollama not available: {e}")

    cache_enabled = False
    if pion_ok:
        try:
            # Probe: if embedding is off, returns ERR; if on, returns $-1 (miss) or a result
            r.execute_command("AI.SEMANTIC_CACHE", "GET", "__probe__")
            cache_enabled = True
        except Exception as e:
            if "not enabled" in str(e).lower() or "requires" in str(e).lower():
                cache_enabled = False
            else:
                cache_enabled = True  # any other error = server is responding

    return pion_ok, ollama_ok, cache_enabled


def run_demo() -> int:
    print("=" * 60)
    print("Pion AI.COMPLETE Demo — Semantic Cache in front of Ollama")
    print("=" * 60)

    pion_ok, ollama_ok, cache_enabled = check_prerequisites()

    if not pion_ok:
        return 1

    print(f"{'✅' if pion_ok else '❌'} Pion (port {PION_PORT})")
    print(f"{'✅' if ollama_ok else '⚠ '} Ollama ({OLLAMA_URL})")
    print(f"{'✅' if cache_enabled else '⚠ '} Semantic cache {'enabled' if cache_enabled else 'disabled (start Ollama + nomic-embed-text)'}")
    print()

    if not cache_enabled:
        print("To enable:")
        print("  ollama pull nomic-embed-text && ollama pull llama3.2:1b")
        print("  ./pion-server -w 1   # auto-detects Ollama on startup")
        return 1

    # ── Query pairs: (first query, semantically-similar follow-up) ────────────
    query_pairs = [
        (
            "What is the capital of France?",
            "Which city is France's capital?",
        ),
        (
            "Explain what a hash map is in one sentence.",
            "What is a hash table? One sentence please.",
        ),
        (
            "What is 2 + 2?",
            "What does 2 plus 2 equal?",
        ),
    ]

    cache_hits   = 0
    total_pairs  = 0
    ollama_times: list[float] = []
    cache_times:  list[float] = []

    for first_q, similar_q in query_pairs:
        print(f"Query 1: {first_q!r}")

        if ollama_ok:
            resp1, ms1 = ai_complete(first_q)
            # First call is always a cache miss (nothing stored yet)
            print(f"  [OLLAMA] {ms1:6.0f}ms → {resp1[:80]!r}{'...' if len(resp1) > 80 else ''}")
            ollama_times.append(ms1)
        else:
            print("  [SKIP] Ollama not available")
            ms1 = 0.0

        print(f"Similar: {similar_q!r}")
        if ollama_ok:
            resp2, ms2 = ai_complete(similar_q)
            speedup = ms1 / ms2 if ms2 > 0.5 else float("inf")
            is_hit  = ms2 < ms1 * 0.5 or ms2 < 50  # <50ms = definitely from cache
            hit_tag = "✅ CACHE HIT" if is_hit else "❌ miss"
            if is_hit:
                cache_hits += 1
                cache_times.append(ms2)
            else:
                ollama_times.append(ms2)
            print(f"  [PION ] {ms2:6.0f}ms → {resp2[:80]!r}{'...' if len(resp2) > 80 else ''}")
            print(f"  {hit_tag}  speedup: {speedup:.0f}×\n")
        total_pairs += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Results: {cache_hits}/{total_pairs} similar queries hit cache")

    if ollama_times:
        avg_ollama = sum(ollama_times) / len(ollama_times)
        print(f"Avg Ollama latency:  {avg_ollama:.0f}ms")
    if cache_times:
        avg_cache = sum(cache_times) / len(cache_times)
        print(f"Avg cache latency:   {avg_cache:.1f}ms")
        if ollama_times:
            print(f"Speedup on hits:     {avg_ollama / avg_cache:.0f}×")

    print()
    print("Pion AI.COMPLETE: one command = cache check + Ollama + cache store.")
    print("Same Redis protocol — any language, zero extra dependencies.")
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
