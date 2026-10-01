#!/usr/bin/env python3
"""Pion Serve — Phase 4: Git-Aware Cache Invalidation Benchmark.

Simulates a coding assistant workflow where:
  1. Cache is populated with completions about various files
  2. Some files change (git push)
  3. Invalidation marks affected cache entries as stale
  4. Subsequent lookups skip stale entries and trigger fresh prefill

Measures: invalidation accuracy, false invalidation rate, stale skip rate.

Usage:
    python benchmark_invalidation.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from vllm_pion.semantic_cache import SemanticCacheManager, CacheConfig
from vllm_pion.kv_serializer import serialize_kv_cache


# ── Test data ────────────────────────────────────────────────────────────────

# Prompts about specific files (will be cached)
FILE_PROMPTS = [
    ("src/auth/middleware.py", "Explain the authentication middleware in auth/middleware.py"),
    ("src/auth/models.py", "What are the user models in auth/models.py?"),
    ("src/api/handlers.py", "How do the API handlers work in api/handlers.py?"),
    ("src/api/routes.py", "Describe the routing in api/routes.py"),
    ("src/db/queries.py", "Explain the database queries in db/queries.py"),
    ("src/db/models.py", "What are the database models in db/models.py?"),
    ("src/frontend/app.tsx", "How does the React app component work in frontend/app.tsx?"),
    ("src/frontend/login.tsx", "Explain the login component in frontend/login.tsx"),
    ("tests/test_auth.py", "What do the auth tests cover in tests/test_auth.py?"),
    ("tests/test_api.py", "Describe the API tests in tests/test_api.py"),
    ("config/settings.yaml", "What configuration is in config/settings.yaml?"),
    ("Dockerfile", "How is the Docker build configured?"),
]

# Unrelated prompts (should NOT be invalidated by any file change)
UNRELATED_PROMPTS = [
    "What is the time complexity of quicksort?",
    "Explain how HTTP/2 multiplexing works",
    "Write a function to check if a string is a palindrome",
]


def make_fake_kv(seq_len: int = 32) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create synthetic KV layers for cache storage."""
    return [
        (np.random.randn(1, 8, seq_len, 64).astype(np.float16),
         np.random.randn(1, 8, seq_len, 64).astype(np.float16))
        for _ in range(4)
    ]


def run_benchmark():
    print("=" * 70)
    print("Pion Serve — Git-Aware Cache Invalidation Benchmark")
    print("=" * 70)

    # Initialize cache with git invalidation enabled
    config = CacheConfig(
        embed_provider="ngram",
        fallback_to_local=True,
        git_invalidation=True,
        git_invalidation_threshold=0.55,
    )

    cache = SemanticCacheManager(config)
    cache.connect()

    # ── Phase 1: Populate cache ──────────────────────────────────────────
    print("\n[1/4] Populating cache with file-related prompts...")

    cache_ids = {}
    for fpath, prompt in FILE_PROMPTS:
        kv = make_fake_kv()
        cache.store(prompt, kv)
        # Retrieve the cache_id that was generated
        cache_ids[fpath] = cache._local_ids[-1]

    for prompt in UNRELATED_PROMPTS:
        kv = make_fake_kv()
        cache.store(prompt, kv)

    total_entries = len(FILE_PROMPTS) + len(UNRELATED_PROMPTS)
    print(f"  Stored {total_entries} cache entries ({len(FILE_PROMPTS)} file-related, {len(UNRELATED_PROMPTS)} unrelated)")

    # Verify all lookups hit
    all_hit = True
    for _, prompt in FILE_PROMPTS:
        result = cache.lookup(prompt)
        if not result.hit:
            all_hit = False
    print(f"  Pre-invalidation: all lookups hit = {all_hit}")

    # ── Phase 2: Simulate git push (change some files) ───────────────────
    print("\n[2/4] Simulating git push — changing auth/ and api/ files...")

    changed_files = [
        "src/auth/middleware.py",
        "src/auth/models.py",
        "src/api/handlers.py",
    ]

    stale_entries = cache.invalidate_for_files(changed_files, source="test")
    print(f"  Changed files: {changed_files}")
    print(f"  Entries invalidated: {len(stale_entries)}")
    for entry in stale_entries:
        print(f"    - {entry.cache_id}: reason={entry.reason}, sim={entry.similarity:.3f}")

    # ── Phase 3: Verify lookups skip stale entries ───────────────────────
    print("\n[3/4] Verifying cache behavior after invalidation...")

    results = []
    for fpath, prompt in FILE_PROMPTS:
        result = cache.lookup(prompt)
        should_be_stale = fpath in changed_files
        actually_stale = result.stale_skip or not result.hit

        correct = (should_be_stale == actually_stale) or (not should_be_stale and result.hit)
        results.append({
            "file": fpath,
            "should_stale": should_be_stale,
            "hit": result.hit,
            "stale_skip": result.stale_skip,
            "correct": correct,
        })

    # Unrelated prompts should still hit
    for prompt in UNRELATED_PROMPTS:
        result = cache.lookup(prompt)
        results.append({
            "file": "(unrelated)",
            "should_stale": False,
            "hit": result.hit,
            "stale_skip": result.stale_skip,
            "correct": result.hit,
        })

    # ── Phase 4: Report ──────────────────────────────────────────────────
    print("\n[4/4] Results:")
    print()
    print(f"{'File':<35} {'Should Stale':>12} {'Hit':>5} {'Stale Skip':>11} {'Correct':>8}")
    print("-" * 75)

    for r in results:
        stale_str = "YES" if r["should_stale"] else "no"
        hit_str = "HIT" if r["hit"] else "MISS"
        skip_str = "SKIP" if r["stale_skip"] else "-"
        ok_str = "Y" if r["correct"] else "N"
        print(f"{r['file']:<35} {stale_str:>12} {hit_str:>5} {skip_str:>11} {ok_str:>8}")

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    stale_correct = sum(1 for r in results if r["should_stale"] and not r["hit"])
    stale_total = sum(1 for r in results if r["should_stale"])
    fresh_correct = sum(1 for r in results if not r["should_stale"] and r["hit"])
    fresh_total = sum(1 for r in results if not r["should_stale"])
    false_invalidations = sum(1 for r in results if not r["should_stale"] and not r["hit"])

    print()
    print(f"Overall accuracy:       {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"Stale correctly missed: {stale_correct}/{stale_total} ({stale_correct/stale_total*100:.0f}%)" if stale_total else "")
    print(f"Fresh correctly hit:    {fresh_correct}/{fresh_total} ({fresh_correct/fresh_total*100:.0f}%)" if fresh_total else "")
    print(f"False invalidations:    {false_invalidations}/{fresh_total}")
    print()

    # Cache stats
    stats = cache.stats
    print(f"Cache stats:")
    print(f"  Lookups: {stats.lookups}, Hits: {stats.hits}, Misses: {stats.misses}")
    print(f"  Stale skips: {stats.stale_skips}")
    print(f"  Invalidation events: {stats.invalidation_events}")
    print(f"  Entries invalidated: {stats.entries_invalidated}")
    print()

    # ── Phase 5: Webhook format test ─────────────────────────────────────
    print("Webhook format test:")
    github_payload = {
        "commits": [
            {"added": [], "modified": ["src/db/queries.py"], "removed": []},
            {"added": ["src/db/new_file.py"], "modified": [], "removed": []},
        ]
    }
    stale2 = cache.invalidate_from_webhook(github_payload)
    print(f"  GitHub webhook: {len(stale2)} entries invalidated for db/ changes")

    # Validation gate
    print()
    print("VALIDATION GATE:")
    stale_pass = stale_correct == stale_total if stale_total else True
    fp_pass = false_invalidations / fresh_total < 0.20 if fresh_total else True
    print(f"  Stale entries skipped:    {stale_correct}/{stale_total} — {'PASS' if stale_pass else 'FAIL'}")
    print(f"  False invalidation <20%:  {false_invalidations}/{fresh_total} — {'PASS' if fp_pass else 'FAIL'}")
    if stale_pass and fp_pass:
        print(f"\n  >>> GIT-AWARE INVALIDATION VALIDATED <<<")

    cache.close()


if __name__ == "__main__":
    run_benchmark()
