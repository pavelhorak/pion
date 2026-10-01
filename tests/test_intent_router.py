#!/usr/bin/env python3
"""Tests for pion-serve/intent_router.py — semantic intent classifier.

Two modes:
  - Offline (default): deterministic fake embedder built from token overlap
    with the seed sets. No external services required.
  - Live (--live):     real embedder via Ollama nomic-embed-text. Validates
    that the production embedding path classifies a held-out set correctly.

Usage:
  python3 tests/test_intent_router.py
  python3 tests/test_intent_router.py --live   # requires Ollama running
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make pion-serve importable regardless of where this is run from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pion-serve"))

from intent_router import (  # noqa: E402
    COMPLEX_SEEDS,
    DEFAULT_ROUTING,
    IntentRouter,
    SIMPLE_SEEDS,
    load_routing_config,
)


# ── Deterministic fake embedder ────────────────────────────────────────────
# Build a 384-dim signal where dim 0 = simple-token mass, dim 1 = complex-token
# mass, rest near-zero. After bootstrap, simple_centroid ≈ e0 and complex_centroid
# ≈ e1, so cosine to each is just the corresponding component (post-normalisation).

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")

def _toks(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s)}

_SIMPLE_VOCAB = set().union(*(_toks(s) for s in SIMPLE_SEEDS))
_COMPLEX_VOCAB = set().union(*(_toks(s) for s in COMPLEX_SEEDS))

DIM = 384


def fake_embed(text: str) -> np.ndarray:
    toks = _toks(text)
    s = len(toks & _SIMPLE_VOCAB)
    c = len(toks & _COMPLEX_VOCAB)
    # Light noise so centroids actually average meaningfully across distinct seeds
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(0, 0.01, DIM).astype(np.float32)
    v[0] += float(s)
    v[1] += float(c)
    n = np.linalg.norm(v)
    return v / (n + 1e-10)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_bootstrap_succeeds():
    r = IntentRouter(embed_fn=fake_embed)
    assert r.bootstrap()
    assert r.simple_centroid is not None
    assert r.complex_centroid is not None
    # The two centroids should be meaningfully separated, not collinear.
    cos = float(np.dot(r.simple_centroid, r.complex_centroid))
    assert cos < 0.5, f"centroids too similar (cos={cos})"
    print(f"  bootstrap ok (centroid cos={cos:.3f})")


def test_classifies_held_out_simple():
    r = IntentRouter(embed_fn=fake_embed)
    assert r.bootstrap()
    held_out = [
        "What is a queue?",
        "Define recursion",
        "How do I print a list?",
        "What does TTL mean?",
        "What is a class in Python?",
    ]
    correct = 0
    for q in held_out:
        d = r.classify(q)
        if d.tier in ("simple", "medium"):  # never complex for these
            correct += 1
    assert correct == len(held_out), f"only {correct}/{len(held_out)} classified non-complex"
    print(f"  held-out simple classified non-complex: {correct}/{len(held_out)}")


def test_classifies_held_out_complex():
    r = IntentRouter(embed_fn=fake_embed)
    assert r.bootstrap()
    held_out = [
        "Refactor a Byzantine fault tolerant algorithm and prove correctness across 30 sharded clusters",
        "Optimise this AVX-512 SIMD kernel under bandwidth-bound dispatch on Apple Silicon UMA",
        "Debug a use-after-free in this lock-free queue with memory ordering primitives",
    ]
    correct = 0
    for q in held_out:
        d = r.classify(q)
        if d.tier in ("complex", "medium"):  # never simple
            correct += 1
    assert correct == len(held_out), f"only {correct}/{len(held_out)} classified non-simple"
    print(f"  held-out complex classified non-simple: {correct}/{len(held_out)}")


def test_code_block_forces_complex():
    r = IntentRouter(embed_fn=fake_embed)
    assert r.bootstrap()
    q = "What is a list?\n```\ndef foo(x):\n    return x + 1\n```\nHow do I call this?"
    d = r.classify(q)
    assert d.tier == "complex", f"code block should force complex, got {d.tier}"
    assert "code" in d.heuristic_boost
    print(f"  code-block boost: tier={d.tier} boost={d.heuristic_boost!r}")


def test_short_query_caps_at_medium():
    r = IntentRouter(embed_fn=fake_embed)
    assert r.bootstrap()
    # A four-word "complex"-vocab query — centroid would say complex, heuristic
    # should drop it to medium.
    q = "Refactor algorithm SIMD kernel"
    d = r.classify(q)
    assert d.tier in ("simple", "medium"), f"short query went to {d.tier}, expected ≤ medium"
    print(f"  short-query cap: tier={d.tier} boost={d.heuristic_boost!r}")


def test_routing_table_lookup():
    r = IntentRouter(embed_fn=fake_embed)
    assert r.bootstrap()
    d = r.classify("What is a hash map?")
    assert d.backend in ("claude", "ollama", "openai", "vllm", "gemini", "llamacpp")
    assert d.model
    assert d.cost_per_mtok >= 0.0
    print(f"  routing entry: {d.tier} → {d.backend}/{d.model} @ ${d.cost_per_mtok}/Mtok")


def test_no_embedding_falls_back_to_medium():
    """If the embedder returns None (e.g. sidecar down), classify must not crash."""
    r = IntentRouter(embed_fn=lambda _t: None)
    # bootstrap returns False but classify still has to be safe
    ok = r.bootstrap()
    assert not ok
    d = r.classify("anything")
    assert d.tier == "medium"
    assert d.backend == DEFAULT_ROUTING["medium"]["backend"]
    print("  no-embedding fallback → medium")


def test_route_config_override():
    cfg = {
        "simple":  {"backend": "ollama", "model": "smollm:135m",  "base_url": "http://x:1"},
        "medium":  {"backend": "ollama", "model": "gemma3:4b"},
        "complex": {"backend": "claude", "model": "claude-opus-4-7"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        path = f.name
    try:
        loaded = load_routing_config(path)
    finally:
        os.unlink(path)
    r = IntentRouter(embed_fn=fake_embed, routing=loaded)
    assert r.bootstrap()
    d = r.classify("What is a list?")
    # simple-leaning → routes to overridden simple model
    if d.tier == "simple":
        assert d.model == "smollm:135m"
        assert d.base_url == "http://x:1"
    print(f"  override: tier={d.tier} model={d.model}")


def test_stats_track_distribution():
    r = IntentRouter(embed_fn=fake_embed)
    assert r.bootstrap()
    queries = [
        "What is a list?",
        "Define a function",
        "Refactor this Byzantine algorithm with formal correctness proof and SIMD kernel rewrite",
        "How do I declare a tuple?",
    ]
    for q in queries:
        r.classify(q)
    stats = r.get_stats()
    assert stats["classified_total"] == len(queries)
    assert stats["simple"] + stats["medium"] + stats["complex"] == len(queries)
    print(f"  distribution: simple={stats['simple']} medium={stats['medium']} complex={stats['complex']}")


# ── Live mode ──────────────────────────────────────────────────────────────

def _ollama_embed(text: str):
    import requests
    try:
        r = requests.post(
            "http://127.0.0.1:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=10,
        )
        if r.ok:
            v = np.array(r.json()["embedding"], dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-10
            return v
    except Exception:
        return None
    return None


def test_live_classifies_real_embeddings():
    if _ollama_embed("hello") is None:
        print("  SKIP — Ollama nomic-embed-text not reachable")
        return
    r = IntentRouter(embed_fn=_ollama_embed)
    assert r.bootstrap()
    held_simple = ["What is a tuple?", "Define overfitting", "What does CPU stand for?"]
    held_complex = [
        "Engineer a fault-tolerant streaming pipeline with exactly-once semantics across Kafka and Flink",
        "Audit this OAuth2 server for token replay and PKCE downgrade attacks",
    ]
    s_correct = sum(1 for q in held_simple if r.classify(q).tier in ("simple", "medium"))
    c_correct = sum(1 for q in held_complex if r.classify(q).tier in ("complex", "medium"))
    assert s_correct >= len(held_simple) - 1, f"simple drift: {s_correct}/{len(held_simple)}"
    assert c_correct >= len(held_complex) - 1, f"complex drift: {c_correct}/{len(held_complex)}"
    print(f"  live: simple ok {s_correct}/{len(held_simple)}, complex ok {c_correct}/{len(held_complex)}")


# ── Runner ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="Also run live test against Ollama")
    args = p.parse_args()

    tests = [
        test_bootstrap_succeeds,
        test_classifies_held_out_simple,
        test_classifies_held_out_complex,
        test_code_block_forces_complex,
        test_short_query_caps_at_medium,
        test_routing_table_lookup,
        test_no_embedding_falls_back_to_medium,
        test_route_config_override,
        test_stats_track_distribution,
    ]
    if args.live:
        tests.append(test_live_classifies_real_embeddings)

    failed = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"[ run] {name}")
            t()
            print(f"[ ok ] {name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
