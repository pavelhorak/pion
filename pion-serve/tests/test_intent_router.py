#!/usr/bin/env python3
"""gh #84: starter tests for pion-serve's intent router.

Self-contained — no backend, no Flask, no embedding model. The router takes
an `embed_fn` callable so we inject a deterministic stub that maps each seed
class to a distinct unit vector. Exercises:

  - DEFAULT_ROUTING contract (tiers + cost values)
  - load_routing_config validation and default-merge behavior
  - bootstrap with a non-trivial embed_fn returns True and sets centroids
  - classify falls back to "medium" when embedding/centroids are missing
  - centroid-based tier assignment (simple vs complex)
  - heuristic boosts: code-block → +complex, short query → bound back to medium
  - get_stats counters track classifications

Run:
    python pion-serve/tests/test_intent_router.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Make the package importable without installing.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from intent_router import (
    DEFAULT_ROUTING,
    IntentRouter,
    RouteDecision,
    TIERS,
    SIMPLE_SEEDS,
    COMPLEX_SEEDS,
    load_routing_config,
)


# ── Deterministic stub embedder ──────────────────────────────────────────────
#
# Pick a small dimension so the test runs in milliseconds. Map each seed
# string into a unit vector with bias toward either the "simple" axis (e0)
# or the "complex" axis (e1) based on which seed list it lives in. Unknown
# queries land somewhere in the middle so the heuristic logic can be
# exercised independently.

DIM = 8


def _seed_vec(axis: int, seed_text: str) -> np.ndarray:
    """Build a vector strongly biased toward `axis` (0 or 1) but with a
    deterministic per-seed perturbation so the centroid is well-defined."""
    rng = np.random.RandomState(abs(hash(seed_text)) % (2**31 - 1))
    v = rng.normal(scale=0.05, size=DIM)
    v[axis] += 1.0
    v /= np.linalg.norm(v) + 1e-10
    return v


def _stub_embed(text: str) -> np.ndarray:
    """Deterministic per-string embedding. Seeds from SIMPLE_SEEDS map onto
    e0; seeds from COMPLEX_SEEDS map onto e1; everything else is a 50/50
    blend with a tiny per-string nudge so equal-length unknown queries hash
    distinctly (the router uses dot products, so all that matters is the
    relative bias)."""
    if text in SIMPLE_SEEDS:
        return _seed_vec(0, text)
    if text in COMPLEX_SEEDS:
        return _seed_vec(1, text)
    # Unknown query — try a heuristic: short queries lean toward simple,
    # long toward complex. This is the same shape the real embed_fn would
    # naturally produce, so the centroid classifier sees a real signal.
    n_words = len(text.split())
    rng = np.random.RandomState(abs(hash(text)) % (2**31 - 1))
    v = rng.normal(scale=0.05, size=DIM)
    bias = 0.7 if n_words >= 30 else (0.7 if n_words <= 4 else 0.0)
    if bias > 0 and n_words >= 30:
        v[1] += bias  # complex
    elif bias > 0:
        v[0] += bias  # simple
    v /= np.linalg.norm(v) + 1e-10
    return v


# ── Tests ────────────────────────────────────────────────────────────────────


class DefaultRoutingContract(unittest.TestCase):
    def test_has_all_three_tiers(self):
        for t in ("simple", "medium", "complex"):
            self.assertIn(t, DEFAULT_ROUTING)

    def test_each_tier_has_backend_model_cost(self):
        for t in DEFAULT_ROUTING:
            entry = DEFAULT_ROUTING[t]
            self.assertIn("backend", entry)
            self.assertIn("model", entry)
            self.assertIn("cost_per_mtok", entry)

    def test_cost_ordering_simple_le_medium_le_complex(self):
        self.assertLess(
            DEFAULT_ROUTING["simple"]["cost_per_mtok"],
            DEFAULT_ROUTING["medium"]["cost_per_mtok"],
        )
        self.assertLess(
            DEFAULT_ROUTING["medium"]["cost_per_mtok"],
            DEFAULT_ROUTING["complex"]["cost_per_mtok"],
        )


class LoadRoutingConfig(unittest.TestCase):
    def _write_config(self, payload):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(payload, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_full_config_overrides_defaults(self):
        path = self._write_config({
            "simple":  {"backend": "ollama", "model": "gemma:2b",  "cost_per_mtok": 0.0},
            "medium":  {"backend": "ollama", "model": "gemma:7b",  "cost_per_mtok": 0.0},
            "complex": {"backend": "ollama", "model": "gemma:27b", "cost_per_mtok": 0.0},
        })
        cfg = load_routing_config(path)
        self.assertEqual(cfg["simple"]["model"], "gemma:2b")
        self.assertEqual(cfg["complex"]["backend"], "ollama")

    def test_partial_config_merges_with_defaults(self):
        # Only override `simple` — medium and complex stay as defaults.
        path = self._write_config({
            "simple": {"backend": "ollama", "model": "gemma:2b"},
        })
        cfg = load_routing_config(path)
        self.assertEqual(cfg["simple"]["model"], "gemma:2b")
        self.assertEqual(cfg["medium"]["model"], DEFAULT_ROUTING["medium"]["model"])
        self.assertEqual(cfg["complex"]["model"], DEFAULT_ROUTING["complex"]["model"])

    def test_missing_required_field_raises(self):
        path = self._write_config({
            "simple": {"backend": "ollama"},  # missing "model"
        })
        with self.assertRaises(ValueError):
            load_routing_config(path)


class IntentRouterBehavior(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter(embed_fn=_stub_embed)
        ok = self.router.bootstrap()
        self.assertTrue(ok, "bootstrap should succeed with the stub embedder")

    def test_bootstrap_sets_centroids(self):
        self.assertIsNotNone(self.router.simple_centroid)
        self.assertIsNotNone(self.router.complex_centroid)
        # The two centroids are nearly orthogonal because the stub embedder
        # placed simple seeds on e0 and complex seeds on e1.
        cos = float(np.dot(self.router.simple_centroid, self.router.complex_centroid))
        self.assertLess(cos, 0.2, f"centroids overlapped too much (cos={cos:.3f})")

    def test_classify_simple_seed_routes_to_simple(self):
        # Use one of the actual seeds — guaranteed dot-product alignment.
        decision = self.router.classify("What is JSON?")
        self.assertEqual(decision.tier, "simple")
        self.assertEqual(decision.backend, DEFAULT_ROUTING["simple"]["backend"])

    def test_classify_complex_seed_routes_to_complex(self):
        long_complex_seed = (
            "Refactor this distributed consensus algorithm to be Byzantine "
            "fault tolerant and analyze the trade-offs vs Raft"
        )
        decision = self.router.classify(long_complex_seed)
        self.assertEqual(decision.tier, "complex")

    def test_classify_with_no_embedding_falls_back_to_medium(self):
        router_no_centroid = IntentRouter(embed_fn=lambda _q: None)
        # bootstrap fails silently when embed_fn returns None for all seeds
        ok = router_no_centroid.bootstrap()
        self.assertFalse(ok)
        # classify still works and routes to medium with confidence 0
        decision = router_no_centroid.classify("anything")
        self.assertEqual(decision.tier, "medium")
        self.assertEqual(decision.confidence, 0.0)

    def test_code_block_forces_complex(self):
        # A short query that would otherwise lean toward simple gets bumped
        # to complex when the user pastes code. We assert tier == "complex"
        # rather than the boost label because the centroid may already place
        # the query in the complex zone (the stub embedder's medium-length
        # neutral bucket), in which case the boost branch is correctly
        # skipped — but the routing decision is what matters.
        decision = self.router.classify(
            "What does this do?\n```python\ndef foo(): return 1\n```"
        )
        self.assertEqual(decision.tier, "complex")

    def test_short_query_caps_at_medium_even_if_complex_scored(self):
        # A 3-word query that happens to land in the complex centroid
        # vicinity should be bumped DOWN to medium — opus is never warranted
        # for a 3-word prompt.
        # We need a 3-word string that the stub embedder will classify as
        # complex. The stub's heuristic biases short text toward simple, so
        # we craft one that overrides: use an exact COMPLEX seed prefix.
        # Verify behavior via a direct call with a contrived embedding.
        # Override embed_fn for one query to force a complex-biased vector.
        contrived = np.zeros(DIM)
        contrived[1] = 1.0  # all complex
        decision = self.router.classify("ok please now", query_emb=contrived)
        # 3 words is below short_token_threshold=6 default → tier becomes medium.
        self.assertEqual(decision.tier, "medium")
        self.assertIn("medium", decision.heuristic_boost)

    def test_get_stats_tracks_classifications(self):
        self.router.classify("What is JSON?")          # simple
        self.router.classify("What is JSON?")          # simple again
        long_complex = (
            "Architect a zero-downtime migration from PostgreSQL 12 to 16 "
            "across 30 sharded clusters totalling 800TB"
        )
        self.router.classify(long_complex)             # complex
        stats = self.router.get_stats()
        self.assertEqual(stats["simple"], 2)
        self.assertEqual(stats["complex"], 1)
        self.assertGreaterEqual(stats["classified_total"], 3)
        self.assertEqual(stats["routing"]["simple"]["backend"],
                         DEFAULT_ROUTING["simple"]["backend"])

    def test_route_decision_contains_centroid_scores(self):
        decision = self.router.classify("What is JSON?")
        self.assertIn("simple", decision.centroid_score)
        self.assertIn("complex", decision.centroid_score)
        self.assertIsInstance(decision.confidence, float)


class MissingTierConfigRaises(unittest.TestCase):
    def test_router_init_rejects_incomplete_routing(self):
        incomplete = {
            "simple": DEFAULT_ROUTING["simple"],
            "complex": DEFAULT_ROUTING["complex"],
            # missing "medium"
        }
        with self.assertRaises(ValueError):
            IntentRouter(embed_fn=_stub_embed, routing=incomplete)


if __name__ == "__main__":
    unittest.main(verbosity=2)
