#!/usr/bin/env python3
"""Pion Intent Router — semantic complexity classifier for tiered model routing.

Classifies every query in <1 ms and routes to the cheapest model that can
answer it well.

Three tiers:
  simple  → cheap model (Haiku / Flash / gemma)        $0.25 / Mtok
  medium  → mid-tier model (Sonnet / gemma4)           $3.00 / Mtok
  complex → top-tier model (Opus / GPT-4)              $15.00 / Mtok

Classification = centroid cosine + length / code heuristics. The centroids
are bootstrapped at startup from a built-in seed set (20 simple + 20 complex
example queries). Embedding is reused from the calling layer (auto-embed
sidecar or Ollama nomic-embed-text).

Routing override is a JSON file mapping tier → {backend, model, [base_url]}:

    {
      "simple":  {"backend": "claude", "model": "claude-haiku-4-5-20251001"},
      "medium":  {"backend": "claude", "model": "claude-sonnet-4-6"},
      "complex": {"backend": "claude", "model": "claude-opus-4-7"}
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("pion-serve.route")

# ── Built-in seed sets ──────────────────────────────────────────────────────

SIMPLE_SEEDS = [
    "What is a hash map?",
    "Define a function in Python",
    "What is the capital of France?",
    "How do I print to stdout?",
    "What does HTTP stand for?",
    "List the primary colors",
    "What is a variable?",
    "Define machine learning",
    "What is JSON?",
    "How do I declare a list?",
    "What is the difference between a list and a tuple?",
    "What is a class?",
    "What does DRY stand for?",
    "What is recursion?",
    "Explain the term 'cache' in one sentence",
    "What is REST?",
    "Define overfitting",
    "What is git?",
    "How do I check Python version?",
    "What is the meaning of TTL?",
]

COMPLEX_SEEDS = [
    "Refactor this distributed consensus algorithm to be Byzantine fault tolerant and analyze the trade-offs vs Raft",
    "Debug a race condition in a lock-free queue using memory ordering primitives and provide a reproducer",
    "Design a multi-tenant rate limiter that supports sliding window quotas with subsecond fairness across replicas",
    "Compare the trade-offs between Paxos, Raft, and ZAB for a write-heavy distributed log store with high tail latency requirements",
    "Optimise this SIMD inner loop for AVX-512 while keeping the scalar fallback path correct on ARM Neon",
    "Walk through the lifecycle of an io_uring SQE under SQPOLL with kernel-bypass NIC submission queues",
    "Analyse this PyTorch backward pass and explain why the gradients are NaN after the first batch",
    "Architect a zero-downtime migration from PostgreSQL 12 to 16 across 30 sharded clusters totalling 800TB",
    "Implement an HNSW index with INT4 quantisation, prefix pruning, and suffix early-exit, and benchmark recall vs QPS",
    "Derive the closed-form variance of importance sampling estimators under Pareto-tailed weights",
    "Reason about the correctness of this cooperative multitasking scheduler when interrupted at arbitrary instruction boundaries",
    "Design and prove correctness of a CRDT for collaborative ordered lists tolerant to concurrent splits",
    "Refactor this monolithic codebase into a hexagonal architecture while preserving the public API surface",
    "Compare LSM tree vs B+ tree storage engines for an append-mostly workload with wide range scans and bursty writes",
    "Engineer a fault-tolerant streaming pipeline with exactly-once semantics across Kafka, Flink, and Cassandra",
    "Trace through a use-after-free in this C++ template metaprogram and propose a refactor that preserves zero overhead",
    "Optimise this transformer attention kernel for an Apple Silicon UMA backend with bandwidth-bound dispatch",
    "Audit this OAuth2 server for token replay, scope escalation, and PKCE downgrade attacks",
    "Design a horizontally scalable approximate-nearest-neighbour service supporting both insert and delete with stable recall",
    "Reverse engineer the protocol of this proprietary binary stream and propose a parser that handles malformed framing safely",
]

# ── Tier defaults ──────────────────────────────────────────────────────────

# Default routing — Anthropic three-tier. Override via --route-config JSON.
DEFAULT_ROUTING: dict[str, dict] = {
    "simple":  {"backend": "claude", "model": "claude-haiku-4-5-20251001", "cost_per_mtok": 0.25},
    "medium":  {"backend": "claude", "model": "claude-sonnet-4-6",         "cost_per_mtok": 3.00},
    "complex": {"backend": "claude", "model": "claude-opus-4-7",           "cost_per_mtok": 15.00},
}

TIERS = ("simple", "medium", "complex")

# Heuristic thresholds
_CODE_BLOCK_RE = re.compile(
    r"```"                       # markdown fence
    r"|^\s{4,}\S"                # 4+ space indent at line start (code block)
    r"|\bdef \w+\("              # python def
    r"|\bclass \w+\s*[:({\[]"    # python/JS class def — requires follow-up : ( { [
    r"|\bSELECT .+ FROM\b",      # SQL
    re.I | re.M,
)
_STACK_TRACE_RE = re.compile(r"Traceback|at \w+\.\w+\(|File \".+\", line \d+", re.I)


# ── Result type ────────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    tier: str                  # "simple" | "medium" | "complex"
    backend: str               # "claude" | "ollama" | "vllm" | "openai" | "gemini" | "llamacpp"
    model: str
    base_url: Optional[str]
    cost_per_mtok: float
    centroid_score: dict[str, float]   # {"simple": cos, "complex": cos}
    heuristic_boost: str               # "" | "+complex (code)" | "+simple (short)" | …
    confidence: float                  # |s - c| margin in [0, 1]


# ── Router ─────────────────────────────────────────────────────────────────

@dataclass
class IntentRouter:
    """Centroid-based intent classifier with heuristic boosts."""
    embed_fn: Callable[[str], Optional[np.ndarray]]
    routing: dict[str, dict] = field(default_factory=lambda: dict(DEFAULT_ROUTING))
    margin: float = 0.05               # |s-c| > margin → confident; else medium
    short_token_threshold: int = 6     # ≤6 whitespace tokens → simple boost
    long_token_threshold: int = 60     # ≥60 tokens → complex boost
    simple_centroid: Optional[np.ndarray] = None
    complex_centroid: Optional[np.ndarray] = None

    def __post_init__(self):
        for tier in TIERS:
            if tier not in self.routing:
                raise ValueError(f"routing config missing tier '{tier}'")
        self._stats = {tier: 0 for tier in TIERS}
        self._stats["unclassified"] = 0  # embed_fn returned None

    def bootstrap(self,
                  simple_seeds: Optional[list[str]] = None,
                  complex_seeds: Optional[list[str]] = None) -> bool:
        """Embed seed sets and compute centroids. Returns True on success."""
        simples = simple_seeds or SIMPLE_SEEDS
        complexes = complex_seeds or COMPLEX_SEEDS

        s_vecs = [v for v in (self.embed_fn(s) for s in simples) if v is not None]
        c_vecs = [v for v in (self.embed_fn(s) for s in complexes) if v is not None]

        if not s_vecs or not c_vecs:
            log.warning("intent router: bootstrap embedding failed (got %d simple, %d complex)",
                        len(s_vecs), len(c_vecs))
            return False

        s_mean = np.mean(np.stack(s_vecs), axis=0)
        c_mean = np.mean(np.stack(c_vecs), axis=0)
        self.simple_centroid = s_mean / (np.linalg.norm(s_mean) + 1e-10)
        self.complex_centroid = c_mean / (np.linalg.norm(c_mean) + 1e-10)
        log.info("intent router: bootstrapped from %d/%d seeds (centroid sim=%.3f)",
                 len(s_vecs), len(c_vecs), float(np.dot(self.simple_centroid, self.complex_centroid)))
        return True

    def classify(self, query: str, query_emb: Optional[np.ndarray] = None) -> RouteDecision:
        """Classify a query and return the route decision. Never raises.

        If centroids are missing or the embedding is unavailable, falls back
        to the medium tier (safe default).
        """
        if query_emb is None:
            query_emb = self.embed_fn(query)
        if query_emb is None or self.simple_centroid is None or self.complex_centroid is None:
            self._stats["unclassified"] += 1
            return self._decision("medium", {"simple": 0.0, "complex": 0.0}, "(no embedding)", 0.0)

        s = float(np.dot(query_emb, self.simple_centroid))
        c = float(np.dot(query_emb, self.complex_centroid))
        scores = {"simple": s, "complex": c}

        # Centroid base tier
        diff = s - c
        if diff > self.margin:
            tier = "simple"
        elif diff < -self.margin:
            tier = "complex"
        else:
            tier = "medium"

        # Heuristic boosts (override centroid in obvious cases)
        boost = ""
        if _CODE_BLOCK_RE.search(query) or _STACK_TRACE_RE.search(query):
            if tier != "complex":
                boost = "+complex (code/trace)"
                tier = "complex"
        else:
            tok_count = len(query.split())
            if tok_count <= self.short_token_threshold and tier == "complex":
                # very short query never warrants opus
                boost = "+medium (short query)"
                tier = "medium"
            elif tok_count >= self.long_token_threshold and tier == "simple":
                boost = "+medium (long query)"
                tier = "medium"

        confidence = min(1.0, abs(diff) / max(self.margin, 1e-6) / 5.0)
        self._stats[tier] += 1
        return self._decision(tier, scores, boost, confidence)

    def _decision(self, tier: str, scores: dict[str, float],
                  boost: str, confidence: float) -> RouteDecision:
        cfg = self.routing[tier]
        return RouteDecision(
            tier=tier,
            backend=cfg["backend"],
            model=cfg["model"],
            base_url=cfg.get("base_url"),
            cost_per_mtok=float(cfg.get("cost_per_mtok", 0.0)),
            centroid_score=scores,
            heuristic_boost=boost,
            confidence=round(confidence, 3),
        )

    def get_stats(self) -> dict:
        total = sum(self._stats[t] for t in TIERS) or 1
        return {
            "classified_total": sum(self._stats[t] for t in TIERS),
            "unclassified": self._stats["unclassified"],
            "simple": self._stats["simple"],
            "medium": self._stats["medium"],
            "complex": self._stats["complex"],
            "simple_pct":  round(self._stats["simple"]  / total * 100, 1),
            "medium_pct":  round(self._stats["medium"]  / total * 100, 1),
            "complex_pct": round(self._stats["complex"] / total * 100, 1),
            "routing": {t: {k: v for k, v in self.routing[t].items()
                            if k in ("backend", "model", "cost_per_mtok")}
                        for t in TIERS},
        }


# ── Routing config loader ──────────────────────────────────────────────────

def load_routing_config(path: str) -> dict[str, dict]:
    """Load a routing override from a JSON file. Validates required fields."""
    with open(path) as f:
        cfg = json.load(f)
    out = dict(DEFAULT_ROUTING)
    for tier in TIERS:
        if tier in cfg:
            entry = cfg[tier]
            if "backend" not in entry or "model" not in entry:
                raise ValueError(f"routing config tier '{tier}' missing backend/model")
            merged = dict(out[tier])
            merged.update(entry)
            out[tier] = merged
    return out
