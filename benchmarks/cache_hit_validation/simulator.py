"""Cache hit rate simulator — exact-prefix (SHA256) vs semantic (cosine) matching.

Embeds all prompts via the configured embedding provider, then simulates both
cache strategies at multiple similarity thresholds.
"""
from __future__ import annotations

import hashlib
import struct
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .workloads import CacheEntry, QueryEntry


@dataclass
class QueryDetail:
    """Per-query result for detailed analysis."""
    query_idx: int
    category: str
    subtype: str
    expected_hit: bool
    exact_hit: bool
    best_similarity: float
    best_match_idx: int
    prompt_preview: str  # first 80 chars


@dataclass
class CategoryResult:
    category: str
    threshold: float
    total: int
    exact_hits: int
    semantic_hits: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def exact_hit_rate(self) -> float:
        return self.exact_hits / self.total if self.total else 0.0

    @property
    def semantic_hit_rate(self) -> float:
        return self.semantic_hits / self.total if self.total else 0.0

    @property
    def fp_rate(self) -> float:
        negatives = self.total - (self.true_positives + self.false_negatives)
        return self.false_positives / negatives if negatives > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 1.0


@dataclass
class SimulationResults:
    thresholds: list[float]
    categories: list[str]
    results: dict[tuple[str, float], CategoryResult]  # (category, threshold) -> result
    details: list[QueryDetail]
    embed_time_s: float
    total_prompts_embedded: int
    embedding_model: str

    # FLOP savings estimates
    exact_flop_savings_pct: float = 0.0
    semantic_flop_savings_pct: dict[float, float] = field(default_factory=dict)


class ExactPrefixCache:
    """SHA256 hash of full prompt — match only on byte-identical text."""

    def __init__(self):
        self._hashes: dict[str, int] = {}

    def store(self, prompt: str, idx: int):
        h = hashlib.sha256(prompt.encode()).hexdigest()
        self._hashes[h] = idx

    def fetch(self, prompt: str) -> tuple[bool, int]:
        h = hashlib.sha256(prompt.encode()).hexdigest()
        if h in self._hashes:
            return True, self._hashes[h]
        return False, -1


class SemanticCache:
    """Cosine similarity brute-force search over embedded prompts."""

    def __init__(self):
        self._embeddings: list[np.ndarray] = []
        self._indices: list[int] = []
        self._matrix: np.ndarray | None = None

    def store(self, embedding: np.ndarray, idx: int):
        self._embeddings.append(embedding)
        self._indices.append(idx)
        self._matrix = None  # invalidate

    def _ensure_matrix(self):
        if self._matrix is None and self._embeddings:
            self._matrix = np.stack(self._embeddings)  # (N, dim)

    def fetch(self, embedding: np.ndarray, threshold: float) -> tuple[bool, int, float]:
        """Returns (hit, matched_cache_idx, best_similarity)."""
        self._ensure_matrix()
        if self._matrix is None:
            return False, -1, 0.0
        # All embeddings are unit-normalized, so dot product = cosine similarity
        sims = self._matrix @ embedding  # (N,)
        best_pos = int(np.argmax(sims))
        best_sim = float(sims[best_pos])
        if best_sim >= threshold:
            return True, self._indices[best_pos], best_sim
        return False, -1, best_sim


def _embed_ngram(prompts: list[str], dim: int = 1536) -> np.ndarray:
    """Local semantic embeddings via character n-gram hashing.

    Preserves text similarity: prompts sharing code/words have high cosine.
    No API key needed. Used as default when no embedding provider is available.
    """
    import hashlib as hl
    embeddings = np.zeros((len(prompts), dim), dtype=np.float32)
    for i, text in enumerate(prompts):
        # Generate character 4-grams and hash each to a dimension
        text_lower = text.lower()
        for n in range(len(text_lower) - 3):
            gram = text_lower[n:n + 4]
            h = int(hl.md5(gram.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 128) & 1 else -1.0
            embeddings[i, idx] += sign
        # Also add word-level features for better semantic grouping
        for word in text_lower.split():
            if len(word) >= 3:
                h = int(hl.md5(word.encode()).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if (h >> 128) & 1 else -1.0
                embeddings[i, idx] += sign * 2.0  # weight words higher
        # L2 normalize
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] /= norm
        if (i + 1) % 50 == 0 or i == len(prompts) - 1:
            print(f"    {i+1}/{len(prompts)} embedded", end="\r")
    print()
    return embeddings


def _embed_all(prompts: list[str], use_mock: bool = False) -> np.ndarray:
    """Embed all prompts, return (N, dim) float32 array."""
    import os

    if use_mock:
        # Use local n-gram embeddings (semantic-preserving, no API needed)
        print(f"  Embedding {len(prompts)} prompts (1536d, provider=ngram-local)...")
        return _embed_ngram(prompts, dim=1536)

    # Try OpenAI, fall back to Ollama, fall back to local n-gram
    provider = os.environ.get("PION_EMBED_PROVIDER", "").lower()
    if not provider:
        # Auto-detect: check for API key or Ollama
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            # Check if Ollama is running
            try:
                import requests
                requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
                provider = "ollama"
            except Exception:
                print("  No OPENAI_API_KEY and Ollama not running — using local n-gram embeddings")
                print(f"  Embedding {len(prompts)} prompts (1536d, provider=ngram-local)...")
                return _embed_ngram(prompts, dim=1536)

    os.environ["PION_EMBED_PROVIDER"] = provider
    if provider == "openai":
        os.environ.setdefault("PION_EMBED_MODEL", "text-embedding-3-small")

    # Import after setting env vars
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from pion_context.embeddings import embed_texts, _dim

    dim = _dim()
    print(f"  Embedding {len(prompts)} prompts ({dim}d, provider={provider})...")

    # Batch in chunks of 100 to respect API limits
    all_bytes: list[bytes] = []
    batch_size = 100
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_bytes = embed_texts(batch)
        all_bytes.extend(batch_bytes)
        done = min(i + batch_size, len(prompts))
        print(f"    {done}/{len(prompts)} embedded", end="\r")
    print()

    # Convert to numpy
    embeddings = np.zeros((len(prompts), dim), dtype=np.float32)
    for i, raw in enumerate(all_bytes):
        n = len(raw) // 4
        floats = struct.unpack(f"{n}f", raw)
        embeddings[i, :n] = floats[:dim]

    return embeddings


def _estimate_tokens(prompt: str) -> int:
    """Rough token count estimate (words * 1.3)."""
    return max(1, int(len(prompt.split()) * 1.3))


def run_simulation(
    cache_entries: list[CacheEntry],
    queries: list[QueryEntry],
    thresholds: list[float] | None = None,
    use_mock: bool = False,
) -> SimulationResults:
    """Run the full simulation.

    1. Embed all prompts (cache entries + queries, deduplicated).
    2. Build exact-prefix and semantic caches.
    3. For each threshold, evaluate all queries.
    4. Compute per-category and overall metrics.
    """
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]

    # Collect all unique prompts for embedding
    prompt_to_idx: dict[str, int] = {}
    all_prompts: list[str] = []
    for entry in cache_entries:
        if entry.prompt not in prompt_to_idx:
            prompt_to_idx[entry.prompt] = len(all_prompts)
            all_prompts.append(entry.prompt)
    for query in queries:
        if query.prompt not in prompt_to_idx:
            prompt_to_idx[query.prompt] = len(all_prompts)
            all_prompts.append(query.prompt)

    print(f"  Unique prompts: {len(all_prompts)} (from {len(cache_entries)} cache + {len(queries)} queries)")

    # Embed everything
    t0 = time.time()
    embeddings = _embed_all(all_prompts, use_mock=use_mock)
    embed_time = time.time() - t0
    print(f"  Embedding time: {embed_time:.1f}s")

    import os
    if use_mock:
        embedding_model = "ngram-local (1536d, no API)"
    else:
        embedding_model = os.environ.get("PION_EMBED_MODEL", os.environ.get("PION_EMBED_PROVIDER", "unknown"))

    # Build caches
    exact_cache = ExactPrefixCache()
    semantic_cache = SemanticCache()

    for i, entry in enumerate(cache_entries):
        exact_cache.store(entry.prompt, i)
        emb_idx = prompt_to_idx[entry.prompt]
        semantic_cache.store(embeddings[emb_idx], i)

    # Evaluate queries at each threshold
    categories = sorted(set(q.category for q in queries))
    all_categories = categories + ["overall"]

    # Pre-compute exact hits and best similarities for all queries
    query_exact: list[tuple[bool, int]] = []
    query_best_sim: list[tuple[float, int]] = []

    for query in queries:
        exact_hit, exact_idx = exact_cache.fetch(query.prompt)
        query_exact.append((exact_hit, exact_idx))

        emb_idx = prompt_to_idx[query.prompt]
        # Get best similarity (use threshold=0 to always get best match)
        _, best_match, best_sim = semantic_cache.fetch(embeddings[emb_idx], threshold=0.0)
        query_best_sim.append((best_sim, best_match))

    # Build details list (threshold-independent)
    details: list[QueryDetail] = []
    for qi, query in enumerate(queries):
        exact_hit = query_exact[qi][0]
        best_sim, best_match = query_best_sim[qi]
        details.append(QueryDetail(
            query_idx=qi,
            category=query.category,
            subtype=query.metadata.get("subtype", ""),
            expected_hit=query.expected_hit,
            exact_hit=exact_hit,
            best_similarity=best_sim,
            best_match_idx=best_match,
            prompt_preview=query.prompt[:80].replace("\n", " "),
        ))

    # Compute results per threshold per category
    results: dict[tuple[str, float], CategoryResult] = {}

    # FLOP savings calculation
    total_possible_flops = sum(_estimate_tokens(q.prompt) ** 2 for q in queries)
    exact_saved_flops = 0
    semantic_saved_flops: dict[float, int] = {t: 0 for t in thresholds}

    for threshold in thresholds:
        # Per-category counters
        counters: dict[str, dict[str, int]] = {}
        for cat in all_categories:
            counters[cat] = {"total": 0, "exact": 0, "semantic": 0, "tp": 0, "fp": 0, "fn": 0}

        for qi, query in enumerate(queries):
            cat = query.category
            exact_hit = query_exact[qi][0]
            best_sim = query_best_sim[qi][0]
            semantic_hit = best_sim >= threshold
            token_flops = _estimate_tokens(query.prompt) ** 2

            for target in [cat, "overall"]:
                counters[target]["total"] += 1
                if exact_hit:
                    counters[target]["exact"] += 1
                if semantic_hit:
                    counters[target]["semantic"] += 1
                    if query.expected_hit:
                        counters[target]["tp"] += 1
                    else:
                        counters[target]["fp"] += 1
                else:
                    if query.expected_hit:
                        counters[target]["fn"] += 1

            if exact_hit and threshold == thresholds[0]:
                exact_saved_flops += token_flops
            if semantic_hit:
                semantic_saved_flops[threshold] += token_flops

        for cat in all_categories:
            c = counters[cat]
            results[(cat, threshold)] = CategoryResult(
                category=cat,
                threshold=threshold,
                total=c["total"],
                exact_hits=c["exact"],
                semantic_hits=c["semantic"],
                true_positives=c["tp"],
                false_positives=c["fp"],
                false_negatives=c["fn"],
            )

    exact_flop_pct = (exact_saved_flops / total_possible_flops * 100) if total_possible_flops else 0
    sem_flop_pcts = {t: (v / total_possible_flops * 100) if total_possible_flops else 0
                     for t, v in semantic_saved_flops.items()}

    return SimulationResults(
        thresholds=thresholds,
        categories=all_categories,
        results=results,
        details=details,
        embed_time_s=embed_time,
        total_prompts_embedded=len(all_prompts),
        embedding_model=embedding_model,
        exact_flop_savings_pct=exact_flop_pct,
        semantic_flop_savings_pct=sem_flop_pcts,
    )
