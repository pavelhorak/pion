#!/usr/bin/env python3
"""Pion L3 Concept Store + Fragment Store — inference distillation memory.

Two complementary stores for inference distillation:

  ConceptStore: Full query→response pairs. Catches paraphrases of known queries.
  FragmentStore: Sentence-level response fragments. Catches novel queries whose
                 answers can be assembled from fragments of previous responses.

Architecture:
  - Concepts/fragments stored in Pion as HSET (concept:<id>, frag:<id>)
  - Embeddings kept in-memory (numpy) for fast brute-force cosine search
  - Avoids single-HNSW-per-worker conflict (no FT.CREATE needed)
  - At <50K entries, numpy brute-force is faster than HNSW anyway (<1ms)

Usage:
    from concept_store import ConceptStore, FragmentStore
    store = ConceptStore(pion_host="127.0.0.1", pion_port=1974)
    frags = FragmentStore(pion_host="127.0.0.1", pion_port=1974, embed_fn=my_embed)
    store.ingest(query_embedding, response_text)
    frags.ingest_response(response_text)
    result = store.try_synthesize(query_embedding)
    frag_result = frags.try_synthesize(query_embedding)
"""

from __future__ import annotations

import json
import logging
import re
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import redis

log = logging.getLogger("pion-serve.l3")


@dataclass
class SynthesisResult:
    """Result from L3 concept synthesis."""
    response: str
    confidence: float
    concept_ids: list[int]
    cosine_similarity: float
    strategy: str  # "direct", "composite", "fragment_augmented"


@dataclass
class ConceptEntry:
    """In-memory concept entry (embedding + metadata pointer)."""
    concept_id: int
    embedding: np.ndarray  # normalized float32
    confidence: float = 1.0
    hit_count: int = 0
    miss_count: int = 0


class ConceptStore:
    """L3 concept memory for inference distillation.

    Three synthesis strategies:
      1. Direct: single concept with high confidence → return stored response
      2. Composite: 2-3 related concepts → concatenate response fragments
      3. Fragment-augmented: partial coverage → return fragments for LLM context

    Concepts are stored in Pion HSET for persistence; embeddings are kept
    in-memory as a numpy matrix for brute-force cosine search.
    """

    def __init__(
        self,
        pion_host: str = "127.0.0.1",
        pion_port: int = 1974,
        synthesis_threshold: float = 0.82,
        direct_threshold: float = 0.90,
        composite_threshold: float = 0.82,
        fragment_threshold: float = 0.70,
        confidence_min: float = 0.50,
        max_concepts: int = 50_000,
    ):
        self.redis = redis.Redis(host=pion_host, port=pion_port, decode_responses=False)
        self.synthesis_threshold = synthesis_threshold
        self.direct_threshold = direct_threshold
        self.composite_threshold = composite_threshold
        self.fragment_threshold = fragment_threshold
        self.confidence_min = confidence_min
        self.max_concepts = max_concepts

        # In-memory embedding index (avoids single-HNSW conflict)
        self._concepts: list[ConceptEntry] = []
        self._emb_matrix: Optional[np.ndarray] = None  # (N, dim) normalized
        self._dirty = True  # rebuild matrix on next search
        self._concept_count = 0

        # Stats
        self.stats = {
            "concepts": 0,
            "synthesis_hits": 0,
            "synthesis_misses": 0,
            "direct_hits": 0,
            "composite_hits": 0,
            "fragment_hits": 0,
            "avg_synthesis_ms": 0.0,
            "total_synthesis_ms": 0.0,
        }

        # Try to restore concept count from Pion
        self._restore_count()

    def _restore_count(self):
        """Restore concept counter from Pion (find highest existing concept:N)."""
        try:
            # Scan for existing concept keys
            cursor = 0
            max_id = -1
            while True:
                cursor, keys = self.redis.scan(cursor, match=b"concept:*", count=100)
                for k in keys:
                    try:
                        cid = int(k.decode().split(":")[1])
                        max_id = max(max_id, cid)
                    except (ValueError, IndexError):
                        pass
                if cursor == 0:
                    break
            if max_id >= 0:
                self._concept_count = max_id + 1
                log.info(f"Restored concept counter: {self._concept_count} concepts")
                self._reload_from_pion()
        except Exception as e:
            log.debug(f"Could not restore concepts: {e}")

    def _reload_from_pion(self):
        """Reload concept embeddings from Pion HSET into memory."""
        loaded = 0
        for cid in range(self._concept_count):
            try:
                data = self.redis.hgetall(f"concept:{cid}")
                if not data:
                    continue
                emb_bytes = data.get(b"embedding")
                if emb_bytes is None:
                    continue
                emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
                confidence = float(data.get(b"confidence", b"1.0"))
                hit_count = int(data.get(b"hit_count", b"0"))
                miss_count = int(data.get(b"miss_count", b"0"))
                self._concepts.append(ConceptEntry(
                    concept_id=cid,
                    embedding=emb,
                    confidence=confidence,
                    hit_count=hit_count,
                    miss_count=miss_count,
                ))
                loaded += 1
            except Exception:
                pass
        if loaded > 0:
            self._dirty = True
            self.stats["concepts"] = loaded
            log.info(f"Reloaded {loaded} concept embeddings into memory")

    def _rebuild_matrix(self):
        """Rebuild the (N, dim) embedding matrix for batch cosine search."""
        if not self._concepts:
            self._emb_matrix = None
            return
        self._emb_matrix = np.stack([c.embedding for c in self._concepts])
        self._dirty = False

    def _cosine_search(self, query_emb: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Brute-force cosine search. Returns [(index_in_concepts, similarity), ...]."""
        if self._dirty:
            self._rebuild_matrix()
        if self._emb_matrix is None or len(self._emb_matrix) == 0:
            return []

        # Normalize query
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        # Batch cosine similarity: (N, dim) @ (dim,) → (N,)
        sims = self._emb_matrix @ query_norm
        # Top-k
        k = min(k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(int(idx), float(sims[idx])) for idx in top_idx]

    def ingest(self, query_emb: np.ndarray, response: str, query_text: str = ""):
        """Store a concept from a completed inference observation.

        Args:
            query_emb: Normalized float32 embedding of the query.
            response: The full LLM response text.
            query_text: Original query text (for debugging/phrase extraction).
        """
        if self._concept_count >= self.max_concepts:
            log.warning(f"Concept store full ({self.max_concepts}), skipping ingest")
            return

        # Check for near-duplicate (don't store if very similar concept exists)
        if len(self._concepts) > 0:
            matches = self._cosine_search(query_emb, k=1)
            if matches and matches[0][1] > 0.95:
                # Very similar concept exists — reinforce it instead
                idx = matches[0][0]
                entry = self._concepts[idx]
                entry.hit_count += 1
                entry.confidence = min(1.0, entry.confidence + 0.01)
                try:
                    self.redis.hset(f"concept:{entry.concept_id}", mapping={
                        b"hit_count": str(entry.hit_count).encode(),
                        b"confidence": f"{entry.confidence:.4f}".encode(),
                        b"last_hit": str(int(time.time())).encode(),
                    })
                except Exception:
                    pass
                return

        cid = self._concept_count
        self._concept_count += 1

        # Normalize embedding
        emb = query_emb.astype(np.float32)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)

        # Store in Pion HSET
        try:
            self.redis.hset(f"concept:{cid}", mapping={
                b"embedding": emb_norm.tobytes(),
                b"response": response.encode("utf-8"),
                b"confidence": b"1.0",
                b"hit_count": b"0",
                b"miss_count": b"0",
                b"created_at": str(int(time.time())).encode(),
                b"last_hit": str(int(time.time())).encode(),
                b"phrase": query_text[:200].encode("utf-8") if query_text else b"",
            })
        except Exception as e:
            log.warning(f"Failed to store concept {cid} in Pion: {e}")
            self._concept_count -= 1
            return

        # Add to in-memory index
        entry = ConceptEntry(concept_id=cid, embedding=emb_norm, confidence=1.0)
        self._concepts.append(entry)
        self._dirty = True
        self.stats["concepts"] = len(self._concepts)

    def try_synthesize(self, query_emb: np.ndarray, k: int = 5) -> Optional[SynthesisResult]:
        """Try to answer from concept memory without LLM.

        Strategies (tried in order):
          1. Direct: best match cosine > direct_threshold and confidence > min
          2. Composite: top-2 concepts both > composite_threshold, compose response
          3. Fragment-augmented: coverage > fragment_threshold, return fragments

        Returns SynthesisResult or None if confidence too low.
        """
        if not self._concepts:
            return None

        t0 = time.perf_counter()
        matches = self._cosine_search(query_emb, k=k)

        if not matches:
            return None

        best_idx, best_sim = matches[0]
        best_entry = self._concepts[best_idx]

        # Strategy 1: Direct return
        if best_sim >= self.direct_threshold and best_entry.confidence >= self.confidence_min:
            response = self._get_response(best_entry.concept_id)
            if response:
                self._record_hit(best_entry, "direct")
                elapsed = (time.perf_counter() - t0) * 1000
                self._update_timing(elapsed)
                return SynthesisResult(
                    response=response,
                    confidence=best_entry.confidence,
                    concept_ids=[best_entry.concept_id],
                    cosine_similarity=best_sim,
                    strategy="direct",
                )

        # Strategy 2: Composite (top-2 concepts)
        if (len(matches) >= 2
                and best_sim >= self.composite_threshold
                and best_entry.confidence >= self.confidence_min):
            second_idx, second_sim = matches[1]
            second_entry = self._concepts[second_idx]
            if second_sim >= self.composite_threshold and second_entry.confidence >= self.confidence_min:
                r1 = self._get_response(best_entry.concept_id)
                r2 = self._get_response(second_entry.concept_id)
                if r1 and r2:
                    composed = self._compose_responses(r1, r2)
                    self._record_hit(best_entry, "composite")
                    self._record_hit(second_entry, "composite")
                    elapsed = (time.perf_counter() - t0) * 1000
                    self._update_timing(elapsed)
                    return SynthesisResult(
                        response=composed,
                        confidence=min(best_entry.confidence, second_entry.confidence),
                        concept_ids=[best_entry.concept_id, second_entry.concept_id],
                        cosine_similarity=best_sim,
                        strategy="composite",
                    )

        # Strategy 3: Fragment-augmented (return fragments for LLM context)
        if best_sim >= self.fragment_threshold and best_entry.confidence >= self.confidence_min:
            fragments = []
            fragment_ids = []
            for idx, sim in matches[:3]:
                if sim < self.fragment_threshold:
                    break
                entry = self._concepts[idx]
                if entry.confidence < self.confidence_min:
                    continue
                r = self._get_response(entry.concept_id)
                if r:
                    fragments.append(r)
                    fragment_ids.append(entry.concept_id)

            if fragments:
                # Build fragment-augmented context
                coverage = best_sim  # approximate
                for e_idx, s in matches[1:3]:
                    if s >= self.fragment_threshold:
                        coverage = (coverage + s) / 2

                self._record_hit(best_entry, "fragment_augmented")
                elapsed = (time.perf_counter() - t0) * 1000
                self._update_timing(elapsed)
                return SynthesisResult(
                    response="\n\n".join(fragments),
                    confidence=best_entry.confidence * coverage,
                    concept_ids=fragment_ids,
                    cosine_similarity=best_sim,
                    strategy="fragment_augmented",
                )

        # No synthesis possible
        self.stats["synthesis_misses"] += 1
        elapsed = (time.perf_counter() - t0) * 1000
        self._update_timing(elapsed)
        return None

    def record_feedback(self, concept_ids: list[int], positive: bool):
        """Record user feedback on a synthesized response.

        positive=True: user accepted (no follow-up correction)
        positive=False: user rephrased or corrected
        """
        for cid in concept_ids:
            # Find in-memory entry
            for entry in self._concepts:
                if entry.concept_id == cid:
                    if positive:
                        entry.hit_count += 1
                        entry.confidence = min(1.0, entry.confidence + 0.01)
                    else:
                        entry.miss_count += 1
                        entry.confidence = max(0.0, entry.confidence - 0.05)
                    try:
                        self.redis.hset(f"concept:{cid}", mapping={
                            b"hit_count": str(entry.hit_count).encode(),
                            b"miss_count": str(entry.miss_count).encode(),
                            b"confidence": f"{entry.confidence:.4f}".encode(),
                        })
                    except Exception:
                        pass
                    break

    def decay_confidence(self, factor: float = 0.99):
        """Apply daily confidence decay to all concepts."""
        for entry in self._concepts:
            entry.confidence *= factor
            if entry.confidence < 0.01:
                continue
            try:
                self.redis.hset(f"concept:{entry.concept_id}",
                                b"confidence", f"{entry.confidence:.4f}".encode())
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Return L3 statistics for /v1/stats."""
        total = self.stats["synthesis_hits"] + self.stats["synthesis_misses"]
        return {
            "concept_count": self.stats["concepts"],
            "synthesis_hits": self.stats["synthesis_hits"],
            "synthesis_misses": self.stats["synthesis_misses"],
            "synthesis_rate": round(self.stats["synthesis_hits"] / max(total, 1), 3),
            "direct_hits": self.stats["direct_hits"],
            "composite_hits": self.stats["composite_hits"],
            "fragment_hits": self.stats["fragment_hits"],
            "avg_synthesis_ms": round(self.stats["avg_synthesis_ms"], 2),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_response(self, concept_id: int) -> Optional[str]:
        """Fetch response text from Pion HSET."""
        try:
            resp = self.redis.hget(f"concept:{concept_id}", "response")
            if resp:
                return resp.decode("utf-8", errors="replace")
        except Exception:
            pass
        return None

    def _compose_responses(self, r1: str, r2: str) -> str:
        """Compose two response fragments into a coherent answer."""
        # Simple concatenation with separator.
        # Production version would use sentence-level deduplication.
        if len(r1) + len(r2) > 4000:
            # Truncate to avoid oversized responses
            r1 = r1[:2000]
            r2 = r2[:2000]
        return f"{r1}\n\nAdditionally:\n\n{r2}"

    def _record_hit(self, entry: ConceptEntry, strategy: str):
        """Record a synthesis hit."""
        entry.hit_count += 1
        entry.confidence = min(1.0, entry.confidence + 0.01)
        self.stats["synthesis_hits"] += 1
        if strategy == "direct":
            self.stats["direct_hits"] += 1
        elif strategy == "composite":
            self.stats["composite_hits"] += 1
        elif strategy == "fragment_augmented":
            self.stats["fragment_hits"] += 1
        try:
            self.redis.hset(f"concept:{entry.concept_id}", mapping={
                b"hit_count": str(entry.hit_count).encode(),
                b"confidence": f"{entry.confidence:.4f}".encode(),
                b"last_hit": str(int(time.time())).encode(),
            })
        except Exception:
            pass

    def _update_timing(self, elapsed_ms: float):
        """Update average synthesis timing."""
        total = self.stats["synthesis_hits"] + self.stats["synthesis_misses"]
        self.stats["total_synthesis_ms"] += elapsed_ms
        self.stats["avg_synthesis_ms"] = self.stats["total_synthesis_ms"] / max(total, 1)


# ── Fragment Store ────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Groups short sentences (1-3 per fragment)."""
    # Split on sentence boundaries
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty and very short
    raw = [s.strip() for s in raw if len(s.strip()) > 15]

    # Group into fragments of 1-3 sentences, targeting ~50-150 words each
    fragments = []
    buf = []
    buf_words = 0
    for sent in raw:
        words = len(sent.split())
        buf.append(sent)
        buf_words += words
        if buf_words >= 40 or len(buf) >= 3:
            fragments.append(" ".join(buf))
            buf = []
            buf_words = 0
    if buf:
        fragments.append(" ".join(buf))

    return fragments


@dataclass
class FragmentEntry:
    """In-memory fragment entry."""
    fragment_id: int
    embedding: np.ndarray
    source_concept_id: int  # which concept/response this came from
    confidence: float = 1.0


@dataclass
class FragmentSynthesisResult:
    """Result from fragment-level synthesis."""
    fragments: list[str]
    coverage: float          # 0.0-1.0, fraction of query covered
    confidence: float
    fragment_ids: list[int]
    strategy: str            # "full_synthesis" or "fragment_augmented"
    augmented_prompt: Optional[str] = None  # pre-built prompt for LLM


class FragmentStore:
    """L3b Fragment Store — sentence-level response caching.

    Decomposes every LLM response into semantic fragments (1-3 sentences),
    embeds each independently, and caches in a flat numpy index. New queries
    search across ALL fragments from ALL previous responses.

    Four-tier response model:
      L1:  Semantic cache hit (cosine > 0.92)         → 0% cost, 25ms
      L3a: Full fragment synthesis (coverage > 0.90)   → 0% cost, 50ms
      L3b: Fragment-augmented generation (30-90%)      → 30-50% cheaper
      Full inference (coverage < 0.30)                 → 100% cost

    The key insight vs naive response-level caching: queries don't repeat,
    but the knowledge fragments that compose answers DO. Even in diverse
    conversations, the same facts appear in different combinations.
    """

    def __init__(
        self,
        pion_host: str = "127.0.0.1",
        pion_port: int = 1974,
        embed_fn: Optional[Callable[[str], Optional[np.ndarray]]] = None,
        batch_embed_fn: Optional[Callable[[list[str]], list[Optional[np.ndarray]]]] = None,
        full_synthesis_threshold: float = 0.88,
        fragment_match_threshold: float = 0.75,
        min_coverage_for_augment: float = 0.30,
        max_fragments: int = 200_000,
    ):
        self.redis = redis.Redis(host=pion_host, port=pion_port, decode_responses=False)
        self.embed_fn = embed_fn
        self.batch_embed_fn = batch_embed_fn
        self.full_synthesis_threshold = full_synthesis_threshold
        self.fragment_match_threshold = fragment_match_threshold
        self.min_coverage_for_augment = min_coverage_for_augment
        self.max_fragments = max_fragments

        self._fragments: list[FragmentEntry] = []
        self._emb_matrix: Optional[np.ndarray] = None
        self._dirty = True
        self._fragment_count = 0

        self.stats = {
            "fragments": 0,
            "full_synthesis": 0,
            "fragment_augmented": 0,
            "misses": 0,
            "avg_ms": 0.0,
            "total_ms": 0.0,
        }

        self._restore_count()

    def _restore_count(self):
        """Restore fragment counter from Pion."""
        try:
            cursor = 0
            max_id = -1
            while True:
                cursor, keys = self.redis.scan(cursor, match=b"frag:*", count=200)
                for k in keys:
                    try:
                        fid = int(k.decode().split(":")[1])
                        max_id = max(max_id, fid)
                    except (ValueError, IndexError):
                        pass
                if cursor == 0:
                    break
            if max_id >= 0:
                self._fragment_count = max_id + 1
                self._reload_from_pion()
        except Exception as e:
            log.debug(f"Could not restore fragments: {e}")

    def _reload_from_pion(self):
        """Reload fragment embeddings from Pion into memory."""
        loaded = 0
        for fid in range(self._fragment_count):
            try:
                data = self.redis.hgetall(f"frag:{fid}")
                if not data:
                    continue
                emb_bytes = data.get(b"embedding")
                if emb_bytes is None:
                    continue
                emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
                src = int(data.get(b"source_concept_id", b"0"))
                conf = float(data.get(b"confidence", b"1.0"))
                self._fragments.append(FragmentEntry(
                    fragment_id=fid, embedding=emb,
                    source_concept_id=src, confidence=conf,
                ))
                loaded += 1
            except Exception:
                pass
        if loaded > 0:
            self._dirty = True
            self.stats["fragments"] = loaded
            log.info(f"Reloaded {loaded} fragment embeddings")

    def _rebuild_matrix(self):
        if not self._fragments:
            self._emb_matrix = None
            return
        self._emb_matrix = np.stack([f.embedding for f in self._fragments])
        self._dirty = False

    def _cosine_search(self, query_emb: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        if self._dirty:
            self._rebuild_matrix()
        if self._emb_matrix is None or len(self._emb_matrix) == 0:
            return []
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        sims = self._emb_matrix @ query_norm
        k = min(k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(int(idx), float(sims[idx])) for idx in top_idx]

    def ingest_response(self, response: str, source_concept_id: int = -1):
        """Decompose a response into fragments, embed each, and store.

        Uses batch embedding when available (10x faster than per-fragment calls).

        Args:
            response: Full LLM response text.
            source_concept_id: ID of the concept this response belongs to.
        """
        if not self.embed_fn and not self.batch_embed_fn:
            return

        fragments = _split_sentences(response)
        if not fragments:
            return

        # Batch embed all fragments at once if possible
        if self.batch_embed_fn:
            embeddings = self.batch_embed_fn(fragments)
        else:
            embeddings = [self.embed_fn(f) for f in fragments]

        now = str(int(time.time())).encode()
        src_bytes = str(source_concept_id).encode()

        for frag_text, emb in zip(fragments, embeddings):
            if self._fragment_count >= self.max_fragments:
                log.warning("Fragment store full")
                return

            if emb is None:
                continue
            emb = emb.astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-10

            # Check for near-duplicate fragment
            if self._fragments:
                matches = self._cosine_search(emb, k=1)
                if matches and matches[0][1] > 0.95:
                    existing = self._fragments[matches[0][0]]
                    existing.confidence = min(1.0, existing.confidence + 0.02)
                    continue

            fid = self._fragment_count
            self._fragment_count += 1

            try:
                self.redis.hset(f"frag:{fid}", mapping={
                    b"embedding": emb.tobytes(),
                    b"text": frag_text.encode("utf-8"),
                    b"source_concept_id": src_bytes,
                    b"confidence": b"1.0",
                    b"created_at": now,
                })
            except Exception as e:
                log.debug(f"Failed to store fragment {fid}: {e}")
                self._fragment_count -= 1
                continue

            self._fragments.append(FragmentEntry(
                fragment_id=fid, embedding=emb,
                source_concept_id=source_concept_id, confidence=1.0,
            ))
            self._dirty = True

        self.stats["fragments"] = len(self._fragments)

    def try_synthesize(self, query_emb: np.ndarray, k: int = 10) -> Optional[FragmentSynthesisResult]:
        """Search fragment index for relevant fragments.

        Returns:
          - Full synthesis (L3a) if coverage > 90% — bypass LLM entirely
          - Fragment-augmented (L3b) if coverage 30-90% — cheaper LLM call
          - None if coverage < 30%
        """
        if not self._fragments:
            return None

        t0 = time.perf_counter()
        matches = self._cosine_search(query_emb, k=k)
        if not matches:
            return None

        # Collect matching fragments above threshold
        matched_fragments = []
        matched_ids = []
        total_sim = 0.0
        seen_sources = set()

        for idx, sim in matches:
            if sim < self.fragment_match_threshold:
                break
            entry = self._fragments[idx]
            if entry.confidence < 0.3:
                continue

            frag_text = self._get_fragment_text(entry.fragment_id)
            if not frag_text:
                continue

            # Deduplicate fragments from same source
            if entry.source_concept_id in seen_sources and len(seen_sources) > 1:
                continue
            seen_sources.add(entry.source_concept_id)

            matched_fragments.append(frag_text)
            matched_ids.append(entry.fragment_id)
            total_sim += sim

        if not matched_fragments:
            self.stats["misses"] += 1
            elapsed = (time.perf_counter() - t0) * 1000
            self._update_stats(elapsed)
            return None

        # Coverage = average similarity of matched fragments
        coverage = total_sim / len(matched_fragments)
        avg_confidence = np.mean([
            self._fragments[idx].confidence
            for idx, sim in matches[:len(matched_fragments)]
            if sim >= self.fragment_match_threshold
        ])

        elapsed = (time.perf_counter() - t0) * 1000

        # L3a: Full synthesis (high coverage, bypass LLM)
        if coverage >= self.full_synthesis_threshold and len(matched_fragments) >= 2:
            self.stats["full_synthesis"] += 1
            self._update_stats(elapsed)
            return FragmentSynthesisResult(
                fragments=matched_fragments,
                coverage=coverage,
                confidence=float(avg_confidence),
                fragment_ids=matched_ids,
                strategy="full_synthesis",
            )

        # L3b: Fragment-augmented (partial coverage, cheaper LLM call)
        if coverage >= self.min_coverage_for_augment:
            context = "\n\n".join(matched_fragments)
            prompt = (
                "Using the following verified facts, answer the user's question. "
                "Only generate content for topics not covered by the facts.\n\n"
                f"Verified facts:\n{context}\n\n---\n\n"
            )
            self.stats["fragment_augmented"] += 1
            self._update_stats(elapsed)
            return FragmentSynthesisResult(
                fragments=matched_fragments,
                coverage=coverage,
                confidence=float(avg_confidence),
                fragment_ids=matched_ids,
                strategy="fragment_augmented",
                augmented_prompt=prompt,
            )

        self.stats["misses"] += 1
        self._update_stats(elapsed)
        return None

    def get_stats(self) -> dict:
        total = self.stats["full_synthesis"] + self.stats["fragment_augmented"] + self.stats["misses"]
        return {
            "fragment_count": self.stats["fragments"],
            "full_synthesis": self.stats["full_synthesis"],
            "fragment_augmented": self.stats["fragment_augmented"],
            "misses": self.stats["misses"],
            "hit_rate": round(
                (self.stats["full_synthesis"] + self.stats["fragment_augmented"]) / max(total, 1), 3
            ),
            "avg_ms": round(self.stats["avg_ms"], 2),
        }

    def _get_fragment_text(self, fragment_id: int) -> Optional[str]:
        try:
            text = self.redis.hget(f"frag:{fragment_id}", "text")
            if text:
                return text.decode("utf-8", errors="replace")
        except Exception:
            pass
        return None

    def _update_stats(self, elapsed_ms: float):
        total = self.stats["full_synthesis"] + self.stats["fragment_augmented"] + self.stats["misses"]
        self.stats["total_ms"] += elapsed_ms
        self.stats["avg_ms"] = self.stats["total_ms"] / max(total, 1)
