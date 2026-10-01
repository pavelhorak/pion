"""SemanticCacheManager — prompt-level KV cache with semantic matching.

The core integration point for Pion Serve. Wraps PionKVClient with:
  - Prompt embedding for semantic matching
  - Configurable similarity threshold
  - RoPE re-rotation on cache hits for position alignment
  - Cache statistics tracking
  - Optional local fallback cache (when Pion server is unavailable)

Usage:
    cache = SemanticCacheManager(config)
    cache.connect()

    # On new request:
    result = cache.lookup(prompt_text)
    if result.hit:
        # Use result.kv_layers with result.rope_offset for zero-prefill
        ...
    else:
        # Run full prefill, then store:
        cache.store(prompt_text, kv_layers, num_tokens)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .client import PionKVClient
from .kv_serializer import serialize_kv_cache, deserialize_kv_cache, KVCacheHeader
from .prompt_embedder import Embedder, create_embedder
from .rope_rerotation import RoPEConfig, rerotate_keys
from .git_invalidator import GitCacheInvalidator, StaleCacheEntry


@dataclass
class CacheConfig:
    """Configuration for SemanticCacheManager."""
    # Pion connection
    pion_host: str = "127.0.0.1"
    pion_port: int = 1974

    # Embedding
    embed_provider: str = "auto"  # "ngram", "openai", "ollama", "auto"
    embed_dim: int = 1536

    # Cache matching
    cosine_threshold: float = 0.90  # minimum similarity for cache hit
    model_tag: str = ""  # filter by model family
    ttl: int = 3600  # cache entry TTL in seconds

    # RoPE re-rotation (set from model config)
    rope_head_dim: int = 0  # 0 = disabled
    rope_base_theta: float = 10000.0
    rope_traditional: bool = False

    # KV serialization
    use_fp16: bool = True  # FP16 reduces wire size by 2x

    # Fallback
    fallback_to_local: bool = True  # use local numpy cache if Pion unavailable
    local_cache_capacity: int = 100  # max entries in local fallback

    # Git-aware invalidation
    git_invalidation: bool = False  # enable git-aware cache invalidation
    git_invalidation_threshold: float = 0.60  # similarity threshold for invalidation
    git_repo_root: str = ""  # git repo root (auto-detected if empty)


@dataclass
class CacheLookupResult:
    """Result of a cache lookup."""
    hit: bool
    similarity: float = 0.0
    kv_layers: Optional[list[tuple[np.ndarray, np.ndarray]]] = None
    header: Optional[KVCacheHeader] = None
    cache_id: str = ""
    lookup_time_ms: float = 0.0
    stale_skip: bool = False  # True if a match was found but skipped due to staleness

    @property
    def num_tokens(self) -> int:
        return self.header.seq_len if self.header else 0


@dataclass
class CacheStats:
    """Cumulative cache statistics."""
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    total_lookup_ms: float = 0.0
    total_store_ms: float = 0.0
    pion_errors: int = 0
    local_fallback_hits: int = 0
    stale_skips: int = 0  # cache matches skipped due to staleness
    invalidation_events: int = 0
    entries_invalidated: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups > 0 else 0.0

    @property
    def avg_lookup_ms(self) -> float:
        return self.total_lookup_ms / self.lookups if self.lookups > 0 else 0.0


class SemanticCacheManager:
    """Prompt-level semantic KV cache manager.

    Embeds prompts, queries Pion's HNSW-indexed KV cache store for
    semantically similar entries, and applies RoPE re-rotation on hits.
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._client: Optional[PionKVClient] = None
        self._embedder: Optional[Embedder] = None
        self._rope_config: Optional[RoPEConfig] = None
        self._invalidator: Optional[GitCacheInvalidator] = None
        self._stats = CacheStats()
        self._connected = False

        # Local fallback cache
        self._local_embeddings: list[np.ndarray] = []
        self._local_blobs: list[bytes] = []
        self._local_ids: list[str] = []

    def connect(self):
        """Initialize embedder and connect to Pion."""
        self._embedder = create_embedder(
            self.config.embed_provider,
            dim=self.config.embed_dim,
        )

        if self.config.rope_head_dim > 0:
            self._rope_config = RoPEConfig.from_base_theta(
                head_dim=self.config.rope_head_dim,
                base=self.config.rope_base_theta,
                traditional=self.config.rope_traditional,
            )

        # Git-aware invalidation
        if self.config.git_invalidation:
            self._invalidator = GitCacheInvalidator(
                embedder=self._embedder,
                similarity_threshold=self.config.git_invalidation_threshold,
                repo_root=self.config.git_repo_root,
            )

        try:
            self._client = PionKVClient(
                host=self.config.pion_host,
                port=self.config.pion_port,
            )
            self._client.connect()
            self._connected = True
        except Exception as e:
            if self.config.fallback_to_local:
                print(f"[SemanticCacheManager] Pion unavailable ({e}), using local fallback")
                self._connected = False
            else:
                raise

    def close(self):
        """Close connections."""
        if self._client:
            self._client.close()
            self._client = None
        self._connected = False

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def lookup(self, prompt_text: str) -> CacheLookupResult:
        """Look up a prompt in the semantic cache.

        Args:
            prompt_text: The full prompt to match.

        Returns:
            CacheLookupResult with hit=True if a similar prompt's KV cache was found.
        """
        t0 = time.perf_counter()
        self._stats.lookups += 1

        embedding = self._embedder.embed(prompt_text)

        # Try Pion first
        if self._connected:
            try:
                blob = self._client.kv_fetch(
                    embedding,
                    threshold=self.config.cosine_threshold,
                    model=self.config.model_tag,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                self._stats.total_lookup_ms += elapsed

                if blob is not None:
                    header, layers = deserialize_kv_cache(blob)
                    self._stats.hits += 1
                    return CacheLookupResult(
                        hit=True,
                        similarity=self.config.cosine_threshold,  # Pion doesn't return exact sim
                        kv_layers=layers,
                        header=header,
                        lookup_time_ms=elapsed,
                    )
                else:
                    self._stats.misses += 1
                    return CacheLookupResult(hit=False, lookup_time_ms=elapsed)

            except Exception:
                self._stats.pion_errors += 1
                if not self.config.fallback_to_local:
                    elapsed = (time.perf_counter() - t0) * 1000
                    self._stats.total_lookup_ms += elapsed
                    self._stats.misses += 1
                    return CacheLookupResult(hit=False, lookup_time_ms=elapsed)

        # Local fallback
        result = self._local_lookup(embedding)
        elapsed = (time.perf_counter() - t0) * 1000
        self._stats.total_lookup_ms += elapsed

        if result.hit:
            self._stats.hits += 1
            self._stats.local_fallback_hits += 1
            result.lookup_time_ms = elapsed
        else:
            self._stats.misses += 1
            result.lookup_time_ms = elapsed

        return result

    def store(
        self,
        prompt_text: str,
        kv_layers: list[tuple[np.ndarray, np.ndarray]],
        cache_id: str = "",
    ) -> bool:
        """Store KV cache tensors for a prompt.

        Args:
            prompt_text: The prompt that produced this KV cache.
            kv_layers: List of (keys, values) per layer.
            cache_id: Optional cache ID (auto-generated if empty).

        Returns:
            True if stored successfully.
        """
        t0 = time.perf_counter()
        self._stats.stores += 1

        if not cache_id:
            cache_id = f"pserve:{uuid.uuid4().hex[:12]}"

        embedding = self._embedder.embed(prompt_text)
        blob = serialize_kv_cache(kv_layers, use_fp16=self.config.use_fp16)

        stored = False

        # Try Pion
        if self._connected:
            try:
                stored = self._client.kv_store(
                    cache_id=cache_id,
                    embedding=embedding,
                    blob=blob,
                    ttl=self.config.ttl,
                    model=self.config.model_tag,
                )
            except Exception:
                self._stats.pion_errors += 1

        # Local fallback (always store locally for resilience)
        if self.config.fallback_to_local:
            self._local_store(embedding, blob, cache_id)
            stored = True

        elapsed = (time.perf_counter() - t0) * 1000
        self._stats.total_store_ms += elapsed
        return stored

    def rerotate_cached_keys(
        self,
        kv_layers: list[tuple[np.ndarray, np.ndarray]],
        original_offset: int,
        target_offset: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Apply RoPE re-rotation to cached KV tensors for position alignment.

        Args:
            kv_layers: Cached (keys, values) per layer.
            original_offset: Position offset when keys were originally computed.
            target_offset: Desired position offset for injection.

        Returns:
            New list of (rerotated_keys, values) per layer.
        """
        if self._rope_config is None:
            raise ValueError("RoPE config not set — set rope_head_dim in CacheConfig")

        result = []
        for keys, values in kv_layers:
            k_float = keys.astype(np.float32)
            k_rerotated = rerotate_keys(
                k_float,
                original_offset=original_offset,
                target_offset=target_offset,
                config=self._rope_config,
            )
            result.append((k_rerotated.astype(keys.dtype), values))
        return result

    # ── Local fallback cache ─────────────────────────────────────────────

    # ── Git-aware invalidation ─────────────────────────────────────────

    @property
    def invalidator(self) -> Optional[GitCacheInvalidator]:
        return self._invalidator

    def invalidate_for_files(self, changed_files: list[str], source: str = "manual") -> list[StaleCacheEntry]:
        """Invalidate cache entries affected by changed files.

        Args:
            changed_files: List of file paths (relative to repo root).
            source: Event source for tracking.

        Returns:
            List of newly invalidated entries.
        """
        if not self._invalidator:
            return []

        stale = self._invalidator.invalidate_for_files(changed_files, source=source)
        self._stats.invalidation_events += 1
        self._stats.entries_invalidated += len(stale)
        return stale

    def invalidate_from_webhook(self, payload: dict) -> list[StaleCacheEntry]:
        """Process a GitHub/GitLab push webhook payload."""
        if not self._invalidator:
            return []

        stale = self._invalidator.invalidate_from_webhook(payload)
        self._stats.invalidation_events += 1
        self._stats.entries_invalidated += len(stale)
        return stale

    def poll_git_changes(self) -> list[StaleCacheEntry]:
        """Poll git for changes since last check."""
        if not self._invalidator:
            return []

        stale = self._invalidator.poll_git_changes()
        if stale:
            self._stats.invalidation_events += 1
            self._stats.entries_invalidated += len(stale)
        return stale

    # ── Local fallback cache ─────────────────────────────────────────────

    def _local_lookup(self, embedding: np.ndarray) -> CacheLookupResult:
        """Brute-force cosine search over local cache, respecting staleness."""
        if not self._local_embeddings:
            return CacheLookupResult(hit=False)

        matrix = np.stack(self._local_embeddings)  # (N, dim)
        sims = matrix @ embedding  # unit-normalized → cosine

        # Sort by similarity descending, skip stale entries
        sorted_indices = np.argsort(-sims)
        for idx in sorted_indices:
            sim = float(sims[idx])
            if sim < self.config.cosine_threshold:
                break  # remaining are all below threshold

            cache_id = self._local_ids[idx]

            # Check staleness
            if self._invalidator and self._invalidator.is_stale(cache_id):
                self._stats.stale_skips += 1
                continue  # skip stale entry, try next

            blob = self._local_blobs[idx]
            header, layers = deserialize_kv_cache(blob)
            return CacheLookupResult(
                hit=True,
                similarity=sim,
                kv_layers=layers,
                header=header,
                cache_id=cache_id,
            )

        # Check if we found a match but it was stale
        best_sim = float(sims[sorted_indices[0]]) if len(sorted_indices) > 0 else 0.0
        if best_sim >= self.config.cosine_threshold:
            best_id = self._local_ids[sorted_indices[0]]
            if self._invalidator and self._invalidator.is_stale(best_id):
                return CacheLookupResult(hit=False, similarity=best_sim, stale_skip=True)

        return CacheLookupResult(hit=False, similarity=best_sim)

    def _local_store(self, embedding: np.ndarray, blob: bytes, cache_id: str):
        """Store in local fallback cache with LRU eviction."""
        if len(self._local_embeddings) >= self.config.local_cache_capacity:
            # Evict oldest
            evicted_id = self._local_ids[0]
            self._local_embeddings.pop(0)
            self._local_blobs.pop(0)
            self._local_ids.pop(0)
            if self._invalidator:
                self._invalidator.untrack_entry(evicted_id)

        self._local_embeddings.append(embedding)
        self._local_blobs.append(blob)
        self._local_ids.append(cache_id)

        # Track for invalidation
        if self._invalidator:
            self._invalidator.track_entry(cache_id, embedding, cache_id)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
