"""Pion KV Connector for vLLM v1.

Implements vLLM's KVConnectorBase_V1 interface to use Pion as an external
KV cache store with semantic prefix matching.

Phase 1 of M14 (Externalized Attention).

Usage with vLLM:
    VLLM_KV_CONNECTOR=vllm_pion.connector.PionKVConnector \\
    python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B

Configuration (via VLLM_KV_CONNECTOR_CONFIG env var or constructor):
    {
        "pion_host": "127.0.0.1",
        "pion_port": 1974,
        "embed_model": "nomic-embed-text",  # or "openai" for OpenAI embeddings
        "embed_host": "127.0.0.1",
        "embed_port": 11434,
        "embed_dim": 768,
        "cosine_threshold": 0.95,
        "model_tag": "llama-3-8b",
        "ttl": 3600,
    }
"""

import hashlib
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .client import PionKVClient

logger = logging.getLogger(__name__)


@dataclass
class PionConnectorConfig:
    """Configuration for the Pion KV connector."""
    pion_host: str = "127.0.0.1"
    pion_port: int = 1974
    embed_model: str = "nomic-embed-text"
    embed_host: str = "127.0.0.1"
    embed_port: int = 11434
    embed_dim: int = 768
    cosine_threshold: float = 0.95
    model_tag: str = ""
    ttl: int = 3600  # 1 hour default

    @classmethod
    def from_dict(cls, d: dict) -> "PionConnectorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PendingStore:
    """State for a request whose KV cache should be stored after generation."""
    prompt_hash: str
    prompt_text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class PendingLoad:
    """State for a request whose KV cache should be loaded from Pion."""
    cache_id: str
    num_tokens: int
    blob: Optional[bytes] = None


class PionKVConnector:
    """vLLM KV Connector that stores/retrieves KV cache in Pion.

    This is a standalone implementation that works with vLLM's connector interface.
    For use without vLLM, use PionKVClient directly.

    The connector:
    1. On new requests: embeds the prompt, checks Pion for a cached KV blob.
    2. On cache hit: loads the cached KV tensors (skipping prefill).
    3. On request completion: stores the KV tensors in Pion for future reuse.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = {}
        self.config = PionConnectorConfig.from_dict(config)
        self.pion = PionKVClient(self.config.pion_host, self.config.pion_port)
        self.pion.connect()

        # Embedding client (lazy-loaded)
        self._embedder = None

        # Tracking state for pending operations
        self._pending_stores: Dict[str, PendingStore] = {}
        self._pending_loads: Dict[str, PendingLoad] = {}

        logger.info(
            f"PionKVConnector initialized: {self.config.pion_host}:{self.config.pion_port}, "
            f"embed_dim={self.config.embed_dim}, threshold={self.config.cosine_threshold}"
        )

    def _get_embedder(self):
        """Lazy-load the embedding client. Falls back to hash-based if Ollama unavailable."""
        if self._embedder is None:
            try:
                import requests
                # Test connectivity to Ollama
                test_url = f"http://{self.config.embed_host}:{self.config.embed_port}/api/tags"
                requests.get(test_url, timeout=2)
                self._embedder = OllamaEmbedder(
                    host=self.config.embed_host,
                    port=self.config.embed_port,
                    model=self.config.embed_model,
                    dimensions=self.config.embed_dim,
                )
                logger.info(f"Using Ollama embedder: {self.config.embed_model}")
            except Exception:
                logger.warning("Ollama unavailable; using hash-based embeddings (no semantic matching)")
                self._embedder = HashEmbedder(dimensions=self.config.embed_dim)
        return self._embedder

    def embed_prompt(self, prompt_text: str) -> np.ndarray:
        """Embed a prompt text into a vector."""
        return self._get_embedder().embed(prompt_text[:512])

    # --- Scheduler-side methods ---

    def check_cache(self, prompt_text: str) -> Optional[Tuple[int, bytes]]:
        """Check if Pion has a cached KV for this prompt.

        Returns:
            (num_cached_tokens, blob) if cache hit, None if miss.
        """
        embedding = self.embed_prompt(prompt_text)
        blob = self.pion.kv_fetch(
            embedding,
            threshold=self.config.cosine_threshold,
            model=self.config.model_tag,
        )
        if blob is None:
            return None

        # Extract num_tokens from blob header (first 4 bytes = uint32 num_tokens)
        if len(blob) >= 4:
            num_tokens = struct.unpack("<I", blob[:4])[0]
            return (num_tokens, blob)

        return None

    def store_cache(self, prompt_text: str, kv_blob: bytes) -> bool:
        """Store KV cache tensors in Pion.

        Args:
            prompt_text: The prompt text (for embedding).
            kv_blob: Serialized KV cache tensors (with 4-byte num_tokens header).

        Returns:
            True if stored successfully.
        """
        embedding = self.embed_prompt(prompt_text)
        cache_id = hashlib.sha256(prompt_text.encode()[:512]).hexdigest()[:16]

        return self.pion.kv_store(
            cache_id=cache_id,
            embedding=embedding,
            blob=kv_blob,
            ttl=self.config.ttl,
            model=self.config.model_tag,
        )

    def get_stats(self) -> dict:
        """Get Pion KV cache statistics."""
        return self.pion.kv_info()

    def close(self):
        """Close the connection."""
        self.pion.close()


class OllamaEmbedder:
    """Embedding client using Ollama's /api/embeddings endpoint."""

    def __init__(self, host: str, port: int, model: str, dimensions: int):
        self.url = f"http://{host}:{port}/api/embeddings"
        self.model = model
        self.dimensions = dimensions

    def embed(self, text: str) -> np.ndarray:
        import requests
        resp = requests.post(self.url, json={"model": self.model, "prompt": text}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        vec = np.array(data["embedding"], dtype=np.float32)
        # Normalize to unit norm
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class HashEmbedder:
    """Deterministic hash-based embedding (no external dependencies).

    Generates a unit-norm vector from the SHA256 hash of the input text.
    Not semantically meaningful — only useful for exact prefix matching.
    For real semantic matching, use OllamaEmbedder or OpenAI.
    """

    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        # Expand hash to fill dimensions using PRNG seeded by hash
        rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
        vec = rng.randn(self.dimensions).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec
