"""Pion Layer Store client — binary protocol for per-layer KV tensor operations.

Phase 2 of M14. Communicates with Pion's binary protocol (0xCA5E framing)
for low-latency per-layer KV cache store/fetch operations.

Also provides a RESP fallback via KV.STORE for environments where the
binary listener isn't available.
"""

import hashlib
import socket
import struct
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .client import PionKVClient

# Binary protocol constants
BINARY_MAGIC = 0xCA5E
CMD_LAYER_STORE = 0x10
CMD_LAYER_FETCH = 0x11
CMD_LAYER_FETCH_BATCH = 0x12
CMD_LAYER_EXTEND = 0x13
CMD_PING = 0xFF
STATUS_OK = 0x00
STATUS_MISS = 0x01
STATUS_ERROR = 0x02


class PionLayerClient:
    """Client for Pion's layer-granular KV store.

    Supports two modes:
    1. Binary protocol (default, sub-ms latency) — for direct binary port
    2. RESP fallback (via KV.STORE/FETCH) — for standard RESP port

    In RESP mode, layers are stored as KV.STORE with cache_id = "layer:{session}:{layer_id}"
    and a unique embedding per (session, layer_id) pair.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1974,
        mode: str = "resp",  # "binary" or "resp"
        embed_dim: int = 768,
        timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.mode = mode
        self.embed_dim = embed_dim
        self.timeout = timeout
        self._resp_client: Optional[PionKVClient] = None

        # Per-session layer tracking
        self._session_embeddings: Dict[str, Dict[int, np.ndarray]] = {}

    def _get_resp_client(self) -> PionKVClient:
        if self._resp_client is None:
            self._resp_client = PionKVClient(self.host, self.port, self.timeout)
            self._resp_client.connect()
        return self._resp_client

    def _get_layer_embedding(self, session_id: str, layer_id: int) -> np.ndarray:
        """Generate a deterministic embedding for a (session, layer) pair.

        Uses a seeded PRNG to produce a consistent unit-norm vector for each
        (session_id, layer_id) combination. This ensures KV.FETCH with the same
        (session, layer) returns the correct blob.
        """
        if session_id not in self._session_embeddings:
            self._session_embeddings[session_id] = {}
        if layer_id not in self._session_embeddings[session_id]:
            # Deterministic seed from session_id + layer_id
            seed_str = f"{session_id}:layer:{layer_id}"
            seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            emb = rng.randn(self.embed_dim).astype(np.float32)
            emb /= np.linalg.norm(emb)
            self._session_embeddings[session_id][layer_id] = emb
        return self._session_embeddings[session_id][layer_id]

    def store_layer(
        self,
        session_id: str,
        layer_id: int,
        tensor: bytes,
    ) -> bool:
        """Store a single layer's KV tensor.

        Args:
            session_id: Inference session identifier.
            layer_id: Transformer layer index (0-based).
            tensor: Raw KV tensor bytes.

        Returns:
            True if stored successfully.
        """
        if self.mode == "resp":
            return self._store_layer_resp(session_id, layer_id, tensor)
        else:
            return self._store_layer_binary(session_id, layer_id, tensor)

    def fetch_layer(
        self,
        session_id: str,
        layer_id: int,
    ) -> Optional[bytes]:
        """Fetch a single layer's KV tensor.

        Returns:
            Raw tensor bytes on hit, None on miss.
        """
        if self.mode == "resp":
            return self._fetch_layer_resp(session_id, layer_id)
        else:
            return self._fetch_layer_binary(session_id, layer_id)

    def store_layers_batch(
        self,
        session_id: str,
        layers: Dict[int, bytes],
    ) -> int:
        """Store multiple layers. Returns number successfully stored."""
        stored = 0
        for layer_id, tensor in layers.items():
            if self.store_layer(session_id, layer_id, tensor):
                stored += 1
        return stored

    def fetch_layers_batch(
        self,
        session_id: str,
        layer_ids: List[int],
    ) -> Dict[int, bytes]:
        """Fetch multiple layers. Returns dict of layer_id -> tensor for hits."""
        results = {}
        for layer_id in layer_ids:
            tensor = self.fetch_layer(session_id, layer_id)
            if tensor is not None:
                results[layer_id] = tensor
        return results

    # ── RESP mode ──

    def _store_layer_resp(self, session_id: str, layer_id: int, tensor: bytes) -> bool:
        """Store via KV.STORE with deterministic per-layer embedding."""
        client = self._get_resp_client()
        cache_id = f"layer:{session_id}:{layer_id}"
        embedding = self._get_layer_embedding(session_id, layer_id)
        return client.kv_store(cache_id, embedding, tensor, ttl=3600)

    def _fetch_layer_resp(self, session_id: str, layer_id: int) -> Optional[bytes]:
        """Fetch via KV.FETCH with the same deterministic embedding."""
        client = self._get_resp_client()
        embedding = self._get_layer_embedding(session_id, layer_id)
        return client.kv_fetch(embedding, threshold=0.99)  # High threshold for deterministic embeddings

    # ── Binary mode (placeholder — needs binary listener in Pion) ──

    def _store_layer_binary(self, session_id: str, layer_id: int, tensor: bytes) -> bool:
        """Store via binary protocol. Not yet implemented — falls back to RESP."""
        return self._store_layer_resp(session_id, layer_id, tensor)

    def _fetch_layer_binary(self, session_id: str, layer_id: int) -> Optional[bytes]:
        """Fetch via binary protocol. Not yet implemented — falls back to RESP."""
        return self._fetch_layer_resp(session_id, layer_id)

    def close(self):
        if self._resp_client:
            self._resp_client.close()
            self._resp_client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
