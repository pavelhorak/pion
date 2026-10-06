"""Embedding providers — shared with pion_mcp.

Supports OpenAI, Ollama (local), and mock (testing).
Defaults to Ollama for local-first usage (no API key required).
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
from typing import Sequence


def _provider() -> str:
    return os.environ.get("PION_EMBED_PROVIDER", "ollama").lower()


def _model() -> str:
    defaults = {
        "ollama": "nomic-embed-text",
        "openai": "text-embedding-3-small",
    }
    return os.environ.get("PION_EMBED_MODEL", defaults.get(_provider(), "nomic-embed-text"))


def _dim() -> int:
    # Must match Pion's server-side Vector Dim (default 1536).
    # Ollama nomic-embed-text outputs 768d — we pad to 1536 with zeros.
    return int(os.environ.get("PION_EMBED_DIM", "1536"))


def _ollama_url() -> str:
    return os.environ.get("PION_OLLAMA_URL", "http://127.0.0.1:11434")


def embed_dim() -> int:
    """Return the configured embedding dimension."""
    return _dim()


def embed_text(text: str) -> bytes:
    """Return raw float32 bytes (little-endian) for text."""
    provider = _provider()
    if provider == "ollama":
        return _embed_ollama(text)
    elif provider == "openai":
        return _embed_openai(text)
    elif provider == "mock":
        return _embed_mock(text)
    else:
        raise ValueError(f"Unknown PION_EMBED_PROVIDER={provider!r}. Use: ollama, openai, mock")


def embed_texts(texts: Sequence[str]) -> list[bytes]:
    """Batch embed."""
    provider = _provider()
    if provider == "openai":
        return _embed_openai_batch(texts)
    if provider == "ollama":
        return _embed_ollama_batch(texts)
    return [embed_text(t) for t in texts]


def embed_to_floats(text: str) -> list[float]:
    """Embed text and return as list of floats."""
    raw = embed_text(text)
    n = len(raw) // 4
    return list(struct.unpack(f"{n}f", raw))


# ── Ollama (default, local) ──────────────────────────────────────────────────

def _finish_ollama(floats: Sequence[float]) -> bytes:
    """Pad to the index dimension (e.g. nomic's 768 -> 1536), L2-normalize, pack."""
    target_dim = _dim()
    floats = list(floats)
    if len(floats) < target_dim:
        floats.extend([0.0] * (target_dim - len(floats)))
    elif len(floats) > target_dim:
        floats = floats[:target_dim]
    norm = math.sqrt(sum(v * v for v in floats)) or 1.0
    return _floats_to_bytes([v / norm for v in floats])


def _embed_ollama(text: str) -> bytes:
    import time
    import requests
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{_ollama_url()}/api/embeddings",
                json={"model": _model(), "prompt": text[:2048]},
                timeout=30,
            )
            resp.raise_for_status()
            return _finish_ollama(resp.json()["embedding"])
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
            else:
                raise


def _embed_ollama_batch(texts: Sequence[str], batch: int = 64) -> list[bytes]:
    """One /api/embed request per `batch` texts instead of one request per
    text. The vectors are the ones _embed_ollama returns: both paths cut the
    text at 2,048 characters, and normalization makes /api/embed's
    unit-length output and /api/embeddings' raw output the same vector."""
    import time
    import requests
    out: list[bytes] = []
    for i in range(0, len(texts), batch):
        part = [t[:2048] for t in texts[i:i + batch]]
        for attempt in range(3):
            try:
                resp = requests.post(f"{_ollama_url()}/api/embed",
                                     json={"model": _model(), "input": part}, timeout=300)
                if resp.status_code == 404:          # an Ollama without /api/embed
                    return [_embed_ollama(t) for t in texts]
                resp.raise_for_status()
                vecs = resp.json()["embeddings"]
                if len(vecs) != len(part):
                    raise ValueError(f"/api/embed returned {len(vecs)} vectors for {len(part)} inputs")
                out += [_finish_ollama(v) for v in vecs]
                break
            except Exception:
                if attempt < 2:
                    time.sleep(0.5)
                else:
                    raise
    return out


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _embed_openai(text: str) -> bytes:
    return _embed_openai_batch([text])[0]


def _embed_openai_batch(texts: Sequence[str]) -> list[bytes]:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=_model(), input=list(texts))
    return [_floats_to_bytes(item.embedding) for item in resp.data]


# ── Mock (testing) ────────────────────────────────────────────────────────────

def _embed_mock(text: str) -> bytes:
    # Seeded Gaussian per dimension. The old sin(h + i*1.618) added a small float to a
    # 256-bit hash — as a float h + i*1.618 == h — so every text embedded to the same
    # constant vector and no search over mock embeddings could tell results apart.
    import random
    rng = random.Random(int(hashlib.sha256(text.encode()).hexdigest(), 16))
    floats = [rng.gauss(0.0, 1.0) for _ in range(_dim())]
    norm = math.sqrt(sum(v * v for v in floats)) or 1.0
    floats = [v / norm for v in floats]
    return _floats_to_bytes(floats)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _floats_to_bytes(floats: list[float]) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)
