"""Embedding provider abstraction.

Supports:
  - OpenAI API (default, requires OPENAI_API_KEY)
  - Ollama (PION_EMBED_PROVIDER=ollama): the same vectors as pion-context's
    codebase index, which defaults to it
  - MAX Serve HTTP (set PION_EMBED_URL=http://localhost:8000/v1)
  - Local numpy random (PION_EMBED_PROVIDER=mock — for testing only)
"""
from __future__ import annotations

import os
import struct
from typing import Sequence


def _provider() -> str:
    return os.environ.get("PION_EMBED_PROVIDER", "openai").lower()


def _model() -> str:
    default = "nomic-embed-text" if _provider() == "ollama" else "text-embedding-3-small"
    return os.environ.get("PION_EMBED_MODEL", default)


def _dim() -> int:
    return int(os.environ.get("PION_EMBED_DIM", "1536"))


def embed_text(text: str) -> bytes:
    """Return raw float32 bytes (little-endian) for `text`."""
    provider = _provider()

    if provider == "openai":
        return _embed_openai(text)
    elif provider == "ollama":
        return _embed_ollama_batch([text])[0]
    elif provider == "max":
        return _embed_max_serve(text)
    elif provider == "mock":
        return _embed_mock(text)
    else:
        raise ValueError(f"Unknown PION_EMBED_PROVIDER={provider!r}. Use: openai, ollama, max, mock")


def embed_texts(texts: Sequence[str]) -> list[bytes]:
    """Batch embed. OpenAI supports native batching; others fall back to loop."""
    provider = _provider()
    if provider == "openai":
        return _embed_openai_batch(texts)
    if provider == "ollama":
        return _embed_ollama_batch(texts)
    return [embed_text(t) for t in texts]


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _embed_openai(text: str) -> bytes:
    return _embed_openai_batch([text])[0]


def _embed_openai_batch(texts: Sequence[str]) -> list[bytes]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  OR set PION_EMBED_PROVIDER=max|mock")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. "
            "Set it, or use PION_EMBED_PROVIDER=mock for testing."
        )

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=_model(), input=list(texts))
    return [_floats_to_bytes(item.embedding) for item in resp.data]


# ── Ollama ─────────────────────────────────────────────────────────────────────

def _embed_ollama_batch(texts: Sequence[str]) -> list[bytes]:
    """pion-context's Ollama embedding when it is installed, so agent memory and
    the codebase index embed text identically; otherwise the same steps here:
    text cut at 2,048 characters, /api/embed, zero-padded to PION_EMBED_DIM,
    L2-normalized."""
    try:
        from pion_context.embeddings import embed_texts as context_embed_texts
    except ImportError:
        context_embed_texts = None
    if context_embed_texts is not None:
        return context_embed_texts(texts)
    import math
    import requests
    url = os.environ.get("PION_OLLAMA_URL", "http://127.0.0.1:11434")
    out = []
    for i in range(0, len(texts), 64):
        resp = requests.post(f"{url}/api/embed", timeout=300,
                             json={"model": _model(), "input": [t[:2048] for t in texts[i:i + 64]]})
        resp.raise_for_status()
        for v in resp.json()["embeddings"]:
            v = list(v)[: _dim()] + [0.0] * max(0, _dim() - len(v))
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append(_floats_to_bytes([x / n for x in v]))
    return out


# ── MAX Serve HTTP ─────────────────────────────────────────────────────────────

def _embed_max_serve(text: str) -> bytes:
    """Call a running `max serve` instance (OpenAI-compatible endpoint)."""
    import httpx

    url = os.environ.get("PION_EMBED_URL", "http://localhost:8000/v1")
    resp = httpx.post(
        f"{url}/embeddings",
        json={"model": _model(), "input": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return _floats_to_bytes(data["data"][0]["embedding"])


# ── Mock (testing) ─────────────────────────────────────────────────────────────

def _embed_mock(text: str) -> bytes:
    """Deterministic mock: hash text to a reproducible float32 vector."""
    import hashlib
    import math

    import random

    # Seeded Gaussian per dimension: distinct texts get distinct, near-orthogonal
    # vectors. The old `sin(h + i * 1.618)` added a small float to a 256-bit
    # hash — as a float, h + i*1.618 == h — so every component was identical
    # and every text embedded to the same constant vector.
    rng = random.Random(int(hashlib.sha256(text.encode()).hexdigest(), 16))
    floats = [rng.gauss(0.0, 1.0) for _ in range(_dim())]
    norm = math.sqrt(sum(v * v for v in floats)) or 1.0
    floats = [v / norm for v in floats]
    return _floats_to_bytes(floats)


# ── helpers ────────────────────────────────────────────────────────────────────

def _floats_to_bytes(floats: list[float]) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)
