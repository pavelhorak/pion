"""Fast lightweight prompt embedding for semantic KV cache matching.

Supports multiple backends:
  - ngram: Local character n-gram hashing (no deps, ~0.1ms per prompt)
  - openai: OpenAI text-embedding-3-small (production quality, ~50ms per prompt)
  - ollama: Ollama nomic-embed-text (local, ~20ms per prompt)

The n-gram embedder preserves text similarity: prompts sharing code/words
have high cosine similarity. Suitable for cache matching where the prompts
contain large shared code blocks.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Protocol for prompt embedders."""

    @property
    def dim(self) -> int: ...
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


class NGramEmbedder:
    """Local semantic embeddings via character n-gram + word hashing.

    No external dependencies. Preserves text similarity for code-heavy prompts.
    Suitable for cache matching where prompts share large code blocks.
    """

    def __init__(self, dim: int = 1536):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single prompt. Returns L2-normalized float32 array."""
        vec = np.zeros(self._dim, dtype=np.float32)
        text_lower = text.lower()

        # Character 4-grams
        for i in range(len(text_lower) - 3):
            gram = text_lower[i:i + 4]
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 128) & 1 else -1.0
            vec[idx] += sign

        # Word-level features (weighted higher for semantic grouping)
        for word in text_lower.split():
            if len(word) >= 3:
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                idx = h % self._dim
                sign = 1.0 if (h >> 128) & 1 else -1.0
                vec[idx] += sign * 2.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple prompts. Returns (N, dim) float32 array."""
        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            result[i] = self.embed(text)
        return result


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small embedder."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        self._model = model
        self._dim = dim
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        result = np.zeros((len(texts), self._dim), dtype=np.float32)
        for item in resp.data:
            result[item.index] = np.array(item.embedding[:self._dim], dtype=np.float32)
        return result


class OllamaEmbedder:
    """Ollama local embedder (nomic-embed-text)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "127.0.0.1",
        port: int = 11434,
        dim: int = 768,
        pad_to: int = 1536,
    ):
        self._model = model
        self._url = f"http://{host}:{port}/api/embeddings"
        self._native_dim = dim
        self._pad_to = pad_to

    @property
    def dim(self) -> int:
        return self._pad_to

    def embed(self, text: str) -> np.ndarray:
        import requests
        resp = requests.post(
            self._url,
            json={"model": self._model, "prompt": text[:2048]},
            timeout=30,
        )
        resp.raise_for_status()
        floats = resp.json()["embedding"]

        vec = np.zeros(self._pad_to, dtype=np.float32)
        vec[:min(len(floats), self._pad_to)] = floats[:self._pad_to]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self._pad_to), dtype=np.float32)
        for i, text in enumerate(texts):
            result[i] = self.embed(text)
        return result


def create_embedder(provider: str = "auto", **kwargs) -> Embedder:
    """Create an embedder by provider name.

    Args:
        provider: "ngram", "openai", "ollama", or "auto" (detect best available).
        **kwargs: Passed to the embedder constructor.
    """
    if provider == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            try:
                import requests
                requests.get("http://127.0.0.1:11434/api/tags", timeout=1)
                provider = "ollama"
            except Exception:
                provider = "ngram"

    if provider == "ngram":
        return NGramEmbedder(**kwargs)
    elif provider == "openai":
        return OpenAIEmbedder(**kwargs)
    elif provider == "ollama":
        return OllamaEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embed provider: {provider!r}")
