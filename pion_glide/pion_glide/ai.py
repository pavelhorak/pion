"""Typed wrappers for Pion's AI.* gateway commands.

All AI commands require Pion started with ``--flare`` (auto-detects Ollama)
or ``--emb-enabled --llm-enabled``.

Quickstart:
    ./pion-server --flare          # auto-detect Ollama at localhost:11434

    from pion_glide import PionClient
    async with await PionClient.connect() as client:
        # Semantic cache in front of LLM
        answer = await client.ai.complete("What is the capital of France?")

        # Manual cache
        await client.ai.semantic_cache_set("capital of France?", "Paris")
        hit = await client.ai.semantic_cache_get("What's the capital of France?")

        # RAG chat
        await client.ft.add_text("kb", "doc:1", "Pion achieves 10K QPS on Linux")
        response = await client.ai.chat(
            "How fast is Pion?", context_index="kb", context_query="performance"
        )
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .client import PionClient


class AIGateway:
    """Typed helpers for Pion's AI.* commands.

    Obtain an instance from :class:`~pion_glide.PionClient`::

        client.ai.complete(...)
        client.ai.semantic_cache_get(...)
    """

    def __init__(self, client: "PionClient") -> None:
        self._client = client

    async def complete(
        self,
        prompt: str,
        tokens: int = 200,
        threshold: float = 0.92,
    ) -> str:
        """Semantic cache check + LLM generation in one command.

        Sends ``AI.COMPLETE prompt TOKENS tokens THRESHOLD threshold``.

        On a semantic cache **hit** (cosine similarity ≥ threshold) returns the
        cached response instantly.  On a **miss** calls the LLM, stores the
        result in the cache, and returns the generated text.

        Parameters
        ----------
        prompt:     The user query or instruction.
        tokens:     Maximum tokens for LLM generation.
        threshold:  Cosine similarity threshold for cache lookup (0–1).
                    Higher = stricter match required for a cache hit.
        """
        raw = await self._client.execute(
            "AI.COMPLETE", prompt,
            "TOKENS", str(tokens),
            "THRESHOLD", str(threshold),
        )
        return _decode(raw)

    async def semantic_cache_set(self, query: str, response: str) -> str:
        """Store a query→response pair in the semantic cache.

        Sends ``AI.SEMANTIC_CACHE SET query response``.
        The query is embedded server-side and inserted into the per-worker HNSW cache.
        """
        raw = await self._client.execute("AI.SEMANTIC_CACHE", "SET", query, response)
        return _decode(raw)

    async def semantic_cache_get(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> Optional[str]:
        """Look up a semantically similar cached response.

        Sends ``AI.SEMANTIC_CACHE GET query [THRESHOLD t]``.

        Parameters
        ----------
        query:      The query to look up.
        threshold:  Optional cosine threshold override (overrides server default).

        Returns
        -------
        Cached response string on hit, ``None`` on miss.
        """
        args = ["AI.SEMANTIC_CACHE", "GET", query]
        if threshold is not None:
            args += ["THRESHOLD", str(threshold)]
        raw = await self._client.execute(*args)
        result = _decode(raw)
        return result if result else None

    async def chat(
        self,
        prompt: str,
        context_index: Optional[str] = None,
        context_query: Optional[str] = None,
        k: int = 3,
    ) -> str:
        """RAG chat: retrieve context vectors → augment prompt → LLM response.

        Sends ``AI.CHAT prompt [CONTEXT index query K k]``.

        Parameters
        ----------
        prompt:           User question or instruction.
        context_index:    FT index to retrieve context from.
        context_query:    Query used for context retrieval (often same as prompt).
        k:                Number of context chunks to inject.
        """
        args: list = ["AI.CHAT", prompt]
        if context_index and context_query:
            args += ["CONTEXT", context_index, context_query, "K", str(k)]
        raw = await self._client.execute(*args)
        return _decode(raw)

    async def flare_run(
        self,
        index_name: str,
        prompt: str,
        max_tokens: int = 200,
    ) -> str:
        """Mid-generation retrieval (FLARE): generate + retrieve + regenerate.

        Sends ``AI.FLARE RUN index_name prompt TOKENS max_tokens``.
        Requires ``--flare`` server flag.
        """
        raw = await self._client.execute(
            "AI.FLARE", "RUN", index_name, prompt, "TOKENS", str(max_tokens)
        )
        return _decode(raw)


def _decode(raw: Any) -> str:
    """Decode a GLIDE response value to str."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)
