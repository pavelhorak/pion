#!/usr/bin/env python3
"""Pion Serve — inference intelligence layer.

OpenAI-compatible proxy that adds semantic caching, RAG, and agent memory
to any LLM backend (ollama, vLLM, OpenAI, etc.).

Architecture:
  Client → Pion Serve (:8080) → Backend (ollama/vLLM/OpenAI)
                │
                ├── Semantic cache: cosine match on query embedding
                ├── RAG injection: FT.SEARCH vector index → prepend context
                ├── Agent memory: recall prior conversations
                └── Response caching: store for future hits

Usage:
    # With Ollama:
    python pion-serve/serve.py --backend ollama

    # With RAG from docs directory:
    python pion-serve/serve.py --backend ollama --rag-dir ./docs

    # With custom model:
    python pion-serve/serve.py --backend ollama --model gemma3:4b

    # Disable caching (pure proxy):
    python pion-serve/serve.py --backend ollama --no-cache

Requires:
    - Pion server: ./pion-server --kvcache -w 1 (or --profile ai)
    - Backend: ollama running with a model loaded
    - pip install flask requests redis numpy

Deployment / threading model (gh #84):

    Flask's default `app.run` is multi-threaded (`threaded=True`). Every
    mutable module global mutated from a request handler is now guarded:

      _stats              → `_stats_lock` (read via `_stats_snapshot`,
                            written via `_incr_stat`).
      _last_l3_response   → `_l3_response_lock` (set + evict are atomic).
      _embed_fn (init)    → no lock needed; `_init_embedder` is called from
                            `main()` before `app.run()`.
      _pion_moe_*  (init) → `@_once_under_init_lock` (idempotent under
                            concurrent first-requests).
      _cag_*       (init) → `@_once_under_init_lock` (same).

    For a stricter "serialize requests" deployment, run under gunicorn-sync:
        gunicorn -w 1 -k sync serve:app
    The locks are still correct under sync workers (no-op fast path), and
    a single gunicorn-sync worker eliminates intra-process concurrency
    entirely if your workload needs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import string
import sys
import threading
import time
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import redis
import requests
from flask import Flask, Response, jsonify, request, stream_with_context

from concept_store import ConceptStore, FragmentStore
from intent_router import (
    DEFAULT_ROUTING,
    IntentRouter,
    RouteDecision,
    load_routing_config,
)

# ── Configuration ────────────────────────────────────────────────────────────

BACKENDS = {
    "ollama": {"url": "http://127.0.0.1:11434", "chat_path": "/api/chat", "models_path": "/api/tags"},
    "vllm": {"url": "http://127.0.0.1:8000", "chat_path": "/v1/chat/completions", "models_path": "/v1/models"},
    "openai": {"url": "https://api.openai.com", "chat_path": "/v1/chat/completions", "models_path": "/v1/models"},
    "gemini": {"url": "https://generativelanguage.googleapis.com", "chat_path": "/v1beta/openai/chat/completions", "models_path": "/v1beta/openai/models"},
    "claude": {"url": "https://api.anthropic.com", "chat_path": "/v1/messages", "models_path": ""},
    "llamacpp": {"url": "http://127.0.0.1:8080", "chat_path": "/v1/chat/completions", "models_path": "/v1/models"},
    # max: Modular MAX Serve. OpenAI-compatible endpoint (`max serve` defaults
    # to port 8000), so it flows through the generic OpenAI-compatible forward
    # path — no special-casing in _forward_to_backend / _forward_stream. This is
    # the request-layer Pion×MAX surface (semantic cache / routing), NOT the KV
    # datapath: MAX 26.4 removed the LMCache connector and its native KVConnector
    # factory is closed (see gh #95). Local MAX needs no auth; if OPENAI_API_KEY
    # is unset no Authorization header is sent, which is correct.
    "max": {"url": "http://127.0.0.1:8000", "chat_path": "/v1/chat/completions", "models_path": "/v1/models"},
    # pion-moe: in-process MoE inference via Pion's MOE.EXPERT.* substrate.
    # The MLX model shell loads in this process; expert weights are fetched
    # on demand from a running pion-server's --moe-cache tier. URL is unused
    # (in-process); the host/port pair comes from --pion-moe-host/--pion-moe-port.
    "pion-moe": {"url": "", "chat_path": "", "models_path": ""},
    # pion-cag-hybrid: in-process CAG+RAG entropy/margin-gated hybrid
    # (gh #23 productization). Preloads a bounded foundation corpus into a
    # MLX prompt cache at startup, calibrates (Te, Tm) on a held-out QA
    # subset, then serves with the multi-signal OR-gate. URL unused.
    "pion-cag-hybrid": {"url": "", "chat_path": "", "models_path": ""},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pion-serve")

app = Flask(__name__)

# gh #84: locks guarding the module-level mutable globals. Flask's default
# server runs handlers concurrently (threaded=True on `app.run`), so every
# read-modify-write of `_stats` / `_last_l3_response` and every lazy-init
# check-then-act on `_embed_fn`, `_pion_moe_*`, `_cag_*` is a race without
# these. RLock so a single thread can re-enter (e.g. lazy-init that itself
# bumps a counter).
#
# Critical sections are kept small — guard only the mutation, not the
# surrounding logic. For one-shot stats reads (`/v1/stats`) we copy the dict
# under the lock and serialise the snapshot outside.
_stats_lock = threading.RLock()
_l3_response_lock = threading.RLock()
_init_lock = threading.RLock()


def _incr_stat(key: str, n: int = 1) -> None:
    """Thread-safe `_stats[key] += n`. Single helper so every counter site is
    serialised under `_stats_lock` without per-call boilerplate."""
    with _stats_lock:
        _stats[key] = _stats.get(key, 0) + n


def _stats_snapshot() -> dict:
    """Return a copy of `_stats` for read-only consumers (e.g. `/v1/stats`)."""
    with _stats_lock:
        return dict(_stats)


def _once_under_init_lock(fn):
    """gh #84: decorator that wraps a lazy-init function so its body runs at
    most once, under `_init_lock`, even when called concurrently from multiple
    request handlers. Used for `_init_pion_moe_lazy` / `_init_pion_cag_lazy` /
    `_init_embedder` where the init does I/O (MLX model load, MoE substrate
    install, sklearn vectoriser fit) and is currently called from the request
    path of the first prompt.

    Semantics: first call runs the body; if it returns normally we mark
    `done` and every later call short-circuits. If the body RAISES the flag
    stays cleared so the next request retries from scratch — matches the
    existing "errors propagate" behaviour, no permanent dead state."""
    done = threading.Event()
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if done.is_set():
            return
        with _init_lock:
            if done.is_set():
                return
            result = fn(*args, **kwargs)
            done.set()
            return result

    return wrapper


# Global state (set in main)
_config: dict = {}
_pion: Optional[redis.Redis] = None
_embed_fn = None
_concept_store: Optional[ConceptStore] = None
_fragment_store: Optional[FragmentStore] = None
_intent_router: Optional[IntentRouter] = None
# pion-moe backend lazy-init state (only populated when --backend pion-moe is used)
_pion_moe_model = None
_pion_moe_tokenizer = None
_pion_moe_tier = None
# pion-cag-hybrid backend lazy-init state (only populated when
# --backend pion-cag-hybrid is used). The model + tokenizer are MLX (Mac);
# _cag_cache is the preloaded foundation prompt cache; _cag_paragraphs is
# the corpus split for TF-IDF RAG fallback; _cag_retriever is the fitted
# TfidfRetriever; _cag_te / _cag_tm are the calibrated OR-gate thresholds.
_cag_model = None
_cag_tokenizer = None
_cag_cache = None
_cag_paragraphs: list = []
_cag_retriever = None
_cag_te: Optional[float] = None
_cag_tm: Optional[float] = None
# Cached newline token IDs for the loaded tokenizer; populated on first warm
# forward. Lets _cag_warm_forward break the gen loop the moment the model
# emits "\n" instead of running the full n_gen and post-stripping.
_cag_newline_set: Optional[set] = None
# Rich calibration result for the `setup` subcommand and operator card —
# populated by `_cag_calibrate` on its return. Schema below.
_cag_last_calibration_info: Optional[dict] = None
_stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "l3_synthesis": 0,
    "l3_direct": 0,
    "l3_composite": 0,
    "l3_fragment": 0,
    "full_inference": 0,
    "rag_injections": 0,
    "tokens_saved": 0,
    "total_latency_ms": 0,
    "route_simple": 0,
    "route_medium": 0,
    "route_complex": 0,
    "route_estimated_cost_usd": 0.0,
    # gh #69: per-backend embed call counters. Each backend bumps its own
    # counter on a successful (non-None) embedding so /v1/stats can
    # distinguish SIE hits from Ollama / Pion / HTTP fallback hits.
    "embed_calls_sie": 0,
    "embed_calls_ollama": 0,
    "embed_calls_pion": 0,
    # pion-cag-hybrid gate decision counters
    "cag_kept": 0,
    "cag_fallback": 0,
}


# ── Embedding ────────────────────────────────────────────────────────────────

def _init_embedder(provider: str = "auto"):
    """Initialize embedding function. Cascade order (when provider='auto'):
      1. Pion auto-embed sidecar (MiniLM-L6-v2, 384-dim — default)
      2. SIE (superlinked/sie) — opt-in via --sie-url
      3. Ollama nomic-embed-text on localhost:11434
      4. hash-only fallback (no semantic match; exact-string cache only)

    Explicit providers ('pion'|'sie'|'ollama'|'none') do not fall through —
    if the named backend isn't reachable, the embedder is left disabled and
    a warning is logged. This is the strict mode the gh #69 acceptance
    criteria asks for ("--embed-backend sie --sie-url ... produces correct
    embeddings on a sample query").
    """
    global _embed_fn

    if provider == "none":
        log.info("Embedder explicitly disabled (--embed-backend none)")
        _embed_fn = None
        _config["embed_backend"] = "none"
        return

    if provider == "pion" or provider == "auto":
        # Use Pion's auto-embed sidecar (MiniLM-L6-v2, 384-dim)
        try:
            r = _pion
            r.execute_command("AI.SEMANTIC_CACHE", "GET", "test_embed_init")
            log.info("Using Pion auto-embed (384-dim MiniLM-L6-v2)")
            _embed_fn = _embed_via_pion
            _config["embed_backend"] = "pion"
            return
        except Exception:
            if provider == "pion":
                log.warning("Pion auto-embed not available — embedder disabled")
                _embed_fn = None
                _config["embed_backend"] = "none"
                return

    # gh #69: SIE (superlinked/sie) — opt-in via --sie-url. Slots ahead of
    # Ollama because SIE is the production-grade option (85 MTEB-verified
    # models, autoscale, OpenAI-compat). We never auto-detect on port 8000
    # since it's a common collision target; explicit URL only.
    sie_url = _config.get("sie_url")
    if provider == "sie" or (provider == "auto" and sie_url):
        if not sie_url:
            log.warning("--embed-backend sie requires --sie-url; embedder disabled")
            _embed_fn = None
            _config["embed_backend"] = "none"
            return
        try:
            resp = requests.post(
                f"{sie_url}/v1/embeddings",
                json={"model": _config.get("sie_model", "BAAI/bge-small-en-v1.5"),
                      "input": "test"},
                timeout=5,
            )
            if resp.ok:
                data = resp.json().get("data") or []
                if data and "embedding" in data[0]:
                    dim = len(data[0]["embedding"])
                    log.info(f"Using SIE at {sie_url} ({_config.get('sie_model')}, {dim}-dim)")
                    _embed_fn = _embed_via_sie
                    _config["embed_backend"] = "sie"
                    return
                else:
                    log.warning(f"SIE at {sie_url} returned no embedding for probe; falling back")
            else:
                log.warning(f"SIE at {sie_url} probe HTTP {resp.status_code}; falling back")
        except Exception as e:
            log.warning(f"SIE at {sie_url} not reachable ({e}); falling back")
        if provider == "sie":
            _embed_fn = None
            _config["embed_backend"] = "none"
            return

    if provider == "ollama" or provider == "auto":
        try:
            resp = requests.post(
                f"{_config.get('ollama_url', 'http://127.0.0.1:11434')}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": "test"},
                timeout=5,
            )
            if resp.ok:
                dim = len(resp.json().get("embedding", []))
                log.info(f"Using Ollama nomic-embed-text ({dim}-dim)")
                _embed_fn = _embed_via_ollama
                _config["embed_backend"] = "ollama"
                return
        except Exception:
            pass
        if provider == "ollama":
            log.warning("Ollama nomic-embed-text not available — embedder disabled")
            _embed_fn = None
            _config["embed_backend"] = "none"
            return

    # Fallback: hash-based pseudo-embedding (no semantic matching, exact only)
    log.warning("No embedding provider available — semantic cache disabled, using hash-only")
    _embed_fn = None
    _config["embed_backend"] = "none"


def _embed_via_sie(text: str) -> Optional[np.ndarray]:
    """gh #69: embed via SIE (superlinked/sie) /v1/embeddings (OpenAI-compat)."""
    try:
        resp = requests.post(
            f"{_config.get('sie_url')}/v1/embeddings",
            json={"model": _config.get("sie_model", "BAAI/bge-small-en-v1.5"),
                  "input": text},
            timeout=10,
        )
        if resp.ok:
            data = resp.json().get("data") or []
            if data and "embedding" in data[0]:
                vec = np.array(data[0]["embedding"], dtype=np.float32)
                vec /= np.linalg.norm(vec) + 1e-10
                _incr_stat("embed_calls_sie")
                return vec
    except Exception as e:
        log.debug(f"SIE embedding failed: {e}")
    return None


def _embed_via_pion(text: str) -> Optional[np.ndarray]:
    """Embed via Pion's semantic cache (piggyback on auto-embed sidecar)."""
    # Use AI.SEMANTIC_CACHE GET with a very low threshold to force embedding
    # This is a hack — proper embedding endpoint would be better
    # For now, use ollama if available
    vec = _embed_via_ollama(text)
    if vec is not None:
        # _embed_via_ollama already counted ollama; reattribute to pion so the
        # active-backend counter matches /v1/stats' embed_backend field.
        _incr_stat("embed_calls_ollama", -1)
        _incr_stat("embed_calls_pion")
    return vec


def _batch_embed_via_ollama(texts: list[str]) -> list[Optional[np.ndarray]]:
    """Batch embed via sequential Ollama calls (Ollama doesn't support true batch).

    Still faster than per-fragment calls from FragmentStore because we avoid
    the overhead of the embed_fn callback layer and can pipeline requests.
    """
    results = []
    for text in texts:
        results.append(_embed_via_ollama(text))
    return results


def _embed_via_ollama(text: str) -> Optional[np.ndarray]:
    """Embed via Ollama's nomic-embed-text."""
    try:
        resp = requests.post(
            f"{_config.get('ollama_url', 'http://127.0.0.1:11434')}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=10,
        )
        if resp.ok:
            vec = np.array(resp.json()["embedding"], dtype=np.float32)
            vec /= np.linalg.norm(vec) + 1e-10
            _incr_stat("embed_calls_ollama")
            return vec
    except Exception as e:
        log.debug(f"Embedding failed: {e}")
    return None


# ── Semantic Cache ───────────────────────────────────────────────────────────

def _cache_check(query: str, threshold: float = 0.85) -> Optional[str]:
    """Check semantic cache for similar query. Returns cached response or None."""
    if not _pion or not _embed_fn:
        return None

    try:
        result = _pion.execute_command(
            "AI.SEMANTIC_CACHE", "GET", query, "THRESHOLD", str(threshold)
        )
        if result and result != b"$-1\r\n" and result != b"(nil)":
            if isinstance(result, bytes):
                return result.decode("utf-8", errors="replace")
            return str(result)
    except Exception as e:
        log.debug(f"Cache check failed: {e}")
    return None


def _cache_store(query: str, response: str):
    """Store query/response pair in semantic cache."""
    if not _pion:
        return
    try:
        _pion.execute_command("AI.SEMANTIC_CACHE", "SET", query, response)
    except Exception as e:
        log.debug(f"Cache store failed: {e}")


# ── RAG Context Injection ───────────────────────────────────────────────────

def _rag_retrieve(query: str, k: int = 3) -> list[str]:
    """Retrieve relevant documents from Pion's vector index."""
    if not _pion or not _embed_fn or not _config.get("rag_index"):
        return []

    vec = _embed_fn(query)
    if vec is None:
        return []

    try:
        index_name = _config["rag_index"]
        vec_blob = vec.tobytes()
        # Standard KNN form. `FT.SEARCH <idx> <blob> KNN k` is not a vector
        # query to Pion — it answered [] — so RAG retrieved nothing. Pion
        # matches the index's own vector field whatever @name says here.
        result = _pion.execute_command(
            "FT.SEARCH", index_name, f"*=>[KNN {k} @vector $vec AS score]",
            "PARAMS", "2", "vec", vec_blob, "DIALECT", "2",
        )
        # Reply: [count, key, fields, key, fields, ...]. The fields carry only
        # id/score, never the document text, so read it from the hash.
        if isinstance(result, list) and len(result) > 1:
            docs = []
            for key in result[1::2]:
                content, text = _pion.hmget(key, "content", "text")
                body = content or text
                if body:
                    docs.append(body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body))
            return docs
    except Exception as e:
        log.debug(f"RAG retrieval failed: {e}")
    return []


def _inject_rag_context(messages: list[dict], context_docs: list[str]) -> list[dict]:
    """Prepend RAG context to the system message."""
    if not context_docs:
        return messages

    context_text = "\n\n---\n\n".join(context_docs)
    rag_prefix = (
        "Use the following context to answer the question. "
        "If the context doesn't contain relevant information, say so.\n\n"
        f"Context:\n{context_text}\n\n---\n\n"
    )

    # Find or create system message
    new_messages = []
    has_system = False
    for msg in messages:
        if msg.get("role") == "system":
            new_messages.append({
                "role": "system",
                "content": rag_prefix + msg["content"],
            })
            has_system = True
        else:
            new_messages.append(msg)

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": rag_prefix + "Answer concisely."})

    return new_messages


# ── Backend Forwarding ───────────────────────────────────────────────────────

def _resolve_backend(body: dict, route: Optional[RouteDecision]) -> tuple[str, str, str]:
    """Resolve (backend_type, backend_url, model) from intent route or config."""
    if route is not None:
        backend_type = route.backend
        backend_url = route.base_url or BACKENDS.get(backend_type, {}).get(
            "url", _config.get("backend_url", "http://127.0.0.1:11434")
        )
        model = route.model
    else:
        backend_type = _config.get("backend_type", "ollama")
        backend_url = _config.get("backend_url", "http://127.0.0.1:11434")
        model = body.get("model", _config.get("model", "llama3.2:3b"))
    return backend_type, backend_url, model


@_once_under_init_lock
def _init_pion_moe_lazy():
    """Lazy init for the pion-moe in-process backend. Loads the MLX model
    shell, opens a wire-backed MoEExpertTierClient against the running
    pion-server, and installs the substrate intercept on the model's MoE
    layers. Idempotent under `@_once_under_init_lock` (gh #84) so two
    concurrent first-requests don't both load the MLX shell."""
    global _pion_moe_model, _pion_moe_tokenizer, _pion_moe_tier
    import sys
    # Both modules ship in examples/. They used to be imported from
    # a research directory that does not ship, so this
    # backend raised ImportError for anyone outside the development repo, and
    # because the init is lazy it did so only once someone selected it.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
    from pion_moe_tier import install_moe_substrate
    from pion_moe_tier_client import PionMoEExpertTierClient
    from mlx_lm import load
    import json as _json
    from pathlib import Path
    model_path = _config.get("pion_moe_model_path")
    model_id = _config.get("pion_moe_model_id")
    host = _config.get("pion_moe_host", "127.0.0.1")
    port = int(_config.get("pion_moe_port", 1974))
    if not model_path or not model_id:
        raise RuntimeError("pion-moe backend requires --pion-moe-model-path + --pion-moe-model-id")
    log.info(f"pion-moe init: loading MLX shell from {model_path}")
    model, tokenizer = load(str(model_path), lazy=True)
    cfg = _json.loads((Path(model_path) / "config.json").read_text())
    name = (cfg.get("model_type") or cfg.get("architectures", ["unknown"])[0]).lower()
    arch = "gemma4_moe" if "gemma" in name else ("phi35_moe" if "phi" in name else "olmoe")
    tier = PionMoEExpertTierClient(model_id=model_id, host=host, port=port,
                                    architecture=arch)
    n_moe = install_moe_substrate(model, tier)
    log.info(f"pion-moe init: substrate installed on {n_moe} MoE layers (arch={arch})")
    _pion_moe_model = model
    _pion_moe_tokenizer = tokenizer
    _pion_moe_tier = tier


def _call_pion_moe(messages: list[dict], body: dict) -> dict:
    """Generate via the in-process MLX model + Pion MoE substrate."""
    import mlx.core as mx
    _init_pion_moe_lazy()
    max_tokens = int(body.get("max_tokens", 64))
    # Apply chat template if available; fall back to last user message
    try:
        formatted = _pion_moe_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = messages[-1].get("content", "")
    ids = _pion_moe_tokenizer.encode(formatted)
    x = mx.array(ids)[None]
    eos = _pion_moe_tokenizer.eos_token_id
    decoded: list[int] = []
    for _ in range(max_tokens):
        out = _pion_moe_model(x); mx.eval(out)
        tok = int(mx.argmax(out[0, -1]).item())
        decoded.append(tok)
        x = mx.concatenate([x, mx.array([[tok]])], axis=1)
        if eos is not None and tok == eos:
            break
    text = _pion_moe_tokenizer.decode(decoded)
    return {
        "content": text,
        "model": _config.get("pion_moe_model_id", "pion-moe"),
        "prompt_tokens": len(ids),
        "completion_tokens": len(decoded),
    }


# ── pion-cag-hybrid backend (gh #23 productization) ────────────────────────

def _cag_entropy_full(logits) -> float:
    """Full-softmax entropy in nats (matches run_8b_multisig.py)."""
    import mlx.core as mx
    lp = logits - mx.logsumexp(logits)
    p = mx.softmax(logits)
    return float(-mx.sum(p * lp).item())


def _cag_margin_full(logits) -> float:
    """log p_top1 - log p_top2 over the full softmax. mx.topk returns ascending,
    so top2 = [#2-largest, #1-largest]; margin = top2[1] - top2[0]."""
    import mlx.core as mx
    lp = logits - mx.logsumexp(logits)
    top2 = mx.topk(lp, 2)
    return float((top2[1] - top2[0]).item())


def _cag_clean_answer(text: str) -> str:
    """Truncate at first newline or question-marker, strip whitespace."""
    text = text.strip()
    for sep in ("\n", "Question:", "Q:"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip()


def _cag_squad_f1(pred: str, golds: list[str]) -> float:
    """Token-level SQuAD F1, max over golds. Used during calibration only."""
    import re as _re, string as _string
    def norm(s):
        s = s.lower()
        s = "".join(c for c in s if c not in _string.punctuation)
        s = _re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())
    if not golds:
        return 0.0
    p_toks = norm(pred).split()
    best = 0.0
    for g in golds:
        g_toks = norm(g).split()
        if not p_toks or not g_toks:
            best = max(best, float(p_toks == g_toks))
            continue
        common = {}
        for t in p_toks:
            common[t] = min(p_toks.count(t), g_toks.count(t))
        shared = sum(common.values())
        if shared == 0:
            continue
        prec = shared / len(p_toks); rec = shared / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def _cag_found(pred: str, golds: list[str]) -> float:
    """Substring answer-found check after normalization."""
    import re as _re, string as _string
    def norm(s):
        s = s.lower()
        s = "".join(c for c in s if c not in _string.punctuation)
        s = _re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())
    p = norm(pred)
    for g in golds:
        gn = norm(g)
        if gn and gn in p:
            return 1.0
    return 0.0


def _cag_warm_forward(question: str, n_gen: int = 20) -> tuple[str, float, float, float]:
    """Run the CAG branch against the preloaded foundation cache.
    Returns (answer, first_token_entropy, first_token_margin, ttft_ms) where
    ttft_ms is **streaming TTFT only** — time from call entry through the
    suffix prefill (the model can stream its first token immediately after).
    Generation latency is excluded so the calibration speedup ratio matches
    the spike's published metric (`run_8b_multisig.warm_forward_multisig`).

    Restores the cache via trim_prompt_cache so it's reusable.

    Mirrors run_8b_multisig.warm_forward_multisig: stops generation at the
    first newline since `_cag_clean_answer` truncates there anyway, which
    keeps the trim/append accounting tight and avoids 17+ wasted forwards
    per call on confident short answers.
    """
    import mlx.core as mx
    import time as _time
    from mlx_lm.models.cache import trim_prompt_cache
    global _cag_newline_set
    if _cag_newline_set is None:
        nl_ids = _cag_tokenizer.encode("\n", add_special_tokens=False)
        _cag_newline_set = set(nl_ids) if nl_ids else set()
    suffix = f"\n\nQuestion: {question}\nShort answer:"
    suffix_ids = _cag_tokenizer.encode(suffix, add_special_tokens=False)
    x = mx.array([suffix_ids])
    _t0 = _time.perf_counter()
    out = _cag_model(x, cache=_cag_cache); mx.eval(out)
    ttft_ms = (_time.perf_counter() - _t0) * 1000
    first_logits = out[0, -1]
    ent = _cag_entropy_full(first_logits)
    marg = _cag_margin_full(first_logits)
    tok_id = int(mx.argmax(first_logits).item())
    decoded = [tok_id]
    consumed = len(suffix_ids) + 1
    eos = _cag_tokenizer.eos_token_id
    for _ in range(n_gen - 1):
        nxt = mx.array([[tok_id]])
        out = _cag_model(nxt, cache=_cag_cache); mx.eval(out)
        nxt_id = int(mx.argmax(out[0, -1]).item())
        if eos is not None and nxt_id == eos:
            break
        decoded.append(nxt_id)
        consumed += 1
        tok_id = nxt_id
        if tok_id in _cag_newline_set:
            break
    trim_prompt_cache(_cag_cache, consumed)
    return _cag_clean_answer(_cag_tokenizer.decode(decoded)), ent, marg, ttft_ms


def _cag_cold_rag(question: str, n_gen: int = 20, k: int = 3) -> tuple[str, float]:
    """RAG fallback path: TF-IDF top-k → cold prefill → greedy decode.
    Returns (answer, ttft_ms) where ttft_ms is streaming TTFT only (time
    through the cold prefill; first token emitted at that boundary).
    Matches the metric definition used by `_cag_warm_forward`."""
    import mlx.core as mx
    import time as _time
    from mlx_lm.models.cache import make_prompt_cache
    chunks = _cag_retriever.topk(question, k)
    prompt = "\n\n".join(chunks) + f"\n\nQuestion: {question}\nShort answer:"
    ids = _cag_tokenizer.encode(prompt)
    cache = make_prompt_cache(_cag_model)
    x = mx.array([ids])
    _t0 = _time.perf_counter()
    out = _cag_model(x, cache=cache); mx.eval(out)
    ttft_ms = (_time.perf_counter() - _t0) * 1000
    tok_id = int(mx.argmax(out[0, -1]).item())
    decoded = [tok_id]
    eos = _cag_tokenizer.eos_token_id
    for _ in range(n_gen - 1):
        nxt = mx.array([[tok_id]])
        out = _cag_model(nxt, cache=cache); mx.eval(out)
        nxt_id = int(mx.argmax(out[0, -1]).item())
        if eos is not None and nxt_id == eos:
            break
        decoded.append(nxt_id)
        tok_id = nxt_id
    return _cag_clean_answer(_cag_tokenizer.decode(decoded)), ttft_ms


def _apply_gate(cag_rows: list[dict], rag_rows: list[dict],
                 te: float, tm: float) -> dict:
    """Apply a fixed (Te, Tm) OR-gate to pre-computed CAG + RAG rows and
    return aggregate metrics. Pure function, no side effects. Used for
    per-fold held-out evaluation in CV."""
    n = len(cag_rows)
    if n == 0:
        return {"f1": 0.0, "found": 0.0, "fallback_rate": 0.0, "speedup": 0.0}
    rag_ttft = sum(r["ttft_ms"] for r in rag_rows) / n
    f1_sum = found_sum = 0.0
    fb = 0
    tt = 0.0
    for c, r in zip(cag_rows, rag_rows):
        if c["entropy"] <= te or c["margin"] >= tm:
            f1_sum += c["f1"]; found_sum += c["found"]; tt += c["ttft_ms"]
        else:
            f1_sum += r["f1"]; found_sum += r["found"]
            tt += c["ttft_ms"] + r["ttft_ms"]
            fb += 1
    tmean = tt / n if n else 0.0
    return {"f1": f1_sum/n, "found": found_sum/n,
            "fallback_rate": fb/n, "speedup": (rag_ttft/tmean) if tmean else 0.0}


def _sweep_over_rows(cag_rows: list[dict], rag_rows: list[dict],
                      cluster: str) -> tuple[float, float, int, dict]:
    """Sweep the OR-gate over (Te, Tm) on the given rows, run the 3-tier
    cascade, return (te, tm, tier_fired, metrics). Pure function. Used both
    on the full calibration set (in-sample) and per-fold (CV training set)."""
    n = len(cag_rows)
    if n == 0:
        return 99.0, 0.0, 3, {"f1": 0.0, "found": 0.0, "fallback_rate": 0.0, "speedup": 0.0}
    rag_f1 = sum(r["f1"] for r in rag_rows) / n
    rag_found = sum(r["found"] for r in rag_rows) / n
    rag_ttft = sum(r["ttft_ms"] for r in rag_rows) / n
    cag_f1_mean = sum(r["f1"] for r in cag_rows) / n
    cag_found_mean = sum(r["found"] for r in cag_rows) / n
    cag_ttft_mean = sum(r["ttft_ms"] for r in cag_rows) / n
    center_Te, center_Tm = (3.3, 1.8) if cluster == "A" else (2.15, 0.3)

    best_strict = None  # tier 1
    best_relaxed = None # tier 2
    for Te_i in range(0, 600, 5):
        Te = Te_i / 100.0
        for Tm_i in range(0, 600, 5):
            Tm = Tm_i / 100.0
            f1_sum = found_sum = 0.0
            fb = 0
            tt = 0.0
            for c, r in zip(cag_rows, rag_rows):
                if c["entropy"] <= Te or c["margin"] >= Tm:
                    f1_sum += c["f1"]; found_sum += c["found"]; tt += c["ttft_ms"]
                else:
                    f1_sum += r["f1"]; found_sum += r["found"]
                    tt += c["ttft_ms"] + r["ttft_ms"]
                    fb += 1
            f1 = f1_sum / n; found = found_sum / n; tmean = tt / n
            sp = rag_ttft / tmean if tmean else 0.0
            gs = sp >= 3.0
            gf = fb / n <= 0.25
            gqf1 = (rag_f1 - f1) <= 0.02
            gqfd = (rag_found - found) <= 0.02

            if gs and gf and gqf1 and gqfd:
                dist = ((Te - center_Te) ** 2 + (Tm - center_Tm) ** 2)
                score = (-dist, sp)
                if best_strict is None or score > best_strict[0]:
                    best_strict = (score, Te, Tm, sp, f1, found, fb / n)
            elif gs and gf and gqf1:
                dist = ((Te - center_Te) ** 2 + (Tm - center_Tm) ** 2)
                score = (found, -dist, sp)
                if best_relaxed is None or score > best_relaxed[0]:
                    best_relaxed = (score, Te, Tm, sp, f1, found, fb / n)

    if best_strict is not None:
        _, Te, Tm, sp, f1, found, fb = best_strict
        return Te, Tm, 1, {"f1": f1, "found": found, "fallback_rate": fb, "speedup": sp}
    if best_relaxed is not None:
        _, Te, Tm, sp, f1, found, fb = best_relaxed
        return Te, Tm, 2, {"f1": f1, "found": found, "fallback_rate": fb, "speedup": sp}
    return 99.0, 0.0, 3, {"f1": cag_f1_mean, "found": cag_found_mean,
                           "fallback_rate": 0.0,
                           "speedup": (rag_ttft/cag_ttft_mean) if cag_ttft_mean else 0.0}


def _cag_calibrate(qa_pairs: list[dict], n_gen: int, k: int, cluster: str) -> tuple[float, float]:
    """Sweep (Te, Tm) over the OR-rule on a calibration QA set, return the
    (Te, Tm) selected on the full set (in-sample optimum). Also runs 10-fold
    held-out CV (or fewer folds for small n) and persists held-out aggregate
    metrics in _cag_last_calibration_info so the operator card / persisted
    state shows an honest generalization expectation alongside the in-sample
    pick. The persisted (Te, Tm) is unchanged from the prior behavior.

    QA pair format: {"question": str, "answers": [str, ...]}.
    """
    # Generate per-query CAG + RAG outputs ONCE, then sweep thresholds
    # against the cached results (no model re-runs per threshold).
    log.info(f"CAG calibration: {len(qa_pairs)} QA pairs, k={k}, cluster={cluster}")
    cag_rows = []
    rag_rows = []
    import time as _time
    t0 = _time.perf_counter()
    for i, qa in enumerate(qa_pairs):
        q = qa["question"]; golds = qa["answers"]
        ans_c, ent, marg, tt_c = _cag_warm_forward(q, n_gen=n_gen)
        ans_r, tt_r = _cag_cold_rag(q, n_gen=n_gen, k=k)
        cag_rows.append({"f1": _cag_squad_f1(ans_c, golds),
                          "found": _cag_found(ans_c, golds),
                          "entropy": ent, "margin": marg,
                          "ttft_ms": tt_c})
        rag_rows.append({"f1": _cag_squad_f1(ans_r, golds),
                          "found": _cag_found(ans_r, golds),
                          "ttft_ms": tt_r})
        if (i + 1) % 25 == 0:
            log.info(f"  CAG cal {i+1}/{len(qa_pairs)}  (wall {_time.perf_counter()-t0:.1f}s)")
    log.info(f"CAG calibration generations done in {_time.perf_counter()-t0:.1f}s")

    n = len(qa_pairs)
    rag_f1 = sum(r["f1"] for r in rag_rows) / n
    rag_found = sum(r["found"] for r in rag_rows) / n
    rag_ttft = sum(r["ttft_ms"] for r in rag_rows) / n
    log.info(f"CAG calibration baselines: RAG F1={rag_f1:.3f} found={rag_found:.3f} ttft={rag_ttft:.0f}ms")
    # Diagnostic: also surface pure-CAG baseline so we can tell whether
    # tier 3 falls through because CAG quality is below RAG-2pp (real
    # quality miss) vs because the OR-gate sweep mis-evaluates.
    cag_f1_mean = sum(r["f1"] for r in cag_rows) / n
    cag_found_mean = sum(r["found"] for r in cag_rows) / n
    cag_ttft_mean = sum(r["ttft_ms"] for r in cag_rows) / n
    cag_ent_mean = sum(r["entropy"] for r in cag_rows) / n
    cag_marg_mean = sum(r["margin"] for r in cag_rows) / n
    log.info(f"CAG calibration pure-CAG (Te=99,Tm=0): F1={cag_f1_mean:.3f} found={cag_found_mean:.3f} "
              f"ttft={cag_ttft_mean:.0f}ms speedup={rag_ttft/cag_ttft_mean if cag_ttft_mean else 0:.2f}× "
              f"mean_entropy={cag_ent_mean:.2f} mean_margin={cag_marg_mean:.2f}")

    # In-sample optimum: full-data sweep (persisted (Te, Tm) comes from this).
    te_sel, tm_sel, tier_fired, in_sample = _sweep_over_rows(cag_rows, rag_rows, cluster)

    # 10-fold held-out CV (or fewer folds for small n). Per-fold: pick the
    # (Te, Tm) on 90% train via the same 3-tier cascade, then evaluate at
    # those thresholds on the 10% eval split. Aggregate across folds.
    import random as _rnd
    K = max(2, min(10, n // 10))
    rng = _rnd.Random(42)
    idx = list(range(n)); rng.shuffle(idx)
    fold_size = n // K
    per_fold = []
    for fi in range(K):
        start = fi * fold_size
        end = (fi + 1) * fold_size if fi < K - 1 else n
        eval_idx = set(idx[start:end])
        train_idx = [j for j in range(n) if j not in eval_idx]
        train_cag = [cag_rows[j] for j in train_idx]
        train_rag = [rag_rows[j] for j in train_idx]
        eval_cag  = [cag_rows[j] for j in eval_idx]
        eval_rag  = [rag_rows[j] for j in eval_idx]
        te_f, tm_f, tier_f, _ = _sweep_over_rows(train_cag, train_rag, cluster)
        held_m = _apply_gate(eval_cag, eval_rag, te_f, tm_f)
        per_fold.append({"fold": fi, "te": te_f, "tm": tm_f, "tier": tier_f,
                          "n_eval": len(eval_idx), **held_m})

    total_eval = sum(p["n_eval"] for p in per_fold)
    held_out = {
        "f1":           sum(p["f1"] * p["n_eval"] for p in per_fold) / total_eval,
        "found":        sum(p["found"] * p["n_eval"] for p in per_fold) / total_eval,
        "fallback_rate":sum(p["fallback_rate"] * p["n_eval"] for p in per_fold) / total_eval,
        "speedup":      sum(p["speedup"] * p["n_eval"] for p in per_fold) / total_eval,
        "k_folds":      K,
        "n_total_eval": total_eval,
    }
    log.info(f"CAG calibration held-out CV (K={K}, n={n}): "
              f"F1={held_out['f1']:.3f} found={held_out['found']:.3f} "
              f"fb={held_out['fallback_rate']:.1%} sp={held_out['speedup']:.2f}×")

    # Log in-sample selection + tier
    tier_lbl = {1: "tier 1 (all 4 strict gates pass)",
                2: "tier 2 (3/4 gates; found drops slightly)",
                3: "tier 3 (pure-CAG fallback; no smarter gate fires)"}[tier_fired]
    if tier_fired == 1:
        log.info(f"CAG calibration in-sample: {tier_lbl}: Te={te_sel:.2f} Tm={tm_sel:.2f} "
                  f"(cluster {cluster}) → F1={in_sample['f1']:.3f} "
                  f"found={in_sample['found']:.3f} fb={in_sample['fallback_rate']:.1%} "
                  f"sp={in_sample['speedup']:.2f}×")
    elif tier_fired == 2:
        log.warning(f"CAG calibration in-sample: {tier_lbl} (found drop "
                     f"{(rag_found-in_sample['found'])*100:.1f}pp): "
                     f"Te={te_sel:.2f} Tm={tm_sel:.2f} (cluster {cluster}) → "
                     f"F1={in_sample['f1']:.3f} found={in_sample['found']:.3f} "
                     f"fb={in_sample['fallback_rate']:.1%} sp={in_sample['speedup']:.2f}× — "
                     f"production: consider a larger calibration set")
    else:
        log.warning(f"CAG calibration in-sample: {tier_lbl} — falling back to pure-CAG (Te=99, Tm=0)")

    global _cag_last_calibration_info
    _cag_last_calibration_info = {
        "cluster": cluster,
        "n_qa": n,
        "rag_baseline": {"f1": rag_f1, "found": rag_found, "ttft_ms": rag_ttft},
        "pure_cag_baseline": {
            "f1": cag_f1_mean, "found": cag_found_mean, "ttft_ms": cag_ttft_mean,
            "speedup_vs_rag": (rag_ttft / cag_ttft_mean) if cag_ttft_mean else 0.0,
            "mean_entropy": cag_ent_mean, "mean_margin": cag_marg_mean,
        },
        "tier_fired": tier_fired,
        "te": te_sel,
        "tm": tm_sel,
        "gated_in_sample": in_sample,
        "gated_heldout_cv": held_out,
        "per_fold": per_fold,
    }
    return (te_sel, tm_sel)


@_once_under_init_lock
def _init_pion_cag_hybrid_lazy():
    """Lazy init for the pion-cag-hybrid backend.
    Loads MLX model + tokenizer, reads the foundation corpus, preloads the
    prompt cache, fits the TF-IDF retriever, and runs the calibration sweep
    over the supplied QA pairs. Idempotent under `@_once_under_init_lock`
    (gh #84) — two concurrent first-requests don't both load the model.

    Required config: cag_foundation_corpus, and one of
    {cag_calibration_qa, cag_calibration_state}. When `cag_calibration_state`
    is set (path to a JSON written by `pion-serve setup`), (Te, Tm) + cluster
    are loaded from the state file and the per-startup calibration sweep is
    skipped — only the foundation preload still runs (a few minutes for ~26K
    tokens on Mac M-series).
    """
    global _cag_model, _cag_tokenizer, _cag_cache, _cag_paragraphs
    global _cag_retriever, _cag_te, _cag_tm
    import time as _time
    import json as _json
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus_path  = _config.get("cag_foundation_corpus")
    cal_path     = _config.get("cag_calibration_qa")
    state_path   = _config.get("cag_calibration_state")

    # Precomputed-state path: load (Te, Tm) from a setup-written JSON instead
    # of running the calibration sweep. Foundation preload still happens.
    precomputed_state = None
    if state_path:
        sp = Path(state_path)
        if not sp.is_file():
            raise RuntimeError(f"--cag-calibration-state file not found: {state_path}")
        precomputed_state = _json.loads(sp.read_text())
        cal = precomputed_state.get("calibration", {})
        if "te" not in cal or "tm" not in cal:
            raise RuntimeError(f"calibration state missing te/tm: {state_path}")
        log.info(f"CAG init: loading precomputed calibration from {state_path} "
                  f"(tier {cal.get('tier_fired', '?')}, cluster {cal.get('cluster', '?')})")

    if not corpus_path or (not cal_path and not state_path):
        raise RuntimeError("pion-cag-hybrid requires --cag-foundation-corpus + "
                            "(--cag-calibration-qa OR --cag-calibration-state)")

    corpus_text = Path(corpus_path).read_text()
    # Paragraphs = blocks separated by 2+ newlines; whitespace trimmed.
    paragraphs = [p.strip() for p in corpus_text.split("\n\n") if p.strip()]
    log.info(f"CAG init: loaded {len(paragraphs)} paragraphs, {len(corpus_text)} chars from {corpus_path}")

    qa_pairs: list[dict] = []
    if cal_path:
        for line in Path(cal_path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            if "question" not in rec or "answers" not in rec:
                raise RuntimeError(f"calibration QA record missing 'question' or 'answers': {rec}")
            qa_pairs.append(rec)
        log.info(f"CAG init: loaded {len(qa_pairs)} calibration QA pairs from {cal_path}")

    model_id = _config["cag_mlx_model"]
    log.info(f"CAG init: loading MLX model {model_id}")
    model, tok = load(model_id)
    _cag_model = model
    _cag_tokenizer = tok

    # Build retriever over the paragraphs (for both fallback path AND
    # held-out RAG during calibration).
    class _TfidfRetriever:
        def __init__(self, paras):
            self.paras = paras
            self.vec = TfidfVectorizer(stop_words="english").fit(paras)
            self.mat = self.vec.transform(paras)
        def topk(self, q, k):
            qv = self.vec.transform([q])
            scores = cosine_similarity(qv, self.mat)[0]
            idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            return [self.paras[i] for i in idx]
    _cag_retriever = _TfidfRetriever(paragraphs)
    _cag_paragraphs = paragraphs

    # Preload foundation cache once.
    cache = make_prompt_cache(model)
    ids = tok.encode("\n\n".join(paragraphs))
    n_corpus_tok = len(ids)
    log.info(f"CAG init: foundation corpus = {n_corpus_tok} tokens; preloading prompt cache...")
    x = mx.array([ids])
    CHUNK = 512
    t0 = _time.perf_counter()
    for i in range(0, x.shape[1], CHUNK):
        _ = model(x[:, i:i + CHUNK], cache=cache)
    mx.eval(*[c.state for c in cache])
    log.info(f"CAG init: preload done in {_time.perf_counter()-t0:.1f}s")
    _cag_cache = cache

    # Calibrate thresholds (skip if state was precomputed by `setup`).
    n_gen = int(_config.get("cag_n_gen", 20))
    k     = int(_config.get("cag_rag_k", 3))

    if precomputed_state is not None:
        cal = precomputed_state["calibration"]
        _cag_te = float(cal["te"])
        _cag_tm = float(cal["tm"])
        log.info(f"CAG init: ready (from state) — Te={_cag_te:.2f}  Tm={_cag_tm:.2f}  "
                  f"(cluster={cal.get('cluster')}, tier={cal.get('tier_fired')}, "
                  f"model={model_id}, {n_corpus_tok} corpus tokens)")
        return

    cluster = _config.get("cag_cluster", "A")

    # Warmup: one discarded forward against the first calibration question.
    # Without this the first scored call pays MLX lazy kernel-compile cost
    # (the same pattern as run_8b_multisig.py's warmup before the scored loop).
    if qa_pairs:
        _ = _cag_warm_forward(qa_pairs[0]["question"], n_gen=n_gen)

    _cag_te, _cag_tm = _cag_calibrate(qa_pairs, n_gen=n_gen, k=k, cluster=cluster)
    log.info(f"CAG init: ready — Te={_cag_te:.2f}  Tm={_cag_tm:.2f}  "
              f"(cluster={cluster}, model={model_id}, "
              f"{n_corpus_tok} corpus tokens)")


def _call_pion_cag_hybrid(messages: list[dict], body: dict) -> dict:
    """Per-request handler: extract user query, run multi-signal OR-gate.
    On gate-keep: return CAG warm-forward answer. On fallback: cold RAG."""
    _init_pion_cag_hybrid_lazy()
    n_gen = int(body.get("max_tokens", _config.get("cag_n_gen", 20)))
    k     = int(_config.get("cag_rag_k", 3))
    # Extract the user query (last user message)
    question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            question = msg.get("content", "")
            break
    if not question:
        return {"content": "", "model": _config["cag_mlx_model"],
                "prompt_tokens": 0, "completion_tokens": 0,
                "cag_branch": "empty_query"}

    ans_cag, ent, marg, _ttft_cag_ms = _cag_warm_forward(question, n_gen=n_gen)
    keep = (ent <= _cag_te) or (marg >= _cag_tm)
    if keep:
        _incr_stat("cag_kept")
        return {
            "content": ans_cag,
            "model": _config["cag_mlx_model"],
            "prompt_tokens": 0,        # warm-forward only sees the suffix
            "completion_tokens": 0,
            "cag_branch": "cag_kept",
            "cag_signals": {"entropy": ent, "margin": marg,
                            "te": _cag_te, "tm": _cag_tm},
        }
    # Fallback to RAG cold prefill
    _incr_stat("cag_fallback")
    ans_rag, _ttft_rag_ms = _cag_cold_rag(question, n_gen=n_gen, k=k)
    return {
        "content": ans_rag,
        "model": _config["cag_mlx_model"],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cag_branch": "rag_fallback",
        "cag_signals": {"entropy": ent, "margin": marg,
                        "te": _cag_te, "tm": _cag_tm},
    }


# ── `setup` subcommand: first-class CAG-hybrid onboarding (gh #23 productize) ─

def _read_corpus_from_path(path: Path) -> tuple[str, list[str]]:
    """Read a corpus from a file (joined by blank lines) or a directory
    (all .txt + .md files concatenated in sorted order, each file's content
    split into paragraphs on blank-line boundaries). Returns (joined_text,
    paragraph_list)."""
    paragraphs: list[str] = []
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    elif path.is_dir():
        files = sorted(list(path.glob("*.txt")) + list(path.glob("*.md")) +
                        list(path.glob("**/*.txt")) + list(path.glob("**/*.md")))
        seen = set()
        for f in files:
            if f in seen: continue
            seen.add(f)
            text = f.read_text(encoding="utf-8", errors="replace")
            for p in text.split("\n\n"):
                p = p.strip()
                if len(p) >= 80:  # skip headers and noise
                    paragraphs.append(p)
    else:
        raise FileNotFoundError(f"corpus path not found: {path}")
    if not paragraphs:
        raise RuntimeError(f"no paragraphs extracted from {path} — corpus appears empty")
    return "\n\n".join(paragraphs), paragraphs


def _gen_calibration_qa_template(paragraphs: list[str], count: int,
                                  seed: int = 42) -> list[dict]:
    """Generate calibration QA pairs from raw paragraphs (no markdown headers
    required). Each generated pair targets one paragraph and uses its first
    informative sentence as the gold answer. Deterministic given (paragraphs,
    seed). Returns up to `count` pairs in {question, answers: [gold]} form."""
    import random as _rnd
    rng = _rnd.Random(seed)
    paras = list(paragraphs)
    rng.shuffle(paras)

    # Pull the first 1-2 sentence prefix from each paragraph as the gold span,
    # and ask a "what does the text say about..." style question anchored to
    # the first ~6 content words of the paragraph.
    qa: list[dict] = []
    for p in paras:
        if len(qa) >= count: break
        # First sentence (rough: split on ". " / "? " / "! " — fall back to first 200 chars)
        sents = re.split(r'(?<=[.!?])\s+', p)
        gold = (sents[0] if sents else p[:200]).strip()
        if len(gold) < 20 or len(gold) > 300:
            continue
        # Anchor phrase: first 5-8 words of the paragraph that look like a noun phrase
        words = p.split()
        anchor = " ".join(words[:6]).strip(string.punctuation)
        if len(anchor) < 10:
            continue
        question = f"According to the text, what is said about {anchor.lower()}?"
        qa.append({"question": question, "answers": [gold]})
    return qa[:count]


def _hash_corpus(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _format_operator_card(state: dict) -> str:
    """One-page summary an operator can paste into onboarding docs."""
    s = state
    cal = s["calibration"]
    cor = s["corpus"]
    pc  = cal["pure_cag_baseline"]
    rag = cal["rag_baseline"]
    gat = cal["gated_in_sample"]
    cv  = cal.get("gated_heldout_cv")
    tier_label = {1: "tier 1 (all 4 strict gates pass)",
                  2: "tier 2 (3/4 gates; found drops slightly)",
                  3: "tier 3 (pure-CAG fallback; no smarter gate fires)"}[cal["tier_fired"]]
    deploy_cmd = (f'python pion-serve/serve.py --backend pion-cag-hybrid '
                   f'--cag-foundation-corpus {cor["path"]} '
                   f'--cag-calibration-state {s["state_path"]}')
    # Held-out is the honest generalization expectation; show it as the
    # headline. In-sample is reported below as the calibration optimum.
    if cv:
        headline = (
            f"│  Held-out CV ({cv['k_folds']}-fold over {cv['n_total_eval']} eval queries) — operator-facing headline:\n"
            f"│    gated policy:  F1={cv['f1']:.3f}  found={cv['found']:.3f}  "
            f"fb={cv['fallback_rate']*100:.1f}%  ({cv['speedup']:.1f}×)\n"
            f"├───────────────────────────────────────────────────────────────────────────────┤\n"
        )
    else:
        headline = ""
    return f"""
╭───────────────────────────────────────────────────────────────────────────────╮
│  pion-cag-hybrid — operator card                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│  Corpus:       {cor['path']}
│                {cor['n_paragraphs']} paragraphs, {cor['n_tokens']:,} tokens, sha256[:16]={cor['hash']}
│  Model:        {cal['model']}
│  Calibration:  {cal['n_qa']} QA pairs, cluster={cal['cluster']}, {tier_label}
│                Selected (Te, Tm) = ({cal['te']:.2f}, {cal['tm']:.2f})  (in-sample optimum)
├───────────────────────────────────────────────────────────────────────────────┤
{headline}│  In-sample baselines on {cal['n_qa']} calibration queries (for reference):
│    pure RAG:      F1={rag['f1']:.3f}  found={rag['found']:.3f}  TTFT={rag['ttft_ms']:.0f}ms
│    pure CAG:      F1={pc['f1']:.3f}  found={pc['found']:.3f}  TTFT={pc['ttft_ms']:.0f}ms  ({pc['speedup_vs_rag']:.1f}×)
│    gated policy:  F1={gat['f1']:.3f}  found={gat['found']:.3f}  fb={gat['fallback_rate']*100:.1f}%  ({gat['speedup']:.1f}×)
├───────────────────────────────────────────────────────────────────────────────┤
│  Deploy:
│    {deploy_cmd}
│
│  State file: {s['state_path']}
╰───────────────────────────────────────────────────────────────────────────────╯
"""


def _setup_command(argv: list[str]) -> int:
    """`pion-serve setup` — onboard a corpus into pion-cag-hybrid.
    Generates calibration QA, runs the cascade, persists state, prints card."""
    global _config
    sp = argparse.ArgumentParser(
        prog="pion-serve setup",
        description="Onboard a corpus into pion-cag-hybrid: generate "
                     "calibration QA, run the 3-tier cascade, persist (Te, Tm).")
    sp.add_argument("--corpus", required=True,
                     help="Path to a corpus file (paragraphs separated by blank lines) "
                          "or directory containing .txt/.md files.")
    sp.add_argument("--qa-source", default=None,
                     help="Optional path to a pre-curated calibration QA JSONL "
                          "({question, answers:[gold,...]} per line). If unset, "
                          "QA is auto-generated from the corpus via the template "
                          "generator.")
    sp.add_argument("--qa-count", type=int, default=150,
                     help="Number of calibration QA pairs to generate (default: 150)")
    sp.add_argument("--qa-seed", type=int, default=42,
                     help="Seed for deterministic QA generation (default: 42)")
    sp.add_argument("--cluster", default="A", choices=["A", "B"],
                     help="Threshold cluster bias (default: A)")
    sp.add_argument("--model", default="mlx-community/Llama-3.1-8B-Instruct-4bit",
                     help="MLX model ID (default: Llama-3.1-8B-Instruct-4bit)")
    sp.add_argument("--n-gen", type=int, default=20,
                     help="Max generation tokens (default: 20)")
    sp.add_argument("--rag-k", type=int, default=3,
                     help="Top-k for RAG fallback (default: 3)")
    sp.add_argument("--out", default=None,
                     help="Path to write the calibration state JSON. Default: "
                          "<corpus_dir>/.pion_cag_calibration.json")
    args = sp.parse_args(argv)

    corpus_path = Path(args.corpus).resolve()
    print(f"[setup] reading corpus from {corpus_path}")
    corpus_text, paragraphs = _read_corpus_from_path(corpus_path)
    corpus_hash = _hash_corpus(corpus_text)
    print(f"[setup] {len(paragraphs)} paragraphs, {len(corpus_text):,} chars, "
          f"sha256[:16]={corpus_hash}")

    # ── Resolve / generate calibration QA ────────────────────────────────────
    if args.qa_source:
        qa_path = Path(args.qa_source).resolve()
        qa_pairs = []
        for line in qa_path.read_text().splitlines():
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if "question" not in rec or "answers" not in rec:
                raise RuntimeError(f"bad QA record: {rec}")
            qa_pairs.append(rec)
        print(f"[setup] loaded {len(qa_pairs)} curated QA pairs from {qa_path}")
        qa_source_label = str(qa_path)
    else:
        qa_pairs = _gen_calibration_qa_template(paragraphs, count=args.qa_count,
                                                 seed=args.qa_seed)
        print(f"[setup] generated {len(qa_pairs)} template-mode QA pairs "
              f"(seed={args.qa_seed})")
        qa_source_label = f"template(seed={args.qa_seed},count={args.qa_count})"

    # Write the (generated or curated) QA to a sidecar file so the operator
    # can inspect / hand-edit it before re-running setup.
    qa_sidecar = Path(args.out).parent / "calibration_qa.jsonl" if args.out else \
                 corpus_path.parent / ".pion_cag_calibration_qa.jsonl"
    qa_sidecar.write_text("\n".join(json.dumps(qa) for qa in qa_pairs) + "\n")
    print(f"[setup] wrote {len(qa_pairs)} QA pairs to {qa_sidecar}")

    # ── Stash a foundation-corpus file the cascade init can read ─────────────
    corpus_sidecar = qa_sidecar.parent / ".pion_cag_foundation_corpus.txt"
    corpus_sidecar.write_text(corpus_text)

    # ── Wire the cascade init with these inputs ──────────────────────────────
    _config = {
        "cag_mlx_model": args.model,
        "cag_foundation_corpus": str(corpus_sidecar),
        "cag_calibration_qa": str(qa_sidecar),
        "cag_cluster": args.cluster,
        "cag_n_gen": args.n_gen,
        "cag_rag_k": args.rag_k,
    }
    print(f"[setup] running cascade preload + calibration (this is the slow step)")
    t0 = time.perf_counter()
    _init_pion_cag_hybrid_lazy()
    cal_wall = time.perf_counter() - t0
    print(f"[setup] cascade init done in {cal_wall:.1f}s")

    info = _cag_last_calibration_info or {}
    info["model"] = args.model
    info["wall_s"] = round(cal_wall, 1)
    info["qa_source"] = qa_source_label

    state = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": {"path": str(corpus_path), "hash": corpus_hash,
                    "n_paragraphs": len(paragraphs),
                    "n_chars": len(corpus_text),
                    "n_tokens": int(info.get("pure_cag_baseline", {}).get("ttft_ms", 0) and 0 or 0)},
        "calibration": info,
    }
    # Patch n_tokens from the cascade's logged value if available
    # (cascade logs "foundation corpus = N tokens" but we don't capture that
    # cleanly — derive from suffix-encode length via the tokenizer instead)
    try:
        state["corpus"]["n_tokens"] = len(_cag_tokenizer.encode(corpus_text))
    except Exception:
        pass

    out_path = Path(args.out).resolve() if args.out else \
                corpus_path.parent / ".pion_cag_calibration.json"
    state["state_path"] = str(out_path)
    out_path.write_text(json.dumps(state, indent=2))
    print(f"\n[setup] wrote state → {out_path}")
    print(_format_operator_card(state))
    return 0


def _forward_to_backend(messages: list[dict], body: dict,
                        route: Optional[RouteDecision] = None) -> dict:
    """Forward chat request to backend and return response."""
    backend_type, backend_url, model = _resolve_backend(body, route)

    if backend_type == "pion-moe":
        return _call_pion_moe(messages, body)

    if backend_type == "pion-cag-hybrid":
        return _call_pion_cag_hybrid(messages, body)

    if backend_type == "ollama":
        # Ollama native chat API
        ollama_body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {},
        }
        if "temperature" in body:
            ollama_body["options"]["temperature"] = body["temperature"]
        if "max_tokens" in body:
            ollama_body["options"]["num_predict"] = body["max_tokens"]

        resp = requests.post(
            f"{backend_url}/api/chat",
            json=ollama_body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data.get("message", {}).get("content", ""),
            "model": model,
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        }
    elif backend_type == "claude":
        # Anthropic Messages API
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=api_key)
        # Extract system message
        system_text = ""
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg["content"]
            else:
                user_messages.append(msg)
        kwargs = {"model": model, "max_tokens": body.get("max_tokens", 1024),
                  "messages": user_messages}
        if system_text:
            kwargs["system"] = system_text
        resp = client.messages.create(**kwargs)
        return {
            "content": resp.content[0].text,
            "model": model,
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        }
    else:
        # OpenAI-compatible API (vLLM, OpenAI, Gemini, llama.cpp)
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        chat_path = BACKENDS.get(backend_type, {}).get("chat_path", "/v1/chat/completions")
        req_body = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        for k in ["temperature", "max_tokens", "top_p"]:
            if k in body:
                req_body[k] = body[k]

        resp = requests.post(
            f"{backend_url}{chat_path}",
            json=req_body,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        return {
            "content": choice.get("message", {}).get("content", ""),
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }


def _forward_stream(messages: list[dict], body: dict,
                    route: Optional[RouteDecision] = None) -> Generator[str, None, None]:
    """Forward as streaming SSE to backend."""
    backend_type, backend_url, model = _resolve_backend(body, route)

    if backend_type == "claude":
        # Anthropic SSE differs from OpenAI; fall back to non-stream + single chunk.
        result = _forward_to_backend(messages, body, route=route)
        chunk = {
            "id": f"pion-{int(time.time())}",
            "object": "chat.completion.chunk",
            "model": result["model"],
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": result["content"]}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        done = {
            "id": f"pion-{int(time.time())}",
            "object": "chat.completion.chunk",
            "model": result["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"
        user_query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if user_query and result["content"] and _config.get("cache_enabled", True):
            _cache_store(user_query, result["content"])
        return

    if backend_type == "ollama":
        ollama_body = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        resp = requests.post(
            f"{backend_url}/api/chat",
            json=ollama_body,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        full_content = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                full_content += token
                chunk = {
                    "id": f"pion-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                continue

        # Final chunk
        done = {
            "id": f"pion-{int(time.time())}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

        # Cache the full response + ingest concept
        user_query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if user_query and full_content and _config.get("cache_enabled", True):
            _cache_store(user_query, full_content)
        if user_query and full_content and _config.get("distill_enabled") and _embed_fn:
            query_emb = _embed_fn(user_query)
            if query_emb is not None:
                if _concept_store:
                    _concept_store.ingest(query_emb, full_content, query_text=user_query)
                if _fragment_store:
                    cid = _concept_store._concept_count - 1 if _concept_store else -1
                    _fragment_store.ingest_response(full_content, source_concept_id=cid)
    else:
        # OpenAI-compatible streaming passthrough
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        chat_path = BACKENDS.get(backend_type, {}).get("chat_path", "/v1/chat/completions")
        req_body = {"model": model, "messages": messages, "stream": True}
        for k in ["temperature", "max_tokens", "top_p"]:
            if k in body:
                req_body[k] = body[k]

        resp = requests.post(
            f"{backend_url}{chat_path}",
            json=req_body, headers=headers, stream=True, timeout=120,
        )
        resp.raise_for_status()
        full_content = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="replace")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    yield f"{line_str}\n\n"
                    break
                try:
                    d = json.loads(data_str)
                    token = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    full_content += token
                except json.JSONDecodeError:
                    pass
                yield f"{line_str}\n\n"

        user_query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if user_query and full_content and _config.get("cache_enabled", True):
            _cache_store(user_query, full_content)
        if user_query and full_content and _config.get("distill_enabled") and _embed_fn:
            query_emb = _embed_fn(user_query)
            if query_emb is not None:
                if _concept_store:
                    _concept_store.ingest(query_emb, full_content, query_text=user_query)
                if _fragment_store:
                    cid = _concept_store._concept_count - 1 if _concept_store else -1
                    _fragment_store.ingest_response(full_content, source_concept_id=cid)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions with semantic cache + RAG."""
    global _stats
    _incr_stat("total_requests")
    t0 = time.perf_counter()

    body = request.get_json(force=True)
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract user query
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg["content"]
            break

    if not user_query:
        return jsonify({"error": "No user message found"}), 400

    # 1. Semantic cache check
    if _config.get("cache_enabled", True):
        cached = _cache_check(user_query, _config.get("cache_threshold", 0.85))
        if cached:
            _incr_stat("cache_hits")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _incr_stat("total_latency_ms", elapsed_ms)
            log.info(f"CACHE HIT ({elapsed_ms:.0f}ms): {user_query[:60]}...")

            resp_body = {
                "id": f"pion-cache-{int(time.time())}",
                "object": "chat.completion",
                "model": body.get("model", _config.get("model", "unknown")),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": cached},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "x_pion": {"source": "l1_cache", "latency_ms": round(elapsed_ms, 1)},
            }

            if stream:
                def cached_sse():
                    chunk = {"id": resp_body["id"], "object": "chat.completion.chunk",
                             "model": resp_body["model"],
                             "choices": [{"index": 0, "delta": {"role": "assistant", "content": cached}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    done = {"id": resp_body["id"], "object": "chat.completion.chunk",
                            "model": resp_body["model"],
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    yield f"data: {json.dumps(done)}\n\n"
                    yield "data: [DONE]\n\n"
                return Response(stream_with_context(cached_sse()), mimetype="text/event-stream")

            return jsonify(resp_body)

    _incr_stat("cache_misses")

    # Check if this is a follow-up correction to an L3 response
    if _config.get("distill_enabled"):
        _check_negative_feedback(user_query)

    # 2. L3 concept synthesis (if --distill enabled)
    if _config.get("distill_enabled") and _concept_store and _embed_fn:
        query_emb = _embed_fn(user_query)
        if query_emb is not None:
            l3_result = _concept_store.try_synthesize(query_emb)
            if l3_result and l3_result.strategy in ("direct", "composite"):
                _incr_stat("l3_synthesis")
                if l3_result.strategy == "direct":
                    _incr_stat("l3_direct")
                else:
                    _incr_stat("l3_composite")
                elapsed_ms = (time.perf_counter() - t0) * 1000
                _incr_stat("total_latency_ms", elapsed_ms)
                log.info(
                    f"L3 {l3_result.strategy.upper()} ({elapsed_ms:.0f}ms, "
                    f"conf={l3_result.confidence:.2f}, "
                    f"sim={l3_result.cosine_similarity:.3f}): {user_query[:60]}..."
                )

                resp_body = {
                    "id": f"pion-l3-{int(time.time())}",
                    "object": "chat.completion",
                    "model": body.get("model", _config.get("model", "unknown")),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": l3_result.response},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "x_pion": {
                        "source": "l3_synthesis",
                        "strategy": l3_result.strategy,
                        "confidence": round(l3_result.confidence, 3),
                        "similarity": round(l3_result.cosine_similarity, 3),
                        "concepts": l3_result.concept_ids,
                        "latency_ms": round(elapsed_ms, 1),
                    },
                }

                if stream:
                    content = l3_result.response
                    def l3_sse():
                        chunk = {"id": resp_body["id"], "object": "chat.completion.chunk",
                                 "model": resp_body["model"],
                                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                        done = {"id": resp_body["id"], "object": "chat.completion.chunk",
                                "model": resp_body["model"],
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                        yield f"data: {json.dumps(done)}\n\n"
                        yield "data: [DONE]\n\n"
                    return Response(stream_with_context(l3_sse()), mimetype="text/event-stream")

                # Also promote to L1 cache + track for feedback
                _cache_store(user_query, l3_result.response)
                _track_l3_response(user_query, l3_result.concept_ids)
                return jsonify(resp_body)

    # 3. L3b fragment synthesis (if --distill enabled)
    _frag_augmented_prompt = None
    if _config.get("distill_enabled") and _fragment_store and _embed_fn:
        query_emb = query_emb if 'query_emb' in dir() else _embed_fn(user_query)
        if query_emb is not None:
            frag_result = _fragment_store.try_synthesize(query_emb)
            if frag_result:
                if frag_result.strategy == "full_synthesis":
                    # L3a: full synthesis from fragments — bypass LLM
                    _incr_stat("l3_synthesis")
                    _incr_stat("l3_fragment")
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    _incr_stat("total_latency_ms", elapsed_ms)
                    response_text = "\n\n".join(frag_result.fragments)
                    log.info(
                        f"L3b FULL_SYNTHESIS ({elapsed_ms:.0f}ms, "
                        f"coverage={frag_result.coverage:.2f}, "
                        f"{len(frag_result.fragments)} frags): {user_query[:60]}..."
                    )
                    resp_body = {
                        "id": f"pion-l3b-{int(time.time())}",
                        "object": "chat.completion",
                        "model": body.get("model", _config.get("model", "unknown")),
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop",
                        }],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "x_pion": {
                            "source": "l3b_full_synthesis",
                            "coverage": round(frag_result.coverage, 3),
                            "fragments": len(frag_result.fragments),
                            "latency_ms": round(elapsed_ms, 1),
                        },
                    }
                    if stream:
                        content = response_text
                        def l3b_sse():
                            chunk = {"id": resp_body["id"], "object": "chat.completion.chunk",
                                     "model": resp_body["model"],
                                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]}
                            yield f"data: {json.dumps(chunk)}\n\n"
                            done_chunk = {"id": resp_body["id"], "object": "chat.completion.chunk",
                                    "model": resp_body["model"],
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                            yield f"data: {json.dumps(done_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                        return Response(stream_with_context(l3b_sse()), mimetype="text/event-stream")
                    _cache_store(user_query, response_text)
                    return jsonify(resp_body)

                elif frag_result.strategy == "fragment_augmented":
                    # L3b: inject fragments as context — cheaper LLM call
                    _frag_augmented_prompt = frag_result.augmented_prompt
                    log.info(
                        f"L3b AUGMENTED (coverage={frag_result.coverage:.2f}, "
                        f"{len(frag_result.fragments)} frags): {user_query[:60]}..."
                    )

    # 4. RAG context injection
    if _config.get("rag_index"):
        rag_docs = _rag_retrieve(user_query, k=_config.get("rag_k", 3))
        if rag_docs:
            messages = _inject_rag_context(messages, rag_docs)
            _incr_stat("rag_injections")
            log.info(f"RAG: injected {len(rag_docs)} docs")

    # 4b. Fragment-augmented context injection (L3b)
    if _frag_augmented_prompt:
        _incr_stat("l3_fragment")
        # Prepend verified fragments to system message
        new_messages = []
        has_system = False
        for msg in messages:
            if msg.get("role") == "system":
                new_messages.append({
                    "role": "system",
                    "content": _frag_augmented_prompt + msg["content"],
                })
                has_system = True
            else:
                new_messages.append(msg)
        if not has_system:
            new_messages.insert(0, {"role": "system", "content": _frag_augmented_prompt + "Answer concisely."})
        messages = new_messages

    # 4c. Intent routing — classify query and pick a tier (model+backend).
    route: Optional[RouteDecision] = None
    if _intent_router is not None:
        cur_emb = query_emb if 'query_emb' in dir() and query_emb is not None else (
            _embed_fn(user_query) if _embed_fn else None
        )
        route = _intent_router.classify(user_query, cur_emb)
        _incr_stat(f"route_{route.tier}")
        log.info(
            f"ROUTE tier={route.tier} backend={route.backend} model={route.model} "
            f"(s={route.centroid_score['simple']:.2f} c={route.centroid_score['complex']:.2f}"
            f"{' ' + route.heuristic_boost if route.heuristic_boost else ''})"
        )

    # 5. Forward to backend
    _incr_stat("full_inference")
    if stream:
        return Response(
            stream_with_context(_forward_stream(messages, body, route=route)),
            mimetype="text/event-stream",
        )

    result = _forward_to_backend(messages, body, route=route)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _incr_stat("total_latency_ms", elapsed_ms)

    # Estimate cost for this request from the route's tier price
    if route is not None and route.cost_per_mtok > 0:
        toks = result.get("prompt_tokens", 0) + result.get("completion_tokens", 0)
        _incr_stat("route_estimated_cost_usd", toks / 1_000_000 * route.cost_per_mtok)

    # 6. Cache response + ingest concept + decompose into fragments
    if _config.get("cache_enabled", True) and result["content"]:
        _cache_store(user_query, result["content"])

    if _config.get("distill_enabled") and result["content"]:
        query_emb = query_emb if 'query_emb' in dir() else (_embed_fn(user_query) if _embed_fn else None)
        if query_emb is not None:
            if _concept_store:
                _concept_store.ingest(query_emb, result["content"], query_text=user_query)
            if _fragment_store:
                cid = _concept_store._concept_count - 1 if _concept_store else -1
                _fragment_store.ingest_response(result["content"], source_concept_id=cid)

    source = "fragment_augmented" if _frag_augmented_prompt else "full_inference"
    log.info(f"BACKEND ({elapsed_ms:.0f}ms): {user_query[:60]}...")

    x_pion: dict = {"source": source, "latency_ms": round(elapsed_ms, 1)}
    if route is not None:
        x_pion["route"] = {
            "tier": route.tier,
            "backend": route.backend,
            "model": route.model,
            "confidence": route.confidence,
            "heuristic_boost": route.heuristic_boost,
        }

    return jsonify({
        "id": f"pion-{int(time.time())}",
        "object": "chat.completion",
        "model": result["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["content"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
        "x_pion": x_pion,
    })


@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available models from backend."""
    backend_type = _config.get("backend_type", "ollama")
    backend_url = _config.get("backend_url", "http://127.0.0.1:11434")

    if backend_type == "ollama":
        try:
            resp = requests.get(f"{backend_url}/api/tags", timeout=5)
            models = resp.json().get("models", [])
            return jsonify({
                "data": [{"id": m["name"], "object": "model"} for m in models],
            })
        except Exception:
            return jsonify({"data": []})
    else:
        try:
            models_path = BACKENDS.get(backend_type, {}).get("models_path", "/v1/models")
            resp = requests.get(f"{backend_url}{models_path}", timeout=5)
            return jsonify(resp.json())
        except Exception:
            return jsonify({"data": []})


@app.route("/v1/stats", methods=["GET"])
def stats():
    """Pion Serve statistics — inference proxy + MLX sidecar + V-Store."""
    # gh #84: snapshot under the lock so the derived rates (l1/l3/inf) all see
    # a coherent set of counter values. Without this, two `_stats[...]` reads
    # interleaved with a request handler's mutations can produce a >100% rate
    # in the same `/v1/stats` response.
    s = _stats_snapshot()
    total = max(s.get("total_requests", 0), 1)
    avg_latency = s.get("total_latency_ms", 0) / total
    l1_rate = s.get("cache_hits", 0) / total
    l3_rate = s.get("l3_synthesis", 0) / total
    inf_rate = s.get("full_inference", 0) / total
    result = {
        "total_requests": s.get("total_requests", 0),
        "l1_cache_hits": s.get("cache_hits", 0),
        "l3_synthesis": s.get("l3_synthesis", 0),
        "l3_direct": s.get("l3_direct", 0),
        "l3_composite": s.get("l3_composite", 0),
        "full_inference": s.get("full_inference", 0),
        "l1_hit_rate": round(l1_rate, 3),
        "l3_synthesis_rate": round(l3_rate, 3),
        "full_inference_rate": round(inf_rate, 3),
        "cost_savings_pct": round((l1_rate + l3_rate) * 100, 1),
        "rag_injections": s.get("rag_injections", 0),
        "avg_latency_ms": round(avg_latency, 1),
        "backend": _config.get("backend_type", "unknown"),
        "model": _config.get("model", "unknown"),
        "cache_enabled": _config.get("cache_enabled", True),
        "distill_enabled": _config.get("distill_enabled", False),
        "rag_index": _config.get("rag_index"),
        # gh #69: embed backend telemetry — active backend + per-backend hits.
        "embedder": {
            "backend": _config.get("embed_backend", "none"),
            "sie_url": _config.get("sie_url"),
            "sie_model": _config.get("sie_model") if _config.get("sie_url") else None,
            "calls": {
                "sie": s.get("embed_calls_sie", 0),
                "ollama": s.get("embed_calls_ollama", 0),
                "pion": s.get("embed_calls_pion", 0),
            },
        },
    }

    # L3 concept store stats
    if _concept_store:
        result["l3_concepts"] = _concept_store.get_stats()
    if _fragment_store:
        result["l3_fragments"] = _fragment_store.get_stats()

    # Intent router stats
    if _intent_router is not None:
        result["intent_router"] = {
            **_intent_router.get_stats(),
            "estimated_cost_usd": round(s.get("route_estimated_cost_usd", 0.0), 4),
        }

    # MLX sidecar stats (if configured)
    mlx_port = _config.get("mlx_tcp_port", 0)
    mlx_host = _config.get("mlx_tcp_host", "127.0.0.1")
    if mlx_port and mlx_port > 0:
        result["mlx_gpu"] = _get_mlx_stats(mlx_host, mlx_port)

    # V-Store stats via Pion RESP
    if _pion:
        result["v_store"] = _get_vstore_stats()

    # pion-cag-hybrid gate stats — only emitted when this backend is active
    if _config.get("backend_type") == "pion-cag-hybrid":
        cag_total = s.get("cag_kept", 0) + s.get("cag_fallback", 0)
        result["cag_hybrid"] = {
            "cag_kept": s.get("cag_kept", 0),
            "cag_fallback": s.get("cag_fallback", 0),
            "fallback_rate": round(s.get("cag_fallback", 0) / max(cag_total, 1), 3),
            "te": _cag_te,
            "tm": _cag_tm,
            "cluster": _config.get("cag_cluster", "A"),
            "model": _config.get("cag_mlx_model"),
        }

    return jsonify(result)


def _get_mlx_stats(host: str, port: int) -> dict:
    """Poll MLX sidecar health over TCP."""
    import socket as sock
    import struct
    try:
        s = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((host, port))
        # Send HEALTH request
        header = struct.pack("<BII", 4, 0, 0)  # MSG_HEALTH=4
        s.sendall(header)
        resp_hdr = b""
        while len(resp_hdr) < 10:
            resp_hdr += s.recv(10 - len(resp_hdr))
        _, _, status, body_len = struct.unpack("<BIBI", resp_hdr)
        body = b""
        while len(body) < body_len:
            body += s.recv(body_len - len(body))
        s.close()
        if status != 0:
            return {"status": "error"}
        result = {"status": "ok"}
        for pair in body.decode().split():
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    if "/" in v:
                        result[k] = v
                    elif "." in v:
                        result[k] = float(v)
                    else:
                        result[k] = int(v)
                except ValueError:
                    result[k] = v
        return result
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


def _get_vstore_stats() -> dict:
    """Get V-Store stats from Pion via V.INFO."""
    try:
        resp = _pion.execute_command("V.INFO")
        if resp and isinstance(resp, bytes):
            text = resp.decode()
            result = {}
            for line in text.split("\r\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        result[k.strip()] = int(v.strip())
                    except ValueError:
                        result[k.strip()] = v.strip()
            return result
        return {"sessions": 0}
    except Exception:
        return {"sessions": 0}


@app.route("/health", methods=["GET"])
def health():
    """Health check — Pion RESP, backend, and optionally MLX sidecar."""
    pion_ok = False
    backend_ok = False
    try:
        _pion.ping()
        pion_ok = True
    except Exception:
        pass
    try:
        backend_url = _config.get("backend_url", "http://127.0.0.1:11434")
        requests.get(backend_url, timeout=2)
        backend_ok = True
    except Exception:
        pass
    result = {"pion": pion_ok, "backend": backend_ok}
    mlx_port = _config.get("mlx_tcp_port", 0)
    if mlx_port and mlx_port > 0:
        mlx_stats = _get_mlx_stats(_config.get("mlx_tcp_host", "127.0.0.1"), mlx_port)
        result["mlx_gpu"] = mlx_stats.get("status") == "ok"
    return jsonify(result)


# ── Confidence Decay + Feedback ──────────────────────────────────────────────

_last_l3_response: dict = {}  # {user_query_hash: (concept_ids, timestamp)}


def _start_decay_timer():
    """Start background thread for daily confidence decay."""
    import threading

    def _decay_loop():
        while True:
            time.sleep(86400)  # 24 hours
            try:
                if _concept_store:
                    _concept_store.decay_confidence(0.99)
                    log.info(f"L3 decay applied to {_concept_store.stats['concepts']} concepts")
            except Exception as e:
                log.debug(f"Decay failed: {e}")

    t = threading.Thread(target=_decay_loop, daemon=True)
    t.start()
    log.info("L3 confidence decay timer started (24h interval)")


def _track_l3_response(user_query: str, concept_ids: list[int]):
    """Track an L3 response for feedback detection. gh #84: mutation under
    `_l3_response_lock` — both `__setitem__` and the eviction sweep are
    concurrent with `_check_negative_feedback`, and a bare `del` mid-iteration
    in another thread would raise `RuntimeError: dict changed size`."""
    qhash = hashlib.md5(user_query.encode()).hexdigest()[:16]
    now = time.time()
    with _l3_response_lock:
        _last_l3_response[qhash] = (concept_ids, now)
        # Evict old entries (we already hold the lock, so iteration is safe).
        stale = [k for k, (_, ts) in _last_l3_response.items() if now - ts > 60]
        for k in stale:
            del _last_l3_response[k]


def _check_negative_feedback(user_query: str):
    """Check if this query is a follow-up correction to an L3 response.

    Heuristic: if the user sends a very similar query within 30s of an L3
    response, it's likely a correction (negative feedback). gh #84: take a
    coherent snapshot of `_last_l3_response` items under the lock, then act
    outside — `_concept_store.record_feedback` may do I/O we don't want to
    serialise behind the dict lock."""
    with _l3_response_lock:
        if not _last_l3_response or not _concept_store:
            return
        snapshot = list(_last_l3_response.items())

    now = time.time()
    for qhash, (concept_ids, ts) in snapshot:
        if now - ts < 30:
            # Recent L3 response exists — this follow-up is likely negative.
            _concept_store.record_feedback(concept_ids, positive=False)
            with _l3_response_lock:
                _last_l3_response.pop(qhash, None)
            log.info(f"L3 negative feedback: concepts {concept_ids}")
            return


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global _config, _pion, _concept_store, _fragment_store, _intent_router

    # ── `setup` subcommand routing (gh #23 productize) ──────────────────────
    # Detected as the first positional arg so the existing serve mode's
    # argparse stays untouched. `pion-serve setup --corpus <dir>` runs the
    # cascade onboarding flow and exits.
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        return _setup_command(sys.argv[2:])

    parser = argparse.ArgumentParser(description="Pion Serve — inference intelligence layer")
    parser.add_argument("--backend", default="ollama", choices=list(BACKENDS.keys()),
                        help="Backend type (default: ollama)")
    parser.add_argument("--backend-url", default=None,
                        help="Backend URL override")
    parser.add_argument("--model", default="llama3.2:3b",
                        help="Default model (default: llama3.2:3b)")
    parser.add_argument("--pion-host", default="127.0.0.1")
    parser.add_argument("--pion-port", type=int, default=1974)
    parser.add_argument("--port", type=int, default=8321,
                        help="Serve port (default: 8321)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable semantic cache")
    parser.add_argument("--cache-threshold", type=float, default=0.85,
                        help="Semantic cache cosine threshold (default: 0.85)")
    parser.add_argument("--rag-index", default=None,
                        help="Pion FT index name for RAG context injection")
    parser.add_argument("--rag-k", type=int, default=3,
                        help="Number of RAG documents to inject (default: 3)")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434",
                        help="Ollama URL for embeddings (default: http://127.0.0.1:11434)")
    # gh #69: SIE (superlinked/sie) embedding backend. Opt-in via --sie-url
    # because port 8000 is shared with too many other tools to auto-detect
    # safely. When set, slots in BEFORE Ollama in the cascade.
    parser.add_argument("--embed-backend", default="auto",
                        choices=["auto", "pion", "sie", "ollama", "none"],
                        help="Embedding backend selection. 'auto' = cascade "
                             "(pion → sie if --sie-url → ollama). Explicit choices "
                             "do not fall through. Default: auto.")
    parser.add_argument("--sie-url", default=None,
                        help="SIE (superlinked/sie) base URL for embeddings, e.g. http://localhost:8000. "
                             "When set, used in the cascade before Ollama. Default: disabled.")
    parser.add_argument("--sie-model", default="BAAI/bge-small-en-v1.5",
                        help="Model name to send to SIE /v1/embeddings (default: BAAI/bge-small-en-v1.5)")
    parser.add_argument("--distill", action="store_true",
                        help="Enable L3 inference distillation (concept synthesis)")
    parser.add_argument("--l3-direct-threshold", type=float, default=0,
                        help="L3 direct synthesis cosine threshold (0 = auto-calibrate)")
    parser.add_argument("--l3-composite-threshold", type=float, default=0,
                        help="L3 composite synthesis cosine threshold (0 = auto-calibrate)")
    parser.add_argument("--mlx-tcp-host", default="127.0.0.1",
                        help="MLX attention sidecar host (default: 127.0.0.1)")
    parser.add_argument("--mlx-tcp-port", type=int, default=0,
                        help="MLX attention sidecar TCP port (0 = disabled)")
    # pion-moe in-process backend (MoE-architecture LLMs served via Pion's
    # MOE.EXPERT.* substrate). Lazy-init at first request.
    parser.add_argument("--pion-moe-host", default="127.0.0.1",
                        help="pion-server host serving the --moe-cache tier")
    parser.add_argument("--pion-moe-port", type=int, default=1974,
                        help="pion-server port serving the --moe-cache tier")
    parser.add_argument("--pion-moe-model-id", default=None,
                        help="model_id (snapshot basename) registered on the pion-server tier")
    parser.add_argument("--pion-moe-model-path", default=None,
                        help="absolute path to the MoE safetensors snapshot dir (must match --pion-moe-model-id)")
    # pion-cag-hybrid in-process backend (CAG+RAG entropy/margin-gated hybrid,
    # gh #23 productization). Lazy-init at first request.
    parser.add_argument("--cag-mlx-model", default="mlx-community/Llama-3.1-8B-Instruct-4bit",
                        help="MLX model ID for the CAG-hybrid backend")
    parser.add_argument("--cag-foundation-corpus", default=None,
                        help="Path to a text file containing the bounded foundation corpus "
                             "(paragraphs separated by blank lines). Required for pion-cag-hybrid.")
    parser.add_argument("--cag-calibration-qa", default=None,
                        help="Path to a JSONL file with calibration questions "
                             "({question, answers: [gold...]} per line). Required for pion-cag-hybrid "
                             "unless --cag-calibration-state is provided.")
    parser.add_argument("--cag-calibration-state", default=None,
                        help="Path to a calibration state JSON written by `pion-serve setup`. "
                             "When set, (Te, Tm) + cluster are loaded from the state file and "
                             "the per-startup calibration sweep is skipped (only the foundation "
                             "preload still happens). Takes precedence over --cag-calibration-qa.")
    parser.add_argument("--cag-cluster", default="A", choices=["A", "B"],
                        help="Threshold cluster to bias toward during calibration: "
                             "A = loose entropy + strict margin (lower fallback); "
                             "B = strict entropy + loose margin (tighter F1). Default: A.")
    parser.add_argument("--cag-rag-k", type=int, default=3,
                        help="Top-k chunks for the CAG-hybrid's RAG fallback path (default: 3)")
    parser.add_argument("--cag-n-gen", type=int, default=20,
                        help="Max generation tokens for both CAG and RAG branches (default: 20)")
    parser.add_argument("--route", action="store_true",
                        help="Enable semantic intent routing (classify queries → tier-specific model)")
    parser.add_argument("--route-config", default=None,
                        help="JSON file with per-tier routing override (simple/medium/complex → backend+model)")
    parser.add_argument("--route-margin", type=float, default=0.05,
                        help="Centroid cosine margin to commit to a tier (default: 0.05)")
    args = parser.parse_args()

    backend_url = args.backend_url or BACKENDS[args.backend]["url"]

    _config = {
        "backend_type": args.backend,
        "backend_url": backend_url,
        "model": args.model,
        "cache_enabled": not args.no_cache,
        "cache_threshold": args.cache_threshold,
        "distill_enabled": args.distill,
        "rag_index": args.rag_index,
        "rag_k": args.rag_k,
        "ollama_url": args.ollama_url,
        "sie_url": args.sie_url,
        "sie_model": args.sie_model,
        "embed_backend_requested": args.embed_backend,
        "mlx_tcp_host": args.mlx_tcp_host,
        "mlx_tcp_port": args.mlx_tcp_port,
        "pion_moe_host": args.pion_moe_host,
        "pion_moe_port": args.pion_moe_port,
        "cag_mlx_model": args.cag_mlx_model,
        "cag_foundation_corpus": args.cag_foundation_corpus,
        "cag_calibration_qa": args.cag_calibration_qa,
        "cag_calibration_state": args.cag_calibration_state,
        "cag_cluster": args.cag_cluster,
        "cag_rag_k": args.cag_rag_k,
        "cag_n_gen": args.cag_n_gen,
        "pion_moe_model_id": args.pion_moe_model_id,
        "pion_moe_model_path": args.pion_moe_model_path,
    }

    # Connect to Pion
    try:
        _pion = redis.Redis(host=args.pion_host, port=args.pion_port, decode_responses=False)
        _pion.ping()
        log.info(f"Connected to Pion at {args.pion_host}:{args.pion_port}")
    except Exception as e:
        log.warning(f"Pion not available ({e}) — caching disabled")
        _config["cache_enabled"] = False

    # Initialize embedder
    _init_embedder(provider=args.embed_backend)

    # Initialize intent router (if --route)
    if args.route:
        if _embed_fn is None:
            log.error("--route requires an embedding provider (Pion auto-embed or Ollama nomic-embed-text)")
            sys.exit(1)
        routing = DEFAULT_ROUTING
        if args.route_config:
            try:
                routing = load_routing_config(args.route_config)
                log.info(f"Loaded routing config from {args.route_config}")
            except Exception as e:
                log.error(f"Failed to load --route-config: {e}")
                sys.exit(1)
        _intent_router = IntentRouter(
            embed_fn=_embed_fn,
            routing=routing,
            margin=args.route_margin,
        )
        if not _intent_router.bootstrap():
            log.error("Intent router bootstrap failed (no embeddings) — disabling")
            _intent_router = None
        else:
            log.info(
                "Intent routing: simple→%s/%s, medium→%s/%s, complex→%s/%s",
                routing["simple"]["backend"],  routing["simple"]["model"],
                routing["medium"]["backend"],  routing["medium"]["model"],
                routing["complex"]["backend"], routing["complex"]["model"],
            )

    # Initialize L3 concept store (if --distill)
    if args.distill:
        # Auto-calibrate thresholds if not explicitly set
        if args.l3_direct_threshold == 0 and _embed_fn:
            cal_pairs = [
                ("What is a hash map?", "How does a hash map work?"),
                ("What is the thread model?", "Describe the thread model."),
                ("How do you build the server?", "What are the build instructions?"),
            ]
            sims = []
            for q1, q2 in cal_pairs:
                e1, e2 = _embed_fn(q1), _embed_fn(q2)
                if e1 is not None and e2 is not None:
                    sims.append(float(np.dot(e1 / (np.linalg.norm(e1) + 1e-10),
                                             e2 / (np.linalg.norm(e2) + 1e-10))))
            if sims:
                median_sim = sorted(sims)[len(sims) // 2]
                args.l3_direct_threshold = round(max(0.65, min(0.92, median_sim - 0.02)), 2)
                args.l3_composite_threshold = round(max(0.55, args.l3_direct_threshold - 0.08), 2)
                log.info(f"L3 auto-calibrated: direct={args.l3_direct_threshold}, "
                         f"composite={args.l3_composite_threshold} "
                         f"(median paraphrase sim={median_sim:.3f})")
            else:
                args.l3_direct_threshold = 0.80
                args.l3_composite_threshold = 0.72
        elif args.l3_direct_threshold == 0:
            args.l3_direct_threshold = 0.80
            args.l3_composite_threshold = 0.72

        _concept_store = ConceptStore(
            pion_host=args.pion_host,
            pion_port=args.pion_port,
            direct_threshold=args.l3_direct_threshold,
            composite_threshold=args.l3_composite_threshold,
        )
        _fragment_store = FragmentStore(
            pion_host=args.pion_host,
            pion_port=args.pion_port,
            embed_fn=_embed_fn,
            batch_embed_fn=_batch_embed_via_ollama if _embed_fn == _embed_via_ollama else None,
        )
        _start_decay_timer()
        log.info(
            f"L3 distillation enabled "
            f"({_concept_store.stats['concepts']} concepts, "
            f"{_fragment_store.stats['fragments']} fragments)"
        )

    # Banner
    print()
    print("=" * 60)
    print("  Pion Serve — inference intelligence layer")
    print("=" * 60)
    print(f"  Backend:  {args.backend} @ {backend_url}")
    print(f"  Model:    {args.model}")
    print(f"  Cache:    {'enabled' if _config['cache_enabled'] else 'disabled'}"
          f" (threshold={args.cache_threshold})")
    print(f"  Distill:  {'enabled (L3 concept synthesis)' if args.distill else 'disabled'}")
    if _intent_router is not None:
        r = _intent_router.routing
        print(f"  Route:    enabled — simple={r['simple']['model']}, "
              f"medium={r['medium']['model']}, complex={r['complex']['model']}")
    else:
        print("  Route:    disabled")
    print(f"  RAG:      {args.rag_index or 'disabled'}")
    active_embed = _config.get("embed_backend", "none")
    if active_embed == "sie":
        print(f"  Embed:    sie @ {args.sie_url} ({args.sie_model})")
    elif active_embed == "none" and args.embed_backend != "auto":
        print(f"  Embed:    {args.embed_backend} requested but unavailable (semantic cache off)")
    else:
        print(f"  Embed:    {active_embed}")
    print(f"  Pion:     {args.pion_host}:{args.pion_port}")
    print(f"  Serving:  http://0.0.0.0:{args.port}/v1/chat/completions")
    print("=" * 60)
    print()

    # Check port is free before starting
    import socket as _sock
    _test = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    try:
        _test.bind(("0.0.0.0", args.port))
        _test.close()
    except OSError:
        log.error(f"FATAL: port {args.port} is already in use")
        sys.exit(1)

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
