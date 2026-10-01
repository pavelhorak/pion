"""
Pion FLARE AI Gateway

An OpenAI-compatible HTTP proxy that adds mid-generation retrieval (FLARE)
to any LLM backend. Transparently intercepts streaming token output, monitors
logprobs, and when confidence drops below τ, queries Pion for grounding
context before regenerating the uncertain chunk.

Architecture:
  Client → FLARE Gateway (:8080) → Upstream LLM (Ollama / vLLM / OpenAI)
                                 ↕
                            Pion (:1974)  [1.28ms retrieval]

Drop-in replacement: swap your LLM base URL from the upstream to the gateway.
The client API is fully OpenAI-compatible (/v1/chat/completions, /v1/completions).

Usage:
    pip install flask requests redis numpy onnxruntime transformers

    # With Ollama upstream:
    python flare_gateway/gateway.py --upstream-type ollama

    # With vLLM upstream:
    python flare_gateway/gateway.py --upstream-type vllm

    # With OpenAI:
    OPENAI_API_KEY=sk-... python flare_gateway/gateway.py --upstream-type openai

Environment:
    UPSTREAM_TYPE       Backend preset: ollama, llama-cpp, vllm, exo, openai (default: ollama)
    UPSTREAM_URL        LLM backend base URL (default: http://localhost:11434)
    UPSTREAM_MODEL      Model name to use (default: llama3.1:8b)
    PION_HOST           Pion server host (default: localhost)
    PION_PORT           Pion server port (default: 1974)
    PION_INDEX          Vector index to search (default: flare_kb)
    FLARE_TAU           Logprob confidence threshold 0-1 (default: 0.3)
    FLARE_CHUNK_TOKENS  Tokens per generation chunk (default: 10)
    FLARE_MAX_RETRIES   Max retrieval attempts per chunk (default: 2)
    GATEWAY_PORT        Port to listen on (default: 8080)
    EMBED_MODEL_PATH    Path to MiniLM ONNX model (default: models/all-MiniLM-L6-v2/onnx/model.onnx)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Generator, Iterator
from urllib.parse import urlparse

import numpy as np
import redis
import requests
from flask import Flask, Response, jsonify, request, stream_with_context

# ── Configuration ─────────────────────────────────────────────────────────────

_UPSTREAM_DEFAULTS: dict[str, dict[str, object]] = {
    "ollama": {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": 11434,
        "path": "/api/generate",
    },
    "llama-cpp": {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": 8080,
        "path": "/v1/chat/completions",
    },
    "vllm": {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": 8000,
        "path": "/v1/chat/completions",
    },
    "exo": {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": 52415,
        "path": "/v1/chat/completions",
    },
    "openai": {
        "scheme": "https",
        "host": "api.openai.com",
        "port": 443,
        "path": "/v1/chat/completions",
    },
}

UPSTREAM_TYPE   = "ollama"
UPSTREAM_SCHEME = "http"
UPSTREAM_HOST   = "127.0.0.1"
UPSTREAM_PORT   = 11434
UPSTREAM_PATH   = "/api/generate"
UPSTREAM_URL    = "http://127.0.0.1:11434"
UPSTREAM_ENDPOINT = "http://127.0.0.1:11434/api/generate"

UPSTREAM_MODEL  = os.environ.get("UPSTREAM_MODEL", "llama3.1:8b")
PION_HOST       = os.environ.get("PION_HOST", "localhost")
PION_PORT       = int(os.environ.get("PION_PORT", "1974"))
PION_INDEX      = os.environ.get("PION_INDEX", "flare_kb")
FLARE_TAU       = float(os.environ.get("FLARE_TAU", "0.3"))
CHUNK_TOKENS    = int(os.environ.get("FLARE_CHUNK_TOKENS", "10"))
MAX_RETRIES     = int(os.environ.get("FLARE_MAX_RETRIES", "2"))
GATEWAY_PORT    = int(os.environ.get("GATEWAY_PORT", "8080"))
EMBED_PATH      = os.environ.get("EMBED_MODEL_PATH", "models/all-MiniLM-L6-v2/onnx/model.onnx")
# EMBED_PROVIDER: onnx (384-dim MiniLM, requires onnxruntime+transformers),
#                 openai (1536-dim, requires OPENAI_API_KEY),
#                 mock (1536-dim deterministic, no deps — use for smoke tests)
EMBED_PROVIDER  = os.environ.get("EMBED_PROVIDER", "onnx")
# EMBED_DIM auto-set: 384 for onnx, 1536 for openai/mock. Must match Pion's config.vector.dimensions.
EMBED_DIM       = int(os.environ.get("EMBED_DIM", "384" if EMBED_PROVIDER == "onnx" else "1536"))
# Key prefix for FLARE KB docs stored in Pion (integer suffix required for HNSW routing)
_FLARE_PREFIX   = "fl:"
_FLARE_SEQ_KEY  = "__flare_seq__"
# Minimum nodes in HNSW before FT.SEARCH is safe (batch-8 kernel reads 8 nodes at once)
_MIN_HNSW_NODES = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("flare-gateway")

app = Flask(__name__)

def _normalize_path(path: str) -> str:
    if not path:
        return "/"
    return path if path.startswith("/") else f"/{path}"

def _build_base_url(scheme: str, host: str, port: int) -> str:
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"

def _init_config(args: argparse.Namespace | None = None) -> None:
    global UPSTREAM_TYPE, UPSTREAM_SCHEME, UPSTREAM_HOST, UPSTREAM_PORT
    global UPSTREAM_PATH, UPSTREAM_URL, UPSTREAM_ENDPOINT

    env_type = os.environ.get("UPSTREAM_TYPE")
    upstream_type = (
        args.upstream_type if args and args.upstream_type else (env_type or "ollama")
    )
    if upstream_type not in _UPSTREAM_DEFAULTS:
        log.warning("Unknown upstream type '%s', falling back to 'ollama'", upstream_type)
        upstream_type = "ollama"

    defaults = _UPSTREAM_DEFAULTS[upstream_type]
    scheme = str(defaults["scheme"])
    host = str(defaults["host"])
    port = int(defaults["port"])
    path = str(defaults["path"])

    env_url = os.environ.get("UPSTREAM_URL")
    if env_url and not (args and args.upstream_type):
        parsed = urlparse(env_url)
        if parsed.scheme:
            scheme = parsed.scheme
        if parsed.hostname:
            host = parsed.hostname
        if parsed.port:
            port = parsed.port
        if parsed.path and parsed.path != "/":
            path = parsed.path

    if args:
        if args.upstream_host:
            host = args.upstream_host
        if args.upstream_port:
            port = args.upstream_port
        if args.upstream_path:
            path = args.upstream_path

    path = _normalize_path(path)
    base_url = _build_base_url(scheme, host, port)

    UPSTREAM_TYPE = upstream_type
    UPSTREAM_SCHEME = scheme
    UPSTREAM_HOST = host
    UPSTREAM_PORT = port
    UPSTREAM_PATH = path
    UPSTREAM_URL = base_url
    UPSTREAM_ENDPOINT = f"{base_url}{path}"

_init_config()

# ── Embedding (in-process) ────────────────────────────────────────────────────

_emb_tok: Any = None
_emb_sess: Any = None


def _load_embedder() -> None:
    """Load MiniLM ONNX embedder (only when EMBED_PROVIDER=onnx)."""
    global _emb_tok, _emb_sess
    if _emb_sess is None:
        from transformers import AutoTokenizer  # type: ignore
        import onnxruntime as ort               # type: ignore
        log.info("Loading MiniLM embedder from %s", EMBED_PATH)
        _emb_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _emb_sess = ort.InferenceSession(EMBED_PATH, providers=["CPUExecutionProvider"])
        log.info("Embedder ready (384-dim)")


def _embed_onnx(text: str) -> bytes:
    """MiniLM ONNX embed — 384-dim, ~1.2ms in-process."""
    _load_embedder()
    enc = _emb_tok(text, return_tensors="np", padding=True,
                   truncation=True, max_length=128)
    out = _emb_sess.run(None, {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc.get("token_type_ids", np.zeros_like(enc["input_ids"])),
    })
    mask = enc["attention_mask"][:, :, np.newaxis].astype(np.float32)
    emb = (out[0] * mask).sum(1) / mask.sum(1).clip(min=1e-9)
    v = emb[0]
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32).tobytes()


def _embed_openai(text: str) -> bytes:
    """OpenAI text-embedding-3-small — 1536-dim. Requires OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        json={"input": text, "model": "text-embedding-3-small"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    r.raise_for_status()
    v = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32).tobytes()


def _embed_mock(text: str) -> bytes:
    """Deterministic mock embed — 1536-dim unit vector seeded from text hash.

    Same text → same vector, so exact-match recall works in smoke tests.
    No semantic meaning (random), but validates the full FLARE pipeline.
    """
    import hashlib
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)
    rng = np.random.RandomState(seed)
    v = rng.randn(1536).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-9)).tobytes()


def embed(text: str) -> bytes:
    """Return float32 vector bytes for Pion FT.SEARCH. Provider: EMBED_PROVIDER."""
    if EMBED_PROVIDER == "openai":
        return _embed_openai(text)
    elif EMBED_PROVIDER == "mock":
        return _embed_mock(text)
    else:
        return _embed_onnx(text)

# ── Pion retrieval ─────────────────────────────────────────────────────────────

_pion: redis.Redis | None = None

def _conn() -> redis.Redis:
    global _pion
    if _pion is None:
        _pion = redis.Redis(host=PION_HOST, port=PION_PORT,
                            decode_responses=False, socket_keepalive=True)
    return _pion

def pion_retrieve(query_text: str, k: int = 3) -> list[str]:
    """Query Pion and return top-k text snippets. Returns [] on any error."""
    t0 = time.perf_counter()
    try:
        r = _conn()
        # Check index exists
        try:
            r.execute_command("FT.INFO", PION_INDEX)
        except Exception:
            return []

        vec_bytes = embed(query_text)

        # Check total loaded docs — fall back to linear scan when < 8
        # (HNSW batch-8 kernel reads out-of-bounds with fewer nodes)
        total_raw = r.get(_FLARE_SEQ_KEY)
        total_count = int(total_raw) if total_raw else 0

        results: list[str] = []

        if total_count < _MIN_HNSW_NODES:
            # Linear scan: iterate fl:1..fl:N, compute cosine similarity in Python
            query_vec = np.frombuffer(vec_bytes, dtype=np.float32)
            candidates: list[tuple[float, str]] = []
            for idx in range(1, total_count + 1):
                key = f"{_FLARE_PREFIX}{idx}"
                emb_raw = r.hget(key, "embedding")
                text_raw = r.hget(key, "text")
                if emb_raw is None or text_raw is None:
                    continue
                doc_vec = np.frombuffer(emb_raw, dtype=np.float32)
                denom = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-9
                sim = float(np.dot(query_vec, doc_vec) / denom)
                text = text_raw.decode() if isinstance(text_raw, bytes) else str(text_raw)
                candidates.append((sim, text))
            candidates.sort(key=lambda x: x[0], reverse=True)
            results = [t for _, t in candidates[:k]]
        else:
            # HNSW search via Pion FT.SEARCH with PARAMS format
            raw = r.execute_command(
                "FT.SEARCH", PION_INDEX,
                f"*=>[KNN {k} @embedding $vec EF_RUNTIME 64]",
                "PARAMS", "2", "vec", vec_bytes,
            )
            if not raw or not isinstance(raw, list):
                return []

            items = list(raw)
            i = 1 if (items and _is_int(items[0])) else 0
            while i + 1 < len(items):
                # Pion returns integer node ID as doc key
                doc_id_raw = items[i]
                doc_id_str = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else str(doc_id_raw)
                # fields_raw = [b"id", id_val, b"score", score_val], or just
                # [b"score", score_val] when the doc has no "id" field on the
                # answering worker (gh #357) — look fields up by name.
                i += 2
                # Pion returns the original key; prefixing it again missed every HGET.
                doc_key = doc_id_str if doc_id_str.startswith(_FLARE_PREFIX) else _FLARE_PREFIX + doc_id_str
                text_raw = r.hget(doc_key, "text")
                if text_raw:
                    results.append(text_raw.decode() if isinstance(text_raw, bytes) else str(text_raw))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug("Pion retrieval: %d results in %.1fms", len(results), elapsed_ms)
        return results
    except Exception as e:
        log.warning("Pion retrieval failed: %s", e)
        return []

def _is_int(v: Any) -> bool:
    try:
        int(v)
        return True
    except (TypeError, ValueError):
        return False

# ── Upstream LLM calls ────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, max_tokens: int, logprobs: bool = True) -> dict:
    """Single non-streaming call to Ollama /api/generate."""
    payload = {
        "model": UPSTREAM_MODEL,
        "prompt": prompt,
        "stream": False,
        "logprobs": logprobs,  # top-level for Ollama ≥0.5 (not inside options)
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
        },
    }
    r = requests.post(UPSTREAM_ENDPOINT, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def _ollama_chat_generate(messages: list[dict], max_tokens: int, logprobs: bool = True) -> dict:
    """Single non-streaming call to Ollama /api/chat."""
    payload = {
        "model": UPSTREAM_MODEL,
        "messages": messages,
        "stream": False,
        "logprobs": logprobs,  # top-level for Ollama ≥0.5 (not inside options)
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
        },
    }
    r = requests.post(UPSTREAM_ENDPOINT, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def _openai_generate(messages: list[dict], max_tokens: int) -> dict:
    """Single non-streaming call to OpenAI-compatible /v1/chat/completions."""
    headers = {}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": UPSTREAM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 1,
        "stream": False,
    }
    r = requests.post(UPSTREAM_ENDPOINT, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()

# ── FLARE core logic ──────────────────────────────────────────────────────────

FEW_SHOT_PREFIX = (
    "Q: Who wrote Hamlet?\nA: William Shakespeare\n\n"
    "Q: What is the capital of Japan?\nA: Tokyo\n\n"
    "Q: In what year did World War II end?\nA: 1945\n\n"
)

def _build_prompt(system: str, user_query: str,
                  context_facts: list[str], generated_so_far: str) -> str:
    """Build a FLARE prompt with retrieved context prepended."""
    ctx = ""
    if context_facts:
        ctx = "Context:\n" + "\n".join(f"- {f}" for f in context_facts) + "\n\n"
    return f"{system}\n\n{ctx}Question: {user_query}\nAnswer: {generated_so_far}"

def _build_messages(system: str, user_query: str,
                    context_facts: list[str], generated_so_far: str) -> list[dict]:
    """Build OpenAI-style messages for FLARE generation."""
    ctx = ""
    if context_facts:
        ctx = "Context:\n" + "\n".join(f"- {f}" for f in context_facts) + "\n\n"
    user_content = f"{ctx}Question: {user_query}\nAnswer: {generated_so_far}"
    messages: list[dict] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    return messages

def _use_ollama_generate() -> bool:
    return UPSTREAM_TYPE == "ollama" and UPSTREAM_PATH.endswith("/api/generate")

def _extract_logprobs(logprobs_raw: Any) -> list[float]:
    if isinstance(logprobs_raw, list):
        out: list[float] = []
        for entry in logprobs_raw:
            if isinstance(entry, dict):
                lp = entry.get("logprob", None)
            elif isinstance(entry, (int, float)):
                lp = float(entry)
            else:
                lp = None
            if lp is not None:
                out.append(float(lp))
        return out

    if isinstance(logprobs_raw, dict):
        candidates: list[Any] = []
        if isinstance(logprobs_raw.get("content"), list):
            candidates = logprobs_raw.get("content", [])
        elif isinstance(logprobs_raw.get("token_logprobs"), list):
            candidates = logprobs_raw.get("token_logprobs", [])
        out = []
        for entry in candidates:
            if isinstance(entry, dict):
                lp = entry.get("logprob", None)
            elif isinstance(entry, (int, float)):
                lp = float(entry)
            else:
                lp = None
            if lp is not None:
                out.append(float(lp))
        return out

    return []

def _extract_chunk(resp: dict) -> tuple[str, list[float], bool]:
    """Normalize upstream responses into (chunk, logprobs, done)."""
    if "response" in resp:
        chunk = resp.get("response", "") or ""
        logprobs = _extract_logprobs(resp.get("logprobs", None))
        done = bool(resp.get("done", False))
        return chunk, logprobs, done

    choices = resp.get("choices", [])
    if choices:
        choice0 = choices[0] if isinstance(choices, list) else {}
        chunk = ""
        if isinstance(choice0, dict):
            msg = choice0.get("message", {})
            if isinstance(msg, dict):
                chunk = msg.get("content", "") or ""
            if not chunk:
                chunk = choice0.get("text", "") or ""
        logprobs = _extract_logprobs(
            choice0.get("logprobs", None) if isinstance(choice0, dict) else None
        )
        done = False
        if isinstance(choice0, dict):
            finish_reason = choice0.get("finish_reason")
            done = finish_reason in ("stop", "length")
        return chunk, logprobs, done

    return "", [], True

def flare_generate(system_prompt: str, user_query: str,
                   max_tokens: int = 200) -> tuple[str, int, int]:
    """
    Run FLARE generation loop.

    Returns:
        (final_text, total_retrievals, total_chunks)
    """
    generated = ""
    retrieved_facts: list[str] = []
    total_retrievals = 0
    total_chunks = 0
    tokens_generated = 0

    while tokens_generated < max_tokens:
        remaining = max_tokens - tokens_generated
        chunk_size = min(CHUNK_TOKENS, remaining)
        if chunk_size <= 0:
            break

        prompt = _build_prompt(system_prompt, user_query, retrieved_facts, generated)
        messages = _build_messages(system_prompt, user_query, retrieved_facts, generated)

        # Generate a chunk
        try:
            if UPSTREAM_TYPE == "ollama":
                if _use_ollama_generate():
                    resp = _ollama_generate(prompt, max_tokens=chunk_size)
                else:
                    resp = _ollama_chat_generate(messages, max_tokens=chunk_size)
            else:
                resp = _openai_generate(messages, max_tokens=chunk_size)
        except Exception as e:
            log.error("Upstream LLM error: %s", e)
            break

        chunk, logprobs_list, done = _extract_chunk(resp)
        if not chunk:
            break

        total_chunks += 1
        tokens_generated += len(chunk.split())  # approximation

        # Extract logprobs if available
        logprobs_raw = logprobs_list
        min_prob = 1.0
        if logprobs_raw:
            probs = [math.exp(float(lp)) for lp in logprobs_raw if lp is not None]
            if probs:
                min_prob = min(probs)

        is_uncertain = (min_prob < FLARE_TAU) and chunk.strip()

        if is_uncertain:
            # FLARE trigger: retrieve and regenerate
            draft_query = (generated + " " + chunk).strip()
            log.info("FLARE trigger (min_prob=%.3f) | query: %s...",
                     min_prob, draft_query[:60])
            new_facts = pion_retrieve(draft_query, k=2)
            total_retrievals += 1

            genuinely_new = [f for f in new_facts if f not in retrieved_facts]
            if genuinely_new:
                retrieved_facts.extend(genuinely_new)
                # Discard uncertain chunk, loop regenerates with new context
                log.info("Retrieved %d new facts, regenerating chunk", len(genuinely_new))
                continue
            # No new facts found — accept the chunk anyway to avoid loop
            log.debug("No new facts found, accepting uncertain chunk")

        generated += chunk

        # Stop on natural end
        if done:
            break

    return generated.strip(), total_retrievals, total_chunks

# ── Gateway routes ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for load balancers."""
    return jsonify({
        "status": "ok",
        "upstream": UPSTREAM_ENDPOINT,
        "pion": f"{PION_HOST}:{PION_PORT}",
        "upstream_type": UPSTREAM_TYPE,
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint with FLARE augmentation."""
    body = request.get_json(force=True)
    messages: list[dict] = body.get("messages", [])
    max_tokens: int = body.get("max_tokens", 200)
    stream: bool = body.get("stream", False)

    # Extract system prompt and user query
    system_prompt = next(
        (m["content"] for m in messages if m.get("role") == "system"),
        "You are a helpful assistant. Answer questions concisely."
    )
    # Find last user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return jsonify({"error": "No user message found"}), 400
    user_query = user_messages[-1]["content"]

    t0 = time.perf_counter()
    text, retrievals, chunks = flare_generate(system_prompt, user_query, max_tokens)
    elapsed = time.perf_counter() - t0

    log.info("FLARE complete: %d chars, %d retrievals, %d chunks, %.2fs",
             len(text), retrievals, chunks, elapsed)

    # Return OpenAI-compatible response
    response_body = {
        "id": f"flare-{int(time.time())}",
        "object": "chat.completion",
        "model": UPSTREAM_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(user_query.split()),
            "completion_tokens": len(text.split()),
            "total_tokens": len(user_query.split()) + len(text.split()),
        },
        "x_flare": {
            "retrievals": retrievals,
            "chunks": chunks,
            "elapsed_s": round(elapsed, 3),
            "tau": FLARE_TAU,
            "pion_index": PION_INDEX,
        },
    }

    if stream:
        # Streaming: emit the full response as a single SSE chunk
        def generate_sse() -> Generator[str, None, None]:
            chunk_data = {
                "id": response_body["id"],
                "object": "chat.completion.chunk",
                "model": UPSTREAM_MODEL,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            done_data = {
                "id": response_body["id"],
                "object": "chat.completion.chunk",
                "model": UPSTREAM_MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_data)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate_sse()),
                        mimetype="text/event-stream")

    return jsonify(response_body)


@app.route("/v1/completions", methods=["POST"])
def completions():
    """OpenAI-compatible text completions endpoint with FLARE augmentation."""
    body = request.get_json(force=True)
    prompt: str = body.get("prompt", "")
    max_tokens: int = body.get("max_tokens", 200)

    t0 = time.perf_counter()
    text, retrievals, chunks = flare_generate("", prompt, max_tokens)
    elapsed = time.perf_counter() - t0

    return jsonify({
        "id": f"flare-{int(time.time())}",
        "object": "text_completion",
        "model": UPSTREAM_MODEL,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "x_flare": {"retrievals": retrievals, "chunks": chunks,
                    "elapsed_s": round(elapsed, 3)},
    })


@app.route("/flare/load", methods=["POST"])
def flare_load():
    """Load documents into the FLARE knowledge base.

    POST /flare/load
    Body: { "documents": [{"id": "doc:1", "text": "..."}, ...] }

    Bulk-embeds and indexes documents into Pion under PION_INDEX.
    Call this once to populate the KB; documents persist across restarts.
    """
    body = request.get_json(force=True)
    docs: list[dict] = body.get("documents", [])
    if not docs:
        return jsonify({"error": "No documents provided"}), 400

    r = _conn()

    # Ensure index — use "embedding" field name to avoid collision with VECTOR keyword
    try:
        r.execute_command("FT.INFO", PION_INDEX)
    except Exception:
        r.execute_command(
            "FT.CREATE", PION_INDEX,
            "SCHEMA", "embedding", "VECTOR", "HNSW",
            "10", "TYPE", "FLOAT32", "DIM", str(EMBED_DIM),
            "DISTANCE_METRIC", "COSINE", "M", "16", "EF_CONSTRUCTION", "128",
        )

    # Filter valid docs first
    valid_docs = [d for d in docs if d.get("text", "").strip()]
    if not valid_docs:
        return jsonify({"loaded": 0, "index": PION_INDEX})

    # Sequential integer keys required for HNSW routing in Pion's fast-path HSET handler
    # (Pion extracts digits from the key name for the HNSW node ID)
    loaded = 0
    for doc in valid_docs:
        text = doc["text"].strip()
        seq_id = int(r.execute_command("INCR", _FLARE_SEQ_KEY))
        key = f"{_FLARE_PREFIX}{seq_id}"
        vec = embed(text)
        r.hset(key, mapping={"text": text, "embedding": vec})
        loaded += 1

    r.execute_command("FT.OPTIMIZE", PION_INDEX)
    return jsonify({"loaded": loaded, "index": PION_INDEX, "embed_provider": EMBED_PROVIDER})


@app.route("/flare/stats", methods=["GET"])
def flare_stats():
    """Return FLARE gateway configuration and KB stats."""
    r = _conn()
    try:
        info_raw = r.execute_command("FT.INFO", PION_INDEX)
        info: dict[str, Any] = {}
        if isinstance(info_raw, list):
            for i in range(0, len(info_raw) - 1, 2):
                k = info_raw[i].decode() if isinstance(info_raw[i], bytes) else str(info_raw[i])
                v = info_raw[i + 1]
                info[k] = v.decode() if isinstance(v, bytes) else v
        num_docs = info.get("num_docs", 0)
    except Exception:
        num_docs = 0

    return jsonify({
        "upstream": UPSTREAM_ENDPOINT,
        "model": UPSTREAM_MODEL,
        "pion": f"{PION_HOST}:{PION_PORT}",
        "index": PION_INDEX,
        "kb_docs": num_docs,
        "tau": FLARE_TAU,
        "chunk_tokens": CHUNK_TOKENS,
        "embed_model": f"{EMBED_PROVIDER} ({EMBED_DIM}-dim)",
        "retrieval_latency": "~1.28ms",
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pion FLARE AI Gateway")
    parser.add_argument(
        "--upstream-type",
        choices=sorted(_UPSTREAM_DEFAULTS.keys()),
        help="Upstream backend type (sets host/port/path defaults).",
    )
    parser.add_argument("--upstream-host", help="Override upstream host.")
    parser.add_argument("--upstream-port", type=int, help="Override upstream port.")
    parser.add_argument("--upstream-path", help="Override upstream path.")
    args = parser.parse_args()

    _init_config(args)
    if EMBED_PROVIDER == "onnx":
        _load_embedder()
    log.info("FLARE Gateway starting on port %d", GATEWAY_PORT)
    log.info("  Upstream: %s (type: %s, model: %s)",
             UPSTREAM_ENDPOINT, UPSTREAM_TYPE, UPSTREAM_MODEL)
    log.info("  Pion:     %s:%d (index: %s)", PION_HOST, PION_PORT, PION_INDEX)
    log.info("  τ (tau):  %.2f | chunk: %d tokens", FLARE_TAU, CHUNK_TOKENS)
    app.run(host="0.0.0.0", port=GATEWAY_PORT, threaded=True)
