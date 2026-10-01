# Pion FLARE AI Gateway

An OpenAI-compatible HTTP proxy that adds **mid-generation retrieval (FLARE)** to any LLM backend.
Transparently monitors streaming token logprobs, and when confidence drops below τ, queries Pion for
grounding context before regenerating the uncertain chunk.

```
Client → FLARE Gateway (:8080) → Upstream LLM (Ollama / vLLM / OpenAI)
                               ↕
                          Pion (:1974)  [1.28ms retrieval — faster than one token]
```

Drop-in replacement: swap your LLM base URL from the upstream to the gateway. The client API is
fully OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`).

---

## Why FLARE + Pion?

Traditional pre-generation RAG retrieves once before generating. FLARE retrieves *during* generation,
triggered only when the model becomes uncertain mid-sentence — the token logprob `min(chunk) < τ`.

This requires retrieval to be nearly free. Network-bound vector stores add 50–200ms per retrieval —
a visible stutter in streaming UIs. Pion's in-process retrieval costs **1.28ms** (embed + HNSW search),
which is **imperceptible** at generation speeds of 8–17 tokens/second.

**Experimental results (HotpotQA 20 questions):**

| Setup | F1 | Correct |
|---|:---:|:---:|
| 1B baseline | 0.15 | 7/20 |
| 1B + FLARE | **0.37** | **7/20** (better partial credit) |
| 8B few-shot | 0.29 | — |
| 8B few-shot + FLARE | **0.35** | **10/20** |

FLARE fires on average 0.6–1.3 retrievals per question, only when genuinely uncertain.

---

## Quickstart

```bash
# Install (no onnxruntime needed for mock/openai providers)
pip install flask requests "redis>=4.6.0,<5.0" numpy

# Start Pion
./pion-server

# Load knowledge base documents
curl -X POST http://localhost:8080/flare/load \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"text": "The Eiffel Tower was built in 1889 in Paris."},
                     {"text": "The speed of light is 299,792,458 m/s."}]}'

# Use OpenAI-compatible API (drop-in replacement)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "When was the Eiffel Tower built?"}],
       "max_tokens": 100}'
```

---

## Running

```bash
# With Ollama (default)
python flare_gateway/gateway.py --upstream-type ollama

# With OpenAI
OPENAI_API_KEY=sk-... python flare_gateway/gateway.py --upstream-type openai

# With vLLM
python flare_gateway/gateway.py --upstream-type vllm
```

---

## Quick Start Per Backend

```bash
# Ollama (/api/generate by default; use --upstream-path /api/chat for chat API)
python flare_gateway/gateway.py --upstream-type ollama

# llama.cpp server (llama-server)
python flare_gateway/gateway.py --upstream-type llama-cpp

# vLLM
python flare_gateway/gateway.py --upstream-type vllm

# exo
python flare_gateway/gateway.py --upstream-type exo

# OpenAI
OPENAI_API_KEY=sk-... python flare_gateway/gateway.py --upstream-type openai
```

---

## Embedding Providers

| Provider | Dimension | Dependency | Use |
|---|:---:|---|---|
| `mock` | 1536 | none | Smoke tests — deterministic hash-based vectors |
| `openai` | 1536 | `OPENAI_API_KEY` | Production — `text-embedding-3-small` |
| `onnx` | 384 | `onnxruntime`, `transformers` | Local MiniLM, ~1.2ms in-process |

```bash
# Smoke test — no API keys, no model downloads
EMBED_PROVIDER=mock python flare_gateway/gateway.py

# Production with OpenAI embeddings
EMBED_PROVIDER=openai OPENAI_API_KEY=sk-... python flare_gateway/gateway.py
```

**Important:** `EMBED_DIM` must match `config.vector.dimensions` in Pion (default: 1536).
The `onnx` provider produces 384-dim vectors — set `EMBED_DIM=384` and ensure Pion's HNSW
config matches, or use `openai`/`mock` for 1536-dim compatibility with the default config.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `UPSTREAM_TYPE` | `ollama` | Backend preset: `ollama`, `llama-cpp`, `vllm`, `exo`, `openai` |
| `UPSTREAM_URL` | `http://localhost:11434` | LLM backend base URL (optional path allowed) |
| `UPSTREAM_MODEL` | `llama3.1:8b` | Model name |
| `PION_HOST` | `localhost` | Pion server host |
| `PION_PORT` | `1974` | Pion server port |
| `PION_INDEX` | `flare_kb` | Vector index name in Pion |
| `FLARE_TAU` | `0.3` | Logprob confidence threshold (0–1). Lower = more retrievals. |
| `FLARE_CHUNK_TOKENS` | `10` | Tokens per generation chunk |
| `EMBED_PROVIDER` | `onnx` | Embedding provider: `onnx`, `openai`, `mock` |
| `EMBED_DIM` | `384` (onnx) / `1536` (other) | Vector dimension |
| `GATEWAY_PORT` | `8080` | Port to listen on |

---

## CLI Flags

| Flag | Description |
|---|---|
| `--upstream-type` | Backend preset: `ollama`, `llama-cpp`, `vllm`, `exo`, `openai` |
| `--upstream-host` | Override upstream host |
| `--upstream-port` | Override upstream port |
| `--upstream-path` | Override upstream path |

---

## API Endpoints

### `POST /v1/chat/completions`
OpenAI-compatible chat completions with FLARE augmentation. Accepts `messages`, `max_tokens`,
`stream`. Returns standard OpenAI response plus `x_flare` metadata:

```json
{
  "choices": [{"message": {"role": "assistant", "content": "..."}}],
  "x_flare": {
    "retrievals": 2,
    "chunks": 5,
    "elapsed_s": 3.4,
    "tau": 0.3,
    "pion_index": "flare_kb"
  }
}
```

### `POST /flare/load`
Bulk-load documents into the FLARE knowledge base:

```json
{"documents": [{"text": "Document content here..."}, ...]}
```

Returns `{"loaded": N, "index": "flare_kb", "embed_provider": "mock"}`.

### `GET /flare/stats`
Returns gateway config, KB doc count, and embedding model info.

### `GET /health`
Returns gateway health payload with upstream and Pion targets.

---

## Implementation

**FLARE loop** (`flare_generate()` in `gateway.py`):
1. Generate `chunk_tokens` tokens from the upstream LLM with `logprobs=true`
2. Compute `min_prob = min(exp(logprob) for logprob in chunk)`
3. If `min_prob < τ` (uncertain): use `(generated_so_far + chunk)` as retrieval query
4. Query Pion: `FT.SEARCH flare_kb "*=>[KNN 2 @embedding $vec EF_RUNTIME 64]" PARAMS 2 vec <bytes>`
5. Prepend retrieved facts as `Context:\n- fact1\n- fact2\n\n` and regenerate the chunk
6. If `min_prob >= τ` (confident): accept the chunk and continue

**Retrieval fallback**: when fewer than 8 documents are loaded (HNSW batch-8 kernel requires ≥8 nodes),
`pion_retrieve()` falls back to Python cosine similarity over all documents.

**Key insight** (Experiment 12b): few-shot prompting alone improved
8B baseline F1 from 0.06 to 0.29. FLARE adds +0.06 on top. The right production setup is:
`8B + few-shot prompt + FLARE` — no fine-tuning required.

## License

Apache-2.0. Pion's satellites are deliberately permissive so they can be vendored
into any stack; the Pion **server** itself is Apache-2.0 too,
with one closed binary library for its tuned vector kernels — see the top-level
`LICENSE` and `doc/licensing.md`.
