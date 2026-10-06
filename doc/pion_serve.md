# Pion Serve — Inference Intelligence Proxy

OpenAI-compatible HTTP proxy that sits between your client and any LLM
backend (Ollama, vLLM, OpenAI, Gemini, Claude, llama.cpp, MAX). Adds three
layers of cached intelligence on top of the backend so you serve
fewer / cheaper / better-grounded inferences.

## At a Glance

```
client → pion-serve :8321 ──┬──► L1 semantic cache    (Pion's auto-embed)
                            │
                            ├──► L3 concept store      (--distill)
                            │
                            ├──► L3b fragment store    (--distill)
                            │
                            ├──► Intent router         (--route)  ← picks tier-specific backend+model
                            │
                            └──► backend  (ollama / vLLM / OpenAI / Claude / Gemini / …)
```

- **L1** matches the user's query against past queries by cosine on
  Pion's auto-embed (MiniLM-L6-v2, 384-dim). Threshold 0.85 default.
  Hits return the cached response — no LLM call.
- **L3** matches against synthesized *concepts* extracted from prior
  responses. Catches some queries that L1 misses (different wording,
  similar intent).
- **L3b** matches at the *fragment* level — sentences extracted from
  prior responses. Designed to recombine fragments when neither L1
  nor L3 hits cleanly. (Hypothesis untested at scale; ships behind
  the same `--distill` flag.)
- **Intent router** classifies every L1/L3 miss by cosine
  against pre-computed simple/complex centroids and dispatches to
  a tier-specific backend+model: simple → cheap (Haiku / Flash /
  smollm), medium → mid (Sonnet / gemma4), complex → top (Opus /
  GPT-4). Heuristic boosts override the centroid for code blocks,
  stack traces, and very short queries. Off by default.
- **Backend** is called only on a full miss. Response goes back through
  ingest so future queries can hit L1/L3/L3b.

L1 ships always. L3 + L3b are opt-in via `--distill`. RAG injection is
opt-in via `--rag-index <index>`. Intent routing is opt-in via
`--route` (with optional `--route-config <json>`).

## Quick Start

```bash
# 1. Pion server with KV cache enabled (in another terminal)
./pion-server --kvcache -w 1

# 2. Your LLM backend (in a third terminal)
ollama serve
ollama pull gemma3:4b   # any model

# 3. Pion Serve
python pion-serve/serve.py --backend ollama --model gemma3:4b
# Listening on http://0.0.0.0:8321
```

OpenAI-compatible clients then point at `http://localhost:8321/v1/...`:

```bash
curl http://localhost:8321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:4b",
    "messages": [{"role": "user", "content": "What is Pion?"}]
  }'
```

The first call goes to Ollama. The second call with a similar question
returns from the L1 cache (≈ ms vs seconds).

## Run Recipes

### L1 only — production default

```bash
python pion-serve/serve.py --backend ollama --model gemma3:4b
```

Semantic cache only. 28-33% hit rate on typical Q&A workloads. Zero
hallucination risk: L1 returns a verbatim past response.

### L1 + L3 + L3b — `--distill` for repetitive workloads

```bash
python pion-serve/serve.py --backend ollama --model gemma3:4b --distill
```

Adds the concept and fragment stores. Catches additional queries on
top of L1, mostly on highly templated FAQ workloads. Cost:
two extra cosine searches per L1 miss. Use when your traffic is
repetitive (FAQ bots, customer support, internal tooling).

**Workload sensitivity.** L3 hit rate is bounded by
the *embedding-similarity ceiling* of the corpus, not by the substrate.
On a codebase-QA set where every question targets a different subsystem,
no pair of questions was similar enough to clear the default thresholds,
so nothing hit (an internal run, not published with a harness). Templated
FAQ workloads cluster much tighter in embedding space; codebase-style
workloads do not. Plan capacity
based on your corpus's pairwise similarity distribution, not on default
projections.

### L1 + RAG injection

```bash
python pion-serve/serve.py --backend ollama --rag-index my_docs
```

Where `my_docs` is a Pion FT index you populated separately (see
`doc/vector_engine.md`). On every L1 miss, Pion Serve does an
`FT.SEARCH` against the index and prepends the top-k results as
context to the backend prompt. Augments inference rather than
skipping it.

### L1 + Intent routing — `--route` for tiered models

```bash
python pion-serve/serve.py --route                                # Anthropic 3-tier defaults
python pion-serve/serve.py --route --route-config routing.json    # custom tiers / backends
```

Each query hits L1 first; on miss, the router picks one of three tiers
(`simple` / `medium` / `complex`) and forwards to that tier's
backend+model. The default routing table maps to Anthropic Haiku 4.5 /
Sonnet 4.6 / Opus 4.7 (`$0.25 / $3 / $15` per million tokens). Override
with a JSON file:

```json
{
  "simple":  {"backend": "ollama", "model": "smollm:135m",  "cost_per_mtok": 0.05},
  "medium":  {"backend": "ollama", "model": "gemma4:e4b",   "cost_per_mtok": 0.50},
  "complex": {"backend": "claude", "model": "claude-opus-4-7", "cost_per_mtok": 15.00}
}
```

Per-tier `base_url` is also supported — set it to point a tier at a
different host (e.g. local vLLM for simple, hosted API for complex).

The route is recorded in the response under `x_pion.route` and counted
in `/v1/stats.intent_router`. Estimated $-cost is accumulated against
the tier's `cost_per_mtok`. See [Intent Routing](#intent-routing)
below for the full classification rules.

### Other backends

```bash
python pion-serve/serve.py --backend vllm     --model meta-llama/Llama-3.2-3B
python pion-serve/serve.py --backend openai   --model gpt-4o-mini
python pion-serve/serve.py --backend gemini   --model gemini-1.5-flash
python pion-serve/serve.py --backend claude   --model claude-3-5-sonnet-20241022
python pion-serve/serve.py --backend llamacpp --model gpt-4
python pion-serve/serve.py --backend max      --model llama-3.1-8b-instruct
```

For paid APIs, set the API key via the standard environment variable
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`).

The `max` backend proxies [Modular MAX Serve](https://docs.modular.com/max/)
(OpenAI-compatible, `max serve` defaults to `:8000`) at the **request layer** —
semantic cache + routing in front of MAX. It is deliberately *not* a KV-datapath
integration: MAX 26.4 removed the LMCache connector Pion previously plugged into,
and MAX's native `KVConnector` factory is a closed enum with no third-party
registration. Start MAX first, then point Pion Serve at it:

```bash
max serve --model-path modularai/Llama-3.1-8B-Instruct-GGUF   # :8000
python pion-serve/serve.py --backend max --model llama-3.1-8b-instruct
```

### SIE embedding backend

If you already run [superlinked/sie](https://github.com/superlinked/sie)
for production embeddings, point Pion Serve at it. SIE is selected
ahead of Ollama in the cascade whenever `--sie-url` is set:

```bash
python pion-serve/serve.py --backend ollama --model gemma3:4b \
    --sie-url http://localhost:8000 \
    --sie-model BAAI/bge-small-en-v1.5
```

Force-select SIE (skip the cascade — error out if SIE is down rather
than fall through to Ollama):

```bash
python pion-serve/serve.py --backend ollama --model gemma3:4b \
    --embed-backend sie --sie-url http://localhost:8000
```

There is no auto-detect on port 8000 — the collision surface with
vLLM / FastAPI / generic web apps is too high. `--sie-url` is the
only opt-in. `/v1/stats.embedder` reports the active backend and per-
backend call counts (`sie` vs `ollama` vs `pion`).

### Pure proxy — no caching

```bash
python pion-serve/serve.py --backend ollama --no-cache
```

Useful for A/B-testing the cache value or for one-off pass-through.

## CLI Reference

| Flag | Default | Purpose |
|---|---|---|
| `--backend` | `ollama` | One of: ollama / vllm / openai / gemini / claude / llamacpp |
| `--backend-url` | per-backend default | Override the backend's base URL |
| `--model` | `llama3.2:3b` | Model name passed through to the backend |
| `--port` | `8321` | Where Pion Serve listens |
| `--pion-host` | `127.0.0.1` | Pion server host |
| `--pion-port` | `1974` | Pion server port |
| `--cache-threshold` | `0.85` | Cosine threshold for L1 hits (lower = more aggressive) |
| `--no-cache` | off | Disable L1 (and therefore L3 / L3b) — pure proxy mode |
| `--distill` | off | Enable L3 concept + L3b fragment layers |
| `--l3-direct-threshold` | auto-calibrated | L3 direct-match threshold |
| `--l3-composite-threshold` | auto-calibrated | L3 composite-match threshold |
| `--rag-index` | unset | Pion FT index name to inject as RAG context on cache misses |
| `--rag-k` | `3` | Top-k documents to inject |
| `--route` | off | Enable intent routing (tier-specific backend+model on L1/L3 miss) |
| `--route-config` | unset | JSON file with per-tier `{backend, model, [base_url], [cost_per_mtok]}` |
| `--route-margin` | `0.05` | Centroid cosine margin to commit to simple/complex (else medium) |
| `--ollama-url` | `http://127.0.0.1:11434` | Ollama URL (used by embedding fallback) |
| `--embed-backend` | `auto` | Embed backend selection: `auto` (cascade pion → sie if `--sie-url` → ollama) / `pion` / `sie` / `ollama` / `none`. Explicit choices do not fall through. |
| `--sie-url` | unset | SIE (`superlinked/sie`) base URL for embeddings, e.g. `http://localhost:8000`. When set, slots ahead of Ollama in the cascade. |
| `--sie-model` | `BAAI/bge-small-en-v1.5` | Model name sent to SIE `/v1/embeddings`. |
| `--mlx-tcp-host` | `127.0.0.1` | MLX sidecar host (for `/v1/stats` GPU section) |
| `--mlx-tcp-port` | `0` | MLX sidecar TCP port (0 = disabled in stats) |

## HTTP Endpoints

OpenAI-compatible:

- `POST /v1/chat/completions` — primary endpoint. Standard request body
  with `model` and `messages`. Streams when `stream=true`.
- `GET  /v1/models` — proxies to the backend's models list.
- `GET  /health` — `{"status": "ok"}` health check.

Pion-specific:

- `GET  /v1/stats` — full pipeline statistics (see below).

## /v1/stats Schema

```json
{
  "total_requests": 1432,
  "l1_cache_hits": 421,
  "l3_synthesis": 87,
  "l3_direct": 64,
  "l3_composite": 23,
  "full_inference": 924,
  "l1_hit_rate": 0.294,
  "l3_synthesis_rate": 0.061,
  "full_inference_rate": 0.645,
  "cost_savings_pct": 35.5,
  "rag_injections": 12,
  "avg_latency_ms": 412.3,
  "backend": "ollama",
  "model": "gemma3:4b",
  "cache_enabled": true,
  "distill_enabled": true,
  "rag_index": null,
  "embedder": {
    "backend": "sie",
    "sie_url": "http://localhost:8000",
    "sie_model": "BAAI/bge-small-en-v1.5",
    "calls": {"sie": 412, "ollama": 0, "pion": 0}
  },
  "l3_concepts": { ... },
  "l3_fragments": { ... },
  "intent_router": {
    "classified_total": 312,
    "unclassified": 0,
    "simple":  187, "simple_pct":  59.9,
    "medium":   42, "medium_pct":  13.5,
    "complex":  83, "complex_pct": 26.6,
    "estimated_cost_usd": 1.4321,
    "routing": {
      "simple":  {"backend": "claude", "model": "claude-haiku-4-5-20251001", "cost_per_mtok": 0.25},
      "medium":  {"backend": "claude", "model": "claude-sonnet-4-6",         "cost_per_mtok": 3.00},
      "complex": {"backend": "claude", "model": "claude-opus-4-7",           "cost_per_mtok": 15.00}
    }
  },
  "mlx_gpu": { ... },
  "v_store": { ... }
}
```

Per-response, every routed completion includes `x_pion.route`:

```json
"x_pion": {
  "source": "full_inference",
  "latency_ms": 412.3,
  "route": {
    "tier": "simple",
    "backend": "claude",
    "model": "claude-haiku-4-5-20251001",
    "confidence": 0.84,
    "heuristic_boost": ""
  }
}
```

`cost_savings_pct = (l1_hit_rate + l3_synthesis_rate) × 100` — the
fraction of requests that did NOT call the backend.

## What hit rate to expect

It depends on how tightly your queries cluster in embedding space, so
measure it on your own traffic (`/v1/stats` reports it). Templated FAQ
workloads cluster tightly and hit often; questions that each target a
different subject — codebase Q&A is the example above — can miss the
default thresholds entirely. L1 is the layer that matters;
L3/L3b ride along behind `--distill` and only contribute on workloads
where the same intent comes back as a meaningfully different sentence.

## When to use --distill

- ✅ FAQ bots, customer support, internal-tool Q&A — high template
  reuse, where L3 can catch paraphrases L1 misses.
- ✅ Any workload where the cost of an extra cosine search is much
  smaller than the cost of an LLM token.
- ❌ Highly-diverse queries (open-ended chat, creative writing) — L1
  is enough; L3 will add cost without measurable catch.
- ❌ Latency-critical paths where the extra ~5-10 ms of the L3 search
  matters and the catch rate doesn't.

L3 / L3b are designed as **off by default**. Enable when you've
measured your traffic and confirmed L1's catch rate is leaving
significant value on the table.

## Intent Routing

Behind `--route`, every L1/L3 miss is classified into one of three
tiers and dispatched to that tier's backend+model. Cost-driven: the
goal is to never call Opus when Haiku will do.

### Pipeline position

Intent routing runs **after** L1/L3 caches and RAG injection, so
cached and RAG-augmented responses bypass it entirely. It only fires
on a real backend call, which is the only place where tier price
actually applies.

### Classifier

Two pre-computed centroids — `simple` and `complex` — are built at
startup by embedding 20 + 20 seed queries (`SIMPLE_SEEDS` /
`COMPLEX_SEEDS` in `pion-serve/intent_router.py`). For each request:

1. Embed the query (auto-embed sidecar or Ollama
   `nomic-embed-text` — same provider as L1).
2. Compute cosine to both centroids; let `diff = simple − complex`.
   - `diff > +margin` → **simple**
   - `diff < −margin` → **complex**
   - else → **medium**
3. Apply heuristic overrides:
   - Code block / `def …(` / `class X:` / `SELECT … FROM` /
     `Traceback …` → force **complex**.
   - Centroid says complex but query is ≤6 whitespace tokens →
     drop to **medium** (no Opus on a four-word question).
   - Centroid says simple but query is ≥60 tokens → bump to
     **medium** (long queries usually have nuance).

Default `margin = 0.05`. Tighten with `--route-margin 0.10` to push
ambiguous queries into `medium` more aggressively.

If the embedder returns `None` (sidecar offline), the classifier
falls back to **medium** rather than failing the request.

### Routing table

```python
DEFAULT_ROUTING = {
  "simple":  {"backend": "claude", "model": "claude-haiku-4-5-20251001", "cost_per_mtok": 0.25},
  "medium":  {"backend": "claude", "model": "claude-sonnet-4-6",         "cost_per_mtok":  3.00},
  "complex": {"backend": "claude", "model": "claude-opus-4-7",           "cost_per_mtok": 15.00},
}
```

Override via `--route-config <path.json>`. Only the tiers you list
are overridden; the rest fall through to the defaults. Each entry
accepts `backend` (one of `ollama / vllm / openai / gemini / claude
/ llamacpp`), `model`, optional `base_url`, optional
`cost_per_mtok`. Cross-backend tiers are supported (simple →
local Ollama, complex → hosted Claude).

### Streaming

Anthropic's SSE format differs from OpenAI's. When a streaming
request routes to the `claude` backend, Pion Serve falls back to a
non-stream call and emits the response as a single SSE chunk. Other
backends (ollama, openai, vllm, gemini, llamacpp) stream natively.

### Cost accounting

`cost_per_mtok` in the routing table is what
`/v1/stats.intent_router.estimated_cost_usd` accumulates against
per-request, so the stats endpoint reports what the routing actually
saved on your traffic.

### Tests

```bash
python3 tests/test_intent_router.py        # offline (no LLM, no Ollama)
python3 tests/test_intent_router.py --live # also runs against Ollama nomic-embed-text
```

Offline tests use a deterministic fake embedder built from token
overlap with the seed sets — they validate centroid bootstrap,
held-out classification, heuristic boosts, config overrides, and
the no-embedding fallback. The live test runs the full embed →
classify path against `nomic-embed-text` on a small held-out set.

### See also

- `pion-serve/intent_router.py` — `IntentRouter`, seed sets,
  routing table, regex heuristics.
- `doc/ai_gateway.md` §AI.ROUTE — the lower-level Mojo routing
  command (`AI.ROUTE.REGISTER/ROUTE/INFO`) that solves the same
  problem at the wire layer. Pion Serve's `--route` is the Python
  / proxy version optimized for OpenAI-compatible deployment.

## Backend Notes

- **Ollama** (default): expects `ollama serve` on `:11434`. Pion
  Serve auto-detects whether `nomic-embed-text` is available for
  embedding; falls back to Pion's auto-embed sidecar (MiniLM-L6-v2,
  384-dim) if not.
- **vLLM**: standard OpenAI-compatible server on `:8000`.
- **OpenAI / Gemini / Claude**: requires the respective API key in
  the environment. Pion Serve passes through the auth header.
- **llama.cpp**: assumes `--server` mode with the OpenAI-compatible
  endpoint on `:8080`.

The Pion server itself (port 1974) is required regardless of which
backend you use — that's where the L1 / L3 / L3b state lives.

## Files

- `pion-serve/serve.py` — Flask proxy (HTTP routes + L1/L3/L3b/RAG
  pipeline + backend dispatch).
- `pion-serve/concept_store.py` — `ConceptStore` and `FragmentStore`
  classes, used when `--distill` is on.
- `pion-serve/intent_router.py` — `IntentRouter` (centroid +
  heuristic classifier), seed sets, default routing table, JSON
  config loader. Used when `--route` is on.
- `pion-serve/generate_qa.py` (+ `generate_qa_claude.py`,
  `generate_qa_gemini.py`) — Q&A dataset generators for replay
  experiments.
- `pion-serve/replay_experiment.py` — replays a Q&A dataset through the
  proxy and logs per-query hits, for measuring your own hit rate.

## Related

- `doc/ai_gateway.md` — the lower-level Pion server commands
  (`AI.SEMANTIC_CACHE`, `FT.SEARCH`, `KV.PREFIX.*`, `ATTEND.PREFIX.*`)
  that Pion Serve calls under the hood.
- `doc/embeddings.md` — Pion's auto-embed sidecar (MiniLM-L6-v2)
  and the `nomic-embed-text` Ollama fallback.
- `doc/vector_engine.md` — `FT.CREATE` / `FT.SEARCH` for the
  `--rag-index` injection path.
- `doc/shared_kv_cache.md` — `KV.PREFIX.*` and the Stage-2
  monkey-patch (separate from Pion Serve; they compose if you
  combine `--distill` with a model that uses `PionPromptCache`).
