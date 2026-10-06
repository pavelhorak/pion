# AI Gateway

Pion implements a native AI pipeline directly in the database engine, eliminating the need for
external orchestration layers. All AI commands are handled in `src/network/slow_path.mojo` via
blocking HTTP/1.0 calls to OpenAI-compatible external services (embedding server + LLM server).

**Ollama auto-detect:** Start with `./pion-server --flare` — Pion probes `localhost:11434` at startup. If `nomic-embed-text` is found, embedding is auto-enabled; `--flare` sets the LLM model to `llama3.1:8b` (plain auto-detect, without `--flare`, uses `llama3.2:1b`). Skip Ollama with `--no-auto-detect` and enable manually via `--emb-enabled --llm-enabled`.

---

## Architecture

### Traditional RAG Pipeline (5+ hops)

```
Client → App Server → Embedding Service → Vector DB → App Server → LLM → App Server → Client
```

### Pion AI Pipeline (2 hops)

```
Client ─RESP3─► Pion
                 ├─ embed text (HTTP → embedding server)
                 ├─ search HNSW (in-memory, zero-copy)
                 ├─ build prompt (in-memory)
                 └─ call LLM (HTTP → MAX Serve / OpenAI)
Pion ─RESP3─► Client
```

No Python runtime. No application server. One TCP connection.

---

## Implemented Commands

### AI.COMPLETE

```
AI.COMPLETE <query> [TOKENS <n>] [THRESHOLD <t>]
```

One-command cache-augmented generation: checks the semantic cache, returns the cached response on a hit, calls the LLM and caches the result on a miss.

```bash
redis-cli AI.COMPLETE "What is the capital of France?" TOKENS 100 THRESHOLD 0.92
# Cache hit (≥ threshold cosine similarity): returns cached response in <1ms
# Cache miss: calls LLM, stores result, returns LLM response
```

Cache hit performance: **275× faster** than a live LLM call (see `examples/pion_vs_ollama_demo.py`).

---

### FT.ADDTEXT

```
FT.ADDTEXT <index> <doc_id> <text>
```

Stores a text document and makes it immediately searchable:

1. Stores `doc_id` + `text` as a HASH in the keyspace
2. Registers `doc_id` in the BM25 doc set, so `FT.OPTIMIZE` indexes it
3. Embeds `text` via `EmbeddingClient` (HTTP/1.0 POST to `/v1/embeddings`)
4. Inserts the embedding into the per-worker semantic HNSW index (`add_and_insert`)
5. Returns `+OK`

Steps 3–4 require `EmbeddingConfig.enabled = True`; steps 1–2 do not, so BM25 ingest works with no embedding backend at all.

Step 2 only applies to **non-negative integer** `doc_id`s — BM25 hits are returned as integer ext_ids. A string id like `doc:1` is still stored and embedded, so `FT.SEARCHTEXT` finds it, but it stays invisible to `FT.SEARCH … BM25`.

```bash
# string ids — semantic search only
redis-cli FT.ADDTEXT products doc:1 "Wireless headphones with active noise cancellation"
redis-cli FT.ADDTEXT products doc:2 "Mechanical keyboard Cherry MX switches"

# integer ids — semantic search AND BM25 (after FT.OPTIMIZE)
redis-cli FT.ADDTEXT products 1 "Wireless headphones with active noise cancellation"
redis-cli FT.OPTIMIZE products
redis-cli FT.SEARCH products BM25 "noise cancellation" K 3
```

### FT.SEARCHTEXT

```
FT.SEARCHTEXT <index> <query_text> [K <k>]
```

Semantic search over text documents added via `FT.ADDTEXT`:

1. Embeds `query_text` via `EmbeddingClient`
2. Searches the per-worker semantic HNSW (ef=32, default k=10)
3. Returns matching `doc_id` strings as a RESP array

```bash
redis-cli FT.SEARCHTEXT products "noise cancelling audio" K 3
# → 1) "doc:1"
```

### AI.CHAT

```
AI.CHAT <prompt> [CONTEXT <index> <text> [K <k>]]
```

Full RAG pipeline in a single RESP3 command:

1. **(optional)** Embeds `text`, searches HNSW for top-k doc_ids, retrieves stored texts
2. Builds LLM prompt: `"Context:\n<docs>\n\nUser: <prompt>"` (or just `<prompt>` if no CONTEXT)
3. Sends HTTP/1.0 POST to `/v1/chat/completions`
4. Parses `"content":"..."` from JSON response, returns as RESP bulk string

```bash
# Simple chat
redis-cli AI.CHAT "What is the capital of France?"

# RAG: retrieve context then generate
redis-cli AI.CHAT "Which audio product do you recommend?" CONTEXT products "audio" K 3
# → "Based on the available products, I recommend the wireless headphones..."
```

---

## Configuration

### Embedding Server (`EmbeddingConfig`)

```
src/common/config.mojo → EmbeddingConfig:
  host:       "127.0.0.1"
  port:       11434           (Ollama default)
  model:      "nomic-embed-text"   (Ollama) or "all-MiniLM-L6-v2" (auto-embed)
  dimensions: 768 (Ollama) or 384 (auto-embed sidecar, default)
  threshold:  0.95 (768-dim) or 0.85 (384-dim auto-embed)
  enabled:    false           ← set True to enable FT.ADDTEXT / FT.SEARCHTEXT
```

Compatible: Ollama, MAX Serve (`max serve --model nomic-embed-text`), any `/v1/embeddings` endpoint.

### LLM Server (`LLMConfig`)

```
src/common/config.mojo → LLMConfig:
  host:    "127.0.0.1"
  port:    8000               (MAX Serve default)
  model:   "meta-llama/Llama-3.1-8B-Instruct"
  enabled: false              ← set True to enable AI.CHAT
```

Compatible: MAX Serve, Ollama, OpenAI API, any `/v1/chat/completions` endpoint.

---

## Implementation Details

**`src/network/llm_client.mojo`** — `LLMClient`:
- Blocking HTTP/1.0 POST (connection-per-request)
- JSON body: `{"model":"...","messages":[{"role":"user","content":"..."}],"stream":false}`
- `_llm_escape_json()` handles `"`, `\`, `\n`, `\r` escaping (module-level to avoid Mojo nested-function mutability constraints)
- Response parser: scans for `"content":"` tag, extracts text with backslash unescape
- 1MB receive buffer; reads until server closes connection

**`src/network/embedding_client.mojo`** — `EmbeddingClient`:
- Blocking HTTP/1.0 POST to `/v1/embeddings`
- Hand-written float parser (`_parse_float32`): no stdlib dependency
- Output: INT8-quantized embedding written to caller-supplied buffer

**Semantic HNSW:**
`FT.ADDTEXT` / `FT.SEARCHTEXT` reuse `SemanticCache.hnsw` (capacity=10K, ef_construction=32).
`add_and_insert()` does immediate insert + compact_buffer rebuild — no deferred FT.OPTIMIZE.
Doc_ids stored in `scache.responses[]` alongside `AI.SEMANTIC_CACHE` entries.

---

## Quick Start

```bash
# 1. Start Ollama with required models
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2:1b   # or llama3.1:8b for higher quality

# 2. Start Pion with AI features (--flare auto-detects Ollama)
./pion-server --flare

# 3. Semantic cache + LLM in one command
redis-cli -p 1974 AI.COMPLETE "What is the capital of France?" TOKENS 100 THRESHOLD 0.92

# 4. Add documents for RAG
redis-cli -p 1974 FT.ADDTEXT kb doc:1 "Pion is a Mojo-native vector database"
redis-cli -p 1974 FT.ADDTEXT kb doc:2 "HNSW achieves 8134 QPS on Apple M4"

# 5. Semantic text search
redis-cli -p 1974 FT.SEARCHTEXT kb "performance benchmark" K 2

# 6. RAG chat (retrieves context then generates)
redis-cli -p 1974 AI.CHAT "How fast is Pion?" CONTEXT kb "performance" K 2
```

---

## Related: AI.SEMANTIC_CACHE

Caches LLM responses keyed by semantic similarity of the input query:

```bash
AI.SEMANTIC_CACHE SET "What is the capital of France?" "+Paris\r\n"
AI.SEMANTIC_CACHE GET "capital of France?"    → +Paris
AI.SEMANTIC_CACHE GET "unrelated query"       → $-1 (cache miss)
```

### Workspace audit trail

A cache hit tells you *what* the model answered, not what it had in mind while
answering. `WORKSPACE` records the caller's snapshot — e.g. the retrieved context or
top-k, base64 — at cache-WRITE time, and `EXPLAIN` reads it back **with no inference
infrastructure required**. That is the point: a compliance reviewer can ask
what was in the workspace without being able to run the model.

```bash
AI.SEMANTIC_CACHE SET "capital of France?" "Paris" WORKSPACE "<base64 lens top-k>"
AI.SEMANTIC_CACHE EXPLAIN "capital of France?"      → "<base64 lens top-k>"
AI.SEMANTIC_CACHE EXPLAIN "never cached"            → nil
AI.SEMANTIC_CACHE GET "capital of France?" WITHWORKSPACE
                                                     → %2 response/workspace (RESP3)
                                                     → *4 flat pairs        (RESP2)
```

The blob is **opaque to the server** — stored and returned, never interpreted —
so the lens format can change without a wire change. An entry written without
`WORKSPACE` reports nil rather than an empty string, so "no audit trail" is
distinguishable from "an empty one".

**`WITHWORKSPACE` is opt-in and the default GET reply is unchanged**, on both
protocols. Always returning a RESP3 map would break every RESP3 client already
reading GET as a bulk string, and a wire break to add observability is the wrong
trade. `EXPLAIN` needs no flag.

Implementation note: `workspaces[]` runs parallel to `responses[]`, and other
callers (MCP memory in `ai.mojo`, RAG ingest in `vector.mojo`) append to
`responses` directly without knowing about workspaces. `cache_set` therefore
PADS to `count` before appending rather than assuming alignment — if the two
ever drift, `EXPLAIN` returns some other entry's audit trail, which is worse
than returning none.

See `src/network/semantic_cache.mojo` and `src/network/embedding_client.mojo`.
Test: `tests/test_gh115_workspace.py`. It ENV_SKIPs without an embedding backend, because every `SET` is a
silent no-op there and the failures would all be about the missing backend.

---

## FLARE AI Gateway (Python, `flare_gateway/`)

A separate Python HTTP proxy that adds **mid-generation retrieval** to any LLM backend.
Unlike the in-Mojo `AI.CHAT` command (which retrieves once before generating), FLARE monitors
token logprobs *during* generation and retrieves only when the model becomes uncertain.

```
Client → FLARE Gateway (:8080) → Ollama / vLLM / OpenAI (:11434)
                               ↕
                          Pion (:1974)  [1.28ms — faster than one token at 8B scale]
```

### How FLARE works

1. Generate `N` tokens (chunk) from the upstream LLM with `logprobs=true`
2. Compute `min_prob = min(exp(logprob) for logprob in chunk)`
3. If `min_prob < τ` (uncertain): use `(context + uncertain_chunk)` as Pion query
4. Retrieve top-k facts from Pion in 1.28ms
5. Prepend facts as context and regenerate the uncertain chunk
6. Accept confident chunks without retrieval

### Quickstart

```bash
pip install flask requests "redis>=4.6.0,<5.0" numpy

# Load knowledge base
curl -X POST http://localhost:8080/flare/load \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"text": "The Eiffel Tower was built in 1889 in Paris."}]}'

# Use as drop-in OpenAI replacement
EMBED_PROVIDER=mock UPSTREAM_URL=http://localhost:11434 \
  python flare_gateway/gateway.py
```

See `flare_gateway/README.md` for full configuration reference.

## Example Scripts

See `examples/` for runnable demos:

| Script | What it shows |
|---|---|
| `examples/pion_vs_ollama_demo.py` | `AI.COMPLETE` semantic cache in front of Ollama |
| `examples/step6_flare.py` | FLARE Python gateway: mid-generation retrieval |
| `examples/step6b_flare_fewshot.py` | FLARE with few-shot prompting on an 8B model |
| `examples/step13_flare_mojo.py` | FLARE Mojo gateway end-to-end test (`AI.FLARE` commands) |
| `examples/agent_memory_demo.py` | Persistent agent memory via MCP tools |
| `examples/session_affinity_demo.py` | Multi-turn session routing via HSET + EXPIRE; TTL expiry → least-loaded fallback |
| `examples/coalescing_proxy.py` | FastAPI proxy using `SET NX` for dedup: 50 concurrent identical queries → 1 inference call |
| `examples/prefix_routing_demo.py` | System-prompt hash → replica routing |
| `examples/exo_session_demo.py` | Pion routing overhead vs direct exo requests; real + stub modes |
| `examples/multinode_prefix_routing.py` | Cross-replica prefix routing with two OpenAI-compatible endpoints |

Requirements: Ollama + `nomic-embed-text` + `llama3.2:1b` (or `llama3.1:8b` for higher quality).

```bash
# Start Pion with AI features
./pion-server --flare

# Run the semantic cache demo
python3 examples/pion_vs_ollama_demo.py
```

---

## Clustered Inference Integration

Distributed inference tools — **exo**, **llama.cpp server**, **vllm-mlx**, and **Hypura** — are
stateless HTTP APIs. When you run more than one of them, you immediately need:

1. **Semantic deduplication** — two nodes answer the same question independently
2. **Session affinity** — multi-turn conversation context is local to the process that handled turn 1
3. **Shared RAG index** — each node embeds and indexes the same documents independently
4. **Request coalescing** — N concurrent identical queries trigger N inference calls
5. **Cross-node prefix cache** — vLLM's PagedAttention prefix cache does not cross replica boundaries
6. **Load visibility** — no atomic way to observe queue depths across nodes without bespoke monitoring


### How Pion Addresses Each Problem

**1. Semantic Deduplication** — `AI.COMPLETE` in front of any inference cluster. Any node's answer
is cached; threshold-similar future queries return in 15ms.

**2. Session Affinity** — HSET/HGET to route multi-turn conversations to the node holding the KV cache:
```
HSET session:abc123  node "exo-head-2:52415"  last_seen 1742900000
EXPIRE session:abc123 3600
HGET session:abc123 node  →  route to "exo-head-2:52415"
```

**3. Shared RAG Index** — `FT.ADDTEXT` / `FT.SEARCHTEXT` give every cluster node one shared HNSW
index. No per-node embedding service; no side-car vector DB.

**4. Request Coalescing** — Pion's atomic `SET NX` deduplicates concurrent identical requests
without changes to the inference backend. 50 concurrent identical queries → 1 inference call.
See `examples/coalescing_proxy.py`.

**5. Cross-Node Prefix Cache Routing** — Hash the system prompt, store warm node in Pion, route
subsequent requests there. Requests sharing a common system prompt hit the replica already warmed
for those KV pages, cutting TTFT by ~30–60% for common prompts (coding assistants, support bots).
See `examples/prefix_routing_demo.py`.

**6. Capacity-Aware Load Balancing** — Each node updates its score via `ZADD inference_fleet
<queue_depth> "node:{id}"`. Atomic least-loaded routing. No separate load balancer process.

### Architecture Patterns

**Pattern A — Pion as AI Memory Layer (recommended first step)**

Change one env var in `flare_gateway/gateway.py` (`UPSTREAM_URL`). No inference backend changes:
```
Client → flare_gateway/gateway.py → Pion:1974 (cache hit < 20ms)
                                          └─ cache miss → exo | llama-server | vllm-mlx
```

**Pattern B — Pion as Cluster Coordinator**

All coordination primitives (session store, prefix cache, queue depths, semantic cache, RAG index)
in a single Pion instance. Each inference node reads/writes via standard Redis commands.

**Pattern C — Pion as Speculative Corpus**

FT.SEARCHTEXT retrieves likely token continuations from the HNSW corpus; exo/vllm-mlx validates
draft tokens in parallel. Acceptance rate ≥20% in grounded conditions (document in context).
See `examples/step7_rest_pion_drafter.py`.

### Per-Tool Integration Notes

**exo:** `FLARE_UPSTREAM_URL=http://exo-head:52415`. Session tracking via HSET. Queue depth via
INCR/DECR (exo doesn't expose queue depth natively). See `examples/exo_session_demo.py`.

**llama.cpp:** Semantic cache via `AI.COMPLETE`. Session: `HSETNX session:{id} node "llama-server-3:8080"`.
Prefix routing gives llama.cpp an implicit prefix cache by routing system-prompt-identical requests
to the same replica, which builds KV cache naturally. See `examples/prefix_routing_demo.py`.

**vllm-mlx:** Cross-replica prefix routing (Pattern B). Pion's semantic cache (application layer)
is complementary to vLLM's PagedAttention (KV layer) — Pion saves full inference on similar
prompts; vLLM saves KV recomputation on exact-prefix continuations.

**Hypura** (Ollama-compatible API): `FLARE_UPSTREAM_URL=http://hypura-host:11434`, no other changes.

## MCP server (pion-mcp, `mcp/`) — experimental

`mcp/` holds an MCP server whose `agent_remember` / `agent_recall` / `agent_forget` tools
store memories in a Pion FT index. It is experimental. Pion serves one FT index at a time,
so these memories, the MCP semantic cache and codebase search replace one another on a
server; give each its own server. pion-mcp also does not accept the `ollama` embedding
provider that pion-context defaults to. See `mcp/README.md`.

---

## Externalized Attention

Replaces O(N^2) transformer attention with O(N log k) HNSW retrieval. Instead of computing full attention over all past tokens, the inference engine stores KV pairs in Pion's per-layer HNSW indices. At query time, top-k nearest keys are retrieved in 86us (vs milliseconds for full attention at 128K context).

### How it works

```
INGEST (once per prompt):
  Inference engine → ATTEND.CREATE(session, key_dim, value_dim)
                   → ATTEND.STORE(session, layer, num_tokens, keys_fp32, values_fp32)  [per layer]
                   → ATTEND.FINALIZE(session, layer)  [builds HNSW per layer]

QUERY (per new token):
  New token's Q vector → ATTEND.QUERY(session, layer, k=64, query_fp32)
                       → Returns top-k V vectors (cosine 1.0 vs full attention)
                       → 86us per layer, 3.4ms for 40 layers at 128K tokens
```

### Commands

**RESP protocol (port 1974):**

| Command | Syntax | Description |
|---|---|---|
| `KV.STORE` ⚠️ | `KV.STORE <cache_id> <embedding_fp32> <blob> [TTL <sec>] [MODEL <name>]` | Store KV cache tensor blob with HNSW-indexed embedding key. **Experimental — TTL parsed and stored but not enforced; MODEL parsed but ignored on FETCH.** |
| `KV.FETCH` ⚠️ | `KV.FETCH <embedding_fp32> [THRESHOLD <cosine>] [MODEL <name>]` | Fetch nearest cached tensor by cosine similarity. **Experimental — MODEL arg silently ignored; capacity-capped at 1000 entries with append-only-on-overflow (no LRU).** |
| `KV.INFO` | `KV.INFO` | KV cache store statistics |
| `ATTEND.CREATE` | `ATTEND.CREATE <session_id> <key_dim> <value_dim>` | Create attention session |
| `ATTEND.STORE` | `ATTEND.STORE <session_id> <layer_id> <num_tokens> <keys_fp32> <values_fp32>` | Stage token KV pairs (3.67M tok/s via binary protocol) |
| `ATTEND.FINALIZE` | `ATTEND.FINALIZE <session_id> <layer_id>` | Batch build HNSW index from staged keys |
| `ATTEND.QUERY` | `ATTEND.QUERY <session_id> <layer_id> <k> <query_fp32>` | Top-k HNSW search; returns the top-k value rows, best first, as one bulk string of `min(k, stored) × value_dim` FP32s; `*0` for an empty layer |
| `ATTEND.INFO` | `ATTEND.INFO` | Index statistics (sessions, tokens, queries) |

**Binary protocol (port 1975, 0xCA5E framing):** ATTEND commands available as 0x20-0x23 for maximum throughput. See [Networking](networking.md) for wire format.

### Performance (binary protocol, 128K tokens, 128d keys)

| Metric | Value |
|---|---|
| Store throughput | 3,667,377 tok/s |
| Query latency (per layer, 128K) | 86us |
| 40-layer query total | 3.4ms |
| Build time (finalize, 128K) | 1.0s |
| Memory (40 layers, 128K) | ~2.5 GB (staging freed after finalize) |
| Quality (k=64 vs full attention) | cosine 1.0 |

### Enabling

```bash
./pion-server --kvcache -w 1
```

### Python client (vllm-pion/)

The `vllm-pion/` package provides drop-in integration:
- `PionKVClient` — KV.STORE / KV.FETCH for raw tensor blobs
- `PionAttentionClient` — ATTEND.* commands with key normalization for INT8 HNSW quantization
- `ExternalizedAttentionLayer` — drop-in replacement for standard attention in vLLM/MLX pipelines

### Mojo modules

| Module | File |
|---|---|
| KVCacheStore | `src/network/kv_cache_store.mojo` |
| LayerStore | `src/network/layer_store.mojo` |
| AttentionIndex | `src/network/attention_index.mojo` |
| BinaryProtocol | `src/network/binary_protocol.mojo` |
| KV.* handlers | `src/commands/kv_cache.mojo` |
| ATTEND.* handlers | `src/commands/attend.mojo` |

---

## Semantic Router

Routes inference queries to the node whose cached KV state is semantically closest. Instead of round-robin or least-connections, Pion routes by meaning.

### Commands

| Command | Syntax | Description |
|---|---|---|
| `AI.ROUTE.REGISTER` | `AI.ROUTE.REGISTER <node_id> <endpoint> <embedding_fp32> [CAPACITY <n>]` | Register inference node with semantic centroid |
| `AI.ROUTE.UPDATE` | `AI.ROUTE.UPDATE <node_id> <new_embedding_fp32>` | Update node centroid (as KV cache evolves) |
| `AI.ROUTE` | `AI.ROUTE <query_embedding_fp32> [EXCLUDE <node_id>]` | Route query to best node by cosine similarity |
| `AI.ROUTE.REMOVE` | `AI.ROUTE.REMOVE <node_id>` | Remove node from routing table |
| `AI.ROUTE.INFO` | `AI.ROUTE.INFO` | Per-node stats: routed count, capacity, endpoint |

### Routing strategy

FP32 brute-force cosine for <=16 nodes (zero recall loss); HNSW O(log N) for >16 nodes. Per-node capacity limits and exclude filters prevent overloading.

### Results

| Metric | Value |
|---|---|
| Routing accuracy | 88% (7/8 queries to correct domain) |
| Latency per route | 0.14ms |
| QPS | 7,176 |

### Mojo modules

| Module | File |
|---|---|
| SemanticRouter | `src/network/semantic_router.mojo` |
| AI.ROUTE.* handlers | `src/commands/route.mojo` |

### Enabling

```bash
./pion-server --kvcache -w 1
```

---

## Speculative RAG

Predicts upcoming RAG queries using embedding-space momentum and pre-executes HNSW search before the query arrives.

### How it works

```
Session starts:
  RAG.SPECULATE.ENABLE session_id  → creates per-session trajectory ring buffer (last 10 embeddings)

Each query:
  RAG.QUERY session_id query_embedding
    1. Append embedding to trajectory ring buffer
    2. Check speculative cache (cosine > 0.9 match against predictions)
       → HIT: return pre-computed HNSW results instantly (0.2ms)
       → MISS: execute live HNSW search
    3. Generate 3 predictions using momentum:
       predicted = current + alpha * (current - previous)
       alpha = [0.5, 1.0, 1.5]
    4. Pre-execute HNSW search for each prediction → store in speculative cache
```

### Commands

| Command | Syntax | Description |
|---|---|---|
| `RAG.SPECULATE.ENABLE` | `RAG.SPECULATE.ENABLE <session_id>` | Enable speculative RAG for a session |
| `RAG.QUERY` | `RAG.QUERY <session_id> <query_embedding_fp32>` | Query with speculation: check cache first, predict next queries |
| `RAG.SPECULATE.INFO` | `RAG.SPECULATE.INFO [session_id]` | Per-session stats: hit rate, trajectory length, predictions |

### Results

| Metric | Value |
|---|---|
| Hit rate (linear trajectory) | 75% (6/8 queries) |
| Prediction latency | 0.2ms |
| Predictions per query | 3 (alpha = 0.5, 1.0, 1.5) |
| Match threshold | cosine > 0.9 |

### Mojo modules

| Module | File |
|---|---|
| SpeculativeRAG | `src/network/speculative_rag.mojo` |
| RAG.* handlers | `src/commands/speculative.mojo` |

### Enabling

```bash
./pion-server --kvcache -w 1
```
