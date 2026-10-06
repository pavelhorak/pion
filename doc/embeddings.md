# Native Embedding Generation

Pion can absorb embedding generation into its data path. Instead of a client passing pre-computed float vectors, the client sends raw text -- Pion generates the vector and indexes it in a single round-trip, eliminating the embedding service as a separate hop.

---

## Current Implementation

FT.ADDTEXT and FT.SEARCHTEXT are implemented and working. Pion supports four embedding backends with a single cascade applied across every embed-using command (`AI.EMBED`, `AI.SEMANTIC_CACHE`, `AI.MEMORY`, `FT.SEARCHTEXT`, `AI.FLARE.*`):

### Embedding Cascade

1. **Apple NLEmbedding (`--nle-embed`, macOS only):** Apple's system `NaturalLanguage` framework. 512-dim sentence embeddings, **no Python sidecar, no model download**. Selected first when available; falls through if NLE doesn't recognize the input. See `src/ffi/nle_wrap.m` and `src/network/nle_embedding_engine.mojo`.
2. **Auto-embed sidecar (cross-platform default):** Pion auto-launches a Python inference sidecar with **MiniLM-L6-v2** (384 dimensions, ~90 MB download on first run). Requires `torch` + `transformers` — install with `pixi run install-inference`. Disable with `--no-auto-embed` or `--profile kv`.
3. **SIE (`superlinked/sie`, opt-in via pion-serve `--sie-url`):** OpenAI-compatible `/v1/embeddings` server with 85+ MTEB-verified models, autoscale-to-zero K8s deployment, Apache 2.0. Recommended when the deployment already runs SIE for production traffic. Explicit URL only — no auto-detect on port 8000 (collision with vLLM / FastAPI / generic web apps). When the URL is set, slots into the pion-serve cascade ahead of Ollama. Force-select via `--embed-backend sie --sie-url http://host:8000`; force-fail if unreachable (no cascade fall-through).
4. **Ollama** (auto-detected): If Ollama is running on port 11434 with `nomic-embed-text` (768 dimensions), Pion uses it via HTTP. Detected automatically at startup unless `--no-auto-detect` is set.
5. **HTTP embedding API**: Any OpenAI-compatible `/v1/embeddings` endpoint or MAX Serve.

**pion-serve telemetry:** `/v1/stats` reports the active embed backend and per-backend call counts under `embedder.{backend, calls.{sie,ollama,pion}}`. This lets ops dashboards split SIE hits from Ollama hits without re-running the proxy.

The auto-embed sidecar uses the InferenceBridge (Unix socket IPC) for sub-millisecond embedding latency. The HTTP path is used as fallback when the sidecar is unavailable. NLE is in-process — no IPC at all.

**Compatibility note:** Mac with `--nle-embed` produces 512-dim semantic-cache vectors; Linux with the PyTorch sidecar produces 384-dim. The two are not cross-compatible; semantic-cache HNSW indexes built on one don't migrate to the other. For any cross-platform persistence requirement, version the cache by embedder identity.

### Commands

```
FT.ADDTEXT  <index> <doc_id> "This is the raw text to embed."
FT.SEARCHTEXT <index> "What is the capital of France?"
```

**FT.ADDTEXT:** Sends the text to the embedding service, receives a float vector, and passes it directly to `hnsw.add_vector()`. The text is also stored in the document hash and registered in the BM25 doc set, so it is indexed at the next `FT.OPTIMIZE`. **Use a non-negative integer `doc_id`** if the doc should be BM25-searchable; BM25 returns integer ext_ids, so string ids are reachable only through `FT.SEARCHTEXT`.

**FT.SEARCHTEXT:** Sends the query text to the embedding service, receives a float vector, and runs `hnsw.search_fp32_scored()`. Returns results in standard FT.SEARCH RESP2 format.

### Default Model

Pion uses **MiniLM-L6-v2** (384 dimensions) by default via the auto-embed sidecar. When Ollama is detected, it uses **nomic-embed-text** (768 dimensions) instead.

**The dimension is not auto-detected.** It is fixed at startup from exactly one of: `768` (the `EmbeddingConfig` default, which is also what the Ollama auto-probe assumes), `512` (`--nle-embed`), `384` (the auto-embed sidecar), or `--emb-dim N`. `FT.CREATE … DIM` sets the *main* HNSW used by `FT.SEARCH`/`FT.HYBRID`; it has no effect on the embedding path.

A width mismatch is silent, not an error: `SemanticCache.embed_into` rejects a vector whose width differs from the configured dimension and falls through to the next cascade tier, which under `--nle-embed` has no HTTP host configured and therefore always fails. The symptom is empty `FT.SEARCHTEXT` results, not a startup complaint — so `--emb-dim` must match the model exactly.

### Selecting a Different Backend

The external embedding backend defaults to Ollama at `127.0.0.1:11434` with model `nomic-embed-text` at 768 dimensions. Four flags change it:

```bash
--emb-model NAME    # model id sent to the backend (default nomic-embed-text)
--emb-host HOST     # default 127.0.0.1
--emb-port N        # default 11434
--emb-dim N         # default 768 — MUST match the model's output width
```

Each one implies `--emb-enabled`. The request is a plain `POST /v1/embeddings` with `{"model": …, "input": …}`, so any OpenAI-compatible server works — Ollama, llama.cpp's `llama-server --embeddings`, SIE, MAX Serve. There is no `Authorization` header, so token-authenticated hosted providers are still out of reach.

**Recommended stronger on-device default — EmbeddingGemma-300M:**

```bash
ollama pull embeddinggemma
./pion-server --emb-enabled --emb-model embeddinggemma --emb-dim 768 --no-auto-detect -w 1
```

768-dim, ~620 MB, and notably it needs no Hugging Face account: `google/embeddinggemma-300m` is a gated repo on HF, so the PyTorch-sidecar route (`--inference-emb-model`) requires an accepted licence and a token, while the Ollama route does not.

The sidecar accepts any Hugging Face encoder via `--inference-emb-model <hf-id>` (it is a generic `AutoTokenizer` + `AutoModel` load with mean pooling and L2 normalisation).

**Instruction prefixes for asymmetric retrievers:** EmbeddingGemma, E5, BGE and similar models are trained with distinct query vs document prefixes and lose accuracy without them. Two opt-in flags prepend a prefix at embed time — the query prefix on `FT.SEARCHTEXT` / `AI.SEMANTIC_CACHE` / `AI.MEMORY`, the document prefix on `FT.ADDTEXT`:

```bash
--emb-query-prefix "task: search result | query: "
--emb-doc-prefix   "title: none | text: "
```

Both default to empty, so symmetric models (MiniLM, nomic) are untouched. Note that Ollama's `embeddinggemma` modelfile is `TEMPLATE {{ .Prompt }}` — it applies no prefix itself, so these flags are how you supply them. EmbeddingGemma's model card specifies these prefixes; no measurement of what they change is published here, so measure on your own retrieval set.

For VectorDBBench benchmarks (Performance1536D50K), the client sends pre-computed 1536-dimensional OpenAI embeddings directly via HSET -- no server-side embedding generation is needed.

### FT.ADDTEXT / FT.SEARCHTEXT limits and gotchas

The dense text path runs against the **per-worker semantic-cache HNSW**, which is a different graph from the one `FT.SEARCH`/`FT.HYBRID` use. That distinction drives most of the surprises:

- **10,000 documents per worker.** `CACHE_MAX_ENTRIES = 10000`; `FT.ADDTEXT` past that still returns `+OK`, still registers the doc for BM25, and silently skips the embedding. The lexical and dense doc sets then diverge with no error.
- **`FT.DROPINDEX` does not clear it.** The handler resets the main HNSW only — it never receives the semantic cache. Documents from a previous corpus stay in the dense results and crowd the top-K, so a caller reusing one server across corpora must namespace its doc ids and filter results. Restart the server for a clean dense index.
- **`FT.HYBRID` does not embed.** It takes a pre-computed vector blob for its dense leg; only `FT.SEARCHTEXT` and `FT.ADDTEXT` call the embedder.
- **Ingest is quadratic.** `add_and_insert` recompacts the whole vector buffer after every insert, so bulk `FT.ADDTEXT` of N docs is O(N²) — a few thousand documents takes minutes.
- **Near-duplicate corpora cap result depth.** Passages that differ only in an index number embed to near-identical vectors; the beam ties on distance and stops expanding after roughly one neighbour list, so `FT.SEARCHTEXT` returns ~2*M results regardless of `K` or `ef`. This is HNSW behaviour on degenerate data, not a bug.
- **Two servers must not share a working directory.** `pion.wal.0` and `pion.hnsw.0` are written relative to cwd, so concurrent instances trample each other's WAL and reload each other's persisted graph. The failure looks like corrupt retrieval results, not an I/O error.

### Hybrid Search

FT.ADDTEXT documents with integer ids are available to both vector search (FT.SEARCH, FT.SEARCHTEXT) and BM25 full-text search (`FT.SEARCH … BM25`) after `FT.OPTIMIZE`. The `FT.HYBRID` command combines both in a single query using Reciprocal Rank Fusion. See `doc/vector_engine.md` for full FT.HYBRID and BM25 documentation, including the per-query `K1`/`B` knobs.

---

## The Problem with Traditional RAG Pipelines

A typical RAG insertion or search today:

1. Client -> Embedding API (raw text)
2. Embedding API -> Client (1536 floats, ~6KB)
3. Client -> Vector DB (floats over network)
4. DB -> Client (search results)

**The cost:** Two large float-array network transfers, two serialization rounds, and an external service dependency for every single query.

## The Pion Solution (Current + Planned)

### Current: HTTP Embedding Path

```
Client -> Pion (text)
Pion -> Ollama/MAX Serve (text, localhost)
Ollama -> Pion (float vector, localhost)
Pion -> HNSW search (zero-copy)
Pion -> Client (results)
```

This eliminates the client-side embedding round-trip. The embedding call is localhost HTTP (~1-5ms) rather than a cross-network call.

### Auto-Embed Sidecar

The auto-embed sidecar (`src/inference/worker.py`) loads MiniLM-L6-v2 via PyTorch/transformers and communicates with Pion via Unix socket IPC (InferenceBridge). This eliminates the need for Ollama or any external embedding service.

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~90 MB download on first run)
- **Startup:** Auto-launched by Pion's main thread; the server listens before the sidecar is ready, so early clients queue rather than fail
- **Semantic cache threshold:** 0.85 (lower than 768-dim's 0.95 to account for reduced precision)
- **Disable:** `--no-auto-embed` or `--profile kv`
- **Test:** `python3 tests/test_auto_embed.py`

### `--nle-embed`: Apple NLEmbedding (Shipped, macOS only)

`src/ffi/nle_wrap.m` wraps Apple's `NaturalLanguage` framework; `src/network/nle_embedding_engine.mojo` is the Mojo wrapper. Same architectural pattern as `MetalAttentionEngine` — `comptime if CompilationTarget.is_macos():` guards keep the Linux build clean.

- **Model:** Apple-bundled English sentence embedding (system framework, no download)
- **Dimension:** 512
- **Acceleration:** chosen by Apple's framework; Pion calls the high-level API and does not pick the compute unit
- **Startup:** initialized at first worker spawn, ~milliseconds (no model load)
- **Semantic cache threshold:** 0.80 (NLE's vector space is less discriminating than MiniLM-L6-v2 between paraphrases vs topic shifts; tune per-deployment)
- **Coverage:** `AI.EMBED`, `AI.SEMANTIC_CACHE`, `AI.MEMORY`, `FT.SEARCHTEXT`, `AI.FLARE.*`
- **Validation:** the same `NLEmbedding` API is validated on iOS for cross-language semantic search.
- **Output:** L2-normalized to unit norm before return — Pion's INT8 HNSW kernel assumes unit-norm input. NLE's raw vectors have norm ≈ 11 at 512-dim; without normalization, the INT8 quantizer saturates and the cosine threshold becomes meaningless.

## Where the time goes

A traditional pipeline crosses the network twice per query: text to an embedding
API, then the returned vector to the vector database, serialized both times.
Pion takes the text, embeds it on the same machine (sidecar, Ollama or
`NLEmbedding`) and searches in the same process. No latency comparison between the
two is published with a harness, so this page states the hops rather than a number.

### The Float Tax Eliminated

- **Traditional**: 768 floats x 4 bytes = ~3KB transferred twice (API -> Client -> DB) plus serialization overhead
- **Current Pion**: 768 floats transferred once (Ollama -> Pion, localhost only) -- client sends only text
