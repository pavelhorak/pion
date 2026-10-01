# pion-mcp

MCP server for [Pion](https://github.com/pavelhorak/pion) — the Redis-compatible vector database
built in Mojo. Exposes Pion's semantic search and key-value store as tools for Claude Code, Cursor,
GitHub Copilot, and any MCP-compatible agent.

## Quickstart

```bash
# Add to Claude Code (one line)
claude mcp add pion -- uvx --from ./mcp pion-mcp

# Or with explicit host/port
claude mcp add pion -- uvx --from ./mcp pion-mcp --host localhost --port 1974
```

Pion must be running before the MCP server connects:
```bash
./pion-server          # starts on port 1974
```

## Tools

### Vector Search & Indexing

| Tool | Description |
|---|---|
| `vector_search` | Semantic search — embeds your query and returns similar documents |
| `add_document` | Add text + auto-embed to a vector index |
| `add_documents_bulk` | Bulk insert with batched embeddings (faster for >10 docs) |
| `create_index` | Create an HNSW vector index |
| `optimize_index` | Build the HNSW graph after inserting documents |
| `index_info` | Inspect index metadata |
| `drop_index` | Delete an index |

### Key-Value Store

| Tool | Description |
|---|---|
| `kv_get` / `kv_set` / `kv_delete` | Standard key-value operations |
| `kv_mget` | Fetch multiple keys in one round-trip |
| `kv_incr` | Atomic integer counter |
| `hash_set` / `hash_get` / `hash_get_all` | Structured document storage |

### Semantic Cache

| Tool | Description |
|---|---|
| `semantic_cache_set` | Cache an LLM response indexed by query meaning |
| `semantic_cache_get` | Return cached response if a similar query was asked before |

### Agent Memory

Persistent cross-session semantic memory. Memories are embedded, stored in Pion's HNSW index,
and retrievable by meaning — not just exact key lookup. Survives process restarts.

| Tool | Description |
|---|---|
| `agent_remember` | Store a memory (fact, decision, context) by semantic content |
| `agent_recall` | Retrieve the most similar memories to a query |
| `agent_forget` | Delete a specific memory by key |
| `agent_forget_session` | Delete all memories from a session |
| `agent_memory_stats` | Show memory count and index status |

### Utility

| Tool | Description |
|---|---|
| `ping` | Check Pion connectivity |
| `server_info` | Pion server stats |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PION_HOST` | `localhost` | Pion server host |
| `PION_PORT` | `1974` | Pion server port |
| `PION_EMBED_PROVIDER` | `openai` | Embedding provider: `openai`, `max`, `mock` |
| `PION_EMBED_MODEL` | `text-embedding-3-small` | Model name |
| `PION_EMBED_DIM` | `1536` | Embedding dimension (must match index) |
| `PION_EF_RUNTIME` | `150` | HNSW search beam width |
| `OPENAI_API_KEY` | — | Required for `openai` provider |
| `PION_EMBED_URL` | `http://localhost:8000/v1` | MAX Serve URL for `max` provider |

### Use with MAX Serve (no OpenAI key needed)

```bash
# Start MAX embedding server
max serve --model sentence-transformers/all-MiniLM-L6-v2 &

# Start Pion MCP with MAX backend (384-dim)
PION_EMBED_PROVIDER=max PION_EMBED_DIM=384 uvx --from . pion-mcp
```

### Testing without any API key

```bash
PION_EMBED_PROVIDER=mock uvx --from . pion-mcp
```

The `mock` provider uses a deterministic hash-based fake embedding — useful for testing tool
integration before setting up a real embedding model.

## Example: Agent Memory Store

With pion-mcp added to Claude Code, you can tell Claude:

> "Remember that the project deadline is March 31st"
> → `agent_remember("Project deadline is March 31st", session_id="my-project")`

> "What do you remember about our build system?"
> → `agent_recall("build system", k=3)` — returns semantically similar memories

Memories persist across sessions and restarts (stored in Pion's WAL + HNSW disk snapshot).

### Storage layout

```
HNSW index:  "__agent_memory__"
Hash keys:   "mem:1", "mem:2", ...  (sequential integer suffix for HNSW routing)
Fields:      text, session_id, timestamp, embedding, [custom metadata]
Counter:     "__mem_seq__"  (sequential ID), "__mem_count__"  (optimize trigger)
```

### Agent Memory vs `kv_set`

| | `kv_set` | `agent_remember` |
|---|---|---|
| Lookup | Exact key | Semantic similarity |
| Persistence | WAL | WAL + HNSW snapshot |
| Cross-session | ✓ | ✓ |
| Natural-language query | ✗ | ✓ |

> "What documents are similar to 'distributed systems performance'?"
> → `vector_search("docs", "distributed systems performance", k=5)`

> "Add this article to my research index"
> → `add_document("research", "article:42", "Full article text here...")`

## Example: Semantic Cache

```python
# In your LLM application, use pion-mcp to cache responses:
cached = semantic_cache_get("What is the capital of France?", threshold=0.95)
if cached:
    return cached  # 70-86% of repeated queries hit cache

response = call_llm("What is the capital of France?")
semantic_cache_set("What is the capital of France?", response)
return response
```

## Performance

Pion at ef=150, 50K vectors, 1536 dimensions (VectorDBBench Performance1536D50K,
head-to-head on same machine — Linux Colima 8-CPU, 2026-03-23):

| Metric | Redis VSET (Redis 8.0) | **Pion V37** | Advantage |
|---|---|---|---|
| Peak QPS (c=10) | 5,441 | **10,283** | **+89%** |
| P99 latency | 0.9ms | **0.9ms** | equal |
| Recall@100 | 0.9197 | **0.9371** | **+1.7pp** |
| Load time | 40.5s | **17.6s** | **2.3× faster** |

macOS (M-series): 8,134 QPS mean (3-run stable), recall 0.9371, load ~16.5s vs Redis 50.4s.

These figures are from an earlier build (spring 2026). The build as of
2026-09-30 measures ~9.4K QPS at recall 0.960 on an M4 Mac mini, at the same
gate configuration (50K × 1536-d, ef=150).

## License

Apache-2.0. Pion's satellites are deliberately permissive so they can be vendored
into any stack; the Pion **server** itself is Apache-2.0 too,
with one closed binary library for its tuned vector kernels — see the top-level
`LICENSE` and `doc/licensing.md`.
