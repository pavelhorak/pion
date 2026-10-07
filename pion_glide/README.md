# pion-glide

> **Experimental.** pion-glide has no test in the suite. No measurement of what it buys is published and nobody outside this project is known to use it, so it is outside Pion's supported surface and may change or be removed. Supported: the prompt cache and `pion-vllm-mlx serve`, the Redis-compatible KV with its WAL, and vector search with the semantic cache ([README](../README.md#experimental)).

Async Python client for [Pion](https://github.com/pavelhorak/pion) built on [Valkey GLIDE](https://github.com/valkey-io/valkey-glide).

Thin wrapper that adds typed `FT.*` (HNSW vector search) and `AI.*` (semantic cache / RAG gateway) helpers on top of GLIDE's standard Redis interface — with full cluster topology discovery, AZ-affinity routing, and OpenTelemetry tracing coming for free from GLIDE.

## Install

```bash
pip install -e pion_glide/   # not published to PyPI; install from a checkout
```

Requires Pion running:

```bash
./pion-server                        # standalone, port 1974
./pion-server --cluster              # cluster mode (for GLIDE cluster client)
./pion-server --flare                # AI features (auto-detect Ollama)
```

## Quickstart

```python
import asyncio
from pion_glide import PionClient

async def main():
    # Standalone connection
    client = await PionClient.connect("127.0.0.1", 1974)

    # Standard KV
    await client.set("greeting", "hello from GLIDE")
    print(await client.get("greeting"))

    # Vector search
    await client.ft.create("products", dim=384, metric="L2")
    await client.ft.add_vector("products", "item:1", [0.1] * 384)
    await client.ft.optimize("products")
    results = await client.ft.search("products", [0.1] * 384, k=5)
    print([r.doc_id for r in results])

    await client.close()

asyncio.run(main())
```

## Cluster Mode

```python
# Start Pion nodes:
#   node-a: ./pion-server --cluster --cluster-host 192.168.1.10 -p 1974
#   node-b: ./pion-server --cluster --cluster-host 192.168.1.11 -p 1974
#              --cluster-nodes 192.168.1.10:1974,192.168.1.11:1974

client = await PionClient.connect_cluster([
    ("192.168.1.10", 1974),
    ("192.168.1.11", 1974),
])
# GLIDE discovers the full slot map via CLUSTER SHARDS — no manual config.
await client.set("{user:42}:session", "tok_xyz")   # hash-tagged for slot affinity
```

## FT.* Vector Search

```python
import struct, random

# 1. Create index
await client.ft.create("items", field="vec", dim=1536, metric="L2")

# 2. Ingest vectors (bulk via HSET)
for i in range(1000):
    vec = [random.random() for _ in range(1536)]
    await client.ft.add_vector("items", f"item:{i}", vec, field="vec")

# 3. Build HNSW graph
await client.ft.optimize("items")

# 4. Search
query = [random.random() for _ in range(1536)]
results = await client.ft.search("items", query, k=10, ef_runtime=150)
for r in results:
    print(r.doc_id, r.score)

# 5. Text search (requires --flare + Ollama)
await client.ft.add_text("docs", "doc:1", "Pion achieves 10K QPS on Linux")
results = await client.ft.search_text("docs", "database performance", k=5)
```

## AI Gateway

Requires `./pion-server --flare` (auto-detects Ollama + `nomic-embed-text`):

```python
# Semantic cache in front of LLM (275× speedup on cache hits)
answer = await client.ai.complete("What is the capital of France?", threshold=0.92)

# Manual cache management
await client.ai.semantic_cache_set("capital of France?", "Paris")
hit = await client.ai.semantic_cache_get("What's the capital of France?")

# RAG chat: retrieve context → augment prompt → LLM
await client.ft.add_text("kb", "doc:1", "Pion achieves 10K QPS on Linux")
response = await client.ai.chat(
    "How fast is Pion?",
    context_index="kb",
    context_query="performance benchmark",
    k=3,
)
```

## Context Manager

```python
async with await PionClient.connect() as client:
    await client.set("key", "value")
    # auto-closes on exit
```

## API Reference

### `PionClient`

| Method | Description |
|---|---|
| `connect(host, port)` | Connect to standalone Pion node |
| `connect_cluster(addresses)` | Connect to Pion cluster |
| `execute(*args)` | Raw command via `custom_command()` |
| `get/set/delete/incr/expire/ttl` | Standard KV |
| `hset/hget/hgetall` | Hash operations |
| `ping()` | Health check |
| `close()` | Close connection |

### `client.ft` — FTIndex

| Method | Description |
|---|---|
| `create(index, field, dim, metric)` | Create HNSW index |
| `optimize(index)` | Build HNSW graph after bulk ingest |
| `drop(index)` | Delete index |
| `info(index)` | Index metadata |
| `add_vector(index, doc_id, vector, field)` | Ingest float32 vector |
| `search(index, query_vec, k, ef_runtime)` | k-NN vector search |
| `add_text(index, doc_id, text)` | Ingest text (server-side embed) |
| `search_text(index, query, k)` | Text search (server-side embed) |

### `client.ai` — AIGateway

| Method | Description |
|---|---|
| `complete(prompt, tokens, threshold)` | Semantic cache + LLM in one call |
| `semantic_cache_set(query, response)` | Store in semantic cache |
| `semantic_cache_get(query, threshold)` | Look up semantic cache |
| `chat(prompt, context_index, context_query, k)` | RAG chat |
| `flare_run(index, prompt, max_tokens)` | FLARE mid-generation retrieval |

## Why GLIDE?

[Valkey GLIDE](https://github.com/valkey-io/valkey-glide) is the reference multi-language client for Valkey/Redis:
- **Cluster topology discovery** — automatically discovers all nodes from a single seed
- **MOVED/ASK redirect handling** — transparent slot migration during live resharding
- **AZ-affinity routing** — routes reads to the nearest replica (cloud cost savings)
- **OpenTelemetry** — built-in distributed tracing, no middleware needed
- **Rust backend** — lower latency, lower CPU vs pure-Python clients

For Pion-specific commands (`FT.*`, `AI.*`), GLIDE's `custom_command()` sends arbitrary RESP arrays — the same as `redis-cli` would, just async and cluster-aware.

## License

Apache 2.0
