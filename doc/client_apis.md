# Client APIs

Pion speaks the Redis wire protocol (RESP2/RESP3). Any Redis-compatible client library works out of the box — no custom SDK needed.

## Recommended Clients

| Language | Library | Install |
|---|---|---|
| **Python** | `redis-py` | `pip install redis` |
| **Python** | `valkey-py` | `pip install valkey` |
| **Go** | `go-redis` | `go get github.com/redis/go-redis/v9` |
| **Go** | `valkey-go` | `go get github.com/valkey-io/valkey-go` |
| **TypeScript/JS** | `ioredis` | `npm install ioredis` |
| **TypeScript/JS** | `Valkey GLIDE` | `npm install @valkey/valkey-glide` |
| **Rust** | `redis-rs` | `redis = "0.25"` in Cargo.toml |
| **Java/Kotlin** | `Valkey GLIDE` | Maven: `valkey-glide` |
| **C# / .NET** | `StackExchange.Redis` | `dotnet add package StackExchange.Redis` |
| **Swift** | `RediStack` | Swift Package Manager |

## Quick Examples

### Python (redis-py)

```python
import redis

r = redis.Redis(host='127.0.0.1', port=1974, decode_responses=True)

# Key-Value
r.set("key", "value")
r.get("key")  # → "value"

# Hash
r.hset("doc:1", mapping={"title": "Pion", "score": "42"})
r.hget("doc:1", "title")  # → "Pion"

# Vector search (RESP3 binary blob)
# Order matters: create the index, add the vectors, optimize (builds HNSW),
# then search. An HSET issued before FT.CREATE — or after FT.OPTIMIZE — is not
# indexed. The vector field is set on a per-document hash key, not the index.
import struct
r.execute_command("FT.CREATE", "my-index", "SCHEMA", "embedding", "VECTOR", "HNSW",
                  "6", "TYPE", "FLOAT32", "DIM", "1536", "DISTANCE_METRIC", "L2")
for i in range(10):
    vec = struct.pack("1536f", *[0.1 * i] * 1536)
    r.execute_command("HSET", f"doc:{i}", "embedding", vec, "title", f"doc {i}")
r.execute_command("FT.OPTIMIZE", "my-index")           # builds the HNSW graph
query = struct.pack("1536f", *[0.1] * 1536)
r.execute_command("FT.SEARCH", "my-index", "*=>[KNN 10 @embedding $vec]",
                  "PARAMS", "2", "vec", query)

# AI Gateway (requires --flare or an embedding server running)
r.execute_command("FT.ADDTEXT", "kb", "doc:1", "Pion is a low-latency KV + vector engine")
r.execute_command("FT.SEARCHTEXT", "kb", "performance benchmark", "K", "3")
r.execute_command("AI.COMPLETE", "What is the capital of France?", "TOKENS", "100", "THRESHOLD", "0.85")
```

### Go (go-redis)

```go
import "github.com/redis/go-redis/v9"

rdb := redis.NewClient(&redis.Options{
    Addr: "127.0.0.1:1974",
})

rdb.Set(ctx, "key", "value", 0)
rdb.Get(ctx, "key")
rdb.Do(ctx, "FT.SEARCH", "my-index", "*=>[KNN 10 @embedding $vec]", "PARAMS", "2", "vec", blob)
```

### TypeScript (ioredis)

```typescript
import Redis from 'ioredis';

const r = new Redis({ host: '127.0.0.1', port: 1974 });

await r.set('key', 'value');
await r.get('key');
await r.call('FT.ADDTEXT', 'kb', 'doc:1', 'Pion is a Mojo-native vector database');
```

## pion-glide (Recommended for Python)

`pion-glide` is the official async Python client — a thin typed wrapper over [Valkey GLIDE](https://github.com/valkey-io/valkey-glide) that adds `FT.*` and `AI.*` helpers.

```bash
pip install -e pion_glide/   # not published to PyPI; install from a checkout
```

```python
import asyncio
from pion_glide import PionClient

async def main():
    # Standalone
    client = await PionClient.connect("127.0.0.1", 1974)

    # Standard KV
    await client.set("key", "hello from GLIDE")
    print(await client.get("key"))

    # Vector search
    await client.ft.create("products", dim=1536, metric="L2")
    await client.ft.add_vector("products", "sku:1", my_vec)
    await client.ft.optimize("products")
    results = await client.ft.search("products", query_vec, k=10)

    # Text search with server-side embedding (requires --flare)
    await client.ft.add_text("kb", "doc:1", "Pion achieves 10K QPS on Linux")
    hits = await client.ft.search_text("kb", "performance", k=5)

    # AI gateway (requires --flare)
    answer = await client.ai.complete("What is the capital of France?")

    await client.close()

asyncio.run(main())
```

See `pion_glide/README.md` for the full API reference and `examples/pion_glide_demo.py` for a runnable demo.

## Valkey GLIDE (Cluster Mode)

For cluster-mode clients (GLIDE, redis-py cluster, Lettuce, Jedis), start Pion with `--cluster`:

```bash
./pion-server --cluster --cluster-host 127.0.0.1
```

**Via `pion-glide`** (recommended — auto-discovers topology):
```python
from pion_glide import PionClient

# Provide one or more seed nodes; GLIDE discovers the rest via CLUSTER SHARDS
client = await PionClient.connect_cluster([("127.0.0.1", 1974)])
await client.set("{user:42}:session", "tok_xyz")  # hash-tagged key for slot affinity
```

**Via `valkey-py` cluster client:**
```python
from valkey.cluster import ValkeyCluster

r = ValkeyCluster(host='127.0.0.1', port=1974)
r.set("key", "value")
```

## CLI

```bash
redis-cli -p 1974 PING
redis-cli -p 1974 SET foo bar
redis-cli -p 1974 FT.CREATE my-index SCHEMA embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC L2
```

## Framework Integrations

Pion ships native packages for popular AI/ML frameworks. No custom SDK needed — all use standard Redis wire protocol.

### RedisVL + LangChain (drop-in compatible)

```python
# RedisVL — no modifications needed
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
index = SearchIndex(schema, redis_url="redis://localhost:1974")
index.create(); index.load(data); results = index.query(VectorQuery(...))

# LangChain — no modifications needed (requires redis-py < 5.0)
from langchain_community.vectorstores.redis import Redis
vs = Redis.from_texts(texts, embedding, redis_url="redis://localhost:1974")
results = vs.similarity_search("query", k=3)
```

### LangGraph — Agent State Persistence

```python
from pion_langgraph import PionSaver
saver = PionSaver(host="127.0.0.1", port=1974)
graph = workflow.compile(checkpointer=saver)
```

Install: `pip install -e pion-langgraph/`

### AutoGen — Semantic Agent Memory

```python
from pion_autogen import PionMemoryStore
memory = PionMemoryStore(host="127.0.0.1", port=1974)
await memory.add("fact to remember")
results = await memory.query("recall fact")
```

Install: `pip install -e pion-autogen/`

### LlamaIndex — Vector Store for RAG

```python
from pion_llamaindex import PionVectorStore
store = PionVectorStore(host="127.0.0.1", port=1974, dimensions=1536)
index = VectorStoreIndex.from_documents(docs, vector_store=store)
```

Install: `pip install -e pion-llamaindex/`

### LMCache — Wire Compatible

Pion is wire-compatible with LMCache's C++ RedisConnector. Zero code changes — just update the config:

```yaml
remote_url: "resp://localhost:1974"
```

Uses standard GET/SET/EXISTS/DEL with SHA256-keyed KV cache tensor blobs (1-16MB per chunk).

Test: `python3 tests/test_lmcache_compat.py --large`

---

## Connection Notes

- **Port**: 1974 (default; change with `-p <port>`)
- **RESP2/RESP3**: both supported; no client configuration needed. `HELLO 3`
  binds the connection to RESP3 and the protocol is per-connection, so a RESP2
  and a RESP3 client can share a server (and a pub/sub channel) without
  interfering. RESP3 connections get the map type (`%`) for HELLO and
  CONFIG GET, the null type (`_`) instead of `$-1`, and the push type (`>`) for
  pub/sub delivery and subscribe confirmations.

  Before RESP3 mode shipped, `HELLO 3` was answered with a RESP2 array — which
  **crashed redis-py ≥ 8 outright** (it defaults to RESP3 and switches its
  parser before reading the reply). If you are on an older Pion, pass
  `redis.Redis(protocol=2)`. On this build, redis-py ≥ 8 connects with defaults.

  Known remainder: score-shaped replies (`ZSCORE`, `ZINCRBY`, `INCRBYFLOAT`,
  `HINCRBYFLOAT`, `GEODIST`) still go out as bulk strings rather than the RESP3
  double type (`,`). RESP3 clients parse these fine — it is a type-fidelity gap,
  not a framing one — but a client relying on the protocol for float conversion
  will hand you bytes where real Redis hands you a float.
- **Pipelining**: fully supported; benchmark with `-P 10` or higher for throughput testing
- **Cluster mode**: CLUSTER INFO/NODES/SLOTS/SHARDS fully implemented for GLIDE compatibility
