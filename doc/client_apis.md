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
r.execute_command("FT.SEARCH", "my-index", "*=>[KNN 10 @embedding $vec AS score]",
                  "PARAMS", "2", "vec", query,
                  "RETURN", "2", "title", "score",   # without RETURN every field comes back,
                  "DIALECT", "2")                     # the binary embedding too

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

## Valkey GLIDE (Cluster Mode)

For cluster-mode clients (GLIDE, redis-py cluster, Lettuce, Jedis), start Pion with `--cluster`:

```bash
./pion-server --cluster --cluster-host 127.0.0.1
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

  `FT.SEARCH`, `FT.HYBRID` and `FT.INFO` answer RESP3 connections with
  RediSearch's RESP3 maps (`FT.SEARCH`: `attributes`, `format`, `results`,
  `total_results`, `warning`, one map per document). Through 0.9.8 they sent
  their RESP2 arrays on every connection, and redis-py ≥ 8's search module,
  which reads the maps, failed in `ft().search()` and `ft().info()`. With 0.9.8
  or older, pass `redis.Redis(protocol=2)`. A raw
  `execute_command("FT.SEARCH", …)` on a RESP3 connection returns the map
  (a dict in redis-py), as from Redis: index it as a list only on RESP2.

  `ZSCORE` and `ZINCRBY` reply with the RESP3 double type (`,`).
  `INCRBYFLOAT`, `HINCRBYFLOAT` and `GEODIST` stay bulk strings under RESP3,
  as they do in Redis 8.
- **Pipelining**: fully supported; benchmark with `-P 10` or higher for throughput testing
- **Cluster mode**: CLUSTER INFO/NODES/SLOTS/SHARDS fully implemented for GLIDE compatibility
