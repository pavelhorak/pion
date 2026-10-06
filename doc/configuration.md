# Configuration

Pion's defaults live in `src/common/config.mojo` and are overridden with CLI flags at startup.
There is no config file. Only `maxmemory` can be changed at runtime with `CONFIG SET` (every other key is refused with an explanatory error)
(`maxmemory` among them).

---

## Config Structs (`src/common/config.mojo`)

### `ServerConfig`

| Field | Default | Description |
|---|---|---|
| `port` | `1974` | TCP listen port |
| `workers` | `1` | Worker thread count. Each worker is an **independent keyspace**, so `-w N` with N > 1 exits unless `--independent-workers` is also passed |
| `profile` | `"auto"` | Smart profile: `"desktop"`, `"cloud"`, `"embedded"`, or deployment profile via `--profile` |
| `use_iouring` | `false` | Force io_uring event loop (Linux default when neither flag set) |
| `use_epoll` | `false` | Force epoll event loop (Linux, best for P=1 w=1 benchmarks) |
| `use_sqpoll` | `false` | io_uring SQPOLL: kernel-side SQ polling (Linux 5.11+, root, experimental) |
| `use_xdp` | `false` | XDP/AF_XDP kernel bypass (Linux 5.4+, CAP_NET_ADMIN) |
| `xdp_interface` | `"eth0"` | NIC interface for XDP attach |
| `use_huge_pages` | `false` | 2MB mmap pages (Linux; ignored on macOS) |
| `strict_affinity` | `false` | Pin each worker to a dedicated P-core |
| `enable_sharding` | `false` | HNSW worker sharding (Linux only; macOS kqueue cost too high) |
| `independent_workers` | `false` | `--independent-workers`: acknowledges that N workers are N keyspaces, and lets `-w N > 1` start |
| `wal_size_mb` | `256` | WAL segment size; a full segment is sealed and a new one opened |
| `wal_max_segments` | `32` | Past this, writes are refused (`-MISCONF`), never silently dropped |
| `blob_threshold` | `1048576` | Values at or above this size go to the mmap'd blob tier |
| `bind_addr` | `""` | Listen address (`--bind`); read the security model before exposing a port |

### `VectorConfig`

| Field | Default | Description |
|---|---|---|
| `dimensions` | `1536` | Embedding dimension (OpenAI default) |
| `max_elements` | `600000` | Max vectors per worker HNSW |
| `M` | `16` | HNSW graph degree |
| `ef_construction` | `100` | Build-time beam width |
| `use_int4` | `false` | Global INT4 quantization (not recommended: low recall; use `polarquant`) |
| `use_bq` | `false` | Binary quantization (not recommended: low recall on embedding data) |
| `polarquant` | `false` | Block-INT4 quantization (`--polarquant`) |
| `turboquant` | `false` | Block-INT3 + QJL quantization (`--turboquant`) |
| `nanoquant` | `false` | Block-INT2 quantization (`--nanoquant`, experimental) |
| `has_gpu` | `false` | GPU path (auto-detected via `env.mojo`) |

Measured recall and QPS for each quantization mode: `doc/vector_engine.md` § Quantized variants.

### `EmbeddingConfig`

Required for `FT.ADDTEXT`, `FT.SEARCHTEXT`, `AI.SEMANTIC_CACHE GET/SET`.

| Field | Default | Description |
|---|---|---|
| `host` | `"127.0.0.1"` | Embedding server host |
| `port` | `11434` | Ollama default |
| `model` | `"nomic-embed-text"` | Model name sent in `/v1/embeddings` request |
| `dimensions` | `768` | Output embedding dimension |
| `threshold` | `0.95` | Cosine similarity gate for `AI.SEMANTIC_CACHE GET` |
| `enabled` | `false` | Auto-enabled when Ollama auto-detect finds `nomic-embed-text`; or set `True` manually |

### `LLMConfig`

Required for `AI.CHAT` and `AI.COMPLETE`.

| Field | Default | Description |
|---|---|---|
| `host` | `"127.0.0.1"` | LLM server host |
| `port` | `8000` | MAX Serve / OpenAI-compatible default |
| `model` | `"meta-llama/Llama-3.1-8B-Instruct"` | Model name in JSON body |
| `enabled` | `false` | Auto-enabled when Ollama auto-detect succeeds; or set `True` manually |

### `ClusterConfig`

Required for GLIDE/cluster-mode clients.

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable cluster mode (CLUSTER INFO/NODES/SLOTS/SHARDS) |
| `my_host` | `"127.0.0.1"` | Advertised IP in CLUSTER NODES output |
| `peer_nodes` | `""` | Comma-separated `"host:port,..."` (multi-node, future) |

### `AIConfig`

| Field | Default | Description |
|---|---|---|
| `enable_gateway` | `false` | AI Gateway activation flag (informational) |
| `enable_max_engine` | `false` | MAX InferenceSession (not implemented) |
| `model_path` | `"models/all-MiniLM-L6-v2.onnx"` | Local model path for future in-process inference |

---

## Smart Profiles

Chosen from the online core count at startup (`src/common/env.mojo`), before
`--profile` and the other flags are applied:

| Profile | Condition | Workers | Huge Pages | Affinity | Notes |
|---|---|---|---|---|---|
| `embedded` | `nproc ≤ 4` | 1 | off | off | BQ disabled |
| `desktop` | `5 ≤ nproc ≤ 15` | 1 | on | off | M=16, INT8 |
| `cloud` | `nproc ≥ 16` | 1 | on | on | `ef_construction=200`, INT8 |

Every profile starts **one** worker. More workers are a throughput option you
opt into with `-w N --independent-workers`, because each worker owns a private
keyspace and a pooled client spreads its connections across them — see
[Running in production](operations.md) §2b. On macOS the count is further capped
at 4, and huge pages fall back to standard `mmap`. `--flare`, `--inference` and
`--profile ai` force a single worker, as does Ollama auto-detection enabling
embeddings (pass `--no-auto-detect --no-auto-embed` for a multi-worker test).

---

## Deployment Profiles (`--profile`)

Override the smart profile with a deployment-specific configuration:

```bash
./pion-server --profile kv       # KV-only: ~50MB/worker (no HNSW, no AI)
./pion-server --profile vector   # KV + HNSW vector search (no AI)
./pion-server --profile ai       # KV + HNSW + AI features (single worker)
./pion-server --profile full     # Everything enabled
```

| Profile | HNSW | AI/Embedding/LLM | Workers | RAM/Worker | Use Case |
|---|:---:|:---:|:---:|:---:|---|
| `kv` | Stub (1 element) | Disabled | 1, or N with `--independent-workers` | ~50 MB | Redis replacement, caching, sessions |
| `vector` | Full (600K) | Disabled | 1, or N with `--independent-workers` | ~700 MB | Semantic search, RAG knowledge base |
| `ai` | Full (600K) | Enabled | 1 (forced) | ~780 MB | AI agent memory, semantic cache |
| `full` | Full (600K) | Enabled | 1, or N with `--independent-workers` | ~920 MB | Everything |

The `kv` profile saves ~650MB/worker by allocating a minimal 1-element HNSW stub instead of the full 600K-element graph. Startup is near-instant vs 10-15s for vector profiles.

`--profile` sets its defaults at its position in argv; a later flag overrides them (so `--profile kv -w 4 --independent-workers -p 6379` runs 4 workers). The one exception is the `ai` profile's single-worker cap, which is enforced after the whole command line is parsed, so it holds regardless of flag order.

---

## CLI Flags

The authoritative list is `./pion-server --help`; the docs site renders the same
printer as its CLI flags reference, so the two cannot disagree.

**Ollama auto-detect:** When `--no-auto-detect` is absent, Pion probes `127.0.0.1:11434` at startup. If `nomic-embed-text` is found in `/api/tags`, embedding and LLM are auto-enabled (model: `llama3.2:1b`).

---

## Memory limit (`--maxmemory`)

```bash
./pion-server --maxmemory 8gb        # bytes, k/kb/m/mb/g/gb
./pion-server --maxmemory 60%        # or a share of physical RAM
redis-cli CONFIG SET maxmemory 4gb   # change it at runtime; 0 turns it off
```

The limit applies to the process **RSS** — what the OS measures when it
decides to kill a process, and what `INFO` reports as `used_memory`. Above it,
commands that grow memory (Redis's `denyoom` set, plus Pion's substrate ingest
commands) are refused with Redis's exact error:

```
-OOM command not allowed when used memory > 'maxmemory'.
```

Reads, `DEL` and the POP family keep working, so a client can free memory.
The policy is `noeviction`: **nothing is ever evicted.** Inside a transaction,
`MULTI` refuses the command at queue time, and a limit crossed after queueing
makes `EXEC` answer `EXECABORT` with nothing applied. A script runs, and only a
memory-growing `redis.call` inside it errors. The binary lane on port+1 refuses
its store opcodes the same way. Every crossing is logged with the measured RSS.

Default is `0` (off): the server grows until the OS kills it, which the crash
breadcrumbs in `doc/operations.md` §1 will show.

---

## Scripts (`--lua-time-limit`, `--lua-memory-limit`)

```bash
./pion-server --lua-time-limit 2000      # ms; default 5000, 0 = never
./pion-server --lua-memory-limit 256mb   # per Lua state; default 1gb, 0 = no cap
```

A worker runs one thing at a time, so while a script runs it cannot answer
another client. That includes the `SCRIPT KILL` and the `BUSY` reply that Redis
uses for a slow script.

Instead, a script that has run for `--lua-time-limit` milliseconds without
writing anything is stopped with
`ERR Script killed: it ran longer than lua-time-limit (… ms) without writing`,
which is when Redis's SCRIPT KILL would be allowed to stop it. A script that has
written keeps running, as an unkillable script does in Redis.

`--lua-memory-limit` caps each Lua state's heap: a worker has one state for EVAL
scripts and one for FUNCTION libraries. Redis has no such cap.

---

## Current Defaults

What `./pion-server --no-auto-detect` printed at startup on a 10-core Apple
Silicon Mac from a source checkout (`desktop` profile). Note the embedding
sidecar: auto-embed starts the MiniLM worker by default whenever
`src/inference/worker.py` resolves, even with no Ollama running. Pass
`--no-auto-embed` for a KV-only process, or `--nle-embed` on macOS for Apple's
in-process embedding.

```
Profile:    desktop
Port:       1974
Workers:    1
Huge Pages: True        (falls back to standard mmap on macOS)
Affinity:   False
Vector Dim: 1536        (M=16, ef_construction=100)
INT4:       False
PolarQuant: False
TurboQuant: False
NanoQuant:  False
BQ:         False
Embedding:  127.0.0.1:11434 model=nomic-embed-text
LLM:        disabled
Inference:  enabled (socket=/tmp/pion_inference.sock)
  Emb Model: sentence-transformers/all-MiniLM-L6-v2
KV Cache:   False
Cluster:    False
```
