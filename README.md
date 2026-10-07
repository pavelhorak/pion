<picture>
  <source media="(prefers-color-scheme: dark)" srcset="dark-logo.png">
  <source media="(prefers-color-scheme: light)" srcset="light-logo.png">
  <img alt="Pion Logo" src="light-logo.png" width="400">
</picture>

# Pion — The Memory Engine for AI Inference

[![Mojo](https://img.shields.io/badge/Mojo-1.1-orange.svg)](https://docs.modular.com/mojo/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

📚 **Documentation:** [`doc/index.md`](doc/index.md) — full reference.

<!-- --8<-- [start:pitch] -->
**Pion remembers what your model already read:** prefill a long prompt once,
and every later request, from any process and even after a crash, starts at
the first new token. It is a memory engine for AI inference — one small Mojo
binary that holds the prompt's K/V cache, and the rest of a model's working
memory, and serves it over the Redis wire.
<!-- --8<-- [end:pitch] -->

<!-- --8<-- [start:open-core] -->
**Pion is open source under Apache-2.0 — with one exception, stated up
front.** The tuned 1536-dim vector search kernels ship as a free, closed
binary library (`libpion_vector`). That is open core. The same algorithms are
open in `src/vector/reference/`, a differential test proves on every release
that the two return identical results, and `pixi run build-open` builds Pion
entirely from source, with vector search at 1536 dims slower — in the default INT8 mode by 26% on
an M4 Mac mini and 40% on an EPYC 8124P, and by 36–55% in the quantized modes on
the Mac — and everything else — the KV engine, persistence, the prompt cache — unchanged.
Details: [`doc/licensing.md`](doc/licensing.md).
<!-- --8<-- [end:open-core] -->

## What works

Three things, each covered by the gate tier that runs on every change. Every
other feature in this repository is [experimental](#experimental): it is
there, it may be tested at the wire level, and it is not something to rely on.

### 1. A prompt cache for mlx-lm, and one endpoint for coding agents

On Apple Silicon it is four lines around `mlx_lm`:

<!-- --8<-- [start:four-lines] -->
```bash
# Homebrew: brew install pavelhorak/tap/pion && brew services start pion   (runs with these flags)
# Tarball or source build:
./pion-server --kvcache --metal-attention -w 1
```

```python
from pion_vllm_mlx import PionPromptCache

pc = PionPromptCache(model, host="127.0.0.1", port=1974)                       # once per loaded model
cache = pc.get_or_prefill(prefix_ids, namespace="app|v1|llama-1b|system_v1")  # MISS: prefill once + store · HIT: fetch
text = generate(model, tok, prompt=suffix_ids, prompt_cache=cache)             # decode as usual, at native speed
```
<!-- --8<-- [end:four-lines] -->

<!-- --8<-- [start:four-lines-explained] -->
`get_or_prefill` returns what `mlx_lm.make_prompt_cache(model)` would, with the
prefix's K/V already in it. `prefix_ids` is the part every request shares (the
system prompt, the few-shot block, the retrieved chunk); `suffix_ids` is the
part that changes. The first process to ask pays the prefill once and stores
it; every later call — this process, another one, or after a restart — fetches
it. That is the cross-process row in the table below. The same-process row
adds the Stage-2 attention patch, in which Pion computes the attention over
the prefix itself ([`pion-vllm-mlx/README.md`](pion-vllm-mlx/README.md)).
<!-- --8<-- [end:four-lines-explained] -->

<!-- --8<-- [start:pip-install] -->
Runnable end to end against a running Pion:
[`examples/prompt_cache_demo.py`](examples/prompt_cache_demo.py) times vanilla
mlx-lm's cold prefill against five cached requests, each followed by a full
answer, and prints every one. Install with `pip install 'pion-vllm-mlx[mlx]'`.
<!-- --8<-- [end:pip-install] -->

<!-- --8<-- [start:serve] -->
**For coding agents:** `pion-vllm-mlx serve` puts one local MLX model behind
Claude Code (Anthropic `/v1/messages`), Codex (OpenAI `/v1/responses`) and any
OpenAI client, with the prompt cache in Pion. Within one live session it does
what a stock local server does. After a restart, in a second session, or from
a second tool, a prompt starts from the prefix Pion stored instead of a full
prefill, and the restored prefix gives the same greedy reply.

```bash
pion-vllm-mlx serve --model mlx-community/Qwen3-4B-4bit --port 8080
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080    # Claude Code; Codex and OpenAI clients: see the guide
```

Setup for each client, what it does not do (hybrid models are not stored yet),
and how it stores a conversation: [`doc/coding_agents.md`](doc/coding_agents.md).
<!-- --8<-- [end:serve] -->

<!-- --8<-- [start:two-numbers] -->
**Time to first token on Apple Silicon, against a cold start and against
mlx-lm's own prompt-cache file:**

| Llama-3.2-1B-4bit, 2,049-token prefix, 16-token question | first token | vs cold |
|---|---:|---:|
| Cold prefill, vanilla mlx-lm | 1,193 ms | |
| mlx-lm's own prompt-cache file, read by a fresh process (`load_prompt_cache`, a 67 MB file) | **37.0 ms** | 32× |
| Pion, the process that stored the prefix, asking again | 46.2 ms | 26× |
| Pion, a separate process, over the wire | 69.0 ms | 17× |

**A file is faster.** `save_prompt_cache` and `load_prompt_cache` ship with
mlx-lm, and mapping a 67 MB file beats fetching the same rows over loopback
TCP. If one program reuses one fixed prefix, use the file. Pion is for what a
file does not do: one server that any process, model object or tool reads
over the Redis wire, with an acked write that survives a crash, and, through
`pion-vllm-mlx serve`, a longest-prefix match over every conversation it has
stored, kept under a byte budget, so a restarted agent or a second session
has no file to find and name.

A shorter prefix saves less: from a separate process the same measurement gives
12× at 1,035 tokens, 4.6× at 268, and 1.4× at 34, where the saving is about
14 ms. A second client (a separate socket and a separate model object) produces
the same text: BLEU 1.000 on a 50-token greedy completion. Sources:
[`cross_process_ttft.py`](benchmarks/reproducers/cross_process_ttft.py) `--same`
for both rows (`--prefix-tokens` for the shorter prefixes) with its
[raw output](benchmarks/reproducers/results/cross_process_ttft_2026_10_07.json),
and `tests/test_kv_prefix_cross_instance.py` with
[its output](benchmarks/results/2026-10-07-mac-m4/kv_prefix_cross_instance.txt),
both on an M4 Mac mini; the file row is
[`file_cache_ttft.py`](benchmarks/reproducers/file_cache_ttft.py) with
[its raw output](benchmarks/reproducers/results/file_cache_ttft_llama_2049_2026_10_07.json),
timed the same way, model load outside the clock. The vanilla side times the first token the way mlx-lm's
own `generate_step` produces it; until 2026-10-02 it also computed logits at
every prompt position, which no generation does, and the ratios published then
were too high. The [changelog](CHANGELOG.md) has the correction. The figures
above are from 2026-10-07, after the harness gained the `<bos>` its prefix lacked;
a same-day run without that fix lands within run-to-run variation of them.

At 64K context a sparse mask still finds the needle. On Gemma-4-E2B-it-4bit,
with a 63,961-token prefix and a 23-token question, vanilla mlx-lm's cold
prefill took 54.4 s to the first token and Pion's warm cache 124.4 ms (437×).
Both answered with the needle's number, while each full-attention layer
attended 512 of the prefix's tokens, 0.80%
([`examples/sparse_mask_64k_niah.py`](examples/sparse_mask_64k_niah.py),
[raw output](benchmarks/results/2026-10-07-mac-m4/sparse_mask_64k_niah.txt), M4
Mac mini, mlx-lm 0.31.3). It is one needle at one depth, warm against cold.
mlx-lm's own prompt-cache file of the same prefix (405 MB) answers in 82.6 ms
([raw output](benchmarks/reproducers/results/file_cache_ttft_gemma_64000_2026_10_07.json)),
faster again, and it attends the whole prefix where Pion's mask attends 0.80%. On
2026-10-06 the example found the needle with neither vanilla nor Pion, because
its prompt carried 397 `<bos>` tokens; it now carries one.
<!-- --8<-- [end:two-numbers] -->

Design, wire protocol, the quantized tiers and the Stage 2 lanes:
[`doc/shared_kv_cache.md`](doc/shared_kv_cache.md). Hybrid models (SSM and
sliding-window layers) are stored by `PionPromptCache` and its `SSM.PREFIX.*`
companion; `pion-vllm-mlx serve` does not store them yet.

### 2. A Redis-compatible KV store with a write-ahead log

354 command names (the generated command table), every one exercised by a
dispatch sweep in five argument shapes, and a differential suite that compares
replies against a real `redis-server`: strings, hashes, lists, sets, sorted
sets, bitmaps, HyperLogLog, geo, streams and consumer groups, pub/sub,
`MULTI`/`EXEC`/`WATCH`, TTLs, Lua. Every write goes to a write-ahead log
before it is acknowledged; an acked write is still there after `SIGKILL`.
`redis-cli` and redis-py, which the test suite uses, talk to it unmodified over
RESP2 or RESP3. Reference:
[`doc/command_matrix.md`](doc/command_matrix.md),
[`doc/persistence.md`](doc/persistence.md).

Against Redis 8.10.2, Pion wins some configurations and loses others. The KV
and vector numbers below are one Linux server, both engines measured the same
day with the same client settings: an AMD EPYC 8124P (16 cores), Redis 8.10.2
built from source, Pion 0.9.5 built as the release builds it. Raw output,
scripts and method:
[`benchmarks/results/2026-10-06-linux-epyc-8124p/`](benchmarks/results/2026-10-06-linux-epyc-8124p/README.md).

**KV throughput** (memtier, 256-byte values, 1:10 SET:GET over 1M keys, 30 s a
run, median of 3, ops/sec):

| One keyspace, ops/sec | Redis, 1 thread | Redis, `io-threads 8` | **Pion `-w 1`** |
|---|---:|---:|---:|
| P=1 | 107,896 | **388,405** | 112,019 |
| P=10 | 643,517 | **1,655,550** | 877,766 |
| P=50 | 987,970 | 1,376,680 | **1,712,725** |

| All 16 cores, 16 independent keyspaces, ops/sec | 16 Redis processes | **Pion `-w 16`** |
|---|---:|---:|
| P=10 | **7,178,453** | 6,097,719 |
| P=50 | **11,128,400** | 9,848,658 |

On one keyspace, Redis with I/O threads is faster than Pion's single worker
except at a deep pipeline (P=50), where Pion leads by 24%; against
single-threaded Redis, Pion leads at every depth. With durability on (P=10),
Redis's AOF (`everysec`, `io-threads 4`) measured 1,213,746 ops/sec and Pion's WAL
847,249, Redis ahead by 43%. Across all cores, sixteen Redis processes
beat Pion's sixteen workers by 13–18%. Using the second hardware thread of every
core — 32 keyspaces each side, 32 client threads, P=50 — Pion leads with 64-byte
values (15,864,842 against 14,780,031) and Redis with 256-byte values (12,372,245
against 11,640,929). The results directory also has Redis at `io-threads 4`. The headline this README used to carry — 14.0M ops/sec, 10.2×
Redis — had no surviving raw log and set Pion's 32 independent keyspaces against
a single Redis instance; it is withdrawn. April's own configuration, re-run on this
server, measured 13,750,187 with 64-byte values — the old number's ballpark, now
with raw output — and the like-for-like comparison is above.

> **Worker-count caveat.** Every number above `w > 1` is measured with
> `--independent-workers`, and that flag changes the semantics: workers are
> shared-nothing, so **`-w N` runs N independent keyspaces**, and a connection is
> bound to whichever worker won `accept()`. A write acked on one connection is
> not readable from another. Pion therefore defaults to `-w 1` (one coherent
> keyspace, what a Redis client expects) and refuses to start with `-w N > 1`
> unless you pass the flag. See [Scaling past one worker](#scaling-past-one-worker).

### 3. Vector search and the semantic cache

HNSW with INT8 SIMD kernels behind the RediSearch `FT.*` API and Redis 8 vector
sets (`VADD`/`VSIM`), BM25 and hybrid fusion (`FT.HYBRID`), and
`AI.SEMANTIC_CACHE`, which embeds on the server (Apple's `NLEmbedding` on macOS,
MiniLM from a source build). Reference: [`doc/vector_engine.md`](doc/vector_engine.md),
[`doc/embeddings.md`](doc/embeddings.md).

**Vector search** (VectorDBBench Performance1536D50K: 50K OpenAI embeddings,
1536 dims; recall@100 against the dataset's ground truth; 1, 5 and 10 clients
for 5 s each; median of 3):

| | Redis 8.10.2 vector sets | **Pion** |
|---|---:|---:|
| **QPS** (10 clients, the peak for both) | **8,039** | 7,339 (−9%) |
| **Recall@100** | 0.920 | **0.960** (+4.0 pp) |
| **P99 latency**, one client | 1.75 ms | **1.4 ms** |
| **Load until searchable** | 51.8 s | **26.4 s** (2.0× faster) |

*Pion serves one index from `-w 16` at ef=150 through VectorDBBench's Redis
client; Redis runs `VADD`/`VSIM` at its defaults through
`benchmarks/VectorDBBench/vset-benchmark.py`, because VectorDBBench has no
vector-set client, so the two clients are different programs sending the same
queries. Pion's load is its insert (≈10.7 s) plus the index build (≈15.5 s);
Redis builds while it inserts.*

At equal or better recall, Pion is the faster of the two. At ef=100 — the
smallest search effort for 100 results, since Pion raises a smaller ef to k —
four runs measured a median **8,544 QPS at recall 0.939**, against Redis's 8,039
at 0.920 (`benchmarks/results/2026-10-06-linux-epyc-8124p/iso/`). Redis was not
run at a higher EF, so the comparison at 0.960 recall is still open.

---

## Quick Start

<!-- --8<-- [start:security] -->
> **Security model, in one paragraph.** Pion listens on **127.0.0.1 by
> default** and there is **no TLS**. Setting a password (`--requirepass-file`,
> `PION_REQUIREPASS`, or `--requirepass`) switches the default to all
> interfaces, following Redis's protected-mode convention — a password is taken
> as the signal that you intend to serve remotely. Override either way with
> `--bind <addr>`; an address that does not parse is a startup failure, never a
> silent fall back to `0.0.0.0`. The bind address applies to **every** listener:
> the RESP port, the binary lane on `port+1`, the WAL replication stream on
> `port+10000`, and gossip/Raft. Note that **replication and gossip are
> unauthenticated** — anyone who can reach `port+10000` can stream the WAL — so
> put those behind a private network, and terminate TLS at a proxy if you need
> encryption in transit. Running with `--bind 0.0.0.0` and no password prints a
> warning at startup and means exactly what it says. See
> [`SECURITY.md`](SECURITY.md). **No
> telemetry:** Pion never phones home — no update check, no analytics — and
> connects out only to what you configure or enable; the full list is in
> [`SECURITY.md`](SECURITY.md#outbound-connections--no-telemetry).
<!-- --8<-- [end:security] -->

### Homebrew (macOS)

macOS 14 or later on Apple Silicon:

```bash
brew install pavelhorak/tap/pion
brew services start pion         # 127.0.0.1:1974; restarts at login and after a crash
redis-cli -p 1974 PING           # +PONG
```

The service starts with `--kvcache --metal-attention --nle-embed`, so the
prompt cache (the four `PionPromptCache` lines) and the semantic cache work
against it as installed. As a service, Pion keeps its WAL, snapshots and crash log in
`$(brew --prefix)/var/pion` and logs to `$(brew --prefix)/var/log/pion.log`.
`pion-server` is also on your PATH; run by hand, it writes its data to the
directory you start it from.

### Prebuilt binary (no build)

macOS 14 or later on Apple Silicon, without Homebrew:

```bash
curl -fsSL https://github.com/pavelhorak/pion/releases/latest/download/pion-macos-arm64.tar.gz | tar xz
cd pion-*-macos-arm64
./pion-server.sh                 # port 1974
redis-cli -p 1974 PING           # +PONG
```

Launch through `pion-server.sh`, not `bin/pion-server`: the wrapper points the
binary at the runtime libraries the tarball bundles. Linux tarballs for x86_64
and arm64 come from the same release (swap the file name for
`pion-linux-x86_64.tar.gz` or `pion-linux-arm64.tar.gz`) and need glibc 2.38 or
newer. Checksums, the Metal shader library and building a tarball yourself:
[`doc/operations.md`](doc/operations.md) §5.

### Docker

Each release publishes a multi-arch image (linux/amd64 and linux/arm64) to
GitHub Container Registry:

```bash
docker run -p 1974:1974 -v pion-data:/data ghcr.io/pavelhorak/pion
redis-cli -p 1974 PING     # +PONG
```

Data (WAL, snapshots, blob arenas) lives in the `/data` volume. The image runs
with `--epoll`, because Docker's default seccomp profile blocks io_uring, and
ships no Python, so features that need an embedding model need a host install.
Building the image yourself, bind mounts and io_uring:
[`doc/operations.md`](doc/operations.md) §5.

### From source

<!-- --8<-- [start:build-from-source] -->
> **Prerequisites:**
> - **[pixi](https://pixi.sh)** builds and runs everything here — `curl -fsSL https://pixi.sh/install.sh | bash`, then restart your shell. It brings its own Mojo toolchain; nothing else needs installing.
> - **`redis-cli`** for the snippets below (`brew install redis` / `apt install redis-tools`). Any Redis client works — Pion speaks RESP2 and RESP3.
> - **macOS:** the Metal shader step (`xcrun metal`) needs full **Xcode** (not just Command Line Tools). Without it the build prints `[Metal] skipped` and reuses the tracked `src/ffi/metal_compute.metallib`, so `--metal-attention` still works — you only need Xcode if you edit `metal_compute.metal`. To rebuild the shader, install Xcode from the App Store, then `xcode-select -s /Applications/Xcode.app/Contents/Developer`.
> - **Linux (GPU):** `pixi run build` calls `/usr/local/cuda/bin/nvcc` and links `cudart`. Install CUDA 12 + cudart first.
> - **Linux (no GPU):** use `pixi run build-portable` — GPU-free build (no nvcc, no cudart, portable x86-64-v2 binary). Disables `--metal-attention` and `--cuda-attention` but all KV / vector / AI commands work. **It produces `./pion-server-dev`, not `./pion-server`** — substitute that name in every command below.

```bash
git clone https://github.com/pavelhorak/pion.git && cd pion
pixi install
pixi run build                        # produces ./pion-server
./pion-server                         # port 1974, one shared keyspace
```
<!-- --8<-- [end:build-from-source] -->

### First five minutes

<!-- --8<-- [start:first-five-minutes] -->
**KV** — `redis-cli` talks to Pion unmodified:

```bash
redis-cli -p 1974 SET user:1 '{"name":"alice"}'
redis-cli -p 1974 GET user:1
```

**Vector search.** A vector is `DIM * 4` raw float32 bytes — 6144 for the
default `DIM 1536` — so this needs a real client rather than `redis-cli`. Three
things the API requires, in this order:

1. `FT.CREATE` **before** any `HSET`. Vectors are only routed into the HNSW once
   the index exists; documents added earlier are ordinary hashes and are not
   searchable.
2. `HSET <key> vec <blob> <field> <value>` — at least two field/value pairs. The
   vector-ingest path is only taken for `HSET` with 6 or more arguments.
3. `DISTANCE_METRIC` accepts **`L2`** and **`COSINE`**. Any other value — `IP`
   included — is refused at `FT.CREATE` rather than silently answered with the
   wrong ordering.
   Omitting the keyword gives `L2`.

`DIM` is per-index in the RediSearch API but the blob must be `DIM * 4` bytes of
float32 either way; the example below uses the server's default 1536.

```bash
pip install redis numpy       # or: pixi run python -m pip install redis numpy
python3 - <<'PY'
import numpy as np, redis
r = redis.Redis(port=1974)
D = 1536                                        # blob is D*4 = 6144 bytes

r.execute_command("FT.CREATE", "idx", "SCHEMA", "vec", "VECTOR", "HNSW", "6",
                  "TYPE", "FLOAT32", "DIM", str(D), "DISTANCE_METRIC", "COSINE")

rng = np.random.default_rng(0)
vecs = {f"doc:{i}": rng.random(D).astype(np.float32) for i in range(20)}
for key, v in vecs.items():
    r.execute_command("HSET", key, "vec", v.tobytes(), "title", key)

r.execute_command("FT.OPTIMIZE", "idx")         # build the graph, then query

hits = r.execute_command("FT.SEARCH", "idx", "*=>[KNN 3 @vec $B]",
                         "PARAMS", "2", "B", vecs["doc:7"].tobytes(),
                         "DIALECT", "2")
print(hits[0], "hits; nearest =", hits[1].decode())   # 3 hits; nearest = doc:7
PY
```

**Semantic cache.** The cache's index is per worker, so it needs a
single-worker server (the default)
and an embedding backend. On macOS use Apple's `NLEmbedding` — no downloads, no
Python (the Homebrew service already runs with `--nle-embed`):

```bash
./pion-server --nle-embed                       # macOS: Apple's sentence embedding, 512-dim
# other platforms: pixi run install-inference   # once; then ./pion-server
#                                               # auto-embeds via MiniLM-L6-v2

redis-cli -p 1974 AI.SEMANTIC_CACHE SET "capital of France?" "Paris"
redis-cli -p 1974 AI.SEMANTIC_CACHE GET "What is the capital of France?"
# -> "Paris"  — a different wording, matched by meaning
```
<!-- --8<-- [end:first-five-minutes] -->

### Server Profiles

> Run `./pion-server --help` for the full flag reference (cluster / AI / inference / tuning groups), grouped and one-line-described.

```bash
./pion-server --profile kv            # KV-only: ~50MB/worker, no HNSW
./pion-server --profile vector        # KV + vector search
./pion-server --profile ai --flare    # KV + vector + AI (semantic cache; RAG and FLARE are experimental)
./pion-server --kvcache -w 1          # + externalized attention (ATTEND.*)
./pion-server --metal-attention -w 1  # + native Metal SDPA on Mac (no Python; fp32, more accurate)
./pion-server --metal-attention-fp16 -w 1  # fp16 kernel — matches vanilla mlx-lm precision
./pion-server --metal-attention --fa-window 2048 -w 1  # sliding-window SDPA: scans only the last N tokens. Safe on Mistral SWA / Longformer; lossy on dense models (Llama, Gemma, GPT).
```

Vector quantization (`--polarquant`, `--turboquant`, `--nanoquant`) is in
[`doc/vector_engine.md`](doc/vector_engine.md); the Linux I/O backends
(`--epoll`, `--iouring`, `--xdp`) are in [`doc/networking.md`](doc/networking.md).

### Scaling past one worker

Pion is **shared-nothing**. A worker owns a private hash map, WAL, HNSW graph and
allocators, and there is no cross-worker request bus. Workers compete for
`accept()` on one listen fd, so a connection is bound to whichever worker won the
race — and:

```
conn A  ->  worker 1   SET user:1 alice   -> +OK
conn B  ->  worker 3   GET user:1         -> (nil)
```

That is not a bug report, it is the model. `-w N` gives you **N independent
keyspaces**, not one keyspace served by N threads. Cross-worker `PUBLISH`
delivers to zero subscribers, `FLUSHALL` clears one worker's slice, and
replication covers worker 0 only.

Because every default connection pool (redis-py, Jedis, go-redis, ioredis) opens
more than one connection, this breaks read-your-writes silently — no error, just
nils. So:

- **The default is `-w 1`.** One worker, one coherent keyspace, every Redis
  client correct out of the box.
- **`-w N > 1` refuses to start** unless you also pass `--independent-workers`,
  which prints the semantics at startup.

Multi-worker is the right choice when each client **pins a single connection**
(or shards keys across per-worker ports itself), and for read-mostly vector
search — the HNSW graph *is* published across workers via `SharedHNSWView`, so
`FT.SEARCH` is coherent at `-w N` even though the KV keyspace is not: every
worker returns the same ranking and the same keys. The documents' hash *fields*
still live on the worker that ingested them, so a result answered elsewhere
carries its key and score but not its `id` field — read the key. That is the
configuration behind the `w=16` / `w=32` benchmark numbers above.

```bash
./pion-server -w 1                              # default: one coherent keyspace
./pion-server -w 16 --independent-workers       # 16 keyspaces, client pins a connection
```

If you want one logical keyspace across cores today, run N single-worker Pions on
N ports and shard client-side, the same way you would shard Redis.

---

## How it compares

### LLM Memory

Every competitor here does something well, and most of them do more than a
table can show. Cells are what each project **ships today** (checked
2026-09-19; the mlx-lm column re-checked against mlx-lm 0.31.3 on 2026-10-07);
the footnotes carry the sources.

| | LMCache (+vLLM) | SGLang HiCache | oMLX | mlx-lm `cache_prompt` | **Pion** |
|---|:---:|:---:|:---:|:---:|:---:|
| Cache shared across **processes** | ✓ via remote store¹ | ✓ L3¹ | ✗ per-process² | manual file | **✓ wire-native** |
| Survives restart | ✓ | ✓ | ✓ SSD tier | ✓ manual | **✓** |
| **Crash-consistent** (acked = durable) | ✗ | ✗ | ✗ | ✗ | **✓ WAL**† |
| KV quantization | FP8; 4-bit demo³ | FP8 | ✓ TurboQuant | ✓ `--kv-bits`⁵ | **fp16/int8/turbo4/INT3/INT2/mlx4g32** |
| Hybrid (Mamba/GDN) prefix state | in-process⁴ | host + storage tiers⁴ | in-process + SSD | ✓ in the file⁵ | **cross-process + persisted** |
| Sparse long-context selector | ✗ | ✗ | ✗ | ✗ | **✓ block-mean, bit-identical decode** |
| MoE expert paging | ✗ | ✗ | ✓ in-process (experimental) | ✗ | cross-process + histograms (experimental) |
| Apple Silicon | ✗ (CUDA) | ✗ (CUDA) | ✓ | ✓ | **✓** (+ Linux CPU) |
| Redis wire / any RESP client | ✗ | ✗ | ✗ | ✗ | **✓** |
| Runs as | vLLM plugin + service | serving engine | menu-bar app / server | library | **one server binary** |

**The two rows that are actually ours** are crash-consistency and the shape of
the sharing. Everything else on this list is a matter of degree.

*Crash-consistent* is not the same as *persistent*. LMCache's disk backend,
SGLang's L3 and oMLX's SSD tier all persist, and all of them recompute on a
miss — that is a perfectly good design. Pion's contract is narrower and
stronger: an acked write is there after a `SIGKILL`.†

*Shared across processes* is not the same as *distributed*. LMCache does
cross-instance sharing through a remote store, and that is its entire pitch;
oMLX clusters by routing a request to the node that already holds the prefix,
which is a different and reasonable answer. Pion is one process on one port
that a **different program, a different model object, or a different machine**
reads directly — no connector, no serving stack, no CUDA.

¹ LMCache ships `local_disk_backend.py`, `p2p_backend.py` and NIXL/remote connectors; SGLang HiCache L3 backs onto Mooncake/3FS/NIXL.
² oMLX per-node caches are private to the process; its cluster scheduler scores nodes on prefix affinity and routes to the hot one. An export/import API was requested by one user on 2026-09-12 ([jundot/omlx#3612](https://github.com/jundot/omlx/issues/3612), converted to a discussion on 2026-10-05 with no votes); the same user's draft implementation ([jundot/omlx#3615](https://github.com/jundot/omlx/pull/3615)) is unmerged, and nothing has shipped.
³ LMCache demoed 4-bit KV with AMD on 2026-08-28.
⁴ vLLM merged hybrid prefix caching 2026-07-12 ([vllm#46384](https://github.com/vllm-project/vllm/pull/46384)), vllm-metal 2026-08-10 ([#584](https://github.com/vllm-project/vllm-metal/pull/584)); both share that state **within** a process. SGLang's HiCache tiers hybrid GDN/Mamba state to host memory and its storage backends as of 2026-09-21 ([sglang#37507](https://github.com/sgl-project/sglang/pull/37507)), inside its own serving stack.

⁵ `mlx_lm.cache_prompt` builds the cache with `make_prompt_cache` and writes it with `save_prompt_cache`, which serializes every cache class's state, the rotating sliding-window and recurrent (`ArraysCache`) layers included; `--kv-bits` quantizes the K/V. [`file_cache_ttft.py`](benchmarks/reproducers/file_cache_ttft.py) `--workload ssm` checks a Mamba model's recurrent state through the file, and `--workload gemma` a sliding-window model's, each against the cold path's first token ([raw output](benchmarks/reproducers/results/)).

† WAL durability is verified on macOS **and Linux**, V-store included: SIGKILL → restart replays the WAL and `V.FETCH` returns bit-equal data (max |Δ| = 0.0), validated on EPYC 8124P. `tests/test_vstore_wal.py` runs in Gate 2c on every gate.

If you run one server and never restart it, vLLM's automatic prefix caching
already does this and you do not need Pion.

### On oMLX specifically

[oMLX](https://github.com/jundot/omlx) is the best way to serve
models locally on a Mac today, and if that is what you want, use it — a
menu-bar app, continuous batching, a RAM+SSD tiered KV cache, TurboQuant KV,
and cache-aware cluster routing across machines.

Pion is not a serving app and does not compete with it. oMLX keeps each
node's cache private to its own process and routes work to whichever node is
warm; Pion is a cache *substrate* that a separate process reads over a wire
protocol, holds hybrid/SSM state across processes rather than within one, and
makes an acked write survive a crash. The two compose, and Pion as a shared or
remote tier underneath oMLX is a thing we would like to build.

One oMLX user asked for durable, portable prefix-cache artifacts so an agent
session would stop re-prefilling 60–120k tokens after a reload
([#3612](https://github.com/jundot/omlx/issues/3612), no votes, since converted
to a discussion). It is one request, not a demand signal. They proposed a
different shape from ours — an agent-owned file with a manifest, exported
through oMLX's own API, rather than a server — so read it as evidence for the
*problem*, not as a vote for Pion. The constraints they arrived at are the
four Pion had to solve: carry the recurrent/GDN state and not only KV, pin the
quantization parameters in the manifest, key on the exact token sequence
rather than the text, and fail closed on any mismatch.

---

## Experimental

These parts of the repository are outside the supported surface above. Each
works as far as its tests go, and none has a published measurement of what it
buys or a user outside this project. They stay in the tree, may change or be
removed, and each carries an **Experimental** banner on its own page.

| Part | Its page | What is tested |
|---|---|---|
| `pion-serve`, the semantic-cache proxy (not `pion-vllm-mlx serve`) | [`doc/pion_serve.md`](doc/pion_serve.md) | its embedding path in the gate tier; router and backends in the full tier |
| MCP server (`mcp/`) | [`mcp/README.md`](mcp/README.md) | its tools, called directly, in the gate tier; no test drives the MCP protocol |
| Cluster: slots, gossip, Raft, replication, failover | [`doc/distributed_systems.md`](doc/distributed_systems.md) | gate and full tier; replication covers worker 0 only and its stream is unauthenticated |
| Tenant binding (`--tenant`) | [`doc/multi_tenant.md`](doc/multi_tenant.md) | gate tier; isolation reviewed only by this project |
| `AI.COMPLETE`, `AI.CHAT`, `AI.ROUTE.*`, `RAG.*`, `AI.FLARE.*` | [`doc/ai_gateway.md`](doc/ai_gateway.md) | wire-tested in the gate tier (needs Ollama) |
| FLARE gateway proxy (`flare_gateway/`) | [`flare_gateway/README.md`](flare_gateway/README.md) | no test of its own |
| `AI.KNN_LM.*`, `NEURON.PKM.*` | [the command reference](https://pion.pavelhorak.com/docs/reference/commands/rag-knn-pkm/) | full tier |
| `MOE.EXPERT.*` expert paging | [the command reference](https://pion.pavelhorak.com/docs/reference/commands/moe-expert/) | wire-tested in the gate tier and the full tier; decode is research-grade |
| `pion-exo` (exo attention hook) | [`pion-exo/README.md`](pion-exo/README.md) | full tier; exo has no upstream hook API |
| `vllm-pion` (vLLM connector client) | [`vllm-pion/README.md`](vllm-pion/README.md) | no test of its own; the connector is not merged upstream |
| LangGraph, AutoGen and LlamaIndex packages | [`pion-langgraph/`](pion-langgraph/README.md), [`pion-autogen/`](pion-autogen/README.md), [`pion-llamaindex/`](pion-llamaindex/README.md) | one gate-tier test of their basic calls |
| LMCache backend (`pion-lmcache/`) | [`pion-lmcache/README.md`](pion-lmcache/README.md) | full tier; the CUDA connector test needs a CUDA box |
| `pion-glide` (GLIDE client) | [`pion_glide/README.md`](pion_glide/README.md) | no test |
| RedisVL and LangChain through their Redis clients | — | no test in the suite |

---

## At a Glance

| | Pion |
|---|---|
| Category | **Memory engine for AI inference** |
| Headline | **Shared KV Cache** — at a 2K prefix on Llama-3.2-1B-4bit, first token in 69.0 ms from a separate process over the wire and 46.2 ms in the same process, against 1,193 ms cold and 37.0 ms from mlx-lm's own prompt-cache file; BLEU 1.000 cross-instance |
| Language | Mojo (SIMD-native, no GC) |
| Protocol | RESP2 / RESP3 (Redis wire-compatible) |
| Peak KV throughput | **15.9M ops/sec** across 32 keyspaces at P=50 with 64-byte values (32 Redis 8.10.2 processes: 14.8M; at 256 bytes Redis leads, 12.4M to 11.6M) · **1.71M** on one keyspace (Redis with I/O threads: 1.38M) — [same server](benchmarks/results/2026-10-06-linux-epyc-8124p/README.md) |
| Vector QPS | 7,339 at 10 clients on Linux at recall 0.960, 8,544 at 0.939 (Redis 8.10.2 vector sets: 8,039 at 0.920); 8,729 on an M4 Mac mini |
| Recall@100 | 0.960 at that QPS (Redis vector sets: 0.920); INT8, ef=150 |
| P99 latency | 1.4 ms on Linux, 0.6 ms on an M4 Mac mini (vector search, one client) |
| Embeddings | Apple NLEmbedding 512-dim with `--nle-embed` on macOS (no Python); MiniLM-L6-v2 384-dim from a source build |
| Networking | kqueue / epoll / io_uring / AF_XDP |
| Persistence | mmap WAL + HNSW snapshot + V-store WAL |
| License | Apache-2.0 · tuned 1536-dim vector kernels a free closed library (`libpion_vector`) — [table](#license) |

---

## More

[Architecture](doc/architecture.md) · [Benchmarking your box](doc/benchmarking_guide.md) ·
[Reproducers](benchmarks/reproducers/README.md) · [Development guide](doc/development_guide.md) ·
[Operations](doc/operations.md) · [Changelog](CHANGELOG.md)

---

## License

| Component | Licence |
|---|---|
| Pion — everything under `src/` (including the open reference vector kernels), the client packages, the tests, tools and docs | [Apache-2.0](LICENSE) |
| `libpion_vector` — the tuned 1536-dim vector kernels under `vendor/pion-vector/`, linked by `pixi run build` and the macOS release tarball | [Pion Vector Binary Licence](vendor/pion-vector/LICENSE) — free, closed |
| Lua 5.1.5 and lua-cjson, vendored under `src/ffi/lua/` | MIT — see [`NOTICE`](NOTICE) |

<!-- --8<-- [start:license-paragraph] -->
Pion is Apache-2.0. The one closed piece, `libpion_vector`, is free for any
use including commercial and in production; it may be redistributed only
unmodified and with Pion, and it may not be used to offer Pion itself as a
managed database, key-value, vector-search or KV-cache service. The open
build (`pixi run build-open`) contains no closed code and carries no such
restriction. The full map, the reasoning, and what is deliberately *not* in
this repository: [`doc/licensing.md`](doc/licensing.md).
<!-- --8<-- [end:license-paragraph] -->
