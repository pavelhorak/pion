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
entirely from source, with vector search at 1536 dims 22–34% slower and
everything else — the KV engine, persistence, the prompt cache — unchanged.
Details: [`doc/licensing.md`](doc/licensing.md).
<!-- --8<-- [end:open-core] -->

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

<!-- --8<-- [start:two-numbers] -->
**Two numbers, both Apple Silicon, both warm cache against a vanilla cold start:**

| Time to first token | vanilla mlx-lm | Pion warm | |
|---|---:|---:|:---:|
| Llama-3.2-1B-4bit, 2,049-token prefix | 1,242 ms | **61.9 ms** | **20×** |
| Gemma-4-E2B-4bit, 64K context, sparse mask | ~58 s | **179 ms** | **326×** |

The 64K row holds **100% needle recall while attending 0.78% of the prefix
budget** — that is needle-in-a-haystack-class retrieval, not a claim about every
long-context task — and it is steady state: the first warm call in a process also
compiles the Metal kernels, about 1.3 s. The first row is same-process; from a separate **process**, over the
wire, the same prefix measures **17×** (1,242 → 73.9 ms). A shorter prefix saves less:
the same measurement gives 11× at 1,035 tokens, 4.5× at 268, and 1.5× at 34,
where the saving is about 15 ms. Cross-instance output is verified
**BLEU 1.000** on a 50-token greedy completion — a separate socket and a separate
model object produce the same text. Sources: [`cross_process_ttft.py`](benchmarks/reproducers/cross_process_ttft.py) `--same` for both 1B rows
(`--prefix-tokens` for the shorter prefixes),
[`examples/sparse_mask_64k_niah.py`](examples/sparse_mask_64k_niah.py).
The vanilla side times the first token the way mlx-lm's own `generate_step`
produces it. Until 2026-10-02 it also computed logits at every prompt position,
which no generation does, and the ratios published then (50.6×, 24×) were too
high — the first also timed a different lane than it named. The
[changelog](CHANGELOG.md) has the correction.
<!-- --8<-- [end:two-numbers] -->

Underneath sits a Redis-wire-compatible KV core and an HNSW vector engine. They
are the on-ramp — every Redis client, LMCache config and RESP tool already speaks
to Pion — and they are fast enough to be interesting on their own
([the engine underneath](#the-engine-underneath)). They are not why you would
install this.

```bash
pixi install && pixi run build
./pion-server --kvcache               # AI memory mode: KV.PREFIX.* + ATTEND.PREFIX.*
redis-cli -p 1974 SET hello world     # the Redis wire still works
```

---

## Why Pion

### LLM Memory

Every competitor here does something well, and most of them do more than a
table can show. Cells are what each project **ships today** (checked
2026-09-19); the footnotes carry the sources.

| | LMCache (+vLLM) | SGLang HiCache | oMLX | mlx-lm `cache_prompt` | **Pion** |
|---|:---:|:---:|:---:|:---:|:---:|
| Cache shared across **processes** | ✓ via remote store¹ | ✓ L3¹ | ✗ per-process² | manual file | **✓ wire-native** |
| Survives restart | ✓ | ✓ | ✓ SSD tier | ✓ manual | **✓** |
| **Crash-consistent** (acked = durable) | ✗ | ✗ | ✗ | ✗ | **✓ WAL**† |
| KV quantization | FP8; 4-bit demo³ | FP8 | ✓ TurboQuant | ✗ | **fp16/int8/turbo4/INT3/INT2/mlx4g32** |
| Hybrid (Mamba/GDN) prefix state | in-process⁴ | host + storage tiers⁴ | in-process + SSD | ✗ | **cross-process + persisted** |
| Sparse long-context selector | ✗ | ✗ | ✗ | ✗ | **✓ block-mean, bit-identical decode** |
| MoE expert paging | ✗ | ✗ | ✓ in-process (experimental) | ✗ | **✓ cross-process + histograms** |
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
² oMLX per-node caches are private to the process; its cluster scheduler scores nodes on prefix affinity and routes to the hot one. An export/import API was requested by a user on 2026-09-12 ([jundot/omlx#3612](https://github.com/jundot/omlx/issues/3612)) and is not shipped.
³ LMCache demoed 4-bit KV with AMD on 2026-08-28.
⁴ vLLM merged hybrid prefix caching 2026-07-12 ([vllm#46384](https://github.com/vllm-project/vllm/pull/46384)), vllm-metal 2026-08-10 ([#584](https://github.com/vllm-project/vllm-metal/pull/584)); both share that state **within** a process. SGLang's HiCache tiers hybrid GDN/Mamba state to host memory and its storage backends as of 2026-09-21 ([sglang#37507](https://github.com/sgl-project/sglang/pull/37507)), inside its own serving stack.

† WAL durability is verified on macOS **and Linux**, V-store included: SIGKILL → restart replays the WAL and `V.FETCH` returns bit-equal data (max |Δ| = 0.0), validated on EPYC 8124P. `tests/test_vstore_wal.py` runs in Gate 2c on every gate.

If you run one server and never restart it, vLLM's automatic prefix caching
already does this and you do not need Pion.

### On oMLX specifically

[oMLX](https://github.com/jundot/omlx) (22K★ in October 2026) is the best way to serve
models locally on a Mac today, and if that is what you want, use it — a
menu-bar app, continuous batching, a RAM+SSD tiered KV cache, TurboQuant KV,
and cache-aware cluster routing across machines.

Pion is not a serving app and does not compete with it. oMLX keeps each
node's cache private to its own process and routes work to whichever node is
warm; Pion is a cache *substrate* that a separate process reads over a wire
protocol, holds hybrid/SSM state across processes rather than within one, and
makes an acked write survive a crash. The two compose, and Pion as a shared or
remote tier underneath oMLX is a thing we would like to build.

That the problem is real is not just our claim: an oMLX user recently asked
for durable, portable prefix-cache artifacts so an agent session would stop
re-prefilling 60–120k tokens after a reload
([#3612](https://github.com/jundot/omlx/issues/3612)). They proposed a
different shape from ours — an agent-owned file with a manifest, exported
through oMLX's own API, rather than a server — so read it as evidence for the
*problem*, not as a vote for Pion. The constraints they arrived at are the
four Pion had to solve: carry the recurrent/GDN state and not only KV, pin the
quantization parameters in the manifest, key on the exact token sequence
rather than the text, and fail closed on any mismatch.

### The engine underneath

A Redis-wire-compatible KV core and an HNSW vector engine carry the substrate.
They are genuinely fast — this is the evidence that Mojo systems code competes
with C at the top end — but they are the on-ramp, not the pitch. Treat the
caveats under each table as part of the number.

**KV throughput.**

| | Redis 8.6 | Dragonfly | **Pion** |
|---|---:|---:|---:|
| **KV ops/sec** (peak — each engine at its best config*) | 1.37M | 4.21M | **14.0M** |
| **KV ops/sec** (P=10, w=16) | 1.49M | 993K | **2.25M** |
| Per-command P=1 | 95–96K | -- | 91–96K — parity, the localhost TCP ceiling |

<sub>* Peak-vs-peak, not a single shared config: Pion's 14.0M is its absolute peak (EPYC 7313P, io_uring, w=32, pipeline 50, 256-byte values); Redis's 1.37M and Dragonfly's 4.21M are each engine's own best in a w=16 run, where Pion measured 12.64M. The P=10 row is that shared w=16 run.</sub>

> **Worker-count caveat.** Every number above `w > 1` is measured with
> `--independent-workers`, and that flag changes the semantics: workers are
> shared-nothing, so **`-w N` runs N independent keyspaces**, and a connection is
> bound to whichever worker won `accept()`. A write acked on one connection is
> not readable from another. Pion therefore defaults to `-w 1` (one coherent
> keyspace, what a Redis client expects) and refuses to start with `-w N > 1`
> unless you pass the flag. See [Scaling past one worker](#scaling-past-one-worker).

> **Pipeline-depth caveat.** Pion's headline wins are at pipelined depth (P≥10) and high concurrency. At **P=1, w=1 the Linux localhost TCP ceiling caps every engine at ~96K ops/sec** — Pion runs at parity with Redis there, not above it. If your workload is single-threaded synchronous request/response, the Redis-baseline column reflects what you'll see.

**Vector search.**

| | Redis VSET | **Pion** |
|---|---:|---:|
| **QPS** (50K, 1536D) | 5,445 | **6,803** (+25%) |
| **Recall@100** | 0.920 | **0.937** (+1.7pp) |
| **P99 latency** | 8.9ms | **1.6ms** (5.6× lower) |
| **Ingest** (50K) | 50.2s | **9.0s** (5.6× faster) |

*Linux bare metal, AMD EPYC 7313P, measured 2026-04-06 with the tuned kernels then in the tree. Linux release tarballs link the tuned library from v0.9.3 on; earlier ones ran the open reference kernels ([licensing](doc/licensing.md)). The current build on an M4 Mac mini measures ~9.4K QPS at recall 0.960 (gate configuration, 2026-09-30). The harnesses to rerun both are in `benchmarks/`; raw run logs are not published.*

### Expert paging — models beyond RAM (substrate validation)

Not a launch claim about speed, and worth reading the italics before the
table: this shows the memory hierarchy works on a consumer machine, not that
decode is interactive today.

| Model | Weights | Device | Result |
|---|---:|---|---|
| Gemma-4-26B-A4B (bf16) | 51.6 GB | 16 GB Mac | **runs** — tiered `MOE.EXPERT.*` cache |
| Phi-3.5-MoE (INT4) | 23.6 GB | 16 GB Mac | runs |
| Mixtral 8x7B (INT4) | 24+ GB | 16 GB Mac | runs |

A cache hit serves an expert in ~**5 ms** whichever tier backs it (RAM LRU → SSD → network). Pruning the 25% least-used experts by access histogram (`HIST`) frees ~10.4 GB on Gemma 4, at a quality cost that depends on the traffic: on prompts unlike the ones the histogram saw, perplexity rose several-fold in our tests, so measure on your own workload before pruning.

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

### Docker

Each release publishes a multi-arch image (linux/amd64 and linux/arm64) to
GitHub Container Registry:

```bash
docker run -p 1974:1974 -v pion-data:/data ghcr.io/pavelhorak/pion
redis-cli -p 1974 PING     # +PONG
```

To build one yourself, hermetically from source or in seconds around a binary
you already have:

```bash
docker build -t pion .                              # from source, ~20 min
docker build --target runtime-prebuilt -t pion .    # wraps ./pion-server, seconds
                                                    # (Linux host only — the image
                                                    #  runs the binary in the context)
```

The published image is built by the release workflow from the tagged source.
The from-source build was last verified end to end by hand on 2026-09-01
(linux/arm64, a 161 MB image): the correctness and Redis-parity suites pass from
outside the container, and the keyspace survives `docker restart` through the
`/data` volume.

Data (WAL, snapshots, blob arenas) lives in the `/data` volume, so it survives
container restarts. Three things worth knowing:

- The default command passes `--epoll`, because Docker's default seccomp profile
  blocks the io_uring syscalls (Docker ≥ 25). For the faster io_uring path, run
  with `--security-opt seccomp=unconfined` and drop `--epoll`.
- The image ships no Python, so it runs with `--no-auto-embed`. Features that
  need an embedding model (semantic cache, auto-embed) need a host install:
  Homebrew, or the tarball with `--nle-embed`, on macOS; a source build elsewhere.
- Use a **named volume** (`-v pion-data:/data`) as shown. The server runs as the
  unprivileged `pion` user, and a named volume inherits `/data`'s ownership from
  the image; a bind mount (`-v $(pwd)/data:/data`) keeps the host directory's
  owner instead, so the WAL cannot be created. For a bind mount, `chown` the
  host directory to the image's `pion` uid first.

### Prebuilt binary (no build)

macOS 14 or later on Apple Silicon, without Homebrew:

```bash
curl -fsSL https://github.com/pavelhorak/pion/releases/latest/download/pion-macos-arm64.tar.gz | tar xz
cd pion-*-macos-arm64
./pion-server.sh                 # port 1974
redis-cli -p 1974 PING           # +PONG
```

Roughly 2 MB: the binary, the three Mojo runtime dylibs it actually links
against, and the Metal shader library that `--metal-attention` needs. No
toolchain, no Python, no model download. Verify the download against the
release's `SHA256SUMS` if you care to (`shasum -a 256 -c SHA256SUMS --ignore-missing`;
without the flag it reports every tarball you did not download as FAILED).

Launch through `pion-server.sh`, not `bin/pion-server` directly — the binary's
rpath points at the build machine's toolchain, and the wrapper is what points it
at the bundled `lib/`. WAL, snapshots and blob arenas are written to the working
directory you launch from.

**Linux** tarballs come from the same release, for x86_64 and arm64: swap the
file name in the `curl` above for `pion-linux-x86_64.tar.gz` or
`pion-linux-arm64.tar.gz`. The x86_64 build is the portable one (x86-64-v2, no
GPU code). From v0.9.3 the Linux tarballs link the same tuned vector library
as the macOS one; earlier ones ran the open reference kernels. They need glibc
2.38 or newer: Ubuntu 24.04, Debian 13, Fedora 39 or later. On an older system
the binary stops with `GLIBC_2.38 not found`; use the Docker image above
instead.

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
./pion-server --profile ai --flare    # KV + vector + AI (semantic cache, RAG, FLARE)
./pion-server --kvcache -w 1          # + externalized attention (ATTEND.*)
./pion-server --metal-attention -w 1  # + native Metal SDPA on Mac (no Python; fp32, more accurate)
./pion-server --metal-attention-fp16 -w 1  # fp16 kernel — matches vanilla mlx-lm precision
./pion-server --metal-attention --fa-window 2048 -w 1  # sliding-window SDPA: scans only the last N tokens. Safe on Mistral SWA / Longformer; lossy on dense models (Llama, Gemma, GPT).
```

### Quantization

```bash
./pion-server --polarquant            # block-INT4: recall 0.965, 6.6K QPS (INT8: 0.960, 8.0K)
./pion-server --turboquant            # block-INT3+QJL: recall 0.953, 4.9K QPS
./pion-server --nanoquant             # block-INT2: EXPERIMENTAL, recall 0.46
```

Gate dataset (Performance1536D50K, ef=150, `-w 10`, Mac). Each variant also keeps an FP32 re-rank copy, so none of them lowers total memory today — they shrink the search beam's working set. Details: [`doc/vector_engine.md`](doc/vector_engine.md).

### I/O Backend (Linux)

```bash
./pion-server --epoll                 # best for P=1 benchmarks
./pion-server --iouring -w 16 --independent-workers   # highest throughput — read the
                                      # scaling note below before using -w > 1
./pion-server --xdp --xdp-iface eth0  # AF_XDP kernel bypass: io_uring's P=1 throughput, p99 −29%
```

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

## Feature Set

### Shared KV Cache — `KV.PREFIX.*` + `ATTEND.PREFIX.*`
The prompt's K/V tensors live in Pion. Future requests on the same prefix skip prefill — backend returns the first token in milliseconds instead of seconds.

```bash
./pion-server --kvcache --metal-attention -w 1   # AI memory mode (Apple Silicon, native Metal end-to-end)
```

```python
from pion_vllm_mlx import PionPromptCache
pc = PionPromptCache(model, host="127.0.0.1", port=1974)
cache = pc.get_or_prefill(prefix_ids, namespace=ns)   # what make_prompt_cache(model) returns, prefix already in it
text = generate(model, tok, prompt=suffix_ids, prompt_cache=cache)
```

- **Stage 1** — the four lines above (`KV.PREFIX.LOOKUP` + `V.FETCH ... RANGE` / `V.FETCH ... BATCH`): cross-instance, persistent, BLEU 1.0, and mlx-lm decodes at native speed because the cache it gets back is an ordinary MLX prompt cache.
- **Stage 2** — `PionPromptCache(model, stage2=True)` + `install_pion_attention_patch()` + `make_pion_prompt_cache(...)`: Pion computes the attention over the prefix itself. Three lanes, picked automatically:
  - **In-process fast lane (default same-process consumer).** Cold prefill stashes per-layer prefix K/V as MLX arrays; warm forwards run `mx.fast.scaled_dot_product_attention` over `concat([prefix | suffix])` with **zero wire roundtrips** and no `mx.eval` barrier per layer. **`bench_w1_stage2.py` Llama-3.2-1B-4bit 5×20: TTFT p50 = 35.2 ms, mean = 43.5 ms, 4.8× vs vanilla cold** on that harness's ~316-token system-prompt prefixes; at a 2,049-token prefix the same lane answers a 16-token question in 61.9 ms against 1,242 ms (20×), the same-process row above.
  - **Binary fast lane on `port+1`** (`0xCA5E` + `CMD_ATTEND_PREFIX_QUERY_FUSED`). Cross-process consumers: single sendmsg scatter-gather, no `.tobytes()` allocs. **TTFT p50 = 127 ms** on the same workload: 32 wire calls a request at 0.79 ms each, because mlx-lm runs the suffix in two passes (all but its last token, then the last) and each pass queries every layer.
  - **RESP fallback** for older servers without the binary listener. **TTFT p50 = 131 ms.**
  Lanes 2/3: native Metal SDPA via `--metal-attention` (no Python sidecar; **0.441 ms** end-to-end at H=8/N=2048/d=128, against 0.489 ms for MLX's own SDPA). Both M=1 decoder and M>1 batched-Q paths; fp16 variant for vanilla mlx-lm precision parity. Same tokens as vanilla mlx-lm on `tests/test_mlx_lm_patch.py` (20/20). Optional `--fa-window N` clamps SDPA to the last `N` tokens for sliding-window models (Mistral SWA, hybrid Qwen3.5 with per-layer routing) — server-wide, lossy on plain dense transformers.
  The in-process lane is the same-process 61.9 ms row. On the wire lanes every decode step pays one round trip per layer, so a consumer that only generates text from another process is faster end to end on Stage 1; Stage 2 is for consumers that want the attention itself to run in Pion — the sparse selectors, `pion-exo`, a custom CacheEngine.
- **WAL-durable** (macOS and Linux): SIGKILL recovers bit-equal, V-store included — replay after SIGKILL is validated on EPYC 8124P and guarded by `tests/test_vstore_wal.py` in Gate 2c. `KV.PREFIX.SAVE` compacts.
- **Authentication**: `--requirepass <password>` requires `AUTH <password>` on every connection before any command is served — on both the RESP port and the binary `port+1` fast lane. Prefer `--requirepass-file <path>` or the `PION_REQUIREPASS` environment variable: the flag spelling puts the password in the process command line, where `ps` and `/proc/<pid>/cmdline` expose it to every local user. See [`SECURITY.md`](SECURITY.md).
- **Tenant isolation**: `--tenant NAME=PASSWORD` (repeatable, requires `--requirepass` as the admin credential) binds each authenticated connection to its tenant's namespace — every key is transparently prefixed, commands outside a fail-closed allowlist are rejected with `-NOPERM`, and `KEYS`/`SCAN` are filtered to the tenant's namespace (see [`doc/multi_tenant.md`](doc/multi_tenant.md)). For hard *resource* isolation (memory/CPU/WAL), run one `pion-server` per tenant. The older `--ns-prefix` flag remains a cooperative namespace guard for `KV.PREFIX.*`/`V.*` only — not an isolation boundary.
- **A mixed workload**: mean TTFT over every request, each prompt's first and cold one included, on Llama-3.2-1B-Instruct-4bit. `tests/test_kv_prefix_workload.py` (5 prompts × 10 queries) measures Stage 1 at **3.1×** for ~316-token prefixes and **5.4×** at ~2,514 tokens; `tests/bench_w1_stage2.py` (5 × 20) measures Stage 2's in-process lane at **4.8×** and **13.2×**; 100% first-token agreement throughout (2026-10-02, M4 Mac mini). It times differently from `cross_process_ttft.py`, the source of the 17× figure, so read the two side by side rather than as one curve.
- **`V.FETCH BATCH`**: a multi-layer fetch that returns all active layers in one round-trip — replaces 32 sequential per-layer calls (16 layers × K + V). Stage 1 warm-path TTFT measured 2.9× faster at 64-token prefix, 1.5× at 2K.

Full design + numbers: [`doc/shared_kv_cache.md`](doc/shared_kv_cache.md).

### Hybrid Retrieval — RAG K/V hydration
RAG pipelines retrieve text chunks from an embedding index, then re-encode the same chunk text through the LLM to fill its K/V cache. `HybridRetrievalCache` caches the chunk's prefilled K/V keyed by chunk_id so the LLM skips that prefill on every subsequent retrieval — same chunks, no re-encoding.

```python
from pion_vllm_mlx import HybridRetrievalCache
from mlx_lm import load

model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
hr = HybridRetrievalCache(model)             # default: in-process, bit-perfect

# one-time per chunk (typically at index time, not in the hot path)
hr.ingest("eiffel_passage", tok.encode("The Eiffel Tower is..."))

# hot path — your existing embedder returned `chunk_id`; we hydrate K/V
cache, suffix = hr.prepare("eiffel_passage", tok.encode("How tall?\nAnswer:"))
# pass `cache` directly to mlx-lm generate — chunk K/V already loaded
```

- **Two backends.** `inproc` (default) keeps K/V as MLX arrays in a process-local dict — bit-perfect vs combined-encoding text-RAG, no Pion server needed. `pion` ships K/V via `KV.PREFIX.*` + `V.STOREBATCH/V.FETCH BATCH` for cross-process / cross-host — fp16 wire precision, requires `--kvcache --metal-attention -w 1`.
- **Measured (Llama-3.2-1B-Instruct-4bit, 3 cases):** inproc 100% token agreement, **TTFT savings 37–73%** (mean 55%); pion functional answer-match parity, TTFT savings 29–61% (mean 46%). Storage: ~8 KB/token at INT4 (1B), ~32 KB (7B), ~80 KB (70B). On a 100-query SQuAD v2 run ([`stage1_hybrid_recall_bench.py`](benchmarks/reproducers/stage1_hybrid_recall_bench.py)) the inproc backend measured 3.0× p50 TTFT with 98.3% token agreement (the `pion` backend 2.7×), the cache hydration inside the clock.
- **No new opcodes** — the wire backend layers entirely on the existing `PionPromptCache` substrate. Chunk_id namespaces are hashed (`chunk_<sha256[:24]>`) so they don't collide with prompt-prefix caches.
- **Several chunks per query.** `set_shared_stub()` + `ingest_pack()` + `prepare_multi()` compose up to 8 chunk packs, each re-rotated exactly to its position ([`doc/shared_kv_cache.md`](doc/shared_kv_cache.md) §Multi-chunk). Keep the packs coarse: with many small ones, distractor chunks collide and quality drops well before 20.

Test: [`pion-vllm-mlx/tests/test_hybrid_retrieval.py`](pion-vllm-mlx/tests/test_hybrid_retrieval.py). Spike harness: [`benchmarks/reproducers/stage0_hybrid_kv_injection.py`](benchmarks/reproducers/stage0_hybrid_kv_injection.py).

### MoE Expert Paging — `MOE.EXPERT.*` (models beyond RAM)
Expert weights for MoE models live in a tiered cache (per-worker RAM LRU → SSD → network) and are served over the wire, so a model larger than physical memory can run; a cache hit serves an expert in ~5 ms.

```bash
./pion-server --moe-cache /path/to/experts --moe-cache-mib 8192 -w 1
redis-cli -p 1974 MOE.EXPERT.LOAD <model> <path>      # idempotent by basename
redis-cli -p 1974 MOE.EXPERT.FETCH <model> <layer> <expert> [NS <traffic-class>]
redis-cli -p 1974 MOE.EXPERT.HIST <model> [NS <name>|NSLIST]  # access histograms
redis-cli -p 1974 MOE.EXPERT.PRUNE <model> <keep-set>          # HIST-guided
```

- **Measured**: Gemma-4-26B-A4B (51.6 GB bf16), Phi-3.5-MoE (23.6 GB INT4), and Mixtral 8x7B (INT4) all run on a 16 GB Mac; ~5 ms cache-hit serving regardless of backing tier.
- **HIST-guided pruning**: per-distribution access histograms (≤8 namespaces per model) drive the prune set; a 25% prune frees ~10.4 GB on Gemma 4. Quality depends on the traffic — collect histograms on traffic like yours and check perplexity on it before pruning.
- **Multi-model**: 4 concurrent models per worker over a shared LRU; stacked bf16, per-expert INT4 and stacked INT4 expert layouts.

### Auto-Embeddings
Pion can embed text itself, so the semantic cache and text search need no Ollama, no OpenAI key and no glue code. Redis, Valkey and the dedicated vector stores expect you to bring vectors or run a separate embedding service; this is a convenience difference, not a capability one, and it matters most on a laptop where the extra service is the whole friction.

```bash
redis-cli -p 1974 AI.SEMANTIC_CACHE SET "capital of France?" "Paris"
redis-cli -p 1974 AI.SEMANTIC_CACHE GET "What is the capital of France?"  # hit
```

Two paths, both zero-Ollama:

- **macOS (genuinely zero-dependency):** pass `--nle-embed` for Apple's `NLEmbedding` (512-dim, no model download, no Python); the Homebrew service runs with it. This is the recommended macOS path. Covers `AI.EMBED`, `AI.SEMANTIC_CACHE`, `AI.MEMORY`, `FT.SEARCHTEXT`, `AI.FLARE.*`.
- **All platforms, from a source build (PyTorch sidecar):** omit `--nle-embed`; Pion auto-spawns `src/inference/worker.py` with MiniLM-L6-v2 (384-dim, ~90 MB download on first run). Requires `torch` + `transformers` — install with `pixi run install-inference`. Falls through to Ollama (768-dim) if available; disable entirely with `--no-auto-embed`.

### Pion Serve — Inference Proxy with L1 Cache
OpenAI-compatible proxy on `:8321`. Drops in front of any backend (Ollama / vLLM / OpenAI / Gemini / Claude / llama.cpp), adds a semantic cache layer that catches **28-33%** of Q&A traffic.

```bash
python pion-serve/serve.py --backend ollama --model gemma3:4b
python pion-serve/serve.py --backend ollama --model gemma3:4b --distill   # +L3 concept store (FAQ workloads)
python pion-serve/serve.py --backend ollama --rag-index my_docs           # +RAG injection
```

`/v1/stats` reports `cost_savings_pct` end-to-end. Full guide: [`doc/pion_serve.md`](doc/pion_serve.md).

### KV (Redis-wire-compatible — the foundation)
292 commands, every one exercised by a dispatch sweep in five argument shapes, and a differential suite that compares replies against a real `redis-server`. Strings, Hashes, Lists, Sets, Sorted Sets, Bitmaps, HyperLogLog, Geo, Streams (XADD / XREAD BLOCK), Pub/Sub, MULTI/EXEC + WATCH, TTL / EXPIRE, Lua (EVAL / FCALL), WAL persistence (SAVE / BGSAVE). 28-command zero-alloc fast path. SSO 23-byte inline strings, SIMD Swiss Table probing.

### Vector Search
HNSW with INT8 SIMD kernels, batch-8 prefix pruning, suffix early-exit. Redis 8 vector sets (VADD / VSIM / …), one set per key and searched exactly (details and limits in the [vector command reference](website/pages/reference/commands/vector.md)) — + full FT.* protocol. Hybrid BM25 + vector fusion (FT.HYBRID). Optional Metal GPU brute-force search on macOS (`--gpu`). Quantization: `--polarquant` (INT4), `--turboquant` (INT3+QJL), `--nanoquant` (INT2, experimental) — see [Quantization](#quantization).

### AI Gateway (wire-native)

`AI.SEMANTIC_CACHE` is the shipped, benchmarked one. The rest are real, wired and
tested, but they are not launch claims — measured on narrower workloads than the
numbers beside them suggest, so read each caveat as part of the entry.

- **AI.SEMANTIC_CACHE** — cosine-similarity cache with auto-embedding.
- **AI.COMPLETE** — cache + LLM in one command.
- **AI.CHAT** — full RAG pipeline (retrieve + generate) in one command.
- **AI.ROUTE** — semantic router, 0.14 ms per route (accuracy so far measured only on an 8-query test: 7 of 8).
- **RAG.SPECULATE** — predict next query via embedding momentum, 75% hit rate on linear-trajectory query streams.
- **AI.KNN_LM.\*** — token-id-tagged kNN datastore substrate for client-side kNN-LM augmentation (CREATE / STORE / STOREBATCH / QUERY / INFO / DROP). Up to 16 named datastores per worker. Brute-force scan below 5K entries; **per-vector SQ8 + asymmetric INT8 SIMD HNSW** above (M=32, heap-based PQ + per-vec batch-4 distance kernel + lazy reciprocal pruning). **dim=768, n=30K, k=10: 0.90 recall, 1.94 ms median latency, 36 s bulk-build.** Use cases: code completion, log generation, in-domain text infill.

### MCP Server — 38 Tools for AI Agents
```bash
claude mcp add pion -- uvx --from ./mcp pion-mcp
```
`agent_remember/recall/forget`, `vector_search`, `kv_*`, `hash_*`, `semantic_cache_*`, `cluster_info`. Pion as agent long-term memory.

### Cluster

> **Not launch-ready, and the gaps are the kind you must know before trying it.**
> Replication runs from worker 0 only, a replica syncs by full resync rather than
> incrementally, and **the replication stream on `port+10000` is unauthenticated**
> — anyone who can reach it can read the WAL. Run it on a private network, treat
> it as a preview, and see [`doc/distributed_systems.md`](doc/distributed_systems.md).

SWIM gossip + Raft metadata consensus + WAL replication + auto-failover (463ms) + CLUSTER commands + CRC16 slot routing + MOVED/ASK redirects + slot migration + replica-reads. Valkey GLIDE / redis-py cluster / Lettuce / Jedis compatible.

---

## Ecosystem

### Python Integrations

```python
# RedisVL -- works unmodified
from redisvl.index import SearchIndex
index = SearchIndex(schema, redis_url="redis://localhost:1974")

# LangChain -- works unmodified
from langchain_community.vectorstores.redis import Redis
vs = Redis.from_texts(texts, embedding, redis_url="redis://localhost:1974")

# LangGraph -- agent state persistence
from pion_langgraph import PionSaver      # pip install -e pion-langgraph/

# AutoGen -- semantic agent memory
from pion_autogen import PionMemoryStore  # pip install -e pion-autogen/

# LlamaIndex -- vector store for RAG
from pion_llamaindex import PionVectorStore  # pip install -e pion-llamaindex/

# LMCache -- wire-compatible, zero code changes
# config: remote_url: "resp://localhost:1974"
```

### Mac Cluster Inference

> Pion's side of both hooks is finished and tested; the other side is not.
> **exo exposes no attention-hook API upstream**,
> and the **vllm-pion v1 connector branch is unmerged** — so these are integration
> substrate, not something you can drop into a running exo cluster today.

```python
# exo integration -- distributed inference with V offloading
# gpu_attention routes through PionPromptCache(stage2=True),
# which uses ATTEND.PREFIX.STORE/QUERY against Pion's native Metal SDPA.
# Start Pion with: ./pion-server --kvcache --metal-attention -w 1
from pion_exo import PionAttentionHook    # pip install -e pion-exo/
hook = PionAttentionHook(pion_host="192.168.1.100", mode="gpu_attention")

# mlx-lm prompt cache; with stage2=True Pion computes the attention (pion-vllm-mlx/README.md)
from pion_vllm_mlx import PionPromptCache           # pip install 'pion-vllm-mlx[mlx]'
```

---

## Benchmarking

```bash
# KV (redis-benchmark, P=10)
python3 benchmarks/valkey-benchmark/valkey-benchmark.py -c 50 -n 100000 -P 10 -w 1 --pion-only

# KV (memtier_benchmark, mixed workload)
python3 benchmarks/memtier-benchmark/memtier-benchmark.py --pion-only --profiles throughput,pipeline -w 16
# (the harness passes --independent-workers for -w > 1 — see "Scaling past one worker")

# Vector (VectorDBBench, 50K 1536D OpenAI)
python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 10

# Correctness + protocol parity
python3 tests/test_raw.py && python3 tests/test_parity.py
```

---

## Architecture

```
       │ RESP2/3 wire (Redis-compatible)                   │ port+1: 0xCA5E binary frame
       ▼                                                    ▼ (cross-process KV.PREFIX
                                                              + ATTEND.PREFIX fast lane)
┌──────────────────────────────────────────────────────────────────────────┐
│  NetworkEngine — one OS thread per worker, kqueue / epoll / io_uring / XDP│
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  TCP recv → client_buffers[fd] (partial frame accumulation)         │ │
│  │              ↓                                                       │ │
│  │   FastPathHandler — 28 zero-heap-alloc commands                     │ │
│  │     GET / SET / MGET / MSET / INCR / DECR / HSET / HGET             │ │
│  │     LPUSH / RPUSH / LPOP / RPOP / LRANGE / LLEN / DEL / EXISTS      │ │
│  │     SADD / SPOP / ZADD / ZPOPMIN / PING / FCALL / FUNCTION LOAD    │ │
│  │     GETBIT / SETBIT / BITCOUNT / PFADD / PFCOUNT                    │ │
│  │              ↓ (fallback when not matched)                          │ │
│  │   SlowPathHandler — RESP3 parse, AI gateway, vector, cluster,       │ │
│  │     Lua 5.1, MULTI/EXEC/WATCH, FT.* + VSET, ATTEND.PREFIX.*,        │ │
│  │     KV.PREFIX.*, V.STOREBATCH/V.FETCH, MOE.EXPERT.*                 │ │
│  │              ↓                                                       │ │
│  │   ResponseWriter (4 MB buffer; -ERR-frame on overflow)               │ │
│  │              ↓                                                       │ │
│  │   send() / writev() / io_uring SQE                                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  Per-worker private state (shared-nothing):                              │
│     SlabHashMap (10M slots) · HNSWGraph · WAL · ObjectPools              │
│     SemanticCache · MoEExpertTier · KvCacheStore · VStoreIndex           │
│                                                                           │
│  Shared (across workers): listen fd · SharedHNSWView · MOE.EXPERT LRU    │
│     V-store cross-worker directory (KV.PREFIX.* visibility)             │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓ writev / mmap.msync(MS_ASYNC)
                          ┌────────────────────────────────┐
                          │   pion.wal.{N}    (256 MB segs) │
                          │   pion.hnsw.{N}   (snapshot)    │
                          │   pion.vstore.{N} (snapshot)    │
                          └────────────────────────────────┘
```

- **Shared-nothing workers**: N OS threads, each owns private hash map, HNSW, WAL, allocators — so `-w N` is N independent keyspaces and is gated behind `--independent-workers`; the default is `-w 1`. See [Scaling past one worker](#scaling-past-one-worker).
- **Zero-alloc hot path**: SSO 23B strings, SIMD Swiss Table, `format_int_to_buf` (no division), 4 MB response buffer per worker; a reply that outgrows it continues in the connection's output queue, so replies of any size arrive whole (#49).
- **GenericValue**: 32-byte tagged union (STRING_SSO / STRING / INT / FLOAT / HASH / LIST / SET / ZSET / BITMAP / HLL / GEO).
- **SlabHashMap**: Swiss Table with `h2` fingerprint SIMD probing, Wyhash, 70 % fill rehash.
- **HNSWGraph**: INT8 batch-8 prefix pruning, suffix early-exit, `l0_compact`, 24-cache-line prefetch.
- **Native Metal SDPA** (`--metal-attention` on macOS): hand-written MSL kernels for M=1 decoder, M>1 prefill, fused, sparse, fp32 + fp16 variants — the same tokens as vanilla mlx-lm on `tests/test_mlx_lm_patch.py` (20/20).

---

## Repository Layout

```
src/main.mojo                     # entry: pion_spawn_workers() -> N pthreads
src/engine/state.mojo             # Pion -- state container / DI root
src/network/fast_path.mojo        # 28 zero-alloc commands
src/network/slow_path.mojo        # RESP3 fallback + AI/cluster/vector
src/commands/                     # command handlers by family
src/common/value.mojo             # GenericValue 32B tagged union
src/common/hash_map.mojo          # Swiss Table
src/vector/hnsw.mojo              # HNSWGraph
src/vector/kernels.mojo           # SIMD distance kernels
src/io/wal.mojo                   # mmap WAL, rotating 256 MB segments
src/ffi/uring_wrap.c              # C shims: gossip, raft, replication, io_uring, xdp

pion-serve/                       # inference proxy (semantic cache + RAG)
pion-exo/                         # exo attention hook (Mac cluster)
pion-vllm-mlx/                    # vllm-mlx attention backend
pion-{langgraph,autogen,llamaindex}/  # framework integrations
vllm-pion/                        # Python attention client
mcp/pion_mcp/                     # MCP server (38 tools)
flare_gateway/                    # FLARE OpenAI-compat proxy
benchmarks/                       # KV + vector benchmark harnesses
tests/                            # correctness + parity + AI gateway
doc/                              # technical reference
```

---

## At a Glance

| | Pion |
|---|---|
| Category | **Memory engine for AI inference** |
| Headline | **Shared KV Cache** — 1,242 ms → 61.9 ms TTFT (20×) at a 2K prefix on Llama-3.2-1B-4bit in the same process, 73.9 ms (17×) from a separate one; BLEU 1.000 cross-instance |
| Expert paging | **MOE.EXPERT.\*** — 51.6 GB MoE runs on a 16 GB Mac; ~5 ms cache-hit serving. *Substrate validation; decode is research-grade* |
| Language | Mojo (SIMD-native, no GC) |
| Protocol | RESP2 / RESP3 (Redis wire-compatible) |
| Peak KV throughput | **14.0M ops/sec** (Linux w=32) |
| Vector QPS | ~9.4K on an M4 Mac mini (6,803 on Linux EPYC, April 2026) |
| Recall@100 | 0.960 at that QPS (INT8, ef=150) |
| P99 latency | 0.7 ms (vector search, M4 Mac mini) |
| Embeddings | Apple NLEmbedding 512-dim with `--nle-embed` on macOS (no Python); MiniLM-L6-v2 384-dim from a source build |
| Inference proxy | Pion Serve — 28-33% L1 hit rate on Q&A, any backend |
| Networking | kqueue / epoll / io_uring / AF_XDP |
| Persistence | mmap WAL + HNSW snapshot + V-store WAL |
| Cluster | SWIM gossip, Raft consensus, WAL replication, auto-failover (463 ms). *Not launch-ready: replication is worker-0-only and full-resync, and the replication port is unauthenticated* |
| AI features | semantic cache + externalized attention (shipped); RAG, FLARE, speculative RAG, `AI.CHAT`/`AI.COMPLETE` are in the box but not launch claims |
| Mac cluster | exo / vllm-mlx hooks, MLX GPU attention sidecar. *Pion side done; the exo hook has no upstream API yet and the vllm-pion v1 connector is unmerged* |
| Integrations | LangGraph, AutoGen, LlamaIndex, RedisVL, LangChain, LMCache |
| MCP | 38 tools for AI agents |
| License | Apache-2.0 · tuned 1536-dim vector kernels a free closed library (`libpion_vector`) — [table](#license) |

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
