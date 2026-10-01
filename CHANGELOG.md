# Changelog

Notable changes. Format loosely follows [Keep a Changelog]; versions before
0.9.0 were an internal `0.BUILD+SHA` counter and are summarised rather than
enumerated — there were roughly 1,100 of them.

## [0.9.1] — 2026-10-01

Packaging and documentation only; the server's behaviour is unchanged.

- **`pion-vllm-mlx` 0.1.1.** The 0.1.0 page on PyPI still said the package was
  not on PyPI, told you to install it from a checkout, and its links resolved
  only inside the repository. Fixed.
- **Homebrew's service runs the prompt cache.** `brew services start pion` now
  starts the server with `--kvcache --metal-attention --nle-embed`, so the four
  `PionPromptCache` lines and the semantic cache work against it as installed;
  before, they failed with `ERR V-store not enabled`. Idle memory is about
  113 MB.
- **The README and the website match their sources again.** Vector numbers carry
  their date and machine (the current build: ~9.4K QPS at recall 0.960 on an M4
  Mac mini); at P=1 Pion is at parity with Redis, not "ties or beats" it; the
  64K row shows its denominators; multi-chunk hybrid retrieval is documented as
  shipped; MoE pruning states its quality cost. Claims with no measurement
  behind them are gone: ANE acceleration for NLEmbedding, a 60% cache hit rate,
  a 60× expert-fetch factor, a P=1 P99 figure, and a Docker image variant that
  does not exist.

## [0.9.0] — 2026-10-01

**The first public release.** Pion is a Redis-wire-compatible memory engine
for AI inference: it keeps your model's KV cache across processes, restarts and
machines.

This is labelled **public preview**, not 1.0, and the label is meant literally.
The engine has been under continuous gate discipline since February and the
correctness surface is well tested, but the operational surface has gaps that a
1.0 should not have. They are listed below rather than discovered.

### Headline

- **Shared KV cache over a wire protocol.** `KV.PREFIX.*`, `ATTEND.*`,
  `SSM.PREFIX.*`, V-store, and a binary lane on `port+1`. A different process,
  a different model object or a different machine reads the same prefix.
  1,530 ms → 30.2 ms warm TTFT (Llama-3.2-1B-4bit, 2,048-token prefix, same
  process); 24× cross-process on the wire (1,558 ms → 64.7 ms); cross-instance
  output verified BLEU 1.000.
- **`PionPromptCache`** — a 4-line drop-in for `mlx_lm.make_prompt_cache` on
  Apple Silicon.
- **A value receipt.** `PION.STATS` (and a `# Pion` section in `INFO`) reports
  cumulative prefill seconds avoided, measured from the client's own reported
  cold prefill (`KV.PREFIX.REGISTER ... PREFILL_MS`) and kept apart from the
  per-token estimate used when nothing was reported. `INFO` no longer
  hardcodes its port, memory or an empty keyspace.
- **Crash-consistent, not merely persistent.** Every store is a WAL record;
  SIGKILL → restart → bit-equal `V.FETCH` (max |Δ| = 0.0), verified on macOS
  and Linux.
- **Hybrid Mamba+Transformer prefix state** shared across processes and
  persisted: 24.7× / 32.7× / 11.6× warm TTFT at 2K / 4K / 8K on Qwen3.5-4B.
- **Sparse long-context selector** — 100% needle recall at 64K attending 0.78%
  of the prefix, bit-identical greedy decode. NIAH-class retrieval only.
- **MoE expert paging** (`MOE.EXPERT.*`) — a 51.6 GB model on a 16 GB Mac, with
  a cross-process API and access histograms.
- **Semantic cache, vector search, BM25, hybrid retrieval**, and built-in text
  embedding on macOS (`--nle-embed`), all behind the same RESP wire.
- **The engine underneath**: 2.2M+ ops/sec on a single Mac worker at P=10,
  14.0M peak on EPYC (w=32, P=50), parity with Redis at P=1. This is the
  on-ramp, not the pitch.

### Licensing

Pion is Apache-2.0: the engine, the tools, the docs and every client
package. The tuned 1536-dim vector kernels ship as `libpion_vector`, a free
closed binary library under its own licence (`vendor/pion-vector/LICENSE`);
`pixi run build-open` builds without it, from source only, with slower vector
search at 1536 dims. Every client package carries its Apache-2.0 text
inside: `pion-vllm-mlx`, `pion-mcp`, `pion-lmcache`, `pion-context`,
`pion-glide`, `pion-langgraph`, `pion-autogen`, `pion-llamaindex`,
`pion-exo`. The full map, including what is deliberately not in the
repository, is `doc/licensing.md`.

### Not in this release

Stated because a preview that hides its gaps is just a 1.0 with better
marketing:

- **No TLS.** Bind is loopback by default; terminate TLS at a proxy.
- **No eviction.** `--maxmemory` refuses memory-growing writes with `-OOM` at an
  RSS ceiling; nothing is evicted, and without it RAM exhaustion is an OOM kill.
- **`-w N` is N independent keyspaces**, not a shared one. The server refuses
  to start with `-w N > 1` unless `--independent-workers` is passed. Default is
  one worker.
- **Cluster mode is single-node.** Slot migration and replication exist;
  production multi-node does not.
- **CUDA is source-build only.** Prebuilt binaries are macOS arm64 and Linux
  CPU.
- **One maintainer**, working evenings. Crash, desync and data-loss reports get
  attention first; feature requests may sit.

### Correctness

The differential harness runs every command against a real `redis-server` and
reports only disagreements. Current state: 1,071 wrong-type probes → 15
divergences, **all of them the deliberate HLL fence**, 0 desyncs; 757/757 on
the correct-type matrix; dispatch sweep 1,485/1,485 probes across 292 commands
and 5 argument shapes; Gate 1 115/115.

That harness found most of what 0.x fixed, including several bugs that returned
a plausible wrong answer rather than an error — the class that no type-confusion
sweep catches.

### What the test program found (Sep 26–27)

Running every test file — not the 16 the old gate ran — surfaced these; each
now has a test that failed before the fix:

- **Vector sets were one server-wide set.** The key was ignored, re-`VADD`
  duplicated, and `VSIM` answered from other keys' elements. They are now one
  set per key with exact search.
- **`FT.SEARCH` filters were never applied.** The documented
  `FILTER @price:[10 50]` ran unfiltered with no error; every documented form
  now filters, anything else is refused. Unrecognised query forms
  answer an error instead of `[]`; `score` is the metric's distance
  instead of a quantized-space number; a multi-field `HSET` keeps the
  vector field readable.
- **Unknown command-line flags were silently accepted**, so a typo in
  `--requirepass` started an unauthenticated server. They now refuse to start
  and suggest the nearest flag.
- **Containers were never freed.** A key's list/hash/set/zset stayed allocated
  after `DEL` or after its last element was popped — ~1 GB of RSS over 60K
  ZADD/ZPOPMIN cycles on one key — and `COPY` shared the container between
  both keys.
- **Lists past 1,024 elements popped the wrong elements.** Draining a large
  list from the right dropped 255 elements per segment (`RPUSH 0..1499`, RPOP
  answered `1022, 767, 511, …`), and LPOP past the head side popped from the
  tail's end. The list tests drained by count, so neither was seen.
- A binary-lane `ATTEND_FINALIZE` of a multi-layer session crashed the server;
  an explicit `THRESHOLD 0` was ignored; a Lua `RENAME`
  left the new key pointing at freed memory.
- **Vector sets now survive a restart**: `VADD`/`VREM`/`VSETATTR`
  are WAL-logged and `SAVE`/`BGREWRITEAOF` serialize them. `TYPE` on a vector
  set answered `string` from the fast path; it answers `vectorset`.
- **The `HSET`-ingest `FT.OPTIMIZE` never calibrated its quantizer**,
  so components outside ±0.2 were clipped at ingest: fine for OpenAI-scale
  embeddings, wrong scores for anything larger (a raw N(0,1) vector scored
  ~1,134 against itself). Every build now calibrates one global range at
  mean ± 8σ of its own data. On the gate corpus that is the old ±0.2 and
  recall is unchanged (0.958). The calibration it borrowed turned out to take
  its square root with three Newton steps that only converge near variance 1,
  so every "3σ" range on embedding data had really been ~15σ.
- The seven example clients under `clients/` did not compile and sent an
  `FT.SEARCH` form the server never parsed, with 8-bit vectors; they are
  deleted. Any Redis client works — see the README.

### Quantized vector indexes

From 2026-03-28 until this release, `--turboquant` and `--nanoquant` silently built
plain INT8 while the startup banner said they were on, so every TurboQuant and
NanoQuant number recorded in that window was an INT8 number. Both now really
run. Measured on the gate dataset: PolarQuant (INT4) recall 0.965, TurboQuant
(INT3+QJL) 0.953, **NanoQuant (INT2) 0.46 — experimental**, not the 0.94 the
docs used to claim. Each keeps an FP32 re-rank copy, so none lowers total
memory yet.

---

## 0.x — February to September 2026

~1,150 commits from the initial engine to the public export. The arc, by theme:

### The engine (Feb–Apr)

Shared-nothing workers on raw pthreads, a Swiss-table hash map with SIMD probing,
zero-allocation fast path for 28 commands, ziplist/quicklist lists, skip-list
sorted sets. Four network tiers — kqueue, epoll, io_uring, AF_XDP — with the
epoll/io_uring crossover measured per CPU class rather than assumed. HNSW vector
search with SIMD distance kernels and INT8/INT4/INT3/INT2 quantization.

### The AI substrate (Apr–Jul)

`KV.PREFIX.*` and the shared KV cache; `ATTEND.*` with native Metal SDPA;
`SSM.PREFIX.*` for recurrent models; `MOE.EXPERT.*` expert paging;
`AI.SEMANTIC_CACHE`, `AI.ROUTE.*`, `RAG.*`, `AI.KNN_LM.*`, `NEURON.PKM.*`. The
substrate touches the shared keyspace zero times — the boundary is
one-directional by design.

### Durability (Jul)

WAL rotation with loud refusal at capacity (it used to return silently while the
client held an `+OK`), a file-backed blob tier for multi-MB values, arena
compaction at startup, aggregate durability for every keyspace type, and
effect-logging so non-deterministic commands replay their *resolved* effect.

### The correctness sweep (Aug)

The largest single body of work in 0.x, driven by the differential oracle. Bit
order in the bitmap family was mirrored; `LTRIM` deleted whole lists past 1,024
entries; `KEYS` and `SCAN MATCH` ignored their pattern entirely, which made the
recommended prefix-delete idiom delete the keyspace; `XADD` replaced any key
with a new stream; equal-score zset members came back in reverse insertion
order; `INCRBY k abc` returned success having added nothing. Dispatch-frame
consumption was made structurally safe after being fixed eleven times one arm at
a time.

### Getting publishable (Sep)

Multi-worker keyspace semantics fenced behind an explicit flag; loopback bind by
default; `CONFIG SET` stopped acknowledging writes it did not perform; the
inference sidecar stopped blocking `listen()` for 30 seconds; `--metal-attention`
made to work from a release tarball; `FT.CREATE` made to honour
`DISTANCE_METRIC`.


[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
