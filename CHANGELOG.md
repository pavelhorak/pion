# Changelog

Notable changes. Format loosely follows [Keep a Changelog]; versions before
0.9.0 were an internal `0.BUILD+SHA` counter and are summarised rather than
enumerated — there were roughly 1,100 of them.

## [Unreleased]

### Added

- **Stream consumer groups** (#40): XGROUP, XREADGROUP (`>`, history, COUNT,
  NOACK, BLOCK), XACK, XCLAIM, XAUTOCLAIM, XPENDING, XINFO GROUPS /
  CONSUMERS / STREAM FULL, XSETID, and Redis 7's stream metadata
  (entries-added, max-deleted-entry-id, recorded-first-entry-id, lag).
  Redis 8.2's XDELEX and XACKDEL, and KEEPREF / DELREF / ACKED on XADD and
  XTRIM trimming, are in too. Groups, consumers and pending entries are
  durable (WAL, snapshot, DUMP, COPY) and replicated. Checked against Redis
  8.10 in RESP2 and RESP3. Not implemented, and refused: XREADGROUP CLAIM,
  MAXCOUNT / MAXSIZE, XNACK, idempotent XADD.
- **Blocking list and sorted-set commands block** (#38): BLPOP and BRPOP
  answered nil at once; BRPOPLPUSH, BLMOVE, BLMPOP, BZPOPMIN, BZPOPMAX and
  BZMPOP did not exist. A blocked client parks, in FIFO order, until a key
  holds what it pops or its timeout passes.
- **Commands Redis 7 clients send** (#39): LCS, LOLWUT, MONITOR, ROLE,
  PFDEBUG, PFSELFTEST, REPLICAOF / SLAVEOF, FAILOVER, REPLCONF,
  RESTORE-ASKING, EVAL_RO / EVALSHA_RO / FCALL_RO, TIME. SYNC and PSYNC
  refuse.

### Changed

- **Scripts run the server's own commands** (#36). `redis.call` used a
  26-command copy of the server; every command now runs through the real
  dispatcher, logs its own WAL record and reaches replicas. Errors,
  conversions, the sandbox and the `redis.*` API follow Redis 8.10. FUNCTION
  libraries persist. `--lua-time-limit` and `--lua-memory-limit` replace the
  old instruction and heap caps.
- **The HNSW index file is the size of the index** (file v4). It held the
  node map and neighbor lists for the server's whole capacity: 329 MB for a
  100-vector index, rewritten on every FT.OPTIMIZE. A 100-vector rebuild went
  from about 410 ms to 6 ms. v3 files still load.
- **Admin and introspection commands do what they answer, or refuse** (#47):
  CLIENT (IDs, LIST, INFO, KILL, PAUSE, REPLY, UNBLOCK), COMMAND (INFO, LIST,
  GETKEYS, DOCS from Redis's own tables), SLOWLOG, ACL, LATENCY, MEMORY,
  MODULE, SHUTDOWN, and CLUSTER outside cluster mode. SHUTDOWN ABORT used to
  shut the server down.

- **Linux x86-64 runs the closed vector library's AVX-512 VNNI build by
  default on CPUs that have it.** `PION_VECTOR_VNNI=0` forces the x86-64-v2
  build, which was the default until now. The VNNI build has passed the
  differential on a Zen 5 and a Zen 4c CPU. On an EPYC 8124P (Zen 4c) it cut
  server CPU per query by 32% and raised QPS by 20–22%, winning all six
  rounds, with identical recall. Intel CPUs with AVX-512 VNNI have not been
  measured. `pixi run test-vector-differential` now pins the x86-64-v2 build,
  and `test-vector-differential-vnni` checks the VNNI build on VNNI hardware.

### Fixed

- **FT.SEARCH on the Linux x86-64 release binaries spent more than half its
  time in libm `fmaf`.** The reply rescoring, which reports the index
  metric's distance for the k rows a reply returns, used `fma()` on 8-wide
  vectors. The release targets x86-64-v2, which has no FMA instruction, so
  every lane became a call into libm: about 300,000 calls per query, more
  server time than the search itself. Those targets now multiply and add.
  macOS, Linux arm64 and builds for a CPU with FMA compile exactly as before.
  On an EPYC 8124P, with the build otherwise unchanged, server CPU per query
  went from 2,180 to 1,012 µs and peak QPS from 2,188 to 3,687 (+70%), in all
  four rounds, with identical recall.
- **A source build finds its Metal library from any directory.**
  `pixi run build` puts `./pion-server` at the repo root and the shader
  library in `src/ffi/`. The search covered that layout only relative to the
  working directory. A source build started from anywhere else, a data
  directory for instance, therefore reported `Metal Attn: NOT ACTIVE` and fell
  back to the MLX bridge. The search now also looks in `src/ffi/` next to the
  binary, and `tests/test_metallib_discovery.py` starts one from a temporary
  directory. Release tarballs were not affected: they find the library in
  `lib/`.
- **The Docker image's version label is Pion's.**
  `org.opencontainers.image.version` was `24.04`, inherited from the Ubuntu
  base image, on every image through 0.9.4. The release now sets it from
  `VERSION` and checks the pushed image's label against the tag, which also
  stops a tag cut without bumping `VERSION`.

- **Linux:**
  - the io_uring loop ticks when idle, so shutdown, WAIT, replica ACKs and
    expiry no longer stall on a quiet server; a failed `io_uring_setup` falls
    back to epoll instead of leaving a port nobody answers (#17, #21);
  - the CPU count comes from Linux's own constant, so Linux machines no
    longer all ran the `embedded` profile (#20);
  - a multi-worker warm restart serves no query before the index has loaded
    (#19);
  - a stopped server frees its port, and a restart waits for one still held
    (#22);
  - no libm `fmaf` call per lane on the x86-64-v2 build, in the vector
    library and in 60 places in the open kernels (#25).
- **Replies Redis clients parse:** ±inf scores no longer read INT64_MIN on
  x86, and scores print as Redis prints them (#18); HGETALL is a RESP3 map
  (#23); a bad password in HELLO's AUTH answers WRONGPASS (#24); RESP3 sets,
  doubles and pairs where Redis sends them (#30).
- **Parsing as Redis parses:** bitmaps (#31), option keywords matched by
  whole names and unknown options refused (#32), geo, now a sorted set as in
  Redis (#33), streams (#34), long-double INCRBYFLOAT (#35).
- **Each command that embeds text sees only its own entries** (#29).
- **INFO and `--version` say which vector build runs** (#26).
- **DUMP and RESTORE carry every type**, with Redis's options and errors;
  MIGRATE works (#41).
- **Pub/sub delivers every message whole**, across workers, with shard
  channels (#42).
- **RESET resets the connection**, and WATCH, QUIT and RESET run inside
  MULTI (#44).
- **A key past its deadline is gone for every command**, and expiry is
  logged, so a restart does not bring an expired key back (#45).
- **FT.SEARCH drops documents that were deleted, renamed or given a new
  vector** (#46). Writing the same vector again keeps a document indexed, and
  an FT.OPTIMIZE with no index defined no longer blocks the next index's
  ingest.
- **Vectors written by every path are indexed**: HSET inside MULTI/EXEC or a
  script, HMSET, HSETNX (#43).
- **OBJECT ENCODING** reports embstr by Redis 8's rule, which depends on
  the key's length and the platform's cache line.
- **Startup refusals exit 1**: an invalid `--tenant` setup exited 0. A
  cluster node whose replication port would pass 65535 refuses to start.

### Security

These were reachable from the wire, and are fixed in this release:

- io_uring: a request that overflowed the client buffer was still read past
  its end; a late completion could act on a new connection that reused the
  fd; the submission ring published entries before they were written.
- Heap overflows in fixed-size buffers: the XREAD BLOCK wake reply (64 KB),
  PUBLISH deliveries (4 KB) and DUMP (1 MB).
- SETBIT read the old bit before growing the bitmap: an out-of-bounds read.
- Replies and WAL records built from freed buffers (AI.KNN_LM.INFO,
  NEURON.PKM.INFO, COMMAND GETKEYSANDFLAGS, ACL LOG, the MoE manifest
  probes).
- XADD truncated a field or value longer than 65,535 bytes and reported
  success.
- Commands pipelined behind XREAD BLOCK were answered before it.
- A replica acknowledged a FULLRESYNC before applying it, so WAIT could count
  writes the replica did not hold.
- A removed key's TTL stayed behind and expired a later key of the same name.
- SELECT n answered +OK and stayed on database 0. It now refuses, as Redis
  does with one database.
- FLUSHALL and FLUSHDB were not written to the WAL, so flushed data came back
  after a restart and stayed on replicas.

## [0.9.4] — 2026-10-04

`pion-vllm-mlx` 0.1.4 works with mlx-lm 0.32, which took code fixes as well as
a wider cap. The server changes only in what it logs when it stops.

### Changed

- **`pion-vllm-mlx` 0.1.4 works with mlx-lm 0.32.** mlx-lm 0.32.0 came out on
  2026-10-01, its first release since April, and the `mlx` extra capped it at
  `<0.32`. Installing the extra beside mlx-lm 0.32 failed to resolve, or
  downgraded mlx-lm to 0.31.3, which lacks every architecture added since
  April. The cap is now `<0.33`. Three things 0.32 broke behind the cap are
  fixed:
  - **`HybridRetrievalCache`** loaded chunk K/V by assigning a cache's
    `state`. 0.32 added the offset to `state`, so hydrating a chunk in
    process raised `ValueError`.
  - **The hybrid-model wire path** (Qwen3.5-style models, and
    `softmax_bitexact`) serialized each cache slot by reading `state` by
    position. On 0.32 that tuple also holds Python scalars and, for an
    attention slot, the step-padded buffers. Slots are now read and written
    through attributes every supported mlx-lm has, so a stored blob is the
    same whichever version wrote it. A restored rotating cache also gets its
    write index back.
  - **`pion-vllm-mlx serve`** wraps mlx-lm's single-request path, which takes
    an extra argument in 0.32, so every request through `serve` would have
    raised `TypeError`.

  Checked on mlx-lm 0.31.3 and 0.32.0 against this release's server (M4 Mac
  mini):
  - The 20 mlx test files whose models are cached here pass on both. Tests
    that need Gemma 4 or Qwen models were not run.
  - A `serve` smoke test passes on both: chat, streaming, the Anthropic
    endpoint, and a restarted `serve` that restores 881 of 882 prompt tokens
    from Pion and gives the same greedy answer.
  - The seam test gained a model-free round trip for each cache class. The
    version matrix runs it on 0.20.1, 0.24.1, 0.28.4, 0.31.3 and 0.32.0.
  - `benchmarks/reproducers/cross_process_ttft.py` on mlx-lm 0.32.0
    (2,049-token prefix) measured 1,242 ms cold, 63–64 ms for a hit from a
    fresh process and 40 ms in the same process. That is at or above the
    published 17× and 20×, which stay as published.
  - The two Qwen3.5 reproducers in `benchmarks/reproducers/` read `state` the
    same way and are ported. Their changed functions were checked on
    synthetic caches; the reproducers were not re-run on the model.

### Measured

- **What the open build costs on Linux x86-64.** The machine was a Ryzen 9
  9950X (Zen 5), and the test was INT8 in the gate configuration with three
  rounds in rotated order.
  - The open reference gives 5,836 QPS.
  - `libpion_vector`'s default x86-64-v2 build gives 7,610 (open is −23%).
  - Its VNNI build gives 8,534 (open is −32%).
  - Recall@100 is 0.960 in every run.
  - The VNNI build also passed the differential on this machine. It stays
    opt-in until its lead over the default build is settled. Details are in
    [vector_engine.md](doc/vector_engine.md#open-build-and-the-closed-vector-library).

### Fixed

- **The vector benchmark ran nowhere but the maintainer's machine.**
  - `pixi run install-vdbbench` installed VectorDBBench into the pixi
    environment, while the harness runs it from `venv_zvec/`.
  - Stock VectorDBBench never sends `FT.OPTIMIZE`, so every query searched an
    index that was never built: recall 0.0 at an impossible QPS.
  - The task now creates `venv_zvec` and patches the client, on every
    platform. The harness refuses a run whose index was never built and names
    the task.
- **The Linux tarballs now state their minimum glibc (2.38).** The README and
  each Linux tarball's `README.txt` say so; the tarball reads the version from
  the binary itself. On an older system, such as Ubuntu 22.04 or Debian 12,
  the binary stops with `GLIBC_2.38 not found`, and the Docker image is the
  way to run it.
- **The Linux gate profile's CPU is labelled correctly.** The EPYC 8124P is a
  Zen 4c "Siena" part, not "Naples-class".
- **A clean stop no longer logs a worker death.** After SIGTERM, SIGINT or
  `SHUTDOWN`, each worker flushes its WAL and returns, and the thread wrapper
  reported every return as `worker 0 DIED (event loop returned)`. Each
  `brew services stop pion` wrote one into the service log. A drained worker
  now logs `worker 0 stopped (shutdown)`; `DIED` is kept for a worker that
  returns without a shutdown request.

## [0.9.3] — 2026-10-03

The Linux binaries link the tuned vector library, and the PyPI page gives the
prefix sweep. The macOS server is unchanged apart from its version.

### Changed

- **The Linux tarballs and the Docker image link `libpion_vector`.** It is the
  free closed library with the tuned 1536-dim beam searches and product-key
  kernels. Until now only the macOS build linked it, and the Linux builds ran
  the open reference kernels in `src/vector/reference/`.
  - Every release leg (macOS arm64, Linux x86-64, Linux arm64) checks the
    library against its open reference before packaging. The check,
    `pixi run test-vector-differential`, requires identical results.
  - On x86-64 the library runs its x86-64-v2 build. Its AVX-512 VNNI build
    stays opt-in (`PION_VECTOR_VNNI=1`) until it has passed that check on VNNI
    hardware.
  - The speed difference on Linux has not been measured. On an M4 Mac the open
    kernels are 22–34% slower at 1536 dims
    ([vector_engine.md](doc/vector_engine.md#open-build-and-the-closed-vector-library)).
  - The image carries the library's licence at `/opt/pion/LICENSE-pion-vector`,
    as the tarballs do, and its licence label names both licences.
  - `pixi run build-open`, the build with no closed code, now exists on both
    Linux targets too.
- **`pion-vllm-mlx` 0.1.3.** Its PyPI page gives the prefix sweep from a
  separate process: 11× at 1,035 tokens, 4.5× at 268 and 1.5× at 34. The 0.1.2
  page said that a prefix under about a thousand tokens has little to save,
  which the 0.9.2 measurements contradict. The README gives the same sweep.
  The page's summary line now describes the prompt cache. It used to describe
  an attention backend for vllm-mlx, a module the package no longer contains.
  The package's code is unchanged.
- **CI runs on Linux arm64 too**, alongside Linux x86-64 and macOS arm64.
  CONTRIBUTING gives the perf-gate commands. Before, it named `/gate`, a
  command that does not ship with the repository, and said that opening a pull
  request runs no checks.

### Fixed

- **`benchmarks/valkey-benchmark/valkey-benchmark.py` removes its temp
  directories at exit.** Every run left a Redis and a Valkey data directory in
  `$TMPDIR`, `--pion-only` runs included.

## [0.9.2] — 2026-10-02

Corrections, the demo, and the PyPI page; the server's behaviour is unchanged.

### Corrected

- **Time-to-first-token ratios were too high, and one row was mislabelled.**
  Every TTFT harness timed vanilla mlx-lm as one forward over the whole prompt,
  with its logits evaluated. That computes the vocabulary projection at every
  prompt position. mlx-lm's own `generate_step` never does: it prefills with only
  the cache evaluated, then runs the last token alone. So the cold side was ~30%
  slow at 2K tokens (1,624 vs 1,246 ms, same first token), and every ratio
  against it was too high.

  Each harness now prefills the way `generate_step` does, checked within 2% of
  it. Re-measured 2026-10-02 on an M4 Mac mini against Pion 0.9.1:
  - **From a separate process, 2,049-token prefix and a 16-token question:**
    1,242 → 73.9 ms, **17×** (was 1,558 → 64.7 ms, 24×). Prefix sweep:
    1.5× / 4.5× / 11× / 17× at 34 / 268 / 1,035 / 2,049 tokens (was
    1.6× / 6.0× / 16.4× / 24×).
  - **Same process, same prompt and question:** 1,242 → 61.9 ms, **20×** (was
    1,530 → 30.2 ms, 50.6×). That row was also mislabelled. `tests/bench_ttft.py`
    built a fresh `PionPromptCache` for every request, so it timed Stage 2's
    wire lane, where Pion computes the attention, and not the in-process lane
    the README named. Its one-token suffix was a best case besides. Both 1B rows
    now come from `cross_process_ttft.py` (`--same` adds this one), and
    `bench_ttft.py` times both lanes; at 2,048 tokens with a one-token suffix
    the wire lane is 28.6× and the in-process lane 86×.
  - **Stage 2 lanes at ~316-token prefixes:** in-process 4.8× (was 6.51×).
    Binary and RESP p50 are 127 and 131 ms (were 92.1 and 108.6 ms); the old
    figures ran the suffix in one pass, and mlx-lm runs it in two.
  - **The mixed workloads (mean over all requests, cold ones included):**
    Stage 1 3.1× at ~316 tokens and 5.4× at ~2,514 (were 5.07× and 6.96×).
    Stage 2's in-process lane 4.8× and 13.2× (were 3.31× and 6.75×). These two
    went up; the harness behind the old Stage 2 pair is not recorded.
  - **Stage-1 workload, 5 prompts × 30 queries:** 612 → 84 ms, 7.3× (was
    846 → 91 ms, 9.26×).
  - **Qwen3.5-4B hybrid at 2K / 4K / 8K:** 14.0× / 25.3× / 29.0× (vanilla
    5.0 / 10.2 / 21.3 s). The 0.9.0 notes below said
    24.7× / 32.7× / 11.6×, and the reproducer README said 24.77× / 36.0× / 29.5×.
  - **Hybrid retrieval, 100 SQuAD v2 queries:** 3.0× p50 in-process and 2.7×
    over the wire (was 4.5×). That bench also left the cache hydration
    (`prepare()`) out of its clock.

  The 64K sparse row is unchanged: its vanilla prefill is chunked and never
  computed the extra logits.

### Changed

- **`examples/prompt_cache_demo.py` shows what a tester will see.**
  - It runs against Pion by default. The old default re-prefilled and printed
    ~1× by construction.
  - It times five requests the way an app makes them and reports their median.
  - It says what to start when no server answers, and exits non-zero on failure.
  - It names the one-time cost of the first request after a store, and says why
    its requests sit above the separate-process row.
- **`pion-vllm-mlx` 0.1.2** carries the corrected numbers to its PyPI page.

### Removed

- **`benchmarks/reproducers/bench_qwen3_5_warm_ttft.py`**, a one-length copy of
  `sweep_qwen3_5_warm_ttft.py`, which runs any length with `--target-tokens N`.
  Against the corrected baseline its 174-token default measures 1.7×, below the
  2× it required to pass. The 3.92× it used to back came from the old baseline.

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
  output verified BLEU 1.000. *(Both ratios were overstated; see Corrected
  under 0.9.2.)*
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
  *(Overstated; see Corrected under 0.9.2.)*
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
