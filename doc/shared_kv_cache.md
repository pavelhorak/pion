# Shared KV Cache Serving — Pion as LLM Memory

Pion stores the LLM's computed KV cache tensors and serves them to future requests on the same prefix, eliminating prefill computation for shared context. Single-instance, MLX runtime, Apple Silicon, fp16 default.

## Quantized tier — `mlx4g32`

`KV.PREFIX.REGISTER <ns> <kv_dim> mlx4g32` (and `V.CREATE ... VQUANT mlx4g32`)
stores K/V in mlx's `QuantizedKVCache` layout: int4, group 32, affine.

```
per token, per layer, D = kv_dim
  [packed uint32 x D/8]   4-bit codes, element j of a group at bits 4*(j%8)
  [scales  fp16  x D/32]
  [biases  fp16  x D/32]
```

640 B/token at D=1024 against fp16's 2048 — **3.2×** smaller, which
`tests/test_gh148_mlx4g32_tier.py` asserts. At that ratio, K/V that takes 12 GB
in fp16 takes under 4 GB.

Both parameters are measured, not conventional:

- **group 32, not 64.** g64 corrupted recalled facts at digit grain in real
  generations (`"2430-04-22"` for `"2030-04-22"`). Knowledge held as KV is
  bits-fragile the same way weight-held knowledge is.
- **affine, not symmetric.** Storing a per-group scale *and* bias reconstructs
  more accurately than symmetric int4 (`tests/test_gh148_mlx4g32_tier.py`
  asserts it), and K's error dominates the end-to-end result.

The scale is round-tripped through fp16 *before* the codes are chosen, so the
quantizer targets the scale the reader will actually see.

## Measured

**Measured — Stage-1 workload harness (5 prompts × 30 queries, 96.7% hit rate, fp16, Apple Silicon):**

| Model | Cold TTFT | Warm TTFT | Reduction | Throughput | First-token |
|---|---:|---:|---:|---:|:---:|
| Llama-3.2-1B-Instruct-4bit | 765 ms | **83 ms** | **89.2%** (9.21×) | 6.49× | 100% (50/50) |

Measured 2026-10-06 on an M4 Mac mini with `tests/test_kv_prefix_workload.py
--queries 30 --prompt-repeats 8`
([raw output](../benchmarks/results/2026-10-06-mac-m4/kv_prefix_workload_q30_r8.txt)),
mean TTFT over all 150 requests, each prompt's cold first one included, with the
cold side prefilled the way mlx-lm's `generate_step` does. Earlier versions of this table timed a cold
side that also computed logits at every prompt position, which no generation
does; the changelog has the correction. A 3B row is not measured, so none is shown.

Cross-instance verified: a fresh second client (separate socket, separate model object) sees `+HIT` before any local work, fetches K/V the first client stored, and produces a **bit-identical 50-token greedy completion (BLEU 1.0000)**.

This row is this harness's own workload (150 requests, 5 prompts × 30 queries, at its `--prompt-repeats 8` prefix), so it does not line up with the README's two headline numbers: **17×** is one separate process hitting a 2,049-token prefix (`benchmarks/reproducers/cross_process_ttft.py`), and **26×** is the process that stored the prefix asking again, through Stage 2's in-process lane (`cross_process_ttft.py --same`; the table below times each lane on its own).

---

## Quick Start

Start the server:

```bash
./pion-server --kvcache -w 1
```

Then, from Python (`pip install 'pion-vllm-mlx[mlx]'`):

```python
from mlx_lm import load, generate
from pion_vllm_mlx import PionPromptCache

model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
pc = PionPromptCache(model, vquant="fp16")

system = "You are a support agent for Acme. Answer in one sentence."   # shared by every request
prefix_ids = tok.encode(system)
ns = PionPromptCache.make_namespace("llama-3.2-1b-4bit", "fp16", system)

# The first call anywhere prefills locally, registers with Pion and stores the K/V.
# Every later call — this process, another one, or after a restart — fetches it.
cache = pc.get_or_prefill(prefix_ids, namespace=ns)
print(generate(model, tok, prompt=tok.encode(" How do I reset my password?", add_special_tokens=False),
               prompt_cache=cache, max_tokens=40))
```

`pc.stats()` reports hits, misses, hit_rate, fetch_ms_total, store_ms_total.

### Heterogeneous KV cache

`PionPromptCache(boundary_protect=N)` switches the per-prefix register from the legacy uniform `KV.PREFIX.REGISTER` to two raw `V.CREATE … SCHEMA` calls (one per K/V side). K stays fp16 across all layers — K drives softmax routing and tolerates quant noise poorly. V uses fp16 for the first N + last N layers ("boundary protection") and the user's `vquant` for the middle layers.

```python
# Boundary-protect K-V split: middle layers in fp8, boundary layers in fp16
pc = PionPromptCache(model, vquant="fp8", boundary_protect=2)
```

fp8 V on the middle layers carries less wire data per layer, and the fp16
boundary layers protect routing. No comparison against uniform fp16 is published
with raw output yet, so measure it on your own workload
(`pion-vllm-mlx/tests/test_prompt_cache_workload.py --vquant fp8 --boundary-protect 2`).

Trade-off: the SCHEMA path skips `KV.PREFIX.REGISTER`'s cross-worker directory publish (single-worker visibility only). A future `KV.PREFIX.REGISTER.SCHEMA` server command would lift this.

`boundary_protect` requires `kv_dim` divisible by 32 when `vquant ∈ {fp8, turbo4, turbo3, turbo2}`. Llama-3.2-1B (`kv_dim=512`) and Llama-3.2-3B (`kv_dim=1024`) both qualify.

---

## Wire Protocol

Three production wrapper commands, plus the underlying V-store path (`V.STOREBATCH` / `V.FETCH ... RANGE`).

### `KV.PREFIX.REGISTER <ns_key> <kv_dim> <vquant> [BLOCKS <block_size> <hash_count> <hash_blob>]`

Creates two V-store sessions, `<ns_key>_pk` (keys) and `<ns_key>_pv` (values), with the given quantization format. After `REGISTER`, standard `V.STOREBATCH` and `V.FETCH ... RANGE` work against the derived sids.

`vquant` ∈ `{int8, turbo4, turbo3, turbo2, fp16, fp8, mlx4g32}` (mlx4g32 = int4 group-32 affine). fp16 is the production default (BLEU 1.0000 cross-instance, a mean of 0.979 over 20 questions against standalone).

```
> KV.PREFIX.REGISTER my_app|v1|llama|fp16|prompt_a 512 fp16
+OK
```

**Optional `BLOCKS` clause** — carry a within-prefix block hash table for cache-aware routers. `<block_size>` is the token count per block (typically 16 or 64), `<hash_count>` is the number of u64 hashes, `<hash_blob>` is `hash_count × 8` bytes of little-endian u64 hashes packed in token order. Hashes are stored on the K-side session and survive WAL replay + snapshot reload. Use the new `KV.PREFIX.BLOCKS` / `KV.PREFIX.MEMBERSHIP` commands below to query them.

```python
import struct
hashes = [hash_block(...) for block in blocks]
blob = struct.pack("<" + "Q" * len(hashes), *hashes)
resp.call("KV.PREFIX.REGISTER", ns_key, str(kv_dim), "fp16",
          "BLOCKS", "64", str(len(hashes)), blob)
```

### `KV.PREFIX.LOOKUP <ns_key> [TOKENS <n>] [PREFILL_MS <ms>]` → `+HIT` or `+MISS`

Tells the client whether both K and V sessions exist in V-store for this namespace.

```
> KV.PREFIX.LOOKUP my_app|v1|llama|fp16|prompt_a
+HIT
```

The options only change what the value receipt (`PION.STATS`) records for a hit; the answer is the same. `TOKENS <n>` credits the n tokens the caller actually restored instead of the namespace's own token count. That is for a client whose reuse spans several namespaces and reads rows with `V.FETCH`: `pion-vllm-mlx serve` stores a conversation as a chain of segments and names the leaf plus the total, once per restore. `PREFILL_MS <ms>` is that restore's measured cold prefill, when the caller has one. An unknown option or a missing value is refused with `-ERR`, and nothing is recorded.

### `KV.PREFIX.DROP <ns_key> [<ns_key> ...]` → `:<dropped>`

Frees the K and V sessions of each prefix and writes the drop to the V-store WAL, so a restart does not replay the rows back. A namespace that is not held counts 0. This is for clients that run their own eviction policy. `pion-vllm-mlx serve --pion-budget-gb` keeps its lineages under a byte budget and evicts the least-recently-used leaf segment. The V-store's own LRU (it evicts when its 256 session slots are full) goes by last access alone. For a lineage, that is the root: it is written once and never touched while the leaf grows.

> **Important — V-store state ≠ Metal session-cache state.** `KV.PREFIX.LOOKUP` reports only V-store registration (Stage-1 path). For Stage-2 wire-mode consumers (sparse-mask / fused-sparse / mlx-lm patch) that read K/V from the Metal SDPA session cache populated by `ATTEND.PREFIX.STORE`, also probe **`ATTEND.PREFIX.LOOKUP <sid> <layer_id>`**. A stale V-store HIT while the Metal cache is cold causes Stage-2 push-cold to skip — checking both is required. `PionPromptCache.lookup(namespace)` does this automatically when `stage2=True`.

### `KV.PREFIX.BLOCKS <ns_key>` → bulk string `[block_size 4B LE][block_count 4B LE][hashes]` or `+UNKNOWN`

Returns the within-prefix block hash table registered via `KV.PREFIX.REGISTER ... BLOCKS`. The bulk-string body is `8 + block_count * 8` bytes: a `block_size` (u32 LE), a `block_count` (u32 LE), then `block_count` u64 LE hashes in token order. Used by cache-aware routers (and audit tooling) to reproduce the residency picture client-side, or to feed a `KV.PREFIX.MEMBERSHIP` probe with the canonical hash set.

Returns the simple string `+UNKNOWN\r\n` when the namespace exists but no block table is registered, OR when the namespace was never registered on any worker. When the namespace lives on a different worker (cross-worker directory hit), returns `-ERR KV.PREFIX.BLOCKS session lives on worker N` so the caller can pin its connection — same pattern as `KV.PREFIX.WARM`.

```
> KV.PREFIX.BLOCKS my_app|v1|llama|fp16|prompt_a
$1256
<binary: block_size=64, block_count=156, 156 × u64 hashes>
```

### `KV.PREFIX.MEMBERSHIP <ns_key> <hash_count> <hash_blob>` → bulk-string bitmap or `+UNKNOWN`

The router supplies the block hashes it's looking for (in any order); the server returns a `ceil(hash_count / 8)`-byte bitmap, with bit `i` set iff probe hash `i` is in the namespace's registered table. Single round-trip, bandwidth-efficient: 100K tokens at `block_size=64` → 1,562 blocks → 196-byte response.

Returns `+UNKNOWN\r\n` when no block table is registered or the namespace doesn't exist. Cross-worker rebound matches `KV.PREFIX.BLOCKS`.

Server-side compute is O(K log N) via binary search over a sorted parallel copy of the registered hashes (built once at `REGISTER` time). Measured end to end over loopback for K=N=1,562 (a 12.5 KB request) on an M4 Mac mini: **128.6 µs p50 / 161.0 µs p99** (`tests/test_kv_prefix_blocks.py`, which fails above 400 µs p50; [raw output](../benchmarks/results/2026-10-06-mac-m4/kv_prefix_blocks.txt)).

```python
probe_blob = struct.pack("<" + "Q" * len(probe_hashes), *probe_hashes)
reply = resp.call("KV.PREFIX.MEMBERSHIP", ns_key, str(len(probe_hashes)), probe_blob)
# Parse bulk-string body as a bitmap; bit i set ↔ probe_hashes[i] is cached.
```

### `KV.PREFIX.INFO` → bulk string

Global stats: registered prefix count, total tokens, total fetches.

```
> KV.PREFIX.INFO
$104
registered_prefixes:5
total_prefix_tokens:5800
vstore_sessions:10
vstore_total_fetches:145
```

Also reported: `vstore_bytes`, the K/V bytes held across every session in its stored format. It's what a byte budget is measured against and what a snapshot writes; resident memory can be up to about 2× that, because buffers grow by doubling. `wal_bytes` is the on-disk size of the V-store WAL, which only grows until `KV.PREFIX.SAVE` snapshots and truncates it.

### `KV.PREFIX.SAVE [path]` → `+OK`

Snapshots every V-store session to `pion.vstore.<worker>` (or a relative `path`), then truncates the V-store WAL. The snapshot is written to `<path>.tmp`, checked write by write, flushed to stable storage (`F_FULLFSYNC` on macOS) and renamed into place. The WAL is truncated only after all of that succeeds. A crash, a full disk or a power cut mid-save leaves the previous snapshot and the WAL as they were.

### `V.FETCH <session_id> <layer_id> RANGE <start_id> <end_id>`

Single round-trip per layer-side, regardless of prefix length. Sidesteps the 64-token RESP frame limit that the legacy id-list form hits at ~60-token prefixes. Returns concatenated dequantized FP32 values.

### `KV.PREFIX.WARM <ns_key> <H> <D> [<attend_sid>]` → `+<N>`

Server-side rehydrate of the Metal SDPA session cache from V-store. Walks both `<ns>_pk` and `<ns>_pv` sessions, dequantizes per layer to fp32, transposes from V-store layout `[N, H*D]` into ATTEND.PREFIX.STORE layout `[H, N, D]`, and pushes each layer back via `pion_metal_sdpa_store_kv`. Returns the number of layers rehydrated as `+N\r\n` (or `+N skipped_dim=K\r\n` when some layers had a kv_dim ≠ H*D and were skipped).

The optional 4th arg overrides the ATTEND-side session id; default = `<ns_key>`. The production consumer (`PionPromptCache._attend_session`) stores ATTEND state under `<namespace>_attn` and passes that here.

Call this after **ATTEND.PREFIX.QUERY** returns `-COLDMISS ...`, or proactively before a hot batch of queries against a namespace that may have been evicted. Its cost grows with the prefix length: one V-store dequant, one CPU transpose and one Metal copy per layer.

```
> KV.PREFIX.WARM my_app|v1|llama|fp16|prompt_a 8 64 my_app|v1|llama|fp16|prompt_a_attn
+16
```

`KV.PREFIX.INFO` reports four cold-tier telemetry fields:
```
sdpa_warm:256        ← live slots in the Metal SDPA session cache
sdpa_cold:1          ← demoted entries in the cold registry
sdpa_demotions:1     ← lifetime WARM→COLD transitions on this worker
sdpa_rehydrates:1    ← lifetime COLD→WARM transitions on this worker
```

### Cold-tier state machine

Three states per `(session_id, layer_id)`:

| State | Where it lives | Wire signal | Cost to query |
|---|---|---|---:|
| **WARM** | Metal SDPA slot (live K_buf/V_buf) | `+HIT` from `ATTEND.PREFIX.LOOKUP` | sub-ms (0.44 ms at H=8, N=2048, D=128) |
| **COLD** | Per-worker cold registry (metadata only); K/V on disk in V-store | `+COLD` from `ATTEND.PREFIX.LOOKUP`; `-COLDMISS …` from `QUERY*` | one `KV.PREFIX.WARM`, then sub-ms |
| **MISSING** | Nowhere | `+MISS` from `LOOKUP`; `-ERR session not found` from `QUERY*` | full cold prefill needed |

Eviction triggers when the per-worker WARM slot table (256 slots) fills. The LRU slot is picked by `last_access_ns` (mach_absolute_time), its Metal `K_buf`/`V_buf` are released, and `(key, H, N, D, last_access_ns)` move to the cold registry (`SDPA_COLD_SLOTS=1024` per worker). A subsequent `ATTEND.PREFIX.QUERY` short-circuits via `session_state(...)==2` and returns `-COLDMISS …`. The consumer issues `KV.PREFIX.WARM` and retries; the warm path stamps `last_access_ns` so the rehydrated session is now the *most-recent*, not the next eviction victim.

The `PionPromptCache` client handles this transparently — `attend_query` / `attend_query_fused` / `attend_query_sparse_auto*` detect `-COLDMISS` (RESP lane) or `STATUS_COLDMISS=0x03` (binary fast lane), call `_warm_namespace(...)`, and retry once. `pcache.cold_rehydrates_observed` exposes the count for telemetry.

---

### The value receipt — `PION.STATS`

The server keeps a per-worker ledger of what the prefix cache actually did for you:

```
PION.STATS            → map of 16 fields (RESP3 %-map; RESP2 flat array)
PION.STATS RESET      → +OK, counters zeroed (uptime is not a counter)
INFO                  → the same numbers under a `# Pion` section
```

| Field | Meaning |
|---|---|
| `kvprefix_hits` / `kvprefix_misses` | `KV.PREFIX.LOOKUP` answers |
| `kvprefix_tokens_served` | prefix tokens whose prefill was skipped (K-side layer-0 token count at hit time, or the `TOKENS` the client named on `KV.PREFIX.LOOKUP`) |
| `kvprefix_bytes_served` | `V.FETCH` payload bytes delivered |
| `prefill_seconds_avoided` | cumulative prefill time skipped, **measured + estimated** |
| `prefill_seconds_avoided_measured` | the part backed by client-reported `PREFILL_MS` only |
| `kvprefix_hits_measured` | hits credited with a reported time |
| `semantic_hits` / `semantic_misses` | `AI.SEMANTIC_CACHE GET` and every other `cache_get` caller |
| `moe_hits` / `moe_misses` | `MOE.EXPERT.FETCH` tier hits |
| `vector_queries` | `FT.SEARCH` + `FT.HYBRID` answered |

Two kinds of number, kept apart on purpose. `PionPromptCache` times its local
cold prefill and sends it as `KV.PREFIX.REGISTER ... PREFILL_MS <ms>`; every
later hit on that prefix is credited exactly that time — a receipt, not an
estimate. A prefix registered without `PREFILL_MS` (a hand-rolled client, or a
server restart, since the reported time is deliberately not persisted) is
credited `tokens × 555 µs` (`LEDGER_EST_PREFILL_US_PER_TOKEN`), a per-token
cost derived once from Llama-3.2-1B-Instruct-4bit on an M-series Mac. Larger models cost more per token, so the estimate is conservative
for them, and it is labelled an estimate wherever it appears.

The counters are **per worker** — they live in the worker that answered — so a
connection pool spanning `-w N` workers reads each worker's own receipt.
Formatting happens on demand in the reply; recording is a counter increment on
an already-slow-path command, so nothing on the KV fast path changed.

## Cache Namespace Contract

The namespace key is the *only* contract — it must encode every load-bearing piece of execution context. Mismatch is silent corruption, not a runtime error.

```python
ns = PionPromptCache.make_namespace(
    model_id,           # e.g. "mlx-community/Llama-3.2-1B-Instruct-4bit"
    tokenizer_hash,     # changes invalidate the cache
    rope_theta,         # rope scaling settings
    quant_format,       # "fp16" / "int8" / "turbo4"
    adapter_id,         # LoRA / adapter identity if any
    prompt_text,        # the actual prompt
)
```

`make_namespace` returns `sha256("|".join(parts))[:32]`.

The namespace is not a secret. Anyone who can reach the server and knows (or guesses) a key can read that prefix, so gate the server with `--requirepass`, and give each tenant its own credentials with `--tenant` (see the [security model](multi_tenant.md)) rather than relying on unguessable keys.

---

## How It Works

```
Cold path (first request per prompt):
  client → mlx_lm.forward(prefix + suffix) → output
                            │
                            └── cache populated locally
  client → KV.PREFIX.REGISTER → V.STOREBATCH per layer (K and V) → Pion

Warm path (every subsequent request on the same namespace):
  client → KV.PREFIX.LOOKUP → +HIT
  client → V.FETCH RANGE per layer → fp32 K and V tensors
  client → MLX KVCache.update_and_fetch(K, V) per layer
  client → mlx_lm.forward(suffix only, cache=rebuilt) → output
```

Pion's V-store stores per-token, per-layer values indexed by token ID. K is treated as just another value array — the wire format is the same. Boundary-layer FP16 protection is exposed via `PionPromptCache(..., boundary_protect=N)` — first/last N layers stay FP16 while middle layers go to the chosen `vquant`, which reduces drift on the quantized formats (no drift measurement is published with raw output yet).

---

## Stage 2: `ATTEND.PREFIX.*` — Pion computes attention on cached K/V

The Stage 1 design above has the *client* run attention: it fetches K/V via `V.FETCH ... RANGE` and runs the model's attention kernel locally. Stage 2 keeps K/V resident in MLX-side memory on Pion's sidecar and runs attention there — only the query Q crosses the wire on each call.

### Wire forms

```
ATTEND.PREFIX.STORE  <sid> <layer> <H> <N> <D> <K_blob> <V_blob>
ATTEND.PREFIX.LOOKUP <sid> <layer>                                       → +HIT / +MISS
ATTEND.PREFIX.QUERY  <sid> <layer> <H> <D> <top_k> <Q_blob> [<fa_window>] → bulk H*D fp32
ATTEND.PREFIX.QUERY_FUSED <sid> <layer> <H_q> <D> <S_suf> <H_kv>
                          <Q> <K_suf> <V_suf> <head_map> [<fa_window>]   → bulk H_q*M*D fp32 (suffix+merge fused)
ATTEND.PREFIX.QUERY_SPARSE <sid> <layer> <H> <D> <K_sparse_max>
                           <Q> <indices> <counts> [<fa_window>]          → bulk H*D fp32 (caller-supplied indices)
ATTEND.PREFIX.QUERY_SPARSE_AUTO <sid> <layer> <H_q> <D> <B> <K_top>
                                <H_kv> <Q> <head_map> [<fa_window>]      → bulk H_q*D fp32 (server picks indices via block-mean top-K)
ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED
       <sid> <layer> <H_q> <D> <B> <K_top> <H_kv> <S_suf>
       <Q> <K_suf> <V_suf> <head_map> [<fa_window>]                       → bulk H_q*D fp32 (sparse-prefix + dense-suffix + merge, single dispatch)
```

Native Metal SDPA via `src/ffi/metal_compute.metal` + `src/ffi/metal_wrap.m`, selected by `--metal-attention` (or `--metal-attention-fp16` for vanilla mlx-lm precision parity). No Python, no Unix socket. **D ∈ {32, 64, 96, 128, 160, 192, 256, 512}** (D=512 is for Gemma 4 full-attention layers; dynamic threadgroup memory scales `s_o` per-PSO). Six kernels per D-PSO: `sdpa_q1_fp32/fp16`, `sdpa_batched_q_fp32/fp16`, `sdpa_batched_q_fused_fp32/fp16`, `sdpa_q1_sparse_fp32/fp16`, `sdpa_q1_sparse_fused_fp32/fp16`. Multi-worker (per-worker session caches with linear probing + tombstones). End-to-end M=1 ATTEND.PREFIX.QUERY median **0.441 ms** at H=8/N=2048/d=128, against 0.456 ms for MLX's own `scaled_dot_product_attention` on the same shape (`tests/bench_pion_metal_attention.py`, [raw output](../benchmarks/results/2026-10-06-mac-m4/metal_attention.txt)). Bit-equivalent to vanilla mlx-lm on `tests/test_mlx_lm_patch.py` (20/20 token agreement).

**Sparse-mask path:** server picks block-mean top-K from resident K/V (block size B, K_top blocks), runs sparse SDPA over the picked indices. Optional fused variant also merges a caller-supplied dense suffix in the same dispatch — the "wire-mode sparse" consumer in `pion-vllm-mlx/pion_vllm_mlx/mlx_lm_patch.py` uses it. `examples/sparse_mask_64k_niah.py` runs single-needle NIAH at 64K on Gemma-4-E2B-it-4bit through this path (in-proc lane, sparse on full layers, K_block=64, K_blocks=8); on 2026-10-07 both vanilla mlx-lm and this path found the needle, the warm call in 124.4 ms against vanilla's 54.4 s cold prefill ([raw output](../benchmarks/results/2026-10-07-mac-m4/sparse_mask_64k_niah.txt)). The 2026-10-06 run, in which neither found it, had 397 `<bos>` tokens in its prompt; the example now puts one at position 0. Wire-lane consumer end-to-end on Llama-3.2-1B (GQA): same magic-number answers as vanilla. Validation gates: `tests/test_attend_sparse_kernel.py`, `tests/test_attend_d512.py`, `tests/test_attend_sparse_auto.py`, `tests/test_attend_sparse_auto_fused.py`, `tests/test_wire_sparse_consumer.py`.

### Why this matters

A single-shot `ATTEND.QUERYBATCH H N D top_k Q K V` marshals the whole K/V on
every call: 8 MB at H=8, N=2048, D=64 in fp32. The two-phase pattern uploads
K/V once and keeps only Q on the wire.

### Measured (Llama-class shape: H=8 N=2048 D=64, top_k=2048, `--metal-attention`)

| Step | Result |
|---|---|
| `ATTEND.PREFIX.STORE` (the 8 MB push, once) | 4.0 ms |
| `ATTEND.PREFIX.QUERY`, Q only, median of 10 | **0.47 ms** (2,143 q/s) |
| Agreement with CPU softmax(QK^T)·V | cosine **1.0000** |

`tests/test_attend_prefix.py` on an M4 Mac mini, 2026-10-06
([raw output](../benchmarks/results/2026-10-06-mac-m4/attend_prefix.txt)); the test
fails below cosine 0.99 or 100 q/s. No published harness times
`ATTEND.QUERYBATCH`, which re-sends K/V on every call.

Test: `tests/test_attend_prefix.py`. Validated via `pion-server --kvcache --metal-attention -w 1`. End-to-end head-to-head bench (Pion-Metal vs MLX raw): `tests/bench_pion_metal_attention.py`.

### When to use Stage 1 vs Stage 2

- **Stage 1 (`KV.PREFIX.*` + `V.FETCH RANGE` + `PionPromptCache`)** — the client's inference engine runs attention locally on fetched K/V. Right when the client is on the same host as Pion (Apple Silicon unified memory) or when the inference engine doesn't expose a hook for offloading attention.
- **Stage 2 (`ATTEND.PREFIX.STORE/QUERY`)** — Pion's MLX sidecar runs attention on K/V that never leaves Pion's process memory after the initial push. Right when the inference engine accepts an external attention output (e.g., custom vLLM CacheEngine, exo's `gpu_attention` mode), or when many queries share the same K/V across sessions and the wire cost of Q+K+V re-marshaling dominates.

The two are complementary; the same prefix can live in both V-store (for clients that fetch K/V) and the MLX sidecar (for clients that offload attention).

### Stage 2 consumer: `install_pion_attention_patch()` (shipped)

The mlx-lm `Attention.__call__` monkey-patch is now shipped in
`pion-vllm-mlx.mlx_lm_patch`. It does the online-softmax merge between
cached prefix attention (run on Pion's MLX sidecar via `ATTEND.PREFIX.QUERY`
with the LSE trailer) and locally computed suffix attention.

```python
from mlx_lm import load
from pion_vllm_mlx import (
    PionPromptCache, install_pion_attention_patch,
    make_pion_prompt_cache,
)

model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
install_pion_attention_patch()                            # one-time patch
pc = PionPromptCache(model, vquant="fp16", stage2=True)
pc.get_or_prefill(prompt_ids, namespace=ns)               # cold path
cache = make_pion_prompt_cache(model, ns, pc, len(prompt_ids))
out = model(suffix_ids, cache=cache)                      # attention runs on sidecar
```

#### Measured TTFT win (warm path, median of 10 runs; 2026-10-06, M4 Mac mini)

| Model | Prompt | Vanilla cold | Stage 1 rebuild (B) | Stage 2, wire lane (C) | Stage 2, in-process (D) |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B-4bit | 256  | 148.6 ms | 12.4 ms (12.0×) | 19.6 ms (7.6×) | 8.5 ms (17.6×) |
| Llama-3.2-1B-4bit | 1024 | 570.1 ms | 22.7 ms (25.2×) | 20.6 ms (27.7×) | 9.3 ms (61.4×) |
| Llama-3.2-1B-4bit | 2048 | 1,170.9 ms | 37.3 ms (31.4×) | 23.9 ms (48.9×) | **11.3 ms (104×)** |

Every warm path restores all prompt tokens but the last, then runs the last
one, and every path reproduces vanilla's first token. A one-token suffix is
the best case for a cache. A real question adds its own prefill: with a
16-token question the in-process lane takes 46.2 ms at 2,049 tokens
(`cross_process_ttft.py --same`).

At 1,024 tokens Stage 1 and Stage 2's wire lane are close; at 2,048 the wire
lane is 1.56× faster. Below ~1K, the wire lane's per-layer round trips
cost more than the transfer, and Stage 1 wins. The in-process lane wins at every
length because it moves nothing, but only the process that prefilled has it.

`tests/bench_ttft.py --runs 11` reproduces the table ([raw output](../benchmarks/results/2026-10-06-mac-m4/), `ttft_r11_*.txt`). Earlier versions timed a vanilla side that computed logits at every prompt
position; the changelog has the correction.

#### Wire-protocol details (for clients implementing their own consumer)

The batched-Q form of `ATTEND.PREFIX.QUERY` (Q shape `(H, M, D)`, M derived
from blob length) returns `[output: H*M*D float32 | LSE: H*M float32]` —
the LSE trailer is required by online-softmax merge. M=1 callers see no
wire-format change (no LSE trailer). See `tests/test_attend_prefix_lse.py`
for the end-to-end verification.

For decode (M=1), the per-layer round-trip dominates. For TTFT (M large),
batched-Q sends one query block per layer instead of one per token. The
monkey-patch uses M=1 in the simple case shipped here;
TTFT-batched M>1 is the optimization that closes the small-N regime.

#### When to use Stage 1 vs Stage 2

- **Stage 1** — short prompts (<512 tok), cold-path-sensitive workloads,
  any inference engine that doesn't expose an attention hook.
- **Stage 2** — long prompts (≥1K tok), shared system prompts across many
  generations, decode-heavy workloads where K/V transfer dominates rebuild.

The two are complementary; the same prefix can live in both V-store
(Stage 1 path) and the MLX sidecar (Stage 2 path).

### Stage 2 in-process fast lane

When the cold prefill happens in the **same Python process** that serves
the warm forwards (single-process inference servers, `bench_w1_stage2.py`,
`pion-exo` running mlx-lm in-process), `PionPromptCache` keeps the per-layer
prefix K/V as MLX arrays (`self._mlx_prefix_kv[namespace]`). The patched
`pion_scaled_dot_product_attention` then concats `[prefix | suffix]` and
calls `mx.fast.scaled_dot_product_attention` directly — **zero wire
roundtrips, no `mx.eval` barrier per layer, no `numpy ↔ MLX` hop, full GPU
pipelining** across all layers in one eval.

Cross-process consumers (separate Python interpreter, container/network
boundary, vLLM CacheEngine on a different host) see no `_mlx_prefix_kv`
entry for the namespace and continue using the wire path — RESP fallback
or the binary fast lane on `port+1` (CMD_ATTEND_PREFIX_QUERY_FUSED = 0x24).

Force the wire path with `PION_PROMPT_CACHE_NO_INPROC=1` for benchmarking
or to validate cross-process behavior on a single machine.

#### Measured (`tests/bench_w1_stage2.py`, Llama-3.2-1B-Instruct-4bit, 5×20 = 100 reqs)

| Config | TTFT mean | **TTFT p50** | wire calls/req | speedup vs vanilla |
|---|---:|---:|---:|---:|
| RESP (legacy), two runs | 101.3 / 109.6 ms | 110.3 / 128.1 ms | 32 (0.65 / 0.72 ms ea) | 1.98× / 1.90× |
| Binary lane | 100.3 ms | 98.2 ms | 32 (0.63 ms ea) | 2.00× |
| **In-process fast lane** | **42.2 ms** | **34.4 ms** | **0** | **4.74×** |

Measured 2026-10-07 on an M4 Mac mini: ~316-token prefixes, vanilla ~200 ms a
request, 100% first-token agreement with vanilla mlx-lm (50/50) on every lane.
The wire lanes are forced with `PION_PROMPT_CACHE_NO_INPROC=1` (binary) and
also `PION_PROMPT_CACHE_NO_BINARY=1` (RESP). They make 32 calls a request
because mlx-lm runs the suffix in two passes (all but its last token, then the
last) and each pass queries every layer. Speedups are mean against mean. Raw
output: [`benchmarks/results/2026-10-07-mac-m4/`](../benchmarks/results/2026-10-07-mac-m4/)
(`w1_stage2*.txt`). Earlier versions of this table timed a vanilla side that
evaluated logits at every prompt position, and read higher; the changelog has
the correction. Until 2026-10-07 the harness also put a second `<bos>` before
every question; removing it moved no figure beyond run-to-run variation
([`before_fix/`](../benchmarks/results/2026-10-07-mac-m4/before_fix/)).

#### Mental model — why the wire path was paying so much

The wire path's per-layer cost on the same host is dominated by the
mid-forward `mx.eval(Q, K, V)` barrier the patched SDPA needed to
materialize tensors before sending them. That barrier drains the GPU
command queue 16 times per forward (once per layer), serializing work
that vanilla MLX would have pipelined. The in-process path removes the
barrier entirely: Q/K/V stay as lazy MLX nodes, the kernel call is
appended to the same command graph, and a single `mx.eval` at the end
of the forward drains everything in parallel.

For cross-process consumers the wire is still the right path — no
shared MLX context means no choice. The binary fast lane
is the optimization for that case (RESP framing → 0xCA5E binary on
`port+1`, single sendmsg scatter-gather: 0.63 ms a call against RESP's 0.65–0.72 ms in the table above).

### SSM.PREFIX.* — recurrent-state companion

The KV.PREFIX.* substrate above handles transformer-attention layers. For **hybrid Mamba+Transformer models** (LoLCATs, MOHAWK, Mamba-in-Llama recipes; or any future external small hybrid), the SSM half of the model has per-layer **recurrent state**, not per-token K/V. SSM.PREFIX.* is the companion substrate for that state.

```
SSM.PREFIX.STORE  <sid> <layer> <state_blob>     → +OK
SSM.PREFIX.FETCH  <sid> <layer>                  → bulk string | $-1
SSM.PREFIX.DROP   <sid> [<layer>]                → +OK (layer omitted = drop all layers for the session)
```

`state_blob` is **opaque to the server** — pure host-side byte storage. The consumer chooses serialization. The reference format (see `tests/test_ssm_prefix_roundtrip.py`):

```
uint32 version = 1
uint32 n_arrays
for each array:
    uint32 ndim
    uint32[ndim] shape
    uint32 dtype_code   (0=fp32, 1=fp16, 2=bf16 — serialized as fp32, lossless)
    raw bytes
```

This keeps the substrate **model-family-agnostic**: Mamba (size=2 `[conv_state, ssm_state]`), RWKV-7 (size=3), and future RetNet / Hedgehog / GLA all serialize differently but the server never parses. Adding a new family is ~1-2 days of consumer-side (de)serializer work.

**Validation:**
- **Drift check** (reproduced by `tests/test_ssm_prefix_roundtrip.py`): bit-perfect state hydration across Mamba-130M-f32 (2048 decode tokens), Mamba-370M-f16 (1024 decode tokens), and RWKV-7 (168M parameters, 512 decode tokens). All show **token agreement 100%, max-abs-diff = 0.000e+00** vs no-snapshot baseline. Deterministic-recurrence property holds.
- **Wire round-trip** (`tests/test_ssm_prefix_roundtrip.py`): end-to-end through pion-server. 64/64 bit-perfect Mamba-130M, 64/64 bit-perfect RWKV-7. 1 MB random blob round-trip + overwrite + multi-layer drop all OK.

**Pickup cost for new families:** each new family needs its own (de)serializer in `tests/test_ssm_prefix_roundtrip.py`'s `serialize_arrays_cache` / `deserialize_arrays_cache`. The RWKV-7 serializer is the reference implementation.

**End-to-end hybrid model.** `mlx-community/Qwen3.5-4B-MLX-4bit` (24 GatedDeltaNet + 8 Qwen3NextAttention, 3:1 ratio, Apache 2.0), a publicly released hybrid, with measured wins:

| Prefix length | Vanilla cold | Pion warm | Speedup | Token agreement |
|---:|---:|---:|:---:|:---:|
| 2,048 tokens |  4,972 ms | 355 ms | 14.0× | 18/18 |
| 4,096 tokens | 10,245 ms | 407 ms | 25.3× | 17/18 |
| 8,192 tokens | 21,328 ms | 936 ms | **29.0×** | 18/18 |

Measured 2026-10-02 on an M4 Mac mini against Pion 0.9.1, vanilla prefilled the
way mlx-lm's `generate_step` does. Each cell is the mean over three queries, and
the speedup is the mean of the per-query ratios
(`benchmarks/reproducers/sweep_qwen3_5_warm_ttft.py`, raw output in
[`stage1_qwen3_5_prefix_sweep_2026_10_02.json`](../benchmarks/reproducers/results/stage1_qwen3_5_prefix_sweep_2026_10_02.json)).
Warm TTFT grows with the bytes shipped: 118.6 MB at 2K and 320 MB at 8K, with a
33.55 MB largest layer. Token agreement misses one token in 18 at 4K,
greedy-argmax noise present on both paths. Earlier versions of this table timed
a vanilla side that computed logits at every prompt position, and read
differently; the changelog has the correction.

18/18 layers bit-perfect across the cleanly-typed split path (24 GatedDeltaNet via `SSM.PREFIX.*`, 8 Qwen3NextAttention via `KV.PREFIX.*` + `V.STOREBATCH`). `PionPromptCache` is hybrid-aware — `_classify_cache` walks the cache list and routes per-slot. Drop-in for mixed-cache models: `pc = PionPromptCache(model, vquant="fp16", port=1974); cache = pc.get_or_prefill(prefix_ids, namespace=ns)`. Reproducer: `benchmarks/reproducers/sweep_qwen3_5_warm_ttft.py`. A layer's K/V has to fit in one request (`CLIENT_BUF_SIZE`, 256 MB); the largest layer at 8K is 33.55 MB, and lengths past 8K are not measured.

### Hybrid Retrieval Cache — RAG K/V hydration

`HybridRetrievalCache` extends the prompt-prefix cache pattern from "the
system prompt that's identical across requests" to "any retrieved chunk
that's been ingested before." The retrieval is still done by whatever
embedding model the consumer already uses (BGE, MiniLM, OpenAI,
text-embedding-3-small, anything stable). Pion's role is to skip the
prefill of the retrieved chunk by holding its K/V tensors keyed by
chunk_id.

```python
from pion_vllm_mlx import HybridRetrievalCache
from mlx_lm import load

model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
hr = HybridRetrievalCache(model)        # inproc backend (default)
hr.ingest("eiffel_passage", tok.encode("The Eiffel Tower is..."))
cache, suffix = hr.prepare("eiffel_passage", tok.encode("How tall?\nAnswer:"))
# pass `cache` to mlx-lm generate — chunk K/V is already loaded
```

#### Backends

| Backend | K/V live | Precision | Server | Best for |
|---|---|---|---|---|
| `inproc` (default) | MLX arrays in a process-local dict | bit-perfect (state-setter pickling) | not required | single-process RAG |
| `pion` | `KV.PREFIX.REGISTER` + `V.STOREBATCH/V.FETCH BATCH` | fp16 (Stage 1's mean BLEU, 0.979 in the table below) | `--kvcache --metal-attention -w 1` | cross-process / cross-host |

#### Measured (Llama-3.2-1B-Instruct-4bit, 100 SQuAD v2 queries)

| Backend | p50 TTFT | vs text-RAG (113.8 ms) | Token agreement | Answer found |
|---|---:|:---:|:---:|:---:|
| inproc | 32.5 ms | **3.5×** | 96.1% | 0.73 (text-RAG: 0.72) |
| pion | 35.8 ms | 3.2× | 96.1% | 0.73 |

`benchmarks/reproducers/stage1_hybrid_recall_bench.py`, cache hydration inside the
clock; raw output in
[`stage1_hybrid_results_2026_10_07.json`](../benchmarks/reproducers/results/stage1_hybrid_results_2026_10_07.json)
(M4 Mac mini, 2026-10-07). Until then every question began with a second `<bos>`: the
2026-10-02 run read 3.0× / 2.7×, 98.3% agreement and 0.68 answers on every path. A
same-day run without the fix gives the same speedups (3.55× / 3.24×,
[`before_fix/`](../benchmarks/results/2026-10-07-mac-m4/before_fix/)), so the speed
difference is the day; the stray token cost answers (0.68) and hid divergence (99.3%).

Test: `pion-vllm-mlx/tests/test_hybrid_retrieval.py`.
First experiment: `benchmarks/reproducers/stage0_hybrid_kv_injection.py`.

#### Storage cost

INT4 K and V per token, summed over the model's layers:

| Model | K-vec dim per layer | Per-token bytes | Per 256-token chunk |
|---|---|---|---|
| Llama-3.2-1B (16 layers, 8 KV heads × 64) | 512 | ~8 KB | ~2 MB |
| Llama-3-8B-class (32 layers, 8 KV heads × 128) | 1024 | ~32 KB | ~8 MB |
| Llama-3-70B (80 layers, 8 KV heads × 128) | 1024 | ~80 KB | ~20 MB |

The hybrid pattern is worth it when the same chunks are retrieved repeatedly (FAQ, knowledge
bases, doc search); the storage blowup over a single 768-dim embedding
is amortized by the prefill saved per hit.

#### Why this is a separate API rather than a flag on `PionPromptCache`

Prefix caching's namespace contract (`make_namespace(model, tokenizer,
rope_theta, quant, adapter, prompt)`) bakes in everything that affects
the prefilled K/V. For RAG, the namespace contract is simpler:
`hash(chunk_id)` — the chunk text is the only thing that varies, the
model+tokenizer are implicit and stable. Mixing the two surfaces would
either pollute the prefix-cache namespace or hide the chunk semantics.
Separate API keeps each surface narrow and the contracts clear.

#### Multi-chunk

Top-K retrieval returns several chunks. `set_shared_stub()` registers a shared
prefix stub once, `ingest_pack()` stores each chunk pack, and `prepare_multi()`
composes up to `max_packs` (default 8) packs with exact positional re-rotation.
Encode each chunk and the suffix separately and concatenate *tokens*, not
strings. Keep the composition coarse: many small packs let distractor chunks
collide, and quality drops well before 20 packs.

---

## Quantization Formats

Measured on a 20-question / 50-token-greedy BLEU eval against the standalone reference:

| Format | Storage vs FP16 | Mean BLEU | First-token | Notes |
|---|---:|---:|:---:|---|
| **fp16** | 1.00× | **0.979** | 100% (20/20) | Bit-identical on 19/20, late drift on 1/20. **Production default.** |
| int8 | ~2× | 0.499 | 95% (19/20) | First token right on 19 of 20, but greedy decode drifts within the 50 tokens. |
| turbo4 | 3.51× | 0.378 | 90% (18/20) | Argmax preserved on most first tokens, but compounds badly over greedy decode. **Single-step / classification only.** |

`tests/test_kv_prefix_bleu.py --vquant {fp16,int8,turbo4}` on an M4 Mac mini, 2026-10-07
([raw output](../benchmarks/results/2026-10-07-mac-m4/), `kv_prefix_bleu*.txt`; the
test's own gate is a mean BLEU of 0.95, which only fp16 passes). Until 2026-10-07 each
question began with a second `<bos>`, and the table read 0.969 / 0.677 / 0.538: the stray
token drew attention away from the stored prefix and hid part of the quantization error
([`before_fix/`](../benchmarks/results/2026-10-07-mac-m4/before_fix/) reproduces those
figures exactly). Storage is per
token at the 1B model's 512-wide K/V rows: int8 is one byte an element against
fp16's two, and turbo4 packs 32 elements into 18 bytes plus a 4-byte row header.

**Recommendation:** ship `vquant=fp16`. Document `int8` and `turbo4` as opt-ins for greedy-tolerant single-step workloads (function-calling tool selection, classification, single-token routing) where the storage win matters more than multi-token fidelity.

---

## Use Cases

Concrete fits for this build (single-instance, MLX, Apple Silicon, a high hit rate, prefix-dominated):

1. **Multi-tenant SaaS with a fixed system prompt.** The Stage-1 workload measurement at the top of this page (5 prompts × 30 queries, 96.7% hit rate, 9.21× mean TTFT) is exactly this shape.
2. **Local LLM apps on Apple Silicon (Mac/iOS).** An embedded engine; chat with a reused system prompt.
3. **Mac cluster inference (exo, vllm-mlx).** Cross-instance verified — multiple Macs share one Pion via TCP.
4. **RAG with a fixed document corpus, contiguous order.** Cache `[system + chunks_in_canonical_order]`. Arbitrary chunk recomposition is **not** safe (causal attention — chunk B's K is rotated for positions it will not occupy).
5. **Few-shot prompts, function-calling, agent loops.** Fixed system + tools + examples; user message varies. First-token agreement is what matters most.
6. **Code assistants with repo context.** 6K-token repo prefix + short user query. Win grows with prefix length.
7. **Prompt-engineering iteration.** 50+ test queries against one prompt; warm cycle dominates.
8. **A/B testing / replay harnesses.** Namespace key ensures fresh cache when any input changes.

---

## Limitations

- **No shared prefix, no win.** Long unique conversations per user need a different optimization.
- **Prefill, not decode.** Tokens-per-second after the first token is unchanged.
- **`turbo4` drifts over multi-token greedy decode.** Use `fp16` (or `mlx4g32`) for generation; `turbo4`/`int8` suit single-step work.
- **Namespace keys are not credentials.** Use `--requirepass` or one process per tenant (`doc/multi_tenant.md`) where tenants must not read each other's cache.
- **Over TCP it will not match RDMA.** GPU clusters with RDMA fabric are better served by an RDMA-native KV store.

### Multi-worker deployment

`pion-server --kvcache -w N` works directly. The cross-worker session
directory makes `KV.PREFIX.LOOKUP` answer correctly regardless of which
worker the connection lands on. V buffers stay per-worker; non-owner
`V.STOREBATCH`/`V.FETCH` return `-ERR session lives on worker N` and
clients reconnect. `pion-lmcache.PionStore` handles this transparently
via `auto_redirect=True` (default).

```bash
./pion-server --kvcache -w 4 --independent-workers   # one Pion, four workers
```

```python
from pion_lmcache import PionStore
with PionStore(port=1974, vquant="fp16") as s:
    s.register("my_app|prompt_a", kv_dim=128)
    s.store_layer("my_app|prompt_a", "V", layer=0,
                  token_offset=0, tensor_fp32=v)
    # auto_redirect transparently handles cross-worker -ERR
```

For multi-tenant isolation, deploy one Pion per tenant; see
`doc/multi_tenant.md` and `pion-lmcache/MULTITENANT.md`.

The legacy "two Pion instances" pattern (`-w N` + `--kvcache -w 1`) still
works for callers that prefer a hard split between vector/KV traffic and
KV-cache traffic — it's no longer required.

---

## Tests

### Stage 1 (cache-rebuild path)

| Test | What it gates |
|---|---|
| `tests/test_kv_prefix_prototype.py` | Single-prompt correctness against HF gpt2 (pre-MLX prototype) |
| `tests/test_kv_prefix_mlx.py` | Single-prompt correctness against Llama-3.2-1B MLX, 5 quant configs |
| `tests/test_kv_prefix_workload.py` | Multi-query workload, lower-level (V.STOREBATCH/V.FETCH RANGE direct) |
| `tests/test_kv_prefix_bleu.py` | BLEU acceptance — 20 queries × 50-token greedy completions vs standalone |
| `tests/test_kv_prefix_cross_instance.py` | Two-client cross-instance verification — proves "shared", not just "cached" |
| `tests/test_kv_prefix_lru.py`, `tests/test_kv_prefix_admission.py` | LRU eviction + 2-hit admission policy |
| `pion-vllm-mlx/tests/test_prompt_cache_workload.py` | Public API test: the headline workload through `PionPromptCache` |

### Stage 2 (mlx-lm Attention monkey-patch)

| Test | What it gates |
|---|---|
| `tests/test_attend_prefix.py` | Stage 2: STORE_KV + QUERY_CACHED basic correctness |
| `tests/test_attend_prefix_batched.py` | Batched-Q `(H, M, D)` cosine 1.0 vs CPU softmax; M=1 backward-compat |
| `tests/test_attend_prefix_lse.py` | LSE trailer end-to-end; merge of two attention halves matches reference |
| `tests/test_attend_prefix_merge.py` | Online softmax merge math (numpy-only proof) |
| `tests/test_mlx_lm_patch.py` | Monkey-patch correctness — Llama-3.2-1B: the first token equal to vanilla's and at least half of 20 greedy tokens (it prints the agreement) |
| `tests/bench_ttft.py` | TTFT A/B/C — Stage-2 wins quantified |
| `tests/bench_bleu_3b_stage2.py` | 3B BLEU comparison — Stage-2 vs cache-rebuild |

### Cross-cutting infrastructure

| Test | What it gates |
|---|---|
| `tests/test_vstore_wal.py` | WAL replay across SIGKILL — bit-equal `V.FETCH` |
| `tests/test_kvprefix_xworker.py` | Cross-worker session directory + cross-worker -ERR contract |
| `tests/test_kvprefix_autoredirect.py` | `PionStore.auto_redirect` handles -ERR transparently |
| `tests/test_multitenant.py` | `--ns-prefix` enforcement matrix |
| `tests/test_dropindex_grace.py` | DROP+REBUILD cycle stress with grace period |
| `tests/bench_g2_strict.py` | Strict numerics — memory savings + MLX-vs-CPU |
| `pion-lmcache/tests/test_pion_lmcache.py` | PionStore / LMCacheRemoteBackend integration (5/5) |
| `tests/test_lmcache_resp_connector.py` | LMCache RESPConnector wire-pattern emulation |

To reproduce the headlines:

**Stage 1 (cache-rebuild) through the public API — 8.92× mean TTFT at 1B**
([raw output](../benchmarks/results/2026-10-07-mac-m4/prompt_cache_workload_q30_r8.txt)):

```bash
./pion-server --kvcache -w 1 &
python3 pion-vllm-mlx/tests/test_prompt_cache_workload.py \
    --vquant fp16 --prompts 5 --queries 30 --prompt-repeats 8   # with pion-vllm-mlx[mlx] installed
# Measured 2026-10-06, M4 Mac mini: mean TTFT 576 → 66 ms (8.77×), throughput 8.73×,
# hit rate 96.7%, first-token agreement 100%
```

**Stage 2 (mlx-lm monkey-patch): both lanes against vanilla and Stage 1:**

```bash
./pion-server --kvcache --metal-attention -w 1 &  # Metal handles both decode (M=1) and batched-Q (M>1) TTFT
python3 tests/bench_ttft.py --prompt-tokens 2048 --runs 11   # with pion-vllm-mlx[mlx] installed
# Measured 2026-10-06, Llama-3.2-1B/2K, M4 Mac mini (warm TTFT median, one-token suffix):
#   Path A vanilla cold:                1171 ms
#   Path B cache-rebuild (Stage 1):       37 ms  (31×)
#   Path C Stage 2, wire lane:            24 ms  (49×)
#   Path D Stage 2, in-process lane:      11 ms  (104×)
```

**Cross-worker (`-w 4`) without auto-cap:**

```bash
./pion-server --kvcache -w 4 --independent-workers &
python3 tests/test_kvprefix_xworker.py
# Expected: REGISTER on one connection, all 12 fresh-connection LOOKUPs HIT
```

---

## Related Docs

- `doc/ai_gateway.md` — `ATTEND.*` and `KV.STORE/FETCH/INFO` (the attention index, distinct from KV.PREFIX).
