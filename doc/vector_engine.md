# Vector Engine: High-Performance Search

**Pion** is a unified, shared-nothing database engine built in Mojo. Its Vector Engine provides real-time approximate nearest-neighbor (ANN) search with Redis Stack-compatible `FT.*` commands, enabling drop-in use with any client that targets Redis as a vector store (VectorDBBench, LangChain, LlamaIndex, etc.).

---

## Architecture Overview

```
Client (VectorDBBench / redis-py / any RESP client)
        │ FT.CREATE / HSET / FT.SEARCH
        ▼
  slow_path.mojo  ←  FT.* dispatcher
  fast_path.mojo  ←  multi-field HSET router
        │
        ├─ keyspace (SlabHashMap)  — stores HASH docs (id, metadata, …)
        └─ HNSWGraph               — stores quantized vectors + graph
```

Each worker owns a private `HNSWGraph` (shared-nothing). A single shared listen fd distributes connections via `accept()` scheduling (no `SO_REUSEPORT`). After `FT.OPTIMIZE`, the building worker publishes read-only index pointers via `SharedHNSWView`; other workers lazy-borrow these pointers on the first `FT.SEARCH`. All mutable search state (`visited_map`, `query_int8`) remains per-worker.

---

## HNSW Index (`src/vector/hnsw.mojo`)

The core index is an HNSW (Hierarchical Navigable Small Worlds) graph.

### Graph structure
- **Level 0** contains all vectors; higher levels contain exponentially fewer nodes (controlled by `M` and `ml = 1/ln(M)`).
- Each node stores a quantized `Int8` vector + neighbor lists (capacity `2*M` at level 0, `M` at higher levels) allocated from a pre-allocated neighbor pool.
- `node_map[id] → internal_idx` allows O(1) lookup by user-supplied integer ID.

### Insert (`add_vector`)
1. Assign a random level (`_random_level`).
2. Quantize the `Float32` input vector to `Int8` via calibrated SQ8.
3. Greedy descent from `max_level` down to `level+1` to find best entry point.
4. For each level from `min(level, max_level)` down to 0: beam-search (`_search_layer`, ef = `ef_construction`), prune neighbors (heuristic), link bidirectionally.

### Search (`search_fp32`, `search_fp32_scored`)
1. **Upper levels**: greedy traversal using `_dist_fp32_int8` (query FP32 × stored Int8).
2. **Level 0**: beam search with `MinHeap` (candidates) + `MaxHeap` (results), ef-bounded.
   - Query is quantized to INT8 once; beam uses `_dist_int8_int8` batch kernels (batch-8 and batch-4) — matches the graph build metric, no per-neighbor dequantization.
   - **Slot-space addressing**: the gather loop reads each neighbor's compact slot
     from `l0_slots` (a 33-u32/node mirror of `l0_compact`) and computes the vector
     address as `compact_buffer + slot*compact_stride + compact_hdr`, so no
     `nodes[]` dereference sits on the critical path. `0xFFFFFFFF` entries fall
     back to the `nodes[]` path. Applies to the INT8 beam and all three quant beams.
   - **Staged prefetch (INT8 beam)**: gather issues 4 lines (norm header + 256B
     prefix) per neighbor; the full slot prefetch is issued one SDOT batch ahead
     inside the scoring loops.
3. `search_fp32_scored` additionally fills a caller-supplied `List[Float32]` with L2 distances in nearest-first order — used by `FT.SEARCH` to return scores.

### Visited-set optimisation
Epoch-stamped visited set: `visited_epoch[UInt16]` per node with a monotonically
increasing `cur_epoch` — a node is visited this query iff `visited_epoch[nid] == cur_epoch`,
so there is no per-query memset (full clear only on `UInt16` wrap).

### Key fields added for FT.* integration

| Field | Type | Purpose |
|---|---|---|
| `vector_field_name` | `InlineArray[UInt8, 32]` | Schema field name holding vectors (set by FT.CREATE) |
| `vector_field_len` | `Int` | Byte length of that name |
| `index_name` | `InlineArray[UInt8, 64]` | Index name captured from FT.CREATE argument |
| `index_name_len` | `Int` | Byte length of stored index name |
| `index_ready` | `Bool` | True after FT.CREATE; gates vector routing in HSET |
| `ef_runtime` | `Int` | Default ef for FT.SEARCH when no per-query override (default 150) |

---

## Calibrated Scalar Quantization — SQ8

All stored vectors are quantized from `Float32` → `Int8` at insert time.

- **Range**: calibrated from the vectors being built, by every build (`HNSWGraph._calibrate`): one global range over mean ± 8σ (Welford). The HSET-ingest build calibrates on up to the first 65,536 ingested vectors, streaming mode on the first 1,000. 8σ is wide enough that the few near-constant outlier dimensions typical of embedding data are clipped (they cancel in every L2 difference) while every component that really varies is kept.
- **Formula**: `q = Int8(clamp((v - min) / range * 254) - 127)`
- **Memory**: 4× compression vs FP32 (1536 dims → 1536 bytes instead of 6144).
- **FP32 buffer ingest**: `add_vector()` buffers raw FP32 vectors; graph is built in batch on `FT.OPTIMIZE` call (6× faster insert — no graph ops during HSET).
- **Compact buffer**: After FT.OPTIMIZE, `compact_vectors()` reorders INT8 vectors in BFS traversal order into a contiguous buffer for cache-friendly beam search. Each build (or snapshot load) also derives `l0_slots` — the slot-space adjacency mirror the beam kernel addresses vectors through — plus `compact_stride`/`compact_hdr` (INT8: `dim+8`/8; quant variants: their own stride/0).

---

## SIMD & Compile-Time-Fused Kernels (`src/vector/kernels.mojo`)

Distance computation is the hottest inner loop. Pion uses Mojo's `comptime` parameters to generate fully-unrolled, fused kernels for common dimensions:

```mojo
fn l2_distance_fp32_int8_fused_jit[dim: Int](
    v1: UnsafePointer[Float32],
    v2: UnsafePointer[Int8],
    min_val: Float32,
    range_val: Float32
) -> Float32:
    # Loop unrolled at compile time for dim ∈ {128, 384, 768, 1536}
    # Dequantization fused into the accumulation: no separate pass
```

Specialised variants exist for `dim ∈ {128, 384, 768, 1536}` — the four most common embedding dimensions. Unknown dimensions fall back to a scalar loop.

Additional kernel variants:
- `l2_distance_int8_int8_batch8_jit[dim]` — INT8×INT8 batch-8, primary search kernel
- `l2_distance_int8_int8_batch4_jit[dim]` — INT8×INT8 batch-4, remainder loop
- `l2_distance_int8_jit[dim]` — scalar INT8×INT8, build-time and fallback
- `l2_distance_int4` — 4-bit packed (when `use_int4=True`)
- `hamming_distance_jit[dim_u64]` — binary quantization (when `use_bq_traversal=True`)

Software prefetching (`sys.intrinsics.prefetch`) is applied to the next neighbor's vector and neighbor-list pointers inside `_search_layer` and `add_vector`.

---

## Wire Protocol — FT.* Commands

Pion is wire-compatible with Redis Stack's vector search subset. Commands are dispatched in `slow_path.mojo` by matching `tl` (token length) and the first bytes of `tp` (token pointer).

### FT.CREATE
```
FT.CREATE <index> ON HASH SCHEMA … <field> VECTOR HNSW 6
    TYPE FLOAT32 DIM <d> DISTANCE_METRIC L2 M <m> EF_CONSTRUCTION <ef>
```
- Captures `<index>` (first token after the command) as `hnsw.index_name`.
- Scans remaining tokens for the `VECTOR` keyword; captures the immediately preceding token as `vector_field_name`.
- Scans for `EF_CONSTRUCTION` keyword; parses the following integer as `hnsw.ef_construction`.
- **`DISTANCE_METRIC`** — resolved in a pre-scan, *before* any index
  state is written, so refusing one is a true no-op. `L2` → `distance_metric =
  0`, `COSINE` → `1`, anything else → `-ERR unsupported DISTANCE_METRIC`.
  Omitting the keyword gives `L2`.

  The search itself is unchanged either way: the beam kernels compute
  `query_norm + node_norm - 2·dot`, i.e. **squared L2 over affinely-quantized
  codes**, so L2 is what this engine has always answered. `COSINE` is
  implemented by L2-normalizing the FP32 vector at ingest *and* the query at
  quantization time — on unit vectors, L2 ordering **is** cosine ordering.

  It is done there and not in the distance function because the quantizer is
  affine **with an offset**: an offset cancels in a difference but not in a dot
  product, so a cosine computed from quantized dots would be wrong. The metric
  is persisted in index-file header word 24 (a warm restart that forgot it
  would query a normalized graph with a raw query) and published to
  `SharedHNSWView.pre_distance_metric` so a cross-worker HSET normalizes the
  same way the querying worker will.

- Sets `hnsw.index_ready = True`.
- Subsequent multi-field HSET calls will route the matching field to the HNSW index.

### HSET (multi-field, fast path)
```
HSET <key> <f1> <v1> <f2> <v2> … <vec_field> <float32_bytes>
```
- Handled in `fast_path.mojo` when `num_args >= 6` and `(num_args - 2) % 2 == 0`.
- Each field is stored in the HASH in keyspace.
- If a field's name matches `vector_field_name` **and** its byte length equals `hnsw.dim * 4`, it is routed to `hnsw.add_vector(key_id, fp32_ptr)`.
- Non-vector fields (id, metadata, …) are stored normally in the hash.
- Returns `:N\r\n` where N = number of fields written.

### FT.SEARCH
```
FT.SEARCH <index> "*=>[KNN <k> @<field> $blob EF_RUNTIME <ef> AS score]"
    PARAMS 2 blob <float32_bytes> [DIALECT 2] [SORTBY score] [LIMIT 0 <k>]
```
- Verifies `<index>` matches `hnsw.index_name` (case-sensitive byte comparison; skipped if no index has been created).
- Extracts `k` by scanning the query string for `KNN `.
- Extracts `EF_RUNTIME <ef>` from the query string if present; falls back to `hnsw.ef_runtime` (default 150).
- Extracts `LIMIT offset count` from the outer token stream; if `count < k`, uses `count` as the result cap.
- Finds the query vector blob in the PARAMS section by matching its byte size to `hnsw.dim * 4` (size-based routing — no name coupling between PARAMS and schema).
- Calls `hnsw.search_fp32_scored(blob_ptr, k, dist_scores, ef_query)` with the resolved ef.
- Returns a RESP2 FT.SEARCH-format response:

```
*(1 + 2*N)\r\n  :N\r\n
[for each result:
  $<id_len>\r\n<id_str>\r\n   ← doc key
  *4\r\n
  $2\r\nid\r\n  $<id_len>\r\n<id_str>\r\n
  $5\r\nscore\r\n  $<len>\r\n<L2_distance>\r\n
]
```

This shape is emitted at every worker count. Doc-key resolution order:
shared `hk_keys_buf` slot→key map (one direct load; written by the fast-path
vector HSET), then the keyspace `__hk__<slot>` probe (keys >31B that don't fit
the shared map's 32-byte slots), then the numeric slot id. The `id` **value**
is the doc hash's own `id` field when the doc hash is resolvable (what
`RETURN 1 id` consumers and `tests/test_multiworker_recall.py` parse), else
the doc key, else the slot number.

### FT.INFO
```
FT.INFO <index>
```
Returns a 4-element array (`index_name`, `<actual_name>`, `num_docs`, `<count>`) when `index_ready`, where `<actual_name>` is the exact index name stored at FT.CREATE time and `<count>` is `hnsw.num_nodes`. Returns `-ERR Unknown index name` when no index has been created.

### FT.OPTIMIZE
```
FT.OPTIMIZE <index>
```
Triggers batch graph construction from the FP32 ingest buffer. Calls `hnsw.build_index_from_shared()` (if shared buffer has data) or `hnsw.build_index()` (local). After build: `compact_vectors()` reorders vectors in BFS order, `publish_to_shared()` makes the index available to other workers via `SharedHNSWView`. Returns `+OK`.

**Ingest → optimize → search is a one-way contract.** Every `HSET` must land *before* `FT.OPTIMIZE`; the FP32 ingest buffer is freed after the build, so an `HSET` issued *after* `FT.OPTIMIZE` is stored as a hash but **not indexed** (no error, and `FT.SEARCH` will not find it). For incremental inserts, `FT.DROPINDEX` → re-ingest everything → `FT.OPTIMIZE` again. Re-issuing `FT.CREATE` for the index being served changes nothing, and does not reopen ingest — unless that index is empty (built with no documents), which has nothing to protect: then ingest opens again. `FT.OPTIMIZE` with no index defined (no `FT.CREATE`, nothing ingested, no graph) answers `Unknown index name`; it used to build and serve an empty index named by its argument, under which the real index's `FT.CREATE` would not open ingest.

**Deletions do follow the keyspace (#46).** A document leaves the results when its hash leaves the keyspace, by DEL, UNLINK, expiry, FLUSHALL, or a write that replaces the key. It also leaves when the vector leaves the document: an `HSET` of a new vector, an `HDEL` of the field, or a field TTL. Writing the vector it already has (the same bytes: an ingest script run twice) keeps it indexed. A renamed document leaves too: its slot names the old key, and after the build the new name is not indexed. Before the build, a renamed, copied (`COPY`) or restored (`RESTORE`) hash is ingested under its new name, as an `HSET` would be.

Mechanics:
- The hash records its slot. The slot dies with the hash, or with its vector field.
- A dead slot is one byte in an array all workers share, so a search on any worker skips it at once.
- The graph is not changed. Dead nodes still route the beam, as HNSW soft deletes do, and `FT.SEARCH` filters them out of the results. The KNN path widens its candidates until it has k live documents.
- Each worker records the slots it killed as members `<build id>:<slot>` of an internal set, `__hk_dead__`. The WAL, snapshots and replication carry it like any set.
- A restart applies the members whose build id matches the loaded index. Each `FT.OPTIMIZE` draws a new build id, saved in index header word 31.

### FT.DROPINDEX
```
FT.DROPINDEX <index>
```
Calls `hnsw.reset_index()` — zeroes `num_nodes`, resets `entry_point_id = -1`, clears `node_map` and `visited_map`, resets `index_name_len = 0`, resets `global_min/max` to calibration defaults, sets `index_ready = False`. Returns `+OK`.

### FT.HYBRID — Hybrid Vector + BM25 Search
```
FT.HYBRID <index> <text_query> <vector_blob> [K <k>] [ALPHA <a>]
```
Executes parallel BM25 full-text search and HNSW vector search in a single command, fusing results via Reciprocal Rank Fusion (RRF).

**Parameters:**
- `<index>` — index name (must match a previously created FT.CREATE index)
- `<text_query>` — plain-text query string for BM25 scoring
- `<vector_blob>` — raw Float32 bytes (`dim * 4` bytes) for HNSW nearest-neighbor search
- `K <k>` — number of results to return (default 10)
- `ALPHA <a>` — blending weight: 0.0 = full vector, 1.0 = full BM25, 0.5 = equal (default 0.5)

**Fusion formula (RRF):**
```
score = vec_weight / (60 + rank_vec) + bm25_weight / (60 + rank_bm25)
```
where `vec_weight = 1 - alpha` and `bm25_weight = alpha`. The constant 60 is the standard RRF damping factor.

**Internal behavior:**
1. HNSW search with `K * 4` oversampling (retrieves 4x candidates for quality)
2. BM25 search with `K * 4` oversampling
3. RRF fusion: candidates from both result sets are merged by document ID, scored, and sorted
4. Top-K results returned

**Response format:** Standard FT.SEARCH RESP2 format with RRF combined scores (same wire format as FT.SEARCH).

**Example:**
```bash
redis-cli FT.HYBRID myindex "machine learning transformers" \
    $(python3 -c "import struct; print(struct.pack('1536f', *[0.01]*1536))") \
    K 10 ALPHA 0.5
```

See also: `doc/embeddings.md` for text embedding generation.

---

## BM25 Full-Text Search

Pion includes a built-in BM25 full-text search engine, constructed during `FT.OPTIMIZE` from the TEXT schema field declared in `FT.CREATE`. The schema's **first** TEXT field is the one indexed.

### Ingest — two paths, both work

The doc set is `union(HNSW nodes, FT.ADDTEXT registrations)`, so a corpus needs no vectors at all to be lexically searchable.

```
FT.CREATE idx SCHEMA body TEXT vec VECTOR HNSW 6 TYPE FLOAT32 DIM 512 DISTANCE_METRIC L2

# (a) text-only — no vector, no embedding sidecar needed
FT.ADDTEXT idx 7 "the --fa-window flag caps the sliding attention span"

# (b) alongside a vector — any key shape; the __hk__<slot> reverse map
#     written by the vector HSET is what points BM25 at the text
HSET doc:7 vec <512 float32 LE>
HSET doc:7 body "the --fa-window flag caps the sliding attention span"

FT.OPTIMIZE idx                # builds the inverted index — required after ingest
FT.SEARCH idx BM25 "sliding attention" K 10
```

**`FT.ADDTEXT` doc ids must be non-negative integers to be BM25-indexable** — hits are returned as integer ext_ids, so a string id has nothing to map back to. A string id still works for `FT.SEARCHTEXT`, which searches the semantic-cache HNSW rather than the inverted index.

`FT.SEARCHTEXT` runs against that **separate, per-worker semantic-cache graph**, and it comes with limits the BM25 side does not have: 10,000 documents per worker, no clearing on `FT.DROPINDEX`, and O(N²) bulk ingest. See `doc/embeddings.md` § "FT.ADDTEXT / FT.SEARCHTEXT limits and gotchas" before building a text knowledge store on it.

`FT.SEARCH … BM25` against an index with **no inverted index at all** returns an error naming the missing step, not an empty array — a caller could not otherwise tell "never optimized" from "matched nothing". A built index that genuinely matches nothing still returns the empty form.

BM25 state is **per-worker and not persisted**: it is never published to the shared view and `save_to_disk` does not carry it, so re-run `FT.OPTIMIZE` after a restart, and prefer `-w 1` for lexical workloads.

### One index per server

A Pion server holds **one index at a time**. `FT.CREATE` overwrites the shared schema, `FT.OPTIMIZE` rebuilds the single HNSW graph and the single set of BM25 postings, and `pion.hnsw.0` is written by whichever index optimized last. Running `FT.OPTIMIZE` on index B therefore *replaces* whatever was serving index A — including a concurrently-serving production index.

This is the contract, not a bug, and a query naming the displaced index gets an error rather than `*0`, which would be indistinguishable from a genuine miss:

```
FT.SEARCH clobA BM25 "alpha bravo"
-ERR index 'clobA' has no BM25 postings - this server holds one index at a time and the
     current postings belong to 'clobB' (a later FT.OPTIMIZE replaced them); re-ingest
     and FT.OPTIMIZE 'clobA' to serve it again

FT.SEARCH clobA *=>[KNN 2 @vec $q] PARAMS 2 q <blob>
-ERR index 'clobA' is not the index loaded on this server - it holds one index at a
     time and currently serves 'clobB' (a later FT.CREATE/FT.OPTIMIZE replaced it)
```

`FT.HYBRID` refuses the same way rather than quietly degrading to a vector-only result scored against another index's postings. Index names are compared **byte-exactly** (case-sensitive), matching the pre-existing KNN check. The `BM25 [<index>]: N terms, M docs` log line names its index so a cross-index rebuild is identifiable after the fact.

Recovery is what the error says: re-ingest the displaced index's documents and re-run `FT.OPTIMIZE` for it. Ownership is recorded at **build** time, so `FT.CREATE B` alone does not displace A's postings — only `FT.OPTIMIZE` does.

**Operational consequence:** do not point a scratch/probe index at a server that is serving a real one. A 2-document probe's `FT.OPTIMIZE` replaces a 300-passage index.

### Indexing
- **Built automatically** by `FT.OPTIMIZE`, from scratch each time
- **Tokenization:** hard-split on whitespace and structural punctuation (`" ' \` ( ) [ ] { } < > | \ , ; : ! ?`); the remaining punctuation (`- . _ + = * / & % @ # ~ $ ^`) is trimmed from token edges but **kept inside** a token, so `--fa-window` indexes as `fa-window` rather than `fa` + `window`. Tokens are lowercased and hashed with FNV-1a. Index-time and query-time tokenization go through one function and cannot drift.
- **No caps.** Vocabulary and per-document term count grow on demand, and token length is unbounded.

### Scoring
- **BM25** with defaults `k1 = 1.2`, `b = 0.75`, tunable **per query** (see below)
- **IDF:** `log(1 + (N - df + 0.5) / (df + 0.5))` where N = documents in the BM25 doc set, df = document frequency. This is Lucene's non-negative form; `rank_bm25`'s `BM25Okapi` instead uses `log(N - df + 0.5) - log(df + 0.5)` with an epsilon floor for negative values. The difference is measurable but small (see below).
- **Lookup:** term hashes are stored sorted; query terms resolve by binary search, O(log V) per term. Up to 64 unique query terms are scored.

### Querying
```
FT.SEARCH <index> BM25 "query text" [K <k>] [K1 <f>] [B <f>]
FT.HYBRID <index> "query text" <vector blob> [K <k>] [ALPHA <a>] [K1 <f>] [B <f>]
```

`K1` (0–100) sets term-frequency saturation; `B` (0–1) sets length normalisation, where `B 0` disables it entirely. Out-of-range values are rejected rather than silently clamped. Omitting both reproduces the documented defaults exactly.

---

## Metadata Filters

`FT.SEARCH` filters a KNN query by TAG and NUMERIC fields declared in the `FT.CREATE` schema. Every form below is applied; anything else is **refused with an error**, never ignored.

```
FT.SEARCH idx "*=>[KNN 10 @vec $v]"               FILTER @price:[10 50]   PARAMS 2 v <blob> DIALECT 2
FT.SEARCH idx "(@price:[10 50] @cat:{a|b})=>[KNN 10 @vec $v]"             PARAMS 2 v <blob> DIALECT 2
FT.SEARCH idx "*=>[KNN 10 @vec $v]"               FILTER price 10 50      PARAMS 2 v <blob>   # RediSearch legacy
FT.SEARCH idx "*=>[KNN 10 @vec $v]"               FILTER cat=a            PARAMS 2 v <blob>
FT.SEARCH idx "*=>[KNN 10 @vec $v]"               FILTER price [10 50]    PARAMS 2 v <blob>
```

- **NUMERIC** `@f:[lo hi]` is inclusive; `-inf` / `+inf` work. Exclusive `(` bounds are refused.
- **TAG** `@f:{a}` matches the value exactly; `@f:{a|b}` matches either.
- Up to **4** clauses, ANDed — several `FILTER` arguments and/or clauses in the query prefix. `|` between clauses (OR) and negation are refused.
- Filtering is applied to KNN candidates, widening the candidate set (4×, 16×, … up to 16,384 or the whole index) until `k` rows pass — a selective filter costs more search, it does not silently return fewer rows while matches exist below that bound.
- Field names are case-sensitive, as in the hash.

The query itself must be `<prefilter>=>[KNN k @field $param [EF_RUNTIME n] [AS alias]]` with the vector in `PARAMS` by name; an unknown form, an unknown vector field, a missing `$param` or a blob of the wrong size is an error.

**Scores** are the index metric's distance for each returned row — squared L2 under `L2`, `1 - cos` under `COSINE` — computed from the stored vectors (FP32 re-rank copy, or the INT8 codes dequantized), not the beam's internal quantized distance. Rows are sorted by that score.

---

## Open build and the closed vector library

The tuned 1536-dim beam searches — INT8 and the three quantized variants —
and the tuned product-key-memory kernels are distributed as a static library,
`libpion_vector`, vendored under `vendor/pion-vector/<platform>/`. Everything
else is open source here: graph build, persistence, the other dimensions,
every quantizer and dequantizer, the Metal shaders, and this document.

| Build | Vector code | Use |
|---|---|---|
| `pixi run build` (macOS arm64, Linux arm64), `build-portable` (Linux x86-64) | links `libpion_vector.a` (`-D PION_HELD_VECTOR`) | default; what releases and the Docker image ship |
| `pixi run build-open` (all three platforms) | `src/vector/reference/` | auditors, ports, anyone who will not run a binary they cannot read |
| any other build (iOS, a bare `mojo build`) | `src/vector/reference/` | — |

**The references are the same algorithms with the tuning removed** — same
traversal, same heaps, same per-lane float formula — and return bit-identical
results. What they leave out is prefetch staging, batch-of-8 scoring and the
ISA dot instructions (SDOT / VNNI). Two checks hold that to be true:
`pixi run test-vector-differential` feeds every closed routine and its
reference the same synthetic state (0 differences required, PKM FP32 scores
within 1e-4 because the tuned kernels sum in a different order), and
`tests/test_vector_build_equivalence.py --binaries <lib> <open>` warm-loads
one real index into both builds and requires identical keys and scores.

**What the open build costs** (gate config: Performance1536D50K, ef=150,
`-w 10`, Mac M4, 3 interleaved ABBA pairs each, 2026-09-25):

| Search | `build` QPS | `build-open` QPS | Open build | Recall@100 (both) |
|---|:---:|:---:|:---:|:---:|
| INT8 (default) | 8,822 | 6,044 | **−30.0%** (3/3 pairs: −30.0, −28.7, −31.5) | 0.958 |
| PolarQuant (INT4) | 6,789 | 4,491 | **−33.8%** (−33.8, −31.0, −35.4) | 0.965 |
| TurboQuant (INT3 + QJL) | 5,035 | 3,776 | **−22.3%** (−26.3, −20.6, −22.3) | 0.95 |
| NanoQuant (INT2) | 3,606 | 2,818 | **−21.9%** (−24.9, −21.9, +2.3) | 0.46 |

QPS is the median lib/open pair. Recall is the same because the results are
the same; per-run recall differs only through HNSW build nondeterminism.
Product-key memory (`NEURON.PKM.*`, 1M slots) was measured the same way on
2026-09-25: server compute +39% exact, +102% with 8 heads, single-query wire
latency +12%. **Nothing else changes between the two builds** — KV
throughput, prompt-cache TTFT and every other headline number come from open
code.

**On Linux x86-64** (INT8, same gate config, 2026-10-03). The machine was a
Ryzen 9 9950X (Zen 5), rented with no other tenant and run with `--epoll`.
The open build was the v0.9.2 release binary and the library build v0.9.3,
which share a toolchain and engine source. Three rounds ran, with the order
rotated:

| Search | Median QPS (3 runs) | Open build (5,836 QPS) | Recall@100 (all) |
|---|:---:|:---:|:---:|
| `libpion_vector`, x86-64-v2 build | 7,610 | **−23.3%** (rounds: −24.4, −14.9, −14.3) | 0.960 |
| `libpion_vector`, VNNI build | 8,534 | **−31.6%** (rounds: −32.6, −14.3, −32.8) | 0.960 |

The VNNI build passed the differential on this machine, with 0 differences
against the reference compiled for `icelake-server`. Its lead over the
x86-64-v2 build there (+12% on the medians, one round tied) did not settle
which build should be the default. A second machine did.

**The VNNI build is the default on CPUs that have it** (since 2026-10-05;
`PION_VECTOR_VNNI=0` forces the x86-64-v2 build). The deciding run was on an
EPYC 8124P (Zen 4c, bare metal, `-w 16`, io_uring), where the VNNI build again
passed the differential. Both arms used one binary built for that CPU, with
the library choosing its build at startup, over six rounds with alternating
order. VNNI won every round:

| | x86-64-v2 build | VNNI build | Median of the per-round changes |
|---|:---:|:---:|:---:|
| Server CPU per query | 939 µs | 623 µs | **−32%** (6 of 6) |
| QPS, 8 clients each pinned to its own worker | 4,578 | 5,601 | **+22%** (6 of 6) |
| Peak QPS (VectorDBBench, C=10) | 3,881 | 4,710 | **+20%** (6 of 6) |
| Recall@100 | 0.960 | 0.960 | equal within 0.001 |

The decision rule was written before the runs. Intel parts with AVX-512 VNNI
(Ice Lake-SP, Sapphire Rapids) have not been measured. Linux arm64 has not
been measured.

The interface is open too: `src/vector/vector_abi.mojo` (the calls),
`beam_view.mojo` and `quant_beam_view.mojo` (the argument structs, whose
field order is ABI). `INFO` reports `pion_vector:` and `--version` a
`vector:` line, both asked of the linked code; the server refuses to start
against a library with a different ABI version.

## Quantized variants

Measured on Performance1536D50K, ef=150, `-w 10`, Mac, 2026-09-25:

| Variant | QPS (c=10) | Recall@100 | Compact bytes/vector |
|---|:---:|:---:|:---:|
| INT8 (default) | 8,021 | 0.960 | 1,600 |
| PolarQuant (INT4) | 6,635 | 0.965 | 868 |
| TurboQuant (INT3+QJL) | 4,924 | 0.953 | 676 + 192 QJL |
| NanoQuant (INT2) | 3,390 | **0.464** | 484 |

Every quant variant also keeps an FP32 re-rank copy (6 KB/vector) in memory
and in the index file, so none of them saves memory or disk overall today;
their compact buffers only shrink the beam's working set. NanoQuant is
experimental: INT2 on these embeddings is coarse (recall@10 is 0.81 at ef=150).

The proof a mode really ran is its FT.OPTIMIZE log line —
`[PolarQuant] Block-INT4 …`, `[TurboQuant] Block-INT3 …`,
`[NanoQuant] Block-INT2 …`. Warm restart and multi-worker serving work for all
three.

---

## PolarQuant — Block-INT4 Quantization

Enabled with the `--polarquant` server flag.

### Encoding
- **Block structure:** 48 blocks x 32 dims = 1536 dimensions
- **Per-block storage:** FP16 scale factor + 32 x 4-bit packed values = 18B/block
- **Total per vector:** 48 x 18B = 868B (vs 1536B for INT8, 43% smaller)
- **WHT rotation:** Walsh-Hadamard Transform applied before quantization for robustness — spreads information across dimensions, reducing quantization error

### Usage
```bash
./pion-server --polarquant -w 10 --independent-workers
```

---

## TurboQuant — Block-INT3 + QJL Error Correction

Enabled with the `--turboquant` server flag.

### Encoding
- **Block-INT3:** 48 blocks x 32 dims, 14B/block = 676B/vector
- **QJL error correction:** random sign flips + Walsh-Hadamard Transform, 192B sign storage per vector
- **Total per vector:** 676B + 192B = 868B

### Search Kernel
- Batch-8 `int3_dot` kernel for distance computation
- QJL Hamming correction applied as a fast error-correction step after initial INT3 distances

### Usage
```bash
./pion-server --turboquant -w 10 --independent-workers
```

---

## NanoQuant — Block-INT2 Quantization

Enabled with the `--nanoquant` server flag. The smallest compact footprint (484B/vector); experimental, see the recall figures above.

### Encoding
- **Block-INT2:** 48 blocks × 32 dims, 10B/block (2B FP16 scale + 8B packed 2-bit data) = 484B/vector
- **4 levels:** unsigned {0,1,2,3} → signed {-3,-1,1,3} for SDOT; stored scale = absmax / 3, so the outer levels decode to ±absmax
- **No QJL needed:** FP32 re-rank suffices to maintain recall — simplest quantization pipeline

### Search Kernel
- Batch-8 `int2_dot` kernel: unpack 2-bit → Int8, asymmetric SDOT with INT8 query
- 8 cache lines per vector (484B) vs 14 (INT4/INT3) — lower memory bandwidth

### Usage
```bash
./pion-server --nanoquant -w 10 --independent-workers
```

---

## Streaming Ingest

Auto-enabled for indices with more than 1M elements. Eliminates the FP32 staging buffer that would otherwise be required during bulk ingest.

- **Memory savings:** at 5M vectors x 1536 dims x 4 bytes = 33.8GB FP32 staging buffer eliminated
- **Mechanism:** vectors are quantized on arrival and inserted directly into the index, bypassing the intermediate FP32 buffer
- **TurboQuant streaming compaction:** when `--turboquant` is active, streaming ingest performs INT8 to FP32 reconstruction on-the-fly for the INT3 quantization pipeline

---

## Default Configuration

| Parameter | Value | Source |
|---|---|---|
| `dimensions` | 1536 | `config.mojo` — matches VectorDBBench Performance1536D50K |
| `max_elements` | 600,000 | `config.mojo` |
| `M` | 16 | `config.mojo` (desktop/cloud profiles) |
| `ef_construction` | 100 (desktop), 200 (cloud); overridden by FT.CREATE | `config.mojo` / `slow_path.mojo` |
| `ef_runtime` | **150** | `hnsw.mojo` field default |
| `use_int4` | False (all profiles — V13 INT4 abandoned, recall 0.74) | `config.mojo` |
| `use_bq` | False (all profiles — Hamming too coarse on 1536-dim) | `config.mojo` |

---

## Benchmarking — VectorDBBench

### Harness (`benchmarks/VectorDBBench/vectordb-benchmark.py`)

Runs VectorDBBench sequentially against Redis, Valkey, Pion (all via the `redis` subcommand of the `vectordbbench` CLI), then zvec (via the `zvec` subcommand — no server required):

```bash
# Single-engine quick run (builds Pion first):
pixi run bench-vdb    # ef_construction=128, ef_runtime=150

# 4-way comparison (Redis, Valkey, Pion, zvec):
cd benchmarks/VectorDBBench
python vectordb-benchmark.py
```

The script auto-starts and stops each server. For each Redis-compatible engine it uses:
- `redis-server` for Redis (vanilla; no FT.* support → will fail gracefully)
- `valkey-server` for Valkey (vanilla; no FT.* support → will fail gracefully)
- `./pion-server -p 6379 -w 1` for Pion (native FT.*)
- `vectordbbench zvec` for zvec (embedded library, no server)

**Constants** (top of script; edit to match your test run):
```python
CASE = "Performance1536D50K"  # VectorDBBench dataset identifier
M = 16                         # HNSW graph degree
EF_CONSTRUCTION = 128          # Build-time beam width
EF_RUNTIME = 200               # Query-time beam width (matches Pion default)
PORT = 6379
```

The `redis` subcommand sends FT.CREATE → HSET (bulk ingest) → FT.OPTIMIZE → FT.SEARCH, all handled natively by Pion; no Lua scripts or modules required. Stock VectorDBBench skips the FT.OPTIMIZE step, because its Redis client's `optimize()` is empty, so the install task patches it.

**CLI entry point**: `vectordbbench redis` (installed by `pixi run install-vdbbench`). Use `vectordbbench --help` to list available backends.

### Install VectorDBBench

```bash
pixi run install-vdbbench   # venv_zvec, VectorDBBench 1.0.22, optimize() patched to send FT.OPTIMIZE
```

---

## GPU Vector Search (Metal)

Metal GPU brute-force search with async pipelining and adaptive CPU/GPU routing. Enabled with `--gpu` flag.

### Architecture

1. **Metal compute shader** (`src/ffi/metal_compute.metal`): INT8 L2 distance, 1 query vs N candidates, char4×4 unrolled + threadgroup shared query cache; optimized multiquery kernel for batch dispatch
2. **Obj-C++ wrapper** (`src/ffi/metal_wrap.m`): persistent Metal context, per-worker buffers (16 workers), async `dispatch_semaphore` completion, `dispatchThreadgroups`
3. **Adaptive routing**: rolling GPU latency tracking → auto-fallback to CPU HNSW when GPU busy (LLM/MAX)
4. **FP32 re-rank buffer**: BFS-ordered FP32 vectors built during `compact_vectors()` for optional GPU oversample re-ranking
5. **Native Mojo kernel** (`src/vector/gpu_search.mojo`) ready for Xcode-enabled builds (`-D ACCELERATOR=apple-m4`)

### Results (Performance1536D50K, ef=150, w=10→4 capped, macOS M4)

| Metric | GPU v2 (Metal) | CPU (HNSW) |
|---|:---:|:---:|
| Peak QPS (c=1) | 1,665 | 1,520 |
| Peak QPS (c=5) | 5,949 | 4,486 |
| Peak QPS (c=10) | 8,030 | 7,722 |
| Recall@100 | 0.937 | 0.937 |
| P99 Latency | 0.8ms | 0.9ms |

GPU +26% over CPU at c=5; converges at c=10 (4 P-core macOS cap). Multiquery kernel positioned for Linux io_uring batch dispatch.

### Usage

```bash
./pion-server --gpu -w 10 --independent-workers
redis-cli XGPU INFO   # GPU device, threadgroup size, dispatch latency, dispatch count
```
