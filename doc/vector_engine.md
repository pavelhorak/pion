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
- **FP32 buffer ingest**: `add_vector()` buffers raw FP32 vectors; graph is built in batch on `FT.OPTIMIZE` call (no graph operations during HSET).
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
FT.CREATE <index> [ON HASH] [PREFIX <n> <prefix>...] [SKIPINITIALSCAN] SCHEMA
    <field> VECTOR HNSW|FLAT <count> TYPE FLOAT32 DIM <d> DISTANCE_METRIC L2|COSINE
        [M <m>] [EF_CONSTRUCTION <ef>] [EF_RUNTIME <ef>] [INITIAL_CAP <n>] ...
    <field> TAG [SEPARATOR <c>] [CASESENSITIVE] [SORTABLE [UNF]] [NOINDEX]
    <field> NUMERIC [SORTABLE [UNF]] [NOINDEX]
    <field> TEXT | GEO | GEOSHAPE ...
```
The whole command is parsed and checked before any index state is written, so
a refused `FT.CREATE` changes nothing. From 1.0.4 every part of it is applied or
refused. Before, the handler picked out the words it knew and took every other
word for a field name, so an option it did not know answered `+OK` and changed
nothing (`tests/test_ft_differential.py` compares every option with
RediSearch's own answer):

- **Applied:** `ON HASH`, `PREFIX`, `SKIPINITIALSCAN`; one `VECTOR` field with
  `TYPE FLOAT32`, `DIM`, `DISTANCE_METRIC L2` or `COSINE`, `EF_CONSTRUCTION`;
  `TAG` with `SEPARATOR` and `CASESENSITIVE`, `NUMERIC`, `TEXT` (for BM25), and
  `NOINDEX` (the field matches no filter, as in RediSearch). `SORTABLE`, `UNF`
  and `M` change nothing Pion answers.
- **Accepted, with no effect on an answer Pion gives:** the full-text index's
  options (`LANGUAGE`, `LANGUAGE_FIELD`, `SCORE`, `SCORE_FIELD`,
  `PAYLOAD_FIELD`, `MAXTEXTFIELDS`, `NOOFFSETS`, `NOHL`, `NOFIELDS`, `NOFREQS`,
  `STOPWORDS`), a `TEXT` field's `WEIGHT`, `NOSTEM` and `PHONETIC`,
  `INDEXEMPTY`, `INDEXMISSING`, `WITHSUFFIXTRIE`, the `FLAT` and `SVS-VAMANA`
  algorithms (built as HNSW) and their attributes, and `GEO` and `GEOSHAPE`
  fields, whose queries Pion refuses.
- **Refused:** `ON JSON`, `FILTER`, `TEMPORARY`, `AS` aliases, a vector `TYPE`
  other than `FLOAT32`, `DISTANCE_METRIC IP`, a second `VECTOR` field, an
  unknown option, field type or vector algorithm, a vector attribute count that
  does not match its attributes, a missing `TYPE` or `DIM`, `DIM 0`, a missing
  or empty `SCHEMA`, a field named twice, a field name over 32 bytes, and more
  than 16 fields besides the vector. The errors are RediSearch's words where it
  has some.

- Captures `<index>` (first token after the command) as `hnsw.index_name`, and
  the `VECTOR` field's name as `vector_field_name`.
- The schema's fields, with their kind, `SEPARATOR` and flags, go to the shared
  view (`schema_field_*`), where every worker's filters read them, and are
  saved with the index (index-file word 25, bit 10), so a restart keeps them.
  An index file written before 1.0.4 has no schema: it loads as before, and
  its filters then read any hash field.
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
- **The hashes already in the keyspace are indexed too**, as RediSearch does:
  every hash whose vector field has `DIM` float32s, on every worker. The
  worker that ran `FT.CREATE` indexes its own before it answers `+OK`; the
  others index theirs at their next housekeeping tick (tens of
  milliseconds). The server log says `[vector] FT.CREATE: indexed N existing
  hashes from worker W's keyspace in M ms`. `SKIPINITIALSCAN`, placed before
  `SCHEMA`, turns this off: only hashes written afterwards are indexed.
  Re-issuing `FT.CREATE` for the index being built adds nothing, since a hash
  already in it is skipped. Over a large keyspace this blocks the worker
  while it runs, as `FT.OPTIMIZE` does.
- **`PREFIX <n> <prefix> …`** (from 1.0.3). Only a hash whose key starts with one of
  the prefixes is indexed, as in RediSearch, whether it was written before
  `FT.CREATE` or after:
  - The match is byte for byte and case-sensitive, and a key equal to a prefix
    matches. Several prefixes are ORed.
  - `PREFIX 0`, an empty prefix, or no `PREFIX` at all means every key.
  - A hash renamed out of the prefixes leaves the index, and one renamed in
    joins it.
  - A write outside the prefixes is not this index's, so it is not counted in
    `vector_unindexed_writes` either.
  - The list is saved in the index file, so a restart keeps it.
  - At most 16 prefixes of up to 63 bytes each. A longer list, a longer
    prefix, or a count that is not a number is refused, and the `FT.CREATE`
    changes nothing.
  - `FT.CREATE` with a different prefix list replaces the index, as a
    different name or `DIM` does. The new index then takes the hashes its list
    covers. Re-issuing the same list changes nothing.

  Before 1.0.3, `PREFIX` was accepted and not applied, so an index took every
  hash with its vector field and `DIM`. An index file written before 1.0.3
  holds no list, so it keeps answering for every key until `FT.DROPINDEX`
  followed by `FT.CREATE`. A client that re-creates its index at startup with
  a `PREFIX` replaces it once, and the new index is built from the hashes
  under the prefix.

  Before 1.0.1, `FT.CREATE` indexed only what was written after it, so an index
  created over existing documents answered `[0]`. That included the restart
  of an index whose `DIM` differs from the server's `--dim`. Its file is not
  restored, and re-issuing `FT.CREATE` after the restart is the way back.

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
FT.SEARCH <index> "<prefilter>=>[KNN <k> @<field> $blob [EF_RUNTIME <ef>] [AS score]]"
    PARAMS 2 blob <float32_bytes> [FILTER ...] [RETURN ...] [NOCONTENT] [WITHSCORES]
    [SORTBY <field> [ASC|DESC]] [LIMIT <offset> <count>] [DIALECT 2] [TIMEOUT <ms>]
```
- `<index>` must be the index this server holds, registered by `FT.CREATE` or
  `FT.OPTIMIZE` and removed by `FT.DROPINDEX`. Another name gets
  `no such index` (or, when another index is served, an error naming it). Before
  1.0.4, after an `FT.DROPINDEX` or a refused `FT.CREATE`, `FT.SEARCH` answered
  from the vectors the dropped definition kept routing while `FT.INFO` said
  `Unknown index name`.
- Extracts `k` from the query's `KNN ` clause, and `EF_RUNTIME <ef>` if present
  (else `hnsw.ef_runtime`). Resolves the query's `$<param>` (and an `EF_RUNTIME
  $<param>`) against PARAMS by name; the vector must be `hnsw.dim * 4` bytes.
- Calls `hnsw.search_fp32_scored(blob_ptr, k, dist_scores, ef_query)` for the k
  nearest. `KNN 0` returns an empty result.
- **`SORTBY <field> [ASC|DESC]`** then orders those k hits: by the KNN score
  (its `AS` alias or `__<field>_score`; `DESC` reverses), or by a schema field,
  NUMERIC as a number, TAG and TEXT by their bytes. As in RediSearch, documents
  without the field come last in both directions, and ties go by document
  order (reversed by `DESC`). A field that is not in the schema is refused
  (`Property ... not loaded nor in schema`). Before 1.0.4 SORTBY was read and
  dropped: the hits came back by distance whatever was asked.
- **`LIMIT <offset> <count>`** then takes the page, and the reply's total is
  every hit, as in RediSearch. Both numbers must be integers of 0 or more.
  Without `LIMIT`, every hit is returned (`KNN 20` returns 20). Before 1.0.4 the
  offset was ignored (`LIMIT 2 3` answered with the first three), and the total
  was the page's size.
- Without `SORTBY`, Pion returns the hits by distance. RediSearch returns them
  in its internal document order, so a `LIMIT` page without `SORTBY` holds
  different hits on the two. Clients that page (RedisVL, LangChain) send
  `SORTBY` on the score.
- Options that change nothing a KNN query answers are accepted: `DIALECT`,
  `TIMEOUT`, and the text query's `VERBATIM`, `NOSTOPWORDS`, `INORDER`, `SLOP`,
  `LANGUAGE`, `SCORER`, `EXPANDER`, `PAYLOAD`, `INFIELDS`, `SUMMARIZE` and
  `HIGHLIGHT`. `INKEYS`, `GEOFILTER`, `WITHPAYLOADS`, `WITHSORTKEYS` and
  `EXPLAINSCORE` are refused, and so is any other option (RediSearch's
  ``Unknown argument `X` ``). Before 1.0.4 an option FT.SEARCH did not know was
  skipped, so `INKEYS` and `GEOFILTER` answered as if absent.
- A document past its key's TTL is gone, as for any read: the search expires
  it (Redis's lazy expiry) and does not return it. It used to come back until
  the active sweep reached the key.
- Every query is read as `DIALECT 2`. RediSearch 8 defaults to dialect 1, where
  the KNN syntax is an error; clients send `DIALECT 2`.
- After an automatic build the graph's hits are scored from the FP32 rows the
  build keeps, so the scores are exact. They came from the INT8 codes, a little
  off the true distance, when no vector had been written since the build.
- Returns RediSearch's reply. In RESP2:

```
*(1 + N*(1 + W + C))\r\n  :N\r\n
[for each result:
  $<len>\r\n<doc key>\r\n
  (WITHSCORES)  $1\r\n0\r\n          ← the query's text score, as below
  (unless NOCONTENT)  *2M\r\n <M field/value pairs>
]
```

`W` is 1 with `WITHSCORES`, `C` is 0 with `NOCONTENT` (or `RETURN 0`). The fields are:

- **`RETURN n field [AS name] ...`**: the named hash fields, renamed by `AS`. The KNN score is the field named by the clause's `AS` alias, or `__<vector field>_score` without one, and comes first when named, as in RediSearch. A field the hash lacks is left out. `n` counts every token after it; a bad or short count answers RediSearch's own error (`Bad arguments for RETURN: ...`).
- **No RETURN**: the KNN score (under the clause's `AS` alias, or `__<vector field>_score`), then every field of the hash, the embedding included, as RediSearch sends. A caller that needs only the keys should pass `RETURN 1 id` or `NOCONTENT`, and one that reads the score by name should name it with `AS`.
- **`WITHSCORES`** reports the query's text score, not the distance, as RediSearch does: `0` for `*=>[KNN …]`. RediSearch reports a text score for a filter prefix too; Pion computes none and reports `0`.

On a RESP3 connection (`HELLO 3`, and redis-py 8 by default) the reply is RediSearch's map: `attributes` `*0`, `format` `+STRING`, `results` (one map per document: `id`, `score` with WITHSCORES as a double, `extra_attributes` with the fields above unless NOCONTENT, `values` `*0`), `total_results`, `warning` `*0`. The BM25 and HYBRID forms of FT.SEARCH, and FT.HYBRID, are Pion's own syntax: they ignore RETURN, NOCONTENT and WITHSCORES and send `id` (the doc hash's own `id` field, when it has one) and `score`.

This shape is emitted at every worker count. Doc-key resolution order:
shared `hk_keys_buf` slot→key map (one direct load; written by the fast-path
vector HSET), then the answering worker's slot map (keys >31B that don't fit
the shared map's 32-byte slots; see "The index's records are not keys"), then
the numeric slot id. Field values come
from the doc hash on the worker answering the query. Under `-w N` the keyspace
is per worker, so a worker that does not hold a document's hash sends none of
its fields, as Redis leaves out a field a document lacks: it never substitutes
the key or slot for a field it cannot see.
`tests/test_ft_search_replies.py` checks every case against RediSearch's
replies (`--oracle` runs the cases against RediSearch itself).

### FT.INFO
```
FT.INFO <index>
```
Returns `index_name` and `num_docs` (the documents FT.SEARCH answers from: the built graph's, plus any it answers exactly, so before a build every vector ingested, less the documents that left: deleted, expired, renamed or given a new vector; 0 before any): a 4-element array of bulk strings in RESP2, a map with `num_docs` an integer in RESP3. Returns `-ERR Unknown index name` for a name the server does not hold. RediSearch's FT.INFO has more fields (`attributes`, sizes, statistics); Pion sends these two.

`num_docs` counts documents with a vector. RediSearch also counts a hash it indexed only by its other fields (one without the vector field, or whose vector was removed). Before 1.0.4 `num_docs` counted every slot ever written, so it only grew: 40 documents, two of them deleted and one updated, read 41.

### FT._LIST
```
FT._LIST
```
The names of the indexes this server holds: the one index, or none. RedisVL calls it to see whether an index exists before `FT.CREATE`. Added in 1.0.4; it was an unknown command.

### FT.OPTIMIZE
```
FT.OPTIMIZE <index>
```
Builds the graph now, from everything in the FP32 ingest buffer. Calls `hnsw.build_index_from_shared()` (if shared buffer has data) or `hnsw.build_index()` (local). After build: `compact_vectors()` reorders vectors in BFS order, `publish_to_shared()` makes the index available to other workers via `SharedHNSWView`. Returns `+OK`. With nothing ingested since the served index was built, it keeps that index and returns `+OK`.

`FT.OPTIMIZE` is optional: without it the index answers exactly and then builds itself (see "Searching without FT.OPTIMIZE" below). Send it to have the graph built at once, and to say that ingest is over.

**After `FT.OPTIMIZE`, ingest is over.** The FP32 ingest buffer is freed after the build, so an `HSET` issued *after* `FT.OPTIMIZE` is stored as a hash but **not indexed**: `FT.SEARCH` will not find it, and `INFO` counts it as `vector_unindexed_writes`. This holds after a restart too (see below). For incremental inserts, leave `FT.OPTIMIZE` out (the automatic build keeps the buffer, across restarts as well), or `FT.DROPINDEX` → re-ingest everything → `FT.OPTIMIZE` again. Re-issuing `FT.CREATE` for the index being served changes nothing, and does not reopen ingest — unless that index is empty (built with no documents), which has nothing to protect: then ingest opens again. `FT.OPTIMIZE` with no index defined (no `FT.CREATE`, nothing ingested, no graph) answers `Unknown index name`; it used to build and serve an empty index named by its argument, under which the real index's `FT.CREATE` would not open ingest.

**Deletions do follow the keyspace (#46).** A document leaves the results when its hash leaves the keyspace, by DEL, UNLINK, expiry, FLUSHALL, or a write that replaces the key. It also leaves when the vector leaves the document: an `HSET` of a new vector, an `HDEL` of the field, or a field TTL. Writing the vector it already has (the same bytes: an ingest script run twice) keeps it indexed. A renamed document leaves too: its slot names the old key, and after the build the new name is not indexed. Before the build, a renamed, copied (`COPY`) or restored (`RESTORE`) hash is ingested under its new name, as an `HSET` would be.

Mechanics:
- The hash records its slot. The slot dies with the hash, or with its vector field.
- A dead slot is one byte in an array all workers share, so a search on any worker skips it at once.
- The graph is not changed. Dead nodes still route the beam, as HNSW soft deletes do, and `FT.SEARCH` filters them out of the results. The KNN path widens its candidates until it has k live documents.
- Each worker records the slots it killed as `<build id>:<slot>`, in a table beside the keyspace (WAL record 47). The WAL, snapshots and replication carry it.
- A restart applies the members whose build id matches the loaded index. Each `FT.OPTIMIZE` draws a new build id, saved in index header word 31.

### The index's records are not keys

Each document the index ingests gets a slot, and the worker that ingested it keeps the slot's key in a slot map, beside the keyspace. The map resolves keys over 31 bytes in `FT.SEARCH` replies, and BM25, RERANK and `FT.DROPINDEX DD` read it. After a restart it links each hash back to its slot, which is what lets a `DEL` after the restart remove the document. The dead slots above live beside it.

Until 1.0.5 both were keys: `__hk__<slot>` held the document's key, and the set `__hk_dead__` held the dead slots. So `DBSIZE`, `KEYS`, `SCAN`, `RANDOMKEY` and `INFO`'s key count saw twice the keys a client wrote, and a client's own key named `__hk__7` was overwritten when slot 7 was ingested. Deleting them, with `SCAN MATCH` and `DEL` or any cleanup of unknown keys, left every document deleted after the next restart in the results. `FLUSHALL` keeps both tables, as it keeps the index; its documents die with their hashes.

Persistence: WAL records 46 (a slot's key), 47 (a dead slot) and 48 (clear the dead slots, or the whole map); every snapshot and `BGREWRITEAOF` writes them. A data directory an older server wrote loads as it is (see `doc/operations.md` §3c). `tests/test_vector_slot_map.py` checks the keyspace, a client's `__hk__7`, restarts through the WAL, `SAVE` and `BGREWRITEAOF`, `DD`, `-w 4`, a replica, and both upgrade paths.

### Searching without FT.OPTIMIZE

A stock Redis client (RedisVL, LangChain, VectorDBBench's own) creates the index, `HSET`s documents and searches. It never sends `FT.OPTIMIZE`. The index answers it anyway (`src/network/vector_auto.mojo`):

1. **Before a graph is built, `FT.SEARCH` is exact.** It scans the FP32 ingest buffer and returns the true nearest neighbours, in the same reply shape and score units as the graph (`1 - cos` under COSINE, squared L2 under L2). Deleted documents and `FILTER` are honoured; `EF_RUNTIME` does not apply. The scan reads every vector once per query, so it slows as the index grows.
2. **Once inserts pause for 200 ms, the graph builds in the background.** One worker runs the same build `FT.OPTIMIZE` runs, on its own thread, while every worker keeps answering exactly. That worker then publishes the graph, and the thread saves it to `pion.hnsw.0`. The server log says `[vector] auto-build: indexing N vectors` and `N vectors indexed in M ms`.
3. **Vectors written after that build are found at once.** The ingest buffer is kept. `FT.SEARCH` scans the vectors written since the build exactly and merges them with the graph's results. Both sides are scored from the FP32 rows for that merge: the graph's own scores come from INT8 codes, which are close to the exact distance but not equal to it, and a merge needs both sides on the same footing. When they reach a tenth of the index and inserts pause again, the graph is rebuilt the same way. During a rebuild, searches are exact again.

`INFO` reports the state process-wide: `vector_index` (`none`, `empty`, `exact`, `building`, `graph`, or `graph+delta`), `vector_graph_rows`, `vector_exact_rows`, `vector_auto_builds` and `vector_unindexed_writes`.

Costs and limits:
- The kept buffer costs `dim × 4` bytes per vector: 6 KB at 1536 dimensions, about 300 MB at 50K. An explicit `FT.OPTIMIZE` frees it.
- An exact scan reads every vector it covers, so an index searched before its graph exists, or during a rebuild, serves far fewer queries per second than the graph does. Measured on an EPYC 8124P at 500K vectors of 1,536 dimensions, one worker and one client (`benchmarks/results/2026-10-20-linux-epyc-8124p/drop_in/`): a query takes 1.6 ms from the graph alone, 3.8 ms with 10,000 vectors written since the build (about 0.18 ms per 1,000), and 190 ms during the 84 s rebuild of 551,000.
- Link lanes: the build uses one per online CPU, from 4 to 16. `PION_BUILD_LANES=N` overrides it.
- **A restart leaves the index as its build left it** (from 1.0.2). The graph loads from `pion.hnsw.0` and serves, and an automatically built index keeps indexing. Before any worker serves, each one puts back the FP32 rows of the documents it holds: the file has the graph, not those rows. The documents written after the build, or rewritten since, become the delta again. A vector written after the restart is found at once, and the index rebuilds itself as before. The log says `[vector] index '<name>' loaded (N vectors, built automatically): it keeps indexing`. The refill reads every hash once, and the server answers nothing until it is done, as during WAL replay. If none of the index's documents is in the keyspace (a server run with `--no-wal`), the index serves as loaded, with ingest closed.
- An index `FT.OPTIMIZE` built keeps ingest closed after a restart, as that command says. A vector written to it afterwards is stored, not indexed, and counted in `vector_unindexed_writes`. The log says `loaded (N vectors) with ingest closed`. A file written before 1.0.2 loads the same way, because it does not record how its index was built. `FT.DROPINDEX` followed by `FT.CREATE` indexes every hash in the keyspace, and the file of the next automatic build records it. Before 1.0.2, every index loaded closed after a restart, and the writes it missed were not counted.

Only the worker whose graph holds the published index rebuilds it. `FT.CREATE`, `FT.DROPINDEX` and `FT.OPTIMIZE` wait for a running build. A worker that borrowed another worker's index gives its buffers back before it builds or resets. It used to build straight into them.

### FT.DROPINDEX
```
FT.DROPINDEX <index> [DD]
```
`DD` also deletes the index's documents, as in RediSearch: each live document's key goes as with `DEL` (WAL record, `WATCH`). Any drop also clears the worker's slot map and dead slots, which describe the dropped index. Before 1.0.4 `DD` was read and dropped, so the documents stayed. Under `-w N` each worker holds its own documents, and one worker cannot delete another's: `DD` is refused there. Any other argument is refused (it used to drop the index anyway).

Calls `hnsw.reset_index()` — zeroes `num_nodes`, resets `entry_point_id = -1`, clears `node_map` and `visited_map`, resets `index_name_len = 0`, resets `global_min/max` to calibration defaults, sets `index_ready = False`. Returns `+OK`. It also deletes the index file, `pion.hnsw.0`, so a restart does not bring the dropped index back. Before 1.0.2 the file stayed, and after a restart `FT.INFO` and `FT.SEARCH` answered from the dropped index. An `FT.CREATE` that replaces an index of another name or `DIM` deletes its file the same way.

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

# (b) alongside a vector — any key shape; the slot map the vector HSET
#     writes is what points BM25 at the text
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

- **NUMERIC** `@f:[lo hi]` is inclusive; `(` makes a bound exclusive (`@f:[(10 20]`, `FILTER f (10 20`); `-inf` / `+inf` work, and so does `1e400` (infinity, as strtod reads it). Bounds and values are doubles: a stored value that is not a number fails the clause. Before 1.0.4 `(` was refused and both sides were Float32, which cannot tell 1,700,000,014 from 1,700,000,000, so a timestamp range matched documents outside it.
- **TAG** values are tag lists, as RediSearch indexes them: the stored value is split at the field's `SEPARATOR` (`,` by default), each tag is trimmed of spaces, and `@f:{a}` matches a document with the tag `a`, ignoring ASCII case unless the field is `CASESENSITIVE`. `@f:{a|b}` matches either. Before 1.0.4 the whole value was compared byte for byte, so `a,b` matched neither `{a}` nor `{b}`, and `{A}` did not match `a`. RediSearch also folds non-ASCII letters; Pion folds only A-Z.
- The field must be in the index's schema, and of the clause's kind: a range on a NUMERIC field, a tag set on a TAG field. Anything else is RediSearch's error (`Unknown field`, `Expected a NUMERIC field`, `Field is not a numeric field`). A `NOINDEX` field matches nothing. An index loaded from a file written before 1.0.4 has no saved schema: there a clause reads any hash field, as before.
- Up to **4** clauses, ANDed — several `FILTER` arguments and/or clauses in the query prefix. `|` between clauses (OR) and negation are refused.
- Filtering is applied to KNN candidates, widening the candidate set (4×, 16×, … up to 16,384 or the whole index) until `k` rows pass — a selective filter costs more search, it does not silently return fewer rows while matches exist below that bound. Each candidate's fields are read from its hash, so a field changed after a build counts at once. The build used to keep a copy of each document's TAG and NUMERIC values, which went stale when a field changed.
- Field names are case-sensitive, as in the hash.
- `FILTER f=value`, `FILTER @f:[lo hi]` and `FILTER f [lo hi]` are Pion's own forms; RediSearch's `FILTER` takes `field lo hi` only.

The query itself must be `<prefilter>=>[KNN k @field $param [EF_RUNTIME n] [AS alias]]` with the vector in `PARAMS` by name; an unknown form, an unknown vector field, a missing `$param` or a blob of the wrong size is an error.

### Differences from RediSearch

`tests/test_ft_differential.py` sends every `FT.*` option to Pion and to RediSearch (Redis 8's search module) and compares the answers. Where they differ on purpose, `tests/ft_differential_known.tsv` says why. Besides what Pion refuses (above), these remain:

- `FT.CREATE` for the index already served answers `+OK` and changes nothing; RediSearch answers `Index already exists`. Clients re-create defensively, and both answers let them.
- `DISTANCE_METRIC` may be left out, and is then `L2`; RediSearch requires it.
- `FLUSHALL` keeps the index definition (`num_docs` 0, no hits); RediSearch drops its indexes.
- A hash whose NUMERIC field is not a number stays in the index for KNN (a range filter skips it); RediSearch does not index the document at all.
- `num_docs` counts documents with a vector; RediSearch also counts hashes it indexed by their other fields only.
- The vector field's name is matched ignoring case (`Vec` and `vec` are one field); RediSearch matches it exactly.
- Punctuation in a tag query is read literally (`@f:{x;y}`); RediSearch needs it escaped.
- Without `SORTBY`, hits come back by distance; RediSearch returns them in its internal document order. `SORTBY` ties go by document order: RediSearch numbers a document again on every write, Pion when its vector is written.
- `RETURN` of a NUMERIC field returns the bytes stored, as `HGET` does; RediSearch reformats the number (`51.0` comes back as `51`).
- Every query is read as `DIALECT 2`.
- Without `LIMIT`, `FT.SEARCH` returns every KNN hit; RediSearch returns 10 (`LIMIT 0 10` is its default). Clients send `LIMIT`.
- FT.INFO sends `index_name` and `num_docs` only. `FT.ALTER`, `FT.AGGREGATE`, `FT.EXPLAIN`, `FT.PROFILE`, `FT.TAGVALS`, `FT.ALIAS*`, `FT.CONFIG`, `FT.SPELLCHECK`, `FT.SYN*`, `FT.DICT*`, `FT.SUG*` and `FT.CURSOR` are unknown commands. A query without a `KNN` clause (`*`, `@cat:{a}`) is refused.

**Scores** are the index metric's distance for each returned row — squared L2 under `L2`, `1 - cos` under `COSINE` — computed from the stored vectors (FP32 re-rank copy, or the INT8 codes dequantized), not the beam's internal quantized distance. Rows are sorted by that score. From INT8 codes, the score compares them with the query clamped to the calibrated range, because the codes are clipped to it: the calibration (mean ± 8 sigma) leaves an embedding's few near-constant outlier dimensions outside, which cancel between documents but used to add their clipped part to every score.

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
`-w 10`, M4 Mac mini, 2026-10-06; per mode the arms ran closed, open, open,
closed, closed, open, and each closed run is paired with the open run next to
it; [raw output](../benchmarks/results/2026-10-06-mac-m4/vec/)):

| Search | `build` QPS | `build-open` QPS | Open build (median of the 3 pairs) | Recall@100 (both) |
|---|:---:|:---:|:---:|:---:|
| INT8 (default) | 8,729 | 6,450 | **−26.1%** (pairs: −24.2, −26.1, −26.1) | 0.960 |
| PolarQuant (INT4) | 7,732 | 4,874 | **−36.7%** (−36.7, −41.8, −31.5) | 0.964–0.968 |
| TurboQuant (INT3 + QJL) | 6,419 | 4,248 | **−35.5%** (−35.5, −37.4, −32.5) | 0.953 |
| NanoQuant (INT2) | 7,385 | 3,449 | **−55.3%** (−57.0, −55.3, −53.2) | 0.463 |

QPS is each build's median over its three runs. Recall is the same because the
results are the same (`tests/test_vector_build_equivalence.py` checks keys and
scores); per-run recall differs only through HNSW build nondeterminism.
**Nothing else changes between the two builds** — KV throughput, prompt-cache
TTFT and every other headline number come from open code.

**On Linux x86-64** (INT8, gate config with `-w 16`, AMD EPYC 8124P, 2026-10-06).
Both binaries were built on the machine from one commit for the same x86-64-v2
target: `pixi run build-portable` (the release's Linux build, with the library)
and `pixi run build-open`. The runs rotated through three arms, the third being
the library binary with `PION_VECTOR_VNNI=0`
([raw output](../benchmarks/results/2026-10-06-linux-epyc-8124p/ovc/)):

| Search | Median QPS (3 runs) | Open build against it | Recall@100 (all runs) |
|---|:---:|:---:|:---:|
| `libpion_vector`, VNNI kernels (the default on this CPU) | 7,502 | **−40.5%** | 0.960 |
| `libpion_vector`, x86-64-v2 kernels (`PION_VECTOR_VNNI=0`) | 5,481 | −18.5% | 0.960 |
| open build | 4,467 | — | 0.960 |

**The VNNI build is the default on CPUs that have it** (since 2026-10-05;
`PION_VECTOR_VNNI=0` forces the x86-64-v2 build). Here it was 36.9% faster than
the library's x86-64-v2 kernels. CI's Linux legs check the x86-64-v2 build
against its open reference on every PR (`pixi run test-vector-differential`).
Intel parts with AVX-512 VNNI (Ice Lake-SP, Sapphire Rapids) have not been
measured. Linux arm64 has not been measured.

The interface is open too: `src/vector/vector_abi.mojo` (the calls),
`beam_view.mojo` and `quant_beam_view.mojo` (the argument structs, whose
field order is ABI). `INFO` reports `pion_vector:` and `--version` a
`vector:` line, both asked of the linked code; the server refuses to start
against a library with a different ABI version.

## Quantized variants

Measured on Performance1536D50K, ef=150, `-w 10`, M4 Mac mini, 2026-10-06 — the
`build` arms of the open-build table above, median of three
([raw output](../benchmarks/results/2026-10-06-mac-m4/vec/)):

| Variant | Peak QPS (≤10 clients) | Recall@100 | Compact bytes/vector |
|---|:---:|:---:|:---:|
| INT8 (default) | 8,729 | 0.960 | 1,600 |
| PolarQuant (INT4) | 7,732 | 0.968 | 868 |
| TurboQuant (INT3+QJL) | 6,419 | 0.953 | 676 + 192 QJL |
| NanoQuant (INT2) | 7,385 | **0.463** | 484 |

Every quant variant also keeps an FP32 re-rank copy (6 KB/vector) in memory
and in the index file, so none of them saves memory or disk overall today;
their compact buffers only shrink the beam's working set. NanoQuant is
experimental: INT2 on these embeddings is coarse (recall@100 0.463 above).

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

- **Memory savings:** at 5M vectors x 1536 dims x 4 bytes = 30.72 GB of FP32 staging buffer eliminated
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
| `use_int4` | False (all profiles — the V13 INT4 path was abandoned for low recall) | `config.mojo` |
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

### Results

No GPU-against-CPU measurement is published with raw output yet. To measure it on
your Mac: `python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --gpu`
against the same command without `--gpu`.

### Usage

```bash
./pion-server --gpu -w 10 --independent-workers
redis-cli XGPU INFO   # GPU device, threadgroup size, dispatch latency, dispatch count
```
