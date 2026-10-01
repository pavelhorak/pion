# Data Types Reference

This document describes the internal representation and supported data types in Pion. All structures are designed for deterministic latency and zero-allocation hot paths.

Source files: `src/common/value.mojo`, `src/common/hash_map.mojo`, `src/common/list.mojo`, `src/common/skip_list.mojo`, `src/vector/hnsw.mojo`, `src/memory/slab_allocator.mojo`, `src/memory/object_pool.mojo`.

---

## GenericValue — 32-Byte Tagged Union

`GenericValue` is the universal value container. Every key and every value in the keyspace is a `GenericValue`. Total size: 32 bytes (4-byte type tag + 24 bytes of data across three `UInt64` words).

### Fields

| Field | Size | Description |
|---|---|---|
| `type` | 4 B | `ValueType` enum discriminant |
| `_data0` | 8 B | Primary data word |
| `_data1` | 8 B | Secondary data word |
| `_data2` | 8 B | Tertiary data word |

### ValueType Enum

| Value | Name | Stored in |
|:---:|---|---|
| 0 | NONE | Sentinel (empty slot) |
| 1 | STRING | Heap-allocated string (>23 bytes) |
| 3 | HASH | Pointer to nested `SlabHashMap` |
| 4 | LIST | Pointer to `SlabList` |
| 5 | SET | Pointer to `SlabHashMap` |
| 6 | ZSET | Pointer to `SlabSkipList` |
| 7 | INT | Inline 64-bit integer in `_data0` |
| 8 | FLOAT | Inline 64-bit float in `_data0` |
| 9 | STRING_SSO | Inline string (<=23 bytes, zero heap) |
| 10 | BITMAP | Pointer to byte array + length |
| 11 | HLL | Pointer to 16 KB register set |
| 12 | GEO | Pointer to `SlabSkipList` (geohash scores) |
| 13 | STREAM | Stream structure |
| 14 | VSET | Pointer to a vector set (`VADD`/`VSIM`, one per key) |

### STRING_SSO Layout (<=23 bytes, zero heap allocation)

The three data words pack the string inline:

| Word | Byte layout |
|---|---|
| `_data0` | Low byte = length (0-23). Bytes 1-7 = chars 0-6. |
| `_data1` | Chars 7-14 (8 bytes) |
| `_data2` | Chars 15-22 (8 bytes) |

Equality for SSO values is a 3-word comparison (`_data0 == _data0 && _data1 == _data1 && _data2 == _data2`), with no byte-level loop.

### STRING Layout (>23 bytes, heap allocated)

| Word | Content |
|---|---|
| `_data0` | Heap pointer to byte data |
| `_data1` | Byte length |
| `_data2` | Reserved |

### Constructors

| Function | Type produced | When to use |
|---|---|---|
| `GenericValue.from_ptr(ptr, len)` | STRING_SSO (<=23B) or heap STRING (>23B) | **All** key/value storage and **all** hash map key lookups |
| `GenericValue.from_ptr_unsafe(ptr, len)` | Always STRING (raw pointer to source buffer) | Temporary values consumed immediately, never stored or used for lookups |
| `GenericValue.from_string(s)` | STRING_SSO (<=23B) or heap STRING | Slow path only, where a heap `String` already exists |

**Critical invariant:** Never use `from_ptr_unsafe` for hash map key lookups. `__eq__` returns `False` immediately when types differ (STRING vs STRING_SSO), causing guaranteed misses against keys stored via `from_ptr`.

### Hash Function — Wyhash

- **SSO path:** 3-round folded multiply over `_data0`, `_data1`, `_data2`. Seed: `0xa0761d6478bd642f`. Each round: `_wyhash_mix(h ^ word, seed)` where mix = 128-bit multiply, XOR high/low halves.
- **Heap path:** 8-byte-step Wyhash over the byte data. Tail bytes handled via overlapping last-8-byte read (safe because heap strings are always >23 bytes).

---

## SlabHashMap — Swiss Table Open-Addressing

Source: `src/common/hash_map.mojo`

Used for the main keyspace (10M slots per worker), per-key hashes, and sets.

### Structure

- **Metadata array:** 1 byte per slot. Values: `h2` fingerprint (0x00-0x7F), `0x80` = EMPTY, `0xFF` = DELETED.
- **Key array:** `UnsafePointer[GenericValue]`, one per slot.
- **Value array:** `UnsafePointer[GenericValue]`, one per slot.
- Capacity is always a power of 2.
- First 16 metadata bytes are mirrored at `metadata[capacity..capacity+16]` for SIMD wrap-around loads.

### Probe Algorithm

1. Compute `h = key.__hash__()`.
2. `h1 = h >> 7` — initial group index. `h2 = h & 0x7F` — 7-bit fingerprint.
3. Load 16-byte SIMD vector from `metadata[idx]`.
4. XOR with `h2_vec` (broadcast h2) — zero lanes are fingerprint matches.
5. XOR with `EMPTY_vec` (broadcast 0x80) — zero lanes are empty slots.
6. On match: compare full key with `__eq__`. On empty: key not found.
7. Linear probing in 16-slot groups: `idx = (idx + 16) & mask`.
8. Next-group prefetch: `prefetch(metadata + next_idx)`.

### Rehash

Triggers at 70% fill (`size * 100 > capacity * 70`). Doubles capacity, re-inserts all non-EMPTY/non-DELETED entries.

### Capacity Defaults

| Use case | Initial capacity |
|---|---|
| Main keyspace | 10M slots per worker |
| Per-key hash (HSET) | 16 slots (from `ObjectPool`) |
| Per-key set (SADD) | 16 slots (from `ObjectPool`) |

---

## SlabList — Ziplist + Quicklist

Source: `src/common/list.mojo`

A dual-mode list that starts as a compact ziplist and converts to a two-sided segmented array (quicklist) when thresholds are exceeded.

### Constants

| Constant | Value |
|---|---|
| `ZIPLIST_MAX_ENTRIES` | 1024 |
| `ZIPLIST_MAX_VALUE_LEN` | 64 bytes |
| `ZIPLIST_INITIAL_CAP` | 8192 bytes |
| `SEG_SIZE` | 256 elements |

### Ziplist Mode (<=1024 entries, values <=64B each)

Contiguous `zip_buf` byte array. Entry format: `[u16 length][data bytes]`, packed sequentially.

- `LPUSH`: prepends via `memmove` of existing data.
- `RPUSH`: appends in place at `zip_len` offset.
- Automatic conversion to quicklist mode via `_convert_to_segmented()` when entry count exceeds `ZIPLIST_MAX_ENTRIES` or a value exceeds `ZIPLIST_MAX_VALUE_LEN`.
- Initial capacity 8192 bytes fits 1024 x ~5-byte benchmark entries without reallocation.

### Quicklist Mode (>1024 entries)

Two-sided segmented array with 256-element segments.

**Head side (LPUSH):**
- `active_head_data[head_off..head_end-1]` are valid entries.
- `head_off` decrements per LPUSH. When `head_off == 0`, the active segment is committed to `head_segs[]` and a fresh buffer is allocated.

**Tail side (RPUSH):**
- `active_tail_data[0..tail_count-1]` are valid entries.
- `tail_count` increments per RPUSH. When `tail_count == SEG_SIZE`, the active segment is committed to `tail_segs[]`.

**LRANGE traversal — 4 phases (sequential scan, no pointer chasing):**
1. Active head buffer (`active_head_data[head_off..head_end-1]`)
2. Committed head segments (newest to oldest: `head_seg_count-1` down to `head_segs_start`)
3. Committed tail segments (oldest to newest: `0` up to `tail_seg_count-1`)
4. Active tail buffer (`active_tail_data[0..tail_count-1]`)

A 600-element LRANGE reads at most 3 sequential arrays.

---

## SlabSkipList — Ordered Score Index

Source: `src/common/skip_list.mojo`

Used for sorted sets (ZSET) and geospatial indices (GEO).

### Structure

- Maximum level: 16.
- Each `SkipListNode` contains: `score: Float64`, `obj: GenericValue`, `forward: InlineArray[UnsafePointer, MAX_LEVEL]`, `level: Int`.
- Nodes allocated from `SlabAllocator[SkipListNode]`.
- Random level generation via `Xoshiro256PlusPlus` PRNG.

### Complexity

| Operation | Time |
|---|---|
| Insert | O(log n) |
| Delete | O(log n) |
| Range query | O(log n + k) where k = result count |
| Score lookup | O(log n) |

---

## Memory Subsystem

### SlabAllocator[T]

Source: `src/memory/slab_allocator.mojo`

- `mmap`-backed bump allocator. `allocate()` = O(1) bump pointer advance or free-list pop.
- `deallocate()` = O(1) free-list push. Caller must call explicitly (no GC).
- Adaptive: doubles `items_per_slab` on exhaustion (up to 10M items), resulting in O(log N) total `mmap` calls.

### ObjectPool[T]

Source: `src/memory/object_pool.mojo`

- Fixed-capacity stack of pre-allocated pointers. `acquire()` pops, `release()` pushes.
- Used for `SlabHashMap(16)` instances (SADD new-key, HSET new-key) to avoid hot-path allocation.
- Always call `.reset()` on acquired objects before use — pool objects retain previous state.

---

## Data Type Details

### Strings & Counters

Commands: `SET`, `GET`, `DEL`, `EXISTS`, `APPEND`, `STRLEN`, `GETRANGE`, `SETRANGE`, `MGET`, `MSET`.

- Values <=23 bytes use STRING_SSO: stored inline in the three `UInt64` words. Zero heap allocation for both storage and retrieval.
- Values >23 bytes are heap-allocated STRING.
- Integer values (detected on SET or INCR) use the INT type: `_data0` holds the raw `Int64`. `INCR`/`DECR` operate directly on `_data0` with zero allocation or parsing overhead.

### Hashes

Commands: `HSET`, `HGET`, `HMGET`, `HGETALL`, `HKEYS`, `HVALS`, `HLEN`, `HDEL`, `HEXISTS`, `HINCRBY`.

- Each hash key points to a nested `SlabHashMap` (initial capacity: 16 slots).
- New hash creation acquires a pre-reset map from `ObjectPool[SlabHashMap]` — no hot-path allocation.
- Multi-field HSET with a vector field routes to `HNSWGraph.add_vector()` when an index is active and the field name matches the configured `vector_field_name`.

### Lists

Commands: `LPUSH`, `RPUSH`, `LPOP`, `RPOP`, `LLEN`, `LRANGE`, `LINDEX`, `LSET`, `LINSERT`, `LREM`, `LTRIM`, `LPOS`, `LMOVE`.

- Backed by `SlabList` (ziplist + quicklist, described above).
- LPUSH/RPUSH/LPOP/RPOP are O(1) in both modes.
- LRANGE is sequential-scan in both modes (ziplist: skip-to-start + write phases; quicklist: 4-phase traversal).

### Sets

Commands: `SADD`, `SPOP`, `SCARD`, `SMEMBERS`, `SREM`, `SINTER`, `SUNION`, `SDIFF`, `SISMEMBER`.

- Each set is a `SlabHashMap` with values set to a sentinel. Same Swiss Table probing as the main keyspace.
- New set creation acquires from `ObjectPool[SlabHashMap]` (16-slot initial capacity).

### Sorted Sets

Commands: `ZADD`, `ZPOPMIN`, `ZPOPMAX`, `ZRANGE`, `ZRANGEBYSCORE`, `ZRANGEBYLEX`, `ZRANK`, `ZREVRANK`, `ZREM`, `ZSCORE`, `ZINCRBY`, `ZCARD`, `ZCOUNT`, `ZUNION`, `ZINTER`, `ZDIFF`, `ZLEXCOUNT`, `ZRANDMEMBER`, `ZMSCORE`, `ZSCAN`, and more (29 commands total).

- Backed by `SlabSkipList`. Elements ordered by `Float64` score.
- O(log n) insert/delete/lookup, O(log n + k) range queries.

### Vectors (HNSW Index)

Commands: `FT.CREATE`, `FT.SEARCH`, `FT.HYBRID`, `FT.INFO`, `FT.OPTIMIZE`, `FT.DROPINDEX`, `FT.ADDTEXT`, `FT.SEARCHTEXT`.

- Vector data ingested via multi-field `HSET` (field name must match the index schema).
- Indexed by `HNSWGraph` — multi-layer approximate nearest neighbor index.
- Configuration: M=16, ef_construction=128, ef_runtime=150 (default).

**Quantization modes:**

| Mode | Flag | Storage | Notes |
|---|---|---|---|
| INT8 (default) | (none) | 1,600 B/vector compact | Batch-8 prefix pruning + suffix early-exit |
| PolarQuant INT4 | `--polarquant` | 868 B/vector compact | Block-INT4 |
| TurboQuant INT3+QJL | `--turboquant` | 676 + 192 B/vector compact | Block-INT3 + QJL correction |
| NanoQuant INT2 | `--nanoquant` | 484 B/vector compact | Experimental, low recall |

Measured recall and QPS, and the FP32 re-rank copy every quant mode keeps: `doc/vector_engine.md` § Quantized variants.

- `FT.OPTIMIZE` builds and compacts the graph on the worker that receives it. HNSW graph persists to disk via `save_to_disk`/`load_from_disk`.
- `FT.HYBRID`: combines vector similarity with BM25 text scoring.

### Vector sets

Commands: `VADD`, `VSIM`, `VCARD`, `VDIM`, `VINFO`, `VISMEMBER`, `VSETATTR`, `VGETATTR`, `VEMB`, `VRANDMEMBER`, `VREM`, `VRANGE`.

- Redis 8 vector sets: one set per key, stored as `ValueType.VSET`, independent of the `FT.*` index.
- Vectors are stored as FP32 unit vectors plus their norm; `VSIM` is an exact search (no graph), scored `(1 + cos) / 2` as Redis reports it.
- Persisted through the WAL and snapshots.

Source: `src/common/vector_set.mojo`, `src/commands/vset.mojo`.

### Streams

Commands: `XADD`, `XLEN`, `XDEL`, `XREAD`, `XREAD BLOCK`, `XRANGE`, `XTRIM`.

- Stream entries keyed by auto-generated or user-provided `<ms>-<seq>` IDs.
- `XREAD BLOCK` supports timeout and wake-up on new entries.

Source: `src/commands/stream.mojo`.

### Bitmaps

Commands: `SETBIT`, `GETBIT`, `BITCOUNT`, `BITOP` (AND/OR/XOR/NOT), `BITPOS`, `BITFIELD`.

- Stored as `ValueType.BITMAP`: `_data0` = pointer to `UInt8` byte array, `_data1` = length in bytes.
- Auto-grows on `SETBIT` beyond current length.
- `BITCOUNT` with no range args is handled on the fast path; range args fall through to slow path.
- **GET on BITMAP keys returns raw bytes** (matching Redis behavior where BITMAP and STRING are interchangeable). BITOP result keys are readable via GET.
- **BITOP supports STRING/STRING_SSO source keys** (not just BITMAP) — treats string bytes as bit arrays.

Source: `src/commands/bitmap.mojo`.

### HyperLogLog

Commands: `PFADD`, `PFCOUNT`, `PFMERGE`.

- Stored as `ValueType.HLL`: `_data0` = pointer to 16 KB register set (16384 registers, 14-bit precision).
- Hash function: Wyhash (not MurmurHash). Consistent with the rest of the engine.
- `PFADD`/`PFCOUNT` handled on the fast path (single-key).

### Geospatial

Commands: `GEOADD`, `GEOPOS`, `GEODIST`, `GEOHASH`, `GEORADIUS`, `GEOSEARCH`, `GEOSEARCHSTORE`.

- Stored as `ValueType.GEO`: `_data0` = pointer to `SlabSkipList`.
- Each member's score is its 52-bit geohash computed from longitude/latitude. The member name is stored as the skip list node's `obj`.
- Range queries (GEOSEARCH, GEORADIUS) exploit skip list score ordering for efficient bounding-box scans.

Source: `src/commands/geo.mojo`.

---

## Fast Path Coverage

The following commands are handled in `FastPathHandler.process_data_plane()` with zero heap allocation for keys <=23 bytes:

```
GET  SET  MGET  MSET  INCR  DECR  HSET  HGET  LPUSH  RPUSH
LPOP  RPOP  LRANGE  LLEN  DEL  EXISTS  SADD  SPOP  ZADD  ZPOPMIN
PING  FUNCTION LOAD  FCALL  GETBIT  SETBIT  BITCOUNT  PFADD  PFCOUNT
```

All other commands (FT.*, GEO*, XADD, SUBSCRIBE, CONFIG, INFO, etc.) are routed through `SlowPathHandler.process_slow_path()`.
