# Persistence: WAL + Snapshots

Pion ensures data durability through a mmap-based Write-Ahead Log (WAL) and binary snapshots, with HNSW index persistence for warm restarts.

---

## WAL (Write-Ahead Log)

> The operational reference — `INFO` fields, crash modes, the blob tier — is
> [`doc/operations.md`](operations.md). The format is defined by
> `src/io/wal.mojo`.

Each worker owns a private WAL, `pion.wal.{worker_id}`. It is **not** a single
ring buffer: a full segment is sealed to `pion.wal.{worker}.{n}`
and a fresh one opened, with recovery replaying sealed segments oldest-first and
then the active one. Segment size is `--wal-size` (default 256 MB) and the
sealed-segment ceiling is `--wal-max-segments` (default 32).

Past that ceiling an append is refused **loudly** — `wal_dropped_entries` in
`INFO` plus a one-shot `WAL: FULL` log line — and keyspace
writes are refused with `-MISCONF` rather than acknowledged. `--wal-full-policy
drop` restores the older accept-and-lose behaviour for cache-only deployments.

### Binary format

```
[4B entry_len LE][1B cmd_id][4B key_len LE][key bytes][4B val_len LE][val bytes]
```

`cmd_id` spans **1–31**. Beyond `1=SET` and `2=DEL` there is `4` (a
24-byte blob-tier pointer record — the blob tier stores values >= `--blob-threshold`
in `pion.blob.{worker}.{seg}` arenas and logs the pointer, not the payload);
`5`–`27` are the aggregate effect records: HSET, LPUSH,
RPUSH, SADD, ZADD, HDEL, SREM, ZREM, LPOP, RPOP, GEOADD, BITMAP/HLL images,
SETBIT, LSET, LTRIM, LREM, LINSERT, XADD, PFADD, XDEL and the TTL pair;
`28`–`30` are the vector-set records (VADD, VREM, VSETATTR); and `31` is a
whole `MSET` in one record (packed varint key/value pairs).

Two rules govern that set:

- **Non-deterministic commands log their RESOLVED effect** — `SPOP` logs an
  `SREM` of the member actually popped, `XADD *` logs the id the server
  generated, `EXPIRE n` logs the absolute deadline. Never replay a random
  choice or a relative time.
- **A new mutating handler MUST append an effect record** (or route through a
  `dispatcher.execute_*` that does), or its writes vanish on replay while every
  test that does not restart the server still passes.

### Operations

| Method | Behaviour |
|---|---|
| `append()` / `append_kv()` | Pure mmap `memcpy` — zero syscalls |
| `sync()` | `msync(MS_ASYNC)` once per event-loop tick (group commit, non-blocking) |
| `recover()` | Replays all entries into keyspace at startup |
| `checkpoint()` | Resets `tail_offset=0` after snapshot |
| `compact_rewrite()` | Walks keyspace, rewrites fresh SET entries, resets WAL |

### Configuration

`wal_sync_mode` in `PionConfig`:
- `"async"` (default) — `msync(MS_ASYNC)`, non-blocking
- `"sync"` — `fdatasync` via C FFI (`pion_fdatasync()`)

**Zero overhead at P=10:** 2M+ RPS across all commands — mmap writes are invisible at pipeline depth.

---

## Snapshots

WAL-compatible binary snapshot with 64-byte header.

### Header

Magic, version, worker_id, kv_count, timestamp — packed KV entries identical to WAL format.

### Commands

| Command | Behaviour |
|---|---|
| `SAVE` | Synchronous snapshot + `WAL.checkpoint()` (resets WAL for minimal delta replay) |
| `BGSAVE` | Same as SAVE + `+Background saving started` response |
| `LASTSAVE` | Returns actual Unix timestamp of last snapshot |
| `BGREWRITEAOF` | `WAL.compact_rewrite()` — walks keyspace, resets WAL, rewrites fresh SET entries |

### Implementation

- `take_snapshot()` — two-pass (count entries, then write), atomic `.tmp` → rename, `fdatasync` before rename
- `load_snapshot()` — reads header, verifies magic, replays the keyspace.
  **Snapshot v2 serializes every type**: STRING, HASH, LIST, SET, ZSET, GEO,
  BITMAP, HLL, INT, FLOAT, STREAM and vector sets, plus TTLs.
- The load loop reads to **end of file**; the header's `kv_count` is only a
  runaway cap.

### Startup recovery

1. **Phase 5a:** Snapshot loaded (if exists)
2. **Phase 5b:** WAL replayed on top (delta entries since last snapshot)

---

## HNSW Persistence

Vector index persistence for warm restarts — skips the expensive `FT.OPTIMIZE` on restart.

| Operation | File | When |
|---|---|---|
| `save_to_disk("pion.hnsw.{worker_id}")` | **Index file v3**: per-node compact slots, the slot→key map, and the INT8 calibration. Older files are refused loudly and rebuilt cold. | After `FT.OPTIMIZE`, from whichever worker handled the request |
| `load_from_disk("pion.hnsw.{worker_id}")` | Reads header, restores index, sets `index_ready=True` | At startup |

Header: magic, version, num_nodes, M, ef_construction, global_min/max, etc.
The slot stride, header size and format bits live in header words 20–22, the
distance metric in word 24, and the optional-section flags (quantized indexes:
FP32 re-rank buffer, QJL signs) in word 25; words 12–19 hold the index name. Warm restore only exists for the server's startup
dimension; a config-mismatch file is refused rather than loaded.

---

## Files

| Path | Role |
|---|---|
| `src/io/wal.mojo` | WAL struct — mmap segments (rotating), append/sync/recover/checkpoint |
| `src/engine/state.mojo` | Snapshot — take_snapshot/load_snapshot integration |
| `src/vector/hnsw.mojo` | HNSW save_to_disk / load_from_disk |
| `src/ffi/uring_wrap.c` | `pion_fdatasync()` C FFI wrapper |
