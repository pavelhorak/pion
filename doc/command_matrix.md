# Pion / Redis / Valkey Command Compatibility Matrix

*Checked against Pion source (`fast_path.mojo`, `slow_path.mojo`), Redis 8, Valkey 8.x and Valkey GLIDE 1.x.*

**Legend:**
- ✅ Full support
- 🟡 Partial (notable limitations noted)
- ❌ Not implemented
- **FAST** = Pion fast path (zero-alloc, `fast_path.mojo`) — typically < 1µs dispatch
- **SLOW** = Pion slow path (`slow_path.mojo`) — RESP3 parsed, ~2–5µs additional overhead
- **GLIDE** = Valkey GLIDE 1.x client support (Java/Python/Node/Go/Rust/C# SDK)

---

## 1. String Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| GET | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| SET | ✅ | ✅ | ✅ | **FAST** | ✅ | NX/XX/EX/PX/EXAT/PXAT/KEEPTTL/GET — all verified against Redis |
| MGET | ✅ | ✅ | ✅ | **FAST** | ✅ | Batch; zero-alloc for keys ≤23 bytes |
| MSET | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| MSETNX | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| INCR | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| DECR | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| INCRBY | ✅ | ✅ | ✅ | **SLOW** | ✅ | redis-py `r.incr(key)` sends INCRBY — use Pion `INCR` directly |
| DECRBY | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| INCRBYFLOAT | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| APPEND | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| STRLEN | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| GETSET | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deprecated in Redis 6.2 (use SET ... GET) |
| GETDEL | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| GETEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SETNX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Use `SET key val NX` instead |
| SETEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Use `SET key val EX secs` instead |
| PSETEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Use `SET key val PX ms` instead |
| GETRANGE / SUBSTR | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SETRANGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| ECHO | ✅ | ✅ | ✅ | **FAST** | ✅ | Required by redis-benchmark 8.x at startup |
| MSETEX | ❌ | ❌ | ✅ | **SLOW** | ❌ | Pion extension: `MSETEX k v ttl [k v ttl ...]`, MSET with a per-pair TTL

---

## 2. Key / Expiry Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| DEL | ✅ | ✅ | ✅ | **FAST** | ✅ | Multi-key; WAL-logged (cmd_id=2) |
| EXISTS | ✅ | ✅ | ✅ | **FAST** | ✅ | Multi-key supported |
| EXPIRE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Per-worker TTL map (GenericValue.INT) |
| PEXPIRE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Millisecond precision |
| EXPIREAT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Unix timestamp seconds |
| PEXPIREAT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Unix timestamp milliseconds |
| TTL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns -1 (no TTL), -2 (not found) |
| PTTL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Millisecond precision |
| PERSIST | ✅ | ✅ | ✅ | **SLOW** | ✅ | Removes TTL |
| EXPIRETIME | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns absolute expiry time |
| PEXPIRETIME | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| UNLINK | ✅ | ✅ | ✅ | **SLOW** | ✅ | Async DEL (falls back to DEL in Redis < 7) |
| TYPE | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| RENAME | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| RENAMENX | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| COPY | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| MOVE | ✅ | ✅ | ❌ | — | ❌ | Single-DB only in GLIDE |
| OBJECT ENCODING | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| OBJECT REFCOUNT | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| OBJECT IDLETIME | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| OBJECT FREQ | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SORT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Lists only; STORE not supported |
| SORT_RO | ✅ | ✅ | ✅ | **SLOW** | ✅ | Delegates to SORT (read-only, no STORE) |
| SCAN | ✅ | ✅ | ✅ | **SLOW** | ✅ | Single-sweep; cursor=0 returns all keys |
| KEYS | ✅ | ✅ | ✅ | **SLOW** | ✅ | O(N), not safe for production use |
| RANDOMKEY | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| TOUCH | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| DUMP | ✅ | ✅ | ✅ | **SLOW** | ❌ | The payload is Pion's own format, not Redis RDB — a DUMP here restores here |
| RESTORE | ✅ | ✅ | ✅ | **SLOW** | ❌ | Accepts a Pion DUMP payload; `REPLACE` supported, `-BUSYKEY` otherwise |
| WAIT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Counts replicas that reached the offset; parks the client. 0 on a single node |
| WAITAOF | ✅ | ✅ | ✅ | **SLOW** | ✅ | `[0, 0]` on a single node |

---

## 3. Hash Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| HSET | ✅ | ✅ | ✅ | **FAST** | ✅ | Multi-field; routes vector fields to HNSW; WAL-logged |
| HGET | ✅ | ✅ | ✅ | **FAST** | ✅ | Two-level keyspace lookup |
| HMSET | ✅ | ✅ | 🟡 | **FAST** | ✅ | Maps to HSET multi-field |
| HMGET | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HGETALL | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HKEYS | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HVALS | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HLEN | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HDEL | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HEXISTS | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HINCRBY | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HINCRBYFLOAT | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HRANDFIELD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Added in Redis 6.2; no-count replies a bulk string, missing key + count an empty array |
| HSCAN | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HSETNX | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| HEXPIRE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Per-field TTL (Redis 7.4+, Valkey 8.1+). `HEXPIRE key ttl FIELDS n f...` → array of per-field codes |
| HPEXPIRE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Millisecond form of HEXPIRE |
| HTTL | ✅ | ✅ | ✅ | **SLOW** | ✅ | `-2` unknown field, `-1` no TTL, else seconds |
| HSTRLEN | ✅ | ✅ | ✅ | **SLOW** | ✅ | `0` for a missing key or field
| HPTTL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Millisecond form of HTTL
| HEXPIREAT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Absolute-deadline form; the WAL logs the resolved deadline, never the relative time
| HPEXPIREAT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Millisecond absolute form
| HEXPIRETIME | ✅ | ✅ | ✅ | **SLOW** | ✅ | Absolute deadline of a field, or `-1`/`-2`
| HPEXPIRETIME | ✅ | ✅ | ✅ | **SLOW** | ✅ | Millisecond form
| HPERSIST | ✅ | ✅ | ✅ | **SLOW** | ✅ | Removes a field TTL; `-2` unknown field, `-1` had none

---

## 4. List Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| LPUSH | ✅ | ✅ | ✅ | **FAST** | ✅ | Ziplist ≤1024 entries, Quicklist above |
| RPUSH | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| LPOP | ✅ | ✅ | ✅ | **FAST** | ✅ | `LPOP key [count]` supported (ziplist + quicklist) |
| RPOP | ✅ | ✅ | ✅ | **FAST** | ✅ | `RPOP key [count]` supported (ziplist + quicklist) |
| LRANGE | ✅ | ✅ | ✅ | **FAST** | ✅ | Ziplist skip+write phases; Quicklist 3-segment sequential scan |
| LLEN | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| LINDEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| LSET | ✅ | ✅ | ✅ | **SLOW** | ✅ | Works on ziplist and quicklist |
| LINSERT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Works on ziplist and quicklist |
| LREM | ✅ | ✅ | ✅ | **SLOW** | ✅ | Works on ziplist and quicklist |
| LTRIM | ✅ | ✅ | ✅ | **SLOW** | ✅ | Works on ziplist and quicklist |
| LPOS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Added in Redis 6.0.6 |
| LMOVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Works on ziplist and quicklist |
| LMPOP | ✅ | ✅ | ✅ | **SLOW** | ✅ | Added in Redis 7.0; ziplist only |
| BLPOP | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Pops correctly from the first non-empty key. **Does not block**: the TIMEOUT argument is parsed and ignored, so an all-empty key set answers nil immediately instead of waiting. `BLPOP k 0` will not wait for a producer |
| BRPOP | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Right-hand form; same non-blocking caveat as BLPOP |
| BLMOVE | ✅ | ✅ | ❌ | — | ✅ | Blocking |
| BLMPOP | ✅ | ✅ | ❌ | — | ✅ | Blocking |
| LPUSHX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Pushes only if the key exists; `0` otherwise
| RPUSHX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Right-hand form
| RPOPLPUSH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deprecated in Redis 6.2 for LMOVE. Validates BOTH keys before moving anything

---

## 5. Set Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| SADD | ✅ | ✅ | ✅ | **FAST** | ✅ | New-key path uses `hash_map_pool` (zero hot-path alloc) |
| SPOP | ✅ | ✅ | ✅ | **FAST** | ✅ | Supports optional count argument (SPOP key count) |
| SCARD | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SISMEMBER | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SMISMEMBER | ✅ | ✅ | ✅ | **SLOW** | ✅ | Added in Redis 7.0 |
| SMEMBERS | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SRANDMEMBER | ✅ | ✅ | ✅ | **SLOW** | ✅ | Count argument supported |
| SREM | ✅ | ✅ | ✅ | **SLOW** | ✅ | Multi-member |
| SMOVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SINTER | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SINTERSTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SINTERCARD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Added in Redis 7.0; LIMIT supported |
| SUNION | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SUNIONSTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SDIFF | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SDIFFSTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Empty result deletes the destination |
| SSCAN | ✅ | ✅ | ✅ | **SLOW** | ✅ | cursor ignored (returns full set) |

---

## 6. Sorted Set Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| ZADD | ✅ | ✅ | ✅ | **FAST** | ✅ | SlabSkipList; NX/XX/GT/LT/CH/INCR verified against Redis; equal scores order lexicographically |
| ZPOPMIN | ✅ | ✅ | ✅ | **FAST** | ✅ | COUNT supported; missing key → empty array, fractional scores exact |
| ZPOPMAX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Collect-reset-rebuild; returns removed elements with scores |
| ZRANGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Rank-based, plus the 6.2 unified form: BYSCORE/BYLEX/REV/LIMIT/WITHSCORES |
| ZRANGEBYSCORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Score range with -inf/+inf/(N exclusive; WITHSCORES/LIMIT supported |
| ZRANGEBYLEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Lex range with [/( inclusive/exclusive; LIMIT supported |
| ZREVRANGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Reverse rank traversal; WITHSCORES supported |
| ZREVRANGEBYSCORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Score range reversed; WITHSCORES/LIMIT supported |
| ZREVRANGEBYLEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Lex range reversed output |
| ZRANGESTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Rank-based copy to destination key |
| ZRANK | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns 0-based rank or nil |
| ZREVRANK | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns 0-based reverse rank or nil |
| ZSCORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns score as bulk string or nil |
| ZMSCORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Multi-member score array response |
| ZINCRBY | ✅ | ✅ | ✅ | **SLOW** | ✅ | Collect-reset-rebuild to update score |
| ZCARD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns skip list length |
| ZCOUNT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Counts members in score range |
| ZLEXCOUNT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Counts members in lex range |
| ZREM | ✅ | ✅ | ✅ | **SLOW** | ✅ | Collect-reset-rebuild; returns removed count |
| ZREMRANGEBYSCORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Collect-reset-rebuild; returns removed count |
| ZREMRANGEBYLEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Collect-reset-rebuild; returns removed count |
| ZREMRANGEBYRANK | ✅ | ✅ | ✅ | **SLOW** | ✅ | Collect-reset-rebuild; returns removed count |
| ZUNION | ✅ | ✅ | ✅ | **SLOW** | ✅ | numkeys arg; WITHSCORES supported; simple sum aggregation |
| ZUNIONSTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Stores union into destination key |
| ZINTER | ✅ | ✅ | ✅ | **SLOW** | ✅ | Intersection by member presence across all keys |
| ZINTERSTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Stores intersection into destination key |
| ZINTERCARD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Cardinality of intersection; optional LIMIT |
| ZDIFF | ✅ | ✅ | ✅ | **SLOW** | ✅ | Members in first key not in any subsequent key |
| ZDIFFSTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Stores diff into destination key |
| ZRANDMEMBER | ✅ | ✅ | ✅ | **SLOW** | ✅ | Random member(s); positive/negative count; WITHSCORES |
| ZMPOP | ✅ | ✅ | ✅ | **SLOW** | ✅ | numkeys/MIN/MAX/COUNT; nested [member, score] pairs; nil when no key has elements |
| BZPOPMIN | ✅ | ✅ | ❌ | — | ✅ | Blocking |
| BZPOPMAX | ✅ | ✅ | ❌ | — | ✅ | Blocking |
| BZMPOP | ✅ | ✅ | ❌ | — | ✅ | Blocking |
| ZSCAN | ✅ | ✅ | ✅ | **SLOW** | ✅ | Cursor ignored; returns all members with scores |

---

## 7. Bitmap Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| GETBIT | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| SETBIT | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| BITCOUNT | ✅ | ✅ | 🟡 | **FAST/SLOW** | ✅ | No-arg and full-key: FAST. Range args (BITCOUNT key start end): SLOW |
| BITPOS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Find first set/clear bit; start/end range supported |
| BITOP | ✅ | ✅ | ✅ | **SLOW** | ✅ | AND/OR/XOR/NOT; zero-pads shorter sources |
| BITFIELD | ✅ | ✅ | ✅ | **SLOW** | ✅ | GET/SET/INCRBY subcommands; u/i type prefix |
| BITFIELD_RO | ✅ | ✅ | ✅ | **SLOW** | ✅ | GET subcommand only |

---

## 8. HyperLogLog Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| PFADD | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| PFCOUNT | ✅ | ✅ | 🟡 | **FAST/SLOW** | ✅ | Single-key: FAST. Multi-key: SLOW |
| PFMERGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Merges multiple HLL keys into destination |

---

## 9. Geospatial Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| GEOADD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Backed by SlabSkipList with geohash encoding |
| GEODIST | ✅ | ✅ | ✅ | **SLOW** | ✅ | Haversine distance; m/km/mi/ft units |
| GEOPOS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Decodes geohash to lon/lat; multi-member |
| GEOSEARCH | ✅ | ✅ | ✅ | **SLOW** | ✅ | FROMMEMBER/FROMLONLAT; BYRADIUS/BYBOX; WITHCOORD/WITHDIST/COUNT |
| GEOSEARCHSTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Stores search results as GEO key |
| GEORADIUS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deprecated but supported; WITHCOORD/WITHDIST/COUNT/ASC/DESC |
| GEORADIUSBYMEMBER | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deprecated but supported; COUNT option |
| GEOHASH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns 11-char base32 geohash strings |

---

## 10. Stream Commands

Streams are fully implemented and WAL-persisted (snapshot v2 + WAL cmd 23 XADD /
27 XDEL; `INFO` reports `streams_persisted:1`). Consumer groups are the one gap —
every `X*GROUP`/pending/claim command refuses explicitly rather than faking state.

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| XADD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Real IDs (`ms-seq`, `*` auto), stored and WAL-persisted (cmd 23) |
| XREAD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns entries; `BLOCK` is parsed but never blocks (returns at once) |
| XLEN | ✅ | ✅ | ✅ | **SLOW** | ✅ | Real length |
| XRANGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Real entries, `-`/`+` bounds |
| XREVRANGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Real entries, reversed |
| XINFO STREAM | ✅ | ✅ | ✅ | **SLOW** | ✅ | `length`, `last-generated-id`, `entries` |
| XDEL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deletes by ID; WAL cmd 27 |
| XTRIM | ✅ | ✅ | ✅ | **SLOW** | ✅ | `MAXLEN` |
| XGROUP | ✅ | ✅ | ❌ | **SLOW** | ✅ | Refuses: `-ERR consumer groups not supported` |
| XREADGROUP | ✅ | ✅ | ❌ | **SLOW** | ✅ | Refuses: `-ERR consumer groups not supported` |
| XACK | ✅ | ✅ | ❌ | **SLOW** | ✅ | Refuses: `-ERR consumer groups not supported` |
| XCLAIM | ✅ | ✅ | ❌ | **SLOW** | ✅ | Refuses: `-ERR consumer groups not supported` |
| XAUTOCLAIM | ✅ | ✅ | ❌ | **SLOW** | ✅ | Refuses: `-ERR consumer groups not supported` |
| XPENDING | ✅ | ✅ | ❌ | **SLOW** | ✅ | Refuses: `-ERR consumer groups not supported` |

---

## 11. Pub/Sub Commands

Message delivery works (RESP2 and RESP3 push). The one limitation is cross-worker:
with `-w N > 1` a publisher and subscriber on different workers do not see each
other (shared-nothing keyspace), so pub/sub is coherent at `-w 1` (the default)
or when both connections land on the same worker — see operations.md §2b.

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| SUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Confirmation + live message delivery |
| UNSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| PUBLISH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns the subscriber count; delivers within a worker |
| PSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Pattern subscribe + delivery |
| PUNSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| PUBSUB | ✅ | ✅ | 🟡 | **SLOW** | ✅ | CHANNELS/NUMSUB/NUMPAT (per-worker view) |
| SSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Shard subscribe + delivery |
| SUNSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SPUBLISH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns the subscriber count |

---

## 12. Scripting & Functions

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| EVAL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Lua 5.1 scripting (sandboxed, cjson, 30 commands) |
| EVALSHA | ✅ | ✅ | ✅ | **SLOW** | ✅ | SHA1-indexed script cache |
| EVAL_RO | ✅ | ✅ | ❌ | — | ✅ | |
| EVALSHA_RO | ✅ | ✅ | ❌ | — | ✅ | |
| SCRIPT LOAD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Compile + cache, returns SHA1 |
| SCRIPT EXISTS | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SCRIPT FLUSH | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| FUNCTION LOAD | ✅ | ✅ | ✅ | **SLOW** | ✅ | `#!lua name=<lib>` shebang, `redis.register_function()`, REPLACE |
| FUNCTION LIST | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns library names, engine, function names |
| FUNCTION DELETE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Delete library by name |
| FUNCTION DUMP | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns empty (no RDB serialization) |
| FUNCTION RESTORE | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns +OK (stub) |
| FUNCTION FLUSH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Clears all libraries |
| FUNCTION STATS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns running_script count |
| FCALL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Executes registered function with KEYS/ARGV |
| FCALL_RO | ✅ | ✅ | ❌ | — | ✅ | |

### Lua Scripting Details

**Engine:** Lua 5.1.5 (PUC-Rio reference implementation), statically linked. One VM per worker (shared-nothing). Coroutine-based: `redis.call()` yields to host for dispatch — no callbacks.

**Sandbox:** Only `base`, `table`, `string`, `math`, `cjson` libraries loaded. No `io`, `os`, `debug`, `package`. Functions removed: `dofile`, `loadfile`, `loadstring`, `print` (use `redis.log` instead). Memory limit: 1 MB per execution. Instruction limit: 1M instructions per execution.

**Lua globals available:** `redis.call()`, `redis.pcall()`, `redis.log()`, `redis.error_reply()`, `redis.status_reply()`, `redis.sha1hex()`, `cjson.encode()`, `cjson.decode()`, `KEYS[]`, `ARGV[]`.

**Commands available inside `redis.call()` / `redis.pcall()`:**

| Category | Commands |
|---|---|
| String | GET, SET, DEL, EXISTS, INCR, DECR, INCRBY, DECRBY, APPEND, STRLEN, SETNX, MGET, MSET |
| Hash | HSET, HGET, HDEL, HEXISTS, HLEN |
| List | LPUSH, RPUSH, LPOP, RPOP, LLEN, LRANGE |
| Set | SADD, SREM, SISMEMBER, SCARD |
| TTL | EXPIRE, TTL, PERSIST |
| Key | TYPE, RENAME |
| Server | PING |

**Commands NOT yet available inside `redis.call()`:**

| Category | Missing commands |
|---|---|
| Hash | HGETALL, HMGET, HKEYS, HVALS, HSETNX, HINCRBY, HINCRBYFLOAT |
| List | LINDEX, LINSERT, LREM, LTRIM |
| Set | SMEMBERS, SRANDMEMBER, SPOP, SINTER, SUNION, SDIFF |
| Sorted Set | ZADD, ZREM, ZSCORE, ZRANK, ZRANGE, ZCARD, ZINCRBY, ZPOPMIN |
| String | GETSET, GETDEL, SETEX, PSETEX, INCRBYFLOAT |
| TTL | PEXPIRE, PTTL, EXPIREAT, PEXPIREAT |
| Key | KEYS, SCAN, RANDOMKEY, COPY, SORT |
| Pub/Sub | SUBSCRIBE, PUBLISH |
| Transactions | MULTI, EXEC |
| Vector/AI | FT.*, AI.*, ATTEND.* |
| Streams | XADD, XREAD, XLEN |

Unsupported commands return `-ERR unknown command '<name>'` when called from Lua.

**Not implemented (Lua features):** `EVAL_RO`/`EVALSHA_RO`, `FCALL_RO`, `FUNCTION DUMP`/`RESTORE` (stubs), `cmsgpack` library, `redis.replicate_commands()`, `redis.set_repl()`, `redis.breakpoint()`/`redis.debug()`, script replication across replicas.

---

## 13. Transaction Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| MULTI | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Per-connection tx state (`tx_in_multi[fd]`): queues commands, validates names at QUEUE time against the generated command table, and answers EXEC with -EXECABORT on an unknown one |
| EXEC | ✅ | ✅ | ✅ | **SLOW** | ✅ | Runs the queued commands atomically and returns their replies as an array; `-EXECABORT` if any was rejected at QUEUE time |
| DISCARD | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns +OK |
| WATCH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Monitors keys for changes between WATCH and EXEC; EXEC returns null if any watched key was modified. Per-fd version tracking via key_versions[65536] array, bumped by SET/DEL/HSET in fast path |
| UNWATCH | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns +OK |

---

## 14. Server / Admin Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| PING | ✅ | ✅ | ✅ | **FAST** | ✅ | Exponential-doubling batch copy for pipelined PINGs |
| FLUSHALL | ✅ | ✅ | ✅ | **FAST/SLOW** | ✅ | Clears this worker's keyspace (one keyspace by default; `-w N > 1` needs `--independent-workers`) |
| FLUSHDB | ✅ | ✅ | ✅ | **SLOW** | ✅ | Clears this worker's keyspace (same as FLUSHALL in shared-nothing) |
| DBSIZE | ✅ | ✅ | ✅ | **FAST** | ✅ | Returns sum of shard sizes for this worker |
| SELECT | ✅ | ✅ | 🟡 | **FAST** | ✅ | Returns +OK; always DB 0 in Pion (shared-nothing) |
| SWAPDB | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns +OK (no-op; single database) |
| SAVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Synchronous WAL flush |
| BGSAVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Async WAL msync |
| BGREWRITEAOF | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns +OK (WAL persists all writes; no AOF needed) |
| LASTSAVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns the real unix timestamp of the last SAVE/BGSAVE |
| INFO | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Server / Memory / Cluster / Replication / Persistence / Stats / Pion / Keyspace sections. Port, RSS, uptime and per-worker key counts are resolved from real state; `redis_version` stays `7.0.0` for client feature gating, `pion_version` carries the build. |
| PION.STATS | n/a | n/a | n/a | **SLOW** | n/a | Pion-only. The value receipt: a 16-pair map (RESP3 `%`, RESP2 flat array) — `kvprefix_hits/misses/tokens_served/bytes_served`, `prefill_seconds_avoided` (measured + estimated) and `prefill_seconds_avoided_measured` (client-reported `PREFILL_MS` only), `semantic_hits/misses`, `moe_hits/misses`, `vector_queries`. Per WORKER. `PION.STATS RESET` zeroes the counters. |
| CONFIG GET | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns real values for known keys (e.g. `maxmemory`); unknown keys reply empty |
| CONFIG SET | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Only `maxmemory` is runtime-settable; every other key refuses with an explanatory `-ERR` |
| CONFIG REWRITE | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns +OK |
| CONFIG RESETSTAT | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns +OK |
| COMMAND | ✅ | ✅ | 🟡 | **FAST** | ✅ | Bare COMMAND returns `*0` (no per-command specs) |
| COMMAND COUNT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns the generated command count (326) |
| COMMAND DOCS | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns *0 (no docs stored) |
| COMMAND INFO | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns *0 |
| DEBUG | ✅ | ✅ | 🟡 | **SLOW** | ❌ | Returns +OK (stub; no debug state) |
| SLOWLOG | ✅ | ✅ | 🟡 | **SLOW** | ✅ | GET→*0, LEN→:0, RESET→+OK |
| LATENCY | ✅ | ✅ | 🟡 | **SLOW** | ✅ | LATEST/HISTORY→*0, RESET→+OK |
| MEMORY USAGE | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns approximate byte estimate for key |
| MEMORY DOCTOR | ✅ | ✅ | 🟡 | **SLOW** | ❌ | Returns fixed health message |
| MODULE | ✅ | ✅ | 🟡 | **SLOW** | ❌ | LIST→*0, others→+OK |
| ACL | ✅ | ✅ | 🟡 | **SLOW** | ✅ | WHOAMI→default, LIST→one entry, USERS→*1, CAT/LOG→*0, others→+OK |
| RESET | ✅ | ✅ | 🟡 | **FAST** | ✅ | Returns +OK |
| QUIT | ✅ | ✅ | 🟡 | **FAST** | ✅ | Returns +OK |
| AUTH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Real auth: with `--requirepass`/`--tenant`, `AUTH <pw>` gates every command (`-NOAUTH` before, `-WRONGPASS` on a bad password); with no password set, `AUTH` replies the Redis error, not +OK |
| HELLO | ✅ | ✅ | ✅ | **SLOW** | ✅ | `HELLO 3` switches the connection to RESP3 (map/push replies); `HELLO`/`HELLO 2` stay RESP2 |
| SHUTDOWN | ✅ | ✅ | ✅ | **SLOW** | ✅ | `NOSAVE` supported. A plain SIGTERM drains the WAL first |
| XGPU | ❌ | ❌ | ✅ | **SLOW** | ❌ | Pion extension: INFO-style GPU availability block |

---

## 15. Cluster Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| CLUSTER INFO | ✅ | ✅ | ✅ | **SLOW** | ✅ | Reports cluster_enabled:0 or :1 per --cluster flag |
| CLUSTER NODES | ✅ | ✅ | ✅ | **SLOW** | ✅ | One-node cluster topology |
| CLUSTER MYID | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| CLUSTER KEYSLOT | ✅ | ✅ | ✅ | **SLOW** | ✅ | CRC16 hash slot computation |
| CLUSTER SLOTS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Legacy format (Redis ≤6 compatibility) |
| CLUSTER SHARDS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Redis 7+ format; GLIDE preferred |
| CLUSTER MEET | ✅ | ✅ | ✅ | **SLOW** | ✅ | Registers peer; Raft prototype |
| CLUSTER RESET | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| CLUSTER REPLICAS | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Lists real replicas under `--cluster`; empty on a single node |
| CLUSTER FAILOVER | ✅ | ✅ | ✅ | **SLOW** | ✅ | Promotes replica to primary; updates cluster_epoch. FORCE variant supported (skips health checks) |
| CLUSTER SETSLOT | ✅ | ✅ | ✅ | **SLOW** | ✅ | IMPORTING/MIGRATING/STABLE/NODE subcommands for slot ownership transfer |
| CLUSTER GETKEYSINSLOT | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns *0 (slot-based key scan not implemented) |
| CLUSTER COUNTKEYSINSLOT | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns :0 |
| CLUSTER STATS | ✅ | n/a | n/a | **SLOW** | n/a | Pion-only. INFO-style bulk-string telemetry: `cluster_stats_local_worker`, `cluster_stats_total_workers`, `kv_prefix_active_total` + per-worker `kv_prefix_active_worker_<id>` (cross-worker via shared directory ACQUIRE), `local_vstore_sessions`, `local_attend_*` (legacy HNSW path; ATTEND.PREFIX.* counters live C-side, scoped out). |
| ASKING | ✅ | ✅ | ✅ | **FAST** | ✅ | One-shot redirect acknowledgement during slot migration |
| READONLY | ✅ | ✅ | ✅ | **FAST** | ✅ | Accepted; Pion has no replica-read split, so it is a no-op that keeps cluster clients happy |
| READWRITE | ✅ | ✅ | ✅ | **FAST** | ✅ | Inverse of READONLY; also a no-op |
| MIGRATE | ✅ | ✅ | ✅ | **SLOW** | ✅ | DUMP + RESTORE over a socket, then DEL. `COPY` / `REPLACE` / `KEYS` supported |
| PSYNC | ✅ | ✅ | 🟡 | **SLOW** | n/a | Replication handshake; worker-0 only |
| REPLCONF | ✅ | ✅ | 🟡 | **SLOW** | n/a | Replication handshake sub-negotiation |

---

## 16. Pion-Native AI Commands (no Redis / Valkey equivalent)

| Command | Pion path | GLIDE | Description |
|---|:---:|:---:|---|
| FT.CREATE | **SLOW** | ❌ | Create vector index (HNSW, SCHEMA, VECTOR field) |
| FT.SEARCH | **SLOW** | ❌ | KNN vector search; PARAMS format; `BM25 <q> [K k] [K1 f] [B f]`; HYBRID (vector+text); TAG/FILTER predicates |
| FT.OPTIMIZE | **SLOW** | ❌ | Build l0_compact, compact_buffer; save HNSW to disk (`pion.hnsw.0`) |
| FT.DROPINDEX | **SLOW** | ❌ | Drop vector index; clears in-memory HNSW. An unknown name is refused, nothing dropped |
| FT.INFO | **SLOW** | ❌ | `[index_name, <name>, num_docs, <n>]`; checks the name against the shared registry, same answer on every worker; unknown name → `Unknown index name` |
| FT.HYBRID | **SLOW** | ❌ | `FT.HYBRID <index> <text_query> <vector_blob> [K <k>] [ALPHA <a>] [K1 <f>] [B <f>]` — BM25+vector fusion with Reciprocal Rank Fusion |
| FT.ADDTEXT | **SLOW** | ❌ | Add text doc: store in keyspace + BM25 doc set (integer ids), embed via HTTP → insert into semantic HNSW |
| FT.SEARCHTEXT | **SLOW** | ❌ | Semantic text search over docs added via FT.ADDTEXT |
| AI.COMPLETE | **SLOW** | ❌ | Semantic cache check → LLM call on miss → cache result |
| AI.SEMANTIC_CACHE GET | **SLOW** | ❌ | Embed query → HNSW search → return cached response if cos sim ≥ threshold |
| AI.SEMANTIC_CACHE SET | **SLOW** | ❌ | Embed query → insert into HNSW → store response |
| AI.CHAT | **SLOW** | ❌ | Full RAG in one command: retrieve context → build prompt → call LLM |
| AI.FLARE LOAD | **SLOW** | ❌ | Add text to in-Mojo FLARE knowledge base |
| AI.FLARE RUN | **SLOW** | ❌ | Mid-generation retrieval loop (FLARE algorithm in Mojo) |
| AI.FLARE INFO | **SLOW** | ❌ | FLARE KB stats |
| AI.KNN_LM.CREATE | **SLOW** | ❌ | Allocate token-id-tagged kNN datastore: `<ds_id> <dim> [<max_entries>]` (default max=100K). Up to 16 datastores per worker. Substrate enabled when `--kvcache` or `--inference` is on. |
| AI.KNN_LM.STORE | **SLOW** | ❌ | Append one (token_id, embedding) pair: `<ds_id> <next_token_id> <emb_blob>`. Auto-builds HNSW once count crosses 5000. |
| AI.KNN_LM.STOREBATCH | **SLOW** | ❌ | Bulk append: `<ds_id> <n> <ids_blob> <emb_blob>` (ids: n × Int32 LE; emb: n × dim × Float32 LE). |
| AI.KNN_LM.QUERY | **SLOW** | ❌ | Top-k kNN: `<ds_id> <k> <emb_blob>`. Returns `k × 8 bytes` packed as `<Int32 LE token_id><Float32 LE distance>`. Brute-force scan below 5K, focused FP32 HNSW above. Sub-linear scaling, ~0.94 ms median at 30K entries. |
| AI.KNN_LM.INFO | **SLOW** | ❌ | `count=N dim=D max_entries=M` |
| AI.KNN_LM.DROP | **SLOW** | ❌ | Free datastore + HNSW graph buffers |
| NEURON.PKM.CREATE | **SLOW** | ❌ | Allocate a product-key memory table: `<table> <dim> <n_slots> [VDIM <v>] [VALTYPE F32\|F16]`. `n_slots` must be a perfect square S² (S ≤ 4096); dim must be even. Up to 8 tables per worker. Enabled by `--kvcache` / `--inference`. |
| NEURON.PKM.SETKEYS | **SLOW** | ❌ | Load codebook half 0 or 1: `<table> <half> <blob>` — S × (dim/2) Float32 LE. Builds the INT8 mirror. Both halves required before QUERY. |
| NEURON.PKM.SETVALS | **SLOW** | ❌ | Write value rows: `<table> <off> <n> <blob>` — n × vdim in VALTYPE. Value matrix is allocated on the first call. |
| NEURON.PKM.QUERY | **SLOW** | ❌ | **Exact** top-k over all n_slots: `<table> <k> <q_blob> [FAST]`. `nq = len(q_blob)/(dim·4)` heads share one codebook pass. Returns `nq·k × 8 bytes` as `<Int32 LE slot_id><Float32 LE score>`, descending, padded `(-1, -inf)`. dim=896 k=32 1M slots: **0.073 ms** (vs 5.13 ms for AI.KNN_LM.QUERY at the same shape). |
| NEURON.PKM.FFN | **SLOW** | ❌ | Fused lookup + softmax-weighted value read: `<table> <k> <q_blob> [FAST] [TEMP <t>]` → `nq × vdim × 4 bytes` Float32 LE. Value rows never cross the wire. |
| NEURON.PKM.INFO | **SLOW** | ❌ | `dim=D half=H s_rows=S n_slots=N d_pad=P keys_ready=0\|1 vdim=V valtype=f32\|f16 val_rows=R queries=Q` |
| NEURON.PKM.DROP | **SLOW** | ❌ | Free codebooks, value matrix, and query scratch |
| AI.EMBED | **SLOW** | ❌ | Embed text with the in-process model. `-ERR AI.EMBED requires --inference or --emb-enabled` when no backend is active |
| AI.GENERATE | **SLOW** | ❌ | Generate via the sidecar; `-ERR ... requires --inference` otherwise |
| AI.LOADMODEL | **SLOW** | ❌ | Load a sidecar model; `-ERR ... requires --inference` otherwise |
| AI.MEMORY | **SLOW** | ❌ | `AI.MEMORY ADD\|RECALL\|CONTEXT ...` — agent-memory surface; a bare call returns the syntax line |

---

## 16b. MoE Expert Paging — `MOE.EXPERT.*` (`--moe-cache <DIR> --moe-cache-mib N`)

Expert weights served from a tiered cache (per-worker RAM LRU → SSD → network)
so a MoE model larger than device RAM runs at interactive cache-hit latency.
Without `--moe-cache` every command answers `-UNAVAILABLE` naming the flag,
rather than a plausible empty result.

| Command | Pion path | Reply | Notes |
|---|:---:|---|---|
| MOE.EXPERT.LOAD `<model_id> <dir>` | **SLOW** | `+OK` / `-UNAVAILABLE` | Loads a manifest and opens the tier |
| MOE.EXPERT.FETCH `<model_id> <layer> <expert>` | **SLOW** | bulk blob / `-UNAVAILABLE` | ~5 ms on a cache hit regardless of backing tier |
| MOE.EXPERT.PREFETCH `<model_id> <layer> <expert>...` | **SLOW** | `+OK` | Asynchronous warm; returns before the fetch completes |
| MOE.EXPERT.PIN `<model_id> <layer> <expert>` | **SLOW** | `+OK` | Protects an expert from LRU eviction |
| MOE.EXPERT.UNPIN `<model_id> <layer> <expert>` | **SLOW** | `+OK` | |
| MOE.EXPERT.INFO `<model_id>` | **SLOW** | JSON / `-UNAVAILABLE` | Per-model manifest |
| MOE.EXPERT.STATS | **SLOW** | JSON | Always available, even with the tier disabled — `{"stage":1,"enabled":false,...}` |
| MOE.EXPERT.HIST `<model_id>` | **SLOW** | JSON / `-UNAVAILABLE` | Per-distribution access histograms |
| MOE.EXPERT.PRUNE `<model_id> <layer> <expert> [on]` | **SLOW** | `+OK` / `-ERR` | **HIST namespaces are per-distribution but PRUNE is global.** A prune driven by one narrow histogram wrecks perplexity on other distributions — use the union-safe client policy |

## 16c. Fixed-Size State Cache — `STATE.*` (`--kvcache`)

A distinct surface from V-store: a fixed-size, overwrite-in-place buffer for
per-request recurrent state, not an evicting cache. Every command answers
`-ERR state cache not enabled (use --kvcache)` when the flag is absent.

| Command | Pion path | Notes |
|---|:---:|---|
| STATE.ALLOC `<id> <bytes>` | **SLOW** | Reserve a fixed-size slot |
| STATE.WRITE `<id> <offset> <data>` | **SLOW** | Overwrite in place; no growth |
| STATE.READ `<id> <offset> <len>` | **SLOW** | |
| STATE.FREE `<id>` | **SLOW** | |
| STATE.INFO | **SLOW** | Slot count and occupancy |

## 17. Externalized Attention Commands (`--kvcache`)

RESP commands on port 1974:

| Command | Pion path | GLIDE | Description |
|---|:---:|:---:|---|
| KV.STORE ⚠️ | **SLOW** | ❌ | Store KV cache tensor with HNSW-indexed embedding key. **Experimental — TTL/MODEL stubs, capacity-capped 1000 entries, no LRU.** See the external-parametric-memory Stage-2 reframe results §5. |
| KV.FETCH ⚠️ | **SLOW** | ❌ | Fetch nearest cached tensor by cosine similarity. **Experimental — MODEL arg silently ignored on FETCH.** |
| KV.INFO | **SLOW** | ❌ | KV cache store statistics (entries, blob bytes, hits, misses, capacity) |
| ~~KV.EVICT~~ | _none_ | ❌ | **Not implemented** — documented in source comments only, no handler. |
| KV.PREFIX.REGISTER | **SLOW** | ❌ | Register namespace for shared prefix KV cache; creates `<ns>_pk` and `<ns>_pv` V-store sessions. Optional `PREFILL_MS <ms>` reports the client's measured cold prefill for the value receipt. See `doc/shared_kv_cache.md`. |
| KV.PREFIX.LOOKUP | **SLOW** | ❌ | Returns `+HIT` if both K and V sessions exist for the namespace, `+MISS` otherwise |
| KV.PREFIX.INFO | **SLOW** | ❌ | Global stats: registered prefix count, total tokens, total fetches |
| V.CREATE | **SLOW** | ❌ | Create V-store session for token-ID-indexed value storage (used by KV.PREFIX.* and externalized attention) |
| V.STOREBATCH | **SLOW** | ❌ | Append a batch of FP32 values quantized to session format (int8/turbo4/turbo3/turbo2/fp16) |
| V.FETCH | **SLOW** | ❌ | Fetch values by token ID list, or contiguous range via `RANGE start end` (single round-trip per layer; bypasses 64-token RESP frame limit) |
| KV.PREFIX.BLOCKS | **SLOW** | ❌ | Per-block visibility for a stored prefix |
| KV.PREFIX.MEMBERSHIP | **SLOW** | ❌ | Which blocks of a prefix are resident |
| KV.PREFIX.OWNER | **SLOW** | ❌ | Which worker holds a prefix. `-w N` is N independent keyspaces, so this is how a client finds the right one |
| KV.PREFIX.WARM | **SLOW** | ❌ | Promote a cold-tier prefix back to RAM |
| KV.PREFIX.SAVE | **SLOW** | ❌ | Persist a prefix to the cold tier |
| V.SNAPSHOT | **SLOW** | ❌ | Point-in-time V-store snapshot |
| V.COMMIT | **SLOW** | ❌ | Seal a snapshot |
| V.RESTORE | **SLOW** | ❌ | Reload a sealed snapshot |
| V.INFO | **SLOW** | ❌ | Per-session or global V-store statistics |
| ATTEND.PREFIX.STORE | **SLOW** | ❌ | Push K/V to native Metal SDPA session cache; resident until DROP / LRU eviction. Body: `<session_id> <layer_id> <H> <N> <D> <K_blob> <V_blob>`. See `doc/shared_kv_cache.md` Stage 2. |
| ATTEND.PREFIX.LOOKUP | **SLOW** | ❌ | Probe the Metal session cache for `(sid, layer_id)`. Returns `+HIT` if slot exists with non-nil K/V, `+MISS` otherwise. Symmetric to `KV.PREFIX.LOOKUP` for V-store state. Required for stage-2-aware client `PionPromptCache.lookup` to avoid stale V-store hits causing skipped `_stage2_push_cold`. |
| ATTEND.PREFIX.QUERY | **SLOW** | ❌ | Run M=1 single-query attention on cached K/V (Q-only on wire). Body: `<session_id> <layer_id> <H> <D> <top_k> <Q_blob>`. Returns `H*D` float32. Native Metal SDPA path (sdpa_q1_fp32/fp16); supports D ∈ {32,64,96,128,160,192,256,**512**}. 146× faster than ATTEND.QUERYBATCH at H=8 N=2048. |
| ATTEND.PREFIX.QUERY_FUSED | **SLOW** | ❌ | M>=1 fused suffix-SDPA + prefix-merge in one dispatch. Body: `<sid> <layer> <H_q> <D> <S_suf> <H_kv> <Q> <K_suf> <V_suf> <head_map> [<fa_window>]`. GQA-aware. Returns merged attention `H_q*M*D*4` bytes; no LSE trailer (merge fused). |
| ATTEND.PREFIX.QUERY_SPARSE | **SLOW** | ❌ | Sparse-mask M=1 attention with caller-supplied per-head indices. Body: `<sid> <layer> <H> <D> <K_sparse_max> <Q> <indices> <counts> [<fa_window>]`. For learned-router v2 consumers. Output `H*D*4` bytes. |
| ATTEND.PREFIX.QUERY_SPARSE_AUTO | **SLOW** | ❌ | Sparse-mask M=1 with server-side block-mean top-K selection. Body: `<sid> <layer> <H_q> <D> <B> <K_top> <H_kv> <Q> <head_map> [<fa_window>]`. K_mean cached server-side per slot. GQA-aware. Output `H_q*D*4` bytes. |
| ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED | **SLOW** | ❌ | Sparse-AUTO + dense-suffix + online-softmax merge in one fused dispatch. Body: `<sid> <layer> <H_q> <D> <B> <K_top> <H_kv> <S_suf> <Q> <K_suf> <V_suf> <head_map> [<fa_window>]`. Closes the "ignore suffix" v1 wire-mode-sparse caveat. Output `H_q*D*4` bytes. |
| SSM.PREFIX.STORE | **SLOW** | ❌ | Store opaque byte blob (serialized SSM recurrent state — Mamba `[conv_state, ssm_state]`, RWKV-7 size=3 tuple, or any future linear-attention family). Body: `<session_id> <layer_id> <state_blob>`. Server never parses the blob. |
| SSM.PREFIX.FETCH | **SLOW** | ❌ | Retrieve previously-stored blob for `(sid, layer_id)`. Returns bulk string or `$-1` (miss). |
| SSM.PREFIX.DROP | **SLOW** | ❌ | Drop stored blob. Body: `<session_id> [<layer_id>]`. Layer omitted = drop all layers for the session. Idempotent. |
| ATTEND.CREATE | **SLOW** | ❌ | Create attention session with key_dim and value_dim |
| ATTEND.STORE | **SLOW** | ❌ | Stage token KV pairs (FP32 memcpy); 3.67M tok/s via binary protocol |
| ATTEND.FINALIZE | **SLOW** | ❌ | Batch build HNSW index from staged keys |
| ATTEND.QUERY | **SLOW** | ❌ | Top-k HNSW search over attention keys; 86us per layer at 128K tokens |
| ATTEND.INFO | **SLOW** | ❌ | Index statistics (sessions, tokens, queries) |

Binary protocol commands on port 1975 (0xCA5E framing):

| Command byte | Command | Description |
|:---:|---|---|
| 0x01 | LAYER.STORE | Store per-layer tensor (binary blob) |
| 0x02 | LAYER.FETCH | Fetch per-layer tensor |
| 0x03 | PING | Binary keepalive |
| 0x20 | ATTEND.CREATE | Create attention session (binary) |
| 0x21 | ATTEND.STORE | Stage token KV pairs (3.67M tok/s) |
| 0x22 | ATTEND.FINALIZE | Batch build HNSW from staged keys |
| 0x23 | ATTEND.QUERY | Top-k HNSW query (86us at 128K) |

Requires `--kvcache` flag. Python client: `vllm-pion/` package (`PionKVClient`, `PionAttentionClient`, `ExternalizedAttentionLayer`).

---

## 18. Semantic Router Commands (`--kvcache`)

| Command | Pion path | GLIDE | Description |
|---|:---:|:---:|---|
| AI.ROUTE.REGISTER | **SLOW** | ❌ | Register inference node with semantic centroid embedding + optional CAPACITY |
| AI.ROUTE.UPDATE | **SLOW** | ❌ | Update a node's centroid embedding (as KV cache evolves) |
| AI.ROUTE | **SLOW** | ❌ | Route query embedding to best node by cosine similarity; 0.14ms, 7K QPS |
| AI.ROUTE.REMOVE | **SLOW** | ❌ | Remove node from routing table |
| AI.ROUTE.INFO | **SLOW** | ❌ | Per-node stats: routed count, capacity, endpoint |

Routing strategy: FP32 brute-force cosine for <=16 nodes (perfect accuracy); HNSW O(log N) for >16 nodes. 88% routing accuracy with Ollama embeddings (7 of 8 test queries).

Requires `--kvcache` flag.

---

## 19. Speculative RAG Commands (`--kvcache`)

| Command | Pion path | GLIDE | Description |
|---|:---:|:---:|---|
| RAG.SPECULATE.ENABLE | **SLOW** | ❌ | Enable speculative RAG for a session (creates trajectory ring buffer of last 10 embeddings) |
| RAG.QUERY | **SLOW** | ❌ | Query with speculation: check pre-computed cache first (cosine > 0.9), fall back to live HNSW search |
| RAG.SPECULATE.INFO | **SLOW** | ❌ | Per-session stats: hit rate, trajectory length, predictions outstanding |

Prediction model: `predicted = current + alpha * (current - previous)`, alpha in [0.5, 1.0, 1.5] (3 predictions per query). 75% hit rate on linear trajectory, 0.2ms prediction+lookup latency.

Requires `--kvcache` flag.

---

## 20. VSET Commands (Redis 8 vector sets — one set per key)

| Command | Redis 8 | Valkey 8 | Pion | Pion path | Notes |
|---|:---:|:---:|:---:|:---:|---|
| VADD | ✅ | ❌ | ✅ | **SLOW** | `VADD key (FP32 <blob> \| VALUES n v…) <element> [SETATTR json]` — `:1` new, `:0` updated; Q8/NOQUANT/BIN/M/EF/CAS accepted (no effect); REDUCE refused |
| VSIM | ✅ | ❌ | ✅ | **SLOW** | `VSIM key (ELE e \| FP32 blob \| VALUES n v…) [WITHSCORES] [WITHATTRIBS] [COUNT n] [EPSILON d]` — exact scan; score `(1+cos)/2`; FILTER refused |
| VCARD | ✅ | ❌ | ✅ | **SLOW** | Elements in the set; `0` for a missing key |
| VDIM | ✅ | ❌ | ✅ | **SLOW** | The set's dimension (fixed by its first VADD) |
| VINFO | ✅ | ❌ | ✅ | **SLOW** | Redis's nine fields; `quant-type f32`, graph fields 0 (no graph) |
| VISMEMBER | ✅ | ❌ | ✅ | **SLOW** | `1` / `0` |
| VSETATTR | ✅ | ❌ | ✅ | **SLOW** | `1` set, `0` missing key or element; empty string removes |
| VGETATTR | ✅ | ❌ | ✅ | **SLOW** | JSON string or nil |
| VEMB | ✅ | ❌ | ✅ | **SLOW** | The stored vector as floats; RAW refused |
| VRANDMEMBER | ✅ | ❌ | ✅ | **SLOW** | Random element(s); count>0 distinct, count<0 with repeats |
| VREM | ✅ | ❌ | ✅ | **SLOW** | `1` / `0`; the last element removes the key |
| VRANGE | ✅ | ❌ | ✅ | **SLOW** | `VRANGE key start end [count]` — lexicographic (`-`, `+`, `[x`, `(x`) |
| VLINKS | ✅ | ❌ | — | **SLOW** | Refused: an exact set has no HNSW graph |

Vector sets are separate from `FT.*` indexes and need no FT.CREATE. Persisted like every other type: VADD/VREM/VSETATTR are WAL-logged and SAVE / BGREWRITEAOF serialize the set.

---

## Summary Statistics

| Product | Commands FAST | Commands SLOW | Total supported (fast+slow) | Total Redis 8 commands |
|---|:---:|:---:|:---:|:---:|
| **Pion** | 33 | ~183 | ~216 (+36 Pion-native) | ~280 |
| **Redis 8** | — | — | ~280 | 280 |
| **Valkey 8** | — | — | ~275 | — |

*Note: "Total supported" counts all commands with ✅ or 🟡 status. Pion-native AI commands (sections 16-19) have no Redis equivalent and are counted separately.*

### Pion Fast Path (33 commands, zero-alloc dispatch)
`GET` `SET` `MGET` `MSET` `INCR` `DECR` `HSET` `HGET` `LPUSH` `RPUSH` `LPOP` `RPOP` `LRANGE` `LLEN` `DEL` `EXISTS` `SADD` `SPOP` `ZADD` `ZPOPMIN` `PING` `FUNCTION LOAD` `FCALL` `GETBIT` `SETBIT` `BITCOUNT`(no-arg) `PFADD` `PFCOUNT`(single-key) `ECHO` `TYPE` `SELECT` `CLIENT`(ID/SETNAME/GETNAME/NO-EVICT) `COMMAND` `DBSIZE` `QUIT` `RESET` (33 commands total)

### Pion Coverage by Category

| Category | Supported / Total Redis | Coverage |
|---|:---:|:---:|
| String | 21 / 21 | 100% |
| Key / Expiry | 23 / 26 | 88% |
| Hash | 13 / 16 | 81% |
| List | 10 / 14 | 71% |
| Set | 17 / 17 | 100% |
| Sorted Set | 29 / 32 | 91% |
| Bitmap | 7 / 7 | 100% |
| HyperLogLog | 3 / 3 | 100% |
| Geo | 8 / 8 | 100% |
| Streams | 14 / 20 | base commands ✓ + WAL-persisted; consumer groups refuse |
| Pub/Sub | 9 / 9 | delivery works; cross-worker is the only limit (`-w 1` default) |
| Scripting | 2 / 16 | 13% |
| Transactions | 5 / 5 | 100% |
| Server/Admin | 31 / 31 | 100% (a few, e.g. DEBUG, are accepted no-ops) |
| Cluster | 13 / 13 | 100% |
| **AI (Pion-native)** | **23 / 23** | **100%** |

---

## Known Compatibility Pitfalls

| Issue | Details |
|---|---|
| `r.incr(key)` → INCRBY | Python redis-py sends `INCRBY key 1`, not `INCR key`. Both work; `INCR` takes the fast path. |
| Multi-worker + AI | When `--flare` / `--emb-enabled` is set, Pion auto-caps to `-w 1`. SemanticCache is per-worker and cannot share state across workers. |
| SCAN cursor semantics | SCAN is implemented but returns all keys in a single sweep (cursor always returns 0 on second call). Applications that rely on incremental cursor-based iteration may need adjustment. |
| Blocking commands | BLPOP/BRPOP pop correctly but **do not block**: an all-empty key set answers nil at once. The other blocking forms are not implemented. Poll with LPOP/RPOP/ZPOPMIN instead. |
| Lua scripting | EVAL/EVALSHA/SCRIPT + FUNCTION LOAD/LIST/DELETE/FLUSH/STATS + FCALL fully implemented with Lua 5.1.5 VM (sandboxed, cjson). `redis.call()` supports 30 commands (see table above). FUNCTION DUMP/RESTORE are stubs. |

---

## Valkey GLIDE Compatibility Notes

GLIDE connects in cluster or standalone mode. With Pion:
- **Standalone mode** works today: `GlideClient.create(GlideClientConfiguration(..., port=1974))`
- **Cluster mode** (`GlideClusterClient`) requires `--cluster` flag on Pion; sends CLUSTER SLOTS/SHARDS at connect time; Pion responds correctly
- **RESP3 negotiation** (`HELLO 3`): supported; the connection switches to RESP3 replies
- **Auth** (`AUTH`): supported when the server runs with `--requirepass`

---

## References

- Pion fast path: `src/network/fast_path.mojo`
- Pion slow path: `src/network/slow_path.mojo`
- Vector search protocol: [Vector Engine](vector_engine.md)
- AI gateway commands: [AI Gateway](ai_gateway.md)
- Client integration: [Client APIs](client_apis.md)
- Redis 8.0 command reference: https://redis.io/commands/
- Valkey 8.x command reference: https://valkey.io/commands/
- Valkey GLIDE API: https://github.com/valkey-io/valkey-glide
