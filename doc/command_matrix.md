# Pion / Redis / Valkey Command Compatibility Matrix

*Checked against Pion source (`fast_path.mojo`, `slow_path.mojo`), Redis 8, Valkey 8.x and Valkey GLIDE 1.x.*

**Legend:**
- ✅ Full support
- 🟡 Partial (notable limitations noted)
- ❌ Not implemented
- **FAST** = Pion fast path (zero-alloc, `fast_path.mojo`)
- **SLOW** = Pion slow path (`slow_path.mojo`) — RESP3 parsed into a token table first
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
| INCRBYFLOAT | ✅ | ✅ | ✅ | **SLOW** | ✅ | In long double and printed `%.17Lf`, as Redis: the platform's long double, so Linux and macOS answer as their Redis does (#35) |
| APPEND | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| STRLEN | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| GETSET | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deprecated in Redis 6.2 (use SET ... GET) |
| GETDEL | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| GETEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SETNX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Use `SET key val NX` instead |
| SETEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Use `SET key val EX secs` instead |
| PSETEX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Use `SET key val PX ms` instead |
| GETRANGE / SUBSTR | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| LCS | ✅ | ✅ | ✅ | **SLOW** | ✅ | `LEN`, `IDX`, `MINMATCHLEN`, `WITHMATCHLEN`; Redis's algorithm and errors, including the 512 MB limit on its table (#39) |
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
| PERSIST | ✅ | ✅ | ✅ | **SLOW** | ✅ | Removes TTL; 0 for an expired key (#45) |
| EXPIRETIME | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns absolute expiry time |
| PEXPIRETIME | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| UNLINK | ✅ | ✅ | ✅ | **SLOW** | ✅ | Async DEL (falls back to DEL in Redis < 7) |
| TYPE | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| RENAME | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| RENAMENX | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| COPY | ✅ | ✅ | ✅ | **SLOW** | ✅ | REPLACE; `DB 0` only — one database, as Redis with `databases 1` |
| MOVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | One database, as Redis with `databases 1`: DB 0 is the source itself, any other index is out of range |
| OBJECT ENCODING | ✅ | ✅ | ✅ | **SLOW** | ✅ | By Redis 8's size rules: int / embstr (≤44 bytes) / raw; a list is listpack while its elements fit 8 KB, else quicklist; hash listpack up to 512 short entries, set intset up to 512 integers or listpack up to 128, sorted set listpack up to 128; stream. Redis also keeps an encoding a value grew into, which Pion does not track (#47) |
| OBJECT REFCOUNT | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| OBJECT IDLETIME | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Refuses: Pion keeps no per-key access time (it answered 0) (#47) |
| OBJECT FREQ | ✅ | ✅ | ✅ | **SLOW** | ✅ | Refuses, as Redis does without an LFU maxmemory policy (Pion has none). On a missing key every OBJECT subcommand answers nil |
| SORT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Lists, sets and sorted sets; BY/LIMIT/GET/ASC/DESC/ALPHA/STORE |
| SORT_RO | ✅ | ✅ | ✅ | **SLOW** | ✅ | SORT without STORE |
| SCAN | ✅ | ✅ | ✅ | **SLOW** | ✅ | MATCH/COUNT/TYPE; single sweep (cursor 0 returns every key, cursor "0" back) |
| KEYS | ✅ | ✅ | ✅ | **SLOW** | ✅ | O(N), not safe for production use; skips expired keys (#45) |
| RANDOMKEY | ✅ | ✅ | ✅ | **SLOW** | ✅ | A random key, never an expired one (#45; it returned the first key in shard order) |
| TOUCH | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| DUMP | ✅ | ✅ | ✅ | **SLOW** | ❌ | Every type, any size (#41). The payload is Pion's records with Redis's footer (format version + CRC-64): Redis refuses it, and Pion refuses Redis's, each with `DUMP payload version or checksum are wrong` |
| RESTORE | ✅ | ✅ | ✅ | **SLOW** | ❌ | As Redis: `REPLACE`, `ABSTTL`, `IDLETIME`/`FREQ` (checked; Pion keeps no LRU/LFU data), the TTL argument, and Redis's errors in Redis's order. Logged, so a restored key survives a restart (#41) |
| WAIT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Counts replicas that reached the offset; parks the client. 0 on a single node |
| WAITAOF | ✅ | ✅ | ✅ | **SLOW** | ✅ | `[0, 0]` on a single node |

**Expiry (#45).** As in Redis, a key past its deadline is gone for every command, read or write. Each lookup compares the key's deadline with one clock per dispatch batch (Redis's command time snapshot), and removes the key when the deadline has passed. A background sweep removes keys nobody reads (`DEBUG SET-ACTIVE-EXPIRE 0` turns it off). Both log the removal as a DEL in the WAL, so a key created again after it expired survives a restart as the new key. On a replica an expired key is hidden, never removed: the primary's DEL removes it, as on a Redis replica. `DBSIZE` and `INFO`'s key count include expired keys until they are removed, as Redis's do.

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
| HINCRBYFLOAT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Long double, `%.17Lf`, Redis's argument order (#35) |
| HRANDFIELD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Added in Redis 6.2; no-count replies a bulk string, missing key + count an empty array |
| HSCAN | ✅ | ✅ | ✅ | **SLOW** | ✅ | MATCH/COUNT/NOVALUES |
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
| LMPOP | ✅ | ✅ | ✅ | **SLOW** | ✅ | Added in Redis 7.0; Redis's argument rules (#32) |
| BLPOP | ✅ | ✅ | ✅ | **SLOW** | ✅ | Blocks until a push or the timeout (seconds, 0 = forever); clients on a key are served in the order they blocked |
| BRPOP | ✅ | ✅ | ✅ | **SLOW** | ✅ | Right-hand form of BLPOP |
| BLMOVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Blocks while the source is empty |
| BLMPOP | ✅ | ✅ | ✅ | **SLOW** | ✅ | Blocks while every key is empty |
| LPUSHX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Pushes only if the key exists; `0` otherwise
| RPUSHX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Right-hand form
| RPOPLPUSH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deprecated in Redis 6.2 for LMOVE. Validates BOTH keys before moving anything
| BRPOPLPUSH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deprecated in Redis 6.2 for BLMOVE; blocks while the source is empty |

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
| ZRANK | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns 0-based rank or nil; `WITHSCORE` returns [rank, score] |
| ZREVRANK | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns 0-based reverse rank or nil; `WITHSCORE` returns [rank, score] |
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
| BZPOPMIN | ✅ | ✅ | ✅ | **SLOW** | ✅ | Blocks until a ZADD or the timeout |
| BZPOPMAX | ✅ | ✅ | ✅ | **SLOW** | ✅ | Blocks until a ZADD or the timeout |
| BZMPOP | ✅ | ✅ | ✅ | **SLOW** | ✅ | Blocks while every key is empty |
| ZSCAN | ✅ | ✅ | ✅ | **SLOW** | ✅ | Cursor ignored; returns all members with scores |

---

## 7. Bitmap Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| GETBIT | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| SETBIT | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| BITCOUNT | ✅ | ✅ | ✅ | **FAST/SLOW** | ✅ | Whole key: FAST. Ranges in BYTE or BIT units: SLOW |
| BITPOS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Ranges in BYTE or BIT units (#31) |
| BITOP | ✅ | ✅ | ✅ | **SLOW** | ✅ | AND/OR/XOR/NOT and Redis 8.2's DIFF/DIFF1/ANDOR/ONE; an empty result deletes the destination (#31) |
| BITFIELD | ✅ | ✅ | ✅ | **SLOW** | ✅ | GET/SET/INCRBY, OVERFLOW WRAP/SAT/FAIL, `#N` offsets, all-or-nothing (#31) |
| BITFIELD_RO | ✅ | ✅ | ✅ | **SLOW** | ✅ | GET only |

---

## 8. HyperLogLog Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| PFADD | ✅ | ✅ | ✅ | **FAST** | ✅ | |
| PFCOUNT | ✅ | ✅ | 🟡 | **FAST/SLOW** | ✅ | Single-key: FAST. Multi-key: SLOW |
| PFMERGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Merges multiple HLL keys into destination |
| PFSELFTEST | ✅ | ✅ | ✅ | **SLOW** | ❌ | Redis's checks against Pion's implementation: the counting kernel against a scalar reference, then the error bound at every power of ten up to 10M elements (#39) |
| PFDEBUG | ✅ | ✅ | ✅ | **SLOW** | ❌ | `GETREG` returns Pion's 16,384 registers. Pion keeps every HyperLogLog dense: `ENCODING` is `dense`, `TODENSE` is 0, `DECODE` answers Redis's error for a dense one. The register values differ from Redis's for the same elements (another hash: the HyperLogLog fence) (#39) |

---

## 9. Geospatial Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| GEOADD | ✅ | ✅ | ✅ | **SLOW** | ✅ | NX/XX/CH; a geo key is a sorted set, as in Redis (#33) |
| GEODIST | ✅ | ✅ | ✅ | **SLOW** | ✅ | m/km/ft/mi |
| GEOPOS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Coordinates printed as Redis prints them |
| GEOSEARCH | ✅ | ✅ | ✅ | **SLOW** | ✅ | FROMMEMBER/FROMLONLAT, BYRADIUS/BYBOX, ASC/DESC, COUNT [ANY], WITHCOORD/WITHDIST/WITHHASH (#33) |
| GEOSEARCHSTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | STOREDIST; an empty result deletes the destination |
| GEORADIUS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deprecated but supported, every option incl. STORE/STOREDIST/ANY/WITHHASH |
| GEORADIUSBYMEMBER | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deprecated but supported, as GEORADIUS |
| GEORADIUS_RO | ✅ | ✅ | ✅ | **SLOW** | ✅ | GEORADIUS without STORE |
| GEORADIUSBYMEMBER_RO | ✅ | ✅ | ✅ | **SLOW** | ✅ | GEORADIUSBYMEMBER without STORE |
| GEOHASH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Returns 11-char base32 geohash strings |

---

## 10. Stream Commands

Streams and their consumer groups follow Redis 7, compared with Redis 8.10
(#40), and are durable: entries, the stream's metadata, groups, consumers and
pending entries reach the WAL as effect records (23, 27, 34, 38-45), the
snapshot, DUMP payloads, COPY and replicas. A pending list stays fast at work
queue sizes: acknowledging an entry is a binary search, not a shift of every
later one.

Redis 8.2's group-reference handling is implemented too: XDELEX, XACKDEL, and
KEEPREF / DELREF / ACKED on XADD and XTRIM trimming (keep the pending entries
that name a deleted entry, remove them with it, or delete only what no group
still references). Redis 8's later stream additions are not: XREADGROUP CLAIM
(8.4), idempotent XADD with XCFGSET and XIDMPRECORD (8.6), XNACK (8.8), and
XREAD / XREADGROUP MAXCOUNT / MAXSIZE. Each is refused with an error, never
ignored. XINFO STREAM leaves out the fields they report (`idmp-*`,
`pids-tracked`, `iids-*`, `nacked-count`) and Redis's internal
`radix-tree-keys` / `radix-tree-nodes` (Pion keeps no radix tree).

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| XADD | ✅ | ✅ | ✅ | **SLOW** | ✅ | `*`, `ms-seq`, `ms-*`; NOMKSTREAM; MAXLEN/MINID with `=`/`~` and LIMIT (`~` trims exactly — see below) (#34); KEEPREF / DELREF / ACKED for the trim (#40) |
| XREAD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Only streams with new entries are returned. `BLOCK` parks the connection until an XADD or the timeout; inside MULTI/EXEC it answers at once |
| XLEN | ✅ | ✅ | ✅ | **SLOW** | ✅ | Real length |
| XRANGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | `-`/`+`, `ms`, exclusive `(id`; COUNT |
| XREVRANGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | As XRANGE, reversed |
| XINFO STREAM | ✅ | ✅ | ✅ | **SLOW** | ✅ | `length`, `last-generated-id`, `max-deleted-entry-id`, `entries-added`, `recorded-first-entry-id`, `groups`, first/last entry; `FULL [COUNT n]` with every group, its pending entries and consumers (#40) |
| XINFO GROUPS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Consumers, pending, last-delivered-id, entries-read and lag, by Redis 7's rules (a deletion past a group makes its lag nil) (#40) |
| XINFO CONSUMERS | ✅ | ✅ | ✅ | **SLOW** | ✅ | Pending, idle and inactive, in name order (#40) |
| XSETID | ✅ | ✅ | ✅ | **SLOW** | ✅ | With ENTRIESADDED and MAXDELETEDID (#40) |
| XDEL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Deletes by ID; every ID validated first |
| XTRIM | ✅ | ✅ | ✅ | **SLOW** | ✅ | MAXLEN/MINID with `=`/`~` and LIMIT. With `~` Redis removes only whole internal nodes (a small stream keeps everything); Pion has no nodes and trims exactly — both within the "at least N kept" contract |
| XGROUP | ✅ | ✅ | ✅ | **SLOW** | ✅ | CREATE (MKSTREAM, ENTRIESREAD), SETID (ENTRIESREAD), DESTROY, CREATECONSUMER, DELCONSUMER, HELP (#40) |
| XREADGROUP | ✅ | ✅ | ✅ | **SLOW** | ✅ | `>` for new entries (COUNT, NOACK), an id for the consumer's history (a deleted entry as `[id, nil]`); `BLOCK` parks the connection until an entry arrives for the group, the group or key goes (NOGROUP), the key changes type (WRONGTYPE) or the timeout; inside MULTI/EXEC it answers at once (#40) |
| XACK | ✅ | ✅ | ✅ | **SLOW** | ✅ | Counts the entries it acknowledged (#40) |
| XCLAIM | ✅ | ✅ | ✅ | **SLOW** | ✅ | IDLE, TIME, RETRYCOUNT, FORCE, JUSTID, LASTID; a claimed entry that was deleted leaves the pending list (#40) |
| XAUTOCLAIM | ✅ | ✅ | ✅ | **SLOW** | ✅ | Cursor, COUNT, JUSTID, and the ids of deleted entries it dropped (#40) |
| XPENDING | ✅ | ✅ | ✅ | **SLOW** | ✅ | Summary, and the extended form with IDLE and a consumer filter (#40) |
| XDELEX | ✅ (8.2) | ❌ | ✅ | **SLOW** | — | KEEPREF / DELREF / ACKED; per id 1 deleted, -1 missing, 2 still referenced (#40) |
| XACKDEL | ✅ (8.2) | ❌ | ✅ | **SLOW** | — | Acknowledges in the group, then deletes as XDELEX (#40) |
| XNACK, XCFGSET, XIDMPRECORD | ✅ (8.6+) | ❌ | ❌ | — | — | Redis 8 additions, not implemented: unknown command |

---

## 11. Pub/Sub Commands

Delivery goes through each subscriber's output buffer, in RESP2 or as RESP3
pushes, with no cap on channels, patterns or subscribers (#42). A subscriber
that is slow to read keeps what it has not read yet; one more than 32 MB behind
is disconnected, as Redis disconnects one past its pubsub output-buffer limit
(32 MB), so it never receives half a message. A RESP2 connection with subscriptions may run
only the pub/sub commands, PING (`[pong, <message>]`), QUIT and RESET, as in
Redis. With `-w N > 1` (`--independent-workers`) a message reaches subscribers
on every worker; PUBLISH counts those of its own worker, as a Redis Cluster
node counts its own, and PUBSUB reports its own worker's subscriptions.

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| SUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| UNSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Without arguments, from every channel |
| PUBLISH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Any size; the publisher gets its own message (RESP3) before the reply. Works from a script |
| PSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Redis's glob, as KEYS |
| PUNSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| PUBSUB | ✅ | ✅ | ✅ | **SLOW** | ✅ | CHANNELS, NUMSUB, NUMPAT, SHARDCHANNELS, SHARDNUMSUB, HELP |
| SSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Shard channels are their own namespace, delivered as `smessage`, with their own counts |
| SUNSUBSCRIBE | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SPUBLISH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Reaches shard subscribers only |

---

## 12. Scripting & Functions

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| EVAL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Lua 5.1; `redis.call()` runs every command (see below) |
| EVALSHA | ✅ | ✅ | ✅ | **SLOW** | ✅ | SHA1-indexed script cache; `NOSCRIPT No matching script. Please use EVAL.` |
| EVAL_RO | ✅ | ✅ | ✅ | **SLOW** | ✅ | A write from the script is refused, as in Redis |
| EVALSHA_RO | ✅ | ✅ | ✅ | **SLOW** | ✅ | As EVAL_RO |
| SCRIPT LOAD | ✅ | ✅ | ✅ | **SLOW** | ✅ | Compile + cache, returns SHA1 |
| SCRIPT EXISTS | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| SCRIPT FLUSH | ✅ | ✅ | ✅ | **SLOW** | ✅ | `ASYNC` / `SYNC` |
| SCRIPT KILL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Always `NOTBUSY`: a worker answers nothing while a script runs (see `--lua-time-limit`) |
| SCRIPT DEBUG | ✅ | ✅ | 🟡 | **SLOW** | ✅ | `NO` is accepted; `YES`/`SYNC` are refused: there is no Lua debugger |
| FUNCTION LOAD | ✅ | ✅ | ✅ | **SLOW** | ✅ | `#!lua name=<lib>`, `REPLACE`, `register_function` with `flags` and `description`; WAL-logged and replicated |
| FUNCTION LIST | ✅ | ✅ | ✅ | **SLOW** | ✅ | `LIBRARYNAME`, `WITHCODE`, flags and descriptions |
| FUNCTION DELETE | ✅ | ✅ | ✅ | **SLOW** | ✅ | WAL-logged and replicated |
| FUNCTION DUMP | ✅ | ✅ | ✅ | **SLOW** | ✅ | The payload is Pion's own format, as DUMP's is: it restores here, not into Redis |
| FUNCTION RESTORE | ✅ | ✅ | ✅ | **SLOW** | ✅ | `APPEND` / `REPLACE` / `FLUSH`, all or nothing |
| FUNCTION FLUSH | ✅ | ✅ | ✅ | **SLOW** | ✅ | `ASYNC` / `SYNC`; WAL-logged and replicated |
| FUNCTION STATS | ✅ | ✅ | ✅ | **SLOW** | ✅ | `running_script` is always nil (see SCRIPT KILL) |
| FUNCTION KILL | ✅ | ✅ | ✅ | **SLOW** | ✅ | Always `NOTBUSY` |
| FCALL | ✅ | ✅ | ✅ | **SLOW** | ✅ | |
| FCALL_RO | ✅ | ✅ | ✅ | **SLOW** | ✅ | Only a function flagged `no-writes` |

### Lua Scripting Details

**Engine:** Lua 5.1.5, statically linked, with Redis's read-only-table patch. Each worker has two states, as Redis has: one for EVAL scripts and one for FUNCTION libraries.

**`redis.call()` runs the server's own commands.** Each call goes through the slow-path dispatcher re-entrantly (`SlowPathHandler.script_dispatch`). Every command and option is therefore available, with the same replies. Every write is logged to the WAL and replicated by the command itself. The checks Redis makes come first:
- an unknown command;
- the arity;
- the `noscript` commands (MULTI, EVAL, SUBSCRIBE, CLIENT, CONFIG, SAVE, …);
- a write from a read-only script.

WAIT and XREAD BLOCK answer at once inside a script, as in Redis.

**What a script sees, checked against redis-server 8.10:**
- Errors carry Redis's suffix: `<error> script: <sha>, on @user_script:<line>.` (`@user_function` for FCALL).
- `pcall(redis.call, …)` catches a command's error and returns its message.
- Lua ↔ RESP conversions follow Redis in both protocols, `redis.setresp(3)` and its `map`/`set`/`double`/`big_number`/`verbatim_string` tables included.
- The globals are Redis's: base minus `print`, `dofile`, `loadfile`, `getfenv`, `setfenv`; `table`, `string`, `math`, `coroutine`; `os` with `clock` only; `cjson`, `cmsgpack`, `struct` and `bit`; and `redis`.
- Everything is read-only, and reading an undefined global is an error.
- Shebang flags (`#!lua flags=no-writes,allow-oom`) are honoured.
- `redis.REDIS_VERSION` is `7.0.0`, the version INFO reports.

**Limits:**
- `--lua-time-limit MS` (default 5000) stops a script that has run that long *without writing*, with `ERR Script killed: it ran longer than lua-time-limit (… ms) without writing`. A worker runs one thing at a time, so it cannot answer the `SCRIPT KILL` or `BUSY` that Redis uses here. A script that has written keeps running, as an unkillable script does in Redis. `0` turns the limit off.
- `--lua-memory-limit SIZE` (default 1gb, `0` = none) caps each Lua state's heap. Redis has no such cap.

**Persistence:** FUNCTION libraries are written to the WAL (records 35 LOAD, 36 DELETE, 37 FLUSH) and into snapshots, and replicated. The EVAL script cache is not persisted, as in Redis.

---

## 13. Transaction Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| MULTI | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Per-connection tx state (`tx_in_multi[fd]`): queues commands, validates names at QUEUE time against the generated command table, and answers EXEC with -EXECABORT on an unknown one |
| EXEC | ✅ | ✅ | ✅ | **SLOW** | ✅ | Runs the queued commands atomically and returns their replies as an array; `-EXECABORT` if any was rejected at QUEUE time |
| DISCARD | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns +OK |
| WATCH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Monitors keys for changes between WATCH and EXEC; EXEC returns null if any watched key was modified or has expired since (#45, as Redis 7). Per-fd version tracking via key_versions[65536] array, bumped by SET/DEL/HSET in fast path |
| UNWATCH | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Returns +OK |

---

## 14. Server / Admin Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| PING | ✅ | ✅ | ✅ | **FAST** | ✅ | Exponential-doubling batch copy for pipelined PINGs |
| FLUSHALL | ✅ | ✅ | ✅ | **FAST/SLOW** | ✅ | Clears this worker's keyspace (one keyspace by default; `-w N > 1` needs `--independent-workers`) |
| FLUSHDB | ✅ | ✅ | ✅ | **SLOW** | ✅ | Clears this worker's keyspace (same as FLUSHALL in shared-nothing) |
| DBSIZE | ✅ | ✅ | ✅ | **FAST** | ✅ | Returns sum of shard sizes for this worker |
| SELECT | ✅ | ✅ | ✅ | **FAST** | ✅ | One database, as Redis with `databases 1`: `SELECT 0` is OK, any other index is refused |
| SWAPDB | ✅ | ✅ | ✅ | **SLOW** | ✅ | One database: `SWAPDB 0 0` is OK, any other index is out of range |
| SAVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Synchronous WAL flush |
| BGSAVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Async WAL msync |
| BGREWRITEAOF | ✅ | ✅ | ✅ | **SLOW** | ✅ | Rewrites the WAL compactly (the live keyspace as fresh records), and answers Redis's `Background append only file rewriting started` (#47) |
| LASTSAVE | ✅ | ✅ | ✅ | **SLOW** | ✅ | The unix time of the last SAVE/BGSAVE, and the startup time before one, as Redis (#47) |
| TIME | ✅ | ✅ | ✅ | **SLOW** | ✅ | [unix seconds, microseconds] |
| INFO | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Server / Memory / Cluster / Replication / Persistence / Stats / Pion / Keyspace sections; `INFO <section> ...` returns only those. Port, RSS, uptime, per-worker key counts and the replication role are resolved from real state; `redis_version` stays `7.0.0` for client feature gating, `pion_version` carries the build. A RESP3 verbatim string under `HELLO 3`. |
| PION.STATS | n/a | n/a | n/a | **SLOW** | n/a | Pion-only. The value receipt: a 16-pair map (RESP3 `%`, RESP2 flat array) — `kvprefix_hits/misses/tokens_served/bytes_served`, `prefill_seconds_avoided` (measured + estimated) and `prefill_seconds_avoided_measured` (client-reported `PREFILL_MS` only), `semantic_hits/misses`, `moe_hits/misses`, `vector_queries`. Per WORKER. `PION.STATS RESET` zeroes the counters. |
| CONFIG GET | ✅ | ✅ | ✅ | **SLOW** | ✅ | Glob patterns over the parameters Pion can state truthfully (`databases`, `maxmemory`, `maxmemory-policy`, `appendonly`, `save`, `port`, `timeout`, `enable-debug-command`, `slowlog-*`, `latency-monitor-threshold` 0, `latency-tracking` no) |
| CONFIG SET | ✅ | ✅ | 🟡 | **SLOW** | ✅ | `maxmemory`, `slowlog-log-slower-than` and `slowlog-max-len` are runtime-settable, several pairs at once (all checked before any is set); `latency-monitor-threshold 0` and `latency-tracking no` are accepted (that is what Pion is); every other key refuses with an explanatory `-ERR` |
| CONFIG REWRITE | ✅ | ✅ | ✅ | **SLOW** | ✅ | `ERR The server is running without a config file`, as Redis without one |
| CONFIG RESETSTAT | ✅ | ✅ | ✅ | **SLOW** | ✅ | Resets the counters INFO reports (PION.STATS) |
| COMMAND | ✅ | ✅ | ✅ | **SLOW** | ✅ | Every command's entry: Redis's own (RESP2 and RESP3, key specs, ACL categories, subcommands) for a command Redis has, one built from Pion's table for a Pion-only command (#47) |
| COMMAND COUNT | ✅ | ✅ | ✅ | **SLOW** | ✅ | The commands in Pion's generated table (not their subcommands, as Redis) |
| COMMAND DOCS | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Each command's group; Pion carries no command documentation text |
| COMMAND INFO | ✅ | ✅ | ✅ | **SLOW** | ✅ | As COMMAND, for the named commands (nil for an unknown one). Also COMMAND LIST [FILTERBY MODULE\|ACLCAT\|PATTERN] (commands and `command\|subcommand`), GETKEYS / GETKEYSANDFLAGS (Redis's key-spec walk, and its own procedures where the specs are incomplete) and HELP (#47) |
| DEBUG | ✅ | ✅ | 🟡 | **SLOW** | ❌ | Refused unless `--enable-debug-command yes\|local` (default no, as Redis 7). Has HELP, SET-ACTIVE-EXPIRE and SLEEP; any other subcommand is refused (#45; it answered +OK to everything) |
| SLOWLOG | ✅ | ✅ | ✅ | **SLOW** | ✅ | Records commands at or over `slowlog-log-slower-than` (10 ms), up to `slowlog-max-len`, in Redis 8's entry shape (with the argument count; 32 arguments, 128 bytes each; credentials redacted; EXEC not logged, what it ran is). Fast-path commands are O(1) and not timed, except that a threshold under 1 ms sends every command through the timed path, and an LRANGE over 4096 elements is timed. GET [count] / LEN / RESET / HELP. Per worker (#47) |
| LATENCY | ✅ | ✅ | ✅ | **SLOW** | ✅ | Pion has no latency monitor and no per-command latency tracking, and answers as Redis does with both off: LATEST/HISTORY empty, RESET 0, GRAPH no samples, DOCTOR the monitoring-disabled report, HISTOGRAM an empty map, HELP (#47) |
| MEMORY USAGE | ✅ | ✅ | ✅ | **SLOW** | ✅ | Pion's own layout: the slot, the key if it does not fit, the value's allocations (containers sampled, `SAMPLES n`, 0 = all); nil for a missing key (#47) |
| MEMORY DOCTOR | ✅ | ✅ | 🟡 | **SLOW** | ❌ | A report of what Pion measures: RSS, its peak, maxmemory. Also MEMORY STATS (those, under Redis's field names), MALLOC-STATS (Redis's answer for an allocator without statistics), PURGE, HELP (#47) |
| MODULE | ✅ | ✅ | ✅ | **SLOW** | ❌ | LIST names `search` (Pion's built-in FT.*), which clients check for; LOAD / LOADEX / UNLOAD refuse; HELP (#47) |
| ACL | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Pion's users are the default user (with `--requirepass`) and the `--tenant` users: GETUSER, LIST, USERS, WHOAMI, CAT, DRYRUN, GENPASS and HELP describe them as Redis does (passwords as SHA-256). LOG records failed AUTHs and tenants' denied commands. SETUSER and DELUSER refuse; SAVE / LOAD give Redis's no-ACL-file error (#47) |
| CLIENT ID / INFO / LIST | ✅ | ✅ | ✅ | **SLOW** | ✅ | IDs are never reused (it was the fd). LIST [TYPE t \| ID id ...] and INFO give Redis's fields that Pion measures: id, addr, laddr, fd, name, age, idle, flags (N P x b O r e T), db, sub/psub/ssub, multi, watch, qbuf, qbuf-free, events, user, redir, resp, lib-name, lib-ver, io-thread; memory and network counters and the last command are left out. Per worker (#47) |
| CLIENT KILL | ✅ | ✅ | ✅ | **SLOW** | ✅ | `<ip:port>`, or ID / ADDR / LADDR / TYPE / USER / SKIPME / MAXAGE filters, as Redis; a client that kills itself gets its reply and is closed (#47) |
| CLIENT PAUSE / UNPAUSE | ✅ | ✅ | ✅ | **SLOW** | ✅ | ALL holds every command, WRITE Redis's write and may-replicate commands (and Pion's ingest), an EXEC that writes; held clients show `b` and resume in order; expiry pauses too (#47) |
| CLIENT REPLY | ✅ | ✅ | ✅ | **SLOW** | ✅ | ON / OFF / SKIP, as Redis (a reply already partly sent cannot be withdrawn) (#47) |
| CLIENT UNBLOCK | ✅ | ✅ | ✅ | **SLOW** | ✅ | A client blocked in BLPOP & co., XREAD BLOCK or WAIT: TIMEOUT answers as its timeout, ERROR with `-UNBLOCKED` (#47) |
| CLIENT SETNAME / GETNAME / SETINFO / NO-EVICT / NO-TOUCH / HELP | ✅ | ✅ | ✅ | **SLOW** | ✅ | As Redis (#30, #47) |
| CLIENT TRACKING / CACHING / GETREDIR / TRACKINGINFO | ✅ | ✅ | 🟡 | **SLOW** | ✅ | Client-side caching is not supported: TRACKING ON refuses, OFF is OK; GETREDIR -1 and TRACKINGINFO `off`, which is true (#47) |
| RESET | ✅ | ✅ | ✅ | **SLOW** | ✅ | As Redis: leaves MONITOR, discards MULTI and WATCH, drops every subscription, back to RESP2, the default user (unauthenticated when a password is set) and no name; READONLY off. Runs at once inside MULTI, as do QUIT and WATCH (#44) |
| MONITOR | ✅ | ✅ | ✅ | **SLOW** | ❌ | Redis's lines: shown after it runs, scripts before what they call, EXEC after its commands, `admin` commands never, AUTH/HELLO credentials redacted. A monitor may not touch the keyspace. With `--independent-workers` a monitor sees its own worker's commands. While a client monitors, every command takes the slow path (#39) |
| LOLWUT | ✅ | ✅ | ✅ | **SLOW** | ✅ | `VERSION 5` (Schotter) and `VERSION 6` (the skyline), ported from Valkey; otherwise `Pion ver. <version>` (#39) |
| ROLE | ✅ | ✅ | ✅ | **SLOW** | ✅ | A standalone server is a primary with offset 0 and no replicas. Under `--cluster`, a primary lists its replicas `[ip, port, acked offset]`, and a replica reports its primary and its link state (#39) |
| REPLICAOF / SLAVEOF | ✅ | ✅ | 🟡 | **SLOW** | ✅ | `NO ONE` is OK on a primary. Refused in cluster mode, as Redis refuses it. Pointing a standalone server at a primary is refused with an error naming the startup flags: Pion replicates in cluster mode, over its own stream (#39) |
| FAILOVER | ✅ | ✅ | ✅ | **SLOW** | ✅ | As a Redis primary with no connected replicas, which a standalone Pion always is (arguments parsed and checked, then `ERR FAILOVER requires connected replicas.`). Refused in cluster mode, as Redis does: Pion's failover there is CLUSTER FAILOVER (#39) |
| SYNC | ✅ | ✅ | 🟡 | **SLOW** | n/a | Refused: a Redis primary streams an RDB file and then commands, which Pion does not produce. Pion replicas follow their primary over its replication port (#39) |
| QUIT | ✅ | ✅ | 🟡 | **FAST** | ✅ | Returns +OK |
| AUTH | ✅ | ✅ | ✅ | **SLOW** | ✅ | Real auth: with `--requirepass`/`--tenant`, `AUTH <pw>` gates every command (`-NOAUTH` before, `-WRONGPASS` on a bad password); with no password set, `AUTH` replies the Redis error, not +OK |
| HELLO | ✅ | ✅ | ✅ | **SLOW** | ✅ | `HELLO 3` switches the connection to RESP3 (map/push replies); `HELLO`/`HELLO 2` stay RESP2 |
| SHUTDOWN | ✅ | ✅ | ✅ | **SLOW** | ✅ | `NOSAVE`, `SAVE`, `NOW`, `FORCE` parsed. A plain SIGTERM drains the WAL first. `ABORT` answers `No shutdown in progress.` (it shut the server down) (#47) |
| XGPU | ❌ | ❌ | ✅ | **SLOW** | ❌ | Pion extension: INFO-style GPU availability block |

---

## 15. Cluster Commands

| Command | Redis 8 | Valkey 8 | Pion | Pion path | GLIDE | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| CLUSTER INFO | ✅ | ✅ | ✅ | **SLOW** | ✅ | With `--cluster`. Outside cluster mode every CLUSTER subcommand but Pion's STATS answers `This instance has cluster support disabled`, as Redis (#47) |
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
| CLUSTER GETKEYSINSLOT | ✅ | ✅ | ✅ | **SLOW** | ✅ | With `--cluster`: the keys in the slot (this worker's keyspace) |
| CLUSTER COUNTKEYSINSLOT | ✅ | ✅ | ✅ | **SLOW** | ✅ | With `--cluster`: the keys in the slot |
| CLUSTER STATS | ✅ | n/a | n/a | **SLOW** | n/a | Pion-only. INFO-style bulk-string telemetry: `cluster_stats_local_worker`, `cluster_stats_total_workers`, `kv_prefix_active_total` + per-worker `kv_prefix_active_worker_<id>` (cross-worker via shared directory ACQUIRE), `local_vstore_sessions`, `local_attend_*` (legacy HNSW path; ATTEND.PREFIX.* counters live C-side, scoped out). |
| ASKING | ✅ | ✅ | ✅ | **FAST** | ✅ | With `--cluster`; outside cluster mode refused as Redis refuses it (#47) |
| READONLY | ✅ | ✅ | ✅ | **FAST** | ✅ | With `--cluster`, allows reads on a replica; outside cluster mode refused as Redis refuses it (#47) |
| READWRITE | ✅ | ✅ | ✅ | **FAST** | ✅ | Inverse of READONLY; outside cluster mode refused (#47) |
| MIGRATE | ✅ | ✅ | ✅ | **SLOW** | ✅ | As Redis: `COPY`, `REPLACE`, `AUTH`, `AUTH2`, `KEYS`, `NOKEY`, the target's own errors, each key's remaining TTL, a host name or IPv6 address. To a Pion target (Redis refuses Pion's payload). The deletions are logged (#41) |
| PSYNC | ✅ | ✅ | 🟡 | **SLOW** | n/a | Refused, as SYNC. It answered +OK, and a Redis replica then waited for an RDB file that never came (#39) |
| REPLCONF | ✅ | ✅ | ✅ | **SLOW** | n/a | Redis's answers for a client that is not a replica: options checked, `ACK`/`GETACK` answer nothing (#39) |
| RESTORE-ASKING | ✅ | ✅ | ✅ | **SLOW** | n/a | RESTORE, as a cluster's MIGRATE sends it (#39, #41) |

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
| AI.KNN_LM.QUERY | **SLOW** | ❌ | Top-k kNN: `<ds_id> <k> <emb_blob>`. Returns `k × 8 bytes` packed as `<Int32 LE token_id><Float32 LE distance>`. Brute-force scan below 5K, focused FP32 HNSW above. |
| AI.KNN_LM.INFO | **SLOW** | ❌ | `count=N dim=D max_entries=M` |
| AI.KNN_LM.DROP | **SLOW** | ❌ | Free datastore + HNSW graph buffers |
| NEURON.PKM.CREATE | **SLOW** | ❌ | Allocate a product-key memory table: `<table> <dim> <n_slots> [VDIM <v>] [VALTYPE F32\|F16]`. `n_slots` must be a perfect square S² (S ≤ 4096); dim must be even. Up to 8 tables per worker. Enabled by `--kvcache` / `--inference`. |
| NEURON.PKM.SETKEYS | **SLOW** | ❌ | Load codebook half 0 or 1: `<table> <half> <blob>` — S × (dim/2) Float32 LE. Builds the INT8 mirror. Both halves required before QUERY. |
| NEURON.PKM.SETVALS | **SLOW** | ❌ | Write value rows: `<table> <off> <n> <blob>` — n × vdim in VALTYPE. Value matrix is allocated on the first call. |
| NEURON.PKM.QUERY | **SLOW** | ❌ | **Exact** top-k over all n_slots: `<table> <k> <q_blob> [FAST]`. `nq = len(q_blob)/(dim·4)` heads share one codebook pass. Returns `nq·k × 8 bytes` as `<Int32 LE slot_id><Float32 LE score>`, descending, padded `(-1, -inf)`. dim=896 k=32 1M slots: **0.075 ms** p50 (vs 5.13 ms for kNN-LM HNSW at the same shape; `tests/test_neuron_pkm.py --bench`, M4 Mac mini, [raw](../benchmarks/results/2026-10-06-mac-m4/neuron_pkm_bench.txt)). |
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
| MOE.EXPERT.FETCH `<model_id> <layer> <expert>` | **SLOW** | bulk blob / `-UNAVAILABLE` | Served from the RAM LRU, SSD or network tier, whichever holds it |
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
| ATTEND.PREFIX.QUERY | **SLOW** | ❌ | Run M=1 single-query attention on cached K/V (Q-only on wire). Body: `<session_id> <layer_id> <H> <D> <top_k> <Q_blob>`. Returns `H*D` float32. Native Metal SDPA path (sdpa_q1_fp32/fp16); supports D ∈ {32,64,96,128,160,192,256,**512**}. |
| ATTEND.PREFIX.QUERY_FUSED | **SLOW** | ❌ | M>=1 fused suffix-SDPA + prefix-merge in one dispatch. Body: `<sid> <layer> <H_q> <D> <S_suf> <H_kv> <Q> <K_suf> <V_suf> <head_map> [<fa_window>]`. GQA-aware. Returns merged attention `H_q*M*D*4` bytes; no LSE trailer (merge fused). |
| ATTEND.PREFIX.QUERY_SPARSE | **SLOW** | ❌ | Sparse-mask M=1 attention with caller-supplied per-head indices. Body: `<sid> <layer> <H> <D> <K_sparse_max> <Q> <indices> <counts> [<fa_window>]`. For learned-router v2 consumers. Output `H*D*4` bytes. |
| ATTEND.PREFIX.QUERY_SPARSE_AUTO | **SLOW** | ❌ | Sparse-mask M=1 with server-side block-mean top-K selection. Body: `<sid> <layer> <H_q> <D> <B> <K_top> <H_kv> <Q> <head_map> [<fa_window>]`. K_mean cached server-side per slot. GQA-aware. Output `H_q*D*4` bytes. |
| ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED | **SLOW** | ❌ | Sparse-AUTO + dense-suffix + online-softmax merge in one fused dispatch. Body: `<sid> <layer> <H_q> <D> <B> <K_top> <H_kv> <S_suf> <Q> <K_suf> <V_suf> <head_map> [<fa_window>]`. Closes the "ignore suffix" v1 wire-mode-sparse caveat. Output `H_q*D*4` bytes. |
| SSM.PREFIX.STORE | **SLOW** | ❌ | Store opaque byte blob (serialized SSM recurrent state — Mamba `[conv_state, ssm_state]`, RWKV-7 size=3 tuple, or any future linear-attention family). Body: `<session_id> <layer_id> <state_blob>`. Server never parses the blob. |
| SSM.PREFIX.FETCH | **SLOW** | ❌ | Retrieve previously-stored blob for `(sid, layer_id)`. Returns bulk string or `$-1` (miss). |
| SSM.PREFIX.DROP | **SLOW** | ❌ | Drop stored blob. Body: `<session_id> [<layer_id>]`. Layer omitted = drop all layers for the session. Idempotent. |
| ATTEND.CREATE | **SLOW** | ❌ | Create attention session with key_dim and value_dim |
| ATTEND.STORE | **SLOW** | ❌ | Stage token KV pairs (FP32 memcpy) |
| ATTEND.FINALIZE | **SLOW** | ❌ | Batch build HNSW index from staged keys |
| ATTEND.QUERY | **SLOW** | ❌ | Top-k HNSW search over attention keys |
| ATTEND.INFO | **SLOW** | ❌ | Index statistics (sessions, tokens, queries) |

Binary protocol commands on port 1975 (0xCA5E framing):

| Command byte | Command | Description |
|:---:|---|---|
| 0x01 | LAYER.STORE | Store per-layer tensor (binary blob) |
| 0x02 | LAYER.FETCH | Fetch per-layer tensor |
| 0x03 | PING | Binary keepalive |
| 0x20 | ATTEND.CREATE | Create attention session (binary) |
| 0x21 | ATTEND.STORE | Stage token KV pairs |
| 0x22 | ATTEND.FINALIZE | Batch build HNSW from staged keys |
| 0x23 | ATTEND.QUERY | Top-k HNSW query |

Requires `--kvcache` flag. Python client: `vllm-pion/` package (`PionKVClient`, `PionAttentionClient`, `ExternalizedAttentionLayer`).

---

## 18. Semantic Router Commands (`--kvcache`)

| Command | Pion path | GLIDE | Description |
|---|:---:|:---:|---|
| AI.ROUTE.REGISTER | **SLOW** | ❌ | Register inference node with semantic centroid embedding + optional CAPACITY |
| AI.ROUTE.UPDATE | **SLOW** | ❌ | Update a node's centroid embedding (as KV cache evolves) |
| AI.ROUTE | **SLOW** | ❌ | Route query embedding to best node by cosine similarity |
| AI.ROUTE.REMOVE | **SLOW** | ❌ | Remove node from routing table |
| AI.ROUTE.INFO | **SLOW** | ❌ | Per-node stats: routed count, capacity, endpoint |

Routing strategy: FP32 brute-force cosine for <=16 nodes (perfect accuracy); HNSW O(log N) for >16 nodes. No routing accuracy is published with a harness yet.

Requires `--kvcache` flag.

---

## 19. Speculative RAG Commands (`--kvcache`)

| Command | Pion path | GLIDE | Description |
|---|:---:|:---:|---|
| RAG.SPECULATE.ENABLE | **SLOW** | ❌ | Enable speculative RAG for a session (creates trajectory ring buffer of last 10 embeddings) |
| RAG.QUERY | **SLOW** | ❌ | Query with speculation: check pre-computed cache first (cosine > 0.9), fall back to live HNSW search |
| RAG.SPECULATE.INFO | **SLOW** | ❌ | Per-session stats: hit rate, trajectory length, predictions outstanding |

Prediction model: `predicted = current + alpha * (current - previous)`, alpha in [0.5, 1.0, 1.5] (3 predictions per query). No hit rate is published with a harness yet.

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

Pion dispatches 354 command names (`PION_COMMAND_COUNT` in the generated
`src/commands/command_table.mojo`), and `tests/test_dispatch_sweep.py` sends
every one of them in five argument shapes. The counts below are of the rows in
this document; ✅ and 🟡 both count as supported, and a 🟡 row's notes say
what is missing.

| Category | Rows in this document | Supported in Pion | Of which partial (🟡) |
|---|:---:|:---:|:---:|
| String | 23 | 23 | 0 |
| Key / Expiry | 31 | 31 | 1 |
| Hash | 25 | 25 | 1 |
| List | 22 | 22 | 0 |
| Set | 17 | 17 | 0 |
| Sorted Set | 35 | 35 | 0 |
| Bitmap | 7 | 7 | 0 |
| HyperLogLog | 5 | 5 | 1 |
| Geo | 10 | 10 | 0 |
| Streams | 20 | 19 | 0 |
| Pub/Sub | 9 | 9 | 0 |
| Scripting & Functions | 19 | 19 | 1 |
| Transactions | 5 | 5 | 3 |
| Server / Admin | 47 | 46 | 10 |
| Cluster | 21 | 20 | 2 |
| Vector sets (VSET) | 13 | 12 | 0 |

### Pion Fast Path (zero-alloc dispatch)
`GET` `SET` `MGET` `MSET` `INCR` `DECR` `HSET` `HGET` `LPUSH` `RPUSH` `LPOP` `RPOP` `LRANGE` `LLEN` `DEL` `EXISTS` `SADD` `SPOP` `ZADD` `ZPOPMIN` `PING` `FUNCTION LOAD` `FCALL` `GETBIT` `SETBIT` `BITCOUNT`(no-arg) `PFADD` `PFCOUNT`(single-key) `ECHO` `TYPE` `SELECT` `DBSIZE` `QUIT`. CLIENT and COMMAND go to the slow path (#47); RESET goes to the slow path, which resets the connection.

---

## Known Compatibility Pitfalls

| Issue | Details |
|---|---|
| `r.incr(key)` → INCRBY | Python redis-py sends `INCRBY key 1`, not `INCR key`. Both work; `INCR` takes the fast path. |
| Multi-worker + AI | When `--flare` / `--emb-enabled` is set, Pion auto-caps to `-w 1`. SemanticCache is per-worker and cannot share state across workers. |
| SCAN cursor semantics | SCAN, HSCAN, SSCAN and ZSCAN iterate incrementally, COUNT-bounded, with a working cursor (#50). A full iteration returns every key present from start to finish, as in Redis. A small keyspace still returns in one call; a large one paginates. |
| Blocking commands | BLPOP, BRPOP, BRPOPLPUSH, BLMOVE, BLMPOP, BZPOPMIN, BZPOPMAX and BZMPOP block as in Redis. The wait is per worker: with `-w N > 1` a push on another worker's keyspace never reaches them (see `--independent-workers`). Inside MULTI/EXEC and inside a script they answer at once. |
| Lua scripting | `redis.call()` runs every command. A script that runs past `--lua-time-limit` without writing is stopped, because a worker cannot answer SCRIPT KILL mid-script (see §12). |

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
