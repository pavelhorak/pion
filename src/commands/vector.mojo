"""Vector search commands: FT.INFO, FT.DROPINDEX, FT.OPTIMIZE, FT.CREATE, FT.ADDTEXT, FT.SEARCHTEXT, FT.SEARCH + BM25."""
from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import UnsafePointer
from std.collections import Array, List
from std.memory import alloc, unsafe_memcpy, unsafe_memset, stack_allocation
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.time import perf_counter_ns
from std.sys.info import CompilationTarget
from std.math import log, sqrt
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.network.dispatcher import CommandDispatcher
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.container_free import hash_get_live
from src.common.value import GenericValue, ValueType
from src.common.config import PionConfig
from src.common.utils import format_int_to_buf, format_float_to_buf, int_string_len, parse_filter_float, is_valid_float_arg, parse_float64, arg_eq, parse_redis_double, DOUBLE_VALUE, DOUBLE_RANGE, parse_int64_strict, ParsedInt
from src.common.lock_free import ShardQueryBus
from src.vector.hnsw import HNSWGraph, SharedHNSWView
from src.vector.hnsw_types import PREFIX_MAX, PREFIX_SLOT, SCHEMA_MAX, SCHEMA_NAME_MAX, SCHEMA_REC, FT_FIELD_TEXT, FT_FIELD_TAG, FT_FIELD_NUMERIC, FT_FIELD_GEO, FT_FIELD_GEOSHAPE, FT_FLAG_SORTABLE, FT_FLAG_UNF, FT_FLAG_NOINDEX, FT_FLAG_CASESENSITIVE
from src.vector.fp32_scan import scan_dot_f32, l2sq_f32_4chain, dot_f32_4chain
# gh #87.1: src/vector/ivf_pq.mojo deleted (a measured dead end).
from src.network.semantic_cache import SemanticCache, CACHE_MAX_ENTRIES, bytes_name
from src.network.rerank_client import RerankClient
from src.memory.object_pool import ObjectPool
from src.common.heap import HeapNode

# IVF-PQ disabled: recall@100 = 0.59 for 50K 1536-dim vectors (a measured dead end).
# gh #87.1: ENABLE_IVF_PQ removed (was permanently False).
comptime MAX_SHARD_K = 100  # matches lock_free.mojo MAX_SHARD_K

# gh #139: BM25 ranking parameters. k1 saturates term frequency, b controls
# length normalisation (0 = off, 1 = full). These are the Lucene/Okapi defaults
# and only the *defaults* — `FT.SEARCH … BM25 … K1 <f> B <f>` overrides per query.
comptime BM25_DEFAULT_K1: Float32 = 1.2
comptime BM25_DEFAULT_B:  Float32 = 0.75
comptime BM25_MAX_QUERY_TERMS = 64  # unique query terms scored (excess ignored)


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

# gh #87.3: _parse_filter_float moved to src.common.utils as parse_filter_float.


@always_inline
def _is_bm25_hard_delim(b: UInt8) -> Bool:
    """Bytes that always terminate a BM25 token: whitespace, control chars, and
    punctuation that never appears inside a word."""
    return (b <= 32                                     # space, tab, CR, LF, controls
            or b == 34 or b == 39 or b == 96            # "  '  `
            or b == 40 or b == 41 or b == 91 or b == 93 # (  )  [  ]
            or b == 123 or b == 125                     # {  }
            or b == 60 or b == 62 or b == 124 or b == 92  # <  >  |  \
            or b == 44 or b == 59 or b == 58            # ,  ;  :
            or b == 33 or b == 63)                      # !  ?


@always_inline
def _is_bm25_edge_trim(b: UInt8) -> Bool:
    """Punctuation stripped from a token's edges but kept *inside* it.

    gh #139: this is what makes flag-like and hyphenated terms index as single
    tokens — `--fa-window` → `fa-window`, `sliding-window.` → `sliding-window`
    — instead of shattering into `fa`/`window`. Splitting on interior hyphens
    was a source of ranking divergence from a reference Okapi BM25 (rank_bm25),
    which tokenizes on whitespace and therefore keeps compounds intact."""
    return (b == 45 or b == 46 or b == 95 or b == 43     # -  .  _  +
            or b == 61 or b == 42 or b == 47 or b == 38  # =  *  /  &
            or b == 37 or b == 64 or b == 35             # %  @  #
            or b == 126 or b == 36 or b == 94)           # ~  $  ^


@fieldwise_init
struct BM25Token(Copyable, Movable, ImplicitlyCopyable):
    """One tokenizer step: `[start, end)` of the trimmed token plus where to
    resume. `valid` is False for a run that trimmed away to nothing (`---`)."""
    var start: Int
    var end: Int
    var next_pos: Int
    var valid: Bool


@always_inline
def _bm25_next_token(buf: UnsafePointer[UInt8, MutUntrackedOrigin], pos: Int, blen: Int) -> BM25Token:
    """Advance one token. Shared by index-time and query-time tokenization —
    the two MUST agree byte-for-byte or terms hash differently and never match,
    so they deliberately go through this one function."""
    var p = pos
    while p < blen and _is_bm25_hard_delim(buf[p]):
        p += 1
    if p >= blen: return BM25Token(0, 0, blen, False)
    var s = p
    while p < blen and not _is_bm25_hard_delim(buf[p]):
        p += 1
    var e = p
    while s < e and _is_bm25_edge_trim(buf[s]):
        s += 1
    while e > s and _is_bm25_edge_trim(buf[e - 1]):
        e -= 1
    return BM25Token(s, e, p, e > s)


@always_inline
def _bm25_hash(buf: UnsafePointer[UInt8, MutUntrackedOrigin], s: Int, e: Int) -> UInt32:
    """FNV-1a over the lowercased token. `| 0x20` is a no-op for digits and
    already-lowercase ASCII; 0 is remapped because it is the empty-slot
    sentinel in the vocabulary hash table."""
    var h: UInt32 = 2166136261
    for bi in range(s, e):
        h = (h ^ UInt32(buf[bi] | 0x20)) * 16777619
    if h == 0: h = 1
    return h


@always_inline
def _bm25_sift_down(idx: UnsafePointer[UInt32, MutUntrackedOrigin],
                    keys: UnsafePointer[UInt32, MutUntrackedOrigin],
                    root_in: Int, n: Int):
    """Heapsort helper: sift `idx[root_in]` down a max-heap ordered by
    `keys[idx[...]]`."""
    var root = root_in
    while True:
        var child = 2 * root + 1
        if child >= n: break
        if child + 1 < n and keys[Int(idx[child])] < keys[Int(idx[child + 1])]: child += 1
        if keys[Int(idx[root])] >= keys[Int(idx[child])]: break
        var t = idx[root]; idx[root] = idx[child]; idx[child] = t
        root = child


@always_inline
def bm25_owner_mismatch(hnsw: HNSWGraph, idx_ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
                        idx_len: Int) -> Bool:
    """gh #145: True when the current postings belong to a different index.

    Pion holds one index per server, so `FT.OPTIMIZE` on index B replaces the
    postings that were serving index A. A query on A then matched nothing and
    returned `*0` — indistinguishable from a genuine miss, which is precisely
    the failure mode gh #139 removed for the not-built case. Byte-exact compare,
    matching the KNN path's existing index-name check (index names are
    case-sensitive)."""
    if hnsw.bm25_index_name_len == 0: return False  # unnamed — nothing to verify
    if idx_len != hnsw.bm25_index_name_len: return True
    for ni in range(idx_len):
        if idx_ptr[ni] != hnsw.bm25_index_name[ni]: return True
    return False


@always_inline
def bm25_owner_error(hnsw: HNSWGraph, idx_ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
                     idx_len: Int) -> String:
    """The error text for a cross-index BM25 query — names both sides so the
    log tells you which FT.OPTIMIZE took the postings away."""
    var msg = String("ERR index '")
    var qlen = idx_len if idx_len < 64 else 64
    for ni in range(qlen): msg += chr(Int(idx_ptr[ni]))
    msg += "' has no BM25 postings - this server holds one index at a time and the current postings belong to '"
    for ni in range(hnsw.bm25_index_name_len): msg += chr(Int(hnsw.bm25_index_name[ni]))
    msg += "' (a later FT.OPTIMIZE replaced them); re-ingest and FT.OPTIMIZE '"
    for ni in range(qlen): msg += chr(Int(idx_ptr[ni]))
    msg += "' to serve it again"
    return msg


@always_inline
def _copy_value_clamped(val: GenericValue, dest: UnsafePointer[UInt8, MutUntrackedOrigin], cap: Int) -> Int:
    """Copy at most `cap` bytes of a string value into `dest`; returns the count.

    `GenericValue.copy_to` always writes `string_len()` bytes, so clamping the
    length variable without clamping the copy — as the BM25 and TAG-metadata
    paths both used to — overflows the destination on any oversized value."""
    var vlen = val.string_len()
    if vlen <= 0: return 0
    if vlen <= cap:
        val.copy_to(dest)
        return vlen
    # Only heap strings can exceed cap (SSO tops out at 23 bytes), so the raw
    # pointer is valid and a truncating memcpy is safe.
    var src = val.as_string()
    if is_null(src): return 0
    unsafe_memcpy(dest=dest, src=src, count=cap)
    return cap


def drop_dead(shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin], mut ids: List[Int],
              mut scores: List[Float32], keep: Int):
    """#46: the results without documents that left the keyspace (or whose
    vector left them), at most `keep` of them, in order."""
    var w = 0
    var any_dead = is_not_null(shared_hnsw) and shared_hnsw[].any_dead()
    for r in range(len(ids)):
        if w >= keep:
            break
        if any_dead and shared_hnsw[].slot_dead(ids[r]):
            continue
        ids[w] = ids[r]
        if r < len(scores):
            scores[w] = scores[r]
        w += 1
    while len(ids) > w:
        _ = ids.pop()
    while len(scores) > w:
        _ = scores.pop()


def _slot_key(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
              keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], slot: Int,
              mut owned: Bool) -> GenericValue:
    """The key of the document at ingest slot `slot`, for a lookup: the shared
    slot->key map (a copy: `owned`, so the caller frees it with
    free_str_payload; inline up to 23 bytes), else this worker's
    `__hk__<slot>` entry (the map's own value: not owned, never freed). NONE
    when neither has it. Not a borrow: the shared map is no RESP token, and
    gh #394's audit proves a borrowed key's bytes alive or refuses it."""
    owned = False
    if is_not_null(shared) and is_not_null(shared[].hk_keys_buf) and slot >= 0 and slot < shared[].hk_max_elements:
        var p = shared[].hk_keys_buf + slot * 32
        var l = Int(p[0])
        if l > 0 and l <= 31:
            owned = True
            return GenericValue.from_ptr(p + 1, l)
    if is_null(keyspace) or slot < 0:
        return GenericValue()
    var hk = stack_allocation[30, UInt8]()
    hk[0]=95;hk[1]=95;hk[2]=104;hk[3]=107;hk[4]=95;hk[5]=95 # __hk__
    var e = format_int_to_buf(hk, 6, Int64(slot))
    # 6 + at most 19 digits: borrow() packs it inline (SSO), no pointer to hk escapes
    return keyspace[].get(GenericValue.borrow(hk, e))


def _slot_hash(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
               keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
               slot: Int) -> UnsafePointer[SlabHashMap, MutUntrackedOrigin]:
    """The hash of the document at `slot` on this worker, or null."""
    if is_null(keyspace):
        return null_ptr[SlabHashMap, MutUntrackedOrigin]()
    var owned = False
    var key = _slot_key(shared, keyspace, slot, owned)
    if key.is_none():
        return null_ptr[SlabHashMap, MutUntrackedOrigin]()
    var v = hash_get_live(keyspace, key)
    if owned: key.free_str_payload()
    if v.is_none() or v.type.value != ValueType.HASH:
        return null_ptr[SlabHashMap, MutUntrackedOrigin]()
    return v.as_hash().bitcast[SlabHashMap]()


@always_inline
def _ttl_any(keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) -> Bool:
    """Whether a key on this worker has a TTL at all (one load when none does)."""
    return is_not_null(keyspace) and Int(keyspace[].ttl_map) != 0 and keyspace[].ttl_map[].size > 0 \
        and keyspace[].clock_ns != 0


def _slot_expired(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                  keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], slot: Int) -> Bool:
    """The document at `slot` is past its key's deadline: it is gone, as for any
    read. The lookup expires the key (Redis's lazy expiry), which kills the
    slot (#46), so later searches skip it without asking. FT.SEARCH used to
    return such a document until the active sweep reached it. Only this
    worker's TTLs are known; under -w N a key on another worker is that
    worker's sweep's."""
    var owned = False
    var key = _slot_key(shared, keyspace, slot, owned)
    if key.is_none():
        return False
    var d = keyspace[].deadline(key)
    var due = d >= 0 and d < keyspace[].clock_ns
    if due:
        _ = keyspace[].get(key)
    if owned: key.free_str_payload()
    return due


def _schema_slot(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                 p: UnsafePointer[UInt8, MutUntrackedOrigin], l: Int) -> Int:
    """The index schema's entry for field name `p[:l]` (case-sensitive, as in
    RediSearch), or -1."""
    if is_null(shared):
        return -1
    for si in range(shared[].schema_field_count):
        if Int(shared[].schema_field_name_lens[si]) != l:
            continue
        var same = True
        for b in range(l):
            if shared[].schema_field_names[si][b] != p[b]:
                same = False
                break
        if same:
            return si
    return -1


struct FtSort(ImplicitlyCopyable, Movable):
    """FT.SEARCH's SORTBY: a field, or the KNN score, and the direction."""
    var p: UnsafePointer[UInt8, MutUntrackedOrigin]
    var l: Int
    var desc: Bool
    var set: Bool

    def __init__(out self):
        self.p = null_ptr[UInt8, MutUntrackedOrigin]()
        self.l = 0
        self.desc = False
        self.set = False


def _sort_less(a: Int, b: Int, has: List[Bool], num: List[Float64], is_num: Bool,
               arena: List[UInt8], s_at: List[Int], s_len: List[Int], ids: List[Int], desc: Bool) -> Bool:
    """RediSearch's SORTBY order (checked against it): documents without the
    field last in both directions; ties, and the missing, by document order,
    reversed with DESC. Pion's document order is the ingest slot."""
    if has[a] != has[b]:
        return has[a]
    if has[a]:
        var c = 0
        if is_num:
            c = -1 if num[a] < num[b] else (1 if num[a] > num[b] else 0)
        else:
            var la = s_len[a]
            var lb = s_len[b]
            var n = la if la < lb else lb
            for k in range(n):
                var x = arena[s_at[a] + k]
                var y = arena[s_at[b] + k]
                if x != y:
                    c = -1 if x < y else 1
                    break
            if c == 0 and la != lb:
                c = -1 if la < lb else 1
        if c != 0:
            return (c > 0) if desc else (c < 0)
    return (ids[a] > ids[b]) if desc else (ids[a] < ids[b])


def _sort_knn(mut ids: List[Int], mut scores: List[Float32], sort: FtSort, opts: FtReplyOpts,
              keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
              shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]) -> String:
    """SORTBY over the KNN results, before LIMIT pages them (RediSearch's
    order of operations). It used to be read and dropped: the results came
    back by distance whatever was asked. The KNN score sorts ascending already;
    DESC reverses it. A schema field sorts by its value: NUMERIC as a number,
    TAG and TEXT by their bytes. Returns an error, or ""."""
    var n = len(ids) if len(ids) < len(scores) else len(scores)
    if opts.is_score_name(sort.p, sort.l):
        if sort.desc:
            for a in range(n // 2):
                var ti = ids[a]; ids[a] = ids[n - 1 - a]; ids[n - 1 - a] = ti
                var ts = scores[a]; scores[a] = scores[n - 1 - a]; scores[n - 1 - a] = ts
        return String("")
    var si = _schema_slot(shared, sort.p, sort.l)
    if si < 0:
        return "Property `" + _bytes_str(sort.p, sort.l) + "` not loaded nor in schema"
    var is_num = shared[].schema_field_types[si] == 3
    var has = List[Bool](capacity=n)
    var num = List[Float64](capacity=n)
    var arena = List[UInt8]()
    var s_at = List[Int](capacity=n)
    var s_len = List[Int](capacity=n)
    var vbuf = alloc[UInt8](256)
    for r in range(n):
        var h = _slot_hash(shared, keyspace, ids[r])
        var v = GenericValue()
        if is_not_null(h):
            v = h[].get(GenericValue.borrow(sort.p, sort.l))
        var ok = not v.is_none()
        var x = Float64(0.0)
        s_at.append(len(arena))
        if ok and is_num:
            var vl = _copy_value_clamped(v, vbuf, 256)
            var pf = parse_redis_double(vbuf, vl, DOUBLE_VALUE)
            ok = pf.ok
            x = pf.value
            s_len.append(0)
        elif ok:
            var vl = v.string_len()
            var tmp = alloc[UInt8](vl + 1)
            v.copy_to(tmp)
            for b in range(vl):
                arena.append(tmp[b])
            tmp.free()
            s_len.append(vl)
        else:
            s_len.append(0)
        has.append(ok)
        num.append(x)
    vbuf.free()
    # stable merge sort of positions
    var order = List[Int](capacity=n)
    for r in range(n):
        order.append(r)
    var tmp_o = List[Int](capacity=n)
    for r in range(n):
        tmp_o.append(0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width if lo + width < n else n
            var hi = lo + 2 * width if lo + 2 * width < n else n
            var a = lo
            var b = mid
            var o = lo
            while a < mid and b < hi:
                if _sort_less(order[b], order[a], has, num, is_num, arena, s_at, s_len, ids, sort.desc):
                    tmp_o[o] = order[b]; b += 1
                else:
                    tmp_o[o] = order[a]; a += 1
                o += 1
            while a < mid:
                tmp_o[o] = order[a]; a += 1; o += 1
            while b < hi:
                tmp_o[o] = order[b]; b += 1; o += 1
            lo = hi
        for r in range(n):
            order[r] = tmp_o[r]
        width *= 2
    var new_ids = List[Int](capacity=n)
    var new_scores = List[Float32](capacity=n)
    for r in range(n):
        new_ids.append(ids[order[r]])
        new_scores.append(scores[order[r]])
    ids = new_ids^
    scores = new_scores^
    return String("")


def _ft_knn_reply(mut writer: ResponseWriter, mut ids: List[Int], mut scores: List[Float32],
                  keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                  shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                  opts: FtReplyOpts, sort: FtSort, offset: Int, count: Int):
    """A KNN query's reply: every hit sorted (SORTBY), then the LIMIT window.
    The total is every hit, as RediSearch reports it. LIMIT used to cap the
    search at its count, so the total was the page's size and the offset was
    ignored (LIMIT 2 3 answered with the first three)."""
    if sort.set:
        var err = _sort_knn(ids, scores, sort, opts, keyspace, shared)
        if err.byte_length() > 0:
            writer.append_error_response(err)
            return
    var total = len(ids)
    var page_ids = List[Int]()
    var page_scores = List[Float32]()
    var end = total if count < 0 or offset + count >= total else offset + count
    var r = offset
    while r < end:
        page_ids.append(ids[r])
        page_scores.append(scores[r] if r < len(scores) else Float32(0.0))
        r += 1
    write_ft_search_response(writer, page_ids, page_scores, keyspace, shared, opts, total)


# RESP3 FT.SEARCH, in RediSearch's shape (checked against RediSearch's own
# replies): a map whose `results` entry holds one map per document (id, then
# score with WITHSCORES, then extra_attributes unless NOCONTENT, then values).
# redis-py 8 negotiates RESP3 by default and its search module reads exactly
# these keys.
comptime _FT_R3_HEAD = "%5\r\n+attributes\r\n*0\r\n+format\r\n+STRING\r\n+results\r\n"
comptime _FT_R3_ID = "+id\r\n"
comptime _FT_R3_FIELDS = "+extra_attributes\r\n"
comptime _FT_R3_DOC_END = "+values\r\n*0\r\n"
comptime _FT_R3_TOTAL = "+total_results\r\n"
comptime _FT_R3_TAIL = "+warning\r\n*0\r\n"
# WITHSCORES reports the query's text score, as RediSearch does, not the
# vector distance: 0 for `*=>[KNN …]`. Pion computes no text score for a
# filter prefix, where RediSearch reports one, and reports 0 there too.
comptime _FT_R2_TEXT_SCORE = "$1\r\n0\r\n"
comptime _FT_R3_TEXT_SCORE = "+score\r\n,0\r\n"

# What an FT.SEARCH reply carries for each document (FtReplyOpts.mode).
comptime FT_REPLY_DEFAULT = UInt8(0)    # Pion's own forms: the doc's `id` field (when its hash has one) and `score`
comptime FT_REPLY_RETURN = UInt8(1)     # the fields RETURN names, as RediSearch answers it
comptime FT_REPLY_NOCONTENT = UInt8(2)  # keys only (NOCONTENT, or RETURN 0)
comptime FT_REPLY_ALL = UInt8(3)        # KNN without RETURN: the score, then every field of the hash


struct FtReplyOpts(ImplicitlyCopyable, Movable):
    """FT.SEARCH's RETURN, NOCONTENT and WITHSCORES, as the reply writer needs
    them. The KNN form starts at FT_REPLY_ALL: without RETURN, RediSearch
    sends the score and then every field of the hash, the embedding included.
    The default, `id` and `score`, is the reply of Pion's own forms (BM25,
    HYBRID, FT.HYBRID), which have no RediSearch equivalent."""
    var mode: UInt8
    var with_scores: Bool
    var tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin]
    var ret_at: Int      # RETURN's first field token
    var ret_n: Int       # RETURN's count: field names, plus each `AS <name>`
    var alias_p: UnsafePointer[UInt8, MutUntrackedOrigin]   # the KNN clause's AS alias
    var alias_l: Int
    var field_p: UnsafePointer[UInt8, MutUntrackedOrigin]   # the KNN @field, for __<field>_score
    var field_l: Int

    def __init__(out self):
        self.mode = FT_REPLY_DEFAULT
        self.with_scores = False
        self.tokens = null_ptr[RESP3Token, MutUntrackedOrigin]()
        self.ret_at = 0
        self.ret_n = 0
        self.alias_p = null_ptr[UInt8, MutUntrackedOrigin]()
        self.alias_l = 0
        self.field_p = null_ptr[UInt8, MutUntrackedOrigin]()
        self.field_l = 0

    def is_score_name(self, p: UnsafePointer[UInt8, MutUntrackedOrigin], l: Int) -> Bool:
        """Whether a RETURN name is the KNN score: the clause's AS alias, else
        RediSearch's default name `__<field>_score`."""
        if self.alias_l > 0:
            if l != self.alias_l:
                return False
            for b in range(l):
                if p[b] != self.alias_p[b]:
                    return False
            return True
        var fl = self.field_l
        if fl == 0 or l != fl + 8 or p[0] != 95 or p[1] != 95:     # "__"
            return False
        for b in range(fl):
            if p[2 + b] != self.field_p[b]:
                return False
        var s = p + 2 + fl                                           # "_score"
        return s[0] == 95 and s[1] == 115 and s[2] == 99 and s[3] == 111 and s[4] == 114 and s[5] == 101


@always_inline
def _put_lit(mut writer: ResponseWriter, s: StaticString):
    writer.append_to_response(s.unsafe_ptr(), s.byte_length())


@always_inline
def _put_score_name(mut writer: ResponseWriter, opts: FtReplyOpts):
    """The KNN score's field name: the clause's AS alias, else RediSearch's
    default `__<field>_score`."""
    if opts.alias_l > 0:
        writer.append_bulk_string_response(opts.alias_p, opts.alias_l)
    else:
        writer.append_bulk_string_response_header(opts.field_l + 8)
        _put_lit(writer, "__")
        writer.append_to_response(opts.field_p, opts.field_l)
        _put_lit(writer, "_score\r\n")


@always_inline
def _put_score_bulk(mut writer: ResponseWriter, dist: Float32):
    """`dist` as a bulk string with 4 decimals. The length is counted the way
    format_float_to_buf writes: a score in (-1, 0) prints `-0.xxxx`, one byte
    more than `int_string_len(Int64(dist))` counts, which broke the frame."""
    var mag = -dist if dist < 0 else dist
    var score_len = (1 if dist < 0 else 0) + int_string_len(Int64(mag)) + 5
    writer.append_bulk_string_response_header(score_len)
    writer.offset = format_float_to_buf(writer.buffer, writer.offset, dist, 4)
    writer.buffer[writer.offset] = 13; writer.buffer[writer.offset+1] = 10; writer.offset += 2


@always_inline
def write_ft_search_response(mut writer: ResponseWriter, results2: List[Int], dist_scores: List[Float32],
                              keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin] = null_ptr[StripedHashMap, MutUntrackedOrigin](),
                              shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin] = null_ptr[SharedHNSWView, MutUntrackedOrigin](),
                              opts: FtReplyOpts = FtReplyOpts(), total: Int = -1):
    """Write a single FT.SEARCH response to writer (does not flush): RESP2's
    flat array, or RediSearch's RESP3 map on a RESP3 connection.

    `results2` is the page to send; `total` is every hit the query had (its
    LIMIT window cut the page from them), and defaults to the page's size.

    Each result is <key>, then (WITHSCORES) the text score, then its fields
    unless NOCONTENT. With RETURN the fields are the ones it names, renamed by
    `AS`, the KNN score first when named (RediSearch's order), each one the
    hash lacks left out. A KNN query without RETURN gets the score and then
    every field of the hash, as from RediSearch. Pion's own forms get
    [id, <id-value>, score, <dist>] when the doc hash has an "id" field on THIS
    worker, else [score, <dist>].
    Key resolution order: shared slot→key map (one direct load — the
    fast-path HSET writes it on every vector ingest), then the keyspace
    __hk__<slot> probe (covers original keys >31B that don't fit the shared
    map's 32B slots), then the numeric slot id.

    gh #357: a missing "id" field is OMITTED, never substituted. It used to
    fall back to the key name, so under `-w N` a query answered by a worker
    that did not ingest the doc returned id="doc:17" where the stored field
    is "17" — a plausible wrong value (the keyspace is per worker, gh #253).
    The same holds for every RETURN field: a worker that does not hold the
    hash returns none of them, as Redis omits a field the doc does not have.
    Clients read fields by name (redis-py drops "id" and uses the key as
    doc.id). Callers must pass `keyspace`, or every result loses its fields."""
    var nr = len(results2)
    var n_total = total if total >= 0 else nr
    var resp3 = writer.proto == 3
    var content = opts.mode != FT_REPLY_NOCONTENT
    var per_doc = 1 + (1 if opts.with_scores else 0) + (1 if content else 0)
    if resp3:
        _put_lit(writer, _FT_R3_HEAD)
        writer.append_array_header(nr)
    else:
        writer.buffer[writer.offset] = 42; writer.offset += 1
        writer.offset = format_int_to_buf(writer.buffer, writer.offset, Int64(1 + nr * per_doc))
        writer.buffer[writer.offset] = 13; writer.buffer[writer.offset+1] = 10; writer.offset += 2
        writer.buffer[writer.offset] = 58; writer.offset += 1
        writer.offset = format_int_to_buf(writer.buffer, writer.offset, Int64(n_total))
        writer.buffer[writer.offset] = 13; writer.buffer[writer.offset+1] = 10; writer.offset += 2

    # RETURN's entries, resolved once: the token naming the hash field, the
    # token naming it in the reply (its AS alias, or itself), and a value slot
    # reused for every document.
    var ret_name = List[Int]()
    var ret_out = List[Int]()
    var ret_vals = List[GenericValue]()
    var score_entry = -1
    if opts.mode == FT_REPLY_RETURN:
        var t = opts.ret_at
        var t_end = opts.ret_at + opts.ret_n
        while t < t_end:
            var out_t = t
            var step = 1
            if t + 2 < t_end and opts.tokens[t + 1].length == 2 and arg_eq(opts.tokens[t + 1].ptr, 2, "as"):
                out_t = t + 2
                step = 3
            if score_entry < 0 and opts.is_score_name(opts.tokens[t].ptr, opts.tokens[t].length):
                score_entry = len(ret_name)
            ret_name.append(t)
            ret_out.append(out_t)
            ret_vals.append(GenericValue())
            t += step

    for ri2 in range(nr):
        var did = results2[ri2]
        var dist = dist_scores[ri2]

        # Resolve the original hash key: shared slot→key map first (single load).
        var key_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        var key_len = 0
        var result_key = GenericValue()
        if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].hk_keys_buf) and did >= 0 and did < shared_hnsw[].hk_max_elements:
            var _shp = shared_hnsw[].hk_keys_buf + did * 32
            var _kl = Int(_shp[0])
            if _kl > 0 and _kl <= 31:
                key_ptr = _shp + 1
                key_len = _kl
        if key_len == 0 and is_not_null(keyspace):
            # __hk__<id> → original hash key (e.g. "doc:0"); keys >31B land here
            var hk_buf = stack_allocation[30, UInt8]()
            hk_buf[0]=95;hk_buf[1]=95;hk_buf[2]=104;hk_buf[3]=107;hk_buf[4]=95;hk_buf[5]=95 # __hk__
            var hk_end = format_int_to_buf(hk_buf, 6, Int64(did))
            var hk_val = keyspace[].get(GenericValue.borrow(hk_buf, hk_end))
            if not hk_val.is_none():
                result_key = hk_val

        # The doc's hash, when this worker holds it: one keyspace probe with
        # the already-resolved key. NOCONTENT needs none.
        var doc_hash = null_ptr[SlabHashMap, MutUntrackedOrigin]()
        if content and is_not_null(keyspace):
            var key_gv = result_key
            var key_copied = False
            if key_len > 0:
                key_gv = GenericValue.from_ptr(key_ptr, key_len)
                key_copied = True
            if not key_gv.is_none():
                var doc_val = hash_get_live(keyspace, key_gv)
                # gh #394: a key over 23 B is a heap copy made only for this
                # lookup — one leak per FT.SEARCH result before.
                if key_copied: key_gv.free_str_payload()
                if not doc_val.is_none() and doc_val.type.value == ValueType.HASH:
                    doc_hash = doc_val.as_hash().bitcast[SlabHashMap]()

        # Doc key
        if resp3:
            writer.append_map_header(per_doc + 1)    # id, [score], [extra_attributes], values
            _put_lit(writer, _FT_R3_ID)
        if key_len > 0:
            writer.append_bulk_string_response(key_ptr, key_len)
        elif not result_key.is_none():
            writer.append_bulk_value_response(result_key)
        else:
            var id_len_n = int_string_len(Int64(did))
            writer.append_bulk_string_response_header(id_len_n)
            writer.offset = format_int_to_buf(writer.buffer, writer.offset, Int64(did))
            writer.buffer[writer.offset] = 13; writer.buffer[writer.offset+1] = 10; writer.offset += 2

        if opts.with_scores:
            if resp3:
                _put_lit(writer, _FT_R3_TEXT_SCORE)
            else:
                _put_lit(writer, _FT_R2_TEXT_SCORE)

        if opts.mode == FT_REPLY_RETURN:
            var m = 0
            for e in range(len(ret_name)):
                if e == score_entry:
                    m += 1
                    continue
                var v = GenericValue()
                if is_not_null(doc_hash):
                    v = doc_hash[].get(GenericValue.borrow(
                        opts.tokens[ret_name[e]].ptr, opts.tokens[ret_name[e]].length))
                ret_vals[e] = v
                if not v.is_none():
                    m += 1
            if resp3:
                _put_lit(writer, _FT_R3_FIELDS)
                writer.append_map_header(m)
            else:
                writer.append_array_header(2 * m)
            if score_entry >= 0:
                writer.append_bulk_string_response(
                    opts.tokens[ret_out[score_entry]].ptr, opts.tokens[ret_out[score_entry]].length)
                _put_score_bulk(writer, dist)
            for e in range(len(ret_name)):
                if e == score_entry or ret_vals[e].is_none():
                    continue
                writer.append_bulk_string_response(
                    opts.tokens[ret_out[e]].ptr, opts.tokens[ret_out[e]].length)
                writer.append_bulk_value_response(ret_vals[e])
        elif opts.mode == FT_REPLY_ALL:
            # hash_get_live purged expired fields, so size counts the live ones
            var nf = doc_hash[].size if is_not_null(doc_hash) else 0
            if resp3:
                _put_lit(writer, _FT_R3_FIELDS)
                writer.append_map_header(1 + nf)
            else:
                writer.append_array_header(2 * (1 + nf))
            _put_score_name(writer, opts)
            _put_score_bulk(writer, dist)
            if nf > 0:
                for slot in range(doc_hash[].capacity):
                    var mt = doc_hash[].metadata[unsafe_offset=slot]
                    if mt != SlabHashMap.EMPTY and mt != SlabHashMap.DELETED:
                        writer.append_bulk_value_response(doc_hash[].keys[unsafe_offset=slot])
                        writer.append_bulk_value_response(doc_hash[].values[unsafe_offset=slot])
        elif content:
            # gh #357: emit "id" only with the doc's own field value; never
            # substitute the key or slot for a field this worker cannot see.
            var id_field_val = GenericValue()
            if is_not_null(doc_hash):
                var id_name_buf = stack_allocation[2, UInt8]()
                id_name_buf[0] = 105; id_name_buf[1] = 100  # "id"
                id_field_val = doc_hash[].get(GenericValue.borrow(id_name_buf, 2))
            var has_id = not id_field_val.is_none()
            if resp3:
                _put_lit(writer, _FT_R3_FIELDS)
                writer.append_map_header(2 if has_id else 1)
            else:
                writer.buffer[writer.offset] = 42
                writer.buffer[writer.offset+1] = 52 if has_id else 50 # *4 | *2
                writer.buffer[writer.offset+2] = 13; writer.buffer[writer.offset+3] = 10; writer.offset += 4
            if has_id:
                writer.buffer[writer.offset]=36; writer.buffer[writer.offset+1]=50 # $2
                writer.buffer[writer.offset+2]=13; writer.buffer[writer.offset+3]=10
                writer.buffer[writer.offset+4]=105; writer.buffer[writer.offset+5]=100 # id
                writer.buffer[writer.offset+6]=13; writer.buffer[writer.offset+7]=10; writer.offset += 8
                writer.append_bulk_value_response(id_field_val)
            writer.buffer[writer.offset]=36; writer.buffer[writer.offset+1]=53 # $5
            writer.buffer[writer.offset+2]=13; writer.buffer[writer.offset+3]=10
            writer.buffer[writer.offset+4]=115; writer.buffer[writer.offset+5]=99 # sc
            writer.buffer[writer.offset+6]=111; writer.buffer[writer.offset+7]=114 # or
            writer.buffer[writer.offset+8]=101; writer.buffer[writer.offset+9]=13 # e
            writer.buffer[writer.offset+10]=10; writer.offset += 11
            _put_score_bulk(writer, dist)
        if resp3:
            _put_lit(writer, _FT_R3_DOC_END)

    if resp3:
        _put_lit(writer, _FT_R3_TOTAL)
        writer.append_int_response(Int64(n_total))
        _put_lit(writer, _FT_R3_TAIL)


# ── gh #367 / #361: FT.SEARCH metadata filters ───────────────────────────────
# Filter kinds held in FtFilters.types:
comptime FT_FILTER_TAG_EQ = UInt8(0)      # field == value (one value)
comptime FT_FILTER_NUMERIC = UInt8(1)     # lo <= field <= hi
comptime FT_FILTER_TAG_ANY = UInt8(2)     # field is one of "a|b|c"
comptime FT_MAX_FILTERS = 4
# A filtered KNN widens its candidate set up to this many before settling for
# fewer than k rows (the whole index, when it is smaller).
comptime FT_FILTER_MAX_CANDIDATES = 16384


def _bytes_str(p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int, cap: Int = 64) -> String:
    """Printable copy of client bytes for an error message (ASCII, capped)."""
    var s = String("")
    var m = n if n < cap else cap
    for i in range(m):
        var c = Int(p[i])
        s += chr(c) if c >= 32 and c < 127 else "?"
    return s


def _parse_bound(p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int, mut out_v: Float64,
                 mut excl: Bool) -> Bool:
    """One NUMERIC bound, as RediSearch reads it: a number (strtod's, so
    1e400 is infinity), inf / +inf / -inf, and a leading `(` for an exclusive
    bound. `(` used to be refused; the value used to be a Float32, which
    cannot tell 1700000014 from 1700000000 (a timestamp range matched
    documents up to 128 s outside it)."""
    excl = False
    var q = p
    var m = n
    if m > 0 and q[0] == 40:                  # '('
        excl = True
        q = q + 1
        m -= 1
    if m == 0:
        return False
    var pf = parse_redis_double(q, m, DOUBLE_RANGE)
    if not pf.ok:
        return False
    out_v = pf.value
    return True


@always_inline
def _ascii_fold(c: UInt8) -> UInt8:
    return c | 0x20 if c >= 65 and c <= 90 else c


struct FtFilters(Movable):
    """Up to FT_MAX_FILTERS ANDed clauses parsed from FILTER arguments and from
    the prefilter half of a `<prefilter>=>[KNN …]` query.

    Before gh #367 the documented `FILTER @price:[10 50]` form reached a parser
    that only knew `name=value` and `name [lo hi]`, fell through it, and the
    query ran UNFILTERED with no error — a tenant/date/category filter that
    returns rows outside its range while looking applied. Every form is now
    either parsed or refused (`error` is set and the query is not run).

    With the index's schema known (FT.CREATE, or an index file that saved
    it), a clause's field must be in it and of its kind: a range on a NUMERIC
    field, a tag set on a TAG field, as RediSearch requires. A clause on a
    field that is not there used to read the hash anyway, and a range on a
    TAG field read "a" as 0.0."""
    var count: Int
    var types: Array[UInt8, 4]
    var field_lens: Array[UInt8, 4]
    var field_names: Array[Array[UInt8, 32], 4]
    var str_ptrs: Array[UnsafePointer[UInt8, MutUntrackedOrigin], 4]
    var str_lens: Array[UInt8, 4]
    var lo: Array[Float64, 4]
    var hi: Array[Float64, 4]
    var lo_excl: Array[Bool, 4]
    var hi_excl: Array[Bool, 4]
    var slots: Array[Int, 4]
    var seps: Array[UInt8, 4]        # a TAG field's SEPARATOR
    var fold: Array[Bool, 4]         # a TAG field without CASESENSITIVE
    var noindex: Array[Bool, 4]      # a NOINDEX field: matches nothing
    var error: String

    def __init__(out self):
        self.count = 0
        self.types = Array[UInt8, 4](uninitialized=True)
        self.field_lens = Array[UInt8, 4](uninitialized=True)
        self.field_names = Array[Array[UInt8, 32], 4](uninitialized=True)
        self.str_ptrs = Array[UnsafePointer[UInt8, MutUntrackedOrigin], 4](uninitialized=True)
        self.str_lens = Array[UInt8, 4](uninitialized=True)
        self.lo = Array[Float64, 4](uninitialized=True)
        self.hi = Array[Float64, 4](uninitialized=True)
        self.lo_excl = Array[Bool, 4](uninitialized=True)
        self.hi_excl = Array[Bool, 4](uninitialized=True)
        self.slots = Array[Int, 4](uninitialized=True)
        self.seps = Array[UInt8, 4](uninitialized=True)
        self.fold = Array[Bool, 4](uninitialized=True)
        self.noindex = Array[Bool, 4](uninitialized=True)
        for i in range(4):
            self.types[i] = 0
            self.field_lens[i] = 0
            self.str_ptrs[i] = null_ptr[UInt8, MutUntrackedOrigin]()
            self.str_lens[i] = 0
            self.lo[i] = 0.0
            self.hi[i] = 0.0
            self.lo_excl[i] = False
            self.hi_excl[i] = False
            self.slots[i] = -1
            self.seps[i] = 44
            self.fold[i] = True
            self.noindex[i] = False
        self.error = String("")

    def failed(self) -> Bool:
        return self.error.byte_length() > 0

    def fail(mut self, msg: String) -> Bool:
        if not self.failed():
            self.error = msg
        return False

    def _begin(mut self, name: UnsafePointer[UInt8, MutUntrackedOrigin], name_len: Int,
               shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin], want: UInt8,
               legacy: Bool) -> Bool:
        """Claim the next slot for field `name`, which must be a `want` field
        of the index schema when the schema is known. `legacy`: the FILTER
        argument form, whose errors RediSearch words differently."""
        if self.count >= FT_MAX_FILTERS:
            return self.fail("ERR FT.SEARCH supports at most 4 filter clauses")
        if name_len == 0 or name_len > 32:
            return self.fail("ERR invalid filter field name '" + _bytes_str(name, name_len) + "'")
        var fi = self.count
        self.field_lens[fi] = UInt8(name_len)
        for b in range(name_len):
            self.field_names[fi][b] = name[b]
        var slot = _schema_slot(shared, name, name_len)
        if is_not_null(shared) and shared[].schema_known:
            if slot < 0:
                if legacy:
                    return self.fail("Unknown Field '" + _bytes_str(name, name_len) + "'")
                return self.fail("Unknown field `" + _bytes_str(name, name_len) + "`")
            var ty = shared[].schema_field_types[slot]
            if ty != want:
                if want == FT_FIELD_NUMERIC:
                    if legacy:
                        return self.fail("Field is not a numeric field, field: " + _bytes_str(name, name_len))
                    return self.fail("Expected a NUMERIC field near " + _bytes_str(name, name_len))
                return self.fail("Expected a TAG field near " + _bytes_str(name, name_len))
        if slot >= 0:
            self.seps[fi] = shared[].schema_field_seps[slot]
            var fl = shared[].schema_field_flags[slot]
            self.fold[fi] = (fl & FT_FLAG_CASESENSITIVE) == 0
            self.noindex[fi] = (fl & FT_FLAG_NOINDEX) != 0
        self.slots[fi] = slot
        return True

    def add_numeric(mut self, name: UnsafePointer[UInt8, MutUntrackedOrigin], name_len: Int,
                    lo_p: UnsafePointer[UInt8, MutUntrackedOrigin], lo_n: Int,
                    hi_p: UnsafePointer[UInt8, MutUntrackedOrigin], hi_n: Int,
                    shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                    legacy: Bool = False) -> Bool:
        var lo = Float64(0.0)
        var hi = Float64(0.0)
        var lo_x = False
        var hi_x = False
        if not _parse_bound(lo_p, lo_n, lo, lo_x):
            if legacy:
                return self.fail("Bad lower range: " + _bytes_str(lo_p, lo_n))
            return self.fail("ERR invalid numeric filter range for '" + _bytes_str(name, name_len)
                             + "': '" + _bytes_str(lo_p, lo_n) + "' '" + _bytes_str(hi_p, hi_n) + "'")
        if not _parse_bound(hi_p, hi_n, hi, hi_x):
            if legacy:
                return self.fail("Bad upper range: " + _bytes_str(hi_p, hi_n))
            return self.fail("ERR invalid numeric filter range for '" + _bytes_str(name, name_len)
                             + "': '" + _bytes_str(lo_p, lo_n) + "' '" + _bytes_str(hi_p, hi_n) + "'")
        if not self._begin(name, name_len, shared, FT_FIELD_NUMERIC, legacy):
            return False
        var fi = self.count
        self.types[fi] = FT_FILTER_NUMERIC
        self.lo[fi] = lo
        self.hi[fi] = hi
        self.lo_excl[fi] = lo_x
        self.hi_excl[fi] = hi_x
        self.count += 1
        return True

    def add_tag(mut self, name: UnsafePointer[UInt8, MutUntrackedOrigin], name_len: Int,
                val: UnsafePointer[UInt8, MutUntrackedOrigin], val_len: Int,
                shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]) -> Bool:
        if val_len == 0 or val_len > 254:
            return self.fail("ERR invalid tag filter value for '" + _bytes_str(name, name_len) + "'")
        if not self._begin(name, name_len, shared, FT_FIELD_TAG, False):
            return False
        var fi = self.count
        var any = False
        for b in range(val_len):
            if val[b] == 124: any = True; break    # '|'
        self.types[fi] = FT_FILTER_TAG_ANY if any else FT_FILTER_TAG_EQ
        self.str_ptrs[fi] = val
        self.str_lens[fi] = UInt8(val_len)
        self.count += 1
        return True

    def parse_expr(mut self, p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int,
                   shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]) -> Bool:
        """RediSearch clause syntax: `*`, or space-separated (ANDed) clauses
        `@field:[lo hi]` / `@field:{a|b}`, optionally parenthesised. OR between
        clauses, negation and free text are refused, not ignored."""
        var i = 0
        var saw_star = False
        while i < n:
            var c = p[i]
            if c == 32 or c == 40 or c == 41:       # ' ' '(' ')'
                i += 1; continue
            if c == 42:                             # '*' — match all
                saw_star = True; i += 1; continue
            if c != 64:                             # must be '@field:…'
                return self.fail("ERR unsupported filter expression '" + _bytes_str(p, n)
                                 + "' - Pion supports '*' or ANDed @field:[lo hi] / @field:{tag|tag} clauses")
            var name_start = i + 1
            var j = name_start
            while j < n and p[j] != 58:             # ':'
                j += 1
            if j >= n or j + 1 >= n:
                return self.fail("ERR unsupported filter expression '" + _bytes_str(p, n) + "'")
            var name_len = j - name_start
            var open = p[j + 1]
            var close = UInt8(93) if open == 91 else UInt8(125)   # ']' or '}'
            if open != 91 and open != 123:
                return self.fail("ERR unsupported filter expression '" + _bytes_str(p, n)
                                 + "' - expected [lo hi] or {tag}")
            var body = j + 2
            var k = body
            while k < n and p[k] != close:
                k += 1
            if k >= n:
                return self.fail("ERR unterminated filter clause in '" + _bytes_str(p, n) + "'")
            if open == 91:
                var a = body
                while a < k and p[a] == 32: a += 1
                var a_end = a
                while a_end < k and p[a_end] != 32: a_end += 1
                var b = a_end
                while b < k and p[b] == 32: b += 1
                var b_end = b
                while b_end < k and p[b_end] != 32: b_end += 1
                var rest = b_end
                while rest < k and p[rest] == 32: rest += 1
                if a_end == a or b_end == b or rest != k:
                    return self.fail("ERR numeric filter needs exactly two bounds: '" + _bytes_str(p, n) + "'")
                if not self.add_numeric(p + name_start, name_len, p + a, a_end - a, p + b, b_end - b, shared):
                    return False
            else:
                var t0 = body
                var t1 = k
                while t0 < t1 and p[t0] == 32: t0 += 1
                while t1 > t0 and p[t1 - 1] == 32: t1 -= 1
                if not self.add_tag(p + name_start, name_len, p + t0, t1 - t0, shared):
                    return False
            i = k + 1
            if i < n and p[i] == 124:               # '|' between clauses = OR
                return self.fail("ERR OR between filter clauses is not supported: '" + _bytes_str(p, n) + "'")
        _ = saw_star
        return True

    def parse_filter_arg(mut self, tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], j: Int,
                         num_tokens: Int,
                         shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]) -> Int:
        """The arguments after a FILTER keyword at token `j`. Returns how many
        tokens were used after FILTER, or -1 with `error` set. Forms:
          FILTER f lo hi                 (RediSearch's numeric filter)
          FILTER @f:[lo hi] | @f:{tag}   (Pion: RediSearch clause syntax)
          FILTER f=value                 (Pion: TAG equality)
          FILTER f [lo hi]               (Pion: NUMERIC, range as one token)"""
        if j + 1 >= num_tokens:
            _ = self.fail("ERR FILTER needs an argument")
            return -1
        var tp = tokens[j + 1].ptr
        var tl = tokens[j + 1].length
        if tl > 0 and tp[0] == 64:
            return 1 if self.parse_expr(tp, tl, shared) else -1
        var eq = -1
        for b in range(tl):
            if tp[b] == 61: eq = b; break
        if eq > 0:
            return 1 if self.add_tag(tp, eq, tp + eq + 1, tl - eq - 1, shared) else -1
        if j + 2 < num_tokens and tokens[j + 2].length > 1 and tokens[j + 2].ptr[0] == 91:
            var rp = tokens[j + 2].ptr
            var rl = tokens[j + 2].length
            if rp[rl - 1] != 93:
                _ = self.fail("ERR unterminated numeric range '" + _bytes_str(rp, rl) + "'")
                return -1
            var a = 1
            while a < rl - 1 and rp[a] == 32: a += 1
            var a_end = a
            while a_end < rl - 1 and rp[a_end] != 32: a_end += 1
            var b = a_end
            while b < rl - 1 and rp[b] == 32: b += 1
            var b_end = b
            while b_end < rl - 1 and rp[b_end] != 32: b_end += 1
            if a_end == a or b_end == b:
                _ = self.fail("ERR numeric filter needs two bounds: '" + _bytes_str(rp, rl) + "'")
                return -1
            return 2 if self.add_numeric(tp, tl, rp + a, a_end - a, rp + b, b_end - b, shared) else -1
        if j + 3 < num_tokens:
            return 3 if self.add_numeric(tp, tl, tokens[j + 2].ptr, tokens[j + 2].length,
                                         tokens[j + 3].ptr, tokens[j + 3].length, shared, True) else -1
        _ = self.fail("ERR unsupported FILTER argument '" + _bytes_str(tp, tl)
                      + "' - use field lo hi, @field:[lo hi], @field:{tag} or field=value")
        return -1


def _rescore_knn(mut hnsw: HNSWGraph, query: UnsafePointer[Float32, MutUntrackedOrigin],
                 mut ids: List[Int], mut scores: List[Float32]):
    """gh #365: metric-unit scores for the rows a KNN reply returns, in
    ascending order of that score (a few rows can swap relative to the
    quantized ranking; the reply must be sorted by the score it shows)."""
    var n = len(ids)
    if len(scores) < n: n = len(scores)
    if n == 0: return
    if not hnsw.metric_scores(query, ids, scores): return
    for a in range(1, n):
        var b = a
        while b > 0 and scores[b] < scores[b - 1]:
            var ts = scores[b]; scores[b] = scores[b - 1]; scores[b - 1] = ts
            var ti = ids[b]; ids[b] = ids[b - 1]; ids[b - 1] = ti
            b -= 1


def _tag_matches(fv: UnsafePointer[UInt8, MutUntrackedOrigin], fl: Int, sep: UInt8, fold: Bool,
                 q: UnsafePointer[UInt8, MutUntrackedOrigin], ql: Int, any: Bool) -> Bool:
    """A TAG field's value against a clause, as RediSearch indexes it: the
    value is a list split at the field's SEPARATOR (',' by default), each tag
    trimmed of spaces, and compared case-insensitively unless the field is
    CASESENSITIVE (ASCII letters; RediSearch also folds the rest of UTF-8).
    `any`: the clause's `a|b` alternatives, any one matching. The whole value
    was compared byte for byte before, so "a,b" matched neither {a} nor {b},
    and {A} matched nothing stored as "a"."""
    var t0 = 0
    while t0 <= fl:
        var t1 = t0
        while t1 < fl and fv[t1] != sep:
            t1 += 1
        var s0 = t0
        var s1 = t1
        while s0 < s1 and fv[s0] == 32: s0 += 1
        while s1 > s0 and fv[s1 - 1] == 32: s1 -= 1
        var a0 = 0
        while a0 <= ql:
            var a1 = a0
            while a1 < ql and (not any or q[a1] != 124):
                a1 += 1
            var b0 = a0
            var b1 = a1
            if any:
                while b0 < b1 and q[b0] == 32: b0 += 1
                while b1 > b0 and q[b1 - 1] == 32: b1 -= 1
            if s1 - s0 == b1 - b0 and s1 > s0:
                var same = True
                for k in range(s1 - s0):
                    var x = fv[s0 + k]
                    var y = q[b0 + k]
                    if fold:
                        x = _ascii_fold(x)
                        y = _ascii_fold(y)
                    if x != y:
                        same = False
                        break
                if same:
                    return True
            a0 = a1 + 1
        t0 = t1 + 1
    return False


def passes_filters(slot: Int, shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                   keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                   filters: FtFilters) -> Bool:
    """Whether the document at ingest slot `slot` passes every clause, read
    from its hash on this worker (a document another worker holds fails:
    the keyspace is per worker, gh #253).

    TAG: _tag_matches. NUMERIC: the value parsed as a double, inside the
    bounds (`(` excludes one); a value that is not a number fails, where
    RediSearch does not index the document at all. A NOINDEX field matches
    nothing, as in RediSearch. The per-node TAG hash and Float32 copies
    taken at build time are gone: they went stale when a field changed after
    the build (HSET doc cat x, then {x} did not find it)."""
    if filters.count == 0: return True
    var hash_ptr = _slot_hash(shared, keyspace, slot)
    if is_null(hash_ptr): return False
    for fi in range(filters.count):
        if filters.noindex[fi]:
            return False
        var flen = Int(filters.field_lens[fi])
        if flen == 0: continue
        # the field name, as a lookup key: a heap copy (a stack buffer must not
        # reach an out-of-line callee, gh #349)
        var fk_buf = alloc[UInt8](32)
        for bi in range(flen): fk_buf[bi] = filters.field_names[fi][bi]
        var field_val = hash_ptr[].get(GenericValue.borrow(fk_buf, flen))
        fk_buf.free()
        if field_val.is_none(): return False
        var ftype = filters.types[fi]
        if ftype == FT_FILTER_TAG_EQ or ftype == FT_FILTER_TAG_ANY:
            var vp = filters.str_ptrs[fi]
            if is_null(vp): return False
            var fv_len = field_val.string_len()
            var fv_buf = alloc[UInt8](fv_len + 1)
            field_val.copy_to(fv_buf)
            var ok = _tag_matches(fv_buf, fv_len, filters.seps[fi], filters.fold[fi],
                                  vp, Int(filters.str_lens[fi]), ftype == FT_FILTER_TAG_ANY)
            fv_buf.free()
            if not ok: return False
        elif ftype == FT_FILTER_NUMERIC:
            var x: Float64
            if field_val.type.value == ValueType.INT:
                x = Float64(field_val.as_int())
            elif field_val.type.value == ValueType.FLOAT:
                x = Float64(field_val.as_float())
            else:
                var fv_len2 = field_val.string_len()
                if fv_len2 <= 0 or fv_len2 > 64: return False
                var vbuf = alloc[UInt8](fv_len2 + 1)
                field_val.copy_to(vbuf)
                var pf = parse_redis_double(vbuf, fv_len2, DOUBLE_VALUE)
                vbuf.free()
                if not pf.ok: return False
                x = pf.value
            var lo = filters.lo[fi]
            var hi = filters.hi[fi]
            if filters.lo_excl[fi]:
                if not (x > lo): return False
            elif not (x >= lo): return False
            if filters.hi_excl[fi]:
                if not (x < hi): return False
            elif not (x <= hi): return False
    return True


# ---------------------------------------------------------------------------
# Drop-in (phase A): the vectors no built graph holds are searched exactly.
# Before the first build that is every vector; after an automatic build it is
# the delta, the slots written after its snapshot (vector_auto.mojo).
# ---------------------------------------------------------------------------

comptime EXACT_BLOCK = 512    # rows per visibility check and per score block


def exact_knn(
    shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    query: UnsafePointer[Float32, MutUntrackedOrigin],
    k: Int, lo: Int, hi: Int,
    mut ids: List[Int], mut scores: List[Float32],
    filters: FtFilters,
    keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
):
    """The k nearest of ingest slots [lo, hi), exactly, in the metric's units
    (`1 - cos` under COSINE, squared L2 under L2, as _rescore_knn reports),
    ascending. Skips slots not yet committed (commit_ingest), dead slots
    (#46), and, with FILTER, documents that fail it: the filter is read only
    for a row that would enter the top k, so a filter costs what it rejects.

    COSINE vectors were normalized at ingest, so the query is normalized and
    one GEMV per block gives the dot products (Accelerate on macOS). L2 has no
    stored norms, so it reads each row once with the four-chain kernel."""
    ids.clear()
    scores.clear()
    if is_null(shared) or k <= 0 or hi <= lo:
        return
    var fp = shared[].ingest_fp32
    var dim = shared[].pre_dim
    if is_null(fp) or dim <= 0:
        return
    var cosine = shared[].pre_distance_metric == 1
    var q = alloc[Float32](dim)
    var ss = Float32(0.0)
    for d in range(dim):
        ss += query[d] * query[d]
    var inv = Float32(1.0) / sqrt(ss) if cosine and ss > 0.0 else Float32(1.0)
    for d in range(dim):
        q[d] = query[d] * inv
    var blk = alloc[Float32](EXACT_BLOCK)
    var live = alloc[UInt8](EXACT_BLOCK)
    var cap = k if k < hi - lo else hi - lo
    var hs = alloc[Float32](cap)      # max-heap on score: the worst kept row on top
    var hi_ = alloc[Int](cap)
    var hn = 0
    var any_dead = shared[].any_dead()
    var ttls = _ttl_any(keyspace)
    var b0 = lo
    while b0 < hi:
        var nb = EXACT_BLOCK if hi - b0 > EXACT_BLOCK else hi - b0
        var nlive = 0
        for r in range(nb):
            var s = b0 + r
            var ok = shared[].row_visible(s) and not (any_dead and shared[].slot_dead(s))
            live[r] = 1 if ok else 0
            if ok: nlive += 1
        if nlive == 0:
            b0 += nb
            continue
        # the ids above were read before the rows below (commit_ingest's pair)
        external_call["pion_full_fence", NoneType]()
        var rows = fp + b0 * dim
        if cosine:
            scan_dot_f32(rows, nb, dim, q, blk)
            for r in range(nb):
                blk[r] = Float32(1.0) - blk[r]
        else:
            for r in range(nb):
                if live[r] != 0:
                    blk[r] = l2sq_f32_4chain(q, rows + r * dim, dim)
        for r in range(nb):
            if live[r] == 0:
                continue
            var sc = blk[r]
            if hn == cap and not (sc < hs[0]):
                continue
            var slot = b0 + r
            if ttls and _slot_expired(shared, keyspace, slot):
                continue
            if filters.count > 0 and not passes_filters(slot, shared, keyspace, filters):
                continue
            if hn < cap:
                var c = hn                     # sift up
                hn += 1
                while c > 0:
                    var par = (c - 1) // 2
                    if hs[par] >= sc:
                        break
                    hs[c] = hs[par]
                    hi_[c] = hi_[par]
                    c = par
                hs[c] = sc
                hi_[c] = slot
            else:
                var c = 0                      # replace the top, sift down
                while True:
                    var l = 2 * c + 1
                    if l >= hn:
                        break
                    var m = l
                    if l + 1 < hn and hs[l + 1] > hs[l]:
                        m = l + 1
                    if hs[m] <= sc:
                        break
                    hs[c] = hs[m]
                    hi_[c] = hi_[m]
                    c = m
                hs[c] = sc
                hi_[c] = slot
        b0 += nb
    # pop the heap from the back: ascending order
    for _ in range(hn):
        ids.append(0)
        scores.append(0.0)
    var n = hn
    while n > 0:
        var top_s = hs[0]
        var top_i = hi_[0]
        n -= 1
        ids[n] = top_i
        scores[n] = top_s
        var sc = hs[n]
        var si = hi_[n]
        var c = 0
        while True:
            var l = 2 * c + 1
            if l >= n:
                break
            var m = l
            if l + 1 < n and hs[l + 1] > hs[l]:
                m = l + 1
            if hs[m] <= sc:
                break
            hs[c] = hs[m]
            hi_[c] = hi_[m]
            c = m
        if n > 0:
            hs[c] = sc
            hi_[c] = si
    q.free()
    blk.free()
    live.free()
    hs.free()
    hi_.free()


def merge_delta(
    shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    built: Int,
    query: UnsafePointer[Float32, MutUntrackedOrigin],
    k: Int,
    filters: FtFilters,
    keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
    mut ids: List[Int], mut scores: List[Float32],
):
    """Drop-in: the graph holds slots [0, built); slots written since (kept by
    an automatic build) are scanned exactly and merged in, both lists sorted
    ascending, k kept. Nothing to do after FT.OPTIMIZE, which frees the buffer."""
    if is_null(shared) or is_null(shared[].ingest_fp32) or built <= 0:
        return
    # The scores are exact whenever the FP32 rows are held (an automatic
    # build keeps them), with a delta or not. Without a delta the reply used
    # to carry the INT8 codes' distances, which RediSearch never reports:
    # +1.3% on squared L2 at DIM 8, about +0.1 of `1 - cos` on 1536-d.
    _exact_rescore(shared, query, built, ids, scores)
    var rows = shared[].ingested()
    if rows <= built:
        return
    var d_ids = List[Int]()
    var d_scores = List[Float32]()
    exact_knn(shared, query, k, built, rows, d_ids, d_scores, filters, keyspace)
    if len(d_ids) == 0:
        return
    var n_a = len(ids) if len(ids) < len(scores) else len(scores)
    var out_ids = List[Int](capacity=k)
    var out_scores = List[Float32](capacity=k)
    var a = 0
    var b = 0
    while len(out_ids) < k and (a < n_a or b < len(d_ids)):
        if b >= len(d_ids) or (a < n_a and scores[a] <= d_scores[b]):
            out_ids.append(ids[a])
            out_scores.append(scores[a])
            a += 1
        else:
            out_ids.append(d_ids[b])
            out_scores.append(d_scores[b])
            b += 1
    ids = out_ids^
    scores = out_scores^


def _exact_rescore(
    shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    query: UnsafePointer[Float32, MutUntrackedOrigin],
    built: Int,
    mut ids: List[Int], mut scores: List[Float32],
):
    """The graph's results, scored exactly from the FP32 ingest rows, ascending.

    _rescore_knn scores them from the INT8 codes, and the quantization noise
    ADDS to every distance: on 1536-d OpenAI embeddings about +0.1 of `1 - cos`
    (0.264 reported against 0.167 exact for the same document). Ranking among
    graph rows survives that; a merge with exact delta rows does not: every
    delta row within 0.1 of the true neighbours outranked all of them, and
    in-stream recall@100 fell to 0.07-0.10. The delta merge needs both sides
    exact, and in auto mode the FP32 rows are all still here."""
    var n = len(ids) if len(ids) < len(scores) else len(scores)
    var fp = shared[].ingest_fp32
    var dim = shared[].pre_dim
    if n == 0 or is_null(fp) or dim <= 0:
        return
    var cosine = shared[].pre_distance_metric == 1
    var q = alloc[Float32](dim)
    var ss = Float32(0.0)
    for d in range(dim):
        ss += query[d] * query[d]
    var inv = Float32(1.0) / sqrt(ss) if cosine and ss > 0.0 else Float32(1.0)
    for d in range(dim):
        q[d] = query[d] * inv
    for i in range(n):
        var slot = ids[i]
        if slot < 0 or slot >= built or not shared[].row_visible(slot):
            continue
        var row = fp + slot * dim
        if cosine:
            scores[i] = Float32(1.0) - dot_f32_4chain(q, row, dim)
        else:
            scores[i] = l2sq_f32_4chain(q, row, dim)
    q.free()
    for a in range(1, n):
        var b = a
        while b > 0 and scores[b] < scores[b - 1]:
            var ts = scores[b]; scores[b] = scores[b - 1]; scores[b - 1] = ts
            var ti = ids[b]; ids[b] = ids[b - 1]; ids[b - 1] = ti
            b -= 1


def refresh_index_view(
    mut hnsw: HNSWGraph,
    shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    worker_id: Int,
    ready: UInt64,
) -> Bool:
    """Whether this worker's graph serves the published index now, after
    making it so where it can. `ready` is the caller's ACQUIRE load of
    ready_atomic.

    - A borrow of an index that was unpublished, or replaced (publish_seq
      moved on), is given back (_unborrow).
    - The owner (the worker whose graph was published) searches its own
      graph; any other worker borrows the published one, also over a stale
      build of its own.
    - Nothing is live while ready_atomic is 0: the owner's graph may be mid
      build on another thread (vector_auto.mojo), or dropped elsewhere.
    Without a shared view (single-graph callers) the graph's own state rules."""
    if is_null(shared) or is_null(shared[].ready_atomic):
        return hnsw.index_ready and hnsw.entry_point_id != -1
    var seq = shared[].publish_seq[] if is_not_null(shared[].publish_seq) else UInt64(0)
    if hnsw.is_borrowed and (ready == 0 or hnsw.borrow_seq != seq):
        hnsw._unborrow()
    if ready == 0:
        return False
    if shared[].index_owner == worker_id and not hnsw.is_borrowed:
        return hnsw.index_ready and hnsw.entry_point_id != -1
    if not hnsw.is_borrowed and shared[].num_shards <= 1:
        hnsw.borrow_from_shared(shared)
    return hnsw.is_borrowed and hnsw.index_ready and hnsw.entry_point_id != -1


def populate_hnsw_metadata(
    mut hnsw: HNSWGraph,
    shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
):
    """Post-FT.OPTIMIZE index build: BM25 over the TEXT fields. The per-node
    TAG/NUMERIC copies it also took went stale when a field changed after the
    build, and filters read the hash now (passes_filters)."""
    build_bm25(hnsw, shared, keyspace)


def build_bm25(
    mut hnsw: HNSWGraph,
    shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
):
    """Phase 3.2: build the BM25 inverted index from the TEXT schema field.

    gh #139, three changes to what "the corpus" means and how much of it survives:

    * **Doc set** is `union(HNSW nodes, FT.ADDTEXT registrations)`, not the node
      list. FT.ADDTEXT creates no node, so an ADDTEXT-only corpus used to build
      an empty index and `FT.SEARCH … BM25` returned `[0]` with no error.
    * **Text lookup** tries `__hk__<ext_id>` → original key first (so
      `HSET doc:7 body …` works) and falls back to the literal decimal key (so
      `FT.ADDTEXT idx 7 …` works). Only the latter used to be tried.
    * **No caps.** Vocabulary (was 32K terms) and per-doc unique terms (was 64)
      both grow on demand, and the per-doc text buffer is sized to the value
      instead of being a fixed 4KB that `copy_to` happily overran. A 64-term
      ceiling truncates ordinary RAG passages mid-document, which silently
      distorts df/idf and is the kind of thing that shows up later as "our
      ranking disagrees with Okapi on paraphrased queries".

    Two passes: pass 1 interns terms and builds a per-doc forward index; pass 2
    inverts it into per-term postings sorted by term hash for O(log V) lookup."""
    if is_null(shared) or is_null(keyspace): return
    var sc = shared[].schema_field_count
    var text_slot = -1
    var text_field_name_len = 0
    for si in range(sc):
        if shared[].schema_field_types[si] == 1:
            text_slot = si
            text_field_name_len = Int(shared[].schema_field_name_lens[si])
            break
    if text_slot < 0 or text_field_name_len == 0 or text_field_name_len > 32: return
    var max_doc_id = hnsw.max_elements
    if max_doc_id <= 0: return

    # ── Pass 0: assemble the deduped doc set ──────────────────────────────────
    # ext_id indexes bm25_doc_lengths and bm25_scores_scratch, so ids outside
    # [0, max_elements) are dropped rather than scribbling out of bounds.
    var n_cand = hnsw.num_nodes + hnsw.bm25_text_count
    if n_cand <= 0: return
    var seen = alloc[UInt8](max_doc_id)
    unsafe_memset(seen, 0, max_doc_id)
    var doc_ids = alloc[Int32](n_cand)
    var n_docs = 0
    for nidx in range(hnsw.num_nodes):
        var e = Int(hnsw.nodes[nidx].id)
        if e < 0 or e >= max_doc_id or seen[e] != 0: continue
        seen[e] = 1; doc_ids[n_docs] = Int32(e); n_docs += 1
    for tix in range(hnsw.bm25_text_count):
        var e = Int(hnsw.bm25_text_ids[tix])
        if e < 0 or e >= max_doc_id or seen[e] != 0: continue
        seen[e] = 1; doc_ids[n_docs] = Int32(e); n_docs += 1
    seen.free()
    if n_docs == 0:
        doc_ids.free(); return

    # ── Vocabulary: open-addressing hash table + parallel term arrays ─────────
    # Both grow by doubling; ht_cap is kept ≥ 2× vocab_cap so the probe stays short.
    var vocab_cap = 4096
    var ht_cap    = 16384
    var ht_mask   = ht_cap - 1
    var ht_hashes = alloc[UInt32](ht_cap)
    var ht_ids    = alloc[UInt32](ht_cap)
    unsafe_memset(ht_hashes.bitcast[UInt8](), 0, ht_cap * 4)
    unsafe_memset(ht_ids.bitcast[UInt8](), 0, ht_cap * 4)
    var term_hashes_tmp = alloc[UInt32](vocab_cap)
    var term_df         = alloc[UInt32](vocab_cap)
    var term_last_doc   = alloc[Int32](vocab_cap)   # last doc that saw this term
    var term_fwd_pos    = alloc[UInt32](vocab_cap)  # its slot in the forward index
    var vocab_count = 0

    # Forward index: flat [term_id, tf] pairs, sliced per doc by start/len.
    # Flat + per-doc offsets (rather than a fixed stride) is what removes the
    # per-doc term cap without paying max_terms × n_docs of memory.
    var fwd_cap   = 65536
    var fwd_buf   = alloc[UInt32](fwd_cap * 2)
    var fwd_count = 0
    var doc_fwd_start = alloc[UInt32](n_docs)
    var doc_fwd_len   = alloc[UInt32](n_docs)

    var doc_lengths_tmp = alloc[UInt16](max_doc_id)
    unsafe_memset(doc_lengths_tmp.bitcast[UInt8](), 0, max_doc_id * 2)
    var total_tokens = 0

    var key_buf  = alloc[UInt8](32)
    var fk_buf   = alloc[UInt8](32)
    var text_cap = 8192
    var text_buf = alloc[UInt8](text_cap)
    for bi in range(text_field_name_len):
        fk_buf[bi] = shared[].schema_field_names[text_slot][bi]

    # ── Pass 1: tokenize, intern terms, build the forward index ──────────────
    for didx in range(n_docs):
        var ext_id = Int(doc_ids[didx])
        doc_fwd_start[didx] = UInt32(fwd_count)
        doc_fwd_len[didx]   = 0
        # __hk__<ext_id> → original key → HASH, else the literal decimal key
        key_buf[0]=95;key_buf[1]=95;key_buf[2]=104;key_buf[3]=107;key_buf[4]=95;key_buf[5]=95  # __hk__
        var hk_end = format_int_to_buf(key_buf, 6, Int64(ext_id))
        var hk_val = keyspace[].get(GenericValue.borrow(key_buf, hk_end))
        var doc_val: GenericValue
        if not hk_val.is_none():
            doc_val = keyspace[].get(hk_val)
        else:
            var kl = format_int_to_buf(key_buf, 0, Int64(ext_id))
            doc_val = keyspace[].get(GenericValue.borrow(key_buf, kl))
        if doc_val.is_none() or doc_val.type.value != ValueType.HASH: continue
        var hash_ptr = doc_val.as_hash().bitcast[SlabHashMap]()
        if is_null(hash_ptr): continue
        var field_val = hash_ptr[].get(GenericValue.borrow(fk_buf, text_field_name_len))
        if field_val.is_none(): continue
        var tlen = field_val.string_len()
        if tlen <= 0: continue
        if tlen > text_cap:
            text_buf.free()
            text_cap = tlen + 1024
            text_buf = alloc[UInt8](text_cap)
        _ = _copy_value_clamped(field_val, text_buf, text_cap)

        var doc_toks = 0
        var tp = 0
        while tp < tlen:
            var tok = _bm25_next_token(text_buf, tp, tlen)
            tp = tok.next_pos
            if not tok.valid: continue
            doc_toks += 1
            var h = _bm25_hash(text_buf, tok.start, tok.end)

            # Grow before probing: a full table would spin forever, and the
            # rehash has to happen while every live term is still reachable.
            if vocab_count + 1 >= vocab_cap or (vocab_count + 1) * 2 >= ht_cap:
                var new_vcap = vocab_cap * 2
                var g_hashes = alloc[UInt32](new_vcap)
                var g_df     = alloc[UInt32](new_vcap)
                var g_last   = alloc[Int32](new_vcap)
                var g_fpos   = alloc[UInt32](new_vcap)
                unsafe_memcpy(dest=g_hashes, src=term_hashes_tmp, count=vocab_count)
                unsafe_memcpy(dest=g_df,     src=term_df,         count=vocab_count)
                unsafe_memcpy(dest=g_last,   src=term_last_doc,   count=vocab_count)
                unsafe_memcpy(dest=g_fpos,   src=term_fwd_pos,    count=vocab_count)
                term_hashes_tmp.free(); term_df.free(); term_last_doc.free(); term_fwd_pos.free()
                term_hashes_tmp = g_hashes; term_df = g_df
                term_last_doc = g_last; term_fwd_pos = g_fpos
                vocab_cap = new_vcap
                var new_htcap = ht_cap * 2
                var new_mask  = new_htcap - 1
                var g_ht_h = alloc[UInt32](new_htcap)
                var g_ht_i = alloc[UInt32](new_htcap)
                unsafe_memset(g_ht_h.bitcast[UInt8](), 0, new_htcap * 4)
                unsafe_memset(g_ht_i.bitcast[UInt8](), 0, new_htcap * 4)
                for t in range(vocab_count):
                    var th = term_hashes_tmp[t]
                    var s2 = Int(th) & new_mask
                    while g_ht_h[s2] != 0:
                        s2 = (s2 + 1) & new_mask
                    g_ht_h[s2] = th; g_ht_i[s2] = UInt32(t)
                ht_hashes.free(); ht_ids.free()
                ht_hashes = g_ht_h; ht_ids = g_ht_i
                ht_cap = new_htcap; ht_mask = new_mask

            # Intern the term
            var slot = Int(h) & ht_mask
            var tid = 0
            while True:
                if ht_hashes[slot] == 0:
                    tid = vocab_count
                    ht_hashes[slot] = h; ht_ids[slot] = UInt32(tid)
                    term_hashes_tmp[tid] = h
                    term_df[tid] = 0
                    term_last_doc[tid] = -1
                    term_fwd_pos[tid] = 0
                    vocab_count += 1
                    break
                elif ht_hashes[slot] == h:
                    tid = Int(ht_ids[slot]); break
                slot = (slot + 1) & ht_mask

            # First hit in this doc appends a posting; repeats bump its tf.
            # term_last_doc gives O(1) per-doc dedup — the old code linear-scanned
            # the doc's term list, which is why it needed a cap to stay quick.
            if term_last_doc[tid] != Int32(didx):
                term_last_doc[tid] = Int32(didx)
                term_df[tid] += 1
                if fwd_count >= fwd_cap:
                    var new_fcap = fwd_cap * 2
                    var g_fwd = alloc[UInt32](new_fcap * 2)
                    unsafe_memcpy(dest=g_fwd, src=fwd_buf, count=fwd_count * 2)
                    fwd_buf.free(); fwd_buf = g_fwd; fwd_cap = new_fcap
                term_fwd_pos[tid] = UInt32(fwd_count)
                fwd_buf[fwd_count * 2]     = UInt32(tid)
                fwd_buf[fwd_count * 2 + 1] = 1
                fwd_count += 1
                doc_fwd_len[didx] += 1
            else:
                fwd_buf[Int(term_fwd_pos[tid]) * 2 + 1] += 1

        if ext_id >= 0 and ext_id < max_doc_id:
            doc_lengths_tmp[ext_id] = UInt16(doc_toks) if doc_toks < 65535 else UInt16(65535)
        total_tokens += doc_toks

    ht_hashes.free(); ht_ids.free()
    key_buf.free(); fk_buf.free(); text_buf.free()

    if vocab_count == 0 or fwd_count == 0:
        # Nothing indexable — leave bm25_is_built False so FT.SEARCH … BM25
        # reports "not built" instead of an indistinguishable empty result.
        term_hashes_tmp.free(); term_df.free(); term_last_doc.free(); term_fwd_pos.free()
        fwd_buf.free(); doc_fwd_start.free(); doc_fwd_len.free()
        doc_lengths_tmp.free(); doc_ids.free()
        return

    # ── Sort the vocabulary by term hash (heapsort, O(V log V)) ──────────────
    # Was an insertion sort, which only survived because vocab was capped at 32K.
    var sort_idx = alloc[UInt32](vocab_count)
    for i in range(vocab_count): sort_idx[i] = UInt32(i)
    var hs_start = vocab_count // 2 - 1
    while hs_start >= 0:
        _bm25_sift_down(sort_idx, term_hashes_tmp, hs_start, vocab_count)
        hs_start -= 1
    var hs_end = vocab_count - 1
    while hs_end > 0:
        var swp = sort_idx[0]; sort_idx[0] = sort_idx[hs_end]; sort_idx[hs_end] = swp
        _bm25_sift_down(sort_idx, term_hashes_tmp, 0, hs_end)
        hs_end -= 1

    var final_term_hashes    = alloc[UInt32](vocab_count)
    var final_term_idf       = alloc[Float32](vocab_count)
    var final_postings_start = alloc[UInt32](vocab_count)
    var final_postings_count = alloc[UInt32](vocab_count)
    unsafe_memset(final_postings_count.bitcast[UInt8](), 0, vocab_count * 4)
    var tid_to_pos = alloc[UInt32](vocab_count)

    var running_start: UInt32 = 0
    for i in range(vocab_count):
        var tid = Int(sort_idx[i])
        tid_to_pos[tid] = UInt32(i)
        final_term_hashes[i] = term_hashes_tmp[tid]
        var df_val = Int(term_df[tid])
        final_postings_start[i] = running_start
        running_start += UInt32(df_val)
        # Lucene-shape IDF over the BM25 doc count (not num_nodes — those differ
        # the moment a text-only doc is in play). Always positive, no flooring.
        var idf_n = Float32(n_docs) - Float32(df_val) + 0.5
        var idf_d = Float32(df_val) + 0.5
        final_term_idf[i] = log(1.0 + idf_n / idf_d)
    sort_idx.free(); term_hashes_tmp.free(); term_df.free()
    term_last_doc.free(); term_fwd_pos.free()

    # ── Pass 2: invert the forward index into postings ───────────────────────
    var final_postings_buf = alloc[UInt32](fwd_count * 2)
    for didx2 in range(n_docs):
        var ext_id2 = UInt32(Int(doc_ids[didx2]))
        var fs = Int(doc_fwd_start[didx2])
        var fl = Int(doc_fwd_len[didx2])
        for pi in range(fl):
            var old_tid = Int(fwd_buf[(fs + pi) * 2])
            var tf      = fwd_buf[(fs + pi) * 2 + 1]
            var spos    = Int(tid_to_pos[old_tid])
            var widx    = Int(final_postings_start[spos]) + Int(final_postings_count[spos])
            final_postings_buf[widx * 2]     = ext_id2
            final_postings_buf[widx * 2 + 1] = tf
            final_postings_count[spos] += 1
    fwd_buf.free(); doc_fwd_start.free(); doc_fwd_len.free(); tid_to_pos.free()

    if is_not_null(hnsw.bm25_term_hashes):    hnsw.bm25_term_hashes.free()
    if is_not_null(hnsw.bm25_term_idf):       hnsw.bm25_term_idf.free()
    if is_not_null(hnsw.bm25_postings_start): hnsw.bm25_postings_start.free()
    if is_not_null(hnsw.bm25_postings_count): hnsw.bm25_postings_count.free()
    if is_not_null(hnsw.bm25_postings_buf):   hnsw.bm25_postings_buf.free()
    if is_not_null(hnsw.bm25_doc_lengths):    hnsw.bm25_doc_lengths.free()
    if is_not_null(hnsw.bm25_doc_ids):        hnsw.bm25_doc_ids.free()

    hnsw.bm25_term_hashes    = final_term_hashes
    hnsw.bm25_term_idf       = final_term_idf
    hnsw.bm25_postings_start = final_postings_start
    hnsw.bm25_postings_count = final_postings_count
    hnsw.bm25_postings_buf   = final_postings_buf
    hnsw.bm25_doc_lengths    = doc_lengths_tmp
    hnsw.bm25_doc_ids        = doc_ids
    hnsw.bm25_doc_count      = n_docs
    hnsw.bm25_vocab_count    = vocab_count
    hnsw.bm25_total_tokens   = total_tokens
    hnsw.bm25_is_built       = True
    # gh #145: stamp the owning index. The shared view carries the name of the
    # schema that produced these postings; fall back to the local copy for a
    # worker that handled FT.CREATE but has not published yet.
    var _own_len = 0
    if shared[].index_name_len > 0:
        _own_len = shared[].index_name_len
        if _own_len > 64: _own_len = 64
        for _oi in range(_own_len): hnsw.bm25_index_name[_oi] = shared[].index_name[_oi]
    elif hnsw.index_name_len > 0:
        _own_len = hnsw.index_name_len
        if _own_len > 64: _own_len = 64
        for _oi in range(_own_len): hnsw.bm25_index_name[_oi] = hnsw.index_name[_oi]
    hnsw.bm25_index_name_len = _own_len
    # Name the index in the log — an unnamed "BM25: N terms" line is what made
    # a cross-index rebuild look like an unrelated event in production (gh #145).
    var _own_str = String("")
    for _oi2 in range(_own_len): _own_str += chr(Int(hnsw.bm25_index_name[_oi2]))
    if _own_len == 0: _own_str = String("<unnamed>")
    print("BM25 [" + _own_str + "]: " + String(vocab_count) + " terms, "
          + String(n_docs) + " docs, " + String(total_tokens) + " tokens")


def search_bm25(
    mut hnsw: HNSWGraph,
    query_ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
    query_len: Int,
    k: Int,
    mut out_scores: List[Float32],
    k1: Float32 = BM25_DEFAULT_K1,
    b: Float32 = BM25_DEFAULT_B,
) -> List[Int]:
    """BM25 search: tokenize query, accumulate scores, return top-k external IDs.

    gh #139: `k1` (term-frequency saturation) and `b` (length normalisation) are
    per-query so a caller can match whatever Okapi variant its evaluation
    harness uses without a server restart."""
    out_scores.clear()
    var result_ids = List[Int]()
    if not hnsw.bm25_is_built or hnsw.bm25_vocab_count == 0 or hnsw.bm25_doc_count == 0:
        return result_ids^
    var vc = hnsw.bm25_vocab_count
    var nd = hnsw.bm25_doc_count
    var avgdl = Float32(hnsw.bm25_total_tokens) / Float32(nd)
    if avgdl < 1.0: avgdl = 1.0

    # Clear only the doc set, not all max_elements floats — postings can only
    # name ids in the doc set, so everything else is already zero.
    for di in range(nd):
        hnsw.bm25_scores_scratch[Int(hnsw.bm25_doc_ids[di])] = 0.0

    # Tokenize the query with the same tokenizer the index was built with.
    var q_hashes = Array[UInt32, BM25_MAX_QUERY_TERMS](uninitialized=True)
    var n_qtok = 0
    var qp = 0
    while qp < query_len and n_qtok < BM25_MAX_QUERY_TERMS:
        var qtok = _bm25_next_token(query_ptr, qp, query_len)
        qp = qtok.next_pos
        if not qtok.valid: continue
        var h = _bm25_hash(query_ptr, qtok.start, qtok.end)
        var seen = False
        for qi2 in range(n_qtok):
            if q_hashes[qi2] == h: seen = True; break
        if not seen: q_hashes[n_qtok] = h; n_qtok += 1

    for qti in range(n_qtok):
        var qh = q_hashes[qti]
        var lo = 0; var hi = vc - 1; var found_pos = -1
        while lo <= hi:
            var mid = (lo + hi) >> 1
            var mh = hnsw.bm25_term_hashes[mid]
            if mh == qh: found_pos = mid; break
            elif mh < qh: lo = mid + 1
            else: hi = mid - 1
        if found_pos < 0: continue
        var idf     = hnsw.bm25_term_idf[found_pos]
        var p_start = Int(hnsw.bm25_postings_start[found_pos])
        var p_count = Int(hnsw.bm25_postings_count[found_pos])
        for pi in range(p_count):
            var ext_id = Int(hnsw.bm25_postings_buf[(p_start + pi) * 2])
            if ext_id < 0 or ext_id >= hnsw.max_elements: continue
            var tf = Float32(hnsw.bm25_postings_buf[(p_start + pi) * 2 + 1])
            var dl = Float32(hnsw.bm25_doc_lengths[ext_id])
            if dl < 1.0: dl = 1.0
            var tf_norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl))
            hnsw.bm25_scores_scratch[ext_id] += idf * tf_norm

    var scored_ids  = List[Int](capacity=512)
    var scored_vals = List[Float32](capacity=512)
    for di2 in range(nd):
        var ext3 = Int(hnsw.bm25_doc_ids[di2])
        if ext3 < 0 or ext3 >= hnsw.max_elements: continue
        var sc = hnsw.bm25_scores_scratch[ext3]
        if sc > 0.0:
            scored_ids.append(ext3)
            scored_vals.append(sc)
    var ns = len(scored_ids)
    var actual_k = k if k < ns else ns
    for ii in range(actual_k):
        var max_idx = ii; var max_sc = scored_vals[ii]
        for jj in range(ii + 1, ns):
            if scored_vals[jj] > max_sc: max_sc = scored_vals[jj]; max_idx = jj
        if max_idx != ii:
            var tmp_id = scored_ids[ii]; scored_ids[ii] = scored_ids[max_idx]; scored_ids[max_idx] = tmp_id
            var tmp_sc = scored_vals[ii]; scored_vals[ii] = scored_vals[max_idx]; scored_vals[max_idx] = tmp_sc
        result_ids.append(scored_ids[ii])
        out_scores.append(scored_vals[ii])
    return result_ids^


# ---------------------------------------------------------------------------
# FT.INFO command handler
# ---------------------------------------------------------------------------

@always_inline
def _shared_index_name_is(
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
    length: Int,
) -> Bool:
    """gh #403: True when `ptr[:length]` names the index registered in the
    shared view — the one FT.CREATE announced to every worker and the one
    FT.SEARCH would borrow. Truncated to 64 bytes, as FT.CREATE stores it."""
    if is_null(shared_hnsw): return False
    var sl = shared_hnsw[].index_name_len
    var nl = length if length < 64 else 64
    if sl == 0 or sl != nl: return False
    for b in range(nl):
        if shared_hnsw[].index_name[b] != ptr[b]: return False
    return True


@always_inline
def _local_index_name_is(
    hnsw: HNSWGraph, ptr: UnsafePointer[UInt8, MutUntrackedOrigin], length: Int,
) -> Bool:
    var nl = length if length < 64 else 64
    if hnsw.index_name_len == 0 or hnsw.index_name_len != nl: return False
    for b in range(nl):
        if hnsw.index_name[b] != ptr[b]: return False
    return True


@always_inline
def _unnamed_live_index(
    hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    ready: UInt64,
) -> Bool:
    """gh #403 review: a SERVED index with no registered name — a pion.hnsw.0
    saved by a pre-#403 binary whose FT.OPTIMIZE ran off the FT.CREATE worker,
    or an FT.OPTIMIZE given no name. FT.SEARCH has always served it under any
    name (there is nothing to compare), so FT.INFO and FT.DROPINDEX address it
    the same way; otherwise nothing could drop it."""
    if is_null(shared_hnsw) or shared_hnsw[].index_name_len > 0:
        return False
    if ready > 0:
        return True
    return hnsw.index_ready and hnsw.entry_point_id != -1 and hnsw.index_name_len == 0


@always_inline
def handle_ft_info(
    tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    num_tokens: Int,
    mut hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    mut writer: ResponseWriter,
) -> Int:
    """FT.INFO <index_name> — return index metadata.
    Returns the new token index after consuming arguments.

    gh #403: this used to answer from the local worker's `index_ready` and
    never read its argument, so `FT.INFO nosuchidx` returned the served
    index's metadata, and a real index was "Unknown" on any worker that had
    not yet borrowed it (FT.SEARCH borrows; FT.INFO did not). The name is now
    checked against the SHARED view — the registry FT.CREATE writes for every
    worker and FT.DROPINDEX clears — so every worker gives the same answer.
    A created-but-unbuilt index exists (num_docs 0), as in RediSearch."""
    if i + 1 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'ft.info' command")
        return i
    var np = tokens[i + 1].ptr
    var nlen = tokens[i + 1].length
    var ready: UInt64 = 0
    if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].ready_atomic):
        ready = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
            shared_hnsw[].ready_atomic, UInt64(0))
    var known: Bool
    if is_not_null(shared_hnsw):
        known = _shared_index_name_is(shared_hnsw, np, nlen) \
                or _unnamed_live_index(hnsw, shared_hnsw, ready)
    else:
        known = _local_index_name_is(hnsw, np, nlen)
    if not known:
        writer.append_error_response("Unknown index name")
        return i + 1
    # num_docs: what this worker's FT.SEARCH serves. The local graph when it is
    # built under this name (the builder, or a live borrow — a builder's live
    # inserts after FT.OPTIMIZE land only in its local count); otherwise what
    # FT.SEARCH would borrow from the shared view; 0 when not built yet.
    var num_docs = 0
    var local_live = hnsw.index_ready and hnsw.entry_point_id != -1 \
                     and not (hnsw.is_borrowed and ready == 0)
    if local_live and _local_index_name_is(hnsw, np, nlen):
        num_docs = hnsw.num_nodes
    elif ready > 0 and is_not_null(shared_hnsw):
        num_docs = shared_hnsw[].num_nodes
    # Drop-in: vectors FT.SEARCH answers exactly (before any build, or written
    # after an automatic one) are documents of the index too.
    if is_not_null(shared_hnsw) and shared_hnsw[].ingested() > num_docs:
        num_docs = shared_hnsw[].ingested()
    # A slot is not a document: one whose document left (DEL, expiry, a new
    # vector, RENAME, FLUSHALL: #46's dead slots) is not counted. num_docs
    # used to count every slot ever written, so it only grew: 40 documents,
    # two of them deleted and one updated, read 41.
    if is_not_null(shared_hnsw) and shared_hnsw[].any_dead():
        var dead = Int(shared_hnsw[].vec_dead_count[])
        num_docs = num_docs - dead if num_docs > dead else 0
    if writer.proto == 3:
        # RediSearch answers FT.INFO with a map on RESP3, and redis-py 8 parses
        # it as one. Same two fields as RESP2; num_docs is an integer there.
        _put_lit(writer, "%2\r\n+index_name\r\n")
        writer.append_bulk_string_response(np, nlen if nlen < 64 else 64)
        _put_lit(writer, "+num_docs\r\n")
        writer.append_int_response(Int64(num_docs))
        return i + 1
    var hdr1 = String("*4\r\n$10\r\nindex_name\r\n")
    writer.append_to_response(hdr1.unsafe_ptr(), hdr1.byte_length())
    writer.append_bulk_string_response(np, nlen if nlen < 64 else 64)
    var hdr2 = String("$8\r\nnum_docs\r\n")
    writer.append_to_response(hdr2.unsafe_ptr(), hdr2.byte_length())
    var nd_str = String(num_docs)
    writer.append_bulk_string_response(nd_str.unsafe_ptr(), nd_str.byte_length())
    return i + 1


# ---------------------------------------------------------------------------
# FT.DROPINDEX command handler
# ---------------------------------------------------------------------------

def wait_for_readers(
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    worker_id: Int,
):
    """gh #14: return once no other worker can still hold a pointer it read
    before this call: what to wait for between unpublishing a buffer (or
    nulling a shared pointer) and freeing it. Retiring an index, freeing the
    ingest buffer after FT.OPTIMIZE, and the automatic rebuild all use it.
    Call it from a worker inside a dispatch batch (its own slot is skipped)
    or from housekeeping, which runs between batches."""
    # gh #14 — PHASE-2 EPOCH RCU (replaces the phase-1 `usleep(10000)`).
    #
    # The caller's `ready_atomic = 0` RELEASE-store is what makes borrower
    # invalidation kick in on the next ACQUIRE. But a search that ALREADY
    # passed its ACQUIRE-load with ready_atomic=1 may still be using the
    # buffers we are about to free. Phase 1 slept 10 ms, on the reasoning that
    # a search takes sub-ms and 10 ms is "longer than any realistic call".
    # That is a statement about the machine, not a guarantee: a page fault, a
    # descheduled worker, or a much larger index makes it false, and nothing
    # detects when it does.
    #
    # Phase 2 waits for the actual condition instead of a proxy for it. Bump
    # the epoch, then wait until every worker is either IDLE or running at an
    # epoch at least as new as the bump — at that point no worker can still
    # hold a pointer borrowed before it.
    var _rcu_done = False
    if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].worker_epoch) \
       and is_not_null(shared_hnsw[].reclaim_epoch) \
       and shared_hnsw[].worker_epoch_slots > 0:
        # `slots > 0` is not defensive noise: with 0 slots the scan below has
        # nothing to check, reports all-clear, and frees with NO grace at all —
        # strictly worse than the phase-1 sleep it replaces. An empty wait must
        # fall through to the fallback, not skip it.
        #
        # fetch_add returns the OLD value, so the epoch every worker must reach
        # is old + 1. ACQUIRE_RELEASE, not ACQUIRE: the release half is what
        # gives a reader that observes the bumped epoch a synchronizes-with
        # edge to the `ready_atomic = 0` store above it. With a plain acquire
        # RMW that edge does not exist, and a worker could enter at the new
        # epoch while still reading ready_atomic as 1 — which is precisely the
        # case the epoch is supposed to make impossible.
        var _prev = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](
            shared_hnsw[].reclaim_epoch, UInt64(1))
        var _target = _prev + UInt64(1)
        var _slots = shared_hnsw[].worker_epoch_slots
        # 100 ms ceiling in 100 µs steps. The bound exists because a wait with
        # no ceiling turns any bug in the enter/exit bracketing — or a worker
        # blocked in a long syscall — into a hung DROPINDEX, which is a worse
        # failure than the one being fixed. In practice this returns after one
        # or two steps.
        var _spins = 0
        while _spins < 1000:
            var _all_clear = True
            for _w in range(_slots):
                # Skip our OWN slot. This handler runs inside a dispatch batch,
                # so this worker is marked busy at the PRE-bump epoch — waiting
                # on ourselves is an unconditional self-deadlock that would
                # silently degrade every DROPINDEX to the 10 ms fallback.
                if _w == worker_id:
                    continue
                var _ws = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                    shared_hnsw[].worker_epoch.unsafe_offset(_w * 8), UInt64(0))
                # 0 = idle between batches. Otherwise (epoch << 1) | 1; a
                # worker that entered at >= _target started after the bump and
                # therefore re-read ready_atomic, so it cannot be holding a
                # stale borrow.
                if _ws != UInt64(0) and (_ws >> 1) < _target:
                    _all_clear = False
                    break
            if _all_clear:
                _rcu_done = True
                break
            _ = external_call["usleep", Int32](Int32(100))
            _spins += 1
    if not _rcu_done:
        # Either the epoch state is not wired (single-worker paths that never
        # allocated it) or a worker did not drain within the ceiling. Fall back
        # to the phase-1 behaviour rather than freeing early — this is strictly
        # what shipped before, so the fallback cannot be worse than the old
        # code.
        _ = external_call["usleep", Int32](Int32(10000))


def retire_index(
    mut hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    worker_id: Int,
):
    """Unpublish the served index, wait out in-flight readers, reset the graph.

    Used by FT.DROPINDEX, and by FT.CREATE when it replaces a built or
    ingesting index with a different name or DIM. The reset keeps the index
    CONFIG (dim, vector field, ef_construction); FT.CREATE overwrites it next.
    """
    # Order matters: clear shared ready_atomic FIRST so other workers'
    # FT.SEARCH borrow check sees not-ready BEFORE this worker frees the
    # buffers that shared.compact_buffer / .prefix_buffer / .node_*norms
    # may still point at. Borrowers that already borrowed see ready_atomic=0
    # in handle_ft_search and reset their own entry_point_id=-1 (forces
    # re-borrow on next valid publish — see borrower-invalidation below).
    # This narrows the UAF window to "between read of ready_atomic=1 and
    # use of compact_buffer in the same FT.SEARCH call", which is small.
    # A proper RCU/epoch-based reclaim would close it entirely.
    if is_not_null(shared_hnsw):
        if is_not_null(shared_hnsw[].ingest_count):
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                shared_hnsw[].ingest_count, UInt64(0)
            )
        # #46: slot numbering starts over, and so do the tombstones
        shared_hnsw[].new_generation()
        shared_hnsw[].ready = False
        if is_not_null(shared_hnsw[].ready_atomic):
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                shared_hnsw[].ready_atomic, UInt64(0))
        # Reset shard_ready so next FT.OPTIMIZE triggers full rebuild
        var n_sr = shared_hnsw[].num_shards
        if is_not_null(shared_hnsw[].shard_ready) and n_sr > 0:
            for sri in range(n_sr):
                Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                    shared_hnsw[].shard_ready + sri * 8, UInt64(0)
                )
    wait_for_readers(shared_hnsw, worker_id)
    hnsw.reset_index()
    # The index's file goes with it. It stayed, so a restart loaded the dropped
    # index back: FT.INFO knew it again and FT.SEARCH answered from it. Every
    # caller holds AUTO_HOLD, so no automatic build is writing the file.
    var _ip = String("pion.hnsw.0")
    _ = external_call["unlink", Int32](_ip.as_c_string_slice())


def handle_ft_dropindex(
    tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    num_tokens: Int,
    mut hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    mut writer: ResponseWriter,
    worker_id: Int,
    keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
    num_workers: Int,
    mut dd_keys: List[UInt8],
    mut dd_ends: List[Int],
) -> Int:
    """FT.DROPINDEX <index_name> [DD] — reset the HNSW index.
    Returns the new token index after consuming arguments.

    gh #403: the name is checked. This server holds one index, and a drop
    used to reset it whatever name was given — `FT.DROPINDEX b` destroyed a
    serving index `a` (the gh #145 cross-index clobber, by the delete route).
    An unknown name is refused and nothing is touched, as in RediSearch.

    DD deletes the index's documents too, as in RediSearch: their keys come
    back in `dd_keys` (the bytes, each key ending at its `dd_ends` entry,
    with each document's `__hk__<slot>` link), and the caller deletes them
    the way DEL does. DD was read and dropped, so the
    documents stayed; any other argument is refused (it dropped the index).
    Under -w N each worker holds its own documents (gh #253), and one worker
    cannot delete another's: DD is refused there."""
    if i + 1 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'ft.dropindex' command")
        return i
    var dd = False
    if i + 2 < num_tokens:
        if i + 3 < num_tokens or not arg_eq(tokens[i + 2].ptr, tokens[i + 2].length, "dd"):
            var bad = i + 3 if i + 3 < num_tokens and arg_eq(tokens[i + 2].ptr, tokens[i + 2].length, "dd") else i + 2
            writer.append_error_response("Unknown argument `" + _bytes_str(tokens[bad].ptr, tokens[bad].length) + "`")
            return num_tokens - 1
        dd = True
    var np = tokens[i + 1].ptr
    var nlen = tokens[i + 1].length
    var known: Bool
    var ready: UInt64 = 0
    if is_not_null(shared_hnsw):
        if is_not_null(shared_hnsw[].ready_atomic):
            ready = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                shared_hnsw[].ready_atomic, UInt64(0))
        known = _shared_index_name_is(shared_hnsw, np, nlen) \
                or _unnamed_live_index(hnsw, shared_hnsw, ready)
    else:
        known = _local_index_name_is(hnsw, np, nlen)
    if not known:
        writer.append_error_response("Unknown index name")
        return num_tokens - 1
    if dd and num_workers > 1:
        writer.append_error_response(
            "ERR FT.DROPINDEX DD needs -w 1: with --independent-workers each worker holds its own documents")
        return num_tokens - 1
    if dd and is_not_null(shared_hnsw):
        # The documents, before the slots start over: every live slot's key.
        var rows = shared_hnsw[].ingested()
        if shared_hnsw[].num_nodes > rows:
            rows = shared_hnsw[].num_nodes
        if rows > shared_hnsw[].hk_max_elements:
            rows = shared_hnsw[].hk_max_elements
        var kbuf = alloc[UInt8](24)
        var hk = alloc[UInt8](32)
        hk[0]=95;hk[1]=95;hk[2]=104;hk[3]=107;hk[4]=95;hk[5]=95 # __hk__
        for s in range(rows):
            if shared_hnsw[].slot_dead(s):
                continue
            var owned = False
            var key = _slot_key(shared_hnsw, keyspace, s, owned)
            if key.is_none():
                continue
            var kl = key.string_len()
            var kp = key.as_string_safe(kbuf)
            for b in range(kl):
                dd_keys.append(kp[b])
            dd_ends.append(len(dd_keys))
            if owned: key.free_str_payload()
            var he = format_int_to_buf(hk, 6, Int64(s))
            for b in range(he):
                dd_keys.append(hk[b])
            dd_ends.append(len(dd_keys))
        kbuf.free()
        hk.free()
    retire_index(hnsw, shared_hnsw, worker_id)
    # The name leaves the registry, so FT.INFO on EVERY worker now says
    # "Unknown index name" — a borrower's stale local name must not keep a
    # dropped index alive (clients probe FT.INFO to decide whether to
    # FT.CREATE). FT.CREATE or FT.OPTIMIZE <name> registers it again.
    if is_not_null(shared_hnsw):
        shared_hnsw[].index_name_len = 0
    # NOTE: pre_index_ready / pre_dim / pre_vector_field_name are NOT cleared
    # — VectorDBBench reuses the FT.CREATE config across DROP/re-INSERT cycles
    # without re-issuing FT.CREATE.
    writer.append_ok_response()
    return num_tokens - 1


# ---------------------------------------------------------------------------
# FT.OPTIMIZE command handler
# ---------------------------------------------------------------------------

@always_inline
def adopt_index_config(
    mut hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
):
    """gh #407: take the index config FT.CREATE registered in the shared view.

    FT.CREATE writes DIM, the vector field, DISTANCE_METRIC and
    EF_CONSTRUCTION into the CREATING worker's graph, and the accept race
    usually hands FT.OPTIMIZE to another worker, which built with its own
    defaults: a DIM 16 ingest buffer read as 1536-d (garbage graph, every
    query refused), COSINE built as L2 — and publish_to_shared then wrote that
    L2 back over the shared metric — and EF_CONSTRUCTION ignored. These are
    the same assignments FT.CREATE makes on its own worker."""
    if is_null(shared_hnsw) or not shared_hnsw[].pre_index_ready:
        return
    # DIM only when the build will read the shared ingest buffer, which is laid
    # out at pre_dim. A re-optimize with nothing ingested rebuilds this
    # worker's own graph, whose vectors are at its own dim.
    var ingested = UInt64(0)
    if is_not_null(shared_hnsw[].ingest_count):
        ingested = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
            shared_hnsw[].ingest_count, UInt64(0))
    if shared_hnsw[].pre_dim > 0 and ingested > 0:
        hnsw.dim = shared_hnsw[].pre_dim
    var vfl = shared_hnsw[].pre_vector_field_len
    if vfl > 0 and vfl <= 32:
        hnsw.vector_field_len = vfl
        for bi in range(vfl): hnsw.vector_field_name[bi] = shared_hnsw[].pre_vector_field_name[bi]
    hnsw.distance_metric = shared_hnsw[].pre_distance_metric
    if shared_hnsw[].pre_ef_construction > 0:
        hnsw.ef_construction = shared_hnsw[].pre_ef_construction
    # FT.CREATE's PREFIX list and schema, which the build saves with the index
    if is_not_null(shared_hnsw[].pre_prefixes):
        hnsw.prefix_count = shared_hnsw[].pre_prefix_count
        unsafe_memcpy(dest=hnsw.prefixes, src=shared_hnsw[].pre_prefixes, count=PREFIX_MAX * PREFIX_SLOT)
    hnsw.adopt_schema(shared_hnsw)


def _close_ingest(shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin], worker_id: Int):
    """FT.OPTIMIZE ends ingest: free the FP32 ingest buffer. Null the pointers,
    wait out every batch that may have read them (an HSET writing a row, an
    exact scan reading one), then free, and restart the count, which nothing
    may use to read a buffer that is gone."""
    var old_fp32 = shared_hnsw[].ingest_fp32
    var old_ids = shared_hnsw[].ingest_ids
    if is_null(old_fp32) and is_null(old_ids):
        return
    shared_hnsw[].ingest_fp32 = null_ptr[Float32, MutUntrackedOrigin]()
    shared_hnsw[].ingest_ids = null_ptr[Int32, MutUntrackedOrigin]()
    wait_for_readers(shared_hnsw, worker_id)
    if is_not_null(old_fp32):
        old_fp32.free()
    if is_not_null(old_ids):
        old_ids.free()
    if is_not_null(shared_hnsw[].ingest_count):
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shared_hnsw[].ingest_count, UInt64(0))


@always_inline
def handle_ft_optimize(
    tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    num_tokens: Int,
    mut hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
    worker_id: Int,
    mut writer: ResponseWriter,
) -> Int:
    """FT.OPTIMIZE <index_name> — build/rebuild the HNSW index from shared ingest buffer.
    Returns the new token index after consuming arguments."""
    var ci = i
    if ci + 1 < num_tokens: ci += 1  # skip index name arg
    # Nothing to build: no FT.CREATE, no vectors ingested, no graph. It used to
    # build and serve an EMPTY index named by its argument; FT.CREATE of the
    # real index then saw an index being served and, since #46, would not
    # reopen ingest for it — so its documents were never indexed and every
    # search answered nothing. Refuse, as FT.INFO and FT.DROPINDEX do.
    var _pending = UInt64(0)
    if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].ingest_count):
        _pending = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared_hnsw[].ingest_count, UInt64(0))
    var _defined = hnsw.index_name_len > 0 or hnsw.num_nodes > 0 or hnsw.index_ready or _pending > 0 \
        or (is_not_null(shared_hnsw) and (shared_hnsw[].index_name_len > 0 or shared_hnsw[].pre_index_ready))
    if not _defined:
        writer.append_error_response("Unknown index name")
        return ci
    # gh #403: the builder names the graph it builds. FT.CREATE registered the
    # name in the shared view, but the accept race usually hands FT.OPTIMIZE to
    # a different worker, whose local name was empty: `save_to_disk` then wrote
    # an UNNAMED index, and after a warm restart FT.INFO <name> is unknown —
    # a client that answers that with FT.CREATE retires the warm index. The
    # registered name always wins: a worker that borrowed an EARLIER index
    # still carries that one's name, and publish_to_shared would otherwise
    # re-register the stale name over the new one. With none registered
    # (FT.DROPINDEX, then ingest and FT.OPTIMIZE without FT.CREATE) the
    # argument names it.
    if is_not_null(shared_hnsw) and shared_hnsw[].index_name_len > 0:
        hnsw.index_name_len = shared_hnsw[].index_name_len
        for _bi in range(hnsw.index_name_len):
            hnsw.index_name[_bi] = shared_hnsw[].index_name[_bi]
    elif ci > i:
        # No registered name: the argument names the build — also over a stale
        # local name, which a worker that BORROWED the dropped index still
        # carries (reset_index returns early for a borrower), and which
        # publish_to_shared would otherwise register again.
        var _onl = tokens[ci].length if tokens[ci].length < 64 else 64
        hnsw.index_name_len = _onl
        for _bi in range(_onl): hnsw.index_name[_bi] = tokens[ci].ptr[_bi]
    adopt_index_config(hnsw, shared_hnsw)
    # Nothing ingested since the served index was built (FT.OPTIMIZE again, or
    # after the automatic build consumed nothing new): keep that index. Only
    # this worker's text-side state is rebuilt (BM25 and filter metadata are
    # per worker). This used to rebuild and republish this worker's graph,
    # which on a worker that had not built the index was an EMPTY one.
    var _served: UInt64 = 0
    if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].ready_atomic):
        _served = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
            shared_hnsw[].ready_atomic, UInt64(0))
    # After an automatic build the ingest count is not reset, so `_pending` is
    # not 0; but when the served graph already holds every vector ingested,
    # there is nothing to build either. FT.OPTIMIZE then only ends ingest: the
    # buffer is freed (below), as after its own build. A client that sends it
    # 200 ms after its last insert no longer pays for a second build.
    var _covered = _served != 0 and is_not_null(shared_hnsw[].ingest_fp32) \
        and shared_hnsw[].num_nodes >= shared_hnsw[].ingested()
    if (_pending == 0 or _covered) and _served != 0:
        populate_hnsw_metadata(hnsw, shared_hnsw, keyspace)
        if _covered:
            _close_ingest(shared_hnsw, worker_id)
            # The served index is an automatic build's, and its file says a
            # restart reopens ingest (index header word 25, bit 8). This
            # command ends ingest, after a restart too: clear the bit in place.
            # (No automatic build is running: this command holds AUTO_HOLD, so
            # the file is whole.)
            var _ip = String("pion.hnsw.0") + "\0"
            _ = external_call["pion_file_patch_u64", Int32](
                _ip.unsafe_ptr(), Int64(25 * 8), ~UInt64(256), UInt64(0))
        writer.append_ok_response()
        return ci
    # Drop-in: build into this graph's OWN arrays. A graph that borrowed the
    # published index used to build straight into the publisher's arrays.
    # And when this worker's graph IS the published index, unpublish it and
    # wait out its readers first: the build rewrites and frees those buffers.
    # (Other workers search exactly over the ingest buffer meanwhile.)
    if _served != 0 and shared_hnsw[].index_owner == worker_id and not hnsw.is_borrowed:
        shared_hnsw[].ready = False
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shared_hnsw[].ready_atomic, UInt64(0))
        wait_for_readers(shared_hnsw, worker_id)
    hnsw._unborrow()
    try:
        var _dbg_ic = UInt64(0)
        if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].ingest_count):
            _dbg_ic = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared_hnsw[].ingest_count, UInt64(0))
        if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].ingest_count) and _dbg_ic > 0:
            var n_shards = shared_hnsw[].num_shards
            if n_shards > 1:
                # Sharded build: this worker builds its own shard, signals others
                hnsw.build_index_from_shared(shared_hnsw, worker_id, n_shards)
                populate_hnsw_metadata(hnsw, shared_hnsw, keyspace)
                hnsw.index_ready = True
                Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                    shared_hnsw[].shard_ready + worker_id * 8, UInt64(1)
                )
                print("FT.OPTIMIZE: worker " + String(worker_id) + " shard built (" + String(hnsw.num_nodes) + " nodes). Waiting for other shards...")
                # Signal all other workers to build their shards.
                # optimize_trigger is a separately allocated UInt64 pointer —
                # pointer VALUE preserved in SharedHNSWView copies.
                _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELEASE](
                    shared_hnsw[].optimize_trigger, UInt64(1))
                # Wait until all shards are ready.
                # usleep(1ms) per iteration: crosses syscall boundary (memory fence),
                # prevents compiler from hoisting shard_ready loads out of the loop,
                # and reduces cache contention so shard workers build faster.
                # Max wait: 120 seconds (sufficient for 50K 1536-dim vectors).
                var max_wait_iters = 24000  # 120s at 5ms per iter
                var wait_iter = 0
                while wait_iter < max_wait_iters:
                    var all_done = True
                    for si in range(n_shards):
                        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared_hnsw[].shard_ready + si * 8, UInt64(0)) == 0:
                            all_done = False
                            break
                    if all_done: break
                    _ = external_call["usleep", Int32](Int32(5000))  # Fix 4: 5ms — reduce cache-line contention vs 1ms
                    wait_iter += 1
                    # Print dbg_counters every 2s (400 * 5ms) so we can see worker progress without stdout contention
                    if wait_iter % 400 == 0 and is_not_null(shared_hnsw[].dbg_counters):
                        var dc = shared_hnsw[].dbg_counters
                        for wi in range(n_shards):
                            var base = wi * 8
                            print("  DBG worker=" + String(wi) + " poll=" + String(dc[base]) + " saw_trig=" + String(dc[base+1]) + " build_done=" + String(dc[base+2]) + " build_fail=" + String(dc[base+3]) + " ready=" + String(Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared_hnsw[].shard_ready + wi * 8, UInt64(0))))
                print("FT.OPTIMIZE: all " + String(n_shards) + " shards ready after " + String(wait_iter * 5) + "ms. Search live.")
                # Reset ingest_count and trigger; keep shard_ready=1 so
                # workers remain in "serve queries" mode.
                Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                    shared_hnsw[].ingest_count, UInt64(0))
                shared_hnsw[].optimize_trigger[0] = 0
            else:
                # Single-worker path: build full index
                hnsw.build_index_from_shared(shared_hnsw)
                populate_hnsw_metadata(hnsw, shared_hnsw, keyspace)
                hnsw.index_ready = True
                print("FT.OPTIMIZE: " + String(hnsw.num_nodes) + " nodes built")
                # Surface the FT.SEARCH auto-scale plan once at build time so logs
                # show the effective ef that will be applied at query time when
                # num_nodes > 50K (formula: ef = max(ef, 150 * (N/50K)^0.25)).
                if hnsw.num_nodes > 50000:
                    var _scaled_ef = Int(150.0 * sqrt(sqrt(Float64(hnsw.num_nodes) / 50000.0)))
                    print("FT.SEARCH ef auto-scale active: num_nodes=" + String(hnsw.num_nodes) + " effective_ef=" + String(_scaled_ef) + " (baseline 150)")
                # ingest_count is reset inside build_index_from_shared (shard_id=-1 path).
                # P5: Build IVF-PQ index alongside HNSW (single-worker path only).
                # gh #87.1: IVF-PQ build removed. Recall@100 was stuck at 0.59 at
                # 50K × 1536-dim regardless of nprobe/nlist (PQ quantization error
                # at 0.33 bits/dim caused rank inversions). HNSW at ef=150 wins on
                # both QPS and recall.
        else:
            hnsw.build_index()
            populate_hnsw_metadata(hnsw, shared_hnsw, keyspace)
            hnsw.index_ready = True
    except:
        pass
    # #46: a fresh id per build, saved with it and recorded with its
    # tombstones, so replay never applies one build's tombstones to another
    var _bid = UInt64(perf_counter_ns()) ^ (UInt64(Int(external_call["random", Int64]())) << 20)
    hnsw.build_id = _bid if _bid != 0 else UInt64(1)
    hnsw.auto_built = False          # ingest ends here, after a restart too
    # Publish full index + PQ data to shared view (all workers borrow via borrow_from_shared)
    if is_not_null(shared_hnsw):
        hnsw.publish_to_shared(shared_hnsw, worker_id)
        # Free the FP32 ingest buffer: FT.OPTIMIZE means ingest is over. An
        # HSET after it is stored but NOT indexed (add_ingest_vector finds no
        # buffer; INFO vector_unindexed_writes counts it). Nothing else indexes
        # it: the automatic build keeps the buffer instead (vector_auto.mojo).
        _close_ingest(shared_hnsw, worker_id)
    # Phase 1: persist HNSW to disk so warm restarts skip FT.OPTIMIZE.
    # FT.OPTIMIZE accept-races to a random worker; with num_shards=1 (current default,
    # see commit 7b90d41) the worker that handled the request has the full index
    # locally — save unconditionally so persistence is reliable at any -w. Filename
    # stays "pion.hnsw.0" so the load path on worker 0 picks it up cold-start (gh #8).
    _ = worker_id  # retained for sharded path resurrection; see issue if re-enabled
    # gh #211: persist the slot→key map with the graph — without it a warm
    # restart resolves every FT.SEARCH result to a raw slot number.
    if is_not_null(shared_hnsw):
        hnsw.save_to_disk("pion.hnsw.0", shared_hnsw[].hk_keys_buf, shared_hnsw[].hk_max_elements)
    else:
        hnsw.save_to_disk("pion.hnsw.0")
    writer.append_ok_response()
    return ci


# ---------------------------------------------------------------------------
# FT.CREATE command handler
# ---------------------------------------------------------------------------

struct FtCreateSpec(Movable):
    """FT.CREATE, parsed and checked before any state is written (_parse_ft_create).
    Fields are token positions in the command, read by the handler after."""
    var nfields: Int
    var f_name: Array[Int, 16]
    var f_type: Array[UInt8, 16]
    var f_flags: Array[UInt8, 16]
    var f_sep: Array[UInt8, 16]
    var vec_at: Int          # the VECTOR field's name token, -1 without one
    var dim: Int
    var metric: Int          # 0 = L2, 1 = COSINE
    var ef_construction: Int # 0: the configured default
    var skip_initial_scan: Bool

    def __init__(out self):
        self.nfields = 0
        self.f_name = Array[Int, 16](fill=0)
        self.f_type = Array[UInt8, 16](fill=UInt8(0))
        self.f_flags = Array[UInt8, 16](fill=UInt8(0))
        self.f_sep = Array[UInt8, 16](fill=UInt8(44))
        self.vec_at = -1
        self.dim = 0
        self.metric = 0
        self.ef_construction = 0
        self.skip_initial_scan = False


def _strict_count(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], j: Int, n: Int) -> Int:
    """Token j as a count (an integer >= 0), or -1."""
    if j >= n:
        return -1
    var v = parse_int64_strict(tokens[j].ptr, tokens[j].length)
    if not v.ok or v.value < 0 or v.value > 1000000000:
        return -1
    return Int(v.value)


def _parse_ft_create(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, n: Int,
                     max_dim: Int, mut spec: FtCreateSpec) -> String:
    """FT.CREATE's grammar, RediSearch's, checked whole: "" when the command is
    well formed and Pion applies every part of it, else the error to reply.

    The handler used to walk the tokens picking out what it knew and taking
    any other word for a field name, so every option it did not know answered
    +OK and changed nothing: ON JSON (indexed the hashes), FILTER, TEMPORARY,
    AS aliases (the alias was taken as the field: nothing indexed), TYPE
    FLOAT16/INT8/FLOAT64 (read as FLOAT32), a second VECTOR field (the first
    one dropped), a missing SCHEMA, TYPE or DIM, DIM 0, a vector attribute
    count that did not match. Each is now applied or refused.
    Options that change only RediSearch's full-text index (LANGUAGE, SCORE,
    STOPWORDS, NOOFFSETS, ...) are accepted: Pion runs no text query they
    could change. GEO and GEOSHAPE fields are accepted; their queries are
    refused. The errors are RediSearch's words where it has some."""
    if i + 1 >= n:
        return "ERR wrong number of arguments for 'FT.CREATE' command"
    var j = i + 2
    var saw_schema = False
    while j < n:
        var tp = tokens[j].ptr
        var tl = tokens[j].length
        if arg_eq(tp, tl, "schema"):
            saw_schema = True
            j += 1
            break
        elif arg_eq(tp, tl, "on"):
            if j + 1 >= n:
                return "Bad arguments for ON: Expected an argument"
            if arg_eq(tokens[j + 1].ptr, tokens[j + 1].length, "json"):
                return "ERR FT.CREATE ON JSON is not supported: Pion indexes hashes (ON HASH)"
            if not arg_eq(tokens[j + 1].ptr, tokens[j + 1].length, "hash"):
                return "Invalid rule type: " + _bytes_str(tokens[j + 1].ptr, tokens[j + 1].length)
            j += 2
        elif arg_eq(tp, tl, "prefix"):
            var pn = _strict_count(tokens, j + 1, n)
            if pn < 0:
                return "ERR Bad arguments for PREFIX: Could not convert argument to expected type"
            if pn > PREFIX_MAX:
                return "ERR PREFIX: at most " + String(PREFIX_MAX) + " prefixes are supported"
            if j + 1 + pn >= n:
                return "ERR Bad arguments for PREFIX: fewer prefixes than its count"
            for k in range(pn):
                var t = tokens[j + 2 + k]
                if arg_eq(t.ptr, t.length, "schema"):
                    return "ERR Bad arguments for PREFIX: fewer prefixes than its count"
                if t.length >= PREFIX_SLOT:
                    return "ERR PREFIX: a prefix of more than " + String(PREFIX_SLOT - 1) + " bytes is not supported"
            j += 2 + pn
        elif arg_eq(tp, tl, "filter"):
            return "ERR FT.CREATE FILTER is not supported (PREFIX is)"
        elif arg_eq(tp, tl, "temporary"):
            return "ERR FT.CREATE TEMPORARY is not supported"
        elif arg_eq(tp, tl, "language") or arg_eq(tp, tl, "language_field") or arg_eq(tp, tl, "score") \
                or arg_eq(tp, tl, "score_field") or arg_eq(tp, tl, "payload_field"):
            if j + 1 >= n:
                return "Bad arguments for " + _bytes_str(tp, tl) + ": Expected an argument"
            j += 2
        elif arg_eq(tp, tl, "maxtextfields") or arg_eq(tp, tl, "nooffsets") or arg_eq(tp, tl, "nohl") \
                or arg_eq(tp, tl, "nofields") or arg_eq(tp, tl, "nofreqs"):
            j += 1
        elif arg_eq(tp, tl, "skipinitialscan"):
            spec.skip_initial_scan = True
            j += 1
        elif arg_eq(tp, tl, "stopwords"):
            var sn = _strict_count(tokens, j + 1, n)
            if sn < 0 or j + 1 + sn >= n:
                return "Bad arguments for STOPWORDS: Could not convert argument to expected type"
            j += 2 + sn
        else:
            return "Unknown argument `" + _bytes_str(tp, tl) + "`"
    if not saw_schema:
        return "No schema found"
    if j >= n:
        return "Fields arguments are missing"
    while j < n:
        var name_at = j
        var name_p = tokens[j].ptr
        var name_l = tokens[j].length
        j += 1
        if j >= n:
            return "Field `" + _bytes_str(name_p, name_l) + "` has no type"
        if arg_eq(tokens[j].ptr, tokens[j].length, "as"):
            return "ERR FT.CREATE: AS aliases are not supported (name the hash field itself)"
        var ty_p = tokens[j].ptr
        var ty_l = tokens[j].length
        j += 1
        # a field named twice
        if spec.vec_at >= 0 and tokens[spec.vec_at].length == name_l:
            var dup = True
            for b in range(name_l):
                if tokens[spec.vec_at].ptr[b] != name_p[b]: dup = False; break
            if dup:
                return "Duplicate field in schema - " + _bytes_str(name_p, name_l)
        for f in range(spec.nfields):
            var ft = tokens[spec.f_name[f]]
            if ft.length != name_l: continue
            var dup2 = True
            for b in range(name_l):
                if ft.ptr[b] != name_p[b]: dup2 = False; break
            if dup2:
                return "Duplicate field in schema - " + _bytes_str(name_p, name_l)
        if arg_eq(ty_p, ty_l, "vector"):
            if spec.vec_at >= 0:
                return "ERR Pion indexes one VECTOR field per index"
            if name_l > SCHEMA_NAME_MAX:
                return "ERR FT.CREATE: a field name of more than " + String(SCHEMA_NAME_MAX) + " bytes is not supported"
            if j >= n:
                return "Bad arguments for vector similarity algorithm: Expected an argument"
            var algo_p = tokens[j].ptr
            var algo_l = tokens[j].length
            if not (arg_eq(algo_p, algo_l, "hnsw") or arg_eq(algo_p, algo_l, "flat")
                    or arg_eq(algo_p, algo_l, "svs-vamana")):
                return "Bad arguments for vector similarity algorithm: Unknown argument"
            var na = _strict_count(tokens, j + 1, n)
            if na < 0:
                return "Bad arguments for vector similarity number of parameters: Could not convert argument to expected type"
            if na % 2 != 0:
                return "Bad number of arguments for vector similarity index: got " + String(na) \
                    + " but expected even number"
            if j + 2 + na > n:
                return "Expected " + String(na) + " parameters but got " + String(n - j - 2)
            var a = j + 2
            var a_end = j + 2 + na
            var got_type = False
            var got_dim = False
            var got_metric = False
            while a < a_end:
                var kp = tokens[a].ptr
                var kl = tokens[a].length
                var vp = tokens[a + 1].ptr
                var vl = tokens[a + 1].length
                if arg_eq(kp, kl, "type"):
                    got_type = True
                    if not arg_eq(vp, vl, "float32"):
                        if arg_eq(vp, vl, "float64") or arg_eq(vp, vl, "float16") or arg_eq(vp, vl, "bfloat16") \
                                or arg_eq(vp, vl, "int8") or arg_eq(vp, vl, "uint8"):
                            return "ERR vector TYPE " + _bytes_str(vp, vl) + " is not supported (FLOAT32 is)"
                        return "Bad arguments for vector similarity " + _bytes_str(algo_p, algo_l) + " index type"
                elif arg_eq(kp, kl, "dim"):
                    got_dim = True
                    var dv = parse_int64_strict(vp, vl)
                    if not dv.ok:
                        return "Bad arguments for vector similarity " + _bytes_str(algo_p, algo_l) \
                            + " index `DIM`: Could not convert argument to expected type"
                    if dv.value <= 0 or dv.value > 1000000:
                        return "Bad arguments for vector similarity " + _bytes_str(algo_p, algo_l) \
                            + " index `DIM`: Value is outside acceptable bounds"
                    if Int(dv.value) > max_dim:
                        return "ERR DIM " + String(dv.value) + " exceeds this server's vector dimension " \
                            + String(max_dim) + " (start it with --dim " + String(dv.value) + ")"
                    spec.dim = Int(dv.value)
                elif arg_eq(kp, kl, "distance_metric"):
                    got_metric = True
                    if arg_eq(vp, vl, "l2"):
                        spec.metric = 0
                    elif arg_eq(vp, vl, "cosine"):
                        spec.metric = 1
                    else:
                        # IP is the realistic case and it is NOT reducible to this
                        # search: the quantizer is affine with an offset, which
                        # cancels in a difference but not in a dot product (gh #271).
                        return "ERR unsupported DISTANCE_METRIC (supported: L2, COSINE)"
                elif arg_eq(kp, kl, "ef_construction") or arg_eq(kp, kl, "m") or arg_eq(kp, kl, "ef_runtime") \
                        or arg_eq(kp, kl, "initial_cap") or arg_eq(kp, kl, "block_size") \
                        or arg_eq(kp, kl, "construction_window_size") or arg_eq(kp, kl, "graph_max_degree") \
                        or arg_eq(kp, kl, "search_window_size") or arg_eq(kp, kl, "training_threshold") \
                        or arg_eq(kp, kl, "leanvec_dim"):
                    var iv = parse_int64_strict(vp, vl)
                    if not iv.ok or iv.value < 0:
                        return "Bad arguments for vector similarity " + _bytes_str(algo_p, algo_l) + " index `" \
                            + _bytes_str(kp, kl) + "`: Could not convert argument to expected type"
                    if arg_eq(kp, kl, "ef_construction") and iv.value > 0:
                        spec.ef_construction = Int(iv.value)
                elif arg_eq(kp, kl, "epsilon") or arg_eq(kp, kl, "reduce"):
                    if not parse_redis_double(vp, vl, DOUBLE_VALUE).ok:
                        return "Bad arguments for vector similarity " + _bytes_str(algo_p, algo_l) + " index `" \
                            + _bytes_str(kp, kl) + "`: Could not convert argument to expected type"
                elif arg_eq(kp, kl, "compression"):
                    pass
                else:
                    return "Bad arguments for algorithm " + _bytes_str(algo_p, algo_l) + ": " + _bytes_str(kp, kl)
                a += 2
            # DISTANCE_METRIC may be left out: it is L2 then, as Pion has
            # always documented. RediSearch requires it.
            _ = got_metric
            var what = String("")
            if not got_type: what = "TYPE"
            elif not got_dim: what = "DIM"
            if what.byte_length() > 0:
                return "Missing mandatory parameter: cannot create " + _bytes_str(algo_p, algo_l) \
                    + " index without specifying " + what + " argument"
            spec.vec_at = name_at
            j = a_end
            while j < n and arg_eq(tokens[j].ptr, tokens[j].length, "indexmissing"):
                j += 1
            continue
        var ftype = UInt8(0)
        if arg_eq(ty_p, ty_l, "text"): ftype = FT_FIELD_TEXT
        elif arg_eq(ty_p, ty_l, "tag"): ftype = FT_FIELD_TAG
        elif arg_eq(ty_p, ty_l, "numeric"): ftype = FT_FIELD_NUMERIC
        elif arg_eq(ty_p, ty_l, "geo"): ftype = FT_FIELD_GEO
        elif arg_eq(ty_p, ty_l, "geoshape"): ftype = FT_FIELD_GEOSHAPE
        else:
            return "Invalid field type for field `" + _bytes_str(name_p, name_l) + "`"
        if spec.nfields >= SCHEMA_MAX:
            return "ERR Pion indexes at most " + String(SCHEMA_MAX) + " fields besides the vector"
        if name_l == 0 or name_l > SCHEMA_NAME_MAX:
            return "ERR FT.CREATE: a field name of more than " + String(SCHEMA_NAME_MAX) + " bytes is not supported"
        var flags = UInt8(0)
        var sep = UInt8(44)
        # the field's options; any other word starts the next field (as in RediSearch)
        while j < n:
            var op = tokens[j].ptr
            var ol = tokens[j].length
            if arg_eq(op, ol, "sortable"):
                flags |= FT_FLAG_SORTABLE
                j += 1
                if j < n and arg_eq(tokens[j].ptr, tokens[j].length, "unf"):
                    flags |= FT_FLAG_UNF
                    j += 1
            elif arg_eq(op, ol, "noindex"):
                flags |= FT_FLAG_NOINDEX
                j += 1
            elif arg_eq(op, ol, "indexmissing") or arg_eq(op, ol, "indexempty") \
                    or arg_eq(op, ol, "withsuffixtrie"):
                j += 1
            elif ftype == FT_FIELD_TAG and arg_eq(op, ol, "separator"):
                if j + 1 >= n or tokens[j + 1].length != 1:
                    return "Tag separator must be a single character. Got `" \
                        + (_bytes_str(tokens[j + 1].ptr, tokens[j + 1].length) if j + 1 < n else String("")) + "`"
                sep = tokens[j + 1].ptr[0]
                j += 2
            elif ftype == FT_FIELD_TAG and arg_eq(op, ol, "casesensitive"):
                flags |= FT_FLAG_CASESENSITIVE
                j += 1
            elif ftype == FT_FIELD_TEXT and (arg_eq(op, ol, "weight") or arg_eq(op, ol, "phonetic")):
                if j + 1 >= n:
                    return "Bad arguments for " + _bytes_str(op, ol) + ": Expected an argument"
                j += 2
            elif ftype == FT_FIELD_TEXT and arg_eq(op, ol, "nostem"):
                j += 1
            elif ftype == FT_FIELD_GEOSHAPE and (arg_eq(op, ol, "flat") or arg_eq(op, ol, "spherical")):
                j += 1
            else:
                break
        var f = spec.nfields
        spec.f_name[f] = name_at
        spec.f_type[f] = ftype
        spec.f_flags[f] = flags
        spec.f_sep[f] = sep
        spec.nfields = f + 1
    return String("")


@always_inline
def handle_ft_create(
    tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    num_tokens: Int,
    mut hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    config: PionConfig,
    mut writer: ResponseWriter,
    worker_id: Int,
    mut backfill: Bool,
) -> Int:
    """FT.CREATE <index> ... ON HASH PREFIX ... SCHEMA <fields> — create vector/text index.
    Returns the new token index after consuming arguments. `backfill` comes back
    True when the index was created with ingest open and without SKIPINITIALSCAN:
    the caller then indexes the hashes already written (vector_backfill_now).
    Whenever ingest is open, not only when this call opened the buffer:
    FT.DROPINDEX keeps the buffer and starts the slots over, so the FT.CREATE
    after it opens nothing and still finds every hash unindexed. A hash this
    index already holds is skipped, so a defensive re-create walks the keyspace
    and changes nothing."""
    var ci = i
    backfill = False
    # The whole command is parsed and checked first (_parse_ft_create), before
    # a byte of index state is written: a refusal must leave the server as it
    # found it, and the loop below overwrites the index name, the vector field
    # and the shared schema as it goes (gh #232's rule, gh #271's pre-scan).
    var spec = FtCreateSpec()
    var perr = _parse_ft_create(tokens, ci, num_tokens, config.vector.dimensions, spec)
    if perr.byte_length() > 0:
        writer.append_error_response(perr)
        return num_tokens - 1
    var skip_initial_scan = spec.skip_initial_scan
    var metric = spec.metric   # 0 = L2 (default: what an index without the keyword has always been)
    # A different index (name or DIM) replacing one that is built or has
    # vectors queued for ingest RETIRES it first, exactly as FT.DROPINDEX
    # would. This used to overwrite the live graph's name and `dim` in place:
    # gh #145's name check then passed for the new index, a search ran a
    # DIM-384 query over the old DIM-4 slots (out of bounds — the heap
    # corruption surfaced as a SIGSEGV in the next FT.OPTIMIZE), it answered
    # with the OTHER index's documents, and queued DIM-4 vectors were built
    # into the DIM-384 graph. Re-issuing FT.CREATE for the index already
    # being served (same name, same DIM) is left alone: clients do that
    # defensively. tests/test_ft_create_redimension.py.
    # gh #407 review: every graph's vector slabs and query scratch are sized
    # for the server's startup --dim and never grow; the parser refused a
    # larger DIM.
    var new_dim = spec.dim
    # PREFIX <n> <p1> ... <pn> (RediSearch): only a hash whose key starts with
    # one of them is indexed, byte for byte. It was accepted and never applied,
    # so an index took every hash with its vector field, and FT.CREATE's
    # backfill (1.0.1) took them all at once. Resolved here, before any state
    # is written, so a refused list changes nothing. n = 0, or an empty
    # prefix, is every key, as in RediSearch.
    var new_prefixes = alloc[UInt8](PREFIX_MAX * PREFIX_SLOT)
    unsafe_memset(new_prefixes, 0, PREFIX_MAX * PREFIX_SLOT)
    var new_prefix_count = 0
    var prefix_err = String("")
    var pj = ci + 2
    while pj < num_tokens:
        if arg_eq(tokens[pj].ptr, tokens[pj].length, "schema"):
            break
        if arg_eq(tokens[pj].ptr, tokens[pj].length, "prefix"):
            var pn = -1
            if pj + 1 < num_tokens and tokens[pj + 1].length > 0 and tokens[pj + 1].length <= 9:
                pn = 0
                for b in range(tokens[pj + 1].length):
                    var c = Int(tokens[pj + 1].ptr[b])
                    if c < 48 or c > 57:
                        pn = -1
                        break
                    pn = pn * 10 + (c - 48)
            if pn < 0:
                prefix_err = "ERR Bad arguments for PREFIX: Could not convert argument to expected type"
            elif pn > PREFIX_MAX:
                prefix_err = "ERR PREFIX: at most " + String(PREFIX_MAX) + " prefixes are supported"
            elif pj + 1 + pn >= num_tokens:
                prefix_err = "ERR Bad arguments for PREFIX: fewer prefixes than its count"
            else:
                var every_key = pn == 0
                for k in range(pn):
                    var t = tokens[pj + 2 + k]
                    if arg_eq(t.ptr, t.length, "schema"):
                        # the count ran into the schema (RediSearch fails too)
                        prefix_err = "ERR Bad arguments for PREFIX: fewer prefixes than its count"
                        break
                    if t.length >= PREFIX_SLOT:
                        prefix_err = "ERR PREFIX: a prefix of more than " + String(PREFIX_SLOT - 1) \
                            + " bytes is not supported"
                        break
                    if t.length == 0:
                        every_key = True
                        continue
                    var dst = new_prefixes + new_prefix_count * PREFIX_SLOT
                    dst[0] = UInt8(t.length)
                    unsafe_memcpy(dest=dst + 1, src=t.ptr, count=t.length)
                    new_prefix_count += 1
                if every_key:
                    unsafe_memset(new_prefixes, 0, PREFIX_MAX * PREFIX_SLOT)
                    new_prefix_count = 0
            break
        pj += 1
    if prefix_err.byte_length() > 0:
        new_prefixes.free()
        writer.append_error_response(prefix_err)
        return num_tokens - 1
    var prefix_differs = False
    if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].pre_prefixes):
        if new_prefix_count != shared_hnsw[].pre_prefix_count:
            prefix_differs = True
        else:
            for b in range(new_prefix_count * PREFIX_SLOT):
                if new_prefixes[b] != shared_hnsw[].pre_prefixes[b]:
                    prefix_differs = True
                    break
    var name_differs = False
    if ci + 1 < num_tokens:
        var tnl = tokens[ci+1].length if tokens[ci+1].length < 64 else 64
        if tnl != hnsw.index_name_len:
            name_differs = True
        else:
            for tni in range(tnl):
                if tokens[ci+1].ptr[tni] != hnsw.index_name[tni]:
                    name_differs = True
                    break
    var has_state = hnsw.entry_point_id != -1 or hnsw.num_nodes > 0
    if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].ingest_count):
        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                shared_hnsw[].ingest_count, UInt64(0)) > UInt64(0):
            has_state = True
    # Another PREFIX list is another index too: the served one holds documents
    # the new list leaves out.
    if has_state and (name_differs or prefix_differs or (new_dim > 0 and new_dim != hnsw.dim)):
        # Stop routing HSET vectors into the old-DIM ingest buffer before the
        # epoch wait, then free it once no batch can hold it; the tail of this
        # handler allocates one sized for the new DIM and re-enables routing
        # (the free-after-wait discipline FT.OPTIMIZE uses for this buffer).
        if is_not_null(shared_hnsw):
            shared_hnsw[].pre_index_ready = False
        retire_index(hnsw, shared_hnsw, worker_id)
        if is_not_null(shared_hnsw):
            if is_not_null(shared_hnsw[].ingest_fp32):
                shared_hnsw[].ingest_fp32.free()
                shared_hnsw[].ingest_fp32 = null_ptr[Float32, MutUntrackedOrigin]()
            if is_not_null(shared_hnsw[].ingest_ids):
                shared_hnsw[].ingest_ids.free()
                shared_hnsw[].ingest_ids = null_ptr[Int32, MutUntrackedOrigin]()
    if prefix_differs:
        # Every worker's HSET reads the list to route a vector: stop routing
        # and wait out the batches that may be reading it before rewriting it
        # (the retire above already did both). pre_index_ready is set again
        # below, after the fence.
        if shared_hnsw[].pre_index_ready:
            shared_hnsw[].pre_index_ready = False
            wait_for_readers(shared_hnsw, worker_id)
        unsafe_memcpy(dest=shared_hnsw[].pre_prefixes, src=new_prefixes, count=PREFIX_MAX * PREFIX_SLOT)
        shared_hnsw[].pre_prefix_count = new_prefix_count
        external_call["pion_full_fence", NoneType]()
    new_prefixes.free()
    # Store index name (token immediately after FT.CREATE)
    if ci + 1 < num_tokens:
        var nm_ptr = tokens[ci+1].ptr
        var nm_len = tokens[ci+1].length if tokens[ci+1].length < 64 else 64
        hnsw.index_name_len = nm_len
        for bi in range(nm_len): hnsw.index_name[bi] = nm_ptr[bi]
    # gh #407: an FT.CREATE without EF_CONSTRUCTION gets the configured
    # default, not whatever an earlier FT.CREATE left on this worker — the
    # value is published below, and a stale one would travel with it.
    hnsw.ef_construction = config.vector.ef_construction
    # The vector field, EF_CONSTRUCTION and DIM, and the schema, as parsed.
    if spec.vec_at >= 0:
        var vt = tokens[spec.vec_at]
        hnsw.vector_field_len = vt.length
        for bi in range(vt.length): hnsw.vector_field_name[bi] = vt.ptr[bi] | 0x20
    if spec.ef_construction > 0:
        hnsw.ef_construction = spec.ef_construction
    if new_dim > 0 and new_dim != hnsw.dim:
        hnsw.dim = new_dim
    var schema_count = spec.nfields
    if is_not_null(shared_hnsw):
        for f in range(spec.nfields):
            var ft = tokens[spec.f_name[f]]
            shared_hnsw[].schema_field_types[f] = spec.f_type[f]
            shared_hnsw[].schema_field_flags[f] = spec.f_flags[f]
            shared_hnsw[].schema_field_seps[f] = spec.f_sep[f]
            shared_hnsw[].schema_field_name_lens[f] = UInt8(ft.length)
            # gh #367: verbatim — `| 0x20` made `a_b` unmatchable
            for bi in range(ft.length): shared_hnsw[].schema_field_names[f][bi] = ft.ptr[bi]
        shared_hnsw[].schema_known = True
    var j = num_tokens
    if is_not_null(shared_hnsw):
        shared_hnsw[].schema_field_count = schema_count
    # gh #271: publish the metric to BOTH the local graph (which quantizes the
    # query) and the shared view (whose add_ingest_vector takes the HSET route
    # on whichever worker won the accept). If only one carried it, ingest and
    # query would sit on opposite sides of the normalization and the index
    # would answer nonsense — the failure this issue reported, reintroduced.
    hnsw.distance_metric = UInt8(metric)
    if is_not_null(shared_hnsw):
        shared_hnsw[].pre_distance_metric = UInt8(metric)
    hnsw.index_ready = True
    # V3.1: propagate index config to shared view so any worker routes HSET vectors
    if is_not_null(shared_hnsw):
        shared_hnsw[].pre_index_ready = True
        shared_hnsw[].pre_vector_field_len = hnsw.vector_field_len
        shared_hnsw[].pre_dim = hnsw.dim
        shared_hnsw[].pre_ef_construction = hnsw.ef_construction   # gh #407
        # Copy field name via memcpy — Array[bi] assignment through UnsafePointer
        # dereference is unreliable in Mojo 0.26.3 (produces corrupted bytes).
        unsafe_memcpy(dest=shared_hnsw[].pre_vector_field_name.unsafe_ptr(), src=hnsw.vector_field_name.unsafe_ptr(), count=hnsw.vector_field_len)
        # Propagate index name to shared view so event-loop shard builds can copy it
        shared_hnsw[].index_name_len = hnsw.index_name_len
        unsafe_memcpy(dest=shared_hnsw[].index_name.unsafe_ptr(), src=hnsw.index_name.unsafe_ptr(), count=hnsw.index_name_len)
    ci = j - 1
    # V3.1: publish vector field name to shared view so all workers can route HSETs
    if is_not_null(shared_hnsw):
        shared_hnsw[].pre_dim = hnsw.dim
        shared_hnsw[].pre_vector_field_len = hnsw.vector_field_len
        unsafe_memcpy(dest=shared_hnsw[].pre_vector_field_name.unsafe_ptr(), src=hnsw.vector_field_name.unsafe_ptr(), count=hnsw.vector_field_len)

        # Pre-allocate shared ingest buffer (capacity = max_elements for dataset scale).
        # Not while a built index is served (#46): re-issuing FT.CREATE for it
        # (clients do, defensively) restarted slot numbering under the live
        # index, so the next HSET renamed its slot 0. An HSET after FT.OPTIMIZE
        # is stored but not indexed — the documented ingest contract.
        # An EMPTY served index has nothing to protect: ingest reopens for it.
        var _serving = is_not_null(shared_hnsw[].ready_atomic) and Atomic[Scalar[DType.uint64]].fetch_add[
            ordering=Ordering.ACQUIRE](shared_hnsw[].ready_atomic, UInt64(0)) != 0 and shared_hnsw[].num_nodes > 0
        if is_null(shared_hnsw[].ingest_fp32) and not _serving:
            var cap = hnsw.max_elements
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                shared_hnsw[].ingest_count, UInt64(0)
            )
            shared_hnsw[].new_generation()      # #46: slots start again from 0
            shared_hnsw[].open_ingest(cap, hnsw.dim)
        backfill = not skip_initial_scan and is_not_null(shared_hnsw[].ingest_fp32)

    writer.append_ok_response()
    return ci


# ---------------------------------------------------------------------------
# FT.ADDTEXT command handler
# ---------------------------------------------------------------------------

@always_inline
def handle_ft_addtext(
    tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    num_tokens: Int,
    mut hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
    mut dispatcher: CommandDispatcher,
    mut scache: SemanticCache,
    mut writer: ResponseWriter,
) -> Int:
    """FT.ADDTEXT index doc_id text — index `text` for both BM25 and FT.SEARCHTEXT.

    Writes the text into the keyspace HASH `<doc_id>` under the schema's TEXT
    field (where build_bm25 reads it) and embeds it into the semantic-cache HNSW
    (where FT.SEARCHTEXT reads it).

    gh #139: the doc is also registered in the BM25 doc set. FT.ADDTEXT creates
    no HNSW node, and build_bm25 used to walk nodes only — so an ADDTEXT-only
    corpus produced an empty inverted index and `FT.SEARCH … BM25` returned `[0]`
    with no error. BM25 registration needs a non-negative integer doc_id because
    hits come back as integer ext_ids; a non-integer id still works for
    FT.SEARCHTEXT but stays invisible to BM25.

    Returns the new token index after consuming arguments."""
    var ci = i
    if ci + 3 < num_tokens:
        # ci+1 = index name (skip), ci+2 = doc_id, ci+3 = text value
        var doc_tok  = tokens[ci + 2]
        var text_tok = tokens[ci + 3]
        # Find TEXT schema field name from shared schema
        var tf_name_len = 0
        var tf_name_buf = alloc[UInt8](32)
        if is_not_null(shared_hnsw):
            var _sc = shared_hnsw[].schema_field_count
            for _si in range(_sc):
                if shared_hnsw[].schema_field_types[_si] == 1:
                    tf_name_len = Int(shared_hnsw[].schema_field_name_lens[_si])
                    for _bi in range(tf_name_len):
                        tf_name_buf[_bi] = shared_hnsw[].schema_field_names[_si][_bi]
                    break
        if tf_name_len > 0:
            # HSET doc_id <text_field_name> <text>
            var doc_key  = GenericValue.from_ptr(doc_tok.ptr, doc_tok.length)
            var hash_ptr2: UnsafePointer[SlabHashMap, MutUntrackedOrigin]
            var existing2 = keyspace[].get(doc_key)
            if existing2.is_none():
                # gh #140: ObjectPool.acquire() past capacity returns raw
                # alloc[T](1) — memory that was never constructed as a
                # SlabHashMap. reset()/set() then walk a garbage capacity and a
                # null metadata pointer, so the 1001st distinct FT.ADDTEXT doc
                # (pool capacity is 1000, state.mojo) SIGSEGV'd the worker.
                # Same guard the HSET fast path already uses.
                if dispatcher.hash_map_pool[].head < dispatcher.hash_map_pool[].capacity:
                    hash_ptr2 = dispatcher.hash_map_pool[].acquire()
                    hash_ptr2[].reset()
                else:
                    hash_ptr2 = alloc[SlabHashMap](1)
                    hash_ptr2.unsafe_write(SlabHashMap(16))
                var _new_hval = GenericValue()
                _new_hval.type = ValueType(ValueType.HASH)
                _new_hval.set_ptr(hash_ptr2.bitcast[NoneType]())
                keyspace[].set(GenericValue.borrow(doc_tok.ptr, doc_tok.length), _new_hval)
            else:
                hash_ptr2 = existing2.as_hash().bitcast[SlabHashMap]()
            if is_not_null(hash_ptr2):
                var field_gv = GenericValue.from_ptr(tf_name_buf, tf_name_len)
                var val_gv   = GenericValue.from_ptr(text_tok.ptr, text_tok.length)
                hash_ptr2[].set(field_gv, val_gv)
            # gh #139: join the BM25 doc set. Length ≤ 18 keeps the accumulator
            # inside Int64 for any all-digit id we would accept anyway.
            var _did_ok = doc_tok.length > 0 and doc_tok.length <= 18
            var _did_val = 0
            if _did_ok:
                for _di in range(doc_tok.length):
                    var _dc = Int(doc_tok.ptr[_di])
                    if _dc < 48 or _dc > 57:
                        _did_ok = False
                        break
                    _did_val = _did_val * 10 + (_dc - 48)
            if _did_ok:
                hnsw.bm25_register_text_doc(_did_val)
        tf_name_buf.free()
        # Phase 4: also embed text + add to semantic cache HNSW for FT.SEARCHTEXT
        if scache.enabled and scache.count < CACHE_MAX_ENTRIES:
            # gh #140: this is the document side of an asymmetric retriever.
            var _emb_ok = scache.embed_into(
                text_tok.ptr,
                text_tok.length,
                scache.embed_buf,
                is_query=False)
            if _emb_ok:
                # #29: the entry belongs to THIS index, and its doc id keeps
                # every byte (non-ASCII bytes used to be dropped).
                var _idx_tok = tokens[ci + 1]
                var _owner = scache.owner_id(bytes_name("t:", _idx_tok.ptr, _idx_tok.length), True)
                try:
                    _ = scache.add_entry(_owner, bytes_name("", doc_tok.ptr, doc_tok.length))
                except:
                    pass
        ci += 3
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR FT.ADDTEXT index doc_id text")
    return ci


# ---------------------------------------------------------------------------
# FT.SEARCHTEXT command handler
# ---------------------------------------------------------------------------

@always_inline
def handle_ft_searchtext(
    tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    num_tokens: Int,
    mut scache: SemanticCache,
    mut writer: ResponseWriter,
) raises -> Int:
    """FT.SEARCHTEXT <index> <text> [K <k>] — embed text, search per-worker text HNSW.
    Returns the new token index after consuming arguments."""
    var ci = i
    if not scache.enabled:
        writer.append_error_response("ERR embedding not enabled (set EmbeddingConfig.enabled=True)")
    elif scache.count == 0:
        writer.append_empty_array_response()
    elif ci + 2 < num_tokens:
        ci += 1
        var _st_idx_tok = tokens[ci]   # #29: only this index's documents
        var _st_text_tok = tokens[ci + 1]
        var _st_k = 10
        ci += 1  # consume text token
        # Parse optional K argument
        if ci + 1 < num_tokens and tokens[ci + 1].length == 1 and (tokens[ci + 1].ptr[0]|0x20)==107:
            if ci + 2 < num_tokens:
                var _kv = 0
                for _ki in range(tokens[ci + 2].length):
                    var _kc = Int(tokens[ci + 2].ptr[_ki])
                    if _kc >= 48 and _kc <= 57: _kv = _kv * 10 + (_kc - 48)
                if _kv > 0: _st_k = _kv
                ci += 2
        var _st_owner = scache.owner_id(bytes_name("t:", _st_idx_tok.ptr, _st_idx_tok.length), False)
        # Embed query text
        var _st_ok = _st_owner >= 0 and scache.embed_into(
            _st_text_tok.ptr,
            _st_text_tok.length,
            scache.embed_buf)
        if not _st_ok:
            writer.append_empty_array_response()
        else:
            # gh #140: K was clamped by a beam hardcoded to 32. #29: the search
            # is now exact over this index's documents, so K is honoured as
            # long as the index holds that many.
            var _st_scores = List[Float32]()
            var _st_results = scache.search_owner(
                scache.embed_buf, _st_owner, _st_k, _st_scores)
            var _st_n = len(_st_results)
            var _st_hdr = String("*") + String(_st_n) + String("\r\n")
            writer.append_to_response(_st_hdr.unsafe_ptr(), _st_hdr.byte_length())
            for _ri in range(_st_n):
                var _rid = _st_results[_ri]
                if _rid >= 0 and _rid < scache.count:
                    var _rdoc = scache.responses[_rid]
                    writer.append_bulk_string_response(_rdoc.unsafe_ptr(), _rdoc.byte_length())
                else:
                    writer.append_null_response()
    else:
        writer.append_error_response("ERR FT.SEARCHTEXT index text [K k]")
    return ci


# ---------------------------------------------------------------------------
# FT.HYBRID command handler
# ---------------------------------------------------------------------------

def handle_ft_hybrid(
    tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    num_tokens: Int,
    mut hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
    mut scache: SemanticCache,
    fd: Int32,
    mut writer: ResponseWriter,
    server: TCPServer,
    kq: Int32,
    consumed_bytes: Int,
    worker_id: Int,
) raises -> Int:
    """FT.HYBRID <index> <text_query> <vector_blob> [K <k>] [ALPHA <a>] [RERANK <host> <port> <model> [<top_n>]]
    Single-command BM25+vector fusion with Reciprocal Rank Fusion.

    gh #70 (RERANK): optional opt-in cross-encoder rerank stage. When the
    keyword is present, the top `<top_n>` (default 50) RRF candidates are
    re-scored by `POST <host>:<port>/v1/score` (superlinked/sie wire shape:
    `{model, query, documents:[...]}` → `{scores:[...]}`). Failure modes —
    network error, non-200, malformed JSON — silently fall back to the RRF
    order; substrate availability beats rerank availability.

    Returns -1 to signal caller to return consumed_bytes (response already flushed)."""
    var ci = i
    # Need at least: FT.HYBRID <index> <text> <blob>
    if ci + 4 > num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'FT.HYBRID' command")
        return ci

    ci += 1  # skip FT.HYBRID token
    var _idx_tok = tokens[ci]; ci += 1   # index name
    var _text_tok = tokens[ci]; ci += 1  # text query
    var _blob_tok = tokens[ci]; ci += 1  # vector blob

    # Parse optional args: K <k>, ALPHA <a>, RERANK <host> <port> <model> [<top_n>]
    var _k = 10
    var _alpha: Float32 = 0.5  # 0=vector-only, 1=BM25-only, 0.5=equal weight
    var _bm25_k1 = BM25_DEFAULT_K1
    var _bm25_b  = BM25_DEFAULT_B
    var _rerank_enabled = False
    var _rerank_host = String("")
    var _rerank_port: Int = 0
    var _rerank_model = String("")
    var _rerank_top_n: Int = 50
    while ci < num_tokens:
        var _otok = tokens[ci]
        var _ol = _otok.length; var _op = _otok.ptr
        # K keyword (1 byte)
        if _ol == 1 and (_op[0]|0x20) == 107 and ci + 1 < num_tokens:
            ci += 1
            var _kv = 0
            for _ki in range(tokens[ci].length):
                var _kc = Int(tokens[ci].ptr[_ki])
                if _kc >= 48 and _kc <= 57: _kv = _kv * 10 + (_kc - 48)
            if _kv > 0: _k = _kv
        # ALPHA keyword (5 bytes)
        elif _ol == 5 and (_op[0]|0x20)==97 and (_op[1]|0x20)==108 and (_op[2]|0x20)==112 and (_op[3]|0x20)==104 and (_op[4]|0x20)==97 and ci + 1 < num_tokens:
            ci += 1
            # Parse float: simple digit.digit format
            var _av: Float32 = 0.0; var _adec = False; var _adiv: Float32 = 1.0
            for _ai in range(tokens[ci].length):
                var _ac = Int(tokens[ci].ptr[_ai])
                if _ac == 46: _adec = True  # '.'
                elif _ac >= 48 and _ac <= 57:
                    if _adec:
                        _adiv *= 10.0; _av += Float32(_ac - 48) / _adiv
                    else:
                        _av = _av * 10.0 + Float32(_ac - 48)
            if _av >= 0.0 and _av <= 1.0: _alpha = _av
        # gh #139: K1 (2 bytes) / B (1 byte) — BM25 ranking parameters.
        elif _ol == 2 and (_op[0]|0x20)==107 and _op[1]==49 and ci + 1 < num_tokens:
            ci += 1
            var _k1v = parse_filter_float(tokens[ci].ptr, tokens[ci].length)
            if _k1v >= 0.0 and _k1v <= 100.0: _bm25_k1 = _k1v
        elif _ol == 1 and (_op[0]|0x20)==98 and ci + 1 < num_tokens:
            ci += 1
            var _bv = parse_filter_float(tokens[ci].ptr, tokens[ci].length)
            if _bv >= 0.0 and _bv <= 1.0: _bm25_b = _bv
        # gh #70: RERANK keyword (6 bytes) — requires <host> <port> <model>; optional <top_n>.
        elif _ol == 6 and (_op[0]|0x20)==114 and (_op[1]|0x20)==101 and (_op[2]|0x20)==114 and (_op[3]|0x20)==97 and (_op[4]|0x20)==110 and (_op[5]|0x20)==107 and ci + 3 < num_tokens:
            var _rh_tok = tokens[ci + 1]
            var _rp_tok = tokens[ci + 2]
            var _rm_tok = tokens[ci + 3]
            _rerank_host = String()
            for _hi in range(_rh_tok.length):
                _rerank_host += chr(Int(_rh_tok.ptr[_hi]))
            var _rport = 0
            for _pi in range(_rp_tok.length):
                var _pc = Int(_rp_tok.ptr[_pi])
                if _pc >= 48 and _pc <= 57: _rport = _rport * 10 + (_pc - 48)
            _rerank_port = _rport
            _rerank_model = String()
            for _mi in range(_rm_tok.length):
                _rerank_model += chr(Int(_rm_tok.ptr[_mi]))
            ci += 3  # host, port, model
            # Optional <top_n> immediately after model — must be all-digits.
            if ci + 1 < num_tokens:
                var _tn_tok = tokens[ci + 1]
                var _all_digits = _tn_tok.length > 0
                for _di in range(_tn_tok.length):
                    var _dc = Int(_tn_tok.ptr[_di])
                    if _dc < 48 or _dc > 57: _all_digits = False; break
                if _all_digits:
                    var _tn = 0
                    for _di2 in range(_tn_tok.length):
                        _tn = _tn * 10 + (Int(_tn_tok.ptr[_di2]) - 48)
                    if _tn > 0: _rerank_top_n = _tn
                    ci += 1
            if _rerank_host.byte_length() > 0 and _rerank_port > 0 and _rerank_model.byte_length() > 0:
                _rerank_enabled = True
        ci += 1

    # The published index, borrowed or owned (refresh_index_view); before a
    # build (drop-in) the vector half is exact over the ingest buffer.
    var _h_ready: UInt64 = 0
    if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].ready_atomic):
        _h_ready = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
            shared_hnsw[].ready_atomic, UInt64(0))
    var _h_live = refresh_index_view(hnsw, shared_hnsw, worker_id, _h_ready)
    var _h_rows = shared_hnsw[].ingested() if is_not_null(shared_hnsw) else 0
    # Validate: need HNSW index for vector search
    if not _h_live and _h_rows == 0 and not hnsw.index_ready:
        writer.append_error_response("ERR no index — run FT.CREATE + FT.OPTIMIZE first")
        writer.flush_response(fd, server, kq)
        return -1

    # gh #145: the BM25 half of the fusion is server-global — refuse rather than
    # quietly degrading to a vector-only result under another index's postings.
    if bm25_owner_mismatch(hnsw, _idx_tok.ptr, _idx_tok.length):
        writer.append_error_response(
            bm25_owner_error(hnsw, _idx_tok.ptr, _idx_tok.length))
        writer.flush_response(fd, server, kq)
        return -1

    # Validate vector blob dimension
    var _blob_ptr = _blob_tok.ptr
    var _blob_len = _blob_tok.length
    var _h_dim = hnsw.dim
    if not _h_live and is_not_null(shared_hnsw) and shared_hnsw[].pre_dim > 0:
        _h_dim = shared_hnsw[].pre_dim
    if _blob_len != _h_dim * 4:
        writer.append_error_response("ERR vector blob size mismatch (expected " + String(_h_dim * 4) + " bytes)")
        writer.flush_response(fd, server, kq)
        return -1

    # ── Run vector search ──
    var _vec_ids = List[Int]()
    var _vec_scores = List[Float32]()
    if _h_live:
        var _hybrid_ef = hnsw.ef_runtime
        if hnsw.num_nodes > 50000:
            var _hr = Float64(hnsw.num_nodes) / 50000.0
            var _hef = Int(150.0 * sqrt(sqrt(_hr)))
            if _hef > _hybrid_ef: _hybrid_ef = _hef
        try:
            var _fp32 = _blob_ptr.bitcast[Float32]()
            _vec_ids = hnsw.search_fp32_scored(_fp32, _k * 4, _vec_scores, _hybrid_ef)
            # drop-in: rows written after an automatic build, in metric units
            if _h_rows > hnsw.num_nodes and is_not_null(shared_hnsw[].ingest_fp32):
                _rescore_knn(hnsw, _fp32, _vec_ids, _vec_scores)
                merge_delta(shared_hnsw, hnsw.num_nodes, _fp32, _k * 4, FtFilters(),
                            keyspace, _vec_ids, _vec_scores)
        except: pass
    elif _h_rows > 0:
        exact_knn(shared_hnsw, _blob_ptr.bitcast[Float32](), _k * 4, 0, _h_rows,
                  _vec_ids, _vec_scores, FtFilters(), keyspace)

    # ── Run BM25 search ──
    var _bm25_ids = List[Int]()
    var _bm25_scores = List[Float32]()
    if _text_tok.length > 0:
        _bm25_ids = search_bm25(hnsw, _text_tok.ptr, _text_tok.length, _k * 4, _bm25_scores,
                                _bm25_k1, _bm25_b)
    # #46: documents that left the keyspace leave the candidates
    drop_dead(shared_hnsw, _vec_ids, _vec_scores, len(_vec_ids))
    drop_dead(shared_hnsw, _bm25_ids, _bm25_scores, len(_bm25_ids))

    # ── RRF fusion with configurable alpha ──
    var _rrf_ids = List[Int]()
    var _rrf_scores = List[Float32]()

    if len(_vec_ids) == 0 and len(_bm25_ids) == 0:
        write_ft_search_response(writer, _rrf_ids, _rrf_scores, keyspace, shared_hnsw)
        writer.flush_response(fd, server, kq)
        return -1

    # Build union of candidate IDs
    var _all_ids = List[Int]()
    for _vi in range(len(_vec_ids)): _all_ids.append(_vec_ids[_vi])
    for _bi in range(len(_bm25_ids)):
        var _bid = _bm25_ids[_bi]
        var _dup = False
        for _vi2 in range(len(_vec_ids)):
            if _vec_ids[_vi2] == _bid: _dup = True; break
        if not _dup: _all_ids.append(_bid)

    var _n_cands = len(_all_ids)
    var _rrf_tmp = List[Float32](capacity=_n_cands)
    var _vec_weight = 1.0 - _alpha  # alpha=0 → full vector, alpha=1 → full BM25
    var _bm25_weight = _alpha

    for _ci in range(_n_cands):
        var _cid = _all_ids[_ci]
        var _rrf_sc: Float32 = 0.0
        # Vector rank contribution
        for _vr in range(len(_vec_ids)):
            if _vec_ids[_vr] == _cid:
                _rrf_sc += _vec_weight / (60.0 + Float32(_vr))
                break
        # BM25 rank contribution
        for _br in range(len(_bm25_ids)):
            if _bm25_ids[_br] == _cid:
                _rrf_sc += _bm25_weight / (60.0 + Float32(_br))
                break
        _rrf_tmp.append(_rrf_sc)

    # gh #70 RERANK: if enabled, selection-sort the larger _rerank_top_n
    # window first, fetch doc TEXT for each, call SIE /v1/score, then re-sort
    # by SIE scores and take top _k. On any failure (no TEXT field, no doc
    # hash, SIE unreachable, malformed JSON), silently fall back to RRF order
    # — substrate availability > rerank availability.
    if _rerank_enabled:
        var _select_n = _rerank_top_n if _rerank_top_n < _n_cands else _n_cands
        if _select_n < _k: _select_n = _k if _k < _n_cands else _n_cands

        # Selection sort top _select_n by RRF (mutates _all_ids + _rrf_tmp in place).
        for _ii in range(_select_n):
            var _mx = _ii; var _ms = _rrf_tmp[_ii]
            for _jj in range(_ii + 1, _n_cands):
                if _rrf_tmp[_jj] > _ms: _ms = _rrf_tmp[_jj]; _mx = _jj
            if _mx != _ii:
                var _ti = _all_ids[_ii]; _all_ids[_ii] = _all_ids[_mx]; _all_ids[_mx] = _ti
                var _ts = _rrf_tmp[_ii]; _rrf_tmp[_ii] = _rrf_tmp[_mx]; _rrf_tmp[_mx] = _ts

        # Find the schema's TEXT field name (type=1). FT.HYBRID requires a TEXT
        # field for BM25 already; if missing we couldn't have RRF'd a result.
        var _tf_name_len = 0
        var _tf_name_buf = stack_allocation[32, UInt8]()
        if is_not_null(shared_hnsw):
            var _sc = shared_hnsw[].schema_field_count
            for _si in range(_sc):
                if shared_hnsw[].schema_field_types[_si] == 1:
                    _tf_name_len = Int(shared_hnsw[].schema_field_name_lens[_si])
                    for _bi in range(_tf_name_len):
                        _tf_name_buf[_bi] = shared_hnsw[].schema_field_names[_si][_bi]
                    break

        # Collect doc text pointers in their RRF-sorted order. Packing into
        # contiguous arrays the C-ish way so RerankClient.score can iterate.
        var _doc_count = 0
        var _doc_ptrs = alloc[UnsafePointer[UInt8, MutUntrackedOrigin]](_select_n)
        var _doc_lens = alloc[Int](_select_n)
        var _doc_bufs = List[UnsafePointer[UInt8, MutUntrackedOrigin]]()  # owns the text buffers
        if _tf_name_len > 0 and is_not_null(keyspace):
            for _ri in range(_select_n):
                var _ext_id = _all_ids[_ri]
                # Match write_ft_search_response's __hk__<id> → hash key lookup.
                var _hash_ptr = null_ptr[SlabHashMap, MutUntrackedOrigin]()
                var _hk_buf = stack_allocation[30, UInt8]()
                _hk_buf[0]=95; _hk_buf[1]=95; _hk_buf[2]=104; _hk_buf[3]=107; _hk_buf[4]=95; _hk_buf[5]=95
                var _hk_end = format_int_to_buf(_hk_buf, 6, Int64(_ext_id))
                var _hk_key = GenericValue.from_ptr(_hk_buf, _hk_end)
                var _hk_val = keyspace[].get(_hk_key)
                if not _hk_val.is_none():
                    var _doc_val = keyspace[].get(_hk_val)
                    if not _doc_val.is_none() and _doc_val.type.value == ValueType.HASH:
                        _hash_ptr = _doc_val.as_hash().bitcast[SlabHashMap]()
                if is_null(_hash_ptr):
                    # Fallback: numeric ext_id as the doc key directly.
                    var _key_buf = stack_allocation[24, UInt8]()
                    var _rev_buf = stack_allocation[20, UInt8]()
                    var _key_len: Int
                    var _tmp = _ext_id
                    if _tmp == 0:
                        _key_buf[0] = 48; _key_len = 1
                    else:
                        var _nd = 0
                        while _tmp > 0:
                            _rev_buf[_nd] = UInt8(_tmp % 10) + 48; _nd += 1; _tmp //= 10
                        for _di in range(_nd): _key_buf[_di] = _rev_buf[_nd - 1 - _di]
                        _key_len = _nd
                    var _num_key = GenericValue.from_ptr(_key_buf, _key_len)
                    var _doc_val2 = keyspace[].get(_num_key)
                    if not _doc_val2.is_none() and _doc_val2.type.value == ValueType.HASH:
                        _hash_ptr = _doc_val2.as_hash().bitcast[SlabHashMap]()
                if is_null(_hash_ptr):
                    continue
                var _fk = GenericValue.from_ptr(_tf_name_buf, _tf_name_len)
                var _fv = _hash_ptr[].get(_fk)
                _fk.free_str_payload()   # gh #394
                if _fv.is_none():
                    continue
                var _tlen = _fv.string_len()
                if _tlen > 4096: _tlen = 4096
                var _tbuf = alloc[UInt8](_tlen) if _tlen > 0 else null_ptr[UInt8, MutUntrackedOrigin]()
                if _tlen > 0:
                    _fv.copy_to(_tbuf)
                _doc_ptrs[_doc_count] = _tbuf
                _doc_lens[_doc_count] = _tlen
                _doc_bufs.append(_tbuf)
                # Move the RRF-sorted entry into the densely-packed prefix so
                # _all_ids[0.._doc_count] aligns with _doc_ptrs[0.._doc_count].
                if _doc_count != _ri:
                    var _tid = _all_ids[_doc_count]; _all_ids[_doc_count] = _all_ids[_ri]; _all_ids[_ri] = _tid
                    var _trrf = _rrf_tmp[_doc_count]; _rrf_tmp[_doc_count] = _rrf_tmp[_ri]; _rrf_tmp[_ri] = _trrf
                _doc_count += 1

        # Call SIE /v1/score. The client and the score array survive the
        # success-branch only; failure ⇒ free + fall through to RRF order.
        if _doc_count > 0:
            var _sie_scores = alloc[Float32](_doc_count)
            var _client = RerankClient(_rerank_host, _rerank_port, _rerank_model)
            var _query_ptr_ext = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_text_tok.ptr))
            var _doc_ptrs_ext = UnsafePointer[UnsafePointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin](unsafe_from_address=Int(_doc_ptrs))
            var _doc_lens_ext = UnsafePointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(_doc_lens))
            var _scores_ext = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_sie_scores))
            var _ok = _client.score(
                _query_ptr_ext, _text_tok.length,
                _doc_ptrs_ext, _doc_lens_ext, _doc_count,
                _scores_ext,
            )
            if _ok:
                # Overwrite the RRF scores for the packed window with SIE
                # scores, then selection-sort top _k by descending SIE score.
                for _i in range(_doc_count):
                    _rrf_tmp[_i] = _sie_scores[_i]
                var _resort_n = _k if _k < _doc_count else _doc_count
                for _i in range(_resort_n):
                    var _mx = _i; var _ms = _rrf_tmp[_i]
                    for _j in range(_i + 1, _doc_count):
                        if _rrf_tmp[_j] > _ms: _ms = _rrf_tmp[_j]; _mx = _j
                    if _mx != _i:
                        var _ti = _all_ids[_i]; _all_ids[_i] = _all_ids[_mx]; _all_ids[_mx] = _ti
                        var _ts = _rrf_tmp[_i]; _rrf_tmp[_i] = _rrf_tmp[_mx]; _rrf_tmp[_mx] = _ts
            _sie_scores.free()

        for _b in _doc_bufs:
            if is_not_null(_b): _b.free()
        _doc_ptrs.free()
        _doc_lens.free()

        # Emit final top-k from whatever order we ended on (SIE if score()
        # succeeded; RRF otherwise).
        var _emit_k = _k if _k < _select_n else _select_n
        if _emit_k > _doc_count and _doc_count > 0: _emit_k = _doc_count
        for _ii in range(_emit_k):
            _rrf_ids.append(_all_ids[_ii])
            _rrf_scores.append(_rrf_tmp[_ii])
    else:
        # Selection sort top-k by RRF score (existing behavior, unchanged).
        var _actual_k = _k if _k < _n_cands else _n_cands
        for _ii in range(_actual_k):
            var _mx = _ii; var _ms = _rrf_tmp[_ii]
            for _jj in range(_ii + 1, _n_cands):
                if _rrf_tmp[_jj] > _ms: _ms = _rrf_tmp[_jj]; _mx = _jj
            if _mx != _ii:
                var _ti = _all_ids[_ii]; _all_ids[_ii] = _all_ids[_mx]; _all_ids[_mx] = _ti
                var _ts = _rrf_tmp[_ii]; _rrf_tmp[_ii] = _rrf_tmp[_mx]; _rrf_tmp[_mx] = _ts
            _rrf_ids.append(_all_ids[_ii])
            _rrf_scores.append(_rrf_tmp[_ii])

    write_ft_search_response(writer, _rrf_ids, _rrf_scores, keyspace, shared_hnsw)
    writer.flush_response(fd, server, kq)
    return -1


# FT.SEARCH command handler
# ---------------------------------------------------------------------------

@always_inline
def handle_ft_search(
    tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    num_tokens: Int,
    mut hnsw: HNSWGraph,
    shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
    keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
    worker_id: Int,
    shard_query_seq: UnsafePointer[UInt64, MutUntrackedOrigin],
    mut scratch_dists: List[Float32],
    mut p3_batch_start_node: Int,
    # Deferred shard state (UnsafePointer to Array fields on SlowPathHandler)
    deferred_fds: UnsafePointer[Array[Int32, 16], MutUntrackedOrigin],
    deferred_seqs: UnsafePointer[Array[UInt64, 16], MutUntrackedOrigin],
    deferred_ks: UnsafePointer[Array[Int32, 16], MutUntrackedOrigin],
    deferred_n_shards: UnsafePointer[Array[Int32, 16], MutUntrackedOrigin],
    deferred_active: UnsafePointer[Array[UInt32, 16], MutUntrackedOrigin],
    deferred_done: UnsafePointer[Array[UInt32, 16], MutUntrackedOrigin],
    deferred_drain_ticks: UnsafePointer[Array[Int32, 16], MutUntrackedOrigin],
    deferred_count: UnsafePointer[Int, MutUntrackedOrigin],
    # I/O
    fd: Int32,
    mut writer: ResponseWriter,
    server: TCPServer,
    kq: Int32,
    consumed_bytes: Int,
) raises -> Int:
    """FT.SEARCH <index> <query> [PARAMS ...] [FILTER ...] [LIMIT ...] [RETURN ...] [NOCONTENT]
    [WITHSCORES] — vector/BM25/hybrid search. RETURN, NOCONTENT and WITHSCORES
    apply to the KNN form; the BM25 and HYBRID forms reply as without them.

    `num_tokens` is this command's token BOUND — the dispatch site passes
    `cmd_end_tok`, not the batch-wide token count. The optional-arg scan below
    advances over every unrecognized token, so a batch-wide bound would walk
    into the next pipelined command (swallowing it, and letting a second
    FT.SEARCH's PARAMS blob overwrite this query's blob).

    Return value: -1 = response already flushed, else a token index. The caller
    ignores it for token accounting and sets `i = cmd_end_tok - 1` itself."""
    var ci = i
    # V2.6: lazy-borrow shared index if local index not built yet (single-worker path only)
    var n_s = 1
    if is_not_null(shared_hnsw): n_s = shared_hnsw[].num_shards
    # Acquire load of ready_atomic pairs with publish_to_shared's RELEASE store
    # so all field writes (compact_buffer/num_nodes/entry_point_id/etc) are
    # visible before borrowing on weak memory order (ARM). Plain-Bool
    # `shared_hnsw[].ready` was the source of intermittent recall≈0 failures.
    var _ready_a: UInt64 = 0
    if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].ready_atomic):
        _ready_a = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
            shared_hnsw[].ready_atomic, UInt64(0))
    # Borrower-side invalidation and (re)borrow: a borrow of an index the
    # publisher dropped (ready_atomic=0) or replaced (publish_seq moved on) is
    # given back, and the published index is borrowed unless this worker's own
    # graph is it (refresh_index_view). `_live`: this worker's graph serves the
    # published index. When it does not, the KNN form answers exactly from the
    # ingest buffer (drop-in: a stock client never sends FT.OPTIMIZE, and the
    # build runs in the background, vector_auto.mojo).
    _ = n_s
    var _live = refresh_index_view(hnsw, shared_hnsw, worker_id, _ready_a)
    var _rows = shared_hnsw[].ingested() if is_not_null(shared_hnsw) else 0
    if ci + 2 < num_tokens:
        ci += 1  # move to index name token
        var idx_name_tok_ptr = tokens[ci].ptr
        var idx_name_tok_len = tokens[ci].length
        # gh #138: no index exists anywhere → error, not an empty array.
        # A client cannot otherwise distinguish "the store lost its index"
        # (process restarted, WAL replayed keys but no FT.CREATE) from
        # "the query genuinely matched nothing" — both used to be `*0`.
        # The index must be REGISTERED (FT.CREATE or FT.OPTIMIZE names it, in
        # the shared view, for every worker; FT.DROPINDEX removes it), as
        # FT.INFO checks. "Defined somewhere" used to be enough: after
        # FT.DROPINDEX, or an FT.CREATE that was refused, the dropped
        # config kept routing HSETs into the ingest buffer, FT.INFO said
        # "Unknown index name" and FT.SEARCH answered from those vectors.
        # Another index being served is gh #145's error, below.
        var _known: Bool
        if is_not_null(shared_hnsw):
            _known = _shared_index_name_is(shared_hnsw, idx_name_tok_ptr, idx_name_tok_len) \
                     or _unnamed_live_index(hnsw, shared_hnsw, _ready_a)
        else:
            _known = _local_index_name_is(hnsw, idx_name_tok_ptr, idx_name_tok_len) \
                     or (hnsw.index_name_len == 0 and hnsw.index_ready)
        var _other = is_not_null(shared_hnsw) and shared_hnsw[].index_name_len > 0
        if not _known and not _other:
            var _nomsg = String("ERR no such index '")
            # Cap at 64 — the same limit FT.CREATE stores, and it keeps an
            # attacker-supplied name from inflating the error response.
            var _nolen = idx_name_tok_len if idx_name_tok_len < 64 else 64
            for _ni in range(_nolen):
                _nomsg += chr(Int(idx_name_tok_ptr[_ni]))
            # ASCII only: RESP error strings get decoded by clients under
            # whatever codec they were configured with.
            _nomsg += "' - not created on this server (FT.CREATE first); if the server restarted, its index was not persisted"
            writer.append_error_response(_nomsg)
            writer.flush_response(fd, server, kq)
            return -1  # signal caller to return consumed_bytes
        # Phase 3.2: check for BM25 or HYBRID sub-mode BEFORE reading qstr
        var _mode_tok_len = tokens[ci + 1].length if ci + 1 < num_tokens else 0
        var _mode_tok_ptr = tokens[ci + 1].ptr   if ci + 1 < num_tokens else null_ptr[UInt8, MutUntrackedOrigin]()
        # BM25 check: b(98),m(109),2(50),5(53) len=4
        var _is_bm25 = _mode_tok_len == 4 and (_mode_tok_ptr[0]|0x20)==98 and (_mode_tok_ptr[1]|0x20)==109 and _mode_tok_ptr[2]==50 and _mode_tok_ptr[3]==53
        # HYBRID check: h(104),y(121),b(98),r(114),i(105),d(100) len=6
        var _is_hybrid = _mode_tok_len == 6 and (_mode_tok_ptr[0]|0x20)==104 and (_mode_tok_ptr[1]|0x20)==121 and (_mode_tok_ptr[2]|0x20)==98 and (_mode_tok_ptr[3]|0x20)==114 and (_mode_tok_ptr[4]|0x20)==105 and (_mode_tok_ptr[5]|0x20)==100
        if _is_bm25 and ci + 2 < num_tokens:
            # FT.SEARCH index BM25 "query text" [K k] [K1 <f>] [B <f>]
            var _bm25_text_tok = tokens[ci + 2]
            var _bm25_k = 10
            var _bm25_k1 = BM25_DEFAULT_K1
            var _bm25_b  = BM25_DEFAULT_B
            var _bm25_bad = String("")
            ci += 2
            # gh #139: K1/B are per-query so a caller can match the Okapi variant
            # its evaluation harness uses without restarting the server.
            var _oi = ci + 1
            while _oi + 1 < num_tokens:
                var _ol = tokens[_oi].length
                var _op = tokens[_oi].ptr
                var _vp = tokens[_oi + 1].ptr
                var _vl = tokens[_oi + 1].length
                if _ol == 1 and (_op[0]|0x20) == 107:      # K
                    var _kv = 0
                    for _ki in range(_vl):
                        var _kc = Int(_vp[_ki])
                        if _kc >= 48 and _kc <= 57: _kv = _kv * 10 + (_kc - 48)
                    if _kv > 0: _bm25_k = _kv
                elif _ol == 2 and (_op[0]|0x20) == 107 and _op[1] == 49:  # K1
                    var _k1v = parse_filter_float(_vp, _vl)
                    if _k1v < 0.0 or _k1v > 100.0: _bm25_bad = "K1 (want 0 <= K1 <= 100)"
                    else: _bm25_k1 = _k1v
                elif _ol == 1 and (_op[0]|0x20) == 98:     # B
                    var _bv = parse_filter_float(_vp, _vl)
                    if _bv < 0.0 or _bv > 1.0: _bm25_bad = "B (want 0 <= B <= 1)"
                    else: _bm25_b = _bv
                else:
                    break
                _oi += 2
            ci = _oi - 1
            if _bm25_bad.byte_length() > 0:
                writer.append_error_response("ERR BM25 parameter out of range: " + _bm25_bad)
                writer.flush_response(fd, server, kq)
                return -1
            # gh #138/#139: distinguish "no inverted index here" from "no match".
            # An unbuilt index used to answer `[0]` — identical to a genuine miss —
            # which is how an FT.ADDTEXT-only corpus looked like a retrieval failure.
            if not hnsw.bm25_is_built:
                var _nbmsg = String("ERR BM25 index for '")
                var _nblen = idx_name_tok_len if idx_name_tok_len < 64 else 64
                for _ni in range(_nblen):
                    _nbmsg += chr(Int(idx_name_tok_ptr[_ni]))
                _nbmsg += ("' is not built - the schema needs a TEXT field, docs need that field set "
                           "(HSET <integer-id> <field> <text>, or FT.ADDTEXT <index> <integer-id> <text>), "
                           "and FT.OPTIMIZE must run after ingest")
                writer.append_error_response(_nbmsg)
                writer.flush_response(fd, server, kq)
                return -1
            # gh #145: postings exist, but for a different index.
            if bm25_owner_mismatch(hnsw, idx_name_tok_ptr, idx_name_tok_len):
                writer.append_error_response(
                    bm25_owner_error(hnsw, idx_name_tok_ptr, idx_name_tok_len))
                writer.flush_response(fd, server, kq)
                return -1
            var _bm25_scores = List[Float32]()
            # #46: with dead documents, look further and drop them
            var _bm25_fetch = _bm25_k
            if is_not_null(shared_hnsw) and shared_hnsw[].any_dead():
                _bm25_fetch = _bm25_k * 4 if _bm25_k * 4 > 64 else 64
            var _bm25_ids = search_bm25(hnsw, _bm25_text_tok.ptr, _bm25_text_tok.length,
                                        _bm25_fetch, _bm25_scores, _bm25_k1, _bm25_b)
            drop_dead(shared_hnsw, _bm25_ids, _bm25_scores, _bm25_k)
            write_ft_search_response(writer, _bm25_ids, _bm25_scores, keyspace, shared_hnsw)
            writer.flush_response(fd, server, kq)
            return -1  # signal caller to return consumed_bytes
        elif _is_hybrid and ci + 1 < num_tokens:
            # FT.SEARCH index HYBRID VECTOR_QUERY <blob> TEXT_QUERY "text" [K k]
            ci += 1  # skip HYBRID keyword
            var _hvec_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
            var _hvec_len = 0
            var _htext_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
            var _htext_len = 0
            var _hk = 10
            var _ji = ci + 1
            while _ji < num_tokens:
                var _jtok = tokens[_ji]
                var _jl = _jtok.length; var _jp = _jtok.ptr
                # VECTOR_QUERY (12 bytes: v,e,c,t,o,r,_,q,u,e,r,y)
                if _jl == 12 and (_jp[0]|0x20)==118 and (_jp[6])==95 and (_jp[7]|0x20)==113:
                    if _ji + 1 < num_tokens:
                        _hvec_ptr = tokens[_ji + 1].ptr
                        _hvec_len = tokens[_ji + 1].length
                        _ji += 1
                # TEXT_QUERY (10 bytes: t,e,x,t,_,q,u,e,r,y)
                elif _jl == 10 and (_jp[0]|0x20)==116 and (_jp[4])==95 and (_jp[5]|0x20)==113:
                    if _ji + 1 < num_tokens:
                        _htext_ptr = tokens[_ji + 1].ptr
                        _htext_len = tokens[_ji + 1].length
                        _ji += 1
                # K keyword
                elif _jl == 1 and (_jp[0]|0x20)==107:
                    if _ji + 1 < num_tokens:
                        var _kv2 = 0
                        for _ki2 in range(tokens[_ji + 1].length):
                            var _kc2 = Int(tokens[_ji + 1].ptr[_ki2])
                            if _kc2 >= 48 and _kc2 <= 57: _kv2 = _kv2 * 10 + (_kc2 - 48)
                        if _kv2 > 0: _hk = _kv2
                        _ji += 1
                _ji += 1
            _ = _ji - 1
            # Run vector search (if blob provided)
            var _hvec_ids    = List[Int]()
            var _hvec_scores = List[Float32]()
            if _live and _hvec_len == hnsw.dim * 4:
                var _st_ef = hnsw.ef_runtime
                if hnsw.num_nodes > 50000:
                    var _str = Float64(hnsw.num_nodes) / 50000.0
                    var _sef = Int(150.0 * sqrt(sqrt(_str)))
                    if _sef > _st_ef: _st_ef = _sef
                try:
                    var _hfp32 = _hvec_ptr.bitcast[Float32]()
                    _hvec_ids = hnsw.search_fp32_scored(_hfp32, _hk * 4, _hvec_scores, _st_ef)
                    # drop-in: rows written after an automatic build (RRF reads
                    # ranks, so both sides go to metric units before merging)
                    if _rows > hnsw.num_nodes and is_not_null(shared_hnsw[].ingest_fp32):
                        _rescore_knn(hnsw, _hfp32, _hvec_ids, _hvec_scores)
                        merge_delta(shared_hnsw, hnsw.num_nodes, _hfp32, _hk * 4, FtFilters(),
                                    keyspace, _hvec_ids, _hvec_scores)
                except: pass
            elif not _live and _rows > 0 and _hvec_len == shared_hnsw[].pre_dim * 4:
                # drop-in: no built index serves here yet; the vector leg is exact
                exact_knn(shared_hnsw, _hvec_ptr.bitcast[Float32](), _hk * 4, 0, _rows,
                          _hvec_ids, _hvec_scores, FtFilters(), keyspace)
            # Run BM25 search (if text provided)
            var _hbm25_scores = List[Float32]()
            var _hbm25_ids    = List[Int]()
            if _htext_len > 0:
                _hbm25_ids = search_bm25(hnsw, _htext_ptr, _htext_len, _hk * 4, _hbm25_scores)
            drop_dead(shared_hnsw, _hvec_ids, _hvec_scores, len(_hvec_ids))      # #46
            drop_dead(shared_hnsw, _hbm25_ids, _hbm25_scores, len(_hbm25_ids))
            # RRF fusion (k=60)
            var _rrf_ids    = List[Int]()
            var _rrf_scores = List[Float32]()
            if len(_hvec_ids) == 0 and len(_hbm25_ids) == 0:
                write_ft_search_response(writer, _rrf_ids, _rrf_scores, keyspace, shared_hnsw)
                writer.flush_response(fd, server, kq)
                return -1  # signal caller to return consumed_bytes
            # Build combined candidate set (union of both result lists)
            var _all_ids = List[Int]()
            for _vi in range(len(_hvec_ids)): _all_ids.append(_hvec_ids[_vi])
            for _bi2 in range(len(_hbm25_ids)):
                var _bid = _hbm25_ids[_bi2]
                var _dup = False
                for _vi2 in range(len(_hvec_ids)):
                    if _hvec_ids[_vi2] == _bid: _dup = True; break
                if not _dup: _all_ids.append(_bid)
            var _n_cands = len(_all_ids)
            var _rrf_tmp = List[Float32](capacity=_n_cands)
            for _ci in range(_n_cands):
                var _cid = _all_ids[_ci]
                var _rrf_sc: Float32 = 0.0
                # Vector rank
                for _vr in range(len(_hvec_ids)):
                    if _hvec_ids[_vr] == _cid: _rrf_sc += 1.0 / (60.0 + Float32(_vr)); break
                # BM25 rank
                for _br in range(len(_hbm25_ids)):
                    if _hbm25_ids[_br] == _cid: _rrf_sc += 1.0 / (60.0 + Float32(_br)); break
                _rrf_tmp.append(_rrf_sc)
            # Top-k by RRF score
            var _actual_k = _hk if _hk < _n_cands else _n_cands
            for _ii2 in range(_actual_k):
                var _mx = _ii2; var _ms = _rrf_tmp[_ii2]
                for _jj2 in range(_ii2 + 1, _n_cands):
                    if _rrf_tmp[_jj2] > _ms: _ms = _rrf_tmp[_jj2]; _mx = _jj2
                if _mx != _ii2:
                    var _ti2 = _all_ids[_ii2]; _all_ids[_ii2] = _all_ids[_mx]; _all_ids[_mx] = _ti2
                    var _ts2 = _rrf_tmp[_ii2]; _rrf_tmp[_ii2] = _rrf_tmp[_mx]; _rrf_tmp[_mx] = _ts2
                _rrf_ids.append(_all_ids[_ii2])
                _rrf_scores.append(_rrf_tmp[_ii2])
            write_ft_search_response(writer, _rrf_ids, _rrf_scores, keyspace, shared_hnsw)
            writer.flush_response(fd, server, kq)
            return -1  # signal caller to return consumed_bytes
        # gh #361: the query must be `<prefilter>=>[KNN k @field $param …]`.
        # Anything else used to fall through to an empty `[]` — not even a
        # valid FT.SEARCH reply shape — which a client cannot tell from "no
        # neighbours"; that is how every pion-mcp vector tool shipped returning
        # nothing. The query token is read in place (no String copy).
        var q_tok = tokens[ci + 1]
        ci += 1  # move to query string token
        var qp = q_tok.ptr
        var ql = q_tok.length
        var filters = FtFilters()
        var q_err = String("")
        var k = 10
        var opts = FtReplyOpts()
        opts.tokens = tokens
        opts.mode = FT_REPLY_ALL     # no RETURN: every field, as RediSearch sends
        var ef_query = hnsw.ef_runtime
        var ef_param_p = null_ptr[UInt8, MutUntrackedOrigin]()
        var ef_param_l = 0
        var vec_param_p = null_ptr[UInt8, MutUntrackedOrigin]()
        var vec_param_l = 0
        var arrow = -1
        for qi in range(ql - 1):
            if qp[qi] == 61 and qp[qi + 1] == 62:   # "=>"
                arrow = qi; break
        if arrow < 0:
            q_err = ("ERR unsupported FT.SEARCH query '" + _bytes_str(qp, ql)
                     + "' - Pion answers '<filter>=>[KNN k @field $param]' with PARAMS, "
                     + "'BM25 <text>' and 'HYBRID VECTOR_QUERY ... TEXT_QUERY ...'")
        elif not filters.parse_expr(qp, arrow, shared_hnsw):
            q_err = filters.error
        else:
            # [KNN <k> @<field> $<param> [EF_RUNTIME <n>|$<p>] [AS <alias>]]
            var p = arrow + 2
            while p < ql and qp[p] == 32: p += 1
            var ok = p < ql and qp[p] == 91          # '['
            p += 1
            while p < ql and qp[p] == 32: p += 1
            ok = ok and p + 3 < ql and (qp[p] | 0x20) == 107 and (qp[p + 1] | 0x20) == 110 \
                 and (qp[p + 2] | 0x20) == 110 and qp[p + 3] == 32
            p += 4
            while p < ql and qp[p] == 32: p += 1
            var kv = 0
            var kd = 0
            while ok and p < ql and qp[p] >= 48 and qp[p] <= 57 and kd < 7:
                kv = kv * 10 + Int(qp[p] - 48); p += 1; kd += 1
            ok = ok and kd > 0          # KNN 0 is valid: an empty result, as in RediSearch
            k = kv
            while p < ql and qp[p] == 32: p += 1
            ok = ok and p < ql and qp[p] == 64       # '@'
            var f0 = p + 1
            p = f0
            while p < ql and qp[p] != 32 and qp[p] != 93: p += 1
            var f_len = p - f0
            opts.field_p = qp + f0
            opts.field_l = f_len
            while p < ql and qp[p] == 32: p += 1
            ok = ok and p < ql and qp[p] == 36       # '$'
            var v0 = p + 1
            p = v0
            while p < ql and qp[p] != 32 and qp[p] != 93: p += 1
            vec_param_p = qp + v0
            vec_param_l = p - v0
            ok = ok and vec_param_l > 0
            # optional clauses up to ']'
            while ok and p < ql and qp[p] != 93:
                if qp[p] == 32:
                    p += 1; continue
                var w0 = p
                while p < ql and qp[p] != 32 and qp[p] != 93: p += 1
                var w_len = p - w0
                while p < ql and qp[p] == 32: p += 1
                var a0 = p
                while p < ql and qp[p] != 32 and qp[p] != 93: p += 1
                var a_len = p - a0
                if a_len == 0:
                    ok = False
                elif w_len == 10 and arg_eq(qp + w0, 10, "ef_runtime"):
                    if qp[a0] == 36:
                        ef_param_p = qp + a0 + 1; ef_param_l = a_len - 1
                    else:
                        var ev = 0
                        for e in range(a_len):
                            var ec = Int(qp[a0 + e])
                            if ec < 48 or ec > 57: ok = False
                            else: ev = ev * 10 + (ec - 48)
                        if ok and ev > 0: ef_query = ev
                elif w_len == 2 and arg_eq(qp + w0, 2, "as"):
                    # The score's field name (else __<field>_score)
                    opts.alias_p = qp + a0
                    opts.alias_l = a_len
                else:
                    ok = False
            ok = ok and p < ql and qp[p] == 93
            if not ok:
                q_err = ("ERR unsupported KNN clause in '" + _bytes_str(qp, ql)
                         + "' - expected [KNN <k> @<field> $<param> [EF_RUNTIME <n>] [AS <alias>]]")
            else:
                # The field must be the index's vector field. Same fold as
                # FT.CREATE stores it (`| 0x20`). The SHARED copy, not this
                # worker's: under -w N only the worker that ran FT.CREATE has
                # the name in its own HNSWGraph; borrowers keep the default.
                var vf_len = hnsw.vector_field_len
                var vf_name = hnsw.vector_field_name.copy()
                if is_not_null(shared_hnsw) and shared_hnsw[].pre_vector_field_len > 0:
                    vf_len = shared_hnsw[].pre_vector_field_len
                    vf_name = shared_hnsw[].pre_vector_field_name.copy()
                var f_ok = f_len == vf_len
                if f_ok:
                    for fb in range(f_len):
                        if (qp[f0 + fb] | 0x20) != vf_name[fb]:
                            f_ok = False; break
                if not f_ok:
                    var _vfn = String("")
                    for fb in range(vf_len): _vfn += chr(Int(vf_name[fb]))
                    q_err = ("ERR KNN field '@" + _bytes_str(qp + f0, f_len)
                             + "' is not this index's vector field '@" + _vfn + "'")
        # Auto-scale ef with dataset size: ef = max(ef, 150 * (N/50K)^0.25)
        # At 50K: ef=150 (baseline). At 5M: ef≈475. Prevents recall regression at scale.
        if _live and hnsw.num_nodes > 50000:
            var scale_ratio = Float64(hnsw.num_nodes) / 50000.0
            # (N/50K)^0.25 = sqrt(sqrt(N/50K))
            var scaled_ef = Int(150.0 * sqrt(sqrt(scale_ratio)))
            if scaled_ef > ef_query:
                ef_query = scaled_ef
        # The remaining tokens, every one of them an option: PARAMS, FILTER,
        # LIMIT, RETURN, NOCONTENT, WITHSCORES, SORTBY, and the options that
        # change nothing a KNN query answers (DIALECT, TIMEOUT, and the text
        # query's: VERBATIM, NOSTOPWORDS, INORDER, SLOP, LANGUAGE, SCORER,
        # EXPANDER, PAYLOAD, INFIELDS, SUMMARIZE, HIGHLIGHT). Anything else is
        # refused. The loop used to step over a token it did not know, so
        # INKEYS, GEOFILTER, WITHPAYLOADS, a typo... answered as if absent.
        var blob_ptr = null_ptr[Float32, MutUntrackedOrigin]()
        var blob_set = False
        var params_at = -1
        var params_n = 0
        var limit_off = 0
        # Without LIMIT, every hit: KNN k returns k rows, as Pion always did.
        # RediSearch pages 10 by default (its clients send LIMIT).
        var limit_cnt = -1
        var sort = FtSort()
        var j2 = ci + 1
        while j2 < num_tokens and not filters.failed() and q_err.byte_length() == 0:
            var jl2 = tokens[j2].length; var jp2 = tokens[j2].ptr
            if jl2 == 6 and arg_eq(jp2, 6, "filter"):
                # gh #367: every documented form parses, anything else refuses.
                var used = filters.parse_filter_arg(tokens, j2, num_tokens, shared_hnsw)
                if used < 0: break
                j2 += used
            elif jl2 == 6 and arg_eq(jp2, 6, "params"):
                if j2 + 1 < num_tokens:
                    var cnt2 = 0
                    var cp2 = tokens[j2 + 1].ptr
                    var cnt_ok = tokens[j2 + 1].length > 0
                    for ci2 in range(tokens[j2 + 1].length):
                        var cc = Int(cp2[ci2])
                        if cc < 48 or cc > 57: cnt_ok = False
                        else: cnt2 = cnt2 * 10 + (cc - 48)
                    if not cnt_ok or cnt2 % 2 != 0 or j2 + 1 + cnt2 >= num_tokens:
                        q_err = "ERR PARAMS needs an even count followed by that many name/value arguments"
                        break
                    params_at = j2 + 2
                    params_n = cnt2 // 2
                    j2 += 1 + cnt2
                else:
                    q_err = "ERR PARAMS needs an even count followed by that many name/value arguments"
                    break
            elif jl2 == 5 and arg_eq(jp2, 5, "limit"):
                # LIMIT <offset> <num>: two integers >= 0 (RediSearch's check
                # and words). It used to read <num> digit by digit, skipping
                # anything else, and ignore <offset>.
                var lo_ok = j2 + 2 < num_tokens
                var lo_v = ParsedInt(0, False)
                var ln_v = ParsedInt(0, False)
                if lo_ok:
                    lo_v = parse_int64_strict(tokens[j2 + 1].ptr, tokens[j2 + 1].length)
                    ln_v = parse_int64_strict(tokens[j2 + 2].ptr, tokens[j2 + 2].length)
                if not lo_ok or not lo_v.ok or not ln_v.ok or lo_v.value < 0 or ln_v.value < 0 \
                        or lo_v.value > 1000000000 or ln_v.value > 1000000000:
                    q_err = "LIMIT needs two numeric arguments"
                    break
                limit_off = Int(lo_v.value)
                limit_cnt = Int(ln_v.value)
                j2 += 2
            elif jl2 == 6 and arg_eq(jp2, 6, "return"):
                # RETURN <n> <field> [AS <name>] ...: n counts every token after
                # it, AS and the name included. RETURN 0 is NOCONTENT. The
                # errors are RediSearch's, word for word.
                var rn = 0
                var rn_ok = j2 + 1 < num_tokens and tokens[j2 + 1].length > 0
                if rn_ok:
                    for rdi in range(tokens[j2 + 1].length):
                        var rdc = Int(tokens[j2 + 1].ptr[rdi])
                        if rdc < 48 or rdc > 57: rn_ok = False
                        else: rn = rn * 10 + (rdc - 48)
                if not rn_ok:
                    q_err = "Bad arguments for RETURN: Could not convert argument to expected type"
                    break
                if j2 + 1 + rn >= num_tokens:
                    q_err = "Bad arguments for RETURN: Expected an argument, but none provided"
                    break
                if rn == 0:
                    opts.mode = FT_REPLY_NOCONTENT
                elif opts.mode != FT_REPLY_NOCONTENT:
                    opts.mode = FT_REPLY_RETURN
                    opts.ret_at = j2 + 2
                    opts.ret_n = rn
                j2 += 1 + rn
            elif jl2 == 9 and arg_eq(jp2, 9, "nocontent"):
                opts.mode = FT_REPLY_NOCONTENT
            elif jl2 == 10 and arg_eq(jp2, 10, "withscores"):
                opts.with_scores = True
            elif jl2 == 6 and arg_eq(jp2, 6, "sortby"):
                # SORTBY <field> [ASC|DESC]: applied below, before LIMIT.
                if j2 + 1 >= num_tokens:
                    q_err = "SORTBY needs a field name"
                    break
                sort.p = tokens[j2 + 1].ptr
                sort.l = tokens[j2 + 1].length
                sort.set = True
                j2 += 1
                if j2 + 1 < num_tokens:
                    var sl = tokens[j2 + 1].length
                    if sl == 3 and arg_eq(tokens[j2 + 1].ptr, 3, "asc"):
                        j2 += 1
                    elif sl == 4 and arg_eq(tokens[j2 + 1].ptr, 4, "desc"):
                        sort.desc = True
                        j2 += 1
            elif arg_eq(jp2, jl2, "dialect") or arg_eq(jp2, jl2, "timeout") or arg_eq(jp2, jl2, "slop") \
                    or arg_eq(jp2, jl2, "language") or arg_eq(jp2, jl2, "scorer") \
                    or arg_eq(jp2, jl2, "expander") or arg_eq(jp2, jl2, "payload"):
                # one argument; none of them changes a KNN answer
                if j2 + 1 >= num_tokens:
                    q_err = _bytes_str(jp2, jl2) + " needs an argument"
                    break
                j2 += 1
            elif arg_eq(jp2, jl2, "verbatim") or arg_eq(jp2, jl2, "nostopwords") or arg_eq(jp2, jl2, "inorder"):
                pass                         # text-query flags, no argument
            elif arg_eq(jp2, jl2, "infields"):
                # INFIELDS <n> <field>...: which TEXT fields a text term searches
                var nf = ParsedInt(0, False)
                if j2 + 1 < num_tokens:
                    nf = parse_int64_strict(tokens[j2 + 1].ptr, tokens[j2 + 1].length)
                if not nf.ok or nf.value < 0 or j2 + 1 + Int(nf.value) >= num_tokens:
                    q_err = "Bad arguments for INFIELDS: Could not convert argument to expected type"
                    break
                j2 += 1 + Int(nf.value)
            elif arg_eq(jp2, jl2, "summarize") or arg_eq(jp2, jl2, "highlight"):
                # SUMMARIZE [FIELDS n f...] [FRAGS n] [LEN n] [SEPARATOR s] /
                # HIGHLIGHT [FIELDS n f...] [TAGS open close]: they rewrite the
                # TEXT a text term matched, and a KNN query matches no term.
                var sub_ok = True
                while j2 + 1 < num_tokens:
                    var sp = tokens[j2 + 1].ptr
                    var sl2 = tokens[j2 + 1].length
                    if arg_eq(sp, sl2, "fields"):
                        var nfs = ParsedInt(0, False)
                        if j2 + 2 < num_tokens:
                            nfs = parse_int64_strict(tokens[j2 + 2].ptr, tokens[j2 + 2].length)
                        if not nfs.ok or nfs.value < 0 or j2 + 2 + Int(nfs.value) >= num_tokens:
                            sub_ok = False
                            break
                        j2 += 2 + Int(nfs.value)
                    elif arg_eq(sp, sl2, "frags") or arg_eq(sp, sl2, "len") or arg_eq(sp, sl2, "separator"):
                        if j2 + 2 >= num_tokens:
                            sub_ok = False
                            break
                        j2 += 2
                    elif arg_eq(sp, sl2, "tags"):
                        if j2 + 3 >= num_tokens:
                            sub_ok = False
                            break
                        j2 += 3
                    else:
                        break
                if not sub_ok:
                    q_err = "Bad arguments for " + _bytes_str(jp2, jl2)
                    break
            elif arg_eq(jp2, jl2, "inkeys") or arg_eq(jp2, jl2, "geofilter") \
                    or arg_eq(jp2, jl2, "withpayloads") or arg_eq(jp2, jl2, "withsortkeys") \
                    or arg_eq(jp2, jl2, "explainscore"):
                q_err = "ERR FT.SEARCH " + _bytes_str(jp2, jl2) + " is not supported"
                break
            else:
                q_err = "Unknown argument `" + _bytes_str(jp2, jl2) + "` at position " \
                    + String(j2 - i - 2) + " for <main>"
                break
            j2 += 1
        ci = j2 - 1
        if filters.failed() and q_err.byte_length() == 0:
            q_err = filters.error
        # Resolve $param references against PARAMS by NAME. The old scan picked
        # whichever value happened to be dim*4 bytes long.
        if q_err.byte_length() == 0 and vec_param_l > 0:
            var found = False
            for pi in range(params_n):
                var nt = tokens[params_at + 2 * pi]
                if nt.length == vec_param_l:
                    var same = True
                    for b in range(vec_param_l):
                        if nt.ptr[b] != vec_param_p[b]: same = False; break
                    if same:
                        found = True
                        var vt = tokens[params_at + 2 * pi + 1]
                        # the served graph's DIM; before a build, the defined index's
                        var _qdim = hnsw.dim
                        if not _live and is_not_null(shared_hnsw) and shared_hnsw[].pre_dim > 0:
                            _qdim = shared_hnsw[].pre_dim
                        if vt.length == _qdim * 4:
                            blob_ptr = vt.ptr.bitcast[Float32]()
                            blob_set = True
                        else:
                            q_err = ("ERR query vector $" + _bytes_str(vec_param_p, vec_param_l) + " is "
                                     + String(vt.length) + " bytes; this index expects "
                                     + String(_qdim * 4) + " (DIM " + String(_qdim) + " x FLOAT32)")
                        break
            if not found and q_err.byte_length() == 0:
                q_err = "ERR query parameter $" + _bytes_str(vec_param_p, vec_param_l) + " is not in PARAMS"
        if q_err.byte_length() == 0 and ef_param_l > 0:
            var ef_found = False
            for pi in range(params_n):
                var nt = tokens[params_at + 2 * pi]
                if nt.length == ef_param_l:
                    var same = True
                    for b in range(ef_param_l):
                        if nt.ptr[b] != ef_param_p[b]: same = False; break
                    if same:
                        var vt = tokens[params_at + 2 * pi + 1]
                        var ev = 0
                        for e in range(vt.length):
                            var ec = Int(vt.ptr[e])
                            if ec >= 48 and ec <= 57: ev = ev * 10 + (ec - 48)
                        if ev > 0: ef_query = ev
                        ef_found = True
                        break
            if not ef_found:
                q_err = "ERR query parameter $" + _bytes_str(ef_param_p, ef_param_l) + " is not in PARAMS"
        # Check index name if we have one stored
        var idx_ok = True
        if hnsw.index_name_len > 0:
            if idx_name_tok_len != hnsw.index_name_len: idx_ok = False
            else:
                for ni in range(idx_name_tok_len):
                    if idx_name_tok_ptr[ni] != hnsw.index_name[ni]: idx_ok = False; break
        elif is_not_null(shared_hnsw) and shared_hnsw[].index_name_len > 0:
            # gh #403: a worker that has not borrowed yet has no local name, and
            # used to accept ANY name here — the same query answered `[0]` on
            # one connection and a name error on another. Check the registry.
            idx_ok = _shared_index_name_is(shared_hnsw, idx_name_tok_ptr, idx_name_tok_len)
        # Ensure ef >= k: ef < k silently returns fewer results → low recall
        if ef_query < k: ef_query = k
        # gh #145: a name mismatch means this server's single index was rebuilt
        # for someone else. That used to fall into the empty-array branch below,
        # so a cross-index clobber looked exactly like "your query matched
        # nothing" — say which index actually holds the graph instead.
        if not idx_ok:
            var _wmsg = String("ERR index '")
            var _wlen = idx_name_tok_len if idx_name_tok_len < 64 else 64
            for _wi in range(_wlen): _wmsg += chr(Int(idx_name_tok_ptr[_wi]))
            _wmsg += "' is not the index loaded on this server - it holds one index at a time and currently serves '"
            if hnsw.index_name_len > 0:
                for _wi in range(hnsw.index_name_len): _wmsg += chr(Int(hnsw.index_name[_wi]))
            else:
                for _wi in range(shared_hnsw[].index_name_len): _wmsg += chr(Int(shared_hnsw[].index_name[_wi]))
            _wmsg += "' (a later FT.CREATE/FT.OPTIMIZE replaced it)"
            writer.append_error_response(_wmsg)
        elif q_err.byte_length() > 0:
            writer.append_error_response(q_err)   # gh #361 / #367
        elif k == 0 or not blob_set or (not _live and _rows == 0):
            # KNN 0, or a defined index with no vectors yet: a well-formed empty
            # reply, `[0]` — `[]` is not an FT.SEARCH reply shape (gh #361).
            var _e_ids = List[Int]()
            var _e_scores = List[Float32]()
            _ft_knn_reply(writer, _e_ids, _e_scores, keyspace, shared_hnsw, opts, sort, limit_off, limit_cnt)
        elif not _live:
            # Drop-in: vectors are in, no built index serves them (none built
            # yet, or this worker's owner is rebuilding it). Exact, in metric
            # units, dead slots and FILTER honoured; EF_RUNTIME does not apply.
            var _x_ids = List[Int]()
            var _x_scores = List[Float32]()
            exact_knn(shared_hnsw, blob_ptr, k, 0, _rows, _x_ids, _x_scores, filters, keyspace)
            _ft_knn_reply(writer, _x_ids, _x_scores, keyspace, shared_hnsw, opts, sort, limit_off, limit_cnt)
        else:
            var n_shards2 = 1
            if is_not_null(shared_hnsw): n_shards2 = shared_hnsw[].num_shards
            # Bug C fix: shard bus has one slot per (coordinator, shard) pair.
            # Multiple concurrent deferred queries from the same coordinator
            # overwrite each other's slots → lost results. Serialize: only
            # one deferred sharded query at a time; concurrent queries fall
            # through to the single-shard CPU path (lower recall but correct).
            # A deferred reply is written later by drain_deferred_shard_responses
            # with the default fields, which no KNN query asks for: it keeps no
            # FtReplyOpts, so every KNN query takes the synchronous path. Carry
            # the options through deferred_* before re-enabling sharding.
            # (Sharding is off: num_shards is 1.)
            var use_sharding = (n_shards2 > 1 and hnsw.index_ready and filters.count == 0
                                and is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].shard_bus)
                                and deferred_count[] == 0
                                and opts.mode == FT_REPLY_DEFAULT and not opts.with_scores)

            if use_sharding:
                var shard_bus = shared_hnsw[].shard_bus
                var query_seq = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                    shard_query_seq, UInt64(1)
                )

                # 1. Search own shard
                scratch_dists.clear()
                var own_ids = hnsw.search_fp32_scored(blob_ptr, MAX_SHARD_K, scratch_dists, ef_query)
                shard_bus[].write_result(worker_id, worker_id, own_ids, scratch_dists, query_seq)

                # 2. Post query only to shards that are confirmed ready.
                # Ghost workers (didn't start due to CPU limit) never set shard_ready
                # and would cause the coordinator to spin-wait forever.
                var active_shards = Array[Bool, 32](uninitialized=True)
                for si in range(32):
                    active_shards[si] = False
                active_shards[worker_id] = True
                var active_count = 1
                for other in range(n_shards2):
                    if other == worker_id: continue
                    var is_ready = (is_not_null(shared_hnsw[].shard_ready) and
                        Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared_hnsw[].shard_ready + other * 8, UInt64(0)) > 0)
                    if is_ready:
                        active_shards[other] = True
                        active_count += 1
                        shard_bus[].post_query(worker_id, other, blob_ptr, MAX_SHARD_K, ef_query, query_seq, hnsw.dim)

                # 3. T3.4: async deferred response — store state, return without flush.
                # drain_deferred_shard_responses() will poll each engine iteration
                # and send the response when all shard results arrive.
                var active_bitmask = UInt32(0)
                for si in range(n_shards2):
                    if active_shards[si]:
                        active_bitmask |= (UInt32(1) << UInt32(si))
                var done_bitmask = UInt32(1) << UInt32(worker_id)  # own shard already done
                if deferred_count[] < 16:
                    var di = deferred_count[]
                    deferred_fds[][di]        = fd
                    deferred_seqs[][di]       = query_seq
                    deferred_ks[][di]         = Int32(k)
                    deferred_n_shards[][di]   = Int32(n_shards2)
                    deferred_active[][di]     = active_bitmask
                    deferred_done[][di]       = done_bitmask
                    deferred_drain_ticks[][di] = Int32(0)
                    deferred_count[] += 1
                    # Flush any responses from commands processed before this FT.SEARCH
                    writer.flush_response(fd, server, kq)
                    return -1  # signal caller to return consumed_bytes (FT.SEARCH response deferred)
                # Deferred table full — fall back to synchronous spin-wait
                var done_arr = Array[Bool, 32](uninitialized=True)
                for si in range(32):
                    done_arr[si] = (not active_shards[si])
                done_arr[worker_id] = True
                var pending = active_count - 1
                var max_spin = 2000000
                var spin = 0
                while pending > 0 and spin < max_spin:
                    for c in range(n_shards2):
                        if c == worker_id: continue
                        var qidx = c * n_shards2 + worker_id
                        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shard_bus[].query_ready + qidx, UInt64(0)) == 0: continue
                        var slot = shard_bus[].query_slots[qidx]
                        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shard_bus[].query_ready + qidx, UInt64(0))
                        scratch_dists.clear()
                        try:
                            var sids = hnsw.search_fp32_scored(slot.query_fp32, Int(slot.k), scratch_dists, Int(slot.ef))
                            shard_bus[].write_result(c, worker_id, sids, scratch_dists, slot.seq)
                        except:
                            shard_bus[].result_counts[c * n_shards2 + worker_id] = 0
                            shard_bus[].result_seq[c * n_shards2 + worker_id] = slot.seq
                            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shard_bus[].result_ready + c * n_shards2 + worker_id, UInt64(1))
                    for other2 in range(n_shards2):
                        if done_arr[other2]: continue
                        var ridx = worker_id * n_shards2 + other2
                        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shard_bus[].result_ready + ridx, UInt64(0)) != 0:
                            if shard_bus[].result_seq[ridx] == query_seq:
                                done_arr[other2] = True
                                pending -= 1
                    spin += 1

                # 4. Fast linear merge of results from all shards
                var all_res_ids = List[Int](capacity=n_shards2 * MAX_SHARD_K)
                var all_res_scores = List[Float32](capacity=n_shards2 * MAX_SHARD_K)

                for shard in range(n_shards2):
                    var idx2 = worker_id * n_shards2 + shard
                    if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shard_bus[].result_ready + idx2, UInt64(0)) == 0: continue
                    if shard_bus[].result_seq[idx2] != query_seq: continue

                    var cnt = Int(shard_bus[].result_counts[idx2])
                    if cnt < 0 or cnt > MAX_SHARD_K: cnt = 0
                    var base_ids = shard_bus[].result_ids + idx2 * MAX_SHARD_K
                    var base_scores = shard_bus[].result_scores + idx2 * MAX_SHARD_K

                    for r in range(cnt):
                        all_res_ids.append(Int(base_ids[r]))
                        all_res_scores.append(base_scores[r])

                # 5. Extract top-k via simple selection sort
                var n_all = len(all_res_ids)
                var final_k = k if k < n_all else n_all

                for fi in range(final_k):
                    var min_idx = fi
                    var min_score = all_res_scores[fi]
                    for fj in range(fi + 1, n_all):
                        if all_res_scores[fj] < min_score:
                            min_score = all_res_scores[fj]
                            min_idx = fj
                    if min_idx != fi:
                        var tmp_id = all_res_ids[fi]
                        all_res_ids[fi] = all_res_ids[min_idx]
                        all_res_ids[min_idx] = tmp_id
                        var tmp_score = all_res_scores[fi]
                        all_res_scores[fi] = min_score
                        all_res_scores[min_idx] = tmp_score

                var merged_results = List[Int]()
                var merged_scores = List[Float32]()
                for ii in range(n_all):
                    merged_results.append(all_res_ids[ii])
                    merged_scores.append(all_res_scores[ii])
                drop_dead(shared_hnsw, merged_results, merged_scores, final_k)   # #46

                write_ft_search_response(writer, merged_results, merged_scores, keyspace, shared_hnsw, opts)
            else:
                # gh #87.1: IVF-PQ search branch removed.
                # §7: Adaptive GPU/CPU routing — GPU brute-force when idle,
                # CPU HNSW when GPU busy (LLM/MAX/high load).
                # pion_metal_should_use_gpu() checks rolling dispatch latency.
                var use_gpu_search = False
                # The §7v2 GPU full-scan kernel reads compact_buffer as INT8; with
                # PolarQuant (INT4) / TurboQuant (INT3+QJL) / NanoQuant (INT2) active
                # those bytes are sub-byte-packed → garbage distances. Fall through
                # to the CPU quant search path; the FP32 GPU rerank still fires from
                # _try_gpu_rerank() inside hnsw.search.
                # #46: dead slots are filtered like a metadata filter (below),
                # through the HNSW path that widens its candidates
                # An expired document is gone too (_slot_expired): when this
                # worker has TTLs, widen like for dead slots, so k live remain.
                var _ttls = _ttl_any(keyspace)
                var _dead = (is_not_null(shared_hnsw) and shared_hnsw[].any_dead()) or _ttls
                if hnsw.has_gpu and is_not_null(hnsw.compact_buffer) and filters.count == 0 and not _dead \
                   and not (hnsw.polarquant or hnsw.turboquant or hnsw.nanoquant):
                    comptime if CompilationTarget.is_macos():
                        use_gpu_search = external_call["pion_metal_should_use_gpu", Int32]() == 1
                scratch_dists.clear()
                var results2 = List[Int]()
                if filters.count > 0 or _dead:
                    # gh #367: post-filter a widening candidate set until k pass
                    # or the whole index has been searched. The old path took
                    # k*2 candidates once, so a selective filter returned a
                    # handful of rows (or none) while matching documents existed.
                    # Each candidate's fields are read from its hash
                    # (passes_filters), so a field changed after the build counts.
                    var filtered_ids = List[Int]()
                    var filtered_scores = List[Float32]()
                    var n_total = hnsw.num_nodes
                    var sk = k * 4 if k * 4 > 64 else 64
                    if sk > n_total: sk = n_total
                    while True:
                        filtered_ids.clear()
                        filtered_scores.clear()
                        scratch_dists.clear()
                        var ef_f = ef_query if ef_query > sk else sk
                        if p3_batch_start_node < 0:
                            p3_batch_start_node = hnsw._upper_level_greedy(blob_ptr)
                        results2 = hnsw.search_fp32_scored(
                            blob_ptr, sk, scratch_dists, ef_f, p3_batch_start_node)
                        for ri3 in range(len(results2)):
                            if len(filtered_ids) >= k: break
                            var _ext_id = results2[ri3]
                            if _dead and (shared_hnsw[].slot_dead(_ext_id)
                                          or (_ttls and _slot_expired(shared_hnsw, keyspace, _ext_id))):
                                continue                       # #46: the document is gone
                            if filters.count == 0:
                                filtered_ids.append(_ext_id)
                                filtered_scores.append(scratch_dists[ri3])
                                continue
                            if passes_filters(_ext_id, shared_hnsw, keyspace, filters):
                                filtered_ids.append(_ext_id)
                                filtered_scores.append(scratch_dists[ri3])
                        if len(filtered_ids) >= k or sk >= n_total or sk >= FT_FILTER_MAX_CANDIDATES:
                            break
                        sk *= 4
                        if sk > n_total: sk = n_total
                        if sk > FT_FILTER_MAX_CANDIDATES: sk = FT_FILTER_MAX_CANDIDATES
                    _rescore_knn(hnsw, blob_ptr, filtered_ids, filtered_scores)
                    merge_delta(shared_hnsw, hnsw.num_nodes, blob_ptr, k, filters, keyspace,
                                filtered_ids, filtered_scores)          # drop-in
                    _ft_knn_reply(writer, filtered_ids, filtered_scores, keyspace, shared_hnsw, opts,
                                  sort, limit_off, limit_cnt)
                else:
                    if use_gpu_search:
                        # §7v2: Metal FFI path (async dispatch_semaphore, proven).
                        # Native Mojo GPU (search_gpu_native) compiled in but disabled:
                        # DeviceContext.synchronize() adds ~5ms vs 100µs Metal semaphore.
                        comptime if CompilationTarget.is_macos():
                            results2 = hnsw.search_gpu_brute_force(blob_ptr, k, scratch_dists, worker_id)
                    else:
                        # HNSW fallback (P3 cached greedy start)
                        if p3_batch_start_node < 0:
                            p3_batch_start_node = hnsw._upper_level_greedy(blob_ptr)
                        results2 = hnsw.search_fp32_scored(
                            blob_ptr, k, scratch_dists, ef_query,
                            p3_batch_start_node)
                    _rescore_knn(hnsw, blob_ptr, results2, scratch_dists)
                    merge_delta(shared_hnsw, hnsw.num_nodes, blob_ptr, k, filters, keyspace,
                                results2, scratch_dists)                # drop-in
                    _ft_knn_reply(writer, results2, scratch_dists, keyspace, shared_hnsw, opts,
                                  sort, limit_off, limit_cnt)
    else:
        # gh #361: `FT.SEARCH <index>` with no query is an arity error, not `[]`.
        writer.append_error_response("ERR wrong number of arguments for 'FT.SEARCH' command")
    return ci
