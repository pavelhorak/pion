"""Sorted set commands: ZREM, ZCARD, ZRANK, ZSCORE, ZCOUNT, ZINCRBY, ZRANGE variants, ZUNION/ZINTER stores, ZSCAN, ZPOPMAX, ZRANDMEMBER, ZMSCORE, ZLEXCOUNT, ZREMRANGE*."""
from src.common.container_free import remove_and_free
from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.utils import rand_count, strict_atol, bytes_to_string, _glob_match, _glob_all, arg_eq, parse_redis_double, DOUBLE_RANGE, DOUBLE_VALUE, parse_int64_strict, scan_cursor, scan_count
from std.memory.unsafe_pointer import UnsafePointer
from std.memory import alloc, unsafe_memcpy, unsafe_memset
from std.collections import Array, List
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.io.wal import WAL, gv_bytes
from src.common.skip_list import SlabSkipList
from src.memory.object_pool import ObjectPool


# ── Multi-key zset operations (gh #232) ───────────────────────────────────────
#
# ZUNION/ZINTER/ZDIFF and their STORE forms were doing none of what Redis does:
#
#   scores were NOT aggregated   `ZADD z1 2 b; ZADD z2 10 b; ZUNION 2 z1 z2`
#                                answered b=2 where Redis answers b=12 (SUM is
#                                the default aggregate). ZUNIONSTORE persisted
#                                that wrong score.
#   WEIGHTS/AGGREGATE ignored    silently — and worse, the option scan then
#                                failed to find WITHSCORES, so asking for
#                                weights dropped the scores from the reply
#                                entirely. Same shape as the gh #251 unified
#                                ZRANGE modifiers being ignored.
#   SETs rejected                a set is a zset whose members all score 1, and
#                                Redis accepts one anywhere a zset is taken.
#
# One accumulator serves all of them. Ordering comes from inserting the results
# into a temp SlabSkipList rather than a hand-rolled sort: it already orders by
# (score, member-lex), which is exactly Redis's order and the thing gh #251
# fixed, so the output matches ZRANGE by construction.

comptime ZAGG_SUM = 0
comptime ZAGG_MIN = 1
comptime ZAGG_MAX = 2


@fieldwise_init
struct ScoreBound(Copyable, Movable, ImplicitlyCopyable):
    var value: Float64
    var excl: Bool
    var ok: Bool


def _score_bound(p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int) -> ScoreBound:
    """A score-range bound (ZCOUNT, ZRANGEBYSCORE, ZREVRANGEBYSCORE,
    ZREMRANGEBYSCORE, ZRANGE BYSCORE) read as Redis's zslParseRange reads it: an
    optional "(" for exclusive, then strtod over the rest (gh #393).

    Five copies of a hand-rolled parser did this before. Each treated ANY
    argument starting with "+" as +inf (so `+100` and `+.5` meant infinity),
    stood in ±1e18 for the infinities (a member scored above 1e18 fell outside
    `-inf +inf`), and read the rest with atof."""
    var excl = False
    var q = p
    var m = n
    if m > 0 and p[0] == 40:          # "("
        excl = True
        q = p.unsafe_offset(1)
        m = n - 1
    var r = parse_redis_double(q, m, DOUBLE_RANGE)
    return ScoreBound(r.value, excl, r.ok)


@always_inline
def _zsetop_member_count(keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                         key: String) -> Int:
    """Cardinality of a ZSET or SET key; 0 for anything else."""
    var v = keyspace[].get(key)
    if v.is_none(): return 0
    if v.type.value == ValueType.ZSET:
        var n = 0
        var c = v.as_zset().bitcast[SlabSkipList]()[].head[].forward[0]
        while is_not_null(c):
            n += 1; c = c[].forward[0]
        return n
    if v.type.value == ValueType.SET:
        return v.as_set().bitcast[SlabHashMap]()[].size
    return 0


@always_inline
def _zsetop_accumulate(keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                       keys: List[String], weights: List[Float64], aggregate: Int,
                       mut out_members: List[GenericValue],
                       mut out_scores: List[Float64],
                       mut out_hits: List[Int]) raises:
    """Fold every key's (member, score) pairs into parallel result lists.

    A SET contributes each member at score 1 — that is Redis's rule, not an
    approximation: `SADD s a b; ZUNION 1 s WITHSCORES` returns a=1 b=1, and a
    member present in both a set and a zset sums as usual.

    `out_hits` counts how many SOURCE keys contained each member, which is what
    ZINTER/ZINTERCARD filter on. Union just ignores it.
    """
    var seen = alloc[SlabHashMap](1)
    seen.unsafe_write(SlabHashMap(64))
    for ki in range(len(keys)):
        var w = weights[ki] if ki < len(weights) else Float64(1.0)
        var v = keyspace[].get(keys[ki])
        if v.is_none(): continue
        # Collect this key's pairs, then fold — the two container shapes differ
        # only in how the pairs are produced.
        var mems = List[GenericValue]()
        var scs = List[Float64]()
        if v.type.value == ValueType.ZSET:
            var c = v.as_zset().bitcast[SlabSkipList]()[].head[].forward[0]
            while is_not_null(c):
                mems.append(c[].obj); scs.append(c[].score)
                c = c[].forward[0]
        elif v.type.value == ValueType.SET:
            var sp = v.as_set().bitcast[SlabHashMap]()
            for slot in range(sp[].capacity):
                var m = sp[].metadata[slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    mems.append(sp[].keys[slot]); scs.append(Float64(1.0))
        for mi in range(len(mems)):
            var contrib = scs[mi] * w
            # Redis's rule for NaN (inf * 0, or inf + -inf when summing): the
            # score becomes 0. MIN and MAX already ignore a NaN operand, as
            # Redis's comparisons do. A NaN score would break the order.
            if contrib != contrib:
                contrib = 0.0
            var at = seen[].get(mems[mi])
            if at.is_none():
                seen[].set(mems[mi], GenericValue.from_int(Int64(len(out_members))))
                out_members.append(mems[mi]); out_scores.append(contrib); out_hits.append(1)
            else:
                var idx = Int(at.as_int())
                out_hits[idx] = out_hits[idx] + 1
                if aggregate == ZAGG_MIN:
                    if contrib < out_scores[idx]: out_scores[idx] = contrib
                elif aggregate == ZAGG_MAX:
                    if contrib > out_scores[idx]: out_scores[idx] = contrib
                else:
                    var _sum = out_scores[idx] + contrib
                    out_scores[idx] = _sum if _sum == _sum else 0.0
    # `seen` BORROWS the sources' members (and out_members does too): it must
    # not free them. Destroying it normally freed every source member, so
    # ZUNION/ZINTER/ZDIFF — read-only — left the sources serving freed memory.
    seen[].forget_borrowed()
    seen.unsafe_deinit_pointee(); seen.free()


@always_inline
def _zsetop_parse_opts(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
                       start: Int, num_tokens: Int, numkeys: Int,
                       mut weights: List[Float64], mut aggregate: Int,
                       mut withscores: Bool) raises -> Int:
    """Parse [WEIGHTS w...] [AGGREGATE SUM|MIN|MAX] [WITHSCORES] in any order.

    Returns the index one past the last option token. Order matters: scanning
    for WITHSCORES only at a fixed offset is what made `ZUNION ... WEIGHTS 2 3
    WITHSCORES` drop the scores.
    """
    var j = start
    while j < num_tokens:
        var tp = tokens[j].ptr; var tl = tokens[j].length
        if tl == 7 and (tp[0]|0x20) == 119:            # WEIGHTS
            j += 1
            for _ in range(numkeys):
                if j < num_tokens:
                    var _wp = parse_redis_double(tokens[j].ptr, tokens[j].length, DOUBLE_VALUE)
                    if not _wp.ok:
                        raise Error("ERR weight value is not a float")
                    weights.append(_wp.value); j += 1
        elif tl == 9 and (tp[0]|0x20) == 97:           # AGGREGATE
            j += 1
            if j < num_tokens:
                var ap = tokens[j].ptr; var al = tokens[j].length
                if al == 3 and (ap[0]|0x20) == 109 and (ap[1]|0x20) == 105: aggregate = ZAGG_MIN
                elif al == 3 and (ap[0]|0x20) == 109 and (ap[1]|0x20) == 97: aggregate = ZAGG_MAX
                else: aggregate = ZAGG_SUM
                j += 1
        elif tl == 10 and (tp[0]|0x20) == 119:         # WITHSCORES
            withscores = True; j += 1
        else:
            break
    return j


@always_inline
def _zsetop_emit_score(mut writer: ResponseWriter, sc: Float64):
    """Integer-valued scores print bare (`2`, not `2.0`), as ZRANGE does."""
    writer.append_bulk_score_response(sc)


@always_inline
def handle_zrem(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin], wal: UnsafePointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """ZREM key member [member ...] -> number of removed members."""
    if i + 2 < num_tokens:
        var _zrm_key = tokens[i+1].value()
        var _zrm_val = keyspace[].get(_zrm_key)
        var _i = i + 1
        if _zrm_val.is_none():
            while _i + 1 < num_tokens and tokens[_i+1].marker != 0:
                _i += 1
            writer.append_int_response(Int64(0))
        elif _zrm_val.type.value == ValueType.ZSET:
            var _zrm_zp = _zrm_val.as_zset().bitcast[SlabSkipList]()
            # gh #394: one O(log n) remove per member. This used to copy every
            # node out, reset() the set and re-insert the survivors (O(n) per
            # ZREM), and freed neither the removed members nor its own copies
            # of the arguments.
            var _zrm_removed = 0
            var _zrm_ni = _i + 1
            while _zrm_ni < num_tokens and tokens[_zrm_ni].marker != 0:
                var _zrm_m = GenericValue.borrow(tokens[_zrm_ni].ptr, tokens[_zrm_ni].length)
                if _zrm_zp[].remove(_zrm_m):
                    _zrm_removed += 1
                    _ = wal[].append_kv(12, tokens[i+1].ptr, tokens[i+1].length,
                                        tokens[_zrm_ni].ptr, tokens[_zrm_ni].length)
                _i = _zrm_ni; _zrm_ni += 1
            writer.append_int_response(Int64(_zrm_removed))
            # gh #234: Redis removes an aggregate the moment its last element goes.
            if _zrm_zp[].length == 0:
                _ = remove_and_free(keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                if is_not_null(wal):
                    _ = wal[].append(2, tokens[i+1].ptr, tokens[i+1].length)
        else:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrem' command")
        return 0


@always_inline
def handle_zcard(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZCARD key -> cardinality of sorted set."""
    if i + 1 < num_tokens:
        var _zcv = keyspace[].get(tokens[i+1].value())
        if _zcv.is_none(): writer.append_int_response(Int64(0))
        elif _zcv.type.value == ValueType.ZSET:
            writer.append_int_response(Int64(_zcv.as_zset().bitcast[SlabSkipList]()[].length))
        else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zcard' command")
        return 0


@always_inline
def handle_zrank(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZRANK key member -> rank (0-based) or nil."""
    if i + 2 < num_tokens:
        var _zrk_v = keyspace[].get(tokens[i+1].value())
        var _zrk_m = GenericValue.borrow(tokens[i+2].ptr, tokens[i+2].length)
        if _zrk_v.is_none(): writer.append_null_response()
        elif _zrk_v.type.value == ValueType.ZSET:
            var _zrk_p = _zrk_v.as_zset().bitcast[SlabSkipList]()
            var _zrk_c = _zrk_p[].head[].forward[0]; var _zrk_r = 0; var _zrk_f = False
            while is_not_null(_zrk_c):
                if _zrk_c[].obj == _zrk_m: _zrk_f = True; break
                _zrk_r += 1; _zrk_c = _zrk_c[].forward[0]
            if _zrk_f: writer.append_int_response(Int64(_zrk_r))
            else: writer.append_null_response()
        else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrank' command")
        return 0


@always_inline
def handle_zrevrank(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZREVRANK key member -> reverse rank or nil."""
    if i + 2 < num_tokens:
        var _zrv_v = keyspace[].get(tokens[i+1].value())
        var _zrv_m = GenericValue.borrow(tokens[i+2].ptr, tokens[i+2].length)
        if _zrv_v.is_none(): writer.append_null_response()
        elif _zrv_v.type.value == ValueType.ZSET:
            var _zrvp = _zrv_v.as_zset().bitcast[SlabSkipList]()
            var _zrv_len = _zrvp[].length
            var _zrv_c = _zrvp[].head[].forward[0]; var _zrv_r = 0; var _zrv_f = False
            while is_not_null(_zrv_c):
                if _zrv_c[].obj == _zrv_m: _zrv_f = True; break
                _zrv_r += 1; _zrv_c = _zrv_c[].forward[0]
            if _zrv_f: writer.append_int_response(Int64(_zrv_len - _zrv_r - 1))
            else: writer.append_null_response()
        else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrevrank' command")
        return 0


@always_inline
def handle_zscore(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZSCORE key member -> score (RESP2 bulk string, RESP3 double) or nil."""
    if i + 2 < num_tokens:
        var _zsv = keyspace[].get(tokens[i+1].value())
        var _zsm = GenericValue.borrow(tokens[i+2].ptr, tokens[i+2].length)
        if _zsv.is_none(): writer.append_null_response()
        elif _zsv.type.value == ValueType.ZSET:
            var _zsp = _zsv.as_zset().bitcast[SlabSkipList]()
            var _zsc = _zsp[].head[].forward[0]; var _zsf = False
            while is_not_null(_zsc):
                if _zsc[].obj == _zsm:
                    writer.append_score_response(_zsc[].score)
                    _zsf = True; break
                _zsc = _zsc[].forward[0]
            if not _zsf: writer.append_null_response()
        else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zscore' command")
        return 0


@always_inline
def handle_zcount(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZCOUNT key min max -> count of members in score range."""
    if i + 3 < num_tokens:
        var _zcv = keyspace[].get(tokens[i+1].value())
        var _zcap = tokens[i+2].ptr; var _zcal = tokens[i+2].length
        var _zcbp = tokens[i+3].ptr; var _zcbl = tokens[i+3].length
        var _zca_excl = False; var _zcb_excl = False
        var _zca: Float64; var _zcb: Float64
        var _sb_zca = _score_bound(_zcap, _zcal)   # gh #393: zslParseRange's rules
        if not _sb_zca.ok:
            raise Error("ERR min or max is not a float")
        _zca = _sb_zca.value
        _zca_excl = _sb_zca.excl
        var _sb_zcb = _score_bound(_zcbp, _zcbl)   # gh #393: zslParseRange's rules
        if not _sb_zcb.ok:
            raise Error("ERR min or max is not a float")
        _zcb = _sb_zcb.value
        _zcb_excl = _sb_zcb.excl
        # gh #232: a key holding a non-zset answered 0, not WRONGTYPE.
        if not _zcv.is_none() and _zcv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 3
        var _zcc = 0
        if not _zcv.is_none() and _zcv.type.value == ValueType.ZSET:
            var _zcp = _zcv.as_zset().bitcast[SlabSkipList]()
            var _zccur = _zcp[].head[].forward[0]
            while is_not_null(_zccur):
                var _s = _zccur[].score
                var _lo_ok = (_zca_excl and _s > _zca) or (not _zca_excl and _s >= _zca)
                var _hi_ok = (_zcb_excl and _s < _zcb) or (not _zcb_excl and _s <= _zcb)
                if _lo_ok and _hi_ok: _zcc += 1
                if _s > _zcb: break
                _zccur = _zccur[].forward[0]
        writer.append_int_response(Int64(_zcc))
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zcount' command")
        return 0


@always_inline
def handle_zincrby(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin], wal: UnsafePointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """ZINCRBY key increment member -> new score."""
    if i + 3 < num_tokens:
        var _zib_key = tokens[i+1].value()
        # gh #393: Redis's value rules (string2d): "inf" is a score, "nan",
        # "1e400" and trailing bytes are not.
        var _zib_p = parse_redis_double(tokens[i+2].ptr, tokens[i+2].length, DOUBLE_VALUE)
        if not _zib_p.ok:
            raise Error("ERR value is not a valid float")
        var _zib_inc = _zib_p.value
        var _zib_mem = GenericValue.from_ptr(tokens[i+3].ptr, tokens[i+3].length)
        var _zib_v = keyspace[].get(_zib_key)
        var _zib_zp: UnsafePointer[SlabSkipList, MutUntrackedOrigin]
        if _zib_v.is_none():
            _zib_zp = skip_list_pool[].acquire()
            _zib_zp.unsafe_write(SlabSkipList(16))
            var _new_v = GenericValue(); _new_v.type = ValueType(ValueType.ZSET)
            _new_v.set_ptr(_zib_zp.bitcast[NoneType]())
            keyspace[].set(_zib_key, _new_v)
            _zib_zp[].insert(Float64(_zib_inc), _zib_mem)
            _ = wal[].append_scored(9, tokens[i+1].ptr, tokens[i+1].length,
                                    Float64(_zib_inc), tokens[i+3].ptr, tokens[i+3].length)
            writer.append_score_response(Float64(_zib_inc))
        elif _zib_v.type.value == ValueType.ZSET:
            _zib_zp = _zib_v.as_zset().bitcast[SlabSkipList]()
            # Find existing score or start from 0
            var _zib_old: Float64 = 0.0; var _zib_found = False
            var _zib_ss = List[Float64](); var _zib_oo = List[GenericValue]()
            var _zib_c = _zib_zp[].head[].forward[0]
            while is_not_null(_zib_c):
                _zib_ss.append(_zib_c[].score); _zib_oo.append(_zib_c[].obj)
                if _zib_c[].obj == _zib_mem: _zib_old = _zib_c[].score; _zib_found = True
                _zib_c = _zib_c[].forward[0]
            var _zib_new = _zib_old + Float64(_zib_inc)
            if _zib_new != _zib_new:
                # inf + -inf. Redis refuses and leaves the member as it was;
                # a NaN score would break the order every range walks.
                writer.append_error_response("ERR resulting score is not a number (NaN)")
                return 3
            _zib_zp[].reset()
            for _ji in range(len(_zib_ss)):
                if _zib_oo[_ji] == _zib_mem: _zib_zp[].insert(_zib_new, _zib_mem)
                else: _zib_zp[].insert(_zib_ss[_ji], _zib_oo[_ji])
            if not _zib_found: _zib_zp[].insert(_zib_new, _zib_mem)
            if _zib_found:
                _ = wal[].append_kv(12, tokens[i+1].ptr, tokens[i+1].length,
                                    tokens[i+3].ptr, tokens[i+3].length)
            _ = wal[].append_scored(9, tokens[i+1].ptr, tokens[i+1].length,
                                    _zib_new, tokens[i+3].ptr, tokens[i+3].length)
            writer.append_score_response(_zib_new)
        else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zincrby' command")
        return 0


@always_inline
def _zr_emit(mut writer: ResponseWriter, objs: List[GenericValue],
             scores: List[Float64], with_scores: Bool,
             off: Int, cnt: Int) raises:
    """Emit `objs` (already in final order) after applying LIMIT.

    LIMIT is applied AFTER ordering, which is why the collect loop below never
    short-circuits on it the way ZRANGEBYSCORE's does: under REV the first N
    matches in skip-list order are the LAST N of the reply, so stopping early
    would return the wrong window."""
    var _n = len(objs)
    if off < 0: _n = 0      # Redis skips `offset--` times: a negative one never ends (gh #393)
    var _start = off if off > 0 else 0
    if _start > _n: _start = _n
    var _end = _n
    if cnt >= 0:
        _end = _start + cnt
        if _end > _n: _end = _n
    var _emitted = _end - _start
    if _emitted < 0: _emitted = 0
    var _h = "*" + String(_emitted * (2 if with_scores else 1)) + "\r\n"
    writer.append_to_response(_h.unsafe_ptr(), _h.byte_length())
    for _zi in range(_start, _start + _emitted):
        writer.append_bulk_value_response(objs[_zi])
        if with_scores:
            writer.append_bulk_score_response(scores[_zi])


def _zrange_index_rev(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], a_in: Int, b_in: Int, with_scores: Bool, cons: Int) raises -> Int:
    """ZRANGE key start stop REV — index range over the reversed order."""
    var _v = keyspace[].get(tokens[i+1].value())
    if _v.is_none(): writer.append_empty_array_response(); return cons
    if _v.type.value != ValueType.ZSET:
        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return cons
    var _p = _v.as_zset().bitcast[SlabSkipList]()
    var _len = _p[].length
    var _a = a_in; var _b = b_in
    if _a < 0: _a = max(0, _len + _a)
    if _b < 0: _b = _len + _b
    if _b >= _len: _b = _len - 1
    if _a > _b or _a >= _len:
        writer.append_empty_array_response(); return cons
    var _objs = List[GenericValue](); var _scores = List[Float64]()
    var _c = _p[].head[].forward[0]
    while is_not_null(_c):
        _objs.append(_c[].obj); _scores.append(_c[].score); _c = _c[].forward[0]
    var _ro = List[GenericValue](); var _rs = List[Float64]()
    # reverse rank r maps to forward index _len-1-r
    for _r in range(_a, _b + 1):
        _ro.append(_objs[_len - 1 - _r]); _rs.append(_scores[_len - 1 - _r])
    _zr_emit(writer, _ro, _rs, with_scores, 0, -1)
    return cons


def _zrange_by(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], bylex: Bool, rev: Bool, lim_off: Int, lim_cnt: Int, with_scores: Bool, cons: Int) raises -> Int:
    """ZRANGE ... BYSCORE|BYLEX [REV] [LIMIT] (gh #251).

    Under REV the two bound arguments arrive HIGH first — `ZRANGE k +inf -inf
    BYSCORE REV` — so the tokens are swapped before parsing rather than the
    comparison being inverted. Getting that backwards yields an empty array,
    not an error, which is why it needs saying."""
    var _lo_i = i + 3 if rev else i + 2
    var _hi_i = i + 2 if rev else i + 3
    var _v = keyspace[].get(tokens[i+1].value())
    if not _v.is_none() and _v.type.value != ValueType.ZSET:
        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return cons

    var _objs = List[GenericValue](); var _scores = List[Float64]()
    if not _v.is_none():
        var _p = _v.as_zset().bitcast[SlabSkipList]()
        var _c = _p[].head[].forward[0]
        if bylex:
            var _lop = tokens[_lo_i].ptr; var _lol = tokens[_lo_i].length
            var _hip = tokens[_hi_i].ptr; var _hil = tokens[_hi_i].length
            var _lo_min = (_lol == 1 and _lop[0] == 45)      # "-"
            var _lo_max = (_lol == 1 and _lop[0] == 43)      # "+"
            var _hi_max = (_hil == 1 and _hip[0] == 43)
            var _hi_min = (_hil == 1 and _hip[0] == 45)
            var _lo_incl = (not _lo_min and not _lo_max and _lol > 0 and _lop[0] == 91)
            var _lo_excl = (not _lo_min and not _lo_max and _lol > 0 and _lop[0] == 40)
            var _hi_incl = (not _hi_max and not _hi_min and _hil > 0 and _hip[0] == 91)
            var _hi_excl = (not _hi_max and not _hi_min and _hil > 0 and _hip[0] == 40)
            var _lo_s = String("")
            _lo_s += bytes_to_string(_lop + 1, _lol - 1)
            var _hi_s = String("")
            _hi_s += bytes_to_string(_hip + 1, _hil - 1)
            while is_not_null(_c):
                var _ms = _c[].obj.__str__()
                var _lo_ok = _lo_min or (_lo_incl and _ms >= _lo_s) or (_lo_excl and _ms > _lo_s)
                var _hi_ok = _hi_max or (_hi_incl and _ms <= _hi_s) or (_hi_excl and _ms < _hi_s)
                if _lo_ok and _hi_ok:
                    _objs.append(_c[].obj); _scores.append(_c[].score)
                _c = _c[].forward[0]
        else:
            var _lop = tokens[_lo_i].ptr; var _lol = tokens[_lo_i].length
            var _hip = tokens[_hi_i].ptr; var _hil = tokens[_hi_i].length
            var _lo_excl = False; var _hi_excl = False
            var _lo: Float64; var _hi: Float64
            var _sb_lo = _score_bound(_lop, _lol)   # gh #393: zslParseRange's rules
            if not _sb_lo.ok:
                raise Error("ERR min or max is not a float")
            _lo = _sb_lo.value
            _lo_excl = _sb_lo.excl
            var _sb_hi = _score_bound(_hip, _hil)   # gh #393: zslParseRange's rules
            if not _sb_hi.ok:
                raise Error("ERR min or max is not a float")
            _hi = _sb_hi.value
            _hi_excl = _sb_hi.excl
            while is_not_null(_c):
                var _s = _c[].score
                var _lo_ok = (_lo_excl and _s > _lo) or (not _lo_excl and _s >= _lo)
                var _hi_ok = (_hi_excl and _s < _hi) or (not _hi_excl and _s <= _hi)
                if _lo_ok and _hi_ok:
                    _objs.append(_c[].obj); _scores.append(_c[].score)
                if _s > _hi: break
                _c = _c[].forward[0]

    if rev:
        var _ro = List[GenericValue](); var _rs = List[Float64]()
        for _r in range(len(_objs) - 1, -1, -1):
            _ro.append(_objs[_r]); _rs.append(_scores[_r])
        _zr_emit(writer, _ro, _rs, with_scores, lim_off, lim_cnt)
    else:
        _zr_emit(writer, _objs, _scores, with_scores, lim_off, lim_cnt)
    return cons


def handle_zrange(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZRANGE key start stop [BYSCORE|BYLEX] [REV] [LIMIT offset count] [WITHSCORES].

    gh #251: the Redis 6.2 modifiers were not parsed at all, and the failure
    was the dangerous kind — `BYSCORE` and `REV` were IGNORED rather than
    rejected, so `ZRANGE k 1 3 BYSCORE` answered with ranks 1..3 and looked
    like a normal reply. The legacy `ZRANGEBYSCORE`/`ZREVRANGE` spellings were
    always correct; this is the unified form modern clients emit
    (redis-py `zrange(..., byscore=True)` / `desc=True`).
    """
    if i + 3 < num_tokens:
        var _zrng_key = tokens[i+1].value()
        var _zrng_with = False; var _zrng_cons = 3
        # ── Modifier scan ──────────────────────────────────────────────────
        var _zr_byscore = False; var _zr_bylex = False; var _zr_rev = False
        var _zr_has_limit = False; var _zr_lim_off = 0; var _zr_lim_cnt = -1
        var _zr_bad = False
        var _oi = i + 4
        while _oi < num_tokens:
            var _op = tokens[_oi].ptr; var _ol = tokens[_oi].length
            if arg_eq(_op, _ol, "withscores"):
                _zrng_with = True; _zrng_cons = _oi - i; _oi += 1
            elif arg_eq(_op, _ol, "byscore"):
                _zr_byscore = True; _zrng_cons = _oi - i; _oi += 1
            elif arg_eq(_op, _ol, "bylex"):
                _zr_bylex = True; _zrng_cons = _oi - i; _oi += 1
            elif arg_eq(_op, _ol, "rev"):
                _zr_rev = True; _zrng_cons = _oi - i; _oi += 1
            elif arg_eq(_op, _ol, "limit"):
                if _oi + 2 >= num_tokens:
                    _zr_bad = True; break
                _zr_has_limit = True
                _zr_lim_off = strict_atol(tokens[_oi+1].value())
                _zr_lim_cnt = strict_atol(tokens[_oi+2].value())
                _oi += 3; _zrng_cons = _oi - i - 1
            else:
                break
        # Redis rejects these combinations rather than guessing an intent.
        # LIMIT-without-BY gets its own message because that is the one a user
        # hits by hand; the rest share the generic syntax error, as Redis does.
        var _zr_limit_misuse = _zr_has_limit and not (_zr_byscore or _zr_bylex)
        if _zr_byscore and _zr_bylex: _zr_bad = True
        if _zr_limit_misuse: _zr_bad = True
        if _zr_bylex and _zrng_with: _zr_bad = True
        if _zr_bad:
            if _zr_limit_misuse:
                writer.append_error_response("ERR syntax error, LIMIT is only supported in combination with either BYSCORE or BYLEX")
            else:
                writer.append_error_response("ERR syntax error")
            return _zrng_cons

        if _zr_byscore or _zr_bylex:
            return _zrange_by(tokens, i, num_tokens, writer, keyspace,
                              _zr_bylex, _zr_rev, _zr_lim_off, _zr_lim_cnt,
                              _zrng_with, _zrng_cons)

        var _zrng_a = strict_atol(tokens[i+2].value()); var _zrng_b = strict_atol(tokens[i+3].value())
        # REV over an index range: start/stop are reverse ranks. Emitting the
        # forward slice reversed is exactly ZREVRANGE, so map onto its rank
        # arithmetic rather than re-deriving it.
        if _zr_rev:
            return _zrange_index_rev(tokens, i, num_tokens, writer, keyspace,
                                     _zrng_a, _zrng_b, _zrng_with, _zrng_cons)
        var _zrng_v = keyspace[].get(_zrng_key)
        if _zrng_v.is_none(): writer.append_empty_array_response()
        elif _zrng_v.type.value == ValueType.ZSET:
            var _zrp = _zrng_v.as_zset().bitcast[SlabSkipList]()
            var _zrlen = _zrp[].length
            if _zrng_a < 0: _zrng_a = max(0, _zrlen + _zrng_a)
            if _zrng_b < 0: _zrng_b = _zrlen + _zrng_b
            if _zrng_b >= _zrlen: _zrng_b = _zrlen - 1
            if _zrng_a > _zrng_b or _zrng_a >= _zrlen:
                writer.append_empty_array_response()
            else:
                var _zrn = (_zrng_b - _zrng_a + 1) * (2 if _zrng_with else 1)
                var _zrh = "*" + String(_zrn) + "\r\n"
                writer.append_to_response(_zrh.unsafe_ptr(), _zrh.byte_length())
                var _zridx = 0; var _zrc = _zrp[].head[].forward[0]
                while is_not_null(_zrc):
                    if _zridx >= _zrng_a and _zridx <= _zrng_b:
                        writer.append_bulk_value_response(_zrc[].obj)
                        if _zrng_with:
                            writer.append_bulk_score_response(_zrc[].score)
                    if _zridx >= _zrng_b: break
                    _zridx += 1; _zrc = _zrc[].forward[0]
        else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return _zrng_cons
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrange' command")
        return 0


@always_inline
def handle_zrevrange(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZREVRANGE key start stop [WITHSCORES] -> list of members in reverse order."""
    if i + 3 < num_tokens:
        var _zrv_key = tokens[i+1].value()
        var _zrva = strict_atol(tokens[i+2].value()); var _zrvb = strict_atol(tokens[i+3].value())
        var _zrv_with = False; var _zrv_cons = 3
        if i + 4 < num_tokens:
            var _op = tokens[i+4].ptr; var _ol = tokens[i+4].length
            if _ol == 10 and (_op[0]|0x20)==119: _zrv_with = True; _zrv_cons = 4
        var _zrvv = keyspace[].get(_zrv_key)
        if _zrvv.is_none(): writer.append_empty_array_response()
        elif _zrvv.type.value == ValueType.ZSET:
            var _zrvp = _zrvv.as_zset().bitcast[SlabSkipList]()
            var _zrvlen = _zrvp[].length
            if _zrva < 0: _zrva = max(0, _zrvlen + _zrva)
            if _zrvb < 0: _zrvb = _zrvlen + _zrvb
            if _zrvb >= _zrvlen: _zrvb = _zrvlen - 1
            if _zrva > _zrvb or _zrva >= _zrvlen:
                writer.append_empty_array_response()
            else:
                # Collect all, then reverse
                var _rvss = List[Float64](); var _rvoo = List[GenericValue]()
                var _rvci = _zrvp[].head[].forward[0]
                while is_not_null(_rvci):
                    _rvss.append(_rvci[].score)
                    _rvoo.append(_rvci[].obj)
                    _rvci = _rvci[].forward[0]
                # Convert from reverse rank to forward rank
                var _rev_a = _zrvlen - 1 - _zrvb; var _rev_b = _zrvlen - 1 - _zrva
                var _zrn = (_rev_b - _rev_a + 1) * (2 if _zrv_with else 1)
                var _zrh = "*" + String(_zrn) + "\r\n"
                writer.append_to_response(_zrh.unsafe_ptr(), _zrh.byte_length())
                # Output in reverse order
                for _ri in range(_rev_b, _rev_a - 1, -1):
                    writer.append_bulk_value_response(_rvoo[_ri])
                    if _zrv_with:
                        writer.append_bulk_score_response(_rvss[_ri])
        else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return _zrv_cons
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrevrange' command")
        return 0


@always_inline
def handle_zrangebyscore(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZRANGEBYSCORE key min max [WITHSCORES] [LIMIT offset count] -> members in score range."""
    if i + 3 < num_tokens:
        var _zbs_key = tokens[i+1].value()
        var _zbsap = tokens[i+2].ptr; var _zbsal = tokens[i+2].length
        var _zbsbp = tokens[i+3].ptr; var _zbsbl = tokens[i+3].length
        var _i = i + 3
        var _zbsa_excl = False; var _zbsb_excl = False
        var _zbsa: Float64; var _zbsb: Float64
        var _sb_zbsa = _score_bound(_zbsap, _zbsal)   # gh #393: zslParseRange's rules
        if not _sb_zbsa.ok:
            raise Error("ERR min or max is not a float")
        _zbsa = _sb_zbsa.value
        _zbsa_excl = _sb_zbsa.excl
        var _sb_zbsb = _score_bound(_zbsbp, _zbsbl)   # gh #393: zslParseRange's rules
        if not _sb_zbsb.ok:
            raise Error("ERR min or max is not a float")
        _zbsb = _sb_zbsb.value
        _zbsb_excl = _sb_zbsb.excl
        var _zbs_with = False; var _zbs_lim_off = 0; var _zbs_lim_cnt = -1
        while _i + 1 < num_tokens:
            var _op = tokens[_i+1].ptr; var _ol = tokens[_i+1].length
            if _ol == 10 and (_op[0]|0x20)==119: _zbs_with = True; _i += 1
            elif _ol == 5 and (_op[0]|0x20)==108:
                _zbs_lim_off = strict_atol(tokens[_i+2].value())
                _zbs_lim_cnt = strict_atol(tokens[_i+3].value()); _i += 3
            else: break
        var _zbsv = keyspace[].get(_zbs_key)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        if not _zbsv.is_none() and _zbsv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _zbs_res = List[Float64](); var _zbs_roo = List[GenericValue]()
        # gh #393: a negative LIMIT offset is an empty reply in Redis (its skip
        # loop runs `offset--` to the end of the set), and a count of 0 is zero
        # elements — both answered as if LIMIT were absent.
        if _zbs_lim_off >= 0 and _zbs_lim_cnt != 0 and not _zbsv.is_none() and _zbsv.type.value == ValueType.ZSET:
            var _zbsp = _zbsv.as_zset().bitcast[SlabSkipList]()
            var _zbsc = _zbsp[].head[].forward[0]; var _skip = _zbs_lim_off
            while is_not_null(_zbsc):
                var _s = _zbsc[].score
                var _lo = (_zbsa_excl and _s > _zbsa) or (not _zbsa_excl and _s >= _zbsa)
                var _hi = (_zbsb_excl and _s < _zbsb) or (not _zbsb_excl and _s <= _zbsb)
                if _lo and _hi:
                    if _skip > 0: _skip -= 1
                    else:
                        _zbs_res.append(_s); _zbs_roo.append(_zbsc[].obj)
                        if _zbs_lim_cnt > 0 and len(_zbs_res) >= _zbs_lim_cnt: break
                if _s > _zbsb: break
                _zbsc = _zbsc[].forward[0]
        var _zbs_n = len(_zbs_roo) * (2 if _zbs_with else 1)
        var _zbs_h = "*" + String(_zbs_n) + "\r\n"
        writer.append_to_response(_zbs_h.unsafe_ptr(), _zbs_h.byte_length())
        for _zi in range(len(_zbs_roo)):
            writer.append_bulk_value_response(_zbs_roo[_zi])
            if _zbs_with:
                writer.append_bulk_score_response(_zbs_res[_zi])
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrangebyscore' command")
        return 0


@always_inline
def handle_zrevrangebyscore(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZREVRANGEBYSCORE key max min [WITHSCORES] [LIMIT offset count] -> members in reverse score range."""
    if i + 3 < num_tokens:
        var _zrvbs_key = tokens[i+1].value()
        var _zrvbsbp = tokens[i+2].ptr; var _zrvbsbl = tokens[i+2].length  # max
        var _zrvbsap = tokens[i+3].ptr; var _zrvbsal = tokens[i+3].length  # min
        var _i = i + 3
        var _zrvbsa_excl = False; var _zrvbsb_excl = False
        var _zrvbsa: Float64; var _zrvbsb: Float64
        var _sb_zrvbsa = _score_bound(_zrvbsap, _zrvbsal)   # gh #393: zslParseRange's rules
        if not _sb_zrvbsa.ok:
            raise Error("ERR min or max is not a float")
        _zrvbsa = _sb_zrvbsa.value
        _zrvbsa_excl = _sb_zrvbsa.excl
        var _sb_zrvbsb = _score_bound(_zrvbsbp, _zrvbsbl)   # gh #393: zslParseRange's rules
        if not _sb_zrvbsb.ok:
            raise Error("ERR min or max is not a float")
        _zrvbsb = _sb_zrvbsb.value
        _zrvbsb_excl = _sb_zrvbsb.excl
        var _zrvbs_with = False; var _zrvbs_lim_off = 0; var _zrvbs_lim_cnt = -1
        while _i + 1 < num_tokens:
            var _op = tokens[_i+1].ptr; var _ol = tokens[_i+1].length
            if _ol == 10 and (_op[0]|0x20)==119: _zrvbs_with = True; _i += 1
            elif _ol == 5 and (_op[0]|0x20)==108:
                _zrvbs_lim_off = strict_atol(tokens[_i+2].value())
                _zrvbs_lim_cnt = strict_atol(tokens[_i+3].value()); _i += 3
            else: break
        var _zrvbsv = keyspace[].get(_zrvbs_key)
        var _zrvbs_res = List[Float64](); var _zrvbs_roo = List[GenericValue]()
        if _zrvbs_lim_off >= 0 and _zrvbs_lim_cnt != 0 and not _zrvbsv.is_none() and _zrvbsv.type.value == ValueType.ZSET:
            var _zrvbsp = _zrvbsv.as_zset().bitcast[SlabSkipList]()
            # Collect all in range in forward order, then reverse
            var _all_ss = List[Float64](); var _all_oo = List[GenericValue]()
            var _zrvbsc = _zrvbsp[].head[].forward[0]
            while is_not_null(_zrvbsc):
                var _s = _zrvbsc[].score
                var _lo = (_zrvbsa_excl and _s > _zrvbsa) or (not _zrvbsa_excl and _s >= _zrvbsa)
                var _hi = (_zrvbsb_excl and _s < _zrvbsb) or (not _zrvbsb_excl and _s <= _zrvbsb)
                if _lo and _hi: _all_ss.append(_s); _all_oo.append(_zrvbsc[].obj)
                if _s > _zrvbsb: break
                _zrvbsc = _zrvbsc[].forward[0]
            # Output in reverse order with LIMIT
            var _skip = _zrvbs_lim_off
            for _ri in range(len(_all_oo) - 1, -1, -1):
                if _skip > 0: _skip -= 1; continue
                _zrvbs_res.append(_all_ss[_ri]); _zrvbs_roo.append(_all_oo[_ri])
                if _zrvbs_lim_cnt > 0 and len(_zrvbs_res) >= _zrvbs_lim_cnt: break
        var _zrvbs_n = len(_zrvbs_roo) * (2 if _zrvbs_with else 1)
        var _zrvbs_h = "*" + String(_zrvbs_n) + "\r\n"
        writer.append_to_response(_zrvbs_h.unsafe_ptr(), _zrvbs_h.byte_length())
        for _zi in range(len(_zrvbs_roo)):
            writer.append_bulk_value_response(_zrvbs_roo[_zi])
            if _zrvbs_with:
                writer.append_bulk_score_response(_zrvbs_res[_zi])
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrevrangebyscore' command")
        return 0


@always_inline
def handle_zunion(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZUNION numkeys key [key ...] [WITHSCORES] -> union of sorted sets."""
    if i + 2 < num_tokens:
        var _nk = strict_atol(tokens[i+1].value()); var _i = i + 2
        # numkeys must be 1..(keys actually given): it was unbounded, so
        # `ZUNION 9223372036854775807 k` looped ~2^63 times (the worker froze).
        if _nk < 1 or _nk > num_tokens - _i:
            writer.append_error_response("ERR numkeys must be at least 1 and at most the number of keys given")
            return num_tokens - 1 - i
        var _zukeys = List[String]()
        for _ki in range(_nk):
            if _i < num_tokens: _zukeys.append(tokens[_i].value())
            _i += 1
        _i -= 1
        # gh #232: every participating key is type-checked before any
        # work. Treating a wrong-type key as an empty set silently
        # changes the answer instead of reporting the error.
        for _wtk in range(len(_zukeys)):
            var _wtv = keyspace[].get(_zukeys[_wtk])
            # A SET is accepted here: Redis treats one as a zset whose members
            # all score 1, so `ZUNION 1 <set>` is legal and returns its members.
            if (not _wtv.is_none() and _wtv.type.value != ValueType.ZSET
                    and _wtv.type.value != ValueType.SET):
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return _i - i
        var _with = False
        var _wts = List[Float64](); var _agg = ZAGG_SUM
        _i = _zsetop_parse_opts(tokens, _i + 1, num_tokens, _nk, _wts, _agg, _with) - 1
        var _mm = List[GenericValue](); var _ss = List[Float64](); var _hh = List[Int]()
        _zsetop_accumulate(keyspace, _zukeys, _wts, _agg, _mm, _ss, _hh)
        # Order via a temp skip list: it sorts by (score, member-lex), which is
        # Redis's order and matches ZRANGE by construction.
        var _sl = alloc[SlabSkipList](1); _sl.unsafe_write(SlabSkipList(16))
        for _zi in range(len(_mm)): _sl[].insert(_ss[_zi], _mm[_zi])
        var _zu_h = "*" + String(len(_mm) * (2 if _with else 1)) + "\r\n"
        writer.append_to_response(_zu_h.unsafe_ptr(), _zu_h.byte_length())
        var _c = _sl[].head[].forward[0]
        while is_not_null(_c):
            writer.append_bulk_value_response(_c[].obj)
            if _with: _zsetop_emit_score(writer, _c[].score)
            _c = _c[].forward[0]
        _sl[].release_borrowed()   # temp list of BORROWED members: unmap slabs only
        _sl.unsafe_deinit_pointee(); _sl.free()
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zunion' command")
        return 0


@always_inline
def handle_zinter(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZINTER numkeys key [key ...] [WITHSCORES] -> intersection of sorted sets."""
    if i + 2 < num_tokens:
        var _nk = strict_atol(tokens[i+1].value()); var _i = i + 2
        # numkeys must be 1..(keys actually given): it was unbounded, so
        # `ZUNION 9223372036854775807 k` looped ~2^63 times (the worker froze).
        if _nk < 1 or _nk > num_tokens - _i:
            writer.append_error_response("ERR numkeys must be at least 1 and at most the number of keys given")
            return num_tokens - 1 - i
        var _zikeys = List[String]()
        for _ki in range(_nk):
            if _i < num_tokens: _zikeys.append(tokens[_i].value())
            _i += 1
        _i -= 1
        # gh #232: every participating key is type-checked before any
        # work. Treating a wrong-type key as an empty set silently
        # changes the answer instead of reporting the error.
        for _wtk in range(len(_zikeys)):
            var _wtv = keyspace[].get(_zikeys[_wtk])
            # A SET is accepted here: Redis treats one as a zset whose members
            # all score 1, so `ZUNION 1 <set>` is legal and returns its members.
            if (not _wtv.is_none() and _wtv.type.value != ValueType.ZSET
                    and _wtv.type.value != ValueType.SET):
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return _i - i
        var _with = False
        var _wts = List[Float64](); var _agg = ZAGG_SUM
        _i = _zsetop_parse_opts(tokens, _i + 1, num_tokens, _nk, _wts, _agg, _with) - 1
        # gh #232: was an O(n*m) nested skip-list walk that also aggregated
        # nothing and ignored WEIGHTS/AGGREGATE. `out_hits` counts how many
        # source keys held each member, so the intersection is a filter on it.
        var _mm = List[GenericValue](); var _ss2 = List[Float64](); var _hh = List[Int]()
        _zsetop_accumulate(keyspace, _zikeys, _wts, _agg, _mm, _ss2, _hh)
        var _sl = alloc[SlabSkipList](1); _sl.unsafe_write(SlabSkipList(16))
        var _n_out = 0
        for _zi in range(len(_mm)):
            if _hh[_zi] == len(_zikeys):
                _sl[].insert(_ss2[_zi], _mm[_zi]); _n_out += 1
        var _zi_h = "*" + String(_n_out * (2 if _with else 1)) + "\r\n"
        writer.append_to_response(_zi_h.unsafe_ptr(), _zi_h.byte_length())
        var _c = _sl[].head[].forward[0]
        while is_not_null(_c):
            writer.append_bulk_value_response(_c[].obj)
            if _with: _zsetop_emit_score(writer, _c[].score)
            _c = _c[].forward[0]
        _sl[].release_borrowed()   # temp list of BORROWED members: unmap slabs only
        _sl.unsafe_deinit_pointee(); _sl.free()
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zinter' command")
        return 0


@always_inline
def handle_zunionstore(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """ZUNIONSTORE dest numkeys key [key ...] -> cardinality of result."""
    if i + 3 < num_tokens:
        var _zus_dest = tokens[i+1].value()
        var _nk = strict_atol(tokens[i+2].value()); var _i = i + 3
        # numkeys must be 1..(keys actually given): it was unbounded, so
        # `ZUNION 9223372036854775807 k` looped ~2^63 times (the worker froze).
        if _nk < 1 or _nk > num_tokens - _i:
            writer.append_error_response("ERR numkeys must be at least 1 and at most the number of keys given")
            return num_tokens - 1 - i
        var _zukeys = List[String]()
        for _ki in range(_nk):
            if _i < num_tokens: _zukeys.append(tokens[_i].value())
            _i += 1
        _i -= 1
        # gh #232: shares ZUNION's accumulator, so the STORED scores aggregate,
        # honour WEIGHTS/AGGREGATE and accept SETs. It used to persist the
        # FIRST score it saw for a member, which is a wrong value written to
        # the keyspace rather than merely a wrong reply.
        for _wtk in range(len(_zukeys)):
            var _wtv = keyspace[].get(_zukeys[_wtk])
            if (not _wtv.is_none() and _wtv.type.value != ValueType.ZSET
                    and _wtv.type.value != ValueType.SET):
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return _i - i
        var _wsc = False
        var _wts = List[Float64](); var _agg = ZAGG_SUM
        _i = _zsetop_parse_opts(tokens, _i + 1, num_tokens, _nk, _wts, _agg, _wsc) - 1
        var _zus_oo = List[GenericValue](); var _zus_ss = List[Float64]()
        var _hh = List[Int]()
        _zsetop_accumulate(keyspace, _zukeys, _wts, _agg, _zus_oo, _zus_ss, _hh)
        var _zus_zp = skip_list_pool[].acquire()
        _zus_zp.unsafe_write(SlabSkipList(16))
        # clone(): the accumulated members are BORROWED from the sources; a
        # stored zset sharing them was a use-after-free (and a crash) on DEL.
        for _zi in range(len(_zus_ss)): _zus_zp[].insert(_zus_ss[_zi], _zus_oo[_zi].clone())
        var _zus_gv = GenericValue(); _zus_gv.type = ValueType(ValueType.ZSET)
        _zus_gv.set_ptr(_zus_zp.bitcast[NoneType]())
        # Free the container this replaces (it may be one of the sources:
        # the result above holds clones, so that is safe). keyspace.set()
        # only swaps the handle — every re-run leaked the old zset.
        var _old_dst = GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length)
        _ = remove_and_free(keyspace, _old_dst)
        _old_dst.free_str_payload()
        keyspace[].set(_zus_dest, _zus_gv)
        writer.append_int_response(Int64(len(_zus_ss)))
        # gh #251: an empty result DELETES the destination (see handle_sinterstore).
        if len(_zus_ss) == 0: _ = remove_and_free(keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zunionstore' command")
        return 0


@always_inline
def handle_zinterstore(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """ZINTERSTORE dest numkeys key [key ...] -> cardinality of result."""
    if i + 3 < num_tokens:
        var _zis_dest = tokens[i+1].value()
        var _nk = strict_atol(tokens[i+2].value()); var _i = i + 3
        # numkeys must be 1..(keys actually given): it was unbounded, so
        # `ZUNION 9223372036854775807 k` looped ~2^63 times (the worker froze).
        if _nk < 1 or _nk > num_tokens - _i:
            writer.append_error_response("ERR numkeys must be at least 1 and at most the number of keys given")
            return num_tokens - 1 - i
        var _zikeys = List[String]()
        for _ki in range(_nk):
            if _i < num_tokens: _zikeys.append(tokens[_i].value())
            _i += 1
        _i -= 1
        # gh #232: same accumulator as ZINTER — aggregates, honours
        # WEIGHTS/AGGREGATE, and accepts SETs.
        for _wtk in range(len(_zikeys)):
            var _wtv = keyspace[].get(_zikeys[_wtk])
            if (not _wtv.is_none() and _wtv.type.value != ValueType.ZSET
                    and _wtv.type.value != ValueType.SET):
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return _i - i
        var _wsc = False
        var _wts = List[Float64](); var _agg = ZAGG_SUM
        _i = _zsetop_parse_opts(tokens, _i + 1, num_tokens, _nk, _wts, _agg, _wsc) - 1
        var _mm = List[GenericValue](); var _all_ss = List[Float64](); var _hh = List[Int]()
        _zsetop_accumulate(keyspace, _zikeys, _wts, _agg, _mm, _all_ss, _hh)
        var _zi_ss = List[Float64](); var _zi_oo = List[GenericValue]()
        for _zi in range(len(_mm)):
            if _hh[_zi] == len(_zikeys):
                _zi_ss.append(_all_ss[_zi]); _zi_oo.append(_mm[_zi])
        var _zis_zp = skip_list_pool[].acquire()
        _zis_zp.unsafe_write(SlabSkipList(16))
        for _zi in range(len(_zi_ss)): _zis_zp[].insert(_zi_ss[_zi], _zi_oo[_zi].clone())   # owned copy
        var _zis_gv = GenericValue(); _zis_gv.type = ValueType(ValueType.ZSET)
        _zis_gv.set_ptr(_zis_zp.bitcast[NoneType]())
        # Free the container this replaces (it may be one of the sources:
        # the result above holds clones, so that is safe). keyspace.set()
        # only swaps the handle — every re-run leaked the old zset.
        var _old_dst = GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length)
        _ = remove_and_free(keyspace, _old_dst)
        _old_dst.free_str_payload()
        keyspace[].set(_zis_dest, _zis_gv)
        writer.append_int_response(Int64(len(_zi_ss)))
        if len(_zi_ss) == 0: _ = remove_and_free(keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))   # gh #251
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zinterstore' command")
        return 0


@always_inline
def handle_zlexcount(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZLEXCOUNT key min max -> count of members in lex range."""
    if i + 3 < num_tokens:
        var _zlv = keyspace[].get(tokens[i+1].value())
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        if not _zlv.is_none() and _zlv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _zlap = tokens[i+2].ptr; var _zlal = tokens[i+2].length
        var _zlbp = tokens[i+3].ptr; var _zlbl = tokens[i+3].length
        var _zla_min = (_zlal == 1 and _zlap[0] == 45)
        var _zla_max = (_zlal == 1 and _zlap[0] == 43)
        var _zlb_min = (_zlbl == 1 and _zlbp[0] == 45)
        var _zlb_max = (_zlbl == 1 and _zlbp[0] == 43)
        var _zla_incl = (not _zla_min and not _zla_max and _zlal > 0 and _zlap[0] == 91)
        var _zla_excl = (not _zla_min and not _zla_max and _zlal > 0 and _zlap[0] == 40)
        var _zlb_incl = (not _zlb_min and not _zlb_max and _zlbl > 0 and _zlbp[0] == 91)
        var _zlb_excl = (not _zlb_min and not _zlb_max and _zlbl > 0 and _zlbp[0] == 40)
        var _zla_str = String("")
        _zla_str += bytes_to_string(_zlap + 1, _zlal - 1)
        var _zlb_str = String("")
        _zlb_str += bytes_to_string(_zlbp + 1, _zlbl - 1)
        var _zlc = 0
        if not _zlv.is_none() and _zlv.type.value == ValueType.ZSET:
            var _zlp = _zlv.as_zset().bitcast[SlabSkipList]()
            var _zlcur = _zlp[].head[].forward[0]
            while is_not_null(_zlcur):
                var _ms = _zlcur[].obj.__str__()
                var _lo_ok = _zla_min or (_zla_max and False) or (_zla_incl and _ms >= _zla_str) or (_zla_excl and _ms > _zla_str)
                var _hi_ok = _zlb_max or (_zlb_min and False) or (_zlb_incl and _ms <= _zlb_str) or (_zlb_excl and _ms < _zlb_str)
                if _lo_ok and _hi_ok: _zlc += 1
                _zlcur = _zlcur[].forward[0]
        writer.append_int_response(Int64(_zlc))
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zlexcount' command")
        return 0


@always_inline
def handle_zrangebylex(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZRANGEBYLEX key min max [LIMIT offset count] -> members in lex range."""
    if i + 3 < num_tokens:
        var _zbl_key = tokens[i+1].value()
        var _zbla_p = tokens[i+2].ptr; var _zbla_l = tokens[i+2].length
        var _zblb_p = tokens[i+3].ptr; var _zblb_l = tokens[i+3].length
        var _i = i + 3
        var _zbla_min = (_zbla_l == 1 and _zbla_p[0] == 45)
        var _zbla_max = (_zbla_l == 1 and _zbla_p[0] == 43)
        var _zblb_max = (_zblb_l == 1 and _zblb_p[0] == 43)
        var _zbla_incl = (not _zbla_min and not _zbla_max and _zbla_l > 0 and _zbla_p[0] == 91)
        var _zbla_excl = (not _zbla_min and not _zbla_max and _zbla_l > 0 and _zbla_p[0] == 40)
        var _zblb_incl = (not _zblb_max and _zblb_l > 0 and _zblb_p[0] == 91)
        var _zblb_excl = (not _zblb_max and _zblb_l > 0 and _zblb_p[0] == 40)
        var _zbla_str = String("")
        _zbla_str += bytes_to_string(_zbla_p + 1, _zbla_l - 1)
        var _zblb_str = String("")
        _zblb_str += bytes_to_string(_zblb_p + 1, _zblb_l - 1)
        var _zbl_lim_off = 0; var _zbl_lim_cnt = -1
        if _i + 1 < num_tokens:
            var _op = tokens[_i+1].ptr; var _ol = tokens[_i+1].length
            if _ol == 5 and (_op[0]|0x20)==108:
                _zbl_lim_off = strict_atol(tokens[_i+2].value())
                _zbl_lim_cnt = strict_atol(tokens[_i+3].value())
                _i += 3
        var _zblv = keyspace[].get(_zbl_key)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        if not _zblv.is_none() and _zblv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _zbl_res = List[GenericValue]()
        if _zbl_lim_off >= 0 and _zbl_lim_cnt != 0 and not _zblv.is_none() and _zblv.type.value == ValueType.ZSET:
            var _zblp = _zblv.as_zset().bitcast[SlabSkipList]()
            var _zblc = _zblp[].head[].forward[0]; var _zbl_skip = _zbl_lim_off
            while is_not_null(_zblc):
                var _ms = _zblc[].obj.__str__()
                var _lo_ok = _zbla_min or (_zbla_incl and _ms >= _zbla_str) or (_zbla_excl and _ms > _zbla_str)
                var _hi_ok = _zblb_max or (_zblb_incl and _ms <= _zblb_str) or (_zblb_excl and _ms < _zblb_str)
                if _lo_ok and _hi_ok:
                    if _zbl_skip > 0: _zbl_skip -= 1
                    else:
                        _zbl_res.append(_zblc[].obj)
                        if _zbl_lim_cnt > 0 and len(_zbl_res) >= _zbl_lim_cnt: break
                _zblc = _zblc[].forward[0]
        var _zbl_h = "*" + String(len(_zbl_res)) + "\r\n"
        writer.append_to_response(_zbl_h.unsafe_ptr(), _zbl_h.byte_length())
        for _zi in range(len(_zbl_res)): writer.append_bulk_value_response(_zbl_res[_zi])
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrangebylex' command")
        return 0


@always_inline
def handle_zrevrangebylex(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZREVRANGEBYLEX key max min [LIMIT offset count] -> members in reverse lex range."""
    if i + 3 < num_tokens:
        var _zrvlk = tokens[i+1].value()
        var _zrvbp = tokens[i+2].ptr; var _zrvbl = tokens[i+2].length  # max (hi)
        var _zrvap = tokens[i+3].ptr; var _zrval = tokens[i+3].length  # min (lo)
        var _i = i + 3
        var _zrvla_min = (_zrval == 1 and _zrvap[0] == 45); var _zrvla_max = (_zrval == 1 and _zrvap[0] == 43)
        var _zrvlb_max = (_zrvbl == 1 and _zrvbp[0] == 43)
        var _zrvla_incl = (not _zrvla_min and not _zrvla_max and _zrval > 0 and _zrvap[0] == 91)
        var _zrvla_excl = (not _zrvla_min and not _zrvla_max and _zrval > 0 and _zrvap[0] == 40)
        var _zrvlb_incl = (not _zrvlb_max and _zrvbl > 0 and _zrvbp[0] == 91)
        var _zrvlb_excl = (not _zrvlb_max and _zrvbl > 0 and _zrvbp[0] == 40)
        var _zrvla_str = String("")
        _zrvla_str += bytes_to_string(_zrvap + 1, _zrval - 1)
        var _zrvlb_str = String("")
        _zrvlb_str += bytes_to_string(_zrvbp + 1, _zrvbl - 1)
        # LIMIT was never parsed here, so every window answered the whole range.
        var _zrvl_lim_off = 0; var _zrvl_lim_cnt = -1
        if _i + 3 < num_tokens and tokens[_i+1].length == 5 and (tokens[_i+1].ptr[0]|0x20) == 108:
            _zrvl_lim_off = strict_atol(tokens[_i+2].value())
            _zrvl_lim_cnt = strict_atol(tokens[_i+3].value())
            _i += 3
        var _zrvlv = keyspace[].get(_zrvlk)
        var _zrvl_res = List[GenericValue]()
        if _zrvl_lim_off >= 0 and _zrvl_lim_cnt != 0 and not _zrvlv.is_none() and _zrvlv.type.value == ValueType.ZSET:
            var _zrvlp = _zrvlv.as_zset().bitcast[SlabSkipList]()
            var _all_oo = List[GenericValue](); var _zrvlc = _zrvlp[].head[].forward[0]
            while is_not_null(_zrvlc):
                var _ms = _zrvlc[].obj.__str__()
                var _lo_ok = _zrvla_min or (_zrvla_incl and _ms >= _zrvla_str) or (_zrvla_excl and _ms > _zrvla_str)
                var _hi_ok = _zrvlb_max or (_zrvlb_incl and _ms <= _zrvlb_str) or (_zrvlb_excl and _ms < _zrvlb_str)
                if _lo_ok and _hi_ok: _all_oo.append(_zrvlc[].obj)
                _zrvlc = _zrvlc[].forward[0]
            var _zrvl_skip = _zrvl_lim_off
            for _ri in range(len(_all_oo) - 1, -1, -1):
                if _zrvl_skip > 0: _zrvl_skip -= 1; continue
                _zrvl_res.append(_all_oo[_ri])
                if _zrvl_lim_cnt > 0 and len(_zrvl_res) >= _zrvl_lim_cnt: break
        var _zrvl_h = "*" + String(len(_zrvl_res)) + "\r\n"
        writer.append_to_response(_zrvl_h.unsafe_ptr(), _zrvl_h.byte_length())
        for _zi in range(len(_zrvl_res)): writer.append_bulk_value_response(_zrvl_res[_zi])
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrevrangebylex' command")
        return 0


@always_inline
def handle_zmpop(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], wal: UnsafePointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """ZMPOP numkeys key [key ...] MIN|MAX [COUNT count] (gh #251).

    Was unimplemented — it errored while its list twin LMPOP worked. Pops from
    the FIRST key that holds a non-empty zset and replies
    `[key, [[member, score], ...]]`; nil when no key has anything, which is why
    a missing key is skipped rather than being an error.

    Note the reply nests each member/score as its own 2-element array — unlike
    ZPOPMIN/ZPOPMAX, which flatten them. Emitting the flat shape here would
    parse as half as many pairs on the client."""
    if i + 3 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'zmpop' command")
        return 0
    var _nk = strict_atol(tokens[i+1].value())
    if _nk <= 0 or i + 1 + _nk >= num_tokens:
        writer.append_error_response("ERR numkeys should be greater than 0")
        return num_tokens - 1 - i
    var _di = i + 2 + _nk                      # MIN|MAX
    if _di >= num_tokens:
        writer.append_error_response("ERR syntax error")
        return num_tokens - 1 - i
    var _from_min = arg_eq(tokens[_di].ptr, tokens[_di].length, "min")
    if not _from_min and not arg_eq(tokens[_di].ptr, tokens[_di].length, "max"):
        writer.append_error_response("ERR syntax error")
        return num_tokens - 1 - i
    var _ci = _di
    var _count = 1
    if _ci + 2 < num_tokens and arg_eq(tokens[_ci+1].ptr, tokens[_ci+1].length, "count"):
        _count = strict_atol(tokens[_ci+2].value()); _ci += 2
    if _count <= 0:
        writer.append_error_response("ERR count should be greater than 0")
        return _ci - i

    for _ki in range(_nk):
        var _kt = i + 2 + _ki
        var _kstr = tokens[_kt].value()
        var _kv = keyspace[].get(_kstr)
        if _kv.is_none(): continue
        if _kv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return _ci - i
        var _zp = _kv.as_zset().bitcast[SlabSkipList]()
        if _zp[].length == 0: continue

        var _popc = _count if _count < _zp[].length else _zp[].length

        var _h = "*2\r\n"
        writer.append_to_response(_h.unsafe_ptr(), _h.byte_length())
        writer.append_bulk_string_response(_kstr.unsafe_ptr(), _kstr.byte_length())
        var _ah = "*" + String(_popc) + "\r\n"
        writer.append_to_response(_ah.unsafe_ptr(), _ah.byte_length())
        # gh #394: pop from the requested end, O(log n) each. This copied every
        # node out, reset() the set and re-inserted the survivors, leaked each
        # popped member, and left an emptied zset's container allocated. MAX
        # pops the highest first (the gh #238 ZPOPMAX lesson).
        var _wb = alloc[UInt8](64)
        for _ in range(_popc):
            var _r = _zp[].pop_min() if _from_min else _zp[].pop_max()
            if not _r.valid:
                break
            var _pair = "*2\r\n"
            writer.append_to_response(_pair.unsafe_ptr(), _pair.byte_length())
            writer.append_bulk_value_response(_r.obj)
            writer.append_bulk_score_response(_r.score)
            # gh #170: log the RESOLVED effect (a ZREM per popped member).
            if is_not_null(wal):
                var _wl = 0
                var _wp = gv_bytes(_r.obj, _wb, _wl)
                _ = wal[].append_kv(12, tokens[_kt].ptr, tokens[_kt].length, _wp, _wl)
            _r.obj.free_str_payload()   # ours now; the reply copied it
        _wb.free()

        # gh #234: an emptied zset is removed — after its ZREMs in the log.
        if _zp[].length == 0:
            _ = remove_and_free(keyspace, GenericValue.borrow(tokens[_kt].ptr, tokens[_kt].length))
            if is_not_null(wal):
                _ = wal[].append(2, tokens[_kt].ptr, tokens[_kt].length)
        return _ci - i

    writer.append_null_response()
    return _ci - i


@always_inline
def handle_zpopmax(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], wal: UnsafePointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """ZPOPMAX key [count] -> highest-score members with scores."""
    if i + 1 < num_tokens:
        var _zpm_key = tokens[i+1].value()
        var _zpm_cnt = 1; var _zpm_cons = 1
        # gh #393: the count was only read when it began with a digit, so
        # `ZPOPMAX k -1` popped ONE member. Redis refuses a negative count.
        if i + 2 < num_tokens:
            _zpm_cnt = strict_atol(tokens[i+2].value()); _zpm_cons = 2
        var _zpm_v = keyspace[].get(_zpm_key)
        if i + 3 < num_tokens:
            writer.append_error_response("ERR syntax error")
        elif _zpm_cnt < 0:
            writer.append_error_response("ERR value is out of range, must be positive")
        elif _zpm_v.is_none(): writer.append_empty_array_response()
        elif _zpm_v.type.value == ValueType.ZSET:
            var _zpmp = _zpm_v.as_zset().bitcast[SlabSkipList]()
            var _popc = min(_zpm_cnt, _zpmp[].length)
            if _popc <= 0:
                writer.append_empty_array_response()
            else:
                # gh #394: pop_max is O(log n) per member. This used to copy
                # every node out, reset() the set and re-insert the survivors —
                # O(n) per ZPOPMAX — and never freed a popped member.
                var _out_h = "*" + String(_popc * 2) + "\r\n"
                writer.append_to_response(_out_h.unsafe_ptr(), _out_h.byte_length())
                var _wb = alloc[UInt8](64)
                for _ in range(_popc):
                    var _r = _zpmp[].pop_max()
                    if not _r.valid:
                        break
                    writer.append_bulk_value_response(_r.obj)
                    writer.append_bulk_score_response(_r.score)
                    var _wl = 0
                    var _wp = gv_bytes(_r.obj, _wb, _wl)
                    _ = wal[].append_kv(12, tokens[i+1].ptr, tokens[i+1].length, _wp, _wl)
                    _r.obj.free_str_payload()   # ours now; the reply copied it
                _wb.free()
                # gh #234: an emptied zset is removed.
                if _zpmp[].length == 0:
                    _ = remove_and_free(keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                    if is_not_null(wal):
                        _ = wal[].append(2, tokens[i+1].ptr, tokens[i+1].length)
        else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return _zpm_cons
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zpopmax' command")
        return 0


@always_inline
def handle_zrandmember(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZRANDMEMBER key [count [WITHSCORES]] -> random member(s)."""
    if i + 1 < num_tokens:
        var _zrm_key = tokens[i+1].value()
        var _zrm_cnt = 1; var _zrm_as_arr = False; var _zrm_with = False; var _zrm_cons = 1
        # gh #238: a NEGATIVE count starts with '-' (45), which this test
        # rejected — so `ZRANDMEMBER k -3` was never seen as having a count
        # and replied with one bare bulk string instead of an array.
        if i + 2 < num_tokens:   # gh #393: always a count when present (see SRANDMEMBER)
            _zrm_cnt = rand_count(tokens[i+2].ptr, tokens[i+2].length); _zrm_as_arr = True; _zrm_cons = 2
            if i + 3 < num_tokens:
                var _op = tokens[i+3].ptr; var _ol = tokens[i+3].length
                if _ol == 10 and (_op[0]|0x20)==119: _zrm_with = True; _zrm_cons = 3
        var _zrmv = keyspace[].get(_zrm_key)
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not _zrmv.is_none() and _zrmv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif _zrmv.is_none() or _zrmv.type.value != ValueType.ZSET:
            if _zrm_as_arr: writer.append_empty_array_response()
            else: writer.append_null_response()
        else:
            var _zrmp = _zrmv.as_zset().bitcast[SlabSkipList]()
            var _zrm_len = _zrmp[].length
            var _abs_cnt = _zrm_cnt if _zrm_cnt >= 0 else -_zrm_cnt
            var _out_n = min(_abs_cnt, _zrm_len) if _zrm_cnt >= 0 else _abs_cnt
            if _zrm_as_arr:
                var _rh = "*" + String(_out_n * (2 if _zrm_with else 1)) + "\r\n"
                writer.append_to_response(_rh.unsafe_ptr(), _rh.byte_length())
                var _emitted = 0; var _rc = _zrmp[].head[].forward[0]
                # The header declares _out_n, and for a negative count _out_n can
                # EXCEED the set size — the walk must cycle back to the head or
                # the body emits fewer elements than declared and desyncs.
                while _emitted < _out_n and is_not_null(_zrmp[].head[].forward[0]):
                    if is_null(_rc): _rc = _zrmp[].head[].forward[0]
                    writer.append_bulk_value_response(_rc[].obj)
                    if _zrm_with:
                        writer.append_bulk_score_response(_rc[].score)
                    _emitted += 1; _rc = _rc[].forward[0]
            else:
                var _rc = _zrmp[].head[].forward[0]
                if is_not_null(_rc): writer.append_bulk_value_response(_rc[].obj)
                else: writer.append_null_response()
        return _zrm_cons
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrandmember' command")
        return 0


@always_inline
def handle_zmscore(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZMSCORE key member [member ...] -> list of scores."""
    if i + 2 < num_tokens:
        var _zms_key = tokens[i+1].value(); var _i = i + 1
        var _zms_v = keyspace[].get(_zms_key)
        var _zms_ni = _i + 1; var _zms_cnt = 0
        while _zms_ni < num_tokens and tokens[_zms_ni].marker != 0:
            _zms_ni += 1
            _zms_cnt += 1
        var _zms_hdr = "*" + String(_zms_cnt) + "\r\n"
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        #
        # The type check is hoisted ABOVE both the header and the per-member
        # loop. Left inside the loop it would emit N errors nested in an array
        # of N — the client reads `*N`, then an error where an element belongs,
        # and the connection desyncs.
        if not _zms_v.is_none() and _zms_v.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return _zms_ni - 1 - i
        writer.append_to_response(_zms_hdr.unsafe_ptr(), _zms_hdr.byte_length())
        for _mi in range(_i + 1, _zms_ni):
            var _mem = GenericValue.borrow(tokens[_mi].ptr, tokens[_mi].length)
            if _zms_v.is_none():
                writer.append_null_response()
            else:
                var _zsp = _zms_v.as_zset().bitcast[SlabSkipList]()
                var _zsc = _zsp[].head[].forward[0]; var _zf = False
                while is_not_null(_zsc):
                    if _zsc[].obj == _mem:
                        writer.append_bulk_score_response(_zsc[].score)
                        _zf = True; break
                    _zsc = _zsc[].forward[0]
                if not _zf: writer.append_null_response()
        return _zms_ni - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zmscore' command")
        return 0


@always_inline
def handle_zscan(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZSCAN key cursor [MATCH pattern] [COUNT count] -> cursor + members."""
    if i + 2 < num_tokens:
        var _zsv = keyspace[].get(tokens[i+1].value())
        _ = scan_cursor(tokens[i+2].ptr, tokens[i+2].length)   # gh #393: it was never read
        var _i = i + 2  # key + cursor
        var zs_pat_p = null_ptr[UInt8, MutUntrackedOrigin]()
        var zs_pat_l = 0
        while _i + 1 < num_tokens and (tokens[_i+1].length == 5 or tokens[_i+1].length == 3):
            var _op = tokens[_i+1].ptr; var _ol = tokens[_i+1].length
            if _ol == 5 and (_op[0]|0x20)==109:
                if _i + 2 < num_tokens:                        # gh #244
                    zs_pat_p = tokens[_i+2].ptr
                    zs_pat_l = tokens[_i+2].length
                _i += 2
            elif _ol == 5 and (_op[0]|0x20)==99:   # COUNT count (gh #393: validated)
                if _i + 2 >= num_tokens: raise Error("ERR syntax error")
                _ = scan_count(tokens[_i+2].ptr, tokens[_i+2].length)
                _i += 2
            else: break
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not _zsv.is_none() and _zsv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif _zsv.is_none() or _zsv.type.value != ValueType.ZSET:
            var _zs_empty = "*2\r\n$1\r\n0\r\n*0\r\n"
            writer.append_to_response(_zs_empty.unsafe_ptr(), _zs_empty.byte_length())
        else:
            var _zsp = _zsv.as_zset().bitcast[SlabSkipList]()
            # gh #244: ZSCAN returns a FLAT [member, score, member, score, ...]
            # list, exactly like HSCAN's field/value. Emitting members only made
            # the reply half the declared length's worth of pairs, so a client
            # walking it two-at-a-time read the NEXT MEMBER as the current
            # member's score. HSCAN and SSCAN were checked and are correct.
            var zs_all = zs_pat_l == 0 or _glob_all(zs_pat_p, zs_pat_l)
            var zs_mb = alloc[UInt8](24)
            var zs_n = 0
            var _zsq = _zsp[].head[].forward[0]
            while is_not_null(_zsq):
                if zs_all: zs_n += 1
                elif _glob_match(zs_pat_p, zs_pat_l, 0, _zsq[].obj.as_string_safe(zs_mb), _zsq[].obj.string_len(), 0):
                    zs_n += 1
                _zsq = _zsq[].forward[0]
            var _zs_hdr = String("*2\r\n$1\r\n0\r\n*") + String(zs_n * 2) + String("\r\n")
            writer.append_to_response(_zs_hdr.unsafe_ptr(), _zs_hdr.byte_length())
            var _zsc = _zsp[].head[].forward[0]
            while is_not_null(_zsc):
                if not zs_all:
                    if not _glob_match(zs_pat_p, zs_pat_l, 0, _zsc[].obj.as_string_safe(zs_mb), _zsc[].obj.string_len(), 0):
                        _zsc = _zsc[].forward[0]
                        continue
                writer.append_bulk_value_response(_zsc[].obj)
                writer.append_bulk_score_response(_zsc[].score)
                _zsc = _zsc[].forward[0]
            zs_mb.unsafe_free()
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zscan' command")
        return 0


@always_inline
def handle_zrangestore(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """ZRANGESTORE dst src min max -> cardinality of result."""
    if i + 4 < num_tokens:
        var _zrs_dst = tokens[i+1].value(); var _zrs_src = tokens[i+2].value()
        var _zrs_a = strict_atol(tokens[i+3].value()); var _zrs_b = strict_atol(tokens[i+4].value())
        var _i = i + 4
        var _zrs_sv = keyspace[].get(_zrs_src)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        if not _zrs_sv.is_none() and _zrs_sv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _zrs_ss = List[Float64](); var _zrs_oo = List[GenericValue]()
        if not _zrs_sv.is_none() and _zrs_sv.type.value == ValueType.ZSET:
            var _zrsp = _zrs_sv.as_zset().bitcast[SlabSkipList]()
            var _zrs_len = _zrsp[].length
            if _zrs_a < 0: _zrs_a = max(0, _zrs_len + _zrs_a)
            if _zrs_b < 0: _zrs_b = _zrs_len + _zrs_b
            if _zrs_b >= _zrs_len: _zrs_b = _zrs_len - 1
            var _idx = 0; var _zrsc = _zrsp[].head[].forward[0]
            while is_not_null(_zrsc):
                if _idx >= _zrs_a and _idx <= _zrs_b:
                    _zrs_ss.append(_zrsc[].score); _zrs_oo.append(_zrsc[].obj)
                if _idx >= _zrs_b: break
                _idx += 1; _zrsc = _zrsc[].forward[0]
        var _zrs_zp = skip_list_pool[].acquire()
        _zrs_zp.unsafe_write(SlabSkipList(16))
        for _zi in range(len(_zrs_ss)): _zrs_zp[].insert(_zrs_ss[_zi], _zrs_oo[_zi].clone())   # owned copy
        var _zrs_gv = GenericValue(); _zrs_gv.type = ValueType(ValueType.ZSET)
        _zrs_gv.set_ptr(_zrs_zp.bitcast[NoneType]())
        # Free the container this replaces (it may be one of the sources:
        # the result above holds clones, so that is safe). keyspace.set()
        # only swaps the handle — every re-run leaked the old zset.
        var _old_dst = GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length)
        _ = remove_and_free(keyspace, _old_dst)
        _old_dst.free_str_payload()
        keyspace[].set(_zrs_dst, _zrs_gv)
        writer.append_int_response(Int64(len(_zrs_ss)))
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zrangestore' command")
        return 0


@always_inline
def handle_zintercard(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZINTERCARD numkeys key [key ...] [LIMIT limit] -> cardinality of intersection."""
    if i + 2 < num_tokens:
        var _nk = strict_atol(tokens[i+1].value()); var _i = i + 2
        # numkeys must be 1..(keys actually given): it was unbounded, so
        # `ZUNION 9223372036854775807 k` looped ~2^63 times (the worker froze).
        if _nk < 1 or _nk > num_tokens - _i:
            writer.append_error_response("ERR numkeys must be at least 1 and at most the number of keys given")
            return num_tokens - 1 - i
        var _zikeys = List[String]()
        for _ki in range(_nk):
            if _i < num_tokens: _zikeys.append(tokens[_i].value())
            _i += 1
        _i -= 1
        # gh #232: every participating key is type-checked before any
        # work. Treating a wrong-type key as an empty set silently
        # changes the answer instead of reporting the error.
        for _wtk in range(len(_zikeys)):
            var _wtv = keyspace[].get(_zikeys[_wtk])
            # A SET is accepted here: Redis treats one as a zset whose members
            # all score 1, so `ZUNION 1 <set>` is legal and returns its members.
            if (not _wtv.is_none() and _wtv.type.value != ValueType.ZSET
                    and _wtv.type.value != ValueType.SET):
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return _i - i
        var _zic_lim = 0  # 0 = no limit
        if _i + 1 < num_tokens:
            var _op = tokens[_i+1].ptr; var _ol = tokens[_i+1].length
            if _ol == 5 and (_op[0]|0x20)==108: _zic_lim = strict_atol(tokens[_i+2].value()); _i += 2
        # gh #232: same accumulator as ZINTER — counts only, and a SET
        # participates. LIMIT still caps the count.
        var _zic_cnt = 0
        var _mm = List[GenericValue](); var _ss2 = List[Float64](); var _hh = List[Int]()
        var _no_w = List[Float64]()
        _zsetop_accumulate(keyspace, _zikeys, _no_w, ZAGG_SUM, _mm, _ss2, _hh)
        for _zi in range(len(_mm)):
            if _zic_lim > 0 and _zic_cnt >= _zic_lim: break
            if _hh[_zi] == len(_zikeys): _zic_cnt += 1
        writer.append_int_response(Int64(_zic_cnt))
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zintercard' command")
        return 0


@always_inline
def handle_zdiff(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """ZDIFF numkeys key [key ...] [WITHSCORES] -> difference of sorted sets."""
    if i + 2 < num_tokens:
        var _nk = strict_atol(tokens[i+1].value()); var _i = i + 2
        # numkeys must be 1..(keys actually given): it was unbounded, so
        # `ZUNION 9223372036854775807 k` looped ~2^63 times (the worker froze).
        if _nk < 1 or _nk > num_tokens - _i:
            writer.append_error_response("ERR numkeys must be at least 1 and at most the number of keys given")
            return num_tokens - 1 - i
        var _keys = List[String]()
        for _ki in range(_nk):
            if _i < num_tokens: _keys.append(tokens[_i].value())
            _i += 1
        _i -= 1
        var _with = False
        if _i + 1 < num_tokens:
            var _op = tokens[_i+1].ptr; var _ol = tokens[_i+1].length
            if _ol == 10 and (_op[0]|0x20)==119: _with = True; _i += 1
        if len(_keys) == 0:
            writer.append_empty_array_response()
        else:
            var _fv = keyspace[].get(_keys[0])
            # gh #232: a MISSING key and a key of the WRONG TYPE were answered
            # identically, with an empty/zero reply. Redis distinguishes them, and
            # the conflation runs in the dangerous direction: a caller who stored
            # the WRONG KIND of value here is told the container is empty, so the
            # bug looks like missing data instead of a type error at the call site.
            # gh #232: a SET is a legal participant (members score 1), and the
            # exclusion side must read one too — otherwise `ZDIFF 2 <zset>
            # <set>` would subtract nothing and silently return too much.
            if (not _fv.is_none() and _fv.type.value != ValueType.ZSET
                    and _fv.type.value != ValueType.SET):
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            elif _fv.is_none():
                writer.append_empty_array_response()
            else:
                var _no_w = List[Float64]()
                # First key alone: ZDIFF keeps ITS scores, never an aggregate.
                var _first = List[String](); _first.append(_keys[0])
                var _doo = List[GenericValue](); var _dss = List[Float64]()
                var _dh1 = List[Int]()
                _zsetop_accumulate(keyspace, _first, _no_w, ZAGG_SUM, _doo, _dss, _dh1)
                # Everything else, purely as an exclusion set.
                var _rest = List[String]()
                for _ki in range(1, len(_keys)): _rest.append(_keys[_ki])
                var _rmm = List[GenericValue](); var _rss = List[Float64]()
                var _rhh = List[Int]()
                _zsetop_accumulate(keyspace, _rest, _no_w, ZAGG_SUM, _rmm, _rss, _rhh)
                var _excl = alloc[SlabHashMap](1); _excl.unsafe_write(SlabHashMap(64))
                for _ri in range(len(_rmm)): _excl[].set(_rmm[_ri], GenericValue.from_int(1))
                var _sl = alloc[SlabSkipList](1); _sl.unsafe_write(SlabSkipList(16))
                var _dn_out = 0
                for _di in range(len(_doo)):
                    if _excl[].get(_doo[_di]).is_none():
                        _sl[].insert(_dss[_di], _doo[_di]); _dn_out += 1
                _excl[].forget_borrowed()   # borrowed members
                _excl.unsafe_deinit_pointee(); _excl.free()
                var _dh = "*" + String(_dn_out * (2 if _with else 1)) + "\r\n"
                writer.append_to_response(_dh.unsafe_ptr(), _dh.byte_length())
                var _c = _sl[].head[].forward[0]
                while is_not_null(_c):
                    writer.append_bulk_value_response(_c[].obj)
                    if _with: _zsetop_emit_score(writer, _c[].score)
                    _c = _c[].forward[0]
                _sl[].release_borrowed()   # temp list of BORROWED members: unmap slabs only
                _sl.unsafe_deinit_pointee(); _sl.free()
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zdiff' command")
        return 0


@always_inline
def handle_zdiffstore(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """ZDIFFSTORE dest numkeys key [key ...] -> cardinality of result."""
    if i + 3 < num_tokens:
        var _zdst_dest = tokens[i+1].value()
        var _nk = strict_atol(tokens[i+2].value()); var _i = i + 3
        # numkeys must be 1..(keys actually given): it was unbounded, so
        # `ZUNION 9223372036854775807 k` looped ~2^63 times (the worker froze).
        if _nk < 1 or _nk > num_tokens - _i:
            writer.append_error_response("ERR numkeys must be at least 1 and at most the number of keys given")
            return num_tokens - 1 - i
        var _zdkeys = List[String]()
        for _ki in range(_nk):
            if _i < num_tokens: _zdkeys.append(tokens[_i].value())
            _i += 1
        _i -= 1
        # gh #232: same two-accumulator shape as ZDIFF, so a SET participates
        # on either side and the stored scores come from the first key.
        var _zd_ss = List[Float64](); var _zd_oo = List[GenericValue]()
        if len(_zdkeys) > 0:
            var _no_w = List[Float64]()
            var _first = List[String](); _first.append(_zdkeys[0])
            var _f_oo = List[GenericValue](); var _f_ss = List[Float64]()
            var _f_hh = List[Int]()
            _zsetop_accumulate(keyspace, _first, _no_w, ZAGG_SUM, _f_oo, _f_ss, _f_hh)
            var _rest = List[String]()
            for _ki in range(1, len(_zdkeys)): _rest.append(_zdkeys[_ki])
            var _r_oo = List[GenericValue](); var _r_ss = List[Float64]()
            var _r_hh = List[Int]()
            _zsetop_accumulate(keyspace, _rest, _no_w, ZAGG_SUM, _r_oo, _r_ss, _r_hh)
            var _excl = alloc[SlabHashMap](1); _excl.unsafe_write(SlabHashMap(64))
            for _ri in range(len(_r_oo)): _excl[].set(_r_oo[_ri], GenericValue.from_int(1))
            for _di in range(len(_f_oo)):
                if _excl[].get(_f_oo[_di]).is_none():
                    _zd_ss.append(_f_ss[_di]); _zd_oo.append(_f_oo[_di])
            _excl[].forget_borrowed()   # borrowed members
            _excl.unsafe_deinit_pointee(); _excl.free()
        # Store result
        var _zdst_zp = skip_list_pool[].acquire()
        _zdst_zp.unsafe_write(SlabSkipList(16))
        for _di in range(len(_zd_ss)): _zdst_zp[].insert(_zd_ss[_di], _zd_oo[_di].clone())   # owned copy
        var _zdst_gv = GenericValue(); _zdst_gv.type = ValueType(ValueType.ZSET)
        _zdst_gv.set_ptr(_zdst_zp.bitcast[NoneType]())
        # Free the container this replaces (it may be one of the sources:
        # the result above holds clones, so that is safe). keyspace.set()
        # only swaps the handle — every re-run leaked the old zset.
        var _old_dst = GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length)
        _ = remove_and_free(keyspace, _old_dst)
        _old_dst.free_str_payload()
        keyspace[].set(_zdst_dest, _zdst_gv)
        writer.append_int_response(Int64(len(_zd_ss)))
        if len(_zd_ss) == 0: _ = remove_and_free(keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))   # gh #251
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zdiffstore' command")
        return 0


@always_inline
def handle_zremrangebylex(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], wal: UnsafePointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """ZREMRANGEBYLEX key min max -> number of removed members."""
    if i + 3 < num_tokens:
        var _zrlk = tokens[i+1].value()
        var _zrlap = tokens[i+2].ptr; var _zrlal = tokens[i+2].length
        var _zrlbp = tokens[i+3].ptr; var _zrlbl = tokens[i+3].length
        var _zrla_min = (_zrlal == 1 and _zrlap[0] == 45); var _zrla_max = (_zrlal == 1 and _zrlap[0] == 43)
        var _zrlb_max = (_zrlbl == 1 and _zrlbp[0] == 43)
        var _zrla_incl = (not _zrla_min and not _zrla_max and _zrlal > 0 and _zrlap[0] == 91)
        var _zrla_excl = (not _zrla_min and not _zrla_max and _zrlal > 0 and _zrlap[0] == 40)
        var _zrlb_incl = (not _zrlb_max and _zrlbl > 0 and _zrlbp[0] == 91)
        var _zrlb_excl = (not _zrlb_max and _zrlbl > 0 and _zrlbp[0] == 40)
        var _zrla_str = String("")
        _zrla_str += bytes_to_string(_zrlap + 1, _zrlal - 1)
        var _zrlb_str = String("")
        _zrlb_str += bytes_to_string(_zrlbp + 1, _zrlbl - 1)
        var _zrlv = keyspace[].get(_zrlk)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        if not _zrlv.is_none() and _zrlv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _zrl_cnt = 0
        if not _zrlv.is_none() and _zrlv.type.value == ValueType.ZSET:
            var _zrlp = _zrlv.as_zset().bitcast[SlabSkipList]()
            var _zrl_ss = List[Float64](); var _zrl_oo = List[GenericValue](); var _zrl_keep = List[Bool]()
            var _zrlc = _zrlp[].head[].forward[0]
            while is_not_null(_zrlc):
                var _ms = _zrlc[].obj.__str__()
                var _lo_ok = _zrla_min or (_zrla_incl and _ms >= _zrla_str) or (_zrla_excl and _ms > _zrla_str)
                var _hi_ok = _zrlb_max or (_zrlb_incl and _ms <= _zrlb_str) or (_zrlb_excl and _ms < _zrlb_str)
                _zrl_ss.append(_zrlc[].score); _zrl_oo.append(_zrlc[].obj)
                _zrl_keep.append(not (_lo_ok and _hi_ok))
                if _lo_ok and _hi_ok: _zrl_cnt += 1
                _zrlc = _zrlc[].forward[0]
            _zrlp[].reset()
            var _wb = alloc[UInt8](64)
            for _ji in range(len(_zrl_ss)):
                if _zrl_keep[_ji]: _zrlp[].insert(_zrl_ss[_ji], _zrl_oo[_ji])
                else:
                    var _wl = 0
                    var _wp = gv_bytes(_zrl_oo[_ji], _wb, _wl)
                    _ = wal[].append_kv(12, tokens[i+1].ptr, tokens[i+1].length, _wp, _wl)
            _wb.free()
        writer.append_int_response(Int64(_zrl_cnt))
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zremrangebylex' command")
        return 0


@always_inline
def handle_zremrangebyrank(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], wal: UnsafePointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """ZREMRANGEBYRANK key start stop -> number of removed members."""
    if i + 3 < num_tokens:
        var _zrrk = tokens[i+1].value()
        var _zrra = strict_atol(tokens[i+2].value()); var _zrrb = strict_atol(tokens[i+3].value())
        var _zrrv = keyspace[].get(_zrrk)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        if not _zrrv.is_none() and _zrrv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _zrr_cnt = 0
        if not _zrrv.is_none() and _zrrv.type.value == ValueType.ZSET:
            var _zrrp = _zrrv.as_zset().bitcast[SlabSkipList]()
            var _zrr_len = _zrrp[].length
            if _zrra < 0: _zrra = max(0, _zrr_len + _zrra)
            if _zrrb < 0: _zrrb = _zrr_len + _zrrb
            if _zrrb >= _zrr_len: _zrrb = _zrr_len - 1
            var _zrr_ss = List[Float64](); var _zrr_oo = List[GenericValue](); var _zrr_keep = List[Bool]()
            var _zrrc = _zrrp[].head[].forward[0]; var _idx = 0
            while is_not_null(_zrrc):
                _zrr_ss.append(_zrrc[].score); _zrr_oo.append(_zrrc[].obj)
                var _in_range = _idx >= _zrra and _idx <= _zrrb
                _zrr_keep.append(not _in_range)
                if _in_range: _zrr_cnt += 1
                _idx += 1; _zrrc = _zrrc[].forward[0]
            _zrrp[].reset()
            var _wb = alloc[UInt8](64)
            for _ji in range(len(_zrr_ss)):
                if _zrr_keep[_ji]: _zrrp[].insert(_zrr_ss[_ji], _zrr_oo[_ji])
                else:
                    var _wl = 0
                    var _wp = gv_bytes(_zrr_oo[_ji], _wb, _wl)
                    _ = wal[].append_kv(12, tokens[i+1].ptr, tokens[i+1].length, _wp, _wl)
            _wb.free()
        writer.append_int_response(Int64(_zrr_cnt))
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zremrangebyrank' command")
        return 0


@always_inline
def handle_zremrangebyscore(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin], wal: UnsafePointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """ZREMRANGEBYSCORE key min max -> number of removed members."""
    if i + 3 < num_tokens:
        var _zrbs_key = tokens[i+1].value()
        var _zrbsap = tokens[i+2].ptr; var _zrbsal = tokens[i+2].length
        var _zrbsbp = tokens[i+3].ptr; var _zrbsbl = tokens[i+3].length
        var _zrbsa_excl = False; var _zrbsb_excl = False
        var _zrbsa: Float64; var _zrbsb: Float64
        var _sb_zrbsa = _score_bound(_zrbsap, _zrbsal)   # gh #393: zslParseRange's rules
        if not _sb_zrbsa.ok:
            raise Error("ERR min or max is not a float")
        _zrbsa = _sb_zrbsa.value
        _zrbsa_excl = _sb_zrbsa.excl
        var _sb_zrbsb = _score_bound(_zrbsbp, _zrbsbl)   # gh #393: zslParseRange's rules
        if not _sb_zrbsb.ok:
            raise Error("ERR min or max is not a float")
        _zrbsb = _sb_zrbsb.value
        _zrbsb_excl = _sb_zrbsb.excl
        var _zrbsv = keyspace[].get(_zrbs_key)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        if not _zrbsv.is_none() and _zrbsv.type.value != ValueType.ZSET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _zrbs_cnt = 0
        if not _zrbsv.is_none() and _zrbsv.type.value == ValueType.ZSET:
            var _zrbsp = _zrbsv.as_zset().bitcast[SlabSkipList]()
            var _zrbs_ss = List[Float64](); var _zrbs_oo = List[GenericValue](); var _zrbs_keep = List[Bool]()
            var _zrbsc = _zrbsp[].head[].forward[0]
            while is_not_null(_zrbsc):
                var _s = _zrbsc[].score
                var _lo = (_zrbsa_excl and _s > _zrbsa) or (not _zrbsa_excl and _s >= _zrbsa)
                var _hi = (_zrbsb_excl and _s < _zrbsb) or (not _zrbsb_excl and _s <= _zrbsb)
                _zrbs_ss.append(_s); _zrbs_oo.append(_zrbsc[].obj)
                _zrbs_keep.append(not (_lo and _hi))
                if _lo and _hi: _zrbs_cnt += 1
                _zrbsc = _zrbsc[].forward[0]
            _zrbsp[].reset()
            var _wb = alloc[UInt8](64)
            for _ji in range(len(_zrbs_ss)):
                if _zrbs_keep[_ji]: _zrbsp[].insert(_zrbs_ss[_ji], _zrbs_oo[_ji])
                else:
                    var _wl = 0
                    var _wp = gv_bytes(_zrbs_oo[_ji], _wb, _wl)
                    _ = wal[].append_kv(12, tokens[i+1].ptr, tokens[i+1].length, _wp, _wl)
            _wb.free()
        writer.append_int_response(Int64(_zrbs_cnt))
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'zremrangebyscore' command")
        return 0
