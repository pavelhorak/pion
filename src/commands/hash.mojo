"""Hash commands: HMGET, HGETALL, HKEYS, HVALS, HLEN, HDEL, HEXISTS, HINCRBY, HINCRBYFLOAT, HRANDFIELD, HSCAN, HSETNX.
R3: Hash field expiration: HEXPIRE, HPEXPIRE, HEXPIREAT, HPEXPIREAT, HTTL, HPTTL, HPERSIST, HEXPIRETIME, HPEXPIRETIME."""
from src.common.container_free import remove_and_free, hash_get_live
from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy
from std.collections import Array
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.fast_path import _get_now_ns
from src.network.dispatcher import CommandDispatcher
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.utils import rand_count, strict_atol, bytes_to_string, _glob_match, _glob_all,  format_int_to_buf, format_float_to_buf, parse_filter_float, parse_float64, parse_int64_strict, is_valid_float_arg, parse_redis_double, DOUBLE_LONG, scan_cursor, scan_count
from src.memory.object_pool import ObjectPool
from src.io.wal import WAL


@always_inline
def handle_hmget(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] = null_ptr[SlabHashMap, MutUntrackedOrigin]()) raises -> Int:
    """HMGET key field [field ...] — returns number of extra tokens consumed."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        var num_fields = num_tokens - i - 2
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        #
        # The array header MUST be written after this branch, not before it.
        # Emitting `*N` and then an error puts the error INSIDE the array: the
        # client reads the header, waits for N elements, and the connection
        # desyncs. Any handler that writes a header before its type check has
        # this bug the moment the check can produce an error.
        if not val.is_none() and val.type.value != ValueType.HASH:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.HASH:
            var arr_hdr2 = String("*") + String(num_fields) + String("\r\n")
            writer.append_to_response(arr_hdr2.unsafe_ptr(), arr_hdr2.byte_length())
            for _ in range(num_fields): writer.append_null_response()
        else:
            var arr_hdr2 = String("*") + String(num_fields) + String("\r\n")
            writer.append_to_response(arr_hdr2.unsafe_ptr(), arr_hdr2.byte_length())
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            for fi in range(num_fields):
                # Expired fields were purged by hash_get_live (gh #392).
                var field_v = GenericValue.borrow(tokens[unsafe_offset=i+2+fi].ptr, tokens[unsafe_offset=i+2+fi].length)
                var fv = hash_ptr[].get(field_v)
                if fv.is_none(): writer.append_null_response()
                else: writer.append_bulk_value_response(fv)
        return num_tokens - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hmget' command")
        return 0


@always_inline
def handle_hgetall(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] = null_ptr[SlabHashMap, MutUntrackedOrigin]()) raises -> Int:
    """HGETALL key — returns number of extra tokens consumed."""
    if i + 1 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.HASH:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.HASH:
            writer.append_map_header(0)    # RESP3 `%0`, RESP2 `*0`
        else:
            # Expired fields were purged by hash_get_live above (gh #392), so
            # every field left is live: size is exact. A RESP3 map, as Redis
            # sends after HELLO 3 (#23); RESP2 gets the flat array as before.
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            writer.append_map_header(hash_ptr[].size)
            for slot in range(hash_ptr[].capacity):
                var m = hash_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    writer.append_bulk_value_response(hash_ptr[].keys[unsafe_offset=slot])
                    writer.append_bulk_value_response(hash_ptr[].values[unsafe_offset=slot])
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hgetall' command")
        return 0


@always_inline
def handle_hkeys(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """HKEYS key — returns number of extra tokens consumed."""
    if i + 1 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.HASH:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.HASH:
            writer.append_empty_array_response()
        else:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var hcount2 = 0
            for slot in range(hash_ptr[].capacity):
                var m = hash_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED: hcount2 += 1
            var hk_hdr = String("*") + String(hcount2) + String("\r\n")
            writer.append_to_response(hk_hdr.unsafe_ptr(), hk_hdr.byte_length())
            for slot in range(hash_ptr[].capacity):
                var m = hash_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    writer.append_bulk_value_response(hash_ptr[].keys[unsafe_offset=slot])
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hkeys' command")
        return 0


@always_inline
def handle_hvals(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """HVALS key — returns number of extra tokens consumed."""
    if i + 1 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.HASH:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.HASH:
            writer.append_empty_array_response()
        else:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var hcount3 = 0
            for slot in range(hash_ptr[].capacity):
                var m = hash_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED: hcount3 += 1
            var hv_hdr = String("*") + String(hcount3) + String("\r\n")
            writer.append_to_response(hv_hdr.unsafe_ptr(), hv_hdr.byte_length())
            for slot in range(hash_ptr[].capacity):
                var m = hash_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    writer.append_bulk_value_response(hash_ptr[].values[unsafe_offset=slot])
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hvals' command")
        return 0


@always_inline
def handle_hlen(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """HLEN key — returns number of extra tokens consumed."""
    if i + 1 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        if val.is_none(): writer.append_int_response(Int64(0))
        elif val.type.value != ValueType.HASH: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            writer.append_int_response(Int64(hash_ptr[].size))
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hlen' command")
        return 0


@always_inline
def handle_hdel(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """HDEL key field [field ...] — returns number of extra tokens consumed."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.HASH:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.HASH:
            _ = num_tokens - i - 2
            writer.append_int_response(Int64(0))
        else:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var hdel_count = 0
            var j_hdel = i + 2
            while j_hdel < num_tokens:
                var field_str = tokens[unsafe_offset=j_hdel].value()
                var field_v = GenericValue.borrow(tokens[unsafe_offset=j_hdel].ptr, tokens[unsafe_offset=j_hdel].length)
                if hash_ptr[].remove_generic(field_v):
                    hdel_count += 1
                    _ = wal[].append_kv(10, key_str.unsafe_ptr(), key_str.byte_length(),
                                        field_str.unsafe_ptr(), field_str.byte_length())
                j_hdel += 1
            writer.append_int_response(Int64(hdel_count))
            # gh #234: Redis removes an aggregate the moment its last element goes.
            if hash_ptr[].size == 0:
                _ = remove_and_free(keyspace, key_v)
                if is_not_null(wal):
                    _ = wal[].append(2, key_str.unsafe_ptr(), key_str.byte_length())
        return num_tokens - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hdel' command")
        return 0


@always_inline
def handle_hexists(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """HEXISTS key field — returns number of extra tokens consumed."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.HASH:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.HASH:
            writer.append_int_response(Int64(0))
        else:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var field_str = tokens[unsafe_offset=i+2].value()
            var field_v = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
            var fv2 = hash_ptr[].get(field_v)
            writer.append_int_response(Int64(0) if fv2.is_none() else Int64(1))
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hexists' command")
        return 0


@always_inline
def handle_hincrby(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut dispatcher: CommandDispatcher) raises -> Int:
    """HINCRBY key field delta — returns number of extra tokens consumed."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var field_str = tokens[unsafe_offset=i+2].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var _hdparse = parse_int64_strict(tokens[unsafe_offset=i+3].ptr, tokens[unsafe_offset=i+3].length)
        if not _hdparse.ok:
            writer.append_error_response("ERR value is not an integer or out of range")
            return 3
        var hdelta: Int64 = _hdparse.value
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.HASH:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.HASH:
            # Create hash via dispatcher (handles pool internally)
            var ibuf_hd = alloc[UInt8](32)
            var ilen_hd = format_int_to_buf(ibuf_hd, 0, hdelta)
            ibuf_hd[unsafe_offset=ilen_hd] = 0
            _ = dispatcher.execute_hset(key_str, field_str, String(ibuf_hd, ilen_hd))
            ibuf_hd.unsafe_free()
            writer.append_int_response(hdelta)
        else:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var field_v2 = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
            var cur_fv = hash_ptr[].get(field_v2)
            var old_hval: Int64 = 0
            if not cur_fv.is_none():
                if cur_fv.type.value == ValueType.INT: old_hval = cur_fv.as_int()
                elif cur_fv.is_string():
                    var flen3 = cur_fv.string_len()
                    var _sbuf = alloc[UInt8](24)
                    var fptr = cur_fv.as_string_safe(_sbuf)
                    # The STORED value needs the same strictness as the delta:
                    # `HSET h f abc; HINCRBY h f 1` used to read "abc" as 0 and
                    # answer 1, silently replacing a non-numeric field.
                    var _fparse = parse_int64_strict(fptr, flen3)
                    _sbuf.unsafe_free()
                    if not _fparse.ok:
                        writer.append_error_response("ERR hash value is not an integer")
                        return 3
                    old_hval = _fparse.value
            # gh #393: Redis refuses a sum past Int64; this WRAPPED, so
            # `HINCRBY h f 9223372036854775807` on 5 answered a large negative.
            if (hdelta < 0 and old_hval < 0 and hdelta < Int64.MIN - old_hval) or \
               (hdelta > 0 and old_hval > 0 and hdelta > Int64.MAX - old_hval):
                writer.append_error_response("ERR increment or decrement would overflow")
                return 3
            var new_hval = old_hval + hdelta
            hash_ptr[].set(field_v2, GenericValue.from_int(new_hval))
            var _hb = alloc[UInt8](32)
            var _hl = format_int_to_buf(_hb, 0, new_hval)
            _ = dispatcher.wal[].append_field_kv(5, key_str.unsafe_ptr(), key_str.byte_length(),
                                                 field_str.unsafe_ptr(), field_str.byte_length(),
                                                 _hb, _hl)
            _hb.unsafe_free()
            writer.append_int_response(new_hval)
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hincrby' command")
        return 0


@always_inline
def handle_hincrbyfloat(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut dispatcher: CommandDispatcher) raises -> Int:
    """HINCRBYFLOAT key field delta — returns number of extra tokens consumed.

    gh #180: a missing key auto-creates the hash (Redis semantics; the old code
    answered WRONGTYPE), and the accumulator is Float64 — the Float32 one lost
    precision within a few increments. Redis computes in long double, so the
    last-bit rounding of e.g. 0.1+0.2 can still differ."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var field_str = tokens[unsafe_offset=i+2].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var _hfp = parse_redis_double(tokens[unsafe_offset=i+3].ptr, tokens[unsafe_offset=i+3].length, DOUBLE_LONG)
        if not _hfp.ok:   # gh #393: string2ld's rules, as INCRBYFLOAT
            writer.append_error_response("ERR value is not a valid float")
            return 3
        var hfdelta = _hfp.value
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        if val.is_none() or val.type.value == ValueType.HASH:
            var cur_hf: Float64 = 0.0
            if not val.is_none():
                var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
                var cur_fv2 = hash_ptr[].get(GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length))
                if not cur_fv2.is_none():
                    if cur_fv2.type.value == ValueType.FLOAT: cur_hf = cur_fv2.as_float()
                    elif cur_fv2.type.value == ValueType.INT: cur_hf = Float64(cur_fv2.as_int())
                    elif cur_fv2.is_string():
                        var _sbuf = alloc[UInt8](24)
                        var fsp = cur_fv2.as_string_safe(_sbuf)
                        # gh #232: the hash twin of the INCRBYFLOAT bug. The
                        # ARGUMENT is validated above but the STORED FIELD was
                        # not, so a non-numeric field parsed as 0.0 and was then
                        # OVERWRITTEN by the delta — `HSET h f notanumber;
                        # HINCRBYFLOAT h f 1.5` replied 1.5 and destroyed the
                        # field. Redis answers `hash value is not a float` and
                        # leaves it alone.
                        var _chp = parse_redis_double(fsp, cur_fv2.string_len(), DOUBLE_LONG)   # gh #393
                        if not _chp.ok:
                            _sbuf.unsafe_free()
                            writer.append_error_response("ERR hash value is not a float")
                            return 3
                        cur_hf = _chp.value
                        _sbuf.unsafe_free()
            var new_hf = cur_hf + hfdelta
            if new_hf != new_hf or new_hf > 1.7976931348623157e308 or new_hf < -1.7976931348623157e308:
                writer.append_error_response("ERR increment would produce NaN or Infinity")
                return 3
            # Shortest round-trip repr, then Redis-style trim: "5.0" → "5".
            # The bytes are copied out of the String IMMEDIATELY: under -O3 the
            # temporary is destroyed after its last formal use, so a laundered
            # pointer into its buffer dangles (release-only garbage replies,
            # caught by test_gh179_180_181.py — dev -O0 passed by luck).
            var hfs = String(new_hf)
            var hfflen = hfs.byte_length()
            var hfp = alloc[UInt8](hfflen)
            unsafe_memcpy(dest=hfp, src=Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(hfs.unsafe_ptr())), count=hfflen)
            var hfftrim = hfflen
            var hff_has_dot = False
            var hff_has_exp = False
            for fk2 in range(hfflen):
                if hfp[unsafe_offset=fk2] == 46: hff_has_dot = True
                elif hfp[unsafe_offset=fk2] == 101: hff_has_exp = True; break
            if hff_has_dot and not hff_has_exp:
                while hfftrim > 1 and hfp[unsafe_offset=hfftrim-1] == 48:
                    hfftrim -= 1
                if hfftrim > 0 and hfp[unsafe_offset=hfftrim-1] == 46: hfftrim -= 1
            var hash_ptr: Pointer[SlabHashMap, MutUntrackedOrigin]
            if val.is_none():
                # Same create path as execute_hset (pool with heap fallback).
                if dispatcher.hash_map_pool[].head < dispatcher.hash_map_pool[].capacity:
                    hash_ptr = dispatcher.hash_map_pool[].acquire(); hash_ptr[].reset()
                else:
                    hash_ptr = alloc[SlabHashMap](1); hash_ptr.unsafe_write(SlabHashMap(16))
                var new_val = GenericValue()
                new_val.type = ValueType(ValueType.HASH)
                new_val.set_ptr(hash_ptr.unsafe_bitcast[NoneType]())
                keyspace[].set(key_v, new_val)
            else:
                hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            hash_ptr[].set(GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length),
                           GenericValue.borrow(hfp, hfftrim))
            # gh #170: effect record (cmd 5), replayed via wal_apply_aggregate
            _ = dispatcher.wal[].append_field_kv(5, key_str.unsafe_ptr(), key_str.byte_length(),
                                                 field_str.unsafe_ptr(), field_str.byte_length(),
                                                 hfp, hfftrim)
            writer.append_bulk_string_response(hfp, hfftrim)
            hfp.unsafe_free()
        else:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hincrbyfloat' command")
        return 0


@always_inline
def handle_hrandfield(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """HRANDFIELD key [count [WITHVALUES]] — returns number of extra tokens consumed."""
    if i + 1 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        var extra = 1
        var hrf_count = 1
        var hrf_withvals = False
        # gh #251: WITH and WITHOUT `count` are different REPLY SHAPES, not just
        # different lengths. `HRANDFIELD k` returns a bulk string; `HRANDFIELD k
        # 1` returns a one-element array. Pion always emitted the array, so a
        # client calling the no-count form got a list where it expected a
        # string — and on a missing key the split runs the other way: no count
        # is nil, with a count it is an empty array.
        var hrf_has_count = False
        if i + 2 < num_tokens:
            # gh #393: always a count when present (see SRANDMEMBER).
            hrf_count = rand_count(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length); extra = 2
            hrf_has_count = True
            if i + 3 < num_tokens and tokens[unsafe_offset=i+3].length >= 4 and (tokens[unsafe_offset=i+3].ptr[unsafe_offset=0]|0x20)==119:
                hrf_withvals = True; extra = 3
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.HASH:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.HASH:
            if hrf_has_count: writer.append_empty_array_response()
            else: writer.append_null_response()
        elif not hrf_has_count:
            # No count: a single field as a BULK STRING, no array wrapper.
            var _hp = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var _found = False
            for slot in range(_hp[].capacity):
                var _m = _hp[].metadata[unsafe_offset=slot]
                if _m != SlabHashMap.EMPTY and _m != SlabHashMap.DELETED:
                    writer.append_bulk_value_response(_hp[].keys[unsafe_offset=slot])
                    _found = True
                    break
            # An empty hash should not exist (gh #234 removes it), but a husk
            # from any path that still leaks one must not desync the reply.
            if not _found: writer.append_null_response()
        else:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var hrf_abs = hrf_count if hrf_count >= 0 else -hrf_count
            # gh #238: a NEGATIVE count returns EXACTLY |count| fields with
            # repeats allowed; only a POSITIVE count is capped at the hash size.
            var out_n = hrf_abs
            if hrf_count >= 0 and hrf_abs > hash_ptr[].size: out_n = hash_ptr[].size
            if hrf_count < 0 and hash_ptr[].size == 0: out_n = 0
            var arr_hdr3 = String("*") + String(out_n * (2 if hrf_withvals else 1)) + String("\r\n")
            writer.append_to_response(arr_hdr3.unsafe_ptr(), arr_hdr3.byte_length())
            var _hlive = List[Int]()
            for slot in range(hash_ptr[].capacity):
                var m0 = hash_ptr[].metadata[unsafe_offset=slot]
                if m0 != SlabHashMap.EMPTY and m0 != SlabHashMap.DELETED:
                    _hlive.append(slot)
                    if hrf_count >= 0 and len(_hlive) >= out_n: break
            var emitted = 0
            # Cycles for the negative form: the header already declared out_n,
            # so emitting fewer would desync the connection.
            while emitted < out_n and len(_hlive) > 0:
                var _sl = _hlive[emitted % len(_hlive)]
                writer.append_bulk_value_response(hash_ptr[].keys[unsafe_offset=_sl])
                if hrf_withvals: writer.append_bulk_value_response(hash_ptr[].values[unsafe_offset=_sl])
                emitted += 1
        return extra
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hrandfield' command")
        return 0


@always_inline
def handle_hscan(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """HSCAN key cursor [MATCH pattern] [COUNT count] — returns number of extra tokens consumed."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var hs_cursor = scan_cursor(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        var extra = 2
        var hs_pat_p = null_ptr[UInt8, MutUntrackedOrigin]()
        var hs_pat_l = 0
        # Skip optional args
        var scan_i = i + 2
        while scan_i + 1 < num_tokens:
            var nxt2 = tokens[unsafe_offset=scan_i+1]; var nxtp2 = nxt2.ptr; var nxtl2 = nxt2.length
            if nxtl2 == 5 and (nxtp2[unsafe_offset=0]|0x20)==109:
                # gh #244: MATCH was skipped, not applied
                if scan_i + 2 < num_tokens:
                    hs_pat_p = tokens[unsafe_offset=scan_i+2].ptr
                    hs_pat_l = tokens[unsafe_offset=scan_i+2].length
                scan_i += 2; extra += 2
            elif nxtl2 == 5 and (nxtp2[unsafe_offset=0]|0x20)==99 and (nxtp2[unsafe_offset=1]|0x20)==111:
                if scan_i + 2 >= num_tokens: raise Error("ERR syntax error")
                _ = scan_count(tokens[unsafe_offset=scan_i+2].ptr, tokens[unsafe_offset=scan_i+2].length)
                scan_i += 2; extra += 2
            else: break
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        if val.is_none() or val.type.value != ValueType.HASH or hs_cursor != 0:
            var hs_empty = "*2\r\n$1\r\n0\r\n*0\r\n"
            writer.append_to_response(hs_empty.unsafe_ptr(), hs_empty.byte_length())
        else:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var hs_all = hs_pat_l == 0 or _glob_all(hs_pat_p, hs_pat_l)
            var hs_mb = alloc[UInt8](24)
            var hs_count2 = 0
            for slot in range(hash_ptr[].capacity):
                var m = hash_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    if hs_all: hs_count2 += 1
                    else:
                        var fk = hash_ptr[].keys[unsafe_offset=slot]
                        if _glob_match(hs_pat_p, hs_pat_l, 0, fk.as_string_safe(hs_mb), fk.string_len(), 0):
                            hs_count2 += 1
            var hs_hdr = String("*2\r\n$1\r\n0\r\n*") + String(hs_count2 * 2) + String("\r\n")
            writer.append_to_response(hs_hdr.unsafe_ptr(), hs_hdr.byte_length())
            for slot in range(hash_ptr[].capacity):
                var m = hash_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    if not hs_all:
                        var fk2 = hash_ptr[].keys[unsafe_offset=slot]
                        if not _glob_match(hs_pat_p, hs_pat_l, 0, fk2.as_string_safe(hs_mb), fk2.string_len(), 0):
                            continue
                    writer.append_bulk_value_response(hash_ptr[].keys[unsafe_offset=slot])
                    writer.append_bulk_value_response(hash_ptr[].values[unsafe_offset=slot])
            hs_mb.unsafe_free()
        return extra
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hscan' command")
        return 0


@always_inline
def handle_hsetnx(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut dispatcher: CommandDispatcher) raises -> Int:
    """HSETNX key field value — returns number of extra tokens consumed."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var field_str = tokens[unsafe_offset=i+2].value()
        var val_str3 = tokens[unsafe_offset=i+3].raw_value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = hash_get_live(keyspace, key_v)   # gh #392: expired fields never show
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.HASH:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.HASH:
            # Create hash via dispatcher; field is new so always sets
            _ = dispatcher.execute_hset(key_str, field_str, val_str3)
            writer.append_int_response(Int64(1))
        else:
            var hash_ptr2 = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var field_v4 = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
            var existing_fv = hash_ptr2[].get(field_v4)
            if existing_fv.is_none():
                hash_ptr2[].set(field_v4, GenericValue.from_string(val_str3))
                _ = dispatcher.wal[].append_field_kv(5, key_str.unsafe_ptr(), key_str.byte_length(),
                                                     field_str.unsafe_ptr(), field_str.byte_length(),
                                                     val_str3.unsafe_ptr(), val_str3.byte_length())
                writer.append_int_response(Int64(1))
            else: writer.append_int_response(Int64(0))
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'hsetnx' command")
        return 0


# ── Hash field expiration (Redis 7.4) — gh #392 ────────────────────────────────
# HEXPIRE key seconds [NX|XX|GT|LT] FIELDS numfields field ...  (and HPEXPIRE,
# HEXPIREAT, HPEXPIREAT, HTTL, HPTTL, HEXPIRETIME, HPEXPIRETIME, HPERSIST).
#
# Each hash keeps its own field → deadline map (SlabHashMap.field_ttl). These
# used to be entries in the GLOBAL TTL table under `key + "::" + field`, which
# was ambiguous (key `a::b` field `c` and key `a` field `b::c` shared one
# entry), outlived DEL, did not follow RENAME/COPY, and was never written to
# the WAL or the snapshot — every field TTL vanished on restart and never
# reached a replica (checked on the pre-fix binary: HTTL -1 after a restart,
# and no `key::field` bytes in its log or snapshot, so there is no old record
# format to read back). Now the deadlines go wherever the hash goes, and
# cmd 32/33 records carry them.
#
# Reply codes (probed against Redis 8.10): HEXPIRE* -2 no field / 0 condition
# not met / 1 set / 2 deleted (deadline already past); HTTL* -2 / -1 no TTL /
# time; HPERSIST -2 / -1 / 1. A missing key answers -2 for every field.
# HSET/HMSET clear an overwritten field's TTL; HINCRBY/HINCRBYFLOAT/HSETNX
# keep it (also probed).


@always_inline
def _parse_fields_offset(tokens: Pointer[RESP3Token, MutUntrackedOrigin], start: Int, num_tokens: Int) -> Int:
    """Find FIELDS keyword and return the index of numfields token. Returns -1 if not found."""
    if start < num_tokens:
        var fp = tokens[unsafe_offset=start].ptr; var fl = tokens[unsafe_offset=start].length
        # FIELDS = 6 bytes: f=102, i=105, e=101, l=108, d=100, s=115
        if fl == 6 and (fp[unsafe_offset=0]|0x20)==102 and (fp[unsafe_offset=1]|0x20)==105 and (fp[unsafe_offset=2]|0x20)==101 and (fp[unsafe_offset=3]|0x20)==108 and (fp[unsafe_offset=4]|0x20)==100 and (fp[unsafe_offset=5]|0x20)==115:
            return start + 1  # index of numfields token
    return -1


def _fields_clause(tokens: Pointer[RESP3Token, MutUntrackedOrigin], at: Int, num_tokens: Int,
                   mut writer: ResponseWriter, mut field_start: Int) -> Int:
    """Parse `FIELDS numfields f1 .. fn` at token `at`. Returns numfields, or
    -1 after writing the error. numfields must be >= 1 and EXACTLY the fields
    given (Redis 7.4): it was once trusted, and a huge value looped ~2^63 times
    (the worker froze) while a too-large one wrote a `*N` header the body never
    filled."""
    var nf_idx = _parse_fields_offset(tokens, at, num_tokens)
    if nf_idx < 0 or nf_idx >= num_tokens:
        writer.append_error_response("ERR syntax error: expected FIELDS numfields field1 ...")
        return -1
    var nf = parse_int64_strict(tokens[unsafe_offset=nf_idx].ptr, tokens[unsafe_offset=nf_idx].length)
    if not nf.ok or nf.value < 1:
        writer.append_error_response("ERR Parameter `numFields` should be greater than 0")
        return -1
    field_start = nf_idx + 1
    if Int(nf.value) != num_tokens - field_start:
        writer.append_error_response("ERR The `numfields` parameter must match the number of arguments")
        return -1
    return Int(nf.value)


def _array_header(mut writer: ResponseWriter, n: Int):
    var h = String("*") + String(n) + String("\r\n")   # named: a temporary's unsafe_ptr can dangle at -O3
    writer.append_to_response(h.unsafe_ptr().unsafe_bitcast[UInt8](), h.byte_length())


def _hexpire_generic(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                     mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                     wal: Pointer[WAL, MutUntrackedOrigin], unit_ns: Int64, absolute: Bool,
                     name: StaticString) raises -> Int:
    """HEXPIRE / HPEXPIRE / HEXPIREAT / HPEXPIREAT. `unit_ns` is 1e9 or 1e6;
    `absolute` makes the argument a unix time instead of a duration."""
    if i + 5 >= num_tokens:
        writer.append_error_response(String("ERR wrong number of arguments for '") + String(name) + "' command")
        return num_tokens - 1 - i
    var tp = parse_int64_strict(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
    if not tp.ok:
        writer.append_error_response("ERR value is not an integer or out of range")
        return num_tokens - 1 - i
    if tp.value < 0:
        writer.append_error_response("ERR invalid expire time, must be >= 0")
        return num_tokens - 1 - i
    var now = _get_now_ns()
    # Overflow-safe deadline: an argument past what Int64 ns can hold is an error.
    var limit = Int64(9_223_372_036_854_775_807)
    if tp.value > (limit - (0 if absolute else now)) // unit_ns:
        writer.append_error_response(String("ERR invalid expire time in '") + String(name) + "' command")
        return num_tokens - 1 - i
    var deadline = tp.value * unit_ns if absolute else now + tp.value * unit_ns
    # NX | XX | GT | LT sits between the time and FIELDS (gh #232: it was not
    # parsed at all, so the whole conditional surface answered a syntax error).
    var cond = 0
    var at = i + 3
    var cp = tokens[unsafe_offset=at].ptr
    var cl = tokens[unsafe_offset=at].length
    if cl == 2:
        var c0 = cp[unsafe_offset=0] | 0x20
        var c1 = cp[unsafe_offset=1] | 0x20
        if c0 == 110 and c1 == 120: cond = 1        # NX
        elif c0 == 120 and c1 == 120: cond = 2      # XX
        elif c0 == 103 and c1 == 116: cond = 3      # GT
        elif c0 == 108 and c1 == 116: cond = 4      # LT
        if cond != 0: at += 1
    var field_start = 0
    var numfields = _fields_clause(tokens, at, num_tokens, writer, field_start)
    if numfields < 0:
        return num_tokens - 1 - i
    var key = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
    var hv = hash_get_live(keyspace, key)
    # gh #232: WRONGTYPE before the header — an error inside the array desyncs.
    if not hv.is_none() and hv.type.value != ValueType.HASH:
        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return num_tokens - 1 - i
    _array_header(writer, numfields)
    if hv.is_none():
        for _ in range(numfields): writer.append_int_response(Int64(-2))
        return num_tokens - 1 - i
    var hp = hv.as_hash().unsafe_bitcast[SlabHashMap]()
    var kt = tokens[unsafe_offset=i+1]
    for fi in range(numfields):
        var ft = tokens[unsafe_offset=field_start + fi]
        var field = GenericValue.borrow(ft.ptr, ft.length)
        if hp[].get(field).is_none():
            writer.append_int_response(Int64(-2))
            continue
        var cur = hp[].field_deadline(field)
        var has = cur != 0
        # No TTL is infinity: GT can never beat it, LT always does.
        var ok = True
        if cond == 1: ok = not has
        elif cond == 2: ok = has
        elif cond == 3: ok = has and deadline > cur
        elif cond == 4: ok = (not has) or deadline < cur
        if not ok:
            writer.append_int_response(Int64(0))
        elif deadline <= now:
            # A deadline already past deletes the field and answers 2.
            _ = hp[].remove_generic(field)
            if is_not_null(wal):
                _ = wal[].append_kv(10, kt.ptr, kt.length, ft.ptr, ft.length)   # HDEL
            writer.append_int_response(Int64(2))
        else:
            hp[].set_field_deadline(field, deadline)
            keyspace[].note_field_ttl(key)
            if is_not_null(wal):
                _ = wal[].append_field_deadline(kt.ptr, kt.length, ft.ptr, ft.length, deadline)
            writer.append_int_response(Int64(1))
    if hp[].size == 0:   # the last field was deleted: so is the key
        _ = remove_and_free(keyspace, key)
        if is_not_null(wal):
            _ = wal[].append(2, kt.ptr, kt.length)
    return num_tokens - 1 - i


def _httl_generic(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                  mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                  mode: Int, name: StaticString) raises -> Int:
    """HTTL (mode 0, seconds left) / HPTTL (1, ms left) / HEXPIRETIME (2, unix
    s) / HPEXPIRETIME (3, unix ms)."""
    if i + 4 >= num_tokens:
        writer.append_error_response(String("ERR wrong number of arguments for '") + String(name) + "' command")
        return num_tokens - 1 - i
    var field_start = 0
    var numfields = _fields_clause(tokens, i + 2, num_tokens, writer, field_start)
    if numfields < 0:
        return num_tokens - 1 - i
    var key = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
    var hv = hash_get_live(keyspace, key)
    if not hv.is_none() and hv.type.value != ValueType.HASH:
        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return num_tokens - 1 - i
    _array_header(writer, numfields)
    if hv.is_none():
        for _ in range(numfields): writer.append_int_response(Int64(-2))
        return num_tokens - 1 - i
    var hp = hv.as_hash().unsafe_bitcast[SlabHashMap]()
    var now = _get_now_ns()
    for fi in range(numfields):
        var ft = tokens[unsafe_offset=field_start + fi]
        var field = GenericValue.borrow(ft.ptr, ft.length)
        if hp[].get(field).is_none():
            writer.append_int_response(Int64(-2))
            continue
        var d = hp[].field_deadline(field)
        if d == 0:
            writer.append_int_response(Int64(-1))
        elif mode == 0:
            # gh #232: Redis ROUNDS UP — truncating made `HEXPIRE k 100` then
            # `HTTL k` answer 99, since some microseconds have always passed.
            writer.append_int_response((d - now + Int64(999_999_999)) // Int64(1_000_000_000))
        elif mode == 1:
            writer.append_int_response((d - now + Int64(999_999)) // Int64(1_000_000))
        elif mode == 2:
            # Round UP, as the differential shows Redis does: the second the
            # field survives to (gh #232 — floor answered one second early).
            writer.append_int_response((d + Int64(999_999_999)) // Int64(1_000_000_000))
        else:
            writer.append_int_response(d // Int64(1_000_000))
    return num_tokens - 1 - i


@always_inline
def handle_hexpire(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    return _hexpire_generic(tokens, i, num_tokens, writer, keyspace, wal, 1_000_000_000, False, "hexpire")


@always_inline
def handle_hpexpire(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    return _hexpire_generic(tokens, i, num_tokens, writer, keyspace, wal, 1_000_000, False, "hpexpire")


@always_inline
def handle_hexpireat(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    return _hexpire_generic(tokens, i, num_tokens, writer, keyspace, wal, 1_000_000_000, True, "hexpireat")


@always_inline
def handle_hpexpireat(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    return _hexpire_generic(tokens, i, num_tokens, writer, keyspace, wal, 1_000_000, True, "hpexpireat")


@always_inline
def handle_httl(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    return _httl_generic(tokens, i, num_tokens, writer, keyspace, 0, "httl")


@always_inline
def handle_hpttl(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    return _httl_generic(tokens, i, num_tokens, writer, keyspace, 1, "hpttl")


@always_inline
def handle_hexpiretime(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    return _httl_generic(tokens, i, num_tokens, writer, keyspace, 2, "hexpiretime")


@always_inline
def handle_hpexpiretime(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    return _httl_generic(tokens, i, num_tokens, writer, keyspace, 3, "hpexpiretime")


@always_inline
def handle_hpersist(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """HPERSIST key FIELDS numfields field ... → -2 no field / -1 no TTL / 1 removed."""
    if i + 4 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'hpersist' command")
        return num_tokens - 1 - i
    var field_start = 0
    var numfields = _fields_clause(tokens, i + 2, num_tokens, writer, field_start)
    if numfields < 0:
        return num_tokens - 1 - i
    var key = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
    var hv = hash_get_live(keyspace, key)
    if not hv.is_none() and hv.type.value != ValueType.HASH:
        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return num_tokens - 1 - i
    _array_header(writer, numfields)
    if hv.is_none():
        for _ in range(numfields): writer.append_int_response(Int64(-2))
        return num_tokens - 1 - i
    var hp = hv.as_hash().unsafe_bitcast[SlabHashMap]()
    var kt = tokens[unsafe_offset=i+1]
    for fi in range(numfields):
        var ft = tokens[unsafe_offset=field_start + fi]
        var field = GenericValue.borrow(ft.ptr, ft.length)
        if hp[].get(field).is_none():
            writer.append_int_response(Int64(-2))
        elif hp[].clear_field_deadline(field):
            if is_not_null(wal):
                _ = wal[].append_kv(33, kt.ptr, kt.length, ft.ptr, ft.length)
            writer.append_int_response(Int64(1))
        else:
            # gh #232: "exists, no TTL" is -1, never 0.
            writer.append_int_response(Int64(-1))
    return num_tokens - 1 - i
