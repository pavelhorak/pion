"""Set commands: SCARD, SMEMBERS, SREM, SMOVE, SINTER, SINTERCARD, SUNION, SDIFF, *STORE, SSCAN, SRANDMEMBER, SISMEMBER, SMISMEMBER."""
from src.common.container_free import remove_and_free
from src.common.ptr import is_not_null, null_ptr
from src.commands.scan_opts import parse_scan_opts, scan_no_opts
from src.common.utils import rand_count, strict_atol, _glob_match, _glob_all, scan_cursor, scan_count, arg_eq, parse_int64_strict
from std.memory.unsafe_pointer import Pointer
from std.collections import Array
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.io.wal import WAL


@always_inline
def handle_scard(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SCARD key → integer reply with cardinality of set."""
    if i + 1 < num_tokens:
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        # gh #232: missing vs wrong-type, same split as handle_sismember below.
        if not val.is_none() and val.type.value != ValueType.SET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none(): writer.append_int_response(Int64(0))
        else: writer.append_int_response(Int64(val.as_set().unsafe_bitcast[SlabHashMap]()[].size))
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'scard' command")
        return 0


@always_inline
def handle_sismember(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SISMEMBER key member → 1 if member exists, 0 otherwise."""
    if i + 2 < num_tokens:
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.SET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.SET:
            writer.append_int_response(Int64(0))
        else:
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            var mem_v = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
            writer.append_int_response(Int64(0) if set_ptr[].get(mem_v).is_none() else Int64(1))
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'sismember' command")
        return 0


@always_inline
def handle_smismember(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SMISMEMBER key member [member ...] → array of 0/1 per member."""
    if i + 2 < num_tokens:
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var num_mems = num_tokens - i - 2
        var smis_hdr = String("*") + String(num_mems) + String("\r\n")
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        #
        # The header goes INSIDE the non-error branches. Emitting `*N` first
        # and then an error puts the error inside the array: the client reads
        # the header, waits for N elements, and the connection desyncs.
        if not val.is_none() and val.type.value != ValueType.SET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.SET:
            writer.append_to_response(smis_hdr.unsafe_ptr(), smis_hdr.byte_length())
            for _ in range(num_mems): writer.append_int_response(Int64(0))
        else:
            writer.append_to_response(smis_hdr.unsafe_ptr(), smis_hdr.byte_length())
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            for mi in range(num_mems):
                var mem_v = GenericValue.borrow(tokens[unsafe_offset=i+2+mi].ptr, tokens[unsafe_offset=i+2+mi].length)
                writer.append_int_response(Int64(0) if set_ptr[].get(mem_v).is_none() else Int64(1))
        return num_tokens - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'smismember' command")
        return 0


@always_inline
def handle_smembers(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SMEMBERS key → array of all members in the set."""
    if i + 1 < num_tokens:
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.SET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.SET:
            writer.append_set_header(0)
        else:
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            writer.append_set_header(set_ptr[].size)   # RESP3 `~`, as Redis
            for slot in range(set_ptr[].capacity):
                var m = set_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    writer.append_bulk_value_response(set_ptr[].keys[unsafe_offset=slot])
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'smembers' command")
        return 0


@always_inline
def handle_srandmember(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SRANDMEMBER key [count] → bulk string or array of random members."""
    if i + 1 < num_tokens:
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var consumed = 1
        var srm_count = 1; var srm_as_array = False
        if i + 2 < num_tokens:
            # gh #393: a count argument is ALWAYS a count. A first-byte sniff
            # decided whether it was one, so `SRANDMEMBER k abc` (and "+1",
            # " 1", "") answered as if no count had been given.
            srm_count = rand_count(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
            srm_as_array = True; consumed = 2
            if num_tokens - i > 3:      # Redis: a count, and nothing after it
                writer.append_error_response("ERR syntax error")
                return num_tokens - 1 - i
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.SET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.SET:
            if srm_as_array: writer.append_empty_array_response()
            else: writer.append_null_response()
        else:
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            var srm_abs = srm_count if srm_count >= 0 else -srm_count
            # gh #238: a NEGATIVE count means "repeats allowed, return EXACTLY
            # |count| elements"; only a POSITIVE count is capped at the set
            # size. Clamping both ways made `SRANDMEMBER k -3` on a 1-member set
            # return 1 element where Redis returns 3.
            var out_n = srm_abs
            if srm_count >= 0 and srm_abs > set_ptr[].size: out_n = set_ptr[].size
            if srm_count < 0 and set_ptr[].size == 0: out_n = 0
            if srm_as_array:
                var srm_hdr = String("*") + String(out_n) + String("\r\n")
                writer.append_to_response(srm_hdr.unsafe_ptr(), srm_hdr.byte_length())
                var _live = List[Int]()
                for slot in range(set_ptr[].capacity):
                    var m0 = set_ptr[].metadata[unsafe_offset=slot]
                    if m0 != SlabHashMap.EMPTY and m0 != SlabHashMap.DELETED:
                        _live.append(slot)
                        if srm_count >= 0 and len(_live) >= out_n: break
                var emitted2 = 0
                while emitted2 < out_n and len(_live) > 0:
                    # cycles for the negative form, walks once for the positive
                    writer.append_bulk_value_response(
                        set_ptr[].keys[unsafe_offset=_live[emitted2 % len(_live)]])
                    emitted2 += 1
            else:
                var found_srm = False
                for slot in range(set_ptr[].capacity):
                    var m = set_ptr[].metadata[unsafe_offset=slot]
                    if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                        writer.append_bulk_value_response(set_ptr[].keys[unsafe_offset=slot])
                        found_srm = True; break
                if not found_srm: writer.append_null_response()
        return consumed
    else:
        writer.append_error_response("ERR wrong number of arguments for 'srandmember' command")
        return 0


@always_inline
def handle_srem(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """SREM key member [member ...] → integer count of removed members."""
    if i + 2 < num_tokens:
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.SET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return num_tokens - 1 - i
        elif val.is_none() or val.type.value != ValueType.SET:
            writer.append_int_response(Int64(0))
            return num_tokens - 1 - i
        else:
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            var srem_cnt = 0; var j_srem = i + 2
            while j_srem < num_tokens:
                var mem_v = GenericValue.borrow(tokens[unsafe_offset=j_srem].ptr, tokens[unsafe_offset=j_srem].length)
                if set_ptr[].remove_generic(mem_v):
                    srem_cnt += 1
                    _ = wal[].append_kv(11, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length,
                                        tokens[unsafe_offset=j_srem].ptr, tokens[unsafe_offset=j_srem].length)
                j_srem += 1
            writer.append_int_response(Int64(srem_cnt))
            # gh #234: Redis removes an aggregate the moment its last element goes.
            if set_ptr[].size == 0:
                _ = remove_and_free(keyspace, key_v)
                if is_not_null(wal):
                    _ = wal[].append(2, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
            return num_tokens - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'srem' command")
        return 0


@always_inline
def handle_smove(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """SMOVE source destination member → 1 if moved, 0 otherwise."""
    if i + 3 < num_tokens:
        var src_vs = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var dst_vs = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        var mem_vs = GenericValue.borrow(tokens[unsafe_offset=i+3].ptr, tokens[unsafe_offset=i+3].length)
        var src_val = keyspace[].get(src_vs)
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        # Redis's precedence, verified against a live server: a wrong-type
        # SOURCE errors; a MISSING source is 0 and the destination is never
        # examined; a wrong-type DESTINATION then errors BEFORE the member is
        # looked up (`SMOVE set wrongtype nosuchmember` is WRONGTYPE, not 0).
        #
        # gh #232: the destination check used to sit AFTER `remove_generic` and
        # after BOTH wal appends, so `SMOVE src wrongtype-dst m` answered
        # WRONGTYPE *and* deleted the member — gone from the source, never added
        # anywhere, and the WAL replayed the same loss. Same shape as the LMOVE
        # bug: a command that refuses must be a no-op.
        var dst_val_pre = keyspace[].get(dst_vs)
        if not src_val.is_none() and src_val.type.value != ValueType.SET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif src_val.is_none():
            writer.append_int_response(Int64(0))
        elif not dst_val_pre.is_none() and dst_val_pre.type.value != ValueType.SET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif tokens[unsafe_offset=i+1].length == tokens[unsafe_offset=i+2].length \
                and src_val._data0 == dst_val_pre._data0:
            # Same key: Redis answers membership and changes nothing.
            var same_set = src_val.as_set().unsafe_bitcast[SlabHashMap]()
            writer.append_int_response(Int64(0) if same_set[].get(mem_vs).is_none() else Int64(1))
        else:
            var src_set = src_val.as_set().unsafe_bitcast[SlabHashMap]()
            if src_set[].get(mem_vs).is_none():
                writer.append_int_response(Int64(0))
            else:
                _ = src_set[].remove_generic(mem_vs)
                _ = wal[].append_kv(11, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length,
                                    tokens[unsafe_offset=i+3].ptr, tokens[unsafe_offset=i+3].length)
                _ = wal[].append_kv(8, tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length,
                                    tokens[unsafe_offset=i+3].ptr, tokens[unsafe_offset=i+3].length)
                var dst_val = keyspace[].get(dst_vs)
                if dst_val.is_none() or dst_val.type.value != ValueType.SET:
                    var ns = alloc[SlabHashMap](1); ns.unsafe_write(SlabHashMap(16))
                    ns[].set(mem_vs, GenericValue.from_int(1))
                    var sv = GenericValue(); sv.type = ValueType(ValueType.SET)
                    sv.set_ptr(ns.unsafe_bitcast[NoneType]()); keyspace[].set(dst_vs, sv)
                else:
                    dst_val.as_set().unsafe_bitcast[SlabHashMap]()[].set(mem_vs, GenericValue.from_int(1))
                writer.append_int_response(Int64(1))
                # gh #234: the source goes with its last member (its TTL too).
                if src_set[].size == 0:
                    _ = remove_and_free(keyspace, src_vs)
                    _ = wal[].append(2, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'smove' command")
        return 0


@always_inline
@always_inline
def _any_not_set(tokens: Pointer[RESP3Token, MutUntrackedOrigin], start: Int, end: Int,
                 keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Bool:
    """True when a key in tokens[start:end] exists and is not a set."""
    for j in range(start, end):
        var v = keyspace[].get(GenericValue.borrow(tokens[unsafe_offset=j].ptr, tokens[unsafe_offset=j].length))
        if not v.is_none() and v.type.value != ValueType.SET:
            return True
    return False


def handle_sinter(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SINTER key [key ...] → array of members in intersection of all sets."""
    if i + 1 < num_tokens:
        var first_si_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var first_si = keyspace[].get(first_si_v)
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        # Every key is checked, as Redis checks them: a wrong-type key after the
        # first used to read as an empty set (and a missing first key hid it).
        if _any_not_set(tokens, i + 1, num_tokens, keyspace):
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif first_si.is_none() or first_si.type.value != ValueType.SET:
            writer.append_set_header(0)
        else:
            var fsi_ptr = first_si.as_set().unsafe_bitcast[SlabHashMap]()
            var sinter_cnt = 0
            for slot in range(fsi_ptr[].capacity):
                var m = fsi_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    var k = fsi_ptr[].keys[unsafe_offset=slot]; var in_all = True
                    var j_si = i + 2
                    while j_si < num_tokens:
                        var ov = GenericValue.borrow(tokens[unsafe_offset=j_si].ptr, tokens[unsafe_offset=j_si].length)
                        var oval = keyspace[].get(ov)
                        if oval.is_none() or oval.type.value != ValueType.SET: in_all = False; break
                        if oval.as_set().unsafe_bitcast[SlabHashMap]()[].get(k).is_none(): in_all = False; break
                        j_si += 1
                    if in_all: sinter_cnt += 1
            writer.append_set_header(sinter_cnt)
            for slot in range(fsi_ptr[].capacity):
                var m = fsi_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    var k = fsi_ptr[].keys[unsafe_offset=slot]; var in_all2 = True
                    var j_si2 = i + 2
                    while j_si2 < num_tokens:
                        var ov2 = GenericValue.borrow(tokens[unsafe_offset=j_si2].ptr, tokens[unsafe_offset=j_si2].length)
                        var oval2 = keyspace[].get(ov2)
                        if oval2.is_none() or oval2.type.value != ValueType.SET: in_all2 = False; break
                        if oval2.as_set().unsafe_bitcast[SlabHashMap]()[].get(k).is_none(): in_all2 = False; break
                        j_si2 += 1
                    if in_all2: writer.append_bulk_value_response(k)
        return num_tokens - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'sinter' command")
        return 0


@always_inline
def handle_sinterstore(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SINTERSTORE destination key [key ...] → integer count of members in resulting set."""
    if i + 2 < num_tokens:
        var dst_sis_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var first_sis_v = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        var first_sis = keyspace[].get(first_sis_v)
        var res_sis = alloc[SlabHashMap](1); res_sis.unsafe_write(SlabHashMap(16))
        if not first_sis.is_none() and first_sis.type.value == ValueType.SET:
            var fsis_ptr = first_sis.as_set().unsafe_bitcast[SlabHashMap]()
            for slot in range(fsis_ptr[].capacity):
                var m = fsis_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    var k = fsis_ptr[].keys[unsafe_offset=slot]; var in_all3 = True
                    var j_sis = i + 3
                    while j_sis < num_tokens:
                        var ov3 = GenericValue.borrow(tokens[unsafe_offset=j_sis].ptr, tokens[unsafe_offset=j_sis].length)
                        var oval3 = keyspace[].get(ov3)
                        if oval3.is_none() or oval3.type.value != ValueType.SET: in_all3 = False; break
                        if oval3.as_set().unsafe_bitcast[SlabHashMap]()[].get(k).is_none(): in_all3 = False; break
                        j_sis += 1
                    if in_all3: res_sis[].set(k.clone(), GenericValue.from_int(1))   # owned copy
        var sis_cnt = res_sis[].size
        var sis_sv = GenericValue(); sis_sv.type = ValueType(ValueType.SET)
        sis_sv.set_ptr(res_sis.unsafe_bitcast[NoneType]()); _ = remove_and_free(keyspace, dst_sis_v)   # free the container this replaces
        keyspace[].set(dst_sis_v, sis_sv)
        writer.append_int_response(Int64(sis_cnt))
        # gh #251: an empty result DELETES the destination — gh #234's rule
        # ("an emptied aggregate must be removed") reaching store destinations,
        # which that pass did not cover. Store-then-remove rather than skipping
        # the store: the destination must be overwritten even when the result
        # is empty (it may have held another type), and this reuses the same
        # free path every other deletion takes.
        if sis_cnt == 0: _ = remove_and_free(keyspace, dst_sis_v)
        return num_tokens - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'sinterstore' command")
        return 0


@always_inline
def handle_sintercard(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SINTERCARD numkeys key [key ...] [LIMIT limit] → integer count of intersection."""
    if i + 2 < num_tokens:
        var nk_r = parse_int64_strict(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        # Bounded BEFORE it indexes anything: `ke_sic = i + 2 + numkeys` was
        # never checked, and INT64_MAX wrapped it negative, so the option scan
        # read the token table at a negative offset — one command, SIGSEGV.
        if not nk_r.ok or nk_r.value <= 0:
            writer.append_error_response("ERR numkeys should be greater than 0")
            return num_tokens - 1 - i
        if nk_r.value > Int64(num_tokens - (i + 2)):
            writer.append_error_response("ERR Number of keys can't be greater than number of args")
            return num_tokens - 1 - i
        var nk_sic = Int(nk_r.value)
        var ks_sic = i + 2; var ke_sic = i + 2 + nk_sic
        var sic_limit = 0; var j_sic = ke_sic
        # LIMIT n and nothing else, as Redis parses it: a word that was five
        # letters starting with "l" was LIMIT, and anything else was skipped.
        while j_sic < num_tokens:
            var op_sic = tokens[unsafe_offset=j_sic]
            if arg_eq(op_sic.ptr, op_sic.length, "limit") and j_sic + 1 < num_tokens:
                var lim = parse_int64_strict(tokens[unsafe_offset=j_sic+1].ptr, tokens[unsafe_offset=j_sic+1].length)
                if not lim.ok or lim.value < 0:
                    writer.append_error_response("ERR LIMIT can't be negative")
                    return num_tokens - 1 - i
                sic_limit = Int(lim.value); j_sic += 1
            else:
                writer.append_error_response("ERR syntax error")
                return num_tokens - 1 - i
            j_sic += 1
        var consumed = j_sic - 1 - i
        if nk_sic <= 0: writer.append_int_response(Int64(0))
        else:
            var first_sic_v = GenericValue.borrow(tokens[unsafe_offset=ks_sic].ptr, tokens[unsafe_offset=ks_sic].length)
            var first_sic = keyspace[].get(first_sic_v)
            var sic_cnt = 0
            if not first_sic.is_none() and first_sic.type.value == ValueType.SET:
                var fsic_ptr = first_sic.as_set().unsafe_bitcast[SlabHashMap]()
                for slot in range(fsic_ptr[].capacity):
                    if sic_limit > 0 and sic_cnt >= sic_limit: break
                    var m = fsic_ptr[].metadata[unsafe_offset=slot]
                    if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                        var k = fsic_ptr[].keys[unsafe_offset=slot]; var in_all_sic = True
                        var js = ks_sic + 1
                        while js < ke_sic:
                            var ov_sic = GenericValue.borrow(tokens[unsafe_offset=js].ptr, tokens[unsafe_offset=js].length)
                            var oval_sic = keyspace[].get(ov_sic)
                            if oval_sic.is_none() or oval_sic.type.value != ValueType.SET: in_all_sic = False; break
                            if oval_sic.as_set().unsafe_bitcast[SlabHashMap]()[].get(k).is_none(): in_all_sic = False; break
                            js += 1
                        if in_all_sic: sic_cnt += 1
            writer.append_int_response(Int64(sic_cnt))
        return consumed
    else:
        writer.append_error_response("ERR wrong number of arguments for 'sintercard' command")
        return 0


@always_inline
def handle_sunion(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SUNION key [key ...] → array of members in union of all sets."""
    if i + 1 < num_tokens:
        # gh #232: wrong type answered like an empty container. Every call
        # site ignores this return and sets i = cmd_end_tok - 1 itself, so
        # returning 0 consumes the frame correctly. Checked for EVERY key before
        # the temporary map exists: the refusal used to return from inside the
        # loop and leak it (~4 KB per wrong-type SUNION, gh #394).
        for j_chk in range(i + 1, num_tokens):
            var chk = keyspace[].get(GenericValue.borrow(tokens[unsafe_offset=j_chk].ptr, tokens[unsafe_offset=j_chk].length))
            if not chk.is_none() and chk.type.value != ValueType.SET:
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return 0
        var tmp_su = alloc[SlabHashMap](1); tmp_su.unsafe_write(SlabHashMap(64))
        var j_su = i + 1
        while j_su < num_tokens:
            var su_v = GenericValue.borrow(tokens[unsafe_offset=j_su].ptr, tokens[unsafe_offset=j_su].length)
            var su_val = keyspace[].get(su_v)
            if not su_val.is_none() and su_val.type.value == ValueType.SET:
                var su_ptr = su_val.as_set().unsafe_bitcast[SlabHashMap]()
                for slot in range(su_ptr[].capacity):
                    var m = su_ptr[].metadata[unsafe_offset=slot]
                    if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                        tmp_su[].set(su_ptr[].keys[unsafe_offset=slot], GenericValue.from_int(1))   # borrowed
            j_su += 1
        writer.append_set_header(tmp_su[].size)
        for slot in range(tmp_su[].capacity):
            var m = tmp_su[].metadata[unsafe_offset=slot]
            if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                writer.append_bulk_value_response(tmp_su[].keys[unsafe_offset=slot])
        # tmp_su BORROWS the sources' members: destroying it must not free them.
        tmp_su[].forget_borrowed()
        tmp_su.unsafe_deinit_pointee(); tmp_su.unsafe_free()
        return num_tokens - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'sunion' command")
        return 0


@always_inline
def handle_sunionstore(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SUNIONSTORE destination key [key ...] → integer count of members in resulting set."""
    if i + 2 < num_tokens:
        var dst_sus_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var res_sus = alloc[SlabHashMap](1); res_sus.unsafe_write(SlabHashMap(64))
        var j_sus = i + 2
        while j_sus < num_tokens:
            var sus_v = GenericValue.borrow(tokens[unsafe_offset=j_sus].ptr, tokens[unsafe_offset=j_sus].length)
            var sus_val = keyspace[].get(sus_v)
            if not sus_val.is_none() and sus_val.type.value == ValueType.SET:
                var sus_ptr = sus_val.as_set().unsafe_bitcast[SlabHashMap]()
                for slot in range(sus_ptr[].capacity):
                    var m = sus_ptr[].metadata[unsafe_offset=slot]
                    if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                        # The destination OWNS its members: a shallow copy shared the
                        # source's payload, so DEL of either freed the other's member.
                        if res_sus[].get(sus_ptr[].keys[unsafe_offset=slot]).is_none():
                            res_sus[].set(sus_ptr[].keys[unsafe_offset=slot].clone(), GenericValue.from_int(1))
            j_sus += 1
        var sus_cnt = res_sus[].size
        var sus_sv = GenericValue(); sus_sv.type = ValueType(ValueType.SET)
        sus_sv.set_ptr(res_sus.unsafe_bitcast[NoneType]()); _ = remove_and_free(keyspace, dst_sus_v)   # free the container this replaces
        keyspace[].set(dst_sus_v, sus_sv)
        writer.append_int_response(Int64(sus_cnt))
        if sus_cnt == 0: _ = remove_and_free(keyspace, dst_sus_v)   # gh #251
        return num_tokens - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'sunionstore' command")
        return 0


@always_inline
def handle_sdiff(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SDIFF key [key ...] → array of members in first set not in any other set."""
    if i + 1 < num_tokens:
        var first_sd_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var first_sd = keyspace[].get(first_sd_v)
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        # Every key is checked, as Redis checks them (see SINTER).
        if _any_not_set(tokens, i + 1, num_tokens, keyspace):
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif first_sd.is_none() or first_sd.type.value != ValueType.SET:
            writer.append_set_header(0)
        else:
            var fsd_ptr = first_sd.as_set().unsafe_bitcast[SlabHashMap]()
            var sd_cnt = 0
            for slot in range(fsd_ptr[].capacity):
                var m = fsd_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    var k = fsd_ptr[].keys[unsafe_offset=slot]; var in_other = False
                    var j_sd = i + 2
                    while j_sd < num_tokens:
                        var ov_sd = GenericValue.borrow(tokens[unsafe_offset=j_sd].ptr, tokens[unsafe_offset=j_sd].length)
                        var oval_sd = keyspace[].get(ov_sd)
                        if not oval_sd.is_none() and oval_sd.type.value == ValueType.SET:
                            if not oval_sd.as_set().unsafe_bitcast[SlabHashMap]()[].get(k).is_none(): in_other = True; break
                        j_sd += 1
                    if not in_other: sd_cnt += 1
            writer.append_set_header(sd_cnt)
            for slot in range(fsd_ptr[].capacity):
                var m = fsd_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    var k = fsd_ptr[].keys[unsafe_offset=slot]; var in_other2 = False
                    var j_sd2 = i + 2
                    while j_sd2 < num_tokens:
                        var ov_sd2 = GenericValue.borrow(tokens[unsafe_offset=j_sd2].ptr, tokens[unsafe_offset=j_sd2].length)
                        var oval_sd2 = keyspace[].get(ov_sd2)
                        if not oval_sd2.is_none() and oval_sd2.type.value == ValueType.SET:
                            if not oval_sd2.as_set().unsafe_bitcast[SlabHashMap]()[].get(k).is_none(): in_other2 = True; break
                        j_sd2 += 1
                    if not in_other2: writer.append_bulk_value_response(k)
        return num_tokens - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'sdiff' command")
        return 0


@always_inline
def handle_sdiffstore(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SDIFFSTORE destination key [key ...] → integer count of members in resulting set."""
    if i + 2 < num_tokens:
        var dst_sds_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var first_sds_v = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        # gh #232: EVERY participating key is type-checked before any work, and
        # before the result set is allocated. Treating a wrong-type source as an
        # empty set silently changes the answer instead of reporting the error,
        # and here it would also write that wrong answer to the destination.
        var _sds_scan = i + 2
        while _sds_scan < num_tokens:
            var _sds_kv = keyspace[].get(GenericValue.borrow(tokens[unsafe_offset=_sds_scan].ptr, tokens[unsafe_offset=_sds_scan].length))
            if not _sds_kv.is_none() and _sds_kv.type.value != ValueType.SET:
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return num_tokens - 1 - i
            _sds_scan += 1
        var first_sds = keyspace[].get(first_sds_v)
        var res_sds = alloc[SlabHashMap](1); res_sds.unsafe_write(SlabHashMap(16))
        if not first_sds.is_none() and first_sds.type.value == ValueType.SET:
            var fsds_ptr = first_sds.as_set().unsafe_bitcast[SlabHashMap]()
            for slot in range(fsds_ptr[].capacity):
                var m = fsds_ptr[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    var k = fsds_ptr[].keys[unsafe_offset=slot]; var in_other_sds = False
                    var j_sds = i + 3
                    while j_sds < num_tokens:
                        var ov_sds = GenericValue.borrow(tokens[unsafe_offset=j_sds].ptr, tokens[unsafe_offset=j_sds].length)
                        var oval_sds = keyspace[].get(ov_sds)
                        if not oval_sds.is_none() and oval_sds.type.value == ValueType.SET:
                            if not oval_sds.as_set().unsafe_bitcast[SlabHashMap]()[].get(k).is_none(): in_other_sds = True; break
                        j_sds += 1
                    if not in_other_sds: res_sds[].set(k.clone(), GenericValue.from_int(1))   # owned copy
        var sds_cnt = res_sds[].size
        var sds_sv = GenericValue(); sds_sv.type = ValueType(ValueType.SET)
        sds_sv.set_ptr(res_sds.unsafe_bitcast[NoneType]()); _ = remove_and_free(keyspace, dst_sds_v)   # free the container this replaces
        keyspace[].set(dst_sds_v, sds_sv)
        writer.append_int_response(Int64(sds_cnt))
        if sds_cnt == 0: _ = remove_and_free(keyspace, dst_sds_v)   # gh #251
        return num_tokens - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'sdiffstore' command")
        return 0


@always_inline
def handle_sscan(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SSCAN key cursor [MATCH pattern] [COUNT count] → cursor + array of members."""
    if i + 2 < num_tokens:
        var key_v_ss = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var ss_cursor = scan_cursor(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        var consumed = num_tokens - 1 - i
        var val_ss = keyspace[].get(key_v_ss)
        # Redis's order: cursor, key (missing: an empty scan, whatever the
        # options; another type: WRONGTYPE, which this answered as an empty
        # scan), then the options.
        if not val_ss.is_none() and val_ss.type.value != ValueType.SET:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return consumed
        var so = scan_no_opts()
        if not val_ss.is_none():
            so = parse_scan_opts(tokens, i + 3, num_tokens, writer, False, False)
            if not so.ok:
                return consumed
        var ss_pat_p = so.pat_p
        var ss_pat_l = so.pat_l
        if val_ss.is_none() or ss_cursor != 0:
            var ss_empty = "*2\r\n$1\r\n0\r\n*0\r\n"
            writer.append_to_response(ss_empty.unsafe_ptr(), ss_empty.byte_length())
        else:
            var ssp = val_ss.as_set().unsafe_bitcast[SlabHashMap]()
            var ss_all = not so.has_match or _glob_all(ss_pat_p, ss_pat_l)
            var ss_mb = alloc[UInt8](24)
            var ss_n = 0
            for slot in range(ssp[].capacity):
                var m0 = ssp[].metadata[unsafe_offset=slot]
                if m0 != SlabHashMap.EMPTY and m0 != SlabHashMap.DELETED:
                    if ss_all: ss_n += 1
                    else:
                        var mk0 = ssp[].keys[unsafe_offset=slot]
                        if _glob_match(ss_pat_p, ss_pat_l, 0, mk0.as_string_safe(ss_mb), mk0.string_len(), 0):
                            ss_n += 1
            var ss_hdr = String("*2\r\n$1\r\n0\r\n*") + String(ss_n) + String("\r\n")
            writer.append_to_response(ss_hdr.unsafe_ptr(), ss_hdr.byte_length())
            for slot in range(ssp[].capacity):
                var m = ssp[].metadata[unsafe_offset=slot]
                if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                    if not ss_all:
                        var mk1 = ssp[].keys[unsafe_offset=slot]
                        if not _glob_match(ss_pat_p, ss_pat_l, 0, mk1.as_string_safe(ss_mb), mk1.string_len(), 0):
                            continue
                    writer.append_bulk_value_response(ssp[].keys[unsafe_offset=slot])
            ss_mb.unsafe_free()
        return consumed
    else:
        writer.append_error_response("ERR wrong number of arguments for 'sscan' command")
        return 0
