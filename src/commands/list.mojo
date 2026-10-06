"""List commands: LINDEX, LSET, LINSERT, LREM, LTRIM, LPOS, LMOVE."""
from src.common.utils import strict_atol, arg_eq, parse_int64_strict, ParsedInt
from src.common.container_free import remove_and_free
from src.common.ptr import is_not_null, is_null
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy
from std.ffi import external_call
from std.collections import Array
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.dispatcher import CommandDispatcher
from src.common.hash_map import StripedHashMap
from src.common.value import GenericValue, ValueType
from src.io.wal import WAL
from src.common.list import SlabList


@no_inline
def _elem_eq(v: GenericValue, p: Pointer[UInt8, MutUntrackedOrigin], l: Int,
             scratch: Pointer[UInt8, MutUntrackedOrigin]) -> Bool:
    """gh #241: byte-compare a list element against a token. Cold path only —
    used by the quicklist branches of LINSERT/LREM/LPOS, which are O(N) anyway.

    `as_string_safe` copies an SSO value into `scratch` (needs >=23B) and
    returns the heap pointer directly for a long string, so this works for both
    representations without materializing a String."""
    if not v.is_string():
        return False
    if v.string_len() != l:
        return False
    var vp = v.as_string_safe(scratch)
    for k in range(l):
        if vp[unsafe_offset=k] != p[unsafe_offset=k]:
            return False
    return True


@always_inline
def handle_lindex(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """LINDEX key index → bulk string at index, or nil."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var idx_str = tokens[unsafe_offset=i+2].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var li_idx = strict_atol(idx_str)   # gh #393: atol took " 1", "+1", "01"
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.LIST:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.LIST:
            writer.append_null_response()
        else:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            var lsz2 = list_ptr[].size
            if li_idx < 0: li_idx += lsz2
            if li_idx < 0 or li_idx >= lsz2:
                writer.append_null_response()
            elif is_not_null(list_ptr[].zip_buf):
                # Ziplist traversal
                var zoff2 = 0
                var found_li = False
                var zi2 = 0
                while zi2 < lsz2:
                    var vlen = Int((list_ptr[].zip_buf.unsafe_offset(zoff2)).unsafe_bitcast[UInt16]()[])
                    if zi2 == li_idx:
                        writer.append_bulk_string_response(list_ptr[].zip_buf.unsafe_offset(zoff2).unsafe_offset(2), vlen)
                        found_li = True; break
                    zoff2 += 2 + vlen; zi2 += 1
                if not found_li: writer.append_null_response()
            else:
                # Quicklist traversal
                var global_li = 0
                var found_li2 = False
                comptime SEG3 = SlabList.SEG_SIZE
                var hi3 = list_ptr[].head_off
                while hi3 < list_ptr[].head_end and not found_li2:   # live range
                    if global_li == li_idx:
                        writer.append_bulk_value_response(list_ptr[].active_head_data[unsafe_offset=hi3])
                        found_li2 = True
                    hi3 += 1; global_li += 1
                var seg3 = list_ptr[].head_seg_count - 1
                while seg3 >= list_ptr[].head_segs_start and not found_li2:
                    var j4 = 0
                    while j4 < SEG3 and not found_li2:
                        if global_li == li_idx:
                            writer.append_bulk_value_response(list_ptr[].head_segs[unsafe_offset=seg3][unsafe_offset=j4])
                            found_li2 = True
                        j4 += 1; global_li += 1
                    seg3 -= 1
                var tseg3 = 0   # tail segs start at 0
                while tseg3 < list_ptr[].tail_seg_count and not found_li2:
                    var j5 = 0
                    while j5 < SEG3 and not found_li2:
                        if global_li == li_idx:
                            writer.append_bulk_value_response(list_ptr[].tail_segs[unsafe_offset=tseg3][unsafe_offset=j5])
                            found_li2 = True
                        j5 += 1; global_li += 1
                    tseg3 += 1
                var tk3 = 0
                while tk3 < list_ptr[].tail_count and not found_li2:
                    if global_li == li_idx:
                        writer.append_bulk_value_response(list_ptr[].active_tail_data[unsafe_offset=tk3])
                        found_li2 = True
                    tk3 += 1; global_li += 1
                if not found_li2: writer.append_null_response()
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'lindex' command")
        return 0


@always_inline
def handle_lset(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """LSET key index value → +OK or error."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var idx_str2 = tokens[unsafe_offset=i+2].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var ls_idx = strict_atol(idx_str2)   # gh #393
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.LIST:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.LIST:
            writer.append_error_response("ERR no such key")
        else:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            var lsz3 = list_ptr[].size
            if ls_idx < 0: ls_idx += lsz3
            if ls_idx < 0 or ls_idx >= lsz3:
                writer.append_error_response("ERR index out of range")
            elif is_not_null(list_ptr[].zip_buf):
                # Rebuild ziplist with new value at index
                var new_val_tok = tokens[unsafe_offset=i+3]
                var zoff3 = 0; var zi3 = 0
                while zi3 < lsz3:
                    var vlen = Int((list_ptr[].zip_buf.unsafe_offset(zoff3)).unsafe_bitcast[UInt16]()[])
                    if zi3 == ls_idx:
                        # Replace: shift remaining data
                        var old_entry_size = 2 + vlen
                        var new_entry_size = 2 + new_val_tok.length
                        var tail_start = zoff3 + old_entry_size
                        var tail_len = list_ptr[].zip_len - tail_start
                        var new_zip_len = list_ptr[].zip_len - old_entry_size + new_entry_size
                        if new_zip_len <= list_ptr[].zip_cap:
                            if new_entry_size != old_entry_size and tail_len > 0:
                                # Same-buffer shift: overlaps whenever the size
                                # delta is smaller than the tail. Measured on
                                # 0.921: 12 elements, `LSET 3 zzz` smeared the
                                # replacement into the tail and turned e11 into
                                # "eee". memmove, not memcpy.
                                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                                    (list_ptr[].zip_buf.unsafe_offset(zoff3).unsafe_offset(new_entry_size)).unsafe_bitcast[NoneType](),
                                    (list_ptr[].zip_buf.unsafe_offset(tail_start)).unsafe_bitcast[NoneType](),
                                    tail_len,
                                )
                            (list_ptr[].zip_buf.unsafe_offset(zoff3)).unsafe_bitcast[UInt16]()[unsafe_offset=0] = UInt16(new_val_tok.length)
                            unsafe_memcpy(dest=list_ptr[].zip_buf.unsafe_offset(zoff3).unsafe_offset(2), src=new_val_tok.ptr, count=new_val_tok.length)
                            list_ptr[].zip_len = new_zip_len
                            _ = wal[].append_u64_val(19, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length,
                                                     UInt64(ls_idx), new_val_tok.ptr, new_val_tok.length)
                        writer.append_ok_response()
                        break
                    zoff3 += 2 + vlen; zi3 += 1
            else:
                # gh #241: quicklist mode. Deep-copy out, swap the one slot,
                # rebuild. `ls_idx` is already range-checked above.
                var ls_new = tokens[unsafe_offset=i+3]
                var ls_elems = list_ptr[].owned_elems()
                ls_elems[ls_idx].free_str_payload()
                ls_elems[ls_idx] = GenericValue.from_ptr(ls_new.ptr, ls_new.length)
                list_ptr[].replace_all(ls_elems)
                _ = wal[].append_u64_val(19, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length,
                                         UInt64(ls_idx), ls_new.ptr, ls_new.length)
                writer.append_ok_response()
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'lset' command")
        return 0


@always_inline
def handle_linsert(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """LINSERT key BEFORE|AFTER pivot value → list length or -1."""
    if i + 4 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var ins_tp = tokens[unsafe_offset=i+2].ptr; var ins_tl = tokens[unsafe_offset=i+2].length
        # BEFORE|AFTER, whole word, checked before the key as Redis does: a
        # 6-byte word starting with "b" was BEFORE and anything else AFTER.
        var before = arg_eq(ins_tp, ins_tl, "before")
        if not before and not arg_eq(ins_tp, ins_tl, "after"):
            writer.append_error_response("ERR syntax error")
            return 4
        var val = keyspace[].get(key_v)
        var pivot_tok = tokens[unsafe_offset=i+3]
        var val_tok2 = tokens[unsafe_offset=i+4]
        if val.is_none():
            writer.append_int_response(Int64(0))
        elif val.type.value != ValueType.LIST:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            if is_null(list_ptr[].zip_buf):
                # gh #241: quicklist mode. This used to return -1 unconditionally
                # — indistinguishable from "pivot not found", so a caller could
                # not tell a missing pivot from an unsupported representation.
                var li_elems = list_ptr[].owned_elems()
                var li_scratch = alloc[UInt8](64)
                var li_at = -1
                for li_j in range(len(li_elems)):
                    if _elem_eq(li_elems[li_j], pivot_tok.ptr, pivot_tok.length, li_scratch):
                        li_at = li_j
                        break
                li_scratch.unsafe_free()
                if li_at < 0:
                    # Pivot absent: nothing changes, so free the copies rather
                    # than rebuilding the list with them.
                    for li_j in range(len(li_elems)):
                        li_elems[li_j].free_str_payload()
                    writer.append_int_response(Int64(-1))
                else:
                    var li_out = List[GenericValue]()
                    li_out.reserve(len(li_elems) + 1)
                    var li_ins = li_at if before else li_at + 1
                    for li_j in range(len(li_elems)):
                        if li_j == li_ins:
                            li_out.append(GenericValue.from_ptr(val_tok2.ptr, val_tok2.length))
                        li_out.append(li_elems[li_j])
                    if li_ins == len(li_elems):
                        li_out.append(GenericValue.from_ptr(val_tok2.ptr, val_tok2.length))
                    list_ptr[].replace_all(li_out)
                    _ = wal[].append_linsert(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length, before,
                                             pivot_tok.ptr, pivot_tok.length,
                                             val_tok2.ptr, val_tok2.length)
                    writer.append_int_response(Int64(list_ptr[].size))
            else:
                # Find pivot in ziplist
                var zoff4 = 0; var zi4 = 0; var lsz4 = list_ptr[].size
                var found_piv = False
                while zi4 < lsz4:
                    var vlen = Int((list_ptr[].zip_buf.unsafe_offset(zoff4)).unsafe_bitcast[UInt16]()[])
                    var match_piv = vlen == pivot_tok.length
                    if match_piv:
                        for pk in range(vlen):
                            if (list_ptr[].zip_buf.unsafe_offset(zoff4).unsafe_offset(2))[unsafe_offset=pk] != pivot_tok.ptr[unsafe_offset=pk]: match_piv = False; break
                    if match_piv:
                        # Insert before or after this entry
                        var ins_at = zoff4 if before else zoff4 + 2 + vlen
                        var ins_entry_sz = 2 + val_tok2.length
                        var new_zip_len2 = list_ptr[].zip_len + ins_entry_sz
                        if new_zip_len2 <= list_ptr[].zip_cap:
                            # Shift tail right
                            var tail_len2 = list_ptr[].zip_len - ins_at
                            if tail_len2 > 0:
                                # MUST be memmove: src and dest are the SAME buffer
                                # and overlap whenever tail_len2 > ins_entry_sz.
                                # unsafe_memcpy smeared the overlap forward, so
                                # `[a,b,c] LINSERT BEFORE b X` produced [a,X,b,b]
                                # and `[one] LINSERT BEFORE one M` returned raw
                                # header bytes ('o',0x03,0x00) as element data —
                                # corruption that reached clients, the WAL and the
                                # snapshot, and could desync RESP when a garbage
                                # byte landed where a length was expected.
                                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                                    (list_ptr[].zip_buf.unsafe_offset(ins_at).unsafe_offset(ins_entry_sz)).unsafe_bitcast[NoneType](),
                                    (list_ptr[].zip_buf.unsafe_offset(ins_at)).unsafe_bitcast[NoneType](),
                                    tail_len2,
                                )
                            (list_ptr[].zip_buf.unsafe_offset(ins_at)).unsafe_bitcast[UInt16]()[unsafe_offset=0] = UInt16(val_tok2.length)
                            unsafe_memcpy(dest=list_ptr[].zip_buf.unsafe_offset(ins_at).unsafe_offset(2), src=val_tok2.ptr, count=val_tok2.length)
                            list_ptr[].zip_len = new_zip_len2
                            list_ptr[].size += 1
                            _ = wal[].append_linsert(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length, before,
                                                     pivot_tok.ptr, pivot_tok.length,
                                                     val_tok2.ptr, val_tok2.length)
                        found_piv = True; break
                    zoff4 += 2 + vlen; zi4 += 1
                if found_piv: writer.append_int_response(Int64(list_ptr[].size))
                else: writer.append_int_response(Int64(-1))
        return 4
    else:
        writer.append_error_response("ERR wrong number of arguments for 'linsert' command")
        return 0


@always_inline
def handle_lrem(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """LREM key count value → number of removed elements."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var count_str2 = tokens[unsafe_offset=i+2].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var lrem_count = strict_atol(count_str2)   # gh #393
        if lrem_count == Int.MIN:
            # Redis reads the count as a range [-LONG_MAX, LONG_MAX]; -2^63 has
            # no magnitude (`-lrem_count` below is itself).
            writer.append_error_response("ERR value is out of range, value must between -9223372036854775807 and 9223372036854775807")
            return 3
        var val_tok3 = tokens[unsafe_offset=i+3]
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.LIST:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.LIST:
            writer.append_int_response(Int64(0))
        else:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            if is_null(list_ptr[].zip_buf):
                # gh #241: quicklist mode. Returning 0 here was worse than an
                # error — the caller sees "no matching elements" and moves on,
                # when in fact the elements are still there.
                var lr_abs = lrem_count if lrem_count >= 0 else -lrem_count
                var lr_elems = list_ptr[].owned_elems()
                var lr_scratch = alloc[UInt8](64)
                var lr_total = 0
                for lr_j in range(len(lr_elems)):
                    if _elem_eq(lr_elems[lr_j], val_tok3.ptr, val_tok3.length, lr_scratch):
                        lr_total += 1
                # count<0 removes the LAST |count| matches, so skip the leading
                # ones; count>0 removes the first |count|; count==0 removes all.
                var lr_skip = 0 if lrem_count >= 0 else (lr_total - lr_abs)
                if lr_skip < 0: lr_skip = 0
                var lr_seen = 0
                var lr_removed = 0
                var lr_out = List[GenericValue]()
                lr_out.reserve(len(lr_elems))
                for lr_j in range(len(lr_elems)):
                    var lr_drop = False
                    if _elem_eq(lr_elems[lr_j], val_tok3.ptr, val_tok3.length, lr_scratch):
                        if lr_abs == 0 or (lr_seen >= lr_skip and lr_removed < lr_abs):
                            lr_drop = True
                            lr_removed += 1
                        lr_seen += 1
                    if lr_drop:
                        lr_elems[lr_j].free_str_payload()
                    else:
                        lr_out.append(lr_elems[lr_j])
                lr_scratch.unsafe_free()
                list_ptr[].replace_all(lr_out)
                if lr_removed > 0:
                    _ = wal[].append_u64_val(21, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length,
                                             Int64(lrem_count).cast[DType.uint64](),
                                             val_tok3.ptr, val_tok3.length)
                writer.append_int_response(Int64(lr_removed))
                # gh #234: Redis removes an aggregate the moment its last element goes.
                if list_ptr[].size == 0:
                    _ = remove_and_free(keyspace, key_v)
                    if is_not_null(wal):
                        _ = wal[].append(2, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
            else:
                var lrem_abs = lrem_count if lrem_count >= 0 else -lrem_count
                var removed = 0
                var new_zip_buf = alloc[UInt8](list_ptr[].zip_cap)
                var new_zip_len3 = 0
                var old_len3 = list_ptr[].zip_len
                var zoff5 = 0; var zi5 = 0; var lsz5 = list_ptr[].size
                # For count>0: head->tail; count<0: tail->head; count=0: all
                # Simple approach: collect all, filter, rebuild
                var entries_buf = alloc[UInt8](old_len3)
                unsafe_memcpy(dest=entries_buf, src=list_ptr[].zip_buf, count=old_len3)
                # Count total entries matching
                var match_count_tot = 0
                zoff5 = 0
                for _ in range(lsz5):
                    var vlen = Int((entries_buf.unsafe_offset(zoff5)).unsafe_bitcast[UInt16]()[])
                    var is_match = vlen == val_tok3.length
                    if is_match:
                        for pk2 in range(vlen):
                            if (entries_buf.unsafe_offset(zoff5).unsafe_offset(2))[unsafe_offset=pk2] != val_tok3.ptr[unsafe_offset=pk2]: is_match = False; break
                    if is_match: match_count_tot += 1
                    zoff5 += 2 + vlen
                # Decide skip range
                var skip_from = 0 if lrem_count >= 0 else (match_count_tot - lrem_abs)
                if skip_from < 0: skip_from = 0
                var match_seen = 0; zoff5 = 0
                for _ in range(lsz5):
                    var vlen = Int((entries_buf.unsafe_offset(zoff5)).unsafe_bitcast[UInt16]()[])
                    var is_match2 = vlen == val_tok3.length
                    if is_match2:
                        for pk3 in range(vlen):
                            if (entries_buf.unsafe_offset(zoff5).unsafe_offset(2))[unsafe_offset=pk3] != val_tok3.ptr[unsafe_offset=pk3]: is_match2 = False; break
                    var should_remove = False
                    if is_match2:
                        if lrem_abs == 0 or match_seen >= skip_from and (lrem_abs == 0 or removed < lrem_abs):
                            should_remove = True; removed += 1
                        match_seen += 1
                    if not should_remove:
                        (new_zip_buf.unsafe_offset(new_zip_len3)).unsafe_bitcast[UInt16]()[unsafe_offset=0] = UInt16(vlen)
                        unsafe_memcpy(dest=new_zip_buf.unsafe_offset(new_zip_len3).unsafe_offset(2), src=entries_buf.unsafe_offset(zoff5).unsafe_offset(2), count=vlen)
                        new_zip_len3 += 2 + vlen
                    zoff5 += 2 + vlen
                unsafe_memcpy(dest=list_ptr[].zip_buf, src=new_zip_buf, count=new_zip_len3)
                list_ptr[].zip_len = new_zip_len3
                list_ptr[].size -= removed
                if removed > 0:
                    _ = wal[].append_u64_val(21, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length,
                                             Int64(lrem_count).cast[DType.uint64](),
                                             val_tok3.ptr, val_tok3.length)
                new_zip_buf.unsafe_free(); entries_buf.unsafe_free()
                writer.append_int_response(Int64(removed))
                # gh #234: Redis removes an aggregate the moment its last element goes.
                if list_ptr[].size == 0:
                    _ = remove_and_free(keyspace, key_v)
                    if is_not_null(wal):
                        _ = wal[].append(2, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'lrem' command")
        return 0


@always_inline
def handle_ltrim(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """LTRIM key start stop → +OK."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var lt_start = strict_atol(tokens[unsafe_offset=i+2].value())
        var lt_stop = strict_atol(tokens[unsafe_offset=i+3].value())
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.LIST:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.LIST:
            writer.append_ok_response()
        else:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            var lsz6 = list_ptr[].size
            if lt_start < 0: lt_start += lsz6
            if lt_stop < 0: lt_stop += lsz6
            if lt_start < 0: lt_start = 0
            if lt_stop >= lsz6: lt_stop = lsz6 - 1
            if is_null(list_ptr[].zip_buf):
                # gh #241: QUICKLIST mode (any list past 1024 entries).
                #
                # This used to fall into the "clear the list" branch below,
                # because `is_null(zip_buf)` — which only means "not a ziplist" —
                # was OR'd with the genuinely-empty-range test. So EVERY LTRIM on
                # a 1025+ entry list DELETED THE WHOLE LIST and replied +OK,
                # including `LTRIM key 0 -1`, which is documented to keep
                # everything. The capped-log idiom (`RPUSH` then
                # `LTRIM key -N -1`) wiped all history at exactly the size where
                # the cap starts to matter.
                #
                # Trimming by mass-popping was tried and REJECTED: it leaves the
                # segmented structure's size and segment bookkeeping
                # inconsistent, after which LRANGE declares an array header
                # larger than the traversal can fill and the client hangs. That
                # is a worse failure than the one being fixed.
                #
                # gh #256: the paragraph above used to end "so decline,
                # consistent with the rest of the family" — and then the code
                # below it did the opposite. LSET/LINSERT/LREM/LPOS no longer
                # refuse either. Kept only as the record of why mass-popping was
                # rejected; the decision it argued for was superseded the same
                # week and a reader hitting it cold would draw the wrong
                # conclusion about what this function does.
                #
                # 2026-08-20: implemented via deep-copy + rebuild
                # (`owned_elems`/`replace_all`), which is what the mass-pop
                # attempt above was trying to avoid and got wrong. Rebuilding
                # cannot leave `size` and the segment bookkeeping disagreeing,
                # because it reconstructs both from one source of truth.
                var lt_elems = list_ptr[].owned_elems()
                var lt_out = List[GenericValue]()
                for lt_j in range(len(lt_elems)):
                    if lt_j >= lt_start and lt_j <= lt_stop:
                        lt_out.append(lt_elems[lt_j])
                    else:
                        lt_elems[lt_j].free_str_payload()
                list_ptr[].replace_all(lt_out)
                # Log the RESOLVED absolute range, and normalize an empty one to
                # (1, 0) the way the ziplist branch does — lt_stop can still be
                # negative here (a stop index that normalized below zero), and
                # UInt64(-1) would replay as a range covering everything.
                var lt_ws = UInt64(lt_start) if lt_start <= lt_stop else UInt64(1)
                var lt_we = UInt64(lt_stop) if lt_start <= lt_stop else UInt64(0)
                _ = wal[].append_u64x2(20, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length,
                                       lt_ws, lt_we)
                writer.append_ok_response()
                # gh #234: an empty range leaves no list, so the key goes too.
                if list_ptr[].size == 0:
                    _ = remove_and_free(keyspace, key_v)
                    if is_not_null(wal):
                        _ = wal[].append(2, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
            elif lt_start > lt_stop:
                # Genuinely empty range on a ZIPLIST — quicklists were handled
                # above, so zip_buf is non-null here.
                list_ptr[].zip_len = 0
                list_ptr[].size = 0
                _ = wal[].append_u64x2(20, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length, 1, 0)
                _ = remove_and_free(keyspace, key_v)       # gh #234: emptied
                _ = wal[].append(2, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
                writer.append_ok_response()
            else:
                var new_zip_buf2 = alloc[UInt8](list_ptr[].zip_cap)
                var new_zip_len4 = 0; var new_size4 = 0
                var zoff6 = 0; var zi6 = 0
                while zi6 < lsz6:
                    var vlen = Int((list_ptr[].zip_buf.unsafe_offset(zoff6)).unsafe_bitcast[UInt16]()[])
                    if zi6 >= lt_start and zi6 <= lt_stop:
                        (new_zip_buf2.unsafe_offset(new_zip_len4)).unsafe_bitcast[UInt16]()[unsafe_offset=0] = UInt16(vlen)
                        unsafe_memcpy(dest=new_zip_buf2.unsafe_offset(new_zip_len4).unsafe_offset(2), src=list_ptr[].zip_buf.unsafe_offset(zoff6).unsafe_offset(2), count=vlen)
                        new_zip_len4 += 2 + vlen; new_size4 += 1
                    zoff6 += 2 + vlen; zi6 += 1
                unsafe_memcpy(dest=list_ptr[].zip_buf, src=new_zip_buf2, count=new_zip_len4)
                list_ptr[].zip_len = new_zip_len4; list_ptr[].size = new_size4
                new_zip_buf2.unsafe_free()
                _ = wal[].append_u64x2(20, tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length,
                                       UInt64(lt_start), UInt64(lt_stop))
                writer.append_ok_response()
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'ltrim' command")
        return 0


@always_inline
def handle_lpos(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """LPOS key element [RANK rank] [COUNT count] [MAXLEN maxlen] → index or array of indices."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var elem_tok = tokens[unsafe_offset=i+2]
        var consumed = 2
        var lpos_rank = 1; var lpos_count = 1; var lpos_all = False
        var lpos_maxlen = 0   # 0 = unlimited, Redis's default
        # Redis's lposCommand: each option a whole word with its value, in
        # any order; anything else is a syntax error. Unknown words used to end
        # the scan silently and COUNT/MAXLEN were matched by their first
        # letters, so `RANKQ 2` and `COUNTQ 2` ran as if the option were absent.
        var opt_j = i + 3
        while opt_j < num_tokens:
            var ot = tokens[unsafe_offset=opt_j]
            var more = opt_j + 1 < num_tokens
            var ov = parse_int64_strict(tokens[unsafe_offset=opt_j + 1].ptr, tokens[unsafe_offset=opt_j + 1].length) \
                if more else ParsedInt(0, False)
            if arg_eq(ot.ptr, ot.length, "rank") and more:
                if not ov.ok:
                    writer.append_error_response("ERR value is not an integer or out of range")
                    return num_tokens - 1 - i
                # gh #393: -2^63 cannot be negated to walk from the tail.
                if ov.value == Int64(-9223372036854775807) - 1:
                    writer.append_error_response("ERR value is out of range, value must between -9223372036854775807 and 9223372036854775807")
                    return num_tokens - 1 - i
                if ov.value == 0:
                    writer.append_error_response("ERR RANK can't be zero: use 1 to start from the first match, 2 from the second ... or use negative to start from the end of the list")
                    return num_tokens - 1 - i
                lpos_rank = Int(ov.value)
            elif arg_eq(ot.ptr, ot.length, "count") and more:
                if not ov.ok or ov.value < 0:
                    writer.append_error_response("ERR COUNT can't be negative")
                    return num_tokens - 1 - i
                lpos_count = Int(ov.value); lpos_all = lpos_count == 0
            elif arg_eq(ot.ptr, ot.length, "maxlen") and more:
                # gh #232: MAXLEN bounds how many elements are COMPARED.
                if not ov.ok or ov.value < 0:
                    writer.append_error_response("ERR MAXLEN can't be negative")
                    return num_tokens - 1 - i
                lpos_maxlen = Int(ov.value)
            else:
                writer.append_error_response("ERR syntax error")
                return num_tokens - 1 - i
            opt_j += 2
            consumed += 2
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        if not val.is_none() and val.type.value != ValueType.LIST:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif val.is_none() or val.type.value != ValueType.LIST:
            writer.append_null_response()
        else:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            var lsz7 = list_ptr[].size
            var rank_seen = 0; var results_found = 0
            var lpos_results = List[Int]()
            var all_matches = List[Int]()
            if is_not_null(list_ptr[].zip_buf):
                var zoff7 = 0; var zi7 = 0
                while zi7 < lsz7:
                    var vlen = Int((list_ptr[].zip_buf.unsafe_offset(zoff7)).unsafe_bitcast[UInt16]()[])
                    var is_match3 = vlen == elem_tok.length
                    if is_match3:
                        for pk4 in range(vlen):
                            if (list_ptr[].zip_buf.unsafe_offset(zoff7).unsafe_offset(2))[unsafe_offset=pk4] != elem_tok.ptr[unsafe_offset=pk4]: is_match3 = False; break
                    if is_match3:
                        rank_seen += 1
                        all_matches.append(zi7)
                    zoff7 += 2 + vlen; zi7 += 1
            else:
                # gh #241: quicklist mode. The scan above is ziplist-only, so
                # `all_matches` stayed empty and LPOS answered nil for every
                # element of any list past 1024 entries — a wrong answer, not a
                # refusal, and the element really is in the list.
                var lp_elems = list_ptr[].get_all()
                var lp_scratch = alloc[UInt8](64)
                for lp_j in range(len(lp_elems)):
                    if _elem_eq(lp_elems[lp_j], elem_tok.ptr, elem_tok.length, lp_scratch):
                        rank_seen += 1
                        all_matches.append(lp_j)
                lp_scratch.unsafe_free()
            # gh #236: a NEGATIVE rank means "count from the tail" (RANK -1 is
            # the LAST occurrence). The old scan tested `rank_seen >= lpos_rank`
            # head-first, which is trivially true for any negative rank, so
            # `LPOS l a RANK -1` returned the FIRST match instead of the last.
            # Collect every match, then select — the list is walked once either
            # way and LPOS carries no gate row.
            # gh #232: apply MAXLEN before selecting. It bounds the COMPARISON
            # window, and the window runs from the end the scan starts at — so
            # a negative rank, which searches from the tail, keeps the LAST
            # `maxlen` elements rather than the first.
            # `all_matches` is ascending, so the MAXLEN window is a contiguous
            # slice of it: a PREFIX for a head-first search and a SUFFIX for a
            # tail-first one. Expressed as bounds because List is not
            # ImplicitlyCopyable, and rebuilding it would copy for nothing.
            var _lo = 0
            var _hi = len(all_matches)
            if lpos_maxlen > 0:
                if lpos_rank >= 0:
                    while _hi > 0 and all_matches[_hi - 1] >= lpos_maxlen: _hi -= 1
                else:
                    while _lo < _hi and all_matches[_lo] < lsz7 - lpos_maxlen: _lo += 1
            var _nm = _hi - _lo
            if lpos_rank > 0:
                var _ri2 = lpos_rank - 1
                while _ri2 < _nm:
                    lpos_results.append(all_matches[_lo + _ri2]); results_found += 1
                    if not lpos_all and results_found >= lpos_count: break
                    _ri2 += 1
            elif lpos_rank < 0:
                var _ri3 = _nm + lpos_rank
                while _ri3 >= 0:
                    lpos_results.append(all_matches[_lo + _ri3]); results_found += 1
                    if not lpos_all and results_found >= lpos_count: break
                    _ri3 -= 1
            if lpos_all or lpos_count > 1:
                var lpos_hdr = String("*") + String(len(lpos_results)) + String("\r\n")
                writer.append_to_response(lpos_hdr.unsafe_ptr(), lpos_hdr.byte_length())
                for ri in range(len(lpos_results)): writer.append_int_response(Int64(lpos_results[ri]))
            elif len(lpos_results) > 0: writer.append_int_response(Int64(lpos_results[0]))
            else: writer.append_null_response()
        return consumed
    else:
        writer.append_error_response("ERR wrong number of arguments for 'lpos' command")
        return 0


@always_inline
def handle_lmove(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut dispatcher: CommandDispatcher) raises -> Int:
    """LMOVE source destination LEFT|RIGHT LEFT|RIGHT → bulk string of moved element."""
    if i + 4 < num_tokens:
        var src_str4 = tokens[unsafe_offset=i+1].value()
        var dst_str4 = tokens[unsafe_offset=i+2].value()
        # LEFT|RIGHT, whole words, checked before either key as Redis does
        # (getListPositionFromObjectOrReply). The first letter alone decided:
        # "LEFTQ" was LEFT and any word not starting with "l" was RIGHT.
        var src_dir_t = tokens[unsafe_offset=i+3]
        var dst_dir_t = tokens[unsafe_offset=i+4]
        var src_dir_from_left = arg_eq(src_dir_t.ptr, src_dir_t.length, "left")
        var dst_dir_to_left = arg_eq(dst_dir_t.ptr, dst_dir_t.length, "left")
        if (not src_dir_from_left and not arg_eq(src_dir_t.ptr, src_dir_t.length, "right")) \
                or (not dst_dir_to_left and not arg_eq(dst_dir_t.ptr, dst_dir_t.length, "right")):
            writer.append_error_response("ERR syntax error")
            return 4
        var src_v4 = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var src_val = keyspace[].get(src_v4)
        # gh #232: a MISSING key and a key of the WRONG TYPE were answered
        # identically, with an empty/zero reply. Redis distinguishes them, and
        # the conflation runs in the dangerous direction: a caller who stored
        # the WRONG KIND of value here is told the container is empty, so the
        # bug looks like missing data instead of a type error at the call site.
        #
        # BOTH keys are validated HERE, before anything is mutated, because
        # Redis does — and because the first version of this fix did not. It
        # checked the destination AFTER the pop and AFTER writing the popped
        # element as the reply, which produced TWO replies for one command
        # (a reply-count desync, gh #156 class) and left the element destroyed:
        # gone from the source, never added to the destination. An LMOVE that
        # refuses must be a no-op, not a partial move with an error stapled on.
        var dst_v4_pre = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        var dst_val_pre = keyspace[].get(dst_v4_pre)
        # Precedence is Redis's and is observable: a wrong-type SOURCE errors,
        # a MISSING source is nil and the destination is never looked at, and
        # only then does a wrong-type destination error. Checking the
        # destination too early makes `LMOVE missing wrongtype` answer
        # WRONGTYPE where Redis answers nil.
        if not src_val.is_none() and src_val.type.value != ValueType.LIST:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        elif src_val.is_none():
            writer.append_null_response()
        else:
            var src_list = src_val.as_list().unsafe_bitcast[SlabList]()
            if src_list[].size == 0:
                writer.append_null_response()
            elif not dst_val_pre.is_none() and dst_val_pre.type.value != ValueType.LIST:
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            else:
                # gh #265: this used to be `elif is_not_null(zip_buf)` with the
                # pop hand-rolled inline, and a trailing `else: nil` — so LMOVE
                # on a >1024-entry list (quicklist mode) reported "source is
                # empty" about a full list. Callers branch on nil, so the
                # reliable-queue idiom stopped moving items at exactly the size
                # where a queue starts to matter, silently.
                #
                # `SlabList.lpop`/`rpop` already implement BOTH representations,
                # and for the ziplist case they are byte-identical to the code
                # that was here — same memmove for the left pop, same
                # truncate-to-prev_offset for the right. So this is a DELETION,
                # not a reimplementation, and quicklist support falls out.
                #
                # Both return an OWNED GenericValue via `from_ptr`, which is
                # required: the ziplist memmove invalidates any borrow into
                # `zip_buf`, and in segmented mode the value lives inline in a
                # segment array a later pop may free.
                var moved_gv = src_list[].lpop() if src_dir_from_left \
                               else src_list[].rpop()
                # gh #170: pop effect on src; the push below logs via the dispatcher
                _ = dispatcher.wal[].append(UInt8(13) if src_dir_from_left else UInt8(14),
                                            tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
                writer.append_bulk_value_response(moved_gv)
                # Push to destination. Push the GenericValue directly — both
                # String(moved_gv) and moved_gv.__str__() stringify the value's
                # internal pointer (e.g. "0x1349…") instead of its bytes,
                # silently corrupting the moved element.
                # The destination type was validated above, before the pop, so
                # there is nothing left to refuse here — only the direction to
                # honour. Re-reading the destination is deliberate: LMOVE k k
                # is a legal rotation, so the pop above may have changed it.
                if dst_dir_to_left:
                    _ = dispatcher.execute_lpush_value(dst_str4, moved_gv)
                else:
                    _ = dispatcher.execute_rpush_value(dst_str4, moved_gv)
                # gh #234 remainder, found by gh #265's differential: Redis
                # removes an aggregate the moment its last element goes, and
                # LMOVE/RPOPLPUSH were missed by that pass because they do not
                # route through the LPOP/RPOP handlers it fixed. Draining a
                # queue left a husk: `EXISTS src` answered 1 for a key holding
                # nothing, and the key could not be reused as another type.
                #
                # AFTER the push, and that ordering is load-bearing: `src == dst`
                # is a legal rotation whose size is 1 again by this point, so
                # checking earlier would delete a list that still has an element.
                if src_list[].size == 0:
                    _ = remove_and_free(keyspace, src_v4)
                    if is_not_null(dispatcher.wal):
                        _ = dispatcher.wal[].append(
                            2, tokens[unsafe_offset=i+1].ptr,
                            tokens[unsafe_offset=i+1].length)
        return 4
    else:
        writer.append_error_response("ERR wrong number of arguments for 'lmove' command")
        return 0
