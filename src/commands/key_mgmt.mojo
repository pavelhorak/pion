"""Key management commands: TYPE, RENAME, RENAMENX, COPY, OBJECT, SORT, SCAN, KEYS, RANDOMKEY, TOUCH, WAIT."""
from src.common.container_free import deep_clone, free_container, remove_and_free, index_field_ttls
from src.commands.scan_opts import parse_scan_opts
from src.common.utils import strict_atol, _glob_match, _glob_all, arg_eq, parse_int64_strict, is_valid_float_arg, parse_float64, scan_cursor, scan_count
from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy
from std.collections import Array, Span, List
from std.ffi import external_call
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.fast_path import _get_now_ns
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.network.cluster import ClusterState


@always_inline
def handle_type(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """TYPE key — return type name as status reply."""
    if i + 1 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var type_resp: String
        var vt = val.type.value
        if vt == ValueType.NONE: type_resp = "+none\r\n"
        elif vt == ValueType.LIST: type_resp = "+list\r\n"
        elif vt == ValueType.SET: type_resp = "+set\r\n"
        elif vt == ValueType.ZSET or vt == ValueType.GEO: type_resp = "+zset\r\n"
        elif vt == ValueType.HASH: type_resp = "+hash\r\n"
        elif vt == ValueType.STREAM: type_resp = "+stream\r\n"
        elif vt == ValueType.VSET: type_resp = "+vectorset\r\n"   # gh #366
        else: type_resp = "+string\r\n"
        writer.append_to_response(type_resp.unsafe_ptr(), type_resp.byte_length())
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'type' command")
        return 0


@always_inline
def handle_rename(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """RENAME key newkey — rename a key."""
    if i + 2 < num_tokens:
        var src_str = tokens[unsafe_offset=i+1].value()
        var dst_str = tokens[unsafe_offset=i+2].value()
        var src_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(src_v)
        if val.is_none(): writer.append_error_response("ERR no such key")
        elif src_str == dst_str:  # gh #123: rename-to-self is a no-op (never free then re-alias)
            writer.append_ok_response()
        else:
            var dst_v = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
            # Read src's TTL first: removing src drops it.
            var exp_v2 = GenericValue()
            if is_not_null(ttl_map):
                exp_v2 = ttl_map[].get(src_v)
            # gh #123: dst gets its own buffer so remove(src) can free src's payload safely.
            keyspace[].set(dst_v, val.clone())
            _ = keyspace[].remove_generic(src_v)
            index_field_ttls(keyspace, dst_v, val)   # gh #392
            if is_not_null(ttl_map):
                if not exp_v2.is_none():
                    ttl_map[].set(dst_v, exp_v2)
                else:
                    # The renamed key carries src's TTL — or none. Keeping dst's
                    # old deadline made the moved value expire on schedule.
                    _ = ttl_map[].remove_generic(dst_v)
            writer.append_ok_response()
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'rename' command")
        return 0


@always_inline
def handle_renamenx(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """RENAMENX key newkey — rename only if newkey does not exist."""
    if i + 2 < num_tokens:
        var src_str2 = tokens[unsafe_offset=i+1].value()
        var dst_str2 = tokens[unsafe_offset=i+2].value()
        var src_v2 = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val2 = keyspace[].get(src_v2)
        if val2.is_none(): writer.append_error_response("ERR no such key")
        elif src_str2 == dst_str2:  # gh #123: rename-to-self no-op; RENAMENX fails (dst exists)
            writer.append_int_response(Int64(0))
        else:
            var dst_v2 = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
            var dst_existing = keyspace[].get(dst_v2)
            if not dst_existing.is_none(): writer.append_int_response(Int64(0))
            else:
                # Read src's TTL first: removing src drops it.
                var exp_v3 = GenericValue()
                if is_not_null(ttl_map):
                    exp_v3 = ttl_map[].get(src_v2)
                keyspace[].set(dst_v2, val2.clone())  # gh #123: dst owns its buffer
                _ = keyspace[].remove_generic(src_v2)
                index_field_ttls(keyspace, dst_v2, val2)   # gh #392
                if is_not_null(ttl_map):
                    if not exp_v3.is_none():
                        ttl_map[].set(dst_v2, exp_v3)
                writer.append_int_response(Int64(1))
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'renamenx' command")
        return 0


@always_inline
def handle_copy(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """COPY source destination [REPLACE] — copy a key."""
    if i + 2 < num_tokens:
        var src_str3 = tokens[unsafe_offset=i+1].value()
        var dst_str3 = tokens[unsafe_offset=i+2].value()
        var src_v3 = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val3 = keyspace[].get(src_v3)
        var do_replace = False
        var j_cp = i + 3
        var consumed = num_tokens - 1 - i
        # Options as Redis parses them: REPLACE, and DB <n> — with one
        # database, DB 0 or nothing. DB 1 used to be ignored, so the
        # copy landed in DB 0 and REPLACE overwrote the key there.
        while j_cp < num_tokens:
            var opt_cp = tokens[unsafe_offset=j_cp]
            if arg_eq(opt_cp.ptr, opt_cp.length, "replace"):
                do_replace = True
            elif arg_eq(opt_cp.ptr, opt_cp.length, "db") and j_cp + 1 < num_tokens:
                var dbr = parse_int64_strict(tokens[unsafe_offset=j_cp + 1].ptr,
                                             tokens[unsafe_offset=j_cp + 1].length)
                if not dbr.ok:
                    writer.append_error_response("ERR value is not an integer or out of range")
                    return consumed
                if dbr.value != 0:
                    writer.append_error_response("ERR DB index is out of range")
                    return consumed
                j_cp += 1
            else:
                writer.append_error_response("ERR syntax error")
                return consumed
            j_cp += 1
        if src_str3 == dst_str3:
            writer.append_error_response("ERR source and destination objects are the same")
            return consumed
        if val3.is_none(): writer.append_int_response(Int64(0))
        else:
            var dst_v3 = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
            var dst_existing2 = keyspace[].get(dst_v3)
            if not dst_existing2.is_none() and not do_replace: writer.append_int_response(Int64(0))
            else:
                # gh #123 / #369: dst owns its buffer AND its container — clone()
                # shared an aggregate's pointer, so COPY aliased the two keys.
                # REPLACE over an aggregate: set() parks the old container in the
                # keyspace graveyard and the engine frees it after the batch
                # (gh #394) — freeing it here as well would free it twice.
                keyspace[].set(dst_v3, deep_clone(val3))
                index_field_ttls(keyspace, dst_v3, val3)   # gh #392: the copy has the same field TTLs
                # gh #242: Redis COPY carries the source's TTL. Pion dropped it,
                # so a copied key silently never expired — the copy outlives the
                # original in any copy-then-expire pattern.
                if is_not_null(ttl_map):
                    var src_ttl = ttl_map[].get(src_v3)
                    if not src_ttl.is_none():
                        ttl_map[].set(dst_v3, src_ttl)
                    else:
                        _ = ttl_map[].remove_generic(dst_v3)
                writer.append_int_response(Int64(1))
        return consumed
    else:
        writer.append_error_response("ERR wrong number of arguments for 'copy' command")
        return 0


def _object_help(mut writer: ResponseWriter):
    """OBJECT HELP: one status line per row, as Redis (it was one bulk string)."""
    var lines = List[String]()
    lines.append("OBJECT <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
    lines.append("ENCODING <key>")
    lines.append("    Return the kind of internal representation used in order to store the value")
    lines.append("    associated with a <key>.")
    lines.append("FREQ <key>")
    lines.append("    Return the access frequency index of the <key>. The returned integer is")
    lines.append("    proportional to the logarithm of the recent access frequency of the key.")
    lines.append("IDLETIME <key>")
    lines.append("    Return the idle time of the <key>, that is the approximated number of")
    lines.append("    seconds elapsed since the last access to the key.")
    lines.append("REFCOUNT <key>")
    lines.append("    Return the number of references of the value associated with the specified")
    lines.append("    <key>.")
    lines.append("HELP")
    lines.append("    Print this help.")
    writer.append_array_header(len(lines))
    for k in range(len(lines)):
        var line = String("+") + lines[k] + "\r\n"
        writer.append_to_response(line.unsafe_ptr(), line.byte_length())


@always_inline
def handle_object(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """OBJECT subcommand key — inspect object internals."""
    if i + 2 < num_tokens:
        var sub_obj = tokens[unsafe_offset=i+1]
        var key_str = tokens[unsafe_offset=i+2].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        var val = keyspace[].get(key_v)
        var sub_p2 = sub_obj.ptr
        var sub_l2 = sub_obj.length
        var _known = (arg_eq(sub_p2, sub_l2, "encoding") or arg_eq(sub_p2, sub_l2, "refcount")
                      or arg_eq(sub_p2, sub_l2, "idletime") or arg_eq(sub_p2, sub_l2, "freq"))
        # A missing key answers nil to every key subcommand, as Redis (#30):
        # ENCODING said "raw", REFCOUNT 1, IDLETIME and FREQ 0.
        if _known and val.is_none():
            writer.append_null_response()
        # ENCODING subcommand
        elif arg_eq(sub_p2, sub_l2, "encoding"):
            var enc_str: String
            var vt2 = val.type.value
            if vt2 == ValueType.INT: enc_str = "int"
            elif vt2 == ValueType.STRING_SSO: enc_str = "embstr"
            elif vt2 == ValueType.STRING: enc_str = "raw"
            elif vt2 == ValueType.LIST: enc_str = "quicklist"
            elif vt2 == ValueType.HASH: enc_str = "hashtable"
            elif vt2 == ValueType.SET: enc_str = "hashtable"
            elif vt2 == ValueType.ZSET or vt2 == ValueType.GEO: enc_str = "skiplist"
            else: enc_str = "raw"
            writer.append_bulk_string_response(enc_str.unsafe_ptr(), enc_str.byte_length())
        elif arg_eq(sub_p2, sub_l2, "refcount"):
            writer.append_int_response(Int64(1))
        elif arg_eq(sub_p2, sub_l2, "idletime"):
            writer.append_int_response(Int64(0))
        elif arg_eq(sub_p2, sub_l2, "freq"):
            # Access frequency exists only under an LFU maxmemory policy, and
            # Pion has none (noeviction): Redis refuses, it does not answer 0.
            writer.append_error_response("ERR An LFU maxmemory policy is not selected, access frequency not tracked. Please note that when switching between policies at runtime LRU and LFU data will take some time to adjust.")
        elif arg_eq(sub_p2, sub_l2, "help"):
            _object_help(writer)
        else: writer.append_error_response("ERR unknown OBJECT subcommand")
        return 2
    else:
        # OBJECT with just subcommand, no key (e.g. OBJECT HELP)
        if i + 1 < num_tokens:
            var sub2 = tokens[unsafe_offset=i+1]
            if arg_eq(sub2.ptr, sub2.length, "help"):
                _object_help(writer)
                return 1
        writer.append_error_response("ERR wrong number of arguments for 'object' command")
        return 0


def _sort_less(a: String, b: String, na: Float64, nb: Float64, alpha: Bool, desc: Bool) -> Bool:
    """Redis's sortCompare: numeric by value, ties broken bytewise (so the
    result is deterministic); ALPHA bytewise. DESC reverses the comparison."""
    var c = 0
    if not alpha:
        if na < nb: c = -1
        elif na > nb: c = 1
    if c == 0:
        if a < b: c = -1
        elif b < a: c = 1
    if desc: c = -c
    return c < 0


def _sort_impl(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
               mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
               ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin], read_only: Bool) raises -> Int:
    """SORT key [BY pattern] [LIMIT offset count] [GET pattern] [ASC|DESC] [ALPHA] [STORE dest]

    Rewritten (found by tests/test_boundary_differential.py and
    test_member_aliasing.py). The old handler matched options by two bytes +
    length, silently ignored BY/GET/STORE (`SORT k STORE d` answered the
    sorted ARRAY and stored nothing), sorted only lists (a set, zset or string
    answered []), parsed numbers with atol (`9.75` sorted as 0) and treated a
    non-number as 0 where Redis errors, and bubble-sorted.

    BY with a pattern containing `*` (external weight keys) and GET are
    REFUSED with an error rather than ignored: an unsupported option must
    never be answered as if it had been applied. `BY <no-star>` is Redis's
    no-sort form and is honoured."""
    if i + 1 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'sort' command")
        return 0
    var consumed = num_tokens - 1 - i
    var alpha = False
    var desc = False
    var dontsort = False
    var lim_off: Int64 = 0
    var lim_cnt: Int64 = -1
    var store_idx = -1
    var j = i + 2
    while j < num_tokens:
        var tp = tokens[unsafe_offset=j].ptr
        var tl = tokens[unsafe_offset=j].length
        if arg_eq(tp, tl, "asc"):
            desc = False
        elif arg_eq(tp, tl, "desc"):
            desc = True
        elif arg_eq(tp, tl, "alpha"):
            alpha = True
        elif arg_eq(tp, tl, "limit") and j + 2 < num_tokens:
            var o = parse_int64_strict(tokens[unsafe_offset=j+1].ptr, tokens[unsafe_offset=j+1].length)
            var c = parse_int64_strict(tokens[unsafe_offset=j+2].ptr, tokens[unsafe_offset=j+2].length)
            if not o.ok or not c.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return consumed
            lim_off = o.value; lim_cnt = c.value
            j += 2
        elif arg_eq(tp, tl, "store") and j + 1 < num_tokens and not read_only:
            store_idx = j + 1
            j += 1
        elif arg_eq(tp, tl, "by") and j + 1 < num_tokens:
            var pp = tokens[unsafe_offset=j+1].ptr
            var has_star = False
            for q in range(tokens[unsafe_offset=j+1].length):
                if pp[unsafe_offset=q] == 42:
                    has_star = True
            if has_star:
                writer.append_error_response("ERR SORT BY <pattern> (external weight keys) is not supported by Pion")
                return consumed
            dontsort = True
            j += 1
        elif arg_eq(tp, tl, "get") and j + 1 < num_tokens:
            writer.append_error_response("ERR SORT GET is not supported by Pion")
            return consumed
        else:
            writer.append_error_response("ERR syntax error")
            return consumed
        j += 1

    var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
    var val = keyspace[].get(key_v)
    key_v.free_str_payload()
    var elems = List[String]()
    var t = val.type.value
    if val.is_none() or t == ValueType.NONE:
        pass
    elif t == ValueType.LIST:
        var owned = val.as_list().unsafe_bitcast[SlabList]()[].owned_elems()
        for e in range(len(owned)):
            elems.append(owned[e].__str__())
            owned[e].free_str_payload()
    elif t == ValueType.SET:
        var sp = val.as_set().unsafe_bitcast[SlabHashMap]()
        for slot in range(sp[].capacity):
            if (sp[].metadata[unsafe_offset=slot] & 0x80) == 0:
                elems.append(sp[].keys[unsafe_offset=slot].__str__())
    elif t == ValueType.ZSET:
        var zn = val.as_zset().unsafe_bitcast[SlabSkipList]()[].head[].forward[0]
        while is_not_null(zn):
            elems.append(zn[].obj.__str__())
            zn = zn[].forward[0]
    else:
        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return consumed

    var n = len(elems)
    var nums = List[Float64]()
    if not alpha and not dontsort:
        for e in range(n):
            var ep = elems[e].unsafe_ptr().unsafe_bitcast[UInt8]()
            var el = elems[e].byte_length()
            var ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ep))
            if not is_valid_float_arg(ptr, el):
                writer.append_error_response("ERR One or more scores can't be converted into double")
                return consumed
            nums.append(parse_float64(ptr, el))
    else:
        for _ in range(n):
            nums.append(0.0)

    # Bottom-up merge sort over an index permutation: O(n log n), stable.
    var idx = List[Int]()
    for e in range(n):
        idx.append(e)
    if not dontsort and n > 1:
        var tmp = List[Int]()
        for e in range(n):
            tmp.append(e)
        var width = 1
        while width < n:
            var lo = 0
            while lo < n:
                var mid = min(lo + width, n)
                var hi = min(lo + 2 * width, n)
                var p = lo
                var q = mid
                var k = lo
                while p < mid and q < hi:
                    if _sort_less(elems[idx[q]], elems[idx[p]], nums[idx[q]], nums[idx[p]], alpha, desc):
                        tmp[k] = idx[q]; q += 1
                    else:
                        tmp[k] = idx[p]; p += 1
                    k += 1
                while p < mid:
                    tmp[k] = idx[p]; p += 1; k += 1
                while q < hi:
                    tmp[k] = idx[q]; q += 1; k += 1
                lo += 2 * width
            for e in range(n):
                idx[e] = tmp[e]
            width *= 2

    var start = Int(lim_off) if lim_off > 0 else 0
    var count = n if lim_cnt < 0 else Int(lim_cnt)
    if start >= n:
        count = 0
    elif start + count > n:
        count = n - start

    if store_idx >= 0:
        var dst = GenericValue.borrow(tokens[unsafe_offset=store_idx].ptr, tokens[unsafe_offset=store_idx].length)
        _ = remove_and_free(keyspace, dst)
        if is_not_null(ttl_map):
            _ = ttl_map[].remove_generic(dst)     # STORE replaces the key: no TTL survives
        if count > 0:
            var owned_out = List[GenericValue]()
            for e in range(start, start + count):
                owned_out.append(GenericValue.from_string(elems[idx[e]]))
            var lp = alloc[SlabList](1)
            lp.unsafe_write(SlabList())
            lp[].replace_all(owned_out)
            var lv = GenericValue()
            lv.type = ValueType(ValueType.LIST)
            lv.set_ptr(lp.bitcast[NoneType]())
            keyspace[].set(dst, lv)
        else:
            dst.free_str_payload()
        writer.append_int_response(Int64(count))
        return consumed

    var hdr = String("*") + String(count) + String("\r\n")
    writer.append_to_response(hdr.unsafe_ptr().unsafe_bitcast[UInt8](), hdr.byte_length())
    for e in range(start, start + count):
        # Straight from the List's heap storage: a local String copy of a
        # short element keeps its bytes INLINE in a stack slot, and that
        # pointer must not cross an out-of-line call (gh #349).
        writer.append_bulk_string_response(elems[idx[e]].unsafe_ptr().unsafe_bitcast[UInt8](),
                                           elems[idx[e]].byte_length())
    return consumed


@always_inline
def handle_sort(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """SORT — see _sort_impl."""
    return _sort_impl(tokens, i, num_tokens, writer, keyspace, ttl_map, False)


@always_inline
def handle_sort_ro(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """SORT_RO — SORT without STORE (STORE is a syntax error here, as in Redis)."""
    return _sort_impl(tokens, i, num_tokens, writer, keyspace, ttl_map, True)


@always_inline
def _key_ns_match(kv: GenericValue, ns_ptr: Pointer[UInt8, MutUntrackedOrigin],
                  ns_len: Int, kbuf: Pointer[UInt8, MutUntrackedOrigin]) -> Int:
    """gh #101: tenant namespace filter for keyspace enumeration. Returns the
    key's byte length when it starts with the ns prefix (always matches when
    ns_len == 0), else -1. kbuf must be ≥24B (SSO extraction scratch); after a
    match the key bytes are readable via kv.as_string_safe(kbuf)."""
    if ns_len == 0:
        return kv.string_len()
    if not kv.is_string():
        return -1
    var klen = kv.string_len()
    if klen <= ns_len:
        return -1
    var kp = kv.as_string_safe(kbuf)
    for k in range(ns_len):
        if kp[unsafe_offset=k] != ns_ptr[unsafe_offset=k]:
            return -1
    return klen


def _type_is(v: GenericValue, tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Bool:
    """SCAN TYPE: does `v` have the type TYPE reports by that name (any case)?
    An unknown name matches nothing, as in Redis 8."""
    var t = v.type.value
    if t == ValueType.LIST: return arg_eq(tp, tl, "list")
    if t == ValueType.SET: return arg_eq(tp, tl, "set")
    if t == ValueType.ZSET or t == ValueType.GEO: return arg_eq(tp, tl, "zset")
    if t == ValueType.HASH: return arg_eq(tp, tl, "hash")
    if t == ValueType.STREAM: return arg_eq(tp, tl, "stream")
    if t == ValueType.VSET: return arg_eq(tp, tl, "vectorset")
    return arg_eq(tp, tl, "string")


@always_inline
def handle_scan(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                ns_ptr: Pointer[UInt8, MutUntrackedOrigin], ns_len: Int) raises -> Int:
    """SCAN cursor [MATCH pattern] [COUNT count] [TYPE type].
    gh #101: when ns_len > 0 (tenant connection), only keys with the tenant
    prefix are returned, with the prefix stripped."""
    if i + 1 < num_tokens:
        var cursor_str = tokens[unsafe_offset=i+1].value()
        var cursor_i2 = scan_cursor(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)   # gh #393
        var consumed = num_tokens - 1 - i
        var so = parse_scan_opts(tokens, i + 2, num_tokens, writer, True, False)
        if not so.ok:
            return consumed
        # cursor 0: return all keys; non-zero: return empty (single sweep)
        if cursor_i2 != 0:
            var scan_empty = "*2\r\n$1\r\n0\r\n*0\r\n"
            writer.append_to_response(scan_empty.unsafe_ptr(), scan_empty.byte_length())
        else:
            var kbuf = alloc[UInt8](24)
            var smbuf = alloc[UInt8](24)
            # An empty MATCH pattern matches only the empty key, as in Redis;
            # "no MATCH" is what matches everything.
            var scan_all = not so.has_match or _glob_all(so.pat_p, so.pat_l)
            # One pass decides which keys answer; the reply's header is their
            # count, so the emit pass cannot disagree with it.
            var hit_shard = List[Int]()
            var hit_slot = List[Int]()
            var hit_len = List[Int]()
            for shard_i in range(8):
                var sp = keyspace[].shards.unsafe_offset(shard_i)
                for slot in range(sp[].capacity):
                    var m = sp[].metadata[unsafe_offset=slot]
                    if m == SlabHashMap.EMPTY or m == SlabHashMap.DELETED:
                        continue
                    var klen = _key_ns_match(sp[].keys[unsafe_offset=slot], ns_ptr, ns_len, kbuf)
                    if klen < 0:
                        continue
                    if not scan_all:
                        var kp1 = sp[].keys[unsafe_offset=slot].as_string_safe(smbuf)
                        if not _glob_match(so.pat_p, so.pat_l, 0, kp1.unsafe_offset(ns_len), klen - ns_len, 0):
                            continue
                    if so.has_type and not _type_is(sp[].values[unsafe_offset=slot], so.type_p, so.type_l):
                        continue
                    hit_shard.append(shard_i)
                    hit_slot.append(slot)
                    hit_len.append(klen)
            var scan_hdr = String("*2\r\n$1\r\n0\r\n*") + String(len(hit_slot)) + String("\r\n")
            writer.append_to_response(scan_hdr.unsafe_ptr(), scan_hdr.byte_length())
            for h in range(len(hit_slot)):
                var sp = keyspace[].shards.unsafe_offset(hit_shard[h])
                if ns_len == 0:
                    writer.append_bulk_value_response(sp[].keys[unsafe_offset=hit_slot[h]])
                else:
                    var kp = sp[].keys[unsafe_offset=hit_slot[h]].as_string_safe(kbuf)
                    writer.append_bulk_string_response(kp.unsafe_offset(ns_len), hit_len[h] - ns_len)
            kbuf.unsafe_free()
            smbuf.unsafe_free()
        return consumed
    else:
        writer.append_error_response("ERR wrong number of arguments for 'scan' command")
        return 0


def handle_keys(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                ns_ptr: Pointer[UInt8, MutUntrackedOrigin], ns_len: Int) raises -> Int:
    """KEYS pattern — glob-matched (gh #243).
    gh #101: when ns_len > 0 (tenant connection), only keys with the tenant
    prefix are returned, with the prefix stripped — and the pattern is matched
    against the STRIPPED key, i.e. what the client actually sees."""
    var consumed = 0
    var pat_p = null_ptr[UInt8, MutUntrackedOrigin]()
    var pat_l = 0
    if i + 1 < num_tokens:
        consumed = 1  # consume pattern arg
        pat_p = tokens[unsafe_offset=i+1].ptr
        pat_l = tokens[unsafe_offset=i+1].length
    # `*` is the overwhelmingly common argument; skip the matcher entirely.
    var match_all = pat_l == 0 or _glob_all(pat_p, pat_l)
    var kbuf = alloc[UInt8](24)
    var mbuf = alloc[UInt8](24)
    var total_keys2 = 0
    for shard_i in range(8):
        var sp2 = keyspace[].shards.unsafe_offset(shard_i)
        for slot in range(sp2[].capacity):
            var m = sp2[].metadata[unsafe_offset=slot]
            if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                var kl0 = _key_ns_match(sp2[].keys[unsafe_offset=slot], ns_ptr, ns_len, kbuf)
                if kl0 >= 0:
                    if match_all:
                        total_keys2 += 1
                    else:
                        var kp0 = sp2[].keys[unsafe_offset=slot].as_string_safe(mbuf)
                        if _glob_match(pat_p, pat_l, 0, kp0.unsafe_offset(ns_len), kl0 - ns_len, 0):
                            total_keys2 += 1
    var keys_hdr = String("*") + String(total_keys2) + String("\r\n")
    writer.append_to_response(keys_hdr.unsafe_ptr(), keys_hdr.byte_length())
    for shard_i in range(8):
        var sp2 = keyspace[].shards.unsafe_offset(shard_i)
        for slot in range(sp2[].capacity):
            var m = sp2[].metadata[unsafe_offset=slot]
            if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                var klen = _key_ns_match(sp2[].keys[unsafe_offset=slot], ns_ptr, ns_len, kbuf)
                if klen < 0:
                    continue
                if not match_all:
                    # MUST use the same predicate as the counting pass above —
                    # a mismatch here would emit a different number of elements
                    # than the array header declares and desync the connection.
                    var kp1 = sp2[].keys[unsafe_offset=slot].as_string_safe(mbuf)
                    if not _glob_match(pat_p, pat_l, 0, kp1.unsafe_offset(ns_len), klen - ns_len, 0):
                        continue
                if ns_len == 0:
                    writer.append_bulk_value_response(sp2[].keys[unsafe_offset=slot])
                else:
                    var kp = sp2[].keys[unsafe_offset=slot].as_string_safe(kbuf)
                    writer.append_bulk_string_response(kp.unsafe_offset(ns_len), klen - ns_len)
    kbuf.unsafe_free()
    mbuf.unsafe_free()
    return consumed


@always_inline
def handle_randomkey(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """RANDOMKEY — return a random key from the keyspace."""
    var found_rk = False
    for shard_i in range(8):
        if found_rk: break
        var sp3 = keyspace[].shards.unsafe_offset(shard_i)
        for slot in range(sp3[].capacity):
            var m = sp3[].metadata[unsafe_offset=slot]
            if m != SlabHashMap.EMPTY and m != SlabHashMap.DELETED:
                writer.append_bulk_value_response(sp3[].keys[unsafe_offset=slot])
                found_rk = True; break
    if not found_rk: writer.append_null_response()
    return 0


@always_inline
def handle_touch(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """TOUCH key [key ...] — return count of existing keys."""
    var touch_count = 0
    var j_tc = i + 1
    while j_tc < num_tokens:
        var k_str2 = tokens[unsafe_offset=j_tc].value()
        var k_v2 = GenericValue.borrow(tokens[unsafe_offset=j_tc].ptr, tokens[unsafe_offset=j_tc].length)
        if not keyspace[].get(k_v2).is_none(): touch_count += 1
        j_tc += 1
    writer.append_int_response(Int64(touch_count))
    return num_tokens - 1 - i


struct ParkedWait(Copyable, Movable, ImplicitlyCopyable):
    """gh #390: a WAIT that could not be answered yet."""
    var fd: Int32
    var num_req: Int
    var target: UInt64      # WAL offset every counted replica must have applied
    var deadline_ns: Int64  # 0 = no deadline: `WAIT n 0` waits until n replicas ack

    def __init__(out self, fd: Int32, num_req: Int, target: UInt64, deadline_ns: Int64):
        self.fd = fd
        self.num_req = num_req
        self.target = target
        self.deadline_ns = deadline_ns


struct ParkedWaits(Movable):
    """gh #390: per-worker parked WAIT clients. A parked connection gets no
    reply and runs nothing more until the event loop answers its WAIT
    (`NetworkEngine._service_parked_waits`); bytes it sends meanwhile stay in
    its receive buffer. `flags[fd]` makes the per-buffer check one byte load."""
    var entries: List[ParkedWait]
    var flags: Pointer[UInt8, MutUntrackedOrigin]
    # Connections parked by park_fd (an XREAD BLOCK): they have no
    # entry, so `count()` does not see them; `any()` does.
    var fd_parks: Int

    def __init__(out self):
        self.entries = List[ParkedWait]()
        self.flags = alloc[UInt8](65536)
        for i in range(65536):
            self.flags[unsafe_offset=i] = 0
        self.fd_parks = 0

    @always_inline
    def count(self) -> Int:
        return len(self.entries)

    @always_inline
    def any(self) -> Bool:
        """Any connection parked, by WAIT or by park_fd: the dispatch loop's
        one-compare guard before the per-fd flag load."""
        return len(self.entries) > 0 or self.fd_parks > 0

    @always_inline
    def is_parked(self, fd: Int) -> Bool:
        return fd >= 0 and fd < 65536 and self.flags[unsafe_offset=fd] != 0

    def park(mut self, w: ParkedWait):
        self.entries.append(w)
        self.flags[unsafe_offset=Int(w.fd)] = 1

    def park_fd(mut self, fd: Int32):
        """Park a connection with no WAIT entry: an XREAD BLOCK,
        answered by NetworkEngine._service_blocked_readers."""
        if fd >= 0 and fd < 65536 and self.flags[unsafe_offset=Int(fd)] == 0:
            self.flags[unsafe_offset=Int(fd)] = 2
            self.fd_parks += 1

    def unpark_fd(mut self, fd: Int32):
        """Undo park_fd."""
        if fd >= 0 and fd < 65536 and self.flags[unsafe_offset=Int(fd)] == 2:
            self.flags[unsafe_offset=Int(fd)] = 0
            self.fd_parks -= 1

    def unpark_at(mut self, k: Int):
        """Remove entry k (swap with the last)."""
        self.flags[unsafe_offset=Int(self.entries[k].fd)] = 0
        var last = len(self.entries) - 1
        if k != last:
            self.entries[k] = self.entries[last]
        _ = self.entries.pop()

    def remove_fd(mut self, fd: Int32):
        """The connection closed while parked: drop its WAIT or its park."""
        if not self.is_parked(Int(fd)):
            return
        if self.flags[unsafe_offset=Int(fd)] == 2:
            self.unpark_fd(fd)
            return
        var k = 0
        while k < len(self.entries):
            if self.entries[k].fd == fd:
                self.unpark_at(k)
            else:
                k += 1
        self.flags[unsafe_offset=Int(fd)] = 0


@always_inline
def handle_wait(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin], cluster: Pointer[ClusterState, MutUntrackedOrigin], fd: Int32, mut parked: ParkedWaits, can_park: Bool) raises -> Bool:
    """WAIT numreplicas timeout → :N, the replicas that have APPLIED every write
    made before this command (gh #390).

    When fewer than `numreplicas` have applied it yet, the connection is
    PARKED and this returns True without writing a reply: the event loop
    answers once enough replicas ACK or the timeout passes, measured on the
    clock, and only then runs anything the client pipelined behind WAIT. It
    used to wait here, polling with usleep inside the event loop, so one WAIT
    stalled every other client of the worker for up to its timeout. A timeout
    of 0 now means what it means in Redis: wait until the replicas ACK.

    It answers at once, with the count as it stands, where it cannot park:
    inside MULTI/EXEC (Redis's CLIENT_DENY_BLOCKING rule), on the XDP lane,
    and on a server with no replication primary (no replica can ever ACK)."""
    var num_req: Int = 0
    var timeout_ms: Int = 0
    if i + 1 < num_tokens:
        try: num_req = strict_atol(tokens[unsafe_offset=i + 1].value())
        except: pass
    if i + 2 < num_tokens:
        try: timeout_ms = strict_atol(tokens[unsafe_offset=i + 2].value())
        except: pass

    var acked = 0
    if is_not_null(cluster) and is_not_null(cluster[].repl_primary_handle):
        var h = cluster[].repl_primary_handle
        # The target is read through the replicator, never through a pointer
        # into a WAL mapping that rotation or SAVE may have replaced.
        var target = external_call["pion_repl_primary_current_tail", UInt64](h)
        acked = Int(external_call["pion_repl_primary_acked_count", Int32](h, target))
        if acked < num_req and can_park and timeout_ms >= 0:
            var deadline = Int64(0)
            if timeout_ms > 0:
                deadline = _get_now_ns() + Int64(timeout_ms) * 1_000_000
            parked.park(ParkedWait(fd, num_req, target, deadline))
            return True
    writer.append_int_response(Int64(acked))
    return False


@always_inline
def handle_waitaof(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """WAITAOF numlocal numreplicas timeout."""
    var consumed = 0
    if i + 3 < num_tokens: consumed = 3
    var waof_resp = "*2\r\n:0\r\n:0\r\n"
    writer.append_to_response(waof_resp.unsafe_ptr(), waof_resp.byte_length())
    return consumed
