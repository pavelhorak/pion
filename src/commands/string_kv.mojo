"""String/KV commands (slow path): SET variants, APPEND, STRLEN, GETRANGE, SETRANGE, GETSET, GETDEL, GETEX, UNLINK, MSETNX, INCRBY, DECRBY, INCRBYFLOAT, EXPIRETIME, PEXPIRETIME, multi-DEL."""
from src.common.container_free import free_container
from std.ffi import external_call
from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.collections import Array
from std.memory import alloc, unsafe_memcpy, unsafe_memset
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.fast_path import _get_now_ns
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.utils import strict_atol, arg_eq, acc_digit_checked, format_int_to_buf, format_float_to_buf, format_float64_to_buf, int_string_len, parse_filter_float, parse_float64, parse_int64_strict, is_valid_float_arg, parse_redis_double, DOUBLE_LONG, set_expiry, SetExpiry, SETEXP_INVALID, SETEXP_EXPIRED
from src.network.dispatcher import CommandDispatcher
from src.io.wal import WAL


@always_inline
def handle_incrby(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """INCRBY key delta → new integer value."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var _dparse = parse_int64_strict(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        var delta: Int64 = _dparse.value
        var new_val: Int64 = 0
        var valid = _dparse.ok
        if not valid: pass
        # gh #232: WRONGTYPE for an aggregate, but only after the ARGUMENT
        # check — Redis validates the delta first (`INCRBY <list> abc` is ERR,
        # `INCRBY <list> 1` is WRONGTYPE), and that order is observable.
        elif val.is_container():
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 2
        elif val.is_none(): new_val = delta
        elif val.type.value == ValueType.INT:
            var cur = val.as_int()
            if (delta > 0 and cur > 9223372036854775807 - delta) or (delta < 0 and cur < -9223372036854775808 - delta):
                valid = False
            else:
                new_val = cur + delta
        elif val.is_string():
            var length = val.string_len()
            var _sbuf = alloc[UInt8](24)
            var ptr = val.as_string_safe(_sbuf)
            var parsed_val: Int64 = 0; var is_neg = False; var start_idx = 0
            if length > 0 and ptr[unsafe_offset=0] == 45: is_neg = True; start_idx = 1
            if start_idx >= length: valid = False
            for j in range(start_idx, length):
                var c = Int(ptr[unsafe_offset=j])
                if c >= 48 and c <= 57:
                    parsed_val = acc_digit_checked(parsed_val, Int64(c - 48), is_neg, j + 1 == length)
                    if parsed_val == -1: valid = False; break
                else: valid = False; break
            if valid:
                if is_neg: parsed_val = -parsed_val
                if (delta > 0 and parsed_val > 9223372036854775807 - delta) or (delta < 0 and parsed_val < -9223372036854775808 - delta):
                    valid = False
                else:
                    new_val = parsed_val + delta
            _sbuf.unsafe_free()
        else: valid = False
        if valid:
            keyspace[].set(key_v, GenericValue.from_int(new_val))
            var _wb = alloc[UInt8](32)
            var _wl = format_int_to_buf(_wb, 0, new_val)
            _ = wal[].append_kv(1, key_str.unsafe_ptr(), key_str.byte_length(), _wb, _wl)
            _wb.unsafe_free()
            writer.append_int_response(new_val)
        else: writer.append_error_response("ERR value is not an integer or out of range")
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'incrby' command")
        return 0


@always_inline
def handle_decrby(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """DECRBY key delta → new integer value."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var _dparse = parse_int64_strict(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        var delta: Int64 = _dparse.value
        var new_val: Int64 = 0
        var valid = _dparse.ok
        if not valid: pass
        elif val.is_container():   # gh #232, argument-first order as in handle_incrby
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 2
        elif val.is_none(): new_val = -delta
        elif val.type.value == ValueType.INT:
            var cur = val.as_int()
            if (delta > 0 and cur < -9223372036854775808 + delta) or (delta < 0 and cur > 9223372036854775807 + delta):
                valid = False
            else:
                new_val = cur - delta
        elif val.is_string():
            var length = val.string_len()
            var _sbuf = alloc[UInt8](24)
            var ptr = val.as_string_safe(_sbuf)
            var parsed_val: Int64 = 0; var is_neg = False; var start_idx = 0
            if length > 0 and ptr[unsafe_offset=0] == 45: is_neg = True; start_idx = 1
            if start_idx >= length: valid = False
            for j in range(start_idx, length):
                var c = Int(ptr[unsafe_offset=j])
                if c >= 48 and c <= 57:
                    parsed_val = acc_digit_checked(parsed_val, Int64(c - 48), is_neg, j + 1 == length)
                    if parsed_val == -1: valid = False; break
                else: valid = False; break
            if valid:
                if is_neg: parsed_val = -parsed_val
                if (delta > 0 and parsed_val < -9223372036854775808 + delta) or (delta < 0 and parsed_val > 9223372036854775807 + delta):
                    valid = False
                else:
                    new_val = parsed_val - delta
            _sbuf.unsafe_free()
        else: valid = False
        if valid:
            keyspace[].set(key_v, GenericValue.from_int(new_val))
            var _wb = alloc[UInt8](32)
            var _wl = format_int_to_buf(_wb, 0, new_val)
            _ = wal[].append_kv(1, key_str.unsafe_ptr(), key_str.byte_length(), _wb, _wl)
            _wb.unsafe_free()
            writer.append_int_response(new_val)
        else: writer.append_error_response("ERR value is not an integer or out of range")
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'decrby' command")
        return 0


@always_inline
def handle_incrbyfloat(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """INCRBYFLOAT key delta → bulk string of new float value."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        # Redis's incrbyfloatCommand order: the key's type, then the current
        # value, then the increment.
        # A container would otherwise be OVERWRITTEN by the new float —
        # measured: `INCRBYFLOAT <list>` turned a 3-element list into the
        # string "1.5" and orphaned the SlabList. gh #232: a BITMAP is a
        # string, so it goes on to the float parse and fails there with "not
        # a valid float", as in Redis.
        if (not val.is_none() and not val.is_string_like()
                and val.type.value != ValueType.INT
                and val.type.value != ValueType.FLOAT):
            writer.append_error_response(
                "WRONGTYPE Operation against a key holding the wrong kind of value")
            return 2
        # The arithmetic is Redis's: long double, in C (pion_ld_incr), so the
        # range and the printed digits are the Redis-for-this-platform ones.
        # gh #232 / #393 still hold — the stored value is validated as
        # strictly as the argument, a non-finite result is refused and the
        # value left alone — they now live in one place.
        var cur_kind = 0
        var cur_d: Float64 = 0.0
        var cur_l = 0
        var cbuf = alloc[UInt8](32)
        var cur_p = cbuf
        if val.type.value == ValueType.FLOAT:
            cur_kind = 2
            cur_d = Float64(val.as_float())
        elif val.type.value == ValueType.INT:
            cur_kind = 1
            cur_l = format_int_to_buf(cbuf, 0, val.as_int())
        elif val.is_string_like():
            cur_kind = 1
            cur_p = val.as_string_safe(cbuf)
            cur_l = val.string_len()
        var fbuf = alloc[UInt8](5200)    # Redis's MAX_LONG_DOUBLE_CHARS, and some
        var flen = Int(external_call["pion_ld_incr", Int64](
            Int32(cur_kind), cur_p, Int64(cur_l), cur_d,
            tokens[unsafe_offset=i+2].ptr, Int64(tokens[unsafe_offset=i+2].length),
            fbuf, Int64(5200)))
        cbuf.unsafe_free()
        if flen == -3:
            writer.append_error_response("ERR increment would produce NaN or Infinity")
        elif flen < 0:
            writer.append_error_response("ERR value is not a valid float")
        else:
            keyspace[].set(key_v, GenericValue.from_ptr(fbuf, flen))
            _ = wal[].append_kv(1, key_str.unsafe_ptr(), key_str.byte_length(), fbuf, flen)
            writer.append_bulk_string_response(fbuf, flen)
        fbuf.unsafe_free()
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'incrbyfloat' command")
        return 0


@always_inline
def handle_append(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """APPEND key value → integer length of new string."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var old_gv = keyspace[].get(key_v)
        var old_len = 0
        # gh #232: BITMAP is string-like; this branch copies via `copy_to`
        # into a fresh buffer and never frees the original, so it is safe
        # for an arena-backed payload (gh #163).
        if not old_gv.is_none() and (old_gv.is_string_like() or old_gv.type.value == ValueType.INT):
            if old_gv.type.value == ValueType.INT:
                var ibuf = alloc[UInt8](32)
                var ibuf_p = ibuf
                var ilen = format_int_to_buf(ibuf_p, 0, old_gv.as_int())
                var new_app_len = ilen + tokens[unsafe_offset=i+2].length
                var app_buf = alloc[UInt8](new_app_len)
                unsafe_memcpy(dest=app_buf, src=ibuf, count=ilen)
                unsafe_memcpy(dest=app_buf.unsafe_offset(ilen), src=tokens[unsafe_offset=i+2].ptr, count=tokens[unsafe_offset=i+2].length)
                var app_gv = GenericValue.from_ptr(app_buf, new_app_len)
                keyspace[].set(key_v, app_gv)
                _ = wal[].append_kv(1, key_str.unsafe_ptr(), key_str.byte_length(), app_buf, new_app_len)
                writer.append_int_response(Int64(new_app_len))
                ibuf.unsafe_free(); app_buf.unsafe_free()
            else:
                old_len = old_gv.string_len()
                var append_len = tokens[unsafe_offset=i+2].length
                var new_len = old_len + append_len
                var new_buf = alloc[UInt8](new_len)
                var new_buf_p = new_buf
                old_gv.copy_to(new_buf_p)
                unsafe_memcpy(dest=new_buf.unsafe_offset(old_len), src=tokens[unsafe_offset=i+2].ptr, count=append_len)
                var new_gv2 = GenericValue.from_ptr(new_buf_p, new_len)
                keyspace[].set(key_v, new_gv2)
                _ = wal[].append_kv(1, key_str.unsafe_ptr(), key_str.byte_length(), new_buf, new_len)
                writer.append_int_response(Int64(new_len))
                new_buf.unsafe_free()
        elif old_gv.is_none():
            # Missing key: APPEND behaves as SET. This is correct Redis
            # semantics and is the ONLY case this branch may handle.
            var append_len = tokens[unsafe_offset=i+2].length
            var new_gv3 = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, append_len)
            keyspace[].set(key_v, new_gv3)
            _ = wal[].append_kv(1, key_str.unsafe_ptr(), key_str.byte_length(), tokens[unsafe_offset=i+2].ptr, append_len)
            writer.append_int_response(Int64(append_len))
        else:
            # The key holds a LIST/HASH/SET/ZSET/STREAM. This branch used to be
            # shared with the missing-key case, so APPEND silently REPLACED the
            # container with a string — measured: a stream at XLEN 1 became
            # `TYPE string` with the appended bytes as its value, the entries
            # gone. The container allocation is orphaned at the same moment
            # (no GC — the caller must deallocate), so it leaks as well.
            writer.append_error_response(
                "WRONGTYPE Operation against a key holding the wrong kind of value")
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'append' command")
        return 0


@always_inline
def handle_strlen(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """STRLEN key → integer length of string value."""
    if i + 1 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        if val.is_none(): writer.append_int_response(Int64(0))
        # gh #232: is_string_like, not is_string — a bitmap IS a string in
        # Redis, so `SETBIT k 10 1; STRLEN k` must answer 2, not WRONGTYPE.
        # string_len() already reads _data1 for anything non-SSO, which is
        # exactly a BITMAP's byte length; only the type check excluded it.
        elif val.is_string_like(): writer.append_int_response(Int64(val.string_len()))
        elif val.type.value == ValueType.INT:
            var ibuf2 = alloc[UInt8](32)
            var ilen2 = format_int_to_buf(ibuf2, 0, val.as_int())
            writer.append_int_response(Int64(ilen2))
            ibuf2.unsafe_free()
        else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'strlen' command")
        return 0


@always_inline
def handle_getset(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut dispatcher: CommandDispatcher,
                  ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] = null_ptr[SlabHashMap, MutUntrackedOrigin](),
                  wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """GETSET key value → old value (bulk string or nil)."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var old_val = keyspace[].get(key_v)
        if not old_val.is_none() and old_val.is_container():
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 2
        # Reply FIRST: `old_val` borrows the payload the SET below frees, so a
        # reply written afterwards served freed memory — the first 8 bytes of
        # every heap-sized old value came back as an allocator pointer.
        if old_val.is_none(): writer.append_null_response()
        else: writer.append_bulk_value_response(old_val)
        var new_str = tokens[unsafe_offset=i+2].raw_value()
        dispatcher.execute_set(key_str, new_str)
        # GETSET is a SET: Redis discards any existing TTL. This arm went
        # through execute_set, which does not touch ttl_map, so the key kept
        # the PREVIOUS value's deadline — the caller writes a fresh value and
        # it silently vanishes at the old expiry. (Fast-path SET clears the TTL
        # itself, which is why plain SET was correct and only GETSET was not.)
        if is_not_null(ttl_map) and not ttl_map[].get(key_v).is_none():
            _ = ttl_map[].remove_generic(key_v)
            if is_not_null(wal):
                _ = wal[].append(26, key_str.unsafe_ptr(), key_str.byte_length())
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'getset' command")
        return 0


@always_inline
def handle_getdel(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """GETDEL key → old value (bulk string or nil), then delete."""
    if i + 1 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var old_val = keyspace[].get(key_v)
        if old_val.is_none(): writer.append_null_response()
        # gh #232: BITMAP is string-like — `SETBIT k 0 1; GETDEL k` answered
        # WRONGTYPE where Redis returns the bytes. Read-only + delete, so
        # there is no arena hazard here at all.
        elif not old_val.is_string_like() and old_val.type.value != ValueType.INT:
            # The WRONGTYPE reply used to come from append_bulk_value_response's
            # own type dispatch — while the remove_generic below ran ANYWAY. So
            # `GETDEL <list>` answered "WRONGTYPE, refused" and deleted the list
            # regardless: the client is told nothing happened and the data is
            # gone. Refuse before touching the keyspace.
            writer.append_error_response(
                "WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            writer.append_bulk_value_response(old_val)
            _ = keyspace[].remove_generic(key_v)
            _ = wal[].append(2, key_str.unsafe_ptr(), key_str.byte_length())
            if is_not_null(ttl_map): _ = ttl_map[].remove_generic(key_v)
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'getdel' command")
        return 0


@always_inline
def handle_getex(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """GETEX key [EX secs | PX ms | EXAT ts | PXAT tsms | PERSIST] → bulk value or nil.

    Validate first, then act. It used to write the value reply and THEN parse
    the option leniently (non-digits skipped), so `GETEX k EX abc` answered
    the value and set a 0-second TTL — the key was gone on the next read.
    Options also matched on three bytes + length."""
    if i + 1 < num_tokens:
        var mode = 0            # 0 none, 1 relative ns, 2 absolute ns, 3 persist
        var amount: Int64 = 0
        if i + 2 < num_tokens:
            var op = tokens[unsafe_offset=i+2].ptr
            var ol = tokens[unsafe_offset=i+2].length
            if arg_eq(op, ol, "persist") and i + 3 == num_tokens:
                mode = 3
            elif i + 4 == num_tokens and (arg_eq(op, ol, "ex") or arg_eq(op, ol, "px")
                                          or arg_eq(op, ol, "exat") or arg_eq(op, ol, "pxat")):
                var pv = parse_int64_strict(tokens[unsafe_offset=i+3].ptr, tokens[unsafe_offset=i+3].length)
                if not pv.ok:
                    writer.append_error_response("ERR value is not an integer or out of range")
                    return num_tokens - 1 - i
                var is_s = arg_eq(op, ol, "ex") or arg_eq(op, ol, "exat")
                if pv.value <= 0 or pv.value > (Int64(9223372036854) if is_s else Int64(9223372036854775)):
                    writer.append_error_response("ERR invalid expire time in 'getex' command")
                    return num_tokens - 1 - i
                amount = pv.value * (Int64(1_000_000_000) if is_s else Int64(1_000_000))
                mode = 1 if (arg_eq(op, ol, "ex") or arg_eq(op, ol, "px")) else 2
            else:
                writer.append_error_response("ERR syntax error")
                return num_tokens - 1 - i
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        if val.is_none():
            writer.append_null_response()
        elif not val.is_string_like() and val.type.value != ValueType.INT and val.type.value != ValueType.FLOAT:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            writer.append_bulk_value_response(val)
            if is_not_null(ttl_map):
                if mode == 3:
                    _ = ttl_map[].remove_generic(key_v)
                elif mode == 1:
                    ttl_map[].set(key_v.clone(), GenericValue.from_int(_get_now_ns() + amount))
                elif mode == 2:
                    ttl_map[].set(key_v.clone(), GenericValue.from_int(amount))
        key_v.free_str_payload()
        return num_tokens - 1 - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'getex' command")
        return 0


@always_inline
def handle_setnx(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut dispatcher: CommandDispatcher) raises -> Int:
    """SETNX key value → 1 if set, 0 if key already exists."""
    if i + 2 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var existing = keyspace[].get(key_v)
        if existing.is_none():
            dispatcher.execute_set(key_str, tokens[unsafe_offset=i+2].raw_value())
            writer.append_int_response(Int64(1))
        else: writer.append_int_response(Int64(0))
        return 2
    else:
        writer.append_error_response("ERR wrong number of arguments for 'setnx' command")
        return 0


@always_inline
def _setex_apply(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, mut dispatcher: CommandDispatcher,
                 ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin], key_str: String, val_str: String,
                 ex: SetExpiry) raises:
    """SETEX / PSETEX once the TTL is known to be legal. A TTL whose deadline
    Redis computes as already past (a relative time overflowing ms, gh #393)
    leaves the key ABSENT — Redis writes it already expired — and still +OK."""
    if ex.status == SETEXP_EXPIRED:
        _ = dispatcher.execute_del(key_str)
        if is_not_null(ttl_map):
            _ = ttl_map[].remove_generic(GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length))
        return
    dispatcher.execute_set(key_str, val_str)
    if is_not_null(ttl_map):
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        ttl_map[].set(key_v, GenericValue.from_int(ex.ns))


@always_inline
def handle_setex(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, mut dispatcher: CommandDispatcher, ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """SETEX key seconds value → +OK."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var secs_str = tokens[unsafe_offset=i+2].value()
        var val_str = tokens[unsafe_offset=i+3].raw_value()
        # Strict (gh #229's rule reaching SETEX): the loop this replaces SKIPPED
        # non-digits, so `SETEX k -5 v` set a 5 s TTL and `SETEX k abc v` a 0 s
        # one, both replying +OK. Redis refuses both, and writes nothing.
        var _sx = parse_int64_strict(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        if not _sx.ok:
            writer.append_error_response("ERR value is not an integer or out of range")
            return 3
        var _now = _get_now_ns()
        var _ex = set_expiry(_sx.value, 1000, True, _now)
        if _ex.status == SETEXP_INVALID:
            writer.append_error_response("ERR invalid expire time in 'setex' command")
            return 3
        _setex_apply(tokens, i, dispatcher, ttl_map, key_str, val_str, _ex)
        writer.append_ok_response()
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'setex' command")
        return 0


@always_inline
def handle_psetex(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, mut dispatcher: CommandDispatcher, ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """PSETEX key milliseconds value → +OK."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var ms_str2 = tokens[unsafe_offset=i+2].value()
        var val_str2 = tokens[unsafe_offset=i+3].raw_value()
        var _px = parse_int64_strict(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
        if not _px.ok:
            writer.append_error_response("ERR value is not an integer or out of range")
            return 3
        var _now = _get_now_ns()
        var _ex = set_expiry(_px.value, 1, True, _now)
        if _ex.status == SETEXP_INVALID:
            writer.append_error_response("ERR invalid expire time in 'psetex' command")
            return 3
        _setex_apply(tokens, i, dispatcher, ttl_map, key_str, val_str2, _ex)
        writer.append_ok_response()
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'psetex' command")
        return 0


@always_inline
def handle_msetex(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, mut dispatcher: CommandDispatcher, ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """R3: MSETEX seconds key1 val1 key2 val2 ... → +OK. Atomic multi-SET with shared TTL."""
    # Need at least: MSETEX seconds key val = 4 tokens (i is at MSETEX)
    if i + 3 < num_tokens:
        # Parse seconds
        var secs_str = tokens[unsafe_offset=i+1].value()
        var _mx = parse_int64_strict(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        if not _mx.ok or _mx.value <= 0 or _mx.value > 9223372036854:
            writer.append_error_response("ERR invalid expire time in 'msetex' command")
            return num_tokens - 1 - i
        var secs_i: Int64 = _mx.value
        var expiry_ns = _get_now_ns() + secs_i * Int64(1_000_000_000)
        # Iterate key-value pairs starting from token i+2
        var j = i + 2
        var consumed = 1  # MSETEX + seconds = 2 tokens, but we return extra consumed
        while j + 1 < num_tokens:
            var key_str = tokens[unsafe_offset=j].value()
            var val_str = tokens[unsafe_offset=j+1].raw_value()
            dispatcher.execute_set(key_str, val_str)
            if is_not_null(ttl_map):
                var key_gv = GenericValue.borrow(tokens[unsafe_offset=j].ptr, tokens[unsafe_offset=j].length)
                ttl_map[].set(key_gv, GenericValue.from_int(expiry_ns))
            j += 2; consumed += 2
        writer.append_ok_response()
        return consumed
    else:
        writer.append_error_response("ERR wrong number of arguments for 'msetex' command")
        return 0


@always_inline
def handle_msetnx(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut dispatcher: CommandDispatcher) raises -> Int:
    """MSETNX key value [key value ...] → 1 if all set, 0 if any exist."""
    if i + 2 < num_tokens and (num_tokens - i - 1) % 2 == 0:
        # Check all keys first
        var all_new = True
        var j_ms = i + 1
        while j_ms < num_tokens:
            var chk_key = tokens[unsafe_offset=j_ms].value()
            var chk_v = GenericValue.borrow(tokens[unsafe_offset=j_ms].ptr, tokens[unsafe_offset=j_ms].length)
            if not keyspace[].get(chk_v).is_none(): all_new = False; break
            j_ms += 2
        if all_new:
            var j_ms2 = i + 1
            while j_ms2 + 1 < num_tokens:
                dispatcher.execute_set(tokens[unsafe_offset=j_ms2].value(), tokens[unsafe_offset=j_ms2+1].raw_value())
                j_ms2 += 2
            writer.append_int_response(Int64(1))
        else:
            writer.append_int_response(Int64(0))
        return num_tokens - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'msetnx' command")
        return 0


@always_inline
def handle_getrange(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """GETRANGE key start end → bulk string substring."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var start_i = strict_atol(tokens[unsafe_offset=i+2].value())
        var end_i = strict_atol(tokens[unsafe_offset=i+3].value())
        # gh #232: is_string_like — a bitmap IS a string, so a range over one
        # must return its bytes. It was falling into the empty-reply branch.
        if val.is_none():
            writer.append_bulk_string_response(String("").unsafe_ptr(), 0)
        elif not val.is_string_like() and val.type.value != ValueType.INT:
            # A missing key and a WRONG-TYPE key are different answers. Both
            # replied "" here, which is the dangerous direction: the caller is
            # told the range is empty when the real fault is a type error at
            # the call site, so it surfaces as missing data rather than a
            # stack trace.
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            var slen = val.string_len()
            if val.type.value == ValueType.INT: slen = int_string_len(val.as_int())
            if start_i < 0: start_i += slen
            if end_i < 0: end_i += slen
            if start_i < 0: start_i = 0
            # gh #232: end was clamped at the TOP but not at the bottom, so an
            # out-of-range negative end stayed negative and the `start > end`
            # test below rejected the whole range. Measured on "hello world":
            # GETRANGE k -100 -99 gave "" where Redis gives "h" (both offsets
            # clamp to 0, so the range is [0,0]). Symmetric with start_i above.
            if end_i < 0: end_i = 0
            if end_i >= slen: end_i = slen - 1
            if start_i > end_i or slen == 0:
                writer.append_bulk_string_response(String("").unsafe_ptr(), 0)
            else:
                var range_len = end_i - start_i + 1
                var range_buf = alloc[UInt8](range_len)
                var range_buf_p = range_buf
                if val.type.value == ValueType.INT:
                    var ibuf3 = alloc[UInt8](32)
                    _ = format_int_to_buf(ibuf3, 0, val.as_int())
                    unsafe_memcpy(dest=range_buf, src=ibuf3.unsafe_offset(start_i), count=range_len)
                    ibuf3.unsafe_free()
                else:
                    var _sbuf = alloc[UInt8](24)
                    var src_p = val.as_string_safe(_sbuf)
                    unsafe_memcpy(dest=range_buf, src=src_p.unsafe_offset(start_i), count=range_len)
                    _sbuf.unsafe_free()
                writer.append_bulk_string_response(range_buf_p, range_len)
                range_buf.unsafe_free()
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'getrange' command")
        return 0


@always_inline
def handle_substr(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """SUBSTR key start end → bulk string substring (comptime for GETRANGE)."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var val = keyspace[].get(key_v)
        var start_s = strict_atol(tokens[unsafe_offset=i+2].value())
        var end_s = strict_atol(tokens[unsafe_offset=i+3].value())
        # gh #232: is_string_like — a bitmap IS a string, so a range over one
        # must return its bytes. It was falling into the empty-reply branch.
        if val.is_none():
            writer.append_bulk_string_response(String("").unsafe_ptr(), 0)
        elif not val.is_string_like() and val.type.value != ValueType.INT:
            # A missing key and a WRONG-TYPE key are different answers. Both
            # replied "" here, which is the dangerous direction: the caller is
            # told the range is empty when the real fault is a type error at
            # the call site, so it surfaces as missing data rather than a
            # stack trace.
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            var slen2 = val.string_len()
            if start_s < 0: start_s += slen2
            if end_s < 0: end_s += slen2
            if start_s < 0: start_s = 0
            if end_s >= slen2: end_s = slen2 - 1
            if start_s > end_s or slen2 == 0:
                writer.append_bulk_string_response(String("").unsafe_ptr(), 0)
            else:
                var rlen2 = end_s - start_s + 1
                var rbuf2 = alloc[UInt8](rlen2 + slen2)
                var rbuf2_p = rbuf2
                val.copy_to(rbuf2_p)
                writer.append_bulk_string_response(rbuf2_p.unsafe_offset(start_s), rlen2)
                rbuf2.unsafe_free()
        return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'substr' command")
        return 0


@always_inline
def handle_setrange(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """SETRANGE key offset value → integer new length."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var old_gv2 = keyspace[].get(key_v)
        var offset_i = strict_atol(tokens[unsafe_offset=i+2].value())
        if offset_i < 0:
            writer.append_error_response("ERR offset is negative")
            return 3
        else:
            var val_tok = tokens[unsafe_offset=i+3]
            var val_len2 = val_tok.length
            # Redis caps a string at proto-max-bulk-len (512 MB). Without the
            # cap `offset + len` overflowed Int64 for offset 2^63-1: the buffer
            # was sized from the WRAPPED length and the value written at
            # ptr + offset — one command, an out-of-bounds heap write (found by
            # tests/test_numeric_args_differential.py; the corruption surfaced
            # later, in an unrelated RPUSH's free()).
            if val_len2 > 0 and offset_i > 536870912 - val_len2:
                writer.append_error_response("ERR string exceeds maximum allowed size (proto-max-bulk-len)")
                return 3
            # gh #232: an EMPTY value is a no-op and must not create the key.
            # `SETRANGE missing 0 ""` returned 0 but MATERIALIZED an empty
            # string, so a later EXISTS said 1 where Redis says 0 — a key that
            # only exists because someone measured it.
            if val_len2 == 0:
                var _srl = 0
                if not old_gv2.is_none() and old_gv2.is_string_like():
                    _srl = old_gv2.string_len()
                writer.append_int_response(Int64(_srl))
                return 3
            var old_len2 = 0
            # Same defect APPEND had: the existing type was consulted ONLY to
            # size the copy, then `keyspace[].set` overwrote unconditionally —
            # so SETRANGE on a list/hash/set/zset/stream silently replaced the
            # container with a string and orphaned its allocation. Refuse.
            # gh #232: BITMAP is string-like. Copies out below, never frees in
            # place, so an arena-backed payload is safe.
            if (not old_gv2.is_none() and not old_gv2.is_string_like()
                    and old_gv2.type.value != ValueType.INT):
                writer.append_error_response(
                    "WRONGTYPE Operation against a key holding the wrong kind of value")
                return 3
            if not old_gv2.is_none() and old_gv2.is_string_like(): old_len2 = old_gv2.string_len()
            var new_len2 = offset_i + val_len2
            if old_len2 > new_len2: new_len2 = old_len2
            var new_buf2 = alloc[UInt8](new_len2)
            var new_buf2_p = new_buf2
            unsafe_memset(new_buf2_p, 0, new_len2)
            if old_len2 > 0: old_gv2.copy_to(new_buf2_p)
            unsafe_memcpy(dest=new_buf2.unsafe_offset(offset_i), src=val_tok.ptr, count=val_len2)
            var sr_gv = GenericValue.from_ptr(new_buf2_p, new_len2)
            keyspace[].set(key_v, sr_gv)
            _ = wal[].append_kv(1, key_str.unsafe_ptr(), key_str.byte_length(), new_buf2, new_len2)
            writer.append_int_response(Int64(new_len2))
            new_buf2.unsafe_free()
            return 3
    else:
        writer.append_error_response("ERR wrong number of arguments for 'setrange' command")
        return 0


@always_inline
def handle_expiretime(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """EXPIRETIME key → unix timestamp in seconds when key expires (-1=no TTL, -2=not found)."""
    if i + 1 < num_tokens and is_not_null(ttl_map):
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var existing = keyspace[].get(key_v)
        if existing.is_none(): writer.append_int_response(Int64(-2))
        else:
            var exp_v = ttl_map[].get(key_v)
            if exp_v.is_none(): writer.append_int_response(Int64(-1))
            else: writer.append_int_response(exp_v.as_int() / Int64(1_000_000_000))
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'expiretime' command")
        return 0


@always_inline
def handle_pexpiretime(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """PEXPIRETIME key → unix timestamp in milliseconds when key expires (-1=no TTL, -2=not found)."""
    if i + 1 < num_tokens and is_not_null(ttl_map):
        var key_str = tokens[unsafe_offset=i+1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        var existing = keyspace[].get(key_v)
        if existing.is_none(): writer.append_int_response(Int64(-2))
        else:
            var exp_v = ttl_map[].get(key_v)
            if exp_v.is_none(): writer.append_int_response(Int64(-1))
            else: writer.append_int_response(exp_v.as_int() / Int64(1_000_000))
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'pexpiretime' command")
        return 0


@always_inline
def handle_unlink(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises -> Int:
    """UNLINK key [key ...] → integer count of deleted keys (same as DEL)."""
    var del_count = 0
    var j_ul = i + 1
    while j_ul < num_tokens:
        var k_str = tokens[unsafe_offset=j_ul].value()
        var k_v = GenericValue.borrow(tokens[unsafe_offset=j_ul].ptr, tokens[unsafe_offset=j_ul].length)
        var ul_taken = GenericValue()
        if keyspace[].remove_generic_taking(k_v, ul_taken):
            free_container(ul_taken)   # gh #369
            del_count += 1
            _ = wal[].append(2, k_str.unsafe_ptr(), k_str.byte_length())
        if is_not_null(ttl_map): _ = ttl_map[].remove_generic(k_v)
        j_ul += 1
    writer.append_int_response(Int64(del_count))
    return num_tokens - i - 1


def _string_bytes(val: GenericValue, scratch: Pointer[UInt8, MutUntrackedOrigin],
                  mut out_len: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    """A read-only view of a string value's bytes: a heap STRING or BITMAP in
    place, a short string or an integer written into `scratch` (>= 32 bytes)."""
    if val.type.value == ValueType.INT:
        out_len = format_int_to_buf(scratch, 0, val.as_int())
        return scratch
    return val.bitmap_view(scratch, out_len)


def handle_lcs(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
               mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) -> Int:
    """LCS key1 key2 [LEN] [IDX] [MINMATCHLEN len] [WITHMATCHLEN] (#39).

    Redis's lcsCommand: both keys must hold strings (a missing key is the
    empty string), then the options, then the work (src/ffi/redis_ports.c).
    `num_tokens` is the command's end."""
    if num_tokens - i < 3:
        writer.append_error_response("ERR wrong number of arguments for 'lcs' command")
        return 0
    var ka = tokens[unsafe_offset=i + 1]
    var kb = tokens[unsafe_offset=i + 2]
    var va = keyspace[].get(GenericValue.borrow(ka.ptr, ka.length))
    var vb = keyspace[].get(GenericValue.borrow(kb.ptr, kb.length))
    var a_ok = va.is_none() or va.is_string_like() or va.type.value == ValueType.INT
    var b_ok = vb.is_none() or vb.is_string_like() or vb.type.value == ValueType.INT
    if not a_ok or not b_ok:
        writer.append_error_response("ERR The specified keys must contain string values")
        return 0
    var getidx = False
    var getlen = False
    var withmatchlen = False
    var minmatchlen = Int64(0)
    var j = i + 3
    while j < num_tokens:
        var o = tokens[unsafe_offset=j]
        var more = num_tokens - 1 - j
        if arg_eq(o.ptr, o.length, "idx"):
            getidx = True
        elif arg_eq(o.ptr, o.length, "len"):
            getlen = True
        elif arg_eq(o.ptr, o.length, "withmatchlen"):
            withmatchlen = True
        elif arg_eq(o.ptr, o.length, "minmatchlen") and more > 0:
            var m = tokens[unsafe_offset=j + 1]
            var r = parse_int64_strict(m.ptr, m.length)
            if not r.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return 0
            minmatchlen = r.value if r.value > 0 else Int64(0)
            j += 1
        else:
            writer.append_error_response("ERR syntax error")
            return 0
        j += 1
    if getidx and getlen:
        writer.append_error_response("ERR If you want both the length and indexes, please just use IDX.")
        return 0
    var sa = alloc[UInt8](32)
    var sb = alloc[UInt8](32)
    var alen = 0
    var blen = 0
    var pa = sa
    var pb = sb
    if not va.is_none():
        pa = _string_bytes(va, sa, alen)
    if not vb.is_none():
        pb = _string_bytes(vb, sb, blen)
    var out = alloc[Pointer[UInt8, MutUntrackedOrigin]](1)
    var mode = Int64(2) if getidx else (Int64(1) if getlen else Int64(0))
    var n = external_call["pion_lcs", Int64](pa, Int64(alen), pb, Int64(blen), mode, minmatchlen,
                                             Int64(1) if withmatchlen else Int64(0), Int64(Int(writer.proto)), out)
    var reply = out[unsafe_offset=0]
    if n < 0:
        writer.append_error_response("ERR Insufficient memory, failed allocating transient memory for LCS")
    else:
        writer.append_to_response(reply, Int(n))
    external_call["pion_lcs_free", NoneType](reply)
    out.unsafe_free()
    sa.unsafe_free()
    sb.unsafe_free()
    return 0
