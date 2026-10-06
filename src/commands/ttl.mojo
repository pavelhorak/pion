"""TTL/Expiry commands: EXPIRE, PEXPIRE, EXPIREAT, PEXPIREAT, TTL, PTTL, PERSIST."""
from src.common.utils import strict_atol, ms_to_deadline_ns, I64_MAX
from src.common.container_free import remove_and_free
from src.common.ptr import is_not_null, null_ptr
from src.io.wal import WAL
from std.memory.unsafe_pointer import Pointer
from std.collections import Array
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.fast_path import _get_now_ns
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue


# An absolute deadline is nanoseconds in an Int64, so it saturates around
# 2262. Past that `now + secs * 1_000_000_000` WRAPPED: measured on 0.915,
# `EXPIRE k 9300000000` left the key present while TTL answered -2, and
# `EXPIRE k 99999999999` reported a TTL of ~246 y. The overflow checks are now
# Redis's own, done in ms (`_expire_generic`), and a deadline past 2262 is
# stored saturated (`ms_to_deadline_ns`) instead of wrapped.


@always_inline
def _delete_expired_now(key: String, key_gv: GenericValue,
                        keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                        ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                        wal: Pointer[WAL, MutUntrackedOrigin]) raises:
    """Drop the key now and log it, for the non-positive-TTL path."""
    _ = remove_and_free(keyspace, key_gv)   # gh #394: an aggregate's container too
    if is_not_null(ttl_map):
        _ = ttl_map[].remove_generic(key_gv)
    if is_not_null(wal):
        _ = wal[].append(2, key.unsafe_ptr(), key.byte_length())


@always_inline
def _parse_expire_flag(tokens: Pointer[RESP3Token, MutUntrackedOrigin], idx: Int, num_tokens: Int) -> Int:
    """0 = no flag, 1 = NX, 2 = XX, 3 = GT, 4 = LT, -1 = unrecognised.

    These were parsed by nobody before gh #236 — the token was simply ignored,
    so every conditional form applied unconditionally."""
    if idx >= num_tokens:
        return 0
    var tp = tokens[unsafe_offset=idx].ptr
    if tokens[unsafe_offset=idx].length != 2:
        return -1
    var a = tp[unsafe_offset=0] | 0x20
    var b = tp[unsafe_offset=1] | 0x20
    if a == 110 and b == 120: return 1      # nx
    if a == 120 and b == 120: return 2      # xx
    if a == 103 and b == 116: return 3      # gt
    if a == 108 and b == 116: return 4      # lt
    return -1


@always_inline
def _expire_flag_allows(flag: Int, has_ttl: Bool, cur_ns: Int64, new_ns: Int64) -> Bool:
    """Redis conditional-expire semantics. A key with NO expiry counts as an
    INFINITE ttl, which is why GT always fails on a persistent key and LT
    always succeeds on one."""
    if flag == 1: return not has_ttl                        # NX
    if flag == 2: return has_ttl                            # XX
    if flag == 3: return has_ttl and new_ns > cur_ns        # GT
    if flag == 4: return (not has_ttl) or new_ns < cur_ns   # LT
    return True


@always_inline
def _expire_generic(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                    wal: Pointer[WAL, MutUntrackedOrigin], unit_ms: Int64, relative: Bool, name: StaticString) raises -> Int:
    """EXPIRE / PEXPIRE / EXPIREAT / PEXPIREAT → 1 if set (or the key was
    deleted by a deadline already past), 0 if the key is missing or the
    NX | XX | GT | LT condition failed.

    Redis's order, which is observable (gh #393): the flag, then the integer,
    then overflow, and only then the key. Overflow is Redis's own arithmetic in
    ms — `EXPIRE k -9223372036854775808` is an error (it cannot be scaled to
    ms), not a delete; `PEXPIREAT k 9223372036854775807` is a legal deadline."""
    if i + 2 >= num_tokens:
        writer.append_error_response(String("ERR wrong number of arguments for '") + String(name) + "' command")
        return 0
    var _flag = _parse_expire_flag(tokens, i + 3, num_tokens)
    if _flag < 0:
        writer.append_error_response("ERR Unsupported option")
        return 2
    var key = tokens[unsafe_offset=i+1].value()
    var when = Int64(strict_atol(tokens[unsafe_offset=i+2].value()))
    if unit_ms != 1:
        if when > I64_MAX // unit_ms or when < -(I64_MAX // unit_ms):   # C truncates; `//` floors
            writer.append_error_response(String("ERR invalid expire time in '") + String(name) + "' command")
            return 2
        when = when * unit_ms
    var now_ns = _get_now_ns()
    if relative:
        var now_ms = now_ns // 1_000_000
        if when > I64_MAX - now_ms:
            writer.append_error_response(String("ERR invalid expire time in '") + String(name) + "' command")
            return 2
        when = when + now_ms
    var key_gv = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
    if keyspace[].get(key_gv).is_none():
        writer.append_int_response(Int64(0))
        return 2
    var _new_ns = ms_to_deadline_ns(when)
    var _cur = ttl_map[].get(key_gv) if is_not_null(ttl_map) else GenericValue()
    var _has_ttl = not _cur.is_none()
    var _cur_ns = _cur.as_int() if _has_ttl else Int64(0)
    if not _expire_flag_allows(_flag, _has_ttl, _cur_ns, _new_ns):
        # gh #236: the conditional forms were applied unconditionally, so
        # `EXPIRE k 100 GT` — the extend-never-shorten idiom — SHRANK a
        # 300 s lease to 100 s.
        writer.append_int_response(Int64(0))
    elif _new_ns <= now_ns:
        # Redis deletes the key at once for a deadline already past and still
        # replies 1. Pion used to set a past deadline and leave the key, so
        # `EXPIRE k 0` — the documented delete idiom — kept the value.
        _delete_expired_now(key, key_gv, keyspace, ttl_map, wal)
        writer.append_int_response(Int64(1))
    else:
        if is_not_null(ttl_map):
            ttl_map[].set(key_gv, GenericValue.from_int(_new_ns))
        # gh #174: log the RESOLVED absolute deadline. `EXPIRE key 60`
        # logged as "60 seconds" would replay as 60 s after recovery,
        # silently extending every TTL by the length of the outage.
        if is_not_null(wal):
            _ = wal[].append_u64_val(25, key.unsafe_ptr(), key.byte_length(),
                                     UInt64(_new_ns),
                                     null_ptr[UInt8, MutUntrackedOrigin](), 0)
        writer.append_int_response(Int64(1))
    return 2


@always_inline
def handle_expire(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                 wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """EXPIRE key seconds [NX|XX|GT|LT]."""
    return _expire_generic(tokens, i, num_tokens, writer, keyspace, ttl_map, wal, 1000, True, "expire")


@always_inline
def handle_pexpire(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                  wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """PEXPIRE key milliseconds [NX|XX|GT|LT]."""
    return _expire_generic(tokens, i, num_tokens, writer, keyspace, ttl_map, wal, 1, True, "pexpire")


@always_inline
def handle_expireat(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                   wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """EXPIREAT key unix-seconds [NX|XX|GT|LT] (the flags were not parsed here before)."""
    return _expire_generic(tokens, i, num_tokens, writer, keyspace, ttl_map, wal, 1000, False, "expireat")


@always_inline
def handle_pexpireat(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                    wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """PEXPIREAT key unix-ms [NX|XX|GT|LT]."""
    return _expire_generic(tokens, i, num_tokens, writer, keyspace, ttl_map, wal, 1, False, "pexpireat")


@always_inline
def handle_ttl(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """TTL key → seconds remaining (-1=no TTL, -2=not found)."""
    if i + 1 < num_tokens:
        var key = tokens[unsafe_offset=i+1].value()
        var key_gv = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        if keyspace[].get(key_gv).is_none():
            writer.append_int_response(Int64(-2))
        elif is_not_null(ttl_map):
            var exp_gv = ttl_map[].get(key_gv)
            if exp_gv.is_none():
                writer.append_int_response(Int64(-1))
            else:
                # #45: measured from the batch clock, as Redis measures from
                # its command time snapshot
                var _now = keyspace[].clock_ns if keyspace[].clock_ns != 0 else _get_now_ns()
                var remaining_ns = exp_gv.as_int() - _now
                if remaining_ns <= 0:
                    writer.append_int_response(Int64(-2))
                else:
                    # gh #236: Redis ROUNDS the remaining time to the nearest
                    # second rather than truncating, so `EXPIRE k 100; TTL k`
                    # answers 100, not 99. Truncating made every TTL read one
                    # second short, which a client doing `if TTL < n: renew`
                    # sees as the lease expiring early.
                    writer.append_int_response(
                        (Int64(remaining_ns) + Int64(500_000_000)) // Int64(1_000_000_000))
        else:
            writer.append_int_response(Int64(-1))
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'ttl' command")
        return 0


@always_inline
def handle_pttl(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """PTTL key → milliseconds remaining (-1=no TTL, -2=not found)."""
    if i + 1 < num_tokens:
        var key = tokens[unsafe_offset=i+1].value()
        var key_gv = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        if keyspace[].get(key_gv).is_none():
            writer.append_int_response(Int64(-2))
        elif is_not_null(ttl_map):
            var exp_gv = ttl_map[].get(key_gv)
            if exp_gv.is_none():
                writer.append_int_response(Int64(-1))
            else:
                var _now = keyspace[].clock_ns if keyspace[].clock_ns != 0 else _get_now_ns()   # #45
                var remaining_ns = exp_gv.as_int() - _now
                if remaining_ns <= 0:
                    writer.append_int_response(Int64(-2))
                else:
                    writer.append_int_response(Int64(remaining_ns) // Int64(1_000_000))
        else:
            writer.append_int_response(Int64(-1))
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'pttl' command")
        return 0


@always_inline
def handle_persist(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                   keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                   wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """PERSIST key → 1 if TTL removed, 0 if no TTL or key not found. The key
    is looked up first: an expired key is gone (#45), and removing its TTL
    alone used to bring it back."""
    if i + 1 < num_tokens:
        var key = tokens[unsafe_offset=i+1].value()
        var key_gv = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
        if not keyspace[].get(key_gv).is_none() and is_not_null(ttl_map) and not ttl_map[].get(key_gv).is_none():
            _ = ttl_map[].remove_generic(key_gv)
            if is_not_null(wal):
                _ = wal[].append(26, key.unsafe_ptr(), key.byte_length())
            writer.append_int_response(Int64(1))
        else:
            writer.append_int_response(Int64(0))
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'persist' command")
        return 0
