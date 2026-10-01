"""Lua 5.1 scripting engine for Pion.

Coroutine-based execution: redis.call() yields to host for command dispatch.
One LuaEngine per worker (shared-nothing).
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.container_free import remove_and_free
from std.memory import alloc, unsafe_memcpy, stack_allocation
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.common.value import GenericValue, ValueType
from src.common.utils import format_int_to_buf
from src.commands.command_table import command_is_denyoom
from src.memory.object_pool import ObjectPool
from src.network.response_writer import ResponseWriter
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from std.collections import Array


# Status codes from lua_wrap.c
comptime PION_LUA_OK = 0
comptime PION_LUA_NEEDS_CMD = 1
comptime PION_LUA_ERROR = -1

# Result buffer for Lua script output
comptime LUA_RESULT_BUF_SIZE = 65536


struct LuaEngine(Movable):
    """Per-worker Lua 5.1 VM with script cache and sandboxed execution."""

    var state: Pointer[NoneType, MutUntrackedOrigin]  # PionLuaState*
    var enabled: Bool
    var result_buf: Pointer[UInt8, MutUntrackedOrigin]
    # Scratch buffer for reading string values from GenericValue
    var val_buf: Pointer[UInt8, MutUntrackedOrigin]

    def __init__(out self):
        self.state = external_call["pion_lua_new_state",
            Pointer[NoneType, MutUntrackedOrigin]](
            Int32(1048576),   # 1MB memory limit
            Int32(1000000),   # 1M instruction limit
        )
        self.enabled = is_not_null(self.state)
        self.result_buf = alloc[UInt8](LUA_RESULT_BUF_SIZE)
        self.val_buf = alloc[UInt8](1024)

    def __init__(out self, *, deinit take: Self):
        self.state = take.state
        self.enabled = take.enabled
        self.result_buf = take.result_buf
        self.val_buf = take.val_buf


# ── Helper: read call arg from Lua coroutine stack ──

@always_inline
def _lua_arg_ptr(state: Pointer[NoneType, MutUntrackedOrigin], idx: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    """Get pointer to idx-th call argument (0 = command name)."""
    var p = external_call["pion_lua_call_arg_ptr",
        Pointer[UInt8, MutUntrackedOrigin]](state, Int32(idx))
    return p

@always_inline
def _lua_arg_len(state: Pointer[NoneType, MutUntrackedOrigin], idx: Int) -> Int:
    """Get length of idx-th call argument."""
    return Int(external_call["pion_lua_call_arg_len", Int32](state, Int32(idx)))


# ── Helper: parse integer from string bytes ──

struct IntParseResult:
    var value: Int64
    var valid: Bool

    def __init__(out self, value: Int64, valid: Bool):
        self.value = value
        self.valid = valid


@always_inline
def _parse_int(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> IntParseResult:
    """Parse integer from byte buffer. Returns (value, valid)."""
    if length == 0:
        return IntParseResult(Int64(0), False)
    var val: Int64 = 0
    var neg = (length > 0 and ptr[unsafe_offset=0] == 45)  # '-'
    var start = 1 if neg else 0
    if start >= length:
        return IntParseResult(Int64(0), False)
    for j in range(start, length):
        var c = Int(ptr[unsafe_offset=j])
        if c >= 48 and c <= 57:
            val = val * 10 + Int64(c - 48)
        else:
            return IntParseResult(Int64(0), False)
    if neg:
        val = -val
    return IntParseResult(val, True)


# ── Command dispatch from redis.call() ──

def _dispatch_lua_cmd(
    state: Pointer[NoneType, MutUntrackedOrigin],
    nargs: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
    ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
    hash_map_pool: Pointer[ObjectPool[SlabHashMap], MutUntrackedOrigin],
    list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin],
    val_buf: Pointer[UInt8, MutUntrackedOrigin],
) -> Int32:
    """Dispatch a redis.call() command. Returns resume status code."""
    if nargs < 1:
        return external_call["pion_lua_push_error_and_resume", Int32](
            state,
            "ERR wrong number of arguments".unsafe_ptr(),
            Int32(34),
        )

    var cmd_ptr = _lua_arg_ptr(state, 0)
    var cmd_len = _lua_arg_len(state, 0)

    if cmd_len == 0:
        return external_call["pion_lua_push_error_and_resume", Int32](
            state,
            "ERR unknown command ''".unsafe_ptr(),
            Int32(21),
        )

    # gh #261: Redis runs a script under --maxmemory and refuses only the
    # memory-growing commands it CALLS (scripts carry no flags here), so a
    # read-only script keeps working. cmd_ptr is Lua-owned heap memory.
    if command_is_denyoom(cmd_ptr, cmd_len) \
       and external_call["pion_maxmemory_check", Int32]() != 0:
        return _push_err(state, "OOM command not allowed when used memory > 'maxmemory'.")

    # Case-insensitive first byte
    var b0 = cmd_ptr[unsafe_offset=0] | 0x20

    # ── GET (3) ──
    if b0 == 103 and cmd_len == 3 and (cmd_ptr[unsafe_offset=1] | 0x20) == 101 and (cmd_ptr[unsafe_offset=2] | 0x20) == 116:
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments for 'get' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_nil_and_resume", Int32](state)
        elif val.type.value == ValueType.INT:
            # gh #87.6: stack_allocation skips the leak-on-error window the
            # alloc/free pair has when the C bridge raises mid-call.
            var ibuf = stack_allocation[24, UInt8]()
            var ival = val.as_int()
            var ilen = format_int_to_buf(ibuf, 0, ival)
            return external_call["pion_lua_push_string_and_resume", Int32](
                state, ibuf, Int32(ilen))
        elif val.is_string():
            var slen = val.string_len()
            val.copy_to(val_buf)
            return external_call["pion_lua_push_string_and_resume", Int32](
                state, val_buf, Int32(slen))
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── SET (3) ──
    elif b0 == 115 and cmd_len == 3 and (cmd_ptr[unsafe_offset=1] | 0x20) == 101 and (cmd_ptr[unsafe_offset=2] | 0x20) == 116:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'set' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var vp = _lua_arg_ptr(state, 2)
        var vl = _lua_arg_len(state, 2)
        var key = GenericValue.borrow(kp, kl)
        var val = GenericValue.borrow(vp, vl)
        keyspace[].set(key, val)
        return external_call["pion_lua_push_ok_and_resume", Int32](state)

    # ── DEL (3) ──
    elif b0 == 100 and cmd_len == 3 and (cmd_ptr[unsafe_offset=1] | 0x20) == 101 and (cmd_ptr[unsafe_offset=2] | 0x20) == 108:
        var count: Int64 = 0
        for a in range(1, nargs):
            var kp = _lua_arg_ptr(state, a)
            var kl = _lua_arg_len(state, a)
            var key = GenericValue.borrow(kp, kl)
            if remove_and_free(keyspace, key):   # gh #394: free a DEL'd aggregate
                count += 1
                # Also remove TTL if present
                if is_not_null(ttl_map):
                    _ = ttl_map[].remove_generic(key)
        return external_call["pion_lua_push_int_and_resume", Int32](state, count)

    # ── EXISTS (6) ──
    elif b0 == 101 and cmd_len == 6 and (cmd_ptr[unsafe_offset=1] | 0x20) == 120:
        var count: Int64 = 0
        for a in range(1, nargs):
            var kp = _lua_arg_ptr(state, a)
            var kl = _lua_arg_len(state, a)
            var key = GenericValue.borrow(kp, kl)
            if not keyspace[].get(key).is_none():
                count += 1
        return external_call["pion_lua_push_int_and_resume", Int32](state, count)

    # ── INCR (4) / DECR (4) ──
    elif (b0 == 105 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 110 and (cmd_ptr[unsafe_offset=2] | 0x20) == 99 and (cmd_ptr[unsafe_offset=3] | 0x20) == 114) or (b0 == 100 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 101 and (cmd_ptr[unsafe_offset=2] | 0x20) == 99 and (cmd_ptr[unsafe_offset=3] | 0x20) == 114):
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments")
        var delta: Int64 = Int64(1) if b0 == 105 else Int64(-1)
        return _do_incrby(state, keyspace, val_buf, 1, delta)

    # ── INCRBY (6) / DECRBY (6) ──
    elif (b0 == 105 and cmd_len == 6 and (cmd_ptr[unsafe_offset=1] | 0x20) == 110) or (b0 == 100 and cmd_len == 6 and (cmd_ptr[unsafe_offset=1] | 0x20) == 101):
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments")
        var dp = _lua_arg_ptr(state, 2)
        var dl = _lua_arg_len(state, 2)
        var delta_r = _parse_int(dp, dl)
        if not delta_r.valid:
            return _push_err(state, "ERR value is not an integer or out of range")
        var delta = delta_r.value
        if b0 == 100:
            delta = -delta
        return _do_incrby(state, keyspace, val_buf, 1, delta)

    # ── EXPIRE (6) ──
    elif b0 == 101 and cmd_len == 6 and (cmd_ptr[unsafe_offset=1] | 0x20) == 120 and (cmd_ptr[unsafe_offset=2] | 0x20) == 112:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'expire' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var sp = _lua_arg_ptr(state, 2)
        var sl = _lua_arg_len(state, 2)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        var secs_r = _parse_int(sp, sl)
        if not secs_r.valid:
            return _push_err(state, "ERR value is not an integer or out of range")
        if is_not_null(ttl_map):
            # Store expiry as nanoseconds from now (monotonic)
            var now_ns = external_call["pion_get_unix_time", Int64]() * 1000000000
            var expiry_ns = now_ns + secs_r.value * 1000000000
            ttl_map[].set(key, GenericValue.from_int(expiry_ns))
        return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(1))

    # ── TTL (3) ──
    elif b0 == 116 and cmd_len == 3 and (cmd_ptr[unsafe_offset=1] | 0x20) == 116 and (cmd_ptr[unsafe_offset=2] | 0x20) == 108:
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments for 'ttl' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(-2))
        if is_not_null(ttl_map):
            var ttl_val = ttl_map[].get(key)
            if not ttl_val.is_none() and ttl_val.type.value == ValueType.INT:
                var now_ns = external_call["pion_get_unix_time", Int64]() * 1000000000
                var remaining = (ttl_val.as_int() - now_ns) // 1000000000
                if remaining <= 0:
                    return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(-2))
                return external_call["pion_lua_push_int_and_resume", Int32](state, remaining)
        return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(-1))

    # ── PERSIST (7) ──
    elif b0 == 112 and cmd_len == 7 and (cmd_ptr[unsafe_offset=1] | 0x20) == 101 and (cmd_ptr[unsafe_offset=2] | 0x20) == 114:
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments for 'persist' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        if is_not_null(ttl_map):
            var removed = ttl_map[].remove_generic(key)
            return external_call["pion_lua_push_int_and_resume", Int32](
                state, Int64(1) if removed else Int64(0))
        return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))

    # ── HSET (4) ──
    elif b0 == 104 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 115 and (cmd_ptr[unsafe_offset=2] | 0x20) == 101 and (cmd_ptr[unsafe_offset=3] | 0x20) == 116:
        if nargs < 4 or (nargs - 2) % 2 != 0:
            return _push_err(state, "ERR wrong number of arguments for 'hset' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        var count: Int64 = 0
        if val.is_none():
            var hash_ptr: Pointer[SlabHashMap, MutUntrackedOrigin]
            if hash_map_pool[].head < hash_map_pool[].capacity:
                hash_ptr = hash_map_pool[].acquire(); hash_ptr[].reset()
            else:
                hash_ptr = alloc[SlabHashMap](1); hash_ptr.unsafe_write(SlabHashMap(16))
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.HASH)
            new_val.set_ptr(hash_ptr.unsafe_bitcast[NoneType]())
            keyspace[].set(key, new_val)
            for f in range(2, nargs, 2):
                var fp = _lua_arg_ptr(state, f)
                var fl = _lua_arg_len(state, f)
                var vp = _lua_arg_ptr(state, f + 1)
                var vl = _lua_arg_len(state, f + 1)
                hash_ptr[].set(GenericValue.borrow(fp, fl), GenericValue.borrow(vp, vl))
                count += 1
        elif val.type.value == ValueType.HASH:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            for f in range(2, nargs, 2):
                var fp = _lua_arg_ptr(state, f)
                var fl = _lua_arg_len(state, f)
                var vp = _lua_arg_ptr(state, f + 1)
                var vl = _lua_arg_len(state, f + 1)
                hash_ptr[].set(GenericValue.borrow(fp, fl), GenericValue.borrow(vp, vl))
                count += 1
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")
        return external_call["pion_lua_push_int_and_resume", Int32](state, count)

    # ── HGET (4) ──
    elif b0 == 104 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 103 and (cmd_ptr[unsafe_offset=2] | 0x20) == 101 and (cmd_ptr[unsafe_offset=3] | 0x20) == 116:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'hget' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_nil_and_resume", Int32](state)
        elif val.type.value == ValueType.HASH:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var fp = _lua_arg_ptr(state, 2)
            var fl = _lua_arg_len(state, 2)
            var fkey = GenericValue.borrow(fp, fl)
            var fval = hash_ptr[].get(fkey)
            if fval.is_none():
                return external_call["pion_lua_push_nil_and_resume", Int32](state)
            return _push_generic_value(state, fval, val_buf)
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── HDEL (4) ──
    elif b0 == 104 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 100 and (cmd_ptr[unsafe_offset=2] | 0x20) == 101 and (cmd_ptr[unsafe_offset=3] | 0x20) == 108:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'hdel' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        elif val.type.value == ValueType.HASH:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var count: Int64 = 0
            for a in range(2, nargs):
                var fp = _lua_arg_ptr(state, a)
                var fl = _lua_arg_len(state, a)
                if hash_ptr[].remove_generic(GenericValue.borrow(fp, fl)):
                    count += 1
            return external_call["pion_lua_push_int_and_resume", Int32](state, count)
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── HEXISTS (7) ──
    elif b0 == 104 and cmd_len == 7 and (cmd_ptr[unsafe_offset=1] | 0x20) == 101 and (cmd_ptr[unsafe_offset=2] | 0x20) == 120:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'hexists' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        elif val.type.value == ValueType.HASH:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var fp = _lua_arg_ptr(state, 2)
            var fl = _lua_arg_len(state, 2)
            var fval = hash_ptr[].get(GenericValue.borrow(fp, fl))
            var exists = Int64(0) if fval.is_none() else Int64(1)
            return external_call["pion_lua_push_int_and_resume", Int32](state, exists)
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── HLEN (4) ──
    elif b0 == 104 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 108 and (cmd_ptr[unsafe_offset=2] | 0x20) == 101 and (cmd_ptr[unsafe_offset=3] | 0x20) == 110:
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments for 'hlen' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        elif val.type.value == ValueType.HASH:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(hash_ptr[].size))
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── LPUSH (5) / RPUSH (5) ──
    elif (b0 == 108 or b0 == 114) and cmd_len == 5 and (cmd_ptr[unsafe_offset=1] | 0x20) == 112 and (cmd_ptr[unsafe_offset=2] | 0x20) == 117 and (cmd_ptr[unsafe_offset=3] | 0x20) == 115 and (cmd_ptr[unsafe_offset=4] | 0x20) == 104:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments")
        var is_lpush = b0 == 108
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        var list_ptr: Pointer[SlabList, MutUntrackedOrigin]
        if val.is_none():
            list_ptr = list_pool[].acquire(); list_ptr[].reset()
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.LIST)
            new_val.set_ptr(list_ptr.unsafe_bitcast[NoneType]())
            keyspace[].set(key, new_val)
        elif val.type.value == ValueType.LIST:
            list_ptr = val.as_list().unsafe_bitcast[SlabList]()
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")
        for a in range(2, nargs):
            var vp = _lua_arg_ptr(state, a)
            var vl = _lua_arg_len(state, a)
            var elem = GenericValue.from_ptr(vp, vl)
            if is_lpush:
                list_ptr[].lpush(elem)
            else:
                list_ptr[].rpush(elem)
        return external_call["pion_lua_push_int_and_resume", Int32](
            state, Int64(list_ptr[].llen()))

    # ── LPOP (4) / RPOP (4) ──
    elif (b0 == 108 or b0 == 114) and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 112 and (cmd_ptr[unsafe_offset=2] | 0x20) == 111 and (cmd_ptr[unsafe_offset=3] | 0x20) == 112:
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments")
        var is_lpop = b0 == 108
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_nil_and_resume", Int32](state)
        elif val.type.value == ValueType.LIST:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            if list_ptr[].llen() == 0:
                return external_call["pion_lua_push_nil_and_resume", Int32](state)
            var elem: GenericValue
            if is_lpop:
                elem = list_ptr[].lpop()
            else:
                elem = list_ptr[].rpop()
            return _push_generic_value(state, elem, val_buf)
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── LLEN (4) ──
    elif b0 == 108 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 108 and (cmd_ptr[unsafe_offset=2] | 0x20) == 101 and (cmd_ptr[unsafe_offset=3] | 0x20) == 110:
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments for 'llen' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        elif val.type.value == ValueType.LIST:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            return external_call["pion_lua_push_int_and_resume", Int32](
                state, Int64(list_ptr[].llen()))
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── LRANGE (6) ──
    elif b0 == 108 and cmd_len == 6 and (cmd_ptr[unsafe_offset=1] | 0x20) == 114 and (cmd_ptr[unsafe_offset=2] | 0x20) == 97:
        if nargs < 4:
            return _push_err(state, "ERR wrong number of arguments for 'lrange' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none() or val.type.value != ValueType.LIST:
            # Empty array for non-existent key
            external_call["pion_lua_array_begin", NoneType](state)
            return external_call["pion_lua_array_end_and_resume", Int32](state)

        var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
        var sp = _lua_arg_ptr(state, 2)
        var sl = _lua_arg_len(state, 2)
        var ep = _lua_arg_ptr(state, 3)
        var el = _lua_arg_len(state, 3)
        var start_r = _parse_int(sp, sl)
        var end_r = _parse_int(ep, el)
        if not start_r.valid or not end_r.valid:
            return _push_err(state, "ERR value is not an integer or out of range")

        var llen = list_ptr[].llen()
        var start = Int(start_r.value)
        var end = Int(end_r.value)
        if start < 0:
            start = llen + start
        if end < 0:
            end = llen + end
        if start < 0:
            start = 0
        if end >= llen:
            end = llen - 1
        if start > end:
            external_call["pion_lua_array_begin", NoneType](state)
            return external_call["pion_lua_array_end_and_resume", Int32](state)

        var elements = list_ptr[].lrange(start, end)
        external_call["pion_lua_array_begin", NoneType](state)
        for idx in range(len(elements)):
            var elem = elements[idx]
            if elem.is_none():
                external_call["pion_lua_array_push_nil", NoneType](state)
            elif elem.is_string():
                var slen = elem.string_len()
                elem.copy_to(val_buf)
                external_call["pion_lua_array_push_string", NoneType](
                    state, val_buf, Int32(slen))
            elif elem.type.value == ValueType.INT:
                # gh #87.6: stack_allocation (see HGET handler note).
                var ibuf = stack_allocation[24, UInt8]()
                var ilen = format_int_to_buf(ibuf, 0, elem.as_int())
                external_call["pion_lua_array_push_string", NoneType](
                    state, ibuf, Int32(ilen))
            else:
                external_call["pion_lua_array_push_nil", NoneType](state)
        return external_call["pion_lua_array_end_and_resume", Int32](state)

    # ── SADD (4) ──
    elif b0 == 115 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 97 and (cmd_ptr[unsafe_offset=2] | 0x20) == 100 and (cmd_ptr[unsafe_offset=3] | 0x20) == 100:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'sadd' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        var count: Int64 = 0
        if val.is_none():
            var set_ptr: Pointer[SlabHashMap, MutUntrackedOrigin]
            if hash_map_pool[].head < hash_map_pool[].capacity:
                set_ptr = hash_map_pool[].acquire(); set_ptr[].reset()
            else:
                set_ptr = alloc[SlabHashMap](1); set_ptr.unsafe_write(SlabHashMap(16))
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.SET)
            new_val.set_ptr(set_ptr.unsafe_bitcast[NoneType]())
            keyspace[].set(key, new_val)
            for a in range(2, nargs):
                var mp = _lua_arg_ptr(state, a)
                var ml = _lua_arg_len(state, a)
                set_ptr[].set(GenericValue.borrow(mp, ml), GenericValue.from_int(1))
                count += 1
        elif val.type.value == ValueType.SET:
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            for a in range(2, nargs):
                var mp = _lua_arg_ptr(state, a)
                var ml = _lua_arg_len(state, a)
                var member = GenericValue.borrow(mp, ml)
                if set_ptr[].get(member).is_none():
                    set_ptr[].set(member, GenericValue.from_int(1))
                    count += 1
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")
        return external_call["pion_lua_push_int_and_resume", Int32](state, count)

    # ── SREM (4) ──
    elif b0 == 115 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 114 and (cmd_ptr[unsafe_offset=2] | 0x20) == 101 and (cmd_ptr[unsafe_offset=3] | 0x20) == 109:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'srem' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        elif val.type.value == ValueType.SET:
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            var count: Int64 = 0
            for a in range(2, nargs):
                var mp = _lua_arg_ptr(state, a)
                var ml = _lua_arg_len(state, a)
                if set_ptr[].remove_generic(GenericValue.borrow(mp, ml)):
                    count += 1
            return external_call["pion_lua_push_int_and_resume", Int32](state, count)
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── SISMEMBER (9) ──
    elif b0 == 115 and cmd_len == 9:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'sismember' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        elif val.type.value == ValueType.SET:
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            var mp = _lua_arg_ptr(state, 2)
            var ml = _lua_arg_len(state, 2)
            var exists = not set_ptr[].get(GenericValue.borrow(mp, ml)).is_none()
            return external_call["pion_lua_push_int_and_resume", Int32](
                state, Int64(1) if exists else Int64(0))
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── SCARD (5) ──
    elif b0 == 115 and cmd_len == 5 and (cmd_ptr[unsafe_offset=1] | 0x20) == 99 and (cmd_ptr[unsafe_offset=2] | 0x20) == 97 and (cmd_ptr[unsafe_offset=3] | 0x20) == 114 and (cmd_ptr[unsafe_offset=4] | 0x20) == 100:
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments for 'scard' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        elif val.type.value == ValueType.SET:
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            return external_call["pion_lua_push_int_and_resume", Int32](
                state, Int64(set_ptr[].size))
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── TYPE (4) ──
    elif b0 == 116 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 121 and (cmd_ptr[unsafe_offset=2] | 0x20) == 112 and (cmd_ptr[unsafe_offset=3] | 0x20) == 101:
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments for 'type' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return _push_status(state, "none")
        elif val.is_string() or val.type.value == ValueType.INT or val.type.value == ValueType.FLOAT:
            return _push_status(state, "string")
        elif val.type.value == ValueType.HASH:
            return _push_status(state, "hash")
        elif val.type.value == ValueType.LIST:
            return _push_status(state, "list")
        elif val.type.value == ValueType.SET:
            return _push_status(state, "set")
        elif val.type.value == ValueType.ZSET:
            return _push_status(state, "zset")
        else:
            return _push_status(state, "none")

    # ── APPEND (6) ──
    elif b0 == 97 and cmd_len == 6 and (cmd_ptr[unsafe_offset=1] | 0x20) == 112 and (cmd_ptr[unsafe_offset=2] | 0x20) == 112:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'append' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var vp = _lua_arg_ptr(state, 2)
        var vl = _lua_arg_len(state, 2)
        var key = GenericValue.borrow(kp, kl)
        var old = keyspace[].get(key)
        if old.is_none():
            keyspace[].set(key, GenericValue.borrow(vp, vl))
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(vl))
        elif old.is_string():
            var olen = old.string_len()
            var newlen = olen + vl
            var buf = alloc[UInt8](newlen)
            old.copy_to(buf)
            unsafe_memcpy(dest=buf.unsafe_offset(olen), src=vp, count=vl)
            keyspace[].set(key, GenericValue.borrow(buf, newlen))
            buf.unsafe_free()
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(newlen))
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── STRLEN (6) ──
    elif b0 == 115 and cmd_len == 6 and (cmd_ptr[unsafe_offset=1] | 0x20) == 116 and (cmd_ptr[unsafe_offset=2] | 0x20) == 114:
        if nargs < 2:
            return _push_err(state, "ERR wrong number of arguments for 'strlen' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(key)
        if val.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        elif val.is_string():
            return external_call["pion_lua_push_int_and_resume", Int32](
                state, Int64(val.string_len()))
        else:
            return _push_err(state, "WRONGTYPE Operation against a key holding the wrong kind of value")

    # ── SETNX (5) ──
    elif b0 == 115 and cmd_len == 5 and (cmd_ptr[unsafe_offset=1] | 0x20) == 101 and (cmd_ptr[unsafe_offset=2] | 0x20) == 116 and (cmd_ptr[unsafe_offset=3] | 0x20) == 110 and (cmd_ptr[unsafe_offset=4] | 0x20) == 120:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'setnx' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var key = GenericValue.borrow(kp, kl)
        var existing = keyspace[].get(key)
        if not existing.is_none():
            return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(0))
        var vp = _lua_arg_ptr(state, 2)
        var vl = _lua_arg_len(state, 2)
        keyspace[].set(key, GenericValue.borrow(vp, vl))
        return external_call["pion_lua_push_int_and_resume", Int32](state, Int64(1))

    # ── MGET (4) ──
    elif b0 == 109 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 103 and (cmd_ptr[unsafe_offset=2] | 0x20) == 101 and (cmd_ptr[unsafe_offset=3] | 0x20) == 116:
        external_call["pion_lua_array_begin", NoneType](state)
        for a in range(1, nargs):
            var kp = _lua_arg_ptr(state, a)
            var kl = _lua_arg_len(state, a)
            var key = GenericValue.borrow(kp, kl)
            var val = keyspace[].get(key)
            if val.is_none():
                external_call["pion_lua_array_push_nil", NoneType](state)
            elif val.is_string():
                var slen = val.string_len()
                val.copy_to(val_buf)
                external_call["pion_lua_array_push_string", NoneType](
                    state, val_buf, Int32(slen))
            elif val.type.value == ValueType.INT:
                # gh #87.6: stack_allocation (see HGET handler note).
                var ibuf = stack_allocation[24, UInt8]()
                var ilen = format_int_to_buf(ibuf, 0, val.as_int())
                external_call["pion_lua_array_push_string", NoneType](
                    state, ibuf, Int32(ilen))
            else:
                external_call["pion_lua_array_push_nil", NoneType](state)
        return external_call["pion_lua_array_end_and_resume", Int32](state)

    # ── MSET (4) ──
    elif b0 == 109 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 115 and (cmd_ptr[unsafe_offset=2] | 0x20) == 101 and (cmd_ptr[unsafe_offset=3] | 0x20) == 116:
        if nargs < 3 or (nargs - 1) % 2 != 0:
            return _push_err(state, "ERR wrong number of arguments for 'mset' command")
        for a in range(1, nargs, 2):
            var kp = _lua_arg_ptr(state, a)
            var kl = _lua_arg_len(state, a)
            var vp = _lua_arg_ptr(state, a + 1)
            var vl = _lua_arg_len(state, a + 1)
            keyspace[].set(GenericValue.borrow(kp, kl), GenericValue.borrow(vp, vl))
        return external_call["pion_lua_push_ok_and_resume", Int32](state)

    # ── PING (4) ──
    elif b0 == 112 and cmd_len == 4 and (cmd_ptr[unsafe_offset=1] | 0x20) == 105 and (cmd_ptr[unsafe_offset=2] | 0x20) == 110 and (cmd_ptr[unsafe_offset=3] | 0x20) == 103:
        return _push_status(state, "PONG")

    # ── RENAME (6) ──
    elif b0 == 114 and cmd_len == 6 and (cmd_ptr[unsafe_offset=1] | 0x20) == 101 and (cmd_ptr[unsafe_offset=2] | 0x20) == 110:
        if nargs < 3:
            return _push_err(state, "ERR wrong number of arguments for 'rename' command")
        var kp = _lua_arg_ptr(state, 1)
        var kl = _lua_arg_len(state, 1)
        var np = _lua_arg_ptr(state, 2)
        var nl = _lua_arg_len(state, 2)
        var old_key = GenericValue.borrow(kp, kl)
        var val = keyspace[].get(old_key)
        if val.is_none():
            return _push_err(state, "ERR no such key")
        var new_key = GenericValue.borrow(np, nl)
        # gh #123: the new key must own its string payload — remove_generic on
        # the old key frees it, so re-storing `val` itself left the new key
        # pointing at freed memory (same fix key_mgmt.mojo's RENAME carries).
        keyspace[].set(new_key, val.clone())
        _ = keyspace[].remove_generic(old_key)
        return external_call["pion_lua_push_ok_and_resume", Int32](state)

    # ── Unknown command ──
    else:
        # Build error message with command name
        var err_prefix = "ERR unknown command '"
        var err_suffix = "'"
        var err_buf = alloc[UInt8](64 + cmd_len)
        unsafe_memcpy(dest=err_buf, src=err_prefix.unsafe_ptr(), count=err_prefix.byte_length())
        unsafe_memcpy(dest=err_buf.unsafe_offset(err_prefix.byte_length()), src=cmd_ptr, count=cmd_len)
        unsafe_memcpy(dest=err_buf.unsafe_offset(err_prefix.byte_length()).unsafe_offset(cmd_len), src=err_suffix.unsafe_ptr(), count=1)
        var total = err_prefix.byte_length() + cmd_len + 1
        var ret = external_call["pion_lua_push_error_and_resume", Int32](
            state, err_buf, Int32(total))
        err_buf.unsafe_free()
        return ret


# ── Helpers ──

@always_inline
def _push_err(state: Pointer[NoneType, MutUntrackedOrigin], msg: String) -> Int32:
    """Push error and resume."""
    return external_call["pion_lua_push_error_and_resume", Int32](
        state, msg.unsafe_ptr(), Int32(msg.byte_length()))


@always_inline
def _push_status(state: Pointer[NoneType, MutUntrackedOrigin], msg: String) -> Int32:
    """Push status reply as {ok=msg} table and resume."""
    return external_call["pion_lua_push_string_and_resume", Int32](
        state, msg.unsafe_ptr(), Int32(msg.byte_length()))


@always_inline
def _push_generic_value(
    state: Pointer[NoneType, MutUntrackedOrigin],
    val: GenericValue,
    val_buf: Pointer[UInt8, MutUntrackedOrigin],
) -> Int32:
    """Push a GenericValue as appropriate Lua type."""
    if val.is_none():
        return external_call["pion_lua_push_nil_and_resume", Int32](state)
    elif val.type.value == ValueType.INT:
        return external_call["pion_lua_push_int_and_resume", Int32](state, val.as_int())
    elif val.is_string():
        var slen = val.string_len()
        val.copy_to(val_buf)
        return external_call["pion_lua_push_string_and_resume", Int32](
            state, val_buf, Int32(slen))
    else:
        return external_call["pion_lua_push_nil_and_resume", Int32](state)


def _do_incrby(
    state: Pointer[NoneType, MutUntrackedOrigin],
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
    val_buf: Pointer[UInt8, MutUntrackedOrigin],
    key_arg_idx: Int,
    delta: Int64,
) -> Int32:
    """INCR/DECR/INCRBY/DECRBY implementation."""
    var kp = _lua_arg_ptr(state, key_arg_idx)
    var kl = _lua_arg_len(state, key_arg_idx)
    var key = GenericValue.borrow(kp, kl)
    var val = keyspace[].get(key)
    var new_val: Int64 = 0
    if val.is_none():
        new_val = delta
    elif val.type.value == ValueType.INT:
        new_val = val.as_int() + delta
    elif val.is_string():
        var slen = val.string_len()
        val.copy_to(val_buf)
        var parsed = _parse_int(val_buf, slen)
        if not parsed.valid:
            return _push_err(state, "ERR value is not an integer or out of range")
        new_val = parsed.value + delta
    else:
        return _push_err(state, "ERR value is not an integer or out of range")
    keyspace[].set(key, GenericValue.from_int(new_val))
    return external_call["pion_lua_push_int_and_resume", Int32](state, new_val)


# ── EVAL / EVALSHA / SCRIPT command handlers ──

def handle_eval(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
    ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
    hash_map_pool: Pointer[ObjectPool[SlabHashMap], MutUntrackedOrigin],
    list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin],
) -> Int:
    """EVAL script numkeys [key ...] [arg ...]."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra

    # Parse: EVAL script numkeys [key ...] [arg ...]
    # tokens[i] = "EVAL", tokens[i+1] = script, tokens[i+2] = numkeys, ...
    if i + 2 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'eval' command")
        return extra

    var script_ptr = tokens[unsafe_offset=i + 1].ptr
    var script_len = Int(tokens[unsafe_offset=i + 1].length)

    var numkeys_ptr = tokens[unsafe_offset=i + 2].ptr
    var numkeys_len = Int(tokens[unsafe_offset=i + 2].length)
    var nk_parsed = _parse_int(numkeys_ptr, numkeys_len)
    if not nk_parsed.valid:
        writer.append_error_response("ERR value is not an integer or out of range")
        return extra
    var numkeys = Int(nk_parsed.value)
    # gh #410: validate before _set_keys_argv sizes an array from numkeys — an
    # unvalidated huge numkeys makes that alloc fail and abort the process.
    if numkeys < 0:
        writer.append_error_response("ERR Number of keys can't be negative")
        return extra
    if numkeys > num_tokens - i - 3:
        writer.append_error_response("ERR Number of keys can't be greater than number of args")
        return extra

    # Load script. gh #87.6: stack_allocation removes the per-EVAL leak
    # window between `alloc` and either `.unsafe_free()` call below — if the C
    # bridge raises or panics, the heap buffer would leak.
    var sha1_buf = stack_allocation[41, UInt8]()
    var load_result = external_call["pion_lua_load_script", Int32](
        lua[].state, script_ptr, Int32(script_len), sha1_buf)

    if load_result < 0:
        writer.append_error_response("ERR error compiling script")
        return extra

    # Set KEYS and ARGV
    _set_keys_argv(lua[].state, tokens, i, num_tokens, numkeys)

    # Execute
    _exec_and_write_result(lua, sha1_buf, keyspace, ttl_map, hash_map_pool,
                           list_pool, writer)
    return extra


def handle_evalsha(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
    ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
    hash_map_pool: Pointer[ObjectPool[SlabHashMap], MutUntrackedOrigin],
    list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin],
) -> Int:
    """EVALSHA sha1 numkeys [key ...] [arg ...]."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra

    if i + 2 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'evalsha' command")
        return extra

    var sha1_ptr = tokens[unsafe_offset=i + 1].ptr
    var sha1_len = Int(tokens[unsafe_offset=i + 1].length)

    # Check SHA1 exists
    if sha1_len != 40:
        writer.append_error_response("NOSCRIPT No matching script. Use EVAL.")
        return extra

    # gh #87.6: stack_allocation (see EVAL handler note).
    var sha1_buf = stack_allocation[41, UInt8]()
    unsafe_memcpy(dest=sha1_buf, src=sha1_ptr, count=40)
    sha1_buf[unsafe_offset=40] = 0  # null terminate

    var exists = external_call["pion_lua_script_exists", Int32](lua[].state, sha1_buf)
    if exists == 0:
        writer.append_error_response("NOSCRIPT No matching script. Use EVAL.")
        return extra

    var numkeys_ptr = tokens[unsafe_offset=i + 2].ptr
    var numkeys_len = Int(tokens[unsafe_offset=i + 2].length)
    var nk_parsed = _parse_int(numkeys_ptr, numkeys_len)
    if not nk_parsed.valid:
        writer.append_error_response("ERR value is not an integer or out of range")
        return extra
    var numkeys = Int(nk_parsed.value)
    # gh #410: validate before _set_keys_argv sizes an array from numkeys.
    if numkeys < 0:
        writer.append_error_response("ERR Number of keys can't be negative")
        return extra
    if numkeys > num_tokens - i - 3:
        writer.append_error_response("ERR Number of keys can't be greater than number of args")
        return extra

    _set_keys_argv(lua[].state, tokens, i, num_tokens, numkeys)
    _exec_and_write_result(lua, sha1_buf, keyspace, ttl_map, hash_map_pool,
                           list_pool, writer)
    return extra


def handle_script(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
) -> Int:
    """SCRIPT LOAD|EXISTS|FLUSH."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra

    if i + 1 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'script' command")
        return extra

    var sub_ptr = tokens[unsafe_offset=i + 1].ptr
    var s0 = sub_ptr[unsafe_offset=0] | 0x20

    if s0 == 108:  # 'l' - LOAD
        if i + 2 >= num_tokens:
            writer.append_error_response("ERR wrong number of arguments for 'script|load' command")
            return extra
        var script_ptr = tokens[unsafe_offset=i + 2].ptr
        var script_len = Int(tokens[unsafe_offset=i + 2].length)
        # gh #87.6: stack_allocation (see EVAL handler note).
        var sha1_buf = stack_allocation[41, UInt8]()
        var result = external_call["pion_lua_load_script", Int32](
            lua[].state, script_ptr, Int32(script_len), sha1_buf)
        if result < 0:
            _ = external_call["pion_lua_get_error",
                Pointer[UInt8, MutUntrackedOrigin]](lua[].state)
            writer.append_error_response("ERR error compiling script")
        else:
            writer.append_bulk_string_response(sha1_buf, 40)
    elif s0 == 101:  # 'e' - EXISTS
        var cnt = num_tokens - i - 2
        if cnt < 1:
            cnt = 1
        # *N\r\n
        writer.buffer[unsafe_offset=writer.offset] = 42  # '*'
        writer.offset += 1
        writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(cnt))
        writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
        writer.offset += 2
        for a in range(2, 2 + cnt):
            if i + a < num_tokens:
                var sha_ptr = tokens[unsafe_offset=i + a].ptr
                var sha_len = Int(tokens[unsafe_offset=i + a].length)
                if sha_len == 40:
                    # gh #87.6: stack_allocation (see EVAL handler note).
                    var sha_buf = stack_allocation[41, UInt8]()
                    unsafe_memcpy(dest=sha_buf, src=sha_ptr, count=40)
                    sha_buf[unsafe_offset=40] = 0
                    var found = external_call["pion_lua_script_exists", Int32](lua[].state, sha_buf)
                    writer.append_int_response(Int64(found))
                else:
                    writer.append_int_response(Int64(0))
            else:
                writer.append_int_response(Int64(0))
    elif s0 == 102:  # 'f' - FLUSH
        external_call["pion_lua_script_flush", NoneType](lua[].state)
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR unknown SCRIPT subcommand")

    return extra


def handle_function(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
) -> Int:
    """FUNCTION LOAD|LIST|DELETE|FLUSH|STATS|DUMP."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra

    if i + 1 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'function' command")
        return extra

    var sub_ptr = tokens[unsafe_offset=i + 1].ptr
    var sub_len = Int(tokens[unsafe_offset=i + 1].length)
    var s0 = sub_ptr[unsafe_offset=0] | 0x20

    if s0 == 108 and sub_len == 4:  # 'l' - LOAD or LIST
        var s1 = sub_ptr[unsafe_offset=1] | 0x20
        if s1 == 111:  # 'o' - LOAD
            # FUNCTION LOAD [REPLACE] <code>
            if i + 2 >= num_tokens:
                writer.append_error_response("ERR wrong number of arguments for 'function|load' command")
                return extra
            var replace: Int32 = 0
            var code_tok_idx = i + 2
            # Check for REPLACE flag
            if i + 3 < num_tokens:
                var maybe_replace = tokens[unsafe_offset=i + 2].ptr
                var mr_len = Int(tokens[unsafe_offset=i + 2].length)
                if mr_len == 7 and (maybe_replace[unsafe_offset=0] | 0x20) == 114:  # 'r' - REPLACE
                    replace = 1
                    code_tok_idx = i + 3
            if code_tok_idx >= num_tokens:
                writer.append_error_response("ERR wrong number of arguments for 'function|load' command")
                return extra
            var code_ptr = tokens[unsafe_offset=code_tok_idx].ptr
            var code_len = Int(tokens[unsafe_offset=code_tok_idx].length)
            # gh #87.6: stack_allocation (see EVAL handler note).
            var name_buf = stack_allocation[64, UInt8]()
            var result = external_call["pion_lua_load_library", Int32](
                lua[].state, code_ptr, Int32(code_len), replace, name_buf, Int32(64))
            if result < 0:
                var err_ptr = external_call["pion_lua_get_error",
                    Pointer[UInt8, MutUntrackedOrigin]](lua[].state)
                var elen = 0
                while elen < 500 and err_ptr[unsafe_offset=elen] != 0:
                    elen += 1
                writer.buffer[unsafe_offset=writer.offset] = 45  # '-'
                writer.offset += 1
                unsafe_memcpy(dest=writer.buffer.unsafe_offset(writer.offset), src=err_ptr, count=elen)
                writer.offset += elen
                writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
                writer.offset += 2
            else:
                # Return library name as bulk string
                var nlen = 0
                while nlen < 63 and name_buf[unsafe_offset=nlen] != 0:
                    nlen += 1
                writer.append_bulk_string_response(name_buf, nlen)
        elif s1 == 105:  # 'i' - LIST
            # FUNCTION LIST → array of library info
            var lib_count = Int(external_call["pion_lua_library_count", Int32](lua[].state))
            if lib_count == 0:
                writer.append_empty_array_response()
            else:
                # *N\r\n where each element is a map with library_name and functions
                writer.buffer[unsafe_offset=writer.offset] = 42  # '*'
                writer.offset += 1
                writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(lib_count))
                writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
                writer.offset += 2
                for li in range(lib_count):
                    # Each library: *6\r\n (3 key-value pairs)
                    # $12\r\nlibrary_name\r\n $<n>\r\n<name>\r\n
                    # $6\r\nengine\r\n $3\r\nlua\r\n
                    # $9\r\nfunctions\r\n *<n>\r\n [func entries]
                    writer.buffer[unsafe_offset=writer.offset] = 42  # '*'
                    writer.offset += 1
                    writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(6))
                    writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
                    writer.offset += 2

                    # library_name key
                    writer.append_bulk_string_response("library_name".unsafe_ptr(), 12)
                    var lname = external_call["pion_lua_library_name",
                        Pointer[UInt8, MutUntrackedOrigin]](lua[].state, Int32(li))
                    var lnlen = 0
                    while lnlen < 63 and lname[unsafe_offset=lnlen] != 0:
                        lnlen += 1
                    writer.append_bulk_string_response(lname, lnlen)

                    # engine key
                    writer.append_bulk_string_response("engine".unsafe_ptr(), 6)
                    writer.append_bulk_string_response("LUA".unsafe_ptr(), 3)

                    # functions key
                    writer.append_bulk_string_response("functions".unsafe_ptr(), 9)
                    var fc = Int(external_call["pion_lua_library_func_count", Int32](
                        lua[].state, Int32(li)))
                    writer.buffer[unsafe_offset=writer.offset] = 42  # '*'
                    writer.offset += 1
                    writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(fc))
                    writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
                    writer.offset += 2
                    for fi in range(fc):
                        # Each function: *2\r\n $4\r\nname\r\n $<n>\r\n<fname>\r\n
                        writer.buffer[unsafe_offset=writer.offset] = 42  # '*'
                        writer.offset += 1
                        writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(2))
                        writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
                        writer.offset += 2
                        writer.append_bulk_string_response("name".unsafe_ptr(), 4)
                        var fname = external_call["pion_lua_library_func_name",
                            Pointer[UInt8, MutUntrackedOrigin]](lua[].state, Int32(li), Int32(fi))
                        var fnlen = 0
                        while fnlen < 63 and fname[unsafe_offset=fnlen] != 0:
                            fnlen += 1
                        writer.append_bulk_string_response(fname, fnlen)
        else:
            writer.append_error_response("ERR unknown FUNCTION subcommand")
    elif s0 == 100:  # 'd' - DELETE (6) or DUMP (4)
        var s1 = sub_ptr[unsafe_offset=1] | 0x20
        if s1 == 101 and sub_len == 6:  # 'e' - DELETE
            if i + 2 >= num_tokens:
                writer.append_error_response("ERR wrong number of arguments for 'function|delete' command")
                return extra
            var name_ptr = tokens[unsafe_offset=i + 2].ptr
            var name_len = Int(tokens[unsafe_offset=i + 2].length)
            var name_buf = alloc[UInt8](name_len + 1)
            unsafe_memcpy(dest=name_buf, src=name_ptr, count=name_len)
            name_buf[unsafe_offset=name_len] = 0
            var result = external_call["pion_lua_delete_library", Int32](lua[].state, name_buf)
            name_buf.unsafe_free()
            if result < 0:
                writer.append_error_response("ERR Library not found")
            else:
                writer.append_ok_response()
        else:  # DUMP
            # FUNCTION DUMP → empty bulk string (not implemented)
            writer.append_bulk_string_response("".unsafe_ptr(), 0)
    elif s0 == 102 and sub_len == 5:  # 'f' - FLUSH
        external_call["pion_lua_flush_libraries", NoneType](lua[].state)
        writer.append_ok_response()
    elif s0 == 115 and sub_len == 5:  # 's' - STATS
        # FUNCTION STATS → *2\r\n $15\r\nrunning_script\r\n :0\r\n
        writer.buffer[unsafe_offset=writer.offset] = 42; writer.offset += 1  # '*'
        writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(2))
        writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10; writer.offset += 2
        writer.append_bulk_string_response("running_script".unsafe_ptr(), 14)
        writer.append_int_response(Int64(0))
    elif s0 == 114:  # 'r' - RESTORE
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR unknown FUNCTION subcommand")

    return extra


def handle_fcall(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
    ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
    hash_map_pool: Pointer[ObjectPool[SlabHashMap], MutUntrackedOrigin],
    list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin],
) -> Int:
    """FCALL function_name numkeys [key ...] [arg ...]."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra

    # Parse: FCALL func_name numkeys [key ...] [arg ...]
    if i + 2 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'fcall' command")
        return extra

    var fname_ptr = tokens[unsafe_offset=i + 1].ptr
    var fname_len = Int(tokens[unsafe_offset=i + 1].length)

    var numkeys_ptr = tokens[unsafe_offset=i + 2].ptr
    var numkeys_len = Int(tokens[unsafe_offset=i + 2].length)
    var nk_parsed = _parse_int(numkeys_ptr, numkeys_len)
    if not nk_parsed.valid:
        writer.append_error_response("ERR value is not an integer or out of range")
        return extra
    var numkeys = Int(nk_parsed.value)
    # gh #410: validate before _set_keys_argv sizes an array from numkeys.
    if numkeys < 0:
        writer.append_error_response("ERR Number of keys can't be negative")
        return extra
    if numkeys > num_tokens - i - 3:
        writer.append_error_response("ERR Number of keys can't be greater than number of args")
        return extra

    # Set KEYS and ARGV (reuse _set_keys_argv but tokens layout is same as EVAL)
    _set_keys_argv(lua[].state, tokens, i, num_tokens, numkeys)

    # Execute function
    var status = external_call["pion_lua_exec_function", Int32](
        lua[].state, fname_ptr, Int32(fname_len))

    # Dispatch loop (same as EVAL)
    var max_iterations = 10000
    var iterations = 0
    while status == PION_LUA_NEEDS_CMD and iterations < max_iterations:
        iterations += 1
        var nargs = Int(external_call["pion_lua_get_call_nargs", Int32](lua[].state))
        status = _dispatch_lua_cmd(lua[].state, nargs, keyspace, ttl_map,
                                   hash_map_pool, list_pool, lua[].val_buf)

    if iterations >= max_iterations:
        writer.append_error_response("ERR Lua script too many redis.call() invocations")
        return extra

    if status == PION_LUA_OK:
        var result_len = external_call["pion_lua_get_result", Int32](
            lua[].state, lua[].result_buf, Int32(LUA_RESULT_BUF_SIZE))
        if result_len > 0:
            unsafe_memcpy(dest=writer.buffer.unsafe_offset(writer.offset),
                   src=lua[].result_buf, count=Int(result_len))
            writer.offset += Int(result_len)
        else:
            writer.append_null_response()
    elif status == PION_LUA_ERROR:
        var err_ptr = external_call["pion_lua_get_error",
            Pointer[UInt8, MutUntrackedOrigin]](lua[].state)
        var err_len = 0
        while err_len < 500 and err_ptr[unsafe_offset=err_len] != 0:
            err_len += 1
        if err_len > 0:
            writer.buffer[unsafe_offset=writer.offset] = 45  # '-'
            writer.offset += 1
            unsafe_memcpy(dest=writer.buffer.unsafe_offset(writer.offset), src=err_ptr, count=err_len)
            writer.offset += err_len
            writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
            writer.offset += 2
        else:
            writer.append_error_response("ERR unknown Lua error")
    else:
        writer.append_error_response("ERR unexpected Lua execution state")

    return extra


# ── Internal helpers ──

def _set_keys_argv(
    state: Pointer[NoneType, MutUntrackedOrigin],
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    i: Int, num_tokens: Int, numkeys: Int,
):
    """Set KEYS and ARGV Lua globals from token array."""
    # Tokens: [i]=CMD, [i+1]=script/sha, [i+2]=numkeys, [i+3..i+2+numkeys]=keys, rest=args
    var key_start = i + 3
    var arg_start = key_start + numkeys

    # Build KEYS array
    var key_ptrs = alloc[Pointer[UInt8, MutUntrackedOrigin]](numkeys if numkeys > 0 else 1)
    var key_lens = alloc[Int32](numkeys if numkeys > 0 else 1)
    for k in range(numkeys):
        var tok_idx = key_start + k
        if tok_idx < num_tokens:
            key_ptrs[unsafe_offset=k] = tokens[unsafe_offset=tok_idx].ptr
            key_lens[unsafe_offset=k] = Int32(tokens[unsafe_offset=tok_idx].length)
        else:
            key_ptrs[unsafe_offset=k] = null_ptr[UInt8, MutUntrackedOrigin]()
            key_lens[unsafe_offset=k] = 0

    external_call["pion_lua_set_keys", NoneType](
        state, key_ptrs, key_lens, Int32(numkeys))

    # Build ARGV array
    var nargv = num_tokens - arg_start
    if nargv < 0:
        nargv = 0
    var arg_ptrs = alloc[Pointer[UInt8, MutUntrackedOrigin]](nargv if nargv > 0 else 1)
    var arg_lens = alloc[Int32](nargv if nargv > 0 else 1)
    for a in range(nargv):
        var tok_idx = arg_start + a
        if tok_idx < num_tokens:
            arg_ptrs[unsafe_offset=a] = tokens[unsafe_offset=tok_idx].ptr
            arg_lens[unsafe_offset=a] = Int32(tokens[unsafe_offset=tok_idx].length)
        else:
            arg_ptrs[unsafe_offset=a] = null_ptr[UInt8, MutUntrackedOrigin]()
            arg_lens[unsafe_offset=a] = 0

    external_call["pion_lua_set_argv", NoneType](
        state, arg_ptrs, arg_lens, Int32(nargv))

    key_ptrs.unsafe_free()
    key_lens.unsafe_free()
    arg_ptrs.unsafe_free()
    arg_lens.unsafe_free()


# @always_inline is load-bearing (gh #349): EVAL/EVALSHA pass a
# `stack_allocation` sha1 buffer here. Mojo 1.0 marks an out-of-line call with
# an untracked-origin pointer argument `tail`, which tells LLVM the callee never
# touches the caller's stack, so the buffer may be treated as dead across the
# call. `tools/audit_tail_alloca.py` finds every such call in the built IR.
@always_inline
def _exec_and_write_result(
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
    sha1_buf: Pointer[UInt8, MutUntrackedOrigin],
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
    ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
    hash_map_pool: Pointer[ObjectPool[SlabHashMap], MutUntrackedOrigin],
    list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin],
    mut writer: ResponseWriter,
):
    """Execute Lua script with dispatch loop and write result to ResponseWriter."""
    var state = lua[].state

    # Start execution
    var status = external_call["pion_lua_exec_sha1", Int32](state, sha1_buf)

    # Dispatch loop: handle redis.call() yields
    var max_iterations = 10000  # safety limit
    var iterations = 0
    while status == PION_LUA_NEEDS_CMD and iterations < max_iterations:
        iterations += 1
        var nargs = Int(external_call["pion_lua_get_call_nargs", Int32](state))
        status = _dispatch_lua_cmd(state, nargs, keyspace, ttl_map,
                                   hash_map_pool, list_pool, lua[].val_buf)

    if iterations >= max_iterations:
        writer.append_error_response("ERR Lua script too many redis.call() invocations")
        return

    if status == PION_LUA_OK:
        # Read result as RESP bytes and append to writer
        var result_len = external_call["pion_lua_get_result", Int32](
            state, lua[].result_buf, Int32(LUA_RESULT_BUF_SIZE))
        if result_len > 0:
            unsafe_memcpy(dest=writer.buffer.unsafe_offset(writer.offset),
                   src=lua[].result_buf, count=Int(result_len))
            writer.offset += Int(result_len)
        else:
            # No result or buffer overflow → nil
            writer.append_null_response()
    elif status == PION_LUA_ERROR:
        var err_ptr = external_call["pion_lua_get_error",
            Pointer[UInt8, MutUntrackedOrigin]](state)
        # Read error string from C
        var err_len = 0
        while err_len < 500 and err_ptr[unsafe_offset=err_len] != 0:
            err_len += 1
        if err_len > 0:
            # Write error response directly: -ERR <msg>\r\n
            writer.buffer[unsafe_offset=writer.offset] = 45  # '-'
            writer.offset += 1
            unsafe_memcpy(dest=writer.buffer.unsafe_offset(writer.offset), src=err_ptr, count=err_len)
            writer.offset += err_len
            writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
            writer.offset += 2
        else:
            writer.append_error_response("ERR unknown Lua error")
    else:
        writer.append_error_response("ERR unexpected Lua execution state")
