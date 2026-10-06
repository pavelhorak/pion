"""Bitmap/HLL commands: BITOP, BITPOS, BITCOUNT, BITFIELD, BITFIELD_RO, PFMERGE.

Redis has no bitmap type: a bitmap IS a string. Pion stores what SETBIT,
BITFIELD and BITOP write as `BITMAP`, which shares a heap STRING's `[ptr, len]`
words, and every command here reads either shape. Bit 0 is the most
significant bit of byte 0 (gh #232).

Arguments follow Redis's bitops.c (public #31): every argument is parsed and
every key type-checked before anything is written, so a refused command
changes nothing, and each error is the one Redis sends for that input.
"""
from src.common.ptr import null_ptr, is_not_null
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy, unsafe_memset
from std.bit import pop_count
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.hll import hll_merge, hll_add, hll_count, HLL_REGISTERS
from src.common.container_free import remove_and_free
from src.io.wal import WAL
from src.common.utils import arg_eq, parse_int64_strict
from std.math import sqrt, log, ceil
from std.ffi import external_call


comptime _E_NOT_INT = "ERR value is not an integer or out of range"
comptime _E_SYNTAX = "ERR syntax error"
comptime _E_WRONGTYPE = "WRONGTYPE Operation against a key holding the wrong kind of value"
comptime _E_OFFSET = "ERR bit offset is not an integer or out of range"
comptime _E_BF_TYPE = "ERR Invalid bitfield type. Use something like i16 u8. Note that u64 is not supported but i64 is."
# Redis caps a bit offset at proto-max-bulk-len (512 MB) worth of bits.
comptime _MAX_BIT_OFFSET = 4294967296


@always_inline
def _tok_int(tokens: Pointer[RESP3Token, MutUntrackedOrigin], j: Int, mut out: Int64) -> Bool:
    """Redis's getLongLongFromObject: an exact integer or nothing."""
    var r = parse_int64_strict(tokens[unsafe_offset=j].ptr, tokens[unsafe_offset=j].length)
    out = r.value
    return r.ok


@always_inline
def _bit_at(p: Pointer[UInt8, MutUntrackedOrigin], pos: Int) -> Int:
    return Int((p[unsafe_offset=pos >> 3] >> UInt8(7 - (pos & 7))) & 1)


def _popcount(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    var c = 0
    var k = 0
    while k + 8 <= n:
        c += Int(pop_count(p.unsafe_offset(k).unsafe_bitcast[UInt64]().load()))
        k += 8
    while k < n:
        c += Int(pop_count(p[unsafe_offset=k]))
        k += 1
    return c


def _count_bit_range(p: Pointer[UInt8, MutUntrackedOrigin], first: Int, last: Int) -> Int:
    """Set bits in the inclusive bit range [first, last]."""
    var c = 0
    var pos = first
    while pos <= last and (pos & 7) != 0:
        c += _bit_at(p, pos)
        pos += 1
    if pos + 7 <= last:
        var nbytes = (last + 1 - pos) >> 3
        c += _popcount(p.unsafe_offset(pos >> 3), nbytes)
        pos += nbytes * 8
    while pos <= last:
        c += _bit_at(p, pos)
        pos += 1
    return c


@always_inline
def _clamp_range(mut start: Int, mut end: Int, totlen: Int):
    """Redis's index conversion for BITCOUNT/BITPOS: negatives count from the
    end, then BOTH ends clamp up to 0 (`BITCOUNT k -100 -99` is byte 0, not an
    empty range) and the end clamps down to the last unit."""
    if start < 0: start = totlen + start
    if end < 0: end = totlen + end
    if start < 0: start = 0
    if end < 0: end = 0
    if end >= totlen: end = totlen - 1


def handle_bitcount(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                    mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) -> Int:
    """BITCOUNT key [start end [BYTE|BIT]] — every form (the fast path answers
    the whole-key form outside transactions)."""
    var argc = num_tokens - i
    if argc < 2:
        writer.append_error_response("ERR wrong number of arguments for 'bitcount' command")
        return 0
    var start = Int64(0)
    var end = Int64(0)
    var isbit = False
    if argc == 4 or argc == 5:
        if not _tok_int(tokens, i + 2, start) or not _tok_int(tokens, i + 3, end):
            writer.append_error_response(_E_NOT_INT)
            return 0
        if argc == 5:
            var u = tokens[unsafe_offset=i + 4]
            if arg_eq(u.ptr, u.length, "bit"):
                isbit = True
            elif not arg_eq(u.ptr, u.length, "byte"):
                writer.append_error_response(_E_SYNTAX)
                return 0
    elif argc != 2:
        writer.append_error_response(_E_SYNTAX)
        return 0
    var v = keyspace[].get(GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length))
    if v.is_none():
        writer.append_int_response(0)
        return 0
    if not v.is_string_like():
        writer.append_error_response(_E_WRONGTYPE)
        return 0
    var scratch = alloc[UInt8](32)
    var blen = 0
    var p = v.bitmap_view(scratch, blen)
    var count = 0
    if argc == 2:
        count = _popcount(p, blen)
    elif not (start < 0 and end < 0 and start > end):
        var s = Int(start)
        var e = Int(end)
        var totlen = blen * 8 if isbit else blen
        _clamp_range(s, e, totlen)
        if s <= e:
            if isbit:
                count = _count_bit_range(p, s, e)
            else:
                count = _popcount(p.unsafe_offset(s), e - s + 1)
    scratch.unsafe_free()
    writer.append_int_response(Int64(count))
    return 0


def handle_bitpos(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                  mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) -> Int:
    """BITPOS key bit [start [end [BYTE|BIT]]].

    A missing key is an endless run of 0 bits: 0 for bit 0, -1 for bit 1.
    Looking for a 0 with no END past a run of 1s answers the first bit after
    the string; with an END it answers -1, since the range holds no 0."""
    var argc = num_tokens - i
    if argc < 3:
        writer.append_error_response("ERR wrong number of arguments for 'bitpos' command")
        return 0
    var bit = Int64(0)
    if not _tok_int(tokens, i + 2, bit):
        writer.append_error_response(_E_NOT_INT)
        return 0
    if bit != 0 and bit != 1:
        writer.append_error_response("ERR The bit argument must be 1 or 0.")
        return 0
    var start = Int64(0)
    var end = Int64(0)
    var isbit = False
    var end_given = False
    if argc >= 4 and argc <= 6:
        if not _tok_int(tokens, i + 3, start):
            writer.append_error_response(_E_NOT_INT)
            return 0
        if argc == 6:
            var u = tokens[unsafe_offset=i + 5]
            if arg_eq(u.ptr, u.length, "bit"):
                isbit = True
            elif not arg_eq(u.ptr, u.length, "byte"):
                writer.append_error_response(_E_SYNTAX)
                return 0
        if argc >= 5:
            if not _tok_int(tokens, i + 4, end):
                writer.append_error_response(_E_NOT_INT)
                return 0
            end_given = True
    elif argc != 3:
        writer.append_error_response(_E_SYNTAX)
        return 0
    var v = keyspace[].get(GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length))
    if v.is_none():
        writer.append_int_response(Int64(-1) if bit == 1 else Int64(0))
        return 0
    if not v.is_string_like():
        writer.append_error_response(_E_WRONGTYPE)
        return 0
    var scratch = alloc[UInt8](32)
    var blen = 0
    var p = v.bitmap_view(scratch, blen)
    # The search range, in bits, inclusive.
    var first = 0
    var last = blen * 8 - 1
    var empty = blen == 0
    if argc > 3:
        var s = Int(start)
        var e = Int(end)
        var totlen = blen * 8 if isbit else blen
        if not end_given:
            e = totlen - 1
        _clamp_range(s, e, totlen)
        empty = s > e
        first = s if isbit else s * 8
        last = e if isbit else e * 8 + 7
    var pos = -1
    if not empty:
        var want = Int(bit)
        var skip = UInt8(0) if want == 1 else UInt8(0xFF)
        var at = first
        while at <= last:
            if (at & 7) == 0 and at + 7 <= last and p[unsafe_offset=at >> 3] == skip:
                at += 8
                continue
            if _bit_at(p, at) == want:
                pos = at
                break
            at += 1
        if pos < 0 and want == 0 and not end_given:
            pos = last + 1
    scratch.unsafe_free()
    writer.append_int_response(Int64(pos))
    return 0


# ── BITFIELD ─────────────────────────────────────────────────────────────────

comptime _BF_GET = 0
comptime _BF_SET = 1
comptime _BF_INCRBY = 2
comptime _OW_WRAP = 0
comptime _OW_SAT = 1
comptime _OW_FAIL = 2


@fieldwise_init
struct _BfOp(Copyable, Movable):
    var opcode: Int
    var signed: Bool
    var bits: Int
    var offset: Int
    var arg: Int64
    var overflow: Int


def _bf_type(t: RESP3Token, mut signed: Bool, mut bits: Int) -> Bool:
    """`i1`..`i64` or `u1`..`u63`, lower-case prefix, as Redis parses it."""
    if t.length < 2:
        return False
    var c = t.ptr[unsafe_offset=0]
    if c == 105:        # 'i'
        signed = True
    elif c == 117:      # 'u'
        signed = False
    else:
        return False
    var w = parse_int64_strict(t.ptr.unsafe_offset(1), t.length - 1)
    if not w.ok or w.value < 1 or w.value > (Int64(64) if signed else Int64(63)):
        return False
    bits = Int(w.value)
    return True


def _bf_offset(t: RESP3Token, bits: Int, mut offset: Int) -> Bool:
    """A bit offset, or `#N` = N fields of this type in."""
    var hashed = t.length > 0 and t.ptr[unsafe_offset=0] == 35   # '#'
    var skip = 1 if hashed else 0
    var o = parse_int64_strict(t.ptr.unsafe_offset(skip), t.length - skip)
    if not o.ok or o.value < 0:
        return False
    var v = Int(o.value)
    if hashed:
        if v > _MAX_BIT_OFFSET // bits:
            return False
        v *= bits
    if v >= _MAX_BIT_OFFSET:
        return False
    offset = v
    return True


def _bf_get(p: Pointer[UInt8, MutUntrackedOrigin], blen: Int, offset: Int, bits: Int) -> UInt64:
    """The field's bits as an unsigned number; bits past `blen` read as 0."""
    var v = UInt64(0)
    for j in range(bits):
        var pos = offset + j
        var b = UInt64(0)
        if (pos >> 3) < blen:
            b = UInt64(_bit_at(p, pos))
        v = (v << 1) | b
    return v


def _bf_set(p: Pointer[UInt8, MutUntrackedOrigin], offset: Int, bits: Int, value: UInt64):
    for j in range(bits):
        var pos = offset + j
        var bitv = UInt8((value >> UInt64(bits - 1 - j)) & 1)
        var sh = UInt8(7 - (pos & 7))
        var byte = p[unsafe_offset=pos >> 3]
        p[unsafe_offset=pos >> 3] = (byte & ~(UInt8(1) << sh)) | (bitv << sh)


@always_inline
def _bf_sext(raw: UInt64, bits: Int) -> Int64:
    """Sign-extend an `i<bits>` field (gh #232: -1234 in an i16 read 64302)."""
    if bits < 64 and (raw & (UInt64(1) << UInt64(bits - 1))) != 0:
        return Int64(raw | (~UInt64(0) << UInt64(bits)))
    return Int64(raw)


def _bf_overflow_unsigned(value: UInt64, incr: Int64, bits: Int, ow: Int, mut limit: UInt64) -> Bool:
    """Redis's checkUnsignedBitfieldOverflow: True on overflow, with `limit` the
    value WRAP or SAT stores (FAIL stores nothing)."""
    var maxv = (UInt64(1) << UInt64(bits)) - 1
    var maxincr = Int64(maxv - value)
    var minincr = Int64(UInt64(0) - value)
    var up = value > maxv or (incr > 0 and incr > maxincr)
    var down = not up and incr < 0 and incr < minincr
    if not up and not down:
        return False
    if ow == _OW_WRAP:
        limit = (value + UInt64(incr)) & ~(~UInt64(0) << UInt64(bits))
    elif ow == _OW_SAT:
        limit = maxv if up else UInt64(0)
    return True


def _bf_overflow_signed(value: Int64, incr: Int64, bits: Int, ow: Int, mut limit: Int64) -> Bool:
    """Redis's checkSignedBitfieldOverflow, its wrapping arithmetic included."""
    var maxv = Int64(9223372036854775807) if bits == 64 else (Int64(1) << Int64(bits - 1)) - 1
    var minv = -maxv - 1
    var maxincr = Int64(UInt64(maxv) - UInt64(value))
    var minincr = Int64(UInt64(minv) - UInt64(value))
    var up = value > maxv or (bits != 64 and incr > maxincr) \
        or (value >= 0 and incr > 0 and incr > maxincr)
    var down = not up and (value < minv or (bits != 64 and incr < minincr)
                           or (value < 0 and incr < 0 and incr < minincr))
    if not up and not down:
        return False
    if ow == _OW_WRAP:
        var c = UInt64(value) + UInt64(incr)
        if bits < 64:
            var mask = ~UInt64(0) << UInt64(bits)
            if (c & (UInt64(1) << UInt64(bits - 1))) != 0:
                c |= mask
            else:
                c &= ~mask
        limit = Int64(c)
    elif ow == _OW_SAT:
        limit = maxv if up else minv
    return True


def handle_bitfield(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                    mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                    readonly: Bool = False) -> Int:
    """BITFIELD key [GET type offset] [SET type offset value]
    [INCRBY type offset increment] [OVERFLOW WRAP|SAT|FAIL] ...
    (BITFIELD_RO when `readonly`: GET only).

    Parses every operation before touching the key, so a bad argument
    anywhere refuses the whole command with nothing written. A write
    creates the key, or grows it, to the farthest bit any operation writes,
    whether or not an OVERFLOW FAIL then skips the write. A string
    key is written as a bitmap copy of its bytes."""
    var name = String("bitfield_ro") if readonly else String("bitfield")
    if num_tokens - i < 2:
        writer.append_error_response("ERR wrong number of arguments for '" + name + "' command")
        return 0
    var ops = List[_BfOp]()
    var ow = _OW_WRAP
    var writes = False
    var highest = -1
    var j = i + 2
    while j < num_tokens:
        var rem = num_tokens - j - 1
        var t = tokens[unsafe_offset=j]
        var opcode = -1
        if arg_eq(t.ptr, t.length, "get") and rem >= 2:
            opcode = _BF_GET
        elif arg_eq(t.ptr, t.length, "set") and rem >= 3:
            opcode = _BF_SET
        elif arg_eq(t.ptr, t.length, "incrby") and rem >= 3:
            opcode = _BF_INCRBY
        elif arg_eq(t.ptr, t.length, "overflow") and rem >= 1:
            var m = tokens[unsafe_offset=j + 1]
            if arg_eq(m.ptr, m.length, "wrap"):
                ow = _OW_WRAP
            elif arg_eq(m.ptr, m.length, "sat"):
                ow = _OW_SAT
            elif arg_eq(m.ptr, m.length, "fail"):
                ow = _OW_FAIL
            else:
                writer.append_error_response("ERR Invalid OVERFLOW type specified")
                return 0
            j += 2
            continue
        else:
            writer.append_error_response(_E_SYNTAX)
            return 0
        var signed = False
        var bits = 0
        if not _bf_type(tokens[unsafe_offset=j + 1], signed, bits):
            writer.append_error_response(_E_BF_TYPE)
            return 0
        var offset = 0
        if not _bf_offset(tokens[unsafe_offset=j + 2], bits, offset):
            writer.append_error_response(_E_OFFSET)
            return 0
        var arg = Int64(0)
        if opcode != _BF_GET:
            writes = True
            if offset + bits - 1 > highest:
                highest = offset + bits - 1
            if not _tok_int(tokens, j + 3, arg):
                writer.append_error_response(_E_NOT_INT)
                return 0
        ops.append(_BfOp(opcode, signed, bits, offset, arg, ow))
        j += 3 if opcode == _BF_GET else 4

    var key = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    var v = keyspace[].get(key)
    if writes and readonly:
        writer.append_error_response("ERR BITFIELD_RO only supports the GET subcommand")
        return 0
    if not v.is_none() and not v.is_string_like():
        writer.append_error_response(_E_WRONGTYPE)
        return 0

    var scratch = alloc[UInt8](32)
    var p = null_ptr[UInt8, MutUntrackedOrigin]()
    var blen = 0
    if writes:
        var need = highest // 8 + 1
        if not v.is_none() and v.type.value == ValueType.BITMAP and v.bitmap_len() >= need:
            p = v.as_bitmap()                   # written in place
            blen = v.bitmap_len()
        else:
            if v.is_none():
                p = alloc[UInt8](need)
                unsafe_memset(p, 0, need)
                blen = need
            else:
                # A shorter bitmap, or a string: write into a copy grown to
                # `need`. The keyspace retires what it replaces.
                p = v.owned_bitmap_copy(need, blen)
            var nv = GenericValue()
            nv.type = ValueType(ValueType.BITMAP)
            nv._data0 = UInt64(Int(p))
            nv._data1 = UInt64(blen)
            keyspace[].set(key, nv)
    elif not v.is_none():
        p = v.bitmap_view(scratch, blen)

    writer.append_array_header(len(ops))
    for k in range(len(ops)):
        var op = ops[k].copy()
        if op.opcode == _BF_GET:
            var raw = _bf_get(p, blen, op.offset, op.bits)
            writer.append_int_response(_bf_sext(raw, op.bits) if op.signed else Int64(raw))
            continue
        var store = UInt64(0)
        var reply = Int64(0)
        var failed = False
        if op.signed:
            var old = _bf_sext(_bf_get(p, blen, op.offset, op.bits), op.bits)
            var limit = Int64(0)
            if op.opcode == _BF_INCRBY:
                var nv = Int64(UInt64(old) + UInt64(op.arg))
                if _bf_overflow_signed(old, op.arg, op.bits, op.overflow, limit):
                    nv = limit
                    failed = op.overflow == _OW_FAIL
                store = UInt64(nv)
                reply = nv
            else:
                var nv = op.arg
                if _bf_overflow_signed(op.arg, 0, op.bits, op.overflow, limit):
                    nv = limit
                    failed = op.overflow == _OW_FAIL
                store = UInt64(nv)
                reply = old
        else:
            var old = _bf_get(p, blen, op.offset, op.bits)
            var limit = UInt64(0)
            if op.opcode == _BF_INCRBY:
                var nv = old + UInt64(op.arg)
                if _bf_overflow_unsigned(old, op.arg, op.bits, op.overflow, limit):
                    nv = limit
                    failed = op.overflow == _OW_FAIL
                store = nv
                reply = Int64(nv)
            else:
                var nv = UInt64(op.arg)
                if _bf_overflow_unsigned(nv, 0, op.bits, op.overflow, limit):
                    nv = limit
                    failed = op.overflow == _OW_FAIL
                store = nv
                reply = Int64(old)
        if failed:
            writer.append_null_response()
        else:
            _bf_set(p, op.offset, op.bits, store)
            writer.append_int_response(reply)
    scratch.unsafe_free()
    return 0


def handle_bitfield_ro(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                       mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) -> Int:
    """BITFIELD_RO key [GET type offset] ..."""
    return handle_bitfield(tokens, i, num_tokens, writer, keyspace, True)


# ── BITOP ────────────────────────────────────────────────────────────────────

comptime _OP_AND = 0
comptime _OP_OR = 1
comptime _OP_XOR = 2
comptime _OP_NOT = 3
comptime _OP_DIFF = 4
comptime _OP_DIFF1 = 5
comptime _OP_ANDOR = 6
comptime _OP_ONE = 7


def handle_bitop(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                 mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                 ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) -> Int:
    """BITOP AND|OR|XOR|NOT|DIFF|DIFF1|ANDOR|ONE destkey key [key ...].

    With X the first source and Y the rest: DIFF is X and not any Y, DIFF1 is
    any Y and not X, ANDOR is X and any Y, ONE keeps the bits set in exactly
    one source. Missing sources are empty strings and shorter ones are
    zero-padded. The result replaces the destination, its TTL included; an
    empty result deletes it. Replies with the result's length in bytes."""
    if num_tokens - i < 4:
        writer.append_error_response("ERR wrong number of arguments for 'bitop' command")
        return 0
    var o = tokens[unsafe_offset=i + 1]
    var op = -1
    var opname = String("")
    if arg_eq(o.ptr, o.length, "and"):
        op = _OP_AND
    elif arg_eq(o.ptr, o.length, "or"):
        op = _OP_OR
    elif arg_eq(o.ptr, o.length, "xor"):
        op = _OP_XOR
    elif arg_eq(o.ptr, o.length, "not"):
        op = _OP_NOT
    elif arg_eq(o.ptr, o.length, "diff"):
        op = _OP_DIFF
        opname = "DIFF"
    elif arg_eq(o.ptr, o.length, "diff1"):
        op = _OP_DIFF1
        opname = "DIFF1"
    elif arg_eq(o.ptr, o.length, "andor"):
        op = _OP_ANDOR
        opname = "ANDOR"
    elif arg_eq(o.ptr, o.length, "one"):
        op = _OP_ONE
    else:
        writer.append_error_response(_E_SYNTAX)
        return 0
    var nsrc = num_tokens - i - 3
    if op == _OP_NOT and nsrc != 1:
        writer.append_error_response("ERR BITOP NOT must be called with a single source key.")
        return 0
    if (op == _OP_DIFF or op == _OP_DIFF1 or op == _OP_ANDOR) and nsrc < 2:
        writer.append_error_response("ERR BITOP " + opname + " must be called with at least two source keys.")
        return 0

    # Every source is checked before anything is computed or written.
    var srcs = List[GenericValue]()
    var maxlen = 0
    for k in range(nsrc):
        var sv = keyspace[].get(GenericValue.borrow(tokens[unsafe_offset=i + 3 + k].ptr, tokens[unsafe_offset=i + 3 + k].length))
        if not sv.is_none() and not sv.is_string_like():
            writer.append_error_response(_E_WRONGTYPE)
            return 0
        srcs.append(sv)
        if not sv.is_none() and sv.string_len() > maxlen:
            maxlen = sv.string_len()

    var dest = GenericValue.borrow(tokens[unsafe_offset=i + 2].ptr, tokens[unsafe_offset=i + 2].length)
    if is_not_null(ttl_map):
        _ = ttl_map[].remove_generic(dest)
    if maxlen == 0:
        _ = remove_and_free(keyspace, dest)
        writer.append_int_response(0)
        return 0

    # res = the op over the sources; acc = OR of the sources after the first
    # (DIFF/DIFF1/ANDOR) or the bits seen at least twice (ONE).
    var res = alloc[UInt8](maxlen)
    var acc = alloc[UInt8](maxlen)
    unsafe_memset(res, 0, maxlen)
    unsafe_memset(acc, 0, maxlen)
    var scratch = alloc[UInt8](32)
    for k in range(nsrc):
        var n = 0
        var sp = null_ptr[UInt8, MutUntrackedOrigin]()
        if not srcs[k].is_none():
            sp = srcs[k].bitmap_view(scratch, n)
        for b in range(maxlen):
            var x = sp[unsafe_offset=b] if b < n else UInt8(0)
            if k == 0:
                res[unsafe_offset=b] = ~x if op == _OP_NOT else x
            elif op == _OP_AND:
                res[unsafe_offset=b] &= x
            elif op == _OP_OR:
                res[unsafe_offset=b] |= x
            elif op == _OP_XOR:
                res[unsafe_offset=b] ^= x
            elif op == _OP_ONE:
                acc[unsafe_offset=b] |= res[unsafe_offset=b] & x
                res[unsafe_offset=b] ^= x
            else:
                acc[unsafe_offset=b] |= x
    for b in range(maxlen):
        if op == _OP_DIFF:
            res[unsafe_offset=b] &= ~acc[unsafe_offset=b]
        elif op == _OP_DIFF1:
            res[unsafe_offset=b] = ~res[unsafe_offset=b] & acc[unsafe_offset=b]
        elif op == _OP_ANDOR:
            res[unsafe_offset=b] &= acc[unsafe_offset=b]
        elif op == _OP_ONE:
            res[unsafe_offset=b] &= ~acc[unsafe_offset=b]
    scratch.unsafe_free()
    acc.unsafe_free()
    var nv = GenericValue()
    nv.type = ValueType(ValueType.BITMAP)
    nv._data0 = UInt64(Int(res))
    nv._data1 = UInt64(maxlen)
    keyspace[].set(dest, nv)
    writer.append_int_response(Int64(maxlen))
    return 0


@always_inline
def handle_pfmerge(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                   wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """PFMERGE destkey sourcekey [sourcekey ...] → +OK."""
    if i + 2 < num_tokens:
        var _pfm_dest = tokens[unsafe_offset=i+1].value()
        var _i = i + 2
        var _pfm_srcs = List[String]()
        while _i < num_tokens and tokens[unsafe_offset=_i].marker != 0:
            _pfm_srcs.append(tokens[unsafe_offset=_i].value()); _i += 1
        _i -= 1
        # gh #232: destination and every source are type-checked before the
        # merge buffer is allocated. A non-HLL source read as an empty
        # register set would silently produce a wrong cardinality — and
        # PFMERGE writes that result to the destination.
        var _pfm_dv = keyspace[].get(GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length))
        if not _pfm_dv.is_none() and _pfm_dv.type.value != ValueType.HLL:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return _i - i
        for _wts in range(len(_pfm_srcs)):
            var _wtv = keyspace[].get(_pfm_srcs[_wts])
            if not _wtv.is_none() and _wtv.type.value != ValueType.HLL:
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return _i - i
        var _pfm_res = alloc[UInt8](HLL_REGISTERS)
        var _pfm_resp = _pfm_res
        unsafe_memset(_pfm_resp, 0, HLL_REGISTERS)
        # Merge dest into result first (if exists)
        var _dest_v = keyspace[].get(_pfm_dest)
        if _dest_v.type.value == ValueType.HLL:
            hll_merge(_pfm_resp, _dest_v.as_hll())
        # Merge all sources
        for _si in range(len(_pfm_srcs)):
            var _sv = keyspace[].get(_pfm_srcs[_si])
            if _sv.type.value == ValueType.HLL: hll_merge(_pfm_resp, _sv.as_hll())
        # Store result at destkey
        var _pfm_gv = GenericValue()
        _pfm_gv.type = ValueType(ValueType.HLL)
        _pfm_gv.set_ptr(_pfm_resp.unsafe_bitcast[NoneType]())
        keyspace[].set(_pfm_dest, _pfm_gv)
        # gh #174: log the merged registers as a cmd-17 image, not cmd-24
        # elements. A merge is a register-wise max over the sources; there is no
        # set of elements whose replay reproduces it, because the source HLLs
        # only ever held registers, never the elements that set them.
        if is_not_null(wal):
            _ = wal[].append_kv(17, _pfm_dest.unsafe_ptr(), _pfm_dest.byte_length(),
                                _pfm_resp, HLL_REGISTERS)
        writer.append_ok_response()
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'pfmerge' command")
        return 0


def handle_pfselftest(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                      mut writer: ResponseWriter) -> Int:
    """PFSELFTEST (#39): Redis's self-test of its HyperLogLog, run against
    Pion's. Test 1: the count over random registers matches the textbook
    harmonic mean computed one register at a time (the counting kernel is
    SIMD, src/common/hll.mojo). Test 2: add 10M distinct elements and check
    the estimate at every power of ten against Redis's bound, 6x the standard
    error (1.04/sqrt(m)), with an error of at most 1 at 10."""
    if num_tokens - i != 1:
        writer.append_error_response("ERR wrong number of arguments for 'pfselftest' command")
        return 0
    var regs = alloc[UInt8](HLL_REGISTERS)
    # Redis's seed: two rand() calls.
    var seed = UInt64(Int(external_call["rand", Int32]())) | (UInt64(Int(external_call["rand", Int32]())) << 32)
    # Test 1: random registers (0..51, the most a 64-bit hash with p=14 sets).
    var x = seed | 1
    for _ in range(1000):
        for r in range(HLL_REGISTERS):
            x ^= x << 13
            x ^= x >> 7
            x ^= x << 17
            regs.store(r, UInt8(Int(x % 52)))
        var E = Float64(0.0)
        var zeros = 0
        for r in range(HLL_REGISTERS):
            var v = Int(regs.load(r))
            if v == 0:
                zeros += 1
            else:
                E += 1.0 / Float64(1 << v)
        var m = Float64(HLL_REGISTERS)
        var est = (0.7213 / (1.0 + 1.079 / m)) * m * m / (E + Float64(zeros))
        if est <= 2.5 * m and zeros != 0:
            est = m * log(m / Float64(zeros))
        var got = hll_count(regs)
        if Int(est) != got:
            writer.append_error_response("ERR TESTFAILED count kernel " + String(got)
                                         + " != reference " + String(Int(est)))
            regs.unsafe_free()
            return 0
    # Test 2: approximation error.
    unsafe_memset(regs, 0, HLL_REGISTERS)
    var relerr = 1.04 / sqrt(Float64(HLL_REGISTERS))
    var checkpoint = 1
    var ele = alloc[UInt8](8)
    for j in range(1, 10_000_001):
        var e = UInt64(j) ^ seed
        ele.unsafe_bitcast[UInt64]().store(0, e)
        _ = hll_add(regs, GenericValue.borrow(ele, 8))
        if j == checkpoint:
            var abserr = checkpoint - hll_count(regs)
            var maxerr = Int(ceil(relerr * 6.0 * Float64(checkpoint)))
            if j == 10:
                maxerr = 1
            if abserr < 0:
                abserr = -abserr
            if abserr > maxerr:
                writer.append_error_response("ERR TESTFAILED Too big error. card:" + String(checkpoint)
                                             + " abserr:" + String(abserr))
                ele.unsafe_free()
                regs.unsafe_free()
                return 0
            checkpoint *= 10
    ele.unsafe_free()
    regs.unsafe_free()
    writer.append_ok_response()
    return 0


def handle_pfdebug(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                   mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) -> Int:
    """PFDEBUG GETREG|DECODE|ENCODING|TODENSE key (#39), Redis's HyperLogLog
    debugging command. Pion keeps every HyperLogLog dense, one byte per
    register: ENCODING is always dense, TODENSE has nothing to convert (0), and
    DECODE, which prints the sparse form, answers Redis's error for a dense
    one. GETREG returns Pion's registers, which differ from Redis's for the same
    elements (a different hash; the documented HyperLogLog fence)."""
    if num_tokens - i != 3:
        writer.append_error_response("ERR wrong number of arguments for 'pfdebug' command")
        return 0
    var sub = tokens[unsafe_offset=i + 1]
    var kt = tokens[unsafe_offset=i + 2]
    var v = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    if v.is_none():
        writer.append_error_response("ERR The specified key does not exist")
        return 0
    if v.type.value != ValueType.HLL:
        if v.is_string_like() or v.type.value == ValueType.INT:
            writer.append_error_response("WRONGTYPE Key is not a valid HyperLogLog string value.")
        else:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return 0
    if arg_eq(sub.ptr, sub.length, "getreg"):
        var regs = v.as_hll()
        writer.append_array_header(HLL_REGISTERS)
        for r in range(HLL_REGISTERS):
            writer.append_int_response(Int64(Int(regs.load(r))))
    elif arg_eq(sub.ptr, sub.length, "decode"):
        writer.append_error_response("ERR HLL encoding is not sparse")
    elif arg_eq(sub.ptr, sub.length, "encoding"):
        writer.append_status_response("dense")
    elif arg_eq(sub.ptr, sub.length, "todense"):
        writer.append_int_response(0)
    else:
        writer.append_error_response("ERR Unknown PFDEBUG subcommand '" + sub.value() + "'")
    return 0
