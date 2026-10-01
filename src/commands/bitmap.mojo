"""Bitmap/HLL commands: BITOP, BITPOS, BITFIELD, BITFIELD_RO, PFMERGE."""
from src.common.ptr import null_ptr, is_not_null
from std.memory.unsafe_pointer import Pointer
from std.collections import Array
from std.memory import alloc, stack_allocation, unsafe_memcpy, unsafe_memset
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.bitmap import getbit, setbit, bitcount, SetBitResult
from src.common.hll import hll_add, hll_count, hll_merge, HLL_REGISTERS
from src.io.wal import WAL
from src.common.utils import strict_atol, format_int_to_buf


@always_inline
def handle_bitop(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """BITOP operation destkey key [key ...] → integer byte length of result."""
    if i + 3 < num_tokens:
        var _bop_op = tokens[unsafe_offset=i+1].value()
        var _i = i + 2
        var _bop_dest = tokens[unsafe_offset=_i].value()
        var _bop_src = List[String]()
        _i += 1
        while _i < num_tokens and tokens[unsafe_offset=_i].marker != 0:
            _bop_src.append(tokens[unsafe_offset=_i].value()); _i += 1
        _i -= 1
        var _bop_n = len(_bop_src)
        if _bop_n == 0:
            writer.append_int_response(Int64(0))
        else:
            # Find max byte length among sources (BITMAP or STRING types)
            var _bop_maxlen = 0
            for _bi in range(_bop_n):
                var _sv = keyspace[].get(_bop_src[_bi])
                if _sv.type.value == ValueType.BITMAP:
                    var _bl = _sv.bitmap_len()
                    if _bl > _bop_maxlen: _bop_maxlen = _bl
                elif _sv.type.value == ValueType.STRING or _sv.type.value == ValueType.STRING_SSO:
                    var _sl = _sv.string_len()
                    if _sl > _bop_maxlen: _bop_maxlen = _sl
            if _bop_maxlen == 0:
                # All sources empty - store empty bitmap
                var _dest_v = GenericValue()
                _dest_v.type = ValueType(ValueType.BITMAP)
                var _ep = alloc[UInt8](1); unsafe_memset(_ep, 0, 1)
                _dest_v._data0 = UInt64(Int(_ep))
                _dest_v._data1 = UInt64(1)
                keyspace[].set(_bop_dest, _dest_v)
                writer.append_int_response(Int64(0))
            else:
                var _bop_res = alloc[UInt8](_bop_maxlen)
                unsafe_memset(_bop_res, 0, _bop_maxlen)
                var _bop_resp = _bop_res
                # Initialize result from first source (BITMAP or STRING)
                var _sv0 = keyspace[].get(_bop_src[0])
                var _sso_buf0 = alloc[UInt8](24)
                if _sv0.type.value == ValueType.BITMAP:
                    var _sp0 = _sv0.as_bitmap(); var _sl0 = _sv0.bitmap_len()
                    unsafe_memcpy(dest=_bop_resp, src=_sp0, count=_sl0)
                elif _sv0.type.value == ValueType.STRING or _sv0.type.value == ValueType.STRING_SSO:
                    var _sp0 = _sv0.as_string_safe(_sso_buf0); var _sl0 = _sv0.string_len()
                    unsafe_memcpy(dest=_bop_resp, src=_sp0, count=_sl0)
                elif (_bop_op.unsafe_ptr()[unsafe_offset=0]|32) == 110:  # NOT - flip bits
                    for _bi in range(_bop_maxlen): _bop_resp[unsafe_offset=_bi] = 0xFF
                # Apply operation
                var _is_not = (_bop_op.unsafe_ptr()[unsafe_offset=0]|32) == 110
                var _sso_buf_op = alloc[UInt8](24)
                if _is_not:
                    # NOT: flip bits of first source
                    var _sv_not = keyspace[].get(_bop_src[0])
                    var _sp_not: Pointer[UInt8, MutUntrackedOrigin]
                    var _sl_not = 0
                    if _sv_not.type.value == ValueType.BITMAP:
                        _sp_not = _sv_not.as_bitmap(); _sl_not = _sv_not.bitmap_len()
                        for _bi in range(_sl_not): _bop_resp[unsafe_offset=_bi] = ~_sp_not[unsafe_offset=_bi]
                    elif _sv_not.type.value == ValueType.STRING or _sv_not.type.value == ValueType.STRING_SSO:
                        _sp_not = _sv_not.as_string_safe(_sso_buf_op); _sl_not = _sv_not.string_len()
                        for _bi in range(_sl_not): _bop_resp[unsafe_offset=_bi] = ~_sp_not[unsafe_offset=_bi]
                    for _bi in range(_sl_not, _bop_maxlen): _bop_resp[unsafe_offset=_bi] = 0
                else:
                    for _ki in range(1, _bop_n):
                        var _skv = keyspace[].get(_bop_src[_ki])
                        var _skp: Pointer[UInt8, MutUntrackedOrigin]
                        var _skl = 0
                        if _skv.type.value == ValueType.BITMAP:
                            _skp = _skv.as_bitmap(); _skl = _skv.bitmap_len()
                        elif _skv.type.value == ValueType.STRING or _skv.type.value == ValueType.STRING_SSO:
                            _skp = _skv.as_string_safe(_sso_buf_op); _skl = _skv.string_len()
                        else:
                            _skp = _bop_resp  # dummy (zero bytes)
                        var _opc = _bop_op.unsafe_ptr()[unsafe_offset=0] | 32
                        for _bi in range(_bop_maxlen):
                            var _b = UInt8(0) if _bi >= _skl else _skp[unsafe_offset=_bi]
                            if _opc == 97:  # 'a' AND
                                _bop_resp[unsafe_offset=_bi] &= _b
                            elif _opc == 111:  # 'o' OR
                                _bop_resp[unsafe_offset=_bi] |= _b
                            elif _opc == 120:  # 'x' XOR
                                _bop_resp[unsafe_offset=_bi] ^= _b
                # Store result
                var _dest_gv = GenericValue()
                _dest_gv.type = ValueType(ValueType.BITMAP)
                _dest_gv._data0 = UInt64(Int(_bop_resp))
                _dest_gv._data1 = UInt64(_bop_maxlen)
                keyspace[].set(_bop_dest, _dest_gv)
                _sso_buf0.unsafe_free()
                _sso_buf_op.unsafe_free()
                writer.append_int_response(Int64(_bop_maxlen))
        return _i - i
    else:
        writer.append_error_response("ERR wrong number of arguments for 'bitop' command")
        return 0


@always_inline
def _bf_signed(raw: Int64, bits: Int, type_str: String) -> Int64:
    """gh #232: sign-extend an `i<N>` BITFIELD value.

    The bit loop accumulates an unsigned magnitude, so `i16` holding -1234 came
    back as 64302 — the right bits read as the wrong number. `u<N>` is returned
    unchanged. Redis rejects a type whose first byte is neither `i` nor `u`; a
    non-`i` prefix here is simply treated as unsigned, matching the existing
    lenient parse of the width."""
    if bits >= 64 or type_str.byte_length() == 0:
        return raw
    if type_str.unsafe_ptr()[unsafe_offset=0] != 105:      # 'i'
        return raw
    var sign_bit = Int64(1) << Int64(bits - 1)
    if (raw & sign_bit) == 0:
        return raw
    return raw - (Int64(1) << Int64(bits))


@always_inline
def _bf_wrap(raw: Int64, bits: Int, type_str: String) -> Int64:
    """gh #232: BITFIELD INCRBY defaults to WRAP overflow, so the REPLY must be
    the value as it now sits in the field.

    Only `bits` bits are written back, so the stored bitmap already wrapped
    correctly — but the reply was computed before the truncation, and
    `INCRBY u16 0 1` on 65535 answered 65536: a number the field cannot hold and
    that a subsequent GET does not return. Mask to the width, then sign-extend
    so `i8` wraps to -128 rather than 128."""
    if bits >= 64:
        return raw
    var mask = (Int64(1) << Int64(bits)) - 1
    return _bf_signed(raw & mask, bits, type_str)


@always_inline
def handle_bitpos(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """BITPOS key bit [start [end [BYTE|BIT]]] → integer position of first bit."""
    if i + 2 < num_tokens:
        var _bpk = tokens[unsafe_offset=i+1].value()
        var _bpbit = strict_atol(tokens[unsafe_offset=i+2].value())  # 0 or 1
        var _bpv = keyspace[].get(_bpk)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        if not _bpv.is_none() and not _bpv.is_string_like():
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _bpp = null_ptr[UInt8, MutUntrackedOrigin]()
        var _bpblen = 0
        var _sbuf_bp = alloc[UInt8](24)
        if _bpv.type.value == ValueType.BITMAP:
            _bpp = _bpv.as_bitmap(); _bpblen = _bpv.bitmap_len()
        elif _bpv.type.value == ValueType.STRING or _bpv.type.value == ValueType.STRING_SSO:
            _bpblen = _bpv.string_len()
            _bpp = _bpv.as_string_safe(_sbuf_bp)
        var _bp_start = 0; var _bp_end = _bpblen - 1
        var _bp_cons = 2
        if i + 3 < num_tokens: _bp_start = strict_atol(tokens[unsafe_offset=i+3].value()); _bp_cons = 3
        if i + 4 < num_tokens: _bp_end = strict_atol(tokens[unsafe_offset=i+4].value()); _bp_cons = 4
        if i + 5 < num_tokens: _bp_cons = 5  # ignore BYTE/BIT modifier
        if _bp_start < 0: _bp_start = max(0, _bpblen + _bp_start)
        if _bp_end < 0: _bp_end = _bpblen + _bp_end
        if _bp_end >= _bpblen: _bp_end = _bpblen - 1
        var _bp_found = -1
        if _bpblen > 0 and _bp_start <= _bp_end:
            for _by in range(_bp_start, _bp_end + 1):
                var _byte = _bpp[unsafe_offset=_by]
                for _bi in range(8):
                    # gh #232: scan MSB-first. Bit 0 of a byte is its most
                    # significant bit, so an LSB-first scan reports the mirror
                    # position — `SETBIT k 2 1` then `BITPOS k 1` answered 5.
                    var _bval = Int((_byte >> UInt8(7 - _bi)) & 1)
                    if _bval == _bpbit:
                        _bp_found = _by * 8 + _bi; break
                if _bp_found >= 0: break
        if _bpbit == 0 and _bp_found < 0 and _bpblen > 0:
            _bp_found = _bpblen * 8  # first 0 bit after all bytes
        writer.append_int_response(Int64(_bp_found))
        _sbuf_bp.unsafe_free()
        return _bp_cons
    else:
        writer.append_error_response("ERR wrong number of arguments for 'bitpos' command")
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


@always_inline
def handle_bitfield(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """BITFIELD key [GET type offset] [SET type offset value] [INCRBY type offset increment] [OVERFLOW ...] → array of integers."""
    if i + 1 < num_tokens:
        var _bfk = tokens[unsafe_offset=i+1].value()
        var _bfv = keyspace[].get(_bfk)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        if not _bfv.is_none() and not _bfv.is_string_like():
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _bfp: Pointer[UInt8, MutUntrackedOrigin]
        var _bfblen = 0
        # gh #232 §4 read-only half, finished: a bitmap IS a string, so
        # `SET k hello; BITFIELD k GET u8 0` must read 104 ('h'). It returned 0
        # — a STRING fell into the "no key yet" branch below and was read as an
        # empty buffer. GETRANGE on the same key already returned 'h', so the
        # bytes were there; only BITFIELD could not see them. A wrong VALUE,
        # not an error, which is the shape that reaches production.
        var _bf_scratch = stack_allocation[32, UInt8]()
        var _bf_str_src = False
        if _bfv.type.value == ValueType.BITMAP:
            _bfp = _bfv.as_bitmap(); _bfblen = _bfv.bitmap_len()
        elif _bfv.is_string():
            # Read-only view. STRING_SSO holds its bytes inside the value, so
            # bitmap_view copies them out to scratch rather than dereferencing
            # a length-and-characters word as an address.
            _bfp = _bfv.bitmap_view(_bf_scratch, _bfblen)
            _bf_str_src = True
        else:
            # gh #232: a missing key used to be materialized as a FIXED 8-byte
            # bitmap, so `BITFIELD k SET u8 0 255` produced an 8-byte string
            # where Redis produces a 1-byte one, and a read-only
            # `BITFIELD k GET u8 0` CREATED the key. Start empty: the growth
            # path below allocates exactly what a write needs, and a pure GET
            # leaves the keyspace untouched.
            _bfblen = 0; _bfp = alloc[UInt8](1); unsafe_memset(_bfp, 0, 1)
        # Count subcommands
        var _bfni = i + 2; var _bfresults = List[Int64]()
        while _bfni < num_tokens:
            var _bfop = tokens[unsafe_offset=_bfni].ptr; var _bfol = tokens[unsafe_offset=_bfni].length
            if _bfol == 8 and (_bfop[unsafe_offset=0]|0x20)==111:  # OVERFLOW - skip next token
                _bfni += 2; continue
            if _bfol < 3: break
            var _bfop_c = _bfop[unsafe_offset=0]|0x20
            if _bfop_c == 103:  # GET type offset
                if _bfni + 2 >= num_tokens: break
                var _bftype = tokens[unsafe_offset=_bfni+1].value()
                var _bfoff = strict_atol(tokens[unsafe_offset=_bfni+2].value())
                # 0 <= offset < 2^32 (Redis's 512 MB bitmap). Unchecked, SET u8 -8 wrote
                # before the buffer and SET u8 9223372036854775800 killed the server.
                if _bfoff < 0 or _bfoff >= 4294967296:
                    raise Error("ERR bit offset is not an integer or out of range")
                var _bfbits = 64
                if _bftype.byte_length() > 1:
                    var _ns = String("")
                    for _ci in range(1, _bftype.byte_length()): _ns += chr(Int(_bftype.unsafe_ptr()[unsafe_offset=_ci]))
                    _bfbits = atol(_ns)
                _bfbits = min(64, max(1, _bfbits))
                var _bfval: Int64 = 0
                for _bi in range(_bfbits):
                    var _bit_off = _bfoff + _bi
                    var _by = _bit_off // 8; var _bitn = 7 - (_bit_off % 8)  # gh #232: Redis numbers bits MSB-first
                    if _by < _bfblen:
                        _bfval |= Int64((_bfp[unsafe_offset=_by] >> UInt8(_bitn)) & 1) << Int64(_bfbits - 1 - _bi)
                _bfresults.append(_bf_signed(_bfval, _bfbits, _bftype)); _bfni += 3
            elif _bfop_c == 115:  # SET type offset value
                # gh #232: the fence is READ-ONLY. `_bfp` here is the STRING's
                # own payload (or the SSO scratch copy), and the write path
                # below both mutates it in place and re-stamps the key as
                # BITMAP on growth — which would corrupt the string, and free
                # a gh #163 blob-arena pointer the heap allocator must never
                # touch. Mutating the union needs arena-aware promotion first.
                if _bf_str_src:
                    writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    return 0
                if _bfni + 3 >= num_tokens: break
                var _bftype = tokens[unsafe_offset=_bfni+1].value()
                var _bfoff = strict_atol(tokens[unsafe_offset=_bfni+2].value())
                # 0 <= offset < 2^32 (Redis's 512 MB bitmap). Unchecked, SET u8 -8 wrote
                # before the buffer and SET u8 9223372036854775800 killed the server.
                if _bfoff < 0 or _bfoff >= 4294967296:
                    raise Error("ERR bit offset is not an integer or out of range")
                var _bfnew = Int64(strict_atol(tokens[unsafe_offset=_bfni+3].value()))
                var _bfbits = 64
                if _bftype.byte_length() > 1:
                    var _ns = String("")
                    for _ci in range(1, _bftype.byte_length()): _ns += chr(Int(_bftype.unsafe_ptr()[unsafe_offset=_ci]))
                    _bfbits = atol(_ns)
                _bfbits = min(64, max(1, _bfbits))
                # Read old value
                var _bfold: Int64 = 0
                for _bi in range(_bfbits):
                    var _bit_off = _bfoff + _bi
                    var _by = _bit_off // 8; var _bitn = 7 - (_bit_off % 8)  # gh #232: Redis numbers bits MSB-first
                    if _by < _bfblen:
                        _bfold |= Int64((_bfp[unsafe_offset=_by] >> UInt8(_bitn)) & 1) << Int64(_bfbits - 1 - _bi)
                # Expand bitmap if needed
                var _need = (_bfoff + _bfbits - 1) // 8 + 1
                if _need > _bfblen:
                    var _newp = alloc[UInt8](_need); unsafe_memcpy(dest=_newp, src=_bfp, count=_bfblen)
                    unsafe_memset(_newp.unsafe_offset(_bfblen), 0, _need - _bfblen)
                    _bfp = _newp
                    _bfblen = _need
                    var _upd = keyspace[].get(_bfk)
                    # The key may not exist yet — a BITFIELD write is what
                    # creates it, so stamp the type rather than assuming it.
                    _upd.type = ValueType(ValueType.BITMAP)
                    _upd._data0 = UInt64(Int(_bfp)); _upd._data1 = UInt64(_bfblen)
                    keyspace[].set(_bfk, _upd)
                # Write new value bit by bit
                for _bi in range(_bfbits):
                    var _bit_off = _bfoff + _bi
                    var _by = _bit_off // 8; var _bitn = 7 - (_bit_off % 8)  # gh #232: Redis numbers bits MSB-first
                    var _bbit = UInt8((_bfnew >> Int64(_bfbits - 1 - _bi)) & 1)
                    if _bbit == 1: _bfp[unsafe_offset=_by] |= UInt8(1 << _bitn)
                    else: _bfp[unsafe_offset=_by] &= ~UInt8(1 << _bitn)
                _bfresults.append(_bf_signed(_bfold, _bfbits, _bftype)); _bfni += 4
            elif _bfop_c == 105:  # INCRBY type offset increment
                if _bf_str_src:   # gh #232: read-only fence, as for SET above
                    writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    return 0
                if _bfni + 3 >= num_tokens: break
                var _bftype = tokens[unsafe_offset=_bfni+1].value()
                var _bfoff = strict_atol(tokens[unsafe_offset=_bfni+2].value())
                # 0 <= offset < 2^32 (Redis's 512 MB bitmap). Unchecked, SET u8 -8 wrote
                # before the buffer and SET u8 9223372036854775800 killed the server.
                if _bfoff < 0 or _bfoff >= 4294967296:
                    raise Error("ERR bit offset is not an integer or out of range")
                var _bfinc = Int64(strict_atol(tokens[unsafe_offset=_bfni+3].value()))
                var _bfbits = 64
                if _bftype.byte_length() > 1:
                    var _ns = String("")
                    for _ci in range(1, _bftype.byte_length()): _ns += chr(Int(_bftype.unsafe_ptr()[unsafe_offset=_ci]))
                    _bfbits = atol(_ns)
                _bfbits = min(64, max(1, _bfbits))
                var _bfcur: Int64 = 0
                for _bi in range(_bfbits):
                    var _bit_off = _bfoff + _bi
                    var _by = _bit_off // 8; var _bitn = 7 - (_bit_off % 8)  # gh #232: Redis numbers bits MSB-first
                    if _by < _bfblen:
                        _bfcur |= Int64((_bfp[unsafe_offset=_by] >> UInt8(_bitn)) & 1) << Int64(_bfbits - 1 - _bi)
                var _bfresval = _bf_wrap(_bf_signed(_bfcur, _bfbits, _bftype) + _bfinc, _bfbits, _bftype)
                var _need = (_bfoff + _bfbits - 1) // 8 + 1
                if _need > _bfblen:
                    var _newp = alloc[UInt8](_need); unsafe_memcpy(dest=_newp, src=_bfp, count=_bfblen)
                    unsafe_memset(_newp.unsafe_offset(_bfblen), 0, _need - _bfblen)
                    _bfp = _newp
                    _bfblen = _need
                    var _upd = keyspace[].get(_bfk)
                    # The key may not exist yet — a BITFIELD write is what
                    # creates it, so stamp the type rather than assuming it.
                    _upd.type = ValueType(ValueType.BITMAP)
                    _upd._data0 = UInt64(Int(_bfp)); _upd._data1 = UInt64(_bfblen)
                    keyspace[].set(_bfk, _upd)
                for _bi in range(_bfbits):
                    var _bit_off = _bfoff + _bi
                    var _by = _bit_off // 8; var _bitn = 7 - (_bit_off % 8)  # gh #232: Redis numbers bits MSB-first
                    var _bbit = UInt8((_bfresval >> Int64(_bfbits - 1 - _bi)) & 1)
                    if _bbit == 1: _bfp[unsafe_offset=_by] |= UInt8(1 << _bitn)
                    else: _bfp[unsafe_offset=_by] &= ~UInt8(1 << _bitn)
                _bfresults.append(_bfresval); _bfni += 4
            else: break
        var _bf_rh = "*" + String(len(_bfresults)) + "\r\n"
        writer.append_to_response(_bf_rh.unsafe_ptr(), _bf_rh.byte_length())
        for _ri in range(len(_bfresults)): writer.append_int_response(_bfresults[_ri])
        return _bfni - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'bitfield' command")
        return 0


@always_inline
def handle_bitfield_ro(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """BITFIELD_RO key [GET type offset] → array of integers (read-only)."""
    if i + 1 < num_tokens:
        var _bfrk = tokens[unsafe_offset=i+1].value()
        var _bfrv = keyspace[].get(_bfrk)
        var _bfrp = null_ptr[UInt8, MutUntrackedOrigin]()
        var _bfrblen = 0
        if _bfrv.type.value == ValueType.BITMAP:
            _bfrp = _bfrv.as_bitmap(); _bfrblen = _bfrv.bitmap_len()
        var _bfrni = i + 2; var _bfrresults = List[Int64]()
        while _bfrni < num_tokens:
            var _bfrop = tokens[unsafe_offset=_bfrni].ptr; var _bfrol = tokens[unsafe_offset=_bfrni].length
            if _bfrol < 3: break
            if (_bfrop[unsafe_offset=0]|0x20) == 103:  # GET
                if _bfrni + 2 >= num_tokens: break
                var _bfrtype = tokens[unsafe_offset=_bfrni+1].value()
                var _bfroff = strict_atol(tokens[unsafe_offset=_bfrni+2].value())
                var _bfrbits = 64
                if _bfrtype.byte_length() > 1:
                    var _ns = String("")
                    for _ci in range(1, _bfrtype.byte_length()): _ns += chr(Int(_bfrtype.unsafe_ptr()[unsafe_offset=_ci]))
                    _bfrbits = atol(_ns)
                _bfrbits = min(64, max(1, _bfrbits))
                var _bfrval: Int64 = 0
                for _bi in range(_bfrbits):
                    var _bit_off = _bfroff + _bi
                    var _by = _bit_off // 8; var _bitn = 7 - (_bit_off % 8)  # gh #232: Redis numbers bits MSB-first
                    if _by < _bfrblen:
                        _bfrval |= Int64((_bfrp[unsafe_offset=_by] >> UInt8(_bitn)) & 1) << Int64(_bfrbits - 1 - _bi)
                _bfrresults.append(_bf_signed(_bfrval, _bfrbits, _bfrtype)); _bfrni += 3
            else: break
        var _bfr_rh = "*" + String(len(_bfrresults)) + "\r\n"
        writer.append_to_response(_bfr_rh.unsafe_ptr(), _bfr_rh.byte_length())
        for _ri in range(len(_bfrresults)): writer.append_int_response(_bfrresults[_ri])
        return _bfrni - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'bitfield_ro' command")
        return 0
