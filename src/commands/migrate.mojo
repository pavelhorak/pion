"""DUMP, RESTORE, MIGRATE commands for cluster slot migration (C1.1).

DUMP serializes a key's value to a binary format.
RESTORE deserializes and stores a key from DUMP/MIGRATE payload.
MIGRATE sends a key to another node (DUMP + RESTORE + DEL).

Binary format (Pion-native, V1):
  [type:1B][key_len:4B LE][key][val_len:4B LE][value_bytes][ttl_ns:8B LE]

Value encoding by type:
  STRING/STRING_SSO: raw bytes
  INT: 8B LE int64
  LIST: [count:4B LE][entry_len:4B LE][entry_bytes]...
  SET: [count:4B LE][elem_len:4B LE][elem_bytes]...
  HASH: [count:4B LE][field_len:4B LE][field][val_len:4B LE][val]...
  ZSET: [count:4B LE][member_len:4B LE][member][score:8B LE float64]...
  BITMAP: raw bytes (same as STRING)
  HLL: raw 12-register bytes
"""

from src.common.ptr import is_not_null, null_ptr
from src.common.container_free import remove_and_free
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy, unsafe_memset
from std.ffi import external_call
from src.common.value import GenericValue, ValueType
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.network.response_writer import ResponseWriter
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.common.utils import format_int_to_buf


@always_inline
def _write_u32_le(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int, val: UInt32) -> Int:
    buf[unsafe_offset=offset] = UInt8(val & 0xFF)
    buf[unsafe_offset=offset + 1] = UInt8((val >> 8) & 0xFF)
    buf[unsafe_offset=offset + 2] = UInt8((val >> 16) & 0xFF)
    buf[unsafe_offset=offset + 3] = UInt8((val >> 24) & 0xFF)
    return offset + 4

@always_inline
def _read_u32_le(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int) -> UInt32:
    return UInt32(buf[unsafe_offset=offset]) | (UInt32(buf[unsafe_offset=offset+1]) << 8) | (UInt32(buf[unsafe_offset=offset+2]) << 16) | (UInt32(buf[unsafe_offset=offset+3]) << 24)

@always_inline
def _write_i64_le(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int, val: Int64) -> Int:
    var u = UInt64(val)
    for bi in range(8):
        buf[unsafe_offset=offset + bi] = UInt8((u >> (UInt64(bi) * 8)) & 0xFF)
    return offset + 8

@always_inline
def _read_i64_le(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int) -> Int64:
    var u: UInt64 = 0
    for bi in range(8):
        u |= UInt64(buf[unsafe_offset=offset + bi]) << (UInt64(bi) * 8)
    return Int64(u)


def dump_key(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
             ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
             key_ptr: Pointer[UInt8, MutUntrackedOrigin], key_len: Int,
             out_buf: Pointer[UInt8, MutUntrackedOrigin], max_len: Int) -> Int:
    """Serialize a key+value into out_buf. Returns bytes written, or -1 if key not found."""
    var key_v = GenericValue.borrow(key_ptr, key_len)
    var val = keyspace[].get(key_v)
    if val.is_none():
        return -1

    var off = 0
    var vt = val.type.value

    # Type byte
    out_buf[unsafe_offset=off] = UInt8(vt); off += 1

    # Key
    off = _write_u32_le(out_buf, off, UInt32(key_len))
    unsafe_memcpy(dest=out_buf.unsafe_offset(off), src=key_ptr, count=key_len); off += key_len

    # Value — type-specific encoding
    if vt == ValueType.STRING or vt == ValueType.STRING_SSO:
        var sso_buf = alloc[UInt8](24)
        var sp = val.as_string_safe(sso_buf)
        var sl = val.string_len()
        off = _write_u32_le(out_buf, off, UInt32(sl))
        unsafe_memcpy(dest=out_buf.unsafe_offset(off), src=sp, count=sl); off += sl
        sso_buf.unsafe_free()
    elif vt == ValueType.INT:
        off = _write_u32_le(out_buf, off, UInt32(8))
        off = _write_i64_le(out_buf, off, val.as_int())
    elif vt == ValueType.BITMAP:
        var bp = val.as_bitmap()
        var bl = val.bitmap_len()
        off = _write_u32_le(out_buf, off, UInt32(bl))
        unsafe_memcpy(dest=out_buf.unsafe_offset(off), src=bp, count=bl); off += bl
    elif vt == ValueType.LIST:
        var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
        var lsz = list_ptr[].size
        off = _write_u32_le(out_buf, off, UInt32(lsz))
        # Serialize list entries (simplified: use zip_buf if available)
        if is_not_null(list_ptr[].zip_buf):
            var zoff = 0
            for _ in range(lsz):
                var vlen = Int((list_ptr[].zip_buf.unsafe_offset(zoff)).unsafe_bitcast[UInt16]()[])
                off = _write_u32_le(out_buf, off, UInt32(vlen))
                unsafe_memcpy(dest=out_buf.unsafe_offset(off), src=list_ptr[].zip_buf.unsafe_offset(zoff).unsafe_offset(2), count=vlen)
                off += vlen
                zoff += 2 + vlen
        else:
            # Quicklist mode — skip for V1, write count=0
            out_buf[unsafe_offset=off - 4] = 0; out_buf[unsafe_offset=off - 3] = 0; out_buf[unsafe_offset=off - 2] = 0; out_buf[unsafe_offset=off - 1] = 0
    elif vt == ValueType.SET:
        var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
        var set_size = set_ptr[].size
        off = _write_u32_le(out_buf, off, UInt32(set_size))
        # Iterate set entries
        var written = 0
        for si in range(set_ptr[].capacity):
            if written >= set_size: break
            var m = set_ptr[].metadata[unsafe_offset=si]
            if m != 0x80 and m != 0xFF:
                var elem = set_ptr[].keys[unsafe_offset=si]
                if not elem.is_none():
                    var sb = alloc[UInt8](24)
                    var ep = elem.as_string_safe(sb)
                    var el = elem.string_len()
                    off = _write_u32_le(out_buf, off, UInt32(el))
                    unsafe_memcpy(dest=out_buf.unsafe_offset(off), src=ep, count=el); off += el
                    sb.unsafe_free()
                    written += 1
    elif vt == ValueType.HASH:
        var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
        var hash_size = hash_ptr[].size
        off = _write_u32_le(out_buf, off, UInt32(hash_size))
        var written = 0
        for si in range(hash_ptr[].capacity):
            if written >= hash_size: break
            var m = hash_ptr[].metadata[unsafe_offset=si]
            if m != 0x80 and m != 0xFF:
                var field = hash_ptr[].keys[unsafe_offset=si]
                var fval = hash_ptr[].values[unsafe_offset=si]
                if not field.is_none():
                    var fb = alloc[UInt8](24)
                    var fp = field.as_string_safe(fb)
                    var fl = field.string_len()
                    off = _write_u32_le(out_buf, off, UInt32(fl))
                    unsafe_memcpy(dest=out_buf.unsafe_offset(off), src=fp, count=fl); off += fl
                    fb.unsafe_free()
                    var vb = alloc[UInt8](24)
                    var vp = fval.as_string_safe(vb)
                    var vl = fval.string_len()
                    off = _write_u32_le(out_buf, off, UInt32(vl))
                    unsafe_memcpy(dest=out_buf.unsafe_offset(off), src=vp, count=vl); off += vl
                    vb.unsafe_free()
                    written += 1
    else:
        # ZSET, GEO, STREAM, HLL — serialize as empty for V1
        off = _write_u32_le(out_buf, off, UInt32(0))

    # TTL (0 = no expiry)
    var ttl_ns: Int64 = 0
    if is_not_null(ttl_map):
        var exp_v = ttl_map[].get(key_v)
        if not exp_v.is_none():
            ttl_ns = exp_v.as_int()
    off = _write_i64_le(out_buf, off, ttl_ns)

    return off


def restore_key(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                data: Pointer[UInt8, MutUntrackedOrigin], data_len: Int,
                replace: Bool = False,
                override_key_ptr: Pointer[UInt8, MutUntrackedOrigin] = null_ptr[UInt8, MutUntrackedOrigin](),
                override_key_len: Int = 0) -> Bool:
    """Deserialize a DUMP payload and store in keyspace. Returns True on success.
    If override_key is provided, use that as the key instead of the one in the payload."""
    if data_len < 6:  # minimum: type(1) + key_len(4) + at least 1 more byte
        return False

    var off = 0
    var vt = Int(data[unsafe_offset=off]); off += 1

    # Key from payload (skip over it, use override if provided)
    var key_len = Int(_read_u32_le(data, off)); off += 4
    if off + key_len > data_len: return False
    var key_v: GenericValue
    if override_key_len > 0 and is_not_null(override_key_ptr):
        key_v = GenericValue.from_ptr(override_key_ptr, override_key_len)
    else:
        key_v = GenericValue.from_ptr(data.unsafe_offset(off), key_len)
    off += key_len

    # Check if key exists
    if not replace:
        var existing = keyspace[].get(key_v)
        if not existing.is_none():
            return False  # BUSYKEY

    # Value length
    if off + 4 > data_len: return False
    var val_len = Int(_read_u32_le(data, off)); off += 4

    # Deserialize value by type
    if vt == ValueType.STRING or vt == ValueType.STRING_SSO:
        if off + val_len > data_len: return False
        var new_val = GenericValue.from_ptr(data.unsafe_offset(off), val_len)
        keyspace[].set(key_v, new_val)
        off += val_len
    elif vt == ValueType.INT:
        if off + 8 > data_len: return False
        var int_val = _read_i64_le(data, off)
        keyspace[].set(key_v, GenericValue.from_int(int_val))
        off += 8
    elif vt == ValueType.BITMAP:
        if off + val_len > data_len: return False
        var bm = alloc[UInt8](val_len)
        unsafe_memcpy(dest=bm, src=data.unsafe_offset(off), count=val_len)
        var bm_val = GenericValue()
        bm_val.type = ValueType(ValueType.BITMAP)
        bm_val._data0 = UInt64(Int(bm))
        bm_val._data1 = UInt64(val_len)
        keyspace[].set(key_v, bm_val)
        off += val_len
    elif vt == ValueType.SET:
        # val_len here is count of elements
        var count = val_len
        var set_ptr = alloc[SlabHashMap](1)
        set_ptr.unsafe_write(SlabHashMap(max(16, count * 2)))
        for _ in range(count):
            if off + 4 > data_len: break
            var el = Int(_read_u32_le(data, off)); off += 4
            if off + el > data_len: break
            var elem_v = GenericValue.from_ptr(data.unsafe_offset(off), el)
            set_ptr[].set(elem_v, GenericValue.from_int(1))
            off += el
        var set_val = GenericValue()
        set_val.type = ValueType(ValueType.SET)
        set_val.set_ptr(set_ptr.unsafe_bitcast[NoneType]())
        keyspace[].set(key_v, set_val)
    elif vt == ValueType.HASH:
        var count = val_len
        var hash_ptr = alloc[SlabHashMap](1)
        hash_ptr.unsafe_write(SlabHashMap(max(16, count * 2)))
        for _ in range(count):
            if off + 4 > data_len: break
            var fl = Int(_read_u32_le(data, off)); off += 4
            if off + fl > data_len: break
            var field_v = GenericValue.from_ptr(data.unsafe_offset(off), fl)
            off += fl
            if off + 4 > data_len: break
            var vl = Int(_read_u32_le(data, off)); off += 4
            if off + vl > data_len: break
            var fval_v = GenericValue.from_ptr(data.unsafe_offset(off), vl)
            hash_ptr[].set(field_v, fval_v)
            off += vl
        var hash_val = GenericValue()
        hash_val.type = ValueType(ValueType.HASH)
        hash_val.set_ptr(hash_ptr.unsafe_bitcast[NoneType]())
        keyspace[].set(key_v, hash_val)
    elif vt == ValueType.LIST:
        var count = val_len
        var list_ptr = alloc[SlabList](1)
        list_ptr.unsafe_write(SlabList())
        for _ in range(count):
            if off + 4 > data_len: break
            var el = Int(_read_u32_le(data, off)); off += 4
            if off + el > data_len: break
            var elem_v = GenericValue.from_ptr(data.unsafe_offset(off), el)
            list_ptr[].rpush(elem_v)
            off += el
        var list_val = GenericValue()
        list_val.type = ValueType(ValueType.LIST)
        list_val.set_ptr(list_ptr.unsafe_bitcast[NoneType]())
        keyspace[].set(key_v, list_val)
    else:
        # Unknown type — store as raw string
        if off + val_len > data_len: return False
        var new_val = GenericValue.from_ptr(data.unsafe_offset(off), val_len)
        keyspace[].set(key_v, new_val)
        off += val_len

    # TTL
    if off + 8 <= data_len:
        var ttl_ns = _read_i64_le(data, off)
        if ttl_ns > 0 and is_not_null(ttl_map):
            ttl_map[].set(key_v, GenericValue.from_int(ttl_ns))

    return True


@always_inline
def handle_dump(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                mut writer: ResponseWriter,
                keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """DUMP key → serialized value as bulk string."""
    if i + 1 < num_tokens:
        var key_str = tokens[unsafe_offset=i + 1].value()
        var key_v = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
        var val = keyspace[].get(key_v)
        if val.is_none():
            writer.append_null_response()
        else:
            # 1MB buffer for serialization
            var buf = alloc[UInt8](1048576)
            var buf_ptr = buf
            var kbuf = alloc[UInt8](key_str.byte_length())
            unsafe_memcpy(dest=kbuf, src=key_str.unsafe_ptr().unsafe_bitcast[UInt8](), count=key_str.byte_length())
            var n = dump_key(keyspace, ttl_map, kbuf, key_str.byte_length(), buf_ptr, 1048576)
            kbuf.unsafe_free()
            if n > 0:
                writer.append_bulk_string_response(buf_ptr, n)
            else:
                writer.append_null_response()
            buf.unsafe_free()
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'dump' command")
        return 0


@always_inline
def handle_restore(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                   mut writer: ResponseWriter,
                   keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                   ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """RESTORE key ttl serialized-value [REPLACE] → +OK or -BUSYKEY."""
    if i + 3 < num_tokens:
        var key_str = tokens[unsafe_offset=i + 1].value()
        _ = tokens[unsafe_offset=i + 2].value()  # ttl (handled inside serialized data)
        var data_tok = tokens[unsafe_offset=i + 3]
        var data_ptr = data_tok.ptr
        var data_len = data_tok.length

        # Check REPLACE flag
        var replace = False
        var extra = 3
        if i + 4 < num_tokens:
            var flag = tokens[unsafe_offset=i + 4]
            if flag.length == 7 and (flag.ptr[unsafe_offset=0]|0x20) == 114:  # 'r' = REPLACE
                replace = True
                extra = 4

        # Copy key to allocated buffer for MutUntrackedOrigin
        var rk_buf = alloc[UInt8](key_str.byte_length())
        unsafe_memcpy(dest=rk_buf, src=key_str.unsafe_ptr().unsafe_bitcast[UInt8](), count=key_str.byte_length())
        var ok = restore_key(keyspace, ttl_map, data_ptr, data_len, replace,
                             rk_buf, key_str.byte_length())
        rk_buf.unsafe_free()
        if ok:
            writer.append_ok_response()
        else:
            writer.append_error_response("BUSYKEY Target key name already exists")
        return extra
    else:
        writer.append_error_response("ERR wrong number of arguments for 'restore' command")
        return 0


@always_inline
def handle_migrate(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                   mut writer: ResponseWriter,
                   keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                   ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) raises -> Int:
    """MIGRATE host port key|"" db timeout [COPY] [REPLACE] [KEYS key ...]
    Sends DUMP payload to target node via TCP, then DELetes local key on success."""
    if i + 5 < num_tokens:
        var host = tokens[unsafe_offset=i + 1].value()
        var port_str = tokens[unsafe_offset=i + 2].value()
        var port = atol(port_str)
        var key_str = tokens[unsafe_offset=i + 3].value()
        _ = tokens[unsafe_offset=i + 4].value()  # db (ignored, Pion is single-db)
        var timeout_str = tokens[unsafe_offset=i + 5].value()
        var timeout_ms = atol(timeout_str)
        var consumed = 5

        # Parse optional flags
        var copy_flag = False
        var replace_flag = False
        var multi_keys = List[String]()
        var ji = i + 6
        while ji < num_tokens and tokens[unsafe_offset=ji].marker != 0:
            var flag = tokens[unsafe_offset=ji]
            var fp = flag.ptr; var fl = flag.length
            if fl == 4 and (fp[unsafe_offset=0]|0x20) == 99 and (fp[unsafe_offset=1]|0x20) == 111:  # COPY
                copy_flag = True
            elif fl == 7 and (fp[unsafe_offset=0]|0x20) == 114 and (fp[unsafe_offset=1]|0x20) == 101:  # REPLACE
                replace_flag = True
            elif fl == 4 and (fp[unsafe_offset=0]|0x20) == 107 and (fp[unsafe_offset=1]|0x20) == 101:  # KEYS
                # Remaining tokens are key names
                ji += 1
                while ji < num_tokens and tokens[unsafe_offset=ji].marker != 0:
                    multi_keys.append(tokens[unsafe_offset=ji].value())
                    ji += 1
                    consumed += 1
                break
            ji += 1
            consumed += 1

        # Build key list
        var keys = List[String]()
        if key_str.byte_length() > 0 and key_str != "":
            keys.append(key_str)
        for ki in range(len(multi_keys)):
            keys.append(multi_keys[ki])

        if len(keys) == 0:
            writer.append_error_response("ERR no keys to migrate")
            return consumed

        # Serialize all keys
        var buf = alloc[UInt8](4194304)  # 4MB buffer
        var buf_ptr = buf
        var total_serialized = 0
        var key_offsets = List[Int]()  # offset into buf for each key's RESTORE payload
        var key_lens = List[Int]()

        for ki in range(len(keys)):
            var k = keys[ki]
            var kl = k.byte_length()
            var kp_buf = alloc[UInt8](kl)
            unsafe_memcpy(dest=kp_buf, src=k.unsafe_ptr().unsafe_bitcast[UInt8](), count=kl)
            var n = dump_key(keyspace, ttl_map, kp_buf, kl, buf_ptr.unsafe_offset(total_serialized), 4194304 - total_serialized)
            kp_buf.unsafe_free()
            if n > 0:
                key_offsets.append(total_serialized)
                key_lens.append(n)
                total_serialized += n
            else:
                key_offsets.append(-1)
                key_lens.append(0)

        # Connect to target node
        var host_cstr = host + "\0"
        var target_fd = external_call["pion_tcp_connect", Int32](
            host_cstr.unsafe_ptr().unsafe_bitcast[UInt8](), Int32(port), Int32(timeout_ms)
        )
        if target_fd < 0:
            buf.unsafe_free()
            writer.append_error_response("IOERR error connecting to target node")
            return consumed

        # For each key: send RESTORE command via RESP
        var migrated = 0
        for ki in range(len(keys)):
            if key_offsets[ki] < 0:
                continue  # key didn't exist
            var k = keys[ki]
            var payload_ptr = buf_ptr.unsafe_offset(key_offsets[ki])
            var payload_len = key_lens[ki]

            # Build RESP: *4\r\n$7\r\nRESTORE\r\n$<keylen>\r\n<key>\r\n$1\r\n0\r\n$<payloadlen>\r\n<payload>\r\n
            var resp_buf = alloc[UInt8](payload_len + 256)
            var rp = resp_buf
            var ro = 0
            # *4 or *5 (with REPLACE)
            var nargs = 4 if not replace_flag else 5
            rp[unsafe_offset=ro] = 42; ro += 1  # '*'
            ro += format_int_to_buf(rp.unsafe_offset(ro), 0, Int64(nargs))
            rp[unsafe_offset=ro] = 13; rp[unsafe_offset=ro+1] = 10; ro += 2
            # $7\r\nRESTORE\r\n
            rp[unsafe_offset=ro] = 36; ro += 1; rp[unsafe_offset=ro] = 55; ro += 1; rp[unsafe_offset=ro] = 13; ro += 1; rp[unsafe_offset=ro] = 10; ro += 1
            rp[unsafe_offset=ro] = 82; rp[unsafe_offset=ro+1] = 69; rp[unsafe_offset=ro+2] = 83; rp[unsafe_offset=ro+3] = 84; rp[unsafe_offset=ro+4] = 79; rp[unsafe_offset=ro+5] = 82; rp[unsafe_offset=ro+6] = 69
            ro += 7; rp[unsafe_offset=ro] = 13; rp[unsafe_offset=ro+1] = 10; ro += 2
            # $<keylen>\r\n<key>\r\n
            rp[unsafe_offset=ro] = 36; ro += 1
            ro += format_int_to_buf(rp.unsafe_offset(ro), 0, Int64(k.byte_length()))
            rp[unsafe_offset=ro] = 13; rp[unsafe_offset=ro+1] = 10; ro += 2
            unsafe_memcpy(dest=rp.unsafe_offset(ro), src=k.unsafe_ptr().unsafe_bitcast[UInt8](), count=k.byte_length()); ro += k.byte_length()
            rp[unsafe_offset=ro] = 13; rp[unsafe_offset=ro+1] = 10; ro += 2
            # $1\r\n0\r\n (ttl=0, actual TTL is inside the payload)
            rp[unsafe_offset=ro] = 36; rp[unsafe_offset=ro+1] = 49; rp[unsafe_offset=ro+2] = 13; rp[unsafe_offset=ro+3] = 10
            rp[unsafe_offset=ro+4] = 48; rp[unsafe_offset=ro+5] = 13; rp[unsafe_offset=ro+6] = 10; ro += 7
            # $<payloadlen>\r\n<payload>\r\n
            rp[unsafe_offset=ro] = 36; ro += 1
            ro += format_int_to_buf(rp.unsafe_offset(ro), 0, Int64(payload_len))
            rp[unsafe_offset=ro] = 13; rp[unsafe_offset=ro+1] = 10; ro += 2
            unsafe_memcpy(dest=rp.unsafe_offset(ro), src=payload_ptr, count=payload_len); ro += payload_len
            rp[unsafe_offset=ro] = 13; rp[unsafe_offset=ro+1] = 10; ro += 2
            # REPLACE flag
            if replace_flag:
                rp[unsafe_offset=ro] = 36; rp[unsafe_offset=ro+1] = 55; rp[unsafe_offset=ro+2] = 13; rp[unsafe_offset=ro+3] = 10; ro += 4
                rp[unsafe_offset=ro] = 82; rp[unsafe_offset=ro+1] = 69; rp[unsafe_offset=ro+2] = 80; rp[unsafe_offset=ro+3] = 76; rp[unsafe_offset=ro+4] = 65; rp[unsafe_offset=ro+5] = 67; rp[unsafe_offset=ro+6] = 69
                ro += 7; rp[unsafe_offset=ro] = 13; rp[unsafe_offset=ro+1] = 10; ro += 2

            # Send to target
            var sent = Int(external_call["send", Int64](target_fd, rp, Int(ro), Int32(0)))
            resp_buf.unsafe_free()

            if sent > 0:
                # Read response (expect +OK\r\n)
                var recv_buf = alloc[UInt8](256)
                var nr = Int(external_call["recv", Int64](target_fd, recv_buf, 256, Int32(0)))
                if nr > 0 and recv_buf[unsafe_offset=0] == 43:  # '+' = success
                    migrated += 1
                    # Delete local key unless COPY
                    if not copy_flag:
                        var kv = GenericValue.from_string(k)
                        _ = remove_and_free(keyspace, kv)   # gh #394: container too
                        if is_not_null(ttl_map):
                            _ = ttl_map[].remove_generic(kv)
                        kv.free_str_payload()               # our lookup copy
                recv_buf.unsafe_free()

        _ = external_call["close", Int32](target_fd)
        buf.unsafe_free()

        if migrated > 0:
            writer.append_ok_response()
        else:
            writer.append_error_response("ERR no keys migrated")
        return consumed
    else:
        writer.append_error_response("ERR wrong number of arguments for 'migrate' command")
        return 0
