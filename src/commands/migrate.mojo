"""DUMP, RESTORE, RESTORE-ASKING and MIGRATE (#41).

A DUMP payload is the key's value as the snapshot writes it: WAL-format records
(src/io/snapshot.mojo, write_key_records), so every type round-trips, followed
by a footer laid out as Redis's: a 2-byte format version and an 8-byte
CRC-64/Jones of everything before it, both little-endian. The version is Pion's
own (PION_DUMP_VERSION), above every RDB version, so a payload Pion wrote is
refused by Redis and one Redis wrote is refused by Pion, each with
"DUMP payload version or checksum are wrong". The TTL is not in the payload:
RESTORE's argument sets it, as in Redis.

This replaced a format that serialized sorted sets, geo keys, streams,
HyperLogLogs, vector sets and long lists as empty, carried the source key's
absolute deadline in place of RESTORE's TTL argument, had no checksum, and was
built in a fixed 1 MB buffer with no bound. RESTORE also logged nothing, so a
restored key was gone after a restart, and MIGRATE did not log the deletion of
the keys it moved, so they came back on the source.
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.container_free import remove_and_free
from std.memory.unsafe_pointer import Pointer
from std.collections import Span
from std.memory import alloc, unsafe_memcpy
from std.ffi import external_call
from src.common.value import GenericValue, ValueType
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.network.response_writer import ResponseWriter
from src.network.resp3 import RESP3Token
from src.network.fast_path import _get_now_ns
from src.common.utils import format_int_to_buf, parse_int64_strict, arg_eq, ms_to_deadline_ns
from src.io.wal import WAL, wal_apply_aggregate
from src.io.snapshot import RecordSink, write_key_records
from src.io.blob_store import BlobStore

comptime PION_DUMP_VERSION = 0x5001


def _u32(p: Pointer[UInt8, MutUntrackedOrigin], at: Int) -> Int:
    return Int(p[at]) | (Int(p[at + 1]) << 8) | (Int(p[at + 2]) << 16) | (Int(p[at + 3]) << 24)


def dump_payload(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                 kp: Pointer[UInt8, MutUntrackedOrigin], kl: Int) -> List[UInt8]:
    """The DUMP payload of a key, or an empty list when it does not exist."""
    var v = keyspace[].get(GenericValue.borrow(kp, kl))
    if v.is_none():
        return List[UInt8]()
    var sink = RecordSink(Int32(-1))
    var ehdr = alloc[UInt8](16)
    var vbuf = alloc[UInt8](64)
    var fbuf = alloc[UInt8](64)
    var scored = alloc[UInt8](8)
    write_key_records(sink, kp, kl, v, null_ptr[BlobStore, MutUntrackedOrigin](), True, ehdr, vbuf, fbuf, scored)
    ehdr.unsafe_free()
    vbuf.unsafe_free()
    fbuf.unsafe_free()
    scored.unsafe_free()
    var out = sink^.take_buf()
    out.append(UInt8(PION_DUMP_VERSION & 0xFF))
    out.append(UInt8(PION_DUMP_VERSION >> 8))
    var crc = external_call["pion_crc64", UInt64](UInt64(0), out.unsafe_ptr(), Int64(len(out)))
    for k in range(8):
        out.append(UInt8((crc >> UInt64(8 * k)) & 0xFF))
    return out^


def handle_dump(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) -> Int:
    """DUMP key → the payload, or nil for a missing key."""
    if num_tokens - i != 2:
        writer.append_error_response("ERR wrong number of arguments for 'dump' command")
        return 0
    var payload = dump_payload(keyspace, tokens[i + 1].ptr, tokens[i + 1].length)
    if len(payload) == 0:
        writer.append_null_response()
    else:
        writer.append_bulk_string_response(Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(payload.unsafe_ptr())), len(payload))
    _ = payload^
    return 1


def _payload_ok(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    """The footer: Pion's version, then the CRC of everything before the CRC
    (the version included, as Redis's verifyDumpPayload)."""
    if n < 10:
        return False
    var ver = Int(p[n - 10]) | (Int(p[n - 9]) << 8)
    if ver != PION_DUMP_VERSION:
        return False
    var want = UInt64(0)
    for k in range(8):
        want |= UInt64(Int(p[n - 8 + k])) << UInt64(8 * k)
    return external_call["pion_crc64", UInt64](UInt64(0), p, Int64(n - 8)) == want


def _record_family(cmd: Int) -> Int:
    """The value type a DUMP record belongs to (0 = not a DUMP record)."""
    if cmd == 1:
        return 1                    # string
    if cmd == 5 or cmd == 32:
        return 2                    # hash, its field TTLs
    if cmd == 7:
        return 3                    # list
    if cmd == 8:
        return 4                    # set
    if cmd == 9:
        return 5                    # sorted set
    if cmd == 15:
        return 6                    # geo
    if cmd == 16:
        return 7                    # bitmap
    if cmd == 17:
        return 8                    # HyperLogLog
    if cmd == 34 or cmd == 38 or cmd == 41 or cmd == 43 or cmd == 45:
        return 9                    # stream: entries, metadata, groups (#40)
    if cmd == 28 or cmd == 30:
        return 10                   # vector set
    return 0


def _u64(p: Pointer[UInt8, MutUntrackedOrigin], at: Int) -> UInt64:
    var v = UInt64(0)
    for k in range(8):
        v |= UInt64(Int(p[at + k])) << UInt64(8 * k)
    return v


def _records_ok(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    """Every record is whole, has the layout its kind expects, and they all
    belong to one value type; a stream's IDs increase and a sorted set's scores
    are numbers. A payload with a valid CRC is not trusted further than that:
    anyone can compute a CRC, and these records reach the replay code that
    otherwise reads only this server's own log."""
    var off = 0
    var family = -1
    var last_ms = UInt64(0)
    var last_seq = UInt64(0)
    var first_entry = True
    while off < n:
        if off + 13 > n:
            return False
        var elen = _u32(p, off)
        var cmd = Int(p[off + 4])
        var kl = _u32(p, off + 5)
        if off + 9 + kl + 4 > n:
            return False
        var vl = _u32(p, off + 9 + kl)
        if elen != 13 + kl + vl or off + 13 + kl + vl > n:
            return False
        var f = _record_family(cmd)
        if f == 0 or (family >= 0 and f != family):
            return False
        family = f
        var v = p + (off + 13 + kl)
        if cmd == 5:                                  # [4B fl][field][4B vl][value]
            if vl < 8:
                return False
            var fl = _u32(v, 0)
            if 8 + fl > vl or 8 + fl + _u32(v, 4 + fl) != vl:
                return False
        elif cmd == 32:                               # [4B fl][field][4B 8][8B deadline]
            if vl < 16:
                return False
            var fl = _u32(v, 0)
            if 16 + fl != vl or _u32(v, 4 + fl) != 8:
                return False
        elif cmd == 9 or cmd == 15:                   # [8B score][member]
            if vl < 8:
                return False
            var score = v.unsafe_bitcast[Float64]().load[volatile=True]()
            if score != score:                        # NaN
                return False
        elif cmd == 16:
            if vl <= 0:
                return False
        elif cmd == 17:
            if vl != 16384:
                return False
        elif cmd == 34:                               # [8B ms][8B seq][pairs]
            if vl < 16:
                return False
            var ms = _u64(v, 0)
            var seq = _u64(v, 8)
            if not first_entry and (ms < last_ms or (ms == last_ms and seq <= last_seq)):
                return False
            first_entry = False
            last_ms = ms
            last_seq = seq
        elif cmd == 45:                               # stream metadata (#40)
            if vl != 40:
                return False
            var lm = _u64(v, 0)
            var ls = _u64(v, 8)
            if not first_entry and (lm < last_ms or (lm == last_ms and ls < last_seq)):
                return False                          # a last id below its own entries
        elif cmd == 38 or cmd == 41 or cmd == 43:     # group, consumer, pending entry (#40)
            var gl = _u32(v, 0) if vl >= 4 else -1
            if gl < 0 or 4 + gl > vl:
                return False
            if cmd == 38:
                if vl != 4 + gl + 24:
                    return False
            else:
                if 8 + gl > vl:
                    return False
                var cl = _u32(v, 4 + gl)
                if vl != 8 + gl + cl + (16 if cmd == 41 else 32):
                    return False
        off += 13 + kl + vl
    return family >= 0


def restore_name(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int) -> String:
    return tokens[i].text_value().lower()


def _atoi(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    """C's atoi, as Redis reads MIGRATE's port: leading spaces, a sign, then
    digits up to the first other byte; 0 when there are none."""
    var k = 0
    while k < n and (p[k] == 32 or (p[k] >= 9 and p[k] <= 13)):
        k += 1
    var neg = False
    if k < n and (p[k] == 43 or p[k] == 45):
        neg = p[k] == 45
        k += 1
    var v = 0
    while k < n and p[k] >= 48 and p[k] <= 57 and v < 1_000_000_000:
        v = v * 10 + Int(p[k] - 48)
        k += 1
    return -v if neg else v


def handle_restore(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                   mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                   ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                   wal: Pointer[WAL, MutUntrackedOrigin]) -> Bool:
    """RESTORE key ttl payload [REPLACE] [ABSTTL] [IDLETIME s] [FREQ f], as
    Redis's restoreCommand: the options, then BUSYKEY, then the TTL, then the
    payload; an expired TTL restores nothing. IDLETIME and FREQ are checked and
    have nothing to set (Pion keeps no LRU or LFU data). True when the key
    changed (the caller bumps WATCH)."""
    var argc = num_tokens - i
    if argc < 4:
        writer.append_error_response("ERR wrong number of arguments for '" + restore_name(tokens, i) + "' command")
        return False
    var replace = False
    var absttl = False
    var idle = Int64(-1)
    var freq = Int64(-1)
    var j = i + 4
    while j < num_tokens:
        var o = tokens[j]
        var more = num_tokens - j - 1
        if arg_eq(o.ptr, o.length, "replace"):
            replace = True
        elif arg_eq(o.ptr, o.length, "absttl"):
            absttl = True
        elif arg_eq(o.ptr, o.length, "idletime") and more >= 1 and freq == -1:
            var r = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not r.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return False
            if r.value < 0:
                writer.append_error_response("ERR Invalid IDLETIME value, must be >= 0")
                return False
            idle = r.value
            j += 1
        elif arg_eq(o.ptr, o.length, "freq") and more >= 1 and idle == -1:
            var r = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not r.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return False
            if r.value < 0 or r.value > 255:
                writer.append_error_response("ERR Invalid FREQ value, must be >= 0 and <= 255")
                return False
            freq = r.value
            j += 1
        else:
            writer.append_error_response("ERR syntax error")
            return False
        j += 1
    var kt = tokens[i + 1]
    var key = GenericValue.borrow(kt.ptr, kt.length)
    var exists = not keyspace[].get(key).is_none()
    if exists and not replace:
        writer.append_error_response("BUSYKEY Target key name already exists.")
        return False
    var ttl = parse_int64_strict(tokens[i + 2].ptr, tokens[i + 2].length)
    if not ttl.ok:
        writer.append_error_response("ERR value is not an integer or out of range")
        return False
    if ttl.value < 0:
        writer.append_error_response("ERR Invalid TTL value, must be >= 0")
        return False
    var pt = tokens[i + 3]
    if not _payload_ok(pt.ptr, pt.length):
        writer.append_error_response("ERR DUMP payload version or checksum are wrong")
        return False
    var body = pt.length - 10
    if not _records_ok(pt.ptr, body):
        writer.append_error_response("ERR Bad data format")
        return False
    if exists:
        _ = remove_and_free(keyspace, key)
        if is_not_null(ttl_map):
            _ = ttl_map[].remove_generic(key)
        if is_not_null(wal):
            _ = wal[].append(2, kt.ptr, kt.length)
    var deadline_ns = Int64(0)
    if ttl.value > 0:
        var when_ms = ttl.value if absttl else Int64(_get_now_ns() // 1_000_000) + ttl.value
        if when_ms <= Int64(_get_now_ns() // 1_000_000):
            writer.append_ok_response()          # already expired: nothing to create
            return exists
        deadline_ns = ms_to_deadline_ns(when_ms)
    var off = 0
    while off < body:
        var cmd = UInt8(pt.ptr[off + 4])
        var rkl = _u32(pt.ptr, off + 5)
        var vl = _u32(pt.ptr, off + 9 + rkl)
        var vp = pt.ptr + (off + 13 + rkl)
        if cmd == 1:
            keyspace[].set(key, GenericValue.borrow(vp, vl))
        else:
            _ = wal_apply_aggregate(cmd, kt.ptr, kt.length, vp, vl, keyspace)
        if is_not_null(wal):
            _ = wal[].append_kv(cmd, kt.ptr, kt.length, vp, vl)
        off += 13 + rkl + vl
    if deadline_ns > 0 and is_not_null(ttl_map):
        ttl_map[].set(key, GenericValue.from_int(deadline_ns))
        if is_not_null(wal):
            _ = wal[].append_u64_val(25, kt.ptr, kt.length, UInt64(deadline_ns),
                                     null_ptr[UInt8, MutUntrackedOrigin](), 0)
    writer.append_ok_response()
    return True


def _put_bulk(mut out: List[UInt8], p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    var tmp = alloc[UInt8](24)
    out.append(36)
    var e = format_int_to_buf(tmp, 0, Int64(n))
    for k in range(e):
        out.append(tmp[k])
    out.append(13)
    out.append(10)
    for k in range(n):
        out.append(p[k])
    out.append(13)
    out.append(10)
    tmp.unsafe_free()


def _put_str(mut out: List[UInt8], s: String):
    _put_bulk(out, Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(s.unsafe_ptr())), s.byte_length())


def _put_header(mut out: List[UInt8], n: Int):
    var tmp = alloc[UInt8](24)
    out.append(42)
    var e = format_int_to_buf(tmp, 0, Int64(n))
    for k in range(e):
        out.append(tmp[k])
    out.append(13)
    out.append(10)
    tmp.unsafe_free()


def handle_migrate(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                   mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                   ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                   wal: Pointer[WAL, MutUntrackedOrigin], cluster_mode: Bool) -> List[Int]:
    """MIGRATE host port key|"" destination-db timeout [COPY] [REPLACE]
    [AUTH password | AUTH2 username password] [KEYS key ...], as Redis's
    migrateCommand: each key's DUMP payload goes to the target as RESTORE (or
    RESTORE-ASKING in cluster mode) with its remaining TTL, after AUTH and
    SELECT when given, in one pipeline; then, unless COPY, the moved keys are
    deleted here and the deletions logged. Blocking, with the timeout bounding
    each wait, as in Redis. Returns the token indexes of the keys it deleted
    (the caller bumps WATCH)."""
    var deleted = List[Int]()
    var argc = num_tokens - i
    if argc < 6:
        writer.append_error_response("ERR wrong number of arguments for 'migrate' command")
        return deleted^
    var copy_flag = False
    var replace_flag = False
    var auth_user = -1
    var auth_pass = -1
    var first_key = i + 3
    var num_keys = 1
    var j = i + 6
    while j < num_tokens:
        var o = tokens[j]
        var more = num_tokens - j - 1
        if arg_eq(o.ptr, o.length, "copy"):
            copy_flag = True
        elif arg_eq(o.ptr, o.length, "replace"):
            replace_flag = True
        elif arg_eq(o.ptr, o.length, "auth"):
            if more < 1:
                writer.append_error_response("ERR syntax error")
                return deleted^
            auth_pass = j + 1
            j += 1
        elif arg_eq(o.ptr, o.length, "auth2"):
            if more < 2:
                writer.append_error_response("ERR syntax error")
                return deleted^
            auth_user = j + 1
            auth_pass = j + 2
            j += 2
        elif arg_eq(o.ptr, o.length, "keys"):
            if tokens[i + 3].length != 0:
                writer.append_error_response("ERR When using MIGRATE KEYS option, the key argument must be set to the empty string")
                return deleted^
            first_key = j + 1
            num_keys = num_tokens - j - 1
            j = num_tokens
            break
        else:
            writer.append_error_response("ERR syntax error")
            return deleted^
        j += 1
    # Redis checks the timeout and the db; the port is read with atoi, so a
    # bad one fails to connect
    var timeout = parse_int64_strict(tokens[i + 5].ptr, tokens[i + 5].length)
    var db = parse_int64_strict(tokens[i + 4].ptr, tokens[i + 4].length)
    if not timeout.ok or not db.ok:
        writer.append_error_response("ERR value is not an integer or out of range")
        return deleted^
    var port = _atoi(tokens[i + 2].ptr, tokens[i + 2].length)
    var timeout_ms = Int(timeout.value) if timeout.value > 0 and timeout.value < 1 << 31 else 1000
    # the keys that exist, with their payloads and remaining TTLs
    var present = List[Int]()
    var payloads = List[List[UInt8]]()
    var ttls = List[Int64]()
    var now_ms = Int64(_get_now_ns() // 1_000_000)
    for k in range(first_key, first_key + num_keys):
        var t = tokens[k]
        var p = dump_payload(keyspace, t.ptr, t.length)
        if len(p) == 0:
            continue
        var ttl = Int64(0)
        if is_not_null(ttl_map):
            var d = ttl_map[].get(GenericValue.borrow(t.ptr, t.length))
            if not d.is_none():
                ttl = d.as_int() // 1_000_000 - now_ms
                if ttl < 0:
                    continue                 # expired: not sent, as in Redis
                if ttl < 1:
                    ttl = 1
        present.append(k)
        payloads.append(p^)
        ttls.append(ttl)
    if len(present) == 0:
        writer.append_status_response("NOKEY")
        return deleted^
    var host = tokens[i + 1].value() + "\0"
    var fd = external_call["pion_tcp_connect_host", Int32](host.unsafe_ptr(), Int32(port), Int32(timeout_ms))
    _ = host^                                  # alive through the call (ASAP destruction)
    if fd < 0:
        writer.append_error_response("IOERR error or timeout connecting to the client")
        return deleted^
    # one pipeline: [AUTH], [SELECT], then a RESTORE per key
    var out = List[UInt8]()
    var expected = 0
    if auth_pass >= 0:
        _put_header(out, 3 if auth_user >= 0 else 2)
        _put_str(out, "AUTH")
        if auth_user >= 0:
            _put_bulk(out, tokens[auth_user].ptr, tokens[auth_user].length)
        _put_bulk(out, tokens[auth_pass].ptr, tokens[auth_pass].length)
        expected += 1
    if db.value != 0:
        _put_header(out, 2)
        _put_str(out, "SELECT")
        _put_str(out, String(db.value))
        expected += 1
    var first_restore = expected
    for k in range(len(present)):
        var t = tokens[present[k]]
        _put_header(out, 5 if replace_flag else 4)
        _put_str(out, "RESTORE-ASKING" if cluster_mode else "RESTORE")
        _put_bulk(out, t.ptr, t.length)
        _put_str(out, String(ttls[k]))
        _put_bulk(out, Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(payloads[k].unsafe_ptr())),
                  len(payloads[k]))
        if replace_flag:
            _put_str(out, "REPLACE")
        expected += 1
    var wrote = external_call["pion_sync_write", Int32](fd, out.unsafe_ptr(), Int64(len(out)),
                                                        Int32(timeout_ms)) == 0
    _ = out^                                   # alive through the call (ASAP destruction)
    var ok = wrote
    var error = String("")
    var line = alloc[UInt8](1024)
    var moved = List[Bool]()
    var setup_failed = False       # AUTH or SELECT refused: Redis deletes no key
    for r in range(expected):
        if not ok:
            break
        var n = Int(external_call["pion_sync_readline", Int64](fd, line, Int64(1024), Int32(timeout_ms)))
        if n < 0:
            ok = False
            break
        var is_err = n > 0 and line[0] == 45      # '-'
        if is_err and error.byte_length() == 0 and n > 1:
            error = String(StringSpan[MutUntrackedOrigin](
                unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=line + 1, length=n - 1)))
        if r < first_restore:
            if is_err:
                setup_failed = True
        else:
            moved.append(not is_err and not setup_failed)
    line.unsafe_free()
    _ = external_call["close", Int32](fd)
    if not ok:
        writer.append_error_response("IOERR error or timeout " + String("reading" if wrote else "writing")
                                     + " to target instance")
        return deleted^
    if not copy_flag:
        for k in range(len(moved)):
            if moved[k]:
                var t = tokens[present[k]]
                var kv = GenericValue.borrow(t.ptr, t.length)
                _ = remove_and_free(keyspace, kv)
                if is_not_null(ttl_map):
                    _ = ttl_map[].remove_generic(kv)
                if is_not_null(wal):
                    _ = wal[].append(2, t.ptr, t.length)
                deleted.append(present[k])
    if error.byte_length() > 0:
        writer.append_error_response("ERR Target instance replied with error: " + error)
    else:
        writer.append_ok_response()
    return deleted^
