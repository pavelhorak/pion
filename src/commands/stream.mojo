"""Stream commands: XADD, XLEN, XRANGE, XREVRANGE, XREAD, XDEL, XTRIM, XINFO, XACK, XGROUP, XCLAIM, XPENDING, XAUTOCLAIM.

Implements a persistent append-only log per key (ValueType.STREAM = 13).
Consumer groups remain stubs.
"""
from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, stack_allocation
from std.collections import Array
from std.memory import unsafe_memcpy, unsafe_memset
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.utils import strict_atol, format_int_to_buf
from src.network.fast_path import _get_now_ns
from src.network.server import TCPServer
# gh #174: StreamEntry/StreamData live in src/common so src/io/wal.mojo can
# replay XADD records without the wal -> stream -> fast_path -> wal cycle.
# Re-exported here so existing importers of this module are unaffected.
from src.common.stream_data import StreamEntry, StreamData
from src.io.wal import WAL

comptime MAX_BLOCKED_READERS = 64
comptime MAX_BLOCKED_KEYS = 4  # max stream keys per blocked XREAD


# ── Blocked XREAD state ──

struct BlockedReader(Copyable, Movable):
    """A pending XREAD BLOCK request waiting for new data."""
    var fd: Int32
    var active: Bool
    var count_limit: Int
    var timeout_ms: Int64       # 0 = infinite, >0 = deadline (absolute ms from _get_now_ns)
    var key_hashes: Array[UInt64, 4]  # hashed key names for fast matching
    var after_ms: Array[UInt64, 4]    # last ID ms per key
    var after_seq: Array[UInt64, 4]   # last ID seq per key
    var num_keys: Int
    # Store key names for response formatting
    var key_ptrs: Array[Pointer[UInt8, MutUntrackedOrigin], 4]
    var key_lens: Array[Int, 4]

    def __init__(out self):
        self.fd = Int32(-1); self.active = False
        self.count_limit = 100; self.timeout_ms = 0; self.num_keys = 0
        self.key_hashes = Array[UInt64, 4](fill=UInt64(0))
        self.after_ms = Array[UInt64, 4](fill=UInt64(0))
        self.after_seq = Array[UInt64, 4](fill=UInt64(0))
        self.key_ptrs = Array[Pointer[UInt8, MutUntrackedOrigin], 4](fill=null_ptr[UInt8, MutUntrackedOrigin]())
        self.key_lens = Array[Int, 4](fill=0)


struct BlockedReaderRegistry(Movable):
    """Per-worker registry of blocked XREAD requests.
    Uses heap-allocated count to ensure mutations persist across mut parameter passing."""
    var readers: Pointer[BlockedReader, MutUntrackedOrigin]
    var count_ptr: Pointer[Int, MutUntrackedOrigin]  # heap-allocated to survive mut borrow

    @always_inline
    def _count(self) -> Int:
        return self.count_ptr[unsafe_offset=0]

    def __init__(out self):
        self.readers = alloc[BlockedReader](MAX_BLOCKED_READERS)
        for i in range(MAX_BLOCKED_READERS):
            (self.readers.unsafe_offset(i)).unsafe_write(BlockedReader())
        self.count_ptr = alloc[Int](1)
        self.count_ptr[unsafe_offset=0] = 0

    @always_inline
    @staticmethod
    def hash_key(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> UInt64:
        var h: UInt64 = 0x736f6d6570736575
        for i in range(length):
            h = (h ^ UInt64(ptr[unsafe_offset=i])) * UInt64(0x100000001b3)
        return h

    def add(mut self, reader: BlockedReader) -> Bool:
        """Register a blocked reader. Returns False if full."""
        for i in range(MAX_BLOCKED_READERS):
            if not self.readers[unsafe_offset=i].active:
                self.readers[unsafe_offset=i] = reader.copy()
                self.count_ptr[unsafe_offset=0] += 1
                return True
        return False

    def remove_fd(mut self, fd: Int32):
        """Remove all blocked readers for a given fd."""
        for i in range(MAX_BLOCKED_READERS):
            if self.readers[unsafe_offset=i].active and self.readers[unsafe_offset=i].fd == fd:
                # Free stored key name copies
                for k in range(self.readers[unsafe_offset=i].num_keys):
                    if is_not_null(self.readers[unsafe_offset=i].key_ptrs[k]):
                        self.readers[unsafe_offset=i].key_ptrs[k].unsafe_free()
                self.readers[unsafe_offset=i].active = False
                self.count_ptr[unsafe_offset=0] -= 1

    def check_timeouts(mut self, now_ms: Int64, server: TCPServer):
        """Check for timed-out blocked readers and send null response."""
        if self.count_ptr[unsafe_offset=0] == 0: return
        for i in range(MAX_BLOCKED_READERS):
            if self.readers[unsafe_offset=i].active and self.readers[unsafe_offset=i].timeout_ms > 0 and now_ms >= self.readers[unsafe_offset=i].timeout_ms:
                # Timed out — send null response ($-1\r\n = 5 bytes)
                var null_buf = alloc[UInt8](5)
                null_buf[unsafe_offset=0] = 36  # '$'
                null_buf[unsafe_offset=1] = 45  # '-'
                null_buf[unsafe_offset=2] = 49  # '1'
                null_buf[unsafe_offset=3] = 13  # '\r'
                null_buf[unsafe_offset=4] = 10  # '\n'
                _ = server.send(self.readers[unsafe_offset=i].fd, null_buf, 5)
                null_buf.unsafe_free()
                for k in range(self.readers[unsafe_offset=i].num_keys):
                    if is_not_null(self.readers[unsafe_offset=i].key_ptrs[k]):
                        self.readers[unsafe_offset=i].key_ptrs[k].unsafe_free()
                self.readers[unsafe_offset=i].active = False
                self.count_ptr[unsafe_offset=0] -= 1


# ── ID helpers ──

struct StreamID(Copyable, Movable, ImplicitlyCopyable):
    var ms: UInt64
    var seq: UInt64
    var explicit: Bool  # True = explicit ID or min/max; False = auto-generate (*)

    def __init__(out self):
        self.ms = 0; self.seq = 0; self.explicit = False

    def __init__(out self, ms: UInt64, seq: UInt64, explicit: Bool):
        self.ms = ms; self.seq = seq; self.explicit = explicit


@always_inline
def parse_stream_id(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> StreamID:
    """Parse 'ms-seq', '*', '-', '+'."""
    if length == 1:
        if ptr[unsafe_offset=0] == 42: return StreamID(0, 0, False)  # '*' → auto
        if ptr[unsafe_offset=0] == 45: return StreamID(0, 0, True)   # '-' → min
        if ptr[unsafe_offset=0] == 43: return StreamID(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, True)  # '+' → max
    var ms: UInt64 = 0
    var seq: UInt64 = 0
    var i = 0
    while i < length and ptr[unsafe_offset=i] != 45:  # '-'
        ms = ms * 10 + UInt64(ptr[unsafe_offset=i] - 48)
        i += 1
    if i < length and ptr[unsafe_offset=i] == 45:
        i += 1
        while i < length:
            seq = seq * 10 + UInt64(ptr[unsafe_offset=i] - 48)
            i += 1
    return StreamID(ms, seq, True)


@always_inline
def id_ge(a_ms: UInt64, a_seq: UInt64, b_ms: UInt64, b_seq: UInt64) -> Bool:
    return a_ms > b_ms or (a_ms == b_ms and a_seq >= b_seq)


@always_inline
def id_le(a_ms: UInt64, a_seq: UInt64, b_ms: UInt64, b_seq: UInt64) -> Bool:
    return a_ms < b_ms or (a_ms == b_ms and a_seq <= b_seq)


@always_inline
def id_gt(a_ms: UInt64, a_seq: UInt64, b_ms: UInt64, b_seq: UInt64) -> Bool:
    return a_ms > b_ms or (a_ms == b_ms and a_seq > b_seq)


@always_inline
def format_stream_id(buf: Pointer[UInt8, MutUntrackedOrigin], ms: UInt64, seq: UInt64) -> Int:
    """Write 'ms-seq' to buf, return bytes written."""
    var off = format_int_to_buf(buf, 0, Int64(ms))
    buf[unsafe_offset=off] = 45  # '-'
    off += 1
    off += format_int_to_buf(buf.unsafe_offset(off), 0, Int64(seq))
    return off


# Outcomes of a stream-slot resolve (gh #232). A null return needs a reason:
# WRONGTYPE is an error reply, NOMKSTREAM is a nil reply, and neither may
# touch the keyspace.
comptime STREAM_OK = 0
comptime STREAM_WRONGTYPE = 1
comptime STREAM_NOMKSTREAM = 2


@always_inline
def get_or_create_stream(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], key_val: GenericValue,
                         no_mkstream: Bool, mut outcome: Int) -> Pointer[StreamData, MutUntrackedOrigin]:
    """Get an existing stream, or create one unless the key is taken or NOMKSTREAM.

    gh #232: this used to `set()` a fresh stream whenever the key held anything
    that was not one, so `SET k hello; XADD k * f v` REPLACED the string and
    answered with an id — the value was gone (`GET k` then said WRONGTYPE) and
    its heap payload was leaked rather than freed. Same for a list, hash, set,
    zset or HLL. Redis answers WRONGTYPE and leaves the key untouched.

    XADD was the ONLY create-if-missing command with this hole; RPUSH, LPUSH,
    SADD, HSET, ZADD, PFADD, GEOADD, SETBIT, APPEND and SETRANGE were all
    probed against every other type and already refuse.

    The outcome rides out in `outcome` rather than costing a second `get()` —
    XADD is a gate row, so the resolve stays one hash lookup.
    """
    var val = keyspace[].get(key_val)
    if not val.is_none():
        if val.type.value == ValueType.STREAM:
            outcome = STREAM_OK
            return val.as_hash().unsafe_bitcast[StreamData]()
        outcome = STREAM_WRONGTYPE
        return null_ptr[StreamData, MutUntrackedOrigin]()
    if no_mkstream:
        outcome = STREAM_NOMKSTREAM
        return null_ptr[StreamData, MutUntrackedOrigin]()
    # Create new stream
    var sd_ptr = alloc[StreamData](1)
    sd_ptr.unsafe_write(StreamData())
    var new_val = GenericValue()
    new_val.type = ValueType(ValueType.STREAM)
    new_val.set_ptr(sd_ptr.unsafe_bitcast[NoneType]())
    keyspace[].set(key_val, new_val)
    outcome = STREAM_OK
    return sd_ptr


@always_inline
def get_stream(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], key_val: GenericValue) -> Pointer[StreamData, MutUntrackedOrigin]:
    """Get existing stream or return null pointer."""
    var val = keyspace[].get(key_val)
    if not val.is_none() and val.type.value == ValueType.STREAM:
        return val.as_hash().unsafe_bitcast[StreamData]()
    return null_ptr[StreamData, MutUntrackedOrigin]()


@always_inline
def stream_key_is_wrongtype(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], key_val: GenericValue) -> Bool:
    """True when the key EXISTS and holds something other than a stream.

    gh #232: `get_stream` collapses "missing" and "wrong type" into one null,
    so XLEN answered 0 and XRANGE answered [] for a key holding a list. Redis
    says WRONGTYPE. The conflation runs in the dangerous direction — a caller
    who stored the wrong kind of value is told the stream is empty, so the bug
    surfaces as missing data rather than as a type error at the call site.

    Kept as a second lookup off the null path rather than widening
    `get_stream`, whose 10 call sites are on the served-stream path where the
    pointer is non-null and this is never reached.
    """
    var val = keyspace[].get(key_val)
    return not val.is_none() and val.type.value != ValueType.STREAM


# ── Write stream entry fields to response ──

@always_inline
def write_entry_to_response(e: StreamEntry, mut writer: ResponseWriter):
    """Write a stream entry as: *2\\r\\n $<id_len>\\r\\n<id>\\r\\n *<2*nf>\\r\\n [field value]..."""
    # Array of 2 elements: [id, fields_array]
    writer.append_to_response("*2\r\n".unsafe_ptr(), 4)
    # ID
    var id_buf = alloc[UInt8](40)
    var id_len = format_stream_id(id_buf, e.id_ms, e.id_seq)
    writer.append_bulk_string_response(id_buf, id_len)
    id_buf.unsafe_free()
    # Fields array
    var nf2 = e.num_fields * 2
    writer.buffer[unsafe_offset=writer.offset] = 42  # '*'
    writer.offset += 1
    writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(nf2))
    writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
    writer.offset += 2
    # Parse packed data: [u16 flen][bytes][u16 vlen][bytes]...
    var doff = 0
    for _ in range(e.num_fields):
        if doff + 2 > e.data_len: break
        var flen = Int((e.data.unsafe_offset(doff)).unsafe_bitcast[UInt16]()[])
        doff += 2
        if doff + flen > e.data_len: break
        writer.append_bulk_string_response(e.data.unsafe_offset(doff), flen)
        doff += flen
        if doff + 2 > e.data_len: break
        var vlen = Int((e.data.unsafe_offset(doff)).unsafe_bitcast[UInt16]()[])
        doff += 2
        if doff + vlen > e.data_len: break
        writer.append_bulk_string_response(e.data.unsafe_offset(doff), vlen)
        doff += vlen


def write_entry_to_buf(e: StreamEntry, buf: Pointer[UInt8, MutUntrackedOrigin], start: Int) -> Int:
    """Write a stream entry to a raw buffer. Returns new offset."""
    var off = start
    # *2\r\n
    buf[unsafe_offset=off] = 42; buf[unsafe_offset=off + 1] = 50; buf[unsafe_offset=off + 2] = 13; buf[unsafe_offset=off + 3] = 10; off += 4
    # ID as bulk string
    var id_buf = alloc[UInt8](40)
    var id_len = format_stream_id(id_buf, e.id_ms, e.id_seq)
    buf[unsafe_offset=off] = 36; off += 1
    off += format_int_to_buf(buf.unsafe_offset(off), 0, Int64(id_len))
    buf[unsafe_offset=off] = 13; buf[unsafe_offset=off + 1] = 10; off += 2
    unsafe_memcpy(dest=buf.unsafe_offset(off), src=id_buf, count=id_len); off += id_len
    buf[unsafe_offset=off] = 13; buf[unsafe_offset=off + 1] = 10; off += 2
    id_buf.unsafe_free()
    # Fields array
    var nf2 = e.num_fields * 2
    buf[unsafe_offset=off] = 42; off += 1
    off += format_int_to_buf(buf.unsafe_offset(off), 0, Int64(nf2))
    buf[unsafe_offset=off] = 13; buf[unsafe_offset=off + 1] = 10; off += 2
    var doff = 0
    for _ in range(e.num_fields):
        if doff + 2 > e.data_len: break
        var flen = Int((e.data.unsafe_offset(doff)).unsafe_bitcast[UInt16]()[])
        doff += 2
        if doff + flen > e.data_len: break
        buf[unsafe_offset=off] = 36; off += 1
        off += format_int_to_buf(buf.unsafe_offset(off), 0, Int64(flen))
        buf[unsafe_offset=off] = 13; buf[unsafe_offset=off + 1] = 10; off += 2
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=e.data.unsafe_offset(doff), count=flen); off += flen
        buf[unsafe_offset=off] = 13; buf[unsafe_offset=off + 1] = 10; off += 2
        doff += flen
        if doff + 2 > e.data_len: break
        var vlen = Int((e.data.unsafe_offset(doff)).unsafe_bitcast[UInt16]()[])
        doff += 2
        if doff + vlen > e.data_len: break
        buf[unsafe_offset=off] = 36; off += 1
        off += format_int_to_buf(buf.unsafe_offset(off), 0, Int64(vlen))
        buf[unsafe_offset=off] = 13; buf[unsafe_offset=off + 1] = 10; off += 2
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=e.data.unsafe_offset(doff), count=vlen); off += vlen
        buf[unsafe_offset=off] = 13; buf[unsafe_offset=off + 1] = 10; off += 2
        doff += vlen
    return off


# ── Command Handlers ──

@always_inline
def handle_xadd(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """XADD key [NOMKSTREAM] [MAXLEN|MINID [=|~] threshold] <id|*> field value [field value ...]"""
    if i + 3 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'xadd' command")
        return 0
    # gh #202: `tokens[i+1].value()` + `GenericValue.from_string(...)` built a
    # heap String only to re-pack it into a GenericValue. `from_ptr` produces
    # the identical value (STRING_SSO at <=23B, heap STRING above) straight
    # from the recv buffer, with no String in between. The WAL sites below take
    # the same bytes from the token rather than from the dead String.
    #
    # INVARIANT — every stream handler keys through `from_ptr` on the token,
    # never through `value()`. `value()` spells bytes >= 128 as '?', so keying
    # XADD off raw bytes while XLEN/XRANGE/XDEL/XREAD/XTRIM/XINFO still keyed
    # off the mangled spelling would make a binary stream key writable but not
    # readable. They were consistently mangled before and are consistently
    # exact now; do not convert one site back.
    var key_ptr = tokens[unsafe_offset=i + 1].ptr
    var key_len = tokens[unsafe_offset=i + 1].length
    var key_val = GenericValue.borrow(key_ptr, key_len)

    # Parse optional flags and find the ID token
    var j = i + 2
    var maxlen = -1
    var no_mkstream = False
    while j < num_tokens:
        var tp = tokens[unsafe_offset=j].ptr; var tl = tokens[unsafe_offset=j].length
        if tl == 6 and (tp[unsafe_offset=0] | 0x20) == 109 and (tp[unsafe_offset=1] | 0x20) == 97 and (tp[unsafe_offset=2] | 0x20) == 120:
            # MAXLEN [~] N
            j += 1
            if j < num_tokens and tokens[unsafe_offset=j].length == 1 and tokens[unsafe_offset=j].ptr[unsafe_offset=0] == 126: j += 1  # skip ~
            if j < num_tokens:
                var ml = strict_atol(tokens[unsafe_offset=j].value())
                if ml > 0: maxlen = ml
            j += 1; continue
        elif tl == 10 and (tp[unsafe_offset=0] | 0x20) == 110:
            # NOMKSTREAM — gh #232: this was `tl == 11` and NOMKSTREAM is 10
            # bytes, so the arm never fired. The flag then fell out of the loop
            # as the ID token: `parse_stream_id` read the literal text into an
            # id (33420896299-0), field/value pairs shifted by one, and the
            # last value was dropped outright. It was also unimplemented — the
            # stream got created either way, where Redis replies nil.
            no_mkstream = True
            j += 1; continue
        elif tl == 5 and (tp[unsafe_offset=0] | 0x20) == 109 and (tp[unsafe_offset=1] | 0x20) == 105:
            # MINID [~] N — skip for now
            j += 1
            if j < num_tokens and tokens[unsafe_offset=j].length == 1 and tokens[unsafe_offset=j].ptr[unsafe_offset=0] == 126: j += 1
            if j < num_tokens: j += 1
            continue
        else:
            break  # This is the ID token

    if j >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'xadd' command")
        return num_tokens - i - 1

    # Parse ID
    var id_tok = tokens[unsafe_offset=j]
    var r = parse_stream_id(id_tok.ptr, id_tok.length)
    var id_ms = r.ms; var id_seq = r.seq; var id_explicit = r.explicit
    j += 1

    # Remaining tokens are field-value pairs
    var field_start = j
    var nf = (num_tokens - field_start) // 2
    if nf < 1:
        writer.append_error_response("ERR wrong number of arguments for 'xadd' command")
        return num_tokens - i - 1

    # Pack field-value data: [u16 flen][bytes][u16 vlen][bytes]...
    var pack_size = 0
    for fi in range(nf):
        pack_size += 2 + tokens[unsafe_offset=field_start + fi * 2].length + 2 + tokens[unsafe_offset=field_start + fi * 2 + 1].length
    var pack_buf = alloc[UInt8](pack_size)
    var poff = 0
    for fi in range(nf):
        var ft = tokens[unsafe_offset=field_start + fi * 2]
        var vt = tokens[unsafe_offset=field_start + fi * 2 + 1]
        (pack_buf.unsafe_offset(poff)).unsafe_bitcast[UInt16]()[] = UInt16(ft.length); poff += 2
        unsafe_memcpy(dest=pack_buf.unsafe_offset(poff), src=ft.ptr, count=ft.length); poff += ft.length
        (pack_buf.unsafe_offset(poff)).unsafe_bitcast[UInt16]()[] = UInt16(vt.length); poff += 2
        unsafe_memcpy(dest=pack_buf.unsafe_offset(poff), src=vt.ptr, count=vt.length); poff += vt.length

    # Get or create stream. Both refusals must free the pack buffer allocated
    # above and leave the keyspace untouched.
    var outcome = STREAM_OK
    var sd = get_or_create_stream(keyspace, key_val, no_mkstream, outcome)
    if outcome == STREAM_WRONGTYPE:
        pack_buf.unsafe_free()
        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        return num_tokens - i - 1
    if outcome == STREAM_NOMKSTREAM:
        pack_buf.unsafe_free()
        writer.append_null_response()
        return num_tokens - i - 1

    # Generate ID if auto
    if not id_explicit:
        id_ms = UInt64(_get_now_ns() // 1000000)
        if id_ms == sd[].last_id_ms:
            id_seq = sd[].last_id_seq + 1
        else:
            id_seq = 0
            if id_ms < sd[].last_id_ms:
                id_ms = sd[].last_id_ms
                id_seq = sd[].last_id_seq + 1

    # gh #242: an EXPLICIT id was appended with no ordering check at all, so
    # `XADD s 5-5` twice, or a smaller id after a larger one, both succeeded —
    # leaving ids like [5-5, 5-5, 3-3, 5-6]. Stream ids being strictly
    # increasing is the invariant every consumer cursor relies on: a client
    # resuming from `(last-id` either loops on the duplicate or skips the
    # out-of-order entry. Only the AUTO path (above) enforced it.
    if id_explicit:
        if id_ms == 0 and id_seq == 0:
            pack_buf.unsafe_free()
            writer.append_error_response("ERR The ID specified in XADD must be greater than 0-0")
            return num_tokens - i - 1
        if sd[].count > 0 and (id_ms < sd[].last_id_ms
                               or (id_ms == sd[].last_id_ms and id_seq <= sd[].last_id_seq)):
            pack_buf.unsafe_free()
            writer.append_error_response("ERR The ID specified in XADD is equal or smaller than the target stream top item")
            return num_tokens - i - 1

    # Append entry
    sd[].append(id_ms, id_seq, pack_buf, pack_size, nf)

    # gh #174: effect-log the entry with its *resolved* ID. `XADD key *` must
    # never replay as "generate an ID now" — recovery runs at a different wall
    # clock, which would renumber the stream and break every consumer cursor.
    if is_not_null(wal):
        _ = wal[].append_u64x2_val(23, key_ptr, key_len,
                                   id_ms, id_seq, pack_buf, pack_size)

    # Apply MAXLEN trimming
    if maxlen > 0 and sd[].alive > maxlen:
        var to_trim = sd[].alive - maxlen
        for ti in range(sd[].count):
            if to_trim <= 0: break
            if not sd[].entries[unsafe_offset=ti].deleted:
                sd[].kill(ti)
                to_trim -= 1
                # Trimming is a resolved effect too: log which entry died, so a
                # replay tombstones the same one instead of re-running a MAXLEN
                # against a differently-ordered rebuild.
                if is_not_null(wal):
                    _ = wal[].append_u64x2_val(
                        27, key_ptr, key_len,
                        sd[].entries[unsafe_offset=ti].id_ms, sd[].entries[unsafe_offset=ti].id_seq,
                        null_ptr[UInt8, MutUntrackedOrigin](), 0)
        sd[].compact()

    # Return ID. gh #202: a 40-byte scratch buffer per XADD does not need a
    # tcmalloc round trip — the reply is copied into the response buffer by
    # append_bulk_string_response before this frame goes away.
    var id_buf = stack_allocation[40, UInt8]()
    var id_len = format_stream_id(id_buf, id_ms, id_seq)
    writer.append_bulk_string_response(id_buf, id_len)
    return num_tokens - i - 1


@always_inline
def handle_xlen(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """XLEN key → :N."""
    if i + 1 < num_tokens:
        var key_val = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
        var sd = get_stream(keyspace, key_val)
        if is_not_null(sd):
            writer.append_int_response(Int64(sd[].alive))
        elif stream_key_is_wrongtype(keyspace, key_val):   # gh #232
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            writer.append_int_response(0)
        return 1
    writer.append_int_response(0)
    return 0


@always_inline
def handle_xrange(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """XRANGE key start end [COUNT count]."""
    if i + 3 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'xrange' command")
        return 0
    var key_val = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    var sd = get_stream(keyspace, key_val)
    if is_null(sd):
        if stream_key_is_wrongtype(keyspace, key_val):   # gh #232
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            writer.append_empty_array_response()
        return num_tokens - i - 1

    var r_start = parse_stream_id(tokens[unsafe_offset=i + 2].ptr, tokens[unsafe_offset=i + 2].length)
    var r_end = parse_stream_id(tokens[unsafe_offset=i + 3].ptr, tokens[unsafe_offset=i + 3].length)
    var start_ms = r_start.ms; var start_seq = r_start.seq
    var end_ms = r_end.ms; var end_seq = r_end.seq

    var count_limit = sd[].count  # default: no limit
    if i + 5 < num_tokens and tokens[unsafe_offset=i + 4].length == 5 and (tokens[unsafe_offset=i + 4].ptr[unsafe_offset=0] | 0x20) == 99:
        count_limit = strict_atol(tokens[unsafe_offset=i + 5].value())

    # Collect matching entries
    var result_count = 0
    # First pass: count
    for ei in range(sd[].count):
        if sd[].entries[unsafe_offset=ei].deleted: continue
        var e = sd[].entries[unsafe_offset=ei]
        if id_ge(e.id_ms, e.id_seq, start_ms, start_seq) and id_le(e.id_ms, e.id_seq, end_ms, end_seq):
            result_count += 1
            if result_count >= count_limit: break

    # Write array header
    writer.buffer[unsafe_offset=writer.offset] = 42  # '*'
    writer.offset += 1
    writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(result_count))
    writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
    writer.offset += 2

    # Second pass: write entries
    var written = 0
    for ei in range(sd[].count):
        if written >= result_count: break
        if sd[].entries[unsafe_offset=ei].deleted: continue
        var e = sd[].entries[unsafe_offset=ei]
        if id_ge(e.id_ms, e.id_seq, start_ms, start_seq) and id_le(e.id_ms, e.id_seq, end_ms, end_seq):
            write_entry_to_response(e, writer)
            written += 1

    return num_tokens - i - 1


@always_inline
def handle_xrevrange(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """XREVRANGE key end start [COUNT count]."""
    if i + 3 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'xrevrange' command")
        return 0
    var key_val = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    var sd = get_stream(keyspace, key_val)
    if is_null(sd):
        if stream_key_is_wrongtype(keyspace, key_val):   # gh #232
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            writer.append_empty_array_response()
        return num_tokens - i - 1

    var r_end = parse_stream_id(tokens[unsafe_offset=i + 2].ptr, tokens[unsafe_offset=i + 2].length)
    var r_start = parse_stream_id(tokens[unsafe_offset=i + 3].ptr, tokens[unsafe_offset=i + 3].length)
    var start_ms = r_start.ms; var start_seq = r_start.seq
    var end_ms = r_end.ms; var end_seq = r_end.seq

    var count_limit = sd[].count
    if i + 5 < num_tokens and tokens[unsafe_offset=i + 4].length == 5 and (tokens[unsafe_offset=i + 4].ptr[unsafe_offset=0] | 0x20) == 99:
        count_limit = strict_atol(tokens[unsafe_offset=i + 5].value())

    # Count matching entries (reverse)
    var result_count = 0
    var ei2 = sd[].count - 1
    while ei2 >= 0:
        if not sd[].entries[unsafe_offset=ei2].deleted:
            var e = sd[].entries[unsafe_offset=ei2]
            if id_ge(e.id_ms, e.id_seq, start_ms, start_seq) and id_le(e.id_ms, e.id_seq, end_ms, end_seq):
                result_count += 1
                if result_count >= count_limit: break
        ei2 -= 1

    writer.buffer[unsafe_offset=writer.offset] = 42
    writer.offset += 1
    writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(result_count))
    writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
    writer.offset += 2

    var written = 0
    ei2 = sd[].count - 1
    while ei2 >= 0:
        if written >= result_count: break
        if not sd[].entries[unsafe_offset=ei2].deleted:
            var e = sd[].entries[unsafe_offset=ei2]
            if id_ge(e.id_ms, e.id_seq, start_ms, start_seq) and id_le(e.id_ms, e.id_seq, end_ms, end_seq):
                write_entry_to_response(e, writer)
                written += 1
        ei2 -= 1

    return num_tokens - i - 1


def notify_blocked_readers(mut registry: BlockedReaderRegistry, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                           key_ptr: Pointer[UInt8, MutUntrackedOrigin], key_len: Int, server: TCPServer):
    """Called after XADD — wake up any blocked readers waiting on this key."""
    if registry.count_ptr[unsafe_offset=0] == 0: return
    var key_hash = BlockedReaderRegistry.hash_key(key_ptr, key_len)
    for ri in range(MAX_BLOCKED_READERS):
        if not registry.readers[unsafe_offset=ri].active: continue
        for ki in range(registry.readers[unsafe_offset=ri].num_keys):
            if registry.readers[unsafe_offset=ri].key_hashes[ki] != key_hash: continue
            # Hash match — verify key name
            if registry.readers[unsafe_offset=ri].key_lens[ki] != key_len: continue
            var name_eq = True
            for ci in range(key_len):
                if registry.readers[unsafe_offset=ri].key_ptrs[ki][unsafe_offset=ci] != key_ptr[unsafe_offset=ci]:
                    name_eq = False; break
            if not name_eq: continue
            # Match! Build and send XREAD response for this blocked reader
            var reader = registry.readers[unsafe_offset=ri].copy()
            var resp_buf = alloc[UInt8](65536)
            var off = 0
            # *1\r\n (one stream)
            resp_buf[unsafe_offset=off] = 42; off += 1; resp_buf[unsafe_offset=off] = 49; off += 1
            resp_buf[unsafe_offset=off] = 13; resp_buf[unsafe_offset=off + 1] = 10; off += 2
            # *2\r\n (key + entries)
            resp_buf[unsafe_offset=off] = 42; off += 1; resp_buf[unsafe_offset=off] = 50; off += 1
            resp_buf[unsafe_offset=off] = 13; resp_buf[unsafe_offset=off + 1] = 10; off += 2
            # $keylen\r\nkey\r\n
            resp_buf[unsafe_offset=off] = 36; off += 1
            off += format_int_to_buf(resp_buf.unsafe_offset(off), 0, Int64(key_len))
            resp_buf[unsafe_offset=off] = 13; resp_buf[unsafe_offset=off + 1] = 10; off += 2
            unsafe_memcpy(dest=resp_buf.unsafe_offset(off), src=key_ptr, count=key_len); off += key_len
            resp_buf[unsafe_offset=off] = 13; resp_buf[unsafe_offset=off + 1] = 10; off += 2
            # Get stream and find new entries
            var key_val = GenericValue.borrow(key_ptr, key_len)
            var sd = get_stream(keyspace, key_val)
            if is_not_null(sd):
                var after_ms = reader.after_ms[ki]
                var after_seq = reader.after_seq[ki]
                var entry_count = 0
                for ei in range(sd[].count):
                    if sd[].entries[unsafe_offset=ei].deleted: continue
                    if id_gt(sd[].entries[unsafe_offset=ei].id_ms, sd[].entries[unsafe_offset=ei].id_seq, after_ms, after_seq):
                        entry_count += 1
                        if entry_count >= reader.count_limit: break
                # Write entries array header
                resp_buf[unsafe_offset=off] = 42; off += 1
                off += format_int_to_buf(resp_buf.unsafe_offset(off), 0, Int64(entry_count))
                resp_buf[unsafe_offset=off] = 13; resp_buf[unsafe_offset=off + 1] = 10; off += 2
                var written = 0
                for ei in range(sd[].count):
                    if written >= entry_count: break
                    if sd[].entries[unsafe_offset=ei].deleted: continue
                    var e = sd[].entries[unsafe_offset=ei]
                    if id_gt(e.id_ms, e.id_seq, after_ms, after_seq):
                        # Write entry inline to resp_buf
                        off = write_entry_to_buf(e, resp_buf, off)
                        written += 1
            else:
                # Empty array
                resp_buf[unsafe_offset=off] = 42; resp_buf[unsafe_offset=off + 1] = 48
                resp_buf[unsafe_offset=off + 2] = 13; resp_buf[unsafe_offset=off + 3] = 10; off += 4
            _ = server.send(reader.fd, resp_buf, off)
            resp_buf.unsafe_free()
            # Remove reader
            for k2 in range(reader.num_keys):
                if is_not_null(reader.key_ptrs[k2]):
                    reader.key_ptrs[k2].unsafe_free()
            registry.readers[unsafe_offset=ri].active = False
            registry.count_ptr[unsafe_offset=0] -= 1
            break  # only wake once per reader


@always_inline
def handle_xread(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                mut blocked_readers: BlockedReaderRegistry,
                fd: Int32,
                server: TCPServer) raises -> Int:
    """XREAD [COUNT count] [BLOCK ms] STREAMS key [key ...] id [id ...]."""
    var j = i + 1
    var count_limit = 100  # default
    var block_ms: Int64 = -1  # -1 = no BLOCK, 0 = infinite, >0 = timeout ms
    var streams_idx = -1

    # Parse options
    while j < num_tokens:
        var tp = tokens[unsafe_offset=j].ptr; var tl = tokens[unsafe_offset=j].length
        if tl == 5 and (tp[unsafe_offset=0] | 0x20) == 99 and (tp[unsafe_offset=1] | 0x20) == 111 and (tp[unsafe_offset=2] | 0x20) == 117:
            # COUNT
            if j + 1 < num_tokens:
                count_limit = strict_atol(tokens[unsafe_offset=j + 1].value())
                j += 2; continue
        elif tl == 5 and (tp[unsafe_offset=0] | 0x20) == 98 and (tp[unsafe_offset=1] | 0x20) == 108:
            # BLOCK
            if j + 1 < num_tokens:
                block_ms = Int64(strict_atol(tokens[unsafe_offset=j + 1].value()))
                j += 2; continue
        elif tl == 7 and (tp[unsafe_offset=0] | 0x20) == 115 and (tp[unsafe_offset=1] | 0x20) == 116 and (tp[unsafe_offset=2] | 0x20) == 114:
            # STREAMS
            streams_idx = j + 1
            break
        j += 1

    if streams_idx < 0 or streams_idx >= num_tokens:
        writer.append_null_response()
        return num_tokens - i - 1

    # Count stream keys: tokens from streams_idx, keys and IDs split evenly
    var remaining = num_tokens - streams_idx
    var num_streams = remaining // 2
    if num_streams < 1:
        writer.append_null_response()
        return num_tokens - i - 1

    # Check if any stream has data
    var has_data = False
    for si in range(num_streams):
        var key_val = GenericValue.borrow(tokens[unsafe_offset=streams_idx + si].ptr, tokens[unsafe_offset=streams_idx + si].length)
        var sd = get_stream(keyspace, key_val)
        if is_not_null(sd) and sd[].alive > 0:
            # Parse the after-ID
            var id_tok = tokens[unsafe_offset=streams_idx + num_streams + si]
            var use_last = id_tok.length == 1 and id_tok.ptr[unsafe_offset=0] == 36  # '$' = last ID
            if use_last:
                continue  # $ means only new entries, which don't exist yet in non-blocking mode
            var r = parse_stream_id(id_tok.ptr, id_tok.length)
            var after_ms = r.ms; var after_seq = r.seq
            # Check if there's any entry after this ID
            for ei in range(sd[].count):
                if sd[].entries[unsafe_offset=ei].deleted: continue
                var e = sd[].entries[unsafe_offset=ei]
                if id_gt(e.id_ms, e.id_seq, after_ms, after_seq):
                    has_data = True
                    break
        if has_data: break

    if not has_data:
        # If BLOCK specified and fd valid, register as blocked reader
        if block_ms >= 0 and Int(fd) >= 0:
            var reader = BlockedReader()
            reader.fd = fd
            reader.active = True
            reader.count_limit = count_limit
            if block_ms > 0:
                reader.timeout_ms = Int64(_get_now_ns() // 1000000) + block_ms
            else:
                reader.timeout_ms = 0  # infinite
            var nk = min(num_streams, MAX_BLOCKED_KEYS)
            reader.num_keys = nk
            for ki in range(nk):
                var kp = tokens[unsafe_offset=streams_idx + ki].ptr
                var kl = tokens[unsafe_offset=streams_idx + ki].length
                reader.key_hashes[ki] = BlockedReaderRegistry.hash_key(kp, kl)
                var id_tok = tokens[unsafe_offset=streams_idx + num_streams + ki]
                var use_last = id_tok.length == 1 and id_tok.ptr[unsafe_offset=0] == 36
                if use_last:
                    # $ = wait for new entries from now
                    var sd2 = get_stream(keyspace, GenericValue.borrow(kp, kl))
                    if is_not_null(sd2):
                        reader.after_ms[ki] = sd2[].last_id_ms
                        reader.after_seq[ki] = sd2[].last_id_seq
                    else:
                        reader.after_ms[ki] = 0; reader.after_seq[ki] = 0
                else:
                    var r2 = parse_stream_id(id_tok.ptr, id_tok.length)
                    reader.after_ms[ki] = r2.ms; reader.after_seq[ki] = r2.seq
                # Copy key name for later response
                var key_copy = alloc[UInt8](kl)
                unsafe_memcpy(dest=key_copy, src=kp, count=kl)
                reader.key_ptrs[ki] = key_copy
                reader.key_lens[ki] = kl
            _ = blocked_readers.add(reader)
            # Don't write any response — client waits for data or timeout
            return num_tokens - i - 1
        writer.append_null_response()
        return num_tokens - i - 1

    # Write outer array: *<num_streams>\r\n
    writer.buffer[unsafe_offset=writer.offset] = 42
    writer.offset += 1
    writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(num_streams))
    writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
    writer.offset += 2

    for si in range(num_streams):
        var key_ptr = tokens[unsafe_offset=streams_idx + si].ptr
        var key_len = tokens[unsafe_offset=streams_idx + si].length
        var key_val = GenericValue.borrow(key_ptr, key_len)
        var sd = get_stream(keyspace, key_val)

        # Each stream: *2\r\n $keylen\r\nkey\r\n *<entries>\r\n [entries...]
        writer.append_to_response("*2\r\n".unsafe_ptr(), 4)
        writer.append_bulk_string_response(key_ptr, key_len)

        if is_null(sd) or sd[].alive == 0:
            writer.append_empty_array_response()
            continue

        var id_tok = tokens[unsafe_offset=streams_idx + num_streams + si]
        var use_last = id_tok.length == 1 and id_tok.ptr[unsafe_offset=0] == 36
        var after_ms: UInt64
        var after_seq: UInt64
        if not use_last:
            var r = parse_stream_id(id_tok.ptr, id_tok.length)
            after_ms = r.ms; after_seq = r.seq
        else:
            after_ms = UInt64(0xFFFFFFFFFFFFFFFF); after_seq = UInt64(0xFFFFFFFFFFFFFFFF)

        # Count entries after ID
        var entry_count = 0
        for ei in range(sd[].count):
            if sd[].entries[unsafe_offset=ei].deleted: continue
            var e = sd[].entries[unsafe_offset=ei]
            if id_gt(e.id_ms, e.id_seq, after_ms, after_seq):
                entry_count += 1
                if entry_count >= count_limit: break

        writer.buffer[unsafe_offset=writer.offset] = 42
        writer.offset += 1
        writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(entry_count))
        writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
        writer.offset += 2

        var written = 0
        for ei in range(sd[].count):
            if written >= entry_count: break
            if sd[].entries[unsafe_offset=ei].deleted: continue
            var e = sd[].entries[unsafe_offset=ei]
            if id_gt(e.id_ms, e.id_seq, after_ms, after_seq):
                write_entry_to_response(e, writer)
                written += 1

    return num_tokens - i - 1


@always_inline
def handle_xdel(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """XDEL key id [id ...] → :N."""
    if i + 2 >= num_tokens:
        writer.append_int_response(0)
        return 0
    var key_ptr = tokens[unsafe_offset=i + 1].ptr
    var key_len = tokens[unsafe_offset=i + 1].length
    var key_val = GenericValue.borrow(key_ptr, key_len)
    var sd = get_stream(keyspace, key_val)
    if is_null(sd):
        if stream_key_is_wrongtype(keyspace, key_val):   # gh #232
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            writer.append_int_response(0)
        return num_tokens - i - 1

    var deleted_count: Int64 = 0
    for j in range(i + 2, num_tokens):
        var r = parse_stream_id(tokens[unsafe_offset=j].ptr, tokens[unsafe_offset=j].length)
        var del_ms = r.ms; var del_seq = r.seq
        for ei in range(sd[].count):
            if not sd[].entries[unsafe_offset=ei].deleted and sd[].entries[unsafe_offset=ei].id_ms == del_ms and sd[].entries[unsafe_offset=ei].id_seq == del_seq:
                sd[].kill(ei)
                deleted_count += 1
                # gh #174: log only IDs that actually matched a live entry, so
                # replay tombstones exactly what the live path did.
                if is_not_null(wal):
                    _ = wal[].append_u64x2_val(
                        27, key_ptr, key_len, del_ms, del_seq,
                        null_ptr[UInt8, MutUntrackedOrigin](), 0)
                break
    sd[].compact()

    writer.append_int_response(deleted_count)
    return num_tokens - i - 1


@always_inline
def handle_xtrim(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """XTRIM key MAXLEN|MINID [~] threshold → :N."""
    if i + 3 >= num_tokens:
        writer.append_int_response(0)
        return 0
    var key_val = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    var sd = get_stream(keyspace, key_val)
    if is_null(sd):
        if stream_key_is_wrongtype(keyspace, key_val):   # gh #232
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            writer.append_int_response(0)
        return num_tokens - i - 1

    var j = i + 2
    var tp = tokens[unsafe_offset=j].ptr; var tl = tokens[unsafe_offset=j].length
    j += 1
    # Skip ~ if present
    if j < num_tokens and tokens[unsafe_offset=j].length == 1 and tokens[unsafe_offset=j].ptr[unsafe_offset=0] == 126: j += 1
    if j >= num_tokens:
        writer.append_int_response(0)
        return num_tokens - i - 1

    var threshold = strict_atol(tokens[unsafe_offset=j].value())
    var trimmed: Int64 = 0

    if tl == 6 and (tp[unsafe_offset=0] | 0x20) == 109 and (tp[unsafe_offset=1] | 0x20) == 97:
        # MAXLEN — one pass from the oldest end (this rescanned from index 0
        # over every tombstone once per trimmed entry).
        var to_trim = sd[].alive - threshold
        for ei in range(sd[].count):
            if to_trim <= 0:
                break
            if not sd[].entries[unsafe_offset=ei].deleted:
                sd[].kill(ei)
                trimmed += 1
                to_trim -= 1
        sd[].compact()
    # MINID not implemented yet

    writer.append_int_response(trimmed)
    return num_tokens - i - 1


# ── Stub handlers (consumer groups) ──

@always_inline
def handle_xack(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """XACK is part of consumer-group surface (gh #81) — not implemented."""
    writer.append_error_response("ERR consumer groups not supported")
    return num_tokens - i - 1


@always_inline
def handle_xreadgroup(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """XREADGROUP — consumer-group surface (gh #81), not implemented."""
    writer.append_error_response("ERR consumer groups not supported")
    return num_tokens - i - 1


@always_inline
def handle_xinfo(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """XINFO STREAM key → basic info. GROUPS/CONSUMERS error out (gh #81)."""
    if i + 2 < num_tokens:
        var sub = tokens[unsafe_offset=i + 1].ptr; var sub_len = tokens[unsafe_offset=i + 1].length
        # XINFO GROUPS | XINFO CONSUMERS — consumer-group surface, not implemented.
        # Fake `*0` would lie to clients that the stream has no groups.
        if sub_len == 6 and (sub[unsafe_offset=0] | 0x20) == 103:
            writer.append_error_response("ERR consumer groups not supported")
            return num_tokens - i - 1
        if sub_len == 9 and (sub[unsafe_offset=0] | 0x20) == 99:
            writer.append_error_response("ERR consumer groups not supported")
            return num_tokens - i - 1
        if sub_len == 6 and (sub[unsafe_offset=0] | 0x20) == 115:
            # XINFO STREAM key
            var key_val = GenericValue.borrow(tokens[unsafe_offset=i + 2].ptr, tokens[unsafe_offset=i + 2].length)
            var sd = get_stream(keyspace, key_val)
            if is_not_null(sd):
                # Return basic info as flat array.
                #
                # Every length here used to be hand-typed and two of the three
                # were wrong, so EVERY `XINFO STREAM` reply Pion has ever sent
                # was malformed: the header claimed 18 bytes for a 16-byte
                # literal (injecting two NULs before the length integer), and
                # the second field declared `$15` for the 17-byte
                # `last-generated-id` while writing 21 of its 24 bytes — so the
                # name arrived truncated with no CRLF. A lenient client papered
                # over it; a strict parser desyncs. Same failure this codebase
                # already ate in pubsub.mojo (five miscounted literals): never
                # hand-type a RESP length. Bind the literal to a name and let
                # `byte_length()` do the counting.
                var hdr = "*6\r\n$6\r\nlength\r\n"
                writer.append_to_response(hdr.unsafe_ptr(), hdr.byte_length())
                writer.append_int_response(Int64(sd[].alive))
                var lgi = "$17\r\nlast-generated-id\r\n"
                writer.append_to_response(lgi.unsafe_ptr(), lgi.byte_length())
                var id_buf = stack_allocation[40, UInt8]()
                var id_len = format_stream_id(id_buf, sd[].last_id_ms, sd[].last_id_seq)
                writer.append_bulk_string_response(id_buf, id_len)
                var ent = "$7\r\nentries\r\n"
                writer.append_to_response(ent.unsafe_ptr(), ent.byte_length())
                writer.append_int_response(Int64(sd[].count))
            elif stream_key_is_wrongtype(keyspace, key_val):   # gh #232
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            else:
                writer.append_error_response("ERR no such key")
            return num_tokens - i - 1
    writer.append_empty_array_response()
    return num_tokens - i - 1


@always_inline
def handle_xgroup(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """XGROUP — consumer-group surface (gh #81), not implemented."""
    writer.append_error_response("ERR consumer groups not supported")
    return num_tokens - i - 1


@always_inline
def handle_xclaim(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """XCLAIM — consumer-group surface (gh #81), not implemented."""
    writer.append_error_response("ERR consumer groups not supported")
    return num_tokens - i - 1


@always_inline
def handle_xpending(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """XPENDING — consumer-group surface (gh #81), not implemented."""
    writer.append_error_response("ERR consumer groups not supported")
    return num_tokens - i - 1


@always_inline
def handle_xrevrange_stub(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """Fallback stub."""
    writer.append_empty_array_response()
    return num_tokens - i - 1


@always_inline
def handle_xautoclaim(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """XAUTOCLAIM — consumer-group surface (gh #81), not implemented."""
    writer.append_error_response("ERR consumer groups not supported")
    return num_tokens - i - 1
