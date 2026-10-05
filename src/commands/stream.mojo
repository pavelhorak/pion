"""Stream commands: XADD, XLEN, XRANGE, XREVRANGE, XREAD, XDEL, XTRIM, XINFO, XACK, XGROUP, XCLAIM, XPENDING, XAUTOCLAIM.

Implements a persistent append-only log per key (ValueType.STREAM = 13).
Consumer groups remain stubs.
"""
from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, stack_allocation
from std.collections import Array, List, Span
from std.memory import unsafe_memcpy, unsafe_memset
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.utils import strict_atol, format_int_to_buf, arg_eq, parse_int64_strict
from src.network.fast_path import _get_now_ns
from src.network.server import TCPServer
# gh #174: StreamEntry/StreamData live in src/common so src/io/wal.mojo can
# replay XADD records without the wal -> stream -> fast_path -> wal cycle.
# Re-exported here so existing importers of this module are unaffected.
from src.common.stream_data import StreamEntry, StreamData
from src.io.wal import WAL

# How many XREAD BLOCK clients one worker parks at once. Past it, XREAD BLOCK
# answers an error instead of leaving the client without a reply.
comptime MAX_BLOCKED_READERS = 4096


# ── Blocked XREAD state ──

struct BlockedReader(Copyable, Movable):
    """An XREAD BLOCK whose connection is parked until one of its streams gets
    an entry after its ID, or its deadline passes. Answered by the event loop
    (`NetworkEngine._service_blocked_readers`), which then runs whatever the
    client pipelined behind it: a blocked client runs nothing else first, as
    in Redis (a PING pipelined behind it used to be answered before it)."""
    var fd: Int32
    var ready: Bool             # an XADD reached one of its streams
    var count_limit: Int        # 0 = no COUNT: every entry
    var timeout_ms: Int64       # 0 = no deadline, else absolute ms (_get_now_ns clock)
    var keys: List[String]      # stream names, byte for byte
    var after_ms: List[UInt64]
    var after_seq: List[UInt64]

    def __init__(out self):
        self.fd = Int32(-1); self.ready = False
        self.count_limit = 0; self.timeout_ms = 0
        self.keys = List[String]()
        self.after_ms = List[UInt64]()
        self.after_seq = List[UInt64]()


struct BlockedReaderRegistry(Movable):
    """Per-worker parked XREAD BLOCK clients."""
    var readers: List[BlockedReader]
    var count_ptr: Pointer[Int, MutUntrackedOrigin]   # len(readers), read by every event-loop tick

    @always_inline
    def _count(self) -> Int:
        return self.count_ptr[unsafe_offset=0]

    def __init__(out self):
        self.readers = List[BlockedReader]()
        self.count_ptr = alloc[Int](1)
        self.count_ptr[unsafe_offset=0] = 0

    def add(mut self, var reader: BlockedReader) -> Bool:
        """Register a blocked reader. Returns False if full."""
        if len(self.readers) >= MAX_BLOCKED_READERS:
            return False
        self.readers.append(reader^)
        self.count_ptr[unsafe_offset=0] = len(self.readers)
        return True

    def remove_at(mut self, k: Int):
        """Drop reader k (swap with the last)."""
        var last = len(self.readers) - 1
        if k != last:
            self.readers[k] = self.readers[last].copy()
        _ = self.readers.pop()
        self.count_ptr[unsafe_offset=0] = len(self.readers)

    def remove_fd(mut self, fd: Int32):
        """The connection closed while blocked."""
        var k = 0
        while k < len(self.readers):
            if self.readers[k].fd == fd:
                self.remove_at(k)
            else:
                k += 1

    def mark_ready(mut self, key_ptr: Pointer[UInt8, MutUntrackedOrigin], key_len: Int):
        """An XADD to `key`: every reader blocked on it is answered on the
        event loop's next tick. Nothing is written here; the XADD's own
        connection is mid-batch."""
        for k in range(len(self.readers)):
            for ki in range(len(self.readers[k].keys)):
                if _bytes_eq(self.readers[k].keys[ki], key_ptr, key_len):
                    self.readers[k].ready = True
                    break


@always_inline
def _bytes_eq(s: String, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    if s.byte_length() != n:
        return False
    var sp = s.unsafe_ptr()
    for j in range(n):
        if sp[j] != p[unsafe_offset=j]:
            return False
    return True


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
def id_ge(a_ms: UInt64, a_seq: UInt64, b_ms: UInt64, b_seq: UInt64) -> Bool:
    return a_ms > b_ms or (a_ms == b_ms and a_seq >= b_seq)


@always_inline
def id_le(a_ms: UInt64, a_seq: UInt64, b_ms: UInt64, b_seq: UInt64) -> Bool:
    return a_ms < b_ms or (a_ms == b_ms and a_seq <= b_seq)


@always_inline
def id_gt(a_ms: UInt64, a_seq: UInt64, b_ms: UInt64, b_seq: UInt64) -> Bool:
    return a_ms > b_ms or (a_ms == b_ms and a_seq > b_seq)


# ── Stream IDs and the XADD / XTRIM arguments, as Redis parses them ──────────

@fieldwise_init
struct ParsedID(Copyable, Movable, ImplicitlyCopyable):
    var ok: Bool
    var ms: UInt64
    var seq: UInt64
    var seq_given: Bool      # False for the `<ms>-*` form


def _string2ull(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, mut out: UInt64) -> Bool:
    """Redis's string2ull: an exact integer that is not negative; failing
    that, what strtoull consumes whole — leading spaces, a sign, digits, no
    overflow, nothing after."""
    var r = parse_int64_strict(p, n)
    if r.ok:
        if r.value < 0:
            return False
        out = UInt64(r.value)
        return True
    var i = 0
    while i < n and (p[unsafe_offset=i] == 32 or (p[unsafe_offset=i] >= 9 and p[unsafe_offset=i] <= 13)):
        i += 1
    var neg = False
    if i < n and (p[unsafe_offset=i] == 43 or p[unsafe_offset=i] == 45):
        neg = p[unsafe_offset=i] == 45
        i += 1
    var start = i
    var v: UInt64 = 0
    while i < n and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
        var d = UInt64(p[unsafe_offset=i] - 48)
        if v > (UInt64.MAX - d) // 10:
            return False
        v = v * 10 + d
        i += 1
    if i == start or i != n:
        return False
    out = (UInt64(0) - v) if neg else v
    return True


def parse_id(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, missing_seq: UInt64,
             strict: Bool, seq_star: Bool) -> ParsedID:
    """streamGenericParseIDOrReply: `-`, `+` (unless `strict`), `<ms>`
    (seq = `missing_seq`), `<ms>-<seq>`, and `<ms>-*` when `seq_star`. The
    parse this replaces skipped every byte that was not a digit."""
    var bad = ParsedID(False, 0, 0, True)
    if n == 0 or n > 127:
        return bad
    if n == 1 and (p[unsafe_offset=0] == 45 or p[unsafe_offset=0] == 43):
        if strict:
            return bad
        if p[unsafe_offset=0] == 45:
            return ParsedID(True, 0, 0, True)
        return ParsedID(True, UInt64.MAX, UInt64.MAX, True)
    var dash = -1
    for k in range(n):
        if p[unsafe_offset=k] == 45:
            dash = k
            break
    var ms: UInt64 = 0
    if not _string2ull(p, dash if dash >= 0 else n, ms):
        return bad
    if dash < 0:
        return ParsedID(True, ms, missing_seq, True)
    var sp = p.unsafe_offset(dash + 1)
    var sn = n - dash - 1
    if seq_star and sn == 1 and sp[unsafe_offset=0] == 42:
        return ParsedID(True, ms, 0, False)
    var seq: UInt64 = 0
    if not _string2ull(sp, sn, seq):
        return bad
    return ParsedID(True, ms, seq, True)


comptime _E_BAD_ID = "ERR Invalid stream ID specified as stream command argument"
comptime TRIM_NONE = 0
comptime TRIM_MAXLEN = 1
comptime TRIM_MINID = 2


@fieldwise_init
struct AddTrimArgs(Copyable, Movable, ImplicitlyCopyable):
    var ok: Bool
    var strategy: Int
    var maxlen: Int
    var minid_ms: UInt64
    var minid_seq: UInt64
    var approx: Bool
    var limit: Int            # entries one call may remove; 0 = no limit
    var no_mkstream: Bool
    var id_idx: Int           # XADD: the token holding the id (or "*")


def parse_add_trim(tokens: Pointer[RESP3Token, MutUntrackedOrigin], start: Int, end: Int,
                   xadd: Bool, mut writer: ResponseWriter) -> AddTrimArgs:
    """streamParseAddOrTrimArgsOrReply: [NOMKSTREAM] [KEEPREF|DELREF|ACKED]
    [MAXLEN|MINID [=|~] threshold [LIMIT count]] — then, for XADD, the id.
    On error the reply is written and `ok` is False.

    MINID was parsed and then ignored (no trim, success reply), `=` was not
    understood, LIMIT not at all, `MAXLEN 0` kept everything, and the
    keywords were matched by their first letters. KEEPREF/DELREF/ACKED choose
    what happens to consumer-group references; with no consumer groups they
    are all the same, and accepted."""
    var a = AddTrimArgs(False, TRIM_NONE, -1, 0, 0, False, -1, False, end)
    var limit_given = False
    var j = start
    while j < end:
        var t = tokens[j]
        var more = end - 1 - j
        if xadd and t.length == 1 and t.ptr[unsafe_offset=0] == 42:     # "*"
            break
        elif (arg_eq(t.ptr, t.length, "maxlen") or arg_eq(t.ptr, t.length, "minid")) and more > 0:
            if a.strategy != TRIM_NONE:
                writer.append_error_response("ERR syntax error, MAXLEN and MINID options at the same time are not compatible")
                return a
            var is_max = arg_eq(t.ptr, t.length, "maxlen")
            a.approx = False
            var nx = tokens[j + 1]
            if more >= 2 and nx.length == 1 and nx.ptr[unsafe_offset=0] == 126:     # "~"
                a.approx = True
                j += 1
            elif more >= 2 and nx.length == 1 and nx.ptr[unsafe_offset=0] == 61:    # "="
                j += 1
            var th = tokens[j + 1]
            if is_max:
                var v = parse_int64_strict(th.ptr, th.length)
                if not v.ok:
                    writer.append_error_response("ERR value is not an integer or out of range")
                    return a
                if v.value < 0:
                    writer.append_error_response("ERR The MAXLEN argument must be >= 0.")
                    return a
                a.maxlen = Int(v.value)
                a.strategy = TRIM_MAXLEN
            else:
                var mid = parse_id(th.ptr, th.length, 0, True, False)
                if not mid.ok:
                    writer.append_error_response(_E_BAD_ID)
                    return a
                a.minid_ms = mid.ms
                a.minid_seq = mid.seq
                a.strategy = TRIM_MINID
            j += 2
            continue
        elif arg_eq(t.ptr, t.length, "limit") and more > 0:
            var lv = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not lv.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return a
            if lv.value < 0:
                writer.append_error_response("ERR The LIMIT argument must be >= 0.")
                return a
            a.limit = Int(lv.value)
            limit_given = True
            j += 2
            continue
        elif xadd and arg_eq(t.ptr, t.length, "nomkstream"):
            a.no_mkstream = True
        elif arg_eq(t.ptr, t.length, "keepref") or arg_eq(t.ptr, t.length, "delref") \
                or arg_eq(t.ptr, t.length, "acked"):
            pass
        elif xadd:
            var id = parse_id(t.ptr, t.length, 0, True, True)
            if not id.ok:
                writer.append_error_response(_E_BAD_ID)
                return a
            break
        else:
            writer.append_error_response("ERR syntax error")
            return a
        j += 1
    a.id_idx = j
    if limit_given and a.limit != 0 and a.strategy == TRIM_NONE:
        writer.append_error_response("ERR syntax error, LIMIT cannot be used without specifying a trimming strategy")
        return a
    if not xadd and a.strategy == TRIM_NONE:
        writer.append_error_response("ERR syntax error, XTRIM must be called with a trimming strategy")
        return a
    if limit_given:
        if not a.approx:
            writer.append_error_response("ERR syntax error, LIMIT cannot be used without the special ~ option")
            return a
    else:
        # ~ without LIMIT removes at most 100 x stream-node-max-entries a call.
        a.limit = 10000 if a.approx else 0
    a.ok = True
    return a


def stream_trim(sd: Pointer[StreamData, MutUntrackedOrigin], a: AddTrimArgs,
                wal: Pointer[WAL, MutUntrackedOrigin], key_ptr: Pointer[UInt8, MutUntrackedOrigin],
                key_len: Int) -> Int:
    """Remove the oldest entries past MAXLEN, or older than MINID, at most
    `a.limit` of them (0 = no limit), logging each (record 27). With `~`
    Redis removes whole internal nodes only, so it may keep more than asked;
    Pion has no nodes and trims exactly, which is within the `~` contract."""
    var removed = 0
    for ei in range(sd[].count):
        if a.limit > 0 and removed >= a.limit:
            break
        if sd[].entries[unsafe_offset=ei].deleted:
            continue
        var e_ms = sd[].entries[unsafe_offset=ei].id_ms
        var e_seq = sd[].entries[unsafe_offset=ei].id_seq
        if a.strategy == TRIM_MAXLEN:
            if sd[].alive <= a.maxlen:
                break
        elif not (e_ms < a.minid_ms or (e_ms == a.minid_ms and e_seq < a.minid_seq)):
            break
        sd[].kill(ei)
        removed += 1
        if is_not_null(wal):
            _ = wal[].append_u64x2_val(27, key_ptr, key_len, e_ms, e_seq,
                                       null_ptr[UInt8, MutUntrackedOrigin](), 0)
    sd[].compact()
    return removed


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

# Packed entry layout: [u32 flen][field][u32 vlen][value]... little-endian.
# The lengths were u16, which held a field or value of at most 64 KB. WAL and
# snapshot records of this layout are cmd 34; cmd 23 records (u16) still replay.
comptime STREAM_LEN_BYTES = 4


@always_inline
def _pack_len(p: Pointer[UInt8, MutUntrackedOrigin]) -> Int:
    return Int((p.unsafe_bitcast[UInt32]())[])


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
    writer.append_array_header(e.num_fields * 2)
    var doff = 0
    for _ in range(e.num_fields):
        if doff + STREAM_LEN_BYTES > e.data_len: break
        var flen = _pack_len(e.data.unsafe_offset(doff))
        doff += STREAM_LEN_BYTES
        if doff + flen > e.data_len: break
        writer.append_bulk_string_response(e.data.unsafe_offset(doff), flen)
        doff += flen
        if doff + STREAM_LEN_BYTES > e.data_len: break
        var vlen = _pack_len(e.data.unsafe_offset(doff))
        doff += STREAM_LEN_BYTES
        if doff + vlen > e.data_len: break
        writer.append_bulk_string_response(e.data.unsafe_offset(doff), vlen)
        doff += vlen


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

    # Redis's streamParseAddOrTrimArgsOrReply: the options, then the id.
    var at = parse_add_trim(tokens, i + 2, num_tokens, True, writer)
    if not at.ok:
        return num_tokens - i - 1
    var field_start = at.id_idx + 1
    var nf = (num_tokens - field_start) // 2
    if num_tokens - field_start < 2 or (num_tokens - field_start) % 2 == 1:
        writer.append_error_response("ERR wrong number of arguments for 'xadd' command")
        return num_tokens - i - 1
    var no_mkstream = at.no_mkstream
    var id_tok = tokens[unsafe_offset=at.id_idx]
    var id_auto = id_tok.length == 1 and id_tok.ptr[unsafe_offset=0] == 42
    var pid = ParsedID(True, 0, 0, True)
    if not id_auto:
        pid = parse_id(id_tok.ptr, id_tok.length, 0, True, True)
    # Before the key is touched, so a new stream is not left empty.
    if not id_auto and pid.seq_given and pid.ms == 0 and pid.seq == 0:
        writer.append_error_response("ERR The ID specified in XADD must be greater than 0-0")
        return num_tokens - i - 1

    # Pack field-value data: [u32 flen][bytes][u32 vlen][bytes]...
    var pack_size = 0
    for fi in range(nf):
        pack_size += (2 * STREAM_LEN_BYTES + tokens[unsafe_offset=field_start + fi * 2].length
                      + tokens[unsafe_offset=field_start + fi * 2 + 1].length)
    var pack_buf = alloc[UInt8](pack_size)
    var poff = 0
    for fi in range(nf):
        var ft = tokens[unsafe_offset=field_start + fi * 2]
        var vt = tokens[unsafe_offset=field_start + fi * 2 + 1]
        (pack_buf.unsafe_offset(poff)).unsafe_bitcast[UInt32]()[] = UInt32(ft.length); poff += STREAM_LEN_BYTES
        unsafe_memcpy(dest=pack_buf.unsafe_offset(poff), src=ft.ptr, count=ft.length); poff += ft.length
        (pack_buf.unsafe_offset(poff)).unsafe_bitcast[UInt32]()[] = UInt32(vt.length); poff += STREAM_LEN_BYTES
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

    if sd[].last_id_ms == UInt64.MAX and sd[].last_id_seq == UInt64.MAX:
        pack_buf.unsafe_free()
        writer.append_error_response("ERR The stream has exhausted the last possible ID, unable to add more items")
        return num_tokens - i - 1
    var id_ms: UInt64
    var id_seq: UInt64
    if id_auto:
        # streamNextID: now, or the last id plus one when the clock is behind.
        id_ms = UInt64(_get_now_ns() // 1000000)
        if id_ms > sd[].last_id_ms:
            id_seq = 0
        elif sd[].last_id_seq == UInt64.MAX:
            id_ms = sd[].last_id_ms + 1
            id_seq = 0
        else:
            id_ms = sd[].last_id_ms
            id_seq = sd[].last_id_seq + 1
    else:
        id_ms = pid.ms
        id_seq = pid.seq
        if not pid.seq_given:
            # `<ms>-*`: the next sequence in that millisecond.
            if sd[].last_id_ms == pid.ms:
                if sd[].last_id_seq == UInt64.MAX:
                    pack_buf.unsafe_free()
                    writer.append_error_response("ERR The ID specified in XADD is equal or smaller than the target stream top item")
                    return num_tokens - i - 1
                id_seq = sd[].last_id_seq + 1
            else:
                id_seq = 0
        # gh #242: ids strictly increase. Compared with the stream's last id
        # even once its entries are deleted, as Redis does.
        if id_ms < sd[].last_id_ms or (id_ms == sd[].last_id_ms and id_seq <= sd[].last_id_seq):
            pack_buf.unsafe_free()
            writer.append_error_response("ERR The ID specified in XADD is equal or smaller than the target stream top item")
            return num_tokens - i - 1

    # Append entry
    sd[].append(id_ms, id_seq, pack_buf, pack_size, nf)

    # gh #174: effect-log the entry with its *resolved* ID. `XADD key *` must
    # never replay as "generate an ID now" — recovery runs at a different wall
    # clock, which would renumber the stream and break every consumer cursor.
    if is_not_null(wal):
        _ = wal[].append_u64x2_val(34, key_ptr, key_len,   # u32 lengths
                                   id_ms, id_seq, pack_buf, pack_size)

    if at.strategy != TRIM_NONE:
        _ = stream_trim(sd, at, wal, key_ptr, key_len)

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


def _xrange(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
            mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
            rev: Bool) -> Int:
    """XRANGE key start end [COUNT n] / XREVRANGE key end start [COUNT n], as
    Redis's xrangeGenericCommand: the ids first (`-`, `+`, `<ms>` with the
    open sequence, `(id` for an exclusive bound), then COUNT, then the key.
    A malformed id used to be read digit by digit as some other id, the key
    was looked up first, and anything after the range was ignored. COUNT 0 or
    below answers a null array, as Redis does."""
    var name = String("xrevrange") if rev else String("xrange")
    if num_tokens - i < 4:
        writer.append_error_response("ERR wrong number of arguments for '" + name + "' command")
        return 0
    var st = tokens[unsafe_offset=i + 3] if rev else tokens[unsafe_offset=i + 2]
    var en = tokens[unsafe_offset=i + 2] if rev else tokens[unsafe_offset=i + 3]
    var s_ex = st.length > 1 and st.ptr[unsafe_offset=0] == 40
    var sid = parse_id(st.ptr.unsafe_offset(1) if s_ex else st.ptr, st.length - 1 if s_ex else st.length,
                       0, s_ex, False)
    if not sid.ok:
        writer.append_error_response(_E_BAD_ID)
        return num_tokens - i - 1
    var s_ms = sid.ms
    var s_seq = sid.seq
    if s_ex:                                          # streamIncrID
        if s_seq == UInt64.MAX:
            if s_ms == UInt64.MAX:
                writer.append_error_response("ERR invalid start ID for the interval")
                return num_tokens - i - 1
            s_ms += 1
            s_seq = 0
        else:
            s_seq += 1
    var e_ex = en.length > 1 and en.ptr[unsafe_offset=0] == 40
    var eid = parse_id(en.ptr.unsafe_offset(1) if e_ex else en.ptr, en.length - 1 if e_ex else en.length,
                       UInt64.MAX, e_ex, False)
    if not eid.ok:
        writer.append_error_response(_E_BAD_ID)
        return num_tokens - i - 1
    var e_ms = eid.ms
    var e_seq = eid.seq
    if e_ex:                                          # streamDecrID
        if e_seq == 0:
            if e_ms == 0:
                writer.append_error_response("ERR invalid end ID for the interval")
                return num_tokens - i - 1
            e_ms -= 1
            e_seq = UInt64.MAX
        else:
            e_seq -= 1
    var count = -1
    var j = i + 4
    while j < num_tokens:
        if arg_eq(tokens[unsafe_offset=j].ptr, tokens[unsafe_offset=j].length, "count") and j + 1 < num_tokens:
            var c = parse_int64_strict(tokens[unsafe_offset=j + 1].ptr, tokens[unsafe_offset=j + 1].length)
            if not c.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return num_tokens - i - 1
            count = Int(c.value) if c.value > 0 else 0
            j += 2
        else:
            writer.append_error_response("ERR syntax error")
            return num_tokens - i - 1
    var key_val = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    var sd = get_stream(keyspace, key_val)
    if is_null(sd):
        if stream_key_is_wrongtype(keyspace, key_val):   # gh #232
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            writer.append_empty_array_response()
        return num_tokens - i - 1
    if count == 0:
        writer.append_null_array_response()
        return num_tokens - i - 1
    # One pass picks the entries, so the header is their count.
    var picked = List[Int]()
    var n = sd[].count
    for k in range(n):
        var ei = n - 1 - k if rev else k
        if sd[].entries[unsafe_offset=ei].deleted:
            continue
        var e = sd[].entries[unsafe_offset=ei]
        if id_ge(e.id_ms, e.id_seq, s_ms, s_seq) and id_le(e.id_ms, e.id_seq, e_ms, e_seq):
            picked.append(ei)
            if count > 0 and len(picked) >= count:
                break
    writer.append_array_header(len(picked))
    for k in range(len(picked)):
        write_entry_to_response(sd[].entries[unsafe_offset=picked[k]], writer)
    return num_tokens - i - 1


@always_inline
def handle_xrange(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """XRANGE key start end [COUNT count]."""
    return _xrange(tokens, i, num_tokens, writer, keyspace, False)


@always_inline
def handle_xrevrange(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises -> Int:
    """XREVRANGE key end start [COUNT count]."""
    return _xrange(tokens, i, num_tokens, writer, keyspace, True)


struct XReadID(Copyable, Movable, ImplicitlyCopyable):
    var ok: Bool
    var ms: UInt64
    var seq: UInt64

    def __init__(out self, ok: Bool, ms: UInt64, seq: UInt64):
        self.ok = ok; self.ms = ms; self.seq = seq


def parse_xread_id(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> XReadID:
    """XREAD's ids: Redis's strict parse (`-` and `+` are not ids here; `$`
    and `+` are handled before this), with `<ms>` meaning `<ms>-0`."""
    var r = parse_id(p, n, 0, True, False)
    return XReadID(r.ok, r.ms, r.seq)


def write_xread_reply(mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                      keys: List[String], after_ms: List[UInt64], after_seq: List[UInt64],
                      count_limit: Int) -> Int:
    """XREAD's reply for the streams that have entries after their IDs, and
    only those, as Redis: a map key -> entries under RESP3, an array of
    [key, entries] pairs under RESP2. `count_limit` 0 means every entry.
    Returns how many streams it wrote; with 0 it writes nothing, and the
    caller answers nil or blocks."""
    var with_data = List[Bool]()
    var n_with = 0
    for si in range(len(keys)):
        var kp = keys[si].unsafe_ptr()
        var sd = get_stream(keyspace, GenericValue.borrow(
            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(kp)), keys[si].byte_length()))
        var has = False
        if is_not_null(sd) and sd[].alive > 0:
            for ei in range(sd[].count):
                if sd[].entries[unsafe_offset=ei].deleted: continue
                if id_gt(sd[].entries[unsafe_offset=ei].id_ms, sd[].entries[unsafe_offset=ei].id_seq,
                         after_ms[si], after_seq[si]):
                    has = True
                    break
        with_data.append(has)
        if has:
            n_with += 1
    if n_with == 0:
        return 0
    var as_map = writer.proto == 3
    if as_map:
        writer.append_map_header(n_with)
    else:
        writer.append_array_header(n_with)
    for si in range(len(keys)):
        if not with_data[si]:
            continue
        var kp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(keys[si].unsafe_ptr()))
        var kl = keys[si].byte_length()
        var sd = get_stream(keyspace, GenericValue.borrow(kp, kl))
        if not as_map:
            writer.append_array_header(2)
        writer.append_bulk_string_response(kp, kl)
        var entry_count = 0
        for ei in range(sd[].count):
            if sd[].entries[unsafe_offset=ei].deleted: continue
            if id_gt(sd[].entries[unsafe_offset=ei].id_ms, sd[].entries[unsafe_offset=ei].id_seq,
                     after_ms[si], after_seq[si]):
                entry_count += 1
                if count_limit > 0 and entry_count >= count_limit: break
        writer.append_array_header(entry_count)
        var written = 0
        for ei in range(sd[].count):
            if written >= entry_count: break
            if sd[].entries[unsafe_offset=ei].deleted: continue
            var e = sd[].entries[unsafe_offset=ei]
            if id_gt(e.id_ms, e.id_seq, after_ms[si], after_seq[si]):
                write_entry_to_response(e, writer)
                written += 1
    return n_with


def handle_xread(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                 keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                 mut blocked_readers: BlockedReaderRegistry,
                 fd: Int32, can_block: Bool) raises -> Bool:
    """XREAD [COUNT count] [BLOCK ms] STREAMS key [key ...] id [id ...].
    Returns True when the connection must park: a BLOCK found no data and the
    reader was registered, so no reply is written yet. `can_block` is False
    inside MULTI/EXEC and on the XDP lane, where BLOCK answers at once, as in
    Redis's transactions.

    Rewritten for #30 and #34: the reply listed every named stream, empty
    ones as `*0` (Redis lists only streams with data); without COUNT it
    stopped at 100 entries (Redis returns all of them); option names matched
    on their first letters; an odd key/ID list and a malformed ID were read
    as something; the no-data reply was a nil bulk string, not a nil array."""
    var j = i + 1
    var count_limit = 0
    var block_ms: Int64 = -1   # -1 = no BLOCK, 0 = forever
    var streams_idx = -1
    while j < num_tokens:
        var tp = tokens[unsafe_offset=j].ptr
        var tl = tokens[unsafe_offset=j].length
        if arg_eq(tp, tl, "count") and j + 1 < num_tokens:
            var c = parse_int64_strict(tokens[unsafe_offset=j + 1].ptr, tokens[unsafe_offset=j + 1].length)
            if not c.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return False
            count_limit = Int(c.value) if c.value > 0 else 0
            j += 2
        elif arg_eq(tp, tl, "block") and j + 1 < num_tokens:
            var b = parse_int64_strict(tokens[unsafe_offset=j + 1].ptr, tokens[unsafe_offset=j + 1].length)
            if not b.ok:
                writer.append_error_response("ERR timeout is not an integer or out of range")
                return False
            if b.value < 0:
                writer.append_error_response("ERR timeout is negative")
                return False
            block_ms = b.value
            j += 2
        elif arg_eq(tp, tl, "streams"):
            streams_idx = j + 1
            break
        else:
            writer.append_error_response("ERR syntax error")
            return False
    if streams_idx < 0:
        writer.append_error_response("ERR syntax error")
        return False
    var remaining = num_tokens - streams_idx
    if remaining <= 0 or remaining % 2 != 0:
        writer.append_error_response("ERR Unbalanced 'xread' list of streams: for each stream key an ID or '$' must be specified.")
        return False
    var num_streams = remaining // 2

    var keys = List[String]()
    var after_ms = List[UInt64]()
    var after_seq = List[UInt64]()
    for si in range(num_streams):
        var kt = tokens[unsafe_offset=streams_idx + si]
        var key_val = GenericValue.borrow(kt.ptr, kt.length)
        if stream_key_is_wrongtype(keyspace, key_val):
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return False
        var sd = get_stream(keyspace, key_val)
        var idt = tokens[unsafe_offset=streams_idx + num_streams + si]
        var ms: UInt64 = 0
        var seq: UInt64 = 0
        if idt.length == 1 and idt.ptr[unsafe_offset=0] == 36:          # '$': only what comes next
            if is_not_null(sd):
                ms = sd[].last_id_ms; seq = sd[].last_id_seq
        elif idt.length == 1 and idt.ptr[unsafe_offset=0] == 43:        # '+': the last entry
            var found = False
            if is_not_null(sd):
                var ei = sd[].count - 1
                while ei >= 0:
                    if not sd[].entries[unsafe_offset=ei].deleted:
                        var lm = sd[].entries[unsafe_offset=ei].id_ms
                        var ls = sd[].entries[unsafe_offset=ei].id_seq
                        if ls > 0:
                            ms = lm; seq = ls - 1
                        elif lm > 0:
                            ms = lm - 1; seq = UInt64.MAX
                        found = True
                        break
                    ei -= 1
                if not found:
                    ms = sd[].last_id_ms; seq = sd[].last_id_seq
        elif idt.length == 1 and idt.ptr[unsafe_offset=0] == 62:        # '>': XREADGROUP's
            writer.append_error_response("ERR The > ID can be specified only when calling XREADGROUP using the GROUP <group> <consumer> option.")
            return False
        else:
            var r = parse_xread_id(idt.ptr, idt.length)
            if not r.ok:
                writer.append_error_response("ERR Invalid stream ID specified as stream command argument")
                return False
            ms = r.ms; seq = r.seq
        keys.append(String(StringSpan[MutUntrackedOrigin](
            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=kt.ptr, length=kt.length))))
        after_ms.append(ms)
        after_seq.append(seq)

    if write_xread_reply(writer, keyspace, keys, after_ms, after_seq, count_limit) > 0:
        return False
    if block_ms >= 0 and can_block and Int(fd) >= 0:
        var reader = BlockedReader()
        reader.fd = fd
        reader.count_limit = count_limit
        reader.timeout_ms = (Int64(_get_now_ns() // 1000000) + block_ms) if block_ms > 0 else Int64(0)
        reader.keys = keys^
        reader.after_ms = after_ms^
        reader.after_seq = after_seq^
        if blocked_readers.add(reader^):
            return True
        writer.append_error_response("ERR too many clients blocked on XREAD")
        return False
    writer.append_null_array_response()
    return False


@always_inline
def handle_xdel(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                wal: Pointer[WAL, MutUntrackedOrigin] = null_ptr[WAL, MutUntrackedOrigin]()) raises -> Int:
    """XDEL key id [id ...] → :N. As Redis: the key first (missing: 0), then
    every id strictly (one bad id refuses the command before anything is
    deleted)."""
    if i + 2 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'xdel' command")
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

    for j in range(i + 2, num_tokens):
        if not parse_id(tokens[unsafe_offset=j].ptr, tokens[unsafe_offset=j].length, 0, True, False).ok:
            writer.append_error_response(_E_BAD_ID)
            return num_tokens - i - 1
    var deleted_count: Int64 = 0
    for j in range(i + 2, num_tokens):
        var r = parse_id(tokens[unsafe_offset=j].ptr, tokens[unsafe_offset=j].length, 0, True, False)
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
    """XTRIM key MAXLEN|MINID [=|~] threshold [LIMIT count] → :N removed.
    MINID used to answer 0 and trim nothing. The caller logs the result as a
    key image."""
    if num_tokens - i < 4:
        writer.append_error_response("ERR wrong number of arguments for 'xtrim' command")
        return 0
    var at = parse_add_trim(tokens, i + 2, num_tokens, False, writer)
    if not at.ok:
        return num_tokens - i - 1
    var key_val = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    var sd = get_stream(keyspace, key_val)
    if is_null(sd):
        if stream_key_is_wrongtype(keyspace, key_val):   # gh #232
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            writer.append_int_response(0)
        return num_tokens - i - 1
    var removed = stream_trim(sd, at, null_ptr[WAL, MutUntrackedOrigin](),
                              tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    writer.append_int_response(Int64(removed))
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
                # A map under RESP3, as Redis sends XINFO STREAM (#30).
                writer.append_map_header(3)
                var hdr = "$6\r\nlength\r\n"
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
