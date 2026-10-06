"""Stream consumer groups and Redis 7's stream metadata (#40).

XGROUP, XREADGROUP, XACK, XCLAIM, XAUTOCLAIM and XPENDING answered
"consumer groups not supported", XINFO GROUPS and CONSUMERS the same, and
XSETID did not exist. They now behave as Redis's (compared with Redis 8.10):

  * groups with a last-delivered id, an entries-read counter and lag;
    consumers with seen and active times; the pending entries list with
    delivery counts and times;
  * XREADGROUP with `>` (new entries, BLOCK, NOACK) and with an id (the
    consumer's history, a deleted entry as [id, nil]);
  * XACK; XCLAIM (IDLE, TIME, RETRYCOUNT, FORCE, JUSTID, LASTID); XAUTOCLAIM
    with its cursor and the ids of deleted entries it dropped; both forms of
    XPENDING;
  * XGROUP CREATE / SETID / DESTROY / CREATECONSUMER / DELCONSUMER / HELP;
  * XINFO STREAM [FULL [COUNT n]] / GROUPS / CONSUMERS / HELP, and XSETID,
    with entries-added, max-deleted-entry-id and recorded-first-entry-id.

Redis 8.2's XDELEX and XACKDEL are here too, with the KEEPREF / DELREF /
ACKED treatment of group references that XADD and XTRIM trimming share.

Every change is effect-logged to the WAL (records 38-45, src/io/wal.mojo),
so a restart, a replica and a DUMP payload carry the groups. Redis 8's later
stream features (XREADGROUP CLAIM / MAXCOUNT / MAXSIZE, XNACK, idempotent
XADD) are not implemented; XINFO STREAM leaves out their fields and Redis's
internal radix-tree counts.
"""

from std.memory import alloc, unsafe_memcpy
from std.memory.unsafe_pointer import Pointer
from std.collections import List
from std.ffi import external_call
from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.value import GenericValue, ValueType
from src.common.hash_map import StripedHashMap
from src.common.utils import arg_eq, parse_int64_strict, bytes_to_string
from src.common.stream_data import (StreamData, StreamEntry, StreamGroup, StreamConsumer, StreamNack,
                                    SCG_INVALID_ENTRIES_READ, PEL_DEAD, sid_lt, sid_cmp,
                                    DEL_NONE, DEL_KEEPREF, DEL_DELREF, DEL_ACKED,
                                    encode_group_rec, encode_consumer_rec, encode_nack_rec,
                                    encode_pel_del_rec, encode_meta_rec, encode_group_name_rec,
                                    encode_delconsumer_rec)
from src.commands.stream import (get_stream, get_or_create_stream, stream_key_is_wrongtype, parse_id,
                                 write_entry_to_response, format_stream_id, STREAM_OK, STREAM_WRONGTYPE,
                                 drop_group_refs, stream_delete_entry)
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.io.wal import WAL

comptime _E_BAD_ID = "ERR Invalid stream ID specified as stream command argument"
comptime _E_WRONGTYPE = "WRONGTYPE Operation against a key holding the wrong kind of value"
comptime _E_NOKEY_XGROUP = ("ERR The XGROUP subcommand requires the key to exist. Note that for CREATE you may "
                            + "want to use the MKSTREAM option to create an empty stream automatically.")

# XREADGROUP's outcomes for its caller
comptime XRG_DONE = 0     # replied
comptime XRG_BLOCK = 1    # nothing to serve and BLOCK given: the caller parks the connection


@always_inline
def _now_ms() -> Int64:
    return external_call["pion_unix_ms", Int64]()


def _text(t: RESP3Token) -> String:
    return bytes_to_string(t.ptr, t.length)


def write_id(mut writer: ResponseWriter, ms: UInt64, seq: UInt64):
    var buf = alloc[UInt8](48)
    var n = format_stream_id(buf, ms, seq)
    writer.append_bulk_string_response(buf, n)
    buf.free()


def _write_name(mut writer: ResponseWriter, name: List[UInt8]):
    writer.append_bulk_string_response(
        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name.unsafe_ptr())), len(name))


def _log(wal: Pointer[WAL, MutUntrackedOrigin], cmd: UInt8, kt: RESP3Token, payload: List[UInt8]):
    if is_null(wal):
        return
    _ = wal[].append_kv(cmd, kt.ptr, kt.length,
                        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(payload.unsafe_ptr())),
                        len(payload))


def _name_bytes(t: RESP3Token) -> List[UInt8]:
    var out = List[UInt8](capacity=t.length)
    for k in range(t.length):
        out.append(t.ptr[k])
    return out^


@always_inline
def _incr(mut ms: UInt64, mut seq: UInt64) -> Bool:
    """streamIncrID; False past the largest id."""
    if seq == UInt64.MAX:
        if ms == UInt64.MAX:
            return False
        ms += 1
        seq = 0
    else:
        seq += 1
    return True


@always_inline
def _decr(mut ms: UInt64, mut seq: UInt64) -> Bool:
    if seq == 0:
        if ms == 0:
            return False
        ms -= 1
        seq = UInt64.MAX
    else:
        seq -= 1
    return True


def _parse_interval(t: RESP3Token, missing_seq: UInt64, mut ms: UInt64, mut seq: UInt64, mut exclusive: Bool) -> Bool:
    """streamParseIntervalIDOrReply: `(id` exclusive (a strict id), else an id
    or `-` / `+`."""
    exclusive = t.length > 1 and t.ptr[0] == 40
    var r = parse_id(t.ptr + 1 if exclusive else t.ptr, t.length - 1 if exclusive else t.length,
                     missing_seq, exclusive, False)
    ms = r.ms
    seq = r.seq
    return r.ok


# ── XGROUP ──

def handle_xgroup(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                  keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises:
    var argc = end - i
    if argc < 2:
        writer.append_error_response("ERR wrong number of arguments for 'xgroup' command")
        return
    var sub = tokens[i + 1]
    var sp = sub.ptr
    var sl = sub.length
    var is_create = arg_eq(sp, sl, "create")
    var is_setid = arg_eq(sp, sl, "setid")
    # each subcommand's arity, as Redis checks it first
    if is_create or is_setid:
        if argc < 5:
            writer.append_error_response("ERR wrong number of arguments for 'xgroup|" + _lower(sub) + "' command")
            return
    elif arg_eq(sp, sl, "destroy"):
        if argc != 4:
            writer.append_error_response("ERR wrong number of arguments for 'xgroup|destroy' command")
            return
    elif arg_eq(sp, sl, "createconsumer") or arg_eq(sp, sl, "delconsumer"):
        if argc != 5:
            writer.append_error_response("ERR wrong number of arguments for 'xgroup|" + _lower(sub) + "' command")
            return
    elif arg_eq(sp, sl, "help"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'xgroup|help' command")
            return
        _xgroup_help(writer)
        return
    else:
        writer.append_error_response("ERR unknown subcommand '" + _text(sub) + "'. Try XGROUP HELP.")
        return

    var mkstream = False
    var entries_read = SCG_INVALID_ENTRIES_READ
    var j = i + 5
    while j < end:
        var o = tokens[j]
        if is_create and arg_eq(o.ptr, o.length, "mkstream"):
            mkstream = True
            j += 1
        elif (is_create or is_setid) and arg_eq(o.ptr, o.length, "entriesread") and j + 1 < end:
            var v = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not v.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return
            if v.value < 0 and v.value != SCG_INVALID_ENTRIES_READ:
                writer.append_error_response("ERR value for ENTRIESREAD must be positive or -1")
                return
            entries_read = v.value
            j += 2
        else:
            writer.append_error_response("ERR unknown subcommand or wrong number of arguments for '" + _text(sub)
                                         + "'. Try XGROUP HELP.")
            return

    var kt = tokens[i + 2]
    var gt = tokens[i + 3]
    var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    var sd = null_ptr[StreamData, MutUntrackedOrigin]()
    if not val.is_none():
        if val.type.value != ValueType.STREAM:
            writer.append_error_response(_E_WRONGTYPE)
            return
        sd = val.as_hash().unsafe_bitcast[StreamData]()
    var g = -1
    if not mkstream:
        if is_null(sd):
            writer.append_error_response(_E_NOKEY_XGROUP)
            return
        g = sd[].group_index(gt.ptr, gt.length)
        if g < 0 and not is_create and not arg_eq(sp, sl, "destroy"):
            writer.append_error_response("NOGROUP No such consumer group '" + _text(gt) + "' for key name '"
                                         + _text(kt) + "'")
            return
    elif is_not_null(sd):
        g = sd[].group_index(gt.ptr, gt.length)

    if is_create:
        if argc > 8:
            writer.append_error_response("ERR unknown subcommand or wrong number of arguments for '" + _text(sub)
                                         + "'. Try XGROUP HELP.")
            return
        var idt = tokens[i + 4]
        var ms = UInt64(0)
        var seq = UInt64(0)
        if idt.length == 1 and idt.ptr[0] == 36:          # $
            if is_not_null(sd):
                ms = sd[].last_id_ms
                seq = sd[].last_id_seq
        else:
            var r = parse_id(idt.ptr, idt.length, 0, True, False)
            if not r.ok:
                writer.append_error_response(_E_BAD_ID)
                return
            ms = r.ms
            seq = r.seq
        if is_null(sd):
            var outcome = STREAM_OK
            sd = get_or_create_stream(keyspace, GenericValue.borrow(kt.ptr, kt.length), False, outcome)
            _log(wal, 45, kt, encode_meta_rec(sd))      # the empty stream exists, durably
        if g >= 0:
            writer.append_error_response("BUSYGROUP Consumer Group name already exists")
            return
        if entries_read != SCG_INVALID_ENTRIES_READ and entries_read > sd[].entries_added:
            entries_read = sd[].entries_added
        sd[].groups.append(StreamGroup(_name_bytes(gt), ms, seq, entries_read))
        _log(wal, 38, kt, encode_group_rec(sd[].groups[len(sd[].groups) - 1]))
        writer.append_ok_response()
    elif is_setid:
        if argc != 5 and argc != 7:
            writer.append_error_response("ERR unknown subcommand or wrong number of arguments for '" + _text(sub)
                                         + "'. Try XGROUP HELP.")
            return
        var idt = tokens[i + 4]
        var ms = sd[].last_id_ms
        var seq = sd[].last_id_seq
        if not (idt.length == 1 and idt.ptr[0] == 36):
            var r = parse_id(idt.ptr, idt.length, 0, False, False)
            if not r.ok:
                writer.append_error_response(_E_BAD_ID)
                return
            ms = r.ms
            seq = r.seq
        if entries_read != SCG_INVALID_ENTRIES_READ and entries_read > sd[].entries_added:
            entries_read = sd[].entries_added
        sd[].groups[g].last_ms = ms
        sd[].groups[g].last_seq = seq
        sd[].groups[g].entries_read = entries_read
        _log(wal, 39, kt, encode_group_rec(sd[].groups[g]))
        writer.append_ok_response()
    elif arg_eq(sp, sl, "destroy"):
        if g < 0:
            writer.append_int_response(0)
            return
        _ = sd[].groups.pop(g)
        _log(wal, 40, kt, encode_group_name_rec(gt.ptr, gt.length))
        writer.append_int_response(1)
    elif arg_eq(sp, sl, "createconsumer"):
        var ct = tokens[i + 4]
        if sd[].groups[g].consumer_index(ct.ptr, ct.length) >= 0:
            writer.append_int_response(0)
            return
        var now = _now_ms()
        var ci = sd[].groups[g].add_consumer(ct.ptr, ct.length, now)
        _settle_consumer(sd, g, ci, True, wal, kt)
        writer.append_int_response(1)
    else:   # delconsumer
        var ct = tokens[i + 4]
        var ci = sd[].groups[g].consumer_index(ct.ptr, ct.length)
        if ci < 0:
            writer.append_int_response(0)
            return
        var pending = sd[].groups[g].delete_consumer(ci)
        _log(wal, 42, kt, encode_delconsumer_rec(gt.ptr, gt.length, ct.ptr, ct.length))
        writer.append_int_response(Int64(pending))


def _lower(t: RESP3Token) -> String:
    return _text(t).lower()


def _xgroup_help(mut writer: ResponseWriter):
    var lines = List[String]()
    lines.append("XGROUP <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
    lines.append("CREATE <key> <groupname> <id|$> [option]")
    lines.append("    Create a new consumer group. Options are:")
    lines.append("    * MKSTREAM")
    lines.append("      Create the empty stream if it does not exist.")
    lines.append("    * ENTRIESREAD entries_read")
    lines.append("      Set the group's entries_read counter (internal use).")
    lines.append("CREATECONSUMER <key> <groupname> <consumer>")
    lines.append("    Create a new consumer in the specified group.")
    lines.append("DELCONSUMER <key> <groupname> <consumer>")
    lines.append("    Remove the specified consumer.")
    lines.append("DESTROY <key> <groupname>")
    lines.append("    Remove the specified group.")
    lines.append("SETID <key> <groupname> <id|$> [ENTRIESREAD entries_read]")
    lines.append("    Set the current group ID and entries_read counter.")
    lines.append("HELP")
    lines.append("    Print this help.")
    writer.append_array_header(len(lines))
    for k in range(len(lines)):
        writer.append_status_response(lines[k])


# ── XREADGROUP ──

struct XReadGroupArgs(Copyable, Movable):
    var ok: Bool
    var count: Int            # 0 = no limit
    var block_ms: Int64       # -1 = no BLOCK; else a deadline in unix ms (0 = forever)
    var noack: Bool
    var group_at: Int         # token of the group name
    var streams_at: Int       # first key token
    var nstreams: Int

    def __init__(out self):
        self.ok = False
        self.count = 0
        self.block_ms = -1
        self.noack = False
        self.group_at = -1
        self.streams_at = -1
        self.nstreams = 0


def parse_xreadgroup(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int,
                     mut writer: ResponseWriter, now_ms: Int64) -> XReadGroupArgs:
    """XREADGROUP's options, as Redis parses them; on an error it has
    replied and `ok` is False."""
    var a = XReadGroupArgs()
    if end - i < 7:   # Redis's arity, checked before any option
        writer.append_error_response("ERR wrong number of arguments for 'xreadgroup' command")
        return a^
    var j = i + 1
    while j < end:
        var o = tokens[j]
        var more = end - j - 1
        if arg_eq(o.ptr, o.length, "block") and more > 0:
            var v = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not v.ok:
                writer.append_error_response("ERR timeout is not an integer or out of range")
                return a^
            if v.value < 0:
                writer.append_error_response("ERR timeout is negative")
                return a^
            if v.value > 0 and v.value > Int64(9223372036854775807) - now_ms:
                writer.append_error_response("ERR timeout is out of range")
                return a^
            a.block_ms = now_ms + v.value if v.value > 0 else 0
            j += 2
        elif arg_eq(o.ptr, o.length, "count") and more > 0:
            var v = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not v.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return a^
            a.count = Int(v.value) if v.value > 0 else 0
            j += 2
        elif arg_eq(o.ptr, o.length, "streams") and more > 0:
            var n = end - (j + 1)
            if n % 2 != 0:
                writer.append_error_response("ERR Unbalanced 'xreadgroup' list of streams: for each stream key an ID "
                                             + "or '>' must be specified.")
                return a^
            a.streams_at = j + 1
            a.nstreams = n // 2
            break
        elif arg_eq(o.ptr, o.length, "group") and more >= 2:
            a.group_at = j + 1
            j += 3
        elif arg_eq(o.ptr, o.length, "noack"):
            a.noack = True
            j += 1
        else:
            writer.append_error_response("ERR syntax error")
            return a^
    if a.streams_at < 0:
        writer.append_error_response("ERR syntax error")
        return a^
    if a.group_at < 0:
        writer.append_error_response("ERR Missing GROUP option for XREADGROUP")
        return a^
    a.ok = True
    return a^


def handle_xreadgroup(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int,
                      mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                      wal: Pointer[WAL, MutUntrackedOrigin], can_block: Bool, mut deadline_ms: Int64,
                      mut keys_at: Int, mut nkeys: Int, mut group_at: Int) raises -> Int:
    """XREADGROUP GROUP g c [COUNT n] [BLOCK ms] [NOACK] STREAMS key... id...

    Returns XRG_BLOCK (nothing written; `deadline_ms` set, 0 = forever) when
    it found nothing to serve, BLOCK was given and the caller can park the
    connection; then the engine runs it again once a stream has new entries
    for the group, the stream or group is gone, or the deadline passes."""
    var now = _now_ms()
    var a = parse_xreadgroup(tokens, i, end, writer, now)
    if not a.ok:
        return XRG_DONE
    var gtok = tokens[a.group_at]
    var ctok = tokens[a.group_at + 1]
    var n = a.nstreams
    var kind = List[Int]()        # 0 new (">"), 1 history
    var ids_ms = List[UInt64]()
    var ids_seq = List[UInt64]()
    for s in range(n):
        var kt = tokens[a.streams_at + s]
        var it = tokens[a.streams_at + n + s]
        var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
        if not val.is_none() and val.type.value != ValueType.STREAM:
            writer.append_error_response(_E_WRONGTYPE)
            return XRG_DONE
        var g = -1
        if not val.is_none():
            g = val.as_hash().unsafe_bitcast[StreamData]()[].group_index(gtok.ptr, gtok.length)
        if g < 0:
            writer.append_error_response("NOGROUP No such key '" + _text(kt) + "' or consumer group '" + _text(gtok)
                                         + "' in XREADGROUP with GROUP option")
            return XRG_DONE
        if it.length == 1 and it.ptr[0] == 36:          # $
            writer.append_error_response("ERR The $ ID is meaningless in the context of XREADGROUP: you want to read "
                                         + "the history of this consumer by specifying a proper ID, or use the > ID "
                                         + "to get new messages. The $ ID would just return an empty result set.")
            return XRG_DONE
        if it.length == 1 and it.ptr[0] == 43:          # +
            writer.append_error_response("ERR The + ID is meaningless in the context of XREADGROUP: you want to read "
                                         + "the history of this consumer by specifying a proper ID, or use the > ID "
                                         + "to get new messages. The + ID would just return an empty result set.")
            return XRG_DONE
        if it.length == 1 and it.ptr[0] == 62:          # >
            kind.append(0)
            ids_ms.append(0)
            ids_seq.append(0)
            continue
        var r = parse_id(it.ptr, it.length, 0, True, False)
        if not r.ok:
            writer.append_error_response(_E_BAD_ID)
            return XRG_DONE
        kind.append(1)
        ids_ms.append(r.ms)
        ids_seq.append(r.seq)

    # Each stream in turn, as Redis serves them: the consumer is created (and
    # seen); a history read is always served; a new read when the stream has
    # an entry past the group's last-delivered id, judged after the streams
    # before it were served, so a key named twice is served once. Serving
    # changes the group here; the reply is written once all are served.
    var served = List[_Served]()
    var included = List[Bool]()
    var nserve = 0
    for s in range(n):
        var kt = tokens[a.streams_at + s]
        var sd = get_stream(keyspace, GenericValue.borrow(kt.ptr, kt.length))
        var g = sd[].group_index(gtok.ptr, gtok.length)
        var ci = sd[].groups[g].consumer_index(ctok.ptr, ctok.length)
        var changed = False
        if ci < 0:
            ci = sd[].groups[g].add_consumer(ctok.ptr, ctok.length, now)
            changed = True
        sd[].groups[g].consumers[ci].seen_time = now
        var yes = kind[s] == 1
        if not yes:
            var ll = sd[].last_live()
            if ll >= 0:
                var e = sd[].entries[unsafe_offset=ll]
                yes = sid_lt(sd[].groups[g].last_ms, sd[].groups[g].last_seq, e.id_ms, e.id_seq)
        included.append(yes)
        if yes:
            nserve += 1
            if kind[s] == 1:
                if _serve_history(sd, g, ci, ids_ms[s], ids_seq[s], a.count, now, wal, kt, s, served):
                    changed = True
            elif _serve_new(sd, g, ci, a.count, a.noack, now, wal, kt, s, served):
                changed = True
        _settle_consumer(sd, g, ci, changed, wal, kt)

    if nserve == 0:
        if a.block_ms >= 0 and can_block:
            deadline_ms = a.block_ms
            keys_at = a.streams_at
            nkeys = n
            group_at = a.group_at
            return XRG_BLOCK
        writer.append_null_array_response()
        return XRG_DONE

    if writer.proto == 3:
        writer.append_map_header(nserve)
    else:
        writer.append_array_header(nserve)
    var at = 0
    for s in range(n):
        if not included[s]:
            continue
        var kt = tokens[a.streams_at + s]
        if writer.proto != 3:
            writer.append_array_header(2)
        writer.append_bulk_string_response(kt.ptr, kt.length)
        var sd = get_stream(keyspace, GenericValue.borrow(kt.ptr, kt.length))
        var cnt = 0
        while at + cnt < len(served) and served[at + cnt].stream == s:
            cnt += 1
        writer.append_array_header(cnt)
        for k in range(at, at + cnt):
            var r = served[k]
            if r.entry >= 0:
                write_entry_to_response(sd[].entries[unsafe_offset=r.entry], writer)
            else:                                   # deleted since it was delivered
                writer.append_array_header(2)
                write_id(writer, r.ms, r.seq)
                writer.append_null_array_response()
        at += cnt
    return XRG_DONE


@fieldwise_init
struct _Served(Copyable, Movable, ImplicitlyCopyable):
    """One entry XREADGROUP serves: which of its streams, the id, and the
    entry's index in that stream (-1: deleted, replied [id, nil])."""
    var stream: Int
    var ms: UInt64
    var seq: UInt64
    var entry: Int


def _settle_consumer(sd: Pointer[StreamData, MutUntrackedOrigin], g: Int, ci: Int, changed: Bool,
                     wal: Pointer[WAL, MutUntrackedOrigin], kt: RESP3Token):
    """Log the consumer (record 41) when the command created or changed it,
    or when its seen time has moved a second past the one last logged: a
    consumer that only polls must not look idle since its last delivery
    after a restart or on a replica (a reaper keyed on idle would delete
    it, and its pending entries with it), and a poll loop must not write a
    record per poll."""
    ref c = sd[].groups[g].consumers[ci]
    if changed or c.seen_time - c.logged_seen >= 1000:
        c.logged_seen = c.seen_time
        _log(wal, 41, kt, encode_consumer_rec(sd[].groups[g].name, sd[].groups[g].consumers[ci]))


def _serve_new(sd: Pointer[StreamData, MutUntrackedOrigin], g: Int, ci: Int, count: Int, noack: Bool, now: Int64,
               wal: Pointer[WAL, MutUntrackedOrigin], kt: RESP3Token, stream: Int,
               mut served: List[_Served]) -> Bool:
    """Deliver the entries past group g's last-delivered id (up to `count`)
    to consumer ci, as Redis's streamReplyWithRange with a group. True when
    the consumer changed (it became active)."""
    var start = sd[].lower_bound(sd[].groups[g].last_ms, sd[].groups[g].last_seq)
    var picks = List[Int]()
    for k in range(start, sd[].count):
        var e = sd[].entries[unsafe_offset=k]
        if e.deleted:
            continue
        if not sid_lt(sd[].groups[g].last_ms, sd[].groups[g].last_seq, e.id_ms, e.id_seq):
            continue
        picks.append(k)
        if count > 0 and len(picks) >= count:
            break
    var cid = sd[].groups[g].consumers[ci].id
    for p in range(len(picks)):
        var e = sd[].entries[unsafe_offset=picks[p]]
        sd[].advance_group(g, e.id_ms, e.id_seq)
        served.append(_Served(stream, e.id_ms, e.id_seq, picks[p]))
        if not noack:
            sd[].groups[g].set_nack(e.id_ms, e.id_seq, cid, now, 1)
            sd[].groups[g].consumers[ci].active_time = now
            _log(wal, 43, kt, encode_nack_rec(sd[].groups[g].name, sd[].groups[g].consumers[ci].name,
                                              e.id_ms, e.id_seq, now, 1))
    if len(picks) > 0:
        _log(wal, 39, kt, encode_group_rec(sd[].groups[g]))
    return len(picks) > 0 and not noack


def _serve_history(sd: Pointer[StreamData, MutUntrackedOrigin], g: Int, ci: Int, after_ms: UInt64, after_seq: UInt64,
                   count: Int, now: Int64, wal: Pointer[WAL, MutUntrackedOrigin], kt: RESP3Token, stream: Int,
                   mut served: List[_Served]) -> Bool:
    """The consumer's own pending entries after the given id: an entry still
    in the stream is delivered again (its count and time move on), a deleted
    one is [id, nil]. True when something was delivered again."""
    var ms = after_ms
    var seq = after_seq
    var picks = List[Int]()
    if _incr(ms, seq):
        var cid = sd[].groups[g].consumers[ci].id
        var k = sd[].groups[g].pel_lower_bound(ms, seq)
        while k < len(sd[].groups[g].pel):
            if sd[].groups[g].pel[k].consumer == cid:
                picks.append(k)
                if count > 0 and len(picks) >= count:
                    break
            k += 1
    var again = False
    for p in range(len(picks)):
        var nk = sd[].groups[g].pel[picks[p]]
        var ei = sd[].find_live(nk.ms, nk.seq)
        served.append(_Served(stream, nk.ms, nk.seq, ei))
        if ei >= 0:
            if sd[].groups[g].pel[picks[p]].delivery_count < Int64(9223372036854775807):
                sd[].groups[g].pel[picks[p]].delivery_count += 1
            sd[].groups[g].pel[picks[p]].delivery_time = now
            var dn = sd[].groups[g].pel[picks[p]]
            _log(wal, 43, kt, encode_nack_rec(sd[].groups[g].name, sd[].groups[g].consumers[ci].name,
                                              dn.ms, dn.seq, dn.delivery_time, dn.delivery_count))
            again = True
    return again


# ── XACK ──

def handle_xack(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises:
    if end - i < 4:
        writer.append_error_response("ERR wrong number of arguments for 'xack' command")
        return
    var kt = tokens[i + 1]
    var gt = tokens[i + 2]
    var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    if not val.is_none() and val.type.value != ValueType.STREAM:
        writer.append_error_response(_E_WRONGTYPE)
        return
    if val.is_none():
        writer.append_int_response(0)
        return
    var sd = val.as_hash().unsafe_bitcast[StreamData]()
    var g = sd[].group_index(gt.ptr, gt.length)
    if g < 0:
        writer.append_int_response(0)
        return
    var ms = List[UInt64]()
    var seq = List[UInt64]()
    for j in range(i + 3, end):
        var r = parse_id(tokens[j].ptr, tokens[j].length, 0, True, False)
        if not r.ok:
            writer.append_error_response(_E_BAD_ID)
            return
        ms.append(r.ms)
        seq.append(r.seq)
    var acked = 0
    for k in range(len(ms)):
        var pk = sd[].groups[g].pel_find(ms[k], seq[k])
        if pk >= 0:
            sd[].groups[g].remove_nack_at(pk)
            _log(wal, 44, kt, encode_pel_del_rec(sd[].groups[g].name, ms[k], seq[k]))
            acked += 1
    sd[].groups[g].pel_tidy()
    writer.append_int_response(Int64(acked))


# ── XDELEX / XACKDEL (Redis 8.2) ──

@fieldwise_init
struct AckDelArgs(Copyable, Movable, ImplicitlyCopyable):
    var ok: Bool
    var strategy: Int
    var ids_at: Int
    var numids: Int


def _parse_ackdel(tokens: Pointer[RESP3Token, MutUntrackedOrigin], start: Int, end: Int,
                  mut writer: ResponseWriter) -> AckDelArgs:
    """[KEEPREF|DELREF|ACKED] IDS numids id..., as Redis's
    streamParseAckDelArgsOrReply: one strategy at most, options after the
    ids are parsed too, KEEPREF by default."""
    var a = AckDelArgs(False, DEL_NONE, -1, 0)
    var j = start
    while j < end:
        var t = tokens[j]
        if a.strategy == DEL_NONE and arg_eq(t.ptr, t.length, "keepref"):
            a.strategy = DEL_KEEPREF
            j += 1
        elif a.strategy == DEL_NONE and arg_eq(t.ptr, t.length, "delref"):
            a.strategy = DEL_DELREF
            j += 1
        elif a.strategy == DEL_NONE and arg_eq(t.ptr, t.length, "acked"):
            a.strategy = DEL_ACKED
            j += 1
        elif arg_eq(t.ptr, t.length, "ids") and j + 1 < end:
            var v = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not v.ok or v.value < 1:
                writer.append_error_response("ERR Number of IDs must be a positive integer")
                return a
            if v.value > Int64(end - j - 2):
                writer.append_error_response("ERR The `numids` parameter must match the number of arguments")
                return a
            a.ids_at = j + 2
            a.numids = Int(v.value)
            j = a.ids_at + a.numids
        else:
            writer.append_error_response("ERR syntax error")
            return a
    if a.ids_at < 0:
        writer.append_error_response("ERR IDS option is required")
        return a
    if a.strategy == DEL_NONE:
        a.strategy = DEL_KEEPREF
    a.ok = True
    return a


def _parse_ids(tokens: Pointer[RESP3Token, MutUntrackedOrigin], at: Int, n: Int, mut ms: List[UInt64],
               mut seq: List[UInt64], mut writer: ResponseWriter) -> Bool:
    """Every id strictly, before anything changes (all or nothing)."""
    for k in range(n):
        var r = parse_id(tokens[at + k].ptr, tokens[at + k].length, 0, True, False)
        if not r.ok:
            writer.append_error_response(_E_BAD_ID)
            return False
        ms.append(r.ms)
        seq.append(r.seq)
    return True


def _minus_ones(mut writer: ResponseWriter, n: Int):
    writer.append_array_header(n)
    for _ in range(n):
        writer.append_int_response(-1)


def handle_xdelex(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                  keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises:
    """XDELEX key [KEEPREF|DELREF|ACKED] IDS numids id... → per id: 1 deleted,
    -1 no such entry, 2 kept because a group still references it (ACKED)."""
    if end - i < 5:
        writer.append_error_response("ERR wrong number of arguments for 'xdelex' command")
        return
    var kt = tokens[i + 1]
    var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    if not val.is_none() and val.type.value != ValueType.STREAM:
        writer.append_error_response(_E_WRONGTYPE)
        return
    var a = _parse_ackdel(tokens, i + 2, end, writer)
    if not a.ok:
        return
    if val.is_none():
        _minus_ones(writer, a.numids)
        return
    var ms = List[UInt64]()
    var seq = List[UInt64]()
    if not _parse_ids(tokens, a.ids_at, a.numids, ms, seq, writer):
        return
    var sd = val.as_hash().unsafe_bitcast[StreamData]()
    var deleted = 0
    writer.append_array_header(a.numids)
    for k in range(a.numids):
        var res = Int64(-1)
        var can_delete = True
        if a.strategy == DEL_ACKED:
            can_delete = not sd[].entry_referenced(ms[k], seq[k])
        elif a.strategy == DEL_DELREF:
            drop_group_refs(sd, ms[k], seq[k], wal, kt.ptr, kt.length)
        if not can_delete:
            res = 2
        elif stream_delete_entry(sd, ms[k], seq[k], wal, kt.ptr, kt.length):
            deleted += 1
            res = 1
        writer.append_int_response(res)
    if deleted > 0:
        sd[].compact()
        _log(wal, 45, kt, encode_meta_rec(sd))


def handle_xackdel(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                   keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises:
    """XACKDEL key group [KEEPREF|DELREF|ACKED] IDS numids id... → per id: the
    entry was pending in the group and is now acknowledged, and 1 deleted
    (even if it was already gone), 2 kept because another group still
    references it (ACKED); -1 it was not pending in the group."""
    if end - i < 6:
        writer.append_error_response("ERR wrong number of arguments for 'xackdel' command")
        return
    var kt = tokens[i + 1]
    var gt = tokens[i + 2]
    var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    if not val.is_none() and val.type.value != ValueType.STREAM:
        writer.append_error_response(_E_WRONGTYPE)
        return
    var a = _parse_ackdel(tokens, i + 3, end, writer)
    if not a.ok:
        return
    var g = -1
    var sd = null_ptr[StreamData, MutUntrackedOrigin]()
    if not val.is_none():
        sd = val.as_hash().unsafe_bitcast[StreamData]()
        g = sd[].group_index(gt.ptr, gt.length)
    if g < 0:
        _minus_ones(writer, a.numids)
        return
    var ms = List[UInt64]()
    var seq = List[UInt64]()
    if not _parse_ids(tokens, a.ids_at, a.numids, ms, seq, writer):
        return
    var deleted = 0
    writer.append_array_header(a.numids)
    for k in range(a.numids):
        var res = Int64(-1)
        var pk = sd[].groups[g].pel_find(ms[k], seq[k])
        if pk >= 0:
            sd[].groups[g].remove_nack_at(pk)
            _log(wal, 44, kt, encode_pel_del_rec(sd[].groups[g].name, ms[k], seq[k]))
            var can_delete = True
            if a.strategy == DEL_ACKED:
                can_delete = not sd[].entry_referenced(ms[k], seq[k])
            elif a.strategy == DEL_DELREF:
                drop_group_refs(sd, ms[k], seq[k], wal, kt.ptr, kt.length)
            if can_delete and stream_delete_entry(sd, ms[k], seq[k], wal, kt.ptr, kt.length):
                deleted += 1
            res = 1 if can_delete else 2
        writer.append_int_response(res)
    sd[].groups[g].pel_tidy()
    if deleted > 0:
        sd[].compact()
        _log(wal, 45, kt, encode_meta_rec(sd))


# ── XPENDING ──

def handle_xpending(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises:
    var argc = end - i
    if argc < 3:
        writer.append_error_response("ERR wrong number of arguments for 'xpending' command")
        return
    if argc != 3 and (argc < 6 or argc > 9):
        writer.append_error_response("ERR syntax error")
        return
    var kt = tokens[i + 1]
    var gt = tokens[i + 2]
    var minidle = Int64(0)
    var count = Int64(0)
    var s_ms = UInt64(0)
    var s_seq = UInt64(0)
    var e_ms = UInt64(0)
    var e_seq = UInt64(0)
    var consumer_at = -1
    if argc >= 6:
        var at = i + 3
        if arg_eq(tokens[at].ptr, tokens[at].length, "idle"):
            var v = parse_int64_strict(tokens[at + 1].ptr, tokens[at + 1].length)
            if not v.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return
            minidle = v.value
            if argc < 8:
                writer.append_error_response("ERR syntax error")
                return
            at += 2
        var cv = parse_int64_strict(tokens[at + 2].ptr, tokens[at + 2].length)
        if not cv.ok:
            writer.append_error_response("ERR value is not an integer or out of range")
            return
        count = cv.value if cv.value > 0 else 0
        var ex = False
        if not _parse_interval(tokens[at], 0, s_ms, s_seq, ex):
            writer.append_error_response(_E_BAD_ID)
            return
        if ex and not _incr(s_ms, s_seq):
            writer.append_error_response("ERR invalid start ID for the interval")
            return
        if not _parse_interval(tokens[at + 1], UInt64.MAX, e_ms, e_seq, ex):
            writer.append_error_response(_E_BAD_ID)
            return
        if ex and not _decr(e_ms, e_seq):
            writer.append_error_response("ERR invalid end ID for the interval")
            return
        if at + 3 < end:
            consumer_at = at + 3
    var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    if not val.is_none() and val.type.value != ValueType.STREAM:
        writer.append_error_response(_E_WRONGTYPE)
        return
    var g = -1
    var sd = null_ptr[StreamData, MutUntrackedOrigin]()
    if not val.is_none():
        sd = val.as_hash().unsafe_bitcast[StreamData]()
        g = sd[].group_index(gt.ptr, gt.length)
    if g < 0:
        writer.append_error_response("NOGROUP No such key '" + _text(kt) + "' or consumer group '" + _text(gt) + "'")
        return
    ref grp = sd[].groups[g]
    if argc == 3:
        writer.append_array_header(4)
        writer.append_int_response(Int64(grp.pel_live()))
        if grp.pel_live() == 0:
            writer.append_null_response()
            writer.append_null_response()
            writer.append_null_array_response()
            return
        var first = grp.pel_next_live(0)
        var last = grp.pel_last_live()
        write_id(writer, grp.pel[first].ms, grp.pel[first].seq)
        write_id(writer, grp.pel[last].ms, grp.pel[last].seq)
        var with_pending = 0
        for k in range(grp.ncons()):
            if grp.consumers[grp.by_name[k]].pending > 0:
                with_pending += 1
        writer.append_array_header(with_pending)
        for k in range(grp.ncons()):
            ref c = grp.consumers[grp.by_name[k]]
            if c.pending > 0:
                writer.append_array_header(2)
                _write_name(writer, c.name)
                var cnt = String(c.pending)
                writer.append_bulk_string_response(cnt.unsafe_ptr(), cnt.byte_length())
        return
    var only = -1
    if consumer_at >= 0:
        var cidx = grp.consumer_index(tokens[consumer_at].ptr, tokens[consumer_at].length)
        if cidx < 0:
            writer.append_array_header(0)
            return
        only = grp.consumers[cidx].id
    var now = _now_ms()
    var rows = List[Int]()
    var k = grp.pel_lower_bound(s_ms, s_seq)
    while count > 0 and k < len(grp.pel):
        var nk = grp.pel[k]
        if sid_lt(e_ms, e_seq, nk.ms, nk.seq):
            break
        k += 1
        if nk.consumer == PEL_DEAD or (only >= 0 and nk.consumer != only):
            continue
        if minidle > 0 and now - nk.delivery_time < minidle:
            continue
        rows.append(k - 1)
        count -= 1
    writer.append_array_header(len(rows))
    for r in range(len(rows)):
        var nk = grp.pel[rows[r]]
        writer.append_array_header(4)
        write_id(writer, nk.ms, nk.seq)
        var oi = grp.consumer_by_id(nk.consumer)
        if oi >= 0:
            _write_name(writer, grp.consumers[oi].name)
        else:
            writer.append_bulk_string_response("".unsafe_ptr(), 0)
        var idle = now - nk.delivery_time
        writer.append_int_response(idle if idle > 0 else 0)
        writer.append_int_response(nk.delivery_count)


# ── XCLAIM / XAUTOCLAIM ──

def _claim(sd: Pointer[StreamData, MutUntrackedOrigin], g: Int, pk: Int, ci: Int, dtime: Int64,
           retrycount: Int64, justid: Bool):
    """Give pending entry pk to consumer ci, as XCLAIM and XAUTOCLAIM do."""
    var cid = sd[].groups[g].consumers[ci].id
    var old = sd[].groups[g].pel[pk].consumer
    if old != cid:
        var oi = sd[].groups[g].consumer_by_id(old)
        if oi >= 0:
            sd[].groups[g].consumers[oi].pending -= 1
        sd[].groups[g].consumers[ci].pending += 1
        sd[].groups[g].pel[pk].consumer = cid
    sd[].groups[g].pel[pk].delivery_time = dtime
    if retrycount >= 0:
        sd[].groups[g].pel[pk].delivery_count = retrycount
    elif not justid and sd[].groups[g].pel[pk].delivery_count < Int64(9223372036854775807):
        sd[].groups[g].pel[pk].delivery_count += 1


def handle_xclaim(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                  keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises:
    """XCLAIM key group consumer min-idle-time id [id ...] [IDLE ms] [TIME ms]
    [RETRYCOUNT n] [FORCE] [JUSTID] [LASTID id]"""
    if end - i < 6:
        writer.append_error_response("ERR wrong number of arguments for 'xclaim' command")
        return
    var kt = tokens[i + 1]
    var gt = tokens[i + 2]
    var ct = tokens[i + 3]
    var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    if not val.is_none() and val.type.value != ValueType.STREAM:
        writer.append_error_response(_E_WRONGTYPE)
        return
    var sd = null_ptr[StreamData, MutUntrackedOrigin]()
    var g = -1
    if not val.is_none():
        sd = val.as_hash().unsafe_bitcast[StreamData]()
        g = sd[].group_index(gt.ptr, gt.length)
    if g < 0:
        writer.append_error_response("NOGROUP No such key '" + _text(kt) + "' or consumer group '" + _text(gt) + "'")
        return
    var mi = parse_int64_strict(tokens[i + 4].ptr, tokens[i + 4].length)
    if not mi.ok:
        writer.append_error_response("ERR Invalid min-idle-time argument for XCLAIM")
        return
    var minidle = mi.value if mi.value > 0 else Int64(0)
    var ids_ms = List[UInt64]()
    var ids_seq = List[UInt64]()
    var j = i + 5
    while j < end:
        var r = parse_id(tokens[j].ptr, tokens[j].length, 0, True, False)
        if not r.ok:
            break
        ids_ms.append(r.ms)
        ids_seq.append(r.seq)
        j += 1
    var now = _now_ms()
    var force = False
    var justid = False
    var dtime = Int64(-1)
    var retrycount = Int64(-1)
    var last_ms = UInt64(0)
    var last_seq = UInt64(0)
    while j < end:
        var o = tokens[j]
        var more = end - 1 - j
        if arg_eq(o.ptr, o.length, "force"):
            force = True
        elif arg_eq(o.ptr, o.length, "justid"):
            justid = True
        elif arg_eq(o.ptr, o.length, "idle") and more > 0:
            j += 1
            var v = parse_int64_strict(tokens[j].ptr, tokens[j].length)
            if not v.ok:
                writer.append_error_response("ERR Invalid IDLE option argument for XCLAIM")
                return
            dtime = now - v.value
        elif arg_eq(o.ptr, o.length, "time") and more > 0:
            j += 1
            var v = parse_int64_strict(tokens[j].ptr, tokens[j].length)
            if not v.ok:
                writer.append_error_response("ERR Invalid TIME option argument for XCLAIM")
                return
            dtime = v.value
        elif arg_eq(o.ptr, o.length, "retrycount") and more > 0:
            j += 1
            var v = parse_int64_strict(tokens[j].ptr, tokens[j].length)
            if not v.ok:
                writer.append_error_response("ERR Invalid RETRYCOUNT option argument for XCLAIM")
                return
            retrycount = v.value
        elif arg_eq(o.ptr, o.length, "lastid") and more > 0:
            j += 1
            var r = parse_id(tokens[j].ptr, tokens[j].length, 0, True, False)
            if not r.ok:
                writer.append_error_response(_E_BAD_ID)
                return
            last_ms = r.ms
            last_seq = r.seq
        else:
            writer.append_error_response("ERR Unrecognized XCLAIM option '" + _text(o) + "'")
            return
        j += 1
    var moved_last = False
    if sid_lt(sd[].groups[g].last_ms, sd[].groups[g].last_seq, last_ms, last_seq):
        sd[].groups[g].last_ms = last_ms
        sd[].groups[g].last_seq = last_seq
        moved_last = True
    if dtime == -1 or dtime < 0 or dtime > now:
        dtime = now
    var ci = sd[].groups[g].consumer_index(ct.ptr, ct.length)
    var created = ci < 0
    if created:
        ci = sd[].groups[g].add_consumer(ct.ptr, ct.length, now)
    sd[].groups[g].consumers[ci].seen_time = now
    var claimed = List[Int]()      # entry index per reply row (-1 never)
    var claimed_ms = List[UInt64]()
    var claimed_seq = List[UInt64]()
    for k in range(len(ids_ms)):
        var pk = sd[].groups[g].pel_find(ids_ms[k], ids_seq[k])
        var ei = sd[].find_live(ids_ms[k], ids_seq[k])
        if ei < 0:
            if pk >= 0:                # the entry is gone: so is its pending entry
                sd[].groups[g].remove_nack_at(pk)
                _log(wal, 44, kt, encode_pel_del_rec(sd[].groups[g].name, ids_ms[k], ids_seq[k]))
            continue
        if force and pk < 0:
            sd[].groups[g].set_nack(ids_ms[k], ids_seq[k], 0, dtime, 0)   # unowned until claimed below
            pk = sd[].groups[g].pel_find(ids_ms[k], ids_seq[k])
        if pk < 0:
            continue
        var owned = sd[].groups[g].consumer_by_id(sd[].groups[g].pel[pk].consumer) >= 0
        if owned and minidle > 0 and now - sd[].groups[g].pel[pk].delivery_time < minidle:
            continue
        _claim(sd, g, pk, ci, dtime, retrycount, justid)
        sd[].groups[g].consumers[ci].active_time = now
        var nk = sd[].groups[g].pel[pk]
        _log(wal, 43, kt, encode_nack_rec(sd[].groups[g].name, sd[].groups[g].consumers[ci].name,
                                          nk.ms, nk.seq, nk.delivery_time, nk.delivery_count))
        claimed.append(ei)
        claimed_ms.append(ids_ms[k])
        claimed_seq.append(ids_seq[k])
    if moved_last:
        _log(wal, 39, kt, encode_group_rec(sd[].groups[g]))
    _settle_consumer(sd, g, ci, created or len(claimed) > 0, wal, kt)
    sd[].groups[g].pel_tidy()
    writer.append_array_header(len(claimed))
    for k in range(len(claimed)):
        if justid:
            write_id(writer, claimed_ms[k], claimed_seq[k])
        else:
            write_entry_to_response(sd[].entries[unsafe_offset=claimed[k]], writer)


def handle_xautoclaim(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                      keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises:
    """XAUTOCLAIM key group consumer min-idle-time start [COUNT n] [JUSTID]
    → [next cursor, claimed entries, ids of deleted entries it dropped]"""
    if end - i < 6:
        writer.append_error_response("ERR wrong number of arguments for 'xautoclaim' command")
        return
    var kt = tokens[i + 1]
    var gt = tokens[i + 2]
    var ct = tokens[i + 3]
    var mi = parse_int64_strict(tokens[i + 4].ptr, tokens[i + 4].length)
    if not mi.ok:
        writer.append_error_response("ERR Invalid min-idle-time argument for XAUTOCLAIM")
        return
    var minidle = mi.value if mi.value > 0 else Int64(0)
    var s_ms = UInt64(0)
    var s_seq = UInt64(0)
    var ex = False
    if not _parse_interval(tokens[i + 5], 0, s_ms, s_seq, ex):
        writer.append_error_response(_E_BAD_ID)
        return
    if ex and not _incr(s_ms, s_seq):
        writer.append_error_response("ERR invalid start ID for the interval")
        return
    var count = Int64(100)
    var justid = False
    var j = i + 6
    while j < end:
        var o = tokens[j]
        var more = end - 1 - j
        if arg_eq(o.ptr, o.length, "count") and more > 0:
            var v = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not v.ok or v.value < 1 or v.value > Int64(9223372036854775807) // 16:
                writer.append_error_response("ERR COUNT must be > 0")
                return
            count = v.value
            j += 1
        elif arg_eq(o.ptr, o.length, "justid"):
            justid = True
        else:
            writer.append_error_response("ERR syntax error")
            return
        j += 1
    var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    if not val.is_none() and val.type.value != ValueType.STREAM:
        writer.append_error_response(_E_WRONGTYPE)
        return
    var sd = null_ptr[StreamData, MutUntrackedOrigin]()
    var g = -1
    if not val.is_none():
        sd = val.as_hash().unsafe_bitcast[StreamData]()
        g = sd[].group_index(gt.ptr, gt.length)
    if g < 0:
        writer.append_error_response("NOGROUP No such key '" + _text(kt) + "' or consumer group '" + _text(gt) + "'")
        return
    var now = _now_ms()
    var ci = sd[].groups[g].consumer_index(ct.ptr, ct.length)
    var created = ci < 0
    if created:
        ci = sd[].groups[g].add_consumer(ct.ptr, ct.length, now)
    sd[].groups[g].consumers[ci].seen_time = now
    var attempts = count * 10
    var claimed = List[Int]()
    var claimed_ms = List[UInt64]()
    var claimed_seq = List[UInt64]()
    var deleted_ms = List[UInt64]()
    var deleted_seq = List[UInt64]()
    var k = sd[].groups[g].pel_next_live(sd[].groups[g].pel_lower_bound(s_ms, s_seq))
    while attempts > 0 and count > 0 and k < len(sd[].groups[g].pel):
        attempts -= 1
        var nk = sd[].groups[g].pel[k]
        var ei = sd[].find_live(nk.ms, nk.seq)
        if ei < 0:
            sd[].groups[g].remove_nack_at(k)
            _log(wal, 44, kt, encode_pel_del_rec(sd[].groups[g].name, nk.ms, nk.seq))
            deleted_ms.append(nk.ms)
            deleted_seq.append(nk.seq)
            count -= 1
            k = sd[].groups[g].pel_next_live(k + 1)
            continue
        var owned = sd[].groups[g].consumer_by_id(nk.consumer) >= 0
        if owned and minidle > 0 and now - nk.delivery_time < minidle:
            k = sd[].groups[g].pel_next_live(k + 1)
            continue
        _claim(sd, g, k, ci, now, -1, justid)
        sd[].groups[g].consumers[ci].active_time = now
        var nk2 = sd[].groups[g].pel[k]
        _log(wal, 43, kt, encode_nack_rec(sd[].groups[g].name, sd[].groups[g].consumers[ci].name,
                                          nk2.ms, nk2.seq, nk2.delivery_time, nk2.delivery_count))
        claimed.append(ei)
        claimed_ms.append(nk.ms)
        claimed_seq.append(nk.seq)
        count -= 1
        k = sd[].groups[g].pel_next_live(k + 1)
    writer.append_array_header(3)
    if k < len(sd[].groups[g].pel):
        write_id(writer, sd[].groups[g].pel[k].ms, sd[].groups[g].pel[k].seq)
    else:
        write_id(writer, 0, 0)
    _settle_consumer(sd, g, ci, created or len(claimed) > 0, wal, kt)
    sd[].groups[g].pel_tidy()
    writer.append_array_header(len(claimed))
    for r in range(len(claimed)):
        if justid:
            write_id(writer, claimed_ms[r], claimed_seq[r])
        else:
            write_entry_to_response(sd[].entries[unsafe_offset=claimed[r]], writer)
    writer.append_array_header(len(deleted_ms))
    for r in range(len(deleted_ms)):
        write_id(writer, deleted_ms[r], deleted_seq[r])


# ── XSETID ──

def handle_xsetid(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                  keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]) raises:
    """XSETID key last-id [ENTRIESADDED n] [MAXDELETEDID id]"""
    if end - i < 3:
        writer.append_error_response("ERR wrong number of arguments for 'xsetid' command")
        return
    var kt = tokens[i + 1]
    var r = parse_id(tokens[i + 2].ptr, tokens[i + 2].length, 0, True, False)
    if not r.ok:
        writer.append_error_response(_E_BAD_ID)
        return
    var entries_added = Int64(-1)
    var md_ms = UInt64(0)
    var md_seq = UInt64(0)
    var j = i + 3
    while j < end:
        var o = tokens[j]
        var more = end - 1 - j
        if arg_eq(o.ptr, o.length, "entriesadded") and more > 0:
            var v = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not v.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return
            if v.value < 0:
                writer.append_error_response("ERR entries_added must be positive")
                return
            entries_added = v.value
            j += 2
        elif arg_eq(o.ptr, o.length, "maxdeletedid") and more > 0:
            var m = parse_id(tokens[j + 1].ptr, tokens[j + 1].length, 0, True, False)
            if not m.ok:
                writer.append_error_response(_E_BAD_ID)
                return
            if sid_lt(r.ms, r.seq, m.ms, m.seq):
                writer.append_error_response("ERR The ID specified in XSETID is smaller than the provided "
                                             + "max_deleted_entry_id")
                return
            md_ms = m.ms
            md_seq = m.seq
            j += 2
        else:
            writer.append_error_response("ERR syntax error")
            return
    var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    if val.is_none():
        writer.append_error_response("ERR no such key")
        return
    if val.type.value != ValueType.STREAM:
        writer.append_error_response(_E_WRONGTYPE)
        return
    var sd = val.as_hash().unsafe_bitcast[StreamData]()
    if sid_lt(r.ms, r.seq, sd[].max_del_ms, sd[].max_del_seq):
        writer.append_error_response("ERR The ID specified in XSETID is smaller than current max_deleted_entry_id")
        return
    if sd[].alive > 0:
        var ll = sd[].last_live()
        var e = sd[].entries[unsafe_offset=ll]
        if sid_lt(r.ms, r.seq, e.id_ms, e.id_seq):
            writer.append_error_response("ERR The ID specified in XSETID is smaller than the target stream top item")
            return
        if entries_added != -1 and Int64(sd[].alive) > entries_added:
            writer.append_error_response("ERR The entries_added specified in XSETID is smaller than the target "
                                         + "stream length")
            return
    sd[].last_id_ms = r.ms
    sd[].last_id_seq = r.seq
    if entries_added != -1:
        sd[].entries_added = entries_added
    if not (md_ms == 0 and md_seq == 0):
        sd[].max_del_ms = md_ms
        sd[].max_del_seq = md_seq
    _log(wal, 45, kt, encode_meta_rec(sd))
    writer.append_ok_response()


# ── XINFO ──

def handle_xinfo(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                 keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) raises:
    """XINFO STREAM key [FULL [COUNT n]] | GROUPS key | CONSUMERS key group | HELP"""
    var argc = end - i
    if argc < 2:
        writer.append_error_response("ERR wrong number of arguments for 'xinfo' command")
        return
    var sub = tokens[i + 1]
    if arg_eq(sub.ptr, sub.length, "help"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'xinfo|help' command")
            return
        var lines = List[String]()
        lines.append("XINFO <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
        lines.append("CONSUMERS <key> <groupname>")
        lines.append("    Show consumers of <groupname>.")
        lines.append("GROUPS <key>")
        lines.append("    Show the stream consumer groups.")
        lines.append("STREAM <key> [FULL [COUNT <count>]")
        lines.append("    Show information about the stream.")
        lines.append("HELP")
        lines.append("    Print this help.")
        writer.append_array_header(len(lines))
        for k in range(len(lines)):
            writer.append_status_response(lines[k])
        return
    var is_stream = arg_eq(sub.ptr, sub.length, "stream")
    var is_groups = arg_eq(sub.ptr, sub.length, "groups")
    var is_consumers = arg_eq(sub.ptr, sub.length, "consumers")
    if not (is_stream or is_groups or is_consumers):
        writer.append_error_response("ERR unknown subcommand '" + _text(sub) + "'. Try XINFO HELP.")
        return
    if (is_stream and argc < 3) or (is_groups and argc != 3) or (is_consumers and argc != 4):
        writer.append_error_response("ERR wrong number of arguments for 'xinfo|" + _lower(sub) + "' command")
        return
    var kt = tokens[i + 2]
    var val = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
    if val.is_none():
        writer.append_error_response("ERR no such key")
        return
    if val.type.value != ValueType.STREAM:
        writer.append_error_response(_E_WRONGTYPE)
        return
    var sd = val.as_hash().unsafe_bitcast[StreamData]()
    var now = _now_ms()
    if is_consumers:
        var gt = tokens[i + 3]
        var g = sd[].group_index(gt.ptr, gt.length)
        if g < 0:
            writer.append_error_response("NOGROUP No such consumer group '" + _text(gt) + "' for key name '"
                                         + _text(kt) + "'")
            return
        ref grp = sd[].groups[g]
        writer.append_array_header(grp.ncons())
        for k in range(grp.ncons()):
            ref c = grp.consumers[grp.by_name[k]]
            writer.append_map_header(4)
            _kv_name(writer, "name", c.name)
            _kv_int(writer, "pending", Int64(c.pending))
            var idle = now - c.seen_time
            _kv_int(writer, "idle", idle if idle > 0 else 0)
            var inactive = Int64(-1)
            if c.active_time != -1:
                inactive = now - c.active_time
                if inactive < 0:
                    inactive = 0
            _kv_int(writer, "inactive", inactive)
        return
    if is_groups:
        writer.append_array_header(len(sd[].groups))
        for g in range(len(sd[].groups)):
            writer.append_map_header(6)
            _kv_name(writer, "name", sd[].groups[g].name)
            _kv_int(writer, "consumers", Int64(sd[].groups[g].ncons()))
            _kv_int(writer, "pending", Int64(sd[].groups[g].pel_live()))
            _key_str(writer, "last-delivered-id")
            write_id(writer, sd[].groups[g].last_ms, sd[].groups[g].last_seq)
            _key_str(writer, "entries-read")
            if sd[].groups[g].entries_read != SCG_INVALID_ENTRIES_READ:
                writer.append_int_response(sd[].groups[g].entries_read)
            else:
                writer.append_null_response()
            _key_str(writer, "lag")
            var valid = False
            var lag = sd[].group_lag(g, valid)
            if valid:
                writer.append_int_response(lag)
            else:
                writer.append_null_response()
        return
    # STREAM [FULL [COUNT n]]
    var full = False
    var full_count = 10
    if argc > 3:
        if not arg_eq(tokens[i + 3].ptr, tokens[i + 3].length, "full"):
            writer.append_error_response("ERR unknown subcommand or wrong number of arguments for '" + _text(sub)
                                         + "'. Try XINFO HELP.")
            return
        full = True
        if argc == 6 and arg_eq(tokens[i + 4].ptr, tokens[i + 4].length, "count"):
            var v = parse_int64_strict(tokens[i + 5].ptr, tokens[i + 5].length)
            if not v.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return
            full_count = Int(v.value) if v.value > 0 else 0
        elif argc != 4:
            writer.append_error_response("ERR unknown subcommand or wrong number of arguments for '" + _text(sub)
                                         + "'. Try XINFO HELP.")
            return
    var f_ms = UInt64(0)
    var f_seq = UInt64(0)
    sd[].first_id(f_ms, f_seq)
    writer.append_map_header(7 if full else 8)
    _kv_int(writer, "length", Int64(sd[].alive))
    _key_str(writer, "last-generated-id")
    write_id(writer, sd[].last_id_ms, sd[].last_id_seq)
    _key_str(writer, "max-deleted-entry-id")
    write_id(writer, sd[].max_del_ms, sd[].max_del_seq)
    _kv_int(writer, "entries-added", sd[].entries_added)
    _key_str(writer, "recorded-first-entry-id")
    write_id(writer, f_ms, f_seq)
    if not full:
        _kv_int(writer, "groups", Int64(len(sd[].groups)))
        _key_str(writer, "first-entry")
        var fl = sd[].first_live()
        if fl >= 0:
            write_entry_to_response(sd[].entries[unsafe_offset=fl], writer)
        else:
            writer.append_null_response()
        _key_str(writer, "last-entry")
        var ll = sd[].last_live()
        if ll >= 0:
            write_entry_to_response(sd[].entries[unsafe_offset=ll], writer)
        else:
            writer.append_null_response()
        return
    _key_str(writer, "entries")
    var shown = 0
    for k in range(sd[].count):
        if not sd[].entries[unsafe_offset=k].deleted:
            shown += 1
            if full_count > 0 and shown >= full_count:
                break
    writer.append_array_header(shown)
    var w = 0
    for k in range(sd[].count):
        if w >= shown:
            break
        if sd[].entries[unsafe_offset=k].deleted:
            continue
        write_entry_to_response(sd[].entries[unsafe_offset=k], writer)
        w += 1
    _key_str(writer, "groups")
    writer.append_array_header(len(sd[].groups))
    for g in range(len(sd[].groups)):
        ref grp = sd[].groups[g]
        writer.append_map_header(7)
        _kv_name(writer, "name", grp.name)
        _key_str(writer, "last-delivered-id")
        write_id(writer, grp.last_ms, grp.last_seq)
        _key_str(writer, "entries-read")
        if grp.entries_read != SCG_INVALID_ENTRIES_READ:
            writer.append_int_response(grp.entries_read)
        else:
            writer.append_null_response()
        _key_str(writer, "lag")
        var valid = False
        var lag = sd[].group_lag(g, valid)
        if valid:
            writer.append_int_response(lag)
        else:
            writer.append_null_response()
        _kv_int(writer, "pel-count", Int64(grp.pel_live()))
        _key_str(writer, "pending")
        var np = grp.pel_live()
        if full_count > 0 and np > full_count:
            np = full_count
        writer.append_array_header(np)
        var pk = grp.pel_next_live(0)
        for _ in range(np):
            var nk = grp.pel[pk]
            pk = grp.pel_next_live(pk + 1)
            writer.append_array_header(4)
            write_id(writer, nk.ms, nk.seq)
            var oi = grp.consumer_by_id(nk.consumer)
            if oi >= 0:
                _write_name(writer, grp.consumers[oi].name)
            else:
                writer.append_bulk_string_response("".unsafe_ptr(), 0)
            writer.append_int_response(nk.delivery_time)
            writer.append_int_response(nk.delivery_count)
        _key_str(writer, "consumers")
        writer.append_array_header(grp.ncons())
        for c in range(grp.ncons()):
            ref con = grp.consumers[grp.by_name[c]]
            writer.append_map_header(5)
            _kv_name(writer, "name", con.name)
            _kv_int(writer, "seen-time", con.seen_time)
            _kv_int(writer, "active-time", con.active_time)
            _kv_int(writer, "pel-count", Int64(con.pending))
            _key_str(writer, "pending")
            var mine = List[Int]()
            for k in range(len(grp.pel)):
                if grp.pel[k].consumer == con.id:
                    mine.append(k)
                    if full_count > 0 and len(mine) >= full_count:
                        break
            writer.append_array_header(len(mine))
            for k in range(len(mine)):
                var nk = grp.pel[mine[k]]
                writer.append_array_header(3)
                write_id(writer, nk.ms, nk.seq)
                writer.append_int_response(nk.delivery_time)
                writer.append_int_response(nk.delivery_count)


def _key_str(mut writer: ResponseWriter, k: StaticString):
    writer.append_bulk_string_response(k.unsafe_ptr(), k.byte_length())


def _kv_int(mut writer: ResponseWriter, k: StaticString, v: Int64):
    _key_str(writer, k)
    writer.append_int_response(v)


def _kv_name(mut writer: ResponseWriter, k: StaticString, name: List[UInt8]):
    _key_str(writer, k)
    _write_name(writer, name)
