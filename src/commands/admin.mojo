"""Admin/server commands: XGPU, PING, ECHO, HELLO, QUIT, AUTH, DBSIZE, SELECT, FLUSHALL, FLUSHDB, SAVE, BGSAVE, LASTSAVE, CONFIG, COMMAND, ACL, RESET, SWAPDB, SHUTDOWN, BGREWRITEAOF, DEBUG, SLOWLOG, LATENCY, MEMORY, MODULE, INFO."""
from src.common.ptr import is_not_null, is_null, null_ptr
from src.commands.command_table import PION_COMMAND_COUNT
from std.memory.unsafe_pointer import Pointer
from std.collections import Array, List, Span, Dict
from std.memory import alloc, unsafe_memcpy
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.common.hll import HLL_REGISTERS
from src.common.value import GenericValue, ValueType
from src.common.config import PionConfig
from src.common.metrics import ValueLedger
from src.common.utils import format_int_to_buf, parse_memory_value, arg_eq, parse_int64_strict, _glob_match, bytes_to_string
# gh #257: whole-name, case-insensitive matching for CONFIG parameter names.
# Not `| 0x20` — that maps '-' fine but mangles '_' (see gh #225), and
# `maxmemory-policy` must match exactly.
from src.network.fast_path import cmd_eq
from src.io.wal import WAL
from src.io.snapshot import SnapshotEngine
from src.network.raft import RaftNode
from src.network.dispatcher import CommandDispatcher
from src.network.cluster import ClusterState
from src.commands.tenant import TenantTable
from std.ffi import external_call
from std.sys.info import CompilationTarget


@always_inline
def handle_xgpu_info(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """XGPU INFO — return GPU diagnostics as bulk string."""
    if i + 1 < num_tokens:
        var sub_tok = tokens[unsafe_offset=i + 1]
        var sub_ptr = sub_tok.ptr; var sub_len = sub_tok.length
        var sub0 = sub_ptr[unsafe_offset=0] | 0x20
        var extra = 1  # consume subcommand

        if sub_len == 4 and sub0 == 105: # INFO (i=105)
            var gi_buf = alloc[UInt8](1024)
            var gi_mb = gi_buf
            var gi_off = 0

            def _gi_sl(s: StringLiteral, b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                var sl = s.byte_length(); unsafe_memcpy(dest=b.unsafe_offset(o), src=s.unsafe_ptr(), count=sl); o += sl
            def _gi_in(n: Int, b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                o += format_int_to_buf(b.unsafe_offset(o), 0, Int64(n))
            def _gi_in64(n: Int64, b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                o += format_int_to_buf(b.unsafe_offset(o), 0, n)
            def _gi_nl2(b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                b[unsafe_offset=o] = 13; b[unsafe_offset=o + 1] = 10; o += 2

            _gi_sl("# GPU", gi_mb, gi_off); _gi_nl2(gi_mb, gi_off)

            var gpu_avail = 0
            comptime if CompilationTarget.is_macos():
                gpu_avail = Int(external_call["pion_metal_available", Int32]())

            _gi_sl("gpu_available:", gi_mb, gi_off)
            _gi_in(gpu_avail, gi_mb, gi_off); _gi_nl2(gi_mb, gi_off)

            if gpu_avail == 1:
                comptime if CompilationTarget.is_macos():
                    _gi_sl("gpu_device_name:", gi_mb, gi_off)
                    var dev_name_ptr = external_call["pion_metal_device_name", Pointer[UInt8, MutUntrackedOrigin]]()
                    var k = 0
                    while k < 64 and dev_name_ptr[unsafe_offset=k] != 0:
                        gi_mb[unsafe_offset=gi_off] = dev_name_ptr[unsafe_offset=k]
                        gi_off += 1; k += 1
                    _gi_nl2(gi_mb, gi_off)

                    _gi_sl("gpu_max_threadgroup_size:", gi_mb, gi_off)
                    _gi_in(Int(external_call["pion_metal_max_threadgroup_size", UInt32]()), gi_mb, gi_off); _gi_nl2(gi_mb, gi_off)

                    _gi_sl("gpu_avg_latency_ns:", gi_mb, gi_off)
                    _gi_in64(Int64(external_call["pion_metal_gpu_latency_ns", UInt64]()), gi_mb, gi_off); _gi_nl2(gi_mb, gi_off)

                    _gi_sl("gpu_dispatch_count:", gi_mb, gi_off)
                    _gi_in(Int(external_call["pion_metal_gpu_dispatch_count", UInt32]()), gi_mb, gi_off); _gi_nl2(gi_mb, gi_off)

                    _gi_sl("gpu_rerank_count:", gi_mb, gi_off)
                    _gi_in64(Int64(external_call["pion_metal_rerank_count", UInt64]()), gi_mb, gi_off); _gi_nl2(gi_mb, gi_off)

            writer.append_bulk_string_response(gi_mb, gi_off)
            gi_buf.unsafe_free()
        else:
            writer.append_error_response("ERR unknown subcommand for GPU command")
        return extra
    else:
        writer.append_error_response("ERR wrong number of arguments for 'gpu' command")
        return 0


@always_inline
def handle_ping(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, mut dispatcher: CommandDispatcher) raises -> Int:
    """PING [message] → +PONG, or the message as a bulk string.

    `num_tokens` is this command's own end (cmd_end_tok). The argument used to
    count only when its RESP marker was NOT '$' — but every argument of an
    array command IS a bulk string, so `PING hello` answered +PONG whenever it
    reached the slow path (e.g. pipelined behind `DEL a b`), and a message
    spelled SET or GET was dropped too. Both were guards against reading the
    NEXT pipelined command as the message, from before commands had their own
    token bound."""
    if i + 2 < num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'ping' command")
        return num_tokens - i - 1
    if i + 1 < num_tokens:
        var t = tokens[unsafe_offset=i+1]
        writer.append_bulk_string_response(t.ptr, t.length)
        return 1
    writer.append_pong_response()
    return 0


@always_inline
def handle_echo(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) raises -> Int:
    """ECHO message → $N\\r\\nmessage\\r\\n."""
    if i + 1 < num_tokens:
        var echo_msg = tokens[unsafe_offset=i+1].value()
        writer.append_bulk_string_response(echo_msg.unsafe_ptr(), echo_msg.byte_length())
        return 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'echo' command")
        return 0


@always_inline
def handle_hello(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, cluster: Pointer[ClusterState, MutUntrackedOrigin],
                 requirepass: String, authed: Pointer[UInt8, MutUntrackedOrigin], fd: Int32,
                 tenants: Pointer[TenantTable, MutUntrackedOrigin],
                 tenant_ids: Pointer[Int16, MutUntrackedOrigin],
                 resp_proto: Pointer[UInt8, MutUntrackedOrigin]) -> Int:
    """HELLO [protover [AUTH user pass]] — RESP2/RESP3 negotiation.
    gh #100 (C2): honors an inline AUTH clause, and when --requirepass is set an
    unauthenticated HELLO returns -NOAUTH instead of leaking the server banner.
    gh #101: `HELLO <proto> AUTH <tenant> <password>` binds the tenant, same
    rules as handle_auth (a tenant-name match never falls through to admin).
    The caller consumes all args via `i = cmd_end_tok - 1`, so `num_tokens` here
    is the command's token-end bound.

    gh #172: `HELLO 3` is now honored — it binds the fd to RESP3 and this reply
    goes out as a RESP3 map. Previously every protover answered with a RESP2
    array carrying `proto:2`, which crashed redis-py >= 8 (RESP3 by default
    since 8.0: it switches its parser before reading the reply and indexes the
    result as a map). The intermediate mitigation answered `-NOPROTO`; that
    turned a crash into a clear error but still left the largest Redis client
    unable to connect with defaults."""
    var extra = 0
    var want_proto: UInt8 = 2
    if i + 1 < num_tokens:
        # Check if next token is a digit (protocol version), not another command
        var nt = tokens[unsafe_offset=i + 1]
        if nt.length > 0 and nt.ptr[unsafe_offset=0] >= 48 and nt.ptr[unsafe_offset=0] <= 57:
            extra = 1
            # Only 2 and 3 exist. Anything else keeps Redis's canonical -NOPROTO.
            if nt.length == 1 and nt.ptr[unsafe_offset=0] == 51:    # "3"
                want_proto = 3
            elif not (nt.length == 1 and nt.ptr[unsafe_offset=0] == 50):  # != "2"
                writer.append_error_response("NOPROTO unsupported protocol version")
                return extra
    # gh #100 (C2): scan for an inline `AUTH <username> <password>` clause.
    if requirepass.byte_length() > 0:
        var j = i + 1
        var auth_failed = False
        while j < num_tokens:
            var t = tokens[unsafe_offset=j]
            if t.length == 4 and (t.ptr[unsafe_offset=0]|0x20)==97 and (t.ptr[unsafe_offset=1]|0x20)==117 and (t.ptr[unsafe_offset=2]|0x20)==116 and (t.ptr[unsafe_offset=3]|0x20)==104:  # AUTH
                if j + 2 < num_tokens:
                    var user = tokens[unsafe_offset=j + 1]
                    var pw = tokens[unsafe_offset=j + 2]  # AUTH <user> <pass>
                    # gh #101: tenant credential first; -2 (name hit, wrong
                    # password) must not fall through to the admin compare.
                    var tm = -1
                    if is_not_null(tenants) and tenants[].count > 0:
                        tm = tenants[].match_credentials(user.ptr, user.length, pw.ptr, pw.length)
                    var ok = False
                    if tm >= 0:
                        ok = True
                        if is_not_null(authed):
                            authed[unsafe_offset=Int(fd)] = 1
                        if is_not_null(tenant_ids):
                            tenant_ids[unsafe_offset=Int(fd)] = Int16(tm)
                    elif tm == -1 and pw.length == requirepass.byte_length():
                        var m = True
                        for k in range(requirepass.byte_length()):
                            if pw.ptr[unsafe_offset=k] != requirepass.unsafe_ptr()[unsafe_offset=k]:
                                m = False; break
                        if m and is_not_null(authed):
                            ok = True
                            authed[unsafe_offset=Int(fd)] = 1
                            if is_not_null(tenant_ids):
                                tenant_ids[unsafe_offset=Int(fd)] = -1  # admin
                    if not ok:
                        auth_failed = True
                    j += 3
                    continue
            j += 1
        # #24: a rejected AUTH clause is WRONGPASS, as AUTH itself answers and
        # as Redis answers HELLO. It also wins over an earlier successful AUTH
        # on this connection: the clause failed, so the banner is not sent.
        # NOAUTH is for a HELLO that carried no credentials at all.
        if auth_failed:
            writer.append_error_response("WRONGPASS invalid username-password pair or user is disabled.")
            return extra
        # Still unauthenticated → refuse to emit the banner.
        if is_null(authed) or authed[unsafe_offset=Int(fd)] == 0:
            writer.append_error_response("NOAUTH HELLO must be called with the client already authenticated, otherwise the HELLO <proto> AUTH <user> <pass> option can be used to authenticate the client and select the RESP protocol version at the same time")
            return extra
    # Negotiation succeeded — bind the fd and switch this writer immediately so
    # the reply below (and every later command in the same recv batch) is
    # already in the agreed protocol.
    if is_not_null(resp_proto):
        resp_proto[unsafe_offset=Int(fd)] = want_proto
    writer.proto = want_proto

    var cluster_enabled2 = is_not_null(cluster) and cluster[].enabled
    var role_str = String("master")
    if cluster_enabled2 and cluster[].is_replica:
        role_str = String("slave")
    var mode_str = String("cluster") if cluster_enabled2 else String("standalone")
    # 7 pairs: server, version, proto, id, mode, role, modules. append_map_header
    # emits `%7` on RESP3 and `*14` on RESP2 — the same byte sequence this
    # handler used to hardcode.
    writer.append_map_header(7)
    # gh #428 (D-13): HELLO's `version` is the version a Redis client gates on, so
    # it reports the same string INFO's `redis_version` does (7.0.0) — not the
    # phantom `1.0.0` that matched neither VERSION nor INFO. `INFO pion_version`
    # still carries the real Pion build. Both strings are 5 bytes, so `$5` holds.
    var hello_hdr = String("$6\r\nserver\r\n$4\r\npion\r\n$7\r\nversion\r\n$5\r\n7.0.0\r\n$5\r\nproto\r\n")
    writer.append_to_response(hello_hdr.unsafe_ptr(), hello_hdr.byte_length())
    writer.append_int_response(Int64(want_proto))
    var hello_id = String("$2\r\nid\r\n:1\r\n$4\r\nmode\r\n")
    writer.append_to_response(hello_id.unsafe_ptr(), hello_id.byte_length())
    writer.append_bulk_string_response(mode_str.unsafe_ptr(), mode_str.byte_length())
    var hello_role = String("$4\r\nrole\r\n")
    writer.append_to_response(hello_role.unsafe_ptr(), hello_role.byte_length())
    writer.append_bulk_string_response(role_str.unsafe_ptr(), role_str.byte_length())
    var hello_mod = String("$7\r\nmodules\r\n*0\r\n")
    writer.append_to_response(hello_mod.unsafe_ptr(), hello_mod.byte_length())
    return extra


def _flush_args_ok(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                   mut writer: ResponseWriter) -> Bool:
    """[ASYNC|SYNC], as Redis's getFlushCommandFlags: one optional word,
    nothing else."""
    var argc = num_tokens - i
    if argc > 2 or (argc == 2 and not arg_eq(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length, "sync")
                    and not arg_eq(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length, "async")):
        writer.append_error_response("ERR syntax error")
        return False
    return True


def _flush(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin]):
    """Empty the keyspace (its TTLs with it) and log it. A flush used to be
    left out of the WAL, so a restart replayed the flushed keys, and replicas,
    which follow the WAL, kept them. Record 250 is the FLUSH the replication
    stream already sends before a resync."""
    keyspace[].reset()
    if is_not_null(wal):
        _ = wal[].append(250, null_ptr[UInt8, MutUntrackedOrigin](), 0)


def handle_flushall(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
                    wal: Pointer[WAL, MutUntrackedOrigin]) -> Int:
    """FLUSHALL [ASYNC|SYNC] — clear all keys in this worker's keyspace."""
    if _flush_args_ok(tokens, i, num_tokens, writer):
        _flush(keyspace, wal)
        writer.append_ok_response()
    return num_tokens - 1 - i


@always_inline
def handle_save(mut snapshot_engine: SnapshotEngine, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], worker_id: Int, mut dispatcher: CommandDispatcher, mut last_save_time: Int64, mut writer: ResponseWriter,
           ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] = null_ptr[SlabHashMap, MutUntrackedOrigin]()) -> Int:
    """SAVE — synchronous snapshot + WAL checkpoint."""
    var save_ts = snapshot_engine.take_snapshot(
        keyspace, worker_id, dispatcher.wal[].seq, dispatcher.blobs, ttl_map)
    if save_ts >= 0:
        dispatcher.wal[].checkpoint()
        last_save_time = save_ts
    writer.append_ok_response()
    return 0


@always_inline
def handle_bgsave(mut snapshot_engine: SnapshotEngine, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], worker_id: Int, mut dispatcher: CommandDispatcher, mut last_save_time: Int64, mut writer: ResponseWriter,
             ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] = null_ptr[SlabHashMap, MutUntrackedOrigin]()) -> Int:
    """BGSAVE — synchronous snapshot (fork-based async is Phase 2); respond immediately."""
    var bgsave_ts = snapshot_engine.take_snapshot(
        keyspace, worker_id, dispatcher.wal[].seq, dispatcher.blobs, ttl_map)
    if bgsave_ts >= 0:
        dispatcher.wal[].checkpoint()
        last_save_time = bgsave_ts
    var bgsave_resp = "+Background saving started\r\n"
    writer.append_to_response(bgsave_resp.unsafe_ptr(), bgsave_resp.byte_length())
    return 0


@always_inline
def handle_time(mut writer: ResponseWriter):
    """TIME -> [unix seconds, microseconds], both bulk strings, as Redis (#30).
    It answered `-ERR unknown command`."""
    var ts = alloc[Int64](2)
    _ = external_call["clock_gettime", Int32](Int32(0), ts)   # CLOCK_REALTIME
    var sec = ts[unsafe_offset=0]
    var usec = ts[unsafe_offset=1] // 1000
    ts.unsafe_free()
    writer.append_array_header(2)
    writer.append_bulk_int_response(sec)
    writer.append_bulk_int_response(usec)


def handle_lolwut(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                  mut writer: ResponseWriter):
    """LOLWUT [VERSION v] [arguments] (#39), as Redis 7: VERSION 5 draws
    Georg Nees's Schotter ([cols] [squares per row] [squares per column]),
    VERSION 6 a skyline ([cols] [rows]), and anything else, the default
    included, prints the version: Pion's own, since the art is Redis's and the
    server is not. The art comes from src/ffi/redis_ports.c. A verbatim string
    in RESP3."""
    from src.common.version import PION_VERSION
    var argc = num_tokens - i
    var version = 0
    var first = i + 1
    if argc >= 3 and arg_eq(tokens[i + 1].ptr, tokens[i + 1].length, "version"):
        var v = parse_int64_strict(tokens[i + 2].ptr, tokens[i + 2].length)
        if not v.ok:
            writer.append_error_response("ERR value is not an integer or out of range")
            return
        # Redis formats the version as "%u.0.0" from an unsigned int.
        var u = Int(UInt64(v.value) & 0xFFFFFFFF)
        version = u if u == 5 or u == 6 else 0
        first = i + 3
    var want = 3 if version == 5 else (2 if version == 6 else 0)
    var args = alloc[Int64](3)
    var nargs = 0
    while nargs < want and first + nargs < num_tokens:
        var t = tokens[first + nargs]
        var a = parse_int64_strict(t.ptr, t.length)
        if not a.ok:
            args.unsafe_free()
            writer.append_error_response("ERR value is not an integer or out of range")
            return
        args[nargs] = a.value
        nargs += 1
    var label = String("Pion ver. ") + PION_VERSION
    var out = alloc[Pointer[UInt8, MutUntrackedOrigin]](1)
    var n = external_call["pion_lolwut", Int64](Int64(version), Int64(nargs), args,
                                               label.unsafe_ptr(), Int64(label.byte_length()), out)
    var text = out[0]
    if n >= 0:
        writer.append_verbatim_response(text, Int(n))
    else:
        writer.append_error_response("ERR out of memory")
    external_call["pion_lcs_free", NoneType](text)
    out.unsafe_free()
    args.unsafe_free()
    _ = label^


@always_inline
def handle_lastsave(last_save_time: Int64, mut writer: ResponseWriter) -> Int:
    """LASTSAVE — return Unix timestamp of last successful snapshot (0 = none)."""
    writer.append_int_response(last_save_time)
    return 0


@always_inline
def handle_info(mut dispatcher: CommandDispatcher, mut writer: ResponseWriter,
                listen_port: Int, keys: Int, expires: Int, uptime_s: Int, extra: String,
                repl_section: String, cluster_enabled: Bool, sections: List[String]) -> Int:
    """INFO [section] — return server info as bulk string.

    gh #262: port, memory, uptime and keyspace are resolved by the caller
    from real state; `extra` is the `# Pion` value-receipt section."""
    var body = dispatcher.execute_info(writer.ctx[].send_stalls, listen_port, keys, expires, uptime_s, extra,
                                       repl_section, cluster_enabled)
    var out = _info_sections(body, sections)
    writer.append_verbatim_response(out.unsafe_ptr(), out.byte_length())
    return 0


def _info_sections(body: String, sections: List[String]) -> String:
    """INFO [section ...]: only the named sections, case-insensitive, as
    Redis (#30); none, `all`, `everything` or `default` mean every section
    (Pion has no section `default` would leave out). An unknown name adds
    nothing, so `INFO nosuch` is empty, as in Redis. INFO ignored its
    arguments and always sent everything."""
    if len(sections) == 0:
        return body
    for k in range(len(sections)):
        var w = sections[k].lower()
        if w == "all" or w == "everything" or w == "default":
            return body
    var out = String("")
    var keep = False
    var bp = body.unsafe_ptr()
    var n = body.byte_length()
    var start = 0
    while start < n:
        var end = start
        while end < n and bp[end] != 10:
            end += 1
        var line = String(StringSpan[MutUntrackedOrigin](
            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bp) + start),
                length=(end + 1 if end < n else end) - start)))
        if end - start >= 2 and bp[start] == 35 and bp[start + 1] == 32:   # "# Name"
            var name_end = end
            if name_end > start and bp[name_end - 1] == 13:
                name_end -= 1
            var name = String(StringSpan[MutUntrackedOrigin](
                unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                    unsafe_ptr=Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bp) + start + 2),
                    length=name_end - start - 2))).lower()
            keep = False
            for k in range(len(sections)):
                if sections[k].lower() == name:
                    keep = True
                    break
        if keep:
            out += line
        start = end + 1
    return out


@always_inline
def _ps_key(mut writer: ResponseWriter, k: StringLiteral):
    writer.append_bulk_string_response(k.unsafe_ptr(), k.byte_length())


@always_inline
def _ps_int(mut writer: ResponseWriter, k: StringLiteral, v: UInt64):
    _ps_key(writer, k)
    writer.append_int_response(Int64(v))


@always_inline
def _ps_str(mut writer: ResponseWriter, k: StringLiteral, v: String):
    _ps_key(writer, k)
    writer.append_bulk_string_response(v.unsafe_ptr(), v.byte_length())


@always_inline
def handle_pion_stats(mut writer: ResponseWriter, ledger: ValueLedger,
                      sem_hits: UInt64, sem_misses: UInt64,
                      moe_hits: UInt64, moe_misses: UInt64, worker_id: Int) -> Int:
    """PION.STATS — the value receipt as a map (RESP3 `%`, RESP2 flat array).

    gh #262. Per WORKER: the counters live in the worker that answered, so
    a client pool spanning workers reads each worker's own receipt.
    `prefill_seconds_avoided` is measured + estimated; the `_measured`
    twin is the part backed by client-reported PREFILL_MS only."""
    writer.append_map_header(16)
    _ps_int(writer, "worker_id", UInt64(worker_id))
    _ps_int(writer, "uptime_in_seconds", UInt64(ledger.uptime_seconds()))
    _ps_int(writer, "kvprefix_hits", ledger.kvprefix_hits)
    _ps_int(writer, "kvprefix_misses", ledger.kvprefix_misses)
    _ps_int(writer, "kvprefix_hits_measured", ledger.kvprefix_hits_measured)
    _ps_int(writer, "kvprefix_tokens_served", ledger.kvprefix_tokens_served)
    _ps_int(writer, "kvprefix_bytes_served", ledger.kvprefix_bytes_served)
    _ps_int(writer, "kvprefix_prefill_us_saved", ledger.kvprefix_prefill_us_saved)
    _ps_int(writer, "kvprefix_prefill_us_measured", ledger.kvprefix_prefill_us_measured)
    _ps_str(writer, "prefill_seconds_avoided", ledger.prefill_seconds_avoided())
    _ps_str(writer, "prefill_seconds_avoided_measured", ledger.prefill_seconds_avoided_measured())
    _ps_int(writer, "semantic_hits", sem_hits)
    _ps_int(writer, "semantic_misses", sem_misses)
    _ps_int(writer, "moe_hits", moe_hits)
    _ps_int(writer, "moe_misses", moe_misses)
    _ps_int(writer, "vector_queries", ledger.vector_queries)
    return 0


@always_inline
def _config_known_value(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int,
                        config: PionConfig, mut found: Bool) -> String:
    """gh #257: the value Pion can TRUTHFULLY report for a config parameter.

    Deliberately small. The old code answered every parameter with an empty
    string, which is a fabricated value dressed as a real one — an operator or
    a client-library probe reads it as "configured to nothing". Redis answers
    an unknown parameter with an empty array, and so does the caller when this
    returns `found = False`.

    A parameter belongs here only if Pion can state its value without
    inventing anything. `databases` is 1: SELECT refuses every other index.
    `tcp-keepalive`, `appendfsync` and friends are absent, since
    no value for them would be true."""
    found = True
    if cmd_eq(p, plen, "databases"):
        return "1"
    # gh #261: the live limit (from --maxmemory or CONFIG SET), in bytes; 0 is
    # unlimited. The policy is always noeviction: over the limit, memory-growing
    # writes are refused with -OOM and nothing is ever evicted.
    if cmd_eq(p, plen, "maxmemory"):
        return String(Int(external_call["pion_get_maxmemory", UInt64]()))
    if cmd_eq(p, plen, "maxmemory-policy"):
        return "noeviction"
    # Pion's durability is a WAL, not a Redis AOF. "no" is the truthful answer
    # to "do you have an append-only file", not a claim about durability.
    if cmd_eq(p, plen, "appendonly"):
        return "no"
    # No scheduled background saves; SAVE/BGSAVE are explicit. Redis reports an
    # empty string when save points are disabled, which is exactly the state.
    if cmd_eq(p, plen, "save"):
        return ""
    if cmd_eq(p, plen, "port"):
        return String(config.server.port)
    # Pion never closes an idle client, which is what timeout 0 means.
    if cmd_eq(p, plen, "timeout"):
        return "0"
    # #45: --enable-debug-command
    if cmd_eq(p, plen, "enable-debug-command"):
        var m = config.server.enable_debug_command
        return "yes" if m == 1 else ("local" if m == 2 else "no")
    # #47: SLOWLOG's two settings (process-wide, settable), and the latency
    # monitor and per-command latency tracking, which Pion does not have
    if cmd_eq(p, plen, "slowlog-log-slower-than"):
        return String(Int(external_call["pion_slowlog_get_slower_than", Int64]()))
    if cmd_eq(p, plen, "slowlog-max-len"):
        return String(Int(external_call["pion_slowlog_get_max_len", Int64]()))
    if cmd_eq(p, plen, "slowlog-entry-max-argc"):
        return "32"           # src/commands/slowlog.mojo's SLOWLOG_MAX_ARGC
    if cmd_eq(p, plen, "slowlog-entry-max-string-len"):
        return "128"          # SLOWLOG_MAX_STRING
    if cmd_eq(p, plen, "latency-monitor-threshold"):
        return "0"
    if cmd_eq(p, plen, "latency-tracking"):
        return "no"
    found = False
    return ""


comptime _CONFIG_COUNT = 14


@always_inline
def _heap_bytes(s: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    """A heap copy of s's bytes, which the caller frees."""
    var n = s.byte_length()
    var h = alloc[UInt8](n + 1)
    unsafe_memcpy(dest=h, src=s.unsafe_ptr(), count=n)
    return h


def _config_name(k: Int) -> StaticString:
    """The parameters CONFIG GET reports, the ones whose value Pion can state
    truthfully (see _config_known_value)."""
    if k == 0: return "databases"
    if k == 1: return "maxmemory"
    if k == 2: return "maxmemory-policy"
    if k == 3: return "appendonly"
    if k == 4: return "save"
    if k == 5: return "port"
    if k == 6: return "timeout"
    if k == 7: return "enable-debug-command"
    if k == 8: return "slowlog-log-slower-than"
    if k == 9: return "slowlog-max-len"
    if k == 10: return "latency-monitor-threshold"
    if k == 11: return "slowlog-entry-max-argc"
    if k == 12: return "slowlog-entry-max-string-len"
    return "latency-tracking"


def handle_config(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                  config: PionConfig, mut stats_reset: Bool) raises -> Int:
    """CONFIG GET|SET|RESETSTAT|REWRITE|HELP (gh #257).

    Both halves used to lie. `CONFIG SET anything anything` replied +OK and did
    nothing, so an operator, a Terraform module or a client library's startup
    probe believed the setting took. `CONFIG GET` answered every parameter with
    an EMPTY VALUE, which reads as "configured to nothing" rather than "I do
    not know this parameter". Same rule as gh #229's numeric parsing: a command
    that cannot honour its contract must error, not acknowledge.

    Subcommands are whole words (the first letter used to decide, so `CONFIG
    GE x` was GET). GET takes one or more glob patterns, matched without
    regard to case as Redis matches them, and answers every known parameter
    any of them names; it read the first argument as an exact name and
    ignored the rest. RESETSTAT sets `stats_reset` for the caller, which owns
    the counters (PION.STATS)."""
    if num_tokens - i < 2:
        writer.append_error_response("ERR wrong number of arguments for 'config' command")
        return 0
    var sub = tokens[unsafe_offset=i+1]
    if arg_eq(sub.ptr, sub.length, "get"):
        if num_tokens - i < 3:
            writer.append_error_response("ERR wrong number of arguments for 'config|get' command")
            return num_tokens - i - 1
        # gh #172: RESP2 emits the flat array it always did; RESP3 a map.
        var hits = List[Int]()
        for k in range(_CONFIG_COUNT):
            # The name's bytes on the heap: a laundered pointer into a local
            # String dangles once the String is destroyed after its last use.
            var nh = _heap_bytes(_config_name(k))
            var nl = _config_name(k).byte_length()
            for j in range(i + 2, num_tokens):
                var pt = tokens[unsafe_offset=j]
                var low = alloc[UInt8](max(pt.length, 1))
                for b in range(pt.length):
                    var c = pt.ptr[unsafe_offset=b]
                    low[unsafe_offset=b] = c | 0x20 if c >= 65 and c <= 90 else c
                var hit = _glob_match(low, pt.length, 0, nh, nl, 0)
                low.unsafe_free()
                if hit:
                    hits.append(k)
                    break
            nh.unsafe_free()
        # Redis replies an empty map to patterns that match nothing (`*0`
        # under RESP2): "no such parameter", not one that exists and is blank.
        writer.append_map_header(len(hits))
        for h in range(len(hits)):
            var nh = _heap_bytes(_config_name(hits[h]))
            var nl = _config_name(hits[h]).byte_length()
            var found = False
            var val = _config_known_value(nh, nl, config, found)
            writer.append_bulk_string_response(nh, nl)
            nh.unsafe_free()
            writer.append_bulk_string_response(val.unsafe_ptr(), val.byte_length())
        return num_tokens - i - 1
    if arg_eq(sub.ptr, sub.length, "set"):
        if num_tokens - i < 4 or (num_tokens - i) % 2 != 0:
            writer.append_error_response("ERR wrong number of arguments for 'config|set' command")
            return num_tokens - i - 1
        # gh #261, #47: the runtime-settable parameters. Several pairs are
        # set together or not at all, as Redis 7 does: every pair is checked
        # before any is applied. Each value lives in C and every worker reads
        # it, so a change is process-wide. maxmemory takes Redis units only
        # (1k = 1000, 1kb = 1024); the `N%` form is a CLI extension that
        # Redis's CONFIG SET rejects too.
        var mm_set = False
        var mm_val = UInt64(0)
        var slt_set = False
        var slt_val = Int64(0)
        var sml_set = False
        var sml_val = Int64(0)
        var j = i + 2
        while j + 1 < num_tokens:
            var nt = tokens[unsafe_offset=j]
            var vt = tokens[unsafe_offset=j + 1]
            if cmd_eq(nt.ptr, nt.length, "maxmemory"):
                var parsed = parse_memory_value(vt.ptr, vt.length, 0)
                if not parsed.ok:
                    writer.append_error_response(
                        "ERR CONFIG SET failed (possibly related to argument 'maxmemory') "
                        + "- argument must be a memory value")
                    return num_tokens - i - 1
                mm_set = True
                mm_val = UInt64(parsed.value)
            elif cmd_eq(nt.ptr, nt.length, "slowlog-log-slower-than") or cmd_eq(nt.ptr, nt.length, "slowlog-max-len"):
                var is_slt = cmd_eq(nt.ptr, nt.length, "slowlog-log-slower-than")
                var name = String("slowlog-log-slower-than") if is_slt else String("slowlog-max-len")
                var pv = parse_int64_strict(vt.ptr, vt.length)
                if not pv.ok:
                    writer.append_error_response("ERR CONFIG SET failed (possibly related to argument '" + name
                                                 + "') - argument couldn't be parsed into an integer")
                    return num_tokens - i - 1
                var lo = Int64(-1) if is_slt else Int64(0)
                if pv.value < lo:
                    writer.append_error_response("ERR CONFIG SET failed (possibly related to argument '" + name
                                                 + "') - argument must be between " + String(Int(lo))
                                                 + " and 9223372036854775807 inclusive")
                    return num_tokens - i - 1
                if is_slt:
                    slt_set = True
                    slt_val = pv.value
                else:
                    sml_set = True
                    sml_val = pv.value
            elif (cmd_eq(nt.ptr, nt.length, "latency-monitor-threshold") and vt.length == 1 and vt.ptr[0] == 48) \
                    or (cmd_eq(nt.ptr, nt.length, "latency-tracking") and arg_eq(vt.ptr, vt.length, "no")):
                pass     # already so: Pion has neither
            else:
                # Nothing else here is settable at runtime: every knob is a
                # CLI flag read once at startup, so acknowledging a SET would
                # claim a change that never happens. Erroring is the whole
                # point of gh #257. (latency-monitor-threshold and
                # latency-tracking land here unless they ask for what Pion is.)
                writer.append_error_response(
                    "ERR CONFIG SET is not supported for '" + bytes_to_string(nt.ptr, nt.length)
                    + "' — Pion is configured by CLI flags at startup (see ./pion-server --help).")
                return num_tokens - i - 1
            j += 2
        if mm_set:
            external_call["pion_set_maxmemory", NoneType](mm_val)
        if slt_set:
            external_call["pion_slowlog_set_slower_than", NoneType](slt_val)
        if sml_set:
            external_call["pion_slowlog_set_max_len", NoneType](sml_val)
        writer.append_ok_response()
        return num_tokens - i - 1
    if arg_eq(sub.ptr, sub.length, "resetstat"):
        if num_tokens - i != 2:
            writer.append_error_response("ERR wrong number of arguments for 'config|resetstat' command")
            return num_tokens - i - 1
        stats_reset = True
        writer.append_ok_response()
        return 1
    if arg_eq(sub.ptr, sub.length, "rewrite"):
        if num_tokens - i != 2:
            writer.append_error_response("ERR wrong number of arguments for 'config|rewrite' command")
            return num_tokens - i - 1
        # Pion reads no config file, which is Redis's answer in that state.
        writer.append_error_response("ERR The server is running without a config file")
        return 1
    if arg_eq(sub.ptr, sub.length, "help"):
        writer.append_array_header(11)
        writer.append_status_response("CONFIG <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
        writer.append_status_response("GET <pattern>")
        writer.append_status_response("    Return parameters matching the glob-like <pattern> and their values.")
        writer.append_status_response("SET <directive> <value>")
        writer.append_status_response("    Set the configuration <directive> to <value>.")
        writer.append_status_response("RESETSTAT")
        writer.append_status_response("    Reset statistics reported by the INFO command.")
        writer.append_status_response("REWRITE")
        writer.append_status_response("    Rewrite the configuration file.")
        writer.append_status_response("HELP")
        writer.append_status_response("    Print this help.")
        return num_tokens - i - 1
    writer.append_error_response("ERR unknown subcommand '" + bytes_to_string(sub.ptr, sub.length)
                                 + "'. Try CONFIG HELP.")
    return num_tokens - i - 1


@always_inline
def handle_quit(mut writer: ResponseWriter) -> Int:
    """QUIT → +OK then close (we return OK; client will close)."""
    writer.append_ok_response()
    return 0


@always_inline
def handle_auth(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                requirepass: String, authed: Pointer[UInt8, MutUntrackedOrigin], fd: Int32,
                tenants: Pointer[TenantTable, MutUntrackedOrigin],
                tenant_ids: Pointer[Int16, MutUntrackedOrigin]) -> Int:
    """AUTH [username] password (gh #100 / C2, gh #101 tenants).

    - No password configured  → Redis-compatible error (never a silent +OK).
    - `AUTH NAME PASSWORD` where NAME is a configured tenant → bind the fd to
      that tenant's namespace (gh #101). A matching name with a wrong password
      is -WRONGPASS — it never falls through to the admin credential, so a
      tenant password colliding with requirepass can't escalate.
    - Otherwise the last argument is compared to requirepass; on match the fd
      authenticates as ADMIN (unprefixed keyspace) and any prior tenant
      binding is cleared."""
    var extra = num_tokens - i - 1
    if requirepass.byte_length() == 0:
        writer.append_error_response("ERR Client sent AUTH, but no password is set. Did you mean AUTH <username> <password>?")
        return extra
    if extra < 1:
        writer.append_error_response("ERR wrong number of arguments for 'auth' command")
        return extra
    # gh #101: tenant credential path.
    if extra == 2 and is_not_null(tenants) and tenants[].count > 0:
        var user = tokens[unsafe_offset=i + 1]
        var tpw = tokens[unsafe_offset=i + 2]
        var m = tenants[].match_credentials(user.ptr, user.length, tpw.ptr, tpw.length)
        if m >= 0:
            if is_not_null(authed):
                authed[unsafe_offset=Int(fd)] = 1
            if is_not_null(tenant_ids):
                tenant_ids[unsafe_offset=Int(fd)] = Int16(m)
            writer.append_ok_response()
            return extra
        if m == -2:
            writer.append_error_response("WRONGPASS invalid username-password pair or user is disabled.")
            return extra
        # m == -1: not a tenant name → fall through to the admin credential.
    var pw = tokens[unsafe_offset=i + extra]  # last token = password
    var rp_ptr = requirepass.unsafe_ptr()
    var rp_len = requirepass.byte_length()
    var matches = pw.length == rp_len
    if matches:
        for k in range(rp_len):
            if pw.ptr[unsafe_offset=k] != rp_ptr[unsafe_offset=k]:
                matches = False
                break
    if matches:
        if is_not_null(authed):
            authed[unsafe_offset=Int(fd)] = 1
        if is_not_null(tenant_ids):
            tenant_ids[unsafe_offset=Int(fd)] = -1  # admin: unprefixed keyspace
        writer.append_ok_response()
    else:
        writer.append_error_response("WRONGPASS invalid username-password pair or user is disabled.")
    return extra


def handle_flushdb(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                   keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
                   wal: Pointer[WAL, MutUntrackedOrigin]) -> Int:
    """FLUSHDB [ASYNC|SYNC] — clear all keys (one database: FLUSHALL's effect)."""
    if _flush_args_ok(tokens, i, num_tokens, writer):
        _flush(keyspace, wal)
        writer.append_ok_response()
    return num_tokens - 1 - i


@always_inline
def handle_dbsize(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter) -> Int:
    """DBSIZE → :N (number of keys in this worker's keyspace)."""
    var _dbsz: Int64 = 0
    for _si in range(8): _dbsz += Int64(keyspace[].shards[unsafe_offset=_si].size)
    writer.append_int_response(_dbsz)
    return 0


@always_inline
def handle_select(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                  mut writer: ResponseWriter, cluster_enabled: Bool = False) -> Int:
    """SELECT index. Pion has one database per server, and answers as Redis
    does when configured with `databases 1`: `SELECT 0` is OK and
    any other index is refused. It used to answer +OK to anything and stay on
    DB 0, so a client keeping data in DB 1 read, overwrote and could FLUSHDB
    DB 0's keys."""
    if num_tokens - i != 2:
        writer.append_error_response("ERR wrong number of arguments for 'select' command")
        return 0
    var r = parse_int64_strict(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    if not r.ok or r.value > 2147483647 or r.value < -2147483648:
        writer.append_error_response("ERR value is not an integer or out of range")
    elif r.value != 0 and cluster_enabled:
        writer.append_error_response("ERR SELECT is not allowed in cluster mode")
    elif r.value != 0:
        writer.append_error_response("ERR DB index is out of range")
    else:
        writer.append_ok_response()
    return 0


@always_inline
def _db_index(t: RESP3Token, mut ok: Bool) -> Int64:
    """getIntFromObject: an integer that fits in 32 bits."""
    var r = parse_int64_strict(t.ptr, t.length)
    ok = r.ok and r.value <= 2147483647 and r.value >= -2147483648
    return r.value


def handle_swapdb(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                  mut writer: ResponseWriter, cluster_enabled: Bool = False) -> Int:
    """SWAPDB index1 index2, with one database: `SWAPDB 0 0` is OK,
    any other index is out of range. It answered +OK and did nothing."""
    if num_tokens - i != 3:
        writer.append_error_response("ERR wrong number of arguments for 'swapdb' command")
        return 0
    if cluster_enabled:
        writer.append_error_response("ERR SWAPDB is not allowed in cluster mode")
        return 0
    var ok = False
    var a = _db_index(tokens[unsafe_offset=i + 1], ok)
    if not ok:
        writer.append_error_response("ERR invalid first DB index")
        return 0
    var b = _db_index(tokens[unsafe_offset=i + 2], ok)
    if not ok:
        writer.append_error_response("ERR invalid second DB index")
        return 0
    if a != 0 or b != 0:
        writer.append_error_response("ERR DB index is out of range")
    else:
        writer.append_ok_response()
    return 0


def handle_move(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                mut writer: ResponseWriter, cluster_enabled: Bool = False) -> Int:
    """MOVE key db, with one database: DB 0 is the source itself,
    any other index is out of range — Redis's answers with `databases 1`."""
    if num_tokens - i != 3:
        writer.append_error_response("ERR wrong number of arguments for 'move' command")
        return 0
    if cluster_enabled:
        writer.append_error_response("ERR MOVE is not allowed in cluster mode")
        return 0
    var ok = False
    var db = _db_index(tokens[unsafe_offset=i + 2], ok)
    if not ok:
        writer.append_error_response("ERR value is not an integer or out of range")
    elif db != 0:
        writer.append_error_response("ERR DB index is out of range")
    else:
        writer.append_error_response("ERR source and destination objects are the same")
    return 0


@always_inline
def handle_bgrewriteaof(mut dispatcher: CommandDispatcher, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
                        ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) -> Int:
    """BGREWRITEAOF — compact the WAL: the live keyspace as fresh records, TTLs included."""
    dispatcher.wal[].compact_rewrite(keyspace, ttl_map)
    # #47: Redis's reply. Pion's rewrite has already run by now.
    writer.append_status_response("Background append only file rewriting started")
    return 0


@always_inline
def handle_debug(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                 keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mode: Int, fd: Int32) -> Int:
    """DEBUG, as far as Pion has it (#45): HELP, SET-ACTIVE-EXPIRE, SLEEP.
    Any other subcommand gets Redis's unknown-subcommand error; it answered
    +OK to every subcommand, run or not.

    Refused unless --enable-debug-command allows it (`mode` 0 no, the
    default, 1 yes, 2 local: loopback connections only), as Redis 7 and later
    refuse it."""
    if mode == 0 or (mode == 2 and external_call["pion_peer_is_local", Int32](fd) == 0):
        writer.append_error_response("ERR DEBUG command not allowed. If the enable-debug-command option is set to "
                                     + "\"local\", you can run it from a local connection, otherwise you need to set "
                                     + "this option in the configuration file, and then restart the server.")
        return 0
    var argc = num_tokens - i
    if argc < 2:
        writer.append_error_response("ERR wrong number of arguments for 'debug' command")
        return 0
    var sub = tokens[i + 1]
    if argc == 2 and arg_eq(sub.ptr, sub.length, "help"):
        var lines = List[String]()
        lines.append("DEBUG <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
        lines.append("SET-ACTIVE-EXPIRE <0|1>")
        lines.append("    Setting it to 0 disables expiring keys in background when they are not")
        lines.append("    accessed (otherwise the Redis behavior). Setting it to 1 reenables back the")
        lines.append("    default.")
        lines.append("SLEEP <seconds>")
        lines.append("    Stop the server for <seconds>. Decimals allowed.")
        lines.append("HELP")
        lines.append("    Print this help.")
        writer.append_array_header(len(lines))
        for k in range(len(lines)):
            writer.append_status_response(lines[k])
    elif argc == 3 and arg_eq(sub.ptr, sub.length, "set-active-expire"):
        # atoi, as Redis reads it: anything not starting with a non-zero
        # number turns the sweep off
        var v = tokens[i + 2].text_value() + "\0"
        keyspace[].active_expire = external_call["atoi", Int32](v.unsafe_ptr()) != 0
        _ = v^
        writer.append_ok_response()
    elif argc == 3 and arg_eq(sub.ptr, sub.length, "sleep"):
        var v = tokens[i + 2].text_value() + "\0"
        var secs = external_call["strtod", Float64](v.unsafe_ptr(), null_ptr[UInt8, MutUntrackedOrigin]())
        _ = v^
        if secs > 0:
            _ = external_call["usleep", Int32](UInt32(Int64(secs * 1_000_000.0)))
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR unknown subcommand or wrong number of arguments for '" + sub.text_value()
                                     + "'. Try DEBUG HELP.")
    return 0


@always_inline
def handle_latency(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """LATENCY (#47). Pion has no latency monitor (latency-monitor-threshold
    is 0, and CONFIG SET refuses another value) and does not track
    per-command latency (latency-tracking is no), so every subcommand answers
    as Redis does in that state: LATEST and HISTORY empty, RESET 0, GRAPH "no
    samples", DOCTOR its monitoring-is-disabled report, HISTOGRAM an empty
    map. RESET answered +OK, and DOCTOR, GRAPH, HISTOGRAM and HELP an empty
    array."""
    var argc = num_tokens - i
    if argc < 2:
        writer.append_error_response("ERR wrong number of arguments for 'latency' command")
        return 0
    var sub = tokens[unsafe_offset=i + 1]
    var sp = sub.ptr
    var sl = sub.length
    if arg_eq(sp, sl, "latest"):
        if argc != 2:
            _subcommand_arity(writer, "latency", "latest")
        else:
            writer.append_empty_array_response()
    elif arg_eq(sp, sl, "history"):
        if argc != 3:
            _subcommand_arity(writer, "latency", "history")
        else:
            writer.append_empty_array_response()
    elif arg_eq(sp, sl, "reset"):
        writer.append_int_response(0)
    elif arg_eq(sp, sl, "graph"):
        if argc != 3:
            _subcommand_arity(writer, "latency", "graph")
        else:
            writer.append_error_response("ERR No samples available for event '"
                                         + tokens[unsafe_offset=i + 2].text_value() + "'")
    elif arg_eq(sp, sl, "doctor"):
        if argc != 2:
            _subcommand_arity(writer, "latency", "doctor")
        else:
            var t = String("I'm sorry, Dave, I can't do that. Latency monitoring is disabled in this Redis instance. "
                           + "You may use \"CONFIG SET latency-monitor-threshold <milliseconds>.\" in order to "
                           + "enable it. If we weren't in a deep space mission I'd suggest to take a look at "
                           + "https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/latency-monitor.\n")
            writer.append_verbatim_response(t.unsafe_ptr(), t.byte_length())
    elif arg_eq(sp, sl, "histogram"):
        writer.append_map_header(0)
    elif arg_eq(sp, sl, "help"):
        if argc != 2:
            _subcommand_arity(writer, "latency", "help")
            return argc - 1
        var lines = List[String]()
        lines.append("DOCTOR")
        lines.append("    Return a human readable latency analysis report.")
        lines.append("GRAPH <event>")
        lines.append("    Return an ASCII latency graph for the <event> class.")
        lines.append("HISTORY <event>")
        lines.append("    Return time-latency samples for the <event> class.")
        lines.append("LATEST")
        lines.append("    Return the latest latency samples for all events.")
        lines.append("RESET [<event> ...]")
        lines.append("    Reset latency data of one or more <event> classes.")
        lines.append("    (default: reset all data for all event classes)")
        lines.append("HISTOGRAM [COMMAND ...]")
        lines.append("    Return a cumulative distribution of latencies in the format of a histogram for the specified command names.")
        lines.append("    If no commands are specified then all histograms are replied.")
        _help_reply(writer, "LATENCY", lines)
    else:
        _unknown_subcommand(writer, sub, "LATENCY")
    return argc - 1


def _memory_usage(v: GenericValue, key_len: Int, samples: Int) -> Int64:
    """MEMORY USAGE's estimate: the slot (metadata byte, 32-byte key and
    value), the key's bytes when they do not fit in it, and the value's own
    allocations: a string's bytes; a hash's, set's or sorted set's table plus
    its elements' bytes, sampled as Redis samples them (`samples` elements, 0
    all) and scaled to the whole; a list's elements."""
    var total = Int64(65)
    if key_len > 23:
        total += Int64(key_len)
    var t = v.type.value
    if t == ValueType.STRING or t == ValueType.BITMAP:
        total += Int64(v.string_len())
    elif t == ValueType.HLL:
        total += Int64(HLL_REGISTERS)
    elif t == ValueType.HASH:
        var m = v.as_hash().bitcast[SlabHashMap]()
        total += Int64(m[].capacity) * 65 + _sampled_payload(m, samples)
    elif t == ValueType.SET:
        var m = v.as_set().bitcast[SlabHashMap]()
        total += Int64(m[].capacity) * 65 + _sampled_payload(m, samples)
    elif t == ValueType.LIST:
        var l = v.as_list().bitcast[SlabList]()
        total += Int64(l[].size) * 34 + 64
    elif t == ValueType.ZSET or t == ValueType.GEO:
        var z = v.as_zset().bitcast[SlabSkipList]()
        var m = Pointer[SlabHashMap, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=z[].members)))
        total += Int64(z[].length) * 96 + Int64(m[].capacity) * 65 + _sampled_payload(m, samples)
    return total


def _sampled_payload(m: Pointer[SlabHashMap, MutUntrackedOrigin], samples: Int) -> Int64:
    """The heap bytes of a hash's or set's fields and values: every element
    when samples is 0, else the first `samples` found, scaled to the size."""
    var seen = 0
    var bytes = Int64(0)
    for slot in range(m[].capacity):
        var md = m[].metadata[slot]
        if md == 0x80 or md == 0xFF:
            continue
        if m[].keys[slot].type.value == ValueType.STRING:
            bytes += Int64(m[].keys[slot].string_len())
        if m[].values[slot].type.value == ValueType.STRING:
            bytes += Int64(m[].values[slot].string_len())
        seen += 1
        if samples > 0 and seen >= samples:
            break
    if seen == 0:
        return 0
    return bytes * Int64(m[].size) // Int64(seen)


@always_inline
def handle_memory(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter) raises -> Int:
    """MEMORY (#47). USAGE of a missing key is nil (it answered 64), and
    USAGE takes SAMPLES. STATS reports what Pion measures, under Redis's field
    names: the process RSS (which INFO's used_memory and --maxmemory also
    use), its peak, and the keys. MALLOC-STATS gives Redis's answer for an
    allocator without statistics, PURGE has nothing to purge, DOCTOR reports
    the same measurements, and HELP has Redis's text. STATS, MALLOC-STATS and
    PURGE answered an empty array."""
    var argc = num_tokens - i
    if argc < 2:
        writer.append_error_response("ERR wrong number of arguments for 'memory' command")
        return 0
    var sub = tokens[unsafe_offset=i + 1]
    var sp = sub.ptr
    var sl = sub.length
    if arg_eq(sp, sl, "usage"):
        if argc < 3:
            _subcommand_arity(writer, "memory", "usage")
            return argc - 1
        var samples = 5
        if argc > 3:
            if argc != 5 or not arg_eq(tokens[unsafe_offset=i + 3].ptr, tokens[unsafe_offset=i + 3].length, "samples"):
                writer.append_error_response("ERR syntax error")
                return argc - 1
            var sv = parse_int64_strict(tokens[unsafe_offset=i + 4].ptr, tokens[unsafe_offset=i + 4].length)
            if not sv.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return argc - 1
            samples = Int(sv.value) if sv.value > 0 else 0
        var kt = tokens[unsafe_offset=i + 2]
        var v = keyspace[].get(GenericValue.borrow(kt.ptr, kt.length))
        if v.is_none():
            writer.append_null_response()
        else:
            writer.append_int_response(_memory_usage(v, kt.length, samples))
    elif arg_eq(sp, sl, "stats"):
        if argc != 2:
            _subcommand_arity(writer, "memory", "stats")
            return argc - 1
        var rss = external_call["pion_rss_sample_now", UInt64]()
        var peak = external_call["pion_crash_rss_peak_bytes", UInt64]()
        if peak < rss:
            peak = rss
        var keys: Int64 = 0
        for k in range(8):
            keys += Int64(keyspace[].shards[k].size)
        writer.append_map_header(5)
        writer.append_bulk_string_response("peak.allocated".unsafe_ptr(), 14)
        writer.append_int_response(Int64(peak))
        writer.append_bulk_string_response("total.allocated".unsafe_ptr(), 15)
        writer.append_int_response(Int64(rss))
        writer.append_bulk_string_response("keys.count".unsafe_ptr(), 10)
        writer.append_int_response(keys)
        writer.append_bulk_string_response("peak.percentage".unsafe_ptr(), 15)
        var pct = String(Float64(rss) * 100.0 / Float64(peak)) if peak > 0 else String("0")
        writer.append_double_response(pct.unsafe_ptr(), pct.byte_length())
        writer.append_bulk_string_response("allocator.resident".unsafe_ptr(), 18)
        writer.append_int_response(Int64(rss))
    elif arg_eq(sp, sl, "malloc-stats"):
        if argc != 2:
            _subcommand_arity(writer, "memory", "malloc-stats")
            return argc - 1
        var t = String("Stats not supported for the current allocator")
        writer.append_bulk_string_response(t.unsafe_ptr(), t.byte_length())
    elif arg_eq(sp, sl, "purge"):
        if argc != 2:
            _subcommand_arity(writer, "memory", "purge")
            return argc - 1
        writer.append_ok_response()
    elif arg_eq(sp, sl, "doctor"):
        if argc != 2:
            _subcommand_arity(writer, "memory", "doctor")
            return argc - 1
        var rss = external_call["pion_rss_sample_now", UInt64]()
        var peak = external_call["pion_crash_rss_peak_bytes", UInt64]()
        if peak < rss:
            peak = rss
        var maxmem = external_call["pion_get_maxmemory", UInt64]()
        var t = String("Pion measures its memory as the process RSS. Now ") + String(Int(rss >> 20)) + " MB, peak " \
                + String(Int(peak >> 20)) + " MB, maxmemory "
        if maxmem > 0:
            t += String(Int(maxmem >> 20)) + " MB (" + String(Int(Float64(rss) * 100.0 / Float64(maxmem))) + "% used).\n"
        else:
            t += "not set.\n"
        writer.append_verbatim_response(t.unsafe_ptr(), t.byte_length())
    elif arg_eq(sp, sl, "help"):
        if argc != 2:
            _subcommand_arity(writer, "memory", "help")
            return argc - 1
        var lines = List[String]()
        lines.append("DOCTOR")
        lines.append("    Return memory problems reports.")
        lines.append("MALLOC-STATS")
        lines.append("    Return internal statistics report from the memory allocator.")
        lines.append("PURGE")
        lines.append("    Attempt to purge dirty pages for reclamation by the allocator.")
        lines.append("STATS")
        lines.append("    Return information about the memory usage of the server.")
        lines.append("USAGE <key> [SAMPLES <count>]")
        lines.append("    Return memory in bytes used by <key> and its value. Nested values are")
        lines.append("    sampled up to <count> times (default: 5, 0 means sample all).")
        _help_reply(writer, "MEMORY", lines)
    else:
        _unknown_subcommand(writer, sub, "MEMORY")
    return argc - 1


@always_inline
def _help_reply(mut writer: ResponseWriter, cmd: StaticString, lines: List[String]):
    """A HELP reply as Redis's addReplyHelp writes it: the header line, the
    command's lines, then HELP's own entry, each a status string."""
    writer.append_array_header(len(lines) + 3)
    writer.append_status_response(String(cmd) + " <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
    for k in range(len(lines)):
        writer.append_status_response(lines[k])
    writer.append_status_response("HELP")
    writer.append_status_response("    Print this help.")


def _unknown_subcommand(mut writer: ResponseWriter, sub: RESP3Token, cmd: StaticString):
    """Redis's reply to a subcommand the command does not have."""
    writer.append_error_response("ERR unknown subcommand '" + sub.text_value() + "'. Try " + String(cmd) + " HELP.")


def _subcommand_arity(mut writer: ResponseWriter, cmd: StaticString, sub: StaticString):
    """Redis's reply to a known subcommand given the wrong number of arguments."""
    writer.append_error_response("ERR wrong number of arguments for '" + String(cmd) + "|" + String(sub) + "' command")


def handle_module(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """MODULE LIST | LOAD | LOADEX | UNLOAD | HELP (#47).

    Pion loads no modules. LIST names `search`, the RediSearch-compatible FT.*
    commands Pion serves (LangChain and RedisVL look for it before they use
    FT.*). It also named ReJSON, whose JSON.* commands Pion does not have, so
    a client that trusted it failed at its first JSON command. LOAD, LOADEX and
    UNLOAD refuse. The subcommand was matched by its first letter and length,
    so `MODULE LOAD x` answered with the list, and the rest answered +OK."""
    var argc = num_tokens - i
    if argc < 2:
        writer.append_error_response("ERR wrong number of arguments for 'module' command")
        return 0
    var sub = tokens[unsafe_offset=i + 1]
    if arg_eq(sub.ptr, sub.length, "list"):
        if argc != 2:
            _subcommand_arity(writer, "module", "list")
            return argc - 1
        writer.append_array_header(1)
        writer.append_map_header(4)
        writer.append_bulk_string_response("name".unsafe_ptr(), 4)
        writer.append_bulk_string_response("search".unsafe_ptr(), 6)
        writer.append_bulk_string_response("ver".unsafe_ptr(), 3)
        writer.append_int_response(20800)
        writer.append_bulk_string_response("path".unsafe_ptr(), 4)
        writer.append_bulk_string_response("".unsafe_ptr(), 0)
        writer.append_bulk_string_response("args".unsafe_ptr(), 4)
        writer.append_empty_array_response()
    elif arg_eq(sub.ptr, sub.length, "help"):
        if argc != 2:
            _subcommand_arity(writer, "module", "help")
            return argc - 1
        var lines = List[String]()
        lines.append("LIST")
        lines.append("    Return a list of loaded modules.")
        lines.append("LOAD <path> [<arg> ...]")
        lines.append("    Load a module library from <path>, passing to it any optional arguments.")
        lines.append("LOADEX <path> [[CONFIG NAME VALUE] [CONFIG NAME VALUE]] [ARGS ...]")
        lines.append("    Load a module library from <path>, while passing it module configurations and optional arguments.")
        lines.append("UNLOAD <name>")
        lines.append("    Unload a module.")
        _help_reply(writer, "MODULE", lines)
    elif arg_eq(sub.ptr, sub.length, "load") or arg_eq(sub.ptr, sub.length, "loadex") \
            or arg_eq(sub.ptr, sub.length, "unload"):
        if argc < 3:
            writer.append_error_response("ERR wrong number of arguments for 'module|" + sub.text_value().lower()
                                         + "' command")
            return argc - 1
        writer.append_error_response("ERR Pion does not load modules: its FT.* search is built in")
    else:
        _unknown_subcommand(writer, sub, "MODULE")
    return argc - 1


@always_inline
def handle_reset(mut writer: ResponseWriter) -> Int:
    """RESET → +RESET\\r\\n (RESP3 connection reset)."""
    writer.append_to_response("+RESET\r\n".unsafe_ptr(), 8)
    return 0
