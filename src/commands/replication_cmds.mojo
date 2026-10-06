"""The replication commands a Redis client may send (#39): ROLE,
REPLICAOF/SLAVEOF, FAILOVER, SYNC/PSYNC and REPLCONF.

Pion replicates in cluster mode. A replica starts with --cluster
--cluster-replica --cluster-primary-host H --cluster-primary-port P and follows
its primary over the replication port (the primary's port + 10000) with Pion's
own stream: a snapshot, then the WAL. So:
- ROLE reports what this server is, as Redis does;
- REPLICAOF/SLAVEOF answer as Redis does where that is possible: refused in
  cluster mode (Redis refuses it there too), `NO ONE` on a primary is a no-op.
  Pointing a standalone server at a primary at run time is refused, with an
  error that names how a Pion replica is set up;
- FAILOVER answers as a Redis primary without connected replicas, which is
  what a standalone Pion always is; in cluster mode Redis refuses it, and
  Pion's failover is CLUSTER FAILOVER;
- SYNC and PSYNC would start a Redis replication stream (an RDB, then
  commands), which Pion does not produce: refused, naming what Pion does;
- REPLCONF accepts and answers its options as Redis does for a client that is
  not a replica: nothing to record, since Pion's replicas use their own port.
"""

from std.ffi import external_call
from std.memory import alloc
from std.memory.unsafe_pointer import Pointer
from src.common.ptr import is_not_null, is_null
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.network.cluster import ClusterState
from src.common.utils import arg_eq, parse_int64_strict


comptime _REPLICA_SETUP = "start a Pion replica with --cluster --cluster-replica --cluster-primary-host <host> --cluster-primary-port <port>"


def _bulk(mut writer: ResponseWriter, s: String):
    writer.append_bulk_string_response(s.unsafe_ptr(), s.byte_length())


def handle_role(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                mut writer: ResponseWriter, cluster: Pointer[ClusterState, MutUntrackedOrigin],
                wal_tail: Int):
    """ROLE: ["master", offset, [[ip, port, acked offset], ...]] on a primary
    (offset 0 and no replicas for a standalone server, which streams nothing),
    or ["slave", primary host, primary port, link state, applied offset]."""
    if num_tokens - i != 1:
        writer.append_error_response("ERR wrong number of arguments for 'role' command")
        return
    if is_null(cluster) or not cluster[].enabled:
        writer.append_array_header(3)
        _bulk(writer, "master")
        writer.append_int_response(0)
        writer.append_array_header(0)
        return
    if cluster[].is_replica:
        var host = String("")
        var port = 0
        var pi = cluster[].primary_peer_idx
        if pi >= 0 and pi < 16:
            var off = pi * 64
            var n = 0
            while n < 63 and cluster[].peer_hosts[off + n] != 0:
                host += chr(Int(cluster[].peer_hosts[off + n]))
                n += 1
            port = cluster[].peer_ports[pi]
        var state = String("connect")
        var applied = Int64(0)
        var h = cluster[].repl_replica_handle
        if is_not_null(h):
            var st = Int(external_call["pion_repl_replica_link_state", Int32](h))
            if st == 2:
                state = "connecting"
            elif st == 3:
                state = "handshake"
            elif st == 4:
                state = "sync"
            elif st == 5:
                state = "connected"
            applied = external_call["pion_repl_replica_applied_offset", Int64](h)
        writer.append_array_header(5)
        _bulk(writer, "slave")
        _bulk(writer, host)
        writer.append_int_response(Int64(port))
        _bulk(writer, state)
        writer.append_int_response(applied)
        return
    writer.append_array_header(3)
    _bulk(writer, "master")
    writer.append_int_response(Int64(wal_tail))
    var h = cluster[].repl_primary_handle
    if is_null(h):
        writer.append_array_header(0)
        return
    var ips = List[String]()
    var ports = List[Int]()
    var acks = List[Int64]()
    var ipbuf = alloc[UInt8](64)
    var pport = alloc[Int32](1)
    var pack = alloc[UInt64](1)
    var k = 0
    while k < 8 and external_call["pion_repl_primary_replica_info", Int32](h, Int32(k), ipbuf, Int32(64), pport, pack) == 1:
        var ip = String("")
        var n = 0
        while n < 63 and ipbuf[n] != 0:
            ip += chr(Int(ipbuf[n]))
            n += 1
        ips.append(ip)
        ports.append(Int(pport[0]))
        acks.append(Int64(pack[0]))
        k += 1
    ipbuf.unsafe_free()
    pport.unsafe_free()
    pack.unsafe_free()
    writer.append_array_header(len(ips))
    for r in range(len(ips)):
        writer.append_array_header(3)
        _bulk(writer, ips[r])
        _bulk(writer, String(ports[r]))
        _bulk(writer, String(acks[r]))


def handle_replicaof(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                     mut writer: ResponseWriter, cluster: Pointer[ClusterState, MutUntrackedOrigin]):
    """REPLICAOF / SLAVEOF host port | NO ONE (Redis's replicaofCommand)."""
    var name = tokens[i].value().lower()
    if num_tokens - i != 3:
        writer.append_error_response("ERR wrong number of arguments for '" + name + "' command")
        return
    if is_not_null(cluster) and cluster[].enabled:
        writer.append_error_response("ERR REPLICAOF not allowed in cluster mode.")
        return
    var h = tokens[i + 1]
    var p = tokens[i + 2]
    if arg_eq(h.ptr, h.length, "no") and arg_eq(p.ptr, p.length, "one"):
        writer.append_ok_response()        # already a primary: nothing to do
        return
    var port = parse_int64_strict(p.ptr, p.length)
    if not port.ok or port.value < 0 or port.value > 65535:
        writer.append_error_response("ERR Invalid master port")
        return
    writer.append_error_response("ERR REPLICAOF is not supported outside cluster mode: Pion replicates in cluster mode, set up at startup; "
                                 + _REPLICA_SETUP)


def handle_failover(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                    mut writer: ResponseWriter, cluster: Pointer[ClusterState, MutUntrackedOrigin]):
    """FAILOVER [TO host port [FORCE]] [ABORT] [TIMEOUT ms] (Redis's
    failoverCommand). A standalone Pion is a primary with no replicas, so
    after the arguments it answers what Redis answers then."""
    if is_not_null(cluster) and cluster[].enabled:
        writer.append_error_response("ERR FAILOVER not allowed in cluster mode.")
        return
    var argc = num_tokens - i
    if argc == 2 and arg_eq(tokens[i + 1].ptr, tokens[i + 1].length, "abort"):
        writer.append_error_response("ERR No failover in progress.")
        return
    var timeout = Int64(0)
    var force = False
    var have_host = False
    var j = 1
    while j < argc:
        var t = tokens[i + j]
        if arg_eq(t.ptr, t.length, "timeout") and j + 1 < argc and timeout == 0:
            var v = parse_int64_strict(tokens[i + j + 1].ptr, tokens[i + j + 1].length)
            if not v.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return
            if v.value <= 0:
                writer.append_error_response("ERR FAILOVER timeout must be greater than 0")
                return
            timeout = v.value
            j += 1
        elif arg_eq(t.ptr, t.length, "to") and j + 2 < argc and not have_host:
            var v = parse_int64_strict(tokens[i + j + 2].ptr, tokens[i + j + 2].length)
            if not v.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return
            have_host = True
            j += 2
        elif arg_eq(t.ptr, t.length, "force") and not force:
            force = True
        else:
            writer.append_error_response("ERR syntax error")
            return
        j += 1
    writer.append_error_response("ERR FAILOVER requires connected replicas.")


def handle_sync(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                mut writer: ResponseWriter):
    """SYNC / PSYNC: refused. A Redis primary answers them by streaming an RDB
    file and then its commands; Pion's replicas use their own stream."""
    var name = tokens[i].value().lower()
    var argc = num_tokens - i
    if (name == "sync" and argc != 1) or (name == "psync" and argc < 3):
        writer.append_error_response("ERR wrong number of arguments for '" + name + "' command")
        return
    writer.append_error_response("ERR " + name.upper() + " is not supported: Pion replicas follow a primary over "
                                 + "Pion's own replication stream; " + _REPLICA_SETUP)


def handle_replconf(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                    mut writer: ResponseWriter):
    """REPLCONF option value ... (Redis's replconfCommand, for a client that
    is not a replica). ACK and GETACK answer nothing, as in Redis."""
    var argc = num_tokens - i
    if argc % 2 == 0:
        writer.append_error_response("ERR syntax error")
        return
    var j = 1
    while j < argc:
        var o = tokens[i + j]
        var v = tokens[i + j + 1]
        if arg_eq(o.ptr, o.length, "listening-port"):
            if not parse_int64_strict(v.ptr, v.length).ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return
        elif arg_eq(o.ptr, o.length, "ip-address"):
            if v.length >= 256:
                writer.append_error_response("ERR REPLCONF ip-address provided by replica instance is too long: "
                                             + String(v.length) + " bytes")
                return
        elif arg_eq(o.ptr, o.length, "capa"):
            pass
        elif arg_eq(o.ptr, o.length, "ack") or arg_eq(o.ptr, o.length, "getack"):
            return                          # no reply
        elif arg_eq(o.ptr, o.length, "rdb-only"):
            var r = parse_int64_strict(v.ptr, v.length)
            if not r.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return
            if r.value < 0 or r.value > 1:
                writer.append_error_response("ERR value is out of range, value must between 0 and 1")
                return
        elif arg_eq(o.ptr, o.length, "rdb-filter-only"):
            # space-separated filters; "functions" is the only one
            var words = v.value().split()
            for w in range(len(words)):
                if words[w].lower() != "functions":
                    writer.append_error_response("ERR Unsupported rdb-filter-only option: " + String(words[w]))
                    return
        else:
            writer.append_error_response("ERR Unrecognized REPLCONF option: " + o.value())
            return
        j += 2
    writer.append_ok_response()
