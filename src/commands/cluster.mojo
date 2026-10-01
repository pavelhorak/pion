"""Cluster commands: CLUSTER INFO/NODES/MYID/KEYSLOT/SLOTS/SHARDS/MEET/FORGET/REPLICATE/FAILOVER/RESET/SETSLOT/STATS."""
from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy
from std.collections import Array
from std.atomic import Atomic, Ordering
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.common.hash_map import StripedHashMap
from src.common.value import GenericValue
from src.network.cluster import ClusterState, REPL_PORT_OFFSET
from src.network.raft import RaftNode
from src.network.v_store import VStoreIndex, VStoreDirectory, MAX_DIR_ENTRIES
from src.network.attention_index import AttentionIndex
from src.vector.hnsw_types import SharedHNSWView
from src.io.wal import WAL
from src.common.utils import strict_atol, bytes_to_string, format_int_to_buf
from std.ffi import external_call


@always_inline
def handle_cluster(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, cluster: Pointer[ClusterState, MutUntrackedOrigin], keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], raft: Pointer[RaftNode, MutUntrackedOrigin], shared_hnsw: Pointer[SharedHNSWView, MutUntrackedOrigin], v_store: VStoreIndex, attn_idx: AttentionIndex, worker_id: Int, num_workers: Int) raises -> Int:
    """Dispatch CLUSTER subcommands. Returns number of extra tokens consumed beyond the subcommand."""
    var cluster_enabled = is_not_null(cluster) and cluster[].enabled
    var n_cluster_nodes = 1 + (cluster[].peer_count if cluster_enabled else 0)
    var consumed = 0
    if i + 1 < num_tokens:
        var sub_tok = tokens[unsafe_offset=i + 1]
        var sub_ptr = sub_tok.ptr
        var sub_len = sub_tok.length
        var sub0 = sub_ptr[unsafe_offset=0] | 0x20
        consumed += 1  # consume subcommand token

        if sub_len == 4 and sub0 == 105:
            # CLUSTER INFO (i=105,n=110,f=102,o=111)
            var ci_buf = alloc[UInt8](600)
            var ci_mb = ci_buf
            var ci_off = 0
            def _ci_sl(s: StringLiteral, b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                var sl = s.byte_length(); unsafe_memcpy(dest=b.unsafe_offset(o), src=s.unsafe_ptr(), count=sl); o += sl
            def _ci_in(n: Int, b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                o += format_int_to_buf(b.unsafe_offset(o), 0, Int64(n))
            def _ci_nl2(b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                b[unsafe_offset=o] = 13; b[unsafe_offset=o + 1] = 10; o += 2
            _ci_sl("cluster_enabled:", ci_mb, ci_off)
            _ci_in(1 if cluster_enabled else 0, ci_mb, ci_off)
            _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_state:ok", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_slots_assigned:16384", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_slots_ok:16384", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            var pfail_cnt = cluster[].count_pfail() if cluster_enabled else 0
            var fail_cnt  = cluster[].count_fail()  if cluster_enabled else 0
            _ci_sl("cluster_slots_pfail:", ci_mb, ci_off)
            _ci_in(pfail_cnt, ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_slots_fail:", ci_mb, ci_off)
            _ci_in(fail_cnt, ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_known_nodes:", ci_mb, ci_off)
            _ci_in(n_cluster_nodes, ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_size:", ci_mb, ci_off)
            _ci_in(n_cluster_nodes, ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_current_epoch:1", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_my_epoch:1", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_stats_messages_sent:0", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("cluster_stats_messages_received:0", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            _ci_sl("total_cluster_links_buffer_limit_exceeded:0", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
            # N3: replication offset + role
            if cluster[].is_replica:
                _ci_sl("cluster_role:slave", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
                _ci_sl("cluster_repl_offset:", ci_mb, ci_off)
                if is_not_null(cluster[].repl_replica_handle):
                    _ci_in(Int(external_call["pion_repl_replica_bytes_received", Int64](cluster[].repl_replica_handle)), ci_mb, ci_off)
                else:
                    _ci_in(0, ci_mb, ci_off)
                _ci_nl2(ci_mb, ci_off)
            else:
                _ci_sl("cluster_role:master", ci_mb, ci_off); _ci_nl2(ci_mb, ci_off)
                _ci_sl("cluster_repl_offset:", ci_mb, ci_off)
                if is_not_null(cluster[].repl_primary_handle):
                    _ci_in(Int(external_call["pion_repl_primary_max_sent_offset", Int64](cluster[].repl_primary_handle)), ci_mb, ci_off)
                else:
                    _ci_in(0, ci_mb, ci_off)
                _ci_nl2(ci_mb, ci_off)
                _ci_sl("cluster_connected_slaves:", ci_mb, ci_off)
                if is_not_null(cluster[].repl_primary_handle):
                    _ci_in(Int(external_call["pion_repl_primary_connected_count", Int32](cluster[].repl_primary_handle)), ci_mb, ci_off)
                else:
                    _ci_in(0, ci_mb, ci_off)
                _ci_nl2(ci_mb, ci_off)
            writer.append_bulk_string_response(ci_mb, ci_off)
            ci_buf.unsafe_free()

        elif sub_len == 5 and sub0 == 110:
            # CLUSTER NODES (n=110,o=111,d=100,e=101,s=115)
            var nb_buf = alloc[UInt8](4096)
            var nb_mb = nb_buf
            var nb_off = 0
            if cluster_enabled:
                unsafe_memcpy(dest=nb_mb.unsafe_offset(nb_off), src=cluster[].node_id.unsafe_ptr(), count=40); nb_off += 40
                nb_mb[unsafe_offset=nb_off] = 32; nb_off += 1
                unsafe_memcpy(dest=nb_mb.unsafe_offset(nb_off), src=cluster[].my_host.unsafe_ptr(), count=cluster[].my_host_len); nb_off += cluster[].my_host_len
                nb_mb[unsafe_offset=nb_off] = 58; nb_off += 1
                nb_off += format_int_to_buf(nb_mb.unsafe_offset(nb_off), 0, Int64(cluster[].my_port))
                nb_mb[unsafe_offset=nb_off] = 64; nb_off += 1
                nb_off += format_int_to_buf(nb_mb.unsafe_offset(nb_off), 0, Int64(cluster[].my_port + REPL_PORT_OFFSET))
                var ms_str = String(" myself,slave - 0 0 1 connected ") if cluster[].is_replica else String(" myself,master - 0 0 1 connected ")
                unsafe_memcpy(dest=nb_mb.unsafe_offset(nb_off), src=ms_str.unsafe_ptr().unsafe_bitcast[UInt8](), count=ms_str.byte_length()); nb_off += ms_str.byte_length()
                nb_off += format_int_to_buf(nb_mb.unsafe_offset(nb_off), 0, Int64(cluster[].my_slot_start))
                nb_mb[unsafe_offset=nb_off] = 45; nb_off += 1
                nb_off += format_int_to_buf(nb_mb.unsafe_offset(nb_off), 0, Int64(cluster[].my_slot_end))
                nb_mb[unsafe_offset=nb_off] = 10; nb_off += 1
                for pi in range(cluster[].peer_count):
                    unsafe_memcpy(dest=nb_mb.unsafe_offset(nb_off), src=cluster[].peer_node_id_ptr(pi), count=40); nb_off += 40
                    nb_mb[unsafe_offset=nb_off] = 32; nb_off += 1
                    unsafe_memcpy(dest=nb_mb.unsafe_offset(nb_off), src=cluster[].peer_host_ptr(pi), count=cluster[].peer_host_lens[pi]); nb_off += cluster[].peer_host_lens[pi]
                    nb_mb[unsafe_offset=nb_off] = 58; nb_off += 1
                    nb_off += format_int_to_buf(nb_mb.unsafe_offset(nb_off), 0, Int64(cluster[].peer_ports[pi]))
                    nb_mb[unsafe_offset=nb_off] = 64; nb_off += 1
                    nb_off += format_int_to_buf(nb_mb.unsafe_offset(nb_off), 0, Int64(cluster[].peer_ports[pi] + REPL_PORT_OFFSET))
                    # Live health: connected vs fail
                    var ph = cluster[].peer_health[pi]
                    var peer_status = String(" master - 0 0 1 connected ")
                    if ph == 2: peer_status = String(" master - 0 0 1 fail ")
                    elif ph == 1: peer_status = String(" master - 0 0 1 pfail ")
                    unsafe_memcpy(dest=nb_mb.unsafe_offset(nb_off), src=peer_status.unsafe_ptr().unsafe_bitcast[UInt8](), count=peer_status.byte_length()); nb_off += peer_status.byte_length()
                    nb_off += format_int_to_buf(nb_mb.unsafe_offset(nb_off), 0, Int64(cluster[].peer_slot_starts[pi]))
                    nb_mb[unsafe_offset=nb_off] = 45; nb_off += 1
                    nb_off += format_int_to_buf(nb_mb.unsafe_offset(nb_off), 0, Int64(cluster[].peer_slot_ends[pi]))
                    nb_mb[unsafe_offset=nb_off] = 10; nb_off += 1
            writer.append_bulk_string_response(nb_mb, nb_off)
            nb_buf.unsafe_free()

        elif sub_len == 4 and sub0 == 109 and (sub_ptr[unsafe_offset=1] | 0x20) == 121:
            # CLUSTER MYID (m=109,y=121,i=105,d=100)
            if cluster_enabled:
                writer.append_bulk_string_response(cluster[].node_id.unsafe_ptr(), 40)
            else:
                var dummy_id = String("0000000000000000000000000000000000000000")
                writer.append_bulk_string_response(dummy_id.unsafe_ptr(), 40)

        elif sub_len == 7 and sub0 == 107:
            # CLUSTER KEYSLOT key (k=107,e=101,y=121,s=115,l=108,o=111,t=116)
            if i + 2 < num_tokens:
                var key_tok = tokens[unsafe_offset=i + 2]
                var slot: Int
                if cluster_enabled:
                    slot = cluster[].keyslot(key_tok.ptr, key_tok.length)
                else:
                    slot = 0
                writer.append_int_response(Int64(slot))
                consumed += 1
            else:
                writer.append_error_response("ERR wrong number of arguments for 'cluster|keyslot' command")

        elif sub_len == 5 and sub0 == 115 and (sub_ptr[unsafe_offset=1] | 0x20) == 108:
            # CLUSTER SLOTS (s=115,l=108,o=111,t=116,s=115)
            var n_total = n_cluster_nodes
            var slots_arr = String("*") + String(n_total) + String("\r\n")
            writer.append_to_response(slots_arr.unsafe_ptr(), slots_arr.byte_length())
            # My node entry: *3 = [start, end, master_node_array]
            var slots_entry = String("*3\r\n")
            writer.append_to_response(slots_entry.unsafe_ptr(), slots_entry.byte_length())
            if cluster_enabled:
                writer.append_int_response(Int64(cluster[].my_slot_start))
                writer.append_int_response(Int64(cluster[].my_slot_end))
            else:
                writer.append_int_response(Int64(0))
                writer.append_int_response(Int64(16383))
            var slots_narr = String("*3\r\n")
            writer.append_to_response(slots_narr.unsafe_ptr(), slots_narr.byte_length())
            if cluster_enabled:
                writer.append_bulk_string_response(cluster[].my_host.unsafe_ptr(), cluster[].my_host_len)
                writer.append_int_response(Int64(cluster[].my_port))
                writer.append_bulk_string_response(cluster[].node_id.unsafe_ptr(), 40)
            else:
                var sh = String("127.0.0.1")
                writer.append_bulk_string_response(sh.unsafe_ptr(), sh.byte_length())
                writer.append_int_response(Int64(1974))
                var sdid = String("0000000000000000000000000000000000000000")
                writer.append_bulk_string_response(sdid.unsafe_ptr(), 40)
            # Peer entries
            if cluster_enabled:
                for pi in range(cluster[].peer_count):
                    var pe_hdr = String("*3\r\n")  # [start, end, master_node]
                    writer.append_to_response(pe_hdr.unsafe_ptr(), pe_hdr.byte_length())
                    writer.append_int_response(Int64(cluster[].peer_slot_starts[pi]))
                    writer.append_int_response(Int64(cluster[].peer_slot_ends[pi]))
                    var pna = String("*3\r\n")
                    writer.append_to_response(pna.unsafe_ptr(), pna.byte_length())
                    writer.append_bulk_string_response(cluster[].peer_host_ptr(pi), cluster[].peer_host_lens[pi])
                    writer.append_int_response(Int64(cluster[].peer_ports[pi]))
                    writer.append_bulk_string_response(cluster[].peer_node_id_ptr(pi), 40)

        elif sub_len == 6 and sub0 == 115 and (sub_ptr[unsafe_offset=1] | 0x20) == 104:
            # CLUSTER SHARDS (s=115,h=104,a=97,r=114,d=100,s=115) — Redis 7+ format
            var n_total = n_cluster_nodes
            var shards_arr = String("*") + String(n_total) + String("\r\n")
            writer.append_to_response(shards_arr.unsafe_ptr(), shards_arr.byte_length())
            # My node entry (inlined)
            var shdr = String("*4\r\n$5\r\nslots\r\n*2\r\n")
            writer.append_to_response(shdr.unsafe_ptr(), shdr.byte_length())
            if cluster_enabled:
                writer.append_int_response(Int64(cluster[].my_slot_start))
                writer.append_int_response(Int64(cluster[].my_slot_end))
            else:
                writer.append_int_response(Int64(0))
                writer.append_int_response(Int64(16383))
            var nhdr = String("$5\r\nnodes\r\n*1\r\n*18\r\n$2\r\nid\r\n")
            writer.append_to_response(nhdr.unsafe_ptr(), nhdr.byte_length())
            if cluster_enabled:
                writer.append_bulk_string_response(cluster[].node_id.unsafe_ptr(), 40)
            else:
                var sdid2 = String("0000000000000000000000000000000000000000")
                writer.append_bulk_string_response(sdid2.unsafe_ptr(), 40)
            var prt_hdr = String("$4\r\nport\r\n")
            writer.append_to_response(prt_hdr.unsafe_ptr(), prt_hdr.byte_length())
            if cluster_enabled:
                writer.append_int_response(Int64(cluster[].my_port))
            else:
                writer.append_int_response(Int64(1974))
            var tls_hdr = String("$8\r\ntls-port\r\n:0\r\n$2\r\nip\r\n")
            writer.append_to_response(tls_hdr.unsafe_ptr(), tls_hdr.byte_length())
            if cluster_enabled:
                writer.append_bulk_string_response(cluster[].my_host.unsafe_ptr(), cluster[].my_host_len)
            else:
                var sh2 = String("127.0.0.1")
                writer.append_bulk_string_response(sh2.unsafe_ptr(), sh2.byte_length())
            var ep_hdr = String("$8\r\nendpoint\r\n")
            writer.append_to_response(ep_hdr.unsafe_ptr(), ep_hdr.byte_length())
            if cluster_enabled:
                writer.append_bulk_string_response(cluster[].my_host.unsafe_ptr(), cluster[].my_host_len)
            else:
                var sh3 = String("127.0.0.1")
                writer.append_bulk_string_response(sh3.unsafe_ptr(), sh3.byte_length())
            var hn_hdr = String("$8\r\nhostname\r\n")
            writer.append_to_response(hn_hdr.unsafe_ptr(), hn_hdr.byte_length())
            if cluster_enabled:
                writer.append_bulk_string_response(cluster[].my_host.unsafe_ptr(), cluster[].my_host_len)
            else:
                var sh4 = String("127.0.0.1")
                writer.append_bulk_string_response(sh4.unsafe_ptr(), sh4.byte_length())
            var tail_hdr = String("$18\r\nreplication-offset\r\n:0\r\n$6\r\nhealth\r\n$6\r\nonline\r\n$4\r\nrole\r\n$6\r\nmaster\r\n")
            writer.append_to_response(tail_hdr.unsafe_ptr(), tail_hdr.byte_length())
            # Peer entries (inlined)
            if cluster_enabled:
                for pi in range(cluster[].peer_count):
                    var pe_shdr = String("*4\r\n$5\r\nslots\r\n*2\r\n")
                    writer.append_to_response(pe_shdr.unsafe_ptr(), pe_shdr.byte_length())
                    writer.append_int_response(Int64(cluster[].peer_slot_starts[pi]))
                    writer.append_int_response(Int64(cluster[].peer_slot_ends[pi]))
                    var pe_nhdr = String("$5\r\nnodes\r\n*1\r\n*18\r\n$2\r\nid\r\n")
                    writer.append_to_response(pe_nhdr.unsafe_ptr(), pe_nhdr.byte_length())
                    writer.append_bulk_string_response(cluster[].peer_node_id_ptr(pi), 40)
                    var pe_prt = String("$4\r\nport\r\n")
                    writer.append_to_response(pe_prt.unsafe_ptr(), pe_prt.byte_length())
                    writer.append_int_response(Int64(cluster[].peer_ports[pi]))
                    var pe_tls = String("$8\r\ntls-port\r\n:0\r\n$2\r\nip\r\n")
                    writer.append_to_response(pe_tls.unsafe_ptr(), pe_tls.byte_length())
                    writer.append_bulk_string_response(cluster[].peer_host_ptr(pi), cluster[].peer_host_lens[pi])
                    var pe_ep = String("$8\r\nendpoint\r\n")
                    writer.append_to_response(pe_ep.unsafe_ptr(), pe_ep.byte_length())
                    writer.append_bulk_string_response(cluster[].peer_host_ptr(pi), cluster[].peer_host_lens[pi])
                    var pe_hn = String("$8\r\nhostname\r\n")
                    writer.append_to_response(pe_hn.unsafe_ptr(), pe_hn.byte_length())
                    writer.append_bulk_string_response(cluster[].peer_host_ptr(pi), cluster[].peer_host_lens[pi])
                    var pe_tail = String("$18\r\nreplication-offset\r\n:0\r\n$6\r\nhealth\r\n$6\r\nonline\r\n$4\r\nrole\r\n$6\r\nmaster\r\n")
                    writer.append_to_response(pe_tail.unsafe_ptr(), pe_tail.byte_length())

        elif sub_len == 4 and sub0 == 109 and (sub_ptr[unsafe_offset=1] | 0x20) == 101:
            # CLUSTER MEET host port — register peer at runtime
            if i + 3 < num_tokens and cluster_enabled:
                var host_tok = tokens[unsafe_offset=i + 2]
                var port_tok = tokens[unsafe_offset=i + 3]
                var meet_port = 0
                for pci in range(port_tok.length):
                    var pb = port_tok.ptr[unsafe_offset=pci]
                    if pb >= 48 and pb <= 57:
                        meet_port = meet_port * 10 + Int(pb - 48)
                var pc = cluster[].peer_count
                if pc < 16:
                    cluster[].set_peer(pc, host_tok.ptr, host_tok.length,
                                                    meet_port, 0, 16383)
                    cluster[].peer_count = pc + 1
                    cluster[].cluster_epoch += 1
                    # Notify gossip of new peer if gossip is running
                    if is_not_null(cluster[].gossip_handle):
                        external_call["pion_gossip_set_peer", NoneType](
                            cluster[].gossip_handle,
                            Int32(pc),
                            host_tok.ptr.unsafe_bitcast[Int8](),
                            Int32(meet_port),
                        )
                consumed += 2
            elif i + 3 < num_tokens:
                consumed += 2
            writer.append_ok_response()

        elif sub_len == 6 and sub0 == 102 and (sub_ptr[unsafe_offset=1] | 0x20) == 111:
            # CLUSTER FORGET node-id — remove peer by node ID
            if i + 2 < num_tokens and cluster_enabled:
                var forget_tok = tokens[unsafe_offset=i + 2]
                var removed = False
                for pi in range(cluster[].peer_count):
                    # Compare 40-char hex node ID
                    var pid_ptr = cluster[].peer_node_id_ptr(pi)
                    var match_ok = True
                    if forget_tok.length != 40: match_ok = False
                    if match_ok:
                        for ci in range(40):
                            if (forget_tok.ptr[unsafe_offset=ci] | 0x20) != (pid_ptr[unsafe_offset=ci] | 0x20):
                                match_ok = False; break
                    if match_ok:
                        # Shift remaining peers down
                        var last = cluster[].peer_count - 1
                        for si in range(pi, last):
                            # Copy peer si+1 → si (host, port, slots, node_id)
                            cluster[].peer_host_lens[si] = cluster[].peer_host_lens[si+1]
                            cluster[].peer_ports[si] = cluster[].peer_ports[si+1]
                            cluster[].peer_slot_starts[si] = cluster[].peer_slot_starts[si+1]
                            cluster[].peer_slot_ends[si] = cluster[].peer_slot_ends[si+1]
                            cluster[].peer_health[si] = cluster[].peer_health[si+1]
                            for bi in range(64):
                                cluster[].peer_hosts[si * 64 + bi] = cluster[].peer_hosts[(si+1) * 64 + bi]
                            for bi in range(41):
                                cluster[].peer_node_ids[si * 41 + bi] = cluster[].peer_node_ids[(si+1) * 41 + bi]
                        cluster[].peer_count = last
                        cluster[].cluster_epoch += 1
                        removed = True
                        break
                if not removed:
                    writer.append_error_response("ERR Unknown node " + tokens[unsafe_offset=i+2].value())
                    consumed += 1
                    return consumed
                consumed += 1
            elif i + 2 < num_tokens:
                consumed += 1
            writer.append_ok_response()

        elif sub_len == 9 and sub0 == 114:
            # CLUSTER REPLICATE node-id — become a replica of the given node
            if i + 2 < num_tokens and cluster_enabled:
                var rep_tok = tokens[unsafe_offset=i + 2]
                var found_idx = -1
                for pi in range(cluster[].peer_count):
                    var pid_ptr = cluster[].peer_node_id_ptr(pi)
                    var match_ok = (rep_tok.length == 40)
                    if match_ok:
                        for ci in range(40):
                            if (rep_tok.ptr[unsafe_offset=ci] | 0x20) != (pid_ptr[unsafe_offset=ci] | 0x20):
                                match_ok = False; break
                    if match_ok: found_idx = pi; break
                if found_idx >= 0:
                    cluster[].is_replica = True
                    cluster[].primary_peer_idx = found_idx
                    cluster[].cluster_epoch += 1
                    writer.append_ok_response()
                else:
                    writer.append_error_response("ERR Unknown node for REPLICATE")
                consumed += 1
            elif i + 2 < num_tokens:
                consumed += 1
                writer.append_ok_response()
            else:
                writer.append_error_response("ERR wrong number of arguments for 'cluster|replicate'")

        elif sub_len == 8 and sub0 == 102 and (sub_ptr[unsafe_offset=1] | 0x20) == 97:
            # CLUSTER FAILOVER [FORCE] — promote replica to primary (N3)
            if not cluster_enabled:
                writer.append_error_response("ERR cluster not enabled")
            elif not cluster[].is_replica:
                writer.append_error_response("ERR not a replica, cannot failover")
            else:
                # Check for FORCE flag
                var force = False
                if i + 2 < num_tokens:
                    var _ff = tokens[unsafe_offset=i+2]
                    if _ff.length == 5 and (_ff.ptr[unsafe_offset=0]|0x20) == 102:  # f=force
                        force = True
                        consumed += 1

                # Health check: primary must be in FAIL state unless FORCE
                var primary_healthy = True
                var _ppidx = cluster[].primary_peer_idx
                if _ppidx >= 0 and _ppidx < 16:
                    var _ph = cluster[].peer_health[_ppidx]
                    if _ph == 2:  # FAIL
                        primary_healthy = False

                if primary_healthy and not force:
                    writer.append_error_response("ERR primary is up, use CLUSTER FAILOVER FORCE to override")
                else:
                    # Stop replica receiver
                    if is_not_null(cluster[].repl_replica_handle):
                        external_call["pion_repl_replica_stop", NoneType](cluster[].repl_replica_handle)
                        cluster[].repl_replica_handle = null_ptr[NoneType, MutUntrackedOrigin]()
                    # Free drain buffer
                    if is_not_null(cluster[].repl_drain_buf):
                        cluster[].repl_drain_buf.unsafe_free()
                        cluster[].repl_drain_buf = null_ptr[UInt8, MutUntrackedOrigin]()

                    # Promote to primary
                    cluster[].is_replica = False
                    cluster[].primary_peer_idx = -1
                    cluster[].cluster_epoch += 1
                    print("N3 FAILOVER: promoted to primary (epoch=" + String(cluster[].cluster_epoch) + ")")

                    # Start primary replicator (listen for future replicas)
                    if is_not_null(cluster[].wal_ptr) and is_null(cluster[].repl_primary_handle):
                        var repl_port = cluster[].server_port + REPL_PORT_OFFSET
                        var _blk = external_call["pion_repl_primary_create",
                                                   Pointer[NoneType, MutUntrackedOrigin]](
                            cluster[].wal_ptr.unsafe_offset(64),  # WAL data section (after 64B header)
                            cluster[].wal_ptr.unsafe_bitcast[UInt64]().unsafe_offset(1),  # &tail_offset (byte 8)
                            Int32(repl_port),
                        )
                        if is_not_null(_blk):
                            var _rc = external_call["pion_repl_primary_start", Int32](_blk)
                            if _rc == 0:
                                cluster[].repl_primary_handle = _blk
                                print("N3 FAILOVER: primary replicator started on port " + String(repl_port))

                    # Save topology
                    cluster[].save_topology("pion-nodes.conf")
                    writer.append_ok_response()

        elif sub_len == 5 and sub0 == 114:
            # CLUSTER RESET (r=114,e=101,s=115,e=101,t=116)
            if cluster_enabled:
                cluster[].is_replica = False
                cluster[].primary_peer_idx = -1
                cluster[].cluster_epoch += 1
            writer.append_ok_response()

        elif sub_len == 8 and sub0 == 114 and (sub_ptr[unsafe_offset=4]|0x20) == 99:
            # CLUSTER REPLICAS node-id → *0 (no replicas in shared-nothing)
            if i + 2 < num_tokens: consumed += 1
            writer.append_empty_array_response()

        elif sub_len == 13 and sub0 == 103:
            # CLUSTER GETKEYSINSLOT slot count — iterate keyspace for matching keys
            if i + 3 < num_tokens:
                var gk_slot = strict_atol(tokens[unsafe_offset=i + 2].value())
                var gk_count = strict_atol(tokens[unsafe_offset=i + 3].value())
                consumed += 2
                # Iterate all 8 shards to find keys in this slot
                var found = List[String]()
                for shard_idx in range(8):
                    if len(found) >= gk_count:
                        break
                    var shard = keyspace[].shards.unsafe_offset(shard_idx)
                    for slot_idx in range(shard[].capacity):
                        if len(found) >= gk_count:
                            break
                        var m = shard[].metadata[unsafe_offset=slot_idx]
                        if m != 0x80 and m != 0xFF:  # not EMPTY, not DELETED
                            var key = shard[].keys[unsafe_offset=slot_idx]
                            if not key.is_none():
                                var sso_buf = alloc[UInt8](24)
                                var kp = key.as_string_safe(sso_buf)
                                var kl = key.string_len()
                                var ks = cluster[].keyslot(kp, kl)
                                if ks == gk_slot:
                                    found.append(bytes_to_string(kp, kl))
                                sso_buf.unsafe_free()
                var arr_hdr = String("*") + String(len(found)) + String("\r\n")
                writer.append_to_response(arr_hdr.unsafe_ptr(), arr_hdr.byte_length())
                for ki in range(len(found)):
                    writer.append_bulk_string_response(found[ki].unsafe_ptr(), found[ki].byte_length())
            else:
                writer.append_error_response("ERR wrong number of arguments for 'cluster getkeysinslot'")

        elif sub_len == 15 and sub0 == 99:
            # CLUSTER COUNTKEYSINSLOT slot — count keys in slot
            if i + 2 < num_tokens:
                var ck_slot = strict_atol(tokens[unsafe_offset=i + 2].value())
                consumed += 1
                var ck_count: Int64 = 0
                for shard_idx in range(8):
                    var shard = keyspace[].shards.unsafe_offset(shard_idx)
                    for slot_idx in range(shard[].capacity):
                        var m = shard[].metadata[unsafe_offset=slot_idx]
                        if m != 0x80 and m != 0xFF:
                            var key = shard[].keys[unsafe_offset=slot_idx]
                            if not key.is_none():
                                var sso_buf = alloc[UInt8](24)
                                var kp = key.as_string_safe(sso_buf)
                                var kl = key.string_len()
                                var ks = cluster[].keyslot(kp, kl)
                                if ks == ck_slot:
                                    ck_count += 1
                                sso_buf.unsafe_free()
                writer.append_int_response(ck_count)
            else:
                writer.append_int_response(0)

        elif sub_len == 8 and sub0 == 97 and (sub_ptr[unsafe_offset=1]|0x20)==100 and (sub_ptr[unsafe_offset=2]|0x20)==100:
            # CLUSTER ADDSLOTS slot [slot ...]
            var si2 = i + 2
            while si2 < num_tokens and tokens[unsafe_offset=si2].marker != 0:
                var sl_str = tokens[unsafe_offset=si2].value()
                var sl_val = atol(sl_str)
                if sl_val >= 0 and sl_val < 16384:
                    cluster[].add_slot(sl_val)
                si2 += 1
                consumed += 1
            cluster[].cluster_epoch += 1
            writer.append_ok_response()

        elif sub_len == 8 and sub0 == 100 and (sub_ptr[unsafe_offset=1]|0x20)==101 and (sub_ptr[unsafe_offset=2]|0x20)==108:
            # CLUSTER DELSLOTS slot [slot ...]
            var si3 = i + 2
            while si3 < num_tokens and tokens[unsafe_offset=si3].marker != 0:
                var sl_str = tokens[unsafe_offset=si3].value()
                var sl_val = atol(sl_str)
                if sl_val >= 0 and sl_val < 16384:
                    cluster[].del_slot(sl_val)
                si3 += 1
                consumed += 1
            cluster[].cluster_epoch += 1
            writer.append_ok_response()

        # §3: CLUSTER SETSLOT <slot> IMPORTING|MIGRATING|NODE|STABLE [node-id]
        elif sub_len == 7 and sub0 == 115 and (sub_ptr[unsafe_offset=1]|0x20)==101 and (sub_ptr[unsafe_offset=2]|0x20)==116:
            # "setslot" (s=115,e=101,t=116,s=115,l=108,o=111,t=116)
            if i + 3 < num_tokens:
                var slot_str = tokens[unsafe_offset=i + 2].value()
                var _ss_slot: Int = 0
                var _ssp = slot_str.unsafe_ptr().unsafe_bitcast[UInt8]()
                for _si in range(slot_str.byte_length()):
                    if _ssp[unsafe_offset=_si] >= 48 and _ssp[unsafe_offset=_si] <= 57:
                        _ss_slot = _ss_slot * 10 + Int(_ssp[unsafe_offset=_si] - 48)
                var action_tok = tokens[unsafe_offset=i + 3]
                var act_p = action_tok.ptr; var act_l = action_tok.length
                consumed += 2
                if act_l == 9 and (act_p[unsafe_offset=0]|0x20)==105:  # IMPORTING
                    if i + 4 < num_tokens:
                        consumed += 1  # consume node-id
                        # Find peer by node-id
                        var nid_tok = tokens[unsafe_offset=i + 4]
                        var peer_idx = -1
                        for pi2 in range(cluster[].peer_count):
                            var match2 = True
                            var pbase2 = pi2 * 41
                            for ci2 in range(min(nid_tok.length, 40)):
                                if cluster[].peer_node_ids.unsafe_ptr()[unsafe_offset=pbase2 + ci2] != nid_tok.ptr[unsafe_offset=ci2]:
                                    match2 = False; break
                            if match2: peer_idx = pi2; break
                        cluster[].set_slot_importing(_ss_slot, peer_idx)
                    writer.append_ok_response()
                elif act_l == 9 and (act_p[unsafe_offset=0]|0x20)==109:  # MIGRATING
                    if i + 4 < num_tokens:
                        consumed += 1
                        var nid_tok = tokens[unsafe_offset=i + 4]
                        var peer_idx = -1
                        for pi2 in range(cluster[].peer_count):
                            var match2 = True
                            var pbase2 = pi2 * 41
                            for ci2 in range(min(nid_tok.length, 40)):
                                if cluster[].peer_node_ids.unsafe_ptr()[unsafe_offset=pbase2 + ci2] != nid_tok.ptr[unsafe_offset=ci2]:
                                    match2 = False; break
                            if match2: peer_idx = pi2; break
                        cluster[].set_slot_migrating(_ss_slot, peer_idx)
                    writer.append_ok_response()
                elif act_l == 4 and (act_p[unsafe_offset=0]|0x20)==110:  # NODE
                    if i + 4 < num_tokens:
                        consumed += 1
                        cluster[].assign_slot_to_node(_ss_slot, tokens[unsafe_offset=i + 4].ptr)
                    writer.append_ok_response()
                elif act_l == 6 and (act_p[unsafe_offset=0]|0x20)==115 and (act_p[unsafe_offset=1]|0x20)==116:  # STABLE
                    cluster[].clear_slot_state(_ss_slot)
                    writer.append_ok_response()
                else:
                    writer.append_error_response("ERR CLUSTER SETSLOT action must be IMPORTING|MIGRATING|NODE|STABLE")
            else:
                writer.append_error_response("ERR wrong number of arguments for CLUSTER SETSLOT")

        elif sub_len == 5 and sub0 == 115 and (sub_ptr[unsafe_offset=1] | 0x20) == 116:
            # CLUSTER STATS (s=115,t=116,a=97,t=116,s=115) — gh #44
            #
            # Returns INFO-style key:value telemetry for the W7 Mac-cluster
            # operator workflow: per-node KV.PREFIX residence (cross-worker
            # via shared directory) + LOCAL V-Store / ATTEND.PREFIX counters
            # for the worker that handled this query. Multi-worker setups
            # under `--kvcache -w >1` see only the handling worker's slice
            # for V-Store / ATTEND counters; KV.PREFIX residence is full.
            var st_cap = 2048
            var st_buf = alloc[UInt8](st_cap)
            var st_mb = st_buf
            var st_off = 0
            def _st_sl(s: StringLiteral, b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                var sl = s.byte_length(); unsafe_memcpy(dest=b.unsafe_offset(o), src=s.unsafe_ptr(), count=sl); o += sl
            def _st_in(n: Int, b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                o += format_int_to_buf(b.unsafe_offset(o), 0, Int64(n))
            def _st_nl(b: Pointer[UInt8, MutUntrackedOrigin], mut o: Int):
                b[unsafe_offset=o] = 13; b[unsafe_offset=o + 1] = 10; o += 2

            _st_sl("# Cluster Stats", st_mb, st_off); _st_nl(st_mb, st_off)
            _st_sl("cluster_stats_local_worker:", st_mb, st_off)
            _st_in(worker_id, st_mb, st_off); _st_nl(st_mb, st_off)
            _st_sl("cluster_stats_total_workers:", st_mb, st_off)
            _st_in(num_workers, st_mb, st_off); _st_nl(st_mb, st_off)

            # Per-worker KV.PREFIX residence — walks the shared directory and
            # bucketises entries by owner_worker_id. Atomic ACQUIRE on
            # `published[i]` matches the lookup path so we only count visible
            # rows. Cross-worker accurate.
            var kv_total = 0
            var per_worker_kv = Array[Int, 64](fill=0)  # safe upper bound
            var have_dir = False
            if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].vstore_directory):
                var dir_ptr = Pointer[VStoreDirectory, MutUntrackedOrigin](
                    unsafe_from_address=Int(shared_hnsw[].vstore_directory))
                if is_not_null(dir_ptr):
                    have_dir = True
                    var entries = dir_ptr[].entries
                    var published = dir_ptr[].published
                    for di in range(MAX_DIR_ENTRIES):
                        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                                published.unsafe_offset(di), UInt64(0)) == 0:
                            continue
                        if not entries[unsafe_offset=di].active:
                            continue
                        kv_total += 1
                        var ow = entries[unsafe_offset=di].owner_worker_id
                        if ow >= 0 and ow < 64:
                            per_worker_kv[ow] = per_worker_kv[ow] + 1

            _st_sl("# KV.PREFIX residence (cross-worker via shared directory)", st_mb, st_off)
            _st_nl(st_mb, st_off)
            _st_sl("kv_prefix_active_total:", st_mb, st_off)
            if have_dir:
                _st_in(kv_total, st_mb, st_off)
            else:
                _st_in(0, st_mb, st_off)
                # No directory means --kvcache is off; mark explicitly so an
                # operator scraping this output can tell "0 because empty"
                # from "0 because feature off".
                _st_nl(st_mb, st_off)
                _st_sl("kv_prefix_directory_enabled:0", st_mb, st_off)
            _st_nl(st_mb, st_off)
            if have_dir:
                _st_sl("kv_prefix_directory_enabled:1", st_mb, st_off)
                _st_nl(st_mb, st_off)
                var nw = num_workers if num_workers > 0 else 1
                if nw > 64:
                    nw = 64
                for wi in range(nw):
                    _st_sl("kv_prefix_active_worker_", st_mb, st_off)
                    _st_in(wi, st_mb, st_off)
                    st_mb[unsafe_offset=st_off] = 58; st_off += 1  # ':'
                    _st_in(per_worker_kv[wi], st_mb, st_off)
                    _st_nl(st_mb, st_off)

            # LOCAL V-Store / ATTEND.* counters. Single worker sees its full
            # state; under -w >1 these are partial — the response makes that
            # explicit so operators don't double-count.
            #
            # Note: `local_attend_*` here is the legacy ATTEND.* (HNSW) path.
            # The Metal-backed ATTEND.PREFIX.* path lives in the C engine and
            # has no Mojo-side counters today; track those via KV.PREFIX
            # residence (every Stage-2 session registers a KV.PREFIX entry
            # first), or wait for the C-side counter export follow-up.
            _st_sl("# V-Store / ATTEND.* (LOCAL worker only)", st_mb, st_off)
            _st_nl(st_mb, st_off)
            _st_sl("local_vstore_enabled:", st_mb, st_off)
            _st_in(1 if v_store.enabled else 0, st_mb, st_off); _st_nl(st_mb, st_off)
            _st_sl("local_vstore_sessions:", st_mb, st_off)
            _st_in(v_store.session_count, st_mb, st_off); _st_nl(st_mb, st_off)
            _st_sl("local_attend_enabled:", st_mb, st_off)
            _st_in(1 if attn_idx.enabled else 0, st_mb, st_off); _st_nl(st_mb, st_off)
            _st_sl("local_attend_sessions:", st_mb, st_off)
            _st_in(attn_idx.session_count, st_mb, st_off); _st_nl(st_mb, st_off)
            _st_sl("local_attend_total_tokens_stored:", st_mb, st_off)
            _st_in(attn_idx.total_tokens_stored, st_mb, st_off); _st_nl(st_mb, st_off)
            _st_sl("local_attend_total_queries:", st_mb, st_off)
            _st_in(attn_idx.total_queries, st_mb, st_off); _st_nl(st_mb, st_off)
            _st_sl("local_attend_total_query_hits:", st_mb, st_off)
            _st_in(attn_idx.total_query_hits, st_mb, st_off); _st_nl(st_mb, st_off)
            var local_misses = attn_idx.total_queries - attn_idx.total_query_hits
            if local_misses < 0: local_misses = 0
            _st_sl("local_attend_total_query_misses:", st_mb, st_off)
            _st_in(local_misses, st_mb, st_off); _st_nl(st_mb, st_off)

            writer.append_bulk_string_response(st_mb, st_off)
            st_buf.unsafe_free()

        else:
            writer.append_error_response("ERR unknown subcommand for cluster command")
    else:
        writer.append_error_response("ERR wrong number of arguments for 'cluster' command")
    return consumed
