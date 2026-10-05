from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.vec_tomb import VecTomb
from std.memory.unsafe_pointer import UnsafePointer
from std.memory import alloc, unsafe_memcpy, unsafe_memset, stack_allocation
from std.collections import Array, List

# IVF-PQ disabled: recall@100 = 0.59 for 50K 1536-dim vectors (a measured dead end).
# gh #87.1: ENABLE_IVF_PQ removed (was permanently False).

from src.network.resp3 import RESP3Parser, RESP3Token, TOKENS_OVERFLOW, MAX_CMD_TOKENS, MAX_CMD_ENDS
from src.network.dispatcher import CommandDispatcher
from src.network.server import TCPServer
from src.network.response_writer import ResponseWriter
from src.network.fast_path import cmd_matches_3, cmd_matches_4, cmd_matches_5, cmd_matches_6, cmd_matches_7, cmd_matches_8, cmd_eq, _get_now_ns
from src.common.container_free import remove_and_free, hash_get_live
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.skip_list import SlabSkipList
from src.common.list import SlabList
from src.memory.slab_allocator import SlabAllocator
from src.common.value import GenericValue, ValueType
from src.common.utils import strict_atol, format_int_to_buf, format_float_to_buf, int_string_len, arg_eq, parse_int64_strict, parse_redis_double, DOUBLE_VALUE, DOUBLE_RANGE, DOUBLE_LONG, set_expiry, SETEXP_INVALID, SETEXP_EXPIRED
from src.common.metrics import ValueLedger
from src.common.config import PionConfig
from src.common.geohash import geohash_encode, geohash_decode, GeoHashBits, GEO_STEP_MAX
from src.memory.object_pool import ObjectPool
from src.vector.hnsw import HNSWGraph, SharedHNSWView
# gh #87.1: src/vector/ivf_pq.mojo deleted.
from src.network.ai_gateway import FLAREGateway
from src.network.semantic_cache import SemanticCache
from src.network.llm_client import LLMClient
from std.atomic import Atomic, Ordering
from src.io.wal import WAL, gv_bytes
from src.network.fast_path import _get_now_ns
from src.io.snapshot import SnapshotEngine
from src.network.raft import RaftNode
from src.common.lock_free import LockFreeRingBuffer, ShardQueryBus
from src.common.heap import HeapNode
from src.network.cluster import ClusterState
from src.network.inference_bridge import InferenceBridge, InferenceResponse, INFER_MSG_EMBED, INFER_MSG_GENERATE, INFER_MSG_LOAD_MODEL, INFER_STATUS_OK, INFER_STATUS_ERROR, INFER_RECV_BUF_SIZE
from src.network.metal_attention_engine import MetalAttentionEngine
from src.network.cuda_attention_engine import CudaAttentionEngine
from src.network.knn_lm import KNNLMIndex
from src.network.pkm import PKMIndex
from std.ffi import external_call
from std.sys.info import CompilationTarget
from std.math import log, sin, cos, sqrt, asin, pi
from src.common.bitmap import getbit, setbit, bitcount, SetBitResult
from src.common.hll import hll_add, hll_count, hll_merge, HLL_REGISTERS

# Command modules (Phase 1 extraction)
from src.commands.transaction import TransactionState, QueuedCommand, handle_multi, handle_exec_start, handle_discard, handle_watch, handle_unwatch, tx_queue_has_denyoom
from src.commands.command_table import command_exists, command_arity, command_is_write, command_is_denyoom, command_is_noscript, command_hidden_from_monitor, command_touches_keyspace, command_monitor_first, PION_COMMAND_COUNT
from src.commands.tenant import TenantTable, tenant_keyspec, apply_tenant_rewrite, TENANT_SCRATCH_CAP, MAX_TENANT_NAME
from src.commands.stream import handle_xadd, handle_xlen, handle_xack, handle_xdel, handle_xread, handle_xtrim, handle_xinfo, handle_xrange, handle_xgroup, handle_xclaim, handle_xpending, handle_xrevrange, handle_xautoclaim, handle_xreadgroup, BlockedReaderRegistry, write_xread_reply
from src.commands.pubsub import PubSubRegistry, handle_pubsub, handle_publish_kind, handle_subscribe_kind, handle_unsubscribe_kind, pubsub_drain, KIND_CHANNEL, KIND_PATTERN, KIND_SHARD
from src.commands.ttl import handle_expire, handle_pexpire, handle_expireat, handle_pexpireat, handle_ttl, handle_pttl, handle_persist
# Command modules (Phase 2 extraction)
from src.commands.list import handle_lindex, handle_lset, handle_linsert, handle_lrem, handle_ltrim, handle_lpos, handle_lmove
from src.commands.mpop import parse_mpop
from src.commands.blocking import BlockedClientRegistry, parse_block_timeout, new_blocked_client
from src.commands.bitmap import handle_bitop, handle_bitpos, handle_bitcount, handle_pfmerge, handle_bitfield, handle_bitfield_ro, handle_pfselftest, handle_pfdebug
from src.commands.monitor import MonitorRegistry, monitor_line
from src.network.vector_ingest import ingest_hash_vector, record_all_dead, ingest_whole_hash, hash_addr, after_rename
from src.commands.replication_cmds import handle_role, handle_replicaof, handle_failover, handle_sync, handle_replconf
from src.commands.key_mgmt import handle_type, handle_rename, handle_renamenx, handle_copy, handle_object, handle_sort, handle_sort_ro, handle_scan, handle_keys, handle_randomkey, handle_touch, handle_wait, handle_waitaof, ParkedWaits
from src.commands.set import handle_scard, handle_sismember, handle_smismember, handle_smembers, handle_srandmember, handle_srem, handle_smove, handle_sinter, handle_sinterstore, handle_sintercard, handle_sunion, handle_sunionstore, handle_sdiff, handle_sdiffstore, handle_sscan
from src.commands.geo import handle_geoadd, handle_geopos, handle_geodist, handle_geohash, handle_georadius, handle_geosearch, handle_geosearchstore, handle_georadiusbymember
from src.commands.hash import handle_hmget, handle_hgetall, handle_hkeys, handle_hvals, handle_hlen, handle_hdel, handle_hexists, handle_hincrby, handle_hincrbyfloat, handle_hrandfield, handle_hscan, handle_hsetnx, handle_hexpire, handle_hpexpire, handle_hexpireat, handle_hpexpireat, handle_httl, handle_hpttl, handle_hpersist, handle_hexpiretime, handle_hpexpiretime
from src.commands.admin import handle_xgpu_info, handle_ping, handle_echo, handle_hello, handle_flushall, handle_save, handle_bgsave, handle_lastsave, handle_info, handle_pion_stats, handle_config, handle_quit, handle_auth, handle_flushdb, handle_dbsize, handle_select, handle_swapdb, handle_move, handle_bgrewriteaof, handle_command, handle_debug, handle_slowlog, handle_latency, handle_memory, handle_module, handle_acl, handle_reset, handle_client, handle_time, handle_lolwut
from src.commands.lua_engine import LuaEngine, handle_eval, handle_evalsha, handle_script, handle_function, handle_fcall
from src.commands.cluster import handle_cluster
from src.commands.migrate import handle_dump, handle_restore, handle_migrate
from src.commands.string_kv import handle_incrby, handle_decrby, handle_incrbyfloat, handle_append, handle_strlen, handle_getset, handle_getdel, handle_getex, handle_setnx, handle_setex, handle_psetex, handle_msetnx, handle_msetex, handle_getrange, handle_substr, handle_setrange, handle_expiretime, handle_pexpiretime, handle_unlink, handle_lcs
from src.commands.ai import handle_ai_chat, handle_ai_flare, handle_ai_complete, handle_ai_semantic_cache, handle_ai_embed, handle_ai_generate, handle_ai_loadmodel, handle_ai_memory
from src.commands.sorted_set import zmpop_pop, handle_zrem, handle_zcard, handle_zrank, handle_zrevrank, handle_zscore, handle_zcount, handle_zincrby, handle_zrange, handle_zrevrange, handle_zrangebyscore, handle_zrevrangebyscore, handle_zunion, handle_zinter, handle_zunionstore, handle_zinterstore, handle_zlexcount, handle_zrangebylex, handle_zrevrangebylex, handle_zpopmax, handle_zmpop, handle_zrandmember, handle_zmscore, handle_zscan, handle_zrangestore, handle_zintercard, handle_zdiff, handle_zdiffstore, handle_zremrangebylex, handle_zremrangebyrank, handle_zremrangebyscore
from src.commands.vector import handle_ft_info, handle_ft_dropindex, handle_ft_optimize, handle_ft_create, handle_ft_addtext, handle_ft_searchtext, handle_ft_search, handle_ft_hybrid, write_ft_search_response
from src.commands.kv_cache import handle_kv_store, handle_kv_fetch, handle_kv_info
from src.commands.attend import handle_attend_create, handle_attend_store, handle_attend_query, handle_attend_finalize, handle_attend_info, ATTEND_MAX_K
from src.commands.v_store import handle_v_create, handle_v_storebatch, handle_v_fetch, handle_v_info, handle_v_snapshot, handle_v_restore, handle_v_commit
from src.commands.state import handle_state_alloc, handle_state_write, handle_state_read, handle_state_free, handle_state_info
from src.network.state_store import StateStore
from src.commands.kv_prefix import handle_kv_prefix_register, handle_kv_prefix_lookup, handle_kv_prefix_info, handle_kv_prefix_save, handle_kv_prefix_commit, handle_kv_prefix_drop, handle_kv_prefix_owner, handle_kv_prefix_warm, handle_kv_prefix_blocks, handle_kv_prefix_membership
from src.commands.attend_prefix import handle_attend_prefix_store, handle_attend_prefix_query, handle_attend_prefix_query_fused, handle_attend_prefix_query_sparse, handle_attend_prefix_query_sparse_auto, handle_attend_prefix_query_sparse_auto_fused, handle_attend_prefix_lookup
from src.commands.ssm_prefix import handle_ssm_prefix_store, handle_ssm_prefix_fetch, handle_ssm_prefix_drop, ssm_durability_startup
from src.commands.moe_expert import handle_moe_expert_fetch, handle_moe_expert_prefetch, handle_moe_expert_pin, handle_moe_expert_unpin, handle_moe_expert_info, handle_moe_expert_stats, handle_moe_expert_hist, handle_moe_expert_prune, handle_moe_expert_load
from src.network.moe_expert_tier import MoEExpertTier
from src.commands.knn_lm import (
    handle_ai_knn_lm_create, handle_ai_knn_lm_store, handle_ai_knn_lm_storebatch,
    handle_ai_knn_lm_query, handle_ai_knn_lm_info, handle_ai_knn_lm_drop,
)
from src.commands.pkm import (
    handle_neuron_pkm_create, handle_neuron_pkm_setkeys, handle_neuron_pkm_setvals,
    handle_neuron_pkm_query, handle_neuron_pkm_ffn, handle_neuron_pkm_info,
    handle_neuron_pkm_drop,
)
from src.network.v_store import VStoreIndex, VStoreDirectory
from src.commands.route import handle_ai_route_register, handle_ai_route_update, handle_ai_route, handle_ai_route_remove, handle_ai_route_info
from src.commands.speculative import handle_rag_speculate_enable, handle_rag_query, handle_rag_speculate_info
from src.commands.vset import handle_vadd, handle_vsim, handle_vcard, handle_vdim, handle_vrem, handle_vemb, handle_vismember, handle_vsetattr, handle_vgetattr, handle_vinfo, handle_vrandmember, handle_vlinks, handle_vrange
from src.network.semantic_router import SemanticRouter
from src.network.speculative_rag import SpeculativeRAG
from src.network.kv_cache_store import KVCacheStore
from src.network.layer_store import LayerStore
from src.network.attention_index import AttentionIndex, ATTEND_QUERY_EF
from src.network.binary_protocol import (
    parse_binary_request, build_binary_response, build_binary_response_ok, build_binary_response_miss,
    parse_layer_store_request, parse_layer_fetch_request,
    parse_attend_create, parse_attend_store, parse_attend_query,
    parse_attend_prefix_query_fused,
    BINARY_MAGIC, BINARY_HEADER_SIZE, BINARY_RESP_HEADER_SIZE,
    CMD_LAYER_STORE, CMD_LAYER_FETCH, CMD_LAYER_FETCH_BATCH, CMD_PING,
    CMD_ATTEND_CREATE, CMD_ATTEND_STORE, CMD_ATTEND_FINALIZE, CMD_ATTEND_QUERY,
    CMD_ATTEND_PREFIX_QUERY_FUSED, CMD_AUTH,
    STATUS_OK, STATUS_MISS, STATUS_ERROR, STATUS_COLDMISS,
    read_u16_le, write_u16_le, write_u32_le,
)

comptime MAX_SHARD_K = 100  # matches lock_free.mojo MAX_SHARD_K


@always_inline
def _binary_send_all(server: TCPServer, fd: Int32,
                     ptr: UnsafePointer[UInt8, MutUntrackedOrigin], total: Int):
    """Blocking send-all for binary-lane bodies that exceed binary_resp_buf.
    Mirrors the gh #76 SSM.PREFIX.FETCH writev loop: server.send may transmit
    partially (or return EAGAIN on a non-blocking socket) for multi-MB blobs,
    so spin-retry until `total` bytes are out. Used by the gh #91 large-layer
    LAYER_FETCH path to stream the 7-byte header + tensor blob directly."""
    var sent = 0
    var stalls = 0
    while sent < total:
        var n = server.send(fd, ptr + sent, total - sent)
        if n > 0:
            sent += n
            stalls = 0
        elif n < 0:
            # gh #404 review: only EAGAIN is worth waiting for. send() has
            # SIGPIPE suppressed, so a client that went away mid-body returns
            # EPIPE/ECONNRESET on every retry — this loop used to sleep on it
            # forever, and the worker with it. And a client that stops reading
            # may not hold the worker for more than ~10 s of no progress.
            var errno_val: Int32
            comptime if CompilationTarget.is_linux():
                errno_val = external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
            else:
                errno_val = external_call["__error", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
            if errno_val != 35 and errno_val != 11:  # 35 = EAGAIN (macOS), 11 = EAGAIN (Linux)
                break
            stalls += 1
            if stalls > 100000:
                break
            _ = external_call["usleep", Int32](Int32(100))  # EAGAIN: 100µs backoff
        else:
            break  # n == 0: connection closed


struct SlowPathHandler:
    var dispatcher: CommandDispatcher
    var parser: RESP3Parser
    # gh #166: the RESP token table and the two command-boundary tables, one set
    # per worker, on the heap. They used to be stack locals in
    # process_slow_path; at MAX_CMD_TOKENS = 2048 that is ~112 KB of a ~512 KB
    # parallelize worker stack, in a frame that also hosts the EXEC replay loop
    # (kept iterative for the same reason — gh #94).
    var tokens_buf: UnsafePointer[RESP3Token, MutUntrackedOrigin]
    var cmd_ends_buf: UnsafePointer[Int, MutUntrackedOrigin]
    var cmd_byte_ends_buf: UnsafePointer[Int, MutUntrackedOrigin]
    var replay_ends_buf: UnsafePointer[Int, MutUntrackedOrigin]
    var replay_byte_ends_buf: UnsafePointer[Int, MutUntrackedOrigin]
    var keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]
    var skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]
    var shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]
    var scratch_ids: List[Int]
    var scratch_dists: List[Float32]
    var worker_id: Int    # V18
    var snapshot_engine: SnapshotEngine
    var last_save_time: Int64
    var num_workers: Int  # V18
    var shard_query_seq: UnsafePointer[UInt64, MutUntrackedOrigin]
    # gh #207: the P3 multi-query staging machinery (p3_batch_start_node
    # cached-start + p3_staged_* deferred-pair fields, flush_staged_search,
    # reset_p3_tick) is DELETED. VectorDBBench is closed-loop per connection:
    # at C=10 a second query never arrives within one kqueue tick, so the
    # cached start node stayed -1 forever and flush_staged_search had no call
    # sites. Same family as the gh #85 KVRequestBus deletion. Do not rebuild —
    # per-tick query batching cannot move the C=10 gate by construction.
    # gh #87.1: IVF-PQ scratch buffers (ivf_adc_table / _u8 / centroid_dists /
    # centroid_order / heap_d / heap_i / query_int8) removed alongside the
    # disabled IVF-PQ path. ~89 KB / worker reclaimed.
    # Phase 2: Semantic cache (per-worker HNSWGraph + EmbeddingClient)
    var scache: SemanticCache
    # M14 Phase 1: KV cache store (per-worker HNSW-indexed blob store)
    var kvcache: KVCacheStore
    # M14 Phase 2: Layer-granular KV store + binary protocol
    var layer_store: LayerStore
    var binary_resp_buf: UnsafePointer[UInt8, MutUntrackedOrigin]
    # M14 Phase 3: Attention index (per-session, per-layer HNSW for token KV pairs)
    var attn_idx: AttentionIndex
    # V-Store: token-ID-indexed V cache (no HNSW, GPU keeps K)
    var v_store: VStoreIndex
    # A4 (issue #31): per-request fixed-size state cache (SWA tail, SSM intermediates).
    # Backed by --kvcache flag — paper §3.6.1 couples KV cache and state cache.
    var state_store: StateStore
    # M13: Semantic router (HNSW-indexed inference load balancer)
    var router: SemanticRouter
    # M9: Speculative RAG (branch prediction for RAG pipeline)
    var spec_rag: SpeculativeRAG
    # Blocked XREAD readers (per-worker)
    var blocked_readers: BlockedReaderRegistry
    # Pub/Sub registry (per-worker) + cross-worker broadcast ring
    var pubsub: PubSubRegistry
    # Transaction state (per-fd MULTI/EXEC queuing)
    var tx_state: TransactionState
    # Phase 4: LLM client for AI.CHAT (HTTP proxy to MAX Serve / OpenAI-compatible)
    var llm_client: LLMClient
    # Phase 4: per-worker scratch buffer for LLM output (1MB)
    var llm_out_buf: UnsafePointer[UInt8, MutUntrackedOrigin]
    # FLARE gateway: in-process HNSW KB + mid-generation retrieval loop
    var flare: FLAREGateway
    # Phase 5: Cluster state (shared, read-only after init)
    var cluster: UnsafePointer[ClusterState, MutUntrackedOrigin]
    # TTL: per-worker map of key → expiry_ns (GenericValue.INT). Shared with FastPathHandler.
    var ttl_map: UnsafePointer[SlabHashMap, MutUntrackedOrigin]
    # T3.4 async shard deferred responses: up to 16 concurrent deferred FT.SEARCH queries.
    # When FT.SEARCH posts shard queries, it stores state here instead of spin-waiting.
    # drain_deferred_shard_responses() is called each engine-loop iteration to check results.
    var deferred_fds:      Array[Int32, 16]   # client fd for each deferred query
    var deferred_seqs:     Array[UInt64, 16]  # query_seq (for result matching)
    var deferred_ks:       Array[Int32, 16]   # k for final top-k merge
    var deferred_n_shards: Array[Int32, 16]   # n_shards for this query
    var deferred_active:   Array[UInt32, 16]  # bitmask of shard IDs queried (incl own)
    var deferred_done:     Array[UInt32, 16]  # bitmask of shard IDs with results ready
    var deferred_count:    Int                       # number of active deferred slots (0..16)
    var deferred_drain_ticks: Array[Int32, 16] # drain-call counter per slot (timeout guard)
    # M1: Inference bridge + deferred inference responses
    var inference_bridge: InferenceBridge
    var infer_deferred_fds:     Array[Int32, 8]   # client fd
    var infer_deferred_req_ids: Array[UInt32, 8]  # bridge req_id for matching
    var infer_deferred_types:   Array[UInt8, 8]   # 1=embed, 2=generate, 3=loadmodel
    var infer_deferred_count:   Int
    var infer_deferred_ticks:   Array[Int32, 8]   # timeout guard
    var infer_resp_buf: UnsafePointer[UInt8, MutUntrackedOrigin]  # scratch for response body
    # In-process Metal SDPA engine (handles ATTEND.PREFIX.* end-to-end).
    var metal_attn_engine: MetalAttentionEngine
    # gh #9: In-process CUDA SDPA engine (Linux GPU equivalent of metal_attn_engine).
    # Same C ABI surface; comptime-guarded so non-Linux builds compile to no-ops.
    var cuda_attn_engine: CudaAttentionEngine
    # AI.KNN_LM.* substrate (token-id-tagged FP32 kNN datastore registry)
    var knn_lm: KNNLMIndex
    # NEURON.PKM.* substrate (gh #146: product-key memory-layer tables)
    var pkm: PKMIndex
    # MOE.EXPERT.* substrate (gh #61 Stage-2 skeleton; backend in flight)
    var moe_tier: MoEExpertTier
    # A1: VSET state (element name ↔ HNSW node ID mapping)
    # Lua 5.1 scripting engine (per-worker, shared-nothing)
    var lua_engine: UnsafePointer[LuaEngine, MutUntrackedOrigin]
    # gh #100 (C2): server password (from config.server.requirepass). Empty = no
    # auth. Cached here so the binary path (process_binary_request, no config
    # param) can gate frames identically to the RESP path.
    var requirepass: String
    # gh #101: tenant table (read-only after startup; parsed per worker from
    # config.server.tenants), the per-worker key-rewrite scratch, and the
    # current command's namespace prefix ("name:") for KEYS/SCAN filtering.
    # cur_tenant_ns_len is 0 for admin/non-tenant commands; the pre-pass in
    # process_slow_path sets it at every command boundary when tenant mode is on.
    var tenant_table: TenantTable
    var tenant_scratch: UnsafePointer[UInt8, MutUntrackedOrigin]
    var tenant_ns_buf: UnsafePointer[UInt8, MutUntrackedOrigin]   # [MAX_TENANT_NAME + 1]
    var cur_tenant_ns_len: Int
    # gh #262: value receipt. New fields go at the END of this struct (gh #149).
    var ledger: ValueLedger
    var listen_port: Int
    # --no-wal: the fast path does not log its writes, so KV.PREFIX.COMMIT has
    # nothing it could make durable and must refuse rather than acknowledge.
    var wal_writes_off: Bool
    # gh #261: RSS above --maxmemory, per the housekeeping tick. Only a hint:
    # `_oom_refuses` re-asks C before refusing anything. Kept LAST (gh #149).
    var over_maxmemory: Bool
    # gh #390: WAIT clients parked until enough replicas ACK (the engine
    # answers them), and whether this worker may park at all — the XDP lane
    # sends through the packet engine and cannot reply to a parked fd later.
    var parked_waits: ParkedWaits
    var can_park_wait: Bool
    # #36: a script's redis.call() re-enters process_slow_path (script_dispatch).
    # The nested call parses into its own token tables, so the EVAL and any
    # command pipelined behind it keep theirs, and writes its reply into
    # script_writer, a capture-only writer (flush keeps the bytes). The
    # script_* context is the outer call's, set by _script_context before a run.
    var script_depth: Int
    var script_tokens_buf: UnsafePointer[RESP3Token, MutUntrackedOrigin]
    var script_cmd_ends_buf: UnsafePointer[Int, MutUntrackedOrigin]
    var script_cmd_byte_ends_buf: UnsafePointer[Int, MutUntrackedOrigin]
    var script_writer: UnsafePointer[ResponseWriter, MutUntrackedOrigin]
    var script_frame: UnsafePointer[UInt8, MutUntrackedOrigin]
    var script_frame_cap: Int
    var script_fd: Int32
    var script_server: UnsafePointer[TCPServer, MutUntrackedOrigin]
    var script_kq: Int32
    var script_hnsw: UnsafePointer[HNSWGraph, MutUntrackedOrigin]
    var script_db_size: UnsafePointer[Int, MutUntrackedOrigin]
    var script_config: UnsafePointer[PionConfig, MutUntrackedOrigin]
    var script_allow_oom: Bool
    # #38: BLPOP & co. parked until a key they wait on has data or they time
    # out; the engine wakes them (NetworkEngine._service_blocked_clients).
    var blocked_clients: BlockedClientRegistry
    # #39 MONITOR (src/commands/monitor.mojo). `fast_path_off` is read by the
    # engine's three process_data_plane call sites (update_dispatch_gate):
    # true while memory is over --maxmemory (gh #261), a client monitors
    # (commands are fed from here) or one is subscribed (#42: a RESP2
    # subscriber may run only the pub/sub commands); then fast_path_ok(fd)
    # decides per connection. `monitor_skip` is set while the engine re-runs a woken
    # blocking command, which monitors saw when it first ran.
    # `monitor_exec_line` holds EXEC's line until the commands it ran are out.
    var monitors: MonitorRegistry
    var fast_path_off: Bool
    var monitor_skip: Bool
    var monitor_exec_line: List[UInt8]
    # The engine's per-fd affinity bytes (3 = READONLY), which RESET clears.
    var local_affinity: UnsafePointer[UInt8, MutUntrackedOrigin]
    # The engine's writer while a script runs: a script's PUBLISH reaches
    # subscribers through it (the script's own writer only captures replies).
    var script_main_writer: UnsafePointer[ResponseWriter, MutUntrackedOrigin]
    # #46: the worker's vector-index tombstones (state.mojo sets it)
    var vec_tomb: UnsafePointer[VecTomb, MutUntrackedOrigin]

    def __init__(
        out self,
        keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
        hash_map_pool: UnsafePointer[ObjectPool[SlabHashMap], MutUntrackedOrigin],
        skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin],
        list_pool: UnsafePointer[ObjectPool[SlabList], MutUntrackedOrigin],
        ai_queue: UnsafePointer[LockFreeRingBuffer, MutUntrackedOrigin],
        wal: UnsafePointer[WAL, MutUntrackedOrigin],
        raft: UnsafePointer[RaftNode, MutUntrackedOrigin],
        shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
        config: PionConfig = PionConfig(),
        worker_id: Int = 0,
        num_workers: Int = 1,
        cluster: UnsafePointer[ClusterState, MutUntrackedOrigin] = null_ptr[ClusterState, MutUntrackedOrigin](),
        ttl_map: UnsafePointer[SlabHashMap, MutUntrackedOrigin] = null_ptr[SlabHashMap, MutUntrackedOrigin](),
    ):
        self.dispatcher = CommandDispatcher(keyspace, hash_map_pool, skip_list_pool, list_pool, ai_queue, wal, raft)
        self.parser = RESP3Parser()
        # gh #166: one heap-resident table set per worker (see the field decls).
        # RESP3Token is a plain marker/ptr/length triple, so the raw allocation
        # is usable directly — parse_stream writes every slot it later reads.
        self.tokens_buf = alloc[RESP3Token](MAX_CMD_TOKENS)
        self.cmd_ends_buf = alloc[Int](MAX_CMD_ENDS)
        self.cmd_byte_ends_buf = alloc[Int](MAX_CMD_ENDS)
        self.replay_ends_buf = alloc[Int](MAX_CMD_ENDS)
        self.replay_byte_ends_buf = alloc[Int](MAX_CMD_ENDS)
        unsafe_memset(self.cmd_ends_buf.bitcast[UInt8](), 0, MAX_CMD_ENDS * 8)
        unsafe_memset(self.cmd_byte_ends_buf.bitcast[UInt8](), 0, MAX_CMD_ENDS * 8)
        unsafe_memset(self.replay_ends_buf.bitcast[UInt8](), 0, MAX_CMD_ENDS * 8)
        unsafe_memset(self.replay_byte_ends_buf.bitcast[UInt8](), 0, MAX_CMD_ENDS * 8)
        self.keyspace = keyspace
        self.skip_list_pool = skip_list_pool
        self.shared_hnsw = shared_hnsw
        self.scratch_ids = List[Int]()
        self.scratch_dists = List[Float32]()
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.requirepass = config.server.requirepass  # gh #100 (C2)
        # gh #101: tenant namespacing state. The scratch is only sized when
        # tenant mode is on (256KB/worker otherwise wasted).
        self.tenant_table = TenantTable(config.server.tenants)
        self.tenant_scratch = alloc[UInt8](TENANT_SCRATCH_CAP) if self.tenant_table.count > 0 else alloc[UInt8](1)
        self.tenant_ns_buf = alloc[UInt8](MAX_TENANT_NAME + 1)
        self.cur_tenant_ns_len = 0
        self.ledger = ValueLedger()
        self.listen_port = config.server.port
        self.wal_writes_off = config.server.no_wal
        self.over_maxmemory = False
        self.parked_waits = ParkedWaits()
        self.can_park_wait = not config.server.use_xdp
        self.script_depth = 0
        self.script_tokens_buf = alloc[RESP3Token](MAX_CMD_TOKENS)
        self.script_cmd_ends_buf = alloc[Int](MAX_CMD_ENDS)
        self.script_cmd_byte_ends_buf = alloc[Int](MAX_CMD_ENDS)
        unsafe_memset(self.script_cmd_ends_buf.bitcast[UInt8](), 0, MAX_CMD_ENDS * 8)
        unsafe_memset(self.script_cmd_byte_ends_buf.bitcast[UInt8](), 0, MAX_CMD_ENDS * 8)
        self.script_writer = null_ptr[ResponseWriter, MutUntrackedOrigin]()
        self.script_frame = null_ptr[UInt8, MutUntrackedOrigin]()
        self.script_frame_cap = 0
        self.script_fd = -1
        self.script_server = null_ptr[TCPServer, MutUntrackedOrigin]()
        self.script_kq = -1
        self.script_hnsw = null_ptr[HNSWGraph, MutUntrackedOrigin]()
        self.script_db_size = null_ptr[Int, MutUntrackedOrigin]()
        self.script_config = null_ptr[PionConfig, MutUntrackedOrigin]()
        self.script_allow_oom = False
        self.blocked_clients = BlockedClientRegistry()
        self.monitors = MonitorRegistry()
        self.fast_path_off = False
        self.monitor_skip = False
        self.monitor_exec_line = List[UInt8]()
        self.local_affinity = null_ptr[UInt8, MutUntrackedOrigin]()
        self.script_main_writer = null_ptr[ResponseWriter, MutUntrackedOrigin]()
        self.vec_tomb = null_ptr[VecTomb, MutUntrackedOrigin]()
        self.cluster = cluster
        self.shard_query_seq = alloc[UInt64](1)
        self.shard_query_seq[0] = 1
        # gh #207: P3 staging field inits removed with the fields.
        # gh #87.1: IVF-PQ scratch alloc block removed.
        # Phase 2: semantic cache (per-worker; disabled by default unless config.embedding.enabled)
        self.scache = SemanticCache(
            config.embedding.host,
            config.embedding.port,
            config.embedding.model,
            config.embedding.dimensions,
            config.embedding.threshold,
            config.embedding.enabled,
            config.embedding.nle,
            config.embedding.query_prefix,
            config.embedding.doc_prefix,
        )
        # M14 Phase 1-3: KV cache store, layer store, attention index
        self.kvcache = KVCacheStore(config.embedding.dimensions, Float32(0.95), config.server.kvcache_enabled)
        self.layer_store = LayerStore(config.server.kvcache_enabled)
        # Only allocate 4MB binary response buffer when kvcache is enabled
        self.binary_resp_buf = alloc[UInt8](4 * 1024 * 1024) if config.server.kvcache_enabled else alloc[UInt8](1)
        self.attn_idx = AttentionIndex(config.server.kvcache_enabled)
        self.v_store = VStoreIndex(config.server.kvcache_enabled)
        self.state_store = StateStore(config.server.kvcache_enabled)
        # Hand the V-store its worker_id and a borrow of the cross-worker
        # session directory (allocated in main(), pointer in SharedHNSWView).
        # NULL when --kvcache is off; safe — directory APIs early-out on NULL.
        self.v_store.my_worker_id = worker_id
        self.v_store.ns_prefix = config.server.ns_prefix
        if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].vstore_directory):
            self.v_store.directory = UnsafePointer[VStoreDirectory, MutUntrackedOrigin](
                unsafe_from_address=Int(shared_hnsw[].vstore_directory))
        # V-store persistence on warm restart:
        #   1. Snapshot load (pion.vstore.<wid>) — base state at last KV.PREFIX.SAVE.
        #   2. WAL replay (pion.vstore.wal.<wid>) — records written after snapshot.
        #   3. WAL open in append mode for ongoing mutations (V.CREATE/STORE/etc).
        # Explicit snapshot via KV.PREFIX.SAVE compacts: writes a fresh snapshot
        # then truncates the WAL. No auto-shutdown hook in Pion today.
        if config.server.kvcache_enabled:
            var vs_path = "pion.vstore." + String(worker_id)
            _ = self.v_store.load_from_disk(vs_path)
            var wal_path = "pion.vstore.wal." + String(worker_id)
            _ = self.v_store.wal_replay(wal_path)
            self.v_store.wal_open(wal_path)
            # gh #94: SSM.PREFIX.* durability mirror — snapshot load + WAL
            # replay + open-for-append. Gated on --kvcache for symmetry with
            # the V-store path; the SSM substrate is itself substrate-only
            # and lives next to it.
            ssm_durability_startup(worker_id)
        # M13: Semantic router (enabled when kvcache is enabled)
        self.router = SemanticRouter(config.embedding.dimensions, config.server.kvcache_enabled)
        # M9: Speculative RAG
        self.spec_rag = SpeculativeRAG(config.embedding.dimensions, config.server.kvcache_enabled)
        self.blocked_readers = BlockedReaderRegistry()
        self.pubsub = PubSubRegistry()
        self.tx_state = TransactionState()
        # Phase 4: LLM client for AI.CHAT (disabled by default; opt-in via --llm-host/port)
        self.llm_client = LLMClient(
            config.llm.host,
            config.llm.port,
            config.llm.model,
            config.llm.enabled,
        )
        self.llm_out_buf = alloc[UInt8](1024 * 1024)  # 1MB scratch for LLM response
        # FLARE gateway (shares embedding + LLM config; enabled only when both are enabled)
        var flare_enabled = config.embedding.enabled and config.llm.enabled
        self.flare = FLAREGateway(
            config.embedding.host,
            config.embedding.port,
            config.embedding.model,
            config.embedding.dimensions,
            config.llm.host,
            config.llm.port,
            config.llm.model,
            config.llm.enabled,
            flare_enabled,
            config.embedding.nle,
        )
        self.ttl_map = ttl_map
        # T3.4 deferred shard state
        self.deferred_fds      = Array[Int32, 16](fill=Int32(-1))
        self.deferred_seqs     = Array[UInt64, 16](fill=UInt64(0))
        self.deferred_ks       = Array[Int32, 16](fill=Int32(0))
        self.deferred_n_shards = Array[Int32, 16](fill=Int32(0))
        self.deferred_active       = Array[UInt32, 16](fill=UInt32(0))
        self.deferred_done         = Array[UInt32, 16](fill=UInt32(0))
        self.deferred_count        = 0
        self.deferred_drain_ticks  = Array[Int32, 16](fill=Int32(0))
        self.snapshot_engine = SnapshotEngine("pion.snapshot")
        self.last_save_time  = Int64(0)
        # M1: Inference bridge
        self.inference_bridge = InferenceBridge(config.inference.enabled, config.inference.socket_path)
        if config.inference.enabled:
            _ = self.inference_bridge.connect()
            # A3: Wire inference bridge to semantic cache for auto-embedding
            self.scache.set_bridge(rebind[UnsafePointer[InferenceBridge, MutUntrackedOrigin]](UnsafePointer(to=self.inference_bridge)))
        self.infer_deferred_fds     = Array[Int32, 8](fill=Int32(-1))
        self.infer_deferred_req_ids = Array[UInt32, 8](fill=UInt32(0))
        self.infer_deferred_types   = Array[UInt8, 8](fill=UInt8(0))
        self.infer_deferred_count   = 0
        self.infer_deferred_ticks   = Array[Int32, 8](fill=Int32(0))
        self.infer_resp_buf = alloc[UInt8](INFER_RECV_BUF_SIZE)
        # In-process Metal SDPA engine. All workers init (Phase 2: per-worker session
        # cache + staging buffers); the device, queue, PSOs, and shared event are still
        # process-global, but each worker owns its own slot[256] and Q/O buffers, so
        # concurrent dispatches don't race.
        self.metal_attn_engine = MetalAttentionEngine(config.metal_attention.enabled, Int(worker_id), config.metal_attention.fp16, config.metal_attention.fa_window)
        # gh #9: CUDA engine — only meaningful on Linux GPU; on macOS/aarch64
        # the FFI symbols aren't linked and self.available stays False.
        self.cuda_attn_engine = CudaAttentionEngine(config.cuda_attention.enabled, Int(worker_id), config.cuda_attention.fa_window)
        # AI.KNN_LM.* enabled whenever the AI surface is available (kvcache or inference).
        # Cheap to init when disabled (no allocations until CREATE).
        self.knn_lm = KNNLMIndex(config.server.kvcache_enabled or config.inference.enabled)
        # gh #146: NEURON.PKM.* rides the same AI-surface gate as AI.KNN_LM.*.
        # Zero allocation when disabled; per-table buffers land at CREATE.
        self.pkm = PKMIndex(config.server.kvcache_enabled or config.inference.enabled)
        # MOE.EXPERT.* tier (gh #61 Stage 2). Enabled when --moe-cache <path>
        # is passed; cache budget from --moe-cache-mib (default 1024).
        # When enabled, the manifest loader reads <path>/config.json to
        # populate the model handle. Per-expert byte offsets land in a
        # follow-on (safetensors-shard parser).
        var moe_enabled = config.server.moe_cache_path.byte_length() > 0
        var moe_max_bytes = UInt64(config.server.moe_cache_mib) * UInt64(1 << 20) if moe_enabled else UInt64(0)
        self.moe_tier = MoEExpertTier(enabled=moe_enabled, cache_max_bytes=moe_max_bytes)
        if moe_enabled:
            _ = self.moe_tier.load_manifest(config.server.moe_cache_path)
            # Stage 4b-1: spawn the warming thread + ring buffers. Idle until
            # Stage 4b-3 wires PREFETCH → enqueue.
            self.moe_tier.init_warm_pool()
        # Initialize Lua engine (one per worker)
        self.lua_engine = alloc[LuaEngine](1)
        self.lua_engine.unsafe_write(LuaEngine())

    def process_binary_request(
        mut self,
        buffer: UnsafePointer[UInt8, MutUntrackedOrigin],
        n: Int,
        server: TCPServer,
        fd: Int32,
    ) raises -> Int:
        """Process a binary protocol request (M14). Returns bytes consumed, 0 if incomplete."""
        var req = parse_binary_request(buffer, n)
        if not req.valid:
            return 0
        var resp_buf = self.binary_resp_buf
        var resp_len: Int
        # gh #100 (C2): binary-lane auth gate. When --requirepass is set, the only
        # frame an unauthenticated binary connection may issue is CMD_AUTH; every
        # other command is refused with STATUS_ERROR. `len(...) > 0` short-circuits
        # to zero cost in the default (no-password) deployment.
        if self.requirepass.byte_length() > 0 and req.cmd == CMD_AUTH:
            var body_len = Int(req.body_len)
            var matches = body_len == self.requirepass.byte_length()
            if matches:
                var rp = self.requirepass.unsafe_ptr()
                for k in range(body_len):
                    if req.body_ptr[k] != rp[k]:
                        matches = False; break
            if matches:
                self.tx_state.authed[Int(fd)] = 1
                resp_len = build_binary_response_ok(resp_buf)
            else:
                resp_len = build_binary_response(resp_buf, STATUS_ERROR,
                    null_ptr[UInt8, MutUntrackedOrigin](), UInt32(0))
            _binary_send_all(server, fd, resp_buf, resp_len)
            return BINARY_HEADER_SIZE + Int(req.body_len)
        if self.requirepass.byte_length() > 0 and self.tx_state.authed[Int(fd)] == 0:
            resp_len = build_binary_response(resp_buf, STATUS_ERROR,
                null_ptr[UInt8, MutUntrackedOrigin](), UInt32(0))
            _binary_send_all(server, fd, resp_buf, resp_len)
            return BINARY_HEADER_SIZE + Int(req.body_len)
        # gh #261: the binary lane is where PionPromptCache ships its K/V —
        # the largest memory consumer Pion has — so --maxmemory must hold here
        # too, not only on RESP. Store opcodes get STATUS_ERROR with the RESP
        # error text as the body; fetches and queries keep working.
        if (req.cmd == CMD_LAYER_STORE or req.cmd == CMD_ATTEND_CREATE
                or req.cmd == CMD_ATTEND_STORE) \
           and external_call["pion_maxmemory_check", Int32]() != 0:
            # A literal: static data, never a stack pointer (gh #349).
            comptime OOM_MSG = "OOM command not allowed when used memory > 'maxmemory'."
            resp_len = build_binary_response(resp_buf, STATUS_ERROR,
                rebind[Pointer[UInt8, MutUntrackedOrigin]](OOM_MSG.unsafe_ptr()),
                UInt32(OOM_MSG.byte_length()))
            _binary_send_all(server, fd, resp_buf, resp_len)
            return BINARY_HEADER_SIZE + Int(req.body_len)
        if req.cmd == CMD_PING:
            resp_len = build_binary_response_ok(resp_buf)
        elif req.cmd == CMD_LAYER_STORE:
            var lreq = parse_layer_store_request(req.body_ptr, Int(req.body_len))
            if lreq.valid:
                var ok = self.layer_store.store_layer(
                    lreq.session_id_ptr, Int(lreq.session_id_len),
                    Int(lreq.layer_id), lreq.tensor_ptr, lreq.tensor_len)
                resp_len = build_binary_response_ok(resp_buf) if ok else build_binary_response_miss(resp_buf)
            else:
                resp_len = build_binary_response_miss(resp_buf)
        elif req.cmd == CMD_LAYER_FETCH:
            var lreq = parse_layer_fetch_request(req.body_ptr, Int(req.body_len))
            if lreq.valid:
                var out_ptr = alloc[UnsafePointer[UInt8, MutUntrackedOrigin]](1)
                var out_len = alloc[Int](1)
                var hit = self.layer_store.fetch_layer(
                    lreq.session_id_ptr, Int(lreq.session_id_len),
                    Int(lreq.layer_id), out_ptr, out_len)
                if hit:
                    var blob_len = out_len[]
                    # gh #91: build_binary_response memcpys the blob into the fixed
                    # 4MB binary_resp_buf with no bound check — a layer ≥ ~4MB
                    # overflows it (heap write OOB + send over-read). For large
                    # blobs, frame the 7-byte header in resp_buf and stream
                    # header + blob via scatter-gather (matches gh #76).
                    if blob_len + BINARY_RESP_HEADER_SIZE > 4 * 1024 * 1024:
                        write_u16_le(resp_buf, 0, BINARY_MAGIC)
                        resp_buf[2] = STATUS_OK
                        write_u32_le(resp_buf, 3, UInt32(blob_len))
                        _binary_send_all(server, fd, resp_buf, BINARY_RESP_HEADER_SIZE)
                        _binary_send_all(server, fd, out_ptr[], blob_len)
                        resp_len = 0  # already streamed; skip the trailing send
                    else:
                        resp_len = build_binary_response(resp_buf, STATUS_OK, out_ptr[], UInt32(blob_len))
                else:
                    resp_len = build_binary_response_miss(resp_buf)
                out_ptr.free()
                out_len.free()
            else:
                resp_len = build_binary_response_miss(resp_buf)
        elif req.cmd == CMD_ATTEND_CREATE:
            var areq = parse_attend_create(req.body_ptr, Int(req.body_len))
            if areq.valid:
                var si = self.attn_idx.create_session(areq.session_id_ptr, Int(areq.session_id_len),
                                                       Int(areq.key_dim), Int(areq.val_dim),
                                                       k_format=areq.k_format, v_format=areq.v_format,
                                                       boundary_layers_n=Int(areq.boundary_n),
                                                       boundary_v_format=areq.boundary_v_format)
                if si >= 0:
                    # Return session index as 4-byte LE body
                    var si_buf = alloc[UInt8](4)
                    write_u32_le(UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(si_buf)), 0, UInt32(si))
                    resp_len = build_binary_response(resp_buf, STATUS_OK,
                        UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(si_buf)), 4)
                else:
                    resp_len = build_binary_response_miss(resp_buf)
            else:
                resp_len = build_binary_response_miss(resp_buf)

        elif req.cmd == CMD_ATTEND_STORE:
            # Need session's key/val dims — find session first
            var sid_len = Int(read_u16_le(req.body_ptr, 0))
            var sid_ptr = req.body_ptr + 2
            var si = self.attn_idx._find_session(sid_ptr, sid_len)
            if si >= 0:
                var meta = self.attn_idx.sessions[si]
                var areq = parse_attend_store(req.body_ptr, Int(req.body_len), meta.key_dim, meta.value_dim)
                if areq.valid:
                    var ok = self.attn_idx.store_tokens(si, Int(areq.layer_id),
                        areq.keys_ptr.bitcast[Float32](), areq.values_ptr.bitcast[Float32](), Int(areq.num_tokens))
                    resp_len = build_binary_response_ok(resp_buf) if ok else build_binary_response_miss(resp_buf)
                else:
                    resp_len = build_binary_response_miss(resp_buf)
            else:
                resp_len = build_binary_response_miss(resp_buf)

        elif req.cmd == CMD_ATTEND_FINALIZE:
            var lreq = parse_layer_fetch_request(req.body_ptr, Int(req.body_len))  # same format: sid + layer_id
            if lreq.valid:
                var si = self.attn_idx._find_session(lreq.session_id_ptr, Int(lreq.session_id_len))
                if si >= 0:
                    var ok = self.attn_idx.finalize_layer(si, Int(lreq.layer_id))
                    resp_len = build_binary_response_ok(resp_buf) if ok else build_binary_response_miss(resp_buf)
                else:
                    resp_len = build_binary_response_miss(resp_buf)
            else:
                resp_len = build_binary_response_miss(resp_buf)

        elif req.cmd == CMD_ATTEND_QUERY:
            var areq = parse_attend_query(req.body_ptr, Int(req.body_len))
            if areq.valid:
                var si = self.attn_idx._find_session(areq.session_id_ptr, Int(areq.session_id_len))
                # The query must be exactly key_dim floats (the RESP arm's
                # rule) — the frame length was never checked, so a short body
                # was read past its end — and k is bounded as on the RESP arm.
                var q_off = 2 + Int(areq.session_id_len) + 4
                var k = Int(areq.k)
                var shape_ok = False
                if si >= 0:
                    shape_ok = (Int(req.body_len) - q_off == self.attn_idx.sessions[si].key_dim * 4
                                and k >= 1 and k <= ATTEND_MAX_K)
                if si >= 0 and shape_ok:
                    var meta = self.attn_idx.sessions[si]
                    var ef = ATTEND_QUERY_EF   # gh #391
                    var out_keys = alloc[Float32](k * meta.key_dim)
                    var out_values = alloc[Float32](k * meta.value_dim)
                    var _tid = alloc[Int](k)
                    var out_tids = UnsafePointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(_tid))
                    var nr = self.attn_idx.query_topk(si, Int(areq.layer_id),
                        areq.query_ptr.bitcast[Float32](), k, ef, out_keys, out_values, out_tids)
                    if nr > 0:
                        # gh #404: all `nr` value rows, best first (was the first
                        # row only, whatever k was). Same body as the RESP arm.
                        var val_bytes = nr * meta.value_dim * 4
                        if val_bytes + BINARY_RESP_HEADER_SIZE > 4 * 1024 * 1024:
                            # Past binary_resp_buf: stream it (gh #91 pattern).
                            write_u16_le(resp_buf, 0, BINARY_MAGIC)
                            resp_buf[2] = STATUS_OK
                            write_u32_le(resp_buf, 3, UInt32(val_bytes))
                            _binary_send_all(server, fd, resp_buf, BINARY_RESP_HEADER_SIZE)
                            _binary_send_all(server, fd, out_values.bitcast[UInt8](), val_bytes)
                            resp_len = 0
                        else:
                            resp_len = build_binary_response(resp_buf, STATUS_OK,
                                out_values.bitcast[UInt8](), UInt32(val_bytes))
                    else:
                        resp_len = build_binary_response_miss(resp_buf)
                    out_keys.free()
                    out_values.free()
                    _tid.free()
                elif si >= 0:
                    # A literal: static data, never a stack pointer (gh #349).
                    comptime AQ_SHAPE_MSG = "ERR ATTEND.QUERY query must be key_dim * 4 bytes and 1 <= k <= 4096"
                    resp_len = build_binary_response(resp_buf, STATUS_ERROR,
                        rebind[Pointer[UInt8, MutUntrackedOrigin]](AQ_SHAPE_MSG.unsafe_ptr()),
                        UInt32(AQ_SHAPE_MSG.byte_length()))
                else:
                    resp_len = build_binary_response_miss(resp_buf)
            else:
                resp_len = build_binary_response_miss(resp_buf)

        elif req.cmd == CMD_ATTEND_PREFIX_QUERY_FUSED:
            # gh #50: Stage 2 fast lane. Same Metal kernel as the RESP variant
            # (handle_attend_prefix_query_fused) but skips RESP framing on
            # send and the per-token bytes-alloc on receive.
            var freq = parse_attend_prefix_query_fused(req.body_ptr, Int(req.body_len))
            if freq.valid and self.metal_attn_engine.available:
                var H_q = Int(freq.H_q)
                var M_q = Int(freq.M)
                var D_q = Int(freq.D)
                var H_kv = Int(freq.H_kv)
                var S_suf = Int(freq.S_suf)
                var out_floats = H_q * M_q * D_q
                var out_bytes = out_floats * 4
                # gh #67: COLDMISS short-circuit. If the (sid, layer) was demoted
                # to the cold tier, return STATUS_COLDMISS so the client can
                # issue KV.PREFIX.WARM + retry on the RESP lane (the binary
                # lane stays light — rehydrate is a heavy infrequent path).
                if self.metal_attn_engine.session_state(freq.session_id_ptr, Int(freq.session_id_len), Int(freq.layer_id)) == 2:
                    resp_len = build_binary_response(resp_buf, STATUS_COLDMISS,
                        null_ptr[UInt8, MutUntrackedOrigin](), 0)
                # binary_resp_buf is 4MB — body must fit alongside the 7-byte header.
                elif out_bytes + BINARY_RESP_HEADER_SIZE > 4 * 1024 * 1024:
                    resp_len = build_binary_response(resp_buf, STATUS_ERROR,
                        null_ptr[UInt8, MutUntrackedOrigin](), 0)
                else:
                    # Write attention output directly into the response body slot,
                    # then patch the header in place — saves the memcpy that
                    # build_binary_response would do otherwise.
                    var out_ptr = UnsafePointer[Float32, MutUntrackedOrigin](
                        unsafe_from_address=Int(resp_buf + BINARY_RESP_HEADER_SIZE))
                    var Q_ptr = freq.Q_ptr.bitcast[Float32]()
                    var Ks_ptr = freq.K_suf_ptr.bitcast[Float32]()
                    var Vs_ptr = freq.V_suf_ptr.bitcast[Float32]()
                    var ok = self.metal_attn_engine.query_batched_fused(
                        freq.session_id_ptr, Int(freq.session_id_len),
                        Int(freq.layer_id),
                        H_q, M_q, D_q, H_kv, S_suf,
                        Q_ptr, Ks_ptr, Vs_ptr, freq.head_map_ptr, out_ptr,
                        fa_window_override=Int(freq.fa_window),
                    )
                    if ok:
                        write_u16_le(resp_buf, 0, BINARY_MAGIC)
                        resp_buf[2] = STATUS_OK
                        write_u32_le(resp_buf, 3, UInt32(out_bytes))
                        resp_len = BINARY_RESP_HEADER_SIZE + out_bytes
                    else:
                        resp_len = build_binary_response(resp_buf, STATUS_ERROR,
                            null_ptr[UInt8, MutUntrackedOrigin](), 0)
            elif not self.metal_attn_engine.available:
                resp_len = build_binary_response(resp_buf, STATUS_ERROR,
                    null_ptr[UInt8, MutUntrackedOrigin](), 0)
            else:
                resp_len = build_binary_response_miss(resp_buf)

        else:
            resp_len = build_binary_response_miss(resp_buf)
        if resp_len > 0:
            # gh #91: a single send() transmits only ~one socket-buffer worth on a
            # non-blocking fd, silently dropping the rest of a multi-MB buffered
            # body (e.g. a ~4MB LAYER_FETCH or fused-attention output). Loop until
            # all bytes are out; small responses still finish in one iteration.
            _binary_send_all(server, fd, resp_buf, resp_len)
        return BINARY_HEADER_SIZE + Int(req.body_len)

    def _info_replication_section(self) -> String:
        """INFO's `# Replication` section from the cluster state: a replica's
        role, primary and offset; a primary's replicas and offset; a
        standalone server is a primary with none."""
        var out = String("# Replication\r\n")
        var _cl = self.cluster
        if is_null(_cl) or not _cl[].enabled:
            out += "role:master\r\nconnected_slaves:0\r\n"
            return out
        if _cl[].is_replica:
            out += "role:slave\r\nmaster_host:"
            if _cl[].primary_peer_idx >= 0 and _cl[].primary_peer_idx < 16:
                var _ph_off = _cl[].primary_peer_idx * 64
                var _ph_len = 0
                while _ph_len < 63 and _cl[].peer_hosts[_ph_off + _ph_len] != 0:
                    out += chr(Int(_cl[].peer_hosts[_ph_off + _ph_len]))
                    _ph_len += 1
            out += "\r\nslave_repl_offset:"
            if is_not_null(_cl[].repl_replica_handle):
                out += String(Int(external_call["pion_repl_replica_bytes_received", Int64](_cl[].repl_replica_handle)))
            else:
                out += "0"
            out += "\r\n"
        else:
            out += "role:master\r\nconnected_slaves:"
            if is_not_null(_cl[].repl_primary_handle):
                out += String(Int(external_call["pion_repl_primary_connected_count", Int32](_cl[].repl_primary_handle)))
            else:
                out += "0"
            out += "\r\nmaster_repl_offset:" + String(Int(self.dispatcher.wal[].tail_offset)) + "\r\n"
        out += "repl_backlog_size:268435456\r\n"
        return out

    def _value_receipt_info(self) -> String:
        """gh #262: the `# Pion` INFO section — THIS worker's value receipt."""
        var s = String("# Pion\r\n")
        s += "stats_scope:worker\r\n"
        s += "worker_id:" + String(self.worker_id) + "\r\n"
        s += "kvprefix_hits:" + String(self.ledger.kvprefix_hits) + "\r\n"
        s += "kvprefix_misses:" + String(self.ledger.kvprefix_misses) + "\r\n"
        s += "kvprefix_hits_measured:" + String(self.ledger.kvprefix_hits_measured) + "\r\n"
        s += "kvprefix_tokens_served:" + String(self.ledger.kvprefix_tokens_served) + "\r\n"
        s += "kvprefix_bytes_served:" + String(self.ledger.kvprefix_bytes_served) + "\r\n"
        s += "prefill_seconds_avoided:" + self.ledger.prefill_seconds_avoided() + "\r\n"
        s += "prefill_seconds_avoided_measured:" + self.ledger.prefill_seconds_avoided_measured() + "\r\n"
        s += "semantic_hits:" + String(self.scache.hits) + "\r\n"
        s += "semantic_misses:" + String(self.scache.misses) + "\r\n"
        s += "moe_hits:" + String(self.moe_tier.hits) + "\r\n"
        s += "moe_misses:" + String(self.moe_tier.misses) + "\r\n"
        s += "vector_queries:" + String(self.ledger.vector_queries) + "\r\n"
        return s

    def update_dispatch_gate(mut self):
        """Recompute `fast_path_off` after anything it depends on changed."""
        self.fast_path_off = (self.over_maxmemory or self.monitors.count() > 0
                              or self.pubsub.subscribed_fds > 0)

    def fast_path_ok(self, ci: Int) -> Bool:
        """Under the gate: may this connection's commands take the fast path?
        Not while memory is over the limit or a client monitors, nor for a
        subscribed connection, whose commands the slow path checks."""
        return (not self.over_maxmemory and self.monitors.count() == 0
                and not self.pubsub.subscribed(Int32(ci)))

    def _reset_connection(mut self, fd: Int32, mut writer: ResponseWriter):
        """RESET, as Redis's clearClientConnectionState (#39): out of MONITOR,
        MULTI discarded and WATCH dropped, every subscription gone, back to
        RESP2 and the default user (unauthenticated when a password is set,
        and no longer bound to a tenant), no name, READONLY off. It answered
        +RESET and reset none of it."""
        _ = self.monitors.remove(fd)
        self.tx_state.cleanup_fd(fd)
        self.pubsub.cleanup_fd(fd)
        self.update_dispatch_gate()
        if is_not_null(self.local_affinity) and self.local_affinity[Int(fd)] == 3:
            self.local_affinity[Int(fd)] = 0
        writer.proto = 2
        writer.append_status_response("RESET")

    def _monitor_line_for(self, tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int,
                          fd: Int32) -> List[UInt8]:
        """#39: the MONITOR line for tokens[i:end], or nothing when Redis
        would not show the command: unknown, a wrong argument count (refused
        before it ran), or `admin`."""
        var tp = tokens[i].ptr
        var tl = tokens[i].length
        if not command_exists(tp, tl):
            return List[UInt8]()
        var argc = end - i
        var ar = command_arity(tp, tl)
        if (ar > 0 and argc != ar) or (ar < 0 and argc < -ar):
            return List[UInt8]()
        var sp = tokens[i + 1].ptr if argc > 1 else tp
        var sl = tokens[i + 1].length if argc > 1 else 0
        if command_hidden_from_monitor(tp, tl, sp, sl):
            return List[UInt8]()
        return monitor_line(tokens, i, end, Int32(-1) if self.script_depth > 0 else fd)

    def _monitor_feed(mut self, tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int,
                      fd: Int32, mut writer: ResponseWriter, server: TCPServer, kq: Int32):
        """#39: show a command to the monitors. A script's commands wait in
        `monitors.pending` until the script's own line is out."""
        var line = self._monitor_line_for(tokens, i, end, fd)
        if len(line) == 0:
            return
        if self.script_depth > 0:
            for b in range(len(line)):
                self.monitors.pending.append(line[b])
            return
        self._monitor_send(line, fd, writer, server, kq)

    def _monitor_send(mut self, line: List[UInt8], fd: Int32, mut writer: ResponseWriter,
                      server: TCPServer, kq: Int32):
        """#39: lines to every monitor. A connection that monitors gets them
        after its own reply; the others through their output buffers."""
        var n = len(line)
        if n == 0:
            return
        var lp = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(line.unsafe_ptr()))
        for k in range(self.monitors.count()):
            var m = self.monitors.fds[k]
            if m == fd:
                writer.append_to_response(lp, n)
            else:
                writer.deliver_to(m, lp, n, server, kq)

    def _rpoplpush(mut self, tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], i: Int,
                   mut writer: ResponseWriter) raises:
        """RPOPLPUSH tokens[i+1] tokens[i+2] (BRPOPLPUSH runs it once its source
        has an element)."""
        var rpl_src = tokens[i+1].value()
        var rpl_dst = tokens[i+2].value()
        # gh #232: neither key was type-checked, and the pop
        # happened first. With a wrong-type DESTINATION the
        # element was popped, answered as a normal success,
        # and then silently dropped — never pushed anywhere.
        # RPOPLPUSH is THE reliable-queue primitive
        # (`RPOPLPUSH work processing`), so that is a job
        # vanishing with a reply that says it moved. Redis
        # validates both keys before touching either.
        var rpl_sv = self.keyspace[].get(GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
        var rpl_dv = self.keyspace[].get(GenericValue.borrow(tokens[i+2].ptr, tokens[i+2].length))
        # Redis's precedence, and it is observable: wrong-type
        # SOURCE errors; a MISSING source is nil and the
        # destination is never examined; only then does a
        # wrong-type destination error.
        var rpl_src_bad = (not rpl_sv.is_none()
                           and rpl_sv.type.value != ValueType.LIST)
        var rpl_dst_bad = (not rpl_sv.is_none() and not rpl_dv.is_none()
                           and rpl_dv.type.value != ValueType.LIST)
        # No `continue` here: the loop tail runs the gh #240
        # forward clamp and `i += 1`, and skipping them is
        # the very desync this arm is being fixed for.
        if rpl_src_bad or rpl_dst_bad:
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
        else:
            var rpl_val = self.dispatcher.execute_rpop(rpl_src)
            if rpl_val.is_none():
                writer.append_null_response()
            else:
                # Push the value directly; both String(gv) and
                # gv.__str__() stringify the internal pointer for
                # heap/SSO string values, corrupting the element.
                writer.append_bulk_value_response(rpl_val)
                _ = self.dispatcher.execute_lpush_value(rpl_dst, rpl_val)
                # gh #234 remainder (found via gh #265):
                # remove the source once its last element
                # leaves, or `EXISTS` lies and the key
                # cannot be reused as another type. gh #234
                # fixed the LPOP/RPOP handlers; this arm
                # reimplements the pop and so was missed.
                #
                # AFTER the push: `src == dst` is a legal
                # rotation and is size 1 again by here.
                var rpl_after = self.keyspace[].get(
                    GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                if (not rpl_after.is_none()
                        and rpl_after.type.value == ValueType.LIST
                        and rpl_after.as_list()
                            .unsafe_bitcast[SlabList]()[].size == 0):
                    _ = remove_and_free(self.keyspace, 
                        GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                    _ = self.dispatcher.wal[].append(
                        2, tokens[i+1].ptr, tokens[i+1].length)

    def _park_blocked(mut self, fd: Int32, buffer: UnsafePointer[UInt8, MutUntrackedOrigin],
                      cmd_idx: Int, num_cmds: Int, cmd_byte_ends: UnsafePointer[Int, MutUntrackedOrigin],
                      on_primary: Bool, tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin],
                      key_first: Int, key_end: Int, deadline_ms: Int64, zset: Bool) -> Bool:
        """#38: park a blocking command that found nothing: register it with a
        copy of its frame and its keys, and stop the batch at its end (the
        caller does that when this returns True). False where it cannot park
        (MULTI/EXEC replay, a script, the XDP lane): the caller answers nil."""
        if not (self.can_park_wait and on_primary and cmd_idx < num_cmds and fd >= 0):
            return False
        var start = cmd_byte_ends[cmd_idx - 1] if cmd_idx > 0 else 0
        var end = cmd_byte_ends[cmd_idx]
        self.blocked_clients.add(new_blocked_client(fd, deadline_ms, zset, buffer + start, end - start,
                                                    tokens, key_first, key_end))
        self.parked_waits.park_fd(fd)
        return True

    def _bpop_lists(mut self, tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], first: Int, end: Int,
                    left: Bool, mut writer: ResponseWriter) raises -> Bool:
        """BLPOP/BRPOP's pop: the keys left to right, a missing one skipped, a
        wrong-type one an error, from the first list an element, replied
        [key, element]. True when it replied."""
        for _bk in range(first, end):
            var _bkey = tokens[_bk].value()
            var _bval = self.keyspace[].get(GenericValue.borrow(tokens[_bk].ptr, tokens[_bk].length))
            if _bval.is_none():
                continue
            if _bval.type.value != ValueType.LIST:
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return True
            var _blp = _bval.as_list().unsafe_bitcast[SlabList]()
            if _blp[].size == 0:
                continue
            var _bpopped = self.dispatcher.execute_lpop(_bkey) if left else self.dispatcher.execute_rpop(_bkey)
            if _bpopped.is_none():
                continue
            writer.append_array_header(2)
            writer.append_bulk_string_response(tokens[_bk].ptr, tokens[_bk].length)
            writer.append_bulk_value_response(_bpopped)
            _bpopped.free_str_payload()   # owned by the caller once popped
            # gh #234: drop an emptied list, and only AFTER the reply.
            if self.dispatcher.execute_llen(_bkey) == 0:
                _ = remove_and_free(self.keyspace, GenericValue.borrow(tokens[_bk].ptr, tokens[_bk].length))
                if is_not_null(self.dispatcher.wal):
                    _ = self.dispatcher.wal[].append(2, tokens[_bk].ptr, tokens[_bk].length)
            return True
        return False

    def _bzpop(mut self, tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], first: Int, end: Int,
               from_min: Bool, mut writer: ResponseWriter) raises -> Bool:
        """BZPOPMIN/BZPOPMAX's pop: from the first sorted set, one member,
        replied [key, member, score]. True when it replied."""
        for k in range(first, end):
            var kv = self.keyspace[].get(GenericValue.borrow(tokens[k].ptr, tokens[k].length))
            if kv.is_none():
                continue
            if kv.type.value != ValueType.ZSET:
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return True
            var zp = kv.as_zset().bitcast[SlabSkipList]()
            if zp[].length == 0:
                continue
            var r = zp[].pop_min() if from_min else zp[].pop_max()
            if not r.valid:
                continue
            writer.append_array_header(3)
            writer.append_bulk_string_response(tokens[k].ptr, tokens[k].length)
            writer.append_bulk_value_response(r.obj)
            writer.append_score_response(r.score)
            if is_not_null(self.dispatcher.wal):            # gh #170: the resolved effect, a ZREM
                var wb = alloc[UInt8](64)
                var wl = 0
                var wp = gv_bytes(r.obj, wb, wl)
                _ = self.dispatcher.wal[].append_kv(12, tokens[k].ptr, tokens[k].length, wp, wl)
                wb.free()
            r.obj.free_str_payload()
            if zp[].length == 0:                             # gh #234
                _ = remove_and_free(self.keyspace, GenericValue.borrow(tokens[k].ptr, tokens[k].length))
                if is_not_null(self.dispatcher.wal):
                    _ = self.dispatcher.wal[].append(2, tokens[k].ptr, tokens[k].length)
            return True
        return False

    def _mpop_lists(mut self, tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], first_key: Int,
                    numkeys: Int, left: Bool, count: Int, mut writer: ResponseWriter) -> Bool:
        """LMPOP's pop, as Redis's mpopGenericCommand: the keys in order, a
        missing one skipped, a wrong-type one an error, and from the first
        non-empty list up to `count` elements, replied `[key, [elements]]`. An
        emptied list is removed (gh #234). True when it replied; False when no
        key held anything, which the caller answers (a null array for LMPOP)."""
        for k in range(numkeys):
            var kt = tokens[first_key + k]
            var kv = GenericValue.borrow(kt.ptr, kt.length)
            var v = self.keyspace[].get(kv)
            if v.is_none():
                continue
            if v.type.value != ValueType.LIST:
                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                return True
            var lp = v.as_list().bitcast[SlabList]()
            if lp[].size == 0:
                continue
            var n = min(count, lp[].size)
            var key_str = kt.value()
            writer.append_array_header(2)
            writer.append_bulk_string_response(kt.ptr, kt.length)
            writer.append_array_header(n)
            for _ in range(n):
                var popped = self.dispatcher.execute_lpop(key_str) if left else self.dispatcher.execute_rpop(key_str)
                writer.append_bulk_value_response(popped)
                popped.free_str_payload()   # the pop handed back an owned value; the reply copied it
            if self.dispatcher.execute_llen(key_str) == 0:
                _ = remove_and_free(self.keyspace, kv)
                if is_not_null(self.dispatcher.wal):
                    _ = self.dispatcher.wal[].append(2, kt.ptr, kt.length)
            return True
        return False

    def process_slow_path(
        mut self,
        buffer: UnsafePointer[UInt8, MutUntrackedOrigin],
        n: Int,
        fd: Int32,
        mut writer: ResponseWriter,
        server: TCPServer,
        kq: Int32,
        mut hnsw: HNSWGraph,
        mut db_size: Int,
        config: PionConfig,
    ) -> Int:
        from src.network.replication import ReplicaReceiver
        # gh #162: hoisted out of the try so the `except` below can recover at
        # command granularity. `cmd_byte_ends[cmd_idx]` gives the byte offset
        # past the command being dispatched when a handler raises, and
        # `cmd_write_start` marks where its (possibly half-written) response
        # began so it can be rolled back. `on_primary` distinguishes the primary
        # recv frame from an EXEC-replay frame, whose byte offsets don't refer
        # to this buffer.
        # gh #166: heap-resident, per worker. Zeroing the whole table per call
        # would be a 16 KB memset on every slow-path frame; parse_stream writes
        # each entry before the dispatch loop reads it, and reads are bounded by
        # num_cmds, so the one-time zeroing at construction is sufficient.
        var cmd_byte_ends = self.cmd_byte_ends_buf if self.script_depth == 0 else self.script_cmd_byte_ends_buf
        var i = 0
        var cmd_idx = 0
        var num_cmds = 0
        var on_primary = True
        var cmd_write_start = 0
        try:
            var tokens = self.tokens_buf if self.script_depth == 0 else self.script_tokens_buf
            var num_tokens = 0
            var consumed_bytes = 0
            var cmd_ends = self.cmd_ends_buf if self.script_depth == 0 else self.script_cmd_ends_buf
            self.parser.parse_stream(buffer, n, tokens, num_tokens, consumed_bytes, cmd_ends, cmd_byte_ends, num_cmds)
            if num_tokens == TOKENS_OVERFLOW:
                # gh #153: a single complete command with more than
                # MAX_CMD_TOKENS arguments. Waiting for more data can never
                # help (that's what used to hang the connection), so reply and
                # skip the frame — `consumed_bytes` covers it exactly, which
                # keeps a pipelined client frame-synced.
                writer.append_error_response(
                    "ERR command has too many arguments (max "
                    + String(MAX_CMD_TOKENS)
                    + "); for high-dimension vectors use the FP32 blob form"
                )
                writer.flush_response(fd, server, kq)
                return consumed_bytes
            if num_tokens == 0:
                # Empty buffer or partial/incomplete command — wait for more data
                return 0
            # MULTI/EXEC replay outer loop. EXEC stages the queued buffers
            # in exec_replay_q/_count below; each replay iteration re-enters
            # this same stack frame instead of recursing into process_slow_path,
            # which would blow the parallelize thread's ~512KB stack (gh #94).
            var exec_replay_count = 0
            var exec_replay_qi = 0
            var exec_replay_q = null_ptr[QueuedCommand, MutUntrackedOrigin]()
            var primary_consumed = consumed_bytes
            while True:
                while i < num_tokens:
                    var token = tokens[i]
                    var tl = token.length
                    var tp = token.ptr
                    # gh #162: where this command's response starts, so the
                    # recovery `except` can drop a half-written frame before
                    # emitting its error.
                    cmd_write_start = writer.offset
                    # cmd_ends holds MAX_CMD_ENDS == MAX_CMD_TOKENS entries and a
                    # command is at least one token, so this can't run off the
                    # end of a real batch (gh #156 — it used to, at 16, and the
                    # `num_tokens` fallback then made the 17th command swallow
                    # every command behind it). The fallback stays for the
                    # degenerate no-command parse.
                    var cmd_end_tok = cmd_ends[cmd_idx] if cmd_idx < num_cmds else num_tokens

                    # #39: a connection in MONITOR mode may not touch the
                    # keyspace, as in Redis (where a monitor is a kind of replica).
                    if self.script_depth == 0 and self.monitors.count() > 0 and self.monitors.contains(fd):
                        var _msp = tokens[i + 1].ptr if cmd_end_tok - i > 1 else tp
                        var _msl = tokens[i + 1].length if cmd_end_tok - i > 1 else 0
                        if command_touches_keyspace(tp, tl, _msp, _msl):
                            writer.append_error_response("ERR Replica can't interact with the keyspace")
                            i = cmd_end_tok
                            while cmd_idx < num_cmds and i >= cmd_ends[cmd_idx]:
                                cmd_idx += 1
                            continue

                    # #42: a RESP2 connection with subscriptions may run only the
                    # pub/sub commands, PING (answered [pong, msg]), QUIT and
                    # RESET, as in Redis. RESP3 has no such limit.
                    if self.script_depth == 0 and writer.proto == 2 and self.pubsub.subscribed(fd):
                        if cmd_eq(tp, tl, "ping"):
                            if cmd_end_tok - i > 2:
                                writer.append_error_response("ERR wrong number of arguments for 'ping' command")
                            else:
                                writer.append_array_header(2)
                                writer.append_bulk_string_response("pong".unsafe_ptr(), 4)
                                if cmd_end_tok - i == 2:
                                    writer.append_bulk_string_response(tokens[i + 1].ptr, tokens[i + 1].length)
                                else:
                                    writer.append_bulk_string_response("".unsafe_ptr(), 0)
                            i = cmd_end_tok
                            while cmd_idx < num_cmds and i >= cmd_ends[cmd_idx]:
                                cmd_idx += 1
                            continue
                        if not (cmd_eq(tp, tl, "subscribe") or cmd_eq(tp, tl, "unsubscribe")
                                or cmd_eq(tp, tl, "psubscribe") or cmd_eq(tp, tl, "punsubscribe")
                                or cmd_eq(tp, tl, "ssubscribe") or cmd_eq(tp, tl, "sunsubscribe")
                                or cmd_eq(tp, tl, "quit") or cmd_eq(tp, tl, "reset")):
                            writer.append_error_response("ERR Can't execute '" + token.value().lower()
                                                         + "': only (P|S)SUBSCRIBE / (P|S)UNSUBSCRIBE / PING / QUIT / RESET are allowed in this context")
                            i = cmd_end_tok
                            while cmd_idx < num_cmds and i >= cmd_ends[cmd_idx]:
                                cmd_idx += 1
                            continue

                    # ── MULTI mode interception: queue commands instead of executing ──
                    # (never for a script's redis.call: it runs inside an EXEC or
                    # on its own, and MULTI is refused inside scripts)
                    if self.script_depth == 0 and self.tx_state.is_multi(fd):
                        # Allow EXEC, DISCARD, MULTI (error), WATCH, UNWATCH through
                        var is_tx_cmd = False
                        if tl == 4 and cmd_matches_4(tp, 101, 120, 101, 99): is_tx_cmd = True   # exec
                        elif tl == 5 and cmd_matches_5(tp, 109, 117, 108, 116, 105): is_tx_cmd = True  # multi
                        elif tl == 7 and cmd_matches_7(tp, 100, 105, 115, 99, 97, 114, 100): is_tx_cmd = True  # discard
                        # Redis runs these at once inside MULTI too (#39): WATCH
                        # to refuse itself, QUIT and RESET to end the transaction.
                        elif cmd_eq(tp, tl, "watch") or cmd_eq(tp, tl, "quit") or cmd_eq(tp, tl, "reset"):
                            is_tx_cmd = True
                        if not is_tx_cmd:
                            # gh #219: queue THIS command's frame only. This used to
                            # enqueue (buffer, n) — the whole recv buffer — and then
                            # `i = num_tokens`, on the assumption of one command per
                            # call. A pipelined transaction (redis-py's default
                            # `pipeline()` sends MULTI + commands + EXEC in ONE write)
                            # therefore got a single +QUEUED for the whole batch,
                            # never reached its EXEC, and silently wrote nothing —
                            # while the client waited on N missing replies.
                            #
                            # The parser's byte boundaries delimit the frame: this
                            # command starts where the previous one ended.
                            var q_start = cmd_byte_ends[cmd_idx - 1] if cmd_idx > 0 else 0
                            var q_end = cmd_byte_ends[cmd_idx] if cmd_idx < num_cmds else n
                            # The enqueue result is load-bearing: replying
                            # +QUEUED for a command that was NOT stored told the
                            # client it was accepted and then dropped it. At the
                            # old fixed cap of 128 a 200-command transaction got
                            # 200 × +QUEUED and executed 128 — 72 acknowledged
                            # writes lost, and the client skewed by 72 replies.
                            # The queue grows now, so this only fires at the hard
                            # ceiling; when it does, poison the transaction so
                            # EXEC aborts rather than applying it in part.
                            # gh #220: validate at QUEUE time, like Redis. An
                            # unknown command used to be queued with +QUEUED and
                            # only error at replay — by which point its siblings
                            # had already been applied, so one typo turned an
                            # atomic transaction into a partially applied one
                            # (redis-py raises, the caller retries, and the
                            # already-applied INCRs run twice). Poisoning the
                            # transaction here makes EXEC answer -EXECABORT and
                            # apply nothing.
                            # gh #220 remainder: Redis validates ARITY at queue
                            # time too, not just the name, and answers EXEC with
                            # -EXECABORT. Pion queued a bad-arity command with
                            # +QUEUED and only failed at replay — by which point
                            # the transaction's earlier commands had already
                            # applied, which is the partial-application this
                            # issue exists to stop. Name-only validation caught
                            # the typo'd COMMAND but not the typo'd ARGUMENT
                            # LIST, and the second is the more common mistake.
                            #
                            # `command_arity` returns 0 for every Pion-specific
                            # command, and 0 means "no opinion" — a wrong
                            # rejection here would break a VALID transaction,
                            # which is worse than the gap being closed.
                            var _tx_argc = cmd_end_tok - i
                            var _tx_ar = command_arity(tp, tl)
                            var _tx_bad_arity = False
                            if _tx_ar > 0 and _tx_argc != _tx_ar:
                                _tx_bad_arity = True
                            elif _tx_ar < 0 and _tx_argc < -_tx_ar:
                                _tx_bad_arity = True
                            # gh #261: Redis refuses a denyoom command at QUEUE
                            # time when over the limit, and poisons the
                            # transaction so EXEC answers -EXECABORT.
                            if self.over_maxmemory and command_is_denyoom(tp, tl) \
                               and external_call["pion_maxmemory_check", Int32]() != 0:
                                self.tx_state.set_dirty(fd)
                                writer.append_error_response("OOM command not allowed when used memory > 'maxmemory'.")
                            elif not command_exists(tp, tl):
                                self.tx_state.set_dirty(fd)
                                writer.append_error_response(
                                    "ERR unknown command '" + token.value() + "'")
                            elif _tx_bad_arity:
                                self.tx_state.set_dirty(fd)
                                writer.append_error_response(
                                    "ERR wrong number of arguments for '"
                                    + token.value() + "' command")
                            else:
                                var queued = True
                                if q_end > q_start:
                                    queued = self.tx_state.enqueue(
                                        fd, buffer.unsafe_offset(q_start), q_end - q_start)
                                if queued:
                                    writer.append_to_response("+QUEUED\r\n".unsafe_ptr(), 9)
                                else:
                                    self.tx_state.set_dirty(fd)
                                    writer.append_error_response(
                                        "ERR transaction queue limit reached; transaction discarded")
                            # This branch `continue`s, skipping the loop tail that
                            # normally does `i += 1` and advances cmd_idx, so it must
                            # do both itself — same idiom as the auth gate below.
                            i = cmd_end_tok
                            while cmd_idx < num_cmds and i >= cmd_ends[cmd_idx]:
                                cmd_idx += 1
                            continue

                    # ── Auth gate (gh #100 / C2) ──
                    # When --requirepass is set, reject every command with -NOAUTH
                    # until the connection has authenticated. AUTH/HELLO/QUIT/RESET
                    # are exempt so the client can authenticate (or negotiate/close).
                    # The `len(...) > 0` short-circuits to near-zero cost when unset.
                    if config.server.requirepass.byte_length() > 0 and not self.tx_state.is_authed(fd):
                        var auth_exempt = False
                        if tl == 4 and cmd_matches_4(tp, 97, 117, 116, 104): auth_exempt = True       # AUTH
                        elif tl == 5 and cmd_matches_5(tp, 104, 101, 108, 108, 111): auth_exempt = True  # HELLO
                        elif tl == 4 and cmd_matches_4(tp, 113, 117, 105, 116): auth_exempt = True     # QUIT
                        elif tl == 5 and cmd_matches_5(tp, 114, 101, 115, 101, 116): auth_exempt = True  # RESET
                        if not auth_exempt:
                            writer.append_error_response("NOAUTH Authentication required.")
                            i = cmd_end_tok  # skip this whole command's tokens
                            while cmd_idx < num_cmds and i >= cmd_ends[cmd_idx]:
                                cmd_idx += 1
                            continue

                    # gh #260: the WAL is full and can no longer persist. Refuse
                    # the keyspace MUTATIONS — acknowledging a write the server
                    # knows will not survive a restart is a lie the client cannot
                    # detect — while still serving reads, which are unaffected.
                    # `command_is_write` is Redis's own `write` flag, generated
                    # into command_table.mojo, so the classification is not a
                    # hand-maintained list that can drift. Substrate commands
                    # (FT.*, KV.PREFIX.*, AI.*, V.*) are deliberately NOT writes
                    # here: they own their own stores and never touch this log.
                    if is_not_null(self.dispatcher.wal) \
                       and self.dispatcher.wal[].refusing_writes() \
                       and command_is_write(tp, tl):
                        writer.append_error_response(
                            "MISCONF WAL is full and cannot persist writes; "
                            + "writes are refused. Run SAVE/BGSAVE to checkpoint, "
                            + "raise --wal-max-segments, or start with "
                            + "--wal-full-policy drop to accept the data loss.")
                        i = cmd_end_tok
                        while cmd_idx < num_cmds and i >= cmd_ends[cmd_idx]:
                            cmd_idx += 1
                        continue

                    # gh #261: memory is over --maxmemory. Refuse the commands
                    # Redis flags `denyoom` (plus Pion's substrate ingest), with
                    # Redis's exact error; reads, DEL and the POP family still
                    # run, so a client can free memory under the limit.
                    if self.over_maxmemory and command_is_denyoom(tp, tl) \
                       and not (self.script_depth > 0 and self.script_allow_oom) \
                       and external_call["pion_maxmemory_check", Int32]() != 0:
                        writer.append_error_response("OOM command not allowed when used memory > 'maxmemory'.")
                        i = cmd_end_tok
                        while cmd_idx < num_cmds and i >= cmd_ends[cmd_idx]:
                            cmd_idx += 1
                        continue

                    # ── Tenant namespacing pre-pass (gh #101) ──
                    # Runs at every command boundary when --tenant is configured.
                    # Tenant-bound fds get a deny-by-default allowlist plus a
                    # keyspec-driven rewrite that prefixes "name:" onto every
                    # key token (fail-closed: reject on overflow, never dispatch
                    # un-rewritten). Admin fds (tenant_id == -1) pass untouched.
                    # MULTI queues raw frames above, so EXEC replay re-enters
                    # here per queued command and rewrites deterministically.
                    if self.tenant_table.count > 0:
                        self.cur_tenant_ns_len = 0
                        var _tid = Int(self.tx_state.tenant_id[Int(fd)])
                        if _tid >= 0:
                            var _tspec = tenant_keyspec(tp, tl)
                            var _treject = not _tspec.allowed
                            if not _treject:
                                var _tnp = self.tenant_table.name_ptr(_tid)
                                var _tnl = self.tenant_table.name_len(_tid)
                                if apply_tenant_rewrite(tokens, i, cmd_end_tok, _tspec,
                                                        _tnp, _tnl, self.tenant_scratch):
                                    # Namespace filter for KEYS/SCAN: "name:".
                                    unsafe_memcpy(dest=self.tenant_ns_buf, src=_tnp, count=_tnl)
                                    self.tenant_ns_buf[_tnl] = 58  # ':'
                                    self.cur_tenant_ns_len = _tnl + 1
                                else:
                                    writer.append_error_response("ERR tenant key rewrite exceeds scratch buffer")
                                    _treject = True
                            else:
                                writer.append_error_response("NOPERM this command is not allowed for tenant connections")
                            if _treject:
                                i = cmd_end_tok
                                while cmd_idx < num_cmds and i >= cmd_ends[cmd_idx]:
                                    cmd_idx += 1
                                continue

                    # #39 MONITOR: a command is shown once it has run (the loop
                    # tail), except a script command, shown before it runs so
                    # that what the script calls follows it.
                    var _mon_i = i
                    var _mon_done = self.monitors.count() == 0 or self.monitor_skip
                    if not _mon_done and command_monitor_first(tp, tl):
                        self._monitor_feed(tokens, i, cmd_end_tok, fd, writer, server, kq)
                        _mon_done = True

                    # ── XGPU ──
                    if tl == 4 and cmd_matches_4(tp, 120, 103, 112, 117): # XGPU (x=120, g=103, p=112, u=117)
                        _ = handle_xgpu_info(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    # ── PING ──
                    elif tl == 4 and cmd_matches_4(tp, 112, 105, 110, 103):
                        _ = handle_ping(tokens, i, cmd_end_tok, writer, self.dispatcher)
                        i = cmd_end_tok - 1
                    # ── ECHO ──
                    elif tl == 4 and cmd_matches_4(tp, 101, 99, 104, 111):
                        _ = handle_echo(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    # ── HELLO ──
                    elif tl == 5 and cmd_matches_5(tp, 104, 101, 108, 108, 111):
                        _ = handle_hello(tokens, i, cmd_end_tok, writer, self.cluster, config.server.requirepass, self.tx_state.authed, fd,
                                         rebind[UnsafePointer[TenantTable, MutUntrackedOrigin]](UnsafePointer(to=self.tenant_table)),
                                         self.tx_state.tenant_id,
                                         self.tx_state.resp_proto)
                        i = cmd_end_tok - 1  # consume all HELLO args (incl. AUTH clause)
                    # ── SET (slow path) ──
                    elif tl == 3 and cmd_matches_3(tp, 115, 101, 116):
                        if i + 2 < cmd_end_tok:
                            var key = tokens[i+1].value(); var val_str = tokens[i+2].raw_value()   # gh #334: byte-exact value
                            i += 2
                            # Parse optional NX/XX/GET/EX/PX/EXAT/PXAT/KEEPTTL/IFEQ/IFNE after SET key value
                            var set_expiry_ns = Int64(0)
                            var has_expiry = False
                            var set_expired = False   # gh #393: Redis writes it already expired
                            var keepttl = False
                            var has_nx = False
                            var has_xx = False
                            var has_get = False
                            var has_ifeq = False
                            var has_ifne = False
                            var ifeq_val = String("")
                            var set_err = String("")
                            var n_expiry = 0
                            var j_sopt = i + 1
                            while j_sopt < cmd_end_tok:   # NOT num_tokens: that ran into the next pipelined command
                                var sopt_tl = tokens[j_sopt].length
                                var sopt_tp = tokens[j_sopt].ptr
                                if sopt_tl == 2 and (sopt_tp[0]|0x20)==101 and (sopt_tp[1]|0x20)==120:  # EX
                                    if j_sopt + 1 < cmd_end_tok:
                                        var _e = set_expiry(Int64(strict_atol(tokens[j_sopt+1].value())), 1000, True, _get_now_ns())
                                        if _e.status == SETEXP_INVALID: set_err = String("ERR invalid expire time in 'set' command")
                                        n_expiry += 1
                                        set_expiry_ns = _e.ns; set_expired = _e.status == SETEXP_EXPIRED
                                        has_expiry = True; i += 2; j_sopt += 2
                                    else: set_err = String("ERR syntax error"); break
                                elif sopt_tl == 2 and (sopt_tp[0]|0x20)==112 and (sopt_tp[1]|0x20)==120:  # PX
                                    if j_sopt + 1 < cmd_end_tok:
                                        var _e = set_expiry(Int64(strict_atol(tokens[j_sopt+1].value())), 1, True, _get_now_ns())
                                        if _e.status == SETEXP_INVALID: set_err = String("ERR invalid expire time in 'set' command")
                                        n_expiry += 1
                                        set_expiry_ns = _e.ns; set_expired = _e.status == SETEXP_EXPIRED
                                        has_expiry = True; i += 2; j_sopt += 2
                                    else: set_err = String("ERR syntax error"); break
                                elif sopt_tl == 4 and (sopt_tp[0]|0x20)==101 and (sopt_tp[1]|0x20)==120 and (sopt_tp[2]|0x20)==97 and (sopt_tp[3]|0x20)==116:  # EXAT
                                    if j_sopt + 1 < cmd_end_tok:
                                        var _e = set_expiry(Int64(strict_atol(tokens[j_sopt+1].value())), 1000, False, _get_now_ns())
                                        if _e.status == SETEXP_INVALID: set_err = String("ERR invalid expire time in 'set' command")
                                        n_expiry += 1
                                        set_expiry_ns = _e.ns; set_expired = _e.status == SETEXP_EXPIRED
                                        has_expiry = True; i += 2; j_sopt += 2
                                    else: set_err = String("ERR syntax error"); break
                                elif sopt_tl == 4 and (sopt_tp[0]|0x20)==112 and (sopt_tp[1]|0x20)==120 and (sopt_tp[2]|0x20)==97 and (sopt_tp[3]|0x20)==116:  # PXAT
                                    if j_sopt + 1 < cmd_end_tok:
                                        var _e = set_expiry(Int64(strict_atol(tokens[j_sopt+1].value())), 1, False, _get_now_ns())
                                        if _e.status == SETEXP_INVALID: set_err = String("ERR invalid expire time in 'set' command")
                                        n_expiry += 1
                                        set_expiry_ns = _e.ns; set_expired = _e.status == SETEXP_EXPIRED
                                        has_expiry = True; i += 2; j_sopt += 2
                                    else: set_err = String("ERR syntax error"); break
                                elif sopt_tl == 7 and (sopt_tp[0]|0x20)==107 and (sopt_tp[1]|0x20)==101 and (sopt_tp[2]|0x20)==101 and (sopt_tp[3]|0x20)==112 and (sopt_tp[4]|0x20)==116 and (sopt_tp[5]|0x20)==116 and (sopt_tp[6]|0x20)==108:  # KEEPTTL
                                    keepttl = True; i += 1; j_sopt += 1
                                # NX (2 bytes: n=110, x=120)
                                elif sopt_tl == 2 and (sopt_tp[0]|0x20)==110 and (sopt_tp[1]|0x20)==120:
                                    has_nx = True; i += 1; j_sopt += 1
                                # XX (2 bytes: x=120, x=120)
                                elif sopt_tl == 2 and (sopt_tp[0]|0x20)==120 and (sopt_tp[1]|0x20)==120:
                                    has_xx = True; i += 1; j_sopt += 1
                                # GET (3 bytes: g=103, e=101, t=116)
                                elif sopt_tl == 3 and (sopt_tp[0]|0x20)==103 and (sopt_tp[1]|0x20)==101 and (sopt_tp[2]|0x20)==116:
                                    has_get = True; i += 1; j_sopt += 1
                                # R3: IFEQ (4 bytes: i=105, f=102, e=101, q=113)
                                elif sopt_tl == 4 and (sopt_tp[0]|0x20)==105 and (sopt_tp[1]|0x20)==102 and (sopt_tp[2]|0x20)==101 and (sopt_tp[3]|0x20)==113:
                                    if j_sopt + 1 < cmd_end_tok:
                                        has_ifeq = True; ifeq_val = tokens[j_sopt+1].raw_value()
                                        i += 2; j_sopt += 2
                                    else: set_err = String("ERR syntax error"); break
                                # R3: IFNE (4 bytes: i=105, f=102, n=110, e=101)
                                elif sopt_tl == 4 and (sopt_tp[0]|0x20)==105 and (sopt_tp[1]|0x20)==102 and (sopt_tp[2]|0x20)==110 and (sopt_tp[3]|0x20)==101:
                                    if j_sopt + 1 < cmd_end_tok:
                                        has_ifne = True; ifeq_val = tokens[j_sopt+1].raw_value()
                                        i += 2; j_sopt += 2
                                    else: set_err = String("ERR syntax error"); break
                                else: set_err = String("ERR syntax error"); break
                            # Redis refuses these combinations; they were silently resolved.
                            if set_err == "" and ((has_nx and has_xx) or n_expiry > 1 or (keepttl and n_expiry > 0)):
                                set_err = String("ERR syntax error")
                            if set_err != "":
                                # Refused: nothing written. (Before, `SET k v EX 0` or
                                # `EX -1` answered +OK with a key that had already expired.)
                                writer.append_error_response(set_err)
                                i = cmd_end_tok - 1
                            else:
                                # Check NX/XX/IFEQ/IFNE conditions before executing SET
                                var should_set = True
                                var cur = self.dispatcher.execute_get(key)
                                # GET option: save old value for response
                                var old_val = cur
                                # NX: only set if key does NOT exist
                                if has_nx and not cur.is_none():
                                    should_set = False
                                # XX: only set if key DOES exist
                                if has_xx and cur.is_none():
                                    should_set = False
                                if should_set and (has_ifeq or has_ifne):
                                    if has_ifeq:
                                        if cur.is_none():
                                            should_set = False
                                        else:
                                            # Compare current value bytes with expected
                                            var cur_len = cur.string_len()
                                            var exp_len = ifeq_val.byte_length()
                                            if cur_len != exp_len:
                                                should_set = False
                                            else:
                                                var sso_buf = stack_allocation[24, UInt8]()
                                                var cur_ptr = cur.as_string_safe(sso_buf)
                                                var exp_ptr = ifeq_val.unsafe_ptr()
                                                var eq = True
                                                for ci in range(cur_len):
                                                    if cur_ptr[ci] != exp_ptr[ci]: eq = False; break
                                                should_set = eq
                                    elif has_ifne:
                                        if not cur.is_none():
                                            var cur_len = cur.string_len()
                                            var exp_len = ifeq_val.byte_length()
                                            if cur_len != exp_len:
                                                should_set = True  # different length → not equal → set
                                            else:
                                                var sso_buf = stack_allocation[24, UInt8]()
                                                var cur_ptr = cur.as_string_safe(sso_buf)
                                                var exp_ptr = ifeq_val.unsafe_ptr()
                                                var eq2 = True
                                                for ci in range(cur_len):
                                                    if cur_ptr[ci] != exp_ptr[ci]: eq2 = False; break
                                                should_set = not eq2  # set if NOT equal
                                if should_set:
                                    # GET's reply goes FIRST: old_val borrows the payload
                                    # execute_set frees (it served freed memory after).
                                    if has_get:
                                        writer.append_bulk_value_response(old_val)
                                    if set_expired:
                                        # A deadline Redis resolves as already past
                                        # (EXAT in the past, or PX overflowing ms):
                                        # it writes the key expired, so it is gone.
                                        _ = self.dispatcher.execute_del(key)
                                        if is_not_null(self.ttl_map):
                                            _ = self.ttl_map[].remove(key)
                                    else:
                                        self.dispatcher.execute_set(key, val_str)
                                    if is_not_null(self.ttl_map) and not set_expired:
                                        # gh #394: the String overloads borrow `key`
                                        # for the call; from_string's copy leaked.
                                        if has_expiry:
                                            self.ttl_map[].set(key, GenericValue.from_int(set_expiry_ns))
                                            # gh #174: `SET k v EX n` is the most common way a
                                            # TTL is ever set — logging only the EXPIRE family
                                            # would leave it durable through SAVE but not
                                            # through a crash.
                                            _ = self.dispatcher.wal[].append_u64_val(
                                                25, key.unsafe_ptr(), key.byte_length(),
                                                UInt64(set_expiry_ns),
                                                null_ptr[UInt8, MutUntrackedOrigin](), 0)
                                        elif not keepttl:
                                            _ = self.ttl_map[].remove(key)
                                            # SET without KEEPTTL clears any existing TTL; that
                                            # clear must replay too, or a crash resurrects it.
                                            _ = self.dispatcher.wal[].append(
                                                26, key.unsafe_ptr(), key.byte_length())
                                    if not has_get:
                                        writer.append_ok_response()
                                else:
                                    if has_get:
                                        writer.append_bulk_value_response(old_val)
                                    else:
                                        writer.append_null_response()
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'set' command")
                            i = cmd_end_tok - 1
                    # ── GET ──
                    elif tl == 3 and cmd_matches_3(tp, 103, 101, 116):
                        if i + 1 < cmd_end_tok:
                            # gh #239: an extra argument used to be left behind and
                            # dispatched as its own command — one command in, two
                            # replies out. Redis rejects the arity; either way the
                            # frame must be consumed to cmd_end_tok.
                            if i + 2 < cmd_end_tok:
                                writer.append_error_response("ERR wrong number of arguments for 'get' command")
                            else:
                                var key = tokens[i+1].value(); var val = self.dispatcher.execute_get(key)
                                writer.append_bulk_value_response(val)
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'get' command")
                            i = cmd_end_tok - 1
                    # ── MGET (gh #101: slow-path coverage — tenant connections and
                    #    MULTI replay never touch the fast path) ──
                    elif tl == 4 and cmd_matches_4(tp, 109, 103, 101, 116):
                        if i + 1 < cmd_end_tok:
                            var mget_hdr = String("*") + String(cmd_end_tok - i - 1) + String("\r\n")
                            writer.append_to_response(mget_hdr.unsafe_ptr(), mget_hdr.byte_length())
                            var j_mget = i + 1
                            while j_mget < cmd_end_tok:
                                writer.append_bulk_value_response(self.dispatcher.execute_get(tokens[j_mget].value()))
                                j_mget += 1
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'mget' command")
                            i = cmd_end_tok - 1
                    # ── MSET (gh #101: slow-path coverage) ──
                    elif tl == 4 and cmd_matches_4(tp, 109, 115, 101, 116):
                        if i + 2 < cmd_end_tok and (cmd_end_tok - i - 1) % 2 == 0:
                            var j_mset = i + 1
                            while j_mset + 1 < cmd_end_tok:
                                self.dispatcher.execute_set(tokens[j_mset].value(), tokens[j_mset+1].value())
                                j_mset += 2
                            writer.append_ok_response()
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'mset' command")
                            i = cmd_end_tok - 1
                    # ── DEL (multi-key, gh #111) ──
                    elif tl == 3 and cmd_matches_3(tp, 100, 101, 108):
                        if i + 1 < cmd_end_tok:
                            var del_count = Int64(0)
                            var j_del = i + 1
                            while j_del < cmd_end_tok:
                                var del_key = tokens[j_del].value()
                                if self.dispatcher.execute_del(del_key):
                                    del_count += 1
                                    if is_not_null(self.ttl_map):
                                        _ = self.ttl_map[].remove_generic(GenericValue.borrow(tokens[j_del].ptr, tokens[j_del].length))
                                j_del += 1
                            writer.append_int_response(del_count)
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'del' command")
                            i = cmd_end_tok - 1
                    # ── EXISTS (multi-key) ──
                    elif tl == 6 and cmd_matches_6(tp, 101, 120, 105, 115, 116, 115):
                        if i + 1 < cmd_end_tok:
                            var count = Int64(0)
                            var j_ex = i + 1
                            # gh #239: was `num_tokens` — the WHOLE recv buffer — so
                            # `EXISTS k1 k2` walked past its own command and ate the
                            # next pipelined command as a key. That command then never
                            # got a reply and the client waited forever: a HANG, not
                            # just a desync. Bound is always cmd_end_tok (gh #218).
                            while j_ex < cmd_end_tok:
                                var ex_key = tokens[j_ex].value()
                                if self.dispatcher.execute_exists(ex_key): count += 1
                                j_ex += 1
                                i += 1
                            writer.append_int_response(count)
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'exists' command")
                            i = cmd_end_tok - 1
                    # ── INCR ──
                    elif tl == 4 and cmd_matches_4(tp, 105, 110, 99, 114):
                        if i + 1 < cmd_end_tok:
                            if i + 2 < cmd_end_tok:   # gh #239
                                writer.append_error_response("ERR wrong number of arguments for 'incr' command")
                            else:
                                var key = tokens[i+1].value()
                                # gh #232: an aggregate is WRONGTYPE (the fast path
                                # already says so); execute_incr folds it into "not an integer".
                                var _ik = self.keyspace[].get(GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                                if not _ik.is_none() and _ik.is_container():
                                    writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                                else:
                                    var res = self.dispatcher.execute_incr(key)
                                    if res.is_valid: writer.append_int_response(res.value)
                                    else: writer.append_error_response("ERR value is not an integer or out of range")
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'incr' command")
                            i = cmd_end_tok - 1
                    # ── DECR ──
                    elif tl == 4 and cmd_matches_4(tp, 100, 101, 99, 114):
                        if i + 1 < cmd_end_tok:
                            if i + 2 < cmd_end_tok:   # gh #239
                                writer.append_error_response("ERR wrong number of arguments for 'decr' command")
                            else:
                                var key = tokens[i+1].value()
                                # gh #232: an aggregate is WRONGTYPE (the fast path
                                # already says so); execute_decr folds it into "not an integer".
                                var _ik = self.keyspace[].get(GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                                if not _ik.is_none() and _ik.is_container():
                                    writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                                else:
                                    var res = self.dispatcher.execute_decr(key)
                                    if res.is_valid: writer.append_int_response(res.value)
                                    else: writer.append_error_response("ERR value is not an integer or out of range")
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'decr' command")
                            i = cmd_end_tok - 1
                    # ── LPUSH (multi-value, gh #111) ──
                    elif tl == 5 and cmd_matches_5(tp, 108, 112, 117, 115, 104):
                        if i + 2 < cmd_end_tok:
                            var lpush_key = tokens[i+1].value()
                            var lpush_len = 0
                            var j_lpush = i + 2
                            while j_lpush < cmd_end_tok:
                                lpush_len = self.dispatcher.execute_lpush(lpush_key, tokens[j_lpush].value())
                                j_lpush += 1
                            if lpush_len < 0: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            else: writer.append_int_response(Int64(lpush_len))
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'lpush' command")
                            i = cmd_end_tok - 1
                    # ── RPUSH (multi-value, gh #111) ──
                    elif tl == 5 and cmd_matches_5(tp, 114, 112, 117, 115, 104):
                        if i + 2 < cmd_end_tok:
                            var rpush_key = tokens[i+1].value()
                            var rpush_len = 0
                            var j_rpush = i + 2
                            while j_rpush < cmd_end_tok:
                                rpush_len = self.dispatcher.execute_rpush(rpush_key, tokens[j_rpush].raw_value())
                                j_rpush += 1
                            if rpush_len < 0: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            else: writer.append_int_response(Int64(rpush_len))
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'rpush' command")
                            i = cmd_end_tok - 1
                    # ── LPUSHX / RPUSHX (gh #101: slow-path coverage; push only if the
                    #    list already exists, else :0) ──
                    elif tl == 6 and (cmd_matches_6(tp, 108, 112, 117, 115, 104, 120) or cmd_matches_6(tp, 114, 112, 117, 115, 104, 120)):
                        if i + 2 < cmd_end_tok:
                            var pushx_key = tokens[i+1].value()
                            var pushx_existing = self.keyspace[].get(pushx_key)
                            # gh #232: "exists but is a hash" is a type error,
                            # not "does not exist yet". Answering :0 tells a
                            # caller the list is simply absent.
                            if not pushx_existing.is_none() and pushx_existing.type.value != ValueType.LIST:
                                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            elif pushx_existing.is_none():
                                writer.append_int_response(0)
                            else:
                                var pushx_left = (tp[0]|0x20) == 108
                                var pushx_len = 0
                                var j_pushx = i + 2
                                while j_pushx < cmd_end_tok:
                                    if pushx_left: pushx_len = self.dispatcher.execute_lpush(pushx_key, tokens[j_pushx].raw_value())
                                    else: pushx_len = self.dispatcher.execute_rpush(pushx_key, tokens[j_pushx].raw_value())
                                    j_pushx += 1
                                if pushx_len < 0: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                                else: writer.append_int_response(Int64(pushx_len))
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'lpushx' command")
                            i = cmd_end_tok - 1
                    # ── LPOP ──
                    elif tl == 4 and cmd_matches_4(tp, 108, 112, 111, 112):
                        if i + 1 < cmd_end_tok:
                            # gh #239: the optional COUNT was neither honoured nor
                            # consumed — `LPOP key 2` popped one element and left
                            # the "2" to be dispatched as its own command (one in,
                            # two replies out). Redis returns an ARRAY for the
                            # count form and nil/array for a missing key.
                            var key = tokens[i+1].value()
                            if i + 2 < cmd_end_tok:
                                var _pc = Int(strict_atol(tokens[i+2].value()))
                                if _pc < 0:
                                    writer.append_error_response("ERR value is out of range, must be positive")
                                else:
                                    var _avail = self.dispatcher.execute_llen(key)
                                    if _avail < 0:
                                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                                    elif _avail == 0:
                                        writer.append_null_response()
                                    else:
                                        var _pn = _pc if _pc < _avail else _avail
                                        var _ph = String("*") + String(_pn) + String("\r\n")
                                        writer.append_to_response(_ph.unsafe_ptr(), _ph.byte_length())
                                        for _ in range(_pn):
                                            var _pv = self.dispatcher.execute_lpop(key)
                                            if _pv.is_none(): break
                                            writer.append_bulk_value_response(_pv)
                                            _pv.free_str_payload()   # the pop handed back an owned value; the reply copied it
                                        # gh #234: the count form drains too, so it
                                        # needs the same emptied-key removal.
                                        if self.dispatcher.execute_llen(key) == 0:
                                            _ = remove_and_free(self.keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                                            _ = self.dispatcher.wal[].append(2, tokens[i+1].ptr, tokens[i+1].length)
                            else:
                                var popped = self.dispatcher.execute_lpop(key)
                                if popped.is_none(): writer.append_null_response()
                                else:
                                    writer.append_bulk_value_response(popped)
                                    popped.free_str_payload()   # the pop handed back an owned value; the reply copied it
                                    # gh #234: the FAST-path arm removes an emptied
                                    # list, but a batch containing any non-fast-path
                                    # command (e.g. multi-value RPUSH) routes every
                                    # command here, so the slow path needs it too.
                                    if self.dispatcher.execute_llen(key) == 0:
                                        _ = remove_and_free(self.keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                                        _ = self.dispatcher.wal[].append(2, tokens[i+1].ptr, tokens[i+1].length)
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'lpop' command")
                            i = cmd_end_tok - 1
                    # ── RPOP ──
                    elif tl == 4 and cmd_matches_4(tp, 114, 112, 111, 112):
                        if i + 1 < cmd_end_tok:
                            # gh #239: the optional COUNT was neither honoured nor
                            # consumed — `RPOP key 2` popped one element and left
                            # the "2" to be dispatched as its own command (one in,
                            # two replies out). Redis returns an ARRAY for the
                            # count form and nil/array for a missing key.
                            var key = tokens[i+1].value()
                            if i + 2 < cmd_end_tok:
                                var _pc = Int(strict_atol(tokens[i+2].value()))
                                if _pc < 0:
                                    writer.append_error_response("ERR value is out of range, must be positive")
                                else:
                                    var _avail = self.dispatcher.execute_llen(key)
                                    if _avail < 0:
                                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                                    elif _avail == 0:
                                        writer.append_null_response()
                                    else:
                                        var _pn = _pc if _pc < _avail else _avail
                                        var _ph = String("*") + String(_pn) + String("\r\n")
                                        writer.append_to_response(_ph.unsafe_ptr(), _ph.byte_length())
                                        for _ in range(_pn):
                                            var _pv = self.dispatcher.execute_rpop(key)
                                            if _pv.is_none(): break
                                            writer.append_bulk_value_response(_pv)
                                            _pv.free_str_payload()   # the pop handed back an owned value; the reply copied it
                                        # gh #234: the count form drains too, so it
                                        # needs the same emptied-key removal.
                                        if self.dispatcher.execute_llen(key) == 0:
                                            _ = remove_and_free(self.keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                                            _ = self.dispatcher.wal[].append(2, tokens[i+1].ptr, tokens[i+1].length)
                            else:
                                var popped = self.dispatcher.execute_rpop(key)
                                if popped.is_none(): writer.append_null_response()
                                else:
                                    writer.append_bulk_value_response(popped)
                                    popped.free_str_payload()   # the pop handed back an owned value; the reply copied it
                                    # gh #234: the FAST-path arm removes an emptied
                                    # list, but a batch containing any non-fast-path
                                    # command (e.g. multi-value RPUSH) routes every
                                    # command here, so the slow path needs it too.
                                    if self.dispatcher.execute_llen(key) == 0:
                                        _ = remove_and_free(self.keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                                        _ = self.dispatcher.wal[].append(2, tokens[i+1].ptr, tokens[i+1].length)
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'rpop' command")
                            i = cmd_end_tok - 1
                    # ── LRANGE ──
                    elif tl == 6 and cmd_matches_6(tp, 108, 114, 97, 110, 103, 101):
                         if i + 3 < cmd_end_tok:
                            var key = tokens[i+1].value(); var start_idx = strict_atol(tokens[i+2].value()); var stop_idx = strict_atol(tokens[i+3].value())
                            var list_ptr = self.dispatcher.get_list(key)
                            if is_not_null(list_ptr):
                                var size = list_ptr[].size; var start = start_idx; var stop = stop_idx
                                if start < 0: start = size + start
                                if start < 0: start = 0
                                if stop < 0: stop = size + stop
                                if stop < 0: stop = -1
                                if stop >= size: stop = size - 1
                                if start > stop or start >= size: writer.append_empty_array_response()
                                else:
                                    var count = stop - start + 1
                                    writer.buffer[writer.offset] = 42 # '*'
                                    writer.offset += 1
                                    writer.offset = format_int_to_buf(writer.buffer, writer.offset, Int64(count))
                                    writer.buffer[writer.offset] = 13 # '\r'
                                    writer.buffer[writer.offset + 1] = 10 # '\n'
                                    writer.offset += 2
                                    if is_not_null(list_ptr[].zip_buf):
                                        var offset = 0
                                        var idx = 0
                                        while idx < start and idx < size:
                                            var v_len = Int((list_ptr[].zip_buf + offset).bitcast[UInt16]()[])
                                            offset += 2 + v_len
                                            idx += 1
                                        while idx <= stop and idx < size:
                                            var v_len = Int((list_ptr[].zip_buf + offset).bitcast[UInt16]()[])
                                            writer.append_bulk_string_response(list_ptr[].zip_buf + offset + 2, v_len)
                                            offset += 2 + v_len
                                            idx += 1
                                    else:
                                        var global_idx = 0
                                        comptime SEG = SlabList.SEG_SIZE
                                        var hi = list_ptr[].head_off
                                        # live range only: head_end < SEG after a pop refill, and head
                                        # segs below head_segs_start were handed to the tail side
                                        while hi < list_ptr[].head_end and global_idx <= stop:
                                            if global_idx >= start:
                                                writer.append_bulk_value_response(list_ptr[].active_head_data[hi])
                                            hi += 1
                                            global_idx += 1
                                        var seg = list_ptr[].head_seg_count - 1
                                        while seg >= list_ptr[].head_segs_start and global_idx <= stop:
                                            var j = 0
                                            while j < SEG and global_idx <= stop:
                                                if global_idx >= start:
                                                    writer.append_bulk_value_response(list_ptr[].head_segs[seg][j])
                                                j += 1
                                                global_idx += 1
                                            seg -= 1
                                        var tseg = 0
                                        while tseg < list_ptr[].tail_seg_count and global_idx <= stop:
                                            var j = 0
                                            while j < SEG and global_idx <= stop:
                                                if global_idx >= start:
                                                    writer.append_bulk_value_response(list_ptr[].tail_segs[tseg][j])
                                                j += 1
                                                global_idx += 1
                                            tseg += 1
                                        var tk = 0
                                        while tk < list_ptr[].tail_count and global_idx <= stop:
                                            if global_idx >= start:
                                                writer.append_bulk_value_response(list_ptr[].active_tail_data[tk])
                                            tk += 1
                                            global_idx += 1
                            else: writer.append_empty_array_response()
                            i += 3
                         else:
                             writer.append_error_response("ERR wrong number of arguments for 'lrange' command")
                             i = cmd_end_tok - 1
                    # ── LLEN ──
                    elif tl == 4 and cmd_matches_4(tp, 108, 108, 101, 110):
                        if i + 1 < cmd_end_tok:
                            if i + 2 < cmd_end_tok:   # gh #239
                                writer.append_error_response("ERR wrong number of arguments for 'llen' command")
                            else:
                                var key = tokens[i+1].value(); var len_res = self.dispatcher.execute_llen(key)
                                if len_res < 0: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                                else: writer.append_int_response(Int64(len_res))
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'llen' command")
                            i = cmd_end_tok - 1
                    # ── HSET ──
                    elif tl == 4 and cmd_matches_4(tp, 104, 115, 101, 116):
                        # Every field-value pair, not just the first: this arm
                        # applied pair one, the gh #240 clamp then skipped the
                        # rest, and `HSET k f1 v1 f2 v2` answered 1 with f2 never
                        # written — on every slow-path route (MULTI/EXEC replay,
                        # tenant connections, a pipeline behind a slow command).
                        if i + 3 < cmd_end_tok and (cmd_end_tok - i - 2) % 2 == 0:
                            var key = tokens[i+1].value()
                            var hs_added = Int64(0)
                            var hs_ok = True
                            var j_hs = i + 2
                            while j_hs + 1 < cmd_end_tok:
                                var res = self.dispatcher.execute_hset(key, tokens[j_hs].value(), tokens[j_hs+1].value())
                                if not res.is_valid:
                                    hs_ok = False
                                    break
                                hs_added += res.value
                                # #43: the index's vector field is indexed here too
                                _ = ingest_hash_vector(self.shared_hnsw, self.keyspace, self.dispatcher.wal, self.vec_tomb,
                                                       tokens[i+1].ptr, tokens[i+1].length, tokens[j_hs].ptr,
                                                       tokens[j_hs].length, tokens[j_hs+1].ptr, tokens[j_hs+1].length)
                                j_hs += 2
                            if hs_ok: writer.append_int_response(hs_added)
                            else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'hset' command")
                            i = cmd_end_tok - 1
                    # ── HGET (with R3 lazy field expiry) ──
                    elif tl == 4 and cmd_matches_4(tp, 104, 103, 101, 116):
                        if i + 2 < cmd_end_tok:
                            # gh #392: hash_get_live deletes this hash's expired
                            # fields (and the key with its last one) before the read.
                            var hg_v = hash_get_live(self.keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                            if not hg_v.is_none() and hg_v.type.value != ValueType.HASH:
                                # gh #232: the fast path already says WRONGTYPE; this
                                # arm answered nil, so the reply depended on the path.
                                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            elif hg_v.is_none():
                                writer.append_null_response()
                            else:
                                var field_val = hg_v.as_hash().bitcast[SlabHashMap]()[].get(GenericValue.borrow(tokens[i+2].ptr, tokens[i+2].length))
                                if field_val.is_none(): writer.append_null_response()
                                else: writer.append_bulk_value_response(field_val)
                            i += 2
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'hget' command")
                            i = cmd_end_tok - 1
                    # ── HMSET (legacy multi-field HSET, replies +OK; gh #101 slow-path coverage) ──
                    elif tl == 5 and cmd_matches_5(tp, 104, 109, 115, 101, 116):
                        if i + 3 < cmd_end_tok and (cmd_end_tok - i - 2) % 2 == 0:
                            var hmset_key = tokens[i+1].value()
                            var hmset_wrong = False
                            var j_hmset = i + 2
                            while j_hmset + 1 < cmd_end_tok:
                                var hmset_res = self.dispatcher.execute_hset(hmset_key, tokens[j_hmset].value(), tokens[j_hmset+1].value())
                                if not hmset_res.is_valid:
                                    hmset_wrong = True
                                    break
                                _ = ingest_hash_vector(self.shared_hnsw, self.keyspace, self.dispatcher.wal, self.vec_tomb,   # #43
                                                       tokens[i+1].ptr, tokens[i+1].length, tokens[j_hmset].ptr,
                                                       tokens[j_hmset].length, tokens[j_hmset+1].ptr, tokens[j_hmset+1].length)
                                j_hmset += 2
                            if hmset_wrong: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            else: writer.append_ok_response()
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'hmset' command")
                            i = cmd_end_tok - 1
                    # ── HSTRLEN (gh #101 slow-path coverage) ──
                    elif tl == 7 and cmd_matches_7(tp, 104, 115, 116, 114, 108, 101, 110):
                        if i + 2 < cmd_end_tok:
                            # gh #232: execute_hget cannot distinguish "no such
                            # hash" from "that key is a list", so both answered
                            # :0. Check the key's type first.
                            var hstrlen_kv = self.keyspace[].get(GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                            if not hstrlen_kv.is_none() and hstrlen_kv.type.value != ValueType.HASH:
                                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            else:
                                var hstrlen_val = self.dispatcher.execute_hget(tokens[i+1].value(), tokens[i+2].value())
                                if hstrlen_val.is_none(): writer.append_int_response(0)
                                else: writer.append_int_response(Int64(hstrlen_val.string_len()))
                            i += 2
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'hstrlen' command")
                            i = cmd_end_tok - 1
                    # ── SADD (multi-member, gh #111) ──
                    elif tl == 4 and cmd_matches_4(tp, 115, 97, 100, 100):
                        if i + 2 < cmd_end_tok:
                            var sadd_key = tokens[i+1].value()
                            var sadd_count = Int64(0)
                            var sadd_wrongtype = False
                            var j_sadd = i + 2
                            while j_sadd < cmd_end_tok:
                                var sadd_res = self.dispatcher.execute_sadd(sadd_key, tokens[j_sadd].value())
                                if sadd_res.is_valid: sadd_count += sadd_res.value
                                else:
                                    sadd_wrongtype = True; break
                                j_sadd += 1
                            if sadd_wrongtype: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            else: writer.append_int_response(sadd_count)
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'sadd' command")
                            i = cmd_end_tok - 1
                    # ── ZADD (multi-pair, gh #111; single-pair no-flag form is served on the
                    #    fast path. Flag forms — NX/XX/GT/LT/CH/INCR — are rejected here rather
                    #    than silently misparsed as scores; full flag support is future work) ──
                    elif tl == 4 and cmd_matches_4(tp, 122, 97, 100, 100):
                        if i + 2 < cmd_end_tok:
                            var zadd_key = tokens[i+1].value()
                            var zadd_added = Int64(0)
                            var zadd_wrongtype = False
                            var zadd_badfloat = False
                            # gh #237: leading flags. Consume them before the
                            # score/member pairs; anything else is a score.
                            var _znx = False; var _zxx = False; var _zgt = False
                            var _zlt = False; var _zch = False; var _zincr = False
                            var j_zadd = i + 2
                            while j_zadd < cmd_end_tok:
                                var ft = tokens[j_zadd]
                                var fp = ft.ptr
                                if ft.length == 2 and (fp[0]|0x20) == 110 and (fp[1]|0x20) == 120: _znx = True
                                elif ft.length == 2 and (fp[0]|0x20) == 120 and (fp[1]|0x20) == 120: _zxx = True
                                elif ft.length == 2 and (fp[0]|0x20) == 103 and (fp[1]|0x20) == 116: _zgt = True
                                elif ft.length == 2 and (fp[0]|0x20) == 108 and (fp[1]|0x20) == 116: _zlt = True
                                elif ft.length == 2 and (fp[0]|0x20) == 99 and (fp[1]|0x20) == 104: _zch = True
                                elif ft.length == 4 and (fp[0]|0x20) == 105 and (fp[1]|0x20) == 110 and (fp[2]|0x20) == 99 and (fp[3]|0x20) == 114: _zincr = True
                                else: break
                                j_zadd += 1
                            var _zchanged = Int64(0)
                            var _zincr_score = Float64(0.0)
                            var _zincr_applied = False
                            var _zincr_nan = False
                            var _zbadopt = False
                            var _zbadmsg = String("")
                            # Redis rejects these combinations outright, each
                            # with its own message.
                            if _znx and _zxx:
                                _zbadopt = True
                                _zbadmsg = "ERR XX and NX options at the same time are not compatible"
                            elif (_znx and (_zgt or _zlt)) or (_zgt and _zlt):
                                _zbadopt = True
                                _zbadmsg = "ERR GT, LT, and/or NX options at the same time are not compatible"
                            elif _zincr and (j_zadd + 2) < cmd_end_tok:
                                _zbadopt = True   # INCR takes exactly one pair
                                _zbadmsg = "ERR INCR option supports a single increment-element pair"
                            # gh #393: every score is parsed — Redis's rules, "inf"
                            # included — BEFORE any pair is applied. A bad score in
                            # pair N used to leave pairs 1..N-1 applied behind the
                            # error; Redis applies nothing.
                            var _zscores = List[Float64]()
                            if not _zbadopt:
                                var j_chk = j_zadd
                                while j_chk + 1 < cmd_end_tok:
                                    var _zps = parse_redis_double(tokens[j_chk].ptr, tokens[j_chk].length, DOUBLE_VALUE)
                                    if not _zps.ok:
                                        zadd_badfloat = True; break
                                    _zscores.append(_zps.value)
                                    j_chk += 2
                            if not _zbadopt and not zadd_badfloat:
                                var _zpair = 0
                                while j_zadd + 1 < cmd_end_tok:
                                    var _zo = self.dispatcher.execute_zadd_cond(
                                        zadd_key, _zscores[_zpair], tokens[j_zadd+1].value(),
                                        _znx, _zxx, _zgt, _zlt, _zincr)
                                    _zpair += 1
                                    if not _zo.ok:
                                        zadd_wrongtype = True; break
                                    if _zo.nan:
                                        _zincr_nan = True; break
                                    if _zo.applied:
                                        zadd_added += _zo.added
                                        _zchanged += _zo.changed
                                        _zincr_score = _zo.new_score
                                        _zincr_applied = True
                                    j_zadd += 2
                            if _zbadopt: writer.append_error_response(_zbadmsg)
                            elif zadd_badfloat: writer.append_error_response("ERR value is not a valid float")
                            elif zadd_wrongtype: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            elif _zincr_nan: writer.append_error_response("ERR resulting score is not a number (NaN)")
                            elif _zincr:
                                # INCR replies with the NEW score, or nil when a
                                # flag suppressed the write.
                                if _zincr_applied:
                                    # Redis answers with addReplyDouble, as ZINCRBY
                                    # does: d2string digits, and a RESP3 double.
                                    # This went through Float32 once (`ZADD z INCR
                                    # 123456789 m` answered 123456792), then through
                                    # 17 fixed decimals (1/3 as 0.33333333333333331).
                                    writer.append_score_response(_zincr_score)
                                else:
                                    writer.append_null_response()
                            elif _zch: writer.append_int_response(_zchanged)
                            else: writer.append_int_response(zadd_added)
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'zadd' command")
                            i = cmd_end_tok - 1
                    # ── SPOP ──
                    # SPOP key [count] — the fast path's semantics. This arm
                    # ignored the count: `SPOP k 3` popped ONE member and answered
                    # a bare bulk string where the fast path (and Redis) answer an
                    # array, so the reply shape depended on which path ran it.
                    elif tl == 4 and cmd_matches_4(tp, 115, 112, 111, 112):
                        if i + 1 < cmd_end_tok and i + 3 >= cmd_end_tok:
                            var key = tokens[i+1].value()
                            var sp_v = self.keyspace[].get(GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                            var sp_has_count = i + 2 < cmd_end_tok
                            var sp_count = Int64(1)
                            var sp_err = String("")
                            if sp_has_count:
                                var _spc = parse_int64_strict(tokens[i+2].ptr, tokens[i+2].length)
                                if not _spc.ok:
                                    sp_err = "ERR value is not an integer or out of range"
                                elif _spc.value < 0:
                                    sp_err = "ERR value is out of range, must be positive"
                                sp_count = _spc.value
                            if sp_err != "":
                                writer.append_error_response(sp_err)
                            elif not sp_v.is_none() and sp_v.type.value != ValueType.SET:
                                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            elif sp_v.is_none():
                                if sp_has_count: writer.append_set_header(0)   # RESP3 `~0`
                                else: writer.append_null_response()
                            else:
                                var sp_set = sp_v.as_set().bitcast[SlabHashMap]()
                                var sp_n = 1
                                if sp_has_count:
                                    sp_n = Int(min(sp_count, Int64(sp_set[].size)))
                                    writer.append_set_header(sp_n)   # RESP3: a set, as Redis
                                for _ in range(sp_n):
                                    var popped = self.dispatcher.execute_spop(key)   # logs the SREM
                                    writer.append_bulk_value_response(popped)
                                    popped.free_str_payload()   # gh #394: pop_random hands the member over
                                # gh #234: an emptied set is removed.
                                if sp_set[].size == 0:
                                    _ = remove_and_free(self.keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                                    _ = self.dispatcher.wal[].append(2, tokens[i+1].ptr, tokens[i+1].length)
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'spop' command")
                        i = cmd_end_tok - 1
                    # ── FLUSHALL ──
                    elif tl == 8 and cmd_matches_8(tp, 102, 108, 117, 115, 104, 97, 108, 108):
                        _ = handle_flushall(tokens, i, cmd_end_tok, self.keyspace, writer, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── SAVE ──
                    elif tl == 4 and cmd_matches_4(tp, 115, 97, 118, 101):
                        _ = handle_save(self.snapshot_engine, self.keyspace, self.worker_id, self.dispatcher, self.last_save_time, writer, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── BGSAVE ──
                    elif tl == 6 and cmd_matches_6(tp, 98, 103, 115, 97, 118, 101):
                        _ = handle_bgsave(self.snapshot_engine, self.keyspace, self.worker_id, self.dispatcher, self.last_save_time, writer, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── LASTSAVE ──
                    elif tl == 8 and cmd_matches_8(tp, 108, 97, 115, 116, 115, 97, 118, 101):
                        _ = handle_lastsave(self.last_save_time, writer)
                        i = cmd_end_tok - 1
                    # ── SHUTDOWN [NOSAVE|SAVE] (gh #259) ──
                    # admin.mojo's docstring has listed this for a long time and
                    # nothing implemented it, so the only way to stop a server was
                    # a signal — which, before gh #259, was the lossy path.
                    #
                    # Routes to the SAME latch as SIGTERM rather than exiting here:
                    # the drain has to happen on the event loop (msync from inside
                    # a dispatch is fine, but exiting mid-batch would abandon the
                    # rest of the pipelined buffer and every other worker's WAL).
                    #
                    # Redis sends NO reply on success — the client observes the
                    # close — and errors only if it cannot comply. Matching that:
                    # SAVE runs synchronously first so its reply ordering is
                    # irrelevant, then the latch is set and we write nothing.
                    elif tl == 8 and cmd_eq(tp, tl, "shutdown"):
                        var _sd_nosave = False
                        if i + 1 < cmd_end_tok:
                            var _ap = tokens[unsafe_offset=i+1].ptr
                            var _al = tokens[unsafe_offset=i+1].length
                            if cmd_eq(_ap, _al, "nosave"):
                                _sd_nosave = True
                        if not _sd_nosave:
                            _ = handle_save(self.snapshot_engine, self.keyspace,
                                            self.worker_id, self.dispatcher,
                                            self.last_save_time, writer, self.ttl_map)
                            # handle_save wrote +OK; SHUTDOWN must not reply, so
                            # roll that back rather than desyncing the client.
                            writer.offset = cmd_write_start
                        external_call["pion_request_shutdown", NoneType]()
                        i = cmd_end_tok - 1
                    # ── TTL/Expiry Commands (src/commands/ttl.mojo) ──
                    elif tl == 6 and cmd_matches_6(tp, 101, 120, 112, 105, 114, 101):
                        _ = handle_expire(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    elif tl == 7 and cmd_matches_7(tp, 112, 101, 120, 112, 105, 114, 101):
                        _ = handle_pexpire(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    elif tl == 8 and cmd_matches_8(tp, 101, 120, 112, 105, 114, 101, 97, 116):
                        _ = handle_expireat(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    elif tl == 9 and (tp[0]|0x20)==112 and (tp[1]|0x20)==101 and (tp[2]|0x20)==120 and (tp[3]|0x20)==112 and (tp[4]|0x20)==105 and (tp[5]|0x20)==114 and (tp[6]|0x20)==101 and (tp[7]|0x20)==97 and (tp[8]|0x20)==116:
                        _ = handle_pexpireat(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    elif tl == 3 and cmd_matches_3(tp, 116, 116, 108):
                        _ = handle_ttl(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    elif tl == 4 and cmd_matches_4(tp, 112, 116, 116, 108):
                        _ = handle_pttl(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    elif tl == 7 and cmd_matches_7(tp, 112, 101, 114, 115, 105, 115, 116):
                        _ = handle_persist(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── INFO ──
                    elif tl == 4 and cmd_matches_4(tp, 105, 110, 102, 111):
                        # INFO [section ...]. One body for every form (#30):
                        # the Replication and Cluster sections now come from the
                        # cluster state for plain INFO too, which printed
                        # role:master on a replica; `INFO replication` was the
                        # only form that read it. Sections filter as in Redis.
                        var _isecs = List[String]()
                        for _ij in range(i + 1, cmd_end_tok):
                            _isecs.append(tokens[_ij].value())
                        var _ik = 0
                        for _is in range(8):
                            _ik += self.keyspace[].shards[unsafe_offset=_is].size
                        var _ie = 0
                        if is_not_null(self.ttl_map): _ie = self.ttl_map[].size
                        var _icl = is_not_null(self.cluster) and self.cluster[].enabled
                        _ = handle_info(self.dispatcher, writer, self.listen_port, _ik, _ie,
                                        self.ledger.uptime_seconds(), self._value_receipt_info(),
                                        self._info_replication_section(), _icl, _isecs)
                        i = cmd_end_tok - 1
                    # ── GETBIT ──
                    elif tl == 6 and cmd_matches_6(tp, 103, 101, 116, 98, 105, 116):
                        if i + 2 < cmd_end_tok:
                            var key = tokens[i+1].value(); var offset = strict_atol(tokens[i+2].value())
                            if offset < 0 or offset >= 4294967296:   # read before the bitmap otherwise
                                writer.append_error_response("ERR bit offset is not an integer or out of range")
                                i = cmd_end_tok - 1; continue
                            var res = self.dispatcher.execute_getbit(key, offset)
                            if res.is_valid: writer.append_int_response(res.value)
                            else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            i += 2
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'getbit' command")
                            i = cmd_end_tok - 1
                    # ── SETBIT ──
                    elif tl == 6 and cmd_matches_6(tp, 115, 101, 116, 98, 105, 116):
                        if i + 3 < cmd_end_tok:
                            var key = tokens[i+1].value(); var offset = strict_atol(tokens[i+2].value()); var value = strict_atol(tokens[i+3].value())
                            # Bounded as on the fast path: a negative offset wrote
                            # before the bitmap (reachable via MULTI / tenant fds).
                            if offset < 0 or offset >= 4294967296:
                                writer.append_error_response("ERR bit offset is not an integer or out of range")
                                i = cmd_end_tok - 1; continue
                            if value != 0 and value != 1:
                                writer.append_error_response("ERR bit is not an integer or out of range")
                                i += 3; continue
                            var res = self.dispatcher.execute_setbit(key, offset, value)
                            if res.is_valid: writer.append_int_response(res.value)
                            else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            i += 3
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'setbit' command")
                            i = cmd_end_tok - 1
                    # ── PFADD ──
                    elif tl == 5 and cmd_matches_5(tp, 112, 102, 97, 100, 100):
                        if i + 1 < cmd_end_tok:
                            var key = tokens[i+1].value(); var elements = List[String](); var element_idx = i + 2
                            while element_idx < cmd_end_tok:   # not num_tokens: the next command's tokens are not elements
                                elements.append(tokens[element_idx].value()); element_idx += 1
                            var res = self.dispatcher.execute_pfadd(key, elements)
                            if res.is_valid: writer.append_int_response(res.value)
                            else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            i = element_idx - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'pfadd' command")
                            i = cmd_end_tok - 1
                    # ── PFCOUNT (multi-key; gh #101 slow-path coverage) ──
                    elif tl == 7 and cmd_matches_7(tp, 112, 102, 99, 111, 117, 110, 116):
                        if i + 1 < cmd_end_tok:
                            var pfc_keys = List[String]()
                            var j_pfc = i + 1
                            while j_pfc < cmd_end_tok:
                                pfc_keys.append(tokens[j_pfc].value()); j_pfc += 1
                            var pfc_res = self.dispatcher.execute_pfcount(pfc_keys)
                            if pfc_res.is_valid: writer.append_int_response(pfc_res.value)
                            else: writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'pfcount' command")
                            i = cmd_end_tok - 1
                    # ── BITCOUNT, every form (public #31). The fast path answers the
                    #    whole-key form outside transactions. ──
                    elif tl == 8 and cmd_matches_8(tp, 98, 105, 116, 99, 111, 117, 110, 116):
                        _ = handle_bitcount(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── GEOADD ── (src/commands/geo.mojo: validates every triple
                    #    first, NX/XX/CH, a sorted set as in Redis)
                    elif tl == 6 and cmd_matches_6(tp, 103, 101, 111, 97, 100, 100):
                        _ = handle_geoadd(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── CONFIG ──
                    elif tl == 6 and cmd_matches_6(tp, 99, 111, 110, 102, 105, 103):
                        var _cfg_reset = False
                        _ = handle_config(tokens, i, cmd_end_tok, writer, config, _cfg_reset)
                        if _cfg_reset:
                            self.ledger.reset()       # CONFIG RESETSTAT: the counters INFO reports
                        i = cmd_end_tok - 1
                    # ── CLUSTER ──
                    elif tl == 7 and cmd_matches_7(tp, 99, 108, 117, 115, 116, 101, 114):
                        _ = handle_cluster(tokens, i, cmd_end_tok, writer, self.cluster, self.keyspace, self.dispatcher.raft, self.shared_hnsw, self.v_store, self.attn_idx, self.worker_id, self.num_workers)
                        i = cmd_end_tok - 1
                    # ── REPLCONF ──
                    elif cmd_eq(tp, tl, "replconf"):
                        # #39: Redis's answers for a client that is not a replica
                        # (ACK/GETACK answer nothing). It answered +OK to all.
                        handle_replconf(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    # ── PSYNC ──
                    elif cmd_eq(tp, tl, "psync"):
                        # #39: refused. It answered +OK, and a Redis replica
                        # then waited forever for an RDB that never came.
                        handle_sync(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    # ── DUMP ──
                    elif tl == 4 and cmd_matches_4(tp, 100, 117, 109, 112):
                        _ = handle_dump(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── RESTORE ──
                    elif tl == 7 and cmd_matches_7(tp, 114, 101, 115, 116, 111, 114, 101):
                        if handle_restore(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.dispatcher.wal):
                            self.tx_state.bump_key_version(tokens[i + 1].ptr, tokens[i + 1].length)
                            _ = ingest_whole_hash(self.shared_hnsw, self.keyspace, self.dispatcher.wal,   # #46
                                                  self.vec_tomb, tokens[i + 1].ptr, tokens[i + 1].length)
                        i = cmd_end_tok - 1
                    # ── MIGRATE ──
                    elif tl == 7 and cmd_matches_7(tp, 109, 105, 103, 114, 97, 116, 101):
                        var _moved = handle_migrate(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map,
                                                    self.dispatcher.wal, is_not_null(self.cluster) and self.cluster[].enabled)
                        for _mk in range(len(_moved)):
                            self.tx_state.bump_key_version(tokens[_moved[_mk]].ptr, tokens[_moved[_mk]].length)
                        i = cmd_end_tok - 1
                    # ── FT.* command family ──
                    # gh #156/#162 family: every handler gets `cmd_end_tok` as its
                    # token bound (their `num_tokens` param is only ever used as a
                    # scan/guard bound), and every site sets `i = cmd_end_tok - 1`.
                    # FT.SEARCH's optional-arg scan used to run to `num_tokens` —
                    # in a pipelined buffer it walked INTO the next command: every
                    # command behind an FT.SEARCH was silently swallowed, and a
                    # second pipelined FT.SEARCH's PARAMS blob overwrote the first
                    # query's blob (request 1 answered with query 2's results).
                    elif tl >= 7 and (tp[0]|0x20)==102 and (tp[1]|0x20)==116 and tp[2]==46:
                        var ft_sub = tp[3] | 0x20
                        if cmd_eq(tp, tl, "ft.info"): # FT.INFO
                            _ = handle_ft_info(tokens, i, cmd_end_tok, hnsw, self.shared_hnsw, writer)
                            i = cmd_end_tok - 1
                        elif cmd_eq(tp, tl, "ft.dropindex"): # FT.DROPINDEX
                            _ = handle_ft_dropindex(tokens, i, cmd_end_tok, hnsw, self.shared_hnsw, writer, self.worker_id)
                            i = cmd_end_tok - 1
                        elif cmd_eq(tp, tl, "ft.optimize"): # FT.OPTIMIZE
                            _ = handle_ft_optimize(tokens, i, cmd_end_tok, hnsw, self.shared_hnsw, self.keyspace, self.worker_id, writer)
                            # #46: the slots already dead, under the new build's id
                            if is_not_null(self.shared_hnsw) and is_not_null(self.shared_hnsw[].ready_atomic) \
                                    and self.shared_hnsw[].ready_atomic[] != 0:
                                record_all_dead(self.shared_hnsw, self.vec_tomb, self.dispatcher)
                            i = cmd_end_tok - 1
                        elif cmd_eq(tp, tl, "ft.create"): # FT.CREATE
                            _ = handle_ft_create(tokens, i, cmd_end_tok, hnsw, self.shared_hnsw, config, writer, self.worker_id)
                            i = cmd_end_tok - 1
                        elif cmd_eq(tp, tl, "ft.addtext"): # FT.ADDTEXT
                            _ = handle_ft_addtext(tokens, i, cmd_end_tok, hnsw, self.shared_hnsw, self.keyspace, self.dispatcher, self.scache, writer)
                            i = cmd_end_tok - 1
                        elif cmd_eq(tp, tl, "ft.searchtext"): # FT.SEARCHTEXT
                            _ = handle_ft_searchtext(tokens, i, cmd_end_tok, self.scache, writer)
                            i = cmd_end_tok - 1
                        elif cmd_eq(tp, tl, "ft.hybrid"): # FT.HYBRID
                            self.ledger.vector_queries += UInt64(1)   # gh #262
                            # A -1 return means the handler already flushed its own
                            # reply — the dispatch loop continues either way so the
                            # rest of a pipelined batch is never dropped (this site
                            # used to `return consumed_bytes`, eating the tail).
                            _ = handle_ft_hybrid(tokens, i, cmd_end_tok, hnsw, self.shared_hnsw,
                                self.keyspace, self.scache, fd, writer, server, kq, consumed_bytes)
                            i = cmd_end_tok - 1
                        elif cmd_eq(tp, tl, "ft.search"): # FT.SEARCH
                            self.ledger.vector_queries += UInt64(1)   # gh #262
                            # Pointers to the per-worker deferred-shard state. `rebind`, not an
                            # Int round-trip: an address laundered through Int loses provenance
                            # to `self`, and -O3 may then keep these fields' stores in
                            # registers across the call (gh #349).
                            var _df_fds_ptr = rebind[UnsafePointer[Array[Int32, 16], MutUntrackedOrigin]](UnsafePointer(to=self.deferred_fds))
                            var _df_seqs_ptr = rebind[UnsafePointer[Array[UInt64, 16], MutUntrackedOrigin]](UnsafePointer(to=self.deferred_seqs))
                            var _df_ks_ptr = rebind[UnsafePointer[Array[Int32, 16], MutUntrackedOrigin]](UnsafePointer(to=self.deferred_ks))
                            var _df_ns_ptr = rebind[UnsafePointer[Array[Int32, 16], MutUntrackedOrigin]](UnsafePointer(to=self.deferred_n_shards))
                            var _df_act_ptr = rebind[UnsafePointer[Array[UInt32, 16], MutUntrackedOrigin]](UnsafePointer(to=self.deferred_active))
                            var _df_done_ptr = rebind[UnsafePointer[Array[UInt32, 16], MutUntrackedOrigin]](UnsafePointer(to=self.deferred_done))
                            var _df_dt_ptr = rebind[UnsafePointer[Array[Int32, 16], MutUntrackedOrigin]](UnsafePointer(to=self.deferred_drain_ticks))
                            var _df_cnt_ptr = rebind[UnsafePointer[Int, MutUntrackedOrigin]](UnsafePointer(to=self.deferred_count))
                            # The handler's token bound is cmd_end_tok (NOT
                            # num_tokens): its FILTER/PARAMS/LIMIT scan walks every
                            # unrecognized token, so a num_tokens bound made it eat
                            # the rest of a pipelined batch. The return value (-1 =
                            # already flushed, else a token index) is deliberately
                            # ignored for accounting — the parser's command boundary
                            # is authoritative at every dispatch site.
                            var _p3_dead_start = -1
                            _ = handle_ft_search(
                                tokens, i, cmd_end_tok, hnsw, self.shared_hnsw, self.keyspace,
                                self.worker_id, self.shard_query_seq, self.scratch_dists,
                                _p3_dead_start,  # gh #207: fresh -1 per command — every query runs its own upper-level greedy (the per-tick cached-start is deleted; reusing another query's entry point was a recall risk for pipelined batches)
                                _df_fds_ptr, _df_seqs_ptr, _df_ks_ptr, _df_ns_ptr,
                                _df_act_ptr, _df_done_ptr, _df_dt_ptr, _df_cnt_ptr,
                                fd, writer, server, kq, consumed_bytes,
                            )
                            i = cmd_end_tok - 1  # -1 because i += 1 at loop bottom
                        else:
                            writer.append_error_response("ERR unknown FT.* subcommand")
                    # ── AI.CHAT ──
                    elif cmd_eq(tp, tl, "ai.chat"):
                        _ = handle_ai_chat(tokens, i, cmd_end_tok, writer, self.keyspace, self.llm_client, self.llm_out_buf, self.scache)
                        i = cmd_end_tok - 1
                    # ── AI.FLARE ──
                    elif cmd_eq(tp, tl, "ai.flare"):
                        _ = handle_ai_flare(tokens, i, cmd_end_tok, writer, self.flare)
                        i = cmd_end_tok - 1
                    # ── AI.COMPLETE ──
                    elif cmd_eq(tp, tl, "ai.complete"):
                        _ = handle_ai_complete(tokens, i, cmd_end_tok, writer, self.scache, self.llm_client, server, fd, kq)
                        i = cmd_end_tok - 1
                    # ── AI.SEMANTIC_CACHE ── (17 bytes, starts with ai.s)
                    elif cmd_eq(tp, tl, "ai.semantic_cache"):
                        _ = handle_ai_semantic_cache(tokens, i, cmd_end_tok, writer, self.scache, server, fd, kq)
                        i = cmd_end_tok - 1
                    # ── AI.EMBED ──
                    elif tl == 8 and (tp[0]|0x20)==97 and (tp[1]|0x20)==105 and tp[2]==46 and (tp[3]|0x20)==101 and (tp[4]|0x20)==109 and (tp[5]|0x20)==98 and (tp[6]|0x20)==101 and (tp[7]|0x20)==100:
                        _ = handle_ai_embed(tokens, i, cmd_end_tok, writer, self.inference_bridge, self.scache)
                        i = cmd_end_tok - 1
                    # ── AI.GENERATE ──
                    elif tl == 11 and (tp[0]|0x20)==97 and (tp[1]|0x20)==105 and tp[2]==46 and (tp[3]|0x20)==103 and (tp[4]|0x20)==101 and (tp[5]|0x20)==110 and (tp[6]|0x20)==101 and (tp[7]|0x20)==114 and (tp[8]|0x20)==97 and (tp[9]|0x20)==116 and (tp[10]|0x20)==101:
                        _ = handle_ai_generate(tokens, i, cmd_end_tok, writer, self.keyspace, self.inference_bridge)
                        i = cmd_end_tok - 1
                    # ── AI.LOADMODEL ──
                    elif tl == 12 and (tp[0]|0x20)==97 and (tp[1]|0x20)==105 and tp[2]==46 and (tp[3]|0x20)==108 and (tp[4]|0x20)==111 and (tp[5]|0x20)==97 and (tp[6]|0x20)==100 and (tp[7]|0x20)==109 and (tp[8]|0x20)==111 and (tp[9]|0x20)==100 and (tp[10]|0x20)==101 and (tp[11]|0x20)==108:
                        _ = handle_ai_loadmodel(tokens, i, cmd_end_tok, writer, self.inference_bridge)
                        i = cmd_end_tok - 1
                    # ── AI.MEMORY ──
                    elif tl == 9 and (tp[0]|0x20)==97 and (tp[1]|0x20)==105 and tp[2]==46 and (tp[3]|0x20)==109 and (tp[4]|0x20)==101 and (tp[5]|0x20)==109 and (tp[6]|0x20)==111 and (tp[7]|0x20)==114 and (tp[8]|0x20)==121:
                        _ = handle_ai_memory(tokens, i, cmd_end_tok, writer, self.keyspace, self.scache, self.dispatcher.list_pool, server, fd, kq)
                        i = cmd_end_tok - 1
                    # ── AI.KNN_LM.STOREBATCH ── (20 bytes)
                    elif cmd_eq(tp, tl, "ai.knn_lm.storebatch"):
                        _ = handle_ai_knn_lm_storebatch(tokens, i, num_tokens, writer, self.knn_lm)
                        i = cmd_end_tok - 1
                    # ── AI.KNN_LM.CREATE ── (16 bytes; case-folded byte 10 == 'c')
                    elif cmd_eq(tp, tl, "ai.knn_lm.create"):
                        _ = handle_ai_knn_lm_create(tokens, i, num_tokens, writer, self.knn_lm)
                        i = cmd_end_tok - 1
                    # ── AI.KNN_LM.STORE / AI.KNN_LM.QUERY ── (both 15 bytes; disambiguate on byte 10)
                    elif cmd_eq(tp, tl, "ai.knn_lm.store"):
                        # AI.KNN_LM.STORE
                        _ = handle_ai_knn_lm_store(tokens, i, num_tokens, writer, self.knn_lm)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "ai.knn_lm.query"):
                        # AI.KNN_LM.QUERY
                        _ = handle_ai_knn_lm_query(tokens, i, num_tokens, writer, self.knn_lm)
                        i = cmd_end_tok - 1
                    # ── AI.KNN_LM.INFO / AI.KNN_LM.DROP ── (both 14 bytes; disambiguate on byte 10)
                    elif cmd_eq(tp, tl, "ai.knn_lm.info"):
                        # AI.KNN_LM.INFO
                        _ = handle_ai_knn_lm_info(tokens, i, num_tokens, writer, self.knn_lm)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "ai.knn_lm.drop"):
                        # AI.KNN_LM.DROP
                        _ = handle_ai_knn_lm_drop(tokens, i, num_tokens, writer, self.knn_lm)
                        i = cmd_end_tok - 1
                    # ── MOE.EXPERT.* (gh #61 Phase-0 Stage-1 wire-surface registration) ──
                    # Common prefix: tp[0..2]="moe", tp[3]='.', tp[10]='.' (after "MOE.EXPERT.")
                    # Stage 1 handlers stub UNAVAILABLE/+OK until backend lands in Stage 2.
                    # All handlers use i = cmd_end_tok - 1 to skip the right number of RESP tokens.
                    elif cmd_eq(tp, tl, "moe.expert.fetch"):
                        # MOE.EXPERT.FETCH — needs fd for writev big-blob response
                        _ = handle_moe_expert_fetch(tokens, i, num_tokens, writer, self.moe_tier, fd)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "moe.expert.stats"):
                        # MOE.EXPERT.STATS
                        _ = handle_moe_expert_stats(tokens, i, num_tokens, writer, self.moe_tier)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "moe.expert.unpin"):
                        # MOE.EXPERT.UNPIN
                        _ = handle_moe_expert_unpin(tokens, i, num_tokens, writer, self.moe_tier)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "moe.expert.info"):
                        # MOE.EXPERT.INFO
                        _ = handle_moe_expert_info(tokens, i, num_tokens, writer, self.moe_tier)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "moe.expert.hist"):
                        # MOE.EXPERT.HIST — per-(layer, expert) usage histogram
                        # (optional SAVE/LOAD subcommand reads start+2/3 — must
                        # bound at cmd_end_tok so pipelined calls don't bleed)
                        _ = handle_moe_expert_hist(tokens, i, cmd_end_tok, writer, self.moe_tier)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "moe.expert.load"):
                        # MOE.EXPERT.LOAD — dynamic model registration
                        _ = handle_moe_expert_load(tokens, i, cmd_end_tok, writer, self.moe_tier)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "moe.expert.prune"):
                        # MOE.EXPERT.PRUNE — optional [on] flag at start+4, must
                        # bound at cmd_end_tok so a pipelined NEXT command's first
                        # token isn't mis-read as our flag.
                        _ = handle_moe_expert_prune(tokens, i, cmd_end_tok, writer, self.moe_tier)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "moe.expert.pin"):
                        # MOE.EXPERT.PIN
                        _ = handle_moe_expert_pin(tokens, i, num_tokens, writer, self.moe_tier)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "moe.expert.prefetch"):
                        # MOE.EXPERT.PREFETCH — variadic trailing expert IDs, must
                        # bound at cmd_end_tok (not num_tokens) when pipelined.
                        _ = handle_moe_expert_prefetch(tokens, i, cmd_end_tok, writer, self.moe_tier)
                        i = cmd_end_tok - 1
                    # ── KV.STORE ── (8 bytes: k=107,v=118,.,s,t,o,r,e)
                    elif tl == 8 and (tp[0]|0x20)==107 and (tp[1]|0x20)==118 and tp[2]==46 and (tp[3]|0x20)==115 and (tp[4]|0x20)==116 and (tp[5]|0x20)==111 and (tp[6]|0x20)==114 and (tp[7]|0x20)==101:
                        _ = handle_kv_store(tokens, i, cmd_end_tok, writer, self.kvcache, server, fd, kq)
                        i = cmd_end_tok - 1
                    # ── KV.FETCH ── (8 bytes: k=107,v=118,.,f,e,t,c,h)
                    elif tl == 8 and (tp[0]|0x20)==107 and (tp[1]|0x20)==118 and tp[2]==46 and (tp[3]|0x20)==102 and (tp[4]|0x20)==101 and (tp[5]|0x20)==116 and (tp[6]|0x20)==99 and (tp[7]|0x20)==104:
                        _ = handle_kv_fetch(tokens, i, cmd_end_tok, writer, self.kvcache, server, fd, kq)
                        i = cmd_end_tok - 1
                    # ── KV.INFO ── (7 bytes: k=107,v=118,.,i,n,f,o)
                    elif tl == 7 and (tp[0]|0x20)==107 and (tp[1]|0x20)==118 and tp[2]==46 and (tp[3]|0x20)==105 and (tp[4]|0x20)==110 and (tp[5]|0x20)==102 and (tp[6]|0x20)==111:
                        _ = handle_kv_info(writer, self.kvcache)
                        i = cmd_end_tok - 1
                    # ── ATTEND.CREATE ── (13 bytes)
                    elif cmd_eq(tp, tl, "attend.create"):
                        _ = handle_attend_create(tokens, i, cmd_end_tok, writer, self.attn_idx)
                        i = cmd_end_tok - 1  # skip all args; i += 1 below
                    # ── ATTEND.STORE ── (12 bytes)
                    elif cmd_eq(tp, tl, "attend.store"):
                        _ = handle_attend_store(tokens, i, cmd_end_tok, writer, self.attn_idx)
                        i = cmd_end_tok - 1
                    # ── ATTEND.QUERY ── (12 bytes)
                    elif cmd_eq(tp, tl, "attend.query"):
                        _ = handle_attend_query(tokens, i, cmd_end_tok, writer, self.attn_idx, server, fd, kq)
                        i = cmd_end_tok - 1
                    # ── ATTEND.FINALIZE ── (15 bytes: a,t,t,e,n,d,.,f,i,n,a,l,i,z,e)
                    elif cmd_eq(tp, tl, "attend.finalize"):
                        _ = handle_attend_finalize(tokens, i, cmd_end_tok, writer, self.attn_idx)
                        i = cmd_end_tok - 1
                    # ── ATTEND.INFO ── (11 bytes)
                    elif cmd_eq(tp, tl, "attend.info"):
                        _ = handle_attend_info(writer, self.attn_idx)
                        i = cmd_end_tok - 1
                    # ── ATTEND.QUERYBATCH removed 2026-05-01 — legacy single-shot path
                    # was only implemented via the (now deleted) MLX sidecar. Use
                    # ATTEND.PREFIX.STORE + ATTEND.PREFIX.QUERY instead — it's faster
                    # AND the same wire shape with K/V resident across queries.
                    # ── V.CREATE ── (8 bytes: v=118,.,c,r,e,a,t,e)
                    elif tl == 8 and (tp[0]|0x20)==118 and tp[1]==46 and (tp[2]|0x20)==99 and (tp[3]|0x20)==114 and (tp[4]|0x20)==101 and (tp[5]|0x20)==97 and (tp[6]|0x20)==116 and (tp[7]|0x20)==101:
                        _ = handle_v_create(tokens, i, num_tokens, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── V.STOREBATCH ── (12 bytes: v=118,.,s,t,o,r,e,b,a,t,c,h)
                    elif cmd_eq(tp, tl, "v.storebatch"):
                        # Bound by cmd_end_tok: the optional FMT F16 scan and the
                        # arity check must never read a pipelined neighbour.
                        _ = handle_v_storebatch(tokens, i, cmd_end_tok, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── V.FETCH ── (7 bytes: v=118,.,f,e,t,c,h)
                    elif tl == 7 and (tp[0]|0x20)==118 and tp[1]==46 and (tp[2]|0x20)==102 and (tp[3]|0x20)==101 and (tp[4]|0x20)==116 and (tp[5]|0x20)==99 and (tp[6]|0x20)==104:
                        # gh #193: pass cmd_end_tok as the handler bound — its
                        # optional-arg scan (FMT NATIVE) must never read into a
                        # pipelined neighbor (the gh #166/FT.* lesson).
                        _ = handle_v_fetch(tokens, i, cmd_end_tok, writer, self.v_store, server, fd, kq, self.ledger)
                        i = cmd_end_tok - 1
                    # ── V.INFO ── (6 bytes: v=118,.,i,n,f,o)
                    elif tl == 6 and (tp[0]|0x20)==118 and tp[1]==46 and (tp[2]|0x20)==105 and (tp[3]|0x20)==110 and (tp[4]|0x20)==102 and (tp[5]|0x20)==111:
                        _ = handle_v_info(tokens, i, num_tokens, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── V.SNAPSHOT ── (10 bytes: v=118,.,s,n,a,p,s,h,o,t)
                    elif tl == 10 and (tp[0]|0x20)==118 and tp[1]==46 and (tp[2]|0x20)==115 and (tp[3]|0x20)==110 and (tp[4]|0x20)==97 and (tp[5]|0x20)==112 and (tp[6]|0x20)==115 and (tp[7]|0x20)==104 and (tp[8]|0x20)==111 and (tp[9]|0x20)==116:
                        _ = handle_v_snapshot(tokens, i, num_tokens, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── V.RESTORE ── (9 bytes: v=118,.,r,e,s,t,o,r,e)
                    elif tl == 9 and (tp[0]|0x20)==118 and tp[1]==46 and (tp[2]|0x20)==114 and (tp[3]|0x20)==101 and (tp[4]|0x20)==115 and (tp[5]|0x20)==116 and (tp[6]|0x20)==111 and (tp[7]|0x20)==114 and (tp[8]|0x20)==101:
                        _ = handle_v_restore(tokens, i, num_tokens, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── V.COMMIT ── (8 bytes: v=118,.,c,o,m,m,i,t)
                    elif tl == 8 and (tp[0]|0x20)==118 and tp[1]==46 and (tp[2]|0x20)==99 and (tp[3]|0x20)==111 and (tp[4]|0x20)==109 and (tp[5]|0x20)==109 and (tp[6]|0x20)==105 and (tp[7]|0x20)==116:
                        _ = handle_v_commit(tokens, i, num_tokens, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── STATE.ALLOC ── (11 bytes: s,t,a,t,e,.,a,l,l,o,c)
                    elif tl == 11 and (tp[0]|0x20)==115 and (tp[1]|0x20)==116 and (tp[2]|0x20)==97 and (tp[3]|0x20)==116 and (tp[4]|0x20)==101 and tp[5]==46 and (tp[6]|0x20)==97 and (tp[7]|0x20)==108 and (tp[8]|0x20)==108 and (tp[9]|0x20)==111 and (tp[10]|0x20)==99:
                        _ = handle_state_alloc(tokens, i, num_tokens, writer, self.state_store)
                        i = cmd_end_tok - 1
                    # ── STATE.WRITE ── (11 bytes: s,t,a,t,e,.,w,r,i,t,e)
                    elif tl == 11 and (tp[0]|0x20)==115 and (tp[1]|0x20)==116 and (tp[2]|0x20)==97 and (tp[3]|0x20)==116 and (tp[4]|0x20)==101 and tp[5]==46 and (tp[6]|0x20)==119 and (tp[7]|0x20)==114 and (tp[8]|0x20)==105 and (tp[9]|0x20)==116 and (tp[10]|0x20)==101:
                        _ = handle_state_write(tokens, i, num_tokens, writer, self.state_store)
                        i = cmd_end_tok - 1
                    # ── STATE.READ ── (10 bytes: s,t,a,t,e,.,r,e,a,d)
                    elif tl == 10 and (tp[0]|0x20)==115 and (tp[1]|0x20)==116 and (tp[2]|0x20)==97 and (tp[3]|0x20)==116 and (tp[4]|0x20)==101 and tp[5]==46 and (tp[6]|0x20)==114 and (tp[7]|0x20)==101 and (tp[8]|0x20)==97 and (tp[9]|0x20)==100:
                        _ = handle_state_read(tokens, i, num_tokens, writer, self.state_store)
                        i = cmd_end_tok - 1
                    # ── STATE.FREE ── (10 bytes: s,t,a,t,e,.,f,r,e,e)
                    elif tl == 10 and (tp[0]|0x20)==115 and (tp[1]|0x20)==116 and (tp[2]|0x20)==97 and (tp[3]|0x20)==116 and (tp[4]|0x20)==101 and tp[5]==46 and (tp[6]|0x20)==102 and (tp[7]|0x20)==114 and (tp[8]|0x20)==101 and (tp[9]|0x20)==101:
                        _ = handle_state_free(tokens, i, num_tokens, writer, self.state_store)
                        i = cmd_end_tok - 1
                    # ── STATE.INFO ── (10 bytes: s,t,a,t,e,.,i,n,f,o)
                    elif tl == 10 and (tp[0]|0x20)==115 and (tp[1]|0x20)==116 and (tp[2]|0x20)==97 and (tp[3]|0x20)==116 and (tp[4]|0x20)==101 and tp[5]==46 and (tp[6]|0x20)==105 and (tp[7]|0x20)==110 and (tp[8]|0x20)==102 and (tp[9]|0x20)==111:
                        _ = handle_state_info(tokens, i, num_tokens, writer, self.state_store)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.REGISTER ── (18 bytes: k,v,.,p,r,e,f,i,x,.,r,e,g,i,s,t,e,r)
                    elif cmd_eq(tp, tl, "kv.prefix.register"):
                        _ = handle_kv_prefix_register(tokens, i, cmd_end_tok, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.LOOKUP ── (16 bytes: k,v,.,p,r,e,f,i,x,.,l,o,o,k,u,p)
                    elif cmd_eq(tp, tl, "kv.prefix.lookup"):
                        _ = handle_kv_prefix_lookup(tokens, i, cmd_end_tok, writer, self.v_store, self.ledger)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.INFO ── (14 bytes: k,v,.,p,r,e,f,i,x,.,i,n,f,o)
                    elif cmd_eq(tp, tl, "kv.prefix.info"):
                        _ = handle_kv_prefix_info(writer, self.v_store, self.metal_attn_engine)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.SAVE ── (14 bytes: k,v,.,p,r,e,f,i,x,.,s,a,v,e)
                    elif cmd_eq(tp, tl, "kv.prefix.save"):
                        _ = handle_kv_prefix_save(tokens, i, cmd_end_tok, self.worker_id, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.COMMIT ── durability barrier over both WALs
                    elif cmd_eq(tp, tl, "kv.prefix.commit"):
                        _ = handle_kv_prefix_commit(writer, self.v_store, self.dispatcher.wal, self.wal_writes_off)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.DROP ── evict prefixes by name, WAL-logged
                    elif cmd_eq(tp, tl, "kv.prefix.drop"):
                        _ = handle_kv_prefix_drop(tokens, i, cmd_end_tok, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.OWNER ── (15 bytes: k,v,.,p,r,e,f,i,x,.,o,w,n,e,r)
                    elif cmd_eq(tp, tl, "kv.prefix.owner"):
                        _ = handle_kv_prefix_owner(tokens, i, cmd_end_tok, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.WARM ── gh #67 (14 bytes: k,v,.,p,r,e,f,i,x,.,w,a,r,m)
                    elif cmd_eq(tp, tl, "kv.prefix.warm"):
                        _ = handle_kv_prefix_warm(tokens, i, num_tokens, writer, self.v_store, self.metal_attn_engine)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.BLOCKS ── gh #71 (16 bytes: k,v,.,p,r,e,f,i,x,.,b,l,o,c,k,s)
                    # Shares tl=16 with KV.PREFIX.LOOKUP — disambiguated on tp[10] (b=98 vs l=108).
                    elif cmd_eq(tp, tl, "kv.prefix.blocks"):
                        _ = handle_kv_prefix_blocks(tokens, i, num_tokens, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── KV.PREFIX.MEMBERSHIP ── gh #71 (20 bytes: k,v,.,p,r,e,f,i,x,.,m,e,m,b,e,r,s,h,i,p)
                    elif cmd_eq(tp, tl, "kv.prefix.membership"):
                        _ = handle_kv_prefix_membership(tokens, i, num_tokens, writer, self.v_store)
                        i = cmd_end_tok - 1
                    # ── ATTEND.PREFIX.STORE ── (19 bytes: a,t,t,e,n,d,.,p,r,e,f,i,x,.,s,t,o,r,e)
                    elif cmd_eq(tp, tl, "attend.prefix.store"):
                        _ = handle_attend_prefix_store(tokens, i, num_tokens, writer, self.metal_attn_engine, self.cuda_attn_engine)
                        i = cmd_end_tok - 1
                    # ── ATTEND.PREFIX.QUERY ── (19 bytes: a,t,t,e,n,d,.,p,r,e,f,i,x,.,q,u,e,r,y)
                    elif cmd_eq(tp, tl, "attend.prefix.query"):
                        _ = handle_attend_prefix_query(tokens, i, num_tokens, writer, self.metal_attn_engine, self.cuda_attn_engine)
                        i = cmd_end_tok - 1
                    # ── ATTEND.PREFIX.LOOKUP ── gh #65 follow-on (20 bytes: a,t,t,e,n,d,.,p,r,e,f,i,x,.,l,o,o,k,u,p)
                    elif cmd_eq(tp, tl, "attend.prefix.lookup"):
                        _ = handle_attend_prefix_lookup(tokens, i, num_tokens, writer, self.metal_attn_engine)
                        i = cmd_end_tok - 1
                    # ── ATTEND.PREFIX.QUERY_FUSED ── gh #49 (25 bytes: a,t,t,e,n,d,.,p,r,e,f,i,x,.,q,u,e,r,y,_,f,u,s,e,d)
                    elif cmd_eq(tp, tl, "attend.prefix.query_fused"):
                        _ = handle_attend_prefix_query_fused(tokens, i, num_tokens, writer, self.metal_attn_engine)
                        i = cmd_end_tok - 1
                    # ── ATTEND.PREFIX.QUERY_SPARSE ── gh #60 Phase 2 (26 bytes: a,t,t,e,n,d,.,p,r,e,f,i,x,.,q,u,e,r,y,_,s,p,a,r,s,e)
                    elif cmd_eq(tp, tl, "attend.prefix.query_sparse"):
                        _ = handle_attend_prefix_query_sparse(tokens, i, num_tokens, writer, self.metal_attn_engine)
                        i = cmd_end_tok - 1
                    # ── ATTEND.PREFIX.QUERY_SPARSE_AUTO ── gh #63 Phase 3b (31 bytes: ...query_sparse_auto)
                    elif cmd_eq(tp, tl, "attend.prefix.query_sparse_auto"):
                        _ = handle_attend_prefix_query_sparse_auto(tokens, i, num_tokens, writer, self.metal_attn_engine, self.cuda_attn_engine)
                        i = cmd_end_tok - 1
                    # ── ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED ── gh #63 follow-on (37 bytes: ...query_sparse_auto_fused)
                    elif cmd_eq(tp, tl, "attend.prefix.query_sparse_auto_fused"):
                        _ = handle_attend_prefix_query_sparse_auto_fused(tokens, i, num_tokens, writer, self.metal_attn_engine)
                        i = cmd_end_tok - 1
                    # ── SSM.PREFIX.STORE ── gh #65 (16 bytes: s,s,m,.,p,r,e,f,i,x,.,s,t,o,r,e)
                    elif cmd_eq(tp, tl, "ssm.prefix.store"):
                        _ = handle_ssm_prefix_store(tokens, i, num_tokens, writer, self.worker_id)
                        i = cmd_end_tok - 1
                    # ── SSM.PREFIX.FETCH ── gh #65 (16 bytes: s,s,m,.,p,r,e,f,i,x,.,f,e,t,c,h)
                    elif cmd_eq(tp, tl, "ssm.prefix.fetch"):
                        _ = handle_ssm_prefix_fetch(tokens, i, num_tokens, writer, self.worker_id, fd)
                        i = cmd_end_tok - 1
                    # ── SSM.PREFIX.DROP ── gh #65 (15 bytes: s,s,m,.,p,r,e,f,i,x,.,d,r,o,p)
                    elif cmd_eq(tp, tl, "ssm.prefix.drop"):
                        _ = handle_ssm_prefix_drop(tokens, i, num_tokens, writer, self.worker_id)
                        i = cmd_end_tok - 1
                    # ── AI.ROUTE.REGISTER ── (17 bytes)
                    elif cmd_eq(tp, tl, "ai.route.register"):
                        _ = handle_ai_route_register(tokens, i, num_tokens, writer, self.router)
                        i = cmd_end_tok - 1
                    # ── AI.ROUTE.UPDATE ── (15 bytes)
                    elif cmd_eq(tp, tl, "ai.route.update"):
                        _ = handle_ai_route_update(tokens, i, num_tokens, writer, self.router)
                        i = cmd_end_tok - 1
                    # ── AI.ROUTE.REMOVE ── (15 bytes)
                    elif cmd_eq(tp, tl, "ai.route.remove"):
                        _ = handle_ai_route_remove(tokens, i, num_tokens, writer, self.router)
                        i = cmd_end_tok - 1
                    # ── AI.ROUTE.INFO ── (13 bytes)
                    elif cmd_eq(tp, tl, "ai.route.info"):
                        _ = handle_ai_route_info(writer, self.router)
                        i = cmd_end_tok - 1
                    # ── AI.ROUTE ── (8 bytes) — must be after longer AI.ROUTE.* commands
                    elif tl == 8 and (tp[0]|0x20)==97 and (tp[1]|0x20)==105 and tp[2]==46 and (tp[3]|0x20)==114 and (tp[4]|0x20)==111 and (tp[5]|0x20)==117 and (tp[6]|0x20)==116 and (tp[7]|0x20)==101:
                        _ = handle_ai_route(tokens, i, num_tokens, writer, self.router, server, fd, kq)
                        i = cmd_end_tok - 1
                    # ── RAG.SPECULATE.ENABLE ── (20 bytes)
                    elif cmd_eq(tp, tl, "rag.speculate.enable"):
                        _ = handle_rag_speculate_enable(tokens, i, num_tokens, writer, self.spec_rag)
                        i = cmd_end_tok - 1
                    # ── RAG.SPECULATE.INFO ── (18 bytes)
                    elif cmd_eq(tp, tl, "rag.speculate.info"):
                        _ = handle_rag_speculate_info(tokens, i, num_tokens, writer, self.spec_rag)
                        i = cmd_end_tok - 1
                    # ── RAG.QUERY ── (9 bytes)
                    elif cmd_eq(tp, tl, "rag.query"):
                        _ = handle_rag_query(tokens, i, num_tokens, writer, self.spec_rag, hnsw, server, fd, kq)
                        i = cmd_end_tok - 1
                    # ── INCRBY ──
                    elif tl == 6 and cmd_matches_6(tp, 105, 110, 99, 114, 98, 121):
                        _ = handle_incrby(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── DECRBY ──
                    elif tl == 6 and cmd_matches_6(tp, 100, 101, 99, 114, 98, 121):
                        _ = handle_decrby(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── INCRBYFLOAT ──
                    elif cmd_eq(tp, tl, "incrbyfloat"):
                        _ = handle_incrbyfloat(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── APPEND ──
                    elif tl == 6 and cmd_matches_6(tp, 97, 112, 112, 101, 110, 100):
                        _ = handle_append(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── STRLEN ──
                    elif tl == 6 and cmd_matches_6(tp, 115, 116, 114, 108, 101, 110):
                        _ = handle_strlen(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── GETSET ──
                    elif tl == 6 and cmd_matches_6(tp, 103, 101, 116, 115, 101, 116):
                        _ = handle_getset(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher, self.ttl_map, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── GETDEL ──
                    elif tl == 6 and cmd_matches_6(tp, 103, 101, 116, 100, 101, 108):
                        _ = handle_getdel(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── GETEX ──
                    elif tl == 5 and cmd_matches_5(tp, 103, 101, 116, 101, 120):
                        _ = handle_getex(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── SETNX ──
                    elif tl == 5 and cmd_matches_5(tp, 115, 101, 116, 110, 120):
                        _ = handle_setnx(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher)
                        i = cmd_end_tok - 1
                    # ── SETEX ──
                    elif tl == 5 and cmd_matches_5(tp, 115, 101, 116, 101, 120):
                        _ = handle_setex(tokens, i, cmd_end_tok, writer, self.dispatcher, self.ttl_map)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── PSETEX ──
                    elif tl == 6 and cmd_matches_6(tp, 112, 115, 101, 116, 101, 120):
                        _ = handle_psetex(tokens, i, cmd_end_tok, writer, self.dispatcher, self.ttl_map)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── MSETNX ──
                    elif tl == 6 and cmd_matches_6(tp, 109, 115, 101, 116, 110, 120):
                        _ = handle_msetnx(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher)
                        i = cmd_end_tok - 1
                    # ── MSETEX (R3) ── (6 bytes: m=109, s=115, e=101, t=116, e=101, x=120)
                    elif tl == 6 and cmd_matches_6(tp, 109, 115, 101, 116, 101, 120):
                        _ = handle_msetex(tokens, i, cmd_end_tok, writer, self.dispatcher, self.ttl_map)
                        if is_not_null(self.dispatcher.wal):   # MSETEX's TTLs were never logged
                            for _mk in range(i + 2, cmd_end_tok, 2):
                                self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=_mk].ptr, tokens[unsafe_offset=_mk].length)
                        i = cmd_end_tok - 1
                    # ── GETRANGE ──
                    elif tl == 8 and cmd_matches_8(tp, 103, 101, 116, 114, 97, 110, 103, 101):
                        _ = handle_getrange(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SUBSTR ──
                    elif tl == 6 and cmd_matches_6(tp, 115, 117, 98, 115, 116, 114):
                        _ = handle_substr(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SETRANGE ──
                    elif tl == 8 and cmd_matches_8(tp, 115, 101, 116, 114, 97, 110, 103, 101):
                        _ = handle_setrange(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── EXPIRETIME ──
                    elif cmd_eq(tp, tl, "expiretime"):
                        _ = handle_expiretime(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── PEXPIRETIME ──
                    elif cmd_eq(tp, tl, "pexpiretime"):
                        _ = handle_pexpiretime(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── UNLINK ──
                    elif tl == 6 and cmd_matches_6(tp, 117, 110, 108, 105, 110, 107):
                        _ = handle_unlink(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── TYPE ──
                    elif tl == 4 and cmd_matches_4(tp, 116, 121, 112, 101):
                        _ = handle_type(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── RENAME ──
                    elif tl == 6 and cmd_matches_6(tp, 114, 101, 110, 97, 109, 101):
                        var _vmv = hash_addr(self.keyspace, tokens[i + 1].ptr, tokens[i + 1].length) if i + 2 < cmd_end_tok else 0
                        _ = handle_rename(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        if _vmv != 0:   # #46: a renamed hash's slot names its old key
                            after_rename(self.shared_hnsw, self.keyspace, self.dispatcher.wal, self.vec_tomb, _vmv,
                                         tokens[i + 1].ptr, tokens[i + 1].length, tokens[i + 2].ptr, tokens[i + 2].length)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        if is_not_null(self.dispatcher.wal) and i + 2 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 2].ptr, tokens[unsafe_offset=i + 2].length)
                        i = cmd_end_tok - 1
                    # ── RENAMENX ──
                    elif tl == 8 and cmd_matches_8(tp, 114, 101, 110, 97, 109, 101, 110, 120):
                        var _vmv = hash_addr(self.keyspace, tokens[i + 1].ptr, tokens[i + 1].length) if i + 2 < cmd_end_tok else 0
                        _ = handle_renamenx(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        if _vmv != 0:   # #46: a renamed hash's slot names its old key
                            after_rename(self.shared_hnsw, self.keyspace, self.dispatcher.wal, self.vec_tomb, _vmv,
                                         tokens[i + 1].ptr, tokens[i + 1].length, tokens[i + 2].ptr, tokens[i + 2].length)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        if is_not_null(self.dispatcher.wal) and i + 2 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 2].ptr, tokens[unsafe_offset=i + 2].length)
                        i = cmd_end_tok - 1
                    # ── COPY ──
                    elif tl == 4 and cmd_matches_4(tp, 99, 111, 112, 121):
                        _ = handle_copy(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        if i + 2 < cmd_end_tok:   # #46: the copy's vector, as HSET's
                            _ = ingest_whole_hash(self.shared_hnsw, self.keyspace, self.dispatcher.wal, self.vec_tomb,
                                                  tokens[i + 2].ptr, tokens[i + 2].length)
                        if is_not_null(self.dispatcher.wal) and i + 2 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 2].ptr, tokens[unsafe_offset=i + 2].length)
                        i = cmd_end_tok - 1
                    # ── OBJECT ──
                    elif tl == 6 and cmd_matches_6(tp, 111, 98, 106, 101, 99, 116):
                        _ = handle_object(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── SORT ──
                    elif tl == 4 and cmd_matches_4(tp, 115, 111, 114, 116):
                        _ = handle_sort(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        if is_not_null(self.dispatcher.wal):   # SORT ... STORE dest: log dest's image
                            for _so in range(i + 2, cmd_end_tok - 1):
                                if arg_eq(tokens[unsafe_offset=_so].ptr, tokens[unsafe_offset=_so].length, "store"):
                                    self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=_so + 1].ptr, tokens[unsafe_offset=_so + 1].length)
                        i = cmd_end_tok - 1
                    # ── SORT_RO ──
                    elif cmd_eq(tp, tl, "sort_ro"):
                        _ = handle_sort_ro(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── SCAN ──
                    elif tl == 4 and cmd_matches_4(tp, 115, 99, 97, 110):
                        _ = handle_scan(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.tenant_ns_buf, self.cur_tenant_ns_len)
                        i = cmd_end_tok - 1
                    # ── KEYS ──
                    elif tl == 4 and cmd_matches_4(tp, 107, 101, 121, 115):
                        _ = handle_keys(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.tenant_ns_buf, self.cur_tenant_ns_len)
                        i = cmd_end_tok - 1
                    # ── RANDOMKEY ──
                    elif cmd_eq(tp, tl, "randomkey"):
                        _ = handle_randomkey(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── TOUCH ──
                    elif tl == 5 and cmd_matches_5(tp, 116, 111, 117, 99, 104):
                        _ = handle_touch(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── WAIT ──
                    elif tl == 4 and cmd_matches_4(tp, 119, 97, 105, 116):
                        # gh #390: park only on the primary frame (not inside an
                        # EXEC replay) and when this command's byte end is known.
                        var can_park = self.can_park_wait and on_primary and cmd_idx < num_cmds and fd >= 0
                        if handle_wait(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map,
                                       self.cluster, fd, self.parked_waits, can_park):
                            # Parked: no reply yet, and nothing pipelined behind
                            # WAIT may run before it. Stop at WAIT's byte end;
                            # the rest stays in the fd's buffer until the engine
                            # answers the WAIT and resumes it (as EXEC does).
                            primary_consumed = cmd_byte_ends[cmd_idx]
                            i = num_tokens
                        else:
                            i = cmd_end_tok - 1
                    # ── WAITAOF ──
                    elif cmd_eq(tp, tl, "waitaof"):
                        _ = handle_waitaof(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── HMGET ──
                    elif tl == 5 and cmd_matches_5(tp, 104, 109, 103, 101, 116):
                        _ = handle_hmget(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── HGETALL ──
                    elif tl == 7 and cmd_matches_7(tp, 104, 103, 101, 116, 97, 108, 108):
                        _ = handle_hgetall(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── HKEYS ──
                    elif tl == 5 and cmd_matches_5(tp, 104, 107, 101, 121, 115):
                        _ = handle_hkeys(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── HVALS ──
                    elif tl == 5 and cmd_matches_5(tp, 104, 118, 97, 108, 115):
                        _ = handle_hvals(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── HLEN ──
                    elif tl == 4 and cmd_matches_4(tp, 104, 108, 101, 110):
                        _ = handle_hlen(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── HDEL ──
                    elif tl == 4 and cmd_matches_4(tp, 104, 100, 101, 108):
                        _ = handle_hdel(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── HEXISTS ──
                    elif tl == 7 and cmd_matches_7(tp, 104, 101, 120, 105, 115, 116, 115):
                        _ = handle_hexists(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── HINCRBY ──
                    elif tl == 7 and cmd_matches_7(tp, 104, 105, 110, 99, 114, 98, 121):
                        _ = handle_hincrby(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher)
                        i = cmd_end_tok - 1
                    # ── HINCRBYFLOAT ──
                    elif cmd_eq(tp, tl, "hincrbyfloat"):
                        # gh #180: cmd_end_tok bound + authoritative skip — the
                        # old num_tokens bound let a short HINCRBYFLOAT satisfy
                        # its arity check with the NEXT pipelined command's
                        # tokens (gh #156 shape), now with a WAL record attached.
                        _ = handle_hincrbyfloat(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher)
                        i = cmd_end_tok - 1
                    # ── HRANDFIELD ──
                    elif cmd_eq(tp, tl, "hrandfield"):
                        _ = handle_hrandfield(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── HSCAN ──
                    elif tl == 5 and cmd_matches_5(tp, 104, 115, 99, 97, 110):
                        _ = handle_hscan(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── HSETNX ──
                    elif tl == 6 and cmd_matches_6(tp, 104, 115, 101, 116, 110, 120):
                        _ = handle_hsetnx(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher, self.shared_hnsw,
                                          self.vec_tomb)
                        i = cmd_end_tok - 1
                    # ── R3: HEXPIRE (7 bytes: h=104,e=101,x=120,p=112,i=105,r=114,e=101) ──
                    elif tl == 7 and (tp[0]|0x20)==104 and (tp[1]|0x20)==101 and (tp[2]|0x20)==120 and (tp[3]|0x20)==112 and (tp[4]|0x20)==105 and (tp[5]|0x20)==114 and (tp[6]|0x20)==101:
                        _ = handle_hexpire(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── R3: HPEXPIRE (8 bytes: h,p,e,x,p,i,r,e) ──
                    elif tl == 8 and (tp[0]|0x20)==104 and (tp[1]|0x20)==112 and (tp[2]|0x20)==101 and (tp[3]|0x20)==120 and (tp[4]|0x20)==112 and (tp[5]|0x20)==105 and (tp[6]|0x20)==114 and (tp[7]|0x20)==101:
                        _ = handle_hpexpire(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── R3: HEXPIREAT (9 bytes: h,e,x,p,i,r,e,a,t) ──
                    elif tl == 9 and (tp[0]|0x20)==104 and (tp[1]|0x20)==101 and (tp[2]|0x20)==120 and (tp[3]|0x20)==112 and (tp[4]|0x20)==105 and (tp[5]|0x20)==114 and (tp[6]|0x20)==101 and (tp[7]|0x20)==97 and (tp[8]|0x20)==116:
                        _ = handle_hexpireat(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── R3: HPEXPIREAT (10 bytes: h,p,e,x,p,i,r,e,a,t) ──
                    elif tl == 10 and (tp[0]|0x20)==104 and (tp[1]|0x20)==112 and (tp[2]|0x20)==101 and (tp[3]|0x20)==120 and (tp[4]|0x20)==112 and (tp[5]|0x20)==105 and (tp[6]|0x20)==114 and (tp[7]|0x20)==101 and (tp[8]|0x20)==97 and (tp[9]|0x20)==116:
                        _ = handle_hpexpireat(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── R3: HTTL (4 bytes: h=104,t=116,t=116,l=108) ──
                    elif tl == 4 and (tp[0]|0x20)==104 and (tp[1]|0x20)==116 and (tp[2]|0x20)==116 and (tp[3]|0x20)==108:
                        _ = handle_httl(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── R3: HPTTL (5 bytes: h=104,p=112,t=116,t=116,l=108) ──
                    elif tl == 5 and (tp[0]|0x20)==104 and (tp[1]|0x20)==112 and (tp[2]|0x20)==116 and (tp[3]|0x20)==116 and (tp[4]|0x20)==108:
                        _ = handle_hpttl(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── R3: HPERSIST (8 bytes: h=104,p=112,e=101,r=114,s=115,i=105,s=115,t=116) ──
                    elif tl == 8 and (tp[0]|0x20)==104 and (tp[1]|0x20)==112 and (tp[2]|0x20)==101 and (tp[3]|0x20)==114 and (tp[4]|0x20)==115 and (tp[5]|0x20)==105 and (tp[6]|0x20)==115 and (tp[7]|0x20)==116:
                        _ = handle_hpersist(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── R3: HEXPIRETIME (11 bytes: h,e,x,p,i,r,e,t,i,m,e) ──
                    elif tl == 11 and (tp[0]|0x20)==104 and (tp[1]|0x20)==101 and (tp[2]|0x20)==120 and (tp[3]|0x20)==112 and (tp[4]|0x20)==105 and (tp[5]|0x20)==114 and (tp[6]|0x20)==101 and (tp[7]|0x20)==116 and (tp[8]|0x20)==105 and (tp[9]|0x20)==109 and (tp[10]|0x20)==101:
                        _ = handle_hexpiretime(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── R3: HPEXPIRETIME (12 bytes: h,p,e,x,p,i,r,e,t,i,m,e) ──
                    elif tl == 12 and (tp[0]|0x20)==104 and (tp[1]|0x20)==112 and (tp[2]|0x20)==101 and (tp[3]|0x20)==120 and (tp[4]|0x20)==112 and (tp[5]|0x20)==105 and (tp[6]|0x20)==114 and (tp[7]|0x20)==101 and (tp[8]|0x20)==116 and (tp[9]|0x20)==105 and (tp[10]|0x20)==109 and (tp[11]|0x20)==101:
                        _ = handle_hpexpiretime(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── LINDEX ──
                    elif tl == 6 and cmd_matches_6(tp, 108, 105, 110, 100, 101, 120):
                        _ = handle_lindex(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── LSET ──
                    elif tl == 4 and cmd_matches_4(tp, 108, 115, 101, 116):
                        _ = handle_lset(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── LINSERT ──
                    elif tl == 7 and cmd_matches_7(tp, 108, 105, 110, 115, 101, 114, 116):
                        _ = handle_linsert(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── LREM ──
                    elif tl == 4 and cmd_matches_4(tp, 108, 114, 101, 109):
                        _ = handle_lrem(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── LTRIM ──
                    elif tl == 5 and cmd_matches_5(tp, 108, 116, 114, 105, 109):
                        _ = handle_ltrim(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── LPOS ──
                    elif tl == 4 and cmd_matches_4(tp, 108, 112, 111, 115):
                        _ = handle_lpos(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── LMOVE ──
                    elif tl == 5 and cmd_matches_5(tp, 108, 109, 111, 118, 101):
                        _ = handle_lmove(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher)
                        i = cmd_end_tok - 1
                    # ── RPOPLPUSH (≡ LMOVE src dst RIGHT LEFT; gh #101 slow-path coverage) ──
                    elif cmd_eq(tp, tl, "rpoplpush"):
                        if i + 2 < cmd_end_tok:
                            self._rpoplpush(tokens, i, writer)
                            i += 2
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'rpoplpush' command")
                            i = cmd_end_tok - 1
                    # ── LMPOP (stub — same pattern as old code) ──
                    elif cmd_eq(tp, tl, "lmpop"):
                        # LMPOP numkeys key [key ...] LEFT|RIGHT [COUNT count]
                        if cmd_end_tok - i < 4:
                            writer.append_error_response("ERR wrong number of arguments for 'lmpop' command")
                        else:
                            var _mp = parse_mpop(tokens, i + 1, cmd_end_tok, False)
                            if _mp.error.byte_length() > 0:
                                writer.append_error_response(_mp.error)
                            elif not self._mpop_lists(tokens, i + 2, _mp.numkeys, _mp.first, _mp.count, writer):
                                writer.append_null_array_response()
                        i = cmd_end_tok - 1
                    # ── ZMPOP (gh #251) ──
                    elif cmd_eq(tp, tl, "zmpop"):
                        _ = handle_zmpop(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── SCARD ──
                    elif tl == 5 and cmd_matches_5(tp, 115, 99, 97, 114, 100):
                        _ = handle_scard(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SISMEMBER ──
                    elif cmd_eq(tp, tl, "sismember"):
                        _ = handle_sismember(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SMISMEMBER ──
                    elif cmd_eq(tp, tl, "smismember"):
                        _ = handle_smismember(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SMEMBERS ──
                    elif tl == 8 and cmd_matches_8(tp, 115, 109, 101, 109, 98, 101, 114, 115):
                        _ = handle_smembers(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SRANDMEMBER ──
                    elif cmd_eq(tp, tl, "srandmember"):
                        _ = handle_srandmember(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SREM ──
                    elif tl == 4 and cmd_matches_4(tp, 115, 114, 101, 109):
                        _ = handle_srem(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── SMOVE ──
                    elif tl == 5 and cmd_matches_5(tp, 115, 109, 111, 118, 101):
                        _ = handle_smove(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── SINTER ──
                    elif tl == 6 and cmd_matches_6(tp, 115, 105, 110, 116, 101, 114):
                        _ = handle_sinter(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SINTERSTORE ──
                    elif cmd_eq(tp, tl, "sinterstore"):
                        _ = handle_sinterstore(tokens, i, cmd_end_tok, writer, self.keyspace)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── SINTERCARD ──
                    elif cmd_eq(tp, tl, "sintercard"):
                        _ = handle_sintercard(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SUNION ──
                    elif tl == 6 and cmd_matches_6(tp, 115, 117, 110, 105, 111, 110):
                        _ = handle_sunion(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SUNIONSTORE ──
                    elif cmd_eq(tp, tl, "sunionstore"):
                        _ = handle_sunionstore(tokens, i, cmd_end_tok, writer, self.keyspace)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── SDIFF ──
                    elif tl == 5 and cmd_matches_5(tp, 115, 100, 105, 102, 102):
                        _ = handle_sdiff(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── SDIFFSTORE ──
                    elif cmd_eq(tp, tl, "sdiffstore"):
                        _ = handle_sdiffstore(tokens, i, cmd_end_tok, writer, self.keyspace)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── SSCAN ──
                    elif tl == 5 and cmd_matches_5(tp, 115, 115, 99, 97, 110):
                        _ = handle_sscan(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZREM ──
                    elif tl == 4 and cmd_matches_4(tp, 122, 114, 101, 109):
                        _ = handle_zrem(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── ZCARD ──
                    elif cmd_eq(tp, tl, "zcard"):
                        _ = handle_zcard(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZRANK ──
                    elif cmd_eq(tp, tl, "zrank"):
                        _ = handle_zrank(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZDIFF (5 bytes, z,d) ──
                    elif cmd_eq(tp, tl, "zdiff"):
                        _ = handle_zdiff(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZSCAN (5 bytes, z,s; gh #101 — was mis-wired to handle_zscore,
                    #    while the 6-byte z,s slot below sent ZSCORE to handle_zscan) ──
                    elif cmd_eq(tp, tl, "zscan"):
                        _ = handle_zscan(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── BITOP ──
                    elif tl == 5 and cmd_matches_5(tp, 98, 105, 116, 111, 112):
                        _ = handle_bitop(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map)
                        if is_not_null(self.dispatcher.wal) and i + 2 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 2].ptr, tokens[unsafe_offset=i + 2].length)
                        i = cmd_end_tok - 1
                    # ── ZRANGE (6 bytes, z,r) ──
                    elif cmd_eq(tp, tl, "zrange"):
                        _ = handle_zrange(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZSCORE (6 bytes, z,s) ──
                    elif cmd_eq(tp, tl, "zscore"):
                        _ = handle_zscore(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZCOUNT (6 bytes, z,c) ──
                    elif cmd_eq(tp, tl, "zcount"):
                        _ = handle_zcount(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZUNION (6 bytes, z,u) ──
                    elif cmd_eq(tp, tl, "zunion"):
                        _ = handle_zunion(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZINTER (6 bytes, z,i) ──
                    elif cmd_eq(tp, tl, "zinter"):
                        _ = handle_zinter(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── BITPOS (6 bytes, b,i) ──
                    elif cmd_eq(tp, tl, "bitpos"):
                        _ = handle_bitpos(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZPOPMIN (7 bytes, z,p,...,i; gh #101 — pre-fix this fell through
                    #    to the z,p ZPOPMAX match below and popped the WRONG end) ──
                    elif cmd_eq(tp, tl, "zpopmin"):
                        if i + 1 < cmd_end_tok:
                            var zpmin_key = tokens[i+1].value()
                            # gh #238: the optional COUNT was neither honoured nor
                            # consumed. `ZPOPMIN key 2` popped one pair and left
                            # the "2" to be dispatched as its own command — one
                            # command in, TWO replies out, desyncing the client
                            # (the gh #214/#218/#223 class, still reachable here
                            # because those sweeps probe 0- and 1-argument forms
                            # and this optional arg is the SECOND).
                            var zpmin_n = 1
                            var zpmin_has_cnt = i + 2 < cmd_end_tok
                            if zpmin_has_cnt:
                                zpmin_n = Int(strict_atol(tokens[i+2].value()))
                            var zpmin_val = self.keyspace[].get(zpmin_key)
                            # gh #393: Redis reads COUNT as a non-negative long,
                            # before the key; a negative one answered `*0`.
                            if i + 3 < cmd_end_tok:
                                writer.append_error_response("ERR syntax error")
                            elif zpmin_n < 0:
                                writer.append_error_response("ERR value is out of range, must be positive")
                            elif zpmin_val.is_none():
                                writer.append_empty_array_response()
                            elif zpmin_val.type.value == ValueType.ZSET:
                                var zpmin_ptr = zpmin_val.as_zset().bitcast[SlabSkipList]()
                                var zpmin_avail = zpmin_ptr[].length
                                var zpmin_out = zpmin_n if zpmin_n < zpmin_avail else zpmin_avail
                                if zpmin_out <= 0:
                                    writer.append_empty_array_response()
                                else:
                                    # RESP3, as Redis: [member, score] without a
                                    # count, a list of such pairs with one.
                                    if zpmin_has_cnt:
                                        writer.append_scored_header(zpmin_out, True)
                                    else:
                                        writer.append_array_header(2)
                                    # heap, not stack_allocation: gv_bytes writes it
                                    # out of line (the gh #349 tail-call hazard).
                                    var _zpm_wb = alloc[UInt8](64)
                                    for _ in range(zpmin_out):
                                        var zpmin_res = zpmin_ptr[].pop_min()
                                        if not zpmin_res.valid or zpmin_res.obj.is_none(): break
                                        # gh #251: was Int64(...), which truncated
                                        # a fractional score (1.5 -> "1").
                                        # #18: nor through Int64() (±inf).
                                        if zpmin_has_cnt:
                                            writer.append_scored_member(zpmin_res.obj, zpmin_res.score, True)
                                        else:
                                            writer.append_bulk_value_response(zpmin_res.obj)
                                            writer.append_score_response(zpmin_res.score)
                                        # The fast path logs the resolved effect (ZREM of
                                        # the popped member, cmd 12); this path logged
                                        # nothing, so a replay resurrected every member
                                        # a slow-path ZPOPMIN had popped.
                                        var _zpm_wl = 0
                                        var _zpm_wp = gv_bytes(zpmin_res.obj, _zpm_wb, _zpm_wl)
                                        _ = self.dispatcher.wal[].append_kv(12, tokens[i+1].ptr, tokens[i+1].length, _zpm_wp, _zpm_wl)
                                        # gh #394: the popped member is ours; the reply copied it.
                                        zpmin_res.obj.free_str_payload()
                                    _zpm_wb.unsafe_free()
                                    # gh #234: an emptied zset is removed.
                                    if zpmin_ptr[].length == 0:
                                        _ = remove_and_free(self.keyspace, GenericValue.borrow(tokens[i+1].ptr, tokens[i+1].length))
                                        _ = self.dispatcher.wal[].append(2, tokens[i+1].ptr, tokens[i+1].length)
                            else:
                                writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            i = cmd_end_tok - 1
                        else:
                            writer.append_error_response("ERR wrong number of arguments for 'zpopmin' command")
                            i = cmd_end_tok - 1
                    # ── ZPOPMAX (7 bytes, z,p) ──
                    elif cmd_eq(tp, tl, "zpopmax"):
                        _ = handle_zpopmax(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── ZINCRBY (7 bytes, z,i) ──
                    elif cmd_eq(tp, tl, "zincrby"):
                        _ = handle_zincrby(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── ZMSCORE (7 bytes, z,m) ──
                    elif cmd_eq(tp, tl, "zmscore"):
                        _ = handle_zmscore(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── PFMERGE (7 bytes, p,f,m) ──
                    elif cmd_eq(tp, tl, "pfmerge"):
                        _ = handle_pfmerge(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── ZREVRANK (8 bytes) ──
                    elif tl == 8 and cmd_matches_8(tp, 122, 114, 101, 118, 114, 97, 110, 107):
                        _ = handle_zrevrank(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── BITFIELD (8 bytes, b,_,_,f) ──
                    elif cmd_eq(tp, tl, "bitfield"):
                        _ = handle_bitfield(tokens, i, cmd_end_tok, writer, self.keyspace)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── ZREVRANGE (9 bytes, z,r) ──
                    elif cmd_eq(tp, tl, "zrevrange"):
                        _ = handle_zrevrange(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZLEXCOUNT (9 bytes, z,l) ──
                    elif cmd_eq(tp, tl, "zlexcount"):
                        _ = handle_zlexcount(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZINTERCARD (10 bytes, z,i) ──
                    elif cmd_eq(tp, tl, "zintercard"):
                        _ = handle_zintercard(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZDIFFSTORE (10 bytes, z,d) ──
                    elif cmd_eq(tp, tl, "zdiffstore"):
                        _ = handle_zdiffstore(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── ZUNIONSTORE (11 bytes, z,u) ──
                    elif cmd_eq(tp, tl, "zunionstore"):
                        _ = handle_zunionstore(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── ZINTERSTORE (11 bytes, z,i) ──
                    elif cmd_eq(tp, tl, "zinterstore"):
                        _ = handle_zinterstore(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── ZRANDMEMBER (11 bytes, z,r,_,_,d) ──
                    elif cmd_eq(tp, tl, "zrandmember"):
                        _ = handle_zrandmember(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZRANGESTORE (11 bytes, z,r,_,_,g,_,s; gh #101 — was mis-wired to
                    #    handle_zrangebyscore, while the 13-byte slot below sent
                    #    ZRANGEBYSCORE to handle_zrangestore) ──
                    elif cmd_eq(tp, tl, "zrangestore"):
                        _ = handle_zrangestore(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    # ── ZRANGEBYLEX (11 bytes, z,r, tp[4]==103, tp[6]==98) ──
                    elif cmd_eq(tp, tl, "zrangebylex"):
                        _ = handle_zrangebylex(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── BITFIELD_RO (11 bytes, b,_,_,f) ──
                    elif cmd_eq(tp, tl, "bitfield_ro"):
                        _ = handle_bitfield_ro(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZRANGEBYSCORE (13 bytes, z) ──
                    elif cmd_eq(tp, tl, "zrangebyscore"):
                        _ = handle_zrangebyscore(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZREMRANGEBYLEX (14 bytes, z, tp[3]==109 'm') ──
                    elif cmd_eq(tp, tl, "zremrangebylex"):
                        _ = handle_zremrangebylex(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── ZREVRANGEBYSCORE (16 bytes, z, tp[3]==118 'v') ──
                    # The length was 14 (ZREVRANGEBYLEX's length) and the LEX arm
                    # below carried 16 — the two were cross-wired, so each command
                    # reached the OTHER's handler: ZREVRANGEBYSCORE answered []
                    # silently and ZREVRANGEBYLEX answered -ERR internal error.
                    # tp[3] ('v') is what separates these from ZREM*; the length is
                    # the only thing separating SCORE from LEX, so it has to be right.
                    elif cmd_eq(tp, tl, "zrevrangebyscore"):
                        _ = handle_zrevrangebyscore(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── ZREMRANGEBYRANK (15 bytes, z) ──
                    elif cmd_eq(tp, tl, "zremrangebyrank"):
                        _ = handle_zremrangebyrank(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── ZREMRANGEBYSCORE (16 bytes, z, tp[3]==109 'm') ──
                    elif cmd_eq(tp, tl, "zremrangebyscore"):
                        _ = handle_zremrangebyscore(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── ZREVRANGEBYLEX (14 bytes, z, tp[3]==118 'v') ── see the
                    # ZREVRANGEBYSCORE arm above: these two had swapped lengths.
                    elif cmd_eq(tp, tl, "zrevrangebylex"):
                        _ = handle_zrevrangebylex(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    # ── GEO commands ──
                    # gh #162: two fixes, both of the gh #156 token-skip family.
                    #  (1) Match the FULL command name. GEOSEARCHSTORE and
                    #      GEORADIUSBYMEMBER used to test only `tl == N and
                    #      (tp[0]|0x20) == 103` ('g'), so any 14- or 17-byte token
                    #      starting with 'g' was routed into a GEO handler.
                    #  (2) The command boundary is authoritative for the skip, as
                    #      for VSET: discard the handler's return and set
                    #      `i = cmd_end_tok - 1`, passing `cmd_end_tok` as the
                    #      handler's bound. The handlers' error paths returned a
                    #      skip that stopped at the token they rejected on (an
                    #      arity error left the key/args to be re-dispatched as
                    #      their own commands — one command in, several replies
                    #      out), and in a deep pipeline the accumulating desync
                    #      eventually mis-framed a bulk length and tripped the
                    #      catch-all `except`, which then dropped the whole batch.
                    # ── GEOPOS ──
                    elif tl == 6 and (tp[0]|0x20)==103 and (tp[1]|0x20)==101 and (tp[2]|0x20)==111 and (tp[3]|0x20)==112 and (tp[4]|0x20)==111 and (tp[5]|0x20)==115:
                        _ = handle_geopos(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool)
                        i = cmd_end_tok - 1
                    # ── GEODIST ──
                    elif tl == 7 and (tp[0]|0x20)==103 and (tp[1]|0x20)==101 and (tp[2]|0x20)==111 and (tp[3]|0x20)==100 and (tp[4]|0x20)==105 and (tp[5]|0x20)==115 and (tp[6]|0x20)==116:
                        _ = handle_geodist(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool)
                        i = cmd_end_tok - 1
                    # ── GEOHASH ──
                    elif tl == 7 and (tp[0]|0x20)==103 and (tp[1]|0x20)==101 and (tp[2]|0x20)==111 and (tp[3]|0x20)==104 and (tp[4]|0x20)==97 and (tp[5]|0x20)==115 and (tp[6]|0x20)==104:
                        _ = handle_geohash(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool)
                        i = cmd_end_tok - 1
                    # ── GEORADIUS ──
                    elif tl == 9 and (tp[0]|0x20)==103 and (tp[1]|0x20)==101 and (tp[2]|0x20)==111 and (tp[3]|0x20)==114 and (tp[4]|0x20)==97 and (tp[5]|0x20)==100 and (tp[6]|0x20)==105 and (tp[7]|0x20)==117 and (tp[8]|0x20)==115:
                        _ = handle_georadius(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool, self.dispatcher.wal, self.ttl_map)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "georadius_ro"):
                        _ = handle_georadius(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool, self.dispatcher.wal, self.ttl_map, True)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "georadiusbymember_ro"):
                        _ = handle_georadiusbymember(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool, self.dispatcher.wal, self.ttl_map, True)
                        i = cmd_end_tok - 1
                    # ── GEOSEARCH ──
                    elif tl == 9 and (tp[0]|0x20)==103 and (tp[1]|0x20)==101 and (tp[2]|0x20)==111 and (tp[3]|0x20)==115 and (tp[4]|0x20)==101 and (tp[5]|0x20)==97 and (tp[6]|0x20)==114 and (tp[7]|0x20)==99 and (tp[8]|0x20)==104:
                        _ = handle_geosearch(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool, self.dispatcher.wal, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── GEOSEARCHSTORE ──
                    elif tl == 14 and (tp[0]|0x20)==103 and (tp[1]|0x20)==101 and (tp[2]|0x20)==111 and (tp[3]|0x20)==115 and (tp[4]|0x20)==101 and (tp[5]|0x20)==97 and (tp[6]|0x20)==114 and (tp[7]|0x20)==99 and (tp[8]|0x20)==104 and (tp[9]|0x20)==115 and (tp[10]|0x20)==116 and (tp[11]|0x20)==111 and (tp[12]|0x20)==114 and (tp[13]|0x20)==101:
                        _ = handle_geosearchstore(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool, self.dispatcher.wal, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── GEORADIUSBYMEMBER ──
                    elif tl == 17 and (tp[0]|0x20)==103 and (tp[1]|0x20)==101 and (tp[2]|0x20)==111 and (tp[3]|0x20)==114 and (tp[4]|0x20)==97 and (tp[5]|0x20)==100 and (tp[6]|0x20)==105 and (tp[7]|0x20)==117 and (tp[8]|0x20)==115 and (tp[9]|0x20)==98 and (tp[10]|0x20)==121 and (tp[11]|0x20)==109 and (tp[12]|0x20)==101 and (tp[13]|0x20)==109 and (tp[14]|0x20)==98 and (tp[15]|0x20)==101 and (tp[16]|0x20)==114:
                        _ = handle_georadiusbymember(tokens, i, cmd_end_tok, writer, self.keyspace, self.skip_list_pool, self.dispatcher.wal, self.ttl_map)
                        i = cmd_end_tok - 1
                    # ── Stream Commands (src/commands/stream.mojo) ──
                    elif cmd_eq(tp, tl, "xlen"):
                        _ = handle_xlen(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    elif tl == 4 and cmd_matches_4(tp, 120, 97, 100, 100):
                        # XADD (x=120,a=97,d=100,d=100)
                        # After XADD, notify blocked XREAD readers waiting on this key
                        var _xadd_key_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
                        var _xadd_key_len = 0
                        if i + 1 < cmd_end_tok:
                            _xadd_key_ptr = tokens[i + 1].ptr
                            _xadd_key_len = tokens[i + 1].length
                        _ = handle_xadd(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                        # The readers are answered by the event loop's next
                        # tick, not here: this connection is mid-batch.
                        if self.blocked_readers.count_ptr[0] > 0 and _xadd_key_len > 0:
                            self.blocked_readers.mark_ready(_xadd_key_ptr, _xadd_key_len)
                    elif cmd_eq(tp, tl, "xack"):
                        # XACK (x=120,a=97,c=99,k=107)
                        _ = handle_xack(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xdel"):
                        _ = handle_xdel(tokens, i, cmd_end_tok, writer, self.keyspace, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xread"):
                        # A BLOCK that found no data parks the
                        # connection, as WAIT does (gh #390): no reply yet, and
                        # nothing pipelined behind it may run first. Stop at its
                        # byte end; the engine answers it and resumes the rest.
                        var can_block = self.can_park_wait and on_primary and cmd_idx < num_cmds and fd >= 0
                        if handle_xread(tokens, i, cmd_end_tok, writer, self.keyspace, self.blocked_readers, fd, can_block):
                            self.parked_waits.park_fd(fd)
                            primary_consumed = cmd_byte_ends[cmd_idx]
                            i = num_tokens
                        else:
                            i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xtrim"):
                        _ = handle_xtrim(tokens, i, cmd_end_tok, writer, self.keyspace)
                        if is_not_null(self.dispatcher.wal) and i + 1 < cmd_end_tok:   # effect not logged by the handler
                            self.dispatcher.wal[].log_key_image(self.keyspace, self.ttl_map, tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xinfo"):
                        _ = handle_xinfo(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xrange"):
                        _ = handle_xrange(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xgroup"):
                        _ = handle_xgroup(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xclaim"):
                        _ = handle_xclaim(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xpending"):
                        _ = handle_xpending(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xrevrange"):
                        _ = handle_xrevrange(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xautoclaim"):
                        # XAUTOCLAIM (x,a,...)
                        _ = handle_xautoclaim(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "xreadgroup"):
                        # XREADGROUP (x,r,...) — was silently routing to handle_xautoclaim (gh #81)
                        _ = handle_xreadgroup(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    # ── Pub/Sub Commands (src/commands/pubsub.mojo) ──
                    elif cmd_eq(tp, tl, "pubsub"):
                        _ = handle_pubsub(tokens, i, cmd_end_tok, writer, self.pubsub)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "publish") or cmd_eq(tp, tl, "spublish"):
                        # Deliveries go through the engine's writer, also from a
                        # script (whose own writer only captures the reply).
                        # Outside a script that is `writer` itself, passed as
                        # null: a second pointer to a `mut` argument breaks
                        # its exclusivity, and at -O3 the reply then lands on
                        # a stale offset, over the message delivered to self.
                        var _dw = null_ptr[ResponseWriter, MutUntrackedOrigin]()
                        var _dkq = kq
                        if self.script_depth > 0:
                            _dw = self.script_main_writer
                            _dkq = self.script_kq
                        _ = handle_publish_kind(tokens, i, cmd_end_tok, writer, _dw, _dkq, fd, self.pubsub, server,
                                                self.tx_state.resp_proto, self.worker_id, self.num_workers,
                                                cmd_eq(tp, tl, "spublish"))
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "subscribe"):
                        _ = handle_subscribe_kind(tokens, i, cmd_end_tok, writer, fd, self.pubsub, KIND_CHANNEL)
                        self.update_dispatch_gate()
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "unsubscribe"):
                        _ = handle_unsubscribe_kind(tokens, i, cmd_end_tok, writer, fd, self.pubsub, KIND_CHANNEL)
                        self.update_dispatch_gate()
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "psubscribe"):
                        _ = handle_subscribe_kind(tokens, i, cmd_end_tok, writer, fd, self.pubsub, KIND_PATTERN)
                        self.update_dispatch_gate()
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "punsubscribe"):
                        _ = handle_unsubscribe_kind(tokens, i, cmd_end_tok, writer, fd, self.pubsub, KIND_PATTERN)
                        self.update_dispatch_gate()
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "ssubscribe"):
                        _ = handle_subscribe_kind(tokens, i, cmd_end_tok, writer, fd, self.pubsub, KIND_SHARD)
                        self.update_dispatch_gate()
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "sunsubscribe"):
                        _ = handle_unsubscribe_kind(tokens, i, cmd_end_tok, writer, fd, self.pubsub, KIND_SHARD)
                        self.update_dispatch_gate()
                        i = cmd_end_tok - 1
                    # ── Transaction Commands (src/commands/transaction.mojo) ──
                    elif tl == 5 and cmd_matches_5(tp, 109, 117, 108, 116, 105):
                        _ = handle_multi(fd, self.tx_state, writer)
                        i = cmd_end_tok - 1
                    elif tl == 4 and cmd_matches_4(tp, 101, 120, 101, 99):
                        # EXEC: stage queued commands for replay AFTER this dispatch
                        # loop completes. handle_exec_start has already written the
                        # *N\r\n array header into `writer`; the per-command replies
                        # accumulate as we walk q[0..tx_cnt) below.
                        #
                        # gh #94 follow-up (WATCH parity): the prior implementation
                        # recursed into process_slow_path here, which blew the stack
                        # on parallelize worker threads (smaller default than the
                        # main thread). Converting to an outer-loop iteration keeps
                        # one stack frame and still gives each queued command a
                        # fresh parse + dispatch via the same `while True:` retry
                        # below.
                        # gh #261: memory crossed --maxmemory after a denyoom
                        # command was queued. Redis aborts the WHOLE transaction;
                        # letting each replayed command refuse itself would
                        # apply it in part. Unlike the per-command check this
                        # asks C directly: EXEC is on the slow path already and
                        # the housekeeping hint may be up to 64 ticks old.
                        if self.tx_state.is_multi(fd) and not self.tx_state.is_dirty(fd) \
                           and external_call["pion_maxmemory_check", Int32]() != 0 \
                           and tx_queue_has_denyoom(fd, self.tx_state):
                            self.tx_state.discard(fd)
                            writer.append_error_response(
                                "EXECABORT Transaction discarded because of: " + "OOM command not allowed when used memory > 'maxmemory'.")
                            exec_replay_count = -1
                        else:
                            exec_replay_count = handle_exec_start(fd, self.tx_state, writer, self._expiry_now())
                        if exec_replay_count > 0:
                            exec_replay_q = self.tx_state.queues[Int(fd)]
                            exec_replay_qi = 0
                            # gh #219: the replayed replies must fill the array
                            # header handle_exec_start just wrote, so NOTHING
                            # else may be dispatched between here and the replay
                            # continuation below. Commands pipelined behind EXEC
                            # in this same recv buffer would otherwise emit
                            # their replies INTO the transaction's array — with
                            # a pipelined MULTI (redis-py's default pipeline)
                            # `MULTI/SET/EXEC/GET/PING` returned `*1` whose sole
                            # element was GET's reply, and the replayed SET's
                            # +OK landed after PING's.
                            #
                            # Stop the primary batch at EXEC's byte end; the
                            # engine's drain loop (`_dispatch_recv_buffer`)
                            # re-enters with the remainder once the replay has
                            # closed out the array.
                            if on_primary and cmd_idx < num_cmds and cmd_byte_ends[cmd_idx] < consumed_bytes:
                                primary_consumed = cmd_byte_ends[cmd_idx]
                                i = num_tokens  # leave the dispatch loop
                    elif tl == 7 and cmd_matches_7(tp, 100, 105, 115, 99, 97, 114, 100):
                        _ = handle_discard(fd, self.tx_state, writer)
                        i = cmd_end_tok - 1
                    elif tl == 5 and cmd_matches_5(tp, 119, 97, 116, 99, 104):
                        _ = handle_watch(tokens, i, cmd_end_tok, fd, self.tx_state, writer, self.keyspace, self._expiry_now())
                        i = cmd_end_tok - 1
                    elif tl == 7 and cmd_matches_7(tp, 117, 110, 119, 97, 116, 99, 104):
                        _ = handle_unwatch(fd, self.tx_state, writer)
                        i = cmd_end_tok - 1
                    # ── Server / Admin Commands ──
                    elif cmd_eq(tp, tl, "quit"):
                        _ = handle_quit(writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "auth"):
                        _ = handle_auth(tokens, i, cmd_end_tok, writer, config.server.requirepass, self.tx_state.authed, fd,
                                         rebind[UnsafePointer[TenantTable, MutUntrackedOrigin]](UnsafePointer(to=self.tenant_table)),
                                         self.tx_state.tenant_id)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "flushdb"):
                        _ = handle_flushdb(tokens, i, cmd_end_tok, self.keyspace, writer, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # All six bytes: `tl == 6 and 'd','b'` matched ANY six-byte
                    # command starting "db" (DBSIZ\0 ran DBSIZE), which is the
                    # gh #162 shape — match the whole command name.
                    elif tl == 6 and (tp[0]|0x20) == 100 and (tp[1]|0x20) == 98 and (tp[2]|0x20) == 115 and (tp[3]|0x20) == 105 and (tp[4]|0x20) == 122 and (tp[5]|0x20) == 101:
                        _ = handle_dbsize(self.keyspace, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "select"):
                        _ = handle_select(tokens, i, cmd_end_tok, writer,
                                          is_not_null(self.cluster) and self.cluster[].enabled)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "swapdb"):
                        _ = handle_swapdb(tokens, i, cmd_end_tok, writer,
                                          is_not_null(self.cluster) and self.cluster[].enabled)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "move"):
                        _ = handle_move(tokens, i, cmd_end_tok, writer,
                                        is_not_null(self.cluster) and self.cluster[].enabled)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "bgrewriteaof"):
                        _ = handle_bgrewriteaof(self.dispatcher, self.keyspace, writer, self.ttl_map)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "command"):
                        _ = handle_command(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "debug"):
                        _ = handle_debug(tokens, i, cmd_end_tok, writer, self.keyspace,
                                         config.server.enable_debug_command, fd)   # #45
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "slowlog"):
                        _ = handle_slowlog(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "latency"):
                        _ = handle_latency(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "memory"):
                        _ = handle_memory(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "module"):
                        _ = handle_module(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "acl"):
                        _ = handle_acl(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "reset"):
                        if cmd_end_tok - i != 1:
                            writer.append_error_response("ERR wrong number of arguments for 'reset' command")
                        else:
                            self._reset_connection(fd, writer)
                        i = cmd_end_tok - 1
                    # ── CLIENT ── (6 bytes: c=99,l=108,i=105,e=101,n=110,t=116)
                    elif tl == 6 and cmd_matches_6(tp, 99, 108, 105, 101, 110, 116):
                        _ = handle_client(tokens, i, cmd_end_tok, fd, writer, self.tx_state.client_names)
                        i = cmd_end_tok - 1
                    # ── BLPOP / BRPOP ── (gh #318, #38)
                    # gh #423: both name literals stay on the `elif` line so
                    # gen_command_table.py sees BRPOP too.
                    elif tl == 5 and (cmd_matches_5(tp, 98, 108, 112, 111, 112) or cmd_matches_5(tp, 98, 114, 112, 111, 112)):
                        var _bpop_left = cmd_matches_5(tp, 98, 108, 112, 111, 112)
                        if cmd_end_tok - i < 3:
                            writer.append_error_response("ERR wrong number of arguments for '"
                                                         + String("blpop" if _bpop_left else "brpop") + "' command")
                        else:
                            var _bdl = parse_block_timeout(tokens[cmd_end_tok - 1], Int64(_get_now_ns() // 1_000_000), writer)
                            if _bdl >= 0 and not self._bpop_lists(tokens, i + 1, cmd_end_tok - 1, _bpop_left, writer):
                                if self._park_blocked(fd, buffer, cmd_idx, num_cmds, cmd_byte_ends, on_primary,
                                                      tokens, i + 1, cmd_end_tok - 1, _bdl, False):
                                    primary_consumed = cmd_byte_ends[cmd_idx]
                                    i = num_tokens
                                else:
                                    writer.append_null_array_response()
                        if i < num_tokens:
                            i = cmd_end_tok - 1
                    # ── EVAL / EVALSHA / FCALL and their _RO forms, SCRIPT, FUNCTION (#36) ──
                    # A script's redis.call() runs through this dispatcher
                    # (script_dispatch), so every command it calls logs its own
                    # WAL record; the old images of the KEYS[] keys are gone.
                    elif tl == 4 and cmd_matches_4(tp, 101, 118, 97, 108):
                        self._script_context(fd, server, kq, hnsw, db_size, config, writer)
                        _ = handle_eval(tokens, i, cmd_end_tok, writer, self.lua_engine, self._host(), False)
                        i = cmd_end_tok - 1
                    elif tl == 7 and cmd_matches_7(tp, 101, 118, 97, 108, 115, 104, 97):
                        self._script_context(fd, server, kq, hnsw, db_size, config, writer)
                        _ = handle_evalsha(tokens, i, cmd_end_tok, writer, self.lua_engine, self._host(), False)
                        i = cmd_end_tok - 1
                    elif tl == 6 and cmd_matches_6(tp, 115, 99, 114, 105, 112, 116):
                        _ = handle_script(tokens, i, cmd_end_tok, writer, self.lua_engine)
                        i = cmd_end_tok - 1
                    elif tl == 5 and cmd_matches_5(tp, 102, 99, 97, 108, 108):
                        self._script_context(fd, server, kq, hnsw, db_size, config, writer)
                        _ = handle_fcall(tokens, i, cmd_end_tok, writer, self.lua_engine, self._host(), False)
                        i = cmd_end_tok - 1
                    elif tl == 8 and cmd_matches_8(tp, 102, 117, 110, 99, 116, 105, 111, 110):
                        _ = handle_function(tokens, i, cmd_end_tok, writer, self.lua_engine, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # ── VSET commands ── (v=118)
                    # gh #156: the handler return value is NOT the token skip.
                    # VADD/VSIM's error paths returned a count that stopped at
                    # the token they rejected on, so the rest of the command
                    # (the scalars, the element name) was re-dispatched and
                    # answered a second time — one command in, two replies out,
                    # and every later reply in a pipeline attributed to the
                    # wrong request. Same shape as NEURON.PKM.* below: the
                    # parser's command boundary is authoritative, so discard the
                    # return and skip to cmd_end_tok unconditionally. Handlers
                    # also get cmd_end_tok as their bound, not num_tokens, so an
                    # optional-argument scan (VSIM's TOPN/WITHSCORES loop,
                    # VEMB's RAW, VRANDMEMBER's count) can't reach into the next
                    # pipelined command's tokens.
                    # VADD (4 bytes: v=118,a=97,d=100,d=100)
                    elif tl == 4 and cmd_matches_4(tp, 118, 97, 100, 100):
                        _ = handle_vadd(tokens, i, cmd_end_tok, self.keyspace, writer, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # VSIM (4 bytes: v=118,s=115,i=105,m=109)
                    elif tl == 4 and cmd_matches_4(tp, 118, 115, 105, 109):
                        _ = handle_vsim(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # VCARD (5 bytes: v=118,c=99,a=97,r=114,d=100)
                    elif tl == 5 and cmd_matches_5(tp, 118, 99, 97, 114, 100):
                        _ = handle_vcard(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # VDIM (4 bytes: v=118,d=100,i=105,m=109)
                    elif tl == 4 and cmd_matches_4(tp, 118, 100, 105, 109):
                        _ = handle_vdim(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # VREM (4 bytes: v=118,r=114,e=101,m=109)
                    elif tl == 4 and cmd_matches_4(tp, 118, 114, 101, 109):
                        _ = handle_vrem(tokens, i, cmd_end_tok, self.keyspace, writer, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # VEMB (4 bytes: v=118,e=101,m=109,b=98)
                    elif tl == 4 and cmd_matches_4(tp, 118, 101, 109, 98):
                        _ = handle_vemb(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # VISMEMBER (9 bytes)
                    elif tl == 9 and (tp[0]|0x20)==118 and (tp[1]|0x20)==105 and (tp[2]|0x20)==115 and (tp[3]|0x20)==109 and (tp[4]|0x20)==101 and (tp[5]|0x20)==109 and (tp[6]|0x20)==98 and (tp[7]|0x20)==101 and (tp[8]|0x20)==114:
                        _ = handle_vismember(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # VSETATTR (8 bytes)
                    elif tl == 8 and (tp[0]|0x20)==118 and (tp[1]|0x20)==115 and (tp[2]|0x20)==101 and (tp[3]|0x20)==116 and (tp[4]|0x20)==97 and (tp[5]|0x20)==116 and (tp[6]|0x20)==116 and (tp[7]|0x20)==114:
                        _ = handle_vsetattr(tokens, i, cmd_end_tok, self.keyspace, writer, self.dispatcher.wal)
                        i = cmd_end_tok - 1
                    # VGETATTR (8 bytes)
                    elif tl == 8 and (tp[0]|0x20)==118 and (tp[1]|0x20)==103 and (tp[2]|0x20)==101 and (tp[3]|0x20)==116 and (tp[4]|0x20)==97 and (tp[5]|0x20)==116 and (tp[6]|0x20)==116 and (tp[7]|0x20)==114:
                        _ = handle_vgetattr(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # VINFO (5 bytes: v=118,i=105,n=110,f=102,o=111)
                    elif tl == 5 and cmd_matches_5(tp, 118, 105, 110, 102, 111):
                        _ = handle_vinfo(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # VRANDMEMBER (11 bytes)
                    elif tl == 11 and (tp[0]|0x20)==118 and (tp[1]|0x20)==114 and (tp[2]|0x20)==97 and (tp[3]|0x20)==110 and (tp[4]|0x20)==100 and (tp[5]|0x20)==109 and (tp[6]|0x20)==101 and (tp[7]|0x20)==109 and (tp[8]|0x20)==98 and (tp[9]|0x20)==101 and (tp[10]|0x20)==114:
                        _ = handle_vrandmember(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # VLINKS (6 bytes: v=118,l=108,i=105,n=110,k=107,s=115)
                    elif tl == 6 and cmd_matches_6(tp, 118, 108, 105, 110, 107, 115):
                        _ = handle_vlinks(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # VRANGE (6 bytes: v=118,r=114,a=97,n=110,g=103,e=101)
                    elif tl == 6 and cmd_matches_6(tp, 118, 114, 97, 110, 103, 101):
                        _ = handle_vrange(tokens, i, cmd_end_tok, self.keyspace, writer)
                        i = cmd_end_tok - 1
                    # ── NEURON.PKM.* (gh #146) ──
                    # Deliberately at the TAIL of the chain: the elif ladder is
                    # ordered by descending frequency, and a memory-layer lookup
                    # is rarer than every stream/pubsub/transaction command above
                    # it. Placing it upstream of XADD cost measurable RPS in the
                    # pre-push gate.
                    # Prefix "NEURON.PKM." is 11 bytes, so tp[11] is the
                    # subcommand's first byte; SETKEYS/SETVALS share a length
                    # and split on tp[14]. All handlers consume a whole command,
                    # so every branch uses the cmd_end_tok skip.
                    elif cmd_eq(tp, tl, "neuron.pkm.create"):
                        # NEURON.PKM.CREATE
                        _ = handle_neuron_pkm_create(tokens, i, cmd_end_tok, writer, self.pkm)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "neuron.pkm.setkeys"):
                        # NEURON.PKM.SETKEYS
                        _ = handle_neuron_pkm_setkeys(tokens, i, cmd_end_tok, writer, self.pkm)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "neuron.pkm.setvals"):
                        # NEURON.PKM.SETVALS
                        _ = handle_neuron_pkm_setvals(tokens, i, cmd_end_tok, writer, self.pkm)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "neuron.pkm.query"):
                        # NEURON.PKM.QUERY
                        _ = handle_neuron_pkm_query(tokens, i, cmd_end_tok, writer, self.pkm)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "neuron.pkm.ffn"):
                        # NEURON.PKM.FFN
                        _ = handle_neuron_pkm_ffn(tokens, i, cmd_end_tok, writer, self.pkm)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "neuron.pkm.info"):
                        # NEURON.PKM.INFO
                        _ = handle_neuron_pkm_info(tokens, i, cmd_end_tok, writer, self.pkm)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "neuron.pkm.drop"):
                        # NEURON.PKM.DROP
                        _ = handle_neuron_pkm_drop(tokens, i, cmd_end_tok, writer, self.pkm)
                        i = cmd_end_tok - 1
                    # ── PION.STATS [RESET] (gh #262) ── the value receipt, per worker
                    elif cmd_eq(tp, tl, "pion.stats"):
                        var _ps_reset = False
                        if i + 1 < cmd_end_tok:
                            var _pa = tokens[unsafe_offset=i + 1]
                            _ps_reset = arg_eq(_pa.ptr, Int(_pa.length), "reset")
                        if _ps_reset:
                            self.ledger.reset()
                            self.scache.hits = UInt64(0)
                            self.scache.misses = UInt64(0)
                            writer.append_ok_response()
                        else:
                            _ = handle_pion_stats(writer, self.ledger, self.scache.hits, self.scache.misses,
                                                  self.moe_tier.hits, self.moe_tier.misses, self.worker_id)
                        i = cmd_end_tok - 1
                    # ── TIME (#30) ── at the tail: see the elif-order note above
                    elif cmd_eq(tp, tl, "time"):
                        if i + 1 < cmd_end_tok:
                            writer.append_error_response("ERR wrong number of arguments for 'time' command")
                        else:
                            handle_time(writer)
                        i = cmd_end_tok - 1
                    # ── EVAL_RO / EVALSHA_RO / FCALL_RO (#36) ── at the tail
                    elif cmd_eq(tp, tl, "eval_ro"):
                        self._script_context(fd, server, kq, hnsw, db_size, config, writer)
                        _ = handle_eval(tokens, i, cmd_end_tok, writer, self.lua_engine, self._host(), True)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "evalsha_ro"):
                        self._script_context(fd, server, kq, hnsw, db_size, config, writer)
                        _ = handle_evalsha(tokens, i, cmd_end_tok, writer, self.lua_engine, self._host(), True)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "fcall_ro"):
                        self._script_context(fd, server, kq, hnsw, db_size, config, writer)
                        _ = handle_fcall(tokens, i, cmd_end_tok, writer, self.lua_engine, self._host(), True)
                        i = cmd_end_tok - 1
                    # ── BRPOPLPUSH / BLMOVE / BLMPOP / BZPOPMIN / BZPOPMAX / BZMPOP (#38) ──
                    # A blocking command that finds nothing parks (_park_blocked)
                    # and stops the batch at its end; the engine runs it again
                    # once a key it waits on has data or its timeout passes.
                    elif cmd_eq(tp, tl, "brpoplpush") or cmd_eq(tp, tl, "blmove"):
                        var _blm = cmd_eq(tp, tl, "blmove")
                        var _bn = 6 if _blm else 4
                        if cmd_end_tok - i != _bn:
                            writer.append_error_response("ERR wrong number of arguments for '"
                                                         + String("blmove" if _blm else "brpoplpush") + "' command")
                        else:
                            var _bdir_ok = True
                            if _blm:
                                for _d in range(i + 3, i + 5):
                                    if not arg_eq(tokens[_d].ptr, tokens[_d].length, "left") \
                                       and not arg_eq(tokens[_d].ptr, tokens[_d].length, "right"):
                                        _bdir_ok = False
                            if not _bdir_ok:
                                writer.append_error_response("ERR syntax error")
                            else:
                                var _bdl = parse_block_timeout(tokens[cmd_end_tok - 1], Int64(_get_now_ns() // 1_000_000), writer)
                                if _bdl >= 0:
                                    var _bsv = self.keyspace[].get(GenericValue.borrow(tokens[i + 1].ptr, tokens[i + 1].length))
                                    if not _bsv.is_none():
                                        # the source exists: the move runs (or errors) now
                                        if _blm:
                                            _ = handle_lmove(tokens, i, i + 5, writer, self.keyspace, self.dispatcher)
                                        else:
                                            self._rpoplpush(tokens, i, writer)
                                    elif self._park_blocked(fd, buffer, cmd_idx, num_cmds, cmd_byte_ends, on_primary,
                                                            tokens, i + 1, i + 2, _bdl, False):
                                        primary_consumed = cmd_byte_ends[cmd_idx]
                                        i = num_tokens
                                    else:
                                        writer.append_null_response()
                        if i < num_tokens:
                            i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "blmpop") or cmd_eq(tp, tl, "bzmpop"):
                        # B?MPOP timeout numkeys key [key ...] LEFT|RIGHT / MIN|MAX [COUNT count]:
                        # the arguments first, then the timeout, as Redis parses them.
                        var _bz = cmd_eq(tp, tl, "bzmpop")
                        if cmd_end_tok - i < 5:
                            writer.append_error_response("ERR wrong number of arguments for '"
                                                         + String("bzmpop" if _bz else "blmpop") + "' command")
                        else:
                            var _bmp = parse_mpop(tokens, i + 2, cmd_end_tok, _bz)
                            if _bmp.error.byte_length() > 0:
                                writer.append_error_response(_bmp.error)
                            else:
                                var _bdl = parse_block_timeout(tokens[i + 1], Int64(_get_now_ns() // 1_000_000), writer)
                                if _bdl >= 0:
                                    var _served: Bool
                                    if _bz:
                                        _served = zmpop_pop(tokens, i + 3, _bmp.numkeys, _bmp.first, _bmp.count,
                                                            writer, self.keyspace, self.dispatcher.wal)
                                    else:
                                        _served = self._mpop_lists(tokens, i + 3, _bmp.numkeys, _bmp.first, _bmp.count, writer)
                                    if not _served:
                                        if self._park_blocked(fd, buffer, cmd_idx, num_cmds, cmd_byte_ends, on_primary,
                                                              tokens, i + 3, i + 3 + _bmp.numkeys, _bdl, _bz):
                                            primary_consumed = cmd_byte_ends[cmd_idx]
                                            i = num_tokens
                                        else:
                                            writer.append_null_array_response()
                        if i < num_tokens:
                            i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "bzpopmin") or cmd_eq(tp, tl, "bzpopmax"):
                        var _bmin = cmd_eq(tp, tl, "bzpopmin")
                        if cmd_end_tok - i < 3:
                            writer.append_error_response("ERR wrong number of arguments for '"
                                                         + String("bzpopmin" if _bmin else "bzpopmax") + "' command")
                        else:
                            var _bdl = parse_block_timeout(tokens[cmd_end_tok - 1], Int64(_get_now_ns() // 1_000_000), writer)
                            if _bdl >= 0 and not self._bzpop(tokens, i + 1, cmd_end_tok - 1, _bmin, writer):
                                if self._park_blocked(fd, buffer, cmd_idx, num_cmds, cmd_byte_ends, on_primary,
                                                      tokens, i + 1, cmd_end_tok - 1, _bdl, True):
                                    primary_consumed = cmd_byte_ends[cmd_idx]
                                    i = num_tokens
                                else:
                                    writer.append_null_array_response()
                        if i < num_tokens:
                            i = cmd_end_tok - 1
                    # ── #39: commands Redis 7 has that Pion lacked ──
                    elif cmd_eq(tp, tl, "lcs"):
                        _ = handle_lcs(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "role"):
                        var _role_tail = Int(self.dispatcher.wal[].tail_offset) if is_not_null(self.dispatcher.wal) else 0
                        handle_role(tokens, i, cmd_end_tok, writer, self.cluster, _role_tail)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "lolwut"):
                        handle_lolwut(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "monitor"):
                        if cmd_end_tok - i != 1:
                            writer.append_error_response("ERR wrong number of arguments for 'monitor' command")
                        elif not on_primary:
                            # queued by MULTI: Redis refuses it at EXEC
                            writer.append_error_response("ERR MONITOR isn't allowed for DENY BLOCKING client")
                        elif not self.monitors.contains(fd):
                            self.monitors.add(fd)
                            self.update_dispatch_gate()
                            writer.append_ok_response()
                        # already monitoring: Redis ignores it, with no reply
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "pfselftest"):
                        _ = handle_pfselftest(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "pfdebug"):
                        _ = handle_pfdebug(tokens, i, cmd_end_tok, writer, self.keyspace)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "replicaof") or cmd_eq(tp, tl, "slaveof"):
                        handle_replicaof(tokens, i, cmd_end_tok, writer, self.cluster)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "failover"):
                        handle_failover(tokens, i, cmd_end_tok, writer, self.cluster)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "sync"):
                        handle_sync(tokens, i, cmd_end_tok, writer)
                        i = cmd_end_tok - 1
                    elif cmd_eq(tp, tl, "restore-asking"):
                        # RESTORE, as a cluster's MIGRATE sends it mid-resharding
                        if handle_restore(tokens, i, cmd_end_tok, writer, self.keyspace, self.ttl_map, self.dispatcher.wal):
                            self.tx_state.bump_key_version(tokens[i + 1].ptr, tokens[i + 1].length)
                            _ = ingest_whole_hash(self.shared_hnsw, self.keyspace, self.dispatcher.wal,   # #46
                                                  self.vec_tomb, tokens[i + 1].ptr, tokens[i + 1].length)
                        i = cmd_end_tok - 1
                    else:
                        writer.append_error_response("ERR unknown command '" + token.value() + "'")
                        # Skip remaining tokens of this command
                        i = cmd_end_tok - 1  # -1 because i += 1 below
                    # gh #240: STRUCTURAL enforcement of "a command consumes its
                    # own frame". Individual arms have violated this eight times
                    # (#156, #162, FT.*, #214, #218, #219, #221, #223) and three
                    # more times in one session (#238 ZPOPMIN, #239 EXISTS/GET/
                    # INCR/DECR/LLEN/LPOP/RPOP, and six 3-argument forms) — every
                    # fix converted the arms one probe could reach, and the next
                    # probe found more. An arm that under-consumes leaves its
                    # surplus tokens to be dispatched as commands: one command in,
                    # several replies out, and every later reply paired with the
                    # wrong request.
                    #
                    # Clamping FORWARD only. An arm that deliberately moved past
                    # this command (EXEC sets i = num_tokens to leave the loop) is
                    # untouched, because the guard only fires when i fell SHORT.
                    if i < cmd_end_tok - 1:
                        i = cmd_end_tok - 1
                    # #39 MONITOR: show the command that just ran. EXEC waits
                    # until the commands it runs (the replay below) are out.
                    if not _mon_done:
                        if exec_replay_count > 0 and tl == 4 and cmd_matches_4(tp, 101, 120, 101, 99):
                            self.monitor_exec_line = self._monitor_line_for(tokens, _mon_i, cmd_end_tok, fd)
                        else:
                            self._monitor_feed(tokens, _mon_i, cmd_end_tok, fd, writer, server, kq)
                    if self.script_depth == 0 and len(self.monitors.pending) > 0:
                        var _mpend = self.monitors.pending.copy()
                        self.monitors.pending.clear()
                        self._monitor_send(_mpend, fd, writer, server, kq)
                    i += 1
                    # Advance command index
                    while cmd_idx < num_cmds and i >= cmd_ends[cmd_idx]:
                        cmd_idx += 1
                # Replay continuation (in the same stack frame).
                if exec_replay_qi < exec_replay_count and is_not_null(exec_replay_q):
                    var rq = exec_replay_q[exec_replay_qi]
                    exec_replay_qi += 1
                    if is_not_null(rq.data) and rq.length > 0:
                        # Re-parse the queued frame, reset dispatch state,
                        # and re-enter via the outer while True.
                        on_primary = False
                        var rnt = 0
                        var rcb = 0
                        var rce = self.replay_ends_buf
                        var rcbe = self.replay_byte_ends_buf
                        var rnc = 0
                        self.parser.parse_stream(rq.data, rq.length, tokens, rnt, rcb, rce, rcbe, rnc)
                        # gh #153: a queued command too large for the token
                        # array can't be replayed. Report it rather than
                        # dropping it silently, and clamp so the dispatch
                        # loop below sees a plain empty parse.
                        if rnt == TOKENS_OVERFLOW:
                            writer.append_error_response(
                                "ERR queued command has too many arguments (max "
                                + String(MAX_CMD_TOKENS)
                                + ")"
                            )
                            rnt = 0
                        num_tokens = rnt
                        num_cmds = rnc
                        unsafe_memcpy(dest=cmd_ends.bitcast[UInt8](),
                               src=rce.bitcast[UInt8](), count=MAX_CMD_ENDS * 8)
                        unsafe_memcpy(dest=cmd_byte_ends.bitcast[UInt8](),
                               src=rcbe.bitcast[UInt8](), count=MAX_CMD_ENDS * 8)
                        i = 0
                        cmd_idx = 0
                        continue
                    continue  # empty queued slot
                # No more replay frames. If we replayed anything, free the queue.
                if exec_replay_count > 0:
                    self.tx_state.discard(fd)
                    if len(self.monitor_exec_line) > 0:      # #39: EXEC after what it ran
                        var _mexec = self.monitor_exec_line.copy()
                        self.monitor_exec_line.clear()
                        self._monitor_send(_mexec, fd, writer, server, kq)
                break
            writer.flush_response(fd, server, kq)
            return primary_consumed
        except e:
            # gh #162: a handler (or a pre-pass) raised mid-batch. Recover at
            # command granularity instead of discarding the whole recv buffer.
            #
            # Drop whatever half-formed response the failed command wrote, emit
            # one error for it, and — when we're on the primary frame and know
            # the command's byte extent — return the offset just past it so the
            # engine's drain loop re-parses and dispatches the rest of the batch
            # this same tick. That turns "one bad command silently eats every
            # command pipelined behind it" into "one bad command, one error".
            #
            # We fall back to consuming the whole buffer (`return n`, the
            # original behaviour) when there's no usable boundary to skip to:
            #   - a parse fault or degenerate parse (cmd_idx >= num_cmds),
            #   - an EXEC-replay frame (on_primary False), whose byte offsets
            #     index the queued frame, not this recv buffer.
            # Roll back only if nothing was flushed since the command began: a
            # handler that already writev'd/flushed part of its reply (FT.SEARCH,
            # SSM.PREFIX.FETCH, …) reset offset to 0, and forcing it back to
            # cmd_write_start would resurrect stale buffer bytes. In that rare
            # case leave offset where it is and just append the error after it.
            if cmd_write_start <= writer.offset:
                writer.offset = cmd_write_start
            # A handler that raised a REDIS error (strict_atol's "ERR value is
            # not an integer or out of range") gets that text as its reply;
            # anything else is an internal fault and says so.
            var emsg = String(e)
            if emsg.startswith("ERR ") or emsg.startswith("WRONGTYPE "):
                writer.append_error_response(emsg)
            else:
                writer.append_error_response("ERR internal error executing command")
            writer.flush_response(fd, server, kq)
            if on_primary and cmd_idx < num_cmds:
                var skip_to = cmd_byte_ends[cmd_idx]
                if skip_to > 0 and skip_to <= n:
                    return skip_to
            return n

    @always_inline
    def _expiry_now(self) -> Int64:
        """#45: the time a deadline is compared with: the batch clock, or the
        wall clock when no key has a TTL (the clock is then off)."""
        var c = self.keyspace[].clock_ns
        return c if c != 0 else Int64(_get_now_ns())

    @always_inline
    def _host(mut self) -> UnsafePointer[NoneType, MutUntrackedOrigin]:
        """This handler's address: the context pion_script_dispatch is given back."""
        return UnsafePointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=self)))

    def _script_context(mut self, fd: Int32, server: TCPServer, kq: Int32, mut hnsw: HNSWGraph,
                        mut db_size: Int, config: PionConfig, mut writer: ResponseWriter):
        """Remember the outer call's context for script_dispatch. Valid for this
        command only: the script runs synchronously inside it."""
        self.script_fd = fd
        self.script_main_writer = UnsafePointer[ResponseWriter, MutUntrackedOrigin](
            unsafe_from_address=Int(UnsafePointer(to=writer)))
        self.script_server = UnsafePointer[TCPServer, MutUntrackedOrigin](
            unsafe_from_address=Int(UnsafePointer(to=server)))
        self.script_kq = kq
        self.script_hnsw = UnsafePointer[HNSWGraph, MutUntrackedOrigin](
            unsafe_from_address=Int(UnsafePointer(to=hnsw)))
        self.script_db_size = UnsafePointer[Int, MutUntrackedOrigin](
            unsafe_from_address=Int(UnsafePointer(to=db_size)))
        self.script_config = UnsafePointer[PionConfig, MutUntrackedOrigin](
            unsafe_from_address=Int(UnsafePointer(to=config)))

    def script_dispatch(mut self, argc: Int,
                        argv: UnsafePointer[UnsafePointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                        lens: UnsafePointer[Int64, MutUntrackedOrigin], flags: Int, resp: Int,
                        reply: UnsafePointer[UnsafePointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                        wrote: UnsafePointer[Int64, MutUntrackedOrigin]) -> Int:
        """One redis.call() from a running script (#36): argv runs as a command
        through process_slow_path, and its reply comes back (reply[0], length)
        from the capture-only script writer. Redis's script checks come first:
        an unknown command, the arity, `noscript`, a write from a read-only
        script. flags: 1 read-only, 2 existence check only, 4 allow-oom."""
        var np = argv[0]
        var nl = Int(lens[0])
        if flags & 2 != 0:
            return 1 if command_exists(np, nl) else 0
        if is_null(self.script_writer):
            self.script_writer = alloc[ResponseWriter](1)
            self.script_writer.unsafe_write(ResponseWriter(capture_only=True))
        var w = self.script_writer
        w[].offset = 0
        w[].overflow_emitted = False
        w[].proto = UInt8(resp)
        reply[0] = w[].buffer
        var ar = command_arity(np, nl)
        var sp = argv[1] if argc > 1 else np
        var sl = Int(lens[1]) if argc > 1 else 0
        if not command_exists(np, nl):
            w[].append_error_response("ERR Unknown Redis command called from script")
        elif (ar > 0 and argc != ar) or (ar < 0 and argc < -ar):
            w[].append_error_response("ERR Wrong number of args calling Redis command from script")
        elif command_is_noscript(np, nl, sp, sl):
            w[].append_error_response("ERR This Redis command is not allowed from script")
        elif flags & 1 != 0 and command_is_write(np, nl):
            w[].append_error_response("ERR Write commands are not allowed from read-only scripts.")
        else:
            var need = 16
            for k in range(argc):
                need += Int(lens[k]) + 24
            if need > self.script_frame_cap:
                if is_not_null(self.script_frame):
                    self.script_frame.free()
                self.script_frame = alloc[UInt8](need)
                self.script_frame_cap = need
            var f = self.script_frame
            var o = 0
            f[o] = 42
            o += 1
            o += format_int_to_buf(f + o, 0, Int64(argc))
            f[o] = 13
            f[o + 1] = 10
            o += 2
            for k in range(argc):
                f[o] = 36
                o += 1
                o += format_int_to_buf(f + o, 0, lens[k])
                f[o] = 13
                f[o + 1] = 10
                o += 2
                unsafe_memcpy(dest=f + o, src=argv[k], count=Int(lens[k]))
                o += Int(lens[k])
                f[o] = 13
                f[o + 1] = 10
                o += 2
            var park = self.can_park_wait
            self.can_park_wait = False        # WAIT and XREAD BLOCK answer at once
            self.script_allow_oom = flags & 4 != 0
            self.script_depth += 1
            # kq = -1: the capture writer's flushes are no-ops (the XDP-lane
            # flush), so the reply stays in its buffer for the script.
            _ = self.process_slow_path(f, o, self.script_fd, w[], self.script_server[], Int32(-1),
                                       self.script_hnsw[], self.script_db_size[], self.script_config[])
            self.script_depth -= 1
            self.script_allow_oom = False
            self.can_park_wait = park
            if command_is_write(np, nl):
                wrote[0] = 1
        return w[].offset

    def drain_pubsub(mut self, mut writer: ResponseWriter, server: TCPServer, kq: Int32):
        """#42: deliver what other workers published since the last tick.
        Called per event-loop tick when there is more than one worker."""
        pubsub_drain(self.pubsub, self.worker_id, writer, server, kq, self.tx_state.resp_proto)

    def drain_deferred_shard_responses(mut self, mut hnsw: HNSWGraph, mut writer: ResponseWriter,
                                       server: TCPServer, kq: Int32) raises:
        """T3.4: poll deferred FT.SEARCH shard queries; send response when all shards reply.
        Called each engine event-loop iteration (both kqueue and io_uring). Non-blocking."""
        if self.deferred_count == 0: return
        if is_null(self.shared_hnsw) or is_null(self.shared_hnsw[].shard_bus): return
        var shard_bus = self.shared_hnsw[].shard_bus
        var n_sh = self.shared_hnsw[].num_shards

        # Serve other coordinators' queries (non-blocking, one pass per drain call).
        if hnsw.index_ready:
            for c in range(n_sh):
                if c == self.worker_id: continue
                var qidx = c * n_sh + self.worker_id
                if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shard_bus[].query_ready + qidx, UInt64(0)) == 0: continue
                var qs = shard_bus[].query_slots[qidx]
                Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shard_bus[].query_ready + qidx, UInt64(0))
                self.scratch_dists.clear()
                try:
                    var sids = hnsw.search_fp32_scored(qs.query_fp32, Int(qs.k), self.scratch_dists, Int(qs.ef))
                    shard_bus[].write_result(c, self.worker_id, sids, self.scratch_dists, qs.seq)
                except:
                    shard_bus[].result_counts[c * n_sh + self.worker_id] = 0
                    shard_bus[].result_seq[c * n_sh + self.worker_id] = qs.seq
                    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shard_bus[].result_ready + c * n_sh + self.worker_id, UInt64(1))

        # Check completion and send deferred responses.
        var i = 0
        while i < self.deferred_count:
            var active = self.deferred_active[i]
            var done   = self.deferred_done[i]
            var seq    = self.deferred_seqs[i]
            var ns     = Int(self.deferred_n_shards[i])
            # Check newly ready shards
            for shard_id in range(ns):
                if not ((active >> UInt32(shard_id)) & UInt32(1)): continue
                if (done >> UInt32(shard_id)) & UInt32(1): continue
                var ridx = self.worker_id * ns + shard_id
                if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shard_bus[].result_ready + ridx, UInt64(0)) == 0: continue
                if shard_bus[].result_seq[ridx] != seq: continue
                done |= (UInt32(1) << UInt32(shard_id))
            self.deferred_done[i] = done
            if done != active:
                # Timeout guard: if this entry has been stuck for >5M drain ticks (~5s),
                # a shard result was lost (race: two concurrent coordinators on same slot).
                # Send empty response to unblock the client and discard the entry.
                self.deferred_drain_ticks[i] += Int32(1)
                if self.deferred_drain_ticks[i] > Int32(5000000):
                    writer.append_empty_array_response()
                    writer.flush_response(self.deferred_fds[i], server, kq)
                    self.deferred_count -= 1
                    if i < self.deferred_count:
                        self.deferred_fds[i]         = self.deferred_fds[self.deferred_count]
                        self.deferred_seqs[i]         = self.deferred_seqs[self.deferred_count]
                        self.deferred_ks[i]           = self.deferred_ks[self.deferred_count]
                        self.deferred_n_shards[i]     = self.deferred_n_shards[self.deferred_count]
                        self.deferred_active[i]       = self.deferred_active[self.deferred_count]
                        self.deferred_done[i]         = self.deferred_done[self.deferred_count]
                        self.deferred_drain_ticks[i]  = self.deferred_drain_ticks[self.deferred_count]
                    continue  # i unchanged — slot now holds swapped-in entry
                i += 1
                continue
            # All shards ready — merge, write response, flush
            var all_ids    = List[Int](capacity=ns * MAX_SHARD_K)
            var all_scores = List[Float32](capacity=ns * MAX_SHARD_K)
            for shard_id in range(ns):
                if not ((active >> UInt32(shard_id)) & UInt32(1)): continue
                var idx2 = self.worker_id * ns + shard_id
                var cnt = Int(shard_bus[].result_counts[idx2])
                if cnt < 0 or cnt > MAX_SHARD_K: cnt = 0
                var base_ids    = shard_bus[].result_ids    + idx2 * MAX_SHARD_K
                var base_scores = shard_bus[].result_scores + idx2 * MAX_SHARD_K
                for r in range(cnt):
                    all_ids.append(Int(base_ids[r]))
                    all_scores.append(base_scores[r])
            var n_all   = len(all_ids)
            var final_k = Int(self.deferred_ks[i]) if Int(self.deferred_ks[i]) < n_all else n_all
            for fi in range(final_k):
                var min_idx   = fi
                var min_score = all_scores[fi]
                for fj in range(fi + 1, n_all):
                    if all_scores[fj] < min_score:
                        min_score = all_scores[fj]
                        min_idx = fj
                if min_idx != fi:
                    var tmp_id = all_ids[fi]
                    all_ids[fi] = all_ids[min_idx]
                    all_ids[min_idx] = tmp_id
                    var tmp_score = all_scores[fi]
                    all_scores[fi] = min_score
                    all_scores[min_idx] = tmp_score
            var merged_ids    = List[Int]()
            var merged_scores = List[Float32]()
            for ii in range(final_k):
                merged_ids.append(all_ids[ii])
                merged_scores.append(all_scores[ii])
            write_ft_search_response(writer, merged_ids, merged_scores, self.keyspace, self.shared_hnsw)
            writer.flush_response(self.deferred_fds[i], server, kq)
            # Remove slot (swap with last)
            self.deferred_count -= 1
            if i < self.deferred_count:
                self.deferred_fds[i]          = self.deferred_fds[self.deferred_count]
                self.deferred_seqs[i]         = self.deferred_seqs[self.deferred_count]
                self.deferred_ks[i]           = self.deferred_ks[self.deferred_count]
                self.deferred_n_shards[i]     = self.deferred_n_shards[self.deferred_count]
                self.deferred_active[i]       = self.deferred_active[self.deferred_count]
                self.deferred_done[i]         = self.deferred_done[self.deferred_count]
                self.deferred_drain_ticks[i]  = self.deferred_drain_ticks[self.deferred_count]
            # Don't increment i — slot i now holds swapped-in entry

