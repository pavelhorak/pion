from src.common.ptr import is_not_null, null_ptr
from src.common.container_free import free_graveyard
from std.collections import List
from std.memory import alloc
from std.memory.unsafe_pointer import Pointer

from src.actor.shard_manager import ShardManager
from src.actor.shard import Shard
from src.actor.vnode import VNode
from src.io.wal import WAL
from src.io.blob_store import BlobStore, BLOB_TIER_OFF
from src.io.snapshot import SnapshotEngine
from src.memory.buffer_pool import BufferPool
from src.vector.hnsw import HNSWGraph, SharedHNSWView
from src.network.gossip import GossipManager
from src.network.replication import PrimaryReplicator, ReplicaReceiver, apply_wal_entries
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.memory.slab_allocator import SlabAllocator
from src.memory.object_pool import ObjectPool
from src.common.topology import ClusterTopology
from src.network.raft import RaftNode
from src.common.config import PionConfig
from src.network.engine import NetworkEngine

from src.common.lock_free import LockFreeRingBuffer, AITask
from src.network.cluster import ClusterState, REPL_PORT_OFFSET
from std.ffi import external_call
from std.atomic import Atomic, Ordering

struct Pion:
    var nodes: List[String]
    var shard_manager: ShardManager
    var wal: Pointer[WAL, MutUntrackedOrigin]
    # gh #163: file-backed arena for values >= --blob-threshold.
    var blobs: Pointer[BlobStore, MutUntrackedOrigin]
    var snapshot_engine: SnapshotEngine
    var buffer_pool: BufferPool
    var hnsw: HNSWGraph
    var gossip: GossipManager
    var replication: PrimaryReplicator
    var replica_recv: ReplicaReceiver
    var local_shard: Shard
    var keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]
    var topology: ClusterTopology
    var raft: Pointer[RaftNode, MutUntrackedOrigin]
    var db_size: Int
    var config: PionConfig
    var engine: NetworkEngine
    var ai_queue: Pointer[LockFreeRingBuffer, MutUntrackedOrigin]
    var shared_hnsw: Pointer[SharedHNSWView, MutUntrackedOrigin]
    var worker_id: Int
    var num_workers: Int
    var cluster: Pointer[ClusterState, MutUntrackedOrigin]
    # TTL: per-worker key→expiry_ns map (GenericValue.INT = nanoseconds from CLOCK_MONOTONIC)
    var ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]
    var last_save_time: Int64     # Unix timestamp of last successful snapshot (0 = none)

    # Object Pools for hot-path allocations
    var hash_map_pool: Pointer[ObjectPool[SlabHashMap], MutUntrackedOrigin]
    var skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]
    var list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin]

    def __init__(out self, var node_list: List[String], config: PionConfig, shared_hnsw: Pointer[SharedHNSWView, MutUntrackedOrigin], shared_listen_fd: Int32 = -1, worker_id: Int = 0, num_workers: Int = 1, secondary_listen_fd: Int32 = Int32(-1), binary_listen_fd: Int32 = Int32(-1), cluster: Pointer[ClusterState, MutUntrackedOrigin] = null_ptr[ClusterState, MutUntrackedOrigin]()):
        self.nodes = node_list.copy()
        var node_list_copy = node_list^
        self.shard_manager = ShardManager(node_list_copy^)

        # === Phase 1: Allocate all Pointers (lightweight — just mmap metadata) ===
        self.wal = alloc[WAL](1)
        self.blobs = alloc[BlobStore](1)
        self.raft = alloc[RaftNode](1)
        self.keyspace = alloc[StripedHashMap](1)
        self.ttl_map = alloc[SlabHashMap](1)
        self.hash_map_pool = alloc[ObjectPool[SlabHashMap]](1)
        self.skip_list_pool = alloc[ObjectPool[SlabSkipList]](1)
        self.list_pool = alloc[ObjectPool[SlabList]](1)
        self.ai_queue = alloc[LockFreeRingBuffer](1)
        _ = self.ai_queue[].__init__(1024)

        # === Phase 2: Lightweight value-type init ===
        self.config = config
        self.db_size = 0
        self.shared_hnsw = shared_hnsw
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.cluster = cluster
        self.snapshot_engine = SnapshotEngine("pion.snapshot")
        self.last_save_time = Int64(0)
        self.buffer_pool = BufferPool(1024)
        self.replication = PrimaryReplicator()
        self.replica_recv = ReplicaReceiver()
        self.topology = ClusterTopology()
        self.gossip = GossipManager()
        self.local_shard = Shard(0)
        for i in range(10):
            self.local_shard.add_vnode(VNode(i))

        # === Phase 3: Create NetworkEngine and listen() BEFORE heavy allocation ===
        # Connections will queue in the kernel backlog (capacity 65535) while
        # Phase 4 heavy init completes. This eliminates Connection Refused on startup.
        self.engine = NetworkEngine(
            config.server.port,
            self.keyspace,
            self.config,
            self.hash_map_pool,
            self.skip_list_pool,
            self.list_pool,
            self.ai_queue,
            self.wal,
            self.raft,
            self.shared_hnsw,
            worker_id=worker_id,
            num_workers=num_workers,
            secondary_listen_fd=secondary_listen_fd,
            binary_listen_fd=binary_listen_fd,
            cluster=cluster,
            ttl_map=self.ttl_map,
        )
        if shared_listen_fd >= 0:
            # V3.1: use the single shared listen socket created in main() before parallelize.
            # All workers register this fd with their own kqueue and race to accept().
            self.engine.server.fd = shared_listen_fd
        else:
            _ = self.engine.server.listen()

        # === Phase 4: Heavy initialization (socket is already listening) ===
        # Per-worker WAL file: pion.wal.0, pion.wal.1, ... so each worker recovers its own data.
        self.wal.unsafe_write(WAL("pion.wal." + String(worker_id), worker_id,
                                       config.server.wal_size_mb * 1024 * 1024,
                                       config.server.wal_max_segments,
                                       not config.server.no_wal))
        # gh #260: --wal-full-policy. Set after construction rather than through
        # the ctor so the WAL keeps its safe default for every other caller
        # (tests, tooling, replication) that builds one without a ServerConfig.
        self.wal[].refuse_when_full = config.server.wal_refuse_when_full
        # gh #163: the blob arena must be mapped before WAL replay — cmd_id 4
        # entries resolve their pointers through it.
        self.blobs.unsafe_write(BlobStore("pion.blob." + String(worker_id), worker_id,
                                               not config.server.no_blob_tier))
        self.engine.fast_path.blobs = self.blobs
        # Disabled = BLOB_TIER_OFF sentinel (not 0), so the fast-path routing
        # stays a single compare. A non-positive --blob-threshold would route
        # every SET to the arena, so it also means "off".
        self.engine.fast_path.blob_threshold = config.server.blob_threshold \
            if not config.server.no_blob_tier and config.server.blob_threshold > 0 \
            else BLOB_TIER_OFF
        self.engine.slow_path.dispatcher.blobs = self.blobs
        self.engine.slow_path.dispatcher.blob_threshold = \
            self.engine.fast_path.blob_threshold
        self.raft.unsafe_write(RaftNode("node_" + String(config.server.port)))
        # Full-capacity HNSWGraph per worker — each worker builds the complete index independently.
        # Sharding (V18d) was reverted: task-1 workers never get CPU time in Mojo green thread model.
        var hnsw_max = config.vector.max_elements
        self.hnsw = HNSWGraph(
            hnsw_max,
            config.vector.dimensions,
            M=config.vector.M,
            ef_construction=config.vector.ef_construction,
            use_int4=config.vector.use_int4,
            use_bq=config.vector.use_bq,
            has_gpu=config.vector.has_gpu,
            use_huge_pages=config.server.use_huge_pages,
            polarquant=config.vector.polarquant,
            turboquant=config.vector.turboquant,
            nanoquant=config.vector.nanoquant,
        )
        # P1-A: lazy init — start at 65536 slots (~4MB/worker vs 650MB), _rehash() doubles on demand.
        # At 1M actual keys: 4 workers × ~65MB = 260MB total (vs 2,600MB at fixed 10M).
        # T3.2: StripedHashMap — 8 shards × 8192 slots = 65536 total. Each shard rehashes
        # independently at ~875K entries, eliminating the O(7M) single-map rehash pause.
        self.keyspace.unsafe_write(StripedHashMap(65536))
        # TTL map: starts small (opt-in feature), grows on demand.
        self.ttl_map.unsafe_write(SlabHashMap(256))
        self.keyspace[].ttl_map = self.ttl_map    # a removed key drops its TTL

        self.hash_map_pool.unsafe_write(ObjectPool[SlabHashMap](1000))
        for i in range(1000):
            self.hash_map_pool[].free_list[unsafe_offset=i].unsafe_write(SlabHashMap(16))

        self.skip_list_pool.unsafe_write(ObjectPool[SlabSkipList](1000))
        for i in range(1000):
            self.skip_list_pool[].free_list[unsafe_offset=i].unsafe_write(SlabSkipList(1024))

        self.list_pool.unsafe_write(ObjectPool[SlabList](1000))
        for i in range(1000):
            self.list_pool[].free_list[unsafe_offset=i].unsafe_write(SlabList())

        # === Phase 5: Snapshot + WAL recovery ===
        # 5a: Load snapshot (base keyspace state); WAL.checkpoint() was called after the snapshot
        #     so the WAL only contains deltas since the last snapshot.
        var snap_ts = self.snapshot_engine.load_snapshot(self.keyspace, worker_id, self.blobs, self.ttl_map)
        if snap_ts > 0:
            self.last_save_time = snap_ts
        # 5b: WAL recovery (replay entries written after last snapshot)
        _ = self.wal[].recover(self.keyspace, self.blobs, self.ttl_map)
        free_graveyard(self.keyspace)   # gh #394: aggregates replay overwrote

        # gh #167: 5c. Reclaim arena bytes stranded by overwrites/DELs. Runs
        # before the event loop starts, so nothing can be mid-GET on a value we
        # are about to move. A compaction invalidates every cmd-4 offset in the
        # log, so re-index immediately: the snapshot writes fresh pointer
        # records and the checkpoint drops the stale ones.
        if self.blobs[].compact(self.keyspace):
            var comp_ts = self.snapshot_engine.take_snapshot(
                self.keyspace, worker_id, self.wal[].seq, self.blobs, self.ttl_map)
            if comp_ts > 0:
                self.wal[].checkpoint()
                self.last_save_time = comp_ts
            else:
                print("Blob tier: compaction re-index FAILED (snapshot error) — "
                      + "the WAL still names pre-compaction offsets")

        # === Phase 6: HNSW load from disk (skips FT.OPTIMIZE on warm restart) ===
        # gh #211: thread the shared slot→key buffer through the load so the
        # persisted map lands in the cross-worker resolver, and publish the
        # loaded graph to the shared view — without the publish only the
        # loading worker can serve FT.SEARCH after a warm restart (ready_atomic
        # stays 0 and the other workers' borrow never fires).
        var hnsw_path = "pion.hnsw." + String(worker_id)
        var _hk_buf = null_ptr[UInt8, MutUntrackedOrigin]()
        var _hk_max = 0
        if is_not_null(self.shared_hnsw):
            _hk_buf = self.shared_hnsw[].hk_keys_buf
            _hk_max = self.shared_hnsw[].hk_max_elements
        if self.hnsw.load_from_disk(hnsw_path, _hk_buf, _hk_max):
            if is_not_null(self.shared_hnsw):
                self.hnsw.publish_to_shared(self.shared_hnsw)
        # #19: this worker's share of the startup load is done (published, or
        # refused as stale). Then wait for every other loader: a worker that
        # loads nothing used to serve at once and answer FT.SEARCH with "no
        # such index" until the loader had published. Clients that connect
        # meanwhile wait in the listen backlog, as they do during WAL replay.
        if is_not_null(self.shared_hnsw) and is_not_null(self.shared_hnsw[].warm_load_pending):
            var _wp = self.shared_hnsw[].warm_load_pending
            if _wp[unsafe_offset=1 + worker_id] == 1:
                # + (2^64 - 1) is - 1: the loader is done.
                _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELEASE](_wp, UInt64.MAX)
            var _waited_ms = 0
            while Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](_wp, UInt64(0)) > 0:
                _ = external_call["usleep", Int32](Int32(1000))
                _waited_ms += 1
                if _waited_ms == 2000:
                    print("Worker " + String(worker_id) + ": waiting for the persisted vector index to load")
                if _waited_ms >= 600_000:
                    # A loader that died never decrements; serving without
                    # the index beats never serving.
                    print("Worker " + String(worker_id) + ": gave up waiting for the vector index load after 600 s")
                    break

        # === Phase 7: Gossip + Replication (worker 0 only) ===
        # Only worker 0 starts background threads; other workers read shared ClusterState.
        # Gossip needs peers; REPLICATION does not. Both used to sit behind
        # `peer_count > 0`, and a replica adds its primary as a peer
        # automatically (main.mojo) but a primary started without
        # --cluster-peers has none — so it never opened its port+10000
        # listener and every replica streamed nothing (0/50 keys in
        # tests/test_replication.py, which had not been run).
        if worker_id == 0 and is_not_null(cluster) and cluster[].enabled:
            if cluster[].peer_count > 0:
                # Gossip: health-probe each peer via TCP PING
                self.gossip.setup(cluster, config.cluster.gossip_ping_ms,
                                  config.cluster.pfail_threshold, config.cluster.fail_threshold)
                cluster[].gossip_handle = self.gossip.handle
                var gossip_ok = self.gossip.start()
                if gossip_ok:
                    print("Gossip: health monitoring started for " + String(cluster[].peer_count) + " peer(s)")
            # Replication
            if cluster[].is_replica and config.cluster.primary_host != "":
                # Replica mode: connect to primary's replication port
                var repl_port = config.cluster.primary_port + REPL_PORT_OFFSET
                cluster[].repl_drain_buf = alloc[UInt8](4194304)  # 4MB drain buffer
                var recv_ok = self.replica_recv.setup(config.cluster.primary_host, repl_port, config.server.port)
                if recv_ok:
                    cluster[].repl_replica_handle = self.replica_recv.handle
                    # The receiver THREAD started; it connects (and retries)
                    # on its own. Saying "connected" here claimed a link that
                    # did not exist when the primary had no listener.
                    print("Replication: replica receiver started, connecting to " + config.cluster.primary_host + ":" + String(repl_port))
                else:
                    print("Replication: replica setup failed — check primary_host/port")
            elif not cluster[].is_replica:
                # Primary mode: listen for replica connections on port + 10000
                var listen_port = config.server.port + REPL_PORT_OFFSET
                var repl_ok = self.replication.setup(self.wal, listen_port)
                if repl_ok:
                    cluster[].repl_primary_handle = self.replication.handle
                    print("Replication: primary listener on port " + String(listen_port))
                else:
                    print("Replication: primary listener on port " + String(listen_port)
                          + " FAILED to start — replicas will receive nothing")
            # gh #149 / gh #163: N3 hands the raw WAL mapping to a C streaming
            # thread and stores it in cluster[].wal_ptr for failover promotion,
            # and the replica side (apply_wal_entries) cannot follow a rotation.
            # So with replication in play: pin the log — rotating it would munmap
            # the region that thread is reading — and keep large values in the
            # WAL where the stream can actually carry them. Both tiers stay off
            # rather than diverging a replica silently. A pinned log that fills
            # still reports it (wal_dropped_entries + "WAL: FULL"), which is
            # what the pre-gh #149 code never did.
            self.wal[].pinned = True
            if self.engine.fast_path.blob_threshold < BLOB_TIER_OFF:
                self.engine.fast_path.blob_threshold = BLOB_TIER_OFF
                self.engine.slow_path.dispatcher.blob_threshold = BLOB_TIER_OFF
                print("Blob tier: disabled under replication "
                      + "(the replication stream carries payloads, not blob pointers)")
            # N3: store WAL pointer and port for failover promotion
            cluster[].wal_ptr = Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(self.wal[].map))
            cluster[].server_port = config.server.port

    def run_server(mut self) raises:
        # V16: single-task per worker — full OS thread dedicated to KQUEUE event loop.
        # V18 dual_task (parallelize[dual_task](2,2)) caused task-1's usleep() to block the
        # OS thread, starving task-0's kqueue loop (Mojo green threads don't yield at OS syscalls).
        self.engine.run_server(self.hnsw, self.db_size)
