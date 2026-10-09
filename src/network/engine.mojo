from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.container_free import free_graveyard
from src.network.vector_ingest import log_dead_slots
from std.sys.info import CompilationTarget
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy, stack_allocation
from std.ffi import external_call
from std.collections import List
from src.common.skip_list import SlabSkipList
from src.network.server import KEvent, TCPServer, EPOLLIN, EPOLLOUT, EPOLLERR, EPOLLHUP, EPOLLET, EPOLLEXCLUSIVE, EPOLL_CTL_ADD, EPOLL_CTL_DEL, EPOLL_CTL_MOD
from src.network.server import epoll_ev_events, epoll_ev_fd, epoll_ctl_fd
from src.common.list import SlabList
from src.memory.slab_allocator import SlabAllocator
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.config import PionConfig
from src.memory.object_pool import ObjectPool
from src.vector.hnsw import HNSWGraph, SharedHNSWView
from src.vector.hnsw_types import HNSW_SHARDED_INGEST_ENABLED
from src.io.wal import WAL
from src.io.snapshot import SNAP_HDR_LEN
from src.io.blob_store import BlobStore
from src.network.raft import RaftNode
from src.common.lock_free import LockFreeRingBuffer, ShardQueryBus
from std.atomic import Atomic, Ordering
from std.memory import unsafe_memset

from src.network.response_writer import ResponseWriter
from src.network.fast_path import FastPathHandler
from src.network.slow_path import SlowPathHandler
from src.network.fast_path import _get_now_ns
from src.io.io_uring import IOUring, PBUF_RING_ENTRIES, PBUF_SIZE, URING_MAX_FDS, UD_RECV, UD_SEND, UD_ACCEPT, UD_TIMEOUT
from src.io.xdp import XDPEngine, XDPFrameInfo, TCP_SYN, TCP_ACK, TCP_FIN, TCP_RST, TCP_PSH
from src.network.cluster import ClusterState, REPL_PORT_OFFSET
from src.commands.stream import write_xread_reply
from src.commands.blocking import blocked_client_ready, UNBLOCK_ERROR, UNBLOCKED_ERROR

# Client receive buffer size. Supports LMCache KV cache blobs (typical 2-4 MB
# per chunk, up to ~16 MB for 70B+ models) and shared-KV-cache tensor frames
# (ATTEND.PREFIX.STORE at H=8 N=16K D=128 = ~128 MB K+V combined; H=32 N=4K
# D=64 = ~64 MB). 256 MB clears every Llama-class shape we ship today and
# extends to N=32K at H=8/D=128. Virtual memory is allocated on-demand, so
# the larger size costs only what active connections actually use.
#
# History: 4 MB → 16 MB (LMCache 70B blobs) → 64 MB (shared-KV-cache phase 1)
# → 256 MB (ATTEND.PREFIX.STORE long-N, 2026-05-02 W3.1 follow-up).
comptime CLIENT_BUF_SIZE = 256 * 1024 * 1024  # 256 MB
# kqueue loop: empty zero-timeout polls after the last event before it blocks
# for 1 ms again. An empty kevent is ~1 us, so this is tens of microseconds.
comptime KQ_SPIN_POLLS = 64


struct NetworkEngine:
    var server: TCPServer
    var kq: Int32
    var client_buffer_lens: Pointer[Int, MutUntrackedOrigin]
    var client_buffers: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    var config: PionConfig
    var fast_path: FastPathHandler
    var slow_path: SlowPathHandler
    var pending_changes: Pointer[KEvent, MutUntrackedOrigin]
    var pending_change_count: Int
    var worker_id: Int
    # gh #14: this worker's epoch slot, resolved ONCE at construction.
    # Resolving it per recv buffer meant three dependent loads
    # (slow_path -> shared_hnsw -> worker_epoch) on the dispatch path for
    # a value that never changes. Null when the shared view has no epoch
    # state, which is also the "nothing to do" fast check.
    var rcu_slot: Pointer[UInt64, MutUntrackedOrigin]
    # gh #259: set by _housekeeping_64tick once SIGTERM/SIGINT has latched and
    # this worker has flushed its WAL durably. Every event loop checks it right
    # after the housekeeping call and returns, which ends the worker thread;
    # main() then falls out of pthread_join and exits normally, so the atexit
    # breadcrumb still runs and the exit status is a clean 0.
    var shutting_down: Bool
    var rcu_epoch_ptr: Pointer[UInt64, MutUntrackedOrigin]
    var num_workers: Int
    # P4: per-fd local-affinity flag. When a connection is accepted on this worker's secondary
    # listen port (base_port + 1 + worker_id), all commands skip cross-worker routing.
    var local_affinity: Pointer[UInt8, MutUntrackedOrigin]
    # P4: secondary listen fd for this worker's affinity port (base_port + 2 + worker_id).
    var secondary_listen_fd: Int32
    # Binary protocol listen fd (shared across workers, base_port + 1).
    var binary_listen_fd: Int32
    # io_uring ring (Linux only; ring_fd=-1 on macOS — ring.setup() never called)
    var ring: Pointer[IOUring, MutUntrackedOrigin]
    # uring_recv_armed[fd]: 1 if a RECV SQE is currently submitted for this fd, 0 otherwise.
    # Prevents double-submitting RECV SQEs when both the RECV handler and SEND completion
    # try to arm the next read.
    var uring_recv_armed: Pointer[UInt8, MutUntrackedOrigin]
    # Multishot recv: provided buffer pool for IORING_RECV_MULTISHOT (kernel 6.0+).
    # 256 × 16KB buffers per worker. When multishot_active=False, falls back to per-fd buffers.
    var multishot_bufs: Pointer[UInt8, MutUntrackedOrigin]
    var multishot_active: Bool
    # Phase 5: Cluster state (shared, read-only after init)
    var cluster: Pointer[ClusterState, MutUntrackedOrigin]
    # N3: auto-failover tick counter (increments when primary is in FAIL state)
    var failover_ticks: Int
    # Active TTL sweep: ticks until next sweep call.
    var ttl_sweep_counter: Int
    # gh #261: RSS is over --maxmemory, per the housekeeping tick. While set,
    # every recv buffer skips the fast path and goes to the slow path, which
    # refuses the memory-growing commands and serves the rest. The check sits
    # HERE, at the call sites, and not inside process_data_plane: a single
    # branch there measurably cost MSET ~7% (interleaved A/B, 16 runs), the
    # same code-layout sensitivity gh #149 measured for one struct field.
    # The call sites read slow_path.fast_path_off, which is set while memory is
    # over the limit, a client monitors (#39) or one is subscribed (#42); then
    # slow_path.fast_path_ok(fd) decides per connection.
    var over_maxmemory: Bool
    # #49: writer is LAST, so a change in its size can never move the hot
    # fast_path/slow_path structs after it (gh #149 layout sensitivity). Its
    # cold state lives behind `writer.ctx` (see WriterCtx for the measured
    # reason), and the release-equivalence MSET A/B was taken with it here.
    var writer: ResponseWriter


    def __init__(
        out self,
        port: Int,
        keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
        config: PionConfig,
        hash_map_pool: Pointer[ObjectPool[SlabHashMap], MutUntrackedOrigin],
        skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin],
        list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin],
        ai_queue: Pointer[LockFreeRingBuffer, MutUntrackedOrigin],
        wal: Pointer[WAL, MutUntrackedOrigin],
        raft: Pointer[RaftNode, MutUntrackedOrigin],
        shared_hnsw: Pointer[SharedHNSWView, MutUntrackedOrigin],
        worker_id: Int = 0,
        num_workers: Int = 1,
        secondary_listen_fd: Int32 = Int32(-1),
        binary_listen_fd: Int32 = Int32(-1),
        cluster: Pointer[ClusterState, MutUntrackedOrigin] = null_ptr[ClusterState, MutUntrackedOrigin](),
        ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] = null_ptr[SlabHashMap, MutUntrackedOrigin](),
    ):
        self.server = TCPServer(port)
        self.kq = -1
        self.config = config
        self.worker_id = worker_id
        self.rcu_slot = null_ptr[UInt64, MutUntrackedOrigin]()
        self.shutting_down = False
        self.rcu_epoch_ptr = null_ptr[UInt64, MutUntrackedOrigin]()
        if is_not_null(shared_hnsw) and is_not_null(shared_hnsw[].worker_epoch) \
           and worker_id < shared_hnsw[].worker_epoch_slots:
            self.rcu_slot = shared_hnsw[].worker_epoch.unsafe_offset(worker_id * 8)
            self.rcu_epoch_ptr = shared_hnsw[].reclaim_epoch
        self.num_workers = num_workers


        self.client_buffer_lens = alloc[Int](65536)
        self.client_buffers = alloc[Pointer[UInt8, MutUntrackedOrigin]](65536)
        for i in range(65536):
            self.client_buffer_lens[unsafe_offset=i] = 0
            self.client_buffers[unsafe_offset=i] = null_ptr[UInt8, MutUntrackedOrigin]()

        # P4: local_affinity[fd] = 1 when connection was accepted on this worker's secondary port.
        self.local_affinity = alloc[UInt8](65536)
        unsafe_memset(self.local_affinity.unsafe_bitcast[UInt8](), 0, 65536)
        self.secondary_listen_fd = secondary_listen_fd
        self.binary_listen_fd = binary_listen_fd
        # io_uring: heap-allocated so ResponseWriter can hold a pointer to it.
        self.ring = alloc[IOUring](1)
        self.ring[unsafe_offset=0] = IOUring()
        self.uring_recv_armed = alloc[UInt8](65536)
        unsafe_memset(self.uring_recv_armed.unsafe_bitcast[UInt8](), 0, 65536)
        self.multishot_bufs = null_ptr[UInt8, MutUntrackedOrigin]()
        self.multishot_active = False

        self.cluster = cluster
        self.failover_ticks = 0
        self.ttl_sweep_counter = 0
        self.writer = ResponseWriter()
        self.fast_path = FastPathHandler(keyspace, hash_map_pool, skip_list_pool, list_pool, ai_queue, wal, raft, shared_hnsw,
                                         worker_id=worker_id, num_workers=num_workers,
                                         local_affinity=self.local_affinity,
                                         cluster=cluster,
                                         ttl_map=ttl_map)
        self.slow_path = SlowPathHandler(keyspace, hash_map_pool, skip_list_pool, list_pool, ai_queue, wal, raft, shared_hnsw, config=config, worker_id=worker_id, num_workers=num_workers, cluster=cluster, ttl_map=ttl_map)
        self.slow_path.local_affinity = self.local_affinity   # RESET clears READONLY (#39)
        self.slow_path.client_buffer_lens = self.client_buffer_lens   # #47: CLIENT LIST qbuf
        self.slow_path.client_buf_cap = CLIENT_BUF_SIZE
        # Wire fast_path's transaction pointers to slow_path's transaction state
        self.fast_path.tx_in_multi = self.slow_path.tx_state.in_multi
        self.fast_path.key_versions = self.slow_path.tx_state.key_versions
        # gh #100 (C2): share the per-fd auth flag and precompute the gate bool.
        self.fast_path.authed = self.slow_path.tx_state.authed
        self.fast_path.auth_required = config.server.requirepass.byte_length() > 0
        # gh #101: share the per-fd tenant binding; tenant_mode routes bound
        # fds to the slow path where the namespacing pre-pass runs.
        self.fast_path.tenant_ids = self.slow_path.tx_state.tenant_id
        self.fast_path.tenant_mode = self.slow_path.tenant_table.count > 0
        # --no-wal: skip WAL append on writes and WAL sync per tick (benchmark mode)
        if config.server.no_wal:
            self.fast_path.has_wal = False
        self.pending_changes = alloc[KEvent](64)
        self.pending_change_count = 0
        self.over_maxmemory = False

    # gh #85 (gh #48 / eae8975): KV_BUS routing, drain_bus_responses and the
    # per-fd P2 stall state (pending_remote[] / pending_slot[] / slot_full_fds[]
    # / pending_remote_count / bus_idle_count) are all gone. Nothing had written
    # a non-(-1) pending_remote since KV_BUS_ENABLED went permanently False, so
    # every stall branch was unreachable — and it was that dead branch, not a
    # real behavioral difference, that kept io_uring on its own dispatch copy.
    # Shared-nothing is the committed model.

    def run_server(mut self, mut hnsw: HNSWGraph, mut db_size: Int) raises:
        if CompilationTarget.is_linux():
            if self.config.server.use_xdp:
                self.run_server_xdp(hnsw, db_size)
            elif self.config.server.use_epoll:
                self.run_server_epoll(hnsw, db_size)
            else:
                # io_uring is default on Linux: better syscall batching at high concurrency.
                # Use --epoll for low-concurrency benchmarks (P=1 per-command).
                self.run_server_uring(hnsw, db_size)
        else:
            self.run_server_kqueue(hnsw, db_size)

    @always_inline
    def _close_fd_common(mut self, fd: Int32, ci: Int):
        """gh #85: cross-poller per-fd cleanup. The poller-specific deregister
        (`kevent_del_write` on kqueue / `epoll_ctl(EPOLL_CTL_DEL)` on epoll) is
        the caller's responsibility — those vary across loops. Everything else
        is identical: pubsub / tx / blocked-reader fd cleanup, close the
        socket, free the per-fd RECV + writer buffers, zero pending_offsets.
        Used by the epoll + kqueue inline close paths, and by io_uring's
        `_uring_finish_close` once nothing is in flight for the fd.

        Extracted from 3 near-identical copies — adding a new per-fd reset
        without updating every loop was the drift bug class the issue called
        out."""
        self.slow_path.pubsub.cleanup_fd(fd)
        self.slow_path.tx_state.cleanup_fd(fd)
        self.slow_path.blocked_readers.remove_fd(fd)
        self.slow_path.blocked_clients.remove_fd(fd)   # #38
        _ = self.slow_path.monitors.remove(fd)        # #39
        self.slow_path.clients.on_close(fd)          # #47 (its REPLY mode goes too)
        self.slow_path.update_dispatch_gate()          # #39, #42, #47
        self.slow_path.parked_waits.remove_fd(fd)   # gh #390
        self.server.close_client(fd)
        self.client_buffer_lens[unsafe_offset=ci] = 0
        if self.client_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
            self.client_buffers[unsafe_offset=ci].unsafe_free()
            self.client_buffers[unsafe_offset=ci] = null_ptr[UInt8, MutUntrackedOrigin]()
        if self.writer.ctx[].pending_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
            self.writer.ctx[].pending_buffers[unsafe_offset=ci].unsafe_free()
            self.writer.ctx[].pending_buffers[unsafe_offset=ci] = null_ptr[UInt8, MutUntrackedOrigin]()
        self.writer.ctx[].pending_offsets[unsafe_offset=ci] = 0
        self.writer.out_free(ci)                     # #49

    @always_inline
    def _set_expiry_clock(mut self):
        """#45: set the keyspace's lazy-expiry clock for the coming batch."""
        var ks = self.fast_path.keyspace
        var tm = self.fast_path.ttl_map
        if is_not_null(tm) and tm[].size > 0:
            ks[].clock_ns = _get_now_ns()
            var cl = self.slow_path.cluster
            # #47: and while CLIENT PAUSE holds writes, as Redis pauses expiry
            ks[].expire_hides_only = (is_not_null(cl) and cl[].enabled and cl[].is_replica) \
                                     or self.slow_path.clients.pause_until_ms != 0
        else:
            ks[].clock_ns = 0

    def _dispatch_recv_buffer(
        mut self,
        fd: Int32,
        client_idx: Int,
        stored_len: Int,
        n: Int,
        kq: Int32,
        mut hnsw: HNSWGraph,
        mut db_size: Int,
    ) raises:
        """gh #85: the single post-recv dispatch + drain body, shared by the
        kqueue, epoll and io_uring loops.

        Path:
          1. Binary protocol (0xCA5E framed) — drain every complete binary
             frame inline, no ResponseWriter, no fast-path involvement.
          2. RESP — fast path first, slow path fallback when the fast path
             returns consumed==0. Drain every complete frame: level-triggered
             pollers re-fire anyway, but draining inline saves a wait
             round-trip per pipelined batch.

        io_uring used to keep its own copy because "its drain shape is
        genuinely different — a single while-loop with break-on-zero rather
        than first-call-then-drain, plus distinct stall semantics" (the
        objection every earlier pass on this issue cited). That difference was
        entirely in the P2 cross-worker stall handling, which is dead code:
        `pending_remote[]` is memset to -1 at init and the only remaining
        writes set it back to -1 — nothing has parked an fd on the bus since
        gh #48 retired KV_BUS. With the stall branches gone the two shapes are
        provably the same loop, so io_uring shares this body and only keeps
        its own RECV re-arm afterwards (the genuinely poller-specific part)."""
        var cur_len = stored_len + n
        if n > 0:
            self.slow_path.clients.touch(fd)    # #47: CLIENT LIST idle
        # #47: a client that killed itself is closed once its reply is out;
        # what it sends meanwhile is dropped, as Redis drops it
        if self.slow_path.clients.close_after[unsafe_offset=client_idx] != 0:
            self.client_buffer_lens[unsafe_offset=client_idx] = 0
            return

        # gh #14 phase-2 RCU: announce that this worker is inside a dispatch
        # batch, and at which epoch. Bracketing HERE rather than inside
        # handle_ft_search is deliberate — that function has many return paths,
        # and a single missed exit would leave a worker permanently marked
        # busy, which the reclaimer would then wait on forever. Workers are
        # single-task (V16), so "inside a dispatch batch" is a complete and
        # unmissable enclosure of every borrowed-pointer read.
        #
        # The enter store is SEQUENTIAL, not RELEASE, and that distinction is
        # the whole correctness argument. A release store orders PRIOR writes;
        # it does not stop a LATER load from being hoisted above it. With a
        # release store the CPU may reorder into:
        #     load ready_atomic (=1, borrow the pointers)
        #     store worker_epoch[w] = busy
        # and between those two the reclaimer reads the slot as idle and frees
        # memory this worker is already using — reintroducing exactly the UAF
        # the epoch is meant to close. SEQUENTIAL puts a full barrier there.
        #
        # Cost is one barrier plus two stores per RECV BUFFER, not per command,
        # so a pipelined batch of 30 pays it once. The epoch load is ACQUIRE so
        # a batch starting after a reclaimer's bump observes it.
        var _rcu_slot = self.rcu_slot
        if is_not_null(_rcu_slot):
            var _rcu_ep = UInt64(1)
            if is_not_null(self.rcu_epoch_ptr):
                _rcu_ep = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                    self.rcu_epoch_ptr, UInt64(0))
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.SEQUENTIAL](
                _rcu_slot, (_rcu_ep << 1) | UInt64(1))

        # #45: the batch's clock for lazy expiry, Redis's command time
        # snapshot: one read per recv buffer while any key has a TTL, 0 (off)
        # otherwise. A replica only hides an expired key (it refuses writes,
        # and the primary's DEL removes it), as a Redis replica does.
        self._set_expiry_clock()

        # Binary protocol connections: route to process_binary_request()
        # instead of the RESP fast_path/slow_path. Binary handler sends
        # responses directly via server.send(), no ResponseWriter needed.
        if self.local_affinity[unsafe_offset=client_idx] == UInt8(2):
            while cur_len > 0:
                var consumed_b = self.slow_path.process_binary_request(
                    self.client_buffers[unsafe_offset=client_idx], cur_len, self.server, fd,
                )
                if consumed_b == 0:
                    break  # incomplete frame — wait for more data
                var leftover_b = cur_len - consumed_b
                if leftover_b > 0:
                    _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                        self.client_buffers[unsafe_offset=client_idx].unsafe_bitcast[NoneType](),
                        (self.client_buffers[unsafe_offset=client_idx].unsafe_offset(consumed_b)).unsafe_bitcast[NoneType](),
                        leftover_b,
                    )
                cur_len = leftover_b
            self.client_buffer_lens[unsafe_offset=client_idx] = cur_len
            if is_not_null(_rcu_slot):    # gh #14: clear on the early return too
                Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](_rcu_slot, UInt64(0))
            return

        # gh #172: stamp the connection's wire protocol onto the per-worker
        # writer once per recv buffer, not once per command — every command in
        # this batch belongs to the same fd, and a `HELLO 3` inside the batch
        # updates `writer.proto` itself so its own reply and everything after it
        # in the same batch is already RESP3.
        self.writer.proto = self.slow_path.tx_state.resp_proto[unsafe_offset=client_idx]
        # #49: and the connection itself, which takes the buffer when it fills
        self.writer.ctx[].cur_fd = fd

        while cur_len > 0:
            # gh #390: a connection whose WAIT is parked runs nothing more
            # until the WAIT is answered; its bytes wait in the buffer.
            if self.slow_path.parked_waits.any() and self.slow_path.parked_waits.is_parked(client_idx):
                break
            var consumed = 0
            if not self.slow_path.fast_path_off or self.slow_path.fast_path_ok(client_idx):   # gh #261, #39, #42
                consumed = self.fast_path.process_data_plane(
                    fd, self.client_buffers[unsafe_offset=client_idx], cur_len,
                    self.writer, self.server, kq,
                    hnsw, db_size,
                )
            if consumed == 0:
                consumed = self.slow_path.process_slow_path(
                    self.client_buffers[unsafe_offset=client_idx], cur_len, fd,
                    self.writer, self.server, kq,
                    hnsw, db_size,
                    self.config,
                )
            if consumed == 0:
                break  # incomplete frame — wait for more data
            var leftover = cur_len - consumed
            if leftover > 0:
                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                    self.client_buffers[unsafe_offset=client_idx].unsafe_bitcast[NoneType](),
                    (self.client_buffers[unsafe_offset=client_idx].unsafe_offset(consumed)).unsafe_bitcast[NoneType](),
                    leftover,
                )
            cur_len = leftover
        # The per-worker response buffer must never carry bytes across an fd
        # boundary. The drain loop above exits on an incomplete frame, and a
        # handler that consumed >0 without flushing leaves its replies queued
        # in `writer.buffer` — which is shared by every fd this worker serves.
        # The next fd's flush would then deliver those bytes to the WRONG
        # connection: the reader that gets them never blocks and runs to
        # completion, while the connection that earned them waits forever for
        # a reply it will never see (concurrent pipelined writes deadlocked
        # 3 of 4 connections; a single connection never tripped it because
        # there was no other fd to mis-deliver to). Drain to THIS fd here.
        if self.writer.offset > 0 or self.writer.ctx[].queued:   # #49: queued, buffer empty
            self.writer.flush_response(fd, self.server, kq)
        # gh #394: free the aggregates this batch overwrote — after the flush,
        # since a reply can borrow from the value a later command replaced.
        # One load and a compare per recv buffer when there are none.
        if len(self.fast_path.keyspace[].graveyard[]) > 0:
            free_graveyard(self.fast_path.keyspace)
        # #46: record the vector slots this batch killed (freeing a hash above,
        # or a field write, kills its slot), so a restart keeps them dead
        if is_not_null(self.slow_path.vec_tomb) and len(self.slow_path.vec_tomb[].pending) > 0:
            log_dead_slots(self.slow_path.shared_hnsw, self.slow_path.vec_tomb, self.slow_path.dispatcher)
        self.client_buffer_lens[unsafe_offset=client_idx] = cur_len
        # gh #14: leave the RCU critical section. Placed after the flush, not
        # before it — a handler's reply can still reference borrowed memory
        # while it sits in the writer buffer.
        if is_not_null(_rcu_slot):
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](_rcu_slot, UInt64(0))

    def _service_parked_waits(mut self, kq: Int32, mut hnsw: HNSWGraph, mut db_size: Int,
                              uring_group: Int = -1) raises:
        """gh #390: answer every parked WAIT whose replicas have ACKed or
        whose timeout has passed, then run what its client pipelined behind
        it. Called from every event loop's tick while any WAIT is parked; the
        loops wake at least every ~1-8 ms, which bounds how late a timeout
        fires.

        `uring_group` >= 0 on the io_uring loop: a parked fd has no RECV armed
        there (a fixed-address RECV would land behind the bytes this drain
        moves), so it is re-armed here once the fd is running again."""
        var pw = Pointer(to=self.slow_path.parked_waits)
        var h = null_ptr[NoneType, MutUntrackedOrigin]()
        if is_not_null(self.cluster):
            h = self.cluster[].repl_primary_handle
        var now = _get_now_ns()
        var k = 0
        while k < pw[].count():
            var w = pw[].entries[k]
            var acked = 0
            if is_not_null(h):
                acked = Int(external_call["pion_repl_primary_acked_count", Int32](h, w.target))
            if w.unblock == 0 and acked < w.num_req and (w.deadline_ns == 0 or now < w.deadline_ns):
                k += 1
                continue
            pw[].unpark_at(k)          # entry k is now a different one: no k += 1
            var fd = w.fd
            var ci = Int(fd)
            self.writer.proto = self.slow_path.tx_state.resp_proto[unsafe_offset=ci]
            self.writer.ctx[].cur_fd = Int32(ci)   # #49
            if w.unblock == UNBLOCK_ERROR:          # #47: CLIENT UNBLOCK id ERROR
                self.writer.append_error_response(UNBLOCKED_ERROR)
            else:
                self.writer.append_int_response(Int64(acked))
            self.writer.flush_response(fd, self.server, kq)
            var stored = self.client_buffer_lens[unsafe_offset=ci]
            if stored > 0 and self.client_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
                self._dispatch_recv_buffer(fd, ci, stored, 0, kq, hnsw, db_size)
            if uring_group >= 0 and self.uring_recv_armed[unsafe_offset=ci] == 0 \
               and not pw[].is_parked(ci) \
               and self.client_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
                self._uring_arm_recv(fd, ci, UInt16(uring_group))

    def _service_blocked_readers(mut self, kq: Int32, mut hnsw: HNSWGraph, mut db_size: Int,
                                 uring_group: Int = -1) raises:
        """Answer every parked XREAD BLOCK whose stream got data (an
        XADD marked it ready) or whose deadline passed, then run what its
        client pipelined behind it, as _service_parked_waits does for WAIT.
        The reply goes through the writer like any other, in order with the
        replies the connection is owed."""
        var reg = Pointer(to=self.slow_path.blocked_readers)
        if reg[]._count() == 0:
            return
        var now_ms = Int64(_get_now_ns() // 1_000_000)
        var k = 0
        while k < reg[]._count():
            var fd = reg[].readers[k].fd
            # #47: CLIENT UNBLOCK answers it as its timeout would, or with an
            # error, whatever its streams hold by now
            var ub = reg[].readers[k].unblock
            var timed_out = ub != 0 or (reg[].readers[k].timeout_ms > 0 and now_ms >= reg[].readers[k].timeout_ms)
            if not reg[].readers[k].ready and not timed_out:
                k += 1
                continue
            var ci = Int(fd)
            self.writer.proto = self.slow_path.tx_state.resp_proto[unsafe_offset=ci]
            self.writer.ctx[].cur_fd = Int32(ci)   # #49
            var wrote = 0
            if ub == 0:
                wrote = write_xread_reply(self.writer, self.slow_path.keyspace,
                                          reg[].readers[k].keys, reg[].readers[k].after_ms,
                                          reg[].readers[k].after_seq, reg[].readers[k].count_limit)
            if wrote == 0:
                if not timed_out:
                    # Woken, but the new entries are gone again (XDEL, XTRIM).
                    reg[].readers[k].ready = False
                    k += 1
                    continue
                if ub == UNBLOCK_ERROR:
                    self.writer.append_error_response(UNBLOCKED_ERROR)
                else:
                    self.writer.append_null_array_response()
            reg[].remove_at(k)            # entry k is now a different one: no k += 1
            self.slow_path.parked_waits.unpark_fd(fd)
            self.writer.flush_response(fd, self.server, kq)
            var stored = self.client_buffer_lens[unsafe_offset=ci]
            if stored > 0 and self.client_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
                self._dispatch_recv_buffer(fd, ci, stored, 0, kq, hnsw, db_size)
            if uring_group >= 0 and self.uring_recv_armed[unsafe_offset=ci] == 0 \
               and not self.slow_path.parked_waits.is_parked(ci) \
               and self.client_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
                self._uring_arm_recv(fd, ci, UInt16(uring_group))

    def _service_blocked_clients(mut self, kq: Int32, mut hnsw: HNSWGraph, mut db_size: Int,
                                 uring_group: Int = -1) raises:
        """#38: wake parked BLPOP & co., oldest first. A client whose key now
        holds what it pops, or whose timeout has passed, has its command run
        again through the slow path, unable to block: served now, or answered
        with the timeout's nil (the command's own reply either way). Then
        whatever it pipelined behind it runs, as for a woken XREAD."""
        var reg = Pointer(to=self.slow_path.blocked_clients)
        if reg[]._count() == 0:
            return
        var now_ms = Int64(_get_now_ns() // 1_000_000)
        var k = 0
        while k < reg[]._count():
            var ub = reg[].clients[k].unblock
            var dl = reg[].clients[k].deadline_ms
            var timed_out = ub != 0 or (dl > 0 and now_ms >= dl)
            var ready = ub == 0 and blocked_client_ready(self.slow_path.keyspace, reg[].clients[k])
            if not timed_out and not ready:
                k += 1
                continue
            var fd = reg[].clients[k].fd
            if not ready:
                # Its timeout passed, or CLIENT UNBLOCK (#47): the timeout's
                # nil, without running the command again. A key that now
                # holds another type would make it answer WRONGTYPE, where
                # Redis answers the timeout.
                var nil_bulk = reg[].clients[k].nil_bulk
                reg[].remove_at(k)                 # the next client is now at k
                self.slow_path.parked_waits.unpark_fd(fd)
                var tci = Int(fd)
                self.writer.proto = self.slow_path.tx_state.resp_proto[unsafe_offset=tci]
                self.writer.ctx[].cur_fd = Int32(tci)   # #49
                if ub == UNBLOCK_ERROR:
                    self.writer.append_error_response(UNBLOCKED_ERROR)
                elif nil_bulk:
                    self.writer.append_null_response()
                else:
                    self.writer.append_null_array_response()
                self.writer.flush_response(fd, self.server, kq)
                self._resume_unparked(fd, kq, hnsw, db_size, uring_group)
                continue
            # The frame goes to a heap buffer this function frees itself: a
            # List's last use is `unsafe_ptr()`, and Mojo destroys a value
            # right after its last use, so the parse below would read freed
            # memory (it did: the allocator reused the first bytes).
            var flen = len(reg[].clients[k].frame)
            var frame = alloc[UInt8](flen + 1)
            unsafe_memcpy(dest=frame, src=reg[].clients[k].frame.unsafe_ptr(), count=flen)
            reg[].remove_at(k)                 # the next client is now at k
            self.slow_path.parked_waits.unpark_fd(fd)
            var ci = Int(fd)
            self.writer.proto = self.slow_path.tx_state.resp_proto[unsafe_offset=ci]
            self.writer.ctx[].cur_fd = Int32(ci)   # #49
            var park = self.slow_path.can_park_wait
            self.slow_path.can_park_wait = False
            # MONITOR showed the command when it first ran (and blocked), as
            # Redis does; running it again is not a new command.
            self.slow_path.monitor_skip = True
            _ = self.slow_path.process_slow_path(frame, flen, fd, self.writer, self.server, kq,
                                                 hnsw, db_size, self.config)
            self.slow_path.monitor_skip = False
            frame.free()
            self.slow_path.can_park_wait = park
            self._resume_unparked(fd, kq, hnsw, db_size, uring_group)

    def _resume_unparked(mut self, fd: Int32, kq: Int32, mut hnsw: HNSWGraph, mut db_size: Int,
                         uring_group: Int) raises:
        """A parked client was answered: run what it pipelined behind the
        command that parked it, and on io_uring arm its receive again."""
        var ci = Int(fd)
        var stored = self.client_buffer_lens[unsafe_offset=ci]
        if stored > 0 and self.client_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
            self._dispatch_recv_buffer(fd, ci, stored, 0, kq, hnsw, db_size)
        if uring_group >= 0 and self.uring_recv_armed[unsafe_offset=ci] == 0 \
           and not self.slow_path.parked_waits.is_parked(ci) \
           and self.client_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
            self._uring_arm_recv(fd, ci, UInt16(uring_group))

    @always_inline
    def _close_after_reply(mut self, fd: Int32):
        """#47: a client that killed itself, once its pending reply is out
        (kqueue / epoll write event): shut down, and its loop closes it."""
        var ci = Int(fd)
        if self.slow_path.clients.close_after[unsafe_offset=ci] != 0 \
           and not self.writer.owes(ci):
            _ = external_call["pion_kill_fd", Int32](fd)

    def _service_pause(mut self, kq: Int32, mut hnsw: HNSWGraph, mut db_size: Int,
                       uring_group: Int = -1) raises:
        """#47 CLIENT PAUSE: when the pause is over (its timeout passed, or
        UNPAUSE) or changed (another PAUSE), the connections it held run their
        commands, as Redis's unblockPostponedClients; under a pause that still
        applies to them they are held again. A held connection CLIENT KILL let
        go runs (and so reads its end of input) at once."""
        var reg = Pointer(to=self.slow_path.clients)
        var ended = reg[].pause_until_ms != 0 and external_call["pion_unix_ms", Int64]() >= reg[].pause_until_ms
        if ended:
            reg[].pause_until_ms = 0
        var go = List[Int32]()
        if ended or reg[].pause_changed:
            reg[].pause_changed = False
            for k in range(len(reg[].paused_fds)):
                var f = reg[].paused_fds[k]
                reg[].postponed[Int(f)] = 0
                go.append(f)
            reg[].paused_fds.clear()
        for k in range(len(reg[].released)):
            go.append(reg[].released[k])
        reg[].released.clear()
        self.slow_path.update_dispatch_gate()
        for k in range(len(go)):
            self.slow_path.parked_waits.unpark_fd(go[k])
            self._resume_unparked(go[k], kq, hnsw, db_size, uring_group)

    def _replica_drain(mut self, cl: Pointer[ClusterState, MutUntrackedOrigin]):
        """gh #390: drain the replica ring, apply every whole record, carry a
        split one into the next drain, and report what was applied — which is
        what the replica ACKs, so WAIT counts it only once it is readable."""
        from src.network.replication import apply_wal_entries
        var carry = cl[].repl_carry
        var cap = cl[].repl_drain_cap
        var buf = cl[].repl_drain_buf
        var n = Int(external_call["pion_repl_replica_drain", Int32](
            cl[].repl_replica_handle, buf.unsafe_offset(carry), Int32(cap - carry)))
        var total = carry + n
        if total == 0:
            return
        # #45: the primary's records apply to what it had: no lazy expiry here
        # (a key this replica's clock calls expired may still be live there).
        var _clk = self.fast_path.keyspace[].clock_ns
        self.fast_path.keyspace[].clock_ns = 0
        var used = apply_wal_entries(self.fast_path.keyspace, buf, total, self.fast_path.ttl_map)
        self.fast_path.keyspace[].clock_ns = _clk
        if used > 0:
            free_graveyard(self.fast_path.keyspace)   # gh #394: a replicated SET over an aggregate
            external_call["pion_repl_replica_applied", NoneType](
                cl[].repl_replica_handle, UInt64(used))
        var left = total - used
        if left > 0 and used > 0:
            _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                buf.unsafe_bitcast[NoneType](),
                buf.unsafe_offset(used).unsafe_bitcast[NoneType](), left)
        if left == cap:
            # One record bigger than the whole buffer: grow it, or it never fits.
            var bigger = alloc[UInt8](cap * 2)
            unsafe_memcpy(dest=bigger, src=buf, count=left)
            buf.unsafe_free()
            cl[].repl_drain_buf = bigger
            cl[].repl_drain_cap = cap * 2
        cl[].repl_carry = left

    def _primary_service(mut self, cl: Pointer[ClusterState, MutUntrackedOrigin]) raises:
        """gh #390: the replicator thread cannot read the keyspace, so a
        replica needing a FULLRESYNC waits for THIS worker to serialize it.
        The records are the snapshot writer's (the replica already decodes
        every one of them), taken between batches, so they are exactly the
        state at the WAL offset handed over with them. Written to a scratch
        file, never the persisted snapshot: replacing that without a WAL
        checkpoint would replay the old log over a newer image."""
        var h = cl[].repl_primary_handle
        var wal = self.fast_path.wal
        if is_null(wal):
            return
        if is_null(wal[].repl_handle):
            wal[].repl_handle = h       # rotation/checkpoint must detach it (see WAL)
        if external_call["pion_repl_primary_snapshot_requested", Int32](h) == 0:
            return
        wal[].publish_header()
        var tail = wal[].tail_offset
        var path = "pion.replsnap." + String(self.worker_id)
        _ = self.slow_path.snapshot_engine.take_snapshot(
            self.fast_path.keyspace, self.worker_id, UInt64(0),
            null_ptr[BlobStore, MutUntrackedOrigin](), self.fast_path.ttl_map, path)
        var fd = external_call["pion_open_rdonly", Int32](path.as_c_string_slice())
        if fd < 0:
            print("Replication: FULLRESYNC snapshot could not be read back from " + path)
            return
        var size = Int(external_call["pion_file_size", Int64](fd))
        var data = alloc[UInt8](max(size, 1))
        var got = 0
        while got < size:
            var r = Int(external_call["pion_read", Int64](fd, data.unsafe_offset(got), size - got))
            if r <= 0:
                break
            got += r
        _ = external_call["close", Int32](fd)
        _ = external_call["unlink", Int32](path.as_c_string_slice())
        if got == size and size >= SNAP_HDR_LEN:
            external_call["pion_repl_primary_provide_snapshot", NoneType](
                h, data.unsafe_offset(SNAP_HDR_LEN), UInt64(size - SNAP_HDR_LEN), tail)
            print("Replication: FULLRESYNC snapshot " + String(size - SNAP_HDR_LEN)
                  + " bytes at WAL offset " + String(tail))
        data.unsafe_free()

    def _housekeeping_64tick(mut self, mut hnsw: HNSWGraph) raises:
        """gh #85: unified per-64-tick housekeeping shared by all four event
        loops (xdp / io_uring / epoll / kqueue). Caller gates on `& 0x3F == 0`.

        Before this extraction each loop carried its own copy of the periodic
        block and they had drifted — the exact bug class the issue calls out
        ("a periodic task added to one loop silently doesn't run on the others"):
          - C1.2 primary auto-start lived ONLY on the xdp loop → a cluster
            master served by kqueue/epoll/io_uring never auto-started its
            primary replicator.
          - N3 auto-failover lived ONLY on epoll/kqueue → a replica served by
            xdp/io_uring never auto-promoted when its primary FAILed.
          - reset_p3_tick (since deleted with the P3 staging machinery, gh #207)
            and the dormant sharded-ingest block were present on some loops,
            absent on others.
        Only WAL sync + replica-drain (the durability path) were shared by all
        four. This body is the UNION of all four. Every task is guarded by a
        cluster/replica/master predicate that no-ops unless clustering is
        configured, so folding the union into every loop is behavior-preserving
        where a loop already ran the task and a latent-bug fix where it did not.
        Nothing here touches the poller (`kq`/ring), so no per-poller parameter
        is needed."""
        # gh #14 RCU self-heal. `_dispatch_recv_buffer` is `raises`; an
        # exception escaping it would skip the exit store and leave this
        # worker's epoch slot marked busy. That is not a correctness problem —
        # the reclaimer's 100 ms ceiling falls back to the phase-1 sleep, so
        # the worst case is old behaviour — but it would be a permanent,
        # hard-to-attribute latency regression on every later DROPINDEX.
        #
        # Housekeeping runs in the event loop, which by construction is NOT
        # inside a dispatch batch, so clearing here can never erase a
        # legitimately-busy marker. Any stuck slot is released within 64 ticks
        # by the worker that owns it.
        if is_not_null(self.rcu_slot):
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                self.rcu_slot, UInt64(0))

        from src.network.replication import apply_wal_entries

        # gh #138: liveness heartbeat (no-op unless --crash-log/--status-file).
        external_call["pion_crash_heartbeat", NoneType](Int32(self.worker_id))

        # gh #261: --maxmemory. One load and a return when no limit is set; with
        # one, at most one worker per 100 ms samples RSS. The flag only routes
        # dispatch to the slow path, which re-confirms before refusing anything.
        var _oom = external_call["pion_maxmemory_check", Int32]() != 0
        self.over_maxmemory = _oom
        self.slow_path.over_maxmemory = _oom
        self.slow_path.update_dispatch_gate()

        # gh #259: graceful shutdown. SIGTERM/SIGINT no longer kill us where
        # they land — they latch, and the drain happens HERE, on the event loop,
        # where calling msync is legal (a signal handler may not).
        #
        # This sits in the shared 64-tick helper for exactly the reason gh #85
        # created it: a drain added to one loop and not the other three is the
        # bug that helper exists to prevent, and "durability only works under
        # kqueue" would be a miserable one to find.
        #
        # Worst-case latency to notice is 64 ticks, bounded well under the
        # SIGALRM grace period. Each worker flushes its OWN WAL — they are
        # shared-nothing, so there is nothing to coordinate.
        # #45: log the DELs of keys that expired with no write behind them (a
        # write logs them itself, ahead of its own record).
        if is_not_null(self.fast_path.wal):
            self.fast_path.wal[].log_expired()
        # #46: and the vector slots the sweep's frees killed
        if is_not_null(self.slow_path.vec_tomb) and len(self.slow_path.vec_tomb[].pending) > 0:
            log_dead_slots(self.slow_path.shared_hnsw, self.slow_path.vec_tomb, self.slow_path.dispatcher)
        # #47: a SLOWLOG threshold set on another worker reaches this one
        if self.slow_path.slowlog_thr != external_call["pion_slowlog_get_slower_than", Int64]():
            self.slow_path.update_dispatch_gate()

        if not self.shutting_down:
            if external_call["pion_shutdown_requested", Int32]() != 0:
                external_call["pion_shutdown_begin_drain", NoneType]()   # #47: past ABORT now
                if is_not_null(self.fast_path.wal):
                    self.fast_path.wal[].sync_durable()
                self.shutting_down = True

        # Cluster: replica WAL drain + apply (non-blocking).
        var _cl = self.slow_path.cluster
        if is_not_null(_cl) and is_not_null(_cl[].repl_replica_handle) and is_not_null(_cl[].repl_drain_buf):
            self._replica_drain(_cl)
        # Cluster: primary side — serve a replica's FULLRESYNC snapshot request.
        if is_not_null(_cl) and is_not_null(_cl[].repl_primary_handle):
            self._primary_service(_cl)

        # WAL group commit: MS_ASYNC msync every 64 ticks (~64ms durability window).
        if self.fast_path.has_wal:
            self.fast_path.wal[].sync()
        # gh #163: the blob arena rides the same group commit. Its pages are
        # file-backed, so this schedules writeback rather than doing it.
        if is_not_null(self.fast_path.blobs):
            self.fast_path.blobs[].sync()

        # N3: Auto-failover — replica detects primary FAIL and auto-promotes.
        # Threshold: 5000 ticks ≈ 5s; failover_ticks increments by 64 per check.
        var _cl2 = self.slow_path.cluster
        if is_not_null(_cl2) and _cl2[].enabled and _cl2[].is_replica and self.worker_id == 0:
            var _ppidx = _cl2[].primary_peer_idx
            if _ppidx >= 0 and _ppidx < 16 and _cl2[].peer_health[_ppidx] == 2:  # FAIL
                self.failover_ticks += 64  # account for gated check interval
                if self.failover_ticks >= 5000:
                    var quorum_ok = True
                    if _cl2[].peer_count >= 2:
                        _ = 1
                        for _pi in range(_cl2[].peer_count):
                            if _pi == _ppidx:
                                continue
                        quorum_ok = True

                    if quorum_ok:
                        print("N3 AUTO-FAILOVER: primary unreachable for " + String(self.failover_ticks) + " ticks, promoting to primary")
                        if is_not_null(_cl2[].repl_replica_handle):
                            external_call["pion_repl_replica_stop", NoneType](_cl2[].repl_replica_handle)
                            _cl2[].repl_replica_handle = null_ptr[NoneType, MutUntrackedOrigin]()
                        if is_not_null(_cl2[].repl_drain_buf):
                            _cl2[].repl_drain_buf.unsafe_free()
                            _cl2[].repl_drain_buf = null_ptr[UInt8, MutUntrackedOrigin]()
                        _cl2[].is_replica = False
                        _cl2[].primary_peer_idx = -1
                        _cl2[].cluster_epoch += 1
                        if is_not_null(_cl2[].wal_ptr) and is_null(_cl2[].repl_primary_handle):
                            var _rport = _cl2[].server_port + REPL_PORT_OFFSET
                            var _blk = external_call["pion_repl_primary_create",
                                                       Pointer[NoneType, MutUntrackedOrigin]](
                                _cl2[].wal_ptr.unsafe_offset(64),
                                _cl2[].wal_ptr.unsafe_bitcast[UInt64]().unsafe_offset(1),
                                Int32(_rport),
                            )
                            if is_not_null(_blk):
                                _ = external_call["pion_repl_primary_start", Int32](_blk)
                                _cl2[].repl_primary_handle = _blk
                        _cl2[].save_topology("pion-nodes.conf")
                        self.failover_ticks = 0
            else:
                self.failover_ticks = 0

        # C1.2: cluster master auto-starts its primary replicator (worker 0).
        var _cl3 = self.slow_path.cluster
        if is_not_null(_cl3) and _cl3[].enabled and not _cl3[].is_replica and self.worker_id == 0:
            if is_not_null(_cl3[].wal_ptr) and is_null(_cl3[].repl_primary_handle):
                var _rport = _cl3[].server_port + REPL_PORT_OFFSET
                var _blk = external_call["pion_repl_primary_create",
                                           Pointer[NoneType, MutUntrackedOrigin]](
                    _cl3[].wal_ptr.unsafe_offset(64),
                    _cl3[].wal_ptr.unsafe_bitcast[UInt64]().unsafe_offset(1),
                    Int32(_rport),
                )
                if is_not_null(_blk):
                    # C1.2: Set replication ID before starting
                    _cl3[].generate_repl_id()
                    var rid = String("")
                    for _ri in range(40):
                        rid += chr(Int(_cl3[].repl_id[_ri]))
                    var rid_cstr = rid + "\0"
                    external_call["pion_repl_primary_set_repl_id", NoneType](
                        _blk, rid_cstr.unsafe_ptr().unsafe_bitcast[UInt8]()
                    )
                    _ = external_call["pion_repl_primary_start", Int32](_blk)
                    _cl3[].repl_primary_handle = _blk
                    print("C1.2: Primary replicator auto-started on port " + String(_rport))

        # gh #5 / #87.2: sharded-build + shard-query-bus serve. Dormant —
        # `num_shards` is pinned at 1; the whole block is comptime-absent unless
        # HNSW_SHARDED_INGEST_ENABLED (False). dbg_counters kept for the eventual
        # sharded re-enable (io_uring carried them; they were the only loop that did).
        comptime if HNSW_SHARDED_INGEST_ENABLED:
            var _sh = self.slow_path.shared_hnsw
            if is_not_null(_sh) and _sh[].num_shards > 1:
                var _n_sh = _sh[].num_shards
                if is_not_null(_sh[].optimize_trigger) and is_not_null(_sh[].shard_ready):
                    var _trig = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](_sh[].optimize_trigger, UInt64(0))
                    if is_not_null(_sh[].dbg_counters):
                        _sh[].dbg_counters[unsafe_offset=self.worker_id * 8] += 1
                    if _trig > 0:
                        if is_not_null(_sh[].dbg_counters):
                            _sh[].dbg_counters[unsafe_offset=self.worker_id * 8 + 1] += 1
                        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](_sh[].shard_ready.unsafe_offset(self.worker_id * 8), UInt64(0)) == 0:
                            try:
                                hnsw.build_index_from_shared(_sh, self.worker_id, _n_sh)
                                hnsw.index_ready = True
                                # Copy index metadata from shared view so FT.SEARCH works on this worker
                                hnsw.index_name_len = _sh[].index_name_len
                                for _ni in range(_sh[].index_name_len):
                                    hnsw.index_name[_ni] = _sh[].index_name[_ni]
                                hnsw.dim = _sh[].pre_dim  # blob parsing uses hnsw.dim * 4
                                Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                                    _sh[].shard_ready.unsafe_offset(self.worker_id * 8), UInt64(1))
                                if is_not_null(_sh[].dbg_counters):
                                    _sh[].dbg_counters[unsafe_offset=self.worker_id * 8 + 2] += 1
                            except:
                                if is_not_null(_sh[].dbg_counters):
                                    _sh[].dbg_counters[unsafe_offset=self.worker_id * 8 + 3] += 1
                if hnsw.index_ready and is_not_null(_sh[].shard_bus):
                    var _sbus = _sh[].shard_bus
                    for _c in range(_n_sh):
                        if _c == self.worker_id: continue
                        var _qidx = _c * _n_sh + self.worker_id
                        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](_sbus[].query_ready.unsafe_offset(_qidx), UInt64(0)) == 0: continue
                        var _slot = _sbus[].query_slots[unsafe_offset=_qidx]
                        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](_sbus[].query_ready.unsafe_offset(_qidx), UInt64(0))
                        self.slow_path.scratch_dists.clear()
                        try:
                            var _sids = hnsw.search_fp32_scored(_slot.query_fp32, Int(_slot.k), self.slow_path.scratch_dists, Int(_slot.ef))
                            _sbus[].write_result(_c, self.worker_id, _sids, self.slow_path.scratch_dists, _slot.seq)
                        except:
                            _sbus[].result_counts[unsafe_offset=_c * _n_sh + self.worker_id] = 0
                            _sbus[].result_seq[unsafe_offset=_c * _n_sh + self.worker_id] = _slot.seq
                            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](_sbus[].result_ready.unsafe_offset(_c * _n_sh).unsafe_offset(self.worker_id), UInt64(1))

    def run_server_xdp(mut self, mut hnsw: HNSWGraph, mut db_size: Int) raises:
        """XDP/AF_XDP event loop — zero-copy kernel bypass path.

        Bypasses the entire Linux TCP/IP stack. Packets are intercepted at the NIC driver
        by an XDP BPF program, delivered via AF_XDP to shared UMEM, and processed directly
        in this event loop. TCP connection management is handled by a minimal TCP-Lite state
        machine (SYN/ACK handshake, data transfer, FIN/RST teardown).

        Falls back to io_uring if XDP setup fails (e.g., no CAP_NET_ADMIN, unsupported NIC).

        Target: 5M+ QPS single worker (vs 3.3M with io_uring).
        """
        from src.network.replication import apply_wal_entries
        var xdp = XDPEngine()
        var xdp_ok: Bool
        if self.config.server.xdp_shared_xskmap_fd >= 0 and self.config.server.xdp_shared_bpf_fd >= 0:
            # Multi-worker mode: use shared BPF/XSKMAP created before parallelize
            xdp_ok = xdp.setup_worker(
                self.config.server.xdp_interface, self.worker_id,
                UInt16(self.config.server.port),
                self.config.server.xdp_shared_xskmap_fd,
                self.config.server.xdp_shared_bpf_fd,
            )
        else:
            # Single-worker mode: create everything ourselves
            xdp_ok = xdp.setup(self.config.server.xdp_interface, self.worker_id, UInt16(self.config.server.port))
        if not xdp_ok:
            print("XDP setup failed (requires Linux 5.4+, CAP_NET_ADMIN, AF_XDP-capable NIC)")
            print("Falling back to io_uring...")
            self.run_server_uring(hnsw, db_size)
            return

        # Also start a TCP listener for non-XDP clients (e.g., loopback connections,
        # redis-cli, health checks). Loopback bypasses the NIC so XDP never sees it.
        # We poll this listener every tick (non-blocking accept + recv/send).
        if self.server.fd < 0 and not self.server.listen():
            pass  # non-fatal — XDP can work without TCP fallback
        var tcp_listen_fd = self.server.fd
        var tcp_active_fds = alloc[Int32](256)  # up to 256 TCP loopback clients
        var tcp_active_count = 0

        var kq = Int32(-1)
        var my_tid = external_call["pthread_self", UInt64]()
        print("--- Pion XDP Engine Active --- worker=" + String(self.worker_id)
              + " tid=" + String(my_tid)
              + " iface=" + self.config.server.xdp_interface
              + " tcp_fallback=" + String(Int(tcp_listen_fd)))

        # Pre-allocate response buffer for XDP path (same as ResponseWriter's 4MB buffer)
        var xdp_response_buf = alloc[UInt8](4194304)
        var xdp_response_len = 0

        while True:
            # ── Phase 1: Poll AF_XDP RX ring for incoming frames ──
            var rx_count = xdp.poll_rx()

            for fi in range(rx_count):
                var addr = xdp.rx_addrs[unsafe_offset=fi]
                var flen = Int(xdp.rx_lens[unsafe_offset=fi])
                var frame = xdp.frame_ptr(addr)

                # Parse TCP header
                var info = xdp.extract_tcp(frame, flen)
                if info.payload_len < 0:
                    # Not a valid TCP frame — release and skip
                    xdp.release_rx_frame(addr)
                    continue

                # ── TCP-Lite state machine ──
                var tcp_flags = info.flags

                if (tcp_flags & TCP_RST) != 0:
                    # RST: immediately tear down connection
                    xdp.handle_rst(info)
                    xdp.release_rx_frame(addr)
                    continue

                if (tcp_flags & TCP_SYN) != 0 and (tcp_flags & TCP_ACK) == 0:
                    # SYN (no ACK): new connection — send SYN+ACK
                    _ = xdp.handle_syn(frame, flen, info)
                    xdp.release_rx_frame(addr)
                    continue

                if (tcp_flags & TCP_FIN) != 0:
                    # FIN: connection teardown — send FIN+ACK
                    _ = xdp.handle_fin(frame, flen, info)
                    xdp.release_rx_frame(addr)
                    continue

                if info.payload_len == 0:
                    # Pure ACK (no data): update connection state
                    xdp.handle_ack(info)
                    xdp.release_rx_frame(addr)
                    continue

                # ── Data segment: extract RESP payload and process ──
                # The payload pointer points directly into the UMEM frame (zero-copy).
                # We feed it to the fast path processor, which writes the response
                # into the ResponseWriter's buffer.
                xdp.handle_ack(info)

                # Use a virtual fd based on source port for the fast_path/slow_path API.
                # This fd is never used for actual I/O — it's an index into client state.
                # We use source port + 32768 to avoid collision with real TCP fds.
                var virtual_fd = Int32(Int(info.sport) | 0x8000)
                var vci = Int(virtual_fd)

                # Allocate client buffer for this virtual connection if needed
                if vci < 65536:
                    if self.client_buffers[unsafe_offset=vci] == null_ptr[UInt8, MutUntrackedOrigin]():
                        self.client_buffers[unsafe_offset=vci] = alloc[UInt8](CLIENT_BUF_SIZE)
                        self.client_buffer_lens[unsafe_offset=vci] = 0

                    # Copy payload into client buffer (needed for partial frame accumulation)
                    var stored = self.client_buffer_lens[unsafe_offset=vci]
                    if stored + info.payload_len <= CLIENT_BUF_SIZE:
                        unsafe_memcpy(dest=self.client_buffers[unsafe_offset=vci].unsafe_offset(stored), src=info.payload_ptr, count=info.payload_len)
                        var total_len = stored + info.payload_len

                        # Process all complete RESP frames
                        var cur_len = total_len
                        # #49: the XDP lane sends `buffer` itself and has no
                        # per-connection queue: a full buffer stays the -ERR frame
                        self.writer.ctx[].cur_fd = -1
                        while cur_len > 0:
                            var consumed = 0
                            if not self.slow_path.fast_path_off or self.slow_path.fast_path_ok(vci):   # gh #261, #39, #42
                                consumed = self.fast_path.process_data_plane(
                                    virtual_fd, self.client_buffers[unsafe_offset=vci], cur_len,
                                    self.writer, self.server, kq,
                                    hnsw, db_size,
                                )
                            if consumed == 0:
                                consumed = self.slow_path.process_slow_path(
                                    self.client_buffers[unsafe_offset=vci], cur_len, virtual_fd,
                                    self.writer, self.server, kq,
                                    hnsw, db_size,
                                    self.config,
                                )
                            if consumed == 0:
                                break
                            var leftover = cur_len - consumed
                            if leftover > 0:
                                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                                    self.client_buffers[unsafe_offset=vci].unsafe_bitcast[NoneType](),
                                    (self.client_buffers[unsafe_offset=vci].unsafe_offset(consumed)).unsafe_bitcast[NoneType](),
                                    leftover,
                                )
                            cur_len = leftover
                        self.client_buffer_lens[unsafe_offset=vci] = cur_len

                        # Send response via XDP TX path (zero-copy)
                        var resp_offset = self.writer.offset
                        if resp_offset > 0:
                            _ = xdp.send_data_ack(
                                frame, flen, info,
                                self.writer.buffer, resp_offset
                            )
                            self.writer.offset = 0

                xdp.release_rx_frame(addr)

            # ── Phase 2: Kick TX ring if we submitted any responses ──
            if rx_count > 0:
                xdp.tx_kick()

            # ── Phase 3: Drain TX completion ring (reclaim sent frames) ──
            _ = xdp.drain_completion()

            # ── Phase 3b: TCP loopback fallback ──
            # Non-blocking accept() + recv/send for loopback clients (redis-cli, health checks).
            # Loopback traffic never hits the NIC, so XDP can't see it.
            if tcp_listen_fd >= 0:
                # Non-blocking accept
                var new_fd = self.server.accept_from(tcp_listen_fd)
                if new_fd >= 0:
                    # Set non-blocking + TCP_NODELAY
                    self.server.set_nonblocking(new_fd)
                    self.server.set_tcp_nodelay(new_fd)
                    if tcp_active_count < 256:
                        tcp_active_fds[unsafe_offset=tcp_active_count] = new_fd
                        self.slow_path.clients.on_accept(new_fd)   # #47
                        tcp_active_count += 1
                        # Allocate client buffer for this TCP fd
                        var ci = Int(new_fd)
                        if ci < 65536 and self.client_buffers[unsafe_offset=ci] == null_ptr[UInt8, MutUntrackedOrigin]():
                            self.client_buffers[unsafe_offset=ci] = alloc[UInt8](CLIENT_BUF_SIZE)
                            self.client_buffer_lens[unsafe_offset=ci] = 0
                    else:
                        self.server.close_client(new_fd)

                # Process active TCP connections (non-blocking recv + process + send)
                var ti = 0
                while ti < tcp_active_count:
                    var tfd = tcp_active_fds[unsafe_offset=ti]
                    var ci = Int(tfd)
                    if ci >= 65536 or self.client_buffers[unsafe_offset=ci] == null_ptr[UInt8, MutUntrackedOrigin]():
                        ti += 1
                        continue
                    var stored = self.client_buffer_lens[unsafe_offset=ci]
                    var n_read = self.server.recv(tfd, self.client_buffers[unsafe_offset=ci].unsafe_offset(stored), CLIENT_BUF_SIZE - stored)
                    if n_read > 0:
                        var cur_len = stored + n_read
                        self.writer.ctx[].cur_fd = -1     # #49: this lane sends `buffer` once, as above
                        while cur_len > 0:
                            var consumed = 0
                            if not self.slow_path.fast_path_off or self.slow_path.fast_path_ok(ci):   # gh #261, #39, #42
                                consumed = self.fast_path.process_data_plane(
                                    tfd, self.client_buffers[unsafe_offset=ci], cur_len,
                                    self.writer, self.server, kq, hnsw, db_size,
                                )
                            if consumed == 0:
                                consumed = self.slow_path.process_slow_path(
                                    self.client_buffers[unsafe_offset=ci], cur_len, tfd,
                                    self.writer, self.server, kq, hnsw, db_size, self.config,
                                )
                            if consumed == 0:
                                break
                            var leftover = cur_len - consumed
                            if leftover > 0:
                                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                                    self.client_buffers[unsafe_offset=ci].unsafe_bitcast[NoneType](),
                                    (self.client_buffers[unsafe_offset=ci].unsafe_offset(consumed)).unsafe_bitcast[NoneType](), leftover)
                            cur_len = leftover
                        self.client_buffer_lens[unsafe_offset=ci] = cur_len
                        # Send TCP response
                        var resp_off = self.writer.offset
                        if resp_off > 0:
                            _ = self.server.send(tfd, self.writer.buffer, resp_off)
                            self.writer.offset = 0
                        # #47: a client that killed itself, its reply now sent
                        if self.slow_path.clients.close_after[unsafe_offset=ci] != 0:
                            _ = external_call["pion_kill_fd", Int32](tfd)
                    elif n_read == 0:
                        # Client disconnected — close and remove from active list.
                        # #47: the per-connection cleanup every other loop does
                        # (subscriptions, MULTI, CLIENT state, buffers): this
                        # closed the socket and freed the buffer only.
                        self._close_fd_common(tfd, ci)
                        # Swap with last active fd
                        tcp_active_count -= 1
                        if ti < tcp_active_count:
                            tcp_active_fds[unsafe_offset=ti] = tcp_active_fds[unsafe_offset=tcp_active_count]
                        continue  # don't increment ti — re-check swapped element
                    # n_read < 0: EAGAIN/EWOULDBLOCK — no data available, skip
                    ti += 1

            # ── Phase 4: Housekeeping (same as io_uring/kqueue paths) ──

            # gh #85b: KV_BUS routing was removed here (shared-nothing, gh #48).

            # Cross-worker pub/sub broadcast drain (every tick for responsive delivery)
            if self.num_workers > 1:
                self.slow_path.drain_pubsub(self.writer, self.server, kq)   # #42

            # TTL sweep
            self.ttl_sweep_counter += 1
            if self.ttl_sweep_counter >= 100:
                self.ttl_sweep_counter = 0
                if self.slow_path.clients.pause_until_ms == 0:   # #47: CLIENT PAUSE holds expiry too
                    self.fast_path.sweep_expired_keys(20)

            # Parked XREAD BLOCK clients, every tick while any exist (the
            # check is one load): answered when an XADD reached them or on timeout.
            if self.slow_path.blocked_readers._count() > 0:
                self._service_blocked_readers(Int32(-1), hnsw, db_size, -1)
            # #38: parked BLPOP & co., the same way.
            if self.slow_path.blocked_clients._count() > 0:
                self._service_blocked_clients(Int32(-1), hnsw, db_size, -1)

            # MOE.EXPERT.* Stage 4b: drain warming-thread completions into the
            # LRU cache. Cheap (atomic load, 0-N memcpy per tick); no-op
            # when --moe-cache isn't on (warm_pool == 0).
            _ = self.slow_path.moe_tier.drain_warm_into_cache()

            # Deferred shard responses: drain every tick when queries are pending
            # (was gated to every 64 ticks — added 2×64-tick round-trip latency).
            if self.slow_path.deferred_count > 0:
                self.slow_path.drain_deferred_shard_responses(hnsw, self.writer, self.server, Int32(-1))

            # Periodic housekeeping: every 64 ticks (gh #85 — unified helper).
            if self.ttl_sweep_counter & 0x3F == 0:
                self._housekeeping_64tick(hnsw)
                # gh #259: WAL is durably flushed; end the worker thread so
                # main() can fall out of pthread_join and exit cleanly.
                if self.shutting_down:
                    return

    def run_server_uring(mut self, mut hnsw: HNSWGraph, mut db_size: Int) raises:
        from src.network.replication import apply_wal_entries
        var use_sqpoll = self.config.server.use_sqpoll
        var ring_ok = self.ring[].setup(UInt32(1024), sqpoll=use_sqpoll)
        if not ring_ok and use_sqpoll:
            print("io_uring SQPOLL setup failed (it needs root or CAP_SYS_NICE); trying without SQPOLL")
            ring_ok = self.ring[].setup(UInt32(1024), sqpoll=False)
        if not ring_ok:
            # #21: this fell back to run_server_kqueue, which returns at once on
            # Linux, so the worker thread ended while the listening socket stayed
            # open: a port that accepted connections and never answered them.
            # io_uring is missing under Docker's default seccomp profile, on old
            # kernels and in sandboxes; epoll is always there.
            print("io_uring unavailable (blocked by seccomp, or an old kernel): worker "
                  + String(self.worker_id) + " uses epoll")
            self.run_server_epoll(hnsw, db_size)
            return

        if self.server.fd < 0 and not self.server.listen():
            return

        # Wire ResponseWriter to the ring so _flush_uring() can submit SENDs.
        self.writer.bind_ring(self.ring)
        var kq = Int32(-1)  # no kqueue on uring path; flush_response ignores this

        # Multishot recv: allocate provided buffer pool and register with kernel.
        # 256 × 16KB = 4MB of provided buffers. Falls back to per-fd recv if kernel
        # doesn't support multishot (< 6.0) or provide_buffers fails.
        var buf_group_id = UInt16(self.worker_id)
        self.multishot_bufs = alloc[UInt8](PBUF_RING_ENTRIES * PBUF_SIZE)
        # gh #478: touch the whole pool now. The kernel hands provided buffers
        # out first-in first-out (a recycled one goes to the back), so traffic
        # walks all 256 of them, and each page became resident the first time
        # a RECV landed in it: RSS crept up by as much as 4 MB per worker over
        # the first 4 MB received, which a leak check (and an operator) reads
        # as a leak, and every first landing was a page fault on the receive
        # path. Faulting them in here costs the same 4 MB, once, before the
        # first client.
        unsafe_memset(self.multishot_bufs, 0, PBUF_RING_ENTRIES * PBUF_SIZE)
        self.ring[].submit_provide_buffers(
            self.multishot_bufs, PBUF_SIZE, PBUF_RING_ENTRIES,
            buf_group_id, UInt16(0))
        self.ring[].enter(Int32(1))
        var pbuf_peek = self.ring[].peek_cqe()
        if pbuf_peek.found and pbuf_peek.cqe.res >= 0:
            self.multishot_active = True
            self.ring[].advance_cq()
            print("Multishot recv: " + String(PBUF_RING_ENTRIES) + " × " + String(PBUF_SIZE) + "B provided buffers registered")
        else:
            if pbuf_peek.found:
                self.ring[].advance_cq()
            self.multishot_active = False
            self.multishot_bufs.unsafe_free()
            self.multishot_bufs = null_ptr[UInt8, MutUntrackedOrigin]()
            print("Multishot recv: not available, using per-fd recv")

        # Submit initial ACCEPT(s).
        self.ring[].submit_accept(self.server.fd)
        if self.secondary_listen_fd >= 0:
            self.ring[].submit_accept(self.secondary_listen_fd)
        if self.binary_listen_fd >= 0:
            self.ring[].submit_accept(self.binary_listen_fd)

        var my_tid = external_call["pthread_self", UInt64]()
        print("--- Pion IO_URING Engine Active --- worker=" + String(self.worker_id) + " tid=" + String(my_tid))

        # #17: one 1 ms OP_TIMEOUT is ALWAYS in flight, so enter() returns at
        # least every millisecond and the loop ticks with no client traffic, as
        # the kqueue (1 ms) and epoll (1 ms) loops do. It used to be armed only
        # while a blocked XREAD or a parked WAIT existed (gh #173), so an idle
        # worker slept in enter() for good: the shutdown drain, the replica's
        # apply-and-ACK, the TTL and field-expiry sweeps, a primary's FULLRESYNC
        # service and the status heartbeat all stopped until a client sent bytes.
        var uring_timeout_armed = False

        while True:
            # shard_active: always non-blocking when num_shards>1.
            # Non-coordinator workers have index_ready=False before shard build; if we wait for
            # trigger_pending they are ALREADY blocked in enter() when the trigger fires → poll=0.
            # Solution: any worker in a sharded config must never sleep in enter().
            var _sh2 = self.slow_path.shared_hnsw
            var shard_active = (is_not_null(_sh2) and _sh2[].num_shards > 1)
            var min_complete = Int32(1)
            if self.num_workers > 1 and shard_active:
                min_complete = Int32(0)

            if not uring_timeout_armed:
                self.ring[].submit_timeout(1)
                uring_timeout_armed = True

            # Submit every SQE written since the last enter; wait for one CQE.
            self.ring[].enter(min_complete)

            # Drain all available CQEs without blocking.
            while True:
                var peek = self.ring[].peek_cqe()
                if not peek.found: break
                var cqe = peek.cqe
                self.ring[].advance_cq()
                var kind = IOUring.ud_kind(cqe.user_data)

                if kind == UD_RECV:
                    var fd = IOUring.fd_from_user_data(cqe.user_data)
                    var ci = Int(fd)
                    var n = Int(cqe.res)
                    var has_buf = IOUring.cqe_has_buffer(cqe.flags)
                    if self.ring[].is_stale(cqe.user_data):
                        # The connection this RECV belonged to is gone.
                        if has_buf:
                            self._uring_recycle_pbuf(cqe.flags, buf_group_id)
                        continue
                    # Multishot: while F_MORE is set the request stays armed.
                    var is_multishot = self.uring_recv_armed[unsafe_offset=ci] == 2
                    if not (is_multishot and IOUring.cqe_has_more(cqe.flags)):
                        self.uring_recv_armed[unsafe_offset=ci] = 0
                    if self.ring[].fd_closing[unsafe_offset=ci] != 0:
                        if has_buf:
                            self._uring_recycle_pbuf(cqe.flags, buf_group_id)
                        self._uring_finish_close(fd, ci)
                        continue

                    if n <= 0:
                        if is_multishot and n == -105 and self.uring_recv_armed[unsafe_offset=ci] == 0:
                            # -ENOBUFS: the provided-buffer pool ran dry. The
                            # connection is fine; re-arm (the buffers recycled
                            # in this pass go to the kernel ahead of it).
                            self.ring[].submit_recv_multishot(fd, buf_group_id)
                            self.uring_recv_armed[unsafe_offset=ci] = 2
                        else:
                            # EOF, or a socket error. Re-arming on any error
                            # (as before) spun forever on a reset connection.
                            self._uring_close_fd(fd, ci)
                        continue

                    var stored = self.client_buffer_lens[unsafe_offset=ci]
                    if is_multishot:
                        if not has_buf:
                            self._uring_close_fd(fd, ci)
                            continue
                        var bid = Int(IOUring.cqe_buffer_id(cqe.flags))
                        var src_buf = self.multishot_bufs.unsafe_offset(bid * PBUF_SIZE)
                        if stored + n > CLIENT_BUF_SIZE:
                            # An unfinished request larger than the client
                            # buffer: close the connection, as the other loops
                            # do when the buffer is full. Nothing is dispatched.
                            self._uring_recycle_pbuf(cqe.flags, buf_group_id)
                            self._uring_close_fd(fd, ci)
                            continue
                        unsafe_memcpy(dest=self.client_buffers[unsafe_offset=ci].unsafe_offset(stored), src=src_buf, count=n)
                        # Recycle: re-provide this single buffer to the kernel
                        self._uring_recycle_pbuf(cqe.flags, buf_group_id)

                    # The buffer holds stored + n bytes (multishot copied them
                    # above; a plain RECV wrote at client_buffers[ci] + stored).
                    # gh #85: dispatch + drain is the shared body.
                    self._dispatch_recv_buffer(
                        fd, ci, stored, n, kq, hnsw, db_size,
                    )
                    # Arm next RECV immediately — overlap with in-flight SEND.
                    # For multishot: re-arm only if exhausted (uring_recv_armed==0).
                    # For regular: always re-arm (saves one io_uring_enter round-trip).
                    # gh #390: a parked fd is re-armed by _service_parked_waits.
                    if self.uring_recv_armed[unsafe_offset=ci] == 0 and not self.slow_path.parked_waits.is_parked(ci):
                        self._uring_arm_recv(fd, ci, buf_group_id)

                elif kind == UD_SEND:
                    var fd = IOUring.fd_from_user_data(cqe.user_data)
                    var ci = Int(fd)
                    if self.ring[].is_stale(cqe.user_data):
                        continue
                    var sent = Int(cqe.res)
                    self.writer.ctx[].uring_inflight[unsafe_offset=ci] = 0   # this SEND is over
                    if self.ring[].fd_closing[unsafe_offset=ci] != 0:
                        self._uring_finish_close(fd, ci)
                        continue
                    if sent <= 0:
                        self._uring_close_fd(fd, ci)
                        continue
                    # Bytes still owed: what this SEND did not take, plus what
                    # was queued while it was in flight (#49: refilled from the
                    # overflow queue).
                    self.writer.uring_sent(ci, sent)
                    if self.writer.ctx[].pending_offsets[unsafe_offset=ci] > 0:
                        self.writer.uring_kick(fd, ci)
                    else:
                        # #47: a client that killed itself, its reply now out
                        if self.slow_path.clients.close_after[unsafe_offset=ci] != 0:
                            self._uring_close_fd(fd, ci)
                            continue
                        # All sent — arm the next RECV if none is armed.
                        # gh #390: not while a WAIT is parked — _service_parked_waits
                        # moves this buffer's bytes and re-arms itself.
                        if self.uring_recv_armed[unsafe_offset=ci] == 0 and self.client_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]() \
                           and not self.slow_path.parked_waits.is_parked(ci):
                            self._uring_arm_recv(fd, ci, buf_group_id)

                elif kind == UD_ACCEPT:
                    var listen_fd = IOUring.fd_from_user_data(cqe.user_data)
                    var new_fd = cqe.res
                    # Resubmit ACCEPT immediately so the next connection isn't missed.
                    self.ring[].submit_accept(listen_fd)
                    if new_fd >= 0:
                        var ci = Int(new_fd)
                        if ci >= URING_MAX_FDS:
                            # Every per-fd table holds URING_MAX_FDS entries.
                            _ = external_call["close", Int32](new_fd)
                            continue
                        self.server.set_nonblocking(new_fd)
                        self.server.set_tcp_nodelay(new_fd)
                        self.client_buffer_lens[unsafe_offset=ci] = 0
                        self.uring_recv_armed[unsafe_offset=ci] = 0
                        self.ring[].fd_closing[unsafe_offset=ci] = 0
                        var is_binary = (self.binary_listen_fd >= 0 and listen_fd == self.binary_listen_fd)
                        # All connections are local-affinity (shared-nothing model).
                        var affinity_val = UInt8(1)
                        if is_binary:
                            affinity_val = UInt8(2)
                        self.local_affinity[unsafe_offset=ci] = affinity_val
                        # Allocate recv buffer and clear any stale send state.
                        if self.client_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
                            self.client_buffers[unsafe_offset=ci].unsafe_free()
                        self.client_buffers[unsafe_offset=ci] = alloc[UInt8](CLIENT_BUF_SIZE)
                        if self.writer.ctx[].pending_buffers[unsafe_offset=ci] != null_ptr[UInt8, MutUntrackedOrigin]():
                            self.writer.ctx[].pending_buffers[unsafe_offset=ci].unsafe_free()
                            self.writer.ctx[].pending_buffers[unsafe_offset=ci] = null_ptr[UInt8, MutUntrackedOrigin]()
                        self.writer.ctx[].pending_offsets[unsafe_offset=ci] = 0
                        self.writer.out_free(ci)                     # #49
                        self.writer.ctx[].uring_inflight[unsafe_offset=ci] = 0
                        self.slow_path.clients.on_accept(new_fd)   # #47
                        self._uring_arm_recv(new_fd, ci, buf_group_id)

                elif kind == UD_TIMEOUT:
                    # The tick timeout fired (res=-ETIME); its only job was to
                    # wake the loop.
                    uring_timeout_armed = False
                # UD_PBUF (buffers are back with the kernel) and UD_CANCEL need
                # nothing: a cancelled request reports through its own CQE.

            # gh #85b: KV_BUS routing was removed here — bus-drain and bus-serve
            # were gated by `comptime if KV_BUS_ENABLED` (False since gh #48).
            # Shared-nothing is the committed model.

            # Cross-worker pub/sub broadcast drain
            if self.num_workers > 1:
                self.slow_path.drain_pubsub(self.writer, self.server, kq)   # #42

            # Active TTL sweep + housekeeping counter.
            self.ttl_sweep_counter += 1
            if self.ttl_sweep_counter >= 100:
                self.ttl_sweep_counter = 0
                if self.slow_path.clients.pause_until_ms == 0:   # #47: CLIENT PAUSE holds expiry too
                    self.fast_path.sweep_expired_keys(20)

            # Parked XREAD BLOCK clients, every tick while any exist (the
            # check is one load): answered when an XADD reached them or on timeout.
            if self.slow_path.blocked_readers._count() > 0:
                self._service_blocked_readers(Int32(-1), hnsw, db_size, Int(buf_group_id))
            # #38: parked BLPOP & co., the same way.
            if self.slow_path.blocked_clients._count() > 0:
                self._service_blocked_clients(Int32(-1), hnsw, db_size, Int(buf_group_id))

            # MOE.EXPERT.* Stage 4b: drain warming-thread completions into the LRU
            # cache. Cheap (atomic load, no-op when --moe-cache isn't on). gh #85:
            # was present on xdp/kqueue only — aligned across all four loops.
            _ = self.slow_path.moe_tier.drain_warm_into_cache()

            # Deferred shard responses: drain every tick when queries are pending
            # (was gated to every 64 ticks — added 2×64-tick round-trip latency).
            if self.slow_path.deferred_count > 0:
                self.slow_path.drain_deferred_shard_responses(hnsw, self.writer, self.server, Int32(-1))

            # gh #390: answer parked WAITs (every tick while any is parked).
            if self.slow_path.parked_waits.count() > 0:
                self._service_parked_waits(Int32(-1), hnsw, db_size, Int(buf_group_id))
            # #47: connections CLIENT PAUSE held, once it is over or changed
            if self.slow_path.clients.pause_until_ms != 0 or self.slow_path.clients.pause_changed \
               or len(self.slow_path.clients.released) > 0:
                self._service_pause(Int32(-1), hnsw, db_size, Int(buf_group_id))

            # Periodic housekeeping: every 64 ticks (gh #85 — unified helper).
            if self.ttl_sweep_counter & 0x3F == 0:
                self._housekeeping_64tick(hnsw)
                # gh #259: WAL is durably flushed; end the worker thread so
                # main() can fall out of pthread_join and exit cleanly.
                if self.shutting_down:
                    self._uring_stop_accepting()
                    return

    @always_inline
    def _uring_arm_recv(mut self, fd: Int32, ci: Int, buf_group_id: UInt16):
        """Arm the next RECV for a connection that has none armed. A plain RECV
        reads straight into the client buffer after the bytes it already holds;
        when an unfinished request has filled that buffer, the connection is
        closed (a zero-length RECV used to stand in for that, by reading as EOF)."""
        if self.ring[].fd_closing[unsafe_offset=ci] != 0:
            return
        if self.multishot_active:
            self.ring[].submit_recv_multishot(fd, buf_group_id)
            self.uring_recv_armed[unsafe_offset=ci] = 2  # 2 = multishot (persistent)
        else:
            var stored = self.client_buffer_lens[unsafe_offset=ci]
            if stored >= CLIENT_BUF_SIZE:
                self._uring_close_fd(fd, ci)
                return
            self.ring[].submit_recv(fd, self.client_buffers[unsafe_offset=ci].unsafe_offset(stored), CLIENT_BUF_SIZE - stored)
            self.uring_recv_armed[unsafe_offset=ci] = 1

    @always_inline
    def _uring_recycle_pbuf(mut self, cqe_flags: UInt32, buf_group_id: UInt16):
        """Hand a provided buffer a RECV completion carried back to the kernel."""
        var bid = Int(IOUring.cqe_buffer_id(cqe_flags))
        self.ring[].submit_provide_buffers(
            self.multishot_bufs.unsafe_offset(bid * PBUF_SIZE), PBUF_SIZE, 1, buf_group_id, UInt16(bid))

    def _uring_stop_accepting(mut self):
        """#22: on a graceful stop, cancel this worker's ACCEPTs and wait for
        them to complete before the worker returns. An ACCEPT in flight holds a
        reference to the listening socket, and the kernel tears a ring down
        asynchronously after the process exits, so the port used to keep
        listening (and accepting into its backlog) for a moment after the
        server was gone. With no request left on it, the socket closes with the
        process. Bounded at ~200 ms: shutdown must not hang on it."""
        var want = 1
        self.ring[].submit_cancel((UD_ACCEPT << 32) | UInt64(UInt32(self.server.fd)))
        if self.secondary_listen_fd >= 0:
            self.ring[].submit_cancel((UD_ACCEPT << 32) | UInt64(UInt32(self.secondary_listen_fd)))
            want += 1
        if self.binary_listen_fd >= 0:
            self.ring[].submit_cancel((UD_ACCEPT << 32) | UInt64(UInt32(self.binary_listen_fd)))
            want += 1
        var done = 0
        var timeout_armed = False
        for _ in range(200):
            if done >= want:
                break
            if not timeout_armed:
                self.ring[].submit_timeout(1)
                timeout_armed = True
            self.ring[].enter(Int32(1))
            while True:
                var peek = self.ring[].peek_cqe()
                if not peek.found: break
                var cqe = peek.cqe
                self.ring[].advance_cq()
                var kind = IOUring.ud_kind(cqe.user_data)
                if kind == UD_ACCEPT:
                    if cqe.res >= 0:
                        # A connection that arrived first: it is not going to be served.
                        _ = external_call["close", Int32](cqe.res)
                    else:
                        done += 1
                elif kind == UD_TIMEOUT:
                    timeout_armed = False

    @always_inline
    def _uring_close_fd(mut self, fd: Int32, ci: Int):
        """Close a connection on the io_uring path, in two phases. This, the
        first, stops it: shutdown() shows the peer the close at once and ends
        its in-flight RECV and SEND, which are also cancelled. The second,
        `_uring_finish_close`, runs once no RECV or SEND is in flight for the
        fd: only then are its buffers freed and its number released, because
        until then the kernel still owns them. The generation stamped on every
        completion backs this up (see UD_* in io_uring.mojo)."""
        if self.ring[].fd_closing[unsafe_offset=ci] == 0:
            self.ring[].fd_closing[unsafe_offset=ci] = 1
            _ = external_call["shutdown", Int32](fd, Int32(2))   # SHUT_RDWR
            if self.uring_recv_armed[unsafe_offset=ci] != 0:
                self.ring[].submit_cancel(self.ring[].make_ud(UD_RECV, fd))
            if self.writer.ctx[].uring_inflight[unsafe_offset=ci] != 0:
                self.ring[].submit_cancel(self.ring[].make_ud(UD_SEND, fd))
        self._uring_finish_close(fd, ci)

    @always_inline
    def _uring_finish_close(mut self, fd: Int32, ci: Int):
        """Second phase of `_uring_close_fd`: nothing is in flight any more."""
        if self.uring_recv_armed[unsafe_offset=ci] != 0 or self.writer.ctx[].uring_inflight[unsafe_offset=ci] != 0:
            return
        self.ring[].retire_fd(fd)
        self.ring[].fd_closing[unsafe_offset=ci] = 0
        self._close_fd_common(fd, ci)

    def run_server_epoll(mut self, mut hnsw: HNSWGraph, mut db_size: Int) raises:
        """epoll event loop — lowest overhead Linux path for P=1 workloads.

        Uses epoll_wait + read() + write() inline — 3 syscalls per command vs
        io_uring's 4+ (enter/CQE drain/SQE resubmit). Level-triggered for both
        listen and client fds (no re-arming needed). EAGAIN on write → register
        EPOLLOUT, handled on next iteration.

        Expected: +25-35% at P=1 vs io_uring path."""
        from src.network.replication import apply_wal_entries
        if self.server.fd < 0 and not self.server.listen():
            return

        # Create epoll instance
        var epfd = external_call["epoll_create1", Int32](Int32(0))
        if epfd < 0:
            print("epoll_create1 failed")
            return

        # Register listen fd for EPOLLIN + EPOLLEXCLUSIVE (level-triggered).
        # EPOLLEXCLUSIVE: only one worker wakes per incoming connection, preventing
        # thundering herd and ensuring even connection distribution across workers.
        var listen_flags = EPOLLIN
        if self.num_workers > 1:
            listen_flags = EPOLLIN | EPOLLEXCLUSIVE
        _ = epoll_ctl_fd(epfd, EPOLL_CTL_ADD, self.server.fd, listen_flags)

        # Secondary listen fd (affinity port)
        if self.secondary_listen_fd >= 0:
            _ = epoll_ctl_fd(epfd, EPOLL_CTL_ADD, self.secondary_listen_fd, listen_flags)

        # Binary protocol listen fd
        if self.binary_listen_fd >= 0:
            _ = epoll_ctl_fd(epfd, EPOLL_CTL_ADD, self.binary_listen_fd, listen_flags)

        # Raw bytes, read through epoll_ev_*: x86-64's epoll_event is packed (12 B).
        var events = alloc[UInt8](1024 * 16)
        # Pass epfd as kq — _flush_kqueue will use kevent_add_write/kevent_del_write
        # which now dispatch to epoll_ctl on Linux when kq >= 0.
        var kq = epfd

        var my_tid = external_call["pthread_self", UInt64]()
        print("--- Pion EPOLL Engine Active --- worker=" + String(self.worker_id) + " tid=" + String(my_tid))

        while True:
            # Adaptive timeout: 0ms (non-blocking) when bus needs polling, else 1ms.
            var _sh2 = self.slow_path.shared_hnsw
            var shard_active = (is_not_null(_sh2) and _sh2[].num_shards > 1)
            var timeout_ms = Int32(1)
            if self.num_workers > 1 and shard_active:
                timeout_ms = Int32(0)

            var nevents = external_call["epoll_wait", Int32](epfd, events, Int32(1024), timeout_ms)

            # Housekeeping (same gating as kqueue/io_uring)
            self.ttl_sweep_counter += 1
            if self.ttl_sweep_counter >= 100:
                self.ttl_sweep_counter = 0
                if self.slow_path.clients.pause_until_ms == 0:   # #47: CLIENT PAUSE holds expiry too
                    self.fast_path.sweep_expired_keys(20)

            # Parked XREAD BLOCK clients, every tick while any exist.
            if self.slow_path.blocked_readers._count() > 0:
                self._service_blocked_readers(kq, hnsw, db_size)
            # #38: parked BLPOP & co., the same way.
            if self.slow_path.blocked_clients._count() > 0:
                self._service_blocked_clients(kq, hnsw, db_size)

            # MOE.EXPERT.* Stage 4b warming-completion drain (no-op without
            # --moe-cache). gh #85: aligned across all four loops.
            _ = self.slow_path.moe_tier.drain_warm_into_cache()

            if nevents <= 0: continue

            for i in range(nevents):
                var fd = epoll_ev_fd(events, Int(i))
                var evmask = epoll_ev_events(events, Int(i))

                # Guard: fd 0/1/2 are stdin/stdout/stderr — never client fds.
                # If they appear in epoll events, something went wrong (e.g., stale
                # registration from fd reuse after close). Skip to avoid corrupting
                # stdout or reading from stdin.
                if fd < 3:
                    continue

                # EPOLLOUT: flush pending writes
                if evmask & EPOLLOUT:
                    self.writer.flush_response(fd, self.server, kq)
                    self._close_after_reply(fd)    # #47
                    # If both EPOLLIN and EPOLLOUT are set, also process EPOLLIN below
                    if not (evmask & EPOLLIN):
                        continue

                var is_listen_fd = (fd == self.server.fd or (self.secondary_listen_fd >= 0 and fd == self.secondary_listen_fd) or (self.binary_listen_fd >= 0 and fd == self.binary_listen_fd))
                if is_listen_fd:
                    var is_binary = (self.binary_listen_fd >= 0 and fd == self.binary_listen_fd)
                    # Accept all pending connections (loop until EAGAIN).
                    # With EPOLLEXCLUSIVE only one worker wakes, so drain fully.
                    while True:
                        var new_fd = self.server.accept_from(fd)
                        if new_fd < 0: break
                        if Int(new_fd) >= URING_MAX_FDS:
                            # Every per-fd table holds 65536 entries; a client
                            # that opened that many connections would index
                            # past them.
                            _ = external_call["close", Int32](new_fd)
                            continue
                        self.server.set_nonblocking(new_fd)
                        self.server.set_tcp_nodelay(new_fd)
                        self.client_buffer_lens[unsafe_offset=Int(new_fd)] = 0
                        var affinity_val = UInt8(1)
                        if is_binary:
                            affinity_val = UInt8(2)
                        self.local_affinity[unsafe_offset=Int(new_fd)] = affinity_val
                        if self.client_buffers[unsafe_offset=Int(new_fd)] != null_ptr[UInt8, MutUntrackedOrigin]():
                            self.client_buffers[unsafe_offset=Int(new_fd)].unsafe_free()
                            self.client_buffers[unsafe_offset=Int(new_fd)] = null_ptr[UInt8, MutUntrackedOrigin]()
                        if self.writer.ctx[].pending_buffers[unsafe_offset=Int(new_fd)] != null_ptr[UInt8, MutUntrackedOrigin]():
                            self.writer.ctx[].pending_buffers[unsafe_offset=Int(new_fd)].unsafe_free()
                            self.writer.ctx[].pending_buffers[unsafe_offset=Int(new_fd)] = null_ptr[UInt8, MutUntrackedOrigin]()
                        self.writer.ctx[].pending_offsets[unsafe_offset=Int(new_fd)] = 0
                        self.writer.out_free(Int(new_fd))                     # #49
                        self.slow_path.clients.on_accept(new_fd)   # #47
                        # Register for EPOLLIN (level-triggered)
                        _ = epoll_ctl_fd(epfd, EPOLL_CTL_ADD, new_fd, EPOLLIN)
                else:
                    # EPOLLERR/EPOLLHUP: always close — even if EPOLLIN is also set.
                    # Trying to read from an errored fd risks stale data or hangs.
                    # gh #85: epoll-specific deregister + shared close cleanup.
                    # (This path previously had a subtle bug: writer.ctx[].pending_buffers
                    # / pending_offsets reset were nested inside the client_buffers
                    # null-check, so they wouldn't fire when client_buffers happened
                    # to be null. Routing through `_close_fd_common` aligns the
                    # EPOLLERR path with the recv-based close path.)
                    if evmask & (EPOLLERR | EPOLLHUP):
                        _ = epoll_ctl_fd(epfd, EPOLL_CTL_DEL, fd, 0)
                        self._close_fd_common(fd, Int(fd))
                        continue

                    var client_idx = Int(fd)
                    if self.client_buffers[unsafe_offset=client_idx] == null_ptr[UInt8, MutUntrackedOrigin]():
                        self.client_buffers[unsafe_offset=client_idx] = alloc[UInt8](CLIENT_BUF_SIZE)
                    var client_buffer = self.client_buffers[unsafe_offset=client_idx]

                    var stored_len = self.client_buffer_lens[unsafe_offset=client_idx]
                    var recv_size = CLIENT_BUF_SIZE - stored_len
                    if recv_size <= 0:
                        # The buffer is full of one unfinished request (every
                        # complete one was dispatched when it arrived), so it
                        # can never complete. `continue` here spun the worker
                        # forever: level-triggered epoll reports the fd again
                        # at once. Close it, as kqueue and io_uring do.
                        _ = epoll_ctl_fd(epfd, EPOLL_CTL_DEL, fd, 0)
                        self._close_fd_common(fd, client_idx)
                        continue
                    var n = self.server.recv(fd, client_buffer.unsafe_offset(stored_len), recv_size)
                    if n <= 0:
                        var do_close = (n == 0)
                        if n == -1:
                            var errno_val = external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
                            if errno_val != 11:  # 11=EAGAIN/EWOULDBLOCK on Linux
                                do_close = True
                            else:
                                continue
                        if do_close:
                            # gh #85: epoll-specific deregister + shared close cleanup.
                            _ = epoll_ctl_fd(epfd, EPOLL_CTL_DEL, fd, 0)
                            self._close_fd_common(fd, client_idx)
                        continue

                    # gh #85: post-recv dispatch + drain. Byte-identical with
                    # the kqueue path — extracted to `_dispatch_recv_buffer`.
                    self._dispatch_recv_buffer(
                        fd, client_idx, stored_len, n, kq, hnsw, db_size,
                    )

            # Multi-worker housekeeping
            # gh #85b: KV_BUS routing was removed here (shared-nothing, gh #48).
            if self.num_workers > 1:
                self.slow_path.drain_pubsub(self.writer, self.server, kq)   # #42

            # Deferred shard responses: drain every tick when queries are pending
            if self.slow_path.deferred_count > 0:
                self.slow_path.drain_deferred_shard_responses(hnsw, self.writer, self.server, kq)

            # gh #390: answer parked WAITs (every tick while any is parked).
            if self.slow_path.parked_waits.count() > 0:
                self._service_parked_waits(kq, hnsw, db_size)
            # #47: connections CLIENT PAUSE held, once it is over or changed
            if self.slow_path.clients.pause_until_ms != 0 or self.slow_path.clients.pause_changed \
               or len(self.slow_path.clients.released) > 0:
                self._service_pause(kq, hnsw, db_size)

            # 64-tick gated housekeeping (gh #85 — unified helper).
            if self.ttl_sweep_counter & 0x3F == 0:
                self._housekeeping_64tick(hnsw)
                # gh #259: WAL is durably flushed; end the worker thread so
                # main() can fall out of pthread_join and exit cleanly.
                if self.shutting_down:
                    return

    def run_server_kqueue(mut self, mut hnsw: HNSWGraph, mut db_size: Int) raises:
        from src.network.replication import apply_wal_entries
        if self.server.fd < 0 and not self.server.listen():
            return

        print("Server socket FD: " + String(self.server.fd))
        self.kq = self.server.kqueue()
        var kq = self.kq
        if kq < 0:
            print("Failed to create kqueue")
            return

        # V22: Level-triggered listen socket — re-fires on each kevent_batch until accept queue is empty.
        # Workers call accept() once per event; with multiple pending connections, kevent fires again.
        # This drains all N connections across up to N kevent_batch iterations without an accept loop,
        # avoiding the blocking second accept() issue with the shared nonblocking socket.
        self.server.kevent_add_read(kq, self.server.fd, edge_triggered=False)
        # P4: register per-worker secondary listen socket (if set) for local-affinity connections.
        if self.secondary_listen_fd >= 0:
            self.server.kevent_add_read(kq, self.secondary_listen_fd, edge_triggered=False)
        # Binary protocol listen socket (shared across workers).
        if self.binary_listen_fd >= 0:
            self.server.kevent_add_read(kq, self.binary_listen_fd, edge_triggered=False)

        var events = alloc[KEvent](1024)
        var timeout = alloc[Int](2)
        timeout[unsafe_offset=0] = 0 # 0 seconds
        timeout[unsafe_offset=1] = 1000000 # 1ms in nanoseconds

        var my_tid = external_call["pthread_self", UInt64]()
        print("--- Pion KQUEUE Engine Active --- worker=" + String(self.worker_id) + " tid=" + String(my_tid))
        # Bounded adaptive polling: consecutive zero-timeout polls since the
        # last event. See the timeout choice below.
        var kq_idle_polls = 0

        while True:
            # P2 async: adaptive kevent timeout.
            # - shard_active: sharding enabled; shard queries can arrive at any time and
            #   coordinators spin-wait — a 1ms kevent block stalls all sharded queries.
            # All conditions use timeout=0 (non-blocking) for ~100-200ns round-trips.
            # When idle for >100 iterations (~10µs), back off to 1ms to avoid busy-spinning.
            var _sh2 = self.slow_path.shared_hnsw
            # shard_active: always non-blocking when num_shards>1 (same fix as io_uring path).
            var shard_active = (is_not_null(_sh2) and _sh2[].num_shards > 1)
            # Bounded adaptive polling: after a tick that saw events, poll with a
            # zero timeout for up to KQ_SPIN_POLLS empty polls before blocking.
            # On macOS loopback a send() also wakes the receiving thread, and the
            # sender pays for it. A worker that dozes off between pipelined
            # batches makes every client send a wake-up, and every reply wake
            # the client. Measured on SADD: the client spent 2.1x more time in
            # sendto and 2.3x more in recvfrom per op against Pion than against
            # a Redis that was 100% busy and never asleep, so Pion lost the row
            # (-13%) by being fast enough to sleep. An idle server still blocks
            # after at most KQ_SPIN_POLLS empty polls, so this costs microseconds
            # of CPU per burst, not a spinning core.
            #
            # Single-worker only. Workers race for accept() on one shared listen
            # socket, and a spinning worker sees each new connection before a
            # blocked one wakes, so it wins every race. That skewed connections
            # onto busy workers, and KV.PREFIX's reconnect-until-owner redirect
            # (test_kvprefix_autoredirect) ran out of retries 1 run in 3.
            # With one worker there is no race to skew.
            var forced_poll = self.num_workers > 1 and shard_active
            var may_spin = self.num_workers == 1 and kq_idle_polls < KQ_SPIN_POLLS
            if forced_poll or may_spin:
                timeout[unsafe_offset=1] = 0
            else:
                timeout[unsafe_offset=1] = 1000000

            # Batch pending kevent changes with the wait syscall (P3.4)
            var nevents = self.server.kevent_batch(kq, self.pending_changes, Int32(self.pending_change_count), events, Int32(1024), timeout)
            self.pending_change_count = 0
            if nevents > 0:
                kq_idle_polls = 0
            elif timeout[unsafe_offset=1] == 0 and not forced_poll:
                # An empty spin poll is not a tick: nothing happened, so the
                # tick-counted work below (TTL sweep, XREAD drain, the 64-tick
                # housekeeping and its WAL msync) keeps the cadence it had.
                kq_idle_polls += 1
                continue

            # Active TTL sweep + housekeeping counter.
            self.ttl_sweep_counter += 1
            if self.ttl_sweep_counter >= 100:
                self.ttl_sweep_counter = 0
                if self.slow_path.clients.pause_until_ms == 0:   # #47: CLIENT PAUSE holds expiry too
                    self.fast_path.sweep_expired_keys(20)

            # Parked XREAD BLOCK clients, every tick while any exist (the
            # check is one load): answered when an XADD reached them or on timeout.
            if self.slow_path.blocked_readers._count() > 0:
                self._service_blocked_readers(kq, hnsw, db_size)
            # #38: parked BLPOP & co., the same way.
            if self.slow_path.blocked_clients._count() > 0:
                self._service_blocked_clients(kq, hnsw, db_size)

            # MOE.EXPERT.* Stage 4b: drain warming-thread completions into the
            # LRU cache. Cheap (atomic load, 0-N memcpy per tick); no-op
            # when --moe-cache isn't on (warm_pool == 0).
            _ = self.slow_path.moe_tier.drain_warm_into_cache()

            if nevents < 0: continue

            for i in range(nevents):
                var fd = Int32(events[unsafe_offset=i].ident)
                if events[unsafe_offset=i].filter == -2:
                    self.writer.flush_response(fd, self.server, kq)
                    self._close_after_reply(fd)    # #47
                    continue
                var is_listen_fd = (fd == self.server.fd or (self.secondary_listen_fd >= 0 and fd == self.secondary_listen_fd) or (self.binary_listen_fd >= 0 and fd == self.binary_listen_fd))
                if is_listen_fd:
                    # V22: level-triggered listen socket — accept one connection per event.
                    # kevent re-fires if more connections remain in the accept queue.
                    var _is_secondary = (self.secondary_listen_fd >= 0 and fd == self.secondary_listen_fd)
                    var is_binary = (self.binary_listen_fd >= 0 and fd == self.binary_listen_fd)
                    var new_fd = self.server.accept_from(fd)
                    if new_fd >= 0 and Int(new_fd) >= URING_MAX_FDS:
                        _ = external_call["close", Int32](new_fd)   # past the per-fd tables
                    elif new_fd >= 0:
                        self.server.set_nonblocking(new_fd)
                        self.server.set_tcp_nodelay(new_fd)
                        # Clear stale buffer from previous connection on this FD (handles RST cleanup)
                        self.client_buffer_lens[unsafe_offset=Int(new_fd)] = 0
                        # All connections are local-affinity (shared-nothing model).
                        var affinity_val = UInt8(1)
                        if is_binary:
                            affinity_val = UInt8(2)
                        self.local_affinity[unsafe_offset=Int(new_fd)] = affinity_val
                        if self.client_buffers[unsafe_offset=Int(new_fd)] != null_ptr[UInt8, MutUntrackedOrigin]():
                            self.client_buffers[unsafe_offset=Int(new_fd)].unsafe_free()
                            self.client_buffers[unsafe_offset=Int(new_fd)] = null_ptr[UInt8, MutUntrackedOrigin]()
                        # Clear pending write state
                        if self.writer.ctx[].pending_buffers[unsafe_offset=Int(new_fd)] != null_ptr[UInt8, MutUntrackedOrigin]():
                            self.writer.ctx[].pending_buffers[unsafe_offset=Int(new_fd)].unsafe_free()
                            self.writer.ctx[].pending_buffers[unsafe_offset=Int(new_fd)] = null_ptr[UInt8, MutUntrackedOrigin]()
                        self.writer.ctx[].pending_offsets[unsafe_offset=Int(new_fd)] = 0
                        self.writer.out_free(Int(new_fd))                     # #49
                        # Register level-triggered READ for client
                        self.slow_path.clients.on_accept(new_fd)   # #47
                        self.server.kevent_add_read(kq, new_fd, edge_triggered=False)
                else:
                    var client_idx = Int(fd)
                    if self.client_buffers[unsafe_offset=client_idx] == null_ptr[UInt8, MutUntrackedOrigin]():
                        self.client_buffers[unsafe_offset=client_idx] = alloc[UInt8](CLIENT_BUF_SIZE)
                    var client_buffer = self.client_buffers[unsafe_offset=client_idx]

                    var stored_len = self.client_buffer_lens[unsafe_offset=client_idx]
                    var n = self.server.recv(fd, client_buffer.unsafe_offset(stored_len), CLIENT_BUF_SIZE - stored_len)
                    if n <= 0:
                        # n==0: clean close. n==-1: check errno — EAGAIN means no data yet (skip),
                        # any other error (ECONNRESET, EPIPE, etc.) means RST — close and clean up
                        # to avoid dirty pending_offsets corrupting the next connection on this fd.
                        var do_close = (n == 0)
                        if n == -1:
                            var errno_val: Int32
                            comptime if CompilationTarget.is_linux():
                                errno_val = external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
                            else:
                                errno_val = external_call["__error", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
                            if errno_val != 35 and errno_val != 11:  # 35=EAGAIN(macOS), 11=EWOULDBLOCK(Linux)
                                do_close = True
                            else:
                                continue # EAGAIN: skip to next event
                        if do_close:
                            # gh #85: kqueue-specific deregister + shared close cleanup.
                            self.server.kevent_del_write(kq, fd)
                            self._close_fd_common(fd, client_idx)
                        continue

                    # gh #85: post-recv dispatch + drain. Byte-identical with
                    # the epoll path — extracted to `_dispatch_recv_buffer`.
                    self._dispatch_recv_buffer(
                        fd, client_idx, stored_len, n, kq, hnsw, db_size,
                    )

            # gh #85b: KV_BUS routing was removed here (shared-nothing, gh #48).

            # Cross-worker pub/sub broadcast drain
            if self.num_workers > 1:
                self.slow_path.drain_pubsub(self.writer, self.server, kq)   # #42

            # Deferred shard responses: drain every tick when queries are pending
            if self.slow_path.deferred_count > 0:
                self.slow_path.drain_deferred_shard_responses(hnsw, self.writer, self.server, kq)

            # gh #390: answer parked WAITs (every tick while any is parked).
            if self.slow_path.parked_waits.count() > 0:
                self._service_parked_waits(kq, hnsw, db_size)
            # #47: connections CLIENT PAUSE held, once it is over or changed
            if self.slow_path.clients.pause_until_ms != 0 or self.slow_path.clients.pause_changed \
               or len(self.slow_path.clients.released) > 0:
                self._service_pause(kq, hnsw, db_size)

            # Periodic housekeeping: every 64 ticks (gh #85 — unified helper).
            # At P=1 (~1ms/tick) this is ~64ms granularity — fine for WAL sync,
            # replica drain, cluster failover (5s threshold), and HNSW shard builds.
            if self.ttl_sweep_counter & 0x3F == 0:
                self._housekeeping_64tick(hnsw)
                # gh #259: WAL is durably flushed; end the worker thread so
                # main() can fall out of pthread_join and exit cleanly.
                if self.shutting_down:
                    return
