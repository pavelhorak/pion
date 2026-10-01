from src.common.ptr import null_ptr
from std.atomic import Atomic, Ordering
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memset
from std.ffi import external_call
from std.collections import List

@fieldwise_init
struct AITask(Copyable, Movable, ImplicitlyCopyable):
    var provider: String
    var prompt: String
    var client_fd: Int32
    var session_id: String

    def copy(self) -> AITask:
        return AITask(self.provider, self.prompt, self.client_fd, self.session_id)

@fieldwise_init
struct PopResult(Copyable, Movable):
    var value: AITask
    var success: Bool

struct LockFreeRingBuffer:
    var buffer: Pointer[AITask, MutUntrackedOrigin]
    var capacity: Int
    var mask: Int
    
    var head: Atomic[Scalar[DType.uint64]]
    var _padding1: SIMD[DType.uint8, 64] 
    var tail: Atomic[Scalar[DType.uint64]]
    var _padding2: SIMD[DType.uint8, 64]

    def __init__(out self, power_of_two_capacity: Int):
        self.capacity = power_of_two_capacity
        self.mask = power_of_two_capacity - 1
        self.buffer = alloc[AITask](self.capacity)
        for i in range(self.capacity):
            self.buffer[unsafe_offset=i] = AITask("", "", 0, "")
            
        self.head = Atomic[Scalar[DType.uint64]](0)
        self.tail = Atomic[Scalar[DType.uint64]](0)
        
        self._padding1 = SIMD[DType.uint8, 64](0)
        self._padding2 = SIMD[DType.uint8, 64](0)

    @always_inline
    def push(mut self, item: AITask) -> Bool:
        var current_tail = self.tail.load[ordering=Ordering.RELAXED]()
        var current_head = self.head.load[ordering=Ordering.ACQUIRE]()
        
        # Queue is full
        if current_tail - current_head >= UInt64(self.capacity):
            return False 
            
        var idx = Int(current_tail) & self.mask
        self.buffer[idx] = item.copy()
        
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
            Pointer(to=self.tail.value), 
            current_tail + 1
        )
        return True

    @always_inline
    def pop(mut self) -> PopResult:
        var current_head = self.head.load[ordering=Ordering.RELAXED]()
        var current_tail = self.tail.load[ordering=Ordering.ACQUIRE]()
        
        # Queue is empty
        if current_head == current_tail:
            return PopResult(AITask("", "", 0, ""), False)
            
        var idx = Int(current_head) & self.mask
        var item = self.buffer[idx].copy()
        
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
            Pointer(to=self.head.value), 
            current_head + 1
        )
        return PopResult(item^, True)

    def deinit(owned self):
        pass


# ─── V18: Worker Sharding — inter-worker query bus ───────────────────────────

comptime MAX_SHARD_K = 100   # max results per shard search (k value for FT.SEARCH)

# Fixed-size slot posted by coordinator i to shard j's query inbox.
# Stores the query pointer directly (Pointer is copyable in Mojo).
@fieldwise_init
struct ShardQuerySlot(Copyable, Movable, ImplicitlyCopyable):
    var query_fp32: Pointer[Float32, MutUntrackedOrigin]
    var k: Int32
    var ef: Int32
    var seq: UInt64

# Shared coordination bus — one instance shared across ALL workers.
# Layout: slots and flags are indexed as [coordinator * num_workers + shard].
#   - Coordinator i posts a query to shard j via slot [i * N + j].
#   - Shard j's task-1 (shard worker) scans its column for pending queries.
#   - After search, shard j writes results to result[i * N + j] and sets result_ready.
struct ShardQueryBus(Movable):
    var num_workers: Int
    var max_dim: Int
    # Query posts: coordinator i → shard j at index [i * N + j]
    var query_slots:  Pointer[ShardQuerySlot, MutUntrackedOrigin]  # [N×N]
    var query_ready:  Pointer[UInt64, MutUntrackedOrigin]            # [N×N] 0=idle 1=pending
    var query_fp32_buf: Pointer[Float32, MutUntrackedOrigin]         # [N×N×max_dim] pre-allocated memory to avoid dangling pointers
    # Results: shard j → coordinator i at index [i * N + j]
    var result_ids:    Pointer[Int32, MutUntrackedOrigin]           # [N×N×MAX_SHARD_K]
    var result_scores: Pointer[Float32, MutUntrackedOrigin]         # [N×N×MAX_SHARD_K]
    var result_counts: Pointer[Int32, MutUntrackedOrigin]           # [N×N]
    var result_ready:  Pointer[UInt64, MutUntrackedOrigin]           # [N×N] 0=pending 1=done
    var result_seq:    Pointer[UInt64, MutUntrackedOrigin]           # [N×N]

    def __init__(out self, num_workers: Int):
        self.num_workers = num_workers
        self.max_dim = 2048 # Pre-allocate with max dim 2048
        var nn = num_workers * num_workers
        self.query_slots  = alloc[ShardQuerySlot](nn)
        self.query_ready  = alloc[UInt64](nn)
        self.query_fp32_buf = alloc[Float32](nn * self.max_dim)
        self.result_ids   = alloc[Int32](nn * MAX_SHARD_K)
        self.result_scores = alloc[Float32](nn * MAX_SHARD_K)
        self.result_counts = alloc[Int32](nn)
        self.result_ready  = alloc[UInt64](nn)
        self.result_seq    = alloc[UInt64](nn)
        unsafe_memset(self.query_ready.unsafe_bitcast[UInt8](), 0, nn * 8)
        unsafe_memset(self.result_ready.unsafe_bitcast[UInt8](), 0, nn * 8)
        unsafe_memset(self.result_seq.unsafe_bitcast[UInt8](), 0, nn * 8)
        for i in range(nn):
            self.result_counts[unsafe_offset=i] = 0

    def __moveinit__(out self, deinit take: Self):
        self.num_workers   = take.num_workers
        self.max_dim       = take.max_dim
        self.query_slots   = take.query_slots
        self.query_ready   = take.query_ready
        self.query_fp32_buf = take.query_fp32_buf
        self.result_ids    = take.result_ids
        self.result_scores = take.result_scores
        self.result_counts = take.result_counts
        self.result_ready  = take.result_ready
        self.result_seq    = take.result_seq

    @always_inline
    def post_query(mut self, coordinator: Int, shard: Int,
                  query_fp32: Pointer[Float32, MutUntrackedOrigin],
                  k: Int, ef: Int, seq: UInt64, dim: Int):
        """Coordinator posts a search query for shard to process."""
        var idx = coordinator * self.num_workers + shard
        var actual_dim = dim if dim <= self.max_dim else self.max_dim
        var q_ptr = self.query_fp32_buf.unsafe_offset((idx * self.max_dim))
        _ = external_call["memcpy", Pointer[NoneType, MutUntrackedOrigin]](
            q_ptr.unsafe_bitcast[NoneType](), query_fp32.unsafe_bitcast[NoneType](), actual_dim * 4
        )
        self.query_slots[unsafe_offset=idx].query_fp32 = q_ptr
        self.query_slots[unsafe_offset=idx].k  = Int32(k)
        self.query_slots[unsafe_offset=idx].ef = Int32(ef)
        self.query_slots[unsafe_offset=idx].seq = seq
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](self.result_ready.unsafe_offset(idx), UInt64(0))
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](self.query_ready.unsafe_offset(idx), UInt64(1))

    @always_inline
    def is_result_ready(self, coordinator: Int, shard: Int) -> Bool:
        var idx = coordinator * self.num_workers + shard
        return Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](self.result_ready + idx, UInt64(0)) != 0

    @always_inline
    def write_result(mut self, coordinator: Int, shard: Int,
                    ids: List[Int], scores: List[Float32], seq: UInt64):
        """Shard writes results for coordinator; sets result_ready last."""
        var idx = coordinator * self.num_workers + shard
        var count = len(ids)
        var n = count if count <= MAX_SHARD_K else MAX_SHARD_K
        self.result_counts[unsafe_offset=idx] = Int32(n)
        var base_ids = self.result_ids.unsafe_offset(idx * MAX_SHARD_K)
        var base_scores = self.result_scores.unsafe_offset(idx * MAX_SHARD_K)
        for r in range(n):
            base_ids[unsafe_offset=r]    = Int32(ids[r])
            base_scores[unsafe_offset=r] = scores[r]
        self.result_seq[unsafe_offset=idx] = seq
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](self.result_ready.unsafe_offset(idx), UInt64(1))

    @always_inline
    def scan_for_query(mut self, shard: Int, start_c: Int
                      ) -> Tuple[Bool, Int, Pointer[Float32, MutUntrackedOrigin], Int32, Int32, UInt64]:
        """Shard worker scans for a pending query (round-robin from start_c).
        Returns (found, coordinator_id, query_fp32_ptr, k, ef, seq)."""
        var N = self.num_workers
        for ci in range(N):
            var c = (start_c + ci) % N
            if c == shard: continue  # coordinator searches own shard directly, not via bus
            var idx = c * N + shard
            if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](self.query_ready + idx, UInt64(0)) != 0:
                var slot = self.query_slots[idx]
                Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](self.query_ready + idx, UInt64(0))
                return (True, c, slot.query_fp32, slot.k, slot.ef, slot.seq)
        return (False, 0, null_ptr[Float32, MutUntrackedOrigin](), Int32(0), Int32(0), UInt64(0))
