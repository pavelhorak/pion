"""Drop-in, phase A: the vector index builds itself.

A stock Redis client (RedisVL, LangChain, VectorDBBench's own) creates an
index, HSETs its documents and searches. It never sends FT.OPTIMIZE, Pion's
"ingest is over, build the graph" command, so it used to get nothing back.
Now:

  1. Until a graph is built, FT.SEARCH answers exactly from the shared ingest
     buffer (vector.mojo exact_knn): same reply, same score units.
  2. Once ingest has been quiet for AUTO_QUIET_NS, one worker builds the graph
     on a background pthread (build_index_from_shared, the same build
     FT.OPTIMIZE runs), while every worker keeps answering exactly.
  3. That worker publishes the graph from its own event loop (filter metadata
     and BM25 read its keyspace), then the thread saves it to disk.
  4. The ingest buffer is KEPT. Vectors written after the build's snapshot
     are the delta: FT.SEARCH merges an exact scan of them into the graph's
     results, so a write after the build is searchable at once. When the delta
     reaches 1/AUTO_REBUILD_DIVISOR of the index and ingest is quiet again,
     the graph is rebuilt the same way; searches are exact while it builds.

An explicit FT.OPTIMIZE keeps its meaning: build now, from everything
ingested, and free the buffer (later HSETs are stored, not indexed, and
counted in INFO vector_unindexed_writes).

Who builds. `index_owner` is the worker whose graph holds the published
index; only it rebuilds, so a build never touches buffers another worker's
graph owns. Before the first publish, the first worker to claim `auto_state`
builds and becomes the owner.

The state machine, in SharedHNSWView.auto_state as `(worker << 8) | phase`:

  IDLE -> BUILDING  the worker claims it (CAS), unpublishes a served index and
                    waits out its readers, resets its graph, starts the thread
  BUILDING -> BUILT the thread finished the build (or FAILED)
  BUILT -> SAVING   the worker published the graph (vector_auto_tick, or a
                    vector_auto_await on that worker)
  SAVING -> DONE    the thread saved pion.hnsw.0
  DONE -> IDLE      the worker joined the thread

Commands that would change the index, or the graph under the thread, wait for
IDLE first: FT.CREATE, FT.DROPINDEX and FT.OPTIMIZE on any worker, which then
hold the state (AUTO_HOLD) while they run so no build starts under them
(vector_auto_hold / vector_auto_release), and RAG.QUERY on the building worker
(vector_auto_await). Waiting workers count themselves in `auto_waiters` and
no build starts while any wait. The building worker never waits for another
worker during a build, so a wait cannot deadlock: it ends at the building
worker's next housekeeping tick after the thread is done.
"""

from std.memory import alloc
from std.memory.unsafe_pointer import UnsafePointer
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.time import perf_counter_ns
from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.hash_map import StripedHashMap
from src.common.vec_tomb import VecTomb
from src.vector.hnsw import HNSWGraph, SharedHNSWView
from src.network.dispatcher import CommandDispatcher
from src.network.vector_ingest import record_all_dead
from src.commands.vector import populate_hnsw_metadata, adopt_index_config, wait_for_readers

comptime AUTO_IDLE = 0
comptime AUTO_BUILDING = 1
comptime AUTO_BUILT = 2
comptime AUTO_SAVING = 3
comptime AUTO_DONE = 4
comptime AUTO_FAILED = 5
comptime AUTO_HOLD = 6       # FT.CREATE / FT.DROPINDEX / FT.OPTIMIZE running (vector_auto_hold)

# Ingest counts as over once no vector has arrived for this long. Housekeeping
# looks every ~64 ms, so a build starts 200-270 ms after the last vector.
comptime AUTO_QUIET_NS = UInt64(200_000_000)
# A built index is rebuilt once its delta is this fraction of it (1/10): the
# exact delta scan then costs about a tenth of a full exact scan.
comptime AUTO_REBUILD_DIVISOR = 10


struct VectorAuto(Movable):
    """One worker's side of the automatic build."""
    var seen: Int             # ingest count at the last tick
    var since_ns: UInt64      # when it last changed
    var job: UnsafePointer[NoneType, MutUntrackedOrigin]   # pion_bg_start_sym handle
    var ctx: UnsafePointer[Int64, MutUntrackedOrigin]      # the thread's arguments
    var snapshot: Int         # rows the running build covers
    var failed_at: Int        # ingest count a build failed at: not retried until it moves
    var t0_ns: UInt64

    def __init__(out self):
        self.seen = -1
        self.since_ns = 0
        self.job = null_ptr[NoneType, MutUntrackedOrigin]()
        self.ctx = null_ptr[Int64, MutUntrackedOrigin]()
        self.snapshot = 0
        self.failed_at = -1
        self.t0_ns = 0


@always_inline
def auto_phase(st: UInt64) -> Int:
    return Int(st & 0xFF)


@always_inline
def auto_worker(st: UInt64) -> Int:
    return Int(st >> 8)


@always_inline
def auto_load(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]) -> UInt64:
    if is_null(shared) or is_null(shared[].auto_state):
        return 0
    return Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared[].auto_state, UInt64(0))


def auto_building_here(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin], worker_id: Int) -> Bool:
    """A background build of THIS worker's graph is running or unfinished."""
    var st = auto_load(shared)
    return auto_phase(st) != AUTO_IDLE and auto_worker(st) == worker_id


def _set(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin], worker_id: Int, phase: Int):
    var v = UInt64(0) if phase == AUTO_IDLE else (UInt64(worker_id) << 8) | UInt64(phase)
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shared[].auto_state, v)


def vector_auto_tick(mut hnsw: HNSWGraph,
                     shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                     keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                     mut dispatcher: CommandDispatcher,
                     tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                     worker_id: Int,
                     mut st: VectorAuto):
    """Every housekeeping tick (engine.mojo): move this worker's build along,
    or start one when ingest has gone quiet with vectors no graph holds."""
    if is_null(shared) or is_null(shared[].auto_state) or is_null(shared[].ingest_count):
        return
    var cur = auto_load(shared)
    if auto_phase(cur) != AUTO_IDLE:
        if auto_worker(cur) == worker_id:
            _advance(hnsw, shared, keyspace, dispatcher, tomb, worker_id, st, False)
        return
    if not shared[].pre_index_ready or is_null(shared[].ingest_fp32):
        st.seen = -1
        return
    var rows = shared[].ingested()
    var now = UInt64(perf_counter_ns())
    if rows != st.seen:
        st.seen = rows
        st.since_ns = now
        return
    if now - st.since_ns < AUTO_QUIET_NS or rows == st.failed_at:
        return
    var ready = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared[].ready_atomic, UInt64(0)) \
        if is_not_null(shared[].ready_atomic) else UInt64(0)
    var built = shared[].num_nodes if ready != 0 else 0
    var delta = rows - built
    if delta <= 0:
        return
    if built > 0 and delta * AUTO_REBUILD_DIVISOR < built:
        return              # small delta: the exact delta scan is cheap
    if shared[].index_owner >= 0 and shared[].index_owner != worker_id:
        return              # only the owner's graph is rebuilt
    if is_not_null(shared[].auto_waiters) and Atomic[Scalar[DType.uint64]].fetch_add[
            ordering=Ordering.ACQUIRE](shared[].auto_waiters, UInt64(0)) != 0:
        return
    var expected = UInt64(0)
    if not Atomic[Scalar[DType.uint64]].compare_exchange(
            shared[].auto_state, expected, (UInt64(worker_id) << 8) | UInt64(AUTO_BUILDING)):
        return
    _start(hnsw, shared, worker_id, rows, ready, st)


def _start(mut hnsw: HNSWGraph,
           shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
           worker_id: Int, rows: Int, ready: UInt64, mut st: VectorAuto):
    """On this worker's loop, holding auto_state = BUILDING."""
    # Ingest was quiet, so every slot of the snapshot should be written; a
    # slot that is not (a writer stalled mid-HSET) postpones the build.
    external_call["pion_full_fence", NoneType]()
    for s in range(rows):
        if not shared[].row_visible(s):
            _set(shared, worker_id, AUTO_IDLE)
            return
    # The served index is this graph (only the owner rebuilds): unpublish it
    # and wait out its readers before the build rewrites and frees it.
    # Every worker answers exactly from the ingest buffer meanwhile.
    if ready != 0:
        shared[].ready = False
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shared[].ready_atomic, UInt64(0))
        wait_for_readers(shared, worker_id)
    hnsw.reset_graph()          # gives back a borrow; BM25 keeps serving
    if shared[].index_name_len > 0:
        hnsw.index_name_len = shared[].index_name_len
        for b in range(hnsw.index_name_len):
            hnsw.index_name[b] = shared[].index_name[b]
    adopt_index_config(hnsw, shared)
    # #46: a fresh id per build, as FT.OPTIMIZE gives one
    var bid = UInt64(perf_counter_ns()) ^ (UInt64(Int(external_call["random", Int64]())) << 20)
    hnsw.build_id = bid if bid != 0 else UInt64(1)
    var ctx = alloc[Int64](4)
    ctx[0] = Int64(Int(UnsafePointer(to=hnsw)))
    ctx[1] = Int64(Int(shared))
    ctx[2] = Int64(rows)
    ctx[3] = Int64(worker_id)
    var sym = String("pion_vector_auto_build") + "\0"
    var job = external_call["pion_bg_start_sym", UnsafePointer[NoneType, MutUntrackedOrigin]](
        sym.unsafe_ptr(), ctx)
    if is_null(job):
        print("[vector] auto-build could not start a thread; FT.SEARCH stays exact (FT.OPTIMIZE builds)")
        ctx.free()
        st.failed_at = rows
        _set(shared, worker_id, AUTO_IDLE)
        return
    st.job = job
    st.ctx = ctx
    st.snapshot = rows
    st.t0_ns = UInt64(perf_counter_ns())
    print("[vector] auto-build: indexing " + String(rows)
          + " vectors in the background; FT.SEARCH answers exactly until it is done")


def _publish(mut hnsw: HNSWGraph,
             shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
             keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
             mut dispatcher: CommandDispatcher,
             tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
             worker_id: Int, st: VectorAuto):
    """BUILT -> SAVING, on this worker's loop: what FT.OPTIMIZE does after its
    build, minus freeing the ingest buffer (the delta lives there)."""
    populate_hnsw_metadata(hnsw, shared, keyspace)
    hnsw.index_ready = True
    hnsw.publish_to_shared(shared, worker_id)
    record_all_dead(shared, tomb, dispatcher)
    if is_not_null(shared[].vec_stats):
        _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELAXED](shared[].vec_stats + 1, UInt64(1))
    var ms = (UInt64(perf_counter_ns()) - st.t0_ns) // 1_000_000
    print("[vector] auto-build: " + String(hnsw.num_nodes) + " vectors indexed in "
          + String(Int(ms)) + " ms; FT.SEARCH uses the graph now")
    _set(shared, worker_id, AUTO_SAVING)


def _advance(mut hnsw: HNSWGraph,
             shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
             keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
             mut dispatcher: CommandDispatcher,
             tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
             worker_id: Int, mut st: VectorAuto, block: Bool):
    """This worker's build, one step (or, with `block`, to the end)."""
    while True:
        var ph = auto_phase(auto_load(shared))
        if ph == AUTO_IDLE:
            return
        if ph == AUTO_BUILT:
            _publish(hnsw, shared, keyspace, dispatcher, tomb, worker_id, st)
            continue
        if ph == AUTO_DONE or ph == AUTO_FAILED:
            if is_not_null(st.job):
                external_call["pion_bg_join", NoneType](st.job)
            st.job = null_ptr[NoneType, MutUntrackedOrigin]()
            if is_not_null(st.ctx):
                st.ctx.free()
            st.ctx = null_ptr[Int64, MutUntrackedOrigin]()
            if ph == AUTO_FAILED:
                print("[vector] auto-build of " + String(st.snapshot)
                      + " vectors failed; FT.SEARCH stays exact (FT.OPTIMIZE builds)")
                st.failed_at = st.snapshot
                hnsw.reset_graph()
            _set(shared, worker_id, AUTO_IDLE)
            return
        if ph != AUTO_BUILDING and ph != AUTO_SAVING:
            return          # AUTO_HOLD: a command of this worker's, not a build
        # BUILDING or SAVING: the thread's turn
        if not block:
            return
        _ = external_call["usleep", Int32](Int32(500))


def vector_auto_await(mut hnsw: HNSWGraph,
                      shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                      keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                      mut dispatcher: CommandDispatcher,
                      tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                      worker_id: Int,
                      mut st: VectorAuto):
    """Return once no automatic build is running: before FT.CREATE,
    FT.DROPINDEX and FT.OPTIMIZE (they change the index the build reads and
    publishes), and before RAG.QUERY on the building worker. On the building
    worker this finishes the build here; elsewhere it waits for that worker."""
    var cur = auto_load(shared)
    if auto_phase(cur) == AUTO_IDLE:
        return
    if auto_worker(cur) == worker_id:
        _advance(hnsw, shared, keyspace, dispatcher, tomb, worker_id, st, True)
        return
    if is_null(shared[].auto_waiters):
        return
    _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](shared[].auto_waiters, UInt64(1))
    while auto_phase(auto_load(shared)) != AUTO_IDLE:
        _ = external_call["usleep", Int32](Int32(1000))
    _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](shared[].auto_waiters, UInt64.MAX)


def auto_worker_of(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]) -> Int:
    """The worker whose graph an unfinished automatic build belongs to (-1: none)."""
    var st = auto_load(shared)
    return auto_worker(st) if auto_phase(st) != AUTO_IDLE else -1


def vector_auto_hold(mut hnsw: HNSWGraph,
                     shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                     keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                     mut dispatcher: CommandDispatcher,
                     tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                     worker_id: Int,
                     mut st: VectorAuto):
    """For FT.CREATE, FT.DROPINDEX and FT.OPTIMIZE: wait out any automatic
    build, then hold the state (AUTO_HOLD) until vector_auto_release, so no
    build starts on another worker while the command frees or rebuilds what
    a build reads. Waiting alone left a window: the owner's tick could claim
    a build between the wait and the command's own work."""
    if is_null(shared) or is_null(shared[].auto_state):
        return
    while True:
        vector_auto_await(hnsw, shared, keyspace, dispatcher, tomb, worker_id, st)
        var expected = UInt64(0)
        if Atomic[Scalar[DType.uint64]].compare_exchange(
                shared[].auto_state, expected, (UInt64(worker_id) << 8) | UInt64(AUTO_HOLD)):
            return


def vector_auto_release(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin], worker_id: Int):
    """End this worker's vector_auto_hold."""
    var st = auto_load(shared)
    if auto_phase(st) == AUTO_HOLD and auto_worker(st) == worker_id:
        _set(shared, worker_id, AUTO_IDLE)
