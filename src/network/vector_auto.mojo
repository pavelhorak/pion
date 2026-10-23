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

A restart keeps each meaning. The index file records an automatic build
(word 25, bit 8). Loading one reopens ingest (reopen_after_load), and before
any worker serves, every worker puts back the FP32 rows of its own hashes
and adds the hashes the graph does not hold as the delta
(vector_restart_refill). The file holds the graph, not those rows. A file
from FT.OPTIMIZE, or from before 1.0.2, loads with ingest closed, as before,
and says so; its later HSETs are counted as unindexed.

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

from std.memory import alloc, unsafe_memcpy
from std.memory.unsafe_pointer import UnsafePointer
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.time import perf_counter_ns
from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.hash_map import StripedHashMap
from src.common.vec_tomb import VecTomb
from src.vector.hnsw import HNSWGraph, SharedHNSWView
from src.vector.hnsw_types import PREFIX_MAX, PREFIX_SLOT
from src.network.dispatcher import CommandDispatcher
from src.io.wal import WAL
from src.network.vector_ingest import record_all_dead, backfill_index
from src.commands.vector import populate_hnsw_metadata, adopt_index_config, wait_for_readers, _close_ingest

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
    var backfill_seen: UInt64 # SharedHNSWView.backfill_gen when this worker last backfilled

    def __init__(out self):
        self.seen = -1
        self.since_ns = 0
        self.job = null_ptr[NoneType, MutUntrackedOrigin]()
        self.ctx = null_ptr[Int64, MutUntrackedOrigin]()
        self.snapshot = 0
        self.failed_at = -1
        self.t0_ns = 0
        self.backfill_seen = 0


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
    # An FT.CREATE on another worker asked every worker to index the hashes
    # its keyspace already holds (vector_backfill_now). Housekeeping runs
    # between dispatch batches, where wait_for_readers counts this worker as
    # idle, so the backfill holds the state instead (AUTO_HOLD): FT.CREATE,
    # FT.DROPINDEX and FT.OPTIMIZE on another worker wait for it before they
    # free the ingest buffer it writes into, and no build starts under it.
    # When the state is taken (a build, or another worker's command), it waits
    # for a later tick rather than for that worker, so it cannot deadlock.
    if is_not_null(shared[].backfill_gen):
        var bg = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared[].backfill_gen, UInt64(0))
        if bg != st.backfill_seen:
            var idle = UInt64(0)
            if Atomic[Scalar[DType.uint64]].compare_exchange(
                    shared[].auto_state, idle, (UInt64(worker_id) << 8) | UInt64(AUTO_HOLD)):
                st.backfill_seen = bg
                _backfill(shared, keyspace, dispatcher.wal, tomb, worker_id)
                _set(shared, worker_id, AUTO_IDLE)
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


def vector_backfill_now(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                        keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                        wal: UnsafePointer[WAL, MutUntrackedOrigin],
                        tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                        worker_id: Int, mut st: VectorAuto):
    """After an FT.CREATE that opened ingest: index the hashes already in this
    worker's keyspace now, before the +OK leaves, and ask every other worker to
    index its own at its next housekeeping tick (vector_auto_tick). The bump's
    release half publishes the index config FT.CREATE just wrote. Called inside
    the FT.CREATE's vector_auto_hold, which keeps the ingest buffer alive."""
    if is_null(shared) or is_null(shared[].backfill_gen):
        return
    var prev = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](
        shared[].backfill_gen, UInt64(1))
    st.backfill_seen = prev + UInt64(1)
    _backfill(shared, keyspace, wal, tomb, worker_id)


def _backfill(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
              keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
              wal: UnsafePointer[WAL, MutUntrackedOrigin],
              tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
              worker_id: Int):
    var t0 = UInt64(perf_counter_ns())
    var n = backfill_index(shared, keyspace, wal, tomb)
    if n > 0:
        print("[vector] FT.CREATE: indexed " + String(n) + " existing hashes from worker "
              + String(worker_id) + "'s keyspace in "
              + String(Int((UInt64(perf_counter_ns()) - t0) // 1_000_000)) + " ms")


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
    hnsw.auto_built = True      # its index file reopens ingest after a restart
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


def reopen_after_load(hnsw: HNSWGraph,
                      shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                      worker_id: Int):
    """Restart: `hnsw` was just warm-loaded and published (state.mojo phase 6,
    on the loading worker, before the others stop waiting for the load).

    An automatic build's index reopens ingest as that build left it: HSETs are
    indexed again, the graph holds slots [0, N), later ones are the delta, and
    the index rebuilds itself as before. Every worker then refills the rows
    (vector_restart_refill) before any serves.

    Any other index (FT.OPTIMIZE ended its ingest, or its file predates the
    flag) keeps ingest closed, as before. It still takes the index config, so
    a vector written to it is counted as unindexed (INFO
    vector_unindexed_writes) rather than dropped without a trace, and the log
    says so: a restart used to look like this from the outside with no sign."""
    if is_null(shared) or is_null(shared[].ingest_count) or is_not_null(shared[].ingest_fp32):
        return
    # The index's PREFIX list (the file's; none for a file from before 1.0.3).
    # Written here, before any worker serves, and otherwise only by FT.CREATE,
    # which stops HSET routing first: every worker's ingest reads it.
    if is_not_null(shared[].pre_prefixes):
        unsafe_memcpy(dest=shared[].pre_prefixes, src=hnsw.prefixes, count=PREFIX_MAX * PREFIX_SLOT)
        shared[].pre_prefix_count = hnsw.prefix_count
    shared[].pre_dim = hnsw.dim
    shared[].pre_ef_construction = hnsw.ef_construction
    shared[].pre_distance_metric = hnsw.distance_metric
    var name = String("")
    for b in range(hnsw.index_name_len):
        name += chr(Int(hnsw.index_name[b]))        # a log line: a name is ASCII in practice
    if hnsw.auto_built and is_not_null(shared[].restart_refill):
        shared[].open_ingest(hnsw.max_elements, hnsw.dim)
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
            shared[].ingest_count, UInt64(hnsw.num_nodes))
        shared[].restart_refill[1] = 1
        external_call["pion_full_fence", NoneType]()
        shared[].pre_index_ready = True
        print("[vector] index '" + name + "' loaded (" + String(hnsw.num_nodes)
              + " vectors, built automatically): it keeps indexing; refilling its rows from the hashes")
        return
    shared[].pre_index_ready = True
    print("[vector] index '" + name + "' loaded (" + String(hnsw.num_nodes)
          + " vectors) with ingest closed (FT.OPTIMIZE built it, or its file predates 1.0.2):"
          + " a vector written from now on is stored, not indexed, and counted in INFO"
          + " vector_unindexed_writes. FT.DROPINDEX then FT.CREATE indexes every hash.")


def vector_restart_refill(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                          keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                          wal: UnsafePointer[WAL, MutUntrackedOrigin],
                          tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                          worker_id: Int):
    """Restart, on every worker after restore_index_state (which refilled the
    rows of the graph's documents this worker holds): add this worker's
    hashes the graph does not hold (written after its build, or rewritten
    since) as the delta, as FT.CREATE indexes existing hashes. Then wait for
    every worker. The last one to get here declares the graph's rows nobody
    refilled dead (their documents are gone), or, when none was refilled at
    all, closes ingest and serves the index as loaded. No worker serves
    before that: a search reads those rows (the delta merge rescores the
    graph's results from them), and a rebuild builds from them."""
    var rr = shared[].restart_refill if is_not_null(shared) else null_ptr[UInt64, MutUntrackedOrigin]()
    if is_null(rr) or Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](rr + 1, UInt64(0)) == 0:
        return
    var t0 = UInt64(perf_counter_ns())
    var added = backfill_index(shared, keyspace, wal, tomb)
    if added > 0:
        print("[vector] restart: worker " + String(worker_id) + " indexed " + String(added)
              + " vectors the graph does not hold, in "
              + String(Int((UInt64(perf_counter_ns()) - t0) // 1_000_000)) + " ms")
    var before = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](rr, UInt64.MAX)
    if before == 1:
        _finish_refill(shared, worker_id)
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](rr + 1, UInt64(0))
        return
    var waited_ms = 0
    while Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](rr + 1, UInt64(0)) != 0:
        _ = external_call["usleep", Int32](Int32(1000))
        waited_ms += 1
        if waited_ms >= 600_000:
            # A worker that died never arrives; serving beats never serving.
            print("Worker " + String(worker_id) + ": gave up waiting for the vector index refill after 600 s")
            break


def _finish_refill(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin], worker_id: Int):
    """The last worker's half of vector_restart_refill."""
    external_call["pion_full_fence", NoneType]()
    var n = shared[].num_nodes
    var refilled = 0
    for s in range(n):
        if shared[].row_visible(s):
            refilled += 1
    if refilled == 0 and n > 0 and shared[].ingested() <= n:
        # None of the index's documents is in the keyspace (a server run with
        # --no-wal, or the index file copied without its data): nothing to
        # rebuild from. Serve the index as loaded, as a warm load always did,
        # with ingest closed as FT.OPTIMIZE leaves it.
        _close_ingest(shared, worker_id)
        print("[vector] restart: none of the index's " + String(n) + " documents is in the keyspace;"
              + " it serves as loaded with ingest closed: a vector written from now on is stored, not"
              + " indexed, and counted in INFO vector_unindexed_writes")
        return
    var gone = 0
    for s in range(n):
        if shared[].row_visible(s):
            continue
        shared[].clear_ingest_row(s)
        shared[].mark_dead(s)
        shared[].commit_ingest(s)
        gone += 1
    print("[vector] restart: " + String(refilled) + " of the graph's rows refilled"
          + (", " + String(gone) + " documents gone" if gone > 0 else String(""))
          + "; " + String(shared[].ingested() - n) + " vectors are the delta; the index keeps indexing")
