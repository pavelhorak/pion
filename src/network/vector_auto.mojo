"""Drop-in: the vector index builds itself, and rebuilds without a gap.

A stock Redis client (RedisVL, LangChain, VectorDBBench's own) creates an
index, HSETs its documents and searches. It never sends FT.OPTIMIZE, Pion's
"ingest is over, build the graph" command, so it used to get nothing back.
Now:

  1. Until a graph is built, FT.SEARCH answers exactly from the shared ingest
     buffer (vector.mojo exact_knn): same reply, same score units.
  2. Once ingest has been quiet for AUTO_QUIET_NS (or AUTO_FIRST_ROWS vectors
     wait while searches run, pause or not), one worker builds the graph on a background
     pthread (build_index_from_shared, the same build FT.OPTIMIZE runs),
     while every worker keeps answering exactly.
  3. The build goes into a REBUILD GRAPH (vector.mojo auto_graph), not into
     a worker's own graph. That worker publishes it from its own event loop,
     and every worker, the builder too, searches it through a borrow. Then
     the thread saves it to disk.
  4. The ingest buffer is KEPT. Vectors written after the build's snapshot
     are the delta: FT.SEARCH merges an exact scan of them into the graph's
     results, so a write after the build is searchable at once.
  5. Phase B: when the delta reaches 1/AUTO_REBUILD_DIVISOR of the index or
     AUTO_DELTA_MAX rows (no pause needed while searches run), or ingest pauses for
     AUTO_SETTLE_MS with a delta of 1/AUTO_SETTLE_DIVISOR, the index is
     rebuilt into the OTHER rebuild graph. The served index, graph
     plus exact delta, keeps answering all the while: the new one replaces it
     at publish, then the builder waits out every reader of the old one
     (wait_for_readers) and frees it. Searches are never exact-only again.

An explicit FT.OPTIMIZE keeps its meaning: build now, from everything
ingested, into the worker's own graph, and free the buffer (later HSETs are
stored, not indexed, and counted in INFO vector_unindexed_writes).

Who builds. `index_owner` is the worker that published the served index;
only it rebuilds, so the per-worker state a publish makes (filter metadata,
BM25: populate_hnsw_metadata) stays on one worker. Before the first publish,
the first worker to claim `auto_state` builds and becomes the owner.

The state machine, in SharedHNSWView.auto_state as `(worker << 8) | phase`:

  IDLE -> BUILDING  the worker claims it (CAS), resets the rebuild graph not
                    served, starts the thread
  BUILDING -> BUILT the thread finished the build (or FAILED)
  BUILT -> SAVING   the worker published the graph and retired the one it
                    replaced (vector_auto_tick, or a vector_auto_await there)
  SAVING -> DONE    the thread saved pion.hnsw.0
  DONE -> IDLE      the worker joined the thread

Commands that would change the index, or the graphs under the thread, wait
for IDLE first: FT.CREATE, FT.DROPINDEX and FT.OPTIMIZE on any worker, which
then hold the state (AUTO_HOLD) while they run so no build starts under them
(vector_auto_hold / vector_auto_release). Waiting workers count themselves in
`auto_waiters` and no build starts while any wait. The building worker never
waits for another worker during a build, so a wait cannot deadlock: it ends
at the building worker's next housekeeping tick after the thread is done.
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
from src.network.vector_ingest import record_all_dead, ingest_unlinked_hashes
from src.io.wal import WAL
from src.commands.vector import populate_hnsw_metadata, adopt_index_config, wait_for_readers, \
    auto_graph, make_auto_graph, served_auto_graph, retire_auto_graphs, _close_ingest

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
# The first build starts without a pause once this many vectors wait and
# searches are running: a stream that never pauses is otherwise searched
# exactly forever (an exact query over 16K rows is ~1.5 ms).
comptime AUTO_FIRST_ROWS = 16384
# "While searches pay for exact scans": an exact scan ran this recently
# (SharedHNSWView.exact_at). The no-pause triggers above and below need it.
comptime AUTO_SEARCHED_NS = UInt64(1_000_000_000)
# A built index is rebuilt once its delta (the rows every query scans
# exactly) reaches 1/AUTO_REBUILD_DIVISOR of it, but never more than
# AUTO_DELTA_MAX rows (PION_VECTOR_DELTA_MAX): past that the scan costs a
# query more than its graph search does, whatever the index's size. After a
# 200 ms pause in the inserts, or with none while searches are running.
comptime AUTO_REBUILD_DIVISOR = 10
comptime AUTO_DELTA_MIN = 256
comptime AUTO_DELTA_MAX = 8192
# Once ingest pauses for AUTO_SETTLE_MS (PION_VECTOR_SETTLE_MS; 0 = never),
# a delta of 1/AUTO_SETTLE_DIVISOR of the index or more is folded in too, so
# what a stream leaves behind does not cost every later query a scan. 3 s, not
# 1: VectorDBBench's stream sends one burst a second, and gaps just over 1 s
# set off 13-14 extra rebuilds per 50K stream (S50, Mac, 2026-10-19).
comptime AUTO_SETTLE_MS = 3000
comptime AUTO_SETTLE_DIVISOR = 100


struct VectorAuto(Movable):
    """One worker's side of the automatic build."""
    var seen: Int             # ingest count at the last tick
    var since_ns: UInt64      # when it last changed
    var job: UnsafePointer[NoneType, MutUntrackedOrigin]   # pion_bg_start_sym handle
    var ctx: UnsafePointer[Int64, MutUntrackedOrigin]      # the thread's arguments
    var snapshot: Int         # rows the running build covers
    var target: Int           # the rebuild graph it builds into (auto_graph)
    var failed_at: Int        # ingest count a build failed at: not retried until it moves
    var t0_ns: UInt64
    var delta_max: Int        # AUTO_DELTA_MAX, or PION_VECTOR_DELTA_MAX
    var settle_ns: UInt64     # AUTO_SETTLE_MS, or PION_VECTOR_SETTLE_MS (0: never)

    def __init__(out self):
        self.seen = -1
        self.since_ns = 0
        self.job = null_ptr[NoneType, MutUntrackedOrigin]()
        self.ctx = null_ptr[Int64, MutUntrackedOrigin]()
        self.snapshot = 0
        self.target = 0
        self.failed_at = -1
        self.t0_ns = 0
        var n1 = String("PION_VECTOR_DELTA_MAX") + "\0"
        var dm = Int(external_call["pion_env_int", Int64](n1.unsafe_ptr(), Int64(AUTO_DELTA_MAX)))
        self.delta_max = dm if dm >= 1 else AUTO_DELTA_MAX
        var n2 = String("PION_VECTOR_SETTLE_MS") + "\0"
        var sm = Int(external_call["pion_env_int", Int64](n2.unsafe_ptr(), Int64(AUTO_SETTLE_MS)))
        self.settle_ns = UInt64(sm) * 1_000_000 if sm > 0 else UInt64(0)


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
    var quiet_ns = now - st.since_ns
    if rows == st.failed_at:
        return
    var ready = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared[].ready_atomic, UInt64(0)) \
        if is_not_null(shared[].ready_atomic) else UInt64(0)
    var built = shared[].num_nodes if ready != 0 else 0
    var delta = rows - built
    if delta <= 0:
        return
    # Without a pause in the inserts, build only while searches are paying for
    # exact scans: a bulk load nobody queries yet waits for its pause, as
    # building at 16K, 23K and 35K rows of a 50K load slowed the load by 45%
    # and made the last build later, not earlier (K50, Mac, 2026-10-19).
    var searched = False
    if is_not_null(shared[].exact_at):
        var at = shared[].exact_at.load[volatile=True]()
        searched = at != 0 and (at >= now or now - at < AUTO_SEARCHED_NS)
    var quiet = quiet_ns >= AUTO_QUIET_NS
    var due: Bool
    if built == 0:
        due = quiet or (rows >= AUTO_FIRST_ROWS and searched)
    else:
        var cap = max(AUTO_DELTA_MIN, min(built // AUTO_REBUILD_DIVISOR, st.delta_max))
        due = (delta >= cap and (quiet or searched)) or (st.settle_ns > 0 and quiet_ns >= st.settle_ns
                                                         and delta * AUTO_SETTLE_DIVISOR >= built)
    if not due:
        return
    if shared[].index_owner >= 0 and shared[].index_owner != worker_id:
        return              # only the owner rebuilds (per-worker publish state)
    if is_not_null(shared[].auto_waiters) and Atomic[Scalar[DType.uint64]].fetch_add[
            ordering=Ordering.ACQUIRE](shared[].auto_waiters, UInt64(0)) != 0:
        return
    # The snapshot is the committed prefix: every slot in it is written (an
    # HSET mid-flight still holds the next one). Rows past it are the delta.
    external_call["pion_full_fence", NoneType]()
    var snap = built
    while snap < rows and shared[].row_visible(snap):
        snap += 1
    if snap <= built:
        return
    var expected = UInt64(0)
    if not Atomic[Scalar[DType.uint64]].compare_exchange(
            shared[].auto_state, expected, (UInt64(worker_id) << 8) | UInt64(AUTO_BUILDING)):
        return
    _start(hnsw, shared, worker_id, snap, ready, st)


def _start(mut hnsw: HNSWGraph,
           shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
           worker_id: Int, rows: Int, ready: UInt64, mut st: VectorAuto):
    """On this worker's loop, holding auto_state = BUILDING: build slots
    [0, rows) into the rebuild graph that is not served. The served index is
    left alone; FT.SEARCH keeps answering from it (and its exact delta)."""
    var t = 1 if served_auto_graph(shared, ready) == 0 else 0
    var g = make_auto_graph(shared, t, hnsw)
    if is_null(g):
        _set(shared, worker_id, AUTO_IDLE)
        return
    g[].reset_graph()           # what it held was retired at the last publish
    g[].auto_built = True       # its index file reopens ingest after a restart
    if shared[].index_name_len > 0:
        g[].index_name_len = shared[].index_name_len
        for b in range(g[].index_name_len):
            g[].index_name[b] = shared[].index_name[b]
    elif hnsw.index_name_len > 0:
        g[].index_name_len = hnsw.index_name_len
        for b in range(hnsw.index_name_len):
            g[].index_name[b] = hnsw.index_name[b]
    adopt_index_config(g[], shared)
    g[].ef_runtime = hnsw.ef_runtime
    # #46: a fresh id per build, as FT.OPTIMIZE gives one
    var bid = UInt64(perf_counter_ns()) ^ (UInt64(Int(external_call["random", Int64]())) << 20)
    g[].build_id = bid if bid != 0 else UInt64(1)
    var ctx = alloc[Int64](4)
    ctx[0] = Int64(Int(g))
    ctx[1] = Int64(Int(shared))
    ctx[2] = Int64(rows)
    ctx[3] = Int64(worker_id)
    var sym = String("pion_vector_auto_build") + "\0"
    var job = external_call["pion_bg_start_sym", UnsafePointer[NoneType, MutUntrackedOrigin]](
        sym.unsafe_ptr(), ctx)
    if is_null(job):
        print("[vector] auto-build could not start a thread; FT.SEARCH stays as it is (FT.OPTIMIZE builds)")
        ctx.free()
        st.failed_at = rows
        _set(shared, worker_id, AUTO_IDLE)
        return
    st.job = job
    st.ctx = ctx
    st.snapshot = rows
    st.target = t
    st.t0_ns = UInt64(perf_counter_ns())
    if ready != 0:
        print("[vector] auto-build: re-indexing " + String(rows)
              + " vectors in the background; the current index keeps serving")
    else:
        print("[vector] auto-build: indexing " + String(rows)
              + " vectors in the background; FT.SEARCH answers exactly until it is done")


def _publish(mut hnsw: HNSWGraph,
             shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
             keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
             mut dispatcher: CommandDispatcher,
             tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
             worker_id: Int, st: VectorAuto):
    """BUILT -> SAVING, on this worker's loop: publish the new graph, retire
    the index it replaces, and make this worker's own graph a borrower of it.
    What FT.OPTIMIZE does after its build, minus freeing the ingest buffer
    (the delta lives there)."""
    var g = auto_graph(shared, st.target)
    g[].index_ready = True
    g[].publish_to_shared(shared, worker_id)
    record_all_dead(shared, tomb, dispatcher)
    if is_not_null(shared[].vec_stats):
        _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELAXED](shared[].vec_stats + 1, UInt64(1))
    # Batches that started before the publish may still read the index it
    # replaced; any later one re-borrows (publish_seq moved on). After the
    # wait nothing reads the old one, so it can go: the other rebuild graph,
    # or this worker's own graph (an FT.OPTIMIZE build here, or a warm load).
    wait_for_readers(shared, worker_id)
    retire_auto_graphs(shared, st.target)
    if hnsw.is_borrowed:
        hnsw._unborrow()
    elif hnsw.num_nodes > 0 or is_not_null(hnsw.compact_buffer):
        hnsw.reset_graph()
    hnsw.borrow_stable(shared)
    # Filter metadata and BM25 for this worker (they are per worker)
    populate_hnsw_metadata(hnsw, shared, keyspace)
    var ms = (UInt64(perf_counter_ns()) - st.t0_ns) // 1_000_000
    print("[vector] auto-build: " + String(g[].num_nodes) + " vectors indexed in "
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
                      + " vectors failed; FT.SEARCH serves what it served (FT.OPTIMIZE builds)")
                st.failed_at = st.snapshot
                var fg = auto_graph(shared, st.target)
                if is_not_null(fg):
                    fg[].reset_graph()      # the served index is untouched
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
        # held elsewhere (another command, or a restart's refill: phase B)
        _ = external_call["usleep", Int32](Int32(500))


def vector_auto_release(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin], worker_id: Int):
    """End this worker's vector_auto_hold."""
    var st = auto_load(shared)
    if auto_phase(st) == AUTO_HOLD and auto_worker(st) == worker_id:
        _set(shared, worker_id, AUTO_IDLE)


def reopen_after_load(hnsw: HNSWGraph,
                      shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                      worker_id: Int):
    """Phase B, restart: `hnsw` was just warm-loaded from an automatic build's
    file and published. Reopen ingest as that build left it: HSETs are indexed
    again, the graph holds slots [0, N), later ones are the delta, and the
    index rebuilds itself as before. Called by the loading worker before the
    other workers stop waiting for the load (state.mojo phase 6).

    The file holds the graph, not the FP32 rows a rebuild and the exact
    rescore read. Every worker puts back the rows of its own hashes
    (restore_index_state) and adds the hashes no live slot holds
    (vector_auto_restored); the hold taken here keeps any build from starting
    until the last of them is done."""
    if is_null(shared) or not hnsw.auto_built or is_not_null(shared[].ingest_fp32) \
            or is_null(shared[].auto_restore) or is_null(shared[].ingest_count) or is_null(shared[].auto_state):
        return
    shared[].pre_dim = hnsw.dim
    shared[].pre_ef_construction = hnsw.ef_construction
    shared[].pre_distance_metric = hnsw.distance_metric
    shared[].open_ingest(hnsw.max_elements, hnsw.dim)
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shared[].ingest_count, UInt64(hnsw.num_nodes))
    _set(shared, worker_id, AUTO_HOLD)
    shared[].auto_restore[1] = 1
    external_call["pion_full_fence", NoneType]()
    shared[].pre_index_ready = True
    print("[vector] warm load of an automatic index (" + String(hnsw.num_nodes)
          + " vectors): ingest reopened; refilling its rows from the hashes")


def vector_auto_restored(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                         keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                         wal: UnsafePointer[WAL, MutUntrackedOrigin],
                         tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                         worker_id: Int):
    """Phase B, restart, on every worker after restore_index_state: add this
    worker's hashes no live slot holds as delta rows. The last worker to get
    here clears the graph's rows nobody refilled (their documents are gone:
    zeros, dead) and releases reopen_after_load's hold."""
    if is_null(shared) or is_null(shared[].auto_restore) or shared[].auto_restore[1] == 0:
        return
    var added = ingest_unlinked_hashes(shared, keyspace, wal, tomb)
    if added > 0:
        print("[vector] restart: " + String(added) + " vectors no built graph held are the delta again")
    var before = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE_RELEASE](
        shared[].auto_restore, UInt64.MAX)
    if before != 1:
        return
    external_call["pion_full_fence", NoneType]()
    var n = shared[].num_nodes
    var refilled = 0
    for s in range(n):
        if shared[].row_visible(s):
            refilled += 1
    if refilled == 0 and n > 0 and shared[].ingested() <= n:
        # None of the index's vectors is in the keyspace (a server run with
        # --no-wal, or the index file copied without its data): there is
        # nothing to rebuild from. Serve the index as loaded, as a warm load
        # always did, and close ingest as FT.OPTIMIZE does.
        shared[].pre_index_ready = False
        _close_ingest(shared, worker_id)
        shared[].auto_restore[1] = 0
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shared[].auto_state, UInt64(0))
        print("[vector] restart: none of the index's " + String(n) + " vectors is in the keyspace;"
              + " it serves as loaded, and later HSETs are stored but not indexed")
        return
    var cleared = 0
    for s in range(n):
        if shared[].row_visible(s):
            continue
        shared[].clear_ingest_row(s)
        shared[].mark_dead(s)
        shared[].commit_ingest(s)
        cleared += 1
    shared[].auto_restore[1] = 0
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](shared[].auto_state, UInt64(0))
    print("[vector] restart: rows of " + String(refilled) + " indexed vectors refilled"
          + (", " + String(cleared) + " gone" if cleared > 0 else String("")))
