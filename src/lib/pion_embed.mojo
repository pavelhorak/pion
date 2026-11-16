"""libpion: the Pion engine inside another process (embedded Pion, phase 0).

doc: embedded/README.md. Built by `pixi run build-embed`
into libpion.dylib / libpion.so. No server: no listen socket, no worker
threads, no signal handlers, no crash log. The host's threads call in under
one engine mutex; everything runs on the calling thread.

How it runs a command. The engine is set up exactly as `--io-threads` sets up
its executor (gh #465): the writer is in io_mode, so a reply never reaches a
socket; a flush queues it in the connection's output block, and pub/sub
deliveries and the replies of parked commands (WAIT, XREAD BLOCK, BLPOP) land
in the same per-connection queues. This library plays the I/O thread: it puts
a connection's request bytes in that connection's input buffer, runs
`_dispatch_io_batch` (the drain the executor runs), and takes what was queued
for the connection. A connection is a virtual id that indexes the engine's
per-fd tables; ids count down from 65535, so they do not collide with a real
descriptor (the engine never treats them as one in io_mode, and this library
never calls the paths that would close one).

C ABI (every export is abi("C") and never raises):
  pion_embed_open(cfg, len, err, err_cap) -> handle (0 on failure; err holds why)
      cfg is "key value" lines: dir (required), appendonly yes|no, wal-size <MB>,
      maxmemory <bytes>, maxmemory-policy <name>
  pion_embed_conn_new(h) -> conn id (< 0 on failure)
  pion_embed_conn_free(h, conn)
  pion_embed_exec(h, conn, req, len, *out, *out_len) -> status
  pion_embed_poll(h, conn, *out, *out_len) -> status
  pion_embed_tick(h)
  pion_embed_close(h) -> 0
status: 0 DONE (every byte consumed), 1 PARKED (a blocking command waits;
poll for its reply), 2 PARTIAL (an incomplete frame is kept for the next
call), 3 CLOSED (QUIT / CLIENT KILL: the connection is done), negative =
error: -1 engine error, -2 closed engine, -3 request too large, -4 bad
connection, -5 called from a forked child.
*out stays valid until the next call on that connection.
"""
from std.memory.alloc import unsafe_alloc
from std.memory import unsafe_memcpy, unsafe_memset
from std.ffi import external_call
from std.collections import List

from src.common.ptr import is_null, is_not_null, null_ptr
from src.common.zalloc import zalloc
from src.common.config import PionConfig
from src.engine.state import Pion
from src.vector.hnsw import SharedHNSWView, HNSWGraph
from src.common.heap import MinHeap as _GH199MinHeap, MaxHeap as _GH199MaxHeap
from std.atomic import Atomic, Ordering
from src.common.data_dir import data_path
from src.vector.hnsw_types import PREFIX_MAX, PREFIX_SLOT, hk_buf_bytes
from src.common.lock_free import ShardQueryBus
from src.network.cluster import ClusterState
from src.network.slow_path import SlowPathHandler
from src.commands.evict import evict_policy_from_string
from src.common.utils import parse_memory_value
from src.common.version import PION_VERSION, PION_BUILD_SHA
from src.network.fast_path import _get_now_ns

comptime EMBED_MAX_CONNS = 4096
comptime EMBED_TOP_ID = 65535
comptime EMBED_BUF = 256 * 1024 * 1024       # the engine's CLIENT_BUF_SIZE
comptime KQ_IO = Int32(-3)                    # run_server_io's kq: flushes queue for the "I/O thread"
comptime NO_LISTEN_FD = Int32(0x3FFFFFFF)     # >= 0, so Pion() does not listen; never used as an fd

comptime ST_DONE = Int64(0)
comptime ST_PARKED = Int64(1)
comptime ST_PARTIAL = Int64(2)
comptime ST_CLOSED = Int64(3)
comptime ST_ERR = Int64(-1)
comptime ST_ENGINE_CLOSED = Int64(-2)
comptime ST_TOO_LARGE = Int64(-3)
comptime ST_BAD_CONN = Int64(-4)
comptime ST_FORKED = Int64(-5)


struct EmbedState(Movable):
    var db: Pointer[Pion, MutUntrackedOrigin]
    var mutex: Pointer[NoneType, MutUntrackedOrigin]
    var lock_fd: Int32
    var pid: Int32
    var closed: Bool
    # per virtual connection, indexed by id: in use, and its staging buffer
    # for the reply bytes handed back to the caller
    var used: Pointer[UInt8, MutUntrackedOrigin]
    var out_bufs: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    var out_caps: Pointer[Int, MutUntrackedOrigin]
    var free_ids: List[Int]
    var next_id: Int
    var live: List[Int]               # ids in use, for the tick's resume pass
    var last_tick_ns: Int
    # wake fds (pion_embed_wake_fd), stored as fd + 1 so a zero page means
    # none; wake_set: a byte is in the pipe and not yet drained
    var wake_r: Pointer[Int32, MutUntrackedOrigin]
    var wake_w: Pointer[Int32, MutUntrackedOrigin]
    var wake_set: Pointer[UInt8, MutUntrackedOrigin]
    var wake_ids: List[Int]

    def __init__(out self):
        self.db = null_ptr[Pion, MutUntrackedOrigin]()
        self.mutex = null_ptr[NoneType, MutUntrackedOrigin]()
        self.lock_fd = -1
        self.pid = 0
        self.closed = False
        self.used = zalloc[UInt8](EMBED_TOP_ID + 1)
        self.out_bufs = zalloc[Pointer[UInt8, MutUntrackedOrigin]](EMBED_TOP_ID + 1)
        self.out_caps = zalloc[Int](EMBED_TOP_ID + 1)
        self.free_ids = List[Int]()
        self.next_id = EMBED_TOP_ID
        self.live = List[Int]()
        self.last_tick_ns = 0
        self.wake_r = zalloc[Int32](EMBED_TOP_ID + 1)
        self.wake_w = zalloc[Int32](EMBED_TOP_ID + 1)
        self.wake_set = zalloc[UInt8](EMBED_TOP_ID + 1)
        self.wake_ids = List[Int]()


comptime _St = Pointer[EmbedState, MutUntrackedOrigin]


@always_inline
def _st(h: Int64) -> _St:
    return _St(unsafe_from_address=Int(h))


@always_inline
def _now_ns() -> Int:
    return Int(_get_now_ns())


def _write_err(err: Pointer[UInt8, MutUntrackedOrigin], cap: Int64, msg: String):
    if is_null(err) or cap <= 0:
        return
    var n = msg.byte_length()
    if n > Int(cap) - 1:
        n = Int(cap) - 1
    unsafe_memcpy(dest=err, src=msg.unsafe_ptr(), count=n)
    err[unsafe_offset=n] = 0


def _cfg_get(cfg: String, key: String) -> String:
    """The value of `key` in "key value" lines (also "key=value"); "" if absent."""
    for line in cfg.split("\n"):
        var l = String(line).strip()
        if l.byte_length() == 0:
            continue
        var sep = -1
        var lb = l.as_bytes()
        for k in range(len(lb)):
            if lb[k] == 32 or lb[k] == 61:     # ' ' or '='
                sep = k
                break
        if sep < 0:
            continue
        if String(l[byte=0:sep]) == key:
            return String(String(l[byte=sep + 1:]).strip())
    return String("")


def _make_shared_hnsw(config: PionConfig) -> Pointer[SharedHNSWView, MutUntrackedOrigin]:
    """main()'s SharedHNSWView setup for one worker. KV profile: max_elements
    is 1, so the slot map and tombstones are a few bytes. TODO(phase 1): one
    function main() and this both call, so the two cannot drift."""
    var n_workers = 1
    var sh = unsafe_alloc[SharedHNSWView](1)
    sh.unsafe_write(SharedHNSWView())
    var ingest_count_buf = unsafe_alloc[UInt64](1)
    ingest_count_buf[unsafe_offset=0] = 0
    sh[].ingest_count = ingest_count_buf
    sh[].num_shards = 1
    var optimize_trigger_buf = unsafe_alloc[UInt64](1)
    optimize_trigger_buf[unsafe_offset=0] = 0
    sh[].optimize_trigger = optimize_trigger_buf
    var ready_atomic_buf = unsafe_alloc[UInt64](1)
    ready_atomic_buf[unsafe_offset=0] = 0
    sh[].ready_atomic = ready_atomic_buf
    var auto_cells = unsafe_alloc[UInt64](8)
    unsafe_memset(auto_cells.unsafe_bitcast[UInt8](), 0, 64)
    sh[].auto_state = auto_cells
    sh[].auto_waiters = auto_cells.unsafe_offset(1)
    sh[].publish_seq = auto_cells.unsafe_offset(2)
    sh[].vec_stats = auto_cells.unsafe_offset(4)
    sh[].backfill_gen = auto_cells.unsafe_offset(3)
    auto_cells[unsafe_offset=6] = UInt64(n_workers)
    sh[].restart_refill = auto_cells.unsafe_offset(6)
    var prefix_cells = unsafe_alloc[UInt8](PREFIX_MAX * PREFIX_SLOT)
    unsafe_memset(prefix_cells, 0, PREFIX_MAX * PREFIX_SLOT)
    sh[].pre_prefixes = prefix_cells
    var warm_pending_buf = unsafe_alloc[UInt64](1 + n_workers)
    warm_pending_buf[unsafe_offset=0] = 0
    warm_pending_buf[unsafe_offset=1] = 0      # KV only: no persisted index is loaded
    sh[].warm_load_pending = warm_pending_buf
    var reclaim_epoch_buf = unsafe_alloc[UInt64](1)
    reclaim_epoch_buf[unsafe_offset=0] = 1
    sh[].reclaim_epoch = reclaim_epoch_buf
    var worker_epoch_buf = unsafe_alloc[UInt64](8 * n_workers)
    unsafe_memset(worker_epoch_buf.unsafe_bitcast[UInt8](), 0, 8 * n_workers * 8)
    sh[].worker_epoch = worker_epoch_buf
    sh[].worker_epoch_slots = n_workers
    var hk_max = config.vector.max_elements
    var hk_keys_buf = unsafe_alloc[UInt8](hk_buf_bytes(hk_max))
    unsafe_memset(hk_keys_buf, 0, hk_buf_bytes(hk_max))
    sh[].hk_keys_buf = hk_keys_buf
    sh[].hk_max_elements = hk_max
    var vec_dead = unsafe_alloc[UInt8](hk_max)
    unsafe_memset(vec_dead, 0, hk_max)
    sh[].vec_dead = vec_dead
    sh[].vec_dead_count = unsafe_alloc[UInt64](1)
    sh[].vec_dead_count[] = 0
    sh[].build_id = unsafe_alloc[UInt64](1)
    sh[].build_id[] = 0
    sh[].vec_gen = unsafe_alloc[UInt64](1)
    sh[].vec_gen[] = 1
    var shard_ready_buf = unsafe_alloc[UInt64](8 * n_workers)
    unsafe_memset(shard_ready_buf.unsafe_bitcast[UInt8](), 0, 8 * n_workers * 8)
    sh[].shard_ready = shard_ready_buf
    var shard_bus_ptr = unsafe_alloc[ShardQueryBus](1)
    shard_bus_ptr.unsafe_write(ShardQueryBus(n_workers))
    sh[].shard_bus = shard_bus_ptr
    var dbg_counters_buf = unsafe_alloc[UInt64](8 * n_workers)
    unsafe_memset(dbg_counters_buf.unsafe_bitcast[UInt8](), 0, 8 * n_workers * 8)
    sh[].dbg_counters = dbg_counters_buf
    return sh


def _memory_value(v: String) -> Int:
    """`--maxmemory`'s parser: bytes, k/kb/m/mb/g/gb, or N% of physical RAM.
    -1 when the value is not one of those. Parsed from a heap copy, as
    main.mojo does: a short String's bytes live inline, and a pointer to them
    handed out of line as MutUntrackedOrigin is miscompiled at -O3."""
    var n = v.byte_length()
    var h = unsafe_alloc[UInt8](n + 1)
    for k in range(n):
        h[unsafe_offset=k] = v.as_bytes()[k]
    var r = parse_memory_value(h, n, Int64(external_call["pion_physical_ram_bytes", UInt64]()))
    h.unsafe_free()
    return Int(r.value) if r.ok else -1


def _open(cfg: String, err: Pointer[UInt8, MutUntrackedOrigin], err_cap: Int64) raises -> Int64:
    var dir = _cfg_get(cfg, "dir")
    if dir.byte_length() == 0:
        _write_err(err, err_cap, "a data directory is required (dir)")
        return 0
    var dirz = dir + "\0"
    if external_call["access", Int32](dirz.unsafe_ptr(), Int32(2)) != 0:   # W_OK
        _write_err(err, err_cap, "the data directory does not exist or is not writable: " + dir)
        return 0
    if external_call["pion_embed_claim", Int32]() != 0:
        _write_err(err, err_cap, "this process already opened a Pion engine; one engine per process")
        return 0
    var lockz = dir + "/pion.lock\0"
    var lock_fd = external_call["pion_dir_lock", Int32](lockz.unsafe_ptr())
    if lock_fd == -2:
        _write_err(err, err_cap, "the data directory is in use by another process (" + dir
                   + "/pion.lock); to share it between processes, run pion-server on it")
        return 0
    if lock_fd < 0:
        _write_err(err, err_cap, "cannot open " + dir + "/pion.lock")
        return 0
    if external_call["pion_set_data_dir", Int32](dir.unsafe_ptr(), Int64(dir.byte_length())) != 0:
        _write_err(err, err_cap, "the data directory path is too long")
        return 0
    # The engine's log lines (WAL replay, snapshot load, warnings) go to a file
    # in the data directory, never to the host's stdout.
    var logfile = _cfg_get(cfg, "logfile")
    if logfile.byte_length() == 0:
        logfile = dir + "/pion.log"
    var logz = logfile + "\0"
    _ = external_call["pion_set_log_file", Int32](logz.unsafe_ptr())

    var config = PionConfig()
    # KV by default. "vector" adds the HNSW index the server's default
    # profile has (open reference kernels: the closed vector library is not
    # linked into libpion). The kv profile refuses FT.CREATE.
    var profile = _cfg_get(cfg, "profile")
    if profile.byte_length() == 0:
        profile = "kv"
    if profile != "kv" and profile != "vector":
        _write_err(err, err_cap, "unknown profile (kv or vector): " + profile)
        return 0
    config.apply_profile(profile)
    var dim = _cfg_get(cfg, "dim")
    if dim.byte_length() > 0:
        config.vector.dimensions = atol(dim)
    # What CONFIG GET port and INFO report. The library listens on nothing;
    # a front end that serves the engine on a port (the test proxy) says so.
    var port = _cfg_get(cfg, "port")
    config.server.workers = 1
    config.server.port = atol(port) if port.byte_length() > 0 else 0
    # The worker's own fallback (pion_worker_entry): macOS has no huge pages,
    # and a SlabAllocator asked for a superpage mapping gets MAP_FAILED, which
    # it does not check. A library has no business asking for them anyway.
    config.server.use_huge_pages = False
    config.server.strict_affinity = False
    config.embedding.enabled = False
    config.llm.enabled = False
    config.inference.enabled = False
    var aof = _cfg_get(cfg, "appendonly")
    if aof == "no":
        config.server.no_wal = True
    var ws = _cfg_get(cfg, "wal-size")
    if ws.byte_length() > 0:
        config.server.wal_size_mb = atol(ws)
    # The rest of the server's options that mean the same thing in a library,
    # named as pion-server's flags without the dashes. A bad value is refused
    # with the server's reason, never ignored.
    var wms = _cfg_get(cfg, "wal-max-segments")
    if wms.byte_length() > 0:
        config.server.wal_max_segments = atol(wms)
    var wfp = _cfg_get(cfg, "wal-full-policy")
    if wfp == "drop":
        config.server.wal_refuse_when_full = False
    elif wfp.byte_length() > 0 and wfp != "refuse":
        _write_err(err, err_cap, "invalid wal-full-policy (expected refuse|drop): " + wfp)
        return 0
    var bt = _cfg_get(cfg, "blob-threshold")
    if bt.byte_length() > 0:
        config.server.blob_threshold = atol(bt)
    if _cfg_get(cfg, "no-blob-tier") == "yes":
        config.server.no_blob_tier = True
    var rp = _cfg_get(cfg, "requirepass")
    if rp.byte_length() > 0:
        config.server.requirepass = rp
    var edc = _cfg_get(cfg, "enable-debug-command").lower()
    if edc == "yes":
        config.server.enable_debug_command = 1
    elif edc == "local":
        config.server.enable_debug_command = 2
    elif edc.byte_length() > 0 and edc != "no":
        _write_err(err, err_cap, "invalid enable-debug-command (no, yes or local): " + edc)
        return 0
    var mss = _cfg_get(cfg, "maxmemory-samples")
    if mss.byte_length() > 0:
        var n = atol(mss)
        if n < 1 or n > 64:
            _write_err(err, err_cap, "invalid maxmemory-samples (1-64): " + mss)
            return 0
        config.server.maxmemory_samples = n
    var lua_time_ms = 5000
    var ltl = _cfg_get(cfg, "lua-time-limit")
    if ltl.byte_length() > 0:
        lua_time_ms = atol(ltl)
        if lua_time_ms < 0:
            _write_err(err, err_cap, "invalid lua-time-limit (milliseconds, 0 = never): " + ltl)
            return 0
    var lua_mem = 1 << 30
    var lml = _cfg_get(cfg, "lua-memory-limit")
    if lml.byte_length() > 0:
        lua_mem = _memory_value(lml)
        if lua_mem < 0:
            _write_err(err, err_cap, "invalid lua-memory-limit (bytes, or with k/kb/m/mb/g/gb, or 1-100%): " + lml)
            return 0
    var mm = _cfg_get(cfg, "maxmemory")
    var pol = _cfg_get(cfg, "maxmemory-policy")
    if pol.byte_length() > 0:
        var p = evict_policy_from_string(pol)
        if p < 0:
            _write_err(err, err_cap, "unknown maxmemory-policy: " + pol)
            return 0
        config.server.maxmemory_policy = p
    external_call["pion_set_maxmemory_policy", NoneType](Int32(config.server.maxmemory_policy))
    external_call["pion_set_maxmemory_samples", NoneType](Int32(config.server.maxmemory_samples))
    if mm.byte_length() > 0:
        config.server.maxmemory = _memory_value(mm)     # Redis units, as --maxmemory
        if config.server.maxmemory < 0:
            _write_err(err, err_cap, "invalid maxmemory (bytes, or with k/kb/m/mb/g/gb, or 1-100%): " + mm)
            return 0
        if config.server.maxmemory > 0:
            external_call["pion_set_maxmemory", NoneType](UInt64(config.server.maxmemory))
    external_call["pion_lua_set_defaults", NoneType](Int64(lua_mem), Int64(lua_time_ms))

    var shared = _make_shared_hnsw(config)
    var cluster = unsafe_alloc[ClusterState](1)
    cluster.unsafe_write(ClusterState())
    var nodes = List[String]()
    nodes.append(String("embedded"))

    var st = unsafe_alloc[EmbedState](1)
    st.unsafe_write(EmbedState())
    st[].lock_fd = lock_fd
    st[].pid = external_call["getpid", Int32]()
    st[].mutex = external_call["pion_mutex_new", Pointer[NoneType, MutUntrackedOrigin]]()
    st[].db = unsafe_alloc[Pion](1)
    st[].db.unsafe_write(Pion(nodes^, config, shared, NO_LISTEN_FD, worker_id=0, num_workers=1,
                              cluster=cluster))

    # The executor's writer, as run_server_io sets it up, with no I/O threads:
    # every fd's owner is -1, so io_tell (KICK, RESUME) is a no-op and this
    # library collects the queued bytes itself.
    ref e = st[].db[].engine
    if is_null(e.writer.ctx[].ovf_lens):
        e.writer.ctx[]._ovf_init()
    var locks = zalloc[UInt32](EMBED_TOP_ID + 1)
    var owners = unsafe_alloc[Int32](EMBED_TOP_ID + 1)
    for i in range(EMBED_TOP_ID + 1):
        owners[unsafe_offset=i] = -1
    e.writer.ctx[].io_locks = locks
    e.writer.ctx[].io_owner = owners
    e.writer.ctx[].io_mode = True
    st[].last_tick_ns = _now_ns()
    # Virtual ids are not descriptors: the socket helpers must never pass one
    # to the kernel (fcntl_wrap.c, pion_set_virtual_floor).
    external_call["pion_set_virtual_floor", NoneType](Int32(EMBED_TOP_ID - EMBED_MAX_CONNS + 1))
    return Int64(Int(st))


@export
def pion_embed_open(cfg: Pointer[UInt8, MutUntrackedOrigin], cfg_len: Int64,
                    err: Pointer[UInt8, MutUntrackedOrigin], err_cap: Int64) abi("C") -> Int64:
    try:
        var s = String(StringSpan[MutUntrackedOrigin](
            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=cfg, length=Int(cfg_len))))
        return _open(s, err, err_cap)
    except e:
        _write_err(err, err_cap, "open failed: " + String(e))
        return 0


@always_inline
def _enter(st: _St) -> Int64:
    """Take the engine lock. 0, or the status the call must return."""
    if external_call["getpid", Int32]() != st[].pid:
        return ST_FORKED
    external_call["pion_mutex_lock", NoneType](st[].mutex)
    if st[].closed:
        external_call["pion_mutex_unlock", NoneType](st[].mutex)
        return ST_ENGINE_CLOSED
    return 0


@always_inline
def _leave(st: _St):
    external_call["pion_mutex_unlock", NoneType](st[].mutex)


def _collect(st: _St, ci: Int) -> Int:
    """Move what is queued for connection ci into its staging buffer."""
    ref ctx = st[].db[].engine.writer.ctx[]
    var total = 0
    while ctx.pending_offsets[unsafe_offset=ci] > 0:
        var n = ctx.pending_offsets[unsafe_offset=ci]
        if total + n > st[].out_caps[unsafe_offset=ci]:
            var ncap = st[].out_caps[unsafe_offset=ci] * 2
            if ncap < total + n:
                ncap = total + n
            if ncap < 4096:
                ncap = 4096
            var nb = unsafe_alloc[UInt8](ncap)
            if total > 0:
                unsafe_memcpy(dest=nb, src=st[].out_bufs[unsafe_offset=ci], count=total)
            if is_not_null(st[].out_bufs[unsafe_offset=ci]):
                st[].out_bufs[unsafe_offset=ci].unsafe_free()
            st[].out_bufs[unsafe_offset=ci] = nb
            st[].out_caps[unsafe_offset=ci] = ncap
        unsafe_memcpy(dest=st[].out_bufs[unsafe_offset=ci].unsafe_offset(total),
                      src=ctx.pending_buffers[unsafe_offset=ci], count=n)
        total += n
        ctx.pending_offsets[unsafe_offset=ci] = 0
        ctx.out_refill(ci)
    return total


def _drop_owed(st: _St, ci: Int):
    """Connection ci was killed (CLIENT KILL, or owed past the 32 MB pub/sub
    limit): what it was owed is dropped, never handed over, as the server
    frees a closed fd's queue. Handing it over made the host push up to
    32 MB to a client that had stopped reading."""
    ref ctx = st[].db[].engine.writer.ctx[]
    ctx.pending_offsets[unsafe_offset=ci] = 0
    st[].db[].engine.writer.out_free(ci)


def _handover(st: _St, ci: Int) -> Int:
    """What exec and poll hand back for connection ci: its output, or
    nothing once the connection was killed."""
    if external_call["pion_virtual_killed", Int32](Int32(ci)) != 0:
        _drop_owed(st, ci)
        return 0
    return _collect(st, ci)


def _wake_drain(st: _St, ci: Int):
    """Connection ci's output was just handed over: its pipe goes quiet."""
    if st[].wake_set[unsafe_offset=ci] != 0:
        external_call["pion_wake_drain", NoneType](st[].wake_r[unsafe_offset=ci] - 1)
        st[].wake_set[unsafe_offset=ci] = 0


def _wake_others(st: _St):
    """Make every wake fd readable whose connection has output waiting: a
    pub/sub message another connection published, a parked command a tick
    answered. Only connections that asked for a wake fd are looked at."""
    if len(st[].wake_ids) == 0:
        return
    ref ctx = st[].db[].engine.writer.ctx[]
    for k in range(len(st[].wake_ids)):
        var ci = st[].wake_ids[k]
        if st[].wake_set[unsafe_offset=ci] == 0 and (ctx.pending_offsets[unsafe_offset=ci] > 0
                or st[].db[].engine.slow_path.clients.close_after[unsafe_offset=ci] != 0
                or external_call["pion_virtual_killed", Int32](Int32(ci)) != 0):
            external_call["pion_wake_signal", NoneType](st[].wake_w[unsafe_offset=ci] - 1)
            st[].wake_set[unsafe_offset=ci] = 1


def _wake_close(st: _St, ci: Int):
    if st[].wake_r[unsafe_offset=ci] != 0:
        external_call["pion_wake_close", NoneType](st[].wake_r[unsafe_offset=ci] - 1,
                                                   st[].wake_w[unsafe_offset=ci] - 1)
        st[].wake_r[unsafe_offset=ci] = 0
        st[].wake_w[unsafe_offset=ci] = 0
        st[].wake_set[unsafe_offset=ci] = 0
        for k in range(len(st[].wake_ids)):
            if st[].wake_ids[k] == ci:
                _ = st[].wake_ids.pop(k)
                break


def _run(st: _St, ci: Int) raises -> Int64:
    """Dispatch what connection ci has buffered. Its status."""
    ref db = st[].db[]
    ref e = db.engine
    if external_call["pion_virtual_killed", Int32](Int32(ci)) != 0:
        return ST_CLOSED              # CLIENT KILL, or past the 32 MB output limit
    var n = e.client_buffer_lens[unsafe_offset=ci]
    if n > 0 and not (e.slow_path.parked_waits.any() and e.slow_path.parked_waits.is_parked(ci)):
        var consumed = e._dispatch_io_batch(Int32(ci), n, KQ_IO, db.hnsw, db.db_size)
        var left = n - consumed
        if left > 0 and consumed > 0:
            var b = e.client_buffers[unsafe_offset=ci]
            _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                b.unsafe_bitcast[NoneType](), b.unsafe_offset(consumed).unsafe_bitcast[NoneType](), left)
        e.client_buffer_lens[unsafe_offset=ci] = left
    if e.slow_path.clients.close_after[unsafe_offset=ci] != 0:
        return ST_CLOSED
    if e.slow_path.parked_waits.any() and e.slow_path.parked_waits.is_parked(ci):
        return ST_PARKED
    if e.client_buffer_lens[unsafe_offset=ci] > 0:
        return ST_PARTIAL
    return ST_DONE


def _tick(st: _St) raises:
    """What run_server_io does per tick, then resume the connections a tick
    unparked (with I/O threads, their thread hands their bytes back)."""
    ref db = st[].db[]
    ref e = db.engine
    # The server ticks about once a millisecond while idle (its kqueue and
    # epoll waits time out after 1 ms) and at most every 100 us while busy.
    # The host calls this every few ms, and later on a loaded machine, so the
    # counter moves by the server ticks that elapsed (at least one): the TTL
    # sweep (every 100) and housekeeping (every 64, which starts automatic
    # vector builds) keep the server's wall-clock cadence. One step per call
    # ran them 5x and more slower, past tests' 1 s windows on CI.
    var now = _now_ns()
    var steps = 1
    if st[].last_tick_ns > 0:
        steps = (now - st[].last_tick_ns) // 1_000_000
        if steps < 1:
            steps = 1
        if steps > 100:
            steps = 100
    st[].last_tick_ns = now
    var sweep = False
    var housekeep = False
    for _ in range(steps):
        e.ttl_sweep_counter += 1
        if e.ttl_sweep_counter >= 100:
            e.ttl_sweep_counter = 0
            sweep = True
        if e.ttl_sweep_counter & 0x3F == 0:
            housekeep = True
    if e.evict_on:
        e._evict_tick()
    if sweep and e.slow_path.clients.pause_until_ms == 0:
        e.fast_path.sweep_expired_keys(20)
    var had_parked = e.slow_path.parked_waits.count() > 0
    if e.slow_path.blocked_readers._count() > 0:
        e._service_blocked_readers(KQ_IO, db.hnsw, db.db_size)
    if e.slow_path.blocked_clients._count() > 0:
        e._service_blocked_clients(KQ_IO, db.hnsw, db.db_size)
    if e.slow_path.parked_waits.count() > 0:
        e._service_parked_waits(KQ_IO, db.hnsw, db.db_size)
    if e.slow_path.clients.pause_until_ms != 0 or e.slow_path.clients.pause_changed \
       or len(e.slow_path.clients.released) > 0:
        e._service_pause(KQ_IO, db.hnsw, db.db_size)
    if housekeep:
        e._housekeeping_64tick(db.hnsw)
        if e.shutting_down:
            st[].closed = True            # SHUTDOWN: the WAL is synced; the engine takes no more
    if had_parked or e.slow_path.clients.pause_until_ms != 0 or len(e.slow_path.clients.released) > 0:
        for k in range(len(st[].live)):
            var ci = st[].live[k]
            if e.client_buffer_lens[unsafe_offset=ci] > 0:
                _ = _run(st, ci)


@always_inline
def _maybe_tick(st: _St) raises:
    if _now_ns() - st[].last_tick_ns >= 100_000:   # run_server_io: at most every 100 us while busy
        _tick(st)


@export
def pion_embed_conn_new(h: Int64) abi("C") -> Int64:
    var st = _st(h)
    var rc = _enter(st)
    if rc != 0:
        return rc
    var ci = -1
    if len(st[].free_ids) > 0:
        ci = st[].free_ids.pop()
    elif len(st[].live) < EMBED_MAX_CONNS:
        ci = st[].next_id
        st[].next_id -= 1
    if ci < 0:
        _leave(st)
        return ST_BAD_CONN
    ref e = st[].db[].engine
    external_call["pion_virtual_reset", NoneType](Int32(ci))   # a reused id starts alive
    e._io_accept(Int32(ci))
    e.slow_path.tx_state.resp_proto[unsafe_offset=ci] = 2
    if is_null(e.client_buffers[unsafe_offset=ci]):
        e.client_buffers[unsafe_offset=ci] = unsafe_alloc[UInt8](EMBED_BUF)
    e.client_buffer_lens[unsafe_offset=ci] = 0
    st[].used[unsafe_offset=ci] = 1
    st[].live.append(ci)
    _leave(st)
    return Int64(ci)


@export
def pion_embed_conn_free(h: Int64, conn: Int64) abi("C"):
    var st = _st(h)
    if _enter(st) != 0:
        return
    var ci = Int(conn)
    if ci < 0 or ci > EMBED_TOP_ID or st[].used[unsafe_offset=ci] == 0:
        _leave(st)
        return
    ref e = st[].db[].engine
    # _close_fd_common without server.close_client: the id is not a descriptor
    var fd = Int32(ci)
    e.slow_path.pubsub.cleanup_fd(fd)
    e.slow_path.tx_state.cleanup_fd(fd)
    e.slow_path.blocked_readers.remove_fd(fd)
    e.slow_path.blocked_clients.remove_fd(fd)
    _ = e.slow_path.monitors.remove(fd)
    e.slow_path.clients.on_close(fd)
    e.slow_path.update_dispatch_gate()
    e.slow_path.parked_waits.remove_fd(fd)
    e.client_buffer_lens[unsafe_offset=ci] = 0
    if is_not_null(e.writer.ctx[].pending_buffers[unsafe_offset=ci]):
        e.writer.ctx[].pending_buffers[unsafe_offset=ci].unsafe_free()
        e.writer.ctx[].pending_buffers[unsafe_offset=ci] = null_ptr[UInt8, MutUntrackedOrigin]()
    e.writer.ctx[].pending_offsets[unsafe_offset=ci] = 0
    e.writer.out_free(ci)
    _wake_close(st, ci)
    st[].used[unsafe_offset=ci] = 0
    for k in range(len(st[].live)):
        if st[].live[k] == ci:
            _ = st[].live.pop(k)
            break
    st[].free_ids.append(ci)
    _leave(st)


@export
def pion_embed_exec(h: Int64, conn: Int64, req: Pointer[UInt8, MutUntrackedOrigin], req_len: Int64,
                    reply_out: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                    out_len: Pointer[Int64, MutUntrackedOrigin]) abi("C") -> Int64:
    var st = _st(h)
    out_len[unsafe_offset=0] = 0
    var rc = _enter(st)
    if rc != 0:
        return rc
    var ci = Int(conn)
    if ci < 0 or ci > EMBED_TOP_ID or st[].used[unsafe_offset=ci] == 0:
        _leave(st)
        return ST_BAD_CONN
    ref e = st[].db[].engine
    var stored = e.client_buffer_lens[unsafe_offset=ci]
    if stored + Int(req_len) > EMBED_BUF:
        _leave(st)
        return ST_TOO_LARGE
    if req_len > 0:
        unsafe_memcpy(dest=e.client_buffers[unsafe_offset=ci].unsafe_offset(stored), src=req, count=Int(req_len))
        e.client_buffer_lens[unsafe_offset=ci] = stored + Int(req_len)
    var status: Int64
    try:
        _maybe_tick(st)
        status = _run(st, ci)
        # As the server's loop does after every batch: a push may have
        # answered a blocked BLPOP/XREAD on another connection. Without this
        # the waiter heard only at the next tick (~5 ms).
        ref sp = e.slow_path
        if sp.blocked_clients._count() > 0:
            e._service_blocked_clients(KQ_IO, st[].db[].hnsw, st[].db[].db_size)
        if sp.blocked_readers._count() > 0:
            e._service_blocked_readers(KQ_IO, st[].db[].hnsw, st[].db[].db_size)
    except:
        status = ST_ERR
    var n = _handover(st, ci)
    _wake_drain(st, ci)
    _wake_others(st)
    reply_out[unsafe_offset=0] = st[].out_bufs[unsafe_offset=ci]
    out_len[unsafe_offset=0] = Int64(n)
    _leave(st)
    return status


@export
def pion_embed_poll(h: Int64, conn: Int64,
                    reply_out: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                    out_len: Pointer[Int64, MutUntrackedOrigin]) abi("C") -> Int64:
    """A tick, then what connection `conn` is owed (a parked reply, pub/sub)."""
    var st = _st(h)
    out_len[unsafe_offset=0] = 0
    var rc = _enter(st)
    if rc != 0:
        return rc
    var ci = Int(conn)
    if ci < 0 or ci > EMBED_TOP_ID or st[].used[unsafe_offset=ci] == 0:
        _leave(st)
        return ST_BAD_CONN
    var status: Int64
    try:
        _tick(st)
        status = _run(st, ci)
    except:
        status = ST_ERR
    var n = _handover(st, ci)
    _wake_drain(st, ci)
    _wake_others(st)
    reply_out[unsafe_offset=0] = st[].out_bufs[unsafe_offset=ci]
    out_len[unsafe_offset=0] = Int64(n)
    _leave(st)
    return status


@export
def pion_embed_wake_fd(h: Int64, conn: Int64) abi("C") -> Int64:
    """A descriptor that is readable while connection `conn` has output the
    caller has not collected (a parked command's reply, a pub/sub message).
    Wait on it in select/poll or an event loop, then call pion_embed_poll.
    It belongs to the connection: closed by pion_embed_conn_free."""
    var st = _st(h)
    var rc = _enter(st)
    if rc != 0:
        return rc
    var ci = Int(conn)
    if ci < 0 or ci > EMBED_TOP_ID or st[].used[unsafe_offset=ci] == 0:
        _leave(st)
        return ST_BAD_CONN
    if st[].wake_r[unsafe_offset=ci] == 0:
        var fds = unsafe_alloc[Int32](2)
        if external_call["pion_wake_pipe", Int32](fds) != 0:
            fds.unsafe_free()
            _leave(st)
            return ST_ERR
        st[].wake_r[unsafe_offset=ci] = fds[unsafe_offset=0] + 1
        st[].wake_w[unsafe_offset=ci] = fds[unsafe_offset=1] + 1
        fds.unsafe_free()
        st[].wake_ids.append(ci)
        _wake_others(st)                 # output already waiting counts too
    var fd = Int64(st[].wake_r[unsafe_offset=ci] - 1)
    _leave(st)
    return fd


@export
def pion_embed_tick(h: Int64) abi("C") -> Int64:
    var st = _st(h)
    var rc = _enter(st)
    if rc != 0:
        return rc
    try:
        _tick(st)
    except:
        rc = ST_ERR
    _wake_others(st)
    _leave(st)
    return rc


@export
def pion_embed_close(h: Int64) abi("C") -> Int64:
    """Sync the WAL to disk and release the directory. The engine's memory is
    not returned (phase 0; one engine per process)."""
    var st = _st(h)
    if external_call["getpid", Int32]() != st[].pid:
        return ST_FORKED
    external_call["pion_mutex_lock", NoneType](st[].mutex)
    if not st[].closed:
        ref e = st[].db[].engine
        if is_not_null(e.fast_path.wal) and e.fast_path.has_wal:
            e.fast_path.wal[].sync_durable()
        if is_not_null(e.fast_path.blobs):
            e.fast_path.blobs[].sync()
        st[].closed = True
    if st[].lock_fd >= 0:
        external_call["pion_dir_unlock", NoneType](st[].lock_fd)
        st[].lock_fd = -1
        _ = external_call["pion_set_log_file", Int32]("".ptr())
    external_call["pion_mutex_unlock", NoneType](st[].mutex)
    return 0


@export
def pion_embed_version(reply_out: Pointer[UInt8, MutUntrackedOrigin], cap: Int64) abi("C") -> Int64:
    var v = PION_VERSION + "+" + PION_BUILD_SHA
    _write_err(reply_out, cap, v)
    return Int64(v.byte_length())


# A script's redis.call() (#36): lua_wrap.c resolves this with dlsym, so the
# library must export it, as main.mojo does for the server. Load the library
# RTLD_GLOBAL so dlsym(RTLD_DEFAULT, ...) finds it.
@export
def pion_script_dispatch(ctx: Pointer[NoneType, MutUntrackedOrigin], argc: Int64,
                         argv: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                         lens: Pointer[Int64, MutUntrackedOrigin], flags: Int64, resp: Int64,
                         reply: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                         wrote: Pointer[Int64, MutUntrackedOrigin]) abi("C") -> Int64:
    var sp = Pointer[SlowPathHandler, MutUntrackedOrigin](unsafe_from_address=Int(ctx))
    return Int64(sp[].script_dispatch(Int(argc), argv, lens, Int(flags), Int(resp), reply, wrote))


# The vector build's threads (FT.OPTIMIZE's link lanes, the automatic build)
# are found with dlsym too. Copies of main.mojo's exports, so a build here
# runs the server's code. TODO(phase 1): one definition both roots export.
@export
def pion_gh199_build_lane(ctx: Pointer[Int64, MutUntrackedOrigin], lane_idx: Int64) abi("C"):
    var g = Pointer[HNSWGraph, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=0]))
    var fp32 = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=1]))
    var n = Int(ctx[unsafe_offset=2])
    var entry_idx = Int(ctx[unsafe_offset=3])
    var n_threads = Int(ctx[unsafe_offset=4])
    var cand = _GH199MinHeap()
    var res = _GH199MaxHeap()
    var visited = unsafe_alloc[UInt16](g[].max_elements)
    unsafe_memset(visited.unsafe_bitcast[UInt8](), 0, g[].max_elements * 2)
    var epoch_cell = unsafe_alloc[UInt16](1)
    epoch_cell[unsafe_offset=0] = 0
    var bi = Int(lane_idx)
    while bi < n:
        if bi != entry_idx:
            try:
                g[]._insert_to_graph_mt(bi, g[].nodes[unsafe_offset=bi].max_level, fp32.unsafe_offset(bi * g[].dim),
                                        cand, res, visited, epoch_cell)
            except:
                pass
        bi += n_threads
    visited.unsafe_free()
    epoch_cell.unsafe_free()


@export
def pion_vector_auto_build(ctx: Pointer[Int64, MutUntrackedOrigin], lane_idx: Int64) abi("C"):
    _ = lane_idx
    var g = Pointer[HNSWGraph, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=0]))
    var sh = Pointer[SharedHNSWView, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=1]))
    var rows = Int(ctx[unsafe_offset=2])
    var w = UInt64(Int(ctx[unsafe_offset=3])) << 8
    var ok = True
    try:
        g[].build_index_from_shared(sh, -1, 1, rows)
    except:
        ok = False
    if ok and (g[].num_nodes == 0 or g[].entry_point_id == -1):
        ok = False
    if not ok:
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](sh[].auto_state, w | UInt64(5))
        return
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](sh[].auto_state, w | UInt64(2))
    while (Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](sh[].auto_state, UInt64(0)) & 0xFF) == 2:
        _ = external_call["usleep", Int32](Int32(1000))
    g[].save_to_disk(data_path("pion.hnsw.0"), sh[].hk_keys_buf, sh[].hk_max_elements)
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](sh[].auto_state, w | UInt64(4))
