"""Transaction commands: MULTI, EXEC, DISCARD, WATCH, UNWATCH.

Per-fd command queuing with WATCH optimistic locking.
MULTI enters queuing mode, EXEC replays atomically.
WATCH snapshots key versions; EXEC aborts if any watched key was modified.
"""
from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.value import GenericValue
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.collections import Array
from std.memory import unsafe_memcpy, unsafe_memset
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.fast_path import cmd_matches_4, cmd_matches_5, cmd_matches_7
from src.common.utils import format_int_to_buf
from src.commands.command_table import command_is_denyoom

comptime MAX_TX_FDS = 65536
# Queue and watch list both GROW (doubling). These are the initial per-fd
# allocations, not limits — they used to be hard caps whose overflow path was a
# silent return, which is the one shape this codebase has been bitten by
# repeatedly (gh #149's WAL append returned silently while the client got +OK).
#
# Measured before the fix, on 0.912:
#   - queueing 200 commands in one MULTI got 200 × +QUEUED and an EXEC array of
#     128 — 72 writes acknowledged and silently dropped, and a client expecting
#     200 elements skewed by 72.
#   - WATCH over 20 keys replied +OK while only 16 were recorded, so a write to
#     key 17..20 did NOT abort EXEC. That is precisely the lost update WATCH
#     exists to prevent, failing silently.
comptime INITIAL_QUEUED_CMDS = 128
comptime INITIAL_WATCHED_KEYS = 16  # per fd
# Real ceilings. Redis is bounded only by memory here; these exist so a runaway
# client can't exhaust the worker, and crossing one is LOUD (see `dirty`).
comptime MAX_QUEUED_CMDS = 1 << 20      # 1M queued commands per fd
comptime MAX_WATCHED_KEYS = 1 << 16     # 65536 watched keys per fd
comptime KEY_VERSION_SLOTS = 65536  # hash-based version tracking (collisions cause safe false aborts)


struct WatchedKey(Copyable, Movable, ImplicitlyCopyable):
    var slot: UInt16      # index into key_versions array
    var version: UInt64   # version at WATCH time

    def __init__(out self):
        self.slot = 0; self.version = 0


struct QueuedCommand(Copyable, Movable, ImplicitlyCopyable):
    var data: Pointer[UInt8, MutUntrackedOrigin]
    var length: Int

    def __init__(out self):
        self.data = null_ptr[UInt8, MutUntrackedOrigin]()
        self.length = 0


struct TransactionState(Movable):
    """Per-worker transaction state. Tracks MULTI mode, command queues, and WATCH state."""
    var in_multi: Pointer[UInt8, MutUntrackedOrigin]      # [MAX_TX_FDS] 0=normal, 1=multi
    var queues: Pointer[Pointer[QueuedCommand, MutUntrackedOrigin], MutUntrackedOrigin]
    var queue_counts: Pointer[Int32, MutUntrackedOrigin]   # [MAX_TX_FDS]
    var queue_caps: Pointer[Int32, MutUntrackedOrigin]     # [MAX_TX_FDS] allocated slots
    # Transaction poisoning. Set when a queue-time step fails (queue ceiling
    # reached). EXEC then replies -EXECABORT and applies NOTHING, which is what
    # Redis does after a queue-time error — a half-applied transaction is worse
    # than a rejected one. Cleared by start_multi/discard.
    var dirty: Pointer[UInt8, MutUntrackedOrigin]          # [MAX_TX_FDS]
    # WATCH: per-key version tracking
    var key_versions: Pointer[UInt64, MutUntrackedOrigin]  # [KEY_VERSION_SLOTS] global version per key-slot
    # Per-fd watched keys
    var watch_keys: Pointer[Pointer[WatchedKey, MutUntrackedOrigin], MutUntrackedOrigin]  # [MAX_TX_FDS]
    # Int32, not UInt8: the list grows now, and a UInt8 count would silently
    # wrap at 256 watched keys — the same class of bug one level down.
    var watch_counts: Pointer[Int32, MutUntrackedOrigin]   # [MAX_TX_FDS]
    var watch_caps: Pointer[Int32, MutUntrackedOrigin]     # [MAX_TX_FDS] allocated slots
    # gh #100 (C2): per-fd authentication flag. 0 = unauthenticated, 1 = AUTH
    # succeeded on this connection. Reset to 0 on connect (fresh alloc / memset)
    # and on close (cleanup_fd) so a reused fd never inherits a prior session's
    # authenticated state. Only consulted when --requirepass is set.
    var authed: Pointer[UInt8, MutUntrackedOrigin]         # [MAX_TX_FDS]
    # gh #101: per-fd tenant binding — index into the worker's TenantTable;
    # -1 = unbound or admin. Same lifecycle as `authed`: -1 on fresh alloc,
    # reset in cleanup_fd so a reused fd never inherits a tenant namespace.
    var tenant_id: Pointer[Int16, MutUntrackedOrigin]      # [MAX_TX_FDS]
    # gh #172: per-fd wire protocol — 2 (default) or 3, set by a successful
    # `HELLO 3`. Same lifecycle as `authed`/`tenant_id`: 2 on fresh alloc, reset
    # to 2 in cleanup_fd so a reused fd never inherits the previous session's
    # protocol and start answering RESP3 to a RESP2 client.
    var resp_proto: Pointer[UInt8, MutUntrackedOrigin]     # [MAX_TX_FDS]

    def __init__(out self):
        self.in_multi = alloc[UInt8](MAX_TX_FDS)
        unsafe_memset(self.in_multi, 0, MAX_TX_FDS)
        self.queues = alloc[Pointer[QueuedCommand, MutUntrackedOrigin]](MAX_TX_FDS)
        unsafe_memset(self.queues.unsafe_bitcast[UInt8](), 0, MAX_TX_FDS * 8)
        self.queue_counts = alloc[Int32](MAX_TX_FDS)
        unsafe_memset(self.queue_counts.unsafe_bitcast[UInt8](), 0, MAX_TX_FDS * 4)
        self.queue_caps = alloc[Int32](MAX_TX_FDS)
        unsafe_memset(self.queue_caps.unsafe_bitcast[UInt8](), 0, MAX_TX_FDS * 4)
        self.dirty = alloc[UInt8](MAX_TX_FDS)
        unsafe_memset(self.dirty, 0, MAX_TX_FDS)
        self.key_versions = alloc[UInt64](KEY_VERSION_SLOTS)
        unsafe_memset(self.key_versions.unsafe_bitcast[UInt8](), 0, KEY_VERSION_SLOTS * 8)
        self.watch_keys = alloc[Pointer[WatchedKey, MutUntrackedOrigin]](MAX_TX_FDS)
        unsafe_memset(self.watch_keys.unsafe_bitcast[UInt8](), 0, MAX_TX_FDS * 8)
        self.watch_counts = alloc[Int32](MAX_TX_FDS)
        unsafe_memset(self.watch_counts.unsafe_bitcast[UInt8](), 0, MAX_TX_FDS * 4)
        self.watch_caps = alloc[Int32](MAX_TX_FDS)
        unsafe_memset(self.watch_caps.unsafe_bitcast[UInt8](), 0, MAX_TX_FDS * 4)
        self.authed = alloc[UInt8](MAX_TX_FDS)
        unsafe_memset(self.authed, 0, MAX_TX_FDS)
        self.tenant_id = alloc[Int16](MAX_TX_FDS)
        # 0xFF in every byte = Int16(-1) in every slot (unbound).
        unsafe_memset(self.tenant_id.unsafe_bitcast[UInt8](), 0xFF, MAX_TX_FDS * 2)
        self.resp_proto = alloc[UInt8](MAX_TX_FDS)
        unsafe_memset(self.resp_proto, 2, MAX_TX_FDS)

    def is_multi(self, fd: Int32) -> Bool:
        return self.in_multi[unsafe_offset=Int(fd)] == 1

    @always_inline
    def is_authed(self, fd: Int32) -> Bool:
        return self.authed[unsafe_offset=Int(fd)] == 1

    @always_inline
    def set_authed(mut self, fd: Int32):
        self.authed[Int(fd)] = 1

    def start_multi(mut self, fd: Int32):
        self.in_multi[unsafe_offset=Int(fd)] = 1
        if is_null(self.queues[unsafe_offset=Int(fd)]):
            self.queues[unsafe_offset=Int(fd)] = alloc[QueuedCommand](INITIAL_QUEUED_CMDS)
            for i in range(INITIAL_QUEUED_CMDS):
                (self.queues[unsafe_offset=Int(fd)].unsafe_offset(i)).unsafe_write(QueuedCommand())
            self.queue_caps[unsafe_offset=Int(fd)] = Int32(INITIAL_QUEUED_CMDS)
        self.queue_counts[unsafe_offset=Int(fd)] = 0
        self.dirty[unsafe_offset=Int(fd)] = 0

    def discard(mut self, fd: Int32):
        self.in_multi[unsafe_offset=Int(fd)] = 0
        var cnt = Int(self.queue_counts[unsafe_offset=Int(fd)])
        if is_not_null(self.queues[unsafe_offset=Int(fd)]):
            for i in range(cnt):
                if is_not_null(self.queues[unsafe_offset=Int(fd)][unsafe_offset=i].data):
                    self.queues[unsafe_offset=Int(fd)][unsafe_offset=i].data.unsafe_free()
                    self.queues[unsafe_offset=Int(fd)][unsafe_offset=i].data = null_ptr[UInt8, MutUntrackedOrigin]()
                    self.queues[unsafe_offset=Int(fd)][unsafe_offset=i].length = 0
        self.queue_counts[unsafe_offset=Int(fd)] = 0
        self.dirty[unsafe_offset=Int(fd)] = 0
        # A transaction that grew past the initial allocation hands the memory
        # back rather than keeping a fat queue parked on an idle fd forever.
        if (self.queue_caps[unsafe_offset=Int(fd)] > Int32(INITIAL_QUEUED_CMDS)
                and is_not_null(self.queues[unsafe_offset=Int(fd)])):
            self.queues[unsafe_offset=Int(fd)].unsafe_free()
            self.queues[unsafe_offset=Int(fd)] = null_ptr[QueuedCommand, MutUntrackedOrigin]()
            self.queue_caps[unsafe_offset=Int(fd)] = 0
        # Clear watch state
        self.watch_counts[unsafe_offset=Int(fd)] = 0
        if (self.watch_caps[unsafe_offset=Int(fd)] > Int32(INITIAL_WATCHED_KEYS)
                and is_not_null(self.watch_keys[unsafe_offset=Int(fd)])):
            self.watch_keys[unsafe_offset=Int(fd)].unsafe_free()
            self.watch_keys[unsafe_offset=Int(fd)] = null_ptr[WatchedKey, MutUntrackedOrigin]()
            self.watch_caps[unsafe_offset=Int(fd)] = 0

    def _grow_queue(mut self, fd: Int32) -> Bool:
        """Double this fd's command queue. False only at MAX_QUEUED_CMDS."""
        var cap = Int(self.queue_caps[unsafe_offset=Int(fd)])
        if cap >= MAX_QUEUED_CMDS: return False
        var new_cap = cap * 2 if cap > 0 else INITIAL_QUEUED_CMDS
        if new_cap > MAX_QUEUED_CMDS: new_cap = MAX_QUEUED_CMDS
        var fresh = alloc[QueuedCommand](new_cap)
        var old = self.queues[unsafe_offset=Int(fd)]
        var cnt = Int(self.queue_counts[unsafe_offset=Int(fd)])
        # Element-wise: QueuedCommand is a POD pair, and growth is amortised
        # O(1) — this is not a hot path, so no bitcast/size assumptions.
        for i in range(new_cap):
            if i < cnt and is_not_null(old):
                (fresh.unsafe_offset(i)).unsafe_write(old[unsafe_offset=i])
            else:
                (fresh.unsafe_offset(i)).unsafe_write(QueuedCommand())
        # Free the descriptor array only — the copied `data` payloads are now
        # owned by `fresh` and must NOT be freed here.
        if is_not_null(old): old.unsafe_free()
        self.queues[unsafe_offset=Int(fd)] = fresh
        self.queue_caps[unsafe_offset=Int(fd)] = Int32(new_cap)
        return True

    def enqueue(mut self, fd: Int32, cmd_data: Pointer[UInt8, MutUntrackedOrigin], cmd_len: Int) -> Bool:
        """Queue one command frame. False ONLY at the hard ceiling — the caller
        must then report an error and poison the transaction, never reply
        +QUEUED for a command it did not store (gh #219 follow-up)."""
        var cnt = Int(self.queue_counts[unsafe_offset=Int(fd)])
        if cnt >= Int(self.queue_caps[unsafe_offset=Int(fd)]):
            if not self._grow_queue(fd): return False
        var copy = alloc[UInt8](cmd_len)
        unsafe_memcpy(dest=copy, src=cmd_data, count=cmd_len)
        self.queues[unsafe_offset=Int(fd)][unsafe_offset=cnt].data = copy
        self.queues[unsafe_offset=Int(fd)][unsafe_offset=cnt].length = cmd_len
        self.queue_counts[unsafe_offset=Int(fd)] = Int32(cnt + 1)
        return True

    @always_inline
    def set_dirty(mut self, fd: Int32):
        self.dirty[unsafe_offset=Int(fd)] = 1

    @always_inline
    def is_dirty(self, fd: Int32) -> Bool:
        return self.dirty[unsafe_offset=Int(fd)] == 1

    def cleanup_fd(mut self, fd: Int32):
        self.discard(fd)
        # gh #100 (C2): clear auth so a reused fd starts unauthenticated.
        self.authed[unsafe_offset=Int(fd)] = 0
        # gh #101: clear tenant binding for the same reason.
        self.tenant_id[unsafe_offset=Int(fd)] = -1
        # gh #172: back to RESP2 so a reused fd never inherits RESP3.
        self.resp_proto[unsafe_offset=Int(fd)] = 2

    # ── Key version tracking ──

    @always_inline
    @staticmethod
    def key_slot_from_hash(h: UInt64) -> Int:
        """Version slot for a key whose GenericValue hash is already known.

        gh #230: the slot used to come from a byte-at-a-time FNV loop — one
        dependent multiply PER KEY BYTE, 16 of them for the benchmark's
        `key:__rand_int__`, on the critical path of every mutation that bumps a
        version. Every hot caller already computes the key's GenericValue hash
        to reach the keyspace, so the slot now falls out of that hash for free.

        The slot never leaves this process (`key_versions` is a plain in-memory
        array, `WatchedKey.slot` a UInt16 beside it), so the derivation is free
        to change. What is NOT free is changing it for only some callers: WATCH
        records a slot and EXEC compares it against what a mutation bumped, so a
        recorder and a bumper that disagree silently stop aborting — the one
        failure WATCH exists to prevent. Both go through this function."""
        return Int(h & UInt64(KEY_VERSION_SLOTS - 1))

    @always_inline
    @staticmethod
    def key_slot(key_ptr: Pointer[UInt8, MutUntrackedOrigin], key_len: Int) -> Int:
        """Version slot from raw key bytes, for callers with no hash in hand.

        Cold paths only (WATCH). `from_ptr` COPIES a key longer than 23 bytes to
        the heap, so the temporary is freed here rather than leaked once per
        WATCH; hot mutation sites pass their existing hash to
        key_slot_from_hash instead and never build a value at all."""
        var kv = GenericValue.borrow(key_ptr, key_len)
        var slot = Self.key_slot_from_hash(UInt64(kv.__hash__()))
        kv.free_str_payload()
        return slot

    @always_inline
    def bump_key_version(mut self, key_ptr: Pointer[UInt8, MutUntrackedOrigin], key_len: Int):
        """Increment version for a key's slot. Called on every mutation (SET, DEL, HSET, etc.)."""
        var slot = Self.key_slot(key_ptr, key_len)
        self.key_versions[slot] += 1

    def _grow_watch(mut self, fd: Int32) -> Bool:
        """Double this fd's watch list. False only at MAX_WATCHED_KEYS."""
        var cap = Int(self.watch_caps[unsafe_offset=Int(fd)])
        if cap >= MAX_WATCHED_KEYS: return False
        var new_cap = cap * 2 if cap > 0 else INITIAL_WATCHED_KEYS
        if new_cap > MAX_WATCHED_KEYS: new_cap = MAX_WATCHED_KEYS
        var fresh = alloc[WatchedKey](new_cap)
        var old = self.watch_keys[unsafe_offset=Int(fd)]
        var wc = Int(self.watch_counts[unsafe_offset=Int(fd)])
        for i in range(new_cap):
            if i < wc and is_not_null(old):
                (fresh.unsafe_offset(i)).unsafe_write(old[unsafe_offset=i])
            else:
                (fresh.unsafe_offset(i)).unsafe_write(WatchedKey())
        if is_not_null(old): old.unsafe_free()
        self.watch_keys[unsafe_offset=Int(fd)] = fresh
        self.watch_caps[unsafe_offset=Int(fd)] = Int32(new_cap)
        return True

    @always_inline
    def watch_key(mut self, fd: Int32, key_ptr: Pointer[UInt8, MutUntrackedOrigin], key_len: Int) -> Bool:
        """Snapshot current version of a key for this fd.

        Returns False ONLY at the hard ceiling. This used to return silently
        past a 16-key cap while WATCH still replied +OK, so key 17+ was not
        watched and EXEC did not abort when it was modified — a silent lost
        update, which is the one failure WATCH exists to prevent.
        """
        var wc = Int(self.watch_counts[unsafe_offset=Int(fd)])
        if wc >= Int(self.watch_caps[unsafe_offset=Int(fd)]):
            if not self._grow_watch(fd): return False
        var slot = Self.key_slot(key_ptr, key_len)
        self.watch_keys[unsafe_offset=Int(fd)][unsafe_offset=wc].slot = UInt16(slot)
        self.watch_keys[unsafe_offset=Int(fd)][unsafe_offset=wc].version = self.key_versions[unsafe_offset=slot]
        self.watch_counts[unsafe_offset=Int(fd)] = Int32(wc + 1)
        return True

    @always_inline
    def check_watch(self, fd: Int32) -> Bool:
        """Check if any watched key was modified. Returns True if all clean (EXEC can proceed)."""
        var wc = Int(self.watch_counts[unsafe_offset=Int(fd)])
        if wc == 0: return True  # no watches = always clean
        if is_null(self.watch_keys[unsafe_offset=Int(fd)]): return True
        for i in range(wc):
            var slot = Int(self.watch_keys[unsafe_offset=Int(fd)][unsafe_offset=i].slot)
            if self.key_versions[unsafe_offset=slot] != self.watch_keys[unsafe_offset=Int(fd)][unsafe_offset=i].version:
                return False  # key was modified
        return True

    @always_inline
    def clear_watch(mut self, fd: Int32):
        """Clear all watches for this fd."""
        self.watch_counts[unsafe_offset=Int(fd)] = 0


# ── Command Handlers ──

@always_inline
def handle_multi(fd: Int32, mut tx: TransactionState, mut writer: ResponseWriter) -> Int:
    """MULTI → +OK (enter queuing mode)."""
    if tx.is_multi(fd):
        writer.append_error_response("ERR MULTI calls can not be nested")
    else:
        tx.start_multi(fd)
        writer.append_ok_response()
    return 0


@always_inline
def tx_queue_has_denyoom(fd: Int32, tx: TransactionState) -> Bool:
    """gh #261: does the queued transaction contain a memory-growing command?

    Redis answers EXEC with -EXECABORT, applying nothing, when memory crossed
    maxmemory after such a command was queued. Checking here rather than letting
    each replayed command refuse itself is the difference between an aborted
    transaction and a HALF-applied one.

    Frames are stored raw: RESP (`*N\\r\\n$L\\r\\nNAME...`) or inline (`NAME ...`).
    The queue is heap memory, so the name pointer is safe to hand on."""
    var cnt = Int(tx.queue_counts[unsafe_offset=Int(fd)])
    var q = tx.queues[unsafe_offset=Int(fd)]
    if is_null(q):
        return False
    for k in range(cnt):
        var d = q[unsafe_offset=k].data
        var n = q[unsafe_offset=k].length
        if is_null(d) or n <= 0:
            continue
        var start = 0
        var nlen = 0
        if d[unsafe_offset=0] == 42:                 # '*': skip "*N\r\n$L\r\n"
            var p = 0
            while p < n and d[unsafe_offset=p] != 10:
                p += 1
            p += 1
            if p >= n or d[unsafe_offset=p] != 36:   # '$'
                continue
            p += 1
            while p < n and d[unsafe_offset=p] >= 48 and d[unsafe_offset=p] <= 57:
                nlen = nlen * 10 + Int(d[unsafe_offset=p]) - 48
                p += 1
            start = p + 2                            # past "\r\n"
        else:
            while nlen < n and d[unsafe_offset=nlen] != 32 and d[unsafe_offset=nlen] != 13 \
                  and d[unsafe_offset=nlen] != 10:
                nlen += 1
        if nlen > 0 and start + nlen <= n and command_is_denyoom(d.unsafe_offset(start), nlen):
            return True
    return False


def handle_exec_start(fd: Int32, mut tx: TransactionState, mut writer: ResponseWriter) -> Int:
    """EXEC → execute queued commands and return array of results.
    Returns the number of queued commands (caller must replay them).
    Returns -1 on error, -2 on WATCH abort."""
    if not tx.is_multi(fd):
        writer.append_error_response("ERR EXEC without MULTI")
        return -1
    # A queue-time failure poisons the transaction: apply nothing and say so.
    # Redis's contract after a queue error is EXECABORT + zero side effects,
    # and a partially applied transaction is worse than a rejected one.
    if tx.is_dirty(fd):
        tx.discard(fd)
        writer.append_error_response(
            "EXECABORT Transaction discarded because of previous errors."
        )
        return -1
    # Check WATCH — abort if any watched key was modified
    if not tx.check_watch(fd):
        tx.in_multi[unsafe_offset=Int(fd)] = 0
        tx.queue_counts[unsafe_offset=Int(fd)] = 0
        tx.clear_watch(fd)
        writer.append_null_response()  # $-1 = transaction aborted
        return -2
    var cnt = Int(tx.queue_counts[unsafe_offset=Int(fd)])
    tx.in_multi[unsafe_offset=Int(fd)] = 0
    tx.clear_watch(fd)
    if cnt == 0:
        writer.append_empty_array_response()
        return 0
    # Write array header *N\r\n
    writer.buffer[unsafe_offset=writer.offset] = 42  # '*'
    writer.offset += 1
    writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(cnt))
    writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
    writer.offset += 2
    return cnt


@always_inline
def handle_discard(fd: Int32, mut tx: TransactionState, mut writer: ResponseWriter) -> Int:
    """DISCARD → +OK (cancel transaction)."""
    if not tx.is_multi(fd):
        writer.append_error_response("ERR DISCARD without MULTI")
    else:
        tx.discard(fd)
        writer.append_ok_response()
    return 0


@always_inline
def handle_watch(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, fd: Int32, mut tx: TransactionState, mut writer: ResponseWriter) -> Int:
    """WATCH key [key ...] → +OK (snapshot key versions for optimistic locking)."""
    if tx.is_multi(fd):
        writer.append_error_response("ERR WATCH inside MULTI is not allowed")
        return num_tokens - i - 1
    var n = num_tokens - i - 1
    if n == 0:
        writer.append_error_response("ERR wrong number of arguments for 'watch' command")
        return 0
    var all_watched = True
    for wi in range(n):
        var kp = tokens[unsafe_offset=i + 1 + wi].ptr
        var kl = tokens[unsafe_offset=i + 1 + wi].length
        if not tx.watch_key(fd, kp, kl):
            all_watched = False
            break
    if not all_watched:
        # Never reply +OK for keys we are not actually watching — that turns
        # WATCH into a no-op the client cannot detect and loses the update it
        # was guarding. Drop the partial set so no false sense of protection
        # survives, and say so.
        tx.clear_watch(fd)
        writer.append_error_response(
            "ERR WATCH exceeds the per-connection limit of "
            + String(MAX_WATCHED_KEYS)
            + " keys; no keys are being watched"
        )
        return n
    writer.append_ok_response()
    return n


@always_inline
def handle_unwatch(fd: Int32, mut tx: TransactionState, mut writer: ResponseWriter) -> Int:
    """UNWATCH → +OK (clear all watches)."""
    tx.clear_watch(fd)
    writer.append_ok_response()
    return 0
