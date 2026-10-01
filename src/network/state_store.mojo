"""StateStore — per-worker fixed-size state cache for sequence-specific buffers.

Issue #31 (A4) + A12 fold-in. Backs the `STATE.*` command family. Holds the
per-request, fixed-size buffers DeepSeek-V4 §3.6.1 calls a "state cache":
SWA recent-window K/V, uncompressed CSA/HCA tail tokens, SSM intermediates
(future). The substrate is opaque: callers write and read raw bytes.

Two modes per session (chosen at ALLOC time):
  - fixed: writes past `size` return -ERR offset out of range
  - ring:  writes wrap via (offset+i) % size — bounded-window state

MVP scope (issue #31):
  - No HNSW, no WAL, no replication, no cross-worker routing
  - Per-worker isolation: a session ALLOC'd on worker N is invisible to
    worker M. Multi-worker consumers are responsible for affinity.

Memory budgets (per worker):
  - MAX_STATE_SESSIONS = 64 concurrent sessions
  - MAX_BUFFER_SIZE    = 64 MB per session
  - MAX_TOTAL_BYTES    = 1 GB total across all sessions
"""

from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset

comptime MAX_STATE_SESSIONS = 64
comptime MAX_BUFFER_SIZE    = 64 * 1024 * 1024     # 64 MB per session
comptime MAX_TOTAL_BYTES    = 1024 * 1024 * 1024   # 1 GB per worker

comptime STATE_MODE_FIXED = UInt8(0)
comptime STATE_MODE_RING  = UInt8(1)


struct StateSession(TrivialRegisterPassable):
    """Per-session metadata for a STATE.ALLOC'd buffer."""
    var active: Bool
    var sid_hash: UInt64
    var sid_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var sid_len: Int
    var size: Int                                              # buffer capacity in bytes
    var mode: UInt8                                            # STATE_MODE_{FIXED,RING}
    var buf: Pointer[UInt8, MutUntrackedOrigin]           # the actual bytes
    var bytes_written: UInt64                                  # diagnostic — cumulative
    var bytes_read: UInt64                                     # diagnostic — cumulative

    def __init__(out self):
        self.active = False
        self.sid_hash = 0
        self.sid_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.sid_len = 0
        self.size = 0
        self.mode = STATE_MODE_FIXED
        self.buf = null_ptr[UInt8, MutUntrackedOrigin]()
        self.bytes_written = 0
        self.bytes_read = 0


struct StateStore(Movable):
    """Per-worker state cache. Owned by SlowPathHandler, never shared.

    Slots are pre-allocated; buffers are allocated lazily on STATE.ALLOC and
    released on STATE.FREE. The total-bytes guard prevents runaway allocation.
    """
    var sessions: Pointer[StateSession, MutUntrackedOrigin]
    var session_count: Int
    var enabled: Bool
    var total_bytes_allocated: Int
    var total_allocs: UInt64       # diagnostic
    var total_frees: UInt64        # diagnostic
    var total_writes: UInt64       # diagnostic
    var total_reads: UInt64        # diagnostic

    def __init__(out self, enabled: Bool = False):
        self.enabled = enabled
        self.session_count = 0
        self.total_bytes_allocated = 0
        self.total_allocs = 0
        self.total_frees = 0
        self.total_writes = 0
        self.total_reads = 0
        if enabled:
            var raw = alloc[StateSession](MAX_STATE_SESSIONS)
            self.sessions = Pointer[StateSession, MutUntrackedOrigin](
                unsafe_from_address=Int(raw))
            for i in range(MAX_STATE_SESSIONS):
                self.sessions[unsafe_offset=i] = StateSession()
        else:
            self.sessions = null_ptr[StateSession, MutUntrackedOrigin]()

    @always_inline
    def _hash_sid(self, ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> UInt64:
        var h: UInt64 = 0
        for i in range(length):
            h = h * UInt64(0x100000001b3) + UInt64(Int(ptr[unsafe_offset=i]))
        return h

    def _find_session(
        self,
        ptr: Pointer[UInt8, MutUntrackedOrigin],
        length: Int,
    ) -> Int:
        """Return slot index of a matching active session, or -1."""
        if not self.enabled:
            return -1
        var h = self._hash_sid(ptr, length)
        for i in range(MAX_STATE_SESSIONS):
            if not self.sessions[unsafe_offset=i].active:
                continue
            if self.sessions[unsafe_offset=i].sid_hash != h:
                continue
            if self.sessions[unsafe_offset=i].sid_len != length:
                continue
            var equal = True
            for j in range(length):
                if self.sessions[unsafe_offset=i].sid_ptr[unsafe_offset=j] != ptr[unsafe_offset=j]:
                    equal = False
                    break
            if equal:
                return i
        return -1

    def alloc_session(
        mut self,
        sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
        sid_len: Int,
        size: Int,
        mode: UInt8,
    ) -> Int:
        """Reserve a fixed-size buffer for `sid`. Returns slot index, or:
            -1 disabled, -2 sid exists, -3 size invalid, -4 budget exhausted,
            -5 no free slot.
        """
        if not self.enabled:
            return -1
        if size <= 0 or size > MAX_BUFFER_SIZE:
            return -3
        if self._find_session(sid_ptr, sid_len) >= 0:
            return -2
        if self.total_bytes_allocated + size > MAX_TOTAL_BYTES:
            return -4
        var slot = -1
        for i in range(MAX_STATE_SESSIONS):
            if not self.sessions[unsafe_offset=i].active:
                slot = i
                break
        if slot < 0:
            return -5

        # Copy sid into owned storage so caller's RESP buffer can be reused.
        var _sid = alloc[UInt8](sid_len)
        var sid_copy = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_sid))
        unsafe_memcpy(dest=sid_copy, src=sid_ptr, count=sid_len)

        var _buf = alloc[UInt8](size)
        var buf = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_buf))
        unsafe_memset(buf, 0, size)

        self.sessions[unsafe_offset=slot].active = True
        self.sessions[unsafe_offset=slot].sid_hash = self._hash_sid(sid_ptr, sid_len)
        self.sessions[unsafe_offset=slot].sid_ptr = sid_copy
        self.sessions[unsafe_offset=slot].sid_len = sid_len
        self.sessions[unsafe_offset=slot].size = size
        self.sessions[unsafe_offset=slot].mode = mode
        self.sessions[unsafe_offset=slot].buf = buf
        self.sessions[unsafe_offset=slot].bytes_written = 0
        self.sessions[unsafe_offset=slot].bytes_read = 0
        self.session_count += 1
        self.total_bytes_allocated += size
        self.total_allocs += 1
        return slot

    def free_session(mut self, slot: Int) -> Bool:
        """Release a session's buffer. Idempotent: returns False if slot inactive."""
        if not self.enabled or slot < 0 or slot >= MAX_STATE_SESSIONS:
            return False
        if not self.sessions[unsafe_offset=slot].active:
            return False
        self.sessions[unsafe_offset=slot].sid_ptr.unsafe_free()
        self.sessions[unsafe_offset=slot].buf.unsafe_free()
        self.total_bytes_allocated -= self.sessions[unsafe_offset=slot].size
        self.sessions[unsafe_offset=slot] = StateSession()
        self.session_count -= 1
        self.total_frees += 1
        return True

    def write_bytes(
        mut self,
        slot: Int,
        offset: Int,
        src: Pointer[UInt8, MutUntrackedOrigin],
        length: Int,
    ) -> Int:
        """Write `length` bytes from `src` into the session buffer starting at `offset`.

        Returns bytes written, or:
            -1 inactive slot, -2 negative offset, -3 fixed-mode overflow,
            -4 zero-length write rejected.
        """
        if not self.enabled or slot < 0 or slot >= MAX_STATE_SESSIONS:
            return -1
        if not self.sessions[unsafe_offset=slot].active:
            return -1
        if offset < 0:
            return -2
        if length <= 0:
            return -4
        var size = self.sessions[unsafe_offset=slot].size
        var buf = self.sessions[unsafe_offset=slot].buf
        if self.sessions[unsafe_offset=slot].mode == STATE_MODE_FIXED:
            if offset + length > size:
                return -3
            unsafe_memcpy(dest=buf.unsafe_offset(offset), src=src, count=length)
        else:
            # Ring mode: wrap each byte. Two memcpy calls in the common case
            # (start segment + wrapped tail), one in the trivial case.
            var first_off = offset % size
            var first_len = length
            if first_off + first_len > size:
                first_len = size - first_off
            unsafe_memcpy(dest=buf.unsafe_offset(first_off), src=src, count=first_len)
            var rem = length - first_len
            var src_off = first_len
            while rem > 0:
                var chunk = rem if rem <= size else size
                unsafe_memcpy(dest=buf, src=src.unsafe_offset(src_off), count=chunk)
                rem -= chunk
                src_off += chunk
                # If a single write is larger than the buffer, the final chunk
                # ends up at offset 0; subsequent chunks would just overwrite
                # what we just wrote. The loop intentionally repeats so that
                # the *last* `size` bytes of input win — matches a true ring.
        self.sessions[unsafe_offset=slot].bytes_written += UInt64(length)
        self.total_writes += 1
        return length

    def read_bytes(
        mut self,
        slot: Int,
        offset: Int,
        length: Int,
        dest: Pointer[UInt8, MutUntrackedOrigin],
    ) -> Int:
        """Read `length` bytes into `dest`. Returns bytes read, or:
            -1 inactive slot, -2 negative offset, -3 fixed-mode overflow,
            -4 zero-length read rejected.
        """
        if not self.enabled or slot < 0 or slot >= MAX_STATE_SESSIONS:
            return -1
        if not self.sessions[unsafe_offset=slot].active:
            return -1
        if offset < 0:
            return -2
        if length <= 0:
            return -4
        var size = self.sessions[unsafe_offset=slot].size
        var buf = self.sessions[unsafe_offset=slot].buf
        if self.sessions[unsafe_offset=slot].mode == STATE_MODE_FIXED:
            if offset + length > size:
                return -3
            unsafe_memcpy(dest=dest, src=buf.unsafe_offset(offset), count=length)
        else:
            var first_off = offset % size
            var first_len = length
            if first_off + first_len > size:
                first_len = size - first_off
            unsafe_memcpy(dest=dest, src=buf.unsafe_offset(first_off), count=first_len)
            var rem = length - first_len
            var dst_off = first_len
            while rem > 0:
                var chunk = rem if rem <= size else size
                unsafe_memcpy(dest=dest.unsafe_offset(dst_off), src=buf, count=chunk)
                rem -= chunk
                dst_off += chunk
        self.sessions[unsafe_offset=slot].bytes_read += UInt64(length)
        self.total_reads += 1
        return length
