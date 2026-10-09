from src.common.ptr import null_ptr, is_null
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.ffi import external_call
from std.memory import unsafe_memset
from std.atomic import Atomic, Ordering

# Linux io_uring syscall numbers (ARM64 and x86_64)
comptime SYS_IO_URING_SETUP  = 425
comptime SYS_IO_URING_ENTER  = 426
comptime SYS_IO_URING_REGISTER = 427

# mmap offsets for io_uring ring buffers
comptime IORING_OFF_SQ_RING  = 0
comptime IORING_OFF_CQ_RING  = 0x8000000
comptime IORING_OFF_SQES     = 0x10000000

# io_uring op codes (Linux 5.6+, stable since then)
comptime IORING_OP_NOP       = 0
comptime IORING_OP_TIMEOUT   = 11
comptime IORING_OP_ACCEPT    = 13
comptime IORING_OP_ASYNC_CANCEL = 14
comptime IORING_OP_SEND      = 26
comptime IORING_OP_RECV      = 27
comptime IORING_OP_PROVIDE_BUFFERS = 31

# io_uring_enter flags
comptime IORING_ENTER_GETEVENTS = 1
comptime IORING_ENTER_SQ_WAKEUP = 2
comptime IORING_ENTER_SQ_WAIT   = 4
comptime IORING_ENTER_REGISTERED_RING = 16   # fd is a registered ring index (5.18)

# io_uring_setup flags
comptime IORING_SETUP_SQPOLL    = 2
comptime IORING_SETUP_SQ_AFF    = 4
# gh #205: one thread creates and drives each ring (shared-nothing), which is
# exactly what these promise. SINGLE_ISSUER (6.0) lets the kernel skip the
# submission locking; DEFER_TASKRUN (6.1, needs SINGLE_ISSUER, not SQPOLL)
# runs completion work only inside io_uring_enter(GETEVENTS) instead of
# interrupting the thread with task-work IPIs. The loop enters with
# GETEVENTS on every pass and a 1 ms timeout is always in flight, so
# completions are still collected while idle.
comptime IORING_SETUP_SINGLE_ISSUER = UInt32(1 << 12)
comptime IORING_SETUP_DEFER_TASKRUN = UInt32(1 << 13)

# SQ ring flags (read from sq_flags pointer in the ring)
comptime IORING_SQ_NEED_WAKEUP  = 1

# SQE flags
comptime IOSQE_FIXED_FILE       = UInt8(1 << 0)  # sqe.fd is a registered-file slot
comptime IOSQE_BUFFER_SELECT    = UInt8(1 << 5)  # select buffer from group (bit 5, not 3!)

# CQE flags
comptime IORING_CQE_F_BUFFER    = UInt32(1 << 0)   # buffer ID in flags >> 16
comptime IORING_CQE_F_MORE      = UInt32(1 << 1)   # multishot: more CQEs to come

# recv flags
comptime IORING_RECV_MULTISHOT  = UInt32(1 << 1)   # multishot recv (kernel 6.0+)

# Buffer ring constants
comptime PBUF_RING_ENTRIES = 256     # number of buffers per group
comptime PBUF_SIZE         = 16384   # 16KB per buffer (matches Redis querybuf)
# Bytes past the last provided buffer. A buffer can be parsed in place
# (gh #206), so a vector load at the end of the last one stays inside the
# allocation.
comptime PBUF_POOL_SLACK   = 4096

# Per-fd tables are indexed by fd, like the engine's (client_buffers etc.).
comptime URING_MAX_FDS = 65536

# User data. Every SQE says WHAT it is and, for a client connection, WHICH
# connection it belongs to:
#
#   bits  0-31  fd (the listening fd for an ACCEPT)
#   bits 32-39  kind (UD_*)
#   bits 40-63  the fd's generation when the SQE was made
#
# The generation is bumped when the engine really closes the fd. A completion
# whose generation is not the fd's current one belongs to a connection that is
# gone: it is dropped, and any provided buffer it carries goes back to the
# kernel. The kind is a field of its own; the old tags overlapped (the timeout
# tag shared bit 32 with the send flag) and were told apart only by the order
# of the checks.
comptime UD_RECV    = UInt64(1)
comptime UD_SEND    = UInt64(2)
comptime UD_ACCEPT  = UInt64(3)
comptime UD_PBUF    = UInt64(4)
comptime UD_TIMEOUT = UInt64(5)
comptime UD_CANCEL  = UInt64(6)
comptime UD_GEN_MASK = UInt32(0xFFFFFF)

# Submission Queue Entry (64 bytes, matches Linux kernel layout)
struct SQE(Copyable, Movable, ImplicitlyCopyable):
    var opcode:       UInt8
    var flags:        UInt8
    var ioprio:       UInt16
    var fd:           Int32
    var off:          UInt64
    var addr:         UInt64
    var len:          UInt32
    var op_flags:     UInt32
    var user_data:    UInt64
    var buf_index:    UInt16
    var personality:  UInt16
    var splice_fd_in: Int32
    var addr3:        UInt64
    var pad:          UInt64

    def __init__(out self):
        self.opcode = 0; self.flags = 0; self.ioprio = 0; self.fd = 0
        self.off = 0; self.addr = 0; self.len = 0; self.op_flags = 0
        self.user_data = 0; self.buf_index = 0; self.personality = 0
        self.splice_fd_in = 0; self.addr3 = 0; self.pad = 0

# Completion Queue Entry (16 bytes)
struct CQE(Copyable, Movable, ImplicitlyCopyable):
    var user_data: UInt64
    var res:       Int32
    var flags:     UInt32

    def __init__(out self, user_data: UInt64, res: Int32, flags: UInt32):
        self.user_data = user_data
        self.res = res
        self.flags = flags

# Return type for peek_cqe
struct CQEPeek(Copyable, Movable, ImplicitlyCopyable):
    var found: Bool
    var cqe:   CQE

    def __init__(out self, found: Bool, cqe: CQE):
        self.found = found
        self.cqe   = cqe

# io_uring_params sq_off and cq_off fields
struct SQRingOffsets(Copyable, Movable, ImplicitlyCopyable):
    var head:         UInt32
    var tail:         UInt32
    var ring_mask:    UInt32
    var ring_entries: UInt32
    var flags:        UInt32
    var dropped:      UInt32
    var array:        UInt32
    var resv0:        UInt32
    var resv1:        UInt64

    def __init__(out self):
        self.head = 0; self.tail = 0; self.ring_mask = 0; self.ring_entries = 0
        self.flags = 0; self.dropped = 0; self.array = 0; self.resv0 = 0; self.resv1 = 0

struct CQRingOffsets(Copyable, Movable, ImplicitlyCopyable):
    var head:         UInt32
    var tail:         UInt32
    var ring_mask:    UInt32
    var ring_entries: UInt32
    var overflow:     UInt32
    var cqes:         UInt32
    var flags:        UInt32
    var resv0:        UInt32
    var resv1:        UInt64

    def __init__(out self):
        self.head = 0; self.tail = 0; self.ring_mask = 0; self.ring_entries = 0
        self.overflow = 0; self.cqes = 0; self.flags = 0; self.resv0 = 0; self.resv1 = 0


@no_inline
def _uring_fixed_update(ext: Pointer[UringExtras, MutUntrackedOrigin],
                        table: Pointer[UInt8, MutUntrackedOrigin], fd: Int32, add: Bool):
    """gh #205: fill (`add`) or empty registered-file slot `fd`. Out of line:
    every close site in the io_uring loop calls it, and only a ring with
    registered files gets past the inlined null check in front of it."""
    var ci = Int(fd)
    if ci < 0 or ci >= ext[].fixed_slots:
        return
    if add:
        var r = external_call["pion_uring_files_update", Int32](ext[].real_fd, Int32(ci), fd)
        table[unsafe_offset=ci] = UInt8(1) if r == 1 else UInt8(0)
    elif table[unsafe_offset=ci] != 0:
        table[unsafe_offset=ci] = 0
        _ = external_call["pion_uring_files_update", Int32](ext[].real_fd, Int32(ci), Int32(-1))


struct UringExtras(Movable):
    """gh #205 / #206: the optional features' state that no default-path SQE
    reads, behind IOUring.ext (see the note on IOUring's fields)."""
    # The ring's own fd. IOUring.ring_fd becomes the registered index once
    # the ring fd is registered; io_uring_register still needs this one.
    var real_fd:      Int32
    # Setup flags granted beyond SQPOLL: 0, SINGLE_ISSUER, or
    # SINGLE_ISSUER | DEFER_TASKRUN.
    var setup_flags:  UInt32
    # Registered-file table size (0 = not registered).
    var fixed_slots:  Int
    # The provided-buffer ring (IORING_REGISTER_PBUF_RING); null when the
    # buffers go back with PROVIDE_BUFFERS SQEs instead.
    var pbuf_ring:    Pointer[UInt8, MutUntrackedOrigin]
    var pbuf_tail:    UInt16
    # A multishot RECV's bytes are parsed in the provided buffer itself when
    # the connection holds no unfinished request.
    var zero_copy:    Bool
    # What each register call answered when it failed (-errno, 0 = no
    # failure), for the features line. pbuf_quirk: the buffer ring took the
    # inverted-reserved-word retry (see pion_uring_register_pbuf_ring).
    var ringfd_err:   Int32
    var files_err:    Int32
    var pbuf_err:     Int32
    var pbuf_quirk:   Bool

    def __init__(out self):
        self.real_fd = -1
        self.setup_flags = 0
        self.fixed_slots = 0
        self.pbuf_ring = null_ptr[UInt8, MutUntrackedOrigin]()
        self.pbuf_tail = 0
        self.zero_copy = False
        self.ringfd_err = 0
        self.files_err = 0
        self.pbuf_err = 0
        self.pbuf_quirk = False


struct IOUring(Movable):
    var ring_fd:      Int32
    var sq_ring:      Pointer[UInt8, MutUntrackedOrigin]
    var cq_ring:      Pointer[UInt8, MutUntrackedOrigin]
    var sqes:         Pointer[SQE, MutUntrackedOrigin]
    # Shared with the kernel: it advances the head, we publish the tail.
    var sq_head:      Pointer[UInt32, MutUntrackedOrigin]
    var sq_tail:      Pointer[UInt32, MutUntrackedOrigin]
    # Our tail: SQEs handed out by _get_sqe. Published to `sq_tail` only once
    # they are fully written (see `publish`), never as each one is handed out.
    var sq_tail_local: UInt32
    var sq_ring_mask: UInt32
    var sq_entries:   UInt32
    var cq_head:      Pointer[UInt32, MutUntrackedOrigin]
    var cq_tail:      Pointer[UInt32, MutUntrackedOrigin]
    var cq_ring_mask: UInt32
    var cqes:         Pointer[CQE, MutUntrackedOrigin]
    var sq_array:     Pointer[UInt32, MutUntrackedOrigin]
    var sq_off:       SQRingOffsets
    var cq_off:       CQRingOffsets
    # SQPOLL: pointer to the SQ ring flags word (kernel writes IORING_SQ_NEED_WAKEUP here)
    var sq_flags_ptr: Pointer[UInt32, MutUntrackedOrigin]
    # True when the ring was created with IORING_SETUP_SQPOLL
    var sqpoll_active: Bool
    # gh #173: persistent __kernel_timespec for IORING_OP_TIMEOUT. [0]=sec, [1]=nsec.
    var timeout_ts: Pointer[Int64, MutUntrackedOrigin]
    # Per-fd generation, embedded in every RECV/SEND user_data (see UD_*).
    var fd_gen:       Pointer[UInt32, MutUntrackedOrigin]
    # Per-fd: 1 once the engine has decided to close the connection and is
    # waiting for its RECV/SEND completions to drain (two-phase close). Shared
    # with the ResponseWriter, which must not submit a SEND for such an fd.
    var fd_closing:   Pointer[UInt8, MutUntrackedOrigin]
    # ── gh #205 / #206: optional features (off unless asked for, each
    # resolved by a probe so the engine reports what the kernel granted).
    # Only what a default-path SQE or enter() reads lives here; the rest is
    # behind `ext`. An out-of-line `mut self` call (_make_sq_room, inlined at
    # every submit site) copies every field of this struct in and out, so it
    # stays small. ──
    # 0, or IORING_ENTER_REGISTERED_RING: `ring_fd` is then the registered
    # ring index and `ext[].real_fd` the fd io_uring_register needs.
    var enter_flags:  UInt32
    # Registered files: null when off. Per fd, 1 while slot fd holds the
    # socket on fd.
    var fd_fixed:     Pointer[UInt8, MutUntrackedOrigin]
    var ext:          Pointer[UringExtras, MutUntrackedOrigin]

    def __init__(out self):
        self.ring_fd      = -1
        self.sq_ring      = null_ptr[UInt8, MutUntrackedOrigin]()
        self.cq_ring      = null_ptr[UInt8, MutUntrackedOrigin]()
        self.sqes         = null_ptr[SQE, MutUntrackedOrigin]()
        self.sq_head      = null_ptr[UInt32, MutUntrackedOrigin]()
        self.sq_tail      = null_ptr[UInt32, MutUntrackedOrigin]()
        self.sq_tail_local = 0
        self.sq_ring_mask = 0
        self.sq_entries   = 0
        self.cq_head      = null_ptr[UInt32, MutUntrackedOrigin]()
        self.cq_tail      = null_ptr[UInt32, MutUntrackedOrigin]()
        self.cq_ring_mask = 0
        self.cqes         = null_ptr[CQE, MutUntrackedOrigin]()
        self.sq_array     = null_ptr[UInt32, MutUntrackedOrigin]()
        self.sq_off       = SQRingOffsets()
        self.cq_off       = CQRingOffsets()
        self.sq_flags_ptr = null_ptr[UInt32, MutUntrackedOrigin]()
        self.sqpoll_active = False
        var _ts = alloc[Int64](2)
        _ts[unsafe_offset=0] = 0; _ts[unsafe_offset=1] = 0
        self.timeout_ts = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(_ts))
        self.fd_gen = null_ptr[UInt32, MutUntrackedOrigin]()
        self.fd_closing = null_ptr[UInt8, MutUntrackedOrigin]()
        self.enter_flags = 0
        self.fd_fixed = null_ptr[UInt8, MutUntrackedOrigin]()
        self.ext = alloc[UringExtras](1)
        self.ext.unsafe_write(UringExtras())

    def __moveinit__(out self, deinit take: Self):
        self.ring_fd      = take.ring_fd
        self.sq_ring      = take.sq_ring
        self.cq_ring      = take.cq_ring
        self.sqes         = take.sqes
        self.sq_head      = take.sq_head
        self.sq_tail      = take.sq_tail
        self.sq_tail_local = take.sq_tail_local
        self.sq_ring_mask = take.sq_ring_mask
        self.sq_entries   = take.sq_entries
        self.cq_head      = take.cq_head
        self.cq_tail      = take.cq_tail
        self.cq_ring_mask = take.cq_ring_mask
        self.cqes         = take.cqes
        self.sq_array     = take.sq_array
        self.sq_off       = take.sq_off
        self.cq_off       = take.cq_off
        self.sq_flags_ptr = take.sq_flags_ptr
        self.sqpoll_active = take.sqpoll_active
        self.timeout_ts   = take.timeout_ts
        self.fd_gen       = take.fd_gen
        self.fd_closing   = take.fd_closing
        self.enter_flags  = take.enter_flags
        self.fd_fixed     = take.fd_fixed
        self.ext          = take.ext

    def setup(mut self, entries: UInt32, sqpoll: Bool = False, defer: Bool = False) -> Bool:
        # Allocate a 128-byte buffer for io_uring_params (kernel layout = 120 bytes).
        # Using alloc instead of address_of(embedded field) avoids origin tracking issues.
        var p = alloc[UInt8](128)

        # gh #205: `defer` asks for SINGLE_ISSUER | DEFER_TASKRUN. A kernel
        # that does not know a flag refuses the whole setup (EINVAL), as does
        # DEFER_TASKRUN with SQPOLL, so each refusal is retried with less:
        # SINGLE_ISSUER alone, then nothing. `setup_flags` keeps what stuck.
        var extra = UInt32(0)
        if defer:
            extra = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN
        var ring_fd = Int32(-1)
        while True:
            unsafe_memset(p, 0, 128)
            var flags_ptr = (p.unsafe_offset(8)).unsafe_bitcast[UInt32]()
            flags_ptr[] = extra
            # SQPOLL: set IORING_SETUP_SQPOLL flag in params.flags (offset 8).
            # The kernel spawns a dedicated SQ polling thread that consumes SQEs without
            # requiring io_uring_enter() for submission — only needed when the kernel thread
            # has gone idle (signalled via IORING_SQ_NEED_WAKEUP in sq_flags).
            # sq_thread_idle (offset 16) = 1000ms — kernel thread sleeps after 1s idle.
            if sqpoll:
                flags_ptr[] = extra | UInt32(IORING_SETUP_SQPOLL)
                var idle_ptr = (p.unsafe_offset(16)).unsafe_bitcast[UInt32]()
                idle_ptr[] = UInt32(1000)  # 1000ms idle timeout

            # pion_io_uring_setup wraps syscall(SYS_io_uring_setup, entries, &params)
            ring_fd = external_call["pion_io_uring_setup", Int32](
                entries, p.unsafe_bitcast[NoneType]()
            )
            if ring_fd >= 0 or extra == 0:
                break
            if extra == (IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN):
                extra = IORING_SETUP_SINGLE_ISSUER
            else:
                extra = 0
        if ring_fd < 0:
            p.unsafe_free()
            return False
        self.ring_fd = Int32(ring_fd)
        self.ext[].real_fd = self.ring_fd
        self.ext[].setup_flags = extra
        self.enter_flags = 0

        # Read params fields from the kernel-filled buffer.
        # io_uring_params layout (all LE, confirmed against kernel headers):
        #   offset  0: sq_entries (u32)   4: cq_entries (u32)
        #   offset  8: flags              12: sq_thread_cpu  16: sq_thread_idle  20: features
        #   offset 24: wq_fd             28-39: resv[3]
        #   offset 40: sq_off (io_sqring_offsets, 40 bytes)
        #   offset 80: cq_off (io_cqring_offsets, 40 bytes)
        var u32p = p.unsafe_bitcast[UInt32]()
        var sq_entries = u32p[unsafe_offset=0]
        var cq_entries = u32p[unsafe_offset=1]
        # sq_off starts at byte 40 = index 10
        self.sq_off.head         = u32p[unsafe_offset=10]
        self.sq_off.tail         = u32p[unsafe_offset=11]
        self.sq_off.ring_mask    = u32p[unsafe_offset=12]
        self.sq_off.ring_entries = u32p[unsafe_offset=13]
        self.sq_off.flags        = u32p[unsafe_offset=14]
        self.sq_off.dropped      = u32p[unsafe_offset=15]
        self.sq_off.array        = u32p[unsafe_offset=16]
        # cq_off starts at byte 80 = index 20
        self.cq_off.head         = u32p[unsafe_offset=20]
        self.cq_off.tail         = u32p[unsafe_offset=21]
        self.cq_off.ring_mask    = u32p[unsafe_offset=22]
        self.cq_off.ring_entries = u32p[unsafe_offset=23]
        self.cq_off.overflow     = u32p[unsafe_offset=24]
        self.cq_off.cqes         = u32p[unsafe_offset=25]
        self.cq_off.flags        = u32p[unsafe_offset=26]
        p.unsafe_free()

        self.sq_entries = sq_entries

        # mmap SQ ring — pion_mmap_uring wraps mmap(NULL, length, prot, flags, fd, offset)
        var sq_ring_sz = Int(self.sq_off.array) + Int(sq_entries) * 4
        var sq_ring_ptr = external_call["pion_mmap_uring", Pointer[NoneType, MutUntrackedOrigin]](
            sq_ring_sz,
            3,  # PROT_READ | PROT_WRITE
            1,  # MAP_SHARED
            self.ring_fd, Int64(IORING_OFF_SQ_RING)
        )
        if Int(sq_ring_ptr) == -1:
            self._abandon()
            return False
        self.sq_ring = sq_ring_ptr.unsafe_bitcast[UInt8]()

        # mmap SQEs
        var sqes_sz = Int(sq_entries) * 64  # sizeof(SQE) = 64
        var sqes_ptr = external_call["pion_mmap_uring", Pointer[NoneType, MutUntrackedOrigin]](
            sqes_sz,
            3, 1, self.ring_fd, Int64(IORING_OFF_SQES)
        )
        if Int(sqes_ptr) == -1:
            self._abandon()
            return False
        self.sqes = sqes_ptr.unsafe_bitcast[SQE]()

        # mmap CQ ring (on Linux 5.4+ CQ shares mmap with SQ ring)
        var cq_ring_sz = Int(self.cq_off.cqes) + Int(cq_entries) * 16
        var cq_ring_ptr = external_call["pion_mmap_uring", Pointer[NoneType, MutUntrackedOrigin]](
            cq_ring_sz,
            3, 1, self.ring_fd, Int64(IORING_OFF_CQ_RING)
        )
        if Int(cq_ring_ptr) == -1:
            self._abandon()
            return False
        self.cq_ring = cq_ring_ptr.unsafe_bitcast[UInt8]()

        # Set up pointers into rings using offsets from params
        var sq_base = self.sq_ring
        self.sq_head      = (sq_base.unsafe_offset(Int(self.sq_off.head))).unsafe_bitcast[UInt32]()
        self.sq_tail      = (sq_base.unsafe_offset(Int(self.sq_off.tail))).unsafe_bitcast[UInt32]()
        self.sq_ring_mask = (sq_base.unsafe_offset(Int(self.sq_off.ring_mask))).unsafe_bitcast[UInt32]()[]
        self.sq_entries   = sq_entries
        self.sq_array     = (sq_base.unsafe_offset(Int(self.sq_off.array))).unsafe_bitcast[UInt32]()
        # SQPOLL: sq_flags is at sq_off.flags offset in the SQ ring. The kernel writes
        # IORING_SQ_NEED_WAKEUP here when the SQ polling thread goes idle.
        self.sq_flags_ptr = (sq_base.unsafe_offset(Int(self.sq_off.flags))).unsafe_bitcast[UInt32]()
        self.sqpoll_active = sqpoll
        self.sq_tail_local = self.sq_tail[]
        # SQE slot i is always submitted through array index i, so the
        # indirection array is the identity map, written once here (as
        # liburing does) instead of on every submission.
        for i in range(Int(sq_entries)):
            self.sq_array[unsafe_offset=i] = UInt32(i)

        var cq_base = self.cq_ring
        self.cq_head      = (cq_base.unsafe_offset(Int(self.cq_off.head))).unsafe_bitcast[UInt32]()
        self.cq_tail      = (cq_base.unsafe_offset(Int(self.cq_off.tail))).unsafe_bitcast[UInt32]()
        self.cq_ring_mask = (cq_base.unsafe_offset(Int(self.cq_off.ring_mask))).unsafe_bitcast[UInt32]()[]
        self.cqes         = (cq_base.unsafe_offset(Int(self.cq_off.cqes))).unsafe_bitcast[CQE]()

        self.fd_gen = alloc[UInt32](URING_MAX_FDS)
        unsafe_memset(self.fd_gen.unsafe_bitcast[UInt8](), 0, URING_MAX_FDS * 4)
        self.fd_closing = alloc[UInt8](URING_MAX_FDS)
        unsafe_memset(self.fd_closing, 0, URING_MAX_FDS)

        if sqpoll:
            print("io_uring SQPOLL active — kernel SQ polling thread spawned (idle_timeout=1000ms)")

        return True

    def _abandon(mut self):
        """setup() failed after the ring fd existed: close it, so a retry (or
        the epoll fallback) does not leak it."""
        _ = external_call["close", Int32](self.ring_fd)
        self.ring_fd = -1

    # ── submission ──────────────────────────────────────────────────────────

    @always_inline
    def _get_sqe(mut self) -> Pointer[SQE, MutUntrackedOrigin]:
        """The next free SQE. The caller fills it, and `publish` hands it to the
        kernel only then, so the kernel never sees a half-written entry. A full
        ring is first flushed to the kernel (`_make_sq_room`): one drain pass
        can queue more SQEs than the ring holds."""
        var tail = self.sq_tail_local
        var head = Atomic[Scalar[DType.uint32]].load[ordering=Ordering.ACQUIRE](self.sq_head)
        if tail - head >= self.sq_entries:
            self._make_sq_room()
        self.sq_tail_local = tail + 1
        return self.sqes.unsafe_offset(Int(tail & self.sq_ring_mask))

    @no_inline
    def _make_sq_room(mut self):
        """The ring is full: hand everything queued to the kernel now. Without
        SQPOLL, io_uring_enter consumes the SQEs before it returns; with it,
        IORING_ENTER_SQ_WAIT waits until the polling thread has made room."""
        while True:
            self.publish()
            var head = Atomic[Scalar[DType.uint32]].load[ordering=Ordering.ACQUIRE](self.sq_head)
            if self.sq_tail_local - head < self.sq_entries:
                return
            var r: Int32
            if self.sqpoll_active:
                var flags = UInt32(IORING_ENTER_SQ_WAIT) | self.enter_flags
                if self.needs_wakeup():
                    flags = flags | UInt32(IORING_ENTER_SQ_WAKEUP)
                r = external_call["pion_io_uring_enter", Int32](self.ring_fd, UInt32(0), UInt32(0), flags)
            else:
                r = external_call["pion_io_uring_enter", Int32](
                    self.ring_fd, self.sq_tail_local - head, UInt32(0), self.enter_flags)
            if r < 0:
                # EINTR: retry. EBUSY / EAGAIN: the kernel is out of completion
                # space; yield and retry while it flushes its overflow list
                # into the room this drain pass has already made.
                _ = external_call["sched_yield", Int32]()

    @always_inline
    def publish(mut self):
        """Make every SQE handed out so far visible to the kernel. A RELEASE
        store: the kernel reads the tail and then the entries, so every write
        to them must be ordered before it."""
        Atomic[Scalar[DType.uint32]].store[ordering=Ordering.RELEASE](self.sq_tail, self.sq_tail_local)

    @always_inline
    def pending(self) -> UInt32:
        """SQEs written and not yet consumed by the kernel."""
        return self.sq_tail_local - Atomic[Scalar[DType.uint32]].load[ordering=Ordering.ACQUIRE](self.sq_head)

    @always_inline
    def make_ud(self, kind: UInt64, fd: Int32) -> UInt64:
        var g = UInt64(self.fd_gen[unsafe_offset=Int(fd)])
        return (g << 40) | (kind << 32) | UInt64(UInt32(fd))

    @always_inline
    def submit_accept(mut self, server_fd: Int32):
        var sqe = self._get_sqe()
        sqe[].opcode       = UInt8(IORING_OP_ACCEPT)
        sqe[].flags        = 0
        sqe[].ioprio       = 0
        sqe[].fd           = server_fd
        sqe[].off          = 0
        sqe[].addr         = 0
        sqe[].len          = 0
        sqe[].op_flags     = 0
        sqe[].buf_index    = 0
        sqe[].personality  = 0
        sqe[].splice_fd_in = 0
        sqe[].addr3        = 0
        sqe[].pad          = 0
        sqe[].user_data    = (UD_ACCEPT << 32) | UInt64(UInt32(server_fd))

    @always_inline
    def submit_timeout(mut self, ms: Int):
        """A relative OP_TIMEOUT: its only job is to make enter() return, so the
        loop ticks even when no client sends anything. Completion res is -ETIME
        by design."""
        self.timeout_ts[unsafe_offset=0] = Int64(ms // 1000)
        self.timeout_ts[unsafe_offset=1] = Int64((ms % 1000) * 1_000_000)
        var sqe = self._get_sqe()
        sqe[].opcode       = UInt8(IORING_OP_TIMEOUT)
        sqe[].flags        = 0
        sqe[].ioprio       = 0
        sqe[].fd           = -1
        sqe[].off          = 0                       # count=0: fire on timeout only
        sqe[].addr         = UInt64(Int(self.timeout_ts))
        sqe[].len          = 1                       # exactly one timespec
        sqe[].op_flags     = 0                       # relative timeout
        sqe[].buf_index    = 0
        sqe[].personality  = 0
        sqe[].splice_fd_in = 0
        sqe[].addr3        = 0
        sqe[].pad          = 0
        sqe[].user_data    = UD_TIMEOUT << 32

    @always_inline
    def submit_cancel(mut self, target_user_data: UInt64):
        """IORING_OP_ASYNC_CANCEL of the request whose user_data matches
        exactly. Its own completion (0, -ENOENT or -EALREADY) is ignored; the
        cancelled request completes with -ECANCELED."""
        var sqe = self._get_sqe()
        sqe[].opcode       = UInt8(IORING_OP_ASYNC_CANCEL)
        sqe[].flags        = 0
        sqe[].ioprio       = 0
        sqe[].fd           = -1
        sqe[].off          = 0
        sqe[].addr         = target_user_data
        sqe[].len          = 0
        sqe[].op_flags     = 0
        sqe[].buf_index    = 0
        sqe[].personality  = 0
        sqe[].splice_fd_in = 0
        sqe[].addr3        = 0
        sqe[].pad          = 0
        sqe[].user_data    = UD_CANCEL << 32

    @always_inline
    def submit_recv(mut self, fd: Int32, buf: Pointer[UInt8, MutUntrackedOrigin], length: Int):
        var sqe = self._get_sqe()
        sqe[].opcode       = UInt8(IORING_OP_RECV)
        sqe[].flags        = 0
        sqe[].ioprio       = 0
        sqe[].fd           = fd
        sqe[].off          = 0
        sqe[].addr         = UInt64(Int(buf))
        sqe[].len          = UInt32(length)
        sqe[].op_flags     = 0
        sqe[].buf_index    = 0
        sqe[].personality  = 0
        sqe[].splice_fd_in = 0
        sqe[].addr3        = 0
        sqe[].pad          = 0
        sqe[].user_data    = self.make_ud(UD_RECV, fd)

    @always_inline
    def submit_recv_multishot(mut self, fd: Int32, buf_group: UInt16):
        """Submit a multishot RECV SQE using provided buffers from buf_group.
        Kernel delivers multiple CQEs per SQE without re-arming (kernel 6.0+).
        Each CQE has IORING_CQE_F_BUFFER set and buffer ID in flags >> 16."""
        var sqe = self._get_sqe()
        sqe[].opcode       = UInt8(IORING_OP_RECV)
        sqe[].flags        = IOSQE_BUFFER_SELECT
        sqe[].ioprio       = UInt16(IORING_RECV_MULTISHOT)  # multishot via ioprio field
        sqe[].fd           = fd
        sqe[].off          = 0
        sqe[].addr         = 0  # kernel selects buffer from group
        sqe[].len          = 0  # kernel uses provided buffer length
        sqe[].op_flags     = 0
        sqe[].buf_index    = buf_group
        sqe[].personality  = 0
        sqe[].splice_fd_in = 0
        sqe[].addr3        = 0
        sqe[].pad          = 0
        sqe[].user_data    = self.make_ud(UD_RECV, fd)

    @always_inline
    def submit_provide_buffers(mut self, buf: Pointer[UInt8, MutUntrackedOrigin],
                               buf_len: Int, count: Int, group_id: UInt16, start_bid: UInt16):
        """Provide a batch of buffers to the kernel for buffer selection.
        Buffers must be contiguous: buf[0..buf_len], buf[buf_len..2*buf_len], etc."""
        var sqe = self._get_sqe()
        sqe[].opcode       = UInt8(IORING_OP_PROVIDE_BUFFERS)
        sqe[].flags        = 0
        sqe[].ioprio       = 0
        sqe[].fd           = Int32(count)
        sqe[].off          = UInt64(start_bid)  # starting buffer ID
        sqe[].addr         = UInt64(Int(buf))
        sqe[].len          = UInt32(buf_len)
        sqe[].op_flags     = 0
        sqe[].buf_index    = group_id
        sqe[].personality  = 0
        sqe[].splice_fd_in = 0
        sqe[].addr3        = 0
        sqe[].pad          = 0
        sqe[].user_data    = UD_PBUF << 32

    @always_inline
    def submit_send(mut self, fd: Int32, buf: Pointer[UInt8, MutUntrackedOrigin], length: Int):
        var sqe = self._get_sqe()
        sqe[].opcode       = UInt8(IORING_OP_SEND)
        sqe[].flags        = self.fixed_flag(fd)
        sqe[].ioprio       = 0
        sqe[].fd           = fd
        sqe[].off          = 0
        sqe[].addr         = UInt64(Int(buf))
        sqe[].len          = UInt32(length)
        sqe[].op_flags     = 0
        sqe[].buf_index    = 0
        sqe[].personality  = 0
        sqe[].splice_fd_in = 0
        sqe[].addr3        = 0
        sqe[].pad          = 0
        sqe[].user_data    = self.make_ud(UD_SEND, fd)

    # ── gh #205: registered ring fd and registered files ────────────────────
    # The register_* calls run once, at loop start, and stay out of line: a
    # single-call-site function gets inlined into run_server_uring, and the
    # buffer ring's 256-entry fill unrolled there grew the loop's function by
    # a fifth.

    @no_inline
    def register_ring_fd(mut self) -> Bool:
        """IORING_REGISTER_RING_FDS (5.18): every enter() then names the ring
        by its registered index, and the kernel skips the fd-table lookup it
        does for a plain fd. Per thread, like the ring itself. False when the
        kernel refuses; enter() keeps using the fd."""
        var idx = external_call["pion_uring_register_ring_fd", Int32](self.ext[].real_fd)
        if idx < 0:
            self.ext[].ringfd_err = idx
            return False
        self.ring_fd = idx
        self.enter_flags = UInt32(IORING_ENTER_REGISTERED_RING)
        return True

    @no_inline
    def register_files(mut self, want: Int) -> Bool:
        """IORING_REGISTER_FILES: an empty table of up to `want` slots
        (clamped to RLIMIT_NOFILE). Slot i serves fd i, so no slot allocator
        is needed: `fixed_add` fills a slot at accept, `fixed_remove` empties
        it at close."""
        var n = external_call["pion_uring_register_files_sparse", Int32](
            self.ext[].real_fd, Int32(want))
        if n <= 0:
            self.ext[].files_err = n
            return False
        var t = alloc[UInt8](URING_MAX_FDS)
        unsafe_memset(t, 0, URING_MAX_FDS)
        self.ext[].fixed_slots = Int(n)
        self.fd_fixed = t
        return True

    @always_inline
    def fixed_add(mut self, fd: Int32):
        """The connection just accepted on `fd` goes into slot `fd`, replacing
        whatever the slot held. If the update fails the fd stays unregistered
        and its SQEs name the plain fd."""
        if is_null(self.fd_fixed):
            return
        _uring_fixed_update(self.ext, self.fd_fixed, fd, True)

    @always_inline
    def fixed_remove(mut self, fd: Int32):
        """Empty slot `fd` before the fd is closed. The table holds its own
        reference to the socket: left in place, the socket would outlive
        close(), and an SQE naming the slot would reach it after the fd
        number had gone to a new connection."""
        if is_null(self.fd_fixed):
            return
        _uring_fixed_update(self.ext, self.fd_fixed, fd, False)

    @always_inline
    def fixed_flag(self, fd: Int32) -> UInt8:
        """IOSQE_FIXED_FILE when slot `fd` holds this fd's socket, else 0.
        Only SENDs use it. A multishot RECV takes its file reference once,
        when armed, so a slot saves it nothing; and on kernels before 6.13 a
        long-lived fixed-file request holds back the release of every file
        removed from the table after it was issued, which would keep closed
        sockets alive for as long as some other connection stays idle."""
        if is_null(self.fd_fixed):
            return 0
        return self.fd_fixed[unsafe_offset=Int(fd)]

    # ── gh #206: provided-buffer ring ────────────────────────────────────────

    @no_inline
    def register_pbuf_ring(mut self, pool: Pointer[UInt8, MutUntrackedOrigin], bgid: UInt16) -> Bool:
        """IORING_REGISTER_PBUF_RING (5.19) for buffer group `bgid`, holding
        all PBUF_RING_ENTRIES buffers of `pool`. A buffer then goes back to
        the kernel with a few stores (`pbuf_recycle`) instead of a
        PROVIDE_BUFFERS SQE. Must run before any PROVIDE_BUFFERS for `bgid`
        (the kernel refuses a group that already has legacy buffers)."""
        var ring = external_call["pion_uring_pbuf_ring_alloc", Pointer[UInt8, MutUntrackedOrigin]](
            Int32(PBUF_RING_ENTRIES))
        if is_null(ring):
            self.ext[].pbuf_err = -12   # ENOMEM
            return False
        var r = external_call["pion_uring_register_pbuf_ring", Int32](
            self.ext[].real_fd, ring.unsafe_bitcast[NoneType](), Int32(PBUF_RING_ENTRIES), Int32(bgid))
        if r < 0:
            self.ext[].pbuf_err = r
            _ = external_call["pion_wal_munmap", Int32](ring.unsafe_bitcast[NoneType](), PBUF_RING_ENTRIES * 16)
            return False
        self.ext[].pbuf_quirk = r == 1
        self.ext[].pbuf_ring = ring
        self.ext[].pbuf_tail = 0
        for bid in range(PBUF_RING_ENTRIES):
            self._pbuf_put(pool, bid)
        self._pbuf_publish()
        return True

    @always_inline
    def _pbuf_put(mut self, pool: Pointer[UInt8, MutUntrackedOrigin], bid: Int):
        """Write buffer `bid` into the next ring entry (struct io_uring_buf:
        addr u64, len u32, bid u16, resv u16). The resv of entry 0 is the
        ring's tail, so it is never written here."""
        var x = self.ext
        var e = x[].pbuf_ring.unsafe_offset(Int(x[].pbuf_tail & UInt16(PBUF_RING_ENTRIES - 1)) * 16)
        e.unsafe_bitcast[UInt64]()[] = UInt64(Int(pool.unsafe_offset(bid * PBUF_SIZE)))
        (e.unsafe_offset(8)).unsafe_bitcast[UInt32]()[] = UInt32(PBUF_SIZE)
        (e.unsafe_offset(12)).unsafe_bitcast[UInt16]()[] = UInt16(bid)
        x[].pbuf_tail += 1

    @always_inline
    def _pbuf_publish(mut self):
        """Hand every entry written so far to the kernel: a RELEASE store of
        the tail, so the entries are visible before it."""
        var x = self.ext
        Atomic[Scalar[DType.uint16]].store[ordering=Ordering.RELEASE](
            (x[].pbuf_ring.unsafe_offset(14)).unsafe_bitcast[UInt16](), x[].pbuf_tail)

    @always_inline
    def pbuf_recycle(mut self, pool: Pointer[UInt8, MutUntrackedOrigin], bid: Int):
        """Give provided buffer `bid` back to the kernel through the ring."""
        self._pbuf_put(pool, bid)
        self._pbuf_publish()

    @always_inline
    def needs_wakeup(self) -> Bool:
        """Check if the kernel SQPOLL thread has gone idle and needs a wakeup.
        Only meaningful when sqpoll_active=True. When the kernel thread is idle,
        it sets IORING_SQ_NEED_WAKEUP in sq_flags; we must call enter() with
        IORING_ENTER_SQ_WAKEUP to wake it."""
        var f = Atomic[Scalar[DType.uint32]].load[ordering=Ordering.ACQUIRE](self.sq_flags_ptr)
        return (f & UInt32(IORING_SQ_NEED_WAKEUP)) != 0

    @always_inline
    def enter(mut self, min_complete: Int32):
        """Submit every SQE written so far and wait for `min_complete`
        completions. The submit count is what the kernel has not consumed yet,
        read from the ring, not a count the caller kept: when an enter returned
        early (EINTR) the old caller-side count lost the remainder, and those
        SQEs waited for unrelated later traffic."""
        self.publish()
        # SQPOLL mode: the kernel SQ polling thread handles submission automatically.
        # We only need to call io_uring_enter when:
        #   1. The kernel thread is idle (NEED_WAKEUP set) — wake it with SQ_WAKEUP flag
        #   2. We need to wait for completions (min_complete > 0) — GETEVENTS flag
        if self.sqpoll_active:
            if min_complete > 0:
                var flags = UInt32(IORING_ENTER_GETEVENTS) | self.enter_flags
                if self.needs_wakeup():
                    flags = flags | UInt32(IORING_ENTER_SQ_WAKEUP)
                _ = external_call["pion_io_uring_enter", Int32](
                    self.ring_fd, UInt32(0), UInt32(min_complete), flags
                )
            elif self.needs_wakeup():
                _ = external_call["pion_io_uring_enter", Int32](
                    self.ring_fd, UInt32(0), UInt32(0),
                    UInt32(IORING_ENTER_SQ_WAKEUP) | self.enter_flags
                )
            return

        # Standard mode: explicit submission + optional wait for completions.
        # A failure (-EINTR, -EBUSY with the CQ overflowing) submits nothing or
        # part; whatever is left stays counted by `pending` for the next call.
        # Always with GETEVENTS, even with nothing to submit and
        # min_complete 0: under DEFER_TASKRUN (gh #205) that is the only place
        # the kernel posts completions.
        _ = external_call["pion_io_uring_enter", Int32](
            self.ring_fd, self.pending(), UInt32(min_complete),
            UInt32(IORING_ENTER_GETEVENTS) | self.enter_flags
        )

    # ── completion ──────────────────────────────────────────────────────────

    @always_inline
    def peek_cqe(self) -> CQEPeek:
        var head = self.cq_head[]
        # ACQUIRE: the kernel writes the entry, then the tail; reading the entry
        # after the tail must see it (ARM64 reorders the plain loads).
        var tail = Atomic[Scalar[DType.uint32]].load[ordering=Ordering.ACQUIRE](self.cq_tail)
        if head == tail:
            return CQEPeek(False, CQE(UInt64(0), Int32(0), UInt32(0)))
        var cqe = self.cqes[unsafe_offset=Int(head & self.cq_ring_mask)]
        return CQEPeek(True, cqe)

    @always_inline
    def advance_cq(mut self):
        # RELEASE: the entry has been read before the kernel may reuse its slot.
        Atomic[Scalar[DType.uint32]].store[ordering=Ordering.RELEASE](self.cq_head, self.cq_head[] + 1)

    @staticmethod
    @always_inline
    def ud_kind(user_data: UInt64) -> UInt64:
        return (user_data >> 32) & 0xFF

    @staticmethod
    @always_inline
    def fd_from_user_data(user_data: UInt64) -> Int32:
        return Int32(UInt32(user_data & UInt64(0xFFFFFFFF)))

    @always_inline
    def is_stale(self, user_data: UInt64) -> Bool:
        """A RECV/SEND completion for a connection this fd no longer holds."""
        var fd = Int(user_data & UInt64(0xFFFFFFFF))
        return UInt32(user_data >> 40) != self.fd_gen[unsafe_offset=fd]

    @always_inline
    def retire_fd(mut self, fd: Int32):
        """The connection on `fd` is closed for good: completions still in the
        ring for it are stale from now on."""
        var ci = Int(fd)
        self.fd_gen[unsafe_offset=ci] = (self.fd_gen[unsafe_offset=ci] + 1) & UD_GEN_MASK

    @staticmethod
    @always_inline
    def cqe_has_more(cqe_flags: UInt32) -> Bool:
        """Check if multishot operation will deliver more CQEs."""
        return (cqe_flags & IORING_CQE_F_MORE) != 0

    @staticmethod
    @always_inline
    def cqe_has_buffer(cqe_flags: UInt32) -> Bool:
        """Check if CQE carries a provided buffer ID."""
        return (cqe_flags & IORING_CQE_F_BUFFER) != 0

    @staticmethod
    @always_inline
    def cqe_buffer_id(cqe_flags: UInt32) -> UInt16:
        """Extract provided buffer ID from CQE flags (bits 16-31)."""
        return UInt16(cqe_flags >> 16)


# Legacy stub kept for compilation compatibility
struct IORequest(Copyable, Movable, ImplicitlyCopyable):
    var fd:     Int32
    var offset: UInt64
    var length: UInt32
    var buffer: Pointer[UInt8, MutUntrackedOrigin]

    def __init__(out self, fd: Int32, offset: UInt64, length: UInt32,
                buffer: Pointer[UInt8, MutUntrackedOrigin]):
        self.fd = fd; self.offset = offset; self.length = length; self.buffer = buffer

struct IORing:
    var fd: Int32

    def __init__(out self, entries: Int):
        self.fd = -1

    def submit_read(mut self, request: IORequest):
        pass

    def submit_write(mut self, request: IORequest):
        pass

    def wait_completion(mut self) -> Int:
        return 1
