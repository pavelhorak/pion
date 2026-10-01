from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.ffi import external_call
from std.memory import unsafe_memset

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
comptime IORING_OP_SEND      = 26
comptime IORING_OP_RECV      = 27
comptime IORING_OP_PROVIDE_BUFFERS = 31

# io_uring_enter flags
comptime IORING_ENTER_GETEVENTS = 1
comptime IORING_ENTER_SQ_WAKEUP = 4

# io_uring_setup flags
comptime IORING_SETUP_SQPOLL    = 2
comptime IORING_SETUP_SQ_AFF    = 4

# SQ ring flags (read from sq_flags pointer in the ring)
comptime IORING_SQ_NEED_WAKEUP  = 1

# SQE flags
comptime IOSQE_BUFFER_SELECT    = UInt8(1 << 5)  # select buffer from group (bit 5, not 3!)

# CQE flags
comptime IORING_CQE_F_BUFFER    = UInt32(1 << 0)   # buffer ID in flags >> 16
comptime IORING_CQE_F_MORE      = UInt32(1 << 1)   # multishot: more CQEs to come

# recv flags
comptime IORING_RECV_MULTISHOT  = UInt32(1 << 1)   # multishot recv (kernel 6.0+)

# Buffer ring constants
comptime PBUF_RING_ENTRIES = 256     # number of buffers per group
comptime PBUF_SIZE         = 16384   # 16KB per buffer (matches Redis querybuf)

# User-data tag bits to distinguish accept/recv/send completions.
# Accept: high 32 bits = 0xFFFFFFFF, low 32 bits = listen_fd.
# Send:   bit 32 set (UDATA_SEND_FLAG), low 32 bits = client_fd.
# Recv:   low 32 bits = client_fd, high 32 bits = 0.
comptime UDATA_SERVER_ACCEPT = UInt64(0xFFFFFFFF00000000)
# gh #173: tick-timeout sentinel (high 32 bits; 0xFFFFFFFE = provide_buffers).
comptime UDATA_TIMEOUT       = UInt64(0xFFFFFFFD00000000)
comptime UDATA_SEND_FLAG     = UInt64(0x0000000100000000)

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


struct IOUring(Movable):
    var ring_fd:      Int32
    var sq_ring:      Pointer[UInt8, MutUntrackedOrigin]
    var cq_ring:      Pointer[UInt8, MutUntrackedOrigin]
    var sqes:         Pointer[SQE, MutUntrackedOrigin]
    var sq_head:      Pointer[UInt32, MutUntrackedOrigin]
    var sq_tail:      Pointer[UInt32, MutUntrackedOrigin]
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
    # gh #173: persistent __kernel_timespec for IORING_OP_TIMEOUT (must outlive
    # the SQE — the kernel reads it at completion time). [0]=sec, [1]=nsec.
    var timeout_ts: Pointer[Int64, MutUntrackedOrigin]

    def __init__(out self):
        self.ring_fd      = -1
        self.sq_ring      = null_ptr[UInt8, MutUntrackedOrigin]()
        self.cq_ring      = null_ptr[UInt8, MutUntrackedOrigin]()
        self.sqes         = null_ptr[SQE, MutUntrackedOrigin]()
        self.sq_head      = null_ptr[UInt32, MutUntrackedOrigin]()
        self.sq_tail      = null_ptr[UInt32, MutUntrackedOrigin]()
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

    def __moveinit__(out self, deinit take: Self):
        self.ring_fd      = take.ring_fd
        self.sq_ring      = take.sq_ring
        self.cq_ring      = take.cq_ring
        self.sqes         = take.sqes
        self.sq_head      = take.sq_head
        self.sq_tail      = take.sq_tail
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

    def setup(mut self, entries: UInt32, sqpoll: Bool = False) -> Bool:
        # Allocate a 128-byte buffer for io_uring_params (kernel layout = 120 bytes).
        # Using alloc instead of address_of(embedded field) avoids origin tracking issues.
        var p = alloc[UInt8](128)
        unsafe_memset(p, 0, 128)

        # SQPOLL: set IORING_SETUP_SQPOLL flag in params.flags (offset 8).
        # The kernel spawns a dedicated SQ polling thread that consumes SQEs without
        # requiring io_uring_enter() for submission — only needed when the kernel thread
        # has gone idle (signalled via IORING_SQ_NEED_WAKEUP in sq_flags).
        # sq_thread_idle (offset 16) = 1000ms — kernel thread sleeps after 1s idle.
        if sqpoll:
            var flags_ptr = (p.unsafe_offset(8)).unsafe_bitcast[UInt32]()
            flags_ptr[] = UInt32(IORING_SETUP_SQPOLL)
            var idle_ptr = (p.unsafe_offset(16)).unsafe_bitcast[UInt32]()
            idle_ptr[] = UInt32(1000)  # 1000ms idle timeout

        # pion_io_uring_setup wraps syscall(SYS_io_uring_setup, entries, &params)
        var ring_fd = external_call["pion_io_uring_setup", Int32](
            entries, p.unsafe_bitcast[NoneType]()
        )
        if ring_fd < 0:
            p.unsafe_free()
            return False
        self.ring_fd = Int32(ring_fd)

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
            return False
        self.sq_ring = sq_ring_ptr.unsafe_bitcast[UInt8]()

        # mmap SQEs
        var sqes_sz = Int(sq_entries) * 64  # sizeof(SQE) = 64
        var sqes_ptr = external_call["pion_mmap_uring", Pointer[NoneType, MutUntrackedOrigin]](
            sqes_sz,
            3, 1, self.ring_fd, Int64(IORING_OFF_SQES)
        )
        if Int(sqes_ptr) == -1:
            return False
        self.sqes = sqes_ptr.unsafe_bitcast[SQE]()

        # mmap CQ ring (on Linux 5.4+ CQ shares mmap with SQ ring)
        var cq_ring_sz = Int(self.cq_off.cqes) + Int(cq_entries) * 16
        var cq_ring_ptr = external_call["pion_mmap_uring", Pointer[NoneType, MutUntrackedOrigin]](
            cq_ring_sz,
            3, 1, self.ring_fd, Int64(IORING_OFF_CQ_RING)
        )
        if Int(cq_ring_ptr) == -1:
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

        var cq_base = self.cq_ring
        self.cq_head      = (cq_base.unsafe_offset(Int(self.cq_off.head))).unsafe_bitcast[UInt32]()
        self.cq_tail      = (cq_base.unsafe_offset(Int(self.cq_off.tail))).unsafe_bitcast[UInt32]()
        self.cq_ring_mask = (cq_base.unsafe_offset(Int(self.cq_off.ring_mask))).unsafe_bitcast[UInt32]()[]
        self.cqes         = (cq_base.unsafe_offset(Int(self.cq_off.cqes))).unsafe_bitcast[CQE]()

        if sqpoll:
            print("io_uring SQPOLL active — kernel SQ polling thread spawned (idle_timeout=1000ms)")

        return True

    @always_inline
    def _get_sqe(mut self) -> Pointer[SQE, MutUntrackedOrigin]:
        var tail = self.sq_tail[]
        var idx = tail & self.sq_ring_mask
        self.sq_array[unsafe_offset=Int(idx)] = idx
        self.sq_tail[] = tail + 1
        return self.sqes.unsafe_offset(Int(idx))

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
        # Embed listen fd in low 32 bits; high 32 bits = 0xFFFFFFFF marks accept.
        sqe[].user_data = UDATA_SERVER_ACCEPT | UInt64(server_fd)

    @always_inline
    def submit_timeout(mut self, ms: Int):
        """gh #173: relative OP_TIMEOUT so enter() can't sleep unboundedly while
        a blocked XREAD needs time-driven expiry. Completion res is -ETIME by
        design; recognized (and swallowed) via UDATA_TIMEOUT."""
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
        sqe[].user_data    = UDATA_TIMEOUT

    @always_inline
    def is_timeout_completion(self, user_data: UInt64) -> Bool:
        return (user_data >> 32) == UInt64(0xFFFFFFFD)

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
        sqe[].user_data    = UInt64(fd)  # recv: user_data = fd (no high bits set)

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
        sqe[].user_data    = UInt64(fd)

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
        sqe[].user_data    = UInt64(0xFFFFFFFE00000000)  # special: provide_buffers completion

    @always_inline
    def is_provide_buffers_completion(self, user_data: UInt64) -> Bool:
        return (user_data >> 32) == UInt64(0xFFFFFFFE)

    @always_inline
    def submit_send(mut self, fd: Int32, buf: Pointer[UInt8, MutUntrackedOrigin], length: Int):
        var sqe = self._get_sqe()
        sqe[].opcode       = UInt8(IORING_OP_SEND)
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
        sqe[].user_data    = UInt64(fd) | UDATA_SEND_FLAG  # bit 32 marks send

    @always_inline
    def needs_wakeup(self) -> Bool:
        """Check if the kernel SQPOLL thread has gone idle and needs a wakeup.
        Only meaningful when sqpoll_active=True. When the kernel thread is idle,
        it sets IORING_SQ_NEED_WAKEUP in sq_flags; we must call enter() with
        IORING_ENTER_SQ_WAKEUP to wake it."""
        return (self.sq_flags_ptr[] & UInt32(IORING_SQ_NEED_WAKEUP)) != 0

    @always_inline
    def enter(mut self, to_submit: Int32, min_complete: Int32):
        # SQPOLL mode: the kernel SQ polling thread handles submission automatically.
        # We only need to call io_uring_enter when:
        #   1. The kernel thread is idle (NEED_WAKEUP set) — wake it with SQ_WAKEUP flag
        #   2. We need to wait for completions (min_complete > 0) — GETEVENTS flag
        # This eliminates io_uring_enter on the hot path when the kernel thread is active.
        if self.sqpoll_active:
            if min_complete > 0:
                # Need CQEs — must enter with GETEVENTS (also wakes SQ thread if needed)
                var flags = UInt32(IORING_ENTER_GETEVENTS)
                if self.needs_wakeup():
                    flags = flags | UInt32(IORING_ENTER_SQ_WAKEUP)
                _ = external_call["pion_io_uring_enter", Int32](
                    self.ring_fd, UInt32(0), UInt32(min_complete), flags
                )
            elif self.needs_wakeup():
                # No CQEs needed but kernel thread is idle — wake it to consume our SQEs
                _ = external_call["pion_io_uring_enter", Int32](
                    self.ring_fd, UInt32(0), UInt32(0), UInt32(IORING_ENTER_SQ_WAKEUP)
                )
            # else: kernel SQ thread is active, it will consume SQEs automatically — no syscall
            return

        # Standard mode: explicit submission + optional wait for completions
        _ = external_call["pion_io_uring_enter", Int32](
            self.ring_fd, UInt32(to_submit), UInt32(min_complete),
            UInt32(IORING_ENTER_GETEVENTS)
        )

    @always_inline
    def peek_cqe(self) -> CQEPeek:
        var head = self.cq_head[]
        var tail = self.cq_tail[]
        if head == tail:
            return CQEPeek(False, CQE(UInt64(0), Int32(0), UInt32(0)))
        var cqe = self.cqes[unsafe_offset=Int(head & self.cq_ring_mask)]
        return CQEPeek(True, cqe)

    @always_inline
    def advance_cq(mut self):
        self.cq_head[] = self.cq_head[] + 1

    @always_inline
    def is_accept_completion(self, user_data: UInt64) -> Bool:
        # High 32 bits = 0xFFFFFFFF means accept; low 32 bits = listen fd.
        return (user_data >> 32) == UInt64(0xFFFFFFFF)

    @always_inline
    def is_send_completion(self, user_data: UInt64) -> Bool:
        return (user_data & UDATA_SEND_FLAG) != 0

    @always_inline
    def fd_from_user_data(self, user_data: UInt64) -> Int32:
        return Int32(user_data & UInt64(0xFFFFFFFF))

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
