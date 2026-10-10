"""#465: the handoff primitives between the executor and its I/O threads.

Single-producer / single-consumer rings of two-word messages, the sleep flags
and the wake-ups. In their own module so the response writer (which queues
out-of-band replies and must tell the owning I/O thread) and io_threads.mojo
(which imports the writer) can both use them without an import cycle.

A wake-up is an fd. On Linux it is an eventfd. On macOS it is a kqueue that
carries one EVFILT_USER event: an I/O thread's own kqueue (so its poller wakes
on it), and a kqueue of its own for the executor.
"""
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.memory import stack_allocation, unsafe_memset
from std.memory.unsafe_pointer import Pointer
from std.sys import CompilationTarget
from src.common.ptr import null_ptr

comptime IO_MSG_DATA = UInt64(1)
comptime IO_MSG_ACCEPT = UInt64(2)
comptime IO_MSG_CLOSE = UInt64(3)
comptime IO_MSG_REPLY = UInt64(4)     # executor -> I/O: the batch is done, arg = consumed (| IO_CLOSE_FLAG)
comptime IO_MSG_KICK = UInt64(5)      # executor -> I/O: bytes were queued out of band (pub/sub, MONITOR)
comptime IO_MSG_RESUME = UInt64(6)    # executor -> I/O: a parked client was answered; send, then hand over its bytes
comptime IO_CLOSE_FLAG = 1 << 62      # in a REPLY's arg: close once the reply is out (QUIT, CLIENT KILL of self)

# pion_worker_entry's index at or above this is an I/O thread, not a worker.
comptime IO_THREAD_BASE = Int64(1 << 20)

comptime IO_RING_CAP = 1 << 16          # messages; more than one thread's connections
comptime IO_RING_MASK = IO_RING_CAP - 1
comptime IO_RING_WORDS = 16 + 2 * IO_RING_CAP   # [head, pad x7, tail, pad x7, slots...]
comptime IO_MAX_FDS = 65536
comptime IO_SPIN = 256                  # empty polls before a thread sleeps
comptime EPOLLRDHUP = UInt32(0x2000)

# macOS kqueue. A struct kevent is 32 bytes: ident u64, filter i16, flags u16,
# fflags u32, data i64, udata ptr.
comptime KEV_BYTES = 32
comptime EVFILT_READ = Int16(-1)
comptime EVFILT_WRITE = Int16(-2)
comptime EVFILT_USER = Int16(-10)
comptime KEV_ADD = UInt16(0x0001)
comptime KEV_DELETE = UInt16(0x0002)
comptime KEV_CLEAR = UInt16(0x0020)
comptime NOTE_TRIGGER = UInt32(0x01000000)
comptime WAKE_IDENT = 1                 # EVFILT_USER's ident: its own namespace, not an fd


@fieldwise_init
struct IOMsg(Copyable, Movable):
    var ok: Bool
    var kind: UInt64
    var fd: Int32
    var arg: Int


@always_inline
def ring_at(base: Pointer[UInt64, MutUntrackedOrigin], t: Int) -> Pointer[UInt64, MutUntrackedOrigin]:
    return base.unsafe_offset(t * IO_RING_WORDS)


@always_inline
def ring_push(r: Pointer[UInt64, MutUntrackedOrigin], kind: UInt64, fd: Int32, arg: Int) -> Bool:
    """Producer side. False when the ring is full."""
    var tail = r[unsafe_offset=8]                     # only the producer writes it
    var head = Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](r)
    if tail - head >= UInt64(IO_RING_CAP):
        return False
    var i = 16 + 2 * (Int(tail) & IO_RING_MASK)
    r[unsafe_offset=i] = (kind << 32) | UInt64(UInt32(fd))
    r[unsafe_offset=i + 1] = UInt64(arg)
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](r.unsafe_offset(8), tail + 1)
    return True


@always_inline
def ring_pop(r: Pointer[UInt64, MutUntrackedOrigin]) -> IOMsg:
    """Consumer side."""
    var head = r[unsafe_offset=0]                     # only the consumer writes it
    var tail = Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](r.unsafe_offset(8))
    if head == tail:
        return IOMsg(False, 0, 0, 0)
    var i = 16 + 2 * (Int(head) & IO_RING_MASK)
    var w0 = r[unsafe_offset=i]
    var w1 = r[unsafe_offset=i + 1]
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](r, head + 1)
    return IOMsg(True, w0 >> 32, Int32(UInt32(w0 & 0xFFFFFFFF)), Int(w1))


@always_inline
def ring_nonempty(r: Pointer[UInt64, MutUntrackedOrigin]) -> Bool:
    return Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](r.unsafe_offset(8)) != \
           Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](r)


@always_inline
def is_sleeping(flag: Pointer[UInt64, MutUntrackedOrigin]) -> Bool:
    """Full barrier, then the flag: pairs with the sleeper's store-then-check."""
    return Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.SEQUENTIAL](flag, UInt64(0)) != 0


@always_inline
def kev_change(kq: Int32, ident: Int, filter: Int16, flags: UInt16, fflags: UInt32) -> Int32:
    """macOS: apply one kevent change, return no events."""
    var ev = stack_allocation[KEV_BYTES, UInt8]()
    unsafe_memset(ev, 0, KEV_BYTES)
    ev.bitcast[UInt64]()[] = UInt64(ident)
    ev.unsafe_offset(8).bitcast[Int16]()[] = filter
    ev.unsafe_offset(10).bitcast[UInt16]()[] = flags
    ev.unsafe_offset(12).bitcast[UInt32]()[] = fflags
    return external_call["kevent", Int32](kq, ev, Int32(1), null_ptr[UInt8, MutUntrackedOrigin](),
                                          Int32(0), null_ptr[NoneType, MutUntrackedOrigin]())


def wake_new() -> Int32:
    """A new wake-up fd (see the module docstring)."""
    comptime if CompilationTarget.is_linux():
        return external_call["eventfd", Int32](UInt32(0), Int32(0x800 | 0x80000))   # EFD_NONBLOCK | EFD_CLOEXEC
    else:
        var kq = external_call["kqueue", Int32]()
        if kq >= 0:
            _ = kev_change(kq, WAKE_IDENT, EVFILT_USER, KEV_ADD | KEV_CLEAR, UInt32(0))
        return kq


def evfd_signal(fd: Int32):
    comptime if CompilationTarget.is_linux():
        var one = stack_allocation[1, UInt64]()
        one[] = 1
        _ = external_call["write", Int](Int(fd), one, 8)
    else:
        _ = kev_change(fd, WAKE_IDENT, EVFILT_USER, UInt16(0), NOTE_TRIGGER)


def evfd_drain(fd: Int32):
    """Linux: reset the eventfd. macOS: nothing to do; EV_CLEAR resets the
    user event when it is delivered."""
    comptime if CompilationTarget.is_linux():
        var v = stack_allocation[1, UInt64]()
        _ = external_call["read", Int](Int(fd), v, 8)


def wake_wait(fd: Int32, timeout_ms: Int):
    """Sleep until fd is signalled or timeout_ms passes (the executor's sleep)."""
    comptime if CompilationTarget.is_linux():
        var pfd = stack_allocation[2, Int32]()   # struct pollfd {int fd; short events; short revents}
        pfd[0] = fd
        pfd[1] = Int32(1)                        # events = POLLIN, revents = 0
        if external_call["poll", Int32](pfd, Int64(1), Int32(timeout_ms)) > 0:
            evfd_drain(fd)
    else:
        var ev = stack_allocation[KEV_BYTES, UInt8]()
        var ts = stack_allocation[2, Int]()      # struct timespec
        ts[0] = 0
        ts[1] = timeout_ms * 1_000_000
        _ = external_call["kevent", Int32](fd, null_ptr[UInt8, MutUntrackedOrigin](), Int32(0),
                                           ev, Int32(1), ts)


@always_inline
def io_errno() -> Int32:
    comptime if CompilationTarget.is_linux():
        return external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
    else:
        return external_call["__error", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]


@always_inline
def io_eagain() -> Int32:
    comptime if CompilationTarget.is_linux():
        return 11
    else:
        return 35
