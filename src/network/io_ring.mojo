"""#465: the handoff primitives between the executor and its I/O threads.

Single-producer / single-consumer rings of two-word messages, the sleep flags
and eventfd wake-ups. In their own module so the response writer (which queues
out-of-band replies and must tell the owning I/O thread) and io_threads.mojo
(which imports the writer) can both use them without an import cycle.
"""
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.memory import stack_allocation
from std.memory.unsafe_pointer import Pointer

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


def evfd_signal(fd: Int32):
    var one = stack_allocation[1, UInt64]()
    one[] = 1
    _ = external_call["write", Int](Int(fd), one, 8)


def evfd_drain(fd: Int32):
    var v = stack_allocation[1, UInt64]()
    _ = external_call["read", Int](Int(fd), v, 8)


@always_inline
def _errno() -> Int32:
    return external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]


