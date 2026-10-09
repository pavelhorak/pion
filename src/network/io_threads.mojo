"""#465: I/O threads for ONE keyspace (prototype: Linux, epoll, -w 1).

The worker thread stays the only thread that touches the keyspace, the WAL,
transactions, ACL/tenant state, blocked clients and pub/sub: it is the
EXECUTOR. `--io-threads N` adds N - 1 I/O threads (Redis counts io-threads the
same way) that own the client sockets. Each I/O thread accepts on the shared
listen socket, receives into the connection's input buffer, hands the bytes
to the executor, and sends the reply the executor queued for the connection.

Per connection at most one batch is with the executor at a time, so replies
leave in request order. While a batch is out, the I/O thread may still append
newly received bytes BEHIND it (the 256 MB input buffer never moves), but it
neither moves nor frees anything the executor reads.

Handoff runs through two single-producer / single-consumer rings per I/O
thread: I/O -> executor carries DATA (a batch), ACCEPT and CLOSE; executor ->
I/O carries REPLY (the batch's reply is queued, so many bytes were consumed).
Reply bytes travel in the writer's existing per-connection output queue
(WriterCtx.pending_buffers / the overflow queue): the executor fills it while
it holds the batch, the I/O thread drains it once the REPLY arrives.

A side with nothing to do sleeps: an I/O thread in epoll_wait, the executor in
poll() on an eventfd. The other side writes the sleeper's eventfd only when
the sleeper has announced it is sleeping, behind a full barrier on both sides
(store flag / load ring vs store ring / load flag), so a wake-up is never lost.

Step 2 of #465 is everything a reply that does not answer a request needs:
pub/sub and MONITOR delivery, blocked and parked clients, CLIENT KILL, the
binary lane and the affinity ports. Until then those paths are not wired here.
"""
from src.common.ptr import is_not_null, is_null, null_ptr
from std.atomic import Atomic, Ordering
from std.ffi import external_call
from std.memory import alloc, unsafe_memset, stack_allocation
from std.memory.unsafe_pointer import Pointer
from std.sys import CompilationTarget
from src.network.server import EPOLLIN, EPOLLOUT, EPOLLERR, EPOLLHUP, EPOLLET, EPOLLEXCLUSIVE
from src.network.server import EPOLL_CTL_ADD, EPOLL_CTL_DEL, EPOLL_CTL_MOD
from src.network.server import epoll_ev_events, epoll_ev_fd, epoll_ctl_fd
from src.network.response_writer import WriterCtx
from src.network.io_ring import (
    IO_MSG_DATA, IO_MSG_ACCEPT, IO_MSG_CLOSE, IO_MSG_REPLY, IO_MSG_KICK, IO_MSG_RESUME, IO_CLOSE_FLAG, IO_THREAD_BASE, IO_RING_WORDS, IO_MAX_FDS, IO_SPIN, EPOLLRDHUP, IOMsg, ring_at, ring_push, ring_pop, ring_nonempty, is_sleeping, evfd_signal, evfd_drain, _errno,
)

struct IOHub(Movable):
    """What the executor and the I/O threads share. Heap-allocated once by the
    executor; lives as long as the process."""
    var n_io: Int
    var listen_fd: Int32
    var listen_fd2: Int32           # the worker's affinity port (port + 2 + worker id), -1 = none
    var buf_cap: Int
    var stop: Pointer[UInt64, MutUntrackedOrigin]
    var rings_in: Pointer[UInt64, MutUntrackedOrigin]     # I/O t -> executor
    var rings_out: Pointer[UInt64, MutUntrackedOrigin]    # executor -> I/O t
    var io_evfd: Pointer[Int32, MutUntrackedOrigin]
    var io_sleep: Pointer[UInt64, MutUntrackedOrigin]     # 8-word stride per thread
    var exec_evfd: Int32
    var exec_sleep: Pointer[UInt64, MutUntrackedOrigin]
    # The engine's per-connection tables.
    var client_buffers: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    var wctx: Pointer[WriterCtx, MutUntrackedOrigin]
    # Per-connection state of the I/O thread that owns the fd.
    var in_len: Pointer[Int, MutUntrackedOrigin]      # bytes in client_buffers[fd]
    var handed: Pointer[Int, MutUntrackedOrigin]      # bytes in the batch with the executor (0 = none)
    var closing: Pointer[UInt8, MutUntrackedOrigin]   # EOF seen while a batch was out
    var out_wait: Pointer[UInt8, MutUntrackedOrigin]  # EPOLLOUT armed: reply bytes left
    var owner_ep: Pointer[Int32, MutUntrackedOrigin]  # the owning thread's epoll fd
    var owner_t: Pointer[Int32, MutUntrackedOrigin]   # the owning thread's index, -1 = none
    var out_lock: Pointer[UInt32, MutUntrackedOrigin] # guards WriterCtx's queue for the fd

    def __init__(out self, n_io: Int, listen_fd: Int32, buf_cap: Int,
                 client_buffers: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                 wctx: Pointer[WriterCtx, MutUntrackedOrigin]):
        self.n_io = n_io
        self.listen_fd = listen_fd
        self.listen_fd2 = Int32(-1)
        self.buf_cap = buf_cap
        self.stop = alloc[UInt64](8)
        self.stop[] = 0
        self.rings_in = alloc[UInt64](n_io * IO_RING_WORDS)
        self.rings_out = alloc[UInt64](n_io * IO_RING_WORDS)
        unsafe_memset(self.rings_in.bitcast[UInt8](), 0, n_io * IO_RING_WORDS * 8)
        unsafe_memset(self.rings_out.bitcast[UInt8](), 0, n_io * IO_RING_WORDS * 8)
        self.io_evfd = alloc[Int32](n_io)
        self.io_sleep = alloc[UInt64](n_io * 8)
        unsafe_memset(self.io_sleep.bitcast[UInt8](), 0, n_io * 64)
        self.exec_evfd = Int32(-1)
        comptime if CompilationTarget.is_linux():
            for t in range(n_io):
                self.io_evfd[unsafe_offset=t] = external_call["eventfd", Int32](UInt32(0), Int32(0x800 | 0x80000))
            self.exec_evfd = external_call["eventfd", Int32](UInt32(0), Int32(0x800 | 0x80000))
        self.exec_sleep = alloc[UInt64](8)
        self.exec_sleep[] = 0
        self.client_buffers = client_buffers
        self.wctx = wctx
        self.in_len = alloc[Int](IO_MAX_FDS)
        self.handed = alloc[Int](IO_MAX_FDS)
        self.closing = alloc[UInt8](IO_MAX_FDS)
        self.out_wait = alloc[UInt8](IO_MAX_FDS)
        self.owner_ep = alloc[Int32](IO_MAX_FDS)
        self.owner_t = alloc[Int32](IO_MAX_FDS)
        self.out_lock = alloc[UInt32](IO_MAX_FDS)
        for i in range(IO_MAX_FDS):
            self.in_len[unsafe_offset=i] = 0
            self.handed[unsafe_offset=i] = 0
            self.closing[unsafe_offset=i] = 0
            self.out_wait[unsafe_offset=i] = 0
            self.owner_ep[unsafe_offset=i] = -1
            self.owner_t[unsafe_offset=i] = -1
            self.out_lock[unsafe_offset=i] = 0


# ── the I/O thread ────────────────────────────────────────────────────────────

struct _IOThread:
    """One I/O thread's loop state; lives on that thread's stack frame."""
    var hub: Pointer[IOHub, MutUntrackedOrigin]
    var t: Int
    var ep: Int32
    var rin: Pointer[UInt64, MutUntrackedOrigin]
    var rout: Pointer[UInt64, MutUntrackedOrigin]
    var pushed: Bool

    def __init__(out self, hub: Pointer[IOHub, MutUntrackedOrigin], t: Int, ep: Int32):
        self.hub = hub
        self.t = t
        self.ep = ep
        self.rin = ring_at(hub[].rings_in, t)
        self.rout = ring_at(hub[].rings_out, t)
        self.pushed = False

    def push(mut self, kind: UInt64, fd: Int32, arg: Int):
        while not ring_push(self.rin, kind, fd, arg):
            # The executor is behind by IO_RING_CAP messages: wake it and wait.
            evfd_signal(self.hub[].exec_evfd)
            _ = external_call["sched_yield", Int32]()
        self.pushed = True

    def wake_executor(mut self):
        if self.pushed:
            self.pushed = False
            if is_sleeping(self.hub[].exec_sleep):
                evfd_signal(self.hub[].exec_evfd)

    def handoff(mut self, fd: Int32):
        var ci = Int(fd)
        self.hub[].handed[unsafe_offset=ci] = self.hub[].in_len[unsafe_offset=ci]
        self.push(IO_MSG_DATA, fd, self.hub[].in_len[unsafe_offset=ci])

    def close_conn(mut self, fd: Int32):
        """Stop serving fd and tell the executor, which closes the socket."""
        var ci = Int(fd)
        _ = epoll_ctl_fd(self.ep, EPOLL_CTL_DEL, fd, 0)
        self.hub[].in_len[unsafe_offset=ci] = 0
        self.hub[].handed[unsafe_offset=ci] = 0
        self.hub[].closing[unsafe_offset=ci] = 0
        self.hub[].out_wait[unsafe_offset=ci] = 0
        self.hub[].owner_ep[unsafe_offset=ci] = -1
        self.hub[].owner_t[unsafe_offset=ci] = -1
        self.push(IO_MSG_CLOSE, fd, 0)

    def accept_all(mut self, lfd: Int32):
        comptime if not CompilationTarget.is_linux():
            return
        while True:
            var nfd = external_call["accept4", Int32](lfd,
                null_ptr[NoneType, MutUntrackedOrigin](), null_ptr[NoneType, MutUntrackedOrigin](),
                Int32(0x800))                                   # SOCK_NONBLOCK
            if nfd < 0:
                return
            if Int(nfd) >= IO_MAX_FDS:
                _ = external_call["close", Int32](nfd)
                continue
            var one = stack_allocation[1, Int32]()
            one[] = 1
            _ = external_call["setsockopt", Int32](nfd, Int32(6), Int32(1), one, UInt32(4))  # TCP_NODELAY
            var ci = Int(nfd)
            self.hub[].in_len[unsafe_offset=ci] = 0
            self.hub[].handed[unsafe_offset=ci] = 0
            self.hub[].closing[unsafe_offset=ci] = 0
            self.hub[].out_wait[unsafe_offset=ci] = 0
            self.hub[].owner_ep[unsafe_offset=ci] = self.ep
            self.hub[].owner_t[unsafe_offset=ci] = Int32(self.t)
            # ACCEPT first: the executor learns of the fd before any of its data.
            self.push(IO_MSG_ACCEPT, nfd, 0)
            _ = epoll_ctl_fd(self.ep, EPOLL_CTL_ADD, nfd, EPOLLIN | EPOLLRDHUP | EPOLLET)

    def read_conn(mut self, fd: Int32):
        """Edge-triggered: read until EAGAIN, appending behind any batch that is out."""
        var ci = Int(fd)
        var buf = self.hub[].client_buffers[unsafe_offset=ci]
        if is_null(buf):
            buf = alloc[UInt8](self.hub[].buf_cap)
            self.hub[].client_buffers[unsafe_offset=ci] = buf
        var eof = False
        while True:
            var have = self.hub[].in_len[unsafe_offset=ci]
            var room = self.hub[].buf_cap - have
            if room <= 0:
                eof = True            # one unfinished request filled the buffer: it can never complete
                break
            var n = Int(external_call["recv", Int64](fd, buf.unsafe_offset(have), room, Int32(0)))
            if n > 0:
                self.hub[].in_len[unsafe_offset=ci] = have + n
                continue
            if n == 0:
                eof = True
            elif _errno() != 11:      # EAGAIN
                eof = True
            break
        if eof:
            # Bytes that came with the FIN still run and get their reply; the
            # close follows the batch (after_reply). A batch already out ends the same way.
            self.hub[].closing[unsafe_offset=ci] = 1
            if self.hub[].handed[unsafe_offset=ci] == 0:
                if self.hub[].out_wait[unsafe_offset=ci] != 0:
                    return                # its reply is still going out; EPOLLOUT closes it
                if self.hub[].in_len[unsafe_offset=ci] > 0:
                    self.handoff(fd)
                else:
                    self.close_conn(fd)
            return
        if self.hub[].handed[unsafe_offset=ci] == 0 and self.hub[].out_wait[unsafe_offset=ci] == 0 \
           and self.hub[].in_len[unsafe_offset=ci] > 0:
            self.handoff(fd)

    def send_conn(mut self, fd: Int32) -> Bool:
        """Send what the connection is owed. True when all of it went out.
        Holds the fd's output lock: the executor may queue an out-of-band
        reply (pub/sub, a woken client) for it at any time."""
        var lk = self.hub[].out_lock.unsafe_offset(Int(fd))
        external_call["pion_spin_lock", NoneType](lk)
        var r = self._send_locked(fd)
        external_call["pion_spin_unlock", NoneType](lk)
        return r

    def _send_locked(mut self, fd: Int32) -> Bool:
        var ci = Int(fd)
        var ctx = self.hub[].wctx
        while True:
            if ctx[].pending_offsets[unsafe_offset=ci] == 0:
                ctx[].out_refill(ci)
                if ctx[].pending_offsets[unsafe_offset=ci] == 0:
                    if self.hub[].out_wait[unsafe_offset=ci] != 0:
                        self.hub[].out_wait[unsafe_offset=ci] = 0
                        _ = epoll_ctl_fd(self.ep, EPOLL_CTL_MOD, fd, EPOLLIN | EPOLLRDHUP | EPOLLET)
                    return True
            var owed = ctx[].pending_offsets[unsafe_offset=ci]
            var buf = ctx[].pending_buffers[unsafe_offset=ci]
            var n = Int(external_call["send", Int64](fd, buf, owed, Int32(0x4000)))   # MSG_NOSIGNAL
            if n >= owed:
                ctx[].pending_offsets[unsafe_offset=ci] = 0
                continue
            if n > 0:
                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                    buf.unsafe_bitcast[NoneType](), buf.unsafe_offset(n).unsafe_bitcast[NoneType](), owed - n)
                ctx[].pending_offsets[unsafe_offset=ci] = owed - n
                continue
            if _errno() == 11:        # EAGAIN: finish on EPOLLOUT
                if self.hub[].out_wait[unsafe_offset=ci] == 0:
                    self.hub[].out_wait[unsafe_offset=ci] = 1
                    _ = epoll_ctl_fd(self.ep, EPOLL_CTL_MOD, fd, EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET)
                return False
            # A dead connection: drop what it is owed; the read side sees the error.
            ctx[].pending_offsets[unsafe_offset=ci] = 0
            ctx[].out_free(ci)
            return True

    def after_reply(mut self, fd: Int32, arg: Int):
        """The executor is done with fd's batch: its reply is queued."""
        var ci = Int(fd)
        var consumed = arg & (IO_CLOSE_FLAG - 1)
        if (arg & IO_CLOSE_FLAG) != 0:
            self.hub[].closing[unsafe_offset=ci] = 1     # QUIT, a protocol error, CLIENT KILL of itself
        var handed = self.hub[].handed[unsafe_offset=ci]
        self.hub[].handed[unsafe_offset=ci] = 0
        var have = self.hub[].in_len[unsafe_offset=ci]
        if consumed > 0:
            var buf = self.hub[].client_buffers[unsafe_offset=ci]
            if have > consumed:
                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                    buf.unsafe_bitcast[NoneType](), buf.unsafe_offset(consumed).unsafe_bitcast[NoneType](),
                    have - consumed)
            have -= consumed
            self.hub[].in_len[unsafe_offset=ci] = have
        var drained = self.send_conn(fd)
        if self.hub[].closing[unsafe_offset=ci] != 0:
            if drained:
                self.close_conn(fd)
            return                        # else: closes once EPOLLOUT drains the rest
        # Bytes the executor has not looked at yet arrived meanwhile: next batch.
        if drained and have > handed - consumed:
            self.handoff(fd)

    def drain_replies(mut self) -> Int:
        var n = 0
        while True:
            var m = ring_pop(self.rout)
            if not m.ok:
                return n
            n += 1
            var ci = Int(m.fd)
            if self.hub[].owner_ep[unsafe_offset=ci] != self.ep:
                continue                  # closed meanwhile (its number may be another thread's now)
            if m.kind == IO_MSG_REPLY:
                self.after_reply(m.fd, m.arg)
            elif m.kind == IO_MSG_KICK:
                _ = self.send_conn(m.fd)
            elif m.kind == IO_MSG_RESUME:
                # A parked client was answered: its reply is queued; then what it
                # pipelined behind the command that parked it.
                if self.send_conn(m.fd) and self.hub[].handed[unsafe_offset=ci] == 0 \
                   and self.hub[].in_len[unsafe_offset=ci] > 0:
                    self.handoff(m.fd)


def io_thread_main(hub: Pointer[IOHub, MutUntrackedOrigin], t: Int):
    comptime if CompilationTarget.is_linux():
        _io_thread_loop(hub, t)


def _io_thread_loop(hub: Pointer[IOHub, MutUntrackedOrigin], t: Int):
    var ep = external_call["epoll_create1", Int32](Int32(0x80000))
    _ = epoll_ctl_fd(ep, EPOLL_CTL_ADD, hub[].listen_fd, EPOLLIN | EPOLLEXCLUSIVE)
    if hub[].listen_fd2 >= 0:
        _ = epoll_ctl_fd(ep, EPOLL_CTL_ADD, hub[].listen_fd2, EPOLLIN | EPOLLEXCLUSIVE)
    var evfd = hub[].io_evfd[unsafe_offset=t]
    _ = epoll_ctl_fd(ep, EPOLL_CTL_ADD, evfd, EPOLLIN)
    var st = _IOThread(hub, t, ep)
    var events = alloc[UInt8](512 * 16)
    var flag = hub[].io_sleep.unsafe_offset(t * 8)
    var idle = 0
    while hub[].stop[] == 0:
        var did = st.drain_replies()
        var timeout = Int32(0)
        if did == 0 and idle >= IO_SPIN:
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.SEQUENTIAL](flag, UInt64(1))
            if not ring_nonempty(st.rout):
                timeout = Int32(1)
        var n = Int(external_call["epoll_wait", Int32](ep, events, Int32(512), timeout))
        if timeout != 0 or did == 0:
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELAXED](flag, UInt64(0))
        if n <= 0:
            if did == 0:
                idle += 1
            else:
                idle = 0
            st.wake_executor()
            continue
        idle = 0
        for i in range(n):
            var fd = epoll_ev_fd(events, i)
            var mask = epoll_ev_events(events, i)
            if fd == evfd:
                evfd_drain(evfd)
                continue
            if fd == hub[].listen_fd or fd == hub[].listen_fd2:
                st.accept_all(fd)
                continue
            var ci = Int(fd)
            if hub[].owner_ep[unsafe_offset=ci] != ep:
                continue              # closed earlier in this batch
            if mask & EPOLLOUT:
                if st.send_conn(fd) and hub[].handed[unsafe_offset=ci] == 0:
                    if hub[].closing[unsafe_offset=ci] != 0:
                        st.close_conn(fd)
                        continue
                    if hub[].in_len[unsafe_offset=ci] > 0:
                        st.handoff(fd)
            if mask & (EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLRDHUP):
                st.read_conn(fd)
        st.wake_executor()
