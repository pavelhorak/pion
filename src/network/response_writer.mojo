from src.common.ptr import null_ptr, is_null, is_not_null
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy
from std.ffi import external_call
from std.sys import CompilationTarget

from src.network.server import TCPServer
from src.common.value import GenericValue, ValueType
from src.common.utils import int_string_len, format_int_to_buf, score_prints_as_int, format_score
from src.io.io_uring import IOUring

# Response buffer size. Kept at 4MB for cache-friendly vector search performance.
# LMCache large-value GET uses writev to bypass this buffer entirely.
comptime RESP_BUF_SIZE = 4 * 1024 * 1024  # 4 MB
# Where an append stops filling `buffer`. The 194 KB past it are slack for the
# fixed-size writes that land between two checks (headers, word stores).
comptime RESP_LIMIT = RESP_BUF_SIZE - 194304
# #49: a connection's pending block. What it is owed past this waits in an
# overflow queue behind it, which has no size limit, as in Redis, where a
# normal client's output buffer is unlimited by default.
comptime OUT_BLOCK = RESP_BUF_SIZE
# #49: what a connection may be owed before a delivery to it (a published
# message, a MONITOR line) shuts it down instead: Redis's pubsub hard limit.
comptime OUT_DELIVER_LIMIT = 32 * 1024 * 1024


@always_inline
def _send_errno() -> Int32:
    comptime if CompilationTarget.is_linux():
        return external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
    else:
        return external_call["__error", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]


@always_inline
def _EAGAIN() -> Int32:
    comptime if CompilationTarget.is_linux():
        return 11
    else:
        return 35


@always_inline
def _send_nowait(fd: Int32, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    """#49: send() that never blocks, whatever the socket's mode (io_uring
    accepts blocking sockets). SIGPIPE is ignored process-wide; Linux also
    gets MSG_NOSIGNAL, as TCPServer.send passes it."""
    comptime if CompilationTarget.is_linux():
        return Int(external_call["send", Int64](fd, p, n, Int32(0x4000 | 0x40)))   # MSG_NOSIGNAL | MSG_DONTWAIT
    else:
        return Int(external_call["send", Int64](fd, p, n, Int32(0x80)))            # MSG_DONTWAIT


def _write_overflow_error_bytes(dst: Pointer[UInt8, MutUntrackedOrigin]):
    """gh #82: write `-ERR response exceeds buffer\r\n` (30 bytes) at dst.
    Free function (not a method) — see `_emit_overflow_error` for why method
    calls in this code path were avoided."""
    dst[unsafe_offset=0]  = 45;  dst[unsafe_offset=1]  = 69;  dst[unsafe_offset=2]  = 82;  dst[unsafe_offset=3]  = 82
    dst[unsafe_offset=4]  = 32;  dst[unsafe_offset=5]  = 114; dst[unsafe_offset=6]  = 101; dst[unsafe_offset=7]  = 115
    dst[unsafe_offset=8]  = 112; dst[unsafe_offset=9]  = 111; dst[unsafe_offset=10] = 110; dst[unsafe_offset=11] = 115
    dst[unsafe_offset=12] = 101; dst[unsafe_offset=13] = 32;  dst[unsafe_offset=14] = 101; dst[unsafe_offset=15] = 120
    dst[unsafe_offset=16] = 99;  dst[unsafe_offset=17] = 101; dst[unsafe_offset=18] = 101; dst[unsafe_offset=19] = 100
    dst[unsafe_offset=20] = 115; dst[unsafe_offset=21] = 32;  dst[unsafe_offset=22] = 98;  dst[unsafe_offset=23] = 117
    dst[unsafe_offset=24] = 102; dst[unsafe_offset=25] = 102; dst[unsafe_offset=26] = 101; dst[unsafe_offset=27] = 114
    dst[unsafe_offset=28] = 13;  dst[unsafe_offset=29] = 10


@fieldwise_init
struct IOVec(Copyable, Movable, ImplicitlyCopyable):
    var iov_base: Pointer[UInt8, MutUntrackedOrigin]
    var iov_len: Int


struct ResponseWriter(Movable):
    var buffer: Pointer[UInt8, MutUntrackedOrigin]
    var offset: Int
    var pending_offsets: Pointer[Int, MutUntrackedOrigin]
    var pending_buffers: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    # io_uring send path (Linux only; use_uring=False on macOS kqueue path)
    var use_uring: Bool
    var ring: Pointer[IOUring, MutUntrackedOrigin]
    # uring_inflight[fd]: byte length of the currently in-flight SEND SQE (0 = none).
    # pending_offsets[fd] may grow beyond uring_inflight[fd] as more responses are appended
    # during an in-flight send; SEND completion drains the accumulated remainder.
    var uring_inflight: Pointer[Int, MutUntrackedOrigin]
    # Set when an append in the current pre-flush batch would have overflowed the
    # 4 MB buffer and we emitted a `-ERR response exceeds buffer\r\n` frame in its
    # place. Subsequent appends in the same batch are dropped (would corrupt the
    # error frame). Reset by flush_response after the buffer is drained.
    # gh #82: replaces the previous silent-drop overflow guards. #49: only on
    # the XDP lane now; everywhere else a full buffer is handed to its
    # connection and the reply goes on.
    var overflow_emitted: Bool
    # gh #172: wire protocol for the connection currently being served — 2 or 3.
    # The writer is per-worker, not per-fd, so the event loop stamps this from
    # `tx_state.resp_proto[fd]` once per recv-buffer dispatch (not per command).
    # Every RESP3-divergent appender branches on it with RESP2 as the
    # fall-through, so the RESP2 hot path keeps its original instruction count.
    #
    # NOTE (gh #149 discipline): this field is LAST on purpose. Inserting a
    # field mid-struct shifts every field after it and measurably costs
    # throughput — `blobs`/`blob_threshold` mid-FastPathHandler cost ~1.5% by
    # itself. New fields go at the end.
    var proto: UInt8
    # gh #192: EAGAIN stalls in the blocking large-response send loops (each
    # one costs a ~150 µs usleep on macOS). Per-worker (writer is per-worker),
    # so plain increments are lockless. Exposed as INFO send_eagain_stalls —
    # the kill-test/observability signal for the substrate large-send path.
    var send_stalls: UInt64
    # #47: how many times the writer flushed. CLIENT REPLY OFF drops a
    # command's reply by cutting the buffer back to where the reply began,
    # which is only right when nothing was sent in between.
    var flush_count: Int
    # #49: the connection whose replies `buffer` holds, stamped wherever the
    # engine starts writing for one (with `proto`). A full buffer is handed to
    # it and the reply goes on. -1 where no connection can take the bytes (the
    # XDP lane): there a full buffer still becomes the -ERR frame.
    var cur_fd: Int32
    # #49: set when bytes were queued for `cur_fd` with `buffer` left empty, so
    # the engine flushes after the batch even though `offset` is 0.
    var queued: Bool
    # #49: what a connection is owed past its pending block, in order:
    # pending_buffers[fd][0, pending_offsets[fd]) then
    # ovf_bufs[fd][ovf_heads[fd], ovf_lens[fd]). Allocated on first use.
    var ovf_bufs: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    var ovf_lens: Pointer[Int, MutUntrackedOrigin]
    var ovf_heads: Pointer[Int, MutUntrackedOrigin]
    var ovf_caps: Pointer[Int, MutUntrackedOrigin]
    # #49: a capture writer's reply past its buffer (a script's redis.call).
    var cap_buf: Pointer[UInt8, MutUntrackedOrigin]
    var cap_len: Int
    var cap_cap: Int

    def __init__(out self):
        self.buffer = alloc[UInt8](RESP_BUF_SIZE)
        self.offset = 0
        self.pending_offsets = alloc[Int](65536)
        self.pending_buffers = alloc[Pointer[UInt8, MutUntrackedOrigin]](65536)
        self.use_uring = False
        self.ring = null_ptr[IOUring, MutUntrackedOrigin]()
        self.uring_inflight = alloc[Int](65536)
        self.overflow_emitted = False
        self.proto = 2
        self.send_stalls = 0
        self.flush_count = 0
        self.cur_fd = -1
        self.queued = False
        self.ovf_bufs = null_ptr[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]()
        self.ovf_lens = null_ptr[Int, MutUntrackedOrigin]()
        self.ovf_heads = null_ptr[Int, MutUntrackedOrigin]()
        self.ovf_caps = null_ptr[Int, MutUntrackedOrigin]()
        self.cap_buf = null_ptr[UInt8, MutUntrackedOrigin]()
        self.cap_len = 0
        self.cap_cap = 0
        for i in range(65536):
            self.pending_offsets[unsafe_offset=i] = 0
            self.pending_buffers[unsafe_offset=i] = null_ptr[UInt8, MutUntrackedOrigin]()
            self.uring_inflight[unsafe_offset=i] = 0

    def __init__(out self, *, capture_only: Bool):
        """#36: a writer that never sends, for a script's redis.call(): its
        reply stays in `buffer` for the engine to read. It has no per-connection
        output state, which is how the paths that would write to a connection
        recognise it (`is_null(pending_offsets)`), and its flushes are given
        kq = -1, the no-op flush the XDP lane uses. (A `capture` field on every
        writer cost the MSET and GET helpers, which take the writer, about 1%
        more instructions.)"""
        self.buffer = alloc[UInt8](RESP_BUF_SIZE)
        self.offset = 0
        self.pending_offsets = null_ptr[Int, MutUntrackedOrigin]()
        self.pending_buffers = null_ptr[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]()
        self.use_uring = False
        self.ring = null_ptr[IOUring, MutUntrackedOrigin]()
        self.uring_inflight = null_ptr[Int, MutUntrackedOrigin]()
        self.overflow_emitted = False
        self.proto = 2
        self.send_stalls = 0
        self.flush_count = 0
        self.cur_fd = -1
        self.queued = False
        self.ovf_bufs = null_ptr[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]()
        self.ovf_lens = null_ptr[Int, MutUntrackedOrigin]()
        self.ovf_heads = null_ptr[Int, MutUntrackedOrigin]()
        self.ovf_caps = null_ptr[Int, MutUntrackedOrigin]()
        self.cap_buf = null_ptr[UInt8, MutUntrackedOrigin]()
        self.cap_len = 0
        self.cap_cap = 0

    @always_inline
    def bind_ring(mut self, ring_ptr: Pointer[IOUring, MutUntrackedOrigin]):
        """Wire this ResponseWriter to an IOUring instance for Linux io_uring sends."""
        self.ring = ring_ptr
        self.use_uring = True

    def _emit_overflow_error(mut self):
        """Cold-path emit of `-ERR response exceeds buffer\r\n` (30 bytes),
        only where a full buffer cannot be handed on: the XDP lane (#49).
        Deliberately NOT `@always_inline`: embedding the emit in an inlined
        appender mis-lowered under recursive dispatcher entry (MULTI/EXEC
        replay), and a non-trivial cold branch made the inliner back off the
        hot appenders."""
        if self.overflow_emitted: return
        self.overflow_emitted = True
        _write_overflow_error_bytes(self.buffer.unsafe_offset(self.offset))
        self.offset += 30

    @no_inline
    def _spill(mut self) -> Bool:
        """#49: the buffer is about to overflow. Hand what it holds to the
        connection it belongs to and start again at 0: the reply goes on, as
        Redis's goes on into its reply list. A capture writer keeps the bytes
        for the script. False where no connection can take them (cur_fd -1,
        the XDP lane)."""
        if is_null(self.pending_offsets):
            if self.offset > 0:
                self._cap_push(self.buffer, self.offset)
                self.offset = 0
            self.flush_count += 1
            return True
        if self.cur_fd < 0:
            return False
        if self.offset > 0:
            self._to_conn(self.cur_fd, self.buffer, self.offset)
            self.offset = 0
        self.flush_count += 1   # what was written is no longer in the buffer
        return True

    @no_inline
    def _put_big(mut self, src: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
        """#49: n payload bytes, however many, after what the buffer holds.
        False only on the XDP lane, where the caller has already been told
        the buffer is full."""
        if self.offset + n <= RESP_LIMIT:
            unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset), src=src, count=n)
            self.offset += n
            return True
        if not self._spill():
            return False
        if n <= RESP_LIMIT:
            unsafe_memcpy(dest=self.buffer, src=src, count=n)
            self.offset = n
            return True
        if is_null(self.pending_offsets):
            self._cap_push(src, n)
        else:
            self._to_conn(self.cur_fd, src, n)
        return True

    @no_inline
    def _framed_cold(mut self, lead: UInt8, prefix: StaticString, data: Pointer[UInt8, MutUntrackedOrigin],
                     length: Int, frame_len: Int):
        """#49: `<lead><frame_len>\r\n<prefix><data>\r\n` for a payload that
        does not fit what is left of the buffer: a bulk string (`$`), a RESP3
        verbatim string (`=`, prefix `txt:`) or a double (`,`, no length)."""
        if self.overflow_emitted:
            return
        if is_not_null(self.pending_offsets) and self.cur_fd < 0:
            self._emit_overflow_error()     # the XDP lane, as before #49
            return
        if self.offset + 64 > RESP_LIMIT:
            _ = self._spill()
        self.buffer[unsafe_offset=self.offset] = lead
        self.offset += 1
        if frame_len >= 0:
            self.offset = format_int_to_buf(self.buffer, self.offset, Int64(frame_len))
            self.buffer[unsafe_offset=self.offset] = 13
            self.buffer[unsafe_offset=self.offset + 1] = 10
            self.offset += 2
        var pl = prefix.byte_length()
        if pl > 0:
            unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset),
                          src=prefix.unsafe_ptr().unsafe_bitcast[UInt8](), count=pl)
            self.offset += pl
        _ = self._put_big(data, length)
        if self.offset + 2 > RESP_LIMIT:
            _ = self._spill()
        self.buffer[unsafe_offset=self.offset] = 13
        self.buffer[unsafe_offset=self.offset + 1] = 10
        self.offset += 2

    @no_inline
    def _raw_cold(mut self, src: Pointer[UInt8, MutUntrackedOrigin], n: Int):
        """#49: raw bytes that do not fit what is left of the buffer."""
        if self.overflow_emitted:
            return
        if is_not_null(self.pending_offsets) and self.cur_fd < 0:
            self._emit_overflow_error()     # the XDP lane, as before #49
            return
        _ = self._put_big(src, n)

    @no_inline
    def _fixed_cold(mut self) -> Bool:
        """#49: a fixed-size append found the buffer full. Spill and go on;
        False on the XDP lane, where the variable-length append that filled
        the buffer has already sent the -ERR frame."""
        if self.overflow_emitted:
            return False
        return self._spill()

    # ── #49: what a connection is owed ────────────────────────────────────────

    def _cap_push(mut self, src: Pointer[UInt8, MutUntrackedOrigin], n: Int):
        """A capture writer's reply past its buffer, kept for the script."""
        if self.cap_len + n > self.cap_cap:
            var ncap = self.cap_cap * 2
            if ncap < self.cap_len + n:
                ncap = self.cap_len + n
            if ncap < RESP_BUF_SIZE:
                ncap = RESP_BUF_SIZE
            var nb = alloc[UInt8](ncap)
            if self.cap_len > 0:
                unsafe_memcpy(dest=nb, src=self.cap_buf, count=self.cap_len)
            if is_not_null(self.cap_buf):
                self.cap_buf.unsafe_free()
            self.cap_buf = nb
            self.cap_cap = ncap
        unsafe_memcpy(dest=self.cap_buf.unsafe_offset(self.cap_len), src=src, count=n)
        self.cap_len += n

    def captured(mut self) -> Int:
        """A capture writer's whole reply, after its buffer spilled: move the
        buffer behind the spilled bytes. Read it at `cap_buf` when cap_len > 0,
        else at `buffer`; the return value is its length either way."""
        if self.cap_len == 0:
            return self.offset
        if self.offset > 0:
            self._cap_push(self.buffer, self.offset)
            self.offset = 0
        return self.cap_len

    def _ovf_init(mut self):
        self.ovf_bufs = alloc[Pointer[UInt8, MutUntrackedOrigin]](65536)
        self.ovf_lens = alloc[Int](65536)
        self.ovf_heads = alloc[Int](65536)
        self.ovf_caps = alloc[Int](65536)
        for i in range(65536):
            self.ovf_bufs[unsafe_offset=i] = null_ptr[UInt8, MutUntrackedOrigin]()
            self.ovf_lens[unsafe_offset=i] = 0
            self.ovf_heads[unsafe_offset=i] = 0
            self.ovf_caps[unsafe_offset=i] = 0

    @always_inline
    def out_overflowed(self, ci: Int) -> Bool:
        """Is the connection owed bytes past its pending block?"""
        return is_not_null(self.ovf_lens) and self.ovf_lens[unsafe_offset=ci] > self.ovf_heads[unsafe_offset=ci]

    @always_inline
    def owes(self, ci: Int) -> Bool:
        """Is the connection owed anything at all?"""
        return self.pending_offsets[unsafe_offset=ci] > 0 or self.out_overflowed(ci)

    def out_owed(self, ci: Int) -> Int:
        """How many bytes the connection is owed."""
        var n = self.pending_offsets[unsafe_offset=ci]
        if is_not_null(self.ovf_lens):
            n += self.ovf_lens[unsafe_offset=ci] - self.ovf_heads[unsafe_offset=ci]
        return n

    def _ovf_push(mut self, ci: Int, src: Pointer[UInt8, MutUntrackedOrigin], n: Int):
        if is_null(self.ovf_lens):
            self._ovf_init()
        var head = self.ovf_heads[unsafe_offset=ci]
        var used = self.ovf_lens[unsafe_offset=ci]
        var cap = self.ovf_caps[unsafe_offset=ci]
        var buf = self.ovf_bufs[unsafe_offset=ci]
        if used + n > cap:
            var live = used - head
            if is_not_null(buf) and live + n <= cap:
                # room enough once what was sent is dropped from the front
                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                    buf.unsafe_bitcast[NoneType](), buf.unsafe_offset(head).unsafe_bitcast[NoneType](), live)
            else:
                var ncap = cap * 2
                if ncap < live + n:
                    ncap = live + n
                if ncap < OUT_BLOCK:
                    ncap = OUT_BLOCK
                var nb = alloc[UInt8](ncap)
                if live > 0:
                    unsafe_memcpy(dest=nb, src=buf.unsafe_offset(head), count=live)
                if is_not_null(buf):
                    buf.unsafe_free()
                buf = nb
                self.ovf_bufs[unsafe_offset=ci] = nb
                self.ovf_caps[unsafe_offset=ci] = ncap
            used = live
            self.ovf_heads[unsafe_offset=ci] = 0
        unsafe_memcpy(dest=buf.unsafe_offset(used), src=src, count=n)
        self.ovf_lens[unsafe_offset=ci] = used + n

    def out_append(mut self, ci: Int, src: Pointer[UInt8, MutUntrackedOrigin], n: Int):
        """Queue n bytes behind everything the connection is owed: the
        pending block while it has room and nothing waits past it, then the
        overflow queue. The front of the pending block may be in a SEND the
        kernel is reading (io_uring); nothing here touches bytes already queued."""
        if n <= 0:
            return
        var p = src
        var left = n
        if not self.out_overflowed(ci):
            if is_null(self.pending_buffers[unsafe_offset=ci]):
                self.pending_buffers[unsafe_offset=ci] = alloc[UInt8](OUT_BLOCK)
            var cur = self.pending_offsets[unsafe_offset=ci]
            var take = OUT_BLOCK - cur
            if take > left:
                take = left
            if take > 0:
                unsafe_memcpy(dest=self.pending_buffers[unsafe_offset=ci].unsafe_offset(cur), src=p, count=take)
                self.pending_offsets[unsafe_offset=ci] = cur + take
                p = p.unsafe_offset(take)
                left -= take
        if left > 0:
            self._ovf_push(ci, p, left)

    def out_refill(mut self, ci: Int):
        """Move what waits past the pending block into it, as far as it has
        room. The engine calls this whenever a send shrinks the block."""
        if not self.out_overflowed(ci):
            return
        if is_null(self.pending_buffers[unsafe_offset=ci]):
            self.pending_buffers[unsafe_offset=ci] = alloc[UInt8](OUT_BLOCK)
        var cur = self.pending_offsets[unsafe_offset=ci]
        var head = self.ovf_heads[unsafe_offset=ci]
        var take = OUT_BLOCK - cur
        if take > self.ovf_lens[unsafe_offset=ci] - head:
            take = self.ovf_lens[unsafe_offset=ci] - head
        if take <= 0:
            return
        unsafe_memcpy(dest=self.pending_buffers[unsafe_offset=ci].unsafe_offset(cur),
                      src=self.ovf_bufs[unsafe_offset=ci].unsafe_offset(head), count=take)
        self.pending_offsets[unsafe_offset=ci] = cur + take
        head += take
        if head == self.ovf_lens[unsafe_offset=ci]:
            self.out_free(ci)       # all moved: give the memory back
        else:
            self.ovf_heads[unsafe_offset=ci] = head

    def out_free(mut self, ci: Int):
        """Drop what the connection was owed past its pending block (it is
        closing, or that queue is empty)."""
        if is_null(self.ovf_lens):
            return
        if is_not_null(self.ovf_bufs[unsafe_offset=ci]):
            self.ovf_bufs[unsafe_offset=ci].unsafe_free()
            self.ovf_bufs[unsafe_offset=ci] = null_ptr[UInt8, MutUntrackedOrigin]()
        self.ovf_lens[unsafe_offset=ci] = 0
        self.ovf_heads[unsafe_offset=ci] = 0
        self.ovf_caps[unsafe_offset=ci] = 0

    def _to_conn(mut self, fd: Int32, src: Pointer[UInt8, MutUntrackedOrigin], n: Int):
        """#49: n bytes for `fd`, in order behind what it is owed. When it is
        owed nothing and no SEND is in flight, the socket takes what it can
        now; the rest is queued, and `queued` makes the engine flush after
        the batch, which arms the write event or submits the SEND."""
        var ci = Int(fd)
        if self.use_uring and self.ring[].fd_closing[unsafe_offset=ci] != 0:
            return
        var sent = 0
        if not self.owes(ci) and (not self.use_uring or self.uring_inflight[unsafe_offset=ci] == 0):
            while sent < n:
                var k = _send_nowait(fd, src.unsafe_offset(sent), n - sent)
                if k <= 0:
                    if k < 0 and _send_errno() != _EAGAIN():
                        return              # a dead connection: its close is the engine's
                    break
                sent += k
        if sent < n:
            self.send_stalls += 1           # gh #192: the socket took less than it was given
            self.out_append(ci, src.unsafe_offset(sent), n - sent)
            self.queued = True

    @always_inline
    def append_to_response[origin: Origin](mut self, src: Pointer[UInt8, origin], length: Int):
        if self.offset + length > RESP_LIMIT:
            self._raw_cold(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(src)), length)
            return
        unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset), src=src, count=length)
        self.offset += length

    @always_inline
    def flush_response(mut self, fd: Int32, server: TCPServer, kq: Int32):
        self.flush_count += 1
        self.queued = False
        if self.use_uring:
            self._flush_uring(fd)
        elif kq == -1:
            # XDP path: don't flush via TCP — the XDP event loop reads
            # self.buffer[0..self.offset] and sends via AF_XDP TX ring.
            pass
        else:
            self._flush_kqueue(fd, server, kq)
        # gh #82: clear the overflow latch once the buffer has been drained
        # (offset == 0 means the bytes were either sent or moved to pending).
        # The next pre-flush batch starts with a clean slate.
        if self.offset == 0:
            self.overflow_emitted = False

    def deliver_to(mut self, fd: Int32, data: Pointer[UInt8, MutUntrackedOrigin], length: Int,
                   server: TCPServer, kq: Int32):
        """Send a whole frame to ANOTHER connection: a published message, a
        MONITOR line (#39, #42). The bytes queued for the current connection
        are not touched. Always the engine's writer, never a script's.

        What the socket cannot take now is queued for that connection, behind
        what it is already owed, and goes out on its write event. A frame that
        would take it past OUT_DELIVER_LIMIT (32 MB, Redis's pubsub hard
        limit) is never cut: the connection is shut down instead, as Redis
        disconnects a client past its output-buffer limit, because a
        subscriber that received half a frame is out of sync for good.
        Not on the XDP lane (kq == -1), which sends nothing over TCP."""
        if length <= 0 or (kq == -1 and not self.use_uring):
            return
        var ci = Int(fd)
        var p = data
        var left = length
        if self.use_uring and self.ring[].fd_closing[unsafe_offset=ci] != 0:
            return
        if not self.use_uring and not self.owes(ci):
            while left > 0:
                var n = server.send(fd, p, left)
                if n <= 0:
                    break
                p = p.unsafe_offset(n)
                left -= n
            if left == 0:
                return
            var err = _send_errno()
            if left == length and err != _EAGAIN():
                return                  # a dead connection: its close is the engine's
        if self.out_owed(ci) + left > OUT_DELIVER_LIMIT:
            _ = external_call["shutdown", Int32](fd, Int32(2))   # SHUT_RDWR: the engine sees EOF
            return
        self.out_append(ci, p, left)
        if self.use_uring:
            self.uring_kick(fd, ci)
        else:
            server.kevent_add_write(kq, fd)

    def uring_kick(mut self, fd: Int32, ci: Int):
        """io_uring: submit a SEND of the pending block when none is in
        flight, refilled from the overflow queue first."""
        if self.uring_inflight[unsafe_offset=ci] != 0:
            return
        if self.pending_offsets[unsafe_offset=ci] < OUT_BLOCK:
            self.out_refill(ci)
        if self.pending_offsets[unsafe_offset=ci] > 0:
            self.uring_inflight[unsafe_offset=ci] = self.pending_offsets[unsafe_offset=ci]
            self.ring[].submit_send(fd, self.pending_buffers[unsafe_offset=ci], self.uring_inflight[unsafe_offset=ci])

    def uring_sent(mut self, ci: Int, sent: Int):
        """io_uring: a SEND of the pending block completed with `sent` bytes.
        Drop them from its front (nothing is in flight now) and refill it."""
        var remaining = self.pending_offsets[unsafe_offset=ci] - sent
        if remaining > 0:
            _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                self.pending_buffers[unsafe_offset=ci].unsafe_bitcast[NoneType](),
                (self.pending_buffers[unsafe_offset=ci].unsafe_offset(sent)).unsafe_bitcast[NoneType](),
                remaining,
            )
            self.pending_offsets[unsafe_offset=ci] = remaining
        else:
            self.pending_offsets[unsafe_offset=ci] = 0
        self.out_refill(ci)

    @always_inline
    def _flush_uring(mut self, fd: Int32):
        """io_uring send path. Queues the response buffer behind what the
        connection is owed and submits a SEND if none is in flight; the SEND
        completion handler (uring_sent) drains the remainder."""
        var ci = Int(fd)
        if self.offset == 0 and not self.owes(ci):
            self.overflow_emitted = False
            return
        if self.ring[].fd_closing[unsafe_offset=ci] != 0:
            # The engine is closing this connection and waits for its last
            # SEND to complete before freeing the buffer a new one would read.
            self.offset = 0
            self.overflow_emitted = False
            return
        if self.offset > 0:
            self.out_append(ci, self.buffer, self.offset)
            self.offset = 0
        self.overflow_emitted = False
        # Submit SEND only if no send is currently in flight for this fd;
        # otherwise its completion sends what was queued meanwhile.
        self.uring_kick(fd, ci)

    @always_inline
    def _flush_kqueue(mut self, fd: Int32, server: TCPServer, kq: Int32):
        var fd_idx = Int(fd)

        # 1. Owed bytes already: queue this batch behind them, then send
        if self.owes(fd_idx):
            if self.offset > 0:
                self.out_append(fd_idx, self.buffer, self.offset)
                self.offset = 0
            self._drain(fd, fd_idx, server, kq)
            return

        # 2. Nothing owed
        if self.offset == 0: return
        var n = server.send(fd, self.buffer, self.offset)
        if n >= self.offset:
            self.offset = 0
        elif n > 0:
            self.out_append(fd_idx, self.buffer.unsafe_offset(n), self.offset - n)
            self.offset = 0
            server.kevent_add_write(kq, fd)
        elif n < 0:
            # EAGAIN/EWOULDBLOCK → queue and retry on EPOLLOUT/EVFILT_WRITE.
            # EPIPE/ECONNRESET → the fd is dead, don't queue (epoll_wait will
            # deliver EPOLLERR).
            if _send_errno() == _EAGAIN():
                self.out_append(fd_idx, self.buffer, self.offset)
                self.offset = 0
                server.kevent_add_write(kq, fd)
            else:
                self.offset = 0

    @no_inline
    def _drain(mut self, fd: Int32, ci: Int, server: TCPServer, kq: Int32):
        """kqueue / epoll: send what the connection is owed, for as long as
        its socket takes it; arm the write event for the rest."""
        while True:
            if self.pending_offsets[unsafe_offset=ci] == 0:
                self.out_refill(ci)
                if self.pending_offsets[unsafe_offset=ci] == 0:
                    server.kevent_del_write(kq, fd)
                    return
            var owed = self.pending_offsets[unsafe_offset=ci]
            var n = server.send(fd, self.pending_buffers[unsafe_offset=ci], owed)
            if n >= owed:
                self.pending_offsets[unsafe_offset=ci] = 0
                continue
            if n > 0:
                var remaining = owed - n
                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                    self.pending_buffers[unsafe_offset=ci].unsafe_bitcast[NoneType](),
                    (self.pending_buffers[unsafe_offset=ci].unsafe_offset(n)).unsafe_bitcast[NoneType](),
                    remaining,
                )
                self.pending_offsets[unsafe_offset=ci] = remaining
                server.kevent_add_write(kq, fd)
                return
            if _send_errno() == _EAGAIN():
                server.kevent_add_write(kq, fd)
            else:
                # EPIPE/ECONNRESET: the fd is dead; drop what it is owed, the
                # event loop closes it
                self.pending_offsets[unsafe_offset=ci] = 0
                self.out_free(ci)
            return

    @always_inline
    def append_status_response(mut self, msg: String):
        """`+msg` — a simple string (status) reply, e.g. one HELP line."""
        var line = String("+") + msg + "\r\n"
        self.append_to_response(line.unsafe_ptr(), line.byte_length())

    @always_inline
    def append_ok_response(mut self):
        # gh #82: hot path — keep the original constant guard. Fixed-byte writes
        # are protected by the safe-zone invariant maintained by variable-length
        # appenders (which emit `-ERR …` before offset crosses into the margin).
        # Probed the alternative — calling `self._emit_overflow_error()` here
        # too, so every silent-drop becomes a `-ERR` frame — and LRANGE_300
        # dropped ~10% on the Mac gate (the cold branch stops being trivial,
        # so Mojo's inliner backs off and the hot path pays the cost). The
        # invariant: any variable-length appender that brings offset into the
        # safe zone has already emitted the `-ERR` frame, so a fixed-byte
        # silent return after that is downstream of an already-overflowed
        # client signal — not a fresh silent loss. Frame-sync test:
        # `tests/test_response_buffer_overflow.py`.
        var off = self.offset
        if off + 5 > RESP_LIMIT:
            if not self._fixed_cold(): return
            off = self.offset
        # '+OK\r\n' packed LE: 0x0000000A0D4B4F2B (writes 8 bytes; extra 3 safely overwritten)
        (self.buffer.unsafe_offset(off)).unsafe_bitcast[UInt64]()[] = UInt64(0x0000000A0D4B4F2B)
        self.offset = off + 5

    @always_inline
    def append_pong_response(mut self):
        var off = self.offset
        if off + 7 > RESP_LIMIT:
            if not self._fixed_cold(): return
            off = self.offset
        # '+PONG\r\n' packed LE: 0x000A0D474E4F502B (writes 8 bytes; extra 1 safely overwritten)
        (self.buffer.unsafe_offset(off)).unsafe_bitcast[UInt64]()[] = UInt64(0x000A0D474E4F502B)
        self.offset = off + 7

    @always_inline
    def append_pong_bulk(mut self, count: Int):
        # Write `count` PONG responses as a single batched memcpy
        # +PONG\r\n = 7 bytes
        var total = count * 7
        if self.offset + total > RESP_LIMIT:
            self._pong_bulk_cold(count)
            return
        var dst = self.buffer.unsafe_offset(self.offset)
        # Seed first response, then double until done
        dst[unsafe_offset=0] = 43; dst[unsafe_offset=1] = 80; dst[unsafe_offset=2] = 79; dst[unsafe_offset=3] = 78; dst[unsafe_offset=4] = 71; dst[unsafe_offset=5] = 13; dst[unsafe_offset=6] = 10
        var written = 7
        while written + written <= total:
            unsafe_memcpy(dest=dst.unsafe_offset(written), src=dst, count=written)
            written += written
        if written < total:
            unsafe_memcpy(dest=dst.unsafe_offset(written), src=dst, count=total - written)
        self.offset += total

    @no_inline
    def _pong_bulk_cold(mut self, count: Int):
        """#49: PONGs past what the buffer holds, a buffer at a time."""
        var left = count
        while left > 0:
            var fit = (RESP_LIMIT - self.offset) // 7
            if fit <= 0:
                if not self._fixed_cold():
                    return
                continue
            var k = fit if fit < left else left
            self.append_pong_bulk(k)
            left -= k

    @always_inline
    def append_null_response(mut self):
        var off = self.offset
        if off + 5 > RESP_LIMIT:
            if not self._fixed_cold(): return
            off = self.offset
        # gh #172: RESP3 replaces both `$-1\r\n` and `*-1\r\n` with the single
        # null type `_\r\n`. RESP2 stays the fall-through so the (hot) GET-miss
        # path keeps its single packed word-store.
        if self.proto == 3:
            # '_\r\n' packed LE: 0x00000000000A0D5F (writes 8 bytes; extra 5 safely overwritten)
            (self.buffer.unsafe_offset(off)).unsafe_bitcast[UInt64]()[] = UInt64(0x00000000000A0D5F)
            self.offset = off + 3
            return
        # '$-1\r\n' packed LE: 0x0000000A0D312D24 (writes 8 bytes; extra 3 safely overwritten)
        (self.buffer.unsafe_offset(off)).unsafe_bitcast[UInt64]()[] = UInt64(0x0000000A0D312D24)
        self.offset = off + 5

    @always_inline
    def append_verbatim_response[origin: Origin](mut self, data: Pointer[UInt8, origin], length: Int):
        """RESP3 verbatim string `=<len + 4>\r\ntxt:<data>\r\n`, as Redis
        sends INFO and CLIENT INFO / LIST; RESP2 has no such type and gets the
        same text as a bulk string."""
        if self.proto != 3:
            self.append_bulk_string_response(data, length)
            return
        if self.offset + length + 32 > RESP_LIMIT:
            self._framed_cold(61, "txt:", Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(data)), length, length + 4)
            return
        self.buffer[unsafe_offset=self.offset] = 61   # '='
        self.offset += 1
        self.offset = format_int_to_buf(self.buffer, self.offset, Int64(length + 4))
        self.buffer[unsafe_offset=self.offset] = 13
        self.buffer[unsafe_offset=self.offset + 1] = 10
        self.buffer[unsafe_offset=self.offset + 2] = 116   # 't'
        self.buffer[unsafe_offset=self.offset + 3] = 120   # 'x'
        self.buffer[unsafe_offset=self.offset + 4] = 116   # 't'
        self.buffer[unsafe_offset=self.offset + 5] = 58    # ':'
        self.offset += 6
        unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset), src=data, count=length)
        self.offset += length
        self.buffer[unsafe_offset=self.offset] = 13
        self.buffer[unsafe_offset=self.offset + 1] = 10
        self.offset += 2

    @always_inline
    def append_null_array_response(mut self):
        """A nil ARRAY: `*-1` under RESP2, `_` under RESP3. Commands whose
        reply is an array (XREAD with no data, a blocking pop that timed out)
        answer nil this way in Redis, not with a nil bulk string (`$-1`)."""
        if self.offset + 8 > RESP_LIMIT:
            if not self._fixed_cold(): return
        if self.proto == 3:
            self.buffer[unsafe_offset=self.offset] = 95   # '_'
            self.buffer[unsafe_offset=self.offset + 1] = 13
            self.buffer[unsafe_offset=self.offset + 2] = 10
            self.offset += 3
        else:
            self.buffer[unsafe_offset=self.offset] = 42   # '*'
            self.buffer[unsafe_offset=self.offset + 1] = 45   # '-'
            self.buffer[unsafe_offset=self.offset + 2] = 49   # '1'
            self.buffer[unsafe_offset=self.offset + 3] = 13
            self.buffer[unsafe_offset=self.offset + 4] = 10
            self.offset += 5

    @always_inline
    def append_array_header(mut self, count: Int):
        """`*<count>\r\n`. Identical in RESP2 and RESP3 — provided so callers
        stop hand-rolling the bytes (several already did, inconsistently)."""
        if self.offset + 16 > RESP_LIMIT:
            if not self._fixed_cold(): return
        self.buffer[self.offset] = 42 # '*'
        self.offset += 1
        self.offset = format_int_to_buf(self.buffer, self.offset, Int64(count))
        self.buffer[self.offset] = 13 # '\r'
        self.buffer[self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_map_header(mut self, pairs: Int):
        """gh #172: RESP3 `%<pairs>\r\n`; RESP2 degrades to a flat array of
        2*pairs elements, which is exactly how RESP2 clients already expect
        HELLO / CONFIG GET / XINFO-class replies. Callers emit key and value
        alternately either way, so one call site serves both protocols."""
        if self.offset + 16 > RESP_LIMIT:
            if not self._fixed_cold(): return
        if self.proto == 3:
            self.buffer[unsafe_offset=self.offset] = 37 # '%'
            self.offset += 1
            self.offset = format_int_to_buf(self.buffer, self.offset, Int64(pairs))
        else:
            self.buffer[unsafe_offset=self.offset] = 42 # '*'
            self.offset += 1
            self.offset = format_int_to_buf(self.buffer, self.offset, Int64(pairs * 2))
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_set_header(mut self, count: Int):
        """RESP3 set `~<count>` (SMEMBERS, SINTER, SUNION, SDIFF, SPOP with a
        count). RESP2 has no set type and sends the same members as an array."""
        if self.offset + 16 > RESP_LIMIT:
            if not self._fixed_cold(): return
        self.buffer[unsafe_offset=self.offset] = 126 if self.proto == 3 else 42   # '~' / '*'
        self.offset += 1
        self.offset = format_int_to_buf(self.buffer, self.offset, Int64(count))
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_scored_header(mut self, members: Int, with_scores: Bool):
        """Header of a reply that lists `members` sorted-set members, with their
        scores when `with_scores` (WITHSCORES, ZPOPMIN/ZPOPMAX with a count,
        ZRANDMEMBER). RESP2 sends one flat array, member then score; RESP3 an
        array of [member, score] pairs, as Redis does. Pair each call with
        append_scored_member."""
        if with_scores and self.proto != 3:
            self.append_array_header(members * 2)
        else:
            self.append_array_header(members)

    @always_inline
    def append_scored_member(mut self, member: GenericValue, score: Float64, with_scores: Bool):
        """One member of an append_scored_header reply: the member, then its
        score as a bulk string (RESP2) or inside a [member, double] pair (RESP3)."""
        if not with_scores:
            self.append_bulk_value_response(member)
            return
        if self.proto == 3:
            self.append_array_header(2)
            self.append_bulk_value_response(member)
            self.append_score_response(score)
        else:
            self.append_bulk_value_response(member)
            self.append_bulk_score_response(score)

    @always_inline
    def append_push_header(mut self, count: Int):
        """gh #172: RESP3 out-of-band push `><count>\r\n` (pub/sub delivery,
        subscribe confirmations). RESP2 has no push type — the same payload
        goes out as a plain array, which is precisely the RESP2 pub/sub wire
        format, so no RESP2 behaviour changes."""
        if self.offset + 16 > RESP_LIMIT:
            if not self._fixed_cold(): return
        if self.proto == 3:
            self.buffer[unsafe_offset=self.offset] = 62 # '>'
        else:
            self.buffer[unsafe_offset=self.offset] = 42 # '*'
        self.offset += 1
        self.offset = format_int_to_buf(self.buffer, self.offset, Int64(count))
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_bool_response(mut self, val: Bool):
        """gh #172: RESP3 `#t\r\n` / `#f\r\n`; RESP2 `:1\r\n` / `:0\r\n`."""
        if self.offset + 8 > RESP_LIMIT:
            if not self._fixed_cold(): return
        if self.proto == 3:
            self.buffer[self.offset] = 35 # '#'
            self.buffer[self.offset + 1] = 116 if val else 102  # 't' / 'f'
        else:
            self.buffer[self.offset] = 58 # ':'
            self.buffer[self.offset + 1] = 49 if val else 48    # '1' / '0'
        self.buffer[self.offset + 2] = 13 # '\r'
        self.buffer[self.offset + 3] = 10 # '\n'
        self.offset += 4

    @always_inline
    def append_double_response[origin: Origin](mut self, text: Pointer[UInt8, origin], length: Int):
        """gh #172: RESP3 `,<value>\r\n` for score-shaped replies (ZSCORE,
        ZINCRBY, INCRBYFLOAT); RESP2 sends the same digits as a bulk string.

        Takes the already-formatted text rather than a Float64 so the two
        protocols emit byte-identical numerals — a separate float formatter
        here would let RESP2 and RESP3 disagree on rounding for the same key.
        Callers that would emit RESP3 `inf`/`-inf`/`nan` must pass those
        spellings; Redis uses them verbatim in the double type."""
        if self.proto != 3:
            self.append_bulk_string_response(text, length)
            return
        if self.offset + length + 16 > RESP_LIMIT:
            self._framed_cold(44, "", Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(text)), length, -1)
            return
        self.buffer[unsafe_offset=self.offset] = 44 # ','
        self.offset += 1
        unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset), src=text, count=length)
        self.offset += length
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_score_response(mut self, score: Float64):
        """gh #177: ZSCORE/ZINCRBY score reply — RESP3 double `,`, RESP2 bulk
        string. Integer-valued scores emit bare digits with no `.0` in both
        protocols, matching Redis (`,4\\r\\n`, `$1\\r\\n4\\r\\n`).

        Scoped to the two commands real Redis answers with the double type:
        INCRBYFLOAT / HINCRBYFLOAT / GEODIST stay bulk strings even under
        RESP3 in Redis 8 (probed 2026-08-03) — do not route them here."""
        if score_prints_as_int(score):   # #18: never Int64() an inf
            var si = Int64(score)
            if self.proto != 3:
                self.append_bulk_int_response(si)
                return
            # ',' + up to 20 digits + '\r\n' = 23 bytes max for Int64.
            if self.offset + 24 > RESP_LIMIT:
                if not self._fixed_cold(): return
            self.buffer[unsafe_offset=self.offset] = 44 # ','
            self.offset += 1
            self.offset = format_int_to_buf(self.buffer, self.offset, si)
            self.buffer[unsafe_offset=self.offset] = 13 # '\r'
            self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
            self.offset += 2
        else:
            var ss = format_score(score)
            self.append_double_response(ss.unsafe_ptr(), ss.byte_length())

    @always_inline
    def append_bulk_score_response(mut self, score: Float64):
        """A sorted-set score inside an array reply (WITHSCORES, the ZPOP and
        ZMPOP families, ZMSCORE, ZSCAN): a bulk string, digits as Redis prints
        them. Every such site used to carry its own `Int64(score)` round trip,
        which read ±inf back as INT64_MIN on x86 (#18)."""
        if score_prints_as_int(score):
            self.append_bulk_int_response(Int64(score))
        else:
            var ss = format_score(score)
            self.append_bulk_string_response(ss.unsafe_ptr(), ss.byte_length())

    @always_inline
    def append_empty_array_response(mut self):
        var off = self.offset
        if off + 4 > RESP_LIMIT:
            if not self._fixed_cold(): return
            off = self.offset
        # '*0\r\n' packed LE: 0x000000000A0D302A (writes 8 bytes; extra 4 safely overwritten)
        (self.buffer.unsafe_offset(off)).unsafe_bitcast[UInt64]()[] = UInt64(0x000000000A0D302A)
        self.offset = off + 4

    @always_inline
    def append_int_response(mut self, val: Int64):
        # ':' + up to 20 digits + '\r\n' = 23 bytes max for Int64.
        if self.offset + 24 > RESP_LIMIT:
            if not self._fixed_cold(): return
        self.buffer[unsafe_offset=self.offset] = 58 # ':'
        self.offset += 1
        self.offset = format_int_to_buf(self.buffer, self.offset, val)
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_bulk_int_response(mut self, val: Int64):
        # '$' + 2-digit len + '\r\n' + up to 20 digits + '\r\n' = 27 bytes max.
        if self.offset + 28 > RESP_LIMIT:
            if not self._fixed_cold(): return
        self.buffer[unsafe_offset=self.offset] = 36 # '$'
        self.offset += 1
        var length = int_string_len(val)
        self.offset = format_int_to_buf(self.buffer, self.offset, Int64(length))
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2
        self.offset = format_int_to_buf(self.buffer, self.offset, val)
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_bulk_string_response_header(mut self, length: Int):
        # '$' + up to 10 digits + '\r\n' = 13 bytes max for any 32-bit length.
        if self.offset + 16 > RESP_LIMIT:
            if not self._fixed_cold(): return
        self.buffer[unsafe_offset=self.offset] = 36 # '$'
        self.offset += 1
        self.offset = format_int_to_buf(self.buffer, self.offset, Int64(length))
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_bulk_string_response[origin: Origin](mut self, data: Pointer[UInt8, origin], length: Int):
        # gh #82: bound the memcpy. Caller may pass arbitrary `length` (V.FETCH,
        # KV.PREFIX.*, stream payloads, etc.); without this guard a large value
        # written to a near-full buffer corrupts adjacent heap. #49: past the
        # bound the reply goes on through the cold path, however long.
        if self.offset + length + 16 > RESP_LIMIT:
            self._framed_cold(36, "", Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(data)), length, length)
            return
        self.buffer[unsafe_offset=self.offset] = 36 # '$'
        self.offset += 1
        self.offset = format_int_to_buf(self.buffer, self.offset, Int64(length))
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2
        unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset), src=data, count=length)
        self.offset += length
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_bulk_value_response(mut self, val: GenericValue):
        if val.is_none():
            self.append_null_response()
            return

        if val.type.value == ValueType.STRING_SSO:
            # gh #82: SSO bitcast writes 24 bytes past the header (3× UInt64 stores
            # at +0/+7/+15). Worst-case footprint = $ + 2-digit len + \r\n + 24 + \r\n = 31 bytes.
            if self.offset + 32 > RESP_LIMIT:
                if not self._fixed_cold(): return
            var length = Int(val._data0 & 0xFF)
            self.buffer[unsafe_offset=self.offset] = 36 # '$'
            self.offset += 1
            # SSO length is always 0-23: inline 1-or-2-digit write, no format_int_to_buf call
            if length < 10:
                self.buffer[unsafe_offset=self.offset] = UInt8(48 + length)
                self.offset += 1
            else: # 10-23
                self.buffer[unsafe_offset=self.offset] = UInt8(48 + length // 10)
                self.buffer[unsafe_offset=self.offset + 1] = UInt8(48 + length % 10)
                self.offset += 2
            self.buffer[unsafe_offset=self.offset] = 13 # '\r'
            self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
            self.offset += 2
            # Fast path: 3 word-stores (may write up to 7 extra bytes, safe in 4MB buffer)
            (self.buffer.unsafe_offset(self.offset)).unsafe_bitcast[UInt64]()[] = val._data0 >> 8
            ((self.buffer.unsafe_offset(self.offset)).unsafe_offset(7)).unsafe_bitcast[UInt64]()[] = val._data1
            ((self.buffer.unsafe_offset(self.offset)).unsafe_offset(15)).unsafe_bitcast[UInt64]()[] = val._data2
            self.offset += length
            self.buffer[unsafe_offset=self.offset] = 13 # '\r'
            self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
            self.offset += 2
        elif val.type.value == ValueType.STRING:
            var length = val.string_len()
            # gh #82: heap STRING length is unbounded; the memcpy below would walk
            # off the 4 MB buffer if offset is already deep into the safe-zone tail.
            if self.offset + length + 16 > RESP_LIMIT:
                self._framed_cold(36, "", val.as_string(), length, length)
                return
            self.buffer[unsafe_offset=self.offset] = 36 # '$'
            self.offset += 1
            self.offset = format_int_to_buf(self.buffer, self.offset, Int64(length))
            self.buffer[unsafe_offset=self.offset] = 13 # '\r'
            self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
            self.offset += 2
            unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset), src=val.as_string(), count=length)
            self.offset += length
            self.buffer[unsafe_offset=self.offset] = 13 # '\r'
            self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
            self.offset += 2
        elif val.type.value == ValueType.INT:
            self.append_bulk_int_response(val.as_int())
        elif val.type.value == ValueType.BITMAP:
            # Redis treats BITMAP as STRING — return raw bytes via GET
            var bm_ptr = val.as_bitmap()
            var bm_len = val.bitmap_len()
            # gh #82: same unbounded-memcpy concern as the STRING branch.
            if self.offset + bm_len + 16 > RESP_LIMIT:
                self._framed_cold(36, "", bm_ptr, bm_len, bm_len)
                return
            self.buffer[unsafe_offset=self.offset] = 36 # '$'
            self.offset += 1
            self.offset = format_int_to_buf(self.buffer, self.offset, Int64(bm_len))
            self.buffer[unsafe_offset=self.offset] = 13 # '\r'
            self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
            self.offset += 2
            unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset), src=bm_ptr, count=bm_len)
            self.offset += bm_len
            self.buffer[unsafe_offset=self.offset] = 13 # '\r'
            self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
            self.offset += 2
        else:
            self.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")

    @always_inline
    def append_bulk_bytes_writev[origin: Origin](
        mut self, fd: Int32, data: Pointer[UInt8, origin], length: Int,
    ):
        """V.FETCH RANGE etc.: a raw byte buffer that may exceed RESP_BUF_SIZE,
        as a bulk string. Since #49 any bulk string may, so this is
        append_bulk_string_response: what the buffer cannot hold goes to the
        connection's output, behind what it is owed, without blocking the
        worker. (A blocking send loop used to write it straight to the socket:
        it stalled every connection on the worker while a slow client read,
        and it jumped ahead of replies already queued for this one.)
        `fd` is the connection being served, which the writer already knows."""
        self.append_bulk_string_response(data, length)

    @always_inline
    def append_large_value_response_writev(mut self, fd: Int32, val: GenericValue):
        """A6: GET of a value of any size (LMCache blobs are 1-16 MB). Since
        #49 this is append_bulk_value_response; see append_bulk_bytes_writev
        for what it replaced."""
        self.append_bulk_value_response(val)

    @always_inline
    def append_error_response(mut self, msg: String):
        var b = msg.as_bytes()
        # Error messages are caller-controlled but in practice bounded — the
        # safe-zone invariant maintained by the variable-length appenders
        # leaves >194 KB of headroom, which is well past any real error string.
        if self.offset + len(b) + 4 > RESP_LIMIT:
            if not self._fixed_cold(): return
            if len(b) + 4 > RESP_LIMIT: return
        self.buffer[unsafe_offset=self.offset] = 45 # '-'
        self.offset += 1
        unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset), src=b.unsafe_ptr(), count=len(b))
        self.offset += len(b)
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_stream_id_response(mut self):
        var off = self.offset
        if off + 22 > RESP_LIMIT:
            if not self._fixed_cold(): return
            off = self.offset
        var ptr = self.buffer + off
        # '$15\r\n1688145214000-0\r\n' packed as three UInt64 word-stores (writes 24 bytes; extra 2 safely overwritten)
        # bytes  0-7:  '$15\r\n168'  LE: 0x3836310A0D353124
        # bytes  8-15: '81452140'   LE: 0x3034313235343138
        # bytes 16-23: '00-0\r\n..' LE: 0x00000A0D302D3030
        ptr.unsafe_bitcast[UInt64]()[] = UInt64(0x3836310A0D353124)
        (ptr + 8).unsafe_bitcast[UInt64]()[] = UInt64(0x3034313235343138)
        (ptr + 16).unsafe_bitcast[UInt64]()[] = UInt64(0x00000A0D302D3030)
        self.offset = off + 22

    @always_inline
    def append_mylib_response(mut self):
        var off = self.offset
        if off + 11 > RESP_LIMIT:
            if not self._fixed_cold(): return
            off = self.offset
        var ptr = self.buffer + off
        # '$5\r\nmylib\r\n' packed as two UInt64 word-stores (writes 16 bytes; extra 5 safely overwritten)
        # bytes 0-7:  [36,53,13,10,109,121,108,105] = '$5\r\nmyli'  LE: 0x696C796D0A0D3524
        # bytes 8-15: [98,13,10, 0,  0,  0,  0,  0] = 'b\r\n.....' LE: 0x000000000A0D62
        ptr.unsafe_bitcast[UInt64]()[] = UInt64(0x696C796D0A0D3524)
        (ptr + 8).unsafe_bitcast[UInt64]()[] = UInt64(0x000000000A0D62)
        self.offset = off + 11
