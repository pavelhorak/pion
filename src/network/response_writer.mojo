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


def _write_overflow_error_bytes(dst: Pointer[UInt8, MutUntrackedOrigin]):
    """gh #82: write `-ERR response exceeds buffer\r\n` (30 bytes) at dst.
    Free function (not a method) — see `_check_overflow` note for why method
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
    # gh #82: replaces the previous silent-drop overflow guards.
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

    @always_inline
    def bind_ring(mut self, ring_ptr: Pointer[IOUring, MutUntrackedOrigin]):
        """Wire this ResponseWriter to an IOUring instance for Linux io_uring sends."""
        self.ring = ring_ptr
        self.use_uring = True

    @always_inline
    def _check_overflow(mut self, length: Int) -> Bool:
        """gh #82: returns True if the caller must skip its write (overflow).
        On first overflow detection the caller is responsible for emitting the
        `-ERR response exceeds buffer\r\n` frame via `_emit_overflow_error()`.

        Hot-path layout: the offset check runs first so a non-overflowing
        append only loads `self.offset` (one 8-byte field already in cache) —
        matches the original silent-guard's cycle count. The flag load and
        emit call are reached only on the cold overflow branch.

        Splitting check from emit also avoids a Mojo @always_inline def
        mis-lowering observed under recursive dispatcher entry (MULTI/EXEC
        replay) when the emit body was embedded directly here."""
        if self.offset + length > RESP_BUF_SIZE - 194304:
            if self.overflow_emitted: return True
            self._emit_overflow_error()
            return True
        return False

    def _emit_overflow_error(mut self):
        """Cold-path emit of `-ERR response exceeds buffer\r\n` (30 bytes).
        Deliberately NOT `@always_inline` — see `_check_overflow` note."""
        if self.overflow_emitted: return
        self.overflow_emitted = True
        _write_overflow_error_bytes(self.buffer.unsafe_offset(self.offset))
        self.offset += 30

    @always_inline
    def append_to_response[origin: Origin](mut self, src: Pointer[UInt8, origin], length: Int):
        # gh #82: variable-length write — must emit `-ERR …` on overflow so
        # pipelined clients stay frame-synced.
        if self._check_overflow(length): return
        unsafe_memcpy(dest=self.buffer.unsafe_offset(self.offset), src=src, count=length)
        self.offset += length

    @always_inline
    def flush_response(mut self, fd: Int32, server: TCPServer, kq: Int32):
        self.flush_count += 1
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

        What the socket cannot take now waits in that connection's pending
        buffer, behind what is already there, and goes out on its write event.
        A frame that fits neither is never cut: the connection is shut down
        instead, as Redis disconnects a client past its output-buffer limit,
        because a subscriber that received half a frame is out of sync for
        good. (Delivery used to send() once and drop the rest at EAGAIN.)
        Not on the XDP lane (kq == -1), which sends nothing over TCP."""
        if length <= 0 or (kq == -1 and not self.use_uring):
            return
        var ci = Int(fd)
        var limit = RESP_BUF_SIZE - 194304
        var p = data
        var left = length
        if not self.use_uring and self.pending_offsets[unsafe_offset=ci] == 0:
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
        var cur = self.pending_offsets[unsafe_offset=ci]
        if self.use_uring and self.ring[].fd_closing[unsafe_offset=ci] != 0:
            return
        if cur + left > limit:
            _ = external_call["shutdown", Int32](fd, Int32(2))   # SHUT_RDWR: the engine sees EOF
            return
        if self.pending_buffers[unsafe_offset=ci] == null_ptr[UInt8, MutUntrackedOrigin]():
            self.pending_buffers[unsafe_offset=ci] = alloc[UInt8](RESP_BUF_SIZE)
        unsafe_memcpy(dest=self.pending_buffers[unsafe_offset=ci].unsafe_offset(cur), src=p, count=left)
        self.pending_offsets[unsafe_offset=ci] = cur + left
        if self.use_uring:
            if self.uring_inflight[unsafe_offset=ci] == 0:
                self.uring_inflight[unsafe_offset=ci] = self.pending_offsets[unsafe_offset=ci]
                self.ring[].submit_send(fd, self.pending_buffers[unsafe_offset=ci], self.uring_inflight[unsafe_offset=ci])
        else:
            server.kevent_add_write(kq, fd)

    @always_inline
    def _flush_uring(mut self, fd: Int32):
        """io_uring send path. Copies response buffer into pending_buffers[fd] and submits
        a SEND SQE if none is currently in flight. Accumulates if a send is in flight;
        the SEND completion handler drains the remainder."""
        if self.offset == 0:
            self.overflow_emitted = False
            return
        var ci = Int(fd)
        if self.ring[].fd_closing[unsafe_offset=ci] != 0:
            # The engine is closing this connection and waits for its last
            # SEND to complete before freeing the buffer a new one would read.
            self.offset = 0
            self.overflow_emitted = False
            return
        if self.pending_buffers[unsafe_offset=ci] == null_ptr[UInt8, MutUntrackedOrigin]():
            self.pending_buffers[unsafe_offset=ci] = alloc[UInt8](RESP_BUF_SIZE)
        var cur = self.pending_offsets[unsafe_offset=ci]
        # gh #82: if appending self.buffer would overflow pending_buffers, try to
        # squeeze the `-ERR response exceeds buffer\r\n` frame in. If even that
        # won't fit, the client has already received frame-synced bytes and the
        # next thing it sees will be a closed connection — losing the trailing
        # responses is unavoidable, but the stream stays well-formed up to the
        # last byte we sent.
        if cur + self.offset > RESP_BUF_SIZE - 194304:
            if cur + 30 <= RESP_BUF_SIZE:
                _write_overflow_error_bytes(self.pending_buffers[unsafe_offset=ci].unsafe_offset(cur))
                self.pending_offsets[unsafe_offset=ci] = cur + 30
                if self.uring_inflight[unsafe_offset=ci] == 0:
                    self.uring_inflight[unsafe_offset=ci] = self.pending_offsets[unsafe_offset=ci]
                    self.ring[].submit_send(fd, self.pending_buffers[unsafe_offset=ci], self.uring_inflight[unsafe_offset=ci])
            self.offset = 0
            self.overflow_emitted = False
            return
        unsafe_memcpy(dest=self.pending_buffers[unsafe_offset=ci].unsafe_offset(cur), src=self.buffer, count=self.offset)
        self.pending_offsets[unsafe_offset=ci] = cur + self.offset
        self.offset = 0
        self.overflow_emitted = False
        # Submit SEND only if no send is currently in flight for this fd.
        # If uring_inflight > 0, the data was appended above; the SEND completion
        # handler will detect pending_offsets > uring_inflight and submit the next SEND.
        if self.uring_inflight[unsafe_offset=ci] == 0:
            self.uring_inflight[unsafe_offset=ci] = self.pending_offsets[unsafe_offset=ci]
            self.ring[].submit_send(fd, self.pending_buffers[unsafe_offset=ci], self.uring_inflight[unsafe_offset=ci])

    @always_inline
    def _flush_kqueue(mut self, fd: Int32, server: TCPServer, kq: Int32):
        var fd_idx = Int(fd)

        # 1. If we have a pending buffer, append new response to it
        if self.pending_offsets[unsafe_offset=fd_idx] > 0:
            if self.offset > 0:
                var current_pending = self.pending_offsets[unsafe_offset=fd_idx]
                if self.pending_buffers[unsafe_offset=fd_idx] == null_ptr[UInt8, MutUntrackedOrigin]():
                    self.pending_buffers[unsafe_offset=fd_idx] = alloc[UInt8](RESP_BUF_SIZE)
                # gh #82: would the merge blow past the 4 MB pending buffer? Try
                # to drop a `-ERR response exceeds buffer\r\n` frame in instead;
                # if even that won't fit, drop self.buffer silently (the client
                # already saw frame-synced bytes; the next thing it sees is FIN).
                if current_pending + self.offset > RESP_BUF_SIZE - 194304:
                    if current_pending + 30 <= RESP_BUF_SIZE:
                        _write_overflow_error_bytes(self.pending_buffers[unsafe_offset=fd_idx].unsafe_offset(current_pending))
                        self.pending_offsets[unsafe_offset=fd_idx] = current_pending + 30
                    self.offset = 0
                else:
                    unsafe_memcpy(dest=self.pending_buffers[unsafe_offset=fd_idx].unsafe_offset(current_pending), src=self.buffer, count=self.offset)
                    self.pending_offsets[unsafe_offset=fd_idx] = current_pending + self.offset
                    self.offset = 0

            # Flush pending buffer
            var n = server.send(fd, self.pending_buffers[unsafe_offset=fd_idx], self.pending_offsets[unsafe_offset=fd_idx])
            if n >= self.pending_offsets[unsafe_offset=fd_idx]:
                self.pending_offsets[unsafe_offset=fd_idx] = 0
                server.kevent_del_write(kq, fd)
            elif n > 0:
                var remaining = self.pending_offsets[unsafe_offset=fd_idx] - n
                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                    self.pending_buffers[unsafe_offset=fd_idx].unsafe_bitcast[NoneType](),
                    (self.pending_buffers[unsafe_offset=fd_idx].unsafe_offset(n)).unsafe_bitcast[NoneType](),
                    remaining,
                )
                self.pending_offsets[unsafe_offset=fd_idx] = remaining
                server.kevent_add_write(kq, fd)  # defensive: re-arm EVFILT_WRITE if still bytes remain
            else:
                # n <= 0: check errno. EAGAIN → re-arm write. EPIPE → fd dead, drop pending.
                var errno_val2: Int32
                comptime if CompilationTarget.is_linux():
                    errno_val2 = external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
                else:
                    errno_val2 = external_call["__error", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
                var eagain_code2: Int32
                comptime if CompilationTarget.is_linux():
                    eagain_code2 = 11
                else:
                    eagain_code2 = 35
                var is_eagain2 = (errno_val2 == eagain_code2)
                if is_eagain2:
                    server.kevent_add_write(kq, fd)
                else:
                    self.pending_offsets[unsafe_offset=fd_idx] = 0  # drop; event loop will close fd
            return

        # 2. No pending buffer
        if self.offset == 0: return
        var n = server.send(fd, self.buffer, self.offset)
        if n >= self.offset:
            self.offset = 0
        elif n > 0:
            var remaining = self.offset - n
            if self.pending_buffers[unsafe_offset=fd_idx] == null_ptr[UInt8, MutUntrackedOrigin]():
                self.pending_buffers[unsafe_offset=fd_idx] = alloc[UInt8](RESP_BUF_SIZE)
            unsafe_memcpy(dest=self.pending_buffers[unsafe_offset=fd_idx], src=self.buffer.unsafe_offset(n), count=remaining)
            self.pending_offsets[unsafe_offset=fd_idx] = remaining
            self.offset = 0
            server.kevent_add_write(kq, fd)
        elif n < 0:
            # Check errno: EAGAIN/EWOULDBLOCK → buffer and retry via EPOLLOUT/EVFILT_WRITE.
            # EPIPE/ECONNRESET → fd is dead, don't buffer (epoll_wait will deliver EPOLLERR).
            var errno_val: Int32
            comptime if CompilationTarget.is_linux():
                errno_val = external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
            else:
                errno_val = external_call["__error", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
            var eagain_code: Int32
            comptime if CompilationTarget.is_linux():
                eagain_code = 11  # EAGAIN/EWOULDBLOCK
            else:
                eagain_code = 35  # EAGAIN (macOS)
            var is_eagain = (errno_val == eagain_code)
            if is_eagain:
                if self.pending_buffers[unsafe_offset=fd_idx] == null_ptr[UInt8, MutUntrackedOrigin]():
                    self.pending_buffers[unsafe_offset=fd_idx] = alloc[UInt8](RESP_BUF_SIZE)
                unsafe_memcpy(dest=self.pending_buffers[unsafe_offset=fd_idx], src=self.buffer, count=self.offset)
                self.pending_offsets[unsafe_offset=fd_idx] = self.offset
                self.offset = 0
                server.kevent_add_write(kq, fd)
            else:
                # EPIPE/ECONNRESET: fd is dead. Drop the response; event loop will close fd.
                self.offset = 0

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
        if off + 5 > RESP_BUF_SIZE - 194304: return
        # '+OK\r\n' packed LE: 0x0000000A0D4B4F2B (writes 8 bytes; extra 3 safely overwritten)
        (self.buffer.unsafe_offset(off)).unsafe_bitcast[UInt64]()[] = UInt64(0x0000000A0D4B4F2B)
        self.offset = off + 5

    @always_inline
    def append_pong_response(mut self):
        var off = self.offset
        if off + 7 > RESP_BUF_SIZE - 194304: return
        # '+PONG\r\n' packed LE: 0x000A0D474E4F502B (writes 8 bytes; extra 1 safely overwritten)
        (self.buffer.unsafe_offset(off)).unsafe_bitcast[UInt64]()[] = UInt64(0x000A0D474E4F502B)
        self.offset = off + 7

    @always_inline
    def append_pong_bulk(mut self, count: Int):
        # Write `count` PONG responses as a single batched memcpy
        # +PONG\r\n = 7 bytes
        var total = count * 7
        if self.offset + total > RESP_BUF_SIZE - 194304: return
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

    @always_inline
    def append_null_response(mut self):
        var off = self.offset
        if off + 5 > RESP_BUF_SIZE - 194304: return
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
        if self._check_overflow(length + 32): return
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
        if self.offset + 8 > RESP_BUF_SIZE - 194304: return
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
        if self.offset + 16 > RESP_BUF_SIZE - 194304: return
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
        if self.offset + 16 > RESP_BUF_SIZE - 194304: return
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
        if self.offset + 16 > RESP_BUF_SIZE - 194304: return
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
        if self.offset + 16 > RESP_BUF_SIZE - 194304: return
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
        if self.offset + 8 > RESP_BUF_SIZE - 194304: return
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
        if self._check_overflow(length + 16): return
        if self.proto != 3:
            self.append_bulk_string_response(text, length)
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
            if self.offset + 24 > RESP_BUF_SIZE - 194304: return
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
        if off + 4 > RESP_BUF_SIZE - 194304: return
        # '*0\r\n' packed LE: 0x000000000A0D302A (writes 8 bytes; extra 4 safely overwritten)
        (self.buffer.unsafe_offset(off)).unsafe_bitcast[UInt64]()[] = UInt64(0x000000000A0D302A)
        self.offset = off + 4

    @always_inline
    def append_int_response(mut self, val: Int64):
        # ':' + up to 20 digits + '\r\n' = 23 bytes max for Int64.
        if self.offset + 24 > RESP_BUF_SIZE - 194304: return
        self.buffer[unsafe_offset=self.offset] = 58 # ':'
        self.offset += 1
        self.offset = format_int_to_buf(self.buffer, self.offset, val)
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2

    @always_inline
    def append_bulk_int_response(mut self, val: Int64):
        # '$' + 2-digit len + '\r\n' + up to 20 digits + '\r\n' = 27 bytes max.
        if self.offset + 28 > RESP_BUF_SIZE - 194304: return
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
        if self.offset + 16 > RESP_BUF_SIZE - 194304: return
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
        # written to a near-full buffer corrupts adjacent heap.
        if self._check_overflow(length + 16): return
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
            if self._check_overflow(32): return
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
            if self._check_overflow(length + 16): return
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
            if self._check_overflow(bm_len + 16): return
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
        """V.FETCH RANGE etc.: bulk-string a raw byte buffer that may exceed
        RESP_BUF_SIZE. Routes through `_send_all_blocking` with header from
        self.buffer + payload direct from caller's pointer. For length ≤ 3 MB
        the normal append_bulk_string_response is faster (no per-call send
        loop) so we delegate; only the over-buffer case takes the writev path.

        Why this exists: append_bulk_string_response does a memcpy into
        self.buffer with no bounds check. V.FETCH RANGE on a ≥2K-token prefix
        returns ≥4 MB which overflows the 4 MB RESP_BUF_SIZE, corrupting the
        adjacent heap and crashing the next slow-path tick. Confirmed via
        bench_w1_2_sweep.py at prompt-repeats=16 (≈2514 tokens × 512 dim ×
        4 bytes = 5 MB) → SIGSEGV in process_slow_path.
        """
        # Fits in buffer (with the same safety margin used elsewhere) → fast path.
        # A capture writer (a script's redis.call) never writes to the fd: a
        # reply too large for it becomes the overflow error.
        if is_null(self.pending_offsets) or self.offset + length + 32 <= RESP_BUF_SIZE - 194304:
            self.append_bulk_string_response(data, length)
            return
        # Build RESP header `$<len>\r\n` in self.buffer.
        self.buffer[unsafe_offset=self.offset] = 36 # '$'
        self.offset += 1
        self.offset = format_int_to_buf(self.buffer, self.offset, Int64(length))
        self.buffer[unsafe_offset=self.offset] = 13 # '\r'
        self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
        self.offset += 2
        var header_len = self.offset

        var crlf = alloc[UInt8](2)
        crlf[unsafe_offset=0] = 13
        crlf[unsafe_offset=1] = 10

        @always_inline
        def _send_all_blocking(sfd: Int32, ptr: Pointer[UInt8, MutUntrackedOrigin], total: Int) -> Int:
            var stalls = 0
            var sent = 0
            while sent < total:
                var n = external_call["pion_write", Int64](sfd, ptr.unsafe_offset(sent), total - sent)
                if n > 0:
                    sent += Int(n)
                elif n < 0:
                    stalls += 1  # gh #192: each retry is a ~150 µs usleep
                    _ = external_call["usleep", Int32](Int32(100))
                else:
                    break
            return stalls

        # Cast caller's data pointer to MutUntrackedOrigin for the C bridge.
        var data_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(data))
        var _st = _send_all_blocking(fd, self.buffer, header_len)
        _st += _send_all_blocking(fd, data_ext, length)
        _st += _send_all_blocking(fd, crlf, 2)
        self.send_stalls += UInt64(_st)

        crlf.unsafe_free()
        self.offset = 0
        self.flush_count += 1   # #47: sent directly, not through flush_response

    @always_inline
    def append_large_value_response_writev(mut self, fd: Int32, val: GenericValue):
        """A6: For values >512B heap STRING, send via writev (header from buffer, value direct
        from heap pointer). Handles partial sends with blocking retry loop.
        Used by GET fast path for LMCache-size blobs (1-16MB)."""
        # Only use writev for values > 3MB that won't fit in the 4MB response buffer.
        # Smaller values go through the normal buffer path (faster, handles pipelining).
        if is_not_null(self.pending_offsets) and val.type.value == ValueType.STRING and val.string_len() > 3 * 1024 * 1024 and val.type.value != ValueType.STRING_SSO:
            # Build RESP header in response buffer: $<len>\r\n
            self.buffer[unsafe_offset=self.offset] = 36 # '$'
            self.offset += 1
            self.offset = format_int_to_buf(self.buffer, self.offset, Int64(val.string_len()))
            self.buffer[unsafe_offset=self.offset] = 13 # '\r'
            self.buffer[unsafe_offset=self.offset + 1] = 10 # '\n'
            self.offset += 2

            var header_len = self.offset

            var crlf = alloc[UInt8](2)
            crlf[unsafe_offset=0] = 13
            crlf[unsafe_offset=1] = 10

            # Send large value via blocking send loop with EAGAIN retry.
            # Non-blocking sockets return EAGAIN after ~128KB; we spin-retry with usleep.
            # This only fires for values >3MB (LMCache path), not on the hot KV/vector path.
            @always_inline
            def _send_all_blocking(sfd: Int32, ptr: Pointer[UInt8, MutUntrackedOrigin], total: Int) -> Int:
                var stalls = 0
                var sent = 0
                while sent < total:
                    var n = external_call["pion_write", Int64](sfd, ptr.unsafe_offset(sent), total - sent)
                    if n > 0:
                        sent += Int(n)
                    elif n < 0:
                        # EAGAIN: socket buffer full, wait and retry
                        stalls += 1  # gh #192
                        _ = external_call["usleep", Int32](Int32(100))  # 100µs
                    else:
                        break  # n==0: connection closed
                return stalls
            var _st = _send_all_blocking(fd, self.buffer, header_len)
            _st += _send_all_blocking(fd, val.as_string(), val.string_len())
            _st += _send_all_blocking(fd, crlf, 2)
            self.send_stalls += UInt64(_st)

            crlf.unsafe_free()
            self.offset = 0
            self.flush_count += 1   # #47: sent directly, not through flush_response
        else:
            self.append_bulk_value_response(val)

    @always_inline
    def append_error_response(mut self, msg: String):
        var b = msg.as_bytes()
        # Error messages are caller-controlled but in practice bounded — the
        # safe-zone invariant maintained by the variable-length appenders
        # leaves >194 KB of headroom, which is well past any real error string.
        if self.offset + len(b) + 4 > RESP_BUF_SIZE - 194304: return
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
        if off + 22 > RESP_BUF_SIZE - 194304: return
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
        if off + 11 > RESP_BUF_SIZE - 194304: return
        var ptr = self.buffer + off
        # '$5\r\nmylib\r\n' packed as two UInt64 word-stores (writes 16 bytes; extra 5 safely overwritten)
        # bytes 0-7:  [36,53,13,10,109,121,108,105] = '$5\r\nmyli'  LE: 0x696C796D0A0D3524
        # bytes 8-15: [98,13,10, 0,  0,  0,  0,  0] = 'b\r\n.....' LE: 0x000000000A0D62
        ptr.unsafe_bitcast[UInt64]()[] = UInt64(0x696C796D0A0D3524)
        (ptr + 8).unsafe_bitcast[UInt64]()[] = UInt64(0x000000000A0D62)
        self.offset = off + 11
