"""Pub/Sub commands: PUBSUB, PUBLISH, SUBSCRIBE, UNSUBSCRIBE, PSUBSCRIBE, PUNSUBSCRIBE, SSUBSCRIBE, SUNSUBSCRIBE, SPUBLISH.

Cross-worker pub/sub via shared broadcast ring. Each PUBLISH writes to the ring;
each worker's event loop drains it and delivers to local subscribers.
"""
from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.collections import Array
from std.memory import unsafe_memcpy, unsafe_memset
from std.atomic import Atomic, Ordering
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.common.utils import format_int_to_buf


# ── PubSub Registry ──
# Per-worker channel registry: maps channel names to subscriber fd sets.
# Max 256 channels, 64 subscribers per channel. Linear scan is fine for typical pub/sub.

comptime MAX_CHANNELS = 256
comptime MAX_SUBS_PER_CHANNEL = 64
comptime MAX_PATTERNS = 256
comptime MAX_FDS = 65536

# ── Cross-Worker Broadcast Ring ──
# Single-producer-multiple-consumer ring buffer. PUBLISH writes to head;
# each worker maintains its own tail cursor. Messages are channel+message
# pairs stored in a fixed-size ring. Fire-and-forget: if a worker is too
# slow, it skips old messages (acceptable for pub/sub semantics).

comptime PUBSUB_RING_SIZE = 1024     # must be power of 2
comptime PUBSUB_MSG_MAX = 2048       # max channel+message bytes per slot
comptime PUBSUB_RING_MASK = PUBSUB_RING_SIZE - 1


struct PubSubBroadcast(Movable):
    """Shared broadcast ring for cross-worker PUBLISH.
    Allocated once before parallelize(), shared across all workers.
    Lock-free: single atomic head, per-worker tail cursors."""
    var head: Pointer[UInt64, MutUntrackedOrigin]  # global write cursor (atomic via static Atomic API)
    var ch_lens: Pointer[Int32, MutUntrackedOrigin]    # channel name length per slot
    var msg_lens: Pointer[Int32, MutUntrackedOrigin]   # message length per slot
    var data: Pointer[UInt8, MutUntrackedOrigin]       # channel+message data per slot
    var origin_worker: Pointer[Int32, MutUntrackedOrigin]  # which worker published (skip self)
    var ready: Bool

    def __init__(out self):
        self.head = alloc[UInt64](1)
        self.head[unsafe_offset=0] = UInt64(0)
        self.ch_lens = alloc[Int32](PUBSUB_RING_SIZE)
        self.msg_lens = alloc[Int32](PUBSUB_RING_SIZE)
        self.data = alloc[UInt8](PUBSUB_RING_SIZE * PUBSUB_MSG_MAX)
        self.origin_worker = alloc[Int32](PUBSUB_RING_SIZE)
        unsafe_memset(self.ch_lens.unsafe_bitcast[UInt8](), 0, PUBSUB_RING_SIZE * 4)
        unsafe_memset(self.msg_lens.unsafe_bitcast[UInt8](), 0, PUBSUB_RING_SIZE * 4)
        unsafe_memset(self.origin_worker.unsafe_bitcast[UInt8](), 0, PUBSUB_RING_SIZE * 4)
        self.ready = True

    def publish(mut self, worker_id: Int, ch_ptr: Pointer[UInt8, MutUntrackedOrigin], ch_len: Int,
               msg_ptr: Pointer[UInt8, MutUntrackedOrigin], msg_len: Int):
        """Post a message to the broadcast ring. Called by PUBLISH handler."""
        if ch_len + msg_len > PUBSUB_MSG_MAX: return  # too large, skip
        var slot = Int(Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELEASE](self.head, UInt64(1)) & UInt64(PUBSUB_RING_MASK))
        var base = slot * PUBSUB_MSG_MAX
        unsafe_memcpy(dest=self.data.unsafe_offset(base), src=ch_ptr, count=ch_len)
        unsafe_memcpy(dest=self.data.unsafe_offset(base).unsafe_offset(ch_len), src=msg_ptr, count=msg_len)
        self.ch_lens[unsafe_offset=slot] = Int32(ch_len)
        self.msg_lens[unsafe_offset=slot] = Int32(msg_len)
        self.origin_worker[unsafe_offset=slot] = Int32(worker_id)

    def drain(mut self, worker_id: Int, mut tail: UInt64, registry: PubSubRegistry, server: TCPServer,
              resp_proto: Pointer[UInt8, MutUntrackedOrigin] = null_ptr[UInt8, MutUntrackedOrigin]()):
        """Drain pending messages from the broadcast ring. Called per event loop tick.
        Delivers to local subscribers only (skip messages from own worker)."""
        var current_head = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](self.head, UInt64(0))
        # Skip if too far behind (ring wrapped — accept message loss)
        if current_head > tail + UInt64(PUBSUB_RING_SIZE):
            tail = current_head - UInt64(PUBSUB_RING_SIZE // 2)
        while tail < current_head:
            var slot = Int(tail & UInt64(PUBSUB_RING_MASK))
            tail += 1
            # Skip own messages (already delivered locally)
            if Int(self.origin_worker[unsafe_offset=slot]) == worker_id: continue
            var ch_len = Int(self.ch_lens[unsafe_offset=slot])
            var msg_len = Int(self.msg_lens[unsafe_offset=slot])
            if ch_len == 0: continue
            var base = slot * PUBSUB_MSG_MAX
            var ch_ptr = self.data.unsafe_offset(base)
            var msg_ptr = self.data.unsafe_offset(base).unsafe_offset(ch_len)
            # Deliver to local exact subscribers
            var ch_idx = registry.find_channel(ch_ptr, ch_len)
            if ch_idx >= 0 and registry.channels[unsafe_offset=ch_idx].fd_count > 0:
                var push_buf = alloc[UInt8](4096)
                var hdr = "*3\r\n$7\r\nmessage\r\n"
                unsafe_memcpy(dest=push_buf, src=hdr.unsafe_ptr(), count=hdr.byte_length()); var off = hdr.byte_length()
                push_buf[unsafe_offset=off] = 36; off += 1
                off += format_int_to_buf(push_buf.unsafe_offset(off), 0, Int64(ch_len))
                push_buf[unsafe_offset=off] = 13; push_buf[unsafe_offset=off + 1] = 10; off += 2
                unsafe_memcpy(dest=push_buf.unsafe_offset(off), src=ch_ptr, count=ch_len); off += ch_len
                push_buf[unsafe_offset=off] = 13; push_buf[unsafe_offset=off + 1] = 10; off += 2
                push_buf[unsafe_offset=off] = 36; off += 1
                off += format_int_to_buf(push_buf.unsafe_offset(off), 0, Int64(msg_len))
                push_buf[unsafe_offset=off] = 13; push_buf[unsafe_offset=off + 1] = 10; off += 2
                unsafe_memcpy(dest=push_buf.unsafe_offset(off), src=msg_ptr, count=msg_len); off += msg_len
                push_buf[unsafe_offset=off] = 13; push_buf[unsafe_offset=off + 1] = 10; off += 2
                for si in range(registry.channels[unsafe_offset=ch_idx].fd_count):
                    # gh #172: per-recipient protocol byte, same as handle_publish.
                    var sfd = registry.channels[unsafe_offset=ch_idx].fds[unsafe_offset=si]
                    push_buf[unsafe_offset=0] = 62 if (is_not_null(resp_proto) and resp_proto[unsafe_offset=Int(sfd)] == 3) else 42
                    send_to_fd(sfd, push_buf, off, server)
                push_buf.unsafe_free()
            # Deliver to local pattern subscribers
            for pi in range(registry.pattern_count):
                if not registry.patterns[unsafe_offset=pi].active: continue
                if glob_match(registry.patterns[unsafe_offset=pi].pattern, registry.patterns[unsafe_offset=pi].pattern_len, ch_ptr, ch_len):
                    var pbuf = alloc[UInt8](4096)
                    var phdr = "*4\r\n$8\r\npmessage\r\n"
                    unsafe_memcpy(dest=pbuf, src=phdr.unsafe_ptr(), count=phdr.byte_length()); var po = phdr.byte_length()
                    pbuf[unsafe_offset=po] = 36; po += 1
                    po += format_int_to_buf(pbuf.unsafe_offset(po), 0, Int64(registry.patterns[unsafe_offset=pi].pattern_len))
                    pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
                    unsafe_memcpy(dest=pbuf.unsafe_offset(po), src=registry.patterns[unsafe_offset=pi].pattern, count=registry.patterns[unsafe_offset=pi].pattern_len)
                    po += registry.patterns[unsafe_offset=pi].pattern_len
                    pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
                    pbuf[unsafe_offset=po] = 36; po += 1
                    po += format_int_to_buf(pbuf.unsafe_offset(po), 0, Int64(ch_len))
                    pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
                    unsafe_memcpy(dest=pbuf.unsafe_offset(po), src=ch_ptr, count=ch_len); po += ch_len
                    pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
                    pbuf[unsafe_offset=po] = 36; po += 1
                    po += format_int_to_buf(pbuf.unsafe_offset(po), 0, Int64(msg_len))
                    pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
                    unsafe_memcpy(dest=pbuf.unsafe_offset(po), src=msg_ptr, count=msg_len); po += msg_len
                    pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
                    var pfd = registry.patterns[unsafe_offset=pi].fd
                    pbuf[unsafe_offset=0] = 62 if (is_not_null(resp_proto) and resp_proto[unsafe_offset=Int(pfd)] == 3) else 42
                    send_to_fd(pfd, pbuf, po, server)
                    pbuf.unsafe_free()


# ── Glob pattern matching (Redis-style: *, ?, [abc], \ escape) ──

@always_inline
def glob_match(pat: Pointer[UInt8, MutUntrackedOrigin], plen: Int, s: Pointer[UInt8, MutUntrackedOrigin], slen: Int) -> Bool:
    var pi = 0; var si = 0
    var star_pi = -1; var star_si = -1
    while si < slen:
        if pi < plen and pat[unsafe_offset=pi] == 42:  # '*'
            star_pi = pi; star_si = si; pi += 1; continue
        if pi < plen and (pat[unsafe_offset=pi] == 63 or pat[unsafe_offset=pi] == s[unsafe_offset=si]):  # '?' or exact match
            pi += 1; si += 1; continue
        if pi < plen and pat[unsafe_offset=pi] == 91:  # '['
            pi += 1
            var negate = False
            if pi < plen and pat[unsafe_offset=pi] == 94:  # '^'
                negate = True; pi += 1
            var matched = False
            while pi < plen and pat[unsafe_offset=pi] != 93:  # ']'
                if pi + 2 < plen and pat[unsafe_offset=pi + 1] == 45:  # range a-z
                    if s[unsafe_offset=si] >= pat[unsafe_offset=pi] and s[unsafe_offset=si] <= pat[unsafe_offset=pi + 2]: matched = True
                    pi += 3
                else:
                    if pat[unsafe_offset=pi] == s[unsafe_offset=si]: matched = True
                    pi += 1
            if pi < plen: pi += 1  # skip ']'
            if matched == negate: # negate XOR matched
                if star_pi >= 0: pi = star_pi + 1; star_si += 1; si = star_si; continue
                return False
            si += 1; continue
        if pi < plen and pat[unsafe_offset=pi] == 92 and pi + 1 < plen:  # '\'  escape
            pi += 1
            if pat[unsafe_offset=pi] == s[unsafe_offset=si]: pi += 1; si += 1; continue
        # Mismatch — backtrack to last '*'
        if star_pi >= 0:
            pi = star_pi + 1; star_si += 1; si = star_si; continue
        return False
    # Consume trailing '*'
    while pi < plen and pat[unsafe_offset=pi] == 42:
        pi += 1
    return pi == plen


struct PatternEntry(Copyable, Movable, ImplicitlyCopyable):
    var pattern: Pointer[UInt8, MutUntrackedOrigin]
    var pattern_len: Int
    var fd: Int32
    var active: Bool

    def __init__(out self):
        self.pattern = null_ptr[UInt8, MutUntrackedOrigin]()
        self.pattern_len = 0; self.fd = Int32(-1); self.active = False


struct ChannelEntry(Copyable, Movable, ImplicitlyCopyable):
    var name: Pointer[UInt8, MutUntrackedOrigin]
    var name_len: Int
    var fds: Pointer[Int32, MutUntrackedOrigin]  # subscriber fds (heap-allocated, max 64)
    var fd_count: Int
    var active: Bool

    def __init__(out self):
        self.name = null_ptr[UInt8, MutUntrackedOrigin]()
        self.name_len = 0
        self.fds = alloc[Int32](MAX_SUBS_PER_CHANNEL)
        self.fd_count = 0
        self.active = False


struct PubSubRegistry(Movable):
    var channels: Pointer[ChannelEntry, MutUntrackedOrigin]
    var channel_count: Int
    var patterns: Pointer[PatternEntry, MutUntrackedOrigin]
    var pattern_count: Int
    var fd_sub_counts: Pointer[Int32, MutUntrackedOrigin]  # per-fd subscription count

    def __init__(out self):
        self.channels = alloc[ChannelEntry](MAX_CHANNELS)
        for i in range(MAX_CHANNELS):
            (self.channels.unsafe_offset(i)).unsafe_write(ChannelEntry())
        self.channel_count = 0
        self.patterns = alloc[PatternEntry](MAX_PATTERNS)
        for i in range(MAX_PATTERNS):
            (self.patterns.unsafe_offset(i)).unsafe_write(PatternEntry())
        self.pattern_count = 0
        self.fd_sub_counts = alloc[Int32](MAX_FDS)
        unsafe_memset(self.fd_sub_counts.unsafe_bitcast[UInt8](), 0, MAX_FDS * 4)

    def find_channel(self, name: Pointer[UInt8, MutUntrackedOrigin], name_len: Int) -> Int:
        """Find channel index by name. Returns -1 if not found."""
        for i in range(self.channel_count):
            if self.channels[unsafe_offset=i].active and self.channels[unsafe_offset=i].name_len == name_len:
                var found = True
                for j in range(name_len):
                    if self.channels[unsafe_offset=i].name[unsafe_offset=j] != name[unsafe_offset=j]:
                        found = False
                        break
                if found: return i
        return -1

    def get_or_create_channel(mut self, name: Pointer[UInt8, MutUntrackedOrigin], name_len: Int) -> Int:
        """Find existing channel or create a new one. Returns channel index."""
        var idx = self.find_channel(name, name_len)
        if idx >= 0: return idx
        # Find an inactive slot or append
        for i in range(self.channel_count):
            if not self.channels[unsafe_offset=i].active:
                idx = i
                break
        if idx < 0:
            if self.channel_count >= MAX_CHANNELS: return -1  # full
            idx = self.channel_count
            self.channel_count += 1
        # Initialize channel
        var ch_name = alloc[UInt8](name_len)
        unsafe_memcpy(dest=ch_name, src=name, count=name_len)
        self.channels[unsafe_offset=idx].name = ch_name
        self.channels[unsafe_offset=idx].name_len = name_len
        self.channels[unsafe_offset=idx].fd_count = 0
        self.channels[unsafe_offset=idx].active = True
        return idx

    def subscribe_fd(mut self, ch_idx: Int, fd: Int32):
        """Add fd to channel's subscriber list."""
        if ch_idx < 0 or ch_idx >= self.channel_count: return
        # Check if already subscribed
        for i in range(self.channels[unsafe_offset=ch_idx].fd_count):
            if self.channels[unsafe_offset=ch_idx].fds[unsafe_offset=i] == fd: return
        if self.channels[unsafe_offset=ch_idx].fd_count >= MAX_SUBS_PER_CHANNEL: return
        self.channels[unsafe_offset=ch_idx].fds[unsafe_offset=self.channels[unsafe_offset=ch_idx].fd_count] = fd
        self.channels[unsafe_offset=ch_idx].fd_count += 1
        self.fd_sub_counts[unsafe_offset=Int(fd)] += 1

    def unsubscribe_fd(mut self, ch_idx: Int, fd: Int32):
        """Remove fd from channel's subscriber list."""
        if ch_idx < 0 or ch_idx >= self.channel_count: return
        for i in range(self.channels[unsafe_offset=ch_idx].fd_count):
            if self.channels[unsafe_offset=ch_idx].fds[unsafe_offset=i] == fd:
                # Swap with last
                self.channels[unsafe_offset=ch_idx].fd_count -= 1
                self.channels[unsafe_offset=ch_idx].fds[unsafe_offset=i] = self.channels[unsafe_offset=ch_idx].fds[unsafe_offset=self.channels[unsafe_offset=ch_idx].fd_count]
                if self.fd_sub_counts[unsafe_offset=Int(fd)] > 0:
                    self.fd_sub_counts[unsafe_offset=Int(fd)] -= 1
                # Deactivate channel if empty
                if self.channels[unsafe_offset=ch_idx].fd_count == 0:
                    if is_not_null(self.channels[unsafe_offset=ch_idx].name):
                        self.channels[unsafe_offset=ch_idx].name.unsafe_free()
                        self.channels[unsafe_offset=ch_idx].name = null_ptr[UInt8, MutUntrackedOrigin]()
                    self.channels[unsafe_offset=ch_idx].active = False
                return

    def psubscribe_fd(mut self, pattern: Pointer[UInt8, MutUntrackedOrigin], plen: Int, fd: Int32):
        """Add a pattern subscription for fd."""
        # Check if already subscribed to this pattern
        for i in range(self.pattern_count):
            if self.patterns[unsafe_offset=i].active and self.patterns[unsafe_offset=i].fd == fd and self.patterns[unsafe_offset=i].pattern_len == plen:
                var dup = True
                for j in range(plen):
                    if self.patterns[unsafe_offset=i].pattern[unsafe_offset=j] != pattern[unsafe_offset=j]: dup = False; break
                if dup: return
        # Find slot
        var idx = -1
        for i in range(self.pattern_count):
            if not self.patterns[unsafe_offset=i].active: idx = i; break
        if idx < 0:
            if self.pattern_count >= MAX_PATTERNS: return
            idx = self.pattern_count; self.pattern_count += 1
        var pat_copy = alloc[UInt8](plen)
        unsafe_memcpy(dest=pat_copy, src=pattern, count=plen)
        self.patterns[unsafe_offset=idx].pattern = pat_copy
        self.patterns[unsafe_offset=idx].pattern_len = plen
        self.patterns[unsafe_offset=idx].fd = fd
        self.patterns[unsafe_offset=idx].active = True
        self.fd_sub_counts[unsafe_offset=Int(fd)] += 1

    def punsubscribe_fd(mut self, pattern: Pointer[UInt8, MutUntrackedOrigin], plen: Int, fd: Int32):
        """Remove a specific pattern subscription for fd."""
        for i in range(self.pattern_count):
            if self.patterns[unsafe_offset=i].active and self.patterns[unsafe_offset=i].fd == fd and self.patterns[unsafe_offset=i].pattern_len == plen:
                var found = True
                for j in range(plen):
                    if self.patterns[unsafe_offset=i].pattern[unsafe_offset=j] != pattern[unsafe_offset=j]: found = False; break
                if found:
                    if is_not_null(self.patterns[unsafe_offset=i].pattern): self.patterns[unsafe_offset=i].pattern.unsafe_free()
                    self.patterns[unsafe_offset=i].active = False
                    if self.fd_sub_counts[unsafe_offset=Int(fd)] > 0: self.fd_sub_counts[unsafe_offset=Int(fd)] -= 1
                    return

    def punsubscribe_all(mut self, fd: Int32):
        """Remove all pattern subscriptions for fd."""
        for i in range(self.pattern_count):
            if self.patterns[unsafe_offset=i].active and self.patterns[unsafe_offset=i].fd == fd:
                if is_not_null(self.patterns[unsafe_offset=i].pattern): self.patterns[unsafe_offset=i].pattern.unsafe_free()
                self.patterns[unsafe_offset=i].active = False
                if self.fd_sub_counts[unsafe_offset=Int(fd)] > 0: self.fd_sub_counts[unsafe_offset=Int(fd)] -= 1

    def pattern_count_active(self) -> Int:
        """Count active pattern subscriptions."""
        var n = 0
        for i in range(self.pattern_count):
            if self.patterns[unsafe_offset=i].active: n += 1
        return n

    def cleanup_fd(mut self, fd: Int32):
        """Remove fd from all channels and patterns (called on connection close)."""
        for i in range(self.channel_count):
            if self.channels[unsafe_offset=i].active:
                self.unsubscribe_fd(i, fd)
        self.punsubscribe_all(fd)
        self.fd_sub_counts[unsafe_offset=Int(fd)] = 0

    def get_sub_count(self, fd: Int32) -> Int:
        """Get total subscription count for an fd."""
        return Int(self.fd_sub_counts[unsafe_offset=Int(fd)])

    def total_channels(self) -> Int:
        """Count active channels."""
        var n = 0
        for i in range(self.channel_count):
            if self.channels[unsafe_offset=i].active: n += 1
        return n


# ── Send message directly to an fd ──

@always_inline
def send_to_fd(fd: Int32, data: Pointer[UInt8, MutUntrackedOrigin], length: Int, server: TCPServer):
    """Send data directly to a subscriber fd. Best-effort (drops on EAGAIN)."""
    var sent = 0
    while sent < length:
        var n = server.send(fd, data.unsafe_offset(sent), length - sent)
        if n <= 0: break  # EAGAIN or error — drop message for this subscriber
        sent += n


# ── Command Handlers ──

@always_inline
def handle_subscribe(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, fd: Int32, mut registry: PubSubRegistry) -> Int:
    """SUBSCRIBE channel [channel ...]."""
    if i + 1 < num_tokens:
        var n = num_tokens - i - 1
        for si in range(n):
            var ch = tokens[unsafe_offset=i + 1 + si].ptr; var cl = tokens[unsafe_offset=i + 1 + si].length
            var ch_idx = registry.get_or_create_channel(ch, cl)
            if ch_idx >= 0:
                registry.subscribe_fd(ch_idx, fd)
            # gh #172: push type on RESP3, plain array on RESP2 (unchanged wire).
            # The literal length used to be hand-counted as 20 for a 19-byte
            # string, so every SUBSCRIBE confirmation shipped one byte of
            # whatever followed the literal in memory — a frame-sync corruption
            # of the gh #156/#162 class. Emitting the header through the writer
            # removes the hand-count entirely.
            writer.append_push_header(3)
            writer.append_to_response("$9\r\nsubscribe\r\n".unsafe_ptr(), 15)
            writer.append_bulk_string_response(ch, cl)
            writer.append_int_response(Int64(registry.get_sub_count(fd)))
        return n
    else:
        writer.append_error_response("ERR wrong number of arguments for 'subscribe' command")
        return 0


@always_inline
def handle_unsubscribe(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, fd: Int32, mut registry: PubSubRegistry) -> Int:
    """UNSUBSCRIBE [channel ...]."""
    var n = num_tokens - i - 1
    if n == 0:
        # Unsubscribe from all
        for ci in range(registry.channel_count):
            if registry.channels[unsafe_offset=ci].active:
                registry.unsubscribe_fd(ci, fd)
        # gh #172: the hand-counted 30 was one short of the 31-byte literal, so
        # this reply lost its final '\n' and desynced the connection. Built from
        # primitives now, which also gets the RESP3 null (`_`) right.
        writer.append_push_header(3)
        writer.append_to_response("$11\r\nunsubscribe\r\n".unsafe_ptr(), 18)
        writer.append_null_response()
        writer.append_int_response(0)
    else:
        for ui in range(n):
            var uch = tokens[unsafe_offset=i + 1 + ui].ptr; var ucl = tokens[unsafe_offset=i + 1 + ui].length
            var ch_idx = registry.find_channel(uch, ucl)
            if ch_idx >= 0:
                registry.unsubscribe_fd(ch_idx, fd)
            writer.append_push_header(3)
            writer.append_to_response("$11\r\nunsubscribe\r\n".unsafe_ptr(), 18)
            writer.append_bulk_string_response(uch, ucl)
            writer.append_int_response(Int64(registry.get_sub_count(fd)))
    return n


@always_inline
def handle_publish(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                  registry: PubSubRegistry, server: TCPServer,
                  broadcast: Pointer[PubSubBroadcast, MutUntrackedOrigin] = null_ptr[PubSubBroadcast, MutUntrackedOrigin](),
                  worker_id: Int = 0,
                  resp_proto: Pointer[UInt8, MutUntrackedOrigin] = null_ptr[UInt8, MutUntrackedOrigin]()) -> Int:
    """PUBLISH channel message → :N (number of subscribers that received the message).
    When broadcast is provided, also posts to cross-worker broadcast ring."""
    if i + 2 >= num_tokens:
        writer.append_int_response(0)
        return 1
    var ch_ptr = tokens[unsafe_offset=i + 1].ptr; var ch_len = tokens[unsafe_offset=i + 1].length
    var msg_ptr = tokens[unsafe_offset=i + 2].ptr; var msg_len = tokens[unsafe_offset=i + 2].length
    var delivered: Int64 = 0

    # 1. Exact channel subscribers — message format: *3\r\n$7\r\nmessage\r\n...
    var ch_idx = registry.find_channel(ch_ptr, ch_len)
    if ch_idx >= 0 and registry.channels[unsafe_offset=ch_idx].fd_count > 0:
        var push_buf = alloc[UInt8](4096)
        var hdr = "*3\r\n$7\r\nmessage\r\n"
        unsafe_memcpy(dest=push_buf, src=hdr.unsafe_ptr(), count=hdr.byte_length()); var off = hdr.byte_length()
        push_buf[unsafe_offset=off] = 36; off += 1
        off += format_int_to_buf(push_buf.unsafe_offset(off), 0, Int64(ch_len))
        push_buf[unsafe_offset=off] = 13; push_buf[unsafe_offset=off + 1] = 10; off += 2
        unsafe_memcpy(dest=push_buf.unsafe_offset(off), src=ch_ptr, count=ch_len); off += ch_len
        push_buf[unsafe_offset=off] = 13; push_buf[unsafe_offset=off + 1] = 10; off += 2
        push_buf[unsafe_offset=off] = 36; off += 1
        off += format_int_to_buf(push_buf.unsafe_offset(off), 0, Int64(msg_len))
        push_buf[unsafe_offset=off] = 13; push_buf[unsafe_offset=off + 1] = 10; off += 2
        unsafe_memcpy(dest=push_buf.unsafe_offset(off), src=msg_ptr, count=msg_len); off += msg_len
        push_buf[unsafe_offset=off] = 13; push_buf[unsafe_offset=off + 1] = 10; off += 2
        for si in range(registry.channels[unsafe_offset=ch_idx].fd_count):
            # gh #172: one buffer, N recipients, and they need not share a
            # protocol — a RESP3 subscriber and a RESP2 subscriber can sit on
            # the same channel. RESP2 array and RESP3 push differ only in the
            # leading byte for this payload, so stamp it per recipient rather
            # than building the frame twice.
            var sfd = registry.channels[unsafe_offset=ch_idx].fds[unsafe_offset=si]
            push_buf[unsafe_offset=0] = 62 if (is_not_null(resp_proto) and resp_proto[unsafe_offset=Int(sfd)] == 3) else 42
            send_to_fd(sfd, push_buf, off, server)
            delivered += 1
        push_buf.unsafe_free()

    # 2. Pattern subscribers — pmessage format: *4\r\n$8\r\npmessage\r\n$<patlen>\r\n<pattern>\r\n$<chlen>\r\n<channel>\r\n$<msglen>\r\n<msg>\r\n
    for pi in range(registry.pattern_count):
        if not registry.patterns[unsafe_offset=pi].active: continue
        if glob_match(registry.patterns[unsafe_offset=pi].pattern, registry.patterns[unsafe_offset=pi].pattern_len, ch_ptr, ch_len):
            var pbuf = alloc[UInt8](4096)
            var phdr = "*4\r\n$8\r\npmessage\r\n"
            unsafe_memcpy(dest=pbuf, src=phdr.unsafe_ptr(), count=phdr.byte_length()); var po = phdr.byte_length()
            # pattern
            pbuf[unsafe_offset=po] = 36; po += 1
            po += format_int_to_buf(pbuf.unsafe_offset(po), 0, Int64(registry.patterns[unsafe_offset=pi].pattern_len))
            pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
            unsafe_memcpy(dest=pbuf.unsafe_offset(po), src=registry.patterns[unsafe_offset=pi].pattern, count=registry.patterns[unsafe_offset=pi].pattern_len)
            po += registry.patterns[unsafe_offset=pi].pattern_len
            pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
            # channel
            pbuf[unsafe_offset=po] = 36; po += 1
            po += format_int_to_buf(pbuf.unsafe_offset(po), 0, Int64(ch_len))
            pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
            unsafe_memcpy(dest=pbuf.unsafe_offset(po), src=ch_ptr, count=ch_len); po += ch_len
            pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
            # message
            pbuf[unsafe_offset=po] = 36; po += 1
            po += format_int_to_buf(pbuf.unsafe_offset(po), 0, Int64(msg_len))
            pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
            unsafe_memcpy(dest=pbuf.unsafe_offset(po), src=msg_ptr, count=msg_len); po += msg_len
            pbuf[unsafe_offset=po] = 13; pbuf[unsafe_offset=po + 1] = 10; po += 2
            var pfd = registry.patterns[unsafe_offset=pi].fd
            pbuf[unsafe_offset=0] = 62 if (is_not_null(resp_proto) and resp_proto[unsafe_offset=Int(pfd)] == 3) else 42
            send_to_fd(pfd, pbuf, po, server)
            pbuf.unsafe_free()
            delivered += 1

    # Cross-worker broadcast (if multi-worker)
    if is_not_null(broadcast) and broadcast[].ready:
        broadcast[].publish(worker_id, ch_ptr, ch_len, msg_ptr, msg_len)

    writer.append_int_response(delivered)
    return 2


@always_inline
def handle_pubsub(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, registry: PubSubRegistry) -> Int:
    """PUBSUB CHANNELS|NUMSUB|NUMPAT|SHARDCHANNELS|SHARDNUMSUB [arg ...]."""
    if i + 1 < num_tokens:
        var sub = tokens[unsafe_offset=i + 1].ptr; var sl = tokens[unsafe_offset=i + 1].length
        if sl == 6 and (sub[unsafe_offset=0] | 0x20) == 110 and (sub[unsafe_offset=3] | 0x20) == 115 and (sub[unsafe_offset=5] | 0x20) == 98:
            # NUMSUB (n,u,m,s,u,b)
            if i + 2 < num_tokens:
                var n = num_tokens - i - 2
                # *2N\r\n [channel :count]...
                writer.buffer[unsafe_offset=writer.offset] = 42; writer.offset += 1
                writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(n * 2))
                writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10; writer.offset += 2
                for ci in range(n):
                    var ch = tokens[unsafe_offset=i + 2 + ci].ptr; var cl = tokens[unsafe_offset=i + 2 + ci].length
                    writer.append_bulk_string_response(ch, cl)
                    var ch_idx = registry.find_channel(ch, cl)
                    if ch_idx >= 0:
                        writer.append_int_response(Int64(registry.channels[unsafe_offset=ch_idx].fd_count))
                    else:
                        writer.append_int_response(0)
            else:
                writer.append_empty_array_response()
        elif sl == 6 and (sub[unsafe_offset=0] | 0x20) == 110 and (sub[unsafe_offset=3] | 0x20) == 112 and (sub[unsafe_offset=5] | 0x20) == 116:
            # NUMPAT (n,u,m,p,a,t)
            writer.append_int_response(Int64(registry.pattern_count_active()))
        elif sl >= 8 and (sub[unsafe_offset=0] | 0x20) == 99:
            # CHANNELS [pattern]
            var count = registry.total_channels()
            writer.buffer[unsafe_offset=writer.offset] = 42; writer.offset += 1
            writer.offset += format_int_to_buf(writer.buffer.unsafe_offset(writer.offset), 0, Int64(count))
            writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10; writer.offset += 2
            for ci in range(registry.channel_count):
                if registry.channels[unsafe_offset=ci].active:
                    writer.append_bulk_string_response(registry.channels[unsafe_offset=ci].name, registry.channels[unsafe_offset=ci].name_len)
        else:
            writer.append_empty_array_response()
        return num_tokens - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'pubsub' command")
        return 0


# ── Stubs for pattern and shard pub/sub (deferred) ──

@always_inline
def handle_psubscribe(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, fd: Int32, mut registry: PubSubRegistry) -> Int:
    """PSUBSCRIBE pattern [pattern ...] — register pattern subscriptions."""
    if i + 1 < num_tokens:
        var n = num_tokens - i - 1
        for si in range(n):
            var ch = tokens[unsafe_offset=i + 1 + si].ptr; var cl = tokens[unsafe_offset=i + 1 + si].length
            registry.psubscribe_fd(ch, cl, fd)
            writer.append_push_header(3)
            writer.append_to_response("$10\r\npsubscribe\r\n".unsafe_ptr(), 17)
            writer.append_bulk_string_response(ch, cl)
            writer.append_int_response(Int64(registry.get_sub_count(fd)))
        return n
    else:
        writer.append_error_response("ERR wrong number of arguments for 'psubscribe' command")
        return 0


@always_inline
def handle_punsubscribe(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, fd: Int32, mut registry: PubSubRegistry) -> Int:
    """PUNSUBSCRIBE [pattern ...]."""
    var n = num_tokens - i - 1
    if n == 0:
        registry.punsubscribe_all(fd)
        # gh #172: was 31 for a 32-byte literal — same truncation as UNSUBSCRIBE.
        writer.append_push_header(3)
        writer.append_to_response("$12\r\npunsubscribe\r\n".unsafe_ptr(), 19)
        writer.append_null_response()
        writer.append_int_response(0)
    else:
        for ui in range(n):
            var uch = tokens[unsafe_offset=i + 1 + ui].ptr; var ucl = tokens[unsafe_offset=i + 1 + ui].length
            registry.punsubscribe_fd(uch, ucl, fd)
            writer.append_push_header(3)
            writer.append_to_response("$12\r\npunsubscribe\r\n".unsafe_ptr(), 19)
            writer.append_bulk_string_response(uch, ucl)
            writer.append_int_response(Int64(registry.get_sub_count(fd)))
    return n


@always_inline
def handle_ssubscribe(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """SSUBSCRIBE shardchannel [shardchannel ...] — stub."""
    if i + 1 < num_tokens:
        var n = num_tokens - i - 1
        for si in range(n):
            var ch = tokens[unsafe_offset=i + 1 + si].ptr; var cl = tokens[unsafe_offset=i + 1 + si].length
            writer.append_push_header(3)
            writer.append_to_response("$10\r\nssubscribe\r\n".unsafe_ptr(), 17)
            writer.append_bulk_string_response(ch, cl)
            writer.append_int_response(Int64(si + 1))
        return n
    else:
        writer.append_error_response("ERR wrong number of arguments for 'ssubscribe' command")
        return 0


@always_inline
def handle_sunsubscribe(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """SUNSUBSCRIBE [shardchannel ...] — stub."""
    var n = num_tokens - i - 1
    if n == 0:
        # gh #172: was 31 for a 32-byte literal — same truncation as UNSUBSCRIBE.
        writer.append_push_header(3)
        writer.append_to_response("$12\r\nsunsubscribe\r\n".unsafe_ptr(), 19)
        writer.append_null_response()
        writer.append_int_response(0)
    else:
        for ui in range(n):
            var uch = tokens[unsafe_offset=i + 1 + ui].ptr; var ucl = tokens[unsafe_offset=i + 1 + ui].length
            writer.append_push_header(3)
            writer.append_to_response("$12\r\nsunsubscribe\r\n".unsafe_ptr(), 19)
            writer.append_bulk_string_response(uch, ucl)
            writer.append_int_response(Int64(n - ui - 1))
    return n


@always_inline
def handle_spublish(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter) -> Int:
    """SPUBLISH shardchannel message → :0 (stub)."""
    writer.append_int_response(0)
    return 2 if i + 2 < num_tokens else 1
