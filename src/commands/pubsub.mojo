"""Pub/Sub: SUBSCRIBE, UNSUBSCRIBE, PSUBSCRIBE, PUNSUBSCRIBE, PUBLISH, PUBSUB,
SSUBSCRIBE, SUNSUBSCRIBE, SPUBLISH.

Each worker keeps its own subscriptions, with no cap on channels, patterns or
subscribers. (It held 256 channels of 64 subscribers and 256 patterns, and
acknowledged a subscription past them that then never received anything, #42.)

A message is built once, in a buffer sized to it, and goes to each subscriber
through the writer's per-connection output buffer (ResponseWriter.deliver_to):
what a slow subscriber cannot take yet waits for its write event, and one that
falls too far behind is disconnected, as Redis does past its output-buffer
limit, rather than cut mid-frame. The connection that publishes gets its own
copy in its reply stream, before PUBLISH's reply. (Messages were built in a
fixed 4 KB buffer with no bound, and send() was tried once.)

Shard channels (SSUBSCRIBE, SPUBLISH) are a namespace of their own, delivered
as `smessage`, as Redis serves them outside cluster mode.

Between workers (`--independent-workers`), PUBLISH and SPUBLISH post the whole
message to every other worker's inbox (src/ffi/fcntl_wrap.c), and each worker
takes its inbox on its tick and delivers to its own subscribers. PUBLISH counts
the subscribers of its own worker, as a Redis Cluster node counts its own.
"""
from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy, unsafe_memset
from std.ffi import external_call
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.common.utils import format_int_to_buf, _glob_match, arg_eq

comptime MAX_FDS = 65536

comptime KIND_CHANNEL = 0
comptime KIND_PATTERN = 1
comptime KIND_SHARD = 2


struct Subscription(Copyable, Movable):
    """A channel, pattern or shard channel, and the connections subscribed to
    it, in the order they subscribed."""
    var name: List[UInt8]
    var fds: List[Int32]

    def __init__(out self, name: List[UInt8]):
        self.name = name.copy()
        self.fds = List[Int32]()


def _bytes(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    for k in range(n):
        out.append(p[k])
    return out^


def _same(a: List[UInt8], p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    if len(a) != n:
        return False
    for k in range(n):
        if a[k] != p[k]:
            return False
    return True


struct PubSubRegistry(Movable):
    var channels: List[Subscription]
    var patterns: List[Subscription]
    var shards: List[Subscription]
    var counts: Pointer[Int32, MutUntrackedOrigin]        # channels + patterns, per fd
    var shard_counts: Pointer[Int32, MutUntrackedOrigin]  # shard channels, per fd
    var subscribed_fds: Int                               # connections with any subscription

    def __init__(out self):
        self.channels = List[Subscription]()
        self.patterns = List[Subscription]()
        self.shards = List[Subscription]()
        self.counts = alloc[Int32](MAX_FDS)
        self.shard_counts = alloc[Int32](MAX_FDS)
        unsafe_memset(self.counts.unsafe_bitcast[UInt8](), 0, MAX_FDS * 4)
        unsafe_memset(self.shard_counts.unsafe_bitcast[UInt8](), 0, MAX_FDS * 4)
        self.subscribed_fds = 0

    def __init__(out self, *, deinit take: Self):
        self.channels = take.channels^
        self.patterns = take.patterns^
        self.shards = take.shards^
        self.counts = take.counts
        self.shard_counts = take.shard_counts
        self.subscribed_fds = take.subscribed_fds

    def _list(mut self, kind: Int) -> Pointer[List[Subscription], MutUntrackedOrigin]:
        if kind == KIND_PATTERN:
            return Pointer[List[Subscription], MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=self.patterns)))
        if kind == KIND_SHARD:
            return Pointer[List[Subscription], MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=self.shards)))
        return Pointer[List[Subscription], MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=self.channels)))

    def find(mut self, kind: Int, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
        var l = self._list(kind)
        for k in range(len(l[])):
            if _same(l[][k].name, p, n):
                return k
        return -1

    def subscribe(mut self, kind: Int, fd: Int32, p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
        """Subscribe fd (no-op when it already is)."""
        var l = self._list(kind)
        var k = self.find(kind, p, n)
        if k < 0:
            l[].append(Subscription(_bytes(p, n)))
            k = len(l[]) - 1
        for j in range(len(l[][k].fds)):
            if l[][k].fds[j] == fd:
                return
        l[][k].fds.append(fd)
        if not self.subscribed(fd):
            self.subscribed_fds += 1
        if kind == KIND_SHARD:
            self.shard_counts[Int(fd)] += 1
        else:
            self.counts[Int(fd)] += 1

    def unsubscribe(mut self, kind: Int, fd: Int32, p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
        var k = self.find(kind, p, n)
        if k >= 0:
            self._drop(kind, k, fd)

    def _drop(mut self, kind: Int, k: Int, fd: Int32):
        var l = self._list(kind)
        for j in range(len(l[][k].fds)):
            if l[][k].fds[j] == fd:
                _ = l[][k].fds.pop(j)
                if kind == KIND_SHARD:
                    self.shard_counts[Int(fd)] -= 1
                else:
                    self.counts[Int(fd)] -= 1
                if not self.subscribed(fd):
                    self.subscribed_fds -= 1
                if len(l[][k].fds) == 0:
                    _ = l[].pop(k)
                return

    def names_of(mut self, kind: Int, fd: Int32) -> List[List[UInt8]]:
        """What fd is subscribed to, of one kind, in the order they were made."""
        var out = List[List[UInt8]]()
        var l = self._list(kind)
        for k in range(len(l[])):
            for j in range(len(l[][k].fds)):
                if l[][k].fds[j] == fd:
                    out.append(l[][k].name.copy())
                    break
        return out^

    def cleanup_fd(mut self, fd: Int32):
        """The connection closed or RESET: every subscription of it goes."""
        for kind in range(3):
            var l = self._list(kind)
            var k = 0
            while k < len(l[]):
                var before = len(l[])
                self._drop(kind, k, fd)
                if len(l[]) == before:
                    k += 1
        self.counts[Int(fd)] = 0
        self.shard_counts[Int(fd)] = 0

    @always_inline
    def get_sub_count(self, fd: Int32) -> Int:
        """Channels + patterns: the count a (P)SUBSCRIBE reply carries, and
        nonzero while a RESP2 connection is in subscribed mode."""
        return Int(self.counts[Int(fd)])

    @always_inline
    def get_shard_count(self, fd: Int32) -> Int:
        return Int(self.shard_counts[Int(fd)])

    def subscribed(self, fd: Int32) -> Bool:
        return self.counts[Int(fd)] > 0 or self.shard_counts[Int(fd)] > 0


# ── Delivery ──

def _put_len(mut out: List[UInt8], prefix: UInt8, n: Int, tmp: Pointer[UInt8, MutUntrackedOrigin]):
    out.append(prefix)
    var e = format_int_to_buf(tmp, 0, Int64(n))
    for k in range(e):
        out.append(tmp[k])
    out.append(13)
    out.append(10)


def _put_bulk(mut out: List[UInt8], p: Pointer[UInt8, MutUntrackedOrigin], n: Int,
              tmp: Pointer[UInt8, MutUntrackedOrigin]):
    _put_len(out, 36, n, tmp)
    for k in range(n):
        out.append(p[k])
    out.append(13)
    out.append(10)


def _frame(kind: Int, pattern: List[UInt8], ch: Pointer[UInt8, MutUntrackedOrigin], cl: Int,
           msg: Pointer[UInt8, MutUntrackedOrigin], ml: Int) -> List[UInt8]:
    """`message`, `pmessage` or `smessage`, RESP2 framed: byte 0 is restamped
    `>` for a RESP3 recipient (the rest of the frame is the same)."""
    var out = List[UInt8](capacity=ml + cl + len(pattern) + 64)
    var tmp = alloc[UInt8](24)
    var word = List[UInt8]()
    var ws = "pmessage" if kind == KIND_PATTERN else ("smessage" if kind == KIND_SHARD else "message")
    for b in ws.as_bytes():
        word.append(b)
    _put_len(out, 42, 4 if kind == KIND_PATTERN else 3, tmp)
    _put_bulk(out, Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(word.unsafe_ptr())), len(word), tmp)
    if kind == KIND_PATTERN:
        _put_bulk(out, Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pattern.unsafe_ptr())), len(pattern), tmp)
    _put_bulk(out, ch, cl, tmp)
    _put_bulk(out, msg, ml, tmp)
    tmp.unsafe_free()
    _ = word^
    return out^


def _send_frame(mut frame: List[UInt8], to: Int32, self_fd: Int32, mut writer: ResponseWriter,
                server: TCPServer, kq: Int32, resp_proto: Pointer[UInt8, MutUntrackedOrigin]):
    frame[0] = 62 if (is_not_null(resp_proto) and resp_proto[Int(to)] == 3) else 42
    var fp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(frame.unsafe_ptr()))
    if to == self_fd:
        writer.append_to_response(fp, len(frame))      # in its own reply stream
    else:
        writer.deliver_to(to, fp, len(frame), server, kq)


def publish_local(mut registry: PubSubRegistry, shard: Bool,
                  ch: Pointer[UInt8, MutUntrackedOrigin], cl: Int,
                  msg: Pointer[UInt8, MutUntrackedOrigin], ml: Int,
                  self_fd: Int32, mut writer: ResponseWriter, server: TCPServer, kq: Int32,
                  resp_proto: Pointer[UInt8, MutUntrackedOrigin]) -> Int:
    """Deliver to this worker's subscribers: the channel's, then each matching
    pattern's (or the shard channel's). Returns how many got it."""
    var delivered = 0
    var none = List[UInt8]()
    var kind = KIND_SHARD if shard else KIND_CHANNEL
    var k = registry.find(kind, ch, cl)
    if k >= 0:
        var frame = _frame(kind, none, ch, cl, msg, ml)
        var l = registry._list(kind)
        var fds = l[][k].fds.copy()
        for j in range(len(fds)):
            _send_frame(frame, fds[j], self_fd, writer, server, kq, resp_proto)
            delivered += 1
    if not shard:
        var pl = registry._list(KIND_PATTERN)
        for pk in range(len(pl[])):
            var pat = pl[][pk].name.copy()
            var pp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pat.unsafe_ptr()))
            if _glob_match(pp, len(pat), 0, ch, cl, 0):
                var frame = _frame(KIND_PATTERN, pat, ch, cl, msg, ml)
                var fds = pl[][pk].fds.copy()
                for j in range(len(fds)):
                    _send_frame(frame, fds[j], self_fd, writer, server, kq, resp_proto)
                    delivered += 1
            _ = pat^
    return delivered


def pubsub_post(worker_id: Int, num_workers: Int, shard: Bool,
                ch: Pointer[UInt8, MutUntrackedOrigin], cl: Int,
                msg: Pointer[UInt8, MutUntrackedOrigin], ml: Int):
    """Hand the message to every other worker (src/ffi/fcntl_wrap.c)."""
    if num_workers > 1:
        _ = external_call["pion_pubsub_post", Int32](Int32(worker_id), Int32(KIND_SHARD if shard else KIND_CHANNEL),
                                                    ch, Int64(cl), msg, Int64(ml))


@always_inline
def _u32_at(p: Pointer[UInt8, MutUntrackedOrigin], at: Int) -> Int:
    """A little-endian u32 at any alignment (the records are packed)."""
    return Int(p[at]) | (Int(p[at + 1]) << 8) | (Int(p[at + 2]) << 16) | (Int(p[at + 3]) << 24)


def pubsub_drain(mut registry: PubSubRegistry, worker_id: Int, mut writer: ResponseWriter,
                 server: TCPServer, kq: Int32, resp_proto: Pointer[UInt8, MutUntrackedOrigin]):
    """Deliver what other workers published since the last tick."""
    var out = alloc[Pointer[UInt8, MutUntrackedOrigin]](1)
    var n = Int(external_call["pion_pubsub_take", Int64](Int32(worker_id), out))
    var buf = out[0]
    out.unsafe_free()
    if n <= 0:
        return
    var off = 0
    while off + 9 <= n:
        var kind = Int(buf[off])
        var cl = _u32_at(buf, off + 1)
        var ml = _u32_at(buf, off + 5)
        var ch = buf.unsafe_offset(off + 9)
        var msg = buf.unsafe_offset(off + 9 + cl)
        _ = publish_local(registry, kind == KIND_SHARD, ch, cl, msg, ml, Int32(-1), writer, server, kq, resp_proto)
        off += 9 + cl + ml
    external_call["pion_lcs_free", NoneType](buf)


# ── Command handlers ──

def _confirm(mut writer: ResponseWriter, word: String, name: Pointer[UInt8, MutUntrackedOrigin], n: Int,
             null_name: Bool, count: Int):
    """A (p|s)(un)subscribe confirmation: [word, name or nil, count]."""
    writer.append_push_header(3)
    writer.append_bulk_string_response(word.unsafe_ptr(), word.byte_length())
    if null_name:
        writer.append_null_response()
    else:
        writer.append_bulk_string_response(name, n)
    writer.append_int_response(Int64(count))


def handle_subscribe_kind(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                          mut writer: ResponseWriter, fd: Int32, mut registry: PubSubRegistry, kind: Int) -> Int:
    """SUBSCRIBE / PSUBSCRIBE / SSUBSCRIBE name [name ...]: one confirmation
    per name, with the connection's count after it."""
    var word = String("psubscribe") if kind == KIND_PATTERN else (String("ssubscribe") if kind == KIND_SHARD else String("subscribe"))
    if num_tokens - i < 2:
        writer.append_error_response("ERR wrong number of arguments for '" + word + "' command")
        return 0
    for k in range(i + 1, num_tokens):
        var t = tokens[k]
        registry.subscribe(kind, fd, t.ptr, t.length)
        var count = registry.get_shard_count(fd) if kind == KIND_SHARD else registry.get_sub_count(fd)
        _confirm(writer, word, t.ptr, t.length, False, count)
    return num_tokens - i - 1


def handle_unsubscribe_kind(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                            mut writer: ResponseWriter, fd: Int32, mut registry: PubSubRegistry, kind: Int) -> Int:
    """UNSUBSCRIBE / PUNSUBSCRIBE / SUNSUBSCRIBE [name ...]: from the names
    given, or from every one of that kind; one confirmation each, or one with
    a nil name when there was nothing to leave."""
    var word = String("punsubscribe") if kind == KIND_PATTERN else (String("sunsubscribe") if kind == KIND_SHARD else String("unsubscribe"))
    if num_tokens - i == 1:
        var names = registry.names_of(kind, fd)
        if len(names) == 0:
            var count = registry.get_shard_count(fd) if kind == KIND_SHARD else registry.get_sub_count(fd)
            _confirm(writer, word, null_ptr[UInt8, MutUntrackedOrigin](), 0, True, count)
        for k in range(len(names)):
            var np = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(names[k].unsafe_ptr()))
            registry.unsubscribe(kind, fd, np, len(names[k]))
            var count = registry.get_shard_count(fd) if kind == KIND_SHARD else registry.get_sub_count(fd)
            _confirm(writer, word, np, len(names[k]), False, count)
        _ = names^
        return 0
    for k in range(i + 1, num_tokens):
        var t = tokens[k]
        registry.unsubscribe(kind, fd, t.ptr, t.length)
        var count = registry.get_shard_count(fd) if kind == KIND_SHARD else registry.get_sub_count(fd)
        _confirm(writer, word, t.ptr, t.length, False, count)
    return num_tokens - i - 1


def handle_publish_kind(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                        mut writer: ResponseWriter, dw: Pointer[ResponseWriter, MutUntrackedOrigin], kq: Int32,
                        fd: Int32, mut registry: PubSubRegistry, server: TCPServer,
                        resp_proto: Pointer[UInt8, MutUntrackedOrigin], worker_id: Int,
                        num_workers: Int, shard: Bool) -> Int:
    """PUBLISH / SPUBLISH channel message → :N, the subscribers on this worker
    that received it. `writer` takes the reply; `dw`, the engine's writer when
    a script runs (its `writer` only captures the reply), delivers the message.
    Null outside a script, where `writer` delivers: a pointer to the same
    object as a `mut` argument would alias it."""
    if num_tokens - i != 3:
        writer.append_error_response("ERR wrong number of arguments for '" + String("spublish" if shard else "publish")
                                     + "' command")
        return 0
    var ch = tokens[i + 1]
    var msg = tokens[i + 2]
    var n: Int
    if is_null(dw):
        n = publish_local(registry, shard, ch.ptr, ch.length, msg.ptr, msg.length, fd, writer, server, kq, resp_proto)
    else:
        n = publish_local(registry, shard, ch.ptr, ch.length, msg.ptr, msg.length, fd, dw[], server, kq, resp_proto)
    pubsub_post(worker_id, num_workers, shard, ch.ptr, ch.length, msg.ptr, msg.length)
    writer.append_int_response(Int64(n))
    return 2


def handle_pubsub(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                  mut writer: ResponseWriter, mut registry: PubSubRegistry) -> Int:
    """PUBSUB CHANNELS|NUMSUB|NUMPAT|SHARDCHANNELS|SHARDNUMSUB|HELP, for this
    worker's subscriptions."""
    if num_tokens - i < 2:
        writer.append_error_response("ERR wrong number of arguments for 'pubsub' command")
        return 0
    var sub = tokens[i + 1]
    var argc = num_tokens - i
    if arg_eq(sub.ptr, sub.length, "channels") or arg_eq(sub.ptr, sub.length, "shardchannels"):
        if argc > 3:
            writer.append_error_response("ERR unknown subcommand or wrong number of arguments for '" + sub.value()
                                         + "'. Try PUBSUB HELP.")
            return 0
        var l = registry._list(KIND_SHARD if arg_eq(sub.ptr, sub.length, "shardchannels") else KIND_CHANNEL)
        var picked = List[Int]()
        for k in range(len(l[])):
            if argc == 3:
                var nm = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(l[][k].name.unsafe_ptr()))
                if not _glob_match(tokens[i + 2].ptr, tokens[i + 2].length, 0, nm, len(l[][k].name), 0):
                    continue
            picked.append(k)
        writer.append_array_header(len(picked))
        for k in range(len(picked)):
            var nm = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(l[][picked[k]].name.unsafe_ptr()))
            writer.append_bulk_string_response(nm, len(l[][picked[k]].name))
    elif arg_eq(sub.ptr, sub.length, "numsub") or arg_eq(sub.ptr, sub.length, "shardnumsub"):
        var kind = KIND_SHARD if arg_eq(sub.ptr, sub.length, "shardnumsub") else KIND_CHANNEL
        var n = argc - 2
        # a flat [name, count, ...] array in both protocols, as Redis
        writer.append_array_header(n * 2)
        for k in range(i + 2, num_tokens):
            var t = tokens[k]
            writer.append_bulk_string_response(t.ptr, t.length)
            var at = registry.find(kind, t.ptr, t.length)
            var l = registry._list(kind)
            writer.append_int_response(Int64(len(l[][at].fds)) if at >= 0 else 0)
    elif arg_eq(sub.ptr, sub.length, "numpat"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'pubsub|numpat' command")
            return 0
        var l = registry._list(KIND_PATTERN)
        writer.append_int_response(Int64(len(l[])))
    elif arg_eq(sub.ptr, sub.length, "help"):
        var lines = List[String]()
        lines.append("PUBSUB <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
        lines.append("CHANNELS [<pattern>]")
        lines.append("    Return the currently active channels matching a <pattern> (default: '*').")
        lines.append("NUMPAT")
        lines.append("    Return number of subscriptions to patterns.")
        lines.append("NUMSUB [<channel> ...]")
        lines.append("    Return the number of subscribers for the specified channels, excluding")
        lines.append("    pattern subscriptions(default: no channels).")
        lines.append("SHARDCHANNELS [<pattern>]")
        lines.append("    Return the currently active shard level channels matching a <pattern> (default: '*').")
        lines.append("SHARDNUMSUB [<shardchannel> ...]")
        lines.append("    Return the number of subscribers for the specified shard level channel(s)")
        lines.append("HELP")
        lines.append("    Print this help.")
        writer.append_array_header(len(lines))
        for k in range(len(lines)):
            writer.append_status_response(lines[k])
    else:
        writer.append_error_response("ERR unknown subcommand '" + sub.value() + "'. Try PUBSUB HELP.")
    return num_tokens - i - 1
