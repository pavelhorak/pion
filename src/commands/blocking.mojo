"""Blocking list and sorted-set commands (#38): BLPOP, BRPOP, BRPOPLPUSH,
BLMOVE, BLMPOP, BZPOPMIN, BZPOPMAX, BZMPOP.

A blocking command that finds nothing to pop parks its connection, as XREAD
BLOCK and WAIT do (gh #390): no reply yet, and nothing the client pipelined
behind it runs. The parked client keeps a copy of its command. On every event
loop tick while any client is parked, NetworkEngine._service_blocked_clients
looks at each in the order they blocked. Once a key it waits on holds a list
(a sorted set for the BZ forms), or its timeout has passed, the command runs
again, unable to block: it is served now, or it gives the timeout's nil.
Either way the reply is the command's own. Redis serves clients blocked on a
key in the order they blocked, and so does this.

Inside MULTI/EXEC, inside a script and on the XDP lane a blocking command
cannot park, and an empty result answers nil at once, as in Redis.
"""

from src.common.ptr import is_not_null, null_ptr
from std.memory import alloc, unsafe_memcpy
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call
from src.common.hash_map import StripedHashMap
from src.common.value import GenericValue, ValueType
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.common.stream_data import StreamData, sid_lt


struct BlockedClient(Copyable, Movable):
    """One parked connection: its command, and the keys it waits on."""
    var fd: Int32
    var frame: List[UInt8]         # the command's RESP frame, run again when it wakes
    var keys: List[UInt8]          # the keys, back to back
    var key_ends: List[Int]        # where each key in `keys` ends
    var deadline_ms: Int64         # 0 = no timeout
    var zset: Bool                 # waits for a sorted set (BZ*), else a list
    var nil_bulk: Bool             # times out with a nil bulk (BRPOPLPUSH, BLMOVE), else a nil array
    var unblock: UInt8             # CLIENT UNBLOCK (#47): UNBLOCK_TIMEOUT or UNBLOCK_ERROR, else 0
    var group: List[UInt8]         # XREADGROUP (#40): its consumer group; empty otherwise

    def __init__(out self, fd: Int32, deadline_ms: Int64, zset: Bool, nil_bulk: Bool):
        self.fd = fd
        self.frame = List[UInt8]()
        self.keys = List[UInt8]()
        self.key_ends = List[Int]()
        self.deadline_ms = deadline_ms
        self.zset = zset
        self.nil_bulk = nil_bulk
        self.unblock = 0
        self.group = List[UInt8]()


# CLIENT UNBLOCK's reasons (#47), kept on a parked client until the event
# loop answers it: TIMEOUT answers as its timeout would, ERROR with
# UNBLOCKED_ERROR. Either way the command does not run again.
comptime UNBLOCK_TIMEOUT = UInt8(1)
comptime UNBLOCK_ERROR = UInt8(2)
comptime UNBLOCKED_ERROR = "UNBLOCKED client unblocked via CLIENT UNBLOCK"


struct BlockedClientRegistry(Movable):
    """Per-worker parked blocking clients, oldest first."""
    var clients: List[BlockedClient]
    var count_ptr: Pointer[Int, MutUntrackedOrigin]   # len(clients), read by every event-loop tick

    def __init__(out self):
        self.clients = List[BlockedClient]()
        self.count_ptr = alloc[Int](1)
        self.count_ptr[unsafe_offset=0] = 0

    def __init__(out self, *, deinit take: Self):
        self.clients = take.clients^
        self.count_ptr = take.count_ptr

    @always_inline
    def _count(self) -> Int:
        return self.count_ptr[unsafe_offset=0]

    def add(mut self, var c: BlockedClient):
        self.clients.append(c^)
        self.count_ptr[unsafe_offset=0] = len(self.clients)

    def remove_at(mut self, k: Int):
        """Drop client k, keeping the others in the order they blocked."""
        _ = self.clients.pop(k)
        self.count_ptr[unsafe_offset=0] = len(self.clients)

    def remove_fd(mut self, fd: Int32):
        """The connection closed while blocked."""
        var k = 0
        while k < len(self.clients):
            if self.clients[k].fd == fd:
                self.remove_at(k)
            else:
                k += 1


def parse_block_timeout(t: RESP3Token, now_ms: Int64, mut writer: ResponseWriter) -> Int64:
    """The deadline in ms (0 = none), or -1 with Redis's error written:
    getTimeoutFromObjectOrReply for seconds (src/ffi/fcntl_wrap.c)."""
    var out = alloc[Int64](1)
    out[unsafe_offset=0] = 0
    var rc = external_call["pion_parse_block_timeout", Int64](t.ptr, Int64(t.length), now_ms, out)
    var v = out[unsafe_offset=0]
    out.free()
    if rc == 1:
        writer.append_error_response("ERR timeout is not a float or out of range")
        return -1
    if rc == 2:
        writer.append_error_response("ERR timeout is out of range")
        return -1
    if rc == 3:
        writer.append_error_response("ERR timeout is negative")
        return -1
    return v


def new_blocked_client(fd: Int32, deadline_ms: Int64, zset: Bool, nil_bulk: Bool,
                       frame: Pointer[UInt8, MutUntrackedOrigin], frame_len: Int,
                       tokens: Pointer[RESP3Token, MutUntrackedOrigin], key_first: Int, key_end: Int) -> BlockedClient:
    """A client for tokens[key_first:key_end] (already the keyspace's keys,
    tenant prefix included) and a copy of its command's frame."""
    var c = BlockedClient(fd, deadline_ms, zset, nil_bulk)
    for b in range(frame_len):
        c.frame.append(frame[unsafe_offset=b])
    for k in range(key_first, key_end):
        var t = tokens[unsafe_offset=k]
        for b in range(t.length):
            c.keys.append(t.ptr[unsafe_offset=b])
        c.key_ends.append(len(c.keys))
    return c^


def blocked_client_ready(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], c: BlockedClient) -> Bool:
    """Does a key the client waits on hold what it pops? (Emptied containers
    are removed, so a key of the right type has an element.) A key of another
    type does not wake it: in Redis the client stays blocked."""
    if len(c.group) > 0:
        return xreadgroup_ready(keyspace, c.keys, c.key_ends, c.group)
    var want = ValueType.ZSET if c.zset else ValueType.LIST
    var start = 0
    var base = c.keys.unsafe_ptr()
    for k in range(len(c.key_ends)):
        var end = c.key_ends[k]
        var v = keyspace[].get(GenericValue.borrow(
            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(base) + start), end - start))
        if not v.is_none() and v.type.value == want:
            return True
        start = end
    return False


def xreadgroup_ready(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                     keys: List[UInt8], key_ends: List[Int], group: List[UInt8]) -> Bool:
    """#40: should a parked XREADGROUP run again? When a stream has an entry
    past the group's last-delivered id, or a stream or the group is gone (the
    run then answers NOGROUP, as Redis's does)."""
    var start = 0
    var base = keys.unsafe_ptr()
    var gp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(group.unsafe_ptr()))
    for k in range(len(key_ends)):
        var end = key_ends[k]
        var v = keyspace[].get(GenericValue.borrow(
            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(base) + start), end - start))
        start = end
        if v.is_none() or v.type.value != ValueType.STREAM:
            return True
        var sd = v.as_hash().unsafe_bitcast[StreamData]()
        var g = sd[].group_index(gp, len(group))
        if g < 0:
            return True
        var ll = sd[].last_live()
        if ll >= 0:
            var e = sd[].entries[unsafe_offset=ll]
            if sid_lt(sd[].groups[g].last_ms, sd[].groups[g].last_seq, e.id_ms, e.id_seq):
                return True
    return False
