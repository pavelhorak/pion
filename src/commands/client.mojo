"""CLIENT (#47): the connections of this worker, as Redis reports them.

ClientRegistry keeps what CLIENT needs per connection: an ID that is never
reused (CLIENT ID used to be the socket fd, so a killed client's ID came back
on the next connection that got its fd), the connect and last-input times,
the library name and version, the REPLY mode, and the pause.

What changed:
  * CLIENT LIST and INFO give the peer as ip:port, the local address, age and
    idle, flags (N, P, x, b, O, r, T, e), the subscription counts, MULTI and
    WATCH, the input buffer, the user (default or a tenant), the protocol and
    the library. Fields Pion does not measure (memory and network counters,
    the last command) are left out rather than invented.
  * KILL (ip:port, or the ID / ADDR / LADDR / TYPE / USER / SKIPME / MAXAGE
    filters), PAUSE / UNPAUSE, REPLY, UNBLOCK and HELP, which were refused.
  * TRACKING (client-side caching) is refused: Pion sends no invalidations.
    GETREDIR and TRACKINGINFO say tracking is off, which is true.

Each worker has its own registry, as it has its own keyspace: with
--independent-workers, CLIENT sees and acts on the connections of the worker
it runs on. IDs are unique across the workers.
"""

from std.memory import alloc, unsafe_memset
from std.memory.unsafe_pointer import Pointer
from std.collections import List, Dict
from std.ffi import external_call
from src.network.response_writer import ResponseWriter

comptime CLIENT_MAX_FDS = 65536

comptime REPLY_ON = UInt8(0)
comptime REPLY_OFF = UInt8(1)
comptime REPLY_SKIP_NEXT = UInt8(2)  # CLIENT REPLY SKIP ran: the next command's reply is dropped
comptime REPLY_SKIP_NOW = UInt8(3)   # this command's reply is dropped


struct ClientRegistry(Movable):
    var ids: Pointer[UInt64, MutUntrackedOrigin]        # 0: no connection
    var ctime_ms: Pointer[Int64, MutUntrackedOrigin]    # unix ms at accept
    var atime: Pointer[UInt64, MutUntrackedOrigin]      # pion_ticks() at the last input
    var reply: Pointer[UInt8, MutUntrackedOrigin]       # REPLY_*
    var no_touch: Pointer[UInt8, MutUntrackedOrigin]
    var no_evict: Pointer[UInt8, MutUntrackedOrigin]
    var postponed: Pointer[UInt8, MutUntrackedOrigin]   # 1: CLIENT PAUSE holds its next command
    var close_after: Pointer[UInt8, MutUntrackedOrigin] # 1: CLIENT KILL of itself: closed once its reply is out
    var lib_name: Dict[Int, String]
    var lib_ver: Dict[Int, String]
    var reply_off_count: Int          # connections not in REPLY ON (the dispatch gate)
    var pause_until_ms: Int64         # 0: not paused
    var pause_all: Bool               # ALL; WRITE otherwise
    var paused_fds: List[Int32]       # held by the pause, in the order they were held
    var released: List[Int32]         # let go before the pause ended (CLIENT KILL): the engine resumes them
    var pause_changed: Bool           # PAUSE / UNPAUSE ran: the engine lets every held client try again
    var ticks_per_ms: Float64

    def __init__(out self):
        self.ids = alloc[UInt64](CLIENT_MAX_FDS)
        unsafe_memset(self.ids.bitcast[UInt8](), 0, CLIENT_MAX_FDS * 8)
        self.ctime_ms = alloc[Int64](CLIENT_MAX_FDS)
        unsafe_memset(self.ctime_ms.bitcast[UInt8](), 0, CLIENT_MAX_FDS * 8)
        self.atime = alloc[UInt64](CLIENT_MAX_FDS)
        unsafe_memset(self.atime.bitcast[UInt8](), 0, CLIENT_MAX_FDS * 8)
        self.reply = alloc[UInt8](CLIENT_MAX_FDS)
        unsafe_memset(self.reply, 0, CLIENT_MAX_FDS)
        self.no_touch = alloc[UInt8](CLIENT_MAX_FDS)
        unsafe_memset(self.no_touch, 0, CLIENT_MAX_FDS)
        self.no_evict = alloc[UInt8](CLIENT_MAX_FDS)
        unsafe_memset(self.no_evict, 0, CLIENT_MAX_FDS)
        self.postponed = alloc[UInt8](CLIENT_MAX_FDS)
        unsafe_memset(self.postponed, 0, CLIENT_MAX_FDS)
        self.close_after = alloc[UInt8](CLIENT_MAX_FDS)
        unsafe_memset(self.close_after, 0, CLIENT_MAX_FDS)
        self.lib_name = Dict[Int, String]()
        self.lib_ver = Dict[Int, String]()
        self.reply_off_count = 0
        self.pause_until_ms = 0
        self.pause_all = False
        self.paused_fds = List[Int32]()
        self.released = List[Int32]()
        self.pause_changed = False
        self.ticks_per_ms = external_call["pion_ticks_per_us", Float64]() * 1000.0

    def __init__(out self, *, deinit take: Self):
        self.ids = take.ids
        self.ctime_ms = take.ctime_ms
        self.atime = take.atime
        self.reply = take.reply
        self.no_touch = take.no_touch
        self.no_evict = take.no_evict
        self.postponed = take.postponed
        self.close_after = take.close_after
        self.lib_name = take.lib_name^
        self.lib_ver = take.lib_ver^
        self.reply_off_count = take.reply_off_count
        self.pause_until_ms = take.pause_until_ms
        self.pause_all = take.pause_all
        self.paused_fds = take.paused_fds^
        self.released = take.released^
        self.pause_changed = take.pause_changed
        self.ticks_per_ms = take.ticks_per_ms

    def on_accept(mut self, fd: Int32):
        var f = Int(fd)
        if f < 0 or f >= CLIENT_MAX_FDS:
            return
        self._forget(f)
        self.ids[f] = external_call["pion_next_client_id", UInt64]()
        self.ctime_ms[f] = external_call["pion_unix_ms", Int64]()
        self.atime[f] = external_call["pion_ticks", UInt64]()

    def on_close(mut self, fd: Int32):
        var f = Int(fd)
        if f < 0 or f >= CLIENT_MAX_FDS:
            return
        self._forget(f)
        self.ids[f] = 0

    def _forget(mut self, f: Int):
        """Whatever the fd's previous connection left behind."""
        if self.reply[f] != REPLY_ON:
            self.reply_off_count -= 1
        self.reply[f] = REPLY_ON
        self.no_touch[f] = 0
        self.no_evict[f] = 0
        self.close_after[f] = 0
        if self.postponed[f] != 0:
            self.postponed[f] = 0
            for k in range(len(self.paused_fds)):
                if Int(self.paused_fds[k]) == f:
                    _ = self.paused_fds.pop(k)
                    break
        for k in range(len(self.released)):
            if Int(self.released[k]) == f:
                _ = self.released.pop(k)
                break
        try:
            if f in self.lib_name:
                _ = self.lib_name.pop(f)
            if f in self.lib_ver:
                _ = self.lib_ver.pop(f)
        except:
            pass

    @always_inline
    def touch(mut self, fd: Int32):
        """Input arrived: CLIENT LIST's idle restarts. One counter read per
        receive buffer."""
        var f = Int(fd)
        if f >= 0 and f < CLIENT_MAX_FDS:
            self.atime[f] = external_call["pion_ticks", UInt64]()

    def id_of(self, fd: Int32) -> UInt64:
        var f = Int(fd)
        if f < 0 or f >= CLIENT_MAX_FDS:
            return 0
        return self.ids[f]

    def fd_of(self, id: UInt64) -> Int32:
        if id == 0:
            return -1
        for f in range(CLIENT_MAX_FDS):
            if self.ids[f] == id:
                return Int32(f)
        return -1

    def idle_s(self, fd: Int32) -> Int64:
        var f = Int(fd)
        if f < 0 or f >= CLIENT_MAX_FDS or self.ticks_per_ms <= 0:
            return 0
        var now = external_call["pion_ticks", UInt64]()
        if now <= self.atime[f]:
            return 0
        return Int64(Float64(now - self.atime[f]) / self.ticks_per_ms) // 1000

    def age_s(self, fd: Int32) -> Int64:
        var f = Int(fd)
        if f < 0 or f >= CLIENT_MAX_FDS:
            return 0
        return (external_call["pion_unix_ms", Int64]() - self.ctime_ms[f]) // 1000

    def set_reply(mut self, fd: Int32, mode: UInt8):
        var f = Int(fd)
        if f < 0 or f >= CLIENT_MAX_FDS:
            return
        var was_on = self.reply[f] == REPLY_ON
        self.reply[f] = mode
        var now_on = mode == REPLY_ON
        if was_on and not now_on:
            self.reply_off_count += 1
        elif not was_on and now_on:
            self.reply_off_count -= 1

    def pause(mut self, until_ms: Int64, all: Bool):
        """CLIENT PAUSE: the mode is the latest one asked for, the end the
        later of the two, as Redis's pauseActions keeps them."""
        self.pause_all = all
        if until_ms > self.pause_until_ms:
            self.pause_until_ms = until_ms
        self.pause_changed = True

    def unpause(mut self):
        self.pause_until_ms = 0
        self.pause_changed = True

    def release(mut self, fd: Int32):
        """Let one held client go now (CLIENT KILL of it)."""
        var f = Int(fd)
        if f < 0 or f >= CLIENT_MAX_FDS or self.postponed[f] == 0:
            return
        self.postponed[f] = 0
        for k in range(len(self.paused_fds)):
            if self.paused_fds[k] == fd:
                _ = self.paused_fds.pop(k)
                break
        self.released.append(fd)

    def hold(mut self, fd: Int32):
        """The pause holds fd's next command."""
        var f = Int(fd)
        if f >= 0 and f < CLIENT_MAX_FDS and self.postponed[f] == 0:
            self.postponed[f] = 1
            self.paused_fds.append(fd)

    @always_inline
    def is_postponed(self, fd: Int32) -> Bool:
        var f = Int(fd)
        return f >= 0 and f < CLIENT_MAX_FDS and self.postponed[f] != 0


def peer_addr(fd: Int32) -> String:
    """ip:port of the connection's peer ("" without one)."""
    var buf = alloc[UInt8](64)
    var n = Int(external_call["pion_peer_id", Int64](fd, buf, Int64(64)))
    var s = String(StringSpan[MutUntrackedOrigin](
        unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=buf, length=n))) if n > 0 else String("")
    buf.free()
    return s


def local_addr(fd: Int32) -> String:
    """ip:port of this end of the connection ("" without one)."""
    var buf = alloc[UInt8](64)
    var n = Int(external_call["pion_local_id", Int64](fd, buf, Int64(64)))
    var s = String(StringSpan[MutUntrackedOrigin](
        unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=buf, length=n))) if n > 0 else String("")
    buf.free()
    return s


struct ClientView(Copyable, Movable):
    """What the slow path knows about one connection, for its CLIENT LIST line."""
    var fd: Int32
    var name: String
    var flags: String
    var sub: Int
    var psub: Int
    var ssub: Int
    var multi: Int
    var watch: Int
    var qbuf: Int
    var qbuf_free: Int
    var events: String
    var user: String
    var resp: Int
    var io_thread: Int

    def __init__(out self, fd: Int32, var name: String, var flags: String, sub: Int, psub: Int, ssub: Int,
                 multi: Int, watch: Int, qbuf: Int, qbuf_free: Int, var events: String, var user: String,
                 resp: Int, io_thread: Int):
        self.fd = fd
        self.name = name^
        self.flags = flags^
        self.sub = sub
        self.psub = psub
        self.ssub = ssub
        self.multi = multi
        self.watch = watch
        self.qbuf = qbuf
        self.qbuf_free = qbuf_free
        self.events = events^
        self.user = user^
        self.resp = resp
        self.io_thread = io_thread


def client_line(reg: ClientRegistry, v: ClientView) raises -> String:
    """One CLIENT LIST / CLIENT INFO line: Redis's fields in Redis's order,
    without those Pion does not measure."""
    var f = Int(v.fd)
    var ln = reg.lib_name[f] if f in reg.lib_name else String("")
    var lv = reg.lib_ver[f] if f in reg.lib_ver else String("")
    return (String("id=") + String(reg.id_of(v.fd)) + " addr=" + peer_addr(v.fd) + " laddr=" + local_addr(v.fd)
            + " fd=" + String(f) + " name=" + v.name + " age=" + String(reg.age_s(v.fd))
            + " idle=" + String(reg.idle_s(v.fd)) + " flags=" + v.flags + " db=0 sub=" + String(v.sub)
            + " psub=" + String(v.psub) + " ssub=" + String(v.ssub) + " multi=" + String(v.multi)
            + " watch=" + String(v.watch) + " qbuf=" + String(v.qbuf) + " qbuf-free=" + String(v.qbuf_free)
            + " events=" + v.events + " user=" + v.user + " redir=-1 resp=" + String(v.resp)
            + " lib-name=" + ln + " lib-ver=" + lv + " io-thread=" + String(v.io_thread) + "\n")


def client_type_of(name_ptr: Pointer[UInt8, MutUntrackedOrigin], name_len: Int) -> Int:
    """Redis's getClientTypeByName: 0 normal, 1 replica (or slave), 2 pubsub,
    3 master, -1 none of them."""
    if _ci_eq(name_ptr, name_len, "normal"):
        return 0
    if _ci_eq(name_ptr, name_len, "replica") or _ci_eq(name_ptr, name_len, "slave"):
        return 1
    if _ci_eq(name_ptr, name_len, "pubsub"):
        return 2
    if _ci_eq(name_ptr, name_len, "master"):
        return 3
    return -1


def _ci_eq(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, lit: StaticString) -> Bool:
    if n != lit.byte_length():
        return False
    var lp = lit.unsafe_ptr()
    for k in range(n):
        var c = p[k]
        if c >= 65 and c <= 90:
            c += 32
        if c != lp[k]:
            return False
    return True


def heapsort_u64(mut v: List[UInt64]):
    """Sort in place (CLIENT LIST's connection order, by ID)."""
    var n = len(v)
    var start = n // 2 - 1
    while start >= 0:
        _sift(v, start, n)
        start -= 1
    var end = n - 1
    while end > 0:
        var t = v[0]
        v[0] = v[end]
        v[end] = t
        _sift(v, 0, end)
        end -= 1


def _sift(mut v: List[UInt64], root0: Int, n: Int):
    var root = root0
    while True:
        var child = 2 * root + 1
        if child >= n:
            return
        if child + 1 < n and v[child] < v[child + 1]:
            child += 1
        if v[root] >= v[child]:
            return
        var t = v[root]
        v[root] = v[child]
        v[child] = t
        root = child


def lower_bytes(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> String:
    """The bytes lowercased (ASCII), as a String."""
    var b = alloc[UInt8](n + 1)
    for k in range(n):
        var c = p[k]
        b[k] = c + 32 if c >= 65 and c <= 90 else c
    var s = String(StringSpan[MutUntrackedOrigin](
        unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=b, length=n))) if n > 0 else String("")
    b.free()
    return s


def client_help(mut writer: ResponseWriter):
    """CLIENT HELP, Redis's text."""
    var lines = List[String]()
    lines.append("CLIENT <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
    lines.append("CACHING (YES|NO)")
    lines.append("    Enable/disable tracking of the keys for next command in OPTIN/OPTOUT modes.")
    lines.append("GETREDIR")
    lines.append("    Return the client ID we are redirecting to when tracking is enabled.")
    lines.append("GETNAME")
    lines.append("    Return the name of the current connection.")
    lines.append("ID")
    lines.append("    Return the ID of the current connection.")
    lines.append("INFO")
    lines.append("    Return information about the current client connection.")
    lines.append("KILL <ip:port>")
    lines.append("    Kill connection made from <ip:port>.")
    lines.append("KILL <option> <value> [<option> <value> [...]]")
    lines.append("    Kill connections. Options are:")
    lines.append("    * ADDR (<ip:port>|<unixsocket>:0)")
    lines.append("      Kill connections made from the specified address")
    lines.append("    * LADDR (<ip:port>|<unixsocket>:0)")
    lines.append("      Kill connections made to specified local address")
    lines.append("    * TYPE (NORMAL|MASTER|REPLICA|PUBSUB)")
    lines.append("      Kill connections by type.")
    lines.append("    * USER <username>")
    lines.append("      Kill connections authenticated by <username>.")
    lines.append("    * SKIPME (YES|NO)")
    lines.append("      Skip killing current connection (default: yes).")
    lines.append("    * ID <client-id>")
    lines.append("      Kill connections by client id.")
    lines.append("    * MAXAGE <maxage>")
    lines.append("      Kill connections older than the specified age.")
    lines.append("LIST [options ...]")
    lines.append("    Return information about client connections. Options:")
    lines.append("    * TYPE (NORMAL|MASTER|REPLICA|PUBSUB)")
    lines.append("      Return clients of specified type.")
    lines.append("UNPAUSE")
    lines.append("    Stop the current client pause, resuming traffic.")
    lines.append("PAUSE <timeout> [WRITE|ALL]")
    lines.append("    Suspend all, or just write, clients for <timeout> milliseconds.")
    lines.append("REPLY (ON|OFF|SKIP)")
    lines.append("    Control the replies sent to the current connection.")
    lines.append("SETNAME <name>")
    lines.append("    Assign the name <name> to the current connection.")
    lines.append("SETINFO <option> <value>")
    lines.append("    Set client meta attr. Options are:")
    lines.append("    * LIB-NAME: the client lib name.")
    lines.append("    * LIB-VER: the client lib version.")
    lines.append("UNBLOCK <clientid> [TIMEOUT|ERROR]")
    lines.append("    Unblock the specified blocked client.")
    lines.append("TRACKING (ON|OFF) [REDIRECT <id>] [BCAST] [PREFIX <prefix> [...]]")
    lines.append("         [OPTIN] [OPTOUT] [NOLOOP]")
    lines.append("    Control server assisted client side caching.")
    lines.append("TRACKINGINFO")
    lines.append("    Report tracking status for the current connection.")
    lines.append("NO-EVICT (ON|OFF)")
    lines.append("    Protect current client connection from eviction.")
    lines.append("NO-TOUCH (ON|OFF)")
    lines.append("    Will not touch LRU/LFU stats when this mode is on.")
    lines.append("HELP")
    lines.append("    Print this help.")
    writer.append_array_header(len(lines))
    for k in range(len(lines)):
        writer.append_status_response(lines[k])
