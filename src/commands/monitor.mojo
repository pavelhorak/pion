"""MONITOR (#39): stream every command the server runs to the clients that
asked for it, as Redis's replicationFeedMonitors does.

A line is `+<sec>.<usec> [0 <ip>:<port>] "cmd" "arg" ...`, each argument
quoted and escaped (src/ffi/fcntl_wrap.c, pion_monitor_line); a script's
commands show `[0 lua]`. The rules follow Redis's `call()`:
- a command is shown once it has run, so its own reply goes first when the
  monitor itself sent it, and EXEC comes after the commands it ran;
- EVAL, EVALSHA, FCALL and their _RO forms (flag `skip_monitor`) are shown
  before they run, so the commands a script calls follow its line;
- `admin` commands are never shown, nor a command refused before it ran (an
  unknown command, a wrong argument count, NOAUTH, OOM, one queued by MULTI);
- AUTH's arguments, and HELLO's or MIGRATE's credentials, read "(redacted)".

Pion's workers are independent (`-w 1` by default; `-w N` needs
--independent-workers), and a monitor sees the commands of the worker that
accepted its connection.

While any client monitors, every recv buffer goes to the slow path (the
engine's `fast_path_off`), which is where commands are fed. A monitor client
may not touch the keyspace ("Replica can't interact with the keyspace"), as in
Redis, where a monitor is a kind of replica.
"""

from std.ffi import external_call
from std.memory import alloc
from std.memory.unsafe_pointer import Pointer
from src.network.resp3 import RESP3Token
from src.common.utils import arg_eq


struct MonitorRegistry(Movable):
    """This worker's monitoring connections, and the lines of a running
    script, which go out once the script's own line has."""
    var fds: List[Int32]
    var pending: List[UInt8]

    def __init__(out self):
        self.fds = List[Int32]()
        self.pending = List[UInt8]()

    def __init__(out self, *, deinit take: Self):
        self.fds = take.fds^
        self.pending = take.pending^

    @always_inline
    def count(self) -> Int:
        return len(self.fds)

    def contains(self, fd: Int32) -> Bool:
        for k in range(len(self.fds)):
            if self.fds[k] == fd:
                return True
        return False

    def add(mut self, fd: Int32):
        if not self.contains(fd):
            self.fds.append(fd)

    def remove(mut self, fd: Int32) -> Bool:
        for k in range(len(self.fds)):
            if self.fds[k] == fd:
                _ = self.fds.pop(k)
                return True
        return False


def monitor_line(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, fd: Int32) -> List[UInt8]:
    """The MONITOR line for tokens[i:end], sent by `fd` (-1: a script)."""
    var argc = end - i
    var argv = alloc[Pointer[UInt8, MutUntrackedOrigin]](argc)
    var lens = alloc[Int64](argc)
    # a literal's bytes are static: the pointer outlives this frame
    var redp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int("(redacted)".unsafe_ptr()))
    var name = tokens[i]
    var is_auth = arg_eq(name.ptr, name.length, "auth")
    var is_hello = arg_eq(name.ptr, name.length, "hello")
    var is_migrate = arg_eq(name.ptr, name.length, "migrate")
    var redact_left = 0
    for k in range(argc):
        var t = tokens[i + k]
        var hide = (is_auth and k > 0) or redact_left > 0
        if redact_left > 0:
            redact_left -= 1
        elif k > 0 and is_hello and arg_eq(t.ptr, t.length, "auth"):
            redact_left = 2
        elif k > 0 and is_migrate and arg_eq(t.ptr, t.length, "auth"):
            redact_left = 1
        elif k > 0 and is_migrate and arg_eq(t.ptr, t.length, "auth2"):
            redact_left = 2
        if hide:
            argv[k] = redp
            lens[k] = Int64(10)
        else:
            argv[k] = t.ptr
            lens[k] = Int64(t.length)
    var out = alloc[Pointer[UInt8, MutUntrackedOrigin]](1)
    var n = external_call["pion_monitor_line", Int64](Int32(fd), Int64(argc), argv, lens, out)
    var line = List[UInt8]()
    var lp = out[0]
    if n > 0:
        line.reserve(Int(n))
        for b in range(Int(n)):
            line.append(lp[b])
    external_call["pion_lcs_free", NoneType](lp)
    out.unsafe_free()
    argv.unsafe_free()
    lens.unsafe_free()
    return line^
