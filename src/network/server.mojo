from src.common.ptr import null_ptr
from std.ffi import external_call
from std.sys import CompilationTarget
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memset, unsafe_memcpy, stack_allocation


@always_inline
def _last_errno() -> Int32:
    comptime if CompilationTarget.is_linux():
        return external_call["__errno_location", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]
    else:
        return external_call["__error", Pointer[Int32, MutUntrackedOrigin]]()[unsafe_offset=0]


@always_inline
def _EADDRINUSE() -> Int32:
    comptime if CompilationTarget.is_linux():
        return 98
    else:
        return 48


def create_listen_socket(port: Int) -> Int32:
    """Create a single shared listen socket. No SO_REUSEPORT — one socket, all workers compete to accept()."""
    var socket_fd = external_call["socket", Int32](Int32(2), Int32(1), Int32(0))  # AF_INET=2, SOCK_STREAM=1
    if socket_fd < 0:
        print("Failed to create shared listen socket")
        return -1

    var opt_ptr = alloc[Int32](1)
    opt_ptr[unsafe_offset=0] = 1
    comptime if CompilationTarget.is_linux():
        _ = external_call["setsockopt", Int32](socket_fd, Int32(1), Int32(2), opt_ptr, Int32(4))      # SOL_SOCKET=1, SO_REUSEADDR=2
    else:
        _ = external_call["setsockopt", Int32](socket_fd, Int32(0xffff), Int32(0x0004), opt_ptr, Int32(4))  # SO_REUSEADDR (macOS)
        _ = external_call["setsockopt", Int32](socket_fd, Int32(0xffff), Int32(0x1022), opt_ptr, Int32(4))  # SO_NOSIGPIPE (macOS)
    opt_ptr.unsafe_free()

    # Set nonblocking using C wrapper (avoids variadic FFI bug on macOS arm64)
    _ = external_call["set_nonblock_c", Int32](socket_fd)

    var addr = alloc[UInt8](16)
    unsafe_memset(addr, 0, 16)
    comptime if CompilationTarget.is_linux():
        # Linux sockaddr_in: no sin_len byte; sin_family is uint16 at offset 0 (little-endian)
        addr[unsafe_offset=0] = 2; addr[unsafe_offset=1] = 0  # AF_INET = 2 as uint16 LE
    else:
        addr[unsafe_offset=0] = 16  # sin_len (BSD/macOS)
        addr[unsafe_offset=1] = 2   # AF_INET
    var p = UInt16(port)
    addr[unsafe_offset=2] = UInt8(p >> 8)
    addr[unsafe_offset=3] = UInt8(p & 0xFF)
    # gh #258: sin_addr. Bytes 4-7 used to be left at zero from the memset,
    # i.e. INADDR_ANY on every listener. The value is already in network byte
    # order, so it is copied out little-end-first.
    var _ba = external_call["pion_get_bind_addr", UInt32]()
    addr[unsafe_offset=4] = UInt8(_ba & 0xFF)
    addr[unsafe_offset=5] = UInt8((_ba >> 8) & 0xFF)
    addr[unsafe_offset=6] = UInt8((_ba >> 16) & 0xFF)
    addr[unsafe_offset=7] = UInt8((_ba >> 24) & 0xFF)

    var res = external_call["bind", Int32](socket_fd, addr, Int32(16))
    # #22: a server that just stopped can still hold the port for a moment. On
    # Linux the kernel tears an io_uring instance down AFTER the process exits,
    # and a pending ACCEPT keeps the listening socket open until it has (a
    # SIGKILLed server never gets to cancel its accepts). A restart used to
    # lose that race and exit with "cannot bind port". Wait for the port, up
    # to 3 s, before giving up.
    var tries = 0
    while res < 0 and tries < 30 and _last_errno() == _EADDRINUSE():
        if tries == 0:
            print("Port " + String(port) + " is still held (a server that just stopped "
                  + "can keep it briefly); retrying for up to 3 s")
        _ = external_call["usleep", Int32](Int32(100000))
        res = external_call["bind", Int32](socket_fd, addr, Int32(16))
        tries += 1
    if res < 0:
        print("Failed to bind shared listen socket to port " + String(port))
        addr.unsafe_free()
        _ = external_call["close", Int32](socket_fd)
        return -1

    res = external_call["listen", Int32](socket_fd, Int32(65535))
    if res < 0:
        print("Failed to listen on shared socket")
        addr.unsafe_free()
        _ = external_call["close", Int32](socket_fd)
        return -1

    addr.unsafe_free()
    print("Shared listen socket: port=" + String(port) + " fd=" + String(socket_fd))
    return socket_fd

comptime EV_CLEAR = 0x0020

# epoll constants (Linux)
comptime EPOLLIN = UInt32(0x001)
comptime EPOLLOUT = UInt32(0x004)
comptime EPOLLERR = UInt32(0x008)
comptime EPOLLHUP = UInt32(0x010)
comptime EPOLLET = UInt32(0x80000000)
comptime EPOLLEXCLUSIVE = UInt32(1 << 28)  # only one epoll wakes per event (Linux 4.5+)
comptime EPOLL_CTL_ADD = Int32(1)
comptime EPOLL_CTL_DEL = Int32(2)
comptime EPOLL_CTL_MOD = Int32(3)

@fieldwise_init
struct EpollEvent(Copyable, Movable):
    """Linux epoll_event: 12 bytes packed (events:u32 + data:u64)."""
    var events: UInt32
    var data: UInt64    # union — low 32 bits = fd

    def __init__(out self):
        self.events = 0
        self.data = 0

@fieldwise_init
struct KEvent(Copyable, Movable):
    var ident: UInt64
    var filter: Int16
    var flags: UInt16
    var fflags: UInt32
    var data: Int64
    var udata: Pointer[NoneType, MutUntrackedOrigin]

    def __init__(out self):
        self.ident = 0
        self.filter = 0
        self.flags = 0
        self.fflags = 0
        self.data = 0
        self.udata = null_ptr[NoneType, MutUntrackedOrigin]()

@fieldwise_init
struct PollFd(Copyable, Movable):
    var fd: Int32
    var events: Int16
    var revents: Int16

    def __init__(out self, fd: Int32, events: Int16):
        self.fd = fd
        self.events = events
        self.revents = 0

@fieldwise_init
struct TCPServer(Copyable, Movable):
    var port: Int
    var fd: Int32

    def __init__(out self, port: Int):
        self.port = port
        self.fd = -1

    def listen(mut self) -> Bool:
        var socket_fd = external_call["socket", Int32](Int32(2), Int32(1), Int32(0)) # AF_INET=2, SOCK_STREAM=1
        if socket_fd < 0:
            print("Failed to create socket")
            return False

        var opt_ptr = alloc[Int32](1)
        opt_ptr[unsafe_offset=0] = 1
        comptime if CompilationTarget.is_linux():
            # SOL_SOCKET=1, SO_REUSEADDR=2, SO_REUSEPORT=15
            _ = external_call["setsockopt", Int32](socket_fd, Int32(1), Int32(2), opt_ptr, Int32(4))
            _ = external_call["setsockopt", Int32](socket_fd, Int32(1), Int32(15), opt_ptr, Int32(4))
        else:
            _ = external_call["setsockopt", Int32](socket_fd, Int32(0xffff), Int32(0x0004), opt_ptr, Int32(4))  # SO_REUSEADDR
            _ = external_call["setsockopt", Int32](socket_fd, Int32(0xffff), Int32(0x0200), opt_ptr, Int32(4))  # SO_REUSEPORT
            _ = external_call["setsockopt", Int32](socket_fd, Int32(0xffff), Int32(0x1022), opt_ptr, Int32(4))  # SO_NOSIGPIPE
        # IPPROTO_TCP = 6, TCP_NODELAY = 1 (same on Linux and macOS)
        _ = external_call["setsockopt", Int32](socket_fd, Int32(6), Int32(1), opt_ptr, Int32(4))
        opt_ptr.unsafe_free()

        self.set_nonblocking(socket_fd)

        var addr = alloc[UInt8](16)
        unsafe_memset(addr, 0, 16)
        comptime if CompilationTarget.is_linux():
            addr[unsafe_offset=0] = 2; addr[unsafe_offset=1] = 0  # sin_family = AF_INET as uint16 LE (no sin_len on Linux)
        else:
            addr[unsafe_offset=0] = 16  # sin_len (BSD/macOS)
            addr[unsafe_offset=1] = 2   # sin_family = AF_INET
        var p = UInt16(self.port)
        addr[unsafe_offset=2] = UInt8(p >> 8)
        addr[unsafe_offset=3] = UInt8(p & 0xFF)

        # gh #258: sin_addr. This comment used to read "no need to set addr[4-7]
        # as they are already 0 from memset" — which is exactly the bug: zero is
        # INADDR_ANY, so every listener was on all interfaces by omission.
        var _ba = external_call["pion_get_bind_addr", UInt32]()
        addr[unsafe_offset=4] = UInt8(_ba & 0xFF)
        addr[unsafe_offset=5] = UInt8((_ba >> 8) & 0xFF)
        addr[unsafe_offset=6] = UInt8((_ba >> 16) & 0xFF)
        addr[unsafe_offset=7] = UInt8((_ba >> 24) & 0xFF)

        var res = external_call["bind", Int32](socket_fd, addr, Int32(16))
        if res < 0:
            print("Failed to bind to port " + String(self.port))
            addr.unsafe_free()
            return False

        res = external_call["listen", Int32](socket_fd, Int32(65535))
        if res < 0:
            print("Failed to listen")
            addr.unsafe_free()
            return False
            
        self.fd = socket_fd
        addr.unsafe_free()
        print("Listening on port " + String(self.port))
        return True

    def set_nonblocking(self, fd: Int32):
        _ = external_call["set_nonblock_c", Int32](fd)

    def set_tcp_nodelay(self, fd: Int32):
        # IPPROTO_TCP = 6, TCP_NODELAY = 0x0001
        var opt = alloc[Int32](1)
        opt[unsafe_offset=0] = 1
        _ = external_call["setsockopt", Int32](fd, Int32(6), Int32(0x0001), opt, Int32(4))
        opt.unsafe_free()

    def accept(self) -> Int32:
        var null_addr = null_ptr[NoneType, MutUntrackedOrigin]()
        var null_len = null_ptr[NoneType, MutUntrackedOrigin]()
        return external_call["accept", Int32](self.fd, null_addr, null_len)

    def accept_from(self, listen_fd: Int32) -> Int32:
        var null_addr = null_ptr[NoneType, MutUntrackedOrigin]()
        var null_len = null_ptr[NoneType, MutUntrackedOrigin]()
        return external_call["accept", Int32](listen_fd, null_addr, null_len)

    def recv(self, client_fd: Int32, buffer: Pointer[UInt8, MutUntrackedOrigin], size: Int) -> Int:
        return Int(external_call["recv", Int64](client_fd, buffer, size, Int32(0)))

    def send[origin: Origin](self, client_fd: Int32, buffer: Pointer[UInt8, origin], size: Int) -> Int:
        # MSG_NOSIGNAL (0x4000) on Linux: prevent SIGPIPE when remote end closed.
        # Without this, send() to a dead fd kills the thread/process via SIGPIPE.
        # On macOS, SO_NOSIGPIPE is set on the socket instead (no per-send flag needed).
        comptime if CompilationTarget.is_linux():
            return Int(external_call["send", Int64](client_fd, buffer, size, Int32(0x4000)))
        else:
            return Int(external_call["send", Int64](client_fd, buffer, size, Int32(0)))

    def close_client(self, client_fd: Int32):
        _ = external_call["close", Int32](client_fd)

    def close(mut self):
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)
            self.fd = -1

    def kqueue(self) -> Int32:
        comptime if CompilationTarget.is_linux():
            return -1  # io_uring used on Linux — no kqueue
        else:
            return external_call["kqueue", Int32]()

    def kevent_add_read(self, kq: Int32, fd: Int32, edge_triggered: Bool = False):
        comptime if not CompilationTarget.is_linux():
            var ev = stack_allocation[1, KEvent]()
            var flags = UInt16(0x0001 | 0x0004) # EV_ADD | EV_ENABLE
            if edge_triggered:
                flags |= EV_CLEAR
            ev[unsafe_offset=0] = KEvent(UInt64(fd), Int16(-1), flags, UInt32(0), Int64(0), null_ptr[NoneType, MutUntrackedOrigin]())
            var null_kevent = null_ptr[KEvent, MutUntrackedOrigin]()
            var null_timeout = null_ptr[NoneType, MutUntrackedOrigin]()
            _ = external_call["kevent", Int32](kq, ev, Int32(1), null_kevent, Int32(0), null_timeout)

    def kevent_add_write(self, kq: Int32, fd: Int32):
        comptime if CompilationTarget.is_linux():
            # epoll: modify to add EPOLLOUT (kq is epoll fd; -1 means io_uring path — skip)
            if kq >= 0:
                var ev = stack_allocation[1, EpollEvent]()
                ev[unsafe_offset=0].events = EPOLLIN | EPOLLOUT
                ev[unsafe_offset=0].data = UInt64(fd)
                _ = external_call["epoll_ctl", Int32](kq, EPOLL_CTL_MOD, fd, ev)
        else:
            var ev = stack_allocation[1, KEvent]()
            ev[unsafe_offset=0] = KEvent(UInt64(fd), Int16(-2), UInt16(0x0001 | 0x0004), UInt32(0), Int64(0), null_ptr[NoneType, MutUntrackedOrigin]())
            var null_kevent = null_ptr[KEvent, MutUntrackedOrigin]()
            var null_timeout = null_ptr[NoneType, MutUntrackedOrigin]()
            _ = external_call["kevent", Int32](kq, ev, Int32(1), null_kevent, Int32(0), null_timeout)

    def kevent_del_write(self, kq: Int32, fd: Int32):
        comptime if CompilationTarget.is_linux():
            # epoll: modify to remove EPOLLOUT (kq is epoll fd; -1 means io_uring path — skip)
            if kq >= 0:
                var ev = stack_allocation[1, EpollEvent]()
                ev[unsafe_offset=0].events = EPOLLIN
                ev[unsafe_offset=0].data = UInt64(fd)
                _ = external_call["epoll_ctl", Int32](kq, EPOLL_CTL_MOD, fd, ev)
        else:
            var ev = stack_allocation[1, KEvent]()
            ev[unsafe_offset=0] = KEvent(UInt64(fd), Int16(-2), UInt16(0x0002), UInt32(0), Int64(0), null_ptr[NoneType, MutUntrackedOrigin]())
            var null_kevent = null_ptr[KEvent, MutUntrackedOrigin]()
            var null_timeout = null_ptr[NoneType, MutUntrackedOrigin]()
            _ = external_call["kevent", Int32](kq, ev, Int32(1), null_kevent, Int32(0), null_timeout)

    def kevent_wait(self, kq: Int32, events: Pointer[KEvent, MutUntrackedOrigin], max_events: Int32, timeout: Pointer[Int, MutUntrackedOrigin] = null_ptr[Int, MutUntrackedOrigin]()) -> Int32:
        comptime if CompilationTarget.is_linux():
            return 0  # no-op on Linux
        else:
            var null_kevent = null_ptr[KEvent, MutUntrackedOrigin]()
            return external_call["kevent", Int32](kq, null_kevent, Int32(0), events, max_events, timeout)

    def kevent_batch(self, kq: Int32, changelist: Pointer[KEvent, MutUntrackedOrigin], nchanges: Int32, eventlist: Pointer[KEvent, MutUntrackedOrigin], nevents: Int32, timeout: Pointer[Int, MutUntrackedOrigin] = null_ptr[Int, MutUntrackedOrigin]()) -> Int32:
        comptime if CompilationTarget.is_linux():
            return 0  # no-op on Linux
        else:
            return external_call["kevent", Int32](kq, changelist, nchanges, eventlist, nevents, timeout)

@fieldwise_init
struct UDPSocket(Copyable, Movable):
    var fd: Int32
    var port: Int

    def __init__(out self, port: Int):
        self.port = port
        self.fd = external_call["socket", Int32](Int32(2), Int32(2), Int32(0)) # AF_INET=2, SOCK_DGRAM=2

    def bind(mut self) -> Bool:
        var addr = alloc[UInt8](16)
        unsafe_memset(addr, 0, 16)
        addr[0] = 16
        addr[1] = 2 # AF_INET

        var p = UInt16(self.port)
        # htons: swap bytes for 16-bit (big-endian)
        addr[2] = UInt8(p >> 8)
        addr[3] = UInt8(p & 0xFF)
        # gh #258: gossip/Raft listened on all interfaces too.
        var _ba = external_call["pion_get_bind_addr", UInt32]()
        addr[4] = UInt8(_ba & 0xFF)
        addr[5] = UInt8((_ba >> 8) & 0xFF)
        addr[6] = UInt8((_ba >> 16) & 0xFF)
        addr[7] = UInt8((_ba >> 24) & 0xFF)

        var res = external_call["bind", Int32](self.fd, addr, Int32(16))
        addr.unsafe_free()
        return res == 0

    def sendto(self, buffer: Pointer[UInt8, MutUntrackedOrigin], length: Int, dest_ip: String, dest_port: Int) -> Int:
        # Simplified sendto for prototype
        # In real system, we'd parse dest_ip
        return Int(external_call["send", Int64](self.fd, buffer, length, Int32(0)))

    def recvfrom(self, buffer: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Int:
        return Int(external_call["recv", Int64](self.fd, buffer, length, Int32(0)))

    def close(mut self):
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)
            self.fd = -1
