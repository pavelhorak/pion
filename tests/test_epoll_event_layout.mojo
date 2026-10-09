"""epoll_wait results decode correctly when several events come back at once.

x86-64 Linux packs `struct epoll_event` to 12 bytes; aarch64 does not (16).
The epoll loop once read every event as a 16-byte Mojo struct, so on x86-64
only the first event of each batch decoded right, and the rest came from the
wrong bytes. Level-triggered epoll re-reported what was missed, so every
functional test still passed and only throughput showed it. This test makes
three fds readable at once, asks for all of them in ONE epoll_wait, and checks
that each decoded fd and mask is one it registered.

    pixi run mojo build -I . tests/test_epoll_event_layout.mojo -o build/t && build/t
"""
from std.ffi import external_call
from std.memory import alloc, stack_allocation
from std.memory.unsafe_pointer import Pointer
from std.sys import CompilationTarget
from src.network.server import EPOLLIN, EPOLL_CTL_ADD, EPOLL_EV_SIZE, EPOLL_EV_DATA
from src.network.server import epoll_ev_events, epoll_ev_fd, epoll_ctl_fd


def main() raises:
    comptime if CompilationTarget.is_linux():
        _run()
    else:
        print("SKIP: epoll is Linux-only")


def _run() raises:
    comptime if CompilationTarget.is_x86():
        if EPOLL_EV_SIZE != 12 or EPOLL_EV_DATA != 4:
            raise Error("x86-64 epoll_event must be 12 bytes with data at 4")
    else:
        if EPOLL_EV_SIZE != 16 or EPOLL_EV_DATA != 8:
            raise Error("aarch64 epoll_event must be 16 bytes with data at 8")

    var ep = external_call["epoll_create1", Int32](Int32(0))
    if ep < 0:
        raise Error("epoll_create1 failed")
    var fds = stack_allocation[6, Int32]()
    var want = List[Int32]()
    for k in range(3):
        if external_call["pipe", Int32](fds.unsafe_offset(2 * k)) != 0:
            raise Error("pipe failed")
        var r = fds[unsafe_offset=2 * k]
        var w = fds[unsafe_offset=2 * k + 1]
        var b = stack_allocation[1, UInt8]()
        b[] = 120
        _ = external_call["write", Int64](w, b, 1)
        if epoll_ctl_fd(ep, EPOLL_CTL_ADD, r, EPOLLIN) != 0:
            raise Error("epoll_ctl failed")
        want.append(r)

    var events = alloc[UInt8](16 * 16)
    var n = Int(external_call["epoll_wait", Int32](ep, events, Int32(16), Int32(1000)))
    if n != 3:
        raise Error("expected 3 ready events in one epoll_wait, got " + String(n))
    var seen = 0
    for i in range(n):
        var fd = epoll_ev_fd(events, i)
        var mask = epoll_ev_events(events, i)
        var found = -1
        for k in range(3):
            if want[k] == fd:
                found = k
        if found < 0:
            raise Error("event " + String(i) + " decoded fd " + String(fd) + ", not a registered fd")
        if (mask & EPOLLIN) == 0:
            raise Error("event " + String(i) + " decoded mask " + String(mask) + " without EPOLLIN")
        if (seen >> found) & 1:
            raise Error("fd " + String(fd) + " decoded twice")
        seen |= 1 << found
    print("PASS: 3 events in one epoll_wait decoded (event size " + String(EPOLL_EV_SIZE) + ")")
