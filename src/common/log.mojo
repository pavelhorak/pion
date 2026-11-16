"""Where the engine's log lines go.

The server prints to stdout. A library (src/lib/pion_embed.mojo) must not write
to its host's stdout, so it points `pion_log_fd` (src/ffi/fcntl_wrap.c) at a
file in its data directory instead. Found by dlsym, not linked: a Mojo test
that compiles an engine module without the C shims still prints to stdout.
"""
from std.io import RawFileDescriptor
from std.ffi import external_call
from std.memory import Pointer
from std.sys import CompilationTarget


@no_inline
def log_out() -> RawFileDescriptor:
    """The fd for one log line: `pion_log_fd` if the process has it, else 1."""
    var handle: Int
    comptime if CompilationTarget.is_macos():
        handle = -2                    # RTLD_DEFAULT
    else:
        handle = 0                     # RTLD_DEFAULT
    var p = external_call["dlsym", Int](handle, "pion_log_fd".ptr())
    if p == 0:
        return RawFileDescriptor(Int32(1))
    return RawFileDescriptor(Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=p).unsafe_load())
