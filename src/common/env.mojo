
from std.ffi import external_call
from std.memory import alloc
from std.sys import CompilationTarget

struct Environment:
    var is_embedded: Bool
    var is_cloud: Bool
    var has_gpu: Bool

    def __init__(out self):
        self.is_embedded = False
        self.is_cloud = False
        self.has_gpu = False  # GPU distance path is a stub; disabled until fully implemented

        var num_cores = online_cpu_count()
        if num_cores <= 4:
            self.is_embedded = True
            self.is_cloud = False
        elif num_cores >= 16:
            self.is_cloud = True
            self.is_embedded = False


def online_cpu_count() -> Int:
    """`sysconf(_SC_NPROCESSORS_ONLN)`. The constant differs per C library:
    58 on macOS and 84 on glibc and musl, where 58 is `_SC_POLL` and returns 1.
    Passing macOS's 58 everywhere made every Linux machine report one core,
    so the smart profile picked `embedded` on a 32-thread EPYC (#20)."""
    var name = Int32(58)
    comptime if CompilationTarget.is_linux():
        name = Int32(84)
    return Int(external_call["sysconf", Int64](name))
