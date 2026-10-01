from std.ffi import external_call
from std.memory import alloc

struct Environment:
    var is_embedded: Bool
    var is_cloud: Bool
    var has_gpu: Bool

    def __init__(out self):
        self.is_embedded = False
        self.is_cloud = False
        self.has_gpu = False  # GPU distance path is a stub; disabled until fully implemented

        var num_cores = Int(external_call["sysconf", Int64](58)) # _SC_NPROCESSORS_ONLN = 58 on Linux/macOS
        if num_cores <= 4:
            self.is_embedded = True
            self.is_cloud = False
        elif num_cores >= 16:
            self.is_cloud = True
            self.is_embedded = False
