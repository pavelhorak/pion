from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.sys import size_of

struct Vector:
    var data: Pointer[Float32, MutUntrackedOrigin]
    var dim: Int

    def __init__(out self, dim: Int):
        self.dim = dim
        self.data = alloc[Float32](dim)

    def __init__(out self, dim: Int, values: List[Float32]):
        self.dim = dim
        self.data = alloc[Float32](dim)
        for i in range(len(values)):
            if i < dim:
                self.data[i] = values[i]

    def __getitem__(self, i: Int) -> Float32:
        return self.data[i]

    def __setitem__(mut self, i: Int, value: Float32):
        self.data[i] = value

    def deinit(owned self):
        self.data.unsafe_free()
