# Mojo 1.0.0 (ed45d567). Same result at -O0 and -O3.
# Expected 199 on every line. `UInt64(p[0].cast[DType.uint8]())` and the
# `.cast[uint8]().cast[uint64]()` chain print 18446744073709551559 (sign
# extension of -57): the int8 -> uint8 -> uint64 chain folds to one sext.
from std.memory import alloc
def main():
    var p = alloc[Int8](4)
    p[0] = Int8(-57)
    var b: UInt8 = 199
    print("UInt64(UInt8 var)", UInt64(b))
    print("UInt64(p[0].cast[uint8])", UInt64(p[0].cast[DType.uint8]()))
    var c = p[0].cast[DType.uint8]()
    print("UInt64(c)", UInt64(c), "c=", c)
    print("p[0].cast[uint8].cast[uint64]", p[0].cast[DType.uint8]().cast[DType.uint64]())
    print("UInt32(p[0].cast[uint8])", UInt32(p[0].cast[DType.uint8]()))
    print("Int(p[0].cast[uint8])", Int(p[0].cast[DType.uint8]()))
