from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memset, unsafe_memcpy

@fieldwise_init
struct SetBitResult:
    var ptr: Pointer[UInt8, MutUntrackedOrigin]
    var len: Int

@always_inline
def _bit_in_byte(bit_offset: Int) -> Int:
    """gh #232: Redis numbers bits from the MOST significant end of each byte.

    `SETBIT k 7 1` yields the string `\\x01` and `SETBIT k 0 1` yields `\\x80`.
    Pion used `bit_offset % 8` directly, i.e. bit 0 = LSB, so it produced
    exactly the reverse of both.

    That was invisible from inside: SETBIT/GETBIT/BITCOUNT agreed with each
    other, and BITCOUNT is a popcount so it cannot see the order at all. It
    only shows up the moment the bytes leave the bitmap commands — a `GET` on
    the key, a snapshot read by another tool, or any client that builds a
    bitmap with SETBIT and parses it as a string (the standard Bloom-filter and
    presence-bitmap idiom).

    Fixed at the primitive, so BITFIELD and every other caller inherit it
    rather than each needing its own flip."""
    return 7 - (bit_offset % 8)

def getbit(bitmap: Pointer[UInt8, MutUntrackedOrigin], bit_offset: Int) -> Int:
    var byte_index = bit_offset // 8
    var bit_index = _bit_in_byte(bit_offset)
    var byte = bitmap.load(byte_index)
    return Int((byte >> UInt8(bit_index)) & 1)

def setbit(byte_len_val: Int, bitmap: Pointer[UInt8, MutUntrackedOrigin], bit_offset: Int, value: Int) -> SetBitResult:
    var byte_len = byte_len_val
    var needed_bytes = bit_offset // 8 + 1
    var current_ptr = bitmap
    if needed_bytes > byte_len:
        var old_len = byte_len
        var new_ptr = alloc[UInt8](needed_bytes)
        unsafe_memcpy(dest=new_ptr, src=bitmap, count=old_len)
        unsafe_memset(new_ptr.unsafe_offset(old_len), 0, needed_bytes - old_len)
        bitmap.unsafe_free()
        current_ptr = new_ptr
        byte_len = needed_bytes

    var byte_index = bit_offset // 8
    var bit_index = _bit_in_byte(bit_offset)
    var byte = current_ptr.load(byte_index)

    if value == 1:
        byte |= UInt8(1 << bit_index)
    else:
        byte &= ~UInt8(1 << bit_index)
    
    current_ptr.store(byte_index, byte)
    return SetBitResult(current_ptr, byte_len)

def bitcount(bitmap: Pointer[UInt8, MutUntrackedOrigin], byte_len: Int) -> Int:
    var count = 0
    for i in range(byte_len):
        var byte = bitmap.load(i)
        while byte > 0:
            byte &= (byte - 1)
            count += 1
    return count
