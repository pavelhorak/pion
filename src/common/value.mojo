from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, stack_allocation, unsafe_memset
from std.memory.unsafe import bitcast
from std.collections import Span
from src.common.vector import Vector

@always_inline
def _low_byte_mask(nbytes: Int) -> UInt64:
    """Mask keeping the low `nbytes` bytes. nbytes in 0..8 (>=8 → all ones)."""
    if nbytes >= 8:
        return ~UInt64(0)
    # nbytes in 0..7 → shift 0..56, never 64 (no UB); nbytes==0 → (1<<0)-1 == 0.
    return (UInt64(1) << UInt64(nbytes * 8)) - 1

@always_inline
def _load_u64_le(ptr: Pointer[UInt8, MutUntrackedOrigin], avail: Int) -> UInt64:
    """Read an 8-byte little-endian word at `ptr` where only `avail` (1..) bytes are
    guaranteed in-bounds. The load is issued whole (a harmless in-bounds over-read that
    the caller masks off) whenever the word is fully in-bounds OR the 8-byte read cannot
    cross a 4 KiB boundary — a load fully inside one mapped page never faults. Only when a
    partial word sits within 8 bytes of a 4 KiB boundary do we fall back to a scalar
    byte assembly, so no read ever crosses into a possibly-unmapped page. 4 KiB is the
    smallest page on any supported target, so this is safe for 16 KiB (Apple Silicon) and
    64 KiB (some ARM) pages too. Needs no caller-side buffer padding.

    The partial-word load is VOLATILE. The hardware allows the over-read, but LLVM
    does not: reading past the end of an allocated object is UB, and where the
    optimizer can see the allocation (Mojo declares `alloc` as `allocsize`), it
    may fold the load to garbage. Under Mojo 1.1, `GenericValue.from_ptr` over a
    4-byte `alloc` came back with every character zeroed
    (tests/test_generic_value.mojo). A volatile load is emitted as written and
    never folded from what the optimizer knows about the object, and it is still
    one instruction."""
    if avail <= 0:
        return 0
    if avail >= 8:
        return ptr.unsafe_bitcast[UInt64]().load()
    if (Int(ptr) & 0xFFF) <= 0xFF8:
        return ptr.unsafe_bitcast[UInt64]().load[volatile=True]()
    var w: UInt64 = 0
    for i in range(avail):
        w |= UInt64(ptr[unsafe_offset=i]) << UInt64(i * 8)
    return w

# gh #163: marks a heap-STRING whose bytes live in the mmap'd blob arena, not on
# the heap. Stored in `_data2`, the word the heap-STRING layout leaves reserved —
# neither __eq__ nor __hash__ reads it on that branch, so the tag is free.
comptime BLOB_TAG = UInt64(0x424C4F4250494F4E)   # "NOIPBLOB" LE

# gh #394: marks a heap-STRING that BORROWS its bytes (the recv buffer, a RESP
# token) instead of owning a heap copy — see GenericValue.borrow(). Same word
# as BLOB_TAG, same reason the tag is free. A borrowed value is never freed,
# and every SlabHashMap store copies it (owned()) before keeping it, so one can
# be handed to get/set/remove but must never be stored anywhere else.
comptime BORROW_TAG = UInt64(0x574F52524F42504E)  # "NPBORROW" LE


@fieldwise_init
struct ValueType(Copyable, Movable, Defaultable, ImplicitlyCopyable, Hashable):
    var value: Int
    comptime NONE = 0
    comptime STRING = 1
    comptime HASH = 3
    comptime LIST = 4
    comptime SET = 5
    comptime ZSET = 6
    comptime INT = 7
    comptime FLOAT = 8
    comptime STRING_SSO = 9
    comptime BITMAP = 10
    comptime HLL = 11
    comptime GEO = 12
    comptime STREAM = 13
    comptime VSET = 14      # gh #366: a Redis 8 vector set (src/common/vector_set.mojo)

    def __init__(out self):
        self.value = Self.NONE

    def __eq__(self, other: ValueType) -> Bool:
        return self.value == other.value
    
    def __hash__(self) -> Int:
        return self.value

@fieldwise_init
struct GenericValue(Copyable, Movable, Defaultable, ImplicitlyCopyable, Hashable, Writable):
    var type: ValueType
    # Data layout (24 bytes):
    # STRING (Heap): [ptr (8), length (8), reserved (8)]
    # STRING_SSO:    [length (1), data (23)]
    # INT:           [int_val (8), reserved (16)]
    # FLOAT:         [float_val (8), reserved (16)]
    # BITMAP:        [ptr (8), length_in_bytes (8), reserved (8)]
    # HLL:           [ptr_to_registers (8), reserved (16)]
    # GEO:           [ptr_to_skiplist (8), reserved (16)]
    # OTHER:         [ptr (8), reserved (16)]
    var _data0: UInt64
    var _data1: UInt64
    var _data2: UInt64

    def __init__(out self):
        self.type = ValueType()
        self._data0 = 0
        self._data1 = 0
        self._data2 = 0

    @always_inline
    def __eq__(self, other: GenericValue) -> Bool:
        if self.type.value != other.type.value:
            return False
        if self.is_none():
            return True
        if self.type.value == ValueType.INT:
            return self.as_int() == other.as_int()
        if self.type.value == ValueType.FLOAT:
            return self.as_float() == other.as_float()
        if self.is_string():
            var l1 = self.string_len()
            var l2 = other.string_len()
            if l1 != l2: return False
            if l1 == 0: return True
            
            if self.type.value == ValueType.STRING_SSO:
                # Both are SSO because types are equal
                return self._data0 == other._data0 and self._data1 == other._data1 and self._data2 == other._data2
            else:
                var p1 = self.as_string()
                var p2 = other.as_string()
                for i in range(l1):
                    if p1[unsafe_offset=i] != p2[unsafe_offset=i]: return False
                return True
        return self._data0 == other._data0

    @staticmethod
    @always_inline
    def _wyhash_mix(a: UInt64, b: UInt64) -> UInt64:
        """Folded multiply: full 128-bit product, XOR high and low halves.
        Inspired by Modular AHash — provides complete bit avalanche vs the
        previous 32-bit split multiply which lost cross-half diffusion."""
        var m = a.cast[DType.uint128]() * b.cast[DType.uint128]()
        var lo = UInt64(m & 0xFFFFFFFFFFFFFFFF)
        var hi = UInt64(m >> 64)
        return lo ^ hi

    @always_inline
    def __hash__(self) -> Int:
        if self.is_none():
            return 0
        if self.type.value == ValueType.INT:
            return Int(self.as_int())
        if self.is_string():
            if self.type.value == ValueType.STRING_SSO:
                var seed: UInt64 = 0xa0761d6478bd642f
                var h = Self._wyhash_mix(seed ^ self._data0, seed ^ self._data1)
                h = Self._wyhash_mix(h, seed ^ self._data2)
                return Int(h)
            else:
                var p = self.as_string().unsafe_bitcast[UInt64]()
                var l = self.string_len()
                var h: UInt64 = 0xa0761d6478bd642f
                # Main 8-byte loop (all full blocks except possibly last)
                var full_blocks = l // 8
                for i in range(full_blocks):
                    h = Self._wyhash_mix(h ^ p.load(i), 0xe7037ed1a0b428db)
                # AHash-style tail: overlapping read of last 8 bytes eliminates
                # per-byte scalar loop. Safe because heap strings are always >23 bytes.
                var tail_bytes = l - full_blocks * 8
                if tail_bytes > 0:
                    var last8 = (self.as_string().unsafe_offset(l) - 8).unsafe_bitcast[UInt64]().load()
                    h = Self._wyhash_mix(h ^ last8, 0xe7037ed1a0b428db)
                return Int(h)
        return Int(self._data0)

    @staticmethod
    @always_inline
    def _pack_sso(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Tuple[UInt64, UInt64, UInt64]:
        """Pack ≤23 raw bytes into SSO layout (d0/d1/d2) via up to three wide loads
        instead of ≤23 serially-dependent shift-ORs. Layout: d0 = length byte + chars
        0..6, d1 = chars 7..14, d2 = chars 15..22. Loads are made safe against reading
        past the key by `_load_u64_le` (no caller-side padding required)."""
        var n0 = min(length, 7)
        var w0 = _load_u64_le(ptr, min(length, 8))
        var d0 = (UInt64(length) | (w0 << 8)) & _low_byte_mask(n0 + 1)
        var d1: UInt64 = 0
        var d2: UInt64 = 0
        if length > 7:
            var a1 = length - 7
            d1 = _load_u64_le(ptr.unsafe_offset(7), a1) & _low_byte_mask(min(a1, 8))
        if length > 15:
            var a2 = length - 15
            d2 = _load_u64_le(ptr.unsafe_offset(15), a2) & _low_byte_mask(min(a2, 8))
        return (d0, d1, d2)

    @staticmethod
    @always_inline
    def hash_and_pack_sso(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Tuple[UInt64, UInt64, UInt64, UInt64]:
        """Pack raw bytes into SSO layout (d0/d1/d2) and compute hash in one pass.
        Returns (hash, d0, d1, d2). Only valid for length ≤ 23."""
        var packed = Self._pack_sso(ptr, length)
        var d0 = packed[0]
        var d1 = packed[1]
        var d2 = packed[2]
        var seed: UInt64 = 0xa0761d6478bd642f
        var h = Self._wyhash_mix(seed ^ d0, seed ^ d1)
        h = Self._wyhash_mix(h, seed ^ d2)
        return (h, d0, d1, d2)

    @always_inline
    def is_string(self) -> Bool:
        return self.type.value == ValueType.STRING or self.type.value == ValueType.STRING_SSO

    @always_inline
    def is_none(self) -> Bool:
        return self.type.value == ValueType.NONE

    @always_inline
    def as_string(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        if self.type.value == ValueType.STRING_SSO:
            return null_ptr[UInt8, MutUntrackedOrigin]()
        return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    @always_inline
    def as_string_safe(self, buf: Pointer[UInt8, MutUntrackedOrigin]) -> Pointer[UInt8, MutUntrackedOrigin]:
        """Return a pointer to string bytes. For heap strings, returns the heap pointer directly.
        For SSO, copies inline data to caller-provided buf (must be ≥24 bytes) and returns buf.
        Caller must not free the returned pointer — it may be either buf or a heap pointer."""
        if self.type.value == ValueType.STRING_SSO:
            self.copy_to(buf)
            return buf
        return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    @always_inline
    def lex_lt(self, other: GenericValue) -> Bool:
        """Redis's ordering for zset members that share a score (gh #251).

        `zslInsert` breaks a score tie with `sdscmp`: compare bytes over the
        common prefix, and if those are equal the SHORTER member sorts first.
        Pion had no tie-break at all, so equal-score members came back in
        reverse-insertion order — which silently broke the score-0 +
        `ZRANGEBYLEX` lexicographic-index idiom, whose entire premise is this
        ordering, and gave tied `ZRANK`s the wrong values.

        Comparison is UNSIGNED byte order, matching `memcmp`: member bytes are
        arbitrary binary, and a signed compare would sort anything with the
        high bit set (UTF-8 continuation bytes, for one) before ASCII.

        Called from the skip-list descent, so it must stay allocation-free —
        `as_string_safe` returns the heap pointer directly for a heap STRING
        and only copies for SSO, whose bytes live inside the value and have no
        address to take."""
        # Members reaching the skip list are always strings (ZADD builds them
        # with from_ptr). A non-string can only arrive through a bug; report
        # "not less than" rather than reading _data1 as a length.
        if not self.is_string() or not other.is_string():
            return False
        var abuf = stack_allocation[24, UInt8]()
        var bbuf = stack_allocation[24, UInt8]()
        var ap = self.as_string_safe(abuf)
        var bp = other.as_string_safe(bbuf)
        var alen = self.string_len()
        var blen = other.string_len()
        var n = alen if alen < blen else blen
        for i in range(n):
            var x = ap[unsafe_offset=i]
            var y = bp[unsafe_offset=i]
            if x != y:
                return x < y
        return alen < blen

    @always_inline
    def string_len(self) -> Int:
        if self.type.value == ValueType.STRING_SSO:
            return Int(self._data0 & 0xFF)
        return Int(self._data1)

    @always_inline
    def copy_to(self, dest: Pointer[UInt8, MutUntrackedOrigin]):
        from std.memory import unsafe_memcpy
        if self.type.value == ValueType.STRING_SSO:
            var length = Int(self._data0 & 0xFF)
            var d0 = self._data0
            var d1 = self._data1
            var d2 = self._data2
            
            if length > 0:
                var c0 = d0 >> 8
                for i in range(min(length, 7)):
                    dest.store(i, UInt8((c0 >> UInt64(i * 8)) & 0xFF))
            if length > 7:
                for i in range(min(length - 7, 8)):
                    dest.store(7 + i, UInt8((d1 >> UInt64(i * 8)) & 0xFF))
            if length > 15:
                for i in range(min(length - 15, 8)):
                    dest.store(15 + i, UInt8((d2 >> UInt64(i * 8)) & 0xFF))
        else:
            var src = self.as_string()
            var length = self.string_len()
            if length > 0:
                unsafe_memcpy(dest=dest, src=src, count=length)

    @always_inline
    def as_int(self) -> Int64:
        return Int64(self._data0)

    @always_inline
    def as_float(self) -> Float64:
        return bitcast[DType.float64, 1](self._data0)

    @always_inline
    def as_vector(self) -> Pointer[Vector, MutUntrackedOrigin]:
        return Pointer[Vector, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    @always_inline
    def as_list(self) -> Pointer[NoneType, MutUntrackedOrigin]:
        return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    @always_inline
    def as_hash(self) -> Pointer[NoneType, MutUntrackedOrigin]:
        return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    @always_inline
    def as_set(self) -> Pointer[NoneType, MutUntrackedOrigin]:
        return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    @always_inline
    def as_zset(self) -> Pointer[NoneType, MutUntrackedOrigin]:
        return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    @always_inline
    def __str__(self) -> String:
        if self.is_none():
            return ""
        if self.type.value == ValueType.INT:
            return String(self.as_int())
        if self.type.value == ValueType.FLOAT:
            return String(self.as_float())
        if self.is_string():
            var length = self.string_len()
            if length == 0: return ""
            var buf = alloc[UInt8](length)
            self.copy_to(buf)
            # gh #181: `String(buf, length)` is not a from-bytes constructor in
            # Mojo b2 — it formatted the POINTER ("0x1349…"), so every __str__
            # of a string value returned its address. Build from the bytes.
            var s = String(StringSpan[MutUntrackedOrigin](
                unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=buf, length=length)))
            buf.unsafe_free()
            return s
        return ""

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    def as_bitmap(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    @always_inline
    def bitmap_len(self) -> Int:
        return Int(self._data1)
    
    @always_inline
    def as_hll(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))
    
    @always_inline
    def as_geo(self) -> Pointer[NoneType, MutUntrackedOrigin]:
        return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    @always_inline
    def set_ptr(mut self, ptr: Pointer[NoneType, MutUntrackedOrigin]):
        self._data0 = UInt64(Int(ptr))

    @always_inline
    def is_string_like(self) -> Bool:
        """gh #232: Redis has no separate bitmap type — a bitmap IS a string.

        BITMAP shares STRING's exact `[ptr, len]` layout (see the layout map
        above), so every read-only string accessor — `string_len`, `copy_to` —
        already produces the right answer for one. Only the type check excluded
        it, which is why `SETBIT k ...; STRLEN k` returned WRONGTYPE where Redis
        returns a length.

        HLL is deliberately NOT included: Pion stores a dense register array
        while Redis stores its own sparse `HYLL` encoding, so exposing those
        bytes as a string would swap a WRONGTYPE divergence for a value one."""
        return self.is_string() or self.type.value == ValueType.BITMAP

    @always_inline
    def is_container(self) -> Bool:
        """True for the aggregate types, i.e. the ones a numeric op must refuse.

        gh #232: INCR/INCRBY/DECRBY on a list, hash, set, zset, geo or stream
        answered `ERR value is not an integer or out of range`, where Redis
        answers WRONGTYPE. Clients branch on the code — redis-py raises a
        different exception class for each — so a type mix-up surfaced as a
        parse complaint about a value the caller never wrote.

        BITMAP and HLL are excluded on purpose. Redis stores both as strings,
        so `SETBIT k 7 1; INCR k` really is a not-an-integer error there, and
        Pion already agrees; adding them here would introduce a divergence."""
        var t = self.type.value
        return (t == ValueType.LIST or t == ValueType.HASH or t == ValueType.SET
                or t == ValueType.ZSET or t == ValueType.GEO or t == ValueType.STREAM
                or t == ValueType.VSET)

    @always_inline
    def bitmap_view(self, scratch: Pointer[UInt8, MutUntrackedOrigin],
                    mut out_len: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        """gh #232: READ-ONLY byte view for the bitmap ops, over either shape.

        BITMAP and heap STRING alias directly — same `[ptr, len]` words.
        STRING_SSO keeps its bytes INSIDE the value, so its `_data0` is a
        length-and-characters word, NOT an address; it must be copied out to
        `scratch` (>= 23 bytes) or the caller would dereference packed
        characters as a pointer.

        Read-only is load-bearing, not a naming choice: `setbit()` frees and
        reallocs when it grows, and a STRING's payload may live in the gh #163
        blob arena, which the heap allocator must never free. Mutating bitmap
        ops therefore still refuse a STRING."""
        if self.type.value == ValueType.STRING_SSO:
            out_len = Int(self._data0 & 0xFF)
            self.copy_to(scratch)
            return scratch
        out_len = Int(self._data1)
        return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self._data0))

    def owned_bitmap_copy(self, min_bytes: Int,
                          mut out_len: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        """gh #232: a FRESH heap buffer holding this value's bytes, for mutation.

        `bitmap_view` is deliberately read-only, because `setbit()` frees and
        reallocs when it grows and a STRING's payload may live in the gh #163
        blob arena — handing an mmap'd address to the allocator. That is why
        SETBIT/APPEND/SETRANGE refused a STRING outright, and why Pion answered
        WRONGTYPE to `SET k "hello"; SETBIT k 10 1`, which real Redis accepts
        (there is no separate bitmap type there: a bitmap IS a string).

        This returns a COPY instead, so the result is always plain heap and
        always safe to grow, free, or hand to `setbit`. Works for every
        string-like shape:

          - STRING_SSO — bytes live inside the value, so `_data0` is a
            length-and-characters word, NOT an address
          - heap STRING — plain copy
          - arena-backed STRING — copied OUT of the mapping, so the original is
            left for the arena to reclaim by compaction
          - BITMAP — same [ptr, len] layout as a heap STRING

        `min_bytes` sizes the result up front (the caller knows the bit offset),
        and the tail past the original length is zeroed the way a grown bitmap
        must be. The CALLER owns the result and must free the ORIGINAL payload
        via `free_str_payload()`, which is already arena- and SSO-safe."""
        var src_len = self.string_len()
        var cap = min_bytes if min_bytes > src_len else src_len
        if cap <= 0:
            cap = 1
        var out = alloc[UInt8](cap)
        if src_len > 0:
            self.copy_to(out)
        if cap > src_len:
            unsafe_memset(out.unsafe_offset(src_len), 0, cap - src_len)
        out_len = cap
        return out

    @always_inline
    def is_blob_backed(self) -> Bool:
        """gh #163: True when this STRING's bytes live in the mmap'd blob arena
        rather than on the heap. `_data2` is the `reserved` word for heap STRING
        (unused by __eq__/__hash__ on this branch), which is what makes the tag
        free."""
        return self.type.value == ValueType.STRING and self._data2 == BLOB_TAG

    @always_inline
    def free_str_payload(self):
        """gh #123: free a heap STRING payload (>23B, alloc'd by from_ptr/from_string).
        No-op for SSO and every non-STRING type — those own no separate heap buffer
        here (LIST/HASH/SET/ZSET/BITMAP/HLL/GEO have their own lifecycle elsewhere).
        Only call on a value the map owns and is dropping (overwrite / remove).

        gh #163: a blob-backed STRING points into the blob arena's mapping, which
        this value does not own — free()ing it would hand a mapped address to the
        allocator. The arena reclaims by compaction instead."""
        if self.type.value == ValueType.STRING and self._data0 != 0 \
           and self._data2 != BLOB_TAG and self._data2 != BORROW_TAG:
            self.as_string().unsafe_free()

    @staticmethod
    @always_inline
    def borrow(src: Pointer[UInt8, _], length: Int) -> Self:
        """A LOOKUP key over bytes the caller keeps alive — no heap copy (gh #394).

        `from_ptr` copies anything over 23 bytes to the heap, and a key built
        only to look something up was almost never freed: every command on a
        key longer than 23 bytes leaked ~48-64 B, plain GET included. This
        builds the same value without the copy: SSO up to 23 bytes (exactly
        `from_ptr`'s bytes), and above that a STRING pointing at `ptr`, tagged
        BORROW_TAG.

        It is safe for the hash map because the two forms never mix: a stored
        key over 23 bytes is a heap STRING and so is this one, and __eq__ and
        __hash__ read only the type, length and bytes — the from_ptr_unsafe
        trap (CLAUDE.md) is a STRING probe against an SSO-stored key, which
        cannot happen here.

        Rules, checked by tools/audit_borrowed_keys.py (must report 0):
          - `ptr` must outlive the value: the recv buffer or a RESP token's
            bytes during its own command. Never a local String's bytes — Mojo
            may destroy the String right after its last use.
          - A borrowed value may be passed to SlabHashMap/StripedHashMap
            (get/set/remove/...), which copy it on insert (owned()), and read.
            It must never be stored anywhere else: a list, skip list, stream,
            struct field or another container would keep a pointer into a
            buffer the next recv overwrites."""
        var ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(src))
        var res = Self()
        if length <= 23:
            # from_ptr's SSO branch, inlined: `ptr` may be a short String's
            # inline bytes, which must not reach an out-of-line call (gh #349).
            res.type = ValueType(ValueType.STRING_SSO)
            var packed = Self._pack_sso(ptr, length)
            res._data0 = packed[0]
            res._data1 = packed[1]
            res._data2 = packed[2]
            return res
        res.type = ValueType(ValueType.STRING)
        res._data0 = UInt64(Int(ptr))
        res._data1 = UInt64(length)
        res._data2 = BORROW_TAG
        return res

    @staticmethod
    @no_inline
    def borrow_buf(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Self:
        """`borrow()` out of line, for the fast path's recv-buffer keys.

        Inlining borrow's SSO pack at every fast-path site grew
        process_data_plane enough to cost the gate's MSET row (interleaved
        A/B, 2026-09-29) — the gh #261 lesson: that function's shape is
        load-bearing. `from_ptr`, which these sites called before, was out of
        line too. The recv buffer is heap, so its pointer may cross a call;
        NEVER pass a stack or short-String pointer here (gh #349) — those
        stay on the inline `borrow()`."""
        return Self.borrow(ptr, length)

    @always_inline
    def is_borrowed(self) -> Bool:
        return self.type.value == ValueType.STRING and self._data2 == BORROW_TAG

    @always_inline
    def owned(self) -> Self:
        """This value as something a container may keep: a borrowed STRING is
        copied to the heap, anything else is returned unchanged (gh #394).
        One compare on the path every hash-map insert takes."""
        if self.type.value == ValueType.STRING and self._data2 == BORROW_TAG:
            return Self.from_ptr(self.as_string(), Int(self._data1))
        return self

    @staticmethod
    @always_inline
    def from_blob_ptr(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Self:
        """gh #163: wrap bytes already living in the blob arena — no copy, no heap.
        The caller guarantees the mapping outlives the value (the arena is mapped
        for the process lifetime)."""
        var res = Self()
        res.type = ValueType(ValueType.STRING)
        res._data0 = UInt64(Int(ptr))
        res._data1 = UInt64(length)
        res._data2 = BLOB_TAG
        return res

    @always_inline
    def clone(self) -> Self:
        """Deep-copy: a heap STRING gets a fresh owned buffer; all other types are
        plain 32-byte value copies (no aliased heap payload here). Used where a value
        read from the map is re-stored under a new key (RENAME/COPY) so the two entries
        never share a buffer that a later free would dangle."""
        if self.type.value == ValueType.STRING:
            return Self.from_ptr(self.as_string(), self.string_len())
        return self

    @staticmethod
    def from_int(val: Int64) -> Self:
        var res = Self()
        res.type = ValueType(ValueType.INT)
        res._data0 = UInt64(val)
        return res

    @staticmethod
    def from_float(val: Float64) -> Self:
        """The Float64's bit pattern, not a conversion. This stored
        `UInt64(Int(val))` — truncating every score to an integer — and the
        zset member dict (the only producer of FLOAT values) records each
        member's score here: ZADD of a new fractional score then unlinked the
        node at the TRUNCATED score, found none, and linked a second node, so
        `ZADD z 1.5 a; ZADD z 2.5 a` left ZCARD 2 and ZSCORE 1.5. FLOAT values
        never reach the keyspace, WAL or snapshot, so no format changes."""
        var res = Self()
        res.type = ValueType(ValueType.FLOAT)
        res._data0 = bitcast[DType.uint64, 1](val)
        return res

    @staticmethod
    def from_ptr(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Self:
        from std.memory import unsafe_memcpy
        var res = Self()
        if length <= 23:
            res.type = ValueType(ValueType.STRING_SSO)
            var packed = Self._pack_sso(ptr, length)
            res._data0 = packed[0]
            res._data1 = packed[1]
            res._data2 = packed[2]
        else:
            var new_ptr = alloc[UInt8](length)
            unsafe_memcpy(dest=new_ptr, src=ptr, count=length)
            res.type = ValueType(ValueType.STRING)
            res._data0 = UInt64(Int(new_ptr))
            res._data1 = UInt64(length)
        return res

    @staticmethod
    def from_ptr_unsafe(b: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Self:
        var res = Self()
        res.type = ValueType(ValueType.STRING)
        res._data0 = UInt64(Int(b))
        res._data1 = UInt64(length)
        return res


    @staticmethod
    def from_string(val: String) -> Self:
        var length = val.byte_length()
        var res = Self()
        
        if length <= 23:
            res.type = ValueType(ValueType.STRING_SSO)
            var d0: UInt64 = UInt64(length)
            var d1: UInt64 = 0
            var d2: UInt64 = 0
            
            var b = val.as_bytes()
            for i in range(min(length, 7)):
                d0 |= UInt64(b[i]) << UInt64((i + 1) * 8)
            if length > 7:
                for i in range(min(length - 7, 8)):
                    d1 |= UInt64(b[7 + i]) << UInt64(i * 8)
            if length > 15:
                for i in range(min(length - 15, 8)):
                    d2 |= UInt64(b[15 + i]) << UInt64(i * 8)
            
            res._data0 = d0
            res._data1 = d1
            res._data2 = d2
        else:
            var ptr = alloc[UInt8](length + 1)
            for i in range(length):
                ptr[unsafe_offset=i] = val.as_bytes()[i]
            ptr[unsafe_offset=length] = 0
            res.type = ValueType(ValueType.STRING)
            res._data0 = UInt64(Int(ptr))
            res._data1 = UInt64(length)
            
        return res
