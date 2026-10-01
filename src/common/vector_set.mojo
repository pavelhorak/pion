"""gh #366: one Redis-8-style vector set — the value behind a VSET key.

Kept apart from src/commands/vset.mojo (the command handlers) so that
container_free.mojo can free a set without importing the network layer.
"""
from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.collections import List, Span
from std.math import sqrt

from src.common.value import GenericValue, ValueType
from src.common.hash_map import SlabHashMap
from src.common.prng import Xoshiro256PlusPlus


struct VectorSet(Movable):
    """One vector set. Slots are append-only; VREM tombstones a slot and the
    name index forgets it (a re-added name takes a fresh slot)."""
    var dim: Int
    var cap: Int
    var n: Int                                   # slots used, tombstones included
    var live: Int
    var vecs: Pointer[Float32, MutUntrackedOrigin]   # [cap * dim], unit vectors
    var norms: Pointer[Float32, MutUntrackedOrigin]  # [cap]
    var alive: Pointer[UInt8, MutUntrackedOrigin]    # [cap]
    var names: List[String]
    var attrs: List[String]                      # "" = no attribute
    var index: SlabHashMap                       # element name -> INT slot
    var rng: Xoshiro256PlusPlus

    def __init__(out self, dim: Int):
        self.dim = dim
        self.cap = 16
        self.n = 0
        self.live = 0
        self.vecs = alloc[Float32](self.cap * dim)
        self.norms = alloc[Float32](self.cap)
        self.alive = alloc[UInt8](self.cap)
        self.names = List[String]()
        self.attrs = List[String]()
        self.index = SlabHashMap(16)
        self.rng = Xoshiro256PlusPlus(UInt64(dim) * 0x9E3779B97F4A7C15 + 366)

    def __moveinit__(out self, deinit take: Self):
        self.dim = take.dim
        self.cap = take.cap
        self.n = take.n
        self.live = take.live
        self.vecs = take.vecs
        self.norms = take.norms
        self.alive = take.alive
        self.names = take.names^
        self.attrs = take.attrs^
        self.index = take.index^
        self.rng = take.rng

    def release(mut self):
        """Free the flat buffers. The Lists and the index are freed by the
        owner's unsafe_deinit_pointee()."""
        if is_not_null(self.vecs): self.vecs.free()
        if is_not_null(self.norms): self.norms.free()
        if is_not_null(self.alive): self.alive.free()
        self.vecs = null_ptr[Float32, MutUntrackedOrigin]()
        self.norms = null_ptr[Float32, MutUntrackedOrigin]()
        self.alive = null_ptr[UInt8, MutUntrackedOrigin]()

    def find(self, p: Pointer[UInt8, MutUntrackedOrigin], l: Int) -> Int:
        var v = self.index.get(GenericValue.borrow(p, l))
        if v.type.value == ValueType.INT:
            return Int(v.as_int())
        return -1

    def _grow(mut self):
        var ncap = self.cap * 2
        var nv = alloc[Float32](ncap * self.dim)
        unsafe_memcpy(dest=nv, src=self.vecs, count=self.n * self.dim)
        self.vecs.free()
        self.vecs = nv
        var nn = alloc[Float32](ncap)
        unsafe_memcpy(dest=nn, src=self.norms, count=self.n)
        self.norms.free()
        self.norms = nn
        var na = alloc[UInt8](ncap)
        unsafe_memcpy(dest=na, src=self.alive, count=self.n)
        self.alive.free()
        self.alive = na
        self.cap = ncap

    def _store(mut self, slot: Int, v: Pointer[Float32, MutUntrackedOrigin]):
        var ss = Float32(0.0)
        for d in range(self.dim):
            ss += v[unsafe_offset=d] * v[unsafe_offset=d]
        var norm = sqrt(ss)
        var inv = Float32(1.0) / norm if norm > 0.0 else Float32(0.0)
        var dst = self.vecs + slot * self.dim
        for d in range(self.dim):
            dst[unsafe_offset=d] = v[unsafe_offset=d] * inv
        self.norms[unsafe_offset=slot] = norm

    def add(mut self, name_p: Pointer[UInt8, MutUntrackedOrigin], name_l: Int,
            v: Pointer[Float32, MutUntrackedOrigin]) -> Bool:
        """True if the element is new; an existing element's vector is replaced."""
        var slot = self.find(name_p, name_l)
        if slot >= 0:
            self._store(slot, v)
            return False
        if self.n == self.cap:
            self._grow()
        slot = self.n
        self.n += 1
        self._store(slot, v)
        self.alive[unsafe_offset=slot] = 1
        var nm = String(StringSpan[MutUntrackedOrigin](
            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=name_p, length=name_l)))
        self.names.append(nm^)
        self.attrs.append(String(""))
        self.index.set(GenericValue.borrow(name_p, name_l), GenericValue.from_int(Int64(slot)))
        self.live += 1
        return True

    def add_stored(mut self, name_p: Pointer[UInt8, MutUntrackedOrigin], name_l: Int,
                   unit: Pointer[Float32, MutUntrackedOrigin], norm: Float32):
        """gh #378: restore an element exactly as it was stored — the unit
        vector and norm `add` computed, not a vector to normalize again. A
        replay that re-ran `add` on `unit * norm` would round differently in
        the last bit, and VEMB/VSIM would no longer agree with the original."""
        var slot = self.find(name_p, name_l)
        if slot < 0:
            if self.n == self.cap:
                self._grow()
            slot = self.n
            self.n += 1
            self.alive[unsafe_offset=slot] = 1
            var nm = String(StringSpan[MutUntrackedOrigin](
                unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=name_p, length=name_l)))
            self.names.append(nm^)
            self.attrs.append(String(""))
            self.index.set(GenericValue.borrow(name_p, name_l), GenericValue.from_int(Int64(slot)))
            self.live += 1
        unsafe_memcpy(dest=self.vecs + slot * self.dim, src=unit, count=self.dim)
        self.norms[unsafe_offset=slot] = norm

    def stored_payload(self, slot: Int, out_buf: Pointer[UInt8, MutUntrackedOrigin]) -> Int:
        """gh #378: the VADD record payload for `slot` — [4B f32 norm][dim x f32
        unit vector] — written into out_buf (needs `payload_len()` bytes).
        One encoding for the WAL, the snapshot and BGREWRITEAOF."""
        out_buf.bitcast[Float32]()[unsafe_offset=0] = self.norms[unsafe_offset=slot]
        unsafe_memcpy(dest=out_buf + 4, src=(self.vecs + slot * self.dim).bitcast[UInt8](),
                      count=self.dim * 4)
        return self.payload_len()

    def payload_len(self) -> Int:
        return 4 + self.dim * 4

    def remove(mut self, name_p: Pointer[UInt8, MutUntrackedOrigin], name_l: Int) -> Bool:
        var slot = self.find(name_p, name_l)
        if slot < 0:
            return False
        self.alive[unsafe_offset=slot] = 0
        self.attrs[slot] = String("")
        _ = self.index.remove_generic(GenericValue.borrow(name_p, name_l))
        self.live -= 1
        return True

    def similarity(self, q: Pointer[Float32, MutUntrackedOrigin], slot: Int) -> Float64:
        """Redis's score: (1 + cos) / 2 against a unit query."""
        var acc = SIMD[DType.float32, 8](0)
        var v = self.vecs + slot * self.dim
        var d = 0
        while d + 8 <= self.dim:
            acc = acc + (q + d).load[width=8]() * (v + d).load[width=8]()
            d += 8
        var dot = acc.reduce_add()
        while d < self.dim:
            dot += q[unsafe_offset=d] * v[unsafe_offset=d]
            d += 1
        return (1.0 + Float64(dot)) / 2.0

    def search(self, q: Pointer[Float32, MutUntrackedOrigin], count: Int,
               mut out_slots: List[Int], mut out_scores: List[Float64]):
        """Top-`count` live slots by similarity, best first (exact)."""
        for s in range(self.n):
            if self.alive[unsafe_offset=s] == 0:
                continue
            var sc = self.similarity(q, s)
            if len(out_slots) < count:
                out_slots.append(s)
                out_scores.append(sc)
            elif sc > out_scores[len(out_scores) - 1]:
                out_slots[len(out_slots) - 1] = s
                out_scores[len(out_scores) - 1] = sc
            else:
                continue
            # keep descending order: bubble the new tail entry up
            var j = len(out_slots) - 1
            while j > 0 and out_scores[j] > out_scores[j - 1]:
                var ts = out_scores[j]; out_scores[j] = out_scores[j - 1]; out_scores[j - 1] = ts
                var ti = out_slots[j]; out_slots[j] = out_slots[j - 1]; out_slots[j - 1] = ti
                j -= 1


def free_vset(vs: Pointer[VectorSet, MutUntrackedOrigin]):
    """Free a vector set that no key references any more (gh #369)."""
    if is_null(vs): return
    vs[].release()
    vs.unsafe_deinit_pointee()
    vs.free()
