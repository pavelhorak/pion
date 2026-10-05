from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memset, unsafe_memcpy
from std.sys.intrinsics import prefetch
from std.bit import count_trailing_zeros
from std.math import iota
from std.collections import List
from src.common.value import GenericValue, ValueType, BLOB_TAG
from src.common.prng import Xoshiro256PlusPlus

# Per-lane weights (1 << lane) for packing a 16-lane bool compare into a
# movemask-style bitmask. Computed once at compile time.
comptime _LANE_BITS = SIMD[DType.uint16, 16](UInt16(1)) << iota[DType.uint16, 16]()

@always_inline
def _movemask16(m: SIMD[DType.bool, 16]) -> Int:
    """Pack a 16-lane bool SIMD compare into an integer bitmask (bit i = lane i).
    Canonical Swiss-table movemask: iterate only matching lanes via count_trailing_zeros
    instead of walking all 16 slots scalar."""
    return Int((m.cast[DType.uint16]() * _LANE_BITS).reduce_add())

struct SlabHashMap(Movable):
    var metadata: Pointer[UInt8, MutUntrackedOrigin]
    var keys: Pointer[GenericValue, MutUntrackedOrigin]
    var values: Pointer[GenericValue, MutUntrackedOrigin]
    var capacity: Int
    var size: Int
    # gh #210: live DELETED-slot count. Counted into the load factor so
    # set/remove churn (size stays small, tombstones accumulate) still
    # triggers the rehash that purges them.
    var tombstones: Int
    # gh #394: where an OVERWRITTEN aggregate goes. Null for every map but the
    # keyspace's shards (hash fields and set members are never aggregates).
    # The map cannot free a list/zset/stream itself — container_free imports
    # this module — so it parks the old value here and the engine frees the
    # batch (free_graveyard) once the replies that might borrow it are out.
    # Before, `SET k v` over a list dropped the list's handle: leaked.
    var graveyard: Pointer[List[GenericValue], MutUntrackedOrigin]
    # gh #392: a HASH value's per-field expiry — field → INT deadline (unix
    # ns). Null until a field of this hash is given a TTL. It used to be the
    # global TTL table under `key + "::" + field`, which aliased (key `a::b`
    # field `c` and key `a` field `b::c` shared ONE entry), survived DEL, and
    # did not follow RENAME. Owned here, a field TTL goes wherever the hash
    # goes and dies with it.
    var field_ttl: Pointer[SlabHashMap, MutUntrackedOrigin]

    # Metadata constants
    comptime EMPTY = UInt8(0b10000000)
    comptime DELETED = UInt8(0b11111111)

    def __init__(out self, capacity: Int, buckets_unused: Int = 0):
        # Capacity should be power of 2 for efficiency
        var real_cap = 16
        while real_cap < capacity:
             real_cap *= 2
        
        self.capacity = real_cap
        self.size = 0
        self.tombstones = 0
        self.graveyard = null_ptr[List[GenericValue], MutUntrackedOrigin]()
        self.field_ttl = null_ptr[SlabHashMap, MutUntrackedOrigin]()
        self.metadata = alloc[UInt8](real_cap + 16)
        unsafe_memset(self.metadata, UInt8(Self.EMPTY), real_cap + 16)
        self.keys = alloc[GenericValue](real_cap)
        self.values = alloc[GenericValue](real_cap)
        for i in range(real_cap):
            (self.keys.unsafe_offset(i)).unsafe_write(GenericValue())
            (self.values.unsafe_offset(i)).unsafe_write(GenericValue())


    @always_inline
    def reset(mut self):
        """Lazy reset: skip entirely when the map is already clean (fresh from pool)."""
        self._drop_field_ttl()
        if self.size == 0:
            return
        # gh #131 §1.4: visit only OCCUPIED slots via the Swiss metadata instead of
        # writing GenericValue() across full capacity — on a sparsely-filled pooled
        # map (HSET path, 1000 maps) the old all-capacity loop dominated. Occupied =
        # metadata high bit clear (EMPTY=0x80 / DELETED=0xFF both set). free_str_payload
        # preserves the gh #123 heap-STRING free; DELETED slots were already reset to
        # GenericValue() at delete time, so skipping them leaks nothing.
        comptime width = 16
        var i = 0
        while i < self.capacity:
            var meta = (self.metadata.unsafe_offset(i)).load[width=width]()
            var occ = _movemask16(
                (meta & SIMD[DType.uint8, width](0x80)).eq(SIMD[DType.uint8, width](0)))
            while occ != 0:
                var idx = i + count_trailing_zeros(occ)
                self.keys[unsafe_offset=idx].free_str_payload()
                self._retire(self.values[unsafe_offset=idx])   # gh #394: FLUSHALL freed no aggregate
                self.keys[unsafe_offset=idx] = GenericValue()
                self.values[unsafe_offset=idx] = GenericValue()
                occ &= occ - 1
            i += width
        unsafe_memset(self.metadata, UInt8(Self.EMPTY), self.capacity + 16)
        self.size = 0
        self.tombstones = 0

    def __moveinit__(out self, deinit take: Self):
        self.metadata = take.metadata
        self.keys = take.keys
        self.values = take.values
        self.capacity = take.capacity
        self.size = take.size
        self.tombstones = take.tombstones
        self.graveyard = take.graveyard
        self.field_ttl = take.field_ttl
    
    # ── gh #392: per-field expiry of a HASH ─────────────────────────────────
    def _drop_field_ttl(mut self):
        if Int(self.field_ttl) != 0:
            self.field_ttl.unsafe_deinit_pointee()
            self.field_ttl.unsafe_free()
            self.field_ttl = null_ptr[SlabHashMap, MutUntrackedOrigin]()

    @always_inline
    def has_field_ttls(self) -> Bool:
        return Int(self.field_ttl) != 0 and self.field_ttl[].size > 0

    def field_deadline(self, field: GenericValue) -> Int64:
        """The field's deadline (unix ns), or 0 when it has none."""
        if Int(self.field_ttl) == 0:
            return 0
        var d = self.field_ttl[].get(field)
        if d.type.value == ValueType.INT:
            return d.as_int()
        return 0

    def set_field_deadline(mut self, field: GenericValue, deadline_ns: Int64):
        if Int(self.field_ttl) == 0:
            self.field_ttl = alloc[SlabHashMap](1)
            self.field_ttl.unsafe_write(SlabHashMap(16))
        self.field_ttl[].set(field, GenericValue.from_int(deadline_ns))

    def clear_field_deadline(mut self, field: GenericValue) -> Bool:
        if Int(self.field_ttl) == 0:
            return False
        return self.field_ttl[].remove_generic(field)

    def expire_field(mut self, field: GenericValue, now_ns: Int64) -> Bool:
        """Delete `field` if its deadline has passed. True when it did."""
        var d = self.field_deadline(field)
        if d != 0 and d <= now_ns:
            _ = self.remove_generic(field)   # also drops the deadline (see remove)
            return True
        return False

    def purge_expired_fields(mut self, now_ns: Int64) -> Int:
        """Delete every field whose deadline has passed; returns how many."""
        if Int(self.field_ttl) == 0 or self.field_ttl[].size == 0:
            return 0
        var n = 0
        var ft = self.field_ttl
        for i in range(ft[].capacity):
            var m = ft[].metadata[unsafe_offset=i]
            if m == Self.EMPTY or m == Self.DELETED:
                continue
            var d = ft[].values[unsafe_offset=i]
            if d.type.value == ValueType.INT and d.as_int() <= now_ns:
                # Removing from self drops ft's entry too, which frees the key
                # payload `f` points at — so `f` is not touched after.
                var f = ft[].keys[unsafe_offset=i]
                _ = self.remove_generic(f)
                n += 1
        return n

    def _h1(self, h: UInt64) -> Int:
        return Int(h >> 7)

    @always_inline
    def _retire(mut self, old: GenericValue):
        """Drop a value this map is overwriting: free a heap STRING payload
        here, park an aggregate / HLL / bitmap in the graveyard (gh #394)."""
        if old.type.value == ValueType.STRING:
            old.free_str_payload()
        elif Int(self.graveyard) != 0 and (old.is_container()
                                           or old.type.value == ValueType.HLL
                                           or old.type.value == ValueType.BITMAP):
            self._park(old)
        # BITMAP was once left out, because setbit() freed the buffer it grew
        # out of, so a bitmap that SET, BITOP or FLUSHALL replaced was never
        # freed at all. setbit() now leaves the old buffer here. The set() paths skip this call when the new
        # value keeps the same buffer (an in-place SETBIT).

    @no_inline
    def _park(mut self, old: GenericValue):
        """Out of line on purpose: _retire is inlined into every store, and
        through set_with_hash into the fast path's MSET loop — List.append's
        grow path there costs the hot loop even though it never runs."""
        self.graveyard[].append(old)

    def _h2(self, h: UInt64) -> UInt8:
        return UInt8(h & 0x7F)

    def set(mut self, key_str: String, var value: GenericValue):
        # gh #394: borrowed; set() copies it only if it inserts.
        self.set(GenericValue.borrow(key_str.unsafe_ptr(), key_str.byte_length()), value^)

    def set(mut self, key: GenericValue, var value: GenericValue):
        if (self.size + self.tombstones) * 100 > self.capacity * 70:
            self._rehash()

        var h = UInt64(key.__hash__())
        var h1 = self._h1(h)
        var h2 = self._h2(h)

        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)
        var deleted_vec = SIMD[DType.uint8, width](Self.DELETED)

        while True:  # re-entered only after the full-table backstop rehash
            var mask = self.capacity - 1
            var idx = h1 & mask
            var start_idx = idx
            # gh #210: first tombstone on the probe path — reused on insert so
            # churn can't exhaust EMPTY slots. Safe anywhere at-or-before the
            # terminating group: lookups scan a full 16-group for h2 matches
            # before honoring its EMPTY.
            var first_del = -1

            while True:
                var next_idx = (idx + 16) & mask
                prefetch(self.metadata.unsafe_offset(next_idx))

                var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()

                var mbits = _movemask16(meta_chunk.eq(h2_vec))
                while mbits != 0:
                    var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                    prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                    if self.keys[unsafe_offset=slot_idx] == key:
                        # gh #123: free the old payload we're replacing. Alias guard: skip
                        # when the new value points at the same buffer (self-overwrite).
                        if self.values[unsafe_offset=slot_idx]._data0 != value._data0:
                            self._retire(self.values[unsafe_offset=slot_idx])
                        self.values[unsafe_offset=slot_idx] = value.owned()
                        return
                    mbits &= mbits - 1

                if first_del < 0:
                    var dbits = _movemask16(meta_chunk.eq(deleted_vec))
                    if dbits != 0:
                        first_del = (idx + count_trailing_zeros(dbits)) & mask

                var ebits = _movemask16(meta_chunk.eq(empty_vec))
                if ebits != 0:
                    var slot_idx = (idx + count_trailing_zeros(ebits)) & mask
                    if first_del >= 0:
                        slot_idx = first_del
                        self.tombstones -= 1
                    self.metadata[unsafe_offset=slot_idx] = h2
                    if slot_idx < 16:
                        self.metadata[unsafe_offset=self.capacity + slot_idx] = h2
                    self.keys[unsafe_offset=slot_idx] = key.owned()
                    self.values[unsafe_offset=slot_idx] = value.owned()
                    self.size += 1
                    return

                idx = next_idx
                if idx == start_idx: break  # gh #210: full wrap, no EMPTY left

            if first_del >= 0:
                self.metadata[unsafe_offset=first_del] = h2
                if first_del < 16:
                    self.metadata[unsafe_offset=self.capacity + first_del] = h2
                self.keys[unsafe_offset=first_del] = key.owned()
                self.values[unsafe_offset=first_del] = value.owned()
                self.size += 1
                self.tombstones -= 1
                return
            # Every slot occupied — unreachable while the load check holds;
            # backstop instead of the pre-#210 infinite spin.
            self._rehash()

    def set_str_reuse_with_hash(mut self, key: GenericValue, h: UInt64,
                                val_ptr: Pointer[UInt8, MutUntrackedOrigin],
                                val_len: Int):
        """gh #175: string-SET that reuses an existing equal-length heap payload
        in place (one memcpy) instead of alloc-new + free-old + memcpy. On the
        pipeline profile every overwrite is same-size, so this removes a
        tcmalloc round-trip (~3.5% of worker time) from the SET hot path.
        Reuse requires: existing value is a heap STRING (not SSO, not
        blob-arena-tagged) of exactly `val_len` bytes, and the new value is
        too long for SSO. Every other shape falls back to free + from_ptr,
        which is byte-for-byte the old behavior."""
        if (self.size + self.tombstones) * 100 > self.capacity * 70:
            self._rehash()

        var h1 = self._h1(h)
        var h2 = self._h2(h)

        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)
        var deleted_vec = SIMD[DType.uint8, width](Self.DELETED)

        while True:  # re-entered only after the full-table backstop rehash
            var mask = self.capacity - 1
            var idx = h1 & mask
            var start_idx = idx
            var first_del = -1  # gh #210: first tombstone on the probe path

            while True:
                var next_idx = (idx + 16) & mask
                prefetch(self.metadata.unsafe_offset(next_idx))

                var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()

                var mbits = _movemask16(meta_chunk.eq(h2_vec))
                while mbits != 0:
                    var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                    prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                    if self.keys[unsafe_offset=slot_idx] == key:
                        var old = self.values[unsafe_offset=slot_idx]
                        if old.type.value == ValueType.STRING and val_len > 23 \
                                and Int(old._data1) == val_len \
                                and old._data2 != BLOB_TAG:
                            # In-place overwrite. src is the recv buffer, dest is
                            # the stored heap payload — they can never alias.
                            unsafe_memcpy(
                                dest=Pointer[UInt8, MutUntrackedOrigin](
                                    unsafe_from_address=Int(old._data0)),
                                src=val_ptr, count=val_len)
                            return
                        self._retire(self.values[unsafe_offset=slot_idx])
                        self.values[unsafe_offset=slot_idx] = GenericValue.from_ptr(val_ptr, val_len)
                        return
                    mbits &= mbits - 1

                if first_del < 0:
                    var dbits = _movemask16(meta_chunk.eq(deleted_vec))
                    if dbits != 0:
                        first_del = (idx + count_trailing_zeros(dbits)) & mask

                var ebits = _movemask16(meta_chunk.eq(empty_vec))
                if ebits != 0:
                    var slot_idx = (idx + count_trailing_zeros(ebits)) & mask
                    if first_del >= 0:
                        slot_idx = first_del
                        self.tombstones -= 1
                    self.metadata[unsafe_offset=slot_idx] = h2
                    if slot_idx < 16:
                        self.metadata[unsafe_offset=self.capacity + slot_idx] = h2
                    self.keys[unsafe_offset=slot_idx] = key.owned()
                    self.values[unsafe_offset=slot_idx] = GenericValue.from_ptr(val_ptr, val_len)
                    self.size += 1
                    return

                idx = next_idx
                if idx == start_idx: break  # gh #210: full wrap, no EMPTY left

            if first_del >= 0:
                self.metadata[unsafe_offset=first_del] = h2
                if first_del < 16:
                    self.metadata[unsafe_offset=self.capacity + first_del] = h2
                self.keys[unsafe_offset=first_del] = key.owned()
                self.values[unsafe_offset=first_del] = GenericValue.from_ptr(val_ptr, val_len)
                self.size += 1
                self.tombstones -= 1
                return
            self._rehash()  # gh #210 backstop: every slot occupied

    def get(self, key_str: String) -> GenericValue:
        # gh #394: borrowed for the call — the from_string copy was never freed.
        return self.get(GenericValue.borrow(key_str.unsafe_ptr(), key_str.byte_length()))

    def get(self, key: GenericValue) -> GenericValue:
        if self.capacity == 0: return GenericValue()
        
        var h = UInt64(key.__hash__())
        var h1 = self._h1(h)
        var h2 = self._h2(h)
        
        var mask = self.capacity - 1
        var idx = h1 & mask
        var start_idx = idx
        
        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)

        while True:
            # Prefetch next group metadata while processing current
            var next_idx = (idx + 16) & mask
            prefetch(self.metadata.unsafe_offset(next_idx))

            var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()

            var mbits = _movemask16(meta_chunk.eq(h2_vec))
            while mbits != 0:
                var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                prefetch((self.values.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                if self.keys[unsafe_offset=slot_idx] == key:
                    return self.values[unsafe_offset=slot_idx]
                mbits &= mbits - 1

            if _movemask16(meta_chunk.eq(empty_vec)) != 0:
                return GenericValue()

            idx = next_idx
            if idx == start_idx: break

        return GenericValue()

    @always_inline
    def get_value_ptr(mut self, key: GenericValue) -> Pointer[GenericValue, MutUntrackedOrigin]:
        """Return pointer to value slot for in-place mutation (e.g. INCR).
        Returns null pointer if key not found."""
        if self.capacity == 0: return null_ptr[GenericValue, MutUntrackedOrigin]()

        var h = UInt64(key.__hash__())
        var h1 = self._h1(h)
        var h2 = self._h2(h)

        var mask = self.capacity - 1
        var idx = h1 & mask
        var start_idx = idx

        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)

        while True:
            var next_idx = (idx + 16) & mask
            prefetch(self.metadata.unsafe_offset(next_idx))

            var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()

            var mbits = _movemask16(meta_chunk.eq(h2_vec))
            while mbits != 0:
                var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                if self.keys[unsafe_offset=slot_idx] == key:
                    return self.values.unsafe_offset(slot_idx)
                mbits &= mbits - 1

            if _movemask16(meta_chunk.eq(empty_vec)) != 0:
                return null_ptr[GenericValue, MutUntrackedOrigin]()

            idx = next_idx
            if idx == start_idx: break

        return null_ptr[GenericValue, MutUntrackedOrigin]()

    @always_inline
    def set_with_hash(mut self, key: GenericValue, var value: GenericValue, h: UInt64):
        """Like set() but skips hash computation — caller provides precomputed hash."""
        if (self.size + self.tombstones) * 100 > self.capacity * 70:
            self._rehash()
        var h1 = self._h1(h)
        var h2 = self._h2(h)
        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)
        var deleted_vec = SIMD[DType.uint8, width](Self.DELETED)
        while True:  # re-entered only after the full-table backstop rehash
            var mask = self.capacity - 1
            var idx = h1 & mask
            var start_idx = idx
            var first_del = -1  # gh #210: first tombstone on the probe path
            while True:
                var next_idx = (idx + 16) & mask
                prefetch(self.metadata.unsafe_offset(next_idx))
                var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()
                var mbits = _movemask16(meta_chunk.eq(h2_vec))
                while mbits != 0:
                    var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                    prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                    if self.keys[unsafe_offset=slot_idx] == key:
                        if self.values[unsafe_offset=slot_idx]._data0 != value._data0:  # gh #123: free replaced payload
                            self._retire(self.values[unsafe_offset=slot_idx])
                        self.values[unsafe_offset=slot_idx] = value.owned()
                        return
                    mbits &= mbits - 1
                if first_del < 0:
                    var dbits = _movemask16(meta_chunk.eq(deleted_vec))
                    if dbits != 0:
                        first_del = (idx + count_trailing_zeros(dbits)) & mask
                var ebits = _movemask16(meta_chunk.eq(empty_vec))
                if ebits != 0:
                    var slot_idx = (idx + count_trailing_zeros(ebits)) & mask
                    if first_del >= 0:
                        slot_idx = first_del
                        self.tombstones -= 1
                    self.metadata[unsafe_offset=slot_idx] = h2
                    if slot_idx < 16:
                        self.metadata[unsafe_offset=self.capacity + slot_idx] = h2
                    self.keys[unsafe_offset=slot_idx] = key.owned()
                    self.values[unsafe_offset=slot_idx] = value.owned()
                    self.size += 1
                    return
                idx = next_idx
                if idx == start_idx: break  # gh #210: full wrap, no EMPTY left
            if first_del >= 0:
                self.metadata[unsafe_offset=first_del] = h2
                if first_del < 16:
                    self.metadata[unsafe_offset=self.capacity + first_del] = h2
                self.keys[unsafe_offset=first_del] = key.owned()
                self.values[unsafe_offset=first_del] = value.owned()
                self.size += 1
                self.tombstones -= 1
                return
            self._rehash()  # gh #210 backstop: every slot occupied

    @always_inline
    def remove_generic_with_hash(mut self, key: GenericValue, h: UInt64) -> Bool:
        """Like remove_generic() but skips hash computation — caller provides precomputed hash."""
        var h1 = self._h1(h)
        var h2 = self._h2(h)
        var mask = self.capacity - 1
        var idx = h1 & mask
        var start_idx = idx
        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)
        while True:
            var next_idx = (idx + 16) & mask
            prefetch(self.metadata.unsafe_offset(next_idx))
            var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()
            var mbits = _movemask16(meta_chunk.eq(h2_vec))
            while mbits != 0:
                var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                if self.keys[unsafe_offset=slot_idx] == key:
                    self.metadata[unsafe_offset=slot_idx] = Self.DELETED
                    if slot_idx < 16:
                        self.metadata[unsafe_offset=self.capacity + slot_idx] = Self.DELETED
                    # gh #392: a removed field loses its TTL. Before the free
                    # below: `key` may be this very slot's stored key.
                    if Int(self.field_ttl) != 0:
                        _ = self.field_ttl[].remove_generic(key)
                    # gh #123: free both heap payloads (no-op for SSO/other types).
                    self.keys[unsafe_offset=slot_idx].free_str_payload()
                    self.values[unsafe_offset=slot_idx].free_str_payload()
                    self.keys[unsafe_offset=slot_idx] = GenericValue()
                    self.values[unsafe_offset=slot_idx] = GenericValue()
                    self.size -= 1
                    self.tombstones += 1
                    return True
                mbits &= mbits - 1
            if _movemask16(meta_chunk.eq(empty_vec)) != 0:
                return False
            idx = next_idx
            if idx == start_idx: break
        return False

    @always_inline
    def remove_generic_with_hash_taking(mut self, key: GenericValue, h: UInt64, mut taken: GenericValue) -> Bool:
        """remove_generic_with_hash that also hands back an AGGREGATE value
        (list/hash/set/zset/geo/stream/vset, and HLL/bitmap) in `taken`, so the caller can free the
        container it points to (gh #369) without a second probe. String
        payloads are freed here as before and never handed back."""
        var h1 = self._h1(h)
        var h2 = self._h2(h)
        var mask = self.capacity - 1
        var idx = h1 & mask
        var start_idx = idx
        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)
        while True:
            var next_idx = (idx + 16) & mask
            prefetch(self.metadata.unsafe_offset(next_idx))
            var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()
            var mbits = _movemask16(meta_chunk.eq(h2_vec))
            while mbits != 0:
                var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                if self.keys[unsafe_offset=slot_idx] == key:
                    self.metadata[unsafe_offset=slot_idx] = Self.DELETED
                    if slot_idx < 16:
                        self.metadata[unsafe_offset=self.capacity + slot_idx] = Self.DELETED
                    # gh #392: a removed field loses its TTL. Before the free
                    # below: `key` may be this very slot's stored key.
                    if Int(self.field_ttl) != 0:
                        _ = self.field_ttl[].remove_generic(key)
                    # gh #123: free both heap payloads (no-op for SSO/other types).
                    self.keys[unsafe_offset=slot_idx].free_str_payload()
                    # container_free.owns_heap(), spelled out: that module
                    # imports this one. HLL/BITMAP are bare heap blocks the
                    # map cannot free itself (gh #394).
                    var _tv = self.values[unsafe_offset=slot_idx]
                    if _tv.is_container() or _tv.type.value == ValueType.HLL \
                            or _tv.type.value == ValueType.BITMAP:
                        taken = _tv
                    self.values[unsafe_offset=slot_idx].free_str_payload()
                    self.keys[unsafe_offset=slot_idx] = GenericValue()
                    self.values[unsafe_offset=slot_idx] = GenericValue()
                    self.size -= 1
                    self.tombstones += 1
                    return True
                mbits &= mbits - 1
            if _movemask16(meta_chunk.eq(empty_vec)) != 0:
                return False
            idx = next_idx
            if idx == start_idx: break
        return False

    @always_inline
    def get_with_hash(self, key: GenericValue, h: UInt64) -> GenericValue:
        """Like get() but skips hash computation — caller provides precomputed hash."""
        if self.capacity == 0: return GenericValue()

        var h1 = self._h1(h)
        var h2 = self._h2(h)

        var mask = self.capacity - 1
        var idx = h1 & mask
        var start_idx = idx

        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)

        while True:
            var next_idx = (idx + 16) & mask
            prefetch(self.metadata.unsafe_offset(next_idx))

            var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()

            var mbits = _movemask16(meta_chunk.eq(h2_vec))
            while mbits != 0:
                var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                prefetch((self.values.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                if self.keys[unsafe_offset=slot_idx] == key:
                    return self.values[unsafe_offset=slot_idx]
                mbits &= mbits - 1

            if _movemask16(meta_chunk.eq(empty_vec)) != 0:
                return GenericValue()

            idx = next_idx
            if idx == start_idx: break

        return GenericValue()

    @always_inline
    def get_with_sso(self, h: UInt64, d0: UInt64, d1: UInt64, d2: UInt64) -> GenericValue:
        """Zero-copy SSO lookup: uses precomputed hash and packed d0/d1/d2 for comparison.
        Avoids constructing a GenericValue for the lookup key entirely."""
        if self.capacity == 0: return GenericValue()

        var h1 = self._h1(h)
        var h2 = self._h2(h)

        var mask = self.capacity - 1
        var idx = h1 & mask
        var start_idx = idx

        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)

        while True:
            var next_idx = (idx + 16) & mask
            prefetch(self.metadata.unsafe_offset(next_idx))

            var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()

            var mbits = _movemask16(meta_chunk.eq(h2_vec))
            while mbits != 0:
                var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                prefetch((self.values.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                var k = self.keys[unsafe_offset=slot_idx]
                if k.type.value == ValueType.STRING_SSO and k._data0 == d0 and k._data1 == d1 and k._data2 == d2:
                    return self.values[unsafe_offset=slot_idx]
                mbits &= mbits - 1

            if _movemask16(meta_chunk.eq(empty_vec)) != 0:
                return GenericValue()

            idx = next_idx
            if idx == start_idx: break

        return GenericValue()

    @always_inline
    def get_with_ptr(self, ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> GenericValue:
        """Zero-copy GET for SSO keys (≤23B). Packs, hashes, and compares without GenericValue."""
        if length <= 23:
            var packed = GenericValue.hash_and_pack_sso(ptr, length)
            return self.get_with_sso(packed[0], packed[1], packed[2], packed[3])
        else:
            # gh #394: borrow, don't copy — the copy was never freed.
            return self.get(GenericValue.borrow(ptr, length))

    def contains(self, key_str: String) -> Bool:
        return self.get(key_str).type.value != ValueType.NONE

    def remove(mut self, key_str: String) -> Bool:
        return self.remove_generic(GenericValue.borrow(key_str.unsafe_ptr(), key_str.byte_length()))

    def remove_generic(mut self, key: GenericValue) -> Bool:
        var h = UInt64(key.__hash__())
        var h1 = self._h1(h)
        var h2 = self._h2(h)
        
        var mask = self.capacity - 1
        var idx = h1 & mask
        var start_idx = idx
        
        comptime width = 16
        var h2_vec = SIMD[DType.uint8, width](h2)
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)

        while True:
            var next_idx = (idx + 16) & mask
            prefetch(self.metadata.unsafe_offset(next_idx))

            var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()

            var mbits = _movemask16(meta_chunk.eq(h2_vec))
            while mbits != 0:
                var slot_idx = (idx + count_trailing_zeros(mbits)) & mask
                prefetch((self.keys.unsafe_offset(slot_idx)).unsafe_bitcast[UInt8]())
                if self.keys[unsafe_offset=slot_idx] == key:
                    self.metadata[unsafe_offset=slot_idx] = Self.DELETED
                    if slot_idx < 16:
                        self.metadata[unsafe_offset=self.capacity + slot_idx] = Self.DELETED
                    # gh #392: a removed field loses its TTL. Before the free
                    # below: `key` may be this very slot's stored key.
                    if Int(self.field_ttl) != 0:
                        _ = self.field_ttl[].remove_generic(key)
                    # gh #123: free both heap payloads (no-op for SSO/other types).
                    self.keys[unsafe_offset=slot_idx].free_str_payload()
                    self.values[unsafe_offset=slot_idx].free_str_payload()
                    self.keys[unsafe_offset=slot_idx] = GenericValue()
                    self.values[unsafe_offset=slot_idx] = GenericValue()
                    self.size -= 1
                    self.tombstones += 1
                    return True
                mbits &= mbits - 1

            if _movemask16(meta_chunk.eq(empty_vec)) != 0:
                return False

            idx = next_idx
            if idx == start_idx: break
        return False

    def pop_random(mut self, mut prng: Xoshiro256PlusPlus) -> GenericValue:
        if self.size == 0:
            return GenericValue()
        var idx = Int(prng.next() & UInt64(self.capacity - 1))
        for i in range(self.capacity):
            var curr = (idx + i) & (self.capacity - 1)
            var m = self.metadata[unsafe_offset=curr]
            if m != Self.EMPTY and m != Self.DELETED:
                var key = self.keys[unsafe_offset=curr]  # ownership of any key payload transfers to caller
                self.metadata[unsafe_offset=curr] = Self.DELETED
                if curr < 16:
                    self.metadata[unsafe_offset=self.capacity + curr] = Self.DELETED
                self.values[unsafe_offset=curr].free_str_payload()  # gh #123: free the dropped value payload
                self.keys[unsafe_offset=curr] = GenericValue()
                self.values[unsafe_offset=curr] = GenericValue()
                self.size -= 1
                self.tombstones += 1
                return key
        return GenericValue()

    def _rehash(mut self):
        var old_cap = self.capacity
        var old_meta = self.metadata
        var old_keys = self.keys
        var old_values = self.values

        # gh #210: doubling is for genuine live-entry pressure. When the table
        # is mostly tombstones (churn workload), rebuild at the same capacity —
        # the rebuild purges them; doubling would grow memory without bound.
        var new_cap = old_cap * 2
        if self.size * 100 <= old_cap * 35:
            new_cap = old_cap
        self.capacity = new_cap
        self.size = 0
        self.tombstones = 0
        self.metadata = alloc[UInt8](self.capacity + 16)
        unsafe_memset(self.metadata, UInt8(Self.EMPTY), self.capacity + 16)
        self.keys = alloc[GenericValue](self.capacity)
        self.values = alloc[GenericValue](self.capacity)
        for i in range(self.capacity):
            (self.keys.unsafe_offset(i)).unsafe_write(GenericValue())
            (self.values.unsafe_offset(i)).unsafe_write(GenericValue())
            
        var mask = self.capacity - 1
        
        comptime width = 16
        var empty_vec = SIMD[DType.uint8, width](Self.EMPTY)

        for i in range(old_cap):
            var m = old_meta[unsafe_offset=i]
            if m != Self.EMPTY and m != Self.DELETED:
                var key = old_keys[unsafe_offset=i]
                var h = UInt64(key.__hash__())
                var h1 = Int(h >> 7)
                var h2 = UInt8(h & 0x7F)
                var idx = h1 & mask
                
                var found = False
                while not found:
                    var meta_chunk = (self.metadata.unsafe_offset(idx)).load[width=width]()
                    var ebits = _movemask16(meta_chunk.eq(empty_vec))
                    if ebits != 0:
                        var slot_idx = (idx + count_trailing_zeros(ebits)) & mask
                        self.metadata[unsafe_offset=slot_idx] = h2
                        if slot_idx < 16:
                            self.metadata[unsafe_offset=self.capacity + slot_idx] = h2
                        self.keys[unsafe_offset=slot_idx] = key.owned()
                        self.values[unsafe_offset=slot_idx] = old_values[unsafe_offset=i]
                        self.size += 1
                        found = True
                    if found: break
                    idx = (idx + 16) & mask
        
        for i in range(old_cap):
             (old_keys.unsafe_offset(i)).unsafe_deinit_pointee()
             (old_values.unsafe_offset(i)).unsafe_deinit_pointee()
        old_meta.unsafe_free()
        old_keys.unsafe_free()
        old_values.unsafe_free()

    def forget_borrowed(mut self):
        """Drop every key and value WITHOUT freeing its heap payload.

        For a temporary map filled with GenericValues BORROWED from another
        container (a union/intersection "seen" set). set() stores the value it
        is given — a shallow copy of a > 23-byte string is the same heap
        pointer — and __del__ frees every key's payload, so destroying such a
        map freed the SOURCE container's members: SUNION, ZUNION, ZINTER and
        ZDIFF (read-only commands) left the source set serving freed memory
        and corrupted the allocator (SIGSEGV inside tcmalloc on a later
        command). Call this before destroying a map that does not own its
        contents; a map that does must be filled with clone()s instead."""
        for i in range(self.capacity):
            self.keys[unsafe_offset=i] = GenericValue()
            self.values[unsafe_offset=i] = GenericValue()
        self.size = 0

    def __del__(deinit self):
        if Int(self.field_ttl) != 0:
            self.field_ttl.unsafe_deinit_pointee()
            self.field_ttl.unsafe_free()
        if Int(self.metadata) != 0:
            self.metadata.unsafe_free()
        if Int(self.keys) != 0:
            for i in range(self.capacity):
                self.keys[unsafe_offset=i].free_str_payload()  # gh #123: free key heap payloads
                (self.keys.unsafe_offset(i)).unsafe_deinit_pointee()
            self.keys.unsafe_free()
        if Int(self.values) != 0:
            for i in range(self.capacity):
                self.values[unsafe_offset=i].free_str_payload()  # gh #123: free value heap payloads
                (self.values.unsafe_offset(i)).unsafe_deinit_pointee()
            self.values.unsafe_free()
struct StripedHashMap(Movable):
    """8-shard Swiss Table. Eliminates O(N) rehash pauses — each shard is 8x smaller.
    Each shard rehashes independently: worst-case pause = 1/8 of an equivalent single-map rehash.
    Shard selection uses bits 0-2 of the hash (same hash used for internal probing)."""
    var shards: Pointer[SlabHashMap, MutUntrackedOrigin]
    # gh #394: aggregates the shards overwrote, freed by the engine after each
    # dispatch batch (container_free.free_graveyard). Shared by all 8 shards.
    var graveyard: Pointer[List[GenericValue], MutUntrackedOrigin]
    # gh #392: keys of hashes that have (or had) field TTLs — what the active
    # expiry sweep walks. An entry can go stale (the key was deleted, renamed,
    # or its last field TTL went); the sweep drops those. Lazy expiry on every
    # read keeps visibility right regardless.
    var field_ttl_index: Pointer[SlabHashMap, MutUntrackedOrigin]
    # The worker's key TTLs (the engine's ttl_map), so that removing
    # a key takes its TTL with it. Before, every route but DEL, UNLINK and
    # expiry (an aggregate emptied by a pop, GETDEL, a *STORE replacing its
    # destination) left the entry behind, and the next key of that name expired
    # at the old deadline. Null for a keyspace that keeps no TTLs.
    var ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]

    def __init__(out self, initial_capacity: Int):
        var shard_cap = max(16, initial_capacity // 8)
        self.ttl_map = null_ptr[SlabHashMap, MutUntrackedOrigin]()
        self.graveyard = alloc[List[GenericValue]](1)
        self.graveyard.unsafe_write(List[GenericValue]())
        self.field_ttl_index = alloc[SlabHashMap](1)
        self.field_ttl_index.unsafe_write(SlabHashMap(16))
        self.shards = alloc[SlabHashMap](8)
        for i in range(8):
            (self.shards.unsafe_offset(i)).unsafe_write(SlabHashMap(shard_cap))
            self.shards[unsafe_offset=i].graveyard = self.graveyard

    def __moveinit__(out self, deinit take: Self):
        self.shards = take.shards
        self.graveyard = take.graveyard
        self.field_ttl_index = take.field_ttl_index
        self.ttl_map = take.ttl_map

    def __del__(deinit self):
        if Int(self.shards) != 0:
            for i in range(8):
                (self.shards.unsafe_offset(i)).unsafe_deinit_pointee()
            self.shards.unsafe_free()
        if Int(self.graveyard) != 0:
            self.graveyard.unsafe_deinit_pointee()
            self.graveyard.unsafe_free()
        if Int(self.field_ttl_index) != 0:
            self.field_ttl_index.unsafe_deinit_pointee()
            self.field_ttl_index.unsafe_free()

    @always_inline
    def _shard(self, h: UInt64) -> Int:
        return Int(h & 7)

    @always_inline
    def get(self, key: GenericValue) -> GenericValue:
        var h = UInt64(key.__hash__())
        return self.shards[unsafe_offset=self._shard(h)].get_with_hash(key, h)

    @always_inline
    def get(self, key_str: String) -> GenericValue:
        # gh #394: borrowed for the call — the from_string copy was never freed.
        return self.get(GenericValue.borrow(key_str.unsafe_ptr(), key_str.byte_length()))

    @always_inline
    def get_with_hash(self, key: GenericValue, h: UInt64) -> GenericValue:
        return self.shards[unsafe_offset=self._shard(h)].get_with_hash(key, h)

    @always_inline
    def get_with_ptr(self, ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> GenericValue:
        """Zero-copy GET for SSO keys (≤23B). Packs bytes, hashes, and compares in one flow
        without constructing a GenericValue for the lookup key."""
        if length <= 23:
            var packed = GenericValue.hash_and_pack_sso(ptr, length)
            var h = packed[0]
            return self.shards[unsafe_offset=self._shard(h)].get_with_sso(h, packed[1], packed[2], packed[3])
        else:
            # gh #394: borrow, don't copy — the copy was never freed.
            return self.get(GenericValue.borrow(ptr, length))

    @always_inline
    def set(mut self, key: GenericValue, var value: GenericValue):
        var h = UInt64(key.__hash__())
        self.shards[unsafe_offset=self._shard(h)].set_with_hash(key, value^, h)

    @always_inline
    def set_with_hash(mut self, key: GenericValue, var value: GenericValue, h: UInt64):
        """set() for a caller that already holds the key's hash (gh #230).

        MSET needs the hash a second time — for the WATCH version slot — so
        computing it here as well would mean hashing every key twice."""
        self.shards[unsafe_offset=self._shard(h)].set_with_hash(key, value^, h)

    @always_inline
    def set_str_reuse(mut self, key: GenericValue,
                      val_ptr: Pointer[UInt8, MutUntrackedOrigin],
                      val_len: Int):
        """gh #175: SET-from-raw-bytes with in-place payload reuse — see
        SlabHashMap.set_str_reuse_with_hash."""
        var h = UInt64(key.__hash__())
        self.shards[unsafe_offset=self._shard(h)].set_str_reuse_with_hash(key, h, val_ptr, val_len)

    @always_inline
    def set(mut self, key_str: String, var value: GenericValue):
        # gh #394: borrowed; set() copies it only if it inserts.
        self.set(GenericValue.borrow(key_str.unsafe_ptr(), key_str.byte_length()), value^)

    @always_inline
    def _drop_ttl(mut self, key: GenericValue):
        """A removed key's TTL goes with it. Called FIRST, while `key`
        is valid: a caller may pass the keyspace's own stored key, which the
        removal frees. (A key read out of the TTL map itself must be an owned
        copy — this frees the map's.)"""
        if Int(self.ttl_map) != 0 and self.ttl_map[].size > 0:
            _ = self.ttl_map[].remove_generic(key)

    @always_inline
    def remove_generic(mut self, key: GenericValue) -> Bool:
        self._drop_ttl(key)
        var h = UInt64(key.__hash__())
        return self.shards[unsafe_offset=self._shard(h)].remove_generic_with_hash(key, h)

    def remove_generic_taking(mut self, key: GenericValue, mut taken: GenericValue) -> Bool:
        """gh #369: remove, and hand back an aggregate value so its container
        can be freed (see container_free.mojo). One probe, like remove_generic."""
        self._drop_ttl(key)
        var h = UInt64(key.__hash__())
        return self.shards[unsafe_offset=self._shard(h)].remove_generic_with_hash_taking(key, h, taken)

    @always_inline
    def remove(mut self, key_str: String) -> Bool:
        return self.remove_generic(GenericValue.borrow(key_str.unsafe_ptr(), key_str.byte_length()))

    @always_inline
    def contains(self, key_str: String) -> Bool:
        return self.get(key_str).type.value != ValueType.NONE

    @always_inline
    def get_value_ptr(mut self, key: GenericValue) -> Pointer[GenericValue, MutUntrackedOrigin]:
        """Return pointer to value slot for in-place mutation (e.g. INCR).
        Returns null pointer if key not found."""
        var h = UInt64(key.__hash__())
        return self.shards[unsafe_offset=self._shard(h)].get_value_ptr(key)

    def reset(mut self):
        for i in range(8):
            self.shards[unsafe_offset=i].reset()
        self.field_ttl_index[].reset()
        # The keys' TTLs go with them, or a key created later under a
        # flushed name expires at the old deadline.
        if Int(self.ttl_map) != 0:
            self.ttl_map[].reset()

    @always_inline
    def note_field_ttl(mut self, key: GenericValue):
        """gh #392: this key's hash now has a field TTL — index it for the sweep."""
        self.field_ttl_index[].set(key, GenericValue.from_int(1))
