"""#46: tombstones for the vector index — a document that left the keyspace
leaves the search results.

The vector index records a document's key when its vector is ingested (the
`__hk__<slot>` map) and nothing updated that record afterwards: FT.SEARCH kept
returning documents that were deleted, expired, flushed, renamed, or given a
new vector. Now the hash that holds an indexed vector carries its slot and a
pointer to its worker's VecTomb (SlabHashMap.vec_slot / vec_tomb), and the
slot dies with it:

  - freeing the hash (DEL, UNLINK, expiry, FLUSHALL, an overwrite, a pop or
    HDEL that empties it: every route ends in SlabHashMap.__del__);
  - setting or removing its vector field (HSET with a new vector, HDEL, a
    field TTL), in SlabHashMap's own set/remove;
  - RENAME, which moves the hash to a key the slot does not name.

A dead slot is one byte in an array every worker shares (the index is
shared), so a search on any worker skips it at once. The search itself is
unchanged — dead nodes still route the beam, as HNSW soft deletes do — and
FT.SEARCH filters them out of the results, widening the candidate set so it
still returns k live documents.

The worker that killed a slot logs it (WAL record 38: the index's build id
and the slot) after the batch; replay applies the records of the index that
loaded, so a restart neither revives a dead slot nor kills one of a later
build. This module imports nothing from Pion, so hash_map.mojo can use it.
"""

from std.memory.unsafe_pointer import Pointer
from std.collections import List
from std.atomic import Atomic, Ordering


struct VecTomb(Movable):
    """One worker's view of the shared tombstones."""
    var dead: Pointer[UInt8, MutUntrackedOrigin]        # a byte per ingest slot, shared
    var dead_count: Pointer[UInt64, MutUntrackedOrigin]  # slots dead, shared
    var n: Int                                            # slots in `dead`
    var pending: List[Int]                                # killed here, not yet logged
    # The slot numbering's generation, shared: it moves on whenever slots
    # start again from 0 (FT.DROPINDEX, an index replaced, a new ingest). A
    # hash remembers the generation it was linked in, and a kill from an older
    # one is ignored — or the hashes of a dropped index, deleted later, would
    # kill the new index's slots of the same numbers.
    var gen: Pointer[UInt64, MutUntrackedOrigin]

    def __init__(out self, dead: Pointer[UInt8, MutUntrackedOrigin],
                 dead_count: Pointer[UInt64, MutUntrackedOrigin], n: Int,
                 gen: Pointer[UInt64, MutUntrackedOrigin]):
        self.dead = dead
        self.dead_count = dead_count
        self.n = n
        self.pending = List[Int]()
        self.gen = gen

    def __init__(out self, *, deinit take: Self):
        self.dead = take.dead
        self.dead_count = take.dead_count
        self.n = take.n
        self.pending = take.pending^
        self.gen = take.gen

    @always_inline
    def generation(self) -> UInt64:
        return self.gen[] if Int(self.gen) != 0 else UInt64(0)

    def kill(mut self, slot: Int, gen: UInt64):
        """Slot `slot`, linked in generation `gen`, no longer names a live
        document."""
        if slot < 0 or slot >= self.n or Int(self.dead) == 0 or gen != self.generation():
            return
        if self.dead[slot] != 0:
            return
        self.dead[slot] = 1
        _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELAXED](self.dead_count, UInt64(1))
        self.pending.append(slot)

    @always_inline
    def any_dead(self) -> Bool:
        return Int(self.dead_count) != 0 and self.dead_count[] != 0
