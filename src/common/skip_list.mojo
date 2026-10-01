from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from src.memory.slab_allocator import SlabAllocator
from src.common.value import GenericValue, ValueType
from src.common.hash_map import SlabHashMap
from src.common.prng import Xoshiro256PlusPlus
from std.collections import Array

# Maximum level for skip list
comptime MAX_LEVEL = 16

@fieldwise_init
struct SkipListNode(Copyable, Movable):
    var score: Float64
    var obj: GenericValue
    var forward: Array[Pointer[Self, MutUntrackedOrigin], MAX_LEVEL]
    var level: Int

    def __init__(out self, level: Int, score: Float64, var obj: GenericValue):
        self.level = level
        self.score = score
        self.obj = obj^
        self.forward = Array[Pointer[Self, MutUntrackedOrigin], MAX_LEVEL](
            uninitialized=True
        )
        # gh #203: null only the slots this node actually owns. `random_level()`
        # is p=0.25, so E[level] = 1.33 of MAX_LEVEL = 16 — the old
        # `range(MAX_LEVEL)` fill wrote 128 B per insert and ~94% of it was
        # dead. (The head node passes level=MAX_LEVEL, so it still gets all 16.)
        #
        # Slots at index >= level are left uninitialized, which is only sound
        # because every reader is bounded by the level discipline: a node is
        # linked at level i ONLY if its level > i, and every `forward[i]` read
        # in this file is reached either from the head (full) or through a link
        # at level i. Concretely — `_insert_node`/`_remove_node` walk and write
        # `update[i]` for i < self.level and read `update[i][].forward[i]`;
        # `pop_min` reads `first[].forward[i]` only inside
        # `if head[].forward[i] == first`, which already proves first.level > i;
        # `get_range` and the iterators use forward[0] and every node has
        # level >= 1. Preserve that invariant in any new traversal.
        for i in range(level):
            self.forward[i] = null_ptr[Self, MutUntrackedOrigin]()

    def deinit(owned self):
        pass

struct ZPopResult(Movable):
    var score: Float64
    var obj: GenericValue
    var valid: Bool
    
    def __init__(out self):
        self.score = 0.0
        self.obj = GenericValue()
        self.valid = False

    def __init__(out self, score: Float64, var obj: GenericValue, valid: Bool):
        self.score = score
        self.obj = obj^
        self.valid = valid

    def __moveinit__(out self, deinit take: Self):
        self.score = take.score
        self.obj = take.obj^
        self.valid = take.valid

struct SlabSkipList(Movable):
    var head: Pointer[SkipListNode, MutUntrackedOrigin]
    var level: Int
    var length: Int
    var allocator: SlabAllocator[SkipListNode]
    var prng: Xoshiro256PlusPlus
    # gh #187: member → score (FLOAT) dict for Redis ZADD upsert semantics.
    # Owns deep copies of heap-STRING member payloads — nodes and the dict must
    # never share a payload, because remove/reset free the stored key's buffer.
    var members: SlabHashMap

    def __init__(out self, capacity: Int):
        self.level = 1
        self.length = 0
        self.allocator = SlabAllocator[SkipListNode](capacity)
        self.prng = Xoshiro256PlusPlus(UInt64(capacity) ^ 0xDEADBEEF)
        self.members = SlabHashMap(16)

        # Head node with MAX_LEVEL
        var head_ptr = self.allocator.allocate()
        head_ptr.unsafe_write(SkipListNode(MAX_LEVEL, 0.0, GenericValue()))
        self.head = head_ptr


    def reset(mut self):
        self.allocator.reset()
        self.level = 1
        self.length = 0
        self.members.reset()
        var head_ptr = self.allocator.allocate()
        head_ptr.unsafe_write(SkipListNode(MAX_LEVEL, 0.0, GenericValue()))
        self.head = head_ptr

    def release(mut self):
        """gh #369: free what this sorted set owns — member payloads held by
        the nodes, and every node slab (`reset()` keeps the first). The dict
        (`members`) and the allocator's bookkeeping Lists are freed by the
        caller's `unsafe_deinit_pointee()`. The list must not be used afterwards."""
        var node = self.head[].forward[0]
        while is_not_null(node):
            node[].obj.free_str_payload()
            node = node[].forward[0]
        self.allocator.release_all()
        self.length = 0
        self.level = 1

    def release_borrowed(mut self):
        """Unmap every node slab WITHOUT freeing the members the nodes hold.

        For a TEMPORARY list of members borrowed from other containers (the
        ZUNION/ZINTER/ZDIFF reply order). The slabs are mmap'd and the
        allocator has no destructor, so `unsafe_deinit_pointee()` alone leaked
        at least a page per call: 20K ZUNIONs grew RSS by ~1 GB
        (tests/test_soak_rss_bounded.py). release() would free the borrowed
        payloads — the owners' data."""
        self.allocator.release_all()
        self.length = 0
        self.level = 1

    def __moveinit__(out self, deinit take: Self):
        self.head = take.head
        self.level = take.level
        self.length = take.length
        self.allocator = take.allocator^
        self.prng = take.prng
        self.members = take.members^

    @always_inline
    def random_level(mut self) -> Int:
        var lvl = 1
        while (self.prng.next() & 0xFFFF) < (0xFFFF // 4) and lvl < MAX_LEVEL:
            lvl += 1
        return lvl

    def insert(mut self, score: Float64, var obj: GenericValue):
        # gh #187: plain insert is now an upsert — every caller (command handlers,
        # WAL replay, snapshot load, ZREM-style rebuilds) gets dedup for free.
        _ = self.upsert(score, obj^)

    @always_inline
    def member_score(self, obj: GenericValue) -> GenericValue:
        """FLOAT = live member at that score. NONE = not a live member.

        Needed by ZADD's conditional flags (gh #237), which must compare against
        the CURRENT score before deciding whether to write. See upsert's dict
        protocol note: an INT value is a popped-member sentinel, not a live
        member, so only FLOAT counts."""
        var e = self.members.get(obj)
        if e.type.value == ValueType.FLOAT:
            return e
        return GenericValue()

    def upsert(mut self, score: Float64, var obj: GenericValue) -> Int:
        """Redis ZADD semantics (gh #187): add a new member or move an existing
        member to the new score. Returns 1 for a new member, 0 for an update
        (including the equal-score no-op). Never leaves duplicate nodes.

        Dict protocol: FLOAT value = live member at that score. The dict key
        is its OWN copy of the member (freed when the entry is removed), the
        node holds another. Pops and removes delete the entry (gh #394); an INT
        value was the old popped-member sentinel and is still read as "not
        live" so a dict from before that change keeps working."""
        var existing = self.members.get(obj)
        var key_in_dict = not existing.is_none()
        var member_live = existing.type.value == ValueType.FLOAT
        if member_live:
            if existing.as_float() == score:
                # gh #394: `obj` is ours (var) and nothing keeps it on the
                # no-op path — every repeated ZADD of an unchanged member
                # leaked its heap copy.
                obj.free_str_payload()
                return 0
            self._remove_node(existing.as_float(), obj)
        # A stored dict key must own its payload independently of the node
        # (reset frees the stored key's buffer). Deep-copy heap STRINGs only
        # when the key will actually be stored (miss); on a hit set() keeps the
        # already-stored key and a deep copy here would leak. SSO has no heap.
        var dict_key = obj
        if not key_in_dict and obj.type.value == ValueType.STRING:
            dict_key = GenericValue.from_ptr(obj.as_string(), Int(obj._data1))
        self._insert_node(score, obj^)
        self.members.set(dict_key, GenericValue.from_float(score))
        return 0 if member_live else 1

    def _insert_node(mut self, score: Float64, var obj: GenericValue):
        # Stack-allocated update array — no heap alloc/free per insert.
        # gh #203: the `fill=` form zero-filled all 16 slots (128 B) per call
        # and every one of them is overwritten before it is read: the descent
        # below writes update[0 .. self.level), the promotion branch writes
        # update[self.level .. lvl), and the link loop reads only i < lvl.
        # Leaving it uninitialized is sound exactly as far as that holds — if
        # you add a read here, bound it by the same discipline.
        var update = Array[Pointer[SkipListNode, MutUntrackedOrigin], MAX_LEVEL](
            uninitialized=True)
        var curr = self.head

        # gh #251: the tie-break is load-bearing. This used to advance only
        # while `score < score`, so the descent stopped at the FIRST node of an
        # equal-score run and the new member was linked ahead of every existing
        # one — equal scores came back in reverse-insertion order where Redis
        # returns them lexicographically. `_remove_node` MUST use this exact
        # predicate; see the note there.
        for i in range(self.level - 1, -1, -1):
            while is_not_null(curr[].forward[i]) and (
                curr[].forward[i][].score < score
                or (curr[].forward[i][].score == score
                    and curr[].forward[i][].obj.lex_lt(obj))
            ):
                curr = curr[].forward[i]
            update[i] = curr

        var lvl = self.random_level()
        if lvl > self.level:
            for i in range(self.level, lvl):
                update[i] = self.head
            self.level = lvl

        var node_ptr = self.allocator.allocate()
        node_ptr.unsafe_write(SkipListNode(lvl, score, obj^))

        for i in range(lvl):
            node_ptr[].forward[i] = update[i][].forward[i]
            update[i][].forward[i] = node_ptr

        self.length += 1

    def _remove_node(mut self, score: Float64, member: GenericValue, free_obj: Bool = True):
        """Unlink and deallocate the node holding (score, member).

        gh #251: the descent predicate MUST match `_insert_node`'s exactly.
        Both now order an equal-score run by member (`lex_lt`), so the walk
        lands on the node's true predecessor at every level instead of
        scanning the whole run — a remover that ordered ties differently from
        the inserter would walk to the wrong place and silently fail to
        unlink, leaving a duplicate node the dict says was removed.

        The old predicate ("advance past every equal-score non-match") was
        correct under any intra-run order but linear in the run length, which
        is the worst case for exactly the workload this fix enables: a
        lex-index where every member shares score 0."""
        # gh #203: uninitialized is sound here for the same reason as in
        # _insert_node — the descent writes update[0 .. self.level) and the
        # unlink loop reads exactly i < self.level.
        var update = Array[Pointer[SkipListNode, MutUntrackedOrigin], MAX_LEVEL](
            uninitialized=True)
        var curr = self.head

        for i in range(self.level - 1, -1, -1):
            while is_not_null(curr[].forward[i]) and (
                curr[].forward[i][].score < score
                or (curr[].forward[i][].score == score
                    and curr[].forward[i][].obj.lex_lt(member))
            ):
                curr = curr[].forward[i]
            update[i] = curr

        var target = curr[].forward[0]
        if is_null(target) or target[].score != score or not (target[].obj == member):
            return
        for i in range(self.level):
            if update[i][].forward[i] == target:
                update[i][].forward[i] = target[].forward[i]
        # The dict holds its own copy of the member, so the node's payload is
        # unreferenced once unlinked — unless the caller takes it (pop_max).
        if free_obj:
            target[].obj.free_str_payload()
        self.allocator.deallocate(target)
        self.length -= 1

    def remove(mut self, member: GenericValue) -> Bool:
        """ZREM one member: O(log n) through the dict and one unlink, freeing
        the node's payload and the dict's own copy (gh #394). `member` may be
        borrowed — it is only compared. ZREM used to copy every node out,
        reset() the set and re-insert the survivors: O(n) per call, and it
        freed neither the removed members nor its own argument copies."""
        var e = self.members.get(member)
        if e.type.value != ValueType.FLOAT:
            return False
        self._remove_node(e.as_float(), member)
        _ = self.members.remove_generic(member)
        return True

    def pop_max(mut self) -> ZPopResult:
        """The highest-scored member, unlinked — O(log n). The returned obj is
        the node's own payload and now belongs to the caller (free it after the
        reply). ZPOPMAX used to rebuild the whole set per call and leak every
        popped member."""
        var curr = self.head
        for i in range(self.level - 1, -1, -1):
            while is_not_null(curr[].forward[i]):
                curr = curr[].forward[i]
        if curr == self.head:
            return ZPopResult()
        var score = curr[].score
        var obj = curr[].obj
        self._remove_node(score, obj, False)
        _ = self.members.remove_generic(obj)   # frees the dict's own copy, not obj
        return ZPopResult(score, obj, True)

    def get_range(self, min_score: Float64, max_score: Float64) -> List[GenericValue]:
        var res = List[GenericValue]()
        var curr = self.head[].forward[0]
        
        # Simple traversal for prototype
        while is_not_null(curr):
            if curr[].score >= min_score and curr[].score <= max_score:
                res.append(curr[].obj)
            elif curr[].score > max_score:
                break
            curr = curr[].forward[0]
            
        return res^

    def pop_min(mut self) -> ZPopResult:
        var first = self.head[].forward[0]
        if is_null(first):
            return ZPopResult()

        var score = first[].score
        var obj = first[].obj

        for i in range(self.level):
            if self.head[].forward[i] == first:
                self.head[].forward[i] = first[].forward[i]
            else:
                break

        # gh #394: REMOVE the dict entry (which frees the dict's own copy of
        # the member). gh #187 kept it as an INT "popped" sentinel because
        # set() could not then reuse DELETED tombstones and churn would spin
        # its probe loop; gh #210 made set() reuse them, and the sentinel made
        # the dict grow by one entry per DISTINCT member ever popped — a queue
        # of unique job ids never shrank. The returned obj is the node's own
        # payload and belongs to the caller: free it after the reply.
        _ = self.members.remove_generic(obj)
        self.allocator.deallocate(first)
        self.length -= 1
        return ZPopResult(score, obj, True)

    def deinit(owned self):
        # In a real system, we'd need to properly deallocate all SkipListNodes 
        # but here SlabAllocator.deinit will free all slabs at once.
        # However, SkipListNode has a 'forward' pointer that MUST be freed.
        # This is a limitation of current prototype: we need a way to call deinit 
        # on all allocated objects in slab.
        pass
