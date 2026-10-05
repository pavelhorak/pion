"""StreamEntry / StreamData — the storage layer for Redis streams.

gh #174: lifted out of `src/commands/stream.mojo` so `src/io/wal.mojo` can
replay XADD records without importing the command module. The command module
pulls in `fast_path` (for `_get_now_ns`) and `fast_path` pulls in `wal`, so a
direct import would close the cycle wal → stream → fast_path → wal. These two
structs are pure data with no protocol or engine dependencies, which is what
makes `src/common/` the right home — the same reason `SlabList` and
`SlabSkipList` live there rather than beside their handlers.

`stream.mojo` re-exports both names, so every existing `from src.commands.stream
import StreamData` keeps working.
"""
from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc


# ── StreamEntry: one entry in the stream ──
struct StreamEntry(Copyable, Movable, ImplicitlyCopyable):
    var id_ms: UInt64
    var id_seq: UInt64
    var data: Pointer[UInt8, MutUntrackedOrigin]  # packed field-value pairs: [u32 flen][bytes][u32 vlen][bytes]...
    var data_len: Int
    var num_fields: Int
    var deleted: Bool

    def __init__(out self):
        self.id_ms = 0; self.id_seq = 0
        self.data = null_ptr[UInt8, MutUntrackedOrigin]()
        self.data_len = 0; self.num_fields = 0; self.deleted = False


# ── StreamData: the stream itself ──
struct StreamData(Copyable, Movable):
    var entries: Pointer[StreamEntry, MutUntrackedOrigin]
    var count: Int
    var alive: Int
    var capacity: Int
    var last_id_ms: UInt64
    var last_id_seq: UInt64

    def __init__(out self):
        self.capacity = 64
        self.entries = alloc[StreamEntry](self.capacity)
        self.count = 0; self.alive = 0
        self.last_id_ms = 0; self.last_id_seq = 0

    def append(mut self, id_ms: UInt64, id_seq: UInt64, data: Pointer[UInt8, MutUntrackedOrigin], data_len: Int, num_fields: Int):
        if self.count >= self.capacity:
            var new_cap = self.capacity * 2
            var new_entries = alloc[StreamEntry](new_cap)
            for i in range(self.count):
                (new_entries.unsafe_offset(i)).unsafe_write(self.entries[unsafe_offset=i])
            self.entries.unsafe_free()
            self.entries = new_entries
            self.capacity = new_cap
        var e = StreamEntry()
        e.id_ms = id_ms; e.id_seq = id_seq
        e.data = data; e.data_len = data_len
        e.num_fields = num_fields; e.deleted = False
        (self.entries.unsafe_offset(self.count)).unsafe_write(e)
        self.count += 1; self.alive += 1
        self.last_id_ms = id_ms; self.last_id_seq = id_seq

    def kill(mut self, i: Int):
        """Delete entry `i` (XDEL, XTRIM, XADD MAXLEN, their replays) and free
        its field data (gh #394). Entries used to be only FLAGGED deleted, so a
        capped stream kept every trimmed entry's data forever — a MAXLEN log
        grew without bound — and never shrank `entries` either."""
        if self.entries[unsafe_offset=i].deleted:
            return
        self.entries[unsafe_offset=i].deleted = True
        var d = self.entries[unsafe_offset=i].data
        if Int(d) != 0:
            d.unsafe_free()
        self.entries[unsafe_offset=i].data = null_ptr[UInt8, MutUntrackedOrigin]()
        self.alive -= 1

    def compact(mut self):
        """Squeeze out deleted entries once they are the majority, keeping the
        order. Call after a batch of kill()s; nothing holds an entry INDEX
        across commands (readers resume from an id), so moving is safe. Keeps
        a capped stream's `entries` bounded and its trim scans short — the
        MAXLEN trim walks from index 0 over every tombstone ever made."""
        var dead = self.count - self.alive
        if dead < 64 or dead * 2 < self.count:
            return
        var w = 0
        for r in range(self.count):
            if not self.entries[unsafe_offset=r].deleted:
                if w != r:
                    self.entries[unsafe_offset=w] = self.entries[unsafe_offset=r]
                w += 1
        self.count = w

    def release(mut self):
        """Free everything this stream owns — for DEL and every other path that
        drops a stream key (gh #394: none did, ~3 KB per stream)."""
        for i in range(self.count):
            var d = self.entries[unsafe_offset=i].data
            if Int(d) != 0:
                d.unsafe_free()
        if Int(self.entries) != 0:
            self.entries.unsafe_free()
        self.entries = null_ptr[StreamEntry, MutUntrackedOrigin]()
        self.count = 0; self.alive = 0; self.capacity = 0

    def deep_copy(self) -> Self:
        """A stream sharing nothing with this one — for COPY. Live entries only;
        the id high-water mark is kept, so new ids stay monotonic."""
        var out = Self()
        for i in range(self.count):
            var e = self.entries[unsafe_offset=i]
            if e.deleted:
                continue
            var d = alloc[UInt8](max(e.data_len, 1))
            for b in range(e.data_len):
                d[unsafe_offset=b] = e.data[unsafe_offset=b]
            out.append(e.id_ms, e.id_seq, d, e.data_len, e.num_fields)
        out.last_id_ms = self.last_id_ms
        out.last_id_seq = self.last_id_seq
        return out^
