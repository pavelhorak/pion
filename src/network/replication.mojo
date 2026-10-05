"""Replication — WAL streaming for primary→replica clusters.

PrimaryReplicator: listens for replica connections, streams new WAL bytes.
ReplicaReceiver:   connects to primary, buffers received WAL bytes in a C ring.
                   Mojo drains the ring each event-loop tick via drain_to_buf().

Replication port = main port + 10000 (e.g., 11974 for port 1974).
Only worker 0 runs these; all workers share the same keyspace pointer.
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.container_free import remove_and_free
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy
from std.ffi import external_call

from src.io.wal import (WAL, WAL_HEADER_SIZE, wal_apply_mset, wal_is_aggregate,
                        wal_apply_aggregate, wal_apply_ttl)
from src.common.hash_map import StripedHashMap, SlabHashMap
from src.common.value import GenericValue


struct PrimaryReplicator(Movable):
    """Wraps C PionReplPrimary. Accepts replica connections, streams WAL."""
    var handle: Pointer[NoneType, MutUntrackedOrigin]

    def __init__(out self):
        self.handle = null_ptr[NoneType, MutUntrackedOrigin]()

    def __moveinit__(out self, deinit take: Self):
        self.handle = take.handle

    def __del__(deinit self):
        if is_not_null(self.handle):
            external_call["pion_repl_primary_stop", NoneType](self.handle)

    def setup(mut self, wal: Pointer[WAL, MutUntrackedOrigin], listen_port: Int) -> Bool:
        """Create and start the primary listener. listen_port = main_port + 10000."""
        if is_null(wal) or is_null(wal[].map):
            return False
        # WAL data section starts after the 64-byte header
        var data_ptr = wal[].map.unsafe_offset(WAL_HEADER_SIZE)
        var tail_ptr = alloc[UInt64](1)  # we pass a pointer that C reads each poll
        # We don't want to alloc a new UInt64 — pass the address of wal.tail_offset directly.
        # Since tail_offset is a field in the WAL struct (heap-allocated), its address is stable.
        # Reconstruct pointer to tail_offset: WAL layout has tail_offset at byte offset 8 in struct.
        # Safest: just pass data_ptr and let C read from WAL map header byte offset 8.
        # The WAL header layout: [8B magic][8B tail_offset][8B worker_id][8B seq][32B reserved]
        # → tail_offset is at wal.map + 8
        tail_ptr.unsafe_free()
        var wal_tail_in_header = wal[].map.unsafe_bitcast[UInt64]().unsafe_offset(1)
        var blk = external_call["pion_repl_primary_create",
                                 Pointer[NoneType, MutUntrackedOrigin]](
            data_ptr,
            wal_tail_in_header,
            Int32(listen_port),
        )
        if is_null(blk):
            return False
        self.handle = blk
        var rc = external_call["pion_repl_primary_start", Int32](blk)
        return rc == 0

    def connected_count(self) -> Int:
        if is_null(self.handle):
            return 0
        return Int(external_call["pion_repl_primary_connected_count", Int32](self.handle))

    def max_sent_offset(self) -> Int:
        """Return the highest WAL offset sent to any replica."""
        if is_null(self.handle):
            return 0
        return Int(external_call["pion_repl_primary_max_sent_offset", Int64](self.handle))

    def set_repl_id(self, repl_id: String):
        """C1.2: Set replication ID (40-char hex). Call before start()."""
        if is_null(self.handle):
            return
        var rid = repl_id
        external_call["pion_repl_primary_set_repl_id", NoneType](self.handle, rid.as_c_string_slice())

    def acked_count(self, target_offset: Int) -> Int:
        """C1.2: Number of replicas that have ACKed at least target_offset."""
        if is_null(self.handle):
            return 0
        return Int(external_call["pion_repl_primary_acked_count", Int32](self.handle, UInt64(target_offset)))

    def set_snapshot(self, buf: Pointer[UInt8, MutUntrackedOrigin], length: Int):
        """C1.2: Set snapshot buffer for FULLRESYNC. Ownership stays with caller."""
        if is_null(self.handle):
            return
        external_call["pion_repl_primary_set_snapshot", NoneType](self.handle, buf, UInt64(length))


struct ReplicaReceiver(Movable):
    """Wraps C PionReplReplicaBlock. Connects to primary, buffers incoming WAL bytes.
    drain_to_buf() copies available bytes to a pre-allocated Mojo buffer for apply."""
    var handle: Pointer[NoneType, MutUntrackedOrigin]

    def __init__(out self):
        self.handle = null_ptr[NoneType, MutUntrackedOrigin]()

    def __moveinit__(out self, deinit take: Self):
        self.handle = take.handle

    def __del__(deinit self):
        if is_not_null(self.handle):
            external_call["pion_repl_replica_stop", NoneType](self.handle)

    def setup(mut self, primary_host: String, primary_repl_port: Int) -> Bool:
        """Create + start the replica receiver thread."""
        var ph = primary_host
        var blk = external_call["pion_repl_replica_create",
                                 Pointer[NoneType, MutUntrackedOrigin]](
            ph.as_c_string_slice(),
            Int32(primary_repl_port),
        )
        if is_null(blk):
            return False
        self.handle = blk
        var rc = external_call["pion_repl_replica_start", Int32](blk)
        return rc == 0

    def drain_to_buf(
        self,
        out_buf: Pointer[UInt8, MutUntrackedOrigin],
        max_bytes: Int,
    ) -> Int:
        """Drain buffered WAL bytes into out_buf. Returns byte count (0 = nothing new)."""
        if is_null(self.handle):
            return 0
        return Int(external_call["pion_repl_replica_drain", Int32](
            self.handle, out_buf, Int32(max_bytes)
        ))

    def bytes_received(self) -> Int:
        """Return total bytes received from primary (monotonic counter)."""
        if is_null(self.handle):
            return 0
        return Int(external_call["pion_repl_replica_bytes_received", Int64](self.handle))

    def repl_offset(self) -> Int:
        """C1.2: Current replication offset (for PSYNC reconnect)."""
        if is_null(self.handle):
            return 0
        return Int(external_call["pion_repl_replica_offset", Int64](self.handle))


@always_inline
def apply_wal_entries(
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
    buf: Pointer[UInt8, MutUntrackedOrigin],
    buf_len: Int,
    ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] = null_ptr[SlabHashMap, MutUntrackedOrigin](),
) -> Int:
    """Parse and apply WAL entries from a raw byte buffer (replicated from primary).
    Entry format: [4B entry_len LE][1B cmd_id][4B key_len LE][key][4B val_len LE][val]
    Skips malformed entries.

    This is the THIRD decoder of the WAL (after replay and the snapshot
    loader) and it must dispatch every record kind they do. It used to apply
    only 1 SET / 2 DEL / 31 MSET: the primary streams its raw WAL, so every
    hash, list, set, zset, stream, HLL, bitmap, vector-set and TTL record was
    silently dropped on every replica — and CLUSTER FAILOVER promoted a node
    holding only the string keys. Now aggregates go through the shared
    wal_apply_aggregate (wal_is_aggregate is the one predicate all three use)
    and TTL records through wal_apply_ttl.

    Returns the bytes consumed: every WHOLE record. A record split across two
    drains stays for the caller to carry into the next one (gh #390 — the
    tail of a drain used to be dropped). Cmd 250 is the replica thread's
    in-band FLUSH ahead of a FULLRESYNC snapshot."""
    var off = 0
    while off + 13 <= buf_len:  # minimum entry: 4+1+4+0+4+0 = 13
        var entry_len = (Int(buf[unsafe_offset=off])
                       | (Int(buf[unsafe_offset=off+1]) << 8)
                       | (Int(buf[unsafe_offset=off+2]) << 16)
                       | (Int(buf[unsafe_offset=off+3]) << 24))
        if entry_len < 13:
            # Not a record boundary: the stream is corrupt. Carrying it would
            # stall the replica for good, so drop the rest and say so.
            print("Replication: corrupt record length " + String(entry_len)
                  + " in the stream — dropping " + String(buf_len - off) + " bytes")
            return buf_len
        if off + entry_len > buf_len:
            break       # split across drains: the caller carries it
        var cmd_id = Int(buf[unsafe_offset=off + 4])
        var key_len = (Int(buf[unsafe_offset=off+5])
                     | (Int(buf[unsafe_offset=off+6]) << 8)
                     | (Int(buf[unsafe_offset=off+7]) << 16)
                     | (Int(buf[unsafe_offset=off+8]) << 24))
        var key_off = off + 9
        if key_len < 0 or key_off + key_len + 4 > off + entry_len:
            off += entry_len; continue
        var val_len_off = key_off + key_len
        var val_len = (Int(buf[unsafe_offset=val_len_off])
                     | (Int(buf[unsafe_offset=val_len_off+1]) << 8)
                     | (Int(buf[unsafe_offset=val_len_off+2]) << 16)
                     | (Int(buf[unsafe_offset=val_len_off+3]) << 24))
        var val_off = val_len_off + 4
        if val_len < 0 or val_off + val_len > off + entry_len:
            off += entry_len; continue

        var key_ptr = buf.unsafe_offset(key_off)

        if cmd_id == 1:  # SET — the key borrows the drain buffer; set() copies on insert
            keyspace[].set(GenericValue.borrow(key_ptr, key_len),
                           GenericValue.from_ptr(buf.unsafe_offset(val_off), val_len))
        elif cmd_id == 2:  # DEL
            _ = remove_and_free(keyspace, GenericValue.borrow(key_ptr, key_len))   # gh #394
        elif cmd_id == 250:  # FULLRESYNC: drop everything, the snapshot follows
            keyspace[].reset()           # aggregates go to the graveyard (gh #394)
            if is_not_null(ttl_map):
                ttl_map[].reset()
        elif cmd_id >= 35 and cmd_id <= 37:   # #36 FUNCTION LOAD / DELETE / FLUSH
            _ = external_call["pion_lua_wal_apply", Int64](
                Int64(cmd_id), key_ptr, Int64(key_len), buf.unsafe_offset(val_off), Int64(val_len))
        elif cmd_id == 25 or cmd_id == 26:   # gh #174 TTL records (EXPIREAT / PERSIST)
            _ = wal_apply_ttl(UInt8(cmd_id), key_ptr, key_len,
                              buf.unsafe_offset(val_off), val_len, ttl_map)
        elif wal_is_aggregate(UInt8(cmd_id)):   # 5-24, 27-34 incl. 31 MSET, 34 XADD
            _ = wal_apply_aggregate(UInt8(cmd_id), key_ptr, key_len,
                                    buf.unsafe_offset(val_off), val_len, keyspace)
        elif cmd_id == 4:  # gh #163 blob pointer — meaningless on a replica
            # The blob tier is switched off wherever replication is configured,
            # so this can only be pre-existing log content on a node promoted
            # after running standalone. Say so: silently skipping is how a
            # replica ends up quietly missing keys the primary has.
            print("Replication: skipped a blob-pointer WAL entry (cmd 4) — the "
                  + "arena bytes are not in the stream; this key is NOT replicated")

        off += entry_len
    return off
