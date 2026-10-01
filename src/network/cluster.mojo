from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memcpy, unsafe_memset
from std.collections import Array
from src.common.utils import format_int_to_buf
from std.ffi import external_call


# gh #87.4: cluster wire-protocol port offsets, centralised.
# A Pion node listening on `port` exposes three auxiliary services derived
# from that base; they were previously hardcoded in 13 call sites across
# engine.mojo, replication.mojo, state.mojo, commands/cluster.mojo, config.mojo.
#
#   replication (TCP) → port + REPL_PORT_OFFSET
#   gossip      (UDP) → port + GOSSIP_PORT_OFFSET   (SWIM)
#   raft        (TCP) → port + RAFT_PORT_OFFSET
#
# Example: --port 1974 → replication 11974, gossip 21974, raft 31974.
comptime REPL_PORT_OFFSET: Int = 10000
comptime GOSSIP_PORT_OFFSET: Int = 20000
comptime RAFT_PORT_OFFSET: Int = 30000


@always_inline
def _hex_char(nibble: UInt8) -> UInt8:
    if nibble < 10:
        return UInt8(48) + nibble  # '0'..'9'
    return UInt8(87) + nibble  # 'a'..'f'


struct ClusterPeer(Movable):
    var node_id: Array[UInt8, 41]   # 40 hex chars + null terminator
    var host: Array[UInt8, 64]
    var host_len: Int
    var port: Int
    var slot_start: Int
    var slot_end: Int

    def __init__(out self):
        self.node_id = Array[UInt8, 41](fill=UInt8(48))  # '0'
        self.node_id[40] = 0
        self.host = Array[UInt8, 64](fill=UInt8(0))
        self.host_len = 0
        self.port = 0
        self.slot_start = 0
        self.slot_end = 0

    def __moveinit__(out self, owned take: Self):
        self.node_id = Array[UInt8, 41](uninitialized=True)
        for i in range(41):
            self.node_id[i] = take.node_id[i]
        self.host = Array[UInt8, 64](uninitialized=True)
        for i in range(64):
            self.host[i] = take.host[i]
        self.host_len = take.host_len
        self.port = take.port
        self.slot_start = take.slot_start
        self.slot_end = take.slot_end


struct ClusterState(Movable):
    var enabled: Bool
    var node_id: Array[UInt8, 41]   # 40 hex chars + null
    var my_host: Array[UInt8, 64]
    var my_host_len: Int
    var my_port: Int
    var my_slot_start: Int
    var my_slot_end: Int
    var peer_count: Int
    # peers stored as flat arrays to avoid Array[ClusterPeer] moveinit issues
    var peer_node_ids: Array[UInt8, 656]   # 16 × 41 bytes
    var peer_hosts: Array[UInt8, 1024]     # 16 × 64 bytes
    var peer_host_lens: Array[Int, 16]
    var peer_ports: Array[Int, 16]
    var peer_slot_starts: Array[Int, 16]
    var peer_slot_ends: Array[Int, 16]
    var crc16_table: Pointer[UInt16, MutUntrackedOrigin]  # 256 entries
    # Health: written by C gossip thread, read by Mojo event loop (per-tick)
    # Values: 0=online, 1=pfail (possible failure), 2=fail (confirmed failure)
    var peer_health: Array[UInt8, 16]
    # Replica mode: when is_replica=True, this node receives WAL from primary
    var is_replica: Bool
    var primary_peer_idx: Int    # index into peers[] of primary; -1 = not replica
    var cluster_epoch: UInt64
    # C1.2: Replication ID — 40-char hex, generated at startup (FNV-1a of node_id + epoch)
    var repl_id: Array[UInt8, 41]
    # §3: Slot migration state — per-slot MIGRATING/IMPORTING tracking
    # slot_migrating[slot] = peer_idx that is importing this slot (-1 = not migrating)
    # slot_importing[slot] = peer_idx that is exporting this slot (-1 = not importing)
    # Only the slots in [my_slot_start..my_slot_end] are relevant for this node.
    var slot_migrating: Pointer[Int16, MutUntrackedOrigin]  # [16384] = 32KB
    var slot_importing: Pointer[Int16, MutUntrackedOrigin]  # [16384] = 32KB
    # C1: Slot ownership bitset — 2048 bytes = 16384 bits. Replaces my_slot_start/my_slot_end
    # for non-contiguous slot ranges (e.g., after ADDSLOTS/DELSLOTS or mid-migration).
    var slot_owned: Pointer[UInt8, MutUntrackedOrigin]  # [2048] = 16384 bits
    # Opaque C handles (worker 0 sets these; other workers read cluster info via peer_health)
    var gossip_handle: Pointer[NoneType, MutUntrackedOrigin]
    var repl_primary_handle: Pointer[NoneType, MutUntrackedOrigin]
    var repl_replica_handle: Pointer[NoneType, MutUntrackedOrigin]
    # 4MB pre-allocated drain buffer for replica WAL apply (only used when is_replica=True)
    var repl_drain_buf: Pointer[UInt8, MutUntrackedOrigin]
    # N3: WAL pointer (for starting PrimaryReplicator after failover promotion)
    var wal_ptr: Pointer[NoneType, MutUntrackedOrigin]
    # N3: Server port (for replication port = port + 10000)
    var server_port: Int
    # gh #390: bytes at the front of repl_drain_buf that are the start of a
    # record the next drain completes. apply_wal_entries stopped at a record
    # split across two drains and the rest of that drain was simply lost.
    var repl_carry: Int
    var repl_drain_cap: Int

    def __init__(out self):
        self.enabled = False
        self.node_id = Array[UInt8, 41](fill=UInt8(48))
        self.node_id[40] = 0
        self.my_host = Array[UInt8, 64](fill=UInt8(0))
        self.my_host_len = 0
        self.my_port = 1974
        self.my_slot_start = 0
        self.my_slot_end = 16383
        self.peer_count = 0
        self.peer_node_ids = Array[UInt8, 656](fill=UInt8(48))
        self.peer_hosts = Array[UInt8, 1024](fill=UInt8(0))
        self.peer_host_lens = Array[Int, 16](fill=Int(0))
        self.peer_ports = Array[Int, 16](fill=Int(0))
        self.peer_slot_starts = Array[Int, 16](fill=Int(0))
        self.peer_slot_ends = Array[Int, 16](fill=Int(0))
        # Init CRC16 table (CCITT, polynomial 0x1021, seed 0)
        var crc_alloc = alloc[UInt16](256)
        self.crc16_table = crc_alloc
        for i in range(256):
            var c = UInt16(i) << 8
            for _ in range(8):
                if (c & 0x8000) != 0:
                    c = (c << 1) ^ 0x1021
                else:
                    c = c << 1
            self.crc16_table[unsafe_offset=i] = c
        self.peer_health = Array[UInt8, 16](fill=UInt8(0))
        self.is_replica = False
        self.primary_peer_idx = -1
        self.cluster_epoch = 1
        # C1.2: Replication ID defaults to "?" — set to real ID after generate_node_id
        self.repl_id = Array[UInt8, 41](fill=UInt8(48))
        self.repl_id[0] = 63  # '?'
        self.repl_id[1] = 0
        self.slot_migrating = alloc[Int16](16384)
        self.slot_importing = alloc[Int16](16384)
        for si in range(16384):
            self.slot_migrating[unsafe_offset=si] = Int16(-1)
            self.slot_importing[unsafe_offset=si] = Int16(-1)
        # C1: Slot ownership bitset — default: own all 16384 slots (all bits set)
        self.slot_owned = alloc[UInt8](2048)
        unsafe_memset(self.slot_owned, 0xFF, 2048)
        self.gossip_handle = null_ptr[NoneType, MutUntrackedOrigin]()
        self.repl_primary_handle = null_ptr[NoneType, MutUntrackedOrigin]()
        self.repl_replica_handle = null_ptr[NoneType, MutUntrackedOrigin]()
        self.repl_drain_buf = null_ptr[UInt8, MutUntrackedOrigin]()
        self.wal_ptr = null_ptr[NoneType, MutUntrackedOrigin]()
        self.repl_carry = 0
        self.repl_drain_cap = 4194304
        self.server_port = 1974

    def __moveinit__(out self, deinit take: Self):
        self.enabled = take.enabled
        self.node_id = Array[UInt8, 41](uninitialized=True)
        for i in range(41):
            self.node_id[i] = take.node_id[i]
        self.my_host = Array[UInt8, 64](uninitialized=True)
        for i in range(64):
            self.my_host[i] = take.my_host[i]
        self.my_host_len = take.my_host_len
        self.my_port = take.my_port
        self.my_slot_start = take.my_slot_start
        self.my_slot_end = take.my_slot_end
        self.peer_count = take.peer_count
        self.peer_node_ids = Array[UInt8, 656](uninitialized=True)
        for i in range(656):
            self.peer_node_ids[i] = take.peer_node_ids[i]
        self.peer_hosts = Array[UInt8, 1024](uninitialized=True)
        for i in range(1024):
            self.peer_hosts[i] = take.peer_hosts[i]
        self.peer_host_lens = Array[Int, 16](uninitialized=True)
        for i in range(16):
            self.peer_host_lens[i] = take.peer_host_lens[i]
        self.peer_ports = Array[Int, 16](uninitialized=True)
        for i in range(16):
            self.peer_ports[i] = take.peer_ports[i]
        self.peer_slot_starts = Array[Int, 16](uninitialized=True)
        for i in range(16):
            self.peer_slot_starts[i] = take.peer_slot_starts[i]
        self.peer_slot_ends = Array[Int, 16](uninitialized=True)
        for i in range(16):
            self.peer_slot_ends[i] = take.peer_slot_ends[i]
        self.crc16_table = take.crc16_table
        self.peer_health = Array[UInt8, 16](uninitialized=True)
        for i in range(16):
            self.peer_health[i] = take.peer_health[i]
        self.is_replica = take.is_replica
        self.primary_peer_idx = take.primary_peer_idx
        self.cluster_epoch = take.cluster_epoch
        self.repl_id = Array[UInt8, 41](uninitialized=True)
        for i in range(41):
            self.repl_id[i] = take.repl_id[i]
        self.slot_migrating = take.slot_migrating
        self.slot_importing = take.slot_importing
        self.slot_owned = take.slot_owned
        self.gossip_handle = take.gossip_handle
        self.repl_primary_handle = take.repl_primary_handle
        self.repl_replica_handle = take.repl_replica_handle
        self.repl_drain_buf = take.repl_drain_buf
        self.wal_ptr = take.wal_ptr
        self.repl_carry = take.repl_carry
        self.repl_drain_cap = take.repl_drain_cap
        self.server_port = take.server_port

    def __del__(deinit self):
        if is_not_null(self.crc16_table):
            self.crc16_table.unsafe_free()
        if is_not_null(self.slot_migrating):
            self.slot_migrating.unsafe_free()
        if is_not_null(self.slot_importing):
            self.slot_importing.unsafe_free()
        if is_not_null(self.slot_owned):
            self.slot_owned.unsafe_free()
        if is_not_null(self.repl_drain_buf):
            self.repl_drain_buf.unsafe_free()

    @always_inline
    def count_pfail(self) -> Int:
        """Count peers in pfail state (health == 1)."""
        var n = 0
        for i in range(self.peer_count):
            if self.peer_health[i] == 1: n += 1
        return n

    @always_inline
    def count_fail(self) -> Int:
        """Count peers in fail state (health == 2)."""
        var n = 0
        for i in range(self.peer_count):
            if self.peer_health[i] == 2: n += 1
        return n

    @always_inline
    def peer_is_connected(self, peer_idx: Int) -> Bool:
        """True if peer health is online (0) or pfail (1); False if fail (2)."""
        if peer_idx < 0 or peer_idx >= self.peer_count: return False
        return self.peer_health[peer_idx] != 2

    @always_inline
    def crc16(self, ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> UInt16:
        var crc: UInt16 = 0
        for i in range(length):
            crc = (crc << 8) ^ self.crc16_table[unsafe_offset=Int(((crc >> 8) ^ UInt16(ptr[unsafe_offset=i])) & 0xFF)]
        return crc

    @always_inline
    def keyslot(self, key_ptr: Pointer[UInt8, MutUntrackedOrigin], key_len: Int) -> Int:
        """Compute Redis hash slot (0-16383) for a key. Handles {hashtag} extraction."""
        var brace_start = -1
        var brace_end = -1
        for i in range(key_len):
            if key_ptr[unsafe_offset=i] == UInt8(123) and brace_start < 0:  # '{'
                brace_start = i + 1
            elif key_ptr[unsafe_offset=i] == UInt8(125) and brace_start >= 0:  # '}'
                brace_end = i
                break
        if brace_start >= 0 and brace_end > brace_start:
            return Int(self.crc16(key_ptr.unsafe_offset(brace_start), brace_end - brace_start) % 16384)
        return Int(self.crc16(key_ptr, key_len) % 16384)

    @always_inline
    def owns_slot(self, slot: Int) -> Bool:
        """Check if this node owns the given slot using the bitset."""
        if slot < 0 or slot >= 16384:
            return False
        return (self.slot_owned[unsafe_offset=slot >> 3] & UInt8(1 << (slot & 7))) != 0

    @always_inline
    def add_slot(mut self, slot: Int):
        """Mark slot as owned by this node."""
        if slot >= 0 and slot < 16384:
            self.slot_owned[unsafe_offset=slot >> 3] = self.slot_owned[unsafe_offset=slot >> 3] | UInt8(1 << (slot & 7))
            # Update legacy range bounds for backward compat
            if slot < self.my_slot_start: self.my_slot_start = slot
            if slot > self.my_slot_end: self.my_slot_end = slot

    @always_inline
    def del_slot(mut self, slot: Int):
        """Remove slot ownership from this node."""
        if slot >= 0 and slot < 16384:
            self.slot_owned[unsafe_offset=slot >> 3] = self.slot_owned[unsafe_offset=slot >> 3] & ~UInt8(1 << (slot & 7))

    def count_owned_slots(self) -> Int:
        """Count the number of slots owned by this node."""
        var count = 0
        for i in range(2048):
            var byte = self.slot_owned[i]
            while byte != 0:
                count += Int(byte & 1)
                byte >>= 1
        return count

    def find_peer_for_slot(self, slot: Int) -> Int:
        """Return peer index [0..peer_count-1] that owns slot, or -1 if unknown."""
        for i in range(self.peer_count):
            if slot >= self.peer_slot_starts[i] and slot <= self.peer_slot_ends[i]:
                return i
        return -1

    def generate_repl_id(mut self):
        """C1.2: Generate 40-char hex replication ID from node_id + epoch."""
        var h1: UInt64 = 14695981039346656037
        var h2: UInt64 = 0xcbf29ce484222325
        for i in range(40):
            h1 = (h1 ^ UInt64(self.node_id[i])) * 1099511628211
            h2 = (h2 ^ UInt64(self.node_id[i])) * 1099511628211
        h1 = (h1 ^ self.cluster_epoch) * 1099511628211
        h2 = (h2 ^ (self.cluster_epoch + 1)) * 1099511628211
        for i in range(8):
            var b1 = UInt8((h1 >> (UInt64(i) * 8)) & 0xFF)
            var b2 = UInt8((h2 >> (UInt64(i) * 8)) & 0xFF)
            self.repl_id[i * 2]     = _hex_char(b1 >> 4)
            self.repl_id[i * 2 + 1] = _hex_char(b1 & 0xF)
            self.repl_id[16 + i * 2]     = _hex_char(b2 >> 4)
            self.repl_id[16 + i * 2 + 1] = _hex_char(b2 & 0xF)
        for i in range(32, 40):
            self.repl_id[i] = UInt8(48)
        self.repl_id[40] = 0

    def set_my_host(mut self, host: String):
        var hlen = min(host.byte_length(), 63)
        for i in range(hlen):
            self.my_host[i] = host.unsafe_ptr()[unsafe_offset=i]
        self.my_host[hlen] = 0
        self.my_host_len = hlen

    def generate_node_id(mut self, host_ptr: Pointer[UInt8, MutUntrackedOrigin], host_len: Int, port: Int):
        """Generate deterministic 40-char hex node ID from host:port using two-pass FNV-1a."""
        var h1: UInt64 = 14695981039346656037
        var h2: UInt64 = 0xcbf29ce484222325
        for i in range(host_len):
            h1 = (h1 ^ UInt64(host_ptr[unsafe_offset=i])) * 1099511628211
            h2 = (h2 ^ UInt64(host_ptr[unsafe_offset=i])) * 1099511628211
        h1 = (h1 ^ UInt64(port)) * 1099511628211
        h2 = (h2 ^ UInt64(port + 1)) * 1099511628211
        for i in range(8):
            var b1 = UInt8((h1 >> (UInt64(i) * 8)) & 0xFF)
            var b2 = UInt8((h2 >> (UInt64(i) * 8)) & 0xFF)
            self.node_id[i * 2]     = _hex_char(b1 >> 4)
            self.node_id[i * 2 + 1] = _hex_char(b1 & 0xF)
            self.node_id[16 + i * 2]     = _hex_char(b2 >> 4)
            self.node_id[16 + i * 2 + 1] = _hex_char(b2 & 0xF)
        for i in range(32, 40):
            self.node_id[i] = UInt8(48)  # '0'
        self.node_id[40] = 0

    def set_peer(mut self, peer_idx: Int, host_ptr: Pointer[UInt8, MutUntrackedOrigin], host_len: Int, port: Int, slot_start: Int, slot_end: Int):
        """Set peer info at peer_idx and generate its node ID."""
        if peer_idx >= 16: return
        # Copy host
        var ph_len = min(host_len, 63)
        for i in range(ph_len):
            self.peer_hosts[peer_idx * 64 + i] = host_ptr[unsafe_offset=i]
        self.peer_hosts[peer_idx * 64 + ph_len] = 0
        self.peer_host_lens[peer_idx] = ph_len
        self.peer_ports[peer_idx] = port
        self.peer_slot_starts[peer_idx] = slot_start
        self.peer_slot_ends[peer_idx] = slot_end
        # Generate node ID into peer_node_ids[peer_idx * 41 .. +41]
        var h1: UInt64 = 14695981039346656037
        var h2: UInt64 = 0xcbf29ce484222325
        for i in range(ph_len):
            h1 = (h1 ^ UInt64(host_ptr[unsafe_offset=i])) * 1099511628211
            h2 = (h2 ^ UInt64(host_ptr[unsafe_offset=i])) * 1099511628211
        h1 = (h1 ^ UInt64(port)) * 1099511628211
        h2 = (h2 ^ UInt64(port + 1)) * 1099511628211
        var base = peer_idx * 41
        for i in range(8):
            var b1 = UInt8((h1 >> (UInt64(i) * 8)) & 0xFF)
            var b2 = UInt8((h2 >> (UInt64(i) * 8)) & 0xFF)
            self.peer_node_ids[base + i * 2]     = _hex_char(b1 >> 4)
            self.peer_node_ids[base + i * 2 + 1] = _hex_char(b1 & 0xF)
            self.peer_node_ids[base + 16 + i * 2]     = _hex_char(b2 >> 4)
            self.peer_node_ids[base + 16 + i * 2 + 1] = _hex_char(b2 & 0xF)
        for i in range(32, 40):
            self.peer_node_ids[base + i] = UInt8(48)
        self.peer_node_ids[base + 40] = 0

    @always_inline
    def peer_node_id_ptr(self, peer_idx: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self.peer_node_ids.unsafe_ptr())).unsafe_offset(peer_idx * 41)

    @always_inline
    def peer_host_ptr(self, peer_idx: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self.peer_hosts.unsafe_ptr())).unsafe_offset(peer_idx * 64)

    @always_inline
    def is_slot_migrating(self, slot: Int) -> Bool:
        """True if this slot is being migrated away from this node."""
        return slot >= 0 and slot < 16384 and self.slot_migrating[slot] >= 0

    @always_inline
    def is_slot_importing(self, slot: Int) -> Bool:
        """True if this slot is being imported to this node."""
        return slot >= 0 and slot < 16384 and self.slot_importing[unsafe_offset=slot] >= 0

    def set_slot_migrating(mut self, slot: Int, target_peer: Int):
        """Mark slot as MIGRATING to target_peer. Source sends -ASK for missing keys."""
        if slot >= 0 and slot < 16384:
            self.slot_migrating[unsafe_offset=slot] = Int16(target_peer)

    def set_slot_importing(mut self, slot: Int, source_peer: Int):
        """Mark slot as IMPORTING from source_peer. Target accepts ASKING-prefixed commands."""
        if slot >= 0 and slot < 16384:
            self.slot_importing[unsafe_offset=slot] = Int16(source_peer)

    def clear_slot_state(mut self, slot: Int):
        """Clear migration state for a slot (STABLE)."""
        if slot >= 0 and slot < 16384:
            self.slot_migrating[unsafe_offset=slot] = Int16(-1)
            self.slot_importing[unsafe_offset=slot] = Int16(-1)

    def assign_slot_to_node(mut self, slot: Int, node_id_ptr: Pointer[UInt8, MutUntrackedOrigin]):
        """Assign a slot to a specific node. If it's us, expand our range. Otherwise, update peer."""
        # Check if it's our node ID
        var is_me = True
        for i in range(40):
            if node_id_ptr[unsafe_offset=i] != self.node_id[i]:
                is_me = False
                break
        if is_me:
            self.add_slot(slot)
        else:
            # Find matching peer and update their range
            for pi in range(self.peer_count):
                var found = True
                var base = pi * 41
                for ci in range(40):
                    if Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self.peer_node_ids.unsafe_ptr()))[unsafe_offset=base + ci] != node_id_ptr[unsafe_offset=ci]:
                        found = False
                        break
                if found:
                    if slot < self.peer_slot_starts[pi]: self.peer_slot_starts[pi] = slot
                    if slot > self.peer_slot_ends[pi]: self.peer_slot_ends[pi] = slot
                    break
        self.clear_slot_state(slot)
        self.cluster_epoch += 1

    def save_topology(self, path: String):
        """Save cluster topology to pion-nodes.conf for auto-rejoin on restart."""
        var cpath = path
        var fd = external_call["pion_creat", Int32](cpath.unsafe_ptr())
        if fd < 0: return
        # Write self
        var line = String("")
        for i in range(40):
            line += chr(Int(self.node_id[i]))
        line += " "
        for i in range(self.my_host_len):
            line += chr(Int(self.my_host[i]))
        line += ":" + String(self.my_port) + " myself,"
        line += "master" if not self.is_replica else "slave"
        line += " - 0 0 " + String(self.cluster_epoch) + " connected "
        line += String(self.my_slot_start) + "-" + String(self.my_slot_end) + "\n"
        _ = external_call["pion_write", Int](fd, line.unsafe_ptr(), line.byte_length())
        # Write peers
        for pi in range(self.peer_count):
            var pline = String("")
            var pbase = pi * 41
            for i in range(40):
                pline += chr(Int(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self.peer_node_ids.unsafe_ptr()))[unsafe_offset=pbase + i]))
            pline += " "
            for i in range(self.peer_host_lens[pi]):
                pline += chr(Int(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self.peer_hosts.unsafe_ptr()))[unsafe_offset=pi * 64 + i]))
            pline += ":" + String(self.peer_ports[pi]) + " "
            var health = self.peer_health[pi]
            if health == 0: pline += "master"
            elif health == 1: pline += "master,pfail"
            else: pline += "master,fail"
            pline += " - 0 0 " + String(self.cluster_epoch) + " connected "
            pline += String(self.peer_slot_starts[pi]) + "-" + String(self.peer_slot_ends[pi]) + "\n"
            _ = external_call["pion_write", Int](fd, pline.unsafe_ptr(), pline.byte_length())
        _ = external_call["close", Int32](fd)
