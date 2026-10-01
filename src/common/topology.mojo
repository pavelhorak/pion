from std.collections import Dict
import std.time

@fieldwise_init
struct OwnershipEntry(Copyable, Movable, ImplicitlyCopyable):
    var node_address: String
    var timestamp: Int

struct ClusterTopology:
    # Map of vNodeID -> OwnershipEntry
    var vnode_map: Dict[Int, OwnershipEntry]
    var version: Int

    def __init__(out self):
        self.vnode_map = Dict[Int, OwnershipEntry]()
        self.version = 0

    def update_ownership(mut self, vnode_id: Int, node_address: String, timestamp: Int):
        try:
            var existing = self.vnode_map[vnode_id]
            if timestamp > existing.timestamp:
                self.vnode_map[vnode_id] = OwnershipEntry(node_address, timestamp)
                self.version += 1
        except:
            self.vnode_map[vnode_id] = OwnershipEntry(node_address, timestamp)
            self.version += 1

    def get_owner(self, vnode_id: Int) -> String:
        try:
            return self.vnode_map[vnode_id].node_address
        except:
            return "UNKNOWN"

    def merge(mut self, other: ClusterTopology):
        # CRDT merge logic (LWW per vNode)
        for entry in other.vnode_map.items():
            var vnode_id = entry.key
            var other_val = entry.value
            self.update_ownership(vnode_id, other_val.node_address, other_val.timestamp)
