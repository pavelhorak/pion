from std.collections import Dict
from .vnode import VNode
from src.common.rendezvous_hash import RendezvousHash

struct ShardManager:
    var shards: List[String]
    var hasher: RendezvousHash
    var vnode_loads: Dict[Int, Int]

    def __init__(out self, var nodes: List[String]):
        # We need to copy because we pass it to hasher too
        var nodes_copy = nodes.copy()
        self.shards = nodes^
        self.hasher = RendezvousHash(nodes_copy^)
        self.vnode_loads = Dict[Int, Int]()

    def get_node_for_key(self, key: String) -> String:
        return self.hasher.get_node(key)

    def get_vnode_id(self, key: String) -> Int:
        return Int(hash(key) % 4096)

    def record_load(mut self, vnode_id: Int, ops: Int):
        try:
            var current_load = self.vnode_loads[vnode_id]
            self.vnode_loads[vnode_id] = current_load + ops
        except:
            self.vnode_loads[vnode_id] = ops

    def rebalance(mut self, threshold: Int):
        print("ShardManager: Evaluating cluster balance...")
        for entry in self.vnode_loads.items():
            var vnode_id = entry.key
            var load = entry.value
            if load > threshold:
                print("ShardManager: Hotspot detected on vNode " + String(vnode_id) + " (Load: " + String(load) + "). Triggering migration.")
                # In a real system, this would calculate the least loaded node
                # and trigger the VNode state machine (ACTIVE -> DRAINING -> TRANSFER)
                self.vnode_loads[vnode_id] = load // 2 # Simulate shedding load
