from std.collections import Dict

struct RendezvousHash:
    var nodes: List[String]

    def __init__(out self, var nodes: List[String]):
        self.nodes = nodes^

    def get_node(self, key: String) -> String:
        if len(self.nodes) == 0:
            return ""
        
        var best_node = self.nodes[0]
        var max_hash: UInt64 = 0
        
        for i in range(len(self.nodes)):
            var node = self.nodes[i]
            # HRW: hash(node + key)
            var current_hash = hash(node + key)
            if i == 0 or current_hash > max_hash:
                max_hash = current_hash
                best_node = node
        
        return best_node
