from std.collections import Dict
from .vnode import VNode

struct Shard:
    var core_id: Int
    var vnodes: Dict[Int, VNode]

    def __init__(out self, core_id: Int):
        self.core_id = core_id
        self.vnodes = Dict[Int, VNode]()

    def add_vnode(mut self, var vnode: VNode):
        var id = vnode.id
        self.vnodes[id] = vnode^

    def get_vnode(mut self, vnode_id: Int) -> Pointer[VNode]:
        # Pointer is old, let's try to get a reference if possible.
        # But for now, let's just use it to see if it compiles or what it suggests.
        # Actually, let's return a Pointer from memory.
        return Pointer[VNode]()
