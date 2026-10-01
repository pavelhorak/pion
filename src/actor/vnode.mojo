from std.collections import Dict
from src.io.wal import WAL
from src.common.value import GenericValue, ValueType
from src.memory.slab_allocator import SlabAllocator
from src.common.hash_map import SlabHashMap
from src.common.list import SlabList
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc

@fieldwise_init
struct VNodeState(Copyable, Movable):
    var value: Int
    comptime ACTIVE = 0
    comptime DRAINING = 1
    comptime TRANSFER = 2
    comptime HYDRATING = 3

    def __eq__(self, other: VNodeState) -> Bool:
        return self.value == other.value

@fieldwise_init
struct VNode(Copyable, Movable):
    var id: Int
    var state: VNodeState
    var kv_store: Dict[String, GenericValue]
    var handoff_queue: List[String]

    def __init__(out self, id: Int):
        self.id = id
        self.state = VNodeState(VNodeState.ACTIVE)
        self.kv_store = Dict[String, GenericValue]()
        self.handoff_queue = List[String]()

    def set_string(mut self, key: String, value: String):
        if self.state == VNodeState(VNodeState.ACTIVE):
            var length = len(value)
            var ptr = alloc[UInt8](length + 1)
            for i in range(length):
                ptr[i] = value.as_bytes()[i]
            ptr[length] = 0
            
            var val = GenericValue(ValueType(ValueType.STRING), ptr.unsafe_bitcast[NoneType]())
            self.kv_store[key] = val^
        elif self.state == VNodeState(VNodeState.DRAINING):
            self.handoff_queue.append("SET:" + key + ":" + value)

    def get_string(self, key: String) -> String:
        try:
            var val = self.kv_store[key]
            if val.type == ValueType(ValueType.STRING):
                var _sbuf = alloc[UInt8](24)
                var ptr = val.as_string_safe(_sbuf)
                var s = String("")
                var i = 0
                while ptr[i] != 0:
                    s += chr(Int(ptr[i]))
                    i += 1
                _sbuf.unsafe_free()
                return s
            return "NIL"
        except:
            return "NIL"

    def hset(mut self, key: String, field: String, value: String):
        if self.state == VNodeState(VNodeState.ACTIVE):
            try:
                var val = self.kv_store[key]
                if val.type == ValueType(ValueType.HASH):
                    var map_ptr = val.ptr.unsafe_bitcast[SlabHashMap]()
                    var length = len(value)
                    var s_ptr = alloc[UInt8](length + 1)
                    for i in range(length):
                        s_ptr[i] = value.as_bytes()[i]
                    s_ptr[length] = 0
                    
                    var g_val = GenericValue(ValueType(ValueType.STRING), s_ptr.unsafe_bitcast[NoneType]())
                    map_ptr[].set(field, g_val^)
                else:
                    pass
            except:
                var map_ptr = alloc[SlabHashMap](1)
                map_ptr.unsafe_write(SlabHashMap(16, 100))
                
                var length = len(value)
                var s_ptr = alloc[UInt8](length + 1)
                for i in range(length):
                    s_ptr[i] = value.as_bytes()[i]
                s_ptr[length] = 0
                
                var g_val = GenericValue(ValueType(ValueType.STRING), s_ptr.unsafe_bitcast[NoneType]())
                map_ptr[].set(field, g_val^)
                
                var val = GenericValue(ValueType(ValueType.HASH), map_ptr.unsafe_bitcast[NoneType]())
                self.kv_store[key] = val^
        elif self.state == VNodeState(VNodeState.DRAINING):
            self.handoff_queue.append("HSET:" + key + ":" + field + ":" + value)

    def hget(self, key: String, field: String) -> String:
        try:
            var val = self.kv_store[key]
            if val.type == ValueType(ValueType.HASH):
                var map_ptr = val.ptr.unsafe_bitcast[SlabHashMap]()
                var g_val = map_ptr[].get(field)
                if g_val.type == ValueType(ValueType.STRING):
                    var _sbuf = alloc[UInt8](24)
                    var ptr = g_val.as_string_safe(_sbuf)
                    var s = String("")
                    var i = 0
                    while ptr[i] != 0:
                        s += chr(Int(ptr[i]))
                        i += 1
                    _sbuf.unsafe_free()
                    return s
            return "NIL"
        except:
            return "NIL"

    def lpush(mut self, key: String, value: String):
        if self.state == VNodeState(VNodeState.ACTIVE):
            try:
                var val = self.kv_store[key]
                if val.type == ValueType(ValueType.LIST):
                    var list_ptr = val.ptr.unsafe_bitcast[SlabList]()
                    var length = len(value)
                    var s_ptr = alloc[UInt8](length + 1)
                    for i in range(length):
                        s_ptr[i] = value.as_bytes()[i]
                    s_ptr[length] = 0
                    
                    var g_val = GenericValue(ValueType(ValueType.STRING), s_ptr.unsafe_bitcast[NoneType]())
                    list_ptr[].lpush(g_val^)
                else:
                    pass
            except:
                var list_ptr = alloc[SlabList](1)
                list_ptr.unsafe_write(SlabList(100))
                
                var length = len(value)
                var s_ptr = alloc[UInt8](length + 1)
                for i in range(length):
                    s_ptr[i] = value.as_bytes()[i]
                s_ptr[length] = 0
                
                var g_val = GenericValue(ValueType(ValueType.STRING), s_ptr.unsafe_bitcast[NoneType]())
                list_ptr[].lpush(g_val^)
                
                var val = GenericValue(ValueType(ValueType.LIST), list_ptr.unsafe_bitcast[NoneType]())
                self.kv_store[key] = val^
        elif self.state == VNodeState(VNodeState.DRAINING):
            self.handoff_queue.append("LPUSH:" + key + ":" + value)

    def rpop(mut self, key: String) -> String:
        try:
            var val = self.kv_store[key]
            if val.type == ValueType(ValueType.LIST):
                var list_ptr = val.ptr.unsafe_bitcast[SlabList]()
                var g_val = list_ptr[].rpop()
                if g_val.type == ValueType(ValueType.STRING):
                    var _sbuf = alloc[UInt8](24)
                    var ptr = g_val.as_string_safe(_sbuf)
                    var s = String("")
                    var i = 0
                    while ptr[i] != 0:
                        s += chr(Int(ptr[i]))
                        i += 1
                    _sbuf.unsafe_free()
                    return s
            return "NIL"
        except:
            return "NIL"

    def start_migration(mut self):
        self.state = VNodeState(VNodeState.DRAINING)
        print("VNode " + String(self.id) + ": Starting migration (DRAINING)")

    def finish_migration(mut self) raises:
        self.state = VNodeState(VNodeState.TRANSFER)
        print("VNode " + String(self.id) + ": Transferring ownership")
        
        # Replay handoff queue
        print("VNode " + String(self.id) + ": Replaying " + String(len(self.handoff_queue)) + " buffered ops")
        for i in range(len(self.handoff_queue)):
            var op = self.handoff_queue[i]
            # Simple replay logic for prototype
            # In real system, this would parse the command and apply it
            print("VNode " + String(self.id) + ": Replayed " + op)
        
        self.handoff_queue = List[String]()
        self.state = VNodeState(VNodeState.ACTIVE)
        print("VNode " + String(self.id) + ": Migration complete (ACTIVE)")

    def serialize(self) -> String:
        # Simplified serialization for prototype
        return "VNODE_DATA:" + String(self.id)
