from src.common.list import SlabList
from src.common.hash_map import SlabHashMap
from src.common.value import GenericValue
from std.time import time

@fieldwise_init
struct StreamEntryId(Copyable, Movable):
    var ms: UInt64
    var seq: UInt64

@fieldwise_init
struct StreamEntry:
    var id: StreamEntryId
    var fields: SlabHashMap

@fieldwise_init
struct Stream:
    var entries: SlabList
    var length: Int
    var last_id: StreamEntryId
    # var groups: SlabHashMap # for consumer groups, to be implemented later

def get_current_timestamp() -> UInt64:
    return int(time() * 1000)

def generate_stream_id(last_id: StreamEntryId, seq_given: Bool, seq: UInt64) -> StreamEntryId:
    var now_ms = get_current_timestamp()
    var new_id = StreamEntryId(now_ms, 0)
    if now_ms > last_id.ms:
        if seq_given:
            new_id.seq = seq
        else:
            new_id.seq = 0
    else:
        new_id.ms = last_id.ms
        new_id.seq = last_id.seq + 1
    return new_id
