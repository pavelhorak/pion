from src.common.utils import acc_digit_checked
from src.common.ptr import null_ptr, is_not_null, is_null
from src.vector.vector_abi import vector_backend_line
from src.common.container_free import free_container, hash_get_live
from src.common.version import PION_VERSION, PION_BUILD_SHA
from std.ffi import external_call
from std.memory.unsafe_pointer import Pointer
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue
from src.common.value import ValueType
from src.common.list import SlabList
from src.memory.slab_allocator import SlabAllocator
from src.memory.object_pool import ObjectPool
from src.vector.hnsw import HNSWGraph
from std.memory import alloc, unsafe_memset, unsafe_memcpy
from src.common.bitmap import getbit, setbit, bitcount
from src.common.skip_list import SlabSkipList
from src.common.geohash import geohash_encode, GEO_STEP_MAX
from src.common.hll import hll_add, hll_count, hll_merge, HLL_REGISTERS

@fieldwise_init
struct ZAddOutcome(Copyable, Movable):
    """Result of a conditional ZADD (gh #237). `ok` False means WRONGTYPE;
    `applied` False means a flag suppressed the write, which is NOT an error."""
    var ok: Bool
    var applied: Bool
    var added: Int64        # 1 only when a NEW member was inserted
    var changed: Int64      # 1 when added OR the score actually moved (CH counts this)
    var new_score: Float64
    var nan: Bool           # INCR would leave NaN (inf + -inf): an error, nothing written


struct IntCmdResult:
    var value: Int64
    var is_valid: Bool

    def __init__(out self, value: Int64, is_valid: Bool):
        self.value = value
        self.is_valid = is_valid

from src.common.lock_free import LockFreeRingBuffer, AITask
from src.io.wal import WAL, gv_bytes
from src.io.blob_store import BlobStore, BLOB_TIER_OFF
from src.network.raft import RaftNode, RaftLogEntry
from src.common.prng import Xoshiro256PlusPlus

struct CommandDispatcher:
    var keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]
    var hash_map_pool: Pointer[ObjectPool[SlabHashMap], MutUntrackedOrigin]
    var skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]
    var list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin]
    var ai_queue: Pointer[LockFreeRingBuffer, MutUntrackedOrigin]
    var wal: Pointer[WAL, MutUntrackedOrigin]
    var raft: Pointer[RaftNode, MutUntrackedOrigin]
    var prng: Xoshiro256PlusPlus
    # gh #163: wired post-construction by Pion.__init__. At the struct tail for
    # the same layout reason as FastPathHandler.blobs.
    var blobs: Pointer[BlobStore, MutUntrackedOrigin]
    var blob_threshold: Int

    def __init__(out self, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], hash_map_pool: Pointer[ObjectPool[SlabHashMap], MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin], list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin], ai_queue: Pointer[LockFreeRingBuffer, MutUntrackedOrigin], wal: Pointer[WAL, MutUntrackedOrigin], raft: Pointer[RaftNode, MutUntrackedOrigin]):
        self.keyspace = keyspace
        self.hash_map_pool = hash_map_pool
        self.skip_list_pool = skip_list_pool
        self.list_pool = list_pool
        self.ai_queue = ai_queue
        self.wal = wal
        self.blobs = null_ptr[BlobStore, MutUntrackedOrigin]()
        self.blob_threshold = BLOB_TIER_OFF
        self.raft = raft
        self.prng = Xoshiro256PlusPlus(0xDEADBEEF)


    @always_inline
    def execute_get(self, key: String) -> GenericValue:
        return self.keyspace[].get(key)

    @always_inline
    def execute_set(self, key: String, val_str: String):
        # gh #163: same routing decision as the fast path — a value at or above
        # the threshold is copied into the file-backed arena and the WAL logs a
        # pointer. Slow-path SET is where oversized values from non-fast-path
        # clients (and MULTI/EXEC replay) land, so it cannot be skipped.
        var vlen = val_str.byte_length()
        var blob_seg = -1
        var blob_off = -1
        if vlen >= self.blob_threshold and is_not_null(self.blobs):
            _ = self.blobs[].append(val_str.unsafe_ptr(), vlen, blob_seg, blob_off)

        var bp = null_ptr[UInt8, MutUntrackedOrigin]()
        if blob_seg >= 0:
            bp = self.blobs[].ptr_at(blob_seg, blob_off, vlen)
        if is_not_null(bp):
            _ = self.wal[].append_blob_ref(key.unsafe_ptr(), key.byte_length(),
                                           blob_seg, blob_off, vlen)
            self.keyspace[].set(key, GenericValue.from_blob_ptr(bp, vlen))
        else:
            # 1. Write-Ahead Logging (WAL) First
            _ = self.wal[].append_kv(1, key.unsafe_ptr(), key.byte_length(),
                                 val_str.unsafe_ptr(), vlen)

            # 2. Optimistic Execution: Apply state change immediately
            var val = GenericValue.borrow(val_str.unsafe_ptr(), val_str.byte_length())
            self.keyspace[].set(key, val)

        # gh #394: this used to append a RaftLogEntry per SET ("pipelined
        # quorum"). Raft here elects leaders and commits topology only
        # (raft.mojo's header: "NOT for data replication") — nothing read,
        # applied or trimmed that log, so every slow-path SET grew it forever.

    @always_inline
    def execute_del(self, key: String) -> Bool:
        var taken = GenericValue()
        var removed = self.keyspace[].remove_generic_taking(GenericValue.borrow(key.unsafe_ptr(), key.byte_length()), taken)
        if removed:
            free_container(taken)   # gh #369: a DEL'd aggregate is freed
            # gh #170: fast-path DEL logs its own record; this covers
            # slow-path multi-DEL and EXEC replay
            _ = self.wal[].append(2, key.unsafe_ptr(), key.byte_length())
        return removed

    @always_inline
    def execute_exists(self, key: String) -> Bool:
        return self.keyspace[].contains(key)

    @always_inline
    def execute_lpush(self, key: String, val_str: String) -> Int:
        var val = self.keyspace[].get(key)
        var list_ptr: Pointer[SlabList, MutUntrackedOrigin]
        if val.is_none():
            list_ptr = self.list_pool[].acquire(); list_ptr[].reset()
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.LIST)
            new_val.set_ptr(list_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
        elif val.type.value == ValueType.LIST:
            list_ptr = val.as_list().unsafe_bitcast[SlabList]()
        else:
            return -1

        list_ptr[].lpush(GenericValue.from_string(val_str))
        _ = self.wal[].append_kv(6, key.unsafe_ptr(), key.byte_length(),
                                 val_str.unsafe_ptr(), val_str.byte_length())
        return list_ptr[].llen()

    @always_inline
    def execute_rpush(self, key: String, val_str: String) -> Int:
        var val = self.keyspace[].get(key)
        var list_ptr: Pointer[SlabList, MutUntrackedOrigin]
        if val.is_none():
            list_ptr = self.list_pool[].acquire(); list_ptr[].reset()
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.LIST)
            new_val.set_ptr(list_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
        elif val.type.value == ValueType.LIST:
            list_ptr = val.as_list().unsafe_bitcast[SlabList]()
        else:
            return -1

        list_ptr[].rpush(GenericValue.from_string(val_str))
        _ = self.wal[].append_kv(7, key.unsafe_ptr(), key.byte_length(),
                                 val_str.unsafe_ptr(), val_str.byte_length())
        return list_ptr[].llen()

    @always_inline
    def execute_lpush_value(self, key: String, var val_gv: GenericValue) -> Int:
        """LPUSH an already-materialized GenericValue (must be an owned copy,
        e.g. from GenericValue.from_ptr). Avoids the String round-trip that
        corrupts binary values — see the LMOVE/RPOPLPUSH callers."""
        var val = self.keyspace[].get(key)
        var list_ptr: Pointer[SlabList, MutUntrackedOrigin]
        if val.is_none():
            list_ptr = self.list_pool[].acquire(); list_ptr[].reset()
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.LIST)
            new_val.set_ptr(list_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
        elif val.type.value == ValueType.LIST:
            list_ptr = val.as_list().unsafe_bitcast[SlabList]()
        else:
            return -1
        var buf = alloc[UInt8](64)
        var vl = 0
        var vp = gv_bytes(val_gv, buf, vl)
        _ = self.wal[].append_kv(6, key.unsafe_ptr(), key.byte_length(), vp, vl)
        buf.unsafe_free()
        list_ptr[].lpush(val_gv)
        return list_ptr[].llen()

    @always_inline
    def execute_rpush_value(self, key: String, var val_gv: GenericValue) -> Int:
        """RPUSH an already-materialized GenericValue (owned copy). See
        execute_lpush_value."""
        var val = self.keyspace[].get(key)
        var list_ptr: Pointer[SlabList, MutUntrackedOrigin]
        if val.is_none():
            list_ptr = self.list_pool[].acquire(); list_ptr[].reset()
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.LIST)
            new_val.set_ptr(list_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
        elif val.type.value == ValueType.LIST:
            list_ptr = val.as_list().unsafe_bitcast[SlabList]()
        else:
            return -1
        var buf = alloc[UInt8](64)
        var vl = 0
        var vp = gv_bytes(val_gv, buf, vl)
        _ = self.wal[].append_kv(7, key.unsafe_ptr(), key.byte_length(), vp, vl)
        buf.unsafe_free()
        list_ptr[].rpush(val_gv)
        return list_ptr[].llen()

    @always_inline
    def execute_lpop(self, key: String) -> GenericValue:
        var val = self.keyspace[].get(key)
        if val.type.value == ValueType.LIST:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            var popped = list_ptr[].lpop()
            if not popped.is_none():
                _ = self.wal[].append(13, key.unsafe_ptr(), key.byte_length())
            return popped
        return GenericValue()

    @always_inline
    def execute_rpop(self, key: String) -> GenericValue:
        var val = self.keyspace[].get(key)
        if val.type.value == ValueType.LIST:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            var popped = list_ptr[].rpop()
            if not popped.is_none():
                _ = self.wal[].append(14, key.unsafe_ptr(), key.byte_length())
            return popped
        return GenericValue()

    @always_inline
    def execute_llen(self, key: String) -> Int:
        var val = self.keyspace[].get(key)
        if val.is_none():
            return 0
        elif val.type.value == ValueType.LIST:
            var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
            return list_ptr[].llen()
        else:
            return -1

    @always_inline
    def get_list(self, key: String) -> Pointer[SlabList, MutUntrackedOrigin]:
        var val = self.keyspace[].get(key)
        if val.type.value == ValueType.LIST:
            return val.as_list().unsafe_bitcast[SlabList]()
        return null_ptr[SlabList, MutUntrackedOrigin]()

    @always_inline
    def execute_ft_searchtext(self, text: String, mut ai_gateway: AIGateway, mut hnsw: HNSWGraph, enable_gateway: Bool) raises -> String:
        if not enable_gateway:
            return ""
        return ai_gateway.generate_augmented_prompt(text, hnsw)

    @always_inline
    def execute_ft_chatt(self, text: String, mut ai_gateway: AIGateway, mut hnsw: HNSWGraph, enable_gateway: Bool) raises -> Bool:
        if not enable_gateway:
            return False
        
        # 1. Perform augmented retrieval on the main thread (fast, SIMD-accelerated)
        var augmented = ai_gateway.generate_augmented_prompt(text, hnsw)
        
        # 2. Push the task to the background queue instead of calling dispatch_to_llm directly
        var task = AITask("openai", augmented, 0, "") # FD placeholder, empty session_id
        if not self.ai_queue[].push(task):
            print("AI Task Queue Full! Dropping request.")
            return False
            
        return True

    @always_inline
    def execute_ft_invoke_tool(self, tool_name: String, args_json: String, mut ai_gateway: AIGateway, enable_gateway: Bool) raises -> String:
        if not enable_gateway:
            return ""
        return ai_gateway.invoke_tool(tool_name, args_json)

    @always_inline
    def execute_ft_mem_push(self, session_id: String, role: String, content: String, mut ai_gateway: AIGateway, enable_gateway: Bool) raises -> Bool:
        if not enable_gateway:
            return False
        ai_gateway.push_to_memory_tunnel(session_id, role, content)
        return True

    @always_inline
    def execute_hset_vector(self, mut hnsw: HNSWGraph, mut db_size: Int, f_vec: Pointer[Float32, MutUntrackedOrigin]) raises:
        hnsw.add_vector(db_size, f_vec)
        db_size += 1

    @always_inline
    def execute_info(self, send_stalls: UInt64 = 0, listen_port: Int = 1974, keys: Int = 0,
                     expires: Int = 0, uptime_s: Int = 0, extra: String = String(""),
                     repl_section: String = String("# Replication\r\nrole:master\r\nconnected_slaves:0\r\n"),
                     cluster_enabled: Bool = False) -> String:
        """INFO's body, every section; handle_info filters and frames it.
        `repl_section` is built by the caller from the cluster state: this
        used to print `role:master` and `cluster_enabled:0` whatever the
        server was, and only `INFO replication` read the real role."""
        # gh #262: resolve first, then report. This used to hardcode
        # tcp_port:1974, used_memory:1048576 and an empty # Keyspace — values
        # dressed as measurements. redis_version stays 7.0.0 because client
        # libraries gate features on it; pion_version carries the truth.
        var rss = external_call["pion_rss_sample_now", UInt64]()
        var rss_peak = external_call["pion_crash_rss_peak_bytes", UInt64]()
        if rss_peak < rss: rss_peak = rss
        var total_ram = external_call["pion_crash_total_ram_bytes", UInt64]()
        var body = String("# Server\r\nredis_version:7.0.0\r\n")
        body += "pion_version:" + PION_VERSION + "+" + PION_BUILD_SHA + "\r\n"
        # D11: which vector implementation is linked — the closed library or
        # the open reference. Asked of the library itself, not the build flag.
        body += "pion_vector:" + vector_backend_line() + "\r\n"
        body += "redis_mode:standalone\r\n"
        # Redis's field: tells a client WHICH server answered (a restart test
        # must not mistake a dying predecessor's listener for the new server).
        body += "process_id:" + String(Int(external_call["getpid", Int32]())) + "\r\n"
        body += "tcp_port:" + String(listen_port) + "\r\n"
        body += "uptime_in_seconds:" + String(uptime_s) + "\r\n"
        body += "# Memory\r\n"
        body += "used_memory:" + String(rss) + "\r\n"
        body += "used_memory_human:" + String(rss >> 20) + "M\r\n"
        body += "used_memory_rss:" + String(rss) + "\r\n"
        body += "used_memory_peak:" + String(rss_peak) + "\r\n"
        if total_ram > UInt64(0):
            body += "total_system_memory:" + String(total_ram) + "\r\n"
        # gh #261: the limit is on RSS — the same number used_memory reports.
        var maxmem = external_call["pion_get_maxmemory", UInt64]()
        body += "maxmemory:" + String(maxmem) + "\r\n"
        body += "maxmemory_human:" + String(maxmem >> 20) + "M\r\n"
        body += "maxmemory_policy:noeviction\r\n"
        body += "maxmemory_refusing_writes:" + String(
            external_call["pion_maxmemory_check", Int32]()) + "\r\n"
        body += "# Cluster\r\ncluster_enabled:" + ("1" if cluster_enabled else "0") + "\r\n"
        body += repl_section
        # gh #149 / gh #163: persistence is where an operator finds out that
        # writes stopped being durable. The old code dropped WAL entries with no
        # counter, no log line and no INFO field — 4.6 GB of acknowledged SETs
        # went missing with nothing anywhere to show for it.
        body += "# Persistence\r\n"
        body += "wal_segments_sealed:" + String(self.wal[].sealed) + "\r\n"
        body += "wal_segment_bytes:" + String(self.wal[].file_size) + "\r\n"
        body += "wal_tail_offset:" + String(self.wal[].tail_offset) + "\r\n"
        body += "wal_dropped_entries:" + String(self.wal[].dropped) + "\r\n"
        body += "wal_dropped_bytes:" + String(self.wal[].dropped_bytes) + "\r\n"
        # gh #260: the active policy and whether it has fired. A monitor that
        # only watches wal_dropped_entries cannot tell "full and refusing"
        # (writes are erroring, no data lost) from "full and dropping" (writes
        # are being acknowledged and lost) — those need different pages.
        body += ("wal_full_policy:"
                 + ("refuse" if self.wal[].refuse_when_full else "drop") + "\r\n")
        body += ("wal_durability_lost:"
                 + ("1" if self.wal[].durability_lost else "0") + "\r\n")
        # gh #170 + gh #174: aggregate durability coverage. Snapshot v2
        # serializes every keyspace type and the WAL now effect-logs every
        # mutation, including the three gaps #170 left open (HLL via cmd 24
        # elements + cmd 17 merge images, streams via cmd 23/27, TTLs via
        # cmd 25/26). These flags stay in INFO rather than being deleted: they
        # are the machine-readable answer to "what does this build actually
        # persist", and a future type will need to declare itself here too.
        body += "snapshot_version:2\r\n"
        body += "aggregate_wal_records:1\r\n"
        body += "hll_wal_logged:1\r\n"
        body += "streams_persisted:1\r\n"
        body += "ttls_persisted:1\r\n"
        # Hash-FIELD TTLs (HEXPIRE) are durable since gh #392: they live IN the
        # hash (SlabHashMap.field_ttl) and are WAL-logged as cmd 32/33 and written
        # into the snapshot, so they survive a restart like any other state.
        body += "hash_field_ttls_persisted:1\r\n"
        if is_not_null(self.blobs) and self.blobs[].enabled \
           and self.blob_threshold < BLOB_TIER_OFF:
            body += "blob_tier_enabled:1\r\n"
            body += "blob_threshold:" + String(self.blob_threshold) + "\r\n"
            body += "blob_segments:" + String(self.blobs[].seg_count) + "\r\n"
            body += "blob_bytes_used:" + String(self.blobs[].bytes_used) + "\r\n"
            body += "blob_records:" + String(self.blobs[].records) + "\r\n"
            body += "blob_bytes_live_at_last_scan:" + String(self.blobs[].live_at_last_scan) + "\r\n"
            body += "blob_compactions:" + String(self.blobs[].compactions) + "\r\n"
        else:
            body += "blob_tier_enabled:0\r\n"
            body += "blob_bytes_used:0\r\n"
        # gh #192: EAGAIN retries in the blocking large-response send loops
        # (each ≈150 µs of usleep on macOS). This worker only — the writer is
        # per-worker. The kill-test/observability signal for substrate TTFT.
        body += "# Stats\r\n"
        body += "send_eagain_stalls:" + String(send_stalls) + "\r\n"
        body += extra   # gh #262: the `# Pion` value-receipt section, built by the caller
        body += "# Keyspace\r\n"
        # Redis omits the db line when the db is empty; keys are THIS worker's.
        if keys > 0:
            body += "db0:keys=" + String(keys) + ",expires=" + String(expires) + ",avg_ttl=0\r\n"
        return body

    @always_inline
    def execute_ft_search(self) -> String:
        return "*3\r\n:1\r\n$5\r\ndoc:1\r\n*2\r\n$3\r\nvec\r\n$10\r\n[VECTOR]\r\n"

    @always_inline
    def execute_ft_info(self) -> String:
        return "*2\r\n$10\r\nindex_name\r\n$3\r\nidx\r\n"

    @always_inline
    def execute_ping(self, msg: String) -> String:
        if msg == "":
            return "+PONG\r\n"
        else:
            return "$" + String(msg.byte_length()) + "\r\n" + msg + "\r\n"

    @always_inline
    def execute_incr(self, key: String) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        var new_val: Int64 = 0
        var valid = True
        if val.is_none():
            new_val = 1
        elif val.type.value == ValueType.INT:
            var cur = val.as_int()
            if cur == 9223372036854775807:
                return IntCmdResult(0, False)
            new_val = cur + 1
        elif val.is_string():
            var _sbuf = alloc[UInt8](24)
            var ptr = val.as_string_safe(_sbuf)
            var length = val.string_len()
            var parsed_val: Int64 = 0
            var is_neg = False
            var start_idx = 0
            if length > 0 and ptr[unsafe_offset=0] == 45:
                is_neg = True
                start_idx = 1
            if start_idx >= length: valid = False
            for j in range(start_idx, length):
                var c = Int(ptr[unsafe_offset=j])
                if c >= 48 and c <= 57:
                    parsed_val = acc_digit_checked(parsed_val, Int64(c - 48), is_neg, j + 1 == length)
                    if parsed_val == -1:
                        valid = False
                        break
                else:
                    valid = False
                    break
            if valid:
                if is_neg: parsed_val = -parsed_val
                if parsed_val == 9223372036854775807:
                    valid = False
                else:
                    new_val = parsed_val + 1
            _sbuf.unsafe_free()
        else:
            valid = False

        if valid:
            self.keyspace[].set(key, GenericValue.from_int(new_val))

        return IntCmdResult(new_val, valid)

    @always_inline
    def execute_decr(self, key: String) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        var new_val: Int64 = 0
        var valid = True
        if val.is_none():
            new_val = -1
        elif val.type.value == ValueType.INT:
            var cur = val.as_int()
            if cur == -9223372036854775808:
                return IntCmdResult(0, False)
            new_val = cur - 1
        elif val.is_string():
            var _sbuf = alloc[UInt8](24)
            var ptr = val.as_string_safe(_sbuf)
            var length = val.string_len()
            var parsed_val: Int64 = 0
            var is_neg = False
            var start_idx = 0
            if length > 0 and ptr[unsafe_offset=0] == 45:
                is_neg = True
                start_idx = 1
            if start_idx >= length: valid = False
            for j in range(start_idx, length):
                var c = Int(ptr[unsafe_offset=j])
                if c >= 48 and c <= 57:
                    parsed_val = acc_digit_checked(parsed_val, Int64(c - 48), is_neg, j + 1 == length)
                    if parsed_val == -1:
                        valid = False
                        break
                else:
                    valid = False
                    break
            if valid:
                if is_neg: parsed_val = -parsed_val
                if parsed_val == -9223372036854775808:
                    valid = False
                else:
                    new_val = parsed_val - 1
            _sbuf.unsafe_free()
        else:
            valid = False

        if valid:
            self.keyspace[].set(key, GenericValue.from_int(new_val))

        return IntCmdResult(new_val, valid)

    @always_inline
    def execute_getbit(self, key: String, offset: Int) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        if val.is_none():
            return IntCmdResult(0, True)
        elif val.type.value == ValueType.BITMAP:
            var bitmap_ptr = val.as_bitmap()
            var byte_len = val.bitmap_len()
            if offset // 8 >= byte_len:
                return IntCmdResult(0, True)
            else:
                return IntCmdResult(Int64(getbit(bitmap_ptr, byte_len, offset)), True)
        else:
            return IntCmdResult(0, False)

    @always_inline
    def execute_setbit(self, key: String, offset: Int, value: Int) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        if val.is_none():
            var byte_len = offset // 8 + 1
            var bitmap_ptr = alloc[UInt8](byte_len)
            unsafe_memset(bitmap_ptr, 0, byte_len)
            
            var old_val = getbit(bitmap_ptr, byte_len, offset)
            
            var result = setbit(byte_len, bitmap_ptr, offset, value)
            var new_bitmap_ptr = result.ptr
            byte_len = result.len

            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.BITMAP)
            new_val._data0 = UInt64(Int(new_bitmap_ptr))
            new_val._data1 = UInt64(byte_len)
            self.keyspace[].set(key, new_val)
            _ = self.wal[].append_u64_val(18, key.unsafe_ptr(), key.byte_length(),
                                          (UInt64(offset) << 1) | UInt64(value & 1),
                                          null_ptr[UInt8, MutUntrackedOrigin](), 0)
            return IntCmdResult(Int64(old_val), True)
        elif val.type.value == ValueType.BITMAP:
            var bitmap_ptr = val.as_bitmap()
            var byte_len = val.bitmap_len()
            var old_val = getbit(bitmap_ptr, byte_len, offset)

            var result = setbit(byte_len, bitmap_ptr, offset, value)
            var new_bitmap_ptr = result.ptr
            byte_len = result.len

            val._data0 = UInt64(Int(new_bitmap_ptr))
            val._data1 = UInt64(byte_len)
            self.keyspace[].set(key, val)
            _ = self.wal[].append_u64_val(18, key.unsafe_ptr(), key.byte_length(),
                                          (UInt64(offset) << 1) | UInt64(value & 1),
                                          null_ptr[UInt8, MutUntrackedOrigin](), 0)
            return IntCmdResult(Int64(old_val), True)
        else:
            return IntCmdResult(0, False)

    @always_inline
    def execute_bitcount(self, key: String) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        if val.is_none():
            return IntCmdResult(0, True)
        elif val.type.value == ValueType.BITMAP:
            var bitmap_ptr = val.as_bitmap()
            var byte_len = val.bitmap_len()
            return IntCmdResult(Int64(bitcount(bitmap_ptr, byte_len)), True)
        else:
            return IntCmdResult(0, False)

    @always_inline
    def execute_hset(self, key: String, field: String, val_str: String) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        var hash_ptr: Pointer[SlabHashMap, MutUntrackedOrigin]
        if val.is_none():
            if self.hash_map_pool[].head < self.hash_map_pool[].capacity:
                hash_ptr = self.hash_map_pool[].acquire(); hash_ptr[].reset()
            else:
                hash_ptr = alloc[SlabHashMap](1); hash_ptr.unsafe_write(SlabHashMap(16))
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.HASH)
            new_val.set_ptr(hash_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
        elif val.type.value == ValueType.HASH:
            hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
        else:
            return IntCmdResult(0, False)
        
        # gh #232 (slow-path half): the reply is the number of fields ADDED —
        # this answered 1 for an update too, as the fast path once did.
        var _hs_before = hash_ptr[].size
        hash_ptr[].set(field, GenericValue.borrow(val_str.unsafe_ptr(), val_str.byte_length()))
        if Int(hash_ptr[].field_ttl) != 0:   # gh #392: HSET clears the field's TTL (Redis)
            _ = hash_ptr[].clear_field_deadline(GenericValue.borrow(field.unsafe_ptr(), field.byte_length()))
        var _hs_added = 1 if hash_ptr[].size > _hs_before else 0

        # gh #170: effect record (cmd 5), replayed via wal_apply_aggregate
        _ = self.wal[].append_field_kv(5, key.unsafe_ptr(), key.byte_length(),
                                       field.unsafe_ptr(), field.byte_length(),
                                       val_str.unsafe_ptr(), val_str.byte_length())

        # gh #394: no Raft log append here either — see execute_set.

        return IntCmdResult(Int64(_hs_added), True)

    @always_inline
    def execute_hget(self, key: String, field: String) -> GenericValue:
        # gh #392: expired fields (and a hash they emptied) are gone first.
        var val = hash_get_live(self.keyspace, GenericValue.borrow(key.unsafe_ptr(), key.byte_length()))
        if val.type.value == ValueType.HASH:
            var hash_ptr = val.as_hash().unsafe_bitcast[SlabHashMap]()
            return hash_ptr[].get(field)
        return GenericValue()

    @always_inline
    def execute_sadd(self, key: String, val_str: String) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        var set_ptr: Pointer[SlabHashMap, MutUntrackedOrigin]
        var val_val = GenericValue.borrow(val_str.unsafe_ptr(), val_str.byte_length())
        
        if val.is_none():
            set_ptr = alloc[SlabHashMap](1)
            set_ptr.unsafe_write(SlabHashMap(16, 100))
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.SET)
            new_val.set_ptr(set_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
            # gh #111: store a non-none marker value (1), matching the fast-path
            # SADD. Membership below is tested via `get(member).is_none()`; an
            # empty GenericValue() would read back as none, so every re-add would
            # be miscounted as new (SCARD deduped correctly, the count did not).
            set_ptr[].set(val_val, GenericValue.from_int(1))
            _ = self.wal[].append_kv(8, key.unsafe_ptr(), key.byte_length(),
                                     val_str.unsafe_ptr(), val_str.byte_length())
            return IntCmdResult(1, True)
        elif val.type.value == ValueType.SET:
            set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            var exists = set_ptr[].get(val_val)
            if exists.is_none():
                set_ptr[].set(val_val, GenericValue.from_int(1))
                _ = self.wal[].append_kv(8, key.unsafe_ptr(), key.byte_length(),
                                         val_str.unsafe_ptr(), val_str.byte_length())
                return IntCmdResult(1, True)
            else:
                return IntCmdResult(0, True)
        else:
            return IntCmdResult(0, False)

    @always_inline
    def execute_spop(mut self, key: String) -> GenericValue:
        var val = self.keyspace[].get(key)
        if val.type.value == ValueType.SET:
            var set_ptr = val.as_set().unsafe_bitcast[SlabHashMap]()
            var popped = set_ptr[].pop_random(self.prng)
            if not popped.is_none():
                # gh #170: non-deterministic — log the resolved effect (SREM)
                var buf = alloc[UInt8](64)
                var ml = 0
                var mp = gv_bytes(popped, buf, ml)
                _ = self.wal[].append_kv(11, key.unsafe_ptr(), key.byte_length(), mp, ml)
                buf.unsafe_free()
            return popped
        return GenericValue()

    @always_inline
    def execute_zadd(self, key: String, score: Float64, member: String) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        var zset_ptr: Pointer[SlabSkipList, MutUntrackedOrigin]
        
        if val.is_none():
            zset_ptr = self.skip_list_pool[].acquire()
            zset_ptr.unsafe_write(SlabSkipList(16))
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.ZSET)
            new_val.set_ptr(zset_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
        elif val.type.value == ValueType.ZSET:
            zset_ptr = val.as_zset().unsafe_bitcast[SlabSkipList]()
        else:
            return IntCmdResult(0, False)
            
        # gh #187: upsert dedups an existing member; reply counts NEW members only
        var zadd_added = zset_ptr[].upsert(score, GenericValue.from_string(member))
        _ = self.wal[].append_scored(9, key.unsafe_ptr(), key.byte_length(),
                                     score, member.unsafe_ptr(), member.byte_length())
        return IntCmdResult(Int64(zadd_added), True)

    @always_inline
    def execute_zadd_cond(self, key: String, score: Float64, member: String,
                          nx: Bool, xx: Bool, gt: Bool, lt: Bool,
                          incr: Bool) -> ZAddOutcome:
        """ZADD with NX/XX/GT/LT/INCR (gh #237). Previously any flag form was
        rejected as a bad float, so leaderboard idioms (`GT`/`LT`) and the very
        common `INCR` simply did not work."""
        var val = self.keyspace[].get(key)
        var zset_ptr: Pointer[SlabSkipList, MutUntrackedOrigin]
        var member_gv = GenericValue.from_string(member)
        if val.is_none():
            # XX must not CREATE the key — checked before allocating anything.
            if xx:
                return ZAddOutcome(True, False, 0, 0, 0.0, False)
            zset_ptr = self.skip_list_pool[].acquire()
            zset_ptr.unsafe_write(SlabSkipList(16))
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.ZSET)
            new_val.set_ptr(zset_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
        elif val.type.value == ValueType.ZSET:
            zset_ptr = val.as_zset().unsafe_bitcast[SlabSkipList]()
        else:
            return ZAddOutcome(False, False, 0, 0, 0.0, False)

        var cur_gv = zset_ptr[].member_score(member_gv)
        var exists = not cur_gv.is_none()
        var cur = cur_gv.as_float() if exists else Float64(0.0)
        # INCR is relative to the current score; on a missing member it is the
        # score itself, matching Redis.
        var target = (cur + score) if (incr and exists) else score

        if nx and exists:
            return ZAddOutcome(True, False, 0, 0, 0.0, False)
        if xx and not exists:
            return ZAddOutcome(True, False, 0, 0, 0.0, False)
        # inf + -inf: Redis refuses with "resulting score is not a number" and
        # changes nothing, checked after NX/XX and before GT/LT as it does. A
        # NaN score has no place in the order at all.
        if target != target:
            return ZAddOutcome(True, False, 0, 0, 0.0, True)
        # GT/LT only gate an UPDATE; against a missing member they always allow
        # the insert (Redis treats "no member" as no bound, not as infinity —
        # the infinity rule is EXPIRE's, not ZADD's).
        if gt and exists and target <= cur:
            return ZAddOutcome(True, False, 0, 0, 0.0, False)
        if lt and exists and target >= cur:
            return ZAddOutcome(True, False, 0, 0, 0.0, False)

        # CH counts members ADDED plus members whose score actually moved. An
        # equal-score write is a no-op and must not count — `upsert` returns 0
        # for it and for a real update alike, so compare here where `cur` is
        # still in scope.
        var moved = (not exists) or (target != cur)
        var added = zset_ptr[].upsert(target, member_gv)
        _ = self.wal[].append_scored(9, key.unsafe_ptr(), key.byte_length(),
                                     target, member.unsafe_ptr(), member.byte_length())
        return ZAddOutcome(True, True, Int64(added), Int64(1) if moved else Int64(0), target, False)

    @always_inline
    def execute_zpopmin(self, key: String) -> GenericValue:
        var val = self.keyspace[].get(key)
        if val.type.value == ValueType.ZSET:
            var zset_ptr = val.as_zset().unsafe_bitcast[SlabSkipList]()
            var min_node = zset_ptr[].pop_min()
            if min_node:
                # gh #170: log the resolved effect (ZREM of the popped member)
                var buf = alloc[UInt8](64)
                var ml = 0
                var mp = gv_bytes(min_node[].value, buf, ml)
                _ = self.wal[].append_kv(12, key.unsafe_ptr(), key.byte_length(), mp, ml)
                buf.unsafe_free()
                return min_node[].value
        return GenericValue()

    @always_inline
    def execute_geoadd(self, key: String, longitude: Float64, latitude: Float64, member: String) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        var zset_ptr: Pointer[SlabSkipList, MutUntrackedOrigin]

        if val.is_none():
            zset_ptr = self.skip_list_pool[].acquire()
            zset_ptr.unsafe_write(SlabSkipList(16))
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.ZSET)     # a geo key is a sorted set, as in Redis
            new_val.set_ptr(zset_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
        elif val.type.value == ValueType.GEO or val.type.value == ValueType.ZSET:
            zset_ptr = val.as_geo().unsafe_bitcast[SlabSkipList]()
        else:
            return IntCmdResult(0, False)

        var hash = geohash_encode(latitude, longitude, GEO_STEP_MAX)
        # gh #187: GEOADD counts new members only (Redis semantics)
        var geo_added = zset_ptr[].upsert(Float64(hash.bits), GenericValue.from_string(member))
        _ = self.wal[].append_scored(9, key.unsafe_ptr(), key.byte_length(),
                                     Float64(hash.bits), member.unsafe_ptr(),
                                     member.byte_length())
        return IntCmdResult(Int64(geo_added), True)

    @always_inline
    def execute_pfadd(self, key: String, elements: List[String]) -> IntCmdResult:
        var val = self.keyspace[].get(key)
        var hll_ptr: Pointer[UInt8, MutUntrackedOrigin]

        if val.is_none():
            hll_ptr = alloc[UInt8](HLL_REGISTERS)
            unsafe_memset(hll_ptr, 0, HLL_REGISTERS)
            var new_val = GenericValue()
            new_val.type = ValueType(ValueType.HLL)
            new_val.set_ptr(hll_ptr.unsafe_bitcast[NoneType]())
            self.keyspace[].set(key, new_val)
        elif val.type.value == ValueType.HLL:
            hll_ptr = val.as_hll()
        else:
            return IntCmdResult(0, False)
        
        var updated = False
        for j in range(len(elements)):
            var element = GenericValue.from_string(elements[j])
            if hll_add(hll_ptr, element):
                updated = True
                # gh #174: the fast path logs PFADD too, but this is the path
                # taken whenever the fast path bails (long key, odd framing), so
                # it needs its own append — a mutation reachable by two routes is
                # only durable if BOTH log it.
                _ = self.wal[].append_kv(24, key.unsafe_ptr(), key.byte_length(),
                                         elements[j].unsafe_ptr(),
                                         elements[j].byte_length())
        return IntCmdResult(Int64(1) if updated else Int64(0), True)

    @always_inline
    def execute_pfcount(self, keys: List[String]) -> IntCmdResult:
        # `PFCOUNT` with no keys reached the multi-key branch and indexed
        # keys[0] on an EMPTY list: a debug_assert abort on a -O0 build, and an
        # out-of-bounds read on release. Callers are supposed to reject the
        # arity first, but a dispatcher primitive must not depend on that —
        # this one is reachable from the fast path, the slow path, and MULTI
        # replay.
        if len(keys) == 0:
            return IntCmdResult(0, False)
        if len(keys) == 1:
            var val = self.keyspace[].get(keys[0])
            if val.is_none():
                return IntCmdResult(0, True)
            elif val.type.value == ValueType.HLL:
                var hll_ptr = val.as_hll()
                return IntCmdResult(Int64(hll_count(hll_ptr)), True)
            else:
                return IntCmdResult(0, False)
        else:
            var first_key = keys[0]
            var val = self.keyspace[].get(first_key)
            var temp_hll = alloc[UInt8](HLL_REGISTERS)
            
            if val.is_none():
                unsafe_memset(temp_hll, 0, HLL_REGISTERS)
            elif val.type.value == ValueType.HLL:
                unsafe_memcpy(dest=temp_hll, src=val.as_hll(), count=HLL_REGISTERS)
            else:
                temp_hll.unsafe_free()
                return IntCmdResult(0, False)

            var merge_error = False
            for j in range(1, len(keys)):
                var hll_val = self.keyspace[].get(keys[j])
                if hll_val.is_none():
                    pass
                elif hll_val.type.value == ValueType.HLL:
                    hll_merge(temp_hll, hll_val.as_hll())
                else:
                    merge_error = True
                    break
            
            var count: Int64 = 0
            if not merge_error:
                count = Int64(hll_count(temp_hll))
            
            temp_hll.unsafe_free()
            
            if merge_error:
                return IntCmdResult(0, False)
            else:
                return IntCmdResult(count, True)

