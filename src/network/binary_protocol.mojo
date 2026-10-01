"""Binary protocol for M14 KV cache operations.

Designed for sub-100us round-trip latency. Used by Phase 2 (layer-granular)
and Phase 3 (externalized attention) where RESP overhead is too high.

Frame format:
  Request:  [magic:2B=0xCA5E][cmd:1B][body_len:4B LE][body:variable]
  Response: [magic:2B=0xCA5E][status:1B][body_len:4B LE][body:variable]

Commands:
  0x01  KV_STORE          Phase 1 (binary path)
  0x02  KV_FETCH          Phase 1 (binary path)
  0x10  LAYER_STORE       Phase 2
  0x11  LAYER_FETCH       Phase 2
  0x12  LAYER_FETCH_BATCH Phase 2
  0x13  LAYER_EXTEND      Phase 2
  0x20  ATTEND_CREATE          Phase 3
  0x21  ATTEND_STORE           Phase 3
  0x22  ATTEND_FINALIZE        Phase 3
  0x23  ATTEND_QUERY           Phase 3
  0x24  ATTEND_PREFIX_QUERY_FUSED  gh #50 — Stage 2 fast lane (mirrors ATTEND.PREFIX.QUERY_FUSED RESP command)
  0xFF  PING              All phases

Status:
  0x00  OK
  0x01  MISS (no data found)
  0x02  ERROR
"""

from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset

# Protocol constants
comptime BINARY_MAGIC = UInt16(0xCA5E)
comptime BINARY_HEADER_SIZE = 7  # magic(2) + cmd(1) + body_len(4)
comptime BINARY_RESP_HEADER_SIZE = 7  # magic(2) + status(1) + body_len(4)

# Commands
comptime CMD_KV_STORE = UInt8(0x01)
comptime CMD_KV_FETCH = UInt8(0x02)
comptime CMD_LAYER_STORE = UInt8(0x10)
comptime CMD_LAYER_FETCH = UInt8(0x11)
comptime CMD_LAYER_FETCH_BATCH = UInt8(0x12)
comptime CMD_LAYER_EXTEND = UInt8(0x13)
comptime CMD_ATTEND_CREATE = UInt8(0x20)
comptime CMD_ATTEND_STORE = UInt8(0x21)
comptime CMD_ATTEND_FINALIZE = UInt8(0x22)
comptime CMD_ATTEND_QUERY = UInt8(0x23)
# gh #50: Stage 2 fast lane — mirrors ATTEND.PREFIX.QUERY_FUSED RESP command
# (handle_attend_prefix_query_fused in src/commands/attend_prefix.mojo) but
# trades the RESP framing tax for a single 0xCA5E binary frame on port+1.
comptime CMD_ATTEND_PREFIX_QUERY_FUSED = UInt8(0x24)
# gh #61 Phase-0 → Stage 1: MOE.EXPERT.* substrate.
# Read-only command surface against a pre-populated tier of MoE expert
# weights — operators can share one tier across N consumers on the same
# host (or across the cluster) instead of each consumer loading the
# expert weights into its own unified memory.
#
# Stage 1 (this drop): read-only handlers — FETCH, INFO, STATS. Returns
# STATUS_MISS / "-UNAVAILABLE" until the backend tier lands in a follow-on.
# Wire surface registered now so consumers (pion_moe_tier in-proc and
# the future cross-process Pion-server-backed variant) can target the
# same protocol.
comptime CMD_MOE_EXPERT_FETCH    = UInt8(0x31)
comptime CMD_MOE_EXPERT_PREFETCH = UInt8(0x32)
comptime CMD_MOE_EXPERT_PIN      = UInt8(0x33)
comptime CMD_MOE_EXPERT_UNPIN    = UInt8(0x34)
comptime CMD_MOE_EXPERT_INFO     = UInt8(0x35)
comptime CMD_MOE_EXPERT_STATS    = UInt8(0x36)
# 0x30 (CMD_MOE_EXPERT_STORE) reserved for Stage 2 writeable tier.
# gh #100 (C2): binary-lane authentication. Body is the raw password bytes.
# Only required when --requirepass is set; until an AUTH frame succeeds on a
# binary connection, every other command returns STATUS_ERROR. Mirrors the RESP
# AUTH gate so the port+1 fast lane is not an unauthenticated bypass.
comptime CMD_AUTH = UInt8(0x37)
comptime CMD_PING = UInt8(0xFF)

# Status codes
comptime STATUS_OK = UInt8(0x00)
comptime STATUS_MISS = UInt8(0x01)
comptime STATUS_ERROR = UInt8(0x02)
# gh #67: distinct status for the cold-tier "session was here, has been
# evicted, V-store can rehydrate" case. The RESP lane returns "-COLDMISS ..."
# for the same condition; the consumer maps both to the same retry path
# (KV.PREFIX.WARM + retry).
comptime STATUS_COLDMISS = UInt8(0x03)


@always_inline
def write_u16_le(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int, value: UInt16):
    """Write a uint16 in little-endian to buffer at offset."""
    buf[unsafe_offset=offset] = UInt8(value & 0xFF)
    buf[unsafe_offset=offset + 1] = UInt8((value >> 8) & 0xFF)

@always_inline
def write_u32_le(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int, value: UInt32):
    """Write a uint32 in little-endian to buffer at offset."""
    buf[unsafe_offset=offset] = UInt8(value & 0xFF)
    buf[unsafe_offset=offset + 1] = UInt8((value >> 8) & 0xFF)
    buf[unsafe_offset=offset + 2] = UInt8((value >> 16) & 0xFF)
    buf[unsafe_offset=offset + 3] = UInt8((value >> 24) & 0xFF)

@always_inline
def read_u16_le(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int) -> UInt16:
    """Read a uint16 in little-endian from buffer at offset."""
    return UInt16(buf[unsafe_offset=offset]) | (UInt16(buf[unsafe_offset=offset + 1]) << 8)

@always_inline
def read_u32_le(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int) -> UInt32:
    """Read a uint32 in little-endian from buffer at offset."""
    return (UInt32(buf[unsafe_offset=offset])
          | (UInt32(buf[unsafe_offset=offset + 1]) << 8)
          | (UInt32(buf[unsafe_offset=offset + 2]) << 16)
          | (UInt32(buf[unsafe_offset=offset + 3]) << 24))


struct BinaryRequest(TrivialRegisterPassable):
    """Parsed binary protocol request."""
    var cmd: UInt8
    var body_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var body_len: UInt32
    var valid: Bool

    def __init__(out self):
        self.cmd = 0
        self.body_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.body_len = 0
        self.valid = False


@always_inline
def parse_binary_request(
    buf: Pointer[UInt8, MutUntrackedOrigin],
    buf_len: Int,
) -> BinaryRequest:
    """Parse a binary protocol request from buffer. Returns invalid request if incomplete."""
    var req = BinaryRequest()
    if buf_len < BINARY_HEADER_SIZE:
        return req

    # Check magic
    var magic = read_u16_le(buf, 0)
    if magic != BINARY_MAGIC:
        return req

    req.cmd = buf[unsafe_offset=2]
    req.body_len = read_u32_le(buf, 3)

    var total_len = BINARY_HEADER_SIZE + Int(req.body_len)
    if buf_len < total_len:
        return req  # Incomplete — need more data

    req.body_ptr = buf.unsafe_offset(BINARY_HEADER_SIZE)
    req.valid = True
    return req


@always_inline
def build_binary_response(
    buf: Pointer[UInt8, MutUntrackedOrigin],
    status: UInt8,
    body_ptr: Pointer[UInt8, MutUntrackedOrigin],
    body_len: UInt32,
) -> Int:
    """Build a binary response into buf. Returns total bytes written."""
    write_u16_le(buf, 0, BINARY_MAGIC)
    buf[unsafe_offset=2] = status
    write_u32_le(buf, 3, body_len)
    if body_len > 0 and is_not_null(body_ptr):
        unsafe_memcpy(dest=buf.unsafe_offset(BINARY_RESP_HEADER_SIZE), src=body_ptr, count=Int(body_len))
    return BINARY_RESP_HEADER_SIZE + Int(body_len)


@always_inline
def build_binary_response_ok(buf: Pointer[UInt8, MutUntrackedOrigin]) -> Int:
    """Build an OK response with no body."""
    return build_binary_response(buf, STATUS_OK, null_ptr[UInt8, MutUntrackedOrigin](), 0)

@always_inline
def build_binary_response_miss(buf: Pointer[UInt8, MutUntrackedOrigin]) -> Int:
    """Build a MISS response with no body."""
    return build_binary_response(buf, STATUS_MISS, null_ptr[UInt8, MutUntrackedOrigin](), 0)


struct LayerStoreRequest(TrivialRegisterPassable):
    """Parsed LAYER.STORE request body."""
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: UInt16
    var layer_id: UInt16
    var tensor_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var tensor_len: Int
    var valid: Bool

    def __init__(out self):
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.layer_id = 0
        self.tensor_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.tensor_len = 0
        self.valid = False


@always_inline
def parse_layer_store_request(
    body: Pointer[UInt8, MutUntrackedOrigin],
    body_len: Int,
) -> LayerStoreRequest:
    """Parse LAYER_STORE body: [session_id_len:2][session_id][layer_id:2][tensor]"""
    var req = LayerStoreRequest()
    if body_len < 4:  # minimum: 2 bytes session_id_len + 2 bytes layer_id
        return req

    req.session_id_len = read_u16_le(body, 0)
    var sid_end = 2 + Int(req.session_id_len)
    if body_len < sid_end + 2:
        return req

    req.session_id_ptr = body.unsafe_offset(2)
    req.layer_id = read_u16_le(body, sid_end)
    var tensor_start = sid_end + 2
    req.tensor_ptr = body.unsafe_offset(tensor_start)
    req.tensor_len = body_len - tensor_start
    req.valid = True
    return req


struct LayerFetchRequest(TrivialRegisterPassable):
    """Parsed LAYER.FETCH request body."""
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: UInt16
    var layer_id: UInt16
    var valid: Bool

    def __init__(out self):
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.layer_id = 0
        self.valid = False


@always_inline
def parse_layer_fetch_request(
    body: Pointer[UInt8, MutUntrackedOrigin],
    body_len: Int,
) -> LayerFetchRequest:
    """Parse LAYER_FETCH body: [session_id_len:2][session_id][layer_id:2]"""
    var req = LayerFetchRequest()
    if body_len < 4:
        return req

    req.session_id_len = read_u16_le(body, 0)
    var sid_end = 2 + Int(req.session_id_len)
    if body_len < sid_end + 2:
        return req

    req.session_id_ptr = body.unsafe_offset(2)
    req.layer_id = read_u16_le(body, sid_end)
    req.valid = True
    return req


# ── ATTEND command parsers ──

struct AttendCreateRequest(TrivialRegisterPassable):
    """Parsed ATTEND_CREATE body:
    [sid_len:2][sid][key_dim:2][val_dim:2]
    M4 extension (detected by body_len > base):
    [settings_flags:1] [kquant:1 if bit0] [vquant:1 if bit1] [boundary_n:1 + boundary_vquant:1 if bit2]
    """
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: UInt16
    var key_dim: UInt16
    var val_dim: UInt16
    var valid: Bool
    # M4 quant settings (default 0 = INT8/INT8/no boundary)
    var k_format: UInt8
    var v_format: UInt8
    var boundary_n: UInt8
    var boundary_v_format: UInt8

    def __init__(out self):
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.key_dim = 0
        self.val_dim = 0
        self.valid = False
        self.k_format = 0
        self.v_format = 0
        self.boundary_n = 0
        self.boundary_v_format = 0

@always_inline
def parse_attend_create(body: Pointer[UInt8, MutUntrackedOrigin], body_len: Int) -> AttendCreateRequest:
    var req = AttendCreateRequest()
    if body_len < 6: return req
    req.session_id_len = read_u16_le(body, 0)
    var sid_end = 2 + Int(req.session_id_len)
    if body_len < sid_end + 4: return req
    req.session_id_ptr = body.unsafe_offset(2)
    req.key_dim = read_u16_le(body, sid_end)
    req.val_dim = read_u16_le(body, sid_end + 2)
    req.valid = True
    # M4: parse optional settings block if body is longer than base size
    var base_end = sid_end + 4
    if body_len > base_end:
        var flags = body[unsafe_offset=base_end]
        var off = base_end + 1
        if (flags & 1) and off < body_len:
            req.k_format = body[unsafe_offset=off]; off += 1
        if (flags & 2) and off < body_len:
            req.v_format = body[unsafe_offset=off]; off += 1
        if (flags & 4) and off + 1 < body_len:
            req.boundary_n = body[unsafe_offset=off]; off += 1
            req.boundary_v_format = body[unsafe_offset=off]; off += 1
    return req


struct AttendStoreRequest(TrivialRegisterPassable):
    """Parsed ATTEND_STORE body: [sid_len:2][sid][layer_id:2][num_tokens:4][keys][values]"""
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: UInt16
    var layer_id: UInt16
    var num_tokens: UInt32
    var keys_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var values_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var valid: Bool

    def __init__(out self):
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.layer_id = 0
        self.num_tokens = 0
        self.keys_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.values_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.valid = False

@always_inline
def parse_attend_store(body: Pointer[UInt8, MutUntrackedOrigin], body_len: Int,
                      key_dim: Int, val_dim: Int) -> AttendStoreRequest:
    var req = AttendStoreRequest()
    if body_len < 8: return req
    req.session_id_len = read_u16_le(body, 0)
    var sid_end = 2 + Int(req.session_id_len)
    if body_len < sid_end + 6: return req
    req.session_id_ptr = body.unsafe_offset(2)
    req.layer_id = read_u16_le(body, sid_end)
    req.num_tokens = read_u32_le(body, sid_end + 2)
    var data_start = sid_end + 6
    var keys_bytes = Int(req.num_tokens) * key_dim * 4
    var vals_bytes = Int(req.num_tokens) * val_dim * 4
    if body_len < data_start + keys_bytes + vals_bytes: return req
    req.keys_ptr = body.unsafe_offset(data_start)
    req.values_ptr = body.unsafe_offset(data_start).unsafe_offset(keys_bytes)
    req.valid = True
    return req


struct AttendQueryRequest(TrivialRegisterPassable):
    """Parsed ATTEND_QUERY body: [sid_len:2][sid][layer_id:2][k:2][query_fp32]"""
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: UInt16
    var layer_id: UInt16
    var k: UInt16
    var query_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var valid: Bool

    def __init__(out self):
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.layer_id = 0
        self.k = 0
        self.query_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.valid = False

@always_inline
def parse_attend_query(body: Pointer[UInt8, MutUntrackedOrigin], body_len: Int) -> AttendQueryRequest:
    var req = AttendQueryRequest()
    if body_len < 6: return req
    req.session_id_len = read_u16_le(body, 0)
    var sid_end = 2 + Int(req.session_id_len)
    if body_len < sid_end + 4: return req
    req.session_id_ptr = body.unsafe_offset(2)
    req.layer_id = read_u16_le(body, sid_end)
    req.k = read_u16_le(body, sid_end + 2)
    req.query_ptr = body.unsafe_offset(sid_end).unsafe_offset(4)
    req.valid = True
    return req


# gh #50: ATTEND.PREFIX.QUERY_FUSED binary frame (CMD_ATTEND_PREFIX_QUERY_FUSED = 0x24).
# Body layout (LE):
#   [sid_len:2][sid]
#   [layer_id:2][H_q:2][D:2][H_kv:2][M:4][S_suf:4]
#   [Q : H_q*M*D float32 = H_q*M*D*4 bytes]
#   [K_suf : H_kv*S_suf*D float32 = H_kv*S_suf*D*4 bytes]   # zero-length when S_suf == 0
#   [V_suf : H_kv*S_suf*D float32 = H_kv*S_suf*D*4 bytes]   # zero-length when S_suf == 0
#   [head_map : H_q uint8]
#   [fa_window : 4 bytes uint32]   # gh #60 Step 1: OPTIONAL trailing field —
#                                    # absent on legacy frames; present means
#                                    # this call's sliding-window override
#                                    # (0 = full attention; >0 = last N tokens).
# M and S_suf are uint32 to handle long-context decode (S_suf > 65535 possible
# at 128K contexts) and prefill batching. The H_q/D/H_kv/layer_id u16 limits
# (≤65535) are safe given every published transformer (≤256 heads, ≤2048 head
# dim, ≤1024 layers).
# Reply body: H_q*M*D float32 = H_q*M*D*4 bytes (merged attention output, no LSE trailer).
struct AttendPrefixQueryFusedRequest(TrivialRegisterPassable):
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: UInt16
    var layer_id: UInt16
    var H_q: UInt16
    var D: UInt16
    var H_kv: UInt16
    var M: UInt32
    var S_suf: UInt32
    var Q_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var K_suf_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var V_suf_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var head_map_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var fa_window: Int32     # gh #60 Step 1: -1 = no override (use engine default).
    var valid: Bool

    def __init__(out self):
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.layer_id = 0
        self.H_q = 0
        self.D = 0
        self.H_kv = 0
        self.M = 0
        self.S_suf = 0
        self.Q_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.K_suf_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.V_suf_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.head_map_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.fa_window = -1
        self.valid = False


@always_inline
def parse_attend_prefix_query_fused(body: Pointer[UInt8, MutUntrackedOrigin],
                                     body_len: Int) -> AttendPrefixQueryFusedRequest:
    var req = AttendPrefixQueryFusedRequest()
    # Minimum: 2 (sid_len) + 16 (4 u16 + 2 u32 fixed fields) — plus sid bytes
    if body_len < 18: return req
    req.session_id_len = read_u16_le(body, 0)
    var sid_end = 2 + Int(req.session_id_len)
    if body_len < sid_end + 16: return req
    req.session_id_ptr = body.unsafe_offset(2)
    req.layer_id = read_u16_le(body, sid_end)
    req.H_q = read_u16_le(body, sid_end + 2)
    req.D = read_u16_le(body, sid_end + 4)
    req.H_kv = read_u16_le(body, sid_end + 6)
    req.M = read_u32_le(body, sid_end + 8)
    req.S_suf = read_u32_le(body, sid_end + 12)
    var H_q_i = Int(req.H_q)
    var D_i = Int(req.D)
    var M_i = Int(req.M)
    var S_suf_i = Int(req.S_suf)
    var H_kv_i = Int(req.H_kv)
    if H_q_i <= 0 or D_i <= 0 or M_i <= 0 or H_kv_i <= 0 or S_suf_i < 0:
        return req
    if (H_q_i % H_kv_i) != 0:
        return req
    var q_bytes = H_q_i * M_i * D_i * 4
    var ks_bytes = H_kv_i * S_suf_i * D_i * 4
    var hm_bytes = H_q_i
    var data_start = sid_end + 16
    var need = q_bytes + ks_bytes + ks_bytes + hm_bytes
    if body_len < data_start + need:
        return req
    req.Q_ptr = body.unsafe_offset(data_start)
    req.K_suf_ptr = body.unsafe_offset(data_start).unsafe_offset(q_bytes)
    req.V_suf_ptr = body.unsafe_offset(data_start).unsafe_offset(q_bytes).unsafe_offset(ks_bytes)
    req.head_map_ptr = body.unsafe_offset(data_start).unsafe_offset(q_bytes).unsafe_offset(ks_bytes) + ks_bytes
    # gh #60 Step 1: optional trailing 4-byte fa_window override.
    # Old consumers send exactly `data_start + need` bytes → fa_window stays -1.
    # New consumers append a u32 → parse it. Forward-compatible.
    if body_len >= data_start + need + 4:
        req.fa_window = Int32(read_u32_le(body, data_start + need))
    req.valid = True
    return req
