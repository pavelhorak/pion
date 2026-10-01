"""KV.PREFIX.* — production wrapper around V.STOREBATCH/V.FETCH RANGE.

Design invariant: the cache namespace must encode the FULL execution context
(model, dtype, quantisation, layer set) — two prefixes that differ in any of
them are different cache entries, or a warm read returns another model's K/V.

Three commands:
  KV.PREFIX.REGISTER <ns_key> <kv_dim> <vquant>
    Creates V-store sessions "<ns_key>_pk" and "<ns_key>_pv". The namespace key
    is the contract — callers must hash full execution context into it
    (model | tokenizer | rope_theta | quant | adapter | prompt). After REGISTER,
    standard V.STOREBATCH / V.FETCH RANGE work against the derived sids.
    Returns +OK on success or -ERR if V-store is full / disabled.

  KV.PREFIX.LOOKUP <ns_key>
    Returns "+HIT" if both sessions exist for this namespace, "+MISS" otherwise.
    Used by clients to decide between cold prefill (then REGISTER + STOREBATCH)
    and warm fetch (V.FETCH RANGE).

  KV.PREFIX.INFO
    Global stats: registered prefix count, total tokens cached.
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from src.io.wal import WAL
from src.network.v_store import (
    VStoreIndex, VFMT_INT8, VFMT_TURBO4, VFMT_TURBO3, VFMT_TURBO2, VFMT_FP16,
    VFMT_MLX4G32, VFMT_FP8,
    vstore_dir_register, vstore_dir_lookup, vstore_dir_drop,
    vstore_dir_get_schema_digest,
    MAX_DIR_ENTRIES, MAX_VS_LAYERS, MAX_VS_SESSIONS,
)
from src.commands.ssm_prefix import ssm_save_snapshot
from src.network.metal_attention_engine import MetalAttentionEngine
from src.network.response_writer import ResponseWriter
from src.common.metrics import ValueLedger
from src.common.utils import strict_atol, arg_eq
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from std.collections import Array
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.atomic import Atomic, Ordering
from std.ffi import external_call


# Returned when the requested format is not a format. Callers MUST check and
# refuse: this used to be `VFMT_INT8`, so `KV.PREFIX.REGISTER ns dim fp_16`
# answered +OK and quietly built an int8 prefix. A prompt cache asking for
# fp16 and silently getting int8 is a fidelity contract broken with no signal
# — the same silent-substitution class gh #148 fixed once already.
comptime VFMT_UNKNOWN = UInt8(255)


@always_inline
def _parse_vquant(v_in: String) -> UInt8:
    var v = v_in.lower()   # lenient about case, strict about the set
    if v == "int8":   return VFMT_INT8
    if v == "turbo4": return VFMT_TURBO4
    if v == "turbo3": return VFMT_TURBO3
    if v == "turbo2": return VFMT_TURBO2
    if v == "fp16":   return VFMT_FP16
    if v == "fp8":    return VFMT_FP8
    # gh #148 Phase 1: mlx QuantizedKVCache layout (int4, group 32, affine).
    # The one config that fits a 4B/86K-token resident cartridge on 16 GB.
    if v == "mlx4g32": return VFMT_MLX4G32
    return VFMT_UNKNOWN


def _safe_relative_path(p: String) -> Bool:
    """True if `p` can only name a file inside the working directory.

    Rejects: empty, absolute (`/...`), home-relative (`~...`), any `..`
    component, and any NUL or control byte (a NUL would also truncate the C
    string handed to `creat`). Ordinary names and subdirectories are allowed,
    so `KV.PREFIX.SAVE snapshots/v1.bin` still works.
    """
    var n = p.byte_length()
    if n == 0:
        return False
    var b = p.as_bytes()
    if b[0] == 47 or b[0] == 126:        # '/' or '~'
        return False
    for i in range(n):
        var c = b[i]
        if c < 32 or c == 127:           # NUL and control bytes
            return False
    # Reject a ".." that is a whole path component: "..", "../x", "x/..",
    # "x/../y". A name that merely contains dots ("v..1") is fine.
    for i in range(n - 1):
        if b[i] == 46 and b[i + 1] == 46:            # ".."
            var at_start = (i == 0) or (b[i - 1] == 47)
            var at_end = (i + 2 >= n) or (b[i + 2] == 47)
            if at_start and at_end:
                return False
    return True


@always_inline
def _ns_prefix_check(ns_ptr: Pointer[UInt8, MutUntrackedOrigin], ns_len: Int,
                     vstore: VStoreIndex) -> Bool:
    """Multi-tenant gate: when --ns-prefix is configured, the supplied
    namespace must start with that exact prefix. Anonymous (un-configured)
    deployments accept any namespace. Returns True if the namespace passes."""
    var pl = vstore.ns_prefix.byte_length()
    if pl == 0:
        return True
    if ns_len < pl:
        return False
    var pp = vstore.ns_prefix.unsafe_ptr()
    for i in range(pl):
        if ns_ptr[unsafe_offset=i] != pp[unsafe_offset=i]:
            return False
    return True


@always_inline
def handle_kv_prefix_register(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """KV.PREFIX.REGISTER <ns_key> <kv_dim> <vquant> [BLOCKS <block_size> <hash_count> <blob>] [PREFILL_MS <ms>]

    gh #262: optional `PREFILL_MS <ms>` is the client's MEASURED cold prefill
    time for this prefix. Every later KV.PREFIX.LOOKUP hit on it is credited
    with exactly that time in the value receipt (`PION.STATS`, `INFO`); a
    prefix registered without it is credited a per-token estimate instead.

    gh #71: optional trailing `BLOCKS` clause carries the within-prefix block
    hash table (u64 LE packed). Stored on the K-side session only; queried by
    `KV.PREFIX.BLOCKS` / `KV.PREFIX.MEMBERSHIP`.
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled (use --kvcache)")
        return 1
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR KV.PREFIX.REGISTER requires: ns_key kv_dim vquant")
        return 1

    var ns_tok = tokens[unsafe_offset=start + 1]
    var ns_ptr = ns_tok.ptr
    var ns_len = Int(ns_tok.length)
    if ns_len == 0 or ns_len > 200:
        writer.append_error_response("ERR ns_key length must be 1..200")
        return 1
    if not _ns_prefix_check(ns_ptr, ns_len, vstore):
        writer.append_error_response("ERR namespace must start with required prefix " +
                                     "'" + vstore.ns_prefix + "' (multi-tenant isolation)")
        return 1

    var kv_dim = strict_atol(tokens[unsafe_offset=start + 2].value())
    if kv_dim <= 0 or kv_dim > 8192:
        writer.append_error_response("ERR kv_dim out of range")
        return 1

    var vquant = _parse_vquant(tokens[unsafe_offset=start + 3].text_value())
    if vquant == VFMT_UNKNOWN:
        writer.append_error_response(
            "ERR KV.PREFIX.REGISTER: unknown vquant '"
            + tokens[unsafe_offset=start + 3].value()
            + "' (want int8|turbo4|turbo3|turbo2|fp16|fp8|mlx4g32)")
        return 1

    # gh #71: optional trailing BLOCKS clause. We expect exactly 4 extra tokens:
    #   tokens[start+4] = "BLOCKS" (case-insensitive)
    #   tokens[start+5] = block_size (decimal)
    #   tokens[start+6] = block_count (decimal)
    #   tokens[start+7] = hash blob (binary bulk string, block_count * 8 bytes LE)
    var have_blocks = False
    var blk_size: UInt32 = UInt32(0)
    var blk_count: UInt32 = UInt32(0)
    var blk_blob_ptr: Pointer[UInt8, MutUntrackedOrigin] = null_ptr[UInt8, MutUntrackedOrigin]()
    if start + 4 < num_tokens:
        var blk_tok = tokens[unsafe_offset=start + 4]
        var bp = blk_tok.ptr
        var bl = Int(blk_tok.length)
        if bl == 6 and (Int(bp[unsafe_offset=0]) | 0x20) == 98 and (Int(bp[unsafe_offset=1]) | 0x20) == 108 \
           and (Int(bp[unsafe_offset=2]) | 0x20) == 111 and (Int(bp[unsafe_offset=3]) | 0x20) == 99 \
           and (Int(bp[unsafe_offset=4]) | 0x20) == 107 and (Int(bp[unsafe_offset=5]) | 0x20) == 115:
            if start + 7 >= num_tokens:
                writer.append_error_response("ERR BLOCKS requires: block_size hash_count hash_blob")
                return 1
            var bs = strict_atol(tokens[unsafe_offset=start + 5].value())
            var bc = strict_atol(tokens[unsafe_offset=start + 6].value())
            if bs <= 0 or bs > 8192:
                writer.append_error_response("ERR BLOCKS block_size out of range (1..8192)")
                return 1
            if bc < 0 or bc > (1 << 20):
                writer.append_error_response("ERR BLOCKS hash_count out of range (0..1048576)")
                return 1
            var blob_tok = tokens[unsafe_offset=start + 7]
            if Int(blob_tok.length) != Int(bc) * 8:
                writer.append_error_response("ERR BLOCKS hash_blob length mismatch")
                return 1
            have_blocks = True
            blk_size = UInt32(bs)
            blk_count = UInt32(bc)
            blk_blob_ptr = blob_tok.ptr

    # gh #262: optional trailing `PREFILL_MS <ms>` (after BLOCKS if present).
    # Bounded by `num_tokens`, which the dispatch arm passes as cmd_end_tok, so
    # the scan cannot read the next pipelined command as an option.
    var prefill_us: UInt64 = UInt64(0)
    var _pj = (start + 8) if have_blocks else (start + 4)
    while _pj + 1 < num_tokens:
        var _pt = tokens[unsafe_offset=_pj]
        if arg_eq(_pt.ptr, Int(_pt.length), "prefill_ms"):
            var _pm = strict_atol(tokens[unsafe_offset=_pj + 1].value())
            if _pm < 0 or _pm > 86_400_000:
                writer.append_error_response("ERR PREFILL_MS out of range (0..86400000)")
                return 1
            prefill_us = UInt64(_pm) * UInt64(1000)
            _pj += 2
        else:
            break

    # Build sid_k = "<ns>_pk"
    var total = ns_len + 3
    var _buf_k = alloc[UInt8](total)
    var sid_k_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_buf_k))
    unsafe_memcpy(dest=sid_k_ptr, src=ns_ptr, count=ns_len)
    sid_k_ptr[unsafe_offset=ns_len] = UInt8(95)      # '_'
    sid_k_ptr[unsafe_offset=ns_len + 1] = UInt8(112) # 'p'
    sid_k_ptr[unsafe_offset=ns_len + 2] = UInt8(107) # 'k'

    var _buf_v = alloc[UInt8](total)
    var sid_v_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_buf_v))
    unsafe_memcpy(dest=sid_v_ptr, src=ns_ptr, count=ns_len)
    sid_v_ptr[unsafe_offset=ns_len] = UInt8(95)
    sid_v_ptr[unsafe_offset=ns_len + 1] = UInt8(112)
    sid_v_ptr[unsafe_offset=ns_len + 2] = UInt8(118) # 'v'

    var idx_k = vstore.create_session(sid_k_ptr, total, Int(kv_dim), vquant)
    var idx_v = vstore.create_session(sid_v_ptr, total, Int(kv_dim), vquant)
    if idx_k >= 0:
        vstore.wal_append_create(sid_k_ptr, total, Int(kv_dim), vquant)
    if idx_v >= 0:
        vstore.wal_append_create(sid_v_ptr, total, Int(kv_dim), vquant)
    if idx_k >= 0:
        vstore.prefix_prefill_us[idx_k] = prefill_us   # gh #262 (0 = not reported)

    # gh #71: attach the optional block-hash table to the K-side session and
    # log it as a WAL op=4 record. Must run AFTER wal_append_create so replay
    # order matches: CREATE → BLOCKS.
    if have_blocks and idx_k >= 0:
        _ = vstore.set_block_hashes(idx_k, blk_size, blk_count, blk_blob_ptr)
        vstore.wal_append_blocks(sid_k_ptr, total, blk_size, blk_count, blk_blob_ptr)

    # Cross-worker directory: publish that worker `my_worker_id` now owns these
    # two sessions. Other workers' KV.PREFIX.LOOKUP / INFO can answer for them
    # without forwarding. ts is just `wal_appended` — not strictly monotonic
    # across workers but good enough for staleness debugging.
    if is_not_null(vstore.directory) and vstore.my_worker_id >= 0 and idx_k >= 0:
        _ = vstore_dir_register(vstore.directory, sid_k_ptr, total,
                                Int(kv_dim), vquant, vstore.my_worker_id,
                                UInt64(vstore.wal_appended))
    if is_not_null(vstore.directory) and vstore.my_worker_id >= 0 and idx_v >= 0:
        _ = vstore_dir_register(vstore.directory, sid_v_ptr, total,
                                Int(kv_dim), vquant, vstore.my_worker_id,
                                UInt64(vstore.wal_appended))

    sid_k_ptr.unsafe_free()
    sid_v_ptr.unsafe_free()

    if idx_k < 0 or idx_v < 0:
        writer.append_error_response("ERR KV.PREFIX.REGISTER failed (no free V-store slots)")
        return 1

    writer.append_ok_response()
    return 1


@always_inline
def handle_kv_prefix_lookup(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
    mut ledger: ValueLedger,
) raises -> Int:
    """KV.PREFIX.LOOKUP <ns_key> [TOKENS <n>] [PREFILL_MS <ms>]  →  +HIT  or  +MISS

    gh #262: every answer is recorded in the value receipt. A local hit is
    credited with the prefix's token count (K-side layer 0) and its reported
    PREFILL_MS; a cross-worker directory hit counts as a hit with no token
    estimate (the tokens live on another worker).

    `TOKENS <n>` credits a hit with the n tokens the caller actually restored
    instead of the namespace's own count. A client whose reuse spans several
    namespaces (pion-vllm-mlx serve stores a lineage as a chain of segments
    and fetches rows with V.FETCH, never asking LOOKUP) names the leaf and the
    total here, once per restore. `PREFILL_MS <ms>` is that restore's measured
    cold prefill, when the caller has one; without it the tokens are credited
    at the labelled per-token estimate. Neither option changes the answer."""
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled")
        return 1
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR KV.PREFIX.LOOKUP requires: ns_key")
        return 1

    var ns_tok = tokens[unsafe_offset=start + 1]
    var ns_ptr = ns_tok.ptr
    var ns_len = Int(ns_tok.length)
    if not _ns_prefix_check(ns_ptr, ns_len, vstore):
        writer.append_error_response("ERR namespace must start with required prefix " +
                                     "'" + vstore.ns_prefix + "' (multi-tenant isolation)")
        return 1
    # Options, bounded by `num_tokens` (the dispatch arm passes cmd_end_tok).
    # Parsed before any lookup so a malformed one answers -ERR and records
    # nothing in the ledger.
    var opt_tokens = -1
    var opt_prefill_us = UInt64(0)
    var _oj = start + 2
    while _oj < num_tokens:
        var _ot = tokens[unsafe_offset=_oj]
        if _oj + 1 >= num_tokens:
            writer.append_error_response("ERR KV.PREFIX.LOOKUP option needs a value")
            return 1
        if arg_eq(_ot.ptr, Int(_ot.length), "tokens"):
            var _tn = strict_atol(tokens[unsafe_offset=_oj + 1].value())
            if _tn < 0 or _tn > 100_000_000:
                writer.append_error_response("ERR TOKENS out of range (0..100000000)")
                return 1
            opt_tokens = _tn
        elif arg_eq(_ot.ptr, Int(_ot.length), "prefill_ms"):
            var _pm = strict_atol(tokens[unsafe_offset=_oj + 1].value())
            if _pm < 0 or _pm > 86_400_000:
                writer.append_error_response("ERR PREFILL_MS out of range (0..86400000)")
                return 1
            opt_prefill_us = UInt64(_pm) * UInt64(1000)
        else:
            writer.append_error_response("ERR KV.PREFIX.LOOKUP unknown option (TOKENS, PREFILL_MS)")
            return 1
        _oj += 2

    var total = ns_len + 3

    var _buf_k = alloc[UInt8](total)
    var sid_k_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_buf_k))
    unsafe_memcpy(dest=sid_k_ptr, src=ns_ptr, count=ns_len)
    sid_k_ptr[unsafe_offset=ns_len] = UInt8(95); sid_k_ptr[unsafe_offset=ns_len + 1] = UInt8(112); sid_k_ptr[unsafe_offset=ns_len + 2] = UInt8(107)

    var _buf_v = alloc[UInt8](total)
    var sid_v_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_buf_v))
    unsafe_memcpy(dest=sid_v_ptr, src=ns_ptr, count=ns_len)
    sid_v_ptr[unsafe_offset=ns_len] = UInt8(95); sid_v_ptr[unsafe_offset=ns_len + 1] = UInt8(112); sid_v_ptr[unsafe_offset=ns_len + 2] = UInt8(118)

    # find_and_touch updates LRU timestamp on hit so a hot prefix doesn't
    # get evicted by REGISTER pressure.
    var idx_k = vstore.find_and_touch(sid_k_ptr, total)
    var idx_v = vstore.find_and_touch(sid_v_ptr, total)
    var hit = (idx_k >= 0 and idx_v >= 0)

    # Cross-worker fallback: if not local, ask the shared directory whether
    # any worker owns this namespace. This is the path that fixed multi-worker
    # KV.PREFIX (replaces the old `--kvcache` -w 1 auto-cap).
    if not hit and is_not_null(vstore.directory):
        var dk = vstore_dir_lookup(vstore.directory, sid_k_ptr, total)
        var dv = vstore_dir_lookup(vstore.directory, sid_v_ptr, total)
        hit = (dk >= 0 and dv >= 0)

    var _lk_tokens = 0
    var _lk_prefill_us = UInt64(0)
    if hit and idx_k >= 0:
        _lk_tokens = vstore.tokens_per_layer[idx_k * MAX_VS_LAYERS]
        _lk_prefill_us = vstore.prefix_prefill_us[idx_k]
    if hit and opt_tokens >= 0:
        # The caller's own count of what it restored replaces the namespace's;
        # so does its measured time (a registered PREFILL_MS was for the
        # namespace's own tokens, not for this restore).
        _lk_tokens = opt_tokens
        _lk_prefill_us = opt_prefill_us
    elif hit and opt_prefill_us > UInt64(0):
        _lk_prefill_us = opt_prefill_us

    sid_k_ptr.unsafe_free()
    sid_v_ptr.unsafe_free()

    if hit:
        ledger.record_kvprefix_hit(_lk_tokens, _lk_prefill_us)
        writer.append_to_response("+HIT\r\n".unsafe_ptr(), 6)
    else:
        ledger.record_kvprefix_miss()
        writer.append_to_response("+MISS\r\n".unsafe_ptr(), 7)
    return 1


def handle_kv_prefix_drop(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """KV.PREFIX.DROP <ns_key> [<ns_key> ...]  →  :N  (namespaces dropped)

    Frees the K and V sessions of each registered prefix and logs the drop to
    the V-store WAL (op 3), so a restart does not replay the rows back. A
    namespace that is not held counts 0 — dropping is idempotent.

    Exists for callers that own an eviction policy the server cannot see:
    pion-vllm-mlx serve keeps its lineages under a byte budget and evicts
    leaf segments first. The V-store's own LRU (session-count pressure in
    create_session) evicts by last access alone, which for a lineage is its
    ROOT — written once, never touched again while the leaf grows — and a
    lineage without its root restores nothing."""
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled")
        return 1
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR KV.PREFIX.DROP requires: ns_key [ns_key ...]")
        return 1
    # Validate every namespace before dropping any (a refusal is a no-op).
    for j in range(start + 1, num_tokens):
        var t = tokens[unsafe_offset=j]
        if not _ns_prefix_check(t.ptr, Int(t.length), vstore):
            writer.append_error_response("ERR namespace must start with required prefix " +
                                         "'" + vstore.ns_prefix + "' (multi-tenant isolation)")
            return 1
    var dropped = 0
    for j in range(start + 1, num_tokens):
        var t = tokens[unsafe_offset=j]
        var ns_ptr = t.ptr
        var ns_len = Int(t.length)
        var total = ns_len + 3
        var _b = alloc[UInt8](total)
        var sid = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_b))
        unsafe_memcpy(dest=sid, src=ns_ptr, count=ns_len)
        sid[unsafe_offset=ns_len] = UInt8(95); sid[unsafe_offset=ns_len + 1] = UInt8(112)
        var any = False
        for side in range(2):
            sid[unsafe_offset=ns_len + 2] = UInt8(107) if side == 0 else UInt8(118)   # _pk / _pv
            var idx = vstore._find_session(sid, total)
            if idx >= 0:
                vstore.wal_append_drop(sid, total)
                _ = vstore.drop_session(idx)
                any = True
        sid.unsafe_free()
        if any:
            dropped += 1
    writer.append_int_response(Int64(dropped))
    return 1


@always_inline
def handle_kv_prefix_owner(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    vstore: VStoreIndex,
) raises -> Int:
    """KV.PREFIX.OWNER <ns_key>  →  +N  (owner worker_id) or +-1 (no owner)

    Cross-worker directory lookup that doesn't touch the local V-store. Lets
    callers discover which worker owns a namespace without paying the cost of
    a full V.* round-trip + redirect. Pairs with client-side auto-redirect in
    pion-lmcache.PionStore: on the first cold connection, OWNER tells the
    client which worker to pin to.
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled (use --kvcache)")
        return 1
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR KV.PREFIX.OWNER requires: ns_key")
        return 1
    var ns_tok = tokens[unsafe_offset=start + 1]
    var ns_ptr = ns_tok.ptr
    var ns_len = Int(ns_tok.length)
    if not _ns_prefix_check(ns_ptr, ns_len, vstore):
        writer.append_error_response("ERR namespace must start with required prefix " +
                                     "'" + vstore.ns_prefix + "' (multi-tenant isolation)")
        return 1
    var total = ns_len + 3

    # The directory entries are keyed on the K-side sid (<ns>_pk). Build it.
    var _buf_k = alloc[UInt8](total)
    var sid_k_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_buf_k))
    unsafe_memcpy(dest=sid_k_ptr, src=ns_ptr, count=ns_len)
    sid_k_ptr[unsafe_offset=ns_len] = UInt8(95); sid_k_ptr[unsafe_offset=ns_len + 1] = UInt8(112); sid_k_ptr[unsafe_offset=ns_len + 2] = UInt8(107)

    var owner = -1
    var schema_digest = UInt32(0)
    if is_not_null(vstore.directory):
        owner = vstore_dir_lookup(vstore.directory, sid_k_ptr, total)
        schema_digest = vstore_dir_get_schema_digest(vstore.directory, sid_k_ptr, total)
    sid_k_ptr.unsafe_free()

    # Bundle gh #29 §8.2: response shape extended from `+<owner>\r\n` to
    # `+<owner> <schema_digest>\r\n` when digest is non-zero. Legacy clients
    # parse `int(line.split()[0][1:])` and pick up just owner — backward
    # compatible. New clients use line.split()[1] to get the digest as a
    # decimal-encoded UInt32 ("0" means uniform/unset). Cross-instance
    # consumers compare digests to detect "same prefix sid, different
    # per-layer schema".
    var s: String
    if schema_digest != UInt32(0):
        s = String("+") + String(owner) + String(" ") + String(schema_digest) + String("\r\n")
    else:
        s = String("+") + String(owner) + String("\r\n")
    writer.append_to_response(s.unsafe_ptr(), s.byte_length())
    return 1


@always_inline
def handle_kv_prefix_blocks(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """KV.PREFIX.BLOCKS <ns_key>  →  bulk-string [block_size 4B LE][block_count 4B LE][hashes]  or  +UNKNOWN

    gh #71: returns the within-prefix block-hash table registered via
    `KV.PREFIX.REGISTER ... BLOCKS`. Used by cache-aware routers (and audit
    tooling) to reproduce the residency picture client-side, or to feed a
    `KV.PREFIX.MEMBERSHIP` probe with the canonical hash set.
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled (use --kvcache)")
        return 1
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR KV.PREFIX.BLOCKS requires: ns_key")
        return 1

    var ns_tok = tokens[unsafe_offset=start + 1]
    var ns_ptr = ns_tok.ptr
    var ns_len = Int(ns_tok.length)
    if ns_len == 0 or ns_len > 200:
        writer.append_error_response("ERR ns_key length must be 1..200")
        return 1
    if not _ns_prefix_check(ns_ptr, ns_len, vstore):
        writer.append_error_response("ERR namespace must start with required prefix " +
                                     "'" + vstore.ns_prefix + "' (multi-tenant isolation)")
        return 1
    var total = ns_len + 3

    # K-side session carries the block table (mirrors KV.PREFIX.OWNER's keying).
    var _buf_k = alloc[UInt8](total)
    var sid_k_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_buf_k))
    unsafe_memcpy(dest=sid_k_ptr, src=ns_ptr, count=ns_len)
    sid_k_ptr[unsafe_offset=ns_len] = UInt8(95); sid_k_ptr[unsafe_offset=ns_len + 1] = UInt8(112); sid_k_ptr[unsafe_offset=ns_len + 2] = UInt8(107)

    var idx_k = vstore.find_and_touch(sid_k_ptr, total)
    if idx_k < 0:
        # Cross-worker hint, same pattern as KV.PREFIX.WARM.
        var owner = -1
        if is_not_null(vstore.directory):
            owner = vstore_dir_lookup(vstore.directory, sid_k_ptr, total)
        sid_k_ptr.unsafe_free()
        if owner >= 0:
            var s = String("ERR KV.PREFIX.BLOCKS session lives on worker ") + String(owner)
            writer.append_error_response(s)
        else:
            writer.append_to_response("+UNKNOWN\r\n".unsafe_ptr(), 10)
        return 1
    sid_k_ptr.unsafe_free()

    var bsz  = vstore.sessions[unsafe_offset=idx_k].block_size
    var bcnt = vstore.sessions[unsafe_offset=idx_k].block_count
    var hp   = vstore.sessions[unsafe_offset=idx_k].block_hashes
    if Int(bsz) == 0 or is_null(hp):
        writer.append_to_response("+UNKNOWN\r\n".unsafe_ptr(), 10)
        return 1

    # Reply is one bulk string: [block_size 4B LE][block_count 4B LE][hashes].
    var blob_bytes = Int(bcnt) * 8
    var total_bytes = 8 + blob_bytes
    var _rbuf = alloc[UInt8](total_bytes)
    var rb = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_rbuf))
    rb.unsafe_bitcast[UInt32]()[unsafe_offset=0]       = bsz
    (rb.unsafe_offset(4)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = bcnt
    if blob_bytes > 0:
        unsafe_memcpy(dest=(rb.unsafe_offset(8)), src=hp.unsafe_bitcast[UInt8](), count=blob_bytes)
    writer.append_bulk_string_response(rb, total_bytes)
    _rbuf.unsafe_free()
    return 1


@always_inline
def handle_kv_prefix_membership(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """KV.PREFIX.MEMBERSHIP <ns_key> <hash_count> <hash_blob>  →  bulk-string bitmap  or  +UNKNOWN

    gh #71: the router supplies the block hashes it's looking for (in token
    order); server returns a ceil(hash_count/8)-byte bitmap, bit i = 1 iff
    probe hash i is cached. Linear probe over the session's registered
    block-hash table; target latency ≤ 100 µs at 1,562 blocks (single worker).
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled (use --kvcache)")
        return 1
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR KV.PREFIX.MEMBERSHIP requires: ns_key hash_count hash_blob")
        return 1

    var ns_tok = tokens[unsafe_offset=start + 1]
    var ns_ptr = ns_tok.ptr
    var ns_len = Int(ns_tok.length)
    if ns_len == 0 or ns_len > 200:
        writer.append_error_response("ERR ns_key length must be 1..200")
        return 1
    if not _ns_prefix_check(ns_ptr, ns_len, vstore):
        writer.append_error_response("ERR namespace must start with required prefix " +
                                     "'" + vstore.ns_prefix + "' (multi-tenant isolation)")
        return 1

    var hc = strict_atol(tokens[unsafe_offset=start + 2].value())
    if hc < 0 or hc > (1 << 20):
        writer.append_error_response("ERR hash_count out of range (0..1048576)")
        return 1
    var blob_tok = tokens[unsafe_offset=start + 3]
    if Int(blob_tok.length) != Int(hc) * 8:
        writer.append_error_response("ERR hash_blob length mismatch")
        return 1
    var probe = blob_tok.ptr.unsafe_bitcast[UInt64]()

    var total = ns_len + 3
    var _buf_k = alloc[UInt8](total)
    var sid_k_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_buf_k))
    unsafe_memcpy(dest=sid_k_ptr, src=ns_ptr, count=ns_len)
    sid_k_ptr[unsafe_offset=ns_len] = UInt8(95); sid_k_ptr[unsafe_offset=ns_len + 1] = UInt8(112); sid_k_ptr[unsafe_offset=ns_len + 2] = UInt8(107)

    var idx_k = vstore.find_and_touch(sid_k_ptr, total)
    if idx_k < 0:
        var owner = -1
        if is_not_null(vstore.directory):
            owner = vstore_dir_lookup(vstore.directory, sid_k_ptr, total)
        sid_k_ptr.unsafe_free()
        if owner >= 0:
            var s = String("ERR KV.PREFIX.MEMBERSHIP session lives on worker ") + String(owner)
            writer.append_error_response(s)
        else:
            writer.append_to_response("+UNKNOWN\r\n".unsafe_ptr(), 10)
        return 1
    sid_k_ptr.unsafe_free()

    var bcnt = Int(vstore.sessions[unsafe_offset=idx_k].block_count)
    var hps  = vstore.sessions[unsafe_offset=idx_k].block_hashes_sorted
    if Int(vstore.sessions[unsafe_offset=idx_k].block_size) == 0 or is_null(hps):
        writer.append_to_response("+UNKNOWN\r\n".unsafe_ptr(), 10)
        return 1

    # Build the bitmap via binary search on the sorted parallel array. At
    # K=N=1562 that's ~17 K compares total vs ~2.4 M for the linear scan that
    # gh #71 originally shipped with (which hit ~6.5 ms p50; we target ≤ 100 µs
    # server-side and ≤ 1 ms e2e over loopback).
    var bitmap_bytes = (Int(hc) + 7) // 8
    var _bm = alloc[UInt8](bitmap_bytes if bitmap_bytes > 0 else 1)
    var bm = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_bm))
    for bi in range(bitmap_bytes):
        bm[unsafe_offset=bi] = UInt8(0)
    for i in range(Int(hc)):
        var needle = probe[unsafe_offset=i]
        # Binary search in [lo, hi).
        var lo = 0
        var hi = bcnt
        var found = False
        while lo < hi:
            var mid = (lo + hi) >> 1
            var v = hps[unsafe_offset=mid]
            if v == needle:
                found = True
                break
            elif v < needle:
                lo = mid + 1
            else:
                hi = mid
        if found:
            bm[unsafe_offset=i >> 3] = bm[unsafe_offset=i >> 3] | UInt8(1 << (i & 7))
    writer.append_bulk_string_response(bm, bitmap_bytes)
    _bm.unsafe_free()
    return 1


@always_inline
def handle_kv_prefix_commit(
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
    wal: Pointer[WAL, MutUntrackedOrigin],
    wal_writes_off: Bool,
) raises -> Int:
    """KV.PREFIX.COMMIT  →  +OK / -ERR

    Replies once every write acknowledged before it — keyspace and V-store
    alike — is on stable storage, power cut and kernel panic included: the
    keyspace WAL is msync(MS_SYNC)ed and both logs are flushed through the
    drive's cache (F_FULLFSYNC on macOS). Without it Pion's logs survive a
    process dying, not the machine: the keyspace WAL is flushed MS_ASYNC once
    per 64 ticks and the V-store WAL is plain write().

    It blocks this worker's event loop for the flush — ~4 ms plus the bytes
    written since the last one on an M4 mini's SSD — so call it at the pace
    durability is needed (serve --durable: every few dozen generated
    tokens), never per write. An error means at least one log could not be
    flushed, or the WAL is disabled (--no-wal), and nothing is promised."""
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled")
        return 1
    if wal_writes_off:
        writer.append_error_response("ERR KV.PREFIX.COMMIT: --no-wal, writes are not logged")
        return 1
    var ok = is_not_null(wal) and wal[].barrier()
    ok = vstore.wal_barrier() and ok
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR KV.PREFIX.COMMIT: a log could not be flushed (or --no-wal)")
    return 1


def handle_kv_prefix_save(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    worker_id: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """KV.PREFIX.SAVE [path]  →  +OK / -ERR

    Snapshot every active V-store session to disk. Used by callers that want
    the cache to survive a restart (e.g. before a planned redeploy). Pion does
    not auto-save on shutdown — call this explicitly.

    Default path: pion.vstore.<worker_id>. The startup auto-load reads this
    same path. Override the path argument to write to a different location
    (useful for offline migration).
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled (use --kvcache)")
        return 1

    var path = String("pion.vstore.") + String(worker_id)
    if start + 1 < num_tokens:
        var path_tok = tokens[unsafe_offset=start + 1]
        if Int(path_tok.length) > 0:
            # SECURITY: this path was used verbatim, so any client that could
            # reach the port could write (and truncate) a file ANYWHERE the
            # server process has permission —
            #   V.CREATE sess 16 VQUANT fp16
            #   KV.PREFIX.SAVE /absolute/path        -> +OK, file written
            #   KV.PREFIX.SAVE ../../../../tmp/x     -> +OK, traversal works
            # No AUTH is required unless --requirepass is set, and the snapshot
            # body embeds caller-chosen session ids, so the attacker also
            # controls some of the bytes landing in the target file. That is
            # the Redis `CONFIG SET dir` + `dbfilename` escalation shape, in a
            # single command with no config change.
            #
            # Confine to the working directory subtree: the documented use is
            # "write to a different location for offline migration", and an
            # operator doing that has shell access to move the file anyway.
            var cand = path_tok.value()
            if not _safe_relative_path(cand):
                writer.append_error_response(
                    "ERR KV.PREFIX.SAVE: path must be relative to the data"
                    + " directory (no leading '/', no '..', no control bytes)")
                return 1
            path = cand

    var ok = vstore.save_to_disk(path)
    # gh #94: SSM.PREFIX.* substrate shares the paired-prefix story (hybrid
    # Mamba-in-Llama / LoLCATs) — snapshot it on the same SAVE so both sides
    # of the prefix cache stay durable together. Best-effort: returns ≥0 on
    # success (slot count) or -1 if no SSM state exists / IO error. We don't
    # gate the +OK on this — the V-store side is the contract caller asked
    # for; SSM is the paired follow-on.
    _ = ssm_save_snapshot(worker_id)
    if ok:
        # Snapshot is durable on disk → WAL is now redundant; truncate it so
        # the next restart only replays records written after this snapshot.
        vstore.wal_truncate()
        writer.append_ok_response()
    else:
        # Either V-store is empty or save_to_disk failed; either way this
        # is benign for callers — they can probe with KV.PREFIX.INFO first.
        writer.append_error_response("ERR KV.PREFIX.SAVE failed (no active sessions or io error)")
    return 1


@always_inline
def handle_kv_prefix_info(
    mut writer: ResponseWriter,
    vstore: VStoreIndex,
    mut metal_engine: MetalAttentionEngine,
) raises -> Int:
    """KV.PREFIX.INFO  →  bulk string with global stats.

    gh #67: includes cold-tier telemetry from the per-worker Metal session
    cache — number of WARM slots, COLD registry entries, lifetime
    WARM→COLD demotions, and lifetime COLD→WARM rehydrates."""
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled")
        return 1

    var prefix_count = 0
    var total_tokens = 0
    for i in range(MAX_VS_SESSIONS):
        if vstore.sessions[unsafe_offset=i].active:
            var sl = vstore.sessions[unsafe_offset=i].session_id_len
            var sp = vstore.sessions[unsafe_offset=i].session_id_ptr
            # sid ends with "_pk" → it's a registered prefix-key session
            if sl >= 3 and Int(sp[unsafe_offset=sl - 3]) == 95 and Int(sp[unsafe_offset=sl - 2]) == 112 and Int(sp[unsafe_offset=sl - 1]) == 107:
                prefix_count += 1
                for li in range(MAX_VS_LAYERS):
                    total_tokens += vstore.tokens_per_layer[unsafe_offset=i * MAX_VS_LAYERS + li]

    var info = String("")
    info += "registered_prefixes:" + String(prefix_count) + "\r\n"
    info += "total_prefix_tokens:" + String(total_tokens) + "\r\n"
    info += "vstore_sessions:" + String(vstore.session_count) + "\r\n"
    info += "vstore_total_fetches:" + String(vstore.total_fetches) + "\r\n"
    info += "vstore_evictions:" + String(vstore.total_evictions) + "\r\n"
    info += "vstore_max_sessions:" + String(MAX_VS_SESSIONS) + "\r\n"
    # Bytes of K/V rows held (every session, every layer, in its stored
    # format) — what a byte budget is enforced against, and what a snapshot
    # writes. Buffers grow by doubling, so resident memory can be up to ~2x.
    var held = 0
    for si in range(MAX_VS_SESSIONS):
        if vstore.sessions[unsafe_offset=si].active:
            for li in range(vstore.sessions[unsafe_offset=si].num_layers):
                var ls = si * MAX_VS_LAYERS + li
                held += vstore.tokens_per_layer[unsafe_offset=ls] * vstore._bytes_per_token_for_slot(ls)
    info += "vstore_bytes:" + String(held) + "\r\n"
    info += "wal_enabled:" + ("1" if vstore.wal_fd >= 0 else "0") + "\r\n"
    # On-disk size of the V-store WAL: it only grows until KV.PREFIX.SAVE
    # snapshots and truncates it (the fd is O_APPEND, so seeking is harmless).
    var wal_bytes = 0
    if vstore.wal_fd >= 0:
        wal_bytes = external_call["lseek", Int](vstore.wal_fd, Int(0), Int32(2))
    info += "wal_bytes:" + String(wal_bytes) + "\r\n"
    info += "wal_appended:" + String(vstore.wal_appended) + "\r\n"
    info += "wal_replayed:" + String(vstore.wal_replayed) + "\r\n"
    # Cross-worker directory: count published entries so callers can see
    # how many prefixes are visible across all workers (vs. just this one).
    info += "directory_enabled:" + ("1" if is_not_null(vstore.directory) else "0") + "\r\n"
    info += "my_worker_id:" + String(vstore.my_worker_id) + "\r\n"
    var dir_active = 0
    if is_not_null(vstore.directory):
        for di in range(MAX_DIR_ENTRIES):
            if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                    vstore.directory[].published.unsafe_offset(di), UInt64(0)) != 0:
                dir_active += 1
    info += "directory_entries:" + String(dir_active) + "\r\n"

    # gh #67: cold-tier telemetry (this worker's Metal SDPA session cache).
    var stats = metal_engine.cold_stats()
    info += "sdpa_warm:"         + String(Int(stats[0])) + "\r\n"
    info += "sdpa_cold:"         + String(Int(stats[1])) + "\r\n"
    info += "sdpa_demotions:"    + String(Int(stats[2])) + "\r\n"
    info += "sdpa_rehydrates:"   + String(Int(stats[3])) + "\r\n"

    var info_bytes = info.as_bytes()
    var info_ptr = info_bytes.unsafe_ptr()
    var info_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(info_ptr))
    writer.append_bulk_string_response(info_ext, info.byte_length())
    return 1


@always_inline
def handle_kv_prefix_warm(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
    mut metal_engine: MetalAttentionEngine,
) raises -> Int:
    """KV.PREFIX.WARM <ns_key> <H> <D> [<attend_sid>]  →  +<rehydrated_layers> / -ERR

    gh #67 cold-tier rehydrate. Walks the V-store K/V sessions for the
    namespace, dequantizes per-layer to fp32, transposes from V-store layout
    [N, H*D] into ATTEND.PREFIX.STORE layout [H, N, D], and pushes each layer
    back into the Metal SDPA session cache. Clears the cold-registry entry
    for each (sid, layer) as a side effect of `store_kv`.

    Use after ATTEND.PREFIX.QUERY returns `-COLDMISS` (or proactively before
    a batch of queries against a namespace that may have been evicted).
    Returns the number of layers successfully rehydrated as `+N` so callers
    can sanity-check (0 = nothing to do, ns was never registered or no V-store
    data; large N = full rehydrate).

    Wire form: `<ns_key>` is the bare namespace (same as KV.PREFIX.REGISTER);
    `<H>` is the kv-head count; `<D>` is the head dim. V-store sessions are
    looked up at `<ns>_pk` / `<ns>_pv`. Layers with `tokens_per_layer == 0`
    are skipped — the loop iterates 0..MAX_VS_LAYERS-1 since the consumer
    might have stored non-contiguous layer ids (rare; the chunked-prefill
    path stores 0..n_layers-1 dense).

    Optional `<attend_sid>` overrides the Metal SDPA session id that gets
    rehydrated; default = `<ns_key>`. The production consumer
    (pion-vllm-mlx prompt_cache.py) stores ATTEND state under
    `<namespace>_attn` to avoid sid collision with V-store, so it passes
    `<ns>_attn` here. Substrate tests use the bare sid.
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled (use --kvcache)")
        return 1
    if not metal_engine.available:
        writer.append_error_response("ERR Metal attention not available (start with --metal-attention)")
        return 1
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR KV.PREFIX.WARM requires: ns_key H D")
        return 1

    var ns_tok = tokens[unsafe_offset=start + 1]
    var ns_ptr = ns_tok.ptr
    var ns_len = Int(ns_tok.length)
    if ns_len == 0 or ns_len > 200:
        writer.append_error_response("ERR ns_key length must be 1..200")
        return 1
    if not _ns_prefix_check(ns_ptr, ns_len, vstore):
        writer.append_error_response("ERR namespace must start with required prefix " +
                                     "'" + vstore.ns_prefix + "' (multi-tenant isolation)")
        return 1

    var H = strict_atol(tokens[unsafe_offset=start + 2].value())
    var D = strict_atol(tokens[unsafe_offset=start + 3].value())
    if H <= 0 or D <= 0 or H > 256 or D > 512:
        writer.append_error_response("ERR invalid H/D (1..256 / 1..512)")
        return 1

    # Build derived sids: "<ns>_pk" / "<ns>_pv".
    var total = ns_len + 3
    var _bk = alloc[UInt8](total)
    var sid_k_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_bk))
    unsafe_memcpy(dest=sid_k_ptr, src=ns_ptr, count=ns_len)
    sid_k_ptr[unsafe_offset=ns_len] = UInt8(95); sid_k_ptr[unsafe_offset=ns_len + 1] = UInt8(112); sid_k_ptr[unsafe_offset=ns_len + 2] = UInt8(107)
    var _bv = alloc[UInt8](total)
    var sid_v_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_bv))
    unsafe_memcpy(dest=sid_v_ptr, src=ns_ptr, count=ns_len)
    sid_v_ptr[unsafe_offset=ns_len] = UInt8(95); sid_v_ptr[unsafe_offset=ns_len + 1] = UInt8(112); sid_v_ptr[unsafe_offset=ns_len + 2] = UInt8(118)

    var idx_k = vstore.find_and_touch(sid_k_ptr, total)
    var idx_v = vstore.find_and_touch(sid_v_ptr, total)
    if idx_k < 0 or idx_v < 0:
        # Cross-worker hint: the V-store directory may say another worker owns
        # this namespace. Mirror the V.* pattern and tell the caller where to
        # pin so they can retry on the right connection.
        var owner = -1
        if is_not_null(vstore.directory):
            owner = vstore_dir_lookup(vstore.directory, sid_k_ptr, total)
        sid_k_ptr.unsafe_free()
        sid_v_ptr.unsafe_free()
        if owner >= 0:
            var s = String("ERR KV.PREFIX.WARM session lives on worker ") + String(owner)
            writer.append_error_response(s)
        else:
            writer.append_error_response("ERR KV.PREFIX.WARM namespace not registered (call KV.PREFIX.REGISTER first)")
        return 1
    sid_k_ptr.unsafe_free()
    sid_v_ptr.unsafe_free()

    # Dimensions: V-store records value_dim per layer (or session-level). We
    # validate against the supplied H*D before allocating buffers — a mismatch
    # means the consumer's tokenizer / model shape diverges from what was
    # stored, and the rehydrate would garble layouts.
    var expected_kv_dim = Int(H) * Int(D)

    # ATTEND.PREFIX.STORE expects an explicit session id; default to the bare
    # namespace, optional 4th arg overrides (production passes `<ns>_attn`).
    var sid_ext: Pointer[UInt8, MutUntrackedOrigin]
    var attend_sid_len: Int
    if start + 4 < num_tokens:
        var asid_tok = tokens[unsafe_offset=start + 4]
        sid_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(asid_tok.ptr))
        attend_sid_len = Int(asid_tok.length)
        if attend_sid_len <= 0:
            sid_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ns_ptr))
            attend_sid_len = ns_len
    else:
        sid_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ns_ptr))
        attend_sid_len = ns_len

    var rehydrated = 0
    var skipped_dim = 0

    # Per-layer rehydrate. tokens_per_layer is a flat array indexed by
    # session_idx * MAX_VS_LAYERS + layer_id.
    for li in range(MAX_VS_LAYERS):
        var n_k = vstore.tokens_per_layer[unsafe_offset=idx_k * MAX_VS_LAYERS + li]
        var n_v = vstore.tokens_per_layer[unsafe_offset=idx_v * MAX_VS_LAYERS + li]
        if n_k == 0 or n_v == 0:
            continue
        # Layer length divergence between K and V — should never happen with
        # the canonical _storebatch path, but guard anyway.
        var N = n_k if n_k < n_v else n_v

        # Validate per-layer value_dim against caller-supplied H*D. A 0 here
        # means the snapshot pre-dates the per-layer field — fall back to
        # session-level value_dim (legacy compat).
        var ld_k = vstore.layer_value_dim[unsafe_offset=idx_k * MAX_VS_LAYERS + li]
        if ld_k <= 0:
            ld_k = vstore.sessions[unsafe_offset=idx_k].value_dim
        var ld_v = vstore.layer_value_dim[unsafe_offset=idx_v * MAX_VS_LAYERS + li]
        if ld_v <= 0:
            ld_v = vstore.sessions[unsafe_offset=idx_v].value_dim
        if ld_k != expected_kv_dim or ld_v != expected_kv_dim:
            skipped_dim += 1
            continue

        var floats = Int(N) * Int(H) * Int(D)
        # Allocate per-layer staging: V-store flat fetch buffer + transposed
        # output. Free at the end of each layer; this keeps peak memory at
        # 4 × floats × 4 bytes ≈ 4 MB at H=8 N=4096 D=128.
        var _ids = alloc[Int32](Int(N))
        var ids_ptr = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(_ids))
        for ti in range(N):
            ids_ptr[unsafe_offset=ti] = Int32(ti)

        var _kf = alloc[Float32](floats)
        var k_flat_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_kf))
        var _vf = alloc[Float32](floats)
        var v_flat_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_vf))

        var got_k = vstore.fetch_tokens(idx_k, li, ids_ptr, Int(N), k_flat_ptr)
        var got_v = vstore.fetch_tokens(idx_v, li, ids_ptr, Int(N), v_flat_ptr)

        ids_ptr.unsafe_free()

        if got_k != Int(N) or got_v != Int(N):
            # Partial fetch — skip this layer rather than push garbage to Metal.
            k_flat_ptr.unsafe_free()
            v_flat_ptr.unsafe_free()
            continue

        # Transpose [N, H*D] → [H, N, D]:
        #   src[n, h, d] = src_flat[n * (H*D) + h * D + d]
        #   dst[h, n, d] = dst_flat[h * (N*D) + n * D + d]
        var _kt = alloc[Float32](floats)
        var k_tp_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_kt))
        var _vt = alloc[Float32](floats)
        var v_tp_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_vt))
        var HD = Int(H) * Int(D)
        var ND = Int(N) * Int(D)
        for h in range(Int(H)):
            var dst_h_k = k_tp_ptr.unsafe_offset(h * ND)
            var dst_h_v = v_tp_ptr.unsafe_offset(h * ND)
            for n in range(Int(N)):
                var src_row_k = k_flat_ptr.unsafe_offset(n * HD).unsafe_offset(h * Int(D))
                var src_row_v = v_flat_ptr.unsafe_offset(n * HD).unsafe_offset(h * Int(D))
                var dst_row_k = dst_h_k.unsafe_offset(n * Int(D))
                var dst_row_v = dst_h_v.unsafe_offset(n * Int(D))
                unsafe_memcpy(dest=dst_row_k.unsafe_bitcast[UInt8](), src=src_row_k.unsafe_bitcast[UInt8](), count=Int(D) * 4)
                unsafe_memcpy(dest=dst_row_v.unsafe_bitcast[UInt8](), src=src_row_v.unsafe_bitcast[UInt8](), count=Int(D) * 4)

        k_flat_ptr.unsafe_free()
        v_flat_ptr.unsafe_free()

        var ok = metal_engine.store_kv(sid_ext, attend_sid_len, li, Int(H), Int(N), Int(D), k_tp_ptr, v_tp_ptr)
        k_tp_ptr.unsafe_free()
        v_tp_ptr.unsafe_free()
        if ok:
            rehydrated += 1

    # Reply: bulk-style `+N` so the caller can parse a single int (compatible
    # with `KV.PREFIX.OWNER`'s simple-string-int reply convention).
    var rs: String
    if skipped_dim > 0:
        rs = String("+") + String(rehydrated) + String(" skipped_dim=") + String(skipped_dim) + String("\r\n")
    else:
        rs = String("+") + String(rehydrated) + String("\r\n")
    writer.append_to_response(rs.unsafe_ptr(), rs.byte_length())
    return 1
