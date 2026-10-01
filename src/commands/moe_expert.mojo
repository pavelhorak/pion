"""MOE.EXPERT.* command handlers — Stage 1 wire surface + Stage 2 skeleton.

gh #61 Phase-0 → Stage 1 (commit `cbe4730`): RESP commands registered.
gh #61 Phase-0 → Stage 2 (this commit + follow-ons): MoEExpertTier struct
wired through; STATS reads real (zero-initialized) counters. FETCH / INFO
still return UNAVAILABLE until the disk-backed manifest loader lands.

Wire surface:

    MOE.EXPERT.FETCH <model_id> <layer_id> <expert_id>
      → -UNAVAILABLE until tier is loaded
      → Bulk-string with packed blobs when tier is loaded (Stage 2+)

    MOE.EXPERT.INFO <model_id>
      → -UNAVAILABLE if model not loaded
      → JSON manifest when found (Stage 2)

    MOE.EXPERT.STATS
      → JSON tier-wide counters (always available)

    MOE.EXPERT.PREFETCH <model_id> <layer_id> <expert_id>...
    MOE.EXPERT.PIN / UNPIN <model_id> <layer_id> <expert_id>
      → +OK no-ops in Stage 1; real semantics in Stage 2.

Binary protocol (CMD_MOE_EXPERT_* in src/network/binary_protocol.mojo at
0x31-0x36) is registered in parallel and dispatched from slow_path.mojo's
binary handler.
"""
from src.common.utils import strict_atol

from src.network.response_writer import ResponseWriter
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.moe_expert_tier import MoEExpertTier, MAX_MOE_NS, MOE_NS_NAME_INLINE

from std.collections import Array
from std.ffi import external_call
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc


@always_inline
def handle_moe_expert_fetch(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut tier: MoEExpertTier,
    client_fd: Int32,
) raises -> Int:
    """MOE.EXPERT.FETCH <model_id> <layer_id> <expert_id>

    Stage 2j: returns the raw bytes of gate_proj.weight for the requested
    (model, layer, expert) as a single bulk string. Multi-blob response
    (all 3 projections + optional scales/biases) per WIRE_FORMAT_DESIGN.md
    is the next refinement; this first cut proves the offset-table + pread
    path end-to-end.
    """
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR MOE.EXPERT.FETCH requires: model_id layer_id expert_id")
        return 1
    if not tier.enabled:
        writer.append_error_response("UNAVAILABLE MOE.EXPERT.* tier not loaded (start server with --moe-cache to enable)")
        return 1

    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = tier.find_model(name_ptr, Int(name_tok.length))
    if slot < 0:
        writer.append_error_response("UNAVAILABLE MOE.EXPERT.* model not loaded")
        return 1

    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    var expert_id = strict_atol(tokens[unsafe_offset=start + 3].value())
    if layer_id < 0 or layer_id >= Int(tier.models[slot].num_layers):
        writer.append_error_response("ERR layer_id out of range")
        return 1
    if expert_id < 0 or expert_id >= Int(tier.models[slot].num_experts):
        writer.append_error_response("ERR expert_id out of range")
        return 1

    # Stage 3: pruned experts are rejected. Consumer-side substrate is expected
    # to skip the FETCH and route around it (e.g. zero output, fall through to
    # next-best expert via routing-decision rewrite). -PRUNED is a distinct
    # status from -UNAVAILABLE so the consumer can disambiguate.
    if tier.is_pruned(slot, layer_id, expert_id):
        writer.append_error_response("PRUNED layer=" + String(layer_id) + " expert=" + String(expert_id))
        return 1

    # Per-(layer, expert) usage telemetry — bumped on every FETCH (hit OR miss)
    # so workload-specific routing concentration is visible via MOE.EXPERT.HIST.
    # Optional `NS <name>` suffix banks the count under a per-distribution
    # namespace (created on first use); omitting it uses ns 0 = "default".
    var ns = 0
    if start + 5 < num_tokens:
        var ns_kw = tokens[unsafe_offset=start + 4]
        if Int(ns_kw.length) == 2 and (ns_kw.ptr[unsafe_offset=0] | 0x20) == 110 and (ns_kw.ptr[unsafe_offset=1] | 0x20) == 115:  # "ns"
            var ns_name_tok = tokens[unsafe_offset=start + 5]
            var ns_name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ns_name_tok.ptr))
            var resolved = tier.resolve_ns(slot, ns_name_ptr, Int(ns_name_tok.length), True)
            if resolved < 0:
                writer.append_error_response("ERR MOE.EXPERT.* namespaces full (max " + String(MAX_MOE_NS) + " per model)")
                return 1
            ns = resolved
    tier.bump_access(slot, ns, layer_id, expert_id)

    # Stage 2k: cache hot-path. If the assembled multi-blob payload is
    # already cached, skip the SSD reads entirely.
    var hit_addr_buf = alloc[UInt64](1)
    var hit_len_buf = alloc[Int](1)
    var is_hit = tier.cache_lookup(slot, layer_id, expert_id, hit_addr_buf, hit_len_buf)
    if is_hit:
        var hit_addr = hit_addr_buf[]
        var hit_len = hit_len_buf[]
        hit_addr_buf.unsafe_free(); hit_len_buf.unsafe_free()
        var hit_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(hit_addr))
        writer.append_bulk_bytes_writev(client_fd, hit_ptr, hit_len)
        return 1
    hit_addr_buf.unsafe_free(); hit_len_buf.unsafe_free()

    # Multi-blob FETCH: iterate all (proj, comp) combinations in the offset
    # table for this (model, layer) and bundle into one bulk-string response.
    # Body format (matches the simplified WIRE_FORMAT_DESIGN.md subset):
    #   [u8 n_blobs] per blob: [u8 proj][u8 comp][u32 LE data_len][data]
    #
    # The offset lookup branches on layout. For stacked layouts (Gemma 4,
    # Phi-3.5) one entry covers all experts of (layer, proj, comp), and the
    # expert is addressed by `data_start + expert * per_expert_bytes`. For
    # per-expert layouts (OLMoE), each (layer, expert, proj, comp) tuple has
    # its own entry: `data_start` is already the absolute byte offset and
    # `per_expert_bytes` is the tensor size.
    var per_expert_layout = tier.uses_per_expert_offsets(slot)

    # First pass: count valid entries + compute total bytes to size the buffer.
    var n_blobs = 0
    var total_bytes = 0
    for p in range(3):     # MOE_PROJ_COUNT
        for c in range(3):   # MOE_COMP_COUNT
            var e = (tier.get_per_expert_offset(slot, layer_id, expert_id, p, c)
                     if per_expert_layout
                     else tier.get_offset(slot, layer_id, p, c))
            if e.shard_id > 0:
                n_blobs += 1
                total_bytes += 6 + Int(e.per_expert_bytes)
    total_bytes += 1   # for the n_blobs prefix byte

    if n_blobs == 0:
        writer.append_error_response("UNAVAILABLE offset table empty for (layer=" + String(layer_id) + ")")
        return 1

    # Single output buffer; writev'd at the end.
    var out_buf = alloc[UInt8](total_bytes)
    out_buf[unsafe_offset=0] = UInt8(n_blobs)
    var write_off = 1

    # Open each shard once per FETCH; reuse fd across (proj, comp) tuples in
    # the same shard. Most stacked-tensor MoEs spread layer-X's projections
    # across the same shard, so a single open suffices for typical cases.
    # Single-shard OLMoE keeps fd open across all 9 blobs.
    var last_shard_id = Int32(-1)
    var fd = Int32(-1)
    for p in range(3):
        for c in range(3):
            var e = (tier.get_per_expert_offset(slot, layer_id, expert_id, p, c)
                     if per_expert_layout
                     else tier.get_offset(slot, layer_id, p, c))
            if e.shard_id <= 0:
                continue
            if e.shard_id != last_shard_id:
                if fd >= 0:
                    _ = external_call["close", Int32](fd)
                var shard_path = tier.get_shard_path(slot, Int(e.shard_id))
                var cpath = shard_path
                fd = external_call["pion_open_rdonly", Int32](cpath.as_c_string_slice())
                last_shard_id = e.shard_id
                if fd < 0:
                    out_buf.unsafe_free()
                    writer.append_error_response("ERR shard open failed at id=" + String(Int(e.shard_id)))
                    return 1
            var per_e = Int(e.per_expert_bytes)
            # Per-expert layout: data_start is THIS expert's absolute offset.
            # Stacked layout: stride by expert_id × per_expert_bytes.
            var target = (Int(e.data_start) if per_expert_layout
                          else Int(e.data_start + UInt64(expert_id) * e.per_expert_bytes))
            # Write the per-blob header (proj, comp, data_len)
            out_buf[unsafe_offset=write_off] = UInt8(p)
            out_buf[unsafe_offset=write_off + 1] = UInt8(c)
            # Little-endian u32 data_len
            out_buf[unsafe_offset=write_off + 2] = UInt8(per_e & 0xFF)
            out_buf[unsafe_offset=write_off + 3] = UInt8((per_e >> 8) & 0xFF)
            out_buf[unsafe_offset=write_off + 4] = UInt8((per_e >> 16) & 0xFF)
            out_buf[unsafe_offset=write_off + 5] = UInt8((per_e >> 24) & 0xFF)
            write_off += 6
            var got = external_call["pion_pread", Int](fd, out_buf.unsafe_offset(write_off), per_e, target)
            if got != per_e:
                if fd >= 0: _ = external_call["close", Int32](fd)
                out_buf.unsafe_free()
                writer.append_error_response("ERR pread short read")
                return 1
            write_off += per_e
    if fd >= 0:
        _ = external_call["close", Int32](fd)

    # Send the assembled multi-blob payload as a single bulk-string via writev.
    writer.append_bulk_bytes_writev(client_fd, out_buf, total_bytes)
    # Stage 2k: hand ownership of out_buf to the LRU cache. Cache will free
    # on eviction. Don't out_buf.unsafe_free() here.
    tier.cache_store(slot, layer_id, expert_id, UInt64(Int(out_buf)), total_bytes)
    tier.misses += UInt64(1)
    return 1


@always_inline
def handle_moe_expert_prefetch(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut tier: MoEExpertTier,
) raises -> Int:
    """MOE.EXPERT.PREFETCH <model_id> <layer> <expert_id>...

    Stage 4 async I/O: for each requested (layer, expert) tuple, look up the
    on-disk byte ranges in the offset table and issue an OS-level read-ahead
    hint (posix_fadvise WILLNEED on Linux, F_RDADVISE on macOS). The kernel
    performs the actual read asynchronously into the page cache; the worker
    returns +OK immediately.

    Pruned experts and (model, layer, expert) tuples already in the LRU
    cache are skipped — no point hinting bytes the cache already owns. Tuples
    outside the offset table are silently skipped as well.

    Cumulative effect over a consumer's pre-routing decision: the next FETCH
    on the same (model, layer, expert) hits page cache → tracks the warm-fetch
    cost (≈12 ms on the Mac M4 reference machine) instead of the cold-fetch
    cost (≈21 ms).
    """
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR MOE.EXPERT.PREFETCH requires: model_id layer_id expert_id...")
        return 1
    if not tier.enabled:
        # Match the prior contract: PREFETCH is fire-and-forget; without a
        # tier loaded it's just a no-op +OK.
        writer.append_ok_response()
        return 1

    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = tier.find_model(name_ptr, Int(name_tok.length))
    if slot < 0:
        writer.append_ok_response()   # unknown model — silent +OK
        return 1

    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    if layer_id < 0 or layer_id >= Int(tier.models[slot].num_layers):
        writer.append_ok_response()
        return 1

    # Stage 4b-3: enqueue each expert on the warming thread. The C worker
    # opens the shard(s), preads the multi-blob payload, and pushes a
    # completion onto the SPSC ring. The engine event loop drains the ring
    # once per tick (moe_tier.drain_warm_into_cache), inserting each warmed
    # buffer into the LRU. The next FETCH on that (layer, expert) hits the
    # cache hot path (~0.5 ms) instead of synchronously preading (~21 ms).
    #
    # Fallback: if the warming thread's request ring is full (returns -1),
    # fall back to the Stage 4a fadvise kernel hint so we still get the
    # page-cache benefit.
    var hit_addr_buf = alloc[UInt64](1)
    var hit_len_buf = alloc[Int](1)
    var n_enqueued = 0
    var n_fallback_fadvise = 0
    var last_shard_id = Int32(-1)
    var fd = Int32(-1)
    var per_expert_layout = tier.uses_per_expert_offsets(slot)
    for k in range(start + 3, num_tokens):
        var expert_id = strict_atol(tokens[unsafe_offset=k].value())
        if expert_id < 0 or expert_id >= Int(tier.models[slot].num_experts):
            continue
        if tier.is_pruned(slot, layer_id, expert_id):
            continue
        # Skip cache hits — bytes are already in RAM
        var is_hit = tier.cache_lookup(slot, layer_id, expert_id, hit_addr_buf, hit_len_buf)
        if is_hit:
            continue
        var rc = tier.enqueue_warm(slot, layer_id, expert_id)
        if rc == 0:
            n_enqueued += 1
            continue
        # Fallback: warming thread queue full / unsupported layout —
        # kernel readahead hint over the per-expert byte ranges.
        for p in range(3):
            for c in range(3):
                var e = (tier.get_per_expert_offset(slot, layer_id, expert_id, p, c)
                         if per_expert_layout
                         else tier.get_offset(slot, layer_id, p, c))
                if e.shard_id <= 0:
                    continue
                if e.shard_id != last_shard_id:
                    if fd >= 0:
                        _ = external_call["close", Int32](fd)
                    var shard_path = tier.get_shard_path(slot, Int(e.shard_id))
                    var cpath = shard_path
                    fd = external_call["pion_open_rdonly", Int32](cpath.as_c_string_slice())
                    last_shard_id = e.shard_id
                    if fd < 0:
                        last_shard_id = Int32(-1)
                        continue
                var per_e = Int(e.per_expert_bytes)
                var target = (Int(e.data_start) if per_expert_layout
                              else Int(e.data_start + UInt64(expert_id) * e.per_expert_bytes))
                _ = external_call["pion_fadvise_willneed", Int32](fd, Int64(target), per_e)
                n_fallback_fadvise += 1
    if fd >= 0:
        _ = external_call["close", Int32](fd)
    hit_addr_buf.unsafe_free(); hit_len_buf.unsafe_free()

    writer.append_ok_response()
    return 1


@always_inline
def handle_moe_expert_pin(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut tier: MoEExpertTier,
) raises -> Int:
    """MOE.EXPERT.PIN — protect cache entry from LRU eviction."""
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR MOE.EXPERT.PIN requires: model_id layer_id expert_id")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = tier.find_model(name_ptr, Int(name_tok.length))
    if slot >= 0:
        var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
        var expert_id = strict_atol(tokens[unsafe_offset=start + 3].value())
        _ = tier.cache_pin(slot, layer_id, expert_id)
    writer.append_ok_response()
    return 1


@always_inline
def handle_moe_expert_unpin(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut tier: MoEExpertTier,
) raises -> Int:
    """MOE.EXPERT.UNPIN — release pin."""
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR MOE.EXPERT.UNPIN requires: model_id layer_id expert_id")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = tier.find_model(name_ptr, Int(name_tok.length))
    if slot >= 0:
        var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
        var expert_id = strict_atol(tokens[unsafe_offset=start + 3].value())
        _ = tier.cache_unpin(slot, layer_id, expert_id)
    writer.append_ok_response()
    return 1


@always_inline
def handle_moe_expert_load(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut tier: MoEExpertTier,
) raises -> Int:
    """MOE.EXPERT.LOAD <path> — load an additional model into the next free
    MoE slot at runtime.

    The substrate supports MAX_MOE_MODELS=4 concurrent registrations per
    worker. Pion starts with at most one model loaded via the --moe-cache
    CLI flag; this command lets operators register additional models on
    a running server (e.g. Gemma 4 26B + Phi-3.5-MoE coexisting on one
    server for memory-tier sharing).

    Reply: +OK on success, -ERR with reason on failure (no free slot,
    invalid manifest, etc.). The model_id is the basename of the path.
    """
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR MOE.EXPERT.LOAD requires: model_dir_path")
        return 1
    if not tier.enabled:
        writer.append_error_response("UNAVAILABLE MOE.EXPERT.* tier not enabled (start server with --moe-cache to enable)")
        return 1
    var path = tokens[unsafe_offset=start + 1].value()
    var ok = tier.load_manifest(path)
    if not ok:
        writer.append_error_response("ERR MOE.EXPERT.LOAD: manifest load failed (see server log; common causes: bad path, no free slot, config.json missing)")
        return 1
    writer.append_ok_response()
    return 1


@always_inline
def handle_moe_expert_prune(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut tier: MoEExpertTier,
) raises -> Int:
    """MOE.EXPERT.PRUNE <model_id> <layer> <expert> [<on>]

    Mark (or unmark) a (model, layer, expert) as pruned. Pruned experts return
    -PRUNED on FETCH. Default `on=1` (prune); pass `0` to un-prune.

    The Stage-3 pruning workflow:
      1. Run inference workload, accumulate per-expert FETCH counts via HIST.
      2. Offline tool (pion-moe-prune) identifies low-usage experts.
      3. Call MOE.EXPERT.PRUNE for each low-usage (layer, expert).
      4. Consumer-side substrate re-routes around -PRUNED experts.

    Reply: integer count of currently-pruned entries for this model.
    """
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR MOE.EXPERT.PRUNE requires: model_id layer expert [on]")
        return 1
    if not tier.enabled:
        writer.append_error_response("UNAVAILABLE MOE.EXPERT.* tier not loaded")
        return 1

    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = tier.find_model(name_ptr, Int(name_tok.length))
    if slot < 0:
        writer.append_error_response("UNAVAILABLE MOE.EXPERT.* model not loaded")
        return 1

    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    var expert_id = strict_atol(tokens[unsafe_offset=start + 3].value())
    if layer_id < 0 or layer_id >= Int(tier.models[slot].num_layers):
        writer.append_error_response("ERR layer_id out of range")
        return 1
    if expert_id < 0 or expert_id >= Int(tier.models[slot].num_experts):
        writer.append_error_response("ERR expert_id out of range")
        return 1

    # Optional on/off flag. Default = prune.
    var on = True
    if start + 4 < num_tokens:
        var flag_str = tokens[unsafe_offset=start + 4].value()
        on = atol(flag_str) != 0

    _ = tier.set_pruned(slot, layer_id, expert_id, on)
    # Also evict any cached payload so the next FETCH-after-PRUNE doesn't
    # accidentally serve stale bytes from the LRU. (On un-prune we just leave
    # the cache as-is; next FETCH will refill if missing.)
    if on:
        _ = tier.cache_evict(slot, layer_id, expert_id)
    writer.append_int_response(Int64(Int(tier.pruned_count)))
    return 1


@always_inline
def handle_moe_expert_info(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut tier: MoEExpertTier,
) raises -> Int:
    """MOE.EXPERT.INFO <model_id> — reports manifest if model is loaded.

    Stage 2: if the tier has a model registered with this id, returns a JSON
    manifest. Otherwise -UNAVAILABLE.
    """
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR MOE.EXPERT.INFO requires: model_id")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = tier.find_model(name_ptr, Int(name_tok.length))
    if slot < 0:
        writer.append_error_response("UNAVAILABLE MOE.EXPERT.* model not loaded")
        return 1
    # Found — emit JSON manifest from the model handle. Access fields
    # via reference to avoid copying the (non-ImplicitlyCopyable) handle.
    var msg = (
        '{"layers":' + String(Int(tier.models[slot].num_layers))
        + ',"experts":' + String(Int(tier.models[slot].num_experts))
        + ',"top_k":' + String(Int(tier.models[slot].top_k))
        + ',"bits":' + String(Int(tier.models[slot].bits))
        + ',"group_size":' + String(Int(tier.models[slot].group_size))
        + ',"hidden_size":' + String(Int(tier.models[slot].hidden_size))
        + ',"moe_intermediate":' + String(Int(tier.models[slot].moe_intermediate))
        + ',"total_tensors":' + String(Int(tier.models[slot].total_tensors))
        + ',"expert_tensors":' + String(Int(tier.models[slot].expert_tensors))
        + ',"hits":' + String(Int(tier.models[slot].hits))
        + ',"misses":' + String(Int(tier.models[slot].misses))
        + ',"prefetched_hits":' + String(Int(tier.models[slot].prefetched_hits))
        + ',"ns_count":' + String(tier.ns_count(slot))
        + '}'
    )
    writer.append_bulk_string_response(msg.unsafe_ptr(), msg.byte_length())
    return 1


@always_inline
def _hist_save(mut tier: MoEExpertTier, slot: Int, ns: Int, L: Int, E: Int,
                path: String, mut writer: ResponseWriter) raises -> Int:
    """Write the access-count snapshot for `(slot, ns)` to `<path>` via
    tmp+rename. One file holds exactly one namespace's counts; the MHST format
    is unchanged (namespace identity lives in the operator's filename)."""
    # Count non-zero entries first
    var n_entries = UInt32(0)
    for layer in range(L):
        for e in range(E):
            if tier.get_access_count(slot, ns, layer, e) > UInt64(0):
                n_entries += UInt32(1)

    # Allocate buffer: 4 (magic) + 1 (version) + 4 (n_entries) + N * 10
    var total = 9 + Int(n_entries) * 10
    var buf = alloc[UInt8](total)
    # Magic: "MHST"
    buf[unsafe_offset=0] = UInt8(77); buf[unsafe_offset=1] = UInt8(72); buf[unsafe_offset=2] = UInt8(83); buf[unsafe_offset=3] = UInt8(84)
    # Version 1
    buf[unsafe_offset=4] = UInt8(1)
    # n_entries u32 LE
    var n = Int(n_entries)
    buf[unsafe_offset=5] = UInt8(n & 0xFF)
    buf[unsafe_offset=6] = UInt8((n >> 8) & 0xFF)
    buf[unsafe_offset=7] = UInt8((n >> 16) & 0xFF)
    buf[unsafe_offset=8] = UInt8((n >> 24) & 0xFF)
    # Entries
    var off = 9
    for layer in range(L):
        for e in range(E):
            var c = tier.get_access_count(slot, ns, layer, e)
            if c > UInt64(0):
                buf[unsafe_offset=off] = UInt8(layer)
                buf[unsafe_offset=off + 1] = UInt8(e)
                var ci = Int(c)
                for k in range(8):
                    buf[unsafe_offset=off + 2 + k] = UInt8((ci >> (k * 8)) & 0xFF)
                off += 10

    # Write to <path>.tmp then atomic rename
    var tmp_path = path + ".tmp\0"
    var fd = external_call["pion_creat", Int32](tmp_path.as_c_string_slice())
    if fd < 0:
        buf.unsafe_free()
        writer.append_error_response("ERR HIST SAVE: cannot create " + path)
        return 1
    var written = external_call["pion_write", Int](fd, buf, total)
    _ = external_call["close", Int32](fd)
    buf.unsafe_free()
    if written != total:
        writer.append_error_response("ERR HIST SAVE: short write")
        return 1
    var src = path + ".tmp\0"
    var dst = path + "\0"
    var rc = external_call["pion_snapshot_rename", Int32](
        src.as_c_string_slice(), dst.as_c_string_slice())
    if rc != Int32(0):
        writer.append_error_response("ERR HIST SAVE: rename failed")
        return 1
    writer.append_ok_response()
    return 1


def _hist_load(mut tier: MoEExpertTier, slot: Int, ns: Int, L: Int, E: Int,
                path: String, mut writer: ResponseWriter) raises -> Int:
    """Read snapshot from `<path>` and replace this `(slot, ns)` band's counts."""
    var cpath = path + "\0"
    var fd = external_call["pion_open_rdonly", Int32](cpath.as_c_string_slice())
    if fd < 0:
        writer.append_error_response("ERR HIST LOAD: cannot open " + path)
        return 1
    # Read header (9 bytes)
    var hdr = alloc[UInt8](9)
    var hr = external_call["pion_read", Int](fd, hdr, 9)
    if hr != 9:
        hdr.unsafe_free(); _ = external_call["close", Int32](fd)
        writer.append_error_response("ERR HIST LOAD: short header")
        return 1
    if hdr[unsafe_offset=0] != UInt8(77) or hdr[unsafe_offset=1] != UInt8(72) or hdr[unsafe_offset=2] != UInt8(83) or hdr[unsafe_offset=3] != UInt8(84):
        hdr.unsafe_free(); _ = external_call["close", Int32](fd)
        writer.append_error_response("ERR HIST LOAD: bad magic (expected MHST)")
        return 1
    if hdr[unsafe_offset=4] != UInt8(1):
        hdr.unsafe_free(); _ = external_call["close", Int32](fd)
        writer.append_error_response("ERR HIST LOAD: unsupported version " + String(Int(hdr[unsafe_offset=4])))
        return 1
    var n_entries = (Int(hdr[unsafe_offset=5])
                    | (Int(hdr[unsafe_offset=6]) << 8)
                    | (Int(hdr[unsafe_offset=7]) << 16)
                    | (Int(hdr[unsafe_offset=8]) << 24))
    hdr.unsafe_free()
    # Read body (n_entries * 10)
    var body_size = n_entries * 10
    if body_size < 0 or body_size > 1_000_000:
        _ = external_call["close", Int32](fd)
        writer.append_error_response("ERR HIST LOAD: n_entries out of range")
        return 1
    var body = alloc[UInt8](body_size) if body_size > 0 else alloc[UInt8](1)
    if body_size > 0:
        var br = external_call["pion_read", Int](fd, body, body_size)
        if br != body_size:
            body.unsafe_free(); _ = external_call["close", Int32](fd)
            writer.append_error_response("ERR HIST LOAD: short body")
            return 1
    _ = external_call["close", Int32](fd)

    # Zero existing counters for this (slot, ns) band, then apply
    tier.zero_access_counts_for_slot(slot, ns)
    for i in range(n_entries):
        var entry_off = i * 10
        var layer = Int(body[unsafe_offset=entry_off])
        var expert = Int(body[unsafe_offset=entry_off + 1])
        if layer >= L or expert >= E:
            continue   # skip out-of-range
        var c = UInt64(0)
        for k in range(8):
            c = c | (UInt64(Int(body[unsafe_offset=entry_off + 2 + k])) << (UInt64(k) * UInt64(8)))
        tier.set_access_count(slot, ns, layer, expert, c)
    body.unsafe_free()

    writer.append_ok_response()
    return 1


def _hist_nslist(mut tier: MoEExpertTier, slot: Int, L: Int, E: Int,
                mut writer: ResponseWriter) raises -> Int:
    """Enumerate the registered namespaces for a model:
      {"namespaces":[[ns,"name",total_fetches,active_count], ...]}
    Only bands with a registered name (name_len > 0) are listed; ns 0
    ("default") is always present."""
    var msg = String('{"namespaces":[')
    var first = True
    for ns in range(MAX_MOE_NS):
        var nl = Int(tier.ns_name_lens[slot * MAX_MOE_NS + ns])
        if nl == 0:
            continue
        # Per-namespace totals.
        var total = UInt64(0)
        var n_active = 0
        for layer in range(L):
            for e in range(E):
                var c = tier.get_access_count(slot, ns, layer, e)
                if c > UInt64(0):
                    total = total + c
                    n_active += 1
        # Namespace name (ASCII; operator-supplied simple identifiers).
        var base = (slot * MAX_MOE_NS + ns) * MOE_NS_NAME_INLINE
        var nm = String()
        for i in range(nl):
            nm += chr(Int(tier.ns_names[base + i]))
        if not first:
            msg += ','
        first = False
        msg += '[' + String(ns) + ',"' + nm + '",' + String(Int(total)) + ',' + String(n_active) + ']'
    msg += ']}'
    writer.append_bulk_string_response(msg.unsafe_ptr(), msg.byte_length())
    return 1


def handle_moe_expert_hist(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut tier: MoEExpertTier,
) raises -> Int:
    """MOE.EXPERT.HIST <model_id> [SAVE|LOAD <path>] — per-(layer, expert)
    FETCH usage histogram.

    Default form (just `<model_id>`) returns a JSON object:
      {
        "n_layers": L,
        "n_experts": E,
        "total_fetches": T,
        "active": [[layer, expert, count], ...]     // sparse, count > 0 only
      }

    `SAVE <path>` writes the access-count snapshot to disk via atomic
    rename (tmp → final). On-disk format:
      4 bytes "MHST" magic
      1 byte  version (=1)
      4 bytes u32 LE n_entries (count of non-zero (layer, expert) pairs)
      per entry: 1 byte layer, 1 byte expert, 8 bytes u64 LE count
    Returns +OK on success.

    `LOAD <path>` zeros the in-memory counters for this model, then
    applies the file's entries. Validates header magic + version.
    Returns +OK on success, -ERR on format mismatch or I/O failure.

    Operator workflow: after a workload-representative run, SAVE captures
    the routing histogram so a fresh server (canary / restart / different
    machine) can LOAD it and generate a HIST-guided prune plan without
    re-running the workload.
    """
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR MOE.EXPERT.HIST requires: model_id [SAVE|LOAD <path>]")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = tier.find_model(name_ptr, Int(name_tok.length))
    if slot < 0:
        writer.append_error_response("UNAVAILABLE MOE.EXPERT.* model not loaded")
        return 1

    var L = Int(tier.models[slot].num_layers)
    var E = Int(tier.models[slot].num_experts)

    # Subcommand dispatch — examined ONLY if a 2nd positional arg is present so
    # the legacy form (`HIST <model_id>`) keeps returning ns-0 JSON unchanged.
    # Grammar:
    #   HIST <model>                        -> JSON for ns 0 (legacy)
    #   HIST <model> NS <name>              -> JSON for that namespace
    #   HIST <model> NSLIST                 -> namespace registry JSON
    #   HIST <model> SAVE <path> [NS <name>]
    #   HIST <model> LOAD <path> [NS <name>]
    var q_ns = 0   # namespace to report on for the JSON path; default = ns 0
    if start + 2 < num_tokens:
        var sub_tok = tokens[unsafe_offset=start + 2]
        var sub_len = Int(sub_tok.length)
        var sub_ptr = sub_tok.ptr
        # SAVE | LOAD (4 bytes, case-insensitive) — require a <path> at +3.
        var is_save = sub_len == 4 and (sub_ptr[unsafe_offset=0]|0x20)==115 and (sub_ptr[unsafe_offset=1]|0x20)==97 and (sub_ptr[unsafe_offset=2]|0x20)==118 and (sub_ptr[unsafe_offset=3]|0x20)==101
        var is_load = sub_len == 4 and (sub_ptr[unsafe_offset=0]|0x20)==108 and (sub_ptr[unsafe_offset=1]|0x20)==111 and (sub_ptr[unsafe_offset=2]|0x20)==97 and (sub_ptr[unsafe_offset=3]|0x20)==100
        if is_save or is_load:
            if start + 3 >= num_tokens:
                writer.append_error_response("ERR MOE.EXPERT.HIST SAVE/LOAD requires <path>")
                return 1
            var path = tokens[unsafe_offset=start + 3].value()
            # Optional `NS <name>` at +4/+5 (default ns 0).
            var io_ns = 0
            if start + 5 < num_tokens:
                var io_kw = tokens[unsafe_offset=start + 4]
                if Int(io_kw.length) == 2 and (io_kw.ptr[unsafe_offset=0]|0x20)==110 and (io_kw.ptr[unsafe_offset=1]|0x20)==115:  # "ns"
                    var io_name_tok = tokens[unsafe_offset=start + 5]
                    var io_name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(io_name_tok.ptr))
                    var resolved = tier.resolve_ns(slot, io_name_ptr, Int(io_name_tok.length), True)
                    if resolved < 0:
                        writer.append_error_response("ERR MOE.EXPERT.* namespaces full (max " + String(MAX_MOE_NS) + " per model)")
                        return 1
                    io_ns = resolved
            if is_save:
                return _hist_save(tier, slot, io_ns, L, E, path, writer)
            return _hist_load(tier, slot, io_ns, L, E, path, writer)
        # NSLIST (6 bytes, case-insensitive)
        if sub_len == 6 and (sub_ptr[unsafe_offset=0]|0x20)==110 and (sub_ptr[unsafe_offset=1]|0x20)==115 and (sub_ptr[unsafe_offset=2]|0x20)==108 and (sub_ptr[unsafe_offset=3]|0x20)==105 and (sub_ptr[unsafe_offset=4]|0x20)==115 and (sub_ptr[unsafe_offset=5]|0x20)==116:
            return _hist_nslist(tier, slot, L, E, writer)
        # NS <name> (2 bytes) — report on one namespace. create=False: an
        # unknown name yields an empty histogram (q_ns = -1 → all counts 0).
        if sub_len == 2 and (sub_ptr[unsafe_offset=0]|0x20)==110 and (sub_ptr[unsafe_offset=1]|0x20)==115:
            if start + 3 >= num_tokens:
                writer.append_error_response("ERR MOE.EXPERT.HIST NS requires <name>")
                return 1
            var ns_name_tok = tokens[unsafe_offset=start + 3]
            var ns_name_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ns_name_tok.ptr))
            q_ns = tier.resolve_ns(slot, ns_name_ptr, Int(ns_name_tok.length), False)
        else:
            writer.append_error_response("ERR MOE.EXPERT.HIST: unknown subcommand (expected NS|NSLIST|SAVE|LOAD)")
            return 1

    # Pre-compute total + active entries for the target namespace.
    var total = UInt64(0)
    var n_active = 0
    for layer in range(L):
        for e in range(E):
            var c = tier.get_access_count(slot, q_ns, layer, e)
            if c > UInt64(0):
                total = total + c
                n_active += 1

    # Build JSON. For Gemma 4 sized models: ~3-4 KB compact output worst case.
    var msg = '{"n_layers":' + String(L) + ',"n_experts":' + String(E)
    msg += ',"total_fetches":' + String(Int(total))
    msg += ',"active_count":' + String(n_active)
    msg += ',"active":['
    var first = True
    for layer in range(L):
        for e in range(E):
            var c = tier.get_access_count(slot, q_ns, layer, e)
            if c > UInt64(0):
                if not first:
                    msg += ','
                first = False
                msg += '[' + String(layer) + ',' + String(e) + ',' + String(Int(c)) + ']'
    msg += ']}'

    writer.append_bulk_string_response(msg.unsafe_ptr(), msg.byte_length())
    return 1


@always_inline
def handle_moe_expert_stats(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut tier: MoEExpertTier,
) raises -> Int:
    """MOE.EXPERT.STATS — tier-wide counters as JSON.

    Always available. Reports real counters from the MoEExpertTier struct
    (currently all zero until backend lands). The structure of the JSON is
    stable: consumers can rely on these field names across stages.
    """
    var enabled_str = "true" if tier.enabled else "false"
    var n_total = tier.hits + tier.misses
    var stage_str = '2' if tier.enabled else '1'
    var msg = (
        '{"stage":' + stage_str
        + ',"enabled":' + enabled_str
        + ',"models_loaded":' + String(tier.model_count())
        + ',"hits":' + String(Int(tier.hits))
        + ',"misses":' + String(Int(tier.misses))
        + ',"prefetched_hits":' + String(Int(tier.prefetched_hits))
        + ',"evictions":' + String(Int(tier.evictions))
        + ',"cache_bytes_used":' + String(Int(tier.cache_bytes_used))
        + ',"cache_max_bytes":' + String(Int(tier.cache_max_bytes))
        + ',"pinned_count":' + String(Int(tier.pinned_count))
        + ',"pruned_count":' + String(Int(tier.pruned_count))
        + '}'
    )
    writer.append_bulk_string_response(msg.unsafe_ptr(), msg.byte_length())
    return 1
