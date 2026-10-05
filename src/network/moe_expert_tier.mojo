"""MoEExpertTier — server-side substrate for MoE expert weights.

gh #61 Phase-0 → Stage 2 (in flight). Stage 1 (commit `cbe4730`) shipped the
RESP + binary wire-surface; this struct is the backend that those handlers
will route through once disk I/O lands. For now the tier holds counters that
MOE.EXPERT.STATS can read.

Architectural plan (mirrors the validated Python `pion_moe_tier.MoEExpertTier`
in examples/pion_moe_tier.py):

  - Per-(model_id, layer_id, expert_id) byte-offset map parsed from
    safetensors at startup. One map per loaded model. Up to MAX_MOE_MODELS
    concurrent models.
  - LRU cache of fetched expert payloads. Cold-fetch path uses
    `os.pread` equivalents (kqueue on macOS, io_uring on Linux) so SSD I/O
    stays off the event-loop thread.
  - Pinning support (PIN/UNPIN) excludes entries from LRU eviction.
  - Manifest available via INFO; per-tier counters via STATS.

This file is the Stage-2 SKELETON: the struct exists, fields are wired,
counters are zero-initialized. Disk I/O + manifest parsing are follow-on
commits. Stage-1 stub RESP handlers in `src/commands/moe_expert.mojo`
continue to return UNAVAILABLE / +OK until those land.
"""

from std.collections import Array
from std.ffi import external_call
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset


# Capacity limits — small for Phase 0; grow when productionization demands it.
comptime MAX_MOE_MODELS = 4
comptime MOE_MODEL_NAME_INLINE = 64
# Offset table sizing — covers every published MoE arch including DeepSeek-V3 (61 layers).
comptime MAX_MOE_LAYERS = 64
comptime MOE_PROJ_COUNT = 3      # gate_proj, up_proj, down_proj
comptime MOE_COMP_COUNT = 3      # weight, scales, biases (only "weight" for bf16/f16/f32)
comptime MOE_OFFSETS_PER_MODEL = MAX_MOE_LAYERS * MOE_PROJ_COUNT * MOE_COMP_COUNT
comptime MAX_SHARD_PATH_INLINE = 256
# Per-(layer, expert) usage counter table — sized to cover every published
# MoE arch. Gemma 4 = 128 experts; DeepSeek-V3 = 256 (would need bump).
comptime MAX_MOE_EXPERTS = 128
# Per-distribution HIST namespaces (traffic classes). The access-count
# histogram is banked per (model, namespace) so operators serving a mixture
# of distributions can compute a prune set that is safe across ALL of them,
# instead of over-fitting to one narrow class (see gh #61 retraction 7b383d4).
# ns 0 is the implicit "default" — omitting NS reproduces the pre-namespace
# behavior. 8 covers ~6-7 real traffic classes + headroom. access_counts at
# max sizing is 4×8×64×128×8B = 2 MB/worker — fine.
comptime MAX_MOE_NS = 8
comptime MOE_NS_NAME_INLINE = 32
# Stage 2k LRU cache: pre-assembled multi-blob payloads keyed by (slot, layer, expert).
comptime MAX_MOE_CACHE_ENTRIES = 256

# Architecture flags for offset table — set during load_manifest
comptime MOE_ARCH_UNKNOWN  = UInt8(0)
comptime MOE_ARCH_GEMMA4   = UInt8(1)   # language_model.model.layers.{L}.experts.switch_glu.{proj}.{comp}
comptime MOE_ARCH_PHI35    = UInt8(2)   # model.layers.{L}.block_sparse_moe.switch_mlp.{proj}.{comp}
comptime MOE_ARCH_OLMOE    = UInt8(3)   # model.layers.{L}.mlp.experts.{E}.{proj}.{comp} — per-expert
comptime MOE_ARCH_MIXTRAL  = UInt8(4)   # model.layers.{L}.block_sparse_moe.experts.{E}.w{1,3,2}.{comp} — per-expert; w1=gate, w3=up, w2=down
# Path shape kinds for shard storage
# config.json files are < 64 KB for every model we've seen (Gemma 4 26B is ~10 KB).
comptime MAX_CONFIG_JSON_BYTES = 65536
# safetensors.index.json gets large for big models — Gemma 4 26B is ~80 KB,
# DeepSeek-V3 can be ~3 MB. Cap to 4 MB.
comptime MAX_INDEX_JSON_BYTES = 4 * 1024 * 1024


# ─── Minimal JSON scanner for the safetensors `config.json` subset ────────────
# Not a real parser — finds `"key":<int>` patterns by byte scan. Sufficient for
# the flat-ish keys we need (num_hidden_layers, num_experts, top_k_experts,
# num_local_experts, num_experts_per_tok). Handles nested keys correctly
# because the search just matches the key→colon→integer triple regardless of
# how deeply nested the parent object is.

@always_inline
def _is_digit(b: UInt8) -> Bool:
    return b >= 48 and b <= 57

@always_inline
def _is_ws(b: UInt8) -> Bool:
    return b == 32 or b == 9 or b == 10 or b == 13

@always_inline
def _find_int_value(buf: Pointer[UInt8, MutUntrackedOrigin], buf_len: Int,
                    key_ptr: Pointer[UInt8, MutUntrackedOrigin], key_len: Int) -> Int:
    """Find `"<key>"<ws>:<ws><int>` in buf. Returns the integer, or -1 if absent.

    Conservative: only matches when the key is wrapped in double-quotes
    immediately followed by colon (allowing whitespace). Won't match keys
    that are prefixes of other keys (e.g. "num_experts" won't accidentally
    match "num_experts_per_tok").
    """
    var i = 0
    while i + key_len + 3 < buf_len:
        # Match opening quote at position i
        if buf[unsafe_offset=i] != 34:  # "
            i += 1
            continue
        # Check key match starting at i+1
        var match_ok = True
        for k in range(key_len):
            if buf[unsafe_offset=i + 1 + k] != key_ptr[unsafe_offset=k]:
                match_ok = False
                break
        if not match_ok:
            i += 1
            continue
        # Check closing quote
        if buf[unsafe_offset=i + 1 + key_len] != 34:  # "
            i += 1
            continue
        # Skip whitespace, expect colon
        var j = i + 2 + key_len
        while j < buf_len and _is_ws(buf[unsafe_offset=j]):
            j += 1
        if j >= buf_len or buf[unsafe_offset=j] != 58:  # :
            i += 1
            continue
        j += 1
        while j < buf_len and _is_ws(buf[unsafe_offset=j]):
            j += 1
        # Parse integer (positive only; safetensors counts are non-negative)
        if j >= buf_len or not _is_digit(buf[unsafe_offset=j]):
            i += 1
            continue
        var n = 0
        while j < buf_len and _is_digit(buf[unsafe_offset=j]):
            n = n * 10 + (Int(buf[unsafe_offset=j]) - 48)
            j += 1
        return n
    return -1


@always_inline
def _count_occurrences(buf: Pointer[UInt8, MutUntrackedOrigin], buf_len: Int,
                       pat_ptr: Pointer[UInt8, MutUntrackedOrigin], pat_len: Int) -> Int:
    """Count non-overlapping occurrences of `pat` in `buf`."""
    if pat_len <= 0 or buf_len < pat_len:
        return 0
    var count = 0
    var i = 0
    while i + pat_len <= buf_len:
        var match_ok = True
        for k in range(pat_len):
            if buf[unsafe_offset=i + k] != pat_ptr[unsafe_offset=k]:
                match_ok = False
                break
        if match_ok:
            count += 1
            i += pat_len
        else:
            i += 1
    return count


@always_inline
def _pad5(n: Int) -> String:
    """Zero-pad an integer to 5 digits: 11 → "00011"."""
    if n >= 10000: return String(n)
    if n >= 1000:  return "0" + String(n)
    if n >= 100:   return "00" + String(n)
    if n >= 10:    return "000" + String(n)
    return "0000" + String(n)


@always_inline
def _read_u64_le(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int) -> UInt64:
    """Read a uint64 in little-endian — matches safetensors header-length convention."""
    return (UInt64(buf[unsafe_offset=offset])
          | (UInt64(buf[unsafe_offset=offset + 1]) << 8)
          | (UInt64(buf[unsafe_offset=offset + 2]) << 16)
          | (UInt64(buf[unsafe_offset=offset + 3]) << 24)
          | (UInt64(buf[unsafe_offset=offset + 4]) << 32)
          | (UInt64(buf[unsafe_offset=offset + 5]) << 40)
          | (UInt64(buf[unsafe_offset=offset + 6]) << 48)
          | (UInt64(buf[unsafe_offset=offset + 7]) << 56))


def _shard_total_count(idx_buf: Pointer[UInt8, MutUntrackedOrigin], idx_len: Int) -> Int:
    """Find the total shard count from index.json by scanning for the
    `-of-NNNNN.safetensors` filename pattern. Returns -1 if not found.
    """
    var pat: StaticString = "-of-"
    var pat_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pat.unsafe_ptr()))
    var pat_len = pat.byte_length()
    var i = 0
    while i + pat_len + 5 < idx_len:
        var match_ok = True
        for k in range(pat_len):
            if idx_buf[unsafe_offset=i + k] != pat_ptr[unsafe_offset=k]:
                match_ok = False
                break
        if match_ok:
            # Skip the "-of-" then parse digits
            var j = i + pat_len
            if _is_digit(idx_buf[unsafe_offset=j]):
                var n = 0
                while j < idx_len and _is_digit(idx_buf[unsafe_offset=j]):
                    n = n * 10 + (Int(idx_buf[unsafe_offset=j]) - 48)
                    j += 1
                return n
        i += 1
    return -1


def _read_shard_header_bytes(shard_path: String) -> Int:
    """Read the first 8 bytes of a safetensors shard, return the header byte
    length (u64 LE). Returns -1 on error.
    """
    var cpath = shard_path
    var fd = external_call["pion_open_rdonly", Int32](cpath.as_c_string_slice())
    if fd < 0:
        return -1
    var hdr_len_buf = alloc[UInt8](8)
    var got = external_call["pion_read", Int](fd, hdr_len_buf, 8)
    _ = external_call["close", Int32](fd)
    if got != 8:
        hdr_len_buf.unsafe_free()
        return -1
    var hl = Int(_read_u64_le(hdr_len_buf, 0))
    hdr_len_buf.unsafe_free()
    return hl


def _read_shard_header_json(shard_path: String,
                            out_buf: Pointer[UInt8, MutUntrackedOrigin],
                            out_capacity: Int) -> Int:
    """Open shard, read 8-byte u64 header_len, then read header_len bytes of JSON
    into out_buf. Returns header_len on success, -1 on error.
    """
    var cpath = shard_path
    var fd = external_call["pion_open_rdonly", Int32](cpath.as_c_string_slice())
    if fd < 0:
        return -1
    var hdr_len_buf = alloc[UInt8](8)
    var got = external_call["pion_read", Int](fd, hdr_len_buf, 8)
    if got != 8:
        hdr_len_buf.unsafe_free()
        _ = external_call["close", Int32](fd)
        return -1
    var hl = Int(_read_u64_le(hdr_len_buf, 0))
    hdr_len_buf.unsafe_free()
    if hl <= 0 or hl > out_capacity:
        _ = external_call["close", Int32](fd)
        return -1
    var total = 0
    while total < hl:
        var n = external_call["pion_read", Int](fd, out_buf.unsafe_offset(total), hl - total)
        if n <= 0:
            break
        total += n
    _ = external_call["close", Int32](fd)
    return total if total == hl else -1


# @always_inline, not for speed: callers pass a String's bytes, which live
# INLINE in the caller's stack frame when the String is <= 23 bytes. An
# out-of-line call with that pointer is emitted as `tail call` and -O3 may
# drop the stores (gh #349 / #384; `pixi run audit-tail-alloca`).
@always_inline
def _find_data_offsets_for_key(buf: Pointer[UInt8, MutUntrackedOrigin], buf_len: Int,
                                key_ptr: Pointer[UInt8, MutUntrackedOrigin], key_len: Int,
                                out_start: Pointer[Int, MutUntrackedOrigin],
                                out_end: Pointer[Int, MutUntrackedOrigin]) -> Bool:
    """Find `"<key>":{...,"data_offsets":[START,END],...}` in a safetensors header.

    Writes START to out_start, END to out_end. Returns True on success.

    Conservative byte-scan: matches the key wrapped in quotes followed by
    colon, then within the next ~512 bytes (covers a typical tensor entry
    object), finds the `"data_offsets":` substring and parses [N,N].
    """
    out_start[] = 0
    out_end[] = 0
    var i = 0
    while i + key_len + 4 < buf_len:
        if buf[unsafe_offset=i] != 34:  # "
            i += 1
            continue
        var match_ok = True
        for k in range(key_len):
            if buf[unsafe_offset=i + 1 + k] != key_ptr[unsafe_offset=k]:
                match_ok = False
                break
        if not match_ok or buf[unsafe_offset=i + 1 + key_len] != 34:
            i += 1
            continue
        # Found "key". Scan forward for "data_offsets":[...]
        var search_end = i + 1024 + key_len
        if search_end > buf_len:
            search_end = buf_len
        var pat: StaticString = "\"data_offsets\":"
        var pat_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pat.unsafe_ptr()))
        var pat_len = pat.byte_length()
        var j = i + 2 + key_len
        while j + pat_len < search_end:
            var pm = True
            for k in range(pat_len):
                if buf[unsafe_offset=j + k] != pat_ptr[unsafe_offset=k]:
                    pm = False
                    break
            if pm:
                # Found "data_offsets":, parse [N,N]
                var p = j + pat_len
                while p < search_end and _is_ws(buf[unsafe_offset=p]):
                    p += 1
                if p < search_end and buf[unsafe_offset=p] == 91:  # [
                    p += 1
                    while p < search_end and _is_ws(buf[unsafe_offset=p]):
                        p += 1
                    var n1 = 0
                    while p < search_end and _is_digit(buf[unsafe_offset=p]):
                        n1 = n1 * 10 + (Int(buf[unsafe_offset=p]) - 48)
                        p += 1
                    while p < search_end and _is_ws(buf[unsafe_offset=p]):
                        p += 1
                    if p < search_end and buf[unsafe_offset=p] == 44:  # ,
                        p += 1
                        while p < search_end and _is_ws(buf[unsafe_offset=p]):
                            p += 1
                        var n2 = 0
                        while p < search_end and _is_digit(buf[unsafe_offset=p]):
                            n2 = n2 * 10 + (Int(buf[unsafe_offset=p]) - 48)
                            p += 1
                        out_start[] = n1
                        out_end[] = n2
                        return True
                break  # data_offsets found but malformed
            j += 1
        i += 1
    return False


@always_inline
def _read_file_all(path: String, buf: Pointer[UInt8, MutUntrackedOrigin],
                   buf_capacity: Int) -> Int:
    """Read up to buf_capacity bytes from path. Returns bytes read, or -1 on error.

    Uses the pion C shim used by snapshot/WAL loaders. No mmap.
    """
    var cpath = path
    var fd = external_call["pion_open_rdonly", Int32](cpath.as_c_string_slice())
    if fd < 0:
        return -1
    var total = 0
    while total < buf_capacity:
        var got = external_call["pion_read", Int](fd, buf.unsafe_offset(total), buf_capacity - total)
        if got <= 0:
            break
        total += got
    _ = external_call["close", Int32](fd)
    return total


struct MoETensorOffset(Movable, Copyable, ImplicitlyCopyable):
    """One entry in the per-(layer, proj, comp) offset table.

    For stacked-tensor MoE archs (Gemma 4, Phi-3.5 mlx-community), each
    (layer, proj, comp) has ONE tensor of shape [num_experts, *].
    `data_start` is the absolute byte offset of that tensor within `shard_id`
    (1-indexed); `per_expert_bytes` is (data_end - data_start) / num_experts.

    A FETCH for (layer, expert, proj, comp) reads:
      pread(shard_fd, per_expert_bytes, data_start + expert * per_expert_bytes)
    """
    var shard_id: Int32         # 1..N from `model-NNNNN-of-NNNNN.safetensors`; 0 = not present
    var data_start: UInt64      # absolute byte offset within the shard (after shard's u64 header_len)
    var per_expert_bytes: UInt64

    def __init__(out self):
        self.shard_id = 0
        self.data_start = UInt64(0)
        self.per_expert_bytes = UInt64(0)


struct MoEModelHandle(Movable, Copyable):
    """One loaded MoE model: identity + manifest stats.

    Stage-2 will add a layout map and shard descriptors. For now this struct
    holds identity + zero counters so the INFO handler has a real model
    record to report when it lands.
    """
    var active: Bool
    var name_len: Int32
    var name: Array[UInt8, MOE_MODEL_NAME_INLINE]
    var num_layers: Int32
    var num_experts: Int32
    var top_k: Int32
    var bits: Int32        # 0=lossless; 4/8/etc for quantized
    var group_size: Int32
    var hidden_size: Int32          # model hidden dimension (e.g. 2816 for Gemma 4 26B)
    var moe_intermediate: Int32     # per-expert FFN intermediate dim (e.g. 704)
    var total_tensors: Int32        # all tensors in safetensors.index.json
    var expert_tensors: Int32       # subset whose name contains "experts"
    # Per-model fetch counters (will tick once the cold path lands).
    var hits: UInt64
    var misses: UInt64
    var prefetched_hits: UInt64

    def __init__(out self):
        self.active = False
        self.name_len = 0
        self.name = Array[UInt8, MOE_MODEL_NAME_INLINE](fill=UInt8(0))
        self.num_layers = 0
        self.num_experts = 0
        self.top_k = 0
        self.bits = 0
        self.group_size = 0
        self.hidden_size = 0
        self.moe_intermediate = 0
        self.total_tensors = 0
        self.expert_tensors = 0
        self.hits = UInt64(0)
        self.misses = UInt64(0)
        self.prefetched_hits = UInt64(0)


struct MoEExpertTier:
    """Per-worker MoE-expert tier registry.

    Holds up to MAX_MOE_MODELS model handles plus a fixed-size offset table.
    The handlers in `src/commands/moe_expert.mojo` read this state to serve
    INFO and STATS, and (Stage 2) FETCH / PREFETCH / PIN / UNPIN.

    Offset-table layout (flat, indexed by helper):
      offsets[model_slot * MOE_OFFSETS_PER_MODEL
              + layer * (MOE_PROJ_COUNT * MOE_COMP_COUNT)
              + proj * MOE_COMP_COUNT
              + comp]
    """
    var enabled: Bool                  # set by --moe-cache flag at startup
    var models: Array[MoEModelHandle, MAX_MOE_MODELS]
    # Per-model shard directory path (where the safetensors files live).
    # Stored as a fixed-size byte buffer indexed by model_slot.
    var shard_dirs: Array[UInt8, MAX_MOE_MODELS * MAX_SHARD_PATH_INLINE]
    var shard_dir_lens: Array[Int32, MAX_MOE_MODELS]
    var shard_counts: Array[Int32, MAX_MOE_MODELS]
    var archs: Array[UInt8, MAX_MOE_MODELS]
    # Per-(model, layer, proj, comp) offset table.
    var offsets: Array[MoETensorOffset, MAX_MOE_MODELS * MOE_OFFSETS_PER_MODEL]
    # Per-(model, namespace, layer, expert) usage counter — incremented on every
    # FETCH. Surfaces per-traffic-class routing concentration so operators can
    # identify pruning candidates that are safe across a mixture of workloads.
    # Indexed by _access_idx(model, ns, layer, expert). ns 0 = "default" (the
    # pre-namespace band). ~2 MB at max sizing — HEAP-allocated, NOT inline: this
    # struct lives by-value on the worker thread stack (Pion → NetworkEngine →
    # SlowPathHandler, constructed in main.mojo's worker_task), and a 2 MB inline
    # table overflows parallelize()'s pooled worker-thread stacks (SIGBUS/SIGSEGV
    # at -w ≥2; worker 0 survives on the main-thread stack). Heap backing keeps
    # the struct small; the alloc is never freed (one per worker, process-lifetime
    # — same discipline as main.mojo's shared allocs). Pointer `[idx]` indexing is
    # source-identical to Array, so the accessor methods are unchanged.
    var access_counts: Pointer[UInt64, MutUntrackedOrigin]
    # Per-(model, layer, expert) prune flag — set via MOE.EXPERT.PRUNE. NOT
    # namespaced: a pruned expert is one physical weight tier, global across
    # traffic classes. The client computes the prune set from the per-ns
    # histograms and issues global PRUNE. Indexed by _pruned_idx (3-D). Bool
    # array is ~32 KB at max sizing (4 × 64 × 128).
    var pruned: Array[Bool, MAX_MOE_MODELS * MAX_MOE_LAYERS * MAX_MOE_EXPERTS]
    var pruned_count: UInt32
    # Per-(model, namespace) name registry. ns 0 is always "default". A namespace
    # is created on first FETCH ... NS <name>. Names are matched case-sensitively.
    var ns_names: Array[UInt8, MAX_MOE_MODELS * MAX_MOE_NS * MOE_NS_NAME_INLINE]
    var ns_name_lens: Array[Int32, MAX_MOE_MODELS * MAX_MOE_NS]
    # Stage 4b: opaque handle to the C-side background-warming thread pool
    # (src/ffi/moe_warm_pool.c). UInt64(0) means uninitialized. Future Stage
    # 4b-3 wires PREFETCH → moe_warm_pool_enqueue and the engine event loop
    # → moe_warm_pool_drain → cache_store. For Stage 4b-1 the pool is
    # initialised on tier construction (when --moe-cache is on) and torn down
    # on shutdown, but no enqueue happens yet.
    var warm_pool: UInt64
    # Per-expert offset table for per-expert MoE layouts (OLMoE). UInt64(0)
    # means "this model uses the stacked offset table" (Gemma 4, Phi-3.5).
    # For OLMoE, holds a heap address to MAX_MOE_LAYERS * MAX_MOE_EXPERTS *
    # MOE_PROJ_COUNT * MOE_COMP_COUNT MoETensorOffset entries = 73,728 × ~24B
    # ≈ 1.77 MB per loaded OLMoE model. Heap-allocated lazily in load_manifest
    # so models with stacked layouts don't pay for it.
    var per_expert_addrs: Array[UInt64, MAX_MOE_MODELS]
    # Stage 2k LRU cache — pre-assembled multi-blob FETCH payloads.
    # Indexed parallel arrays for simple linear scan; capped at MAX_MOE_CACHE_ENTRIES.
    var cache_active: Array[Bool, MAX_MOE_CACHE_ENTRIES]
    var cache_pinned: Array[Bool, MAX_MOE_CACHE_ENTRIES]
    var cache_slot: Array[Int32, MAX_MOE_CACHE_ENTRIES]
    var cache_layer: Array[Int32, MAX_MOE_CACHE_ENTRIES]
    var cache_expert: Array[Int32, MAX_MOE_CACHE_ENTRIES]
    var cache_ts: Array[UInt64, MAX_MOE_CACHE_ENTRIES]
    # Store the raw byte-buffer pointers as UInt64 addresses (Mojo Array
    # of Pointer is awkward; addresses round-trip cleanly).
    var cache_data_addr: Array[UInt64, MAX_MOE_CACHE_ENTRIES]
    var cache_data_len: Array[Int32, MAX_MOE_CACHE_ENTRIES]
    var cache_clock: UInt64
    var cache_bytes_used: UInt64
    var cache_max_bytes: UInt64
    # Aggregate tier-wide counters (sum across models).
    var hits: UInt64
    var misses: UInt64
    var prefetched_hits: UInt64
    var evictions: UInt64
    var pinned_count: UInt32

    def __init__(out self, enabled: Bool = False, cache_max_bytes: UInt64 = UInt64(0)):
        self.enabled = enabled
        self.models = Array[MoEModelHandle, MAX_MOE_MODELS](fill=MoEModelHandle())
        self.shard_dirs = Array[UInt8, MAX_MOE_MODELS * MAX_SHARD_PATH_INLINE](fill=UInt8(0))
        self.shard_dir_lens = Array[Int32, MAX_MOE_MODELS](fill=Int32(0))
        self.shard_counts = Array[Int32, MAX_MOE_MODELS](fill=Int32(0))
        self.archs = Array[UInt8, MAX_MOE_MODELS](fill=MOE_ARCH_UNKNOWN)
        self.offsets = Array[MoETensorOffset, MAX_MOE_MODELS * MOE_OFFSETS_PER_MODEL](
            fill=MoETensorOffset())
        # Heap-backed (see field comment): keep this 2 MB table off the worker
        # thread stack. alloc() does not zero, so memset the whole band to 0.
        comptime _ACCESS_N = MAX_MOE_MODELS * MAX_MOE_NS * MAX_MOE_LAYERS * MAX_MOE_EXPERTS
        self.access_counts = alloc[UInt64](_ACCESS_N)
        unsafe_memset(self.access_counts.unsafe_bitcast[UInt8](), 0, _ACCESS_N * 8)
        self.pruned = Array[Bool, MAX_MOE_MODELS * MAX_MOE_LAYERS * MAX_MOE_EXPERTS](
            fill=False)
        self.pruned_count = UInt32(0)
        self.ns_names = Array[UInt8, MAX_MOE_MODELS * MAX_MOE_NS * MOE_NS_NAME_INLINE](fill=UInt8(0))
        self.ns_name_lens = Array[Int32, MAX_MOE_MODELS * MAX_MOE_NS](fill=Int32(0))
        self.warm_pool = UInt64(0)   # init_warm_pool() initialises on demand
        self.per_expert_addrs = Array[UInt64, MAX_MOE_MODELS](fill=UInt64(0))
        self.cache_active = Array[Bool, MAX_MOE_CACHE_ENTRIES](fill=False)
        self.cache_pinned = Array[Bool, MAX_MOE_CACHE_ENTRIES](fill=False)
        self.cache_slot = Array[Int32, MAX_MOE_CACHE_ENTRIES](fill=Int32(-1))
        self.cache_layer = Array[Int32, MAX_MOE_CACHE_ENTRIES](fill=Int32(-1))
        self.cache_expert = Array[Int32, MAX_MOE_CACHE_ENTRIES](fill=Int32(-1))
        self.cache_ts = Array[UInt64, MAX_MOE_CACHE_ENTRIES](fill=UInt64(0))
        self.cache_data_addr = Array[UInt64, MAX_MOE_CACHE_ENTRIES](fill=UInt64(0))
        self.cache_data_len = Array[Int32, MAX_MOE_CACHE_ENTRIES](fill=Int32(0))
        self.cache_clock = UInt64(0)
        self.cache_bytes_used = UInt64(0)
        self.cache_max_bytes = cache_max_bytes
        self.hits = UInt64(0)
        self.misses = UInt64(0)
        self.prefetched_hits = UInt64(0)
        self.evictions = UInt64(0)
        self.pinned_count = UInt32(0)

    @always_inline
    def _offset_idx(self, model_slot: Int, layer: Int, proj: Int, comp: Int) -> Int:
        return (model_slot * MOE_OFFSETS_PER_MODEL
                + layer * (MOE_PROJ_COUNT * MOE_COMP_COUNT)
                + proj * MOE_COMP_COUNT
                + comp)

    @always_inline
    def _access_idx(self, model_slot: Int, ns: Int, layer: Int, expert: Int) -> Int:
        return (((model_slot * MAX_MOE_NS + ns) * MAX_MOE_LAYERS
                 + layer) * MAX_MOE_EXPERTS
                + expert)

    @always_inline
    def _pruned_idx(self, model_slot: Int, layer: Int, expert: Int) -> Int:
        return (model_slot * (MAX_MOE_LAYERS * MAX_MOE_EXPERTS)
                + layer * MAX_MOE_EXPERTS
                + expert)

    def bump_access(mut self, model_slot: Int, ns: Int, layer: Int, expert: Int):
        """Increment the per-(model, ns, layer, expert) FETCH counter. Caller is
        responsible for bounds-checking layer/expert.
        """
        if ns < 0 or ns >= MAX_MOE_NS:
            return
        var idx = self._access_idx(model_slot, ns, layer, expert)
        if idx >= 0 and idx < MAX_MOE_MODELS * MAX_MOE_NS * MAX_MOE_LAYERS * MAX_MOE_EXPERTS:
            self.access_counts[unsafe_offset=idx] = self.access_counts[unsafe_offset=idx] + UInt64(1)

    @always_inline
    def get_access_count(self, model_slot: Int, ns: Int, layer: Int, expert: Int) -> UInt64:
        if ns < 0 or ns >= MAX_MOE_NS:
            return UInt64(0)
        return self.access_counts[unsafe_offset=self._access_idx(model_slot, ns, layer, expert)]

    def set_access_count(mut self, model_slot: Int, ns: Int, layer: Int, expert: Int, count: UInt64):
        """Direct setter — used by MOE.EXPERT.HIST LOAD to restore counters
        from disk. Bounds-checks before write."""
        if ns < 0 or ns >= MAX_MOE_NS:
            return
        var idx = self._access_idx(model_slot, ns, layer, expert)
        if idx >= 0 and idx < MAX_MOE_MODELS * MAX_MOE_NS * MAX_MOE_LAYERS * MAX_MOE_EXPERTS:
            self.access_counts[unsafe_offset=idx] = count

    def zero_access_counts_for_slot(mut self, model_slot: Int, ns: Int):
        """Clear all (layer, expert) counters for one (model, ns) band. Used by
        MOE.EXPERT.HIST LOAD before applying the snapshot's values."""
        if model_slot < 0 or model_slot >= MAX_MOE_MODELS or ns < 0 or ns >= MAX_MOE_NS:
            return
        var base = (model_slot * MAX_MOE_NS + ns) * (MAX_MOE_LAYERS * MAX_MOE_EXPERTS)
        for i in range(MAX_MOE_LAYERS * MAX_MOE_EXPERTS):
            self.access_counts[unsafe_offset=base + i] = UInt64(0)

    # ─── Per-distribution HIST namespace registry ─────────────────────────────

    def init_ns_registry_for_slot(mut self, model_slot: Int):
        """Reset a model slot's namespace registry to just ns 0 = "default".
        Called from load_manifest when a slot is (re)claimed, and zeroes every
        namespace band's counters so a reused slot starts clean."""
        if model_slot < 0 or model_slot >= MAX_MOE_MODELS:
            return
        for ns in range(MAX_MOE_NS):
            self.ns_name_lens[model_slot * MAX_MOE_NS + ns] = Int32(0)
            self.zero_access_counts_for_slot(model_slot, ns)
        # ns 0 = "default"
        var base = (model_slot * MAX_MOE_NS + 0) * MOE_NS_NAME_INLINE
        self.ns_names[base + 0] = UInt8(100)  # 'd'
        self.ns_names[base + 1] = UInt8(101)  # 'e'
        self.ns_names[base + 2] = UInt8(102)  # 'f'
        self.ns_names[base + 3] = UInt8(97)   # 'a'
        self.ns_names[base + 4] = UInt8(117)  # 'u'
        self.ns_names[base + 5] = UInt8(108)  # 'l'
        self.ns_names[base + 6] = UInt8(116)  # 't'
        self.ns_name_lens[model_slot * MAX_MOE_NS + 0] = Int32(7)

    def resolve_ns(mut self, model_slot: Int, name_ptr: Pointer[UInt8, MutUntrackedOrigin],
                  name_len: Int, create: Bool) -> Int:
        """Map a namespace name to its slot index (0..MAX_MOE_NS-1) for a model.
        Returns the existing slot on match; on miss, assigns the next free slot
        if `create` else returns -1; returns -1 if the registry is full.
        Names longer than MOE_NS_NAME_INLINE are truncated on store."""
        if model_slot < 0 or model_slot >= MAX_MOE_MODELS:
            return -1
        var store_len = name_len
        if store_len > MOE_NS_NAME_INLINE:
            store_len = MOE_NS_NAME_INLINE
        # Scan for an existing match.
        for ns in range(MAX_MOE_NS):
            var nl = Int(self.ns_name_lens[model_slot * MAX_MOE_NS + ns])
            if nl != store_len:
                continue
            var base = (model_slot * MAX_MOE_NS + ns) * MOE_NS_NAME_INLINE
            var match_ok = True
            for i in range(store_len):
                if self.ns_names[base + i] != name_ptr[unsafe_offset=i]:
                    match_ok = False
                    break
            if match_ok:
                return ns
        if not create:
            return -1
        # Assign the first free slot (name_len == 0). ns 0 is always "default"
        # so a freshly-claimed slot never hands out ns 0 to a named namespace.
        for ns in range(MAX_MOE_NS):
            if Int(self.ns_name_lens[model_slot * MAX_MOE_NS + ns]) == 0:
                var base = (model_slot * MAX_MOE_NS + ns) * MOE_NS_NAME_INLINE
                for i in range(store_len):
                    self.ns_names[base + i] = name_ptr[unsafe_offset=i]
                self.ns_name_lens[model_slot * MAX_MOE_NS + ns] = Int32(store_len)
                return ns
        return -1   # registry full

    @always_inline
    def ns_count(self, model_slot: Int) -> Int:
        """Number of registered namespaces for a model (ns 0 always counts)."""
        if model_slot < 0 or model_slot >= MAX_MOE_MODELS:
            return 0
        var n = 0
        for ns in range(MAX_MOE_NS):
            if Int(self.ns_name_lens[model_slot * MAX_MOE_NS + ns]) > 0:
                n += 1
        return n

    @always_inline
    def is_pruned(self, model_slot: Int, layer: Int, expert: Int) -> Bool:
        var idx = self._pruned_idx(model_slot, layer, expert)
        if idx < 0 or idx >= MAX_MOE_MODELS * MAX_MOE_LAYERS * MAX_MOE_EXPERTS:
            return False
        return self.pruned[idx]

    def set_pruned(mut self, model_slot: Int, layer: Int, expert: Int, on: Bool) -> Bool:
        """Mark/unmark an expert as pruned. Returns the new state. Updates
        tier-wide pruned_count counter."""
        var idx = self._pruned_idx(model_slot, layer, expert)
        if idx < 0 or idx >= MAX_MOE_MODELS * MAX_MOE_LAYERS * MAX_MOE_EXPERTS:
            return False
        var was = self.pruned[idx]
        if on and not was:
            self.pruned[idx] = True
            self.pruned_count = self.pruned_count + UInt32(1)
        elif not on and was:
            self.pruned[idx] = False
            if self.pruned_count > UInt32(0):
                self.pruned_count = self.pruned_count - UInt32(1)
        return on

    # ─── Stage 4b background-warming thread pool ──────────────────────────────
    # FFI surface for src/ffi/moe_warm_pool.c. Stage 4b-1 ships init/shutdown
    # only; the pool exists but no Mojo code enqueues to it yet. Stage 4b-2
    # ports the pread/blob-assembly inner loop into the C worker (replaces the
    # ENOSYS stub). Stage 4b-3 wires PREFETCH → enqueue and the engine event
    # loop → drain → cache_store.

    def init_warm_pool(mut self):
        """Spawn the warming thread + allocate ring buffers. Called from
        slow_path's CommandDispatcher init after load_manifest succeeds.
        Idempotent: safe to call when warm_pool is already initialised."""
        if self.warm_pool != UInt64(0):
            return
        var addr = external_call["moe_warm_pool_init", UInt64](
            UInt64(64), UInt64(64))
        self.warm_pool = addr

    def shutdown_warm_pool(mut self):
        """Join the warming thread + free buffers. Called on engine teardown.
        Safe to call before init."""
        if self.warm_pool == UInt64(0):
            return
        var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self.warm_pool))
        _ = external_call["moe_warm_pool_shutdown", NoneType](p)
        self.warm_pool = UInt64(0)

    def enqueue_warm(mut self, model_slot: Int, layer: Int, expert: Int) -> Int:
        """Enqueue a (model_slot, layer, expert) prefetch request on the
        warming thread. Packs the 9-entry offset table for this layer into
        the 180-byte LE-packed layout the C worker expects + passes the
        model's shard_dir as a stable pointer.

        Returns 0 on success, -1 if the warming thread's request ring is
        full (caller should fall back to fadvise / sync FETCH path).
        """
        if self.warm_pool == UInt64(0):
            return -1
        if model_slot < 0 or model_slot >= MAX_MOE_MODELS:
            return -1
        if layer < 0 or layer >= MAX_MOE_LAYERS:
            return -1
        if expert < 0 or expert >= Int(self.models[model_slot].num_experts):
            return -1
        # Stage 4b's warm-pool C worker is hard-wired to the stacked layout
        # (data_start + expert * per_expert_bytes). For per-expert layouts
        # (OLMoE) the offsets aren't strided, so warming would compute the
        # wrong byte range. Reject here; PREFETCH falls back to fadvise.
        if self.uses_per_expert_offsets(model_slot):
            return -1
        # Pack offset table for this (model, layer): 9 entries × 20 bytes.
        # Layout per entry: i32 shard_id | u64 data_start | u64 per_expert_bytes
        # The C worker reads these as LE-packed; we serialise byte-by-byte
        # for endianness safety even though M-series is LE.
        var packed = Array[UInt8, 180](fill=UInt8(0))
        for i in range(9):
            var p = i // 3
            var c = i % 3
            var e = self.get_offset(model_slot, layer, p, c)
            var base = i * 20
            var sid = UInt32(Int32(e.shard_id))
            packed[base + 0] = UInt8(Int(sid) & 0xFF)
            packed[base + 1] = UInt8((Int(sid) >> 8) & 0xFF)
            packed[base + 2] = UInt8((Int(sid) >> 16) & 0xFF)
            packed[base + 3] = UInt8((Int(sid) >> 24) & 0xFF)
            var ds = e.data_start
            for k in range(8):
                packed[base + 4 + k] = UInt8(Int(ds >> (UInt64(k) * UInt64(8))) & 0xFF)
            var pe = e.per_expert_bytes
            for k in range(8):
                packed[base + 12 + k] = UInt8(Int(pe >> (UInt64(k) * UInt64(8))) & 0xFF)
        # Shard dir: build a null-terminated copy so the C side gets a valid string
        # (strncpy in C copies up to MOE_WARM_SHARD_PATH_MAX-1 bytes regardless).
        var dir_str = self.get_shard_dir(model_slot) + "\0"
        var pool = Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(self.warm_pool))
        var rc = external_call["moe_warm_pool_enqueue", Int32](
            pool,
            Int32(model_slot), Int32(layer), Int32(expert),
            dir_str.unsafe_ptr(),
            Int32(Int(self.shard_counts[model_slot])),
            packed.unsafe_ptr(),
            Int(180),
        )
        return Int(rc)

    def drain_warm_into_cache(mut self) -> Int:
        """Drain up to N completions from the warming thread and insert each
        successfully-warmed buffer into the LRU cache via cache_store. Called
        once per event-loop tick. Returns the count of successful inserts.

        Completion records arrive on a SPSC ring; this is the ONLY consumer.
        Status<0 means the worker failed (file open, pread short read, etc.);
        in that case we increment a counter and move on (caller's downstream
        FETCH will fall back to the synchronous path).
        """
        if self.warm_pool == UInt64(0):
            return 0
        # Drain up to 16 completions per tick — keeps event-loop latency
        # bounded; remaining completions wait for the next tick.
        comptime DRAIN_BATCH = 16
        # Each moe_warm_completion_t is 32 bytes (4*int32 + uint64 + 2*int32 padded).
        # Concretely: slot:i32 layer:i32 expert:i32 addr:u64 data_len:i32 status:i32.
        # Mojo doesn't directly model C structs so we allocate a flat byte buffer
        # and reach into it field-by-field.
        var COMP_BYTES = 32
        var buf = alloc[UInt8](DRAIN_BATCH * COMP_BYTES)
        var pool = Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(self.warm_pool))
        var n = Int(external_call["moe_warm_pool_drain", UInt64](
            pool, buf, UInt64(DRAIN_BATCH)))
        var inserted = 0
        for i in range(n):
            var off = i * COMP_BYTES
            # Layout (matches struct moe_warm_completion_t):
            #   0..4   : slot (i32)
            #   4..8   : layer (i32)
            #   8..12  : expert (i32)
            #   12..16 : padding (compiler may insert; check below)
            #   16..24 : addr (u64)
            #   24..28 : data_len (i32)
            #   28..32 : status (i32)
            # Note on alignment: with 3×i32 then a u64, the compiler typically
            # pads to 8-byte alignment for addr → 4 bytes padding at off 12.
            # If a future C compiler reorders, this read pattern needs revisiting.
            var slot_i = Int((buf.unsafe_offset(off)).unsafe_bitcast[Int32]()[])
            var layer_i = Int((buf.unsafe_offset(off).unsafe_offset(4)).unsafe_bitcast[Int32]()[])
            var expert_i = Int((buf.unsafe_offset(off).unsafe_offset(8)).unsafe_bitcast[Int32]()[])
            var addr_u = (buf.unsafe_offset(off).unsafe_offset(16)).unsafe_bitcast[UInt64]()[]
            var data_len_i = Int((buf.unsafe_offset(off).unsafe_offset(24)).unsafe_bitcast[Int32]()[])
            var status_i = Int((buf.unsafe_offset(off).unsafe_offset(28)).unsafe_bitcast[Int32]()[])
            if status_i == 0 and addr_u != UInt64(0) and data_len_i > 0:
                # The worker malloc'd the buffer via libc; Mojo's cache uses
                # tcmalloc. Copy the bytes into a Mojo-allocated buffer so the
                # cache's eventual eviction calls Mojo's free against a
                # tcmalloc-owned pointer. Then libc-free the C buffer.
                # Cost: one memcpy of ~12 MB on M4 ≈ 3 ms — well under the
                # 21 ms cold-FETCH cost we'd otherwise pay.
                var src_ptr = Pointer[UInt8, MutUntrackedOrigin](
                    unsafe_from_address=Int(addr_u))
                var mojo_buf = alloc[UInt8](data_len_i)
                unsafe_memcpy(dest=mojo_buf, src=src_ptr, count=data_len_i)
                _ = external_call["moe_warm_free_buffer", NoneType](addr_u)
                self.cache_store(slot_i, layer_i, expert_i,
                                  UInt64(Int(mojo_buf)), data_len_i)
                inserted += 1
            else:
                # Worker errored — free the C buffer if any.
                if addr_u != UInt64(0):
                    _ = external_call["moe_warm_free_buffer", NoneType](addr_u)
        buf.unsafe_free()
        return inserted

    @always_inline
    def get_offset(self, model_slot: Int, layer: Int, proj: Int, comp: Int) -> MoETensorOffset:
        return self.offsets[self._offset_idx(model_slot, layer, proj, comp)]

    @always_inline
    def _set_offset(mut self, model_slot: Int, layer: Int, proj: Int, comp: Int,
                    shard_id: Int, data_start: UInt64, per_expert_bytes: UInt64):
        var idx = self._offset_idx(model_slot, layer, proj, comp)
        self.offsets[idx].shard_id = Int32(shard_id)
        self.offsets[idx].data_start = data_start
        self.offsets[idx].per_expert_bytes = per_expert_bytes

    def get_shard_dir(self, model_slot: Int) -> String:
        """Return the model directory path (where safetensors shards live)."""
        var n = Int(self.shard_dir_lens[model_slot])
        if n <= 0:
            return String("")
        var base = model_slot * MAX_SHARD_PATH_INLINE
        var out = String("")
        for i in range(n):
            out += chr(Int(self.shard_dirs[base + i]))
        return out

    def get_shard_path(self, model_slot: Int, shard_id: Int) -> String:
        """Construct `<model_dir>/model-NNNNN-of-NNNNN.safetensors`, or
        `<model_dir>/model.safetensors` if the model is single-shard
        (shard_count == 1)."""
        var dir = self.get_shard_dir(model_slot)
        var n = Int(self.shard_counts[model_slot])
        if n == 1:
            return dir + "/model.safetensors"
        return dir + "/model-" + _pad5(shard_id) + "-of-" + _pad5(n) + ".safetensors"

    # ─── Per-expert offset table (OLMoE arch) ─────────────────────────────────
    # Per-expert layouts store one tensor per (layer, expert, proj, comp) — far
    # more entries than the stacked layout. We heap-allocate a per-model offset
    # table when the arch is OLMOE so models with stacked layouts (Gemma 4,
    # Phi-3.5) don't pay the ~1.77 MB cost.

    @always_inline
    def _per_expert_idx(self, layer: Int, expert: Int, proj: Int, comp: Int) -> Int:
        """Index within ONE model's per-expert offset table."""
        return (layer * (MAX_MOE_EXPERTS * MOE_PROJ_COUNT * MOE_COMP_COUNT)
                + expert * (MOE_PROJ_COUNT * MOE_COMP_COUNT)
                + proj * MOE_COMP_COUNT
                + comp)

    def _alloc_per_expert_table(mut self, model_slot: Int):
        """Heap-allocate the per-expert offset table for this model (if not
        already allocated). Used by load_manifest when arch == OLMOE."""
        if self.per_expert_addrs[model_slot] != UInt64(0):
            return
        var n = MAX_MOE_LAYERS * MAX_MOE_EXPERTS * MOE_PROJ_COUNT * MOE_COMP_COUNT
        var buf = alloc[MoETensorOffset](n)
        for i in range(n):
            (buf.unsafe_offset(i)).unsafe_write(MoETensorOffset())
        self.per_expert_addrs[model_slot] = UInt64(Int(buf))

    def _set_per_expert_offset(mut self, model_slot: Int, layer: Int, expert: Int,
                                proj: Int, comp: Int,
                                shard_id: Int, data_start: UInt64, per_expert_bytes: UInt64):
        var addr = self.per_expert_addrs[model_slot]
        if addr == UInt64(0):
            return
        var table = Pointer[MoETensorOffset, MutUntrackedOrigin](unsafe_from_address=Int(addr))
        var idx = self._per_expert_idx(layer, expert, proj, comp)
        table[unsafe_offset=idx].shard_id = Int32(shard_id)
        table[unsafe_offset=idx].data_start = data_start
        table[unsafe_offset=idx].per_expert_bytes = per_expert_bytes

    def get_per_expert_offset(self, model_slot: Int, layer: Int, expert: Int,
                                proj: Int, comp: Int) -> MoETensorOffset:
        """Read offset from the per-expert table. Returns a default
        MoETensorOffset (shard_id=0) if not allocated / not populated."""
        var addr = self.per_expert_addrs[model_slot]
        if addr == UInt64(0):
            return MoETensorOffset()
        var table = Pointer[MoETensorOffset, MutUntrackedOrigin](unsafe_from_address=Int(addr))
        var idx = self._per_expert_idx(layer, expert, proj, comp)
        return table[unsafe_offset=idx]

    @always_inline
    def uses_per_expert_offsets(self, model_slot: Int) -> Bool:
        """True if this model uses the per-expert offset table (OLMoE);
        False if it uses the stacked offset table (Gemma 4, Phi-3.5)."""
        return self.per_expert_addrs[model_slot] != UInt64(0)

    @always_inline
    def find_model(self, name_ptr: Pointer[UInt8, MutUntrackedOrigin], name_len: Int) -> Int:
        """Return slot index for a loaded model by id-prefix match, or -1."""
        for s in range(MAX_MOE_MODELS):
            if not self.models[s].active:
                continue
            if Int(self.models[s].name_len) != name_len:
                continue
            var match_ok = True
            for i in range(name_len):
                if self.models[s].name[i] != name_ptr[unsafe_offset=i]:
                    match_ok = False
                    break
            if match_ok:
                return s
        return -1

    @always_inline
    def model_count(self) -> Int:
        """Number of currently-loaded models."""
        var n = 0
        for s in range(MAX_MOE_MODELS):
            if self.models[s].active:
                n += 1
        return n

    # ─── Stage 2k LRU cache ───────────────────────────────────────────────────
    # The cache stores pre-assembled multi-blob FETCH payloads keyed by
    # (model_slot, layer, expert). Linear-scan lookup (bounded at
    # MAX_MOE_CACHE_ENTRIES); LRU eviction via cache_ts comparison.

    def _cache_find(self, slot: Int, layer: Int, expert: Int) -> Int:
        """Return cache slot idx for (slot, layer, expert), or -1 if not present."""
        for i in range(MAX_MOE_CACHE_ENTRIES):
            if (self.cache_active[i]
                    and Int(self.cache_slot[i]) == slot
                    and Int(self.cache_layer[i]) == layer
                    and Int(self.cache_expert[i]) == expert):
                return i
        return -1

    def _cache_touch(mut self, idx: Int):
        """Mark cache entry as most-recently-used."""
        self.cache_clock += UInt64(1)
        self.cache_ts[idx] = self.cache_clock

    def _cache_evict_one(mut self) -> Bool:
        """Free the LRU non-pinned cache entry. Returns True if evicted."""
        var victim = -1
        var min_ts = UInt64.MAX
        for i in range(MAX_MOE_CACHE_ENTRIES):
            if not self.cache_active[i] or self.cache_pinned[i]:
                continue
            if self.cache_ts[i] < min_ts:
                min_ts = self.cache_ts[i]
                victim = i
        if victim < 0:
            return False
        var addr = self.cache_data_addr[victim]
        if addr != UInt64(0):
            var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(addr))
            p.unsafe_free()
        self.cache_bytes_used -= UInt64(Int(self.cache_data_len[victim]))
        self.cache_active[victim] = False
        self.cache_pinned[victim] = False
        self.cache_data_addr[victim] = UInt64(0)
        self.cache_data_len[victim] = Int32(0)
        self.evictions += UInt64(1)
        return True

    def _cache_insert(mut self, slot: Int, layer: Int, expert: Int,
                      data_addr: UInt64, data_len: Int) -> Int:
        """Insert a new cache entry. Evicts LRU non-pinned entries until the
        new payload fits within cache_max_bytes. Returns inserted slot idx,
        or -1 if all entries are pinned and the new payload doesn't fit.
        """
        # Evict until there's room
        while (self.cache_bytes_used + UInt64(data_len) > self.cache_max_bytes
                and self.cache_max_bytes > UInt64(0)):
            if not self._cache_evict_one():
                break  # all pinned; insert anyway (overshoots budget)
        # Find a free slot
        var free_slot = -1
        for i in range(MAX_MOE_CACHE_ENTRIES):
            if not self.cache_active[i]:
                free_slot = i
                break
        if free_slot < 0:
            # All slots occupied — evict LRU and retry
            if not self._cache_evict_one():
                return -1
            for i in range(MAX_MOE_CACHE_ENTRIES):
                if not self.cache_active[i]:
                    free_slot = i
                    break
            if free_slot < 0:
                return -1
        self.cache_active[free_slot] = True
        self.cache_slot[free_slot] = Int32(slot)
        self.cache_layer[free_slot] = Int32(layer)
        self.cache_expert[free_slot] = Int32(expert)
        self.cache_data_addr[free_slot] = data_addr
        self.cache_data_len[free_slot] = Int32(data_len)
        self.cache_bytes_used += UInt64(data_len)
        self._cache_touch(free_slot)
        return free_slot

    def cache_lookup(mut self, slot: Int, layer: Int, expert: Int,
                     out_addr: Pointer[UInt64, MutUntrackedOrigin],
                     out_len: Pointer[Int, MutUntrackedOrigin]) -> Bool:
        """Hot path. Returns True on cache hit and writes (addr, len)."""
        var idx = self._cache_find(slot, layer, expert)
        if idx < 0:
            return False
        self._cache_touch(idx)
        out_addr[] = self.cache_data_addr[idx]
        out_len[] = Int(self.cache_data_len[idx])
        self.hits += UInt64(1)
        return True

    def cache_store(mut self, slot: Int, layer: Int, expert: Int,
                     data_addr: UInt64, data_len: Int):
        """Insert into cache. Caller's malloc'd buffer is now owned by the
        cache; do NOT free it (the cache frees on eviction).
        """
        _ = self._cache_insert(slot, layer, expert, data_addr, data_len)

    def cache_pin(mut self, slot: Int, layer: Int, expert: Int) -> Bool:
        var idx = self._cache_find(slot, layer, expert)
        if idx < 0: return False
        if not self.cache_pinned[idx]:
            self.cache_pinned[idx] = True
            self.pinned_count += UInt32(1)
        return True

    def cache_unpin(mut self, slot: Int, layer: Int, expert: Int) -> Bool:
        var idx = self._cache_find(slot, layer, expert)
        if idx < 0: return False
        if self.cache_pinned[idx]:
            self.cache_pinned[idx] = False
            self.pinned_count -= UInt32(1)
        return True

    def cache_evict(mut self, slot: Int, layer: Int, expert: Int) -> Bool:
        """Force-evict a specific (slot, layer, expert) entry from the LRU
        cache, releasing its buffer. Used by MOE.EXPERT.PRUNE to ensure stale
        bytes don't outlive the prune decision. Returns True if evicted."""
        var idx = self._cache_find(slot, layer, expert)
        if idx < 0:
            return False
        var addr = self.cache_data_addr[idx]
        if addr != UInt64(0):
            var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(addr))
            p.unsafe_free()
        if self.cache_pinned[idx] and self.pinned_count > UInt32(0):
            self.pinned_count -= UInt32(1)
        self.cache_bytes_used -= UInt64(Int(self.cache_data_len[idx]))
        self.cache_active[idx] = False
        self.cache_pinned[idx] = False
        self.cache_data_addr[idx] = UInt64(0)
        self.cache_data_len[idx] = Int32(0)
        self.evictions += UInt64(1)
        return True

    def _enumerate_into_offsets(mut self, model_slot: Int, arch: UInt8, num_experts: Int,
                                shard_id: Int,
                                hbuf: Pointer[UInt8, MutUntrackedOrigin], hbuf_len: Int,
                                data_base: UInt64) -> Int:
        """Iterate every (layer, proj, comp) key that could live in this shard.
        For each match, fill the offset table at [model_slot][layer][proj][comp].

        Returns the number of tensor entries found in this shard.

        `data_base` = 8 + header_len, i.e. the byte position where tensor
        data starts in the shard file. data_offsets in the JSON header are
        relative to this base, so absolute = data_base + json_data_start.
        """
        # ── OLMoE per-expert layout ────────────────────────────────────────
        # Each (layer, expert, proj, comp) is its own tensor in safetensors.
        # Naming pattern: model.layers.{L}.mlp.experts.{E}.{proj}.{comp}
        # Total: L × E × 3 × 3 distinct tensor names (e.g. OLMoE-1B-7B has
        # 16 × 64 × 9 = 9,216). We fan-out to populate the per-expert
        # offset table allocated in load_manifest.
        if arch == MOE_ARCH_OLMOE:
            var num_found_olmoe = 0
            var ds2 = alloc[Int](1); var de2 = alloc[Int](1)
            for L in range(Int(self.models[model_slot].num_layers)):
                for E in range(num_experts):
                    for p in range(MOE_PROJ_COUNT):
                        var proj_name = String("gate_proj") if p == 0 else (String("up_proj") if p == 1 else String("down_proj"))
                        for c in range(MOE_COMP_COUNT):
                            var comp = String(".weight") if c == 0 else (String(".scales") if c == 1 else String(".biases"))
                            var key = (String("model.layers.") + String(L)
                                      + String(".mlp.experts.") + String(E)
                                      + String(".") + proj_name + comp)
                            var f = _find_data_offsets_for_key(hbuf, hbuf_len,
                                Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key.unsafe_ptr())),
                                key.byte_length(), ds2, de2)
                            _ = key^    # read through a pointer that does not keep it alive
                            if f:
                                var json_start = UInt64(ds2[])
                                var json_end = UInt64(de2[])
                                var tensor_bytes = json_end - json_start
                                var abs_offset = data_base + json_start
                                self._set_per_expert_offset(model_slot, L, E, p, c,
                                                              shard_id, abs_offset, tensor_bytes)
                                num_found_olmoe += 1
            ds2.unsafe_free(); de2.unsafe_free()
            return num_found_olmoe

        # ── Mixtral per-expert layout ─────────────────────────────────────
        # Same per-expert offset-table shape as OLMoE, but tensors are named
        # `model.layers.{L}.block_sparse_moe.experts.{E}.w{1,3,2}.{comp}`.
        # Mapping: p=0 (gate_proj) ← w1, p=1 (up_proj) ← w3, p=2 (down_proj)
        # ← w2 — the conventional Llama / Mixtral SwiGLU naming.
        if arch == MOE_ARCH_MIXTRAL:
            var num_found_mix = 0
            var ds3 = alloc[Int](1); var de3 = alloc[Int](1)
            for L in range(Int(self.models[model_slot].num_layers)):
                for E in range(num_experts):
                    for p in range(MOE_PROJ_COUNT):
                        var w_name = String("w1") if p == 0 else (String("w3") if p == 1 else String("w2"))
                        for c in range(MOE_COMP_COUNT):
                            var comp = String(".weight") if c == 0 else (String(".scales") if c == 1 else String(".biases"))
                            var key = (String("model.layers.") + String(L)
                                      + String(".block_sparse_moe.experts.") + String(E)
                                      + String(".") + w_name + comp)
                            var f = _find_data_offsets_for_key(hbuf, hbuf_len,
                                Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key.unsafe_ptr())),
                                key.byte_length(), ds3, de3)
                            _ = key^    # read through a pointer that does not keep it alive
                            if f:
                                var json_start = UInt64(ds3[])
                                var json_end = UInt64(de3[])
                                var tensor_bytes = json_end - json_start
                                var abs_offset = data_base + json_start
                                self._set_per_expert_offset(model_slot, L, E, p, c,
                                                              shard_id, abs_offset, tensor_bytes)
                                num_found_mix += 1
            ds3.unsafe_free(); de3.unsafe_free()
            return num_found_mix

        # ── Stacked layouts (Gemma 4, Phi-3.5) ────────────────────────────
        # One tensor per (layer, proj, comp), strided across experts.
        var arch_prefix = String("")
        var arch_suffix = String("")
        if arch == MOE_ARCH_GEMMA4:
            arch_prefix = String("language_model.model.layers.")
            arch_suffix = String(".experts.switch_glu.")
        elif arch == MOE_ARCH_PHI35:
            arch_prefix = String("model.layers.")
            arch_suffix = String(".block_sparse_moe.switch_mlp.")
        else:
            return 0  # unknown arch

        var num_found = 0
        var ds = alloc[Int](1); var de = alloc[Int](1)
        for L in range(Int(self.models[model_slot].num_layers)):
            for p in range(MOE_PROJ_COUNT):
                var proj_name = String("gate_proj") if p == 0 else (String("up_proj") if p == 1 else String("down_proj"))
                for c in range(MOE_COMP_COUNT):
                    var comp = String(".weight") if c == 0 else (String(".scales") if c == 1 else String(".biases"))
                    var key = arch_prefix + String(L) + arch_suffix + proj_name + comp
                    var f = _find_data_offsets_for_key(hbuf, hbuf_len,
                        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key.unsafe_ptr())),
                        key.byte_length(), ds, de)
                    _ = key^    # read through a pointer that does not keep it alive
                    if f:
                        var json_start = UInt64(ds[])
                        var json_end = UInt64(de[])
                        var per_expert = (json_end - json_start) // UInt64(num_experts)
                        var abs_offset = data_base + json_start
                        self._set_offset(model_slot, L, p, c,
                                          shard_id, abs_offset, per_expert)
                        num_found += 1
        ds.unsafe_free(); de.unsafe_free()
        return num_found

    def load_manifest(mut self, model_dir: String) -> Bool:
        """Parse `<model_dir>/config.json` and populate one MoEModelHandle slot.

        Sufficient for `MOE.EXPERT.INFO` to return real manifest values
        (num_layers, num_experts, top_k, bits, group_size). Disk offsets +
        per-expert tensor map are populated separately when the
        safetensors-shard parser lands.

        Returns True on success. Errors silently set False (server keeps
        running with the tier still enabled but no models loaded).
        """
        # Compute basename (model_id) BEFORE any disk I/O so we can dedup
        # idempotently and short-circuit. Calling MOE.EXPERT.LOAD <path> twice
        # with the same path (or a trailing-slash variant) must be a no-op
        # success, not a second slot consumption or a redundant config.json
        # read.
        var name_bytes = model_dir.as_bytes()
        var path_len = model_dir.byte_length()
        var basename_start = 0
        for k in range(path_len):
            if name_bytes[k] == 47:  # '/'
                basename_start = k + 1
        var basename_len = path_len - basename_start
        if basename_len == 0:
            # Path ends with '/'; back up one and try again
            basename_start = 0
            var k2 = path_len - 2
            while k2 >= 0:
                if name_bytes[k2] == 47:  # '/'
                    basename_start = k2 + 1
                    break
                k2 -= 1
            basename_len = (path_len - 1) - basename_start
        # Cap to inline storage
        var inline_n = basename_len if basename_len < MOE_MODEL_NAME_INLINE else MOE_MODEL_NAME_INLINE

        # Dedup check: if a model with this basename is already loaded, return
        # idempotent success without reading config.json or consuming a slot.
        # Inline byte comparison against each active slot's name Array.
        for s in range(MAX_MOE_MODELS):
            if not self.models[s].active:
                continue
            if Int(self.models[s].name_len) != inline_n:
                continue
            var match_ok = True
            for k in range(inline_n):
                if self.models[s].name[k] != name_bytes[basename_start + k]:
                    match_ok = False
                    break
            if match_ok:
                print("MoE Cache:  load_manifest idempotent — basename already loaded in slot "
                      + String(s))
                return True

        # Allocate config.json scratch
        var buf = alloc[UInt8](MAX_CONFIG_JSON_BYTES)
        unsafe_memset(buf, UInt8(0), MAX_CONFIG_JSON_BYTES)
        var cfg_path = model_dir + "/config.json"
        var n = _read_file_all(cfg_path, buf, MAX_CONFIG_JSON_BYTES)
        if n <= 0:
            buf.unsafe_free()
            print("MoE Cache:  load_manifest failed — could not read " + cfg_path)
            return False

        # Find a free slot
        var slot = -1
        for s in range(MAX_MOE_MODELS):
            if not self.models[s].active:
                slot = s
                break
        if slot < 0:
            buf.unsafe_free()
            print("MoE Cache:  load_manifest failed — no free model slot")
            return False

        # Probe keys (handles both Gemma 4 nested + Phi-3.5/OLMoE flat layouts)
        var key_num_layers: StaticString = "num_hidden_layers"
        var nl = _find_int_value(buf, n,
                                   Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_num_layers.unsafe_ptr())),
                                   key_num_layers.byte_length())
        var key_num_experts_a: StaticString = "num_experts"
        var ne_a = _find_int_value(buf, n,
                                     Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_num_experts_a.unsafe_ptr())),
                                     key_num_experts_a.byte_length())
        var key_num_experts_b: StaticString = "num_local_experts"
        var ne_b = _find_int_value(buf, n,
                                     Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_num_experts_b.unsafe_ptr())),
                                     key_num_experts_b.byte_length())
        var key_topk_a: StaticString = "top_k_experts"
        var tk_a = _find_int_value(buf, n,
                                     Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_topk_a.unsafe_ptr())),
                                     key_topk_a.byte_length())
        var key_topk_b: StaticString = "num_experts_per_tok"
        var tk_b = _find_int_value(buf, n,
                                     Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_topk_b.unsafe_ptr())),
                                     key_topk_b.byte_length())
        var key_bits: StaticString = "bits"
        var bits = _find_int_value(buf, n,
                                     Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_bits.unsafe_ptr())),
                                     key_bits.byte_length())
        var key_gs: StaticString = "group_size"
        var gs = _find_int_value(buf, n,
                                   Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_gs.unsafe_ptr())),
                                   key_gs.byte_length())
        var key_hidden: StaticString = "hidden_size"
        var hidden = _find_int_value(buf, n,
                                       Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_hidden.unsafe_ptr())),
                                       key_hidden.byte_length())
        var key_intermediate: StaticString = "moe_intermediate_size"
        var inter = _find_int_value(buf, n,
                                      Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_intermediate.unsafe_ptr())),
                                      key_intermediate.byte_length())
        # Fallback for archs that store the regular intermediate_size (Phi-3.5
        # uses intermediate_size for the MoE FFN since it has no separate
        # shared MLP).
        if inter <= 0:
            var key_int2: StaticString = "intermediate_size"
            inter = _find_int_value(buf, n,
                                       Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(key_int2.unsafe_ptr())),
                                       key_int2.byte_length())
        buf.unsafe_free()

        # Validate we found the load-bearing fields
        var ne = ne_a if ne_a > 0 else ne_b
        var tk = tk_a if tk_a > 0 else tk_b
        if nl <= 0 or ne <= 0 or tk <= 0:
            print("MoE Cache:  load_manifest failed — missing num_hidden_layers/num_experts/top_k in config.json (nl="
                  + String(nl) + " ne=" + String(ne) + " tk=" + String(tk) + ")")
            return False

        # Populate the slot. model_id = basename of the directory path
        # (e.g. HF snapshot hash) — what consumers will pass to FETCH/INFO.
        # basename_start / basename_len / inline_n / name_bytes were computed
        # above (pre-slot) for the dedup check; reuse them here.
        self.models[slot].active = True
        # Reset the namespace registry for this (possibly reused) slot: ns 0 =
        # "default", all other bands cleared. Zeroes access counters too.
        self.init_ns_registry_for_slot(slot)
        for k in range(inline_n):
            self.models[slot].name[k] = name_bytes[basename_start + k]
        self.models[slot].name_len = Int32(inline_n)
        self.models[slot].num_layers = Int32(nl)
        self.models[slot].num_experts = Int32(ne)
        self.models[slot].top_k = Int32(tk)
        self.models[slot].bits = Int32(bits) if bits > 0 else Int32(0)   # 0 = lossless
        self.models[slot].group_size = Int32(gs) if gs > 0 else Int32(0)
        self.models[slot].hidden_size = Int32(hidden) if hidden > 0 else Int32(0)
        self.models[slot].moe_intermediate = Int32(inter) if inter > 0 else Int32(0)

        # Stage 2d: also scan model.safetensors.index.json to count tensors.
        # This validates the I/O path at scale (1 MB+ JSON) and gives operators
        # an early signal that the model directory is well-formed. Per-tensor
        # byte-offset extraction happens in the per-shard header parser.
        var idx_buf = alloc[UInt8](MAX_INDEX_JSON_BYTES)
        unsafe_memset(idx_buf, UInt8(0), MAX_INDEX_JSON_BYTES)
        var idx_path = model_dir + "/model.safetensors.index.json"
        var idx_n = _read_file_all(idx_path, idx_buf, MAX_INDEX_JSON_BYTES)
        var n_shards = -1
        if idx_n > 0:
            # Count tensor entries: each maps to a "...safetensors" filename.
            var pat_st: StaticString = ".safetensors\""
            var total_t = _count_occurrences(idx_buf, idx_n,
                Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pat_st.unsafe_ptr())),
                pat_st.byte_length())
            # Count expert tensors: any key containing "experts.".
            var pat_exp: StaticString = "experts."
            var exp_t = _count_occurrences(idx_buf, idx_n,
                Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pat_exp.unsafe_ptr())),
                pat_exp.byte_length())
            self.models[slot].total_tensors = Int32(total_t)
            self.models[slot].expert_tensors = Int32(exp_t)
            # Stage 2e: extract total shard count from `-of-NNNNN` pattern.
            n_shards = _shard_total_count(idx_buf, idx_n)
            # Single-shard model (OLMoE-1B-7B-4bit etc.) has no -of- pattern in
            # index.json. If index.json was readable but no shard suffix, the
            # file is named plain `model.safetensors`; treat as n_shards = 1.
            if n_shards < 0 and total_t > 0:
                n_shards = 1
        idx_buf.unsafe_free()

        # Stages 2e–2j: Detect architecture, loop ALL N shards, populate the
        # per-(layer, proj, comp) offset table on the tier.
        if n_shards > 0:
            # Store shard_count + shard_dir on the tier
            self.shard_counts[slot] = Int32(n_shards)
            var dir_bytes_n = path_len if path_len < MAX_SHARD_PATH_INLINE else MAX_SHARD_PATH_INLINE
            self.shard_dir_lens[slot] = Int32(dir_bytes_n)
            var dir_base = slot * MAX_SHARD_PATH_INLINE
            for k in range(dir_bytes_n):
                self.shard_dirs[dir_base + k] = name_bytes[k]

            # Architecture detection: probe shard 1 with one key per arch.
            # Single-shard models name the file `model.safetensors`; multi-shard
            # name it `model-NNNNN-of-NNNNN.safetensors`.
            var shard1_path_local = (
                model_dir + "/model.safetensors" if n_shards == 1
                else model_dir + "/model-00001-of-" + _pad5(n_shards) + ".safetensors"
            )
            var shard1_hdr_len = _read_shard_header_bytes(shard1_path_local)
            var arch = MOE_ARCH_UNKNOWN
            if shard1_hdr_len > 0:
                var hdr_buf = alloc[UInt8](shard1_hdr_len + 16)
                var got = _read_shard_header_json(shard1_path_local, hdr_buf, shard1_hdr_len + 16)
                if got == shard1_hdr_len:
                    var probe_a: StaticString = "language_model.model.layers.0.experts.switch_glu.gate_proj.weight"
                    var probe_b: StaticString = "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"
                    var probe_c: StaticString = "model.layers.0.mlp.experts.0.gate_proj.weight"
                    var probe_d: StaticString = "model.layers.0.block_sparse_moe.experts.0.w1.weight"
                    var ds = alloc[Int](1); var de = alloc[Int](1)
                    if _find_data_offsets_for_key(hdr_buf, got,
                            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(probe_a.unsafe_ptr())),
                            probe_a.byte_length(), ds, de):
                        arch = MOE_ARCH_GEMMA4
                    elif _find_data_offsets_for_key(hdr_buf, got,
                            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(probe_b.unsafe_ptr())),
                            probe_b.byte_length(), ds, de):
                        arch = MOE_ARCH_PHI35
                    elif _find_data_offsets_for_key(hdr_buf, got,
                            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(probe_c.unsafe_ptr())),
                            probe_c.byte_length(), ds, de):
                        arch = MOE_ARCH_OLMOE
                    elif _find_data_offsets_for_key(hdr_buf, got,
                            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(probe_d.unsafe_ptr())),
                            probe_d.byte_length(), ds, de):
                        arch = MOE_ARCH_MIXTRAL
                    ds.unsafe_free(); de.unsafe_free()
                hdr_buf.unsafe_free()
            self.archs[slot] = arch

            # OLMoE / Mixtral: heap-allocate the per-expert offset table.
            if arch == MOE_ARCH_OLMOE or arch == MOE_ARCH_MIXTRAL:
                self._alloc_per_expert_table(slot)

            # Stage 2h+2i: loop all shards, fill offset table.
            if arch == MOE_ARCH_GEMMA4 or arch == MOE_ARCH_PHI35 or arch == MOE_ARCH_OLMOE or arch == MOE_ARCH_MIXTRAL:
                var total_found = 0
                var total_bytes = UInt64(0)
                for shard_id in range(1, n_shards + 1):
                    var shard_path = (
                        model_dir + "/model.safetensors" if n_shards == 1
                        else model_dir + "/model-" + _pad5(shard_id) + "-of-" + _pad5(n_shards) + ".safetensors"
                    )
                    var hlen = _read_shard_header_bytes(shard_path)
                    if hlen <= 0:
                        continue
                    var hbuf = alloc[UInt8](hlen + 16)
                    var hgot = _read_shard_header_json(shard_path, hbuf, hlen + 16)
                    if hgot == hlen:
                        # 8-byte u64 + header bytes = data starts at offset (8 + hlen)
                        var data_base = UInt64(8 + hlen)
                        var n_in_shard = self._enumerate_into_offsets(
                            slot, arch, ne, shard_id, hbuf, hgot, data_base)
                        total_found += n_in_shard
                    hbuf.unsafe_free()
                # Sum total bytes for log (branch on layout type — per-expert
                # vs stacked).
                if arch == MOE_ARCH_OLMOE or arch == MOE_ARCH_MIXTRAL:
                    for L in range(Int(self.models[slot].num_layers)):
                        for E in range(ne):
                            for p in range(MOE_PROJ_COUNT):
                                for c in range(MOE_COMP_COUNT):
                                    var e = self.get_per_expert_offset(slot, L, E, p, c)
                                    if e.shard_id > 0:
                                        total_bytes += e.per_expert_bytes
                else:
                    for L in range(Int(self.models[slot].num_layers)):
                        for p in range(MOE_PROJ_COUNT):
                            for c in range(MOE_COMP_COUNT):
                                var e = self.get_offset(slot, L, p, c)
                                if e.shard_id > 0:
                                    total_bytes += e.per_expert_bytes * UInt64(ne)
                print("MoE Cache:  offset table built: " + String(total_found)
                      + " expert tensors across " + String(n_shards) + " shards, "
                      + String(Int(total_bytes // UInt64(1024 * 1024))) + " MiB total")
            else:
                print("MoE Cache:  arch unknown — offset table not built")

        print("MoE Cache:  loaded manifest from " + model_dir + " (layers=" + String(nl)
              + " experts=" + String(ne) + " top_k=" + String(tk)
              + " bits=" + String(bits) + " group_size=" + String(gs)
              + " tensors=" + String(Int(self.models[slot].total_tensors))
              + " expert_tensors=" + String(Int(self.models[slot].expert_tensors)) + ")")
        return True
