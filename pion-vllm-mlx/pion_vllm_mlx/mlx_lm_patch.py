"""mlx-lm Attention monkey-patch — Stage-2 TTFT/decode path via Pion.

Where this fits:

  PionPromptCache (already shipped, see prompt_cache.py): on a warm prefix,
  client downloads K/V from Pion via V.FETCH RANGE, rebuilds the local mlx-lm
  cache, mlx-lm runs attention locally over the rebuilt cache. Saves prefill
  compute, but pays K/V transfer + local cache rebuild on every fresh session.

  This module: on a warm prefix, K/V STAY in Pion's MLX sidecar memory across
  sessions. mlx-lm's own attention is monkey-patched so each layer's
  scaled_dot_product_attention call routes the prefix portion to Pion's
  sidecar (ATTEND.PREFIX.QUERY) and merges with locally computed suffix
  attention via online softmax. K/V never leave the sidecar — Q on the wire
  per layer per forward pass.

The §28.3 batched-Q protocol (M>1) plus the §31 LSE trailer make this
feasible: mlx-lm calls scaled_dot_product_attention(Q, suffix_K, suffix_V),
the patched SDPA computes the suffix part with LSE locally, asks Pion for
prefix attention output + prefix LSE, online-merges the two.

Online softmax merge:
    p_w = exp(p_lse - max(p_lse, s_lse))
    s_w = exp(s_lse - max(p_lse, s_lse))
    output = (p_w * p_out + s_w * s_out) / (p_w + s_w)

That's mathematically equivalent to softmax over [prefix_K | suffix_K] @
[prefix_V | suffix_V] — no quality loss vs the reference attention.

Usage:
    from mlx_lm import load
    from pion_vllm_mlx import PionPromptCache, install_pion_attention_patch
    from pion_vllm_mlx.mlx_lm_patch import make_pion_prompt_cache

    model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    install_pion_attention_patch()             # one-time global patch

    pc = PionPromptCache(model, vquant="fp16")
    # First call: prefill + register + push K/V to sidecar.
    pc.get_or_prefill(prompt_ids, namespace=ns)
    # Now build a Pion-aware cache list for the next forward pass:
    cache = make_pion_prompt_cache(model, namespace=ns, prompt_cache=pc,
                                    prefix_len=len(prompt_ids))
    out = model(suffix_ids, cache=cache)        # attention runs on the sidecar

Limitations (this delivery):
- Decode step uses M=1 calls. For long generation (100+ tokens) the per-step
  wire round-trip (1 ms/layer × N_layers × 100 = several seconds) dominates.
  The TTFT-style M-tokens-at-once path is what wins big.
- Attention sinks (newer Llama variants) not handled — the sidecar's softmax
  doesn't know about sinks. Falls back to no-sink merge; safe for Llama-3.2.
- Sliding-window attention not supported here; use vanilla cache for SWA layers.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, List

import mlx.core as mx
import numpy as np

# gh #49 escape hatch — set PION_DISABLE_FUSED_ATTEND=1 to force the legacy
# 2-step path (suffix SDPA + host merge). For A/B benchmarking only; the fused
# path is mathematically identical (validated by tests/test_attend_fused_*.py).
_DISABLE_FUSED = os.environ.get("PION_DISABLE_FUSED_ATTEND", "").lower() in ("1", "true", "yes")


# ─────────────────────────────────────────────────────────────────────────────
# Cache class — stores suffix locally, marker for "prefix is in Pion"
# ─────────────────────────────────────────────────────────────────────────────


class PionPrefixCache:
    """Drop-in replacement for mlx-lm's KVCache that delegates the prefix
    portion of attention to Pion's MLX sidecar via ATTEND.PREFIX.QUERY.

    The model's Attention layer still calls cache.update_and_fetch(K, V) and
    receives K/V back. We return ONLY the locally accumulated suffix K/V; the
    patched scaled_dot_product_attention then handles the merge with the
    Pion-resident prefix.

    Identification: the patched SDPA detects this class via isinstance().
    """

    def __init__(
        self,
        layer_idx: int,
        namespace: str,
        prompt_cache,           # PionPromptCache from prompt_cache.py
        prefix_len: int,
        fa_window: Optional[int] = None,
        sparse_mode: Optional[Dict[str, int]] = None,
    ) -> None:
        self.layer_idx = layer_idx
        self.namespace = namespace
        self.prompt_cache = prompt_cache
        self.prefix_len = prefix_len
        # gh #60: sliding-window size for this layer; None = full attention.
        # On hybrid archs (Gemma 4 / Mistral SWA / Qwen3.5) sliding layers must
        # only attend to the last `fa_window` tokens; otherwise the in-proc lane
        # over-attends across the whole prefix and the model collapses to
        # repetition. tests/test_hybrid_per_layer_agreement.py is the gate.
        self.fa_window = fa_window
        # gh #60 Phase 3: block-mean top-K sparse selection. None = dense.
        # When set, sparse_mode = {"K_block": B, "K_blocks": K_top}. On the
        # in-proc lane decode path (M=1) the patched SDPA picks `K_top` blocks
        # of `B` tokens from the prefix portion using block-mean QK scoring,
        # and the model attends to (K_top*B + suffix_len) tokens instead of
        # (prefix_len + suffix_len). The last partial block (≤B tokens) is
        # always included for recency. Suffix is always kept dense.
        self.sparse_mode = sparse_mode
        # Local suffix K/V (populated by update_and_fetch) — shape (1, n_kv_heads, M, D).
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        # Token offset for RoPE — mlx-lm's Attention reads cache.offset.
        self.offset = prefix_len
        # gh #50 in-process fast lane: pull the MLX-resident prefix K/V for
        # this layer if PionPromptCache stashed them on cold prefill. When
        # set, pion_scaled_dot_product_attention runs attention locally via
        # mx.fast.scaled_dot_product_attention over [prefix | suffix] —
        # zero wire roundtrips, full GPU pipelining (no mx.eval barrier per
        # layer). When None (cross-process consumer), the wire path stays
        # in service.
        self._mlx_prefix_K: Optional[mx.array] = None
        self._mlx_prefix_V: Optional[mx.array] = None
        kv = getattr(prompt_cache, "_mlx_prefix_kv", {}).get(namespace)
        if kv is not None and 0 <= layer_idx < len(kv) and kv[layer_idx] is not None:
            # gh #59: hybrid archs may have None entries for KV-shared cache
            # positions — guard against unpacking those.
            self._mlx_prefix_K, self._mlx_prefix_V = kv[layer_idx]

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """mlx-lm's KVCache contract: append keys/values, return the current K/V.

        On the in-proc lane (cache._mlx_prefix_K populated), returns the
        FULL [prefix | suffix] K/V — sliced to the last `fa_window` tokens
        for sliding-window layers (gh #60). This matters for hybrid archs:
        KV-shared downstream layers (Gemma 4: 20 of 35) read the donor
        layer's return value via mlx-lm's `intermediates[prev_idx]` →
        `shared_kv` plumbing and pass it straight into vanilla SDPA, so the
        donor's `update_and_fetch` MUST return what the donor's attention
        actually attends to. Suffix-only returns silently corrupt the
        downstream KV-shared layers' attention.

        On the wire lane (no _mlx_prefix_K), returns suffix-only — the wire
        path doesn't materialize the prefix locally, and KV-shared layers
        aren't yet supported on the wire path (tracked separately).
        """
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concat([self.keys, keys], axis=2)
            self.values = mx.concat([self.values, values], axis=2)
        self.offset = self.prefix_len + self.keys.shape[2]
        if self._mlx_prefix_K is None:
            return self.keys, self.values
        full_K = mx.concat([self._mlx_prefix_K, self.keys], axis=2)
        full_V = mx.concat([self._mlx_prefix_V, self.values], axis=2)
        if self.fa_window and full_K.shape[2] > self.fa_window:
            full_K = full_K[..., -self.fa_window:, :]
            full_V = full_V[..., -self.fa_window:, :]
        return full_K, full_V

    # mlx-lm's generate_step does `mx.eval([c.state for c in prompt_cache])`
    # after the prompt is processed, on every cache entry, on every version
    # from 0.20.1 to 0.32.0. Without this property a make_pion_prompt_cache()
    # list could only be driven by a hand-rolled model(x, cache=...) loop —
    # which is what bench_ttft.py and test_mlx_lm_patch.py do, and why the
    # README's "drop-in" claim went unexercised: the one call an mlx-lm user
    # actually makes, `generate(..., prompt_cache=cache)`, raised
    # AttributeError. The state is the locally accumulated SUFFIX only; the
    # prefix lives in Pion and is merged by the patched SDPA, so this mirrors
    # KVCache.state's contract for the part of the sequence this object owns.
    @property
    def state(self):
        if self.keys is None:
            return ()
        return self.keys, self.values

    @state.setter
    def state(self, v) -> None:
        self.keys, self.values = v
        self.offset = self.prefix_len + (self.keys.shape[2] if self.keys is not None else 0)


# ─────────────────────────────────────────────────────────────────────────────
# Patched scaled_dot_product_attention
# ─────────────────────────────────────────────────────────────────────────────


def _quest_topk_select(
    queries: mx.array,   # (1, Hq, M=1, D)  decode-step query
    K_full: mx.array,    # (1, Hkv, prefix_len + suffix_len, D)
    V_full: mx.array,
    prefix_len: int,
    B: int,              # block size, e.g. 64
    K_top: int,          # number of blocks to pick, e.g. 8
):
    """Quest-style upper-bound top-K selector (W11 — gh #9 follow-up).

    Reference: Tang et al., "Quest: Query-Aware Sparsity for Efficient
    Long-Context LLM Inference", arXiv:2406.10774, MIT 2024.

    For each block of B contiguous prefix tokens, compute a per-block
    upper bound on `max_t (Q · K_t)` from the per-block per-dim min/max
    of K:

        UB(block) = Σ_d max(Q[d] · K_min[block, d], Q[d] · K_max[block, d])

    This is provably ≥ the actual softmax-weight-determining max within
    the block (the term that dominates dense SDPA). Selecting top-K by
    UB guarantees no false negatives among the actual top-K blocks.

    Same shape contract as `_block_mean_topk_select` — drop-in
    replacement at the decode-path selector boundary. Differs only in
    the per-block score computation; everything else (recency-keep tail,
    GQA broadcast, sort-and-concat, suffix passthrough) is identical.

    Cost vs block-mean:
      - storage: 2× per-block stats (K_min + K_max, two D-vectors per
        block) but everything is recomputed per call here — no
        precomputed K-store-side artifact (yet). Selector compute is
        ~2-3× block-mean due to the maximum() instead of mean().
      - both costs are still ≪ the SDPA itself; the selector is a
        small fraction of total decode-step time at any K_top.
    """
    Hq = queries.shape[1]
    Hkv = K_full.shape[1]
    D = K_full.shape[3]
    full_len = K_full.shape[2]
    suffix_len = full_len - prefix_len

    budget = B * K_top
    if prefix_len <= budget:
        return K_full, V_full

    prefix_K = K_full[:, :, :prefix_len, :]
    prefix_V = V_full[:, :, :prefix_len, :]
    suffix_K = K_full[:, :, prefix_len:, :] if suffix_len > 0 else None
    suffix_V = V_full[:, :, prefix_len:, :] if suffix_len > 0 else None

    N_full = (prefix_len // B) * B
    N_remain = prefix_len - N_full

    sel_K = prefix_K[:, :, :N_full, :]                            # (1, Hkv, N_full, D)
    sel_K = sel_K.reshape(1, Hkv, N_full // B, B, D)
    K_min = mx.min(sel_K, axis=3)                                 # (1, Hkv, N_full/B, D)
    K_max = mx.max(sel_K, axis=3)
    if Hq != Hkv:
        rep = Hq // Hkv
        K_min = mx.repeat(K_min, rep, axis=1)                     # (1, Hq, N_full/B, D)
        K_max = mx.repeat(K_max, rep, axis=1)

    # Quest upper bound per block: Σ_d max(Q[d]*K_min, Q[d]*K_max).
    # Both terms broadcast Q over the block axis. Take elementwise max,
    # then sum across the D axis to collapse to a single score per block.
    q = queries[:, :, 0, :]                                       # (1, Hq, D)
    scale = 1.0 / mx.sqrt(mx.array(D, dtype=mx.float32))
    q_b = q[:, :, None, :]                                        # (1, Hq, 1, D) — broadcasts over N_full/B
    term_min = q_b * K_min                                        # (1, Hq, N_full/B, D)
    term_max = q_b * K_max
    ub_per_dim = mx.maximum(term_min, term_max)                   # (1, Hq, N_full/B, D)
    scores = (mx.sum(ub_per_dim, axis=-1) * scale)                # (1, Hq, N_full/B)

    # v1 shared mask: average over heads, single top-K shared across heads.
    # Same v1 choice as block-mean; per-head plumbing is a v2 win that the
    # MSL kernel already supports but the in-proc lane intentionally keeps
    # cheap-and-shared until we measure head-divergence wins.
    shared_scores = mx.mean(scores, axis=1)                       # (1, N_full/B)

    sorted_idx = mx.argsort(-shared_scores, axis=-1)
    K_top_eff = min(K_top, N_full // B)
    top = sorted_idx[:, :K_top_eff]
    top = mx.sort(top, axis=-1)                                   # logical token order

    arange_B = mx.arange(B, dtype=top.dtype)
    token_idx = top[:, :, None] * B + arange_B[None, None, :]     # (1, K_top_eff, B)
    token_idx = token_idx.reshape(1, K_top_eff * B)

    if N_remain > 0:
        tail = mx.arange(N_full, prefix_len, dtype=token_idx.dtype)
        token_idx = mx.concat([token_idx, tail[None, :]], axis=1)

    idx_1d = token_idx.squeeze(0)
    sparse_prefix_K = mx.take(prefix_K, idx_1d, axis=2)
    sparse_prefix_V = mx.take(prefix_V, idx_1d, axis=2)

    if suffix_K is not None:
        out_K = mx.concat([sparse_prefix_K, suffix_K], axis=2)
        out_V = mx.concat([sparse_prefix_V, suffix_V], axis=2)
    else:
        out_K = sparse_prefix_K
        out_V = sparse_prefix_V
    return out_K, out_V


def _block_mean_topk_select(
    queries: mx.array,   # (1, Hq, M=1, D)  decode-step query
    K_full: mx.array,    # (1, Hkv, prefix_len + suffix_len, D)
    V_full: mx.array,
    prefix_len: int,
    B: int,              # block size, e.g. 64
    K_top: int,          # number of blocks to pick, e.g. 8
):
    """gh #60 Phase 3: block-mean top-K selector — pick `K_top` blocks of `B`
    contiguous tokens from the prefix portion by Q · mean(K_block) score.

    The last partial block (`prefix_len % B`) is always included for recency,
    so the caller doesn't lose recent context to a stale top-K pick.

    Suffix is left dense and concatenated as-is. Returns (K_sel, V_sel) with
    shape (1, Hkv, K_top*B + remainder + suffix_len, D).

    Shared mask across query heads (v1): scores are averaged over Hq, so the
    same set of prefix tokens is attended to by every head. Head-divergent
    masks are a v2 win — Pion's MSL kernel `sdpa_q1_sparse_fp32` already
    supports them; only the in-proc MLX path stays shared in v1 to keep the
    selector dispatch count low.
    """
    Hq = queries.shape[1]
    Hkv = K_full.shape[1]
    D = K_full.shape[3]
    full_len = K_full.shape[2]
    suffix_len = full_len - prefix_len

    # Cheap-out: when the prefix is already smaller than the budget, no
    # selection saves any tokens — just pass the full K/V through.
    budget = B * K_top
    if prefix_len <= budget:
        return K_full, V_full

    prefix_K = K_full[:, :, :prefix_len, :]
    prefix_V = V_full[:, :, :prefix_len, :]
    suffix_K = K_full[:, :, prefix_len:, :] if suffix_len > 0 else None
    suffix_V = V_full[:, :, prefix_len:, :] if suffix_len > 0 else None

    N_full = (prefix_len // B) * B   # truncate to a clean block boundary
    N_remain = prefix_len - N_full

    sel_K = prefix_K[:, :, :N_full, :]                            # (1, Hkv, N_full, D)
    sel_K = sel_K.reshape(1, Hkv, N_full // B, B, D)
    K_mean = mx.mean(sel_K, axis=3)                               # (1, Hkv, N_full/B, D)
    if Hq != Hkv:
        # GQA: broadcast kv-head means to query heads
        rep = Hq // Hkv
        K_mean = mx.repeat(K_mean, rep, axis=1)                   # (1, Hq, N_full/B, D)

    # Score Q (decode M=1) against block means.
    q = queries[:, :, 0, :]                                       # (1, Hq, D)
    scale = 1.0 / mx.sqrt(mx.array(D, dtype=mx.float32))
    scores = mx.matmul(q[:, :, None, :], K_mean.transpose(0, 1, 3, 2))  # (1, Hq, 1, N_full/B)
    scores = (scores.squeeze(2) * scale)                          # (1, Hq, N_full/B)

    # v1 shared mask: average across query heads, single top-K shared by all
    # heads. Pion-MSL `sdpa_q1_sparse_fp32` accepts per-head masks; v2 will
    # plumb per-head all the way through and switch to it.
    shared_scores = mx.mean(scores, axis=1)                       # (1, N_full/B)

    # Top-K blocks. MLX has argpartition but its semantics flipped between
    # 0.18 and 0.21; argsort + slice is robust and the selector cost is
    # ~negligible (N_full/B ≤ 1024 even at 64K prefix).
    sorted_idx = mx.argsort(-shared_scores, axis=-1)              # (1, N_full/B) descending
    K_top_eff = min(K_top, N_full // B)
    top = sorted_idx[:, :K_top_eff]
    # Sort selected block IDs ascending — preserves logical token order so
    # rotary positions stay in their original order when concatenated.
    top = mx.sort(top, axis=-1)                                   # (1, K_top_eff)

    # Expand block IDs to token indices: each block contributes B contiguous tokens.
    arange_B = mx.arange(B, dtype=top.dtype)
    token_idx = top[:, :, None] * B + arange_B[None, None, :]     # (1, K_top_eff, B)
    token_idx = token_idx.reshape(1, K_top_eff * B)

    # Always include the last partial block (recency keep).
    if N_remain > 0:
        tail = mx.arange(N_full, prefix_len, dtype=token_idx.dtype)
        token_idx = mx.concat([token_idx, tail[None, :]], axis=1)

    # Gather K/V at the selected indices — shared across heads.
    idx_1d = token_idx.squeeze(0)
    sparse_prefix_K = mx.take(prefix_K, idx_1d, axis=2)
    sparse_prefix_V = mx.take(prefix_V, idx_1d, axis=2)

    if suffix_K is not None:
        out_K = mx.concat([sparse_prefix_K, suffix_K], axis=2)
        out_V = mx.concat([sparse_prefix_V, suffix_V], axis=2)
    else:
        out_K = sparse_prefix_K
        out_V = sparse_prefix_V
    return out_K, out_V


def _resolve_mask(mask, M: int, S: int):
    """mlx-lm 0.31+ may pass mask="causal" (string sentinel) instead of an
    array. Translate it into an additive bias of shape (M, S) that
    broadcasts onto (B, H, M, S) scores. None / array passthrough."""
    if mask is None or not isinstance(mask, str):
        return mask
    if mask != "causal":
        raise NotImplementedError(f"mask sentinel {mask!r} not supported")
    # Suffix has M new queries against S total accumulated K (S >= M).
    # Query at row q sees K positions 0..(S - M + q) inclusive.
    offset = S - M
    rows = mx.arange(M)[:, None]
    cols = mx.arange(S)[None, :]
    return mx.where(cols > offset + rows, -1e9, 0.0)


def _suffix_sdpa_with_lse(
    Q: mx.array,        # (B, H, M, D)
    K: mx.array,        # (B, H, S, D)  S = suffix tokens so far
    V: mx.array,        # (B, H, S, D)
    scale: float,
    mask,
):
    """Manual softmax SDPA that ALSO returns rowwise LSE.

    mlx's mx.fast.scaled_dot_product_attention is fused & faster but only
    returns the output. For the merge we need LSE on the suffix side too —
    so this path. Suffix is always small (decode: 1..N_decoded; TTFT: M
    new tokens), so the slowdown vs. fused SDPA is minor in practice."""
    scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) * scale  # (B, H, M, S)
    M_q, S_k = Q.shape[2], K.shape[2]
    resolved = _resolve_mask(mask, M_q, S_k)
    if resolved is not None:
        scores = scores + resolved
    m = mx.max(scores, axis=-1, keepdims=True)              # (B, H, M, 1)
    exp_scores = mx.exp(scores - m)
    sum_exp = mx.sum(exp_scores, axis=-1, keepdims=True)
    weights = exp_scores / sum_exp
    output = mx.matmul(weights, V)                          # (B, H, M, D)
    lse = (m + mx.log(sum_exp)).squeeze(-1)                 # (B, H, M)
    return output, lse


def _online_softmax_merge(
    p_out: mx.array, p_lse: mx.array,
    s_out: mx.array, s_lse: mx.array,
):
    """Combine two attention results computed over disjoint K/V partitions
    into the equivalent of a single softmax over their union.

      m       = max(p_lse, s_lse)
      p_w     = exp(p_lse - m)
      s_w     = exp(s_lse - m)
      output  = (p_w * p_out + s_w * s_out) / (p_w + s_w)

    Shapes: p_out/s_out (B, H, M, D), p_lse/s_lse (B, H, M). All broadcast on D.
    """
    m = mx.maximum(p_lse, s_lse)
    p_w = mx.exp(p_lse - m)[..., None]
    s_w = mx.exp(s_lse - m)[..., None]
    total = p_w + s_w
    return (p_w * p_out + s_w * s_out) / total


def _legacy_pion_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: "PionPrefixCache",
    scale: float,
    mask,
):
    """Pre-gh-#49 two-step path: suffix SDPA on the host, prefix attention on
    Pion, online-softmax merge on the host. ~50 ms / 16 layers
    measured. Kept for the custom-array-mask fallback
    case the fused kernel doesn't yet handle."""
    B, Hq, M, D = queries.shape
    Hkv = keys.shape[1]
    if Hkv != Hq:
        rep = Hq // Hkv
        keys_full = mx.repeat(keys, rep, axis=1)
        values_full = mx.repeat(values, rep, axis=1)
    else:
        rep = 1
        keys_full = keys
        values_full = values
    s_out, s_lse = _suffix_sdpa_with_lse(queries, keys_full, values_full, scale, mask)
    if cache.prefix_len <= 0:
        return s_out
    Q_np = np.array(queries[0]).astype(np.float32)
    if rep > 1:
        Q_send = Q_np.reshape(Hkv, rep, M, D).reshape(Hkv, rep * M, D)
    else:
        Q_send = Q_np if M > 1 else Q_np.reshape(Hq, 1, D)
    # gh #60 Step 1: per-layer window also reaches the legacy non-fused path.
    p_out_np, p_lse_np = cache.prompt_cache.attend_query(
        cache.namespace, cache.layer_idx, Q_send, with_lse=True,
        fa_window=cache.fa_window,
    )
    if rep > 1:
        p_out_np = p_out_np.reshape(Hkv, rep, M, D).reshape(Hq, M, D)
        p_lse_np = p_lse_np.reshape(Hkv, rep, M).reshape(Hq, M)
    p_out = mx.array(p_out_np)[None, ...]
    p_lse = mx.array(p_lse_np)[None, ...]
    return _online_softmax_merge(p_out, p_lse, s_out, s_lse)


def pion_scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: Any,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
):
    """Replacement for mlx_lm.models.base.scaled_dot_product_attention.

    Pion-aware path triggers iff `cache` is a PionPrefixCache. Else falls
    through to the original implementation (preserved as ``_orig_sdpa``).

    gh #49: routes through Pion's server-side fused suffix-SDPA kernel
    (``ATTEND.PREFIX.QUERY_FUSED``) when the prefix is non-empty and mask is
    None or the "causal" sentinel. The legacy two-step host-side merge
    (``_suffix_sdpa_with_lse`` + ``_online_softmax_merge``) is kept as
    ``_legacy_pion_attention`` and used only when an explicit array mask
    forces it."""
    if not isinstance(cache, PionPrefixCache):
        return _orig_sdpa(queries, keys, values, cache, scale, mask, sinks)
    if sinks is not None:
        # Attention sinks aren't routed through the sidecar (sidecar's softmax
        # doesn't model them). Fall through to the original on these layers;
        # cache must therefore not be PionPrefixCache for sink-using models —
        # caller should have built a vanilla cache for them.
        raise NotImplementedError(
            "PionPrefixCache + attention sinks not supported. Use a vanilla "
            "KVCache for sink layers.")

    # queries: (B, H_q, M, D). keys/values: (B, H_kv, S, D) — already the
    # SUFFIX-only K/V returned by PionPrefixCache.update_and_fetch.
    B, Hq, M, D = queries.shape
    if B != 1:
        # mlx-lm always runs B=1 in practice — keep this loud if someone ever
        # batches at the model level so we know to revisit the wire path.
        raise NotImplementedError("PionPrefixCache assumes B=1; got B={}".format(B))

    # Empty prefix → no merge needed; let mx.fast handle suffix-only.
    if cache.prefix_len <= 0:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask)

    # gh #50 + gh #60 in-process fast lane. When the prefix K/V is resident
    # in MLX memory in *this* process, PionPrefixCache.update_and_fetch already
    # returned the full [prefix | suffix] K/V (sliced to fa_window on sliding
    # layers, full on dense layers). Call mx.fast.SDPA directly — same path
    # vanilla mlx-lm uses with a normal KVCache. No wire, no mx.eval barrier,
    # no numpy↔MLX copy, no merge. Routing this BEFORE the fused-eligibility
    # check is what makes the legacy fallback path safe when in-proc is active
    # — _legacy_pion_attention assumes keys/values are suffix-only (it adds the
    # prefix via attend_query + online merge), and would double-count prefix
    # if handed the full-K/V tensor that update_and_fetch now returns.
    if cache._mlx_prefix_K is not None:
        # gh #60 Phase 3: optional sparse selection. Only fires on M=1 (decode
        # path) — M>1 forwards over a warm prefix use a causal mask whose shape
        # depends on the dense prefix length, so reshaping the K/V under it
        # would break the mask. Decode-step queries see `mask=None` because
        # they attend to all keys, so sparse + mask=None is the easy case.
        sparse_mode = getattr(cache, "sparse_mode", None)
        if sparse_mode is not None and queries.shape[2] == 1 and mask is None:
            B_blk = int(sparse_mode.get("K_block", 64))
            K_top = int(sparse_mode.get("K_blocks", 8))
            # W11 / gh #9 follow-up: selector dispatch. "block_mean" is the
            # historical default (gh #60 Phase 3); "quest" is the Quest
            # upper-bound variant (W11 Phase 1) — drop-in same shape contract.
            selector = sparse_mode.get("selector", "block_mean")
            if selector == "quest":
                keys, values = _quest_topk_select(
                    queries, keys, values, cache.prefix_len, B_blk, K_top)
            else:
                keys, values = _block_mean_topk_select(
                    queries, keys, values, cache.prefix_len, B_blk, K_top)
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask)

    # Wire path (cross-process consumer): keys/values are suffix-only.

    # gh #63 wire-mode sparse routing: when M=1 (decode step) and the cache
    # has a sparse_mode configured, route to ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED.
    # Server picks top-K from resident K/V via block-mean, then merges with
    # caller-supplied suffix K/V in one kernel dispatch — proper online-softmax
    # over (sparse-prefix ∪ dense-suffix). Closes the "ignore suffix" caveat
    # that the v1 wire-mode sparse path had.
    sparse_mode = getattr(cache, "sparse_mode", None)
    if (sparse_mode is not None and M == 1 and
            (mask is None or (isinstance(mask, str) and mask == "causal"))):
        B_blk = int(sparse_mode.get("K_block", 64))
        K_top = int(sparse_mode.get("K_blocks", 8))
        Hkv_local = keys.shape[1]
        S_suf_local = keys.shape[2]                # suffix tokens so far

        q_mx = queries[0, :, 0, :]                 # (Hq, D)
        if q_mx.dtype != mx.float32: q_mx = q_mx.astype(mx.float32)
        k_mx = keys[0]                             # (Hkv, S_suf, D)
        v_mx = values[0]
        if k_mx.dtype != mx.float32: k_mx = k_mx.astype(mx.float32)
        if v_mx.dtype != mx.float32: v_mx = v_mx.astype(mx.float32)
        mx.eval(q_mx, k_mx, v_mx)
        Q_np     = np.array(q_mx, copy=False)
        K_suf_np = np.array(k_mx, copy=False)
        V_suf_np = np.array(v_mx, copy=False)

        out_np = cache.prompt_cache.attend_query_sparse_auto_fused(
            cache.namespace, cache.layer_idx,
            Q_np, B=B_blk, K_top=K_top,
            K_suf=K_suf_np, V_suf=V_suf_np,
            H_kv=Hkv_local,
            fa_window=cache.fa_window,
            # gh #371: the in-proc branch above honours `selector`; the wire
            # branch dropped it, so `selector: "quest"` silently ran block-mean.
            selector=sparse_mode.get("selector", "block_mean"),
        )
        out = mx.array(out_np).reshape(1, Hq, 1, D)
        return out

    # Custom array mask: the fused kernel applies "no-mask on prefix +
    # causal-on-suffix" by construction. Anything else falls back.
    fused_eligible = (mask is None) or (isinstance(mask, str) and mask == "causal")
    if _DISABLE_FUSED or not fused_eligible:
        return _legacy_pion_attention(queries, keys, values, cache, scale, mask)

    Hkv = keys.shape[1]
    if Hq % Hkv != 0:
        return _legacy_pion_attention(queries, keys, values, cache, scale, mask)
    rep = Hq // Hkv

    # head_map[h_q] = h_kv. Cached on the layer cache: it's a constant of
    # (Hq, Hkv) and recomputing every call is silly. uint8 is safe up to
    # H_kv=256 (every published GQA model is <= 64 kv heads).
    head_map = getattr(cache, "_head_map", None)
    if head_map is None or head_map.shape[0] != Hq:
        head_map = np.repeat(np.arange(Hkv, dtype=np.uint8), rep)
        cache._head_map = head_map

    # Phase 2A zero-copy interop (Codex review of gh #49 hot-path overhead).
    # Cast inside MLX (no-op when already fp32 — MLX short-circuits), force
    # ONE explicit eval barrier so the three lazy graphs sync in parallel,
    # then take a zero-copy numpy VIEW via the buffer protocol. Replaces
    # the prior `np.array(...).astype(np.float32)` pattern which under
    # NumPy 2.0 ran two host copies per array (np.array copy=True default,
    # astype copy=True default) and forced three serial implicit evals.
    q_mx = queries[0]
    k_mx = keys[0]
    v_mx = values[0]
    if q_mx.dtype != mx.float32: q_mx = q_mx.astype(mx.float32)
    if k_mx.dtype != mx.float32: k_mx = k_mx.astype(mx.float32)
    if v_mx.dtype != mx.float32: v_mx = v_mx.astype(mx.float32)
    # PION_NO_EXPLICIT_EVAL=1 skips the explicit eval barrier — np.array()
    # below would trigger eval implicitly. The explicit single barrier costs
    # nothing if the lazy graph is already realized; it serializes the three
    # arrays' eval into one parallel sync if they aren't. Keep on by default
    # (correctness identical) and disable via env var to A/B.
    if not os.environ.get("PION_NO_EXPLICIT_EVAL"):
        mx.eval(q_mx, k_mx, v_mx)
    Q_np     = np.array(q_mx, copy=False)
    K_suf_np = np.array(k_mx, copy=False)
    V_suf_np = np.array(v_mx, copy=False)
    # gh #60 Step 1: per-layer fa_window — sliding layers pass 512 (or
    # whatever the model config specifies), full layers pass None (= server
    # default, typically full attention). The server caps prefix attention
    # to the last `fa_window` tokens, matching vanilla mlx-lm's sliding mask.
    out_np = cache.prompt_cache.attend_query_fused(
        cache.namespace, cache.layer_idx,
        Q_np, K_suf_np, V_suf_np, head_map,
        fa_window=cache.fa_window,
    )
    return mx.array(out_np)[None, ...]


# ─────────────────────────────────────────────────────────────────────────────
# Install / uninstall
# ─────────────────────────────────────────────────────────────────────────────


from pion_vllm_mlx._compat import (  # noqa: E402  (gh #263 seam guard)
    PionMlxCompatError,
    check_mlx_lm_seam,
)

_orig_sdpa = None
_INSTALLED = False


def install_pion_attention_patch() -> None:
    """Replace mlx_lm.models.base.scaled_dot_product_attention with the
    Pion-aware version. Idempotent. Affects every model module that
    `from .base import scaled_dot_product_attention` AT IMPORT TIME — the
    monkey-patch reaches into each one's namespace.

    The seam is checked BEFORE anything is patched, so an
    incompatible mlx-lm raises `PionMlxCompatError` naming the observed
    signature and leaves mlx-lm untouched — rather than surfacing as a
    TypeError inside generation, after the patch is already installed."""
    global _orig_sdpa, _INSTALLED
    if _INSTALLED:
        return
    check_mlx_lm_seam()
    import mlx_lm.models.base as base_mod
    _orig_sdpa = base_mod.scaled_dot_product_attention
    base_mod.scaled_dot_product_attention = pion_scaled_dot_product_attention
    # Walk every already-imported model module and replace its reference too,
    # since `from .base import scaled_dot_product_attention` snapshot-bound it.
    import sys
    for name, mod in list(sys.modules.items()):
        if not name.startswith("mlx_lm.models."):
            continue
        if name.endswith(".base"):
            continue
        if hasattr(mod, "scaled_dot_product_attention") and \
                getattr(mod, "scaled_dot_product_attention") is _orig_sdpa:
            mod.scaled_dot_product_attention = pion_scaled_dot_product_attention
    _INSTALLED = True


def uninstall_pion_attention_patch() -> None:
    """Reverse install_pion_attention_patch()."""
    global _INSTALLED
    if not _INSTALLED or _orig_sdpa is None:
        return
    import mlx_lm.models.base as base_mod
    base_mod.scaled_dot_product_attention = _orig_sdpa
    import sys
    for name, mod in list(sys.modules.items()):
        if not name.startswith("mlx_lm.models."):
            continue
        if hasattr(mod, "scaled_dot_product_attention") and \
                getattr(mod, "scaled_dot_product_attention") is pion_scaled_dot_product_attention:
            mod.scaled_dot_product_attention = _orig_sdpa
    _INSTALLED = False


# ─────────────────────────────────────────────────────────────────────────────
# Cache-list builder
# ─────────────────────────────────────────────────────────────────────────────


def make_pion_prompt_cache(
    model,
    namespace: str,
    prompt_cache,
    prefix_len: int,
    sparse_full_layers: Optional[Dict[str, int]] = None,
) -> List[PionPrefixCache]:
    """Build a per-layer cache list pointing each layer at Pion.

    Mirrors mlx_lm.models.cache.make_prompt_cache(model) but returns
    PionPrefixCache instances. Pass the result as `cache=` to model() for
    forward passes after a successful PionPromptCache.get_or_prefill().

    Hybrid architectures (Gemma 4 / Mistral SWA / Qwen3.5) build a
    cache list that can be SHORTER than n_layers because KV-shared layers
    don't carry their own cache entry — the model routes them via
    `shared_kv` from earlier layers and pads the cache list with None
    internally. Probe `model.model.make_cache()` first so the length
    matches what mlx-lm actually expects; fall back to n_layers for
    flat-uniform models that don't expose make_cache().

    Pass `sparse_full_layers={"K_block": 64, "K_blocks": 8}`
    to enable block-mean top-K sparse selection on FULL-attention layers
    only. Sliding-attention layers are already capped at `sliding_window`
    (typically 512) tokens by design and sparsifying below that crosses out
    of the training distribution, so they stay dense. Flat models without
    `layer_types`: every layer is treated as full and gets sparse.
    """
    # The outer mlx-lm Model class exposes make_cache() for hybrid archs
    # (Gemma4 / Mistral SWA / Qwen3.5); it's the source of truth for the cache
    # list length the model expects. Fall back to len(layers) for flat models
    # where the wrapper isn't present.
    n_caches = None
    if hasattr(model, "make_cache"):
        try:
            n_caches = len(model.make_cache())
        except Exception:
            n_caches = None
    if n_caches is None:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            n_caches = len(model.model.layers)
        else:
            n_caches = len(model.layers)
    # gh #60 Step 1: per-layer sliding-window window.
    # Hybrid archs expose `args.layer_types` (per-layer "sliding_attention" /
    # "full_attention") and `args.sliding_window`. Cache slot `i` corresponds
    # to layer `i` (KV-shared layers route via `previous_kvs` and don't carry
    # their own slot, so the cache list spans only the first
    # `n_layers - n_kv_shared` layers — mlx-lm convention). For flat models
    # without `layer_types`, every cache slot is full attention (fa_window=None).
    args = getattr(model, "args", None)
    layer_types = getattr(args, "layer_types", None) or [] if args else []
    sliding_window = getattr(args, "sliding_window", 0) or 0 if args else 0
    def _window_for(i: int):
        if i >= len(layer_types):
            return None
        return sliding_window if (layer_types[i] == "sliding_attention" and sliding_window > 0) else None
    def _sparse_for(i: int):
        if sparse_full_layers is None:
            return None
        # No layer_types ⇒ flat model ⇒ every layer is full ⇒ apply sparse.
        if i >= len(layer_types):
            return sparse_full_layers
        # Hybrid: only the explicitly-full layers get sparse.
        return sparse_full_layers if layer_types[i] == "full_attention" else None
    return [
        PionPrefixCache(layer_idx=i, namespace=namespace,
                        prompt_cache=prompt_cache, prefix_len=prefix_len,
                        fa_window=_window_for(i),
                        sparse_mode=_sparse_for(i))
        for i in range(n_caches)
    ]
