#!/usr/bin/env python3
"""gh #82 follow-up — deterministic routing gate for Gemma 4 12B.

Locks in `make_pion_prompt_cache`'s per-layer routing for Gemma-4-12B-it-4bit
without downloading the 6.7 GB model. The HF config.json values are
hard-coded here so a future mlx-lm / patch refactor that breaks the routing
for hybrid 12B fails this test instead of silently corrupting long-context
generation.

Reference: `mlx-community/gemma-4-12B-it-4bit` config.json (2026-06-03 release):
- num_hidden_layers = 48
- layer_types: 5× sliding_attention then 1× full_attention, repeating
  → 40 sliding + 8 full
- sliding_window = 1024
- sliding head_dim = 256, full head_dim (= global_head_dim) = 512
- max_position_embeddings = 262144 (256K)

The patch is model-agnostic and reads these straight from `model.args`; this
test stubs `model.args` with the production values and asserts every
PionPrefixCache slot was wired correctly. No mlx-lm required — pure Python.

Usage:
    python pion-vllm-mlx/tests/test_gemma4_12b_layer_routing.py

Exit code 0 = all asserts passed, 1 = failure.
"""
from __future__ import annotations

import sys
import types

sys.path.insert(0, "pion-vllm-mlx")

from pion_vllm_mlx.mlx_lm_patch import make_pion_prompt_cache


# Verbatim from mlx-community/gemma-4-12B-it-4bit/config.json text_config.layer_types.
# 48 entries, pattern = (sliding × 5, full × 1) repeating eight times.
GEMMA4_12B_LAYER_TYPES = (
    ["sliding_attention"] * 5 + ["full_attention"]
) * 8
assert len(GEMMA4_12B_LAYER_TYPES) == 48, "config drift — layer count changed"
assert GEMMA4_12B_LAYER_TYPES.count("full_attention") == 8
assert GEMMA4_12B_LAYER_TYPES.count("sliding_attention") == 40
GEMMA4_12B_SLIDING_WINDOW = 1024


def _make_stub_model(layer_types, sliding_window, n_caches=None):
    """Return a minimal stand-in for an mlx-lm hybrid model.

    `make_pion_prompt_cache` accesses:
      - `model.args.layer_types` (list[str])
      - `model.args.sliding_window` (int)
      - `model.make_cache()` (preferred, for hybrid cache-list length) OR
        `model.model.layers` / `model.layers` (fallback).
    """
    args = types.SimpleNamespace(
        layer_types=layer_types,
        sliding_window=sliding_window,
    )
    model = types.SimpleNamespace(args=args)
    n_caches = n_caches if n_caches is not None else len(layer_types)
    model.make_cache = lambda: [None] * n_caches
    return model


def test_dense_layers_skip_sparse_when_no_config():
    """With `sparse_full_layers=None`, every slot gets sparse_mode=None and the
    only differentiation is fa_window."""
    model = _make_stub_model(GEMMA4_12B_LAYER_TYPES, GEMMA4_12B_SLIDING_WINDOW)
    caches = make_pion_prompt_cache(
        model, namespace="t1", prompt_cache=None, prefix_len=1024,
        sparse_full_layers=None,
    )
    assert len(caches) == 48, f"cache slot count: got {len(caches)}, want 48"
    for i, c in enumerate(caches):
        want = GEMMA4_12B_SLIDING_WINDOW if GEMMA4_12B_LAYER_TYPES[i] == "sliding_attention" else None
        assert c.fa_window == want, (
            f"layer {i} ({GEMMA4_12B_LAYER_TYPES[i]}): "
            f"fa_window got {c.fa_window}, want {want}"
        )
        assert c.sparse_mode is None, f"layer {i}: sparse_mode should be None when not configured"
        assert c.layer_idx == i
        assert c.prefix_len == 1024
        assert c.namespace == "t1"


def test_sparse_full_layers_only():
    """gh #60 invariant: sparse-mask applies to full_attention layers only;
    sliding layers stay dense (capped at sliding_window) and must NOT get
    sparse_mode set — sparsifying below window crosses out of the training
    distribution and breaks logits."""
    sparse_cfg = {"K_block": 64, "K_blocks": 8}
    model = _make_stub_model(GEMMA4_12B_LAYER_TYPES, GEMMA4_12B_SLIDING_WINDOW)
    caches = make_pion_prompt_cache(
        model, namespace="t2", prompt_cache=None, prefix_len=65536,
        sparse_full_layers=sparse_cfg,
    )
    n_sparse = 0
    n_sliding_with_window = 0
    for i, c in enumerate(caches):
        if GEMMA4_12B_LAYER_TYPES[i] == "full_attention":
            assert c.sparse_mode == sparse_cfg, (
                f"layer {i} (full): sparse_mode got {c.sparse_mode}, want {sparse_cfg}"
            )
            assert c.fa_window is None, (
                f"layer {i} (full): fa_window must be None, got {c.fa_window}"
            )
            n_sparse += 1
        else:  # sliding_attention
            assert c.sparse_mode is None, (
                f"layer {i} (sliding): sparse_mode must be None, got {c.sparse_mode} "
                f"— sparsifying sliding layers below the 1024 window breaks distribution"
            )
            assert c.fa_window == GEMMA4_12B_SLIDING_WINDOW, (
                f"layer {i} (sliding): fa_window got {c.fa_window}, want 1024"
            )
            n_sliding_with_window += 1
    assert n_sparse == 8, f"want 8 sparse full layers, got {n_sparse}"
    assert n_sliding_with_window == 40, f"want 40 windowed sliding layers, got {n_sliding_with_window}"


def test_quest_selector_passes_through():
    """gh #9 Phase 1+2: when caller adds `selector` to sparse_full_layers, the
    same dict reaches every full-attention slot verbatim — the patch must not
    drop or rewrite extra keys."""
    sparse_cfg = {"K_block": 64, "K_blocks": 8, "selector": "quest"}
    model = _make_stub_model(GEMMA4_12B_LAYER_TYPES, GEMMA4_12B_SLIDING_WINDOW)
    caches = make_pion_prompt_cache(
        model, namespace="t3", prompt_cache=None, prefix_len=128 * 1024,
        sparse_full_layers=sparse_cfg,
    )
    for i, c in enumerate(caches):
        if GEMMA4_12B_LAYER_TYPES[i] == "full_attention":
            assert c.sparse_mode is sparse_cfg or c.sparse_mode == sparse_cfg, (
                f"layer {i} (full): selector key lost — got {c.sparse_mode}"
            )


def test_make_cache_drives_cache_list_length():
    """Hybrid archs with KV-shared layers expose a make_cache() shorter than
    n_layers (mlx-lm pads via shared_kv routing). The patch must trust
    make_cache() over len(layer_types)."""
    model = _make_stub_model(
        GEMMA4_12B_LAYER_TYPES, GEMMA4_12B_SLIDING_WINDOW, n_caches=45,
    )
    caches = make_pion_prompt_cache(
        model, namespace="t4", prompt_cache=None, prefix_len=1024,
        sparse_full_layers={"K_block": 32, "K_blocks": 4},
    )
    assert len(caches) == 45, (
        f"cache slot count must match model.make_cache(): got {len(caches)}, want 45"
    )


def main() -> int:
    tests = [
        ("dense layers skip sparse when no config", test_dense_layers_skip_sparse_when_no_config),
        ("sparse_full_layers only", test_sparse_full_layers_only),
        ("Quest selector passes through", test_quest_selector_passes_through),
        ("make_cache() drives cache list length", test_make_cache_drives_cache_list_length),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\nGemma 4 12B routing gate: {len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
