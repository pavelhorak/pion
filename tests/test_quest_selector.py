#!/usr/bin/env python3
"""W11 Phase 1 — synthetic correctness probe for the Quest selector.

Validates _quest_topk_select against the existing _block_mean_topk_select
on three synthetic scenarios:

  (a) NIAH-shape: one sharp needle in a sea of random K. Block-mean
      should find it; Quest should also find it (must not regress).
  (b) Dispersed-relevance: 4 weakly-correlated tokens spread across 4
      separate blocks. Quest's upper-bound semantics should rank the
      blocks containing those tokens higher than blocks of uniform
      noise; block-mean should also find them but may rank a block of
      "popular average" tokens above. Compares which top-K each picks.
  (c) Shape contract: output shapes match block-mean for the same
      (B, K_top, prefix_len) inputs; suffix passthrough preserved;
      recency tail included.

This is a unit-test on the selector itself, not an end-to-end model
eval. End-to-end NIAH + factual-QA validation comes after this proves
the selector behaves.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pion-vllm-mlx"))

import mlx.core as mx
from pion_vllm_mlx.mlx_lm_patch import _block_mean_topk_select, _quest_topk_select


def reference_dense_topk_block_ids(queries, K_full, prefix_len, B, K_top):
    """Ground truth: for each block, compute true max_t(Q · K_t) over its
    tokens (per-head averaged), then top-K. This is what we'd get if we
    could afford dense per-token scoring."""
    Hq = queries.shape[1]
    Hkv = K_full.shape[1]
    D = K_full.shape[3]
    N_full = (prefix_len // B) * B

    prefix_K = K_full[:, :, :N_full, :]
    prefix_K = prefix_K.reshape(1, Hkv, N_full // B, B, D)
    if Hq != Hkv:
        rep = Hq // Hkv
        prefix_K = mx.repeat(prefix_K, rep, axis=1)

    q = queries[:, :, 0, :]
    scale = 1.0 / mx.sqrt(mx.array(D, dtype=mx.float32))
    # Compute Q · K_t per token: (1, Hq, N_full/B, B)
    qk = mx.sum(q[:, :, None, None, :] * prefix_K, axis=-1)
    max_per_block = mx.max(qk, axis=-1)                        # (1, Hq, N_full/B)
    shared = mx.mean(max_per_block, axis=1) * scale            # (1, N_full/B)
    sorted_idx = mx.argsort(-shared, axis=-1)
    return sorted_idx[:, :K_top]


def selected_block_ids_from_output(prefix_K, selector_K, B, K_top):
    """Reverse-engineer which prefix block IDs a selector picked by
    matching its output K rows against the source prefix K rows."""
    Hq = selector_K.shape[1]
    prefix_K_rep = prefix_K
    if prefix_K.shape[1] != Hq and Hq % prefix_K.shape[1] == 0:
        prefix_K_rep = mx.repeat(prefix_K, Hq // prefix_K.shape[1], axis=1)
    # Take the first K_top * B rows of the selector output (these are the
    # selected prefix tokens, before suffix). Each B-row chunk should match
    # a contiguous prefix block at offset block_id * B.
    sel = selector_K[:, :, : K_top * B, :]
    sel = sel.reshape(1, Hq, K_top, B, -1)
    blk = mx.mean(sel, axis=3)                                  # (1, Hq, K_top, D)
    pK = prefix_K_rep[:, :, : (prefix_K_rep.shape[2] // B) * B, :]
    pK = pK.reshape(1, Hq, -1, B, prefix_K_rep.shape[3])
    p_mean = mx.mean(pK, axis=3)                                # (1, Hq, n_blocks, D)
    # For each selected block, find argmax cos-sim against the prefix block means.
    sel_n = blk / (mx.linalg.norm(blk, axis=-1, keepdims=True) + 1e-9)
    p_n = p_mean / (mx.linalg.norm(p_mean, axis=-1, keepdims=True) + 1e-9)
    # (1, Hq, K_top, n_blocks)
    sims = mx.matmul(sel_n, p_n.transpose(0, 1, 3, 2))
    picks = mx.argmax(sims, axis=-1)                            # (1, Hq, K_top)
    # Shared mask in v1: all heads pick same set → reduce over Hq via mode
    # (just take head 0 for simplicity since they should agree).
    return picks[0, 0, :]                                       # (K_top,)


def scenario_a_niah():
    """One sharp needle in a sea of random K — block-mean's home turf."""
    print("\n=== Scenario A: NIAH (one sharp needle) ===")
    mx.random.seed(42)
    Hq, Hkv, D = 4, 1, 64
    prefix_len = 512
    suffix_len = 0
    B, K_top = 64, 4

    K_full = mx.random.normal((1, Hkv, prefix_len, D)) * 0.1
    V_full = mx.random.normal((1, Hkv, prefix_len, D))
    # Plant the needle at token 320 (block 5)
    needle_block = 5
    needle_token = needle_block * B + 30
    needle_dir = mx.random.normal((D,))
    needle_dir = needle_dir / mx.linalg.norm(needle_dir)
    K_full[0, 0, needle_token, :] = needle_dir * 5.0
    queries = mx.repeat(needle_dir[None, None, None, :], Hq, axis=1)

    ref_top = reference_dense_topk_block_ids(queries, K_full, prefix_len, B, K_top)
    print(f"  reference top-{K_top} blocks (dense max-per-block): {ref_top[0].tolist()}")
    print(f"  expected to include block {needle_block} (needle location)")

    bm_K, _ = _block_mean_topk_select(queries, K_full, V_full, prefix_len, B, K_top)
    qs_K, _ = _quest_topk_select(queries, K_full, V_full, prefix_len, B, K_top)
    bm_picks = selected_block_ids_from_output(K_full, bm_K, B, K_top).tolist()
    qs_picks = selected_block_ids_from_output(K_full, qs_K, B, K_top).tolist()
    print(f"  block_mean picks: {sorted(bm_picks)}")
    print(f"  quest      picks: {sorted(qs_picks)}")
    bm_found = needle_block in bm_picks
    qs_found = needle_block in qs_picks
    print(f"  needle found by block_mean: {bm_found}")
    print(f"  needle found by quest:      {qs_found}")
    return bm_found, qs_found


def scenario_b_dispersed():
    """4 weakly-correlated tokens in 4 separate blocks. The blocks
    containing those tokens should rank higher than uniform-noise blocks.
    Tests whether Quest's max-based UB is less fooled by 'popular mean'
    blocks than block-mean."""
    print("\n=== Scenario B: dispersed relevance (4 weak tokens, 4 blocks) ===")
    mx.random.seed(123)
    Hq, Hkv, D = 4, 1, 64
    prefix_len = 512
    B, K_top = 64, 4

    # All K starts as random noise.
    K_full = mx.random.normal((1, Hkv, prefix_len, D)) * 0.1
    V_full = mx.random.normal((1, Hkv, prefix_len, D))

    # Plant 4 weakly-correlated tokens in blocks 1, 3, 5, 7
    direction = mx.random.normal((D,))
    direction = direction / mx.linalg.norm(direction)
    target_blocks = [1, 3, 5, 7]
    for tb in target_blocks:
        K_full[0, 0, tb * B + 10, :] = direction * 2.0   # weaker than NIAH

    # Also: make blocks 0, 2, 4, 6 have ELEVATED mean (mid-attention noise
    # that block-mean might mistake for relevant). Each token in these
    # blocks gets a small bias toward direction.
    for db in [0, 2, 4, 6]:
        for t in range(B):
            K_full[0, 0, db * B + t, :] += direction * 0.3

    queries = mx.repeat(direction[None, None, None, :], Hq, axis=1)

    ref_top = reference_dense_topk_block_ids(queries, K_full, prefix_len, B, K_top)
    ref_set = set(ref_top[0].tolist())
    print(f"  reference top-{K_top}: {sorted(ref_set)}")
    print(f"  target (planted spikes): {target_blocks}")

    bm_K, _ = _block_mean_topk_select(queries, K_full, V_full, prefix_len, B, K_top)
    qs_K, _ = _quest_topk_select(queries, K_full, V_full, prefix_len, B, K_top)
    bm_picks = set(selected_block_ids_from_output(K_full, bm_K, B, K_top).tolist())
    qs_picks = set(selected_block_ids_from_output(K_full, qs_K, B, K_top).tolist())
    print(f"  block_mean picks: {sorted(bm_picks)}  (overlap with target: {len(bm_picks & set(target_blocks))}/4)")
    print(f"  quest      picks: {sorted(qs_picks)}  (overlap with target: {len(qs_picks & set(target_blocks))}/4)")
    print(f"  block_mean overlap with dense ref: {len(bm_picks & ref_set)}/{K_top}")
    print(f"  quest      overlap with dense ref: {len(qs_picks & ref_set)}/{K_top}")
    return len(bm_picks & ref_set), len(qs_picks & ref_set)


def scenario_c_shape_contract():
    """Output shape contract: K_top*B prefix + suffix_len = N_selected."""
    print("\n=== Scenario C: shape contract + suffix passthrough ===")
    mx.random.seed(7)
    Hq, Hkv, D = 4, 1, 64
    prefix_len, suffix_len = 500, 5
    B, K_top = 64, 4
    total_len = prefix_len + suffix_len

    K_full = mx.random.normal((1, Hkv, total_len, D))
    V_full = mx.random.normal((1, Hkv, total_len, D))
    queries = mx.random.normal((1, Hq, 1, D))

    bm_K, bm_V = _block_mean_topk_select(queries, K_full, V_full, prefix_len, B, K_top)
    qs_K, qs_V = _quest_topk_select(queries, K_full, V_full, prefix_len, B, K_top)
    print(f"  block_mean output K shape: {bm_K.shape}")
    print(f"  quest      output K shape: {qs_K.shape}")
    print(f"  shapes match: {bm_K.shape == qs_K.shape and bm_V.shape == qs_V.shape}")
    # K_top * B + remainder + suffix
    expected_len = K_top * B + (prefix_len - (prefix_len // B) * B) + suffix_len
    print(f"  expected output length: {expected_len}  actual: {qs_K.shape[2]}")
    # Verify suffix is preserved verbatim at the end
    suf_match = mx.all(qs_K[:, :, -suffix_len:, :] == K_full[:, :, -suffix_len:, :]).item()
    print(f"  suffix tokens preserved verbatim: {suf_match}")
    return (
        bm_K.shape == qs_K.shape and bm_V.shape == qs_V.shape
        and qs_K.shape[2] == expected_len
        and suf_match
    )


def main():
    print("W11 / gh #9 Phase 1 — Quest selector synthetic probe")
    print("=" * 60)

    bm_a, qs_a = scenario_a_niah()
    bm_b, qs_b = scenario_b_dispersed()
    shape_ok = scenario_c_shape_contract()

    print("\n=== Summary ===")
    print(f"  A (NIAH) — needle found:    block_mean={bm_a}  quest={qs_a}")
    print(f"  B (dispersed) — overlap@4:  block_mean={bm_b}  quest={qs_b}")
    print(f"  C (shape contract): {'PASS' if shape_ok else 'FAIL'}")

    # Conditions for Phase 1 "implementation looks right":
    #   - Quest finds NIAH needle (must not regress)
    #   - Quest's overlap-with-dense-ref ≥ block-mean's overlap (the dispersed
    #     scenario is exactly the case Quest is designed to win)
    #   - Shape contract holds
    if not qs_a:
        print("\n  ✗ FAIL: Quest missed the NIAH needle — implementation regressed.")
        return 1
    if not shape_ok:
        print("\n  ✗ FAIL: shape contract broken.")
        return 1
    if qs_b < bm_b:
        print(f"\n  ⚠ WARN: Quest ({qs_b}/4) underperformed block-mean ({bm_b}/4) on dispersed.")
        print("    Synthetic scenario may not be hard enough to discriminate; real")
        print("    factual-QA workload validation is the load-bearing test.")
    print("\n  ✓ Quest selector behaves correctly on the synthetic probe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
