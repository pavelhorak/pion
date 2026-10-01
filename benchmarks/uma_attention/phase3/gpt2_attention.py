#!/usr/bin/env python3
"""
UMMA Phase 3 — Real GPT-2 Attention Benchmark

Extracts Q/K/V from GPT-2 124M and runs sparse attention with real
(correlated) key vectors. Validates that random-data results from
Phases 0-2 hold for real transformer weights.

Key question: Do correlated keys (mean pairwise cosine ~0.67) change
the top-k selection quality or timing behavior?

Usage:
    python3 gpt2_attention.py
    python3 gpt2_attention.py --context 16384 --markdown
    python3 gpt2_attention.py --context 4096 --layers 0,5,11
"""

import argparse
import time
import sys
import numpy as np

# -----------------------------------------------------------------------
# GPT-2 weight extraction
# -----------------------------------------------------------------------
def extract_qkv(layer_idx=0, context_len=4096, seed=42):
    """Extract Q, K, V from GPT-2 124M for a given layer.

    Returns per-head Q[1,d], K[N,d], V[N,d] and metadata.
    """
    import torch
    from transformers import GPT2Model, GPT2Tokenizer

    print(f"  Loading GPT-2 124M...", end="", flush=True)
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    print(" done.")

    d_model = 768
    n_heads = 12
    d_head = d_model // n_heads  # 64

    # GPT-2 max position embeddings = 1024. For larger contexts, we run
    # the model at 1024 and tile the K/V to simulate longer sequences.
    max_pos = 1024
    run_len = min(context_len, max_pos)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    text = "The quick brown fox jumps over the lazy dog. " * (run_len // 8 + 1)
    tokens = tokenizer.encode(text, max_length=run_len, truncation=True)
    if len(tokens) < run_len:
        tokens = (tokens * (run_len // len(tokens) + 1))[:run_len]
    input_ids = torch.tensor([tokens[:run_len]])

    print(f"  Running forward pass (context={context_len})...", end="", flush=True)
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
    print(" done.")

    # Extract the attention layer's Q, K, V projections
    layer = model.h[layer_idx].attn
    hidden = outputs.hidden_states[layer_idx]  # [1, N, 768]

    with torch.no_grad():
        # GPT-2 uses Conv1D for QKV projection
        qkv = layer.c_attn(hidden)  # [1, N, 3*768]
        q, k, v = qkv.split(d_model, dim=2)

        # Reshape to per-head: [1, run_len, 12, 64] → [12, run_len, 64]
        q = q.view(1, run_len, n_heads, d_head).squeeze(0).permute(1, 0, 2)
        k = k.view(1, run_len, n_heads, d_head).squeeze(0).permute(1, 0, 2)
        v = v.view(1, run_len, n_heads, d_head).squeeze(0).permute(1, 0, 2)

    # Use last token as query (autoregressive decode scenario)
    Q_np = q[:, -1:, :].numpy().astype(np.float32)   # [12, 1, 64]
    K_np = k.numpy().astype(np.float32)                # [12, run_len, 64]
    V_np = v.numpy().astype(np.float32)                # [12, run_len, 64]

    # Tile K/V to reach requested context_len (with small random perturbation to avoid trivial repeats)
    if context_len > run_len:
        reps = (context_len + run_len - 1) // run_len
        np.random.seed(seed)
        K_tiles = [K_np + np.random.randn(*K_np.shape).astype(np.float32) * 0.01 for _ in range(reps)]
        V_tiles = [V_np + np.random.randn(*V_np.shape).astype(np.float32) * 0.01 for _ in range(reps)]
        K_np = np.concatenate(K_tiles, axis=1)[:, :context_len, :]
        V_np = np.concatenate(V_tiles, axis=1)[:, :context_len, :]

    Q = Q_np
    K = K_np
    V = V_np

    # Key correlation stats
    head0_K = K[0]  # [N, 64]
    norms = np.linalg.norm(head0_K, axis=1, keepdims=True)
    normed = head0_K / (norms + 1e-10)
    # Sample 1000 pairs for mean cosine
    np.random.seed(seed)
    idx_a = np.random.randint(0, context_len, 1000)
    idx_b = np.random.randint(0, context_len, 1000)
    cosines = np.sum(normed[idx_a] * normed[idx_b], axis=1)
    mean_cos = float(np.mean(np.abs(cosines)))

    meta = {
        "model": "gpt2-124M",
        "layer": layer_idx,
        "n_heads": n_heads,
        "d_head": d_head,
        "context_len": context_len,
        "mean_pairwise_cosine": mean_cos,
    }
    return Q, K, V, meta

# -----------------------------------------------------------------------
# Attention implementations
# -----------------------------------------------------------------------
def full_attention(Q, K, V):
    """Full dense attention per head. Q:[H,1,d] K:[H,N,d] V:[H,N,d] → [H,1,d]"""
    d = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d)  # [H, 1, N]
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.matmul(weights, V)  # [H, 1, d]


def sparse_attention_cpu(Q, K, V, top_k=32):
    """CPU sparse attention: Q@K^T → top-k → softmax → V[top_k]"""
    H, _, d = Q.shape
    N = K.shape[1]
    scale = 1.0 / np.sqrt(d)
    outputs = np.zeros((H, 1, d), dtype=np.float32)

    for h in range(H):
        scores = (Q[h] @ K[h].T) * scale  # [1, N]
        topk_idx = np.argpartition(scores[0], -top_k)[-top_k:]
        topk_scores = scores[0, topk_idx]
        topk_scores -= topk_scores.max()
        weights = np.exp(topk_scores)
        weights /= weights.sum()
        V_topk = V[h, topk_idx]  # [k, d]
        outputs[h, 0] = weights @ V_topk

    return outputs


def sparse_attention_gpu(Q_mx, K_mx, V_mx, top_k=32):
    """MLX GPU batched sparse attention"""
    import mlx.core as mx
    H = Q_mx.shape[0]
    d = Q_mx.shape[-1]
    scale = 1.0 / np.sqrt(d)

    scores = mx.matmul(Q_mx, mx.transpose(K_mx, (0, 2, 1))) * scale  # [H, 1, N]
    topk_idx = mx.argpartition(scores[:, 0, :], kth=-top_k, axis=-1)[:, -top_k:]
    topk_scores = mx.take_along_axis(scores[:, 0, :], topk_idx, axis=-1)
    topk_scores = topk_scores - mx.max(topk_scores, axis=-1, keepdims=True)
    weights = mx.exp(topk_scores)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)

    # Gather V
    V_gathered = mx.take_along_axis(
        V_mx,
        topk_idx[:, :, None] * mx.ones((1, 1, d), dtype=mx.int32),
        axis=1
    )
    output = mx.matmul(weights[:, None, :], V_gathered)
    return output[:, 0, :]


def gpu_dense_attention(Q_mx, K_mx, V_mx):
    """MLX full dense attention"""
    import mlx.core as mx
    d = Q_mx.shape[-1]
    scores = mx.matmul(Q_mx, mx.transpose(K_mx, (0, 2, 1))) / np.sqrt(d)
    scores = scores - mx.max(scores, axis=-1, keepdims=True)
    weights = mx.exp(scores)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return mx.matmul(weights, V_mx)

# -----------------------------------------------------------------------
# Benchmark harness
# -----------------------------------------------------------------------
def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def bench(fn, args, warmup=5, iters=20, sync_mlx=False):
    import mlx.core as mx
    for _ in range(warmup):
        r = fn(*args)
        if sync_mlx: mx.eval(r)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        r = fn(*args)
        if sync_mlx: mx.eval(r)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)

    arr = np.array(times)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "p95": float(np.percentile(arr, 95)),
        "std": float(arr.std()),
    }

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="UMMA Phase 3: GPT-2 Attention")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--layers", type=str, default="0")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    top_k = args.topk

    try:
        import mlx.core as mx
        HAS_MLX = True
        print(f"  MLX: {mx.__version__}, Metal: {mx.metal.is_available()}")
    except ImportError:
        HAS_MLX = False
        print("  WARNING: MLX not available, GPU benchmarks skipped")

    print(f"\n  UMMA Phase 3 — GPT-2 Real Attention Benchmark")
    print(f"  context={args.context}, top_k={top_k}, layers={layers}\n")

    for layer_idx in layers:
        print(f"  --- Layer {layer_idx} ---")
        Q, K, V, meta = extract_qkv(layer_idx, args.context)
        H, N, d = K.shape

        print(f"  Shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
        print(f"  Mean pairwise |cosine| (head 0): {meta['mean_pairwise_cosine']:.3f}")

        # Reference: full attention
        ref = full_attention(Q, K, V)

        # --- CPU sparse ---
        cpu_stats = bench(sparse_attention_cpu, (Q, K, V, top_k), warmup=3, iters=10)
        cpu_out = sparse_attention_cpu(Q, K, V, top_k)
        cpu_cos = cosine_sim(cpu_out, ref)

        # --- CPU dense ---
        cpu_dense_stats = bench(full_attention, (Q, K, V), warmup=3, iters=10)

        results = [
            ("CPU-sparse", cpu_stats, cpu_cos),
            ("CPU-dense", cpu_dense_stats, 1.0),
        ]

        if HAS_MLX:
            Q_mx = mx.array(Q)
            K_mx = mx.array(K)
            V_mx = mx.array(V)

            # --- GPU sparse ---
            gpu_sp_stats = bench(sparse_attention_gpu, (Q_mx, K_mx, V_mx, top_k),
                                 warmup=5, iters=20, sync_mlx=True)
            gpu_sp_out = sparse_attention_gpu(Q_mx, K_mx, V_mx, top_k)
            mx.eval(gpu_sp_out)
            gpu_sp_cos = cosine_sim(np.array(gpu_sp_out), ref[:, 0, :])

            # --- GPU dense ---
            gpu_dense_stats = bench(gpu_dense_attention, (Q_mx, K_mx, V_mx),
                                    warmup=5, iters=20, sync_mlx=True)
            gpu_dense_out = gpu_dense_attention(Q_mx, K_mx, V_mx)
            mx.eval(gpu_dense_out)
            gpu_dense_cos = cosine_sim(np.array(gpu_dense_out), ref)

            results.extend([
                ("GPU-sparse(MLX)", gpu_sp_stats, gpu_sp_cos),
                ("GPU-dense(MLX)", gpu_dense_stats, gpu_dense_cos),
            ])

        if args.markdown:
            print(f"\n| Strategy | Median µs | Min µs | P95 µs | Cosine vs full | Speedup vs CPU-sparse |")
            print(f"|---|---:|---:|---:|---:|---:|")
            for name, stats, cos in results:
                speedup = cpu_stats["median"] / stats["median"] if stats["median"] > 0 else 0
                print(f"| {name} | {stats['median']:.0f} | {stats['min']:.0f} | {stats['p95']:.0f} | {cos:.4f} | {speedup:.2f}x |")
        else:
            print(f"\n  {'Strategy':<20s} {'Median µs':>10s} {'Min µs':>10s} {'P95 µs':>10s} {'Cosine':>8s} {'vs CPU':>8s}")
            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
            for name, stats, cos in results:
                speedup = cpu_stats["median"] / stats["median"] if stats["median"] > 0 else 0
                print(f"  {name:<20s} {stats['median']:>10.0f} {stats['min']:>10.0f} {stats['p95']:>10.0f} {cos:>8.4f} {speedup:>7.2f}x")

        # Top-k quality analysis: how many of the true top-32 does sparse attention find?
        scale = 1.0 / np.sqrt(d)
        for h in [0, 5, 11]:
            if h >= H: continue
            scores_full = (Q[h] @ K[h].T) * scale  # [1, N]
            true_topk = set(np.argsort(scores_full[0])[-top_k:])
            found_topk = set(np.argpartition(scores_full[0], -top_k)[-top_k:])
            overlap = len(true_topk & found_topk) / top_k
            score_range = float(scores_full.max() - scores_full.min())
            top_margin = float(scores_full[0, sorted(true_topk)[-1]] - scores_full[0, sorted(true_topk)[0]])
            print(f"\n  Head {h}: top-{top_k} overlap={overlap:.0%}, score range={score_range:.3f}, "
                  f"top-k margin={top_margin:.4f}")

        print()

    print("  Phase 3 complete.")


if __name__ == "__main__":
    main()
