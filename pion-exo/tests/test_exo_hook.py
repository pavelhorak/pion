#!/usr/bin/env python3
"""Test PionAttentionHook against a running Pion server.

Requires:
    ./pion-server --kvcache --metal-attention -w 1

Tests:
    1. Health check (RESP + ATTEND.PREFIX.QUERY availability)
    2. Prefill v_offload → V.STOREBATCH
    3. Decode v_offload → V.FETCH + CPU attention
    4. Prefill+Decode gpu_attention → ATTEND.PREFIX.STORE/QUERY
       (via PionPromptCache(stage2=True))
    5. Session cleanup
"""

import os
import sys

import numpy as np

# Add pion-exo and (sibling) pion-vllm-mlx paths for in-tree development.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "pion-vllm-mlx"))
from pion_exo import PionAttentionHook


def cpu_reference_attention(Q, K, V, top_k):
    """Reference CPU sparse attention for v_offload correctness check."""
    if Q.ndim == 3:
        Q = Q[:, 0, :]
    H, D = Q.shape
    scale = D ** -0.5
    output = np.zeros((H, D), dtype=np.float32)
    for h in range(H):
        K_h = K[h] if K.ndim == 3 else K
        scores = (Q[h] @ K_h.T) * scale
        actual_k = min(top_k, len(scores))
        topk_idx = np.argpartition(scores, -actual_k)[-actual_k:]
        topk_scores = scores[topk_idx]
        topk_scores -= topk_scores.max()
        weights = np.exp(topk_scores)
        weights /= weights.sum()
        output[h] = weights @ V[h, topk_idx] if V.ndim == 3 else weights @ V[topk_idx]
    return output


def cpu_dense_attention(Q, K, V):
    """Reference dense softmax attention for gpu_attention correctness check."""
    if Q.ndim == 3:
        Q = Q[:, 0, :]
    H, D = Q.shape
    scale = D ** -0.5
    output = np.zeros((H, D), dtype=np.float32)
    for h in range(H):
        scores = (Q[h] @ K[h].T) * scale
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        output[h] = weights @ V[h]
    return output


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    pion_host = os.environ.get("PION_HOST", "127.0.0.1")
    pion_port = int(os.environ.get("PION_PORT", "1974"))

    passed = 0
    failed = 0

    # --- Test 1: Health check ---
    print("[1] Health check...", end=" ")
    hook = PionAttentionHook(
        pion_host=pion_host, pion_port=pion_port, mode="v_offload"
    )
    health = hook.health_check()
    if health["pion_resp"]:
        metal_str = "Metal=" + ("OK" if health["metal_attention"] else "N/A")
        print(f"OK (RESP=OK, {metal_str})")
        passed += 1
    else:
        print(f"FAIL (Pion not reachable at {pion_host}:{pion_port})")
        print("  Start Pion: ./pion-server --kvcache --metal-attention -w 1")
        failed += 1
        hook.close()
        print(f"\nResults: {passed} passed, {failed} failed")
        return 1

    # Test parameters. H=1 for v_offload (the offload path stores H*N tokens
    # flat at one offset and fetches by head-relative IDs — meaningful only
    # when H=1; multi-head v_offload requires per-head session_ids and is
    # exo's responsibility to plumb). gpu_attention re-uses these K/V at
    # the same H for a fair Stage-2 comparison.
    H, N, D, top_k = 1, 256, 128, 16
    np.random.seed(42)
    K = np.random.randn(H, N, D).astype(np.float32) * 0.1
    V = np.random.randn(H, N, D).astype(np.float32)
    Q = np.random.randn(H, D).astype(np.float32) * 0.1
    session_id = f"exo_test_{int(np.random.randint(100000))}"

    # --- Test 2: Prefill v_offload (V.STOREBATCH) ---
    print("[2] Prefill v_offload (V.STOREBATCH)...", end=" ")
    try:
        hook.on_prefill(session_id, layer_id=0, K=K, V=V)
        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAIL ({e})")
        failed += 1

    # --- Test 3: Decode v_offload ---
    print("[3] Decode v_offload (V.FETCH)...", end=" ")
    try:
        output = hook.on_decode_attention(session_id, layer_id=0, Q=Q, K_local=K, top_k=top_k)
        ref = cpu_reference_attention(Q, K, V, top_k)
        cos = cosine_sim(output, ref)
        if cos > 0.90:
            print(f"OK (cosine={cos:.4f})")
            passed += 1
        else:
            print(f"FAIL (cosine={cos:.4f}, expected > 0.90)")
            failed += 1
    except Exception as e:
        print(f"FAIL ({e})")
        failed += 1

    # --- Test 4: gpu_attention via PionPromptCache(stage2=True) ---
    if health.get("metal_attention"):
        print("[4] gpu_attention (ATTEND.PREFIX.STORE+QUERY)...", end=" ")
        gpu_hook = PionAttentionHook(
            pion_host=pion_host, pion_port=pion_port, mode="gpu_attention"
        )
        try:
            gpu_session = session_id + "_gpu"
            gpu_hook.on_prefill(gpu_session, layer_id=0, K=K, V=V)
            output = gpu_hook.on_decode_attention(
                gpu_session, layer_id=0, Q=Q, K_local=K, top_k=top_k
            )
            ref = cpu_dense_attention(Q, K, V)
            cos = cosine_sim(output, ref)
            if cos > 0.90:
                print(f"OK (cosine={cos:.4f})")
                passed += 1
            else:
                print(f"FAIL (cosine={cos:.4f}, expected > 0.90)")
                failed += 1
            gpu_hook.drop_session(gpu_session)
        except Exception as e:
            print(f"FAIL ({e})")
            failed += 1
        gpu_hook.close()
    else:
        print("[4] gpu_attention — SKIPPED (Pion not built with --metal-attention)")

    # --- Test 5: Session cleanup ---
    print("[5] Drop session...", end=" ")
    try:
        hook.drop_session(session_id)
        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAIL ({e})")
        failed += 1

    hook.close()

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
