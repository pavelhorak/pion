#!/usr/bin/env python3
"""M14 Phase 3: Quality validation — verify HNSW attention retrieval correctness.

Tests that HNSW top-k retrieval preserves attention quality by:
1. Generating synthetic "attention-like" KV pairs with known structure
2. Storing them in Pion
3. Querying and verifying the correct tokens are retrieved
4. Computing attention output and comparing to ground truth (full attention)

No real model needed — uses synthetic data with controlled similarity structure.
"""

import sys
import time
import math

sys.path.insert(0, "vllm-pion")

import numpy as np
from vllm_pion.attention_client import PionAttentionClient

HOST = "127.0.0.1"
PORT = 1974


def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def full_attention(query, keys, values, head_dim=128):
    """Compute full attention: softmax(Q·K^T / sqrt(d)) · V"""
    scores = keys @ query / math.sqrt(head_dim)
    weights = softmax(scores)
    return weights @ values


def topk_attention(query, topk_keys, topk_values, head_dim=128):
    """Compute attention over just top-k tokens."""
    scores = topk_keys @ query / math.sqrt(head_dim)
    weights = softmax(scores)
    return weights @ topk_values


def test_retrieval_accuracy(client, n_tokens=4000, key_dim=128, val_dim=128):
    """Test that Pion returns the correct best-match value via HNSW.

    Stores tokens with known unique values, then queries and verifies
    the returned value matches the expected best-match token.
    """
    print(f"\n=== Retrieval Accuracy Test (n={n_tokens}) ===")

    session_id = "quality_retrieval_v2"
    client.create_session(session_id, key_dim, val_dim)

    np.random.seed(42)
    keys = np.random.randn(n_tokens, key_dim).astype(np.float32)
    # Each value row encodes its token id in BITS (0.0/1.0 per dim). Values are
    # INT8-quantized with ONE scale per layer, so a row holding the integer i
    # comes back as roughly i +/- range/255 — for 4000 ids that is +/-8 and
    # "exact" could never hold. 0 and 1 survive the quantizer exactly.
    values = ((np.arange(n_tokens)[:, None] >> np.arange(val_dim)) & 1).astype(np.float32)

    # Store all tokens in batches, then finalize once
    batch = 500
    for b in range(0, n_tokens, batch):
        n = min(batch, n_tokens - b)
        client.store_tokens(session_id, 0, keys[b:b+n], values[b:b+n])
    client.finalize_layer(session_id, 0)

    # Find the ground-truth best match via brute force (on normalized keys)
    norm_keys = keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-8) * 0.15

    # Test 10 random queries
    correct = 0
    tested = 0
    for qi in range(10):
        # Use a key from the dataset as query (should match itself)
        target_idx = qi * (n_tokens // 10)
        query = keys[target_idx].copy()

        result = client.query_topk(session_id, 0, query, k=1)
        if result is None:
            print(f"  Query {qi}: MISS")
            tested += 1
            continue

        # The returned value should be [target_idx, target_idx, ...]
        retrieved_val = np.frombuffer(result, dtype=np.float32)
        retrieved_id = int(sum(int(round(float(b))) << bi for bi, b in enumerate(retrieved_val[:16])))
        retrieved_id = min(retrieved_id, n_tokens - 1)

        # Check: is the retrieved token close to the expected?
        # HNSW with INT8 quantization may return a nearby token, not exact.
        # Accept if the retrieved token's key has high cosine to the query.
        norm_query = query / (np.linalg.norm(query) + 1e-8) * 0.15
        cos_sim = np.dot(norm_keys[retrieved_id], norm_query) / (
            np.linalg.norm(norm_keys[retrieved_id]) * np.linalg.norm(norm_query) + 1e-8)

        is_correct = (retrieved_id == target_idx)
        is_close = cos_sim > 0.90
        if is_correct:
            correct += 1
        tested += 1

        if qi < 3:  # Print first 3
            print(f"  Query {qi}: target={target_idx} retrieved={retrieved_id} "
                  f"cosine={cos_sim:.3f} {'EXACT' if is_correct else ('CLOSE' if is_close else 'WRONG')}")

    # A query with a STORED key must return that key's own value (rule 5:
    # known-answer data, the stored item's exact vector, required FIRST).
    # This used to accept 50%, and main() ignored the result anyway.
    print(f"  Exact match: {correct}/{tested}")
    return tested == 10 and correct == tested


def test_attention_quality_at_scale(client, n_tokens=4000, ks=[16, 32, 64, 128, 256], key_dim=128, val_dim=128):
    """Measure attention quality (cosine similarity) at different k values.

    This validates the plan's claim that k=128 preserves 95%+ quality.
    """
    print(f"\n=== Attention Quality vs k (n={n_tokens}) ===")

    np.random.seed(123)
    keys = np.random.randn(n_tokens, key_dim).astype(np.float32)
    # Make attention sparse: a few tokens have high similarity to query
    query = np.random.randn(key_dim).astype(np.float32)
    # Inject 50 "important" tokens that are very similar to the query
    for i in range(50):
        keys[i * 80] = query + np.random.randn(key_dim).astype(np.float32) * 0.05

    values = np.random.randn(n_tokens, val_dim).astype(np.float32)

    # Ground truth: full attention
    full_output = full_attention(query, keys, values, key_dim)

    print(f"  {'k':>6} | {'Cosine':>8} | {'MSE':>12} | {'Quality':>8}")
    print(f"  {'-'*6}|{'-'*10}|{'-'*14}|{'-'*10}")

    for k in ks:
        scores = keys @ query
        topk_idx = np.argsort(scores)[-k:]
        tk = keys[topk_idx]
        tv = values[topk_idx]
        topk_out = topk_attention(query, tk, tv, key_dim)

        cos = np.dot(full_output, topk_out) / (np.linalg.norm(full_output) * np.linalg.norm(topk_out) + 1e-8)
        mse = np.mean((full_output - topk_out) ** 2)
        quality = "PASS" if cos > 0.95 else "MARGINAL" if cos > 0.90 else "FAIL"

        print(f"  {k:>6} | {cos:>8.4f} | {mse:>12.6f} | {quality:>8}")

    return True


def test_sparse_attention_property(n_tokens=4000, key_dim=128):
    """Validate the H2O paper's claim: 10-20% of tokens capture 90%+ of attention mass.

    This is the theoretical foundation for externalized attention.
    """
    print(f"\n=== Sparse Attention Property (n={n_tokens}) ===")

    np.random.seed(77)
    query = np.random.randn(key_dim).astype(np.float32)
    keys = np.random.randn(n_tokens, key_dim).astype(np.float32)

    # Inject heavy-hitter tokens (like H2O paper)
    n_heavy = 50  # 50 tokens with very high attention
    for i in range(n_heavy):
        keys[i * (n_tokens // n_heavy)] = query * (1.0 + np.random.randn() * 0.1)

    scores = keys @ query / math.sqrt(key_dim)
    weights = softmax(scores)

    # Sort weights descending
    sorted_w = np.sort(weights)[::-1]
    cum_mass = np.cumsum(sorted_w)

    for pct in [0.05, 0.10, 0.20, 0.50]:
        k = int(n_tokens * pct)
        mass = cum_mass[k - 1] if k > 0 else 0
        print(f"  Top {pct*100:.0f}% tokens ({k:>4}): capture {mass*100:.1f}% of attention mass")

    # Find k for 90% mass
    k_90 = np.searchsorted(cum_mass, 0.90) + 1
    print(f"  Tokens for 90% mass: {k_90} ({k_90/n_tokens*100:.1f}% of {n_tokens})")
    print(f"  Result: {'PASS' if k_90 / n_tokens < 0.20 else 'FAIL'} (<20% threshold)")

    return k_90 / n_tokens < 0.20


def main():
    print("M14 Phase 3: Quality Validation")
    print(f"Server: {HOST}:{PORT}")
    print()

    # Informational only: these two are numpy arithmetic on synthetic data and
    # never touch the server, so they cannot detect a Pion regression.
    test_sparse_attention_property()
    client = PionAttentionClient(HOST, PORT)
    test_attention_quality_at_scale(client, n_tokens=4000)

    # The server test. A failure here is a FAIL — it used to be caught and
    # printed as "skipped", so a dead or wrong ATTEND path passed the gate.
    ok = test_retrieval_accuracy(client, n_tokens=4000)
    client.close()
    print("\nPASS" if ok else "\nFAIL: stored keys did not retrieve their own values")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
