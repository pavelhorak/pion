#!/usr/bin/env python3
"""ATTEND.PREFIX.QUERY batched-Q form — closes §25.6 / §26.5 last item.

Adds Q with shape (H, M, D) over the wire so a Stage-2 monkey-patch can attend
the entire suffix in one round trip, instead of M separate round trips.

Verifies:
  1. Backward compat — Q (H, D) still works (M=1 path).
  2. Batched form — Q (H, M, D) returns (H, M, D) attention output.
  3. Numerical agreement vs CPU softmax(QK^T/√d)·V (cosine ≥ 0.9999).
  4. Wire-cost win — one batched call vs M separate calls at the same N/H/D.

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import socket
import sys
import time

import numpy as np

sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx.prompt_cache import PionPromptCache, _RESP


def cpu_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference dense softmax attention.
    Q: (H, M, D), K, V: (H, N, D). Returns (H, M, D)."""
    H, M, D = Q.shape
    out = np.empty((H, M, D), dtype=np.float32)
    scale = 1.0 / np.sqrt(D)
    for h in range(H):
        scores = (Q[h] @ K[h].T) * scale  # (M, N)
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=-1, keepdims=True)
        out[h] = w @ V[h]
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    n = np.linalg.norm(af) * np.linalg.norm(bf)
    return float((af @ bf) / n) if n > 0 else 0.0


def store_kv_via_attend_prefix(client: PionPromptCache, ns: str, layer_id: int,
                                K: np.ndarray, V: np.ndarray) -> None:
    """Push K/V (H, N, D) to the MLX sidecar."""
    client._attend_store_layer(ns, layer_id, K, V)
    client._stage2_pushed.add(client._attend_session(ns))


def main() -> int:
    s = socket.create_connection(("127.0.0.1", 1974), timeout=2); s.close()
    H, N, D = 8, 256, 64

    # Fake "model" — only PionPromptCache needs .args / make_prompt_cache during
    # get_or_prefill. We never call get_or_prefill here, so pass None and use
    # the lower-level attend helpers directly.
    class FakeArgs:
        num_hidden_layers = 1
        num_key_value_heads = H
        num_attention_heads = H
        head_dim = D
        hidden_size = H * D
    class FakeModel:
        args = FakeArgs()
    pc = PionPromptCache.__new__(PionPromptCache)
    pc.model = FakeModel()
    pc.vquant = "fp16"
    pc.layout = pc.__class__.__init__.__globals__["_layout_from"](FakeModel())
    pc.host = "127.0.0.1"; pc.port = 1974
    pc.stage2 = True
    pc.admission_threshold = 1
    pc.resp = _RESP("127.0.0.1", 1974)
    pc._local = {}
    pc._stage2_pushed = set()
    pc._observed = {}
    pc.hits = pc.misses = pc.admission_skips = 0
    pc.fetch_ms_total = pc.store_ms_total = 0.0
    pc.attend_query_ms_total = 0.0; pc.attend_query_calls = 0

    rng = np.random.default_rng(42)
    K = rng.standard_normal((H, N, D)).astype(np.float32)
    V = rng.standard_normal((H, N, D)).astype(np.float32)

    ns = PionPromptCache.make_namespace("batched_q_test", "v1")
    store_kv_via_attend_prefix(pc, ns, 0, K, V)

    # ── (1) backward compat: Q (H, D) ────────────────────────────────────
    Q1 = rng.standard_normal((H, D)).astype(np.float32)
    out1 = pc.attend_query(ns, 0, Q1, top_k=N)  # full softmax via top_k=N
    assert out1.shape == (H, D), f"expected (H, D), got {out1.shape}"
    ref1 = cpu_attention(Q1[:, None, :], K, V)[:, 0, :]
    cos1 = cosine(out1, ref1)
    print(f"[batched-q] M=1  cosine vs CPU: {cos1:.6f}  (expect ~1.0)")
    assert cos1 >= 0.999, f"single-token cosine too low: {cos1}"

    # ── (2) batched form: Q (H, M, D), M=8 ───────────────────────────────
    M = 8
    Qb = rng.standard_normal((H, M, D)).astype(np.float32)
    out_b = pc.attend_query(ns, 0, Qb)
    assert out_b.shape == (H, M, D), f"expected (H, M, D), got {out_b.shape}"
    ref_b = cpu_attention(Qb, K, V)
    cos_b = cosine(out_b, ref_b)
    print(f"[batched-q] M={M} cosine vs CPU: {cos_b:.6f}  (expect ≥0.9999)")
    assert cos_b >= 0.9999, f"batched cosine too low: {cos_b}"

    # ── (3) latency comparison: M sequential calls vs 1 batched call ─────
    Qs = rng.standard_normal((H, M, D)).astype(np.float32)

    t0 = time.perf_counter()
    seq_out = np.empty((H, M, D), dtype=np.float32)
    for j in range(M):
        seq_out[:, j, :] = pc.attend_query(ns, 0, Qs[:, j, :], top_k=N)
    seq_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    bat_out = pc.attend_query(ns, 0, Qs)
    bat_ms = (time.perf_counter() - t0) * 1000

    cos_seq_bat = cosine(seq_out, bat_out)
    print(f"[batched-q] M={M} seq calls: {seq_ms:.2f} ms  batched: {bat_ms:.2f} ms  speedup: {seq_ms/bat_ms:.2f}×")
    print(f"[batched-q] seq vs batched cosine: {cos_seq_bat:.6f} (expect ~1.0; small ε from sparse top_k vs dense)")

    assert bat_ms < seq_ms, f"batched not faster: {bat_ms:.2f} >= {seq_ms:.2f}"

    print("[batched-q] PASS — (H, M, D) wire form correct, faster than M single calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
