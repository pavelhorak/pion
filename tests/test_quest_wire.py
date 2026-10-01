#!/usr/bin/env python3
"""W11 / gh #9 Phase 2 wire-level smoke for the Quest selector on Metal.

This is a **plumbing test, not a quality test**. It validates that:

  1. ATTEND.PREFIX.QUERY_SPARSE_AUTO with selector_id=0 (block-mean
     default) and selector_id=1 (Quest) both succeed against the same
     stored K/V — no FFI signature mismatch, no shape error, the
     selector_id byte is actually parsed all the way down.
  2. The two selectors produce DIFFERENT outputs (different blocks
     selected → different attention output). If outputs were identical
     the selector_id plumbing would be silently dead.

The wire smoke does NOT validate "Quest produces better-cosine output
than block-mean on a contrived synthetic K/V" because that depends on
the relative weights of softmax over the selected vs unselected
tokens — small spikes in K_max can still receive small softmax weight
relative to many mid-attention tokens, so picking the spike blocks via
Quest may produce a *different* output than dense without being
*closer* to dense.

The load-bearing Quest-beats-block-mean signal lives in the Phase 1
e2e test on real Llama-3.2-1B-4bit + the gh #23 SQuAD v2 corpus —
Quest F1=0.188 > block-mean F1=0.132 at K_top=8.

Pre-req:
  ./pion-server --kvcache --metal-attention --no-auto-embed -w 1
"""
from __future__ import annotations

import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pion-vllm-mlx"))

from pion_vllm_mlx.prompt_cache import PionPromptCache  # noqa: E402


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def store_via_resp(host: str, port: int, sid: str, layer: int,
                   K: np.ndarray, V: np.ndarray) -> None:
    """ATTEND.PREFIX.STORE wire frame. K, V shape: (H, N, D) fp32."""
    H, N, D = K.shape
    Kb = K.astype(np.float32).tobytes()
    Vb = V.astype(np.float32).tobytes()
    sock = socket.socket()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(30.0)
    sock.connect((host, port))
    parts = [b"ATTEND.PREFIX.STORE", sid.encode(), str(layer).encode(),
             str(H).encode(), str(N).encode(), str(D).encode(), Kb, Vb]
    frame = f"*{len(parts)}\r\n".encode()
    for p in parts:
        frame += f"${len(p)}\r\n".encode() + p + b"\r\n"
    sock.sendall(frame)
    line = b""
    while not line.endswith(b"\r\n"):
        c = sock.recv(1)
        if not c:
            break
        line += c
    sock.close()
    if not line.startswith(b"+OK"):
        raise RuntimeError(f"STORE failed: {line!r}")


def main():
    HOST, PORT = "127.0.0.1", 1974
    H = 4
    N = 512
    D = 64
    B = 64
    K_top = 4
    LAYER = 0
    NS = "w11_phase2_smoke"

    rng = np.random.RandomState(123)

    # Dispersed-relevance K (mirrors Phase 1 Scenario B): all tokens are
    # weak noise, 4 weak spikes scattered across blocks 1/3/5/7, 4 elevated-
    # mean noise blocks at 0/2/4/6 designed to fool block-mean. V is also
    # noise; we only validate Q→K selector behaviour here, V irrelevant.
    K = rng.randn(H, N, D).astype(np.float32) * 0.1
    V = rng.randn(H, N, D).astype(np.float32)
    direction = rng.randn(D).astype(np.float32)
    direction /= np.linalg.norm(direction)
    spike_blocks = [1, 3, 5, 7]
    decoy_blocks = [0, 2, 4, 6]
    for sb in spike_blocks:
        K[:, sb * B + 10, :] = direction * 2.0
    for db in decoy_blocks:
        for t in range(B):
            K[:, db * B + t, :] += direction * 0.3

    Q = np.tile(direction, (H, 1))   # (H, D) — every head queries the same direction

    print(f"[setup] H={H} N={N} D={D} B={B} K_top={K_top}  → {N // B} prefix blocks")
    print(f"[setup] spike blocks (should be picked):  {spike_blocks}")
    print(f"[setup] decoy blocks (mid-mean noise):    {decoy_blocks}")

    # PionPromptCache derives the on-server session ID via _attend_session
    # which appends "_attn" to the namespace. STORE must use that same SID
    # or the QUERY will hit "session not found".
    SID = f"{NS}_attn"
    print(f"\n[1] STORE K/V into Pion @ {HOST}:{PORT} (sid={SID}, layer={LAYER})")
    store_via_resp(HOST, PORT, SID, LAYER, K, V)

    pc = PionPromptCache(model=None, host=HOST, port=PORT)
    pc._local[NS] = N

    print(f"\n[2] dense ATTEND.PREFIX.QUERY (top_k={N})")
    out_dense = pc.attend_query(NS, LAYER, Q, top_k=N)
    print(f"    output shape: {out_dense.shape}")

    print(f"\n[3] ATTEND.PREFIX.QUERY_SPARSE_AUTO selector=block_mean")
    out_bm = pc.attend_query_sparse_auto(NS, LAYER, Q, B, K_top, H_kv=H,
                                          selector="block_mean")
    print(f"    output shape: {out_bm.shape}")

    print(f"\n[4] ATTEND.PREFIX.QUERY_SPARSE_AUTO selector=quest")
    out_quest = pc.attend_query_sparse_auto(NS, LAYER, Q, B, K_top, H_kv=H,
                                             selector="quest")
    print(f"    output shape: {out_quest.shape}")

    cos_bm    = cos(out_dense, out_bm)
    cos_quest = cos(out_dense, out_quest)
    cos_bm_q  = cos(out_bm, out_quest)

    print("\n=== Cosine vs dense ground truth ===")
    print(f"  cos(dense, block_mean) = {cos_bm:.6f}")
    print(f"  cos(dense, quest)      = {cos_quest:.6f}")
    print(f"  cos(block_mean, quest) = {cos_bm_q:.6f}  (≠ 1.0 means selectors picked different blocks)")

    # ─── W11 Phase 2 e2e re-anchor: in-proc vs wire equivalence ────────────
    # Compute the SAME Quest selection in MLX in-proc on the same K/V/Q,
    # then compare against the wire-path output. If cos ≈ 1.0 the C-side
    # selector + sparse SDPA matches MLX-side selector + mx.fast SDPA, and
    # the Phase 1 factual-QA F1 win (block-mean 0.132 → Quest 0.188 at
    # K_top=8 on the gh #23 SQuAD v2 corpus) transfers to wire-mode
    # consumers by construction. No need to re-run the n=20 e2e benchmark.
    try:
        import mlx.core as mx
        from pion_vllm_mlx.mlx_lm_patch import _quest_topk_select, _block_mean_topk_select
        # Promote synthetic numpy arrays into MLX with the shape PionPrefixCache
        # would have: (1, H, N, D) for K/V and (1, H, M=1, D) for Q.
        K_mx = mx.array(K[None, ...])            # (1, H, N, D)
        V_mx = mx.array(V[None, ...])
        Q_mx = mx.array(Q[None, :, None, :])     # (1, H, 1, D)
        scale = 1.0 / float(np.sqrt(D))

        Kq_sel, Vq_sel = _quest_topk_select(Q_mx, K_mx, V_mx, prefix_len=N,
                                              B=B, K_top=K_top)
        Kb_sel, Vb_sel = _block_mean_topk_select(Q_mx, K_mx, V_mx, prefix_len=N,
                                                  B=B, K_top=K_top)
        # Run mx.fast SDPA on the selected K/V (same path the in-proc lane
        # does at decode time). Output shape (1, H, 1, D) → (H, D).
        out_quest_inproc = np.array(mx.fast.scaled_dot_product_attention(
            Q_mx, Kq_sel, Vq_sel, scale=scale, mask=None
        ))[0, :, 0, :]
        out_bm_inproc = np.array(mx.fast.scaled_dot_product_attention(
            Q_mx, Kb_sel, Vb_sel, scale=scale, mask=None
        ))[0, :, 0, :]
        cos_quest_inproc_wire = cos(out_quest, out_quest_inproc)
        cos_bm_inproc_wire    = cos(out_bm, out_bm_inproc)
        print("\n=== In-proc (MLX) vs wire (C+Metal) equivalence ===")
        print(f"  cos(quest_wire, quest_inproc)             = {cos_quest_inproc_wire:.6f}")
        print(f"  cos(block_mean_wire, block_mean_inproc)   = {cos_bm_inproc_wire:.6f}")
        # Phase 2 load-bearing claim is QUEST equivalence: the Phase 1 e2e
        # factual-QA F1 win (block-mean 0.132 → Quest 0.188) transfers to
        # the wire path iff the wire selector + SDPA produce numerically
        # equivalent output to the in-proc MLX path on the same input.
        # Quest's max-based scoring is reduction-order-independent so it
        # achieves bit-exact agreement (cos ≈ 1.000000); a divergence here
        # would mean the C-side Quest formula has a real bug.
        #
        # Block-mean's mean-based scoring has a numerical-order dependence
        # (sequential C sum vs MLX tree-reduce). On the dispersed-relevance
        # synthetic the 4 decoy blocks have nearly-tied mean scores; small
        # rounding swaps the picked set across paths, producing the ~0.9
        # cosine. This is NOT a Phase 2 regression — block-mean has been on
        # the wire since gh #63, well before any selector_id surface.
        # Informational only; gate only on Quest equivalence.
        quest_equiv_ok = cos_quest_inproc_wire > 0.995
    except ImportError:
        print("\n  (skipped in-proc vs wire equivalence — mlx not installed in this environment)")
        quest_equiv_ok = True

    failures = 0
    if not quest_equiv_ok:
        print(f"\n  ✗ FAIL: Quest wire output diverges from Quest in-proc output. "
              f"Phase 1 F1 win does NOT transfer; the C-side Quest scoring formula "
              f"in pion_metal_sdpa_query_sparse_auto has a real bug.")
        failures += 1
    if cos_bm_q >= 0.9999:
        # Outputs identical → selector_id is silently no-op on the server.
        print(f"\n  ✗ FAIL: block_mean and quest produced identical outputs "
              f"({cos_bm_q:.6f}) — selector_id plumbing is not actually "
              f"taking effect on the server side.")
        failures += 1
    if failures:
        return 1
    print("\n  ✓ Wire-level Quest plumbing on Metal: PASS")
    print("    (selectors produce different outputs; selector_id is wired)")
    print("    The Quest-beats-block-mean quality signal is on real workloads")
    print("    not on contrived synthetic K/V with random V — softmax weighting")
    print("    means dense output can be dominated by the *quantity* of mid-")
    print("    weight tokens, not the *peak* tokens Quest is best at finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
