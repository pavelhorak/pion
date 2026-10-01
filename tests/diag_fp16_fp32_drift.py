"""Isolate the fp16-vs-fp32 attention drift between Pion's fp32 Metal SDPA
and vanilla mlx-lm's fp16 attention.

Three diagnostics, all running the SAME random K/V/Q at Llama-3.2-1B-shape
(H=8 GQA-replicated to 32 Hq, N=512 N_prefix, D=64), comparing attention
outputs computed at different precisions.

  1. attn(K_fp32, V_fp32, Q_fp32) — Pion's wire/kernel path.
  2. attn(K_fp16, V_fp16, Q_fp16) — vanilla mlx-lm-style.
  3. attn(K_fp64, V_fp64, Q_fp64) — gold standard CPU reference.

All three should produce the same RESULT in real arithmetic. Their
floating-point drift is the question.

If Pion's fp32 path matches fp64 reference within ~1e-7 (cosine 1.0)
but matches fp16 path only within ~1e-3 (cosine 0.99x), then the
test_mlx_lm_patch.py 5pp gap is structural fp16-vs-fp32 drift — not a
bug, just a precision-contract mismatch. The fix is either to round
Pion's output to fp16 (cheap, ships today) or run the kernel in fp16
(roadmap step 3, multi-day).
"""
from __future__ import annotations
import sys
import numpy as np

H_KV, H_Q, N, D = 8, 32, 512, 64  # Llama-3.2-1B with GQA factor 4


def attn(K, V, Q, dtype):
    """Standard softmax(QK^T/sqrt(D))V at the requested dtype."""
    K = K.astype(dtype)
    V = V.astype(dtype)
    Q = Q.astype(dtype)
    scale = (1.0 / np.sqrt(D)).astype(dtype)
    # Q [H, M, D], K [H, N, D], V [H, N, D]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale  # [H, M, N]
    sm = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(sm.astype(np.float64) if dtype == np.float64 else sm)
    e = e.astype(dtype) if dtype != np.float64 else e
    a = e / e.sum(axis=-1, keepdims=True)
    return np.matmul(a, V)


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def stats(name, ref_fp64, candidate, dtype_label):
    diff = candidate.astype(np.float64) - ref_fp64.astype(np.float64)
    abs_max = float(np.abs(diff).max())
    abs_max_ref = float(np.abs(ref_fp64).max())
    rel_max = abs_max / abs_max_ref if abs_max_ref > 0 else 0.0
    cos = cosine(candidate, ref_fp64)
    print(f"  {name} (dtype={dtype_label})  max|Δ|={abs_max:.2e}  max rel={rel_max:.2e}  cosine vs fp64 ref={cos:.10f}")


def gqa_repack(Q_hq, H_kv):
    """Replicate the patch's GQA repack: Q[Hq, M, D] → Q[Hkv, rep*M, D]."""
    Hq, M, D_ = Q_hq.shape
    rep = Hq // H_kv
    return Q_hq.reshape(H_kv, rep, M, D_).reshape(H_kv, rep * M, D_)


def main():
    rng = np.random.default_rng(0xC0FFEE)
    # Realistic ranges — RMSNorm + projection makes activations roughly N(0, 0.5).
    K = (rng.standard_normal((H_KV, N, D)) * 0.5).astype(np.float64)
    V = (rng.standard_normal((H_KV, N, D)) * 0.5).astype(np.float64)
    # Single decode-step Q, GQA repacked per the mlx-lm patch's wire convention.
    Q_hq = (rng.standard_normal((H_Q, 1, D)) * 0.5).astype(np.float64)
    Q_send = gqa_repack(Q_hq, H_KV)  # [H_kv, rep*M, D] = [8, 4, 64] for M=1
    print(f"shape: K [H_kv={H_KV}, N={N}, D={D}], "
          f"Q_hq [H_q={H_Q}, M=1, D={D}] → wire Q_send [H_kv={H_KV}, rep*M=4, D={D}]")
    print()

    # Gold standard
    out_fp64 = attn(K, V, Q_send, np.float64)

    # Candidates
    out_fp32 = attn(K, V, Q_send, np.float32)
    out_fp16 = attn(K, V, Q_send, np.float16)
    # bf16 not native to numpy; emulate via fp32 with explicit truncation.
    # Skip bf16 here — fp16 is what mlx-lm actually uses for Llama-3.2-1B-4bit.

    print(f"Reference: attn(K, V, Q_send) at fp64")
    print(f"  shape: {out_fp64.shape}, max|out|={float(np.abs(out_fp64).max()):.4f}")
    print()
    print("Drift vs fp64 gold:")
    stats("fp32 (Pion's wire+kernel path)", out_fp64, out_fp32, "fp32")
    stats("fp16 (vanilla mlx-lm path)",     out_fp64, out_fp16, "fp16")
    print()
    print("Cross-precision drift (Pion vs vanilla):")
    print(f"  cosine(fp32, fp16) = {cosine(out_fp32, out_fp16):.10f}")
    diff_3216 = (out_fp32.astype(np.float64) - out_fp16.astype(np.float64))
    print(f"  max|fp32 - fp16|   = {float(np.abs(diff_3216).max()):.2e}")
    print(f"  max relative       = {float(np.abs(diff_3216).max() / np.abs(out_fp64).max()):.2e}")
    print()
    print("Fix-candidate strategies (each emulates a Pion-side precision contract):")
    print()
    # (a) Round Pion's fp32 output to fp16 before returning.
    out_fp32_rounded = out_fp32.astype(np.float16).astype(np.float32)
    print(f"  (a) Pion fp32 attention, output cast to fp16 then fp32:")
    diff_a = out_fp32_rounded.astype(np.float64) - out_fp16.astype(np.float64)
    print(f"      cosine(this, vanilla fp16)  = {cosine(out_fp32_rounded, out_fp16):.10f}")
    print(f"      max|this - vanilla fp16|    = {float(np.abs(diff_a).max()):.2e}")

    # (b) Round K/V/Q to fp16 BEFORE the kernel runs (so kernel operates on fp16-precision
    #     values cast back to fp32 for register-width compute).
    K16 = K.astype(np.float16).astype(np.float64)
    V16 = V.astype(np.float16).astype(np.float64)
    Q16 = Q_send.astype(np.float16).astype(np.float64)
    out_kv16 = attn(K16, V16, Q16, np.float32)
    print(f"  (b) K/V/Q rounded to fp16 then attention computed at fp32 (Pion-side cast):")
    diff_b = out_kv16.astype(np.float64) - out_fp16.astype(np.float64)
    print(f"      cosine(this, vanilla fp16)  = {cosine(out_kv16, out_fp16):.10f}")
    print(f"      max|this - vanilla fp16|    = {float(np.abs(diff_b).max()):.2e}")

    # (c) Run attention itself at fp16 (true vanilla path; this is roadmap step 3).
    print(f"  (c) Native fp16 kernel (roadmap step 3):")
    print(f"      cosine(this, vanilla fp16)  = {cosine(out_fp16, out_fp16):.10f}  (definitionally 1.0)")

    print()
    print("Verdict:")
    rel_3216 = float(np.abs(diff_3216).max() / np.abs(out_fp64).max())
    if rel_3216 > 1e-2:
        print(f"  CONFIRMED — Pion fp32 vs vanilla fp16 differ by {rel_3216:.1%} relative.")
        print(f"  This is the size of drift expected from fp16 ULP rounding cascading through")
        print(f"  softmax+matmul. Greedy decode amplifies this into different tokens.")
        print()
        print(f"  Fix options:")
        print(f"    (a) Cheap: round Pion's wire output to fp16 → fp32 before returning.")
        print(f"        One memcpy + cast in the FFI; ships today.")
        print(f"    (b) Proper: run the Metal kernel in fp16. Roadmap step 3.")
    elif rel_3216 > 1e-4:
        print(f"  Drift {rel_3216:.1%} is on fp16's ULP boundary — borderline. fp32 vs fp16")
        print(f"  is the right hypothesis. Greedy decode is sensitive enough to amplify this.")
    else:
        print(f"  Drift {rel_3216:.1%} is below fp16 ULP — hypothesis FALSIFIED.")
        print(f"  Look elsewhere (algorithm, masking, scale dtype, online vs full softmax).")


if __name__ == "__main__":
    sys.exit(main() or 0)
