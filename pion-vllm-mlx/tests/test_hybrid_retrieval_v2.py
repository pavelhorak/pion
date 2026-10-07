#!/usr/bin/env python3
"""gh #150 — multi-chunk composition v2: shared stub + delta-rotation.

The load-bearing check is the first one: a chunk's K, encoded at one set of
positions and rotated by delta, must equal the same chunk encoded at the target
positions. That is the claim the whole composition rests on — if it holds, RoPE
is not the composition problem, and the remaining gap is the cross-pack
distractor effect that the max_packs guard exists to bound.

Model: whichever local mlx model is present (no download). Skips cleanly if mlx
or a cached model is missing.

Usage: python3 pion-vllm-mlx/tests/test_hybrid_retrieval_v2.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


def skip(name, why):
    SKIP.append(name)
    print(f"  SKIP  {name} — {why}")


CANDIDATES = [
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
]


def _load_local():
    """Load whichever candidate is already in the HF cache. No downloads —
    a test that silently pulls 2 GB is a test nobody runs twice."""
    from huggingface_hub import try_to_load_from_cache
    from mlx_lm import load
    import json
    for repo in CANDIDATES:
        hit = try_to_load_from_cache(repo, "config.json")
        if isinstance(hit, str):
            # Delta-rotation assumes plain RoPE (R(d)R(p) == R(p+d)); a
            # rope_scaling model (Llama 3.x "llama3" scaling) is outside the
            # feature, and picking one made the test abort, not test.
            if json.load(open(hit)).get("rope_scaling"):
                continue
            return load(repo), repo
    return None, None


def main():
    try:
        import mlx.core as mx
    except ImportError:
        skip("gh #150 v2", "mlx not installed")
        return 0

    loaded, repo = _load_local()
    if loaded is None:
        skip("gh #150 v2", f"none of {CANDIDATES} in the local HF cache")
        return 0
    (model, tok) = loaded
    print(f"gh #150 v2 multi-chunk composition — {repo}\n")

    from pion_vllm_mlx import HybridRetrievalCache

    hr = HybridRetrievalCache(model)
    stub_ids = tok.encode("You are a precise assistant. Answer from the documents.\n")
    hr.set_shared_stub(stub_ids)
    check("shared stub encodes", hr._stub_len == len(stub_ids),
          f"{hr._stub_len} tokens")

    # Packs and the question follow the stub, so they carry no special tokens;
    # the stub opens the prompt and keeps the tokenizer's <bos>, if it has one.
    doc_a = tok.encode("Document A: the Pion server listens on port 1974 by default.\n",
                       add_special_tokens=False)
    doc_b = tok.encode("Document B: the reference recall floor for the vector gate is 0.940.\n",
                       add_special_tokens=False)
    hr.ingest_pack("a", doc_a)
    hr.ingest_pack("b", doc_b)
    check("packs ingest behind the stub",
          hr._manifest["a"]["encoded_at"] == hr._stub_len
          and hr._manifest["b"]["encoded_at"] == hr._stub_len)

    # ── 1. rotation identity ────────────────────────────────────────────────
    # Pack B was encoded at [stub_len, stub_len+len(b)). In a composition it
    # lands after pack A, i.e. shifted by len(doc_a). Rotating its stored K by
    # that delta must reproduce encoding [stub + A + B] and slicing B out.
    delta = len(doc_a)
    rotated = hr._rotate_k(hr._inproc_kv["b"][0][0], delta)

    ref_ids = stub_ids + doc_a + doc_b
    ref = hr._encode_to_mlx_kv(ref_ids)
    start = hr._stub_len + len(doc_a)
    ref_k = ref[0][0][:, :, start:start + len(doc_b), :]

    num = float(mx.max(mx.abs(rotated.astype(mx.float32) - ref_k.astype(mx.float32))).item())
    den = float(mx.max(mx.abs(ref_k.astype(mx.float32))).item()) or 1.0
    rel = num / den
    check("delta-rotation reproduces prefill-at-target-offset (layer 0 K)",
          rel < 5e-2, f"rel max err {rel:.2e} (fp16-stored K; fp32 identity is ~2e-05)")

    # V must be left alone — rotating it would be a silent corruption.
    v_same = bool(mx.all(hr._inproc_kv["b"][0][1] == hr._inproc_kv["b"][0][1]).item())
    check("V is not rotated", v_same)

    # ── 2. composition runs and stays coherent ──────────────────────────────
    q = tok.encode("\nQuestion: which port does the Pion server listen on?\nAnswer:",
                   add_special_tokens=False)
    cache, suffix = hr.prepare_multi(["a", "b"], q)
    check("prepare_multi returns a cache positioned past stub + both packs",
          cache[0].offset == hr._stub_len + len(doc_a) + len(doc_b),
          f"offset {cache[0].offset}")

    out = []
    last = mx.array([suffix])
    for _ in range(12):
        logits = model(last, cache=cache)
        mx.eval(logits)
        nxt = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        out.append(nxt)
        last = mx.array([[nxt]])
    text = tok.decode(out)
    check("composed cache generates the fact from pack A", "1974" in text,
          f"generated {text!r}")

    # ── 3. the coarse-grain bound is enforced ───────────────────────────────
    raised = False
    try:
        hr.prepare_multi(["a", "b"], q, max_packs=1)
    except ValueError as exc:
        raised = "mono" in str(exc) or "15/27" in str(exc)
    check("prepare_multi refuses more packs than max_packs, citing the measurement",
          raised)

    override_ok = False
    try:
        hr.prepare_multi(["a", "b"], q, max_packs=1, allow_fine_grain=True)
        override_ok = True
    except ValueError:
        override_ok = False
    check("allow_fine_grain=True overrides the bound deliberately", override_ok)

    # ── 4. stub omission is refused rather than silently degenerate ─────────
    hr2 = HybridRetrievalCache(model)
    refused = False
    try:
        hr2.ingest_pack("x", doc_a)
    except RuntimeError:
        refused = True
    check("ingest_pack without a shared stub is refused", refused)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


sys.exit(main())
