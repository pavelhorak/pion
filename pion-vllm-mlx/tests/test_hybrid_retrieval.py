"""End-to-end test for HybridRetrievalCache (gh #54).

Two backends are tested:

  inproc  — in-process MLX K/V dict; **bit-perfect** vs combined-encoding
            text-RAG baseline.
  pion    — wire via PionPromptCache (KV.PREFIX.* + V.STOREBATCH/V.FETCH);
            fp16 precision, gated on **lexical answer match** in first 30
            generated tokens (BLEU ~0.97 documented in
            kv_prefix_cache_ship.md). Skipped if pion-server is not up.

Both backends must show ≥ 20% TTFT savings vs the text-RAG baseline.

To run the pion backend:
    ./pion-server --kvcache --metal-attention -w 1 -p 1984
    PION_KVCACHE_PORT=1984 python3 pion-vllm-mlx/tests/test_hybrid_retrieval.py
"""
from __future__ import annotations

import os
import socket
import sys
import time

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx import HybridRetrievalCache


PORT = int(os.environ.get("PION_KVCACHE_PORT", "1984"))
HOST = "127.0.0.1"


def _server_up() -> bool:
    s = socket.socket()
    s.settimeout(0.3)
    try:
        s.connect((HOST, PORT))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _greedy_step(model, cache, last):
    logits = model(last, cache=cache)
    mx.eval(logits)
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


CASES = [
    (
        "eiffel",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
        "Paris, France. It is named after the engineer Gustave Eiffel, whose company "
        "designed and built the tower. Locally nicknamed 'La dame de fer', it was "
        "constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. "
        "The tower is 330 metres tall, about the same height as an 81-storey building, "
        "and the tallest structure in Paris. ",
        "Question: How tall is the Eiffel Tower?\nAnswer:",
        "330",
    ),
    (
        "photosynthesis",
        "Photosynthesis is a process used by plants and other organisms to convert "
        "light energy into chemical energy that, through cellular respiration, can "
        "later be released to fuel the organism's activities. Most plants and algae "
        "perform photosynthesis. Photosynthesis is largely responsible for producing "
        "and maintaining the oxygen content of Earth's atmosphere. ",
        "Question: What gas do plants release during photosynthesis?\nAnswer:",
        "xygen",  # match Oxygen / oxygen case-insensitively via substring
    ),
    (
        "pacific",
        "The Pacific Ocean is the largest and deepest of Earth's five oceanic "
        "divisions. The Mariana Trench in the western North Pacific is the deepest "
        "point in the world, reaching a depth of 10,928 meters. ",
        "Question: What is the deepest point in the Pacific Ocean?\nAnswer:",
        "Mariana",
    ),
]
N_GEN = 30


def _run_method_a(model, tok, chunk_ids, query_ids):
    """Combined encoding text-RAG baseline. Returns (out_tokens, prefill_ms)."""
    from mlx_lm.models.cache import make_prompt_cache
    full = chunk_ids + query_ids
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    _ = model(mx.array([full]), cache=cache)
    mx.eval([c.state for c in cache])
    prefill_ms = (time.perf_counter() - t0) * 1000
    last = mx.array([[full[-1]]])
    out: list[int] = []
    for _ in range(N_GEN):
        nid = _greedy_step(model, cache, last)
        out.append(nid)
        last = mx.array([[nid]])
    return out, prefill_ms


def _run_method_b(model, tok, hr, chunk_id, chunk_ids, query_ids):
    """Hybrid retrieval. Returns (out_tokens, prefill_ms)."""
    hr.ingest(chunk_id, chunk_ids)
    t0 = time.perf_counter()
    cache, suffix = hr.prepare(chunk_id, query_ids)
    _ = model(mx.array([suffix]), cache=cache)
    mx.eval([c.state for c in cache])
    prefill_ms = (time.perf_counter() - t0) * 1000
    last = mx.array([[suffix[-1]]])
    out: list[int] = []
    for _ in range(N_GEN):
        nid = _greedy_step(model, cache, last)
        out.append(nid)
        last = mx.array([[nid]])
    return out, prefill_ms


def run_backend(model, tok, backend: str) -> tuple[int, list[str]]:
    print(f"\n--- backend={backend} ---")
    kwargs = {"backend": backend}
    if backend == "pion":
        kwargs.update({"host": HOST, "port": PORT})
    hr = HybridRetrievalCache(model, **kwargs)
    failures: list[str] = []
    ttft_ratios: list[float] = []
    for chunk_id, chunk_text, query_text, answer_substr in CASES:
        chunk_ids = tok.encode(chunk_text)
        query_ids = tok.encode(query_text)
        out_a, t_a = _run_method_a(model, tok, chunk_ids, query_ids)
        out_b, t_b = _run_method_b(model, tok, hr, chunk_id, chunk_ids, query_ids)
        agreement = sum(1 for a, b in zip(out_a, out_b) if a == b) / N_GEN
        ratio = (t_a - t_b) / t_a * 100
        ttft_ratios.append(ratio)
        text_a = tok.decode(out_a)
        text_b = tok.decode(out_b)

        # Per-backend pass criteria.
        if backend == "inproc":
            # bit-perfect — gate on full token agreement
            quality_ok = agreement >= 0.98
            quality_label = f"agreement={agreement:.3f}"
        else:
            # fp16 wire — gate on functional answer match in first 30 tokens
            answer_in_a = answer_substr.lower() in text_a.lower()
            answer_in_b = answer_substr.lower() in text_b.lower()
            quality_ok = answer_in_a and answer_in_b
            quality_label = (
                f"answer={'Y' if answer_in_b else 'N'} "
                f"(agreement={agreement:.3f})"
            )
        latency_ok = ratio >= 20.0
        ok = quality_ok and latency_ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {chunk_id:14s} {quality_label} "
            f"TTFT_save={ratio:5.1f}% (A={t_a:.1f}ms B={t_b:.1f}ms)"
        )
        if not ok:
            print(f"          A: {text_a[:60]!r}")
            print(f"          B: {text_b[:60]!r}")
            failures.append(chunk_id)
    if ttft_ratios:
        print(
            f"  TTFT savings: mean={sum(ttft_ratios)/len(ttft_ratios):.1f}% "
            f"min={min(ttft_ratios):.1f}% max={max(ttft_ratios):.1f}%"
        )
    print(f"  stats: {hr.stats()}")
    return len(failures), failures


def main() -> int:
    print(f"=== HybridRetrievalCache integration test ===")
    print("Loading mlx-community/Llama-3.2-1B-Instruct-4bit...")
    model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

    # warmup MLX kernels
    from mlx_lm.models.cache import make_prompt_cache
    warm = make_prompt_cache(model)
    _ = model(mx.array([tok.encode("warmup")]), cache=warm)
    mx.eval([c.state for c in warm])

    n_failed = 0
    all_failures: list[str] = []

    # inproc — always run; no server required.
    f, names = run_backend(model, tok, "inproc")
    n_failed += f
    all_failures += [f"inproc::{n}" for n in names]

    # pion — only if server is reachable.
    if _server_up():
        f, names = run_backend(model, tok, "pion")
        n_failed += f
        all_failures += [f"pion::{n}" for n in names]
    else:
        print(f"\n--- backend=pion: SKIPPED (no server on {HOST}:{PORT}) ---")

    if n_failed:
        print(f"\n{n_failed} case(s) failed: {all_failures}")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
