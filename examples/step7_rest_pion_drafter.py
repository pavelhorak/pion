"""
Step 7: REST — Pion as Speculative Drafter (Experiment 13)

Novel approach from Gemini expert consultation (2026-03-24).

Instead of using a smaller LLM to draft tokens (NEST), use Pion corpus
directly as the draft engine.

Key insight: In a grounded RAG setting (document in the model's context),
the model tends to quote/reproduce phrases from that document. Pion retrieves
semantically similar corpus chunks and proposes their token continuations as
speculative drafts. If the model was going to say those tokens anyway →
acceptance is high WITHOUT requiring model memorization.

Difference from NEST:
  NEST oracle:   forced exact match from chunk_map (unrealistic)
  NEST realistic: no document in context, Wikipedia corpus → 2% acceptance
  REST:          document IN context, retrieve from same corpus → ?% acceptance

Two test conditions:
  GROUNDED:   model prompt includes 2-3 paragraphs from the corpus document
              Expected: high acceptance (model reproduces document text)
  UNGROUNDED: open-ended generation, no document in context
              Expected: ~2% (same as NEST realistic — baseline)

Pass criteria: grounded acceptance ≥ 20% (10× better than ungrounded baseline)
"""

import sys, time, os
import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")
from pion_memory import PionMemory
import onnxruntime as ort
from transformers import AutoTokenizer

# ── Embedding ─────────────────────────────────────────────────────────────────

_tok = None
_sess = None

def _load():
    global _tok, _sess
    if _sess is None:
        _tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _sess = ort.InferenceSession(os.environ.get("PION_MINILM_ONNX", "models/all-MiniLM-L6-v2/onnx/model.onnx"),
                                     providers=["CPUExecutionProvider"])

def embed(text: str) -> np.ndarray:
    _load()
    enc = _tok(text, return_tensors="np", padding=True, truncation=True, max_length=128)
    out = _sess.run(None, {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc.get("token_type_ids", np.zeros_like(enc["input_ids"])),
    })
    mask = enc["attention_mask"][:, :, np.newaxis].astype(np.float32)
    emb = (out[0] * mask).sum(1) / mask.sum(1).clip(min=1e-9)
    v = emb[0]
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

# ── Llama Loading ─────────────────────────────────────────────────────────────

def load_llm():
    from llama_cpp import Llama
    gguf = os.path.expanduser(
        "~/.ollama/models/blobs/"
        "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
    )
    print(f"Loading GGUF: {os.path.basename(gguf)} ({os.path.getsize(gguf)/1e9:.1f}GB)...")
    llm = Llama(model_path=gguf, n_ctx=4096, n_threads=8, verbose=False, logits_all=True)
    print("Model loaded.")
    return llm

# ── Corpus: load dev_refrence.md ─────────────────────────────────────────────

def load_corpus(llm, path="doc/architecture.md", max_paragraphs=200):
    with open(path) as f:
        raw = f.read()

    # Split into paragraphs (non-empty, skip markdown headers)
    paragraphs = []
    for p in raw.split("\n\n"):
        p = p.strip()
        if len(p) > 80 and not p.startswith("#") and not p.startswith("```") and not p.startswith("|"):
            paragraphs.append(p)
        if len(paragraphs) >= max_paragraphs:
            break

    print(f"Loaded {len(paragraphs)} paragraphs from corpus.")

    mem = PionMemory(dim=384, max_elements=len(paragraphs) + 10, M=16, ef_construction=64)
    chunk_map = {}  # idx → token list

    print("Embedding & indexing corpus...")
    for i, p in enumerate(tqdm(paragraphs)):
        emb = embed(p)
        mem.remember(i, emb)
        toks = llm.tokenize(p.encode("utf-8"))
        if toks and toks[0] == llm.token_bos():
            toks = toks[1:]
        chunk_map[i] = toks

    mem.optimize()
    print("Corpus ready.")
    return mem, chunk_map, paragraphs

# ── Speculative decoding helpers ──────────────────────────────────────────────

def find_draft_in_chunk(out_tokens, chunk_tokens, max_drafts=24, min_match=3):
    """Find where out_tokens[-k:] matches inside chunk_tokens, return continuation."""
    for match_len in range(min(12, len(out_tokens)), min_match - 1, -1):
        suffix = out_tokens[-match_len:]
        for i in range(len(chunk_tokens) - match_len):
            if chunk_tokens[i:i + match_len] == suffix:
                draft = chunk_tokens[i + match_len: i + match_len + max_drafts]
                if draft:
                    return draft
    return []

def run_rest(llm, mem, chunk_map, prompt_tokens, max_tokens=200, draft_size=24,
             label=""):
    """Generate with REST (Pion-as-drafter) speculative decoding."""
    gen = llm.generate(prompt_tokens, temp=0.0)
    out_tokens = []
    t0 = time.perf_counter()

    # Warm up: generate a few tokens normally before trying REST
    WARMUP = 8
    for _ in range(WARMUP):
        try:
            out_tokens.append(next(gen))
        except StopIteration:
            break

    drafts = []
    drafts_proposed = 0
    drafts_accepted = 0
    total_retrievals = 0

    while len(out_tokens) < max_tokens:
        # Always retrieve fresh when draft queue empty
        if not drafts:
            recent_text = llm.detokenize(out_tokens[-48:]).decode("utf-8", errors="ignore")
            emb = embed(recent_text)
            total_retrievals += 1
            results = mem.recall(emb, k=5, ef=32)

            for best_id, _ in results:
                chunk_toks = chunk_map[best_id]
                candidate = find_draft_in_chunk(out_tokens, chunk_toks,
                                                max_drafts=draft_size)
                if candidate:
                    drafts = candidate
                    break

        if drafts:
            drafts_proposed += 1
            try:
                t = gen.send(drafts)
            except StopIteration:
                break

            out_tokens.append(t)
            if t == drafts[0]:
                # First draft token accepted — try to verify more
                drafts_accepted += 1
                for i in range(1, len(drafts)):
                    if len(out_tokens) >= max_tokens:
                        break
                    try:
                        t = next(gen)
                    except StopIteration:
                        break
                    out_tokens.append(t)
                    if t == drafts[i]:
                        drafts_accepted += 1
                    else:
                        break
            drafts = []
        else:
            # No draft found — generate normally, probe every 4 tokens
            for _ in range(4):
                if len(out_tokens) >= max_tokens:
                    break
                try:
                    out_tokens.append(next(gen))
                except StopIteration:
                    break

    gen_time = time.perf_counter() - t0
    return out_tokens, gen_time, drafts_proposed, drafts_accepted, total_retrievals

def run_normal(llm, prompt_tokens, max_tokens=200):
    """Baseline: no speculative decoding."""
    gen = llm.generate(prompt_tokens, temp=0.0)
    out_tokens = []
    t0 = time.perf_counter()
    try:
        for _ in range(max_tokens):
            out_tokens.append(next(gen))
    except StopIteration:
        pass
    return out_tokens, time.perf_counter() - t0

# ── Main ──────────────────────────────────────────────────────────────────────

def run_condition(llm, mem, chunk_map, paragraphs, prompts, label, max_tokens=150):
    print(f"\n{'='*60}")
    print(f"CONDITION: {label}")
    print(f"{'='*60}")

    total_proposed = 0
    total_accepted = 0
    total_retrievals = 0
    speedups = []

    for i, (prompt_text, description) in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {description[:60]}")
        prompt_tokens = llm.tokenize(prompt_text.encode("utf-8"))

        # Baseline
        norm_toks, norm_time = run_normal(llm, prompt_tokens, max_tokens=max_tokens)
        norm_speed = len(norm_toks) / norm_time

        # REST
        rest_toks, rest_time, proposed, accepted, retrievals = run_rest(
            llm, mem, chunk_map, prompt_tokens, max_tokens=max_tokens)
        rest_speed = len(rest_toks) / rest_time

        speedup = rest_speed / norm_speed
        accept_rate = (accepted / proposed * 100) if proposed > 0 else 0.0

        total_proposed += proposed
        total_accepted += accepted
        total_retrievals += retrievals
        speedups.append(speedup)

        print(f"  Normal: {norm_speed:.1f} tok/s  |  REST: {rest_speed:.1f} tok/s  "
              f"|  speedup: {speedup:.2f}x")
        print(f"  Drafts proposed: {proposed}  accepted: {accepted}  "
              f"({accept_rate:.1f}%)  retrievals: {retrievals}")

    overall_accept = (total_accepted / total_proposed * 100) if total_proposed > 0 else 0.0
    mean_speedup = sum(speedups) / len(speedups)
    print(f"\n{label} SUMMARY:")
    print(f"  Overall acceptance rate: {overall_accept:.1f}%")
    print(f"  Mean speedup: {mean_speedup:.2f}x")
    print(f"  Total drafts: {total_proposed} proposed, {total_accepted} accepted")

    return overall_accept, mean_speedup

def main():
    _load()
    llm = load_llm()
    mem, chunk_map, paragraphs = load_corpus(llm)

    # ── GROUNDED prompts ──────────────────────────────────────────────────────
    # Each prompt directly includes a paragraph from the corpus.
    # The model should continue with text that is IN the corpus.
    grounded_prompts = []
    for idx in [5, 15, 25, 40, 55]:
        if idx < len(paragraphs):
            p = paragraphs[idx]
            words = p.split()
            # Give first ~40 words as prompt — model should continue from corpus
            prompt_prefix = " ".join(words[:40])
            grounded_prompts.append(
                (prompt_prefix, f"Corpus para #{idx}: {prompt_prefix[:50]}...")
            )

    # ── UNGROUNDED prompts ────────────────────────────────────────────────────
    # Open-ended generation with no document context.
    # Baseline: should behave like NEST realistic (~2% acceptance).
    ungrounded_prompts = [
        ("The history of ancient Rome began with", "Open: Rome history"),
        ("Machine learning algorithms can be categorized into", "Open: ML categories"),
        ("The economic impact of renewable energy includes", "Open: renewable energy"),
        ("In quantum mechanics, the wave function describes", "Open: quantum mechanics"),
        ("The process of photosynthesis converts", "Open: photosynthesis"),
    ]

    grounded_accept, grounded_speedup = run_condition(
        llm, mem, chunk_map, paragraphs, grounded_prompts, "GROUNDED (doc in context)")

    ungrounded_accept, ungrounded_speedup = run_condition(
        llm, mem, chunk_map, paragraphs, ungrounded_prompts, "UNGROUNDED (no doc context)")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXPERIMENT 13 — REST FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Grounded acceptance:   {grounded_accept:.1f}%  speedup: {grounded_speedup:.2f}x")
    print(f"Ungrounded acceptance: {ungrounded_accept:.1f}%  speedup: {ungrounded_speedup:.2f}x")
    print(f"Lift from grounding:   {grounded_accept - ungrounded_accept:+.1f}pp")
    print()

    if grounded_accept >= 20:
        print("VERDICT: PASS — REST is viable for grounded RAG generation")
        print("  => FLARE Gateway with REST drafter is the production architecture")
    elif grounded_accept >= 10:
        print("VERDICT: MARGINAL — some lift from grounding, needs larger corpus or domain tuning")
    else:
        print("VERDICT: FAIL — grounding does not increase acceptance rate")
        print("  => REST not viable; FLARE (without speculative decoding) remains the path")

if __name__ == "__main__":
    main()
