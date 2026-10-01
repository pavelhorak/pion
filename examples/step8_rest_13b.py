"""
Step 8: REST Experiment 13b — Speedup Confirmation

Fixes two critical limitations of step7 (Exp 13):
  1. Corpus size: 74 paragraphs → 1,000+ (all repo .md files)
  2. Batch verification: sequential gen.send() → TRUE one-forward-pass batch verify

True batch verify: instead of k separate llm.forward() calls to check k draft tokens,
run eval(draft_tokens) ONCE and read logits for all positions from llm.scores.
At k=16 drafts and 50% acceptance: 1 forward pass yields ~9 new tokens vs 9 sequential passes.

Pass criteria (Exp 13b):
  - Grounded acceptance rate ≥ 30%
  - Mean speedup ≥ 1.5×
"""

import os
import sys, os, glob, time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")
from pion_memory import PionMemory
import onnxruntime as ort
from transformers import AutoTokenizer


# Directories to skip when scanning the tree. Was a hardcoded private dir
# name; a public checkout has no such directory, and the list is now
# extensible via PION_SCAN_EXCLUDE (colon-separated).
_EXCLUDED_DIRS = [d for d in os.environ.get(
    "PION_SCAN_EXCLUDE", "build:dist:.git:node_modules").split(":") if d]


def _in_excluded_dir(path):
    return any(("/" + d + "/") in path or path.startswith(d + "/")
               for d in _EXCLUDED_DIRS)


# ── Embedding (MiniLM-L6-v2 ONNX, 384-dim) ──────────────────────────────────

_tok = None
_sess = None

def _load_embedder():
    global _tok, _sess
    if _sess is None:
        _tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _sess = ort.InferenceSession(
            os.environ.get("PION_MINILM_ONNX", "models/all-MiniLM-L6-v2/onnx/model.onnx"),
            providers=["CPUExecutionProvider"],
        )

def embed(text: str) -> np.ndarray:
    _load_embedder()
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

# ── LLM ──────────────────────────────────────────────────────────────────────

def load_llm():
    from llama_cpp import Llama
    gguf = os.path.expanduser(
        "~/.ollama/models/blobs/"
        "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
    )
    print(f"Loading GGUF ({os.path.getsize(gguf)/1e9:.1f}GB)...")
    llm = Llama(
        model_path=gguf,
        n_ctx=4096,
        n_threads=8,
        verbose=False,
        logits_all=True,   # Required for batch verify — all position scores available
    )
    print("Model loaded.")
    return llm

# ── Corpus: all .md files in the repo ────────────────────────────────────────

def load_corpus(llm, root=".", target_paragraphs=1200):
    """Load paragraphs from all .md files under root.

    Returns (PionMemory, chunk_map, paragraphs).
    chunk_map: {idx: token_list} for speculative matching.
    """
    _load_embedder()

    # Collect all .md files
    md_files = sorted(glob.glob(os.path.join(root, "**/*.md"), recursive=True))
    # Exclude generated/benchmark result files (mostly tables, not prose)
    exclude = {"benchmark_results.md", "winning_benchmarks.md"}
    md_files = [f for f in md_files if os.path.basename(f) not in exclude
                and not _in_excluded_dir(f)]

    print(f"Found {len(md_files)} markdown files.")

    paragraphs = []
    sources = []

    for path in md_files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except OSError:
            continue

        for p in raw.split("\n\n"):
            p = p.strip()
            # Keep substantive prose paragraphs only
            if (len(p) > 80
                    and not p.startswith("#")
                    and not p.startswith("```")
                    and not p.startswith("|")
                    and not p.startswith("-  ")
                    and p.count("\n") < 8):       # skip big code/table blocks
                paragraphs.append(p)
                sources.append(os.path.relpath(path, root))

        if len(paragraphs) >= target_paragraphs:
            break

    print(f"Extracted {len(paragraphs)} paragraphs "
          f"(target {target_paragraphs}) from {len(set(sources))} files.")

    mem = PionMemory(dim=384, max_elements=len(paragraphs) + 32, M=16, ef_construction=64)
    chunk_map = {}

    print("Embedding & indexing corpus (this takes ~1–2 min)...")
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

# ── Draft finding ─────────────────────────────────────────────────────────────

def find_draft(out_tokens, chunk_tokens, max_drafts=16, min_match=3):
    """Scan chunk_tokens for out_tokens[-k:] suffix; return continuation."""
    for match_len in range(min(12, len(out_tokens)), min_match - 1, -1):
        suffix = out_tokens[-match_len:]
        for i in range(len(chunk_tokens) - match_len):
            if chunk_tokens[i:i + match_len] == suffix:
                draft = chunk_tokens[i + match_len: i + match_len + max_drafts]
                if draft:
                    return list(draft)
    return []

# ── TRUE batch verification ───────────────────────────────────────────────────

def batch_verify(llm, draft_tokens):
    """One forward pass to verify all draft_tokens.

    Requires model was loaded with logits_all=True.

    Approach:
      - scores[n_before - 1] already has the prediction for draft_tokens[0]
        (computed in the previous generation step — FREE)
      - eval(draft_tokens) runs ONE forward pass for all k tokens
      - scores[n_before + i - 1] = prediction for draft_tokens[i], i ≥ 1

    Returns:
      (accepted_count, new_tokens)
      new_tokens = draft_tokens[:accepted_count] + [correction_token]
    """
    n_before = llm.n_tokens
    state = llm.save_state()

    # Prediction for d[0] — already computed, no cost
    pred_d0 = int(np.argmax(llm.scores[n_before - 1]))

    # One forward pass for all drafts
    llm.eval(draft_tokens)

    # Verify each draft token greedily
    accepted = 0

    if pred_d0 == draft_tokens[0]:
        accepted = 1
        for i in range(1, len(draft_tokens)):
            # scores[n_before + i - 1] was written by eval() at position n_before + (i-1)
            predicted = int(np.argmax(llm.scores[n_before + i - 1]))
            if predicted == draft_tokens[i]:
                accepted += 1
            else:
                break

    # Build the list of tokens to accept
    if accepted < len(draft_tokens):
        # First rejection is at position `accepted`
        if accepted == 0:
            correction = pred_d0  # already have this
        else:
            correction = int(np.argmax(llm.scores[n_before + accepted - 1]))
        new_tokens = list(draft_tokens[:accepted]) + [correction]

        # Roll back KV cache and replay accepted tokens + correction
        llm.load_state(state)
        if accepted > 0:
            llm.eval(draft_tokens[:accepted])
        llm.eval([correction])
    else:
        # All accepted — bonus token from the last scores position
        bonus = int(np.argmax(llm.scores[n_before + len(draft_tokens) - 1]))
        new_tokens = list(draft_tokens) + [bonus]
        llm.eval([bonus])

    return accepted, new_tokens

# ── REST generation ───────────────────────────────────────────────────────────

def run_rest(llm, mem, chunk_map, prompt_tokens, max_tokens=200, draft_size=16, label=""):
    """Generate with REST + true batch verification."""

    # Prime the KV cache with prompt
    llm.reset()
    llm.eval(prompt_tokens)

    out_tokens = []
    t0 = time.perf_counter()

    # Warmup: 8 greedy tokens (fill KV cache, stabilise generation direction)
    WARMUP = 8
    for _ in range(WARMUP):
        if len(out_tokens) >= max_tokens:
            break
        next_tok = int(np.argmax(llm.scores[llm.n_tokens - 1]))
        llm.eval([next_tok])
        out_tokens.append(next_tok)

    batches_attempted = 0
    drafts_accepted = 0
    drafts_proposed = 0
    total_retrievals = 0

    while len(out_tokens) < max_tokens:
        # Retrieve similar chunk from Pion
        recent_text = llm.detokenize(out_tokens[-48:]).decode("utf-8", errors="ignore")
        q_emb = embed(recent_text)
        total_retrievals += 1
        results = mem.recall(q_emb, k=5, ef=32)

        draft = []
        for best_id, _ in results:
            d = find_draft(out_tokens, chunk_map[best_id], max_drafts=draft_size)
            if d:
                draft = d
                break

        if draft:
            batches_attempted += 1
            drafts_proposed += len(draft)
            accepted, new_toks = batch_verify(llm, draft)
            drafts_accepted += accepted
            out_tokens.extend(new_toks)
        else:
            # No draft match — generate 4 tokens normally (probe often)
            for _ in range(4):
                if len(out_tokens) >= max_tokens:
                    break
                next_tok = int(np.argmax(llm.scores[llm.n_tokens - 1]))
                llm.eval([next_tok])
                out_tokens.append(next_tok)

    gen_time = time.perf_counter() - t0
    return out_tokens, gen_time, drafts_proposed, drafts_accepted, total_retrievals, batches_attempted

def run_normal(llm, prompt_tokens, max_tokens=200):
    """Baseline: pure autoregressive generation (no speculative decoding)."""
    llm.reset()
    llm.eval(prompt_tokens)

    out_tokens = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        next_tok = int(np.argmax(llm.scores[llm.n_tokens - 1]))
        llm.eval([next_tok])
        out_tokens.append(next_tok)
        if next_tok == llm.token_eos():
            break

    return out_tokens, time.perf_counter() - t0

# ── Experiment runner ─────────────────────────────────────────────────────────

def run_condition(llm, mem, chunk_map, prompts, label, max_tokens=150):
    print(f"\n{'='*60}")
    print(f"CONDITION: {label}")
    print(f"{'='*60}")

    total_proposed = 0
    total_accepted = 0
    total_batches = 0
    speedups = []

    for i, (prompt_text, description) in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {description[:70]}")
        prompt_tokens = llm.tokenize(prompt_text.encode("utf-8"))
        if prompt_tokens and prompt_tokens[0] == llm.token_bos():
            prompt_tokens = prompt_tokens[1:]

        # Baseline
        norm_toks, norm_time = run_normal(llm, prompt_tokens, max_tokens=max_tokens)
        norm_speed = len(norm_toks) / norm_time

        # REST (true batch verify)
        rest_toks, rest_time, proposed, accepted, retrievals, batches = run_rest(
            llm, mem, chunk_map, prompt_tokens, max_tokens=max_tokens,
            draft_size=16, label=label)
        rest_speed = len(rest_toks) / rest_time

        speedup = rest_speed / norm_speed
        accept_rate = (accepted / proposed * 100) if proposed > 0 else 0.0
        toks_per_batch = (accepted / batches) if batches > 0 else 0.0

        total_proposed += proposed
        total_accepted += accepted
        total_batches += batches
        speedups.append(speedup)

        print(f"  Normal:  {norm_speed:.1f} tok/s   REST: {rest_speed:.1f} tok/s"
              f"   speedup: {speedup:.2f}×")
        print(f"  Batches: {batches}  proposed: {proposed}  accepted: {accepted}"
              f"   ({accept_rate:.1f}%)   tok/batch: {toks_per_batch:.1f}"
              f"   retrievals: {retrievals}")

    overall_accept = (total_accepted / total_proposed * 100) if total_proposed > 0 else 0.0
    mean_speedup = sum(speedups) / len(speedups) if speedups else 0.0
    avg_toks_per_batch = (total_accepted / total_batches) if total_batches > 0 else 0.0

    print(f"\n{label} SUMMARY:")
    print(f"  Overall acceptance rate:  {overall_accept:.1f}%")
    print(f"  Mean speedup:             {mean_speedup:.2f}×")
    print(f"  Avg accepted/batch:       {avg_toks_per_batch:.1f}")

    return overall_accept, mean_speedup

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _load_embedder()
    llm = load_llm()
    mem, chunk_map, paragraphs = load_corpus(llm, root=".", target_paragraphs=1200)

    print(f"\nCorpus size: {len(paragraphs)} paragraphs")

    # ── GROUNDED prompts ──────────────────────────────────────────────────────
    # Prompt = first ~40 words of a corpus paragraph.
    # Model should continue using the document language → high acceptance rate.
    grounded_prompts = []
    step = max(1, len(paragraphs) // 10)
    for idx in range(0, min(len(paragraphs), step * 10), step):
        p = paragraphs[idx]
        words = p.split()
        if len(words) < 20:
            continue
        prefix = " ".join(words[:40])
        grounded_prompts.append(
            (prefix, f"Para #{idx}: {prefix[:60]}...")
        )
        if len(grounded_prompts) >= 10:
            break

    # ── UNGROUNDED prompts ────────────────────────────────────────────────────
    # Open-ended — model has no document context; should behave like NEST (~2%).
    ungrounded_prompts = [
        ("The history of ancient Rome began with the founding of the city",
         "Open: Roman history"),
        ("Machine learning algorithms can be broadly categorized into supervised",
         "Open: ML categories"),
        ("The economic impact of renewable energy on global electricity markets",
         "Open: renewable energy"),
        ("In quantum mechanics, the uncertainty principle states that",
         "Open: quantum mechanics"),
        ("The process of photosynthesis converts sunlight into chemical energy",
         "Open: photosynthesis"),
    ]

    grounded_accept, grounded_speedup = run_condition(
        llm, mem, chunk_map, grounded_prompts, "GROUNDED", max_tokens=150)

    ungrounded_accept, ungrounded_speedup = run_condition(
        llm, mem, chunk_map, ungrounded_prompts, "UNGROUNDED", max_tokens=150)

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXPERIMENT 13b — REST FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Corpus:                {len(paragraphs)} paragraphs")
    print(f"Grounded acceptance:   {grounded_accept:.1f}%   speedup: {grounded_speedup:.2f}×")
    print(f"Ungrounded acceptance: {ungrounded_accept:.1f}%   speedup: {ungrounded_speedup:.2f}×")
    print(f"Lift from grounding:   {grounded_accept - ungrounded_accept:+.1f}pp")
    print()

    if grounded_accept >= 30 and grounded_speedup >= 1.5:
        print("VERDICT: PASS — REST viable, speedup confirmed")
        print("  Gate: Blog post claim 'Pion as speculative drafter' is publishable")
    elif grounded_accept >= 20:
        print("VERDICT: MARGINAL — acceptance ok but speedup below 1.5×")
        print("  Consider: larger draft_size, domain-tuned corpus, or more tokens")
    elif grounded_speedup >= 1.5:
        print("VERDICT: MARGINAL — speedup ok but acceptance below 30%")
    else:
        print("VERDICT: FAIL — REST not viable at this scale")
        print("  Next: Exp 14 kNN-LM (per-token logit injection via llama.cpp logits_all)")

if __name__ == "__main__":
    main()
