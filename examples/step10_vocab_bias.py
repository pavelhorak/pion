"""
Step 10: Dynamic Vocabulary Biasing (Experiment 15) — Token-Level RAG

Gemini expert recommendation after Exp 13b (REST) and Exp 14 (kNN-LM) failures.

Insight: Both failures stem from the same mismatch — we tried to map semantic
retrieval directly to next-token prediction. The fix: use retrieval to shape the
LOGIT SPACE (vocabulary preference), not predict specific next tokens.

Mechanism:
  Every N tokens during generation:
    1. Embed last 64 tokens → MiniLM → 384-dim (1.28ms)
    2. Retrieve top-k semantically similar corpus paragraphs from Pion
    3. Tokenize all retrieved text → build a set of relevant token IDs
    4. Add +β to logits of those token IDs before sampling/argmax

Why this works where kNN-LM failed:
  - kNN-LM: "what exact token comes next" → needs hidden states
  - Vocab biasing: "what words are relevant to this context" → semantic match is enough
  - Retrieved paragraph about "HNSW M=16" → boosts probability of tokens like "16",
    "parameter", "graph", "neighbors" → model still paraphrases but uses right vocabulary

Why this works where REST failed:
  - REST: required verbatim reproduction
  - Vocab biasing: model generates freely, just with boosted domain vocabulary
  - No save/load overhead: purely additive forward pass, no rollbacks

Evaluation:
  A. Perplexity on held-out corpus (β sweep: 0, 0.5, 1.0, 2.0, 3.0)
  B. Domain factual accuracy: ask 10 Pion-specific questions, score whether
     the correct answer token(s) appear in the first 20 generated tokens
  C. Retrieval frequency: measure % of vocab-bias updates that include the
     correct answer token in the retrieved token set

Pass criteria:
  - PPL improvement ≥ 2%  OR  factual accuracy improvement ≥ 2 correct/10
"""

import os
import sys, os, glob, math, time
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


# ── Embedding ─────────────────────────────────────────────────────────────────

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
    llm = Llama(model_path=gguf, n_ctx=2048, n_threads=8, verbose=False,
                logits_all=True)
    print("Model loaded.")
    return llm

# ── Corpus + Pion index ───────────────────────────────────────────────────────

def load_corpus_and_index(root=".", n_train=150, n_test=30):
    md_files = sorted(glob.glob(os.path.join(root, "**/*.md"), recursive=True))
    exclude = {"benchmark_results.md", "winning_benchmarks.md"}
    md_files = [f for f in md_files
                if os.path.basename(f) not in exclude and not _in_excluded_dir(f)]

    paragraphs = []
    for path in md_files:
        try:
            raw = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for p in raw.split("\n\n"):
            p = p.strip()
            if len(p) > 100 and not p.startswith("#") and not p.startswith("```") \
                    and not p.startswith("|") and p.count("\n") < 6:
                paragraphs.append(p)
        if len(paragraphs) >= n_train + n_test:
            break

    train = paragraphs[:n_train]
    test  = paragraphs[n_train:n_train + n_test]

    # Build Pion index: paragraph_id → paragraph_embedding
    print(f"Indexing {len(train)} train paragraphs...")
    mem = PionMemory(dim=384, max_elements=len(train) + 32, M=16, ef_construction=64)
    for i, p in enumerate(tqdm(train, desc="  Embedding")):
        mem.remember(i, embed(p))
    mem.optimize()
    print("Index ready.")
    return mem, train, test

# ── Vocab bias helpers ────────────────────────────────────────────────────────

def get_bias_token_set(llm, mem, train_paras, q_emb, k=3, ef=32):
    """Retrieve k paragraphs and return the set of all their token IDs."""
    results = mem.recall(q_emb, k=k, ef=ef)
    token_set = set()
    for idx, _ in results:
        para = train_paras[idx % len(train_paras)]
        toks = llm.tokenize(para.encode("utf-8"))
        token_set.update(toks)
    return token_set

def apply_bias(logits: np.ndarray, token_set: set, beta: float) -> np.ndarray:
    """Add +beta to logits of all tokens in token_set."""
    if not token_set or beta == 0.0:
        return logits
    biased = logits.copy()
    ids = np.array([t for t in token_set if 0 <= t < len(logits)], dtype=np.int64)
    if ids.size:
        biased[ids] += beta
    return biased

# ── Perplexity evaluation ─────────────────────────────────────────────────────

def eval_perplexity(llm, mem, train_paras, test_paras, betas,
                    every_n=8, k=3, max_para=20):
    """Compute perplexity on test paragraphs for each beta value.

    every_n: refresh the vocabulary bias set every N token positions.
    """
    _load_embedder()
    log_probs = {b: [] for b in betas}
    retrieve_times = []

    for para in tqdm(test_paras[:max_para], desc="  Evaluating PPL"):
        toks = llm.tokenize(para.encode("utf-8"))
        if toks and toks[0] == llm.token_bos():
            toks = toks[1:]
        if len(toks) < 74:   # need 64 ctx window + a few target tokens
            continue

        # One forward pass for all positions
        llm.reset()
        llm.eval(toks)

        ctx_window = 64
        bias_token_set = set()

        for i in range(ctx_window, len(toks) - 1):
            true_next = toks[i + 1]
            raw_logits = llm.scores[i].copy()

            # Refresh bias token set every N positions
            if (i - ctx_window) % every_n == 0:
                t0 = time.perf_counter()
                ctx_text = llm.detokenize(toks[i - ctx_window + 1: i + 1]).decode(
                    "utf-8", errors="ignore")
                q_emb = embed(ctx_text)
                bias_token_set = get_bias_token_set(llm, mem, train_paras, q_emb, k=k)
                retrieve_times.append(time.perf_counter() - t0)

            for beta in betas:
                biased = apply_bias(raw_logits, bias_token_set, beta)
                # Stable softmax
                biased -= biased.max()
                p = np.exp(biased)
                p /= p.sum()
                p_true = float(p[true_next])
                if p_true > 0:
                    log_probs[beta].append(math.log(p_true))

    ppls = {}
    for b in betas:
        lps = log_probs[b]
        ppls[b] = math.exp(-sum(lps) / len(lps)) if lps else float("inf")

    avg_ms = np.mean(retrieve_times) * 1000 if retrieve_times else 0
    n_updates = len(retrieve_times)
    print(f"  Retrieval updates: {n_updates}  avg {avg_ms:.2f}ms  "
          f"(tokens evaluated: {len(log_probs[betas[0]])})")
    return ppls

# ── Domain factual accuracy ───────────────────────────────────────────────────

# (prompt, list_of_acceptable_answer_substrings, description)
FACTUAL_QA = [
    ("What is Pion's peak queries per second on Apple M4?",
     ["8,134", "8134", "8134 QPS", "8,134 QPS"], "macOS peak QPS"),

    ("What is the default port number for Pion?",
     ["1974"], "Pion port"),

    ("What is the HNSW M parameter in Pion's default configuration?",
     ["16", "M=16", "M = 16"], "HNSW M parameter"),

    ("How large is each WAL ring buffer in Pion?",
     ["256 MB", "256MB", "256 megabyte"], "WAL size"),

    ("What is Pion's retrieval latency including embedding?",
     ["1.28", "1.28ms", "1.28 ms"], "retrieval latency"),

    ("What recall@100 does Pion achieve at ef=150?",
     ["0.9371", "93.71", "0.937"], "recall@100"),

    ("What is the default ef_construction parameter for HNSW?",
     ["128", "ef_construction=128"], "ef_construction"),

    ("How many workers does Pion run on a desktop machine?",
     ["8", "eight workers", "8 workers"], "num workers"),

    ("What is the maximum HNSW level (max_level) in Pion?",
     ["6", "max_level=6", "max level of 6"], "max_level"),

    ("What embedding model does Pion use by default?",
     ["nomic-embed-text", "nomic", "MiniLM"], "default embed model"),
]

def eval_factual(llm, mem, train_paras, betas, max_gen_tokens=60, k=3):
    """For each question, generate up to max_gen_tokens. Score whether any
    acceptable answer string appears in the generated text."""
    _load_embedder()

    # System prompt + few-shot for concise answers (reuse Exp 12b style)
    FEW_SHOT = (
        "Answer factual questions concisely with a short phrase or number.\n"
        "Q: What year was Redis released? A: 2009\n"
        "Q: What data structure does HNSW stand for? A: Hierarchical Navigable Small World\n"
    )

    results = {b: [] for b in betas}

    print("\n  Factual QA:")
    for prompt_text, answers, desc in FACTUAL_QA:
        full_prompt = FEW_SHOT + f"Q: {prompt_text} A:"
        prompt_toks = llm.tokenize(full_prompt.encode("utf-8"))
        if prompt_toks and prompt_toks[0] == llm.token_bos():
            prompt_toks = prompt_toks[1:]

        # Get vocabulary bias from the question itself
        q_emb = embed(prompt_text)
        bias_token_set = get_bias_token_set(llm, mem, train_paras, q_emb, k=k)

        for beta in betas:
            llm.reset()
            llm.eval(prompt_toks)
            out_toks = []
            for _ in range(max_gen_tokens):
                raw = llm.scores[llm.n_tokens - 1].copy()
                biased = apply_bias(raw, bias_token_set, beta)
                next_tok = int(np.argmax(biased))
                if next_tok == llm.token_eos():
                    break
                llm.eval([next_tok])
                out_toks.append(next_tok)

            generated = llm.detokenize(out_toks).decode("utf-8", errors="ignore").strip()
            correct = any(ans.lower() in generated.lower() for ans in answers)
            results[beta].append(correct)

            if beta == betas[0] or beta == betas[-1]:   # print baseline and best beta
                mark = "✓" if correct else "✗"
                print(f"  β={beta:.1f} {mark} [{desc}]: {generated[:60]!r}")

    scores = {b: sum(results[b]) for b in betas}
    return scores, results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _load_embedder()
    llm = load_llm()
    mem, train_paras, test_paras = load_corpus_and_index(root=".", n_train=150, n_test=30)

    BETAS = [0.0, 0.5, 1.0, 2.0, 3.0]

    # ── A: Perplexity ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("A. PERPLEXITY on held-out test paragraphs")
    print("   (bias refreshed every 8 tokens, k=3 retrieved paragraphs)")
    print("=" * 60)
    ppls = eval_perplexity(llm, mem, train_paras, test_paras,
                           betas=BETAS, every_n=8, k=3, max_para=20)

    best_beta = min(BETAS, key=lambda b: ppls[b])
    baseline_ppl = ppls[0.0]
    print(f"\n  β=0.0 (baseline): {baseline_ppl:.2f}")
    for b in BETAS[1:]:
        delta = (ppls[b] - baseline_ppl) / baseline_ppl * 100
        mark = " ✓ IMPROVE" if delta < -2 else (" ✗ WORSE" if delta > 2 else " ~ neutral")
        print(f"  β={b:.1f}:           {ppls[b]:.2f}  ({delta:+.1f}%){mark}")
    ppl_improvement = (baseline_ppl - ppls[best_beta]) / baseline_ppl * 100

    # ── B: Factual accuracy ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("B. DOMAIN FACTUAL ACCURACY (10 Pion-specific questions)")
    print("=" * 60)
    scores, _ = eval_factual(llm, mem, train_paras, betas=BETAS, k=3)

    print("\n  Summary:")
    for b in BETAS:
        mark = " ✓" if scores[b] > scores[0.0] else (" =" if scores[b] == scores[0.0] else " ✗")
        print(f"  β={b:.1f}: {scores[b]}/10 correct{mark}")
    fact_gain = scores[best_beta] - scores[0.0]

    # ── Verdict ───────────────────────────────────────────────────────────────
    ppl_pass  = ppl_improvement >= 2.0
    fact_pass = fact_gain >= 2

    print("\n" + "=" * 60)
    print("EXPERIMENT 15 — VOCAB BIASING FINAL RESULTS")
    print("=" * 60)
    print(f"Best β:              {best_beta}")
    print(f"PPL improvement:     {ppl_improvement:.1f}%  (baseline {baseline_ppl:.2f} → best {ppls[best_beta]:.2f})")
    print(f"Factual gain:        {fact_gain:+d}/10  (baseline {scores[0.0]}/10 → best {scores[best_beta]}/10)")
    print()

    if ppl_pass or fact_pass:
        print("VERDICT: PASS")
        if ppl_pass:
            print(f"  ✓ PPL improved {ppl_improvement:.1f}% at β={best_beta}")
        if fact_pass:
            print(f"  ✓ Factual accuracy +{fact_gain} correct answers at β={best_beta}")
        print("  Claim: 'Retrieved vocabulary boosts domain factual accuracy'")
        print("  Next: run full HotpotQA comparison vs FLARE baseline (F1=0.35)")
    else:
        print("VERDICT: FAIL")
        print(f"  PPL change: {ppl_improvement:.1f}% (need ≥2%)")
        print(f"  Factual gain: {fact_gain}/10 (need ≥2)")
        if ppl_improvement < -2:
            print("  Vocab bias hurts perplexity — retrieved tokens are off-domain")
            print("  → Try Approach 3: Look-Ahead FLARE (embed hallucination, not prefix)")
        else:
            print("  → Neutral effect; try larger β or different k")

if __name__ == "__main__":
    main()
