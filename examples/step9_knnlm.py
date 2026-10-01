"""
Step 9: kNN-LM Experiment 14 — Per-Token Logit Augmentation

Hypothesis: Pion's 1.28ms retrieval makes interactive kNN-LM viable.
Prior work (Khandelwal et al. 2019): FAISS kNN-LM requires 50ms/token →
impractical for interactive generation. Pion at 1.28ms = 2.5% overhead per
token on 8B.

Mechanism:
  At each generation step:
    1. Embed current context (last 64 tokens → MiniLM → 384-dim)
    2. Retrieve k nearest (context_embedding, next_token) pairs from Pion (1.28ms)
    3. Build P_kNN = softmax(-dist / T) distributed over retrieved next tokens
    4. Interpolate: P_final = (1-λ) * P_LM + λ * P_kNN
    5. Sample / compute log-probability from P_final

Evaluation:
  A. Perplexity on held-out corpus paragraphs
     - Lower PPL with kNN augmentation → distribution improvement
  B. Factual probability probes
     - P(correct_domain_token | context) with vs without kNN
     - Tests whether retrieved domain knowledge boosts specific correct tokens
  C. λ sweep: 0.0 (baseline), 0.1, 0.25, 0.5

Pass criteria (Exp 14):
  - Perplexity improvement ≥ 2% on held-out corpus
  OR
  - Factual probability boost ≥ 2× on domain-specific probes
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


# ── Embedding (MiniLM-L6-v2, 384-dim) ────────────────────────────────────────

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
    llm = Llama(
        model_path=gguf,
        n_ctx=2048,
        n_threads=8,
        verbose=False,
        logits_all=True,
    )
    print("Model loaded.")
    return llm

# ── Corpus loading ────────────────────────────────────────────────────────────

def load_corpus(root=".", n_train=150, n_test=30):
    """Load paragraphs; return (train_paragraphs, test_paragraphs)."""
    md_files = sorted(glob.glob(os.path.join(root, "**/*.md"), recursive=True))
    exclude = {"benchmark_results.md", "winning_benchmarks.md"}
    md_files = [f for f in md_files if os.path.basename(f) not in exclude
                and not _in_excluded_dir(f)]

    paragraphs = []
    for path in md_files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except OSError:
            continue
        for p in raw.split("\n\n"):
            p = p.strip()
            if (len(p) > 100
                    and not p.startswith("#")
                    and not p.startswith("```")
                    and not p.startswith("|")
                    and p.count("\n") < 6):
                paragraphs.append(p)
        if len(paragraphs) >= n_train + n_test:
            break

    train = paragraphs[:n_train]
    test  = paragraphs[n_train:n_train + n_test]
    print(f"Corpus: {len(train)} train, {len(test)} test paragraphs.")
    return train, test

# ── Datastore construction ────────────────────────────────────────────────────

def build_datastore(llm, train_paragraphs, stride=4, ctx_window=64):
    """Build (context_embedding, next_token_id) datastore.

    stride=4 means we take a snapshot every 4 tokens. For 150 paragraphs
    × ~100 tokens avg / 4 = ~3,750 entries.
    """
    _load_embedder()

    entries = []   # list of (embedding, next_token_id)

    print(f"Building datastore (stride={stride}, ctx={ctx_window} tokens)...")
    for para in tqdm(train_paragraphs, desc="  Encoding paragraphs"):
        toks = llm.tokenize(para.encode("utf-8"))
        if toks and toks[0] == llm.token_bos():
            toks = toks[1:]
        if len(toks) < 8:   # skip very short paragraphs
            continue

        # Take a snapshot every `stride` tokens (excluding the last token
        # which would require the next paragraph as target)
        for i in range(ctx_window, len(toks) - 1, stride):
            ctx_toks = toks[i - ctx_window : i]
            next_tok = toks[i]
            ctx_text = llm.detokenize(ctx_toks).decode("utf-8", errors="ignore")
            emb = embed(ctx_text)
            entries.append((emb, next_tok))

    print(f"Datastore: {len(entries)} (context, next_token) entries.")

    mem = PionMemory(dim=384, max_elements=len(entries) + 64, M=16, ef_construction=64)
    id_to_token = {}   # hnsw_id → next_token_id

    for idx, (emb, tok) in enumerate(entries):
        mem.remember(idx, emb)
        id_to_token[idx] = tok

    mem.optimize()
    print("Datastore indexed.")
    return mem, id_to_token

# ── kNN distribution ──────────────────────────────────────────────────────────

def knn_dist(mem, id_to_token, q_emb, vocab_size, k=8, T=10.0):
    """Retrieve k nearest neighbours and return sparse P_kNN over vocabulary."""
    results = mem.recall(q_emb, k=k, ef=64)
    if not results:
        return np.zeros(vocab_size, dtype=np.float32)

    # Compute weights: softmax(-dist / T)
    dists = np.array([d for _, d in results], dtype=np.float32)
    weights = np.exp(-dists / T)
    weights /= weights.sum() + 1e-12

    p_knn = np.zeros(vocab_size, dtype=np.float32)
    for (idx, _), w in zip(results, weights):
        tok = id_to_token.get(idx, -1)
        if 0 <= tok < vocab_size:
            p_knn[tok] += w

    return p_knn

# ── Perplexity evaluation ─────────────────────────────────────────────────────

def eval_perplexity(llm, mem, id_to_token, paragraphs, lambdas, k=8, T=10.0,
                    ctx_window=64, max_para=20):
    """Compute perplexity on test paragraphs for each λ in lambdas.

    Returns dict: lambda → perplexity.
    """
    _load_embedder()
    vocab_size = llm.n_vocab()

    log_probs = {lam: [] for lam in lambdas}
    token_times = []

    for para in tqdm(paragraphs[:max_para], desc="  Evaluating PPL"):
        toks = llm.tokenize(para.encode("utf-8"))
        if toks and toks[0] == llm.token_bos():
            toks = toks[1:]
        if len(toks) < ctx_window + 2:
            continue

        # One forward pass over the full paragraph
        llm.reset()
        llm.eval(toks)
        # llm.scores[i] predicts toks[i+1] (with logits_all=True)

        # For each token position (skip first ctx_window tokens — no context yet)
        for i in range(ctx_window, len(toks) - 1):
            true_next = toks[i + 1]
            raw_logits = llm.scores[i].copy()   # shape: (vocab_size,)

            # Baseline LM distribution
            max_l = raw_logits.max()
            p_lm = np.exp(raw_logits - max_l)
            p_lm /= p_lm.sum()

            # kNN distribution
            t0 = time.perf_counter()
            ctx_text = llm.detokenize(toks[i - ctx_window + 1: i + 1]).decode("utf-8", errors="ignore")
            q_emb = embed(ctx_text)
            p_knn = knn_dist(mem, id_to_token, q_emb, vocab_size, k=k, T=T)
            token_times.append(time.perf_counter() - t0)

            for lam in lambdas:
                p_final = (1.0 - lam) * p_lm + lam * p_knn
                p_true = float(p_final[true_next])
                if p_true > 0:
                    log_probs[lam].append(math.log(p_true))

    # Perplexity = exp(-mean(log P))
    ppls = {}
    for lam in lambdas:
        lps = log_probs[lam]
        if lps:
            ppls[lam] = math.exp(-sum(lps) / len(lps))
        else:
            ppls[lam] = float("inf")

    avg_retrieve_ms = np.mean(token_times) * 1000 if token_times else 0.0
    print(f"  Avg retrieval+embed per token: {avg_retrieve_ms:.2f}ms  "
          f"(tokens evaluated: {len(log_probs[lambdas[0]])})")

    return ppls

# ── Factual probability probes ────────────────────────────────────────────────

# Each probe: (context_text, target_token_string, description)
# We measure P(target_token | context) with and without kNN augmentation.
FACTUAL_PROBES = [
    ("Pion achieves a mean of 8,",            "134",    "macOS peak QPS 8,134"),
    ("The default HNSW M parameter is ",      "16",     "HNSW M=16"),
    ("WAL ring buffers are ",                 "256",    "WAL 256MB per worker"),
    ("Pion retrieval latency is 1.",          "28",     "latency 1.28ms"),
    ("ef_construction default is ",          "128",    "ef_construction=128"),
    ("The number of HNSW layers max_level=", "6",      "max_level=6"),
    ("HNSW batch size kernel uses ",         "8",      "batch-8 kernel"),
    ("Recall@100 score is 0.9",              "371",    "recall 0.9371"),
    ("Pion port default is ",                "1974",   "port 1974"),
    ("workers per machine desktop profile: ","8",      "8 workers"),
]

def eval_factual_probes(llm, mem, id_to_token, lambdas, k=8, T=10.0):
    """For each probe, measure P(target_token | context) at each λ."""
    _load_embedder()
    vocab_size = llm.n_vocab()

    print("\n  Factual probes:")
    results = []

    for ctx_text, target_str, desc in FACTUAL_PROBES:
        # Tokenize target — take first token only
        target_toks = llm.tokenize(target_str.encode("utf-8"))
        if target_toks and target_toks[0] == llm.token_bos():
            target_toks = target_toks[1:]
        if not target_toks:
            continue
        target_tok = target_toks[0]

        # Run model forward on context
        ctx_toks = llm.tokenize(ctx_text.encode("utf-8"))
        if ctx_toks and ctx_toks[0] == llm.token_bos():
            ctx_toks = ctx_toks[1:]

        llm.reset()
        llm.eval(ctx_toks)
        raw_logits = llm.scores[llm.n_tokens - 1].copy()

        max_l = raw_logits.max()
        p_lm = np.exp(raw_logits - max_l)
        p_lm /= p_lm.sum()

        # kNN distribution
        q_emb = embed(ctx_text)
        p_knn = knn_dist(mem, id_to_token, q_emb, vocab_size, k=k, T=T)

        probe_row = {"desc": desc, "target": target_str}
        for lam in lambdas:
            p_final = (1.0 - lam) * p_lm + lam * p_knn
            probe_row[f"lam_{lam}"] = float(p_final[target_tok])

        # Rank of target token in baseline distribution
        rank = int((p_lm > p_lm[target_tok]).sum()) + 1
        probe_row["baseline_rank"] = rank

        results.append(probe_row)
        p_base = probe_row["lam_0.0"]
        p_best = max(probe_row[f"lam_{lam}"] for lam in lambdas if lam > 0)
        boost = p_best / p_base if p_base > 0 else float("inf")
        print(f"  {desc:<35} P_base={p_base:.5f}  P_kNN_best={p_best:.5f}  "
              f"boost={boost:.2f}×  rank={rank}")

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _load_embedder()
    llm = load_llm()

    train_paras, test_paras = load_corpus(root=".", n_train=150, n_test=30)

    mem, id_to_token = build_datastore(llm, train_paras, stride=4, ctx_window=64)
    print(f"Datastore ready: {len(id_to_token)} entries\n")

    LAMBDAS = [0.0, 0.1, 0.25, 0.5]
    K = 8
    T = 10.0

    # ── A: Perplexity evaluation ──────────────────────────────────────────────
    print("=" * 60)
    print("A. PERPLEXITY on held-out test paragraphs")
    print("=" * 60)
    ppls = eval_perplexity(llm, mem, id_to_token, test_paras,
                           lambdas=LAMBDAS, k=K, T=T, max_para=20)

    print("\n  λ=0.00 (baseline):", f"{ppls[0.0]:.2f}")
    for lam in LAMBDAS[1:]:
        delta = (ppls[lam] - ppls[0.0]) / ppls[0.0] * 100
        marker = " ✓ IMPROVE" if delta < -2 else (" ✗ WORSE" if delta > 2 else " ~ neutral")
        print(f"  λ={lam:.2f}:           {ppls[lam]:.2f}  ({delta:+.1f}%){marker}")

    best_lam = min(LAMBDAS[1:], key=lambda l: ppls[l])
    best_improvement = (ppls[0.0] - ppls[best_lam]) / ppls[0.0] * 100

    # ── B: Factual probes ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("B. FACTUAL PROBABILITY PROBES (domain-specific tokens)")
    print("=" * 60)
    probe_results = eval_factual_probes(llm, mem, id_to_token, lambdas=LAMBDAS, k=K, T=T)

    # Count probes with ≥2× boost
    boosted = 0
    for row in probe_results:
        p_base = row["lam_0.0"]
        p_best = max(row[f"lam_{lam}"] for lam in LAMBDAS if lam > 0)
        if p_base > 0 and p_best / p_base >= 2.0:
            boosted += 1

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 14 — kNN-LM FINAL RESULTS")
    print("=" * 60)
    print(f"Datastore:        {len(id_to_token)} entries ({len(train_paras)} train paras, stride=4)")
    print(f"Best λ:           {best_lam}  (PPL: {ppls[best_lam]:.2f})")
    print(f"PPL improvement:  {best_improvement:.1f}%  (baseline: {ppls[0.0]:.2f})")
    print(f"Factual boost:    {boosted}/{len(probe_results)} probes ≥2× boost")
    print()

    ppl_pass  = best_improvement >= 2.0
    fact_pass = boosted >= 5

    if ppl_pass or fact_pass:
        print("VERDICT: PASS")
        if ppl_pass:
            print(f"  ✓ PPL improved {best_improvement:.1f}% at λ={best_lam}")
        if fact_pass:
            print(f"  ✓ {boosted}/10 domain factual probes boosted ≥2×")
        print("  Gate: kNN-LM claim is publishable")
        print("  Next: tune λ/T, scale datastore, write blog section")
    else:
        print("VERDICT: FAIL")
        print(f"  PPL change: {best_improvement:.1f}% (need ≥2%)")
        print(f"  Factual boost: {boosted}/10 probes (need ≥5)")
        print("  Root cause hypothesis: MiniLM embeddings don't align with")
        print("  LLM's next-token prediction space. Need model's own hidden")
        print("  states (requires HuggingFace or MAX inference API).")
        print("  Next: write Exp 14b with llama.cpp hidden-state extraction")

if __name__ == "__main__":
    main()
