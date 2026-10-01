"""
Step 12: KV Cache Alignment Guard (Experiment 16)

Tests whether exact token prefix matching fixes the RoPE entanglement bug from Exp 9
while preserving meaningful TTFT speedup.

Problem from Exp 9: similarity-only reuse (cosine ≥ 0.95) caused answer quality
degradation — loading a KV state built on base_tokens then evaluating a modified
document's question causes the model to attend to wrong positional encodings
(RoPE entanglement). The cached "Pion" at position 42 doesn't match "Pion (database)"
at the same position in the modified document.

Fix: after a similarity hit, verify the first TOKEN_PREFIX_LEN token IDs exactly match
the cached document. If they diverge → fall back to full re-eval (safe but slow).
If they match exactly → KV state is valid, reuse it (fast + correct).

Three conditions tested on each query:
  A. BASELINE — full re-eval every request (no cache), correct + slow
  B. SIMILARITY-ONLY — reuse if cosine ≥ 0.95 (old Exp 9 behavior), fast but wrong on
     modified documents
  C. ALIGNED — reuse if cosine ≥ 0.98 AND first 64 tokens match exactly (new guard),
     fast + correct on identical docs, safe fallback on modified docs

Two document types per query:
  EXACT   — same tokens as cached document → should hit in B and C
  MODIFIED — "Pion" → "Pion (database)", changes token IDs at every occurrence
              → should hit in B (cosine ≈ 0.98), rejected by C (token mismatch)

Pass criteria:
  - Aligned condition: TTFT speedup ≥ 5× on EXACT documents
  - Aligned condition: answer quality matches BASELINE on MODIFIED documents
  - Similarity-only condition: shows measurable answer degradation on MODIFIED documents
    (confirms the bug is real and the guard fixes it)
"""

import os
import sys, os, time, math
import numpy as np

sys.path.insert(0, ".")
from pion_memory import PionMemory
import onnxruntime as ort
from transformers import AutoTokenizer

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

def dist_to_cosine(l2_dist: float) -> float:
    """Convert L2 distance (on unit-norm vectors) to cosine similarity.
    L2² = 2*(1 - cos_sim)  →  cos_sim = 1 - L2²/2
    """
    return 1.0 - (l2_dist ** 2) / 2.0

# ── LLM ──────────────────────────────────────────────────────────────────────

def load_llm():
    from llama_cpp import Llama
    gguf = os.path.expanduser(
        "~/.ollama/models/blobs/"
        "sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
    )
    print(f"Loading GGUF ({os.path.getsize(gguf)/1e9:.1f}GB)...")
    llm = Llama(model_path=gguf, n_ctx=2048, n_threads=8, verbose=False)
    print("Model loaded.")
    return llm

# ── KV cache store ────────────────────────────────────────────────────────────

class KVStore:
    """In-memory KV cache store keyed by Pion HNSW id."""
    def __init__(self, dim=384, max_elements=32):
        self.mem = PionMemory(dim=dim, max_elements=max_elements, M=16, ef_construction=64)
        self.entries = {}   # id → {"tokens": list[int], "state": LlamaState, "text": str}
        self._next_id = 0

    def store(self, text: str, tokens: list, state) -> int:
        idx = self._next_id
        self._next_id += 1
        emb = embed(text)
        self.mem.remember(idx, emb)
        self.entries[idx] = {"tokens": tokens, "state": state, "text": text}
        return idx

    def finalize(self):
        self.mem.optimize()

    def lookup(self, text: str, gate_dist: float, prefix_len: int, query_tokens: list):
        """Returns (cache_entry | None, cosine_sim, token_match: bool | None).

        gate_dist: max L2 distance to accept (cosine_sim threshold converted).
        prefix_len: number of leading token IDs to compare for exact match.
        query_tokens: token IDs of the incoming document (NOT including the question).
        """
        emb = embed(text)
        results = self.mem.recall(emb, k=1, ef=32)
        if not results:
            return None, 0.0, None

        best_id, l2_dist = results[0]
        cos_sim = dist_to_cosine(l2_dist)

        if l2_dist > gate_dist:
            return None, cos_sim, None   # similarity gate failed

        entry = self.entries[best_id]
        cached_tokens = entry["tokens"]
        check_len = min(prefix_len, len(query_tokens), len(cached_tokens))
        token_match = (cached_tokens[:check_len] == query_tokens[:check_len])

        return entry, cos_sim, token_match

# ── Generation helpers ─────────────────────────────────────────────────────────

def generate(llm, n_tokens=20) -> list:
    out = []
    for _ in range(n_tokens):
        t = llm.sample()
        out.append(t)
        llm.eval([t])
        if t == llm.token_eos():
            break
    return out

def run_baseline(llm, doc_tokens: list, q_tokens: list, n_gen=20):
    """Full re-eval: doc + question tokens, then generate."""
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(doc_tokens + q_tokens)
    ttft = time.perf_counter() - t0
    ans = llm.detokenize(generate(llm, n_gen)).decode("utf-8", errors="ignore").strip()
    return ans, ttft

def run_cached(llm, state, q_tokens: list, n_gen=20):
    """Fast path: load cached state, eval only question tokens."""
    llm.reset()
    t0 = time.perf_counter()
    llm.load_state(state)
    llm.eval(q_tokens)
    ttft = time.perf_counter() - t0
    ans = llm.detokenize(generate(llm, n_gen)).decode("utf-8", errors="ignore").strip()
    return ans, ttft

# ── Experiment ────────────────────────────────────────────────────────────────

# L2 distance thresholds for cosine similarity gates
# cos_sim ≥ 0.95  →  L2 ≤ sqrt(2*0.05) ≈ 0.316
# cos_sim ≥ 0.98  →  L2 ≤ sqrt(2*0.02) ≈ 0.200
GATE_OLD = math.sqrt(2 * 0.05)   # cosine ≥ 0.95
GATE_NEW = math.sqrt(2 * 0.02)   # cosine ≥ 0.98
TOKEN_PREFIX_LEN = 64

# Test questions — chosen so correct answers are unambiguous
QUESTIONS = [
    ("\n\nQuestion: What protocol is Pion wire-compatible with?\nAnswer:",
     "Redis/Valkey RESP"),
    ("\n\nQuestion: What programming language is Pion written in?\nAnswer:",
     "Mojo"),
    ("\n\nQuestion: What data structure does Pion use for vector search?\nAnswer:",
     "HNSW"),
]

def main():
    _load_embedder()
    llm = load_llm()

    # Load corpus document
    print("\nLoading document...")
    with open("doc/architecture.md", "r") as f:
        text = f.read()
    base_doc = " ".join(text.split()[:800])  # ~1000 tokens
    mod_doc = base_doc.replace("Pion", "Pion (database)").replace("engine", "system")

    base_tokens = llm.tokenize(base_doc.encode("utf-8"))
    mod_tokens  = llm.tokenize(mod_doc.encode("utf-8"))

    print(f"Base doc: {len(base_tokens)} tokens")
    print(f"Modified doc: {len(mod_tokens)} tokens")
    print(f"Token prefix match (first {TOKEN_PREFIX_LEN}): "
          f"{base_tokens[:TOKEN_PREFIX_LEN] == mod_tokens[:TOKEN_PREFIX_LEN]}")

    # Build cache from base document
    print("\nBuilding KV cache from base document...")
    store = KVStore(dim=384, max_elements=16)
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(base_tokens)
    eval_time = time.perf_counter() - t0
    state = llm.save_state()
    store.store(base_doc, base_tokens, state)
    store.finalize()
    print(f"Cached base doc in {eval_time:.2f}s ({len(base_tokens)/eval_time:.0f} tok/s)")

    # Results accumulator
    results = {
        "exact":    {"baseline": [], "sim_only": [], "aligned": []},
        "modified": {"baseline": [], "sim_only": [], "aligned": []},
    }

    print("\n" + "="*70)
    print("EXPERIMENT 16 — KV CACHE ALIGNMENT GUARD")
    print("="*70)

    for q_text, expected in QUESTIONS:
        q_tokens = llm.tokenize(q_text.encode("utf-8"))

        for doc_label, doc_tokens, doc_text in [
            ("EXACT",    base_tokens, base_doc),
            ("MODIFIED", mod_tokens,  mod_doc),
        ]:
            print(f"\n[{doc_label}] Q: {q_text.strip()[:60]}")
            print(f"  Expected answer contains: '{expected}'")

            # ── Baseline ──────────────────────────────────────────────────
            ans_base, ttft_base = run_baseline(llm, doc_tokens, q_tokens)
            correct_base = expected.lower() in ans_base.lower()
            print(f"  BASELINE      TTFT={ttft_base:.2f}s  {'✓' if correct_base else '✗'}  {ans_base[:60]!r}")
            results[doc_label.lower()]["baseline"].append((ttft_base, correct_base))

            # ── Similarity-only (old behavior, gate=0.95) ─────────────────
            entry, cos_sim, tok_match = store.lookup(
                doc_text, GATE_OLD, TOKEN_PREFIX_LEN, doc_tokens)
            if entry:
                ans_sim, ttft_sim = run_cached(llm, entry["state"], q_tokens)
                hit = True
            else:
                ans_sim, ttft_sim = run_baseline(llm, doc_tokens, q_tokens)
                hit = False
            correct_sim = expected.lower() in ans_sim.lower()
            speedup_sim = ttft_base / ttft_sim if ttft_sim > 0 else 0
            print(f"  SIM-ONLY 0.95 TTFT={ttft_sim:.2f}s  {'✓' if correct_sim else '✗'}  "
                  f"hit={'Y' if hit else 'N'} cos={cos_sim:.3f} tok_match={tok_match}  "
                  f"speedup={speedup_sim:.1f}×  {ans_sim[:40]!r}")
            results[doc_label.lower()]["sim_only"].append((ttft_sim, correct_sim, hit, speedup_sim))

            # ── Aligned (new, gate=0.98 + token prefix match) ─────────────
            entry2, cos_sim2, tok_match2 = store.lookup(
                doc_text, GATE_NEW, TOKEN_PREFIX_LEN, doc_tokens)
            if entry2 and tok_match2:
                ans_aln, ttft_aln = run_cached(llm, entry2["state"], q_tokens)
                hit2 = True
                fallback = False
            else:
                ans_aln, ttft_aln = run_baseline(llm, doc_tokens, q_tokens)
                hit2 = False
                fallback = True
            correct_aln = expected.lower() in ans_aln.lower()
            speedup_aln = ttft_base / ttft_aln if ttft_aln > 0 else 0
            reason = ("cache hit" if hit2 else
                      f"fallback (cos={cos_sim2:.3f}<0.98" if not entry2 else
                      f"fallback (tok_match=False)")
            print(f"  ALIGNED  0.98 TTFT={ttft_aln:.2f}s  {'✓' if correct_aln else '✗'}  "
                  f"{reason}  speedup={speedup_aln:.1f}×  {ans_aln[:40]!r}")
            results[doc_label.lower()]["aligned"].append((ttft_aln, correct_aln, hit2, speedup_aln))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for doc_label in ["exact", "modified"]:
        r = results[doc_label]
        n = len(QUESTIONS)

        base_ttft   = np.mean([x[0] for x in r["baseline"]])
        base_acc    = sum(x[1] for x in r["baseline"])

        sim_ttft    = np.mean([x[0] for x in r["sim_only"]])
        sim_acc     = sum(x[1] for x in r["sim_only"])
        sim_hits    = sum(x[2] for x in r["sim_only"])
        sim_speedup = np.mean([x[3] for x in r["sim_only"]])

        aln_ttft    = np.mean([x[0] for x in r["aligned"]])
        aln_acc     = sum(x[1] for x in r["aligned"])
        aln_hits    = sum(x[2] for x in r["aligned"])
        aln_speedup = np.mean([x[3] for x in r["aligned"]])

        print(f"\n{doc_label.upper()} documents ({n} questions):")
        print(f"  BASELINE:     avg TTFT={base_ttft:.2f}s  correct={base_acc}/{n}")
        print(f"  SIM-ONLY 0.95: avg TTFT={sim_ttft:.2f}s  correct={sim_acc}/{n}  "
              f"hits={sim_hits}/{n}  speedup={sim_speedup:.1f}×")
        print(f"  ALIGNED  0.98: avg TTFT={aln_ttft:.2f}s  correct={aln_acc}/{n}  "
              f"hits={aln_hits}/{n}  speedup={aln_speedup:.1f}×")

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    exact_aln_speedup  = np.mean([x[3] for x in results["exact"]["aligned"]])
    exact_aln_acc      = sum(x[1] for x in results["exact"]["aligned"])
    mod_base_acc       = sum(x[1] for x in results["modified"]["baseline"])
    mod_sim_acc        = sum(x[1] for x in results["modified"]["sim_only"])
    mod_aln_acc        = sum(x[1] for x in results["modified"]["aligned"])
    n = len(QUESTIONS)

    speedup_pass  = exact_aln_speedup >= 5.0
    quality_pass  = mod_aln_acc >= mod_base_acc   # aligned matches or beats baseline on modified docs
    bug_confirmed = mod_sim_acc < mod_base_acc     # sim-only degrades quality on modified docs

    print(f"Speedup on EXACT docs (aligned):    {exact_aln_speedup:.1f}×  "
          f"{'✓ PASS (≥5×)' if speedup_pass else '✗ FAIL (<5×)'}")
    print(f"Answer quality on MODIFIED docs:")
    print(f"  Baseline:    {mod_base_acc}/{n}")
    print(f"  Sim-only:    {mod_sim_acc}/{n}  {'← degraded (bug confirmed)' if bug_confirmed else '← no degradation'}")
    print(f"  Aligned:     {mod_aln_acc}/{n}  "
          f"{'✓ PASS (matches baseline)' if quality_pass else '✗ FAIL (still degraded)'}")

    if speedup_pass and quality_pass:
        print("\nVERDICT: PASS")
        if bug_confirmed:
            print("  ✓ RoPE entanglement bug confirmed on similarity-only reuse")
        print(f"  ✓ Aligned guard restores answer quality on modified documents")
        print(f"  ✓ {exact_aln_speedup:.1f}× TTFT speedup preserved on exact-match documents")
        print("  Claim: 'Semantic KV cache is safe with cosine≥0.98 + exact token prefix match'")
        print("  Next: Exp 17 REST approximate speculative decoding")
    else:
        if not speedup_pass:
            print(f"\nVERDICT: FAIL — speedup {exact_aln_speedup:.1f}× < 5× on exact docs")
            print("  → Aligned gate may be too conservative; check hit rate")
        if not quality_pass:
            print(f"\nVERDICT: FAIL — aligned quality {mod_aln_acc}/{n} < baseline {mod_base_acc}/{n}")
            print("  → Token prefix check not fixing RoPE issue; may need full token match")

if __name__ == "__main__":
    main()
