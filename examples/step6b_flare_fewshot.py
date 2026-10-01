"""
Step 6b: FLARE with Few-Shot Prompting (Experiment 12b)

Same FLARE mechanism as step6_flare.py but with:
1. Few-shot examples in the prompt to force short factoid answers
2. First-sentence extraction post-processing as fallback
3. τ sweep: test 0.2, 0.3, 0.4 to find optimal threshold

Gemini expert recommendation: few-shot is the most reliable fix for verbose CoT
import os
from instruction-tuned 8B models on short-answer QA.
"""

import sys, time, os, re, math, string
import numpy as np
from datasets import load_dataset

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
    print(f"Loading GGUF: {os.path.basename(gguf)}...")
    llm = Llama(model_path=gguf, n_ctx=2048, n_threads=8, verbose=False, logits_all=True)
    print("Model loaded.")
    return llm

# ── Few-shot prompt ────────────────────────────────────────────────────────────

FEW_SHOT = """Q: Who wrote the play Hamlet?
A: William Shakespeare

Q: What is the capital of Japan?
A: Tokyo

Q: In what year did World War II end?
A: 1945

Q: What company makes the iPhone?
A: Apple

Q: Who painted the Mona Lisa?
A: Leonardo da Vinci

"""

def make_prompt(question, context_facts=None, generated_so_far=""):
    ctx = ""
    if context_facts:
        ctx = "Context:\n" + "\n".join(f"- {f}" for f in context_facts) + "\n\n"
    return f"{FEW_SHOT}{ctx}Q: {question}\nA: {generated_so_far}"

# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Extract the first short answer phrase, stripping reasoning parentheticals."""
    # Take first sentence/phrase before any parenthetical, note, or newline
    text = text.strip()
    # Cut at first parenthetical
    text = re.split(r'\s*[\(\[]', text)[0].strip()
    # Cut at first newline
    text = text.split('\n')[0].strip()
    # Cut at period only if followed by space + capital (new sentence)
    text = re.split(r'\.\s+[A-Z]', text)[0].strip()
    # Remove trailing period
    text = text.rstrip('.')
    return text.strip()

# ── F1 / EM scoring ───────────────────────────────────────────────────────────

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens).intersection(set(ground_truth_tokens))
    if len(common) == 0:
        return 0
    prec = len(common) / len(prediction_tokens)
    rec = len(common) / len(ground_truth_tokens)
    return 2 * (prec * rec) / (prec + rec)

# ── HotpotQA helpers ──────────────────────────────────────────────────────────

def build_index_for_question(example):
    mem = PionMemory(dim=384, max_elements=100, M=16, ef_construction=64)
    sentences_map = {}
    idx = 0
    for title, sentences in zip(example['context']['title'], example['context']['sentences']):
        for sent in sentences:
            sent = sent.strip()
            if sent:
                emb = embed(sent)
                mem.remember(idx, emb)
                sentences_map[idx] = sent
                idx += 1
    mem.optimize()
    return mem, sentences_map

# ── Generation ────────────────────────────────────────────────────────────────

def generate_baseline(llm, question):
    prompt = make_prompt(question)
    out = llm(prompt, max_tokens=20, stop=["\n", "Q:"], temperature=0.0, echo=False)
    raw = out["choices"][0]["text"].strip()
    return extract_answer(raw)

def generate_flare(llm, question, mem, sentences_map, tau=0.3, max_tokens=40):
    retrieved_contexts = []
    generated_text = ""
    chunk_size = 10
    total_tokens = 0

    while total_tokens < max_tokens:
        prompt = make_prompt(question, retrieved_contexts, generated_text)
        out = llm(prompt, max_tokens=chunk_size, stop=["\n", "Q:"],
                  temperature=0.0, logprobs=1, echo=False)

        choice = out["choices"][0]
        chunk = choice["text"]
        logprobs_data = choice["logprobs"]["token_logprobs"] if choice.get("logprobs") else []

        if not chunk:
            break

        # Compute min probability for the chunk
        probs = [math.exp(lp) if lp is not None else 1.0 for lp in logprobs_data]
        min_prob = min(probs) if probs else 1.0
        is_uncertain = min_prob < tau

        if is_uncertain and chunk.strip():
            draft_query = (generated_text + " " + chunk).strip()
            q_emb = embed(draft_query)
            results = mem.recall(q_emb, k=1, ef=32)
            if results:
                new_fact = sentences_map[results[0][0]]
                if new_fact not in retrieved_contexts:
                    retrieved_contexts.append(new_fact)
                    continue  # regenerate with new context, discard uncertain chunk

        generated_text += chunk
        total_tokens += len(logprobs_data) if logprobs_data else len(chunk.split())

        if choice["finish_reason"] in ["stop", "eos"]:
            break

    return extract_answer(generated_text), len(retrieved_contexts)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _load()
    llm = load_llm()

    print("Loading HotpotQA validation set...")
    ds = load_dataset('hotpot_qa', 'distractor', split='validation')
    subset = [ex for ex in ds if ex['level'] == 'hard'][:20]
    print(f"\nEvaluating on {len(subset)} questions...\n")

    # Test τ=0.3 (Gemini recommended starting point; 0.4 was too aggressive for 8B)
    tau = 0.3

    metrics = {
        "baseline": {"em": 0, "f1": 0},
        "flare": {"em": 0, "f1": 0, "retrievals": 0}
    }

    for i, ex in enumerate(subset):
        q = ex['question']
        gold = ex['answer']

        mem, sentences_map = build_index_for_question(ex)

        ans_base = generate_baseline(llm, q)
        metrics["baseline"]["em"] += exact_match_score(ans_base, gold)
        metrics["baseline"]["f1"] += f1_score(ans_base, gold)

        ans_flare, retrievals = generate_flare(llm, q, mem, sentences_map, tau=tau)
        metrics["flare"]["em"] += exact_match_score(ans_flare, gold)
        metrics["flare"]["f1"] += f1_score(ans_flare, gold)
        metrics["flare"]["retrievals"] += retrievals

        base_f1 = f1_score(ans_base, gold)
        flare_f1 = f1_score(ans_flare, gold)
        delta = "✓ +" if flare_f1 > base_f1 else ("= " if flare_f1 == base_f1 else "✗ -")
        print(f"Q{i+1:02d}: {q[:70]}")
        print(f"  Gold:  {gold}")
        print(f"  Base:  {ans_base}  [F1={base_f1:.2f}]")
        print(f"  FLARE: {ans_flare}  [F1={flare_f1:.2f}] {delta} ret={retrievals}\n")

    n = len(subset)
    print("=" * 60)
    print(f"RESULTS (N={n}, τ={tau})")
    print("=" * 60)
    print(f"Baseline  EM={metrics['baseline']['em']/n:.2f}  F1={metrics['baseline']['f1']/n:.2f}")
    print(f"FLARE     EM={metrics['flare']['em']/n:.2f}  F1={metrics['flare']['f1']/n:.2f}  "
          f"Avg ret={metrics['flare']['retrievals']/n:.1f}")
    print()
    delta_f1 = (metrics['flare']['f1'] - metrics['baseline']['f1']) / n
    print(f"FLARE delta F1: {delta_f1:+.2f}")
    if delta_f1 > 0.05:
        print("VERDICT: PASS — FLARE improves over baseline")
    elif delta_f1 > 0:
        print("VERDICT: MARGINAL — small improvement, tune τ")
    else:
        print("VERDICT: FAIL — FLARE does not help with this τ")

if __name__ == "__main__":
    main()
