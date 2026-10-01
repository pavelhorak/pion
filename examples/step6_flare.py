"""
Step 6: FLARE (Forward-Looking Active REtrieval) Prototype

This script implements a simplified version of FLARE.
Instead of relying on the model to use explicit tools like "Search: <query>",
the inference engine monitors the internal token probabilities (logprobs) of the LLM.

Algorithm:
1. Start generating the answer sentence by sentence (or chunk by chunk).
2. If the model generates a chunk where the confidence (probability) of any token 
   drops below a threshold (tau), the model is "uncertain" or "hallucinating".
3. We pause, discard the uncertain chunk, and use that draft chunk as a search query 
   against Pion.
4. We inject the retrieved context invisibly into the prompt and restart the generation 
   of that chunk.
5. If the model is confident, we simply accept the chunk and continue.

This achieves mid-generation retrieval (Active RAG) without requiring the model 
to have cognitive overhead for tool-use, making it ideal for small, stock models.
"""

import os
import sys, time, os, re, math
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

# ── HotpotQA Helpers ──────────────────────────────────────────────────────────

def build_index_for_question(example):
    mem = PionMemory(dim=384, max_elements=100, M=16, ef_construction=64)
    sentences_map = {}
    
    idx = 0
    for title, sents in zip(example['context']['title'], example['context']['sentences']):
        text = " ".join(sents)
        emb = embed(text)
        mem.remember(idx, emb)
        sentences_map[idx] = f"Title: {title}. Text: {text}"
        idx += 1
        
    mem.optimize()
    return mem, sentences_map

def normalize_answer(s):
    import string
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
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

# ── Generation Strategies ─────────────────────────────────────────────────────

def generate_baseline(llm, question):
    prompt = f"Answer the question with a short, exact phrase.\nQuestion: {question}\nAnswer:"
    out = llm(prompt, max_tokens=30, stop=["\n", "Question:"], temperature=0.0, echo=False)
    return out["choices"][0]["text"].strip()

def generate_flare(llm, question, mem, sentences_map, tau=0.2, max_tokens=40):
    # tau: probability threshold. If min_prob < tau, we are uncertain.
    
    retrieved_contexts = []
    generated_text = ""
    
    # We will generate in chunks of up to 10 tokens
    chunk_size = 10
    
    while len(llm.tokenize(generated_text.encode("utf-8"))) < max_tokens:
        # Build prompt
        context_str = "\n".join(retrieved_contexts)
        if context_str:
            prompt = f"Context facts:\n{context_str}\n\nAnswer the question with a short, exact phrase.\nQuestion: {question}\nAnswer: {generated_text}"
        else:
            prompt = f"Answer the question with a short, exact phrase.\nQuestion: {question}\nAnswer: {generated_text}"
            
        out = llm(prompt, max_tokens=chunk_size, stop=["\n", "Question:"], temperature=0.0, logprobs=1, echo=False)
        
        chunk = out["choices"][0]["text"]
        logprobs_data = out["choices"][0]["logprobs"]["token_logprobs"]
        tokens_data = out["choices"][0]["logprobs"]["tokens"]
        
        if not logprobs_data or not chunk:
            break
            
        # Ignore the first token's logprob if it's just a space or continuation artifact sometimes
        probs = [math.exp(lp) if lp is not None else 1.0 for lp in logprobs_data]
        
        # Are we uncertain?
        min_prob = min(probs)
        is_uncertain = min_prob < tau
        
        if is_uncertain:
            # We are uncertain! We discard this chunk (or use it as a query)
            draft_query = generated_text + " " + chunk
            q_emb = embed(draft_query)
            results = mem.recall(q_emb, k=1, ef=32)
            
            if results:
                new_fact = sentences_map[results[0][0]]
                if new_fact not in retrieved_contexts:
                    retrieved_contexts.append(new_fact)
                    # We continue the while loop WITHOUT appending the uncertain chunk,
                    # so it will regenerate with the new context.
                    continue
            
            # If no new facts found, we just accept the chunk and move on
            generated_text += chunk
        else:
            # We are confident! Accept the chunk
            generated_text += chunk
            
        finish_reason = out["choices"][0]["finish_reason"]
        if finish_reason in ["stop", "eos"]:
            break

    return generated_text.strip(), len(retrieved_contexts)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _load()
    llm = load_llm()
    
    print("Loading HotpotQA validation set...")
    ds = load_dataset('hotpot_qa', 'distractor', split='validation')
    subset = [ex for ex in ds if ex['level'] == 'hard'][:20]
    
    print(f"\nEvaluating on {len(subset)} questions...\n")
    
    metrics = {
        "baseline": {"em": 0, "f1": 0},
        "flare": {"em": 0, "f1": 0, "avg_retrievals": 0}
    }
    
    for i, ex in enumerate(subset):
        q = ex['question']
        gold = ex['answer']
        
        mem, sentences_map = build_index_for_question(ex)
        
        # 1. Baseline
        ans_base = generate_baseline(llm, q)
        metrics["baseline"]["em"] += exact_match_score(ans_base, gold)
        metrics["baseline"]["f1"] += f1_score(ans_base, gold)
        
        # 2. FLARE
        ans_flare, retrievals = generate_flare(llm, q, mem, sentences_map, tau=0.4)
        metrics["flare"]["em"] += exact_match_score(ans_flare, gold)
        metrics["flare"]["f1"] += f1_score(ans_flare, gold)
        metrics["flare"]["avg_retrievals"] += retrievals
        
        print(f"Q{i+1}: {q}")
        print(f"  Gold: {gold}")
        print(f"  Base: {ans_base}")
        print(f"  FLARE: {ans_flare} (Retrievals triggered: {retrievals})\n")

    N = len(subset)
    print("\n" + "="*50)
    print("RESULTS (N=20)")
    print("="*50)
    print(f"Baseline - EM: {metrics['baseline']['em']/N:.2f}, F1: {metrics['baseline']['f1']/N:.2f}")
    print(f"FLARE    - EM: {metrics['flare']['em']/N:.2f}, F1: {metrics['flare']['f1']/N:.2f} (Avg ret: {metrics['flare']['avg_retrievals']/N:.1f})")

if __name__ == "__main__":
    main()
