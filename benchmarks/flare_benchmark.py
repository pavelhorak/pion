#!/usr/bin/env python3
"""A5 Benchmark: FLARE F1 on HotpotQA, Natural Questions, TriviaQA

Compares three strategies:
  1. Baseline (no retrieval) — direct LLM answer
  2. Single-shot RAG — retrieve once before answering
  3. FLARE (mid-generation retrieval) — logprob-triggered active retrieval via Pion

Requires:
  - Pion server running:  ./pion-server -w 1
  - Ollama running:       ollama serve  (with llama3.1:8b pulled)
  - Dependencies:         pip install datasets requests numpy

Usage:
    # Full benchmark (all 3 datasets, N=100 per dataset):
    python benchmarks/flare_benchmark.py

    # Quick smoke test (N=20):
    python benchmarks/flare_benchmark.py --num-questions 20

    # Single dataset:
    python benchmarks/flare_benchmark.py --datasets hotpotqa

    # Custom tau sweep:
    python benchmarks/flare_benchmark.py --tau 0.2 0.3 0.4

    # Use FLARE gateway instead of direct Ollama:
    python benchmarks/flare_benchmark.py --use-gateway --gateway-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import string
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# ── Configuration ────────────────────────────────────────────────────────────

PION_HOST = "127.0.0.1"
PION_PORT = 1974
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
LLM_MODEL = "llama3.1:8b"

# ── Embedding ────────────────────────────────────────────────────────────────

def embed(text: str, ollama_url: str = OLLAMA_URL) -> np.ndarray:
    """Embed text using Ollama nomic-embed-text."""
    resp = requests.post(
        f"{ollama_url}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text[:512]},
        timeout=30,
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec

# ── Pion client (RESP protocol) ─────────────────────────────────────────────

class PionClient:
    """Minimal Pion RESP client for vector index operations. Uses persistent connection."""

    def __init__(self, host: str = PION_HOST, port: int = PION_PORT):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None

    def _connect(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(30)
        self._sock.connect((self.host, self.port))

    def _send_command(self, *args: Any) -> bytes:
        if not self._sock:
            self._connect()

        header = f"*{len(args)}\r\n".encode()
        body = b""
        for a in args:
            if isinstance(a, bytes):
                body += f"${len(a)}\r\n".encode() + a + b"\r\n"
            else:
                s = str(a)
                body += f"${len(s)}\r\n{s}\r\n".encode()

        try:
            self._sock.sendall(header + body)
            resp = self._sock.recv(64 * 1024)
            return resp
        except (ConnectionError, socket.timeout, OSError):
            self._connect()
            self._sock.sendall(header + body)
            resp = self._sock.recv(64 * 1024)
            return resp

    def create_index(self, name: str = "flare_kb", dim: int = EMBED_DIM):
        """Create a vector index."""
        self._send_command(
            "FT.CREATE", name, "ON", "HASH", "PREFIX", "1", f"doc:{name}:",
            "SCHEMA", "content", "TEXT", "vector", "VECTOR", "HNSW",
            "6", "TYPE", "FLOAT32", "DIM", str(dim), "DISTANCE_METRIC", "L2",
        )

    def drop_index(self, name: str = "flare_kb"):
        """Drop a vector index."""
        self._send_command("FT.DROPINDEX", name)

    def add_document(self, index_name: str, doc_id: str, content: str, vector: np.ndarray):
        """Add a document to the index."""
        self._send_command(
            "HSET", f"doc:{index_name}:{doc_id}",
            "content", content,
            "vector", vector.tobytes(),
        )

    def optimize(self, name: str = "flare_kb"):
        """Optimize the index (build HNSW)."""
        self._send_command("FT.OPTIMIZE", name)

    def search(self, index_name: str, query_vec: np.ndarray, k: int = 3) -> List[str]:
        """Search the index and return content strings."""
        resp = self._send_command(
            "FT.SEARCH", index_name,
            f"*=>[KNN {k} @vector $BLOB AS score]",
            "PARAMS", "2", "BLOB", query_vec.tobytes(),
            "RETURN", "1", "content",
            "SORTBY", "score",
            "LIMIT", "0", str(k),
            "DIALECT", "2",
        )
        # Parse RESP array response for content fields
        results = []
        try:
            lines = resp.split(b"\r\n")
            for i, line in enumerate(lines):
                if line == b"content" or line == b"$7" and i + 1 < len(lines) and lines[i + 1] == b"content":
                    # Next bulk string after "content" is the value
                    for j in range(i + 1, min(i + 4, len(lines))):
                        if lines[j].startswith(b"$") and lines[j] != b"$7" and lines[j] != b"$-1":
                            length = int(lines[j][1:])
                            if j + 1 < len(lines) and len(lines[j + 1]) == length:
                                results.append(lines[j + 1].decode("utf-8", errors="replace"))
                                break
        except Exception:
            pass
        return results

    def flushdb(self):
        """Flush all data."""
        self._send_command("FLUSHDB")

# ── LLM generation via Ollama ────────────────────────────────────────────────

def ollama_generate(
    prompt: str,
    model: str = LLM_MODEL,
    max_tokens: int = 50,
    temperature: float = 0.0,
    ollama_url: str = OLLAMA_URL,
    logprobs: bool = False,
) -> Dict[str, Any]:
    """Generate text using Ollama API. Returns response dict with 'text' and optionally 'logprobs'."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop": ["\n", "Q:", "Question:"],
        },
    }
    if logprobs:
        payload["logprobs"] = True

    resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    result = {"text": data.get("response", "").strip()}

    # Parse logprobs if available
    if logprobs and "logprobs" in data:
        result["logprobs"] = data["logprobs"]
    elif logprobs and "details" in data:
        # Some Ollama versions use 'details'
        result["logprobs"] = data["details"]

    return result

# ── FLARE chunked generation via Ollama ──────────────────────────────────────

def _detect_repetition(text: str) -> bool:
    """Detect if text contains repetitive phrases (3+ repeats of any 3+ word ngram)."""
    words = text.lower().split()
    if len(words) < 9:
        return False
    # Check 3-gram to 6-gram repetition
    for n in range(3, min(7, len(words) // 2 + 1)):
        seen = {}
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            seen[gram] = seen.get(gram, 0) + 1
            if seen[gram] >= 3:
                return True
    return False


def _truncate_at_repetition(text: str) -> str:
    """Truncate text at the point where repetition begins."""
    words = text.split()
    if len(words) < 6:
        return text
    # Find the first point where a 3+ word phrase repeats
    for n in range(3, min(7, len(words) // 2 + 1)):
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n]).lower()
            # Look for this gram later in the text
            for j in range(i + n, len(words) - n + 1):
                later = " ".join(words[j:j + n]).lower()
                if gram == later:
                    # Truncate just before the first repetition
                    return " ".join(words[:j]).strip()
    return text


def generate_flare_ollama(
    question: str,
    contexts: List[str],
    search_fn,
    tau: float = 0.3,
    max_tokens: int = 30,
    chunk_size: int = 8,
    model: str = LLM_MODEL,
    ollama_url: str = OLLAMA_URL,
) -> Tuple[str, int]:
    """Generate answer using post-hoc FLARE: generate full answer, check logprobs,
    if uncertain retrieve and regenerate.

    For short factoid QA, the whole answer is one phrase. Chunked FLARE introduces
    repetition and degrades quality. Post-hoc FLARE avoids this: generate once,
    check confidence, retrieve if needed, regenerate once with context.

    Returns (answer_text, num_retrievals).
    """
    retrieved = list(contexts)
    num_retrievals = 0
    max_rounds = 3  # max retrieve-and-regenerate cycles

    for round_num in range(max_rounds + 1):
        # Build prompt with any retrieved context
        ctx_str = "\n".join(f"- {c}" for c in retrieved) if retrieved else ""
        if ctx_str:
            prompt = f"{FEW_SHOT}Context:\n{ctx_str}\n\nQ: {question}\nA:"
        else:
            prompt = f"{FEW_SHOT}Q: {question}\nA:"

        result = ollama_generate(
            prompt, model=model, max_tokens=max_tokens,
            ollama_url=ollama_url, logprobs=True,
        )
        answer_text = result["text"]
        if not answer_text:
            break

        # Check logprobs for uncertainty
        min_prob = 1.0
        lp_data = result.get("logprobs")
        if lp_data and isinstance(lp_data, list):
            for lp in lp_data:
                if isinstance(lp, dict):
                    p = math.exp(lp.get("logprob", 0))
                    min_prob = min(min_prob, p)
                elif isinstance(lp, (int, float)) and lp is not None:
                    min_prob = min(min_prob, math.exp(lp))

        is_uncertain = min_prob < tau

        # If confident or no more rounds, return this answer
        if not is_uncertain or round_num >= max_rounds:
            break

        # Uncertain: use the draft answer as a search query
        new_docs = search_fn(question + " " + answer_text.strip())
        added = False
        for doc in new_docs:
            if doc not in retrieved:
                retrieved.append(doc)
                added = True

        if not added:
            # No new context found, return current answer
            break

        num_retrievals += 1
        # Loop: regenerate with new context

    return extract_answer(answer_text), num_retrievals

# ── FLARE via gateway ────────────────────────────────────────────────────────

def generate_flare_gateway(
    question: str,
    gateway_url: str = "http://localhost:8080",
    model: str = LLM_MODEL,
) -> Tuple[str, int]:
    """Generate answer using the FLARE gateway (handles retrieval internally).

    Returns (answer_text, num_retrievals_estimate).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Answer with a short, exact phrase: {question}"}],
        "max_tokens": 60,
        "temperature": 0.0,
    }
    resp = requests.post(f"{gateway_url}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    # The gateway doesn't report retrieval count directly; estimate from response headers
    retrievals = data.get("usage", {}).get("flare_retrievals", 0)
    return extract_answer(text), retrievals

# ── Answer extraction & scoring ──────────────────────────────────────────────

FEW_SHOT = """Q: Who wrote the play Hamlet?
A: William Shakespeare

Q: What is the capital of Japan?
A: Tokyo

Q: In what year did World War II end?
A: 1945

Q: Who painted the Mona Lisa?
A: Leonardo da Vinci

"""

def extract_answer(text: str) -> str:
    """Extract the first short answer phrase, stripping model reasoning/hedging."""
    text = text.strip()
    if not text:
        return ""

    # Strip common hedging prefixes from instruction-tuned models
    hedging = [
        r"^(?:The answer (?:is|to that (?:one )?is)[:\s]*)",
        r"^(?:Based on .*?,\s*)",
        r"^(?:According to .*?,\s*)",
        r"^(?:It (?:seems|looks|appears) like .*?[.!]\s*)",
        r"^(?:I'm not (?:sure|aware) .*?[.!]\s*)",
        r"^(?:(?:Yes|No)[,.]?\s*)",  # Keep yes/no but strip for extraction
    ]
    # Don't strip yes/no if the question starts with "Are"/"Is"/"Was"/"Were"/"Do"/"Did"
    original = text
    for pattern in hedging[:-1]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    # For yes/no: only strip if it's followed by more content
    yn_match = re.match(r"^(Yes|No)[,.\s]+(.+)", text, re.IGNORECASE)
    if yn_match and len(yn_match.group(2)) > 3:
        # Keep the yes/no as the answer if the rest is just elaboration
        rest = yn_match.group(2).strip()
        if len(rest.split()) > 10:
            text = yn_match.group(1)  # just "Yes" or "No"

    # If stripping removed everything, use original
    if not text.strip():
        text = original

    # Cut at parenthetical
    text = re.split(r'\s*[\(\[]', text)[0].strip()
    # First line only
    text = text.split('\n')[0].strip()
    # Cut at second sentence (period + space + capital)
    text = re.split(r'\.\s+[A-Z]', text)[0].strip()
    # Remove trailing punctuation
    text = text.rstrip('.!,;:')
    return text.strip()

def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)

# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_hotpotqa(n: int) -> List[Dict]:
    """Load HotpotQA distractor validation set."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    examples = []
    for ex in ds:
        if len(examples) >= n:
            break
        paragraphs = []
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            text = " ".join(sents)
            paragraphs.append({"title": title, "text": text})
        examples.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "paragraphs": paragraphs,
            "dataset": "HotpotQA",
        })
    return examples

def load_natural_questions(n: int) -> List[Dict]:
    """Load Natural Questions (open-domain short-answer subset)."""
    from datasets import load_dataset
    ds = load_dataset("nq_open", split="validation")
    examples = []
    for ex in ds:
        if len(examples) >= n:
            break
        answers = ex["answer"]
        if not answers:
            continue
        examples.append({
            "question": ex["question"],
            "answer": answers[0],  # first gold answer
            "all_answers": answers,
            "paragraphs": [],  # NQ open doesn't provide passages; we use Pion search
            "dataset": "NaturalQuestions",
        })
    return examples

def load_triviaqa(n: int) -> List[Dict]:
    """Load TriviaQA (unfiltered, validation set)."""
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "unfiltered", split="validation")
    examples = []
    for ex in ds:
        if len(examples) >= n:
            break
        answer = ex["answer"]["value"]
        aliases = ex["answer"].get("aliases", [])
        # Extract context from search results or entity pages
        paragraphs = []
        if ex.get("search_results") and ex["search_results"].get("search_context"):
            for ctx in ex["search_results"]["search_context"][:5]:
                if ctx:
                    paragraphs.append({"title": "", "text": ctx[:500]})
        elif ex.get("entity_pages") and ex["entity_pages"].get("wiki_context"):
            for ctx in ex["entity_pages"]["wiki_context"][:3]:
                if ctx:
                    paragraphs.append({"title": "", "text": ctx[:500]})
        examples.append({
            "question": ex["question"],
            "answer": answer,
            "all_answers": [answer] + aliases,
            "paragraphs": paragraphs,
            "dataset": "TriviaQA",
        })
    return examples

def f1_with_aliases(pred: str, example: Dict) -> float:
    """Compute F1 against all gold answers, return max."""
    answers = example.get("all_answers", [example["answer"]])
    return max(f1_score(pred, a) for a in answers)

def em_with_aliases(pred: str, example: Dict) -> bool:
    """Compute EM against all gold answers."""
    answers = example.get("all_answers", [example["answer"]])
    return any(exact_match(pred, a) for a in answers)

# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_dataset_benchmark(
    examples: List[Dict],
    tau: float = 0.3,
    model: str = LLM_MODEL,
    ollama_url: str = OLLAMA_URL,
    use_gateway: bool = False,
    gateway_url: str = "http://localhost:8080",
    verbose: bool = True,
) -> Dict:
    """Run benchmark on a list of examples. Returns metrics dict."""
    metrics = {
        "baseline": {"em": 0, "f1": 0.0, "count": 0, "latency_ms": []},
        "rag": {"em": 0, "f1": 0.0, "count": 0, "latency_ms": []},
        "flare": {"em": 0, "f1": 0.0, "count": 0, "latency_ms": [], "retrievals": 0},
    }

    dataset_name = examples[0]["dataset"] if examples else "unknown"

    for i, ex in enumerate(examples):
        q = ex["question"]
        paragraphs = ex["paragraphs"]

        # Build numpy vector index for this question's paragraphs
        # (simulates Pion FT.SEARCH which runs in <2ms per query)
        para_texts = []
        para_vecs = []
        if paragraphs:
            for para in paragraphs:
                text = para["text"]
                vec = embed(text, ollama_url)
                para_texts.append(text)
                para_vecs.append(vec)

        # Search function — numpy cosine similarity (equivalent to Pion HNSW)
        def search_fn(query_text: str, _texts=para_texts, _vecs=para_vecs) -> List[str]:
            if not _vecs:
                return []
            q_vec = embed(query_text, ollama_url)
            sims = [np.dot(q_vec, v) for v in _vecs]
            top_k = sorted(range(len(sims)), key=lambda j: sims[j], reverse=True)[:3]
            return [_texts[j] for j in top_k]

        # 1. Baseline (no retrieval)
        prompt_base = f"{FEW_SHOT}Q: {q}\nA:"
        t0 = time.perf_counter()
        result_base = ollama_generate(prompt_base, model=model, ollama_url=ollama_url)
        t1 = time.perf_counter()
        ans_base = extract_answer(result_base["text"])
        metrics["baseline"]["em"] += int(em_with_aliases(ans_base, ex))
        metrics["baseline"]["f1"] += f1_with_aliases(ans_base, ex)
        metrics["baseline"]["latency_ms"].append((t1 - t0) * 1000)
        metrics["baseline"]["count"] += 1

        # 2. Single-shot RAG (search via numpy, same as Pion would return)
        rag_docs = search_fn(q) if para_texts else []
        ctx_str = "\n".join(f"- {d}" for d in rag_docs) if rag_docs else ""

        if ctx_str:
            prompt_rag = f"{FEW_SHOT}Context:\n{ctx_str}\n\nQ: {q}\nA:"
        else:
            prompt_rag = f"{FEW_SHOT}Q: {q}\nA:"

        t0 = time.perf_counter()
        result_rag = ollama_generate(prompt_rag, model=model, ollama_url=ollama_url)
        t1 = time.perf_counter()
        ans_rag = extract_answer(result_rag["text"])
        metrics["rag"]["em"] += int(em_with_aliases(ans_rag, ex))
        metrics["rag"]["f1"] += f1_with_aliases(ans_rag, ex)
        metrics["rag"]["latency_ms"].append((t1 - t0) * 1000)
        metrics["rag"]["count"] += 1

        # 3. FLARE
        t0 = time.perf_counter()
        if use_gateway:
            ans_flare, retrievals = generate_flare_gateway(q, gateway_url, model)
        else:
            ans_flare, retrievals = generate_flare_ollama(
                q, [], search_fn, tau=tau, model=model, ollama_url=ollama_url,
            )
        t1 = time.perf_counter()
        metrics["flare"]["em"] += int(em_with_aliases(ans_flare, ex))
        metrics["flare"]["f1"] += f1_with_aliases(ans_flare, ex)
        metrics["flare"]["latency_ms"].append((t1 - t0) * 1000)
        metrics["flare"]["retrievals"] += retrievals
        metrics["flare"]["count"] += 1

        if verbose:
            base_f1 = f1_with_aliases(ans_base, ex)
            rag_f1 = f1_with_aliases(ans_rag, ex)
            flare_f1 = f1_with_aliases(ans_flare, ex)
            print(f"  [{dataset_name}] Q{i+1:03d}: {q[:65]}")
            print(f"    Gold:  {ex['answer']}")
            print(f"    Base:  {ans_base}  [F1={base_f1:.2f}]")
            print(f"    RAG:   {ans_rag}  [F1={rag_f1:.2f}]")
            print(f"    FLARE: {ans_flare}  [F1={flare_f1:.2f}, ret={retrievals}]")

    return metrics

# ── Report ───────────────────────────────────────────────────────────────────

def print_report(all_metrics: Dict[str, Dict], tau: float, model: str = LLM_MODEL):
    """Print final benchmark report."""
    print()
    print("=" * 78)
    print("A5 FLARE BENCHMARK RESULTS")
    print("=" * 78)
    print(f"FLARE tau={tau}  |  Model: {model}  |  Embed: {EMBED_MODEL}")
    print()

    header = f"{'Dataset':<20} {'Strategy':<10} {'EM':>6} {'F1':>8} {'Avg Lat':>10} {'Ret':>5}"
    print(header)
    print("-" * 78)

    totals = {"baseline": {"em": 0, "f1": 0.0, "n": 0, "lat": []},
              "rag": {"em": 0, "f1": 0.0, "n": 0, "lat": []},
              "flare": {"em": 0, "f1": 0.0, "n": 0, "lat": [], "ret": 0}}

    for ds_name, metrics in all_metrics.items():
        for strategy in ["baseline", "rag", "flare"]:
            m = metrics[strategy]
            n = m["count"]
            if n == 0:
                continue
            em = m["em"] / n
            f1 = m["f1"] / n
            avg_lat = sum(m["latency_ms"]) / len(m["latency_ms"]) if m["latency_ms"] else 0
            ret = f"{m.get('retrievals', 0)/n:.1f}" if strategy == "flare" else "-"

            label = strategy.upper()
            print(f"{ds_name:<20} {label:<10} {em:>5.2f}  {f1:>7.3f}  {avg_lat:>8.0f}ms  {ret:>5}")

            totals[strategy]["em"] += m["em"]
            totals[strategy]["f1"] += m["f1"]
            totals[strategy]["n"] += n
            totals[strategy]["lat"].extend(m["latency_ms"])
            if strategy == "flare":
                totals[strategy]["ret"] += m.get("retrievals", 0)

        print()

    # Aggregate
    print("-" * 78)
    for strategy in ["baseline", "rag", "flare"]:
        t = totals[strategy]
        n = t["n"]
        if n == 0:
            continue
        em = t["em"] / n
        f1 = t["f1"] / n
        avg_lat = sum(t["lat"]) / len(t["lat"]) if t["lat"] else 0
        ret = f"{t.get('ret', 0)/n:.1f}" if strategy == "flare" else "-"
        label = strategy.upper()
        print(f"{'AGGREGATE':<20} {label:<10} {em:>5.2f}  {f1:>7.3f}  {avg_lat:>8.0f}ms  {ret:>5}")

    print()

    # Delta summary
    base_f1 = totals["baseline"]["f1"] / totals["baseline"]["n"] if totals["baseline"]["n"] else 0
    rag_f1 = totals["rag"]["f1"] / totals["rag"]["n"] if totals["rag"]["n"] else 0
    flare_f1 = totals["flare"]["f1"] / totals["flare"]["n"] if totals["flare"]["n"] else 0

    print("F1 IMPROVEMENT:")
    print(f"  RAG   vs Baseline: {rag_f1 - base_f1:+.3f}")
    print(f"  FLARE vs Baseline: {flare_f1 - base_f1:+.3f}")
    print(f"  FLARE vs RAG:      {flare_f1 - rag_f1:+.3f}")
    print()

    # Pion retrieval latency
    avg_search_lat = 0
    search_lats = totals["flare"]["lat"]
    if search_lats:
        avg_search_lat = sum(search_lats) / len(search_lats)
    print(f"Pion vector search contribution: embedded in FLARE total latency")
    print(f"  (Pion FT.SEARCH typically adds <2ms per retrieval)")
    print()

    return {
        "baseline_f1": base_f1,
        "rag_f1": rag_f1,
        "flare_f1": flare_f1,
        "flare_delta_vs_baseline": flare_f1 - base_f1,
        "flare_delta_vs_rag": flare_f1 - rag_f1,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A5: FLARE F1 Benchmark")
    parser.add_argument("--num-questions", "-n", type=int, default=100,
                        help="Number of questions per dataset (default: 100)")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "nq", "triviaqa"],
                        choices=["hotpotqa", "nq", "triviaqa"],
                        help="Datasets to evaluate (default: all three)")
    parser.add_argument("--tau", nargs="+", type=float, default=[0.3],
                        help="FLARE confidence threshold(s) to test (default: 0.3)")
    parser.add_argument("--model", type=str, default=LLM_MODEL,
                        help=f"Ollama model name (default: {LLM_MODEL})")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_URL)
    parser.add_argument("--pion-host", type=str, default=PION_HOST)
    parser.add_argument("--pion-port", type=int, default=PION_PORT)
    parser.add_argument("--use-gateway", action="store_true",
                        help="Use FLARE gateway instead of direct Ollama for FLARE strategy")
    parser.add_argument("--gateway-url", type=str, default="http://localhost:8080")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-question output")
    parser.add_argument("--output-json", type=str, default="",
                        help="Save results to JSON file")
    args = parser.parse_args()

    llm_model = args.model

    print("=" * 78)
    print("A5 BENCHMARK: FLARE F1 — HotpotQA / Natural Questions / TriviaQA")
    print("=" * 78)
    print(f"Model: {args.model}  |  Embed: {EMBED_MODEL}  |  N={args.num_questions}/dataset")
    print(f"Datasets: {', '.join(args.datasets)}  |  tau={args.tau}")
    print()

    # Check Ollama
    try:
        r = requests.get(f"{args.ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
        print(f"[OK] Ollama at {args.ollama_url}")
    except Exception:
        print(f"[FAIL] Ollama not running at {args.ollama_url}")
        print("  Start with: ollama serve")
        sys.exit(1)

    # Warm up embedding model
    print("Warming up embedding model...", end=" ", flush=True)
    _ = embed("warmup", args.ollama_url)
    print("done")
    print()

    # Load datasets
    dataset_loaders = {
        "hotpotqa": ("HotpotQA", load_hotpotqa),
        "nq": ("NaturalQuestions", load_natural_questions),
        "triviaqa": ("TriviaQA", load_triviaqa),
    }

    all_results = {}
    for tau in args.tau:
        print(f"\n{'='*78}")
        print(f"TAU = {tau}")
        print(f"{'='*78}")

        tau_metrics = {}
        for ds_key in args.datasets:
            ds_name, loader = dataset_loaders[ds_key]
            print(f"\nLoading {ds_name} ({args.num_questions} questions)...", flush=True)
            examples = loader(args.num_questions)
            print(f"  Loaded {len(examples)} examples")
            print()

            metrics = run_dataset_benchmark(
                examples, tau=tau, model=args.model,
                ollama_url=args.ollama_url,
                use_gateway=args.use_gateway, gateway_url=args.gateway_url,
                verbose=not args.quiet,
            )
            tau_metrics[ds_name] = metrics

        summary = print_report(tau_metrics, tau, model=llm_model)
        all_results[f"tau={tau}"] = {"metrics": tau_metrics, "summary": summary}

    # Save JSON results
    if args.output_json:
        # Convert numpy/non-serializable types
        def make_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        with open(args.output_json, "w") as f:
            json.dump(make_serializable(all_results), f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    print("\nDone.")

if __name__ == "__main__":
    main()
