#!/usr/bin/env python3
"""Corpus reuse-density / near-duplicate measurement harness.

Reusable on any corpus to answer the three questions a "should we build
an embedding-keyed K/V cache?" decision needs:

  A.1 — chunk-ID reuse rate across queries
        (how often does query N's top-k overlap with query M's top-k?)
  A.2 — near-duplicate cluster density in the corpus
        (what fraction of chunks have a cosine ≥ θ twin elsewhere?)
  A.3 — quality-on-substitution
        (does swapping a near-neighbor chunk's K/V preserve answer
        quality at high embedding similarity?)

Originally built for the External Parametric Memory Stage 2 reframe
Stage A (external-parametric-memory reframe results),
which returned NEGATIVE on Pion's own docs corpus (0.9% near-dup
density vs ≥5% gate). Kept after that lineage closed because the
measurement applies to any future design that proposes embedding-
keyed cache reuse. Renamed from `benchmarks/knnlm/stage2_reality_check.py`
during 2026-04-28 cleanup.

Inputs:
- A query workload (default: `pion-serve/qa_dataset_gemma4.jsonl`).
- A corpus chunked by `## ` headings (default: `doc/*.md`).
- A retriever endpoint (default: Pion `FT.SEARCH KNN k=5`).
- An embedding model (default: sentence-transformers/all-MiniLM-L6-v2).
- A generator for A.3 (default: GPT-2 medium, consistency-only).

Usage:
    pkill -9 -x pion-server
    rm -f pion.wal.0 pion.hnsw.0
    ./pion-server --profile vector --no-auto-embed --dim 384 -w 1 &
    python3 benchmarks/corpus_reuse_density.py
"""
import argparse
import collections
import json
import os
import re
import sys
import time
from typing import List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PION_ROOT = os.path.dirname(os.path.dirname(HERE))
INDEX_NAME = "stage_a_idx"
KEY_PREFIX = "c:"
EMBED_DIM = 384
N_QUERIES = 200       # full qa_dataset_gemma4.jsonl
TOP_K = 5             # retriever top-k
NN_THRESHOLD = 0.90   # near-duplicate cosine threshold
SUBSTITUTION_PAIRS = 50  # for A.3 quality measurement
RAG_GEN_TOKENS = 16   # short answer for A.3


# ── Corpus loader ─────────────────────────────────────────────────────
def load_corpus(doc_dir: str, project_root: str) -> List[Tuple[str, str, str]]:
    """Return [(chunk_id, source_doc_relative_path, chunk_text), ...].

    chunk_id format: '<source_doc>#<heading>' to match qa_dataset_gemma4
    `source_doc` + `heading` fields exactly.
    """
    chunks = []
    paths = []
    # All doc/*.md
    for f in sorted(os.listdir(doc_dir)):
        if f.endswith(".md"):
            paths.append(os.path.join(doc_dir, f))
    # Plus CLAUDE.md at project root, when present
    claude_md = os.path.join(project_root, "CLAUDE.md")
    if os.path.exists(claude_md):
        paths.append(claude_md)

    for p in paths:
        rel = os.path.relpath(p, project_root)
        with open(p) as fh:
            text = fh.read()
        # Split by `## ` headings (h2). Keep h1 + preamble as 'preamble'.
        sections = re.split(r"\n(?=## )", text)
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            # Find heading
            m = re.match(r"^## (.+?)$", sec, re.MULTILINE)
            heading = m.group(1).strip() if m else "(preamble)"
            chunk_id = f"{rel}#{heading}"
            # Limit to ~2000 chars (~500 tokens) so embeddings stay focused
            chunk_text = sec[:2000]
            chunks.append((chunk_id, rel, chunk_text))
    return chunks


def load_queries(jsonl_path: str, n: int) -> List[dict]:
    qa = [json.loads(l) for l in open(jsonl_path)]
    return qa[:n]


# ── Embedding (sentence-transformers MiniLM-L6-v2) ────────────────────
_EMB_MODEL = None
def get_embedder():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"[stage_a] loading sentence-transformers/all-MiniLM-L6-v2...", flush=True)
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL


def embed(texts: List[str]) -> np.ndarray:
    return get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── Pion client (redis-py) ─────────────────────────────────────────────
def pion_connect(host: str, port: int):
    import redis
    return redis.Redis(host=host, port=port, socket_timeout=60)


def build_index(r, chunks: List[Tuple[str, str, str]]) -> np.ndarray:
    """Embed all chunks, load into Pion via FT.CREATE+HSET, return embeddings array."""
    print(f"[stage_a] embedding {len(chunks)} chunks...", flush=True)
    texts = [c[2] for c in chunks]
    embs = embed(texts).astype(np.float32)
    print(f"[stage_a] embeddings shape: {embs.shape}", flush=True)

    # Reset
    try:
        r.execute_command("FT.DROPINDEX", INDEX_NAME)
    except Exception:
        pass
    keys = r.keys(KEY_PREFIX + "*")
    if keys:
        r.delete(*keys)
    r.execute_command(
        "FT.CREATE", INDEX_NAME,
        "ON", "HASH", "PREFIX", "1", KEY_PREFIX,
        "SCHEMA",
        "vector", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32",
            "DIM", str(EMBED_DIM),
            "DISTANCE_METRIC", "COSINE",
        "chunk_id", "TAG",
    )
    print(f"[stage_a] FT.CREATE ok", flush=True)
    pipe = r.pipeline(transaction=False)
    for i, (chunk_id, _, _) in enumerate(chunks):
        blob = embs[i].tobytes()
        pipe.hset(f"{KEY_PREFIX}{i}", mapping={"vector": blob, "chunk_id": chunk_id})
        if (i + 1) % 200 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
    pipe.execute()
    r.execute_command("FT.OPTIMIZE", INDEX_NAME)
    print(f"[stage_a] index built", flush=True)
    return embs


def vsim_topk(r, q_emb: np.ndarray, k: int) -> List[int]:
    """Return top-k chunk indices (the i in c:i) for query embedding."""
    blob = q_emb.astype(np.float32).tobytes()
    res = r.execute_command(
        "FT.SEARCH", INDEX_NAME,
        f"*=>[KNN {k} @vector $vec EF_RUNTIME 150 as score]",
        "SORTBY", "score",
        "PARAMS", "2", "vec", blob,
        "DIALECT", "2",
    )
    # res = [count, key1, [field, val, ...], key2, [...], ...]
    out = []
    count = res[0] if isinstance(res[0], int) else int(res[0])
    for ri in range(min(count, k)):
        key = res[1 + ri * 2]
        if isinstance(key, bytes):
            key = key.decode()
        # key looks like "c:42" → 42
        out.append(int(key.split(":", 1)[1]))
    return out


# ── A.1: deterministic chunk-ID reuse rate ────────────────────────────
def measure_a1(r, queries: List[dict], chunks: List[Tuple[str, str, str]],
               q_embs: np.ndarray) -> dict:
    """For all (q_i, q_j) pairs: count fraction sharing ≥1 chunk-ID in top-K.

    Also: retrieval recall vs gold source_doc (when query has source_doc set,
    is the gold chunk in the top-K?).
    """
    print(f"\n[A.1] retrieving top-{TOP_K} per query for {len(queries)} queries...", flush=True)
    chunk_id_by_idx = [c[0] for c in chunks]
    src_doc_by_idx = [c[1] for c in chunks]
    per_query_topk: List[List[int]] = []
    gold_in_topk = 0
    gold_in_top1 = 0
    n_with_gold = 0

    for qi, q in enumerate(queries):
        topk = vsim_topk(r, q_embs[qi], TOP_K)
        per_query_topk.append(topk)
        # Gold check: query's source_doc + heading should match some retrieved chunk_id
        gold_chunk_id = f"{q.get('source_doc','')}#{q.get('heading','')}"
        if q.get("source_doc"):
            n_with_gold += 1
            top_ids = [chunk_id_by_idx[i] for i in topk]
            if gold_chunk_id in top_ids:
                gold_in_topk += 1
            if topk and chunk_id_by_idx[topk[0]] == gold_chunk_id:
                gold_in_top1 += 1
        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(queries)}", flush=True)

    # Pairwise overlap
    n = len(queries)
    pair_count = 0
    overlap_count = 0
    for i in range(n):
        si = set(per_query_topk[i])
        for j in range(i + 1, n):
            sj = set(per_query_topk[j])
            pair_count += 1
            if si & sj:
                overlap_count += 1

    reuse_rate = overlap_count / pair_count if pair_count else 0.0
    recall_at_k = gold_in_topk / max(n_with_gold, 1)
    recall_at_1 = gold_in_top1 / max(n_with_gold, 1)

    print(f"\n[A.1] RESULT")
    print(f"  pairwise chunk-ID overlap rate (any of top-{TOP_K} shared): {reuse_rate*100:.2f}%")
    print(f"  retriever recall@{TOP_K} vs gold: {recall_at_k*100:.2f}%  ({gold_in_topk}/{n_with_gold})")
    print(f"  retriever recall@1 vs gold: {recall_at_1*100:.2f}%  ({gold_in_top1}/{n_with_gold})")

    gate_pass = reuse_rate < 0.70
    print(f"  GATE A.1 (< 70%): {'PASS' if gate_pass else 'FAIL — hash-key already wins'}")

    return {
        "pairwise_reuse_rate": reuse_rate,
        "recall_at_k": recall_at_k,
        "recall_at_1": recall_at_1,
        "n_queries": n,
        "n_with_gold": n_with_gold,
        "gate_pass": gate_pass,
        "per_query_topk": per_query_topk,
    }


# ── A.2: near-duplicate cluster density ───────────────────────────────
def measure_a2(chunks: List[Tuple[str, str, str]], embs: np.ndarray) -> dict:
    """For each chunk, find the nearest non-self chunk; report fraction with
    nearest cosine sim ≥ NN_THRESHOLD."""
    print(f"\n[A.2] computing pairwise cosine sim over {len(chunks)} chunks...", flush=True)
    # Embeddings are already L2-normalized → cosine = dot product
    sim = embs @ embs.T  # [N, N]
    np.fill_diagonal(sim, -1.0)
    nn_sim = sim.max(axis=1)
    nn_idx = sim.argmax(axis=1)
    n_with_dup = int((nn_sim >= NN_THRESHOLD).sum())
    density = n_with_dup / len(chunks)
    print(f"\n[A.2] RESULT")
    print(f"  chunks with a near-duplicate (cosine ≥ {NN_THRESHOLD}): {n_with_dup}/{len(chunks)} = {density*100:.2f}%")
    # Show top examples
    print(f"  sample near-duplicate pairs:")
    top = np.argsort(-nn_sim)[:5]
    for i in top:
        if nn_sim[i] >= NN_THRESHOLD:
            print(f"    {chunks[i][0]}  ⇄  {chunks[nn_idx[i]][0]}   cos={nn_sim[i]:.3f}")
    gate_pass = density >= 0.05
    print(f"  GATE A.2 (≥ 5%): {'PASS' if gate_pass else 'FAIL — corpus has no duplicates to address'}")
    return {
        "n_chunks": len(chunks),
        "n_with_duplicate": n_with_dup,
        "density": density,
        "gate_pass": gate_pass,
        "nn_sim": nn_sim.tolist(),
        "nn_idx": nn_idx.tolist(),
    }


# ── A.3: quality on near-duplicate substitution ───────────────────────
_GEN_MODEL = None
_GEN_TOK = None
def get_generator():
    global _GEN_MODEL, _GEN_TOK
    if _GEN_MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[A.3] loading gpt2-medium for generation...", flush=True)
        _GEN_TOK = AutoTokenizer.from_pretrained("gpt2-medium")
        if _GEN_TOK.pad_token_id is None:
            _GEN_TOK.pad_token = _GEN_TOK.eos_token
        _GEN_MODEL = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        _GEN_MODEL.eval()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _GEN_MODEL.to(device)
    return _GEN_MODEL, _GEN_TOK


def rag_generate(question: str, chunk_text: str, max_new: int = RAG_GEN_TOKENS) -> str:
    import torch
    model, tok = get_generator()
    device = next(model.parameters()).device
    prompt = f"Context: {chunk_text[:1200]}\n\nQuestion: {question}\nAnswer:"
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=900).to(device)["input_ids"]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    if "Answer:" in full:
        ans = full.split("Answer:", 1)[1]
    else:
        ans = full
    return ans.split("\n", 1)[0].strip()


def rouge_l(pred: str, ref: str) -> float:
    """Cheap ROUGE-L F1 (LCS-based) — no external deps."""
    a = pred.lower().split()
    b = ref.lower().split()
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


def measure_a3(queries, chunks, q_embs, a1, a2) -> dict:
    """For SUBSTITUTION_PAIRS pairs: pick (query, retrieved_top1=X) where the
    retriever returned the gold chunk-X; find Y ≠ X with cos(X,Y) ≥ NN_THRESHOLD;
    generate two answers and measure ROUGE-L delta.
    """
    print(f"\n[A.3] selecting {SUBSTITUTION_PAIRS} substitution pairs...", flush=True)
    embs_arr = np.array([])  # not used; we use a2 nn_sim/nn_idx
    nn_sim = np.asarray(a2["nn_sim"])
    nn_idx = np.asarray(a2["nn_idx"])

    chunk_id_by_idx = [c[0] for c in chunks]
    chunk_text_by_idx = [c[2] for c in chunks]

    pairs = []  # (query, X_idx, Y_idx)
    for qi, q in enumerate(queries):
        if len(pairs) >= SUBSTITUTION_PAIRS:
            break
        topk = a1["per_query_topk"][qi]
        if not topk:
            continue
        x = topk[0]
        if nn_sim[x] < NN_THRESHOLD:
            continue
        y = int(nn_idx[x])
        if y == x:
            continue
        pairs.append((q, x, y))

    if not pairs:
        print(f"[A.3] NO ELIGIBLE PAIRS — corpus has no near-duplicates the retriever lands on")
        return {
            "n_pairs": 0,
            "mean_rouge_x": 0.0,
            "mean_rouge_y": 0.0,
            "mean_drop": 0.0,
            "gate_pass": False,
            "samples": [],
        }
    print(f"[A.3] {len(pairs)} eligible pairs found", flush=True)

    drops = []
    rouge_x_list = []
    rouge_y_list = []
    samples = []
    t0 = time.perf_counter()
    for pi, (q, x, y) in enumerate(pairs):
        ans_x = rag_generate(q["question"], chunk_text_by_idx[x])
        ans_y = rag_generate(q["question"], chunk_text_by_idx[y])
        # Reference: the gold answer from the qa dataset
        gold = q.get("answer", "")
        rouge_x = rouge_l(ans_x, gold)
        rouge_y = rouge_l(ans_y, gold)
        drops.append(rouge_x - rouge_y)
        rouge_x_list.append(rouge_x)
        rouge_y_list.append(rouge_y)
        if pi < 5:
            samples.append({
                "q": q["question"][:100],
                "X_chunk": chunk_id_by_idx[x],
                "Y_chunk": chunk_id_by_idx[y],
                "cos_xy": float(nn_sim[x]),
                "ans_X": ans_x[:80],
                "ans_Y": ans_y[:80],
                "gold": gold[:80],
                "rouge_X": rouge_x,
                "rouge_Y": rouge_y,
                "drop": rouge_x - rouge_y,
            })
        if (pi + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"  {pi+1}/{len(pairs)}  mean_drop={np.mean(drops):+.4f}  ({(pi+1)/el:.1f} pair/s)", flush=True)

    mean_drop = float(np.mean(drops))
    print(f"\n[A.3] RESULT")
    print(f"  pairs evaluated: {len(pairs)}")
    print(f"  mean ROUGE-L using X (retrieved gold chunk): {np.mean(rouge_x_list):.4f}")
    print(f"  mean ROUGE-L using Y (near-duplicate): {np.mean(rouge_y_list):.4f}")
    print(f"  mean drop (X − Y): {mean_drop:+.4f}")
    gate_pass = mean_drop <= 0.10
    print(f"  GATE A.3 (drop ≤ 0.10): {'PASS' if gate_pass else 'FAIL — wrong-chunk K/V corrupts answers'}")
    return {
        "n_pairs": len(pairs),
        "mean_rouge_x": float(np.mean(rouge_x_list)),
        "mean_rouge_y": float(np.mean(rouge_y_list)),
        "mean_drop": mean_drop,
        "gate_pass": gate_pass,
        "samples": samples,
    }


# ── Writeup ────────────────────────────────────────────────────────────
def write_artefact(workload_path: str, corpus_chunks: int,
                   a1: dict, a2: dict, a3: dict, out_path: str) -> None:
    with open(out_path, "w") as f:
        f.write("# Stage A — Reality Check Results\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n")
        f.write(f"**Workload:** `{workload_path}` ({a1['n_queries']} queries; "
                f"{a1['n_with_gold']} have ground-truth source_doc)\n")
        f.write(f"**Corpus:** {corpus_chunks} chunks (## headings of doc/*.md)\n")
        f.write(f"**Embedding:** sentence-transformers/all-MiniLM-L6-v2 (384-dim, normalized)\n")
        f.write(f"**Retriever:** Pion FT.SEARCH KNN k={TOP_K} ef_runtime=150\n\n")
        f.write("---\n\n")
        f.write("## Per-gate result\n\n")
        f.write("| Gate | Threshold | Measured | Verdict |\n|---|---|---:|---|\n")
        f.write(f"| A.1 — chunk-ID reuse rate | < 70% | "
                f"**{a1['pairwise_reuse_rate']*100:.2f}%** | "
                f"{'**PASS** (embedding-key has room)' if a1['gate_pass'] else '**FAIL** — hash-key already wins'} |\n")
        f.write(f"| A.2 — near-duplicate density | ≥ 5% | "
                f"**{a2['density']*100:.2f}%** ({a2['n_with_duplicate']}/{a2['n_chunks']}) | "
                f"{'**PASS** (addressable surface exists)' if a2['gate_pass'] else '**FAIL** — corpus has no duplicates'} |\n")
        if a3["n_pairs"] > 0:
            f.write(f"| A.3 — substitution quality drop | drop ≤ 0.10 | "
                    f"**{a3['mean_drop']:+.4f}** ROUGE-L (X={a3['mean_rouge_x']:.3f}, Y={a3['mean_rouge_y']:.3f}, n={a3['n_pairs']}) | "
                    f"{'**PASS** (near-dups preserve quality)' if a3['gate_pass'] else '**FAIL** — wrong-chunk K/V corrupts answers'} |\n")
        else:
            f.write(f"| A.3 — substitution quality drop | drop ≤ 0.10 | NO ELIGIBLE PAIRS | "
                    "**N/A** (A.2 pass was paper-thin or retriever didn't land on near-duplicate chunks) |\n")
        f.write("\n---\n\n")
        f.write("## Decision\n\n")
        all_pass = a1["gate_pass"] and a2["gate_pass"] and (a3["n_pairs"] == 0 or a3["gate_pass"])
        if all_pass:
            f.write("**ALL GATES PASS — proceed to Stage B (7-day demo).**\n\n"
                    "Stage 2 reframe survives the reality check. The embedding-keyed K/V "
                    "lookup has both an addressable surface (chunks rotate) and tolerable "
                    "quality on near-duplicate substitution. Next step: §6 Day 1 — wire "
                    "`KV.FETCH` into `PionPromptCache`.\n\n")
        else:
            f.write("**AT LEAST ONE GATE FAILS — kill the reframe, fall back to Option 3.**\n\n")
            for k, name in (("a1", "A.1 chunk-ID reuse"), ("a2", "A.2 near-duplicate density"), ("a3", "A.3 substitution quality")):
                d = locals()[k]
                if not d.get("gate_pass", True):
                    f.write(f"- {name} failed — see §{k.upper()} reading below.\n")
            f.write("\nPion's already-shipped `KV.PREFIX.*` 89.2% TTFT win + `ATTEND.PREFIX.*` "
                    "146× win + `AI.KNN_LM.*` PPL win remain the differentiated story without further R&D.\n\n")

        f.write("---\n\n## Detail\n\n")
        f.write(f"### A.1 Retrieval recall vs gold\n\n"
                f"- recall@{TOP_K}: **{a1['recall_at_k']*100:.2f}%**\n"
                f"- recall@1: **{a1['recall_at_1']*100:.2f}%**\n\n"
                f"This is a sanity check — if retriever can't find the gold chunk, the entire pipeline is broken. "
                f"A high recall@1 also means the retriever is *deterministic enough* that hash-key would match.\n\n")
        f.write(f"### A.2 Top near-duplicate samples\n\n")
        if a3.get("samples"):
            f.write(f"### A.3 Generation samples (first 5 of {a3['n_pairs']} pairs)\n\n")
            for s in a3["samples"]:
                f.write(f"- **Q:** {s['q']}\n")
                f.write(f"  - X = `{s['X_chunk']}` → `{s['ans_X']}` (ROUGE={s['rouge_X']:.3f})\n")
                f.write(f"  - Y = `{s['Y_chunk']}` (cos={s['cos_xy']:.3f}) → `{s['ans_Y']}` (ROUGE={s['rouge_Y']:.3f})\n")
                f.write(f"  - gold: `{s['gold']}`\n")
                f.write(f"  - drop: {s['drop']:+.3f}\n\n")
    print(f"\n[stage_a] wrote {out_path}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--queries", type=int, default=N_QUERIES)
    ap.add_argument("--out", default=os.path.join(PION_ROOT, "stage2_reality_check_results.md"))
    args = ap.parse_args()

    workload_path = os.path.join(PION_ROOT, "pion-serve", "qa_dataset_gemma4.jsonl")
    doc_dir = os.path.join(PION_ROOT, "doc")

    print(f"[stage_a] workload: {workload_path}")
    print(f"[stage_a] corpus root: {doc_dir}")

    chunks = load_corpus(doc_dir, PION_ROOT)
    print(f"[stage_a] corpus: {len(chunks)} chunks from {len(set(c[1] for c in chunks))} docs")

    queries = load_queries(workload_path, args.queries)
    print(f"[stage_a] queries: {len(queries)}")

    r = pion_connect(args.host, args.port)
    try:
        r.execute_command("PING")
    except Exception as e:
        print(f"[stage_a] FATAL: cannot reach Pion at {args.host}:{args.port}: {e}", file=sys.stderr)
        return 2

    # Build index
    chunk_embs = build_index(r, chunks)
    # Embed queries
    print(f"[stage_a] embedding {len(queries)} queries...", flush=True)
    q_texts = [q["question"] for q in queries]
    q_embs = embed(q_texts).astype(np.float32)

    a1 = measure_a1(r, queries, chunks, q_embs)
    a2 = measure_a2(chunks, chunk_embs)
    a3 = measure_a3(queries, chunks, q_embs, a1, a2)

    write_artefact(workload_path, len(chunks), a1, a2, a3, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
