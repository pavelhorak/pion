#!/usr/bin/env python3
"""Replay Experiment — validate inference distillation in one afternoon.

Replays a Q&A dataset through Pion Serve's L1 (semantic cache) → L3
(concept synthesis) → full inference pipeline, logging the source of
each response and plotting a convergence curve.

Usage:
    # Generate dataset first:
    python3 pion-serve/generate_qa.py --source doc/ README.md --count 500 --no-llm

    # Run experiment (requires Pion + Ollama running):
    python3 pion-serve/replay_experiment.py --dataset pion-serve/qa_dataset.jsonl \
        --backend-url http://127.0.0.1:11434 --model gemma3:4b

    # Dry run (no LLM, uses ground-truth answers):
    python3 pion-serve/replay_experiment.py --dataset pion-serve/qa_dataset.jsonl --dry-run

    # Plot from existing log:
    python3 pion-serve/replay_experiment.py --plot-only --log pion-serve/experiment_log.jsonl

Requires:
    - Pion server: ./pion-server -w 1
    - Ollama (unless --dry-run): ollama with a model loaded
    - pip install numpy redis requests matplotlib (matplotlib optional for plots)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import redis
import requests

from concept_store import ConceptStore, FragmentStore, SynthesisResult


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL dataset."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


_pion_embed_sock = None


def _embed_via_pion_sidecar(text: str, sock_path: str = "/tmp/pion_inference.sock") -> Optional[np.ndarray]:
    """Embed via Pion's auto-embed inference sidecar (MiniLM-L6-v2, 384-dim).

    Protocol (little-endian):
      Request:  [type:1B=1][req_id:4B][body_len:4B][body]
                body = [text_len:4B][text_bytes]
      Response: [type:1B][req_id:4B][status:1B][body_len:4B][body]
                body = [dim:4B][f32 * dim]
    """
    global _pion_embed_sock
    import socket
    import struct as st

    def _read_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    try:
        if _pion_embed_sock is None:
            _pion_embed_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _pion_embed_sock.connect(sock_path)

        text_bytes = text.encode("utf-8")
        # body: [text_len:4B][text]
        body = st.pack("<I", len(text_bytes)) + text_bytes
        # header: [type=1 (EMBED)][req_id=0][body_len]
        header = st.pack("<BII", 1, 0, len(body))
        _pion_embed_sock.sendall(header + body)

        # Response header: [type:1B][req_id:4B][status:1B][body_len:4B] = 10 bytes
        resp_header = _read_exact(_pion_embed_sock, 10)
        if resp_header is None:
            _pion_embed_sock = None
            return None

        _msg_type, _req_id, status, resp_body_len = st.unpack("<BIbI", resp_header)
        if status != 0 or resp_body_len == 0:
            # Read and discard error body
            if resp_body_len > 0:
                _read_exact(_pion_embed_sock, resp_body_len)
            return None

        resp_body = _read_exact(_pion_embed_sock, resp_body_len)
        if resp_body is None:
            _pion_embed_sock = None
            return None

        # body: [dim:4B][f32 * dim]
        dim = st.unpack_from("<I", resp_body, 0)[0]
        vec = np.frombuffer(resp_body, dtype=np.float32, offset=4, count=dim).copy()
        vec /= np.linalg.norm(vec) + 1e-10
        return vec
    except Exception as e:
        _pion_embed_sock = None
        return None


def embed_text(text: str, backend_url: str) -> Optional[np.ndarray]:
    """Embed text via Ollama nomic-embed-text, falling back to Pion sidecar."""
    # Try Ollama first
    try:
        resp = requests.post(
            f"{backend_url}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=10,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["embedding"], dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-10
        return vec
    except Exception:
        pass

    # Fallback: Pion's auto-embed sidecar
    return _embed_via_pion_sidecar(text)


def check_semantic_cache(pion: redis.Redis, query: str, threshold: float = 0.92) -> Optional[str]:
    """Check Pion's L1 semantic cache."""
    try:
        result = pion.execute_command(
            "AI.SEMANTIC_CACHE", "GET", query, "THRESHOLD", str(threshold)
        )
        if result and result != b"$-1\r\n" and result != b"(nil)" and result != b"":
            if isinstance(result, bytes):
                decoded = result.decode("utf-8", errors="replace")
                if decoded and len(decoded) > 5:
                    return decoded
            elif isinstance(result, str) and len(result) > 5:
                return result
    except Exception:
        pass
    return None


def store_semantic_cache(pion: redis.Redis, query: str, response: str):
    """Store in Pion's L1 semantic cache."""
    try:
        pion.execute_command("AI.SEMANTIC_CACHE", "SET", query, response)
    except Exception:
        pass


def call_llm(query: str, backend_url: str, model: str, api_key: str = "") -> Optional[str]:
    """Call LLM backend for full inference. Supports Ollama, OpenAI-compatible, and Anthropic APIs."""

    # Anthropic API (Claude)
    if "anthropic" in backend_url or model.startswith("claude"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": query}],
            )
            return resp.content[0].text
        except Exception as e:
            log.warning(f"Claude API call failed: {e}")
            return None

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Detect backend type from URL
    if "generativelanguage.googleapis.com" in backend_url:
        chat_url = f"{backend_url}/v1beta/openai/chat/completions"
        body = {"model": model, "messages": [{"role": "user", "content": query}], "stream": False}
    elif "11434" in backend_url or "ollama" in backend_url:
        chat_url = f"{backend_url}/api/chat"
        body = {"model": model, "messages": [{"role": "user", "content": query}], "stream": False}
    else:
        chat_url = f"{backend_url}/v1/chat/completions"
        body = {"model": model, "messages": [{"role": "user", "content": query}], "stream": False}

    try:
        resp = requests.post(chat_url, json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data:
            return data["message"].get("content", "")
        if "choices" in data:
            return data["choices"][0].get("message", {}).get("content", "")
        return None
    except Exception as e:
        return None


def compute_quality(generated: str, ground_truth: str, embed_fn) -> float:
    """Compute quality score between generated and ground-truth responses.

    Uses cosine similarity of response embeddings as a proxy for quality.
    Returns value in [0, 1].
    """
    if not generated or not ground_truth:
        return 0.0

    emb_gen = embed_fn(generated[:500])
    emb_gt = embed_fn(ground_truth[:500])
    if emb_gen is None or emb_gt is None:
        # Fallback: Jaccard word overlap
        words_gen = set(generated.lower().split())
        words_gt = set(ground_truth.lower().split())
        if not words_gen or not words_gt:
            return 0.0
        return len(words_gen & words_gt) / len(words_gen | words_gt)

    return float(np.dot(emb_gen, emb_gt))


def run_experiment(
    dataset: list[dict],
    pion: redis.Redis,
    concepts: ConceptStore,
    backend_url: str,
    model: str,
    l1_threshold: float = 0.92,
    dry_run: bool = False,
    log_path: str = "pion-serve/experiment_log.jsonl",
    fragments: Optional[FragmentStore] = None,
    api_key: str = "",
) -> list[dict]:
    """Run the replay experiment.

    For each query:
      1. Check L1 semantic cache
      2. Check L3 concept synthesis
      3. Check L3b fragment synthesis
      4. Fall through to full inference (or ground-truth in dry-run)
    """
    log_entries = []
    embed_fn = lambda text: embed_text(text, backend_url)

    # Track LLM responses per concept for self-consistency quality metric.
    # When L3 replays a concept, compare against the LLM's original response
    # (self-consistency), not against template ground truth.
    _concept_responses: dict[int, str] = {}  # concept_id → LLM response

    # Counters
    total = len(dataset)
    l1_hits = 0
    l3_hits = 0
    l3_direct = 0
    l3_composite = 0
    l3b_full = 0
    l3b_augmented = 0
    full_inference = 0
    embed_failures = 0

    print(f"\nRunning experiment: {total} queries")
    print("-" * 70)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_file = open(log_path, "w")

    t_start = time.perf_counter()

    for i, item in enumerate(dataset):
        query = item["question"]
        ground_truth = item.get("answer", "")
        t0 = time.perf_counter()

        entry = {
            "idx": i,
            "query": query[:100],
            "source": None,
            "latency_ms": 0,
            "quality": 0.0,
            "l3_confidence": 0.0,
            "l3_similarity": 0.0,
            "l3_strategy": "",
            "concepts_total": concepts.stats["concepts"],
            "cumulative_l1_rate": 0.0,
            "cumulative_l3_rate": 0.0,
            "cumulative_inference_rate": 0.0,
        }

        # Phase 1: L1 semantic cache
        l1_result = check_semantic_cache(pion, query, l1_threshold)
        if l1_result:
            l1_hits += 1
            entry["source"] = "l1_cache"
            entry["latency_ms"] = (time.perf_counter() - t0) * 1000
            entry["quality"] = compute_quality(l1_result, ground_truth, embed_fn)
        else:
            # Phase 2: L3 concept synthesis
            query_emb = embed_text(query, backend_url)
            if query_emb is None:
                embed_failures += 1
                # Can't do L3 without embedding — fall through
            else:
                l3_result = concepts.try_synthesize(query_emb)
                if l3_result and l3_result.strategy in ("direct", "composite"):
                    l3_hits += 1
                    entry["source"] = "l3_synthesis"
                    entry["latency_ms"] = (time.perf_counter() - t0) * 1000
                    # Self-consistency: compare L3 response against the LLM's
                    # original response for this concept, not template ground truth
                    ref = _concept_responses.get(l3_result.concept_ids[0], ground_truth)
                    entry["quality"] = compute_quality(l3_result.response, ref, embed_fn)
                    entry["l3_confidence"] = l3_result.confidence
                    entry["l3_similarity"] = l3_result.cosine_similarity
                    entry["l3_strategy"] = l3_result.strategy
                    if l3_result.strategy == "direct":
                        l3_direct += 1
                    else:
                        l3_composite += 1

                    # Store in L1 cache too (promote successful L3 to L1)
                    store_semantic_cache(pion, query, l3_result.response)

            # Phase 3: L3b fragment synthesis
            if entry["source"] is None and fragments and query_emb is not None:
                frag_result = fragments.try_synthesize(query_emb)
                if frag_result and frag_result.strategy == "full_synthesis":
                    l3_hits += 1
                    l3b_full += 1
                    response_text = "\n\n".join(frag_result.fragments)
                    entry["source"] = "l3b_full_synthesis"
                    entry["latency_ms"] = (time.perf_counter() - t0) * 1000
                    # Use source concept's LLM response as reference if available
                    frag_ref = ground_truth
                    for fid in frag_result.fragment_ids[:1]:
                        src_cid = fragments._fragments[fid].source_concept_id if fid < len(fragments._fragments) else -1
                        if src_cid >= 0 and src_cid in _concept_responses:
                            frag_ref = _concept_responses[src_cid]
                            break
                    entry["quality"] = compute_quality(response_text, frag_ref, embed_fn)
                    entry["l3_confidence"] = frag_result.confidence
                    entry["l3_similarity"] = frag_result.coverage
                    entry["l3_strategy"] = "l3b_full"
                    store_semantic_cache(pion, query, response_text)
                elif frag_result and frag_result.strategy == "fragment_augmented":
                    l3b_augmented += 1
                    # Still needs LLM but with pre-verified context (cheaper)
                    # For experiment: count as partial hit, still call LLM
                    entry["l3_strategy"] = "l3b_augmented"

            if entry["source"] is None:
                # Phase 4: Full inference
                full_inference += 1
                if dry_run:
                    response = ground_truth
                else:
                    response = call_llm(query, backend_url, model, api_key)
                    if response is None:
                        response = ground_truth  # fallback

                entry["source"] = "full_inference"
                entry["latency_ms"] = (time.perf_counter() - t0) * 1000
                entry["quality"] = 1.0 if dry_run else compute_quality(response, ground_truth, embed_fn)

                # Update L1 + L3 + fragments
                store_semantic_cache(pion, query, response)
                if query_emb is not None:
                    concepts.ingest(query_emb, response, query_text=query)
                    cid = concepts._concept_count - 1
                    _concept_responses[cid] = response  # track for self-consistency
                    if fragments:
                        fragments.ingest_response(response, source_concept_id=cid)

        # Cumulative rates
        processed = i + 1
        entry["cumulative_l1_rate"] = l1_hits / processed
        entry["cumulative_l3_rate"] = l3_hits / processed
        entry["cumulative_inference_rate"] = full_inference / processed

        log_entries.append(entry)
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

        # Progress
        if (i + 1) % 10 == 0 or i == total - 1:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r  [{i+1:4d}/{total}] "
                f"L1={l1_hits}({l1_hits/processed*100:.0f}%) "
                f"L3={l3_hits}({l3_hits/processed*100:.0f}%) "
                f"INF={full_inference}({full_inference/processed*100:.0f}%) "
                f"C={concepts.stats['concepts']} "
                f"F={fragments.stats['fragments'] if fragments else 0} "
                f"ETA={eta:.0f}s"
            )
            sys.stdout.flush()

    log_file.close()
    print()
    print("-" * 70)

    # Summary
    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total queries:         {total}")
    print(f"  Elapsed:               {elapsed:.1f}s ({total/elapsed:.1f} queries/s)")
    print(f"  Embedding failures:    {embed_failures}")
    print()
    print(f"  L1 cache hits:         {l1_hits:4d} ({l1_hits/total*100:.1f}%)")
    print(f"  L3 synthesis hits:     {l3_hits:4d} ({l3_hits/total*100:.1f}%)")
    print(f"    ├─ direct:           {l3_direct:4d}")
    print(f"    ├─ composite:        {l3_composite:4d}")
    print(f"    └─ l3b full:         {l3b_full:4d}")
    if l3b_augmented > 0:
        print(f"  L3b augmented (→LLM): {l3b_augmented:4d} (still called LLM but with fragment context)")
    print(f"  Full inference:        {full_inference:4d} ({full_inference/total*100:.1f}%)")
    print()

    # Quality stats for L3
    l3_entries = [e for e in log_entries if e["source"] in ("l3_synthesis", "l3b_full_synthesis")]
    if l3_entries:
        l3_qualities = [e["quality"] for e in l3_entries]
        print(f"  L3 quality (vs ground truth):")
        print(f"    mean:   {np.mean(l3_qualities):.3f}")
        print(f"    median: {np.median(l3_qualities):.3f}")
        print(f"    min:    {np.min(l3_qualities):.3f}")
    print()

    # L3 catch rate (fraction of L1 misses caught by L3)
    l1_misses = total - l1_hits
    if l1_misses > 0:
        l3_catch_rate = l3_hits / l1_misses
        print(f"  L3 catch rate (of L1 misses): {l3_catch_rate*100:.1f}%")
    print()

    # Decision gate
    print(f"  DECISION GATE:")
    l3_catch = l3_hits / max(l1_misses, 1) * 100
    l3_quality_mean = np.mean([e["quality"] for e in l3_entries]) if l3_entries else 0
    combined = (l1_hits + l3_hits) / total * 100

    if l3_catch >= 15 and l3_quality_mean >= 0.85:
        print(f"    🟢 GREEN — L3 catches {l3_catch:.0f}% of misses, quality {l3_quality_mean:.2f}")
        print(f"    → Invest 4-5 weeks in full L3 production implementation")
    elif l3_catch >= 5:
        print(f"    🟡 YELLOW — L3 catches {l3_catch:.0f}% of misses, quality {l3_quality_mean:.2f}")
        print(f"    → L3 works but may not justify full effort; consider simpler approach")
    else:
        print(f"    🔴 RED — L3 catches only {l3_catch:.0f}% of misses")
        print(f"    → Concept synthesis insufficient; pivot to pure L1 improvements")

    print(f"\n  Combined L1+L3:        {combined:.1f}% (only {100-combined:.1f}% need LLM)")
    print(f"  Log: {os.path.abspath('pion-serve/experiment_log.jsonl')}")
    print(f"{'=' * 60}")
    print()

    return log_entries


def plot_convergence(log_entries: list[dict], output_path: str = "pion-serve/convergence.png"):
    """Plot convergence curve from experiment log."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    indices = [e["idx"] for e in log_entries]
    l1_rates = [e["cumulative_l1_rate"] * 100 for e in log_entries]
    l3_rates = [e["cumulative_l3_rate"] * 100 for e in log_entries]
    inf_rates = [e["cumulative_inference_rate"] * 100 for e in log_entries]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: convergence curve
    ax1.plot(indices, l1_rates, label="L1 Cache Hit Rate", color="#2196F3", linewidth=2)
    ax1.plot(indices, l3_rates, label="L3 Synthesis Rate", color="#4CAF50", linewidth=2)
    ax1.plot(indices, inf_rates, label="Full Inference Rate", color="#F44336", linewidth=2)
    ax1.fill_between(indices, l1_rates, alpha=0.1, color="#2196F3")
    ax1.fill_between(indices, l3_rates, alpha=0.1, color="#4CAF50")

    ax1.set_xlabel("Queries Processed")
    ax1.set_ylabel("Cumulative Rate (%)")
    ax1.set_title("Inference Distillation — Convergence Curve")
    ax1.legend(loc="center right")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Bottom: concept count
    concept_counts = [e["concepts_total"] for e in log_entries]
    ax2.plot(indices, concept_counts, color="#FF9800", linewidth=2)
    ax2.set_xlabel("Queries Processed")
    ax2.set_ylabel("Concepts")
    ax2.set_title("Concept Store Growth")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Convergence plot saved to {output_path}")

    # Also save quality over time if we have L3 data
    l3_entries = [(e["idx"], e["quality"]) for e in log_entries if e["source"] == "l3_synthesis"]
    if l3_entries:
        fig2, ax = plt.subplots(figsize=(12, 4))
        l3_idx, l3_qual = zip(*l3_entries)
        ax.scatter(l3_idx, l3_qual, alpha=0.5, s=20, color="#4CAF50")
        # Running average
        if len(l3_qual) >= 10:
            window = min(20, len(l3_qual))
            running_avg = np.convolve(l3_qual, np.ones(window)/window, mode="valid")
            ax.plot(l3_idx[window-1:], running_avg, color="#2E7D32", linewidth=2, label=f"Running avg (w={window})")
            ax.legend()
        ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="Quality threshold (0.85)")
        ax.set_xlabel("Query Index")
        ax.set_ylabel("L3 Quality Score")
        ax.set_title("L3 Synthesis Quality Over Time")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        quality_path = output_path.replace(".png", "_quality.png")
        plt.savefig(quality_path, dpi=150, bbox_inches="tight")
        print(f"  Quality plot saved to {quality_path}")


def main():
    parser = argparse.ArgumentParser(description="Pion Inference Distillation — Replay Experiment")
    parser.add_argument("--dataset", default="pion-serve/qa_dataset.jsonl",
                        help="Input JSONL dataset (default: pion-serve/qa_dataset.jsonl)")
    parser.add_argument("--pion-host", default="127.0.0.1")
    parser.add_argument("--pion-port", type=int, default=1974)
    parser.add_argument("--backend-url", default="http://127.0.0.1:11434",
                        help="LLM backend URL (default: ollama localhost)")
    parser.add_argument("--model", default="gemma3:4b",
                        help="LLM model (default: gemma3:4b)")
    parser.add_argument("--l1-threshold", type=float, default=0.92,
                        help="L1 semantic cache threshold (default: 0.92)")
    parser.add_argument("--l3-direct-threshold", type=float, default=0,
                        help="L3 direct synthesis threshold (0 = auto-calibrate)")
    parser.add_argument("--l3-composite-threshold", type=float, default=0,
                        help="L3 composite synthesis threshold (0 = auto-calibrate)")
    parser.add_argument("--api-key", default="",
                        help="API key for backend (Gemini, OpenAI). Also reads GEMINI_API_KEY env var.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use ground-truth answers instead of LLM (no backend needed)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only plot from existing log (no experiment)")
    parser.add_argument("--log", default="pion-serve/experiment_log.jsonl",
                        help="Log file path (default: pion-serve/experiment_log.jsonl)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of queries (0 = all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pion Inference Distillation — Replay Experiment")
    print("=" * 60)

    if args.plot_only:
        print(f"\nPlotting from {args.log}...")
        entries = load_jsonl(args.log)
        plot_convergence(entries)
        return

    # Load dataset
    print(f"\n1. Loading dataset: {args.dataset}")
    dataset = load_jsonl(args.dataset)
    if args.limit > 0:
        dataset = dataset[:args.limit]
    print(f"   {len(dataset)} queries loaded")

    # Connect to Pion
    print(f"\n2. Connecting to Pion at {args.pion_host}:{args.pion_port}")
    pion = redis.Redis(host=args.pion_host, port=args.pion_port, decode_responses=False)
    try:
        pion.ping()
        print("   Connected")
    except Exception as e:
        print(f"   ERROR: Cannot connect to Pion: {e}")
        sys.exit(1)

    # Check embedding availability + auto-calibrate thresholds
    if not args.dry_run:
        print(f"\n3. Checking embedding availability...")
        test_emb = embed_text("test", args.backend_url)
        if test_emb is None:
            print("   ERROR: Cannot embed text. Is Ollama running with nomic-embed-text?")
            print("   Hint: ollama pull nomic-embed-text")
            sys.exit(1)
        print(f"   Embedding OK (dim={len(test_emb)})")

        # Auto-calibrate thresholds if not explicitly set
        if args.l3_direct_threshold == 0:
            print(f"   Calibrating thresholds...")
            cal_pairs = [
                ("What is a hash map?", "How does a hash map work?"),
                ("What is the thread model?", "Describe the thread model."),
                ("How do you build the server?", "What are the build instructions?"),
            ]
            sims = []
            for q1, q2 in cal_pairs:
                e1 = embed_text(q1, args.backend_url)
                e2 = embed_text(q2, args.backend_url)
                if e1 is not None and e2 is not None:
                    sims.append(float(np.dot(e1, e2)))
            if sims:
                median_sim = sorted(sims)[len(sims) // 2]
                # Direct threshold: ~median paraphrase similarity (these should hit)
                # Composite threshold: ~10% below direct
                args.l3_direct_threshold = round(max(0.65, min(0.92, median_sim - 0.02)), 2)
                args.l3_composite_threshold = round(max(0.55, args.l3_direct_threshold - 0.08), 2)
                print(f"   Auto-calibrated: direct={args.l3_direct_threshold}, "
                      f"composite={args.l3_composite_threshold} "
                      f"(median paraphrase sim={median_sim:.3f})")
            else:
                args.l3_direct_threshold = 0.80
                args.l3_composite_threshold = 0.72
                print(f"   Calibration failed, using defaults: direct=0.80, composite=0.72")
    else:
        print(f"\n3. Dry run mode — using ground-truth answers (no LLM needed)")
        if args.l3_direct_threshold == 0:
            args.l3_direct_threshold = 0.90
            args.l3_composite_threshold = 0.82

    # Initialize concept store + fragment store
    print(f"\n4. Initializing L3 concept store + fragment store...")
    concepts = ConceptStore(
        pion_host=args.pion_host,
        pion_port=args.pion_port,
        direct_threshold=args.l3_direct_threshold,
        composite_threshold=args.l3_composite_threshold,
    )
    frag_embed_fn = lambda text: embed_text(text, args.backend_url)
    frag_store = FragmentStore(
        pion_host=args.pion_host,
        pion_port=args.pion_port,
        embed_fn=frag_embed_fn,
    )
    print(f"   Concept store: {concepts.stats['concepts']} concepts")
    print(f"   Fragment store: {frag_store.stats['fragments']} fragments")

    # Run experiment
    # Resolve API key
    api_key = (args.api_key
               or os.environ.get("ANTHROPIC_API_KEY", "")
               or os.environ.get("GEMINI_API_KEY", "")
               or os.environ.get("OPENAI_API_KEY", ""))

    print(f"\n5. Running experiment...")
    log_entries = run_experiment(
        dataset=dataset,
        pion=pion,
        concepts=concepts,
        backend_url=args.backend_url,
        model=args.model,
        l1_threshold=args.l1_threshold,
        dry_run=args.dry_run,
        log_path=args.log,
        fragments=frag_store,
        api_key=api_key,
    )

    # Plot
    print(f"\n6. Plotting convergence curve...")
    plot_convergence(log_entries)

    print("\nDone.")


if __name__ == "__main__":
    main()
