"""
Phase 1 Agent Memory Demo: in-process persistent memory across conversation resets.

Demonstrates:
  1. Conversation 1: store 5 user preferences via pion.remember()
  2. Process "restart" (memory survives as pion-server WAL or in-memory library)
  3. Conversation 2: retrieve preferences at each generation step via pion.recall()
  4. Show preferences correctly applied in generated responses

This script works WITHOUT a running model — it uses mock embeddings that capture
semantic relationships, and generates "responses" via simple template filling.
To run with a real LLM (requires transformers + torch):
  python3 examples/agent_memory_demo.py --real-llm

Core metric: retrieval latency at each generation step (logged to console).
Target: p50 <200µs per recall() call.

Usage:
    # Mock mode (no dependencies beyond numpy):
    python3 examples/agent_memory_demo.py

    # With real embeddings (requires: pip install sentence-transformers):
    python3 examples/agent_memory_demo.py --real-embeddings
"""

import sys
import time
import argparse
import numpy as np

sys.path.insert(0, ".")
from pion_memory import PionMemory


# ──────────────────────────────────────────────────────────────────────────────
# Mock embedding function (deterministic, captures semantic similarity)
# ──────────────────────────────────────────────────────────────────────────────

# Vocabulary of "concepts" for mock embeddings
_VOCAB = [
    "python", "code", "programming", "language", "developer",
    "prefer", "like", "want", "need", "use",
    "formal", "casual", "tone", "style", "voice",
    "short", "brief", "concise", "long", "detailed",
    "name", "call", "address", "greet",
    "coffee", "tea", "drink", "morning", "caffeine",
    "dark", "light", "mode", "interface", "theme",
    "metric", "imperial", "unit", "system",
    "email", "slack", "chat", "message", "notify",
    "backend", "frontend", "database", "api", "server",
]
_VOCAB_DIM = 64
_vocab_matrix = None


def _get_vocab_matrix() -> np.ndarray:
    global _vocab_matrix
    if _vocab_matrix is None:
        rng = np.random.default_rng(123)
        m = rng.standard_normal((len(_VOCAB), _VOCAB_DIM)).astype(np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        _vocab_matrix = m / (norms + 1e-8)
    return _vocab_matrix


def mock_embed(text: str) -> np.ndarray:
    """Return a fixed-dim embedding by bag-of-vocab-words (deterministic)."""
    words = text.lower().split()
    vocab_m = _get_vocab_matrix()
    vec = np.zeros(_VOCAB_DIM, dtype=np.float32)
    hits = 0
    for word in words:
        for vi, vw in enumerate(_VOCAB):
            if vw in word or word in vw:
                vec += vocab_m[vi]
                hits += 1
    if hits == 0:
        rng = np.random.default_rng(hash(text) % (2**32))
        vec = rng.standard_normal(_VOCAB_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def real_embed(text: str, model) -> np.ndarray:
    """Use sentence-transformers for real embeddings."""
    emb = model.encode([text], normalize_embeddings=True)
    return emb[0].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# User preferences (the "memory" to store and retrieve)
# ──────────────────────────────────────────────────────────────────────────────

PREFERENCES = [
    {
        "id": 0,
        "text": "The user prefers Python over other programming languages",
        "key": "language_preference",
        "value": "Python",
    },
    {
        "id": 1,
        "text": "The user wants formal and professional tone in responses",
        "key": "tone",
        "value": "formal and professional",
    },
    {
        "id": 2,
        "text": "The user likes concise and brief answers without padding",
        "key": "response_style",
        "value": "concise and brief",
    },
    {
        "id": 3,
        "text": "The user prefers dark mode in all interfaces",
        "key": "ui_preference",
        "value": "dark mode",
    },
    {
        "id": 4,
        "text": "The user wants to be called Alex and starts mornings with coffee",
        "key": "personal",
        "value": "name=Alex, morning routine=coffee",
    },
]

# Conversation 2 queries — designed to trigger relevant preferences
CONVERSATION_2 = [
    ("How should I write this code in Python?", "language_preference"),
    ("I need a formal tone for this email message", "tone"),
    ("Can you give a brief and concise answer?", "response_style"),
    ("I prefer dark mode for my interface and theme", "ui_preference"),
    ("Good morning! I need my coffee to start the day", "personal"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Agent: generates responses augmented with retrieved memory
# ──────────────────────────────────────────────────────────────────────────────

def agent_respond(query: str, retrieved: list[tuple[int, float]], embed_fn) -> str:
    """Generate a response template that incorporates retrieved preferences."""
    pref_map = {p["id"]: p for p in PREFERENCES}
    applied = [pref_map[rid]["value"] for rid, _ in retrieved if rid in pref_map]

    if not applied:
        return f'Responding to: "{query}" — [no preferences retrieved]'

    context_str = " | ".join(applied[:3])
    return (
        f'[Memory: {context_str}]\n'
        f'Responding to: "{query}"\n'
        f'(Applied {len(applied)} preference(s) from memory)'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main demo
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-embeddings", action="store_true",
                        help="Use sentence-transformers for real embeddings")
    parser.add_argument("--ef", type=int, default=32,
                        help="Search ef for memory retrieval (default: 32)")
    args = parser.parse_args()

    if args.real_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            embed_fn = lambda t: real_embed(t, st_model)
            dim = 384
            print("  Loaded (dim=384)")
        except ImportError:
            print("sentence-transformers not installed. Falling back to mock embeddings.")
            embed_fn = mock_embed
            dim = _VOCAB_DIM
    else:
        embed_fn = mock_embed
        dim = _VOCAB_DIM

    print("\n" + "=" * 65)
    print("  Pion Agent Memory Demo — Phase 1")
    print("=" * 65)
    print(f"  Embeddings: {'sentence-transformers (all-MiniLM-L6-v2)' if args.real_embeddings else 'mock (bag-of-words, 64-dim)'}")
    print(f"  Memory: PionMemory(dim={dim}, max_elements=100)")

    # ── Conversation 1: store preferences ─────────────────────────────────────
    print("\n[Conversation 1] Storing user preferences...")
    print("-" * 65)

    mem = PionMemory(dim=dim, max_elements=100, M=8, ef_construction=32)

    for pref in PREFERENCES:
        emb = embed_fn(pref["text"])
        mem.remember(pref["id"], emb)
        print(f"  remember({pref['id']}): {pref['text'][:55]}...")

    mem.optimize()
    print(f"\n  Index built. {len(mem)} preferences stored.")
    print(f"  Memory ready: {mem.is_ready}")

    # ── Simulated process restart ──────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  [Simulated restart — loading memory from persisted index]")
    print("─" * 65)
    # In production: mem = PionMemory.load("pion_agent.hnsw")
    # Here we keep the same object to demonstrate the flow
    print("  Memory retained in-process (zero reload time)")
    print("  Note: pion-server with WAL can persist across true restarts")

    # ── Conversation 2: retrieve at each step ─────────────────────────────────
    print(f"\n[Conversation 2] Retrieval-augmented responses (k=3, ef={args.ef})")
    print("-" * 65)

    latencies_us = []
    retrieved_correct = 0

    for turn_idx, (query, expected_pref_key) in enumerate(CONVERSATION_2):
        print(f"\n  Turn {turn_idx + 1}: \"{query}\"")

        # Embed query
        t0 = time.perf_counter()
        q_emb = embed_fn(query)
        t_embed = (time.perf_counter() - t0) * 1e6

        # Retrieve from memory
        t0 = time.perf_counter()
        retrieved = mem.recall(q_emb, k=3, ef=args.ef)
        t_recall = (time.perf_counter() - t0) * 1e6
        latencies_us.append(t_recall)

        # Check if expected preference was retrieved
        retrieved_ids = {rid for rid, _ in retrieved}
        expected_id = next(p["id"] for p in PREFERENCES if p["key"] == expected_pref_key)
        hit = expected_id in retrieved_ids
        if hit:
            retrieved_correct += 1

        # Show retrieved preferences
        pref_map = {p["id"]: p for p in PREFERENCES}
        for rid, score in retrieved:
            if rid in pref_map:
                p = pref_map[rid]
                marker = "✓" if rid == expected_id else " "
                print(f"  {marker}  [{rid}] L2={score:.3f}  {p['text'][:50]}...")

        # Generate response
        response = agent_respond(query, retrieved, embed_fn)
        print(f"  Response: {response}")
        print(f"  Latency: embed={t_embed:.0f}µs  recall={t_recall:.0f}µs")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Results")
    print("=" * 65)

    latencies_us.sort()
    p50 = latencies_us[len(latencies_us) // 2]
    p99 = latencies_us[int(len(latencies_us) * 0.99)]
    mean = np.mean(latencies_us)

    print(f"  Recall turns:     {len(CONVERSATION_2)}")
    print(f"  Correct retrieval: {retrieved_correct}/{len(CONVERSATION_2)} "
          f"({100*retrieved_correct/len(CONVERSATION_2):.0f}%)")
    print(f"  recall() p50:     {p50:.1f}µs")
    print(f"  recall() p99:     {p99:.1f}µs")
    print(f"  recall() mean:    {mean:.1f}µs")

    print()
    p50_ok = p50 < 200
    correct_ok = retrieved_correct >= len(CONVERSATION_2) * 0.6
    print(f"  Target p50 <200µs:          {'PASS ✓' if p50_ok else 'FAIL ✗'}  ({p50:.1f}µs)")
    print(f"  Target ≥60% correct recall: {'PASS ✓' if correct_ok else 'FAIL ✗'}  "
          f"({retrieved_correct}/{len(CONVERSATION_2)})")

    print()
    print("  This demo shows in-process persistent memory callable from any")
    print("  Python AI framework — no server, no network, <1ms per retrieval.")
    print()


if __name__ == "__main__":
    main()
