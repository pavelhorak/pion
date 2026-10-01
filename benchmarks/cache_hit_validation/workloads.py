"""Synthetic coding workload generator for cache hit rate validation.

Generates four workload categories:
  A. Multi-turn conversations (same code context, rephrased questions)
  B. Inline completions (same file with minor edits)
  C. Cross-user queries (different developers, same code)
  D. Unrelated queries (negative control for false positive measurement)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from .code_snippets import CodeSnippet, get_snippet_pool


@dataclass
class CacheEntry:
    prompt: str
    category: str
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryEntry:
    prompt: str
    category: str
    expected_hit: bool
    source_cache_idx: int  # index into cache_entries list, -1 if no match expected
    metadata: dict = field(default_factory=dict)


# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are an expert coding assistant. Be concise and precise."

TURN1_QUESTIONS = [
    "Explain what this code does.",
    "Walk me through this code step by step.",
    "What is the purpose of this code?",
    "Describe the functionality of this code.",
    "How does this code work?",
    "Give me a high-level overview of this code.",
    "Summarize what this function does.",
    "What does this implementation do?",
]

TURN2_QUESTIONS = [
    "How would I add error handling to this?",
    "Can you refactor this for better readability?",
    "What are the potential bugs in this code?",
    "How would you add unit tests for this?",
    "What edge cases should I handle?",
    "Can you optimize this for performance?",
    "How would I add logging to this?",
    "What would you change to make this production-ready?",
]

TURN3_QUESTIONS = [
    "Now write the actual implementation of that change.",
    "Show me the complete refactored version.",
    "Write the test file for this.",
    "Add docstrings to all the functions.",
    "Convert this to use async/await.",
    "Add type annotations throughout.",
]

CROSS_USER_PREFIXES = [
    "I'm working on the Pion project. Here's some code I need help with:",
    "Looking at this code from our codebase:",
    "Can you help me understand this module?",
    "I'm reviewing this code for a PR:",
    "I need to modify this code. Here it is:",
    "Our team wrote this — I need to extend it:",
    "I'm debugging an issue in this code:",
    "I'm new to this codebase. Here's a file I need to understand:",
]

CROSS_USER_SUFFIXES = [
    "How does the main logic work?",
    "Explain the core algorithm here.",
    "What's the overall architecture of this code?",
    "Where are the key data structures defined?",
    "What are the important functions?",
    "How is error handling done here?",
    "What dependencies does this have?",
    "How would I extend this?",
]

UNRELATED_QUERIES = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a recipe for chocolate chip cookies.",
    "What were the main causes of World War I?",
    "How does photosynthesis work?",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about programming.",
    "What is the time complexity of quicksort?",
    "How do I train a neural network from scratch?",
    "Explain the CAP theorem in distributed systems.",
    "Write a React component for a login form.",
    "How do I set up a Kubernetes cluster?",
    "What is the difference between REST and GraphQL?",
    "Explain how garbage collection works in Java.",
    "Write a SQL query to find duplicate records.",
    "How does the Bitcoin blockchain work?",
    "What is the difference between Docker and VMs?",
    "Explain CRISPR gene editing technology.",
    "Write a bash script to monitor disk usage.",
    "How does HTTP/3 differ from HTTP/2?",
    "Write a shopping cart implementation in TypeScript.",
    "Explain how transformers work in machine learning.",
    "What are the SOLID principles in OOP?",
    "How does a B-tree index work in databases?",
    "Write a Python script to scrape a website.",
    "Explain the Raft consensus algorithm.",
    "How do I implement OAuth 2.0?",
    "What is the difference between threads and processes?",
    "Write a Rust function to parse JSON.",
    "How does memory management work in Swift?",
]


def _make_code_block(snippet: CodeSnippet) -> str:
    return f"```{snippet.language}\n# {snippet.file_path}:{snippet.start_line}-{snippet.end_line}\n{snippet.content}\n```"


def _make_prompt(system: str, context: str, question: str) -> str:
    return f"System: {system}\n\nContext:\n{context}\n\nUser: {question}"


# ── Mutation engine for inline completions ───────────────────────────────────

def _mutate_snippet(content: str, rng: random.Random, num_edits: int = 2) -> str:
    """Apply small deterministic edits to simulate typing."""
    lines = content.split("\n")
    mutations_applied = 0

    for _ in range(num_edits * 3):  # try up to 3x to find mutable lines
        if mutations_applied >= num_edits or not lines:
            break
        idx = rng.randint(0, len(lines) - 1)
        line = lines[idx]
        if len(line.strip()) < 5:
            continue

        mutation = rng.choice(["var_rename", "add_comment", "change_literal", "add_blank"])

        if mutation == "var_rename":
            # Rename a short identifier
            words = re.findall(r'\b[a-z_][a-z0-9_]{2,8}\b', line)
            if words:
                old = rng.choice(words)
                new = old[:-1] + rng.choice("_xyz")
                lines[idx] = line.replace(old, new, 1)
                mutations_applied += 1

        elif mutation == "add_comment":
            indent = len(line) - len(line.lstrip())
            comment_char = "#" if not any(line.strip().startswith(k) for k in ["fn ", "struct ", "var "]) else "#"
            lines.insert(idx, " " * indent + f"{comment_char} TODO: review this")
            mutations_applied += 1

        elif mutation == "change_literal":
            # Change a number or short string
            if re.search(r'\b\d+\b', line):
                lines[idx] = re.sub(r'\b(\d+)\b', lambda m: str(int(m.group()) + rng.randint(1, 5)), line, count=1)
                mutations_applied += 1
            elif '"' in line:
                lines[idx] = line.replace('"', '"', 1)  # no-op but counts for diversity
                mutations_applied += 1

        elif mutation == "add_blank":
            lines.insert(idx + 1, "")
            mutations_applied += 1

    return "\n".join(lines)


# ── Workload generators ──────────────────────────────────────────────────────

def _gen_multi_turn(snippets: list[CodeSnippet], rng: random.Random) -> tuple[list[CacheEntry], list[QueryEntry]]:
    """Category A: Multi-turn conversations."""
    cache = []
    queries = []

    for i, snippet in enumerate(snippets[:12]):
        code_block = _make_code_block(snippet)

        # Cache entry: turn 1 with first question variant
        q1 = TURN1_QUESTIONS[i % len(TURN1_QUESTIONS)]
        prompt1 = _make_prompt(SYSTEM_PROMPT, code_block, q1)
        cache_idx = len(cache)
        cache.append(CacheEntry(prompt1, "multi_turn", {"file": snippet.file_path, "turn": 1}))

        # Query: rephrased turn 1 (semantic match, NOT exact match)
        alt_q1 = TURN1_QUESTIONS[(i + 3) % len(TURN1_QUESTIONS)]
        queries.append(QueryEntry(
            _make_prompt(SYSTEM_PROMPT, code_block, alt_q1),
            "multi_turn", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "rephrased_turn1", "file": snippet.file_path},
        ))

        # Query: turn 2 (shares code context, different question)
        q2 = TURN2_QUESTIONS[i % len(TURN2_QUESTIONS)]
        turn2_prompt = _make_prompt(SYSTEM_PROMPT, code_block, f"Regarding the code above: {q2}")
        queries.append(QueryEntry(
            turn2_prompt,
            "multi_turn", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "turn2", "file": snippet.file_path},
        ))

        # Query: turn 3 (shares code context, follow-up)
        q3 = TURN3_QUESTIONS[i % len(TURN3_QUESTIONS)]
        turn3_prompt = _make_prompt(SYSTEM_PROMPT, code_block, f"Following up on the code: {q3}")
        queries.append(QueryEntry(
            turn3_prompt,
            "multi_turn", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "turn3", "file": snippet.file_path},
        ))

        # Query: slightly different system prompt (still same code)
        alt_system = "You are a senior software engineer doing code review. Be thorough."
        queries.append(QueryEntry(
            _make_prompt(alt_system, code_block, q1),
            "multi_turn", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "alt_system_prompt", "file": snippet.file_path},
        ))

    return cache, queries


def _gen_inline_completions(snippets: list[CodeSnippet], rng: random.Random) -> tuple[list[CacheEntry], list[QueryEntry]]:
    """Category B: Inline completions with minor file edits."""
    cache = []
    queries = []

    for i, snippet in enumerate(snippets[:10]):
        # Cache entry: completion request with original file
        original_prompt = f"Complete the following {snippet.language} code:\n{_make_code_block(snippet)}\n\n# Continue from line {snippet.end_line}:"
        cache_idx = len(cache)
        cache.append(CacheEntry(original_prompt, "inline_completion", {"file": snippet.file_path}))

        # Query: file with 1 small edit (should be semantic hit)
        mutated1 = CodeSnippet(snippet.file_path, snippet.language,
                               _mutate_snippet(snippet.content, rng, num_edits=1),
                               snippet.start_line, snippet.end_line)
        queries.append(QueryEntry(
            f"Complete the following {snippet.language} code:\n{_make_code_block(mutated1)}\n\n# Continue from line {snippet.end_line}:",
            "inline_completion", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "1_edit", "file": snippet.file_path},
        ))

        # Query: file with 2 edits
        mutated2 = CodeSnippet(snippet.file_path, snippet.language,
                               _mutate_snippet(snippet.content, rng, num_edits=2),
                               snippet.start_line, snippet.end_line)
        queries.append(QueryEntry(
            f"Complete the following {snippet.language} code:\n{_make_code_block(mutated2)}\n\n# Continue from line {snippet.end_line}:",
            "inline_completion", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "2_edits", "file": snippet.file_path},
        ))

        # Query: file with 3 edits (might be borderline at high thresholds)
        mutated3 = CodeSnippet(snippet.file_path, snippet.language,
                               _mutate_snippet(snippet.content, rng, num_edits=3),
                               snippet.start_line, snippet.end_line)
        queries.append(QueryEntry(
            f"Complete the following {snippet.language} code:\n{_make_code_block(mutated3)}\n\n# Continue from line {snippet.end_line}:",
            "inline_completion", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "3_edits", "file": snippet.file_path},
        ))

        # Query: different cursor position (same file, different wrapping)
        mid_line = (snippet.start_line + snippet.end_line) // 2
        queries.append(QueryEntry(
            f"Complete the {snippet.language} code at line {mid_line}:\n{_make_code_block(snippet)}\n\n# Insert at line {mid_line}:",
            "inline_completion", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "cursor_shift", "file": snippet.file_path},
        ))

    return cache, queries


def _gen_cross_user(snippets: list[CodeSnippet], rng: random.Random) -> tuple[list[CacheEntry], list[QueryEntry]]:
    """Category C: Different developers asking about the same code."""
    cache = []
    queries = []

    for i, snippet in enumerate(snippets[:10]):
        code_block = _make_code_block(snippet)
        prefix_a = CROSS_USER_PREFIXES[i % len(CROSS_USER_PREFIXES)]
        suffix_a = CROSS_USER_SUFFIXES[i % len(CROSS_USER_SUFFIXES)]
        prompt_a = f"{prefix_a}\n\n{code_block}\n\n{suffix_a}"
        cache_idx = len(cache)
        cache.append(CacheEntry(prompt_a, "cross_user", {"file": snippet.file_path, "user": "A"}))

        # Query: developer B asks about same code, different framing
        prefix_b = CROSS_USER_PREFIXES[(i + 4) % len(CROSS_USER_PREFIXES)]
        suffix_b = CROSS_USER_SUFFIXES[(i + 3) % len(CROSS_USER_SUFFIXES)]
        prompt_b = f"{prefix_b}\n\n{code_block}\n\n{suffix_b}"
        queries.append(QueryEntry(
            prompt_b, "cross_user", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "diff_framing", "file": snippet.file_path, "user": "B"},
        ))

        # Query: developer C, more different framing
        prefix_c = CROSS_USER_PREFIXES[(i + 6) % len(CROSS_USER_PREFIXES)]
        suffix_c = CROSS_USER_SUFFIXES[(i + 5) % len(CROSS_USER_SUFFIXES)]
        prompt_c = f"{prefix_c}\n\n{code_block}\n\n{suffix_c}"
        queries.append(QueryEntry(
            prompt_c, "cross_user", expected_hit=True, source_cache_idx=cache_idx,
            metadata={"subtype": "diff_framing_2", "file": snippet.file_path, "user": "C"},
        ))

    return cache, queries


def _gen_unrelated(cache_size: int, rng: random.Random) -> tuple[list[CacheEntry], list[QueryEntry]]:
    """Category D: Unrelated queries (negative control)."""
    queries = []
    # These should NOT match any of the code-related cache entries
    for i, text in enumerate(UNRELATED_QUERIES):
        queries.append(QueryEntry(
            text, "unrelated", expected_hit=False, source_cache_idx=-1,
            metadata={"subtype": "negative_control"},
        ))
    return [], queries


# ── Main generator ───────────────────────────────────────────────────────────

def generate_workload(seed: int = 42) -> tuple[list[CacheEntry], list[QueryEntry]]:
    """Generate full synthetic workload.

    Returns (cache_entries, queries) with ground truth labels.
    """
    rng = random.Random(seed)
    snippets = get_snippet_pool(n=20)

    if len(snippets) < 10:
        raise RuntimeError(
            f"Only found {len(snippets)} code snippets. Need at least 10. "
            "Check that the Pion source files exist."
        )

    all_cache: list[CacheEntry] = []
    all_queries: list[QueryEntry] = []

    # A: Multi-turn conversations
    c, q = _gen_multi_turn(snippets, rng)
    # Offset source_cache_idx by current cache size
    base = len(all_cache)
    for entry in q:
        if entry.source_cache_idx >= 0:
            entry.source_cache_idx += base
    all_cache.extend(c)
    all_queries.extend(q)

    # B: Inline completions
    c, q = _gen_inline_completions(snippets, rng)
    base = len(all_cache)
    for entry in q:
        if entry.source_cache_idx >= 0:
            entry.source_cache_idx += base
    all_cache.extend(c)
    all_queries.extend(q)

    # C: Cross-user queries
    c, q = _gen_cross_user(snippets, rng)
    base = len(all_cache)
    for entry in q:
        if entry.source_cache_idx >= 0:
            entry.source_cache_idx += base
    all_cache.extend(c)
    all_queries.extend(q)

    # D: Unrelated (no cache entries, just queries)
    _, q = _gen_unrelated(len(all_cache), rng)
    all_queries.extend(q)

    # Shuffle queries for realistic interleaving (but keep seed deterministic)
    rng.shuffle(all_queries)

    return all_cache, all_queries
