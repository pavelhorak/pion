#!/usr/bin/env python3
"""Generate 500 Q&A pairs about Python using Claude Haiku.

Generates in batches of 25 (20 API calls), with paraphrase clusters.
Uses claude-haiku-4-5-20251001 for speed and cost efficiency.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python3 pion-serve/generate_qa_claude.py --output pion-serve/qa_dataset_claude.jsonl

Cost estimate: ~500 pairs × ~200 tokens/pair = ~100K tokens ≈ $0.10 with Haiku.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import anthropic


TOPICS = [
    # Core types (8)
    "Python lists (append, extend, slicing, sorting)",
    "Python dictionaries (get, items, comprehensions, defaultdict)",
    "Python sets (add, intersection, union, difference)",
    "Python tuples (immutability, unpacking, named tuples)",
    "Python strings (f-strings, split, join, formatting)",
    "Python numbers (int, float, decimal, complex)",
    "Python None and bool (truthy, falsy, is vs ==)",
    "Python type conversion (int(), str(), list(), casting)",
    # Control flow (6)
    "Python if/else and ternary expressions",
    "Python for loops and iteration (enumerate, zip, range)",
    "Python while loops and break/continue",
    "Python match/case (structural pattern matching)",
    "Python try/except/finally (exception handling)",
    "Python with statement (context managers)",
    # Functions (8)
    "Python function definitions (def, return, docstrings)",
    "Python *args and **kwargs",
    "Python lambda functions",
    "Python closures and nonlocal",
    "Python decorators (@property, @staticmethod, custom)",
    "Python generators and yield",
    "Python recursion and stack limits",
    "Python function annotations and type hints",
    # Classes (8)
    "Python __init__ and constructors",
    "Python inheritance and super()",
    "Python dunder/magic methods (__str__, __repr__, __eq__)",
    "Python dataclasses (@dataclass)",
    "Python abstract base classes (abc, ABC)",
    "Python __slots__ and memory optimization",
    "Python @property and descriptors",
    "Python class methods vs static methods",
    # Stdlib (12)
    "Python os and sys modules",
    "Python pathlib (Path, file operations)",
    "Python json module (loads, dumps, custom encoders)",
    "Python re module (regex, match, search, findall)",
    "Python datetime (date, time, timedelta, strftime)",
    "Python collections (Counter, deque, OrderedDict)",
    "Python itertools (chain, product, combinations)",
    "Python functools (partial, lru_cache, reduce)",
    "Python logging module (levels, handlers, formatters)",
    "Python typing module (Optional, Union, TypeVar, Generic)",
    "Python enum module (Enum, IntEnum, auto)",
    "Python copy module (shallow vs deep copy)",
    # Async (6)
    "Python asyncio basics (async def, await)",
    "Python asyncio event loop",
    "Python asyncio.gather and tasks",
    "Python asyncio queues and synchronization",
    "Python aiohttp (async HTTP client/server)",
    "Python async generators and async for",
    # Packaging (5)
    "Python pip and package installation",
    "Python virtual environments (venv, virtualenv)",
    "Python pyproject.toml and setup.cfg",
    "Python requirements.txt and dependency management",
    "Python wheels and distribution",
    # Testing (5)
    "Python pytest basics (test discovery, assertions)",
    "Python pytest fixtures and conftest",
    "Python pytest parametrize",
    "Python unittest and TestCase",
    "Python mocking (unittest.mock, patch, MagicMock)",
    # Web (4)
    "Python Flask (routes, templates, request handling)",
    "Python FastAPI (type hints, async, OpenAPI)",
    "Python requests library (GET, POST, sessions)",
    "Python HTTP and REST API concepts",
    # Data (4)
    "Python csv module (reader, writer, DictReader)",
    "Python sqlite3 (connect, execute, fetchall)",
    "Python struct module (pack, unpack, binary data)",
    "Python pickle (serialization, security risks)",
    # Concurrency (4)
    "Python threading (Thread, Lock, Event)",
    "Python multiprocessing (Process, Pool, shared memory)",
    "Python GIL (Global Interpreter Lock)",
    "Python concurrent.futures (ThreadPoolExecutor, ProcessPoolExecutor)",
    # Common errors (5)
    "Python ImportError and ModuleNotFoundError",
    "Python AttributeError",
    "Python TypeError",
    "Python KeyError and IndexError",
    "Python ValueError and RuntimeError",
    # Patterns (4)
    "Python context managers (__enter__, __exit__, contextlib)",
    "Python metaclasses (type, __new__, __init_subclass__)",
    "Python descriptor protocol (__get__, __set__)",
    "Python mixin classes and multiple inheritance",
    # Performance (4)
    "Python profiling (cProfile, timeit, line_profiler)",
    "Python caching (lru_cache, functools.cache)",
    "Python memory management and garbage collection",
    "Python performance tips (list vs generator, set lookup)",
    # File I/O (4)
    "Python file I/O (open, read, write, modes)",
    "Python binary file handling",
    "Python encoding and Unicode (utf-8, errors, BOM)",
    "Python tempfile and file-like objects",
    # Misc (5)
    "Python list/dict/set comprehensions",
    "Python walrus operator (:=)",
    "Python unpacking and starred expressions",
    "Python __name__ == '__main__'",
    "Python virtual environments vs conda",
    # Debugging (4)
    "Python pdb and breakpoint()",
    "Python traceback and stack traces",
    "Python logging for debugging",
    "Python common debugging strategies",
]


def generate_batch(client: anthropic.Anthropic, model: str, topics: list[str]) -> list[dict]:
    """Generate Q&A pairs for a batch of topics (5 questions each, with paraphrases)."""
    topic_list = "\n".join(f"- {t}" for t in topics)
    prompt = f"""Generate exactly {len(topics) * 5} question-answer pairs about these Python topics.

Topics:
{topic_list}

For EACH topic, generate exactly 5 questions:
- 3 paraphrased versions of the same core question (testing if a cache can match them)
- 2 different questions about the same topic

Rules:
- Answers: 2-4 sentences, factual, concise. Include a short code example where helpful (1-3 lines).
- Each question must be unique text.
- Output ONLY valid JSON array. No markdown, no commentary.

Output format (JSON array):
[
  {{"question": "What is a list in Python?", "answer": "A list is a mutable ordered collection..."}},
  {{"question": "How do Python lists work?", "answer": "Python lists store ordered sequences..."}},
  ...
]"""

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()

        # Extract JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            pairs = json.loads(text[start:end + 1])
            return [p for p in pairs if "question" in p and "answer" in p]
    except json.JSONDecodeError as e:
        print(f"\n  Warning: JSON parse error: {e}")
    except Exception as e:
        print(f"\n  Warning: API error: {e}")
    return []


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs using Claude API")
    parser.add_argument("--output", default="pion-serve/qa_dataset_claude.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Claude model (default: claude-haiku-4-5-20251001)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Topics per API call (default: 5 = 25 Q&A pairs per call)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("=" * 60)
    print("  Q&A Generator — Claude API")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Topics:   {len(TOPICS)}")
    print(f"  Expected: ~{len(TOPICS) * 5} Q&A pairs")
    print(f"  Batch:    {args.batch_size} topics/call = {len(TOPICS) // args.batch_size} API calls")
    print()

    all_pairs = []
    batches = [TOPICS[i:i + args.batch_size] for i in range(0, len(TOPICS), args.batch_size)]

    for bi, batch in enumerate(batches):
        sys.stdout.write(f"\r  Batch {bi + 1}/{len(batches)} ({len(all_pairs)} pairs so far)...")
        sys.stdout.flush()
        pairs = generate_batch(client, args.model, batch)
        all_pairs.extend(pairs)
        time.sleep(0.2)  # rate limit courtesy

    print(f"\r  Generated {len(all_pairs)} Q&A pairs from {len(batches)} API calls")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"  Output: {args.output}")
    print(f"  Total: {len(all_pairs)} pairs")
    print()


if __name__ == "__main__":
    main()
