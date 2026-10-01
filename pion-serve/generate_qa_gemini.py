#!/usr/bin/env python3
"""Generate 500 Q&A pairs about Python using Gemini API (OpenAI-compatible endpoint).

Usage:
    export GEMINI_API_KEY="..."
    python3 pion-serve/generate_qa_gemini.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests

TOPICS = [
    "Python lists (append, extend, slicing, sorting)",
    "Python dictionaries (get, items, comprehensions, defaultdict)",
    "Python sets (add, intersection, union, difference)",
    "Python tuples (immutability, unpacking, named tuples)",
    "Python strings (f-strings, split, join, formatting)",
    "Python numbers (int, float, decimal, complex)",
    "Python None and bool (truthy, falsy, is vs ==)",
    "Python type conversion (int(), str(), list(), casting)",
    "Python if/else and ternary expressions",
    "Python for loops and iteration (enumerate, zip, range)",
    "Python while loops and break/continue",
    "Python match/case (structural pattern matching)",
    "Python try/except/finally (exception handling)",
    "Python with statement (context managers)",
    "Python function definitions (def, return, docstrings)",
    "Python *args and **kwargs",
    "Python lambda functions",
    "Python closures and nonlocal",
    "Python decorators (@property, @staticmethod, custom)",
    "Python generators and yield",
    "Python recursion and stack limits",
    "Python function annotations and type hints",
    "Python __init__ and constructors",
    "Python inheritance and super()",
    "Python dunder/magic methods (__str__, __repr__, __eq__)",
    "Python dataclasses (@dataclass)",
    "Python abstract base classes (abc, ABC)",
    "Python __slots__ and memory optimization",
    "Python @property and descriptors",
    "Python class methods vs static methods",
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
    "Python asyncio basics (async def, await)",
    "Python asyncio event loop",
    "Python asyncio.gather and tasks",
    "Python asyncio queues and synchronization",
    "Python aiohttp (async HTTP client/server)",
    "Python async generators and async for",
    "Python pip and package installation",
    "Python virtual environments (venv, virtualenv)",
    "Python pyproject.toml and setup.cfg",
    "Python requirements.txt and dependency management",
    "Python wheels and distribution",
    "Python pytest basics (test discovery, assertions)",
    "Python pytest fixtures and conftest",
    "Python pytest parametrize",
    "Python unittest and TestCase",
    "Python mocking (unittest.mock, patch, MagicMock)",
    "Python Flask (routes, templates, request handling)",
    "Python FastAPI (type hints, async, OpenAPI)",
    "Python requests library (GET, POST, sessions)",
    "Python HTTP and REST API concepts",
    "Python csv module (reader, writer, DictReader)",
    "Python sqlite3 (connect, execute, fetchall)",
    "Python struct module (pack, unpack, binary data)",
    "Python pickle (serialization, security risks)",
    "Python threading (Thread, Lock, Event)",
    "Python multiprocessing (Process, Pool, shared memory)",
    "Python GIL (Global Interpreter Lock)",
    "Python concurrent.futures (ThreadPoolExecutor, ProcessPoolExecutor)",
    "Python ImportError and ModuleNotFoundError",
    "Python AttributeError",
    "Python TypeError",
    "Python KeyError and IndexError",
    "Python ValueError and RuntimeError",
    "Python context managers (__enter__, __exit__, contextlib)",
    "Python metaclasses (type, __new__, __init_subclass__)",
    "Python descriptor protocol (__get__, __set__)",
    "Python mixin classes and multiple inheritance",
    "Python profiling (cProfile, timeit, line_profiler)",
    "Python caching (lru_cache, functools.cache)",
    "Python memory management and garbage collection",
    "Python performance tips (list vs generator, set lookup)",
    "Python file I/O (open, read, write, modes)",
    "Python binary file handling",
    "Python encoding and Unicode (utf-8, errors, BOM)",
    "Python tempfile and file-like objects",
    "Python list/dict/set comprehensions",
    "Python walrus operator (:=)",
    "Python unpacking and starred expressions",
    "Python __name__ == '__main__'",
    "Python virtual environments vs conda",
    "Python pdb and breakpoint()",
    "Python traceback and stack traces",
    "Python logging for debugging",
    "Python common debugging strategies",
]


def generate_batch(api_key: str, topics: list[str]) -> list[dict]:
    topic_list = "\n".join(f"- {t}" for t in topics)
    prompt = f"""Generate exactly {len(topics) * 5} question-answer pairs about these Python topics.

Topics:
{topic_list}

For EACH topic, generate exactly 5 questions:
- 3 paraphrased versions of the same core question (testing if a cache can detect they ask the same thing)
- 2 different questions about the same topic

Rules:
- Answers: 2-4 sentences, factual, concise. Include a short code example where helpful.
- Each question must be unique text.
- Output ONLY a valid JSON array. No markdown fences, no commentary.

Format: [{{"question": "...", "answer": "..."}}, ...]"""

    try:
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]

        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

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
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    output = "pion-serve/qa_dataset_gemini.jsonl"
    batch_size = 5  # 5 topics × 5 questions = 25 pairs per call

    print("=" * 60)
    print("  Q&A Generator — Gemini 2.5 Flash")
    print("=" * 60)
    print(f"  Topics:   {len(TOPICS)}")
    print(f"  Expected: ~{len(TOPICS) * 5} Q&A pairs")
    print(f"  Batches:  {len(TOPICS) // batch_size} API calls")
    print()

    all_pairs = []
    batches = [TOPICS[i:i + batch_size] for i in range(0, len(TOPICS), batch_size)]

    for bi, batch in enumerate(batches):
        sys.stdout.write(f"\r  Batch {bi + 1}/{len(batches)} ({len(all_pairs)} pairs)...")
        sys.stdout.flush()
        pairs = generate_batch(api_key, batch)
        all_pairs.extend(pairs)
        time.sleep(0.5)  # rate limit

    print(f"\r  Generated {len(all_pairs)} Q&A pairs from {len(batches)} API calls")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"  Output: {output}")
    print()


if __name__ == "__main__":
    main()
