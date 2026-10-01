# Prompt for Gemini: Generate 500 Q&A pairs

Generate exactly 500 question-answer pairs about **Python programming** (stdlib, common patterns, debugging, packaging, async, data structures, web frameworks, testing).

## Requirements

1. **Paraphrase clusters**: Group questions into ~100 topics. Each topic should have 3-5 paraphrased versions of the same question. This is critical — the system being tested detects when different wordings ask the same thing.

2. **Diverse question styles**: Mix of:
   - "What is X?" / "Explain X" / "How does X work?"
   - "How do I do X?" / "What's the best way to X?"
   - "What's the difference between X and Y?"
   - "Why does X happen?" / "What causes X?"
   - "When should I use X vs Y?"
   - Troubleshooting: "Why am I getting error X?"

3. **Answers**: 2-5 sentences each. Factual, concise, no filler. Include a short code snippet where relevant (1-3 lines max).

4. **Output format**: Raw JSONL (one JSON object per line, no markdown fences, no extra text before or after):

```
{"question": "What is a list comprehension in Python?", "answer": "A list comprehension is a concise way to create lists using a single line of code. The syntax is [expression for item in iterable if condition]. For example, `squares = [x**2 for x in range(10)]` creates a list of squares. It's generally faster than an equivalent for loop because it's optimized internally."}
{"question": "How do list comprehensions work?", "answer": "List comprehensions iterate over an iterable, optionally filter with a condition, and collect the results of an expression into a new list. The syntax `[expr for x in iterable if cond]` replaces a multi-line for loop with append. They support nested loops: `[x*y for x in range(3) for y in range(3)]`."}
{"question": "Can you explain Python list comprehensions?", "answer": "A list comprehension is a compact syntax for building lists: `[expression for item in iterable]`. You can add filtering with `if`: `[x for x in range(20) if x % 2 == 0]`. They're preferred over map/filter for readability and are slightly faster than equivalent for loops."}
```

5. **Topic distribution** (~100 topics, 5 questions each = 500):
   - Core types: list, dict, set, tuple, string, int/float, None, bool (8 topics)
   - Control flow: if/else, for, while, match, try/except, with (6 topics)
   - Functions: def, args, kwargs, lambda, closures, decorators, generators, yield (8 topics)
   - Classes: init, inheritance, super, dunder methods, dataclasses, abstract classes, slots, properties (8 topics)
   - Stdlib: os, sys, pathlib, json, re, datetime, collections, itertools, functools, logging, typing, enum (12 topics)
   - Async: asyncio, await, event loop, tasks, gather, aiohttp (6 topics)
   - Packaging: pip, venv, pyproject.toml, requirements.txt, wheels (5 topics)
   - Testing: pytest, unittest, mocking, fixtures, parametrize (5 topics)
   - Web: Flask, FastAPI, requests, HTTP methods (4 topics)
   - Data: csv, sqlite3, struct, pickle (4 topics)
   - Concurrency: threading, multiprocessing, GIL, concurrent.futures (4 topics)
   - Common errors: ImportError, AttributeError, TypeError, KeyError, IndexError (5 topics)
   - Patterns: context managers, descriptors, metaclasses, abc (4 topics)
   - Performance: profiling, caching, lru_cache, memory (4 topics)
   - File I/O: open, read, write, binary, encoding (4 topics)
   - Misc: comprehensions, walrus operator, f-strings, unpacking, type hints (5 topics)
   - Advanced: GC, memory model, C extensions, ctypes (4 topics)
   - Debugging: pdb, breakpoint, traceback, logging (4 topics)

6. **No duplicates**. Each of the 500 questions must be unique text, even within paraphrase clusters.

7. **Output only JSONL**. No commentary, no headers, no markdown. Just 500 lines of JSON.
