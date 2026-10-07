#!/usr/bin/env python3
"""No prompt is assembled with a plain `tok.encode()` piece after another piece.

WHY
`tok.encode(text)` adds special tokens: Llama 3 prepends <|begin_of_text|>
and the Gemma 4 loader turns <bos> on, so every piece after the first put a
<bos> in the middle of the prompt. Twenty-one harnesses did it until
2026-10-07. The 64K NIAH prompt held 397 and neither vanilla mlx-lm nor Pion
found the needle; RULER's multi-value NIAH lost a value on every path; the
published Stage 1 / Stage 2 workloads, BLEU and hybrid-retrieval figures were
all measured on such prompts. tests/test_prompt_bos.py checks the factored
builders against real tokenizers; this audit covers every tracked .py,
including prompts built inline, and needs no model.

It canaries itself first: each shape that shipped must be reported, and the
fixed spellings must not be.

    python3 tests/test_audit_prompt_bos.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import audit_prompt_bos as A  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)


SHIPPED = {
    "a query after a system prompt (bench_w1_stage2, test_kv_prefix_workload)":
        "def f(tok, sys_ids, q):\n    return sys_ids + tok.encode(q)\n",
    "needle and question pieces (test_long_context_niah)":
        "def f(tok, filler, needle, q, i):\n"
        "    needle_tokens = tok.encode(needle)\n"
        "    question_tokens = tok.encode(q)\n"
        "    return filler[:i] + needle_tokens + filler[i:] + question_tokens\n",
    "a doubled encoded filler (test_hybrid_per_layer_agreement)":
        "def f(tok, n):\n    base = tok.encode('x ' * 64)\n    while len(base) < n:\n        base = base + base\n    return base[:n]\n",
    "statements from a list comprehension (test_ruler_subset)":
        "def f(tok, out, ss):\n    lists = [tok.encode(s) for s in ss]\n    for s in lists:\n        out.extend(s)\n    return out\n",
    "an appended paragraph (bench_nemotron_h_ttft)":
        "def f(tok, ids, para):\n    ids += tok.encode(para)\n    return ids\n",
    "a chunk and a suffix (stage1_hybrid_recall_bench)":
        "def f(tokenizer, c, q):\n    chunk_ids = tokenizer.encode(c)\n    suffix_ids = tokenizer.encode(q)\n    return chunk_ids + suffix_ids\n",
}
FIXED = {
    "pieces without special tokens, one <bos> first":
        "def f(tok, filler, needle, q, i):\n"
        "    bos = [tok.bos_token_id]\n"
        "    needle_tokens = tok.encode(needle, add_special_tokens=False)\n"
        "    question_tokens = tok.encode(q, add_special_tokens=False)\n"
        "    return bos + filler[:i] + needle_tokens + filler[i:] + question_tokens\n",
    "the first piece keeps its <bos>":
        "def f(tok, c, q):\n    return tok.encode(c) + tok.encode(q, add_special_tokens=False)\n",
    "the flag decided at run time (stage0's special=)":
        "def f(tokenizer, a, b, special):\n    return tokenizer.encode(a) + tokenizer.encode(b, add_special_tokens=special)\n",
    "str.encode is not a tokenizer":
        "def f(a, b):\n    return a.encode('utf-8') + b.encode('utf-8')\n",
    "an opted-out line":
        "def f(tok):\n    return tok.encode('a') + tok.encode('b')  # bos-audit: ok — a canary\n",
}

print("[1] every shape that shipped is reported (canary)")
for name, src in SHIPPED.items():
    hits = A.audit_source(src, "canary.py")
    check(name, len(hits) >= 1, "not reported")
print("[2] the fixed spellings are not")
for name, src in FIXED.items():
    hits = A.audit_source(src, "canary.py")
    check(name, not hits, "; ".join(hits))

print("[3] the tree")
hits = []
for rel in A.tracked_python():
    try:
        hits += A.audit_source((ROOT / rel).read_text(encoding="utf-8"), rel)
    except SyntaxError:
        continue
check(f"no plain encode() piece after another piece ({len(A.tracked_python())} files)", not hits,
      "\n      " + "\n      ".join(hits))

print()
print("PASS — 0 failure(s)" if not failures else f"FAIL — {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
