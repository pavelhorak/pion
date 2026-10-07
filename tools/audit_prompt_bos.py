#!/usr/bin/env python3
"""Find token lists from a plain `tok.encode()` placed after something else.

`tok.encode(text)` adds the tokenizer's special tokens: Llama 3 prepends
<|begin_of_text|> by default, and tests/_gemma4_text_filter_load.py turns
Gemma 4's <bos> on. Only the first piece of a prompt may carry one. A plain
encode() result on the right of a `+`, in `+=`, or passed to `.extend()` puts
a <bos> in the middle of the prompt, and so does doubling an encoded filler
(`base = base + base`). Twenty-one harnesses had this shape until 2026-10-07;
at 64K it hid the needle from vanilla mlx-lm and Pion alike.

A name counts as a plain piece when the same function assigns it from a plain
encode() (or a list comprehension of them). An encode() with
`add_special_tokens=False`, or with the flag decided at run time, is fine.
A line may opt out with the comment `bos-audit: ok` and a reason.

    python3 tools/audit_prompt_bos.py          # every tracked .py; exit 1 on a hit
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPT_OUT = "bos-audit: ok"


def plain_encode(node) -> bool:
    """`<tokenizer>.encode(...)` that keeps the tokenizer's special tokens."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "encode"):
        return False
    if "tok" not in ast.unparse(node.func.value).lower():
        return False          # str.encode("utf-8") and friends
    for kw in node.keywords:
        if kw.arg == "add_special_tokens":
            return isinstance(kw.value, ast.Constant) and kw.value.value is not False
    return True


def audit_source(src: str, path: str = "<src>") -> list[str]:
    tree = ast.parse(src)
    lines = src.splitlines()
    hits: set[str] = set()
    scopes = [n for n in ast.walk(tree)
              if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef))]
    for scope in scopes:
        nodes = list(ast.walk(scope))
        plain: set[str] = set()        # names holding one plain piece
        plain_lists: set[str] = set()  # names holding a list of them, or derived from one
        for n in nodes:
            if isinstance(n, ast.Assign):
                v = n.value
                names = {t.id for t in n.targets if isinstance(t, ast.Name)}
                if plain_encode(v):
                    plain |= names
                elif isinstance(v, ast.ListComp) and plain_encode(v.elt):
                    plain_lists |= names

        def mentions(expr, pool) -> bool:
            return any(isinstance(m, ast.Name) and m.id in pool for m in ast.walk(expr))

        changed = True
        while changed:     # plan = sorted(zip(depths, stmt_lists)); for d, stmt in plan: ...
            changed = False
            for n in nodes:
                if (isinstance(n, ast.Assign) and not plain_encode(n.value)
                        and mentions(n.value, plain_lists)):
                    new = {t.id for t in n.targets if isinstance(t, ast.Name)} - plain_lists - plain
                    if new:
                        plain_lists |= new
                        changed = True
                if isinstance(n, (ast.For, ast.comprehension)) and mentions(n.iter, plain_lists):
                    new = {m.id for m in ast.walk(n.target) if isinstance(m, ast.Name)} - plain
                    if new:
                        plain |= new
                        changed = True

        def piece(x) -> bool:
            if plain_encode(x):
                return True
            if isinstance(x, ast.Name):
                return x.id in plain
            if isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name):
                return x.value.id in plain
            return False

        for n in nodes:
            after = None
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                after = n.right
            elif isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add):
                after = n.value
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "extend" and n.args):
                after = n.args[0]
            if after is None or not piece(after):
                continue
            if OPT_OUT in (lines[n.lineno - 1] if n.lineno - 1 < len(lines) else ""):
                continue
            hits.add(f"{path}:{n.lineno}: {ast.unparse(n)[:100]}")
    return sorted(hits)


def tracked_python() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.py"],
                         capture_output=True, text=True, check=True).stdout
    return out.split()


def main() -> int:
    hits: list[str] = []
    for rel in tracked_python():
        try:
            hits += audit_source((ROOT / rel).read_text(encoding="utf-8"), rel)
        except SyntaxError:
            continue
    for h in hits:
        print(h)
    print(f"{len(hits)} plain encode() pieces placed after another piece")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
