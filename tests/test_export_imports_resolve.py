#!/usr/bin/env python3
"""Every Python file the export ships must be able to find its local imports.

WHY THIS EXISTS

Three times on 2026-09-19 a file shipped that could not run, each time because
it imported something the export strips, and each time it was found by hand
after the fact:

  1. examples/moe_expert_substrate_demo.py put experiments/moe_expert_phase0/
     on sys.path and imported pion_moe_tier from it. The capstone demo — the
     one the "51.6 GB model on a 16 GB Mac" claim points at — raised
     ImportError for every reader outside the development repo.
  2. Two reproducers copied into benchmarks/reproducers/ imported
     gh23_cag_hybrid, pion_moe_tier_client and eval_prune_quality. Published,
     they would have been reproducers that fail on their first line.
  3. pion_memory.py was stripped as a "zombie root". Nine examples import it,
     including agent_memory_demo.py, which is a first-class indexed demo.

None of these were catchable by the existing checks. They are code, not links,
so the dead-link scanner cannot see them; no test imports the examples; and
one of the dependencies is written `REPO / 'experiments' / 'moe_expert_phase0'`
-- split into path components, so even a grep for the literal string misses it.

WHAT IT CHECKS

For each shipped .py file, every `import X` / `from X import ...` where X is
neither stdlib nor a known third-party package must resolve to something that
also ships. Resolution follows the file's own sys.path manipulation, because
that is how the real failures were constructed.

Usage:
    python3 tests/test_export_imports_resolve.py <export_tree>
    python3 tests/test_export_imports_resolve.py .        # the private tree

Exit 0 when every local import resolves, 1 otherwise.
"""
import ast
import os
import sys

# Packages a user installs. Their absence is a documented prerequisite, not a
# packaging bug, so they are not this test's business.
THIRD_PARTY = {
    "numpy", "mlx", "mlx_lm", "torch", "transformers", "redis", "requests",
    "datasets", "fastapi", "httpx", "uvicorn", "llama_cpp", "onnxruntime",
    "tqdm", "sentence_transformers", "pydantic", "yaml", "tomli", "aiohttp",
    "openai", "anthropic", "huggingface_hub", "safetensors", "scipy",
    "matplotlib", "pandas", "sklearn", "pytest", "setuptools", "build",
    "twine", "valkey", "glide", "psutil", "pyarrow", "flask", "mcp",
    "autogen_core", "langgraph", "llama_index", "ann_benchmarks",
    "mkdocs", "mkdocs_gen_files", "markdown",   # the website tooling,
                    # pinned in website/requirements.txt (gh #303)
    "benchmarks",   # benchmarks/cache_hit_validation/run.py imports the repo
                    # root as a package; it is run from the root by design.
}

SKIP_DIRS = {".git", ".pixi", "localtemp", "memory", "experiments",
             "node_modules", "__pycache__", "venv_zvec", ".venv", "dist",
             "build", "site-packages"}


def stdlib_names() -> set:
    names = set(sys.stdlib_module_names)
    names.update({"__future__", "typing_extensions"})
    return names


STDLIB = stdlib_names()


def shipped_modules(root: str) -> set:
    """Top-level module names importable from the tree, by any shipped file."""
    mods = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                mods.add(fn[:-3])
        # a directory with __init__.py is an importable package
        if "__init__.py" in filenames:
            mods.add(os.path.basename(dirpath))
        for d in dirnames:
            if os.path.exists(os.path.join(dirpath, d, "__init__.py")):
                mods.add(d)
    return mods


def python_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def guarded_import_lines(tree: ast.AST) -> set:
    """Line numbers of imports wrapped in try/except ImportError.

    A module imported inside such a block is an optional dependency the file
    handles itself — tests/test_moe_ns_prune_selection.py prints SKIP and
    exits 0 when its research-tree module is absent. Flagging those would
    train readers to ignore this check, which is how the audit that never
    fires gets written.
    """
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import = any(
            (isinstance(h.type, ast.Name) and h.type.id in
             ("ImportError", "ModuleNotFoundError"))
            or (isinstance(h.type, ast.Tuple) and any(
                isinstance(e, ast.Name) and e.id in
                ("ImportError", "ModuleNotFoundError") for e in h.type.elts))
            for h in node.handlers)
        if catches_import:
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    guarded.add(sub.lineno)
    return guarded


def top_level_imports(path: str) -> set:
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except SyntaxError:
        return set()
    guarded = guarded_import_lines(tree)
    out = set()
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) in guarded:
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:          # relative import, resolves within its package
                continue
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def main() -> int:
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    ships = shipped_modules(root)
    failures = []
    checked = 0

    for path in sorted(python_files(root)):
        rel = os.path.relpath(path, root)
        checked += 1
        for mod in sorted(top_level_imports(path)):
            if mod in STDLIB or mod in THIRD_PARTY or mod in ships:
                continue
            failures.append((rel, mod))

    print(f"tree:    {root}")
    print(f"scanned: {checked} Python files")
    print(f"shipped top-level modules: {len(ships)}")

    if failures:
        print(f"\nUNRESOLVABLE LOCAL IMPORTS: {len(failures)}\n")
        print(f"  {'file':<58} {'imports'}")
        for rel, mod in failures:
            print(f"  {rel:<58} {mod}")
        print("\nEach of these raises ImportError for anyone who has only this")
        print("tree. Either the dependency must ship, or the file must not.")
        return 1

    print("\nevery local import resolves within the tree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
