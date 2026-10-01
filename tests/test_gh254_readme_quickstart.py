#!/usr/bin/env python3
"""gh #254 — the README quick start must work as written.

The acceptance criterion on the issue is "every snippet in the README executed
verbatim succeeds". This EXTRACTS the snippets from README.md rather than
restating them, so the test cannot drift away from the document it is meant to
guard: if someone edits the README into something that does not run, this fails.

What was wrong (verified 2026-08-26, all six confirmed before fixing):

  1. `HSET` came BEFORE `FT.CREATE`, so nothing was ever routed into the HNSW.
  2. The `HSET` was 5 tokens; vector ingest requires >= 6 arguments. And the
     README said "1536-byte-blob" where `DIM 1536` needs **6144** bytes
     (dim * 4). Neither could work.
  3. The semantic-cache example ran against a server started without an
     embedding backend.
  4. `pixi` and `redis-cli` were used but never introduced.
  5. `uvx pion-mcp` 404s — the package is not on PyPI.
  6. Linux-without-GPU was routed to `pixi run build-portable`, which produces
     `pion-server-dev`, and the next line said `./pion-server`.

A seventh was found while fixing and has since been FIXED in the engine:
`DIM` mismatches used to be accepted and answered with nonsense (`DIM 8`
against a 1536-dim server returning three hits all scored 0.0000 in the wrong
order). Re-probed 2026-09-19 with 40 non-colinear vectors and a planted exact
match at D = 8 / 64 / 1536: every one returns the correct nearest neighbour.
`DIM` is per-index, as the README says. The check that pinned the old warning
is retired inline below, with the reproduction attempt recorded.

Usage: python3 tests/test_gh254_readme_quickstart.py [--port 1974]
"""
import argparse, re, os, socket, subprocess, sys, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")

failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def readme():
    with open(README) as f:
        return f.read()


def quickstart_blocks():
    """Fenced code blocks between '## Quick Start' and the next '## '."""
    t = readme()
    start = t.index("## Quick Start")
    end = t.index("\n## ", start + 10)
    return re.findall(r"```(?:bash|python)?\n(.*?)```", t[start:end], re.S)


def cmd(port, *args):
    s = socket.create_connection(("127.0.0.1", port), timeout=15)
    try:
        parts = [f"*{len(args)}\r\n".encode()]
        for a in args:
            a = a if isinstance(a, bytes) else str(a).encode()
            parts.append(b"$%d\r\n%s\r\n" % (len(a), a))
        s.sendall(b"".join(parts))
        import time; time.sleep(0.4)
        return s.recv(8192)
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()

    try:
        socket.create_connection(("127.0.0.1", args.port), timeout=5).close()
    except OSError as e:
        print(f"FATAL: no Pion on {args.port} ({e})")
        return 2

    blocks = quickstart_blocks()
    text = readme()

    print("\n[1] Documented prerequisites exist")
    check("pixi has an install link", "pixi.sh" in text,
          "pixi is used but never introduced")
    check("redis-cli is named as a prerequisite",
          "redis-cli" in text and ("brew install redis" in text
                                   or "redis-tools" in text))
    check("the no-GPU build's binary name is stated",
          "pion-server-dev" in text,
          "build-portable produces pion-server-dev; README said ./pion-server")

    print("\n[2] No dead `uvx pion-mcp` in shipped docs OR source docstrings")
    # .py is in the sweep because the FIRST instance to ship was a docstring:
    # mcp/pion_mcp/server.py told the reader to `pip install pion-mcp` / `uvx
    # pion-mcp`, both 404, and that file is the entry point someone reads when
    # wiring Pion into Claude Code or Cursor. A .md-only guard could not see it,
    # so the check passed for weeks while the most-read install instruction in
    # the package was wrong. Widened 2026-09-19.
    dead = []
    SKIP_DIRS = ("localtemp", "/.git", "/.pixi", "node_modules", "__pycache__",
                 "/venv_zvec", "/experiments")
    for dirpath, _, names in os.walk(ROOT):
        if any(d in dirpath for d in SKIP_DIRS):
            continue
        for n in names:
            if not (n.endswith(".md") or n.endswith(".py")) or n == "FableReview.md":
                continue
            if n == os.path.basename(__file__):
                continue          # this file names the string in order to ban it
            p = os.path.join(dirpath, n)
            try:
                body = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if re.search(r"uvx pion-mcp", body):
                dead.append(os.path.relpath(p, ROOT))
    check("no `uvx pion-mcp` (404 on PyPI) in any shipped doc or docstring", not dead,
          f"still present in {dead}")

    print("\n[3] The vector snippet runs verbatim, and its ORDER is correct")
    vec_block = next((b for b in blocks if "FT.CREATE" in b), None)
    check("a vector snippet exists in Quick Start", vec_block is not None)
    if vec_block:
        ic, ih = vec_block.index("FT.CREATE"), vec_block.index("HSET")
        check("FT.CREATE comes BEFORE HSET", ic < ih,
              "HSET first means nothing is ever indexed")
        # Count the arguments in the execute_command("HSET", ...) call rather
        # than pattern-matching the literal: `v.tobytes()` contains parentheses,
        # so a naive regex reads the call as ending early.
        m = re.search(r'execute_command\(\s*"HSET"(.*?)\)\s*$', vec_block, re.M | re.S)
        n_args = 0
        if m:
            depth, cur, parts = 0, "", []      # NOT `args` — that is argparse's
            for ch in m.group(1):
                if ch in "([": depth += 1
                elif ch in ")]": depth -= 1
                if ch == "," and depth == 0:
                    parts.append(cur); cur = ""
                else:
                    cur += ch
            if cur.strip(): parts.append(cur)
            n_args = 1 + len([a for a in parts if a.strip()])   # +1 for "HSET"
        check("HSET has >= 6 arguments (vector ingest requires it)", n_args >= 6,
              f"counted {n_args}; a 5-token HSET never reaches the vector path")
        check("blob size is stated as DIM*4, not DIM bytes",
              "DIM * 4" in text or "dim * 4" in text.lower(),
              "README said '1536-byte-blob'; DIM 1536 needs 6144")
        # RETIRED 2026-09-19 — this asserted the README warn that DIM must match
        # the server's configured dimension, because a mismatch was ACCEPTED and
        # answered with nonsense ("DIM 8 against a 1536-dim server returned three
        # hits all scored 0.0000 in the wrong order", per this file's docstring).
        #
        # That no longer reproduces. Re-probed against a default server with 40
        # non-colinear random vectors per index and a planted exact match:
        #
        #     D=   8  accepted; nearest == target  OK
        #     D=  64  accepted; nearest == target  OK
        #     D=1536  accepted; nearest == target  OK
        #
        # So DIM really is per-index, which is what the README says. The check
        # was failing on main too — it had been red for long enough that nobody
        # noticed the underlying bug had been fixed underneath it, and satisfying
        # it would have meant writing a warning about behaviour the engine no
        # longer has. What survives is the blob-size rule above (DIM * 4), which
        # is the trap that IS still real.
        #
        # Restore this check only with a fresh reproduction attached.
        check("the snippet states the DIM/blob-size relationship",
              "DIM * 4" in text or "dim * 4" in text.lower(),
              "the per-index DIM must still be paired with its DIM*4 blob size")

        py = vec_block
        if py.lstrip().startswith("pip install"):
            py = "\n".join(py.splitlines()[1:])
        py = py.replace("python3 - <<'PY'", "").replace("\nPY", "")
        r = subprocess.run([sys.executable, "-c", textwrap.dedent(py)],
                           capture_output=True, text=True, timeout=300, cwd=ROOT)
        ok = r.returncode == 0 and "nearest = doc:7" in r.stdout
        check("the vector snippet runs and finds the exact match", ok,
              f"rc={r.returncode} out={r.stdout[-200:]!r} err={r.stderr[-300:]!r}")

    print("\n[4] The KV snippet runs verbatim")
    check("SET/GET round-trips", cmd(args.port, "SET", "user:1", '{"name":"alice"}')
          .startswith(b"+OK") and b"alice" in cmd(args.port, "GET", "user:1"))

    print("\n[5] The semantic-cache snippet is documented against a working server")
    check("semantic cache names an embedding backend",
          "--nle-embed" in text and "install-inference" in text,
          "the example needs a server started with an embedding backend")
    check("it does not claim a multi-worker server",
          "-w 4" not in text.split("## Quick Start")[1].split("\n## ")[0],
          "the cache index is per-worker")

    print("\n[6] The flagship PionPromptCache snippet has the shape that actually runs")
    # Found 2026-09-21. The four Pion lines at the top of the README, in the
    # Feature Set, in the thesis post, in pion_vllm_mlx/__init__.py and in
    # pion-vllm-mlx/README.md read `cache = PionPromptCache(host=…, port=…)` +
    # "hand `cache` to mlx-lm where you would pass make_prompt_cache(model)".
    # PionPromptCache is a MANAGER, not a cache: `model` is its first
    # parameter, it has no update_and_fetch/state, and the object mlx-lm
    # consumes comes from make_pion_prompt_cache() (Stage 2 — the path the
    # 50.6× number is measured on) or get_or_prefill() (Stage 1). None of the
    # copies sits inside "## Quick Start", so sections [1]-[5] could not see
    # them. The shape is pinned against the SOURCE (ast), not against another
    # doc — a doc agreeing with a doc proves nothing.
    import ast

    def _read(rel):
        with open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore") as f:
            return f.read()

    tree = ast.parse(_read("pion-vllm-mlx/pion_vllm_mlx/prompt_cache.py"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "PionPromptCache")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    first = init.args.args[1].arg if len(init.args.args) > 1 else None
    check("PionPromptCache.__init__'s first parameter is `model`", first == "model",
          f"got {first!r}")
    gop = next((n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "get_or_prefill"), None)
    check("PionPromptCache.get_or_prefill(prefix_token_ids, namespace) exists",
          gop is not None and [a.arg for a in gop.args.args][1:3] == ["prefix_token_ids", "namespace"])
    patch_src = _read("pion-vllm-mlx/pion_vllm_mlx/mlx_lm_patch.py")
    check("PionPrefixCache exposes `state` (mlx-lm's generate_step evals it on every cache entry)",
          re.search(r"class PionPrefixCache[\s\S]*?def state\(self\)", patch_src) is not None,
          "without it, generate(..., prompt_cache=make_pion_prompt_cache(...)) raises AttributeError")

    top_blocks = re.findall(r"```python\n(.*?)```", text.split("## Why Pion")[0], re.S)
    check("a Python snippet exists above ## Why Pion", bool(top_blocks))
    feature = text.split("### Shared KV Cache")[1].split("\n### ")[0] if "### Shared KV Cache" in text else ""
    copies = {
        "README.md (top)": top_blocks[0] if top_blocks else "",
        "README.md (Feature Set)": feature,
        "pion-vllm-mlx/pion_vllm_mlx/__init__.py": _read("pion-vllm-mlx/pion_vllm_mlx/__init__.py"),
        "pion-vllm-mlx/README.md": _read("pion-vllm-mlx/README.md"),
    }
    # The launch post exists only in the private tree; check it where it exists.
    post = "doc/blog/2026-10-pion-memory-engine.md"
    if os.path.isfile(os.path.join(ROOT, post)):
        copies[post] = _read(post)
    for name, body in copies.items():
        check(f"{name}: no `PionPromptCache(host=` — a manager is not a cache",
              "PionPromptCache(host=" not in body)
    for name in [n for n in ("README.md (top)", post,
                 "pion-vllm-mlx/pion_vllm_mlx/__init__.py", "pion-vllm-mlx/README.md") if n in copies]:
        body = copies[name]
        check(f"{name}: constructs PionPromptCache(model, ...)", "PionPromptCache(model" in body)
        check(f"{name}: builds the cache mlx-lm consumes",
              "make_pion_prompt_cache(" in body or "get_or_prefill(" in body,
              "neither make_pion_prompt_cache nor get_or_prefill appears")

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
