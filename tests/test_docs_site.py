#!/usr/bin/env python3
"""The website says what the repository says (launch S2/S3).

The landing page and the docs are derived from README.md regions, doc/*.md,
the --help printer in src/main.mojo, the generated command table and the
package READMEs. `mkdocs build --strict` already turns a dead link, an orphan
page or a nav entry without a file into a failure; this test covers what the
build cannot see:

  [1] every README snippet region the site includes still exists;
  [2] every `<!-- include-... -->` directive under website/pages resolves;
  [3] no nav entry names a file the public export strips (the private tree has
      more files than the public one, and a nav that referenced one would build
      here and fail on the public repo);
  [4] the generated CLI page lists every flag the help printer knows, and the
      command index counts what the command table counts;
  [5] the landing page carries the four Pion lines, the two numbers with their
      factors, the honesty figure and the install commands — and no unfilled
      placeholder.

Usage:
    python3 tests/test_docs_site.py                # builds into a temp dir
    python3 tests/test_docs_site.py --built _site  # checks an existing build
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "website"))
import sitelib as S  # noqa: E402

failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def nav_files(node, out):
    if isinstance(node, dict):
        for v in node.values():
            nav_files(v, out)
    elif isinstance(node, list):
        for v in node:
            nav_files(v, out)
    elif isinstance(node, str):
        out.append(node)
    return out


def main():
    built = None
    if "--built" in sys.argv:
        built = Path(sys.argv[sys.argv.index("--built") + 1]).resolve()

    print("\n[1] README regions the site includes")
    readme = S.read("README.md")
    used = set(re.findall(r"include-region:\s*README\.md\s*\|\s*([\w-]+)", "\n".join(
        p.read_text(encoding="utf-8") for p in (REPO / "website" / "pages").rglob("*.md"))))
    used |= {"pitch", "four-lines", "four-lines-explained", "pip-install", "two-numbers",
             "security", "build-from-source", "first-five-minutes", "license-paragraph"}
    for name in sorted(used):
        try:
            body = S.region(readme, name)
            check(f"region {name} present and non-empty", len(body.strip()) > 20)
        except KeyError as e:
            check(f"region {name} present", False, str(e))
    four = S.fences(S.region(readme, "four-lines"), "python")
    check("the four lines are the canonical get_or_prefill form",
          bool(four) and "PionPromptCache(model" in four[0] and "get_or_prefill" in four[0] and "prompt_cache=cache" in four[0])

    print("\n[2] include directives under website/pages resolve")
    inc = re.compile(r"<!--\s*include-(section|region|file):\s*(.+?)\s*-->")
    n = 0
    for page in sorted((REPO / "website" / "pages").rglob("*.md")):
        for m in inc.finditer(page.read_text(encoding="utf-8")):
            n += 1
            kind, parts = m.group(1), [x.strip() for x in m.group(2).split("|")]
            src = parts[0]
            try:
                text = S.doc_text(src[4:]) if src == "doc/index.md" else S.read(src)
                if kind == "section":
                    S.section(text, parts[1])
                elif kind == "region":
                    S.region(text, parts[1])
            except (KeyError, FileNotFoundError) as e:
                check(f"{page.relative_to(REPO)}: {kind} {parts[:2]}", False, str(e))
    check(f"all {n} include directives resolve", not any(f[0].startswith("website/pages") for f in failures))

    print("\n[3] nav names only files the public export ships")
    import yaml
    cfg = yaml.safe_load((REPO / "mkdocs.yml").read_text(encoding="utf-8"))
    files = nav_files(cfg["nav"], [])
    strip = REPO / "publication" / "strip_list.txt"
    patterns = []
    if strip.exists():
        for ln in strip.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#", 1)[0].strip()
            if ln.startswith("doc/"):
                patterns.append(ln[4:])
    stripped_in_nav = [f for f in files if any(fnmatch.fnmatch(f, pat) for pat in patterns)]
    check("no nav entry matches a strip-list pattern", not stripped_in_nav, stripped_in_nav)
    allow_file = REPO / "publication" / "doc_allowlist.txt"
    if allow_file.exists():
        allowed = {ln.split("#", 1)[0].strip() for ln in allow_file.read_text(encoding="utf-8").splitlines()} - {""}
        private_in_nav = [f for f in files if (REPO / "doc" / f).exists()
                          and not (REPO / "website" / "pages" / f).exists() and f not in allowed]
        check("every doc/ file in the nav is in doc_allowlist.txt", not private_in_nav, private_in_nav)
    generated = {"index.md", "reference/cli-flags.md", "reference/command-index.md"}
    missing = [f for f in files if not (REPO / "doc" / f).exists() and not (REPO / "website" / "pages" / f).exists() and f not in generated]
    check("every nav entry is a doc file, a website page, or generated", not missing, missing)

    print("\n[4] generated pages agree with their sources")
    if built is None:
        tmp = Path(tempfile.mkdtemp(prefix="pion_site_"))
        env = dict(os.environ, DISABLE_MKDOCS_2_WARNING="true")
        r = subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(tmp / "docs")],
                           cwd=REPO, capture_output=True, text=True, env=env)
        check("mkdocs build --strict succeeds", r.returncode == 0, r.stderr[-800:])
        r = subprocess.run([sys.executable, "website/landing/build_landing.py", "--out", str(tmp)],
                           cwd=REPO, capture_output=True, text=True)
        check("landing builds", r.returncode == 0, r.stderr[-800:])
        built = tmp
    cli = (built / "docs" / "reference" / "cli-flags" / "index.html").read_text(encoding="utf-8")
    n_rows = len(re.findall(r"<td><code>-", cli))
    check(f"CLI page lists every help-printer flag ({S.cli_flag_count()})", n_rows == S.cli_flag_count(), f"page rows {n_rows}")
    cmds = (built / "docs" / "reference" / "command-index" / "index.html").read_text(encoding="utf-8")
    names = S.command_names()
    check(f"command index lists all {len(names)} commands",
          all(f"<code>{n.upper()}</code>" in cmds for n in names),
          [n for n in names if f"<code>{n.upper()}</code>" not in cmds][:10])
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cmds))
    check("command index total matches PION_COMMAND_COUNT",
          S.command_table_count() == len(names) and re.search(r"Total \(\s*PION_COMMAND_COUNT\s*\)\s*%d\b" % S.command_table_count(), text) is not None)
    for page in ("getting-started/install", "getting-started/first-five-minutes", "reference/commands/kv-prefix",
                 "reference/python/pion-vllm-mlx", "project/changelog"):
        check(f"docs page {page} built", (built / "docs" / page / "index.html").exists())
    kv = (built / "docs" / "reference" / "commands" / "kv-prefix" / "index.html").read_text(encoding="utf-8")
    check("KV.PREFIX page carries REGISTER, LOOKUP, PREFILL_MS and the value receipt",
          all(t in kv for t in ("KV.PREFIX.REGISTER", "KV.PREFIX.LOOKUP", "PREFILL_MS", "PION.STATS")))
    api = (built / "docs" / "reference" / "python" / "pion-vllm-mlx" / "index.html").read_text(encoding="utf-8")
    check("Python API page renders PionPromptCache.get_or_prefill from source", "get_or_prefill" in api and "HybridRetrievalCache" in api)

    print("\n[5] the landing page")
    landing = (built / "index.html").read_text(encoding="utf-8")
    check("no unfilled placeholder", not re.search(r"\{\{\w+\}\}", landing))
    check("the four lines", "get_or_prefill" in landing and "PionPromptCache(model" in landing)
    # The second row was "9.26×" (846 → 91 ms) until 2026-09-23: a real run from a
    # different workload, shown beside a 2,048-token row it did not match. Both rows
    # now come from cross_process_ttft.py (--same adds the first), and since
    # 2026-10-02 their vanilla side prefills the way mlx-lm's generate_step does —
    # the earlier 50.6× / 24× computed logits at every prompt position. The
    # 2026-10-07 rerun (the harness's prefix gained its <bos>) reads 26× / 17×.
    check("both TTFT rows with factors", "26×" in landing and "17×" in landing and "69.0 ms" in landing)
    # The correction sentence names the retired factors, so check the retired cells.
    check("the retired TTFT cells are gone", all(c not in landing for c in ("1,530 ms", "64.7 ms", "61.9 ms", "73.9 ms")))
    # The 64K sparse-mask row (326×) was withdrawn on 2026-10-06: its reproducer
    # missed the needle, and its vanilla time was never recorded. The figure that
    # replaced it on 2026-10-07 comes from a published run, and links to it.
    check("the withdrawn 64K row stays withdrawn", "326×" not in landing)
    check("the 64K figure links its raw output",
          "437×" in landing and "2026-10-07-mac-m4/sparse_mask_64k_niah.txt" in landing)
    check("the honesty figure: what a short prefix saves", "1.4×" in landing)  # 1.5× until the 2026-10-07 rerun
    check("install: Homebrew first on macOS", "brew install pavelhorak/tap/pion" in landing)
    check("install: the tarball is one click away", 'href="/docs/getting-started/install/"' in landing)
    check("install: docker run", "docker run" in landing)
    check("the comparison table names its competitors", all(t in landing for t in ("LMCache", "SGLang", "oMLX")))
    check("the licence footer", "Apache-2.0" in landing and "libpion_vector" in landing)
    check("links into the docs", 'href="/docs/' in landing)
    check("version stamped from VERSION", S.read("VERSION").strip() in landing)

    print("\n[6] analytics: Cloudflare's cookieless beacon, only with a configured token")
    token = S.analytics_token()
    check("the beacon is empty without a token and carries it with one",
          S.analytics_beacon("") == "" and "tok123" in S.analytics_beacon("tok123"))
    install = (built / "docs" / "getting-started" / "install" / "index.html").read_text(encoding="utf-8")
    for where, page_html in (("landing page", landing), ("docs pages", install)):
        check(f"beacon on the {where} iff a token is set (token {'set' if token else 'empty'})",
              ("cloudflareinsights.com" in page_html) == bool(token))

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for n_, d in failures:
        print(f"  - {n_}: {d}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
