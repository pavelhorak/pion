#!/usr/bin/env python3
"""Every measured number in the public docs must name its evidence.

A claim is a number with a performance unit in prose: a ratio (20×), a
throughput (8,801 QPS, 2.4M ops/sec, 30 tok/s), a latency or duration
(61.9 ms, 86 µs), a percentage or percentage points, a memory size (50 MB),
a bare decimal next to a metric (recall 0.937, F1 0.37), or a bare number in a
table about rates or latency (| QPS | 8,801 |), table headers included.
Numbers in code blocks, inline code and link targets are commands and paths,
not claims. A number that names an input rather than a result — "64K
context", "a 2K prefix", "1M slots", "a 16 GB Mac" — is not a claim either.

The unit of checking is a paragraph, a list item or a table row, so a
sentence wrapped over several lines is one claim. Each claim must be covered
by an entry in benchmarks/claims.toml:

    [[claim]]
    file = "README.md"
    where = "2,049-token prefix"         # a substring of the paragraph or row
    numbers = ["1,242 ms", "61.9 ms", "20×"]
    kind = "result"
    evidence = ["benchmarks/reproducers/results/cross_process_ttft_2026_10_02.json"]
    expect = ["61.9", "20.08"]           # strings the evidence must contain

Kinds:
  result      raw output under benchmarks/results/ or benchmarks/reproducers/results/,
              from a harness in this repo that the result's README names;
  test        a test in tests/manifest.toml that asserts the number;
  code        a constant or default in the source;
  external    a third-party fact (a model's download size, a release asset); evidence may be a URL;
  arithmetic  computed from other numbers in the same claim, or from a stated shape.

The check fails when a claim has no entry, when an entry's evidence file is
missing or lacks an `expect` string, or when an entry matches nothing any more.

    python3 tools/check_doc_claims.py            # report; exit 1 on any problem
    python3 tools/check_doc_claims.py --list     # every claim, covered or not
"""
from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "benchmarks" / "claims.toml"
KINDS = {"result", "test", "code", "external", "arithmetic"}

# Dated records and legal text, not statements about the product today.
EXEMPT_FILES = {"CHANGELOG.md", "CODE_OF_CONDUCT.md", "CLA.md", "website/README.md"}
EXEMPT_DIRS = ("tests/", "benchmarks/results/", "benchmarks/reproducers/results/", "held/", "vendor/")

NUM = r"(?<![\w.#/@$=:-])[+\-−~≈]?\d(?:[\d,]*\d)?(?:\.\d+)?"
UNIT = (r"(?:(?:[x×](?![\w])(?!\s?\d))|\s?(?:%|pp(?![\w])|ms(?![\w])|µs|us(?![\w])|ns(?![\w])|"
        r"s(?![\w])|sec(?:onds?)?(?![\w])|min(?:utes?)?(?![\w])|"
        r"[KMG]?\s?(?:QPS|qps|ops/s(?:ec)?|req/s|RPS|rps|q/s|tok/s|tokens?/s(?:ec(?:ond)?)?|"
        r"tokens per second|ops per second|queries per second)|"
        r"[KMG]i?B(?![\w])|TB(?![\w])))")
CLAIM = re.compile(NUM + UNIT)
BARE_KM = re.compile(NUM + r"[KM](?![\w+])")      # 350K, 2.55M (unit in the header); not 2K+1, a formula
# A bare decimal is a claim when the paragraph, row or table header names a metric.
METRIC_DEC = re.compile(r"(?<![\w.,])(?:0\.\d{2,}|1\.0{2,})(?![\w.])")
METRIC_WORD = re.compile(r"\b(?:recall|F1|accuracy|precision|BLEU|agreement|EM|hit rate|cosine|NDCG|MRR|"
                         r"perplexity|PPL|score)\b", re.I)
# In a table whose row or header names a rate or a latency, a bare number is a
# measurement too (| QPS | 8,801 | 6,689 |).
TABLE_METRIC = re.compile(r"(?:\bQPS\b|\bqps\b|ops/s|\bRPS\b|tok/s|q/s|\blatency\b|\bthroughput\b|\bTTFT\b)", re.I)
TABLE_NUM = re.compile(r"(?<![\w.,#/$=@:-])(?:\d{1,3}(?:,\d{3})+|\d{3,})(?:\.\d+)?(?![\w.%×])"
                       r"(?!\s?(?:ms|µs|us|ns|s\b|sec|min|%|×|x\b|[KMG]i?B|TB|QPS|qps|q/s|tok/s|tokens?/s|ops|RPS|rps|req/s))")
# A number followed by one of these names an input, not a result.
INPUT_NOUN = re.compile(
    r"\s*-?\s*(?:context|prefix(?:es)?|tokens?|tok\b|-token|slots?|vectors?|keys?|entries|elements|nodes|items|"
    r"documents?|docs|queries|requests?|reqs|rows?|dims?|dimensions|members|fields|connections|clients|parameters?|params|bytes?|-byte|"
    r"Mac\b|Mac mini|MacBook|machine|RAM|of RAM|box|laptop|unified|GPU|VRAM|card|device|NIAH|sparse|★)", re.I)


def git_files() -> set[str] | None:
    """Every file git tracks or would add (untracked, not ignored); None outside a
    git tree. A working checkout can hold gitignored notes, symlinked in, that no
    clean checkout publishes. `git check-ignore` cannot sort those out: it aborts
    on the first path that runs through a symlinked directory."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                             capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return set(out.stdout.decode("utf-8", errors="replace").split("\0")) - {""}


def published_files() -> list[Path]:
    files = sorted(ROOT.glob("*.md")) + sorted(ROOT.glob("*/README.md")) + sorted(ROOT.glob("*/*/README.md"))
    files += sorted((ROOT / "doc").glob("**/*.md")) + sorted((ROOT / "website").glob("**/*.md"))
    files += sorted((ROOT / "website" / "landing").glob("*.html"))
    known = git_files()
    out, seen = [], set()
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        if rel in seen or rel in EXEMPT_FILES or f.name in EXEMPT_FILES or rel.startswith(EXEMPT_DIRS) or not f.is_file():
            continue
        if known is not None and rel not in known:
            continue
        seen.add(rel)
        out.append(f)
    return out


def _clean(line: str) -> str:
    line = re.sub(r"`[^`]*`", " ", line)                 # inline code
    line = re.sub(r"\]\([^)]*\)", "]", line)              # link targets
    line = re.sub(r"<[^>]+>", " ", line)                  # inline HTML
    line = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", line)     # dates
    line = re.sub(r"\bv?\d+\.\d+\.\d+\b", " ", line)       # versions
    return line


def blocks(f: Path):
    """(first line number, text) per paragraph, list item or table row, with
    fenced code, HTML comments and (for HTML) style/script removed."""
    text = f.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    cur: list[str] = []
    start = 0
    in_code = in_style = False
    header = prev_row = ""
    prev_is_row = False

    def flush():
        nonlocal cur
        if cur:
            yield start, " ".join(cur), header
        cur = []

    for i, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if s.startswith("```") or s.startswith("~~~"):
            yield from flush()
            in_code = not in_code
            continue
        if in_code:
            continue
        if f.suffix == ".html":
            if re.search(r"<(style|script)\b", s):
                in_style = True
            if in_style:
                if re.search(r"</(style|script)>", s):
                    in_style = False
                continue
            line = html.unescape(re.sub(r"<[^>]+>", " ", raw)).strip()
            if line:
                yield i, _clean(line), ""
            continue
        line = _clean(raw).strip()
        if not line:
            yield from flush()
            prev_is_row = False
            continue
        new_unit = (s.startswith("|") or re.match(r"(#{1,6}\s|[-*+]\s|\d+[.)]\s|>)", s) is not None)
        if new_unit or s.startswith("|"):
            yield from flush()
        if not cur:
            start = i
        cur.append(line)
        if s.startswith("|"):
            if re.fullmatch(r"\|?[\s:|-]+\|?", s):          # the separator row: the row before was a header
                header = prev_row
                cur = []
                continue
            if not prev_is_row:
                header = ""
            prev_row = line
            yield from flush()
        prev_is_row = s.startswith("|")
    yield from flush()


def norm(tok: str) -> str:
    t = tok.strip().replace(",", "").replace("−", "-").replace(" ", "").replace(" ", "")
    t = t.lstrip("+~≈")
    return t.replace("x", "×") if re.fullmatch(r"-?[\d.]+x", t) else t


def claims_in(text: str, header: str = "") -> list[str]:
    found = []
    for pat in (CLAIM, BARE_KM):
        for m in pat.finditer(text):
            if INPUT_NOUN.match(text, m.end()):
                continue
            found.append(m.group(0))
    if METRIC_WORD.search(text) or METRIC_WORD.search(header):
        found += [m.group(0) for m in METRIC_DEC.finditer(text)]
    if text.lstrip().startswith("|") and (TABLE_METRIC.search(text) or TABLE_METRIC.search(header)):
        found += [m.group(0) for m in TABLE_NUM.finditer(text) if not INPUT_NOUN.match(text, m.end())]
    return list(dict.fromkeys(norm(t) for t in found))


def load_register() -> list[dict]:
    if not REGISTER.exists():
        return []
    return tomllib.loads(REGISTER.read_text(encoding="utf-8")).get("claim", [])


def check_entry(e: dict) -> list[str]:
    where = f"benchmarks/claims.toml: {e.get('file')} '{e.get('where')}'"
    probs = []
    if e.get("kind") not in KINDS:
        probs.append(f"{where}: unknown kind {e.get('kind')!r}")
    evidence = e.get("evidence", [])
    if e.get("kind") in {"result", "test", "code"} and not evidence:
        probs.append(f"{where}: kind {e['kind']} needs evidence")
    bodies = []
    for ev in evidence:
        if ev.startswith(("http://", "https://")):
            if e.get("kind") != "external":
                probs.append(f"{where}: only external evidence may be a URL ({ev})")
            continue
        p = ROOT / ev
        if not p.exists():
            probs.append(f"{where}: evidence {ev} does not exist")
        elif p.is_file():
            bodies.append(p.read_text(encoding="utf-8", errors="replace"))
    if bodies:
        for want in e.get("expect", []):
            if not any(want in b for b in bodies):
                probs.append(f"{where}: no evidence file contains {want!r}")
    return probs


def ignored_by_git(paths: list[str]) -> set[str]:
    """The paths .gitignore would keep out of a commit: evidence that exists here
    and in no clean checkout (benchmarks/results/ once fell under `*.txt`)."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--stdin"], input="\n".join(paths),
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return set()
    return set(out.stdout.split())


def check(list_all: bool = False) -> int:
    reg = load_register()
    problems: list[str] = []
    for e in reg:
        problems += check_entry(e)
    local = sorted({ev for e in reg for ev in e.get("evidence", []) if not ev.startswith(("http://", "https://"))})
    for ev in sorted(ignored_by_git(local)):
        problems.append(f"benchmarks/claims.toml: evidence {ev} is gitignored, so no clean checkout has it")
    used = [False] * len(reg)
    by_file: dict[str, list[tuple[int, dict]]] = {}
    for idx, e in enumerate(reg):
        by_file.setdefault(e["file"], []).append((idx, e))
    n_claims = unsourced = 0
    files = published_files()
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        entries = by_file.get(rel, [])
        for line_no, text, header in blocks(f):
            nums = claims_in(text, header)
            if not nums:
                continue
            n_claims += len(nums)
            covered: set[str] = set()
            for idx, e in entries:
                if e["where"] in text:
                    used[idx] = True
                    covered |= {norm(n) for n in e.get("numbers", [])}
            missing = [n for n in nums if n not in covered]
            if list_all or missing:
                print(f"{'UNSOURCED' if missing else 'ok':9} {rel}:{line_no}: {', '.join(missing or nums)} | {text[:150]}")
            if missing:
                unsourced += 1
                problems.append(f"{rel}:{line_no}: no evidence for {', '.join(missing)}")
    for idx, e in enumerate(reg):
        if not used[idx]:
            problems.append(f"benchmarks/claims.toml: {e['file']} '{e['where']}' matches nothing (stale entry)")
    print(f"\n{n_claims} numbers in {len(files)} files; {len(reg)} register entries; "
          f"{unsourced} paragraphs or rows unsourced; {len(problems) - unsourced} register problems")
    for p in problems:
        if "no evidence for" not in p:
            print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print every claim, covered or not")
    sys.exit(check(ap.parse_args().list))
