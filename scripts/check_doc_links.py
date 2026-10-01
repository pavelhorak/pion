#!/usr/bin/env python3
"""check_doc_links.py — find dead relative markdown links in a Pion tree.

Usage:
    python3 scripts/check_doc_links.py <tree_root> [--all]

Exit status is 0 when no dead links are found and 1 when any are, so this is
usable as a CI check and as a gate in the flip runbook.

WHY THIS LIVES IN THE REPO
--------------------------
This started as a one-session audit script in a scratchpad under /tmp, and the
release runbook called it from there. macOS purges /tmp, so the runbook had a
step that would simply stop existing. A dead-link gate that lives in a
scratchpad is a gate that is not there on the day you need it. Promoted into
the repo on 2026-09-17, unchanged in its link-resolution logic.

RUN IT AGAINST THE EXPORT TREE, NOT THIS ONE
--------------------------------------------
The private tree and the export tree give DIFFERENT answers, and the private
one understates the problem: a file the export strips is alive here and dead
for every stranger. A clean run in the private tree proves nothing about what
ships. The flip runbook's copy of this check runs against `/tmp/pub`.

WHAT IT SCANS
-------------
By default, the "entry point" set a stranger actually reads: README.md,
doc/index.md, examples/README.md, CONTRIBUTING.md, SECURITY.md, plus every
doc/*.md and doc/blog/*.md. This is the set the 2026-09-16 export audit used,
and it reports 7 dead links on that tree — keep the default stable so that
baseline stays comparable. `--all` sweeps every *.md in the tree instead
(satellite READMEs included), which is broader and noisier.

For each markdown link [text](target) it skips http(s):, mailto: and bare
#anchor targets, strips any #anchor suffix, and accepts the link if the target
exists either relative to the linking file's own directory or relative to the
tree root.

THE PARSING TRAP THIS ALREADY HANDLES
-------------------------------------
Mojo generic-constructor syntax looks exactly like a markdown link to a naive
regex: `SlabAllocator[ListNode](10M)` and `alloc[Int](4)` both parse as
[text](target) with targets "10M" and "4". Fenced blocks and inline code spans
are stripped before scanning, or every doc carrying a Mojo sample reports a
pile of fake dead links. Do not "simplify" that away.
"""
import re
import sys
from pathlib import Path

# Matches [text](target) — non-greedy text, target up to the first
# unescaped ')'. Good enough for standard markdown docs (no nested parens
# handling, which is the common case in this repo).
LINK_RE = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')

FENCE_RE = re.compile(r'^```.*?^```', re.DOTALL | re.MULTILINE)
INLINE_CODE_RE = re.compile(r'`[^`\n]+`')

# Directories that never ship and whose links are not a stranger's problem.
SKIP_DIRS = {
    ".git", ".pixi", "localtemp", "memory", "node_modules", "__pycache__",
    "venv_zvec", "dist", "logs", ".venv",
}


def strip_code(text: str) -> str:
    text = FENCE_RE.sub('', text)
    text = INLINE_CODE_RE.sub('', text)
    return text


def find_entry_files(root: Path) -> list[Path]:
    """The files a stranger reads first. Stable by design — see the docstring."""
    fixed = [
        "README.md",
        "doc/index.md",
        "examples/README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
    ]
    files = []
    for rel in fixed:
        p = root / rel
        if p.is_file():
            files.append(p)
    doc_dir = root / "doc"
    if doc_dir.is_dir():
        files.extend(sorted(doc_dir.glob("*.md")))
        blog_dir = doc_dir / "blog"
        if blog_dir.is_dir():
            files.extend(sorted(blog_dir.glob("*.md")))
    return dedup(files)


def find_all_files(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*.md")):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        files.append(p)
    return dedup(files)


def dedup(files: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(f)
    return out


def is_skippable(target: str) -> bool:
    t = target.strip()
    if not t:
        return True
    if t.startswith("#"):
        return True
    if t.startswith("http://") or t.startswith("https://"):
        return True
    if t.startswith("mailto:"):
        return True
    return False


def check_target(root: Path, linking_file: Path, target: str) -> bool:
    """Return True if the target resolves to an existing file/dir."""
    path_part = target.split("#", 1)[0].strip()
    if not path_part:
        # was a bare "path#anchor" with empty path -> treat as self-reference
        return True
    # some authors write (<path>)
    path_part = path_part.strip("<>")
    candidate1 = (linking_file.parent / path_part)   # relative to the doc
    candidate2 = (root / path_part)                  # relative to the tree root
    for c in (candidate1, candidate2):
        try:
            if c.exists():
                return True
        except OSError:
            pass
    return False


def main() -> int:
    args = [a for a in sys.argv[1:]]
    scan_all = "--all" in args
    positional = [a for a in args if not a.startswith("--")]
    if len(positional) != 1:
        print("usage: check_doc_links.py <tree_root> [--all]", file=sys.stderr)
        return 2

    root = Path(positional[0]).resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    entry_files = find_all_files(root) if scan_all else find_entry_files(root)

    dead_rows = []
    per_file_dead_count = {}
    per_file_total_count = {}

    for f in entry_files:
        rel_f = str(f.relative_to(root))
        per_file_total_count.setdefault(rel_f, 0)
        per_file_dead_count.setdefault(rel_f, 0)
        text = strip_code(f.read_text(errors="replace"))
        for m in LINK_RE.finditer(text):
            target = m.group(2).strip()
            if is_skippable(target):
                continue
            per_file_total_count[rel_f] += 1
            if not check_target(root, f, target):
                per_file_dead_count[rel_f] += 1
                dead_rows.append((rel_f, target))

    mode = "all *.md" if scan_all else "entry points"
    print("# Dead local links report\n")
    print(f"Tree: {root}")
    print(f"Mode: {mode} — {len(entry_files)} files scanned\n")
    if dead_rows:
        print("| linking file | link target | exists? |")
        print("|---|---|---|")
        for rel_f, target in dead_rows:
            print(f"| {rel_f} | `{target}` | NO |")
    else:
        print("(no dead links found)")

    print("\n## Per-file summary (files with at least one link checked)\n")
    print("| file | links checked | dead |")
    print("|---|---|---|")
    for rel_f in sorted(per_file_total_count):
        total = per_file_total_count[rel_f]
        if total == 0:
            continue
        print(f"| {rel_f} | {total} | {per_file_dead_count[rel_f]} |")

    print(f"\nTotal dead links: {len(dead_rows)}")
    return 1 if dead_rows else 0


if __name__ == "__main__":
    sys.exit(main())
