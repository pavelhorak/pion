#!/usr/bin/env python3
"""Compiler-guided Mojo 1.0 deprecation migrations.

`x[i]` and `a + b` are the same syntax on Pointer, List, InlineArray, SIMD, Dict
and every user struct, and only the pointer ones migrate — so a regex cannot do
this. But the compiler has already resolved the ambiguity: every warning carries
an exact file:line:col, and for binary operators the caret diagram encodes the
full expression extent:

    nr = external_call["pion_read", Int64](fd, buf + total_n, 65535 - total_n)
                                               ~~~~^~~~~~~~~

The `~` run gives both operand boundaries and `^` sits on the operator, so the
rewrite is mechanical without ever inferring a receiver's type. (Note the
compiler flagged `buf + total_n` and left `65535 - total_n` alone — it is
discriminating, not pattern-matching.)

Kinds:
    getitem   p[i]     -> p[unsafe_offset=i]        (insert at a column)
    add       p + n    -> p.unsafe_offset(n)        (rewrite using the extent)

Usage:
    pixi run build-dev > build.log 2>&1
    python3 scripts/migrate_deprecations.py build.log --kind getitem [--apply]
    python3 scripts/migrate_deprecations.py build.log --kind add     [--apply]
      [--only SUBSTR]... [--exclude SUBSTR]...

Default is a dry run; nothing is written without --apply. Rebuild before every
run — the log's line/column numbers must match the tree on disk.

Safety, in both kinds:
  * Every edit verifies the character the compiler pointed at is what it should
    be ('[' or '+') before writing. A mismatch is skipped and counted, never
    guessed at. A non-zero skip count means the log is stale — rebuild.
  * Sites are de-duplicated; the compiler reports a site once per generic
    instantiation and editing twice would corrupt the line.
  * Within a line, edits apply right-to-left so an earlier edit cannot shift a
    later site's recorded column.
  * `add` additionally skips OVERLAPPING extents (nested `a + b + c`), because
    rewriting the inner expression invalidates the outer one's recorded extent.
    Those are reported; rebuild and re-run to converge.
"""
import collections
import re
import sys

WARN_GETITEM = re.compile(
    r"^(?P<file>/.*?):(?P<line>\d+):(?P<col>\d+): warning: positional "
    r"`__getitem__` is deprecated")
WARN_ADD = re.compile(
    r"^(?P<file>/.*?):(?P<line>\d+):(?P<col>\d+): warning: '__add__' is "
    r"deprecated")
INSERT = "unsafe_offset="


def collect_getitem(lines):
    sites = collections.defaultdict(list)
    for raw in lines:
        m = WARN_GETITEM.match(raw)
        if m:
            sites[m.group("file")].append((int(m.group("line")), int(m.group("col"))))
    return {f: sorted(set(v)) for f, v in sites.items()}


def collect_add(lines):
    """Needs the caret diagram, so it reads the two lines after each warning."""
    sites = collections.defaultdict(list)
    for i, raw in enumerate(lines):
        m = WARN_ADD.match(raw)
        if not m or i + 2 >= len(lines):
            continue
        caret = lines[i + 2].rstrip("\n")
        if "^" not in caret or set(caret.strip()) - set("~^"):
            continue  # not a caret diagram
        start = min(caret.index("~") if "~" in caret else len(caret), caret.index("^"))
        end = max(caret.rindex("~") if "~" in caret else -1, caret.rindex("^"))
        sites[m.group("file")].append((int(m.group("line")), start, caret.index("^"), end))
    return {f: sorted(set(v)) for f, v in sites.items()}


def apply_getitem(path, sites, apply):
    with open(path) as f:
        lines = f.readlines()
    edits, skips = 0, []
    by_line = collections.defaultdict(list)
    for ln, col in sites:
        by_line[ln].append(col)

    for ln, cols in by_line.items():
        if ln - 1 >= len(lines):
            skips.append(f"{path}:{ln}: past end of file (stale log?)")
            continue
        text = lines[ln - 1]
        for col in sorted(cols, reverse=True):
            i = col - 1
            if i >= len(text) or text[i] != "[":
                got = text[i] if i < len(text) else "EOL"
                skips.append(f"{path}:{ln}:{col}: expected '[', found {got!r}")
                continue
            if text[i + 1:i + 1 + len(INSERT)] == INSERT:
                continue
            text = text[:i + 1] + INSERT + text[i + 1:]
            edits += 1
        lines[ln - 1] = text

    if apply and edits:
        with open(path, "w") as f:
            f.writelines(lines)
    return edits, len(by_line), skips


def apply_add(path, sites, apply):
    with open(path) as f:
        lines = f.readlines()
    edits, skips = 0, []
    by_line = collections.defaultdict(list)
    for ln, start, op, end in sites:
        by_line[ln].append((start, op, end))

    for ln, spans in by_line.items():
        if ln - 1 >= len(lines):
            skips.append(f"{path}:{ln}: past end of file (stale log?)")
            continue
        text = lines[ln - 1].rstrip("\n")
        done = []  # ranges already rewritten on this line
        for start, op, end in sorted(spans, key=lambda s: s[0], reverse=True):
            # Overlap FIRST. A nested `a + b + c` produces two sites whose
            # extents contain one another; once the inner one is rewritten the
            # outer one's recorded columns are stale, so checking the character
            # first would report "expected '+'" — which this script documents as
            # meaning the log is out of date, sending the reader after the wrong
            # problem. Nested is normal and converges on a rebuild; stale is not.
            if any(not (end < ds or start > de) for ds, de in done):
                skips.append(f"{path}:{ln}:{op + 1}: nested/overlapping extent — "
                             f"rebuild and re-run to converge")
                continue
            if op >= len(text) or text[op] != "+":
                got = text[op] if op < len(text) else "EOL"
                skips.append(f"{path}:{ln}:{op + 1}: expected '+', found {got!r}")
                continue
            lhs = text[start:op].rstrip()
            rhs = text[op + 1:end + 1].strip()
            if not lhs or not rhs:
                skips.append(f"{path}:{ln}:{op + 1}: empty operand, refusing")
                continue
            text = f"{text[:start]}{lhs}.unsafe_offset({rhs}){text[end + 1:]}"
            done.append((start, end))
            edits += 1
        lines[ln - 1] = text + "\n"

    if apply and edits:
        with open(path, "w") as f:
            f.writelines(lines)
    return edits, len(by_line), skips


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    log = args[0]
    apply = "--apply" in args
    kind = "getitem"
    for i, a in enumerate(args):
        if a == "--kind" and i + 1 < len(args):
            kind = args[i + 1]
    only = [args[i + 1] for i, a in enumerate(args) if a == "--only" and i + 1 < len(args)]
    exclude = [args[i + 1] for i, a in enumerate(args)
               if a == "--exclude" and i + 1 < len(args)]
    if kind not in ("getitem", "add"):
        print(f"unknown --kind {kind!r}; expected 'getitem' or 'add'")
        return 1

    with open(log, errors="replace") as f:
        loglines = f.readlines()

    sites = (collect_getitem(loglines) if kind == "getitem"
             else collect_add(loglines))
    if not sites:
        print(f"no '{kind}' warnings found in {log}")
        return 1

    applier = apply_getitem if kind == "getitem" else apply_add
    total_edits = total_skips = 0
    for path in sorted(sites):
        if only and not any(o in path for o in only):
            continue
        if exclude and any(e in path for e in exclude):
            continue
        try:
            edits, nlines, skips = applier(path, sites[path], apply)
        except OSError as e:
            print(f"    SKIP {path}: {e}")
            total_skips += 1
            continue
        total_edits += edits
        total_skips += len(skips)
        if edits or skips:
            print(f"{'EDIT' if apply else 'would edit'} {edits:5d} site(s) "
                  f"on {nlines:4d} line(s)  {path}")
        for s in skips[:4]:
            print(f"    SKIP {s}")
        if len(skips) > 4:
            print(f"    ... {len(skips) - 4} more skips")

    print(f"\nkind={kind}  {'applied' if apply else 'dry run'}: "
          f"{total_edits} edit(s), {total_skips} skip(s)")
    if total_skips:
        print("Skips are expected for nested extents (rebuild and re-run to "
              "converge). Any 'expected X, found Y' skip means the log is stale.")
    if not apply:
        print("Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
