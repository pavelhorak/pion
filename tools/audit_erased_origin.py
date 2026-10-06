#!/usr/bin/env python3
"""Find a local buffer read through a pointer that does not keep it alive.

Mojo destroys a value right after its LAST USE, and a call's arguments are
evaluated before the call runs. A pointer from `x.unsafe_ptr()` carries x's
origin, so while that pointer is in use x stays alive. Cast it through `Int`
(`Pointer[...](unsafe_from_address=Int(x.unsafe_ptr()))`) or `rebind` it and
the origin is gone: if `len(x)` in the same argument list is x's last use, x
is destroyed BEFORE the callee reads the pointer.

    f(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr())), len(x))

reads freed memory. It shipped (#40): consumer names replayed from the WAL
came back as allocator garbage, and XDEL's stream-metadata record was written
from a freed buffer, so a restart read a nonsense last-generated-id.

The rule this audit checks, for each such cast of a local `x` (a parameter or
a field lives for the whole call, a StaticString forever):
  * if the cast is bound to a variable (`var p = ...`), x must be named again
    AFTER the last statement that names p;
  * otherwise x must be named again after the statement holding the cast.
A keep-alive `_ = x^` placed after the last use of the pointer satisfies it;
so does passing the buffer itself (a List or String argument stays alive for
the whole call) instead of a raw pointer to it.

A later use counts only at the block level of everything between it and
the pointer's last use: one inside a later `if` or in a sibling `else` does
not keep x alive on every path, and a branch whose last use of x is the
pointer read is exactly how the WAL replay bug looked.

    python3 tools/audit_erased_origin.py [--src src] [-v]
"""
import argparse
import os
import re
import sys

CAST = re.compile(r"(?:\bInt\(\s*|\brebind\[[^\]]*\]\]?\(\s*)(?<![\w.])(\w+)\.unsafe_ptr\(\)")
DECL = re.compile(r"\bvar\s+(\w+)\s*(:\s*([\w\[\], ]+))?\s*=")
STATIC_TYPES = ("StaticString", "StringLiteral")


def statements(body):
    """(first, last) line index of each statement, by bracket balance."""
    out, depth, s0, open_ = [], 0, 0, False
    for i, l in enumerate(body):
        if not open_:
            s0 = i
        depth += l.count("(") + l.count("[") - l.count(")") - l.count("]")
        open_ = depth > 0 or l.rstrip().endswith("\\")
        if not open_:
            out.append((s0, i))
            depth = 0
    if open_:
        out.append((s0, len(body) - 1))
    return out


def audit_file(path):
    found = []
    lines = open(path, encoding="utf-8").read().split("\n")
    starts = [i for i, l in enumerate(lines) if re.match(r"\s*(def|fn)\s", l)] + [len(lines)]
    for a, b in zip(starts, starts[1:]):
        body = [re.sub(r"#.*$", "", l) for l in lines[a:b]]
        text = "\n".join(body)
        locals_ = {}
        for m in DECL.finditer(text):
            locals_.setdefault(m.group(1), (m.group(3) or "").strip())
        st = statements(body)
        for (s, e) in st:
            stmt = "\n".join(body[s:e + 1])
            for m in CAST.finditer(stmt):
                x = m.group(1)
                if x not in locals_ or locals_[x].startswith(STATIC_TYPES):
                    continue
                end = e
                # bound to a variable only when the right-hand side IS the
                # pointer (not a call that takes it: `var f = g(ptr, n)`)
                bind = re.match(r"\s*var\s+(\w+)\s*(:[^=]*)?=\s*(Pointer\[|UnsafePointer\[|Int\(|rebind\[)", body[s])
                if bind and bind.group(1) != x:
                    p = bind.group(1)
                    for (s2, e2) in st:
                        if s2 > e and re.search(r"\b%s\b" % re.escape(p), "\n".join(body[s2:e2 + 1])):
                            end = e2
                # `x = f(<pointer to x>)`: the old x dies at the assignment, a
                # later `x` is the new value
                reassigned = re.match(r"\s*%s\s*=[^=]" % re.escape(x), body[s]) is not None
                if reassigned or not _kept_alive(body, st, end, x):
                    found.append((path, a + s + 1, x, stmt.strip().split("\n")[0][:120]))
    return found


def _indent(l):
    return len(l) - len(l.lstrip())


def _kept_alive(body, st, end, x):
    """Is x named after line `end` on every path from it? A use counts when it
    sits at the block level of everything between (not inside a later `if`
    or loop, not in a sibling `elif`/`else` clause)."""
    first = max(s for (s, e) in st if s <= end)
    level = _indent(body[first])
    sibling = False
    k = end + 1
    while k < len(body):
        l = body[k]
        if not l.strip():
            k += 1
            continue
        ind = _indent(l)
        if ind < level:
            level = ind
            sibling = bool(re.match(r"\s*(elif\b|else\b|except\b|finally\b)", l))
        if not sibling and ind <= level and re.search(r"\b%s\b" % re.escape(x), l):
            return True
        k += 1
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    found = []
    for dp, _, files in os.walk(args.src):
        for f in sorted(files):
            if f.endswith(".mojo"):
                found += audit_file(os.path.join(dp, f))
    for path, line, x, stmt in found:
        print(f"ERASED ORIGIN {os.path.relpath(path)}:{line}: `{x}` may be destroyed before this reads it: {stmt}")
    print(f"{len(found)} site(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
