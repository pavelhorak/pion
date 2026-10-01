#!/usr/bin/env python3
"""Find Mojo `tail call`s that receive a pointer into the caller's stack (gh #349).

    mojo build -I . src/main.mojo -D PION_HELD_VECTOR --emit llvm -o /tmp/pion.ll
    python3 tools/audit_tail_alloca.py /tmp/pion.ll [--all]

Why IR and not source: Mojo 1.0 marks a call `tail` when its pointer arguments
carry an untracked/external origin (`MutUntrackedOrigin`, `stack_allocation`).
In LLVM, `tail` promises the callee does not touch the caller's allocas, so at
-O3 the optimizer deletes the caller's stores into a stack buffer it then passes
(and forgets the callee's stores into one it passes out). Measured 2026-09-27:

    var s = stack_allocation[2, Int]();  s[0] = 40;  s[1] = 2
    consume(s)             # `tail call`, no stores emitted: returns garbage at -O3

Short Strings (<= 23 bytes, stored inline) and InlineArray locals hit it as
soon as their address is rebound to `MutUntrackedOrigin`. Whether a hit
actually misbehaves depends on whether LLVM later inlines the callee, which is
why the tree mostly works today and why that is not a property to rely on.

This reads the UNOPTIMIZED IR (`--emit llvm`), where `@always_inline` bodies are
already inlined by Mojo and every remaining call is one LLVM may keep out of
line. A hit is a call site Mojo told LLVM it may miscompile.

A stack address is followed through GEPs, casts, select/phi, ptrtoint and
integer arithmetic, and through a store to a slot and a load back from it
(the -O0 spill that hid container_free.deep_clone -> VectorSet.add, 2026-09-28).
Gated by tests/test_audit_tail_alloca.py, which also proves the audit still
FINDS each shape; still live on Mojo 1.1.0 and nightly (gh #384).
"""
import re
import sys
from collections import Counter

DEF_RE = re.compile(r'^define [^@]*@("(?:[^"\\]|\\.)*"|[\w.$]+)\(')
ALLOCA_RE = re.compile(r"^\s*(%[\w.]+) = alloca ")
# Any instruction deriving a value from other values: GEP, casts, select,
# phi (a short String's data pointer is `phi [heap_ptr, %string_alloca]`),
# and the ptrtoint/inttoptr/integer-arithmetic chain that carries an address
# as an `Int` (`UnsafePointer(unsafe_from_address=Int(p))` still gets `tail`).
PTRDEF_RE = re.compile(
    r"^\s*(%[\w.]+) = (getelementptr|bitcast|addrspacecast|select|phi \w+|"
    r"ptrtoint|inttoptr|add|sub|or|and|xor)\b(.*)$")
# A pointer spilled to a slot and loaded back (-O0 spills every local).
STORE_RE = re.compile(r"^\s*store (?:ptr|i64) (%[\w.]+), ptr (%[\w.]+)")
LOAD_RE = re.compile(r"^\s*(%[\w.]+) = load (?:ptr|i64), ptr (%[\w.]+)")
VAL_RE = re.compile(r"%[\w.]+")
TAIL_RE = re.compile(r'\btail call [^@]*@("(?:[^"\\]|\\.)*"|[\w.$]+)\((.*)\)')
ARG_RE = re.compile(r"(?:ptr|i64)(?: [a-z]+)* (%[\w.]+)")


def demangle(name):
    name = name.strip('"')
    return re.sub(r"\(.*", "", name)


def _flush(fn, allocas, edges, calls, hits):
    stack = set(allocas)
    changed = True
    while changed:                       # fixpoint: phis can name later values
        changed = False
        for dst, srcs in edges:
            if dst not in stack and any(v in stack for v in srcs):
                stack.add(dst)
                changed = True
    for callee, args in calls:
        if any(p in stack for p in ARG_RE.findall(args)):
            hits.append((fn, callee))


def scan(path):
    hits = []
    fn = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("define "):
                m = DEF_RE.match(line)
                fn = demangle(m.group(1)) if m else "?"
                allocas, edges, calls = [], [], []
                continue
            if fn is None:
                continue
            if line.startswith("}"):
                _flush(fn, allocas, edges, calls, hits)
                fn = None
                continue
            a = ALLOCA_RE.match(line)
            if a:
                allocas.append(a.group(1))
                continue
            st = STORE_RE.match(line)
            if st:
                edges.append(("slot:" + st.group(2), [st.group(1)]))
                continue
            ld = LOAD_RE.match(line)
            if ld:
                edges.append((ld.group(1), ["slot:" + ld.group(2)]))
                continue
            d = PTRDEF_RE.match(line)
            if d:
                # GEP's first operand after the type is the base; the others
                # are indices (never allocas), so taking all %values is safe.
                edges.append((d.group(1), VAL_RE.findall(d.group(3))))
                continue
            t = TAIL_RE.search(line)
            if t:
                callee = demangle(t.group(1))
                if "::" in callee:           # C / runtime symbols: buffer escapes to C
                    calls.append((callee, t.group(2)))
    return hits


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    hits = scan(sys.argv[1])
    pairs = Counter(hits)
    for (caller, callee), n in sorted(pairs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"{n:3d}x  {caller}\n       -> tail call {callee}")
    print(f"\n{len(hits)} tail call(s) pass a stack address to a Mojo function "
          f"({len(pairs)} distinct caller/callee pairs, "
          f"{len({c for _, c in pairs})} distinct callees)")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
