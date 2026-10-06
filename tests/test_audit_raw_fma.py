#!/usr/bin/env python3
"""No `fma(` outside src/vector/fma_mad.mojo, and no `fmaf` in a vendored
vector library (#15, #25).

WHY
`fma(a, b, c)` rounds once. The release's Linux x86 build targets
x86-64-v2, which has no FMA instruction, so LLVM lowers each lane of a vector
`fma` to a libm `fmaf` call: in FT.SEARCH's metric_scores that was more time
than the search itself (#15), and 60 more call sites in the open distance,
PKM and speculative-RAG kernels, plus the closed library's PKM scoring
(#25, `fmaf` in its x86 MANIFEST), paid the same on every Linux x86 binary.
`fma_mad[w]` is `a * b + c` there and `fma` everywhere else.

The check runs over a scratch tree first (a planted raw `fma(` must be
reported, `fma_mad[...](` and prose must not), then over src/ and the
vendored MANIFESTs.

    python3 tests/test_audit_raw_fma.py
"""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"src/vector/fma_mad.mojo"}
RAW = re.compile(r"(?<![\w.\[`])fma\(")


def raw_fma(tree: Path, base: Path):
    hits = []
    for p in sorted(tree.rglob("*.mojo")):
        rel = p.relative_to(base).as_posix()
        if rel in ALLOWED:
            continue
        in_doc = False
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            quotes = line.count('"""')
            if in_doc or quotes:
                if quotes % 2 == 1:
                    in_doc = not in_doc
                continue
            code = line.split("#", 1)[0]
            if RAW.search(code):
                hits.append(f"{rel}:{n}: {line.strip()[:100]}")
    return hits


def main():
    fails = []
    with tempfile.TemporaryDirectory(prefix="fma_canary_") as d:
        t = Path(d)
        (t / "src").mkdir()
        (t / "src" / "bad.mojo").write_text("def f(a: Float32) -> Float32:\n    return fma(a, a, a)\n")
        (t / "src" / "good.mojo").write_text(
            '"""fma(a, b, c) in a docstring is prose."""\n'
            "def g(a: Float32) -> Float32:\n"
            "    # fma(a, a, a) in a comment is prose\n"
            "    return fma_mad[1](a, a, a)\n")
        hits = raw_fma(t / "src", t)
        print(f"[canary] planted 1 raw fma, found {len(hits)}")
        if len(hits) != 1 or "bad.mojo" not in hits[0]:
            fails.append("canary")
            print("\n".join(hits))
    hits = raw_fma(ROOT / "src", ROOT)
    print(f"[src]    raw fma outside {sorted(ALLOWED)}: {len(hits)}")
    if hits:
        fails.append("src")
        print("\n".join(hits))
    manifests = sorted((ROOT / "vendor" / "pion-vector").glob("*/MANIFEST"))
    bad = [m.relative_to(ROOT).as_posix() for m in manifests if re.search(r"^\s*fmaf\s*$", m.read_text(), re.M)]
    print(f"[vendor] MANIFESTs naming fmaf: {len(bad)} of {len(manifests)}")
    if bad or not manifests:
        fails.append("vendor")
        print("\n".join(bad) or "no MANIFEST found")
    print("PASS" if not fails else f"FAIL: {', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
