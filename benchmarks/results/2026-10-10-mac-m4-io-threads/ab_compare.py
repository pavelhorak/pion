"""Paired A/B of the KV gate harness: main vs branch, per row.

Each run is one markdown table; runs of the two binaries alternate (ABBA ABBA).
Per row: the median of each side, the branch/main ratio of medians, and how
many of the 4 adjacent pairs the branch won, lost or tied.
"""
import glob
import os
import re
import statistics as st
import sys

S = sys.argv[1]


def table(path):
    rows = {}
    for line in open(path):
        m = re.match(r"\| (.+?) \| [\d,]+ \| [\d,]+ \| [\d,]+ \| \*\*([\d,]+)\*\*", line)
        if m:
            rows[m.group(1)] = int(m.group(2).replace(",", ""))
    return rows


runs = {}
for f in glob.glob(os.path.join(S, "*.md")):
    tag, i = re.match(r"(main|branch)_(\d+)\.md", os.path.basename(f)).groups()
    runs[(tag, int(i))] = table(f)
idx = sorted({i for _, i in runs})
names = list(runs[("main", idx[0])].keys())
print(f"runs: {len(idx)} pairs\n")
print(f"| row | main median | branch median | branch/main | pairs won/lost/tied |")
print(f"|---|---:|---:|---:|---|")
ratios = []
for n in names:
    a = [runs[("main", i)][n] for i in idx if ("main", i) in runs]
    b = [runs[("branch", i)][n] for i in idx if ("branch", i) in runs]
    w = sum(1 for i in idx if runs[("branch", i)][n] > runs[("main", i)][n])
    l = sum(1 for i in idx if runs[("branch", i)][n] < runs[("main", i)][n])
    t = len(idx) - w - l
    r = st.median(b) / st.median(a)
    ratios.append(r)
    print(f"| {n} | {st.median(a):,.0f} | {st.median(b):,.0f} | {r:.3f} | {w}/{l}/{t} |")
print(f"\ngeometric mean of branch/main over {len(ratios)} rows: "
      f"{st.geometric_mean(ratios):.4f}")
