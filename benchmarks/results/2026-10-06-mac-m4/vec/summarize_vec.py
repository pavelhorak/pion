#!/usr/bin/env python3
"""Summarize the closed-vs-open vector runs in this directory.

Reads each vec_<mode>_<n>_<arm>.txt (a vectordb-benchmark.py log), takes the
final "Performance case got result" line, and prints per-run QPS, recall@100 and
serial P99, then per mode: each build's median QPS, and the open build's change
against the closed run it is paired with (runs ran closed, open, open, closed,
closed, open, so the pairs are runs 1-2, 4-3 and 5-6 in each mode's order).

    python3 summarize_vec.py > summary.md
"""
import glob
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = re.compile(r"Performance case got result: Metric\(.*?load_duration=([\d.]+), qps=([\d.]+), "
                    r"serial_latency_p99=np\.float64\(([\d.]+)\).*?recall=np\.float64\(([\d.]+)\)")
runs = {}
for path in glob.glob(os.path.join(HERE, "vec_*_*_*.txt")):
    m = re.match(r"vec_(\w+?)_(\d+)_(closed|open)\.txt$", os.path.basename(path))
    if not m:
        continue
    found = RESULT.findall(open(path, errors="replace").read())
    if found:
        load, qps, p99, recall = map(float, found[-1])
        runs[int(m.group(2))] = (m.group(1), m.group(3), qps, recall, p99, load)

print("| run | mode | build | QPS | recall@100 | serial P99 | load |")
print("|---:|---|---|---:|---:|---:|---:|")
for n in sorted(runs):
    mode, arm, qps, recall, p99, load = runs[n]
    print(f"| {n} | {mode} | {arm} | {qps:,.1f} | {recall:.4f} | {p99 * 1000:.1f} ms | {load:.1f} s |")

print("\n| mode | closed median QPS | open median QPS | open vs closed, per pair | median of the pairs |")
print("|---|---:|---:|---|---:|")
for mode in ("int8", "polarquant", "turboquant", "nanoquant"):
    rs = [(n, r) for n, r in sorted(runs.items()) if r[0] == mode]
    closed = [r[2] for _, r in rs if r[1] == "closed"]
    opened = [r[2] for _, r in rs if r[1] == "open"]
    gaps = [(o / c - 1) * 100 for c, o in zip(closed, opened)]
    print(f"| {mode} | {st.median(closed):,.1f} | {st.median(opened):,.1f} | "
          f"{', '.join(f'{g:+.1f}%' for g in gaps)} | {st.median(gaps):+.1f}% |")
