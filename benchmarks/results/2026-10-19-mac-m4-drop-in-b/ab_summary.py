#!/usr/bin/env python3
"""Summarize ab_dropin.sh output: per arm, the full run and the search-only runs.

Usage: python3 ab_summary.py <OUT dir>
Prints one row per run (arm, run, QPS at concurrency 1/5/10, max QPS, recall,
insert / optimize seconds) and, per arm kind, the median of the search-only max QPS,
then new/old and newp/old ratios of those medians.
"""
import glob
import json
import os
import re
import statistics
import sys


def result(rdir):
    files = sorted(glob.glob(os.path.join(rdir, "**", "result*.json"), recursive=True))
    if not files:
        return None
    d = json.load(open(files[-1]))
    return d["results"][0]["metrics"]


def main():
    out = sys.argv[1]
    arms = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)) and "_p" in d)
    print("| arm | run | QPS c1 / c5 / c10 | max QPS | recall | insert s | optimize s |")
    print("|---|---|---|---:|---:|---:|---:|")
    by_kind = {}
    for a in arms:
        kind = a.split("_p")[0]
        for run in ["full"] + sorted(r for r in os.listdir(os.path.join(out, a)) if re.fullmatch(r"s\d+", r)):
            m = result(os.path.join(out, a, run))
            if m is None:
                print(f"| {a} | {run} | (no result) | | | | |")
                continue
            conc = " / ".join(f"{q:,.0f}" for q in (m.get("conc_qps_list") or []))
            print(f"| {a} | {run} | {conc} | {m['qps']:,.0f} | {m['recall']:.4f} | "
                  f"{m.get('insert_duration', 0):.1f} | {m.get('optimize_duration', 0):.1f} |")
            if run != "full":
                by_kind.setdefault(kind, []).append(m["qps"])
        log = os.path.join(out, a, "build_log.txt")
        if os.path.exists(log):
            for line in open(log, errors="replace"):
                if "[vector] auto-build:" in line and "indexed in" in line:
                    print(f"|  | {a} log | {line.strip()[:90]} | | | | |")
    print()
    med = {k: statistics.median(v) for k, v in by_kind.items()}
    for k, v in sorted(by_kind.items()):
        print(f"{k}: search-only max QPS {', '.join(f'{x:,.0f}' for x in v)}; median {med[k]:,.0f}")
    if "old" in med:
        for k in ("new", "newp"):
            if k in med:
                print(f"{k}/old = {med[k] / med['old']:.3f}")


if __name__ == "__main__":
    main()
