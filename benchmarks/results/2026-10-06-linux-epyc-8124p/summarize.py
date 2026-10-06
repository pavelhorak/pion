#!/usr/bin/env python3
"""Medians over repetitions from run_on_box.sh's memtier JSON files, as markdown.

Sharded Redis runs are one memtier per instance; their ops/sec are summed per
repetition (latency: the worst instance's p50/p99) before the median.
"""
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path


def totals(path: Path):
    d = json.loads(path.read_text())
    t = d["ALL STATS"]["Totals"]
    lat = t.get("Percentile Latencies", {})
    return t["Ops/sec"], lat.get("p50.00", float("nan")), lat.get("p99.00", float("nan"))


def main(out: str) -> None:
    raw = Path(out) / "raw"
    runs = collections.defaultdict(lambda: collections.defaultdict(list))   # config -> rep -> [(ops, p50, p99)]
    for f in sorted(raw.glob("*.json")):
        m = re.match(r"(.+)_r(\d+)(?:_i\d+)?$", f.stem)
        if not m:
            continue
        try:
            runs[m.group(1)][int(m.group(2))].append(totals(f))
        except (KeyError, ValueError) as e:
            print(f"<!-- unreadable {f.name}: {e} -->")
    print("| config | reps | ops/sec (median) | min–max | p50 ms | p99 ms |")
    print("|---|---:|---:|---|---:|---:|")
    for cfg in sorted(runs):
        per_rep = []
        for rep, parts in sorted(runs[cfg].items()):
            per_rep.append((sum(p[0] for p in parts), max(p[1] for p in parts), max(p[2] for p in parts)))
        ops = [r[0] for r in per_rep]
        print(f"| {cfg} | {len(per_rep)} | {st.median(ops):,.0f} | {min(ops):,.0f}–{max(ops):,.0f} | "
              f"{st.median(r[1] for r in per_rep):.3f} | {st.median(r[2] for r in per_rep):.3f} |")


if __name__ == "__main__":
    main(sys.argv[1])
