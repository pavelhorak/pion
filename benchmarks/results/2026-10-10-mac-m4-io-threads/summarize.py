"""Medians of the Mac io-threads sweep: Pion --io-threads N vs Redis io-threads N."""
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

raw = Path(sys.argv[1]) / "raw"
ops = collections.defaultdict(list)
p99 = collections.defaultdict(list)
for f in raw.glob("*.json"):
    m = re.match(r"(pion|redis)_io(\d+)_P(\d+)_r(\d+)$", f.stem)
    if not m:
        continue
    t = json.loads(f.read_text())["ALL STATS"]["Totals"]
    key = (m.group(1), int(m.group(2)), int(m.group(3)))
    ops[key].append(t["Ops/sec"])
    p99[key].append(t.get("Percentile Latencies", {}).get("p99.00", float("nan")))

print("| P | threads | Redis ops/sec | Pion ops/sec | Pion ÷ Redis | Pion ÷ Pion io1 | Redis p99 ms | Pion p99 ms | runs |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for P in (1, 10, 50):
    base = st.median(ops[("pion", 1, P)]) if ops[("pion", 1, P)] else float("nan")
    for io in (1, 2, 4):
        r, p = ops[("redis", io, P)], ops[("pion", io, P)]
        if not r or not p:
            continue
        mr, mp = st.median(r), st.median(p)
        print(f"| {P} | {io} | {mr:,.0f} | {mp:,.0f} | {mp / mr:.2f} | {mp / base:.2f} | "
              f"{st.median(p99[('redis', io, P)]):.3f} | {st.median(p99[('pion', io, P)]):.3f} | "
              f"{len(r)}/{len(p)} |")
