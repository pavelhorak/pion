#!/usr/bin/env python3
"""Per-thread CPU of each Pion run, from run_on_box.sh's /proc snapshots.

Each thread's (utime + stime) delta over the memtier run, in cores (the run's
length comes from settings.txt). Busiest first. With --io-threads N > 1 the
busiest thread is normally the executor (the worker thread) and the rest are
I/O threads, so the row says which side ran out first.

    python3 benchmarks/kv-io-threads/thread_cpu.py <OUT>/raw
"""
import os
import re
import sys
from pathlib import Path

HZ = os.sysconf("SC_CLK_TCK")


def snap(p: Path) -> dict:
    d = {}
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f) == 3:
            d[f[0]] = int(f[1]) + int(f[2])
    return d


def main(raw_dir: str) -> None:
    raw = Path(raw_dir)
    m = re.search(r"TEST_TIME=(\d+)", (raw.parent / "settings.txt").read_text())
    secs = int(m.group(1)) if m else 30
    print("| run | threads by CPU, cores (user + sys over the run) |")
    print("|---|---|")
    for b in sorted(raw.glob("*.threads_before")):
        a = b.with_name(b.name.replace("_before", "_after"))
        if not a.exists():
            continue
        x, y = snap(b), snap(a)
        use = sorted(((y[t] - x.get(t, 0)) / HZ / secs for t in y), reverse=True)
        cells = " · ".join(f"{u:.2f}" for u in use if u > 0.01)
        print(f"| {b.name.replace('.threads_before', '')} | {cells} |")


if __name__ == "__main__":
    main(sys.argv[1])
