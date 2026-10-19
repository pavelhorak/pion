#!/usr/bin/env python3
"""The S50 table for phase B: per run, recall and max QPS at each stage (50%, 80%, write
finished, after optimize()), builds the server ran, and per-worker recall after the run
(post_check.py). Same columns as phase A's (benchmarks/results/2026-10-19-mac-m4-drop-in).
Usage: python3 s50_summary.py raw/s50/<series>"""
import glob, json, os, re, sys

root = sys.argv[1]
print("| run | 50% | 80% | write finished | after optimize() | max QPS by stage | builds | per-worker recall after |")
print("|---|---:|---:|---:|---:|---|---:|---|")
for d in sorted(os.listdir(root)):
    fs = glob.glob(os.path.join(root, d, "**", "result_*.json"), recursive=True) \
        or glob.glob(os.path.join(root, d, "result.json"))
    if not fs:
        continue
    m = json.load(open(fs[0]))["results"][0]["metrics"]
    rec = m["st_recall_list"]
    qps = " / ".join(f"{x:,.0f}" for x in m["st_max_qps_list_list"])
    log = os.path.join(root, d, "server_log.txt")
    if not os.path.exists(log):
        log = os.path.join(root, d, "build_log.txt")
    builds = sum(1 for line in open(log, errors="replace") if "indexed in" in line) if os.path.exists(log) else -1
    pc = os.path.join(root, d, "post_check.txt")
    wr = re.findall(r"recall@100 ([0-9.]+)", open(pc).read()) if os.path.exists(pc) else []
    print(f"| {d} | " + " | ".join(f"{x:.4f}" for x in rec) + f" | {qps} | {builds} | {' '.join(wr)} |")
