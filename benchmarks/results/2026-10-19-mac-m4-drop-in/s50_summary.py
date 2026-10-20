#!/usr/bin/env python3
"""The S50 table: per run, recall and max QPS at each stage (50%, 80%, write
finished, after optimize()), and how many automatic builds the server ran.
Usage: python3 s50_summary.py raw/s50"""
import glob, json, os, sys

root = sys.argv[1]
print("| run | 50% | 80% | write finished | after optimize() | max QPS by stage | builds |")
print("|---|---:|---:|---:|---:|---|---:|")
for d in sorted(os.listdir(root)):
    f = os.path.join(root, d, "result.json")
    if not os.path.exists(f):
        continue
    m = json.load(open(f))["results"][0]["metrics"]
    rec = m["st_recall_list"]
    qps = " / ".join(f"{x:,.0f}" for x in m["st_max_qps_list_list"])
    builds = sum(1 for line in open(os.path.join(root, d, "build_log.txt")) if "indexed in" in line)
    print(f"| {d} | " + " | ".join(f"{x:.4f}" for x in rec) + f" | {qps} | {builds} |")
