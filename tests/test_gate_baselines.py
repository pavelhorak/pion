#!/usr/bin/env python3
"""Gate-harness regression tests. No server, no benchmark — pure logic.

Two things are under test:

1. **Floor neutrality.** The thresholds moved from hardcoded dicts in
   valkey-benchmark.py into benchmarks/gate_baselines.json. Every floor must
   survive that move byte-for-byte. Thresholds are never adjusted
   to make a run pass, and a refactor is the easiest place to move one
   by accident, so the pre-refactor tables are pinned here verbatim.

2. **A baseline row that was never measured must FAIL the gate.** The old code
   read `if actual > 0 and actual < min_rps`, so a row missing from the results
   dict silently passed: `--gate -t get,set` printed "all 20 commands meet
   baseline" having checked two.

Run: python3 tests/test_gate_baselines.py
"""

import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "vkbench", os.path.join(REPO, "benchmarks", "valkey-benchmark", "valkey-benchmark.py"))
assert _spec and _spec.loader, "cannot locate valkey-benchmark.py"
vkbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vkbench)

# ── The tables exactly as they were before the JSON move (commit 3947f47) ──────
PRE_REFACTOR_MAC_P10 = {
    "GET": 1_979_166, "SET": 2_065_217, "INCR": 1_979_166, "HSET": 2_065_217,
    "LPUSH": 2_021_276, "RPUSH": 1_979_166, "LPOP": 1_979_166, "RPOP": 1_979_166,
    "SADD": 1_532_258, "SPOP": 2_065_217, "ZADD": 2_021_276, "ZPOPMIN": 1_666_666,
    "MSET (10 keys)": 1_900_000, "PING_INLINE": 1_397_058, "PING_MBULK": 2_021_276,
    "XADD": 1_696_428,
    "LRANGE_100 (first 100 elements)": 251_322,
    "LRANGE_300 (first 300 elements)": 97_236,
    "LRANGE_500 (first 500 elements)": 53_429,
    "LRANGE_600 (first 600 elements)": 49_633,
}
# Ratcheted 2026-10-09 (#28, raise only): HSET LPUSH RPUSH ZADD ZPOPMIN MSET PING_*
# and a first XADD floor, from the median of 5 recorded rounds on an EPYC 8124P.
PRE_REFACTOR_LINUX_P10 = {
    "GET": 589_950, "SET": 698_250, "INCR": 575_700, "HSET": 562_130,
    "LPUSH": 708_955, "RPUSH": 646_258, "LPOP": 589_950, "RPOP": 582_350,
    "SADD": 561_450, "SPOP": 589_950, "ZADD": 703_703, "ZPOPMIN": 572_289,
    "MSET (10 keys)": 641_891, "PING_INLINE": 669_014, "PING_MBULK": 669_014,
    "XADD": 637_583,
    "LRANGE_100 (first 100 elements)": 167_200,
    "LRANGE_300 (first 300 elements)": 53_200,
    "LRANGE_500 (first 500 elements)": 31_350,
    "LRANGE_600 (first 600 elements)": 26_600,
}
PRE_REFACTOR_MAC_FALLBACK = {
    1: {"GET": 209_000, "SET": 209_000, "INCR": 209_000, "HSET": 209_000,
        "LPUSH": 209_000, "RPUSH": 209_000},
    100: {"GET": 8_550_000, "SET": 8_550_000, "INCR": 9_500_000, "HSET": 9_500_000,
          "LPUSH": 9_500_000, "RPUSH": 8_550_000},
    1000: {"GET": 9_500_000, "SET": 9_500_000, "INCR": 11_400_000, "HSET": 13_300_000,
           "LPUSH": 13_300_000, "RPUSH": 11_400_000},
}

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


def mins(profile, pipeline):
    t = vkbench.load_gate_baselines(profile, pipeline)
    return {c: s["min"] for c, s in t["commands"].items()}


print("=== Floor neutrality: JSON must reproduce the pre-refactor tables ===")
for label, profile, depth, expect in [
    ("mac P=10", "mac", 10, PRE_REFACTOR_MAC_P10),
    ("linux-epyc-8124p P=10", "linux-epyc-8124p", 10, PRE_REFACTOR_LINUX_P10),
    ("mac P=1", "mac", 1, PRE_REFACTOR_MAC_FALLBACK[1]),
    ("mac P=100", "mac", 100, PRE_REFACTOR_MAC_FALLBACK[100]),
    ("mac P=1000", "mac", 1000, PRE_REFACTOR_MAC_FALLBACK[1000]),
]:
    got = mins(profile, depth)
    check(f"{label}: same command set", set(got) == set(expect),
          f"only-in-json={sorted(set(got)-set(expect))} only-in-old={sorted(set(expect)-set(got))}")
    drift = {c: (expect[c], got[c]) for c in expect if c in got and got[c] != expect[c]}
    check(f"{label}: every floor unchanged", not drift, f"moved: {drift}")

# XADD got its first Linux floor on 2026-10-09 (#28); it must stay in the table.
check("linux table has XADD", "XADD" in mins("linux-epyc-8124p", 10))

print("\n=== Unlisted depth must error, not silently reuse another table ===")
try:
    vkbench.load_gate_baselines("mac", 50)
    check("mac P=50 raises GateConfigError", False, "returned a table instead of raising")
except vkbench.GateConfigError as e:
    check("mac P=50 raises GateConfigError", True)
    check("error names the depths that do exist", "P=10" in str(e), str(e))
try:
    vkbench.load_gate_baselines("linux-epyc-8124p", 1)
    check("linux P=1 raises GateConfigError", False, "returned a table instead of raising")
except vkbench.GateConfigError:
    check("linux P=1 raises GateConfigError", True)
try:
    vkbench.load_gate_baselines("nonesuch", 10)
    check("unknown profile raises GateConfigError", False)
except vkbench.GateConfigError:
    check("unknown profile raises GateConfigError", True)

print("\n=== Unmeasured rows fail the gate (the gh-#189-era hole) ===")
base = {"GET": 1_000_000, "SET": 1_000_000, "XADD": 500_000}

v = vkbench.evaluate_gate(base, {"GET": 1_100_000, "SET": 1_100_000, "XADD": 600_000})
check("all rows measured and above floor -> pass", not v["failures"] and v["checked"] == 3)

v = vkbench.evaluate_gate(base, {"GET": 900_000, "SET": 1_100_000, "XADD": 600_000})
check("a row below floor -> fail", len(v["failures"]) == 1 and "GET" in v["failures"][0])

# The regression: two of three rows measured, no --tests. Old code printed PASS.
v = vkbench.evaluate_gate(base, {"GET": 1_100_000, "SET": 1_100_000})
check("missing row without --tests -> fail", len(v["failures"]) == 1,
      f"failures={v['failures']}")
check("missing row is reported as NOT MEASURED",
      v["failures"] and "NOT MEASURED" in v["failures"][0], str(v["failures"]))
check("missing row is listed", v["missing"] == ["XADD"], str(v["missing"]))

# Same input, but the operator asked for a subset explicitly.
v = vkbench.evaluate_gate(base, {"GET": 1_100_000, "SET": 1_100_000}, tests_given=True)
check("missing row with --tests -> partial, not failure",
      not v["failures"] and v["missing"] == ["XADD"] and v["checked"] == 2)

# A subset run must still catch a real miss inside the subset.
v = vkbench.evaluate_gate(base, {"GET": 900_000}, tests_given=True)
check("--tests still fails a measured row below floor",
      len(v["failures"]) == 1 and "GET" in v["failures"][0], str(v["failures"]))

# Zero is a real measurement (a dead command), not an absence — the old
# `actual > 0` guard let exactly this through.
v = vkbench.evaluate_gate(base, {"GET": 0, "SET": 1_100_000, "XADD": 600_000})
check("a measured 0 RPS fails rather than being skipped",
      len(v["failures"]) == 1 and "GET" in v["failures"][0], str(v["failures"]))

print("\n=== Multi-run aggregation + ratchet-only floor proposal ===")
import statistics as _st

# Aggregation is median-of-kept, and warm-up discard must drop the low first run.
runs = [
    {"Pion": {"GET": 1_000_000}},   # warm-up, low — must not pull the median down
    {"Pion": {"GET": 2_000_000}},
    {"Pion": {"GET": 2_100_000}},
    {"Pion": {"GET": 2_050_000}},
]
kept = runs[1:]
med = _st.median([r["Pion"]["GET"] for r in kept])
check("median of kept runs ignores the discarded warm-up", med == 2_050_000, str(med))
check("warm-up would have moved it", _st.median([r["Pion"]["GET"] for r in runs]) != med)

# The ratchet rule: propose up, never down.
TOL = 0.95


def propose(floor_now, median):
    """Mirror of the rule in report_run_spread: ratchet up or hold, never lower."""
    proposed = int(median * TOL)
    if median < floor_now:
        return floor_now, "regression"
    if proposed > floor_now:
        return proposed, "ratchet"
    return floor_now, "keep"


f, v = propose(1_979_166, 2_400_000)
check("median well above floor -> ratchet up", v == "ratchet" and f == 2_280_000, f"{v} {f}")

f, v = propose(1_979_166, 2_083_333)
check("median at the original known-good -> holds (0.95x is below floor)",
      v == "keep" and f == 1_979_166, f"{v} {f}")

f, v = propose(1_979_166, 1_900_000)
check("median BELOW the floor -> flagged regression, floor unchanged",
      v == "regression" and f == 1_979_166, f"{v} {f}")

f, v = propose(1_979_166, 1_979_166)
check("median exactly at floor -> holds, never lowers",
      f == 1_979_166 and v in ("keep", "regression"), f"{v} {f}")

# The property that matters most: no input can ever produce a lower floor.
lowered = [m for m in range(100_000, 3_000_000, 50_000)
           if propose(1_979_166, m)[0] < 1_979_166]
check("no median value can lower a floor", not lowered, f"lowered at {lowered[:3]}")

print(f"\n{'='*60}")
print(f"Gate baseline tests: {passed} passed, {failed} failed")
print(f"{'='*60}")
sys.exit(1 if failed else 0)
