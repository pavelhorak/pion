#!/usr/bin/env python3
"""Pion Serve — Phase 1: Cache Hit Rate Validation.

Validates that semantic KV cache matching achieves significantly higher
hit rates than exact-prefix (SHA256) matching on realistic coding
assistant workloads.

Usage:
    # Real embeddings (requires OPENAI_API_KEY):
    python -m benchmarks.cache_hit_validation.run

    # Mock embeddings (no API key, for testing pipeline):
    python -m benchmarks.cache_hit_validation.run --mock

    # Custom thresholds:
    python -m benchmarks.cache_hit_validation.run --thresholds 0.85,0.90,0.95
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cache_hit_validation.workloads import generate_workload
from benchmarks.cache_hit_validation.simulator import run_simulation
from benchmarks.cache_hit_validation.reporter import generate_report


def main():
    parser = argparse.ArgumentParser(description="Pion Serve — Cache Hit Rate Validation")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for workload generation (default: 42)")
    parser.add_argument("--thresholds", type=str, default="0.80,0.85,0.88,0.90,0.92,0.95",
                        help="Comma-separated cosine similarity thresholds")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results.md in same directory)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock embeddings (no API key needed, for pipeline testing)")
    args = parser.parse_args()

    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    output_path = Path(args.output) if args.output else Path(__file__).parent / "results.md"

    print("=" * 60)
    print("Pion Serve — Phase 1: Cache Hit Rate Validation")
    print("=" * 60)
    print()

    # Step 1: Generate workload
    print("[1/3] Generating synthetic coding workload...")
    t0 = time.time()
    cache_entries, queries = generate_workload(seed=args.seed)
    print(f"  {len(cache_entries)} cache entries, {len(queries)} queries")
    print(f"  Categories: {', '.join(sorted(set(q.category for q in queries)))}")
    expected_hits = sum(1 for q in queries if q.expected_hit)
    print(f"  Expected hits: {expected_hits}/{len(queries)} ({expected_hits/len(queries)*100:.0f}%)")
    print(f"  Generated in {time.time()-t0:.1f}s")
    print()

    # Step 2: Run simulation
    print("[2/3] Running cache simulation...")
    if args.mock:
        print("  (Using MOCK embeddings — results are for pipeline validation only)")
    results = run_simulation(cache_entries, queries, thresholds=thresholds, use_mock=args.mock)
    print()

    # Step 3: Generate report
    print("[3/3] Generating report...")
    report = generate_report(results)
    output_path.write_text(report)
    print(f"  Full report: {output_path}")
    print()

    # Print key results to stdout
    print("=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    print()
    print(f"{'Threshold':>10s} | {'Exact Hit':>10s} | {'Semantic Hit':>12s} | {'Delta':>7s} | {'FP Rate':>8s} | {'FLOP Savings':>12s}")
    print(f"{'-'*10} | {'-'*10} | {'-'*12} | {'-'*7} | {'-'*8} | {'-'*12}")
    for t in thresholds:
        r = results.results.get(("overall", t))
        if not r:
            continue
        exact = f"{r.exact_hit_rate*100:.1f}%"
        semantic = f"{r.semantic_hit_rate*100:.1f}%"
        delta = f"+{(r.semantic_hit_rate - r.exact_hit_rate)*100:.0f}pp"
        fp = f"{r.fp_rate*100:.1f}%"
        flops = f"{results.semantic_flop_savings_pct.get(t, 0):.1f}%"
        print(f"{t:>10.2f} | {exact:>10s} | {semantic:>12s} | {delta:>7s} | {fp:>8s} | {flops:>12s}")
    print()

    # Validation gate check
    rec_t = 0.90 if 0.90 in thresholds else thresholds[len(thresholds) // 2]
    r_gate = results.results.get(("overall", rec_t))
    if r_gate:
        hit_pass = r_gate.semantic_hit_rate > 0.50
        fp_pass = r_gate.fp_rate < 0.10
        print(f"VALIDATION GATE (t={rec_t}):")
        print(f"  Hit rate >50%:     {r_gate.semantic_hit_rate*100:.1f}% — {'PASS' if hit_pass else 'FAIL'}")
        print(f"  FP rate <10%:      {r_gate.fp_rate*100:.1f}% — {'PASS' if fp_pass else 'FAIL'}")
        if hit_pass and fp_pass:
            print(f"\n  >>> PHASE 1 VALIDATED — proceed to Phase 2 (vLLM integration) <<<")
        elif not hit_pass:
            print(f"\n  >>> BELOW TARGET — investigate workload mix or lower threshold <<<")
        else:
            print(f"\n  >>> FP RATE TOO HIGH — raise threshold or improve embedding model <<<")

    return 0


if __name__ == "__main__":
    sys.exit(main())
