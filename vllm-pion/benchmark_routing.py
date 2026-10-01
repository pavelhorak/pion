#!/usr/bin/env python3
"""Pion Serve — Phase 3: Fleet Routing Benchmark.

Simulates a 4-GPU fleet serving coding assistant requests.
Each GPU specializes in a different domain (frontend, backend, ML, infra).
Measures routing accuracy, centroid drift, and cache affinity improvement.

Usage:
    python benchmark_routing.py                          # local routing (no Pion)
    python benchmark_routing.py --pion-host 127.0.0.1    # with Pion M13 router
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from vllm_pion.fleet_manager import FleetManager, FleetConfig, WorkerConfig
from vllm_pion.routing_policy import RoutingPolicy, Strategy
from vllm_pion.prompt_embedder import create_embedder


# ── Simulated GPU fleet ──────────────────────────────────────────────────────

FLEET = [
    WorkerConfig(
        worker_id="gpu-0",
        endpoint="http://gpu0:8080",
        capacity=16,
        tags={"description": "React TypeScript frontend components JSX CSS styling Next.js"},
    ),
    WorkerConfig(
        worker_id="gpu-1",
        endpoint="http://gpu1:8080",
        capacity=16,
        tags={"description": "Go HTTP handlers PostgreSQL database queries auth middleware backend API"},
    ),
    WorkerConfig(
        worker_id="gpu-2",
        endpoint="http://gpu2:8080",
        capacity=16,
        tags={"description": "Python PyTorch machine learning model training data pipeline neural network"},
    ),
    WorkerConfig(
        worker_id="gpu-3",
        endpoint="http://gpu3:8080",
        capacity=16,
        tags={"description": "Terraform Kubernetes Docker infrastructure deployment CI/CD monitoring"},
    ),
]

# Coding requests with expected routing target
REQUESTS = [
    # Frontend requests → gpu-0
    ("Fix the CSS styling on the login button component", "gpu-0"),
    ("Add a React hook for managing form state in TypeScript", "gpu-0"),
    ("Debug the Next.js SSR hydration mismatch in the dashboard", "gpu-0"),
    ("Create a new JSX component for the navigation sidebar", "gpu-0"),
    ("Fix responsive layout breakpoints in the user profile page", "gpu-0"),

    # Backend requests → gpu-1
    ("Fix the authentication middleware to handle expired tokens", "gpu-1"),
    ("Optimize the PostgreSQL query for fetching user orders", "gpu-1"),
    ("Add rate limiting to the Go HTTP API handler", "gpu-1"),
    ("Debug the database connection pool exhaustion issue", "gpu-1"),
    ("Implement the REST API endpoint for user preferences", "gpu-1"),

    # ML requests → gpu-2
    ("Fix the gradient explosion in the training loop", "gpu-2"),
    ("Add data augmentation to the image classification pipeline", "gpu-2"),
    ("Debug why the PyTorch model isn't converging on the validation set", "gpu-2"),
    ("Implement early stopping for the neural network training", "gpu-2"),
    ("Optimize the data loading pipeline for GPU training", "gpu-2"),

    # Infra requests → gpu-3
    ("Fix the Terraform module for the new VPC configuration", "gpu-3"),
    ("Debug why the Kubernetes pod keeps crash-looping", "gpu-3"),
    ("Add monitoring alerts for the production deployment", "gpu-3"),
    ("Create a Docker multi-stage build for the application", "gpu-3"),
    ("Fix the CI/CD pipeline failing on the integration tests", "gpu-3"),

    # Ambiguous / cross-domain requests
    ("Review the pull request for the new feature", ""),
    ("What are the best practices for code review?", ""),
    ("How do I set up the development environment?", ""),
]


def run_benchmark(fleet: FleetManager, strategy: str) -> list[dict]:
    """Run routing benchmark with the given strategy."""
    policy = RoutingPolicy(fleet, strategy=strategy)
    results = []

    for prompt, expected_gpu in REQUESTS:
        decision = policy.route(prompt=prompt)

        correct = False
        if expected_gpu:
            correct = decision.worker_id == expected_gpu
        else:
            correct = True  # ambiguous, any routing is acceptable

        results.append({
            "prompt": prompt[:60],
            "expected": expected_gpu or "(any)",
            "actual": decision.worker_id,
            "endpoint": decision.endpoint,
            "strategy": decision.strategy_used,
            "fallback": decision.fallback,
            "hit": decision.hit,
            "correct": correct,
            "scores": decision.scores,
        })

        # Simulate completion → centroid update
        if decision.hit:
            embedding = fleet._embedder.embed(prompt)
            fleet.report_completion(decision.worker_id, embedding)

    return results


def print_report(results: list[dict], fleet: FleetManager, strategy: str):
    """Print benchmark report."""
    print()
    print("=" * 75)
    print(f"FLEET ROUTING BENCHMARK — Strategy: {strategy}")
    print("=" * 75)

    # Accuracy
    domain_results = [r for r in results if r["expected"] != "(any)"]
    ambiguous_results = [r for r in results if r["expected"] == "(any)"]
    domain_correct = sum(1 for r in domain_results if r["correct"])
    total_hit = sum(1 for r in results if r["hit"])

    print(f"\nDomain routing accuracy: {domain_correct}/{len(domain_results)} "
          f"({domain_correct/len(domain_results)*100:.0f}%)" if domain_results else "")
    print(f"Total routed: {total_hit}/{len(results)}")
    print()

    # Per-domain breakdown
    domains = {"gpu-0": "Frontend", "gpu-1": "Backend", "gpu-2": "ML", "gpu-3": "Infra"}
    print(f"{'Domain':<12} {'Expected':>8} {'Correct':>8} {'Accuracy':>9}")
    print("-" * 40)
    for gpu_id, domain_name in domains.items():
        domain_reqs = [r for r in domain_results if r["expected"] == gpu_id]
        if domain_reqs:
            correct = sum(1 for r in domain_reqs if r["correct"])
            print(f"{domain_name:<12} {len(domain_reqs):>8} {correct:>8} {correct/len(domain_reqs)*100:>8.0f}%")
    print()

    # Detailed results
    print(f"{'#':>2} {'Prompt':<55} {'Expected':>8} {'Actual':>8} {'OK':>4}")
    print("-" * 80)
    for i, r in enumerate(results):
        ok = "Y" if r["correct"] else "N"
        print(f"{i:>2} {r['prompt']:<55} {r['expected']:>8} {r['actual']:>8} {ok:>4}")
    print()

    # Worker distribution
    print("Worker load distribution:")
    for wid, state in fleet.workers.items():
        domain = domains.get(wid, "?")
        print(f"  {wid} ({domain}): {state.total_routed} routed, "
              f"{state.total_completed} completed, "
              f"{state.active_requests} active")
    print()

    # Fleet stats
    stats = fleet.stats
    print(f"Fleet stats: {stats.total_routed} routed, {stats.route_hits} hits, "
          f"{stats.route_misses} misses, {stats.centroid_updates} centroid updates")

    # Centroid drift analysis
    print("\nCentroid drift (initial → current, cosine similarity):")
    embedder = fleet._embedder
    for wid, state in fleet.workers.items():
        if state.centroid is not None:
            initial = embedder.embed(state.config.tags.get("description", ""))
            drift = float(np.dot(initial, state.centroid))
            domain = domains.get(wid, "?")
            print(f"  {wid} ({domain}): {drift:.4f}")


def run_strategy_comparison(fleet_config: FleetConfig):
    """Compare all routing strategies side by side."""
    strategies = ["semantic_affinity", "round_robin", "least_loaded", "random"]
    all_results = {}

    for strategy in strategies:
        # Fresh fleet for each strategy (clean centroid state)
        fleet = FleetManager(fleet_config)
        fleet.start()
        for worker_config in FLEET:
            fleet.add_worker(worker_config)

        results = run_benchmark(fleet, strategy)
        all_results[strategy] = results

        domain_results = [r for r in results if r["expected"] != "(any)"]
        domain_correct = sum(1 for r in domain_results if r["correct"])
        accuracy = domain_correct / len(domain_results) * 100 if domain_results else 0

        print_report(results, fleet, strategy)
        fleet.stop()

    # Comparison summary
    print()
    print("=" * 75)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 75)
    print(f"\n{'Strategy':<22} {'Domain Accuracy':>16} {'Total Routed':>14}")
    print("-" * 55)
    for strategy in strategies:
        results = all_results[strategy]
        domain_results = [r for r in results if r["expected"] != "(any)"]
        domain_correct = sum(1 for r in domain_results if r["correct"])
        total_routed = sum(1 for r in results if r["hit"])
        accuracy = domain_correct / len(domain_results) * 100 if domain_results else 0
        print(f"{strategy:<22} {accuracy:>14.0f}% {total_routed:>14}")


def main():
    parser = argparse.ArgumentParser(description="Pion Serve — Fleet Routing Benchmark")
    parser.add_argument("--pion-host", type=str, default="127.0.0.1")
    parser.add_argument("--pion-port", type=int, default=1974)
    parser.add_argument("--use-pion", action="store_true",
                        help="Connect to Pion M13 router (default: local routing)")
    parser.add_argument("--strategy", type=str, default="all",
                        help="Strategy to benchmark: semantic_affinity, round_robin, least_loaded, random, all")
    parser.add_argument("--embed-provider", type=str, default="ngram")
    args = parser.parse_args()

    fleet_config = FleetConfig(
        pion_host=args.pion_host,
        pion_port=args.pion_port,
        use_pion=args.use_pion,
        embed_provider=args.embed_provider,
        embed_dim=1536,  # match ngram default
    )

    print("=" * 75)
    print("Pion Serve — Phase 3: Fleet Routing Benchmark")
    print("=" * 75)
    print(f"Fleet: {len(FLEET)} GPUs (Frontend, Backend, ML, Infra)")
    print(f"Requests: {len(REQUESTS)} ({len([r for r in REQUESTS if r[1]])} domain-specific, "
          f"{len([r for r in REQUESTS if not r[1]])} ambiguous)")
    print(f"Embedding: {args.embed_provider}")
    print(f"Pion M13: {'enabled' if args.use_pion else 'local routing'}")

    if args.strategy == "all":
        run_strategy_comparison(fleet_config)
    else:
        fleet = FleetManager(fleet_config)
        fleet.start()
        for worker_config in FLEET:
            fleet.add_worker(worker_config)

        results = run_benchmark(fleet, args.strategy)
        print_report(results, fleet, args.strategy)
        fleet.stop()


if __name__ == "__main__":
    sys.exit(main() or 0)
