#!/usr/bin/env python3
"""memtier_benchmark harness — fair comparison of Redis, Valkey, Dragonfly, and Pion.

Runs memtier_benchmark with identical parameters against each server, collects
throughput (ops/sec), latency (p50/p99/p99.9), and reports side-by-side results.

Usage:
    python3 benchmarks/memtier-benchmark/memtier-benchmark.py                    # all 4 servers
    python3 benchmarks/memtier-benchmark/memtier-benchmark.py --pion-only        # Pion only
    python3 benchmarks/memtier-benchmark/memtier-benchmark.py --test-time 30     # longer run
    python3 benchmarks/memtier-benchmark/memtier-benchmark.py --pipeline 10      # pipelined
"""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from bench_proc import kill_pion_servers, kill_inference_workers  # gh #347
RESULTS_MD = os.path.join(SCRIPT_DIR, "benchmark_results.md")

# ── Server configs ────────────────────────────────────────────────────────────

SERVERS = {
    "redis": {
        "name": "Redis",
        "port": 6379,
        "cmd": None,  # auto-detect
        "args": ["--save", "", "--appendonly", "no", "--loglevel", "warning",
                 "--io-threads", str(max(1, __import__('os').cpu_count() - 1)),
                 "--dir", "/tmp/pion-bench-redis"],
    },
    "valkey": {
        "name": "Valkey",
        "port": 6380,
        "cmd": None,
        "args": ["--save", "", "--appendonly", "no", "--loglevel", "warning",
                 "--dir", "/tmp/pion-bench-valkey"],
    },
    "dragonfly": {
        "name": "Dragonfly",
        "port": 6381,
        "cmd": None,
        "args": ["--logtostderr", "--cache_mode", "--maxmemory", "8gb",
                 "--dir", "/tmp/pion-bench-dragonfly", "--dbfilename", ""],
    },
    "pion": {
        "name": "Pion",
        "port": 1974,
        "cmd": None,
        "args": ["--no-auto-detect", "--no-auto-embed"],
    },
}


def _clean_pion_state():
    """Remove stale Pion mmap state files (WAL/HNSW/snapshot).

    Mirrors the hardened version in vectordb-benchmark.py: kill stragglers
    first, retry once if the unlink races a still-mmap'd file, and log loud
    warnings rather than silently swallowing errors. A silent swallow here
    previously left WAL data that polluted vector recall.
    """
    import glob as _glob
    # By executable name, not cmdline: a cmdline match also kills a running
    # `mojo build ... -o pion-server` (gh #347).
    kill_pion_servers()
    kill_inference_workers()
    time.sleep(0.5)

    def _try_remove():
        leftover = []
        for pattern in ["pion.hnsw.*", "pion.wal.*", "pion.snapshot.*", "pion.blob.*"]:
            for f in _glob.glob(os.path.join(PROJECT_ROOT, pattern)):
                try:
                    os.remove(f)
                except OSError as e:
                    leftover.append((f, e))
        return leftover

    leftover = _try_remove()
    if leftover:
        time.sleep(1.0)
        leftover = _try_remove()
    for f, e in leftover:
        print(f"  [clean] WARNING: could not remove {os.path.basename(f)}: {e}")


def find_binary(name, search_paths):
    """Find a server binary."""
    path = shutil.which(name)
    if path:
        return path
    for p in search_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def detect_servers(args):
    """Auto-detect server binaries."""
    parent = os.path.dirname(PROJECT_ROOT)

    SERVERS["redis"]["cmd"] = find_binary("redis-server", [
        "/opt/homebrew/bin/redis-server",
        "/usr/bin/redis-server",
        "/opt/redis-8.0/src/redis-server",
    ])

    SERVERS["valkey"]["cmd"] = find_binary("valkey-server", [
        os.path.join(parent, "valkey/src/valkey-server"),
        "/opt/homebrew/bin/valkey-server",
    ])

    SERVERS["dragonfly"]["cmd"] = find_binary("dragonfly", [
        os.path.join(parent, "dragonfly/build-opt/dragonfly"),
        os.path.join(parent, "dragonfly/build-release/dragonfly"),
        "/opt/homebrew/bin/dragonfly",
    ])

    SERVERS["pion"]["cmd"] = os.path.join(PROJECT_ROOT, "pion-server")
    if not os.path.isfile(SERVERS["pion"]["cmd"]):
        SERVERS["pion"]["cmd"] = None

    # Apply CLI overrides
    if args.redis_bin:
        SERVERS["redis"]["cmd"] = args.redis_bin
    if args.valkey_bin:
        SERVERS["valkey"]["cmd"] = args.valkey_bin
    if args.dragonfly_bin:
        SERVERS["dragonfly"]["cmd"] = args.dragonfly_bin
    if args.pion_bin:
        SERVERS["pion"]["cmd"] = args.pion_bin


# ── Server lifecycle ──────────────────────────────────────────────────────────

def wait_for_port(port, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def start_server(key, workers=1, startup_wait=5):
    cfg = SERVERS[key]
    if not cfg["cmd"]:
        return None

    cmd = [cfg["cmd"]]
    port = cfg["port"]

    if key == "pion":
        cmd += ["-p", str(port), "-w", str(workers)] + cfg["args"]
        # gh #253: -w N > 1 refuses to start without this. memtier generates
        # independent random keys per connection and never reads another
        # connection's writes, so a split keyspace does not change the numbers.
        if workers > 1:
            cmd += ["--independent-workers"]
    elif key == "dragonfly":
        cmd += ["--port", str(port)] + cfg["args"]
    else:
        cmd += ["--port", str(port)] + cfg["args"]

    print(f"  Starting {cfg['name']}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    if proc.poll() is not None:
        print(f"  ERROR: {cfg['name']} exited immediately (code {proc.returncode})")
        return None

    if not wait_for_port(port, timeout=startup_wait):
        print(f"  ERROR: {cfg['name']} not responding on port {port}")
        proc.kill()
        return None

    # Flush any stale data
    try:
        subprocess.run(["redis-cli", "-p", str(port), "FLUSHALL"],
                       capture_output=True, timeout=5)
    except Exception:
        pass

    print(f"  {cfg['name']} ready on port {port}")
    return proc


def stop_server(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


# ── memtier_benchmark runner ─────────────────────────────────────────────────

def run_memtier(port, threads, clients, test_time, pipeline, ratio, data_size,
                key_pattern="R:R", key_max=1000000, extra_args=None):
    """Run memtier_benchmark and parse results."""
    cmd = [
        "memtier_benchmark",
        "-s", "127.0.0.1",
        "-p", str(port),
        "-t", str(threads),
        "-c", str(clients),
        "--test-time", str(test_time),
        "--ratio", ratio,
        "-d", str(data_size),
        "--key-pattern", key_pattern,
        "--key-maximum", str(key_max),
        "--distinct-client-seed",
        "--hide-histogram",
        "--json-out-file", "/tmp/memtier_result.json",
    ]
    if pipeline > 1:
        cmd += ["--pipeline", str(pipeline)]
    if extra_args:
        cmd += extra_args

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=test_time + 60)
    output = result.stdout + "\n" + result.stderr

    # Parse JSON output
    try:
        with open("/tmp/memtier_result.json") as f:
            data = json.load(f)
        return parse_memtier_json(data)
    except Exception:
        # Fallback: parse text output
        return parse_memtier_text(output)


def parse_memtier_json(data):
    """Parse memtier JSON output."""
    results = {}
    for json_key, result_key in [("Totals", "total"), ("Sets", "sets"), ("Gets", "gets")]:
        op_data = data.get("ALL STATS", {}).get(json_key, {})
        if not op_data:
            continue
        pct = op_data.get("Percentile Latencies", {})
        results[result_key] = {
            "ops_sec": op_data.get("Ops/sec", 0),
            "avg_lat": op_data.get("Latency", 0),
            "p50": pct.get("p50.00", 0),
            "p99": pct.get("p99.00", 0),
            "p999": pct.get("p99.90", 0),
            "kb_sec": op_data.get("KB/sec", 0),
        }
    return results


def parse_memtier_text(output):
    """Fallback: parse memtier text output."""
    results = {}
    for line in output.split("\n"):
        line = line.strip()
        # Match: Totals    123456.78    1.234    2.345    3.456
        m = re.match(r"(Totals|Sets|Gets)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            key = m.group(1).lower()
            results[key] = {
                "ops_sec": float(m.group(2)),
                "p50": float(m.group(5)),
                "p99": float(m.group(6)),
                "p999": float(m.group(7)),
            }
    return results


# ── Benchmark profiles ────────────────────────────────────────────────────────

PROFILES = {
    "throughput": {
        "description": "Max throughput (SET:GET 1:10, 256B values)",
        "threads": 4,
        "clients": 50,
        "test_time": 15,
        "pipeline": 1,
        "ratio": "1:10",
        "data_size": 256,
    },
    "pipeline": {
        "description": "Pipelined throughput (P=10, SET:GET 1:10, 256B)",
        "threads": 4,
        "clients": 50,
        "test_time": 15,
        "pipeline": 10,
        "ratio": "1:10",
        "data_size": 256,
    },
    "pipeline_deep": {
        "description": "Deep pipeline (P=30, SET:GET 1:10, 256B)",
        "threads": 4,
        "clients": 30,
        "test_time": 15,
        "pipeline": 30,
        "ratio": "1:10",
        "data_size": 256,
    },
    "latency": {
        "description": "Latency focused (1 thread, 1 client, P=1, SET:GET 1:1)",
        "threads": 1,
        "clients": 1,
        "test_time": 10,
        "pipeline": 1,
        "ratio": "1:1",
        "data_size": 256,
    },
    "write_heavy": {
        "description": "Write heavy (SET:GET 1:0, 256B)",
        "threads": 4,
        "clients": 50,
        "test_time": 15,
        "pipeline": 1,
        "ratio": "1:0",
        "data_size": 256,
    },
    "large_values": {
        "description": "Large values (SET:GET 1:10, 4KB)",
        "threads": 4,
        "clients": 50,
        "test_time": 15,
        "pipeline": 1,
        "ratio": "1:10",
        "data_size": 4096,
    },
    "high_concurrency": {
        "description": "High concurrency (200 clients, P=10, 256B)",
        "threads": 4,
        "clients": 200,
        "test_time": 15,
        "pipeline": 10,
        "ratio": "1:10",
        "data_size": 256,
    },
}


# ── Main ──────────────────────────────────────────────────────────────────────

def format_ops(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))


def main():
    parser = argparse.ArgumentParser(description="memtier_benchmark harness — Pion vs Redis vs Valkey vs Dragonfly")
    parser.add_argument("--pion-only", action="store_true", help="Benchmark Pion only")
    parser.add_argument("--profiles", type=str, default="throughput,pipeline,latency",
                        help="Comma-separated profiles to run (default: throughput,pipeline,latency)")
    parser.add_argument("--all-profiles", action="store_true", help="Run all profiles")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Pion worker count (default: 1)")
    parser.add_argument("--test-time", type=int, default=None, help="Override test duration (seconds)")
    parser.add_argument("--pipeline", type=int, default=None, help="Override pipeline depth")
    parser.add_argument("--threads", type=int, default=None, help="Override memtier threads")
    parser.add_argument("--clients", type=int, default=None, help="Override clients per thread")
    parser.add_argument("--data-size", type=int, default=None, help="Override value size in bytes")
    parser.add_argument("--redis-bin", type=str, default=None)
    parser.add_argument("--valkey-bin", type=str, default=None)
    parser.add_argument("--dragonfly-bin", type=str, default=None)
    parser.add_argument("--pion-bin", type=str, default=None)
    parser.add_argument("--gate", action="store_true",
                        help="Gate mode: exit 1 if Pion ops/sec below baseline for pipeline depth")
    parser.add_argument("--gate-profile", choices=["mac", "linux-epyc-8124p"], default="mac",
                        help="Which CPU-class baseline to enforce. mac=Apple Silicon "
                             "-w 1 (throughput P=1 ≥ 330K, pipeline P=10 ≥ 2.4M, default); "
                             "linux-epyc-8124p=EPYC 8124P @ 2.45 GHz Zen 4c Siena -w 32 "
                             "(throughput P=1 ≥ 252K, pipeline P=10 ≥ 1.95M). "
                             "See gh #56.")
    args = parser.parse_args()
    # gh #305: a contaminated box reads exactly like a regression. Refuse to
    # gate on one; transient causes (load, indexing, a CI job) get 3 min to clear.
    if args.gate:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from preflight import preflight
        if not preflight(wait=180):
            sys.exit(1)

    detect_servers(args)

    # Determine which servers to test
    if args.pion_only:
        server_keys = ["pion"]
    else:
        server_keys = ["redis", "valkey", "dragonfly", "pion"]

    # Filter to available servers
    available = []
    for k in server_keys:
        if SERVERS[k]["cmd"]:
            available.append(k)
        else:
            print(f"  SKIP {SERVERS[k]['name']}: binary not found")
    server_keys = available

    if not server_keys:
        print("ERROR: No servers found. Install redis-server, valkey-server, dragonfly, or build pion-server.")
        sys.exit(1)

    # Determine profiles
    if args.all_profiles:
        profile_names = list(PROFILES.keys())
    else:
        profile_names = [p.strip() for p in args.profiles.split(",")]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_results = {}  # {profile: {server: results}}

    print(f"\n{'='*70}")
    print(f"  memtier_benchmark — {timestamp}")
    print(f"  Servers: {', '.join(SERVERS[k]['name'] for k in server_keys)}")
    print(f"  Profiles: {', '.join(profile_names)}")
    print(f"  Pion workers: {args.workers}")
    print(f"{'='*70}\n")

    for profile_name in profile_names:
        if profile_name not in PROFILES:
            print(f"  Unknown profile: {profile_name}")
            continue

        profile = PROFILES[profile_name].copy()

        # Apply CLI overrides
        if args.test_time is not None:
            profile["test_time"] = args.test_time
        if args.pipeline is not None:
            profile["pipeline"] = args.pipeline
        if args.threads is not None:
            profile["threads"] = args.threads
        if args.clients is not None:
            profile["clients"] = args.clients
        if args.data_size is not None:
            profile["data_size"] = args.data_size

        print(f"{'='*70}")
        print(f"  Profile: {profile_name} — {profile['description']}")
        print(f"  threads={profile['threads']} clients={profile['clients']} "
              f"pipeline={profile['pipeline']} ratio={profile['ratio']} "
              f"data_size={profile['data_size']}B test_time={profile['test_time']}s")
        print(f"{'='*70}")

        profile_results = {}
        for key in server_keys:
            cfg = SERVERS[key]
            print(f"\n  --- {cfg['name']} ---")

            # Clean up any leftover
            if key == "pion":
                kill_pion_servers()   # gh #347: never a cmdline match
            else:
                # Armored: redis rewrites its title, so -x cannot match it,
                # and a bare -f pattern matches the shell running pkill.
                name = os.path.basename(cfg["cmd"])
                subprocess.run(["pkill", "-9", "-f", name[:-1] + "[" + name[-1] + "]"],
                               capture_output=True)
            time.sleep(1)

            # Clean WAL/HNSW/snapshot for Pion (defensive: silent swallow here
            # was the root cause of a recall regression in vectordb-benchmark;
            # same lesson applies here).
            if key == "pion":
                _clean_pion_state()

            startup_wait = 15 if key == "pion" else 5
            proc = start_server(key, workers=args.workers, startup_wait=startup_wait)
            if not proc:
                print(f"  SKIP {cfg['name']}: failed to start")
                continue

            try:
                results = run_memtier(
                    port=cfg["port"],
                    threads=profile["threads"],
                    clients=profile["clients"],
                    test_time=profile["test_time"],
                    pipeline=profile["pipeline"],
                    ratio=profile["ratio"],
                    data_size=profile["data_size"],
                )
                profile_results[key] = results

                if "total" in results:
                    t = results["total"]
                    print(f"  Total: {format_ops(t['ops_sec'])} ops/sec, "
                          f"p50={t.get('p50',0):.3f}ms, p99={t.get('p99',0):.3f}ms")
                if "sets" in results:
                    s = results["sets"]
                    print(f"  SET:   {format_ops(s['ops_sec'])} ops/sec, p99={s.get('p99',0):.3f}ms")
                if "gets" in results:
                    g = results["gets"]
                    print(f"  GET:   {format_ops(g['ops_sec'])} ops/sec, p99={g.get('p99',0):.3f}ms")
            except Exception as e:
                print(f"  ERROR running memtier: {e}")
            finally:
                stop_server(proc)
                time.sleep(1)
                # Clean WAL/HNSW/snapshot for Pion — leave a clean state for
                # the next run (the next caller may not clean before starting).
                if key == "pion":
                    _clean_pion_state()

        all_results[profile_name] = profile_results

    # ── Generate report ───────────────────────────────────────────────────────

    print(f"\n\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}\n")

    md_lines = []
    md_lines.append(f"### memtier_benchmark Run: {timestamp}")
    md_lines.append(f"**Pion workers:** {args.workers}\n")

    for profile_name, profile_results in all_results.items():
        profile = PROFILES[profile_name]
        md_lines.append(f"#### {profile_name}: {profile['description']}")
        md_lines.append(f"t={profile['threads']} c={profile['clients']} "
                        f"P={profile['pipeline']} ratio={profile['ratio']} "
                        f"d={profile['data_size']}B time={profile['test_time']}s\n")

        # Total ops/sec table
        md_lines.append("| Metric | " + " | ".join(
            SERVERS[k]["name"] for k in profile_results) + " |")
        md_lines.append("| :--- | " + " | ".join(
            ":---:" for _ in profile_results) + " |")

        # Collect metrics
        metrics = [
            ("Total ops/sec", "total", "ops_sec", format_ops),
            ("SET ops/sec", "sets", "ops_sec", format_ops),
            ("GET ops/sec", "gets", "ops_sec", format_ops),
            ("Total p50 (ms)", "total", "p50", lambda x: f"{x:.3f}"),
            ("Total p99 (ms)", "total", "p99", lambda x: f"{x:.3f}"),
            ("Total p99.9 (ms)", "total", "p999", lambda x: f"{x:.3f}"),
            ("SET p99 (ms)", "sets", "p99", lambda x: f"{x:.3f}"),
            ("GET p99 (ms)", "gets", "p99", lambda x: f"{x:.3f}"),
        ]

        for label, op_key, metric_key, fmt in metrics:
            vals = []
            raw_vals = []
            for k in profile_results:
                v = profile_results[k].get(op_key, {}).get(metric_key, 0)
                raw_vals.append(v)
                vals.append(fmt(v) if v else "—")

            # Bold the winner (highest ops, lowest latency)
            if raw_vals and any(v > 0 for v in raw_vals):
                if "ops" in metric_key:
                    best_idx = max(range(len(raw_vals)), key=lambda i: raw_vals[i])
                else:
                    non_zero = [v for v in raw_vals if v > 0]
                    if non_zero:
                        best_val = min(non_zero)
                        best_idx = next(i for i, v in enumerate(raw_vals) if v == best_val)
                    else:
                        best_idx = -1
                if best_idx >= 0:
                    vals[best_idx] = f"**{vals[best_idx]}**"

            md_lines.append(f"| {label} | " + " | ".join(vals) + " |")

        md_lines.append("")

        # Print to console
        print(f"\n  {profile_name}: {profile['description']}")
        header = f"  {'Metric':<20}"
        for k in profile_results:
            header += f" {SERVERS[k]['name']:>12}"
        print(header)
        print("  " + "-" * len(header))
        for label, op_key, metric_key, fmt in metrics:
            row = f"  {label:<20}"
            for k in profile_results:
                v = profile_results[k].get(op_key, {}).get(metric_key, 0)
                row += f" {fmt(v) if v else '—':>12}"
            print(row)

    # Write markdown
    md_content = "\n".join(md_lines) + "\n\n"
    old_content = ""
    if os.path.exists(RESULTS_MD):
        with open(RESULTS_MD) as f:
            old_content = f.read()
    with open(RESULTS_MD, "w") as f:
        f.write(md_content + old_content)
    print(f"\nResults written to {RESULTS_MD}")

    # Gate mode: check Pion results against conservative baselines
    if args.gate:
        # Per-CPU-class gate-floor baselines for mixed SET/GET workload (ops/sec),
        # keyed by pipeline depth. See gh #56.
        if args.gate_profile == "linux-epyc-8124p":
            # EPYC 8124P @ 2.45 GHz, Zen 4c Siena, -w 32. Source: 2026-05-02 commit
            # b214fbe — throughput P=1 = 265.6K, pipeline P=10 = 2.06M, pipeline_deep
            # P=30 = 4.48M. 95% thresholds below.
            GATE_BASELINES = {
                1:  252_000,    # 95% of 265.6K (throughput profile)
                10: 1_950_000,  # ~95% of 2.06M (pipeline profile)
                30: 4_250_000,  # ~95% of 4.48M (pipeline_deep, when run)
            }
            profile_label = "Linux EPYC 8124P @ 2.45 GHz (Zen 4c Siena, -w 32)"
        else:
            # Mac-side baselines, Apple Silicon -w 1. Aligned with /gate.md (Gate 3)
            # on 2026-05-03.
            GATE_BASELINES = {
                1:  330_000,    # P=1 w=1 floor (throughput profile)
                # Ratcheted 2.2M -> 2.4M on 2026-08-13 by explicit decision.
                # Recorded gate-PASSING runs include 2.35M and 2.39M, which this
                # floor would now reject, so a miss here is not by itself a
                # regression — re-run and A/B before treating it as one.
                10: 2_400_000,  # P=10 w=1 floor (pipeline profile)
                50: 500_000,    # P=50 w=1 floor (not exercised by /gate)
            }
            profile_label = "Mac (Apple Silicon, -w 1)"
        print(f"\n  Gate profile: {profile_label}")
        gate_failures = []
        for profile_name, profile_results in all_results.items():
            if "pion" not in profile_results:
                continue
            pion_data = profile_results["pion"]
            total_ops = pion_data.get("total", {}).get("ops_sec", 0)
            pipeline = PROFILES.get(profile_name, {}).get("pipeline", 1)
            # Use exact match or nearest lower baseline
            baseline = None
            for p in sorted(GATE_BASELINES.keys(), reverse=True):
                if pipeline >= p:
                    baseline = GATE_BASELINES[p]
                    break
            if baseline is None:
                baseline = GATE_BASELINES[1]
            if total_ops < baseline:
                gate_failures.append(
                    f"  {profile_name} (P={pipeline}): {total_ops:,.0f} ops/sec < {baseline:,} (baseline)")
            else:
                print(f"  GATE PASS  {profile_name} (P={pipeline}): "
                      f"{total_ops:,.0f} ops/sec >= {baseline:,}")
        if gate_failures:
            print(f"\n{'='*60}")
            print(f"GATE FAILED — {len(gate_failures)} profile(s) below baseline:")
            for f in gate_failures:
                print(f)
            print(f"{'='*60}")
            sys.exit(1)
        else:
            print(f"\nGATE PASSED — all Pion profiles meet ops/sec baselines.")


if __name__ == "__main__":
    main()
