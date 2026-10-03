#!/usr/bin/env python3

import argparse
import atexit
import json
import subprocess
import time
import os
import signal
import datetime
import shlex
import tempfile
import socket
import sys
import shutil
import statistics

def _valkey_root():
    """Sibling checkout by default; override with VALKEY_ROOT."""
    # Four dirnames: file -> valkey-benchmark/ -> benchmarks/ -> repo -> the
    # repo's PARENT. Three resolved to <repo>/valkey, so Valkey was silently
    # skipped ("Command not found ... Skipping") and reported as 0 RPS.
    return os.environ.get(
        "VALKEY_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), "valkey"))


def _dragonfly_bin():
    """Override with DRAGONFLY_BIN; defaults to a sibling build-opt tree."""
    return os.environ.get(
        "DRAGONFLY_BIN",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), "dragonfly", "build-opt", "dragonfly"))


def _find_benchmark_bin():
    """Find valkey-benchmark or redis-benchmark on the system."""
    for name in ["valkey-benchmark", "redis-benchmark"]:
        path = shutil.which(name)
        if path:
            return path
    # Fallback: a sibling valkey checkout, overridable via VALKEY_ROOT. Was an
    # absolute path under one developer's home directory.
    return os.path.join(_valkey_root(), "src", "valkey-benchmark")

BENCHMARK_BIN = _find_benchmark_bin()

GATE_BASELINES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gate_baselines.json")


class GateConfigError(Exception):
    """The baseline table is missing, malformed, or has no entry for this run."""


def load_gate_baselines(profile, pipeline):
    """Load the floor table for (profile, pipeline) from benchmarks/gate_baselines.json.

    Returns {"commands": {cmd: {"min": int, "known_good": int|None}},
             "profile_name": str, "partial": bool}.

    Raises GateConfigError rather than falling back to another depth's table:
    gating a P=50 run against the P=10 floors silently reports a pass/fail for a
    workload the floors were never measured on.
    """
    try:
        with open(GATE_BASELINES_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise GateConfigError(f"{GATE_BASELINES_PATH} not found")
    except json.JSONDecodeError as e:
        raise GateConfigError(f"{GATE_BASELINES_PATH} is not valid JSON: {e}")

    prof = data.get("profiles", {}).get(profile)
    if prof is None:
        raise GateConfigError(f"no profile '{profile}' in {GATE_BASELINES_PATH}")

    table = prof.get("pipelines", {}).get(str(pipeline))
    if table is None:
        have = ", ".join(f"P={p}" for p in sorted(prof.get("pipelines", {}), key=int))
        raise GateConfigError(
            f"profile '{profile}' has no baselines for P={pipeline} (has {have}). "
            f"Mac and Linux tables are not interchangeable and depths are not "
            f"substitutable — measure and record a floor table for this depth first.")

    commands = table.get("commands") or {}
    if not commands:
        raise GateConfigError(f"profile '{profile}' P={pipeline} has an empty command table")

    # Self-check: `min` is stored explicitly so a tolerance edit can never move a
    # shipped floor, but flag any row that has drifted away from the tolerance.
    tol = data.get("tolerance", 0.95)
    for cmd, spec in commands.items():
        kg = spec.get("known_good")
        if kg:
            expected = kg * tol
            if abs(spec["min"] - expected) > 1:
                print(f"WARN: baseline drift — {profile} P={pipeline} {cmd}: "
                      f"min {spec['min']:,} vs {tol:.0%} of known_good {kg:,} "
                      f"= {expected:,.0f}. Thresholds are never lowered; if this is "
                      f"an intentional ratchet, update known_good too.")

    return {
        "commands": commands,
        "profile_name": prof.get("description", profile),
        "partial": bool(table.get("partial")),
    }


def evaluate_gate(baselines, measured, tests_given=False):
    """Compare measured RPS against the floor table.

    Returns {"failures": [str], "missing": [str], "checked": int}.

    A baseline row absent from `measured` is a FAILURE, not a pass — otherwise a
    typo in -t, a renamed command, or a harness change that drops a row lets the
    gate certify a table it never measured. The one legitimate way to measure a
    subset is -t/--tests; that degrades the run to a PARTIAL gate instead.
    """
    failures = []
    missing = [cmd for cmd in baselines if cmd not in measured]
    for cmd, min_rps in baselines.items():
        if cmd not in measured:
            continue
        if measured[cmd] < min_rps:
            failures.append(f"  {cmd}: {int(measured[cmd]):,} < {min_rps:,} (baseline)")
    if missing and not tests_given:
        for cmd in missing:
            failures.append(f"  {cmd}: NOT MEASURED (baseline {baselines[cmd]:,})")
    return {"failures": failures, "missing": missing,
            "checked": len(baselines) - len(missing)}

def parse_args():
    parser = argparse.ArgumentParser(description="Run valkey-benchmark against Redis, Valkey, Dragonfly, and Pion.")
    parser.add_argument("-P", "--pipeline", type=int, help="Pipeline <numreq> requests")
    parser.add_argument("-t", "--tests", type=str, help="Only run the comma separated list of tests")
    parser.add_argument("-n", "--requests", type=int, help="Total number of requests")
    parser.add_argument("-c", "--clients", type=int, help="Number of parallel connections")
    parser.add_argument("-w", "--workers", type=int, help="Number of workers for Pion")
    parser.add_argument("--output", type=str, default="benchmark_results.md", help="Output markdown file")
    parser.add_argument("--pion-only", action="store_true", help="Only benchmark Pion (skip Redis/Valkey/Dragonfly)")
    parser.add_argument("--gate", action="store_true", help="Gate mode: exit 1 if any command falls below baseline")
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat the whole measure cycle N times (fresh server each run) "
                             "and aggregate per-command to the MEDIAN. Use for floor-table "
                             "sessions; 1 (default) behaves exactly as before.")
    parser.add_argument("--warmup-runs", type=int, default=None,
                        help="Discard the first K runs from the aggregate (they still run). "
                             "Defaults to 1 when --runs > 1, else 0 — the first run of a "
                             "series is consistently the low one.")
    parser.add_argument("--baseline-report", action="store_true",
                        help="Print a per-row floor proposal: current floor vs measured "
                             "median, RATCHET-ONLY (never proposes a lower floor). Writes a "
                             "candidate table to --baseline-out for a human to ratify.")
    parser.add_argument("--baseline-out", type=str, default="gate_baselines.candidate.json",
                        help="Where --baseline-report writes its candidate table. Never "
                             "overwrites the live gate_baselines.json.")
    parser.add_argument("--gate-profile", choices=["mac", "linux-epyc-8124p"], default="mac",
                        help="Which CPU-class baseline table to use. mac=Apple Silicon (default), "
                             "linux-epyc-8124p=EPYC 8124P @ 2.45 GHz Zen 4c Siena (gh #56). "
                             "Mac and Linux gates are NOT interchangeable.")
    return parser.parse_args()

def wait_for_port(port, timeout=30):
    """Poll until the port accepts connections or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def start_server(cmd, port, timeout=30):
    print(f"Starting server: {cmd}")
    try:
        process = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if not wait_for_port(port, timeout=timeout):
            print(f"Warning: server on port {port} did not become ready in {timeout}s.")
            process.kill()
            return None
        print(f"  Server on port {port} ready.")
        return process
    except FileNotFoundError:
        print(f"Warning: Command not found '{cmd}'. Skipping.")
        return None
    except Exception as e:
        print(f"Error starting server: {e}")
        return None

def stop_server(process):
    if process is None:
        return
    print("Stopping server...")
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    except Exception as e:
        print(f"Error stopping server: {e}")
    time.sleep(1)

def flushall(base_port, num_workers=4):
    """Issue FLUSHALL to all worker ports (base + secondary ports) to clear all keyspaces.
    With P2 routing, each worker owns its own keyspace — must flush all of them.
    Secondary ports: base+1+worker_id (e.g. 1975–1978 for base=1974, workers=4).
    """
    ports = [base_port] + [base_port + 1 + w for w in range(num_workers)]
    for p in ports:
        try:
            s = socket.create_connection(("127.0.0.1", p), timeout=2)
            s.sendall(b"*1\r\n$8\r\nFLUSHALL\r\n")
            s.recv(16)
            s.close()
        except Exception:
            pass  # secondary port may not be reachable; tolerate

# Test groups ordered by key type — FLUSHALL issued between groups to prevent WRONGTYPE errors.
# valkey-benchmark reuses the same key pool across all test types; without flushing, a key
# written as a LIST by LPUSH will be seen as WRONGTYPE by a subsequent SADD.
#
# LRANGE is separated from lpush/rpush/lpop/rpop because at P>=100 the LRANGE_600 response
# (~5.4KB each) × pipeline_depth exceeds Pion's 4MB response buffer, crashing the server.
# LRANGE is only run when pipeline depth < 100.
DEFAULT_TEST_GROUPS = [
    "ping",
    "set,get,incr",
    "lpush,rpush,lpop,rpop",
    "lrange",   # skipped at P>=100 (response buffer overflow)
    "sadd,spop",
    "hset",
    "zadd,zpopmin",
    "mset",
    "xadd",
]

def run_benchmark_group(port, name, args, tests):
    cmd = f"{BENCHMARK_BIN} -p {port} -q"
    if args.clients:
        cmd += f" -c {args.clients}"
    if args.requests:
        cmd += f" -n {args.requests}"
    if args.pipeline:
        cmd += f" -P {args.pipeline}"
    cmd += f" -t {tests}"

    print(f"  [{name}] group '{tests}': {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return parse_quiet_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"  Error in group '{tests}' for {name}: {e.stderr[:200]}")
        return parse_quiet_output(e.stdout)

def run_benchmark(port, name, args):
    if args.tests:
        # User specified explicit test list — run as-is (no grouping)
        cmd = f"{BENCHMARK_BIN} -p {port} -q"
        if args.clients:
            cmd += f" -c {args.clients}"
        if args.requests:
            cmd += f" -n {args.requests}"
        if args.pipeline:
            cmd += f" -P {args.pipeline}"
        cmd += f" -t {args.tests}"
        print(f"Running benchmark for {name}: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return parse_quiet_output(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for {name}: {e.stderr}")
            return parse_quiet_output(e.stdout)

    # Default: run each type group separately with FLUSHALL between groups
    print(f"Running benchmark for {name} (grouped, FLUSHALL between groups):")
    combined = {}
    num_workers = args.workers if args.workers else 4
    pipeline = args.pipeline if args.pipeline else 1
    for i, group in enumerate(DEFAULT_TEST_GROUPS):
        # Skip lrange at high pipeline depths to avoid response buffer overflow
        # (LRANGE_600 produces ~5.4KB/response; P>=100 exceeds the 4MB response buffer)
        if group == "lrange" and pipeline >= 100:
            print(f"  [{name}] skipping lrange group (P={pipeline} >= 100, response buffer limit)")
            continue
        if i > 0:
            flushall(port, num_workers)
        group_results = run_benchmark_group(port, name, args, group)
        # Don't overwrite already-measured commands: a later partial/error run may produce
        # stale low numbers for commands that were correctly measured in an earlier group.
        for k, v in group_results.items():
            if k not in combined:
                combined[k] = v
    return combined

def parse_quiet_output(output):
    results = {}
    for line in output.splitlines():
        if "requests per second" in line:
            parts = line.split(":")
            if len(parts) == 2:
                cmd = parts[0].strip()
                rps = float(parts[1].split()[0])
                results[cmd] = rps
    return results

def report_run_spread(kept, args):
    """Per-command spread across kept runs, then a RATCHET-ONLY floor proposal.

    The proposal never emits a floor below the one in force. The rule is
    absolute: a threshold is never lowered
    to accommodate a measurement. So a row whose median sits at or under its
    current floor is reported as something to DIAGNOSE, not as a calibration
    input — which is what keeps a refresh from laundering a regression into a
    new baseline.
    """
    samples = {}
    for run in kept:
        for cmd, rps in run.get("Pion", {}).items():
            samples.setdefault(cmd, []).append(rps)
    if not samples:
        return

    print(f"\n{'=' * 96}")
    print(f"PER-RUN SPREAD — Pion, {len(kept)} kept run(s)")
    print(f"{'=' * 96}")
    print(f"{'command':<36}{'median':>12}{'min':>12}{'max':>12}{'spread':>10}")
    for cmd in sorted(samples):
        v = samples[cmd]
        med, lo, hi = statistics.median(v), min(v), max(v)
        spread = (hi - lo) / med * 100 if med else 0.0
        print(f"{cmd:<36}{med:>12,.0f}{lo:>12,.0f}{hi:>12,.0f}{spread:>9.1f}%")

    if not args.baseline_report:
        return

    pipeline = args.pipeline if args.pipeline else 1
    try:
        table = load_gate_baselines(args.gate_profile, pipeline)
    except GateConfigError as e:
        print(f"\nCannot produce a baseline report: {e}")
        return
    current = table["commands"]
    tol = 0.95

    print(f"\n{'=' * 96}")
    print(f"FLOOR PROPOSAL (ratchet-only) — {table['profile_name']}, P={pipeline}")
    print(f"{'=' * 96}")
    print(f"{'command':<36}{'floor now':>12}{'median':>12}{'proposed':>12}{'':>4}verdict")

    candidate, ratchets, regressions = {}, 0, []
    for cmd, spec in current.items():
        floor_now = spec["min"]
        if cmd not in samples:
            print(f"{cmd:<36}{floor_now:>12,}{'—':>12}{'—':>12}    NOT MEASURED")
            candidate[cmd] = dict(spec)
            continue
        med = statistics.median(samples[cmd])
        proposed = int(med * tol)
        if med < floor_now:
            verdict = "REGRESSION? median is BELOW the floor in force — diagnose"
            regressions.append(cmd)
            candidate[cmd] = dict(spec)          # unchanged; never follow it down
        elif proposed > floor_now:
            verdict = f"RATCHET UP  (+{(proposed / floor_now - 1) * 100:.1f}%)"
            ratchets += 1
            candidate[cmd] = {"min": proposed, "known_good": int(med)}
        else:
            verdict = "keep (proposal not above current floor)"
            candidate[cmd] = dict(spec)
        print(f"{cmd:<36}{floor_now:>12,}{med:>12,.0f}{proposed:>12,}    {verdict}")

    print(f"\n{ratchets} row(s) would ratchet up; {len(regressions)} below their floor.")
    if regressions:
        print("BELOW FLOOR — do NOT lower these, diagnose them: " + ", ".join(regressions))
    print("Nothing is applied automatically. Review, then merge into "
          "benchmarks/gate_baselines.json and record the runs behind it.")

    try:
        with open(args.baseline_out, "w") as f:
            json.dump({
                "_generated": "candidate only — NOT the live table; a human ratifies this",
                "profile": args.gate_profile,
                "pipeline": pipeline,
                "kept_runs": len(kept),
                "samples": {c: sorted(v) for c, v in samples.items()},
                "commands": candidate,
            }, f, indent=2)
        print(f"Candidate table written to {args.baseline_out}")
    except Exception as e:
        print(f"Could not write {args.baseline_out}: {e}")


def main():
    args = parse_args()
    # gh #305: a contaminated box reads exactly like a regression. Refuse to
    # gate on one; transient causes (load, indexing, a CI job) get 3 min to clear.
    if args.gate:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from preflight import preflight
        if not preflight(wait=180):
            sys.exit(1)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pion_bin = os.path.join(project_root, "pion-server")
    
    if not os.path.exists(pion_bin):
        print(f"Warning: {pion_bin} not found. Please build Pion first using 'pixi run build'. Attempting to run anyway...")

    pion_cmd = f"{pion_bin} -p 1974 --no-auto-detect --no-auto-embed"
    if args.workers is not None:
        pion_cmd += f" -w {args.workers}"
        # gh #253: -w N > 1 refuses to start without this. valkey-benchmark's
        # clients each drive their own keys and never read across connections.
        if args.workers > 1:
            pion_cmd += " --independent-workers"

    # Each oracle gets a fresh private --dir. With the shared CWD, Valkey 9.0
    # loaded a dump.rdb that Redis 8.10 had written (RDB v15), exited at once,
    # and showed up as 0 RPS in every column. --save '' stops new dumps, but
    # nothing stopped an existing one from being LOADED.
    redis_dir = tempfile.mkdtemp(prefix="pion-bench-redis-")
    valkey_dir = tempfile.mkdtemp(prefix="pion-bench-valkey-")
    # Removed at exit, on sys.exit() and Ctrl-C too. Every run used to leave
    # this pair behind, --pion-only included: ~150 had piled up in $TMPDIR.
    for d in (redis_dir, valkey_dir):
        atexit.register(shutil.rmtree, d, ignore_errors=True)
    servers = {
        "Redis": {"cmd": f"redis-server --port 6379 --save '' --appendonly no --dir {redis_dir}", "port": 6379},
        "Valkey": {"cmd": f"{os.path.join(_valkey_root(), 'src', 'valkey-server')} --port 6380 --save '' --appendonly no --dir {valkey_dir}", "port": 6380},
        "Dragonfly": {"cmd": f"{_dragonfly_bin()} --bind 127.0.0.1 --port 6381 --maxmemory=3gb", "port": 6381},
        "Pion": {"cmd": pion_cmd, "port": 1974}
    }

    results = {"Redis": {}, "Valkey": {}, "Dragonfly": {}, "Pion": {}}

    skip = {"Redis", "Valkey", "Dragonfly"} if args.pion_only else set()

    # Track active server process for cleanup on SIGINT
    active_proc = [None]  # mutable container for signal handler access

    def sigint_handler(signum, frame):
        print("\nInterrupted — stopping server...")
        stop_server(active_proc[0])
        sys.exit(1)

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    def measure_once():
        """One full pass over the servers. Fresh server per pass, as a lone run gets."""
        out = {}
        for name, config in servers.items():
            if name in skip:
                continue
            timeout = 30 if name != "Pion" else 60
            proc = start_server(config["cmd"], config["port"], timeout=timeout)
            if proc is None:
                continue
            active_proc[0] = proc
            try:
                out[name] = run_benchmark(config["port"], name, args)
            finally:
                stop_server(proc)
                active_proc[0] = None
        return out

    n_runs = max(1, args.runs)
    warmup = args.warmup_runs if args.warmup_runs is not None else (1 if n_runs > 1 else 0)
    warmup = min(warmup, n_runs - 1)  # never discard everything

    per_run = []
    for r in range(n_runs):
        if n_runs > 1:
            tag = "warm-up, discarded" if r < warmup else f"kept {r - warmup + 1}/{n_runs - warmup}"
            print(f"\n{'=' * 60}\nRUN {r + 1}/{n_runs} ({tag})\n{'=' * 60}")
        per_run.append(measure_once())

    kept = per_run[warmup:]
    for name in results:
        samples = {}
        for run in kept:
            for cmd, rps in run.get(name, {}).items():
                samples.setdefault(cmd, []).append(rps)
        if samples:
            results[name] = {cmd: statistics.median(v) for cmd, v in samples.items()}

    if n_runs > 1:
        print(f"\nAggregated {len(kept)} kept run(s) of {n_runs} to per-command MEDIAN "
              f"({warmup} warm-up discarded).")
        report_run_spread(kept, args)

    signal.signal(signal.SIGINT, original_sigint)

    # Collect all commands tested
    all_commands = set()
    for name in results:
        all_commands.update(results[name].keys())
    all_commands = sorted(list(all_commands))

    if not all_commands:
        print("No results gathered. Please check if valkey-benchmark is installed and servers can start.")
        return

    # Prepare markdown table
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    params = []
    if args.clients: params.append(f"Clients: {args.clients}")
    if args.requests: params.append(f"Requests: {args.requests}")
    if args.pipeline: params.append(f"Pipeline: {args.pipeline}")
    if args.tests: params.append(f"Tests: {args.tests}")
    if args.workers: params.append(f"Pion workers: {args.workers}")
    if n_runs > 1:
        # The recorded numbers are an aggregate, not a single run. Say so in the
        # artifact itself — a bare table reads as one run to everyone later.
        params.append(f"**MEDIAN of {len(kept)} kept runs** ({n_runs} run, "
                      f"{warmup} warm-up discarded)")
    params_str = ", ".join(params) if params else "Default valkey-benchmark and Pion parameters"
    
    md = []
    md.append(f"### Benchmark Run: {now}")
    md.append(f"**Parameters:** {params_str}")
    md.append("")
    md.append("| Command | Valkey (RPS) | Redis (RPS) | Dragonfly (RPS) | Pion (RPS) | Winner | Pion vs Best (%) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for cmd in all_commands:
        valkey_rps = results["Valkey"].get(cmd, 0.0)
        redis_rps = results["Redis"].get(cmd, 0.0)
        dragonfly_rps = results["Dragonfly"].get(cmd, 0.0)
        pion_rps = results["Pion"].get(cmd, 0.0)

        best_other = max(valkey_rps, redis_rps, dragonfly_rps)
        if best_other > 0:
            pion_vs_best = ((pion_rps - best_other) / best_other) * 100
        else:
            pion_vs_best = 0.0

        # Determine winner
        max_rps = max(valkey_rps, redis_rps, dragonfly_rps, pion_rps)
        winners = []
        if max_rps > 0:
            if valkey_rps == max_rps: winners.append("Valkey")
            if redis_rps == max_rps: winners.append("Redis")
            if dragonfly_rps == max_rps: winners.append("Dragonfly")
            if pion_rps == max_rps: winners.append("Pion")
        winner_str = "/".join(winners) if winners else "N/A"

        # Format row
        def fmt_rps(rps, is_max):
            s = f"{int(rps):,}"
            return f"**{s}**" if is_max and rps > 0 else s
        
        valkey_str = fmt_rps(valkey_rps, valkey_rps == max_rps)
        redis_str = fmt_rps(redis_rps, redis_rps == max_rps)
        dragonfly_str = fmt_rps(dragonfly_rps, dragonfly_rps == max_rps)
        pion_str = fmt_rps(pion_rps, pion_rps == max_rps)
        winner_col = f"**{winner_str}**" if winner_str != "N/A" else "N/A"

        md.append(f"| {cmd} | {valkey_str} | {redis_str} | {dragonfly_str} | {pion_str} | {winner_col} | {pion_vs_best:.1f}% |")

    md.append("")
    md_str = "\n".join(md)

    # Output to screen
    print("\n" + md_str)

    # Prepend to file (New results at the top)
    existing_content = ""
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing_content = f.read()

    with open(args.output, "w") as f:
        f.write(md_str + "\n\n" + existing_content)
    
    print(f"Results prepended to {args.output}")

    # Cleanup .dfs and .wal files in the current directory. Pion's own
    # persistence files (pion.wal.0, sealed pion.wal.0.N segments, blob arenas,
    # snapshots) never matched the suffix checks — with gh #149 rotation they
    # would otherwise accumulate across runs and grow the next run's replay.
    print("Cleaning up temporary data files...")
    for filename in os.listdir("."):
        if (filename.endswith(".dfs") or filename.endswith(".wal")
                or filename.startswith("pion.wal.")
                or filename.startswith("pion.blob.")
                or filename.startswith("pion.snapshot.")):
            try:
                os.remove(filename)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")

    # Gate mode: check Pion results against baselines (95% of known-good run)
    if args.gate and "Pion" in results and results["Pion"]:
        pipeline = args.pipeline if args.pipeline else 1

        try:
            table = load_gate_baselines(args.gate_profile, pipeline)
        except GateConfigError as e:
            print(f"\n{'='*60}")
            print(f"GATE FAILED — cannot load baselines: {e}")
            print(f"{'='*60}")
            sys.exit(1)

        baselines = {cmd: spec["min"] for cmd, spec in table["commands"].items()}
        profile_name = table["profile_name"]

        # A row present in the table but absent from the results is NOT a pass.
        # `-t/--tests` legitimately restricts the run, so that case degrades to an
        # explicitly-labelled PARTIAL gate rather than a silent full-gate PASS.
        partial_run = bool(args.tests)
        verdict = evaluate_gate(baselines, results["Pion"], tests_given=partial_run)
        gate_failures = verdict["failures"]
        missing = verdict["missing"]
        checked = verdict["checked"]

        if gate_failures:
            print(f"\n{'='*60}")
            print(f"GATE FAILED [{profile_name}] — {len(gate_failures)} command(s) below/missing at P={pipeline}:")
            for f in gate_failures:
                print(f)
            print(f"{'='*60}")
            sys.exit(1)

        if partial_run and missing:
            print(f"\n{'='*60}")
            print(f"PARTIAL GATE [{profile_name}] — {checked}/{len(baselines)} rows checked at P={pipeline}; "
                  f"{len(missing)} not measured because --tests was given.")
            print("This is NOT a push certification. Re-run without --tests to certify.")
            print(f"{'='*60}")
        elif table.get("partial"):
            print(f"\nPARTIAL GATE [{profile_name}] — all {checked} core commands meet the P={pipeline} "
                  f"fallback thresholds. This depth has no certified floor table; "
                  f"only P=10 certifies a push.")
        else:
            print(f"\nGATE PASSED [{profile_name}] — all {checked} commands meet P={pipeline} baseline (95% threshold).")

if __name__ == "__main__":
    main()