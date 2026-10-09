#!/usr/bin/env python3
"""VectorDBBench comparison harness.

Runs VectorDBBench against Redis (FT and VSET), zvec, and Pion.
Results are reported in benchmark_results.md in the script directory.
"""

import subprocess
import time
import os
import sys
import re
import argparse
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from bench_proc import kill_pion_servers, kill_inference_workers  # gh #347
ZVEC_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "zvec")
VALKEY_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "valkey")
RESULTS_MD = os.path.join(SCRIPT_DIR, "benchmark_results.md")

CASE = "Performance1536D50K"
M = 16
EF_CONSTRUCTION = 128
EF_RUNTIME = 100
PORT = 6379              # redis/redis-stack default
PION_PORT = 6395         # dedicated — avoids conflict with redis auto-restart on 6379
VALKEY_PORT = 6397       # dedicated for valkey (VADD/VSIM)
VALKEY_SEARCH_PORT = 6398  # dedicated for valkey-search (FT.*)
CONCURRENCY_DURATION = 5  # seconds per concurrency level
BENCH_TIMEOUT = 7200       # seconds before killing a hung benchmark subprocess (5M needs ~60min ingest+optimize)

ZVEC_VENV_BENCH = os.path.join(ZVEC_ROOT, "venv_zvec", "bin", "vectordbbench")
VALKEY_SERVER = os.path.join(VALKEY_ROOT, "src", "valkey-server")
# Derived like ZVEC_ROOT/VALKEY_ROOT above (sibling checkout), overridable so a
# stranger can point at their own build. Was an absolute path under one
# developer's home directory, which made --all-profiles unrunnable elsewhere.
VALKEY_SEARCH_ROOT = os.environ.get(
    "VALKEY_SEARCH_ROOT", os.path.join(os.path.dirname(PROJECT_ROOT), "valkey-search"))
VALKEY_SEARCH_LIB = os.environ.get(
    "VALKEY_SEARCH_LIB", os.path.join(VALKEY_SEARCH_ROOT, ".build-release", "libsearch.dylib"))


def clear_port(port):
    subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(0.5)


def check_index_count(port, expected_min):
    """Check if Pion/Redis already has an index with >= expected_min docs.
    Returns True if load can be skipped."""
    try:
        import redis
        # gh #172: redis-py >=8 defaults to RESP3 (HELLO 3) and Pion answers
        # -NOPROTO — the check then always fails and search-only reloads 50K
        # docs. Pin protocol=2 where the kwarg exists (redis-py >=5).
        kw = {"protocol": 2} if int(redis.__version__.split(".")[0]) >= 5 else {}
        r = redis.Redis(host="localhost", port=port, socket_timeout=5, **kw)
        info = r.execute_command("FT.INFO", "idx")
        # FT.INFO returns a flat list: [..., 'num_docs', '5000000', ...]
        if isinstance(info, list):
            for i, v in enumerate(info):
                if isinstance(v, bytes):
                    v = v.decode()
                if v == "num_docs" and i + 1 < len(info):
                    count = int(info[i + 1])
                    print(f"  [search-only] Index has {count:,} docs (need {expected_min:,})")
                    return count >= expected_min
        r.close()
    except Exception as e:
        print(f"  [search-only] Could not check index: {e}")
    return False


def clean_pion_state():
    """Remove stale HNSW/WAL/snapshot files to prevent warm-restart recall corruption.

    Kills any leftover pion-server / inference worker first; otherwise mmap-held
    WAL files can be silently un-removable on macOS (the unlink doesn't fail
    *if* the holder has fully exited, but a sub-second race after pkill can
    leave files locked). Retries once after a brief wait.
    """
    import glob as _glob
    # Defensive kill of any straggler — silent if none.
    # By executable name, not cmdline: a cmdline match also kills a running
    # `mojo build ... -o pion-server` (gh #347).
    kill_pion_servers()
    kill_inference_workers()
    time.sleep(0.5)

    def _try_remove():
        leftover = []
        # gh #138: the bench-spawned servers also drop breadcrumb files
        # (pion-<port>.status / .crash.log) into the repo root.
        for pattern in ["pion.hnsw.*", "pion.wal.*", "pion.snapshot.*",
                        "pion.blob.*", "pion-*.status", "pion-*.crash.log"]:
            for f in _glob.glob(os.path.join(PROJECT_ROOT, pattern)):
                try:
                    os.remove(f)
                    print(f"  [clean] Removed {os.path.basename(f)}")
                except OSError as e:
                    leftover.append((f, e))
        return leftover

    leftover = _try_remove()
    if leftover:
        time.sleep(1.0)
        leftover = _try_remove()
    if leftover:
        # Loud failure: silent swallowing here previously masked a recall regression
        # caused by a stale WAL file polluting the keyspace.
        for f, e in leftover:
            print(f"  [clean] WARNING: could not remove {os.path.basename(f)}: {e}")


def start_server(cmd, name, port=PORT, startup_wait=3):
    clear_port(port)
    print(f"[{name}] Starting: {cmd}")
    # PION_BENCH_SERVER_LOG=<path>: keep the Pion server's own output (build
    # link-drop accounting, warnings) instead of discarding it — the only way
    # to read what the server said during a run that dipped.
    log_path = os.environ.get("PION_BENCH_SERVER_LOG") if name == "Pion" else None
    sink = open(log_path, "a") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, shell=True, stdout=sink, stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL)
    time.sleep(startup_wait)
    if proc.poll() is not None:
        print(f"[{name}] Server exited immediately — skipping.")
        return None
    return proc


def stop_server(proc, name, port=PORT):
    if proc is None:
        return
    print(f"[{name}] Stopping (waiting up to 60s for HNSW save)...")
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    clear_port(port)


def run_bench(cmd, timeout=BENCH_TIMEOUT):
    print(f"  Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=timeout,
        )
        combined_output = result.stdout + "\n" + result.stderr
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return combined_output
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Benchmark exceeded {timeout}s — killed.", file=sys.stderr)
        return ""


def extract_results(output, db_label):
    """Extract metrics using regex from combined stdout/stderr.

    vectordbbench's summary row is load_dur | qps | p99 | p95 | recall. The
    4th column used to be read as "optimize_time", so the FT.OPTIMIZE gate
    compared a p95 LATENCY (~0.0007 s) against its 25 s ceiling and could
    never fail. The build time is `optimize_duration=` in the load summary."""
    pattern = rf"{db_label}\s+.*?\|\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+"
    match = re.search(pattern, output)
    if match:
        opt = re.search(r"optimize_duration=([\d\.]+)", output)
        return {
            "load_dur": match.group(1),
            "qps": match.group(2),
            "p99": match.group(3),
            "p95": match.group(4),
            "optimize_time": opt.group(1) if opt else "",
            "recall": match.group(5)
        }
    return None


def index_never_built(res):
    """True when VectorDBBench never sent FT.OPTIMIZE, so every query ran
    against an index that was never built. Stock VectorDBBench's Redis client
    has an empty optimize(), and the run then reports recall 0.0 at an
    impossible QPS (182K, measured 2026-10-03) after a 0.0001 s "build"."""
    try:
        return float(res["optimize_time"]) < 0.05 and float(res["recall"]) < 0.1
    except (KeyError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pion-only", action="store_true",
                        help="Skip all competitors; benchmark Pion only")
    parser.add_argument("--valkey-only", action="store_true",
                        help="Skip all competitors; benchmark Valkey only")
    parser.add_argument("--ef-runtime", type=int, default=EF_RUNTIME,
                        help=f"ef_runtime for search (default: {EF_RUNTIME})")
    parser.add_argument("--workers", type=int, default=10,
                        help="Pion worker count (default: 10)")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable Metal GPU vector search (macOS Apple Silicon)")
    parser.add_argument("--polarquant", action="store_true",
                        help="Enable M6 PolarQuant: WHT rotation + INT4 quantization")
    parser.add_argument("--turboquant", action="store_true",
                        help="Enable M7 TurboQuant: 3-bit + QJL error correction")
    parser.add_argument("--nanoquant", action="store_true",
                        help="Enable N4 NanoQuant: block-INT2 quantization (484B/vec)")
    parser.add_argument("--profile", type=str, default=None,
                        help="Pion deployment profile: kv, vector, ai, full")
    parser.add_argument("--case", type=str, default=CASE,
                        help=f"VectorDBBench case type (default: {CASE})")
    parser.add_argument("--gate", action="store_true",
                        help="Gate mode: exit 1 if recall/QPS/optimize miss the profile's baseline")
    parser.add_argument("--gate-profile", choices=["mac", "linux-epyc-8124p"], default="mac",
                        help="Which CPU-class baseline to enforce. mac=Apple Silicon "
                             "(recall ≥ 0.940, QPS ≥ 7900, FT.OPTIMIZE ≤ 25s — default), "
                             "linux-epyc-8124p=EPYC 8124P @ 2.45 GHz Zen 4c Siena "
                             "(recall ≥ 0.940, QPS ≥ 4280, FT.OPTIMIZE ≤ 28s, 50K w=16). "
                             "See gh #56.")
    parser.add_argument("--dim", type=int, default=None,
                        help="Override vector dimensions (auto-detected from case if not set)")
    parser.add_argument("--max-elements", type=int, default=None,
                        help="Override max_elements for HNSW (auto-detected from case if not set)")
    parser.add_argument("--redis-host", type=str, default=None,
                        help="Use an already-running Redis Stack at this host (skips local start)")
    parser.add_argument("--redis-port", type=int, default=PORT,
                        help=f"Redis Stack port when using --redis-host (default: {PORT})")
    parser.add_argument("--redis-vset-host", type=str, default=None,
                        help="Use an already-running Redis 8.0 at this host for VSET (skips local start)")
    parser.add_argument("--redis-vset-port", type=int, default=PORT,
                        help=f"Redis 8.0 port when using --redis-vset-host (default: {PORT})")
    parser.add_argument("--search-only", action="store_true",
                        help="Skip load if server already has vectors (check FT.INFO doc count)")
    args = parser.parse_args()
    # gh #305: a contaminated box reads exactly like a regression. Refuse to
    # gate on one; transient causes (load, indexing, a CI job) get 3 min to clear.
    if args.gate:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from preflight import preflight
        if not preflight(wait=180):
            sys.exit(1)
    case = args.case

    ef = args.ef_runtime
    bench_results = []
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Redis FT (Query Engine)
    if not args.pion_only and not args.valkey_only:
        if args.redis_host:
            # Use externally running Redis Stack (e.g. separate container on Linux)
            proc = True  # sentinel — nothing to start/stop
            redis_host = args.redis_host
            redis_port = args.redis_port
        else:
            proc = start_server(f"redis-stack-server --port {PORT} --daemonize no", "Redis_FT")
            redis_host = "localhost"
            redis_port = PORT
    else:
        proc = None
        redis_host = "localhost"
        redis_port = PORT
    if proc:
        out = run_bench(
            f"pixi run vectordbbench redis --host {redis_host} --port {redis_port} --cmd --no-ssl "
            f"--case-type {case} --m {M} --ef-construction {EF_CONSTRUCTION} --ef-runtime {ef} "
            f"--db-label Redis_FT --concurrency-duration {CONCURRENCY_DURATION}"
        )
        res = extract_results(out, "Redis_FT")
        if res:
            res["name"] = "Redis Query Engine (FT.SEARCH)"
            bench_results.append(res)
        if proc is not True:
            stop_server(proc, "Redis_FT")

    # 2. Redis Vector Sets (VADD/VSIM)
    if not args.pion_only and not args.valkey_only:
        if args.redis_vset_host:
            vset_proc = True  # sentinel — nothing to start/stop
            vset_host = args.redis_vset_host
            vset_port = args.redis_vset_port
        else:
            vset_proc = start_server(f"redis-server --port {PORT}", "Redis_VSET")
            vset_host = "localhost"
            vset_port = PORT
        if vset_proc:
            out = run_bench(
                f"pixi run vectordbbench redis_vset --host {vset_host} --port {vset_port} --cmd --no-ssl "
                f"--case-type {case} --m {M} --ef-construction {EF_CONSTRUCTION} --ef-runtime {ef} "
                f"--db-label Redis_VSET --concurrency-duration {CONCURRENCY_DURATION} --num-concurrency 1,5,10"
            )
            res = extract_results(out, "Redis_VSET")
            if res:
                res["name"] = "Redis Vector Sets (VADD/VSIM)"
                bench_results.append(res)
            if vset_proc is not True:
                stop_server(vset_proc, "Redis_VSET")

    # 3. Valkey (VADD/VSIM) — SKIPPED: Valkey is a Redis 7.x fork; VADD/VSIM are Redis 8.0+ only.
    # Valkey-Search (FT.* module) is benchmarked separately below.

    # 4. Valkey-Search (FT.SEARCH via module)
    if not args.pion_only and os.path.exists(VALKEY_SERVER) and os.path.exists(VALKEY_SEARCH_LIB):
        proc = start_server(
            f"{VALKEY_SERVER} --loadmodule {VALKEY_SEARCH_LIB} --port {VALKEY_SEARCH_PORT} --save '' --dir /tmp --loglevel warning",
            "Valkey_Search", port=VALKEY_SEARCH_PORT,
        )
        if proc:
            out = run_bench(
                f"pixi run vectordbbench valkey --host localhost --port {VALKEY_SEARCH_PORT} --cmd --no-ssl "
                f"--case-type {case} --m {M} --ef-construction {EF_CONSTRUCTION} --ef-runtime {ef} "
                f"--db-label valkey_search --concurrency-duration {CONCURRENCY_DURATION}"
            )
            res = extract_results(out, "valkey_search")
            if res:
                res["name"] = "Valkey-Search (FT.SEARCH)"
                bench_results.append(res)
            stop_server(proc, "Valkey_Search", port=VALKEY_SEARCH_PORT)
    elif not args.pion_only:
        print(f"[Valkey-Search] Not found — skipping.")

    # 6. zvec (in-process library)
    if not args.pion_only and not args.valkey_only and os.path.exists(ZVEC_VENV_BENCH):
        print(f"[zvec] Running zvec bench...")
        out = run_bench(
            f"{ZVEC_VENV_BENCH} zvec --path /tmp/zvec_bench "
            f"--case-type {case} --m {M} --ef-construction {EF_CONSTRUCTION} --ef-search {ef} "
            f"--db-label zvec --concurrency-duration {CONCURRENCY_DURATION}"
        )
        res = extract_results(out, "zvec")
        if res:
            res["name"] = "zvec (In-Process)"
            bench_results.append(res)
    elif not args.pion_only:
        print(f"[zvec] Not found at {ZVEC_VENV_BENCH} — skipping.")

    # Auto-detect dimensions and max_elements from case type
    dim = args.dim
    max_elements = args.max_elements
    if dim is None:
        # Parse dimension from case name (e.g., Performance1536D50K → 1536)
        import re as _re
        m = _re.search(r'(\d+)D', case)
        if m:
            dim = int(m.group(1))
    if max_elements is None:
        # Parse scale from case name and add 10% headroom
        m = _re.search(r'D(\d+)([KM])', case)
        if m:
            scale = int(m.group(1))
            if m.group(2) == 'K':
                scale *= 1000
            elif m.group(2) == 'M':
                scale *= 1000000
            max_elements = int(scale * 1.1)

    # 7. Pion — dedicated port; 20s startup for 10-worker hashmap init + neighbor_pool alloc
    clean_pion_state()  # prevent warm-restart from stale HNSW files
    # gh #253: -w N > 1 is fenced behind --independent-workers. A vector
    # benchmark is exactly the shape the fence permits — the HNSW graph is
    # published cross-worker via SharedHNSWView, and the harness never expects
    # one connection to read another's KV writes.
    pion_cmd = (f"{PROJECT_ROOT}/pion-server -p {PION_PORT} -w {args.workers} "
                f"--independent-workers --no-auto-detect --no-auto-embed")
    if args.gpu: pion_cmd += " --gpu"
    if args.polarquant: pion_cmd += " --polarquant"
    if args.turboquant: pion_cmd += " --turboquant"
    if args.nanoquant: pion_cmd += " --nanoquant"
    if args.profile: pion_cmd += f" --profile {args.profile}"
    if dim and dim != 1536: pion_cmd += f" --dim {dim}"
    if max_elements and max_elements > 600000: pion_cmd += f" --max-elements {max_elements}"
    # Scale startup wait: 30s base + 15s per million max_elements
    startup_wait = 30 + (max_elements // 1000000) * 15 if max_elements else 30
    proc = start_server(
        pion_cmd, "Pion", port=PION_PORT, startup_wait=startup_wait,
    ) if not args.valkey_only else None
    if proc:
        # Check if we can skip load (index already populated from a previous run)
        skip_load_flags = ""
        if args.search_only:
            # Parse expected count from case name (e.g. Performance1536D5M → 5000000)
            import re as _re2
            _m2 = _re2.search(r'D(\d+)([KM])', case)
            expected = 0
            if _m2:
                expected = int(_m2.group(1)) * (1000 if _m2.group(2) == 'K' else 1000000)
            if expected > 0 and check_index_count(PION_PORT, expected):
                skip_load_flags = " --skip-drop-old --skip-load"
                print(f"  [search-only] Skipping load — index already has >= {expected:,} vectors")
            else:
                print(f"  [search-only] Index not ready — running full load")
        out = run_bench(
            f"pixi run vectordbbench redis --host localhost --port {PION_PORT} --cmd --no-ssl "
            f"--case-type {case} --m {M} --ef-construction {EF_CONSTRUCTION} --ef-runtime {ef} "
            f"--db-label pion --concurrency-duration {CONCURRENCY_DURATION} --num-concurrency 1,5,10"
            # vectordb-bench >= 1.0.x loads performance cases through
            # ConcurrentInsertRunner (4 connections by default) instead of the
            # SerialInsertRunner older releases used. Pion is shared-nothing:
            # each connection is served by one worker that owns a private HNSW
            # graph, and with num_shards=1 a query only searches its own
            # worker's graph. Ingesting over 4 connections therefore scatters
            # the 50K vectors across 4 graphs and recall collapses (0.7828 vs
            # 0.9603 measured, 2026-08-04) — a benchmark-topology artifact, not
            # an engine regression. Pin single-connection ingest so the load
            # shape matches the one every gate baseline was measured with.
            f" --load-concurrency 1"
            f"{skip_load_flags}"
        )
        res = extract_results(out, "pion")
        never_built = bool(res) and index_never_built(res)
        if res and not never_built:
            gpu_tag = " GPU" if args.gpu else ""
            res["name"] = f"Pion V27{gpu_tag} (ef={ef}, w={args.workers})"
            bench_results.append(res)
        # Skip stop_server + cleanup when PION_BENCH_NO_CLEANUP=1 (leaves
        # server alive on PION_PORT for the caller to drive a follow-up bench
        # — e.g. /gate's Gate 4b runs pipelined_bench against this same Pion).
        if not os.environ.get("PION_BENCH_NO_CLEANUP"):
            stop_server(proc, "Pion", port=PION_PORT)
            clean_pion_state()
        else:
            print(f"[Pion] Leaving server alive on port {PION_PORT} (PION_BENCH_NO_CLEANUP=1)")
        # After the server is stopped, so a refusal leaks no process.
        if never_built:
            print(f"\nERROR: FT.OPTIMIZE never ran (index build {res['optimize_time']} s, "
                  f"recall {res['recall']}), so the queries searched an empty index.\n"
                  "Stock VectorDBBench's Redis client has an empty optimize(). Run\n"
                  "`pixi run install-vdbbench`: it installs VectorDBBench where this\n"
                  "harness runs it and patches optimize() to send FT.OPTIMIZE.",
                  file=sys.stderr)
            sys.exit(1)

    # Prepend to benchmark_results.md
    if bench_results:
        new_content = f"### VectorDBBench Run: {run_timestamp}\n"
        new_content += f"**Case:** {case}, **M:** {M}, **EF_CON:** {EF_CONSTRUCTION}, **EF_RUN:** {ef}\n\n"
        new_content += "| Engine | Load Time (s) | Peak QPS | P99 Latency (s) | Recall | \n"
        new_content += "| :--- | :---: | :---: | :---: | :---: |\n"
        for r in bench_results:
            new_content += f"| {r['name']} | {r['load_dur']} | {r['qps']} | {r['p99']} | {r['recall']} |\n"
        new_content += "\n"

        old_content = ""
        if os.path.exists(RESULTS_MD):
            with open(RESULTS_MD, "r") as f:
                old_content = f.read()
        
        with open(RESULTS_MD, "w") as f:
            f.write(new_content + old_content)
        print(f"\nResults appended to {RESULTS_MD}")
    else:
        print("\nNo valid metrics captured in this run.")

    # Gate mode: check Pion results against baselines
    if args.gate:
        # Mac (Apple Silicon) baselines — historical default. ef=150, w=10, 50K case.
        # Linux EPYC 8124P @ 2.45 GHz (Zen 4c Siena) baselines — gh #56, re-derived 2026-10-09 (#28):
        # native build at 4b672bb (VNNI default), 50K case, w=16, ef=150, median of 5 recorded rounds →
        # 7,697 QPS / recall 0.960 / FT.OPTIMIZE 7.32s (raw runs: benchmarks/results/2026-10-09-linux-epyc-8124p/gate_floors/).
        # QPS ≥ 7312 (95% of 7697); FT.OPTIMIZE ≤ 7.7s (≈ 105% of 7.32); recall stays ≥ 0.940 (raise only).
        if args.gate_profile == "linux-epyc-8124p":
            RECALL_MIN = 0.940
            QPS_MIN = 7312
            OPTIMIZE_MAX = 7.7
            profile_name = "Linux EPYC 8124P @ 2.45 GHz (Zen 4c Siena, w=16)"
        else:
            RECALL_MIN = 0.940
            # Ratcheted 7400 -> 7900 on 2026-08-13 by explicit decision. Note
            # this floor is ABOVE the upper edge of the noise band recorded
            # for this machine (6684-7850 across identical binaries), so a legitimate
            # build on a loaded or thermally-limited machine will miss it — the
            # same day this was set, clean main measured 7982 and, the night
            # before, 5524-6056. Treat a miss here as "re-run and A/B against
            # the previous release binary" first; it is a tighter tripwire, not
            # evidence of a regression on its own.
            QPS_MIN = 7900
            OPTIMIZE_MAX = 25.0
            profile_name = "Mac (Apple Silicon, w=10)"
        pion_res = None
        for r in bench_results:
            if "Pion" in r.get("name", "") or "pion" in r.get("name", ""):
                pion_res = r
                break
        if pion_res is None:
            print("\nGATE FAILED — no Pion results captured.")
            sys.exit(1)
        gate_failures = []
        try:
            recall = float(pion_res["recall"])
            if recall < RECALL_MIN:
                gate_failures.append(f"  Recall: {recall:.4f} < {RECALL_MIN} (baseline)")
        except (ValueError, KeyError):
            gate_failures.append("  Recall: not captured")
        try:
            qps = float(pion_res["qps"])
            if qps < QPS_MIN:
                gate_failures.append(f"  Peak QPS: {qps:.0f} < {QPS_MIN} (baseline)")
        except (ValueError, KeyError):
            gate_failures.append("  Peak QPS: not captured")
        try:
            opt_time = float(pion_res["optimize_time"])
            if opt_time > OPTIMIZE_MAX:
                gate_failures.append(f"  FT.OPTIMIZE time: {opt_time:.1f}s > {OPTIMIZE_MAX}s (baseline)")
        except (ValueError, KeyError):
            gate_failures.append("  FT.OPTIMIZE time: not captured")
        if gate_failures:
            print(f"\n{'='*60}")
            print(f"GATE FAILED [{profile_name}] — vector baselines not met:")
            for f in gate_failures:
                print(f)
            print(f"{'='*60}")
            sys.exit(1)
        else:
            print(f"\nGATE PASSED [{profile_name}] — recall={pion_res['recall']}, QPS={pion_res['qps']}")


if __name__ == "__main__":
    main()
