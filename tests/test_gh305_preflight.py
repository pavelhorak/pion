#!/usr/bin/env python3
"""gh #305: benchmarks/preflight.py refuses a contaminated box and kills nothing.

    python3 tests/test_gh305_preflight.py

Part 1 is pure: classify_processes over synthetic process tables, covering each
spelling that the prose checklist got wrong (pion-server-dev escaping
`pkill -x pion-server`, redis rewriting its title so `pgrep -x` never matches).

Part 2 is live and hermetic: it plants a process whose executable is NAMED
pion-server (a compiled sleep(60)), runs the CLI, and requires exit 1, a
finding naming that pid, and the process still alive afterwards. Other machine
state may add findings; the assertions only concern the planted one.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import preflight as pf  # noqa: E402

fails = 0


def check(name, ok, note=""):
    global fails
    fails += not ok
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + note if note else ''}")


def kinds(findings):
    return [(f.headline.split(":")[0], f.owner, f.blocking, f.transient) for f in findings]


QUIET = [
    (1, 0.0, "/sbin/launchd", "/sbin/launchd"),
    (200, 24.0, "/Applications/Claude.app/Contents/MacOS/claude", "claude"),
    (300, 3.0, "/System/Library/WindowServer", "WindowServer -daemon"),
    (400, 0.1, "/opt/homebrew/bin/redis-cli", "redis-cli -p 6399 ping"),
]


def part1():
    print("classify_processes (pure)")
    check("quiet table -> no findings", pf.classify_processes(QUIET) == [])

    t = QUIET + [(501, 0.0, "/x/pion-server-dev", "./pion-server-dev -p 1975")]
    f = pf.classify_processes(t)
    check("pion-server-dev is a leaked server", len(f) == 1 and f[0].owner == "ours"
          and f[0].blocking and "pion-server-dev" in f[0].detail, str(kinds(f)))
    check("...and the fix names pkill -x pion-server-dev",
          bool(f) and "pkill -x pion-server-dev" in f[0].detail)
    check("allow_pion suppresses it", pf.classify_processes(t, allow_pion=True) == [])

    t = QUIET + [(502, 0.2, "/opt/homebrew/bin/redis-server",
                  "/opt/homebrew/bin/redis-server *:6399")]
    f = pf.classify_processes(t)
    check("retitled redis-server is caught", len(f) == 1 and "oracle" in f[0].headline
          and "kill 502" in f[0].detail, str(kinds(f)))
    check("redis-cli is not a server", pf.classify_processes(QUIET) == [])

    t = QUIET + [(503, 150.0, "/Users/x/runner/bin/Runner.Worker", "Runner.Worker spawnclient"),
                 (504, 120.0, "mojo", "mojo run -I . tests/x.mojo")]
    f = pf.classify_processes(t)
    check("CI job is theirs, transient, blocking",
          len(f) == 1 and f[0].owner == "theirs" and f[0].transient and f[0].blocking,
          str(kinds(f)))
    check("...and says never kill", bool(f) and "NEVER kill" in f[0].detail)

    t = QUIET + [(505, 99.0, "/usr/bin/python3", "python3 research/s127/exp3_sweep.py")]
    f = pf.classify_processes(t)
    check("s1??/exp*.py research job is theirs",
          len(f) == 1 and f[0].owner == "theirs", str(kinds(f)))

    t = QUIET + [(506, 1.0, "/Applications/Ollama.app/ollama", "ollama serve")]
    f = pf.classify_processes(t)
    check("Ollama warns but does not block",
          len(f) == 1 and not f[0].blocking, str(kinds(f)))

    f = pf.classify_processes(QUIET + [(502, 0.2, "redis-server", "redis-server *:6399")],
                              skip_pids={502})
    check("own lineage is skipped", f == [])


def part2():
    print("CLI against a planted pion-server (live)")
    d = tempfile.mkdtemp()
    fake = os.path.join(d, "pion-server")
    # Compiled, not copied: macOS SIGKILLs a copy of a signed system binary
    # (/bin/sleep) on exec. Same stand-in as test_gh347_bench_kill.py.
    src = os.path.join(d, "s.c")
    with open(src, "w") as f:
        f.write("#include <unistd.h>\nint main(void){sleep(60);return 0;}\n")
    subprocess.run(["cc", "-o", fake, src], check=True)
    proc = subprocess.Popen([fake])
    try:
        time.sleep(0.5)
        env = {k: v for k, v in os.environ.items() if k != "PION_GATE_FORCE"}
        r = subprocess.run([sys.executable, os.path.join(ROOT, "benchmarks", "preflight.py")],
                           capture_output=True, text=True, env=env, timeout=60)
        check("exit 1", r.returncode == 1, f"rc={r.returncode}")
        check("finding names the planted pid",
              f"pion-server pid {proc.pid}" in r.stderr, r.stderr.splitlines()[:3].__str__())
        check("planted process NOT killed", proc.poll() is None)

        env["PION_GATE_FORCE"] = "1"
        r = subprocess.run([sys.executable, os.path.join(ROOT, "benchmarks", "preflight.py")],
                           capture_output=True, text=True, env=env, timeout=60)
        check("PION_GATE_FORCE=1 bypasses, and says so",
              r.returncode == 0 and "not a gate pass" in r.stderr)

        r = subprocess.run([sys.executable, os.path.join(ROOT, "benchmarks", "preflight.py"),
                            "--allow-pion"], capture_output=True, text=True,
                           env={k: v for k, v in env.items() if k != "PION_GATE_FORCE"},
                           timeout=60)
        check("--allow-pion drops the planted finding",
              f"pid {proc.pid}" not in r.stderr)
    finally:
        proc.kill()
        proc.wait()
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    part1()
    part2()
    print("\n" + (f"{fails} FAILED" if fails else "ALL PASS"))
    sys.exit(1 if fails else 0)
