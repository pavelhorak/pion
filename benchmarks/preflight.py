#!/usr/bin/env python3
"""Refuse to benchmark on a contaminated machine (gh #305).

    python3 benchmarks/preflight.py            # exit 1 on any blocking finding
    python3 benchmarks/preflight.py --wait 180 # re-check transient findings (load,
                                               # indexing, a CI job) until clear

Every check here comes from a session that lost hours to it. A contaminated run
is not a noisy run: it depresses rows selectively and reads exactly like a
regression. Each finding says whether the cause is OURS to remove (a server we
leaked, disk we filled) or SOMEONE ELSE'S to wait for (a CI job, a research
loop, Spotlight). This script never kills anything itself; it tells you what
to kill.

The bench harnesses run this at the top of every `--gate` run, and the Claude
Code bench guard imports it, so there is one copy of the rules. Bypass with
PION_GATE_FORCE=1, and then the result is not a gate pass.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_proc import pion_server_pids  # noqa: E402

GATE_PORT = 1974
MIN_FREE_GB = 2.0
# This box idles near 1.3-2.0 with a Claude session and the desktop app up.
# Spotlight indexing a fresh venv left it at 6.31 with 0% instantaneous CPU.
MAX_LOAD1 = 4.0
FOREIGN_CPU_PCT = 25.0
# audiomxd's lifetime CPU share. Healthy: 0.09 s over 29 h. Livelocked: ~96%.
AUDIOMXD_SPIN_RATE = 0.05

BENIGN = (
    "claude", "Claude Helper", "WindowServer", "Terminal", "iTerm",
    "Flurry", "loginwindow", "kernel_task", "pion-server", "ollama",
)

# Oracle/competitor servers the harnesses and differential tests start. Redis
# rewrites its process title to `redis-server *:6399`, so match the COMMAND
# LINE, never an exact comm: `pgrep -x redis-server` never matches.
ORACLES = ("redis-server", "valkey-server", "dragonfly")

# Long-running experiment scripts that share this box.
RESEARCH_JOB = re.compile(r"\bs1\d\d/exp[^/\s]*\.py\b")


@dataclass
class Finding:
    headline: str
    detail: str
    owner: str          # "ours" (remove it) | "theirs" (wait; never kill)
    transient: bool = False  # clears on its own; --wait re-checks these
    blocking: bool = True


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def _cpu_seconds(t):
    """Parse ps TIME/ETIME ([DD-]HH:MM:SS.ss or MM:SS.ss). None if unparseable."""
    try:
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
        parts = [float(p) for p in t.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        return None


def _own_lineage():
    """This process and its ancestors: their command lines name what we match."""
    pids, pid = set(), os.getpid()
    for _ in range(32):
        pids.add(pid)
        ppid = _sh(f"ps -o ppid= -p {pid}")
        if not ppid.isdigit() or int(ppid) <= 1:
            break
        pid = int(ppid)
    return pids


def process_table():
    """[(pid, pcpu, comm, command)] from two ps snapshots joined on pid.

    comm and command are fetched separately because both can contain spaces
    ("Claude Helper"), so one combined row cannot be split reliably.
    """
    comms, cmds = {}, {}
    for line in _sh("ps -Ao pid=,pcpu=,comm=").splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit():
            try:
                comms[int(parts[0])] = (float(parts[1]), parts[2].strip())
            except ValueError:
                pass
    for line in _sh("ps -Ao pid=,command=").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            cmds[int(parts[0])] = parts[1]
    return [(pid, pc, comm, cmds.get(pid, comm)) for pid, (pc, comm) in comms.items()]


def foreign_consumer(ps_output, benign=BENIGN, threshold=FOREIGN_CPU_PCT):
    """First non-benign process above `threshold`% CPU in `ps -Ao pcpu,comm -r`
    output (header included), or None. Returns (name, pcpu).

    Scans every row and ignores load average. On 2026-08-15 a foreign job at
    48.6% slipped past a check that looked only at the hottest row (benign) and
    gated on load1 >= 2.0 (it was 1.93): one busy process on an idle box does
    not move the 1-minute average.
    """
    for row in (ps_output or "").splitlines()[1:]:
        parts = row.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pcpu = float(parts[0])
        except ValueError:
            continue
        name = parts[1].strip()
        if pcpu >= threshold and not any(b in name for b in benign):
            return name, pcpu
    return None


def classify_processes(table, skip_pids=(), allow_pion=False):
    """Findings about resident processes. Pure over `table`, so it is testable."""
    out = []
    skip = set(skip_pids)
    rows = [r for r in table if r[0] not in skip]

    if not allow_pion:
        pion = [(pid, os.path.basename(comm)) for pid, _, comm, _ in rows
                if os.path.basename(comm).startswith("pion-server")]
        if pion:
            names = sorted({n for _, n in pion})
            out.append(Finding(
                f"leaked Pion server(s): {', '.join(f'{n} pid {p}' for p, n in pion)}",
                "Three forgotten servers once cost ~30% on every WRITE row for hours "
                "while reads barely moved; that read/write split is the tell. Each holds "
                "a 10M-slot hash map and a 256 MB WAL mapping.\n"
                f"  Remove: pkill -x {' ; pkill -x '.join(names)}  then re-check "
                "(`pkill -x pion-server` does NOT match pion-server-dev).",
                "ours"))

    oracles = [(pid, cmd) for pid, _, _, cmd in rows
               if any(o in cmd for o in ORACLES) and "preflight" not in cmd]
    if oracles:
        out.append(Finding(
            "leaked oracle/competitor server(s): "
            + "; ".join(f"pid {p} `{c[:60]}`" for p, c in oracles),
            "A stray redis-server from a differential run cost ~56% on MSET. "
            "`pgrep -x redis-server` never sees one because Redis rewrites its "
            "process title; this check matches the command line.\n"
            f"  Remove: kill {' '.join(str(p) for p, _ in oracles)}",
            "ours"))

    research = [(pid, cmd) for pid, _, _, cmd in rows if RESEARCH_JOB.search(cmd)]
    if research:
        out.append(Finding(
            "research job running: "
            + "; ".join(f"pid {p} `{c[:70]}`" for p, c in research),
            "Another experiment is using this box. NEVER kill it: wait for it to "
            "finish.",
            "theirs", transient=True))

    worker = [pid for pid, _, comm, cmd in rows
              if "Runner.Worker" in comm or "Runner.Worker" in cmd]
    mojo = [pid for pid, _, comm, _ in rows if os.path.basename(comm) == "mojo"]
    if worker or mojo:
        what = (f"CI job ACTIVE (Runner.Worker pid {','.join(map(str, worker))})"
                if worker else f"`mojo` compile running (pid {','.join(map(str, mojo))})")
        out.append(Finding(
            what,
            "The self-hosted CI runner on this box spawns `mojo` compilers at "
            "100-170% CPU, and jobs queue back to back: require ~60-90 s of "
            "sustained quiet, and re-check right after the run. A stray local build "
            "looks the same.\n  NEVER kill a CI job. Wait for it to finish.",
            "theirs", transient=True))

    sup = [(pid, cmd) for pid, _, _, cmd in rows if "pion_sup" in cmd]
    live_sup = [p for p, _ in sup
                if not _sh(f"ps -o stat= -p {p}").lstrip().startswith(("T", "t"))]
    if live_sup:
        pids = " ".join(map(str, live_sup))
        out.append(Finding(
            f"research supervisor (pion_sup.sh) running, pid {pids}",
            "It respawns a server on 1974 whenever a harness kills one, and that "
            "server's startup burns a core mid-benchmark.\n"
            f"  Freeze it for the window: kill -STOP {pids}  (then kill -CONT {pids}). "
            "NEVER kill it.",
            "theirs"))

    ollama = [pid for pid, _, comm, _ in rows if os.path.basename(comm).startswith("ollama")]
    if ollama:
        out.append(Finding(
            f"Ollama running (pid {','.join(map(str, ollama))})",
            "Resident Ollama costs ~6% vector QPS. Quit it for a vector gate run, or "
            "report the number as taken with it up.",
            "ours", blocking=False))

    return out


def check_audiomxd():
    if platform.system() != "Darwin":
        return []
    console = _sh("stat -f%Su /dev/console")
    at_login_window = bool(console) and console != os.environ.get("USER", "")
    line = _sh("ps -Ao pid,etime,time,comm | grep '[a]udiomxd' | head -1")
    spinning, note = None, ""
    parts = line.split(None, 3)
    if len(parts) >= 3:
        cpu, elapsed = _cpu_seconds(parts[2]), _cpu_seconds(parts[1])
        if cpu is not None and elapsed:
            rate = cpu / elapsed
            spinning = rate >= AUDIOMXD_SPIN_RATE
            note = (f"audiomxd: {cpu:.2f}s CPU over {elapsed / 3600:.1f}h "
                    f"({rate * 100:.3f}% of one core)")
    fix = "  Fix: log in at the console (Screen Sharing counts)."
    if spinning:
        return [Finding(f"audiomxd IS LIVELOCKED ({note})",
                        "At the login window it retries forever at ~96% of a core. On "
                        "2026-07-30, 16 of 20 KV rows fell below baseline on an "
                        "UNMODIFIED binary.\n" + fix, "ours")]
    if at_login_window and spinning is None:
        return [Finding(f"audiomxd livelock likely (console user '{console}', CPU "
                        "unmeasurable)", "The Mac is at the login window.\n" + fix, "ours")]
    if at_login_window:
        return [Finding(f"console user is '{console}' (login window); audiomxd not "
                        "spinning yet", note + ". It can enter the loop at any time; "
                        "re-check its CPU after the run.\n" + fix, "ours", blocking=False)]
    return []


def check_port():
    """Port 1974 held by something that is not a Pion server (e.g. an iOS app)."""
    pids = _sh(f"lsof -nP -iTCP:{GATE_PORT} -sTCP:LISTEN -t").split()
    pion = {str(p) for p in pion_server_pids()}
    foreign = [p for p in pids if p not in pion]
    if not foreign:
        return []
    names = ", ".join(f"{p} {_sh(f'ps -o comm= -p {p}') or '?'}" for p in foreign)
    return [Finding(f"port {GATE_PORT} held by a non-Pion process: {names}",
                    "The harness's server then FATALs on bind, and the bench measures "
                    "whatever holds the port. Quit that app.", "theirs")]


def check_disk(path="/"):
    try:
        free_gb = shutil.disk_usage(path).free / 1e9
    except OSError:
        return []
    if free_gb >= MIN_FREE_GB:
        return []
    return [Finding(f"disk pressure: {free_gb:.1f} GB free on / (want >= {MIN_FREE_GB:.0f})",
                    "Near-full, the WAL's mmap/msync stalls both perf gates, and a FULL "
                    "disk makes `pixi run build` exit 0 with stale machine code. Swapfiles "
                    "in /System/Volumes/VM can swing free space by gigabytes within minutes.",
                    "ours")]


def check_load(table, max_load=MAX_LOAD1):
    out = []
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        return out
    mds = [(pid, pc) for pid, pc, comm, _ in table
           if os.path.basename(comm).startswith("mds") and pc >= 20.0]
    if mds:
        out.append(Finding(
            "Spotlight indexing: " + ", ".join(f"pid {p} at {pc:.0f}%" for p, pc in mds),
            "A big install (a venv writes ~10k files) is indexed at ~250% CPU for "
            "minutes. Wait for it.", "theirs", transient=True))
    if load1 > max_load:
        out.append(Finding(
            f"1-minute load average {load1:.2f} > {max_load:.1f}",
            "Watch load, not instantaneous CPU: after Spotlight showed 0% the load was "
            "still 6.31 and falling, and a run started there is uniformly depressed.",
            "theirs", transient=True))
    elif not pion_server_pids():
        ps = _sh("ps -Ao pcpu,comm -r | head -9")
        hit = foreign_consumer(ps)
        if hit and os.path.basename(hit[0]).startswith("mds") and not mds:
            out.append(Finding(
                f"Spotlight indexing: {os.path.basename(hit[0])} at {hit[1]:.0f}%",
                "A big install (a venv writes ~10k files) is indexed at ~250% CPU for "
                "minutes. Wait for it.", "theirs", transient=True))
        elif hit and not os.path.basename(hit[0]).startswith("mds"):
            out.append(Finding(
                f"unidentified CPU consumer: {hit[0]} at {hit[1]:.0f}% (load {load1:.2f})",
                "Identify it before benchmarking: a uniform 4-25% depression is what "
                "this produces.", "theirs", transient=True))
    return out


def run_checks(allow_pion=False, max_load=MAX_LOAD1):
    table = process_table()
    found = []
    found += classify_processes(table, skip_pids=_own_lineage(), allow_pion=allow_pion)
    found += check_audiomxd()
    found += check_port()
    found += check_disk()
    found += check_load(table, max_load)
    return found


def report(found, stream=sys.stderr):
    hard = [f for f in found if f.blocking]
    soft = [f for f in found if not f.blocking]
    if not hard:
        print("preflight: clean" + (" (with warnings)" if soft else ""), file=stream)
    else:
        print("preflight: BLOCKED. A benchmark here would be uninterpretable, "
              "not merely noisy.", file=stream)
    for i, f in enumerate(hard + soft, 1):
        tag = "BLOCK" if f.blocking else "warn "
        who = "ours: remove it" if f.owner == "ours" else "not ours: wait, never kill"
        print(f"  {i}. [{tag}] {f.headline}  ({who})", file=stream)
        for line in f.detail.splitlines():
            print(f"       {line.strip()}", file=stream)
    if hard:
        print("  Clear these and retry, or set PION_GATE_FORCE=1 to run anyway (then it "
              "is not a gate pass). Never move a gate floor.", file=stream)


def preflight(wait=0.0, allow_pion=False, max_load=MAX_LOAD1, stream=sys.stderr):
    """Run the checks and print the report. True if the box is clean.

    With `wait`, re-check every 10 s while every blocking finding is transient.
    A leaked server never clears on its own, so it fails immediately.
    """
    if os.environ.get("PION_GATE_FORCE") == "1":
        print("preflight: SKIPPED (PION_GATE_FORCE=1). This run is not a gate pass.",
              file=stream)
        return True
    deadline = time.monotonic() + wait
    while True:
        found = run_checks(allow_pion, max_load)
        hard = [f for f in found if f.blocking]
        if not hard or not all(f.transient for f in hard) or time.monotonic() >= deadline:
            report(found, stream)
            return not hard
        print(f"preflight: waiting on {'; '.join(f.headline for f in hard)}", file=stream)
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description="Refuse to benchmark on a contaminated machine (gh #305).")
    ap.add_argument("--wait", type=float, default=0,
                    help="seconds to keep re-checking transient findings (default 0)")
    ap.add_argument("--allow-pion", action="store_true",
                    help="a running pion-server is expected (e.g. --search-only)")
    ap.add_argument("--max-load", type=float, default=MAX_LOAD1,
                    help=f"1-minute load average ceiling (default {MAX_LOAD1})")
    a = ap.parse_args()
    sys.exit(0 if preflight(a.wait, a.allow_pion, a.max_load) else 1)


if __name__ == "__main__":
    main()
