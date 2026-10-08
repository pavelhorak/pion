#!/usr/bin/env python3
"""Run the test suite from tests/manifest.toml — every test file, classified.

WHY
Until 2026-09-26 the gate ran 16 of 180 test files by name. The other 164
ran when someone remembered, so a regression in their area passed the gate
by construction — and the tests that did run had ways to pass without
checking anything. This runner closes the
structural half:

  * EVERY test file — tests/*.py, tests/*.mojo and each package's tests/ —
    must appear in the manifest (or in [not_tests]). An unclassified file
    FAILS the run: a new test cannot silently stay out.
  * It kills every pion-server* process between tests and owns port 1974
    while it runs. Do not run it on a box serving anything you want kept.
  * Tests declare what they need (`requires`). Unmet requirements SKIP with
    the reason printed; `--require-all` turns every such skip into a failure
    (for a provisioned machine, where "not installed" is itself the bug).
  * The runner owns the server for tests that need one: a fresh
    `pion-server <flags>` per test, state files cleared. AFTER the test, the
    server log is scanned for a fault (SIGSEGV/SIGBUS/abort) and the server is
    PINGed: a test that "passes" while the server died FAILS. That is how the
    FT.CREATE heap corruption (#364) hid — the one test that triggered it was
    never run, and nothing watched the server.
  * A pion-server left running by a test FAILS that test (leaked servers cost
    ~30% on the KV gate's write rows).
  * `xfail = "gh #N"` marks a known bug: it must fail. If it passes, that is
    XPASS and it FAILS the run — remove the marker on purpose.

USAGE
  python3 tests/run_all.py                      # tier "gate" (every push)
  python3 tests/run_all.py --tier full          # everything runnable here
  python3 tests/run_all.py --only gh232 wrongtype    # substring filter
  python3 tests/run_all.py --list               # show classification, run nothing
  python3 tests/run_all.py --binary ./pion-server --report out.json

Exit 0 only if nothing FAILED (skips are listed, not hidden).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "tests", "manifest.toml")
PORT = 1974          # many tests hardcode it; the runner owns it for their duration
FAULT_MARKERS = ("SIGSEGV", "SIGBUS", "SIGABRT", "SIGILL", "SIGFPE", "ABORT:")
STATE_GLOBS = ("pion.wal.*", "pion.hnsw.*", "pion.snapshot.*", "pion.blob.*",
               "pion.vstore.wal.*", "pion.ssm.wal.*", "pion-1974.status", "pion-1974.crash.log")


# ── requirements ────────────────────────────────────────────────────────────
def _port_open(port, host="127.0.0.1"):
    try:
        socket.create_connection((host, port), timeout=0.5).close()
        return True
    except OSError:
        return False


def _ollama_model(name):
    """True if Ollama serves `name`. Starts `ollama serve` when the binary is
    installed but not running — the old /gate did, and without it Gate 2b
    (test_ai_gateway) would skip on any box where Ollama is not already up."""
    if not shutil.which("ollama"):
        return False
    def listed():
        try:
            cp = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            return cp.returncode == 0, name in cp.stdout
        except (OSError, subprocess.TimeoutExpired):
            return False, False
    up, has = listed()
    if not up:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        for _ in range(40):
            time.sleep(0.5)
            up, has = listed()
            if up:
                break
    return has


def _py(mod):
    return subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode == 0


def _hf_cached(repo_id):
    base = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.isdir(os.path.join(base, "models--" + repo_id.replace("/", "--")))


def requirement_met(req: str) -> tuple[bool, str]:
    """Each requirement is `kind` or `kind:arg`. Returns (met, reason)."""
    kind, _, arg = req.partition(":")
    if kind == "linux":
        return sys.platform.startswith("linux"), "Linux only"
    if kind == "macos":
        return sys.platform == "darwin", "macOS only"
    if kind == "cuda":
        return shutil.which("nvcc") is not None, "needs CUDA (nvcc)"
    if kind == "ollama":
        return _ollama_model(arg or "nomic-embed-text"), f"needs Ollama model {arg or 'nomic-embed-text'}"
    if kind == "py":
        return _py(arg), f"needs Python module {arg}"
    if kind == "hf":
        return _hf_cached(arg), f"needs Hugging Face model {arg} cached"
    if kind == "bin":
        return shutil.which(arg) is not None, f"needs `{arg}` on PATH"
    if kind == "file":
        return os.path.exists(os.path.join(REPO, arg)), f"needs {arg}"
    if kind == "env":
        return bool(os.environ.get(arg)), f"needs ${arg}"
    return False, f"unknown requirement {req!r}"


# ── server lifecycle ────────────────────────────────────────────────────────
def pion_pids():
    out = subprocess.run(["ps", "-Ao", "pid=,comm="], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and os.path.basename(parts[1].strip()).startswith("pion-server"):
            pids.append(int(parts[0]))
    return pids


def redis_pids():
    """redis-server rewrites its process title (`redis-server *:6399`), so
    match the command line, not the exact name — `pgrep -x` finds nothing."""
    out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and os.path.basename(parts[1].split()[0]).startswith("redis-server"):
            pids.append(int(parts[0]))
    return pids


def kill_all_servers():
    for pid in pion_pids():
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    time.sleep(0.3)


# Files OUTSIDE the repo a server leaves behind. A stale inference socket made
# the next server's sidecar look "ready" in 0.01 s (test_gh282 failed only when
# it ran after test_auto_embed) — isolation has to cover these too.
FOREIGN_STATE = ("/tmp/pion_inference.sock",)


def clear_state():
    for f in FOREIGN_STATE:
        try:
            os.remove(f)
        except OSError:
            pass
    for pat in STATE_GLOBS:
        for f in glob.glob(os.path.join(REPO, pat)):
            try:
                os.remove(f)
            except OSError:
                pass
    # gh #468: V.EXPORT's files, written beside the server's state.
    shutil.rmtree(os.path.join(REPO, "pion-export"), ignore_errors=True)


def start_server(binary, flags, log_path):
    clear_state()
    log = open(log_path, "w")
    cmd = [binary, "-p", str(PORT), "--no-crash-log"] + shlex.split(flags)
    if "--no-auto-detect" not in flags and "--flare" not in flags and "--inference" not in flags:
        cmd.append("--no-auto-detect")
    if not any(f in flags for f in ("--no-auto-embed", "--inference", "--nle-embed", "--auto-embed")):
        cmd.append("--no-auto-embed")   # opt back in with --auto-embed in `server`
    p = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    # Ready means ANSWERING, not accepting: the server listens before it
    # initialises (so clients queue instead of being refused), and a server
    # that dies during init after listen() used to hand the test a socket
    # that never replied. Same rule as every restart test (wait for PING).
    deadline = time.time() + 60
    while time.time() < deadline:
        if p.poll() is not None:
            return p, "server exited during startup"
        if _pong(PORT, _password(flags)):
            return p, None
        time.sleep(0.2)
    return p, "server did not answer PING within 60 s"


def _pong(port, pw=None):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=2)
    except OSError:
        return False
    try:
        if pw:   # a password-protected server answers NOAUTH to a bare PING
            s.sendall(b"*2\r\n$4\r\nAUTH\r\n$%d\r\n%s\r\n" % (len(pw), pw.encode()))
            s.recv(64)
        s.sendall(b"*1\r\n$4\r\nPING\r\n")
        return s.recv(16).startswith(b"+PONG")
    except OSError:
        return False
    finally:
        s.close()


def _password(flags):
    toks = shlex.split(flags or "")
    return toks[toks.index("--requirepass") + 1] if "--requirepass" in toks[:-1] else None


def server_fault(log_path, proc, flags=""):
    text = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""
    hit = next((m for m in FAULT_MARKERS if m in text), None)
    if hit:
        tail = "\n".join(text.splitlines()[-8:])
        return f"server fault ({hit}):\n{tail}"
    if proc.poll() is not None:
        return f"server exited (code {proc.returncode}) while the test ran"
    if not _pong(PORT, _password(flags)):
        return "server did not answer PING after the test"
    return None


def _size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ── running ─────────────────────────────────────────────────────────────────
def _logname(t):
    """One log per manifest ENTRY: a file listed twice (e.g. once per --quant)
    must not overwrite its own logs."""
    base = os.path.basename(t["file"])
    if t.get("args"):
        base += "." + "".join(ch if ch.isalnum() else "_" for ch in t["args"])[:60]
    return base


def run_one(t, binary, logdir, require_all):
    name = t["file"]
    logname = _logname(t)
    reqs = t.get("requires", [])
    for r in reqs:
        met, why = requirement_met(r)
        if not met:
            if require_all:
                return "FAIL", f"requirement unmet under --require-all: {why}", 0.0
            return "SKIP", why, 0.0
    kill_all_servers()
    mode = t.get("server", "default")
    log_path = os.path.join(logdir, logname + ".server.log")
    proc = None
    if mode not in ("self", "none"):
        flags = "" if mode == "default" else mode
        proc, err = start_server(binary, flags, log_path)
        if err:
            proc.kill()
            return "FAIL", err, 0.0
    shell = False
    if t.get("cmd"):
        # Special invocations (link lines, pixi tasks, a fresh export tree).
        cmd, shell = t["cmd"], True
    elif name.endswith(".mojo"):
        # Build, then run: a test that no longer COMPILES is a failure, and
        # seven of them had silently stopped compiling since Mojo 1.0.
        exe = os.path.join(logdir, logname[:-5] if logname.endswith(".mojo") else logname)
        cmd = (f"pixi run mojo build -I . {' '.join(t.get('mojo_flags', []))} {shlex.quote(name)} "
               f"-o {shlex.quote(exe)} && {shlex.quote(exe)}")
        shell = True
    else:
        cmd = [sys.executable, name] + shlex.split(t.get("args", ""))
    # Isolation for what a test leaves behind (rule 11): a private TMPDIR that
    # is deleted afterwards, and any new /tmp/pion* entry is removed and
    # reported. Before this, one full run leaked 4.6 GB of server temp dirs
    # (each holds a 256 MB WAL) and degraded every benchmark after it.
    test_tmp = os.path.join(logdir, "tmp-" + logname)
    os.makedirs(test_tmp, exist_ok=True)
    tmp_before = set(glob.glob("/tmp/pion*"))
    redis_before = set(redis_pids())
    env = dict(os.environ, PION_BIN=binary, PYTHONUNBUFFERED="1", TMPDIR=test_tmp)
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, shell=shell,
                            timeout=t.get("timeout", 120))
        rc, out = cp.returncode, cp.stdout + cp.stderr
    except subprocess.TimeoutExpired as e:
        rc = "timeout"
        out = (e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
    dt = time.time() - t0
    with open(os.path.join(logdir, logname + ".out"), "w") as f:
        f.write(out)
    fault = server_fault(log_path, proc, "" if mode == "default" else mode) if proc else None
    if proc:
        proc.kill()
        proc.wait()
    leaked = pion_pids()
    kill_all_servers()
    # An oracle a test started and forgot costs ~56% on the KV gate's MSET row.
    leaked_redis = sorted(set(redis_pids()) - redis_before)
    for pid in leaked_redis:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    clear_state()
    # #27: keep what a test's own servers logged before the cleanup below
    # deletes it. The warm-restart and ssm tests start their servers in a temp
    # directory, and their logs went with it, so a Linux failure in those
    # tests left nothing to read. Text logs only, never a WAL or a snapshot.
    _save_logs([test_tmp] + sorted(set(glob.glob("/tmp/pion*")) - tmp_before - set(FOREIGN_STATE)),
               os.path.join(logdir, logname + ".logs"))
    litter = 0
    for f in set(glob.glob("/tmp/pion*")) - tmp_before - set(FOREIGN_STATE):
        litter += _size(f)
        shutil.rmtree(f, ignore_errors=True) if os.path.isdir(f) else _rm(f)
    litter += _size(test_tmp)
    shutil.rmtree(test_tmp, ignore_errors=True)
    if litter > 50 * 1024 * 1024:
        print(f"    note: {name} left {litter // (1024 * 1024)} MB of temp files behind (cleaned)")
    tail = "\n".join(out.strip().splitlines()[-6:])
    if rc == "timeout":
        verdict, why = "FAIL", f"timeout after {t.get('timeout', 120)} s\n{tail}"
    elif fault:
        verdict, why = "FAIL", fault
    elif rc != 0:
        verdict, why = "FAIL", f"exit {rc}\n{tail}"
    elif leaked:
        # Any mode: the runner's own server was killed and reaped above, so
        # whatever is still alive was started by the test and not stopped.
        verdict, why = "FAIL", f"left pion-server running (pids {leaked})"
    elif leaked_redis:
        verdict, why = "FAIL", f"left redis-server running (pids {leaked_redis})"
    else:
        verdict, why = "PASS", ""
    if t.get("xfail"):
        if verdict == "FAIL":
            return "XFAIL", f"{t['xfail']}: {why.splitlines()[0] if why else ''}", dt
        return "FAIL", f"XPASS — {t['xfail']} looks fixed; remove the xfail marker", dt
    return verdict, why, dt


def _save_logs(roots, dest, max_bytes=8 << 20):
    """Copy the text logs under `roots` (files ending .log, .out, .status or
    .txt, or named `log`) into `dest`, each at most `max_bytes`."""
    for root in roots:
        paths = [root] if os.path.isfile(root) else [
            os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
        for path in paths:
            name = os.path.basename(path)
            if not (name == "log" or name.endswith((".log", ".out", ".status", ".txt"))):
                continue
            try:
                if os.path.getsize(path) > max_bytes:
                    continue
                os.makedirs(dest, exist_ok=True)
                flat = os.path.relpath(path, os.path.dirname(root) if os.path.isdir(root) else
                                       os.path.dirname(path)).replace(os.sep, "__")
                shutil.copyfile(path, os.path.join(dest, flat))
            except OSError:
                pass


def discover():
    """Every test file the repo holds: tests/ plus each package's own tests/
    (pion-exo, pion-lmcache, pion-serve, pion-vllm-mlx, …)."""
    pats = ["tests/*.py", "tests/*.mojo", "pion-*/tests/*.py", "pion_*/tests/*.py", "mcp/tests/*.py"]
    files = sorted({f for pat in pats for f in glob.glob(os.path.join(REPO, pat))})
    return [os.path.relpath(f, REPO) for f in files]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run tests/manifest.toml")
    ap.add_argument("--tier", default="gate", choices=["gate", "full"])
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--binary", default=os.path.join(REPO, "pion-server"))
    ap.add_argument("--require-all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report")
    args = ap.parse_args()
    binary = os.path.abspath(args.binary)

    manifest = tomllib.load(open(MANIFEST, "rb"))
    tests = manifest["test"]
    listed = {t["file"] for t in tests}
    unclassified = [f for f in discover() if f not in listed and f not in manifest.get("not_tests", {}).get("files", [])]
    missing = [t["file"] for t in tests if not os.path.exists(os.path.join(REPO, t["file"]))]

    tiers = {"gate": ["gate"], "full": ["gate", "full"]}[args.tier]
    selected = [t for t in tests if t.get("tier", "full") in tiers
                and (not args.only or any(o in t["file"] for o in args.only))]

    if args.list:
        for t in tests:
            print(f"{t.get('tier', 'full'):6s} {t.get('server', 'default')[:24]:24s} {t['file']}"
                  + (f"  args={t['args']!r}" if t.get("args") else "")
                  + (f"  requires={t['requires']}" if t.get("requires") else "")
                  + (f"  xfail={t['xfail']}" if t.get("xfail") else ""))
        print(f"\n{len(tests)} classified, {len(unclassified)} unclassified, {len(missing)} missing")
        return 0

    logdir = os.path.join(REPO, "build", "test-logs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    if not os.path.exists(binary):
        print(f"FATAL: no server binary at {binary}")
        return 2
    print(f"run_all: tier={args.tier} tests={len(selected)} binary={binary}\nlogs: {logdir}\n")
    results = []
    for t in selected:
        verdict, why, dt = run_one(t, binary, logdir, args.require_all)
        label = t["file"] + (f" [{t['args']}]" if t.get("args") and sum(x["file"] == t["file"] for x in tests) > 1 else "")
        results.append({"file": label, "verdict": verdict, "why": why, "sec": round(dt, 1)})
        print(f"  {verdict:5s} {dt:6.1f}s  {label}" + (f"  — {why.splitlines()[0]}" if why and verdict != "PASS" else ""),
              flush=True)

    counts = {v: sum(1 for r in results if r["verdict"] == v) for v in ("PASS", "FAIL", "SKIP", "XFAIL")}
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped, {counts['XFAIL']} known-bug")
    for r in results:
        if r["verdict"] == "FAIL":
            print(f"\n--- FAIL {r['file']}\n{r['why']}")
    skips = {}
    for r in results:
        if r["verdict"] == "SKIP":
            skips.setdefault(r["why"], []).append(os.path.basename(r["file"]))
    for why, files in skips.items():
        print(f"SKIP ({why}): {len(files)} — {', '.join(files[:6])}{' …' if len(files) > 6 else ''}")
    if unclassified:
        print(f"\nFAIL: {len(unclassified)} test file(s) are not in tests/manifest.toml:")
        for f in unclassified:
            print(f"  {f}")
    if missing:
        print(f"\nFAIL: manifest names {len(missing)} file(s) that do not exist: {missing}")
    if args.report:
        json.dump({"results": results, "unclassified": unclassified, "missing": missing}, open(args.report, "w"), indent=1)
    return 1 if counts["FAIL"] or unclassified or missing else 0


if __name__ == "__main__":
    sys.exit(main())
