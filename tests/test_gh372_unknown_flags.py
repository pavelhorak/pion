#!/usr/bin/env python3
"""gh #372 — pion-server must REFUSE a command line it does not understand.

Before the fix every unrecognised argument was skipped without a word, so a
typo on a security or durability flag produced a server that looked healthy
and was not what the operator asked for:

    ./pion-server --requirepas secret     -> an UNAUTHENTICATED server
    ./pion-server --wal-sise 1024         -> the default WAL size
    ./pion-server --mlx-attention         -> a flag deleted months ago, "accepted"

What is asserted:
  1. an unknown flag exits 1 before listening, names the flag, and suggests the
     nearest real one (`--requirepas` -> `--requirepass`)
  2. a stray positional argument exits 1
  3. a value-taking flag with no value exits 1 and says it needs a value
     (it used to fall through as "unknown" or be silently ignored)
  4. an unparseable number (`-p abc`, `-w two`) exits 1
  5. invalid --tenant setups and cluster ports with no room for the replication
     port (+10000) exit 1 (they exited 0, or started and replicated nothing)
  6. a correct command line still starts and serves (the refusal must not
     reject what is valid — a guard needs a probe for the case it ALLOWS)
  7. source check: _known_flags() / _value_flags() in src/main.mojo list exactly
     the flags the parser matches (they drive the "did you mean" and
     "requires a value" messages, so drift makes the messages lie)

Usage: python3 tests/test_gh372_unknown_flags.py [./pion-server]
"""
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6472
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def listening(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return True
    except OSError:
        return False


def run(args, timeout=20):
    """Start the server with `args`; return (exit_code or None if still up, output)."""
    d = tempfile.mkdtemp(prefix="pion_gh372_")
    log = os.path.join(d, "out.log")
    p = subprocess.Popen([BINARY, "--no-auto-detect", "--no-auto-embed", *args],
                         cwd=d, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        p.wait(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = None
    up = listening(PORT) if rc is None else False
    if p.poll() is None:
        p.kill()
        p.wait()
    out = open(log, errors="replace").read()
    shutil.rmtree(d, ignore_errors=True)
    return rc, up, out


def refused(name, args, must_contain):
    rc, up, out = run(["-p", str(PORT), *args])
    check(f"{name}: exits 1", rc == 1, f"rc={rc} up={up}\n{out[-400:]}")
    check(f"{name}: never listened", not up)
    for s in must_contain:
        check(f"{name}: message contains {s!r}", s in out, out[-400:])


def source_check():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "main.mojo")).read()
    body = src[src.index("\ndef main():"):]
    body = body[:body.index("# gh #138: default breadcrumb paths")]
    parsed, valued = set(), set()
    for line in body.split("\n"):
        if re.match(r"\s+(el)?if ", line):
            fl = re.findall(r'args\[i\] == "(-[^"]*)"', line)
            parsed.update(fl)
            if re.search(r"i ?\+ ?1 < len\(args\)", line):
                valued.update(fl)
    parsed |= {"--help", "-h", "--version", "-v"}

    def listed(fn):
        m = re.search("def " + fn + r"\(\) -> List\[String\]:\n    return \[(.*?)\n    \]", src, re.S)
        return set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()
    known, value = listed("_known_flags"), listed("_value_flags")
    check("source: _known_flags() == flags the parser matches", known == parsed,
          f"missing {sorted(parsed - known)} extra {sorted(known - parsed)}")
    check("source: _value_flags() == flags that take a value", value == valued,
          f"missing {sorted(valued - value)} extra {sorted(value - valued)}")


def main():
    source_check()
    if not os.path.exists(BINARY):
        print(f"binary not found: {BINARY}")
        sys.exit(1)
    if listening(PORT):
        print(f"port {PORT} already in use")
        sys.exit(1)
    print(f"[gh #372] {BINARY}")

    refused("unknown flag", ["--this-flag-does-not-exist"],
            ["FATAL", "--this-flag-does-not-exist"])
    refused("typo of --requirepass", ["--requirepas", "secret"],
            ["--requirepas", "did you mean --requirepass"])
    refused("typo of --independent-workers", ["-w", "1", "--independant-workers"],
            ["did you mean --independent-workers"])
    refused("removed flag --mlx-attention", ["--mlx-attention"], ["--mlx-attention"])
    refused("stray positional", ["6380"], ["6380"])
    refused("value flag without a value", ["--requirepass"],
            ["--requirepass", "requires a value"])
    refused("unparseable port", ["-w", "1", "-p", "abc"], ["abc"])
    refused("unparseable worker count", ["-w", "two"], ["two"])
    # Refusals that exited 0 (a bare `return` from main), so a supervisor or
    # a CI step read them as a successful start
    refused("--tenant without --requirepass", ["-p", str(PORT), "--tenant", "acme=pw"],
            ["--tenant requires --requirepass"])
    refused("invalid --tenant argument", ["-p", str(PORT), "--requirepass", "x", "--tenant", "nopassword"],
            ["invalid --tenant argument"])
    # A cluster node replicates on port + 10000: one above 55535 used to start
    # and replicate nothing
    refused("cluster port with no room for replication", ["-p", "60001", "--cluster"],
            ["port + 10000", "60001"])
    refused("replica of a primary with no room for replication",
            ["-p", str(PORT), "--cluster", "--cluster-replica", "--cluster-primary-host", "127.0.0.1",
             "--cluster-primary-port", "60001"], ["port + 10000", "60001"])

    # The valid case — several value-taking and boolean flags together.
    rc, up, out = run(["-p", str(PORT), "-w", "1", "--no-crash-log", "--wal-size", "64",
                       "--profile", "kv"], timeout=4)
    check("valid command line starts and listens", rc is None and up, f"rc={rc}\n{out[-400:]}")

    print(f"\n{'FAILED' if FAIL else 'PASSED'}: {len(FAIL)} failure(s)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
