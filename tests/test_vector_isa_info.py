#!/usr/bin/env python3
"""INFO and --version say which build of the vector routines runs (#26).

WHY
The x86-64 vector library carries an AVX-512 VNNI build and an x86-64-v2 build
and picks one at startup; PION_VECTOR_VNNI=0 forces x86-64-v2. INFO and
--version named only the library and its ABI, so a benchmark log or a bug
report could not tell which build had run, and the two differ by ~20% in
vector QPS.

CHECKS
  1. `--version` and INFO's `pion_vector:` line both end in `isa=<name>`, and
     agree.
  2. The name fits the platform: vnni / x86-64-v2 / avx2 on x86-64, neon on
     ARM64.
  3. With the closed library on x86-64, PION_VECTOR_VNNI=0 reports x86-64-v2:
     the line reports the build the process chose, not the archive.

    python3 tests/test_vector_isa_info.py [--binary pion-server]
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = 6435
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def isa_of(line):
    m = re.search(r"\bisa=([A-Za-z0-9_-]+)\s*$", line.strip())
    return m.group(1) if m else None


def info_line(binary, env):
    work = tempfile.mkdtemp(prefix="isa_")
    proc = subprocess.Popen([binary, "-p", str(PORT), "-w", "1", "--no-auto-detect",
                             "--no-auto-embed", "--no-crash-log", "--no-wal"],
                            cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        wait_ready(PORT, 60, proc=proc)
        with Conn(PORT, timeout=10) as c:
            info = c.cmd("INFO").decode()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        try:
            wait_port_free(PORT)
        except RuntimeError:
            pass
        shutil.rmtree(work, ignore_errors=True)
    return next((l for l in info.splitlines() if l.startswith("pion_vector:")), "")


def main() -> int:
    binary = os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server"))
    ver = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30).stdout
    vline = next((l for l in ver.splitlines() if l.startswith("vector:")), "")
    print(f"  --version: {vline}")
    iline = info_line(binary, dict(os.environ))
    print(f"  INFO:      {iline}")
    v_isa, i_isa = isa_of(vline), isa_of(iline)
    check("--version names the ISA", v_isa is not None, vline)
    check("INFO names the ISA", i_isa is not None, iline)
    check("they agree", v_isa == i_isa, f"{v_isa} vs {i_isa}")
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        check("an x86-64 ISA", i_isa in ("vnni", "x86-64-v2", "avx2"), str(i_isa))
        if "closed" in iline:
            forced = isa_of(info_line(binary, dict(os.environ, PION_VECTOR_VNNI="0")))
            check("PION_VECTOR_VNNI=0 reports x86-64-v2", forced == "x86-64-v2", str(forced))
    else:
        check("an ARM64 ISA", i_isa == "neon", str(i_isa))
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
