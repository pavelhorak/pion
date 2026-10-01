#!/usr/bin/env python3
"""gh #347: the bench cleanup must kill every pion-server* and never a build.

`pkill -9 -f 'pion-serve[r]'` matched `mojo build ... -o pion-server` and
SIGKILLed two release builds on 2026-09-25. benchmarks/bench_proc.py matches
the executable name instead. This spawns stand-ins under the real names --
pion-server, pion-server-dev, pion-server-iouring, and a `mojo` whose command
line ends in `-o pion-server` -- and asserts exactly the servers die.

Needs a C compiler (`cc`); copied system binaries do not run from another
path on macOS, so the stand-ins are built rather than copied.

Kills any REAL pion-server* running on this machine too -- same as a bench.

Usage: python3 tests/test_gh347_bench_kill.py     (exit 0 = pass)
"""
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from bench_proc import kill_pion_servers, pion_server_pids  # noqa: E402

SERVERS = ("pion-server", "pion-server-dev", "pion-server-iouring")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.c")
        with open(src, "w") as f:
            f.write("#include <unistd.h>\nint main(void){sleep(60);return 0;}\n")
        fake = os.path.join(d, "fake")
        subprocess.run(["cc", "-o", fake, src], check=True)
        for name in SERVERS + ("mojo",):
            subprocess.run(["cp", fake, os.path.join(d, name)], check=True)

        servers = [subprocess.Popen([os.path.join(d, n)]) for n in SERVERS]
        build = subprocess.Popen([os.path.join(d, "mojo"), "build",
                                  "src/main.mojo", "-o", "pion-server"])
        try:
            time.sleep(0.5)
            checks = [
                ("stand-ins started", all(p.poll() is None for p in servers + [build])),
                ("finds exactly the servers",
                 set(p.pid for p in servers) <= set(pion_server_pids())
                 and build.pid not in pion_server_pids()),
            ]
            kill_pion_servers()
            time.sleep(0.3)
            checks += [
                ("every pion-server* killed", all(p.poll() is not None for p in servers)),
                ("mojo build survives", build.poll() is None),
            ]
        finally:
            for p in servers + [build]:
                if p.poll() is None:
                    p.kill()

    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
