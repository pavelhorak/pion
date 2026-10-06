#!/usr/bin/env python3
"""Every dispatch arm pins every byte of its command name (gh #225).

tools/audit_loose_command_matchers.py must report 0. It is only worth running
if it can still find one, so a canary goes first: a temp copy of a loose arm
(prefix plus length, the shape that ran `PINX` as PING) must be reported.

    python3 tests/test_audit_loose_matchers.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOL = os.path.join(ROOT, "tools", "audit_loose_command_matchers.py")


def count(path=None):
    out = subprocess.run([sys.executable, TOOL] + ([path] if path else []),
                         capture_output=True, text=True, check=True).stdout
    m = re.match(r"(\d+) dispatch arms", out)
    return (int(m.group(1)) if m else -1), out


def main():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "fast_path.mojo"), "w") as f:
            f.write("                elif b0_lower == 112 and cmd_len == 4: # 'p' - PING\n"
                    "                    writer.append_pong_response()\n")
        n, out = count(d)
        if n != 1:
            print(f"FAIL canary: the audit did not find a planted loose arm\n{out}")
            return 1
    n, out = count()
    print(out.strip())
    if n != 0:
        print(f"FAIL: {n} dispatch arm(s) do not pin their whole command name")
        return 1
    print("PASS: 0 loose arms (canary found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
