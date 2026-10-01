#!/usr/bin/env python3
"""Every borrowed lookup key must be used safely: tools/audit_borrowed_keys.py
reports 0 (gh #394).

WHY
`GenericValue.borrow()` builds a key that points at the recv buffer or a RESP
token instead of copying it — the fix for the long-key leak on every command.
It is safe exactly as long as the value only reaches the hash map (which
copies what it keeps) or is read, and its bytes outlive it. Stored anywhere
else it is a pointer into a buffer the next read overwrites; built over a
local String's bytes it can outlive the String. The audit checks both rules at
every `GenericValue.borrow(` in src/.

A 0 from an audit that cannot see the bug is worth nothing, so this first runs
it over a scratch tree with one planted violation of each kind and requires it
to report every one — then runs it over the real tree and requires 0.

    python3 tests/test_audit_borrowed_keys.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_borrowed_keys.py"

CANARY = '''
def handle_canary(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], list_ptr: Pointer[SlabList, MutUntrackedOrigin]):
    # 1. a borrowed key STORED in a list — dangles after the next recv
    var k1 = GenericValue.borrow(tokens[unsafe_offset=i+1].ptr, tokens[unsafe_offset=i+1].length)
    list_ptr[].rpush(k1)
    # 2. borrowed over a LOCAL String's bytes — Mojo may destroy it first
    var s = String("abcdefghijklmnopqrstuvwxyz0123456789")
    var k2 = GenericValue.borrow(s.unsafe_ptr(), s.byte_length())
    _ = keyspace[].get(k2)
    # 3. a borrowed value copied into another variable the audit cannot follow
    var k3 = GenericValue.borrow(tokens[unsafe_offset=i+2].ptr, tokens[unsafe_offset=i+2].length)
    var k4 = k3
    _ = keyspace[].get(k4)
'''
EXPECTED = 3


def run(args):
    r = subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    fails = []
    d = Path(tempfile.mkdtemp(prefix="borrow_canary_"))
    try:
        (d / "common").mkdir()
        (d / "commands").mkdir()
        (d / "commands" / "canary.mojo").write_text(CANARY)
        rc, out = run(["--src", str(d)])
        found = out.count("UNSAFE BORROW")
        print(f"[canary] planted {EXPECTED}, audit found {found}")
        if found != EXPECTED or rc == 0:
            fails.append(f"canary: expected {EXPECTED} findings and a failing exit, got {found} (rc={rc})\n{out}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    rc, out = run([])
    print("[src] " + out.strip().splitlines()[-1])
    if rc != 0:
        fails.append("src/ has unsafe borrowed keys:\n" + out)
    for f in fails:
        print("FAIL", f)
    print(f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
