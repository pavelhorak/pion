#!/usr/bin/env python3
"""Gate: no out-of-line Mojo call may receive a pointer into the caller's stack.

gh #349 / #384: Mojo emits an out-of-line call as LLVM `tail call` when its
pointer arguments carry an untracked origin, and `tail` promises the callee
never touches the caller's allocas. At -O3 LLVM then deletes the caller's
stores into a stack buffer the callee reads (and the callee's writes into one
the caller reads back). Still live on Mojo 1.1.0 and the 2026-09-28 nightly.
Whether a hit misbehaves depends on inlining decisions, so the tree working
today proves nothing. `tools/audit_tail_alloca.py` reads the unoptimized IR,
where every call LLVM may keep out of line is visible.

Two halves, both required:
  1. CANARIES. The audit must FIND each shape in tools/mojo_repros/ (real
     Mojo output) and the hand-written IR shapes below (Int address, spill).
     An audit whose patterns drifted from the IR Mojo emits would report 0
     on anything, and a 0 from it would mean nothing.
  2. THE TREE. The server (and, when held/ is present, the closed vector
     library) must report 0.

    python3 tests/test_audit_tail_alloca.py
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(REPO, "tools", "audit_tail_alloca.py")

# Shapes Mojo emits only sometimes (whether a given call carries `tail` depends
# on the surrounding code), so they are pinned as hand-written IR: each must be
# flagged, which tests the audit's tracking rather than Mojo's codegen.
IR_CANARIES = {
    # address carried as an Int: UnsafePointer(unsafe_from_address=Int(p))
    "ptrtoint": """define internal void @"m::main()"() {
  %1 = alloca [2 x i64], align 8
  %2 = ptrtoint ptr %1 to i64
  %3 = tail call i64 @"m::sum_addr(Int)"(i64 %2)
  ret void
}
""",
    # -O0 spills a local pointer and reloads it before the call (the COPY of a
    # vector set, container_free.deep_clone, reached `VectorSet.add` this way)
    "spill": """define internal void @"m::main()"() {
  %1 = alloca ptr, i64 1, align 8
  %2 = alloca { ptr, i64, i64 }, align 8
  store ptr %2, ptr %1, align 8
  %3 = load ptr, ptr %1, align 8
  %4 = tail call i64 @"m::consume(Ptr)"(ptr %3)
  ret void
}
""",
}


def emit_ir(src, out, flags=()):
    cmd = ["pixi", "run", "mojo", "build", "-I", ".", *flags, "--emit", "llvm", src, "-o", out]
    cp = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if cp.returncode != 0:
        tail = "\n".join((cp.stdout + cp.stderr).strip().splitlines()[-8:])
        sys.exit(f"FAIL: could not emit IR for {src}\n{tail}")


def audit(ll):
    cp = subprocess.run([sys.executable, AUDIT, ll], capture_output=True, text=True)
    return cp.returncode, cp.stdout.strip()


def main():
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, ir in IR_CANARIES.items():
            ll = os.path.join(tmp, name + ".ll")
            with open(ll, "w") as f:
                f.write(ir)
            rc, out = audit(ll)
            if rc == 0:
                print(f"FAIL IR canary {name}: the audit no longer tracks this shape")
                failures += 1
            else:
                print(f"ok   IR canary {name}")
        canaries = [
            "tools/mojo_repros/tail_call_drops_stack_stores.mojo",
            "tools/mojo_repros/short_string_ptr_tail_call.mojo",
        ]
        for src in canaries:
            ll = os.path.join(tmp, os.path.basename(src) + ".ll")
            emit_ir(src, ll)
            rc, out = audit(ll)
            name = os.path.basename(src)
            if rc == 0:
                print(f"FAIL canary {name}: audit found nothing — its patterns no longer match Mojo's IR")
                failures += 1
            else:
                print(f"ok   canary {name}: {out.splitlines()[-1]}")

        targets = [("src/main.mojo", ["-D", "PION_HELD_VECTOR"])]
        if os.path.exists(os.path.join(REPO, "held", "pion_vector.mojo")):
            targets.append(("held/pion_vector.mojo", ["-D", "ASSERT=none"]))
        else:
            print("note held/ absent (export tree): auditing the server only")
            targets[0] = ("src/main.mojo", [])
        for src, flags in targets:
            ll = os.path.join(tmp, os.path.basename(src) + ".ll")
            emit_ir(src, ll, flags)
            rc, out = audit(ll)
            if rc != 0:
                print(f"FAIL {src}:\n{out}")
                failures += 1
            else:
                print(f"ok   {src}: {out.splitlines()[-1]}")

    if failures:
        sys.exit(f"\n{failures} failure(s). A stack pointer may cross only into an "
                 f"@always_inline Mojo function or into C (gh #349).")
    print("\nPASS")


if __name__ == "__main__":
    main()
