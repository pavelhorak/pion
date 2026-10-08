#!/usr/bin/env python3
"""The tracked metallib must be built from the tracked shader source.

src/ffi/metal_wrap.m loads src/ffi/metal_compute.metallib before it would
compile src/ffi/metal_compute.metal at run time. So a commit that changes the
.metal without the .metallib leaves every source checkout running the OLD
kernels, with nothing to say so; with a changed kernel signature (gh #398
changed K/V from float4 to half4) it reads the wrong bytes. Release and CI
builds delete the tracked copy and rebuild it, so they never see the problem,
which is exactly why it needs a check of its own.

The rule: the newest commit that touches the .metal must also touch the
.metallib, or be older than the newest commit that does, and neither may have
an uncommitted change the other lacks. Without full Xcode, take the metallib
from CI's `metal_compute.metallib` artifact (macOS leg).
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = "src/ffi/metal_compute.metal"
LIB = "src/ffi/metal_compute.metallib"


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def main():
    if git("rev-parse", "--git-dir").returncode != 0:
        print("SKIP: not a git checkout")
        return 0
    dirty_src = git("diff", "--quiet", "HEAD", "--", SRC).returncode != 0
    dirty_lib = git("diff", "--quiet", "HEAD", "--", LIB).returncode != 0
    if dirty_src and not dirty_lib:
        print(f"FAIL: {SRC} has uncommitted changes and {LIB} does not")
        return 1
    c_src = git("log", "-1", "--format=%H", "--", SRC).stdout.strip()
    c_lib = git("log", "-1", "--format=%H", "--", LIB).stdout.strip()
    if not c_src or not c_lib:
        print("SKIP: history too shallow to compare")
        return 0
    # Fresh when the shader's newest commit is the metallib's, or an ancestor of it.
    fresh = c_src == c_lib or git("merge-base", "--is-ancestor", c_src, c_lib).returncode == 0
    if not fresh:
        print(f"FAIL: {SRC} changed in {c_src[:8]} after {LIB} was last committed ({c_lib[:8]})")
        return 1
    print(f"PASS: metallib committed at {c_lib[:8]}, shader at {c_src[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
