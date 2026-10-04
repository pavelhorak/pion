#!/usr/bin/env python3
"""A source build finds its Metal shader library from any working directory.

`pixi run build` leaves ./pion-server at the repo root and the shader library at
src/ffi/metal_compute.metallib. The search in src/ffi/metal_wrap.m (gh #281)
looked next to the binary, in ../lib, ../share/pion and ../src/ffi relative to
it, and in src/ffi relative to the WORKING directory. That last entry is the
only one a checkout layout matches, so a source build started from a data
directory found no library. `--metal-attention` then fell back to the MLX
bridge, and the log said `Metal Attn: NOT ACTIVE`.

The server starts in a fresh temporary directory and must report
`Metal Attn: ACTIVE (<repo>/src/ffi/metal_compute.metallib)`.

Usage: python3 tests/test_metallib_discovery.py [./pion-server]
"""
import os, re, shutil, socket, subprocess, sys, tempfile, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
LIB = os.path.join(REPO, "src", "ffi", "metal_compute.metallib")
PORT = 19281


def main() -> int:
    if sys.platform != "darwin" or not os.path.exists(LIB):
        print("SKIP: needs macOS and a built src/ffi/metal_compute.metallib")
        return 0
    if os.path.dirname(BINARY) != REPO:
        print(f"SKIP: {BINARY} is not a source build at the repo root")
        return 0

    work = tempfile.mkdtemp(prefix="metallib_discovery_")
    log_path = os.path.join(work, "server.log")
    env = {k: v for k, v in os.environ.items() if k != "PION_METAL_LIB"}
    proc = subprocess.Popen(
        [BINARY, "-p", str(PORT), "-w", "1", "--kvcache", "--metal-attention",
         "--no-auto-detect", "--no-auto-embed", "--no-crash-log"],
        cwd=work, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    try:
        verdict = None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and verdict is None:
            text = open(log_path, errors="replace").read()
            m = re.search(r"Metal Attn: (ACTIVE|NOT ACTIVE)[^\n]*", text)
            if m:
                verdict = m.group(0)
            elif proc.poll() is not None:
                break
            else:
                time.sleep(0.25)
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=2).close()
            up = True
        except OSError:
            up = False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10)
        shutil.rmtree(work, ignore_errors=True)

    # Compare files, not strings: the disk may be case-insensitive.
    found = re.search(r"ACTIVE \((.+)\)", verdict or "")
    ok = (found is not None and os.path.exists(found.group(1))
          and os.path.samefile(found.group(1), LIB) and up)
    print(f"  {'PASS' if ok else 'FAIL'}  started from a temporary directory: {verdict or 'no Metal Attn line'}"
          + ("" if up else " (server not answering)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
