"""The `cloud` smart profile must not enable naive INT4 — found 2026-09-23.

`_apply_smart_profile` picks `cloud` for any box with >= 16 online cores, and
since the first smart-config commit that branch set `vector.use_int4 = True`.
V13 abandoned naive INT4 (recall 0.74) and fixed only the DESKTOP branch. The
cloud half was worse than low recall: INT4 stores dim/2-byte rows while
`_beam_search_1536` reads dim-byte INT8 rows, so the first FT.SEARCH after
FT.OPTIMIZE SIGSEGV'd the worker — on exactly the machines a stranger reaches
for first (a 16-vCPU cloud VM, an 8C/16T EPYC, a 16-core M-series Max).

It survived because no test box had 16 cores: this Mac has 10, so every gate
ran the desktop profile. This test FAKES the core count by interposing
`sysconf(_SC_NPROCESSORS_ONLN)` (DYLD_INSERT_LIBRARIES on macOS, LD_PRELOAD on
Linux) — so the profile a big box gets is testable on a small one.

Two gotchas it encodes:
  * macOS SIP strips DYLD_* when exec'ing /bin/sh, so the server is exec'd
    DIRECTLY (no shell=True) or the shim silently never loads.
  * It asserts the banner says `Profile: cloud` before trusting anything —
    a shim that did not load would pass vacuously on the desktop profile.

    python3 tests/test_cloud_profile_int4.py [--binary pion-server]
"""
from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time

try:
    import numpy as np
    import redis
except ImportError as e:  # pragma: no cover
    print(f"SKIP: needs numpy + redis ({e})")
    sys.exit(0)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = 6421
D = 1536          # _beam_search_1536 is the kernel that crashed
N = 2000

SHIM_MAC = r"""
#include <unistd.h>
#include <stdlib.h>
static long fake_sysconf(int name) {
    if (name == _SC_NPROCESSORS_ONLN) { const char *v = getenv("FAKE_NCPU"); if (v) return atol(v); }
    return sysconf(name);
}
__attribute__((used)) static struct { const void *r; const void *o; } interposers[]
  __attribute__((section("__DATA,__interpose"))) = { { (const void*)fake_sysconf, (const void*)sysconf } };
"""
SHIM_LINUX = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
long sysconf(int name) {
    static long (*real)(int) = 0;
    if (!real) real = (long (*)(int))dlsym(RTLD_NEXT, "sysconf");
    if (name == _SC_NPROCESSORS_ONLN) { const char *v = getenv("FAKE_NCPU"); if (v) return atol(v); }
    return real(name);
}
"""

failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def build_shim(work):
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        print("SKIP: no C compiler to build the sysconf shim")
        sys.exit(0)
    mac = platform.system() == "Darwin"
    src = os.path.join(work, "fakecores.c")
    lib = os.path.join(work, "libfakecores." + ("dylib" if mac else "so"))
    with open(src, "w") as f:
        f.write(SHIM_MAC if mac else SHIM_LINUX)
    cmd = [cc, "-O2", "-dynamiclib" if mac else "-shared", "-fPIC", src, "-o", lib]
    if not mac:
        cmd.append("-ldl")
    subprocess.run(cmd, check=True)
    return lib, ("DYLD_INSERT_LIBRARIES" if mac else "LD_PRELOAD")


def main() -> int:
    binary = os.environ.get("PION_BIN", "pion-server")
    if "--binary" in sys.argv:
        binary = sys.argv[sys.argv.index("--binary") + 1]
    work = tempfile.mkdtemp(prefix="cloud_int4_")
    lib, var = build_shim(work)
    env = dict(os.environ, FAKE_NCPU="16", **{var: lib})
    log_path = os.path.join(work, "server.log")
    # Exec directly: through /bin/sh, SIP would strip DYLD_INSERT_LIBRARIES.
    proc = subprocess.Popen([os.path.join(ROOT, binary), "-p", str(PORT), "-w", "1",
                             "--no-auto-detect", "--no-auto-embed", "--no-crash-log"],
                            cwd=work, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            try:
                socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
                break
            except OSError:
                if proc.poll() is not None:
                    break
                time.sleep(0.25)
        banner = open(log_path).read()

        print("[1] the shim took: the server believes it has 16 cores")
        if not check("banner says Profile: cloud", "Profile:    cloud" in banner,
                     "shim did not load — every later check would be vacuous"):
            return 1

        print("[2] the cloud profile does not enable naive INT4")
        check("banner says INT4: False", "INT4:       False" in banner)

        print(f"[3] FT.CREATE / {N} HSET / FT.OPTIMIZE / FT.SEARCH at D={D} survives")
        r = redis.Redis(port=PORT, socket_timeout=60)
        rng = np.random.default_rng(3)
        vecs = rng.normal(size=(N, D)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        r.execute_command("FT.CREATE", "idx", "SCHEMA", "vec", "VECTOR", "HNSW", "6",
                          "TYPE", "FLOAT32", "DIM", str(D), "DISTANCE_METRIC", "L2")
        pipe = r.pipeline(transaction=False)
        for i in range(N):
            pipe.execute_command("HSET", f"d{i}", "vec", vecs[i].tobytes())
        pipe.execute()
        r.execute_command("FT.OPTIMIZE", "idx")
        hits = 0
        probes = list(range(0, N, N // 20))
        for i in probes:
            try:
                res = r.execute_command("FT.SEARCH", "idx", "*=>[KNN 1 @vec $B]",
                                        "PARAMS", "2", "B", vecs[i].tobytes(), "DIALECT", "2")
            except (redis.ConnectionError, redis.TimeoutError) as e:
                check("FT.SEARCH answers", False, repr(e))
                break
            hits += int(len(res) > 1 and res[1].decode() == f"d{i}")
        time.sleep(0.5)
        check("server still alive after searching", proc.poll() is None,
              f"exit {proc.returncode}: " + open(log_path).read()[-400:])
        check(f"self-match top-1 on {len(probes)} probes", hits == len(probes), f"{hits}/{len(probes)}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        # The server leaves a 256 MB WAL segment behind; ten runs is 2.5 GB.
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
