#!/usr/bin/env python3
"""gh #271 — FT.CREATE must honour DISTANCE_METRIC.

Pre-fix, `DISTANCE_METRIC` appeared ZERO times in `src/**/*.mojo`:
`handle_ft_create` parsed VECTOR / TEXT / TAG / NUMERIC / EF_CONSTRUCTION / DIM,
and the keyword fell through to the field-name catch-all and was dropped. `L2`
and `COSINE` therefore built and queried byte-identical indexes — a caller who
asked for one metric got the other with nothing to indicate it, the
silent-wrong-answer class of gh #229, gh #257 and gh #260.

WHICH metric they both got is the part the issue got backwards, and it matters
enough to record. The issue said "the index is always cosine",
on the strength of a kernel named `cosine_distance_int8_jit`. That kernel is a
bare negative dot product, it is imported by `hnsw.mojo` and never called, and
the beam kernels compute `query_norm + node_norm - 2*dot` — squared L2. Nothing
in the ingest path normalized. Measured on 0.982 with the construction below:
BOTH metrics returned the L2 answer. So L2 was accidentally correct all along
and COSINE was the broken one.

The fix: COSINE L2-normalizes at ingest and at query, which makes the existing
squared-L2 search compute cosine ordering exactly. L2 is unchanged. IP and any
other spelling are refused at FT.CREATE.

CONSTRUCTION — read this before editing the vectors. The original repro used
`np.full(D, i)`: colinear vectors, tied under cosine, so any ordering was
correct and the "wrong" answer was the test's fault. That misdiagnosis is what
this file exists to prevent recurring. Here the index holds N random UNIT
vectors so the quantizer's Welford calibration has a sane distribution, and two
planted candidates force the metrics apart:

    q = u                      (unit)
    A = 3u                     same direction, far in magnitude
                                 cosine dist 0.000    L2 dist 2.0
    B = u + 0.3v,  v _|_ u     near in magnitude, off direction
                                 cosine dist 0.042    L2 dist 0.3

    COSINE must rank A first.  L2 must rank B first.

Usage:
    python3 tests/test_gh271_distance_metric.py [--port 1974]
    python3 tests/test_gh271_distance_metric.py --restart --binary pion-server
        also spawns its own server and checks the metric survives a restart
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

try:
    import numpy as np
    import redis
except ImportError as e:  # pragma: no cover
    print(f"SKIP: needs numpy + redis ({e})")
    sys.exit(0)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESTART_PORT = 6412
failures: list = []
passes: list = []


def check(name, cond, detail="") -> bool:
    if cond:
        passes.append(name)
        print(f"  PASS  {name}")
    else:
        failures.append((name, detail))
        print(f"  FAIL  {name}   {detail}")
    return bool(cond)


def unit(x) -> "np.ndarray":
    return (x / np.linalg.norm(x)).astype(np.float32)


def discriminating(D, seed=7):
    """(query, A, B) constructed so cosine and L2 must disagree. See module docstring."""
    rng = np.random.default_rng(seed)
    u = unit(rng.normal(size=D))
    v = rng.normal(size=D)
    v = unit(v - np.dot(v, u) * u)
    return u, (3.0 * u).astype(np.float32), (u + 0.3 * v).astype(np.float32)


def build(r, name, D, metric, docs, drop_first=True):
    """FLUSHALL, create, ingest, optimize. `metric=None` omits the keyword."""
    if drop_first:
        r.execute_command("FLUSHALL")
    args = ["FT.CREATE", name, "SCHEMA", "vec", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", str(D)]
    if metric is not None:
        args += ["DISTANCE_METRIC", metric]
    r.execute_command(*args)
    for k, v in docs.items():
        r.execute_command("HSET", k, "vec", v.tobytes(), "title", k)
    r.execute_command("FT.OPTIMIZE", name)


def search(r, name, q, k=3):
    res = r.execute_command("FT.SEARCH", name, f"*=>[KNN {k} @vec $B]",
                            "PARAMS", "2", "B", q.tobytes(), "DIALECT", "2")
    return [res[i].decode() for i in range(1, len(res), 2)]


def ranked(r, name, D, metric, n_bg=1500, seed=7):
    """Build a discriminating index under `metric`; return the top-k key order."""
    u, A, B = discriminating(D, seed)
    rng = np.random.default_rng(seed + 1000)
    docs = {f"bg{i}": unit(rng.normal(size=D)) for i in range(n_bg)}
    docs["A"], docs["B"] = A, B
    build(r, name, D, metric, docs)
    return search(r, name, u, k=3)


# --------------------------------------------------------------------------
def section_metric_is_honoured(r):
    print("[1] L2 and COSINE must give DIFFERENT orderings on a discriminating input")
    print("    A = 3u (cosine dist 0.000, L2 dist 2.0) · B = u+0.3v (cosine 0.042, L2 0.3)")
    for D in (128, 1536):
        l2 = ranked(r, f"m_l2_{D}", D, "L2")
        cos = ranked(r, f"m_cos_{D}", D, "COSINE")
        print(f"    D={D:<5} L2 -> {l2[:2]}    COSINE -> {cos[:2]}")
        check(f"D={D}: L2 ranks B first (nearest in L2)", l2[:1] == ["B"], f"got {l2}")
        check(f"D={D}: COSINE ranks A first (nearest in direction)", cos[:1] == ["A"], f"got {cos}")
        # The one assertion that actually pins the bug: pre-fix these were equal.
        check(f"D={D}: the two metrics disagree (DISTANCE_METRIC is read)", l2[:1] != cos[:1],
              f"both returned {l2[:1]} — metric ignored, this is gh #271")


def section_default(r):
    print("\n[2] Omitting DISTANCE_METRIC keeps the pre-#271 behaviour (L2)")
    print("    The engine has always computed squared L2, so an index created")
    print("    without the keyword must not silently change meaning.")
    for D in (128, 1536):
        u, A, B = discriminating(D)
        rng = np.random.default_rng(1007)
        docs = {f"bg{i}": unit(rng.normal(size=D)) for i in range(1500)}
        docs["A"], docs["B"] = A, B
        build(r, f"m_def_{D}", D, None, docs)
        got = search(r, f"m_def_{D}", u, k=3)
        check(f"D={D}: no DISTANCE_METRIC behaves as L2", got[:1] == ["B"], f"got {got}")


def section_refusal(r):
    print("\n[3] An unsupported metric is REFUSED, as a no-op, without desyncing")
    print("    Answering an IP query with L2 ordering would be the same silent")
    print("    wrong answer this issue is about; the quantizer is affine WITH an")
    print("    offset, which cancels in a difference but not in a dot product, so")
    print("    there is no honest way to serve IP on this search.")
    D = 16
    rng = np.random.default_rng(3)
    docs = {f"d{i}": unit(rng.normal(size=D)) for i in range(50)}
    build(r, "good", D, "COSINE", docs)
    q = unit(rng.normal(size=D))
    before = search(r, "good", q, k=3)

    for bad in ("IP", "INNER_PRODUCT", "l1", ""):
        try:
            r.execute_command("FT.CREATE", f"bad{len(bad)}", "SCHEMA", "vec", "VECTOR",
                              "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(D),
                              "DISTANCE_METRIC", bad)
            check(f"DISTANCE_METRIC {bad!r} is refused", False, "returned OK")
        except redis.ResponseError as e:
            check(f"DISTANCE_METRIC {bad!r} is refused",
                  "unsupported DISTANCE_METRIC" in str(e), str(e))

    # On the pre-fix binary the bogus FT.CREATE SUCCEEDS, and because a server
    # holds one index at a time (gh #145) it displaces 'good' — so this search
    # raises rather than returning a different answer. Both are the same
    # failure: the refusal was not a no-op.
    try:
        after = search(r, "good", q, k=3)
    except redis.ResponseError as e:
        after = f"<error: {e}>"
    check("the refusal did not disturb the existing index", before == after,
          f"before={before} after={after}")
    check("the connection is still usable", r.ping())

    # Frame integrity: an arm that errors must still consume its own command
    # (gh #223 / #240). Pipelined behind the refusal, PING and ECHO must land.
    import socket

    def enc(*a):
        out = b"*%d\r\n" % len(a)
        for x in a:
            xb = x.encode() if isinstance(x, str) else x
            out += b"$%d\r\n%s\r\n" % (len(xb), xb)
        return out

    s = socket.create_connection(("127.0.0.1", r.connection_pool.connection_kwargs["port"]))
    f = s.makefile("rb")
    s.sendall(enc("FT.CREATE", "pdesync", "SCHEMA", "v", "VECTOR", "HNSW", "6", "TYPE",
                  "FLOAT32", "DIM", "8", "DISTANCE_METRIC", "IP")
              + enc("PING") + enc("ECHO", "tail"))
    got = [f.readline() for _ in range(4)]
    s.close()
    check("pipelined: refusal is exactly one reply, then PONG, then the echo",
          got[0].startswith(b"-ERR") and got[1] == b"+PONG\r\n"
          and got[2] == b"$4\r\n" and got[3] == b"tail\r\n", got)


def section_dim(r):
    print("\n[4] A DIM that disagrees with the server's dimension still WORKS")
    print("    Asserted because this issue was FILED on the opposite claim and a")
    print("    README warning shipped saying so; if someone 'fixes' DIM on the")
    print("    strength of that, this catches it.")
    for D in (8, 128, 1536, 4096):
        rng = np.random.default_rng(7)
        docs = {f"d:{i}": unit(rng.normal(size=D)) for i in range(30)}
        build(r, f"dim{D}", D, "COSINE", docs)
        got = search(r, f"dim{D}", docs["d:5"], k=1)
        check(f"DIM {D} returns the exact match first", got[:1] == ["d:5"], f"got {got}")


# --------------------------------------------------------------------------
def start_pion(port, binary, log_path):
    proc = subprocess.Popen(
        [os.path.join(ROOT, binary), "-p", str(port), "-w", "1",
         "--no-auto-detect", "--no-auto-embed"],
        cwd=ROOT, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid)
    deadline = time.time() + 40
    while time.time() < deadline:
        try:
            c = redis.Redis(port=port, socket_connect_timeout=1)
            c.ping()
            c.close()
            return proc
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"{binary} did not come up on port {port}")


def stop_pion(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def cleanup_state():
    for name in os.listdir(ROOT):
        if name.startswith(("pion.hnsw.", "pion.wal.", "pion.snapshot.", "pion.blob.")):
            try:
                os.remove(os.path.join(ROOT, name))
            except OSError:
                pass


def section_restart(binary):
    """The metric decides whether the QUERY is normalized, so a warm restart
    that forgot it would query a normalized graph with a raw query and serve
    quietly wrong results — the same shape as gh #211's BFS-permuted labels."""
    print("\n[5] The metric survives a restart (persisted in the index header)")
    D = 1536
    log = os.path.join(ROOT, "gh271_restart.log")
    cleanup_state()
    proc = start_pion(RESTART_PORT, binary, log)
    try:
        r = redis.Redis(port=RESTART_PORT)
        cold = ranked(r, "widx", D, "COSINE")
        check("cold: COSINE ranks A first", cold[:1] == ["A"], f"got {cold}")
        r.close()
    finally:
        stop_pion(proc)

    proc = start_pion(RESTART_PORT, binary, log + ".2")
    try:
        r = redis.Redis(port=RESTART_PORT)
        u, _A, _B = discriminating(D)
        warm = search(r, "widx", u, k=3)
        check("warm: COSINE ordering is preserved across restart",
              warm[:1] == ["A"], f"cold={cold} warm={warm}")
        r.close()
        # Without this the section passes vacuously the moment the warm path
        # stops loading and silently cold-rebuilds instead — same trap gh #250
        # closed by asserting a WAL rotation actually happened.
        loaded = "HNSW loaded from disk" in open(log + ".2").read()
        check("warm: the index was really LOADED, not cold-rebuilt", loaded,
              "no 'HNSW loaded from disk' in the restart log")
    finally:
        stop_pion(proc)
        cleanup_state()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--restart", action="store_true",
                    help="also spawn a server and check restart persistence")
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", "pion-server"))
    a = ap.parse_args()

    # --restart spawns its own server, so it does not need one on --port. Run it
    # first: it FLUSHes the repo's pion.* state files, which would pull the WAL
    # out from under a server already running there.
    if a.restart:
        section_restart(a.binary)

    try:
        r = redis.Redis(port=a.port)
        r.ping()
    except Exception as e:
        if a.restart:
            print(f"\n(no Pion on {a.port}; ran the restart section only — {e})")
            print(f"\n{len(passes)} passed, {len(failures)} failed")
            for name, detail in failures:
                print(f"  FAILED: {name}  {detail}")
            return 1 if failures else 0
        print(f"FATAL: no Pion on {a.port} ({e})")
        return 2

    # Each section is isolated: on a pre-fix binary an unsupported metric is
    # ACCEPTED and displaces the serving index, so a later query raises. That
    # must be reported as a failure of that section, not abort the run.
    for fn in (section_metric_is_honoured, section_default, section_refusal, section_dim):
        try:
            fn(r)
        except Exception as e:  # noqa: BLE001 - a raising section IS a failure
            failures.append((fn.__name__, f"raised {type(e).__name__}: {e}"))
            print(f"  FAIL  {fn.__name__} raised {type(e).__name__}: {e}")
    r.execute_command("FLUSHALL")

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
