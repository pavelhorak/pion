#!/usr/bin/env python3
"""gh #403 — FT.INFO, FT.DROPINDEX and FT.SEARCH read the index NAME, and every
worker gives the same answer.

Before the fix `handle_ft_info` answered from the local worker's `index_ready`
and never looked at its argument:

  * `FT.INFO nosuchidx` returned the served index's metadata, so a client that
    probes for an index before creating it (RedisVL, LangChain, VectorDBBench)
    was told it existed;
  * a real index was "Unknown index name" on any worker that had not borrowed it
    yet (FT.SEARCH borrows, FT.INFO did not) — `pipelined_bench.py` aborted on
    its opening FT.INFO at `-w 10`.

Found on the way and fixed with it:

  * `FT.DROPINDEX <any name>` dropped the one index this server holds;
  * FT.SEARCH on a worker that had not borrowed yet accepted ANY index name;
  * FT.OPTIMIZE on a worker other than FT.CREATE's saved an UNNAMED index, so
    after a warm restart the index was "unknown" by name.

Every worker is reached deterministically through its affinity port
(`port + 2 + worker_id`) — an accept-race probe that happens to land on the
builder proves nothing (CLAUDE.md, gh #253).

Usage:
    python3 tests/test_gh403_ft_info_name.py [--binary ./pion-server] [--port 7403]
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, ok, detail: object = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


class Client:
    def __init__(self, port, timeout=30):
        import socket
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        # makefile('rb'), never manual slicing — see python_resp_array_slicing_trap
        self.f = self.sock.makefile("rb")

    @staticmethod
    def encode(*args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        return out

    def cmd(self, *args):
        self.sock.sendall(self.encode(*args))
        return self.read()

    def read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        tag, body = line[:1], line[1:-2]
        if tag == b"*":
            n = int(body)
            return [] if n < 0 else [self.read() for _ in range(n)]
        if tag == b"$":
            n = int(body)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if tag == b"-":
            return Exception(body.decode(errors="replace"))
        if tag == b":":
            return int(body)
        return body

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def serve(binary, port, workers, cwd, log_name):
    cmd = [binary, "--profile", "vector", "-w", str(workers), "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    if workers > 1:
        cmd.append("--independent-workers")
    log = open(os.path.join(cwd, log_name), "w")
    return subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def stop(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass


def wait_ready(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = Client(port, timeout=2)
            r = c.cmd("PING")
            c.close()
            if r == b"PONG":
                return True
        except (OSError, ConnectionError):
            pass
        time.sleep(0.3)
    return False


def is_err(r, text=None):
    return isinstance(r, Exception) and (text is None or text in str(r))


def info_docs(r):
    """num_docs from an FT.INFO reply, or None when it is not an info reply."""
    if not isinstance(r, list):
        return None
    kv = {r[i]: r[i + 1] for i in range(0, len(r) - 1, 2)}
    if kv.get(b"index_name") is None or kv.get(b"num_docs") is None:
        return None
    return int(kv[b"num_docs"])


def worker_of(c):
    info = c.cmd("INFO")
    for line in info.decode(errors="replace").split("\r\n"):
        if line.startswith("worker_id:"):
            return int(line.split(":", 1)[1])
    return -1


def knn(c, index, blob, k=5):
    return c.cmd("FT.SEARCH", index, f"*=>[KNN {k} @vec $q]", "PARAMS", "2", "q", blob)


def schema(dim, metric="L2"):
    return ["ON", "HASH", "PREFIX", "1", "doc:", "SCHEMA", "vec", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", str(dim), "DISTANCE_METRIC", metric]


def ingest(c, vecs):
    buf = b"".join(Client.encode("HSET", f"doc:{i}", "vec", v.tobytes())
                   for i, v in enumerate(vecs))
    c.sock.sendall(buf)
    return [c.read() for _ in range(len(vecs))]


# ── Phase A: single worker ───────────────────────────────────────────────────
def phase_single(binary, port, workdir):
    print("\n[A] single worker: the name is read")
    d = os.path.join(workdir, "single")
    os.makedirs(d)
    proc = serve(binary, port, 1, d, "server.log")
    try:
        if not check("server ready", wait_ready(port)):
            return
        c = Client(port)
        dim = 16
        rng = np.random.default_rng(403)
        vecs = rng.standard_normal((200, dim)).astype(np.float32)

        check("FT.INFO before any FT.CREATE is an error",
              is_err(c.cmd("FT.INFO", "idx"), "Unknown index name"))
        check("FT.DROPINDEX with no index is an error",
              is_err(c.cmd("FT.DROPINDEX", "idx"), "Unknown index name"))
        check("FT.CREATE idx", c.cmd("FT.CREATE", "idx", *schema(dim)) == b"OK")
        r = c.cmd("FT.INFO", "idx")
        check("created-but-unbuilt index exists, num_docs 0", info_docs(r) == 0, repr(r)[:120])
        check("FT.INFO nosuchidx is an error (was: idx's metadata)",
              is_err(c.cmd("FT.INFO", "nosuchidx"), "Unknown index name"))

        ok = all(x == 1 or x == 2 for x in ingest(c, vecs))
        check("ingest 200 vectors", ok)
        check("FT.OPTIMIZE idx", c.cmd("FT.OPTIMIZE", "idx") == b"OK")
        r = c.cmd("FT.INFO", "idx")
        check("FT.INFO idx num_docs 200", info_docs(r) == 200, repr(r)[:120])
        r = c.cmd("FT.INFO", "nosuchidx")
        check("FT.INFO nosuchidx still an error after the build", is_err(r, "Unknown index name"),
              repr(r)[:120])
        check("FT.INFO IDX is an error (names are case-sensitive)",
              is_err(c.cmd("FT.INFO", "IDX")))

        r = c.cmd("FT.DROPINDEX", "nosuchidx")
        check("FT.DROPINDEX nosuchidx is refused", is_err(r, "Unknown index name"), repr(r)[:120])
        r = knn(c, "idx", vecs[7].tobytes())
        check("... and idx still serves (the refused drop touched nothing)",
              isinstance(r, list) and len(r) >= 1 and r[0] == 5 and r[1] == b"doc:7",
              repr(r)[:120])
        check("FT.INFO idx still num_docs 200", info_docs(c.cmd("FT.INFO", "idx")) == 200)

        check("FT.DROPINDEX idx", c.cmd("FT.DROPINDEX", "idx") == b"OK")
        check("FT.INFO idx after the drop is an error",
              is_err(c.cmd("FT.INFO", "idx"), "Unknown index name"))

        # A served index with NO registered name (what a pre-#403 file whose
        # build ran off the FT.CREATE worker loads as; here: FT.OPTIMIZE with
        # no name after the name was dropped). FT.SEARCH has always served it
        # under any name, so FT.INFO and FT.DROPINDEX must reach it too — or
        # nothing can ever drop it.
        c.cmd("FT.CREATE", "u", *schema(dim))
        check("FT.DROPINDEX u before any build (name gone, ingest config kept)",
              c.cmd("FT.DROPINDEX", "u") == b"OK")
        buf = b"".join(Client.encode("HSET", f"doc:{1000 + i}", "vec", v.tobytes())
                       for i, v in enumerate(vecs[:100]))
        c.sock.sendall(buf)
        _ = [c.read() for _ in range(100)]
        check("FT.OPTIMIZE with no name builds an unnamed index", c.cmd("FT.OPTIMIZE") == b"OK")
        r = knn(c, "anything", vecs[3].tobytes())
        check("unnamed index: FT.SEARCH under any name serves it", isinstance(r, list) and len(r) > 1,
              repr(r)[:80])
        r = c.cmd("FT.INFO", "anything")
        check("unnamed index: FT.INFO under any name answers", info_docs(r) == 100, repr(r)[:80])
        check("unnamed index: FT.DROPINDEX can drop it", c.cmd("FT.DROPINDEX", "anything") == b"OK")
        check("... and afterwards FT.INFO is an error again",
              is_err(c.cmd("FT.INFO", "anything"), "Unknown index name"))

        # Arity errors still consume their own frame (gh #223): the PING behind
        # them must answer PONG, and nothing else may come back.
        c.sock.sendall(Client.encode("FT.INFO") + Client.encode("PING"))
        a, b = c.read(), c.read()
        check("FT.INFO with no name: arity error, then PONG",
              is_err(a, "wrong number of arguments") and b == b"PONG", f"{a!r} {b!r}")
        c.sock.sendall(Client.encode("FT.DROPINDEX") + Client.encode("PING"))
        a, b = c.read(), c.read()
        check("FT.DROPINDEX with no name: arity error, then PONG",
              is_err(a, "wrong number of arguments") and b == b"PONG", f"{a!r} {b!r}")
        c.close()
    finally:
        stop(proc)


# ── Phase B/C: four workers, then a warm restart ─────────────────────────────
DIM_WARM = 1536  # load_from_disk only restores the server's startup dim
N_WARM = 300
WORKERS = 4


def per_worker(port, fn):
    out = []
    for w in range(WORKERS):
        c = Client(port + 2 + w)
        try:
            out.append(fn(c))
        finally:
            c.close()
    return out


def phase_multi(binary, port, workdir):
    print(f"\n[B] -w {WORKERS}: every worker gives the same answer")
    d = os.path.join(workdir, "multi")
    os.makedirs(d)
    rng = np.random.default_rng(4031)
    vecs = rng.standard_normal((N_WARM, DIM_WARM)).astype(np.float32)
    proc = serve(binary, port, WORKERS, d, "server.log")
    try:
        if not check("server ready", wait_ready(port)):
            return
        ids = per_worker(port, worker_of)
        if not check("affinity port port+2+w reaches worker w", ids == list(range(WORKERS)), ids):
            return

        c0 = Client(port + 2)       # worker 0 creates
        check("FT.CREATE idx on worker 0", c0.cmd("FT.CREATE", "idx", *schema(DIM_WARM, "COSINE")) == b"OK")
        c0.close()
        r = per_worker(port, lambda c: info_docs(c.cmd("FT.INFO", "idx")))
        check("unbuilt idx exists on EVERY worker (was: only on the creator)", r == [0] * WORKERS, r)
        r = per_worker(port, lambda c: is_err(c.cmd("FT.INFO", "nosuchidx"), "Unknown index name"))
        check("FT.INFO nosuchidx is an error on every worker", all(r), r)
        r = per_worker(port, lambda c: is_err(knn(c, "nosuchidx", vecs[0].tobytes())))
        check("FT.SEARCH nosuchidx is an error on every worker (was `[0]` off the creator)", all(r), r)

        c2 = Client(port + 4)       # worker 2 ingests
        check("ingest on worker 2", all(x == 1 or x == 2 for x in ingest(c2, vecs)))
        c2.close()
        c1 = Client(port + 3)       # worker 1 builds — NOT the creator
        check("FT.OPTIMIZE idx on worker 1", c1.cmd("FT.OPTIMIZE", "idx") == b"OK")
        c1.close()

        # The issue's case: a FRESH connection whose first command is FT.INFO,
        # on every worker, before any FT.SEARCH has borrowed anything there.
        r = per_worker(port, lambda c: info_docs(c.cmd("FT.INFO", "idx")))
        check(f"FT.INFO idx before any borrow: num_docs {N_WARM} on every worker",
              r == [N_WARM] * WORKERS, r)
        r = per_worker(port, lambda c: is_err(c.cmd("FT.INFO", "nosuchidx"), "Unknown index name"))
        check("FT.INFO nosuchidx still an error on every worker", all(r), r)

        # And through the shared port, many connections opened CONCURRENTLY
        # (serially-opened ones all land on one worker).
        results, lock = [], threading.Lock()

        def probe():
            try:
                c = Client(port)
                w = worker_of(c)
                n = info_docs(c.cmd("FT.INFO", "idx"))
                c.close()
            except Exception as e:  # noqa: BLE001
                w, n = -1, repr(e)
            with lock:
                results.append((w, n))
        ts = [threading.Thread(target=probe) for _ in range(40)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        bad = [x for x in results if x[1] != N_WARM]
        check(f"40 concurrent connections: FT.INFO idx ok on all "
              f"(workers seen {sorted({w for w, _ in results})})", not bad, bad[:5])

        r = per_worker(port, lambda c: knn(c, "idx", vecs[11].tobytes()))
        check("FT.SEARCH idx answers doc:11 first on every worker",
              all(isinstance(x, list) and len(x) > 1 and x[1] == b"doc:11" for x in r),
              [repr(x)[:60] for x in r])
    finally:
        stop(proc)

    print(f"\n[C] warm restart: the saved index carries its name")
    proc = serve(binary, port, WORKERS, d, "server-warm.log")
    try:
        if not check("warm server ready", wait_ready(port)):
            return
        # The load is published by the loading worker; give it a moment.
        deadline = time.time() + 30
        r = None
        while time.time() < deadline:
            r = per_worker(port, lambda c: info_docs(c.cmd("FT.INFO", "idx")))
            if r == [N_WARM] * WORKERS:
                break
            time.sleep(0.5)
        check(f"after restart FT.INFO idx num_docs {N_WARM} on every worker "
              "(was: unnamed file, 'Unknown index name')", r == [N_WARM] * WORKERS, r)
        r = per_worker(port, lambda c: knn(c, "idx", vecs[42].tobytes()))
        check("warm FT.SEARCH idx answers doc:42 first on every worker",
              all(isinstance(x, list) and len(x) > 1 and x[1] == b"doc:42" for x in r),
              [repr(x)[:60] for x in r])
    finally:
        stop(proc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7403)
    args = ap.parse_args()
    args.binary = os.path.abspath(args.binary)  # the server runs in a temp cwd
    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}")
        return 2
    workdir = tempfile.mkdtemp(prefix="pion-gh403-")
    print(f"gh #403 FT index-name tests — binary={args.binary} port={args.port} dir={workdir}")
    try:
        phase_single(args.binary, args.port, workdir)
        phase_multi(args.binary, args.port + 10, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)   # 256 MB of WAL per worker
    print(f"\n{'=' * 60}")
    print(f"PASS {len(PASS)}  FAIL {len(FAIL)}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
