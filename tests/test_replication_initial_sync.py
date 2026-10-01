#!/usr/bin/env python3
"""A replica that connects AFTER data was written must still receive it.

WHY
Redis replicas start with a full resync (an RDB transfer) and then follow the
command stream. Pion's replica only follows the WAL stream from the moment it
connects: anything the primary held before that — the whole dataset, when a
replica is added to a running primary, or after any reconnect — never arrives,
and nothing reports it. test_replication.py started its replica first and
wrote afterwards, so it could not see this.

    python3 tests/test_replication_initial_sync.py [--port 2401]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready  # noqa: E402

BIN = os.environ.get("PION_BIN", "./pion-server")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2401)
    args = ap.parse_args()
    pa, pb = args.port, args.port + 20
    work = tempfile.mkdtemp(prefix="pion-repl-sync-")
    procs = []

    def start(port, extra, sub):
        d = os.path.join(work, sub); os.makedirs(d)
        p = subprocess.Popen([os.path.abspath(BIN), "-p", str(port), "-w", "1", "--no-crash-log",
                              "--no-auto-detect", "--no-auto-embed", "--cluster",
                              "--cluster-host", "127.0.0.1"] + extra, cwd=d,
                             stdout=open(os.path.join(d, "log"), "w"), stderr=subprocess.STDOUT)
        procs.append(p)
        wait_ready(port, 30, proc=p)

    try:
        start(pa, [], "primary")
        a = Conn(pa)
        before = {f"{{s}}pre:{i}": f"v{i}" for i in range(20)}
        a.pipeline([("SET", k, v) for k, v in before.items()])
        start(pb, ["--cluster-replica", "--cluster-primary-host", "127.0.0.1",
                   "--cluster-primary-port", str(pa)], "replica")
        b = Conn(pb)
        b.cmd("READONLY")
        # Prove the stream itself is live, so a miss below is about SYNC.
        deadline = time.time() + 20
        live = False
        while time.time() < deadline:
            a.cmd("SET", "{s}live", "1")
            if b.cmd("GET", "{s}live") == b"1":
                live = True
                break
            time.sleep(0.2)
        if not live:
            print("FAIL: the replication stream never delivered a live write (not a sync question)")
            return 1
        missing = [k for k, v in before.items() if b.cmd("GET", k) != v.encode()]
        print(f"pre-existing keys on the replica: {len(before) - len(missing)}/{len(before)}")
        if missing:
            print(f"FAIL: keys written before the replica connected never arrived: {missing[:4]} …")
            return 1
        print("PASS")
        return 0
    finally:
        for p in procs:
            p.kill(); p.wait()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
