#!/usr/bin/env python3
"""gh #14 — phase-2 epoch RCU for FT.DROPINDEX reclamation.

Phase 1 slept 10 ms before freeing the published index buffers, on the
reasoning that a search takes sub-ms so 10 ms outlasts any in-flight reader.
That is a statement about the machine, not a guarantee — a page fault, a
descheduled worker, or a much larger index makes it false, and nothing detects
when it does.

Phase 2 waits for the actual condition: bump an epoch, then wait until every
OTHER worker is either idle between dispatch batches or running at an epoch at
least as new as the bump. At that point no worker can still hold a pointer it
borrowed before the drop.

What this test can and cannot show
----------------------------------
A use-after-free is not directly observable from the wire — the old code was
*usually* right, which is exactly why it survived. So this asserts the two
things that ARE observable and that a broken implementation would violate:

  1. The server survives DROPINDEX racing concurrent FT.SEARCH traffic, with
     every connection still framed afterwards. A premature free shows up as a
     crash, a hang, or garbage in a reply.
  2. DROPINDEX does not become slower. The failure mode of a wait-based
     reclaimer is waiting on a condition that never clears — most likely the
     reclaimer waiting on ITSELF, since it runs inside a dispatch batch and is
     therefore marked busy at the pre-bump epoch. That bug is invisible to a
     correctness check (it silently degrades to the 10 ms fallback) and shows
     up only as latency, so the timing assertion is the real regression guard.

Run:  ./pion-server -p 1974 -w 1 &   (or -w 4 --independent-workers --no-auto-detect --no-auto-embed)
      python3 tests/test_gh14_epoch_rcu.py
"""
import argparse
import socket
import struct
import sys
import threading
import time


class Client:
    def __init__(self, port, host="127.0.0.1", timeout=30):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.f = self.sock.makefile("rb")

    def cmd(self, *args):
        out = [b"*%d\r\n" % len(args)]
        for a in args:
            b = a if isinstance(a, bytes) else str(a).encode()
            out.append(b"$%d\r\n%s\r\n" % (len(b), b))
        self.sock.sendall(b"".join(out))
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        t, body = line[:1], line[1:-2]
        if t == b"$":
            n = int(body)
            return None if n == -1 else self.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(body)
            return None if n == -1 else [self._read() for _ in range(n)]
        # Strip the type byte for +simple, -error and :integer. Returning
        # "+PONG" instead of "PONG" here manufactured a fake desync on every
        # reader — the harness-false-signal trap, hit again.
        if t == b":":
            return int(body)
        return body.decode()

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


DIM = 8
NVEC = 400


def build_index(c, name="rcu_idx"):
    c.cmd("FT.CREATE", name, "SCHEMA", "v", "VECTOR", "HNSW", "6",
          "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "COSINE")
    for i in range(NVEC):
        vec = struct.pack("<%df" % DIM, *[(i % 17) + j * 0.01 for j in range(DIM)])
        c.cmd("HSET", "doc:%d" % i, "v", vec)
    c.cmd("FT.OPTIMIZE", name)


def query_blob():
    return struct.pack("<%df" % DIM, *[1.0] * DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--readers", type=int, default=6)
    ap.add_argument("--cycles", type=int, default=12)
    args = ap.parse_args()

    main_c = Client(args.port)
    failures = []
    stop = threading.Event()
    reader_errors = []

    def reader():
        try:
            rc = Client(args.port)
        except OSError as e:
            reader_errors.append("connect: %s" % e)
            return
        blob = query_blob()
        try:
            while not stop.is_set():
                # The reply may legitimately be an empty array or an error
                # (index dropped) — what must never happen is a malformed
                # frame, a hang, or a closed connection.
                try:
                    rc.cmd("FT.SEARCH", "rcu_idx", "*=>[KNN 10 @v $B]",
                           "PARAMS", "2", "B", blob, "DIALECT", "2")
                except ConnectionError as e:
                    reader_errors.append("during search: %s" % e)
                    return
                if rc.cmd("PING") != "PONG":
                    reader_errors.append("connection desynced after FT.SEARCH")
                    return
        except Exception as e:                                # noqa: BLE001
            reader_errors.append("%s: %s" % (type(e).__name__, e))
        finally:
            rc.close()

    print("building index (%d vectors, dim %d)" % (NVEC, DIM))
    build_index(main_c)

    threads = [threading.Thread(target=reader, daemon=True)
               for _ in range(args.readers)]
    for t in threads:
        t.start()
    time.sleep(0.5)

    print("racing %d DROPINDEX cycles against %d concurrent readers"
          % (args.cycles, args.readers))
    drop_ms = []
    for n in range(args.cycles):
        t0 = time.perf_counter()
        r = main_c.cmd("FT.DROPINDEX", "rcu_idx")
        dt = (time.perf_counter() - t0) * 1000.0
        drop_ms.append(dt)
        if r != "OK":
            failures.append("cycle %d: DROPINDEX replied %r" % (n, r))
        build_index(main_c)

    stop.set()
    for t in threads:
        t.join(timeout=10)

    if main_c.cmd("PING") != "PONG":
        failures.append("control connection desynced")
    main_c.cmd("FT.DROPINDEX", "rcu_idx")
    main_c.close()

    failures.extend(reader_errors)

    drop_ms.sort()
    median = drop_ms[len(drop_ms) // 2]
    worst = drop_ms[-1]
    print("  DROPINDEX median %.2f ms, worst %.2f ms" % (median, worst))

    # Phase 1 always slept 10 ms. If the epoch wait works, the median is well
    # under that; if the reclaimer is waiting on a condition that never clears
    # (the self-wait bug), every call falls back to the 10 ms sleep and the
    # median lands at or above it.
    if median >= 9.0:
        failures.append(
            "DROPINDEX median %.2f ms — at or above the 10 ms phase-1 fallback, "
            "so the epoch wait is timing out rather than draining" % median)

    if failures:
        print("\ngh #14 epoch RCU: FAIL")
        for f in failures[:15]:
            print("  " + f)
        return 1
    print("\ngh #14 epoch RCU: PASS "
          "(%d cycles x %d readers, no desync, median %.2f ms)"
          % (args.cycles, args.readers, median))
    return 0


if __name__ == "__main__":
    sys.exit(main())
