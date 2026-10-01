#!/usr/bin/env python3
"""gh #78 — MOE.EXPERT.FETCH cold-latency driver (sync-pread vs io_uring A/B).

Drives cold FETCHes over unique (layer, expert) pairs at a given concurrency
against an already-running pion-server with a model LOADed. Cold discipline
(page-cache drop, fresh server) is the orchestrator's job, not this driver's.

    python3 tests/bench_moe_fetch_iouring.py --model olmoe \
        --layers 16 --experts 64 --pairs 256 --concurrency 32

Prints: per-FETCH p50/p95/p99/max (ms), total wall, aggregate MB/s.
"""

import argparse
import os
import random
import socket
import threading
import time


def encode(*args) -> bytes:
    out = [f"*{len(args)}\r\n".encode()]
    for a in args:
        b = a if isinstance(a, bytes) else str(a).encode()
        out.append(f"${len(b)}\r\n".encode() + b + b"\r\n")
    return b"".join(out)


class Conn:
    """Buffered RESP reader — never slice a growing recv buffer (O(N^2))."""

    def __init__(self, host: str, port: int):
        self.s = socket.create_connection((host, port), timeout=30)
        self.f = self.s.makefile("rb")

    def fetch(self, model: str, layer: int, expert: int) -> int:
        """Returns payload byte count. Raises on -ERR/-UNAVAILABLE."""
        self.s.sendall(encode("MOE.EXPERT.FETCH", model, layer, expert))
        line = self.f.readline()
        if line.startswith(b"-"):
            raise RuntimeError(line.decode().strip())
        if line.startswith(b"$"):
            n = int(line[1:])
            if n < 0:
                raise RuntimeError("nil FETCH reply")
            payload = self.f.read(n + 2)
            return n
        if line.startswith(b"*"):
            # multi-blob reply: read each bulk part
            parts = int(line[1:])
            total = 0
            for _ in range(parts):
                hdr = self.f.readline()
                if not hdr.startswith(b"$"):
                    raise RuntimeError(f"unexpected part header {hdr!r}")
                n = int(hdr[1:])
                if n >= 0:
                    self.f.read(n + 2)
                    total += n
            return total
        raise RuntimeError(f"unexpected reply {line!r}")

    def close(self):
        try:
            self.f.close()
            self.s.close()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PION_PORT", "1974")))
    ap.add_argument("--model", default="olmoe")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=256, help="unique (layer,expert) cold fetches")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42, help="same seed both variants = same pair set")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_pairs = [(l, e) for l in range(args.layers) for e in range(args.experts)]
    rng.shuffle(all_pairs)
    pairs = all_pairs[: args.pairs]

    lock = threading.Lock()
    latencies: list[float] = []
    bytes_total = [0]
    errors: list[str] = []
    idx = [0]

    def worker():
        conn = Conn(args.host, args.port)
        while True:
            with lock:
                if idx[0] >= len(pairs) or errors:
                    break
                l, e = pairs[idx[0]]
                idx[0] += 1
            t0 = time.perf_counter()
            try:
                n = conn.fetch(args.model, l, e)
            except Exception as exc:
                with lock:
                    errors.append(f"({l},{e}): {exc}")
                break
            dt = time.perf_counter() - t0
            with lock:
                latencies.append(dt)
                bytes_total[0] += n
        conn.close()

    t_start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    if errors:
        print(f"FAIL: {errors[0]}  ({len(errors)} errors)")
        return 1

    lat = sorted(latencies)
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] * 1000
    mb = bytes_total[0] / 1e6
    print(
        f"pairs={len(lat)} c={args.concurrency} wall={wall:.2f}s "
        f"p50={q(0.50):.2f}ms p95={q(0.95):.2f}ms p99={q(0.99):.2f}ms "
        f"max={lat[-1]*1000:.2f}ms agg={mb/wall:.0f}MB/s total={mb:.0f}MB"
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
