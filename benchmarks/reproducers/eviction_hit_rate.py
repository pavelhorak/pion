#!/usr/bin/env python3
"""Hit rate of a cache under Pion's eviction policies, against Redis at the same size.

A look-aside cache on a Zipf-distributed key universe: GET a batch, SET what
missed. Redis's limit is on allocator bytes and Pion's on RSS, so a byte limit
means different things: each server's limit is steered until it holds the same
number of keys, and hit rates are compared at that size. `mem MB` is what each
reports as used_memory: allocator bytes for Redis, RSS for Pion. An ideal LRU over the
same requests, simulated in Python, is the ceiling for the LRU rows.

    python3 benchmarks/reproducers/eviction_hit_rate.py ./pion-server

Needs numpy and redis-server on the PATH. Standard RESP over a socket, no client
library. Prints one table; nothing is written.
"""
import collections
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

BINARY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "./pion-server")
UNIVERSE = 2_000_000
ZIPF = 1.0
VALUE = b"v" * 200
HOLD_KEYS = 200_000      # the cache's size, in keys, on both servers
BATCH = 500
WARM_OPS = 4_000_000
MEASURE_OPS = 4_000_000
PION_PORT, REDIS_PORT = 6520, 6521


class Conn:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=60)
        self.f = self.s.makefile("rb", buffering=1 << 20)

    def _read(self):
        line = self.f.readline()
        t, b = line[:1], line[1:-2]
        if t in (b"+",):
            return b
        if t == b"-":
            raise RuntimeError(b.decode())
        if t == b":":
            return int(b)
        if t == b"$":
            n = int(b)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if t == b"*":
            return [self._read() for _ in range(int(b))]
        raise RuntimeError(repr(line))

    def many(self, cmds):
        out = []
        for c in cmds:
            out.append(b"*%d\r\n" % len(c))
            for a in c:
                a = a if isinstance(a, bytes) else str(a).encode()
                out.append(b"$%d\r\n%s\r\n" % (len(a), a))
        self.s.sendall(b"".join(out))
        return [self._read() for _ in cmds]

    def __call__(self, *a):
        return self.many([a])[0]

    def info(self, field):
        for line in self("INFO").decode().split("\r\n"):
            if line.startswith(field + ":"):
                return line.split(":", 1)[1]
        return None


def zipf_keys(rng, n):
    """Bounded Zipf over the universe, by inverse CDF on a precomputed table."""
    u = rng.random(n)
    return np.searchsorted(CDF, u)


ranks = np.arange(1, UNIVERSE + 1, dtype=np.float64)
CDF = np.cumsum(1.0 / ranks ** ZIPF)
CDF /= CDF[-1]
# A key's name must not reveal its rank to the hash, and a fixed permutation
# keeps the runs comparable.
PERM = np.random.default_rng(7).permutation(UNIVERSE)


def run(conn, requests, steer=None):
    hits = total = 0
    for i in range(0, len(requests), BATCH):
        ids = requests[i:i + BATCH]
        names = [b"k:%d" % PERM[k] for k in ids]
        got = conn.many([(b"GET", n) for n in names])
        miss = [n for n, v in zip(names, got) if v is None]
        hits += len(names) - len(miss)
        total += len(names)
        if miss:
            conn.many([(b"SET", n, VALUE) for n in set(miss)])
        if steer and i % (BATCH * 50) == 0:
            steer()
    return hits, total


def steered_limit(conn, is_redis):
    """A controller of its own: nudge the limit until the server holds HOLD_KEYS.
    It keeps running through the measured phase, or the sizes drift apart and the
    comparison with them. Records every size it saw."""
    seen = []
    def steer():
        keys = conn("DBSIZE")
        seen.append(keys)
        limit = int(conn("CONFIG", "GET", "maxmemory")[1])
        err = (keys - HOLD_KEYS) / HOLD_KEYS
        if abs(err) > 0.01 and keys > HOLD_KEYS // 4:
            bytes_per_key = 300
            conn("CONFIG", "SET", "maxmemory", max(1 << 20, limit - int(err * HOLD_KEYS * bytes_per_key * 0.7)))
    steer.seen = seen
    return steer


def bench(start, is_redis, policy, reqs_warm, reqs_measure):
    proc, d, port = start()
    try:
        c = Conn(port)
        base = int(c.info("used_memory"))
        c("CONFIG", "SET", "maxmemory-policy", policy)
        c("CONFIG", "SET", "maxmemory", base + HOLD_KEYS * 300)
        time.sleep(0.2)
        steer = steered_limit(c, is_redis)
        run(c, reqs_warm, steer)
        steer.seen.clear()
        h, t = run(c, reqs_measure, steer)
        seen = steer.seen
        rss = int(c.info("used_memory")) >> 20
        return (h / t, sum(seen) // len(seen), min(seen), max(seen), rss, base >> 20,
                int(c("CONFIG", "GET", "maxmemory")[1]) >> 20, int(c.info("evicted_keys")))
    finally:
        proc.kill()
        proc.wait()
        shutil.rmtree(d, ignore_errors=True)


def start_pion():
    d = tempfile.mkdtemp(prefix="pion_hit_")
    p = subprocess.Popen([BINARY, "-p", str(PION_PORT), "-w", "1", "--no-wal", "--no-auto-detect", "--no-auto-embed",
                          "--no-crash-log"], cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _wait(PION_PORT)
    return p, d, PION_PORT


def start_redis():
    d = tempfile.mkdtemp(prefix="redis_hit_")
    p = subprocess.Popen(["redis-server", "--port", str(REDIS_PORT), "--save", "", "--appendonly", "no"],
                         cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _wait(REDIS_PORT)
    return p, d, REDIS_PORT


def _wait(port):
    for _ in range(300):
        try:
            Conn(port)("PING")
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("no server on port %d" % port)


def ideal_lru(reqs_warm, reqs_measure):
    cache = collections.OrderedDict()
    def step(ids):
        hits = 0
        for k in ids.tolist():
            if k in cache:
                cache.move_to_end(k)
                hits += 1
            else:
                cache[k] = 1
                if len(cache) > HOLD_KEYS:
                    cache.popitem(last=False)
        return hits
    step(reqs_warm)
    return step(reqs_measure) / len(reqs_measure)


def main():
    rng = np.random.default_rng(42)
    warm = zipf_keys(rng, WARM_OPS)
    measure = zipf_keys(rng, MEASURE_OPS)
    print(f"universe {UNIVERSE:,}, Zipf {ZIPF}, cache {HOLD_KEYS:,} keys, {len(VALUE)}-byte values, "
          f"{WARM_OPS:,} warm-up + {MEASURE_OPS:,} measured requests")
    print(f"ideal LRU (simulated): {ideal_lru(warm, measure):.4f}\n")
    print(f"{'policy':16} {'server':6} {'hit rate':>9} {'mean keys':>10} {'min-max':>17} {'mem MB':>7} "
          f"{'idle MB':>8} {'limit MB':>9} {'evicted':>10}")
    for policy in ("allkeys-lru", "allkeys-lfu", "allkeys-random"):
        for name, start, is_redis in (("Redis", start_redis, True), ("Pion", start_pion, False)):
            hr, mean, lo, hi, mem, idle, lim, ev = bench(start, is_redis, policy, warm, measure)
            print(f"{policy:16} {name:6} {hr:9.4f} {mean:>10,} {lo:>8,}-{hi:<8,} {mem:>7} {idle:>8} {lim:>9} {ev:>10,}",
                  flush=True)


if __name__ == "__main__":
    main()
