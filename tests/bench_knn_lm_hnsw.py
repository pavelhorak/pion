#!/usr/bin/env python3
"""Bench AI.KNN_LM.* HNSW acceleration vs brute-force.

For three datastore sizes (under threshold, just over, well over) we:
  1. AI.KNN_LM.CREATE a fresh datastore.
  2. Stream synthetic random unit-norm FP32 embeddings via AI.KNN_LM.STOREBATCH
     (each entry tagged with its index as the next_token_id).
  3. Run NUM_QUERIES random queries via AI.KNN_LM.QUERY, time the wire.
  4. Compute brute-force ground truth in numpy on the same data.
  5. Report:
     - recall@k (fraction of true top-k IDs returned in any order)
     - median query latency
     - the cross-over: at what N does HNSW beat brute-force on wall time

The HNSW path activates automatically inside the server once
count ≥ KNN_HNSW_BUILD_THRESHOLD (default 5000). Below that, both paths
are brute-force; we expect identical numbers.

Run:
  ./pion-server --kvcache -w 1
  .pixi/envs/default/bin/python tests/bench_knn_lm_hnsw.py [--start]
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ─── RESP-2 client (self-contained, single bulk reply) ─────────────────────


class Wire:
    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.sock.settimeout(600)
        self.sock.connect((host, port))
        self.buf = b""

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    @staticmethod
    def _encode(parts):
        out = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            if isinstance(p, (bytes, bytearray)):
                out.append(f"${len(p)}\r\n".encode())
                out.append(bytes(p))
                out.append(b"\r\n")
            else:
                s = str(p).encode()
                out.append(f"${len(s)}\r\n".encode())
                out.append(s)
                out.append(b"\r\n")
        return b"".join(out)

    def _recv(self, n_min: int):
        while len(self.buf) < n_min:
            chunk = self.sock.recv(64 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("pion closed")
            self.buf += chunk

    def _read_line(self) -> bytes:
        while True:
            nl = self.buf.find(b"\r\n")
            if nl >= 0:
                line, self.buf = self.buf[:nl], self.buf[nl + 2:]
                return line
            self._recv(len(self.buf) + 1)

    def call(self, *parts):
        self.sock.sendall(self._encode(parts))
        line = self._read_line()
        if not line:
            raise ConnectionError("empty reply")
        t = line[:1]
        if t == b"+":
            return line[1:]
        if t == b"-":
            raise RuntimeError(line[1:].decode("utf-8", errors="replace"))
        if t == b":":
            return int(line[1:])
        if t == b"$":
            n = int(line[1:])
            if n < 0:
                return None
            self._recv(n + 2)
            data, self.buf = self.buf[:n], self.buf[n + 2:]
            return data
        raise RuntimeError(f"unexpected reply: {line!r}")


# ─── Server lifecycle ─────────────────────────────────────────────────────


def start_pion(port: int, log_path: str) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")  # gh #429
    cmd = [binary, "--kvcache", "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, stdout=log, stderr=log, preexec_fn=os.setsid)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                time.sleep(1)
                return proc
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("pion-server didn't come up")


def stop_pion(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# ─── Bench logic ──────────────────────────────────────────────────────────


def make_dataset(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, dim)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


def make_queries(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, dim)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


def brute_force_topk(vecs: np.ndarray, query: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    """L2² distances; matches the server's _l2_distance_fp32."""
    diff = vecs - query
    d = np.einsum("nd,nd->n", diff, diff)
    order = np.argsort(d)[:k]
    return [int(i) for i in order], [float(d[i]) for i in order]


def parse_query_reply(blob: bytes, k: int) -> tuple[list[int], list[float]]:
    """Server reply: k * (Int32 token_id, Float32 distance) little-endian."""
    if blob is None:
        return [], []
    expected = k * 8
    if len(blob) < expected:
        raise RuntimeError(f"short query reply: {len(blob)} < {expected}")
    tids = []
    dists = []
    for i in range(k):
        tid, dist = struct.unpack_from("<if", blob, i * 8)
        tids.append(tid)
        dists.append(dist)
    return tids, dists


def run_bench_for_n(w: Wire, n: int, dim: int, k: int, num_queries: int, ds_id: str) -> dict:
    print(f"\n  ===== n={n} (HNSW threshold = 5000) =====")
    # Drop any prior datastore with this name (idempotent).
    try:
        w.call("AI.KNN_LM.DROP", ds_id)
    except RuntimeError:
        pass

    # Create + STOREBATCH the full dataset.
    vecs = make_dataset(n, dim, seed=0xDA7A0001)
    w.call("AI.KNN_LM.CREATE", ds_id, str(dim), str(max(n + 1, 6000)))
    print(f"    [load] storebatch n={n} dim={dim}...")
    t0 = time.perf_counter()
    # Server's STOREBATCH takes (n, token_ids_blob, embeddings_blob).
    token_ids = np.arange(n, dtype=np.int32).tobytes()
    emb_blob = vecs.tobytes()
    w.call("AI.KNN_LM.STOREBATCH", ds_id, str(n), token_ids, emb_blob)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"    [load] {load_ms:.1f} ms total ({load_ms / n * 1000:.2f} µs/entry)")

    # Sanity: AI.KNN_LM.INFO should show count == n.
    info = w.call("AI.KNN_LM.INFO", ds_id)
    print(f"    [info] {info!r}" if info is not None else "    [info] None (unexpected)")

    # Run queries.
    queries = make_queries(num_queries, dim, seed=0xCA11A123)
    server_results: list[tuple[list[int], list[float]]] = []
    latencies: list[float] = []
    for qi in range(num_queries):
        q_blob = queries[qi].tobytes()
        t0 = time.perf_counter()
        reply = w.call("AI.KNN_LM.QUERY", ds_id, str(k), q_blob)
        latencies.append((time.perf_counter() - t0) * 1000)
        ids, dists = parse_query_reply(reply, k)
        server_results.append((ids, dists))

    # Ground truth.
    gt_results: list[list[int]] = []
    for qi in range(num_queries):
        gt_ids, _ = brute_force_topk(vecs, queries[qi], k)
        gt_results.append(gt_ids)

    # Recall@k = fraction of GT ids returned in any order.
    recalls: list[float] = []
    valid_returns: list[int] = []  # count of non-sentinel ids returned per query
    for (got_ids, _), gt in zip(server_results, gt_results):
        valid = [i for i in got_ids if i >= 0]
        valid_returns.append(len(valid))
        if not gt:
            recalls.append(1.0)
            continue
        hit = sum(1 for i in valid if i in set(gt))
        recalls.append(hit / k)

    median_lat = statistics.median(latencies)
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))] if len(latencies) >= 20 else max(latencies)
    median_recall = statistics.median(recalls)

    print(f"    queries: median={median_lat:.3f} ms  p95={p95_lat:.3f} ms")
    print(f"    recall@{k}: median={median_recall:.4f}  min={min(recalls):.4f}")
    print(f"    valid returns/query: median={statistics.median(valid_returns)}/{k}")

    return {
        "n": n,
        "load_ms": load_ms,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "median_recall": median_recall,
        "min_recall": min(recalls),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1974)
    p.add_argument("--start", action="store_true")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--sizes", default="2000,8000,30000",
                   help="Comma-separated dataset sizes. Default brackets the HNSW threshold (5000).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    proc = None
    log_path = "/tmp/pion_knn_lm_bench.log"
    sizes = [int(s) for s in args.sizes.split(",")]
    try:
        if args.start:
            for f in ("pion.wal.0", "pion.snapshot.0", "pion.hnsw.0",
                      "pion.vstore.0", "pion.vstore.wal.0"):
                p = os.path.join(PROJECT_ROOT, f)
                if os.path.exists(p):
                    os.remove(p)
            print(f"Starting pion-server (log: {log_path})")
            proc = start_pion(args.port, log_path)
        else:
            with socket.create_connection(("127.0.0.1", args.port), timeout=2):
                pass

        w = Wire("127.0.0.1", args.port)
        results = []
        for n in sizes:
            results.append(run_bench_for_n(w, n, args.dim, args.k, args.queries, f"bench_n{n}"))
        w.close()

        print("\n" + "=" * 78)
        print(f"Summary  dim={args.dim}  k={args.k}  queries={args.queries}")
        print("=" * 78)
        print(f"{'n':>10} {'load (ms)':>12} {'median q (ms)':>16} {'p95 q (ms)':>13} {'recall':>8}")
        for r in results:
            print(f"{r['n']:>10}  {r['load_ms']:>10.1f}  "
                  f"{r['median_latency_ms']:>14.3f}  "
                  f"{r['p95_latency_ms']:>11.3f}  "
                  f"{r['median_recall']:>8.4f}")

        # Honest verdict checks:
        below_threshold = [r for r in results if r["n"] < 5000]
        above_threshold = [r for r in results if r["n"] >= 5000]
        if below_threshold:
            print(f"\nBelow threshold (brute-force): "
                  f"recall median = "
                  f"{statistics.median(r['median_recall'] for r in below_threshold):.4f} "
                  f"(should be 1.0)")
        if above_threshold:
            recall_above = statistics.median(r["median_recall"] for r in above_threshold)
            print(f"Above threshold (HNSW): "
                  f"recall median = {recall_above:.4f} "
                  f"(target ≥ 0.95)")
            if recall_above < 0.90:
                print("  ⚠ Recall low — consider raising ef during search.")

        # Regression gate (gh issue #2): recall@k must stay ≥ 0.85 at n=30K
        # once the ef-auto-scale fix is in place. Pre-fix this bench measured
        # 0.50 at 30K. If you see this fire after a knn_lm change, the
        # ef-scaling formula in src/network/knn_lm.mojo regressed.
        n30k_results = [r for r in results if r["n"] == 30000]
        if n30k_results:
            recall_30k = n30k_results[0]["median_recall"]
            if recall_30k < 0.85:
                print(f"\n  ❌ REGRESSION: recall@{args.k} at n=30000 = "
                      f"{recall_30k:.4f}, gate requires ≥ 0.85")
                return 1
            print(f"  ✓ Gate (n=30000, recall@{args.k} ≥ 0.85): "
                  f"{recall_30k:.4f}")

            # Regression gate (gh issue #42): median query latency at n=30K
            # ≤ 2.5 ms. Captures the full SQ8 + heap-PQ + batch-4 stack:
            # FP32 baseline 5.17 ms → SQ8 2.9 ms → +heap 2.7 ms →
            # +per-vec batch-4 distance kernel ~1.94 ms. Threshold sits
            # at observed-plus-headroom for thermal noise; the 2 ms target
            # in the issue is met when the chip is cool. If you see this
            # fire, the SQ8 / batch-4 path in src/network/knn_lm.mojo
            # regressed — check that KNNHNSW._search_layer is still
            # collecting unvisited neighbors and dispatching to
            # `l2_distance_fp32_int8_pervec_batch4` (kernels.mojo) for
            # groups of 4, with embeddings_int8_ref/sq_min_ref/sq_range_ref
            # populated by store_one/store_batch.
            lat_30k = n30k_results[0]["median_latency_ms"]
            if lat_30k > 2.5:
                print(f"\n  ❌ REGRESSION: median latency at n=30000 = "
                      f"{lat_30k:.3f} ms, gate requires ≤ 2.5 ms")
                return 1
            print(f"  ✓ Gate (n=30000, median latency ≤ 2.5 ms): "
                  f"{lat_30k:.3f} ms")

            # Regression gate (gh issue #42): bulk-build load time at
            # n=30K ≤ 60 s. Captures the lazy-reciprocal-pruning + scratch
            # pre-alloc win (251 s → ~36 s with the rebuild moved out of
            # the hot insert path and into compact_overflows() at end of
            # bulk insert). Threshold sits at observed-plus-headroom (60 s
            # vs ~36 s observed) for thermal/system noise; the issue's
            # 150 s target is comfortably met. If you see this fire, the
            # eager M² rebuild is back in `insert()`'s for-k loop — check
            # that the `else:` branch only fires at nb_count >=
            # _max_neighbors_storage(L) (= 2× M0/M), not at
            # _max_neighbors(L), and that compact_overflows() is still
            # called from `_maybe_build_hnsw` after bulk insert.
            load_30k_s = n30k_results[0]["load_ms"] / 1000.0
            if load_30k_s > 60.0:
                print(f"\n  ❌ REGRESSION: 30K bulk-build load time = "
                      f"{load_30k_s:.1f} s, gate requires ≤ 60 s")
                return 1
            print(f"  ✓ Gate (n=30000, load time ≤ 60 s): "
                  f"{load_30k_s:.1f} s")

        if len(results) >= 2:
            # Latency comparison between adjacent sizes — cross the threshold.
            for i in range(len(results) - 1):
                a, b = results[i], results[i + 1]
                ratio = b["median_latency_ms"] / max(0.001, a["median_latency_ms"])
                size_ratio = b["n"] / a["n"]
                print(f"Latency scaling n={a['n']} → n={b['n']}: "
                      f"latency ratio {ratio:.2f}× for size ratio {size_ratio:.2f}× "
                      f"({'sub-linear' if ratio < size_ratio * 0.7 else 'roughly linear'})")
    except Exception:
        import traceback
        traceback.print_exc()
        return 2
    finally:
        if args.start:
            stop_pion(proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
