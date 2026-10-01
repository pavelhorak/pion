# Benchmarking Guide

How to measure Pion on your own machine, and how not to fool yourself doing it.

## Before you measure

A benchmark on a busy machine produces a *wrong number*, not an error. Run the
pre-flight check first:

```bash
python3 benchmarks/preflight.py
```

It exits non-zero when something would distort the run — a leaked
`pion-server` or `redis-server` from an earlier test, a compile job, high load
average, low disk, Spotlight indexing — and says which of those you can remove
and which you have to wait out. It never kills anything. Every benchmark script
run with `--gate` calls it.

The rules it encodes:

- **Leaked servers cost ~30% on write-heavy rows** and almost nothing on reads.
  A write-row miss with healthy read rows means a resident process, not a slow
  disk. Stop leftover servers by exact name (`pkill -x pion-server`, then
  `pgrep -x pion-server` to confirm).
- **Warm up.** The first few runs on an idle machine are consistently low.
- **Compare against the previous commit on the same machine**, interleaved
  (A B B A), rather than against an absolute number from somewhere else.

---

## KV throughput — memtier_benchmark

`benchmarks/memtier-benchmark/memtier-benchmark.py` wraps `memtier_benchmark`
and measures mixed SET/GET at several pipeline depths.

```bash
# Pion only (throughput + pipeline profiles)
python3 benchmarks/memtier-benchmark/memtier-benchmark.py --pion-only --profiles throughput,pipeline -w 1

# Redis + Valkey + Dragonfly + Pion
python3 benchmarks/memtier-benchmark/memtier-benchmark.py --all-profiles -w 1
```

---

## KV per-command — valkey-benchmark

`benchmarks/valkey-benchmark/valkey-benchmark.py` wraps `redis-benchmark` and
reports 20 commands individually.

```bash
python3 benchmarks/valkey-benchmark/valkey-benchmark.py -c 50 -n 100000 -P 10 -w 1 --pion-only
python3 benchmarks/valkey-benchmark/valkey-benchmark.py -c 50 -n 100000 -P 10 -w 1 --pion-only --gate
```

With `--gate`, each row is checked against its floor in
`benchmarks/gate_baselines.json` and the script exits 1 on a miss. Floors are
per machine class (`--gate-profile mac` or `linux-epyc-8124p`); a Linux server
CPU and an Apple M-series chip are not interchangeable.

| Parameter | Value | Notes |
|---|---|---|
| `-c 50` | 50 concurrent clients | |
| `-n 100000` | 100K requests per command | |
| `-P 10` | Pipeline depth 10 | Pion's advantage is at pipeline depth |
| `-w 1` | 1 worker | One keyspace; the per-command floors are single-worker |

Some rows sit close to their floor, so ordinary run-to-run noise of a few
percent can flip them. `MSET` in particular depends on its position in the run
(it is measured warm, eighth of the command groups); do not reorder the groups
or read a standalone `-t mset` against the floor.

### Linux backend selection

- **`--epoll`** runs at parity with Redis at P=1 (91-96K RPS on Linux
  localhost, where every engine hits the same TCP ceiling). Best for per-command
  `-w 1` runs.
- **io_uring (default)** is behind at P=1 on some CPUs but wins at high
  concurrency (P>=10). Which one wins at P=1 depends on the CPU generation, so
  measure both on a new machine.

**P=1 ceiling:** ~250K RPS on macOS localhost, ~96K on Linux localhost.

---

## Vector search — VectorDBBench

`benchmarks/VectorDBBench/vectordb-benchmark.py`, on the Performance1536D50K
dataset (1536 dims, 50K vectors).

```bash
pixi run install-vdbbench   # pip install 'vectordb-bench[redis]'; use Python 3.11 — 3.14 breaks vectordbbench
python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 10
python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 10 --gate
```

| Parameter | Value |
|---|---|
| Dataset | Performance1536D50K |
| HNSW M | 16 |
| ef_construction | 128 |
| ef_runtime | 150 |
| Workers | 10 (`-w 10 --independent-workers`) |
| Quantization | INT8 |

Vector QPS on a laptop is noisy: identical binaries have measured anywhere in
a ~15% band, and the first run of a series is usually the low one. Recall
varies by a few tenths of a point between runs because the graph build is
randomized, and an occasional single run dips further; check whether a low
number reproduces before reading anything into it.

---

## Protocol parity

```bash
python3 tests/test_parity.py
```

Checks Redis protocol compatibility across ~45 command families.

---

## Notes

- **Startup wait:** a multi-worker Pion takes several seconds to initialise its
  10M-slot hash maps; the harnesses wait for it.
- **Linux io_uring** requires `--security-opt seccomp=unconfined` in Docker.
- **Linux build:** `gcc -c src/ffi/uring_wrap.c -o src/ffi/uring_wrap.o` before
  `pixi run build`.
