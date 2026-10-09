# VectorDBBench Performance1536D50K — Pion vs Redis 8.10.2 + RediSearch

One comparison on the same API and the same harness: VectorDBBench's own CLI and
harness, run identically against Pion (through a `pion` client added to
VectorDBBench 1.0.22 for this run: the stock Redis client plus `FT.OPTIMIZE`
after the load; it is to be proposed upstream) and against real Redis 8.10.2 with the RediSearch module (FT.* /
HNSW), 3 rounds each, alternating P R P R P R.

## Method

- **Case:** `Performance1536D50K` (OpenAI, 50,000 vectors, 1536 dims,
  `DISTANCE_METRIC=COSINE`), `k=100`, concurrency levels `[1, 5, 10]` at
  `concurrency_duration=5s` per level — see any `*_cmd.txt` here for the
  exact command line.
- **HNSW params, identical for both engines:** `M=16`, `EF_CONSTRUCTION=128`,
  `EF_RUNTIME=150`.
- **Hardware:** one bare-metal server — AMD EPYC
  8124P (Zen 4c, 16C/32T, 2.45 GHz), Ubuntu 24.04, NVMe. See `setup.txt` for
  the exact `lscpu` output and binary versions
  (`pion-server 0.9.7+4b672bb`, `vector: libpion_vector abi=1 (closed,
  dims=1536) isa=vnni`, `Redis server v=8.10.2`).
- **CPU pinning:** the server (Pion or Redis) pinned to physical cores 0-7
  plus their SMT siblings (`setup.txt`'s `server cpus:` line); the
  `vectordbbench` client process pinned to cores 8-15 plus siblings
  (`client cpus:` line), via `taskset` — visible in each `*_cmd.txt`.
- **Redis:** 8.10.2 + RediSearch, started at its OWN defaults — no index or
  concurrency flags passed to the server. `redis_config_*.txt` is
  `CONFIG GET search-workers` / `CONFIG GET io-threads` from each round:
  `search-workers 16` (RediSearch's built-in default on this box),
  `io-threads 1`.
- **Pion:** `pion-server -w 16 --independent-workers --no-auto-detect
  --no-auto-embed`.

### Load time is not compared

Pion's rounds ran VectorDBBench with `--load-concurrency 1`, the setting Pion's
own vector benchmark uses; Redis's rounds ran VectorDBBench's default
concurrent loader. `insert_duration` and `load_duration` therefore come from
different loaders and are not a load-time comparison. Recall, NDCG, QPS and
latency are measured after the load and do not depend on it.

Separately, `optimize_duration` itself isn't a fair ratio in the other
direction: RediSearch has no `FT.OPTIMIZE` — it builds its HNSW graph
incrementally on every insert — so Redis's `optimize_duration` is just the
no-op client call's overhead (~0.0001s), not an engine build cost. Pion's
~7.4s there is real: ingest buffers raw vectors and builds the graph in one
batch when `FT.OPTIMIZE` is sent.

## Files

| File | What it is |
|---|---|
| `pion_r{1,2,3}.json`, `redis_r{1,2,3}.json` | VectorDBBench's own `TestResult` JSON per round (unmodified, just renamed out of `pion_N/result_*.json` / `redis_N/result_*.json`) |
| `pion_r{1,2,3}_cmd.txt`, `redis_r{1,2,3}_cmd.txt` | The exact `vectordbbench` command line for that round, including the `taskset` pinning |
| `setup.txt` | CPU pinning, `pion-server --version`, Redis version, `lscpu` |
| `redis_config_{1,2,3}.txt` | `CONFIG GET search-workers` / `CONFIG GET io-threads` for each Redis round |
| `summary.md` | `analyze_m3.py`'s per-run and median table for both engines, the Pion/Redis ratio on the medians, and this same method footer — data only, no verdict |

`summary.md` was produced from these six JSON files (median of three per engine).
