# 2026-10-06 — Pion against Redis 8.10.2 on one Linux server

Both engines on the same machine, the same day, with the same client settings.

- Machine: AMD EPYC 8124P, 16 cores / 32 threads, 124 GB, bare metal, Ubuntu
  ([`lscpu.txt`](lscpu.txt), [`uname.txt`](uname.txt), [`free.txt`](free.txt)).
- Pion: `pion-server 0.9.5+d921ceb`, built on the machine with
  `pixi run build-portable` — the release's linux-x86_64 build (x86-64-v2, the
  closed vector library, which picked its VNNI kernels on this CPU)
  ([`pion_version.txt`](pion_version.txt), [`pion_build_task.txt`](pion_build_task.txt)).
- Redis: 8.10.2 built from the tag ([`redis_version.txt`](redis_version.txt)).
- Client: `memtier_benchmark` built from source at `f8d4458d`
  ([`memtier_version.txt`](memtier_version.txt)).

## How it ran

[`run_on_box.sh`](run_on_box.sh) runs every phase and writes this directory;
[`summarize.py`](summarize.py) turns the memtier JSON into
[`summary.md`](summary.md) (medians over the 3 repetitions; for the 16 Redis
instances, the 16 clients' ops/sec are summed per repetition first). After the
run, the script's two paths became overridable (`PION`, `PION_REPO`); their
defaults are what ran.

Common memtier settings ([`settings.txt`](settings.txt)): 256-byte values,
1:10 SET:GET, keys drawn at random from 1M, 30 s per run,
`--distinct-client-seed`. Neither engine persists in the throughput runs
(Redis `--save "" --appendonly no`, Pion `--no-wal`).

- **One keyspace:** 8 client threads × 50 connections against Redis with
  `io-threads` 1, 4 and 8, and against `pion-server -w 1 --profile kv`.
- **Durable:** the same at P=10 with Redis `appendonly yes, appendfsync everysec,
  io-threads 4` and Pion with its WAL on.
- **All cores:** 16 Redis processes (one client thread × 50 connections each)
  against `pion-server -w 16 --independent-workers` (16 client threads × 50
  connections). Both are 16 independent keyspaces.
- **Vector:** VectorDBBench Performance1536D50K (50K OpenAI embeddings, 1536
  dimensions, recall@100 against the dataset's ground truth), concurrency 1, 5
  and 10 for 5 s each. Pion through
  `benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 16`
  (VectorDBBench's Redis client over `FT.*`, M=16, EF_CONSTRUCTION=128); Redis
  through `benchmarks/VectorDBBench/vset-benchmark.py`, which drives `VADD`/`VSIM`
  at Redis's defaults (VectorDBBench has no vector-set client).

Raw output: `raw/*.txt` (memtier's own report per run), `raw_json.tar.gz`
(memtier's JSON, which `summarize.py` reads), `vec/` (each vector run's log).

## KV: one keyspace, no persistence (ops/sec, median of 3)

| Pipeline | Redis, 1 thread | Redis, `io-threads 4` | Redis, `io-threads 8` | Pion `-w 1` |
|---|---:|---:|---:|---:|
| P=1 | 107,896 | 257,182 | 388,405 | 112,019 |
| P=10 | 643,517 | 1,405,547 | 1,655,550 | 877,766 |
| P=50 | 987,970 | 1,239,044 | 1,376,680 | 1,712,725 |

## KV: durable, P=10 (ops/sec, median of 3)

| Redis AOF `everysec`, `io-threads 4` | Pion WAL, `-w 1` |
|---:|---:|
| 1,213,746 | 847,249 |

## KV: all 16 cores, 16 independent keyspaces (ops/sec, median of 3)

| Pipeline | 16 Redis processes | Pion `-w 16 --independent-workers` |
|---|---:|---:|
| P=10 | 7,178,453 | 6,097,719 |
| P=50 | 11,128,400 | 9,848,658 |

## Vector (median of 3)

| | Redis 8.10.2 vector sets | Pion |
|---|---:|---:|
| QPS, 10 clients (the peak for both) | 8,039 | 7,339 |
| Recall@100 | 0.920 | 0.960 |
| P99 latency, one client | 1.75 ms | 1.4 ms |
| Load until searchable | 51.8 s | 26.4 s |

Pion's load is its insert (≈10.7 s) plus the index build (≈15.5 s, `FT.OPTIMIZE`);
Redis builds its graph while it inserts. The two clients are different programs
that send the same queries at the same concurrency, so read the QPS row with that
in mind.
