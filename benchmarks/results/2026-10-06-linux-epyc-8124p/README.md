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

- **Closed vector library against the open build:** [`open_vs_closed_on_box.sh`](open_vs_closed_on_box.sh),
  run after the above on the same machine. It builds `pixi run build-portable`
  (the library) and `pixi run build-open` from the same commit, both for
  x86-64-v2, and runs the vector harness at `--ef-runtime 150 --workers 16` with
  the arms in the order closed, open, closed-v2, open, closed-v2, closed,
  closed-v2, closed, open — where closed-v2 is the library binary with
  `PION_VECTOR_VNNI=0`.

Raw output: `raw/*.txt` (memtier's own report per run), `raw_json.tar.gz`
(memtier's JSON, which `summarize.py` reads), `vec/` (each vector run's log),
`ovc/` (each open-vs-closed run's log, `results.txt` the script's one-line
extract per run, `version_*.txt` each binary's `--version`), `iso/` (Pion's
search `ef` swept by [`iso_recall_on_box.sh`](iso_recall_on_box.sh), same harness
and settings otherwise).

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

## Vector: closed library against the open build (INT8, median of 3)

| Build | QPS (10 clients) | Recall@100 |
|---|---:|---:|
| `libpion_vector`, VNNI kernels (the default on this CPU) | 7,502 | 0.960 |
| `libpion_vector`, x86-64-v2 kernels (`PION_VECTOR_VNNI=0`) | 5,481 | 0.960 |
| open build | 4,467 | 0.960 |

## Vector: Pion's search effort against recall (INT8, `-w 16`, one run each)

| `--ef-runtime` | QPS (10 clients) | Recall@100 |
|---|---:|---:|
| 32 | 8,462 | 0.939 |
| 48 | 8,482 | 0.939 |
| 64 | 8,607 | 0.939 |
| 100 | 8,718 | 0.939 |
| 150 | 7,496 | 0.960 |

`FT.SEARCH` raises an ef below k to k (`src/commands/vector.mojo`), so with k=100
the first four rows are one configuration run four times: median 8,544 QPS at
recall 0.939. Redis vector sets at their defaults measured 8,039 QPS at 0.920 on
this machine (table above).

## KV: 32 keyspaces each side (P=50, ops/sec, median of 3). Not a comparison: read why below

Run after everything above, on the same machine, by [`peak_on_box.sh`](peak_on_box.sh)
and [`peak64_on_box.sh`](peak64_on_box.sh) (raw output in `peak/` and `peak64/`).
The machine has 16 cores and 32 hardware threads; both sides get 32 keyspaces,
32 client threads and 800 connections: `pion-server -w 32 --independent-workers`
under one `memtier -t 32 -c 25`, against 32 Redis processes under one
`memtier -t 1 -c 25` each.

| Values | 32 Redis processes | Pion `-w 32` |
|---|---:|---:|
| 64 bytes | 14,780,031 | 15,864,842 |
| 256 bytes | 12,372,245 | 11,640,929 |

April's configuration for the old "14.0M ops/sec" headline — Pion `-w 32` under
`memtier -t 16 -c 50`, P=50, Pion only — measured 13,750,187 with 64-byte values
and 9,529,047 with 256-byte values (`peak/`). memtier reported its own threads at
100% CPU in these runs, so the client, not the server, may set the ceiling.

### Why these runs do not compare the servers (found 2026-10-12)

**Unequal connection spread.** Each Redis process served exactly its own client's 25
connections. Pion's 800 connections went to whichever worker won `accept()`, and
the GET hit rates show they landed unevenly:
- **The model:** with random keys over 1M, a keyspace that receives S writes per
  second over a 30 s run hits on average 1 − (1 − e^−x)/x of its GETs, where
  x = 30·S/1M.
- **Redis:** the model reproduces all four Redis hit rates exactly (43.2% and 38.2%
  here, 42.3% and 55.2% for the 16-process runs).
- **Pion:** its 64-byte run hit 67% of its GETs where an even spread over 32 workers
  gives 45%, as if about 15 workers held the traffic. The 256-byte run behaves like
  about 19 workers, and April's configuration like about 10.
- **Where Pion spread evenly:** in the 16-worker runs Pion's spread was close to even
  (14 to 15 of 16), and there Redis leads by 13–18%.

**The shared CPUs went to the client.** Client and server share the machine's 32
hardware threads, so what the server leaves idle goes to the client:
- In the 64-byte run, the client used 21.6 of 32 threads against Pion and 16.4
  against the 32 Redis processes (memtier's `Cores used`, in `raw/`).
- The hits and misses per second are in each run's `Totals` line and in
  `raw_json.tar.gz`.

So the 64-byte row measures the uneven connection spread, not either server.

**Also not one database.** A third of Pion's GETs (and more of Redis's) came back
empty, because each keyspace only holds what its own connections wrote.
