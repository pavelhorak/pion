# 2026-10-18: Pion's vector search against RediSearch and valkey-search, 50K and 500K

One machine, one harness, one session. Pion against Redis 8.10.2 + RediSearch
(HNSW FLOAT32, and SVS-VAMANA with 8-bit compression) and Valkey 9.1.2 +
valkey-search 1.3.0-rc1 (HNSW FLOAT16 and FLOAT32), on VectorDBBench's
`Performance1536D50K` and `Performance1536D500K` cases (OpenAI embeddings, 1536
dimensions, cosine, k = 100), plus its streaming case at 50K.

**Every Pion row is a build-once index with no live inserts.** Pion buffers the
vectors and builds its graph once, when the client sends `FT.OPTIMIZE`. An `HSET`
after that is stored but not indexed. RediSearch and valkey-search index every
insert as it arrives. A stock Redis client never sends `FT.OPTIMIZE`, so pointed at
Pion it finds nothing. Pion also searches INT8 codes, where the HNSW columns
search FP32 or FP16.

## Machine and software

- **Machine:** AMD EPYC 8124P, 16 cores / 32 threads, 4 L3 domains, 124 GiB, NVMe
  RAID1, Ubuntu 24.04, bare metal ([`setup.txt`](setup.txt) has `lscpu`, the kernel
  knobs and every version).
- **Pion:** `pion-server 0.9.9+091f6be`, built on the machine with
  `pixi run build-portable`, the release's linux-x86_64 build (x86-64-v2 plus the
  closed vector library, which picked its VNNI build on this CPU:
  `libpion_vector abi=1 (closed, dims=1536) isa=vnni`).
- **Redis:** 8.10.2 built from the tag, RediSearch from `make modules-update` on
  the 8.10 branch (`4c9333b`).
- **Valkey:** 9.1.2 built from the tag, valkey-search 1.3.0-rc1 (`e31273a`, a
  prerelease) built from source.
- **Harness:** VectorDBBench 1.0.22 and redis-py 4.6.0, in two installs
  ([`client/`](client/)):
  - Pion's adds one call: `FT.OPTIMIZE` after the load
    ([`pion_client.diff`](client/pion_client.diff)).
  - The competitors' client takes its index type, vector type and search knob
    from environment variables; with none set it sends exactly what stock 1.0.22
    sends ([`competitor_client.diff`](client/competitor_client.diff)).
- **Datasets:** VectorDBBench's own S3 copies; hashes in
  [`datasets.sha256.txt`](datasets.sha256.txt).

## Method

- **Pinning:** every server on CPUs 0–7 plus their SMT siblings, the client on 8–15
  plus siblings (`taskset`, visible in every `r<n>_cmd.txt`). Each half spans two of
  the four L3 domains. The placement effect is not isolated.
- **Threads:** 16 on every server: Pion `-w 16 --independent-workers`, RediSearch
  `--search-workers 16`, valkey-search `--search.reader-threads 16
  --search.writer-threads 16`.
- **Index:** HNSW `M 16 EF_CONSTRUCTION 128` on every HNSW column. SVS-VAMANA at
  its documented defaults with `COMPRESSION LVQ8`. On AMD hardware and Redis Open
  Source that falls back to 8-bit global scalar quantization; `FT.INFO` reported
  `compression GlobalSQ8` ([`smoke/smoke_redis_svs_lvq8.txt`](smoke/smoke_redis_svs_lvq8.txt)).
  SVS takes the harness's `--ef-runtime` value as `SEARCH_WINDOW_SIZE`.
- **Blocks:** each block is one engine at one size on a fresh server. Run 1 loads
  (default concurrent loader, the same for every engine) and searches at the first
  grid value; runs 2–15 search only. The 5 grid values run 3 times each, in an
  order that never puts two repeats of a value side by side. Tables report the
  median of the 3 repeats.
- **Peak QPS** is the best of concurrency 1, 5 and 10, 5 s each. **Recall** is
  VectorDBBench's serial recall@100 over 1,000 queries.
- **Order:** Pion ran first and last at each size; the gap between its two blocks is
  the drift control. The ratios below use the lower of the two.
- **Time until searchable:** Pion's `load_duration` includes `FT.OPTIMIZE`. The
  stock client's `optimize()` is a no-op, so a competitor could be searched while
  its server still builds in the background. The competitor client therefore waits
  until the server reports its index built, and that wait counts in
  `load_duration`. For both HNSW engines it was 0.0 s: the index was complete when
  the insert returned.
- **Effective ef:** at 500K, Pion raises ef to at least 266 (the server logs
  `effective_ef=266`). Its 500K grid starts at 267, so every value ran as requested.
- **Machine state:** the RAID1 resync was throttled to 1 MB/s for the whole session
  (it moved from 6.2% to 7.8%). Every block's pre-flight was clean: no other
  server, load under 2 ([`blocks/*/preflight.txt`](blocks/)). The 60-second health
  log is [`health.txt`](health.txt).

## Two things that happened

- **The SVS 50K block timed out, with 2 of its 3 repeats.** The competitor client's
  wait required RediSearch's tiered frontend buffer to drain to 0. SVS-VAMANA
  keeps vectors below its flush threshold in that flat buffer, which every search
  also scans: 562 of 50,000 stayed there, with no job pending and no worker active,
  so the wait never ended. Mid-session the wait was changed to also accept
  "indexing done, no jobs, no workers, buffer unchanged for 10 s"
  ([`svs_steady_patch_2026-10-18.py`](client/svs_steady_patch_2026-10-18.py); the
  record is [`chain_progress.txt`](chain_progress.txt)). The block's load was retried
  on a fresh server, and its 40-minute limit cut it after 10 usable runs. The 500K
  SVS block ran with the change from the start. The change adds 10 s to SVS's
  `load_duration`, and the tables below subtract it. For the HNSW columns it
  changed nothing: their buffer was 0.
- **SVS-VAMANA LVQ8 does not reach recall 0.97.** It plateaus at 0.962 at 50K and
  0.954 at 500K (search window 600), as it did on a Mac before the session. Its
  0.97 cells are "not comparable" by construction. That was decided before any
  data, so no 0.97 statement here rests on SVS.

## Results

QPS at the first grid value whose recall reaches the target. The full grids, every
repeat and the serial P99 are in [`summary.md`](summary.md).

**50K**

| Engine | Precision | QPS ≥ 0.95 (ef) | QPS ≥ 0.97 (ef) | Time until searchable |
|---|---|---:|---:|---:|
| Pion, first block | INT8 | 6,910 (150) | 4,120 (300) | 23.9 s |
| Pion, last block | INT8 | 6,846 (150) | 4,099 (300) | 23.8 s |
| RediSearch HNSW | FP32 | 1,880 (150) | 1,725 (200) | 12.9 s |
| RediSearch SVS-VAMANA | 8-bit SQ | 1,888 (150) | not reached | 20.7 s |
| valkey-search HNSW | FP16 | 1,555 (150) | 1,468 (200) | 30.6 s |
| valkey-search HNSW | FP32 | 1,301 (150) | 1,207 (200) | 33.0 s |

**500K**

| Engine | Precision | QPS ≥ 0.95 (ef) | QPS ≥ 0.97 (ef) | Time until searchable |
|---|---|---:|---:|---:|
| Pion, first block | INT8 | 3,779 (267) | 3,368 (400) | 259.8 s |
| Pion, last block | INT8 | 3,766 (267) | 3,250 (400) | 265.6 s |
| RediSearch HNSW | FP32 | 1,545 (200) | 1,363 (300) | 126.7 s |
| RediSearch SVS-VAMANA | 8-bit SQ | 1,163 (400) | not reached | 240.0 s |
| valkey-search HNSW | FP16 | 1,230 (200) | 1,119 (300) | 360.8 s |

SVS's time until searchable is its `load_duration` minus the 10 s confirmation
window above (30.7 s and 250.0 s as recorded).

**Pion / competitor, at matched recall (Pion = the lower of its two blocks)**

| Competitor | 50K, 0.95 | 50K, 0.97 | 500K, 0.95 | 500K, 0.97 |
|---|---:|---:|---:|---:|
| RediSearch HNSW FP32 | 3.64x | 2.38x | 2.44x | 2.38x |
| RediSearch SVS-VAMANA 8-bit | 3.63x | not comparable | 3.24x | not comparable |
| valkey-search HNSW FP16 | 4.40x | 2.79x | 3.06x | 2.90x |
| valkey-search HNSW FP32 | 5.26x | 3.40x | not run | not run |

**Pion drift**, the median over grid values of last block / first block − 1 on peak
QPS: −0.9% at 50K and −0.3% at 500K.

**Time until searchable** is where Pion loses to RediSearch.
- Against RediSearch HNSW, Pion takes about twice as long at both sizes: 23.9 s
  against 12.9 s at 50K, and 259.8 s against 126.7 s at 500K.
- RediSearch SVS is also faster than Pion: 20.7 s and 240.0 s.
- The gap is the build. Pion's insert takes no longer than RediSearch's (12.1 s
  against 12.9 s at 50K, 106.5 s against 126.7 s at 500K), and `FT.OPTIMIZE` adds
  11.8 s and 153.3 s.
- Pion is faster than valkey-search at both sizes.

## Streaming case, 50K

VectorDBBench's `StreamingPerformanceCase`: 200 rows/s for 250 s, with searches
while 50% and 80% of the data is in, at the end of the write, and after
`optimize()`.

| Engine | Recall at 50% | at 80% | end of write | after `optimize()` |
|---|---:|---:|---:|---:|
| Pion | 0.0 | 0.0 | 0.0 | 0.9600 |
| RediSearch HNSW FP32 | 0.5106 | 0.7974 | 0.9628 | 0.9628 |
| valkey-search HNSW FP16 | 0.5084 | 0.7974 | 0.9631 | 0.9631 |

Pion answers nothing until `FT.OPTIMIZE`, so its in-stream QPS counts empty answers
and is not reported. RediSearch served 1,806–1,895 QPS and valkey-search
1,379–1,548 QPS throughout the stream.

## Not measured

- other engines (Qdrant, Elasticsearch, Milvus, pgvector, Dragonfly);
- the streaming case at 500K;
- 1M vectors or any other dimension (Pion's tuned search runs only at 1536);
- vendor-default index parameters;
- Pion's open build (`build-open`);
- valkey-search FP32 at 500K.

## Files

| Path | What |
|---|---|
| [`summary.md`](summary.md) | Every block's full grid, repeats and serial P99; matched-recall tables; ratios; drift; streaming; the chain record |
| `blocks/<block>/r<n>.json` | VectorDBBench's own result JSON per run, unmodified |
| `blocks/<block>/r<n>_cmd.txt` | The exact command line, pinning included, plus a one-line check of the result |
| `blocks/<block>/` other files | Server log, `FT.INFO` and memory after the load, engine facts, pre-flight, Pion's effective ef and index-file identity |
| [`engines/`](engines/) | Each column's server flags, client settings and vendor documentation links |
| [`client/`](client/) | Both client diffs against stock VectorDBBench 1.0.22, and the mid-session wait change |
| [`setup.txt`](setup.txt), [`build_summary.txt`](build_summary.txt), [`smoke/`](smoke/) | Machine, versions, builds, and the 16-vector smoke test of each engine |
| [`chain_progress.txt`](chain_progress.txt), [`health.txt`](health.txt) | The run's timeline, and load, memory, disk and RAID state every 60 s |
| [`harness_logs.tar.gz`](harness_logs.tar.gz) | VectorDBBench's log for every run, and each block's log |

If a number does not reproduce on your machine, open an issue with your hardware
and the raw output.
