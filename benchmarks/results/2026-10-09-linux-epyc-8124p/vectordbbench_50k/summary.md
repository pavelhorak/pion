# VectorDBBench Performance1536D50K: Pion vs Redis 8.10.2 + RediSearch, per-run results

Per-run values and the median of three for each engine; the ratio is Pion over Redis on the medians.

### pion

metric | pion_1 | pion_2 | pion_3 | median
--- | --- | --- | --- | ---
recall | 0.9601 | 0.9600 | 0.9598 | 0.9600
ndcg | 0.9678 | 0.9675 | 0.9676 | 0.9676
qps (peak) | 6673.0 | 7491.1 | 6445.9 | 6673.0
conc_qps C=1 | 898.9 | 653.0 | 865.6 | 865.6
conc_qps C=5 | 3406.8 | 3938.4 | 3370.3 | 3406.8
conc_qps C=10 | 6673.0 | 7491.1 | 6445.9 | 6673.0
serial p99 (s) | 0.001300 | 0.001300 | 0.001300 | 0.001300
insert_duration (s) | 10.83 | 11.76 | 10.60 | 10.83
optimize_duration (s) | 7.45 | 7.41 | 7.46 | 7.45
load_duration (s) | 18.28 | 19.17 | 18.06 | 18.28

### redis

metric | redis_1 | redis_2 | redis_3 | median
--- | --- | --- | --- | ---
recall | 0.9629 | 0.9627 | 0.9629 | 0.9629
ndcg | 0.9696 | 0.9694 | 0.9695 | 0.9695
qps (peak) | 1887.0 | 1878.7 | 1903.7 | 1887.0
conc_qps C=1 | 222.9 | 244.3 | 250.6 | 244.3
conc_qps C=5 | 1053.9 | 1043.4 | 1045.0 | 1045.0
conc_qps C=10 | 1887.0 | 1878.7 | 1903.7 | 1887.0
serial p99 (s) | 0.005100 | 0.004900 | 0.005000 | 0.005000
insert_duration (s) | 12.47 | 13.32 | 12.33 | 12.47
optimize_duration (s) | 0.00 | 0.00 | 0.00 | 0.00
load_duration (s) | 12.47 | 13.32 | 12.33 | 12.47

### Pion / Redis ratio (on the medians above)

metric | ratio | direction
--- | --- | ---
recall | 0.997x | higher is better
ndcg | 0.998x | higher is better
qps (peak) | 3.536x | higher is better
conc_qps C=1 | 3.543x | higher is better
conc_qps C=5 | 3.260x | higher is better
conc_qps C=10 | 3.536x | higher is better
serial p99 (s) | 0.260x | lower is better
insert_duration (s) | 0.869x | lower is better
optimize_duration (s) | n/a | Redis/RediSearch has no FT.OPTIMIZE; its duration here is client-side no-op overhead, not an engine build cost
load_duration (s) | 1.466x | lower is better

### Footer

- **CPU pinning** (from `setup.txt`): server on cpus `0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23`, client (vectordbbench itself, via `VDBB_TASKSET`) on cpus `8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31`.
- **Load-concurrency difference, not comparable:** Pion's rows ran with `--load-concurrency 1` (Pion's own vector benchmark loads through one connection). Redis's rows ran at VectorDBBench's default concurrent loader. `insert_duration`/`load_duration` therefore measure different ingest topologies between the two engines and are NOT a fair load-time comparison; `optimize_duration`, the search rows and recall/ndcg are unaffected by it.
- **HNSW params:** M=16, EF_CONSTRUCTION=128, EF_RUNTIME=150, DISTANCE_METRIC=COSINE, k=100, concurrency levels=[1, 5, 10], concurrency_duration=5s.
