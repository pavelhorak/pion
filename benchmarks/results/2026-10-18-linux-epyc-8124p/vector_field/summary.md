# Vector field session: summary (tables only)

Every Pion row: build-once index, no live inserts (PLAN §0.2). Medians of three repeats per
grid value. Peak QPS = best of concurrency 1, 5, 10 (5 s each). Recall = serial recall@100,
1000 queries. Raw files: each block's r<n>_cmd.txt and r<n>/**/result_*.json.

## Performance1536D50K

### K50_1_pion: Pion (INT8, build-once index, no live inserts)

Run 1 load: insert 12.1 s, optimize/build 11.8 s, load_duration (time until searchable) 23.9 s

Effective ef: no auto-scale at N<=50000: effective ef = requested ef (ef < k=100 runs at k)

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 150 | 0.9597 | 6910 | 823 | 1.30 | 6051, 6910, 7300 |
| 200 | 200 | 0.9691 | 5508 | 684 | 1.50 | 5508, 6040, 5495 |
| 300 | 300 | 0.9768 | 4120 | 494 | 1.70 | 4207, 4090, 4120 |
| 400 | 400 | 0.9796 | 3824 | 445 | 2.60 | 3824, 3461, 3855 |
| 600 | 600 | 0.9818 | 3020 | 396 | 2.60 | 3040, 3020, 2881 |

### K50_2_redis_hnsw_fp32: Redis 8.10.2 + RediSearch HNSW FLOAT32

Run 1 load: insert 12.9 s, optimize/build 0.0 s, load_duration (time until searchable) 12.9 s

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 150 | 0.9630 | 1880 | 265 | 5.10 | 1880, 1911, 1862 |
| 200 | 200 | 0.9777 | 1725 | 198 | 5.70 | 1724, 1725, 1799 |
| 300 | 300 | 0.9892 | 1513 | 170 | 7.00 | 1534, 1513, 1498 |
| 400 | 400 | 0.9939 | 1334 | 158 | 8.10 | 1351, 1318, 1334 |
| 600 | 600 | 0.9973 | 1067 | 129 | 9.90 | 1064, 1069, 1067 |

### K50_3_redis_svs_lvq8: Redis 8.10.2 + RediSearch SVS-VAMANA LVQ8 (8-bit SQ on AMD)

Flags: block timed out; no DONE

Run 1 load: insert 20.7 s, optimize/build 10.0 s, load_duration (time until searchable) 30.7 s

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 150 | 0.9506 | 1888 | 248 | 4.80 | 1882, 1894 |
| 200 | 200 | 0.9561 | 1690 | 202 | 5.35 | 1696, 1683 |
| 300 | 300 | 0.9600 | 1519 | 168 | 5.90 | 1477, 1562 |
| 400 | 400 | 0.9614 | 1339 | 146 | 7.60 | 1314, 1364 |
| 600 | 600 | 0.9622 | 1049 | 108 | 9.80 | 1026, 1072 |

Unusable runs: [11]

### K50_4_valkey_fp16: Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16

Run 1 load: insert 30.6 s, optimize/build 0.0 s, load_duration (time until searchable) 30.6 s

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 150 | 0.9628 | 1555 | 231 | 5.20 | 1516, 1555, 1555 |
| 200 | 200 | 0.9774 | 1468 | 201 | 6.10 | 1468, 1460, 1494 |
| 300 | 300 | 0.9890 | 1236 | 172 | 6.90 | 1291, 1217, 1236 |
| 400 | 400 | 0.9932 | 922 | 149 | 7.90 | 861, 1065, 922 |
| 600 | 600 | 0.9966 | 740 | 121 | 10.00 | 830, 740, 738 |

### K50_5_pion: Pion (INT8, build-once index, no live inserts)

Run 1 load: insert 11.9 s, optimize/build 11.9 s, load_duration (time until searchable) 23.8 s

Effective ef: no auto-scale at N<=50000: effective ef = requested ef (ef < k=100 runs at k)

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 150 | 0.9593 | 6846 | 613 | 1.30 | 6730, 7300, 6846 |
| 200 | 200 | 0.9690 | 5295 | 606 | 1.50 | 5560, 5248, 5295 |
| 300 | 300 | 0.9769 | 4099 | 483 | 1.70 | 4200, 4038, 4099 |
| 400 | 400 | 0.9795 | 3636 | 456 | 2.60 | 3590, 3933, 3636 |
| 600 | 600 | 0.9816 | 3215 | 376 | 2.60 | 2973, 3241, 3215 |

### K50_6_valkey_fp32: Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT32

Run 1 load: insert 33.0 s, optimize/build 0.0 s, load_duration (time until searchable) 33.0 s

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 150 | 0.9629 | 1301 | 212 | 5.00 | 1280, 1311, 1301 |
| 200 | 200 | 0.9777 | 1207 | 194 | 5.90 | 1194, 1207, 1226 |
| 300 | 300 | 0.9894 | 1069 | 173 | 6.90 | 1033, 1069, 1077 |
| 400 | 400 | 0.9939 | 945 | 145 | 8.40 | 945, 986, 863 |
| 600 | 600 | 0.9973 | 712 | 118 | 10.40 | 712, 861, 644 |

### 50K: QPS at matched recall

| engine | block | precision | QPS at recall >= 0.95 (ef) | QPS at recall >= 0.97 (ef) |
|---|---|---|---:|---:|
| Pion (INT8, build-once index, no live inserts) | K50_1_pion | INT8 | 6910 (150) | 4120 (300) |
| Redis 8.10.2 + RediSearch HNSW FLOAT32 | K50_2_redis_hnsw_fp32 | FP32 | 1880 (150) | 1725 (200) |
| Redis 8.10.2 + RediSearch SVS-VAMANA LVQ8 (8-bit SQ on AMD) | K50_3_redis_svs_lvq8 | 8-bit SQ | 1888 (150) | not reached at 600 |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16 | K50_4_valkey_fp16 | FP16 | 1555 (150) | 1468 (200) |
| Pion (INT8, build-once index, no live inserts) | K50_5_pion | INT8 | 6846 (150) | 4099 (300) |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT32 | K50_6_valkey_fp32 | FP32 | 1301 (150) | 1207 (200) |

### 50K: Pion / competitor at matched recall (Pion = the lower of its blocks)

| competitor | precision | ratio at 0.95 | ratio at 0.97 |
|---|---|---:|---:|
| Redis 8.10.2 + RediSearch HNSW FLOAT32 | FP32 | 3.64x | 2.38x |
| Redis 8.10.2 + RediSearch SVS-VAMANA LVQ8 (8-bit SQ on AMD) | 8-bit SQ | 3.63x | not comparable (competitor did not reach it) |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16 | FP16 | 4.40x | 2.79x |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT32 | FP32 | 5.26x | 3.40x |

### 50K: Pion drift within the session

Median over grid values of last/first - 1 on peak QPS: **-0.9%** (ef 150: -0.9%, ef 200: -3.9%, ef 300: -0.5%, ef 400: -4.9%, ef 600: +6.5%). Within 10%.

## Performance1536D500K

### K500_1_pion: Pion (INT8, build-once index, no live inserts)

Run 1 load: insert 106.5 s, optimize/build 153.3 s, load_duration (time until searchable) 259.8 s

Effective ef: FT.SEARCH ef auto-scale active: num_nodes=500000 effective_ef=266 (baseline 150)

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 267 | 267 | 0.9622 | 3779 | 447 | 2.10 | 3650, 3799, 3779 |
| 300 | 300 | 0.9649 | 3681 | 516 | 2.10 | 3681, 3656, 3691 |
| 400 | 400 | 0.9703 | 3368 | 397 | 3.10 | 3377, 3202, 3368 |
| 600 | 600 | 0.9750 | 2442 | 348 | 3.20 | 2742, 2442, 2435 |
| 800 | 800 | 0.9768 | 2295 | 303 | 4.50 | 2295, 2179, 2300 |

### K500_2_redis_hnsw_fp32: Redis 8.10.2 + RediSearch HNSW FLOAT32

Run 1 load: insert 126.7 s, optimize/build 0.0 s, load_duration (time until searchable) 126.7 s

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 150 | 0.9442 | 1688 | 210 | 5.50 | 1707, 1680, 1688 |
| 200 | 200 | 0.9605 | 1545 | 191 | 6.50 | 1531, 1566, 1545 |
| 300 | 300 | 0.9762 | 1363 | 155 | 8.00 | 1363, 1372, 1362 |
| 400 | 400 | 0.9836 | 1187 | 144 | 9.60 | 1187, 1170, 1221 |
| 600 | 600 | 0.9902 | 950 | 108 | 12.40 | 955, 926, 950 |

### K500_3_redis_svs_lvq8: Redis 8.10.2 + RediSearch SVS-VAMANA LVQ8 (8-bit SQ on AMD)

Run 1 load: insert 240.0 s, optimize/build 10.0 s, load_duration (time until searchable) 250.0 s

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 150 | 0.9344 | 1709 | 201 | 5.30 | 1737, 1701, 1709 |
| 200 | 200 | 0.9426 | 1569 | 176 | 6.00 | 1530, 1623, 1569 |
| 300 | 300 | 0.9490 | 1365 | 153 | 7.30 | 1365, 1351, 1381 |
| 400 | 400 | 0.9521 | 1163 | 124 | 8.70 | 1135, 1170, 1163 |
| 600 | 600 | 0.9544 | 913 | 94 | 11.40 | 913, 901, 921 |

### K500_4_valkey_fp16: Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16

Run 1 load: insert 360.8 s, optimize/build 0.0 s, load_duration (time until searchable) 360.8 s

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 150 | 150 | 0.9441 | 1338 | 210 | 5.90 | 1384, 1283, 1338 |
| 200 | 200 | 0.9608 | 1230 | 191 | 6.60 | 1232, 1220, 1230 |
| 300 | 300 | 0.9764 | 1119 | 162 | 7.60 | 1172, 1119, 1116 |
| 400 | 400 | 0.9837 | 1059 | 147 | 8.90 | 1070, 1025, 1060 |
| 600 | 600 | 0.9903 | 723 | 112 | 11.50 | 858, 723, 720 |

### K500_5_pion: Pion (INT8, build-once index, no live inserts)

Run 1 load: insert 111.6 s, optimize/build 154.0 s, load_duration (time until searchable) 265.6 s

Effective ef: FT.SEARCH ef auto-scale active: num_nodes=500000 effective_ef=266 (baseline 150)

| ef (requested) | ef (effective) | recall | peak QPS | QPS at C=1 | serial P99 ms | repeats (peak QPS) |
|---:|---:|---:|---:|---:|---:|---|
| 267 | 267 | 0.9623 | 3766 | 500 | 2.50 | 3704, 3916, 3766 |
| 300 | 300 | 0.9647 | 3710 | 440 | 2.80 | 3710, 3601, 3791 |
| 400 | 400 | 0.9705 | 3250 | 393 | 3.10 | 3283, 3250, 3210 |
| 600 | 600 | 0.9750 | 2599 | 386 | 3.70 | 2738, 2598, 2559 |
| 800 | 800 | 0.9769 | 2220 | 293 | 4.50 | 2345, 2198, 2220 |

### 500K: QPS at matched recall

| engine | block | precision | QPS at recall >= 0.95 (ef) | QPS at recall >= 0.97 (ef) |
|---|---|---|---:|---:|
| Pion (INT8, build-once index, no live inserts) | K500_1_pion | INT8 | 3779 (267) | 3368 (400) |
| Redis 8.10.2 + RediSearch HNSW FLOAT32 | K500_2_redis_hnsw_fp32 | FP32 | 1545 (200) | 1363 (300) |
| Redis 8.10.2 + RediSearch SVS-VAMANA LVQ8 (8-bit SQ on AMD) | K500_3_redis_svs_lvq8 | 8-bit SQ | 1163 (400) | not reached at 600 |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16 | K500_4_valkey_fp16 | FP16 | 1230 (200) | 1119 (300) |
| Pion (INT8, build-once index, no live inserts) | K500_5_pion | INT8 | 3766 (267) | 3250 (400) |

### 500K: Pion / competitor at matched recall (Pion = the lower of its blocks)

| competitor | precision | ratio at 0.95 | ratio at 0.97 |
|---|---|---:|---:|
| Redis 8.10.2 + RediSearch HNSW FLOAT32 | FP32 | 2.44x | 2.38x |
| Redis 8.10.2 + RediSearch SVS-VAMANA LVQ8 (8-bit SQ on AMD) | 8-bit SQ | 3.24x | not comparable (competitor did not reach it) |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16 | FP16 | 3.06x | 2.90x |

### 500K: Pion drift within the session

Median over grid values of last/first - 1 on peak QPS: **-0.3%** (ef 267: -0.3%, ef 300: +0.8%, ef 400: -3.5%, ef 600: +6.4%, ef 800: -3.3%). Within 10%.

## Streaming case, 50K (StreamingPerformanceCase, 200 rows/s, stages 0.5 / 0.8, then end of write, then after optimize)

| engine | stage | recall | max QPS | QPS by concurrency (5, 10) |
|---|---|---:|---:|---|
| Pion (INT8, build-once index, no live inserts) | 50% inserted | 0.0000 | no results (QPS of empty answers not reported) |  |
| Pion (INT8, build-once index, no live inserts) | 80% inserted | 0.0000 | no results (QPS of empty answers not reported) |  |
| Pion (INT8, build-once index, no live inserts) | write finished | 0.0000 | no results (QPS of empty answers not reported) |  |
| Pion (INT8, build-once index, no live inserts) | after optimize() | 0.9600 | 7118 | 3527, 7118 |
| Pion (INT8, build-once index, no live inserts) | insert 250 s, optimize 11.8 s | | | |
| Redis 8.10.2 + RediSearch HNSW FLOAT32 | 50% inserted | 0.5106 | 1842 | 1036, 1842 |
| Redis 8.10.2 + RediSearch HNSW FLOAT32 | 80% inserted | 0.7974 | 1806 | 1034, 1806 |
| Redis 8.10.2 + RediSearch HNSW FLOAT32 | write finished | 0.9628 | 1881 | 1063, 1881 |
| Redis 8.10.2 + RediSearch HNSW FLOAT32 | after optimize() | 0.9628 | 1895 | 1075, 1895 |
| Redis 8.10.2 + RediSearch HNSW FLOAT32 | insert 250 s, optimize 0.0 s | | | |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16 | 50% inserted | 0.5084 | 1388 | 741, 1388 |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16 | 80% inserted | 0.7974 | 1379 | 727, 1379 |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16 | write finished | 0.9631 | 1548 | 828, 1548 |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16 | after optimize() | 0.9631 | 1542 | 805, 1542 |
| Valkey 9.1.2 + valkey-search 1.3.0-rc1 HNSW FLOAT16 | insert 250 s, optimize 0.0 s | | | |

## Chain record (cuts, stops, timeouts, preflight findings)

- 2026-10-18 16:30:56 start B_build (timeout 90 min, estimate 60 min, hours used 1, cap 7)
- 2026-10-18 16:55:10 start SETUP (timeout 10 min, estimate 5 min, hours used 1, cap 7)
- 2026-10-18 16:55:13 start K50_1_pion (timeout 40 min, estimate 20 min, hours used 1, cap 7)
- 2026-10-18 17:07:19 start K50_2_redis_hnsw_fp32 (timeout 40 min, estimate 15 min, hours used 1, cap 7)
- 2026-10-18 17:20:19 start K50_3_redis_svs_lvq8 (timeout 40 min, estimate 15 min, hours used 2, cap 7)
- 2026-10-18 18:00:22 end K50_3_redis_svs_lvq8: TIMEOUT at 40 min: did not finish
- 2026-10-18 18:00:22 start K50_4_valkey_fp16 (timeout 40 min, estimate 25 min, hours used 2, cap 7)
- 2026-10-18 18:13:10 start K50_5_pion (timeout 40 min, estimate 22 min, hours used 2, cap 7)
- 2026-10-18 18:24:36 start K50_6_valkey_fp32 (timeout 40 min, estimate 21 min, hours used 3, cap 7)
- 2026-10-18 18:37:54 start S50_1_pion (timeout 25 min, estimate 8 min, hours used 3, cap 7)
- 2026-10-18 18:46:13 start S50_2_redis_hnsw_fp32 (timeout 25 min, estimate 10 min, hours used 3, cap 7)
- 2026-10-18 18:55:29 start S50_3_valkey_fp16 (timeout 25 min, estimate 11 min, hours used 3, cap 7)
- 2026-10-18 19:04:24 start K500_1_pion (timeout 90 min, estimate 35 min, hours used 3, cap 7)
- 2026-10-18 19:20:34 start K500_2_redis_hnsw_fp32 (timeout 90 min, estimate 19 min, hours used 4, cap 7)
- 2026-10-18 19:36:07 start K500_3_redis_svs_lvq8 (timeout 90 min, estimate 19 min, hours used 4, cap 7)
- 2026-10-18 19:53:18 start K500_4_valkey_fp16 (timeout 90 min, estimate 19 min, hours used 4, cap 7)
- 2026-10-18 20:12:02 start K500_5_pion (timeout 90 min, estimate 20 min, hours used 4, cap 7)

