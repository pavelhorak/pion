# Phase B box run: summary

# Vector field session 2026-10-20, box setup  
## host  
pion-node-1  
6.8.0-146-generic  
Ubuntu 24.04.5 LTS  
32  
CPU(s):                                  32  
Model name:                              AMD EPYC 8124P 16-Core Processor  
BIOS Model name:                         AMD EPYC 8124P 16-Core Processor                 CPU @ 2.4GHz  
Thread(s) per core:                      2  
Core(s) per socket:                      16  
CPU(s) scaling MHz:                      77%  
CPU max MHz:                             3003.9109  
L3 cache:                                64 MiB (4 instances)  
NUMA node(s):                            1  
NUMA node0 CPU(s):                       0-31  
flags: avx512_bf16 avx512_vnni avx512f   
## lscpu -e=CPU,CORE,CACHE (last CACHE id = L3 domain)  
CPU CORE L1d:L1i:L2:L3  
  0    0 0:0:0:0  
  1    1 1:1:1:0  
  2    2 2:2:2:0  
  3    3 3:3:3:0  
  4    4 16:16:16:4  
  5    5 17:17:17:4  
  6    6 18:18:18:4  
  7    7 19:19:19:4  
  8    8 8:8:8:2  
  9    9 9:9:9:2  
 10   10 10:10:10:2  
 11   11 11:11:11:2  
 12   12 24:24:24:6  
 13   13 25:25:25:6  
 14   14 26:26:26:6  
 15   15 27:27:27:6  
 16    0 0:0:0:0  
 17    1 1:1:1:0  
 18    2 2:2:2:0  
 19    3 3:3:3:0  
 20    4 16:16:16:4  
 21    5 17:17:17:4  
 22    6 18:18:18:4  
 23    7 19:19:19:4  
 24    8 8:8:8:2  
 25    9 9:9:9:2  
 26   10 10:10:10:2  
 27   11 11:11:11:2  
 28   12 24:24:24:6  
 29   13 25:25:25:6  
 30   14 26:26:26:6  
 31   15 27:27:27:6  
## pinning (constant all session)  
server cpus: 0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23  
client cpus: 8,9,10,11,12,13,14

## S50: StreamingPerformanceCase, OpenAI 50K, 200 rows/s, stock client for Pion

| block | 50% | 80% | write finished | after optimize() | max QPS by stage | builds | per-worker recall after (min / max) |
|---|---:|---:|---:|---:|---|---:|---|
| S50_1_pion_a | 0.5138 | 0.7993 | 0.9669 | 0.9599 | 2,155 / 2,928 / 6,709 / 7,077 | 40 | 0.9573 / 0.9573 (16 workers) |
| S50_2_redis_hnsw_fp32 | 0.5084 | 0.7974 | 0.9629 | 0.9629 | 1,797 / 1,752 / 1,852 / 1,836 | - | - |
| S50_3_pion_b | 0.5097 | 0.7998 | 0.9672 | 0.9597 | 3,125 / 3,128 / 6,694 / 6,721 | 35 | 0.9572 / 0.9572 (16 workers) |
| S50_4_pion_a | 0.5132 | 0.8001 | 0.9671 | 0.9602 | 2,126 / 2,905 / 6,954 / 6,662 | 40 | 0.9581 / 0.9581 (16 workers) |
| S50_5_pion_b | 0.5086 | 0.7993 | 0.9678 | 0.9601 | 3,049 / 3,276 / 6,856 / 6,935 | 35 | 0.9581 / 0.9581 (16 workers) |

## D500_1_pion_b: Pion B at 500K, -w 1, one client (delta_steps.py)

```
[   39.1s] ingested 500000 rows in 15.5 s; state (index, graph, exact, builds, running) ('exact', 0, 500000, 0, 0)
[  107.7s] first build: settled 68.6 s after the ingest ended; state ('graph', 500000, 0, 1, 0)
[  109.3s] RESULT delta 0: p50 1.613 ms  p90 1.849 ms (graph only)
[  113.6s] RESULT delta 10000: p50 3.444 ms  p90 3.699 ms  added 1.831 ms = 113.5% of the delta-0 p50 (client RTT included in it)
[  120.0s] RESULT delta 25000: p50 5.470 ms  p90 5.729 ms  added 3.857 ms = 239.1% of the delta-0 p50 (client RTT included in it)
[  129.3s] RESULT delta 45000: p50 8.103 ms  p90 8.361 ms  added 6.489 ms = 402.3% of the delta-0 p50 (client RTT included in it)
[  129.3s] RESULT fit over the 3 delta points: added = 0.512 ms fixed + 0.1330 ms per 1,000 delta rows
[  129.3s] RESULT at a 34000-row delta the fit gives +5.035 ms = 312.1% of the delta-0 p50
[  129.3s] RESULT at a 61000-row delta the fit gives +8.626 ms = 534.7% of the delta-0 p50
[  210.5s] RESULT rebuild published 81.0 s after the last row was added (client clock; the server's 'indexed in' line is the build's own time); state ('graph+delta', 550000, 1000, 2, 0)
[  210.5s] RESULT during the rebuild: 23 batches of 200 queries, delta 51000-1000, p50 median 30.763 ms, max 31.711 ms
[  212.5s] RESULT after the rebuild: p50 2.021 ms  p90 2.263 ms; state ('graph+delta', 550000, 1000, 2, 0)
--- server
[vector] auto-build: indexing 500000 vectors in the background; FT.SEARCH answers exactly until it is done
[vector] auto-build: 500000 vectors indexed in 67323 ms; FT.SEARCH uses the graph now
[vector] auto-build: re-indexing 550000 vectors in the background; the current index keeps serving
[vector] auto-build: 550000 vectors indexed in 78794 ms; FT.SEARCH uses the graph now
```

## D500_2_pion_a: Pion A at 500K, -w 1, one client (delta_steps.py)

```
[   35.5s] ingested 500000 rows in 13.5 s; state (index, graph, exact, builds, running) ('exact', 0, 500000, 0, 0)
[  103.6s] first build: settled 68.1 s after the ingest ended; state ('graph', 500000, 0, 1, 0)
[  105.2s] RESULT delta 0: p50 1.616 ms  p90 1.851 ms (graph only)
[  109.8s] RESULT delta 10000: p50 3.820 ms  p90 4.083 ms  added 2.204 ms = 136.4% of the delta-0 p50 (client RTT included in it)
[  117.2s] RESULT delta 25000: p50 6.481 ms  p90 6.753 ms  added 4.865 ms = 301.0% of the delta-0 p50 (client RTT included in it)
[  128.3s] RESULT delta 45000: p50 9.969 ms  p90 10.214 ms  added 8.353 ms = 516.9% of the delta-0 p50 (client RTT included in it)
[  128.3s] RESULT fit over the 3 delta points: added = 0.457 ms fixed + 0.1756 ms per 1,000 delta rows
[  128.3s] RESULT at a 34000-row delta the fit gives +6.429 ms = 397.8% of the delta-0 p50
[  128.3s] RESULT at a 61000-row delta the fit gives +11.171 ms = 691.2% of the delta-0 p50
[  213.2s] RESULT rebuild published 84.7 s after the last row was added (client clock; the server's 'indexed in' line is the build's own time); state ('building', 551000, 0, 2, 0)
[  214.8s] RESULT after the rebuild: p50 1.631 ms  p90 1.880 ms; state ('building', 551000, 0, 2, 0)
--- server
[vector] auto-build: indexing 500000 vectors in the background; FT.SEARCH answers exactly until it is done
[vector] auto-build: 500000 vectors indexed in 66731 ms; FT.SEARCH uses the graph now
[vector] auto-build: indexing 551000 vectors in the background; FT.SEARCH answers exactly until it is done
[vector] auto-build: 551000 vectors indexed in 84091 ms; FT.SEARCH uses the graph now
```
