| run | mode | build | QPS | recall@100 | serial P99 | load |
|---:|---|---|---:|---:|---:|---:|
| 1 | int8 | closed | 7,920.9 | 0.9600 | 0.7 ms | 10.0 s |
| 2 | int8 | open | 6,007.6 | 0.9598 | 1.0 ms | 9.4 s |
| 3 | int8 | open | 6,449.6 | 0.9598 | 1.0 ms | 9.0 s |
| 4 | int8 | closed | 8,728.9 | 0.9599 | 0.6 ms | 8.9 s |
| 5 | int8 | closed | 8,787.0 | 0.9602 | 0.6 ms | 8.8 s |
| 6 | int8 | open | 6,493.2 | 0.9603 | 1.0 ms | 8.8 s |
| 7 | polarquant | closed | 7,732.0 | 0.9639 | 0.8 ms | 8.9 s |
| 8 | polarquant | open | 4,893.6 | 0.9682 | 1.3 ms | 8.9 s |
| 9 | polarquant | open | 4,551.1 | 0.9675 | 1.1 ms | 9.0 s |
| 10 | polarquant | closed | 7,818.2 | 0.9681 | 0.8 ms | 9.0 s |
| 11 | polarquant | closed | 7,116.5 | 0.9680 | 0.7 ms | 9.0 s |
| 12 | polarquant | open | 4,873.8 | 0.9639 | 1.1 ms | 9.0 s |
| 13 | turboquant | closed | 6,583.5 | 0.9529 | 1.0 ms | 8.9 s |
| 14 | turboquant | open | 4,248.4 | 0.9531 | 1.4 ms | 9.0 s |
| 15 | turboquant | open | 4,017.4 | 0.9528 | 1.3 ms | 9.1 s |
| 16 | turboquant | closed | 6,418.9 | 0.9528 | 1.0 ms | 9.1 s |
| 17 | turboquant | closed | 6,362.3 | 0.9531 | 1.0 ms | 9.0 s |
| 18 | turboquant | open | 4,294.0 | 0.9532 | 1.4 ms | 9.1 s |
| 19 | nanoquant | closed | 7,335.0 | 0.4632 | 0.9 ms | 9.0 s |
| 20 | nanoquant | open | 3,150.4 | 0.4631 | 2.0 ms | 8.9 s |
| 21 | nanoquant | open | 3,448.7 | 0.4629 | 1.9 ms | 8.9 s |
| 22 | nanoquant | closed | 7,710.6 | 0.4631 | 0.9 ms | 8.8 s |
| 23 | nanoquant | closed | 7,384.6 | 0.4636 | 0.9 ms | 8.9 s |
| 24 | nanoquant | open | 3,454.9 | 0.4619 | 2.0 ms | 8.9 s |

| mode | closed median QPS | open median QPS | open vs closed, per pair | median of the pairs |
|---|---:|---:|---|---:|
| int8 | 8,728.9 | 6,449.6 | -24.2%, -26.1%, -26.1% | -26.1% |
| polarquant | 7,732.0 | 4,873.8 | -36.7%, -41.8%, -31.5% | -36.7% |
| turboquant | 6,418.9 | 4,248.4 | -35.5%, -37.4%, -32.5% | -35.5% |
| nanoquant | 7,384.6 | 3,448.7 | -57.0%, -55.3%, -53.2% | -55.3% |
