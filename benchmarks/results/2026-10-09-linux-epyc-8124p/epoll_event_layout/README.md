# x86-64 `--epoll` read epoll events with the wrong layout

EPYC 8124P (Zen 4c, 16 cores), Ubuntu 24.04, kernel 6.8. One worker
(`-w 1 --profile kv --no-wal`), portable builds. Load: memtier, 8 threads x 50
connections, 256-byte values, 1:10 SET:GET over 1M keys, 30 s per run, 3 runs.
Captured 2026-10-09.

The release reads each `struct epoll_event` as 16 bytes; x86-64 Linux packs it
to 12. Only the first event of each `epoll_wait` batch decodes correctly; the
rest are read from the wrong bytes. A garbage event mask can carry
EPOLLERR/EPOLLHUP, and the loop then closes a healthy connection: memtier
reports `Connection reset by peer`, and the run's totals become meaningless
(the `⚠` rows).

ops/sec per run (run 1 / 2 / 3):

| Build | P | ops/sec |
|---|---:|---|
| release `--epoll` (16-byte reads) | 1 | 639,100,882 ⚠ 7 resets / 110,945 / 648,341,025 ⚠ 7 resets |
| release `--epoll` (16-byte reads) | 10 | 8,431,928,313 ⚠ 6 resets / 913,082 ⚠ 2 resets / 6,377,754,127 ⚠ 5 resets |
| release `--epoll` (16-byte reads) | 50 | 5,974,073,999 ⚠ 7 resets / 2,205,953 ⚠ 2 resets / 2,232,973 ⚠ 3 resets |
| fixed `--epoll` (this branch) | 1 | 125,265 / 125,578 / 126,468 |
| fixed `--epoll` (this branch) | 10 | 996,532 / 967,994 / 968,545 |
| fixed `--epoll` (this branch) | 50 | 1,986,335 / 1,918,954 / 1,919,850 |
| release io_uring (default) | 1 | 115,470 / 115,276 / 115,216 |
| release io_uring (default) | 10 | 878,906 / 851,995 / 854,984 |
| release io_uring (default) | 50 | 1,621,304 / 1,595,737 / 1,591,360 |

Files: `<arm>_P<depth>_clean_r<run>.txt`, memtier's own output.
