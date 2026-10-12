# 2026-10-12: `--io-threads` against Redis's `io-threads` on one Linux server

One keyspace: Pion `--io-threads N` against Redis 8.10.2 `io-threads N`, on the
same machine, the same day, with the same client settings as the
[2026-10-06 run](../2026-10-06-linux-epyc-8124p/README.md).

- **Machine:** AMD EPYC 8124P, 16 cores / 32 threads, bare metal, Ubuntu 24.04
  ([`lscpu.txt`](lscpu.txt), [`uname.txt`](uname.txt)). This is the same machine type
  as 2026-10-06, but a different machine.
- **Pion:** `pion-server 0.9.9+b74b82a` (the v0.9.9 release commit), built on the
  machine with `pixi run build-portable`, which is the release's linux-x86_64 build
  (x86-64-v2) ([`pion_version.txt`](pion_version.txt)).
- **Redis:** 8.10.2, built from the tag ([`redis_version.txt`](redis_version.txt)).
- **Client:** `memtier_benchmark` built from source at `0333be07`
  ([`memtier_version.txt`](memtier_version.txt)). The 2026-10-06 run used `f8d4458d`.
- **Correctness first:** `tests/test_gh465_io_threads.py` passed 60/0 against this
  binary on this machine before any measurement
  ([`smoke_test_gh465_io_threads.log`](smoke_test_gh465_io_threads.log)).

## How it ran

[`benchmarks/kv-io-threads/run_on_box.sh`](../../kv-io-threads/run_on_box.sh), at
the commit above, ran every config and wrote this directory.
[`summary.md`](summary.md) holds the medians over 3 repetitions, and
[`thread_cpu.md`](thread_cpu.md) the CPU time of each Pion thread over each run,
read from `/proc/<pid>/task`.

Settings ([`settings.txt`](settings.txt)):
- 8 client threads x 50 connections;
- 256-byte values, 1:10 SET:GET, keys drawn at random from 1M;
- 30 s per run, `--distinct-client-seed`;
- client and server on the same machine, no CPU pinning.

Neither engine persists in the throughput runs (Redis `--save "" --appendonly no`,
Pion `--no-wal`).

- **One keyspace:** Redis with `io-threads` 1, 4 and 8, and `pion-server -w 1 --profile kv`
  with `--io-threads` 1, 2, 4 and 8. Both count threads the same way: N is one
  main (Redis) or executor (Pion) thread plus N - 1 I/O threads.
  `--io-threads 1` is Pion's single-threaded event loop, the `-w 1` of 2026-10-06.
- **Durable, P=10:** Redis with `appendonly yes, appendfsync everysec` at `io-threads`
  4 and 8; Pion with its WAL on at `--io-threads` 1, 4 and 8.

Raw memtier output: [`raw/`](raw/). The JSON, per-thread CPU samples and server
logs are in [`raw_json.tar.gz`](raw_json.tar.gz).

**One condition to know:** the machine's RAID1 was resyncing after a fresh install.
It was throttled to 1 MB/s before every run except the first; that run's median is
unaffected ([`raid_resync.txt`](raid_resync.txt)).

## Results (ops/sec, median of 3)

| One keyspace | Redis, 1 thread | Redis, `io-threads 8` | Pion, 1 thread | **Pion, `--io-threads 8`** | io8 against io8 |
|---|---:|---:|---:|---:|---:|
| P=1 | 110,044 | 386,278 | 114,909 | **515,267** | 1.33x |
| P=10 | 640,985 | 1,661,596 | 895,149 | **2,972,713** | 1.79x |
| P=50 | 979,984 | 1,368,935 | 1,703,363 | **3,909,190** | 2.86x |

At P=50, `--io-threads 4` measured 4,431,024, faster than 8. `io-threads 4` for
Redis, and `--io-threads 2` and 4 for Pion, are in [`summary.md`](summary.md).

| Durable, P=10 | Redis AOF `everysec` | Pion WAL | Ratio |
|---|---:|---:|---:|
| `io-threads 4` | 1,237,988 | 2,409,492 | 1.95x |
| `io-threads 8` | 1,297,102 | **2,724,529** | 2.10x |

Pion's single thread with its WAL measured 856,140.

**Spread:** Pion at `--io-threads 8` and P=10 varied 15% over its 3 runs
(2,730,502 to 3,135,046); every Redis row stayed within 4%.

## What limits each depth

[`thread_cpu.md`](thread_cpu.md) lists each run's threads by CPU time, without
names. The executor is the one thread whose load differs from the others: the I/O
threads share the connections, so their loads come out about even.

- **P=1, `--io-threads 4`:** 0.99, 0.99, 0.99 and 0.37 cores. The three I/O threads
  are saturated and the executor is not, so more I/O threads help. `--io-threads 8`
  is 1.8x `--io-threads 4` here.
- **P=50, `--io-threads 8`:** one thread at 0.99, the other seven at 0.66 to 0.87.
  The executor is the limit. More I/O threads add hand-off cost, which is why 8
  comes in under 4.
- **P=50, `--io-threads 4`:** 1.00, 0.95, 0.94 and 0.94. The threads are too close
  to tell apart.

## What this does not show

- **No Mac numbers.** On a Mac with the client on the same machine, I/O threads
  lower throughput for both engines
  ([2026-10-10](../2026-10-10-mac-m4-io-threads/README.md)). They pay on a server
  with cores to spare.
- **No multi-keyspace comparison.** Across all cores, the 2026-10-06 run is the
  comparison, and there 16 Redis processes lead 16 Pion workers.
- **One machine type, one client.** On another CPU, run the same script and use
  its own numbers.
