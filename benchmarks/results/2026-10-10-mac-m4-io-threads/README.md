# 2026-10-10: `--io-threads` on a Mac (one keyspace)

The Mac half of the I/O-thread work: what `--io-threads N` does on macOS
(kqueue), and whether the default (`--io-threads 1`) got slower. The Linux
acceptance numbers come from `benchmarks/kv-io-threads/run_on_box.sh` on the
EPYC 8124P that measured 2026-10-06, and are not in this directory.

- Machine: Apple M4 Mac mini, 10 cores (4 performance, 6 efficiency)
  ([`cpu.txt`](cpu.txt)). Client and server on the same machine, over loopback.
- Pion: `pion-server 0.9.8+58e5bcb`, `pixi run build`
  ([`pion_version.txt`](pion_version.txt)). Redis 8.10.2, Homebrew
  ([`redis_version.txt`](redis_version.txt)). memtier 2.5.1
  ([`memtier_version.txt`](memtier_version.txt)).
- **A Chrome renderer held one core at 100% for the whole session.** It was
  not ours to stop, so every number here was taken with it running. Both
  comparisons below are interleaved and paired, so both sides ran under it; the
  absolute numbers are not gate numbers.

## 1. Pion `--io-threads N` against Redis `io-threads N`

[`sweep.sh`](sweep.sh), summarized by [`summarize.py`](summarize.py) into
[`summary.md`](summary.md): the 2026-10-06 memtier settings (256-byte values,
1:10 SET:GET over 1M keys, 50 connections per client thread, persistence off on
both), but **4 client threads instead of 8, and 10 s runs**. With 8 client
threads, the client alone would take most of a 10-core machine. Each config
ran 3 times, rounds interleaved; medians shown. Raw output is in
[`raw/`](raw/) and [`raw_json.tar.gz`](raw_json.tar.gz).

| P | threads | Redis ops/sec | Pion ops/sec | Pion ÷ Redis | Pion ÷ Pion at 1 thread |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 296,529 | 372,633 | 1.26 | 1.00 |
| 1 | 2 | 320,012 | 301,489 | 0.94 | 0.81 |
| 1 | 4 | 213,052 | 217,819 | 1.02 | 0.58 |
| 10 | 1 | 1,592,002 | 2,667,748 | 1.68 | 1.00 |
| 10 | 2 | 2,107,459 | 2,508,837 | 1.19 | 0.94 |
| 10 | 4 | 2,013,187 | 2,202,174 | 1.09 | 0.83 |
| 50 | 1 | 2,382,716 | 4,981,035 | 2.09 | 1.00 |
| 50 | 2 | 3,013,713 | 4,884,903 | 1.62 | 0.98 |
| 50 | 4 | 2,160,531 | 5,318,386 | 2.46 | 1.07 |

**On this Mac, I/O threads do not raise Pion's throughput, and they lower it
at P=1 and P=10.** Pion's single thread leads Redis's best configuration at
every pipeline depth (P=1: 372,633 vs 320,012 at Redis `io-threads 2`).

[`cpu_probe.sh`](cpu_probe.sh) ([`cpu_probe.txt`](cpu_probe.txt)) shows why,
sampling 5 s into a run:

| P=1 | `--io-threads 1` | `--io-threads 4` |
|---|---:|---:|
| ops/sec | 369,704 | 217,483 |
| memtier CPU | 186% | 374% |
| server threads, % of a core | 99 | 100 · 99 · 82 · 47 |
| machine idle | 48% | 8.6% (75.7% system time) |

At `--io-threads 1` the server's one thread is saturated while half the machine
is idle. At 4, the machine is saturated, mostly in the kernel, and the
client spends about three times the CPU per request. Redis loses the same way
at `io-threads 4` (P=1: 213,052 against 296,529 at 1). On this machine, socket
I/O spread over more threads costs more in the kernel and in client wake-ups
than it saves the server. The EPYC has 16 cores, the client on separate cores,
and a Linux network stack, so it is where the design gets measured.

## 2. `--io-threads 1` against main: no regression

[`ab_io1.sh`](ab_io1.sh): the KV gate harness (`valkey-benchmark.py -c 50
-n 100000 -P 10 -w 1 --pion-only --gate`), main's binary (`0.9.7+805636b`, the
code of release 0.9.8) against the branch, in the order main, branch, branch,
main, twice. [`ab_compare.py`](ab_compare.py) produced
[`ab_io1_summary.md`](ab_io1_summary.md) from the per-run tables in
[`ab_io1/`](ab_io1/).

- Geometric mean of branch ÷ main over the 21 rows: **1.0000**.
- Largest row differences: INCR +3.6% (the branch faster in 3 of 4 pairs) and
  RPOP −2.3%. At 100,000 requests a row's result falls on the harness's timer
  steps: 2,325,581, 2,380,952 and 2,439,024 ops/sec are adjacent steps, about
  2.4% apart. INCR moved by one to two steps, RPOP by one.
- The only row under its floor was MSET (10 keys), on **both** binaries: main
  in 3 of 4 runs, the branch in 2 of 4. Its median was 1,869,322 on main and
  1,889,482 on the branch. MSET has the least margin of any row, and one core
  was busy with Chrome.
- Codegen: `process_data_plane`, `_get_burst`, `_mset_frame`,
  `_dispatch_recv_buffer` and `process_slow_path` are byte-for-byte the same
  size in both binaries (`nm -n`). `_flush_kqueue` grew from 796 to 992 bytes,
  all of it on the early exit taken only with I/O threads.
