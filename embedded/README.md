# Embedded Pion

The Pion engine inside your Python process, behind redis-py. No server.

```python
import pionlite as pion                # the pionlite wheel, attached to each GitHub release
r = pion.Redis(dir="./data")           # a redis.Redis: pipelines, Lock, pub/sub, scripts
ar = pion.asyncio.Redis(dir="./data")  # a redis.asyncio.Redis
r = pion.from_url("pion:///abs/path/to/data")
pool = pion.asyncio.ConnectionPool.from_url("pion:///abs/path/to/data")  # for redis.asyncio.Redis(connection_pool=pool)
```

- Every write goes to the server's write-ahead log in `dir` as the command
  runs; an acknowledged write survives `kill -9`.
- One engine and one data directory per process. A second process opening the
  directory is refused (`dir/pion.lock`): to share data between processes, run
  `pion-server` on it.
- KV only (no vector search). Built and tested on macOS arm64, Linux x86-64
  and Linux arm64 by `.github/workflows/embedded.yml` (pull requests that touch
  the library, and manual runs). Each release built from a `v*` tag carries
  the three wheels (`pionlite-<version>-cp311-abi3-*.whl`, one per
  platform, for CPython 3.11 and later). The Linux wheels
  are `manylinux_2_35` (glibc 2.35 or later, Ubuntu 22.04 for one), built on
  Ubuntu 22.04; macOS needs 14 or later.
- The engine's log lines (WAL replay and snapshot load at open, `Snapshot:
  saved` on SAVE, warnings) go to `dir/pion.log`, never to the host's stdout.
  `logfile PATH` in the engine's config moves them.

## Known gaps

- One engine per process; `memory://`-style isolated instances per URL are
  not possible.
- `pion-server` takes `pion.lock` only with `--dir`. A server started without
  it, in the directory an app has open, is not stopped.

## Engine options

`pion.Redis(dir=..., appendonly=..., wal_size_mb=..., maxmemory=...,
maxmemory_policy=...)`. The C ABI's config (`pion_embed_open`, one `key value`
per line) also takes `wal-max-segments`, `wal-full-policy`, `blob-threshold`,
`no-blob-tier`, `requirepass`, `enable-debug-command`, `maxmemory-samples`,
`lua-time-limit`, `lua-memory-limit` and `logfile`, with the values
`pion-server`'s flags of the same names take (`maxmemory` and
`lua-memory-limit` in Redis units). A bad value fails the open with the reason.

## Waiting for output

`pion_embed_wake_fd(h, conn)` returns a descriptor that is readable while the
connection has output the caller has not collected: a parked command's reply
(BLPOP, XREADGROUP BLOCK, WAIT), a pub/sub message another connection
published. The sync adapter waits on it with `select`, the asyncio adapter
with `loop.add_reader`, so neither polls. A push answers a blocked client on
another connection before the pushing call returns, as the server's loop
does; a timeout is answered by the engine's periodic tick.

## Build and test

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 pixi run build-embed   # -> libpion.dylib (~6 min)
PYTHONPATH=embedded/python python3 embedded/tests/test_smoke.py
bash embedded/build_wheel.sh python3      # -> embedded/dist/*-cp311-abi3-macosx_14_0_arm64.whl
python3 embedded/tests/test_graduation.py ./pion-server   # app <-> pion-server --dir, one at a time
# the server's test tier, through the library: a stand-in for pion-server
# serves each test from libpion behind a loopback proxy. Tests that need a
# flag the library has no equivalent for (--kvcache, --cluster, -w N, ...)
# are reported as skipped, with the flag named.
python3 tests/run_all.py --server-tests-only --binary embedded/tests/pion-server-embedded
# or one suite by hand:
PYTHONPATH=embedded/python python3 -m pionlite.proxy --dir /tmp/d --port 16700 &
python3 tests/test_raw.py --port 16700 && python3 tests/test_parity.py --port 16700
```

The library looks for `libpion.dylib` next to the package (`_lib/`), at the
repo root, or at `$PION_EMBED_LIB`.

## How it works

`src/lib/pion_embed.mojo` sets the engine up as the `--io-threads` executor
(#84) with no I/O threads, and plays the I/O thread itself: a request goes
into the connection's input buffer, `_dispatch_io_batch` runs it, and the
reply is taken from the connection's output queue. Pub/sub deliveries and
parked commands (BLPOP, XREADGROUP BLOCK, WAIT) use the same queue. Calls run
on the caller's thread under one engine mutex, through the stable-ABI
extension `_pion` (one cp311-abi3 wheel for CPython >= 3.11; it releases the
GIL; `PION_EMBED_CTYPES=1` falls back to ctypes); a
daemon thread ticks the engine (`Engine._tick_loop`).

