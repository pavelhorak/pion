# Embedded Pion (phase 0 prototype)

The Pion engine inside your Python process, behind redis-py. No server.

```python
import pion_embedded as pion           # working name; not on PyPI
r = pion.Redis(dir="./data")           # a redis.Redis: pipelines, Lock, pub/sub, scripts
ar = pion.asyncio.Redis(dir="./data")  # a redis.asyncio.Redis
r = pion.from_url("pion:///abs/path/to/data")
```

- Every write goes to the server's write-ahead log in `dir` as the command
  runs; an acknowledged write survives `kill -9`.
- One engine and one data directory per process. A second process opening the
  directory is refused (`dir/pion.lock`): to share data between processes, run
  `pion-server` on it.
- KV only (no vector search). macOS arm64 is the only platform built and run;
  the Linux targets compile.

## Known gaps

- **The engine prints its log lines to the host's stdout** (WAL replay and
  snapshot load at open, `Snapshot: saved` on SAVE, warnings). The plan's §5.6
  wants a silent library; that needs the engine's prints routed through one
  switch.
- One engine per process; `memory://`-style isolated instances per URL are
  not possible.
- No Linux wheel yet (the targets compile; linking and running them needs CI).
- `pion-server` takes `pion.lock` only with `--dir`. A server started without
  it, in the directory an app has open, is not stopped.

## Build and test

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 pixi run build-embed   # -> libpion.dylib (~6 min)
PYTHONPATH=embedded/python python3 embedded/tests/test_smoke.py
bash embedded/build_wheel.sh python3      # -> embedded/dist/*-py3-none-macosx_14_0_arm64.whl
python3 embedded/tests/test_graduation.py ./pion-server   # app <-> pion-server --dir, one at a time
# the server's own suites, through the library:
PYTHONPATH=embedded/python python3 -m pion_embedded.proxy --dir /tmp/d --port 16700 &
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

