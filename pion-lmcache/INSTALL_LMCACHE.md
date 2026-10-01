# Validating Pion against real LMCache (Linux runbook)

LMCache's wheel hard-depends on CUDA at install time
(`torch.utils.cpp_extension._join_cuda_home → CUDA_HOME not set`), so
this can't run on a Mac. On a Linux host with NVIDIA GPU + CUDA toolkit
installed, here's the validation runbook for the wire-compat path
(`remote_url: resp://...` against a running pion-server).

## 0. Prerequisites

```bash
# CUDA install verification
nvidia-smi
echo "$CUDA_HOME"          # must be set, e.g. /usr/local/cuda

# Python 3.10..3.13 (LMCache 0.4.x requires <3.14)
python3 --version

# A pion-server binary built for Linux (see README.md, "Quick Start").
ls pion-server
```

## 1. Install LMCache

```bash
python3 -m venv lmc_venv
source lmc_venv/bin/activate
pip install lmcache
```

If `pip install lmcache` builds from sdist and CUDA detection fails,
verify `nvcc --version` and `$CUDA_HOME/bin/nvcc` resolve. The C++
RESPClient extension is required for `remote_url: resp://`.

## 2. Start pion-server with kvcache and the standard listen port

```bash
./pion-server --kvcache -w 1 -p 1974 &
sleep 2
```

A6 wire compatibility serves LMCache's GET/SET/EXISTS/DEL out of the
box; no further Pion-side configuration is needed.

## 3. Wire pion-lmcache (optional — only if NOT using `remote_url`)

The simplest LMCache integration uses the stock `RESPConnector`:

```yaml
# lmcache_config.yaml
chunk_size: 256
local_cpu: true
remote_url: resp://localhost:1974
```

LMCache constructs its own `RESPConnector` from this URL — `pion-lmcache`'s
Python adapters aren't in the call path. Path 2/3 (LMCacheRemoteBackend
or PionStore) are for callers who want to bypass LMCache entirely.

## 4. Hello-world: round-trip a fake KV cache through Pion

Save as `lmcache_pion_smoke.py`:

```python
import asyncio
import torch
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.utils import CacheEngineKey

# Build a config pointing at Pion via its wire-compat path.
cfg = LMCacheEngineConfig.from_dict({
    "chunk_size": 256,
    "local_cpu": True,
    "remote_url": "resp://localhost:1974",
})

# A tiny synthetic chunk — real workloads use the engine's allocator.
metadata = LMCacheMetadata(
    model_name="meta-llama/Llama-3.2-1B",
    world_size=1, worker_id=0, fmt="2LTD",
    kv_shape=torch.Size([2, 1, 16, 256, 64]), kv_dtype=torch.float16,
    use_mla=False,
)

# (Use the engine builder to construct a RemoteBackend wrapping the resp://
# connector — full code path lives in lmcache.v1.cache_engine.LMCacheEngineBuilder.)
# For the smoke test it's enough to confirm the connector handshake works:
from lmcache.v1.storage_backend.connector import CreateConnector
loop = asyncio.new_event_loop()
conn = CreateConnector("resp://localhost:1974", local_cpu_backend=None,
                        config=cfg, metadata=metadata, loop=loop)
print("connected:", type(conn).__name__)
print("ping:", asyncio.run_coroutine_threadsafe(conn.ping(), loop).result())
```

Expected output:

```
connected: RESPConnector
ping: 0
```

If `ping` returns 0 the wire path is live. From here, full LMCacheEngine
integration is upstream-LMCache territory — the Pion side is verified.

## 5. Validate persistence across pion-server restart

```bash
# Run a workload that writes a few chunks via LMCache, then:
pkill pion-server
sleep 2
./pion-server --kvcache -w 1 -p 1974 &
# The KV string keyspace WAL replay reconstructs everything LMCache wrote.
# Confirm with: redis-cli -p 1974 KEYS '*' | wc -l
```

Pion's KV path WAL (separate from V-store WAL, see §29) covers SET/DEL,
so chunks LMCache stored before the kill are still queryable after
restart.

## 6. Where this leaves the shared-KV-cache design

Marks `LMCache adapter wrapper | ✅` in §13.3 by virtue of three pieces:

1. Wire compat (A6) handles LMCache traffic.
2. `pion-lmcache.LMCacheRemoteBackend` exposes the same surface to
   non-LMCache Python callers.
3. `pion-lmcache.PionStore` exposes the structured features that
   LMCache's blob interface can't surface.

The Linux/CUDA full-LMCache integration test is captured here as a
runbook rather than as a CI-runnable script — the upstream library
choice to require CUDA at install time means there's no portable way
to keep it green from a Mac dev box.
