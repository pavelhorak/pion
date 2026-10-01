# pion-lmcache

Pion as an LMCache remote backend — three integration paths depending on
what you need.

## Why three paths?

LMCache's standard remote tier is opaque blob storage. Pion can serve in
that role today (wire-compatible RESP via A6), and that's the simplest
path to ship. But Pion also has a *structured* KV cache (per-block
quantization, prefix sharing, snapshot+WAL durability, cross-worker
LOOKUP) that LMCache's blob interface can't surface. So this package
exposes both.

| Path | Use when | Pion features available |
|---|---|---|
| **(1) Wire-compat** (`remote_url: resp://`) | You're already using LMCache and just want to swap Redis for Pion | Persistent blob storage, GET/SET batching |
| **(2) `LMCacheRemoteBackend`** | You want a Python blob client to Pion (NOT through LMCache) | Same as (1), plus optional `ns_prefix` for multi-tenant isolation |
| **(3) `PionStore`** | You want quantized + prefix-shared KV cache and don't need LMCache | All of Pion's structured features: turbo4 / fp16 / int8, KV.PREFIX.\*, V.STOREBATCH/V.FETCH, KV.PREFIX.SAVE, cross-worker auto-redirect |

## Path 1 — Wire-compat (recommended for existing LMCache users)

Already shipped via A6 in the Pion server. No Python adapter needed.

```yaml
# lmcache_config.yaml
chunk_size: 256
local_cpu: true
remote_url: resp://localhost:1974
```

Pion's RESP path handles LMCache's `RESPConnector` traffic (GET/SET/EXISTS/DEL
on SHA256-keyed blobs) up to 16 MB per blob. See `tests/test_lmcache_compat.py`
for the wire-level acceptance test (8 sections covering every command LMCache
uses, including 8 MB chunks for 70B models).

**Pros:** zero code change for LMCache users. WAL-durable persistence comes
for free (Pion's regular KV-string WAL covers SET/DEL).
**Cons:** opaque blobs — none of Pion's structured cache features are exposed.

## Path 2 — `LMCacheRemoteBackend` (Python blob client)

Direct Python access to Pion's wire-compat path. Useful when you want
explicit control (timeouts, retry, namespace prefix) or when you're
embedding Pion in something that isn't LMCache.

```python
from pion_lmcache import LMCacheRemoteBackend

backend = LMCacheRemoteBackend(host="127.0.0.1", port=1974,
                                ns_prefix="tenant_a")
backend.put("model@1@0@deadbeef@fp16", blob_bytes)
got = backend.get("model@1@0@deadbeef@fp16")
backend.contains("...")  # True / False
backend.remove("...")
backend.ping()           # PONG
```

Methods (`put / get / contains / remove / mput / mget / ping`) match the
**stable subset** of LMCache's `RemoteBackendInterface`. Where LMCache's
exact upstream signature has churned across versions, this adapter sticks
to that subset and avoids importing `lmcache` directly — so the package
stays installable on hosts without CUDA.

## Path 3 — `PionStore` (structured KV cache)

For new integrations that want Pion's full feature set: per-block
quantization, prefix sharing across instances, `KV.PREFIX.SAVE`-durable
snapshots, WAL-durable writes, and transparent cross-worker auto-redirect.

```python
from pion_lmcache import PionStore
import numpy as np

with PionStore(host="...", port=1974, vquant="fp16") as s:
    s.register("my_app|model_v1|prompt_a", kv_dim=128)
    K = np.random.randn(28, 100, 128).astype(np.float32)  # (n_layers, n_tokens, kv_dim)
    V = np.random.randn(28, 100, 128).astype(np.float32)
    s.store_prefix("my_app|model_v1|prompt_a", K, V)

    # Cross-instance / cross-process: another client looks it up.
    if s.lookup("my_app|model_v1|prompt_a"):
        K2, V2 = s.fetch_prefix("my_app|model_v1|prompt_a",
                                 n_layers=28, n_tokens=100, kv_dim=128)

    s.save()  # KV.PREFIX.SAVE: snapshot + WAL truncate

    print(s.info())  # {wal_appended, wal_replayed, vstore_evictions, ...}
```

`PionStore` is what `pion-vllm-mlx.PionPromptCache` uses internally to
hand out drop-in mlx-lm caches.

### Cross-worker auto-redirect

Pion runs `--kvcache` with `-w >1`: V buffers stay per-worker but the
session directory is shared. PionStore catches `-ERR session lives on
worker N` transparently — opens new connections until the kernel's
accept race lands on the owner, then pins. Disable with
`auto_redirect=False`.

```python
with PionStore(port=1974) as s:
    s.register("ns_a", 128)
    s.store_layer("ns_a", "V", 0, 0, tensor)  # 0-2 redirects on average
    print(s.redirect_count)                   # diagnostic
    print(s.owner("ns_a"))                    # KV.PREFIX.OWNER → worker_id
```

### Persistence model

- `V.STOREBATCH` writes are WAL-durable (`pion.vstore.wal.<worker_id>`,
  see Pion §29). Replayed on startup; SIGKILL is recovered with
  bit-equal `V.FETCH` round-trip.
- `KV.PREFIX.SAVE` writes a fresh snapshot (`pion.vstore.<worker_id>`)
  and truncates the WAL — the canonical compaction.
- The cross-worker directory is also re-published from both snapshot
  load and WAL replay, so warm restart preserves cross-worker LOOKUP.

## Validating against real LMCache (Linux/CUDA)

LMCache's wheel currently requires CUDA at install time, so you can't
exercise it on Mac. See [`INSTALL_LMCACHE.md`](INSTALL_LMCACHE.md) for the
Linux runbook (`pip install lmcache` + a 30-line script that runs the
same wire pattern as `LMCacheRedisClient`).

The local validation that doesn't need CUDA:

- `pion-lmcache/tests/test_pion_lmcache.py` — 5 tests (PASS) covering
  PionStore round-trip, blob put/get, ns_prefix isolation, KV.PREFIX.SAVE,
  WAL persistence across SIGKILL.
- `tests/test_lmcache_compat.py` — wire acceptance tests for the exact
  RESP commands LMCache's RESPConnector emits (run with `--large` for
  the 8 MB blob path).
- `tests/test_lmcache_resp_connector.py` — emulates LMCache's RESPConnector
  call sequence (chunked GET/SET/EXISTS/DEL with realistic key shapes).

## Install

```bash
pip install -e pion-lmcache/
# Optional: lmcache (Linux + CUDA only)
pip install -e pion-lmcache/[lmcache]
```

The package itself depends only on `numpy`. The `lmcache` extra is for
when you actually want to plug into LMCache; on Mac without CUDA you
should skip it and use Path 1 (wire-compat) or Path 3 (PionStore) instead.

## License

Apache-2.0. Pion's satellites are deliberately permissive so they can be vendored
into any stack; the Pion **server** itself is Apache-2.0 too,
with one closed binary library for its tuned vector kernels — see the top-level
`LICENSE` and `doc/licensing.md`.
