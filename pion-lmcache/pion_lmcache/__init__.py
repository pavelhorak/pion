"""pion-lmcache — Pion as an LMCache remote backend.

Two layers, depending on what you need:

1. **PionStore** (structured): high-level Python wrapper around Pion's
   KV.PREFIX.REGISTER / KV.PREFIX.LOOKUP / V.STOREBATCH / V.FETCH RANGE /
   KV.PREFIX.SAVE. Built for new integrations that want quantization
   (turbo4/turbo3/turbo2/fp16), prefix-sharing across instances, and
   durable persistence (snapshot + WAL).

2. **LMCacheRemoteBackend** (blob): adapter that exposes LMCache's
   expected remote-backend interface (`put / get / contains / remove`)
   while routing through Pion's wire-compatible RESP path. Drop-in
   for `LMCacheEngineConfig(remote_url=resp://localhost:1974)` style
   configs and for LMCache's `RemoteBackendInterface` plugin slot.

Usage (structured):
    from pion_lmcache import PionStore
    s = PionStore(host="127.0.0.1", port=1974, vquant="fp16")
    s.register("my_app|model|prompt_a", kv_dim=128)
    s.store_layer("my_app|model|prompt_a", side="V", layer=0,
                  token_offset=0, tensor_fp32=v_tensor)
    s.save()                                         # snapshot + WAL truncate

Usage (LMCache plugin):
    from pion_lmcache import LMCacheRemoteBackend
    backend = LMCacheRemoteBackend(host="...", port=1974)
    # Pass `backend` where LMCache expects a remote-backend instance.

See the shared-KV-cache design (§13.3 + §29) for the architecture this
adapts to.
"""
from pion_lmcache.store import PionStore
from pion_lmcache.adapter import LMCacheRemoteBackend

__all__ = ["PionStore", "LMCacheRemoteBackend"]
