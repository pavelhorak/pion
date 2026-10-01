# `SSM.PREFIX.*` — recurrent-state companion

Hybrid Mamba / GatedDeltaNet / RWKV models carry state that is not K/V.
`SSM.PREFIX.*` stores that state under the same namespace contract as
`KV.PREFIX.*`, so a hybrid model's whole prefix — softmax layers via the
V-store, linear layers here — is shared across processes and survives a
restart.

<!-- include-section: doc/shared_kv_cache.md | ### SSM.PREFIX.* — recurrent-state companion -->

