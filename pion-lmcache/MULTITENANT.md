# Multi-tenant deployment for Pion KV cache

Pion's KV.PREFIX namespace is content-addressable — anyone with a
connection can read or write any namespace they know the name of. For
deployments where tenants must be isolated from each other, use the
following pattern:

## Pattern: one Pion per tenant + namespace-prefix enforcement

Run a separate `pion-server` process per tenant, each gated by
`--ns-prefix <prefix>`. The prefix becomes the lockbox: every
`KV.PREFIX.REGISTER`, `LOOKUP`, `OWNER`, and `SAVE` checks that the
supplied namespace starts with this exact byte string. Mismatch → `-ERR`.

Server side:

```bash
# tenant_a's Pion (port 6000)
./pion-server --kvcache -w 4 --independent-workers -p 6000 --ns-prefix "tenant_a:" &

# tenant_b's Pion (port 6001)
./pion-server --kvcache -w 4 --independent-workers -p 6001 --ns-prefix "tenant_b:" &
```

Client side:

```python
from pion_lmcache import PionStore

# tenant_a's app — connects to its dedicated Pion, prepends prefix automatically.
s = PionStore(host="...", port=6000, tenant_prefix="tenant_a:")
s.register("my_app|prompt_a", kv_dim=128)   # → "tenant_a:my_app|prompt_a" on the wire

# tenant_b's app — different Pion, different prefix.
s = PionStore(host="...", port=6001, tenant_prefix="tenant_b:")
s.register("my_app|prompt_a", kv_dim=128)   # → "tenant_b:my_app|prompt_a", no collision.
```

## What's enforced vs what's not

**Enforced** (server-side, by `--ns-prefix`):

- `KV.PREFIX.REGISTER` rejects mismatched namespaces.
- `KV.PREFIX.LOOKUP` rejects mismatched namespaces.
- `KV.PREFIX.OWNER` rejects mismatched namespaces.
- `KV.PREFIX.SAVE` is unaffected (server-wide snapshot).

**NOT enforced** (out of scope for this layer):

- Connection-level authentication is available separately: `--requirepass`
  gates every connection behind AUTH, and `--tenant NAME=PASSWORD`
  adds per-connection tenant binding with transparent key
  namespacing for the plain KV command families — see `doc/multi_tenant.md`.
  Note the tenant allowlist **rejects** `KV.PREFIX.*`/`V.*` for tenant
  connections (KV-cache tenancy still uses the process-per-tenant pattern
  in this document); without those flags, deploy network policies
  (firewall, VPC, mTLS-terminating proxy) to gate connection-level access.
- `V.STOREBATCH` / `V.FETCH` direct invocations — these still take the
  full session id (`<ns>_pk` / `<ns>_pv`) so a client that goes around
  `KV.PREFIX.*` can hit any session by name. PionStore's high-level API
  goes through the gated commands, so simply use it.
- Cross-tenant V-store LRU eviction — V-store sessions across tenants
  share the same MAX_VS_SESSIONS pool. Heavy-traffic tenants can
  evict light-traffic tenants. If you need fairness here, deploy
  separate Pion processes per tenant (recommended pattern above) so
  each gets its own MAX_VS_SESSIONS=32 slots.

## Validation

`tests/test_multitenant.py` exercises the full enforcement matrix:
correct prefix succeeds; wrong / missing prefix is rejected on
REGISTER, LOOKUP, OWNER; legitimate tenant retains FETCH access.

## When NOT to use this

- **Single-tenant deployments**: leave `--ns-prefix` unset (default
  empty string). All commands accept any namespace; no overhead.
- **Trusted multi-tenant on shared infrastructure** (e.g. multiple
  internal services on one Pion): set `--ns-prefix` per logical
  service to keep namespaces from colliding accidentally, but treat
  this as hygiene rather than security.
- **Hard isolation across mutually distrusting tenants**: the
  one-Pion-per-tenant pattern above is the correct deployment
  topology. Don't try to multiplex them on a single Pion.
