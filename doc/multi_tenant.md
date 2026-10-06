# Multi-tenancy in Pion

**Short version:** Pion offers two multi-tenant models.

1. **One `pion-server` process per tenant** — the canonical deployment wherever *hard resource isolation* matters (memory, CPU, WAL, blast radius).
2. **Per-connection tenant binding** (`--tenant NAME=PASSWORD`) — in-process namespace isolation for high-density hosting of many small tenants. Every key a tenant connection touches is transparently prefixed with its namespace; commands outside a fail-closed allowlist are rejected.

The `--ns-prefix` flag is only a cooperative namespace guard, not an isolation boundary — do not rely on it to keep tenants apart.

## What `--ns-prefix` actually does

`--ns-prefix tenant_a` makes the `KV.PREFIX.*` and `V.*` command family require the client-supplied namespace to start with the byte string `tenant_a`. It is a `startswith` check applied at a handful of call sites (`src/commands/kv_prefix.mojo`).

What it does **not** do:

- It does **not** gate plain `GET` / `SET` / `HSET` / `FT.*` / stream / pub-sub commands. Those share one per-worker keyspace regardless of `--ns-prefix`.
- It does **not** bind a namespace to a connection. Any client may pass any namespace value; a client that names its namespace `tenant_a...` passes the check.

In other words, `--ns-prefix` guards against *accidental* cross-namespace access (a typo, a misconfigured client), not against a motivated or hostile tenant. Treat it as a soft convention, not a security control.

## The isolation model: one process per tenant

Pion is shared-nothing per worker and cheap to run (`--profile kv` ≈ 50 MB/worker). The supported way to isolate tenants is to give each tenant its own process:

```bash
pion-server -p 1974 --requirepass "$TENANT_A_PW" --profile kv -w 4 --independent-workers   # tenant A
pion-server -p 1975 --requirepass "$TENANT_B_PW" --profile kv -w 4 --independent-workers   # tenant B
```

Each process has a private keyspace, WAL, HNSW graph, and (with `--requirepass`) its own authentication boundary. Route each tenant's clients to their own port. This is the model `pion-lmcache/MULTITENANT.md` describes.

## Authentication

Set `--requirepass <password>` to require `AUTH <password>` on every connection before any other command is served (both the RESP port and the binary `port+1` fast lane, which uses a `0x37` AUTH frame). Without it, the server accepts unauthenticated commands from anyone who can reach the port — bind to loopback or a trusted network in that case.

## Per-connection tenant binding (`--tenant`)

Real in-process tenant isolation. Enable with repeatable `--tenant NAME=PASSWORD` flags; `--requirepass` is **required** and becomes the admin credential:

```bash
pion-server -w 4 --independent-workers --requirepass "$ADMIN_PW" \
    --tenant "acme=$ACME_PW" --tenant "globex=$GLOBEX_PW"
```

- **Binding.** `AUTH acme <password>` (or `HELLO 2 AUTH acme <password>`) binds the connection to tenant `acme`. `AUTH <admin-password>` binds it as **admin** (unprefixed keyspace, full command set). There is no anonymous path: until a connection binds, every command is rejected with `-NOAUTH`.
- **Transparent namespacing.** Every key a tenant connection touches is invisibly prefixed with `acme:` — clients need no changes. Tenant names are restricted to `[A-Za-z0-9_-]{1,64}`, so `:` can never appear in a name and the name→prefix map is *prefix-free*: tenant A cannot forge a key in tenant B's namespace even by sending a key that literally contains `B:` (it becomes `A:B:...`).
- **Deny-by-default allowlist.** Tenant connections may run the keyed KV / hash / list / set / zset / bitmap / stream(XADD-family) / geo / HLL families, `KEYS`/`SCAN` (filtered to the tenant's namespace, prefix stripped), TTL commands, `MULTI`/`EXEC`/`WATCH`, and connection-scope commands (`PING`, `ECHO`, `INFO`, `CLIENT`, ...). Everything else — `EVAL*`/`SCRIPT`/`FUNCTION` (runtime-computed keys can't be gated), `FLUSHALL`/`FLUSHDB`, `CONFIG`/`DEBUG`/`SAVE`/`CLUSTER`, `FT.*`/vector/`AI.*`/`KV.*` (ANN neighbor sets are prefix-blind; vector isolation needs per-tenant indexes), `SUBSCRIBE`/`PUBLISH` (channels are a separate namespace), `SORT`, `XREAD`, stream consumer groups (`XGROUP`, `XREADGROUP`, `XACK`, `XPENDING`, `XCLAIM`, `XAUTOCLAIM`, `XINFO`, `XSETID`, `XDELEX`, `XACKDEL`: group reads name their keys after `STREAMS`, as `XREAD` does), numkeys-form multi-key commands (`ZUNIONSTORE`, `BITOP`, `LMPOP`, ...) — is rejected with `-NOPERM`. Fail-closed at the command level.
- **Multi-key safety.** Allowed multi-key commands (`MSET`, `DEL`, `RENAME`, `COPY`, `SMOVE`, `LMOVE`, `S*STORE`, ...) are safe by construction: *every* key argument is rewritten, so all of them land in the caller's own namespace.
- **Performance.** Tenant-bound connections are served entirely on the slow path (~345K ops/s P=1 per worker); the default/admin path is byte-for-byte untouched, so the KV gate baselines are unaffected. Namespacing cost is paid only by namespaced connections.
- **Binary lane (`port+1`).** The `0x37` AUTH frame accepts only the admin credential; tenants cannot use the binary lane (fail-closed; tenant binding for the binary lane is future work).

### Caveats

- **Per-worker keyspace scatter.** Pion workers own private keyspaces and connections are distributed by `accept()` competition, so a tenant's keys are spread across per-worker maps. `KEYS`/`SCAN` see only the *current worker's* slice — existing Pion semantics for every client, but visible under a tenant banner. Use `-w 1` (the default) if a tenant needs a complete `SCAN` view.
- **Shared resources.** Tenants in one process still share the worker's memory, CPU, WAL, and snapshot files (noisy-neighbor and blast-radius remain). Where that matters, use one process per tenant (below).
- **`WATCH` versions** are bumped by fast-path mutations only; slow-path writes (which includes all tenant writes) do not bump them, so `WATCH`-based optimistic locking is not currently conflict-detecting for tenant connections.

Regression test: `tests/test_tenant_isolation.py`.
