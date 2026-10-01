# Security model

Loopback by default, no TLS, a password that flips the default bind, and an
unauthenticated replication stream. Read this before the server sees a network.

<!-- include-region: README.md | security -->

## Reporting a vulnerability

<!-- include-file: SECURITY.md | strip-h1 | shift:1 -->

## Tenants and isolation

<!-- include-section: doc/multi_tenant.md | ## The isolation model: one process per tenant -->

The per-connection tenant binding (`--tenant NAME=PASSWORD`) is documented in
[Multi-tenant](/multi_tenant.md); the flags themselves are in the
[CLI reference](/reference/cli-flags.md).
