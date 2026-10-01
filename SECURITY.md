# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report privately via GitHub's [Security Advisories](https://github.com/pavelhorak/pion/security/advisories/new),
or by email to pion@pavelhorak.com with `[pion-security]` in the subject.

Please include:

- affected version (`./pion-server --version`) and platform
- a minimal reproduction — a RESP command sequence is ideal
- what an attacker gains (crash, data disclosure, data loss, bypass)

Pion is maintained by one person. Expect an acknowledgement within a few days,
not within hours. If you need a faster response for a live incident, say so in
the subject line.

## Scope

Pion speaks an unauthenticated wire protocol by default, exactly as Redis does.
The following are **expected behaviour, not vulnerabilities**:

- Any client that can reach the port can read and write all data. Bind to
  localhost or a private network, and use `--requirepass` if you need
  authentication.
- `KEYS` and `SCAN` walk the whole keyspace and are O(N). That is the documented
  Redis semantics.
- `--tenant` provides *namespace* isolation, not resource isolation. A tenant can
  exhaust memory, CPU or WAL for every other tenant on the same process. For hard
  isolation, run one server per tenant. This is stated in `doc/multi_tenant.md`
  and is a design boundary, not a bug.
- `EVAL` runs Lua in-process. Scripts are sandboxed against the filesystem and
  network, but a script can still block the event loop.

The following **are** in scope and worth reporting:

- memory-safety faults reachable from the wire (crash, out-of-bounds read or
  write, use-after-free) — including via malformed RESP frames
- any path where one connection can read or destroy another connection's or
  another tenant's data
- a reply that does not correspond to the command that produced it (protocol
  desync), since it can cause a client to attribute one key's value to another
- silent data loss: a command that reports success without durably applying, or
  that destroys data it was not asked to touch
- authentication or allowlist bypass under `--requirepass` / `--tenant`

## Outbound connections — no telemetry

Pion never contacts us. There is no update check, no usage analytics, and no
crash reporting: crash breadcrumbs (`pion-<port>.crash.log`, `.status`) are
written to local files and go nowhere. `PION.STATS` is computed and served
locally.

The website is different, and only the website: pion.pavelhorak.com counts
page views with Cloudflare Web Analytics, which sets no cookies and collects
no personal data. Nothing in the server, the Python packages or the release
tarballs reports anywhere.

The server opens an outbound connection only in these cases, each of which
you enable or configure:

| When | Where it connects | Turn it off |
|---|---|---|
| Startup, by default | `127.0.0.1:11434` — a local Ollama probe, loopback only | `--no-auto-detect` |
| An embedding, LLM or MAX endpoint is enabled | the host you configured (defaults to `127.0.0.1`) | leave it disabled |
| `FT.HYBRID … RERANK <host> <port> …` | the host named in that command | do not send it |
| `MIGRATE` | the target named in that command | do not send it |
| Replication, cluster, Raft/gossip | the peers you configured | single node (default) |
| The Python inference sidecar is enabled | a local Unix socket; the Hugging Face library inside it downloads the model on first use | `HF_HUB_OFFLINE=1`, or `--no-auto-embed` |
| `--nle-embed` (macOS) | nothing from Pion — Apple's on-device NaturalLanguage framework; macOS manages its own assets | leave it off |

This list was taken from the source, not from memory: every outbound
connection in the server goes through `pion_connect_tcp`,
`pion_connect_unix` or `pion_tcp_connect` (`src/ffi/`). A new call site
belongs in this table in the same change.

## Supported versions

Pre-1.0, only the latest release receives fixes. There are no backports.
