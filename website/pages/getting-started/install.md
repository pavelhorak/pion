# Install

Pion is one binary. Pick the path that matches your machine; every path ends
at a server answering `PING` on port 1974. The text on this page is the
README's own install section, included at build time.

=== "macOS · Homebrew"

<!-- include-section: README.md | ### Homebrew (macOS) | indent:4 -->

=== "macOS · Apple Silicon (prebuilt)"

<!-- include-section: README.md | ### Prebuilt binary (no build) | indent:4 -->

=== "Python client"

    The client for the prompt-cache path. It needs a running
    `pion-server --kvcache --metal-attention -w 1` from the macOS tab.

<!-- include-region: README.md | pip-install | indent:4 -->

=== "Docker"

<!-- include-section: README.md | ### Docker | indent:4 -->

=== "From source"

<!-- include-region: README.md | build-from-source | indent:4 -->

## Before you expose it to a network

<!-- include-region: README.md | security -->

Next: [First five minutes](first-five-minutes.md) — the Redis wire, vector
search, the semantic cache and the four Pion lines.
