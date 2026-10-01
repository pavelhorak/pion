#!/bin/bash
# Stamp the version into src/common/version.mojo.
# Called by `pixi run build` via the stamp-version task.
#
# TWO MODES, chosen by the shape of VERSION itself.
#
#   semver   VERSION contains "0.9.0" (three components)  -> stamped verbatim
#   counter  VERSION contains "0.991" (two components)    -> incremented
#
# Pre-1.0 this project used an auto-incrementing build counter, which was the
# right tool while every build was internal: the number answered "is this the
# binary I just built?" and nothing else. It stops being right the moment
# strangers can read the version, because it communicates nothing about
# compatibility and changes on every local compile.
#
# From 0.9.0 (D2, the public preview) VERSION holds a semantic version and the
# build stamps it verbatim. Tags drive releases. A local build no longer
# rewrites VERSION, which also means `pixi run build` stops dirtying the
# working tree — it used to leave two modified files after every compile,
# including in CI.
#
# If you want the old per-build counter back for internal bookkeeping, it
# belongs in a separate file that is not the published version.

set -e
VERSION_FILE="VERSION"
VERSION_MOJO="src/common/version.mojo"

CURRENT=$(tr -d '[:space:]' < "$VERSION_FILE")

# Three components => semver, stamp as-is. Two => legacy counter, increment.
if [ "$(echo "$CURRENT" | tr -cd '.' | wc -c | tr -d ' ')" -ge 2 ]; then
    MODE="semver"
    NEW_VERSION="$CURRENT"
else
    MODE="counter"
    BUILD_NUM=$(echo "$CURRENT" | cut -d. -f2)
    NEW_VERSION="0.$((BUILD_NUM + 1))"
    echo "$NEW_VERSION" > "$VERSION_FILE"
fi

# Which commit is this binary? Two sources, in precedence order.
#
# 1. `git rev-parse` — an ordinary build in a git tree. The common case.
#
# 2. `.export_sha` — a build INSIDE an export tree, which `git archive`
#    produces as a plain directory where `git rev-parse` fails. Without it the
#    binary stamps `+unknown`, and a bug report carrying `+unknown` cannot be
#    tied to a commit — exactly the provenance you need from someone you have
#    never met and cannot ask.
#
# PION_PUBLIC_SHA used to be a third source here, at the top. It existed
# because official binaries were built from the PRIVATE tree, so `git
# rev-parse` named a commit a stranger could not fetch, and the variable let a
# release claim the public SHA instead. D3 (2026-09-17) moved release builds
# into public CI, where the binary is built from the commit it claims. The
# override is gone rather than left dormant: a second stamping path that
# nobody exercises is a thing that silently rots, and its honesty depended on
# private and public source staying identical.
SHA=$(git rev-parse --short HEAD 2>/dev/null \
      || { [ -f .export_sha ] && cat .export_sha; } \
      || echo "unknown")
DATE=$(date +%Y-%m-%d)

cat > "$VERSION_MOJO" << EOF
"""Pion version — auto-stamped by build script.

Format: MAJOR.MINOR.PATCH+SHA. Set VERSION by hand; releases are driven by
git tags (see .github/workflows/release.yml).
SHA is the short git commit hash at build time.
"""

comptime PION_VERSION = "$NEW_VERSION"
comptime PION_BUILD_SHA = "$SHA"
comptime PION_BUILD_DATE = "$DATE"
EOF

echo "[version] $NEW_VERSION+$SHA ($DATE) [$MODE]"
