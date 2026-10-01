#!/usr/bin/env bash
# Package a prebuilt Pion release tarball for the current platform (gh #108).
#
# Produces dist/pion-<version>-<platform>.tar.gz containing:
#   bin/pion-server            the release binary (already built — this script
#                              does NOT build unless --build is passed, so the
#                              gated binary is what ships)
#   lib/                       the Mojo runtime libraries the binary needs
#   pion-server.sh             wrapper that sets the library path and execs
#   LICENSE, README.txt       Apache-2.0 for Pion
#   LICENSE-pion-vector        only when the binary links libpion_vector
#
# The binary's rpath points into the build machine's .pixi env, so a
# downloaded tarball MUST launch through pion-server.sh (or set
# LD_LIBRARY_PATH / DYLD_LIBRARY_PATH to the tarball's lib/ manually).
#
# Usage:
#   scripts/package_release.sh                # package existing ./pion-server
#   scripts/package_release.sh --build        # pixi run build first
#   scripts/package_release.sh --allow-dirty  # package anyway (local testing)

set -euo pipefail
cd "$(dirname "$0")/.."

DO_BUILD=0
ALLOW_DIRTY=0
for arg in "$@"; do
    case "$arg" in
        --build)       DO_BUILD=1 ;;
        --allow-dirty) ALLOW_DIRTY=1 ;;
        *) echo "usage: package_release.sh [--build] [--allow-dirty]"; exit 1 ;;
    esac
done

if [ "$DO_BUILD" = "1" ]; then pixi run build; fi

[ -x ./pion-server ] || { echo "FATAL: ./pion-server not found — run 'pixi run build' (or pass --build)"; exit 1; }

# ── Provenance guard (gh #108) ──────────────────────────────────────────────
# This script ships ./pion-server exactly as it finds it, and the binary
# carries whatever SHA was HEAD when it was BUILT. doc/operations.md states the
# rule as prose — "build at HEAD before packaging" — which does not run.
#
# The dirty-tree half is the half that bites, and a SHA check alone will not
# catch it: a binary built from a dirty tree stamps the CURRENT HEAD and still
# contains code that HEAD does not have. Measured on 0.975 — binary stamped
# `0.975+5897e50` with 5897e50 checked out, while five modified source files
# sat in the tree. A bug report against that tarball would name a commit whose
# code it does not contain, which is the whole failure mode.
#
# VERSION and src/common/version.mojo are excluded because `pixi run build`
# stamps them itself, so the documented flow (build at HEAD, package, then
# commit both files) leaves exactly those two modified and nothing else.
#
# src/ffi/metal_compute.metallib is excluded for the same reason. It is a build
# product that is also tracked, so a Command Line Tools checkout still has one,
# and `pixi run build` recompiles it wherever full Xcode is installed. Every
# release runner has full Xcode, so there it always differs from the tracked
# bytes, and the release would refuse to package its own build. Its source,
# metal_compute.metal, stays in the check.
#
# An EXPORT tree is not a git repo — `git archive` produces a plain directory
# — so `git rev-parse` fails there and this guard compared the binary's real
# SHA against the literal string "unknown" and refused every time. The export
# records its source commit in `.export_sha` for exactly this reason (it is
# what stamp_version.sh reads to avoid stamping `+unknown`), so use it as the
# fallback rather than inventing a second provenance mechanism.
#
# Order matters: prefer git when there is a git tree, because in CI the
# checkout IS the repo and `.export_sha` will not exist.
HEAD_SHA="$(git rev-parse --short HEAD 2>/dev/null \
            || { [ -f .export_sha ] && tr -d "[:space:]" < .export_sha; } \
            || echo unknown)"
BIN_SHA="$(./pion-server --version 2>/dev/null | sed -n 's/.*+\([0-9a-f][0-9a-f]*\).*/\1/p')"
# Likewise skipped in an export tree: `git status` reports nothing there, so
# the check is vacuous rather than wrong, but saying so beats a silent pass.
if git rev-parse --git-dir >/dev/null 2>&1; then
    DIRTY="$(git status --porcelain 2>/dev/null \
             | grep -v '^??' \
             | grep -vE ' (VERSION|src/common/version\.mojo|src/ffi/metal_compute\.metallib)$' || true)"
else
    DIRTY=""
    echo "  note: not a git tree (export?); the dirty-tree check is skipped."
fi

# "FATAL" when it stops, "WARNING" when --allow-dirty carries on: a message
# that says FATAL and then keeps going teaches the reader to skim past it.
if [ "$ALLOW_DIRTY" = "1" ]; then LVL="WARNING"; else LVL="FATAL"; fi

if [ -n "$DIRTY" ]; then
    echo "$LVL: working tree has changes the binary's commit does not contain:"
    printf '%s\n' "$DIRTY" | sed 's/^/    /'
    if [ "$ALLOW_DIRTY" != "1" ]; then
        echo "  Commit them and rebuild, or pass --allow-dirty for a local-only tarball."
        exit 1
    fi
    echo "  --allow-dirty: continuing. DO NOT PUBLISH this tarball."
fi

if [ "$BIN_SHA" != "$HEAD_SHA" ]; then
    echo "$LVL: ./pion-server stamps '$BIN_SHA' but HEAD is '$HEAD_SHA'."
    echo "  (A binary built before its own commit names the PREVIOUS commit.)"
    if [ "$ALLOW_DIRTY" != "1" ]; then
        echo "  Rebuild at HEAD before packaging."
        exit 1
    fi
    echo "  --allow-dirty: continuing. DO NOT PUBLISH this tarball."
fi

VERSION="$(cat VERSION)"   # semver since 0.9.0 (D2); opaque string here
OS="$(uname -s)"
ARCH="$(uname -m)"
# gh #415 (I-7): on ubuntu-24.04-arm `uname -m` is `aarch64`, so the tarball was
# named pion-<v>-linux-aarch64.tar.gz while release.yml's alias loop and the
# docs promise `linux-arm64`. Normalise so the produced name matches the alias
# and the documented /releases/latest/download/pion-linux-arm64.tar.gz.
case "$ARCH" in aarch64) ARCH="arm64" ;; esac
case "$OS" in
    Darwin) PLATFORM="macos-${ARCH}"; LIBEXT="dylib"; LIBPATH_VAR="DYLD_LIBRARY_PATH" ;;
    Linux)  PLATFORM="linux-${ARCH}"; LIBEXT="so";    LIBPATH_VAR="LD_LIBRARY_PATH" ;;
    *) echo "FATAL: unsupported OS $OS"; exit 1 ;;
esac

NAME="pion-${VERSION}-${PLATFORM}"
STAGE="dist/${NAME}"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/lib"

cp pion-server "$STAGE/bin/"
cp LICENSE "$STAGE/"
# gh #412 (I-5): ship NOTICE. liblua.a is statically linked and Lua/lua-cjson
# are MIT, which requires the copyright+permission notice "in all copies"; the
# Modular runtime dylibs in lib/ carry their own redistribution notice. NOTICE
# now reproduces all three verbatim, so it must travel with the tarball.
[ -f NOTICE ] && cp NOTICE "$STAGE/"

# A `pixi run build` binary links the closed libpion_vector, whose own licence
# must travel with it (its section 3(c)). ASK THE BINARY which backend it
# carries rather than inferring it from the platform: a build-open binary, or
# any platform not yet vendored (gh #348), contains no closed code and ships
# Apache-2.0 alone. An unreadable --version is a refusal, not a guess — shipping
# the library without its licence is the one mistake here that is not cosmetic.
VEC_LINE="$(./pion-server --version 2>/dev/null | grep '^vector:' || true)"
case "$VEC_LINE" in
    *libpion_vector*)
        [ -f vendor/pion-vector/LICENSE ] || { echo "FATAL: binary links libpion_vector but vendor/pion-vector/LICENSE is missing"; exit 1; }
        cp vendor/pion-vector/LICENSE "$STAGE/LICENSE-pion-vector"
        LICENSE_LINE="License: Apache-2.0 (LICENSE). This binary links libpion_vector, the
closed tuned vector kernels, under the Pion Vector Binary Licence
(LICENSE-pion-vector): free for any use, not for offering Pion as a managed
service." ;;
    *reference*|*open*)
        LICENSE_LINE="License: Apache-2.0 (LICENSE). Open build: no closed code is linked." ;;
    *)
        echo "FATAL: cannot tell which vector backend ./pion-server links ('$VEC_LINE')"; exit 1 ;;
esac

# Runtime libraries: ASK THE BINARY, don't hardcode a list.
#
# The old hardcoded set still named libAsyncRTMojoBindings, which the Mojo 1.0
# migration dropped — so every package run printed "tarball may not run
# standalone — missing runtime libs" for a tarball that runs fine. A release
# script that cries wolf trains you to ignore it, which is worse than no check,
# and the list would drift again at the next toolchain bump.
#
# The derivation lives in collect_runtime_libs.sh because the Dockerfile needs
# the identical answer, and when it was written out twice the two copies
# drifted: this one was fixed for Mojo 1.0 and the Dockerfile's was not.
LIBDIR=".pixi/envs/default/lib"
MISSING=0
scripts/collect_runtime_libs.sh pion-server "$LIBDIR" "$STAGE/lib" || MISSING=1
# gh #415 (I-16): a missing runtime lib is FATAL — a tarball whose wrapper cannot
# find its dylibs is dead on arrival, and release.yml does not grep for a
# warning, so the old warn-and-continue silently shipped a broken release. On
# THIS box the baked LC_RPATH still resolves (doc/operations.md), so a raw run
# hides it; the wrapper is the only thing a stranger has. --allow-dirty (local
# testing, e.g. no pixi env in a scratch copy) downgrades it to a warning.
if [ "$MISSING" != "0" ]; then
    if [ "$ALLOW_DIRTY" = "1" ]; then
        echo "WARNING: tarball may not run standalone — missing runtime libs (--allow-dirty)"
    else
        echo "FATAL: missing runtime libs — the tarball would not run standalone."
        echo "       collect_runtime_libs.sh could not resolve every dependency from"
        echo "       ${LIBDIR}. Fix the lib set (or pass --allow-dirty for local testing)."
        exit 1
    fi
fi

# ── Metal shader library (gh #281) ──────────────────────────────────────────
# --metal-attention is the README's FIRST command and it could not work from a
# tarball: metal_wrap.m loads metal_compute.metallib by path, and this script
# shipped bin/ + lib/ + docs and nothing else. The server started, printed
# "Metal Attn: enabled", then fell back to the MLX bridge twenty lines later.
#
# Two ways that happened, and the second is the one that shipped v0.985:
#   1. the file was never copied into the tarball (this block);
#   2. `pixi run build` printed "[Metal] skipped" — one line inside a 20-minute
#      build — because the box had Command Line Tools instead of full Xcode, so
#      no metallib was produced at all, and packaging never asked.
#
# So this is a hard failure on macOS, not a warning. The binary links
# -framework Metal unconditionally there, `--metal-attention` is documented,
# and shipping an artifact that cannot honour a documented flag is exactly the
# class of quiet degradation gh #257 and gh #281 are about.
if [ "$OS" = "Darwin" ]; then
    METALLIB="src/ffi/metal_compute.metallib"
    if [ ! -f "$METALLIB" ]; then
        echo "FATAL: $METALLIB is missing — this binary cannot do --metal-attention."
        echo "  'pixi run build' skips the shader step without FULL Xcode (Command"
        echo "  Line Tools is not enough) and says so in one line. Fix with:"
        echo "      xcode-select -s /Applications/Xcode.app/Contents/Developer"
        echo "      pixi run build"
        echo "  Current: $(xcode-select -p 2>/dev/null || echo '(xcode-select failed)')"
        exit 1
    fi
    # A metallib older than its source is the same failure wearing a disguise:
    # the shader step was skipped and a previous build's artifact is still lying
    # around. That is precisely how v0.985 looked on disk.
    if [ "src/ffi/metal_compute.metal" -nt "$METALLIB" ]; then
        echo "FATAL: $METALLIB is OLDER than src/ffi/metal_compute.metal."
        echo "  The shader step was skipped and a stale artifact remains. Rebuild"
        echo "  with full Xcode selected (see above) — do not ship this."
        exit 1
    fi
    cp "$METALLIB" "$STAGE/lib/"
    echo "  bundled: metal_compute.metallib ($(wc -c < "$METALLIB" | tr -d ' ') bytes)"
fi

cat > "$STAGE/pion-server.sh" <<WRAPPER
#!/usr/bin/env bash
# Launcher: points the dynamic loader at the bundled Mojo runtime libs.
DIR="\$(cd "\$(dirname "\$0")" && pwd)"
export ${LIBPATH_VAR}="\$DIR/lib\${${LIBPATH_VAR}:+:\$${LIBPATH_VAR}}"
exec "\$DIR/bin/pion-server" "\$@"
WRAPPER
chmod +x "$STAGE/pion-server.sh"

cat > "$STAGE/README.txt" <<README
Pion ${VERSION} (${PLATFORM})

Run:
    ./pion-server.sh                 # port 1974
    ./pion-server.sh --profile kv    # KV-only, ~50MB/worker
    ./pion-server.sh --kvcache -w 1  # + shared KV cache (KV.PREFIX.*)

Any Redis client connects: redis-cli -p 1974

The wrapper sets ${LIBPATH_VAR} to the bundled lib/ directory; the raw
binary in bin/ will not start without it. WAL, snapshots, and blob arenas
are written to the current working directory.

lib/metal_compute.metallib is the Metal shader library for
--metal-attention (macOS). It is found relative to the binary, so you can
run this from any working directory; PION_METAL_LIB overrides the path.
The server prints "Metal Attn: ACTIVE" or "Metal Attn: NOT ACTIVE" at
startup — the earlier banner line reports only what you asked for.

Docs: https://github.com/pavelhorak/pion (doc/index.md)
${LICENSE_LINE}
README

mkdir -p dist
tar -czf "dist/${NAME}.tar.gz" -C dist "$NAME"
rm -rf "$STAGE"

echo "packaged: dist/${NAME}.tar.gz"
tar -tzf "dist/${NAME}.tar.gz" | sed 's/^/  /'
