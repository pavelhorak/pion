#!/usr/bin/env bash
# Build the pion-embedded wheel for this platform: one cp311-abi3 wheel (the
# _pion extension) for every CPython >= 3.11.
#   macOS: MACOSX_DEPLOYMENT_TARGET=14.0 pixi run build-embed && bash embedded/build_wheel.sh <python>
#   Linux: bash embedded/build_linux.sh && bash embedded/build_wheel.sh <python>   (needs patchelf)
# The wheel carries libpion and the Mojo runtime closure it needs
# (scripts/collect_runtime_libs.sh), found through @loader_path: never the
# build machine's pixi env. Verify it with the toolchain hidden.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${1:-python3}"
if [ "$(uname -s)" = "Darwin" ]; then
    LIB="$ROOT/libpion.dylib"
    [ -f "$LIB" ] || { echo "build it first: MACOSX_DEPLOYMENT_TARGET=14.0 pixi run build-embed" >&2; exit 1; }
    MINOS=$(otool -l "$LIB" | awk '/minos/{print $2; exit}')
    [ "$MINOS" = "14.0" ] || { echo "libpion.dylib minos is $MINOS, not 14.0: rebuild with MACOSX_DEPLOYMENT_TARGET=14.0" >&2; exit 1; }
else
    LIB="$ROOT/libpion.so"
    [ -f "$LIB" ] || { echo "build it first: bash embedded/build_linux.sh" >&2; exit 1; }
    command -v patchelf >/dev/null || { echo "patchelf is required (pip install patchelf)" >&2; exit 1; }
fi
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$ROOT/embedded/python/." "$STAGE/"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$STAGE/pion_embedded/_lib"
cp "$LIB" "$STAGE/pion_embedded/_lib/"
bash "$ROOT/scripts/collect_runtime_libs.sh" "$LIB" "$ROOT/.pixi/envs/default/lib" "$STAGE/pion_embedded/_lib"
if [ "$(uname -s)" = "Darwin" ]; then
    for f in "$STAGE"/pion_embedded/_lib/*.dylib; do
        chmod u+w "$f"
        for rp in $(otool -l "$f" | awk '/LC_RPATH/{getline; getline; print $2}'); do
            install_name_tool -delete_rpath "$rp" "$f"
        done
        install_name_tool -add_rpath @loader_path "$f" 2>/dev/null || true
        codesign -f -s - "$f" 2>/dev/null
    done
    PAT='*-abi3-macosx_14_0_arm64.whl'
else
    for f in "$STAGE"/pion_embedded/_lib/*.so; do
        chmod u+w "$f"
        patchelf --set-rpath '$ORIGIN' "$f"
    done
    PAT='*-abi3-linux_*.whl'
fi
export MACOSX_DEPLOYMENT_TARGET=14.0    # the extension's minos, as libpion's
OUT="$ROOT/embedded/dist"
mkdir -p "$OUT"
(cd "$STAGE" && "$PY" -m pip wheel --no-deps --no-build-isolation -w "$STAGE/wheel" . >/dev/null)
cp "$STAGE"/wheel/$PAT "$OUT/"
ls -la "$OUT"
