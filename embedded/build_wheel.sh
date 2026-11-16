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
# The wheel carries the engine, so it carries the engine's version: the wheel
# attached to release v1.1.3 is pion-embedded 1.1.3.
VER="$(tr -d '[:space:]' < "$ROOT/VERSION")"
sed -i.bak "s/^version = \"0.0.0\".*/version = \"$VER\"/" "$STAGE/pyproject.toml" && rm "$STAGE/pyproject.toml.bak"
grep -q "^version = \"$VER\"" "$STAGE/pyproject.toml" || { echo "could not stamp version $VER" >&2; exit 1; }
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
    PAT='*-abi3-manylinux_2_*.whl'
fi
export MACOSX_DEPLOYMENT_TARGET=14.0    # the extension's minos, as libpion's
if [ "$(uname -s)" = "Darwin" ]; then
    # A python.org framework Python (what setup-python installs on the macOS
    # runners) is universal2: it compiles the extension for both arches and
    # tags the wheel universal2. Everything in _lib/ is arm64 only, so build
    # and tag arm64.
    export ARCHFLAGS="-arch arm64" _PYTHON_HOST_PLATFORM="macosx-14.0-arm64"
fi
OUT="$ROOT/embedded/dist"
mkdir -p "$OUT"
(cd "$STAGE" && "$PY" -m pip wheel --no-deps --no-build-isolation -w "$STAGE/wheel" . >/dev/null)
if [ "$(uname -s)" != "Darwin" ]; then
    # setuptools tags a Linux wheel linux_<arch>, which PyPI refuses. Tag it
    # manylinux_2_<N>, N being the highest GLIBC_2.* any shipped file needs
    # (the extension included), so the tag states the floor rather than
    # claiming one. Built on Ubuntu 22.04 that is 2.35, set by Modular's
    # libAsyncRTRuntimeGlobals.so; on 24.04 libpion itself needs 2.38.
    W=$(ls "$STAGE"/wheel/*-abi3-linux_*.whl)
    SCAN="$(mktemp -d)"
    (cd "$SCAN" && "$PY" -m zipfile -e "$W" .)
    N=0
    for f in $(find "$SCAN" -name '*.so*' -type f); do
        n=$( (strings -a "$f" | grep -o 'GLIBC_2\.[0-9]*' || true) | sed 's/GLIBC_2\.//' | sort -n | tail -1)
        echo "$(basename "$f"): GLIBC_2.${n:-none}"
        if [ -n "$n" ] && [ "$n" -gt "$N" ]; then N=$n; fi
    done
    rm -rf "$SCAN"
    [ "$N" -gt 0 ] || { echo "no GLIBC_2.* version found in the wheel" >&2; exit 1; }
    "$PY" -m wheel tags --remove --platform-tag "manylinux_2_${N}_$(uname -m)" "$W" >/dev/null
fi
compgen -G "$STAGE/wheel/$PAT" >/dev/null || { echo "no $PAT; built: $(ls "$STAGE/wheel")" >&2; exit 1; }
cp "$STAGE"/wheel/$PAT "$OUT/"
ls -la "$OUT"
