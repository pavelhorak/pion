#!/usr/bin/env bash
# Collect the Mojo runtime libraries a built pion-server needs (gh #108).
#
#   collect_runtime_libs.sh <binary> <libdir> <outdir>
#
# ASK THE BINARY, DON'T HARDCODE A LIST. This exists because the list was
# hardcoded in two places and they drifted apart: Mojo 1.0 dropped
# libAsyncRTMojoBindings, package_release.sh was fixed to derive the set, and
# the Dockerfile was not — so `docker build --target runtime-prebuilt` kept
# COPYing a file that no longer exists (hard build failure), while the
# from-source target's `|| true` loop silently shipped 3 of the 5 it named.
# One derivation, two callers, no third instance.
#
# Exit status: 0 if every library the binary needs was copied; 1 if any is
# missing (or none could be determined). Callers decide how loud that is —
# the Dockerfile fails the build, package_release.sh warns.

set -uo pipefail

BINARY="${1:?usage: collect_runtime_libs.sh <binary> <libdir> <outdir>}"
LIBDIR="${2:?usage: collect_runtime_libs.sh <binary> <libdir> <outdir>}"
OUTDIR="${3:?usage: collect_runtime_libs.sh <binary> <libdir> <outdir>}"

# Only libraries we actually ship out of the pixi env. System libs (libc,
# libm, libpthread) come from the base image / host and must not be bundled —
# copying those is how you get a binary that segfaults against the wrong libc.
MOJO_LIB_RE='^lib(KGEN|AsyncRT|MSupport|NVPTX)'

deps_of() {   # deps_of <file> -> bare library filenames it needs from LIBDIR
    if [ "$(uname -s)" = "Darwin" ]; then
        otool -L "$1" 2>/dev/null | awk 'NR>1 {print $1}' | grep '@rpath/' | sed 's|@rpath/||'
    else
        # `ldd` resolves through rpath and prints "<soname> => <path>"; for an
        # unresolvable dep it prints "<soname> => not found", which still gives
        # us the soname in $1 — so a missing lib is reported, not skipped.
        ldd "$1" 2>/dev/null | awk '{print $1}' | grep -E "$MOJO_LIB_RE"
    fi
}

# Direct dependencies, plus one transitive level: the Mojo runtime libs
# reference each other (binary -> libKGENCompilerRTShared ->
# libAsyncRTRuntimeGlobals + libMSupportGlobals), and only the first level
# shows up in the binary itself.
WANTED="$(deps_of "$BINARY")"
for lib in $WANTED; do
    [ -f "$LIBDIR/$lib" ] && WANTED="$WANTED $(deps_of "$LIBDIR/$lib")"
done
WANTED="$(printf '%s\n' $WANTED | sort -u)"

mkdir -p "$OUTDIR"
MISSING=0

for lib in $WANTED; do
    if [ -f "$LIBDIR/$lib" ]; then
        cp "$LIBDIR/$lib" "$OUTDIR/"
        echo "  bundled: $lib"
    else
        echo "  MISSING: $lib (required by $(basename "$BINARY"), not in $LIBDIR)"
        MISSING=1
    fi
done

if [ -z "$WANTED" ]; then
    echo "  MISSING: could not read the library dependencies of $BINARY"
    MISSING=1
fi

exit "$MISSING"
