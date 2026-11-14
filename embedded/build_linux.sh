#!/usr/bin/env bash
# Build libpion.so for this Linux host (x86-64 or aarch64), the counterpart of
# `pixi run build-embed` on macOS. A shared library needs position-independent
# C objects, and the server's build-ffi compiles without -fPIC, so the shims
# and Lua are compiled here into their own directory.
#   bash embedded/build_linux.sh            # -> ./libpion.so
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OBJ="$(mktemp -d)"
trap 'rm -rf "$OBJ"' EXIT
CF="-O3 -fPIC"
for c in crash_wrap fcntl_wrap geo_math uring_wrap xdp_wrap ssm_state_wrap build_pool_wrap worker_spawn_wrap; do
    gcc $CF -c "src/ffi/$c.c" -o "$OBJ/$c.o"
done
mkdir -p "$OBJ/lua"
for f in src/ffi/lua/l*.c src/ffi/lua/strbuf.c src/ffi/lua/fpconv.c src/ffi/lua/lua_cjson.c; do
    [ -f "$f" ] || continue
    gcc -O2 -fPIC -DLUA_USE_POSIX -Isrc/ffi/lua -c "$f" -o "$OBJ/lua/$(basename "${f%.c}").o"
done
ar rcs "$OBJ/liblua.a" "$OBJ"/lua/*.o
gcc -O2 -fPIC -c src/ffi/lua_wrap.c -Isrc/ffi/lua -Isrc/ffi -o "$OBJ/lua_wrap.o"
case "$(uname -m)" in
    x86_64) CPU=x86-64-v2 ;;     # the release's portable target
    aarch64|arm64) CPU=neoverse-n1 ;;
    *) echo "unsupported machine $(uname -m)" >&2; exit 1 ;;
esac
LINK=""
for o in crash_wrap build_pool_wrap worker_spawn_wrap fcntl_wrap geo_math uring_wrap xdp_wrap ssm_state_wrap lua_wrap; do
    LINK="$LINK -Xlinker $OBJ/$o.o"
done
pixi run mojo build --emit shared-lib -O3 --target-cpu "$CPU" -I . src/lib/pion_embed.mojo \
    $LINK -Xlinker "$OBJ/liblua.a" -Xlinker -lm -Xlinker -lpthread -Xlinker -ldl -o libpion.so
ls -la libpion.so
echo "highest GLIBC version required: $(strings -a libpion.so | grep -o 'GLIBC_2\.[0-9]*' | sort -t. -k2 -n | tail -1)"
