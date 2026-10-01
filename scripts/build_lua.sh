#!/bin/bash
# Build Lua 5.1.5 static library (with cjson) and Pion Lua bridge
set -e
cd src/ffi/lua
for f in l*.c strbuf.c fpconv.c lua_cjson.c; do
    [ -f "$f" ] && gcc -O2 -DLUA_USE_POSIX -I. -c "$f" -o "${f%.c}.o" 2>/dev/null
done
ar rcs liblua.a *.o
echo '[Lua] Built liblua.a'
cd ../../..
gcc -O2 -c src/ffi/lua_wrap.c -I src/ffi/lua -I src/ffi -o src/ffi/lua_wrap.o
echo '[Lua] Built lua_wrap.o'
