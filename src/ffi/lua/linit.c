/*
** Pion custom linit.c — sandboxed library loading.
** Only loads: base, table, string, math (no io, os, debug, package, loadlib).
** See Copyright Notice in lua.h
*/

#define linit_c
#define LUA_LIB

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

static const luaL_Reg lualibs[] = {
  {"", luaopen_base},
  {LUA_TABLIBNAME, luaopen_table},
  {LUA_STRLIBNAME, luaopen_string},
  {LUA_MATHLIBNAME, luaopen_math},
  {NULL, NULL}
};

LUALIB_API void luaL_openlibs (lua_State *L) {
  const luaL_Reg *lib = lualibs;
  for (; lib->func; lib++) {
    lua_pushcfunction(L, lib->func);
    lua_pushstring(L, lib->name);
    lua_call(L, 1, 0);
  }

  /* Remove dangerous base functions */
  lua_pushnil(L); lua_setglobal(L, "dofile");
  lua_pushnil(L); lua_setglobal(L, "loadfile");
  lua_pushnil(L); lua_setglobal(L, "load");       /* Lua 5.1: loadstring is the alias */
  lua_pushnil(L); lua_setglobal(L, "loadstring");
}
