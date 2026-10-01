/*
 * Pion ↔ Lua 5.1 bridge header.
 * Coroutine-based execution: redis.call() yields to host for command dispatch.
 */

#ifndef PION_LUA_WRAP_H
#define PION_LUA_WRAP_H

#include <stdint.h>

/* Opaque handle to a per-worker Lua state */
typedef struct PionLuaState PionLuaState;

/* Execution status codes */
#define PION_LUA_OK           0   /* Script completed, result on stack */
#define PION_LUA_NEEDS_CMD    1   /* redis.call() yielded, read cmd args */
#define PION_LUA_ERROR       -1   /* Script error, error message available */

/* --- Lifecycle --- */
PionLuaState* pion_lua_new_state(int mem_limit, int insn_limit);
void          pion_lua_close(PionLuaState *S);

/* --- Script cache --- */
/* Load and compile a script. Writes 40-byte hex SHA1 to out_sha1.
   Returns 0 on success, -1 on compile error (error msg via pion_lua_get_error). */
int pion_lua_load_script(PionLuaState *S, const char *script, int script_len,
                         char *out_sha1);

/* Check if a SHA1 hex string is cached. Returns 1 if found, 0 if not. */
int pion_lua_script_exists(PionLuaState *S, const char *sha1_hex);

/* Flush all cached scripts. */
void pion_lua_script_flush(PionLuaState *S);

/* --- Execution (coroutine-based) --- */
/* Set KEYS table before execution. */
void pion_lua_set_keys(PionLuaState *S, const char **keys, const int *key_lens, int nkeys);

/* Set ARGV table before execution. */
void pion_lua_set_argv(PionLuaState *S, const char **argv, const int *argv_lens, int nargv);

/* Begin executing a cached script by SHA1 hex.
   Returns PION_LUA_OK, PION_LUA_NEEDS_CMD, or PION_LUA_ERROR. */
int pion_lua_exec_sha1(PionLuaState *S, const char *sha1_hex);

/* --- Command dispatch (when status == PION_LUA_NEEDS_CMD) --- */
/* Get number of args (including command name) from the yielded redis.call(). */
int pion_lua_get_call_nargs(PionLuaState *S);

/* Get the i-th argument (0 = command name, 1+ = args).
   Writes pointer and length. Pointer valid until next pion_lua call. */
void pion_lua_get_call_arg(PionLuaState *S, int idx, const char **out_ptr, int *out_len);

/* After dispatching a command, push the result and resume. */

/* Push bulk string result, then resume. */
int pion_lua_push_string_and_resume(PionLuaState *S, const char *s, int len);

/* Push integer result, then resume. */
int pion_lua_push_int_and_resume(PionLuaState *S, int64_t val);

/* Push nil/false result (Redis nil → Lua false), then resume. */
int pion_lua_push_nil_and_resume(PionLuaState *S);

/* Push +OK status result, then resume. */
int pion_lua_push_ok_and_resume(PionLuaState *S);

/* Push error and resume (for redis.pcall — wraps in {err=...} table). */
int pion_lua_push_error_and_resume(PionLuaState *S, const char *msg, int msg_len);

/* Push array header (creates table), pushes N elements, then resume.
   Elements must be pushed via pion_lua_array_push_* before calling _end. */
void pion_lua_array_begin(PionLuaState *S);
void pion_lua_array_push_string(PionLuaState *S, const char *s, int len);
void pion_lua_array_push_int(PionLuaState *S, int64_t val);
void pion_lua_array_push_nil(PionLuaState *S);
int  pion_lua_array_end_and_resume(PionLuaState *S);

/* --- Result reading (when status == PION_LUA_OK) --- */
/* Write the Lua result as RESP bytes into out_buf.
   Returns number of bytes written, or -1 if buffer too small. */
int pion_lua_get_result(PionLuaState *S, char *out_buf, int buf_size);

/* --- Error reading (when status == PION_LUA_ERROR) --- */
const char* pion_lua_get_error(PionLuaState *S);

/* --- pcall mode flag --- */
/* Set to 1 before a pcall dispatch, 0 for call. Affects error handling. */
void pion_lua_set_pcall_mode(PionLuaState *S, int pcall);
int  pion_lua_get_pcall_mode(PionLuaState *S);

/* --- Simple accessors (easier to call from Mojo FFI) --- */
/* Return pointer to i-th call argument string (0=cmd name). Valid until next call. */
const char* pion_lua_call_arg_ptr(PionLuaState *S, int idx);
/* Return length of i-th call argument. */
int pion_lua_call_arg_len(PionLuaState *S, int idx);

/* --- Functions API (Redis 7+ compatible) --- */

/* Load a library from code with #!lua name=<libname> shebang.
   If replace != 0, overwrite existing library with same name.
   Writes library name to out_name (max out_name_size bytes).
   Returns 0 on success, -1 on error (msg via pion_lua_get_error). */
int pion_lua_load_library(PionLuaState *S, const char *code, int code_len,
                          int replace, char *out_name, int out_name_size);

/* Delete a library by name. Returns 0 on success, -1 if not found. */
int pion_lua_delete_library(PionLuaState *S, const char *name);

/* Flush all libraries. */
void pion_lua_flush_libraries(PionLuaState *S);

/* Get number of loaded libraries. */
int pion_lua_library_count(PionLuaState *S);

/* Get library name by index. Returns pointer (valid until next call). */
const char* pion_lua_library_name(PionLuaState *S, int idx);

/* Get number of functions in library by index. */
int pion_lua_library_func_count(PionLuaState *S, int lib_idx);

/* Get function name in library. Returns pointer (valid until next call). */
const char* pion_lua_library_func_name(PionLuaState *S, int lib_idx, int func_idx);

/* Execute a named function (set KEYS/ARGV before calling).
   Returns PION_LUA_OK, PION_LUA_NEEDS_CMD, or PION_LUA_ERROR. */
int pion_lua_exec_function(PionLuaState *S, const char *func_name, int func_name_len);

#endif /* PION_LUA_WRAP_H */
