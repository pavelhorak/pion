/*
 * Pion ↔ Lua 5.1 bridge header. See lua_wrap.c: redis.call() runs
 * synchronously through the host's `pion_script_dispatch` (an @export in
 * src/main.mojo, resolved with dlsym), so a script runs the server's own
 * commands.
 *
 * Every run and FUNCTION operation that answers the client builds its RESP
 * reply in the state's output buffer: pion_lua_out() points at it, and the
 * call returns its length (or -1 when no reply could be built).
 */

#ifndef PION_LUA_WRAP_H
#define PION_LUA_WRAP_H

#include <stdint.h>

typedef struct PionLuaState PionLuaState;

/* --- Lifecycle --- */
/* mem_limit: bytes per Lua state, 0 = none. time_limit_ms: a script that runs
 * longer without writing is stopped, 0 = never. */
PionLuaState *pion_lua_new_state(int64_t mem_limit, int64_t time_limit_ms);
/* Process-wide defaults for states created with negative limits. */
void          pion_lua_set_defaults(int64_t mem_limit, int64_t time_limit_ms);
void          pion_lua_close(PionLuaState *S);
/* The context handed to pion_script_dispatch for the run that follows. */
void          pion_lua_set_host(PionLuaState *S, void *host);
int64_t       pion_lua_memory(PionLuaState *S);
const char   *pion_lua_out(PionLuaState *S);
int64_t       pion_lua_out_len(PionLuaState *S);

/* --- Scripts (EVAL) --- */
/* Compile and cache; writes the lower-case sha (41 bytes, NUL-terminated).
 * 0 on success, -1 with an error reply in pion_lua_out. */
int     pion_lua_load_script(PionLuaState *S, const char *code, int64_t len, char *out_sha);
int     pion_lua_script_exists(PionLuaState *S, const char *sha);
void    pion_lua_script_flush(PionLuaState *S);
int64_t pion_lua_run_script(PionLuaState *S, const char *sha,
                            const char **keys, const int64_t *key_lens, int64_t nkeys,
                            const char **args, const int64_t *arg_lens, int64_t nargs,
                            int64_t ro, int64_t client_resp);

/* --- Functions --- */
/* 1 = loaded (the library name is the reply), 0 = an error reply. */
int64_t     pion_lua_function_load(PionLuaState *S, const char *code, int64_t len, int64_t replace);
int64_t     pion_lua_function_delete(PionLuaState *S, const char *name, int64_t nlen);
void        pion_lua_function_flush(PionLuaState *S);
int64_t     pion_lua_function_list(PionLuaState *S, const char *pat, int64_t plen, int64_t withcode, int64_t resp);
int64_t     pion_lua_function_stats(PionLuaState *S, int64_t resp);
int64_t     pion_lua_function_dump(PionLuaState *S);
int64_t     pion_lua_dump_count(const char *p, int64_t n);
const char *pion_lua_dump_entry(const char *p, int64_t n, int64_t k, int64_t *len);
int64_t     pion_lua_library_name_of(const char *code, int64_t len, const char **name, int64_t *nlen);
int64_t     pion_lua_library_exists(PionLuaState *S, const char *name, int64_t nlen);
int64_t     pion_lua_library_count(PionLuaState *S);
const char *pion_lua_library_name(PionLuaState *S, int64_t i);
const char *pion_lua_library_code(PionLuaState *S, int64_t i, int64_t *len);
int64_t     pion_lua_run_function(PionLuaState *S, const char *name, int64_t nlen,
                                  const char **keys, const int64_t *key_lens, int64_t nkeys,
                                  const char **args, const int64_t *arg_lens, int64_t nargs,
                                  int64_t ro, int64_t client_resp);

int64_t     pion_lua_function_exists(PionLuaState *S, const char *name, int64_t nlen);

/* --- Persistence: this worker thread's state --- */
/* WAL/snapshot/replication function records: 35 LOAD, 36 DELETE, 37 FLUSH. */
int64_t     pion_lua_wal_apply(int64_t cmd_id, const char *key, int64_t kl, const char *val, int64_t vl);
int64_t     pion_lua_tls_library_count(void);
const char *pion_lua_tls_library_name(int64_t i);
const char *pion_lua_tls_library_code(int64_t i, int64_t *len);

#endif /* PION_LUA_WRAP_H */
