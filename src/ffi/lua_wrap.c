/*
 * Pion ↔ Lua 5.1 bridge.
 *
 * Coroutine-based execution model:
 *   1. Script runs inside a Lua coroutine (lua_newthread).
 *   2. redis.call("CMD", ...) yields the coroutine back to the host.
 *   3. Host (Mojo) reads the command args, dispatches, pushes result, resumes.
 *   4. Repeat until the script returns or errors.
 *
 * This avoids C→Mojo callbacks entirely.
 */

#include "lua/lua.h"
#include "lua/lauxlib.h"
#include "lua/lualib.h"
#include "lua_wrap.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* lua-cjson: luaopen_cjson declared here (defined in lua_cjson.c) */
extern int luaopen_cjson(lua_State *L);

/* ── SHA1 (minimal implementation for script hashing) ── */

typedef struct {
    uint32_t state[5];
    uint64_t count;
    uint8_t  buffer[64];
} SHA1_CTX;

static void sha1_transform(uint32_t state[5], const uint8_t block[64]) {
    uint32_t a, b, c, d, e, w[80];
    int i;
    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8)  | (uint32_t)block[i*4+3];
    }
    for (i = 16; i < 80; i++) {
        uint32_t t = w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16];
        w[i] = (t << 1) | (t >> 31);
    }
    a = state[0]; b = state[1]; c = state[2]; d = state[3]; e = state[4];
    for (i = 0; i < 80; i++) {
        uint32_t f, k, temp;
        if (i < 20)      { f = (b & c) | ((~b) & d); k = 0x5A827999; }
        else if (i < 40) { f = b ^ c ^ d;             k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
        else              { f = b ^ c ^ d;             k = 0xCA62C1D6; }
        temp = ((a << 5) | (a >> 27)) + f + e + k + w[i];
        e = d; d = c; c = (b << 30) | (b >> 2); b = a; a = temp;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e;
}

static void sha1_init(SHA1_CTX *ctx) {
    ctx->state[0] = 0x67452301; ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE; ctx->state[3] = 0x10325476;
    ctx->state[4] = 0xC3D2E1F0; ctx->count = 0;
}

static void sha1_update(SHA1_CTX *ctx, const uint8_t *data, size_t len) {
    size_t i = 0, idx = (size_t)(ctx->count & 63);
    ctx->count += len;
    if (idx) {
        size_t left = 64 - idx;
        if (len < left) { memcpy(ctx->buffer + idx, data, len); return; }
        memcpy(ctx->buffer + idx, data, left);
        sha1_transform(ctx->state, ctx->buffer);
        i = left;
    }
    for (; i + 64 <= len; i += 64) sha1_transform(ctx->state, data + i);
    if (i < len) memcpy(ctx->buffer, data + i, len - i);
}

static void sha1_final(SHA1_CTX *ctx, uint8_t digest[20]) {
    uint8_t pad[64]; uint64_t bits = ctx->count * 8;
    size_t idx = (size_t)(ctx->count & 63);
    memset(pad, 0, 64); pad[0] = 0x80;
    size_t padlen = (idx < 56) ? (56 - idx) : (120 - idx);
    sha1_update(ctx, pad, padlen);
    uint8_t bits_be[8];
    for (int i = 0; i < 8; i++) bits_be[i] = (uint8_t)(bits >> (56 - i*8));
    sha1_update(ctx, bits_be, 8);
    for (int i = 0; i < 5; i++) {
        digest[i*4+0] = (uint8_t)(ctx->state[i] >> 24);
        digest[i*4+1] = (uint8_t)(ctx->state[i] >> 16);
        digest[i*4+2] = (uint8_t)(ctx->state[i] >> 8);
        digest[i*4+3] = (uint8_t)(ctx->state[i]);
    }
}

static void sha1_hex(const uint8_t digest[20], char hex[41]) {
    static const char hc[] = "0123456789abcdef";
    for (int i = 0; i < 20; i++) {
        hex[i*2]   = hc[digest[i] >> 4];
        hex[i*2+1] = hc[digest[i] & 0xF];
    }
    hex[40] = '\0';
}

/* ── Script cache ── */

#define MAX_CACHED_SCRIPTS 256

typedef struct {
    char sha1_hex[41];
    int  ref;           /* Lua registry reference to compiled chunk */
} CachedScript;

/* ── Function library registry ── */

#define MAX_LIBRARIES 64
#define MAX_FUNCS_PER_LIB 32
#define MAX_NAME_LEN 64

typedef struct {
    char name[MAX_NAME_LEN];
    int  ref;           /* Lua registry reference to function closure */
} RegisteredFunc;

typedef struct {
    char           name[MAX_NAME_LEN];
    int            code_ref;  /* registry ref to library chunk (for reload) */
    RegisteredFunc funcs[MAX_FUNCS_PER_LIB];
    int            func_count;
} Library;

/* ── Memory-limited allocator ── */

/*
 * The cap binds only while SCRIPT code runs (`enforce`, set around every
 * resume/pcall of user code). Host bookkeeping — copying KEYS/ARGV in,
 * creating the coroutine, registry refs, pushing redis.call() replies —
 * runs outside any protected call, where a failed allocation is not a Lua
 * error but a panic that ends the process. Script garbage left behind at the
 * cap used to make exactly that happen on the NEXT command (gh #410).
 */
typedef struct {
    size_t used;
    size_t limit;
    int    enforce;
} LuaMemCtx;

static void *lua_mem_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
    LuaMemCtx *ctx = (LuaMemCtx *)ud;
    if (nsize == 0) {
        free(ptr);
        ctx->used -= osize;
        return NULL;
    }
    if (nsize > osize && ctx->enforce && ctx->used - osize + nsize > ctx->limit) {
        return NULL;  /* inside a script: Lua raises "not enough memory" */
    }
    void *p = realloc(ptr, nsize);
    if (p == NULL) {
        /* Lua assumes a shrink never fails: keep the larger block. */
        if (nsize <= osize) { ctx->used = ctx->used - osize + nsize; return ptr; }
        return NULL;
    }
    ctx->used = ctx->used - osize + nsize;
    return p;
}

/* An error outside every protected call. The bridge keeps script code and the
 * memory cap inside protected calls, so reaching this is a bug: say so and
 * abort, so the crash log records a signal and a backtrace instead of the
 * silent exit(1) that Lua would otherwise take after a panic function returns. */
static int pion_lua_panic(lua_State *L) {
    const char *msg = lua_tostring(L, -1);
    fprintf(stderr, "[Lua] PANIC: unprotected error in the Lua bridge: %s\n",
            msg ? msg : "(no message)");
    fflush(stderr);
    abort();
    return 0;
}

/* ── PionLuaState ── */

struct PionLuaState {
    lua_State   *L;             /* main Lua state */
    lua_State   *co;            /* coroutine thread for current execution */
    int          co_ref;        /* registry ref for coroutine (prevents GC) */
    LuaMemCtx    mem_ctx;
    int          insn_limit;
    CachedScript cache[MAX_CACHED_SCRIPTS];
    int          cache_count;
    int          pcall_mode;    /* 1 = redis.pcall (errors as tables, not raises) */
    char         error_buf[1024];
    /* Array builder state */
    int          array_count;   /* number of elements pushed via array_push_* */
    /* Functions API (Redis 7+ libraries) */
    Library      libs[MAX_LIBRARIES];
    int          lib_count;
    /* Temporary storage for register_function during library load */
    Library     *loading_lib;   /* points to libs[lib_count] during load */
};

/* ── Instruction limit hook ── */

static void insn_hook(lua_State *L, lua_Debug *ar) {
    (void)ar;
    luaL_error(L, "script exceeded instruction limit");
}

/* Run user code on the coroutine with the memory cap in force. */
static int run_script(PionLuaState *S, int nargs) {
    S->mem_ctx.enforce = 1;
    int status = lua_resume(S->co, nargs);
    S->mem_ctx.enforce = 0;
    return status;
}

/* Drop the previous execution's coroutine. A script that ran into the cap
 * leaves its garbage behind, and Lua 5.1 has no emergency collection, so
 * without a full cycle here every later script would fail on memory that is
 * already unreachable. Below half the cap the incremental collector is left
 * to do its normal work. */
static void release_coroutine(PionLuaState *S) {
    if (S->co_ref != LUA_NOREF) {
        luaL_unref(S->L, LUA_REGISTRYINDEX, S->co_ref);
        S->co_ref = LUA_NOREF;
        S->co = NULL;
    }
    if (S->mem_ctx.used > S->mem_ctx.limit / 2) {
        lua_gc(S->L, LUA_GCCOLLECT, 0);
    }
}

/* Set a global without consulting a metatable a script may have put on _G:
 * the host writes KEYS/ARGV outside any protected call. */
static void raw_setglobal(lua_State *L, const char *name) {
    lua_pushstring(L, name);
    lua_insert(L, -2);
    lua_rawset(L, LUA_GLOBALSINDEX);
}

static void raw_getglobal(lua_State *L, const char *name) {
    lua_pushstring(L, name);
    lua_rawget(L, LUA_GLOBALSINDEX);
}

/* ── redis.call / redis.pcall ── */

/*
 * redis.call("CMD", arg1, arg2, ...) → yields all args to the host.
 * The host dispatches the command and pushes the result before resuming.
 */
static int lua_redis_call(lua_State *L) {
    /* Yield all arguments on the stack to the host */
    return lua_yield(L, lua_gettop(L));
}

/*
 * redis.pcall — same as redis.call, but the host wraps errors in {err=...}
 * instead of raising. We signal pcall mode via a flag on the PionLuaState.
 * Since redis.pcall uses the same yield mechanism, we set a flag in the
 * registry before calling to differentiate.
 */
static int lua_redis_pcall(lua_State *L) {
    /* Set pcall flag in registry */
    lua_pushboolean(L, 1);
    lua_setfield(L, LUA_REGISTRYINDEX, "__pion_pcall");
    return lua_yield(L, lua_gettop(L));
}

/* redis.log(level, msg) — simple print */
static int lua_redis_log(lua_State *L) {
    int nargs = lua_gettop(L);
    if (nargs >= 2) {
        const char *msg = lua_tostring(L, 2);
        if (msg) fprintf(stderr, "[Lua] %s\n", msg);
    }
    return 0;
}

/* redis.error_reply(msg) → {err=msg} */
static int lua_redis_error_reply(lua_State *L) {
    lua_newtable(L);
    lua_pushvalue(L, 1);
    lua_setfield(L, -2, "err");
    return 1;
}

/* redis.status_reply(msg) → {ok=msg} */
static int lua_redis_status_reply(lua_State *L) {
    lua_newtable(L);
    lua_pushvalue(L, 1);
    lua_setfield(L, -2, "ok");
    return 1;
}

/*
 * redis.register_function(name, callback)
 * OR redis.register_function{function_name=name, callback=func}
 * Called during FUNCTION LOAD to register named functions.
 * Stores the callback ref in the loading library's func table.
 */
static int lua_redis_register_function(lua_State *L) {
    const char *fname = NULL;
    int func_stack_idx = 0;

    if (lua_type(L, 1) == LUA_TTABLE) {
        /* Table form: {function_name="name", callback=func} */
        lua_getfield(L, 1, "function_name");
        fname = lua_tostring(L, -1);
        lua_getfield(L, 1, "callback");
        func_stack_idx = lua_gettop(L);
    } else {
        /* Simple form: register_function("name", func) */
        fname = luaL_checkstring(L, 1);
        luaL_checktype(L, 2, LUA_TFUNCTION);
        func_stack_idx = 2;
    }

    if (!fname) return luaL_error(L, "register_function: missing function name");

    /* Get PionLuaState from registry */
    lua_getfield(L, LUA_REGISTRYINDEX, "__pion_state");
    PionLuaState *S = (PionLuaState *)lua_touserdata(L, -1);
    lua_pop(L, 1);

    if (!S || !S->loading_lib) {
        return luaL_error(L, "register_function: not inside FUNCTION LOAD");
    }
    if (S->loading_lib->func_count >= MAX_FUNCS_PER_LIB) {
        return luaL_error(L, "register_function: too many functions in library");
    }

    /* Store function reference */
    RegisteredFunc *rf = &S->loading_lib->funcs[S->loading_lib->func_count];
    strncpy(rf->name, fname, MAX_NAME_LEN - 1);
    rf->name[MAX_NAME_LEN - 1] = '\0';

    lua_pushvalue(L, func_stack_idx);
    rf->ref = luaL_ref(L, LUA_REGISTRYINDEX);
    S->loading_lib->func_count++;

    return 0;
}

/* redis.sha1hex(s) → SHA1 hex of string */
static int lua_redis_sha1hex(lua_State *L) {
    size_t len;
    const char *s = luaL_checklstring(L, 1, &len);
    SHA1_CTX ctx; uint8_t digest[20]; char hex[41];
    sha1_init(&ctx);
    sha1_update(&ctx, (const uint8_t *)s, len);
    sha1_final(&ctx, digest);
    sha1_hex(digest, hex);
    lua_pushlstring(L, hex, 40);
    return 1;
}

/* ── Register redis.* table ── */

static void register_redis_table(lua_State *L) {
    lua_newtable(L);

    lua_pushcfunction(L, lua_redis_call);
    lua_setfield(L, -2, "call");

    lua_pushcfunction(L, lua_redis_pcall);
    lua_setfield(L, -2, "pcall");

    lua_pushcfunction(L, lua_redis_log);
    lua_setfield(L, -2, "log");

    lua_pushcfunction(L, lua_redis_error_reply);
    lua_setfield(L, -2, "error_reply");

    lua_pushcfunction(L, lua_redis_status_reply);
    lua_setfield(L, -2, "status_reply");

    lua_pushcfunction(L, lua_redis_sha1hex);
    lua_setfield(L, -2, "sha1hex");

    lua_pushcfunction(L, lua_redis_register_function);
    lua_setfield(L, -2, "register_function");

    /* Log levels (match Redis) */
    lua_pushinteger(L, 0); lua_setfield(L, -2, "LOG_DEBUG");
    lua_pushinteger(L, 1); lua_setfield(L, -2, "LOG_VERBOSE");
    lua_pushinteger(L, 2); lua_setfield(L, -2, "LOG_NOTICE");
    lua_pushinteger(L, 3); lua_setfield(L, -2, "LOG_WARNING");

    lua_setglobal(L, "redis");
}

/* ── RESP result serialization ── */

/*
 * Serialize a Lua value at stack index `idx` to RESP bytes.
 * Returns bytes written, or -1 on overflow.
 */
static int lua_to_resp(lua_State *L, int idx, char *buf, int buf_size) {
    int pos = 0;
    int type = lua_type(L, idx);

    switch (type) {
    case LUA_TSTRING: {
        size_t slen;
        const char *s = lua_tolstring(L, idx, &slen);
        /* $<len>\r\n<data>\r\n */
        int hdr = snprintf(buf + pos, buf_size - pos, "$%d\r\n", (int)slen);
        if (hdr < 0 || pos + hdr + (int)slen + 2 > buf_size) return -1;
        pos += hdr;
        memcpy(buf + pos, s, slen); pos += (int)slen;
        buf[pos++] = '\r'; buf[pos++] = '\n';
        return pos;
    }
    case LUA_TNUMBER: {
        lua_Number n = lua_tonumber(L, idx);
        int64_t ival = (int64_t)n;
        /* Check if it's an integer */
        if ((lua_Number)ival == n) {
            int w = snprintf(buf + pos, buf_size - pos, ":%lld\r\n", (long long)ival);
            if (w < 0 || pos + w > buf_size) return -1;
            return pos + w;
        } else {
            /* Float as bulk string */
            char tmp[64];
            int tl = snprintf(tmp, sizeof(tmp), "%.17g", (double)n);
            int w = snprintf(buf + pos, buf_size - pos, "$%d\r\n%s\r\n", tl, tmp);
            if (w < 0 || pos + w > buf_size) return -1;
            return pos + w;
        }
    }
    case LUA_TBOOLEAN: {
        if (lua_toboolean(L, idx)) {
            /* true → :1 */
            if (pos + 5 > buf_size) return -1;
            memcpy(buf + pos, ":1\r\n", 4); return pos + 4;
        } else {
            /* false → $-1 (nil in Redis) */
            if (pos + 6 > buf_size) return -1;
            memcpy(buf + pos, "$-1\r\n", 5); return pos + 5;
        }
    }
    case LUA_TTABLE: {
        /* Check for {err=...} or {ok=...}. Raw access, never lua_getfield:
         * result serialization runs OUTSIDE any protected call, so a returned
         * table with a metamethod that errors (e.g. __index) would panic the
         * process rather than raise a Lua error (gh #410). idx is absolute. */
        lua_pushliteral(L, "err"); lua_rawget(L, idx);
        if (!lua_isnil(L, -1)) {
            size_t elen;
            const char *e = lua_tolstring(L, -1, &elen);
            int w = snprintf(buf + pos, buf_size - pos, "-ERR %.*s\r\n",
                             (int)elen, e ? e : "");
            lua_pop(L, 1);
            if (w < 0 || pos + w > buf_size) return -1;
            return pos + w;
        }
        lua_pop(L, 1);

        lua_pushliteral(L, "ok"); lua_rawget(L, idx);
        if (!lua_isnil(L, -1)) {
            size_t olen;
            const char *o = lua_tolstring(L, -1, &olen);
            int w = snprintf(buf + pos, buf_size - pos, "+%.*s\r\n",
                             (int)olen, o ? o : "");
            lua_pop(L, 1);
            if (w < 0 || pos + w > buf_size) return -1;
            return pos + w;
        }
        lua_pop(L, 1);

        /* Array table: count elements via rawlen / iteration */
        int len = 0;
        /* Count sequential integer keys starting from 1 */
        while (1) {
            lua_rawgeti(L, idx, len + 1);
            if (lua_isnil(L, -1)) { lua_pop(L, 1); break; }
            lua_pop(L, 1);
            len++;
        }

        int hdr = snprintf(buf + pos, buf_size - pos, "*%d\r\n", len);
        if (hdr < 0 || pos + hdr > buf_size) return -1;
        pos += hdr;

        for (int i = 1; i <= len; i++) {
            lua_rawgeti(L, idx, i);
            int w = lua_to_resp(L, lua_gettop(L), buf + pos, buf_size - pos);
            lua_pop(L, 1);
            if (w < 0) return -1;
            pos += w;
        }
        return pos;
    }
    case LUA_TNIL:
    default:
        /* nil → $-1 */
        if (pos + 6 > buf_size) return -1;
        memcpy(buf + pos, "$-1\r\n", 5);
        return pos + 5;
    }
}

/* ── Public API ── */

PionLuaState* pion_lua_new_state(int mem_limit, int insn_limit) {
    PionLuaState *S = (PionLuaState *)calloc(1, sizeof(PionLuaState));
    if (!S) return NULL;

    S->mem_ctx.used = 0;
    S->mem_ctx.limit = mem_limit > 0 ? (size_t)mem_limit : (size_t)(1 << 20); /* default 1MB */
    S->mem_ctx.enforce = 0;
    S->insn_limit = insn_limit > 0 ? insn_limit : 1000000;     /* default 1M instructions */

    S->L = lua_newstate(lua_mem_alloc, &S->mem_ctx);
    if (!S->L) { free(S); return NULL; }
    lua_atpanic(S->L, pion_lua_panic);

    /* Open sandboxed libraries */
    luaL_openlibs(S->L);

    /* Register redis.* table */
    register_redis_table(S->L);

    /* Register cjson library as global */
    lua_pushcfunction(S->L, luaopen_cjson);
    lua_call(S->L, 0, 1);       /* returns module table on stack */
    lua_setglobal(S->L, "cjson"); /* set as global */

    /* Remove print (use redis.log instead) */
    lua_pushnil(S->L); lua_setglobal(S->L, "print");

    S->co = NULL;
    S->co_ref = LUA_NOREF;
    S->cache_count = 0;
    S->pcall_mode = 0;
    S->array_count = 0;
    S->lib_count = 0;
    S->loading_lib = NULL;

    /* Store PionLuaState* in registry for redis.register_function access */
    lua_pushlightuserdata(S->L, S);
    lua_setfield(S->L, LUA_REGISTRYINDEX, "__pion_state");

    return S;
}

void pion_lua_close(PionLuaState *S) {
    if (!S) return;
    if (S->L) lua_close(S->L);
    free(S);
}

int pion_lua_load_script(PionLuaState *S, const char *script, int script_len,
                         char *out_sha1) {
    if (!S || !S->L) return -1;

    /* Compute SHA1 */
    SHA1_CTX sha; uint8_t digest[20];
    sha1_init(&sha);
    sha1_update(&sha, (const uint8_t *)script, (size_t)script_len);
    sha1_final(&sha, digest);
    sha1_hex(digest, out_sha1);

    /* Check if already cached */
    for (int i = 0; i < S->cache_count; i++) {
        if (memcmp(S->cache[i].sha1_hex, out_sha1, 40) == 0) {
            return 0; /* already loaded */
        }
    }

    /* Compile */
    if (luaL_loadbuffer(S->L, script, (size_t)script_len, "user_script") != 0) {
        snprintf(S->error_buf, sizeof(S->error_buf), "%s", lua_tostring(S->L, -1));
        lua_pop(S->L, 1);
        return -1;
    }

    /* Store in registry */
    int ref = luaL_ref(S->L, LUA_REGISTRYINDEX);

    if (S->cache_count < MAX_CACHED_SCRIPTS) {
        memcpy(S->cache[S->cache_count].sha1_hex, out_sha1, 41);
        S->cache[S->cache_count].ref = ref;
        S->cache_count++;
    }

    return 0;
}

int pion_lua_script_exists(PionLuaState *S, const char *sha1_hex) {
    if (!S) return 0;
    for (int i = 0; i < S->cache_count; i++) {
        if (memcmp(S->cache[i].sha1_hex, sha1_hex, 40) == 0) return 1;
    }
    return 0;
}

void pion_lua_script_flush(PionLuaState *S) {
    if (!S || !S->L) return;
    for (int i = 0; i < S->cache_count; i++) {
        luaL_unref(S->L, LUA_REGISTRYINDEX, S->cache[i].ref);
    }
    S->cache_count = 0;
}

void pion_lua_set_keys(PionLuaState *S, const char **keys, const int *key_lens, int nkeys) {
    if (!S || !S->L) return;
    lua_newtable(S->L);
    for (int i = 0; i < nkeys; i++) {
        lua_pushlstring(S->L, keys[i], (size_t)key_lens[i]);
        lua_rawseti(S->L, -2, i + 1);
    }
    raw_setglobal(S->L, "KEYS");
}

void pion_lua_set_argv(PionLuaState *S, const char **argv, const int *argv_lens, int nargv) {
    if (!S || !S->L) return;
    lua_newtable(S->L);
    for (int i = 0; i < nargv; i++) {
        lua_pushlstring(S->L, argv[i], (size_t)argv_lens[i]);
        lua_rawseti(S->L, -2, i + 1);
    }
    raw_setglobal(S->L, "ARGV");
}

int pion_lua_exec_sha1(PionLuaState *S, const char *sha1_hex) {
    if (!S || !S->L) return PION_LUA_ERROR;

    /* Find cached script */
    int ref = LUA_NOREF;
    for (int i = 0; i < S->cache_count; i++) {
        if (memcmp(S->cache[i].sha1_hex, sha1_hex, 40) == 0) {
            ref = S->cache[i].ref;
            break;
        }
    }
    if (ref == LUA_NOREF) {
        snprintf(S->error_buf, sizeof(S->error_buf), "NOSCRIPT No matching script. Use EVAL to load.");
        return PION_LUA_ERROR;
    }

    /* Clean up previous coroutine (frees its garbage, GCs past half the cap) */
    release_coroutine(S);

    /* Create new coroutine */
    S->co = lua_newthread(S->L);
    S->co_ref = luaL_ref(S->L, LUA_REGISTRYINDEX);  /* prevent GC */
    S->pcall_mode = 0;

    /* Set instruction limit hook on the coroutine */
    lua_sethook(S->co, insn_hook, LUA_MASKCOUNT, S->insn_limit);

    /* Reset memory counter for this execution */
    /* (We don't reset S->mem_ctx.used because Lua state itself uses memory) */

    /* Push the cached function onto the coroutine stack */
    lua_rawgeti(S->L, LUA_REGISTRYINDEX, ref);
    lua_xmove(S->L, S->co, 1);

    /* Copy KEYS and ARGV globals to coroutine's environment */
    /* (Globals are shared in Lua 5.1 — KEYS/ARGV set on main state are visible) */

    /* Resume the coroutine (0 arguments) — memory cap in force for script code */
    int status = run_script(S, 0);

    if (status == 0) {
        /* Script completed normally */
        return PION_LUA_OK;
    } else if (status == LUA_YIELD) {
        /* redis.call/pcall yielded — check pcall flag */
        lua_getfield(S->co, LUA_REGISTRYINDEX, "__pion_pcall");
        S->pcall_mode = lua_toboolean(S->co, -1);
        lua_pop(S->co, 1);
        /* Clear the flag for next time */
        lua_pushnil(S->co);
        lua_setfield(S->co, LUA_REGISTRYINDEX, "__pion_pcall");
        return PION_LUA_NEEDS_CMD;
    } else {
        /* Error */
        const char *err = lua_tostring(S->co, -1);
        snprintf(S->error_buf, sizeof(S->error_buf), "%s", err ? err : "unknown error");
        return PION_LUA_ERROR;
    }
}

int pion_lua_get_call_nargs(PionLuaState *S) {
    if (!S || !S->co) return 0;
    return lua_gettop(S->co);
}

void pion_lua_get_call_arg(PionLuaState *S, int idx, const char **out_ptr, int *out_len) {
    if (!S || !S->co || idx < 0 || idx >= lua_gettop(S->co)) {
        *out_ptr = NULL; *out_len = 0; return;
    }
    size_t len;
    /* Lua stack is 1-indexed */
    const char *s = lua_tolstring(S->co, idx + 1, &len);
    *out_ptr = s;
    *out_len = (int)len;
}

/* Resume the coroutine after pushing a result.
   The result value should already be on top of co's stack (pushed by caller).
   Returns PION_LUA_OK, PION_LUA_NEEDS_CMD, or PION_LUA_ERROR. */
static int do_resume(PionLuaState *S) {
    /* Clear all yielded args from stack, keep only the result value on top */
    /* Actually, lua_resume handles this: narg=1 means "1 value pushed as result" */
    int status = run_script(S, 1);

    if (status == 0) {
        return PION_LUA_OK;
    } else if (status == LUA_YIELD) {
        lua_getfield(S->co, LUA_REGISTRYINDEX, "__pion_pcall");
        S->pcall_mode = lua_toboolean(S->co, -1);
        lua_pop(S->co, 1);
        lua_pushnil(S->co);
        lua_setfield(S->co, LUA_REGISTRYINDEX, "__pion_pcall");
        return PION_LUA_NEEDS_CMD;
    } else {
        const char *err = lua_tostring(S->co, -1);
        snprintf(S->error_buf, sizeof(S->error_buf), "%s", err ? err : "unknown error");
        return PION_LUA_ERROR;
    }
}

int pion_lua_push_string_and_resume(PionLuaState *S, const char *s, int len) {
    if (!S || !S->co) return PION_LUA_ERROR;
    /* Clear yielded args */
    lua_settop(S->co, 0);
    lua_pushlstring(S->co, s, (size_t)len);
    return do_resume(S);
}

int pion_lua_push_int_and_resume(PionLuaState *S, int64_t val) {
    if (!S || !S->co) return PION_LUA_ERROR;
    lua_settop(S->co, 0);
    lua_pushinteger(S->co, (lua_Integer)val);
    return do_resume(S);
}

int pion_lua_push_nil_and_resume(PionLuaState *S) {
    if (!S || !S->co) return PION_LUA_ERROR;
    lua_settop(S->co, 0);
    /* Redis nil → Lua false (not nil, because nil terminates tables) */
    lua_pushboolean(S->co, 0);
    return do_resume(S);
}

int pion_lua_push_ok_and_resume(PionLuaState *S) {
    if (!S || !S->co) return PION_LUA_ERROR;
    lua_settop(S->co, 0);
    lua_newtable(S->co);
    lua_pushstring(S->co, "OK");
    lua_setfield(S->co, -2, "ok");
    return do_resume(S);
}

int pion_lua_push_error_and_resume(PionLuaState *S, const char *msg, int msg_len) {
    if (!S || !S->co) return PION_LUA_ERROR;
    lua_settop(S->co, 0);
    if (S->pcall_mode) {
        /* pcall: return {err=...} table */
        lua_newtable(S->co);
        lua_pushlstring(S->co, msg, (size_t)msg_len);
        lua_setfield(S->co, -2, "err");
    } else {
        /* call: raise error. EVAL reports S->error_buf, so the message must
         * land there too — it used to be pushed only onto the dead coroutine,
         * and every redis.call() error (WRONGTYPE, arity, -OOM from gh #261)
         * reached the client as "ERR unknown Lua error". */
        lua_pushlstring(S->co, msg, (size_t)msg_len);
        int n = msg_len < (int)sizeof(S->error_buf) - 1 ? msg_len : (int)sizeof(S->error_buf) - 1;
        memcpy(S->error_buf, msg, (size_t)n);
        S->error_buf[n] = '\0';
        return PION_LUA_ERROR;
    }
    return do_resume(S);
}

void pion_lua_array_begin(PionLuaState *S) {
    if (!S || !S->co) return;
    lua_settop(S->co, 0);
    lua_newtable(S->co);
    S->array_count = 0;
}

void pion_lua_array_push_string(PionLuaState *S, const char *s, int len) {
    if (!S || !S->co) return;
    S->array_count++;
    lua_pushlstring(S->co, s, (size_t)len);
    lua_rawseti(S->co, -2, S->array_count);
}

void pion_lua_array_push_int(PionLuaState *S, int64_t val) {
    if (!S || !S->co) return;
    S->array_count++;
    lua_pushinteger(S->co, (lua_Integer)val);
    lua_rawseti(S->co, -2, S->array_count);
}

void pion_lua_array_push_nil(PionLuaState *S) {
    if (!S || !S->co) return;
    S->array_count++;
    lua_pushboolean(S->co, 0);  /* Redis nil → Lua false */
    lua_rawseti(S->co, -2, S->array_count);
}

int pion_lua_array_end_and_resume(PionLuaState *S) {
    if (!S || !S->co) return PION_LUA_ERROR;
    /* Table is already on top of stack */
    return do_resume(S);
}

int pion_lua_get_result(PionLuaState *S, char *out_buf, int buf_size) {
    if (!S || !S->co) return -1;
    int top = lua_gettop(S->co);
    if (top == 0) {
        /* No return value → nil */
        if (buf_size < 5) return -1;
        memcpy(out_buf, "$-1\r\n", 5);
        return 5;
    }
    return lua_to_resp(S->co, 1, out_buf, buf_size);
}

const char* pion_lua_get_error(PionLuaState *S) {
    if (!S) return "null state";
    return S->error_buf;
}

void pion_lua_set_pcall_mode(PionLuaState *S, int pcall) {
    if (S) S->pcall_mode = pcall;
}

int pion_lua_get_pcall_mode(PionLuaState *S) {
    return S ? S->pcall_mode : 0;
}

/* ── Functions API ── */

int pion_lua_load_library(PionLuaState *S, const char *code, int code_len,
                          int replace, char *out_name, int out_name_size) {
    if (!S || !S->L) return -1;

    /* Parse shebang: #!lua name=<libname> */
    const char *name_start = NULL;
    int name_len = 0;
    if (code_len > 10 && code[0] == '#' && code[1] == '!') {
        /* Find "name=" */
        for (int i = 2; i < code_len - 5; i++) {
            if (code[i] == 'n' && code[i+1] == 'a' && code[i+2] == 'm' &&
                code[i+3] == 'e' && code[i+4] == '=') {
                name_start = code + i + 5;
                /* Find end of name (newline or space) */
                for (int j = i + 5; j < code_len; j++) {
                    if (code[j] == '\n' || code[j] == '\r' || code[j] == ' ') break;
                    name_len++;
                }
                break;
            }
        }
    }
    if (!name_start || name_len == 0 || name_len >= MAX_NAME_LEN) {
        snprintf(S->error_buf, sizeof(S->error_buf),
                 "ERR Missing or invalid library name. Library must start with #!lua name=<name>");
        return -1;
    }

    /* Copy name to output */
    int copy_len = name_len < out_name_size - 1 ? name_len : out_name_size - 1;
    memcpy(out_name, name_start, copy_len);
    out_name[copy_len] = '\0';

    /* Check if library already exists */
    for (int i = 0; i < S->lib_count; i++) {
        if (strncmp(S->libs[i].name, name_start, name_len) == 0 &&
            S->libs[i].name[name_len] == '\0') {
            if (!replace) {
                snprintf(S->error_buf, sizeof(S->error_buf),
                         "ERR Library '%s' already exists", S->libs[i].name);
                return -1;
            }
            /* Replace: unref old functions, reset */
            for (int f = 0; f < S->libs[i].func_count; f++) {
                luaL_unref(S->L, LUA_REGISTRYINDEX, S->libs[i].funcs[f].ref);
            }
            if (S->libs[i].code_ref != LUA_NOREF)
                luaL_unref(S->L, LUA_REGISTRYINDEX, S->libs[i].code_ref);
            S->libs[i].func_count = 0;
            S->libs[i].code_ref = LUA_NOREF;
            S->loading_lib = &S->libs[i];
            goto do_load;
        }
    }

    if (S->lib_count >= MAX_LIBRARIES) {
        snprintf(S->error_buf, sizeof(S->error_buf), "ERR too many libraries loaded");
        return -1;
    }

    /* New library slot */
    S->loading_lib = &S->libs[S->lib_count];
    memset(S->loading_lib, 0, sizeof(Library));
    memcpy(S->loading_lib->name, name_start, name_len);
    S->loading_lib->name[name_len] = '\0';
    S->loading_lib->code_ref = LUA_NOREF;

do_load:
    ;
    /* Skip shebang line before compiling */
    const char *lua_code = code;
    int lua_code_len = code_len;
    if (code_len > 2 && code[0] == '#' && code[1] == '!') {
        for (int i = 0; i < code_len; i++) {
            if (code[i] == '\n') {
                lua_code = code + i + 1;
                lua_code_len = code_len - i - 1;
                break;
            }
        }
    }

    /* Compile and execute the library code */
    if (luaL_loadbuffer(S->L, lua_code, (size_t)lua_code_len, out_name) != 0) {
        snprintf(S->error_buf, sizeof(S->error_buf), "%s", lua_tostring(S->L, -1));
        lua_pop(S->L, 1);
        S->loading_lib = NULL;
        return -1;
    }

    /* Execute — this should call redis.register_function() */
    if (lua_pcall(S->L, 0, 0, 0) != 0) {
        snprintf(S->error_buf, sizeof(S->error_buf), "%s", lua_tostring(S->L, -1));
        lua_pop(S->L, 1);
        S->loading_lib = NULL;
        return -1;
    }

    /* If this was a new library, bump count */
    if (S->loading_lib == &S->libs[S->lib_count]) {
        S->lib_count++;
    }
    S->loading_lib = NULL;
    return 0;
}

int pion_lua_delete_library(PionLuaState *S, const char *name) {
    if (!S) return -1;
    for (int i = 0; i < S->lib_count; i++) {
        if (strcmp(S->libs[i].name, name) == 0) {
            /* Unref all functions */
            for (int f = 0; f < S->libs[i].func_count; f++) {
                luaL_unref(S->L, LUA_REGISTRYINDEX, S->libs[i].funcs[f].ref);
            }
            if (S->libs[i].code_ref != LUA_NOREF)
                luaL_unref(S->L, LUA_REGISTRYINDEX, S->libs[i].code_ref);
            /* Shift remaining libraries down */
            for (int j = i; j < S->lib_count - 1; j++) {
                S->libs[j] = S->libs[j + 1];
            }
            S->lib_count--;
            return 0;
        }
    }
    snprintf(S->error_buf, sizeof(S->error_buf), "ERR Library not found");
    return -1;
}

void pion_lua_flush_libraries(PionLuaState *S) {
    if (!S || !S->L) return;
    for (int i = 0; i < S->lib_count; i++) {
        for (int f = 0; f < S->libs[i].func_count; f++) {
            luaL_unref(S->L, LUA_REGISTRYINDEX, S->libs[i].funcs[f].ref);
        }
        if (S->libs[i].code_ref != LUA_NOREF)
            luaL_unref(S->L, LUA_REGISTRYINDEX, S->libs[i].code_ref);
    }
    S->lib_count = 0;
}

int pion_lua_library_count(PionLuaState *S) {
    return S ? S->lib_count : 0;
}

const char* pion_lua_library_name(PionLuaState *S, int idx) {
    if (!S || idx < 0 || idx >= S->lib_count) return "";
    return S->libs[idx].name;
}

int pion_lua_library_func_count(PionLuaState *S, int lib_idx) {
    if (!S || lib_idx < 0 || lib_idx >= S->lib_count) return 0;
    return S->libs[lib_idx].func_count;
}

const char* pion_lua_library_func_name(PionLuaState *S, int lib_idx, int func_idx) {
    if (!S || lib_idx < 0 || lib_idx >= S->lib_count) return "";
    if (func_idx < 0 || func_idx >= S->libs[lib_idx].func_count) return "";
    return S->libs[lib_idx].funcs[func_idx].name;
}

int pion_lua_exec_function(PionLuaState *S, const char *func_name, int func_name_len) {
    if (!S || !S->L) return PION_LUA_ERROR;

    /* Find function across all libraries */
    int ref = LUA_NOREF;
    for (int i = 0; i < S->lib_count; i++) {
        for (int f = 0; f < S->libs[i].func_count; f++) {
            if (strncmp(S->libs[i].funcs[f].name, func_name, func_name_len) == 0 &&
                S->libs[i].funcs[f].name[func_name_len] == '\0') {
                ref = S->libs[i].funcs[f].ref;
                goto found;
            }
        }
    }
found:
    if (ref == LUA_NOREF) {
        snprintf(S->error_buf, sizeof(S->error_buf),
                 "ERR Function not found");
        return PION_LUA_ERROR;
    }

    /* Clean up previous coroutine (frees its garbage, GCs past half the cap) */
    release_coroutine(S);

    /* Create coroutine */
    S->co = lua_newthread(S->L);
    S->co_ref = luaL_ref(S->L, LUA_REGISTRYINDEX);
    S->pcall_mode = 0;

    lua_sethook(S->co, insn_hook, LUA_MASKCOUNT, S->insn_limit);

    /* Push the function */
    lua_rawgeti(S->L, LUA_REGISTRYINDEX, ref);
    lua_xmove(S->L, S->co, 1);

    /* Push KEYS and ARGV as arguments to the function (raw — a script may have
     * put a metatable on _G) */
    raw_getglobal(S->co, "KEYS");
    raw_getglobal(S->co, "ARGV");

    /* Resume with 2 arguments (keys, args) — memory cap in force */
    int status = run_script(S, 2);

    if (status == 0) {
        return PION_LUA_OK;
    } else if (status == LUA_YIELD) {
        lua_getfield(S->co, LUA_REGISTRYINDEX, "__pion_pcall");
        S->pcall_mode = lua_toboolean(S->co, -1);
        lua_pop(S->co, 1);
        lua_pushnil(S->co);
        lua_setfield(S->co, LUA_REGISTRYINDEX, "__pion_pcall");
        return PION_LUA_NEEDS_CMD;
    } else {
        const char *err = lua_tostring(S->co, -1);
        snprintf(S->error_buf, sizeof(S->error_buf), "%s", err ? err : "unknown error");
        return PION_LUA_ERROR;
    }
}

const char* pion_lua_call_arg_ptr(PionLuaState *S, int idx) {
    if (!S || !S->co || idx < 0 || idx >= lua_gettop(S->co)) return NULL;
    size_t len;
    return lua_tolstring(S->co, idx + 1, &len);
}

int pion_lua_call_arg_len(PionLuaState *S, int idx) {
    if (!S || !S->co || idx < 0 || idx >= lua_gettop(S->co)) return 0;
    size_t len;
    lua_tolstring(S->co, idx + 1, &len);
    return (int)len;
}
