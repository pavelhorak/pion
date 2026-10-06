/*
 * Pion ↔ Lua 5.1 bridge: EVAL, EVALSHA, their _RO forms, SCRIPT, FUNCTION and
 * FCALL, with the scripting environment Redis 7+ gives a script.
 *
 * redis.call() runs SYNCHRONOUSLY: it calls back into the host (Mojo's
 * `pion_script_dispatch`, an @export resolved with dlsym), which runs the
 * command through the server's own slow-path dispatcher and hands back its RESP
 * reply, converted here to Lua values. Every command, option, error and WAL
 * record is the server's own. The bridge used to yield a coroutine to a
 * separate 26-command dispatcher instead; Lua 5.1 cannot yield across pcall,
 * so a script could not even catch a redis.call() error (#36).
 *
 * Errors use Lua's longjmp only between C and Lua frames: by the time a
 * redis.call() error is raised, the host callback has returned.
 *
 * Two Lua states per worker, as in Redis: one for EVAL scripts, one for
 * FUNCTION libraries.
 */

#include "lua/lua.h"
#include "lua/lauxlib.h"
#include "lua/lualib.h"
#include "lua_wrap.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

extern int luaopen_cjson(lua_State *L);
extern int luaopen_struct(lua_State *L);
extern int luaopen_cmsgpack(lua_State *L);
extern int luaopen_bit(lua_State *L);

/* The version Pion reports in INFO (`redis_version`), which clients gate on. */
#define PION_REDIS_VERSION     "7.0.0"
#define PION_REDIS_VERSION_NUM 0x00070000

/* ── SHA1 (script hashing) ── */

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

static void sha1_hex_of(const char *s, size_t len, char hex[41]) {
    static const char hc[] = "0123456789abcdef";
    SHA1_CTX ctx; uint8_t digest[20];
    sha1_init(&ctx);
    sha1_update(&ctx, (const uint8_t *)s, len);
    sha1_final(&ctx, digest);
    for (int i = 0; i < 20; i++) {
        hex[i*2]   = hc[digest[i] >> 4];
        hex[i*2+1] = hc[digest[i] & 0xF];
    }
    hex[40] = '\0';
}

/* ── Memory-limited allocator ── */

/* The cap binds only while script code runs (`enforce`, set around every
 * protected call into user code). Host bookkeeping outside a protected call
 * — creating the states, registry refs — must never see a failed allocation:
 * there it is a panic, not a Lua error (gh #410). */
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
    if (nsize > osize && ctx->enforce && ctx->limit > 0 && ctx->used - osize + nsize > ctx->limit) {
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

/* An error outside every protected call is a bridge bug: say so and abort, so
 * the crash log records a signal and a backtrace. */
static int pion_lua_panic(lua_State *L) {
    const char *msg = lua_tostring(L, -1);
    fprintf(stderr, "[Lua] PANIC: unprotected error in the Lua bridge: %s\n",
            msg ? msg : "(no message)");
    fflush(stderr);
    abort();
    return 0;
}

/* ── State ── */

typedef struct {
    char sha[41];
    int  ref;              /* registry ref to the compiled chunk */
    int  flags;            /* shebang flags; FN_SHEBANG when it had one */
} CachedScript;

#define FN_NO_WRITES            (1 << 0)
#define FN_ALLOW_OOM            (1 << 1)
#define FN_ALLOW_STALE          (1 << 2)
#define FN_NO_CLUSTER           (1 << 3)
#define FN_ALLOW_CROSS_SLOT     (1 << 4)
#define FN_SHEBANG              (1 << 8)

typedef struct {
    char *name;
    char *desc;            /* NULL = none */
    int   ref;             /* registry ref (functions state) to the callback */
    int   flags;
} RegFunc;

typedef struct {
    char    *name;
    char    *code;
    size_t   code_len;
    RegFunc *fns;
    int      nfns;
} Library;

/* The host's command dispatcher (Mojo `pion_script_dispatch`): runs argv as a
 * command and returns its RESP reply (valid until the next call) in *out.
 * flags: PION_LUA_DISPATCH_RO refuses writes; PION_LUA_DISPATCH_CHECK only
 * answers whether the command exists (1/0). *wrote is set to 1 when a write
 * command ran. Returns the reply length, or -1. */
#define PION_LUA_DISPATCH_RO    1
#define PION_LUA_DISPATCH_CHECK 2
#define PION_LUA_DISPATCH_OOM   4   /* the function may run deny-oom commands */
typedef int64_t (*pion_dispatch_fn)(void *ctx, int64_t argc, const char **argv,
                                    const int64_t *lens, int64_t flags, int64_t resp,
                                    const char **out, int64_t *wrote);

struct PionLuaState {
    lua_State    *L;           /* EVAL scripts */
    lua_State    *FL;          /* FUNCTION libraries */
    int           fl_globals;  /* registry ref (FL): the real globals behind its proxy _G */
    LuaMemCtx     mem;
    LuaMemCtx     fmem;
    int64_t       time_limit_ms;

    CachedScript *scripts;
    int           nscripts, cap_scripts;

    Library      *libs;
    int           nlibs, cap_libs;
    Library      *loading;     /* the library being loaded (register_function) */

    void             *host;
    pion_dispatch_fn  dispatch;

    /* the run in progress */
    int           running;
    int           resp;        /* redis.setresp: 2 or 3 */
    int           ro;          /* writes refused */
    int           allow_oom;
    int           wrote;
    int           killed;
    struct timespec started;
    char          run_name[128];   /* sha, or function name: the error suffix */
    const char   *run_source;      /* "@user_script" / "@user_function" */

    /* argv scratch for redis.call */
    const char  **argv;
    int64_t      *lens;
    int           cap_argv;
    char        **numbufs;         /* number arguments converted to strings */
    int           cap_numbufs;

    /* the reply being built */
    char         *out;
    size_t        out_len, out_cap;
    int           out_oom;
};

/* ── Output buffer ── */

static void out_reset(PionLuaState *S) { S->out_len = 0; S->out_oom = 0; }

static void out_add(PionLuaState *S, const char *p, size_t n) {
    if (S->out_oom) return;
    if (S->out_len + n > S->out_cap) {
        size_t nc = S->out_cap ? S->out_cap : 4096;
        while (nc < S->out_len + n) nc *= 2;
        char *q = (char *)realloc(S->out, nc);
        if (!q) { S->out_oom = 1; return; }
        S->out = q; S->out_cap = nc;
    }
    memcpy(S->out + S->out_len, p, n);
    S->out_len += n;
}

static void out_str(PionLuaState *S, const char *s) { out_add(S, s, strlen(s)); }

static void out_hdr(PionLuaState *S, char type, long long n) {
    char b[32];
    int l = snprintf(b, sizeof(b), "%c%lld\r\n", type, n);
    out_add(S, b, (size_t)l);
}

static void out_bulk(PionLuaState *S, const char *p, size_t n) {
    out_hdr(S, '$', (long long)n);
    out_add(S, p, n);
    out_add(S, "\r\n", 2);
}

/* A simple-string or error line: CR and LF become spaces, as Redis does. */
static void out_line(PionLuaState *S, char type, const char *p, size_t n) {
    out_add(S, &type, 1);
    size_t start = S->out_len;
    out_add(S, p, n);
    if (!S->out_oom)
        for (size_t i = start; i < S->out_len; i++)
            if (S->out[i] == '\r' || S->out[i] == '\n') S->out[i] = ' ';
    out_add(S, "\r\n", 2);
}

static void out_null(PionLuaState *S, int resp) { out_str(S, resp == 3 ? "_\r\n" : "$-1\r\n"); }

/* An error reply of Pion's own (not a script's): "-<msg>\r\n". */
static int64_t out_error(PionLuaState *S, const char *msg) {
    out_reset(S);
    out_line(S, '-', msg, strlen(msg));
    return S->out_oom ? -1 : (int64_t)S->out_len;
}

const char *pion_lua_out(PionLuaState *S) { return S ? S->out : NULL; }

/* ── Doubles ── */

/* The shortest representation that reads back to the same double, as Redis's
 * fpconv_dtoa prints a number passed to redis.call() and d2string a double
 * reply: integral values print without an exponent, -0 as 0. */
static int fmt_double(double v, char *buf, size_t cap) {
    if (isnan(v)) return snprintf(buf, cap, "nan");
    if (isinf(v)) return snprintf(buf, cap, v > 0 ? "inf" : "-inf");
    if (v == 0) return snprintf(buf, cap, "0");
    if (v == floor(v) && fabs(v) < 1e17) return snprintf(buf, cap, "%.0f", v);
    for (int prec = 1; prec <= 17; prec++) {
        int l = snprintf(buf, cap, "%.*g", prec, v);
        if (strtod(buf, NULL) == v) return l;
    }
    return snprintf(buf, cap, "%.17g", v);
}

/* ── Lua helpers ── */

static PionLuaState *state_of(lua_State *L) {
    lua_getfield(L, LUA_REGISTRYINDEX, "__pion_state");
    PionLuaState *S = (PionLuaState *)lua_touserdata(L, -1);
    lua_pop(L, 1);
    return S;
}

/* Push {err="<code> <msg>"} the way Redis's luaPushErrorBuff builds it: a
 * message starting with '-' carries its own code ("-WRONGTYPE ..."), anything
 * else gets "ERR". A trailing CR/LF is trimmed. */
static void push_error_table(lua_State *L, const char *msg, size_t len) {
    luaL_Buffer b;
    while (len > 0 && (msg[len - 1] == '\r' || msg[len - 1] == '\n')) len--;
    luaL_buffinit(L, &b);
    if (len > 0 && msg[0] == '-') {
        const char *sp = memchr(msg, ' ', len);
        if (!sp) {
            luaL_addstring(&b, "ERR ");
            luaL_addlstring(&b, msg + 1, len - 1);
        } else {
            luaL_addlstring(&b, msg + 1, (size_t)(sp - msg - 1));
            luaL_addlstring(&b, sp, len - (size_t)(sp - msg));
        }
    } else {
        luaL_addstring(&b, "ERR ");
        luaL_addlstring(&b, msg, len);
    }
    lua_newtable(L);
    lua_pushliteral(L, "err");
    luaL_pushresult(&b);
    lua_rawset(L, -3);
}

/* Raise (call) or return (pcall) an error table. */
static int error_or_return(lua_State *L, int raise, const char *msg) {
    push_error_table(L, msg, strlen(msg));
    if (raise) return lua_error(L);
    return 1;
}

/* Raise {err="ERR <msg>"}: how Redis's redis.* functions fail (no position
 * prefix; the handler adds the script line). */
static int raise_err(lua_State *L, const char *msg) {
    push_error_table(L, msg, strlen(msg));
    return lua_error(L);
}

/* Lua's own tostring for an error object of any type. */
static void push_tostring(lua_State *L, int idx) {
    switch (lua_type(L, idx)) {
    case LUA_TNUMBER:
    case LUA_TSTRING:  lua_pushvalue(L, idx); lua_tostring(L, -1); break;
    case LUA_TBOOLEAN: lua_pushstring(L, lua_toboolean(L, idx) ? "true" : "false"); break;
    case LUA_TNIL:     lua_pushliteral(L, "nil"); break;
    default:
        lua_pushfstring(L, "%s: %p", luaL_typename(L, idx), lua_topointer(L, idx));
    }
}

/* ── RESP → Lua (a command's reply, as Redis's redisProtocolToLuaType) ── */

static int resp_line(const char *p, size_t n, size_t pos, size_t *end) {
    for (size_t i = pos; i + 1 < n; i++)
        if (p[i] == '\r' && p[i + 1] == '\n') { *end = i; return 1; }
    return 0;
}

static long long resp_int(const char *p, size_t a, size_t b) {
    long long v = 0; int neg = 0; size_t i = a;
    if (i < b && p[i] == '-') { neg = 1; i++; }
    for (; i < b; i++) v = v * 10 + (p[i] - '0');
    return neg ? -v : v;
}

/* Push the reply at *pos and advance past it. *is_err is set for a top-level
 * error reply. Returns 0 on a malformed reply. */
static int resp_to_lua(lua_State *L, const char *p, size_t n, size_t *pos, int *is_err, int depth) {
    size_t end;
    if (*pos >= n || !resp_line(p, n, *pos + 1, &end)) return 0;
    if (depth > 1000 || !lua_checkstack(L, 4)) return 0;
    char t = p[*pos];
    size_t a = *pos + 1;
    switch (t) {
    case '+':
        lua_newtable(L);
        lua_pushliteral(L, "ok");
        lua_pushlstring(L, p + a, end - a);
        lua_rawset(L, -3);
        *pos = end + 2;
        return 1;
    case '-':
        push_error_table(L, p + *pos, end - *pos);   /* keeps its code */
        if (is_err && depth == 0) *is_err = 1;
        *pos = end + 2;
        return 1;
    case ':':
        lua_pushnumber(L, (lua_Number)resp_int(p, a, end));
        *pos = end + 2;
        return 1;
    case '$': {
        long long len = resp_int(p, a, end);
        *pos = end + 2;
        if (len < 0) { lua_pushboolean(L, 0); return 1; }
        if (*pos + (size_t)len + 2 > n) return 0;
        lua_pushlstring(L, p + *pos, (size_t)len);
        *pos += (size_t)len + 2;
        return 1;
    }
    case '=': {   /* verbatim: {verbatim_string={format=, string=}} */
        long long len = resp_int(p, a, end);
        *pos = end + 2;
        if (len < 4 || *pos + (size_t)len + 2 > n) return 0;
        lua_newtable(L);
        lua_pushliteral(L, "verbatim_string");
        lua_newtable(L);
        lua_pushliteral(L, "format");
        lua_pushlstring(L, p + *pos, 3);
        lua_rawset(L, -3);
        lua_pushliteral(L, "string");
        lua_pushlstring(L, p + *pos + 4, (size_t)len - 4);
        lua_rawset(L, -3);
        lua_rawset(L, -3);
        *pos += (size_t)len + 2;
        return 1;
    }
    case '*':
    case '>': {
        long long cnt = resp_int(p, a, end);
        *pos = end + 2;
        if (cnt < 0) { lua_pushboolean(L, 0); return 1; }
        lua_newtable(L);
        for (long long j = 1; j <= cnt; j++) {
            if (!resp_to_lua(L, p, n, pos, NULL, depth + 1)) return 0;
            lua_rawseti(L, -2, (int)j);
        }
        return 1;
    }
    case '%': {
        long long cnt = resp_int(p, a, end);
        *pos = end + 2;
        lua_newtable(L);
        lua_pushliteral(L, "map");
        lua_newtable(L);
        for (long long j = 0; j < cnt; j++) {
            if (!resp_to_lua(L, p, n, pos, NULL, depth + 1)) return 0;
            if (!resp_to_lua(L, p, n, pos, NULL, depth + 1)) return 0;
            lua_rawset(L, -3);
        }
        lua_rawset(L, -3);
        return 1;
    }
    case '~': {
        long long cnt = resp_int(p, a, end);
        *pos = end + 2;
        lua_newtable(L);
        lua_pushliteral(L, "set");
        lua_newtable(L);
        for (long long j = 0; j < cnt; j++) {
            if (!resp_to_lua(L, p, n, pos, NULL, depth + 1)) return 0;
            lua_pushboolean(L, 1);
            lua_rawset(L, -3);
        }
        lua_rawset(L, -3);
        return 1;
    }
    case '_':
        lua_pushnil(L);
        *pos = end + 2;
        return 1;
    case '#':
        lua_pushboolean(L, end > a && p[a] == 't');
        *pos = end + 2;
        return 1;
    case ',': {
        char tmp[128];
        size_t l = end - a < sizeof(tmp) - 1 ? end - a : sizeof(tmp) - 1;
        memcpy(tmp, p + a, l); tmp[l] = 0;
        lua_newtable(L);
        lua_pushliteral(L, "double");
        lua_pushnumber(L, strtod(tmp, NULL));
        lua_rawset(L, -3);
        *pos = end + 2;
        return 1;
    }
    case '(':
        lua_newtable(L);
        lua_pushliteral(L, "big_number");
        lua_pushlstring(L, p + a, end - a);
        lua_rawset(L, -3);
        *pos = end + 2;
        return 1;
    case '|': {   /* attributes are dropped; the value follows */
        long long cnt = resp_int(p, a, end);
        *pos = end + 2;
        for (long long j = 0; j < 2 * cnt; j++) {
            if (!resp_to_lua(L, p, n, pos, NULL, depth + 1)) return 0;
            lua_pop(L, 1);
        }
        return resp_to_lua(L, p, n, pos, is_err, depth);
    }
    default:
        return 0;
    }
}

/* ── Lua → RESP (a script's return value, as Redis's luaReplyToRedisReply) ── */

static void reply_value(PionLuaState *S, lua_State *L, int resp, int depth);

/* The value on top of the stack, consumed. `resp` is the CLIENT's protocol;
 * S->resp is the script's (redis.setresp), which decides how a boolean
 * converts, as in Redis. */
static void reply_value(PionLuaState *S, lua_State *L, int resp, int depth) {
    int t = lua_type(L, -1);
    if (depth > 1000 || !lua_checkstack(L, 4)) {
        out_str(S, "-ERR reached lua stack limit\r\n");
        lua_pop(L, 1);
        return;
    }
    switch (t) {
    case LUA_TSTRING: {
        size_t l; const char *s = lua_tolstring(L, -1, &l);
        out_bulk(S, s, l);
        break;
    }
    case LUA_TBOOLEAN:
        if (S->resp == 2) {
            if (lua_toboolean(L, -1)) out_str(S, ":1\r\n"); else out_null(S, resp);
        } else if (resp == 2) {
            out_str(S, lua_toboolean(L, -1) ? ":1\r\n" : ":0\r\n");
        } else {
            out_str(S, lua_toboolean(L, -1) ? "#t\r\n" : "#f\r\n");
        }
        break;
    case LUA_TNUMBER:
        out_hdr(S, ':', (long long)lua_tonumber(L, -1));
        break;
    case LUA_TTABLE: {
        int tbl = lua_gettop(L);
        /* {err=string}: an error reply */
        lua_pushliteral(L, "err"); lua_rawget(L, tbl);
        if (lua_type(L, -1) == LUA_TSTRING) {
            size_t l; const char *s = lua_tolstring(L, -1, &l);
            if (l > 0 && s[0] == '-') { s++; l--; }
            out_line(S, '-', s, l);
            lua_pop(L, 2);
            return;
        }
        lua_pop(L, 1);
        /* {ok=string}: a status reply */
        lua_pushliteral(L, "ok"); lua_rawget(L, tbl);
        if (lua_type(L, -1) == LUA_TSTRING) {
            size_t l; const char *s = lua_tolstring(L, -1, &l);
            out_line(S, '+', s, l);
            lua_pop(L, 2);
            return;
        }
        lua_pop(L, 1);
        /* {double=number} */
        lua_pushliteral(L, "double"); lua_rawget(L, tbl);
        if (lua_type(L, -1) == LUA_TNUMBER) {
            char b[64]; int l = fmt_double(lua_tonumber(L, -1), b, sizeof(b));
            if (resp == 3) { out_line(S, ',', b, (size_t)l); }
            else out_bulk(S, b, (size_t)l);
            lua_pop(L, 2);
            return;
        }
        lua_pop(L, 1);
        /* {big_number=string} */
        lua_pushliteral(L, "big_number"); lua_rawget(L, tbl);
        if (lua_type(L, -1) == LUA_TSTRING) {
            size_t l; const char *s = lua_tolstring(L, -1, &l);
            if (resp == 3) out_line(S, '(', s, l); else out_bulk(S, s, l);
            lua_pop(L, 2);
            return;
        }
        lua_pop(L, 1);
        /* {verbatim_string={format=, string=}} */
        lua_pushliteral(L, "verbatim_string"); lua_rawget(L, tbl);
        if (lua_type(L, -1) == LUA_TTABLE) {
            int vt = lua_gettop(L);
            lua_pushliteral(L, "format"); lua_rawget(L, vt);
            lua_pushliteral(L, "string"); lua_rawget(L, vt);
            if (lua_type(L, -2) == LUA_TSTRING && lua_type(L, -1) == LUA_TSTRING) {
                size_t fl, sl;
                const char *f = lua_tolstring(L, -2, &fl);
                const char *s = lua_tolstring(L, -1, &sl);
                if (resp == 3) {
                    char fmt3[3] = {'t', 'x', 't'};
                    for (size_t k = 0; k < 3 && k < fl; k++) fmt3[k] = f[k];
                    out_hdr(S, '=', (long long)(sl + 4));
                    out_add(S, fmt3, 3);
                    out_add(S, ":", 1);
                    out_add(S, s, sl);
                    out_add(S, "\r\n", 2);
                } else {
                    out_bulk(S, s, sl);
                }
                lua_pop(L, 4);
                return;
            }
            lua_pop(L, 2);
        }
        lua_pop(L, 1);
        /* {map={...}} */
        lua_pushliteral(L, "map"); lua_rawget(L, tbl);
        if (lua_type(L, -1) == LUA_TTABLE) {
            int mt = lua_gettop(L);
            long long cnt = 0;
            lua_pushnil(L);
            while (lua_next(L, mt)) { cnt++; lua_pop(L, 1); }
            out_hdr(S, resp == 3 ? '%' : '*', resp == 3 ? cnt : 2 * cnt);
            lua_pushnil(L);
            while (lua_next(L, mt)) {
                lua_pushvalue(L, -2);
                reply_value(S, L, resp, depth + 1);   /* key */
                reply_value(S, L, resp, depth + 1);   /* value */
            }
            lua_pop(L, 2);
            return;
        }
        lua_pop(L, 1);
        /* {set={...}} */
        lua_pushliteral(L, "set"); lua_rawget(L, tbl);
        if (lua_type(L, -1) == LUA_TTABLE) {
            int st = lua_gettop(L);
            long long cnt = 0;
            lua_pushnil(L);
            while (lua_next(L, st)) { cnt++; lua_pop(L, 1); }
            out_hdr(S, resp == 3 ? '~' : '*', cnt);
            lua_pushnil(L);
            while (lua_next(L, st)) {
                lua_pop(L, 1);
                lua_pushvalue(L, -1);
                reply_value(S, L, resp, depth + 1);
            }
            lua_pop(L, 2);
            return;
        }
        lua_pop(L, 1);
        /* an array: elements 1.. up to the first nil */
        long long cnt = 0;
        for (;;) {
            lua_rawgeti(L, tbl, (int)(cnt + 1));
            int nil = lua_isnil(L, -1);
            lua_pop(L, 1);
            if (nil) break;
            cnt++;
        }
        out_hdr(S, '*', cnt);
        for (long long j = 1; j <= cnt; j++) {
            lua_rawgeti(L, tbl, (int)j);
            reply_value(S, L, resp, depth + 1);
        }
        break;
    }
    default:
        out_null(S, resp);
    }
    lua_pop(L, 1);
}

/* ── The time limit ── */

static int64_t elapsed_ms(PionLuaState *S) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t)(now.tv_sec - S->started.tv_sec) * 1000 +
           (now.tv_nsec - S->started.tv_nsec) / 1000000;
}

/* Every 100K instructions. A script that has run past lua-time-limit and has
 * not written anything is stopped, as SCRIPT KILL would stop it in Redis: a
 * worker runs one thing at a time, so it cannot answer SCRIPT KILL (or BUSY)
 * while a script runs. One that has written keeps running, as Redis keeps an
 * unkillable script running. */
static void time_hook(lua_State *L, lua_Debug *ar) {
    (void)ar;
    PionLuaState *S = state_of(L);
    if (!S || S->time_limit_ms <= 0 || S->wrote || S->killed) return;
    if (elapsed_ms(S) < S->time_limit_ms) return;
    S->killed = 1;
    char msg[160];
    snprintf(msg, sizeof(msg),
             "ERR Script killed: it ran longer than lua-time-limit (%lld ms) without writing",
             (long long)S->time_limit_ms);
    lua_newtable(L);
    lua_pushliteral(L, "err");
    lua_pushstring(L, msg);
    lua_rawset(L, -3);
    lua_error(L);
}

/* ── The error handler (Redis's __redis__err__handler) ── */

/* Turns any error into {err=..., source=..., line=...}: a non-table becomes
 * {err="ERR " .. tostring(e)}, and the position is the script line that
 * raised it (the caller of a C function such as redis.call or error). */
static int err_handler(lua_State *L) {
    lua_Debug ar;
    int have = 0;
    if (lua_getstack(L, 1, &ar)) {
        lua_getinfo(L, "Sl", &ar);
        have = 1;
        if (ar.what && strcmp(ar.what, "C") == 0) {
            have = lua_getstack(L, 2, &ar) ? (lua_getinfo(L, "Sl", &ar), 1) : 0;
        }
    }
    if (lua_type(L, 1) != LUA_TTABLE) {
        push_tostring(L, 1);
        lua_newtable(L);
        lua_pushliteral(L, "err");
        lua_pushliteral(L, "ERR ");
        lua_pushvalue(L, -4);
        lua_concat(L, 2);
        lua_rawset(L, -3);
        lua_replace(L, 1);
        lua_settop(L, 1);
    }
    if (have && ar.currentline > 0 && ar.source && ar.source[0] == '@' && !lua_isreadonlytable(L, 1)) {
        lua_pushliteral(L, "source");
        lua_pushstring(L, ar.source);
        lua_rawset(L, 1);
        lua_pushliteral(L, "line");
        lua_pushinteger(L, ar.currentline);
        lua_rawset(L, 1);
    }
    return 1;
}

/* The reply for a failed run: "-<err> script: <name>, on <source>:<line>." */
static int64_t reply_run_error(PionLuaState *S, lua_State *L) {
    out_reset(S);
    luaL_Buffer b;
    luaL_buffinit(L, &b);
    if (lua_type(L, -1) == LUA_TTABLE) {
        int et = lua_gettop(L);
        lua_pushliteral(L, "err"); lua_rawget(L, et);
        if (lua_type(L, -1) == LUA_TSTRING) {
            size_t l; const char *s = lua_tolstring(L, -1, &l);
            if (l > 0 && s[0] == '-') { s++; l--; }
            luaL_addlstring(&b, s, l);
        } else {
            luaL_addstring(&b, "ERR unknown error");
        }
        lua_pop(L, 1);
        lua_pushliteral(L, "source"); lua_rawget(L, et);
        lua_pushliteral(L, "line"); lua_rawget(L, et);
        if (lua_type(L, -2) == LUA_TSTRING && lua_isnumber(L, -1)) {
            char suf[256];
            snprintf(suf, sizeof(suf), " script: %s, on %s:%d.", S->run_name,
                     lua_tostring(L, -2), (int)lua_tointeger(L, -1));
            lua_pop(L, 2);
            luaL_addstring(&b, suf);
        } else {
            lua_pop(L, 2);
        }
    } else {
        /* not through the handler (a memory error while raising one) */
        push_tostring(L, -1);
        luaL_addstring(&b, "ERR ");
        luaL_addvalue(&b);
    }
    luaL_pushresult(&b);
    size_t l; const char *m = lua_tolstring(L, -1, &l);
    out_line(S, '-', m, l);
    lua_pop(L, 1);
    return S->out_oom ? -1 : (int64_t)S->out_len;
}

/* ── redis.* ── */

static int redis_call_generic(lua_State *L, int raise) {
    PionLuaState *S = state_of(L);
    int argc = lua_gettop(L);
    if (argc == 0)
        return error_or_return(L, raise, "Please specify at least one argument for this redis lib call");
    if (argc > S->cap_argv) {
        int nc = argc < 16 ? 16 : argc * 2;
        const char **na = (const char **)realloc((void *)S->argv, sizeof(char *) * (size_t)nc);
        if (na) S->argv = na;
        int64_t *nl = (int64_t *)realloc(S->lens, sizeof(int64_t) * (size_t)nc);
        if (nl) S->lens = nl;
        if (!na || !nl) return luaL_error(L, "not enough memory");
        S->cap_argv = nc;
    }
    for (int j = 1; j <= argc; j++) {
        int t = lua_type(L, j);
        if (t == LUA_TNUMBER) {
            /* Redis's fpconv_dtoa: the shortest form that reads back */
            char nb[64];
            int l = fmt_double(lua_tonumber(L, j), nb, sizeof(nb));
            lua_pushlstring(L, nb, (size_t)l);
            lua_replace(L, j);
        } else if (t != LUA_TSTRING) {
            return error_or_return(L, raise, "Lua redis lib command arguments must be strings or integers");
        }
        size_t l;
        S->argv[j - 1] = lua_tolstring(L, j, &l);
        S->lens[j - 1] = (int64_t)l;
    }
    if (!S->dispatch)
        return error_or_return(L, raise, "This Redis command is not allowed from script");
    const char *rep = NULL;
    int64_t wrote = 0;
    int64_t flags = (S->ro ? PION_LUA_DISPATCH_RO : 0) | (S->allow_oom ? PION_LUA_DISPATCH_OOM : 0);
    int64_t n = S->dispatch(S->host, argc, S->argv, S->lens, flags, S->resp, &rep, &wrote);
    if (wrote) S->wrote = 1;
    if (n < 0 || !rep)
        return error_or_return(L, raise, "internal error: the command produced no reply");
    size_t pos = 0;
    int is_err = 0;
    lua_settop(L, 0);
    if (!resp_to_lua(L, rep, (size_t)n, &pos, &is_err, 0))
        return error_or_return(L, raise, "internal error: the command's reply could not be read");
    if (is_err && raise) return lua_error(L);
    return 1;
}

static int lua_redis_call(lua_State *L)  { return redis_call_generic(L, 1); }
static int lua_redis_pcall(lua_State *L) { return redis_call_generic(L, 0); }

/* Redis's pcall: an error table raised by redis.call() comes back as its
 * message string, as scripts written for Redis < 7 expect. */
static int lua_redis_lua_pcall(lua_State *L) {
    int argc = lua_gettop(L);
    luaL_checkany(L, 1);
    lua_pushboolean(L, 1);
    lua_insert(L, 1);
    if (lua_pcall(L, argc - 1, LUA_MULTRET, 0)) {
        lua_remove(L, 1);
        if (lua_istable(L, -1)) {
            lua_pushliteral(L, "err");
            lua_rawget(L, -2);
            if (lua_isstring(L, -1)) lua_replace(L, -2);
            else lua_pop(L, 1);
        }
        lua_pushboolean(L, 0);
        lua_insert(L, 1);
    }
    return lua_gettop(L);
}

static int lua_redis_log(lua_State *L) {
    int argc = lua_gettop(L);
    if (argc < 2) return raise_err(L, "redis.log() requires two arguments or more.");
    if (!lua_isnumber(L, 1)) return raise_err(L, "First argument must be a number (log level).");
    int level = (int)lua_tonumber(L, 1);
    if (level < 0 || level > 3) return raise_err(L, "Invalid log level.");
    luaL_Buffer b;
    luaL_buffinit(L, &b);
    for (int j = 2; j <= argc; j++) {
        size_t l; const char *s = lua_tolstring(L, j, &l);
        if (s) {
            if (j != 2) luaL_addchar(&b, ' ');
            luaL_addlstring(&b, s, l);
        }
    }
    luaL_pushresult(&b);
    if (level >= 2) fprintf(stderr, "[Lua] %s\n", lua_tostring(L, -1));
    return 0;
}

/* redis.error_reply / status_reply: a table the script returns. */
static int lua_redis_error_reply(lua_State *L) {
    if (lua_gettop(L) != 1 || lua_type(L, -1) != LUA_TSTRING) {
        push_error_table(L, "wrong number or type of arguments", strlen("wrong number or type of arguments"));
        return 1;
    }
    size_t l; const char *s = lua_tolstring(L, 1, &l);
    if (l > 0 && s[0] == '-') {
        push_error_table(L, s, l);
    } else {
        lua_pushliteral(L, "-");
        lua_pushvalue(L, 1);
        lua_concat(L, 2);
        size_t l2; const char *s2 = lua_tolstring(L, -1, &l2);
        push_error_table(L, s2, l2);
    }
    return 1;
}

static int lua_redis_status_reply(lua_State *L) {
    if (lua_gettop(L) != 1 || lua_type(L, -1) != LUA_TSTRING) {
        push_error_table(L, "wrong number or type of arguments", strlen("wrong number or type of arguments"));
        return 1;
    }
    lua_newtable(L);
    lua_pushliteral(L, "ok");
    lua_pushvalue(L, 1);
    lua_rawset(L, -3);
    return 1;
}

static int lua_redis_sha1hex(lua_State *L) {
    if (lua_gettop(L) != 1) return raise_err(L, "wrong number of arguments");
    size_t len;
    const char *s = lua_tolstring(L, 1, &len);
    char hex[41];
    sha1_hex_of(s ? s : "", s ? len : 0, hex);
    lua_pushlstring(L, hex, 40);
    return 1;
}

static int lua_redis_setresp(lua_State *L) {
    PionLuaState *S = state_of(L);
    if (lua_gettop(L) != 1) return raise_err(L, "redis.setresp() requires one argument.");
    int v = (int)lua_tonumber(L, 1);
    if (v != 2 && v != 3) return raise_err(L, "RESP version must be 2 or 3.");
    S->resp = v;
    return 0;
}

static int lua_redis_set_repl(lua_State *L) {
    if (lua_gettop(L) != 1) return raise_err(L, "redis.set_repl() requires one argument.");
    int v = (int)lua_tonumber(L, 1);
    if (v < 0 || v > 3)
        return raise_err(L, "Invalid replication flags. Use REPL_AOF, REPL_REPLICA, REPL_ALL or REPL_NONE.");
    return 0;
}

static int lua_redis_true(lua_State *L)  { lua_pushboolean(L, 1); return 1; }
static int lua_redis_false(lua_State *L) { lua_pushboolean(L, 0); return 1; }
static int lua_redis_nil(lua_State *L)   { (void)L; return 0; }

static int lua_redis_acl_check_cmd(lua_State *L) {
    PionLuaState *S = state_of(L);
    int argc = lua_gettop(L);
    if (argc == 0) return raise_err(L, "Please specify at least one argument for this redis lib call");
    size_t l; const char *name = lua_tolstring(L, 1, &l);
    int exists = 0;
    if (name && S->dispatch) {
        const char *argv[1] = {name};
        int64_t lens[1] = {(int64_t)l};
        const char *rep = NULL; int64_t wrote = 0;
        exists = S->dispatch(S->host, 1, argv, lens, PION_LUA_DISPATCH_CHECK, 2, &rep, &wrote) == 1;
    }
    if (!exists) return raise_err(L, "Invalid command passed to redis.acl_check_cmd()");
    lua_pushboolean(L, 1);
    return 1;
}

static int lua_redis_register_function(lua_State *L);

/* The `redis` table: the full API for scripts and functions, or (`load`) the
 * part FUNCTION LOAD gives library code. */
static void push_redis_table(lua_State *L, int load) {
    lua_newtable(L);
    if (!load) {
        lua_pushcfunction(L, lua_redis_call);           lua_setfield(L, -2, "call");
        lua_pushcfunction(L, lua_redis_pcall);          lua_setfield(L, -2, "pcall");
        lua_pushcfunction(L, lua_redis_sha1hex);        lua_setfield(L, -2, "sha1hex");
        lua_pushcfunction(L, lua_redis_error_reply);    lua_setfield(L, -2, "error_reply");
        lua_pushcfunction(L, lua_redis_status_reply);   lua_setfield(L, -2, "status_reply");
        lua_pushcfunction(L, lua_redis_set_repl);       lua_setfield(L, -2, "set_repl");
        lua_pushcfunction(L, lua_redis_true);           lua_setfield(L, -2, "replicate_commands");
        lua_pushcfunction(L, lua_redis_false);          lua_setfield(L, -2, "breakpoint");
        lua_pushcfunction(L, lua_redis_nil);            lua_setfield(L, -2, "debug");
        lua_pushcfunction(L, lua_redis_acl_check_cmd);  lua_setfield(L, -2, "acl_check_cmd");
        lua_pushinteger(L, 3); lua_setfield(L, -2, "REPL_ALL");
        lua_pushinteger(L, 1); lua_setfield(L, -2, "REPL_AOF");
        lua_pushinteger(L, 2); lua_setfield(L, -2, "REPL_SLAVE");
        lua_pushinteger(L, 2); lua_setfield(L, -2, "REPL_REPLICA");
        lua_pushinteger(L, 0); lua_setfield(L, -2, "REPL_NONE");
    } else {
        lua_pushcfunction(L, lua_redis_register_function); lua_setfield(L, -2, "register_function");
    }
    lua_pushcfunction(L, lua_redis_log);     lua_setfield(L, -2, "log");
    if (!load) { lua_pushcfunction(L, lua_redis_setresp); lua_setfield(L, -2, "setresp"); }
    lua_pushinteger(L, 0); lua_setfield(L, -2, "LOG_DEBUG");
    lua_pushinteger(L, 1); lua_setfield(L, -2, "LOG_VERBOSE");
    lua_pushinteger(L, 2); lua_setfield(L, -2, "LOG_NOTICE");
    lua_pushinteger(L, 3); lua_setfield(L, -2, "LOG_WARNING");
    lua_pushstring(L, PION_REDIS_VERSION);   lua_setfield(L, -2, "REDIS_VERSION");
    lua_pushinteger(L, PION_REDIS_VERSION_NUM); lua_setfield(L, -2, "REDIS_VERSION_NUM");
}

/* ── The sandbox ── */

/* Redis's error for reading an undefined global. */
static int protected_global_index(lua_State *L) {
    if (lua_gettop(L) != 2) return luaL_error(L, "Wrong number of arguments to luaProtectedTableError");
    if (!lua_isstring(L, -1) && !lua_isnumber(L, -1))
        return luaL_error(L, "Second argument to luaProtectedTableError must be a string or number");
    return luaL_error(L, "Script attempted to access nonexistent global variable '%s'", lua_tostring(L, -1));
}

static void set_error_metatable(lua_State *L, int idx) {
    idx = idx < 0 ? lua_gettop(L) + idx + 1 : idx;
    lua_newtable(L);
    lua_pushcfunction(L, protected_global_index);
    lua_setfield(L, -2, "__index");
    lua_setmetatable(L, idx);
}

/* Make the table at idx, and every table reachable from it, readonly. */
static void protect_recursively(lua_State *L, int idx) {
    idx = idx < 0 ? lua_gettop(L) + idx + 1 : idx;
    if (!lua_checkstack(L, 4) || lua_isreadonlytable(L, idx)) return;
    lua_enablereadonlytable(L, idx, 1);
    if (lua_getmetatable(L, idx)) {
        protect_recursively(L, -1);
        lua_pop(L, 1);
    }
    lua_pushnil(L);
    while (lua_next(L, idx)) {
        if (lua_istable(L, -1)) protect_recursively(L, -1);
        lua_pop(L, 1);
    }
}

static void open_lib(lua_State *L, const char *name, lua_CFunction f) {
    lua_pushcfunction(L, f);
    lua_pushstring(L, name);
    lua_call(L, 1, 0);
}

static void remove_global(lua_State *L, const char *name) {
    lua_pushnil(L);
    lua_setglobal(L, name);
}

/* The environment Redis 7+ gives a script (the same globals, as probed against
 * redis-server 8.10): base minus print, dofile, loadfile, getfenv, setfenv
 * and newproxy; table, string, math, coroutine; os with only clock; cjson,
 * struct, cmsgpack and bit; the redis table; Redis's pcall; then everything
 * readonly, and reading an undefined global an error. */
static lua_State *new_sandbox(LuaMemCtx *mem, PionLuaState *S, int functions) {
    lua_State *L = lua_newstate(lua_mem_alloc, mem);
    if (!L) return NULL;
    lua_atpanic(L, pion_lua_panic);

    open_lib(L, "", luaopen_base);
    open_lib(L, LUA_TABLIBNAME, luaopen_table);
    open_lib(L, LUA_STRLIBNAME, luaopen_string);
    open_lib(L, LUA_MATHLIBNAME, luaopen_math);
    open_lib(L, LUA_OSLIBNAME, luaopen_os);
    open_lib(L, "cjson", luaopen_cjson);
    open_lib(L, "struct", luaopen_struct);
    open_lib(L, "cmsgpack", luaopen_cmsgpack);
    open_lib(L, "bit", luaopen_bit);
    /* luaopen_cjson returns its table without setting a global */
    lua_getglobal(L, "cjson");
    if (lua_isnil(L, -1)) {
        lua_pop(L, 1);
        lua_pushcfunction(L, luaopen_cjson);
        lua_call(L, 0, 1);
        lua_setglobal(L, "cjson");
    } else {
        lua_pop(L, 1);
    }
    lua_getglobal(L, "cmsgpack");
    if (lua_isnil(L, -1)) {
        lua_pop(L, 1);
        lua_pushcfunction(L, luaopen_cmsgpack);
        lua_call(L, 0, 1);
        lua_setglobal(L, "cmsgpack");
    } else {
        lua_pop(L, 1);
    }

    remove_global(L, "print");
    remove_global(L, "dofile");
    remove_global(L, "loadfile");
    remove_global(L, "getfenv");
    remove_global(L, "setfenv");
    remove_global(L, "newproxy");
    remove_global(L, "module");
    remove_global(L, "require");
    remove_global(L, "package");

    /* os: clock only */
    lua_getglobal(L, "os");
    lua_getfield(L, -1, "clock");
    lua_newtable(L);
    lua_insert(L, -2);
    lua_setfield(L, -2, "clock");
    lua_setglobal(L, "os");
    lua_pop(L, 1);

    lua_pushcfunction(L, lua_redis_lua_pcall);
    lua_setglobal(L, "pcall");

    push_redis_table(L, 0);
    lua_setglobal(L, "redis");

    /* the handler Redis exposes as a global; ours is C, this keeps _G's shape */
    lua_pushcfunction(L, err_handler);
    lua_setglobal(L, "__redis__err__handler");

    if (!functions) {
        lua_newtable(L); lua_setglobal(L, "KEYS");
        lua_newtable(L); lua_setglobal(L, "ARGV");
    }

    lua_pushlightuserdata(L, S);
    lua_setfield(L, LUA_REGISTRYINDEX, "__pion_state");

    /* readonly: _G and everything reachable from it, the string metatable too */
    lua_pushvalue(L, LUA_GLOBALSINDEX);
    set_error_metatable(L, -1);
    protect_recursively(L, -1);
    lua_pop(L, 1);
    lua_pushliteral(L, "");
    if (lua_getmetatable(L, -1)) {
        protect_recursively(L, -1);
        lua_pop(L, 1);
    }
    lua_pop(L, 1);

    if (functions) {
        /* Redis's functions engine: _G is an empty proxy whose __index is the
         * real globals when a function runs, and only the library API while
         * FUNCTION LOAD runs the library body (see pion_lua_function_load). */
        lua_pushvalue(L, LUA_GLOBALSINDEX);
        S->fl_globals = luaL_ref(L, LUA_REGISTRYINDEX);
        lua_newtable(L);
        lua_newtable(L);
        lua_pushvalue(L, LUA_GLOBALSINDEX);
        lua_setfield(L, -2, "__index");
        lua_setmetatable(L, -2);
        lua_enablereadonlytable(L, -1, 1);
        lua_replace(L, LUA_GLOBALSINDEX);
    }
    return L;
}

/* ── Lifecycle ── */

/* --lua-memory-limit / --lua-time-limit, set once by main before the workers
 * start; a state created with a negative limit takes these. */
static int64_t g_mem_limit = (int64_t)1 << 30;
static int64_t g_time_limit_ms = 5000;

/* This worker thread's state: WAL replay, the snapshot writer and replication
 * apply function records through it (they run on the worker's own thread). */
static __thread PionLuaState *tls_lua = NULL;

void pion_lua_set_defaults(int64_t mem_limit, int64_t time_limit_ms) {
    g_mem_limit = mem_limit;
    g_time_limit_ms = time_limit_ms;
}

PionLuaState *pion_lua_new_state(int64_t mem_limit, int64_t time_limit_ms) {
    PionLuaState *S = (PionLuaState *)calloc(1, sizeof(PionLuaState));
    if (!S) return NULL;
    if (mem_limit < 0) mem_limit = g_mem_limit;
    if (time_limit_ms < 0) time_limit_ms = g_time_limit_ms;
    S->mem.limit = mem_limit > 0 ? (size_t)mem_limit : 0;
    S->fmem.limit = S->mem.limit;
    S->time_limit_ms = time_limit_ms;
    S->resp = 2;
    S->L = new_sandbox(&S->mem, S, 0);
    S->FL = new_sandbox(&S->fmem, S, 1);
    if (!S->L || !S->FL) {
        if (S->L) lua_close(S->L);
        if (S->FL) lua_close(S->FL);
        free(S);
        return NULL;
    }
    S->dispatch = (pion_dispatch_fn)dlsym(RTLD_DEFAULT, "pion_script_dispatch");
    tls_lua = S;
    return S;
}

void pion_lua_set_host(PionLuaState *S, void *host) {
    if (S) S->host = host;
}

int64_t pion_lua_memory(PionLuaState *S) {
    return S ? (int64_t)(S->mem.used + S->fmem.used) : 0;
}

static void free_library(PionLuaState *S, Library *lib) {
    for (int f = 0; f < lib->nfns; f++) {
        luaL_unref(S->FL, LUA_REGISTRYINDEX, lib->fns[f].ref);
        free(lib->fns[f].name);
        free(lib->fns[f].desc);
    }
    free(lib->fns);
    free(lib->name);
    free(lib->code);
    memset(lib, 0, sizeof(*lib));
}

void pion_lua_close(PionLuaState *S) {
    if (!S) return;
    if (tls_lua == S) tls_lua = NULL;
    for (int i = 0; i < S->nlibs; i++) free_library(S, &S->libs[i]);
    free(S->libs);
    free(S->scripts);
    if (S->L) lua_close(S->L);
    if (S->FL) lua_close(S->FL);
    free((void *)S->argv);
    free(S->lens);
    free(S->out);
    free(S);
}

/* ── Running user code ── */

/* Set KEYS and ARGV (globals of the EVAL state, written past its readonly
 * protection), or push them as the function's two arguments. */
static void push_array(lua_State *L, const char **p, const int64_t *l, int64_t n) {
    lua_createtable(L, (int)n, 0);
    for (int64_t j = 0; j < n; j++) {
        lua_pushlstring(L, p[j], (size_t)l[j]);
        lua_rawseti(L, -2, (int)(j + 1));
    }
}

typedef struct {
    PionLuaState *S;
    int ref;                      /* the function to run */
    int as_function;              /* FCALL: (keys, args) as arguments */
    const char **keys; const int64_t *key_lens; int64_t nkeys;
    const char **args; const int64_t *arg_lens; int64_t nargs;
} RunCtx;

/* Inside a protected call, so a memory error building KEYS/ARGV is a script
 * error, not a panic. */
static int run_trampoline(lua_State *L) {
    RunCtx *rc = (RunCtx *)lua_touserdata(L, 1);
    lua_settop(L, 0);
    lua_pushcfunction(L, err_handler);
    lua_rawgeti(L, LUA_REGISTRYINDEX, rc->ref);
    if (rc->as_function) {
        push_array(L, rc->keys, rc->key_lens, rc->nkeys);
        push_array(L, rc->args, rc->arg_lens, rc->nargs);
    } else {
        lua_pushvalue(L, LUA_GLOBALSINDEX);
        lua_enablereadonlytable(L, -1, 0);
        lua_pushliteral(L, "KEYS");
        push_array(L, rc->keys, rc->key_lens, rc->nkeys);
        lua_rawset(L, -3);
        lua_pushliteral(L, "ARGV");
        push_array(L, rc->args, rc->arg_lens, rc->nargs);
        lua_rawset(L, -3);
        lua_enablereadonlytable(L, -1, 1);
        lua_pop(L, 1);
    }
    int status = lua_pcall(L, rc->as_function ? 2 : 0, 1, 1);
    lua_pushboolean(L, status == 0);
    return 2;   /* result or error object, then the success flag */
}

static int64_t run_user(PionLuaState *S, lua_State *L, LuaMemCtx *mem, RunCtx *rc, int64_t client_resp) {
    out_reset(S);
    S->running = 1;
    S->resp = 2;
    S->wrote = 0;
    S->killed = 0;
    clock_gettime(CLOCK_MONOTONIC, &S->started);
    lua_sethook(L, time_hook, LUA_MASKCOUNT, 100000);
    mem->enforce = 1;
    int top = lua_gettop(L);
    lua_pushcfunction(L, run_trampoline);
    lua_pushlightuserdata(L, rc);
    int status = lua_pcall(L, 1, 2, 0);
    mem->enforce = 0;
    lua_sethook(L, NULL, 0, 0);
    int64_t n;
    if (status != 0) {
        /* the trampoline itself failed (memory while setting up) */
        n = reply_run_error(S, L);
    } else if (lua_toboolean(L, -1)) {
        lua_pop(L, 1);
        reply_value(S, L, (int)client_resp, 0);
        n = S->out_oom ? -1 : (int64_t)S->out_len;
    } else {
        lua_pop(L, 1);
        n = reply_run_error(S, L);
    }
    lua_settop(L, top);
    S->running = 0;
    S->resp = 2;
    /* A script that ran into the cap leaves garbage; Lua 5.1 has no emergency
     * collection, so collect fully past half the cap. */
    if (mem->limit > 0 && mem->used > mem->limit / 2) lua_gc(L, LUA_GCCOLLECT, 0);
    return n;
}

/* ── Scripts (EVAL) ── */

static CachedScript *find_script(PionLuaState *S, const char *sha) {
    for (int i = 0; i < S->nscripts; i++)
        if (strncasecmp(S->scripts[i].sha, sha, 40) == 0) return &S->scripts[i];
    return NULL;
}

/* Strip a `#!lua [flags=...]` line; returns the flags, or -1 (with an error
 * reply in S->out) when the shebang names another engine or an unknown flag. */
static int script_shebang(PionLuaState *S, const char *code, size_t len, size_t *body_off) {
    *body_off = 0;
    if (len < 2 || code[0] != '#' || code[1] != '!') return 0;
    size_t eol = 0;
    while (eol < len && code[eol] != '\n') eol++;
    const char *p = code + 2, *e = code + eol;
    if (e - p < 3 || strncmp(p, "lua", 3) != 0 || (e - p > 3 && p[3] != ' ' && p[3] != '\r')) {
        char msg[200];
        size_t nl = 0;
        while (code + nl < e && code[nl] != ' ' && code[nl] != '\r') nl++;
        snprintf(msg, sizeof(msg), "ERR Unexpected engine in script shebang: %.*s", (int)nl, code);
        out_error(S, msg);
        return -1;
    }
    p += 3;
    int flags = 0;
    while (p < e) {
        while (p < e && (*p == ' ' || *p == '\r')) p++;
        if (p >= e) break;
        const char *w = p;
        while (p < e && *p != ' ' && *p != '\r') p++;
        size_t wl = (size_t)(p - w);
        if (wl >= 6 && strncmp(w, "flags=", 6) == 0) {
            const char *f = w + 6, *fe = w + wl;
            while (f < fe) {
                const char *c = f;
                while (f < fe && *f != ',') f++;
                size_t cl = (size_t)(f - c);
                if (cl == 9 && strncmp(c, "no-writes", 9) == 0) flags |= FN_NO_WRITES;
                else if (cl == 9 && strncmp(c, "allow-oom", 9) == 0) flags |= FN_ALLOW_OOM;
                else if (cl == 11 && strncmp(c, "allow-stale", 11) == 0) flags |= FN_ALLOW_STALE;
                else if (cl == 10 && strncmp(c, "no-cluster", 10) == 0) flags |= FN_NO_CLUSTER;
                else if (cl == 21 && strncmp(c, "allow-cross-slot-keys", 21) == 0) flags |= FN_ALLOW_CROSS_SLOT;
                else if (cl > 0) {
                    char msg[200];
                    snprintf(msg, sizeof(msg), "ERR Unexpected flag in script shebang: %.*s", (int)cl, c);
                    out_error(S, msg);
                    return -1;
                }
                if (f < fe) f++;
            }
        } else {
            char msg[200];
            snprintf(msg, sizeof(msg), "ERR Unknown lua shebang option: %.*s", (int)wl, w);
            out_error(S, msg);
            return -1;
        }
    }
    *body_off = eol;   /* keep the newline: line numbers stay the source's */
    return flags | FN_SHEBANG;
}

/* SCRIPT LOAD / EVAL: compile and cache. 0 = cached (sha in out_sha), -1 = an
 * error reply in S->out. */
int pion_lua_load_script(PionLuaState *S, const char *code, int64_t len, char *out_sha) {
    if (!S || !S->L) return -1;
    sha1_hex_of(code, (size_t)len, out_sha);
    if (find_script(S, out_sha)) return 0;
    size_t off = 0;
    int flags = script_shebang(S, code, (size_t)len, &off);
    if (flags < 0) return -1;
    S->mem.enforce = 1;
    int st = luaL_loadbuffer(S->L, code + off, (size_t)len - off, "@user_script");
    S->mem.enforce = 0;
    if (st != 0) {
        const char *m = lua_tostring(S->L, -1);
        char msg[1200];
        snprintf(msg, sizeof(msg), "ERR Error compiling script (new function): %s", m ? m : "?");
        lua_pop(S->L, 1);
        out_error(S, msg);
        return -1;
    }
    if (S->nscripts == S->cap_scripts) {
        int nc = S->cap_scripts ? S->cap_scripts * 2 : 64;
        CachedScript *ns = (CachedScript *)realloc(S->scripts, sizeof(CachedScript) * (size_t)nc);
        if (!ns) { lua_pop(S->L, 1); out_error(S, "ERR out of memory"); return -1; }
        S->scripts = ns; S->cap_scripts = nc;
    }
    CachedScript *cs = &S->scripts[S->nscripts++];
    memcpy(cs->sha, out_sha, 41);
    cs->flags = flags;
    cs->ref = luaL_ref(S->L, LUA_REGISTRYINDEX);
    return 0;
}

int pion_lua_script_exists(PionLuaState *S, const char *sha) {
    return (S && find_script(S, sha)) ? 1 : 0;
}

void pion_lua_script_flush(PionLuaState *S) {
    if (!S || !S->L) return;
    for (int i = 0; i < S->nscripts; i++) luaL_unref(S->L, LUA_REGISTRYINDEX, S->scripts[i].ref);
    S->nscripts = 0;
    lua_gc(S->L, LUA_GCCOLLECT, 0);
}

/* EVAL / EVALSHA and their _RO forms: run a cached script. The reply (the
 * script's result, or its error) is in pion_lua_out; returns its length, or
 * -1 when no reply could be built. */
int64_t pion_lua_run_script(PionLuaState *S, const char *sha,
                            const char **keys, const int64_t *key_lens, int64_t nkeys,
                            const char **args, const int64_t *arg_lens, int64_t nargs,
                            int64_t ro, int64_t client_resp) {
    if (!S || !S->L) return -1;
    CachedScript *cs = find_script(S, sha);
    if (!cs) return out_error(S, "NOSCRIPT No matching script. Please use EVAL.");
    if (ro && (cs->flags & FN_SHEBANG) && !(cs->flags & FN_NO_WRITES))
        return out_error(S, "ERR Can not execute a script with write flag using *_ro command.");
    memcpy(S->run_name, cs->sha, 41);
    S->run_source = "@user_script";
    S->ro = (ro || (cs->flags & FN_NO_WRITES)) ? 1 : 0;
    S->allow_oom = (cs->flags & FN_ALLOW_OOM) ? 1 : 0;
    RunCtx rc = {S, cs->ref, 0, keys, key_lens, nkeys, args, arg_lens, nargs};
    return run_user(S, S->L, &S->mem, &rc, client_resp);
}

/* ── Functions ── */

static RegFunc *find_function(PionLuaState *S, const char *name, size_t nlen, Library **lib) {
    for (int i = 0; i < S->nlibs; i++)
        for (int f = 0; f < S->libs[i].nfns; f++) {
            RegFunc *rf = &S->libs[i].fns[f];
            if (strlen(rf->name) == nlen && memcmp(rf->name, name, nlen) == 0) {
                if (lib) *lib = &S->libs[i];
                return rf;
            }
        }
    return NULL;
}

static int valid_name(const char *s, size_t n) {
    if (n == 0) return 0;
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'))
            return 0;
    }
    return 1;
}

/* redis.register_function(name, callback) or
 * redis.register_function{function_name=, callback=, flags=, description=} */
static int lua_redis_register_function(lua_State *L) {
    PionLuaState *S = state_of(L);
    if (!S || !S->loading)
        return raise_err(L, "redis.register_function can only be called on FUNCTION LOAD command");
    Library *lib = S->loading;
    const char *name = NULL; size_t nlen = 0;
    const char *desc = NULL; size_t dlen = 0;
    int flags = 0;
    int cb = 0;
    int argc = lua_gettop(L);
    if (argc == 1) {
        if (!lua_istable(L, 1))
            return raise_err(L, "calling redis.register_function with a single argument is only applicable to Lua table (representing named arguments).");
        lua_pushnil(L);
        while (lua_next(L, 1)) {
            if (lua_type(L, -2) != LUA_TSTRING)
                return raise_err(L, "named argument key given to redis.register_function is not a string");
            const char *k = lua_tostring(L, -2);
            if (strcasecmp(k, "function_name") == 0) {
                if (lua_type(L, -1) != LUA_TSTRING)
                    return raise_err(L, "function_name argument given to redis.register_function must be a string");
                name = lua_tolstring(L, -1, &nlen);
            } else if (strcasecmp(k, "description") == 0) {
                if (lua_type(L, -1) != LUA_TSTRING)
                    return raise_err(L, "description argument given to redis.register_function must be a string");
                desc = lua_tolstring(L, -1, &dlen);
            } else if (strcasecmp(k, "callback") == 0) {
                if (!lua_isfunction(L, -1))
                    return raise_err(L, "callback argument given to redis.register_function must be a function");
            } else if (strcasecmp(k, "flags") == 0) {
                if (!lua_istable(L, -1))
                    return raise_err(L, "flags argument to redis.register_function must be a table representing function flags");
                int ft = lua_gettop(L);
                for (int j = 1;; j++) {
                    lua_rawgeti(L, ft, j);
                    if (lua_isnil(L, -1)) { lua_pop(L, 1); break; }
                    const char *fs = lua_type(L, -1) == LUA_TSTRING ? lua_tostring(L, -1) : NULL;
                    if (fs && strcmp(fs, "no-writes") == 0) flags |= FN_NO_WRITES;
                    else if (fs && strcmp(fs, "allow-oom") == 0) flags |= FN_ALLOW_OOM;
                    else if (fs && strcmp(fs, "allow-stale") == 0) flags |= FN_ALLOW_STALE;
                    else if (fs && strcmp(fs, "no-cluster") == 0) flags |= FN_NO_CLUSTER;
                    else if (fs && strcmp(fs, "allow-cross-slot-keys") == 0) flags |= FN_ALLOW_CROSS_SLOT;
                    else return raise_err(L, "unknown flag given");
                    lua_pop(L, 1);
                }
            } else {
                return raise_err(L, "unknown argument given to redis.register_function");
            }
            lua_pop(L, 1);
        }
        if (!name) return raise_err(L, "redis.register_function must get a function name argument");
        lua_pushnil(L);
        while (lua_next(L, 1)) {   /* the callback, by its case-insensitive key */
            if (lua_type(L, -2) == LUA_TSTRING && strcasecmp(lua_tostring(L, -2), "callback") == 0) {
                cb = lua_gettop(L);
                lua_pushvalue(L, -2);
                break;
            }
            lua_pop(L, 1);
        }
        if (!cb) return raise_err(L, "redis.register_function must get a callback argument");
    } else if (argc == 2) {
        if (lua_type(L, 1) != LUA_TSTRING)
            return raise_err(L, "first argument to redis.register_function must be a string");
        if (!lua_isfunction(L, 2))
            return raise_err(L, "second argument to redis.register_function must be a function");
        name = lua_tolstring(L, 1, &nlen);
        cb = 2;
    } else {
        return raise_err(L, "wrong number of arguments to redis.register_function");
    }
    if (!valid_name(name, nlen))
        return raise_err(L, "Library names can only contain letters, numbers, or underscores(_) and must be at least one character long");
    for (int f = 0; f < lib->nfns; f++)
        if (strlen(lib->fns[f].name) == nlen && memcmp(lib->fns[f].name, name, nlen) == 0)
            return raise_err(L, "Function already exists in the library");
    RegFunc *nf = (RegFunc *)realloc(lib->fns, sizeof(RegFunc) * (size_t)(lib->nfns + 1));
    if (!nf) return luaL_error(L, "not enough memory");
    lib->fns = nf;
    RegFunc *rf = &lib->fns[lib->nfns];
    memset(rf, 0, sizeof(*rf));
    rf->name = (char *)malloc(nlen + 1);
    if (!rf->name) return luaL_error(L, "not enough memory");
    memcpy(rf->name, name, nlen); rf->name[nlen] = 0;
    if (desc) {
        rf->desc = (char *)malloc(dlen + 1);
        if (rf->desc) { memcpy(rf->desc, desc, dlen); rf->desc[dlen] = 0; }
    }
    rf->flags = flags;
    lua_pushvalue(L, cb);
    rf->ref = luaL_ref(L, LUA_REGISTRYINDEX);
    lib->nfns++;
    return 0;
}

/* Parse "#!<engine> name=<lib>" — returns 0, or -1 with an error reply. */
static int library_shebang(PionLuaState *S, const char *code, size_t len, const char **name, size_t *nlen, size_t *body_off) {
    if (len < 2 || code[0] != '#' || code[1] != '!') {
        out_error(S, "ERR Missing library metadata");
        return -1;
    }
    size_t eol = 0;
    while (eol < len && code[eol] != '\n') eol++;
    const char *p = code + 2, *e = code + eol;
    const char *eng = p;
    while (p < e && *p != ' ' && *p != '\r') p++;
    size_t englen = (size_t)(p - eng);
    if (englen != 3 || strncasecmp(eng, "lua", 3) != 0) {
        char msg[200];
        snprintf(msg, sizeof(msg), "ERR Engine '%.*s' not found", (int)englen, eng);
        out_error(S, msg);
        return -1;
    }
    *name = NULL; *nlen = 0;
    while (p < e) {
        while (p < e && (*p == ' ' || *p == '\r')) p++;
        if (p >= e) break;
        const char *w = p;
        while (p < e && *p != ' ' && *p != '\r') p++;
        size_t wl = (size_t)(p - w);
        if (wl >= 5 && strncmp(w, "name=", 5) == 0) {
            *name = w + 5; *nlen = wl - 5;
        } else {
            char msg[200];
            snprintf(msg, sizeof(msg), "ERR Invalid metadata value given: %.*s", (int)wl, w);
            out_error(S, msg);
            return -1;
        }
    }
    if (!*name) { out_error(S, "ERR Library name was not given"); return -1; }
    if (!valid_name(*name, *nlen)) {
        out_error(S, "ERR Library names can only contain letters, numbers, or underscores(_) and must be at least one character long");
        return -1;
    }
    *body_off = eol;
    return 0;
}

static int find_library(PionLuaState *S, const char *name, size_t nlen) {
    for (int i = 0; i < S->nlibs; i++)
        if (strlen(S->libs[i].name) == nlen && memcmp(S->libs[i].name, name, nlen) == 0) return i;
    return -1;
}

typedef struct { int ref; } LoadCtx;

static int load_trampoline(lua_State *L) {
    LoadCtx *lc = (LoadCtx *)lua_touserdata(L, 1);
    lua_settop(L, 0);
    lua_rawgeti(L, LUA_REGISTRYINDEX, lc->ref);
    lua_call(L, 0, 0);
    return 0;
}

/* FUNCTION LOAD [REPLACE] code. Returns 1 when a library was (re)loaded, with
 * the library name as the reply in S->out; 0 with an error reply. */
int64_t pion_lua_function_load(PionLuaState *S, const char *code, int64_t len, int64_t replace) {
    if (!S || !S->FL) return 0;
    const char *name; size_t nlen, off;
    if (library_shebang(S, code, (size_t)len, &name, &nlen, &off) < 0) return 0;
    int existing = find_library(S, name, nlen);
    if (existing >= 0 && !replace) {
        char msg[200];
        snprintf(msg, sizeof(msg), "ERR Library '%.*s' already exists", (int)nlen, name);
        out_error(S, msg);
        return 0;
    }
    lua_State *L = S->FL;
    S->fmem.enforce = 1;
    int st = luaL_loadbuffer(L, code + off, (size_t)len - off, "@user_function");
    S->fmem.enforce = 0;
    if (st != 0) {
        char msg[1200];
        snprintf(msg, sizeof(msg), "ERR Error compiling function: %s", lua_tostring(L, -1));
        lua_pop(L, 1);
        out_error(S, msg);
        return 0;
    }
    int chunk_ref = luaL_ref(L, LUA_REGISTRYINDEX);
    Library nl;
    memset(&nl, 0, sizeof(nl));
    nl.name = (char *)malloc(nlen + 1);
    nl.code = (char *)malloc((size_t)len + 1);
    if (!nl.name || !nl.code) {
        free(nl.name); free(nl.code);
        luaL_unref(L, LUA_REGISTRYINDEX, chunk_ref);
        out_error(S, "ERR out of memory");
        return 0;
    }
    memcpy(nl.name, name, nlen); nl.name[nlen] = 0;
    memcpy(nl.code, code, (size_t)len); nl.code[len] = 0;
    nl.code_len = (size_t)len;

    /* The library body sees only `redis`, with the library API: register_function,
     * log, the LOG_* levels and the versions. Any other global, or any other
     * field of redis, is "nonexistent", as in Redis. */
    lua_pushvalue(L, LUA_GLOBALSINDEX);
    lua_getmetatable(L, -1);
    lua_newtable(L);
    push_redis_table(L, 1);
    set_error_metatable(L, -1);
    lua_enablereadonlytable(L, -1, 1);
    lua_setfield(L, -2, "redis");
    set_error_metatable(L, -1);
    lua_enablereadonlytable(L, -1, 1);
    lua_setfield(L, -2, "__index");
    lua_pop(L, 2);

    S->loading = &nl;
    LoadCtx lc = {chunk_ref};
    lua_sethook(L, time_hook, LUA_MASKCOUNT, 100000);
    S->wrote = 0; S->killed = 0;
    clock_gettime(CLOCK_MONOTONIC, &S->started);
    S->fmem.enforce = 1;
    lua_pushcfunction(L, load_trampoline);
    lua_pushlightuserdata(L, &lc);
    int rs = lua_pcall(L, 1, 0, 0);
    S->fmem.enforce = 0;
    lua_sethook(L, NULL, 0, 0);
    S->loading = NULL;

    lua_pushvalue(L, LUA_GLOBALSINDEX);
    lua_getmetatable(L, -1);
    lua_rawgeti(L, LUA_REGISTRYINDEX, S->fl_globals);
    lua_setfield(L, -2, "__index");
    lua_pop(L, 2);
    luaL_unref(L, LUA_REGISTRYINDEX, chunk_ref);

    const char *fail = NULL;
    char msg[1200];
    if (rs != 0) {
        const char *m = NULL;
        if (lua_istable(L, -1)) {
            lua_pushliteral(L, "err"); lua_rawget(L, -2);
            m = lua_tostring(L, -1);
            snprintf(msg, sizeof(msg), "ERR Error registering functions: %s", m ? m : "unknown error");
            lua_pop(L, 1);
        } else {
            m = lua_tostring(L, -1);
            snprintf(msg, sizeof(msg), "ERR Error registering functions: ERR %s", m ? m : "unknown error");
        }
        lua_pop(L, 1);
        fail = msg;
    } else if (nl.nfns == 0) {
        fail = "ERR No functions registered";
    } else {
        for (int f = 0; f < nl.nfns && !fail; f++) {
            Library *owner = NULL;
            RegFunc *other = find_function(S, nl.fns[f].name, strlen(nl.fns[f].name), &owner);
            if (other && (existing < 0 || owner != &S->libs[existing])) {
                snprintf(msg, sizeof(msg), "ERR Function %s already exists", nl.fns[f].name);
                fail = msg;
            }
        }
    }
    if (fail) {
        free_library(S, &nl);
        out_error(S, fail);
        return 0;
    }
    if (existing >= 0) {
        free_library(S, &S->libs[existing]);
        S->libs[existing] = nl;
    } else {
        if (S->nlibs == S->cap_libs) {
            int nc = S->cap_libs ? S->cap_libs * 2 : 16;
            Library *nlibs = (Library *)realloc(S->libs, sizeof(Library) * (size_t)nc);
            if (!nlibs) { free_library(S, &nl); out_error(S, "ERR out of memory"); return 0; }
            S->libs = nlibs; S->cap_libs = nc;
        }
        S->libs[S->nlibs++] = nl;
    }
    out_reset(S);
    out_bulk(S, name, nlen);
    return 1;
}

/* FUNCTION DELETE name: 1 when deleted, 0 when there is no such library. */
int64_t pion_lua_function_delete(PionLuaState *S, const char *name, int64_t nlen) {
    if (!S) return 0;
    int i = find_library(S, name, (size_t)nlen);
    if (i < 0) return 0;
    free_library(S, &S->libs[i]);
    memmove(&S->libs[i], &S->libs[i + 1], sizeof(Library) * (size_t)(S->nlibs - i - 1));
    S->nlibs--;
    return 1;
}

void pion_lua_function_flush(PionLuaState *S) {
    if (!S) return;
    for (int i = 0; i < S->nlibs; i++) free_library(S, &S->libs[i]);
    S->nlibs = 0;
    if (S->FL) lua_gc(S->FL, LUA_GCCOLLECT, 0);
}

int64_t pion_lua_library_count(PionLuaState *S) { return S ? S->nlibs : 0; }

const char *pion_lua_library_name(PionLuaState *S, int64_t i) {
    return (S && i >= 0 && i < S->nlibs) ? S->libs[i].name : "";
}

const char *pion_lua_library_code(PionLuaState *S, int64_t i, int64_t *len) {
    if (!S || i < 0 || i >= S->nlibs) { *len = 0; return ""; }
    *len = (int64_t)S->libs[i].code_len;
    return S->libs[i].code;
}

/* FCALL / FCALL_RO. The reply is in pion_lua_out. */
int64_t pion_lua_run_function(PionLuaState *S, const char *name, int64_t nlen,
                              const char **keys, const int64_t *key_lens, int64_t nkeys,
                              const char **args, const int64_t *arg_lens, int64_t nargs,
                              int64_t ro, int64_t client_resp) {
    if (!S || !S->FL) return -1;
    RegFunc *rf = find_function(S, name, (size_t)nlen, NULL);
    if (!rf) return out_error(S, "ERR Function not found");
    if (ro && !(rf->flags & FN_NO_WRITES))
        return out_error(S, "ERR Can not execute a script with write flag using *_ro command.");
    snprintf(S->run_name, sizeof(S->run_name), "%s", rf->name);
    S->run_source = "@user_function";
    S->ro = (ro || (rf->flags & FN_NO_WRITES)) ? 1 : 0;
    S->allow_oom = (rf->flags & FN_ALLOW_OOM) ? 1 : 0;
    RunCtx rc = {S, rf->ref, 1, keys, key_lens, nkeys, args, arg_lens, nargs};
    return run_user(S, S->FL, &S->fmem, &rc, client_resp);
}

/* Glob match for FUNCTION LIST LIBRARYNAME (Redis's stringmatchlen subset). */
static int glob(const char *p, size_t pl, const char *s, size_t sl) {
    while (pl > 0) {
        if (*p == '*') {
            while (pl > 1 && p[1] == '*') { p++; pl--; }
            if (pl == 1) return 1;
            for (size_t i = 0; i <= sl; i++)
                if (glob(p + 1, pl - 1, s + i, sl - i)) return 1;
            return 0;
        }
        if (sl == 0) return 0;
        if (*p == '?') { p++; pl--; s++; sl--; continue; }
        if (*p == '[') {
            size_t j = 1; int neg = 0, hit = 0;
            if (j < pl && p[j] == '^') { neg = 1; j++; }
            for (; j < pl && p[j] != ']'; j++) {
                if (p[j] == '\\' && j + 1 < pl) { j++; if (p[j] == *s) hit = 1; }
                else if (j + 2 < pl && p[j + 1] == '-' && p[j + 2] != ']') {
                    char lo = p[j], hi = p[j + 2];
                    if (lo > hi) { char t = lo; lo = hi; hi = t; }
                    if (*s >= lo && *s <= hi) hit = 1;
                    j += 2;
                } else if (p[j] == *s) hit = 1;
            }
            if (neg) hit = !hit;
            if (!hit) return 0;
            if (j < pl) j++;
            p += j; pl -= j; s++; sl--;
            continue;
        }
        if (*p == '\\' && pl > 1) { p++; pl--; }
        if (*p != *s) return 0;
        p++; pl--; s++; sl--;
    }
    return sl == 0;
}

static void out_flag_names(PionLuaState *S, int flags, int resp) {
    const char *names[5]; int n = 0;
    if (flags & FN_NO_WRITES) names[n++] = "no-writes";
    if (flags & FN_ALLOW_OOM) names[n++] = "allow-oom";
    if (flags & FN_ALLOW_STALE) names[n++] = "allow-stale";
    if (flags & FN_NO_CLUSTER) names[n++] = "no-cluster";
    if (flags & FN_ALLOW_CROSS_SLOT) names[n++] = "allow-cross-slot-keys";
    out_hdr(S, resp == 3 ? '~' : '*', n);
    for (int i = 0; i < n; i++) { out_add(S, "+", 1); out_str(S, names[i]); out_add(S, "\r\n", 2); }
}

/* FUNCTION LIST [LIBRARYNAME pattern] [WITHCODE] */
int64_t pion_lua_function_list(PionLuaState *S, const char *pat, int64_t plen, int64_t withcode, int64_t resp) {
    out_reset(S);
    int n = 0;
    for (int i = 0; i < S->nlibs; i++)
        if (!pat || glob(pat, (size_t)plen, S->libs[i].name, strlen(S->libs[i].name))) n++;
    out_hdr(S, '*', n);
    for (int i = 0; i < S->nlibs; i++) {
        Library *lib = &S->libs[i];
        if (pat && !glob(pat, (size_t)plen, lib->name, strlen(lib->name))) continue;
        int fields = withcode ? 4 : 3;
        out_hdr(S, resp == 3 ? '%' : '*', resp == 3 ? fields : 2 * fields);
        out_bulk(S, "library_name", 12); out_bulk(S, lib->name, strlen(lib->name));
        out_bulk(S, "engine", 6); out_bulk(S, "LUA", 3);
        out_bulk(S, "functions", 9);
        out_hdr(S, '*', lib->nfns);
        for (int f = 0; f < lib->nfns; f++) {
            RegFunc *rf = &lib->fns[f];
            out_hdr(S, resp == 3 ? '%' : '*', resp == 3 ? 3 : 6);
            out_bulk(S, "name", 4); out_bulk(S, rf->name, strlen(rf->name));
            out_bulk(S, "description", 11);
            if (rf->desc) out_bulk(S, rf->desc, strlen(rf->desc)); else out_null(S, (int)resp);
            out_bulk(S, "flags", 5); out_flag_names(S, rf->flags, (int)resp);
        }
        if (withcode) { out_bulk(S, "library_code", 12); out_bulk(S, lib->code, lib->code_len); }
    }
    return S->out_oom ? -1 : (int64_t)S->out_len;
}

/* FUNCTION STATS */
int64_t pion_lua_function_stats(PionLuaState *S, int64_t resp) {
    out_reset(S);
    int nf = 0;
    for (int i = 0; i < S->nlibs; i++) nf += S->libs[i].nfns;
    out_hdr(S, resp == 3 ? '%' : '*', resp == 3 ? 2 : 4);
    out_bulk(S, "running_script", 14);
    out_null(S, (int)resp);
    out_bulk(S, "engines", 7);
    out_hdr(S, resp == 3 ? '%' : '*', resp == 3 ? 1 : 2);
    out_bulk(S, "LUA", 3);
    out_hdr(S, resp == 3 ? '%' : '*', resp == 3 ? 2 : 4);
    out_bulk(S, "libraries_count", 15); out_hdr(S, ':', S->nlibs);
    out_bulk(S, "functions_count", 15); out_hdr(S, ':', nf);
    return S->out_oom ? -1 : (int64_t)S->out_len;
}

/* FUNCTION DUMP: Pion's own payload (as DUMP's is), restorable by FUNCTION
 * RESTORE here: "PIONFN1\n", then per library [u32 len][code]. */
int64_t pion_lua_function_dump(PionLuaState *S) {
    out_reset(S);
    out_add(S, "PIONFN1\n", 8);
    for (int i = 0; i < S->nlibs; i++) {
        uint32_t l = (uint32_t)S->libs[i].code_len;
        unsigned char b[4] = {(unsigned char)l, (unsigned char)(l >> 8), (unsigned char)(l >> 16), (unsigned char)(l >> 24)};
        out_add(S, (const char *)b, 4);
        out_add(S, S->libs[i].code, S->libs[i].code_len);
    }
    return S->out_oom ? -1 : (int64_t)S->out_len;
}

/* Validate a FUNCTION DUMP payload: the number of libraries, or -1. */
int64_t pion_lua_dump_count(const char *p, int64_t n) {
    if (n < 8 || memcmp(p, "PIONFN1\n", 8) != 0) return -1;
    int64_t pos = 8, cnt = 0;
    while (pos < n) {
        if (pos + 4 > n) return -1;
        uint32_t l = (uint32_t)(unsigned char)p[pos] | ((uint32_t)(unsigned char)p[pos + 1] << 8) |
                     ((uint32_t)(unsigned char)p[pos + 2] << 16) | ((uint32_t)(unsigned char)p[pos + 3] << 24);
        pos += 4;
        if (pos + (int64_t)l > n) return -1;
        pos += l;
        cnt++;
    }
    return cnt;
}

/* The k-th library's code in a validated payload. */
const char *pion_lua_dump_entry(const char *p, int64_t n, int64_t k, int64_t *len) {
    int64_t pos = 8;
    for (int64_t i = 0; pos + 4 <= n; i++) {
        uint32_t l = (uint32_t)(unsigned char)p[pos] | ((uint32_t)(unsigned char)p[pos + 1] << 8) |
                     ((uint32_t)(unsigned char)p[pos + 2] << 16) | ((uint32_t)(unsigned char)p[pos + 3] << 24);
        if (i == k) { *len = l; return p + pos + 4; }
        pos += 4 + l;
    }
    *len = 0;
    return "";
}

/* The library name a library's code declares, for FUNCTION RESTORE's
 * conflict check: 1 with name/nlen set, or 0. */
int64_t pion_lua_library_name_of(const char *code, int64_t len, const char **name, int64_t *nlen) {
    if (len < 2 || code[0] != '#' || code[1] != '!') return 0;
    int64_t eol = 0;
    while (eol < len && code[eol] != '\n') eol++;
    for (int64_t i = 2; i + 5 <= eol; i++) {
        if (memcmp(code + i, "name=", 5) == 0 && (code[i - 1] == ' ')) {
            int64_t s = i + 5, e = s;
            while (e < eol && code[e] != ' ' && code[e] != '\r') e++;
            *name = code + s; *nlen = e - s;
            return 1;
        }
    }
    return 0;
}

int64_t pion_lua_library_exists(PionLuaState *S, const char *name, int64_t nlen) {
    return (S && find_library(S, name, (size_t)nlen) >= 0) ? 1 : 0;
}


/* ── Persistence (this thread's state) ── */

/* Apply a function record from the WAL, a snapshot or the replication stream:
 * 35 FUNCTION LOAD (value = the library code; replaces a library of the same
 * name), 36 FUNCTION DELETE (key = the library name), 37 FUNCTION FLUSH.
 * 1 when applied. */
int64_t pion_lua_wal_apply(int64_t cmd_id, const char *key, int64_t kl, const char *val, int64_t vl) {
    PionLuaState *S = tls_lua;
    if (!S) return 0;
    if (cmd_id == 35) return pion_lua_function_load(S, val, vl, 1) == 1;
    if (cmd_id == 36) return pion_lua_function_delete(S, key, kl);
    if (cmd_id == 37) { pion_lua_function_flush(S); return 1; }
    return 0;
}

int64_t pion_lua_tls_library_count(void) { return tls_lua ? tls_lua->nlibs : 0; }

const char *pion_lua_tls_library_name(int64_t i) { return pion_lua_library_name(tls_lua, i); }

const char *pion_lua_tls_library_code(int64_t i, int64_t *len) { return pion_lua_library_code(tls_lua, i, len); }

int64_t pion_lua_function_exists(PionLuaState *S, const char *name, int64_t nlen) {
    return (S && find_function(S, name, (size_t)nlen, NULL)) ? 1 : 0;
}

int64_t pion_lua_out_len(PionLuaState *S) { return S ? (int64_t)S->out_len : 0; }
