"""gh #101: per-connection tenant binding — transparent key namespacing.

Tenant credentials come from repeatable `--tenant NAME=PASSWORD` flags
(serialized by main.mojo into config.server.tenants as newline-joined
"NAME=PASSWORD" entries). When any tenant is configured:

- `--requirepass` is REQUIRED and becomes the admin credential (unprefixed
  keyspace, full command set). There is no anonymous path: the gh #100
  NOAUTH gate rejects every command until the connection binds.
- `AUTH NAME PASSWORD` binds the connection to tenant NAME; every key the
  connection touches is transparently prefixed with "NAME:" by the
  slow-path pre-pass (apply_tenant_rewrite below). Tenant connections are
  routed off the fast path in fast_path.mojo.
- Tenant names are restricted to [A-Za-z0-9_-]{1,64}. ':' is therefore
  impossible in a name, which makes the name→prefix map prefix-free:
  tenant A can never forge a key in tenant B's namespace — a key that
  literally contains "B:" becomes "A:B:..." after rewrite.
- Commands are DENY-BY-DEFAULT: tenant_keyspec() is an allowlist; anything
  not on it (EVAL/FLUSHALL/CONFIG/FT.*/pub-sub/admin/...) is rejected with
  -NOPERM. Fail-closed at the command level, not just the key level.

Non-goals (documented in doc/multi_tenant.md): tenants still share the
worker's WAL, memory, and CPU. One-process-per-tenant remains the
deployment for hard resource isolation.
"""
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.collections import Array
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS

comptime MAX_TENANTS = 64
comptime MAX_TENANT_NAME = 64
comptime MAX_TENANT_PASS = 128
# Per-worker scratch for rewritten key tokens (reset per command). A command
# whose prefixed keys exceed this is rejected — fail-closed, never truncated.
comptime TENANT_SCRATCH_CAP = 256 * 1024


@always_inline
def _lit_eq_ci(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int,
               lit: StringLiteral) -> Bool:
    """Case-insensitive ASCII match of a token against a lowercase literal."""
    if tl != lit.byte_length():
        return False
    var lp = lit.unsafe_ptr()
    for k in range(tl):
        if (tp[unsafe_offset=k] | 0x20) != lp[unsafe_offset=k]:
            return False
    return True


def valid_tenant_name_bytes(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    """[A-Za-z0-9_-]{1,MAX_TENANT_NAME}. Banning ':' is the prefix-freeness
    guarantee; banning glob metacharacters keeps names safe to splice into
    MATCH patterns."""
    if n < 1 or n > MAX_TENANT_NAME:
        return False
    for k in range(n):
        var c = p[unsafe_offset=k]
        var ok = (c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (
            c >= 97 and c <= 122) or c == 95 or c == 45
        if not ok:
            return False
    return True


def tenant_arg_error(arg: String) -> String:
    """Validate one `--tenant NAME=PASSWORD` argument. Returns "" if valid,
    else a human-readable reason for the FATAL startup message."""
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(arg.unsafe_ptr()))
    var n = arg.byte_length()
    var eq = -1
    for k in range(n):
        if p[unsafe_offset=k] == 61:  # '='
            eq = k
            break
    if eq < 0:
        return String("expected NAME=PASSWORD")
    if not valid_tenant_name_bytes(p, eq):
        return String("tenant name must match [A-Za-z0-9_-]{1,64}")
    var pw_len = n - eq - 1
    if pw_len < 1:
        return String("password must be non-empty")
    if pw_len > MAX_TENANT_PASS:
        return String("password too long (max 128 bytes)")
    for k in range(eq + 1, n):
        if p[unsafe_offset=k] == 10 or p[unsafe_offset=k] == 13:
            return String("password may not contain newlines")
    return String("")


struct TenantTable(Movable):
    """Read-only after startup (shared-nothing safe: each worker parses its
    own copy from config). Flat fixed-stride storage — no per-lookup allocs."""
    var count: Int
    var name_buf: Pointer[UInt8, MutUntrackedOrigin]   # MAX_TENANTS × MAX_TENANT_NAME
    var name_lens: Pointer[Int32, MutUntrackedOrigin]  # [MAX_TENANTS]
    var pass_buf: Pointer[UInt8, MutUntrackedOrigin]   # MAX_TENANTS × MAX_TENANT_PASS
    var pass_lens: Pointer[Int32, MutUntrackedOrigin]  # [MAX_TENANTS]

    def __init__(out self, spec: String):
        """spec = newline-joined "NAME=PASSWORD" entries, pre-validated by
        tenant_arg_error at arg-parse time. Malformed entries are skipped
        (never partially loaded). Duplicate names: first entry wins."""
        self.count = 0
        self.name_buf = alloc[UInt8](MAX_TENANTS * MAX_TENANT_NAME)
        self.name_lens = alloc[Int32](MAX_TENANTS)
        self.pass_buf = alloc[UInt8](MAX_TENANTS * MAX_TENANT_PASS)
        self.pass_lens = alloc[Int32](MAX_TENANTS)
        var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(spec.unsafe_ptr()))
        var n = spec.byte_length()
        var start = 0
        var k = 0
        while k <= n:
            if k == n or p[unsafe_offset=k] == 10:  # '\n' or end
                if k > start and self.count < MAX_TENANTS:
                    var eq = -1
                    for j in range(start, k):
                        if p[unsafe_offset=j] == 61:  # '='
                            eq = j
                            break
                    if eq > start:
                        var nl = eq - start
                        var pl = k - eq - 1
                        if (nl <= MAX_TENANT_NAME and pl >= 1
                                and pl <= MAX_TENANT_PASS
                                and self._find_name(p.unsafe_offset(start), nl) < 0):
                            unsafe_memcpy(dest=self.name_buf.unsafe_offset(self.count * MAX_TENANT_NAME),
                                   src=p.unsafe_offset(start), count=nl)
                            self.name_lens[unsafe_offset=self.count] = Int32(nl)
                            unsafe_memcpy(dest=self.pass_buf.unsafe_offset(self.count * MAX_TENANT_PASS),
                                   src=p.unsafe_offset(eq).unsafe_offset(1), count=pl)
                            self.pass_lens[unsafe_offset=self.count] = Int32(pl)
                            self.count += 1
                start = k + 1
            k += 1

    @always_inline
    def name_ptr(self, t: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        return self.name_buf.unsafe_offset(t * MAX_TENANT_NAME)

    @always_inline
    def name_len(self, t: Int) -> Int:
        return Int(self.name_lens[unsafe_offset=t])

    def _find_name(self, user_ptr: Pointer[UInt8, MutUntrackedOrigin],
                   user_len: Int) -> Int:
        """Tenant id whose name exactly matches (case-sensitive), or -1."""
        for t in range(self.count):
            if Int(self.name_lens[unsafe_offset=t]) != user_len:
                continue
            var np = self.name_buf.unsafe_offset(t * MAX_TENANT_NAME)
            var same = True
            for k in range(user_len):
                if np[unsafe_offset=k] != user_ptr[unsafe_offset=k]:
                    same = False
                    break
            if same:
                return t
        return -1

    def match_credentials(self, user_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          user_len: Int,
                          pw_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          pw_len: Int) -> Int:
        """Tenant id on name+password match; -2 when the name exists but the
        password is wrong (caller must NOT fall through to the admin
        credential); -1 when no tenant has that name."""
        var t = self._find_name(user_ptr, user_len)
        if t < 0:
            return -1
        if Int(self.pass_lens[unsafe_offset=t]) != pw_len:
            return -2
        var pp = self.pass_buf.unsafe_offset(t * MAX_TENANT_PASS)
        for k in range(pw_len):
            if pp[unsafe_offset=k] != pw_ptr[unsafe_offset=k]:
                return -2
        return t


@fieldwise_init
struct TenantKeySpec(Copyable, Movable, ImplicitlyCopyable):
    """Redis-style keyspec for the tenant rewrite. firstkey/lastkey are
    1-based arg positions relative to the command token; lastkey == -1 means
    keys run through the command's last token."""
    var allowed: Bool
    var firstkey: Int32   # 0 = allowed but no key args (PING, MULTI, ...)
    var lastkey: Int32
    var keystep: Int32


def tenant_keyspec(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> TenantKeySpec:
    """Deny-by-default command allowlist for tenant connections.

    Anything not enumerated here — EVAL*/SCRIPT/FUNCTION/FCALL (runtime-computed
    keys Lua can't be gated on), FLUSHALL/FLUSHDB, SAVE/BGSAVE/CONFIG/DEBUG/
    CLUSTER/SHUTDOWN (admin surface), FT.*/V*/AI.*/KV.*/ATTEND.*/RAG.*/MOE.*/
    SSM.* (vector isolation needs per-tenant indexes — ANN neighbor sets are
    prefix-blind), SUBSCRIBE/PUBLISH (channels are a separate namespace),
    XREAD/SORT/OBJECT/ZUNIONSTORE/ZINTERSTORE/BITOP/SINTERCARD/LMPOP (numkeys/
    pattern key forms), DBSIZE/RANDOMKEY-like leaks — is rejected with -NOPERM.

    Multi-key commands where EVERY key argument is rewritten (RENAME, COPY,
    SMOVE, LMOVE, S*STORE, MSET, DEL, ...) are safe by construction: all keys
    land in the caller's own namespace."""
    # Dotted commands (FT.*, AI.*, KV.*, V.*, ...) are never tenant-callable.
    for k in range(tl):
        if tp[unsafe_offset=k] == 46:  # '.'
            return TenantKeySpec(False, 0, 0, 0)

    # ── No-key commands (connection/session scope) ──
    if (_lit_eq_ci(tp, tl, "ping") or _lit_eq_ci(tp, tl, "echo")
            or _lit_eq_ci(tp, tl, "auth") or _lit_eq_ci(tp, tl, "hello")
            or _lit_eq_ci(tp, tl, "quit") or _lit_eq_ci(tp, tl, "reset")
            or _lit_eq_ci(tp, tl, "select") or _lit_eq_ci(tp, tl, "swapdb")
            or _lit_eq_ci(tp, tl, "multi") or _lit_eq_ci(tp, tl, "exec")
            or _lit_eq_ci(tp, tl, "discard") or _lit_eq_ci(tp, tl, "unwatch")
            or _lit_eq_ci(tp, tl, "command") or _lit_eq_ci(tp, tl, "client")
            or _lit_eq_ci(tp, tl, "info")
            # KEYS/SCAN take a pattern, not a key; the handlers filter+strip
            # via the caller-supplied namespace (see key_mgmt.mojo).
            or _lit_eq_ci(tp, tl, "keys") or _lit_eq_ci(tp, tl, "scan")):
        return TenantKeySpec(True, 0, 0, 0)

    # ── Single key at arg 1 ──
    if (_lit_eq_ci(tp, tl, "get") or _lit_eq_ci(tp, tl, "set")
            or _lit_eq_ci(tp, tl, "setnx") or _lit_eq_ci(tp, tl, "setex")
            or _lit_eq_ci(tp, tl, "psetex") or _lit_eq_ci(tp, tl, "getset")
            or _lit_eq_ci(tp, tl, "getdel") or _lit_eq_ci(tp, tl, "getex")
            or _lit_eq_ci(tp, tl, "append") or _lit_eq_ci(tp, tl, "strlen")
            or _lit_eq_ci(tp, tl, "incr") or _lit_eq_ci(tp, tl, "decr")
            or _lit_eq_ci(tp, tl, "incrby") or _lit_eq_ci(tp, tl, "decrby")
            or _lit_eq_ci(tp, tl, "incrbyfloat")
            or _lit_eq_ci(tp, tl, "setrange") or _lit_eq_ci(tp, tl, "getrange")
            or _lit_eq_ci(tp, tl, "substr")
            or _lit_eq_ci(tp, tl, "type") or _lit_eq_ci(tp, tl, "ttl")
            or _lit_eq_ci(tp, tl, "pttl") or _lit_eq_ci(tp, tl, "expire")
            or _lit_eq_ci(tp, tl, "pexpire") or _lit_eq_ci(tp, tl, "expireat")
            or _lit_eq_ci(tp, tl, "pexpireat") or _lit_eq_ci(tp, tl, "persist")
            or _lit_eq_ci(tp, tl, "expiretime") or _lit_eq_ci(tp, tl, "pexpiretime")):
        return TenantKeySpec(True, 1, 1, 1)
    if (_lit_eq_ci(tp, tl, "hset") or _lit_eq_ci(tp, tl, "hsetnx")
            or _lit_eq_ci(tp, tl, "hget") or _lit_eq_ci(tp, tl, "hmget")
            or _lit_eq_ci(tp, tl, "hmset") or _lit_eq_ci(tp, tl, "hdel")
            or _lit_eq_ci(tp, tl, "hlen") or _lit_eq_ci(tp, tl, "hexists")
            or _lit_eq_ci(tp, tl, "hkeys") or _lit_eq_ci(tp, tl, "hvals")
            or _lit_eq_ci(tp, tl, "hgetall") or _lit_eq_ci(tp, tl, "hincrby")
            or _lit_eq_ci(tp, tl, "hincrbyfloat")
            or _lit_eq_ci(tp, tl, "hrandfield") or _lit_eq_ci(tp, tl, "hstrlen")
            or _lit_eq_ci(tp, tl, "hscan")):
        return TenantKeySpec(True, 1, 1, 1)
    if (_lit_eq_ci(tp, tl, "lpush") or _lit_eq_ci(tp, tl, "rpush")
            or _lit_eq_ci(tp, tl, "lpushx") or _lit_eq_ci(tp, tl, "rpushx")
            or _lit_eq_ci(tp, tl, "lpop") or _lit_eq_ci(tp, tl, "rpop")
            or _lit_eq_ci(tp, tl, "llen") or _lit_eq_ci(tp, tl, "lrange")
            or _lit_eq_ci(tp, tl, "lindex") or _lit_eq_ci(tp, tl, "lset")
            or _lit_eq_ci(tp, tl, "linsert") or _lit_eq_ci(tp, tl, "lrem")
            or _lit_eq_ci(tp, tl, "ltrim") or _lit_eq_ci(tp, tl, "lpos")):
        return TenantKeySpec(True, 1, 1, 1)
    if (_lit_eq_ci(tp, tl, "sadd") or _lit_eq_ci(tp, tl, "srem")
            or _lit_eq_ci(tp, tl, "spop") or _lit_eq_ci(tp, tl, "scard")
            or _lit_eq_ci(tp, tl, "sismember") or _lit_eq_ci(tp, tl, "smismember")
            or _lit_eq_ci(tp, tl, "smembers") or _lit_eq_ci(tp, tl, "srandmember")
            or _lit_eq_ci(tp, tl, "sscan")):
        return TenantKeySpec(True, 1, 1, 1)
    if (_lit_eq_ci(tp, tl, "zadd") or _lit_eq_ci(tp, tl, "zscore")
            or _lit_eq_ci(tp, tl, "zincrby") or _lit_eq_ci(tp, tl, "zcard")
            or _lit_eq_ci(tp, tl, "zcount") or _lit_eq_ci(tp, tl, "zrange")
            or _lit_eq_ci(tp, tl, "zrangebyscore") or _lit_eq_ci(tp, tl, "zrevrange")
            or _lit_eq_ci(tp, tl, "zrevrangebyscore") or _lit_eq_ci(tp, tl, "zrank")
            or _lit_eq_ci(tp, tl, "zrevrank") or _lit_eq_ci(tp, tl, "zrem")
            or _lit_eq_ci(tp, tl, "zpopmin") or _lit_eq_ci(tp, tl, "zpopmax")
            or _lit_eq_ci(tp, tl, "zrangebylex") or _lit_eq_ci(tp, tl, "zremrangebyrank")
            or _lit_eq_ci(tp, tl, "zremrangebyscore") or _lit_eq_ci(tp, tl, "zremrangebylex")
            or _lit_eq_ci(tp, tl, "zlexcount") or _lit_eq_ci(tp, tl, "zscan")
            or _lit_eq_ci(tp, tl, "zmscore") or _lit_eq_ci(tp, tl, "zrandmember")):
        return TenantKeySpec(True, 1, 1, 1)
    if (_lit_eq_ci(tp, tl, "setbit") or _lit_eq_ci(tp, tl, "getbit")
            or _lit_eq_ci(tp, tl, "bitcount") or _lit_eq_ci(tp, tl, "bitpos")
            or _lit_eq_ci(tp, tl, "bitfield") or _lit_eq_ci(tp, tl, "bitfield_ro")
            or _lit_eq_ci(tp, tl, "pfadd")):
        return TenantKeySpec(True, 1, 1, 1)
    if (_lit_eq_ci(tp, tl, "xadd") or _lit_eq_ci(tp, tl, "xlen")
            or _lit_eq_ci(tp, tl, "xrange") or _lit_eq_ci(tp, tl, "xrevrange")
            or _lit_eq_ci(tp, tl, "xdel") or _lit_eq_ci(tp, tl, "xtrim")):
        return TenantKeySpec(True, 1, 1, 1)
    if (_lit_eq_ci(tp, tl, "geoadd") or _lit_eq_ci(tp, tl, "geodist")
            or _lit_eq_ci(tp, tl, "geopos") or _lit_eq_ci(tp, tl, "geohash")
            or _lit_eq_ci(tp, tl, "geosearch")):
        return TenantKeySpec(True, 1, 1, 1)

    # ── Every argument is a key ──
    if (_lit_eq_ci(tp, tl, "del") or _lit_eq_ci(tp, tl, "unlink")
            or _lit_eq_ci(tp, tl, "exists") or _lit_eq_ci(tp, tl, "touch")
            or _lit_eq_ci(tp, tl, "mget") or _lit_eq_ci(tp, tl, "watch")
            or _lit_eq_ci(tp, tl, "pfcount") or _lit_eq_ci(tp, tl, "pfmerge")
            or _lit_eq_ci(tp, tl, "sinter") or _lit_eq_ci(tp, tl, "sunion")
            or _lit_eq_ci(tp, tl, "sdiff") or _lit_eq_ci(tp, tl, "sinterstore")
            or _lit_eq_ci(tp, tl, "sunionstore") or _lit_eq_ci(tp, tl, "sdiffstore")
            or _lit_eq_ci(tp, tl, "rename") or _lit_eq_ci(tp, tl, "renamenx")):
        return TenantKeySpec(True, 1, -1, 1)

    # ── Exactly two keys (args 1 and 2), options follow ──
    if (_lit_eq_ci(tp, tl, "smove") or _lit_eq_ci(tp, tl, "rpoplpush")
            or _lit_eq_ci(tp, tl, "lmove") or _lit_eq_ci(tp, tl, "copy")
            or _lit_eq_ci(tp, tl, "geosearchstore")):
        return TenantKeySpec(True, 1, 2, 1)

    # ── Alternating key/value ──
    if _lit_eq_ci(tp, tl, "mset") or _lit_eq_ci(tp, tl, "msetnx"):
        return TenantKeySpec(True, 1, -1, 2)

    return TenantKeySpec(False, 0, 0, 0)


def tenant_rewrite_need(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    cmd_end_tok: Int,
    spec: TenantKeySpec,
    name_len: Int,
) -> Int:
    """#52: the scratch bytes apply_tenant_rewrite needs for this command,
    so the caller can size the scratch first: a command may have any number
    of keys, and rewriting cannot grow the buffer it repoints tokens into."""
    if not spec.allowed or spec.firstkey <= 0:
        return 0
    var need = 0
    var k = i + Int(spec.firstkey)
    var last = cmd_end_tok - 1
    if spec.lastkey > 0:
        var abs_last = i + Int(spec.lastkey)
        if abs_last < last:
            last = abs_last
    while k <= last:
        need += name_len + 1 + tokens[unsafe_offset=k].length
        k += Int(spec.keystep)
    return need


def apply_tenant_rewrite(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    i: Int,
    cmd_end_tok: Int,
    spec: TenantKeySpec,
    name_ptr: Pointer[UInt8, MutUntrackedOrigin],
    name_len: Int,
    scratch: Pointer[UInt8, MutUntrackedOrigin],
    scratch_cap: Int = TENANT_SCRATCH_CAP,
) -> Bool:
    """Prefix every key token of tokens[i..cmd_end_tok) with "NAME:",
    repointing the token into `scratch` (reset per command — safe because a
    command is fully dispatched before the next command's rewrite runs).
    Returns False on scratch overflow: caller must reject the command,
    never dispatch it un-rewritten."""
    if not spec.allowed or spec.firstkey <= 0:
        return True
    var prefix_len = name_len + 1
    var off = 0
    var k = i + Int(spec.firstkey)
    var last = cmd_end_tok - 1
    if spec.lastkey > 0:
        var abs_last = i + Int(spec.lastkey)
        if abs_last < last:
            last = abs_last
    while k <= last:
        var klen = tokens[unsafe_offset=k].length
        if off + prefix_len + klen > scratch_cap:
            return False
        unsafe_memcpy(dest=scratch.unsafe_offset(off), src=name_ptr, count=name_len)
        scratch[unsafe_offset=off + name_len] = 58  # ':'
        if klen > 0:
            unsafe_memcpy(dest=scratch.unsafe_offset(off).unsafe_offset(prefix_len), src=tokens[unsafe_offset=k].ptr, count=klen)
        tokens[unsafe_offset=k] = RESP3Token(tokens[unsafe_offset=k].marker, scratch.unsafe_offset(off), prefix_len + klen)
        off += prefix_len + klen
        k += Int(spec.keystep)
    return True
