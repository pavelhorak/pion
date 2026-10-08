"""Pion's command surface — GENERATED, do not hand-edit (gh #220).

Regenerate with `python3 tools/gen_command_table.py`; the drift test
`tests/test_command_table_drift.py` fails the build if this file and the
dispatch chains disagree.

Purpose: queue-time validation for MULTI. Redis rejects an unknown command when
it is QUEUED and then answers EXEC with -EXECABORT, applying nothing. Pion used
to queue anything with +QUEUED and only fail at replay, by which point the
other commands in the transaction had already been applied — one typo silently
turned an atomic transaction into a partially applied one.

Case folding here is A-Z only, NOT the usual `| 0x20`: 18 of these
names contain '_' (AI.KNN_LM.QUERY), and '_' | 0x20 is 0x7F, so the cheap fold
would fail to match every substrate command.
"""

from src.common.ptr import null_ptr, is_null, is_not_null


comptime PION_COMMAND_COUNT = 355


@always_inline
def _cmd_eq_ci(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int,
               lit: StringLiteral) -> Bool:
    """Case-insensitive ASCII match, folding only A-Z (see the note above)."""
    if tl != lit.byte_length():
        return False
    var lp = lit.unsafe_ptr()
    for k in range(tl):
        var c = tp[unsafe_offset=k]
        if c >= 65 and c <= 90:
            c |= 0x20
        if c != lp[unsafe_offset=k]:
            return False
    return True


def command_exists(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Bool:
    """Is this a command Pion dispatches? Bucketed by length so a lookup
    compares against only the names of that length."""
    if tl == 3:
        return (
            _cmd_eq_ci(tp, tl, "acl") or
            _cmd_eq_ci(tp, tl, "del") or
            _cmd_eq_ci(tp, tl, "get") or
            _cmd_eq_ci(tp, tl, "lcs") or
            _cmd_eq_ci(tp, tl, "set") or
            _cmd_eq_ci(tp, tl, "ttl")
        )
    elif tl == 4:
        return (
            _cmd_eq_ci(tp, tl, "auth") or
            _cmd_eq_ci(tp, tl, "copy") or
            _cmd_eq_ci(tp, tl, "decr") or
            _cmd_eq_ci(tp, tl, "dump") or
            _cmd_eq_ci(tp, tl, "echo") or
            _cmd_eq_ci(tp, tl, "eval") or
            _cmd_eq_ci(tp, tl, "exec") or
            _cmd_eq_ci(tp, tl, "hdel") or
            _cmd_eq_ci(tp, tl, "hget") or
            _cmd_eq_ci(tp, tl, "hlen") or
            _cmd_eq_ci(tp, tl, "hset") or
            _cmd_eq_ci(tp, tl, "httl") or
            _cmd_eq_ci(tp, tl, "incr") or
            _cmd_eq_ci(tp, tl, "info") or
            _cmd_eq_ci(tp, tl, "keys") or
            _cmd_eq_ci(tp, tl, "llen") or
            _cmd_eq_ci(tp, tl, "lpop") or
            _cmd_eq_ci(tp, tl, "lpos") or
            _cmd_eq_ci(tp, tl, "lrem") or
            _cmd_eq_ci(tp, tl, "lset") or
            _cmd_eq_ci(tp, tl, "mget") or
            _cmd_eq_ci(tp, tl, "move") or
            _cmd_eq_ci(tp, tl, "mset") or
            _cmd_eq_ci(tp, tl, "ping") or
            _cmd_eq_ci(tp, tl, "pttl") or
            _cmd_eq_ci(tp, tl, "quit") or
            _cmd_eq_ci(tp, tl, "role") or
            _cmd_eq_ci(tp, tl, "rpop") or
            _cmd_eq_ci(tp, tl, "sadd") or
            _cmd_eq_ci(tp, tl, "save") or
            _cmd_eq_ci(tp, tl, "scan") or
            _cmd_eq_ci(tp, tl, "sort") or
            _cmd_eq_ci(tp, tl, "spop") or
            _cmd_eq_ci(tp, tl, "srem") or
            _cmd_eq_ci(tp, tl, "sync") or
            _cmd_eq_ci(tp, tl, "time") or
            _cmd_eq_ci(tp, tl, "type") or
            _cmd_eq_ci(tp, tl, "vadd") or
            _cmd_eq_ci(tp, tl, "vdim") or
            _cmd_eq_ci(tp, tl, "vemb") or
            _cmd_eq_ci(tp, tl, "vrem") or
            _cmd_eq_ci(tp, tl, "vsim") or
            _cmd_eq_ci(tp, tl, "wait") or
            _cmd_eq_ci(tp, tl, "xack") or
            _cmd_eq_ci(tp, tl, "xadd") or
            _cmd_eq_ci(tp, tl, "xdel") or
            _cmd_eq_ci(tp, tl, "xgpu") or
            _cmd_eq_ci(tp, tl, "xlen") or
            _cmd_eq_ci(tp, tl, "zadd") or
            _cmd_eq_ci(tp, tl, "zrem")
        )
    elif tl == 5:
        return (
            _cmd_eq_ci(tp, tl, "bitop") or
            _cmd_eq_ci(tp, tl, "blpop") or
            _cmd_eq_ci(tp, tl, "brpop") or
            _cmd_eq_ci(tp, tl, "debug") or
            _cmd_eq_ci(tp, tl, "fcall") or
            _cmd_eq_ci(tp, tl, "getex") or
            _cmd_eq_ci(tp, tl, "hello") or
            _cmd_eq_ci(tp, tl, "hkeys") or
            _cmd_eq_ci(tp, tl, "hmget") or
            _cmd_eq_ci(tp, tl, "hmset") or
            _cmd_eq_ci(tp, tl, "hpttl") or
            _cmd_eq_ci(tp, tl, "hscan") or
            _cmd_eq_ci(tp, tl, "hvals") or
            _cmd_eq_ci(tp, tl, "lmove") or
            _cmd_eq_ci(tp, tl, "lmpop") or
            _cmd_eq_ci(tp, tl, "lpush") or
            _cmd_eq_ci(tp, tl, "ltrim") or
            _cmd_eq_ci(tp, tl, "multi") or
            _cmd_eq_ci(tp, tl, "pfadd") or
            _cmd_eq_ci(tp, tl, "psync") or
            _cmd_eq_ci(tp, tl, "reset") or
            _cmd_eq_ci(tp, tl, "rpush") or
            _cmd_eq_ci(tp, tl, "scard") or
            _cmd_eq_ci(tp, tl, "sdiff") or
            _cmd_eq_ci(tp, tl, "setex") or
            _cmd_eq_ci(tp, tl, "setnx") or
            _cmd_eq_ci(tp, tl, "smove") or
            _cmd_eq_ci(tp, tl, "sscan") or
            _cmd_eq_ci(tp, tl, "touch") or
            _cmd_eq_ci(tp, tl, "vcard") or
            _cmd_eq_ci(tp, tl, "vinfo") or
            _cmd_eq_ci(tp, tl, "watch") or
            _cmd_eq_ci(tp, tl, "xinfo") or
            _cmd_eq_ci(tp, tl, "xread") or
            _cmd_eq_ci(tp, tl, "xtrim") or
            _cmd_eq_ci(tp, tl, "zcard") or
            _cmd_eq_ci(tp, tl, "zdiff") or
            _cmd_eq_ci(tp, tl, "zmpop") or
            _cmd_eq_ci(tp, tl, "zrank") or
            _cmd_eq_ci(tp, tl, "zscan")
        )
    elif tl == 6:
        return (
            _cmd_eq_ci(tp, tl, "append") or
            _cmd_eq_ci(tp, tl, "asking") or
            _cmd_eq_ci(tp, tl, "bgsave") or
            _cmd_eq_ci(tp, tl, "bitpos") or
            _cmd_eq_ci(tp, tl, "blmove") or
            _cmd_eq_ci(tp, tl, "blmpop") or
            _cmd_eq_ci(tp, tl, "bzmpop") or
            _cmd_eq_ci(tp, tl, "client") or
            _cmd_eq_ci(tp, tl, "config") or
            _cmd_eq_ci(tp, tl, "dbsize") or
            _cmd_eq_ci(tp, tl, "decrby") or
            _cmd_eq_ci(tp, tl, "exists") or
            _cmd_eq_ci(tp, tl, "expire") or
            _cmd_eq_ci(tp, tl, "geoadd") or
            _cmd_eq_ci(tp, tl, "geopos") or
            _cmd_eq_ci(tp, tl, "getbit") or
            _cmd_eq_ci(tp, tl, "getdel") or
            _cmd_eq_ci(tp, tl, "getset") or
            _cmd_eq_ci(tp, tl, "hsetnx") or
            _cmd_eq_ci(tp, tl, "incrby") or
            _cmd_eq_ci(tp, tl, "lindex") or
            _cmd_eq_ci(tp, tl, "lolwut") or
            _cmd_eq_ci(tp, tl, "lpushx") or
            _cmd_eq_ci(tp, tl, "lrange") or
            _cmd_eq_ci(tp, tl, "memory") or
            _cmd_eq_ci(tp, tl, "module") or
            _cmd_eq_ci(tp, tl, "msetex") or
            _cmd_eq_ci(tp, tl, "msetnx") or
            _cmd_eq_ci(tp, tl, "object") or
            _cmd_eq_ci(tp, tl, "psetex") or
            _cmd_eq_ci(tp, tl, "pubsub") or
            _cmd_eq_ci(tp, tl, "rename") or
            _cmd_eq_ci(tp, tl, "rpushx") or
            _cmd_eq_ci(tp, tl, "script") or
            _cmd_eq_ci(tp, tl, "select") or
            _cmd_eq_ci(tp, tl, "setbit") or
            _cmd_eq_ci(tp, tl, "sinter") or
            _cmd_eq_ci(tp, tl, "strlen") or
            _cmd_eq_ci(tp, tl, "substr") or
            _cmd_eq_ci(tp, tl, "sunion") or
            _cmd_eq_ci(tp, tl, "swapdb") or
            _cmd_eq_ci(tp, tl, "unlink") or
            _cmd_eq_ci(tp, tl, "v.info") or
            _cmd_eq_ci(tp, tl, "vlinks") or
            _cmd_eq_ci(tp, tl, "vrange") or
            _cmd_eq_ci(tp, tl, "xclaim") or
            _cmd_eq_ci(tp, tl, "xdelex") or
            _cmd_eq_ci(tp, tl, "xgroup") or
            _cmd_eq_ci(tp, tl, "xrange") or
            _cmd_eq_ci(tp, tl, "xsetid") or
            _cmd_eq_ci(tp, tl, "zcount") or
            _cmd_eq_ci(tp, tl, "zinter") or
            _cmd_eq_ci(tp, tl, "zrange") or
            _cmd_eq_ci(tp, tl, "zscore") or
            _cmd_eq_ci(tp, tl, "zunion")
        )
    elif tl == 7:
        return (
            _cmd_eq_ci(tp, tl, "ai.chat") or
            _cmd_eq_ci(tp, tl, "cluster") or
            _cmd_eq_ci(tp, tl, "command") or
            _cmd_eq_ci(tp, tl, "discard") or
            _cmd_eq_ci(tp, tl, "eval_ro") or
            _cmd_eq_ci(tp, tl, "evalsha") or
            _cmd_eq_ci(tp, tl, "flushdb") or
            _cmd_eq_ci(tp, tl, "ft.info") or
            _cmd_eq_ci(tp, tl, "geodist") or
            _cmd_eq_ci(tp, tl, "geohash") or
            _cmd_eq_ci(tp, tl, "hexists") or
            _cmd_eq_ci(tp, tl, "hexpire") or
            _cmd_eq_ci(tp, tl, "hgetall") or
            _cmd_eq_ci(tp, tl, "hincrby") or
            _cmd_eq_ci(tp, tl, "hstrlen") or
            _cmd_eq_ci(tp, tl, "kv.info") or
            _cmd_eq_ci(tp, tl, "latency") or
            _cmd_eq_ci(tp, tl, "linsert") or
            _cmd_eq_ci(tp, tl, "migrate") or
            _cmd_eq_ci(tp, tl, "monitor") or
            _cmd_eq_ci(tp, tl, "persist") or
            _cmd_eq_ci(tp, tl, "pexpire") or
            _cmd_eq_ci(tp, tl, "pfcount") or
            _cmd_eq_ci(tp, tl, "pfdebug") or
            _cmd_eq_ci(tp, tl, "pfmerge") or
            _cmd_eq_ci(tp, tl, "publish") or
            _cmd_eq_ci(tp, tl, "restore") or
            _cmd_eq_ci(tp, tl, "slaveof") or
            _cmd_eq_ci(tp, tl, "slowlog") or
            _cmd_eq_ci(tp, tl, "sort_ro") or
            _cmd_eq_ci(tp, tl, "unwatch") or
            _cmd_eq_ci(tp, tl, "v.fetch") or
            _cmd_eq_ci(tp, tl, "waitaof") or
            _cmd_eq_ci(tp, tl, "xackdel") or
            _cmd_eq_ci(tp, tl, "zincrby") or
            _cmd_eq_ci(tp, tl, "zmscore") or
            _cmd_eq_ci(tp, tl, "zpopmax") or
            _cmd_eq_ci(tp, tl, "zpopmin")
        )
    elif tl == 8:
        return (
            _cmd_eq_ci(tp, tl, "ai.embed") or
            _cmd_eq_ci(tp, tl, "ai.flare") or
            _cmd_eq_ci(tp, tl, "ai.route") or
            _cmd_eq_ci(tp, tl, "bitcount") or
            _cmd_eq_ci(tp, tl, "bitfield") or
            _cmd_eq_ci(tp, tl, "bzpopmax") or
            _cmd_eq_ci(tp, tl, "bzpopmin") or
            _cmd_eq_ci(tp, tl, "expireat") or
            _cmd_eq_ci(tp, tl, "failover") or
            _cmd_eq_ci(tp, tl, "fcall_ro") or
            _cmd_eq_ci(tp, tl, "flushall") or
            _cmd_eq_ci(tp, tl, "function") or
            _cmd_eq_ci(tp, tl, "getrange") or
            _cmd_eq_ci(tp, tl, "hpersist") or
            _cmd_eq_ci(tp, tl, "hpexpire") or
            _cmd_eq_ci(tp, tl, "kv.fetch") or
            _cmd_eq_ci(tp, tl, "kv.store") or
            _cmd_eq_ci(tp, tl, "lastsave") or
            _cmd_eq_ci(tp, tl, "readonly") or
            _cmd_eq_ci(tp, tl, "renamenx") or
            _cmd_eq_ci(tp, tl, "replconf") or
            _cmd_eq_ci(tp, tl, "setrange") or
            _cmd_eq_ci(tp, tl, "shutdown") or
            _cmd_eq_ci(tp, tl, "smembers") or
            _cmd_eq_ci(tp, tl, "spublish") or
            _cmd_eq_ci(tp, tl, "v.commit") or
            _cmd_eq_ci(tp, tl, "v.create") or
            _cmd_eq_ci(tp, tl, "v.export") or
            _cmd_eq_ci(tp, tl, "vgetattr") or
            _cmd_eq_ci(tp, tl, "vsetattr") or
            _cmd_eq_ci(tp, tl, "xpending") or
            _cmd_eq_ci(tp, tl, "zrevrank")
        )
    elif tl == 9:
        return (
            _cmd_eq_ci(tp, tl, "ai.memory") or
            _cmd_eq_ci(tp, tl, "ft.create") or
            _cmd_eq_ci(tp, tl, "ft.hybrid") or
            _cmd_eq_ci(tp, tl, "ft.search") or
            _cmd_eq_ci(tp, tl, "georadius") or
            _cmd_eq_ci(tp, tl, "geosearch") or
            _cmd_eq_ci(tp, tl, "hexpireat") or
            _cmd_eq_ci(tp, tl, "pexpireat") or
            _cmd_eq_ci(tp, tl, "rag.query") or
            _cmd_eq_ci(tp, tl, "randomkey") or
            _cmd_eq_ci(tp, tl, "readwrite") or
            _cmd_eq_ci(tp, tl, "replicaof") or
            _cmd_eq_ci(tp, tl, "rpoplpush") or
            _cmd_eq_ci(tp, tl, "sismember") or
            _cmd_eq_ci(tp, tl, "subscribe") or
            _cmd_eq_ci(tp, tl, "v.restore") or
            _cmd_eq_ci(tp, tl, "vismember") or
            _cmd_eq_ci(tp, tl, "xrevrange") or
            _cmd_eq_ci(tp, tl, "zlexcount") or
            _cmd_eq_ci(tp, tl, "zrevrange")
        )
    elif tl == 10:
        return (
            _cmd_eq_ci(tp, tl, "brpoplpush") or
            _cmd_eq_ci(tp, tl, "evalsha_ro") or
            _cmd_eq_ci(tp, tl, "expiretime") or
            _cmd_eq_ci(tp, tl, "ft.addtext") or
            _cmd_eq_ci(tp, tl, "hpexpireat") or
            _cmd_eq_ci(tp, tl, "hrandfield") or
            _cmd_eq_ci(tp, tl, "pfselftest") or
            _cmd_eq_ci(tp, tl, "pion.stats") or
            _cmd_eq_ci(tp, tl, "psubscribe") or
            _cmd_eq_ci(tp, tl, "sdiffstore") or
            _cmd_eq_ci(tp, tl, "sintercard") or
            _cmd_eq_ci(tp, tl, "smismember") or
            _cmd_eq_ci(tp, tl, "ssubscribe") or
            _cmd_eq_ci(tp, tl, "state.free") or
            _cmd_eq_ci(tp, tl, "state.info") or
            _cmd_eq_ci(tp, tl, "state.read") or
            _cmd_eq_ci(tp, tl, "v.snapshot") or
            _cmd_eq_ci(tp, tl, "xautoclaim") or
            _cmd_eq_ci(tp, tl, "xreadgroup") or
            _cmd_eq_ci(tp, tl, "zdiffstore") or
            _cmd_eq_ci(tp, tl, "zintercard")
        )
    elif tl == 11:
        return (
            _cmd_eq_ci(tp, tl, "ai.complete") or
            _cmd_eq_ci(tp, tl, "ai.generate") or
            _cmd_eq_ci(tp, tl, "attend.info") or
            _cmd_eq_ci(tp, tl, "bitfield_ro") or
            _cmd_eq_ci(tp, tl, "ft.optimize") or
            _cmd_eq_ci(tp, tl, "hexpiretime") or
            _cmd_eq_ci(tp, tl, "incrbyfloat") or
            _cmd_eq_ci(tp, tl, "pexpiretime") or
            _cmd_eq_ci(tp, tl, "sinterstore") or
            _cmd_eq_ci(tp, tl, "srandmember") or
            _cmd_eq_ci(tp, tl, "state.alloc") or
            _cmd_eq_ci(tp, tl, "state.write") or
            _cmd_eq_ci(tp, tl, "sunionstore") or
            _cmd_eq_ci(tp, tl, "unsubscribe") or
            _cmd_eq_ci(tp, tl, "vrandmember") or
            _cmd_eq_ci(tp, tl, "zinterstore") or
            _cmd_eq_ci(tp, tl, "zrandmember") or
            _cmd_eq_ci(tp, tl, "zrangebylex") or
            _cmd_eq_ci(tp, tl, "zrangestore") or
            _cmd_eq_ci(tp, tl, "zunionstore")
        )
    elif tl == 12:
        return (
            _cmd_eq_ci(tp, tl, "ai.loadmodel") or
            _cmd_eq_ci(tp, tl, "attend.query") or
            _cmd_eq_ci(tp, tl, "attend.store") or
            _cmd_eq_ci(tp, tl, "bgrewriteaof") or
            _cmd_eq_ci(tp, tl, "ft.dropindex") or
            _cmd_eq_ci(tp, tl, "georadius_ro") or
            _cmd_eq_ci(tp, tl, "hincrbyfloat") or
            _cmd_eq_ci(tp, tl, "hpexpiretime") or
            _cmd_eq_ci(tp, tl, "punsubscribe") or
            _cmd_eq_ci(tp, tl, "sunsubscribe") or
            _cmd_eq_ci(tp, tl, "v.storebatch")
        )
    elif tl == 13:
        return (
            _cmd_eq_ci(tp, tl, "ai.route.info") or
            _cmd_eq_ci(tp, tl, "attend.create") or
            _cmd_eq_ci(tp, tl, "ft.searchtext") or
            _cmd_eq_ci(tp, tl, "zrangebyscore")
        )
    elif tl == 14:
        return (
            _cmd_eq_ci(tp, tl, "ai.knn_lm.drop") or
            _cmd_eq_ci(tp, tl, "ai.knn_lm.info") or
            _cmd_eq_ci(tp, tl, "geosearchstore") or
            _cmd_eq_ci(tp, tl, "kv.prefix.drop") or
            _cmd_eq_ci(tp, tl, "kv.prefix.info") or
            _cmd_eq_ci(tp, tl, "kv.prefix.save") or
            _cmd_eq_ci(tp, tl, "kv.prefix.warm") or
            _cmd_eq_ci(tp, tl, "moe.expert.pin") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.ffn") or
            _cmd_eq_ci(tp, tl, "restore-asking") or
            _cmd_eq_ci(tp, tl, "zremrangebylex") or
            _cmd_eq_ci(tp, tl, "zrevrangebylex")
        )
    elif tl == 15:
        return (
            _cmd_eq_ci(tp, tl, "ai.knn_lm.query") or
            _cmd_eq_ci(tp, tl, "ai.knn_lm.store") or
            _cmd_eq_ci(tp, tl, "ai.route.remove") or
            _cmd_eq_ci(tp, tl, "ai.route.update") or
            _cmd_eq_ci(tp, tl, "attend.finalize") or
            _cmd_eq_ci(tp, tl, "kv.prefix.owner") or
            _cmd_eq_ci(tp, tl, "moe.expert.hist") or
            _cmd_eq_ci(tp, tl, "moe.expert.info") or
            _cmd_eq_ci(tp, tl, "moe.expert.load") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.drop") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.info") or
            _cmd_eq_ci(tp, tl, "ssm.prefix.drop") or
            _cmd_eq_ci(tp, tl, "zremrangebyrank")
        )
    elif tl == 16:
        return (
            _cmd_eq_ci(tp, tl, "ai.knn_lm.create") or
            _cmd_eq_ci(tp, tl, "kv.prefix.blocks") or
            _cmd_eq_ci(tp, tl, "kv.prefix.commit") or
            _cmd_eq_ci(tp, tl, "kv.prefix.lookup") or
            _cmd_eq_ci(tp, tl, "moe.expert.fetch") or
            _cmd_eq_ci(tp, tl, "moe.expert.prune") or
            _cmd_eq_ci(tp, tl, "moe.expert.stats") or
            _cmd_eq_ci(tp, tl, "moe.expert.unpin") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.query") or
            _cmd_eq_ci(tp, tl, "ssm.prefix.fetch") or
            _cmd_eq_ci(tp, tl, "ssm.prefix.store") or
            _cmd_eq_ci(tp, tl, "zremrangebyscore") or
            _cmd_eq_ci(tp, tl, "zrevrangebyscore")
        )
    elif tl == 17:
        return (
            _cmd_eq_ci(tp, tl, "ai.route.register") or
            _cmd_eq_ci(tp, tl, "ai.semantic_cache") or
            _cmd_eq_ci(tp, tl, "georadiusbymember") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.create")
        )
    elif tl == 18:
        return (
            _cmd_eq_ci(tp, tl, "kv.prefix.register") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.setkeys") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.setvals") or
            _cmd_eq_ci(tp, tl, "rag.speculate.info")
        )
    elif tl == 19:
        return (
            _cmd_eq_ci(tp, tl, "attend.prefix.query") or
            _cmd_eq_ci(tp, tl, "attend.prefix.store") or
            _cmd_eq_ci(tp, tl, "moe.expert.prefetch")
        )
    elif tl == 20:
        return (
            _cmd_eq_ci(tp, tl, "ai.knn_lm.storebatch") or
            _cmd_eq_ci(tp, tl, "attend.prefix.lookup") or
            _cmd_eq_ci(tp, tl, "georadiusbymember_ro") or
            _cmd_eq_ci(tp, tl, "kv.prefix.membership") or
            _cmd_eq_ci(tp, tl, "rag.speculate.enable")
        )
    elif tl == 25:
        return (
            _cmd_eq_ci(tp, tl, "attend.prefix.query_fused")
        )
    elif tl == 26:
        return (
            _cmd_eq_ci(tp, tl, "attend.prefix.query_sparse")
        )
    elif tl == 31:
        return (
            _cmd_eq_ci(tp, tl, "attend.prefix.query_sparse_auto")
        )
    elif tl == 37:
        return (
            _cmd_eq_ci(tp, tl, "attend.prefix.query_sparse_auto_fused")
        )
    return False


@always_inline
def command_arity(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Int:
    """Redis's arity for this command, or 0 when Pion does not know one.

    266 of 355 commands have an entry; the rest are Pion-specific
    (FT.*, KV.PREFIX.*, AI.*, ATTEND.*) and are deliberately NOT validated.

    Encoding is Redis's own, kept verbatim so it can be checked against
    `COMMAND INFO` rather than re-derived: positive N means exactly N tokens
    INCLUDING the command name; negative N means at least |N|.

    0 means "no opinion" and MUST be treated as valid. The conservative
    direction matters here — a wrong arity rejects a transaction that would
    have worked, which is a worse failure than the missing check it replaces.
    """
    if tl == 3:
        if _cmd_eq_ci(tp, tl, "acl"): return -2
        if _cmd_eq_ci(tp, tl, "del"): return -2
        if _cmd_eq_ci(tp, tl, "get"): return 2
        if _cmd_eq_ci(tp, tl, "lcs"): return -3
        if _cmd_eq_ci(tp, tl, "set"): return -3
        if _cmd_eq_ci(tp, tl, "ttl"): return 2
    elif tl == 4:
        if _cmd_eq_ci(tp, tl, "auth"): return -2
        if _cmd_eq_ci(tp, tl, "copy"): return -3
        if _cmd_eq_ci(tp, tl, "decr"): return 2
        if _cmd_eq_ci(tp, tl, "dump"): return 2
        if _cmd_eq_ci(tp, tl, "echo"): return 2
        if _cmd_eq_ci(tp, tl, "eval"): return -3
        if _cmd_eq_ci(tp, tl, "exec"): return 1
        if _cmd_eq_ci(tp, tl, "hdel"): return -3
        if _cmd_eq_ci(tp, tl, "hget"): return 3
        if _cmd_eq_ci(tp, tl, "hlen"): return 2
        if _cmd_eq_ci(tp, tl, "hset"): return -4
        if _cmd_eq_ci(tp, tl, "httl"): return -5
        if _cmd_eq_ci(tp, tl, "incr"): return 2
        if _cmd_eq_ci(tp, tl, "info"): return -1
        if _cmd_eq_ci(tp, tl, "keys"): return 2
        if _cmd_eq_ci(tp, tl, "llen"): return 2
        if _cmd_eq_ci(tp, tl, "lpop"): return -2
        if _cmd_eq_ci(tp, tl, "lpos"): return -3
        if _cmd_eq_ci(tp, tl, "lrem"): return 4
        if _cmd_eq_ci(tp, tl, "lset"): return 4
        if _cmd_eq_ci(tp, tl, "mget"): return -2
        if _cmd_eq_ci(tp, tl, "move"): return 3
        if _cmd_eq_ci(tp, tl, "mset"): return -3
        if _cmd_eq_ci(tp, tl, "ping"): return -1
        if _cmd_eq_ci(tp, tl, "pttl"): return 2
        if _cmd_eq_ci(tp, tl, "quit"): return -1
        if _cmd_eq_ci(tp, tl, "role"): return 1
        if _cmd_eq_ci(tp, tl, "rpop"): return -2
        if _cmd_eq_ci(tp, tl, "sadd"): return -3
        if _cmd_eq_ci(tp, tl, "save"): return 1
        if _cmd_eq_ci(tp, tl, "scan"): return -2
        if _cmd_eq_ci(tp, tl, "sort"): return -2
        if _cmd_eq_ci(tp, tl, "spop"): return -2
        if _cmd_eq_ci(tp, tl, "srem"): return -3
        if _cmd_eq_ci(tp, tl, "sync"): return 1
        if _cmd_eq_ci(tp, tl, "time"): return 1
        if _cmd_eq_ci(tp, tl, "type"): return 2
        if _cmd_eq_ci(tp, tl, "vadd"): return -5
        if _cmd_eq_ci(tp, tl, "vdim"): return 2
        if _cmd_eq_ci(tp, tl, "vemb"): return -3
        if _cmd_eq_ci(tp, tl, "vrem"): return 3
        if _cmd_eq_ci(tp, tl, "vsim"): return -4
        if _cmd_eq_ci(tp, tl, "wait"): return 3
        if _cmd_eq_ci(tp, tl, "xack"): return -4
        if _cmd_eq_ci(tp, tl, "xadd"): return -5
        if _cmd_eq_ci(tp, tl, "xdel"): return -3
        if _cmd_eq_ci(tp, tl, "xlen"): return 2
        if _cmd_eq_ci(tp, tl, "zadd"): return -4
        if _cmd_eq_ci(tp, tl, "zrem"): return -3
    elif tl == 5:
        if _cmd_eq_ci(tp, tl, "bitop"): return -4
        if _cmd_eq_ci(tp, tl, "blpop"): return -3
        if _cmd_eq_ci(tp, tl, "brpop"): return -3
        if _cmd_eq_ci(tp, tl, "debug"): return -2
        if _cmd_eq_ci(tp, tl, "fcall"): return -3
        if _cmd_eq_ci(tp, tl, "getex"): return -2
        if _cmd_eq_ci(tp, tl, "hello"): return -1
        if _cmd_eq_ci(tp, tl, "hkeys"): return 2
        if _cmd_eq_ci(tp, tl, "hmget"): return -3
        if _cmd_eq_ci(tp, tl, "hmset"): return -4
        if _cmd_eq_ci(tp, tl, "hpttl"): return -5
        if _cmd_eq_ci(tp, tl, "hscan"): return -3
        if _cmd_eq_ci(tp, tl, "hvals"): return 2
        if _cmd_eq_ci(tp, tl, "lmove"): return 5
        if _cmd_eq_ci(tp, tl, "lmpop"): return -4
        if _cmd_eq_ci(tp, tl, "lpush"): return -3
        if _cmd_eq_ci(tp, tl, "ltrim"): return 4
        if _cmd_eq_ci(tp, tl, "multi"): return 1
        if _cmd_eq_ci(tp, tl, "pfadd"): return -2
        if _cmd_eq_ci(tp, tl, "psync"): return -3
        if _cmd_eq_ci(tp, tl, "reset"): return 1
        if _cmd_eq_ci(tp, tl, "rpush"): return -3
        if _cmd_eq_ci(tp, tl, "scard"): return 2
        if _cmd_eq_ci(tp, tl, "sdiff"): return -2
        if _cmd_eq_ci(tp, tl, "setex"): return 4
        if _cmd_eq_ci(tp, tl, "setnx"): return 3
        if _cmd_eq_ci(tp, tl, "smove"): return 4
        if _cmd_eq_ci(tp, tl, "sscan"): return -3
        if _cmd_eq_ci(tp, tl, "touch"): return -2
        if _cmd_eq_ci(tp, tl, "vcard"): return 2
        if _cmd_eq_ci(tp, tl, "vinfo"): return 2
        if _cmd_eq_ci(tp, tl, "watch"): return -2
        if _cmd_eq_ci(tp, tl, "xinfo"): return -2
        if _cmd_eq_ci(tp, tl, "xread"): return -4
        if _cmd_eq_ci(tp, tl, "xtrim"): return -4
        if _cmd_eq_ci(tp, tl, "zcard"): return 2
        if _cmd_eq_ci(tp, tl, "zdiff"): return -3
        if _cmd_eq_ci(tp, tl, "zmpop"): return -4
        if _cmd_eq_ci(tp, tl, "zrank"): return -3
        if _cmd_eq_ci(tp, tl, "zscan"): return -3
    elif tl == 6:
        if _cmd_eq_ci(tp, tl, "append"): return 3
        if _cmd_eq_ci(tp, tl, "asking"): return 1
        if _cmd_eq_ci(tp, tl, "bgsave"): return -1
        if _cmd_eq_ci(tp, tl, "bitpos"): return -3
        if _cmd_eq_ci(tp, tl, "blmove"): return 6
        if _cmd_eq_ci(tp, tl, "blmpop"): return -5
        if _cmd_eq_ci(tp, tl, "bzmpop"): return -5
        if _cmd_eq_ci(tp, tl, "client"): return -2
        if _cmd_eq_ci(tp, tl, "config"): return -2
        if _cmd_eq_ci(tp, tl, "dbsize"): return 1
        if _cmd_eq_ci(tp, tl, "decrby"): return 3
        if _cmd_eq_ci(tp, tl, "exists"): return -2
        if _cmd_eq_ci(tp, tl, "expire"): return -3
        if _cmd_eq_ci(tp, tl, "geoadd"): return -5
        if _cmd_eq_ci(tp, tl, "geopos"): return -2
        if _cmd_eq_ci(tp, tl, "getbit"): return 3
        if _cmd_eq_ci(tp, tl, "getdel"): return 2
        if _cmd_eq_ci(tp, tl, "getset"): return 3
        if _cmd_eq_ci(tp, tl, "hsetnx"): return 4
        if _cmd_eq_ci(tp, tl, "incrby"): return 3
        if _cmd_eq_ci(tp, tl, "lindex"): return 3
        if _cmd_eq_ci(tp, tl, "lolwut"): return -1
        if _cmd_eq_ci(tp, tl, "lpushx"): return -3
        if _cmd_eq_ci(tp, tl, "lrange"): return 4
        if _cmd_eq_ci(tp, tl, "memory"): return -2
        if _cmd_eq_ci(tp, tl, "module"): return -2
        if _cmd_eq_ci(tp, tl, "msetex"): return -4
        if _cmd_eq_ci(tp, tl, "msetnx"): return -3
        if _cmd_eq_ci(tp, tl, "object"): return -2
        if _cmd_eq_ci(tp, tl, "psetex"): return 4
        if _cmd_eq_ci(tp, tl, "pubsub"): return -2
        if _cmd_eq_ci(tp, tl, "rename"): return 3
        if _cmd_eq_ci(tp, tl, "rpushx"): return -3
        if _cmd_eq_ci(tp, tl, "script"): return -2
        if _cmd_eq_ci(tp, tl, "select"): return 2
        if _cmd_eq_ci(tp, tl, "setbit"): return 4
        if _cmd_eq_ci(tp, tl, "sinter"): return -2
        if _cmd_eq_ci(tp, tl, "strlen"): return 2
        if _cmd_eq_ci(tp, tl, "substr"): return 4
        if _cmd_eq_ci(tp, tl, "sunion"): return -2
        if _cmd_eq_ci(tp, tl, "swapdb"): return 3
        if _cmd_eq_ci(tp, tl, "unlink"): return -2
        if _cmd_eq_ci(tp, tl, "vlinks"): return -3
        if _cmd_eq_ci(tp, tl, "vrange"): return -4
        if _cmd_eq_ci(tp, tl, "xclaim"): return -6
        if _cmd_eq_ci(tp, tl, "xdelex"): return -5
        if _cmd_eq_ci(tp, tl, "xgroup"): return -2
        if _cmd_eq_ci(tp, tl, "xrange"): return -4
        if _cmd_eq_ci(tp, tl, "xsetid"): return -3
        if _cmd_eq_ci(tp, tl, "zcount"): return 4
        if _cmd_eq_ci(tp, tl, "zinter"): return -3
        if _cmd_eq_ci(tp, tl, "zrange"): return -4
        if _cmd_eq_ci(tp, tl, "zscore"): return 3
        if _cmd_eq_ci(tp, tl, "zunion"): return -3
    elif tl == 7:
        if _cmd_eq_ci(tp, tl, "cluster"): return -2
        if _cmd_eq_ci(tp, tl, "command"): return -1
        if _cmd_eq_ci(tp, tl, "discard"): return 1
        if _cmd_eq_ci(tp, tl, "eval_ro"): return -3
        if _cmd_eq_ci(tp, tl, "evalsha"): return -3
        if _cmd_eq_ci(tp, tl, "flushdb"): return -1
        if _cmd_eq_ci(tp, tl, "geodist"): return -4
        if _cmd_eq_ci(tp, tl, "geohash"): return -2
        if _cmd_eq_ci(tp, tl, "hexists"): return 3
        if _cmd_eq_ci(tp, tl, "hexpire"): return -6
        if _cmd_eq_ci(tp, tl, "hgetall"): return 2
        if _cmd_eq_ci(tp, tl, "hincrby"): return 4
        if _cmd_eq_ci(tp, tl, "hstrlen"): return 3
        if _cmd_eq_ci(tp, tl, "latency"): return -2
        if _cmd_eq_ci(tp, tl, "linsert"): return 5
        if _cmd_eq_ci(tp, tl, "migrate"): return -6
        if _cmd_eq_ci(tp, tl, "monitor"): return 1
        if _cmd_eq_ci(tp, tl, "persist"): return 2
        if _cmd_eq_ci(tp, tl, "pexpire"): return -3
        if _cmd_eq_ci(tp, tl, "pfcount"): return -2
        if _cmd_eq_ci(tp, tl, "pfdebug"): return 3
        if _cmd_eq_ci(tp, tl, "pfmerge"): return -2
        if _cmd_eq_ci(tp, tl, "publish"): return 3
        if _cmd_eq_ci(tp, tl, "restore"): return -4
        if _cmd_eq_ci(tp, tl, "slaveof"): return 3
        if _cmd_eq_ci(tp, tl, "slowlog"): return -2
        if _cmd_eq_ci(tp, tl, "sort_ro"): return -2
        if _cmd_eq_ci(tp, tl, "unwatch"): return 1
        if _cmd_eq_ci(tp, tl, "waitaof"): return 4
        if _cmd_eq_ci(tp, tl, "xackdel"): return -6
        if _cmd_eq_ci(tp, tl, "zincrby"): return 4
        if _cmd_eq_ci(tp, tl, "zmscore"): return -3
        if _cmd_eq_ci(tp, tl, "zpopmax"): return -2
        if _cmd_eq_ci(tp, tl, "zpopmin"): return -2
    elif tl == 8:
        if _cmd_eq_ci(tp, tl, "bitcount"): return -2
        if _cmd_eq_ci(tp, tl, "bitfield"): return -2
        if _cmd_eq_ci(tp, tl, "bzpopmax"): return -3
        if _cmd_eq_ci(tp, tl, "bzpopmin"): return -3
        if _cmd_eq_ci(tp, tl, "expireat"): return -3
        if _cmd_eq_ci(tp, tl, "failover"): return -1
        if _cmd_eq_ci(tp, tl, "fcall_ro"): return -3
        if _cmd_eq_ci(tp, tl, "flushall"): return -1
        if _cmd_eq_ci(tp, tl, "function"): return -2
        if _cmd_eq_ci(tp, tl, "getrange"): return 4
        if _cmd_eq_ci(tp, tl, "hpersist"): return -5
        if _cmd_eq_ci(tp, tl, "hpexpire"): return -6
        if _cmd_eq_ci(tp, tl, "lastsave"): return 1
        if _cmd_eq_ci(tp, tl, "readonly"): return 1
        if _cmd_eq_ci(tp, tl, "renamenx"): return 3
        if _cmd_eq_ci(tp, tl, "replconf"): return -1
        if _cmd_eq_ci(tp, tl, "setrange"): return 4
        if _cmd_eq_ci(tp, tl, "shutdown"): return -1
        if _cmd_eq_ci(tp, tl, "smembers"): return 2
        if _cmd_eq_ci(tp, tl, "spublish"): return 3
        if _cmd_eq_ci(tp, tl, "vgetattr"): return 3
        if _cmd_eq_ci(tp, tl, "vsetattr"): return 4
        if _cmd_eq_ci(tp, tl, "xpending"): return -3
        if _cmd_eq_ci(tp, tl, "zrevrank"): return -3
    elif tl == 9:
        if _cmd_eq_ci(tp, tl, "georadius"): return -6
        if _cmd_eq_ci(tp, tl, "geosearch"): return -7
        if _cmd_eq_ci(tp, tl, "hexpireat"): return -6
        if _cmd_eq_ci(tp, tl, "pexpireat"): return -3
        if _cmd_eq_ci(tp, tl, "randomkey"): return 1
        if _cmd_eq_ci(tp, tl, "readwrite"): return 1
        if _cmd_eq_ci(tp, tl, "replicaof"): return 3
        if _cmd_eq_ci(tp, tl, "rpoplpush"): return 3
        if _cmd_eq_ci(tp, tl, "sismember"): return 3
        if _cmd_eq_ci(tp, tl, "subscribe"): return -2
        if _cmd_eq_ci(tp, tl, "vismember"): return 3
        if _cmd_eq_ci(tp, tl, "xrevrange"): return -4
        if _cmd_eq_ci(tp, tl, "zlexcount"): return 4
        if _cmd_eq_ci(tp, tl, "zrevrange"): return -4
    elif tl == 10:
        if _cmd_eq_ci(tp, tl, "brpoplpush"): return 4
        if _cmd_eq_ci(tp, tl, "evalsha_ro"): return -3
        if _cmd_eq_ci(tp, tl, "expiretime"): return 2
        if _cmd_eq_ci(tp, tl, "hpexpireat"): return -6
        if _cmd_eq_ci(tp, tl, "hrandfield"): return -2
        if _cmd_eq_ci(tp, tl, "pfselftest"): return 1
        if _cmd_eq_ci(tp, tl, "psubscribe"): return -2
        if _cmd_eq_ci(tp, tl, "sdiffstore"): return -3
        if _cmd_eq_ci(tp, tl, "sintercard"): return -3
        if _cmd_eq_ci(tp, tl, "smismember"): return -3
        if _cmd_eq_ci(tp, tl, "ssubscribe"): return -2
        if _cmd_eq_ci(tp, tl, "xautoclaim"): return -6
        if _cmd_eq_ci(tp, tl, "xreadgroup"): return -7
        if _cmd_eq_ci(tp, tl, "zdiffstore"): return -4
        if _cmd_eq_ci(tp, tl, "zintercard"): return -3
    elif tl == 11:
        if _cmd_eq_ci(tp, tl, "bitfield_ro"): return -2
        if _cmd_eq_ci(tp, tl, "hexpiretime"): return -5
        if _cmd_eq_ci(tp, tl, "incrbyfloat"): return 3
        if _cmd_eq_ci(tp, tl, "pexpiretime"): return 2
        if _cmd_eq_ci(tp, tl, "sinterstore"): return -3
        if _cmd_eq_ci(tp, tl, "srandmember"): return -2
        if _cmd_eq_ci(tp, tl, "sunionstore"): return -3
        if _cmd_eq_ci(tp, tl, "unsubscribe"): return -1
        if _cmd_eq_ci(tp, tl, "vrandmember"): return -2
        if _cmd_eq_ci(tp, tl, "zinterstore"): return -4
        if _cmd_eq_ci(tp, tl, "zrandmember"): return -2
        if _cmd_eq_ci(tp, tl, "zrangebylex"): return -4
        if _cmd_eq_ci(tp, tl, "zrangestore"): return -5
        if _cmd_eq_ci(tp, tl, "zunionstore"): return -4
    elif tl == 12:
        if _cmd_eq_ci(tp, tl, "bgrewriteaof"): return 1
        if _cmd_eq_ci(tp, tl, "georadius_ro"): return -6
        if _cmd_eq_ci(tp, tl, "hincrbyfloat"): return 4
        if _cmd_eq_ci(tp, tl, "hpexpiretime"): return -5
        if _cmd_eq_ci(tp, tl, "punsubscribe"): return -1
        if _cmd_eq_ci(tp, tl, "sunsubscribe"): return -1
    elif tl == 13:
        if _cmd_eq_ci(tp, tl, "zrangebyscore"): return -4
    elif tl == 14:
        if _cmd_eq_ci(tp, tl, "geosearchstore"): return -8
        if _cmd_eq_ci(tp, tl, "restore-asking"): return -4
        if _cmd_eq_ci(tp, tl, "zremrangebylex"): return 4
        if _cmd_eq_ci(tp, tl, "zrevrangebylex"): return -4
    elif tl == 15:
        if _cmd_eq_ci(tp, tl, "zremrangebyrank"): return 4
    elif tl == 16:
        if _cmd_eq_ci(tp, tl, "zremrangebyscore"): return 4
        if _cmd_eq_ci(tp, tl, "zrevrangebyscore"): return -4
    elif tl == 17:
        if _cmd_eq_ci(tp, tl, "georadiusbymember"): return -5
    elif tl == 20:
        if _cmd_eq_ci(tp, tl, "georadiusbymember_ro"): return -5
    return 0


@always_inline
def command_is_write(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Bool:
    """True when this command mutates the keyspace, per real Redis's `write` flag.

    110 of 355 commands are writes. Used by gh #260 to refuse mutations
    once the WAL can no longer persist them, instead of acknowledging writes
    that will not survive a restart.

    A command absent from this table is NOT a write. For Pion's substrate
    surface (FT.*, KV.PREFIX.*, AI.*, ATTEND.*, V.*) that is correct, not just
    conservative: those planes own their own stores and never append to the
    shared keyspace WAL, so a full keyspace log is not a statement about them.

    EVAL/EVALSHA/FCALL are forced true. Redis classifies scripts per-invocation
    from what they actually call; Pion cannot, so it takes the safe side.
    """
    if tl == 3:
        return (
            _cmd_eq_ci(tp, tl, "del") or
            _cmd_eq_ci(tp, tl, "set")
        )
    elif tl == 4:
        return (
            _cmd_eq_ci(tp, tl, "copy") or
            _cmd_eq_ci(tp, tl, "decr") or
            _cmd_eq_ci(tp, tl, "hdel") or
            _cmd_eq_ci(tp, tl, "hset") or
            _cmd_eq_ci(tp, tl, "incr") or
            _cmd_eq_ci(tp, tl, "lpop") or
            _cmd_eq_ci(tp, tl, "lrem") or
            _cmd_eq_ci(tp, tl, "lset") or
            _cmd_eq_ci(tp, tl, "move") or
            _cmd_eq_ci(tp, tl, "mset") or
            _cmd_eq_ci(tp, tl, "rpop") or
            _cmd_eq_ci(tp, tl, "sadd") or
            _cmd_eq_ci(tp, tl, "sort") or
            _cmd_eq_ci(tp, tl, "spop") or
            _cmd_eq_ci(tp, tl, "srem") or
            _cmd_eq_ci(tp, tl, "vadd") or
            _cmd_eq_ci(tp, tl, "vrem") or
            _cmd_eq_ci(tp, tl, "xack") or
            _cmd_eq_ci(tp, tl, "xadd") or
            _cmd_eq_ci(tp, tl, "xdel") or
            _cmd_eq_ci(tp, tl, "zadd") or
            _cmd_eq_ci(tp, tl, "zrem")
        )
    elif tl == 5:
        return (
            _cmd_eq_ci(tp, tl, "bitop") or
            _cmd_eq_ci(tp, tl, "blpop") or
            _cmd_eq_ci(tp, tl, "brpop") or
            _cmd_eq_ci(tp, tl, "getex") or
            _cmd_eq_ci(tp, tl, "hmset") or
            _cmd_eq_ci(tp, tl, "lmove") or
            _cmd_eq_ci(tp, tl, "lmpop") or
            _cmd_eq_ci(tp, tl, "lpush") or
            _cmd_eq_ci(tp, tl, "ltrim") or
            _cmd_eq_ci(tp, tl, "pfadd") or
            _cmd_eq_ci(tp, tl, "rpush") or
            _cmd_eq_ci(tp, tl, "setex") or
            _cmd_eq_ci(tp, tl, "setnx") or
            _cmd_eq_ci(tp, tl, "smove") or
            _cmd_eq_ci(tp, tl, "xtrim") or
            _cmd_eq_ci(tp, tl, "zmpop")
        )
    elif tl == 6:
        return (
            _cmd_eq_ci(tp, tl, "append") or
            _cmd_eq_ci(tp, tl, "blmove") or
            _cmd_eq_ci(tp, tl, "blmpop") or
            _cmd_eq_ci(tp, tl, "bzmpop") or
            _cmd_eq_ci(tp, tl, "decrby") or
            _cmd_eq_ci(tp, tl, "expire") or
            _cmd_eq_ci(tp, tl, "geoadd") or
            _cmd_eq_ci(tp, tl, "getdel") or
            _cmd_eq_ci(tp, tl, "getset") or
            _cmd_eq_ci(tp, tl, "hsetnx") or
            _cmd_eq_ci(tp, tl, "incrby") or
            _cmd_eq_ci(tp, tl, "lpushx") or
            _cmd_eq_ci(tp, tl, "msetex") or
            _cmd_eq_ci(tp, tl, "msetnx") or
            _cmd_eq_ci(tp, tl, "psetex") or
            _cmd_eq_ci(tp, tl, "rename") or
            _cmd_eq_ci(tp, tl, "rpushx") or
            _cmd_eq_ci(tp, tl, "setbit") or
            _cmd_eq_ci(tp, tl, "swapdb") or
            _cmd_eq_ci(tp, tl, "unlink") or
            _cmd_eq_ci(tp, tl, "xclaim") or
            _cmd_eq_ci(tp, tl, "xdelex") or
            _cmd_eq_ci(tp, tl, "xsetid")
        )
    elif tl == 7:
        return (
            _cmd_eq_ci(tp, tl, "flushdb") or
            _cmd_eq_ci(tp, tl, "hexpire") or
            _cmd_eq_ci(tp, tl, "hincrby") or
            _cmd_eq_ci(tp, tl, "linsert") or
            _cmd_eq_ci(tp, tl, "migrate") or
            _cmd_eq_ci(tp, tl, "persist") or
            _cmd_eq_ci(tp, tl, "pexpire") or
            _cmd_eq_ci(tp, tl, "pfdebug") or
            _cmd_eq_ci(tp, tl, "pfmerge") or
            _cmd_eq_ci(tp, tl, "restore") or
            _cmd_eq_ci(tp, tl, "xackdel") or
            _cmd_eq_ci(tp, tl, "zincrby") or
            _cmd_eq_ci(tp, tl, "zpopmax") or
            _cmd_eq_ci(tp, tl, "zpopmin")
        )
    elif tl == 8:
        return (
            _cmd_eq_ci(tp, tl, "bitfield") or
            _cmd_eq_ci(tp, tl, "bzpopmax") or
            _cmd_eq_ci(tp, tl, "bzpopmin") or
            _cmd_eq_ci(tp, tl, "expireat") or
            _cmd_eq_ci(tp, tl, "flushall") or
            _cmd_eq_ci(tp, tl, "hpersist") or
            _cmd_eq_ci(tp, tl, "hpexpire") or
            _cmd_eq_ci(tp, tl, "renamenx") or
            _cmd_eq_ci(tp, tl, "setrange") or
            _cmd_eq_ci(tp, tl, "vsetattr")
        )
    elif tl == 9:
        return (
            _cmd_eq_ci(tp, tl, "georadius") or
            _cmd_eq_ci(tp, tl, "hexpireat") or
            _cmd_eq_ci(tp, tl, "pexpireat") or
            _cmd_eq_ci(tp, tl, "rpoplpush")
        )
    elif tl == 10:
        return (
            _cmd_eq_ci(tp, tl, "brpoplpush") or
            _cmd_eq_ci(tp, tl, "hpexpireat") or
            _cmd_eq_ci(tp, tl, "sdiffstore") or
            _cmd_eq_ci(tp, tl, "xautoclaim") or
            _cmd_eq_ci(tp, tl, "xreadgroup") or
            _cmd_eq_ci(tp, tl, "zdiffstore")
        )
    elif tl == 11:
        return (
            _cmd_eq_ci(tp, tl, "incrbyfloat") or
            _cmd_eq_ci(tp, tl, "sinterstore") or
            _cmd_eq_ci(tp, tl, "sunionstore") or
            _cmd_eq_ci(tp, tl, "zinterstore") or
            _cmd_eq_ci(tp, tl, "zrangestore") or
            _cmd_eq_ci(tp, tl, "zunionstore")
        )
    elif tl == 12:
        return (
            _cmd_eq_ci(tp, tl, "hincrbyfloat")
        )
    elif tl == 14:
        return (
            _cmd_eq_ci(tp, tl, "geosearchstore") or
            _cmd_eq_ci(tp, tl, "restore-asking") or
            _cmd_eq_ci(tp, tl, "zremrangebylex")
        )
    elif tl == 15:
        return (
            _cmd_eq_ci(tp, tl, "zremrangebyrank")
        )
    elif tl == 16:
        return (
            _cmd_eq_ci(tp, tl, "zremrangebyscore")
        )
    elif tl == 17:
        return (
            _cmd_eq_ci(tp, tl, "georadiusbymember")
        )
    return False


def command_is_noscript(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int,
                        sp: Pointer[UInt8, MutUntrackedOrigin], sl: Int) -> Bool:
    """True when a script's redis.call() may not run this command (#36): real
    Redis's `noscript` flag. `sp`/`sl` is the first argument, for the container
    commands Redis flags per subcommand; `container|*` means every subcommand
    but HELP. 42 entries.
    """
    if tl == 4:
        if (
            _cmd_eq_ci(tp, tl, "auth") or
            _cmd_eq_ci(tp, tl, "eval") or
            _cmd_eq_ci(tp, tl, "exec") or
            _cmd_eq_ci(tp, tl, "quit") or
            _cmd_eq_ci(tp, tl, "role") or
            _cmd_eq_ci(tp, tl, "save") or
            _cmd_eq_ci(tp, tl, "sync")
        ):
            return True
    elif tl == 5:
        if (
            _cmd_eq_ci(tp, tl, "debug") or
            _cmd_eq_ci(tp, tl, "fcall") or
            _cmd_eq_ci(tp, tl, "hello") or
            _cmd_eq_ci(tp, tl, "multi") or
            _cmd_eq_ci(tp, tl, "psync") or
            _cmd_eq_ci(tp, tl, "reset") or
            _cmd_eq_ci(tp, tl, "watch")
        ):
            return True
    elif tl == 6:
        if (
            _cmd_eq_ci(tp, tl, "bgsave")
        ):
            return True
    elif tl == 7:
        if (
            _cmd_eq_ci(tp, tl, "discard") or
            _cmd_eq_ci(tp, tl, "eval_ro") or
            _cmd_eq_ci(tp, tl, "evalsha") or
            _cmd_eq_ci(tp, tl, "monitor") or
            _cmd_eq_ci(tp, tl, "slaveof") or
            _cmd_eq_ci(tp, tl, "unwatch")
        ):
            return True
    elif tl == 8:
        if (
            _cmd_eq_ci(tp, tl, "failover") or
            _cmd_eq_ci(tp, tl, "fcall_ro") or
            _cmd_eq_ci(tp, tl, "replconf") or
            _cmd_eq_ci(tp, tl, "shutdown")
        ):
            return True
    elif tl == 9:
        if (
            _cmd_eq_ci(tp, tl, "replicaof") or
            _cmd_eq_ci(tp, tl, "subscribe")
        ):
            return True
    elif tl == 10:
        if (
            _cmd_eq_ci(tp, tl, "evalsha_ro") or
            _cmd_eq_ci(tp, tl, "psubscribe") or
            _cmd_eq_ci(tp, tl, "ssubscribe")
        ):
            return True
    elif tl == 11:
        if (
            _cmd_eq_ci(tp, tl, "unsubscribe")
        ):
            return True
    elif tl == 12:
        if (
            _cmd_eq_ci(tp, tl, "bgrewriteaof") or
            _cmd_eq_ci(tp, tl, "punsubscribe") or
            _cmd_eq_ci(tp, tl, "sunsubscribe")
        ):
            return True
    if _cmd_eq_ci(tp, tl, "acl"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "client"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "cluster"):
        return _cmd_eq_ci(sp, sl, "reset")
    if _cmd_eq_ci(tp, tl, "config"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "function"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "latency"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "module"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "script"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    return False


@always_inline
def command_is_denyoom(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Bool:
    """True when this command is refused while memory is over --maxmemory (gh #261).

    83 of 355 commands: real Redis's `denyoom` flag plus Pion's
    substrate ingest commands (PION_DENYOOM in tools/gen_command_table.py).
    Reads, DEL and the POP family stay served under the limit, as in Redis.
    """
    if tl == 3:
        return (
            _cmd_eq_ci(tp, tl, "set")
        )
    elif tl == 4:
        return (
            _cmd_eq_ci(tp, tl, "copy") or
            _cmd_eq_ci(tp, tl, "decr") or
            _cmd_eq_ci(tp, tl, "hset") or
            _cmd_eq_ci(tp, tl, "incr") or
            _cmd_eq_ci(tp, tl, "lset") or
            _cmd_eq_ci(tp, tl, "mset") or
            _cmd_eq_ci(tp, tl, "sadd") or
            _cmd_eq_ci(tp, tl, "sort") or
            _cmd_eq_ci(tp, tl, "vadd") or
            _cmd_eq_ci(tp, tl, "xadd") or
            _cmd_eq_ci(tp, tl, "zadd")
        )
    elif tl == 5:
        return (
            _cmd_eq_ci(tp, tl, "bitop") or
            _cmd_eq_ci(tp, tl, "hmset") or
            _cmd_eq_ci(tp, tl, "lmove") or
            _cmd_eq_ci(tp, tl, "lpush") or
            _cmd_eq_ci(tp, tl, "pfadd") or
            _cmd_eq_ci(tp, tl, "rpush") or
            _cmd_eq_ci(tp, tl, "setex") or
            _cmd_eq_ci(tp, tl, "setnx")
        )
    elif tl == 6:
        return (
            _cmd_eq_ci(tp, tl, "append") or
            _cmd_eq_ci(tp, tl, "blmove") or
            _cmd_eq_ci(tp, tl, "decrby") or
            _cmd_eq_ci(tp, tl, "geoadd") or
            _cmd_eq_ci(tp, tl, "getset") or
            _cmd_eq_ci(tp, tl, "hsetnx") or
            _cmd_eq_ci(tp, tl, "incrby") or
            _cmd_eq_ci(tp, tl, "lpushx") or
            _cmd_eq_ci(tp, tl, "msetex") or
            _cmd_eq_ci(tp, tl, "msetnx") or
            _cmd_eq_ci(tp, tl, "psetex") or
            _cmd_eq_ci(tp, tl, "rpushx") or
            _cmd_eq_ci(tp, tl, "setbit") or
            _cmd_eq_ci(tp, tl, "xsetid")
        )
    elif tl == 7:
        return (
            _cmd_eq_ci(tp, tl, "hincrby") or
            _cmd_eq_ci(tp, tl, "linsert") or
            _cmd_eq_ci(tp, tl, "pfdebug") or
            _cmd_eq_ci(tp, tl, "pfmerge") or
            _cmd_eq_ci(tp, tl, "restore") or
            _cmd_eq_ci(tp, tl, "zincrby")
        )
    elif tl == 8:
        return (
            _cmd_eq_ci(tp, tl, "bitfield") or
            _cmd_eq_ci(tp, tl, "kv.store") or
            _cmd_eq_ci(tp, tl, "setrange") or
            _cmd_eq_ci(tp, tl, "v.commit") or
            _cmd_eq_ci(tp, tl, "v.create")
        )
    elif tl == 9:
        return (
            _cmd_eq_ci(tp, tl, "ft.create") or
            _cmd_eq_ci(tp, tl, "georadius") or
            _cmd_eq_ci(tp, tl, "rpoplpush") or
            _cmd_eq_ci(tp, tl, "subscribe") or
            _cmd_eq_ci(tp, tl, "v.restore")
        )
    elif tl == 10:
        return (
            _cmd_eq_ci(tp, tl, "brpoplpush") or
            _cmd_eq_ci(tp, tl, "ft.addtext") or
            _cmd_eq_ci(tp, tl, "psubscribe") or
            _cmd_eq_ci(tp, tl, "sdiffstore") or
            _cmd_eq_ci(tp, tl, "ssubscribe") or
            _cmd_eq_ci(tp, tl, "zdiffstore")
        )
    elif tl == 11:
        return (
            _cmd_eq_ci(tp, tl, "ft.optimize") or
            _cmd_eq_ci(tp, tl, "incrbyfloat") or
            _cmd_eq_ci(tp, tl, "sinterstore") or
            _cmd_eq_ci(tp, tl, "sunionstore") or
            _cmd_eq_ci(tp, tl, "zinterstore") or
            _cmd_eq_ci(tp, tl, "zrangestore") or
            _cmd_eq_ci(tp, tl, "zunionstore")
        )
    elif tl == 12:
        return (
            _cmd_eq_ci(tp, tl, "ai.loadmodel") or
            _cmd_eq_ci(tp, tl, "attend.store") or
            _cmd_eq_ci(tp, tl, "hincrbyfloat") or
            _cmd_eq_ci(tp, tl, "v.storebatch")
        )
    elif tl == 13:
        return (
            _cmd_eq_ci(tp, tl, "attend.create")
        )
    elif tl == 14:
        return (
            _cmd_eq_ci(tp, tl, "geosearchstore") or
            _cmd_eq_ci(tp, tl, "kv.prefix.warm") or
            _cmd_eq_ci(tp, tl, "restore-asking")
        )
    elif tl == 15:
        return (
            _cmd_eq_ci(tp, tl, "ai.knn_lm.store") or
            _cmd_eq_ci(tp, tl, "moe.expert.load")
        )
    elif tl == 16:
        return (
            _cmd_eq_ci(tp, tl, "ai.knn_lm.create") or
            _cmd_eq_ci(tp, tl, "kv.prefix.commit")
        )
    elif tl == 17:
        return (
            _cmd_eq_ci(tp, tl, "ai.route.register") or
            _cmd_eq_ci(tp, tl, "georadiusbymember") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.create")
        )
    elif tl == 18:
        return (
            _cmd_eq_ci(tp, tl, "kv.prefix.register") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.setkeys") or
            _cmd_eq_ci(tp, tl, "neuron.pkm.setvals")
        )
    elif tl == 19:
        return (
            _cmd_eq_ci(tp, tl, "attend.prefix.store")
        )
    elif tl == 20:
        return (
            _cmd_eq_ci(tp, tl, "ai.knn_lm.storebatch")
        )
    return False


def command_hidden_from_monitor(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int,
                                sp: Pointer[UInt8, MutUntrackedOrigin], sl: Int) -> Bool:
    """True when MONITOR never shows this command (#39): Redis's `admin` flag.
    `sp`/`sl` is the first argument, for the container commands Redis flags per
    subcommand; `container|*` means every subcommand but HELP. 52 entries
    (tools/redis_admin.txt).
    """
    if tl == 4:
        if (
            _cmd_eq_ci(tp, tl, "save") or
            _cmd_eq_ci(tp, tl, "sync")
        ):
            return True
    elif tl == 5:
        if (
            _cmd_eq_ci(tp, tl, "debug") or
            _cmd_eq_ci(tp, tl, "psync")
        ):
            return True
    elif tl == 6:
        if (
            _cmd_eq_ci(tp, tl, "bgsave")
        ):
            return True
    elif tl == 7:
        if (
            _cmd_eq_ci(tp, tl, "monitor") or
            _cmd_eq_ci(tp, tl, "pfdebug") or
            _cmd_eq_ci(tp, tl, "slaveof")
        ):
            return True
    elif tl == 8:
        if (
            _cmd_eq_ci(tp, tl, "failover") or
            _cmd_eq_ci(tp, tl, "replconf") or
            _cmd_eq_ci(tp, tl, "shutdown")
        ):
            return True
    elif tl == 9:
        if (
            _cmd_eq_ci(tp, tl, "replicaof")
        ):
            return True
    elif tl == 10:
        if (
            _cmd_eq_ci(tp, tl, "pfselftest")
        ):
            return True
    elif tl == 12:
        if (
            _cmd_eq_ci(tp, tl, "bgrewriteaof")
        ):
            return True
    if _cmd_eq_ci(tp, tl, "acl"):
        return _cmd_eq_ci(sp, sl, "deluser") or _cmd_eq_ci(sp, sl, "dryrun") or _cmd_eq_ci(sp, sl, "getuser") or _cmd_eq_ci(sp, sl, "list") or _cmd_eq_ci(sp, sl, "load") or _cmd_eq_ci(sp, sl, "log") or _cmd_eq_ci(sp, sl, "save") or _cmd_eq_ci(sp, sl, "setuser") or _cmd_eq_ci(sp, sl, "users")
    if _cmd_eq_ci(tp, tl, "client"):
        return _cmd_eq_ci(sp, sl, "kill") or _cmd_eq_ci(sp, sl, "list") or _cmd_eq_ci(sp, sl, "no-evict") or _cmd_eq_ci(sp, sl, "pause") or _cmd_eq_ci(sp, sl, "unblock") or _cmd_eq_ci(sp, sl, "unpause")
    if _cmd_eq_ci(tp, tl, "cluster"):
        return _cmd_eq_ci(sp, sl, "addslots") or _cmd_eq_ci(sp, sl, "addslotsrange") or _cmd_eq_ci(sp, sl, "bumpepoch") or _cmd_eq_ci(sp, sl, "count-failure-reports") or _cmd_eq_ci(sp, sl, "delslots") or _cmd_eq_ci(sp, sl, "delslotsrange") or _cmd_eq_ci(sp, sl, "failover") or _cmd_eq_ci(sp, sl, "flushslots") or _cmd_eq_ci(sp, sl, "forget") or _cmd_eq_ci(sp, sl, "meet") or _cmd_eq_ci(sp, sl, "migration") or _cmd_eq_ci(sp, sl, "replicas") or _cmd_eq_ci(sp, sl, "replicate") or _cmd_eq_ci(sp, sl, "reset") or _cmd_eq_ci(sp, sl, "saveconfig") or _cmd_eq_ci(sp, sl, "set-config-epoch") or _cmd_eq_ci(sp, sl, "setslot") or _cmd_eq_ci(sp, sl, "slaves") or _cmd_eq_ci(sp, sl, "syncslots")
    if _cmd_eq_ci(tp, tl, "config"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "latency"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "module"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "slowlog"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    return False


def command_touches_keyspace(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int,
                             sp: Pointer[UInt8, MutUntrackedOrigin], sl: Int) -> Bool:
    """True when a monitoring connection may not run this command (#39):
    Redis's `readonly`, `write` or `may_replicate` flag ("Replica can't interact
    with the keyspace").
    `sp`/`sl` is the first argument, for the container commands Redis flags per
    subcommand; `container|*` means every subcommand but HELP. 214 entries
    (tools/redis_keyspace.txt).
    """
    if tl == 3:
        if (
            _cmd_eq_ci(tp, tl, "del") or
            _cmd_eq_ci(tp, tl, "get") or
            _cmd_eq_ci(tp, tl, "lcs") or
            _cmd_eq_ci(tp, tl, "set") or
            _cmd_eq_ci(tp, tl, "ttl")
        ):
            return True
    elif tl == 4:
        if (
            _cmd_eq_ci(tp, tl, "copy") or
            _cmd_eq_ci(tp, tl, "decr") or
            _cmd_eq_ci(tp, tl, "dump") or
            _cmd_eq_ci(tp, tl, "hdel") or
            _cmd_eq_ci(tp, tl, "hget") or
            _cmd_eq_ci(tp, tl, "hlen") or
            _cmd_eq_ci(tp, tl, "hset") or
            _cmd_eq_ci(tp, tl, "httl") or
            _cmd_eq_ci(tp, tl, "incr") or
            _cmd_eq_ci(tp, tl, "keys") or
            _cmd_eq_ci(tp, tl, "llen") or
            _cmd_eq_ci(tp, tl, "lpop") or
            _cmd_eq_ci(tp, tl, "lpos") or
            _cmd_eq_ci(tp, tl, "lrem") or
            _cmd_eq_ci(tp, tl, "lset") or
            _cmd_eq_ci(tp, tl, "mget") or
            _cmd_eq_ci(tp, tl, "move") or
            _cmd_eq_ci(tp, tl, "mset") or
            _cmd_eq_ci(tp, tl, "pttl") or
            _cmd_eq_ci(tp, tl, "rpop") or
            _cmd_eq_ci(tp, tl, "sadd") or
            _cmd_eq_ci(tp, tl, "scan") or
            _cmd_eq_ci(tp, tl, "sort") or
            _cmd_eq_ci(tp, tl, "spop") or
            _cmd_eq_ci(tp, tl, "srem") or
            _cmd_eq_ci(tp, tl, "type") or
            _cmd_eq_ci(tp, tl, "vadd") or
            _cmd_eq_ci(tp, tl, "vdim") or
            _cmd_eq_ci(tp, tl, "vemb") or
            _cmd_eq_ci(tp, tl, "vrem") or
            _cmd_eq_ci(tp, tl, "vsim") or
            _cmd_eq_ci(tp, tl, "xack") or
            _cmd_eq_ci(tp, tl, "xadd") or
            _cmd_eq_ci(tp, tl, "xdel") or
            _cmd_eq_ci(tp, tl, "xlen") or
            _cmd_eq_ci(tp, tl, "zadd") or
            _cmd_eq_ci(tp, tl, "zrem")
        ):
            return True
    elif tl == 5:
        if (
            _cmd_eq_ci(tp, tl, "bitop") or
            _cmd_eq_ci(tp, tl, "blpop") or
            _cmd_eq_ci(tp, tl, "brpop") or
            _cmd_eq_ci(tp, tl, "getex") or
            _cmd_eq_ci(tp, tl, "hkeys") or
            _cmd_eq_ci(tp, tl, "hmget") or
            _cmd_eq_ci(tp, tl, "hmset") or
            _cmd_eq_ci(tp, tl, "hpttl") or
            _cmd_eq_ci(tp, tl, "hscan") or
            _cmd_eq_ci(tp, tl, "hvals") or
            _cmd_eq_ci(tp, tl, "lmove") or
            _cmd_eq_ci(tp, tl, "lmpop") or
            _cmd_eq_ci(tp, tl, "lpush") or
            _cmd_eq_ci(tp, tl, "ltrim") or
            _cmd_eq_ci(tp, tl, "pfadd") or
            _cmd_eq_ci(tp, tl, "rpush") or
            _cmd_eq_ci(tp, tl, "scard") or
            _cmd_eq_ci(tp, tl, "sdiff") or
            _cmd_eq_ci(tp, tl, "setex") or
            _cmd_eq_ci(tp, tl, "setnx") or
            _cmd_eq_ci(tp, tl, "smove") or
            _cmd_eq_ci(tp, tl, "sscan") or
            _cmd_eq_ci(tp, tl, "touch") or
            _cmd_eq_ci(tp, tl, "vcard") or
            _cmd_eq_ci(tp, tl, "vinfo") or
            _cmd_eq_ci(tp, tl, "xread") or
            _cmd_eq_ci(tp, tl, "xtrim") or
            _cmd_eq_ci(tp, tl, "zcard") or
            _cmd_eq_ci(tp, tl, "zdiff") or
            _cmd_eq_ci(tp, tl, "zmpop") or
            _cmd_eq_ci(tp, tl, "zrank") or
            _cmd_eq_ci(tp, tl, "zscan")
        ):
            return True
    elif tl == 6:
        if (
            _cmd_eq_ci(tp, tl, "append") or
            _cmd_eq_ci(tp, tl, "bitpos") or
            _cmd_eq_ci(tp, tl, "blmove") or
            _cmd_eq_ci(tp, tl, "blmpop") or
            _cmd_eq_ci(tp, tl, "bzmpop") or
            _cmd_eq_ci(tp, tl, "dbsize") or
            _cmd_eq_ci(tp, tl, "decrby") or
            _cmd_eq_ci(tp, tl, "exists") or
            _cmd_eq_ci(tp, tl, "expire") or
            _cmd_eq_ci(tp, tl, "geoadd") or
            _cmd_eq_ci(tp, tl, "geopos") or
            _cmd_eq_ci(tp, tl, "getbit") or
            _cmd_eq_ci(tp, tl, "getdel") or
            _cmd_eq_ci(tp, tl, "getset") or
            _cmd_eq_ci(tp, tl, "hsetnx") or
            _cmd_eq_ci(tp, tl, "incrby") or
            _cmd_eq_ci(tp, tl, "lindex") or
            _cmd_eq_ci(tp, tl, "lolwut") or
            _cmd_eq_ci(tp, tl, "lpushx") or
            _cmd_eq_ci(tp, tl, "lrange") or
            _cmd_eq_ci(tp, tl, "msetex") or
            _cmd_eq_ci(tp, tl, "msetnx") or
            _cmd_eq_ci(tp, tl, "psetex") or
            _cmd_eq_ci(tp, tl, "rename") or
            _cmd_eq_ci(tp, tl, "rpushx") or
            _cmd_eq_ci(tp, tl, "setbit") or
            _cmd_eq_ci(tp, tl, "sinter") or
            _cmd_eq_ci(tp, tl, "strlen") or
            _cmd_eq_ci(tp, tl, "substr") or
            _cmd_eq_ci(tp, tl, "sunion") or
            _cmd_eq_ci(tp, tl, "swapdb") or
            _cmd_eq_ci(tp, tl, "unlink") or
            _cmd_eq_ci(tp, tl, "vlinks") or
            _cmd_eq_ci(tp, tl, "vrange") or
            _cmd_eq_ci(tp, tl, "xclaim") or
            _cmd_eq_ci(tp, tl, "xdelex") or
            _cmd_eq_ci(tp, tl, "xrange") or
            _cmd_eq_ci(tp, tl, "xsetid") or
            _cmd_eq_ci(tp, tl, "zcount") or
            _cmd_eq_ci(tp, tl, "zinter") or
            _cmd_eq_ci(tp, tl, "zrange") or
            _cmd_eq_ci(tp, tl, "zscore") or
            _cmd_eq_ci(tp, tl, "zunion")
        ):
            return True
    elif tl == 7:
        if (
            _cmd_eq_ci(tp, tl, "eval_ro") or
            _cmd_eq_ci(tp, tl, "flushdb") or
            _cmd_eq_ci(tp, tl, "geodist") or
            _cmd_eq_ci(tp, tl, "geohash") or
            _cmd_eq_ci(tp, tl, "hexists") or
            _cmd_eq_ci(tp, tl, "hexpire") or
            _cmd_eq_ci(tp, tl, "hgetall") or
            _cmd_eq_ci(tp, tl, "hincrby") or
            _cmd_eq_ci(tp, tl, "hstrlen") or
            _cmd_eq_ci(tp, tl, "linsert") or
            _cmd_eq_ci(tp, tl, "migrate") or
            _cmd_eq_ci(tp, tl, "persist") or
            _cmd_eq_ci(tp, tl, "pexpire") or
            _cmd_eq_ci(tp, tl, "pfcount") or
            _cmd_eq_ci(tp, tl, "pfdebug") or
            _cmd_eq_ci(tp, tl, "pfmerge") or
            _cmd_eq_ci(tp, tl, "restore") or
            _cmd_eq_ci(tp, tl, "sort_ro") or
            _cmd_eq_ci(tp, tl, "xackdel") or
            _cmd_eq_ci(tp, tl, "zincrby") or
            _cmd_eq_ci(tp, tl, "zmscore") or
            _cmd_eq_ci(tp, tl, "zpopmax") or
            _cmd_eq_ci(tp, tl, "zpopmin")
        ):
            return True
    elif tl == 8:
        if (
            _cmd_eq_ci(tp, tl, "bitcount") or
            _cmd_eq_ci(tp, tl, "bitfield") or
            _cmd_eq_ci(tp, tl, "bzpopmax") or
            _cmd_eq_ci(tp, tl, "bzpopmin") or
            _cmd_eq_ci(tp, tl, "expireat") or
            _cmd_eq_ci(tp, tl, "fcall_ro") or
            _cmd_eq_ci(tp, tl, "flushall") or
            _cmd_eq_ci(tp, tl, "getrange") or
            _cmd_eq_ci(tp, tl, "hpersist") or
            _cmd_eq_ci(tp, tl, "hpexpire") or
            _cmd_eq_ci(tp, tl, "renamenx") or
            _cmd_eq_ci(tp, tl, "setrange") or
            _cmd_eq_ci(tp, tl, "smembers") or
            _cmd_eq_ci(tp, tl, "vgetattr") or
            _cmd_eq_ci(tp, tl, "vsetattr") or
            _cmd_eq_ci(tp, tl, "xpending") or
            _cmd_eq_ci(tp, tl, "zrevrank")
        ):
            return True
    elif tl == 9:
        if (
            _cmd_eq_ci(tp, tl, "georadius") or
            _cmd_eq_ci(tp, tl, "geosearch") or
            _cmd_eq_ci(tp, tl, "hexpireat") or
            _cmd_eq_ci(tp, tl, "pexpireat") or
            _cmd_eq_ci(tp, tl, "randomkey") or
            _cmd_eq_ci(tp, tl, "rpoplpush") or
            _cmd_eq_ci(tp, tl, "sismember") or
            _cmd_eq_ci(tp, tl, "vismember") or
            _cmd_eq_ci(tp, tl, "xrevrange") or
            _cmd_eq_ci(tp, tl, "zlexcount") or
            _cmd_eq_ci(tp, tl, "zrevrange")
        ):
            return True
    elif tl == 10:
        if (
            _cmd_eq_ci(tp, tl, "brpoplpush") or
            _cmd_eq_ci(tp, tl, "evalsha_ro") or
            _cmd_eq_ci(tp, tl, "expiretime") or
            _cmd_eq_ci(tp, tl, "hpexpireat") or
            _cmd_eq_ci(tp, tl, "hrandfield") or
            _cmd_eq_ci(tp, tl, "sdiffstore") or
            _cmd_eq_ci(tp, tl, "sintercard") or
            _cmd_eq_ci(tp, tl, "smismember") or
            _cmd_eq_ci(tp, tl, "xautoclaim") or
            _cmd_eq_ci(tp, tl, "xreadgroup") or
            _cmd_eq_ci(tp, tl, "zdiffstore") or
            _cmd_eq_ci(tp, tl, "zintercard")
        ):
            return True
    elif tl == 11:
        if (
            _cmd_eq_ci(tp, tl, "bitfield_ro") or
            _cmd_eq_ci(tp, tl, "hexpiretime") or
            _cmd_eq_ci(tp, tl, "incrbyfloat") or
            _cmd_eq_ci(tp, tl, "pexpiretime") or
            _cmd_eq_ci(tp, tl, "sinterstore") or
            _cmd_eq_ci(tp, tl, "srandmember") or
            _cmd_eq_ci(tp, tl, "sunionstore") or
            _cmd_eq_ci(tp, tl, "vrandmember") or
            _cmd_eq_ci(tp, tl, "zinterstore") or
            _cmd_eq_ci(tp, tl, "zrandmember") or
            _cmd_eq_ci(tp, tl, "zrangebylex") or
            _cmd_eq_ci(tp, tl, "zrangestore") or
            _cmd_eq_ci(tp, tl, "zunionstore")
        ):
            return True
    elif tl == 12:
        if (
            _cmd_eq_ci(tp, tl, "georadius_ro") or
            _cmd_eq_ci(tp, tl, "hincrbyfloat") or
            _cmd_eq_ci(tp, tl, "hpexpiretime")
        ):
            return True
    elif tl == 13:
        if (
            _cmd_eq_ci(tp, tl, "zrangebyscore")
        ):
            return True
    elif tl == 14:
        if (
            _cmd_eq_ci(tp, tl, "geosearchstore") or
            _cmd_eq_ci(tp, tl, "restore-asking") or
            _cmd_eq_ci(tp, tl, "zremrangebylex") or
            _cmd_eq_ci(tp, tl, "zrevrangebylex")
        ):
            return True
    elif tl == 15:
        if (
            _cmd_eq_ci(tp, tl, "zremrangebyrank")
        ):
            return True
    elif tl == 16:
        if (
            _cmd_eq_ci(tp, tl, "zremrangebyscore") or
            _cmd_eq_ci(tp, tl, "zrevrangebyscore")
        ):
            return True
    elif tl == 17:
        if (
            _cmd_eq_ci(tp, tl, "georadiusbymember")
        ):
            return True
    elif tl == 20:
        if (
            _cmd_eq_ci(tp, tl, "georadiusbymember_ro")
        ):
            return True
    if _cmd_eq_ci(tp, tl, "function"):
        return _cmd_eq_ci(sp, sl, "delete") or _cmd_eq_ci(sp, sl, "flush") or _cmd_eq_ci(sp, sl, "load") or _cmd_eq_ci(sp, sl, "restore")
    if _cmd_eq_ci(tp, tl, "memory"):
        return _cmd_eq_ci(sp, sl, "usage")
    if _cmd_eq_ci(tp, tl, "object"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "xgroup"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    if _cmd_eq_ci(tp, tl, "xinfo"):
        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")
    return False


def command_monitor_first(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Bool:
    """True when MONITOR shows this command BEFORE it runs (#39): Redis's
    `skip_monitor` flag, the script commands, so that what a script calls
    follows the script's own line. 6 entries (tools/redis_skip_monitor.txt).
    """
    if _cmd_eq_ci(tp, tl, "eval"):
        return True
    if _cmd_eq_ci(tp, tl, "eval_ro"):
        return True
    if _cmd_eq_ci(tp, tl, "evalsha"):
        return True
    if _cmd_eq_ci(tp, tl, "evalsha_ro"):
        return True
    if _cmd_eq_ci(tp, tl, "fcall"):
        return True
    if _cmd_eq_ci(tp, tl, "fcall_ro"):
        return True
    return False
