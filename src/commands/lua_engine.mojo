"""Lua scripting: EVAL, EVALSHA, EVAL_RO, EVALSHA_RO, SCRIPT, FUNCTION, FCALL,
FCALL_RO.

The engine is src/ffi/lua_wrap.c. A script's redis.call() runs synchronously
through the server's own slow-path dispatcher (SlowPathHandler.script_dispatch,
reached through the @export pion_script_dispatch in main.mojo), so it runs
every command with its options, errors and WAL records (#36). The C side builds
each reply; these handlers parse the arguments as Redis does and copy it out.

One LuaEngine per worker, shared-nothing. FUNCTION libraries are persisted as
WAL records 35 (LOAD), 36 (DELETE) and 37 (FLUSH), and written into snapshots.
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory import alloc, unsafe_memcpy
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call
from src.common.utils import arg_eq, parse_int64_strict, bytes_to_string
from src.network.response_writer import ResponseWriter
from src.network.resp3 import RESP3Token
from src.io.wal import WAL


struct LuaEngine(Movable):
    """Per-worker Lua states (EVAL scripts and FUNCTION libraries)."""

    var state: Pointer[NoneType, MutUntrackedOrigin]  # PionLuaState*
    var enabled: Bool

    def __init__(out self):
        # Negative limits take the process-wide --lua-memory-limit and
        # --lua-time-limit that main set before the workers started.
        self.state = external_call["pion_lua_new_state",
            Pointer[NoneType, MutUntrackedOrigin]](Int64(-1), Int64(-1))
        self.enabled = is_not_null(self.state)

    def __init__(out self, *, deinit take: Self):
        self.state = take.state
        self.enabled = take.enabled


def _help_script() -> List[String]:
    return [
        "SCRIPT <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
        "DEBUG (YES|SYNC|NO)",
        "    Set the debug mode for subsequent scripts executed.",
        "EXISTS <sha1> [<sha1> ...]",
        "    Return information about the existence of the scripts in the script cache.",
        "FLUSH [ASYNC|SYNC]",
        "    Flush the Lua scripts cache. Very dangerous on replicas.",
        "    When called without the optional mode argument, the behavior is determined by the",
        "    lazyfree-lazy-user-flush configuration directive. Valid modes are:",
        "    * ASYNC: Asynchronously flush the scripts cache.",
        "    * SYNC: Synchronously flush the scripts cache.",
        "KILL",
        "    Kill the currently executing Lua script.",
        "LOAD <script>",
        "    Load a script into the scripts cache without executing it.",
        "HELP",
        "    Print this help.",
    ]


def _help_function() -> List[String]:
    return [
        "FUNCTION <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
        "LOAD [REPLACE] <FUNCTION CODE>",
        "    Create a new library with the given library name and code.",
        "DELETE <LIBRARY NAME>",
        "    Delete the given library.",
        "LIST [LIBRARYNAME PATTERN] [WITHCODE]",
        "    Return general information on all the libraries:",
        "    * Library name",
        "    * The engine used to run the Library",
        "    * Functions list",
        "    * Library code (if WITHCODE is given)",
        "    It also possible to get only function that matches a pattern using LIBRARYNAME argument.",
        "STATS",
        "    Return information about the current function running:",
        "    * Function name",
        "    * Command used to run the function",
        "    * Duration in MS that the function is running",
        "    If no function is running, return nil",
        "    In addition, returns a list of available engines.",
        "KILL",
        "    Kill the current running function.",
        "FLUSH [ASYNC|SYNC]",
        "    Delete all the libraries.",
        "    When called without the optional mode argument, the behavior is determined by the",
        "    lazyfree-lazy-user-flush configuration directive. Valid modes are:",
        "    * ASYNC: Asynchronously flush the libraries.",
        "    * SYNC: Synchronously flush the libraries.",
        "DUMP",
        "    Return a serialized payload representing the current libraries, can be restored using FUNCTION RESTORE command",
        "RESTORE <PAYLOAD> [FLUSH|APPEND|REPLACE]",
        "    Restore the libraries represented by the given payload, it is possible to give a restore policy to",
        "    control how to handle existing libraries (default APPEND):",
        "    * FLUSH: delete all existing libraries.",
        "    * APPEND: appends the restored libraries to the existing libraries. On collision, abort.",
        "    * REPLACE: appends the restored libraries to the existing libraries, On collision, replace the old",
        "      libraries with the new libraries (notice that even on this option there is a chance of failure",
        "      in case of functions name collision with another library).",
        "HELP",
        "    Print this help.",
    ]


# ── Helpers ──

def _emit_out(lua: Pointer[LuaEngine, MutUntrackedOrigin], n: Int64, mut writer: ResponseWriter):
    """Copy the reply the C side built (pion_lua_out) into the writer."""
    if n > 0:
        var p = external_call["pion_lua_out", Pointer[UInt8, MutUntrackedOrigin]](lua[].state)
        writer.append_to_response(p, Int(n))
    else:
        writer.append_error_response("ERR internal error building the script reply")


def _help(lines: List[String], mut writer: ResponseWriter):
    writer.append_array_header(len(lines))
    for k in range(len(lines)):
        writer.append_status_response(lines[k])


def _tok_text(t: RESP3Token) -> String:
    return bytes_to_string(t.ptr, t.length)


def _subcommand_arity_error(container: StringLiteral, sub: RESP3Token, mut writer: ResponseWriter):
    var name = _tok_text(sub).lower()
    writer.append_error_response("ERR wrong number of arguments for '" + String(container) + "|"
                                 + name + "' command")


def _numkeys(tokens: Pointer[RESP3Token, MutUntrackedOrigin], idx: Int, avail: Int,
             mut writer: ResponseWriter, fcall: Bool) -> Int:
    """numkeys as evalGenericCommand / fcallCommandGeneric read it: the
    number, or -1 with the error written."""
    var t = tokens[unsafe_offset=idx]
    var p = parse_int64_strict(t.ptr, t.length)
    if not p.ok:
        if fcall:
            writer.append_error_response("ERR Bad number of keys provided")
        else:
            writer.append_error_response("ERR value is not an integer or out of range")
        return -1
    if p.value > Int64(avail):
        writer.append_error_response("ERR Number of keys can't be greater than number of args")
        return -1
    if p.value < 0:
        writer.append_error_response("ERR Number of keys can't be negative")
        return -1
    return Int(p.value)


struct _Args(Movable):
    """KEYS and ARGV as pointer/length arrays for the C side (heap: never a
    stack buffer across an out-of-line call, gh #349)."""
    var kp: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    var kl: Pointer[Int64, MutUntrackedOrigin]
    var nk: Int
    var ap: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    var al: Pointer[Int64, MutUntrackedOrigin]
    var na: Int

    def __init__(out self, tokens: Pointer[RESP3Token, MutUntrackedOrigin], first: Int, nk: Int, end: Int):
        self.nk = nk
        self.na = end - first - nk
        self.kp = alloc[Pointer[UInt8, MutUntrackedOrigin]](nk + 1)
        self.kl = alloc[Int64](nk + 1)
        self.ap = alloc[Pointer[UInt8, MutUntrackedOrigin]](self.na + 1)
        self.al = alloc[Int64](self.na + 1)
        for k in range(nk):
            self.kp[unsafe_offset=k] = tokens[unsafe_offset=first + k].ptr
            self.kl[unsafe_offset=k] = Int64(tokens[unsafe_offset=first + k].length)
        for k in range(self.na):
            self.ap[unsafe_offset=k] = tokens[unsafe_offset=first + nk + k].ptr
            self.al[unsafe_offset=k] = Int64(tokens[unsafe_offset=first + nk + k].length)

    def __init__(out self, *, deinit take: Self):
        self.kp = take.kp
        self.kl = take.kl
        self.nk = take.nk
        self.ap = take.ap
        self.al = take.al
        self.na = take.na

    def free(mut self):
        self.kp.unsafe_free()
        self.kl.unsafe_free()
        self.ap.unsafe_free()
        self.al.unsafe_free()


def _run_script(lua: Pointer[LuaEngine, MutUntrackedOrigin], sha: Pointer[UInt8, MutUntrackedOrigin],
                tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                numkeys: Int, ro: Bool, mut writer: ResponseWriter,
                host: Pointer[NoneType, MutUntrackedOrigin]):
    var a = _Args(tokens, i + 3, numkeys, num_tokens)
    external_call["pion_lua_set_host", NoneType](lua[].state, host)
    var n = external_call["pion_lua_run_script", Int64](
        lua[].state, sha, a.kp, a.kl, Int64(a.nk), a.ap, a.al, Int64(a.na),
        Int64(1 if ro else 0), Int64(Int(writer.proto)))
    a.free()
    _emit_out(lua, n, writer)


# ── EVAL / EVALSHA (and _RO) ──

def handle_eval(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
    host: Pointer[NoneType, MutUntrackedOrigin],
    ro: Bool,
) -> Int:
    """EVAL / EVAL_RO script numkeys [key ...] [arg ...]."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra
    if num_tokens - i < 3:
        writer.append_error_response("ERR wrong number of arguments for '" + String("eval_ro" if ro else "eval") + "' command")
        return extra
    var numkeys = _numkeys(tokens, i + 2, num_tokens - i - 3, writer, False)
    if numkeys < 0:
        return extra
    var sha = alloc[UInt8](41)
    var st = external_call["pion_lua_load_script", Int32](
        lua[].state, tokens[unsafe_offset=i + 1].ptr, Int64(tokens[unsafe_offset=i + 1].length), sha)
    if st < 0:
        sha.unsafe_free()
        _emit_out(lua, external_call["pion_lua_out_len", Int64](lua[].state), writer)
        return extra
    _run_script(lua, sha, tokens, i, num_tokens, numkeys, ro, writer, host)
    sha.unsafe_free()
    return extra


def handle_evalsha(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
    host: Pointer[NoneType, MutUntrackedOrigin],
    ro: Bool,
) -> Int:
    """EVALSHA / EVALSHA_RO sha1 numkeys [key ...] [arg ...]."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra
    if num_tokens - i < 3:
        writer.append_error_response("ERR wrong number of arguments for '" + String("evalsha_ro" if ro else "evalsha") + "' command")
        return extra
    var numkeys = _numkeys(tokens, i + 2, num_tokens - i - 3, writer, False)
    if numkeys < 0:
        return extra
    var t = tokens[unsafe_offset=i + 1]
    if t.length != 40:
        writer.append_error_response("NOSCRIPT No matching script. Please use EVAL.")
        return extra
    var sha = alloc[UInt8](41)
    unsafe_memcpy(dest=sha, src=t.ptr, count=40)
    sha[unsafe_offset=40] = 0
    _run_script(lua, sha, tokens, i, num_tokens, numkeys, ro, writer, host)
    sha.unsafe_free()
    return extra


# ── SCRIPT ──

def handle_script(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
) -> Int:
    """SCRIPT LOAD | EXISTS | FLUSH | KILL | DEBUG | HELP, as Redis answers them."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra
    if num_tokens - i < 2:
        writer.append_error_response("ERR wrong number of arguments for 'script' command")
        return extra
    var sub = tokens[unsafe_offset=i + 1]
    var argc = num_tokens - i
    if arg_eq(sub.ptr, sub.length, "load"):
        if argc != 3:
            _subcommand_arity_error("script", sub, writer)
            return extra
        var sha = alloc[UInt8](41)
        var st = external_call["pion_lua_load_script", Int32](
            lua[].state, tokens[unsafe_offset=i + 2].ptr, Int64(tokens[unsafe_offset=i + 2].length), sha)
        if st < 0:
            _emit_out(lua, external_call["pion_lua_out_len", Int64](lua[].state), writer)
        else:
            writer.append_bulk_string_response(sha, 40)
        sha.unsafe_free()
    elif arg_eq(sub.ptr, sub.length, "exists"):
        if argc < 3:
            _subcommand_arity_error("script", sub, writer)
            return extra
        writer.append_array_header(argc - 2)
        var sha = alloc[UInt8](41)
        for a in range(i + 2, num_tokens):
            var t = tokens[unsafe_offset=a]
            var found = Int32(0)
            if t.length == 40:
                unsafe_memcpy(dest=sha, src=t.ptr, count=40)
                sha[unsafe_offset=40] = 0
                found = external_call["pion_lua_script_exists", Int32](lua[].state, sha)
            writer.append_int_response(Int64(found))
        sha.unsafe_free()
    elif arg_eq(sub.ptr, sub.length, "flush"):
        var ok = argc == 2
        if argc == 3:
            var m = tokens[unsafe_offset=i + 2]
            ok = arg_eq(m.ptr, m.length, "sync") or arg_eq(m.ptr, m.length, "async")
        if not ok:
            writer.append_error_response("ERR SCRIPT FLUSH only support SYNC|ASYNC option")
            return extra
        external_call["pion_lua_script_flush", NoneType](lua[].state)
        writer.append_ok_response()
    elif arg_eq(sub.ptr, sub.length, "kill"):
        if argc != 2:
            _subcommand_arity_error("script", sub, writer)
            return extra
        # A worker runs one thing at a time: nothing is running while this
        # command is being answered (see --lua-time-limit).
        writer.append_error_response("NOTBUSY No scripts in execution right now.")
    elif arg_eq(sub.ptr, sub.length, "debug"):
        if argc != 3:
            _subcommand_arity_error("script", sub, writer)
            return extra
        var m = tokens[unsafe_offset=i + 2]
        if arg_eq(m.ptr, m.length, "no"):
            writer.append_ok_response()
        elif arg_eq(m.ptr, m.length, "yes") or arg_eq(m.ptr, m.length, "sync"):
            writer.append_error_response("ERR SCRIPT DEBUG YES|SYNC is not supported: Pion has no Lua debugger")
        else:
            writer.append_error_response("ERR Use SCRIPT DEBUG YES/SYNC/NO")
    elif arg_eq(sub.ptr, sub.length, "help"):
        if argc != 2:
            _subcommand_arity_error("script", sub, writer)
            return extra
        _help(_help_script(), writer)
    else:
        writer.append_error_response("ERR unknown subcommand '" + _tok_text(sub) + "'. Try SCRIPT HELP.")
    return extra


# ── FUNCTION ──

def _log_library(wal: Pointer[WAL, MutUntrackedOrigin], code: Pointer[UInt8, MutUntrackedOrigin], clen: Int):
    """WAL record 35: the library's name as the key, its code as the value."""
    if is_null(wal):
        return
    var name = alloc[Pointer[UInt8, MutUntrackedOrigin]](1)
    var nlen = alloc[Int64](1)
    nlen[unsafe_offset=0] = 0
    if external_call["pion_lua_library_name_of", Int64](code, Int64(clen), name, nlen) == 1:
        _ = wal[].append_kv(35, name[unsafe_offset=0], Int(nlen[unsafe_offset=0]), code, clen)
    name.unsafe_free()
    nlen.unsafe_free()


def _restore(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
             mut writer: ResponseWriter, lua: Pointer[LuaEngine, MutUntrackedOrigin],
             wal: Pointer[WAL, MutUntrackedOrigin]):
    """FUNCTION RESTORE payload [FLUSH|APPEND|REPLACE]: all or nothing."""
    var policy = 0   # 0 APPEND, 1 REPLACE, 2 FLUSH
    if num_tokens - i == 4:
        var m = tokens[unsafe_offset=i + 3]
        if arg_eq(m.ptr, m.length, "flush"):
            policy = 2
        elif arg_eq(m.ptr, m.length, "replace"):
            policy = 1
        elif not arg_eq(m.ptr, m.length, "append"):
            writer.append_error_response("ERR Wrong restore policy given, value should be either FLUSH, APPEND or REPLACE.")
            return
    elif num_tokens - i > 4:
        writer.append_error_response("ERR Wrong restore policy given, value should be either FLUSH, APPEND or REPLACE.")
        return
    var pt = tokens[unsafe_offset=i + 2]
    var count = external_call["pion_lua_dump_count", Int64](pt.ptr, Int64(pt.length))
    if count < 0:
        writer.append_error_response("ERR DUMP payload version or checksum are wrong")
        return
    var ep = alloc[Int64](1)
    var namep = alloc[Pointer[UInt8, MutUntrackedOrigin]](1)
    var nlen = alloc[Int64](1)
    if policy == 0:
        # APPEND: a library that already exists aborts the whole restore
        for k in range(Int(count)):
            var code = external_call["pion_lua_dump_entry", Pointer[UInt8, MutUntrackedOrigin]](
                pt.ptr, Int64(pt.length), Int64(k), ep)
            if external_call["pion_lua_library_name_of", Int64](code, ep[unsafe_offset=0], namep, nlen) == 1 \
               and external_call["pion_lua_library_exists", Int64](
                   lua[].state, namep[unsafe_offset=0], nlen[unsafe_offset=0]) == 1:
                var nm = bytes_to_string(namep[unsafe_offset=0], Int(nlen[unsafe_offset=0]))
                writer.append_error_response("ERR Library " + nm + " already exists")
                ep.unsafe_free(); namep.unsafe_free(); nlen.unsafe_free()
                return
    # Keep the current libraries, so a failure can put them back.
    var saved_n = external_call["pion_lua_function_dump", Int64](lua[].state)
    var saved = alloc[UInt8](Int(saved_n) + 1)
    unsafe_memcpy(dest=saved, src=external_call["pion_lua_out", Pointer[UInt8, MutUntrackedOrigin]](lua[].state),
                  count=Int(saved_n))
    if policy == 2:
        external_call["pion_lua_function_flush", NoneType](lua[].state)
    var failed = False
    for k in range(Int(count)):
        var code = external_call["pion_lua_dump_entry", Pointer[UInt8, MutUntrackedOrigin]](
            pt.ptr, Int64(pt.length), Int64(k), ep)
        if external_call["pion_lua_function_load", Int64](lua[].state, code, ep[unsafe_offset=0], Int64(1)) != 1:
            failed = True
            break
    if failed:
        # the error the failed load built, then the previous libraries back
        var en = external_call["pion_lua_out_len", Int64](lua[].state)
        var err = alloc[UInt8](Int(en) + 1)
        unsafe_memcpy(dest=err, src=external_call["pion_lua_out", Pointer[UInt8, MutUntrackedOrigin]](lua[].state),
                      count=Int(en))
        external_call["pion_lua_function_flush", NoneType](lua[].state)
        var sc = external_call["pion_lua_dump_count", Int64](saved, saved_n)
        for k in range(Int(sc)):
            var code = external_call["pion_lua_dump_entry", Pointer[UInt8, MutUntrackedOrigin]](saved, saved_n, Int64(k), ep)
            _ = external_call["pion_lua_function_load", Int64](lua[].state, code, ep[unsafe_offset=0], Int64(1))
        writer.append_to_response(err, Int(en))
        err.unsafe_free()
    else:
        if is_not_null(wal):
            if policy == 2:
                _ = wal[].append(37, null_ptr[UInt8, MutUntrackedOrigin](), 0)
            for k in range(Int(count)):
                var code = external_call["pion_lua_dump_entry", Pointer[UInt8, MutUntrackedOrigin]](
                    pt.ptr, Int64(pt.length), Int64(k), ep)
                _log_library(wal, code, Int(ep[unsafe_offset=0]))
        writer.append_ok_response()
    saved.unsafe_free()
    ep.unsafe_free(); namep.unsafe_free(); nlen.unsafe_free()


def handle_function(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
    wal: Pointer[WAL, MutUntrackedOrigin],
) -> Int:
    """FUNCTION LOAD | DELETE | FLUSH | LIST | STATS | DUMP | RESTORE | KILL | HELP.
    Every change to the libraries is written to the WAL."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra
    if num_tokens - i < 2:
        writer.append_error_response("ERR wrong number of arguments for 'function' command")
        return extra
    var sub = tokens[unsafe_offset=i + 1]
    var argc = num_tokens - i
    var proto = Int64(Int(writer.proto))
    if arg_eq(sub.ptr, sub.length, "load"):
        if argc < 3:
            _subcommand_arity_error("function", sub, writer)
            return extra
        var replace = False
        var pos = i + 2
        while pos < num_tokens - 1:
            var t = tokens[unsafe_offset=pos]
            pos += 1
            if arg_eq(t.ptr, t.length, "replace"):
                replace = True
                continue
            writer.append_error_response("ERR Unknown option given: " + _tok_text(t))
            return extra
        var code = tokens[unsafe_offset=pos]
        var ok = external_call["pion_lua_function_load", Int64](
            lua[].state, code.ptr, Int64(code.length), Int64(1 if replace else 0))
        _emit_out(lua, external_call["pion_lua_out_len", Int64](lua[].state), writer)
        if ok == 1:
            _log_library(wal, code.ptr, code.length)
    elif arg_eq(sub.ptr, sub.length, "delete"):
        if argc != 3:
            _subcommand_arity_error("function", sub, writer)
            return extra
        var nm = tokens[unsafe_offset=i + 2]
        if external_call["pion_lua_function_delete", Int64](lua[].state, nm.ptr, Int64(nm.length)) == 1:
            if is_not_null(wal):
                _ = wal[].append(36, nm.ptr, nm.length)
            writer.append_ok_response()
        else:
            writer.append_error_response("ERR Library not found")
    elif arg_eq(sub.ptr, sub.length, "flush"):
        var ok = argc == 2
        if argc == 3:
            var m = tokens[unsafe_offset=i + 2]
            ok = arg_eq(m.ptr, m.length, "sync") or arg_eq(m.ptr, m.length, "async")
        if not ok:
            writer.append_error_response("ERR FUNCTION FLUSH only supports SYNC|ASYNC option")
            return extra
        external_call["pion_lua_function_flush", NoneType](lua[].state)
        if is_not_null(wal):
            _ = wal[].append(37, null_ptr[UInt8, MutUntrackedOrigin](), 0)
        writer.append_ok_response()
    elif arg_eq(sub.ptr, sub.length, "list"):
        var withcode = False
        var have_pat = False
        var pat = null_ptr[UInt8, MutUntrackedOrigin]()
        var plen = 0
        var a = i + 2
        while a < num_tokens:
            var t = tokens[unsafe_offset=a]
            if not withcode and arg_eq(t.ptr, t.length, "withcode"):
                withcode = True
                a += 1
                continue
            if not have_pat and arg_eq(t.ptr, t.length, "libraryname"):
                if a >= num_tokens - 1:
                    writer.append_error_response("ERR library name argument was not given")
                    return extra
                have_pat = True
                pat = tokens[unsafe_offset=a + 1].ptr
                plen = tokens[unsafe_offset=a + 1].length
                a += 2
                continue
            writer.append_error_response("ERR Unknown argument " + _tok_text(t))
            return extra
        var n = external_call["pion_lua_function_list", Int64](
            lua[].state, pat, Int64(plen if have_pat else -1), Int64(1 if withcode else 0), proto)
        _emit_out(lua, n, writer)
    elif arg_eq(sub.ptr, sub.length, "stats"):
        if argc != 2:
            _subcommand_arity_error("function", sub, writer)
            return extra
        _emit_out(lua, external_call["pion_lua_function_stats", Int64](lua[].state, proto), writer)
    elif arg_eq(sub.ptr, sub.length, "dump"):
        if argc != 2:
            _subcommand_arity_error("function", sub, writer)
            return extra
        var n = external_call["pion_lua_function_dump", Int64](lua[].state)
        writer.append_bulk_string_response(
            external_call["pion_lua_out", Pointer[UInt8, MutUntrackedOrigin]](lua[].state), Int(n))
    elif arg_eq(sub.ptr, sub.length, "restore"):
        if argc < 3:
            _subcommand_arity_error("function", sub, writer)
            return extra
        _restore(tokens, i, num_tokens, writer, lua, wal)
    elif arg_eq(sub.ptr, sub.length, "kill"):
        if argc != 2:
            _subcommand_arity_error("function", sub, writer)
            return extra
        writer.append_error_response("NOTBUSY No scripts in execution right now.")
    elif arg_eq(sub.ptr, sub.length, "help"):
        if argc != 2:
            _subcommand_arity_error("function", sub, writer)
            return extra
        _help(_help_function(), writer)
    else:
        writer.append_error_response("ERR unknown subcommand '" + _tok_text(sub) + "'. Try FUNCTION HELP.")
    return extra


# ── FCALL / FCALL_RO ──

def handle_fcall(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    mut writer: ResponseWriter,
    lua: Pointer[LuaEngine, MutUntrackedOrigin],
    host: Pointer[NoneType, MutUntrackedOrigin],
    ro: Bool,
) -> Int:
    """FCALL / FCALL_RO function numkeys [key ...] [arg ...]."""
    var extra = num_tokens - i - 1
    if is_null(lua) or not lua[].enabled:
        writer.append_error_response("ERR Lua scripting is not available")
        return extra
    if num_tokens - i < 3:
        writer.append_error_response("ERR wrong number of arguments for '" + String("fcall_ro" if ro else "fcall") + "' command")
        return extra
    var fname = tokens[unsafe_offset=i + 1]
    # Redis looks the function up before it reads numkeys.
    if external_call["pion_lua_function_exists", Int64](lua[].state, fname.ptr, Int64(fname.length)) != 1:
        writer.append_error_response("ERR Function not found")
        return extra
    var numkeys = _numkeys(tokens, i + 2, num_tokens - i - 3, writer, True)
    if numkeys < 0:
        return extra
    var a = _Args(tokens, i + 3, numkeys, num_tokens)
    external_call["pion_lua_set_host", NoneType](lua[].state, host)
    var n = external_call["pion_lua_run_function", Int64](
        lua[].state, fname.ptr, Int64(fname.length), a.kp, a.kl, Int64(a.nk), a.ap, a.al, Int64(a.na),
        Int64(1 if ro else 0), Int64(Int(writer.proto)))
    a.free()
    _emit_out(lua, n, writer)
    return extra
