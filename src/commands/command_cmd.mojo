"""COMMAND and its subcommands (#47).

Every subcommand but COUNT answered an empty array, bare COMMAND too: a
cluster client or a proxy that learns a command's keys from COMMAND INFO or
COMMAND GETKEYS learned nothing. They now answer from the generated tables in
command_info.mojo:

  * COMMAND, COMMAND INFO: Redis's own entry for a command Redis has (as
    Redis encodes it in RESP2 or RESP3), and for a Pion-only command an entry
    built from Pion's table: its arity, `write`/`denyoom` or `readonly`, no
    key positions, @read/@write and @slow.
  * COMMAND LIST [FILTERBY MODULE <m> | ACLCAT <c> | PATTERN <glob>].
    MODULE `search` lists the FT.* commands, the module MODULE LIST names.
  * COMMAND GETKEYS / GETKEYSANDFLAGS: Redis's getKeysUsingKeySpecs over the
    command's key specs, and for the commands whose specs are incomplete,
    unknown or flag-variable, Redis's own procedures (_proc_keys).
  * COMMAND DOCS: each command's group; Pion carries no command documentation.
"""

from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.collections import List
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.common.utils import arg_eq, _glob_match
from src.commands.command_table import (PION_COMMAND_COUNT, command_exists, command_arity,
                                        command_is_write, command_is_denyoom)
from src.commands.command_info import (CMDINFO2, CMDINFO3, cmdinfo_span, cmd_keyspecs, cmd_doc_group,
                                       ACL_CATEGORIES, acl_category_commands, cmd_no_mandatory_keys,
                                       PION_COMMAND_NAMES, PION_SUBCOMMAND_NAMES)


def _words(s: StaticString) -> List[String]:
    """The space-separated words of `s`."""
    var out = List[String]()
    var p = s.unsafe_ptr()
    var n = s.byte_length()
    var start = 0
    for k in range(n + 1):
        if k == n or p[k] == 32:
            if k > start:
                out.append(String(StringSpan[MutUntrackedOrigin](
                    unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                        unsafe_ptr=Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(p) + start),
                        length=k - start))))
            start = k + 1
    return out^


def _pp(s: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(s.unsafe_ptr()))


def write_command_info(mut writer: ResponseWriter, tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Bool:
    """One command's COMMAND INFO entry; False (nothing written) when there is
    no such command."""
    if not command_exists(tp, tl):
        return False
    var sp = cmdinfo_span(tp, tl)
    if sp.len2 > 0:
        if writer.proto == 3:
            writer.append_to_response(Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=Int(CMDINFO3.unsafe_ptr()) + sp.off3), sp.len3)
        else:
            writer.append_to_response(Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=Int(CMDINFO2.unsafe_ptr()) + sp.off2), sp.len2)
        return True
    # A Pion-only command: what Pion's own table knows.
    var lower = String("")
    for k in range(tl):
        var c = tp[k]
        if c >= 65 and c <= 90:
            c += 32
        lower += chr(Int(c))
    var write = command_is_write(tp, tl)
    var denyoom = command_is_denyoom(tp, tl)
    writer.append_array_header(10)
    writer.append_bulk_string_response(_pp(lower), lower.byte_length())
    writer.append_int_response(Int64(command_arity(tp, tl)))
    writer.append_set_header(2 if write and denyoom else 1)
    if write:
        writer.append_status_response("write")
        if denyoom:
            writer.append_status_response("denyoom")
    else:
        writer.append_status_response("readonly")
    writer.append_int_response(0)
    writer.append_int_response(0)
    writer.append_int_response(0)
    writer.append_set_header(2)
    writer.append_status_response("@write" if write else "@read")
    writer.append_status_response("@slow")
    writer.append_set_header(0)      # tips
    writer.append_set_header(0)      # key specs
    writer.append_set_header(0)      # subcommands
    _ = lower^
    return True


def _write_all_info(mut writer: ResponseWriter):
    var names = _words(PION_COMMAND_NAMES)
    writer.append_array_header(len(names))
    for k in range(len(names)):
        _ = write_command_info(writer, _pp(names[k]), names[k].byte_length())


# ── key specs ────────────────────────────────────────────────────────────────

struct KeyRef(Copyable, Movable, ImplicitlyCopyable):
    var pos: Int          # token offset from the command name
    var flags: Int        # index into the flags of the spec string (start, len)
    var flen: Int

    def __init__(out self, pos: Int, flags: Int, flen: Int):
        self.pos = pos
        self.flags = flags
        self.flen = flen


def _num(s: Pointer[UInt8, MutUntrackedOrigin], mut at: Int, end: Int) -> Int:
    var neg = False
    if at < end and s[at] == 45:
        neg = True
        at += 1
    var v = 0
    while at < end and s[at] >= 48 and s[at] <= 57:
        v = v * 10 + Int(s[at]) - 48
        at += 1
    return -v if neg else v


def _tok_eq_ci(t: RESP3Token, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    if t.length != n:
        return False
    for k in range(n):
        var a = t.ptr[k]
        var b = p[k]
        if a >= 97 and a <= 122:
            a -= 32
        if b >= 97 and b <= 122:
            b -= 32
        if a != b:
            return False
    return True


def _parse_ll(t: RESP3Token, mut out: Int) -> Bool:
    """string2ll: an optional '-', then digits, nothing else."""
    if t.length == 0 or t.length > 20:
        return False
    var k = 0
    var neg = False
    if t.ptr[0] == 45:
        if t.length == 1:
            return False
        neg = True
        k = 1
    var v = 0
    while k < t.length:
        var c = t.ptr[k]
        if c < 48 or c > 57:
            return False
        v = v * 10 + Int(c) - 48
        k += 1
    out = -v if neg else v
    return True


def _spec_keys(specs: StaticString, tokens: Pointer[RESP3Token, MutUntrackedOrigin], base: Int, argc: Int,
               mut keys: List[KeyRef]) -> Bool:
    """Redis's getKeysUsingKeySpecs. False when a spec cannot be applied (the
    arguments do not fit it); `keys` is then empty."""
    var s = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(specs.unsafe_ptr()))
    var n = specs.byte_length()
    var at = 0
    while at < n:
        var end = at
        while end < n and s[end] != 59:     # ';'
            end += 1
        var colon = at
        while colon < end and s[colon] != 58:
            colon += 1
        var c = at
        var first = 0
        if s[c] == 73:                     # 'I' index
            c += 1
            first = _num(s, c, colon)
        elif s[c] == 75:                   # 'K' startfrom '=' keyword
            c += 1
            var startfrom = _num(s, c, colon)
            c += 1                         # '='
            var kw = c
            # The keyword runs up to FK: FK's letter, then digits, commas and
            # '-'. Step back over those to find the letter.
            var scan = colon - 1
            while scan > kw and ((s[scan] >= 48 and s[scan] <= 57) or s[scan] == 44 or s[scan] == 45):
                scan -= 1
            var kwlen = scan - kw
            c = scan
            var start_index = startfrom if startfrom > 0 else argc + startfrom
            var end_index = argc - 1 if startfrom > 0 else 1
            var i = start_index
            while True:
                if i >= argc or i < 1:
                    break
                if _tok_eq_ci(tokens[base + i], s + kw, kwlen):
                    first = i + 1
                    break
                if i == end_index:
                    break
                i = i + 1 if start_index <= end_index else i - 1
            if first == 0:
                at = end + 1
                continue                   # keyword not there: this spec has no keys
        else:
            keys.clear()
            return False                   # unknown begin_search
        var last = 0
        var step = 1
        if s[c] == 82:                     # 'R' lastkey,keystep,limit
            c += 1
            var lastkey = _num(s, c, colon)
            c += 1
            step = _num(s, c, colon)
            c += 1
            var limit = _num(s, c, colon)
            if lastkey >= 0:
                last = first + lastkey
            elif limit == 0:
                last = argc + lastkey
            else:
                last = first + ((argc - first) // limit + lastkey)
        elif s[c] == 78:                   # 'N' keynumidx,firstkey,keystep
            c += 1
            var keynumidx = _num(s, c, colon)
            c += 1
            var firstkey = _num(s, c, colon)
            c += 1
            step = _num(s, c, colon)
            if keynumidx >= argc - first:
                keys.clear()
                return False
            var numkeys = 0
            if not _parse_ll(tokens[base + first + keynumidx], numkeys) or numkeys < 0:
                keys.clear()
                return False
            first += firstkey
            last = first + numkeys - 1
        else:
            keys.clear()
            return False
        if last >= argc or last < first or first >= argc:
            keys.clear()
            return False
        var i = first
        while i <= last:
            if i < argc:
                keys.append(KeyRef(i, colon + 1, end - colon - 1))
            i += step
        at = end + 1
    return True


comptime _F_RO_ACCESS = "RO,access"
comptime _F_OW_UPDATE = "OW,update"
comptime _F_RW_ACCESS_UPDATE = "RW,access,update"
comptime _F_RW_ACCESS_DELETE = "RW,access,delete"


struct ProcKey(Copyable, Movable, ImplicitlyCopyable):
    var pos: Int
    var flags: StaticString

    def __init__(out self, pos: Int, flags: StaticString):
        self.pos = pos
        self.flags = flags


def _proc_keys(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int, tokens: Pointer[RESP3Token, MutUntrackedOrigin],
               base: Int, argc: Int, mut out: List[ProcKey]) -> Bool:
    """Redis's getkeys procedures, for the commands whose key specs are
    incomplete, unknown or flag-variable. False when the command has none."""
    if arg_eq(tp, tl, "sort") or arg_eq(tp, tl, "sort_ro"):
        out.append(ProcKey(1, _F_RO_ACCESS))
        if arg_eq(tp, tl, "sort_ro"):
            return True
        var store = -1
        var i = 2
        while i < argc:
            var t = tokens[base + i]
            if arg_eq(t.ptr, t.length, "limit"):
                i += 2
            elif arg_eq(t.ptr, t.length, "get") or arg_eq(t.ptr, t.length, "by"):
                i += 1
            elif arg_eq(t.ptr, t.length, "store") and i + 1 < argc:
                store = i + 1              # the last STORE wins, as SORT reads it
            i += 1
        if store >= 0:
            out.append(ProcKey(store, _F_OW_UPDATE))
        return True
    if arg_eq(tp, tl, "migrate"):
        var first = 3
        var num = 1
        if argc > 6:
            var i = 6
            while i < argc:
                var t = tokens[base + i]
                if arg_eq(t.ptr, t.length, "keys"):
                    if tokens[base + 3].length > 0:
                        return True        # Redis: the key argument must be empty; no keys
                    first = i + 1
                    num = argc - first
                    break
                if arg_eq(t.ptr, t.length, "auth"):
                    i += 1
                elif arg_eq(t.ptr, t.length, "auth2"):
                    i += 2
                i += 1
        for k in range(num):
            out.append(ProcKey(first + k, _F_RW_ACCESS_DELETE))
        return True
    if arg_eq(tp, tl, "xread") or arg_eq(tp, tl, "xreadgroup"):
        var streams = -1
        var i = 1
        while i < argc:
            var t = tokens[base + i]
            if arg_eq(t.ptr, t.length, "block") or arg_eq(t.ptr, t.length, "count"):
                i += 1
            elif arg_eq(t.ptr, t.length, "group"):
                i += 2
            elif arg_eq(t.ptr, t.length, "noack"):
                pass
            elif arg_eq(t.ptr, t.length, "streams"):
                streams = i
                break
            else:
                break
            i += 1
        var num = argc - streams - 1
        if streams == -1 or num == 0 or num % 2 != 0:
            return True
        num //= 2
        for k in range(num):
            out.append(ProcKey(streams + 1 + k, _F_RO_ACCESS))
        return True
    if arg_eq(tp, tl, "georadius") or arg_eq(tp, tl, "georadiusbymember"):
        var stored = -1
        var i = 5
        while i < argc:
            var t = tokens[base + i]
            if (arg_eq(t.ptr, t.length, "store") or arg_eq(t.ptr, t.length, "storedist")) and i + 1 < argc:
                stored = i + 1
                i += 1
            i += 1
        out.append(ProcKey(1, _F_RO_ACCESS))
        if stored >= 0:
            out.append(ProcKey(stored, _F_OW_UPDATE))
        return True
    if arg_eq(tp, tl, "set"):
        for i in range(3, argc):
            var t = tokens[base + i]
            if arg_eq(t.ptr, t.length, "get"):
                out.append(ProcKey(1, _F_RW_ACCESS_UPDATE))
                return True
        out.append(ProcKey(1, _F_OW_UPDATE))
        return True
    if arg_eq(tp, tl, "bitfield"):
        var readonly = True
        var i = 2
        while i < argc:
            var rem = argc - i - 1
            var t = tokens[base + i]
            if arg_eq(t.ptr, t.length, "get") and rem >= 2:
                i += 2
            elif (arg_eq(t.ptr, t.length, "set") or arg_eq(t.ptr, t.length, "incrby")) and rem >= 3:
                readonly = False
                break
            elif arg_eq(t.ptr, t.length, "overflow") and rem >= 1:
                i += 1
            else:
                readonly = False
                break
            i += 1
        if readonly:
            out.append(ProcKey(1, _F_RO_ACCESS))
        else:
            out.append(ProcKey(1, _F_RW_ACCESS_UPDATE))
        return True
    return False


def _write_flags(mut writer: ResponseWriter, s: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    """Comma-separated key-spec flags as a set of status strings."""
    var count = 0
    if n > 0:
        count = 1
        for k in range(n):
            if s[k] == 44:
                count += 1
    writer.append_set_header(count)
    var start = 0
    for k in range(n + 1):
        if k == n or s[k] == 44:
            writer.append_status_response(String(StringSpan[MutUntrackedOrigin](
                unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                    unsafe_ptr=s + start, length=k - start))))
            start = k + 1


def _getkeys(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
             mut writer: ResponseWriter, with_flags: Bool):
    var base = i + 2
    var argc = num_tokens - base
    var name = tokens[base]
    var tp = name.ptr
    var tl = name.length
    if not command_exists(tp, tl):
        writer.append_error_response("ERR Invalid command specified")
        return
    var specs = cmd_keyspecs(tp, tl)
    var plist = List[ProcKey]()
    var has_proc = _proc_keys(tp, tl, tokens, base, argc, plist)
    if specs.byte_length() == 0 and not has_proc:
        writer.append_error_response("ERR The command has no key arguments")
        return
    var arity = command_arity(tp, tl)
    if (arity > 0 and arity != argc) or argc < -arity:
        writer.append_error_response("ERR Invalid number of arguments specified for command")
        return
    var sp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(specs.unsafe_ptr()))
    if has_proc:
        if len(plist) == 0:
            if cmd_no_mandatory_keys(tp, tl):
                writer.append_array_header(0)
            else:
                writer.append_error_response("ERR Invalid arguments specified for command")
            return
        writer.append_array_header(len(plist))
        for k in range(len(plist)):
            var t = tokens[base + plist[k].pos]
            if with_flags:
                writer.append_array_header(2)
                writer.append_bulk_string_response(t.ptr, t.length)
                var f = plist[k].flags
                _write_flags(writer, Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(f.unsafe_ptr())),
                             f.byte_length())
            else:
                writer.append_bulk_string_response(t.ptr, t.length)
        return
    var keys = List[KeyRef]()
    var ok = _spec_keys(specs, tokens, base, argc, keys)
    if not ok or len(keys) == 0:
        if cmd_no_mandatory_keys(tp, tl):
            writer.append_array_header(0)
        else:
            writer.append_error_response("ERR Invalid arguments specified for command")
        return
    writer.append_array_header(len(keys))
    for k in range(len(keys)):
        var t = tokens[base + keys[k].pos]
        if with_flags:
            writer.append_array_header(2)
            writer.append_bulk_string_response(t.ptr, t.length)
            _write_flags(writer, sp + keys[k].flags, keys[k].flen)
        else:
            writer.append_bulk_string_response(t.ptr, t.length)


def handle_command(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                   mut writer: ResponseWriter) -> Int:
    """COMMAND [COUNT | INFO [name ...] | LIST [FILTERBY ...] | DOCS [name ...]
    | GETKEYS cmd ... | GETKEYSANDFLAGS cmd ... | HELP] (#47)."""
    var argc = num_tokens - i
    if argc == 1:
        _write_all_info(writer)
        return 0
    var sub = tokens[i + 1]
    var sp = sub.ptr
    var sl = sub.length
    if arg_eq(sp, sl, "count"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'command|count' command")
        else:
            writer.append_int_response(PION_COMMAND_COUNT)
    elif arg_eq(sp, sl, "info"):
        if argc == 2:
            _write_all_info(writer)
        else:
            writer.append_array_header(argc - 2)
            for k in range(i + 2, num_tokens):
                if not write_command_info(writer, tokens[k].ptr, tokens[k].length):
                    writer.append_null_response()
    elif arg_eq(sp, sl, "docs"):
        var names = List[String]()
        if argc == 2:
            names = _words(PION_COMMAND_NAMES)
        else:
            for k in range(i + 2, num_tokens):
                if command_exists(tokens[k].ptr, tokens[k].length):
                    names.append(tokens[k].text_value().lower())
        writer.append_map_header(len(names))
        for k in range(len(names)):
            writer.append_bulk_string_response(_pp(names[k]), names[k].byte_length())
            writer.append_map_header(1)
            writer.append_bulk_string_response("group".unsafe_ptr(), 5)
            var g = cmd_doc_group(_pp(names[k]), names[k].byte_length())
            writer.append_bulk_string_response(g.unsafe_ptr(), g.byte_length())
    elif arg_eq(sp, sl, "list"):
        # the commands, then their subcommands as `command|subcommand`, as
        # Redis lists them (COMMAND COUNT counts the commands only)
        var names = _words(PION_COMMAND_NAMES)
        var subs = _words(PION_SUBCOMMAND_NAMES)
        for k in range(len(subs)):
            names.append(subs[k])
        if argc == 2:
            writer.append_array_header(len(names))
            for k in range(len(names)):
                writer.append_bulk_string_response(_pp(names[k]), names[k].byte_length())
        elif argc == 5 and arg_eq(tokens[i + 2].ptr, tokens[i + 2].length, "filterby"):
            var kind = tokens[i + 3]
            var arg = tokens[i + 4]
            var picked = List[String]()
            if arg_eq(kind.ptr, kind.length, "module"):
                if arg_eq(arg.ptr, arg.length, "search"):
                    for k in range(len(names)):
                        if names[k].startswith("ft."):
                            picked.append(names[k])
            elif arg_eq(kind.ptr, kind.length, "aclcat"):
                var members = acl_category_commands(arg.ptr, arg.length)
                if members != "?":
                    picked = _words(members)
            elif arg_eq(kind.ptr, kind.length, "pattern"):
                # Redis matches the pattern without regard to case; the names
                # are lowercase, so the pattern is lowered once
                # (on the heap: a short String's bytes live in the String itself,
                # on this stack, and may not go to an out-of-line call, gh #349)
                var low = alloc[UInt8](arg.length + 1)
                for b in range(arg.length):
                    var c = arg.ptr[b]
                    low[b] = c + 32 if c >= 65 and c <= 90 else c
                for k in range(len(names)):
                    if _glob_match(low, arg.length, 0, _pp(names[k]), names[k].byte_length(), 0):
                        picked.append(names[k])
                low.free()
            else:
                writer.append_error_response("ERR syntax error")
                return argc - 1
            writer.append_array_header(len(picked))
            for k in range(len(picked)):
                writer.append_bulk_string_response(_pp(picked[k]), picked[k].byte_length())
        else:
            writer.append_error_response("ERR syntax error")
    elif arg_eq(sp, sl, "getkeys") or arg_eq(sp, sl, "getkeysandflags"):
        if argc < 3:
            writer.append_error_response("ERR wrong number of arguments for 'command|" + sub.text_value().lower()
                                         + "' command")
        else:
            _getkeys(tokens, i, num_tokens, writer, arg_eq(sp, sl, "getkeysandflags"))
    elif arg_eq(sp, sl, "help"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'command|help' command")
        else:
            var lines = List[String]()
            lines.append("(no subcommand)")
            lines.append("    Return details about all Redis commands.")
            lines.append("COUNT")
            lines.append("    Return the total number of commands in this Redis server.")
            lines.append("LIST")
            lines.append("    Return a list of all commands in this Redis server.")
            lines.append("INFO [<command-name> ...]")
            lines.append("    Return details about multiple Redis commands.")
            lines.append("    If no command names are given, documentation details for all")
            lines.append("    commands are returned.")
            lines.append("DOCS [<command-name> ...]")
            lines.append("    Return documentation details about multiple Redis commands.")
            lines.append("    If no command names are given, documentation details for all")
            lines.append("    commands are returned.")
            lines.append("GETKEYS <full-command>")
            lines.append("    Return the keys from a full Redis command.")
            lines.append("GETKEYSANDFLAGS <full-command>")
            lines.append("    Return the keys and the access flags from a full Redis command.")
            writer.append_array_header(len(lines) + 3)
            writer.append_status_response("COMMAND <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
            for k in range(len(lines)):
                writer.append_status_response(lines[k])
            writer.append_status_response("HELP")
            writer.append_status_response("    Print this help.")
    else:
        writer.append_error_response("ERR unknown subcommand '" + sub.text_value() + "'. Try COMMAND HELP.")
    return argc - 1
