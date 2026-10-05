"""ACL (#47): the users Pion has, described as Redis describes users.

Pion's users come from its command line: the default user (with the
--requirepass password, or none) and one user per --tenant NAME=PASSWORD,
whose commands are the tenant allowlist and whose keys are NAME:*. ACL
answered +OK to SETUSER, GETUSER, DELUSER, GENPASS and HELP without doing
anything, and an empty LIST and CAT. Now:

  * GETUSER, LIST, USERS and WHOAMI describe those users; passwords appear
    as their SHA-256, as Redis shows them.
  * CAT lists the categories and the Pion commands in each; DRYRUN answers
    whether a user may run a command; GENPASS generates a password from the
    system's entropy; HELP has Redis's text.
  * LOG records failed AUTHs (and HELLO AUTH) and commands a tenant was
    refused, grouped and shaped as Redis records them.
  * SETUSER and DELUSER refuse: the users are fixed at startup. SAVE and LOAD
    answer Redis's no-ACL-file error, which is Pion's situation.
"""

from std.collections import List
from std.memory import alloc
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call
from src.common.ptr import is_not_null
from src.common.utils import bytes_to_string
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.commands.tenant import TenantTable, tenant_keyspec, MAX_TENANT_PASS
from src.commands.command_info import ACL_CATEGORIES, acl_category_commands, PION_COMMAND_NAMES
from src.commands.command_table import command_exists, command_arity

comptime ACL_LOG_MAX = 128
comptime ACL_LOG_GROUP_MS = 60000


struct AclLogEntry(Copyable, Movable):
    var count: Int64
    var reason: String       # auth, command, key
    var context: String      # toplevel, multi, lua
    var object: String
    var username: String
    var client_info: String
    var entry_id: Int64
    var created_ms: Int64
    var updated_ms: Int64

    def __init__(out self, var reason: String, var context: String, var object: String, var username: String,
                 var client_info: String, entry_id: Int64, now_ms: Int64):
        self.count = 1
        self.reason = reason^
        self.context = context^
        self.object = object^
        self.username = username^
        self.client_info = client_info^
        self.entry_id = entry_id
        self.created_ms = now_ms
        self.updated_ms = now_ms


struct AclLog(Movable):
    var entries: List[AclLogEntry]   # newest first
    var next_id: Int64

    def __init__(out self):
        self.entries = List[AclLogEntry]()
        self.next_id = 0

    def __init__(out self, *, deinit take: Self):
        self.entries = take.entries^
        self.next_id = take.next_id

    def add(mut self, var reason: String, var context: String, var object: String, var username: String,
            var client_info: String):
        """Redis's addACLLogEntry: an entry like a recent one (same reason,
        context, object and user, within 60 s) counts up instead."""
        var now = external_call["pion_unix_ms", Int64]()
        for k in range(len(self.entries)):
            ref e = self.entries[k]
            if (e.reason == reason and e.context == context and e.object == object and e.username == username
                    and now - e.updated_ms < ACL_LOG_GROUP_MS):
                e.count += 1
                e.updated_ms = now
                e.client_info = client_info^
                var moved = self.entries.pop(k)
                self.entries.insert(0, moved^)
                return
        self.entries.insert(0, AclLogEntry(reason^, context^, object^, username^, client_info^, self.next_id, now))
        self.next_id += 1
        while len(self.entries) > ACL_LOG_MAX:
            _ = self.entries.pop()

    def write(self, mut writer: ResponseWriter, count: Int):
        var n = len(self.entries)
        if count < n:
            n = count if count > 0 else 0
        var now = external_call["pion_unix_ms", Int64]()
        writer.append_array_header(n)
        for k in range(n):
            ref e = self.entries[k]
            writer.append_map_header(10)
            _kv_int(writer, "count", e.count)
            _kv_str(writer, "reason", e.reason)
            _kv_str(writer, "context", e.context)
            _kv_str(writer, "object", e.object)
            _kv_str(writer, "username", e.username)
            writer.append_bulk_string_response("age-seconds".unsafe_ptr(), 11)
            var age = _seconds_text(now - e.created_ms)
            writer.append_double_response(age.unsafe_ptr(), age.byte_length())
            _kv_str(writer, "client-info", e.client_info)
            _kv_int(writer, "entry-id", e.entry_id)
            _kv_int(writer, "timestamp-created", e.created_ms)
            _kv_int(writer, "timestamp-last-updated", e.updated_ms)


def _seconds_text(ms: Int64) -> String:
    """Milliseconds as seconds, the shortest way Redis prints the double:
    218 -> "0.218", 1430 -> "1.43", 0 -> "0"."""
    var m = ms if ms > 0 else Int64(0)
    var whole = String(Int(m // 1000))
    var frac = Int(m % 1000)
    if frac == 0:
        return whole^
    var digits = String(frac + 1000)      # "1xyz": keeps the leading zeros
    var d = digits.as_bytes()
    var end = 4
    while end > 1 and d[end - 1] == 48:
        end -= 1
    var out = whole + "."
    for k in range(1, end):
        out += chr(Int(d[k]))
    return out^


def _kv_int(mut writer: ResponseWriter, k: StaticString, v: Int64):
    writer.append_bulk_string_response(k.unsafe_ptr(), k.byte_length())
    writer.append_int_response(v)


def _kv_str(mut writer: ResponseWriter, k: StaticString, v: String):
    writer.append_bulk_string_response(k.unsafe_ptr(), k.byte_length())
    writer.append_bulk_string_response(v.unsafe_ptr(), v.byte_length())


def _ci(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, lit: StaticString) -> Bool:
    if n != lit.byte_length():
        return False
    var lp = lit.unsafe_ptr()
    for k in range(n):
        var c = p[k]
        if c >= 65 and c <= 90:
            c += 32
        if c != lp[k]:
            return False
    return True


@always_inline
def sha256_hex(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> String:
    var out = alloc[UInt8](64)
    external_call["pion_sha256_hex", NoneType](p, Int64(n), out)
    var s = bytes_to_string(out, 64)
    out.free()
    return s


struct AclUsers:
    """The users of this server, read from its configuration."""
    var requirepass: String
    var tenants: Pointer[TenantTable, MutUntrackedOrigin]

    def __init__(out self, requirepass: String, tenants: Pointer[TenantTable, MutUntrackedOrigin]):
        self.requirepass = requirepass
        self.tenants = tenants

    def tenant_count(self) -> Int:
        return self.tenants[].count if is_not_null(self.tenants) else 0

    def find(self, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
        """-1 default, a tenant's index, or -2 for no such user. Names are
        case-sensitive, as Redis's are."""
        if n == 7 and _same(p, n, "default"):
            return -1
        for t in range(self.tenant_count()):
            if self.tenants[].name_len(t) == n:
                var np = self.tenants[].name_ptr(t)
                var eq = True
                for k in range(n):
                    if np[k] != p[k]:
                        eq = False
                        break
                if eq:
                    return t
        return -2

    def name(self, u: Int) -> String:
        if u < 0:
            return "default"
        return bytes_to_string(self.tenants[].name_ptr(u), self.tenants[].name_len(u))

    def password_hash(self, u: Int) -> String:
        """The user's password as SHA-256 hex, "" when it has none."""
        if u < 0:
            if self.requirepass.byte_length() == 0:
                return ""
            var rp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self.requirepass.unsafe_ptr()))
            var h = sha256_hex(rp, self.requirepass.byte_length())
            _ = self.requirepass
            return h
        var pp = self.tenants[].pass_buf.unsafe_offset(u * MAX_TENANT_PASS)
        return sha256_hex(pp, Int(self.tenants[].pass_lens[u]))

    def command_rules(self, u: Int) -> String:
        """The user's commands as ACL rules: +@all for default, -@all and
        each allowed command for a tenant."""
        if u < 0:
            return "+@all"
        var out = String("-@all")
        var names = PION_COMMAND_NAMES
        var np = names.unsafe_ptr()
        var nl = names.byte_length()
        var start = 0
        for k in range(nl + 1):
            if k == nl or np[k] == 32:
                if k > start:
                    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(np) + start)
                    if tenant_keyspec(p, k - start).allowed:
                        out += " +" + bytes_to_string(p, k - start)
                start = k + 1
        return out^

    def key_rule(self, u: Int) -> String:
        if u < 0:
            return "~*"
        return "~" + self.name(u) + ":*"

    def flags(self, u: Int) -> List[String]:
        var f = List[String]()
        f.append("on")
        if u < 0 and self.requirepass.byte_length() == 0:
            f.append("nopass")
        f.append("sanitize-payload")
        return f^

    def list_line(self, u: Int) -> String:
        var s = String("user ") + self.name(u)
        var f = self.flags(u)
        for k in range(len(f)):
            s += " " + f[k]
        var h = self.password_hash(u)
        if h.byte_length() > 0:
            s += " #" + h
        s += " " + self.key_rule(u)
        s += " &*" if u < 0 else " resetchannels"
        s += " " + self.command_rules(u)
        return s^


def _same(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, lit: StaticString) -> Bool:
    if n != lit.byte_length():
        return False
    var lp = lit.unsafe_ptr()
    for k in range(n):
        if p[k] != lp[k]:
            return False
    return True


def _parse_long(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, mut out: Int64) -> Bool:
    """Redis's string2ll."""
    if n == 0 or n > 20:
        return False
    var k = 0
    var neg = False
    if p[0] == 45:
        neg = True
        k = 1
        if n == 1:
            return False
    if p[k] == 48 and n - k > 1:
        return False
    var v = UInt64(0)
    while k < n:
        var c = p[k]
        if c < 48 or c > 57:
            return False
        var d = UInt64(c - 48)
        if v > (UInt64(0xFFFFFFFFFFFFFFFF) - d) // 10:
            return False
        v = v * 10 + d
        k += 1
    if neg:
        if v > UInt64(9223372036854775808):
            return False
        out = Int64(0) - Int64(v - 1) - 1 if v > 0 else Int64(0)
    elif v > UInt64(9223372036854775807):
        return False
    else:
        out = Int64(v)
    return True


comptime _NO_ACL_FILE = ("ERR This Redis instance is not configured to use an ACL file. You may want to specify users "
                         + "via the ACL SETUSER command and then issue a CONFIG REWRITE (assuming you have a Redis "
                         + "configuration file set) in order to store users in the Redis configuration.")


def handle_acl(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
               users: AclUsers, mut log: AclLog, caller: Int) raises:
    """ACL <subcommand> (#47). `caller` is the connection's user: -1
    default, else its tenant."""
    var argc = end - i
    if argc < 2:
        writer.append_error_response("ERR wrong number of arguments for 'acl' command")
        return
    var sub = tokens[i + 1]
    var sp = sub.ptr
    var sl = sub.length
    if _ci(sp, sl, "whoami"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'acl|whoami' command")
            return
        var n = users.name(caller)
        writer.append_bulk_string_response(n.unsafe_ptr(), n.byte_length())
    elif _ci(sp, sl, "users"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'acl|users' command")
            return
        var names = List[String]()
        names.append("default")
        for t in range(users.tenant_count()):
            names.append(users.name(t))
        _sort(names)
        writer.append_array_header(len(names))
        for k in range(len(names)):
            writer.append_bulk_string_response(names[k].unsafe_ptr(), names[k].byte_length())
    elif _ci(sp, sl, "list"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'acl|list' command")
            return
        var order = List[String]()
        order.append("default")
        for t in range(users.tenant_count()):
            order.append(users.name(t))
        _sort(order)
        writer.append_array_header(len(order))
        for k in range(len(order)):
            var op = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(order[k].unsafe_ptr()))
            var line = users.list_line(users.find(op, order[k].byte_length()))
            writer.append_bulk_string_response(line.unsafe_ptr(), line.byte_length())
    elif _ci(sp, sl, "getuser"):
        if argc != 3:
            writer.append_error_response("ERR wrong number of arguments for 'acl|getuser' command")
            return
        var u = users.find(tokens[i + 2].ptr, tokens[i + 2].length)
        if u == -2:
            writer.append_null_response()
            return
        writer.append_map_header(6)
        writer.append_bulk_string_response("flags".unsafe_ptr(), 5)
        var f = users.flags(u)
        writer.append_set_header(len(f))
        for k in range(len(f)):
            writer.append_bulk_string_response(f[k].unsafe_ptr(), f[k].byte_length())
        writer.append_bulk_string_response("passwords".unsafe_ptr(), 9)
        var h = users.password_hash(u)
        if h.byte_length() > 0:
            writer.append_array_header(1)
            writer.append_bulk_string_response(h.unsafe_ptr(), h.byte_length())
        else:
            writer.append_array_header(0)
        var cr = users.command_rules(u)
        _kv_str(writer, "commands", cr)
        _kv_str(writer, "keys", users.key_rule(u))
        _kv_str(writer, "channels", String("&*") if u < 0 else String(""))
        writer.append_bulk_string_response("selectors".unsafe_ptr(), 9)
        writer.append_array_header(0)
    elif _ci(sp, sl, "cat"):
        if argc > 3:
            writer.append_error_response("ERR wrong number of arguments for 'acl|cat' command")
            return
        if argc == 2:
            _write_words(writer, ACL_CATEGORIES)
            return
        var c = tokens[i + 2]
        var low = alloc[UInt8](c.length + 1)
        for k in range(c.length):
            var b = c.ptr[k]
            low[k] = b + 32 if b >= 65 and b <= 90 else b
        var members = acl_category_commands(low, c.length)
        low.free()
        if members == "?":
            writer.append_error_response("ERR Unknown category '" + bytes_to_string(c.ptr, c.length) + "'")
            return
        _write_words(writer, members)
    elif _ci(sp, sl, "genpass"):
        if argc > 3:
            writer.append_error_response("ERR wrong number of arguments for 'acl|genpass' command")
            return
        var bits = Int64(256)
        if argc == 3:
            if not _parse_long(tokens[i + 2].ptr, tokens[i + 2].length, bits):
                writer.append_error_response("ERR value is not an integer or out of range")
                return
            if bits <= 0 or bits > 4096:
                writer.append_error_response("ERR ACL GENPASS argument must be the number of bits for the output "
                                             + "password, a positive number up to 4096")
                return
        var out = alloc[UInt8](1025)
        var n = Int(external_call["pion_genpass_hex", Int64](bits, out))
        if n <= 0:
            out.free()
            writer.append_error_response("ERR failed to read random bytes")
            return
        writer.append_bulk_string_response(out, n)
        out.free()
    elif _ci(sp, sl, "dryrun"):
        if argc < 4:
            writer.append_error_response("ERR wrong number of arguments for 'acl|dryrun' command")
            return
        var u = users.find(tokens[i + 2].ptr, tokens[i + 2].length)
        if u == -2:
            writer.append_error_response("ERR User '" + bytes_to_string(tokens[i + 2].ptr, tokens[i + 2].length)
                                         + "' not found")
            return
        var c = tokens[i + 3]
        if not command_exists(c.ptr, c.length):
            writer.append_error_response("ERR Command '" + bytes_to_string(c.ptr, c.length) + "' not found")
            return
        var ar = command_arity(c.ptr, c.length)
        var cargc = end - (i + 3)
        if (ar > 0 and cargc != ar) or (ar < 0 and cargc < -ar):
            writer.append_error_response("ERR wrong number of arguments for '" + _lower(c.ptr, c.length)
                                         + "' command")
            return
        if u >= 0 and not tenant_keyspec(c.ptr, c.length).allowed:
            var m = String("User ") + users.name(u) + " has no permissions to run the '" + _lower(c.ptr, c.length) \
                    + "' command"
            writer.append_bulk_string_response(m.unsafe_ptr(), m.byte_length())
            return
        writer.append_ok_response()
    elif _ci(sp, sl, "log"):
        if argc > 3:
            writer.append_error_response("ERR wrong number of arguments for 'acl|log' command")
            return
        var count = 10
        if argc == 3:
            var a = tokens[i + 2]
            if _ci(a.ptr, a.length, "reset"):
                log.entries.clear()
                writer.append_ok_response()
                return
            var v = Int64(0)
            if not _parse_long(a.ptr, a.length, v):
                writer.append_error_response("ERR value is not an integer or out of range")
                return
            count = Int(v) if v < 1 << 30 else 1 << 30
        log.write(writer, count)
    elif _ci(sp, sl, "setuser"):
        if argc < 3:
            writer.append_error_response("ERR wrong number of arguments for 'acl|setuser' command")
            return
        writer.append_error_response("ERR ACL SETUSER is not supported: Pion's users are the default user "
                                     + "and the --tenant users on its command line")
    elif _ci(sp, sl, "deluser"):
        if argc < 3:
            writer.append_error_response("ERR wrong number of arguments for 'acl|deluser' command")
            return
        for k in range(i + 2, end):
            if _same(tokens[k].ptr, tokens[k].length, "default"):
                writer.append_error_response("ERR The 'default' user cannot be removed")
                return
        for k in range(i + 2, end):
            if users.find(tokens[k].ptr, tokens[k].length) >= 0:
                writer.append_error_response("ERR ACL DELUSER is not supported: Pion's --tenant users are "
                                             + "fixed at startup")
                return
        writer.append_int_response(0)
    elif _ci(sp, sl, "save") or _ci(sp, sl, "load"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'acl|" + _lower(sp, sl) + "' command")
            return
        writer.append_error_response(_NO_ACL_FILE)
    elif _ci(sp, sl, "help"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'acl|help' command")
            return
        var lines = List[String]()
        lines.append("ACL <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
        lines.append("CAT [<category>]")
        lines.append("    List all commands that belong to <category>, or all command categories")
        lines.append("    when no category is specified.")
        lines.append("DELUSER <username> [<username> ...]")
        lines.append("    Delete a list of users.")
        lines.append("DRYRUN <username> <command> [<arg> ...]")
        lines.append("    Returns whether the user can execute the given command without executing the command.")
        lines.append("GETUSER <username>")
        lines.append("    Get the user's details.")
        lines.append("GENPASS [<bits>]")
        lines.append("    Generate a secure 256-bit user password. The optional `bits` argument can")
        lines.append("    be used to specify a different size.")
        lines.append("LIST")
        lines.append("    Show users details in config file format.")
        lines.append("LOAD")
        lines.append("    Reload users from the ACL file.")
        lines.append("LOG [<count> | RESET]")
        lines.append("    Show the ACL log entries.")
        lines.append("SAVE")
        lines.append("    Save the current config to the ACL file.")
        lines.append("SETUSER <username> <attribute> [<attribute> ...]")
        lines.append("    Create or modify a user with the specified attributes.")
        lines.append("USERS")
        lines.append("    List all the registered usernames.")
        lines.append("WHOAMI")
        lines.append("    Return the current connection username.")
        lines.append("HELP")
        lines.append("    Print this help.")
        writer.append_array_header(len(lines))
        for k in range(len(lines)):
            writer.append_status_response(lines[k])
    else:
        writer.append_error_response("ERR unknown subcommand '" + bytes_to_string(sp, sl) + "'. Try ACL HELP.")


def _lower(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> String:
    var b = alloc[UInt8](n + 1)
    for k in range(n):
        var c = p[k]
        b[k] = c + 32 if c >= 65 and c <= 90 else c
    var s = bytes_to_string(b, n)
    b.free()
    return s


def _write_words(mut writer: ResponseWriter, words: StaticString):
    """A space-separated list as an array of bulk strings."""
    var p = words.unsafe_ptr()
    var n = words.byte_length()
    var count = 0
    var start = 0
    for k in range(n + 1):
        if k == n or p[k] == 32:
            if k > start:
                count += 1
            start = k + 1
    writer.append_array_header(count)
    start = 0
    for k in range(n + 1):
        if k == n or p[k] == 32:
            if k > start:
                writer.append_bulk_string_response(
                    Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(p) + start), k - start)
            start = k + 1


def _sort(mut v: List[String]):
    for a in range(1, len(v)):
        var j = a
        while j > 0 and v[j] < v[j - 1]:
            var t = v[j]
            v[j] = v[j - 1]
            v[j - 1] = t
            j -= 1
