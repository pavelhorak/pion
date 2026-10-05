"""SLOWLOG (#47): the commands that took longer than slowlog-log-slower-than.

SLOWLOG GET answered an empty array whatever ran, because nothing was ever
recorded. The slow path now times each command it runs (two reads of the
CPU's tick counter, pion_ticks) and keeps the ones at or over the threshold,
newest first, up to slowlog-max-len, as Redis's slowlogPushEntryIfNeeded
does: at most slowlog-entry-max-argc (32) arguments, the last replaced by
"... (N more arguments)", each cut to slowlog-entry-max-string-len (128)
bytes plus "... (N more bytes)", and, as Redis 8 adds, the command's
argument count. AUTH's arguments, HELLO's AUTH credentials, MIGRATE's AUTH /
AUTH2 ones and CONFIG SET's requirepass / masterauth values are logged as
"(redacted)", as Redis does. EXEC is not logged (Redis flags it
skip_slowlog); the commands it runs are. A command queued inside MULTI, or
refused before it ran (NOAUTH), is not logged either.

Commands the fast path serves (GET, SET, INCR, LPUSH, ...) are O(1) and are
not timed, so that GET and SET pay nothing for this.

slowlog-log-slower-than and slowlog-max-len are process-wide (CONFIG SET on
any worker); each worker keeps its own entries.
"""

from std.collections import List
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.common.utils import bytes_to_string

comptime SLOWLOG_MAX_ARGC = 32
comptime SLOWLOG_MAX_STRING = 128


struct SlowLogEntry(Copyable, Movable):
    var id: Int64
    var ts: Int64           # unix seconds
    var us: Int64           # duration, microseconds
    var args: List[String]
    var peer: String
    var name: String
    var argc: Int64         # the command's arguments, before args was cut to 32 (Redis 8's 7th field)

    def __init__(out self, id: Int64, ts: Int64, us: Int64, var args: List[String], var peer: String,
                 var name: String, argc: Int64):
        self.id = id
        self.ts = ts
        self.us = us
        self.args = args^
        self.peer = peer^
        self.name = name^
        self.argc = argc


struct SlowLog(Movable):
    var entries: List[SlowLogEntry]   # oldest first; GET answers newest first
    var next_id: Int64
    var ticks_per_us: Float64

    def __init__(out self):
        self.entries = List[SlowLogEntry]()
        self.next_id = 0
        self.ticks_per_us = external_call["pion_ticks_per_us", Float64]()

    def __init__(out self, *, deinit take: Self):
        self.entries = take.entries^
        self.next_id = take.next_id
        self.ticks_per_us = take.ticks_per_us

    @always_inline
    def threshold_us(self) -> Int64:
        return external_call["pion_slowlog_get_slower_than", Int64]()

    def elapsed_us(self, t0: UInt64, t1: UInt64) -> Int64:
        if t1 <= t0 or self.ticks_per_us <= 0:
            return 0
        return Int64(Float64(t1 - t0) / self.ticks_per_us)

    def push(mut self, tokens: Pointer[RESP3Token, MutUntrackedOrigin], first: Int, end: Int, us: Int64,
             var peer: String, var name: String):
        """Record tokens[first:end] as one entry, if the log keeps any."""
        var max_len = Int(external_call["pion_slowlog_get_max_len", Int64]())
        if max_len <= 0:
            self.entries.clear()
            return
        var args = List[String]()
        var argc = end - first
        var shown = argc if argc <= SLOWLOG_MAX_ARGC else SLOWLOG_MAX_ARGC
        var redact_from = _redact_from(tokens, first, end)
        for k in range(shown):
            if shown != argc and k == shown - 1:
                args.append(String("... (") + String(argc - shown + 1) + " more arguments)")
                break
            var t = tokens[first + k]
            if redact_from >= 0 and first + k >= redact_from and _redacted(tokens, first, first + k):
                args.append("(redacted)")
                continue
            if t.length > SLOWLOG_MAX_STRING:
                args.append(_text(t.ptr, SLOWLOG_MAX_STRING) + "... (" + String(t.length - SLOWLOG_MAX_STRING)
                            + " more bytes)")
            else:
                args.append(_text(t.ptr, t.length))
        var ts = external_call["pion_unix_ms", Int64]() // 1000
        self.entries.append(SlowLogEntry(self.next_id, ts, us, args^, peer^, name^, Int64(argc)))
        self.next_id += 1
        while len(self.entries) > max_len:
            _ = self.entries.pop(0)

    def write_get(self, mut writer: ResponseWriter, count: Int):
        """The newest `count` entries (all for -1), newest first."""
        var n = len(self.entries)
        if count >= 0 and count < n:
            n = count
        writer.append_array_header(n)
        for k in range(n):
            ref e = self.entries[len(self.entries) - 1 - k]
            writer.append_array_header(7)
            writer.append_int_response(e.id)
            writer.append_int_response(e.ts)
            writer.append_int_response(e.us)
            writer.append_array_header(len(e.args))
            for a in range(len(e.args)):
                writer.append_bulk_string_response(e.args[a].unsafe_ptr(), e.args[a].byte_length())
            writer.append_bulk_string_response(e.peer.unsafe_ptr(), e.peer.byte_length())
            writer.append_bulk_string_response(e.name.unsafe_ptr(), e.name.byte_length())
            writer.append_int_response(e.argc)


def _text(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> String:
    """The bytes as a String, unchanged (bytes_to_string)."""
    return bytes_to_string(p, n)


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


def _redact_from(tokens: Pointer[RESP3Token, MutUntrackedOrigin], first: Int, end: Int) -> Int:
    """The first argument that may be redacted, -1 when none is."""
    var c = tokens[first]
    if _ci(c.ptr, c.length, "auth"):
        return first + 1
    if _ci(c.ptr, c.length, "hello") or _ci(c.ptr, c.length, "migrate") or _ci(c.ptr, c.length, "config"):
        return first + 1
    return -1


def _redacted(tokens: Pointer[RESP3Token, MutUntrackedOrigin], first: Int, k: Int) -> Bool:
    """Is tokens[k] a credential? AUTH: every argument. HELLO: the two after
    AUTH. MIGRATE: the one after AUTH, the two after AUTH2. CONFIG SET: the
    value of requirepass or masterauth."""
    var c = tokens[first]
    if _ci(c.ptr, c.length, "auth"):
        return True
    if _ci(c.ptr, c.length, "config"):
        var p = tokens[k - 1]
        return k - 1 > first + 1 and (_ci(p.ptr, p.length, "requirepass") or _ci(p.ptr, p.length, "masterauth"))
    for back in range(1, 3):
        if k - back <= first:
            break
        var p = tokens[k - back]
        if _ci(p.ptr, p.length, "auth"):
            if _ci(c.ptr, c.length, "hello"):
                return back <= 2
            return back == 1
        if _ci(p.ptr, p.length, "auth2"):
            return back <= 2
    return False


def handle_slowlog(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, end: Int, mut writer: ResponseWriter,
                   mut log: SlowLog):
    """SLOWLOG GET [count] | LEN | RESET | HELP."""
    if end - i < 2:
        writer.append_error_response("ERR wrong number of arguments for 'slowlog' command")
        return
    var sub = tokens[i + 1]
    var argc = end - i
    if _ci(sub.ptr, sub.length, "get"):
        if argc > 3:
            # slowlog|get's arity is -2, so Redis's own check passes and its
            # handler answers this
            writer.append_error_response("ERR unknown subcommand or wrong number of arguments for '"
                                         + _text(sub.ptr, sub.length) + "'. Try SLOWLOG HELP.")
            return
        var count = 10
        if argc == 3:
            var c = tokens[i + 2]
            var v = Int64(0)
            var ok = _parse_ll(c.ptr, c.length, v)
            if not ok or v < -1:
                writer.append_error_response("ERR count should be greater than or equal to -1")
                return
            count = Int(v)
        log.write_get(writer, count)
    elif _ci(sub.ptr, sub.length, "len"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'slowlog|len' command")
            return
        writer.append_int_response(Int64(len(log.entries)))
    elif _ci(sub.ptr, sub.length, "reset"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'slowlog|reset' command")
            return
        log.entries.clear()
        writer.append_ok_response()
    elif _ci(sub.ptr, sub.length, "help"):
        if argc != 2:
            writer.append_error_response("ERR wrong number of arguments for 'slowlog|help' command")
            return
        var lines = List[String]()
        lines.append("SLOWLOG <subcommand> [<arg> [value] [opt] ...]. Subcommands are:")
        lines.append("GET [<count>]")
        lines.append("    Return top <count> entries from the slowlog (default: 10, -1 mean all).")
        lines.append("    Entries are made of:")
        lines.append("    id, timestamp, time in microseconds, arguments array, client IP and port,")
        lines.append("    client name")
        lines.append("LEN")
        lines.append("    Return the length of the slowlog.")
        lines.append("RESET")
        lines.append("    Reset the slowlog.")
        lines.append("HELP")
        lines.append("    Print this help.")
        writer.append_array_header(len(lines))
        for k in range(len(lines)):
            writer.append_status_response(lines[k])
    else:
        writer.append_error_response("ERR unknown subcommand '" + _text(sub.ptr, sub.length)
                                     + "'. Try SLOWLOG HELP.")


def _parse_ll(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, mut out: Int64) -> Bool:
    """Redis's string2ll: optional '-', digits, no leading zeros or spaces,
    within 64 bits."""
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
    else:
        if v > UInt64(9223372036854775807):
            return False
        out = Int64(v)
    return True
