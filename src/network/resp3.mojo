from src.common.ptr import null_ptr, is_not_null
from std.memory import alloc
from std.collections import List, Array, Span
from std.memory.unsafe_pointer import Pointer
from std.sys import simd_width_of

# gh #102 (C4): RESP length bounds — a `proto-max-bulk-len` equivalent.
# Without an upper bound, `bl = bl*10 + digit` on a ~19-digit prefix overflows
# `Int` negative, which defeats the `data_start + bl + 2 > length` bounds check
# below and yields an OOB read into `RESP3Token(buffer + data_start, bl)`.
# Guarding the accumulator *during* the digit loop bails before any overflow
# (worst case reaches MAX_BULK_LEN*10+9 ≈ 5.4e9, far inside Int64). 512 MB
# matches the Redis default ceiling; the array element count is capped to bound
# the per-command verification loop against a wrapped/huge count.
comptime MAX_BULK_LEN = 536870912   # 512 MB
comptime MAX_ARRAY_LEN = 1048576    # 1M elements per command array

# gh #153: `parse_stream` reports "no tokens" for two different situations —
# an incomplete frame (wait for more data) and a single complete command that
# does not fit the 64-token array (never satisfiable by waiting). The caller
# treated both as "wait", so the oversized command parked its fd forever.
# num_tokens == TOKENS_OVERFLOW distinguishes the second: the frame is complete
# and `consumed_bytes` covers it. Since #52 the slow path then parses that one
# command into a table of its own size (`need_tokens`) and runs it, up to
# MAX_ARRAY_LEN arguments, as Redis does; it used to answer -ERR.
comptime TOKENS_OVERFLOW = -1
# gh #166: 2048, not 64. `VADD key VALUES 1536 <v1..v1536> elem` is ~1539 RESP
# tokens; at 64 it could only ever be the gh #153 error. The array no longer
# lives on the `parallelize` worker stack — SlowPathHandler owns one heap
# instance per worker (~112 KB for the token + boundary tables together), which
# is what makes this bump safe: 2048 x 24 B inline on a ~512 KB worker stack is
# the shape that already produced SIGSEGV at -w >= 2 elsewhere in this codebase.
comptime MAX_CMD_TOKENS = 2048     # capacity of the parse_stream token array

# gh #156: capacity of the per-batch command-boundary table. Every command is
# at least one token, so a parse that stores at most MAX_CMD_TOKENS tokens can
# never see more than that many commands — sizing the two together means
# `num_cmds` is always exact and `cmd_ends[cmd_idx]` is always the real end of
# command `cmd_idx`. It was 16, and the dispatch loop fell back to
# `cmd_end_tok = num_tokens` past that, so the 17th command in a pipelined
# batch consumed every command after it: 20 in, 17 replies out. Any handler
# that skips with `i = cmd_end_tok - 1` was exposed.
comptime MAX_CMD_ENDS = MAX_CMD_TOKENS

def _is_valid_utf8(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    """Strict UTF-8 (RFC 3629): no overlong forms, no surrogates, nothing past
    U+10FFFF — the bytes a Mojo String may hold."""
    var i = 0
    while i < n:
        var b0 = Int(p[unsafe_offset=i])
        if b0 < 0x80:
            i += 1
            continue
        var need: Int
        var lo = 0x80
        var hi = 0xBF
        if b0 >= 0xC2 and b0 <= 0xDF:
            need = 1
        elif b0 == 0xE0:
            need = 2; lo = 0xA0
        elif (b0 >= 0xE1 and b0 <= 0xEC) or b0 == 0xEE or b0 == 0xEF:
            need = 2
        elif b0 == 0xED:
            need = 2; hi = 0x9F
        elif b0 == 0xF0:
            need = 3; lo = 0x90
        elif b0 >= 0xF1 and b0 <= 0xF3:
            need = 3
        elif b0 == 0xF4:
            need = 3; hi = 0x8F
        else:
            return False
        if i + need >= n:          # a truncated sequence at the end
            return False
        var b1 = Int(p[unsafe_offset=i + 1])
        if b1 < lo or b1 > hi:
            return False
        for k in range(2, need + 1):
            var bk = Int(p[unsafe_offset=i + k])
            if bk < 0x80 or bk > 0xBF:
                return False
        i += need + 1
    return True


struct RESP3Token(Copyable, Movable, ImplicitlyCopyable):
    var marker: UInt8
    var ptr: Pointer[UInt8, MutUntrackedOrigin]
    var length: Int

    def __init__(out self, marker: UInt8, ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int):
        self.marker = marker
        self.ptr = ptr
        self.length = length

    def copy(self) -> Self:
        return self

    def value(self) -> String:
        """Token bytes as a String, BYTE FOR BYTE, valid UTF-8 or not.

        Redis keys, values, list elements, hash fields and set/zset members
        are binary-safe, and this is what ~470 slow-path call sites use to
        read them. It used to spell every byte of an invalid UTF-8 sequence
        as '?' (gh #334 kept that half), which aliased different keys onto one
        (b"k\\xff" and b"k\\xfe" both became "k?" — `SET k\\xff v EX 9`
        wrote key "k?" while a plain `SET k\\xff` on the fast path wrote the
        real key), and stored "??" for every binary member of a multi-element
        LPUSH/SADD/ZADD while replying success. Found by
        tests/test_binary_safety_differential.py (~90 commands diverged from
        Redis on binary keys).

        The String is a byte container here: byte_length / unsafe_ptr /
        GenericValue.from_string / byte-wise ==. A handler that does TEXT
        operations on a token (upper/lower, codepoint iteration) must take
        text_value() instead.
        """
        if self.length == 0:
            return String("")
        return String(StringSpan[MutUntrackedOrigin](
            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=self.ptr, length=self.length)))

    def text_value(self) -> String:
        """Token bytes as TEXT: exact whenever they are valid UTF-8, and with
        bytes >= 128 spelled '?' only when they are not.

        ONLY for a token the handler treats as text — case-folds, prints or
        parses as a keyword (`PING <msg>`'s SET/GET check, a VQUANT/format
        name). A Mojo String is documented to hold valid UTF-8, and its
        codepoint operations (upper/lower) are not safe on anything else.
        Keys, members, fields and values are NOT text: use value().

        gh #334: every non-ASCII byte used to be spelled '?', valid UTF-8
        included. A key like "clé" became "cl??" on every slow-path command, so
        20 of 27 read commands missed a key the fast path had stored byte-exact,
        and SET ... EX / SETEX / SETNX / GETSET stored "café" as "caf??". A
        Mojo String must hold valid UTF-8, so text passes through unchanged and
        only a genuinely invalid (binary) sequence keeps the old spelling;
        handlers that store VALUES take raw_value() instead, which is byte-exact
        for binary too.

        gh #202: this used to append one character at a time (`s += chr(b)`),
        an O(n) realloc chain per token — heap-String construction was 14.8% of
        server CPU on the XADD row, the one clearly server-bound row in the
        valkey gate table. Every slow-path handler pays it for every token it
        names. Bulk-construct instead.

        The '?' mapping is preserved exactly, not dropped: SIMD-scan for a high
        byte first and only fall back to the per-character spelling when one is
        actually present. Command names, keys and IDs are ASCII on every hot
        path, so the scan is the common case and the slow spelling is rare.
        """
        if self.length == 0:
            return String("")

        comptime W = 16
        var i = 0
        var has_high = False
        while i + W <= self.length:
            var chunk = (self.ptr.unsafe_offset(i)).load[width=W]()
            if (chunk & UInt8(0x80)).reduce_max() != UInt8(0):
                has_high = True
                break
            i += W
        if not has_high:
            while i < self.length:
                if self.ptr[unsafe_offset=i] >= UInt8(128):
                    has_high = True
                    break
                i += 1

        if not has_high:
            # All-ASCII, so the bytes are valid UTF-8 by construction and the
            # String copies them out — no borrow into the recv buffer survives.
            return String(StringSpan[MutUntrackedOrigin](
                unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                    unsafe_ptr=self.ptr, length=self.length)))

        if _is_valid_utf8(self.ptr, self.length):
            return String(StringSpan[MutUntrackedOrigin](
                unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                    unsafe_ptr=self.ptr, length=self.length)))

        var s = String("")
        for k in range(self.length):
            var b = Int(self.ptr[unsafe_offset=k])
            if b < 128:
                s += chr(b)
            else:
                s += "?"
        return s

    @always_inline
    def raw_value(self) -> String:
        """Same as value() (byte for byte). Kept because gh #334 introduced it
        for the storing handlers, when value() was not byte-exact."""
        return self.value()

    @always_inline
    def value_bytes_ptr(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        return self.ptr

# #53: Redis's limit for a request line: an inline command, or the count
# line of a multibulk request or of one of its bulk strings.
comptime PROTO_INLINE_MAX_SIZE = 64 * 1024
# #53: a request Redis refuses with a protocol error and then closes the
# connection for. num_tokens == PROTOCOL_ERROR, need_tokens == the PE_* code.
comptime PROTOCOL_ERROR = -2
comptime PE_UNBALANCED_QUOTES = 1
comptime PE_BIG_INLINE = 2
comptime PE_MULTIBULK_LEN = 3
comptime PE_BULK_LEN = 4
comptime PE_BIG_MBULK_COUNT = 5
comptime PE_BIG_BULK_COUNT = 6
comptime PE_EXPECTED_DOLLAR = 256    # + the byte found where a `$` belonged


def protocol_error_text(code: Int) -> String:
    """The error Redis answers a PE_* protocol error with, without `-ERR `."""
    if code == PE_UNBALANCED_QUOTES:
        return "Protocol error: unbalanced quotes in request"
    if code == PE_BIG_INLINE:
        return "Protocol error: too big inline request"
    if code == PE_MULTIBULK_LEN:
        return "Protocol error: invalid multibulk length"
    if code == PE_BULK_LEN:
        return "Protocol error: invalid bulk length"
    if code == PE_BIG_MBULK_COUNT:
        return "Protocol error: too big mbulk count string"
    if code == PE_BIG_BULK_COUNT:
        return "Protocol error: too big bulk count string"
    if code >= PE_EXPECTED_DOLLAR:
        var b = code - PE_EXPECTED_DOLLAR
        var shown = chr(b) if b >= 32 and b < 127 else String("\\x") + chr(_hex_digit(b >> 4)) + chr(_hex_digit(b & 15))
        return "Protocol error: expected '$', got '" + shown + "'"
    return "Protocol error"


@always_inline
def _hex_digit(v: Int) -> Int:
    return v + 48 if v < 10 else v + 87


@always_inline
def _string2ll(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, mut ok: Bool) -> Int:
    """Redis's string2ll: an optional '-', then digits, no leading zero (but
    "0"), nothing else, within Int64. ok is False otherwise."""
    ok = False
    if n <= 0 or n > 20:
        return 0
    var i = 0
    var neg = False
    if p[unsafe_offset=0] == 45:          # '-'
        neg = True
        i = 1
        if n == 1:
            return 0
    var d0 = Int(p[unsafe_offset=i])
    if d0 < 48 or d0 > 57:
        return 0
    if d0 == 48:
        if n == i + 1 and not neg:
            ok = True
        return 0
    var v = 0
    while i < n:
        var d = Int(p[unsafe_offset=i])
        if d < 48 or d > 57:
            return 0
        if v > (9223372036854775807 - (d - 48)) // 10:
            return 0
        v = v * 10 + (d - 48)
        i += 1
    ok = True
    return -v if neg else v


@always_inline
def _is_space(b: Int) -> Bool:
    """C isspace: space, \t, \n, \v, \f, \r."""
    return b == 32 or (b >= 9 and b <= 13)


@always_inline
def _hex_val(b: Int) -> Int:
    if b >= 48 and b <= 57:
        return b - 48
    if b >= 97 and b <= 102:
        return b - 87
    if b >= 65 and b <= 70:
        return b - 55
    return -1


struct RESP3Parser:
    # #53: an inline argument written in quotes is not a slice of the request
    # (its quotes are gone, its escapes decoded), so it is decoded here and its
    # token points here. Tokens are only read until the next parse_stream call,
    # which reuses the bytes. Grown to the request's length at the first quoted
    # argument of a parse, so it never moves while a parse uses it: a script's
    # redis.call, which parses while its EVAL still uses its tokens, sends RESP
    # arrays and never reaches the inline path.
    var scratch: Pointer[UInt8, MutUntrackedOrigin]
    var scratch_cap: Int
    var scratch_used: Int

    def __init__(out self):
        self.scratch = null_ptr[UInt8, MutUntrackedOrigin]()
        self.scratch_cap = 0
        self.scratch_used = 0

    @always_inline
    def find_newline_simd(self, buffer: Pointer[UInt8, MutUntrackedOrigin], start: Int, length: Int) -> Int:
        var pos = start
        comptime simd_width = 16
        
        while pos + simd_width <= length:
            var chunk = (buffer.unsafe_offset(pos)).load[width=simd_width]()
            # Fast check if any newline (10) exists in this chunk
            if (chunk ^ 10).reduce_min() == 0:
                for i in range(simd_width):
                    if chunk[i] == 10:
                        return pos + i
            pos += simd_width
            
        for i in range(pos, length):
            if Int(buffer[unsafe_offset=i]) == 10:
                return i
        return -1

    def _split_inline(mut self, buffer: Pointer[UInt8, MutUntrackedOrigin], start: Int, end: Int,
                      length: Int, tokens: Pointer[RESP3Token, MutUntrackedOrigin],
                      mut num_tokens: Int, token_cap: Int) -> Int:
        """#53: split buffer[start, end) as Redis's sdssplitargs does: blanks
        separate arguments; "..." decodes \\xHH \\n \\r \\t \\b \\a and
        \\<c> -> c; '...' decodes \\'; a closing quote must be followed by a
        blank or the end; a NUL ends the line, as it ends the C string Redis
        splits. Stores tokens while num_tokens < token_cap. Returns how many
        arguments the line holds, or -1 for unbalanced quotes."""
        var p = start
        var argc = 0
        var lim = end
        for z in range(start, end):
            if buffer[unsafe_offset=z] == 0:
                lim = z
                break
        while True:
            while p < lim and _is_space(Int(buffer[unsafe_offset=p])):
                p += 1
            if p >= lim:
                return argc
            var tok_start = p
            var quoted = False
            var inq = False
            var insq = False
            # first pass: find where the argument ends and whether it is quoted
            var q = p
            while True:
                if q >= lim:
                    if inq or insq:
                        return -1
                    break
                var c = Int(buffer[unsafe_offset=q])
                if inq:
                    if c == 92 and q + 1 < lim:          # backslash: skip what it escapes
                        q += 2
                        continue
                    if c == 34:                          # closing "
                        if q + 1 < lim and not _is_space(Int(buffer[unsafe_offset=q + 1])):
                            return -1
                        q += 1
                        break
                    q += 1
                elif insq:
                    if c == 92 and q + 1 < lim and buffer[unsafe_offset=q + 1] == 39:
                        q += 2
                        continue
                    if c == 39:                          # closing '
                        if q + 1 < lim and not _is_space(Int(buffer[unsafe_offset=q + 1])):
                            return -1
                        q += 1
                        break
                    q += 1
                else:
                    if _is_space(c):
                        break
                    if c == 34:
                        inq = True
                        quoted = True
                    elif c == 39:
                        insq = True
                        quoted = True
                    q += 1
            var stored = num_tokens < token_cap
            if not quoted:
                if stored:
                    tokens[unsafe_offset=num_tokens] = RESP3Token(73, buffer.unsafe_offset(tok_start), q - tok_start)
            elif stored:
                # decode into the scratch (never longer than the raw argument)
                if self.scratch_cap < length:
                    if is_not_null(self.scratch):
                        self.scratch.free()
                    self.scratch = alloc[UInt8](length)
                    self.scratch_cap = length
                var out = self.scratch.unsafe_offset(self.scratch_used)
                var o = 0
                var r = tok_start
                inq = False
                insq = False
                while r < q:
                    var c = Int(buffer[unsafe_offset=r])
                    if inq:
                        if c == 92 and r + 3 < q and buffer[unsafe_offset=r + 1] == 120 \
                           and _hex_val(Int(buffer[unsafe_offset=r + 2])) >= 0 and _hex_val(Int(buffer[unsafe_offset=r + 3])) >= 0:
                            out[unsafe_offset=o] = UInt8(_hex_val(Int(buffer[unsafe_offset=r + 2])) * 16 + _hex_val(Int(buffer[unsafe_offset=r + 3])))
                            o += 1
                            r += 4
                            continue
                        if c == 92 and r + 1 < q:
                            var e = Int(buffer[unsafe_offset=r + 1])
                            var d = e
                            if e == 110: d = 10        # \n
                            elif e == 114: d = 13      # \r
                            elif e == 116: d = 9       # \t
                            elif e == 98: d = 8        # \b
                            elif e == 97: d = 7        # \a
                            out[unsafe_offset=o] = UInt8(d)
                            o += 1
                            r += 2
                            continue
                        if c == 34:
                            inq = False
                            r += 1
                            continue
                        out[unsafe_offset=o] = UInt8(c)
                        o += 1
                        r += 1
                    elif insq:
                        if c == 92 and r + 1 < q and buffer[unsafe_offset=r + 1] == 39:
                            out[unsafe_offset=o] = 39
                            o += 1
                            r += 2
                            continue
                        if c == 39:
                            insq = False
                            r += 1
                            continue
                        out[unsafe_offset=o] = UInt8(c)
                        o += 1
                        r += 1
                    else:
                        if c == 34:
                            inq = True
                        elif c == 39:
                            insq = True
                        else:
                            out[unsafe_offset=o] = UInt8(c)
                            o += 1
                        r += 1
                tokens[unsafe_offset=num_tokens] = RESP3Token(73, out, o)
                self.scratch_used += o
            if stored:
                num_tokens += 1
            argc += 1
            p = q

    def parse_stream(mut self, buffer: Pointer[UInt8, MutUntrackedOrigin], length: Int, tokens: Pointer[RESP3Token, MutUntrackedOrigin], mut num_tokens: Int, mut consumed_bytes: Int, cmd_ends: Pointer[Int, MutUntrackedOrigin], cmd_byte_ends: Pointer[Int, MutUntrackedOrigin], mut num_cmds: Int, mut need_tokens: Int, token_cap: Int = MAX_CMD_TOKENS) raises:
        # cmd_ends[k]      = token index one past command k's last token.
        # cmd_byte_ends[k] = buffer byte offset one past command k's last byte.
        # gh #162: the byte end lets the slow-path recover from a handler that
        # raises mid-batch by consuming exactly the offending command and
        # re-dispatching the rest, instead of discarding the whole recv buffer.
        #
        # #52: `tokens` holds `token_cap` entries. A command that does not fit
        # them sets num_tokens = TOKENS_OVERFLOW and need_tokens = its count, so
        # the caller can parse it alone into a table that size.
        # #53: a request Redis refuses with a protocol error sets num_tokens =
        # PROTOCOL_ERROR and need_tokens = its PE_* code; the caller answers
        # and closes the connection. Like an overflow, it is only reported
        # once no complete command precedes it in the buffer: those run first.
        num_tokens = 0
        consumed_bytes = 0
        num_cmds = 0
        need_tokens = 0
        self.scratch_used = 0
        if length <= 0: return

        var pos = 0
        while pos < length:
            var m_int = Int(buffer[unsafe_offset=pos])
            var tokens_checkpoint = num_tokens

            if m_int != 42:
                # #53: every line that does not start with `*` is an inline
                # command, as in Redis — not only lines that start with a
                # letter (`"PING"`, `  PING` and `123` got no reply at all, and
                # a `$3` line read the next line as its payload).
                var nl = self.find_newline_simd(buffer, pos, length)
                if nl == -1:
                    if length - pos > PROTO_INLINE_MAX_SIZE:
                        if tokens_checkpoint == 0:
                            num_tokens = PROTOCOL_ERROR
                            need_tokens = PE_BIG_INLINE
                    return
                var argc = self._split_inline(buffer, pos, nl, length, tokens, num_tokens, token_cap)
                if argc < 0:
                    num_tokens = tokens_checkpoint
                    if tokens_checkpoint == 0:
                        num_tokens = PROTOCOL_ERROR
                        need_tokens = PE_UNBALANCED_QUOTES
                    return
                if tokens_checkpoint + argc > token_cap:
                    # #52: more arguments than the table holds
                    num_tokens = tokens_checkpoint
                    if tokens_checkpoint == 0:
                        num_tokens = TOKENS_OVERFLOW
                        need_tokens = argc
                        consumed_bytes = nl + 1
                    return
                pos = nl + 1
                consumed_bytes = pos
                # An empty line is no command (Redis ignores it). gh #156: a
                # command needs a boundary entry, or every `i = cmd_end_tok - 1`
                # handler reads the end of the batch as its own.
                if argc > 0 and num_cmds < MAX_CMD_ENDS:
                    cmd_ends[unsafe_offset=num_cmds] = num_tokens
                    cmd_byte_ends[unsafe_offset=num_cmds] = pos
                    num_cmds += 1
                continue

            # `*<count>\r\n`, then <count> bulk strings: verify ALL are complete
            var line_end = self.find_newline_simd(buffer, pos + 1, length)
            if line_end == -1:
                if length - pos > PROTO_INLINE_MAX_SIZE and tokens_checkpoint == 0:
                    num_tokens = PROTOCOL_ERROR
                    need_tokens = PE_BIG_MBULK_COUNT
                return
            # #53: the count as Redis reads it (string2ll up to the \r);
            # anything else is a protocol error. The old loop skipped every
            # non-digit, so `*-1` read as 1 and `*2x` as 2.
            var count_ok = False
            var count = 0
            if Int(buffer[unsafe_offset=line_end - 1]) == 13:
                count = _string2ll(buffer.unsafe_offset(pos + 1), line_end - 1 - (pos + 1), count_ok)
            if not count_ok or count > MAX_ARRAY_LEN:   # gh #102 (C4): bound the per-command loop
                if tokens_checkpoint == 0:
                    num_tokens = PROTOCOL_ERROR
                    need_tokens = PE_MULTIBULK_LEN
                return
            if count <= 0:
                # `*0`, `*-1`: no command, as in Redis
                pos = line_end + 1
                consumed_bytes = pos
                continue
            var temp_pos = line_end + 1
            var all_ok = True
            var proto_err = 0
            # gh #153: set when this command has more tokens than the array
            # holds. We keep verifying (advancing temp_pos) without storing,
            # so the frame's true end is known and can be skipped.
            var overflowed = False
            # Verify and add each of the N bulk-string tokens
            for _ in range(count):
                if temp_pos >= length:
                    all_ok = False; break
                if buffer[unsafe_offset=temp_pos] != 36:
                    proto_err = PE_EXPECTED_DOLLAR + Int(buffer[unsafe_offset=temp_pos])
                    break
                var bl_end = self.find_newline_simd(buffer, temp_pos + 1, length)
                if bl_end == -1:
                    if length - temp_pos > PROTO_INLINE_MAX_SIZE:
                        proto_err = PE_BIG_BULK_COUNT
                    else:
                        all_ok = False
                    break
                var bl_ok = False
                var bl = 0
                if Int(buffer[unsafe_offset=bl_end - 1]) == 13:
                    bl = _string2ll(buffer.unsafe_offset(temp_pos + 1), bl_end - 1 - (temp_pos + 1), bl_ok)
                if not bl_ok or bl < 0 or bl > MAX_BULK_LEN:   # gh #102: bail before Int overflow
                    proto_err = PE_BULK_LEN
                    break
                var data_start = bl_end + 1
                if data_start + bl + 2 > length: all_ok = False; break
                if num_tokens < token_cap:
                    tokens[unsafe_offset=num_tokens] = RESP3Token(buffer[unsafe_offset=temp_pos], buffer.unsafe_offset(data_start), bl)
                    num_tokens += 1
                else:
                    # Token array full. Don't store, but keep walking the
                    # frame so `temp_pos` ends up past it. No `overflowed`
                    # guard needed above: num_tokens is frozen at the cap
                    # from here on, so the bound test alone keeps us here —
                    # and the per-token loop stays exactly as cheap as it
                    # was before gh #153.
                    overflowed = True
                temp_pos = data_start + bl + 2
            if proto_err != 0:
                num_tokens = tokens_checkpoint
                if tokens_checkpoint == 0:
                    num_tokens = PROTOCOL_ERROR
                    need_tokens = proto_err
                return
            if all_ok and overflowed:
                # gh #153: the frame is complete, it just doesn't fit.
                if tokens_checkpoint == 0:
                    # Nothing earlier to drain, so re-parsing can never make
                    # this fit — waiting for more data would hang the fd on
                    # a command that already arrived in full. Hand the
                    # caller the frame's length and the overflow signal (#52:
                    # and its size, to parse it alone into a table that big).
                    consumed_bytes = temp_pos
                    num_tokens = TOKENS_OVERFLOW
                    need_tokens = count
                else:
                    # Earlier commands in this batch did produce tokens.
                    # Let the caller execute and drain those; this command
                    # is re-parsed as the first one next call and then takes
                    # the branch above. consumed_bytes stays at the end of
                    # the last fully-parsed command.
                    num_tokens = tokens_checkpoint
                return
            if all_ok:
                pos = temp_pos
                consumed_bytes = pos  # complete command consumed up to here
                if num_cmds < MAX_CMD_ENDS:
                    cmd_ends[unsafe_offset=num_cmds] = num_tokens
                    cmd_byte_ends[unsafe_offset=num_cmds] = pos
                    num_cmds += 1
            else:
                # Partial command: undo any tokens added and stop parsing
                num_tokens = tokens_checkpoint
                return

        return
