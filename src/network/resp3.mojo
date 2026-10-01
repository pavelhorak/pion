from src.common.ptr import null_ptr
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
# and `consumed_bytes` covers it, so reply -ERR and skip it.
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

struct RESP3Parser:
    def __init__(out self):
        pass

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

    def parse_stream(self, buffer: Pointer[UInt8, MutUntrackedOrigin], length: Int, tokens: Pointer[RESP3Token, MutUntrackedOrigin], mut num_tokens: Int, mut consumed_bytes: Int, cmd_ends: Pointer[Int, MutUntrackedOrigin], cmd_byte_ends: Pointer[Int, MutUntrackedOrigin], mut num_cmds: Int) raises:
        # cmd_ends[k]      = token index one past command k's last token.
        # cmd_byte_ends[k] = buffer byte offset one past command k's last byte.
        # gh #162: the byte end lets the slow-path recover from a handler that
        # raises mid-batch by consuming exactly the offending command and
        # re-dispatching the rest, instead of discarding the whole recv buffer.
        num_tokens = 0
        consumed_bytes = 0
        num_cmds = 0
        if length <= 0: return

        var pos = 0
        while pos < length:
            var marker = buffer[unsafe_offset=pos]
            var m_int = Int(marker)

            var line_end = self.find_newline_simd(buffer, pos + 1, length)

            if line_end == -1: break

            if m_int == 42: # * — RESP array command: verify ALL N bulk strings are complete
                # Parse array count N
                var count = 0
                for i in range(pos + 1, line_end):
                    var ch = Int(buffer[unsafe_offset=i])
                    if ch >= 48 and ch <= 57:
                        count = count * 10 + (ch - 48)
                        if count > MAX_ARRAY_LEN: break  # gh #102: bound the per-command loop
                # Checkpoint: save token count before this command
                var tokens_checkpoint = num_tokens
                # gh #102 (C4): reject an oversized/wrapped array count outright
                # rather than iterating `range(count)` billions of times.
                if count > MAX_ARRAY_LEN:
                    num_tokens = tokens_checkpoint
                    return
                var temp_pos = line_end + 1
                var all_ok = True
                # gh #153: set when this command has more tokens than the array
                # holds. We keep verifying (advancing temp_pos) without storing,
                # so the frame's true end is known and can be skipped.
                var overflowed = False
                # Verify and add each of the N bulk-string tokens
                for _ in range(count):
                    if temp_pos >= length or buffer[unsafe_offset=temp_pos] != 36:
                        all_ok = False; break
                    var bl_end = self.find_newline_simd(buffer, temp_pos + 1, length)
                    if bl_end == -1: all_ok = False; break
                    var bl = 0
                    var bl_ovf = False
                    for i in range(temp_pos + 1, bl_end):
                        var ch = Int(buffer[unsafe_offset=i])
                        if ch >= 48 and ch <= 57:
                            bl = bl * 10 + (ch - 48)
                            if bl > MAX_BULK_LEN:  # gh #102: bail before Int overflow
                                bl_ovf = True; break
                    if bl_ovf: all_ok = False; break
                    var data_start = bl_end + 1
                    if data_start + bl + 2 > length: all_ok = False; break
                    if num_tokens < MAX_CMD_TOKENS:
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
                if all_ok and overflowed:
                    # gh #153: the frame is complete, it just doesn't fit.
                    if tokens_checkpoint == 0:
                        # Nothing earlier to drain, so re-parsing can never make
                        # this fit — waiting for more data would hang the fd on
                        # a command that already arrived in full. Hand the
                        # caller the frame's length and the overflow signal so
                        # it can reply -ERR and stay frame-synced.
                        consumed_bytes = temp_pos
                        num_tokens = TOKENS_OVERFLOW
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

            elif m_int == 36: # $ standalone bulk string (outside array)
                var bl = 0
                var is_neg = False
                for i in range(pos + 1, line_end):
                    var ch = Int(buffer[unsafe_offset=i])
                    if ch >= 48 and ch <= 57:
                        bl = bl * 10 + (ch - 48)
                        if bl > MAX_BULK_LEN:  # gh #102: bail before Int overflow → OOB read
                            break
                    elif ch == 45: # -
                        is_neg = True

                if is_neg: bl = -1
                elif bl > MAX_BULK_LEN: break  # gh #102: oversized bulk length — stop parsing

                pos = line_end + 1
                if bl >= 0:
                    if pos + bl + 2 > length: break  # incomplete standalone bulk string
                    tokens[unsafe_offset=num_tokens] = RESP3Token(marker, buffer.unsafe_offset(pos), bl)
                    num_tokens += 1
                    if num_tokens >= 64: return
                    pos += bl
                    if pos < length and Int(buffer[unsafe_offset=pos]) == 13: pos += 1
                    if pos < length and Int(buffer[unsafe_offset=pos]) == 10: pos += 1
                    consumed_bytes = pos
                else:
                    tokens[unsafe_offset=num_tokens] = RESP3Token(marker, null_ptr[UInt8, MutUntrackedOrigin](), 0)
                    num_tokens += 1
                    if num_tokens >= 64: return
                    consumed_bytes = pos

            elif m_int == 43 or m_int == 45 or m_int == 58: # +, -, :
                var s_start = pos + 1
                var s_end = line_end
                if s_end > s_start and Int(buffer[unsafe_offset=s_end-1]) == 13:
                    s_end -= 1

                tokens[unsafe_offset=num_tokens] = RESP3Token(marker, buffer.unsafe_offset(s_start), s_end - s_start)
                num_tokens += 1
                if num_tokens >= 64: return
                pos = line_end + 1
                consumed_bytes = pos

            elif (m_int >= 65 and m_int <= 90) or (m_int >= 97 and m_int <= 122):
                var s_end = line_end
                if s_end > pos and Int(buffer[unsafe_offset=s_end-1]) == 13:
                    s_end -= 1

                # Split inline command by spaces
                var tok_start = pos
                for i in range(pos, s_end):
                    if Int(buffer[unsafe_offset=i]) == 32: # space
                        if i > tok_start:
                            tokens[unsafe_offset=num_tokens] = RESP3Token(73, buffer.unsafe_offset(tok_start), i - tok_start)
                            num_tokens += 1
                            if num_tokens >= 64: return
                        tok_start = i + 1

                if s_end > tok_start:
                    tokens[unsafe_offset=num_tokens] = RESP3Token(73, buffer.unsafe_offset(tok_start), s_end - tok_start)
                    num_tokens += 1
                    if num_tokens >= 64: return

                # gh #156: an inline command is a command, so it needs a
                # boundary entry too. Without one `num_cmds` stayed 0 for an
                # all-inline batch and every `i = cmd_end_tok - 1` handler read
                # the fallback `num_tokens` — i.e. the end of the *batch* — and
                # ate every command behind it.
                if num_cmds < MAX_CMD_ENDS:
                    cmd_ends[unsafe_offset=num_cmds] = num_tokens
                    cmd_byte_ends[unsafe_offset=num_cmds] = line_end + 1
                    num_cmds += 1

                pos = line_end + 1
                consumed_bytes = pos
            else:
                # Unknown marker: skip to next newline
                pos = line_end + 1
                consumed_bytes = pos

        return
