from std.collections import Span
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call
from std.memory import alloc, stack_allocation
from std.sys import CompilationTarget

# A flattened string containing "00010203...9899"
comptime DIGIT_LUT = "00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899"
comptime INT64_MIN_STR = "-9223372036854775808"

@always_inline
def arg_eq(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int, lit: StringLiteral) -> Bool:
    """Whole-name, case-insensitive match for an option KEYWORD (gh #251).

    The same discipline gh #225 imposed on command names, applied to the
    keyword arguments that select a command's behaviour. Handlers here have
    matched option keywords as "length plus first byte", which cannot tell
    `BYLEX` from `BYFOO` and would silently pick a branch on a typo — and for
    `ZRANGE` the branch decides whether `min`/`max` mean ranks or scores.

    `lit` must be lowercase ASCII. The fold is an explicit A-Z range test, not
    `| 0x20`, for the reason recorded on `cmd_eq`: that trick maps `_` to 0x7F,
    so a keyword containing an underscore cannot be written with it."""
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


@always_inline
def int_string_len(val: Int64) -> Int:
    var n = val
    if n == 0: return 1
    var length = 0
    if n < 0:
        # -n is a NO-OP for Int64 MIN (its magnitude has no positive Int64), so
        # n stays negative and every `n < 10^k` test below matches immediately.
        # See format_int_to_buf: this is the length half of the same bug.
        if n == -9223372036854775808: return 20
        length += 1
        n = -n

    # Check small values first (most common: string lengths, array counts, small integers)
    if n < 10: return length + 1
    if n < 100: return length + 2
    if n < 1000: return length + 3
    if n < 10000: return length + 4
    if n < 100000: return length + 5
    if n < 1000000: return length + 6
    if n < 10000000: return length + 7
    if n < 100000000: return length + 8
    if n < 1000000000: return length + 9
    if n < 10000000000: return length + 10
    if n < 100000000000: return length + 11
    if n < 1000000000000: return length + 12
    if n < 10000000000000: return length + 13
    if n < 100000000000000: return length + 14
    if n < 1000000000000000: return length + 15
    if n < 10000000000000000: return length + 16
    if n < 100000000000000000: return length + 17
    if n < 1000000000000000000: return length + 18
    return length + 19

@always_inline
def format_int_to_buf(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int, val: Int64) -> Int:
    var n = val
    var new_offset = offset

    if n == 0:
        buf[unsafe_offset=new_offset] = 48 # '0'
        return new_offset + 1

    if n < 0:
        # Int64 MIN: `-n` overflows back to itself, leaving n negative. The
        # `n < 10` fast path below then matched and emitted UInt8(48 + n),
        # which truncated to '0' — so a value that reached exactly
        # -9223372036854775808 was reported as 0 and STORED as the string "-0"
        # (measured: SET k -9223372036854775807 ; DECRBY k 1 -> 0, GET -> "-0").
        # Emit the literal; it is the one Int64 with no negatable magnitude.
        if n == -9223372036854775808:
            var min_lit = INT64_MIN_STR.unsafe_ptr()
            for mi in range(20):
                buf[unsafe_offset=new_offset + mi] = min_lit[unsafe_offset=mi]
            return new_offset + 20
        buf[unsafe_offset=new_offset] = 45 # '-'
        new_offset += 1
        n = -n

    # Fast path: single digit (most common for string lengths and small counts)
    if n < 10:
        buf[unsafe_offset=new_offset] = UInt8(48 + n)
        return new_offset + 1

    var num_len = int_string_len(n) 
    var curr_idx = new_offset + num_len - 1
    
    var lut_ptr = DIGIT_LUT.unsafe_ptr()
    
    # Process two digits per iteration using multiplication tricks instead of division
    while n >= 100:
        var next_n = n // 100 
        var rem = Int(n - (next_n * 100)) # Avoid modulo!
        
        var lut_idx = rem * 2
        buf[unsafe_offset=curr_idx] = lut_ptr[unsafe_offset=lut_idx + 1]
        buf[unsafe_offset=curr_idx - 1] = lut_ptr[unsafe_offset=lut_idx]
        
        curr_idx -= 2
        n = next_n
        
    # Handle the remaining 1 or 2 digits
    if n < 10:
        buf[unsafe_offset=curr_idx] = UInt8(48 + n)
    else:
        var lut_idx = Int(n) * 2
        buf[unsafe_offset=curr_idx] = lut_ptr[unsafe_offset=lut_idx + 1]
        buf[unsafe_offset=curr_idx - 1] = lut_ptr[unsafe_offset=lut_idx]
        
    return new_offset + num_len

@always_inline
def _format_big_integral(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int, v_in: Float64) -> Int:
    """Exact decimal digits of a finite, non-negative, integral double >= 2^63.

    format_float64_to_buf took the integer part with Int64(v), which does not
    hold past 2^63: `INCRBYFLOAT k 1e19` stored "9223372036854775807.////…"
    (the saturated integer, then '/' = '0' - 1 digits from the bad fraction).
    Redis prints such values in full, with no exponent. Cold path: division
    and a List are fine here, unlike the per-reply integer formatter.

    v = m * 2^t with m < 2^53 exactly (halving an integral double is exact),
    then the base-10^9 limbs of m are doubled t times, 29 bits at a time."""
    var v = v_in
    var t = 0
    while v >= 9007199254740992.0:          # 2^53
        v = v / 2.0
        t += 1
    var m = UInt64(v)
    var limbs = List[UInt64]()
    limbs.append(m % 1000000000)
    limbs.append((m // 1000000000) % 1000000000)
    limbs.append(m // 1000000000000000000)
    while t > 0:
        var sh = min(t, 29)
        t -= sh
        var carry: UInt64 = 0
        for li in range(len(limbs)):
            var x = (limbs[li] << UInt64(sh)) + carry
            limbs[li] = x % 1000000000
            carry = x // 1000000000
        while carry > 0:
            limbs.append(carry % 1000000000)
            carry = carry // 1000000000
    var top = len(limbs) - 1
    while top > 0 and limbs[top] == 0:
        top -= 1
    var off = format_int_to_buf(buf, offset, Int64(limbs[top]))
    var li2 = top - 1
    while li2 >= 0:
        var d = limbs[li2]
        for k in range(8, -1, -1):
            buf[unsafe_offset=off + k] = UInt8(48 + Int(d % 10))
            d = d // 10
        off += 9
        li2 -= 1
    return off


@always_inline   # callers pass stack_allocation buffers (gh #349): must not stay out of line
def format_float64_to_buf(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int, val: Float64, precision: Int = 17, trim: Bool = True) -> Int:
    """Float64 twin of `format_float_to_buf`, for values that are STORED back.

    gh #232: INCRBYFLOAT did its arithmetic in Float32, whose 24-bit mantissa
    cannot represent 100000001 — so `SET k 100000000; INCRBYFLOAT k 1` replied
    `100000000`, silently dropping the increment while looking like a success.
    Any counter past ~16.7M simply stopped moving. The result is re-serialized
    into the keyspace, so the format is the storage precision too and 6 decimals
    truncated pi to 3.141592 on a round-trip.

    Kept separate rather than widening `format_float_to_buf`, whose Float32
    signature is shared with the GEO and vector paths where the narrower type
    is deliberate and the output is not stored.
    """
    var v = val
    var off = offset
    if v != v:                               # NaN — callers that store refuse it first
        buf[unsafe_offset=off] = 110; buf[unsafe_offset=off + 1] = 97; buf[unsafe_offset=off + 2] = 110
        return off + 3
    if v < 0:
        buf[unsafe_offset=off] = 45 # '-'
        off += 1
        v = -v
    if v > 1.7976931348623157e308:           # inf
        buf[unsafe_offset=off] = 105; buf[unsafe_offset=off + 1] = 110; buf[unsafe_offset=off + 2] = 102
        return off + 3
    if v >= 9223372036854775808.0:          # 2^63: integral, and past Int64
        return _format_big_integral(buf, off, v)

    # Digit extraction below TRUNCATES, but printf-style formatting ROUNDS —
    # `%.4f` of 166.27415… is 166.2742, not 166.2741. Adding half a unit in the
    # last place first turns truncation into round-half-up. At precision 17 the
    # addend is far below double resolution, so the shortest-repr path is
    # unaffected.
    if precision > 0:
        var half = 0.5
        for _ in range(precision): half /= 10.0
        v += half

    var ipart = Int64(v)
    off = format_int_to_buf(buf, off, ipart)

    if precision > 0:
        var frac_start = off
        buf[unsafe_offset=off] = 46 # '.'
        off += 1
        var fpart = v - Float64(ipart)
        for _ in range(precision):
            fpart *= 10
            var digit = Int(fpart)
            buf[unsafe_offset=off] = UInt8(48 + digit)
            off += 1
            fpart -= Float64(digit)
        # Trim trailing zeros, then a bare '.' — an integral result must come
        # back as "9", not "9.00000000000000000". GEODIST opts OUT: Redis
        # prints distances with `%.4f`, so a zero distance is "0.0000" and the
        # trailing zeros are part of the contract.
        if trim:
            while off > frac_start + 1 and buf[unsafe_offset=off - 1] == 48:
                off -= 1
            if off == frac_start + 1: off = frac_start

    return off


@always_inline
def format_float_to_buf(buf: Pointer[UInt8, MutUntrackedOrigin], offset: Int, val: Float32, precision: Int = 4) -> Int:
    var v = val
    var off = offset
    if v < 0:
        buf[unsafe_offset=off] = 45 # '-'
        off += 1
        v = -v
    
    var ipart = Int64(v)
    off = format_int_to_buf(buf, off, ipart)
    
    if precision > 0:
        buf[unsafe_offset=off] = 46 # '.'
        off += 1
        var fpart = v - Float32(ipart)
        for _ in range(precision):
            fpart *= 10
            var digit = Int(fpart)
            buf[unsafe_offset=off] = UInt8(48 + digit)
            off += 1
            fpart -= Float32(digit)
            
    return off


@always_inline
def score_prints_as_int(score: Float64) -> Bool:
    """True when Redis prints a sorted-set score as a bare integer: its d2string
    takes the integer path only for integral values within ±2^62 (double2ll's
    LLONG_MAX/2 bound). Everything else goes through `format_score`.

    #18: every score emitter used to test `Float64(Int64(score)) == score`.
    Converting ±inf (or anything past Int64) to Int64 is undefined; LLVM folds
    the round trip into a truncation test that ±inf passes, and x86's
    cvttsd2si then yields INT64_MIN, so `ZADD k inf m` read back as
    -9223372036854775808. The range test comes first and short-circuits, so
    the conversion only ever sees values it can represent (NaN fails it too)."""
    return score >= -4611686018427387904.0 and score <= 4611686018427387904.0 \
        and Float64(Int64(score)) == score


def format_score(score: Float64) -> String:
    """Redis's d2string for a score `score_prints_as_int` rejects: `inf`,
    `-inf`, `nan`, and otherwise the shortest round-trip digits laid out as its
    fpconv_dtoa lays them out: plain digits up to 6 places past the last
    significant one (`9223372036854776000`), plain decimals down to 1e-6
    (`0.00001`), and `1e+20` / `1.5e-7` beyond those.

    Known residual: fpconv is Grisu2, which for about 0.1% of full-precision
    doubles emits a longer or differently-rounded digit string than the
    shortest one (`-6016.9512179398635` for -6016.951217939863). Both parse
    back to the same double; Pion emits the shortest, as Python's repr does."""
    if score != score:
        return "nan"
    if score > 1.7976931348623157e308:
        return "inf"
    if score < -1.7976931348623157e308:
        return "-inf"
    if score == 0.0:
        return "0"
    # Shortest digits and the decimal exponent come from Mojo's own formatter
    # ("1.5", "1e-05", "4.611686018427388e+18", "9007199254740992.0").
    var s = String(score)
    var p = s.unsafe_ptr()
    var n = s.byte_length()
    var i = 0
    var neg = False
    if n > 0 and p[0] == 45:   # '-'
        neg = True
        i = 1
    var digits = String("")
    var point = -1            # count of mantissa digits before the '.'
    var nd_raw = 0
    var exp10 = 0
    while i < n:
        var c = p[i]
        if c == 46:           # '.'
            point = nd_raw
        elif c == 101 or c == 69:   # 'e' / 'E'
            i += 1
            var eneg = False
            if i < n and (p[i] == 45 or p[i] == 43):
                eneg = p[i] == 45
                i += 1
            while i < n:
                exp10 = exp10 * 10 + Int(p[i] - 48)
                i += 1
            if eneg:
                exp10 = -exp10
            break
        else:
            digits += chr(Int(c))
            nd_raw += 1
        i += 1
    if point < 0:
        point = nd_raw
    # value = digits * 10^K with digits stripped of leading and trailing zeros
    var dp = digits.unsafe_ptr()
    var lo = 0
    while lo < nd_raw - 1 and dp[lo] == 48:
        lo += 1
    var hi = nd_raw
    var K = point - nd_raw + exp10
    while hi > lo + 1 and dp[hi - 1] == 48:
        hi -= 1
        K += 1
    var nd = hi - lo
    var out = String("-") if neg else String("")
    var e = K + nd - 1
    var ae = e if e >= 0 else -e
    if K >= 0 and ae < nd + 7:
        for j in range(lo, hi):
            out += chr(Int(dp[j]))
        for _ in range(K):
            out += "0"
        return out
    if K < 0 and (K > -7 or ae < 4):
        var offset = nd + K
        if offset <= 0:
            out += "0."
            for _ in range(-offset):
                out += "0"
            for j in range(lo, hi):
                out += chr(Int(dp[j]))
        else:
            for j in range(lo, lo + offset):
                out += chr(Int(dp[j]))
            out += "."
            for j in range(lo + offset, hi):
                out += chr(Int(dp[j]))
        return out
    out += chr(Int(dp[lo]))
    if nd > 1:
        out += "."
        for j in range(lo + 1, hi):
            out += chr(Int(dp[j]))
    out += "e-" if e < 0 else "e+"
    out += String(ae)
    return out


@always_inline
def set_thread_qos_user_interactive():
    # Step 5: Pin this thread to P-cores (high-performance cores) on Apple Silicon.
    # pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE=0x21, relative_priority=0)
    # macOS scheduler maps QOS_CLASS_USER_INTERACTIVE exclusively to P-core cluster.
    # Must be called before heavy compute — QoS overrides all other scheduler hints.
    comptime if not CompilationTarget.is_linux():
        _ = external_call["pthread_set_qos_class_self_np", Int32](UInt32(0x21), Int32(0))

@always_inline
def set_thread_affinity(cpu_id: Int) -> String:
    """Apply `--affinity` to the calling worker thread and say what was applied
    ("" when nothing was). Linux pins the thread to one CPU; macOS has no hard
    pinning and sets an affinity tag, a scheduling hint (#20: Linux used to do
    nothing here while the caller logged "pinned to CPU i")."""
    comptime if CompilationTarget.is_linux():
        var c = Int(external_call["pion_pin_current_thread", Int32](Int32(cpu_id)))
        if c < 0:
            return ""
        return "pinned to CPU " + String(c)
    else:
        # macOS THREAD_AFFINITY_POLICY = 4, THREAD_AFFINITY_POLICY_COUNT = 1
        var policy = cpu_id
        var thread = external_call["mach_thread_self", UInt32]()
        var policy_ptr = alloc[Int](1)
        policy_ptr[unsafe_offset=0] = policy
        var kr = external_call["thread_policy_set", Int32](
            thread,
            4, # THREAD_AFFINITY_POLICY
            policy_ptr,
            1  # THREAD_AFFINITY_POLICY_COUNT
        )
        policy_ptr.unsafe_free()
        if kr != 0:
            return ""
        return "affinity tag " + String(cpu_id) + " (a macOS scheduling hint)"

comptime CMD_GET = 0x00746567
comptime CMD_MGET = 0x7465676d
comptime CMD_MSET = 0x7465736d
comptime CMD_SET = 0x00746573
comptime CMD_PING = 0x676e6970
comptime CMD_INCR = 0x72636e69
comptime CMD_HSET = 0x74657368
comptime CMD_XADD = 0x64646178
comptime CMD_LPUSH = 0x000000687375706c
comptime CMD_RPUSH = 0x0000006873757072
comptime CMD_LPOP = 0x706f706c
comptime CMD_RPOP = 0x706f7072
comptime CMD_SADD = 0x64646173
comptime CMD_SPOP = 0x706f7073
comptime CMD_ZADD = 0x6464617a
comptime CMD_ZPOPMIN = 0x006e696d706f707a
comptime CMD_LRANGE = 0x000065676e61726c
comptime CMD_FCALL = 0x0000006c6c616366
comptime CMD_FUNCTION = 0x6e6f6974636e7566


# NOT @always_inline — this recurses on '*' and Mojo rejects a recursive
# always_inline function. KEYS/SCAN are O(N) by definition; the call overhead
# is irrelevant next to the keyspace walk.
def _glob_match(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int, pi_in: Int,
                s: Pointer[UInt8, MutUntrackedOrigin], slen: Int, si_in: Int) -> Bool:
    """Redis-compatible glob: `*`, `?`, `[abc]`, `[a-z]`, `[^abc]`, `\\` escape.

    gh #243: KEYS and SCAN MATCH previously IGNORED the pattern and returned the
    whole keyspace — `KEYS zzz` listed every key. Not a hot path (KEYS is O(N)
    by definition), so this is written for clarity rather than speed."""
    var pi = pi_in
    var si = si_in
    while pi < plen:
        var pc = p[unsafe_offset=pi]
        if pc == 42:                      # '*' — try every split point
            while pi < plen and p[unsafe_offset=pi] == 42:
                pi += 1
            if pi == plen:
                return True
            var k = si
            while k <= slen:
                if _glob_match(p, plen, pi, s, slen, k):
                    return True
                k += 1
            return False
        if si >= slen:
            return False
        if pc == 63:                      # '?'
            pi += 1; si += 1; continue
        if pc == 91:                      # '[' char class
            var j = pi + 1
            var neg = False
            if j < plen and p[unsafe_offset=j] == 94:
                neg = True; j += 1
            var matched = False
            var first = True
            while j < plen and (p[unsafe_offset=j] != 93 or first):
                first = False
                if j + 2 < plen and p[unsafe_offset=j+1] == 45 and p[unsafe_offset=j+2] != 93:
                    if s[unsafe_offset=si] >= p[unsafe_offset=j] and s[unsafe_offset=si] <= p[unsafe_offset=j+2]:
                        matched = True
                    j += 3
                else:
                    if p[unsafe_offset=j] == s[unsafe_offset=si]:
                        matched = True
                    j += 1
            if j < plen:
                j += 1                    # step past ']'
            if neg:
                matched = not matched
            if not matched:
                return False
            pi = j; si += 1; continue
        if pc == 92 and pi + 1 < plen:    # '\' escape
            pi += 1
            pc = p[unsafe_offset=pi]
        if pc != s[unsafe_offset=si]:
            return False
        pi += 1; si += 1
    return si == slen


@always_inline
def _glob_all(pat: Pointer[UInt8, MutUntrackedOrigin], plen: Int) -> Bool:
    """True when the pattern is `*` (or all stars) — the match-everything case,
    worth short-circuiting since it is by far the most common KEYS argument."""
    if plen == 0:
        return False
    for k in range(plen):
        if pat[unsafe_offset=k] != 42:
            return False
    return True


comptime I64_MAX = Int64(9223372036854775807)
comptime I64_MIN = Int64(-9223372036854775807) - 1


@always_inline
def ms_to_deadline_ns(when_ms: Int64) -> Int64:
    """Redis keeps a deadline in ms, Pion in ns, and ns run out in 2262. A
    deadline Redis accepts but ns cannot hold is stored as the latest one ns
    can (gh #393) — the key keeps a TTL that outlives anything running now —
    instead of the error it used to be, or the wrap it was before that."""
    if when_ms > I64_MAX // 1_000_000:
        return I64_MAX
    if when_ms < -(I64_MAX // 1_000_000):     # C's truncating bound; `//` floors
        return I64_MIN
    return when_ms * 1_000_000


comptime SETEXP_INVALID = 0    # "ERR invalid expire time in '<cmd>' command"; nothing written
comptime SETEXP_DEADLINE = 1   # set the key with `ns` as its deadline
comptime SETEXP_EXPIRED = 2    # reply as if set, and leave the key absent


@fieldwise_init
struct SetExpiry(Copyable, Movable, ImplicitlyCopyable):
    var status: Int
    var ns: Int64


@always_inline
def set_expiry(v: Int64, unit_ms: Int64, relative: Bool, now_ns: Int64) -> SetExpiry:
    """The TTL argument of SET EX|PX|EXAT|PXAT, SETEX and PSETEX, resolved the
    way Redis's getExpireMillisecondsOrReply does (gh #393): non-positive, or
    seconds that cannot scale to ms, is an error, and so is a relative time
    whose sum with now overflows ms (its "Overflow detected" check). That last
    check reads a signed overflow, so a clang build of Redis (macOS) may drop
    it and answer +OK with the key already expired; Linux builds answer the
    error, which is what the check is for. A deadline already past is +OK and
    the key is gone."""
    if v <= 0 or (unit_ms != 1 and v > I64_MAX // unit_ms):
        return SetExpiry(SETEXP_INVALID, 0)
    var ms = v * unit_ms
    var now_ms = now_ns // 1_000_000
    if relative:
        if ms > I64_MAX - now_ms:
            return SetExpiry(SETEXP_INVALID, 0)
        ms += now_ms
    if ms <= now_ms:
        return SetExpiry(SETEXP_EXPIRED, 0)
    return SetExpiry(SETEXP_DEADLINE, ms_to_deadline_ns(ms))


@fieldwise_init
struct ParsedFloat(Copyable, Movable, ImplicitlyCopyable):
    """A Redis-faithful double parse (gh #393): the value and whether the
    argument was one. Branch on `ok`."""
    var value: Float64
    var ok: Bool


# pion_parse_double modes (src/ffi/fcntl_wrap.c holds the rules).
comptime DOUBLE_VALUE = 0      # a value argument: ZADD score, ZINCRBY, GEO*, …
comptime DOUBLE_RANGE = 1      # a range bound after "(": ZCOUNT, Z*RANGEBYSCORE
comptime DOUBLE_LONG = 2       # INCRBYFLOAT / HINCRBYFLOAT (long double rules)


@always_inline
def parse_redis_double(p: Pointer[UInt8, _], n: Int, mode: Int) -> ParsedFloat:
    """Parse a float argument exactly as Redis does — strtod plus the checks
    of the matching Redis entry point (see pion_parse_double). The hand-rolled
    parsers rejected what Redis accepts ("inf", "0x10", "+.5") and took what it
    refuses ("1e400", "nan", "1.5 "). The out-parameter is a stack slot handed
    to C, which is safe (gh #349 is about out-of-line MOJO callees)."""
    var out = stack_allocation[1, Float64]()
    var ok = external_call["pion_parse_double", Int32](p, Int64(n), Int32(mode), out) == 1
    return ParsedFloat(out[0] if ok else 0.0, ok)


@fieldwise_init
struct ParsedInt(Copyable, Movable):
    """A strict integer parse: the value, plus whether the input was an integer
    at all. Callers MUST branch on `ok` — that is the whole point of the type."""
    var value: Int64
    var ok: Bool


@always_inline
@always_inline
def strict_atol(s: String) raises -> Int:
    """A client's integer ARGUMENT, parsed as Redis parses it — or raise
    "ERR value is not an integer or out of range" (the slow path's recovery
    `except` forwards that text as the reply).

    `atol` accepts " 1", "1 ", "+1", "01" and "-0", all of which Redis
    refuses, and handlers took them as numbers: `LTRIM k " 1" -1` trimmed,
    `EXPIREAT k " 1"` deleted the key. The loop is inline on purpose: `s` is
    usually a short String whose bytes live INLINE in a stack slot, and that
    pointer must not cross an out-of-line call (gh #349)."""
    var n = s.byte_length()
    var p = s.unsafe_ptr()
    if n == 0:
        raise Error("ERR value is not an integer or out of range")
    var i = 0
    var neg = False
    if p[0] == 45:
        neg = True
        i = 1
        if n == 1 or (n == 2 and p[1] == 48):          # "-" and "-0"
            raise Error("ERR value is not an integer or out of range")
    if p[i] == 48 and n - i > 1:                         # leading zero: "01"
        raise Error("ERR value is not an integer or out of range")
    var acc: Int64 = 0
    while i < n:
        var c = Int(p[i])
        if c < 48 or c > 57:
            raise Error("ERR value is not an integer or out of range")
        var d = Int64(c - 48)
        if acc > 922337203685477580 or (acc == 922337203685477580 and d > 7):
            if neg and acc == 922337203685477580 and d == 8 and i + 1 == n:
                return Int(Int64(-9223372036854775807) - 1)
            raise Error("ERR value is not an integer or out of range")
        acc = acc * 10 + d
        i += 1
    return Int(-acc) if neg else Int(acc)


def rand_count(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) raises -> Int:
    """SRANDMEMBER / HRANDFIELD / ZRANDMEMBER count: an integer in
    [-LONG_MAX, LONG_MAX] — Redis refuses -2^63 because a negative count is
    negated (gh #393)."""
    var r = parse_int64_strict(p, n)
    if not r.ok:
        raise Error("ERR value is not an integer or out of range")
    if r.value == Int64(-9223372036854775807) - 1:
        raise Error("ERR value is out of range")
    return Int(r.value)


def scan_cursor(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) raises -> Int:
    """A SCAN-family cursor as Redis reads it — strtoull: leading whitespace, a
    "+" and leading zeros are fine ("-0" too), anything left over or a
    negative number is `ERR invalid cursor` (gh #393; `atol` took "-1" and
    "1 ", and the differential showed Redis takes " 1", "+1", "01")."""
    var i = 0
    while i < n and (p[unsafe_offset=i] == 32 or (p[unsafe_offset=i] >= 9 and p[unsafe_offset=i] <= 13)):
        i += 1
    var neg = False
    if i < n and (p[unsafe_offset=i] == 43 or p[unsafe_offset=i] == 45):
        neg = p[unsafe_offset=i] == 45
        i += 1
    if i >= n:
        raise Error("ERR invalid cursor")
    var acc: Int64 = 0
    while i < n:
        var c = Int(p[unsafe_offset=i])
        if c < 48 or c > 57:
            raise Error("ERR invalid cursor")
        if acc > 922337203685477580 or (acc == 922337203685477580 and c - 48 > 7):
            raise Error("ERR invalid cursor")
        acc = acc * 10 + Int64(c - 48)
        i += 1
    if neg and acc != 0:
        raise Error("ERR invalid cursor")
    return Int(acc)


def scan_count(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) raises -> Int:
    """A SCAN-family COUNT: an integer (else the integer error) of at least 1
    (else `ERR syntax error`, as Redis). SSCAN/ZSCAN skipped the value without
    reading it, so `COUNT abc` and `COUNT -1` were accepted (gh #393)."""
    var r = parse_int64_strict(p, n)
    if not r.ok:
        raise Error("ERR value is not an integer or out of range")
    if r.value < 1:
        raise Error("ERR syntax error")
    return Int(r.value)


@always_inline
def bytes_to_string(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> String:
    """A String holding exactly these n bytes (n <= 0 -> "").

    The loops this replaces spelled each byte `chr(Int(b))`, which is the
    CODEPOINT b: every byte >= 0x80 became two UTF-8 bytes. A lex bound like
    `(b\\xff` therefore never equalled the stored member b"b\\xff" and the
    exclusive range included it; HGETALL/HGET never matched a non-ASCII
    field's TTL entry; CLUSTER GETKEYSINSLOT returned re-encoded key names.
    The String is a byte container — compare it bytewise, never decode it."""
    if n <= 0:
        return String("")
    return String(StringSpan[MutUntrackedOrigin](
        unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=p, length=n)))


@always_inline
def acc_digit_checked(acc: Int64, d: Int64, neg: Bool, last: Bool) -> Int64:
    """One step of a STORED-value integer parse: acc * 10 + d, or -1 when that
    would leave Int64 (acc is the non-negative magnitude, so -1 is free).

    gh #229 made the ARGUMENT parse strict; the stored-value loops in INCR,
    DECR, INCRBY and DECRBY still accumulated with no bound, so a counter
    stored as "9999999999999999999" (SET, or APPEND to a digit string) went
    INCR -> -8446744073709551616 and was written back — Redis answers
    "ERR value is not an integer or out of range" and leaves it. Found by
    tests/test_boundary_differential.py. The one legal overflow, the final
    digit of -9223372036854775808, returns Int64 MIN, which the caller's
    `-parsed_val` leaves unchanged. No division: this sits on the fast path."""
    if acc > 922337203685477580 or (acc == 922337203685477580 and d > 7):
        if neg and last and acc == 922337203685477580 and d == 8:
            return Int64(-9223372036854775807) - 1
        return -1
    return acc * 10 + d


def parse_int64_strict(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int) -> ParsedInt:
    """Parse a RESP integer ARGUMENT, rejecting anything that is not exactly an
    optionally-signed run of digits that fits in an Int64.

    The hand-rolled loops this replaces accumulated digits and *skipped*
    everything else, with no overflow check, so on 0.915 every one of these was
    accepted silently (measured):

        INCRBY k 1abc2  -> +12      (letters skipped, digits concatenated)
        INCRBY k 1e3    -> +13      INCRBY k 0x10  -> +10
        INCRBY k 1,000  -> +1000    INCRBY k '  5' -> +5
        INCRBY k abc    -> +0       INCRBY k ''    -> +0   (reply looks like success)
        INCRBY k 99999999999999999999 -> +7766279631452241920  (wrapped)

    The `abc`/empty cases are the dangerous ones: a client whose delta variable
    stringified to garbage gets an integer reply and believes the counter moved.
    Redis errors on all of the above. Note the *stored value* parse in the same
    handlers already rejected non-digits correctly — only the argument parse
    was unchecked, which is why this went unnoticed.

    Overflow is detected before it happens (compare against MAX/10 rather than
    letting the multiply wrap), so no wrapped value can escape."""
    if plen == 0:
        return ParsedInt(0, False)
    var i = 0
    var neg = False
    if p[unsafe_offset=0] == 45:            # '-'
        neg = True
        i = 1
    if i >= plen:                            # a lone "-" is not a number
        return ParsedInt(0, False)
    # Redis's string2ll: only "0" itself may start with 0, so "01", "007" and
    # "-0" are not integers (INCRBY k 01 is an error there, and added 1 here).
    if p[unsafe_offset=i] == 48 and (plen - i > 1 or neg):
        return ParsedInt(0, False)
    var acc: Int64 = 0
    while i < plen:
        var c = Int(p[unsafe_offset=i])
        if c < 48 or c > 57:
            return ParsedInt(0, False)       # reject, never skip
        var d = Int64(c - 48)
        # Int64 max is 9223372036854775807. Check BEFORE multiplying.
        if acc > 922337203685477580 or (acc == 922337203685477580 and d > 7):
            # -9223372036854775808 is one past the positive limit; allow it.
            if not (neg and acc == 922337203685477580 and d == 8 and i + 1 == plen):
                return ParsedInt(0, False)
            return ParsedInt(-9223372036854775808, True)
        acc = acc * 10 + d
        i += 1
    return ParsedInt(-acc, True) if neg else ParsedInt(acc, True)


def parse_memory_value(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int,
                       pct_of_bytes: Int64) -> ParsedInt:
    """gh #261: a memory size as Redis spells it — `<digits>[b|k|kb|m|mb|g|gb]`,
    case-insensitive, where `k` is 1000 and `kb` is 1024 (Redis's memtoull).

    `pct_of_bytes > 0` additionally accepts `<1..100>%` of that many bytes (the
    caller passes physical RAM). That form is a
    Pion extension for the command line only: Redis's CONFIG SET rejects it,
    and so does Pion's, so a client script cannot come to depend on it.

    Cold path (startup, CONFIG SET): no allocation constraints, but it follows
    parse_int64_strict's rule — reject anything malformed, never skip it."""
    if plen == 0:
        return ParsedInt(0, False)
    var ndig = 0
    while ndig < plen and p[unsafe_offset=ndig] >= 48 and p[unsafe_offset=ndig] <= 57:
        ndig += 1
    if ndig == 0:
        return ParsedInt(0, False)          # no digits, or a sign: not a size
    var num = parse_int64_strict(p, ndig)
    if not num.ok:
        return ParsedInt(0, False)
    var ulen = plen - ndig
    var u0: Int = 0
    var u1: Int = 0
    if ulen >= 1:
        u0 = Int(p[unsafe_offset=ndig])
        if u0 >= 65 and u0 <= 90:
            u0 += 32
    if ulen >= 2:
        u1 = Int(p[unsafe_offset=ndig + 1])
        if u1 >= 65 and u1 <= 90:
            u1 += 32
    var mult: Int64 = 0
    if ulen == 0:
        mult = 1
    elif ulen == 1 and u0 == 37:            # '%'
        if pct_of_bytes <= 0 or num.value < 1 or num.value > 100:
            return ParsedInt(0, False)
        return ParsedInt((pct_of_bytes // 100) * num.value, True)
    elif ulen == 1 and u0 == 98:            # b
        mult = 1
    elif ulen == 1 and u0 == 107:           # k
        mult = 1000
    elif ulen == 1 and u0 == 109:           # m
        mult = 1000000
    elif ulen == 1 and u0 == 103:           # g
        mult = 1000000000
    elif ulen == 2 and u1 == 98 and u0 == 107:
        mult = 1024
    elif ulen == 2 and u1 == 98 and u0 == 109:
        mult = 1048576
    elif ulen == 2 and u1 == 98 and u0 == 103:
        mult = 1073741824
    else:
        return ParsedInt(0, False)
    if num.value > 9223372036854775807 // mult:
        return ParsedInt(0, False)
    return ParsedInt(num.value * mult, True)


@always_inline
def is_valid_float_arg(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int) -> Bool:
    """True only for a well-formed decimal float: [+-]?digits[.digits][eE[+-]digits].

    The float parsers (`parse_filter_float` / `parse_float64`) skip characters
    they don't recognise, exactly like the integer loops did, so on 0.915
    `INCRBYFLOAT k abc` and `INCRBYFLOAT k ''` both added 0.0 and replied with
    the unchanged value as if it had worked.

    This is a separate VALIDATOR rather than a change to those parsers on
    purpose: `parse_filter_float`'s behaviour is what vector FILTER comparisons
    are built on, and re-pointing it here would drag an unrelated subsystem
    into a string-command fix."""
    if plen == 0:
        return False
    var i = 0
    if p[unsafe_offset=0] == 45 or p[unsafe_offset=0] == 43:   # '-' / '+'
        i = 1
    var mantissa_digits = 0
    var seen_dot = False
    # No `break` in this loop: with `@always_inline`, Mojo 1.0.0 (ed45d567)
    # miscompiled the break-out-of-the-mantissa-loop shape followed by the
    # exponent loop — `is_valid_float_arg("10")` returned False at -O0 AND -O3
    # in some inlining contexts (FT.SEARCH filter bounds, a standalone call)
    # while the INCRBYFLOAT sites happened to be right. The loop condition
    # carries the stop instead. tests/test_float_arg_validator.mojo pins it.
    var in_mantissa = True
    while i < plen and in_mantissa:
        var c = Int(p[unsafe_offset=i])
        if c >= 48 and c <= 57:
            mantissa_digits += 1
            i += 1
        elif c == 46:                     # '.'
            if seen_dot:
                return False              # two decimal points
            seen_dot = True
            i += 1
        elif c == 101 or c == 69:         # 'e' / 'E' — the exponent starts at i
            in_mantissa = False
        else:
            return False
    if mantissa_digits == 0:
        return False                      # ".", "-", "e5" are not numbers
    if i >= plen:
        return True                       # no exponent
    i += 1                                # skip the e/E
    if i < plen and (p[unsafe_offset=i] == 45 or p[unsafe_offset=i] == 43):
        i += 1
    var exp_digits = 0
    while i < plen:
        var c2 = Int(p[unsafe_offset=i])
        if c2 < 48 or c2 > 57:
            return False
        exp_digits += 1
        i += 1
    return exp_digits > 0                 # a bare trailing "e" is not a number


@always_inline
def parse_float64(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int) -> Float64:
    """Float64-returning variant of parse_filter_float (gh #180 — HINCRBYFLOAT
    accumulated in Float32 and lost precision within a few increments). Kept
    separate so parse_filter_float's Float32 rounding, which vector FILTER
    comparisons depend on, is untouched."""
    var i = 0
    var sign: Float64 = 1.0
    if i < plen and p[unsafe_offset=i] == 45: sign = -1.0; i += 1
    elif i < plen and p[unsafe_offset=i] == 43: i += 1
    var integer: Float64 = 0.0
    while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
        integer = integer * 10.0 + Float64(p[unsafe_offset=i] - 48); i += 1
    var frac: Float64 = 0.0; var frac_mul: Float64 = 0.1
    if i < plen and p[unsafe_offset=i] == 46:
        i += 1
        while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
            frac = frac + frac_mul * Float64(p[unsafe_offset=i] - 48); frac_mul *= 0.1; i += 1
    var exp_val: Int = 0; var exp_sign: Int = 1
    if i < plen and (p[unsafe_offset=i] == 101 or p[unsafe_offset=i] == 69):
        i += 1
        if i < plen and p[unsafe_offset=i] == 45: exp_sign = -1; i += 1
        elif i < plen and p[unsafe_offset=i] == 43: i += 1
        while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
            exp_val = exp_val * 10 + Int(p[unsafe_offset=i] - 48); i += 1
        exp_val *= exp_sign
    var result = sign * (integer + frac)
    if exp_val > 0:
        var scale: Float64 = 1.0
        for _ in range(exp_val): scale *= 10.0
        result *= scale
    elif exp_val < 0:
        var scale: Float64 = 1.0
        for _ in range(-exp_val): scale /= 10.0
        result *= scale
    return result


# gh #373: "THRESHOLD was not given". The old sentinel was 0.0, so an explicit
# `THRESHOLD 0` ("match anything") silently meant "use the default" — and the
# parsers returned 0.0 for an unparseable value too. Any real cosine threshold
# is in [-1, 1], so a value this far below cannot be one a caller sent.
comptime THRESHOLD_UNSET = Float32(-1.0e30)


@always_inline
def resolve_threshold(override: Float32, default: Float32) -> Float32:
    """The caller's THRESHOLD when one was given (including 0), else `default`."""
    return override if override > Float32(-1.0e29) else default


@always_inline
def parse_filter_float(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int) -> Float32:
    """gh #87.3: parse a single ASCII float from buffer[0..plen-1].
    Canonical copy — previously duplicated as `_parse_filter_float` /
    `_parse_float_buf` / `_parse_hash_float` across vector.mojo / ai.mojo /
    string_kv.mojo / hash.mojo. Same shape as embedding_client._parse_float32.

    Handles optional `[+-]`, integer part, optional `.frac`, optional
    `[eE][+-]?digits` exponent. Returns 0.0 on empty or all-non-digit input
    (consistent with the original behaviour — no exception)."""
    var i = 0
    var sign: Float64 = 1.0
    if i < plen and p[unsafe_offset=i] == 45: sign = -1.0; i += 1
    elif i < plen and p[unsafe_offset=i] == 43: i += 1
    var integer: Float64 = 0.0
    while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
        integer = integer * 10.0 + Float64(p[unsafe_offset=i] - 48); i += 1
    var frac: Float64 = 0.0; var frac_mul: Float64 = 0.1
    if i < plen and p[unsafe_offset=i] == 46:
        i += 1
        while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
            frac = frac + frac_mul * Float64(p[unsafe_offset=i] - 48); frac_mul *= 0.1; i += 1
    var exp_val: Int = 0; var exp_sign: Int = 1
    if i < plen and (p[unsafe_offset=i] == 101 or p[unsafe_offset=i] == 69):
        i += 1
        if i < plen and p[unsafe_offset=i] == 45: exp_sign = -1; i += 1
        elif i < plen and p[unsafe_offset=i] == 43: i += 1
        while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
            exp_val = exp_val * 10 + Int(p[unsafe_offset=i] - 48); i += 1
        exp_val *= exp_sign
    var result = Float32(sign * (integer + frac))
    if exp_val > 0:
        var scale: Float64 = 1.0
        for _ in range(exp_val): scale *= 10.0
        result = Float32(Float64(result) * scale)
    elif exp_val < 0:
        var scale: Float64 = 1.0
        for _ in range(-exp_val): scale /= 10.0
        result = Float32(Float64(result) * scale)
    return result
