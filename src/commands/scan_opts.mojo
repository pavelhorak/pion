"""Options of SCAN, HSCAN, SSCAN and ZSCAN, read as Redis's scanGenericCommand
reads them: MATCH, COUNT (an integer of at least 1), TYPE (SCAN only),
NOVALUES (HSCAN only), in any order, and anything else a syntax error. Each
scan used to match these by length and first letter and stop silently at the
first word it did not know, and SCAN parsed TYPE only to ignore it."""
from std.memory.unsafe_pointer import Pointer
from src.common.ptr import null_ptr
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.common.utils import arg_eq, parse_int64_strict


@fieldwise_init
struct ScanOpts(Copyable, Movable, ImplicitlyCopyable):
    var ok: Bool
    var has_match: Bool
    var pat_p: Pointer[UInt8, MutUntrackedOrigin]
    var pat_l: Int
    var has_type: Bool
    var type_p: Pointer[UInt8, MutUntrackedOrigin]
    var type_l: Int
    var novalues: Bool
    var count: Int            # #50: COUNT, 10 when absent, as in Redis


def scan_no_opts() -> ScanOpts:
    """No options: what a scan of a missing key answers with, unparsed."""
    return ScanOpts(True, False, null_ptr[UInt8, MutUntrackedOrigin](), 0,
                    False, null_ptr[UInt8, MutUntrackedOrigin](), 0, False, 10)


def parse_scan_opts(tokens: Pointer[RESP3Token, MutUntrackedOrigin], start: Int, end: Int,
                    mut writer: ResponseWriter, keyspace_scan: Bool, hash_scan: Bool) -> ScanOpts:
    """The options from `start` to `end`. On error the reply is written and
    `ok` is False."""
    var o = ScanOpts(False, False, null_ptr[UInt8, MutUntrackedOrigin](), 0,
                     False, null_ptr[UInt8, MutUntrackedOrigin](), 0, False, 10)
    var j = start
    while j < end:
        var t = tokens[j]
        var left = end - j
        if arg_eq(t.ptr, t.length, "count") and left >= 2:
            var c = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not c.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return o
            if c.value < 1:
                writer.append_error_response("ERR syntax error")
                return o
            o.count = Int(c.value)
            j += 2
        elif arg_eq(t.ptr, t.length, "match") and left >= 2:
            o.has_match = True
            o.pat_p = tokens[j + 1].ptr
            o.pat_l = tokens[j + 1].length
            j += 2
        elif keyspace_scan and arg_eq(t.ptr, t.length, "type") and left >= 2:
            o.has_type = True
            o.type_p = tokens[j + 1].ptr
            o.type_l = tokens[j + 1].length
            j += 2
        elif arg_eq(t.ptr, t.length, "novalues"):
            if not hash_scan:
                writer.append_error_response("ERR NOVALUES option can only be used in HSCAN")
                return o
            o.novalues = True
            j += 1
        else:
            writer.append_error_response("ERR syntax error")
            return o
    o.ok = True
    return o
