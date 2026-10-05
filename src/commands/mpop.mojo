"""Arguments of LMPOP and ZMPOP (and their blocking forms), parsed in the order
Redis's lmpopGenericCommand / zmpopGenericCommand check them: numkeys, the
direction, then at most one COUNT, with anything else a syntax error. The keys
are looked at only after all of that, so a bad argument is refused before a
key's type is."""
from std.memory.unsafe_pointer import UnsafePointer
from src.network.resp3 import RESP3Token
from src.common.utils import arg_eq, parse_int64_strict


@fieldwise_init
struct MPopArgs(Copyable, Movable):
    var error: String        # empty when the arguments parsed
    var numkeys: Int
    var first: Bool          # LEFT for lists, MIN for sorted sets
    var count: Int


def parse_mpop(tokens: UnsafePointer[RESP3Token, MutUntrackedOrigin], numkeys_idx: Int, end: Int,
               zset: Bool) -> MPopArgs:
    """`tokens[numkeys_idx]` is numkeys; the keys follow it, then LEFT|RIGHT
    (MIN|MAX when `zset`), then an optional COUNT n."""
    var nk = parse_int64_strict(tokens[numkeys_idx].ptr, tokens[numkeys_idx].length)
    if not nk.ok or nk.value < 1:
        return MPopArgs(String("ERR numkeys should be greater than 0"), 0, False, 0)
    if nk.value >= Int64(end - numkeys_idx - 1):     # no token left for the direction
        return MPopArgs(String("ERR syntax error"), 0, False, 0)
    var numkeys = Int(nk.value)
    var w = tokens[numkeys_idx + numkeys + 1]
    var first = True
    if zset:
        if arg_eq(w.ptr, w.length, "max"):
            first = False
        elif not arg_eq(w.ptr, w.length, "min"):
            return MPopArgs(String("ERR syntax error"), 0, False, 0)
    else:
        if arg_eq(w.ptr, w.length, "right"):
            first = False
        elif not arg_eq(w.ptr, w.length, "left"):
            return MPopArgs(String("ERR syntax error"), 0, False, 0)
    var count = -1
    var j = numkeys_idx + numkeys + 2
    while j < end:
        if count == -1 and arg_eq(tokens[j].ptr, tokens[j].length, "count") and j + 1 < end:
            j += 1
            var c = parse_int64_strict(tokens[j].ptr, tokens[j].length)
            if not c.ok or c.value < 1:
                return MPopArgs(String("ERR count should be greater than 0"), 0, False, 0)
            count = Int(c.value)
        else:
            return MPopArgs(String("ERR syntax error"), 0, False, 0)
        j += 1
    if count == -1:
        count = 1
    return MPopArgs(String(""), numkeys, first, count)
