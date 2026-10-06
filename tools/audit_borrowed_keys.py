#!/usr/bin/env python3
"""Borrowed lookup keys (gh #394): rewrite the sites that can use one, and
audit that every borrowed key is used safely.

WHY
`GenericValue.from_ptr` / `from_string` copy anything over 23 bytes to the
heap, and a key built only to look something up was almost never freed:
every command on a key longer than 23 bytes leaked ~48-64 B, plain GET
included. `GenericValue.borrow(ptr, len)` builds the same value without the
copy (SSO up to 23 bytes, a BORROW_TAG'd STRING pointing at `ptr` above), and
every SlabHashMap store copies a borrowed value before keeping it.

That makes a borrowed value safe to hand to the hash map and to read — and
unsafe everywhere else: stored in a list, a skip list, a stream, a struct
field or a List[GenericValue], it is a pointer into a recv buffer the next
read overwrites. A per-site hand pass is how gh #225 learned it misses sites,
so both halves are mechanical:

    python3 tools/audit_borrowed_keys.py            # audit: must report 0
    python3 tools/audit_borrowed_keys.py --plan -v  # what --rewrite would do, and why not
    python3 tools/audit_borrowed_keys.py --rewrite  # convert every provable site

A site is converted only when BOTH hold:
  1. its bytes outlive the command: a RESP token (`tokens[E].ptr`, or a
     `var s = tokens[E].value()` String — rewritten to the token itself, since
     Mojo may destroy the String right after its last use), the fast path's
     recv buffer (`buffer + ...`, or a local assigned from it), a read-only
     String or pointer PARAMETER (the caller keeps it alive for the call), or
     a Lua `redis.call` argument (`_lua_arg_ptr`: on the coroutine's stack
     until the handler resumes it, which is its return);
  2. EVERY use of the value is a hash-map call (get/set/remove/...), a
     read-only accessor, a comparison, or one of the audited helpers below.
Anything else is left as an owned copy. The audit re-applies the same two
rules to every `GenericValue.borrow(` in src/ and reports each violation.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# value.mojo defines borrow(); hash_map.mojo borrows its own String arguments
# for the duration of the call, which is the one place a String is safe.
EXEMPT = {SRC / "common" / "value.mojo", SRC / "common" / "hash_map.mojo"}

# Hash-map methods: any argument position is safe — keys and values are both
# copied by owned() on insert, and lookups never keep what they are given.
MAP_METHODS = {
    "get", "get_value_ptr", "get_with_hash", "remove_generic",
    "remove_generic_taking", "remove_generic_with_hash",
    "remove_generic_with_hash_taking", "set", "set_with_hash", "set_str_reuse",
    # gh #392 per-field TTL methods on SlabHashMap / StripedHashMap: map calls inside
    "field_deadline", "set_field_deadline", "clear_field_deadline", "expire_field",
    "note_field_ttl",
    # #45: StripedHashMap's TTL reads (a ttl_map get; keep nothing)
    "deadline", "is_expired",
}
# Read-only methods on the value itself.
READ_METHODS = {
    "__hash__", "string_len", "as_string", "as_string_safe", "copy_to",
    "is_none", "type", "is_string", "free_str_payload", "clone", "is_borrowed",
    "owned", "lex_lt",
}
# Helpers audited by hand: each passes the key only to hash-map calls or reads
# its bytes, and none keeps it. Name -> the argument positions that may be a
# borrowed key. Adding one here is a claim about its body — read it first.
SAFE_FUNCS = {
    "remove_and_free": {1},          # container_free.mojo: get + remove_generic
    "get_stream": {1},               # stream.mojo: get
    "stream_key_is_wrongtype": {1},  # stream.mojo: get
    "get_or_create_stream": {1},     # stream.mojo: get + set
    "_delete_expired_now": {1},      # ttl.mojo: remove_generic on two maps
    "gv_bytes": {0},                 # wal.mojo: reads the bytes
    "_set_blob_value": {0},          # fast_path.mojo: set
    "remove": {0},                   # SlabSkipList.remove: dict get/remove + compare
    "member_score": {0},             # SlabSkipList.member_score: dict get
    "_lookup": {1},                  # vset.mojo: keyspace get + type check
    "hash_get_live": {1},            # container_free: keyspace get, purge, remove_and_free
    "_lex_in": {2, 4},               # sorted_set.mojo: lex_lt comparisons only
    "index_field_ttls": {1},         # container_free: note_field_ttl (a map set)
    # wal.mojo replay lookups: keyspace get, and set on create
    "_replay_hash": {1}, "_replay_list": {1}, "_replay_set": {1}, "_replay_zset": {1},
    "_replay_stream": {1}, "_replay_vset": {1},
    "hll_add": {1},                  # hll.mojo: hashes the element, keeps nothing
    "key_slot_from_hash": set(),     # only ever sees X.__hash__()
    "UInt64": {0},                   # UInt64(X.__hash__()) — never X itself
}

TOKEN_VALUE = re.compile(r"^tokens\[(unsafe_offset=)?(?P<idx>[^\]]+)\]\.value\(\)$")
TOKEN_PTR = re.compile(r"^tokens\[(unsafe_offset=)?(?P<idx>[^\]]+)\]\.ptr$")
TOKEN_LEN = re.compile(r"^tokens\[(unsafe_offset=)?(?P<idx>[^\]]+)\]\.length$")
CTOR = re.compile(r"GenericValue\.(from_string|from_ptr|borrow_buf|borrow)\(")
IDENT = re.compile(r"[A-Za-z_]\w*")


def mask(text):
    """Blank comments and string literals (same length) so paren matching and
    identifier searches only see code."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "#":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c in "\"'":
            q3 = text[i:i + 3]
            if q3 in ('"""', "'''"):
                j = text.find(q3, i + 3)
                j = n if j < 0 else j + 3
            else:
                j = i + 1
                while j < n and text[j] != c and text[j] != "\n":
                    j += 2 if text[j] == "\\" else 1
                j += 1
            for k in range(i, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def split_args(s):
    """Split a call's argument text on top-level commas."""
    depth, cur, out = 0, [], []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def close_paren(m, open_at):
    depth = 0
    for k in range(open_at, len(m)):
        if m[k] == "(":
            depth += 1
        elif m[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    return -1


def enclosing_call(m, pos):
    """(callee name, argument index) of the innermost call whose argument list
    contains offset `pos`, or None."""
    depth = 0
    k = pos - 1
    while k >= 0:
        ch = m[k]
        if ch in ")]}":
            depth += 1
        elif ch in "([{":
            if depth == 0:
                if ch != "(":
                    return None       # inside an index or literal, not a call
                j = k - 1
                while j >= 0 and m[j] == " ":
                    j -= 1
                e = j + 1
                while j >= 0 and (m[j].isalnum() or m[j] == "_"):
                    j -= 1
                name = m[j + 1:e]
                if not name:
                    return None
                end = close_paren(m, k)
                args_text = m[k + 1:end]
                argi = len(split_args(m[k + 1:pos] + "\x00")) - 1
                del args_text
                return name, argi
            depth -= 1
        k -= 1
    return None


def line_of(text, pos):
    return text.count("\n", 0, pos)


def indent(line):
    return len(line) - len(line.lstrip())


def scope_lines(lines, def_line):
    """Line numbers of the block a `var` on def_line is visible in, skipping
    nested blocks that shadow it with their own `var` of the same name."""
    ind = indent(lines[def_line])
    out = []
    k = def_line + 1
    while k < len(lines):
        s = lines[k]
        if s.strip() and not s.strip().startswith("#") and indent(s) < ind:
            break
        out.append(k)
        k += 1
    return out


class Site:
    def __init__(self, path, line, kind, text, reason=None):
        self.path, self.line, self.kind, self.text, self.reason = path, line, kind, text, reason


def token_source(lines, mlines, site_line, s_expr):
    """If the String `s_expr` is a RESP token's value — directly, or through a
    `var S = tokens[E].value()` whose inputs are unchanged since — return
    (tokens expression prefix, idx) for borrow(); else None."""
    s_expr = s_expr.strip()
    m = TOKEN_VALUE.match(s_expr)
    if m:
        return s_expr[: -len(".value()")]
    if not IDENT.fullmatch(s_expr):
        return None
    name = s_expr
    pat = re.compile(r"^\s*var\s+%s\s*(?::\s*String\s*)?=\s*(tokens\[[^\]]+\])\.value\(\)\s*$" % re.escape(name))
    for k in range(site_line, -1, -1):
        mk = None
        for stmt in mlines[k].split(";"):
            mk = mk or pat.match(" " * indent(mlines[k]) + stmt.strip())
        if mk:
            if indent(lines[k]) > indent(lines[site_line]):
                return None              # defined in a block the site is not in
            tok = mk.group(1)
            idx_vars = set(IDENT.findall(tok)) - {"tokens", "unsafe_offset"} | {name}
            for j in range(k + 1, site_line):
                for v in idx_vars:
                    if re.search(r"(?<![\w.])%s\s*(\+|-|\*)?=(?!=)" % re.escape(v), mlines[j]):
                        return None      # index or the String rebound in between
            return tok
        if re.match(r"^\s*(?:@\w+\s*)*def\s", lines[k]):
            if readonly_param(lines, k, name, "String") and not rebound(mlines, k, site_line, name):
                return f"{name}.unsafe_ptr(), {name}.byte_length()"
            return None
    return None


def tok_src_is_token(expr):
    return bool(TOKEN_VALUE.match(expr.strip()))


def signature(lines, def_line):
    sig = []
    for k in range(def_line, min(def_line + 30, len(lines))):
        sig.append(lines[k])
        if re.search(r"\)\s*(->[^:]*)?(raises\s*)?(->[^:]*)?:\s*(#.*)?$", lines[k]):
            break
    return " ".join(sig)


def readonly_param(lines, def_line, name, typ):
    """`name: typ` is a parameter the CALLER owns (no var/mut/out/deinit), so
    it is alive for the whole call. An owned (`var`) parameter is not: Mojo
    may destroy it right after its last use."""
    sig = signature(lines, def_line)
    m = re.search(r"(?:^|[(,])\s*((?:var|mut|out|deinit|owned|inout)\s+)?%s\s*:\s*%s\b" % (re.escape(name), typ), sig)
    return bool(m) and not m.group(1)


def rebound(mlines, a, b, name):
    return any(re.search(r"(?<![\w.])%s\s*(\+|-|\*)?=(?!=)" % re.escape(name), mlines[j]) for j in range(a + 1, b))


def recv_derived(mlines, site_line, p_expr):
    """True when a pointer expression points into the fast path's recv buffer
    or at a RESP token's bytes."""
    p = p_expr.strip()
    if TOKEN_PTR.match(p):
        return True
    mo = re.fullmatch(r"(\w+)\.unsafe_offset\(.*\)", p)
    if mo:                            # an offset from an alive pointer is alive
        return recv_derived(mlines, site_line, mo.group(1))
    mt = re.fullmatch(r"(\w+)\.ptr", p)
    if mt:                            # a local copy of a RESP token: its bytes
        tv = mt.group(1)
        for k in range(site_line, -1, -1):
            if re.match(r"^\s*var\s+%s\s*=\s*tokens\[[^\]]+\]\s*$" % re.escape(tv), mlines[k]):
                return True
            if re.match(r"^\s*(?:@\w+\s*)*def\s", mlines[k]):
                return False
        return False
    root = IDENT.match(p)
    if not root:
        return False
    r = root.group(0)
    if r == "buffer":
        return True
    mp = re.fullmatch(r"(\w+)\.unsafe_ptr\(\)", p)
    if mp:   # a read-only String parameter's bytes (the audit's view of case 1)
        for k in range(site_line, -1, -1):
            if re.match(r"^\s*(?:@\w+\s*)*def\s", mlines[k]):
                return readonly_param(mlines, k, mp.group(1), "String") and not rebound(mlines, k, site_line, mp.group(1))
        return False
    # a local assigned only from the recv buffer / a token pointer
    assigned = []
    for k in range(site_line, -1, -1):
        mk = re.match(r"^\s*(?:var\s+)?%s\s*(?::\s*[\w\[\], ]+)?=\s*(.+?)\s*$" % re.escape(r), mlines[k])
        if mk and "==" not in mlines[k].split("=", 1)[0]:
            assigned.append(mk.group(1))
            if mlines[k].lstrip().startswith("var "):
                break
        if re.match(r"^\s*(?:@\w+\s*)*def\s", mlines[k]):
            # a pointer PARAMETER: the caller keeps it alive for the call
            if not assigned and re.fullmatch(r"\w+", p) and readonly_param(mlines, k, r, "(?:Unsafe)?Pointer"):
                return True
            break
    if not assigned:
        return False
    return all(a.startswith("buffer") or TOKEN_PTR.match(a) or a.startswith("_lua_arg_ptr(")
               for a in assigned)


def use_ok(m, text_pos, name):
    """Is this occurrence of `name` at offset text_pos a safe use?"""
    after = m[text_pos + len(name):]
    before = m[:text_pos]
    ma = re.match(r"\s*\.\s*(\w+)", after)
    if ma:
        return ma.group(1) in READ_METHODS, f".{ma.group(1)}"
    if re.match(r"\s*(==|!=)", after) or re.search(r"(==|!=)\s*$", before):
        return True, "=="
    call = enclosing_call(m, text_pos)
    if call is None:
        return False, "not a call argument"
    fn, argi = call
    if fn in MAP_METHODS:
        return True, fn
    if fn in SAFE_FUNCS and argi in SAFE_FUNCS[fn]:
        return True, fn
    return False, f"{fn}(arg {argi})"


def analyze(path, rewrite=False):
    text = path.read_text()
    m = mask(text)
    lines = text.split("\n")
    mlines = m.split("\n")
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln) + 1)
    converted, skipped, violations = [], [], []
    edits = []   # (start, end, replacement)

    for cm in CTOR.finditer(m):
        kind = "borrow" if cm.group(1) == "borrow_buf" else cm.group(1)
        open_at = cm.end() - 1
        close = close_paren(m, open_at)
        args = split_args(text[open_at + 1:close])
        ln = line_of(text, cm.start())
        line = lines[ln]
        # named: `var X = GenericValue.ctor(...)` with nothing after it
        named = re.match(r"^\s*var\s+(\w+)\s*(?::\s*GenericValue\s*)?=\s*$", m[offs[ln]:cm.start()])
        if named and m[close + 1:offs[ln + 1] if ln + 1 < len(offs) else len(m)].strip():
            named = None       # `var x = GenericValue.ctor(...).something`
        # --- inline argument of a hash-map call ---
        # The value cannot outlive the call (the map copies what it keeps), so
        # any pointer that is valid at the call will do. A String is different:
        # borrowing its bytes inline lets Mojo destroy it before the call runs,
        # so from_string(S) becomes the map's own String overload instead,
        # which holds S for the whole call.
        if kind in ("from_ptr", "from_string") and not named:
            call = enclosing_call(m, cm.start())
            direct = call and call[0] in MAP_METHODS and \
                re.search(r"[(,]\s*$", m[:cm.start()]) is not None
            if direct and kind == "from_ptr" and len(args) == 2 and \
                    not re.match(r"\s*(\.|\[)", m[close + 1:]):
                converted.append(Site(path, ln, kind, line.strip()))
                edits.append((cm.start(), close + 1, f"GenericValue.borrow({args[0]}, {args[1]})"))
                continue
            if direct and kind == "from_string" and len(args) == 1 and call[1] == 0 and \
                    call[0] in ("get", "set", "remove_generic"):
                # rewrite `.get(GenericValue.from_string(S)` -> `.get(S`
                head = m[:cm.start()]
                mh = re.search(r"\.(get|set|remove_generic)\(\s*$", head)
                if mh and not tok_src_is_token(args[0]):
                    meth = {"get": "get", "set": "set", "remove_generic": "remove"}[mh.group(1)]
                    converted.append(Site(path, ln, kind, line.strip()))
                    edits.append((mh.start(), close + 1, f".{meth}({args[0]}"))
                    continue

        # --- source check ---
        new_ctor = None
        if kind == "from_string":
            if len(args) != 1:
                continue
            tok = token_source(lines, mlines, ln, args[0])
            if tok is None:
                if rewrite:
                    skipped.append(Site(path, ln, kind, line.strip(), "String not a token's value"))
                continue
            new_ctor = (f"GenericValue.borrow({tok})" if "unsafe_ptr()" in tok
                        else f"GenericValue.borrow({tok}.ptr, {tok}.length)")
        elif kind == "from_ptr":
            if len(args) != 2 or not recv_derived(mlines, ln, args[0]):
                if rewrite:
                    skipped.append(Site(path, ln, kind, line.strip(), "pointer not the recv buffer or a token"))
                continue
            new_ctor = f"GenericValue.borrow({args[0]}, {args[1]})"
        else:  # borrow
            if path in EXEMPT:
                continue
            call = enclosing_call(m, cm.start())
            if not named and call and (call[0] in MAP_METHODS or
                                       (call[0] in SAFE_FUNCS and call[1] in SAFE_FUNCS[call[0]])) and \
                    re.search(r"[(,]\s*$", m[:cm.start()]) and not re.match(r"\s*(\.|\[)", m[close + 1:]):
                continue     # a direct argument of a call that keeps nothing: cannot outlive it
            if len(args) != 2 or not recv_derived(mlines, ln, args[0]):
                violations.append(Site(path, ln, kind, line.strip(), f"source not provably alive: {args[0] if args else '?'}"))
                continue
        # --- use check ---
        bad = None
        if named:
            name = named.group(1)
            for k in scope_lines(lines, ln):
                ml = mlines[k]
                if re.match(r"^\s*var\s+%s\b" % re.escape(name), ml):
                    bad = f"line {k + 1}: shadowed"      # conservative
                    break
                for om in re.finditer(r"(?<![\w.])%s\b" % re.escape(name), ml):
                    if re.match(r"\s*(\+|-)?=(?!=)", ml[om.end():]):
                        bad = f"line {k + 1}: rebound"
                        break
                    if re.match(r"\s*\^", ml[om.end():]):
                        bad = f"line {k + 1}: moved"
                        break
                    ok, why = use_ok(m, offs[k] + om.start(), name)
                    if not ok:
                        bad = f"line {k + 1}: {why}"
                        break
                if bad:
                    break
        else:
            call = enclosing_call(m, cm.start())
            if call is None or not (call[0] in MAP_METHODS or
                                    (call[0] in SAFE_FUNCS and call[1] in SAFE_FUNCS[call[0]])):
                bad = f"inline in {call[0] + '()' if call else 'a non-call'}"
        if kind == "borrow":
            if bad:
                violations.append(Site(path, ln, kind, line.strip(), bad))
            continue
        if bad:
            if rewrite:
                skipped.append(Site(path, ln, kind, line.strip(), bad))
            continue
        converted.append(Site(path, ln, kind, line.strip()))
        edits.append((cm.start(), close + 1, new_ctor))

    if rewrite == "write" and edits:
        out = text
        for s, e, rep in sorted(edits, reverse=True):
            out = out[:s] + rep + out[e:]
        path.write_text(out)
    return converted, skipped, violations


def main():
    global SRC, EXEMPT
    rewrite = "write" if "--rewrite" in sys.argv else ("plan" if "--plan" in sys.argv else False)
    verbose = "-v" in sys.argv
    if "--src" in sys.argv:          # audit another tree (the test's canary)
        SRC = Path(sys.argv[sys.argv.index("--src") + 1]).resolve()
        EXEMPT = {SRC / "common" / "value.mojo", SRC / "common" / "hash_map.mojo"}
    files = sorted(p for p in SRC.rglob("*.mojo"))
    tc, ts, tv = [], [], []
    for p in files:
        c, s, v = analyze(p, rewrite)
        tc += c; ts += s; tv += v
    if rewrite:
        print(f"converted {len(tc)} site(s) to GenericValue.borrow; left {len(ts)} as owned copies")
        if verbose:
            for x in tc:
                print(f"  CONV  {x.path}:{x.line + 1}  {x.text[:100]}")
            for x in ts:
                print(f"  kept  {x.path}:{x.line + 1}  [{x.reason}]  {x.text[:90]}")
        # audit what the rewrite produced
        tv = []
        for p in files:
            tv += analyze(p, False)[2]
    for x in tv:
        print(f"UNSAFE BORROW {x.path}:{x.line + 1}: {x.reason}\n    {x.text}")
    print(f"{len(tv)} unsafe borrowed key(s)")
    return 1 if tv else 0


if __name__ == "__main__":
    sys.exit(main())
