#!/usr/bin/env python3
"""Generate src/commands/command_table.mojo from the dispatch chains (gh #220).

WHY THIS IS GENERATED, NOT HAND-WRITTEN
---------------------------------------
The table backs queue-time validation inside MULTI. A hand-maintained list
drifts, and the drift is asymmetric and nasty: a command missing from the table
still works normally but is rejected inside a transaction. So the list is
derived from the dispatch arms themselves and re-derivable at any time —
`tests/test_command_table_drift.py` regenerates and diffs.

HOW A NAME IS ESTABLISHED
-------------------------
Two independent signals, and every name is verified against the arm it came
from:

  1. `cmd_matches_N(tp, 109,117,108,116,105)` decodes directly to "multi".
  2. Prefix-matched arms (the substrate families) encode no decodable name, so
     the candidate comes from the arm body's `handle_<name>(...)` call — then
     every `.`/`_` placement of that handler name is filtered by the arm's OWN
     constraints: exact length (`tl == 15`), case-folded bytes
     (`(tp[0]|0x20) == 110`), and literal bytes (`tp[6] == 46`). A candidate
     that survives is the only string that arm can match, not a guess.

Ambiguous survivors are resolved against the documented wire name in the
repository docs. Anything still ambiguous or unverifiable is reported and the
generator FAILS rather than emitting a table that is quietly wrong.

That verification step is not just bookkeeping: it is what found gh #221, where
ZREVRANGEBYSCORE and ZREVRANGEBYLEX had each other's lengths and so ran each
other's handlers.

Usage:  python3 tools/gen_command_table.py [--check]
        --check exits non-zero if the committed table differs (drift gate).
"""

import argparse
import itertools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "network"
OUT = ROOT / "src" / "commands" / "command_table.mojo"

CM = re.compile(r"cmd_matches_(\d+)\s*\(\s*tp\s*,\s*([0-9,\s]+?)\)")
# gh #225: an arm rewritten to `cmd_eq(tp, tl, "name")` states its name
# outright, so there is nothing to infer. Prefer it over every heuristic below.
CEQ = re.compile(r'cmd_eq\s*\(\s*tp\s*,\s*tl\s*,\s*"([^"]+)"\s*\)')
HANDLER = re.compile(r"\bhandle_([a-z0-9_]+)\s*\(")
TL = re.compile(r"\btl\s*==\s*(\d+)")
FOLDED = re.compile(r"\(\s*tp\[\s*(\d+)\s*\]\s*\|\s*0x20\s*\)\s*==\s*(\d+)")
EXACT = re.compile(r"(?<![|\w])tp\[\s*(\d+)\s*\]\s*==\s*(\d+)")
FP = re.compile(r"b0_lower\s*==\s*(\d+)\s*and\s*cmd_len\s*==\s*(\d+)")

_DOCS = None

# Arms that dispatch inline (dispatcher.execute_*) call no handle_*, so there
# is no name to read off the body. This is a HINT POOL, not a table: a hint is
# accepted only when the arm's own length/byte constraints admit exactly one of
# them, and an arm no hint fits fails the build. Adding a wrong name here
# cannot corrupt the output — it simply will not match anything.
INLINE_HINTS = {
    "replconf", "psync", "rpoplpush", "lmpop", "zmpop", "smismember",
    "sintercard", "lcs", "getdel", "getex", "sintercard", "object",
    "zpopmin", "zpopmax", "blmpop", "bzmpop", "lpos", "zrandmember",
    "hrandfield", "srandmember", "waitaof", "failover", "psetex",
}


def docs():
    global _DOCS
    if _DOCS is None:
        blob = []
        for p in [ROOT / "CLAUDE.md"] + sorted((ROOT / "doc").glob("*.md")):
            try:
                blob.append(p.read_text())
            except OSError:
                pass
        _DOCS = "\n".join(blob)
    return _DOCS


def decode_literal(nums):
    try:
        s = "".join(chr(int(n)) for n in nums)
        return s if s.isascii() and s.isalpha() else None
    except (ValueError, TypeError):
        return None


def candidates(handler):
    parts = handler.split("_")
    if len(parts) == 1:
        return {parts[0]}
    out = set()
    for seps in itertools.product("._", repeat=len(parts) - 1):
        s = parts[0]
        for sep, p in zip(seps, parts[1:]):
            s += sep + p
        out.add(s)
    return out


def satisfies(name, tl, folded, exact):
    if tl is not None and len(name) != tl:
        return False
    for idx, val in folded.items():
        if idx >= len(name) or (ord(name[idx]) | 0x20) != val:
            return False
    for idx, val in exact.items():
        if idx >= len(name) or ord(name[idx]) != val:
            return False
    return True


def collect():
    names, problems = {}, []
    lines = (SRC / "slow_path.mojo").read_text().splitlines()

    for idx, line in enumerate(lines):
        s = line.strip()
        if not s.startswith(("elif", "if")):
            continue
        # gh #225: a rewritten arm is `cmd_eq(tp, tl, "name")` with no `==` in
        # it at all, so the old "must contain ==" filter silently dropped 137 of
        # them and the table shrank from 322 to 185 commands. A command missing
        # from the table still works normally but is REJECTED inside MULTI,
        # which is exactly the asymmetric drift this generator exists to stop.
        if "==" not in s and not CEQ.search(s):
            continue
        if "req.cmd" in s or "CMD_" in s:
            continue  # 0xCA5E binary lane: no RESP name, no transactions

        hit = False
        for m in CEQ.finditer(s):
            names.setdefault(m.group(1).lower(), "cmd_eq")
            hit = True
        if hit:
            continue

        for m in CM.finditer(s):
            n, nums = int(m.group(1)), [x for x in m.group(2).split(",") if x.strip()]
            if len(nums) == n and (d := decode_literal(nums)):
                names.setdefault(d.lower(), "literal")
                hit = True
        if hit:
            continue

        tl_m = TL.search(s)
        tl = int(tl_m.group(1)) if tl_m else None
        folded = {int(i): int(v) for i, v in FOLDED.findall(s)}
        exact = {int(i): int(v) for i, v in EXACT.findall(s)}
        if tl is None and not folded and not exact:
            continue  # not a command-name arm (guards, option parsing)

        # Scan the whole arm body, to the next arm at the SAME indentation.
        # A fixed 6-line window silently missed FT.SEARCH, whose handler call
        # sits ~16 lines into its body — and a silently missing command is
        # precisely the drift this table is meant to eliminate, so an arm that
        # yields no handler is now REPORTED, never skipped.
        arm_indent = len(line) - len(line.lstrip())
        hs = []
        for look in lines[idx + 1:]:
            if not look.strip():
                continue
            ind = len(look) - len(look.lstrip())
            ls = look.strip()
            if ind <= arm_indent and ls.startswith(("elif", "else", "if ")):
                break
            hs += HANDLER.findall(ls)
        if not hs:
            # Some arms dispatch inline via dispatcher.execute_* and call no
            # handle_*. Their names come from a hint pool and are accepted ONLY
            # if the arm's own constraints admit exactly one — a wrong hint
            # fails the build rather than poisoning the table.
            fits = sorted(c for c in INLINE_HINTS
                          if satisfies(c, tl, folded, exact))
            if len(fits) == 1:
                names.setdefault(fits[0], "inline-hint")
            else:
                problems.append(("no-handler-in-arm", s[:70], tl, fits or sorted(folded.items())))
            continue

        # An arm with no length constraint whose body holds several distinct
        # handlers is a FAMILY DISPATCHER (e.g. the `FT.` prefix arm wrapping
        # per-subcommand arms), not a command. Its children carry their own
        # `tl` and are visited separately by this same loop, so skip it rather
        # than reporting every sibling as an ambiguity.
        if tl is None and len(set(hs)) > 1:
            continue

        survivors = sorted({c for h in hs for c in candidates(h)
                            if satisfies(c, tl, folded, exact)})
        if len(survivors) == 1:
            names.setdefault(survivors[0], "verified")
        elif len(survivors) > 1:
            corroborated = [c for c in survivors if c.upper() in docs()]
            if len(corroborated) == 1:
                names.setdefault(corroborated[0], "doc")
            else:
                problems.append(("ambiguous", hs[0], tl, survivors))
        else:
            problems.append(("unverified", hs[0], tl, sorted(folded.items())))

    # fast_path arms name their command in a trailing comment
    for line in (SRC / "fast_path.mojo").read_text().splitlines():
        s = line.strip()
        if not FP.search(s):
            continue
        m = re.search(r"#\s*.*?-\s*([A-Z][A-Z0-9\-\.]+)", s) or \
            re.search(r"#\s*([A-Z][A-Z0-9\-\.]{1,15})\b", s)
        if m:
            names.setdefault(m.group(1).lower(), "fastpath")
    return names, problems


HEADER = '''"""Pion's command surface — GENERATED, do not hand-edit (gh #220).

Regenerate with `python3 tools/gen_command_table.py`; the drift test
`tests/test_command_table_drift.py` fails the build if this file and the
dispatch chains disagree.

Purpose: queue-time validation for MULTI. Redis rejects an unknown command when
it is QUEUED and then answers EXEC with -EXECABORT, applying nothing. Pion used
to queue anything with +QUEUED and only fail at replay, by which point the
other commands in the transaction had already been applied — one typo silently
turned an atomic transaction into a partially applied one.

Case folding here is A-Z only, NOT the usual `| 0x20`: {underscores} of these
names contain '_' (AI.KNN_LM.QUERY), and '_' | 0x20 is 0x7F, so the cheap fold
would fail to match every substrate command.
"""

from src.common.ptr import null_ptr, is_null, is_not_null


comptime PION_COMMAND_COUNT = {count}


@always_inline
def _cmd_eq_ci(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int,
               lit: StringLiteral) -> Bool:
    """Case-insensitive ASCII match, folding only A-Z (see the note above)."""
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


def command_exists(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Bool:
    """Is this a command Pion dispatches? Bucketed by length so a lookup
    compares against only the names of that length."""
'''


ARITY_FILE = ROOT / "tools" / "redis_arity.txt"


def redis_arity():
    """Redis's own arity for every command Pion shares with it.

    Committed as data rather than queried at generation time, so the generator
    stays runnable with no redis-server. Refresh with `--refresh-arity` against
    a live one. Positive N means "exactly N tokens including the name";
    negative N means "at least |N|" — Redis's own encoding, kept verbatim so
    the semantics are checkable against `COMMAND INFO` rather than re-derived.

    A command with no entry here is NOT validated. Every Pion-specific command
    (FT.*, KV.PREFIX.*, AI.*, ATTEND.*, ...) falls in that bucket, and the
    conservative direction matters: a wrong arity here would reject a VALID
    transaction, which is worse than the missing check it replaces.
    """
    out = {}
    if not ARITY_FILE.exists():
        return out
    for line in ARITY_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


def _command_rows():
    """Raw `COMMAND` reply from a live redis-server on $REDIS_PORT (default 6379).

    Shared by the arity and write-flag refreshers so there is exactly one RESP
    reader to get wrong. Row layout is Redis's: [name, arity, flags, ...].
    """
    import os
    import socket
    port = int(os.environ.get("REDIS_PORT", "6379"))
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    f = s.makefile("rb")

    def rd():
        line = f.readline()
        t, b = line[:1], line[1:-2]
        if t == b"$":
            n = int(b)
            return None if n == -1 else f.read(n + 2)[:-2]
        if t in (b"*", b"~", b">"):
            n = int(b)
            return None if n == -1 else [rd() for _ in range(n)]
        if t == b"%":
            n = int(b)
            return [rd() for _ in range(2 * n)]
        if t == b":":
            return int(b)
        return b.decode()

    s.sendall(b"*1\r\n$7\r\nCOMMAND\r\n")
    rows = rd()
    # `rd` is a general RESP reader whose return type is a union; COMMAND always
    # replies with an array, so normalise here rather than at each caller.
    return rows if isinstance(rows, list) else []


def refresh_arity():
    """Re-query a live redis-server for every command's arity."""
    got = {}
    for c in _command_rows():
        if not isinstance(c, list) or len(c) < 2 or not isinstance(c[1], int):
            continue
        name = c[0].decode() if isinstance(c[0], bytes) else str(c[0])
        got[name.lower()] = c[1]
    return got


WRITE_FILE = ROOT / "tools" / "redis_write.txt"

# EVAL/EVALSHA/FCALL are NOT flagged `write` by Redis — it decides per script,
# at runtime, from the commands the script actually calls. Pion does the same
# since #36: a script's redis.call() goes through the slow-path dispatcher, so
# each command it calls meets the WAL-full and maxmemory gates itself. (They
# were forced `write` here while scripts ran a private command copy.)
FORCE_WRITE = set()


def redis_write_flags():
    """The set of commands real Redis marks `write` in `COMMAND INFO` flags.

    Committed as data rather than queried at generation time, exactly like
    `redis_arity()` — the generator must stay runnable with no redis-server.
    Refresh with `--refresh-write-flags` against a live one.

    A command with NO entry here is treated as a non-write, and for Pion's
    substrate surface that is correct rather than merely conservative: FT.*,
    KV.PREFIX.*, AI.*, ATTEND.*, V.* write their OWN stores, never the shared
    keyspace WAL (the substrate/engine boundary), so a full keyspace
    log says nothing about their durability and must not refuse them.
    """
    out = set()
    if not WRITE_FILE.exists():
        return out
    for line in WRITE_FILE.read_text().splitlines():
        n = line.strip()
        if n:
            out.add(n)
    return out


def refresh_write_flags():
    """Re-query a live redis-server on $REDIS_PORT for `write`-flagged names."""
    got = set()
    for c in _command_rows():
        if not isinstance(c, list) or len(c) < 3:
            continue
        name = c[0].decode() if isinstance(c[0], bytes) else str(c[0])
        flags = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in (c[2] or [])]
        if "write" in flags:
            got.add(name.lower())
    return got


DENYOOM_FILE = ROOT / "tools" / "redis_denyoom.txt"

# gh #261: Pion commands that ingest new data into a store of their own and so
# grow memory without bound. Redis has no opinion on them (they do not exist
# there), and leaving them out would make --maxmemory a limit that the
# substrate — Pion's largest memory consumer — simply walks past. Commands that
# mix reads and writes behind a subcommand (AI.SEMANTIC_CACHE GET|SET) are NOT
# listed: refusing their reads under memory pressure is the worse failure.
PION_DENYOOM = {
    "kv.store", "kv.prefix.register", "kv.prefix.commit", "kv.prefix.warm",
    "v.create", "v.commit", "v.storebatch", "v.restore",
    "attend.create", "attend.store", "attend.prefix.store",
    "ft.create", "ft.addtext", "ft.optimize",
    "ai.knn_lm.create", "ai.knn_lm.store", "ai.knn_lm.storebatch",
    "ai.route.register", "ai.loadmodel",
    "neuron.pkm.create", "neuron.pkm.setkeys", "neuron.pkm.setvals",
    "moe.expert.load",
}


def redis_denyoom_flags():
    """Commands real Redis marks `denyoom`: refused while used memory > maxmemory.

    Committed data, like the write flags; refresh with --refresh-denyoom-flags.
    Note what is NOT here: DEL, the POP family, EXPIRE — they free memory or
    cannot grow it, so Redis keeps serving them under the limit, and so does
    Pion. EVAL/EVALSHA/FCALL are absent too: Redis runs a script under the limit
    and refuses only the denyoom commands it CALLS, which Pion's Lua dispatch
    mirrors.
    """
    out = set()
    if DENYOOM_FILE.exists():
        out = {l.strip() for l in DENYOOM_FILE.read_text().splitlines() if l.strip()}
    return out


def refresh_flag(flag):
    """Re-query a live redis-server on $REDIS_PORT for commands carrying `flag`."""
    got = set()
    for c in _command_rows():
        if not isinstance(c, list) or len(c) < 3:
            continue
        name = c[0].decode() if isinstance(c[0], bytes) else str(c[0])
        flags = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in (c[2] or [])]
        if flag in flags:
            got.add(name.lower())
    return got


NOSCRIPT_FILE = ROOT / "tools" / "redis_noscript.txt"


def redis_noscript_flags():
    """Commands (`name`) and subcommands (`container|sub`, or `container|*` for
    every subcommand but HELP) real Redis flags `noscript`: refused when a
    script's redis.call() names them (#36). Committed data like the write
    flags; refresh with --refresh-noscript-flags."""
    out = set()
    if NOSCRIPT_FILE.exists():
        out = {l.strip() for l in NOSCRIPT_FILE.read_text().splitlines() if l.strip()}
    return out


def refresh_noscript():
    """Re-query a live redis-server for `noscript`, subcommands included."""
    got = set()
    for c in _command_rows():
        if not isinstance(c, list) or len(c) < 3:
            continue
        name = (c[0].decode() if isinstance(c[0], bytes) else str(c[0])).lower()
        flags = [x.decode() if isinstance(x, bytes) else str(x) for x in (c[2] or [])]
        subs = c[9] if len(c) > 9 and isinstance(c[9], list) else []
        if not subs:
            if "noscript" in flags:
                got.add(name)
            continue
        ns, others = [], []
        for sc in subs:
            sn = (sc[0].decode() if isinstance(sc[0], bytes) else str(sc[0])).lower()
            sf = [x.decode() if isinstance(x, bytes) else str(x) for x in (sc[2] or [])]
            (ns if "noscript" in sf else others).append(sn)
        if ns and others == [name + "|help"]:
            got.add(name + "|*")
        else:
            got.update(ns)
    return got


NOSCRIPT_HEADER = """

def command_is_noscript(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int,
                        sp: Pointer[UInt8, MutUntrackedOrigin], sl: Int) -> Bool:
    \"\"\"True when a script's redis.call() may not run this command (#36): real
    Redis's `noscript` flag. `sp`/`sl` is the first argument, for the container
    commands Redis flags per subcommand; `container|*` means every subcommand
    but HELP. {known} entries.
    \"\"\"
"""


DENYOOM_HEADER = """

@always_inline
def command_is_denyoom(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Bool:
    \"\"\"True when this command is refused while memory is over --maxmemory (gh #261).

    {known} of {total} commands: real Redis's `denyoom` flag plus Pion's
    substrate ingest commands (PION_DENYOOM in tools/gen_command_table.py).
    Reads, DEL and the POP family stay served under the limit, as in Redis.
    \"\"\"
"""


WRITE_HEADER = """

@always_inline
def command_is_write(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Bool:
    \"\"\"True when this command mutates the keyspace, per real Redis's `write` flag.

    {known} of {total} commands are writes. Used by gh #260 to refuse mutations
    once the WAL can no longer persist them, instead of acknowledging writes
    that will not survive a restart.

    A command absent from this table is NOT a write. For Pion's substrate
    surface (FT.*, KV.PREFIX.*, AI.*, ATTEND.*, V.*) that is correct, not just
    conservative: those planes own their own stores and never append to the
    shared keyspace WAL, so a full keyspace log is not a statement about them.

    EVAL/EVALSHA/FCALL are forced true. Redis classifies scripts per-invocation
    from what they actually call; Pion cannot, so it takes the safe side.
    \"\"\"
"""


ARITY_HEADER = """

@always_inline
def command_arity(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) -> Int:
    \"\"\"Redis's arity for this command, or 0 when Pion does not know one.

    {known} of {total} commands have an entry; the rest are Pion-specific
    (FT.*, KV.PREFIX.*, AI.*, ATTEND.*) and are deliberately NOT validated.

    Encoding is Redis's own, kept verbatim so it can be checked against
    `COMMAND INFO` rather than re-derived: positive N means exactly N tokens
    INCLUDING the command name; negative N means at least |N|.

    0 means "no opinion" and MUST be treated as valid. The conservative
    direction matters here — a wrong arity rejects a transaction that would
    have worked, which is a worse failure than the missing check it replaces.
    \"\"\"
"""


def emit(names):
    by_len = {}
    for n in sorted(names):
        by_len.setdefault(len(n), []).append(n)

    body = HEADER.format(count=len(names),
                         underscores=sum(1 for n in names if "_" in n))
    first = True
    for ln in sorted(by_len):
        kw = "if" if first else "elif"
        first = False
        body += f"    {kw} tl == {ln}:\n        return (\n"
        entries = by_len[ln]
        for k, n in enumerate(entries):
            tail = "" if k == len(entries) - 1 else " or"
            body += f'            _cmd_eq_ci(tp, tl, "{n}"){tail}\n'
        body += "        )\n"
    body += "    return False\n"

    # ---- arity table (gh #220 remainder) -------------------------------
    ar = redis_arity()
    have = sorted(n for n in names if n in ar)
    body += ARITY_HEADER.format(known=len(have), total=len(names))
    ar_by_len = {}
    for n in have:
        ar_by_len.setdefault(len(n), []).append(n)
    first = True
    for ln in sorted(ar_by_len):
        kw = "if" if first else "elif"
        first = False
        body += f"    {kw} tl == {ln}:\n"
        for n in ar_by_len[ln]:
            body += f'        if _cmd_eq_ci(tp, tl, "{n}"): return {ar[n]}\n'
    body += "    return 0\n"

    # ---- write-flag table (gh #260) ------------------------------------
    wr = redis_write_flags() | FORCE_WRITE
    writes = sorted(n for n in names if n in wr)
    body += WRITE_HEADER.format(known=len(writes), total=len(names))
    w_by_len = {}
    for n in writes:
        w_by_len.setdefault(len(n), []).append(n)
    first = True
    for ln in sorted(w_by_len):
        kw = "if" if first else "elif"
        first = False
        body += f"    {kw} tl == {ln}:\n        return (\n"
        entries = w_by_len[ln]
        for k, n in enumerate(entries):
            tail = "" if k == len(entries) - 1 else " or"
            body += f'            _cmd_eq_ci(tp, tl, "{n}"){tail}\n'
        body += "        )\n"
    body += "    return False\n"

    # ---- noscript table (#36) ------------------------------------------
    nsf = redis_noscript_flags()
    top = sorted(n for n in names if n in nsf)
    subs = {}
    for e in sorted(nsf):
        if "|" in e:
            cont, sub = e.split("|", 1)
            if cont in names:
                subs.setdefault(cont, []).append(sub)
    body += NOSCRIPT_HEADER.format(known=len(top) + sum(len(v) for v in subs.values()))
    ns_by_len = {}
    for n in top:
        ns_by_len.setdefault(len(n), []).append(n)
    first = True
    for ln in sorted(ns_by_len):
        kw = "if" if first else "elif"
        first = False
        body += f"    {kw} tl == {ln}:\n        if (\n"
        entries = ns_by_len[ln]
        for k, n in enumerate(entries):
            tail = "" if k == len(entries) - 1 else " or"
            body += f'            _cmd_eq_ci(tp, tl, "{n}"){tail}\n'
        body += "        ):\n            return True\n"
    for cont in sorted(subs):
        body += f'    if _cmd_eq_ci(tp, tl, "{cont}"):\n'
        if subs[cont] == ["*"]:
            body += '        return sl > 0 and not _cmd_eq_ci(sp, sl, "help")\n'
        else:
            conds = " or ".join(f'_cmd_eq_ci(sp, sl, "{x}")' for x in subs[cont])
            body += f"        return {conds}\n"
    body += "    return False\n"

    # ---- denyoom table (gh #261) ----------------------------------------
    dn = redis_denyoom_flags() | PION_DENYOOM
    missing = sorted(PION_DENYOOM - set(names))
    if missing:
        raise SystemExit(f"PION_DENYOOM names no dispatched command: {missing}")
    deny = sorted(n for n in names if n in dn)
    body += DENYOOM_HEADER.format(known=len(deny), total=len(names))
    d_by_len = {}
    for n in deny:
        d_by_len.setdefault(len(n), []).append(n)
    first = True
    for ln in sorted(d_by_len):
        kw = "if" if first else "elif"
        first = False
        body += f"    {kw} tl == {ln}:\n        return (\n"
        entries = d_by_len[ln]
        for k, n in enumerate(entries):
            tail = "" if k == len(entries) - 1 else " or"
            body += f'            _cmd_eq_ci(tp, tl, "{n}"){tail}\n'
        body += "        )\n"
    body += "    return False\n"
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed table is stale (drift gate)")
    ap.add_argument("--refresh-arity", action="store_true",
                    help="re-query a live redis-server and rewrite tools/redis_arity.txt")
    ap.add_argument("--refresh-write-flags", action="store_true",
                    help="re-query a live redis-server and rewrite tools/redis_write.txt")
    ap.add_argument("--refresh-noscript-flags", action="store_true",
                    help="re-query a live redis-server and rewrite tools/redis_noscript.txt")
    ap.add_argument("--refresh-denyoom-flags", action="store_true",
                    help="re-query a live redis-server and rewrite tools/redis_denyoom.txt")
    args = ap.parse_args()

    if args.refresh_arity:
        got = refresh_arity()
        names_only, _ = collect()
        keep = {n: got[n] for n in names_only if n in got}
        ARITY_FILE.write_text("\n".join(f"{k} {v}" for k, v in sorted(keep.items())) + "\n")
        print(f"wrote {ARITY_FILE.relative_to(ROOT)} — {len(keep)} arities")

    if args.refresh_write_flags:
        got = refresh_write_flags()
        names_only, _ = collect()
        keep = sorted(n for n in names_only if n in got)
        WRITE_FILE.write_text("\n".join(keep) + "\n")
        print(f"wrote {WRITE_FILE.relative_to(ROOT)} — {len(keep)} write commands")

    if args.refresh_noscript_flags:
        got = refresh_noscript()
        names_only, _ = collect()
        keep = sorted(e for e in got if e.split("|", 1)[0] in names_only)
        NOSCRIPT_FILE.write_text("\n".join(keep) + "\n")
        print(f"wrote {NOSCRIPT_FILE.relative_to(ROOT)} — {len(keep)} noscript entries")

    if args.refresh_denyoom_flags:
        got = refresh_flag("denyoom")
        names_only, _ = collect()
        keep = sorted(n for n in names_only if n in got)
        DENYOOM_FILE.write_text("\n".join(keep) + "\n")
        print(f"wrote {DENYOOM_FILE.relative_to(ROOT)} — {len(keep)} denyoom commands")

    names, problems = collect()
    if problems:
        print(f"REFUSING to emit: {len(problems)} unresolved arm(s)", file=sys.stderr)
        for kind, h, tl, extra in problems[:20]:
            print(f"  {kind}: handler={h} tl={tl} {extra}", file=sys.stderr)
        return 2

    text = emit(names)
    if args.check:
        if not OUT.exists():
            print(f"DRIFT: {OUT} does not exist", file=sys.stderr)
            return 1
        if OUT.read_text() != text:
            print("DRIFT: command_table.mojo is stale — regenerate with "
                  "`python3 tools/gen_command_table.py`", file=sys.stderr)
            return 1
        print(f"command table is current ({len(names)} commands)")
        return 0

    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} — {len(names)} commands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
