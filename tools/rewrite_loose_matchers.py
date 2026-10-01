#!/usr/bin/env python3
"""Rewrite prefix+length dispatch arms into whole-name `cmd_eq` calls (gh #225).

THE PROBLEM
-----------
`cmd_matches_N` stops at 8 bytes. Every longer command name — the whole AI
substrate surface, `FT.*`, `KV.PREFIX.*`, `ATTEND.*` — was matched by hand as
"length plus whichever bytes the author felt were distinctive". That accepts
names that are not the command, and more importantly it SHADOWS: any future
command sharing the checked prefix and length is silently swallowed by the
existing arm.

WHY THIS IS A SCRIPT AND NOT 166 HAND EDITS
-------------------------------------------
Hand-editing 166 conditions is exactly the failure mode gh #240 was about: a
per-site sweep converts the sites you happened to look at, and the next probe
finds the ones you missed. A rewriter either resolves an arm to exactly one
name or refuses to touch it, and reports what it refused.

HOW A NAME IS ESTABLISHED
-------------------------
Identical to `gen_command_table.py`, deliberately: the candidate comes from the
arm body's `handle_<name>` call (or the inline-dispatch hint pool), and every
`.`/`_` placement is then filtered by the arm's OWN constraints — exact length,
case-folded bytes, literal bytes. A candidate that survives is the only string
that arm could ever have matched, so substituting `cmd_eq` for the loose test
cannot change which inputs the arm accepts, except to stop accepting the ones
that were never the command.

That shared derivation is also the safety net: `tests/test_command_table_drift.py`
regenerates the command table from these same arms, so a rewrite that changed
an arm's meaning shows up there as drift.

Usage:  python3 tools/rewrite_loose_matchers.py --dry-run
        python3 tools/rewrite_loose_matchers.py --apply
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_command_table import (  # noqa: E402
    CM, HANDLER, TL, FOLDED, EXACT, INLINE_HINTS, ROOT, SRC,
    candidates, decode_literal, satisfies,
)

TARGET = SRC / "slow_path.mojo"

# Names the `.`/`_` permutation search cannot reach, because a segment of the
# handler name itself contains an underscore (`knn_lm`, `semantic_cache`,
# `query_sparse_auto_fused`, `bitfield_ro`). Keyed by handler so it survives
# line moves. These are NOT trusted blindly — each still has to satisfy the
# arm's own length and byte constraints below, so a typo here fails loudly
# instead of silently rewriting an arm to match something else.
EXPLICIT = {
    "handle_ai_semantic_cache": "ai.semantic_cache",
    "handle_ai_knn_lm_storebatch": "ai.knn_lm.storebatch",
    "handle_ai_knn_lm_create": "ai.knn_lm.create",
    "handle_ai_knn_lm_store": "ai.knn_lm.store",
    "handle_ai_knn_lm_query": "ai.knn_lm.query",
    "handle_ai_knn_lm_info": "ai.knn_lm.info",
    "handle_ai_knn_lm_drop": "ai.knn_lm.drop",
    "handle_attend_prefix_query_sparse_auto_fused":
        "attend.prefix.query_sparse_auto_fused",
    "handle_rag_speculate_enable": "rag.speculate.enable",
    "handle_bitfield_ro": "bitfield_ro",
}

# A conjunct that is purely a command-name test — these get replaced wholesale.
NAME_CONJUNCT = re.compile(
    r"^\(?\s*(?:"
    r"tl\s*==\s*\d+"
    r"|\(\s*tp\[\s*\d+\s*\]\s*\|\s*0x20\s*\)\s*==\s*\d+"
    r"|tp\[\s*\d+\s*\]\s*==\s*\d+"
    r"|cmd_matches_\d+\s*\(\s*tp\s*,[^()]*\)"
    r")\s*\)?$"
)


def split_conjuncts(cond):
    """Split on top-level ` and `. Returns None if the condition has a top-level
    `or`, or unbalanced brackets — those are not safe to rewrite mechanically."""
    parts, depth, cur, i = [], 0, "", 0
    while i < len(cond):
        c = cond[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
            if depth < 0:
                return None
        elif c == '"':
            j = cond.index('"', i + 1)
            cur += cond[i:j + 1]
            i = j + 1
            continue
        if depth == 0 and cond.startswith(" and ", i):
            parts.append(cur.strip())
            cur = ""
            i += 5
            continue
        if depth == 0 and cond.startswith(" or ", i):
            return None
        cur += c
        i += 1
    if depth != 0:
        return None
    parts.append(cur.strip())
    return parts


def resolve(lines, idx, s, tl, folded, exact):
    """The arm's single possible command name, or None."""
    arm_indent = len(lines[idx]) - len(lines[idx].lstrip())
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
        fits = sorted(c for c in INLINE_HINTS if satisfies(c, tl, folded, exact))
        return fits[0] if len(fits) == 1 else None
    if tl is None:
        return None                     # family dispatcher, not a command
    survivors = sorted({c for h in hs for c in candidates(h)
                        if satisfies(c, tl, folded, exact)})
    if len(survivors) == 1:
        return survivors[0]
    for h in hs:
        cand = EXPLICIT.get("handle_" + h)
        if cand and satisfies(cand, tl, folded, exact):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (args.apply or args.dry_run):
        ap.error("pass --apply or --dry-run")

    lines = TARGET.read_text().splitlines()
    out = list(lines)
    rewritten, skipped = [], []

    for idx, line in enumerate(lines):
        s = line.strip()
        if not s.startswith(("elif ", "if ")) or "==" not in s or not s.endswith(":"):
            continue
        if "req.cmd" in s or "CMD_" in s:
            continue

        tl_m = TL.search(s)
        tl = int(tl_m.group(1)) if tl_m else None
        folded = {int(i): int(v) for i, v in FOLDED.findall(s)}
        exact = {int(i): int(v) for i, v in EXACT.findall(s)}
        if tl is None:
            continue

        # Already exact? cmd_matches_N covering every byte needs no change.
        exact_already = False
        for m in CM.finditer(s):
            n = int(m.group(1))
            nums = [x for x in m.group(2).split(",") if x.strip()]
            if len(nums) == n == tl and decode_literal(nums):
                exact_already = True
        pinned = set(folded) | set(exact)
        if exact_already or pinned >= set(range(tl)):
            continue
        if "cmd_eq(" in s:
            continue

        kw, cond = s.split(" ", 1)
        cond = cond.rstrip(":").strip()
        parts = split_conjuncts(cond)
        if parts is None:
            skipped.append((idx + 1, "non-trivial boolean", s[:70]))
            continue

        name = resolve(lines, idx, s, tl, folded, exact)
        if name is None:
            skipped.append((idx + 1, "name not uniquely resolvable", s[:70]))
            continue
        if len(name) != tl:
            skipped.append((idx + 1, "resolved name length != tl", s[:70]))
            continue

        keep = [p for p in parts if not NAME_CONJUNCT.match(p)]
        if len(keep) == len(parts):
            skipped.append((idx + 1, "no name conjuncts found", s[:70]))
            continue

        new_cond = " and ".join([f'cmd_eq(tp, tl, "{name}")'] + keep)
        indent = line[:len(line) - len(line.lstrip())]
        out[idx] = f"{indent}{kw} {new_cond}:"
        rewritten.append((idx + 1, name, len(parts) - len(keep)))

    print(f"{len(rewritten)} arms rewritten to cmd_eq, {len(skipped)} skipped\n")
    for ln, name, n in rewritten[:15]:
        print(f"  line {ln:5d}  {name:28s} ({n} conjuncts folded)")
    if len(rewritten) > 15:
        print(f"  ... and {len(rewritten)-15} more")
    if skipped:
        print("\nSkipped (left exactly as they were):")
        for ln, why, src in skipped[:20]:
            print(f"  line {ln:5d}  {why}")
            print(f"           {src}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped)-20} more")

    if args.apply and rewritten:
        TARGET.write_text("\n".join(out) + "\n")
        print(f"\nwrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
