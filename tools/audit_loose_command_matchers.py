#!/usr/bin/env python3
"""Audit: dispatch arms that do NOT pin every byte of their command name.

gh #162's lesson: "match the whole command name". An arm
that checks a prefix plus a length accepts names that are not the command —
`PINX` and `PI\\x00G` execute as PING, `DBSIZ\\x00` as DBSIZE — and silently
shadows any future command that shares the prefix and length.

`cmd_matches_N(tp, ...)` pins every byte, so those arms are exact by
construction. This finds the hand-rolled ones.
"""
import re
from pathlib import Path

# Derived from this file's own location, not hardcoded: an absolute path
# into one developer's home directory makes the tool unrunnable for anyone
# else, and leaks that path into any public export.
SRC = Path(__file__).resolve().parents[1] / "src" / "network"
if len(__import__("sys").argv) > 1:          # a directory holding the two files (the test's canary)
    SRC = Path(__import__("sys").argv[1])
TL = re.compile(r"\btl\s*==\s*(\d+)")
CL = re.compile(r"\bcmd_len\s*==\s*(\d+)")
FOLDED = re.compile(r"\(\s*(?:tp|buffer)\[\s*(?:cmd_start \+ )?(\d+)\s*\]\s*\|\s*0x20\s*\)\s*==\s*(\d+)")
EXACT = re.compile(r"(?<![|\w])(?:tp|buffer)\[\s*(?:cmd_start \+ )?(\d+)\s*\]\s*==\s*(\d+)")
CM = re.compile(r"cmd_matches_(\d+)\s*\(")
B0 = re.compile(r"\bb0_lower\s*==\s*(\d+)")

rows = []
CEQ = re.compile(r"cmd_eq\s*\(")


def body_pins_the_name(lines, idx):
    """True when this arm is a FAMILY GATE, not a command match.

    `elif b0_lower == 102 and cmd_len == 8:` looks loose in isolation, but its
    body is `if cmd_matches_8(...) FLUSHALL else -> slow path` — the name IS
    pinned, one level down, and the fall-through routes to a dispatcher that
    matches exactly. Reporting those forever would train readers to ignore a
    permanently non-zero audit, which is worse than not running it.
    """
    arm_indent = len(lines[idx]) - len(lines[idx].lstrip())
    body = []
    for look in lines[idx + 1:]:
        if not look.strip():
            continue
        ind = len(look) - len(look.lstrip())
        ls = look.strip()
        if ind <= arm_indent and ls.startswith(("elif", "else", "if ")):
            break
        if CM.search(ls) or CEQ.search(ls):
            return True
        body.append(ls)
    # An arm whose entire body hands the frame to the slow path executes
    # NOTHING on a near-miss, so it cannot shadow: the slow path re-matches the
    # name exactly. `elif b0_lower == 102 and cmd_len == 5: return consumed` is
    # a routing hint, not a command match.
    if body and all(b.startswith(("return consumed", "fast_path_ok = False",
                                  "break", "#")) for b in body):
        return True
    return False


for fname in ("fast_path.mojo", "slow_path.mojo"):
    if not (SRC / fname).exists():
        continue
    src_lines = (SRC / fname).read_text().splitlines()
    for ln, line in enumerate(src_lines, 1):
        s = line.strip()
        if not s.startswith(("elif", "if ")) or "==" not in s:
            continue
        if "req.cmd" in s or "CMD_" in s:
            continue
        if CM.search(s) or CEQ.search(s):
            continue                      # pins every byte by construction
        if body_pins_the_name(src_lines, ln - 1):
            continue
        m = TL.search(s) or CL.search(s)
        if not m:
            continue
        n = int(m.group(1))
        pinned = {int(i) for i, _ in FOLDED.findall(s)} | {int(i) for i, _ in EXACT.findall(s)}
        if B0.search(s):
            pinned.add(0)                 # b0_lower pins index 0
        missing = sorted(set(range(n)) - pinned)
        if missing:
            name = "?"
            mm = re.search(r"#\s*'?\w'?.*?-?\s*([A-Z][A-Z0-9_\.]{1,20})", s)
            if mm:
                name = mm.group(1)
            rows.append((fname, ln, n, len(pinned), missing, name, s[:96]))

rows.sort(key=lambda r: len(r[4]), reverse=True)
print(f"{len(rows)} dispatch arms do not pin every byte of their name\n")
print(f"{'file':16s} {'line':>5s} {'len':>4s} {'pinned':>6s}  {'unchecked idx':22s} name")
for f, ln, n, p, miss, name, src in rows[:40]:
    print(f"{f:16s} {ln:5d} {n:4d} {p:6d}  {str(miss)[:22]:22s} {name}")
if len(rows) > 40:
    print(f"... and {len(rows)-40} more")
