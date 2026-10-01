#!/usr/bin/env python3
"""gh #220 — the command table must never drift from the dispatch chains.

`src/commands/command_table.mojo` backs queue-time validation inside MULTI. If
it falls behind the dispatcher, the failure is asymmetric and easy to miss: a
newly added command works normally but is REJECTED inside a transaction. That
is worse than the bug the table fixes, so the table is generated and this test
regenerates it and fails on any difference.

It is a pure source check — no server, no build, no benchmark — so it can run
anywhere, any time.

Usage: python3 tests/test_command_table_drift.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "tools" / "gen_command_table.py"
TABLE = ROOT / "src" / "commands" / "command_table.mojo"


def main():
    if not GEN.exists():
        print(f"FAIL: generator missing at {GEN}")
        return 1
    if not TABLE.exists():
        print(f"FAIL: {TABLE} missing — run `python3 tools/gen_command_table.py`")
        return 1

    r = subprocess.run([sys.executable, str(GEN), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.strip() or r.stderr.strip())

    if r.returncode == 2:
        print("\nFAIL: the generator REFUSED to emit — one or more dispatch arms\n"
              "could not be resolved to exactly one command name. That is the\n"
              "check working: it is how gh #221 (ZREVRANGEBYSCORE/ZREVRANGEBYLEX\n"
              "carrying each other's lengths) was found. Fix the arm, or add a\n"
              "hint to INLINE_HINTS if the arm dispatches inline.")
        return 1
    if r.returncode != 0:
        print("\nFAIL: command_table.mojo is stale.\n"
              "Regenerate with `python3 tools/gen_command_table.py` and commit it.")
        return 1

    # A table that somehow lost the basics would still be "current"; assert the
    # floor explicitly so an empty or truncated emit cannot pass.
    text = TABLE.read_text()
    required = ["get", "set", "del", "multi", "exec", "discard", "watch",
                "ft.search", "zrevrangebyscore", "zrevrangebylex",
                "ai.knn_lm.query", "replconf"]
    missing = [c for c in required if f'"{c}"' not in text]
    if missing:
        print(f"FAIL: table is current but missing core commands: {missing}")
        return 1

    count_line = [l for l in text.splitlines() if "PION_COMMAND_COUNT" in l]
    print(f"table current, {count_line[0].split('=')[-1].strip() if count_line else '?'} commands, "
          f"all {len(required)} spot-checks present")
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
