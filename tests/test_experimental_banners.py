#!/usr/bin/env python3
"""Every part the README lists as experimental says so on its own page.

The README's "Experimental" table is the list of everything outside the
supported surface (the prompt cache and serve, the Redis-compatible KV, vector
search and the semantic cache). A reader who lands on one of those pages from
a search engine never sees the README, so each page has to carry the banner
itself. This pins the two together: every row's page must open with an
**Experimental** banner, and the table must not shrink to nothing.

A row's page is its first link: a repository path, or a docs-site URL, which
maps to the assembled page under website/pages/. No server, no network.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://pion.pavelhorak.com/docs/"
BANNER = re.compile(r"\*\*Experimental\.\*\*|\*\*experimental\*\*")
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        fails.append(name)


def page_of(link: str) -> Path | None:
    if link.startswith(SITE):
        rel = link[len(SITE):].strip("/")
        for cand in (ROOT / "website/pages" / f"{rel}.md", ROOT / "doc" / f"{rel}.md"):
            if cand.exists():
                return cand
        return None
    if link.startswith(("http://", "https://")):
        return None
    p = ROOT / link.split("#")[0]
    return p if p.exists() else None


print("[1] the banner pattern (canary)")
check("matches the banner", bool(BANNER.search("> **Experimental.** The exo hook ...")))
check("does not match a page without one", not BANNER.search("# pion-exo\n\nPion attention hook for exo."))

print("[2] every experimental part's page")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
m = re.search(r"^## Experimental\n(.*?)(?=^## )", readme, re.S | re.M)
check("the README has an Experimental section", m is not None)
rows = [ln for ln in (m.group(1) if m else "").splitlines()
        if ln.startswith("|") and not ln.startswith(("| Part", "|---"))]
check("it lists at least ten parts", len(rows) >= 10, f"{len(rows)} rows")
for row in rows:
    cells = [c.strip() for c in row.strip("|").split("|")]
    links = re.findall(r"\]\(([^)]+)\)", cells[1] if len(cells) > 1 else "")
    name = re.sub(r"`", "", cells[0])[:60]
    if not links:
        check(f"{name}: no page to carry a banner (allowed only for 'no test' rows)",
              "no test" in row, row[:120])
        continue
    for link in links:
        page = page_of(link)
        if page is None:
            check(f"{name}: {link} resolves to a file in the repository", False, link)
            continue
        head = "\n".join(page.read_text(encoding="utf-8").splitlines()[:15])
        check(f"{name}: {page.relative_to(ROOT)} opens with an Experimental banner", bool(BANNER.search(head)))

print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} failure(s)")
sys.exit(1 if fails else 0)
