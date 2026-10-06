#!/usr/bin/env python3
"""Every measured number in the published docs names its evidence.

Runs tools/check_doc_claims.py over README.md, the landing page, website/pages,
doc/ and the package READMEs, against benchmarks/claims.toml. Fails when a
number with a performance unit has no register entry, when an entry's evidence
file is gone or no longer contains what the entry expects, or when an entry
matches nothing any more.

It canaries itself first: the detector must find the claim shapes that were
published without evidence before this check existed, and must skip the
shapes that name an input rather than a result. Without that, a detector that
found nothing would pass every doc.

No server, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import check_doc_claims as C  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)


print("[1] the detector finds claims (canary)")
for text, want in [
    ("Pion measures **14.0M ops/sec** at P=50", "14.0Mops/sec"),
    ("| **QPS** (50K, 1536D, 10 clients) | **5,445** | 6,803 (+25%) |", "25%"),
    ("| **QPS** (50K, 1536D, 10 clients) | **5,445** | 6,803 (+25%) |", "5445"),
    ("Cache hit performance: **275× faster** than a live LLM call", "275×"),
    ("retrieved in 86us per layer", "86us"),
    ("| Recall@100 | 0.920 | **0.937** |", "0.937"),
    ("RAM/Worker ~700 MB", "700MB"),
    ("| Redis 8.6 | 1.37M | 4.21M |", "1.37M"),
]:
    got = C.claims_in(text)
    check(f"finds {want!r}", want in got, f"got {got}")

print("[2] the detector skips inputs and shapes (canary)")
for text in [
    "at 64K context with a sparse mask",
    "a 2,049-token prefix",
    "1M slots, dim 896",
    "on a 16 GB Mac",
    "5×20 = 100 requests",
    "2K+1 syscalls per batch of K ready fds",
    "48 x 18B blocks",
]:
    got = C.claims_in(text)
    check(f"skips {text!r}", not got, f"got {got}")

print("[3] evidence a clean checkout would not have is caught (canary)")
ign = C.ignored_by_git(["scratch-not-evidence.txt", "benchmarks/results/any/run.txt"])
check("a .txt outside benchmarks/results/ is reported as gitignored", "scratch-not-evidence.txt" in ign, f"got {ign}")
check("raw results under benchmarks/results/ are not", "benchmarks/results/any/run.txt" not in ign, f"got {ign}")

print("[4] the published docs")
rc = C.check()
check("every measured number in the published docs names its evidence", rc == 0,
      "run `python3 tools/check_doc_claims.py` for the list")

print(f"\n{'PASS' if not failures else 'FAIL'} — {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
