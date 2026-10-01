#!/usr/bin/env python3
"""gh #23 — structural smoke test for `examples/cag_legal_demo/`.

The demo backs the `pion-serve --backend pion-cag-hybrid` getting-started flow
and is referenced from the README. None of the artifacts get touched by the regular Mojo build or `/gate`,
so an accidental edit/delete/schema-drift would only surface when a future
operator tried the docs' spin-up command — by which point we've shipped a
broken demo.

This test is intentionally LIGHTWEIGHT — no MLX, no model load, no pion-serve
subprocess:

  [1] All five artifact files exist at the documented paths.
  [2] The corpus is non-trivial (≥ 10 KB) — catches accidental truncation.
  [3] `calibration_qa.jsonl` parses; every row has `{question, answers}`.
  [4] `sample_queries.jsonl` parses; every row has `{id, case, question}`.
  [5] `calibration_state.json` parses + has the schema `pion-serve` will read
      (schema_version, calibration.{tier_fired, te, tm, gated_heldout_cv,
      per_fold}).
  [6] `README.md` mentions the `pion-serve --backend pion-cag-hybrid`
      spin-up command — catches docs drift.

When the cascade is restructured (different cluster naming, additional gate
metrics, etc.) the calibration-state schema check is the canary: edit the
test and the docs at the same time as the producer.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_DIR = os.path.join(PROJECT_ROOT, "examples", "cag_legal_demo")


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def main() -> int:
    failures = 0

    # ── [1] All artifact files present ─────────────────────────────────
    print("[1] artifact files present")
    artifacts = {
        "corpus":      os.path.join(DEMO_DIR, "corpus", "opinions.txt"),
        "qa":          os.path.join(DEMO_DIR, "calibration_qa.jsonl"),
        "state":       os.path.join(DEMO_DIR, "calibration_state.json"),
        "samples":     os.path.join(DEMO_DIR, "sample_queries.jsonl"),
        "readme":      os.path.join(DEMO_DIR, "README.md"),
    }
    for k, p in artifacts.items():
        if not os.path.exists(p):
            failures += _fail(f"missing: {k} ({p})")
    if failures:
        return failures
    print("   OK — 5/5 present")

    # ── [2] Corpus non-trivial ─────────────────────────────────────────
    print("[2] corpus ≥ 10 KB")
    sz = os.path.getsize(artifacts["corpus"])
    if sz < 10 * 1024:
        failures += _fail(f"corpus only {sz} bytes (< 10 KB) — likely truncated")
    else:
        print(f"   OK — {sz} bytes")

    # ── [3] calibration_qa.jsonl shape ─────────────────────────────────
    print("[3] calibration_qa.jsonl rows have {question, answers}")
    try:
        qa = _read_jsonl(artifacts["qa"])
    except Exception as e:
        return _fail(f"could not parse calibration_qa.jsonl: {e}") or failures
    if len(qa) < 10:
        failures += _fail(f"only {len(qa)} QA rows (< 10) — not a useful calibration set")
    for i, row in enumerate(qa):
        if "question" not in row or "answers" not in row:
            failures += _fail(f"row {i} missing keys; got {list(row.keys())}")
            break
        if not isinstance(row["answers"], list) or not row["answers"]:
            failures += _fail(f"row {i} 'answers' must be non-empty list, got {row['answers']!r}")
            break
    else:
        print(f"   OK — {len(qa)} rows, every row has both fields")

    # ── [4] sample_queries.jsonl shape ─────────────────────────────────
    print("[4] sample_queries.jsonl rows have {id, case, question}")
    try:
        samples = _read_jsonl(artifacts["samples"])
    except Exception as e:
        return _fail(f"could not parse sample_queries.jsonl: {e}") or failures
    if len(samples) < 5:
        failures += _fail(f"only {len(samples)} sample queries (< 5) — too few for hands-on")
    for i, row in enumerate(samples):
        missing = [k for k in ("id", "case", "question") if k not in row]
        if missing:
            failures += _fail(f"sample row {i} missing {missing}; got {list(row.keys())}")
            break
    else:
        print(f"   OK — {len(samples)} sample queries")

    # ── [5] calibration_state.json schema ──────────────────────────────
    print("[5] calibration_state.json schema (pion-serve reader contract)")
    try:
        state = json.load(open(artifacts["state"]))
    except Exception as e:
        return _fail(f"could not parse calibration_state.json: {e}") or failures
    for k in ("schema_version", "corpus", "calibration"):
        if k not in state:
            failures += _fail(f"calibration state missing top-level '{k}'")
    cal = state.get("calibration", {})
    for k in ("tier_fired", "te", "tm", "gated_heldout_cv", "per_fold"):
        if k not in cal:
            failures += _fail(f"calibration state missing 'calibration.{k}'")
    if "tier_fired" in cal and cal["tier_fired"] not in (1, 2, 3):
        failures += _fail(f"tier_fired={cal['tier_fired']} not in {{1,2,3}}")
    cv = cal.get("gated_heldout_cv", {})
    for k in ("f1", "found", "speedup", "k_folds"):
        if k not in cv:
            failures += _fail(f"calibration state missing 'gated_heldout_cv.{k}'")
    if cv.get("k_folds", 0) < 2:
        failures += _fail(f"k_folds={cv.get('k_folds')} not enough for CV (< 2)")
    if not failures:
        print(f"   OK — schema_version={state.get('schema_version')}, "
              f"tier={cal.get('tier_fired')}, "
              f"f1={cv.get('f1', 0):.3f}, found={cv.get('found', 0):.3f}, "
              f"speedup={cv.get('speedup', 0):.2f}×, k_folds={cv.get('k_folds')}")

    # ── [6] README mentions the spin-up command ────────────────────────
    print("[6] README contains pion-serve spin-up command")
    readme = open(artifacts["readme"]).read()
    needed = "--backend pion-cag-hybrid"
    if needed not in readme:
        failures += _fail(f"README missing '{needed}' — docs drift")
    else:
        print("   OK")

    if failures:
        print(f"\nFAIL — {failures} structural issue(s) in examples/cag_legal_demo/")
        return failures
    print("\nPASS — gh #23 CAG-hybrid demo artifacts structurally sound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
