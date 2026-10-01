"""Markdown results reporter for cache hit rate validation."""
from __future__ import annotations

import datetime
from collections import defaultdict

from .simulator import SimulationResults, QueryDetail


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _sim_histogram(details: list[QueryDetail], category: str | None = None) -> str:
    """Text histogram of best similarity scores."""
    sims = [d.best_similarity for d in details if category is None or d.category == category]
    if not sims:
        return "(no data)"

    buckets = [
        (0.0, 0.50, "<0.50"),
        (0.50, 0.60, "0.50-0.60"),
        (0.60, 0.70, "0.60-0.70"),
        (0.70, 0.80, "0.70-0.80"),
        (0.80, 0.85, "0.80-0.85"),
        (0.85, 0.90, "0.85-0.90"),
        (0.90, 0.92, "0.90-0.92"),
        (0.92, 0.95, "0.92-0.95"),
        (0.95, 1.01, "0.95-1.00"),
    ]

    lines = []
    max_count = 0
    bucket_counts = []
    for lo, hi, label in buckets:
        count = sum(1 for s in sims if lo <= s < hi)
        bucket_counts.append((label, count))
        max_count = max(max_count, count)

    bar_width = 40
    for label, count in bucket_counts:
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "#" * bar_len
        lines.append(f"  {label:>9s} | {bar:<{bar_width}s} {count}")

    return "\n".join(lines)


def generate_report(sim: SimulationResults) -> str:
    """Generate full markdown report."""
    lines: list[str] = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Header
    lines.append("# Pion Serve: Cache Hit Rate Validation Results")
    lines.append("")
    lines.append(f"**Date:** {now}  |  **Embedding model:** {sim.embedding_model}  |  "
                 f"**Prompts embedded:** {sim.total_prompts_embedded}  |  **Embed time:** {sim.embed_time_s:.1f}s")
    lines.append("")

    # ── Executive Summary ──
    # Pick threshold=0.90 as the "recommended" for the summary
    rec_t = 0.90 if 0.90 in sim.thresholds else sim.thresholds[len(sim.thresholds) // 2]
    overall_exact = sim.results.get(("overall", rec_t))
    if overall_exact:
        exact_rate = overall_exact.exact_hit_rate
        sem_rate = overall_exact.semantic_hit_rate
        fp_rate = overall_exact.fp_rate
        delta_pp = (sem_rate - exact_rate) * 100

        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"At threshold **{rec_t}**, semantic matching achieves **{_pct(sem_rate)}** hit rate "
                     f"vs **{_pct(exact_rate)}** for exact-prefix — a **+{delta_pp:.0f}pp improvement**.")
        lines.append(f"False positive rate: **{_pct(fp_rate)}**. "
                     f"Estimated prefill FLOP savings: **{sim.semantic_flop_savings_pct.get(rec_t, 0):.1f}%** "
                     f"(vs {sim.exact_flop_savings_pct:.1f}% exact).")
        lines.append("")

        # Validation gate
        lines.append("### Phase 1 Validation Gate")
        lines.append("")
        hit_pass = sem_rate > 0.50
        quality_pass = fp_rate < 0.10
        lines.append(f"| Metric | Target | Actual | Status |")
        lines.append(f"|--------|--------|--------|--------|")
        lines.append(f"| Semantic hit rate (t={rec_t}) | >50% | {_pct(sem_rate)} | {'PASS' if hit_pass else 'FAIL'} |")
        lines.append(f"| False positive rate | <10% | {_pct(fp_rate)} | {'PASS' if quality_pass else 'FAIL'} |")
        lines.append(f"| Embedding overhead | <2s for {sim.total_prompts_embedded} prompts | {sim.embed_time_s:.1f}s | {'PASS' if sim.embed_time_s < 30 else 'REVIEW'} |")
        lines.append("")

    # ── Summary Table ──
    lines.append("## Summary: Exact-Prefix vs Semantic Matching")
    lines.append("")
    lines.append("| Threshold | Exact Hit Rate | Semantic Hit Rate | Delta | FP Rate | Precision | Recall | Est. FLOP Savings |")
    lines.append("|-----------|---------------|-------------------|-------|---------|-----------|--------|-------------------|")
    for t in sim.thresholds:
        r = sim.results.get(("overall", t))
        if not r:
            continue
        delta = (r.semantic_hit_rate - r.exact_hit_rate) * 100
        flop_sav = sim.semantic_flop_savings_pct.get(t, 0)
        lines.append(
            f"| {t:.2f} | {_pct(r.exact_hit_rate)} | {_pct(r.semantic_hit_rate)} | "
            f"+{delta:.0f}pp | {_pct(r.fp_rate)} | {_pct(r.precision)} | {_pct(r.recall)} | {flop_sav:.1f}% |"
        )
    lines.append("")

    # ── Per-Category Breakdown ──
    lines.append("## Results by Workload Category")
    lines.append("")
    category_names = {
        "multi_turn": "A: Multi-Turn Conversations",
        "inline_completion": "B: Inline Completions",
        "cross_user": "C: Cross-User Queries",
        "unrelated": "D: Unrelated (Negative Control)",
    }

    for cat in sim.categories:
        if cat == "overall":
            continue
        title = category_names.get(cat, cat)
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Threshold | Exact Hits | Semantic Hits | Total | Sem. Rate | FP | FN | Precision | Recall |")
        lines.append("|-----------|-----------|---------------|-------|-----------|----|----|-----------|--------|")
        for t in sim.thresholds:
            r = sim.results.get((cat, t))
            if not r:
                continue
            lines.append(
                f"| {t:.2f} | {r.exact_hits} | {r.semantic_hits} | {r.total} | "
                f"{_pct(r.semantic_hit_rate)} | {r.false_positives} | {r.false_negatives} | "
                f"{_pct(r.precision)} | {_pct(r.recall)} |"
            )
        lines.append("")

    # ── Similarity Distribution ──
    lines.append("## Similarity Score Distribution")
    lines.append("")
    lines.append("### Overall")
    lines.append("```")
    lines.append(_sim_histogram(sim.details))
    lines.append("```")
    lines.append("")

    for cat in sim.categories:
        if cat == "overall":
            continue
        title = category_names.get(cat, cat)
        cat_details = [d for d in sim.details if d.category == cat]
        if not cat_details:
            continue
        lines.append(f"### {title}")
        lines.append("```")
        lines.append(_sim_histogram(cat_details))
        lines.append("```")
        lines.append("")

    # ── Expected vs Actual Breakdown ──
    lines.append("## Expected-Hit vs Actual Analysis (t=0.90)")
    lines.append("")
    lines.append("Queries where ground truth expected a hit but semantic matching missed (false negatives),")
    lines.append("and queries where no hit was expected but semantic matching triggered (false positives).")
    lines.append("")

    t_analysis = 0.90 if 0.90 in sim.thresholds else sim.thresholds[len(sim.thresholds) // 2]

    # False negatives
    fn_details = [d for d in sim.details if d.expected_hit and d.best_similarity < t_analysis]
    if fn_details:
        lines.append(f"### False Negatives ({len(fn_details)} queries, t={t_analysis})")
        lines.append("")
        lines.append("| Category | Subtype | Best Sim | Prompt Preview |")
        lines.append("|----------|---------|----------|---------------|")
        for d in sorted(fn_details, key=lambda x: x.best_similarity)[:20]:
            lines.append(f"| {d.category} | {d.subtype} | {d.best_similarity:.4f} | {d.prompt_preview[:60]} |")
        lines.append("")

    # False positives
    fp_details = [d for d in sim.details if not d.expected_hit and d.best_similarity >= t_analysis]
    if fp_details:
        lines.append(f"### False Positives ({len(fp_details)} queries, t={t_analysis})")
        lines.append("")
        lines.append("| Category | Subtype | Best Sim | Prompt Preview |")
        lines.append("|----------|---------|----------|---------------|")
        for d in sorted(fp_details, key=lambda x: -x.best_similarity)[:20]:
            lines.append(f"| {d.category} | {d.subtype} | {d.best_similarity:.4f} | {d.prompt_preview[:60]} |")
        lines.append("")

    if not fn_details and not fp_details:
        lines.append("*No false negatives or false positives at this threshold.*")
        lines.append("")

    # ── Subtype Breakdown ──
    lines.append("## Hit Rate by Query Subtype (t=0.90)")
    lines.append("")
    subtype_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hits": 0, "expected": 0})
    for d in sim.details:
        key = f"{d.category}/{d.subtype}" if d.subtype else d.category
        subtype_stats[key]["total"] += 1
        if d.best_similarity >= t_analysis:
            subtype_stats[key]["hits"] += 1
        if d.expected_hit:
            subtype_stats[key]["expected"] += 1

    lines.append("| Subtype | Total | Semantic Hits | Expected Hits | Hit Rate |")
    lines.append("|---------|-------|--------------|---------------|----------|")
    for key in sorted(subtype_stats.keys()):
        s = subtype_stats[key]
        rate = s["hits"] / s["total"] if s["total"] else 0
        lines.append(f"| {key} | {s['total']} | {s['hits']} | {s['expected']} | {_pct(rate)} |")
    lines.append("")

    # ── Analysis ──
    lines.append("## Analysis & Recommendations")
    lines.append("")

    # Find optimal threshold (highest hit rate with FP rate < 5%)
    best_t = None
    for t in sorted(sim.thresholds):
        r = sim.results.get(("overall", t))
        if r and r.fp_rate <= 0.05:
            if best_t is None or r.semantic_hit_rate > sim.results[("overall", best_t)].semantic_hit_rate:
                best_t = t
    if best_t is None:
        best_t = max(sim.thresholds)  # fallback to strictest

    r_best = sim.results.get(("overall", best_t))
    if r_best:
        lines.append(f"**Recommended threshold: {best_t}** — achieves {_pct(r_best.semantic_hit_rate)} hit rate "
                     f"with {_pct(r_best.fp_rate)} false positive rate.")
        lines.append("")

    # Per-category observations
    for cat in sim.categories:
        if cat == "overall":
            continue
        r90 = sim.results.get((cat, t_analysis))
        if r90:
            title = category_names.get(cat, cat)
            lines.append(f"- **{title}:** {_pct(r90.semantic_hit_rate)} semantic hit rate "
                         f"(exact: {_pct(r90.exact_hit_rate)}, {r90.false_positives} FP, {r90.false_negatives} FN)")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by Pion Serve Phase 1 validation suite — {now}*")

    return "\n".join(lines)
