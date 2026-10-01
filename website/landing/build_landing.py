#!/usr/bin/env python3
"""Render the landing page (https://pion.pavelhorak.com/) from the repository.

The landing page cannot say something the README does not: every paragraph,
command, number and caveat below is pulled from README.md snippet regions
and headings, from the two sections in website/landing/sections.md, and from
the version file. Edit those; never this page's output.

    python3 website/landing/build_landing.py            # → _site/index.html
    python3 website/landing/build_landing.py --out DIR
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sitelib as S  # noqa: E402

HERE = Path(__file__).resolve().parent
SECTIONS = "website/landing/sections.md"


def md(text: str, src_dir: str = "") -> str:
    """Markdown → HTML with links rewritten to GitHub / the docs site."""
    text = S.strip_html_comments(S.relink(text, src_dir))
    text = text.replace("](/", "](https://pion.pavelhorak.com/docs/")   # docs-absolute → site
    text = re.sub(r"\.md(#|\))", r"/\1", text)                           # docs .md → directory URLs
    return markdown.markdown(text, extensions=["tables", "fenced_code", "footnotes"])


def code(block: str) -> str:
    """A fenced block's body as a highlighted <pre>: comments dimmed."""
    out = []
    for ln in block.rstrip("\n").splitlines():
        if ln.lstrip().startswith("#"):
            out.append(f'<span class="c">{html.escape(ln)}</span>')
        elif "#" in ln:
            head, _, tail = ln.partition("#")
            out.append(html.escape(head) + f'<span class="c">#{html.escape(tail)}</span>')
        else:
            out.append(html.escape(ln))
    return "\n".join(out)


def first_paragraph(text: str) -> str:
    for para in re.split(r"\n\s*\n", S.strip_html_comments(text).strip()):
        p = para.strip()
        if p and not p.startswith(("```", "|", "#", ">", "-", "*", "<")):
            return p
    return ""


def table_rows(md_text: str) -> list[list[str]]:
    rows = []
    for ln in md_text.splitlines():
        if ln.startswith("|") and not re.match(r"^\|\s*:?-", ln):
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return rows[1:] if rows else []   # drop the header


def inline(s: str) -> str:
    """Inline markdown (bold, code) → HTML, no <p>."""
    h = markdown.markdown(s)
    return re.sub(r"^<p>|</p>$", "", h.strip())


def build(out_dir: Path) -> Path:
    readme = S.read("README.md")
    post = S.read(SECTIONS)
    version = S.read("VERSION").strip()

    # ── hero ──
    pitch = S.region(readme, "pitch").replace("\n", " ").strip()
    # The hero H1 is the problem ("Your model re-reads the same prompt on every
    # request."); the README pitch region is the answer under it.
    hero_lede = inline(pitch)

    # ── install tabs ──
    brew = S.section(readme, "### Homebrew (macOS)")
    mac_code = code(S.fences(brew, "bash")[0])
    mac_note = (md(first_paragraph(brew.split("```")[2]))
                + '<p>Without Homebrew, the release tarball is on the '
                  '<a href="/docs/getting-started/install/">install page</a>.</p>')
    pip_html = md(S.region(readme, "pip-install"))
    docker = S.section(readme, "### Docker")
    docker_code = code(S.fences(docker, "bash")[0])
    docker_note = md(first_paragraph(docker))
    src_region = S.region(readme, "build-from-source")
    src_code = code(S.fences(src_region, "bash")[-1])
    src_note = md(src_region.split("```")[0])

    # ── the four lines ──
    four = S.region(readme, "four-lines")
    four_code = code(S.fences(four, "python")[0])
    four_explained = md(S.region(readme, "four-lines-explained"))

    # ── numbers ──
    two = S.section(post, "## Two numbers, with their denominators")
    rows = table_rows(two)
    numbers_rows = ""
    for r in rows:
        if len(r) < 4:
            continue
        numbers_rows += (f"<tr><td>{inline(r[0])}</td><td class=\"num\">{inline(r[1])}</td>"
                         f"<td class=\"num\">{inline(r[2])}</td><td class=\"x\">{inline(r[3])}</td></tr>\n")
    r64 = [r for r in table_rows(S.region(readme, "two-numbers")) if "64K" in r[0]]
    if r64:
        numbers_rows += (f"<tr><td>{inline(r64[0][0])} — 100% needle recall attending <strong>0.78%</strong> of the prefix</td>"
                         f"<td class=\"num\"></td><td class=\"num\"></td><td class=\"x\">{inline(r64[0][3])}<small>NIAH-class only</small></td></tr>\n")
    numbers_note = md("\n\n".join(p for p in re.split(r"\n\s*\n", two) if not p.startswith("|")), "website/landing")

    # ── where it does not help ──
    nohelp = S.section(post, "## Where this does not help")
    nohelp_lead = first_paragraph(nohelp)
    big = re.search(r"\*\*([0-9.]+×)\*\*", nohelp_lead)
    not_help_big = big.group(1) if big else ""
    not_help_cap = inline(nohelp_lead)
    items = "".join(f"<li>{inline(m.group(1).strip())}</li>\n"
                    for m in re.finditer(r"^- (.+?)(?=^- |\Z)", nohelp, re.S | re.M))
    items = re.sub(r"\n\s+", " ", items)

    # ── what is different ──
    different = md(S.section(readme, "### LLM Memory"))
    omlx = md(S.section(readme, "### On oMLX specifically"))

    # ── tiles: figures from README's at-a-glance table ──
    glance = {r[0]: r[1] for r in table_rows(S.section(readme, "## At a Glance")) if len(r) == 2}
    def g(key):
        return inline(glance.get(key, ""))
    tile_kv = f"{g('Peak KV throughput')} · P99 {g('P99 latency')}"
    tile_vec = f"Recall@100 {g('Recall@100')} · QPS {g('Vector QPS')}"
    tile_dur = g("Persistence")

    # ── footer ──
    license_html = md(S.region(readme, "license-paragraph"))

    tpl = (HERE / "index.template.html").read_text(encoding="utf-8")
    fill = {
        "version": html.escape(version),
        "hero_lede": hero_lede,
        "install_mac_code": mac_code, "install_mac_note": mac_note,
        "install_pip_html": pip_html,
        "install_docker_code": docker_code, "install_docker_note": docker_note,
        "install_src_code": src_code, "install_src_note": src_note,
        "four_lines_code": four_code, "four_lines_explained": four_explained,
        "numbers_rows": numbers_rows, "numbers_note": numbers_note,
        "not_help_big": not_help_big, "not_help_cap": not_help_cap, "not_help_items": items,
        "different_html": different, "omlx_html": omlx,
        "tile_kv_fig": tile_kv, "tile_vec_fig": tile_vec, "tile_dur_fig": tile_dur,
        "license_html": license_html,
        "analytics": S.analytics_beacon(S.analytics_token()),
    }
    page = tpl
    for k, v in fill.items():
        page = page.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{(\w+)\}\}", page)
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    assets = out_dir / "assets"
    assets.mkdir(exist_ok=True)
    for f in (HERE.parent / "assets").iterdir():
        shutil.copy(f, assets / f.name)
    return out_dir / "index.html"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(S.ROOT / "_site"))
    a = ap.parse_args()
    p = build(Path(a.out))
    print(f"wrote {p} ({p.stat().st_size:,} bytes)")
