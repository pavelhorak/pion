#!/usr/bin/env python3
"""Render website/assets/og.png (1280 x 630, the link-preview image) from the landing page.

It reuses the landing template's own styles, its hero diagram (the Pion mark
traced from the original logo) and headline, and the README pitch sentence, so
the preview cannot drift from the page it advertises. Needs Google Chrome for
the screenshot; without it, it writes the HTML and says where.

    python3 website/landing/build_og.py            # -> website/assets/og.png
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sitelib as S  # noqa: E402

CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium")


def block(text: str, start: str, end: str) -> str:
    a = text.index(start)
    return text[a:text.index(end, a) + len(end)]


def main() -> int:
    tpl = (HERE / "index.template.html").read_text(encoding="utf-8")
    fonts = block(tpl, '<link rel="stylesheet" href="https://fonts.googleapis.com', ">")
    style = block(tpl, "<style>", "</style>")
    defs = block(tpl, '<svg width="0" height="0"', "</svg>")
    hero = block(tpl, '<svg class="fabric"', "</svg>")
    m = re.search(r"<h1>(.*?)</h1>", tpl, re.S)
    if m is None:
        raise SystemExit("landing template has no <h1>")
    h1 = m.group(1)
    pitch = S.region((S.ROOT / "README.md").read_text(encoding="utf-8"), "pitch")
    first = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", " ".join(pitch.split()).split(". ")[0] + ".")

    page = f"""<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8">{fonts}{style}
<style>
html,body{{width:1280px;height:630px;margin:0;overflow:hidden}}
.og{{display:grid;grid-template-columns:600px 1fr;gap:24px;align-items:center;height:630px;padding:0 48px 0 64px;box-sizing:border-box}}
.og .eyebrow{{font-size:15px;margin-bottom:14px}}
.og h1{{font-family:var(--font-display);font-weight:600;font-size:58px;line-height:1.02;letter-spacing:-.022em;color:var(--ink);margin:0 0 20px}}
.og p{{font-size:22px;line-height:1.42;color:var(--ink-2);margin:0}}
.og p strong{{color:var(--ink)}}
.og .fabric{{max-width:none;width:100%}}
</style></head><body>{defs}
<div class="og"><div><p class="eyebrow">Pion · memory engine for AI inference</p><h1>{h1}</h1><p>{first}</p></div>
{hero}</div></body></html>"""
    tmp = Path(tempfile.mkdtemp(prefix="pion_og_"))
    src = tmp / "og.html"
    src.write_text(page, encoding="utf-8")
    chrome = next((c for c in CHROME if shutil.which(c) or Path(c).exists()), None)
    if chrome is None:
        print(f"no Chrome found; open {src} and screenshot it at 1280x630")
        return 1
    out = HERE.parent / "assets" / "og.png"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1280,630",
                    "--virtual-time-budget=4000", f"--screenshot={out}", src.as_uri()],
                   check=True, stderr=subprocess.DEVNULL)
    print(f"wrote {out.relative_to(S.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
