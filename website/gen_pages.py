"""Virtual docs pages, generated at build time (mkdocs-gen-files).

Nothing this script produces is ever committed: the reference pages are
re-derived from the artefacts that already have to be right (the --help
printer, the command table, README.md, the package READMEs, the root files),
so a page here cannot drift from the code without the build noticing.

Two kinds of page:
  * generated outright — CLI flags (src/main.mojo), the command index
    (tools/gen_command_table.py), the theme assets;
  * assembled — every file under website/pages/ is copied into the docs tree
    after its `<!-- include-...: -->` directives are expanded:
        <!-- include-section: README.md | ### Docker -->
        <!-- include-region:  README.md | security -->
        <!-- include-file:    CONTRIBUTING.md | strip-h1 -->
    Options after the source: strip-h1, shift:N (demote headings), indent:N
    (for content under a tab), raw (keep HTML comments). Links in the included text are rewritten so they
    resolve from the page they land on.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import mkdocs_gen_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sitelib as S  # noqa: E402

PAGES = S.ROOT / "website" / "pages"
GENERATED_NOTE = ('\n\n<p class="pion-source">Generated at build time from <code>{src}</code> — '
                  'edit the source, not this page.</p>\n')


def write(path: str, content: str, edit_path: str | None = None) -> None:
    with mkdocs_gen_files.open(path, "w") as f:
        f.write(content)
    if edit_path:
        mkdocs_gen_files.set_edit_path(path, edit_path)


# ── 1. theme assets ─────────────────────────────────────────────────────────
for _asset in ("pion-mark.svg", "pion-icon.svg"):
    with mkdocs_gen_files.open(f"assets/{_asset}", "wb") as f:
        f.write((S.ROOT / "website" / "assets" / _asset).read_bytes())
with mkdocs_gen_files.open("assets/extra.css", "w") as f:
    f.write(S.read("website/extra.css"))

# ── 2. the doc file the export replaces with a curated public version ──
for _name in ("index.md",):
    _pub = S.ROOT / "publication" / "public" / "doc" / _name
    if _pub.exists():
        write(_name, S.relink(_pub.read_text(encoding="utf-8"), "doc"), _name)

# ── 3. assembled pages from website/pages/ ──────────────────────────────────
_INC = re.compile(r"<!--\s*include-(section|region|file):\s*(.+?)\s*-->")


def _source_text(src: str) -> str:
    if src == "doc/index.md":
        return S.doc_text(src[4:])
    return S.read(src)


def _expand(m: re.Match) -> str:
    kind, arg = m.group(1), m.group(2)
    parts = [p.strip() for p in arg.split("|")]
    src = parts[0]
    text = _source_text(src)
    src_dir = Path(src).parent.as_posix()
    src_dir = "" if src_dir == "." else src_dir
    if kind == "section":
        body, opts = S.section(text, parts[1]), parts[2:]
    elif kind == "region":
        body, opts = S.region(text, parts[1]), parts[2:]
    else:
        body, opts = text, parts[1:]
    if "raw" not in opts:
        body = S.strip_html_comments(body)
    for o in opts:
        if o == "strip-h1":
            body = re.sub(r"^\s*# [^\n]*\n+", "", body, count=1)
        elif o.startswith("shift:"):
            body = S.shift_headings(body, int(o[6:]))
    body = S.relink(body, src_dir).rstrip("\n") + "\n"
    for o in opts:
        if o.startswith("indent:"):   # for content under a `=== "tab"`
            pad = " " * int(o[7:])
            body = "".join((pad + ln if ln.strip() else ln) for ln in body.splitlines(keepends=True))
    return body


for _src in sorted(PAGES.rglob("*.md")):
    _rel = _src.relative_to(PAGES).as_posix()
    _text = _src.read_text(encoding="utf-8")
    _out = _INC.sub(_expand, _text)
    write(_rel, _out, f"../website/pages/{_rel}")

# ── 4. CLI flags, from the binary's own --help printer ──────────────────────

def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


_groups = S.cli_help_groups()
_n_flags = sum(len(g["flags"]) for g in _groups)
_md = ["# CLI flags", "",
       f"Every flag `pion-server` accepts — **{_n_flags}** of them, in the groups "
       "`pion-server --help` prints. This page is generated from the help printer in "
       "`src/main.mojo` at build time, so a flag is documented here exactly when the "
       "binary knows it. Defaults and profiles are explained in "
       "[Configuration and profiles](/configuration.md); the security-relevant flags "
       "(`--bind`, `--requirepass*`, `--tenant`) in the [security model](/concepts/security-model.md).",
       "",
       "```", "pion-server [flags]", "```", ""]
for g in _groups:
    _md += [f"## {g['name']}", ""]
    if g["flags"]:
        _md += ['<div class="pion-cli" markdown>', "", "| Flag | What it does |", "|---|---|"]
        _md += [f"| `{_cell(f['spec'])}` | {_cell(f['desc'])} |" for f in g["flags"]]
        _md += ["", "</div>", ""]
    for note in g["notes"]:
        _md += [f"*{_cell(note)}*", ""]
_md.append(GENERATED_NOTE.format(src="src/main.mojo · _print_help()"))
write("reference/cli-flags.md", "\n".join(_md), "../src/main.mojo")

# ── 5. the command index, from the generated command table ─────────────────
_FAMILY_PAGE = {
    "KV.PREFIX": "kv-prefix", "KV": "kv-prefix", "V": "v-store", "STATE": "v-store",
    "ATTEND": "attend", "ATTEND.PREFIX": "attend", "SSM.PREFIX": "ssm-prefix",
    "MOE.EXPERT": "moe-expert", "FT": "vector", "VSET": "vector",
    "AI": "ai", "AI.ROUTE": "ai", "AI.SEMANTIC_CACHE": "ai", "AI.FLARE": "ai", "AI.MEMORY": "ai",
    "AI.KNN_LM": "rag-knn-pkm", "RAG": "rag-knn-pkm", "RAG.SPECULATE": "rag-knn-pkm",
    "NEURON.PKM": "rag-knn-pkm", "PION": "stats",
}
_VSET = {"vadd", "vsim", "vcard", "vdim", "vinfo", "vismember", "vsetattr", "vgetattr", "vemb",
         "vrandmember", "vrem", "vrange", "vlinks"}


def _family(n: str) -> str:
    if n in _VSET:
        return "VSET"
    return n.rsplit(".", 1)[0].upper() if "." in n else "OTHER"


_names = S.command_names()
_arity = S.redis_arity()
_writes = S.redis_write_set()
_secs = S.matrix_sections()


def _doc_link(n: str) -> str:
    for title, slug, ns in _secs:
        if n in ns:
            short = re.sub(r"^\d+[a-z]?\.\s*", "", re.sub(r"[`*]", "", title))
            short = short.split(" — ")[0].split(" (")[0][:48]
            return f"[{short}](/command_matrix.md#{slug})"
    return "—"


_redis = [n for n in _names if n in _arity]
_native = [n for n in _names if n not in _arity]
_count = S.command_table_count()
_md = ["# Command index", "",
       f"All **{len(_names)}** commands the server dispatches, derived at build time from the same "
       "source as `src/commands/command_table.mojo` (the table `MULTI` uses to reject unknown "
       "commands at queue time). A command that exists in the engine appears here; a command that "
       "appears here exists in the engine. The per-command semantics, return shapes and known "
       "divergences from Redis are in the [Redis-compatible command matrix](/command_matrix.md); the "
       "Pion-native families each have a [reference page](/reference/commands/index.md).",
       "",
       f"| | count |", "|---|---:|",
       f"| Redis-compatible (a Redis 8 command of the same name exists) | {len(_redis)} |",
       f"| Pion-native | {len(_native)} |",
       f"| Total (`PION_COMMAND_COUNT`) | {_count} |", "",
       "## Pion-native", "",
       "| Command | Family | Reference | Documented in the matrix |", "|---|---|---|---|"]
for n in _native:
    fam = _family(n)
    page = _FAMILY_PAGE.get(fam)
    ref = f"[{fam}](/reference/commands/{page}.md)" if page else fam
    _md.append(f"| `{n.upper()}` | {fam} | {ref} | {_doc_link(n)} |")
_md += ["", "## Redis-compatible", "",
        "Arity is Redis's own (negative = *at least* that many); *writes* marks commands Redis "
        "flags as write commands, which is what decides WAL logging and `WATCH` bumps.", "",
        "| Command | Arity | Writes | Documented in the matrix |", "|---|---:|:---:|---|"]
for n in _redis:
    _md.append(f"| `{n.upper()}` | {_arity[n]} | {'✓' if n in _writes else ''} | {_doc_link(n)} |")
_md.append(GENERATED_NOTE.format(src="tools/gen_command_table.py · doc/command_matrix.md"))
write("reference/command-index.md", "\n".join(_md), "../tools/gen_command_table.py")
