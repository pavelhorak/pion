"""Single-source helpers for the site build.

Everything the site says comes from a file that already has to be right —
README.md, doc/*.md, src/main.mojo's --help printer, the generated command
table, the package READMEs. These helpers pull text out of those files so the
docs (website/gen_pages.py) and the landing page (website/landing/build_landing.py)
include it rather than restate it. Nothing here is imported by the server.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc"
GITHUB = "https://github.com/pavelhorak/pion"
BLOB = GITHUB + "/blob/main/"
SITE = "https://pion.pavelhorak.com"


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def doc_text(name: str) -> str:
    """A doc/ file as the PUBLIC tree sees it.

    index.md is replaced at export by a curated public version under
    publication/public/doc/. When that directory exists (the
    private tree) read the replacement, so the site built here is the site
    that ships. In the export tree the directory is absent and doc/ already
    holds the public version."""
    pub = ROOT / "publication" / "public" / "doc" / name
    if pub.exists():
        return pub.read_text(encoding="utf-8")
    return (DOC / name).read_text(encoding="utf-8")


def region(text: str, name: str) -> str:
    """Text between `<!-- --8<-- [start:name] -->` and `<!-- --8<-- [end:name] -->`."""
    pat = (r"<!--\s*--8<--\s*\[start:" + re.escape(name) + r"\]\s*-->\n?(.*?)"
           r"<!--\s*--8<--\s*\[end:" + re.escape(name) + r"\]\s*-->")
    m = re.search(pat, text, re.S)
    if not m:
        raise KeyError(f"region {name!r} not found")
    return m.group(1).strip("\n") + "\n"


def section(text: str, heading: str, include_heading: bool = False) -> str:
    """Body of the markdown section whose heading line is exactly `heading`
    (e.g. '## Quick Start'), up to the next heading of the same or higher
    level. Headings inside code fences are ignored."""
    lines = text.splitlines(keepends=True)
    level = len(heading) - len(heading.lstrip("#"))
    start = None
    end = len(lines)
    in_fence = False
    for i, ln in enumerate(lines):
        if ln.startswith("```"):
            in_fence = not in_fence
        if in_fence:
            continue
        if start is None:
            if ln.rstrip() == heading:
                start = i
            continue
        m = re.match(r"^(#{1,6})\s", ln)
        if m and len(m.group(1)) <= level:
            end = i
            break
    if start is None:
        raise KeyError(f"section {heading!r} not found")
    body = lines[start + (0 if include_heading else 1):end]
    return "".join(body).strip("\n") + "\n"


def fences(md: str, lang: str | None = None) -> list[str]:
    """Fenced code blocks in `md`, optionally only those tagged `lang`."""
    out = []
    for m in re.finditer(r"```([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", md, re.S):
        if lang is None or m.group(1) == lang:
            out.append(m.group(2))
    return out


def strip_html_comments(md: str) -> str:
    return re.sub(r"<!--.*?-->\n?", "", md, flags=re.S)


def shift_headings(md: str, by: int) -> str:
    """Demote every heading outside code fences by `by` levels."""
    out, in_fence = [], False
    for ln in md.splitlines(keepends=True):
        if ln.startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else re.match(r"^(#{1,6})(\s.*)$", ln)
        if m:
            ln = "#" * max(1, min(6, len(m.group(1)) + by)) + m.group(2) + "\n"
        out.append(ln)
    return "".join(out)


_LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")


def relink(md: str, src_dir: str) -> str:
    """Rewrite relative links in text that came from `src_dir` (a repo-relative
    directory: '' for README.md, 'doc' for doc/*.md) so they still resolve
    once the text is rendered as a docs page somewhere else. Targets inside
    doc/ become docs-absolute (`/x.md`, which mkdocs resolves relative to the
    docs dir); targets elsewhere in the repository become GitHub links."""
    def fix(m):
        target = m.group(2)
        if re.match(r"^(https?:|mailto:|#|/)", target):
            return m.group(0)
        path, _, frag = target.partition("#")
        if not path:
            return m.group(0)
        p = (ROOT / src_dir / path).resolve()
        try:
            rel = p.relative_to(ROOT).as_posix()
        except ValueError:
            return m.group(0)
        if rel.startswith("doc/") and rel.endswith(".md"):
            new = "/" + rel[4:]
        else:
            new = BLOB + rel
        if frag:
            new += "#" + frag
        return m.group(1) + new + m.group(3)
    return _LINK.sub(fix, md)


# ── CLI flags, from the --help printer in src/main.mojo ─────────────────────

def cli_help_groups() -> list[dict]:
    """Parse `_print_help()` in src/main.mojo into groups of flags.

    The printer is the flag reference the binary itself prints, so a flag
    exists on the page exactly when it exists in the help — that is the whole
    reason this page is generated rather than typed."""
    src = read("src/main.mojo")
    m = re.search(r"def _print_help\(\):\n(.*?)\n(?=def |\S)", src, re.S)
    if not m:
        raise RuntimeError("_print_help() not found in src/main.mojo")
    body = m.group(1)
    lines = [ln for ln in re.findall(r'print\("((?:[^"\\]|\\.)*)"\)', body)]
    lines = [ln.replace('\\"', '"') for ln in lines]
    groups: list[dict] = []
    cur = None
    notes_before: list[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        if not ln.startswith(" "):
            if ln.startswith(("Usage:", "Wire-compatible", "More:", "doc/")):
                notes_before.append(ln)
                continue
            cur = {"name": ln.strip(), "flags": [], "notes": []}
            groups.append(cur)
            continue
        if cur is None:
            notes_before.append(ln.strip())
            continue
        # `  -p, --port N   desc` — the ARGUMENT is one space after the flag and
        # UPPERCASE; the description follows after two or more spaces (so a
        # description that starts with AF_XDP is not mistaken for an argument).
        # Continuation lines are indented 20+ spaces and may mention a flag; the
        # indent bound keeps them out.
        fm = re.match(r"^\s{2,8}((?:-[A-Za-z],\s+)?--[\w-]+(?:\s*/\s*--[\w-]+)?(?: [A-Z][A-Z0-9_]*)?)\s+(\S.*)$", ln)
        if fm and ln.lstrip().startswith("-"):
            cur["flags"].append({"spec": fm.group(1).strip(), "desc": fm.group(2).strip()})
        elif cur["flags"] and ln.startswith(" " * 20):
            cur["flags"][-1]["desc"] += " " + ln.strip()
        else:
            cur["notes"].append(ln.strip())
    return groups


def cli_flag_count() -> int:
    return sum(len(g["flags"]) for g in cli_help_groups())


# ── The command surface, from the generated command table ───────────────────

def command_names() -> list[str]:
    """Every wire command the dispatch chains accept, lowercase, from the same
    derivation that generates src/commands/command_table.mojo."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_command_table", ROOT / "tools" / "gen_command_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    names, problems = mod.collect()
    if problems:
        raise RuntimeError(f"gen_command_table.collect() reported {len(problems)} unresolved arm(s)")
    return sorted(names)


def redis_arity() -> dict[str, int]:
    out = {}
    for ln in read("tools/redis_arity.txt").splitlines():
        if ln.strip():
            n, a = ln.split()
            out[n] = int(a)
    return out


def redis_write_set() -> set[str]:
    return {ln.strip() for ln in read("tools/redis_write.txt").splitlines() if ln.strip()}


def command_table_count() -> int:
    m = re.search(r"comptime PION_COMMAND_COUNT = (\d+)", read("src/commands/command_table.mojo"))
    return int(m.group(1)) if m else -1


def matrix_sections() -> list[tuple[str, str, set[str]]]:
    """(heading text, slug, {command names mentioned in the section's tables})
    for every `##`/`###` section of doc/command_matrix.md."""
    from markdown.extensions.toc import slugify
    text = read("doc/command_matrix.md")
    out = []
    cur_title, cur_names = None, set()
    for ln in text.splitlines():
        h = re.match(r"^(##|###)\s+(.*)$", ln)
        if h:
            if cur_title is not None:
                out.append((cur_title, slugify(re.sub(r"`", "", cur_title), "-"), cur_names))
            cur_title, cur_names = h.group(2).strip(), set()
            continue
        if ln.startswith("|") and cur_title is not None:
            first = ln.split("|")[1]
            for tok in re.findall(r"[A-Z][A-Z0-9_.]*(?:\s[A-Z][A-Z0-9_]*)?", first):
                cur_names.add(tok.strip().lower())
                cur_names.add(tok.split()[0].lower())
    if cur_title is not None:
        out.append((cur_title, slugify(re.sub(r"`", "", cur_title), "-"), cur_names))
    return out


def analytics_token() -> str:
    """The Cloudflare Web Analytics site token (mkdocs.yml `extra`), or ""."""
    m = re.search(r'^\s+cloudflare_web_analytics_token:\s*"([^"]*)"', read("mkdocs.yml"), re.M)
    return m.group(1) if m else ""


def analytics_beacon(token: str) -> str:
    """Cloudflare's cookieless beacon, or "" without a token, so a fork's build
    of the site reports to no one. One definition serves the docs (hooks.py)
    and the landing page (build_landing.py)."""
    if not token:
        return ""
    return ('<script defer src="https://static.cloudflareinsights.com/beacon.min.js" '
            f"data-cf-beacon='{{\"token\": \"{token}\"}}'></script>")
