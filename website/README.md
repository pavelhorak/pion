# The website

`https://pion.pavelhorak.com/` — a one-page landing at `/` and the docs at
`/docs/`, built from this repository on every push to `main` by
`.github/workflows/pages.yml` and deployed to GitHub Pages from the public repo.

**There is no second copy of anything.** The site is derived from files that
already have to be right:

| On the site | Comes from |
|---|---|
| Docs pages | `doc/*.md`, rendered by MkDocs (Material) |
| Overview (docs home) | `doc/index.md` as the public export sees it |
| Install, first five minutes, security, licence | `README.md` snippet regions (`<!-- --8<-- [start:name] -->`) and headings |
| CLI flags | the `--help` printer in `src/main.mojo` |
| Command index | `tools/gen_command_table.py`, the same derivation as `src/commands/command_table.mojo` |
| Pion-native command pages | the wire-protocol sections of `doc/shared_kv_cache.md`, `doc/command_matrix.md`, `doc/vector_engine.md`, `doc/ai_gateway.md` |
| Python API | docstrings, via mkdocstrings |
| Package pages, Contributing, Security, CLA, Changelog | each package's `README.md` and the root files |
| Landing page | the same README regions, the launch post's two honesty sections, `VERSION` |

Pages that are *assembled* live in `website/pages/` and contain
`<!-- include-section: FILE | HEADING -->`, `<!-- include-region: FILE | NAME -->`
or `<!-- include-file: FILE -->` directives, expanded by `website/gen_pages.py`
at build time. Links in included text are rewritten so they resolve from the
page they land on; links that leave the docs tree become GitHub links.

## Build and preview

```bash
uv venv .venv-site && uv pip install --python .venv-site/bin/python -r website/requirements.txt
export DISABLE_MKDOCS_2_WARNING=true
.venv-site/bin/mkdocs serve                              # docs at http://127.0.0.1:8000/docs/
.venv-site/bin/mkdocs build --strict                     # → _site/docs/
.venv-site/bin/python website/landing/build_landing.py   # → _site/index.html
.venv-site/bin/python tests/test_docs_site.py --built _site
```

`--strict` is the contract: a dead link, a nav entry without a file, a file
without a nav entry, or a README region that went missing fails the build.
`tests/test_docs_site.py` checks what the build cannot — that the generated
pages agree with their sources and that the landing page carries the numbers.
Both run in CI on every PR.

## The mark

The database drum in the landing hero and the favicon (`assets/pion-icon.svg`,
navy that turns light under a dark browser theme) and the whole-logo icons —
the 30 px brand mark and the docs header (`assets/pion-mark.svg`, white for the
navy header; antennas at 75% length so the drum still reads at 30 px) — share
one drum, **traced with potrace from the original logo artwork**
(`site/assets/pion-icon-light.png`, kept outside git), not redrawn by hand.
An earlier hand-drawn recreation did not survive comparison with the PNG, so
a new size or variant should be traced the same way, not drawn. The hero's
node positions and connector curves were measured from
`site/assets/light-logo.png` the same way. The link-preview image
`assets/og.png` is rendered from the landing's own hero, headline and pitch
sentence: `python3 website/landing/build_og.py` (needs Google Chrome).

## Where things are

```
mkdocs.yml                     nav, theme, plugins; docs_dir is doc/
website/
  sitelib.py                   extraction helpers (regions, sections, --help, command table)
  gen_pages.py                 mkdocs-gen-files script: generated + assembled pages
  hooks.py                     rewrites links that leave doc/ to GitHub URLs
  extra.css                    the visual system on top of Material
  pages/                       assembled page sources (include directives)
  landing/index.template.html  the landing page
  landing/build_landing.py     fills the template from README / the launch post
  assets/                      logo, mark
```

## Rules

- **No number on the site that does not divide out of a published row.** The
  landing takes its numbers from the README and the launch post; edit those.
- **Name + category everywhere**: "Pion — memory engine for AI inference" in
  every title and OG tag.
- **The honesty panel is a section, not fine print.** It is included verbatim
  from the launch post.
- Nothing generated is committed. `_site/` and `.venv-site/` are ignored.
