#!/usr/bin/env python3
"""pulse — a daily snapshot of who is finding, installing and talking about Pion.

WHY IT MUST RUN FROM DAY ONE
GitHub keeps repository traffic (views, clones, referrers, popular paths) for
14 days and then it is gone. A launch week that nobody snapshotted cannot be
reconstructed. Run this daily from the moment the repository is public.

WHAT IT COLLECTS (one JSON line per source per run)
  github     stars, forks, watchers; traffic views/clones per day, referrers,
             popular paths (needs push access; works on private repos too);
             release asset download counts; issues/PRs opened by anyone other
             than the owner; stargazers (to count distinct engaged accounts)
  pypi       recent downloads via pypistats.org (mirrors excluded)
  hn         Hacker News stories and comments matching the queries (Algolia)
  backlinks  GitHub issues/PRs and code elsewhere that mention the repo
  reddit, x  NOT collected: both refuse unauthenticated search (Reddit 403s,
             X's search API is paid). Use F5Bot email alerts for Reddit/HN/
             Lobsters and a saved X search. Recorded as such, not guessed.

OUTPUT
  <out>/YYYY-MM.jsonl   append-only log, {"date","ts","source","key","data"}
  <out>/index.html      dashboard regenerated from the whole log. Traffic is
                        merged per day across snapshots, so the history
                        outlives GitHub's 14-day window.

The dashboard's headline pair is the launch kill criterion: strangers who
INSTALLED (release downloads + PyPI) and strangers who ENGAGED (distinct
non-owner accounts that starred, opened an issue/PR, or were counted by a
source above).

USAGE
  python3 scripts/pulse.py                              # snapshot + dashboard
  python3 scripts/pulse.py --repo owner/a --repo owner/b
  python3 scripts/pulse.py --dashboard-only
  python3 scripts/pulse.py --launchd-plist > ~/Library/LaunchAgents/com.pion.pulse.plist
  launchctl load ~/Library/LaunchAgents/com.pion.pulse.plist    # daily 07:30

Needs: Python 3.9+, the `gh` CLI logged in (push access for traffic). No
third-party packages. Nothing here runs inside the Pion server.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import html
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_REPOS = ["pavelhorak/pion"]
DEFAULT_PYPI = ["pion-vllm-mlx"]
DEFAULT_HN_QUERIES = ["pavelhorak/pion", "pion mojo"]
DEFAULT_OUT = os.path.expanduser("~/.pion/pulse")
UA = "pion-pulse/1.0 (+https://github.com/pavelhorak/pion)"


# ── fetching ────────────────────────────────────────────────────────────────
def gh(path: str, paginate: bool = False, accept: str | None = None):
    """GET a GitHub REST path via the gh CLI. Returns parsed JSON or raises."""
    cmd = ["gh", "api", "-X", "GET", path]
    if paginate:
        cmd.insert(2, "--paginate")
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:300])
    text = out.stdout.strip()
    if paginate and text.startswith("[") and "][" in text:
        text = text.replace("][", ",")   # --paginate concatenates arrays
    return json.loads(text) if text else None


def http_json(url: str, retry_429: bool = True):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429 and retry_429:   # pypistats rate-limits bursts
            import time
            time.sleep(20)
            return http_json(url, retry_429=False)
        raise


def guarded(fn, *a, **kw):
    """Run one collector; a failure is recorded, never fatal to the run."""
    try:
        return fn(*a, **kw)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001 — one dead source must not sink the snapshot
        return {"error": str(e)[:300]}


def collect_github(repo: str) -> dict:
    owner = repo.split("/")[0]
    meta = gh(f"repos/{repo}")
    d = {
        "private": meta.get("private"),
        "stars": meta.get("stargazers_count"),
        "forks": meta.get("forks_count"),
        "watchers": meta.get("subscribers_count"),
        "open_issues": meta.get("open_issues_count"),
    }
    d["views"] = guarded(gh, f"repos/{repo}/traffic/views")
    d["clones"] = guarded(gh, f"repos/{repo}/traffic/clones")
    d["referrers"] = guarded(gh, f"repos/{repo}/traffic/popular/referrers")
    d["paths"] = guarded(gh, f"repos/{repo}/traffic/popular/paths")
    rel = guarded(gh, f"repos/{repo}/releases?per_page=100")
    if isinstance(rel, list):
        d["releases"] = [{
            "tag": r.get("tag_name"),
            "assets": {a["name"]: a.get("download_count", 0) for a in r.get("assets", [])},
        } for r in rel]
    else:
        d["releases"] = rel
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    items = guarded(gh, f"repos/{repo}/issues?state=all&since={since}&per_page=100")
    if isinstance(items, list):
        d["community"] = [{
            "number": i["number"],
            "kind": "pr" if "pull_request" in i else "issue",
            "author": i["user"]["login"],
            "created_at": i["created_at"],
            "title": i["title"][:120],
        } for i in items if i["user"]["login"] != owner and not i["user"]["login"].endswith("[bot]")]
    else:
        d["community"] = items
    stars = guarded(gh, f"repos/{repo}/stargazers?per_page=100", paginate=True,
                    accept="application/vnd.github.star+json")
    if isinstance(stars, list):
        d["stargazers"] = [{"login": s["user"]["login"], "at": s.get("starred_at")}
                           for s in stars if s.get("user") and s["user"]["login"] != owner]
    else:
        d["stargazers"] = stars
    return d


def collect_pypi(package: str) -> dict:
    try:
        recent = http_json(f"https://pypistats.org/api/packages/{package}/recent")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"published": False}
        raise
    overall = guarded(http_json, f"https://pypistats.org/api/packages/{package}/overall?mirrors=false")
    total = None
    if isinstance(overall, dict) and "data" in overall:
        total = sum(row.get("downloads", 0) for row in overall["data"])
    return {"published": True, "recent": recent.get("data"), "total_without_mirrors": total}


def collect_hn(query: str) -> dict:
    after = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)).timestamp())
    q = urllib.parse.quote(query)
    res = http_json("https://hn.algolia.com/api/v1/search_by_date?"
                    f"query={q}&tags=(story,comment)&numericFilters=created_at_i>{after}"
                    "&typoTolerance=false&hitsPerPage=100")
    # Algolia still matches loosely (prefixes, stemming): keep a hit only when
    # every query word appears in its text, or "pion mojo" finds unrelated news.
    words = query.lower().split()

    def mentions(h):
        text = " ".join(str(h.get(k) or "") for k in ("title", "story_title", "comment_text", "url", "story_url")).lower()
        return all(w in text for w in words)

    return {"hits": [{
        "id": h.get("objectID"),
        "type": "story" if "story" in h.get("_tags", []) else "comment",
        "title": (h.get("title") or h.get("story_title") or "")[:120],
        "points": h.get("points"),
        "comments": h.get("num_comments"),
        "author": h.get("author"),
        "created_at": h.get("created_at"),
    } for h in res.get("hits", []) if mentions(h)]}


def collect_backlinks(repo: str) -> dict:
    q = urllib.parse.quote(f'"{repo}" -repo:{repo}')
    issues = gh(f"search/issues?q={q}&per_page=50")
    code = guarded(gh, f"search/code?q={urllib.parse.quote(chr(34) + repo + chr(34))}&per_page=50")
    owner = repo.split("/")[0].lower()
    # Strangers only: the owner's own repos and issues are not backlinks.
    found = [{"url": i["html_url"], "title": i["title"][:120], "author": i["user"]["login"]}
             for i in issues.get("items", []) if i["user"]["login"].lower() != owner]
    d = {"issues_total": len(found), "issues": found}
    if isinstance(code, dict) and "items" in code:
        d["code"] = sorted({i["repository"]["full_name"] for i in code["items"]
                            if i["repository"]["owner"]["login"].lower() != owner})
    else:
        d["code"] = code
    return d


# ── storage ─────────────────────────────────────────────────────────────────
def append(out: str, rows: list[dict]) -> str:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, dt.date.today().strftime("%Y-%m") + ".jsonl")
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return path


def load(out: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "*.jsonl"))):
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def snapshot(args) -> list[dict]:
    now = dt.datetime.now(dt.timezone.utc)
    base = {"date": now.strftime("%Y-%m-%d"), "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    rows = []
    for repo in args.repo:
        rows.append({**base, "source": "github", "key": repo, "data": guarded(collect_github, repo)})
        rows.append({**base, "source": "backlinks", "key": repo, "data": guarded(collect_backlinks, repo)})
    for pkg in args.pypi:
        rows.append({**base, "source": "pypi", "key": pkg, "data": guarded(collect_pypi, pkg)})
    for q in args.hn:
        rows.append({**base, "source": "hn", "key": q, "data": guarded(collect_hn, q)})
    for src, why in (("reddit", "unauthenticated search refused (HTTP 403) — use F5Bot alerts"),
                     ("x", "search API is paid — use a saved search")):
        rows.append({**base, "source": src, "key": "-", "data": {"collected": False, "why": why}})
    return rows


# ── dashboard ───────────────────────────────────────────────────────────────
def latest(rows, source, key):
    for r in reversed(rows):
        if r["source"] == source and r["key"] == key:
            return r["data"]
    return None


def merged_traffic(rows, repo, kind):
    """Per-day {count, uniques} across every snapshot — max wins, since a day's
    number only grows while it is inside GitHub's window."""
    days = {}
    for r in rows:
        if r["source"] != "github" or r["key"] != repo:
            continue
        block = (r["data"] or {}).get(kind) or {}
        for d in block.get(kind, []) if isinstance(block, dict) else []:
            day = d["timestamp"][:10]
            prev = days.get(day, {"count": 0, "uniques": 0})
            days[day] = {"count": max(prev["count"], d["count"]), "uniques": max(prev["uniques"], d["uniques"])}
    return dict(sorted(days.items()))


def dashboard(out: str, rows: list[dict], repos, packages, queries) -> str:
    e = html.escape
    parts = []
    installs = 0
    engaged = set()
    for repo in repos:
        g = latest(rows, "github", repo) or {}
        if "error" in g:
            parts.append(f"<h2>{e(repo)}</h2><p class=err>{e(g['error'])}</p>")
            continue
        rel = g.get("releases") if isinstance(g.get("releases"), list) else []
        dl = sum(sum(r["assets"].values()) for r in rel)
        installs += dl
        for s in g.get("stargazers") or [] if isinstance(g.get("stargazers"), list) else []:
            engaged.add(s["login"])
        for c in g.get("community") or [] if isinstance(g.get("community"), list) else []:
            engaged.add(c["author"])
        views, clones = merged_traffic(rows, repo, "views"), merged_traffic(rows, repo, "clones")
        tr = "".join(
            f"<tr><td>{d}</td><td>{views.get(d, {}).get('count', '')}</td><td>{views.get(d, {}).get('uniques', '')}</td>"
            f"<td>{clones.get(d, {}).get('count', '')}</td><td>{clones.get(d, {}).get('uniques', '')}</td></tr>"
            for d in sorted(set(views) | set(clones), reverse=True))
        refs = g.get("referrers") if isinstance(g.get("referrers"), list) else []
        ref_rows = "".join(f"<tr><td>{e(r['referrer'])}</td><td>{r['count']}</td><td>{r['uniques']}</td></tr>" for r in refs)
        comm = g.get("community") if isinstance(g.get("community"), list) else []
        comm_rows = "".join(f"<tr><td>{c['created_at'][:10]}</td><td>{c['kind']} #{c['number']}</td><td>{e(c['author'])}</td><td>{e(c['title'])}</td></tr>" for c in comm)
        rel_rows = "".join(f"<tr><td>{e(r['tag'])}</td><td>{sum(r['assets'].values())}</td></tr>" for r in rel)
        bl = latest(rows, "backlinks", repo) or {}
        bl_rows = "".join(f"<li><a href='{e(i['url'])}'>{e(i['title'])}</a> — {e(i['author'])}</li>" for i in bl.get("issues", []) or [])
        code = bl.get("code") if isinstance(bl.get("code"), list) else []
        parts.append(f"""
<h2>{e(repo)}{' <small>(private)</small>' if g.get('private') else ''}</h2>
<div class=cards><div><b>{g.get('stars', 0)}</b>stars</div><div><b>{g.get('forks', 0)}</b>forks</div>
<div><b>{dl}</b>release downloads</div><div><b>{len(comm)}</b>non-owner issues/PRs (30 d)</div>
<div><b>{bl.get('issues_total', 0) if isinstance(bl.get('issues_total'), int) else 0}</b>backlinks</div></div>
<h3>Traffic by day (merged across snapshots)</h3><table><tr><th>day</th><th>views</th><th>uniq</th><th>clones</th><th>uniq</th></tr>{tr or '<tr><td colspan=5>none yet</td></tr>'}</table>
<h3>Referrers (latest 14 d)</h3><table><tr><th>referrer</th><th>views</th><th>uniq</th></tr>{ref_rows or '<tr><td colspan=3>none</td></tr>'}</table>
<h3>Releases</h3><table><tr><th>tag</th><th>downloads</th></tr>{rel_rows or '<tr><td colspan=2>none</td></tr>'}</table>
<h3>Non-owner issues and PRs (30 d)</h3><table>{comm_rows or '<tr><td>none</td></tr>'}</table>
<h3>Mentions elsewhere on GitHub</h3><ul>{bl_rows or '<li>none</li>'}</ul>
<p>Code referencing the repo: {', '.join(e(c) for c in code) or 'none'}</p>""")
    for pkg in packages:
        p = latest(rows, "pypi", pkg) or {}
        if p.get("published"):
            wk = (p.get("recent") or {}).get("last_week", 0)
            installs += p.get("total_without_mirrors") or 0
            parts.append(f"<h2>PyPI {e(pkg)}</h2><p>last week {wk} · total without mirrors {p.get('total_without_mirrors')}</p>")
        else:
            parts.append(f"<h2>PyPI {e(pkg)}</h2><p>{'not published yet' if p.get('published') is False else e(str(p.get('error', 'no data')))}</p>")
    hn_rows = ""
    for q in queries:
        h = latest(rows, "hn", q) or {}
        for hit in h.get("hits", []) or []:
            hn_rows += (f"<tr><td>{(hit.get('created_at') or '')[:10]}</td><td>{hit['type']}</td>"
                        f"<td><a href='https://news.ycombinator.com/item?id={e(str(hit['id']))}'>{e(hit['title'] or '(comment)')}</a></td>"
                        f"<td>{hit.get('points') or ''}</td><td>{e(hit.get('author') or '')}</td></tr>")
    parts.append(f"<h2>Hacker News (30 d)</h2><table>{hn_rows or '<tr><td>no mentions</td></tr>'}</table>")
    parts.append("<h2>Not collected</h2><ul>"
                 + "".join(f"<li>{e(s)}: {e((latest(rows, s, '-') or {}).get('why', ''))}</li>" for s in ("reddit", "x"))
                 + "</ul>")
    last = rows[-1]["ts"] if rows else "never"
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Pion pulse</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--mute:#666;--line:#ddd;--card:#f5f5f5;--err:#b00020}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121212;--fg:#eee;--mute:#999;--line:#333;--card:#1e1e1e;--err:#ff6b6b}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;width:100%;margin:4px 0 16px}}td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}}
.cards{{display:flex;flex-wrap:wrap;gap:8px}}.cards div{{background:var(--card);padding:8px 12px;border-radius:8px;min-width:120px}}
.cards b{{display:block;font-size:22px}}.err{{color:var(--err)}}a{{color:inherit}}small{{color:var(--mute)}}
</style></head><body>
<h1>Pion pulse</h1><p><small>last snapshot {e(last)} · {len(rows)} records</small></p>
<div class=cards><div><b>{installs}</b>installed (release downloads + PyPI; GitHub counts your own downloads too)</div>
<div><b>{len(engaged)}</b>engaged strangers (distinct non-owner accounts: stars, issues, PRs)</div></div>
<p><small>Day-10 kill criterion: ≥3 engaged → Beat 1 · 1–2 → rework the framing · 0 → the pitch, not the timing.</small></p>
{''.join(parts)}
</body></html>"""
    path = os.path.join(out, "index.html")
    with open(path, "w") as f:
        f.write(page)
    return path


def launchd_plist() -> str:
    script = os.path.abspath(__file__)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.pion.pulse</string>
  <key>ProgramArguments</key><array><string>/usr/bin/env</string><string>python3</string><string>{script}</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
  <key>StandardOutPath</key><string>{DEFAULT_OUT}/launchd.log</string>
  <key>StandardErrorPath</key><string>{DEFAULT_OUT}/launchd.log</string>
</dict></plist>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", action="append", help=f"owner/name (repeatable; default {DEFAULT_REPOS})")
    ap.add_argument("--pypi", action="append", help=f"package (repeatable; default {DEFAULT_PYPI})")
    ap.add_argument("--hn", action="append", help=f"HN query (repeatable; default {DEFAULT_HN_QUERIES})")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dashboard-only", action="store_true")
    ap.add_argument("--launchd-plist", action="store_true", help="print a daily launchd agent and exit")
    args = ap.parse_args()
    if args.launchd_plist:
        print(launchd_plist(), end="")
        return 0
    args.repo = args.repo or DEFAULT_REPOS
    args.pypi = args.pypi or DEFAULT_PYPI
    args.hn = args.hn or DEFAULT_HN_QUERIES
    if not args.dashboard_only:
        rows = snapshot(args)
        path = append(args.out, rows)
        errors = [f"{r['source']}:{r['key']}: {r['data']['error']}" for r in rows
                  if isinstance(r["data"], dict) and "error" in r["data"]]
        print(f"appended {len(rows)} records to {path}")
        for err in errors:
            print(f"  source failed — {err}", file=sys.stderr)
    page = dashboard(args.out, load(args.out), args.repo, args.pypi, args.hn)
    print(f"dashboard {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
