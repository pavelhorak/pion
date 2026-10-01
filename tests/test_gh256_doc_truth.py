#!/usr/bin/env python3
"""gh #256 — the repo must not assert things about itself that are false.

Each of these is what a skeptical evaluator greps for first, and each was wrong:

  1. CONTRIBUTING.md linked `.github/workflows/ci.yml` and said "the CI gate
     covers PR builds". That workflow was deleted in `3ec2ff5` and `.github/`
     does not exist, so a first-time contributor's PR ran NOTHING while the doc
     said otherwise. Not restored: hosted-runner CI on this private repo is
     both unpaid-for and payment-blocked, so the local gate IS the gate.
  2. CONTRIBUTING.md said "a CLA applies from the first external PR". No CLA
     exists; CLA-vs-DCO is still undecided.
  3. `pion-langgraph`/`-autogen`/`-llamaindex` pyprojects declared MIT under a
     BSL 1.1 repo. Two MORE were found the same way (`pion-vllm-mlx`,
     `pion-exo`) whose own READMEs already said BSL.
  4. doc/persistence.md described a single 256 MB ring with three cmd ids —
     the pre-gh #149 design. The WAL rotates segments and has 27 cmd ids.
  5. CLAUDE.md said LTRIM "declines in quicklist mode" and that LPOS/LINSERT/
     LREM return plausible-but-wrong values there. All five were implemented on
     2026-08-20 and are verified working here against a live server.
  6. doc/command_matrix.md said MULTI has "no per-connection tx state".

The point of a truth-sync is that it STAYS true, so these are assertions rather
than a one-time edit. Claims 5 and 6 are checked against the running server, not
against other prose — a doc agreeing with a doc proves nothing.

Usage: python3 tests/test_gh256_doc_truth.py [--port 1974]
"""
import argparse, os, re, socket, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def read(rel):
    try:
        with open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


class R:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=15)
        self.f = self.s.makefile("rb")

    def __call__(self, *a):
        p = [f"*{len(a)}\r\n".encode()]
        for x in a:
            x = x if isinstance(x, bytes) else str(x).encode()
            p.append(b"$%d\r\n%s\r\n" % (len(x), x))
        self.s.sendall(b"".join(p)); return self._r()

    def _r(self):
        line = self.f.readline()
        t, b = line[:1], line[1:-2]
        if t == b":":
            return int(b)
        if t in b"+-":
            return line[:-2]
        if t == b"$":
            n = int(b); return None if n == -1 else self.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(b); return [] if n <= 0 else [self._r() for _ in range(n)]
        return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()

    print("[1] CONTRIBUTING.md does not claim CI that does not exist")
    c = read("CONTRIBUTING.md")
    has_gh = os.path.isdir(os.path.join(ROOT, ".github", "workflows"))
    # Forbid a markdown LINK to the workflow, not the string itself: the
    # corrected text mentions the path in the sentence explaining that it was
    # deleted, which is exactly the honesty being asked for.
    linked = re.search(r"\]\(\.github/workflows", c) is not None
    check(".github/workflows exists, or CONTRIBUTING doesn't LINK a workflow",
          has_gh or not linked,
          "CONTRIBUTING links a workflow file that is not in the repo")
    check("CONTRIBUTING doesn't claim a CI gate covers PRs",
          has_gh or "CI gate covers PR builds" not in c)
    check("CONTRIBUTING states the local gate is the gate",
          has_gh or "there is no ci" in c.lower(),
          "the doc should say plainly that nothing runs on a PR")

    print("\n[2] CONTRIBUTING.md does not claim a CLA that does not exist")
    cla_files = [p for p in ("CLA.md", "CLA.txt", ".github/CLA.md")
                 if os.path.exists(os.path.join(ROOT, p))]
    check("no CLA is asserted unless one exists",
          bool(cla_files) or "a CLA applies from the first" not in c,
          "CONTRIBUTING says a CLA applies; no CLA file exists")

    print("\n[3] Package licence metadata agrees with the package's own README")
    bad = []
    for name in sorted(os.listdir(ROOT)):
        pj = os.path.join(ROOT, name, "pyproject.toml")
        if not os.path.isfile(pj):
            continue
        lic = re.search(r'license\s*=\s*\{\s*text\s*=\s*"([^"]+)"', read(f"{name}/pyproject.toml"))
        if not lic:
            continue
        rd = read(f"{name}/README.md")
        # The package's OWN licence is the first one its "## License" section
        # names. Since the 2026-09-19 licence split the Apache-2.0 satellites
        # explain in that same section that the SERVER is BSL 1.1, so a
        # whole-file "mentions BSL" test flagged every one of them (red on main
        # from #300 until 2026-09-21, unnoticed because this file is not in CI).
        m = re.search(r"^##+\s*Licen[cs]e\b(.*?)(?=^##\s|\Z)", rd, re.S | re.M)
        sect = m.group(1) if m else rd
        hits = [(sect.find(t), t) for t in ("Apache", "Business Source", "BSL 1.1", "BUSL")
                if t in sect]
        if not hits:
            continue
        readme_says = "Apache" if min(hits)[1] == "Apache" else "BUSL"
        pyproject_says = "Apache" if "Apache" in lic.group(1) else "BUSL"
        if readme_says != pyproject_says:
            bad.append(f"{name}: pyproject={lic.group(1)} but its README licence section leads with {readme_says}")
    check("no package contradicts its own README's licence", not bad, "; ".join(bad))

    print("\n[3b] Licence surfaces state the D11 licences, not the retired BSL")
    # D11 (2026-09-24): Apache-2.0 engine + the closed libpion_vector under its
    # own binary licence. BSL 1.1 was the licence of record for five months and
    # was written into ~35 files; these are the ones a reader or a scanner
    # consults for "what licence is this". Dated history is deliberately
    # not listed. Files absent from a tree (the public export) are skipped.
    surfaces = ["LICENSE", "README.md", "CONTRIBUTING.md", "CLA.md", "CHANGELOG.md",
                "NOTICE", "mkdocs.yml", "doc/licensing.md", "doc/index.md",
                "doc/pion_one_pager.md", "doc/blog/2026-10-pion-memory-engine.md",
                "publication/public/doc/index.md", "scripts/package_release.sh",
                ".github/SUPPORT.md", ".github/workflows/cla.yml",
                "vendor/pion-vector/README.md", "pion-serve/README.md"]
    for name in sorted(os.listdir(ROOT)):
        if os.path.isfile(os.path.join(ROOT, name, "pyproject.toml")) and \
           os.path.isfile(os.path.join(ROOT, name, "README.md")):
            surfaces.append(f"{name}/README.md")
    stale = [f for f in surfaces if os.path.isfile(os.path.join(ROOT, f))
             and any(t in read(f) for t in ("BSL 1.1", "Business Source", "Change Date"))]
    check("no licence surface still says BSL", not stale, ", ".join(stale))
    check("root LICENSE is Apache-2.0", read("LICENSE").lstrip().startswith("Apache License"))
    vl = os.path.join(ROOT, "vendor/pion-vector/LICENSE")
    check("the binary licence text exists",
          os.path.isfile(vl) and read("vendor/pion-vector/LICENSE").startswith("Pion Vector Binary Licence"))
    check("the vendor README no longer says the text is owed",
          "Owed before the public release" not in read("vendor/pion-vector/README.md"))
    check("README links the binary licence", "vendor/pion-vector/LICENSE" in read("README.md"))

    print("\n[4] doc/persistence.md is not the pre-gh#149 design")
    pdoc = read("doc/persistence.md")
    check("does not claim a single 256 MB ring buffer",
          "256 MB mmap ring buffer" not in pdoc)
    check("does not claim only 3 cmd ids",
          "1=SET (key+val), 2=DEL (key only), 3=HSET" not in pdoc)
    check("mentions segment rotation", "seal" in pdoc.lower() and "--wal-max-segments" in pdoc)
    check("mentions the blob tier pointer record", "blob" in pdoc.lower())

    print("\n[5] The quicklist claims match the SERVER, not other prose")
    try:
        r = R(args.port)
    except OSError as e:
        print(f"FATAL: no Pion on {args.port} ({e})")
        return 2
    r("DEL", "gh256:q")
    for i in range(1500):                     # past the 1024 ziplist threshold
        r("RPUSH", "gh256:q", f"e{i}")
    implemented = {
        "LPOS":    r("LPOS", "gh256:q", "e500") == 500,
        # LINSERT returns the new length; -1 is the "not supported here" answer
        # this issue is about. `or True` would have made this assertion vacuous.
        "LINSERT": r("LINSERT", "gh256:q", "BEFORE", "e500", "X") == 1501,
        "LREM":    r("LREM", "gh256:q", "1", "e501") == 1,
        "LSET":    r("LSET", "gh256:q", "0", "first") == b"+OK",
    }
    r("LTRIM", "gh256:q", "0", "9")
    implemented["LTRIM"] = r("LLEN", "gh256:q") == 10
    r("DEL", "gh256:q")
    for name, ok in implemented.items():
        check(f"{name} works in quicklist mode", ok)
    cl = read("CLAUDE.md")
    check("CLAUDE.md does not still say LTRIM declines",
          "LTRIM now declines in quicklist mode" not in cl,
          "doc contradicts the server, which just did it")

    print("\n[6] MULTI really has per-connection tx state")
    a, b = R(args.port), R(args.port)
    a("MULTI"); a("SET", "gh256:tx", "queued")
    check("a queued command has NOT applied on another connection",
          b("GET", "gh256:tx") is None, "MULTI is not queueing per connection")
    a("EXEC")
    check("EXEC applies it", b("GET", "gh256:tx") == b"queued")
    b("DEL", "gh256:tx")
    cm = read("doc/command_matrix.md")
    check("command_matrix.md does not claim 'no per-connection tx state'",
          "no per-connection tx state" not in cm)

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
