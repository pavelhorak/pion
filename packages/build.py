#!/usr/bin/env python3
"""Build Pion's reusable Mojo packages from src/ and run their tests.

Each packages/<name>/package.toml names the src/ files a package is made of.
This script copies them into build/mojo-packages/src/<name>/, rewrites Pion's
absolute imports (`from src.common.ptr import ...`) to package-relative ones
(`from .ptr import ...`), writes an `__init__.mojo` from the manifest's
exports, precompiles the package with `mojo precompile`, and runs every
packages/<name>/tests/test_*.mojo against it.

The packages have no source of their own: src/ is the only copy, so a package
cannot drift from the server it came from. An import of a src/ module the
manifest does not list fails the build, rather than producing a package that
compiles here and not elsewhere.

    python3 packages/build.py                 # every package
    python3 packages/build.py pion_resp       # one
    python3 packages/build.py --mojo /path/to/mojo
    python3 packages/build.py pion_resp --assemble-only   # sources only (the recipes use this)

Needs the Mojo toolchain: `pixi run mojo` from the repository, or --mojo.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKGS = ROOT / "packages"
OUT = ROOT / "build" / "mojo-packages"
SRC_IMPORT = re.compile(r"^(\s*from\s+)src\.(?:\w+\.)*(\w+)(\s+import\b)", re.M)


def find_mojo(explicit: str | None) -> list[str]:
    """The env's mojo binary cannot find `std` unless pixi activated the
    environment, so a checkout with a pixi env goes through `pixi run`."""
    if explicit:
        return [explicit]
    if (ROOT / ".pixi" / "envs" / "default").exists() and shutil.which("pixi"):
        return ["pixi", "run", "mojo"]
    if shutil.which("mojo"):
        return ["mojo"]
    return ["pixi", "run", "mojo"]


def assemble(name: str) -> Path:
    meta = tomllib.loads((PKGS / name / "package.toml").read_text(encoding="utf-8"))
    dest = OUT / "src" / name
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    members = {Path(s["as"]).stem for s in meta["sources"]}
    for s in meta["sources"]:
        text = (ROOT / s["from"]).read_text(encoding="utf-8")

        def rewrite(m: re.Match) -> str:
            module = m.group(2)
            if module not in members:
                raise SystemExit(f"{name}: {s['from']} imports src module {module!r}, "
                                 f"which packages/{name}/package.toml does not list")
            return f"{m.group(1)}.{module}{m.group(3)}"

        text = SRC_IMPORT.sub(rewrite, text)
        if "from src." in text or "import src." in text:
            raise SystemExit(f"{name}: {s['from']} still imports from src/ after the rewrite")
        (dest / s["as"]).write_text(
            f"# Assembled by packages/build.py from {s['from']} (Pion {version()}); edit that file, not this one.\n"
            + text, encoding="utf-8")
    (dest / "__init__.mojo").write_text(
        f'"""{meta["summary"]}.\n\nBuilt from Pion {version()} (https://github.com/pavelhorak/pion), Apache-2.0."""\n\n'
        + "\n".join(meta["exports"]) + "\n", encoding="utf-8")
    return dest


def version() -> str:
    return (ROOT / "VERSION").read_text().strip()


def run(cmd: list[str]) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="packages to build (default: all)")
    ap.add_argument("--mojo", help="the mojo executable (default: the repo's pixi env)")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--assemble-only", action="store_true",
                    help="write build/mojo-packages/src/<name>/ and stop (a recipe precompiles it itself)")
    a = ap.parse_args()
    names = a.names or sorted(p.name for p in PKGS.iterdir() if (p / "package.toml").exists())
    mojo = find_mojo(a.mojo)
    failed = 0
    for name in names:
        src = assemble(name)
        if a.assemble_only:
            print(f"assembled {name} -> {src.relative_to(ROOT)}")
            continue
        pkg = OUT / f"{name}.mojoc"
        rc, out = run([*mojo, "precompile", str(src), "-o", str(pkg)])
        ok = rc == 0 and pkg.exists()
        print(f"{'PASS' if ok else 'FAIL'}  {name}: precompile -> {pkg.relative_to(ROOT)}")
        if not ok:
            print(out[-4000:])
            failed += 1
            continue
        if a.no_tests:
            continue
        for test in sorted((PKGS / name / "tests").glob("test_*.mojo")):
            rc, out = run([*mojo, "run", "-I", str(OUT), str(test)])
            print(f"{'PASS' if rc == 0 else 'FAIL'}  {name}: {test.relative_to(ROOT)}")
            if rc != 0:
                print(out[-4000:])
                failed += 1
    print(f"\n{'ALL PASS' if not failed else f'{failed} FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
