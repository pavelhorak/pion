#!/usr/bin/env python3
"""The Mojo packages under packages/ build from src/ and pass their tests.

packages/build.py assembles pion_resp, pion_slab and pion_simd from the files
the server compiles, precompiles each, and runs its tests. Running it here
means a change to one of those src/ files that breaks a package (a new import
of a module the package does not carry, an API the package exports going away)
fails the gate instead of the next release's recipe build.

    python3 tests/test_mojo_packages.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
r = subprocess.run([sys.executable, str(ROOT / "packages" / "build.py")], cwd=ROOT)
sys.exit(r.returncode)
