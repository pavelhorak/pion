#!/bin/bash
# Install VectorDBBench where the vector harness runs it, patched for Pion.
#
#   pixi run install-vdbbench
#
# vectordb-benchmark.py starts it with `pixi run vectordbbench`, which runs
# venv_zvec/bin/vectordbbench on every platform. Stock VectorDBBench's Redis
# client has an empty optimize(), so it never sends FT.OPTIMIZE: Pion never
# builds the index, and the run reports recall 0.0 at an impossible QPS. The
# patch below makes optimize() send it, as the gate's own setup always has.
#
# The version is pinned to the one the Mac gate runs (1.0.22, checked
# 2026-10-03). VectorDBBench needs Python 3.11 or 3.12; 3.14 breaks it.
# Safe to re-run: an existing venv is kept and only checked and patched.
set -euo pipefail
cd "$(dirname "$0")/../.."

VENV="${VDBBENCH_VENV:-venv_zvec}"
PKGS=('vectordb-bench[redis]==1.0.22' 'redis==4.6.0')

if [ ! -x "$VENV/bin/vectordbbench" ]; then
    PY="${VDBBENCH_PYTHON:-$(command -v python3.12 || command -v python3.11 || true)}"
    if [ -n "$PY" ]; then
        echo "creating $VENV with $("$PY" --version 2>&1)"
        "$PY" -m venv "$VENV"
        "$VENV/bin/pip" install -q "${PKGS[@]}"
    elif command -v uv >/dev/null 2>&1; then
        echo "creating $VENV with uv (Python 3.12)"
        uv venv -q -p 3.12 "$VENV"
        uv pip install -q --python "$VENV/bin/python" "${PKGS[@]}"
    else
        echo "error: need python3.12, python3.11 or uv (https://docs.astral.sh/uv/) to create $VENV" >&2
        exit 1
    fi
fi

"$VENV/bin/python" - <<'PY'
import glob
import sys

paths = glob.glob(sys.prefix + "/lib/python3*/site-packages/vectordb_bench/backend/clients/redis/redis.py")
if not paths:
    sys.exit("error: vectordb_bench's Redis client is not in this venv")
old = "def optimize(self, data_size: int | None = None):\n        pass"
new = ("def optimize(self, data_size: int | None = None):\n"
       "        assert self.conn is not None\n"
       '        self.conn.execute_command("FT.OPTIMIZE", "index")')
for p in paths:
    text = open(p).read()
    if 'execute_command("FT.OPTIMIZE", "index")' in text:
        print("optimize() already sends FT.OPTIMIZE:", p)
    elif old in text:
        open(p, "w").write(text.replace(old, new))
        print("patched optimize() to send FT.OPTIMIZE:", p)
    else:
        sys.exit("error: optimize() no longer matches the patch; this VectorDBBench "
                 "version changed it. Make it send FT.OPTIMIZE by hand: " + p)
PY
echo "ready: $VENV/bin/vectordbbench"
