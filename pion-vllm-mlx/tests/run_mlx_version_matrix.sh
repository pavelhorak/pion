#!/usr/bin/env bash
# mlx-lm version matrix for the attention-patch seam (gh #263).
#
# Builds a throwaway venv per mlx-lm version, installs that version, and runs
# the two seam gates in it:
#
#   pion-vllm-mlx/tests/test_mlx_lm_seam.py            (needs mlx-lm, no model)
#   pion-vllm-mlx/tests/test_gemma4_12b_layer_routing.py  (pure python)
#
# Both are download-free in the model sense — no weights, no pion-server, no
# network beyond pip. `tests/test_mlx_lm_patch.py` is NOT in the matrix: it
# needs a running `--kvcache` server and Llama-3.2-1B cached, so it is the
# end-to-end gate you run once by hand, not once per version.
#
# Apple Silicon only (mlx ships no wheels elsewhere). Each venv is ~400 MB and
# is deleted as soon as its run finishes, so peak disk is one venv, not N.
#
# Usage:
#   pion-vllm-mlx/tests/run_mlx_version_matrix.sh                  # default matrix
#   pion-vllm-mlx/tests/run_mlx_version_matrix.sh 0.29.1 0.31.3    # explicit versions
#   PY=python3.12 pion-vllm-mlx/tests/run_mlx_version_matrix.sh    # pick interpreter
#
# Exit 0 only if every version in the matrix passed both gates.

set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root

PY="${PY:-python3.12}"
# Floor, three points in between (0.31.3 was the ceiling until pion-vllm-mlx
# 0.1.4), and the tested ceiling. Keep the last entry in step with
# pion-vllm-mlx/pyproject.toml and pion_vllm_mlx/_compat.py.
DEFAULT_VERSIONS=(0.20.1 0.24.1 0.28.4 0.31.3 0.32.0)
VERSIONS=("${@:-}")
[ -z "${VERSIONS[0]:-}" ] && VERSIONS=("${DEFAULT_VERSIONS[@]}")

command -v "$PY" >/dev/null || { echo "FATAL: $PY not found (set PY=)"; exit 1; }
[ "$(uname -s)" = "Darwin" ] || { echo "SKIP: mlx is Apple Silicon only"; exit 0; }

WORK="$(mktemp -d -t pion-mlx-matrix)"
trap 'rm -rf "$WORK"' EXIT

PASSED=(); FAILED=(); SKIPPED=()

for V in "${VERSIONS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  mlx-lm $V  ($PY)"
    echo "════════════════════════════════════════════════════════════"
    VENV="$WORK/v$V"
    "$PY" -m venv "$VENV" || { SKIPPED+=("$V (venv)"); continue; }

    if ! "$VENV/bin/pip" install -q --disable-pip-version-check \
            "mlx-lm==$V" numpy 2>"$WORK/pip.err"; then
        echo "  INSTALL FAILED — no wheel for this interpreter, or a resolver conflict:"
        sed 's/^/    /' "$WORK/pip.err" | tail -4
        SKIPPED+=("$V (install)")
        rm -rf "$VENV"
        continue
    fi

    ACTUAL="$("$VENV/bin/python" -c 'import mlx_lm;print(getattr(mlx_lm,"__version__","?"))' 2>/dev/null)"
    echo "  installed: mlx-lm $ACTUAL"

    OK=1
    "$VENV/bin/python" pion-vllm-mlx/tests/test_mlx_lm_seam.py || OK=0
    echo ""
    "$VENV/bin/python" pion-vllm-mlx/tests/test_gemma4_12b_layer_routing.py >/dev/null 2>&1 \
        && echo "  PASS  gemma4 12B layer routing (pure python)" \
        || { echo "  FAIL  gemma4 12B layer routing"; OK=0; }

    if [ "$OK" = "1" ]; then PASSED+=("$V"); else FAILED+=("$V"); fi
    rm -rf "$VENV"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  MATRIX SUMMARY  ($PY)"
echo "════════════════════════════════════════════════════════════"
[ ${#PASSED[@]}  -gt 0 ] && echo "  passed:  ${PASSED[*]}"
[ ${#FAILED[@]}  -gt 0 ] && echo "  FAILED:  ${FAILED[*]}"
[ ${#SKIPPED[@]} -gt 0 ] && echo "  skipped: ${SKIPPED[*]}"
echo "════════════════════════════════════════════════════════════"

# A skip is not a pass. If nothing ran, say so rather than exiting green.
[ ${#PASSED[@]} -eq 0 ] && { echo "NOTHING RAN — treat as a failure."; exit 1; }
[ ${#FAILED[@]} -gt 0 ] && exit 1
exit 0
