"""mlx-lm seam compatibility guard (gh #263).

`install_pion_attention_patch()` replaces an *internal* mlx-lm function and
rebinds the snapshot-bound name inside every already-imported
``mlx_lm.models.*`` module. That is a private seam with no stability promise,
and it has already moved once (``sinks`` was added to the signature). Without a
check, an upstream release that changes the signature surfaces as a ``TypeError``
somewhere inside generation — far from the cause, and after the patch has
already been installed.

Two checks, deliberately weighted differently:

* **The signature is the contract, so a mismatch is a hard error.** It is the
  thing that actually breaks.
* **The version range is a proxy, so being outside it is a warning** (below the
  floor is still an error — those releases lack symbols we need). Refusing to
  run on a newer mlx-lm whose seam is still intact would be a false failure, and
  the pyproject ceiling already stops pip from resolving an untested one by
  default. What must never happen is a *silent* run on an untested version.

Tested: mlx-lm 0.20.1, 0.22.5, 0.24.1, 0.28.4, 0.29.1, 0.31.3, 0.32.0 (0.32.0 is
the latest on PyPI as of 2026-10-04) via
``pion-vllm-mlx/tests/run_mlx_version_matrix.sh``.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Any, Optional

# Mirrors pion-vllm-mlx/pyproject.toml's `mlx` extra. Change both together.
MIN_MLX_LM = "0.20.1"   # 0.20.0 was never published to PyPI
MAX_TESTED_MLX_LM = "0.32.0"
CEILING_MLX_LM = "0.33"          # exclusive, as in `mlx-lm>=0.20.1,<0.33`

#: Parameter names of ``mlx_lm.models.base.scaled_dot_product_attention`` as of
#: the tested ceiling, in order. ``sinks`` is the recent addition.
EXPECTED_SDPA_PARAMS = (
    "queries", "keys", "values", "cache", "scale", "mask", "sinks",
)

#: Every upstream symbol this package reaches into at run time. ``generate_step``
#: is deliberately absent — the package itself does not use it (only the demo
#: does), and it moved module between tested versions.
REQUIRED_SYMBOLS = (
    ("mlx_lm.models.base", "scaled_dot_product_attention"),
    ("mlx_lm.models.cache", "make_prompt_cache"),
)


class PionMlxCompatError(RuntimeError):
    """mlx-lm is present but its internals are not the shape the patch needs.

    Deliberately NOT an ``ImportError``: ``pion_vllm_mlx/__init__.py`` catches
    ImportError to make the patch optional, and an incompatibility swallowed
    there would resurface as a confusing missing-name error instead of this
    message.
    """


def _parse(version: str) -> tuple:
    parts = []
    for chunk in version.split(".")[:3]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def installed_mlx_lm_version() -> Optional[str]:
    """The installed mlx-lm version, or None if mlx-lm is not importable."""
    try:
        import mlx_lm
    except Exception:
        return None
    return getattr(mlx_lm, "__version__", None) or "unknown"


def _sdpa_params() -> Optional[tuple]:
    try:
        import mlx_lm.models.base as base_mod
        fn = base_mod.scaled_dot_product_attention
        return tuple(inspect.signature(fn).parameters)
    except Exception:
        return None


def describe() -> dict:
    """A report of the seam as it currently stands. Never raises.

    Useful in bug reports: ``python -c "import pion_vllm_mlx._compat as c;
    print(c.describe())"``.
    """
    version = installed_mlx_lm_version()
    params = _sdpa_params()
    missing = []
    for mod_name, attr in REQUIRED_SYMBOLS:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            if not hasattr(mod, attr):
                missing.append(f"{mod_name}.{attr}")
        except Exception:
            missing.append(f"{mod_name} (not importable)")
    return {
        "mlx_lm_version": version,
        "tested_range": f">={MIN_MLX_LM},<{CEILING_MLX_LM}",
        "max_tested": MAX_TESTED_MLX_LM,
        "sdpa_params": params,
        "expected_sdpa_params": EXPECTED_SDPA_PARAMS,
        "missing_symbols": missing,
        "signature_ok": params is not None and _params_compatible(params),
    }


def _params_compatible(params: tuple) -> bool:
    """Upstream's parameter list must be a PREFIX of what we accept.

    A prefix — not equality — because the replacement declares every parameter
    the ceiling version has, with defaults on the optional tail. An older
    mlx-lm that predates ``sinks`` calls us with six arguments and that is
    fine. Upstream growing a parameter we do not declare is NOT fine: it would
    be passed positionally or by keyword into a function that cannot take it.
    """
    if len(params) > len(EXPECTED_SDPA_PARAMS):
        return False
    return tuple(params) == EXPECTED_SDPA_PARAMS[:len(params)]


def check_mlx_lm_seam(*, warn_untested: bool = True) -> dict:
    """Validate the monkey-patch seam. Raises PionMlxCompatError on mismatch.

    Called by ``install_pion_attention_patch()`` BEFORE anything is patched, so
    a failure leaves mlx-lm untouched.
    """
    report = describe()
    version = report["mlx_lm_version"]

    if version is None:
        raise PionMlxCompatError(
            "mlx-lm is not installed. The Pion attention patch needs it:\n"
            "    pip install 'pion-vllm-mlx[mlx]'\n"
            "(Apple Silicon only — mlx has no wheels for other platforms.)"
        )

    if report["missing_symbols"]:
        raise PionMlxCompatError(
            "mlx-lm %s is missing internals the Pion attention patch needs: %s\n"
            "Tested range is mlx-lm %s (ceiling %s). Pin a tested version:\n"
            "    pip install 'mlx-lm<%s'"
            % (version, ", ".join(report["missing_symbols"]),
               report["tested_range"], MAX_TESTED_MLX_LM, CEILING_MLX_LM)
        )

    params = report["sdpa_params"]
    if not _params_compatible(params):
        raise PionMlxCompatError(
            "mlx-lm %s changed the signature the Pion attention patch replaces.\n"
            "  mlx_lm.models.base.scaled_dot_product_attention%s\n"
            "  the patch provides                             (%s)\n"
            "Nothing has been patched. Pin a tested version:\n"
            "    pip install 'mlx-lm>=%s,<%s'\n"
            "and please report the new signature at "
            "https://github.com/pavelhorak/pion/issues"
            % (version, "(" + ", ".join(params) + ")",
               ", ".join(EXPECTED_SDPA_PARAMS), MIN_MLX_LM, CEILING_MLX_LM)
        )

    if _parse(version) < _parse(MIN_MLX_LM):
        raise PionMlxCompatError(
            "mlx-lm %s is below the supported floor %s.\n"
            "    pip install 'mlx-lm>=%s,<%s'"
            % (version, MIN_MLX_LM, MIN_MLX_LM, CEILING_MLX_LM)
        )

    if warn_untested and _parse(version) >= _parse(CEILING_MLX_LM):
        warnings.warn(
            "mlx-lm %s is newer than the tested ceiling %s. The attention-patch "
            "signature still matches, so this is very likely fine — but it is "
            "untested. If generation misbehaves, pin 'mlx-lm<%s' and report it."
            % (version, MAX_TESTED_MLX_LM, CEILING_MLX_LM),
            RuntimeWarning,
            stacklevel=3,
        )

    return report


# ── Cache slot contents ───────────────────────────────────────────────────────
#
# Up to 0.31, a cache slot's ``state`` was its contents: an ArraysCache's list of
# arrays, or a KVCache's (keys, values) sliced to the tokens it holds. The
# scalars (offset, window, quantization) lived in a separate ``meta_state``.
# mlx-lm 0.32 removed ``meta_state`` and folded those scalars into ``state``, so
# ``state`` now has a different length for every cache class, and a KVCache's
# ``state`` returns its step-padded buffers. Reading ``state`` by position, or
# assigning it a list, therefore breaks on 0.32: loudly on a length mismatch,
# silently on a padded buffer. These two functions go through attributes that
# every supported version has, so the SSM.PREFIX blobs built from them are the
# same whichever mlx-lm wrote them.


def slot_arrays(c) -> list:
    """The arrays one cache slot holds, laid out as mlx-lm <=0.31's ``state``.

    ArraysCache / MambaCache: its list of arrays (entries may be None).
    KVCache / RotatingKVCache: ``[keys, values]``, sliced to the tokens held.
    """
    if isinstance(getattr(c, "cache", None), list):
        return list(c.cache)
    if hasattr(c, "keys_and_values"):       # mlx-lm >= 0.32
        return list(c.keys_and_values())
    return list(c.state)                    # mlx-lm <= 0.31: already sliced


def set_slot_arrays(c, arrays, offset: Optional[int] = None) -> None:
    """Load ``arrays``, laid out as ``slot_arrays`` returns them, into a fresh slot.

    A KV slot then holds ``offset`` tokens, by default as many as the keys
    carry. A RotatingKVCache also gets its write index: once its window is
    full, its next token is written there, and from index 0 it would overwrite
    a token the window keeps. Restoring a rotating cache that has already
    wrapped is out of scope: two arrays cannot say where the window starts.
    The slot takes the arrays themselves, not copies, as assigning ``state``
    did.
    """
    if isinstance(getattr(c, "cache", None), list):
        c.cache = list(arrays)
        return
    keys, values = arrays
    c.keys, c.values = keys, values
    n = int(keys.shape[2])
    c.offset = n if offset is None else int(offset)
    if hasattr(c, "_idx"):
        c._idx = n
