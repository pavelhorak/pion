#!/usr/bin/env python3
"""mlx-lm seam conformance gate (gh #263).

`install_pion_attention_patch()` monkey-patches a PRIVATE mlx-lm function and
rebinds the snapshot-bound name inside every already-imported
``mlx_lm.models.*`` module. Nothing upstream promises that shape, and it has
already moved once (``sinks``). This test is what makes an upstream break show
up as a red test on a version bump rather than as a TypeError inside somebody's
generation loop.

Deliberately DOWNLOAD-FREE and SERVER-FREE: no model weights, no pion-server,
no network. That is what lets it run once per mlx-lm version in a matrix.
(`tests/test_mlx_lm_patch.py` is the end-to-end sibling and is neither — it
needs a running `--kvcache` server and Llama-3.2-1B cached, so it cannot be
the matrix job. gh #263 assumed both were download-free; only this one is.)

    python3 pion-vllm-mlx/tests/test_mlx_lm_seam.py

Exit 0 = seam intact for the installed mlx-lm, 1 = it moved.
"""
from __future__ import annotations

import inspect
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PASS, FAIL = 0, 0
FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}{(' — ' + detail) if detail else ''}")
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    try:
        import mlx_lm
    except ImportError:
        print("SKIP: mlx-lm not installed (Apple Silicon only).")
        return 0

    version = getattr(mlx_lm, "__version__", "unknown")
    print(f"mlx-lm {version} · python {sys.version.split()[0]}\n")

    from pion_vllm_mlx import _compat
    from pion_vllm_mlx._compat import (
        EXPECTED_SDPA_PARAMS,
        PionMlxCompatError,
        REQUIRED_SYMBOLS,
        _params_compatible,
        check_mlx_lm_seam,
        describe,
    )
    from pion_vllm_mlx.mlx_lm_patch import (
        install_pion_attention_patch,
        pion_scaled_dot_product_attention,
        uninstall_pion_attention_patch,
    )

    print("§1 the declared contract matches our own replacement")
    ours = tuple(inspect.signature(pion_scaled_dot_product_attention).parameters)
    check("_compat.EXPECTED_SDPA_PARAMS == pion_scaled_dot_product_attention's params",
          ours == EXPECTED_SDPA_PARAMS, f"{ours} vs {EXPECTED_SDPA_PARAMS}")

    print("\n§2 every upstream symbol the package reaches into still exists")
    for mod_name, attr in REQUIRED_SYMBOLS:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            check(f"{mod_name}.{attr}", hasattr(mod, attr))
        except ImportError as e:
            check(f"{mod_name}.{attr}", False, str(e))

    print("\n§3 upstream's signature is a prefix of ours")
    import mlx_lm.models.base as base_mod
    up = tuple(inspect.signature(base_mod.scaled_dot_product_attention).parameters)
    check("upstream scaled_dot_product_attention params compatible",
          _params_compatible(up), f"upstream {up}, expected prefix of {EXPECTED_SDPA_PARAMS}")

    print("\n§4 the guard agrees")
    try:
        check_mlx_lm_seam(warn_untested=False)
        check("check_mlx_lm_seam() passes", True)
    except PionMlxCompatError as e:
        check("check_mlx_lm_seam() passes", False, str(e).splitlines()[0])
    rep = describe()
    check("describe() reports no missing symbols", rep["missing_symbols"] == [],
          str(rep["missing_symbols"]))
    check("describe() reports signature_ok", rep["signature_ok"] is True)

    print("\n§5 _params_compatible rejects what it must")
    check("exact match accepted", _params_compatible(EXPECTED_SDPA_PARAMS))
    check("shorter prefix accepted (mlx-lm predating `sinks`)",
          _params_compatible(EXPECTED_SDPA_PARAMS[:-1]))
    check("extra trailing param rejected",
          not _params_compatible(EXPECTED_SDPA_PARAMS + ("scale_factor",)))
    check("reordered params rejected",
          not _params_compatible(("keys", "queries") + EXPECTED_SDPA_PARAMS[2:]))
    check("renamed param rejected",
          not _params_compatible(EXPECTED_SDPA_PARAMS[:-1] + ("sink",)))

    print("\n§6 install/uninstall actually rebind — base module AND a snapshot binder")
    orig = base_mod.scaled_dot_product_attention
    # A model module that did `from .base import scaled_dot_product_attention`
    # holds its OWN reference; the walk over sys.modules is what catches those,
    # and it is the half most likely to rot silently.
    fake = types.ModuleType("mlx_lm.models._pion_seam_probe")
    fake.scaled_dot_product_attention = orig
    sys.modules["mlx_lm.models._pion_seam_probe"] = fake
    try:
        install_pion_attention_patch()
        check("base module rebound",
              base_mod.scaled_dot_product_attention is pion_scaled_dot_product_attention)
        check("snapshot-binding module rebound",
              fake.scaled_dot_product_attention is pion_scaled_dot_product_attention)

        install_pion_attention_patch()   # idempotent
        check("second install is a no-op",
              base_mod.scaled_dot_product_attention is pion_scaled_dot_product_attention)

        uninstall_pion_attention_patch()
        check("base module restored", base_mod.scaled_dot_product_attention is orig)
        check("snapshot-binding module restored",
              fake.scaled_dot_product_attention is orig)
    finally:
        sys.modules.pop("mlx_lm.models._pion_seam_probe", None)
        try:
            uninstall_pion_attention_patch()
        except Exception:
            pass

    print("\n§7 symbols the shipped demos and pion-serve import")
    try:
        from mlx_lm.models.cache import trim_prompt_cache  # noqa: F401
        check("mlx_lm.models.cache.trim_prompt_cache (pion-serve CAG path)", True)
    except ImportError as e:
        check("mlx_lm.models.cache.trim_prompt_cache (pion-serve CAG path)", False, str(e))

    where = []
    for mod_name in ("mlx_lm.generate", "mlx_lm.utils"):
        try:
            mod = __import__(mod_name, fromlist=["generate_step"])
            if hasattr(mod, "generate_step"):
                where.append(mod_name)
        except ImportError:
            pass
    check("generate_step resolvable (examples/prompt_cache_demo.py)", bool(where),
          "found in neither mlx_lm.generate nor mlx_lm.utils")
    if where:
        print(f"        (generate_step lives in {', '.join(where)})")

    print(f"\n{'=' * 60}")
    print(f"mlx-lm {version}: {PASS} passed, {FAIL} failed")
    if FAILURES:
        print("\nThe seam moved:")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nUpdate pion_vllm_mlx/_compat.py and the pyproject ceiling together.")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
