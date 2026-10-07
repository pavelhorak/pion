#!/usr/bin/env python3
"""Every prompt the long-context harnesses build holds exactly one <bos>, first.

`tok.encode(text)` adds special tokens: Llama 3 prepends <|begin_of_text|>,
and tests/_gemma4_text_filter_load.py turns Gemma 4's <bos> on. The harnesses
built prompts from separately encoded pieces, so every seam carried a <bos>:
the 64K NIAH prompt held 397, and neither vanilla mlx-lm nor Pion found the
needle. tests/_prompt_ids.py is the fix; this test runs each harness's own
prompt builder against the real tokenizers.

It canaries itself first: a plain two-piece prompt must show two <bos>, and
`one_bos` must refuse it. A tokenizer that stopped prepending would otherwise
pass every check here without testing anything.

Tokenizers only (no model weights are loaded), so it takes seconds.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pion-vllm-mlx"))
sys.path.insert(0, str(REPO / "tests"))

from _gemma4_text_filter_load import _resolve_snapshot, load_tokenizer_only  # noqa: E402
from _prompt_ids import bos, bos_count, one_bos, piece  # noqa: E402
from mlx_lm.tokenizer_utils import load as load_tokenizer  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def well_formed(tok, ids, suffix_ids=None) -> tuple[bool, str]:
    n = bos_count(tok, ids)
    if n != 1 or ids[0] != bos(tok)[0]:
        return False, f"{n} <bos> tokens, first id {ids[0]}"
    if suffix_ids is not None and list(ids[-len(suffix_ids):]) != list(suffix_ids):
        return False, "the warm suffix is not the prompt's tail, so the prefix split is wrong"
    return True, ""


gemma = load_tokenizer_only("mlx-community/gemma-4-e2b-it-4bit")
llama = load_tokenizer(_resolve_snapshot("mlx-community/Llama-3.2-1B-Instruct-4bit"))

print("[1] the tokenizers prepend <bos> on a plain encode (canary)")
for name, tok in (("gemma-4 (loader)", gemma), ("llama-3.2", llama)):
    naive = tok.encode("Some filler text.") + tok.encode("Question: what?")
    check(f"{name}: two plain pieces carry two <bos>", bos_count(tok, naive) == 2,
          f"got {bos_count(tok, naive)}")
    try:
        one_bos(tok, naive)
        refused = False
    except ValueError:
        refused = True
    check(f"{name}: one_bos refuses that prompt", refused)
    check(f"{name}: a piece carries none", bos_count(tok, piece(tok, "Question: what?")) == 0)

print("[2] each harness builds a prompt with one <bos>, at position 0")
niah = load_module("niah", REPO / "tests" / "test_long_context_niah.py")
ids, _ = niah.build_prompt(niah.Trial(length=4096, depth=0.5, city="Bratislava", number=65125), gemma)
q = ("\n\nQuestion: What is the magic number for the city of Bratislava? "
     "Answer with only the number.\nAnswer:")
ok, why = well_formed(gemma, ids, piece(gemma, q))
check("test_long_context_niah.build_prompt", ok, why)

multi = load_module("niah_multi", REPO / "tests" / "test_long_context_niah_multi.py")
t = multi.Trial(length=4096, needles=[("Bratislava", 11111), ("Trondheim", 22222), ("Asmara", 33333)],
                target_idx=1, depths=[0.25, 0.5, 0.75])
ids, _ = multi.build_prompt(t, gemma)
q = ("\n\nQuestion: What is the magic number for the city of Trondheim? "
     "Answer with only the number.\nAnswer:")
ok, why = well_formed(gemma, ids, piece(gemma, q))
check("test_long_context_niah_multi.build_prompt", ok, why)

ruler = load_module("ruler", REPO / "tests" / "test_ruler_subset.py")
mv = ruler.MVTrial(length=4096, city="Vilnius", values=[12345, 23456, 34567], depths=[0.25, 0.5, 0.75])
ids, _ = ruler.build_mv_prompt(mv, gemma)
ok, why = well_formed(gemma, ids, piece(gemma, ruler.mv_suffix(mv)))
check("test_ruler_subset.build_mv_prompt", ok, why)
vt = ruler.VTTrial(length=4096, var_names=["X1", "X2", "X3"], root_value=54321, depths=[0.25, 0.5, 0.75])
ids, _ = ruler.build_vt_prompt(vt, gemma)
ok, why = well_formed(gemma, ids, piece(gemma, ruler.vt_suffix(vt)))
check("test_ruler_subset.build_vt_prompt", ok, why)

example = load_module("niah_example", REPO / "examples" / "sparse_mask_64k_niah.py")
ids, q_ids = example.build_prompt(4096, 0.5, "Bratislava", 65125, gemma)
ok, why = well_formed(gemma, ids, q_ids)
check("examples/sparse_mask_64k_niah.build_prompt", ok, why)

# bench_w1_stage2 builds its requests with test_kv_prefix_workload's request_ids.
w1 = load_module("w1_stage2", REPO / "tests" / "bench_w1_stage2.py")
for name, tok in (("llama-3.2", llama), ("gemma-4 (loader)", gemma)):
    sys_ids = tok.encode(w1.system_prompt(0, 2))
    q = w1.USER_QUERIES[0]
    ok, why = well_formed(tok, w1.request_ids(tok, sys_ids, q), piece(tok, q))
    check(f"test_kv_prefix_workload / bench_w1_stage2 request_ids ({name})", ok, why)

print()
print("PASS — 0 failure(s)" if not failures else f"FAIL — {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
