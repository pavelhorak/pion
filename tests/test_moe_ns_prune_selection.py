#!/usr/bin/env python3
"""Model-free unit test of the per-distribution prune-set selection logic (gh #61).

The ΔPPL headline in per_dist_hist_namespaces.py needs an MLX MoE model. The
*mechanism* it relies on — that a distribution-aware prune set spares experts a
single-class (naive) set would catastrophically prune — is pure logic and is
verified here on synthetic per-class histograms. This is the runnable proof that
namespaces fix the +96.67 catastrophe at the selection layer.

Run:
  python3 tests/test_moe_ns_prune_selection.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The four functions under test are pure logic, but they live in the private
# research tree (`experiments/` is stripped from the public export). Skip
# explicitly rather than letting the ImportError escape: the substrate gate
# does classify a raw ModuleNotFoundError as ENV_SKIP, but only by matching the
# text of a traceback, and a test that means "skip" should say so itself.
sys.path.insert(0, str(REPO / 'experiments' / 'moe_expert_phase0'))
try:
    from per_dist_hist_namespaces import (  # noqa: E402
        bottom_frac_per_layer, naive_prune_set, maxpool_prune_set, union_safe_prune_set,
    )
except ModuleNotFoundError:
    print("SKIP: per_dist_hist_namespaces is in the private research tree "
          "(experiments/moe_expert_phase0/), which the public export strips.")
    sys.exit(0)


def main() -> int:
    failures: list[str] = []
    L, E, frac = 1, 4, 0.25   # 1 layer, 4 experts, prune bottom 1 (k = max(1, 4*0.25))

    # english hammers experts 0,1; code hammers 2,3. Under english-only HIST,
    # expert 3 looks unused (count 0) and gets pruned — but it is the hottest
    # expert for code. That is exactly the narrow-HIST catastrophe.
    english = {(0, 0): 100, (0, 1): 80, (0, 2): 5, (0, 3): 0}
    code    = {(0, 0): 0,   (0, 1): 3,  (0, 2): 90, (0, 3): 120}
    per_class = {'english': english, 'code': code}

    naive = naive_prune_set(english, L, E, frac)          # bottom-1 by english
    maxpool = maxpool_prune_set(per_class, L, E, frac)     # bottom-1 by max-over-class
    unionsafe = union_safe_prune_set(per_class, L, E, frac)

    # naive prunes expert 3 (english count 0) — the code-critical expert.
    if (0, 3) not in naive:
        failures.append(f'naive should prune code-critical expert (0,3): {naive}')
    # maxpool ranks by max(english,code): expert 3 = max(0,120)=120 (safe);
    # the true global-least is expert with lowest max. english=100/80/5/0,
    # code=0/3/90/120 → max=100/80/90/120 → least is (0,1)=80. So maxpool
    # prunes (0,1), NOT the code-critical (0,3).
    if (0, 3) in maxpool:
        failures.append(f'maxpool must NOT prune code-critical (0,3): {maxpool}')
    if maxpool != {(0, 1)}:
        failures.append(f'maxpool expected {{(0,1)}}, got {maxpool}')
    # union-safe: bottom-1(english)={(0,3)}, bottom-1(code)={(0,0)}; intersection
    # is empty — no expert is unused in BOTH classes, so nothing is pruned.
    if unionsafe != set():
        failures.append(f'union-safe expected empty, got {unionsafe}')

    # Same budget: naive and maxpool both prune exactly k=1 per layer.
    if len(naive) != 1 or len(maxpool) != 1:
        failures.append(f'naive/maxpool budget: {len(naive)}, {len(maxpool)}')
    # union-safe never prunes more than maxpool (it is an intersection).
    if not unionsafe.issubset(maxpool | naive):
        failures.append('union-safe should be a subset of the pooled candidates')

    # bottom_frac_per_layer sanity: k respects frac and ties break by eid.
    tie = {(0, 0): 5, (0, 1): 5, (0, 2): 5, (0, 3): 5}
    if bottom_frac_per_layer(tie, 1, 4, 0.5) != {(0, 0), (0, 1)}:
        failures.append('tie-break should pick lowest expert ids')

    # Multi-layer independence: selection is per-layer.
    ml = {(0, 0): 0, (0, 1): 9, (1, 0): 9, (1, 1): 0}
    got = bottom_frac_per_layer(ml, 2, 2, 0.5)
    if got != {(0, 0), (1, 1)}:
        failures.append(f'per-layer selection wrong: {got}')

    if failures:
        print('FAIL:')
        for f in failures:
            print('  -', f)
        return 1
    print('PASS: per-distribution prune-set selection (naive/maxpool/union-safe)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
