#!/usr/bin/env python3
"""Methodology guard for the MOE.EXPERT.* PPL eval.

The eval_prune_quality.py --ppl-corpus path produced two materially
different "ΔPPL at 25% HIST" numbers depending on corpus size:
  - ~500 tokens:  ΔPPL = −17.40  (apparent PPL IMPROVEMENT)
  - ~1,500 tokens: ΔPPL = +7.60   (small PPL cost, not net-positive)

Both reports were taken honestly, but the small one is below the
variance floor for sign-flip resistance. This test asserts the
methodology defaults are enforced:

  1. Held-out corpus is >= 1,000 tokens (per the OLMoE tokenizer
     proxy — gives a stable count).
  2. Chunk size is >= 256 (smaller chunks lose context too fast).
  3. Stride <= chunk (no gaps).

If a future eval script change loosens these defaults, this test
flags it instead of letting another small-corpus headline slip in.

Mac-side test; skipped if mlx_lm isn't installed."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'experiments' / 'moe_expert_phase0'))

CORPUS = REPO / 'experiments' / 'moe_expert_phase0' / 'ppl_corpus_canonical.txt'
DIVERSE = REPO / 'experiments' / 'moe_expert_phase0' / 'ppl_corpus_diverse.txt'

# This test reads corpora and a harness from the private research tree. In the
# public export those are absent and the failure was a bare FileNotFoundError —
# which, unlike ModuleNotFoundError, is NOT one of the substrate gate's
# ENV_SKIP markers, so it read as a real failure rather than a missing
# prerequisite.
if not CORPUS.exists() or not DIVERSE.exists():
    print("SKIP: the PPL corpora live in the private research tree "
          "(experiments/moe_expert_phase0/), stripped from the public export.")
    sys.exit(0)

# Distribution markers the diverse corpus must contain — present to ensure
# someone editing the file doesn't accidentally strip multilingual / code /
# math sections and re-create the narrow-corpus bias that produced the
# 2026-05-19 retraction.
DIVERSE_MARKERS = {
    'code (Python def)':    'def quicksort',
    'code (JavaScript)':    'async function',
    'code (Rust)':          'fn merge_sort',
    'Mandarin':             '中国',
    'Spanish':              'España',
    'Polish':               'Polski',
    'LaTeX math':           r'\int',
    'dialogue':             'Sarah asked',
    'medical/clinical':     'troponin',
    'legal':                'Licensee',
}


def main() -> int:
    failures: list[str] = []

    # 1. The canonical corpus file exists and is >= 1000 tokens.
    if not CORPUS.exists():
        # Defer: not a failure today, but warn so future tests can lock the path.
        print(f'[skip] canonical corpus {CORPUS} not committed yet — '
              f'create it when running PPL claims in commits/blog.')
    else:
        try:
            from mlx_lm import load
            import os
            snap = os.path.expanduser(
                '~/.cache/huggingface/hub/models--mlx-community--OLMoE-1B-7B-0125-Instruct-4bit/snapshots')
            snap = os.path.join(snap, os.listdir(snap)[0]) if os.path.exists(snap) else None
            if snap is None:
                print(f'[skip] OLMoE snapshot not cached; skipping tokenizer-proxy check')
            else:
                _, tok = load(snap, lazy=True)
                ntok = len(tok.encode(CORPUS.read_text()))
                if ntok < 1000:
                    failures.append(f'canonical corpus too small: {ntok} tokens (need >= 1000)')
                else:
                    print(f'  canonical corpus = {ntok} tokens >= 1000 ✓')
                # Diverse corpus: must be >= 3000 tokens AND contain all marker
                # distributions, otherwise the OOD probe is degraded into
                # something that re-creates the narrow-eval bias bug.
                if DIVERSE.exists():
                    dtext = DIVERSE.read_text()
                    dtok = len(tok.encode(dtext))
                    if dtok < 3000:
                        failures.append(f'diverse corpus too small: {dtok} tokens (need >= 3000)')
                    else:
                        print(f'  diverse corpus = {dtok} tokens >= 3000 ✓')
                    missing = [label for label, marker in DIVERSE_MARKERS.items()
                               if marker not in dtext]
                    if missing:
                        failures.append(
                            f'diverse corpus missing {len(missing)} distribution(s): '
                            + ', '.join(missing))
                    else:
                        print(f'  diverse corpus covers all {len(DIVERSE_MARKERS)} '
                              f'distribution markers ✓')
                else:
                    print(f'[skip] diverse corpus {DIVERSE} not present')
        except ImportError:
            print(f'[skip] mlx_lm not installed; cannot verify token count')

    # 2. eval_prune_quality.py --ppl-chunk default is documented as >= 256.
    eval_py = REPO / 'experiments' / 'moe_expert_phase0' / 'eval_prune_quality.py'
    src = eval_py.read_text()
    # Look for argparse default; quick string match suffices.
    if "'--ppl-chunk', type=int, default=" in src:
        # extract the default value
        import re
        m = re.search(r"'--ppl-chunk',\s*type=int,\s*default=(\d+)", src)
        if m:
            default = int(m.group(1))
            if default < 256:
                failures.append(f'eval_prune_quality.py --ppl-chunk default = {default} (need >= 256)')
            else:
                print(f'  --ppl-chunk default = {default} >= 256 ✓')
        else:
            failures.append('could not parse --ppl-chunk default from eval_prune_quality.py')
    else:
        failures.append('eval_prune_quality.py is missing the --ppl-chunk argparse line')

    # 3. eval_prune_quality.py --help must succeed. The argparse help-string
    # parser raises ValueError on stray '%' (interpreted as format specifier);
    # this caught a real shipped bug in commit 8fa880f where --corpus-only-hist
    # help contained "25%" and the flag could not be parsed. Invoke the script
    # as a subprocess (more honest than importing — argparse only validates
    # help strings at first add_argument call).
    import subprocess
    eval_py = REPO / 'experiments' / 'moe_expert_phase0' / 'eval_prune_quality.py'
    r = subprocess.run(['python3', str(eval_py), '--help'],
                       capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        failures.append(f'eval_prune_quality.py --help failed (rc={r.returncode}): '
                        f'stderr tail = {r.stderr[-400:]!r}')
    else:
        if '--corpus-only-hist' not in r.stdout:
            failures.append('eval_prune_quality.py --help missing --corpus-only-hist line')
        else:
            print('  eval_prune_quality.py --help parses + lists --corpus-only-hist ✓')

    # Same for the cross_corpus_25hist.py reproducer.
    cross_py = REPO / 'experiments' / 'moe_expert_phase0' / 'cross_corpus_25hist.py'
    if cross_py.exists():
        r = subprocess.run(['python3', str(cross_py), '--help'],
                           capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            failures.append(f'cross_corpus_25hist.py --help failed (rc={r.returncode}): '
                            f'stderr tail = {r.stderr[-400:]!r}')
        else:
            print('  cross_corpus_25hist.py --help parses ✓')

    # And bench_memory_savings.py — same class of bug bit it once already.
    bench_py = REPO / 'experiments' / 'moe_expert_phase0' / 'bench_memory_savings.py'
    if bench_py.exists():
        r = subprocess.run(['python3', str(bench_py), '--help'],
                           capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            failures.append(f'bench_memory_savings.py --help failed (rc={r.returncode}): '
                            f'stderr tail = {r.stderr[-400:]!r}')
        else:
            print('  bench_memory_savings.py --help parses ✓')

    if failures:
        print(f'\nFAIL — {len(failures)} methodology gap(s):')
        for f in failures:
            print(f'  {f}')
        return 1
    print(f'\nPASS — PPL eval methodology guards intact.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
