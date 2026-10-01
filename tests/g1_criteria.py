"""gh #171 — G1 pass criteria, split by quantization tier.

Why this file exists
--------------------
The original G1 gate asserted one set of numbers regardless of `--vquant`:
logprob drift < 0.01 AND storage ratio > 3.0. Those two are mutually
unsatisfiable in any single configuration, and the gate only ever passed by
accident: before gh #148, `VQUANT` was silently ignored on the uniform path, so
a run labelled `turbo4` actually stored fp16. The <0.01 target was calibrated
against that accidental losslessness. Once #148 made block-INT4 really apply,
the same test measured the honest INT4 cost and failed unconditionally — a
threshold artifact, not a substrate regression.

The split
---------
Each tier asserts what it is actually responsible for:

* **fp16** is the *infrastructure* gate. Nothing is quantized, so any drift is
  the wire, the store, and the cache rebuild — it must be ~0. This is the run
  that answers "does the substrate corrupt K/V". It asserts drift and does NOT
  assert compression, because fp16's storage ratio is 1.00 by definition.
* **turbo4 / int8** are the *compression* gates. They assert the storage win
  and first-token identity, plus a drift ceiling.

On the drift ceilings
---------------------
The ceilings below are **regression tripwires, not quality certifications**.
A quantizer that gets meaningfully worse trips them; a number under them does
not by itself mean the tier is good enough to ship for a workload. Quality is
certified by first-token identity here (the argmax must be unchanged), and at
the model tier by the MLX edition's downstream metrics — the gh #9 K_top
retraction is the standing reminder that a wire-level number is not a
workload-level claim.

Values are set with headroom over what was measured on gpt2 (2026-08-02,
release build, `tests/test_kv_prefix_prototype.py`):

    tier      measured drift    ceiling    measured storage    floor
    fp16          0.000205        0.01           1.00x          n/a
    int8          0.076149        0.12           2.00x          1.9x
    turbo4        0.158624        0.25           3.52x          3.0x

`drift_oracle_vs_cold` (cache-rebuild noise, no quantization involved) measured
0.000011 in all three — five orders of magnitude under the fp16 ceiling, which
is what lets us attribute the fp16 number to the wire rather than to the
harness.
"""

# tier -> (max logprob drift, min storage ratio or None, asserts_compression)
G1_CRITERIA = {
    "fp16":   (0.01, None, False),
    "int8":   (0.12, 1.9,  True),
    "turbo4": (0.25, 3.0,  True),
}

# gh #370: the ceilings above were measured on gpt2 and then applied, unchanged,
# to the MLX edition's Llama-3.2-1B-Instruct-4bit, where the turbo4 run has
# never passed. Drift is a property of MODEL x QUANTIZER, not of the quantizer
# alone, so a ceiling is only meaningful for the model it was measured on.
# Measured 2026-09-27 on current main (release build, 5 repeats):
#
#    model                       tier     drift      ceiling (x1.58, gpt2's headroom)
#    Llama-3.2-1B-Instruct-4bit  fp16     0.000000   0.01   (lossless — unchanged)
#    Llama-3.2-1B-Instruct-4bit  int8     0.159302   0.25
#    Llama-3.2-1B-Instruct-4bit  turbo4   0.352295   0.55
#
# Why this is calibration and not a relaxed gate hiding a regression: the same
# session re-ran the gpt2 prototype and reproduced its 2026-08-02 numbers to six
# digits (int8 0.076149, turbo4 0.158624) — the quantizer and wire are
# bit-for-bit what they were; fp16 drift on Llama is exactly 0 (the wire, store
# and cache rebuild add nothing); and int8 and turbo4 sit at the SAME ~2.1-2.2x
# of their gpt2 values, which is a model's sensitivity, not one tier breaking.
# Storage floors are the quantizer's and do not change with the model.
G1_DRIFT_BY_MODEL = {
    "gpt2": {"fp16": 0.01, "int8": 0.12, "turbo4": 0.25},
    "mlx-community/Llama-3.2-1B-Instruct-4bit": {"fp16": 0.01, "int8": 0.25, "turbo4": 0.55},
}

# Shared across tiers.
TTFT_RATIO_MAX = 0.30


def evaluate(vquant, ratio, warm_id, cold_id, oracle_id, drift, storage_ratio, model="gpt2"):
    """Return (overall_pass, list of (label, passed, detail)) for one G1 run.

    `model` selects the drift ceiling (G1_DRIFT_BY_MODEL). A model nobody has
    measured is an error, not a silent borrow of another model's number."""
    if vquant not in G1_CRITERIA:
        raise ValueError(f"no G1 criteria defined for vquant={vquant!r}")
    if model not in G1_DRIFT_BY_MODEL:
        raise ValueError(f"no G1 drift ceiling measured for model {model!r} — measure fp16/int8/turbo4 "
                         f"on it and add a row to G1_DRIFT_BY_MODEL (gh #370)")
    _, storage_min, asserts_compression = G1_CRITERIA[vquant]
    drift_max = G1_DRIFT_BY_MODEL[model][vquant]

    results = [
        ("TTFT ratio", ratio < TTFT_RATIO_MAX,
         f"{ratio:.3f} < {TTFT_RATIO_MAX}"),
        ("first-token", warm_id == cold_id == oracle_id,
         f"warm={warm_id} cold={cold_id} oracle={oracle_id}"),
        ("logprob drift", drift < drift_max,
         f"{drift:.6f} < {drift_max}  ({'losslessness' if vquant == 'fp16' else 'regression tripwire'})"),
    ]
    if asserts_compression:
        results.append(("storage", storage_ratio > storage_min,
                        f"{storage_ratio:.2f}x > {storage_min}x"))
    else:
        results.append(("storage", True,
                        f"{storage_ratio:.2f}x — not asserted for fp16 (baseline tier)"))

    return all(p for _, p, _ in results), results


def print_verdict(vquant, results, overall):
    print("\n──────── verdict ────────")
    for label, passed, detail in results:
        print(f"  {label:<14} {'PASS' if passed else 'FAIL'}   {detail}")
    print(f"  G1 GATE ({vquant})  {'PASS' if overall else 'FAIL'}")
    if not overall and any(label == "TTFT ratio" and not p for label, p, _ in results):
        print("  NOTE: TTFT ratio is only meaningful against a RELEASE server "
              "(./pion-server). A -O0 --build-dev server inflates fetch latency "
              "enough to fail this on its own.")
