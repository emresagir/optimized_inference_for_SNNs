#!/usr/bin/env python3
"""
Generate a sparsity-sweep test header in the same format as the 200-sample
test_data.h (TestSample table, int8 spikes [timestep][200], .ext_weights).

Every sample has a FIXED number of active inputs per timestep (k out of 200),
so the input sparsity ratio is exact:

    sparsity = 1 - k / 200

Samples are grouped by level (k ascending), REPS samples per level.
  k = 0    -> all-zero input: measures the fixed cost (decay + scans + reset)
  k = 200  -> fully dense input: gives the worst case / sops_max

Non-zero values are random ints in [VALUE_MIN, VALUE_MAX] (the real data holds
event counts like 2..17, not binary). Positions are uniform random over the
200 inputs (2 x 10 x 10), so border/center taps average out.

Usage:
  python make_sparsity_sweep_testdata.py
  python make_sparsity_sweep_testdata.py --timesteps 60 --reps 3 \
         --levels 0,2,5,10,20,50,100,200 --table-name test_samples
Outputs:
  test_data_sweep.h        the header (rename / replace your test_data.h)
  sweep_manifest.csv       sample index -> k, density, sparsity, T
"""

import argparse
import csv
import random
from pathlib import Path

NUM_INPUTS = 200          # CH * H * W = 2 * 10 * 10
NUM_CLASSES = 10
VALUE_MIN, VALUE_MAX = 2, 16
DEFAULT_LEVELS = "0,1,2,4,6,10,15,20,30,40,60,100,200"


def make_sample(rng, T, k):
    steps = []
    for _ in range(T):
        row = [0] * NUM_INPUTS
        for pos in rng.sample(range(NUM_INPUTS), k):
            row[pos] = rng.randint(VALUE_MIN, VALUE_MAX)
        steps.append(row)
    return steps


def fmt_row(row):
    return "{" + ", ".join(f"{v:4d}" for v in row) + "}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default=DEFAULT_LEVELS,
                    help="comma separated k values (active inputs per timestep, 0..200)")
    ap.add_argument("--reps", type=int, default=5, help="samples per level")
    ap.add_argument("--timesteps", type=int, default=40, help="T of every sample")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--table-name", default="test_samples",
                    help="name of the TestSample array at the end of the header "
                         "(must match the name your main.c uses)")
    ap.add_argument("--out", default="test_data_sweep.h")
    ap.add_argument("--manifest", default="sweep_manifest.csv")
    args = ap.parse_args()

    levels = [int(v) for v in args.levels.split(",")]
    for k in levels:
        if not 0 <= k <= NUM_INPUTS:
            raise SystemExit(f"level k={k} out of range 0..{NUM_INPUTS}")

    rng = random.Random(args.seed)
    plan = [(k, r) for k in levels for r in range(args.reps)]
    n = len(plan)
    T = args.timesteps

    lines = []
    w = lines.append
    w("#ifndef TEST_DATA_H")
    w("#define TEST_DATA_H")
    w("")
    w("#include <stdint.h>")
    w("")
    w("/* Sparsity sweep: fixed number of active inputs per timestep per sample.")
    w(f" * levels (k of {NUM_INPUTS}) = {args.levels}, reps = {args.reps}, T = {T}, seed = {args.seed}")
    w(" * Labels are dummy (0): PASS/FAIL is meaningless for this file. */")
    w("")
    w("/* ── Constants ──────────────────────────────────────── */")
    w(f"#define TD_NUM_INPUTS    {NUM_INPUTS}   /* CH * H * W = 2 * 10 * 10 */")
    w(f"#define TD_NUM_CLASSES   {NUM_CLASSES}")
    w(f"#define TD_NUM_SAMPLES   {n}")
    w("")
    w("/* ── Per-sample descriptor ───────────────────────────── */")
    w("typedef struct {")
    w("    const int8_t  (*spikes)[TD_NUM_INPUTS]; /* [timestep][input_idx] */")
    w("    uint16_t       num_timesteps;")
    w("    uint8_t        label;                   /* ground-truth class 0-9 */")
    w("} TestSample;")
    w("")

    manifest = []
    for idx, (k, rep) in enumerate(plan):
        steps = make_sample(rng, T, k)
        dens = 100.0 * k / NUM_INPUTS
        w(f"/* ── Sample {idx}  (sweep k={k}/{NUM_INPUTS}, density={dens:.1f}%, "
          f"sparsity={100 - dens:.1f}%, rep {rep})  label=0  T={T} ── */")
        w('__attribute__((section(".ext_weights")))')
        w(f"static const int8_t _spikes_{idx}[{T}][TD_NUM_INPUTS] = {{")
        for t, row in enumerate(steps):
            sep = "," if t < T - 1 else " "
            w(f"    {fmt_row(row)}{sep} /* t={t} */")
        w("};")
        w("")
        manifest.append([idx, k, f"{dens:.2f}", f"{100 - dens:.2f}", T, rep])

    w("/* ── Sample table ────────────────────────────────────── */")
    w(f"static const TestSample {args.table_name}[TD_NUM_SAMPLES] = {{")
    for idx in range(n):
        sep = "," if idx < n - 1 else ""
        w(f"    {{ _spikes_{idx}, {T}, 0 }}{sep}")
    w("};")
    w("")
    w("#endif /* TEST_DATA_H */")

    Path(args.out).write_text("\n".join(lines) + "\n")
    with open(args.manifest, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["sample_idx", "k_active_per_step", "input_density_pct",
                     "input_sparsity_pct", "T", "rep"])
        cw.writerows(manifest)

    print(f"samples: {n}  (levels={len(levels)} x reps={args.reps}), T={T}")
    print(f"data size in QSPI: {n * T * NUM_INPUTS / 1024:.0f} KiB")
    print(f"wrote {args.out} and {args.manifest}")


if __name__ == "__main__":
    main()
