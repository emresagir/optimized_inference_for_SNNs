#!/usr/bin/env python3
"""Thesis figure: cycles per timestep of the tap table in Conv 1 vs L1 input sparsity (sparsity sweep).

Only Conv 1: the sweep controls the L1 input, the L2 input sparsity follows from L1 firing and stays
above ~87 %, so the sweep does not cover Conv 2.

Sweep log : <folder>/sweep/sweep_tap.txt (tap project with test_sweep.h, k active inputs out of 200
            per timestep, 5 samples per level, T = 40). Parsed with sweep_test/plot_sparsity_sweep.py.
Dense     : Basic and CMSIS-NN do not depend on the input, so they are drawn as the mean cycles per
            timestep of base_694.txt and cmsis_694.txt (same session, same build settings).
Natural   : the L1 input sparsity range of the 694 test samples (tap_694.txt), min to max.

Usage:  python3 plot_sweep_thesis.py [folder]        (default folder: test_14_11_for_tap_CSR_base)
Output: <folder>/sweep/thesis_sweep.pdf/.png  (next to sweep_tap.txt)
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sweep_test"))
from plot_sparsity_sweep import parse_log, aggregate, linfit   # noqa: E402

FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "test_14_11_for_tap_CSR_base"
OUT = FOLDER / "sweep"
IN1, IN2 = 200, 392
LAYER_RE = re.compile(r"^(L\d) layer update cycles:\s+calls=(\d+).*total=(\d+)")


def dense_mean(path):
    """Mean cycles per timestep of a 694-sample log, per layer."""
    v = {"L1": [], "L2": []}
    for line in path.read_text(errors="ignore").splitlines():
        m = LAYER_RE.match(line.strip())
        if m and int(m.group(2)) > 0 and m.group(1) in v:
            v[m.group(1)].append(int(m.group(3)) / int(m.group(2)))
    return {k: float(np.mean(x)) for k, x in v.items()}


recs, clk = parse_log(FOLDER / "sweep" / "sweep_tap.txt", IN1, IN2)
nat, _ = parse_log(FOLDER / "tap_694.txt", IN1, IN2)
agg = aggregate(recs)
dense = {"Basic": dense_mean(FOLDER / "base_694.txt"), "CMSIS-NN": dense_mean(FOLDER / "cmsis_694.txt")}
nat_sp = np.array([r["sp1"] for r in nat])
nat_lo, nat_med, nat_mean, nat_hi = nat_sp.min(), np.median(nat_sp), nat_sp.mean(), nat_sp.max()

x = np.array([a["sp1"] for a in agg])
tap = np.array([a["cyc1"] for a in agg])
c0, c1, r2 = linfit(np.array([r["sops1"] for r in recs]), np.array([r["cyc1"] for r in recs]))


def break_even(ref):
    """L1 input sparsity [%] where the tap cost reaches `ref` (linear between levels), or None."""
    y = tap
    for i in range(len(x) - 1):            # x goes from sparse to dense, y increases
        if y[i] <= ref <= y[i + 1]:
            return x[i] + (ref - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    return None


# ---- terminal summary ----
print(f"{len(recs)} sweep samples, {len(agg)} levels, natural L1 sparsity: "
      f"mean {nat_mean:.1f} %, median {nat_med:.1f} %, range {nat_lo:.1f}-{nat_hi:.1f} %")
print(f"{'k':>4}{'L1 sp%':>8}{'Conv1 cyc':>11}")
for a in agg:
    print(f"{a['k']:>4.0f}{a['sp1']:>8.1f}{a['cyc1']:>11,.0f}")
print(f"Conv 1: fit cycles = {c0:,.0f} + {c1:.2f} * sops  (R^2 = {r2:.4f}), "
      f"k=0 measured {tap[0]:,.0f}, densest level {tap[-1]:,.0f}")
for name, d in dense.items():
    be = break_even(d["L1"])
    print(f"   vs {name:8s} {d['L1']:>9,.0f} cyc: " +
          (f"tap is slower below {be:.1f} % L1 input sparsity" if be is not None
           else f"tap is faster at every level"))

# ---- figure ----
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
DENSE_STYLE = {"Basic": ("0.35", "--"), "CMSIS-NN": ("0.35", ":")}
fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.axvspan(nat_lo, nat_hi, color="0.88", lw=0, label="Test samples")
ax.axvline(nat_mean, color="k", lw=0.8, label=f"Test samples, mean ({nat_mean:.1f} %)")
for name, (c, ls) in DENSE_STYLE.items():
    ax.axhline(dense[name]["L1"] / 1e3, color=c, ls=ls, lw=0.9, label=name)
ax.plot(x, tap / 1e3, "o-", color="#1f4e9c", ms=3, lw=1.2, label="Tap table")
for name in dense:
    be = break_even(dense[name]["L1"])
    if be is not None:
        ax.plot(be, dense[name]["L1"] / 1e3, "x", color="k", ms=5, mew=0.9)
        ax.annotate(f"{be:.0f} %", (be, dense[name]["L1"] / 1e3), textcoords="offset points",
                    xytext=(-4, 4 if name == "Basic" else -9), ha="right", fontsize=7.5)
ax.set_xlabel("L1 input sparsity [%]")
ax.set_ylabel(r"Cycles per timestep [$\times 10^3$]")
ax.set_xlim(-2, 101); ax.set_ylim(0, 130)
ax.grid(True, ls=":", lw=0.4, color="0.8"); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
sec = ax.secondary_yaxis("right", functions=(lambda c: c * 1e3 / clk * 1e3, lambda ms: ms * clk / 1e6))
sec.set_ylabel(f"Time per timestep [ms] at {clk / 1e6:.0f} MHz")
h, l = ax.get_legend_handles_labels()
order = [4, 2, 3, 0, 1]                     # Tap, Basic, CMSIS-NN, test range, test mean
ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=2, frameon=False)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"thesis_sweep.{ext}", dpi=200)
print("[SAVED]", OUT / "thesis_sweep.pdf", "(+ .png preview)")
