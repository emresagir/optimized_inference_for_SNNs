#!/usr/bin/env python3
"""
plot_tradeoff.py - Dense base vs Tap kernel: cycles vs input spike density.

Parses the UART logs of the two sweep runs, groups samples by input density
(nnz1 / (T * N_IN)), and finds the break-even density where the tap kernel
stops being faster than the base kernel. With --real, it also reads a tap-kernel
log of the real ST-MNIST test set and marks where that workload sits.

Usage:
    python plot_tradeoff.py --base sweep_base.txt --tap sweep_tap.txt
    python plot_tradeoff.py --base sweep_base.txt --tap sweep_tap.txt --panels a   # only Layer 1 graph
    python plot_tradeoff.py --base sweep_base.txt --tap sweep_tap.txt --panels ac  # Layer 1 + speedup
    python plot_tradeoff.py --base sweep_base.txt --tap sweep_tap.txt --real test_tap.txt --panels a
"""
import argparse
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- parsing
RE_SAMPLE = re.compile(r"^\[(\d+)\]\s+T=\s*(\d+)\s+GT=(\d+)\s+Pred=(\d+)\s+(PASS|FAIL)\s+spikes\s*:\s*([\d\s]*)$")
RE_CYC = re.compile(r"^(L\d) layer update cycles:\s*calls=(\d+)\s+last=\d+\s+min=(\d+)\s+max=(\d+)\s+avg=(\d+)\s+total=(\d+)")
RE_SNSP = re.compile(r"^snSP \[(\d+)\]\s+T=(\d+)\s+nnz1=(\d+)\s+sops1=(\d+)\s+nnz2=(\d+)\s+sops2=(\d+)")
RE_PLC = re.compile(r"^PLC (\S+)\s+addr=0x[0-9A-Fa-f]+\s+size=\s*(\d+)\s+(\w+)")
RE_CLK = re.compile(r"CLK=(\d+)")
RE_INDEX_ARRAY = re.compile(r"(idx|ptr|tap_)")   # format-specific index structures


def parse_log(path):
    """Return (samples, memory, clk). Cycle stats printed after [k] belong to sample k."""
    samples = defaultdict(dict)
    memory, clk, current = {}, None, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if m := RE_PLC.match(line):
                memory[m[1]] = (int(m[2]), m[3])
            elif m := RE_CLK.search(line):
                clk = int(m[1])
            elif m := RE_SNSP.match(line):
                samples[int(m[1])].update(nnz1=int(m[3]), sops1=int(m[4]),
                                          nnz2=int(m[5]), sops2=int(m[6]))
            elif m := RE_SAMPLE.match(line):
                current = int(m[1])
                samples[current].update(T=int(m[2]), gt=int(m[3]), pred=int(m[4]),
                                        ok=(m[5] == "PASS"), spikes=tuple(map(int, m[6].split())))
            elif m := RE_CYC.match(line):
                if current is None or int(m[2]) == 0:      # the calls=0 line is the reset print
                    continue
                L = m[1].lower()
                samples[current].update({f"{L}_avg": int(m[5]), f"{L}_min": int(m[3]),
                                         f"{L}_max": int(m[4]), f"{L}_total": int(m[6])})
    return dict(samples), memory, clk


# ----------------------------------------------------------------------------- analysis
def crossover(x, tap, base):
    """First x where tap goes from faster (<) to slower (>=) than base, linear interpolation."""
    diff = np.asarray(tap) - np.asarray(base)
    for i in range(len(diff) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            return x[i] - diff[i] * (x[i + 1] - x[i]) / (diff[i + 1] - diff[i])
    return None


def group(levels_of_sample, values):
    by = defaultdict(list)
    for lv, v in zip(levels_of_sample, values):
        by[lv].append(v)
    lv = sorted(by)
    return (np.array(lv),
            np.array([np.mean(by[k]) for k in lv]),
            np.array([np.min(by[k]) for k in lv]),
            np.array([np.max(by[k]) for k in lv]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sweep_base.txt")
    ap.add_argument("--tap", default="sweep_tap.txt")
    ap.add_argument("--n-in", type=int, default=200,
                    help="input elements per timestep (2 pol x 10 x 10 = 200; nnz1=8000 at T=40 is 100%%)")
    ap.add_argument("--out", default="tradeoff", help="output file stem (.png and .pdf)")
    ap.add_argument("--panels", default="abc",
                    help="which graphs to draw, any of: a = Layer 1, b = Layer 1+2, c = speedup "
                         "(e.g. --panels a, --panels ac). Default: abc")
    ap.add_argument("--real", default=None,
                    help="tap-kernel log of the real test set (with snSP lines) to mark on the graphs")
    args = ap.parse_args()

    base, mem_b, clk_b = parse_log(args.base)
    tap, mem_t, clk_t = parse_log(args.tap)
    clk = clk_t or clk_b or 64_000_000

    need = ("l1_avg", "l2_avg")
    idx = sorted(i for i in tap
                 if i in base and "nnz1" in tap[i]
                 and all(k in tap[i] for k in need) and all(k in base[i] for k in need))
    dropped = sorted((set(tap) | set(base)) - set(idx))
    if dropped:
        print(f"note: skipped samples without complete cycle data: {dropped}")
    if not idx:
        sys.exit("no samples with cycle data in both logs")

    dens = np.array([tap[i]["nnz1"] / (tap[i]["T"] * args.n_in) for i in idx]) * 100  # percent
    if dens.max() > 100.0 + 1e-9:
        print(f"warning: density > 100% (max {dens.max():.1f}%), check --n-in")
    dens = np.round(dens, 4)

    b_l1 = np.array([base[i]["l1_avg"] for i in idx], float)
    t_l1 = np.array([tap[i]["l1_avg"] for i in idx], float)
    b_tot = b_l1 + np.array([base[i]["l2_avg"] for i in idx], float)
    t_tot = t_l1 + np.array([tap[i]["l2_avg"] for i in idx], float)

    lv, bl1_m, bl1_lo, bl1_hi = group(dens, b_l1)
    _, tl1_m, tl1_lo, tl1_hi = group(dens, t_l1)
    _, btot_m, btot_lo, btot_hi = group(dens, b_tot)
    _, ttot_m, ttot_lo, ttot_hi = group(dens, t_tot)

    # Break-even: interpolation on level means, plus a linear cost model for L1
    x_l1 = crossover(lv, tl1_m, bl1_m)
    x_tot = crossover(lv, ttot_m, btot_m)
    slope, icpt = np.polyfit(dens, t_l1, 1)
    x_l1_model = (b_l1.mean() - icpt) / slope if slope > 0 else None
    r2 = 1 - np.sum((t_l1 - (slope * dens + icpt)) ** 2) / np.sum((t_l1 - t_l1.mean()) ** 2)

    # Real workload: where does the actual test set sit on the density axis?
    real = None
    if args.real:
        rs, _, _ = parse_log(args.real)
        r = [s for s in rs.values() if "nnz1" in s and s.get("T", 0) > 0]
        if not r:
            print(f"warning: no snSP lines found in {args.real}, real workload not marked")
        else:
            T = np.array([s["T"] for s in r], float)
            nnz = np.array([s["nnz1"] for s in r], float)
            d = nnz / (T * args.n_in) * 100
            cyc = [s for s in r if "l1_total" in s and "l2_total" in s]
            steps = sum(s["T"] for s in cyc)
            real = dict(n=len(r), pooled=nnz.sum() / (T.sum() * args.n_in) * 100,
                        p5=np.percentile(d, 5), p50=np.median(d), p95=np.percentile(d, 95),
                        dmax=d.max(), l1=None, l12=None, peak_d=None)
            if cyc:
                real["l1"] = sum(s["l1_total"] for s in cyc) / steps            # cycles per timestep
                real["l12"] = real["l1"] + sum(s["l2_total"] for s in cyc) / steps
                l1_peak = max(s["l1_max"] for s in cyc)                         # busiest single timestep
                real["peak_d"] = (l1_peak - icpt) / slope                        # density via cost model

    # ------------------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    panels = [p for p in "abc" if p in args.panels.lower()]
    if not panels:
        sys.exit("--panels must contain at least one of a, b, c")
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels) + (1 if len(panels) == 1 else -1), 5),
                             constrained_layout=True, squeeze=False)
    ax_of = dict(zip(panels, axes[0]))
    C_BASE, C_TAP, C_REAL = "#d1495b", "#2e86ab", "#e09f3e"
    xfit = np.linspace(0, 100, 200)

    def setup(ax, title, ylabel, cycles_axis=True):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Input spike density (%)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-2, 102)
        top = ax.secondary_xaxis("top", functions=(lambda x: 100 - x, lambda x: 100 - x))
        top.set_xlabel("Input sparsity (%)")
        if cycles_axis:
            right = ax.secondary_yaxis("right", functions=(lambda c: c / clk * 1e6,
                                                           lambda u: u * clk / 1e6))
            right.set_ylabel(f"µs per timestep @ {clk / 1e6:.0f} MHz")

    def mark(ax, x, label):
        if x is None:
            ax.text(0.97, 0.05, "no break-even in measured range\n(tap faster everywhere)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    bbox=dict(boxstyle="round", fc="white", ec="0.6"))
            return
        ax.axvspan(-2, x, color="green", alpha=0.06)
        ax.axvspan(x, 102, color="red", alpha=0.06)
        ax.axvline(x, color="k", ls="--", lw=1.2)
        ax.annotate(f"{label}\n{x:.1f}% density\n({100 - x:.1f}% sparsity)",
                    xy=(x, ax.get_ylim()[1] * 0.30), xytext=(10, 0), textcoords="offset points",
                    ha="left", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.6"))

    def hide_ticks_near(axis, value, lo, hi, log=False, frac=0.06):
        """Blank the default tick labels that would collide with a value label placed on the axis."""
        if not hasattr(axis, "_near"):
            axis._near = []
            orig = axis.get_major_formatter()
            base = orig if isinstance(orig, plt.FuncFormatter) else plt.FuncFormatter(lambda v, _: f"{v:g}")
            tr = (lambda v: np.log10(v)) if log else (lambda v: v)
            span = tr(hi) - tr(lo)
            axis.set_major_formatter(plt.FuncFormatter(
                lambda v, pos: "" if any(abs(tr(v) - tr(n)) / span < frac for n in axis._near)
                else base(v, pos)))
        axis._near.append(value)

    def guides(ax, x, y, color, ytext, below=False, xtext=True):
        """Dashed lines from the star to both axes, with its values written on the axes."""
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        ax.plot([x, x], [y0, y], ls="--", lw=1.1, color=color, zorder=5)
        ax.plot([x0, x], [y, y], ls="--", lw=1.1, color=color, zorder=5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        log = ax.get_yscale() == "log"
        ax.annotate(ytext, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va=("top" if below else "bottom") if log else "center",
                    color=color, fontsize=9, fontweight="bold", annotation_clip=False)
        hide_ticks_near(ax.yaxis, y, y0, y1, log=log)
        if xtext:
            ax.annotate(f"{x:.1f}%", xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, -4),
                        textcoords="offset points", ha="center", va="top",
                        color=color, fontsize=9, fontweight="bold", annotation_clip=False)
            hide_ticks_near(ax.xaxis, x, x0, x1)

    def mark_real(ax, y=None):
        if real is None or y is None:
            return
        ax.plot(real["pooled"], y, "*", ms=15, color=C_REAL, mec="k", zorder=6,
                label="Measured on test set")
        guides(ax, real["pooled"], y, C_REAL, f"{y / 1000:.1f}k")

    def draw(ax, raw_t, m_b, lo_b, hi_b, m_t, lo_t, hi_t):
        ax.scatter(dens, raw_t, s=10, color=C_TAP, alpha=0.3, zorder=2)
        ax.fill_between(lv, lo_t, hi_t, color=C_TAP, alpha=0.15)
        ax.plot(lv, m_t, "o-", color=C_TAP, label="Tap kernel (mean per level)", zorder=3)
        ax.fill_between(lv, lo_b, hi_b, color=C_BASE, alpha=0.15)
        ax.plot(lv, m_b, "s-", color=C_BASE, ms=4, label="Base kernel (mean per level)", zorder=3)

    # (a) Layer 1
    if "a" in ax_of:
        ax = ax_of["a"]
        draw(ax, t_l1, bl1_m, bl1_lo, bl1_hi, tl1_m, tl1_lo, tl1_hi)
        ax.plot(xfit, slope * xfit + icpt, ":", color=C_TAP, lw=1.2,
                label=f"Tap linear fit (R²={r2:.3f})")
        setup(ax, "(a) Layer 1 update", "Cycles per timestep")
        ax.set_ylim(0, max(tl1_hi.max(), bl1_hi.max()) * 1.1)
        mark(ax, x_l1, "Break-even")
        mark_real(ax, real and real["l1"])
        ax.legend(loc="upper left", fontsize=8)

    # (b) Layer 1 + Layer 2
    if "b" in ax_of:
        ax = ax_of["b"]
        draw(ax, t_tot, btot_m, btot_lo, btot_hi, ttot_m, ttot_lo, ttot_hi)
        setup(ax, "(b) Layer 1 + Layer 2 update", "Cycles per timestep")
        ax.set_ylim(0, max(ttot_hi.max(), btot_hi.max()) * 1.3)  # room for legend
        mark(ax, x_tot, "Break-even")
        mark_real(ax, real and real["l12"])
        ax.legend(loc="upper left", fontsize=8)

    # (c) Speedup
    if "c" in ax_of:
        ax = ax_of["c"]
        ax.plot(lv, bl1_m / tl1_m, "o-", color=C_TAP, label="Layer 1")
        ax.plot(lv, btot_m / ttot_m, "^-", color="#6a4c93", label="Layer 1 + Layer 2")
        ax.axhline(1.0, color="k", ls="--", lw=1.2, label="Break-even (1×)")
        ax.set_yscale("log")
        ax.set_yticks([0.5, 1, 2, 5, 10, 20])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        setup(ax, "(c) Speedup of tap over base", "Base cycles / Tap cycles", cycles_axis=False)
        if x_l1 is not None:
            ax.axvline(x_l1, color="k", ls=":", lw=1)
        if real and real["l1"]:
            sp1, sp12 = bl1_m.mean() / real["l1"], btot_m.mean() / real["l12"]
            ax.plot(real["pooled"], sp1, "*", ms=15, color=C_TAP, mec="k", zorder=6)
            ax.plot(real["pooled"], sp12, "*", ms=15, color="#6a4c93", mec="k",
                    zorder=6, label="Measured on test set")
            guides(ax, real["pooled"], sp1, C_TAP, f"{sp1:.1f}×", below=sp1 < sp12, xtext=False)
            guides(ax, real["pooled"], sp12, "#6a4c93", f"{sp12:.1f}×", below=sp12 <= sp1)
        ax.legend(loc="upper right", fontsize=8)

    if len(panels) > 1:
        fig.suptitle("Base vs Tap kernel: latency trade-off over input spike density", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"saved {args.out}.png and {args.out}.pdf")

    # ------------------------------------------------------------------------- summary
    acc = defaultdict(lambda: [0, 0, 0])   # level -> [n, base_ok, tap_ok]
    for d, i in zip(dens, idx):
        a = acc[d]
        a[0] += 1
        a[1] += base[i]["ok"]
        a[2] += tap[i]["ok"]

    print("\n density  sparsity |  base L1   tap L1  L1 spd |  base L1+2  tap L1+2  tot spd | acc base  acc tap")
    print("-" * 104)
    for k, d in enumerate(lv):
        n, ob, ot = acc[d]
        print(f"{d:7.2f}%  {100 - d:7.2f}% | {bl1_m[k]:8.0f} {tl1_m[k]:8.0f} {bl1_m[k] / tl1_m[k]:6.2f}x |"
              f" {btot_m[k]:10.0f} {ttot_m[k]:9.0f} {btot_m[k] / ttot_m[k]:7.2f}x |"
              f"    {ob}/{n}      {ot}/{n}")

    print(f"\nTap L1 cost model: cycles ≈ {icpt:.0f} + {slope:.1f} * density%   (R² = {r2:.4f})")
    print(f"  -> fixed overhead {icpt:.0f} cycles/timestep, "
          f"{slope * 100 / args.n_in:.2f} cycles per input spike")
    if x_l1 is not None:
        print(f"L1 break-even (interpolated): {x_l1:.1f}% density = {100 - x_l1:.1f}% sparsity")
    if x_l1_model is not None:
        print(f"L1 break-even (linear model): {x_l1_model:.1f}% density = {100 - x_l1_model:.1f}% sparsity")
    print("L1+L2 break-even: " + (f"{x_tot:.1f}% density" if x_tot is not None
                                  else "none in measured range (tap faster everywhere)"))

    if real:
        print(f"\nReal test set ({real['n']} samples, {args.real}):")
        print(f"  input density: mean {real['pooled']:.2f}% (time-weighted), median {real['p50']:.2f}%, "
              f"5–95% {real['p5']:.2f}–{real['p95']:.2f}%, densest sample {real['dmax']:.2f}%")
        if x_l1 is not None:
            print(f"  margin to L1 break-even: {x_l1 / real['pooled']:.0f}x the mean density")
        if real["l1"]:
            print(f"  measured tap L1: {real['l1']:.0f} cycles/timestep "
                  f"(model predicts {icpt + slope * real['pooled']:.0f}); "
                  f"base L1: {bl1_m.mean():.0f}  ->  {bl1_m.mean() / real['l1']:.2f}x faster")
            print(f"  measured tap L1+L2: {real['l12']:.0f} cycles/timestep; "
                  f"base: {btot_m.mean():.0f}  ->  {btot_m.mean() / real['l12']:.2f}x faster")
            print(f"  busiest single timestep ≈ {real['peak_d']:.1f}% density (from L1 max cycles), "
                  f"{'below' if x_l1 is None or real['peak_d'] < x_l1 else 'ABOVE'} the break-even")

    # Memory of format-specific index structures
    def idx_bytes(mem):
        return sum(sz for name, (sz, _) in mem.items() if RE_INDEX_ARRAY.search(name))
    mb, mt = idx_bytes(mem_b), idx_bytes(mem_t)
    if mb or mt:
        print(f"\nIndex memory: base {mb} B, tap {mt} B" + (f"  ({mb / mt:.1f}x smaller)" if mt else ""))

    # Bit-exactness check: same inputs should give the same output spikes
    diff_spk = [i for i in sorted(set(base) & set(tap))
                if "spikes" in base[i] and "spikes" in tap[i] and base[i]["spikes"] != tap[i]["spikes"]]
    diff_pred = [i for i in diff_spk if base[i]["pred"] != tap[i]["pred"]]
    if diff_spk:
        print(f"\nWARNING: output spikes differ between kernels on {len(diff_spk)} samples: {diff_spk}")
        print(f"         prediction differs on: {diff_pred}")
    else:
        print("\nOutput spikes identical between kernels on all samples.")


if __name__ == "__main__":
    main()