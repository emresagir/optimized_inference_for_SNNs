#!/usr/bin/env python3
"""
Graph the TAP sparsity-sweep log (timing lines + "snSP" counting lines in one
file, produced with the sweep test_data header).

Sparsity ratio = fraction of zero inputs = 1 - k/200 for L1 (k = active inputs
per timestep, derived from nnz1/T in the log). L2 input sparsity is measured
(1 - nnz2 / (T * 392)) because it follows from how L1 fires.

Figure (2 x 3):
  (a) cycles per call vs L1 input sparsity (L1, L2, average of both)
  (b) zoom on 90-100 % sparsity, where the natural data lives
  (c) cycles per call vs synaptic ops per call, linear fits (+ zoom inset)
  (d) share of event-driven cycles vs sparsity (rest = fixed cost)
  (e) L2 input sparsity vs L1 input sparsity (why L2 saturates)
  (f) cycles per synaptic op vs sparsity (flat = linear cost model holds)

Optional: --natural FILE overlays the samples of the natural-data TAP log
(same format) and shades their sparsity range, to check that the sweep
matches the real data.

Usage:
  python plot_sparsity_sweep.py sweep_log.txt
  python plot_sparsity_sweep.py sweep_log.txt --natural tap_spiking_counts_included_60_samples.txt
  python plot_sparsity_sweep.py sweep_log.txt --out sweep.png --no-show
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RE = re.compile(
    r"^\[(\d+)\]\s+T=\s*(\d+)\s+GT=(\d+)\s+Pred=(\d+)\s+(PASS|FAIL)"
)
LAYER_RE = re.compile(
    r"^(L\d) layer update cycles:\s+calls=(\d+)\s+last=(\d+)\s+min=(\d+)"
    r"\s+max=(\d+)\s+avg=(\d+)\s+total=(\d+)"
)
SP_RE = re.compile(
    r"^snSP\s+\[(\d+)\]\s+T=(\d+)\s+nnz1=(\d+)\s+sops1=(\d+)"
    r"\s+nnz2=(\d+)\s+sops2=(\d+)"
)
CLK_RE = re.compile(r"CLK=(\d+)")

LAYERS = ["L1", "L2"]
COLORS = {"L1": "#4C72B0", "L2": "#DD8452", "AVG": "#55A868"}
DEFAULT_CLK = 64_000_000


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_log(path: Path, in1: int, in2: int):
    """Return (records, clk_hz). One record per complete sample."""
    samples, pending, current, clk = [], None, None, None

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()

            m = CLK_RE.search(line)
            if m and clk is None:
                clk = int(m.group(1))

            m = SP_RE.match(line)          # snSP line comes BEFORE its sample line
            if m:
                i, T, n1, s1, n2, s2 = (int(v) for v in m.groups())
                pending = {"idx": i, "T": T, "nnz1": n1, "sops1": s1,
                           "nnz2": n2, "sops2": s2}
                continue

            m = SAMPLE_RE.match(line)
            if m:
                current = {"idx": int(m.group(1)), "T": int(m.group(2)),
                           "sp": pending, "L1": None, "L2": None}
                pending = None
                samples.append(current)
                continue

            m = LAYER_RE.match(line)
            if m and current is not None:
                layer, calls, total = m.group(1), int(m.group(2)), int(m.group(7))
                if layer in LAYERS and calls > 0 and current[layer] is None:
                    current[layer] = {"calls": calls, "total": total}

    records, dropped = [], 0
    for s in samples:
        sp = s["sp"]
        if not (s["L1"] and s["L2"] and sp) or sp["idx"] != s["idx"] or sp["T"] != s["T"]:
            dropped += 1
            continue
        T = s["T"]
        records.append({
            "idx": s["idx"], "T": T,
            "k": sp["nnz1"] / T,                                  # active L1 inputs / step
            "sp1": 100.0 * (1.0 - sp["nnz1"] / (T * in1)),        # L1 input sparsity [%]
            "sp2": 100.0 * (1.0 - sp["nnz2"] / (T * in2)),        # L2 input sparsity [%]
            "cyc1": s["L1"]["total"] / s["L1"]["calls"],
            "cyc2": s["L2"]["total"] / s["L2"]["calls"],
            "sops1": sp["sops1"] / T,
            "sops2": sp["sops2"] / T,
        })
    if dropped:
        print(f"[warn] {path.name}: {dropped} incomplete sample(s) skipped "
              f"(missing timing or snSP lines)", file=sys.stderr)
    return records, (clk or DEFAULT_CLK)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def linfit(x, y):
    c1, c0 = np.polyfit(x, y, 1)
    pred = c0 + c1 * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return c0, c1, (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def aggregate(recs):
    """Group records by L1 sparsity level. Returns list of dicts (sparse -> dense)."""
    keys = sorted({round(r["sp1"], 2) for r in recs}, reverse=True)
    out = []
    for kv in keys:
        rs = [r for r in recs if round(r["sp1"], 2) == kv]

        def ms(name):
            v = np.array([r[name] for r in rs])
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

        a = {"sp1": kv, "n": len(rs), "k": float(np.mean([r["k"] for r in rs]))}
        for name in ["cyc1", "cyc2", "sops1", "sops2", "sp2"]:
            a[name], a[name + "_std"] = ms(name)
        avg_vals = np.array([(r["cyc1"] + r["cyc2"]) / 2 for r in rs])
        a["cyc_avg"] = float(avg_vals.mean())
        a["cyc_avg_std"] = float(avg_vals.std(ddof=1)) if len(rs) > 1 else 0.0
        out.append(a)
    return out


def print_table(agg, fixed, fits):
    print("\nSweep summary (mean over repeats at each level)")
    print(f"{'k/step':>7}{'L1 sp%':>8}{'n':>3}{'L1 cyc':>10}{'L2 cyc':>10}"
          f"{'avg cyc':>10}{'L1 sops':>10}{'L2 sops':>9}{'L2 sp%':>8}"
          f"{'L1 c/sop':>10}{'L2 c/sop':>10}")
    for a in agg:
        def cps(cyc, sops, f):
            return f"{(cyc - f) / sops:>10.2f}" if (f is not None and sops > 0) else f"{'-':>10}"
        print(f"{a['k']:>7.0f}{a['sp1']:>8.1f}{a['n']:>3}{a['cyc1']:>10,.0f}{a['cyc2']:>10,.0f}"
              f"{a['cyc_avg']:>10,.0f}{a['sops1']:>10,.0f}{a['sops2']:>9,.0f}{a['sp2']:>8.1f}"
              f"{cps(a['cyc1'], a['sops1'], fixed['L1'])}{cps(a['cyc2'], a['sops2'], fixed['L2'])}")
    print("\nFixed cost (all-zero input, k = 0) and linear fit cycles = c0 + c1 * sops")
    for layer in LAYERS:
        c0, c1, r2 = fits[layer]
        f = fixed[layer]
        fs = f"{f:,.0f}" if f is not None else "n/a (no k=0 level)"
        print(f"  {layer}: measured fixed = {fs} cyc | fit c0 = {c0:,.0f}, "
              f"c1 = {c1:.2f} cyc/sop, R^2 = {r2:.3f}")
    print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_sweep(recs, agg, nat, clk, out_path, title_extra=""):
    x = np.array([a["sp1"] for a in agg])
    fixed = {"L1": None, "L2": None}
    k0 = [a for a in agg if a["k"] == 0]
    if k0:
        fixed = {"L1": k0[0]["cyc1"], "L2": k0[0]["cyc2"]}

    fits = {}
    for layer, ck, sk in [("L1", "cyc1", "sops1"), ("L2", "cyc2", "sops2")]:
        fits[layer] = linfit(np.array([r[sk] for r in recs]),
                             np.array([r[ck] for r in recs]))
    for layer in LAYERS:                       # fall back to fit intercept
        if fixed[layer] is None:
            fixed[layer] = fits[layer][0]

    print_table(agg, fixed, fits)

    nat_range = None
    if nat:
        s = np.array([r["sp1"] for r in nat])
        nat_range = (s.min(), s.max())

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

    def shade(ax):
        if nat_range:
            ax.axvspan(nat_range[0], nat_range[1], color="gray", alpha=0.13,
                       label="natural data range (L1)")

    def cycles_plot(ax, zoom=False):
        for key, name, col in [("cyc1", "L1", COLORS["L1"]),
                               ("cyc2", "L2", COLORS["L2"]),
                               ("cyc_avg", "Overall (avg of L1, L2)", COLORS["AVG"])]:
            m = np.array([a[key] for a in agg])
            sd = np.array([a[key + "_std"] for a in agg])
            ls = "--" if key == "cyc_avg" else "-"
            ax.errorbar(x, m, yerr=sd, marker="o", ms=4.5, capsize=3, lw=1.6,
                        ls=ls, color=col, label=f"{name} (sweep mean)")
        if nat:
            for layer, key in [("L1", "cyc1"), ("L2", "cyc2")]:
                ax.scatter([r["sp1"] for r in nat], [r[key] for r in nat], s=16,
                           facecolors="none", edgecolors=COLORS[layer], alpha=0.55,
                           linewidths=0.8, label=f"{layer} natural samples")
        shade(ax)
        ax.set_xlabel("L1 input sparsity ratio [%]  (zeros / all inputs)")
        ax.set_ylabel("Avg cycles per call (timestep)")
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    # (a) full range
    cycles_plot(ax_a)
    ax_a.set_title("(a) Cycles per call vs input sparsity")
    ax_a.legend(fontsize=7.5, loc="upper right")
    sec = ax_a.secondary_yaxis("right", functions=(lambda c: c / clk * 1e3,
                                                   lambda ms: ms * clk / 1e3))
    sec.set_ylabel(f"Time per call [ms] @ {clk / 1e6:.0f} MHz")

    # (b) zoom 90-100 %
    cycles_plot(ax_b, zoom=True)
    lo = 89.0
    ax_b.set_xlim(lo, 100.4)
    vis = [a for a in agg if a["sp1"] >= lo]
    ymax = max(max(a["cyc1"], a["cyc2"]) for a in vis) * 1.15
    if nat:
        ymax = max(ymax, max(max(r["cyc1"], r["cyc2"]) for r in nat if r["sp1"] >= lo) * 1.05)
    ax_b.set_ylim(0, ymax)
    ax_b.set_title("(b) Zoom: 89-100 % sparsity (natural-data regime)")
    for layer in LAYERS:
        ax_b.axhline(fixed[layer], color=COLORS[layer], ls=":", lw=1.1, alpha=0.8)
        ax_b.text(lo + 0.1, fixed[layer], f" {layer} fixed = {fixed[layer]:,.0f}",
                  color=COLORS[layer], va="bottom", fontsize=8)

    # (c) cycles vs sops with fits (linear axes + zoom inset for the natural regime)
    inset = ax_c.inset_axes([0.46, 0.07, 0.51, 0.40])
    for layer, ck, sk in [("L1", "cyc1", "sops1"), ("L2", "cyc2", "sops2")]:
        xs = np.array([r[sk] for r in recs])
        ys = np.array([r[ck] for r in recs])
        c0, c1, r2 = fits[layer]
        for axx, xmax in ((ax_c, xs.max() * 1.03), (inset, 2600)):
            axx.scatter(xs, ys, s=16 if axx is ax_c else 12, color=COLORS[layer],
                        alpha=0.7, edgecolor="black", linewidth=0.3,
                        label=f"{layer} sweep samples" if axx is ax_c else None)
            gx = np.linspace(0, xmax, 50)
            axx.plot(gx, c0 + c1 * gx, "k--", lw=1.0)
            if nat:
                axx.scatter([r[sk] for r in nat], [r[ck] for r in nat],
                            s=14 if axx is ax_c else 10, facecolors="none",
                            edgecolors=COLORS[layer], alpha=0.6, linewidths=0.8,
                            label=f"{layer} natural samples" if axx is ax_c else None)
    inset.set_xlim(0, 2600)
    near = [r for r in recs if r["sops1"] <= 2600] + [r for r in recs if r["sops2"] <= 2600]
    inset.set_ylim(0, max([r["cyc1"] for r in recs if r["sops1"] <= 2600] +
                          [r["cyc2"] for r in recs if r["sops2"] <= 2600] + [1]) * 1.1)
    inset.tick_params(labelsize=7)
    inset.set_title("zoom: low-activity region", fontsize=7.5)
    inset.grid(linestyle="--", alpha=0.4)
    txt = "\n".join(f"{l}: c0={fits[l][0]:,.0f} cyc, c1={fits[l][1]:.2f} cyc/sop, "
                    f"R\u00b2={fits[l][2]:.3f}" for l in LAYERS)
    ax_c.text(0.03, 0.80, txt, transform=ax_c.transAxes, ha="left", va="top",
              fontsize=8.5, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax_c.set_xlabel("Synaptic ops per call (sops / T)")
    ax_c.set_ylabel("Avg cycles per call")
    ax_c.set_title("(c) Cycles vs synaptic ops (linear fit)")
    ax_c.set_xlim(left=0)
    ax_c.set_ylim(bottom=0)
    ax_c.legend(fontsize=7.5, loc="upper left")
    ax_c.grid(linestyle="--", alpha=0.4)
    ax_c.set_axisbelow(True)

    # (d) event-driven share
    for layer, key in [("L1", "cyc1"), ("L2", "cyc2")]:
        share = np.array([100.0 * (1 - fixed[layer] / a[key]) for a in agg])
        ax_d.plot(x, share, marker="o", ms=4.5, lw=1.6, color=COLORS[layer],
                  label=f"{layer} event-driven share")
    shade(ax_d)
    ax_d.set_xlabel("L1 input sparsity ratio [%]")
    ax_d.set_ylabel("Event-driven share of cycles [%]\n(100 % - fixed cost share)")
    ax_d.set_title("(d) How much of the cost depends on spikes")
    ax_d.set_ylim(-3, 100)
    ax_d.legend(fontsize=8)
    ax_d.grid(linestyle="--", alpha=0.4)
    ax_d.set_axisbelow(True)

    # (e) L2 input sparsity vs L1 input sparsity
    ax_e.errorbar(x, [a["sp2"] for a in agg], yerr=[a["sp2_std"] for a in agg],
                  marker="o", ms=4.5, capsize=3, lw=1.6, color=COLORS["L2"],
                  label="sweep (mean)")
    if nat:
        ax_e.scatter([r["sp1"] for r in nat], [r["sp2"] for r in nat], s=16,
                     facecolors="none", edgecolors="black", alpha=0.55,
                     linewidths=0.8, label="natural samples")
    shade(ax_e)
    ax_e.set_xlabel("L1 input sparsity ratio [%]  (controlled)")
    ax_e.set_ylabel("L2 input sparsity ratio [%]  (measured)")
    ax_e.set_title("(e) L2 sparsity follows from L1 firing")
    ax_e.legend(fontsize=8)
    ax_e.grid(linestyle="--", alpha=0.4)
    ax_e.set_axisbelow(True)

    # (f) cycles per synaptic op
    for layer, ck, sk in [("L1", "cyc1", "sops1"), ("L2", "cyc2", "sops2")]:
        pts = [(a["sp1"], (a[ck] - fixed[layer]) / a[sk])
               for a in agg if a[sk] > 0]
        if pts:
            px, py = zip(*pts)
            ax_f.plot(px, py, marker="o", ms=4.5, lw=1.6, color=COLORS[layer],
                      label=f"{layer} (mean {np.mean(py):.2f})")
    shade(ax_f)
    ax_f.set_xlabel("L1 input sparsity ratio [%]")
    ax_f.set_ylabel("(cycles - fixed) / sops   [cycles per synaptic op]")
    ax_f.set_title("(f) Cost per synaptic op (flat = linear model)")
    top = max([(a[ck] - fixed[l]) / a[sk] for l, ck, sk in
               (("L1", "cyc1", "sops1"), ("L2", "cyc2", "sops2"))
               for a in agg if a[sk] > 0] + [1.0])
    ax_f.set_ylim(0, top * 1.15)
    ax_f.legend(fontsize=8)
    ax_f.grid(linestyle="--", alpha=0.4)
    ax_f.set_axisbelow(True)

    T_vals = sorted({r["T"] for r in recs})
    reps = sorted({a["n"] for a in agg})
    fig.suptitle(
        f"TAP sparsity sweep | {len(recs)} samples, {len(agg)} sparsity levels, "
        f"repeats per level {reps}, T = {T_vals} | @ {clk / 1e6:.0f} MHz {title_extra}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="sweep log (timing + snSP lines)")
    ap.add_argument("--natural", default=None,
                    help="natural-data TAP log (same format) to overlay; "
                         "defaults to tap_spiking_counts_included_60_samples.txt "
                         "next to the sweep log if it exists")
    ap.add_argument("--l1-inputs", type=int, default=200, help="L1 input neurons")
    ap.add_argument("--l2-inputs", type=int, default=392, help="L2 input neurons")
    ap.add_argument("--out", default="sparsity_sweep.png")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    log = Path(args.log)
    if not log.exists():
        sys.exit(f"file not found: {log}")
    recs, clk = parse_log(log, args.l1_inputs, args.l2_inputs)
    if len(recs) < 3:
        sys.exit("Not enough complete samples parsed. Does the log contain "
                 "'snSP [..]' lines and the L1/L2 timing lines?")

    nat_path = Path(args.natural) if args.natural else \
        log.parent / "tap_spiking_counts_included_60_samples.txt"
    nat = None
    if nat_path.exists() and nat_path.resolve() != log.resolve():
        nat, _ = parse_log(nat_path, args.l1_inputs, args.l2_inputs)
        print(f"natural-data overlay: {len(nat)} samples from {nat_path.name}")

    agg = aggregate(recs)
    plot_sweep(recs, agg, nat, clk, args.out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()