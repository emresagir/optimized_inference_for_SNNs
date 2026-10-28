#!/usr/bin/env python3
"""
Compare SNN inference cycle counts across methods (base, cmsisnn, csr, tap).

Main graph:
  Grouped bar chart of average cycles per layer-update call (per timestep),
  L1, L2 and overall (average of L1 and L2) side by side for each method.
    - Bar height  : weighted mean = sum(total) / sum(calls) over all samples
    - Error bars  : std of the per-sample mean cycles/call across samples
    - Bar labels  : speedup vs base (base_mean / method_mean)
    - Right axis  : same values converted to milliseconds at CLK
    - X tick label: accuracy (PASS count) of each method

Optional (--with-total):
  Second figure with total cycles per sample (L1 + L2 per inference),
  with mean timesteps T shown, since T varies between samples.

Usage:
  python compare_snn_cycles.py                       # files in current dir
  python compare_snn_cycles.py /path/to/tests_dir    # files in another dir
  python compare_snn_cycles.py --with-total --out cmp.png --no-show
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

METHODS = ["base", "cmsisnn", "csr", "tap"]
FILE_PATTERN = "{m}_test_60_samples.txt"
LAYERS = ["L1", "L2"]
DEFAULT_CLK_HZ = 64_000_000

SAMPLE_RE = re.compile(
    r"^\[(\d+)\]\s+T=\s*(\d+)\s+GT=(\d+)\s+Pred=(\d+)\s+(PASS|FAIL)"
)
LAYER_RE = re.compile(
    r"^(L\d) layer update cycles:\s+calls=(\d+)\s+last=(\d+)\s+min=(\d+)"
    r"\s+max=(\d+)\s+avg=(\d+)\s+total=(\d+)"
)
CLK_RE = re.compile(r"CLK=(\d+)")


def parse_file(path: Path):
    """Return (samples, clk_hz). samples = list of dicts, one per timestep-run."""
    samples = []
    clk = None
    current = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()

            m = CLK_RE.search(line)
            if m and clk is None:
                clk = int(m.group(1))

            m = SAMPLE_RE.match(line)
            if m:
                current = {
                    "idx": int(m.group(1)),
                    "T": int(m.group(2)),
                    "gt": int(m.group(3)),
                    "pred": int(m.group(4)),
                    "pass": m.group(5) == "PASS",
                    "L1": None,
                    "L2": None,
                }
                samples.append(current)
                continue

            m = LAYER_RE.match(line)
            if m and current is not None:
                layer = m.group(1)
                calls = int(m.group(2))
                total = int(m.group(7))
                # The header lines before sample [00] have calls=0 and no
                # current sample yet, so they are skipped by the check above.
                if layer in LAYERS and calls > 0 and current[layer] is None:
                    current[layer] = {"calls": calls, "total": total}

    # Keep only complete samples
    samples = [s for s in samples if s["L1"] and s["L2"]]
    return samples, clk or DEFAULT_CLK_HZ


def layer_stats(samples, layer):
    calls = np.array([s[layer]["calls"] for s in samples], dtype=float)
    total = np.array([s[layer]["total"] for s in samples], dtype=float)
    per_sample_mean = total / calls
    return {
        "mean": total.sum() / calls.sum(),         # weighted mean cycles/call
        "std": per_sample_mean.std(ddof=1) if len(samples) > 1 else 0.0,
        "per_sample": per_sample_mean,
    }


def overall_stats(samples):
    """Average of L1 and L2 cycles per call, i.e. (L1 + L2) / 2."""
    calls = np.array([s["L1"]["calls"] for s in samples], dtype=float)
    total = np.array([(s["L1"]["total"] + s["L2"]["total"]) / 2.0
                      for s in samples], dtype=float)
    per_sample_mean = total / calls
    return {
        "mean": total.sum() / calls.sum(),
        "std": per_sample_mean.std(ddof=1) if len(samples) > 1 else 0.0,
        "per_sample": per_sample_mean,
    }


def load_all(folder: Path):
    data = {}
    clk = DEFAULT_CLK_HZ
    for m in METHODS:
        p = folder / FILE_PATTERN.format(m=m)
        if not p.exists():
            print(f"[warn] missing file, skipping: {p}", file=sys.stderr)
            continue
        samples, clk_found = parse_file(p)
        if not samples:
            print(f"[warn] no samples parsed in {p}", file=sys.stderr)
            continue
        clk = clk_found
        data[m] = {
            "samples": samples,
            "L1": layer_stats(samples, "L1"),
            "L2": layer_stats(samples, "L2"),
            "ALL": overall_stats(samples),
            "n_pass": sum(s["pass"] for s in samples),
            "n": len(samples),
        }
    return data, clk


def print_summary(data):
    print(f"\n{'method':<9}{'acc':>9}{'L1 cyc/call':>14}{'L1 spdup':>10}"
          f"{'L2 cyc/call':>14}{'L2 spdup':>10}"
          f"{'AVG cyc/call':>14}{'AVG spdup':>11}")
    base = data.get("base")
    for m, d in data.items():
        sp1 = base["L1"]["mean"] / d["L1"]["mean"] if base else float("nan")
        sp2 = base["L2"]["mean"] / d["L2"]["mean"] if base else float("nan")
        spa = base["ALL"]["mean"] / d["ALL"]["mean"] if base else float("nan")
        print(f"{m:<9}{d['n_pass']:>4}/{d['n']:<4}"
              f"{d['L1']['mean']:>14.0f}{sp1:>9.2f}x"
              f"{d['L2']['mean']:>14.0f}{sp2:>9.2f}x"
              f"{d['ALL']['mean']:>14.0f}{spa:>10.2f}x")
    print()


def plot_per_call(data, clk, out_path):
    methods = list(data.keys())
    x = np.arange(len(methods))
    width = 0.27
    series = ["L1", "L2", "ALL"]
    colors = {"L1": "#4C72B0", "L2": "#DD8452", "ALL": "#55A868"}
    names = {"L1": "L1", "L2": "L2", "ALL": "Overall (avg of L1, L2)"}
    base = data.get("base")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    all_tops = [data[m][k]["mean"] + data[m][k]["std"]
                for m in methods for k in series]
    for i, layer in enumerate(series):
        means = np.array([data[m][layer]["mean"] for m in methods])
        stds = np.array([data[m][layer]["std"] for m in methods])
        pos = x + (i - 1) * width

        ax.bar(pos, means, width, yerr=stds, capsize=4, label=names[layer],
               color=colors[layer], edgecolor="black", linewidth=0.6,
               error_kw={"elinewidth": 1.0, "alpha": 0.8})

        for p, m, mean, std in zip(pos, methods, means, stds):
            txt = f"{mean:,.0f}"
            if base and m != "base":
                txt += f"\n{base[layer]['mean'] / mean:.2f}x"
            ax.text(p, mean + std + 0.01 * max(all_tops), txt,
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold" if layer == "ALL" else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m}\nacc {data[m]['n_pass']}/{data[m]['n']}" for m in methods]
    )
    ax.set_ylabel("Avg cycles per layer-update call (per timestep)")
    ax.set_title("Layer update cost per timestep by method\n"
                 "(weighted mean, error bars = std across samples, "
                 "label = speedup vs base)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(title="Series")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)

    # Secondary axis in milliseconds
    to_ms = lambda c: c / clk * 1e3
    from_ms = lambda ms: ms * clk / 1e3
    sec = ax.secondary_yaxis("right", functions=(to_ms, from_ms))
    sec.set_ylabel(f"Time per call [ms] @ {clk / 1e6:.0f} MHz")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")
    return fig


def plot_total(data, clk, out_path):
    methods = list(data.keys())
    x = np.arange(len(methods))
    base = data.get("base")

    # total cycles per sample (L1 + L2), plus mean T for context
    totals = {}
    for m in methods:
        t = np.array([s["L1"]["total"] + s["L2"]["total"] for s in data[m]["samples"]],
                     dtype=float)
        totals[m] = t

    fig, ax = plt.subplots(figsize=(8, 5.5))
    means = np.array([totals[m].mean() for m in methods])
    stds = np.array([totals[m].std(ddof=1) for m in methods])

    ax.bar(x, means, 0.55, yerr=stds, capsize=4, color="#55A868",
           edgecolor="black", linewidth=0.6)
    for xi, m, mean, std in zip(x, methods, means, stds):
        txt = f"{mean:,.0f}"
        if base and m != "base":
            txt += f"\n{totals['base'].mean() / mean:.2f}x"
        ax.text(xi, mean + std + 0.01 * means.max(), txt,
                ha="center", va="bottom", fontsize=8)

    mean_T = np.mean([s["T"] for s in data[methods[0]]["samples"]])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Total cycles per sample (L1 + L2)")
    ax.set_title(f"Total cycles per inference (mean T = {mean_T:.0f} timesteps)\n"
                 "error bars = std across samples; depends on sample length")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)

    to_ms = lambda c: c / clk * 1e3
    from_ms = lambda ms: ms * clk / 1e3
    sec = ax.secondary_yaxis("right", functions=(to_ms, from_ms))
    sec.set_ylabel(f"Time per sample [ms] @ {clk / 1e6:.0f} MHz")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=".",
                    help="folder containing the *_test_60_samples.txt files")
    ap.add_argument("--out", default="cycles_per_call_comparison.png",
                    help="output PNG for the main graph")
    ap.add_argument("--with-total", action="store_true",
                    help="also create the total-cycles-per-sample figure")
    ap.add_argument("--no-show", action="store_true",
                    help="save figures without opening a window")
    args = ap.parse_args()

    data, clk = load_all(Path(args.folder))
    if not data:
        sys.exit("No data loaded. Check the folder and file names.")

    print_summary(data)
    plot_per_call(data, clk, args.out)

    if args.with_total:
        out2 = Path(args.out).with_name(Path(args.out).stem + "_total.png")
        plot_total(data, clk, out2)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()