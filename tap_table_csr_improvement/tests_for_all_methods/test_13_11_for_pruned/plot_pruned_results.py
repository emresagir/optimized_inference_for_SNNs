#!/usr/bin/env python3
"""Thesis figure + LaTeX table for the pruned-network tests.

Runs (all CSR method, 694 samples, same firmware settings):
  dense  : the dense model                                  (CSR_table_694_previous_model_not_fine_tuned.txt)
  full   : the pruned model with the FULL CSR table          (CSR_table_pruned_but_full_CSR_table.txt)
  pruned : the pruned model with the PRUNED CSR table        (CSR_table_pruned_694.txt)

"full" vs "pruned" is the clean comparison: same network, same spikes, only the table differs.
The dense model is shown for context only (different weights -> different spike activity).

Outputs (in this folder):
  thesis_pruned_cycles.pdf/.png  mean cycles per timestep per layer (dense model, pruned model full/pruned table)
  thesis_pruned_table.tex        LaTeX table (booktabs)

Usage: python3 plot_pruned_results.py
"""
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RUNS = {
    "dense":  HERE / "CSR_table_694_previous_model_not_fine_tuned.txt",
    "full":   HERE / "CSR_table_pruned_but_full_CSR_table.txt",
    "pruned": HERE / "CSR_table_pruned_694.txt",
}
LABELS = {"dense": "dense model", "full": "pruned model, full table", "pruned": "pruned model, pruned table"}
# Floating-point accuracy on the same 694 samples, real timesteps only (golden reference, no padding):
FLOAT_CORRECT = {"dense": 468, "full": 490, "pruned": 490}
FULL_CONN = {"L1": 12544, "L2": 10368}           # 392*2*4*4 and 144*8*3*3 (no connection removed)
ROW_PTR_BYTES = {"L1": (200 + 1) * 2, "L2": (392 + 1) * 2}

SAMPLE = re.compile(r"^\[(\d+)\]\s+T=\s*(\d+)\s+GT=(\d+)\s+Pred=(\d+)\s+(PASS|FAIL)\s+spikes:\s*([\d ]+)")
CYC = re.compile(r"^(L\d) layer update cycles:\s+calls=(\d+).*total=(\d+)")
CONN = re.compile(r"CONN=L1:(\d+),L2:(\d+)")


def parse(path):
    rows, cyc, cur, conn = {}, {}, None, dict(FULL_CONN)
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        m = CONN.search(line)
        if m:
            conn = {"L1": int(m.group(1)), "L2": int(m.group(2))}
        m = SAMPLE.match(line)
        if m:
            cur = int(m.group(1))
            rows[cur] = dict(gt=int(m.group(3)), pred=int(m.group(4)), spikes=np.array(m.group(6).split(), int))
            continue
        m = CYC.match(line)
        if m and cur is not None and int(m.group(2)) > 0:      # stats of sample `cur` (printed after it)
            cyc.setdefault(cur, {})[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return rows, cyc, conn


data = {k: parse(p) for k, p in RUNS.items()}
N = len(data["pruned"][0])

# ---- bit-exactness: full vs pruned table must give identical outputs ----
rf, rp = data["full"][0], data["pruned"][0]
identical = sum(np.array_equal(rf[i]["spikes"], rp[i]["spikes"]) and rf[i]["pred"] == rp[i]["pred"] for i in rf)
print(f"full vs pruned table: identical spike vectors and predictions in {identical}/{N} samples")

# ---- per-layer cycles: weighted mean (total cycles / total calls) and per-sample means ----
stats = {}
for k, (rows, cyc, conn) in data.items():
    ks = sorted(cyc)
    s = {"acc": sum(r["gt"] == r["pred"] for r in rows.values()), "conn": conn}
    for L in ("L1", "L2"):
        s[L] = sum(cyc[i][L][1] for i in ks) / sum(cyc[i][L][0] for i in ks)
        s[L + "_ps"] = {i: cyc[i][L][1] / cyc[i][L][0] for i in ks}
    stats[k] = s
for k, s in stats.items():
    print(f"{LABELS[k]:28s} acc {s['acc']}/{N}  L1 {s['L1']:7.0f}  L2 {s['L2']:7.0f} cycles/timestep  "
          f"connections L1 {s['conn']['L1']}  L2 {s['conn']['L2']}")
for L in ("L1", "L2"):
    a, b = stats["full"][L], stats["pruned"][L]
    r = stats["pruned"]["conn"][L] / stats["full"]["conn"][L]
    fixed = (b - r * a) / (1 - r)                 # two-point estimate of the table-independent part
    print(f"{L}: {100 * (1 - b / a):.1f}% fewer cycles (speed-up {a / b:.2f}x), "
          f"{100 * (1 - r):.1f}% fewer connections; table-independent part ~{fixed:.0f} of {a:.0f} cycles")

# ---- LaTeX table ----
def tbytes(conn, L):
    return ROW_PTR_BYTES[L] + 2 * 2 * conn[L]     # row_ptr + out_idx + weight_idx (uint16)

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{CSR method on the dense and the pruned network (694 test samples). The pruned network",
    r"is run with the full and with the pruned CSR table; both give identical outputs.}",
    r"\label{tab:pruned_results}",
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r" & Dense model & \multicolumn{2}{c}{Pruned model} \\",
    r"\cmidrule(lr){3-4}",
    r" & & Full table & Pruned table \\",
    r"\midrule",
]
def row(name, f):
    return f"{name} & " + " & ".join(f(k) for k in ("dense", "full", "pruned")) + r" \\"
lines += [
    row("Connections Conv 1", lambda k: f"{stats[k]['conn']['L1']:,}".replace(",", r"\,")),
    row("Connections Conv 2", lambda k: f"{stats[k]['conn']['L2']:,}".replace(",", r"\,")),
    row(r"CSR table size [B]", lambda k: f"{tbytes(stats[k]['conn'], 'L1') + tbytes(stats[k]['conn'], 'L2'):,}".replace(",", r"\,")),
    row(r"Accuracy, float [\%]", lambda k: f"{100 * FLOAT_CORRECT[k] / N:.2f}"),
    row(r"Accuracy, MCU [\%]", lambda k: f"{100 * stats[k]['acc'] / N:.2f}"),
    row("Conv 1 cycles / timestep", lambda k: f"{stats[k]['L1']:,.0f}".replace(",", r"\,")),
    row("Conv 2 cycles / timestep", lambda k: f"{stats[k]['L2']:,.0f}".replace(",", r"\,")),
    r"\midrule",
    row("Speed-up Conv 1 (vs.\\ full table)", lambda k: "--" if k != "pruned" else f"{stats['full']['L1'] / stats['pruned']['L1']:.2f}$\\times$"),
    row("Speed-up Conv 2 (vs.\\ full table)", lambda k: "--" if k != "pruned" else f"{stats['full']['L2'] / stats['pruned']['L2']:.2f}$\\times$"),
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
(HERE / "thesis_pruned_table.tex").write_text("\n".join(lines) + "\n")
print("[SAVED]", HERE / "thesis_pruned_table.tex", "(needs \\usepackage{booktabs})")

# ---- figure ----
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"dense": "#c8c8c8", "full": "#1f4e9c", "pruned": "#e07b00"}
fig, ax = plt.subplots(figsize=(5.0, 2.8))
BAR_LABELS = {"dense": "dense model", "full": "pruned model, full table", "pruned": "pruned model, pruned table"}

# mean cycles per timestep per layer
x = np.arange(2); w = 0.26
for j, k in enumerate(("dense", "full", "pruned")):
    vals = [stats[k]["L1"], stats[k]["L2"]]
    ax.bar(x + (j - 1) * w, vals, w, color=C[k], edgecolor="0.3" if k == "dense" else C[k],
           hatch="///" if k == "dense" else None, lw=0.5, label=BAR_LABELS[k])
for i, L in enumerate(("L1", "L2")):           # reduction full -> pruned table, above that pair of bars
    red = 100 * (1 - stats["pruned"][L] / stats["full"][L])
    top = max(stats["full"][L], stats["pruned"][L])
    ax.plot([i, i, i + w, i + w], [top * 1.03, top * 1.06, top * 1.06, stats["pruned"][L] * 1.03],
            color="0.25", lw=0.7)
    ax.text(i + w / 2, top * 1.075, f"$-${red:.1f}%", ha="center", va="bottom", fontsize=8)
# x labels also give the fraction of removed connections, to compare with the cycle reduction
removed = {L: 100 * (1 - stats["pruned"]["conn"][L] / stats["full"]["conn"][L]) for L in ("L1", "L2")}
ax.set_xticks(x, [f"Conv 1\n({removed['L1']:.1f}% connections removed)",
                  f"Conv 2\n({removed['L2']:.1f}% connections removed)"])
ax.set_ylabel("Cycles per timestep")
ax.set_ylim(0, max(stats[k][L] for k in stats for L in ("L1", "L2")) * 1.22)
ax.grid(True, axis="y", ls=":", lw=0.4, color="0.8"); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
          handlelength=1.4, columnspacing=1.0, fontsize=7.5)

fig.savefig(HERE / "thesis_pruned_cycles.pdf")
fig.savefig(HERE / "thesis_pruned_cycles.png", dpi=200)
print("[SAVED]", HERE / "thesis_pruned_cycles.pdf", "(+ .png preview)")
