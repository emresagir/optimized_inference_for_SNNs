#!/usr/bin/env python3
"""Render presentation-ready comparison tables (PNG, transparent-free white bg, 300 dpi).

Usage: python make_table.py [run_a.txt] [run_b.txt]
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
A = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "CSR_table_694_previous_model_not_fine_tuned.txt"
B = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "CSR_table_pruned_694.txt"
LABEL_A, LABEL_B = "CSR (dense)", "CSR (pruned)"

SAMPLE = re.compile(r"^\[(\d+)\]\s+T=\s*\d+\s+GT=(\d+)\s+Pred=(\d+)\s+(PASS|FAIL)\s+spikes:\s*([\d ]+)")


def parse(p):
    r = {}
    for line in p.read_text(errors="ignore").splitlines():
        m = SAMPLE.match(line)
        if m:
            i, gt, pr, st, sp = m.groups()
            r[int(i)] = (int(gt), int(pr), st == "PASS", np.array(sp.split(), dtype=int))
    return r


ra, rb = parse(A), parse(B)
idx = sorted(set(ra) & set(rb)); n = len(idx)
gt = np.array([ra[i][0] for i in idx])
pa = np.array([ra[i][1] for i in idx]); pb = np.array([rb[i][1] for i in idx])
oka = np.array([ra[i][2] for i in idx]); okb = np.array([rb[i][2] for i in idx])
sa = np.stack([ra[i][3] for i in idx]); sb = np.stack([rb[i][3] for i in idx])

HEAD_BG, HEAD_FG = "#1f3b57", "white"
ROW_BG = ["#ffffff", "#f2f5f8"]
GOOD, BAD, NEUTRAL = "#1a7f37", "#c62828", "#333333"


def delta_color(v, higher_is_better=True):
    if abs(v) < 1e-9:
        return NEUTRAL
    return GOOD if (v > 0) == higher_is_better else BAD


def draw(rows, header, colors, fname, col_widths, title=None):
    nr = len(rows) + 1
    fig_h = 0.42 * nr + (0.45 if title else 0.1)
    fig, ax = plt.subplots(figsize=(sum(col_widths), fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=[header] + rows, colWidths=[w / sum(col_widths) for w in col_widths],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d0d7de"); cell.set_linewidth(0.8)
        cell.set_fontsize(12)
        if r == 0:
            cell.set_facecolor(HEAD_BG); cell.get_text().set_color(HEAD_FG); cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(ROW_BG[(r - 1) % 2])
            if c == 0:
                cell.get_text().set_ha("left")
            col = colors[r - 1][c]
            if col:
                cell.get_text().set_color(col); cell.get_text().set_weight("bold")
    if title:
        ax.set_title(title, fontsize=14, weight="bold", pad=10)
    out = HERE / fname
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


# ---- summary table ----
acc_a, acc_b = 100 * oka.mean(), 100 * okb.mean()
spk_a, spk_b = sa.sum(1).mean(), sb.sum(1).mean()
rows = [
    ["Accuracy", f"{acc_a:.2f} %", f"{acc_b:.2f} %", f"{acc_b - acc_a:+.2f} pp"],
    ["Correct samples", f"{oka.sum()} / {n}", f"{okb.sum()} / {n}", f"{okb.sum() - oka.sum():+d}"],
    ["Avg. output spikes / sample", f"{spk_a:.1f}", f"{spk_b:.1f}", f"{100 * (spk_b / spk_a - 1):+.2f} %"],
    ["Same prediction", "", f"{(pa == pb).sum()} / {n}", f"{100 * (pa == pb).mean():.1f} %"],
    ["PASS → FAIL", "", f"{(oka & ~okb).sum()}", ""],
    ["FAIL → PASS", "", f"{(~oka & okb).sum()}", ""],
]
colors = [[None] * 4 for _ in rows]
colors[0][3] = delta_color(acc_b - acc_a)
colors[1][3] = delta_color(okb.sum() - oka.sum())
colors[2][3] = NEUTRAL
colors[4][2] = BAD
colors[5][2] = GOOD
draw(rows, ["Metric", LABEL_A, LABEL_B, "Δ"], colors, "table_summary.png", [3.4, 2.0, 2.0, 1.5],
     title=f"{LABEL_A} vs {LABEL_B}  (N = {n})")

# ---- per-class accuracy table ----
rows, colors = [], []
for c in range(10):
    m = gt == c
    a, b = 100 * oka[m].mean(), 100 * okb[m].mean()
    rows.append([f"{c}", f"{m.sum()}", f"{a:.1f} %", f"{b:.1f} %", f"{b - a:+.1f} pp"])
    colors.append([None, None, None, None, delta_color(b - a)])
rows.append(["All", f"{n}", f"{acc_a:.2f} %", f"{acc_b:.2f} %", f"{acc_b - acc_a:+.2f} pp"])
colors.append([None, None, None, None, delta_color(acc_b - acc_a)])
draw(rows, ["Class", "N", LABEL_A, LABEL_B, "Δ"], colors, "table_per_class.png", [1.0, 0.8, 2.0, 2.0, 1.4],
     title="Per-class accuracy")
