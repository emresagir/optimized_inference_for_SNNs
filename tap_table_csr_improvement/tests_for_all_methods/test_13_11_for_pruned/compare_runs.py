#!/usr/bin/env python3
"""Compare two STM32 run logs (per-sample predictions + output spike counts) and plot the differences.

Usage: python compare_runs.py [run_a.txt] [run_b.txt]
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
A = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "CSR_table_694_previous_model_not_fine_tuned.txt"
B = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "CSR_table_pruned_694.txt"

SAMPLE = re.compile(r"^\[(\d+)\]\s+T=\s*(\d+)\s+GT=(\d+)\s+Pred=(\d+)\s+(PASS|FAIL)\s+spikes:\s*([\d ]+)")
CYC = re.compile(r"^(L\d) layer update cycles: calls=(\d+) .*avg=(\d+) total=(\d+)")


def parse(path):
    rows, cyc = {}, {"L1": [], "L2": []}
    for line in path.read_text(errors="ignore").splitlines():
        m = SAMPLE.match(line)
        if m:
            i, T, gt, pr, st, sp = m.groups()
            rows[int(i)] = (int(T), int(gt), int(pr), st == "PASS", np.array(sp.split(), dtype=int))
            continue
        m = CYC.match(line)
        if m and int(m.group(2)) > 0:
            cyc[m.group(1)].append((int(m.group(2)), int(m.group(4))))   # (calls, total) of one sample
    return rows, cyc


ra, ca = parse(A)
rb, cb = parse(B)
idx = sorted(set(ra) & set(rb))
n = len(idx)
gt = np.array([ra[i][1] for i in idx])
pa = np.array([ra[i][2] for i in idx]); pb = np.array([rb[i][2] for i in idx])
oka = np.array([ra[i][3] for i in idx]); okb = np.array([rb[i][3] for i in idx])
sa = np.stack([ra[i][4] for i in idx]); sb = np.stack([rb[i][4] for i in idx])
d = sb - sa

na, nb = A.stem, B.stem
print(f"A = {na}\nB = {nb}\ncommon samples: {n}")
print(f"accuracy  A: {oka.sum()}/{n} ({100*oka.mean():.2f}%)   B: {okb.sum()}/{n} ({100*okb.mean():.2f}%)   "
      f"delta: {100*(okb.mean()-oka.mean()):+.2f} pp")
print(f"same prediction: {(pa == pb).sum()}/{n} ({100*(pa == pb).mean():.2f}%)")
print(f"  A PASS -> B FAIL: {(oka & ~okb).sum()}   A FAIL -> B PASS: {(~oka & okb).sum()}   "
      f"both PASS: {(oka & okb).sum()}   both FAIL: {(~oka & ~okb).sum()}")
print(f"total output spikes  A: {sa.sum()}  B: {sb.sum()}  ({100*(sb.sum()/max(sa.sum(),1)-1):+.2f}%)")
print(f"spike diff (B-A) per class-neuron: mean={d.mean():+.3f}  mean|d|={np.abs(d).mean():.3f}  max|d|={np.abs(d).max()}")
print(f"identical spike vectors: {(d == 0).all(1).sum()}/{n}")
print("per-class mean spikes  A:", np.round(sa.mean(0), 2))
print("per-class mean spikes  B:", np.round(sb.mean(0), 2))
print("per-class accuracy (A -> B):")
for c in range(10):
    m = gt == c
    if m.any():
        print(f"  class {c}: {100*oka[m].mean():6.2f}% -> {100*okb[m].mean():6.2f}%  ({100*(okb[m].mean()-oka[m].mean()):+.2f} pp, n={m.sum()})")
for L in ("L1", "L2"):
    if ca[L] and cb[L]:
        # weighted mean over all timesteps: total cycles / total calls (as compare_snn_cycles.py)
        a = sum(t for _, t in ca[L]) / sum(c for c, _ in ca[L])
        b = sum(t for _, t in cb[L]) / sum(c for c, _ in cb[L])
        print(f"{L} avg cycles/step  A: {a:.0f}  B: {b:.0f}  ({100*(b/a-1):+.2f}%)")
    elif ca[L] or cb[L]:
        print(f"{L} cycles only present in {'A' if ca[L] else 'B'} log")

flip_ab = [i for i, x in zip(idx, oka & ~okb) if x]
flip_ba = [i for i, x in zip(idx, ~oka & okb) if x]
print("A PASS -> B FAIL samples:", flip_ab)
print("A FAIL -> B PASS samples:", flip_ba)

# ---- plots ----
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(f"{nb} vs {na}  (B - A)")

ax[0, 0].bar(["A PASS→B FAIL", "A FAIL→B PASS", "both PASS", "both FAIL"],
             [(oka & ~okb).sum(), (~oka & okb).sum(), (oka & okb).sum(), (~oka & ~okb).sum()],
             color=["#d62728", "#2ca02c", "#7f7f7f", "#bcbcbc"])
ax[0, 0].set_title(f"Outcome transitions  (acc {100*oka.mean():.2f}% → {100*okb.mean():.2f}%)")
ax[0, 0].set_ylabel("samples")

acc_a = [100 * oka[gt == c].mean() if (gt == c).any() else 0 for c in range(10)]
acc_b = [100 * okb[gt == c].mean() if (gt == c).any() else 0 for c in range(10)]
x = np.arange(10); w = 0.4
ax[0, 1].bar(x - w/2, acc_a, w, label=na)
ax[0, 1].bar(x + w/2, acc_b, w, label=nb)
ax[0, 1].set_xticks(x); ax[0, 1].set_xlabel("class"); ax[0, 1].set_ylabel("accuracy %")
ax[0, 1].set_title("Per-class accuracy"); ax[0, 1].legend(fontsize=8)

ax[1, 0].bar(x - w/2, sa.mean(0), w, label=na)
ax[1, 0].bar(x + w/2, sb.mean(0), w, label=nb)
ax[1, 0].set_xticks(x); ax[1, 0].set_xlabel("output neuron"); ax[1, 0].set_ylabel("mean spikes / sample")
ax[1, 0].set_title("Mean output spike count"); ax[1, 0].legend(fontsize=8)

tot = d.sum(1)
ax[1, 1].plot(idx, tot, lw=0.7, color="#1f77b4")
ax[1, 1].axhline(0, color="k", lw=0.5)
ax[1, 1].axhline(tot.mean(), color="r", ls="--", lw=0.8, label=f"mean {tot.mean():+.1f}")
ax[1, 1].set_xlabel("sample"); ax[1, 1].set_ylabel("Σ spikes (B - A)")
ax[1, 1].set_title("Total output spike difference per sample"); ax[1, 1].legend(fontsize=8)

fig.tight_layout()
out = HERE / f"diff_{na}_vs_{nb}.png"
fig.savefig(out, dpi=130)
print(f"saved plot: {out}")
if "--show" in sys.argv:
    plt.show()
