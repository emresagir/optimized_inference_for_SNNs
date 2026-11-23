#!/usr/bin/env python3
"""Thesis figure: soft reset, MCU (Q15) vs snnTorch golden reference (float), traced hidden neuron, test sample 0.

MCU log: simone_network_membrane_soft_reset.txt (for-soft-reset project, SNN_Print_Membrane, s = 60).
Reference: Delvecchio's network (delvecchio_retrained_snntorch_20251007_161214.pt, not retrained),
reset_mechanism="subtract", reset_delay=True (same delayed reset as the MCU kernel).
Same style as the ST-MNIST membrane figures (overlay, spike raster, difference).
Output: thesis_braille_mcu_vs_ref.pdf/.png next to this script.
"""
import re
from pathlib import Path

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # Soft_Reset_recurrent_for_STM
MODEL, SAMPLE, SCALE = HERE.parent / "delvecchio_retrained_snntorch_20251007_161214.pt", 0, 60.0
MEM_RE = re.compile(r"MEM t=(\d+) H(\d+) V:(-?\d+) S:(\d)")

# MCU trace
a = np.array([list(map(int, m.groups())) for m in map(MEM_RE.match, open(HERE / "simone_network_membrane_soft_reset.txt")) if m])
u_m, s_m, n = a[:, 2] * SCALE / 32768.0, a[:, 3] > 0, a[0, 1]

# snnTorch reference, same neuron and sample
sd = torch.load(MODEL, map_location="cpu")
fc1 = nn.Linear(12, 38, bias=False)
fc1.weight.data = sd["fc1.weight"]
lif1 = snn.RLeaky(beta=0.9, threshold=1.0, reset_mechanism="subtract", reset_delay=True,
                  all_to_all=False, V=sd["lif1.V"], learn_recurrent=False)
x = torch.load(ROOT / "ds_test.pt", weights_only=False)[SAMPLE][0].float()
u_r, s_r = [], []
with torch.no_grad():
    spk, mem = lif1.init_rleaky()
    for t in range(x.shape[0]):
        spk, mem = lif1(fc1(x[t:t + 1]), spk, mem)
        u_r.append(float(mem[0, n]))
        s_r.append(bool(spk[0, n]))
u_r, s_r = np.array(u_r), np.array(s_r)
print(f"hidden neuron {n}: max|dU| = {np.abs(u_m - u_r).max():.4f}, mean|dU| = {np.abs(u_m - u_r).mean():.4f}, "
      f"spikes snnTorch/MCU = {s_r.sum()}/{s_m.sum()}, different spike decisions = {(s_r != s_m).sum()}")

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.1, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
c_ref, c_mcu, c_diff, c_miss = "#1f4e9c", "#e07b00", "#444444", "#f2c9c9"
t = np.arange(len(u_m))
fig, (ax_u, ax_s, ax_d) = plt.subplots(3, 1, figsize=(6.3, 3.3), sharex=True,
                                       gridspec_kw={"height_ratios": [2.2, 0.55, 1.0], "hspace": 0.12})
ax_u.plot(t, u_r, color=c_ref)
ax_u.plot(t, u_m, color=c_mcu, ls="--")
ax_u.axhline(1.0, color="0.35", ls=":", lw=0.8)
ax_u.set_ylabel(r"$U$")
for xx in t[s_r != s_m]:
    ax_s.axvspan(xx - 0.5, xx + 0.5, color=c_miss, lw=0)
ax_s.vlines(t[s_r], 1.6, 2.4, color=c_ref, lw=1.0)
ax_s.vlines(t[s_m], 0.6, 1.4, color=c_mcu, lw=1.0)
ax_s.set_ylim(0.3, 2.7)
ax_s.set_yticks([2, 1], ["ref", "MCU"])
ax_s.tick_params(axis="y", length=0)
ax_d.axhline(0, color="0.6", lw=0.6)
ax_d.plot(t, u_m - u_r, color=c_diff, lw=0.9)
ax_d.set_ylabel(r"$U_{\mathrm{MCU}}-U_{\mathrm{ref}}$")
ax_d.set_xlabel("Timestep")
for ax in (ax_u, ax_s, ax_d):
    ax.grid(True, ls=":", lw=0.4, color="0.8")
    ax.set_xlim(t[0] - 0.5, t[-1] + 0.5)
    ax.spines[["top", "right"]].set_visible(False)
ax_s.grid(False)
handles = [Line2D([], [], color=c_ref, label="snnTorch (float)"),
           Line2D([], [], color=c_mcu, ls="--", label="MCU (Q15)"),
           Line2D([], [], color="0.35", ls=":", lw=0.8, label=r"threshold $U_{\mathrm{thr}}$"),
           Patch(color=c_miss, label="spike decision differs")]
ax_u.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
            handlelength=1.8, columnspacing=1.2)
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"thesis_braille_mcu_vs_ref.{ext}", dpi=200)
print("[SAVED]", HERE / "thesis_braille_mcu_vs_ref.pdf", "(+ .png preview)")
