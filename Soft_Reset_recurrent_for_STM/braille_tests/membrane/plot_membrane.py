#!/usr/bin/env python3
"""Thesis figures: soft vs hard reset on the MCU, same network and input (test sample 0).

Logs from the for-soft-reset project with Delvecchio's network (SNN_Print_Membrane, s = 60), only the reset in
SNN_Run_Timestep changed:
  simone_network_membrane_soft_reset.txt, simone_network_membrane_hard_reset.txt
  (traced hidden neuron and output neuron 1 on every line)
Zoomed to timesteps 115-170, where the two resets differ most.
Output: thesis_braille_reset_hidden.pdf/.png next to this script.
"""
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SCALE = 60.0
T0, T1 = 115, 170
MEM_RE = re.compile(r"MEM t=(\d+) H(\d+) V:(-?\d+) S:(\d) O(\d+) V:(-?\d+) S:(\d)")


def load(log):
    a = np.array([list(map(int, m.groups())) for m in map(MEM_RE.match, open(HERE / log)) if m])
    q = SCALE / 32768.0
    return {"hidden": (a[:, 2] * q, a[:, 3] > 0, a[0, 1]), "output": (a[:, 5] * q, a[:, 6] > 0, a[0, 4])}


soft, hard = load("simone_network_membrane_soft_reset.txt"), load("simone_network_membrane_hard_reset.txt")

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.1, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
c_soft, c_hard = "#1f4e9c", "#e07b00"

for key in ("hidden",):   # output layer left out: its input (hidden spikes) differs between the two runs
    (u_s, s_s, n), (u_h, s_h, n_h) = soft[key], hard[key]
    assert n == n_h, "both logs must trace the same neuron"
    t = np.arange(len(u_s))
    w = (t >= T0) & (t <= T1)
    print(f"{key} neuron {n}: spikes soft/hard = {s_s.sum()}/{s_h.sum()} (all), "
          f"{s_s[w].sum()}/{s_h[w].sum()} (timesteps {T0}-{T1})")
    top = max(u_s[w].max(), u_h[w].max(), 1.0)
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    ax.plot(t[w], u_s[w], color=c_soft, marker="o", ms=2.2, label="Soft reset")
    ax.plot(t[w], u_h[w], color=c_hard, ls="--", marker="o", ms=2.2, label="Hard reset")
    ax.axhline(1.0, color="0.35", ls=":", lw=0.8, label=r"threshold $U_{\mathrm{thr}}$")
    ax.plot(t[w & s_s], np.full((w & s_s).sum(), top + 0.25), "|", color=c_soft, ms=7)
    ax.plot(t[w & s_h], np.full((w & s_h).sum(), top + 0.12), "|", color=c_hard, ms=7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"$U$")
    ax.set_xlim(T0 - 0.5, T1 + 0.5)
    ax.grid(True, ls=":", lw=0.4, color="0.8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, handlelength=2.2,
              columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"thesis_braille_reset_{key}.{ext}", dpi=200)
    plt.close(fig)
    print("[SAVED]", HERE / f"thesis_braille_reset_{key}.pdf", "(+ .png preview)")
