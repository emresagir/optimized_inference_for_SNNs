#!/usr/bin/env python3
"""Golden reference for the recurrent Braille network (12 -> 38 recurrent -> 7) on the 140 test samples.

Same model and settings as the board: retrained_snntorch_20260404_185209.pt, soft reset with
reset_delay=False, no bias, 256 steps. Prediction = output neuron with the most spikes; ties go to the
lowest index (same as the MCU and torch.argmax).

Usage:  python3 golden_ref_recurrent.py [--hard | --delvecchio | --delvecchio-soft] [mcu_log]
        --hard:       hard-reset model retrained_snntorch_20260404_172148.pt (reset to zero, reset_delay=False).
        --delvecchio: Delvecchio's model delvecchio_retrained_snntorch_20251007_161214.pt (hard reset,
                      one recurrent weight per neuron), copied from TesiSNN-main/NIR-to-C.
        --delvecchio-soft: the same model, not retrained, run with the soft reset (subtract, reset_delay=True).
        With an MCU log from test_sender.py, it also prints the agreement per sample.
Output: printed only.
"""
import re
import sys
from pathlib import Path

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                        # Soft_Reset_recurrent_for_STM: models and ds_test.pt
MODES = {"": (ROOT / "retrained_snntorch_20260404_185209.pt", "subtract", ""),
         "--hard": (ROOT / "retrained_snntorch_20260404_172148.pt", "zero", "_hardreset"),
         "--delvecchio": (HERE / "delvecchio_retrained_snntorch_20251007_161214.pt", "zero", "_delvecchio"),
         "--delvecchio-soft": (HERE / "delvecchio_retrained_snntorch_20251007_161214.pt", "subtract", "_delvecchio_soft")}
mode = next((a for a in sys.argv if a in MODES), "")
sys.argv = [a for a in sys.argv if a not in MODES]
model_path, RESET, tag = MODES[mode]
MODEL = model_path.name
sd = torch.load(model_path, map_location="cpu")
ds = torch.load(ROOT / "ds_test.pt", weights_only=False)
X = torch.stack([ds[i][0] for i in range(len(ds))]).float().swapaxes(0, 1)   # (256, 140, 12)
Y = np.array([int(ds[i][1]) for i in range(len(ds))])

fc1 = nn.Linear(12, 38, bias=False)
fc2 = nn.Linear(38, 7, bias=False)
fc1.weight.data = sd["fc1.weight"]
fc2.weight.data = sd["fc2.weight"]
# MCU kernels: hard reset = same step (reset_delay=False); soft reset = delayed, like ST-MNIST (reset_delay=True)
DELAY = RESET == "subtract"
lif1 = snn.RLeaky(beta=0.9, threshold=1.0, reset_mechanism=RESET, reset_delay=DELAY,
                  all_to_all=False, V=sd["lif1.V"], learn_recurrent=False)
lif2 = snn.Leaky(beta=0.55, threshold=1.0, reset_mechanism=RESET, reset_delay=DELAY)

with torch.no_grad():
    spk1, mem1 = lif1.init_rleaky()
    mem2 = lif2.init_leaky()
    counts = 0
    for t in range(X.shape[0]):
        spk1, mem1 = lif1(fc1(X[t]), spk1, mem1)
        spk2, mem2 = lif2(fc2(spk1), mem2)
        counts = counts + spk2
pred = counts.argmax(1).numpy()
print(f"{MODEL} ({RESET}) Python (float): {(pred == Y).sum()}/{len(Y)} = {100 * (pred == Y).mean():.2f}%")
print("Python misses:", [i for i in range(len(Y)) if pred[i] != Y[i]])

if len(sys.argv) > 1:
    mcu = {}
    for line in open(sys.argv[1], errors="ignore"):
        m = re.search(r"Sample\s+(\d+)/\d+: Pred=(\d+), Exp=(\d+)", line)
        if m:
            mcu[int(m.group(1)) - 1] = int(m.group(2))
    idx = sorted(mcu)
    p_mcu = np.array([mcu[i] for i in idx])
    print(f"MCU: {(p_mcu == Y[idx]).sum()}/{len(idx)} = {100 * (p_mcu == Y[idx]).mean():.2f}%")
    print("MCU misses:", [i for i in idx if mcu[i] != Y[i]])
    diff = [i for i in idx if mcu[i] != pred[i]]
    print(f"Agreement MCU vs Python: {len(idx) - len(diff)}/{len(idx)} = {100 * (1 - len(diff) / len(idx)):.1f}%")
    print("Different predictions (sample: MCU / Python / label):",
          [(i, mcu[i], int(pred[i]), int(Y[i])) for i in diff])
