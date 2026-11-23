#!/usr/bin/env python3
"""MAC count of the tap table method: MCU log vs golden reference (Python float model).

MCU     : the firmware (MAC_COUNTER 1) prints, for every sample, the MACs per timestep of each conv layer:
              [k] T= 62  GT=8  Pred=8 ...
               Layer 1 average mac count of the sample = 1650
               Layer 2 average mac count of the sample = 6462
          The two lines after [k] belong to sample k (printed at the reset before the next sample),
          so the last sample has none. The value is total/T, truncated to an integer.
Python  : the same test samples run through the float model. A MAC happens for every
          (active input, output neuron it reaches) pair, so the MACs of a layer are
          conv2d(input != 0, ones kernel).sum() -- exactly what the tap table does.
Dense   : the MACs of the basic method, every output neuron x (C_in * Kh * Kw), every timestep.

Averages are weighted by the timesteps: sum(MACs) / sum(T) over the samples present in the log.

Usage:  python3 MAC_count.py [big|small] [log]      (default: big, tap_694_big_mac.txt)
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tonic
import snntorch as snn
from snntorch import utils

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / "SCNN_STMNIST" / "cache" / "stmnist" / "test"

NETWORKS = {
    # Main network: 2 -> 8 -> 16 channels
    "small": dict(C1=8, C2=16,
                  model=ROOT / "opt_SCNN_MNSIT" / "best_scnn_net_20260817_171949_acc83.4.pth",
                  log=HERE / "tap_694_small_mac.txt"),
    # Larger network: 2 -> 32 -> 64 channels
    "big": dict(C1=32, C2=64,
                model=ROOT / "tap_table_csr_improvement" / "bigger_network" / "best_scnn_net_20260426_104230_acc91.7.pth",
                log=HERE / "tap_694_big_mac.txt"),
}
NET = sys.argv[1] if len(sys.argv) > 1 else "big"
if NET not in NETWORKS:
    sys.exit(f"unknown network '{NET}', use one of {list(NETWORKS)}")
CFG = NETWORKS[NET]
C1, C2 = CFG["C1"], CFG["C2"]
LOG = Path(sys.argv[2]) if len(sys.argv) > 2 else CFG["log"]
N = 694

# Geometry (same for both networks): 2x10x10 input, Conv1 4x4 s1 -> 7x7, Conv2 3x3 s2 -> 3x3
IN_C, IN_HW = 2, 10
K1, S1, H1 = 4, 1, 7
K2, S2, H2 = 3, 2, 3
DENSE = {1: C1 * H1 * H1 * IN_C * K1 * K1,     # MACs per timestep of the basic method
         2: C2 * H2 * H2 * C1 * K2 * K2}
IN_SIZE = {1: IN_C * IN_HW * IN_HW, 2: C1 * H1 * H1}   # number of inputs of each conv layer

# ---------------------------------------------------------------- MCU log
SAMPLE = re.compile(r"^\[(\d+)\]\s+T=\s*(\d+)")
MAC = re.compile(r"Layer\s+(\d)\s+average mac count of the sample\s*=\s*(\d+)")

T_mcu, mac_mcu = {}, {1: {}, 2: {}}
cur = None
for line in LOG.read_text(errors="ignore").splitlines():
    line = line.strip()
    m = SAMPLE.match(line)
    if m:
        cur = int(m.group(1)); T_mcu[cur] = int(m.group(2))
        continue
    m = MAC.search(line)
    if m and cur is not None:
        mac_mcu[int(m.group(1))][cur] = int(m.group(2))

ids = sorted(k for k in T_mcu if k in mac_mcu[1] and k in mac_mcu[2])
if not ids:
    sys.exit(f"no MAC lines found in {LOG.name}")
print(f"Network: {NET} (2->{C1}->{C2})   log: {LOG.name}   samples with MAC counts: {len(ids)}/{len(T_mcu)}")
if all(mac_mcu[1][k] == 0 for k in ids) or all(mac_mcu[2][k] == 0 for k in ids):
    print("[WARN] a layer counted 0 MACs in every sample: check the out_ch test of MAC_COUNTER in the firmware")

# ---------------------------------------------------------------- Python (golden reference)
beta = 0.95
net = nn.Sequential(
    nn.Conv2d(IN_C, C1, K1, S1, bias=False), snn.Leaky(beta=beta, init_hidden=True, reset_mechanism="subtract"),
    nn.Conv2d(C1, C2, K2, S2, bias=False), snn.Leaky(beta=beta, init_hidden=True, reset_mechanism="subtract"),
    nn.Flatten(), nn.Linear(C2 * H2 * H2, 10, bias=False),
    snn.Leaky(beta=beta, init_hidden=True, output=True, reset_mechanism="subtract"))
net.load_state_dict(torch.load(CFG["model"], map_location="cpu")); net.eval()

ds = tonic.DiskCachedDataset(dataset=None, cache_path=str(CACHE))
xs = [torch.as_tensor(np.asarray(ds[i][0]), dtype=torch.float32) for i in range(N)]
L = np.array([x.shape[0] for x in xs])
TMAX = int(L.max())
X = torch.zeros(TMAX, N, IN_C, IN_HW, IN_HW)
for i, x in enumerate(xs):
    X[:x.shape[0], i] = x

ones1 = torch.ones(C1, IN_C, K1, K1)
ones2 = torch.ones(C2, C1, K2, K2)
mac_py = {1: np.zeros(N), 2: np.zeros(N)}      # total MACs per sample (real length only)
act_py = {1: np.zeros(N), 2: np.zeros(N)}      # total active inputs per sample
with torch.no_grad():
    utils.reset(net)
    for t in range(TMAX):
        live = torch.as_tensor(t < L, dtype=torch.float32)          # only the real timesteps count
        a1 = (X[t] != 0).float()                                     # Conv1 input: events
        spk1 = net[1](net[0](X[t]))                                  # Conv1 output spikes
        a2 = (spk1 != 0).float()                                     # Conv2 input
        for layer, a, ones, s in ((1, a1, ones1, S1), (2, a2, ones2, S2)):
            macs = F.conv2d(a, ones, stride=s).sum(dim=(1, 2, 3))
            mac_py[layer] += (macs * live).numpy()
            act_py[layer] += (a.sum(dim=(1, 2, 3)) * live).numpy()

if any(T_mcu[k] != L[k] for k in ids):
    print("[WARN] timesteps differ between the MCU log and the dataset for some samples")

# ---------------------------------------------------------------- comparison
idx = np.array(ids)
T_sum = L[idx].sum()
print(f"Timesteps: {T_sum} (sample {ids[0]}..{ids[-1]})\n")

hdr = f"{'':26s} | {'Conv 1':>14s} | {'Conv 2':>14s} | {'Both':>14s}"
print(hdr); print("-" * len(hdr))


def row(name, v1, v2, vb, fmt):
    print(f"{name:26s} | {fmt.format(v1):>14s} | {fmt.format(v2):>14s} | {fmt.format(vb):>14s}")


dense_b = DENSE[1] + DENSE[2]
mcu = {l: sum(mac_mcu[l][k] * T_mcu[k] for k in ids) / T_sum for l in (1, 2)}
py = {l: mac_py[l][idx].sum() / T_sum for l in (1, 2)}
act = {l: act_py[l][idx].sum() / T_sum / IN_SIZE[l] for l in (1, 2)}

row("Dense MACs / timestep", DENSE[1], DENSE[2], dense_b, "{:,.0f}")
row("MCU MACs / timestep", mcu[1], mcu[2], mcu[1] + mcu[2], "{:,.1f}")
row("Python MACs / timestep", py[1], py[2], py[1] + py[2], "{:,.1f}")
row("MCU / Python", mcu[1] / py[1], mcu[2] / py[2], (mcu[1] + mcu[2]) / (py[1] + py[2]), "{:.4f}")
print("-" * len(hdr))
row("Input sparsity (Python) %", 100 * (1 - act[1]), 100 * (1 - act[2]), float("nan"), "{:.2f}")
row("MAC sparsity MCU %", 100 * (1 - mcu[1] / DENSE[1]), 100 * (1 - mcu[2] / DENSE[2]),
    100 * (1 - (mcu[1] + mcu[2]) / dense_b), "{:.2f}")
row("MAC sparsity Python %", 100 * (1 - py[1] / DENSE[1]), 100 * (1 - py[2] / DENSE[2]),
    100 * (1 - (py[1] + py[2]) / dense_b), "{:.2f}")
row("MAC reduction MCU (x)", DENSE[1] / mcu[1], DENSE[2] / mcu[2], dense_b / (mcu[1] + mcu[2]), "{:.1f}")
row("MAC reduction Python (x)", DENSE[1] / py[1], DENSE[2] / py[2], dense_b / (py[1] + py[2]), "{:.1f}")

# per-sample agreement (MCU value is truncated to an integer, so compare with floor of the Python mean)
for l in (1, 2):
    py_avg = np.floor(mac_py[l][idx] / L[idx])
    mc_avg = np.array([mac_mcu[l][k] for k in ids])
    same = int((py_avg == mc_avg).sum())
    rel = np.abs(mc_avg - py_avg) / np.maximum(py_avg, 1)
    print(f"\nConv {l}: per-sample MACs/timestep equal to Python in {same}/{len(ids)} samples, "
          f"mean |diff| {100 * rel.mean():.2f}%, max {100 * rel.max():.2f}%")
print("\nConv 1 input is the same on both sides, so it should match exactly; "
      "Conv 2 input is the Conv 1 spikes, which differ slightly after quantization.")
