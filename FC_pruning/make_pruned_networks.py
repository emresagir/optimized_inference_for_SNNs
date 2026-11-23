#!/usr/bin/env python3
"""FC pruning test: a random one-layer FC network (144 -> 10, LIF), saved once per pruning ratio.

Weights : Gaussian, mean 0, std 0.5675 (= std of the trained FC of the small network), seed 0.
LIF     : like the trained network: beta 0.95, threshold 1, reset by subtraction.
          NIR stores beta as tau = dt / (1 - beta), dt = 1e-4 (like the snnTorch export).
Pruning : magnitude pruning, the smallest |w| are set to 0.

Output  : networks/random_fc_144x10_p<ratio>.nir   (ratio = 0, 20, 40, 60, 80, 90, 95, 100)
"""
from pathlib import Path

import numpy as np
import nir

N_IN = 144
N_OUT = 10
STD = 0.5675
SEED = 0
BETA = 0.95
THRESHOLD = 1.0
DT = 1e-4
RATIOS = [0, 20, 40, 60, 80, 90, 95, 100]

out_dir = Path(__file__).resolve().parent / "networks"
out_dir.mkdir(exist_ok=True)

# The random weights, the same for every ratio (NIR layout: weight[output][input])
rng = np.random.default_rng(SEED)
weight = rng.normal(0.0, STD, size=(N_OUT, N_IN)).astype(np.float32)


def prune(w, ratio):
    """Set the ratio % smallest |w| to 0."""
    w = w.copy().flatten()
    n_prune = round(w.size * ratio / 100)
    smallest_first = np.argsort(np.abs(w), kind="stable")
    w[smallest_first[:n_prune]] = 0.0
    return w.reshape(N_OUT, N_IN)


def save_network(w, path):
    tau = DT / (1.0 - BETA)
    nodes = {
        "input": nir.Input(input_type={"input": np.array([N_IN])}),
        "0": nir.Linear(weight=w),
        "1": nir.LIF(tau=np.full(N_OUT, tau, dtype=np.float32),
                     r=np.ones(N_OUT, dtype=np.float32),
                     v_leak=np.zeros(N_OUT, dtype=np.float32),
                     v_threshold=np.full(N_OUT, THRESHOLD, dtype=np.float32),
                     v_reset=np.zeros(N_OUT, dtype=np.float32)),
        "output": nir.Output(output_type={"output": np.array([N_OUT])}),
    }
    edges = [("input", "0"), ("0", "1"), ("1", "output")]
    nir.write(path, nir.NIRGraph(nodes=nodes, edges=edges, metadata={"reset_mechanism": "subtract"}))


for ratio in RATIOS:
    w = prune(weight, ratio)
    path = out_dir / f"random_fc_{N_IN}x{N_OUT}_p{ratio}.nir"
    save_network(w, path)
    print(f"[SAVED] {path.name}: {np.count_nonzero(w)}/{w.size} nonzero weights")
