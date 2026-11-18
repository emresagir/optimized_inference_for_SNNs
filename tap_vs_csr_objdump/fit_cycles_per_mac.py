#!/usr/bin/env python3
"""Measured cost per MAC: least-squares fit of the per-sample layer cycles of the 694-sample runs.

    total_cycles(sample) = a * T  +  b * sops

T = timesteps of the sample, sops = MACs (synaptic operations) of the sample, both printed by the
firmware ('snSP' line, only in the tap log). a = fixed cost per timestep (decay and reset loops over
all neurons, input scan), b = cycles per MAC. CSR and tap produce bit-identical spikes (0/694 spike
vectors differ), so the CSR samples have the same sops; they are taken from the tap log.
"""
import re
import sys
from pathlib import Path
import numpy as np

LOGS = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parents[1] / "tap_table_csr_improvement/tests_for_all_methods/test_14_11_for_tap_CSR_base"
SAMPLE = re.compile(r"^\[(\d+)\] T=\s*(\d+)")
SNSP = re.compile(r"snSP \[(\d+)\] T=(\d+) nnz1=\d+ sops1=(\d+) nnz2=\d+ sops2=(\d+)")
CYC = re.compile(r"(L[12]) layer update cycles: calls=(\d+) .* total=(\d+)")

def parse(path):
    """{sample: {"T", "L1", "L2"}} from the cycle lines after each '[k] T=' line, plus sops if present."""
    rows, k, sops = {}, None, {}
    for line in path.read_text(errors="ignore").splitlines():
        m = SNSP.search(line)
        if m:
            sops[int(m[1])] = {"L1": int(m[3]), "L2": int(m[4])}
            continue
        m = SAMPLE.match(line.strip())
        if m:
            k = int(m[1]); rows[k] = {"T": int(m[2])}
            continue
        m = CYC.search(line)
        if m and k is not None and int(m[2]) == rows[k]["T"]:
            rows[k][m[1]] = int(m[3])
    return {i: r for i, r in rows.items() if "L1" in r and "L2" in r}, sops


_, SOPS = parse(LOGS / "tap_694.txt")
print(f"{'log':6s} {'layer':5s} {'n':>4s} {'a [cyc/step]':>13s} {'b [cyc/MAC]':>12s} {'R^2':>7s}")
for name in ("csr_694.txt", "tap_694.txt"):
    rows, _ = parse(LOGS / name)
    for L in ("L1", "L2"):
        ks = sorted(set(rows) & set(SOPS))
        A = np.array([[rows[i]["T"], SOPS[i][L]] for i in ks], float)
        y = np.array([rows[i][L] for i in ks], float)
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ [a, b]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"{name.split('_')[0]:6s} {L:5s} {len(ks):>4d} {a:>13.0f} {b:>12.2f} {r2:>7.4f}")
