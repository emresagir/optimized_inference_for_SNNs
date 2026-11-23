#!/usr/bin/env python3
"""Thesis figure: accuracy of the Python (float) model and the MCU, at real length and with a 100-step window.

Real length    : each sample runs only its own timesteps (all speed tests).    logs: <method>_694.txt
100-step window: each sample runs 100 timesteps (2 s); input is zero after the end.
                 `#define SNN_WINDOW 100` in main.c.                             logs: window100_<method>.txt
Methods: base/basic, cmsis, csr, tap. The figure uses CSR; the table uses every log that exists.

The Python accuracy is computed here with the float model on the same 694 test samples.
Missing or incomplete logs are skipped (shown as "--").

Usage:  python3 plot_accuracy.py [folder]        (default folder: test_14_11_for_tap_CSR_base)
Output: <folder>/thesis_accuracy.pdf/.png and <folder>/thesis_accuracy_table.tex
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tonic
import snntorch as snn
from snntorch import utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# ---- Main network (2->8, 8->16 channels) ----
# FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "test_14_11_for_tap_CSR_base"
# MODEL = ROOT / "opt_SCNN_MNSIT" / "best_scnn_net_20260817_171949_acc83.4.pth"
# CACHE = ROOT / "SCNN_STMNIST" / "cache" / "stmnist" / "test"
# LOGS = {"real": FOLDER / "csr_694.txt", "window": FOLDER / "window100_csr.txt"}
# WINDOW, N = 100, 694
# C1, C2 = 8, 16   # output channels of Conv 1 and Conv 2
# # (row name, real-length log, window log)
# METHOD_LOGS = [("Basic", "base_694.txt", "window100_basic.txt"), ("CMSIS-NN", "cmsis_694.txt", "window100_cmsis.txt"),
#                ("CSR table", "csr_694.txt", "window100_csr.txt"), ("Tap table", "tap_694.txt", "window100_tap.txt")]

# ---- Larger network (2->32, 32->64 channels); no CSR, its tables do not fit ----
FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "big_network_logs"
MODEL = ROOT / "tap_table_csr_improvement" / "bigger_network" / "best_scnn_net_20260426_104230_acc91.7.pth"
CACHE = ROOT / "SCNN_STMNIST" / "cache" / "stmnist" / "test"
LOGS = {"real": FOLDER / "tap_694_big.txt", "window": FOLDER / "window100_tap_big.txt"}
WINDOW, N = 100, 694
C1, C2 = 32, 64   # output channels of Conv 1 and Conv 2
# (row name, real-length log, window log)
METHOD_LOGS = [("Basic", "base_694_big.txt", "window100_basic_big.txt"),
               ("CMSIS-NN", "cmsisnn_694_big.txt", "window100_cmsisnn_big.txt"),
               ("Tap table", "tap_694_big.txt", "window100_tap_big.txt")]

SAMPLE = re.compile(r"^\[(\d+)\]\s+T=\s*\d+\s+GT=(\d+)\s+Pred=(\d+)")


def mcu_preds(path):
    """Predictions of a complete MCU log, or None if the log is missing or incomplete."""
    if not path.exists():
        return None
    p = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = SAMPLE.match(line.strip())
        if m:
            p[int(m.group(1))] = int(m.group(3))
    if len(p) != N:
        print(f"[skip] {path.name}: {len(p)} samples, expected {N}")
        return None
    return np.array([p[i] for i in range(N)])


def mcu_correct(path):
    p = mcu_preds(path)
    return None if p is None else int((p == Y).sum())


# ---- Python (float) accuracy, both protocols, from one run of WINDOW steps ----
beta = 0.95
net = nn.Sequential(
    nn.Conv2d(2, C1, 4, 1, bias=False), snn.Leaky(beta=beta, init_hidden=True, reset_mechanism="subtract"),
    nn.Conv2d(C1, C2, 3, 2, bias=False), snn.Leaky(beta=beta, init_hidden=True, reset_mechanism="subtract"),
    nn.Flatten(), nn.Linear(C2 * 3 * 3, 10, bias=False),
    snn.Leaky(beta=beta, init_hidden=True, output=True, reset_mechanism="subtract"))
net.load_state_dict(torch.load(MODEL, map_location="cpu")); net.eval()
ds = tonic.DiskCachedDataset(dataset=None, cache_path=str(CACHE))
X = torch.zeros(WINDOW, N, 2, 10, 10); Y = np.zeros(N, int); L = np.zeros(N, int)
for i in range(N):
    x, y = ds[i]
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    X[:x.shape[0], i] = x; Y[i] = int(y); L[i] = x.shape[0]
with torch.no_grad():
    utils.reset(net)
    cum = torch.stack([net(X[t])[0] for t in range(WINDOW)]).cumsum(0).numpy()   # (WINDOW, N, 10)
py_pred = {"real": cum[L - 1, np.arange(N)].argmax(1), "window": cum[WINDOW - 1].argmax(1)}
py = {k: int((p == Y).sum()) for k, p in py_pred.items()}
mcu = {k: mcu_correct(p) for k, p in LOGS.items()}
for k in py:
    print(f"{k:7s} Python {py[k]}/{N} ({100 * py[k] / N:.2f}%)   MCU " +
          (f"{mcu[k]}/{N} ({100 * mcu[k] / N:.2f}%)" if mcu[k] is not None else "no log yet"))

# ---- figure ----
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
groups = [("real", "Real sample length"), ("window", f"{WINDOW}-step window (2 s)")]
series = [("Python (float)", py, "#bdbdbd"), ("MCU", mcu, "#1f4e9c")]
fig, ax = plt.subplots(figsize=(3.6, 2.6))
x = np.arange(len(groups)); w = 0.34
for j, (label, vals, color) in enumerate(series):
    for xi, (k, _) in zip(x, groups):
        if vals[k] is None:
            continue
        v = 100 * vals[k] / N
        ax.bar(xi + (j - 0.5) * w, v, w, color=color, lw=0, label=label if xi == 0 else None)
        ax.text(xi + (j - 0.5) * w, v + 1.2, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x, [g for _, g in groups]); ax.set_ylabel("Accuracy [%]")
ax.set_ylim(0, 100); ax.grid(True, axis="y", ls=":", lw=0.4, color="0.8"); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
for ext in ("pdf", "png"):
    fig.savefig(FOLDER / f"thesis_accuracy.{ext}", dpi=200)
print("[SAVED]", FOLDER / "thesis_accuracy.pdf", "(+ .png preview)")


# ---- LaTeX table: accuracy of every method, agreement with Python ----
# METHOD_LOGS is set at the top, together with the network.


def cells(pred, ref):
    if pred is None:
        return ["--", "--"]
    return [f"{100 * (pred == Y).mean():.2f}", f"{100 * (pred == ref).mean():.1f}"]


rows = [("Python (float)", ["{:.2f}".format(100 * py["real"] / N), "--",
                            "{:.2f}".format(100 * py["window"] / N), "--"])]
ACC_LINE = re.compile(r"Accuracy:\s*(\d+)/(\d+)")


def fw_accuracy(path):
    """The firmware's own 'Accuracy: k/N' line, as a cross-check of the parsed predictions."""
    if not path.exists():
        return None
    m = ACC_LINE.search(path.read_text(errors="ignore"))
    return int(m.group(1)) if m else None


def counts(pred, ref, path):
    """'correct/N  same/N  fw=k' for the terminal (raw numbers, so the percentages can be checked)."""
    if pred is None:
        return f"{'--':>9} {'--':>9} {'':>7}"
    fw = fw_accuracy(path)
    flag = "" if fw is None or fw == int((pred == Y).sum()) else " MISMATCH"
    return f"{int((pred == Y).sum()):>4}/{N} {int((pred == ref).sum()):>4}/{N} fw={fw}{flag}"


term = [("Python (float)", f"{py['real']:>4}/{N} {'--':>9} {'':>7}", f"{py['window']:>4}/{N} {'--':>9} {'':>7}")]
missing_window = []
for name, real_log, win_log in METHOD_LOGS:
    pr = mcu_preds(FOLDER / real_log)
    pw = mcu_preds(FOLDER / win_log)
    if pw is None:
        missing_window.append(name)
    rows.append((f"MCU, {name}", cells(pr, py_pred["real"]) + cells(pw, py_pred["window"])))
    term.append((f"MCU, {name}", counts(pr, py_pred["real"], FOLDER / real_log),
                 counts(pw, py_pred["window"], FOLDER / win_log)))
note = (f" No window run for: {', '.join(missing_window)}." if missing_window else "")

# ---- terminal: the table as it goes into LaTeX, then the raw counts behind it ----
print(f"\n{'':16s} | {'Real length':^21s} | {f'{WINDOW}-step window':^21s}")
print(f"{'':16s} | {'Acc [%]':>10s} {'Agree [%]':>10s} | {'Acc [%]':>10s} {'Agree [%]':>10s}")
print("-" * 64)
for name, c in rows:
    print(f"{name:16s} | {c[0]:>10s} {c[1]:>10s} | {c[2]:>10s} {c[3]:>10s}")
print(f"\nRaw counts: correct/N, same prediction as Python/N, fw = 'Accuracy:' line of the MCU log")
print(f"{'':16s} | {'Real length':^28s} | {f'{WINDOW}-step window':^28s}")
print("-" * 80)
for name, a, b in term:
    print(f"{name:16s} | {a:28s} | {b:28s}")
print()

tex = [
    r"\begin{table}[htbp]",
    r"\centering",
    rf"\caption{{Accuracy on the {N} test samples, with each sample run for its real length and with a fixed",
    rf"{WINDOW}-step (2\,s) window. Agreement is the share of samples with the same prediction as the Python model."
    + note + "}",
    r"\label{tab:accuracy}",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    rf" & \multicolumn{{2}}{{c}}{{Real length}} & \multicolumn{{2}}{{c}}{{{WINDOW}-step window}} \\",
    r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
    r" & Accuracy [\%] & Agreement [\%] & Accuracy [\%] & Agreement [\%] \\",
    r"\midrule",
]
tex += [f"{name} & " + " & ".join(c) + r" \\" for name, c in rows]
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(FOLDER / "thesis_accuracy_table.tex").write_text("\n".join(tex) + "\n")
print("[SAVED]", FOLDER / "thesis_accuracy_table.tex", "(needs \\usepackage{booktabs})")
