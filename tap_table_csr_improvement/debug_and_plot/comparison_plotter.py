import argparse
import numpy as np
import matplotlib.pyplot as plt

# Usage:
#   python comparison_plotter.py --python membrane_potentials_layer1.npz \
#       --mcu for_tap_membrane/membrane_potentials_MCU_L1.npz \
#       --out for_tap_membrane/comparison_L1.png --title "Layer 1 (neuron 364)"
#
# Thesis figures (vector PDF + PNG preview, all layers of one method at once):
#   python comparison_plotter.py --thesis --method tap
#   -> for_tap_membrane/thesis_membrane_L1_L2.pdf/.png and thesis_membrane_output.pdf/.png
cli = argparse.ArgumentParser(description="Compare PyTorch vs MCU membrane potentials")
cli.add_argument("--python", default="membrane_potentials_python.npz", help="golden reference .npz")
cli.add_argument("--mcu", default="membrane_potentials_MCU.npz", help="parsed MCU log .npz")
cli.add_argument("--out", default=None, help="save the figure to this path (png/pdf); "
                 "with --thesis: output prefix (default for_<method>_membrane/thesis_membrane)")
cli.add_argument("--title", default=None, help="extra text appended to the figure title")
cli.add_argument("--no-show", action="store_true", help="don't open a plot window")
cli.add_argument("--thesis", action="store_true",
                 help="produce the thesis figures (L1+L2 as (a)/(b), output neuron separately)")
cli.add_argument("--dir", default=".", help="with --thesis: folder holding the golden .npz files "
                 "and the for_<method>_membrane subfolders")
cli.add_argument("--method", default="tap", help="with --thesis: which for_<method>_membrane to plot")
args = cli.parse_args()


def thesis_figures(args):
    """Overlay of float reference and MCU, spike raster (shaded where the spike decisions
    differ) and the difference U_MCU - U_ref, for each layer. Vector PDF at text width."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    c_ref, c_mcu, c_diff, c_miss = "#1f4e9c", "#e07b00", "#444444", "#f2c9c9"
    threshold = 1.0
    layers = {  # key: (golden npz, MCU npz suffix, draw the threshold line)
        "L1":   ("membrane_potentials_layer1.npz", "L1",   False),
        "L2":   ("membrane_potentials_layer2.npz", "L2",   True),
        "OUT8": ("membrane_potentials_layer3.npz", "OUT8", True),
    }
    mdir = f"{args.dir}/for_{args.method}_membrane"
    prefix = args.out or f"{mdir}/thesis_membrane"

    def load(key):
        ref_f, mcu_k, thr = layers[key]
        ref = np.load(f"{args.dir}/{ref_f}")
        mcu = np.load(f"{mdir}/membrane_potentials_MCU_{mcu_k}.npz")
        n = min(len(ref["membrane_potentials"]), len(mcu["membrane_potentials"]))
        return (np.arange(n), ref["membrane_potentials"][:n], mcu["membrane_potentials"][:n],
                ref["spikes"][:n] > 0, mcu["spikes"][:n] > 0, thr)

    def draw(axes, key, label):
        ax_u, ax_s, ax_d = axes
        t, u_ref, u_mcu, s_ref, s_mcu, thr = load(key)
        ax_u.plot(t, u_ref, color=c_ref)
        ax_u.plot(t, u_mcu, color=c_mcu, ls="--")
        if thr:
            ax_u.axhline(threshold, color="0.35", ls=":", lw=0.8)
        ax_u.set_ylabel(r"$U$")
        ax_u.text(-0.105, 1.0, label, transform=ax_u.transAxes, fontweight="bold", va="top")
        for x in t[s_ref != s_mcu]:                       # shade differing spike decisions
            ax_s.axvspan(x - 0.5, x + 0.5, color=c_miss, lw=0)
        ax_s.vlines(t[s_ref], 1.6, 2.4, color=c_ref, lw=1.0)
        ax_s.vlines(t[s_mcu], 0.6, 1.4, color=c_mcu, lw=1.0)
        ax_s.set_ylim(0.3, 2.7)
        ax_s.set_yticks([2, 1], ["ref", "MCU"])
        ax_s.tick_params(axis="y", length=0)
        ax_d.axhline(0, color="0.6", lw=0.6)
        ax_d.plot(t, u_mcu - u_ref, color=c_diff, lw=0.9)
        ax_d.set_ylabel(r"$U_{\mathrm{MCU}}-U_{\mathrm{ref}}$")
        for ax in axes:
            ax.grid(True, ls=":", lw=0.4, color="0.8")
            ax.set_xlim(t[0] - 0.5, t[-1] + 0.5)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[:2]:
            ax.tick_params(labelbottom=False)
        ax_s.grid(False)

    def figure(keys, labels, path, height):
        ratios = []
        for _ in keys:
            ratios += [2.2, 0.55, 1.0]
        fig, axs = plt.subplots(len(ratios), 1, figsize=(6.3, height), sharex=True,
                                gridspec_kw={"height_ratios": ratios, "hspace": 0.12})
        for i, (k, lab) in enumerate(zip(keys, labels)):
            draw(axs[3 * i:3 * i + 3], k, lab)
        for j in range(3, len(axs)):          # gap between the layer groups
            p = axs[j].get_position()
            axs[j].set_position([p.x0, p.y0 - 0.035, p.width, p.height])
        for i in range(len(keys)):            # every group shows its own time axis
            axs[3 * i + 2].tick_params(labelbottom=True)
        axs[-1].set_xlabel("Timestep")
        handles = [Line2D([], [], color=c_ref, label="snnTorch (float)"),
                   Line2D([], [], color=c_mcu, ls="--", label="MCU (Q15)"),
                   Line2D([], [], color="0.35", ls=":", lw=0.8, label=r"threshold $U_{\mathrm{thr}}$"),
                   Patch(color=c_miss, label="spike decision differs")]
        axs[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                      ncol=4, frameon=False, handlelength=1.8, columnspacing=1.2)
        fig.savefig(path + ".pdf")
        fig.savefig(path + ".png", dpi=200)
        plt.close(fig)
        print(f"[SAVED] {path}.pdf (+ .png preview)")

    figure(["L1", "L2"], ["(a)", "(b)"], prefix + "_L1_L2", height=6.4)
    figure(["OUT8"], [""], prefix + "_output", height=3.3)


if args.thesis:
    thesis_figures(args)
    raise SystemExit(0)

# 1. Load both datasets
try:
    python_data = np.load(args.python)
    t_py = python_data['timesteps']
    mem_py = python_data['membrane_potentials']
    spikes_py = python_data['spikes']
except FileNotFoundError:
    print(f"Error: '{args.python}' not found.")
    raise

try:
    hw_data = np.load(args.mcu)
    t_hw = hw_data['timesteps']
    mem_hw = hw_data['membrane_potentials']
    spikes_hw = hw_data['spikes']
except FileNotFoundError:
    print(f"Error: '{args.mcu}' not found.")
    raise

# Ensure arrays are matching lengths before calculating differences
min_len = min(len(mem_py), len(mem_hw))
t_error = t_py[:min_len]
divergence_over_time = mem_hw[:min_len] - mem_py[:min_len]

# 2. Setup Dashboard Layout (2 small top, 1 middle wide, 1 bottom wide)
fig = plt.figure(figsize=(15, 12))
suptitle = 'SNN Neuron Dynamics & Hardware Divergence Analysis'
if args.title:
    suptitle += f' — {args.title}'
fig.suptitle(suptitle, fontsize=14, fontweight='bold')

# Define grid structure
ax1 = plt.subplot(3, 2, 1)  # Top Left
ax2 = plt.subplot(3, 2, 2, sharey=ax1)  # Top Right
ax3 = plt.subplot(3, 1, 2)  # Middle row, full width
ax4 = plt.subplot(3, 1, 3, sharex=ax3)  # Bottom row, full width (shares X axis)

# --- GRAPH 1: Python / snnTorch Simulation ---
ax1.plot(t_py, mem_py, label='PyTorch $V_m$', color='blue', linewidth=1.5)
py_spike_times = t_py[spikes_py > 0]
py_spike_heights = mem_py[spikes_py > 0]
ax1.scatter(py_spike_times, py_spike_heights, color='crimson', marker='o', s=40, label='Sim Spike', zorder=3)
ax1.set_title('PyTorch / snnTorch Simulation')
ax1.set_ylabel('Normalized Potential')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left')

# --- GRAPH 2: Hardware / Parsed Log Execution ---
ax2.plot(t_hw, mem_hw, label='Hardware $V_m$', color='darkorange', linewidth=1.5)
hw_spike_times = t_hw[spikes_hw > 0]
hw_spike_heights = mem_hw[spikes_hw > 0]
ax2.scatter(hw_spike_times, hw_spike_heights, color='black', marker='x', s=50, label='HW Spike', zorder=3)
ax2.set_title('Normalized Hardware Execution Log')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper left')

# --- GRAPH 3: DIRECT OVERLAP ALIGNMENT ---
ax3.plot(t_py, mem_py, label='PyTorch Simulation', color='blue', linewidth=2.0, alpha=0.7)
ax3.plot(t_hw, mem_hw, label='Hardware Execution', color='darkorange', linewidth=1.5, linestyle='--', alpha=0.8)
ax3.scatter(py_spike_times, py_spike_heights, color='crimson', marker='o', s=60, label='Sim Spike', zorder=4)
ax3.scatter(hw_spike_times, hw_spike_heights, color='black', marker='x', s=70, label='HW Spike', zorder=5)
ax3.set_title('Direct Overlap Alignment Verification')
ax3.set_ylabel('Normalized Potential')
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.legend(loc='upper right')

# --- GRAPH 4: DIVERGENCE OVER TIMESTEPS ---
# Shading the area to highlight structural accumulating error vs oscillations
ax4.plot(t_error, divergence_over_time, label=r'Error ($\Delta V = V_{hw} - V_{py}$)', color='purple', linewidth=1.75)
ax4.fill_between(t_error, divergence_over_time, 0, color='purple', alpha=0.15)
ax4.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)  # Perfect tracking baseline
ax4.set_title('Divergence Track Over Time')
ax4.set_xlabel('Timestep')
ax4.set_ylabel(r'Potential Delta ($\Delta V$)')
ax4.grid(True, linestyle=':', alpha=0.7)
ax4.legend(loc='upper left')

# 3. Clean up margins and render display
plt.tight_layout()

# --- PRINT STATISTICAL SUMMARY ---
final_idx = min_len - 1
final_v_py = mem_py[final_idx]
final_v_hw = mem_hw[final_idx]
final_divergence = divergence_over_time[final_idx]
percentage_error = (final_divergence / final_v_py) * 100 if final_v_py != 0 else 0.0

print(f"==========================================================")
print(f"--- DIVERGENCE METRICS AT LAST TIMESTEP (T = {final_idx}) ---")
print(f"==========================================================")
print(f"PyTorch Simulation V_m : {final_v_py:.6f}")
print(f"Hardware Execution V_m : {final_v_hw:.6f}")
print(f"Absolute Divergence    : {final_divergence:+.6f}")
print(f"Relative Deviation     : {percentage_error:+.2f}%")
print(f"----------------------------------------------------------")

# Identify max deviation over the run duration
max_error_idx = np.argmax(np.abs(divergence_over_time))
max_error_val = divergence_over_time[max_error_idx]
print(f"Peak Deviation across whole run:")
print(f"  Happened at T = {max_error_idx}")
print(f"  Value delta   = {max_error_val:+.6f}")
print(f"==========================================================")

if args.out:
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: '{args.out}'")

if not args.no_show:
    plt.show()