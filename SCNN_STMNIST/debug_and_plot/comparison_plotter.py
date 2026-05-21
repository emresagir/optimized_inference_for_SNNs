import numpy as np
import matplotlib.pyplot as plt

# 1. Load both datasets
try:
    python_data = np.load('membrane_potentials_python.npz')
    t_py = python_data['timesteps']
    mem_py = python_data['membrane_potentials']
    spikes_py = python_data['spikes']
except FileNotFoundError:
    print("Error: 'membrane_potentials_python.npz' not found.")
    raise

try:
    hw_data = np.load('membrane_potentials_MCU.npz')
    t_hw = hw_data['timesteps']
    mem_hw = hw_data['membrane_potentials']
    spikes_hw = hw_data['spikes']
except FileNotFoundError:
    print("Error: 'hardware_membrane_potentials.npz' not found.")
    raise

# Ensure arrays are matching lengths before calculating differences
min_len = min(len(mem_py), len(mem_hw))
t_error = t_py[:min_len]
divergence_over_time = mem_hw[:min_len] - mem_py[:min_len]

# 2. Setup Dashboard Layout (2 small top, 1 middle wide, 1 bottom wide)
fig = plt.figure(figsize=(15, 12))
fig.suptitle('SNN Neuron Dynamics & Hardware Divergence Analysis', fontsize=14, fontweight='bold')

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
ax4.plot(t_error, divergence_over_time, label='Error ($\Delta V = V_{hw} - V_{py}$)', color='purple', linewidth=1.75)
ax4.fill_between(t_error, divergence_over_time, 0, color='purple', alpha=0.15)
ax4.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)  # Perfect tracking baseline
ax4.set_title('Divergence Track Over Time')
ax4.set_xlabel('Timestep')
ax4.set_ylabel('Potential Delta ($\Delta V$)')
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

plt.show()