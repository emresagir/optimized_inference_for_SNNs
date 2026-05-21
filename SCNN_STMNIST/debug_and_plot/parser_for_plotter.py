import re
import numpy as np

# Configuration - Update these paths as needed
input_log_file = "neuron_log_OUT8.txt"
output_npz_file = "membrane_potentials_MCU.npz"

# Conversion factor: scales the fixed-point integer back to normalized float
SCALING_FACTOR = 360.0 / 32767.0

# Storage lists
timesteps = []
membrane_potentials = []
spikes = []

# Regex patterns to capture the digits after 'V:' and 'S:'
# Using -?\d+ ensures it still captures properly if V ever drops below 0
v_pattern = re.compile(r"V:(-?\d+)")
s_pattern = re.compile(r"S:(\d+)")

print(f"--- Starting Parser ---")

with open(input_log_file, "r") as f:
    for t, line in enumerate(f):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
            
        v_match = v_pattern.search(line)
        s_match = s_pattern.search(line)
        
        if v_match and s_match:
            # Extract integers from text
            v_quantized = int(v_match.group(1))
            #debug
            #print(v_quantized)
            spike_val = int(s_match.group(1))
            #debug
            #print(spike_val)
            # Apply your normalization math
            v_normalized = v_quantized * SCALING_FACTOR
            
            # Append data to lists
            timesteps.append(t)
            membrane_potentials.append(v_normalized)
            #debug
            #print(v_normalized)
            spikes.append(spike_val)
        else:
            print(f"[WARNING] Line {t} did not match expected pattern: {line}")

# Convert lists to NumPy arrays
timesteps_arr = np.array(timesteps)
mem_arr = np.array(membrane_potentials)
spikes_arr = np.array(spikes)

# Save using the exact key format as your PyTorch script tracking
np.savez(
    output_npz_file, 
    timesteps=timesteps_arr, 
    membrane_potentials=mem_arr, 
    spikes=spikes_arr
)

print(f"\n[SUCCESS] Successfully parsed {len(timesteps)} steps.")
print(f"  - Quantized V range: {int(mem_arr.min()/SCALING_FACTOR)} to {int(mem_arr.max()/SCALING_FACTOR)}")
print(f"  - Normalized V range: {mem_arr.min():.4f} to {mem_arr.max():.4f}")
print(f"  - Total Spikes detected: {spikes_arr.sum()}")
print(f"  - Saved file to: '{output_npz_file}'")