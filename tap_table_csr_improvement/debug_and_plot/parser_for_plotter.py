import re
import argparse
import numpy as np

# Configuration - defaults, can be overridden from the command line:
#   python parser_for_plotter.py <input_log.txt> <output.npz>
cli = argparse.ArgumentParser(description="Parse MCU UART neuron log into an .npz")
cli.add_argument("input_log_file", nargs="?", default="neuron_log_OUT8.txt")
cli.add_argument("output_npz_file", nargs="?", default="membrane_potentials_MCU.npz")
args = cli.parse_args()
input_log_file = args.input_log_file
output_npz_file = args.output_npz_file

# Conversion factor: scales the fixed-point integer back to normalized float
SCALING_FACTOR = 300.0 / 32768.0   # scale s = 300 (Load_NIR_Weights), Q15 full scale = 2^15

# Storage lists
timesteps = []
membrane_potentials = []
spikes = []

# Regex patterns to capture the digits after 'V:' and 'S:'
# Using -?\d+ ensures it still captures properly if V ever drops below 0
v_pattern = re.compile(r"V:(-?\d+)")
s_pattern = re.compile(r"S:(\d+)")

print(f"--- Starting Parser ---")

skipped_lines = 0
t = 0  # timestep counter: only advances on V:/S: data lines, so header/footer lines don't shift timesteps

with open(input_log_file, "r") as f:
    for line_no, line in enumerate(f, start=1):
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
            t += 1
        else:
            # UART header/footer (memory map, build info, cycle counters, result line)
            skipped_lines += 1

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

print(f"\n[SUCCESS] Successfully parsed {len(timesteps)} steps from '{input_log_file}'.")
print(f"  - Skipped {skipped_lines} non-data lines (header/footer)")
print(f"  - Quantized V range: {int(mem_arr.min()/SCALING_FACTOR)} to {int(mem_arr.max()/SCALING_FACTOR)}")
print(f"  - Normalized V range: {mem_arr.min():.4f} to {mem_arr.max():.4f}")
print(f"  - Total Spikes detected: {spikes_arr.sum()}")
print(f"  - Saved file to: '{output_npz_file}'")