"""
Golden reference inference for a pruned ST-MNIST SCNN.

Same flow as ../opt_SCNN_MNSIT/golden_ref_for_opt.py, pointed at a checkpoint
produced by prune_SCNN_STMNIST.py (plain state_dict, pruned weights == 0).

Usage:
    python3 golden_ref_pruned.py                                  # newest .pth in ./pruned_models
    python3 golden_ref_pruned.py --model pruned_models/pruned_scnn_s70_....pth
"""

import argparse
import glob
import os

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
import numpy as np
import tonic


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Golden reference inference for a pruned SCNN.")
parser.add_argument("--model", default=None,
                    help="Pruned checkpoint (default: newest .pth in ./pruned_models)")
parser.add_argument("--num_test_samples", type=int, default=200)
parser.add_argument("--no_more_samples", action="store_true", help="Skip the multi-sample accuracy run")
args = parser.parse_args()


# ============================================================
# 1. Setup
# ============================================================

device = torch.device("cpu")
beta = 0.95
reset_mechanism = "subtract"

model_path = args.model
if model_path is None:
    candidates = glob.glob(os.path.join(SCRIPT_DIR, "pruned_models", "*.pth"))
    if not candidates:
        raise FileNotFoundError("No pruned models in ./pruned_models. Run prune_SCNN_STMNIST.py first, or pass --model.")
    model_path = max(candidates, key=os.path.getmtime)

print(f"Model: {model_path}")

testset = tonic.DiskCachedDataset(
    dataset=None,
    cache_path=os.path.join(SCRIPT_DIR, "..", "SCNN_STMNIST", "cache", "stmnist", "test")
)


# ============================================================
# 2. Exact architecture from training
# ============================================================

scnn_net = nn.Sequential(
    nn.Conv2d(2, 8, kernel_size=4, stride=1, bias=False),
    snn.Leaky(
        beta=beta,
        init_hidden=True,
        reset_mechanism=reset_mechanism
    ),
    nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False),
    snn.Leaky(
        beta=beta,
        init_hidden=True,
        reset_mechanism=reset_mechanism
    ),
    nn.Flatten(),
    nn.Linear(16 * 3 * 3, 10, bias=False),
    snn.Leaky(
        beta=beta,
        init_hidden=True,
        output=True,
        reset_mechanism=reset_mechanism
    )
).to(device)


# Load weights (plain state_dict; prune.remove already applied by the pruning script)
state_dict = torch.load(model_path, map_location=device)
if any(k.endswith("weight_orig") or k.endswith("weight_mask") for k in state_dict):
    raise ValueError("Checkpoint still holds pruning masks (weight_orig/weight_mask). "
                     "Use the files saved by prune_SCNN_STMNIST.py.")
scnn_net.load_state_dict(state_dict)
scnn_net.eval()


# Access layers
conv1 = scnn_net[0]
lif1  = scnn_net[1]
conv2 = scnn_net[2]
lif2  = scnn_net[3]
fc    = scnn_net[5]
lif3  = scnn_net[6]


def print_sparsity(title):
    print(title)
    zeros_total, numel_total = 0, 0
    for name, layer in (("conv1", conv1), ("conv2", conv2), ("fc", fc)):
        w = layer.weight
        zeros = int((w == 0).sum().item())
        zeros_total += zeros
        numel_total += w.numel()
        print(f"  {name:>5}: {zeros:5d}/{w.numel():5d} zero ({zeros / w.numel() * 100:5.1f}%)")
    print(f"  total: {zeros_total:5d}/{numel_total:5d} zero ({zeros_total / numel_total * 100:5.1f}%)")


print_sparsity("--- Weight sparsity (float, as pruned) ---")


# ------------------------------------------------------------
# Weight quantization, bit-exact with the generated C code
# ------------------------------------------------------------
# nir_to_c_generator_V2.py emits each weight as a "%.4e" float literal, then
# Load_NIR_Weights() does:
#     float scaled = w / scale;                      // scale = 60.0f
#     arm_float_to_q15(&scaled, &weights[j], 1);
# Without ARM_MATH_ROUNDING (not defined in these projects) arm_float_to_q15 is
#     q = __SSAT((q31_t)(scaled * 32768.0f), 16)    // C cast = truncate toward 0
# The C membrane potential lives in the same /60 Q15 domain, so the effective
# float weight the MCU uses is q * 60 / 32768.
Q15_SCALE = 60.0


def quantize_like_c(w: torch.Tensor):
    w_lit = np.array([float(f"{v:.4e}") for v in w.detach().cpu().numpy().ravel()],
                     dtype=np.float32).reshape(w.shape)          # "%.4e" literal
    scaled = w_lit / np.float32(Q15_SCALE)                       # float32 divide
    prod = scaled * np.float32(32768.0)                          # float32 multiply
    q = np.clip(np.trunc(prod), -32768, 32767).astype(np.int32)  # (q31_t) cast + __SSAT
    w_eff = q.astype(np.float64) * Q15_SCALE / 32768.0
    return torch.from_numpy(w_eff).to(w.dtype).to(w.device), q


with torch.no_grad():
    for name, layer in (("conv1", conv1), ("conv2", conv2), ("fc", fc)):
        w_eff, q = quantize_like_c(layer.weight)
        n_sat = int(((q == 32767) | (q == -32768)).sum())
        if n_sat:
            print(f"[WARN] {name}: {n_sat} weights saturated in Q15 (|w| >= {Q15_SCALE})")
        layer.weight.copy_(w_eff)

print_sparsity("--- Weight sparsity (after C-equivalent Q15 quantization) ---")
print("  (zeros here = taps the pruned CSR map can skip; should match the generator's connection count)")


# ============================================================
# 3. First test sample
# ============================================================

data, target = testset[0]

# T, C, H, W -> T, 1, C, H, W
data = torch.from_numpy(data).unsqueeze(1).to(device).float()

print("\n--- Starting Golden Reference Inference ---")
print(f"Target Label: {target}")

utils.reset(scnn_net)

total_spikes = torch.zeros(10)
layer1_spike_count = None
layer2_spike_count = None


# ============================================================
# 4. Manual inference
# ============================================================

with torch.no_grad():

    for t in range(data.size(0)):

        input_spikes = data[t]

        # Layer 1
        c1_out = conv1(input_spikes)
        s1 = lif1(c1_out)
        mem1 = lif1.mem

        if layer1_spike_count is None:
            layer1_spike_count = torch.zeros_like(
                s1[0], dtype=torch.float32
            )

        layer1_spike_count += s1[0].detach().cpu()

        # Layer 2
        c2_out = conv2(s1)
        s2 = lif2(c2_out)
        mem2 = lif2.mem

        if layer2_spike_count is None:
            layer2_spike_count = torch.zeros_like(
                s2[0], dtype=torch.float32
            )

        layer2_spike_count += s2[0].detach().cpu()

        # Output
        cur3 = fc(s2.flatten(1))
        s3, mem3 = lif3(cur3)

        total_spikes += s3.squeeze().cpu()


# ============================================================
# 5. Most spiking neurons
# ============================================================

idx = torch.argmax(layer1_spike_count).item()
ch, y, x = torch.unravel_index(
    torch.tensor(idx),
    layer1_spike_count.shape
)
most_l1 = (ch.item(), y.item(), x.item())

print("Most spiking Layer 1 neuron:")
print(f"  Channel = {ch.item()}, Y = {y.item()}, X = {x.item()}")
print(f"  Total spikes = {layer1_spike_count[ch, y, x].item()}")


idx = torch.argmax(layer2_spike_count).item()
ch, y, x = torch.unravel_index(
    torch.tensor(idx),
    layer2_spike_count.shape
)
most_l2 = (ch.item(), y.item(), x.item())

print("Most spiking Layer 2 neuron:")
print(f"  Channel = {ch.item()}, Y = {y.item()}, X = {x.item()}")
print(f"  Total spikes = {layer2_spike_count[ch, y, x].item()}")


max_idx = torch.argmax(total_spikes).item()

print("Most spiking output neuron:")
print(f"Neuron = {max_idx}")
print(f"Total spikes = {total_spikes[max_idx].item()}")


# ============================================================
# 6. Timestep debugging
# ============================================================

utils.reset(scnn_net)

# Set which layer(s) to debug. Use one value like [1], or all three like [1, 2, 3].
layer_debug_list = [1, 2, 3]
layer_debug_data = {
    layer: {
        "timesteps": [],
        "membrane_potentials": [],
        "spikes": []
    }
    for layer in layer_debug_list
}

# Tracked neurons. Pruning changes which neurons are active, so by default
# track the most spiking ones found above. Set to fixed tuples / ints to
# match a neuron you are tracing on the MCU instead.
TRACK_L1 = None   # e.g. (7, 3, 0)
TRACK_L2 = None   # e.g. (11, 0, 1)
TRACK_OUT = None  # e.g. 8

track_ch_1, track_y_1, track_x_1 = TRACK_L1 if TRACK_L1 is not None else most_l1
track_ch_2, track_y_2, track_x_2 = TRACK_L2 if TRACK_L2 is not None else most_l2
track = TRACK_OUT if TRACK_OUT is not None else max_idx

print(f"\nTracking L1 (ch={track_ch_1}, y={track_y_1}, x={track_x_1}), "
      f"L2 (ch={track_ch_2}, y={track_y_2}, x={track_x_2}), output neuron {track}")


with torch.no_grad():

    for t in range(data.size(0)):

        # Layer 1
        c1_out = conv1(data[t])
        s1 = lif1(c1_out)
        mem1 = lif1.mem

        # Layer 2
        c2_out = conv2(s1)
        s2 = lif2(c2_out)
        mem2 = lif2.mem

        # Output
        cur3 = fc(s2.flatten(1))
        s3, mem3 = lif3(cur3)

        for layer in layer_debug_list:

            if layer == 1:

                current_input = c1_out[
                    0, track_ch_1, track_y_1, track_x_1
                ].item()

                current_mem = lif1.mem[
                    0, track_ch_1, track_y_1, track_x_1
                ].item()

                current_spike = s1[
                    0, track_ch_1, track_y_1, track_x_1
                ].item()

                layer_debug_data[1]["timesteps"].append(t)
                layer_debug_data[1]["membrane_potentials"].append(current_mem)
                layer_debug_data[1]["spikes"].append(current_spike)

                print(
                    f"L1 | T={t} | Input: {current_input:.4f} | "
                    f"Mem: {current_mem:.4f} | "
                    f"Spike: {current_spike}"
                )

            if layer == 2:

                current_input = c2_out[
                    0, track_ch_2, track_y_2, track_x_2
                ].item()

                current_mem = lif2.mem[
                    0, track_ch_2, track_y_2, track_x_2
                ].item()

                current_spike = s2[
                    0, track_ch_2, track_y_2, track_x_2
                ].item()

                layer_debug_data[2]["timesteps"].append(t)
                layer_debug_data[2]["membrane_potentials"].append(current_mem)
                layer_debug_data[2]["spikes"].append(current_spike)

                print(
                    f"L2 | T={t} | Input: {current_input:.4f} | "
                    f"Mem: {current_mem:.4f} | "
                    f"Spike: {current_spike}"
                )

            if layer == 3:

                current_input = cur3[0, track].item()
                current_mem = mem3[0, track].item()
                current_spike = s3[0, track].item()

                layer_debug_data[3]["timesteps"].append(t)
                layer_debug_data[3]["membrane_potentials"].append(current_mem)
                layer_debug_data[3]["spikes"].append(current_spike)

                print(
                    f"L3 | T={t} | Input: {current_input:.4f} | "
                    f"Mem: {current_mem:.4f} | "
                    f"Spike: {current_spike}"
                )


debug_dir = os.path.join(SCRIPT_DIR, "debug_and_plot")
os.makedirs(debug_dir, exist_ok=True)

for layer, history in layer_debug_data.items():
    out_file = os.path.join(debug_dir, f"membrane_potentials_layer{layer}.npz")
    np.savez(
        out_file,
        timesteps=np.array(history["timesteps"]),
        membrane_potentials=np.array(history["membrane_potentials"]),
        spikes=np.array(history["spikes"])
    )
    print(f"\n[INFO] Saved to '{out_file}'")

print("\n--- Result ---")
print(f"Accumulated Spikes: {total_spikes.numpy()}")
print(f"Prediction: {total_spikes.argmax().item()}")
print(f"Actual: {target}")


# ============================================================
# 7. Test multiple samples
# ============================================================

moresamples = not args.no_more_samples
num_test_samples = min(args.num_test_samples, len(testset))

if moresamples:

    correct_count = 0

    print(
        f"\n--- Starting Inference on "
        f"{num_test_samples} samples ---"
    )

    for i in range(num_test_samples):

        data, target = testset[i]
        data = torch.from_numpy(data).unsqueeze(1).to(device).float()

        utils.reset(scnn_net)
        total_spikes = torch.zeros(10)

        with torch.no_grad():

            for t in range(data.size(0)):

                spk_out, mem_out = scnn_net(data[t])
                total_spikes += spk_out.squeeze().cpu()

        pred = total_spikes.argmax().item()

        if pred == target:
            correct_count += 1

        spike_list = total_spikes.int().tolist()

        print(
            f"Sample {i} | Target: {target} | "
            f"Pred: {pred} | "
            f"{'✓' if pred == target else '✗'}"
            f"   Spike Counts: {spike_list}"
        )

    print(
        f"\nFinal Accuracy: "
        f"{correct_count}/{num_test_samples}, "
        f"{correct_count / num_test_samples * 100:.2f}%"
    )
