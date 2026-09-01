import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
import numpy as np
import tonic


# ============================================================
# 1. Setup
# ============================================================

device = torch.device("cpu")
beta = 0.95
reset_mechanism = "subtract"

model_path = "best_scnn_net_20260817_171949_acc83.4.pth"

testset = tonic.DiskCachedDataset(
    dataset=None,
    cache_path="../SCNN_STMNIST/cache/stmnist/test"
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


# Load weights
scnn_net.load_state_dict(
    torch.load(model_path, map_location=device)
)
scnn_net.eval()


# Same quantization as your original golden reference
with torch.no_grad():
    for param in scnn_net.parameters():
        param.data = torch.round(param.data * 60) / 60


# Access layers
conv1 = scnn_net[0]
lif1  = scnn_net[1]
conv2 = scnn_net[2]
lif2  = scnn_net[3]
fc    = scnn_net[5]
lif3  = scnn_net[6]


# ============================================================
# 3. First test sample
# ============================================================

data, target = testset[0]

# T, C, H, W -> T, 1, C, H, W
data = torch.from_numpy(data).unsqueeze(1).to(device).float()

print("--- Starting Golden Reference Inference ---")
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

print("Most spiking Layer 1 neuron:")
print(f"  Channel = {ch.item()}, Y = {y.item()}, X = {x.item()}")
print(f"  Total spikes = {layer1_spike_count[ch, y, x].item()}")


idx = torch.argmax(layer2_spike_count).item()
ch, y, x = torch.unravel_index(
    torch.tensor(idx),
    layer2_spike_count.shape
)

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

timesteps = []
neuron_mem_history = []
neuron_spike_history = []

layer = 2

# Layer 1
# track_ch = 7
# track_y = 3
# track_x = 0

# Layer 2
track_ch = 11
track_y = 0
track_x = 1

# Output
# track = 8


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

    timesteps.append(t)

    if layer == 1:

        current_input = c1_out[
            0, track_ch, track_y, track_x
        ].item()

        current_mem = lif1.mem[
            0, track_ch, track_y, track_x
        ].item()

        current_spike = s1[
            0, track_ch, track_y, track_x
        ].item()

        neuron_mem_history.append(current_mem)
        neuron_spike_history.append(current_spike)

        print(
            f"T={t} | Input: {current_input:.4f} | "
            f"Mem: {current_mem:.4f} | "
            f"Spike: {current_spike}"
        )

    elif layer == 2:

        current_input = c2_out[
            0, track_ch, track_y, track_x
        ].item()

        current_mem = lif2.mem[
            0, track_ch, track_y, track_x
        ].item()

        current_spike = s2[
            0, track_ch, track_y, track_x
        ].item()

        neuron_mem_history.append(current_mem)
        neuron_spike_history.append(current_spike)

        print(
            f"T={t} | Input: {current_input:.4f} | "
            f"Mem: {current_mem:.4f} | "
            f"Spike: {current_spike}"
        )

    elif layer == 3:

        current_input = cur3[0, track].item()
        current_mem = mem3[0, track].item()
        current_spike = s3[0, track].item()

        neuron_mem_history.append(current_mem)
        neuron_spike_history.append(current_spike)

        print(
            f"T={t} | Input: {current_input:.4f} | "
            f"Mem: {current_mem:.4f} | "
            f"Spike: {current_spike}"
        )


np.savez(
    "./debug_and_plot/membrane_potentials_python.npz",
    timesteps=np.array(timesteps),
    membrane_potentials=np.array(neuron_mem_history),
    spikes=np.array(neuron_spike_history)
)

print(
    "\n[INFO] Saved to "
    "'./debug_and_plot/membrane_potentials_python.npz'"
)

print("\n--- Result ---")
print(f"Accumulated Spikes: {total_spikes.numpy()}")
print(f"Prediction: {total_spikes.argmax().item()}")
print(f"Actual: {target}")


# ============================================================
# 7. Test multiple samples
# ============================================================

moresamples = True
num_test_samples = 50

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