import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
import numpy as np

# 1. Setup - MUST match your training parameters
device = torch.device("cpu") # Use CPU for easier debugging/comparison
beta = 0.95
reset_mechanism = "subtract"
model_path = "best_scnn_net_20260426_104230_acc91.7.pth" # Update this with the best model file

# 2. Re-define the exact architecture
scnn_net = nn.Sequential(
    nn.Conv2d(2, 32, kernel_size=4, stride=1, bias=False), 
    snn.Leaky(beta=beta, init_hidden=True, reset_mechanism=reset_mechanism),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=False), 
    snn.Leaky(beta=beta, init_hidden=True, reset_mechanism=reset_mechanism),
    nn.Flatten(),
    nn.Linear(64 * 3 * 3, 10, bias=False), 
    snn.Leaky(beta=beta, init_hidden=True, output=True, reset_mechanism=reset_mechanism)
).to(device)

# Load weights
scnn_net.load_state_dict(torch.load(model_path, map_location=device))
scnn_net.eval()

# 3. Load one sample from your testset
# We use the dataset object directly to avoid batching shuffle
from torch.utils.data import DataLoader
import tonic
testset = tonic.DiskCachedDataset(dataset=None, cache_path='./cache/stmnist/test')
# Get the very first sample (index 0)
data, target = testset[0] 
# Tonic data is (Time, Channel, H, W). Add Batch dimension -> (T, 1, C, H, W)
data = torch.from_numpy(data).unsqueeze(1).to(device).float()

# 4. Step-by-Step Manual Inference
print(f"--- Starting Golden Reference Inference ---")
print(f"Target Label: {target}")

# Reset hidden states
utils.reset(scnn_net)
# We access layers individually to peek at intermediate "acc" values
conv1 = scnn_net[0]
lif1  = scnn_net[1]
conv2 = scnn_net[2]
lif2  = scnn_net[3]
fc    = scnn_net[5]
lif3  = scnn_net[6]

total_spikes = torch.zeros(10)

layer1_spike_count = None
layer2_spike_count = None

with torch.no_grad():
    for param in scnn_net.parameters():
        param.data = torch.round(param.data * 60) / 60

for t in range(data.size(0)):
    input_spikes = data[t] # Shape: (1, 2, 10, 10)
    
    # --- Layer 1 ---
    c1_out = conv1(input_spikes) 
    s1 = lif1(c1_out)          # Returns only spikes because init_hidden=True
    mem1 = lif1.mem            # Access the hidden membrane potential state


    # Initialize spike counter on first timestep
    if layer1_spike_count is None:
        layer1_spike_count = torch.zeros_like(s1[0], dtype=torch.float32)  # (32, H, W)
    
    # Accumulate spikes for each neuron in layer 1
    layer1_spike_count += s1[0].detach().cpu()

    
    # --- Layer 2 ---
    c2_out = conv2(s1)
    s2 = lif2(c2_out)
    mem2 = lif2.mem

    # For counting spikes in Layer 2
    if layer2_spike_count is None:
        layer2_spike_count = torch.zeros_like(s2[0], dtype=torch.float32)

    layer2_spike_count += s2[0].detach().cpu()


    # --- Layer 3 (Output) ---
    cur3 = fc(s2.flatten(1))
    s3, mem3 = lif3(cur3)      # Returns both because output=True in your training script
    
    total_spikes += s3.squeeze().cpu()


# Find the most spiking neuron in layer 1
max_flat_idx = torch.argmax(layer1_spike_count).item()
ch, y, x = torch.unravel_index(torch.tensor(max_flat_idx), layer1_spike_count.shape)
print(f"Most spiking Layer 1 neuron:")
print(f"  Channel = {ch}, Y = {y}, X = {x}")
print(f"  Total spikes = {layer1_spike_count[ch, y, x].item()}")

# Find the most spiking neuron in layer 2
max_flat_idx = torch.argmax(layer2_spike_count).item()
ch, y, x = torch.unravel_index(
    torch.tensor(max_flat_idx),
    layer2_spike_count.shape
)

print("Most spiking Layer 2 neuron:")
print(f"  Channel = {ch}, Y = {y}, X = {x}")
print(f"  Total spikes = {layer2_spike_count[ch, y, x].item()}")

# Find the most spiking neuron in the output layer
max_idx = torch.argmax(total_spikes).item()

print("Most spiking output neuron:")
print(f"Neuron (class index) = {max_idx}")
print(f"Total spikes = {total_spikes[max_idx].item()}")

utils.reset(scnn_net)

# SETUP

# Lists to collect data across timesteps for your plot
timesteps = []
neuron_mem_history = []
neuron_spike_history = []

# Which layer will be tracked
layer = 3
# More than one sample? 
moresamples = True
num_test_samples = 200  

# Choose the neuron you want to track
if(layer == 1):
    # For Layer 1 most spiking,
    track_ch = 10
    track_y = 0
    track_x = 0

if(layer == 2):
    # For Layer 2 most spiking,
    track_ch = 14
    track_y = 1
    track_x = 2

if(layer == 3):
    # For output most spiking,
    track = 8



for t in range(data.size(0)):
    # --- LAYER 1 ---
    # The spikes coming from the dataset
    l1_input = data[t] 
    
    c1_out = conv1(l1_input)
    s1 = lif1(c1_out)
    mem1 = lif1.mem
    
    # --- LAYER 2 ---
    # The "Input" to Layer 2 is the "Output" (s1) of Layer 1
    l2_input = s1 
    
    c2_out = conv2(l2_input)
    s2 = lif2(c2_out)
    mem2 = lif2.mem


    # --- OUTPUT LAYER ---
    cur3 = fc(s2.flatten(1))
    s3, mem3 = lif3(cur3)



    timesteps.append(t)
    #Store values for this timestep

    if(layer == 1):
        # Layer 1 Conv output for the first neuron before it hits the LIF
        current_input_to_lif1 = c1_out[0, track_ch, track_y, track_x].item()
        # State of the neuron AFTER the LIF update
        current_mem_lif1 = lif1.mem[0, track_ch, track_y, track_x].item()
        current_spike_lif1 = s1[0, track_ch, track_y, track_x].item()
        neuron_mem_history.append(current_mem_lif1)
        neuron_spike_history.append(current_spike_lif1)
        print(f"T={t} | Input: {current_input_to_lif1:.4f} | Mem: {current_mem_lif1:.4f} | Spike: {current_spike_lif1}")

    if(layer == 2):
        # Layer 2 neuron
        l2_input = c2_out[0, track_ch, track_y, track_x].item()
        l2_mem = lif2.mem[0, track_ch, track_y, track_x].item()
        l2_spk = s2[0, track_ch, track_y, track_x].item()
        neuron_mem_history.append(l2_mem)
        neuron_spike_history.append(l2_spk)
        print(f"T={t} | Input: {l2_input:.4f} | Mem: {l2_mem:.4f} | Spike: {l2_spk}")
    
    if layer == 3:
        # Output neuron 
        current_input = cur3[0, track].item()
        current_mem = mem3[0, track].item()
        current_spike = s3[0, track].item()

        neuron_mem_history.append(current_mem)
        neuron_spike_history.append(current_spike)

        print(
            f"T={t} | Input: {current_input:.4f} | "
            f"Mem: {current_mem:.4f} | Spike: {current_spike}"
        )



# Save collected history to a file
np.savez(
    './debug_and_plot/membrane_potentials_python.npz', 
    timesteps=np.array(timesteps), 
    membrane_potentials=np.array(neuron_mem_history), 
    spikes=np.array(neuron_spike_history)
)
print("\n[INFO] Saved membrane potentials and spikes to './debug_and_plot/membrane_potentials_python.npz'")

with torch.no_grad():
    print(f"--- Result ---")
    print(f"Accumulated Spikes: {total_spikes.detach().cpu().numpy()}")
    print(f"Prediction: {total_spikes.argmax()} | Actual: {target}")


#TESTING FOR MORE THAN ONE.
if(moresamples):
    correct_count = 0

    print(f"--- Starting Inference on {num_test_samples} samples ---")

    for i in range(num_test_samples):
        # 1. Get sample
        data, target = testset[i]
        data = torch.from_numpy(data).unsqueeze(1).to(device).float()
        
        # 2. RESET hidden states for the new sample
        utils.reset(scnn_net)
        total_spikes = torch.zeros(10)
        
        # 3. Temporal Loop (Inference)
        with torch.no_grad():
            for t in range(data.size(0)):
                # Forward pass through the whole sequence
                # We use the full net here for simplicity, but you can keep your manual steps
                spk_out, mem_out = scnn_net(data[t])
                total_spikes += spk_out.squeeze().cpu()
        
        # 4. Calculate Prediction
        pred = total_spikes.argmax().item()
        is_correct = (pred == target)
        if is_correct:
            correct_count += 1
        
        # Converting to int and then a list for a clean [0, 5, 22...] look
        spike_list = total_spikes.int().tolist()
        print(f"Sample {i} | Target: {target} | Pred: {pred} | {'✓' if is_correct else '✗'}" + f"   Spike Counts: {spike_list}")

    print(f"\nFinal Accuracy: {correct_count}/{num_test_samples}, {correct_count/num_test_samples * 100:.2f}%")