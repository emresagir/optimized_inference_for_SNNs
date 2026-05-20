import torch
import torch.nn as nn
import snntorch as snn

beta = 0.8
threshold = 1.0

# 3 inputs -> 5 outputs
linear = nn.Linear(3, 5, bias=False)

leaky = snn.Leaky(
    beta=beta,
    threshold=threshold,
    reset_mechanism='subtract'
)

# Hard-coded weights: shape = (5, 3)
weights = torch.tensor([
    [0.5,  0.2, -0.1],   # neuron 0
    [0.3, -0.4,  0.8],   # neuron 1
    [-0.6, 0.1,  0.2],   # neuron 2
    [0.7,  0.5, -0.3],   # neuron 3
    [0.2, -0.2,  0.4],   # neuron 4
], dtype=torch.float32)

linear.weight.data = weights

# Input sequence: 5 test vectors
# Last two are zeros -> membrane only decays
#
# Shape: (Time, Input)
inputs = torch.tensor([
    [1.0, 0.0, 0.0],   # t=0
    [0.0, 1.0, 0.0],   # t=1
    [0.0, 0.0, 1.0],   # t=2
    [0.0, 0.0, 0.0],   # t=3 -> decay only
    [0.0, 0.0, 0.0],   # t=4 -> decay only
], dtype=torch.float32)

# Initial membrane state: batch=1, neurons=5
mem = torch.zeros(1, 5)

print("--- Python Gold Reference (Linear) ---")

for t in range(5):

    # Add batch dimension
    cur = linear(inputs[t].unsqueeze(0))

    mem_before = mem.clone()

    spk, mem = leaky(cur, mem)

    print(
        f"T={t} | "
        f"input: {inputs[t][0]:.4f} | "
        f"cur: {cur[0,3]:.4f} | "
        f"mem_before: {mem_before[0,3]:.4f} | "
        f"mem_after: {mem[0,3]:.4f} | "
        f"spk: {spk[0,3]:.4f}"
    )