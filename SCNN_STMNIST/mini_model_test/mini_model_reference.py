import torch
import torch.nn as nn
import snntorch as snn

beta = 0.8
threshold = 1.0

# 1 input channel, 2 output channels, square kernel (2,2)
# input (1, 3, 3) → output (2, 2, 2)
conv = nn.Conv2d(1, 2, kernel_size=(2, 2), bias=False)
leaky = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism='subtract')

# Hard-coded weights (2, 1, 2, 2)
weights = torch.tensor([
    [[[0.5,  0.2],
      [-0.1, 0.3]]],   # channel 0
    [[[0.1, -0.5],
      [0.8,  0.4]]]    # channel 1
])
conv.weight.data = weights

# Input: (Time, Batch=1, Channel=1, H=3, W=3)
inputs = torch.tensor([
    [[1, 0, 1],
     [0, 1, 0],
     [1, 0, 1]],  # T=0
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],  # T=1
    [[1, 1, 0],
     [0, 0, 1],
     [1, 0, 0]],  # T=2
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],  # T=3 decay
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],  # T=4 decay
]).float().unsqueeze(1).unsqueeze(1)  # (T, B=1, C=1, H=3, W=3)

mem = torch.zeros(1, 2, 2, 2)  # (B=1, C=2, H=2, W=2)

print("--- Python Gold Reference (Conv2d) ---")
for t in range(5):
    cur = conv(inputs[t])
    mem_before = mem.clone()
    spk, mem = leaky(cur, mem)
    print(f"T={t} | cur: {cur[0,0,0,0].item():.4f} | mem_before: {mem_before[0,0,0,0].item():.4f} | mem_after: {mem[0,0,0,0].item():.4f} | spk: {spk[0,0,0,0].item()}")