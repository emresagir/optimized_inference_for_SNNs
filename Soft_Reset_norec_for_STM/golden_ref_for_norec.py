import torch
import torch.nn as nn
import snntorch as snn
from torch.utils.data import DataLoader

# =====================================================================
# MODEL DEFINITION (must match training architecture)
# =====================================================================
num_inputs  = 12
num_hidden  = 42
num_outputs = 7
num_steps   = 256

beta_r   = 0.90
beta_out = 0.55

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta_r,   threshold=1.0, reset_mechanism="subtract", reset_delay=False)
        self.fc2  = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta_out, threshold=1.0, reset_mechanism="subtract", reset_delay=False)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []
        for step in range(num_steps):
            x_step     = x[:, step, :]
            cur1       = self.fc1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2       = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# =====================================================================
# LOAD MODEL & DATASET
# =====================================================================
MODEL_PATH = "./retrained_snntorch_20260411_175507.pt"   # <-- replace with your .pt filename

device = (torch.device("cuda")  if torch.cuda.is_available()  else
          torch.device("mps")   if torch.backends.mps.is_available() else
          torch.device("cpu"))

net = Net().to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Zero out biases to match the inference behavior
net.fc1.bias.data.zero_()
net.fc2.bias.data.zero_()
net.eval()

ds_test     = torch.load("./ds_test.pt", weights_only=False)
ds_test     = torch.utils.data.Subset(ds_test, range(20))
test_loader = DataLoader(ds_test, batch_size=20, shuffle=False, drop_last=False)

# =====================================================================
# INFERENCE
# =====================================================================
all_spike_counts = []
all_predicted    = []
all_expected     = []

with torch.no_grad():
    for data, targets in test_loader:
        data    = data.to(device)
        spk_rec, _ = net(data)              # (num_steps, batch, num_outputs)
        counts     = spk_rec.sum(dim=0)     # (batch, num_outputs)
        predicted  = counts.max(dim=1).indices

        all_spike_counts.append(counts.cpu())
        all_predicted.append(predicted.cpu())
        all_expected.append(targets)

all_spike_counts = torch.cat(all_spike_counts, dim=0)   # (N, num_outputs)
all_predicted    = torch.cat(all_predicted,    dim=0)   # (N,)
all_expected     = torch.cat(all_expected,     dim=0)   # (N,)

# =====================================================================
# PRINT RESULTS
# =====================================================================
for i in range(len(all_expected)):
    counts   = all_spike_counts[i].numpy().astype(int)
    pred_cls = all_predicted[i].item()
    exp_cls  = all_expected[i].item()
    match    = "✓" if pred_cls == exp_cls else "✗"

    print(f"[{i:3d}] Output spikes accumulation: {' '.join(str(c) for c in counts)}")
    print(f"       Predicted class: {pred_cls}  |  Expected class: {exp_cls}  {match}")
    print()

# =====================================================================
# OVERALL ACCURACY
# =====================================================================
correct = (all_predicted == all_expected).sum().item()
total   = len(all_expected)
print(f"Overall accuracy: {correct}/{total}  ({100*correct/total:.2f}%)")