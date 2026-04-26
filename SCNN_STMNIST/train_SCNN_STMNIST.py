import tonic
import tonic.transforms as transforms  # Not to be mistaken with torchdata.transfroms
from tonic import DiskCachedDataset

# torch imports
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn


# snntorch imports
import snntorch as snn
from snntorch import surrogate
import snntorch.spikeplot as splt
from snntorch import functional as SF
from snntorch import utils


# other imports
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.display import display
import numpy as np
import torchdata
import os

# For determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  

from ipywidgets import IntProgress
import time
import statistics

from datetime import datetime
import random

# For determinism
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

# For CUDA
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # for multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create a timestamp string (YearMonthDay_HourMinuteSecond)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the dynamic filename
model_name = f"scnn_net_{timestamp}.pth"

# Reset Mechanism
reset_mechanism = "subtract"

# Initialize tracking variables before the loop
best_val_acc = 0.0

# 1. Point to your existing cache folders
# We set dataset=None because the data is already transformed and saved on disk.
# Tonic will simply load the processed .pt or .npy files.
trainset = tonic.DiskCachedDataset(dataset=None, cache_path='./cache/stmnist/train')
valset   = tonic.DiskCachedDataset(dataset=None, cache_path='./cache/stmnist/val')
testset  = tonic.DiskCachedDataset(dataset=None, cache_path='./cache/stmnist/test')


batch_size = 32


# 3. Create DataLoaders
# CRITICAL: We use PadTensors because spike sequences have variable time lengths.
# batch_first=False is standard for snnTorch (Time, Batch, Channel, X, Y)
collate = tonic.collation.PadTensors(batch_first=False)

trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, shuffle=True, worker_init_fn=seed_worker, generator=g)
valloader  = DataLoader(valset,  batch_size=batch_size, collate_fn=collate, worker_init_fn=seed_worker, generator=g)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=collate, worker_init_fn=seed_worker, generator=g)



# Query the shape of a sample: time x batch x dimensions
data_tensor, targets = next(iter(trainloader))
print(data_tensor.shape)




device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# neuron and simulation parameters
beta = 0.95

# This is the same architecture that was used in the STMNIST Paper
# Modified architecture for easier C-generation, discarded maxpool2d and used strides. 
scnn_net = nn.Sequential(
    # Input: 10x10x2
    nn.Conv2d(2, 32, kernel_size=4, stride=1, bias=False), # Output: 7x7
    snn.Leaky(beta=beta, init_hidden=True, reset_mechanism=reset_mechanism),
    
    # We add stride=2 here to replace the MaxPool
    nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=False), # Output: 3x3
    snn.Leaky(beta=beta, init_hidden=True, reset_mechanism=reset_mechanism),
    
    nn.Flatten(),
    # Update the linear layer input: 64 channels * 3 * 3 = 576
    nn.Linear(64 * 3 * 3, 10, bias=False), 
    snn.Leaky(beta=beta, init_hidden=True, output=True, reset_mechanism=reset_mechanism)
).to(device)

optimizer = torch.optim.Adam(scnn_net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps

        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)

start_time = time.time()

num_epochs = 30

loss_hist = []
acc_hist = []

# training loop
for epoch in range(num_epochs):
    train_batch_acc = []

    for i, (data, targets) in enumerate(iter(trainloader)):
        data = data.to(device)
        targets = targets.to(device)

        scnn_net.train()
        spk_rec = forward_pass(scnn_net, data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Only calculate accuracy occasionally to save time
        if i % 10 == 0:
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Epoch {epoch} Iter {i} | Loss: {loss_val.item():.4f} | Acc: {acc*100:.2f}%")

        # Print accuracy every 4 iterations
        if i%10 == 0:
            print(f"Accuracy: {acc * 100:.2f}%\n")

    # VALIDATION PHASE: Run this after every epoch
    scnn_net.eval()
    val_acc_hist = []
    with torch.no_grad():
        for val_data, val_targets in valloader:
            val_data, val_targets = val_data.to(device), val_targets.to(device)
            val_spk = forward_pass(scnn_net, val_data)
            val_acc_hist.append(SF.accuracy_rate(val_spk, val_targets))

    current_val_acc = np.mean(val_acc_hist)
    print(f"Epoch {epoch} complete. Validation Accuracy: {current_val_acc * 100:.2f}%")
    
    # --- CHECKPOINT LOGIC ---
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        
        # Save the best model with a dynamic name
        model_path = f"best_scnn_net_{timestamp}_acc{best_val_acc*100:.1f}.pth"
        
        # Optional: Remove previous best files if you don't want a folder full of .pth files
        torch.save(scnn_net.state_dict(), model_path)
        print(f"New Best Model saved to {model_path}!")
    else:
        print(f"Validation accuracy did not improve (Best: {best_val_acc * 100:.2f}%)")
    

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Convert elapsed time to minutes, seconds, and milliseconds
minutes, seconds = divmod(elapsed_time, 60)
seconds, milliseconds = divmod(seconds, 1)
milliseconds = round(milliseconds * 1000)

# Print the elapsed time
print(f"Elapsed time: {int(minutes)} minutes, {int(seconds)} seconds, {milliseconds} milliseconds")
torch.save(scnn_net.state_dict(), model_name)
print(f"Model successfully saved as: {model_name}")

# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()


# Make sure your model is in evaluation mode
scnn_net.eval()

# Initialize variables to store predictions and ground truth labels
acc_hist = []

# Iterate over batches in the testloader
with torch.no_grad():
    for data, targets in testloader:
        # Move data and targets to the device (GPU or CPU)
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        spk_rec = forward_pass(scnn_net, data)

        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)

        # if i%10 == 0:
        # print(f"Accuracy: {acc * 100:.2f}%\n")

print("The average loss across the testloader is:", statistics.mean(acc_hist))
