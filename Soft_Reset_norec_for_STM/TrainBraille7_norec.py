import snntorch as snn
from snntorch import spikeplot as splt  # Visualization tools for spikes
from snntorch import spikegen          # Spike generation utilities

import datetime

# PyTorch core libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # MNIST dataset and preprocessing

# Visualization and utility libraries
import matplotlib.pyplot as plt
import numpy as np
import itertools

import nir
from snntorch.export_nir import export_to_nir
from snntorch import functional as SF


training_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# =====================================================================
# DATA LOADER CONFIGURATION
# =====================================================================                        
# Load the actual TensorDataset objects
ds_train = torch.load("./ds_train.pt", weights_only=False)
ds_test  = torch.load("./ds_test.pt",  weights_only=False)
ds_val   = torch.load("./ds_val.pt",   weights_only=False)

batch_size = 64

dtype = torch.float                     # Data type for tensors
# Determine device: use GPU (CUDA) if available, else Mac GPU (MPS), else CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Create DataLoaders for batching and shuffling data
train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

num_inputs = 12
num_hidden = 42
num_outputs = 7

num_steps = 256

beta_r   = 0.90   # hidden layer — moderate memory
beta_out = 0.55   # output layer — fast decay, sharp classification

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta_r,   threshold=1.0, reset_mechanism="subtract", reset_delay=False)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta_out, threshold=1.0, reset_mechanism="subtract", reset_delay=False)

    def forward(self, x):
        # x shape expected: (batch_size, 256, 12)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        # We iterate through the TIME dimension (256 steps)
        for step in range(num_steps):
            # x[:, step, :] gets the 12 features for THIS specific time step
            x_step = x[:, step, :] 
            
            cur1 = self.fc1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


net = Net().to(device)


# class NetForNIR(nn.Module):
#     """Single-timestep wrapper for NIR export — shares weights with Net."""
#     def __init__(self, trained_net):
#         super().__init__()
#         self.fc1  = trained_net.fc1
#         self.lif1 = trained_net.lif1
#         self.fc2  = trained_net.fc2
#         self.lif2 = trained_net.lif2

#     def forward(self, x):
#         # x: (batch, features) — single timestep, no loop needed
#         mem1 = self.lif1.init_leaky()
#         mem2 = self.lif2.init_leaky()
#         cur1       = self.fc1(x)
#         spk1, mem1 = self.lif1(cur1, mem1)
#         cur2       = self.fc2(spk1)
#         spk2, mem2 = self.lif2(cur2, mem2)
#         return spk2


# =====================================================================
# UTILITY FUNCTIONS FOR EVALUATION
# =====================================================================
def print_batch_accuracy(data, targets, train=False):
    """Calculate and print accuracy for a single batch.
    
    Method: Sum spikes over all time steps and select the output neuron
    with the highest spike count as the predicted class.
    
    Args:
        data: Input batch of images
        targets: Ground truth labels
        train: If True, print as train accuracy; else as test accuracy
    """
    # Forward pass: get spike recordings
    output, _ = net(data)
    
    # Sum spikes across time steps and find neuron with max spikes
    _, idx = output.sum(dim=0).max(1)
    
    # Calculate accuracy by comparing predictions with targets
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(current_data, current_targets, current_test_data, current_test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[-1]:.2f}")
    print_batch_accuracy(current_data, current_targets, train=True)
    print_batch_accuracy(current_test_data, current_test_targets, train=False)
    print("\n")


def evaluate(loader, model):
    """Evaluate model accuracy over an entire DataLoader."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, targets in loader:
            data    = data.to(device)
            targets = targets.to(device)

            spk_rec, mem_rec = model(data)

            # Loss across time steps
            batch_loss = loss_fn(spk_rec, targets)

            total_loss += batch_loss.item()

            # Accuracy via spike count
            _, predicted = spk_rec.sum(dim=0).max(1)
            correct += (predicted == targets).sum().item()
            total   += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# =====================================================================
# LOSS FUNCTION AND OPTIMIZER
# =====================================================================
# count loss
loss_fn = SF.ce_count_loss()

# Adam optimizer: adaptive learning rate optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, betas=(0.9, 0.999))




num_epochs = 250                 # Number of epochs to train
loss_hist = []                  # Track training loss over iterations
test_loss_hist = []             # Track test loss over iterations
counter = 0                     # Counter for printing progress

test_batch_iter = iter(test_loader)

best_val_acc = 0.0  
# Outer epoch loop
for epoch in range(num_epochs):
    iter_counter = 0            # Reset iteration counter for this epoch
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        # Move data to device
        data = data.to(device)
        targets = targets.to(device)

        # Set network to training mode (enables dropout, etc.)
        net.train()
        
        # Forward pass: 
        spk_rec, mem_rec = net(data)

        loss_val = loss_fn(spk_rec, targets)

        # Backpropagation and optimization
        optimizer.zero_grad()       # Clear previous gradients
        loss_val.backward()          # Compute gradients
        optimizer.step()             # Update weights

        # Record training loss
        loss_hist.append(loss_val.item())

        # ===== Test Set Evaluation =====
        with torch.no_grad():  # Disable gradient computation for testing
            # Set network to evaluation mode (disables dropout, etc.)
            net.eval()
            
            # Get test batch
            try:
                batch_test_data, batch_test_targets = next(test_batch_iter)
            except StopIteration:
                test_batch_iter = iter(test_loader)
                batch_test_data, batch_test_targets = next(test_batch_iter)

            # Test forward pass
            test_spk, test_mem = net(batch_test_data)
            # Calculate test loss
            test_loss = loss_fn(test_spk, batch_test_targets)

            test_loss_hist.append(test_loss.item())

            # Print progress every 50 iterations
            if counter % 50 == 0:
                train_printer(data, targets, batch_test_data, batch_test_targets)            
            counter += 1
            iter_counter += 1
    
    val_loss, val_acc = evaluate(val_loader, net)
    print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), f"./retrained_snntorch_{training_datetime}.pt")
        print(f"  → New best model saved ({val_acc:.2f}%)")

# =====================================================================
# VISUALIZATION: LOSS CURVES
# =====================================================================
# Plot training and test loss over iterations
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# =====================================================================
# FINAL EVALUATION: FULL TEST SET ACCURACY
# =====================================================================
total = 0                       # Total number of test samples
correct = 0                     # Number of correctly classified samples

# Recreate test loader without dropping last batch (to evaluate all samples)
final_test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=True, drop_last=False)

# Evaluate on entire test set
with torch.no_grad():           # No gradient computation needed
    net.eval()                    # Set to evaluation mode
  
    for data, targets in final_test_loader:
    # Move data to device
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass: get spike recordings
        test_spk, _ = net(data)

        # Find predicted class: neuron with max spike count
        _, predicted = test_spk.sum(dim=0).max(1)
        
        # Update accuracy counts
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")



net.load_state_dict(torch.load(f"./retrained_snntorch_{training_datetime}.pt"))
final_loss, final_acc = evaluate(final_test_loader, net)
print(f"Final Test Accuracy (best model): {final_acc:.2f}%")


# net_for_nir = NetForNIR(net)                          # wraps same weights, no copy
# sample_data = torch.zeros(1, num_inputs).to(device)   # (1, 12) — single timestep
# sample_data = torch.zeros(num_inputs).to(device)
# nir_graph   = export_to_nir(net_for_nir, sample_data)

# # 2. Repair the metadata manually
# for name, node in nir_graph.nodes.items():
#     # Fix the Input node
#     if isinstance(node, nir.Input):
#         node.input_type = {'input': np.array([num_inputs])}
    
#     # Fix the Linear/Affine nodes
#     elif isinstance(node, (nir.Affine, nir.Linear)):
#         # Extract shapes from the weights already stored in the node
#         # weight shape is usually (out_features, in_features)
#         out_f, in_f = node.weight.shape
#         node.input_type = {'input': np.array([in_f])}
#         node.output_type = {'output': np.array([out_f])}
        
#     # Fix the Spiking/LIF nodes
#     elif isinstance(node, (nir.LIF)):
#         # Match the size to the hidden or output count
#         size = num_hidden if "1" in name else num_outputs
#         node.input_type = {'input': np.array([size])}
#         node.output_type = {'output': np.array([size])}
        
#         # # Requirement #3 for your C-generator: Ensure parameters are NumPy arrays
#         # node.beta = np.array([node.beta]) if isinstance(node.beta, (float, int)) else node.beta
#         # node.v_threshold = np.array([1.0])
#         # node.v_reset = np.array([0.0])

# # This forces NIR to rebuild the edges using the nodes we just repaired
# repaired_graph = nir.NIRGraph(nodes=nir_graph.nodes, edges=nir_graph.edges)

# nir_graph.infer_types()
# nir.write("braille_model.nir", nir_graph)
# print("*** NIR model saved ***")






# # Mapping the architecture: 12 (In) -> 42 (Hidden) -> 7 (Out)
# snntorch_network = nn.Sequential(
#     nn.Linear(12, 42, bias=False), 
#     snn.Leaky(beta=0.90, threshold=1.0, init_hidden=True),
#     nn.Linear(42, 7, bias=False),    
#     snn.Leaky(beta=0.55, threshold=1.0, init_hidden=True, output=True)
# )

# checkpoint = torch.load(f"./retrained_snntorch_{training_datetime}.pt")

# # Sequential[0] is your fc1 (12 -> 42)
# snntorch_network[0].weight.data = checkpoint['fc1.weight']

# # Sequential[2] is your fc2 (42 -> 7)
# snntorch_network[2].weight.data = checkpoint['fc2.weight']

# sample_data = torch.randn(12)

# # Now the exporter will actually 'see' the dimensions
# nir_graph = export_to_nir(snntorch_network, sample_data)

# nir.write("braille_model.nir", nir_graph)
# print("*** NIR model saved ***")


