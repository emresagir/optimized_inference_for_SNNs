import copy
import datetime
import json
import os
import numpy as np
import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

training_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

### TRAINED WEIGHTS STORING
store_weights = True

### TRAINING/VALIDATION HISTORIES STORING
store_training = True
store_validation = True

### DEVICE SETTINGS
use_gpu = False

if use_gpu:
    gpu_sel = 0
    device = torch.device("cuda:"+str(gpu_sel))
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
else:
    device = torch.device("cpu")
### SPECIFY THE RESET MECHANISM TO USE AND WHETHER TO DELAY IT OR NOT
reset_mechanism = "subtract" # "zero" or "subtract"
reset_delay = False # True or False

### SET BIAS USAGE DEPENDING ON THE TARGET PLATFORM
use_bias = False

### OPTIMAL HYPERPARAMETERS
if reset_mechanism == "subtract":
   #parameters_path = "./parameters_noDelay_noBias_ref_subtract.json"
   #Not planning to change any parameters for this test to be done, the same parameters file will be used.
   parameters_path = "./parameters_noDelay_bias_ref_zero.json"
elif reset_mechanism == "zero":
   parameters_path = "./parameters_noDelay_bias_ref_zero.json"

with open(parameters_path) as f:
   parameters = json.load(f)

parameters["reset"] = reset_mechanism
parameters["reset_delay"] = reset_delay

parameters["use_bias"] = use_bias

regularization = [parameters["reg_l1"], parameters["reg_l2"]]
### LOAD DATA
ds_train = torch.load("./ds_train.pt", weights_only=False)
ds_val = torch.load("./ds_val.pt", weights_only=False)
ds_test = torch.load("./ds_test.pt", weights_only=False)

letter_written = ['Space', 'A', 'E', 'I', 'O', 'U', 'Y']

def model_build(settings, input_size, num_steps, device):

    input_channels = int(input_size)
    num_hidden = int(settings["nb_hidden"])
    num_outputs = 7

    ### Surrogate gradient setting
    spike_grad = surrogate.fast_sigmoid(slope=int(settings["slope"]))

    ### Put things together
    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            ##### Initialize layers #####
            self.fc1 = nn.Linear(input_channels, num_hidden)
            if not settings["use_bias"]:
                self.fc1.__setattr__("bias",None)
            self.lif1 = snn.RLeaky(beta=settings["beta_r"], threshold= 1.0, spike_grad=spike_grad, reset_mechanism=settings["reset"], reset_delay=settings["reset_delay"], all_to_all=False)
            if not settings["use_bias"]:
                self.lif1.recurrent.__setattr__("bias",None)
            ### Output layer
            self.fc2 = nn.Linear(num_hidden, num_outputs)
            if not settings["use_bias"]:
                self.fc2.__setattr__("bias",None)
            self.lif2 = snn.Leaky(beta=settings["beta_out"], threshold= 1.0,spike_grad=spike_grad, reset_mechanism=settings["reset"], reset_delay=settings["reset_delay"])

        def forward(self, x):

            # Initialize hidden states at t=0
            spk1, mem1 = self.lif1.init_rleaky()
            mem2 = self.lif2.init_leaky()

            # Record the spikes from the hidden layer
            spk1_rec = [] 
            # Record the final layer
            spk2_rec = []

            for step in range(num_steps):
                ### Recurrent layer
                cur1 = self.fc1(x[step])
                spk1, mem1 = self.lif1(cur1, spk1, mem1)
                ### Output layer
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)

                spk1_rec.append(spk1)
                spk2_rec.append(spk2)

            return torch.stack(spk2_rec, dim=0), torch.stack(spk1_rec, dim=0)

    return Net().to(device)

def training_loop(dataset, batch_size, net, optimizer, loss_fn, device, regularization=None):
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    batch_loss = []
    batch_acc = []

    for data, labels in train_loader:
      
      data = data.to(device).swapaxes(1, 0)
      labels = labels.to(device)

      net.train()
      spk_rec, hid_rec = net(data)

      # Training loss
      if regularization != None:
        # L1 loss on spikes per neuron from the hidden layer
        reg_loss = regularization[0]*torch.mean(torch.sum(hid_rec, 0))
        # L2 loss on total number of spikes from the hidden layer
        reg_loss = reg_loss + regularization[1]*torch.mean(torch.sum(torch.sum(hid_rec, dim=0), dim=1)**2)
        loss_val = loss_fn(spk_rec, labels) + reg_loss
      else:
        loss_val = loss_fn(spk_rec, labels)

      batch_loss.append(loss_val.detach().cpu().item())

      # Training accuracy
      act_total_out = torch.sum(spk_rec, 0)  # sum over time
      _, neuron_max_act_total_out = torch.max(act_total_out, 1)  # argmax over output units to compare to labels
      batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())

      # Gradient calculation + weight update
      optimizer.zero_grad()
      loss_val.backward()
      optimizer.step()

    epoch_loss = np.mean(batch_loss)
    epoch_acc = np.mean(batch_acc)
    
    return [epoch_loss, epoch_acc]


def val_test_loop(dataset, batch_size, net, loss_fn, device, shuffle=True, saved_state_dict=None, label_probabilities=False, regularization=None):
  
  with torch.no_grad():
    if saved_state_dict != None:
        net.load_state_dict(saved_state_dict)
    net.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    batch_loss = []
    batch_acc = []

    for data, labels in loader:
        data = data.to(device).swapaxes(1, 0)
        labels = labels.to(device)

        spk_out, hid_rec = net(data)

        # Validation loss
        if regularization != None:
            # L1 loss on spikes per neuron from the hidden layer
            reg_loss = regularization[0]*torch.mean(torch.sum(hid_rec, 0))
            # L2 loss on total number of spikes from the hidden layer
            reg_loss = reg_loss + regularization[1]*torch.mean(torch.sum(torch.sum(hid_rec, dim=0), dim=1)**2)
            loss_val = loss_fn(spk_out, labels) + reg_loss
        else:
            loss_val = loss_fn(spk_out, labels)

        batch_loss.append(loss_val.detach().cpu().item())

        # Accuracy
        act_total_out = torch.sum(spk_out, 0)  # sum over time
        _, neuron_max_act_total_out = torch.max(act_total_out, 1)  # argmax over output units to compare to labels
        batch_acc.extend((neuron_max_act_total_out == labels).detach().cpu().numpy())
    
    if label_probabilities:
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        log_p_y = log_softmax_fn(act_total_out)
        return [np.mean(batch_loss), np.mean(batch_acc)], torch.exp(log_p_y)
    else:
        return [np.mean(batch_loss), np.mean(batch_acc)]
    
### PREPARE FOR TRAINING

num_epochs = 500

batch_size = 64

input_size = 12 
num_steps = next(iter(ds_test))[0].shape[0]

net = model_build(parameters, input_size, num_steps, device)

loss_fn = SF.ce_count_loss()

optimizer = torch.optim.Adam(net.parameters(), lr=parameters["lr"], betas=(0.9, 0.999))

### TRAINING (with validation and test)

print("Training started on: {}-{}-{} {}:{}:{}\n".format(
    training_datetime[:4],
    training_datetime[4:6],
    training_datetime[6:8],
    training_datetime[-6:-4],
    training_datetime[-4:-2],
    training_datetime[-2:])
    )

training_results = []
validation_results = []

for ee in range(num_epochs):

    train_loss, train_acc = training_loop(ds_train, batch_size, net, optimizer, loss_fn, device, regularization=regularization)
    val_loss, val_acc = val_test_loop(ds_val, batch_size, net, loss_fn, device, regularization=regularization)

    training_results.append([train_loss, train_acc])
    validation_results.append([val_loss, val_acc])

    if (ee == 0) | ((ee+1)%10 == 0):
        print("\tepoch {}/{} done \t --> \ttraining accuracy (loss): {}% ({}), \tvalidation accuracy (loss): {}% ({})".format(ee+1,num_epochs,np.round(training_results[-1][1]*100,4), training_results[-1][0], np.round(validation_results[-1][1]*100,4), validation_results[-1][0]))
        
    if val_acc >= np.max(np.array(validation_results)[:,1]):
        best_val_layers = copy.deepcopy(net.state_dict())
training_hist = np.array(training_results)
validation_hist = np.array(validation_results)

# best training and validation at best training
acc_best_train = np.max(training_hist[:,1])
epoch_best_train = np.argmax(training_hist[:,1])
acc_val_at_best_train = validation_hist[epoch_best_train][1]

# best validation and training at best validation
acc_best_val = np.max(validation_hist[:,1])
epoch_best_val = np.argmax(validation_hist[:,1])
acc_train_at_best_val = training_hist[epoch_best_val][1]

print("\n")
print("Overall results:")
print("\tBest training accuracy: {}% ({}% corresponding validation accuracy) at epoch {}/{}".format(
    np.round(acc_best_train*100,4), np.round(acc_val_at_best_train*100,4), epoch_best_train+1, num_epochs))
print("\tBest validation accuracy: {}% ({}% corresponding training accuracy) at epoch {}/{}".format(
    np.round(acc_best_val*100,4), np.round(acc_train_at_best_val*100,4), epoch_best_val+1, num_epochs))
print("\n")
    
# Test
test_results = val_test_loop(ds_test, batch_size, net, loss_fn, device, shuffle=False, saved_state_dict=best_val_layers, regularization=regularization)
print("Test accuracy: {}%\n".format(np.round(test_results[1]*100,2)))

# Ns single-sample inferences to check label probabilities
Ns = 10
for ii in range(Ns):
    single_sample = next(iter(DataLoader(ds_test, batch_size=1, shuffle=True)))
    _, lbl_probs = val_test_loop(TensorDataset(single_sample[0],single_sample[1]), 1, net, loss_fn, device, shuffle=False, saved_state_dict=best_val_layers, label_probabilities=True, regularization=regularization)
    print("Single-sample inference {}/{} from test set:".format(ii+1,Ns))
    print("Sample: {} \tPrediction: {}".format(letter_written[single_sample[1]],letter_written[torch.max(lbl_probs.cpu(),1)[1]]))
    print("Label probabilities (%): {}\n".format(np.round(np.array(lbl_probs.detach().cpu().numpy())*100,2)))
# Store the trained weights
if store_weights:
    torch.save(best_val_layers, "./retrained_snntorch_{}.pt".format(training_datetime))
    print("*** weights stored ***")

# Store the training/validation histories
if store_training:
    #array of shape (num_epochs, 2) with columns [loss, accuracy]
    torch.save(training_hist, "./training_history_{}.pt".format(training_datetime))
if store_validation:
    #array of shape (num_epochs, 2) with columns [loss, accuracy]
    torch.save(validation_hist, "./validation_history_{}.pt".format(training_datetime))