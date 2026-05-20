import torch
import nir
import numpy as np

# 1. LOAD YOUR TRAINED WEIGHTS
# Replace with your actual filename
checkpoint = torch.load('./retrained_snntorch_20260411_175507.pt', map_location='cpu')

# 2. DEFINE ARCHITECTURE DIMENSIONS
# Adjust NUM_HID to 38 or 42 based on your specific trained model
NUM_IN = 12
NUM_HID = 42  
NUM_OUT = 7

# 3. EXTRACT PARAMETERS AS NUMPY ARRAYS
# We pull the raw numbers out of the PyTorch tensors
w_fc1 = checkpoint['fc1.weight'].detach().cpu().numpy()
w_fc2 = checkpoint['fc2.weight'].detach().cpu().numpy()
beta1 = checkpoint['lif1.beta'].detach().cpu().numpy()
beta2 = checkpoint['lif2.beta'].detach().cpu().numpy()
vthr = 1.0

# 4. MANUALLY DEFINE THE NODES
nodes = {
    'input': nir.Input(input_type=np.array([NUM_IN])),
    
    'fc1': nir.Affine(
        weight=w_fc1, 
        bias=checkpoint.get('fc1.bias', torch.zeros(NUM_HID)).detach().cpu().numpy()
    ),
    
    'lif1': nir.LIF(
        tau=np.full(NUM_HID, 1e-4 / (1- beta1)),
        v_threshold=np.full(NUM_HID, vthr),
        v_leak=np.zeros(NUM_HID),
        v_reset=np.zeros(NUM_HID),
        r=np.ones(NUM_HID),
    ),
    
    'fc2': nir.Affine(
        weight=w_fc2, 
        bias=checkpoint.get('fc2.bias', torch.zeros(NUM_OUT)).detach().cpu().numpy()
    ),
    
    'lif2': nir.LIF(
        tau=np.full(NUM_OUT, 1e-4/(1-beta2)),
        v_threshold=np.full(NUM_OUT, vthr),
        v_leak=np.zeros(NUM_OUT),
        v_reset=np.zeros(NUM_OUT),
        r=np.ones(NUM_OUT),
    ),
    
    'output': nir.Output(output_type=np.array([NUM_OUT]))
}

# 5. MANUALLY DEFINE THE EDGES (The Wiring)
edges = [
    ('input', 'fc1'),
    ('fc1', 'lif1'),
    ('lif1', 'fc2'),
    ('fc2', 'lif2'),
    ('lif2', 'output')
]

# 6. ASSEMBLE AND SAVE
# This avoids the automated tracer and creates a clean file
# Metadata is needed for subtract to work. Otherwise the mechanism will be reset by default. 
nir_graph = nir.NIRGraph(nodes=nodes, edges=edges, metadata={'reset_mechanism': 'subtract'})
nir.write("manual_braille_model.nir", nir_graph)

print(f"Success! Manual NIR graph saved to 'manual_braille_model.nir'")