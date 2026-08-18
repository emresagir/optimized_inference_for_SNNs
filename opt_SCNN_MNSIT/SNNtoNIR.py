import torch
import nir
import snntorch as snn
import numpy as np

# 1. Setup & Load Model
device = torch.device("cpu")
beta = 0.95
reset_mechanism = "subtract"

# Define the network exactly as before
scnn_net = torch.nn.Sequential(
    torch.nn.Conv2d(2, 8, kernel_size=4, stride=1, bias=False), # Node '0'
    snn.Leaky(beta=beta, init_hidden=True, reset_mechanism=reset_mechanism), # Node '1'
    torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False), # Node '2'
    snn.Leaky(beta=beta, init_hidden=True, reset_mechanism=reset_mechanism), # Node '3'
    torch.nn.Flatten(start_dim=1), # Node '4'
    torch.nn.Linear(16 * 3 * 3, 10, bias=False), # Node '5'
    snn.Leaky(beta=beta, init_hidden=True, output=True, reset_mechanism=reset_mechanism) # Node '6'
)
scnn_net.load_state_dict(torch.load('best_scnn_net_20260817_171949_acc83.4.pth', map_location=device))
scnn_net.eval()

# 2. Manually create NIR nodes
# We pull the weights directly from the state_dict
sd = scnn_net.state_dict()

nodes = {
    "input": nir.Input(np.array([2, 10, 10])),
    
    "0": nir.Conv2d(
        input_shape=np.array([10, 10]),
        weight=sd['0.weight'].numpy(),
        bias=np.zeros(8, dtype=np.float32), 
        stride=np.array([1, 1]),
        padding=np.array([0, 0]),
        dilation=np.array([1, 1]),
        groups=1
    ),
    
    "1": nir.LIF(
        tau=np.full((8, 7, 7), 1e-4/(1.0-beta)), 
        v_threshold=np.ones((8, 7, 7)),
        v_leak=np.zeros((8, 7, 7)),
        r=np.ones((8, 7, 7))
    ),
    
    "2": nir.Conv2d(
        input_shape=np.array([7, 7]),
        weight=sd['2.weight'].numpy(),
        bias=np.zeros(16, dtype=np.float32), 
        stride=np.array([2, 2]),
        padding=np.array([0, 0]),
        dilation=np.array([1, 1]),
        groups=1
    ),
    
    "3": nir.LIF(
        tau=np.full((16, 3, 3), 1e-4/(1.0-beta)),
        v_threshold=np.ones((16, 3, 3)),
        v_leak=np.zeros((16, 3, 3)),
        r=np.ones((16, 3, 3))
    ),
    
    "4": nir.Flatten(
        input_type={'input': np.array([16,3,3])},
        output_type={'output': np.array([144])},
        # Starts flattening from the beginning otherwise it flattens to 16,9.
        start_dim=0  
    ),
    
    "5": nir.Affine(
        weight=sd['5.weight'].numpy(),
        bias=np.zeros(10, dtype=np.float32),
        input_type={'input': np.array([144])},
        output_type={'output': np.array([10])}
    ),
    
    "6": nir.LIF(
        tau=np.full((10,), 1e-4/(1.0-beta)),
        v_threshold=np.ones((10,)),
        v_leak=np.zeros((10,)),
        r=np.ones((10,)),
        input_type={'input': np.array([10])},
        output_type={'output': np.array([10])}
    ),
    
    "output": nir.Output(np.array([10]))
}

# 3. Define the edges (connections)
edges = [
    ("input", "0"), ("0", "1"), ("1", "2"), ("2", "3"),
    ("3", "4"), ("4", "5"), ("5", "6"), ("6", "output")
]

# 4. Assemble the Graph
nir_graph = nir.NIRGraph(nodes, edges, metadata={'reset_mechanism': 'subtract'})


# 5. Save and Verify
nir.write("stmnist_with_reset.nir", nir_graph)
print("Manual Export Success!")

# This should now work perfectly without any type inference errors
reloaded_graph = nir.read("stmnist_with_reset.nir")
print("Verification Success: nir.read() passed.")