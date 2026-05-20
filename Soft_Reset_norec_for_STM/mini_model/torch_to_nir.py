import nir
import numpy as np

beta = 0.8
dt = 1e-4

# Linear weights: (out_features, in_features) = (5, 3)
weights = np.array([
    [0.5,  0.2, -0.1],   # neuron 0
    [0.3, -0.4,  0.8],   # neuron 1
    [-0.6, 0.1,  0.2],   # neuron 2
    [0.7,  0.5, -0.3],   # neuron 3
    [0.2, -0.2,  0.4],   # neuron 4
], dtype=np.float32)

nodes = {

    # Input vector of size 3
    "input": nir.Input(np.array([3])),

    # Fully-connected layer: 3 -> 5
    "linear": nir.Affine(
        weight=weights,
        bias=np.zeros(5, dtype=np.float32)
    ),

    # LIF neurons
    "lif": nir.LIF(
        tau=np.full((5,), dt / (1.0 - beta), dtype=np.float32),
        v_threshold=np.ones((5,), dtype=np.float32),
        v_leak=np.zeros((5,), dtype=np.float32),
        r=np.ones((5,), dtype=np.float32)
    ),

    # Output vector of size 5
    "output": nir.Output(np.array([5]))
}

edges = [
    ("input",  "linear"),
    ("linear", "lif"),
    ("lif",    "output")
]

graph = nir.NIRGraph(nodes, edges, metadata={'reset_mechanism': 'subtract'})

# Write graph
nir.write("mini_linear.nir", graph)

# Read back for verification
nir.read("mini_linear.nir")

print("nir.read passed!")