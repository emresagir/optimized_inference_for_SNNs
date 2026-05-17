import nir
import numpy as np

beta = 0.8
dt   = 1e-4

# (out_c, in_c, kH, kW) = (2, 1, 2, 2) — square kernel, so NIR bug is harmless
weights = np.array([
    [[[0.5,  0.2],
      [-0.1, 0.3]]],
    [[[0.1, -0.5],
      [0.8,  0.4]]]
], dtype=np.float32)

nodes = {
    "input": nir.Input(np.array([1, 3, 3])),        # (C, H, W)

    "conv": nir.Conv2d(
        input_shape=np.array([3, 3]),                # input spatial — kH=kW=2 so formula correct for both dims
        weight=weights,
        bias=np.zeros(2, dtype=np.float32),
        stride=np.array([1, 1]),
        padding=np.array([0, 0]),
        dilation=np.array([1, 1]),
        groups=1
    ),                                               # NIR infers output: (2, 2, 2)

    "1": nir.LIF(
        tau=np.full((2, 2, 2), dt / (1.0 - beta)),
        v_threshold=np.ones((2, 2, 2)),
        v_leak=np.zeros((2, 2, 2)),
        r=np.ones((2, 2, 2))
    ),

    "flatten": nir.Flatten(
        input_type={'input': np.array([2, 2, 2])},
        output_type={'output': np.array([8])},
        start_dim=0
    ),

    "output": nir.Output(np.array([8]))
}

edges = [
    ("input",   "conv"),
    ("conv",    "1"),
    ("1",       "flatten"),
    ("flatten", "output")
]

graph = nir.NIRGraph(nodes, edges)
nir.write("mini_conv2d.nir", graph)
nir.read("mini_conv2d.nir")
print("nir.read passed!")