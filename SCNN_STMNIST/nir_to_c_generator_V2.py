# Copyright (C) 2025 Simone Delvecchio
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This work is part of the MSc Thesis: 
# "Optimization of Spiking Neural Networks execution on low-power microcontrollers."
# Politecnico di Torino.
#
# Thesis: https://webthesis.biblio.polito.it/38593/
# GitHub: https://github.com/BlackAqualad/snn2mcu

"""
NIR to C Code Generator for STM32H7 with ARM CMSIS-DSP
This program converts NIR network descriptions to embedded C code (.c and .h files)
following the LIF neuron implementation patterns.

Supports:
- Leaky LIF neurons
- RLeaky (recurrent) LIF neurons with 1-to-1 self-connections
- Fully connected feed-forward layers
- 1-to-1 connections (diagonal matrices stored as vectors)
- Q15 fixed-point arithmetic with ARM CMSIS-DSP

Weight Storage Pattern:
- NIR matrices are [outputs, inputs] = [neurons, inputs]
- C code uses INPUT-MAJOR order: [in0→n0, in0→n1, ..., in1→n0, in1→n1, ...]
- Conversion: weight.T.flatten() (transpose then flatten)
- For 1-to-1 connections: only diagonal values are stored as a vector
- For recurrent: only 1-to-1 is supported, stored as vector

Weight Formatting:
- Scientific notation with 4 decimal digits (e.g., 1.2345e-02f)
- Ensures portability across different architectures
- Matches base file format pattern

Limitations:
- Bias is NOT supported (must be zero in NIR file)
"""

import nir
import numpy as np
import os
from typing import Dict, List, Tuple

DEBUG = True

class NIRToCGenerator:
    def __init__(self, nir_file_path: str, output_prefix: str = "snn"):
        """
        Initialize the NIR to C code generator.
        
        Args:
            nir_file_path: Path to the .nir file
            output_prefix: Prefix for output files (default: "snn")
        """
        self.nir_graph = nir.read(nir_file_path)
        self.output_prefix = output_prefix
        self.scale_factor = 360.0  # Q15 scaling factor
        
        # Extract network architecture
        self.layers = []
        self.analyze_network()
    
    @staticmethod
    def _format_weight(value: float) -> str:
        """
        Format weight value in scientific notation with 4 decimal digits.
        This ensures portability across architectures and matches the base file format.
        
        Args:
            value: Weight value to format
            
        Returns:
            Formatted string like "1.2345e-02f" or "-3.4567e+00f"
        """
        # Use scientific notation with 4 decimal places
        return f"{value:.4e}f"
        
    def analyze_network(self):
        """Analyze the NIR graph to extract layer information."""
        print("Analyzing NIR graph structure...")

        # Read the reset mechanism
        metadata = getattr(self.nir_graph, 'metadata', {}) or {}
        self.reset_mechanism = metadata.get('reset_mechanism', 'zero')
        print(f"Reset mechanism: {self.reset_mechanism}")
        
        # Build adjacency list
        adjacency = {}
        for src, dst in self.nir_graph.edges:
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(dst)
        
        # Find input size
        input_node = self.nir_graph.nodes.get('input')
        if input_node is None:
            raise ValueError("No input node found in NIR graph")
        
        # TODO: Check the changes works with linear input. 
        # input_type is a dictionary with 'input' key containing numpy array
        if isinstance(input_node.input_type, dict):
            # Get the first value from the dictionary
            input_type_value = list(input_node.input_type.values())[0]
            if isinstance(input_type_value, np.ndarray) and input_type_value.ndim > 0:
                if len(input_type_value) == 1:
                    # Flat input: scalar size (original FC-only case)
                    self.num_inputs = int(input_type_value[0])
                    self.input_shape = (self.num_inputs,)
                else:
                    # Spatial input: e.g. [2, 10, 10] → (C, H, W)
                    self.input_shape = tuple(int(x) for x in input_type_value)
                    self.num_inputs = int(np.prod(self.input_shape))  # total elements = 200
            else:
                # Scalar value
                self.num_inputs = int(input_type_value)
                self.input_shape = (self.num_inputs,)

        elif isinstance(input_node.input_type, np.ndarray):
                # input_type is directly a numpy array (no dict wrapper)
            if len(input_node.input_type) == 1:
                self.num_inputs = int(input_node.input_type[0])
                self.input_shape = (self.num_inputs,)
            else:
                self.input_shape = tuple(int(x) for x in input_node.input_type)
                self.num_inputs = int(np.prod(self.input_shape))
        else:
            # If it's a scalar
            self.num_inputs = int(input_node.input_type)
        
        print(f"Input shape: {self.input_shape}, total elements: {self.num_inputs}")
        # Traverse the graph to identify layers (avoiding recurrent loops)
        current = 'input'
        layer_idx = 0
        visited_lif_nodes = set()  # Track visited LIF nodes to avoid infinite loops
        
        while current in adjacency:
            next_nodes = adjacency[current]
            
            # Look for Affine or Linear (weight) or Conv2d node (skip recurrent connections)
            affine_node = None
            for node_name in next_nodes:
                node = self.nir_graph.nodes[node_name]
                if (isinstance(node, (nir.Affine, nir.Linear, nir.Conv2d)) and 'rec' not in node_name):  # Skip recurrent nodes
                    affine_node = node_name
                    break
                # For the flatten layer.
                if isinstance(node, nir.Flatten):
                    # Find the node AFTER Flatten
                    flatten_successors = adjacency.get(node_name, [])
                    for f_node_name in flatten_successors:
                        f_node = self.nir_graph.nodes[f_node_name]
                        if isinstance(f_node, (nir.Affine, nir.Linear, nir.Conv2d)):
                            affine_node = f_node_name
                            break
                    if affine_node: break # Found it through the Flatten node
            
            if affine_node is None:
                break
            
            # Get the LIF node following the Affine
            if affine_node not in adjacency:
                break
                
            lif_node = None
            for node_name in adjacency[affine_node]:
                if isinstance(self.nir_graph.nodes[node_name], nir.LIF):
                    lif_node = node_name
                    break
            
            if lif_node is None or lif_node in visited_lif_nodes:
                break
            
            visited_lif_nodes.add(lif_node)  # Mark as visited
            
            # Check for recurrent connection
            has_recurrent = False
            recurrent_weights = None
            
            if lif_node in adjacency:
                for next_node in adjacency[lif_node]:
                    node = self.nir_graph.nodes[next_node]
                    if 'rec' in next_node and isinstance(node, (nir.Affine, nir.Linear)):
                        # Found recurrent connection
                        has_recurrent = True
                        recurrent_weights = node.weight
                        # Check if it's diagonal (1-to-1)
                        if not np.allclose(recurrent_weights, np.diag(np.diag(recurrent_weights))):
                            raise ValueError(f"Layer {layer_idx}: Only diagonal (1-to-1) recurrent connections are supported")
                        break
            
            # Extract layer information
            affine = self.nir_graph.nodes[affine_node]
            lif = self.nir_graph.nodes[lif_node]
            weights_to_store = affine.weight
            is_conv = (weights_to_store.ndim == 4)

            # Convolutional Case
            if is_conv:
                # weights_to_store shape: [out_channels, in_channels, k_height, k_width]
                out_c, in_c, kh, kw = weights_to_store.shape
                in_shape = self.nir_graph.nodes[affine_node].input_type['input']
                # print("out_c =", out_c, "in_c =", in_c,"kh =", kh, "kw =", kw)
                is_one_to_one = False
                stride_h, stride_w = getattr(affine, 'stride', (1, 1))
                padding_h, padding_w = getattr(affine, 'padding', (0, 0))
                dil_h, dil_w = getattr(affine, 'dilation', (1, 1))

                in_h = in_shape[1]
                in_w = in_shape[2]
                # Standard conv output size formula:
                # out = floor((in + 2*pad - dilation*(kernel-1) - 1) / stride + 1)
                out_h = int((in_h + 2 * padding_h - dil_h * (kh - 1) - 1) / stride_h + 1)
                out_w = int((in_w + 2 * padding_w - dil_w * (kw - 1) - 1) / stride_w + 1)
                print(stride_h, stride_w, padding_h, padding_w, dil_h, dil_w, out_h, out_w, in_h, in_w)

            # Linear Case
            else:
                # Check connection type: fully connected or 1-to-1
                is_one_to_one = (affine.weight.shape[0] == affine.weight.shape[1] and 
                            np.allclose(affine.weight, np.diag(np.diag(affine.weight))))
                
                # Extract weights: if 1-to-1, just take diagonal; if fully connected, keep full matrix
                if is_one_to_one:
                    weights_to_store = np.diag(affine.weight)  # Extract diagonal as 1D vector
                else:
                    weights_to_store = affine.weight  # Keep full matrix
            
            # Check if bias exists and is non-zero (only for Affine, Linear has no bias)
            has_bias = hasattr(affine, 'bias') and affine.bias is not None
            if has_bias and not np.allclose(affine.bias, 0.0):
                print(f"WARNING: Layer {layer_idx} has non-zero bias values. Bias is NOT supported and will be ignored!")
            
            # Calculate beta from NIR tau parameter
            # SNNTorch export_nir.py uses dt = 1e-4 (hardcoded) and tau = dt/(1-beta)
            # To recover beta: beta = 1 - dt/tau
            dt = 1e-4  # Fixed timestep used by snntorch export_nir.py
            beta = 1.0 - dt / lif.tau  # Discrete-time decay factor
            
            # --- Inputs/Neurons Calculation ---
            if is_conv:

                # For n_inputs of conv.
                in_shape = self.nir_graph.nodes[affine_node].input_type['input']
                total_inputs = int(np.prod(in_shape))
                
                # Overwriting
                n_inputs = total_inputs
                n_neurons = int(np.prod(lif.v_threshold.shape))
            else:
                n_inputs = weights_to_store.shape[1]
                n_neurons = weights_to_store.shape[0]

            # Store per-neuron parameters (each neuron can have different values)
            layer_info = {
                'index': layer_idx,
                'affine_name': affine_node,
                'lif_name': lif_node,
                'type': 'conv2d' if is_conv else 'linear',
                'num_inputs':n_inputs,
                'num_neurons': n_neurons,
                'weights': weights_to_store,  # Either 1D vector (1-to-1) or 2D matrix (fully connected) or 4D matrix for conv2d
                # NOTE: bias is NOT supported in the embedded C implementation

                # Conv2d specific attributes (default to None for Linear)
                'kernel_size': (weights_to_store.shape[2], weights_to_store.shape[3]) if is_conv else None,
                'stride_h': stride_h if is_conv else None,
                'stride_w': stride_w if is_conv else None,
                'padding_h': padding_h if is_conv else None,
                'padding_w': padding_w if is_conv else None,

                'out_c': out_c if is_conv else None,
                'in_c': in_c if is_conv else None,
                'in_h': in_h if is_conv else None,
                'in_w': in_w if is_conv else None,
                'kw': kw if is_conv else None,
                'kh': kh if is_conv else None,
                'out_h': out_h if is_conv else None,
                'out_w': out_w if is_conv else None,

                'is_one_to_one': is_one_to_one,
                'is_conv' : is_conv,
                # Per-neuron parameters
                'tau': lif.tau,  # Array of tau values (one per neuron)
                'threshold': lif.v_threshold,  # Array
                'v_leak': lif.v_leak,  # Array
                'v_reset': lif.v_reset,  # Array
                'beta': beta,  # Decay factor: beta = 1 - dt/tau (matches snntorch)
                'has_recurrent': has_recurrent,
                'recurrent_weights': np.diag(recurrent_weights) if has_recurrent else None  # 1D vector
            }

            if DEBUG:
                print(f"--- Layer {layer_info['index']} Information ---")
                print(f"Affine Node:    {layer_info['affine_name']}")
                print(f"LIF Node:       {layer_info['lif_name']}")
                print(f"Is One-to-One:  {layer_info['is_one_to_one']}")
                print(f"Is conv:        {layer_info['is_conv']}")
                print(f"Num Inputs:     {layer_info['num_inputs']}")
                print(f"Num Neurons:    {layer_info['num_neurons']}")
                print(f"Weights Shape:  {layer_info['weights'].shape}")
                #print(f"Tau:            {layer_info['tau']}")
                #print(f"Threshold:      {layer_info['threshold']}")
                #print(f"Beta (Decay):   {layer_info['beta']}")
                print(f"kernel_size:    {layer_info['kernel_size']}")
                print(f"stride:         {layer_info['stride_h']}")
                print(f"padding:        {layer_info['padding_h']}")
                print(f"Has Recurrent:  {layer_info['has_recurrent']}")

                if layer_info['has_recurrent']:
                    print(f"Rec. Weights:   {layer_info['recurrent_weights'].shape}")
                print("-" * 30)
            
            # Check if all neurons in layer have same parameters (for optimization)
            layer_info['uniform_params'] = (
                np.all(lif.tau == lif.tau[0]) and
                np.all(lif.v_threshold == lif.v_threshold[0]) and
                np.all(lif.v_leak == lif.v_leak[0]) and
                np.all(lif.v_reset == lif.v_reset[0])
            )
            
            self.layers.append(layer_info)
            
            if is_conv:
                conn_type = "convolutional"
            elif is_one_to_one:
                conn_type = "1-to-1"
            else:
                conn_type = "fully connected"

            param_type = "uniform" if layer_info['uniform_params'] else "per-neuron"

            if DEBUG:
                print(f"Layer {layer_idx} type: {conn_type}")
                print(f"Parameters:   {param_type}")
                if is_conv:
                    print(f"Stride_h:       {layer_info['stride_h']}")
                    print(f"Stride_w:       {layer_info['stride_w']}")

            
            # To be able to print tau, need to flatten before because the conv tau is a matrix.
            tau_val = layer_info['tau'].flatten()[0]
            if np.isscalar(layer_info['beta']):
                beta_val = layer_info['beta']
            else:
                beta_val = layer_info['beta'].flatten()[0]

            
            # Print layer details for verification
            if layer_info['uniform_params']:
                print(f"Layer {layer_idx}: {layer_info['num_inputs']} -> {layer_info['num_neurons']} neurons, "
                      f"Connection: {conn_type}, Recurrent: {has_recurrent}, Params: {param_type}")
                print(f"  tau={tau_val:.6f}, beta={beta_val:.6f}")
            else:
                print(f"Layer {layer_idx}: {layer_info['num_inputs']} -> {layer_info['num_neurons']} neurons, "
                      f"Connection: {conn_type}, Recurrent: {has_recurrent}, Params: {param_type}")
            
            layer_idx += 1
            current = lif_node
        
        print(f"Found {len(self.layers)} layers")
        
    def generate_header_file(self) -> str:
        """Generate the .h header file content."""
        h_content = f"""#ifndef LIF_NEURON_GEN_H
#define LIF_NEURON_GEN_H

#include <stdint.h>
#include "arm_math.h"

typedef struct {{
    q15_t threshold;     // Firing threshold in Q15
    q15_t reset_value;   // Reset potential in Q15
    q15_t membrane_potential; // Current membrane potential in Q15
    q15_t decay_factor;  // Precomputed beta (decay factor) in Q15
}} LIFNeuron;

// Utility functions
void usart1_print(const char* str);
void print_float(const char* prefix, float_t value);

// LIF Neuron functions
void LIFNeuron_Init(LIFNeuron* neuron, q15_t threshold, q15_t reset_value);

// Layer update functions
void LIFNeuron_Layer_Update_Vectorized(LIFNeuron* neurons, const q7_t* input_spikes, 
                                     const q15_t* weights, uint16_t num_inputs, 
                                     uint16_t num_neurons, q7_t* output_spikes,
                                     const q7_t* recurrent_spikes, const q15_t* recurrent_weights,
                                     uint8_t is_one_to_one);

void LIFNeuron_Layer_Update_Vectorized_NoRecurrent(LIFNeuron* neurons, const q7_t* input_spikes, 
                                                  const q15_t* weights, uint16_t num_inputs, 
                                                  uint16_t num_neurons, q7_t* output_spikes,
                                                  uint8_t is_one_to_one);

// Layer Update functions for Reset-by-Subtract
void LIFNeuron_Layer_Update_Subtract(LIFNeuron* neurons, const q7_t* input_spikes,
                                     const q15_t* weights, uint16_t num_inputs,
                                     uint16_t num_neurons, q7_t* output_spikes,
                                     const q7_t* recurrent_spikes, const q15_t* recurrent_weights,
                                     uint8_t is_one_to_one);

void LIFNeuron_Layer_Update_Subtract_NoRecurrent(LIFNeuron* neurons, const q7_t* input_spikes,
                                                  const q15_t* weights, uint16_t num_inputs,
                                                  uint16_t num_neurons, q7_t* output_spikes,
                                                  uint8_t is_one_to_one);

void LIFNeuron_Conv2d_Update_Subtract_Base(LIFNeuron* neurons,         
    const q7_t* input_spikes,  // Input feature map [In_CH * In_H * In_W]
    const q15_t* weights,       // Weights [Out_CH * In_CH * KH * KW]
    q7_t* output_spikes,        // Output spikes [Out_CH * Out_H * Out_W]
    uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w, uint16_t out_ch,
    uint16_t kh, uint16_t kw,
    uint16_t stride, uint16_t padding
);                                                  

// Weight loading function
void Load_NIR_Weights(void);

// SNN main functions
void SNN_Init(void);
void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes);
void SNN_Reset_State(void);

#endif // LIF_NEURON_GEN_H
"""
        return h_content
    
    def generate_c_file(self) -> str:
        """Generate the .c implementation file content."""
        
        # Generate layer definitions
        layer_defs = self._generate_layer_definitions()
        
        # Generate weight array definitions
        weight_defs = self._generate_weight_definitions()
        
        # Generate utility functions
        utility_funcs = self._generate_utility_functions()
        
        # Generate LIF neuron functions
        lif_funcs = self._generate_lif_functions()

        # Generate Naive Conv LIF neuron functions
        naive_conv_lif_funcs = self._generate_naive_conv_lif_functions()
        
        # Generate weight loading function
        weight_load_func = self._generate_weight_loading_function()
        
        # Generate SNN initialization
        snn_init = self._generate_snn_init()
        
        # Generate SNN timestep function
        snn_timestep = self._generate_snn_timestep()
        
        # Generate SNN reset function
        snn_reset = self._generate_snn_reset()
        
        c_content = f"""#include "stm32h7xx_hal.h"
#include "../Inc/lif_neuron_gen.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "../Inc/usart.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Network architecture from NIR
// Input size: {self.num_inputs}
// Layers: {len(self.layers)}
{self._generate_architecture_comment()}

{layer_defs}

{weight_defs}

{utility_funcs}

{lif_funcs}

{naive_conv_lif_funcs}

{weight_load_func}

{snn_init}

{snn_timestep}

{snn_reset}
"""
        return c_content
    
    def _generate_architecture_comment(self) -> str:
        """Generate architecture description comment."""
        lines = []
        for layer in self.layers:
            rec_str = "with recurrent" if layer['has_recurrent'] else "no recurrent"
            #conn_str = "1-to-1" if layer['is_one_to_one'] else "fully connected"
            if layer['is_one_to_one']:
                conn_str = "1-to-1"
            elif layer ['is_conv']:
                conn_str = "convolutional"
            else:
                conn_str = "fully connected"
            param_str = "uniform params" if layer['uniform_params'] else "per-neuron params"
            lines.append(f"// Layer {layer['index']}: {layer['num_inputs']} -> {layer['num_neurons']} ({conn_str}, {rec_str}, {param_str})")
        return '\n'.join(lines)
    
    # EMRE TODO: Need to check this func
    def _generate_layer_definitions(self) -> str:
        """Generate layer array definitions."""
        lines = [f"// Global variables for the SNN"]
        lines.append(f"#define NUM_INPUTS {self.num_inputs}")
        lines.append(f"#define NUM_INPUT_CHANNEL {self.input_shape[0]}")
        
        for i, layer in enumerate(self.layers):
            if layer['is_conv']:
                lines.append(f"#define L{i+1}_OUT_CH      {layer['out_c']}")
                lines.append(f"#define L{i+1}_IN_CH       {layer['in_c']}")
                lines.append(f"#define L{i+1}_KERNEL_H    {layer['kh']}")
                lines.append(f"#define L{i+1}_KERNEL_W    {layer['kw']}")
                lines.append(f"#define L{i+1}_KERNEL_SIZE {layer['kh'] * layer['kw']}")
                lines.append(f"#define L{i+1}_STRIDE_H    {layer['stride_h']}")
                lines.append(f"#define L{i+1}_STRIDE_W    {layer['stride_w']}")
                lines.append(f"#define L{i+1}_PAD_H       {layer['padding_h']}")
                lines.append(f"#define L{i+1}_PAD_W       {layer['padding_w']}")
                lines.append(f"#define L{i+1}_OUT_H       {layer['out_h']}")
                lines.append(f"#define L{i+1}_OUT_W       {layer['out_w']}")
                # im2col scratch buffer size: 2 * in_ch * kH * kW (CMSIS-NN requirement)
                col_buf_size = 2 * layer['in_c'] * layer['kh'] * layer['kw']
                lines.append(f"#define L{i+1}_COL_BUF_SIZE {col_buf_size}  // 2 * in_ch * kH * kW")

            lines.append(f"#define NUM_NEURONS_LAYER{i+1} {layer['num_neurons']}")
        
        lines.append("")
        
        # Layer neuron arrays
        layer_arrays = ", ".join([f"layer{i+1}[NUM_NEURONS_LAYER{i+1}]" for i in range(len(self.layers))])
        lines.append(f"static __attribute__((aligned(32))) LIFNeuron {layer_arrays};")
        
        # Spike arrays with memory alignment
        for i in range(len(self.layers)):
            lines.append(f"static __attribute__((aligned(32))) q7_t l{i+1}_spikes[NUM_NEURONS_LAYER{i+1}];")
        
        # Previous spike arrays for recurrent layers with memory alignment
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"static __attribute__((aligned(32))) q7_t l{i+1}_spikes_prev[NUM_NEURONS_LAYER{i+1}];")
        
        return '\n'.join(lines)
    
    # Emre: This is okay 
    def _generate_weight_definitions(self) -> str:
        """Generate weight array definitions."""
        lines = []
        
        # Weight arrays with memory alignment
        for i, layer in enumerate(self.layers):
            if layer['is_one_to_one']:
                # 1-to-1 connection: only diagonal values (vector)
                if i == 0:
                    lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[NUM_INPUTS]; // 1-to-1 connection (vector)")
                else:
                    lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[NUM_NEURONS_LAYER{i}]; // 1-to-1 connection (vector)")
            elif layer['is_conv']:
                # Conv2d connection
                lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[L{i+1}_OUT_CH * L{i+1}_IN_CH * L{i+1}_KERNEL_H * L{i+1}_KERNEL_W]; // Conv connected")
            else:
                # Fully connected: full weight matrix
                if i == 0:
                    lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[NUM_INPUTS*NUM_NEURONS_LAYER{i+1}]; // Fully connected")
                else:
                    lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[NUM_NEURONS_LAYER{i}*NUM_NEURONS_LAYER{i+1}]; // Fully connected")
        
        # Recurrent weight arrays (always 1-to-1, stored as vectors) with memory alignment
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"static __attribute__((aligned(32))) q15_t recurrent_weights{i+1}[NUM_NEURONS_LAYER{i+1}]; // Recurrent 1-to-1 (vector)")
        
        return '\n'.join(lines)
    
    def _generate_utility_functions(self) -> str:
        """Generate utility functions for USART printing."""
        return """// Utility functions for USART printing
void usart1_print(const char* str) {
    HAL_UART_Transmit(&huart3, (uint8_t*)str, strlen(str), 1000);
}

void print_float(const char* prefix, float_t value) {
    char buf[100];
    int int_part = (int)value;
    int frac_part = (int)((fabs(value) - fabs((float)int_part)) * 10000); // 4 decimal places
    
    // Handle negative numbers between -1 and 0
    if (value < 0.0f && int_part == 0) {
        snprintf(buf, sizeof(buf), "%s-%d.%04d\\r\\n", prefix, int_part, frac_part);
    } else {
        snprintf(buf, sizeof(buf), "%s%d.%04d\\r\\n", prefix, int_part, frac_part);
    }
    usart1_print(buf);
}"""
    # TODO: NEED TO CHECK
    def _generate_lif_functions(self) -> str:
        """Generate LIF neuron update functions."""
        return """
void LIFNeuron_Init(LIFNeuron* neuron, q15_t threshold, q15_t reset_value) {
    neuron->threshold = threshold;
    neuron->reset_value = reset_value;
    neuron->membrane_potential = reset_value;
    // decay_factor (beta) will be set in SNN_Init
}

void LIFNeuron_Layer_Update_Vectorized(LIFNeuron* neurons, const q7_t* input_spikes, 
                                     const q15_t* weights, uint16_t num_inputs, 
                                     uint16_t num_neurons, q7_t* output_spikes,
                                     const q7_t* recurrent_spikes, const q15_t* recurrent_weights,
                                     uint8_t is_one_to_one) {
    q15_t membrane_potentials[num_neurons];
    q15_t reset_values[num_neurons];
    q15_t decay_factors[num_neurons];
    q15_t thresholds[num_neurons];
    q15_t weighted_inputs[num_neurons];

    // Extract neuron parameters
    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    // Initialize weighted_inputs to zero
    arm_fill_q15(0, weighted_inputs, num_neurons);

    // Calculate weighted input currents (feedforward)
    if (is_one_to_one) {
        // 1-to-1 connection: weights are stored as a vector, each input connects to corresponding neuron
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                // For 1-to-1, weight vector: weights[i] corresponds to connection i->i
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        // Fully connected: each input connects to all neurons
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }
    
    // Add recurrent connections (self-connections from previous timestep, always 1-to-1)
    if (recurrent_spikes != NULL && recurrent_weights != NULL) {
        for (uint16_t i = 0; i < num_neurons; i++) {
            if (recurrent_spikes[i]) {
                // Recurrent weights are stored as vector: recurrent_weights[i] for neuron i's self-loop
                arm_add_q15(&weighted_inputs[i], &recurrent_weights[i], &weighted_inputs[i], 1);
            }
        }
    }

    // Vectorized membrane potential update: V = reset + (V - reset) * beta + weighted_input
    q15_t temp1[num_neurons], temp2[num_neurons], temp3[num_neurons];
    
    arm_sub_q15(membrane_potentials, reset_values, temp1, num_neurons);
    arm_mult_q15(temp1, decay_factors, temp2, num_neurons);
    arm_add_q15(reset_values, temp2, temp3, num_neurons);
    arm_add_q15(temp3, weighted_inputs, membrane_potentials, num_neurons);

    // Check for spikes and reset
    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            membrane_potentials[i] = reset_values[i];
        } else {
            output_spikes[i] = 0;
        }
        neurons[i].membrane_potential = membrane_potentials[i];
    }
}

void LIFNeuron_Layer_Update_Vectorized_NoRecurrent(LIFNeuron* neurons, const q7_t* input_spikes, 
                                                  const q15_t* weights, uint16_t num_inputs, 
                                                  uint16_t num_neurons, q7_t* output_spikes,
                                                  uint8_t is_one_to_one) {
    q15_t membrane_potentials[num_neurons];
    q15_t reset_values[num_neurons];
    q15_t decay_factors[num_neurons];
    q15_t thresholds[num_neurons];
    q15_t weighted_inputs[num_neurons];

    // Extract neuron parameters
    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    // Initialize weighted_inputs to zero
    arm_fill_q15(0, weighted_inputs, num_neurons);

    // Calculate weighted input currents (no recurrent)
    if (is_one_to_one) {
        // 1-to-1 connection: weights are stored as a vector, each input connects to corresponding neuron
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                // For 1-to-1, weight vector: weights[i] corresponds to connection i->i
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        // Fully connected: each input connects to all neurons
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }

    // Vectorized membrane potential update
    q15_t temp1[num_neurons], temp2[num_neurons], temp3[num_neurons];
    
    arm_sub_q15(membrane_potentials, reset_values, temp1, num_neurons);
    arm_mult_q15(temp1, decay_factors, temp2, num_neurons);
    arm_add_q15(reset_values, temp2, temp3, num_neurons);
    arm_add_q15(temp3, weighted_inputs, membrane_potentials, num_neurons);

    // Check for spikes and reset
    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            membrane_potentials[i] = reset_values[i];
        } else {
            output_spikes[i] = 0;
        }
        neurons[i].membrane_potential = membrane_potentials[i];
    }
}

// Same layer update functions, for Reset-by-Subtract
void LIFNeuron_Layer_Update_Subtract(LIFNeuron* neurons, const q7_t* input_spikes, 
                                     const q15_t* weights, uint16_t num_inputs, 
                                     uint16_t num_neurons, q7_t* output_spikes,
                                     const q7_t* recurrent_spikes, const q15_t* recurrent_weights,
                                     uint8_t is_one_to_one) {
    q15_t membrane_potentials[num_neurons];
    q15_t reset_values[num_neurons];
    q15_t decay_factors[num_neurons];
    q15_t thresholds[num_neurons];
    q15_t weighted_inputs[num_neurons];

    // Extract neuron parameters
    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    // Initialize weighted_inputs to zero
    arm_fill_q15(0, weighted_inputs, num_neurons);

    // Calculate weighted input currents (feedforward)
    if (is_one_to_one) {
        // 1-to-1 connection: weights are stored as a vector, each input connects to corresponding neuron
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                // For 1-to-1, weight vector: weights[i] corresponds to connection i->i
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        // Fully connected: each input connects to all neurons
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }
    
    // Add recurrent connections (self-connections from previous timestep, always 1-to-1)
    if (recurrent_spikes != NULL && recurrent_weights != NULL) {
        for (uint16_t i = 0; i < num_neurons; i++) {
            if (recurrent_spikes[i]) {
                // Recurrent weights are stored as vector: recurrent_weights[i] for neuron i's self-loop
                arm_add_q15(&weighted_inputs[i], &recurrent_weights[i], &weighted_inputs[i], 1);
            }
        }
    }

    // Vectorized membrane potential update: V = reset + (V - reset) * beta + weighted_input
    // Update membrane for soft reset:
    // v(t+1) = decay * v(t) + weighted_inputs - reset_value(previous step)
    q15_t temp1[num_neurons], temp2[num_neurons];
    
    arm_mult_q15(membrane_potentials, decay_factors, temp1, num_neurons);
    arm_add_q15(temp1, weighted_inputs, temp2, num_neurons);
    arm_sub_q15(temp2, reset_values, membrane_potentials, num_neurons);

    // Spike check, then store reset_value for the NEXT step
    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            neurons[i].reset_value = thresholds[i];   // subtract next step
        } else {
            output_spikes[i] = 0;
            neurons[i].reset_value = 0;
        }
        neurons[i].membrane_potential = membrane_potentials[i];

        // Debug print for the most spiking output neuron.
        // if(i == 8){
        //     char buf[200];
        //     snprintf(buf, sizeof(buf), "V:%ld = Reset:%d + acc: %hu| threshold: %d S:%d | nindex = %d \\r\\n", 
        //             (long)neurons[i].membrane_potential, neurons[i].reset_value, weighted_inputs[i] , neurons[i].threshold, 
        //                 output_spikes[i], i);
        //     usart1_print(buf);
        // }
        
    }
}

void LIFNeuron_Layer_Update_Subtract_NoRecurrent(LIFNeuron* neurons, const q7_t* input_spikes, 
                                                  const q15_t* weights, uint16_t num_inputs, 
                                                  uint16_t num_neurons, q7_t* output_spikes,
                                                  uint8_t is_one_to_one) {
    q15_t membrane_potentials[num_neurons];
    q15_t reset_values[num_neurons];
    q15_t decay_factors[num_neurons];
    q15_t thresholds[num_neurons];
    q15_t weighted_inputs[num_neurons];

    // Extract neuron parameters
    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    // Initialize weighted_inputs to zero
    arm_fill_q15(0, weighted_inputs, num_neurons);

    // Calculate weighted input currents (no recurrent)
    if (is_one_to_one) {
        // 1-to-1 connection: weights are stored as a vector, each input connects to corresponding neuron
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                // For 1-to-1, weight vector: weights[i] corresponds to connection i->i
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        // Fully connected: each input connects to all neurons
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }

    // Vectorized membrane potential update
    // Update membrane for soft reset:
    // v(t+1) = decay * v(t) + weighted_inputs - reset_value(previous step)
    q15_t temp1[num_neurons], temp2[num_neurons];
    
    arm_mult_q15(membrane_potentials, decay_factors, temp1, num_neurons);
    arm_add_q15(temp1, weighted_inputs, temp2, num_neurons);
    arm_sub_q15(temp2, reset_values, membrane_potentials, num_neurons);

    // Spike check, then store reset_value for the NEXT step
    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            neurons[i].reset_value = thresholds[i];   // subtract next step
        } else {
            output_spikes[i] = 0;
            neurons[i].reset_value = 0;
        }
        neurons[i].membrane_potential = membrane_potentials[i];

        // TODO : Delete this part when the all tests are done.
        // Debug print for the most spiking output neuron.
        // if(i == 8){
        //     char buf[200];
        //     snprintf(buf, sizeof(buf), "V:%ld = Reset:%d + acc: %hu| threshold: %d S:%d | nindex = %d \\r\\n", 
        //             (long)neurons[i].membrane_potential, neurons[i].reset_value, weighted_inputs[i] , neurons[i].threshold, 
        //                 output_spikes[i], i);
        //     usart1_print(buf);
        // }
        
    }
}
"""
    # EMRE TODO: Add hard-reset function.
    def _generate_naive_conv_lif_functions(self) -> str:
        """Generates the C code for Convolutional SNN layers."""
        return """
void LIFNeuron_Conv2d_Update_Subtract_Base(LIFNeuron* neurons,         // Array of neurons for this layer
    const q7_t* input_spikes,  // Input feature map [In_CH * In_H * In_W]
    const q15_t* weights,       // Weights [Out_CH * In_CH * KH * KW]
    q7_t* output_spikes,        // Output spikes [Out_CH * Out_H * Out_W]
    uint16_t in_h, uint16_t in_w,
    uint16_t in_ch,
    uint16_t out_h, uint16_t out_w,
    uint16_t out_ch,
    uint16_t kh, uint16_t kw,
    uint16_t stride,
    uint16_t padding
) {
    // 1. Iterate over every output "pixel" (which is one LIF neuron)
    for (uint16_t oc = 0; oc < out_ch; oc++) {
        for (uint16_t oh = 0; oh < out_h; oh++) {
            for (uint16_t ow = 0; ow < out_w; ow++) {
                
                // Accumulator for the current (this is the weighted input)
                q31_t acc = 0; 

                // 2. Perform the Convolution (Sliding Window)
                for (uint16_t ic = 0; ic < in_ch; ic++) {
                    for (uint16_t fy = 0; fy < kh; fy++) {
                        for (uint16_t fx = 0; fx < kw; fx++) {
                            
                            // Calculate input coordinates
                            int16_t ih = oh * stride + fy - padding;
                            int16_t iw = ow * stride + fx - padding;

                            // Check boundaries (Padding logic)
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                // Indexing for [CH][H][W] format
                                uint32_t input_idx = (ic * in_h * in_w) + (ih * in_w) + iw;
                                // Indexing for [OutCH][InCH][KH][KW] format
                                uint32_t weight_idx = (oc * in_ch * kh * kw) + (ic * kh * kw) + (fy * kw) + fx;

                                acc += (q31_t)input_spikes[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }

                // 3. LIF Neuron Update Logic
                // Index of the specific neuron in the flat array
                uint32_t n_idx = (oc * out_h * out_w) + (oh * out_w) + ow;

                // Pull parameters into q31 to avoid premature saturation
                q31_t v_prev    = (q31_t)neurons[n_idx].membrane_potential;
                q31_t reset     = (q31_t)neurons[n_idx].reset_value; // I will use this reset to achieve soft-reset subtraction in the next timestep.
                q31_t decay     = (q31_t)neurons[n_idx].decay_factor;
                q31_t threshold = (q31_t)neurons[n_idx].threshold;

                // All arithmetic stays in q31 — acc is already in Q15 scale (spike * Q15_weight)
                q31_t v_shifted = (v_prev * decay) >> 15;
                q31_t v_new     = v_shifted + acc - reset;  // acc added here before any saturation
                // Reset value consist the threshold from the last timestep if there was any spike, otherwise its zero. (Uth*S(t)).
                // v_new = ((v_prev)*decay) + acc - reset
                // U(t+1) = (U(t)*Beta) + W*X(t+1) - Uth*S(t)
                // This is the equation from the snntorch tutorial 3. 

                // Only saturate when writing back to the q15_t struct field
                neurons[n_idx].membrane_potential = (q15_t)__SSAT(v_new, 16);


                // SOFT RESET
                if (neurons[n_idx].membrane_potential > neurons[n_idx].threshold) {
                    output_spikes[n_idx] = 1;
                    neurons[n_idx].reset_value    = (q15_t)threshold;  // will subtract next step
                } else {
                    output_spikes[n_idx] = 0;
                    neurons[n_idx].reset_value    = 0;                 // clear if there is no spike
                }

                // TODO: DELETE THIS DEBUG PRINT FROM GENERATOR WHEN ITS FULLY WORKING
                // WILL FOLLOW WITH DEBUG TO SEE THE MEMBRANE POTENTIAL FOR THAT SPECIFIC NEURON.
                // oc == 10 oh == 0 ow == 0, makes index 490 for the first layer. 90 for the second layer.
                // I will watch the membrane potential of the firts layer's this neuron.
                //if (n_idx == 490 && oc == 10 && oh == 0 && ow == 0 ) {
                
                // For Layer 2 most spiking one is C14, H1, W2 which makes the n_idx = (14×3×3)+(1×3)+2 = 131
                // if (n_idx == 131 && oc == 14 && oh == 1 && ow == 2 ) {
                //     char buf[200];
                //     // Use %ld for q31_t (long int) to avoid format warnings
                //     // We print the raw integer. 60 = 1.0 in float terms.
                //     snprintf(buf, sizeof(buf), "V:%ld = Reset:%ld + v_shifted:%ld + acc: %ld| threshold: %d S:%d | nindex = %ld | v_prev = %ld | decay = %ld \\r\\n", 
                //             (long)neurons[n_idx].membrane_potential, reset, v_shifted, acc, neurons[n_idx].threshold, 
                //              output_spikes[n_idx], n_idx, v_prev, decay);
                //     usart1_print(buf);
                // }


            }
        }
    }
}
        """

    
    def _generate_weight_loading_function(self) -> str:
        """Generate function to load weights from NIR data."""
        lines = [
            "void Load_NIR_Weights(void) {",
            f"    const float scale = {self.scale_factor}f;",
            ""
        ]
        
        # Generate weight arrays for each layer
        for i, layer in enumerate(self.layers):
            if layer['is_one_to_one']:
                # 1-to-1: weights is a 1D vector (diagonal values)
                lines.append(f"    // Layer {i+1} weights - 1-to-1 connection (vector of {layer['weights'].shape[0]} values)")
                lines.append(f"    static const float fc{i+1}_weights_vector[{layer['weights'].shape[0]}] = {{")
                
                weights_flat = layer['weights']  # Already 1D vector
                
                for idx, w in enumerate(weights_flat):
                    if idx % 8 == 0:
                        lines.append("        " if idx > 0 else "        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(weights_flat) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(weights_flat) - 1:
                        lines.append("")
                
                lines.append("    };")

            elif layer['is_conv']:
                # Convolutional connection, weights are 4D matrix, 
                #  Conv2D weights (flattened)                
                #  Original shape:
                #    weights[out_ch][in_ch][kh][kw]
                #  Flattened as:
                #    [oc][ic][kh][kw] → 1D array
                #  Index:
                #    idx = oc*(IN_CH*KH*KW) + ic*(KH*KW) + kh*KW + kw
                #  Each output neuron uses:
                #    IN_CH × KH × KW weights (receptive field)
                #  Usage:
                #    sum += weights[oc * RF_SIZE + k] * input_patch[k];
                
                out_ch, in_ch, kh, kw = layer['weights'].shape
                total_w = out_ch * in_ch * kh * kw

                lines.append(f"    // Layer {i+1} conv weights - Conv2d ({out_ch}x{in_ch}x{kh}x{kw})")
                lines.append(f"    // Stored in OUT_CH-MAJOR order: [oc][ic][kh][kw]")
                lines.append(f"    static const float conv{i+1}_weights_vector[{total_w}] = {{")

                weights_flat = layer['weights'].reshape(-1)  # shape: [out_ch, in_ch, kh, kw]

                for idx, w in enumerate(weights_flat):
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(weights_flat) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(weights_flat) - 1:
                        lines.append("")
                lines.append("    };")

            else:
                # Fully connected: weights is 2D matrix, flatten in input-major order
                # NIR format: [neurons, inputs] - need to TRANSPOSE to get [inputs, neurons]
                # Pattern: in0→n0, in0→n1, in0→n2, ..., in1→n0, in1→n1, ...
                lines.append(f"    // Layer {i+1} feedforward weights - fully connected ({layer['num_inputs']}x{layer['num_neurons']})")
                lines.append(f"    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]")
                lines.append(f"    static const float fc{i+1}_weights_vector[{layer['num_inputs'] * layer['num_neurons']}] = {{")
                
                # Transpose NIR matrix [neurons, inputs] to [inputs, neurons], then flatten
                weights_flat = layer['weights'].T.flatten()  # .T converts to input-major order
                
                for idx, w in enumerate(weights_flat):
                    if idx % 8 == 0:
                        lines.append("        " if idx > 0 else "        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(weights_flat) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(weights_flat) - 1:
                        lines.append("")
                
                lines.append("    };")
            lines.append("")
        
        # Generate recurrent weights (always 1-to-1, stored as vectors)
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    // Layer {i+1} recurrent weights - 1-to-1 (vector of {layer['num_neurons']} values)")
                lines.append(f"    static const float recurrent_weights_layer{i+1}[{layer['num_neurons']}] = {{")
                
                rec_weights = layer['recurrent_weights']  # Already 1D vector
                for idx, w in enumerate(rec_weights):
                    if idx % 8 == 0:
                        lines.append("        " if idx > 0 else "        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(rec_weights) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(rec_weights) - 1:
                        lines.append("")
                
                lines.append("    };")
                lines.append("")
        
        # Convert and store weights
        lines.append("    // Convert and store feedforward weights")
        for i, layer in enumerate(self.layers):
            if layer['is_one_to_one']:
                size = layer['weights'].shape[0]
            
                lines.append(f"    for (int i = 0; i < {size}; i++) {{")
                lines.append(f"        float scaled = fc{i+1}_weights_vector[i] / scale;")
                lines.append(f"        arm_float_to_q15(&scaled, &weights{i+1}[i], 1);")
                lines.append("    }")
                lines.append("")

            elif layer['is_conv']:
                out_ch, in_ch, kh, kw = layer['weights'].shape
                size = out_ch * in_ch * kh * kw

                lines.append(f"    for (int j = 0; j < {size}; j++) {{")
                lines.append(f"        float scaled = conv{i+1}_weights_vector[j] / scale;")
                lines.append(f"        arm_float_to_q15(&scaled, &weights{i+1}[j], 1);")
                lines.append("    }")
                lines.append("")
                
            else:
                size = layer['num_inputs'] * layer['num_neurons']
            
                lines.append(f"    for (int i = 0; i < {size}; i++) {{")
                lines.append(f"        float scaled = fc{i+1}_weights_vector[i] / scale;")
                lines.append(f"        arm_float_to_q15(&scaled, &weights{i+1}[i], 1);")
                lines.append("    }")
                lines.append("")

        
        # Convert recurrent weights
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    // Convert recurrent weights (1-to-1)")
                lines.append(f"    for (int i = 0; i < {layer['num_neurons']}; i++) {{")
                lines.append(f"        float scaled = recurrent_weights_layer{i+1}[i] / scale;")
                lines.append(f"        arm_float_to_q15(&scaled, &recurrent_weights{i+1}[i], 1);")
                lines.append("    }")
                lines.append("")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _generate_snn_init(self) -> str:
        """Generate SNN initialization function."""
        lines = [
            "void SNN_Init(void) {",
            f"    const float scale = {self.scale_factor}f;",
            ""
        ]
        
        # Initialize neurons layer by layer with their specific parameters
        for i, layer in enumerate(self.layers):
            lines.append(f"    // Layer {i+1} initialization")
            
            if layer['uniform_params']:
                # All neurons have same parameters - optimize
                lines.append(f"    // Uniform parameters for all neurons")
                lines.append(f"    q15_t threshold_{i+1}, reset_value_{i+1}, decay_factor_{i+1};")
                lines.append(f"    float threshold_f_{i+1} = {self._format_weight(layer['threshold'].flat[0])[:-1]} / scale;")  # Remove 'f' suffix
                lines.append(f"    float reset_value_f_{i+1} = {self._format_weight(layer['v_reset'].flat[0])[:-1]} / scale;")
                lines.append(f"    float beta_{i+1} = {self._format_weight(layer['beta'].flat[0])};")
                lines.append("")
                lines.append(f"    arm_float_to_q15(&threshold_f_{i+1}, &threshold_{i+1}, 1);")
                lines.append(f"    arm_float_to_q15(&reset_value_f_{i+1}, &reset_value_{i+1}, 1);")
                lines.append(f"    arm_float_to_q15(&beta_{i+1}, &decay_factor_{i+1}, 1);")
                lines.append("")
                lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
                lines.append(f"        LIFNeuron_Init(&layer{i+1}[i], threshold_{i+1}, reset_value_{i+1});")
                lines.append(f"        layer{i+1}[i].decay_factor = decay_factor_{i+1};")
                lines.append("    }")
            else:
                # Each neuron has different parameters
                lines.append(f"    // Per-neuron parameters")
                lines.append(f"    float thresholds_f[NUM_NEURONS_LAYER{i+1}] = {{")
                for idx, val in enumerate(layer['threshold']):
                    if idx % 8 == 0 and idx > 0:
                        lines.append("")
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(val)
                    if idx < len(layer['threshold']) - 1:
                        lines[-1] += ", "
                lines.append("    };")
                
                lines.append(f"    float reset_values_f[NUM_NEURONS_LAYER{i+1}] = {{")
                for idx, val in enumerate(layer['v_reset']):
                    if idx % 8 == 0 and idx > 0:
                        lines.append("")
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(val)
                    if idx < len(layer['v_reset']) - 1:
                        lines[-1] += ", "
                lines.append("    };")
                
                lines.append(f"    float betas_f[NUM_NEURONS_LAYER{i+1}] = {{")
                for idx, val in enumerate(layer['beta']):
                    if idx % 8 == 0 and idx > 0:
                        lines.append("")
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(val)
                    if idx < len(layer['beta']) - 1:
                        lines[-1] += ", "
                lines.append("    };")
                lines.append("")
                
                lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
                lines.append(f"        q15_t threshold_q15, reset_q15, beta_q15;")
                lines.append(f"        float thresh_scaled = thresholds_f[i] / scale;")
                lines.append(f"        float reset_scaled = reset_values_f[i] / scale;")
                lines.append(f"        arm_float_to_q15(&thresh_scaled, &threshold_q15, 1);")
                lines.append(f"        arm_float_to_q15(&reset_scaled, &reset_q15, 1);")
                lines.append(f"        arm_float_to_q15(&betas_f[i], &beta_q15, 1);")
                lines.append(f"        LIFNeuron_Init(&layer{i+1}[i], threshold_q15, reset_q15);")
                lines.append(f"        layer{i+1}[i].decay_factor = beta_q15;")
                lines.append("    }")
            
            lines.append("")
        
        # Load weights
        lines.append("    // Load weights from NIR")
        lines.append("    Load_NIR_Weights();")
        lines.append("")
        
        # Initialize previous spike arrays
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    arm_fill_q7(0, l{i+1}_spikes_prev, NUM_NEURONS_LAYER{i+1});")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _generate_snn_timestep(self) -> str:
        """Generate SNN timestep execution function."""
        lines = [
            "void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes) {"
        ]

        # Reset mechanism for function calls
        if self.reset_mechanism == 'subtract':
            update_func_rec    = "LIFNeuron_Layer_Update_Subtract"
            update_func_norec  = "LIFNeuron_Layer_Update_Subtract_NoRecurrent"
        else:  # 'zero' (default)
            update_func_rec    = "LIFNeuron_Layer_Update_Vectorized"
            update_func_norec  = "LIFNeuron_Layer_Update_Vectorized_NoRecurrent"
        
        # Process each layer
        for i, layer in enumerate(self.layers):
            if i == 0:
                input_var = "input_spikes"
                input_size = "NUM_INPUTS"
            else:
                input_var = f"l{i}_spikes"
                input_size = f"NUM_NEURONS_LAYER{i}"
            
            is_one_to_one_flag = "1" if layer['is_one_to_one'] else "0"
            
            if layer['has_recurrent']:
                lines.append(f"    // Layer {i+1} with recurrent connections ({'1-to-1' if layer['is_one_to_one'] else 'fully connected'})")
                lines.append(f"    {update_func_rec}(layer{i+1}, {input_var}, weights{i+1}, {input_size}, NUM_NEURONS_LAYER{i+1}, l{i+1}_spikes, l{i+1}_spikes_prev, recurrent_weights{i+1}, {is_one_to_one_flag});")

            # EMRE TODO: Edit for hard-reset.
            elif layer['is_conv']:
                lines.append(f"    // Layer {i+1} (convolutional)")
                lines.append(f"    LIFNeuron_Conv2d_Update_Subtract_Base(layer{i+1}, {input_var}, weights{i+1}, l{i+1}_spikes, "
                             f"{layer['in_h']}, {layer['in_w']}, {layer['in_c']}, "
                             f"{layer['out_h']}, {layer['out_w']}, {layer['out_c']}, "
                             f"{layer['kh']}, {layer['kw']}, "
                             f"{layer['stride_h']}, {layer['padding_h']});")

            else:
                lines.append(f"    // Layer {i+1} (no recurrent, {'1-to-1' if layer['is_one_to_one'] else 'fully connected'})")
                lines.append(f"    {update_func_norec}(layer{i+1}, {input_var}, weights{i+1}, {input_size}, NUM_NEURONS_LAYER{i+1}, l{i+1}_spikes, {is_one_to_one_flag});")
            lines.append("")
        
        # Store current spikes as previous for recurrent layers
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    // Store spikes for layer {i+1} recurrent connections")
                lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
                lines.append(f"        l{i+1}_spikes_prev[i] = l{i+1}_spikes[i];")
                lines.append("    }")
                lines.append("")
        
        # Copy output
        last_layer_idx = len(self.layers)
        lines.append(f"    // Copy output spikes")
        lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{last_layer_idx}; i++) {{")
        lines.append(f"        output_spikes[i] = l{last_layer_idx}_spikes[i];")
        lines.append("    }")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _generate_snn_reset(self) -> str:
        """Generate SNN state reset function."""
        lines = [
            "void SNN_Reset_State(void) {"
        ]
        
        for i, layer in enumerate(self.layers):
            lines.append(f"    // Reset layer {i+1}")
            lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
            lines.append(f"        layer{i+1}[i].membrane_potential = layer{i+1}[i].reset_value;")
            lines.append(f"        l{i+1}_spikes[i] = 0;")
            if layer['has_recurrent']:
                lines.append(f"        l{i+1}_spikes_prev[i] = 0;")
            lines.append("    }")
            lines.append("")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def generate_files(self, output_dir: str = "."):
        """Generate the .h and .c files."""
        h_file = os.path.join(output_dir, f"lif_neuron_gen.h")
        c_file = os.path.join(output_dir, f"lif_neuron_gen.c")
        
        print(f"\nGenerating files...")
        
        # Generate header
        h_content = self.generate_header_file()
        with open(h_file, 'w') as f:
            f.write(h_content)
        print(f"✓ Generated: {h_file}")
        
        # Generate implementation
        c_content = self.generate_c_file()
        with open(c_file, 'w') as f:
            f.write(c_content)
        print(f"✓ Generated: {c_file}")
        
        # Generate usage example
        self._generate_usage_example(output_dir)
        
        print("\n" + "="*70)
        print("CODE GENERATION COMPLETE")
        print("="*70)
        self._print_summary()
    
    def _generate_usage_example(self, output_dir: str):
        """Generate example usage code."""
        example = f"""/*
 * Example usage of the generated SNN code
 * 
 * This shows how to use the NIR-generated SNN in your main.c
 * 
 * NOTE: Input must be binary spikes (q7_t: 0 or 1)
 */

#include "lif_neuron_gen.h"

void example_usage(void) {{
    // 1. Initialize the SNN (call once at startup)
    SNN_Init();
    
    // 2. Prepare input spikes (size: {self.num_inputs})
    // Input must be binary values: 0 (no spike) or 1 (spike)
    q7_t input_spikes[{self.num_inputs}];
    q7_t output_spikes[{self.layers[-1]['num_neurons']}];
    
    // Example: Set some input spikes
    for (int i = 0; i < {self.num_inputs}; i++) {{
        input_spikes[i] = (i % 2 == 0) ? 1 : 0; // Example pattern
    }}
    
    // 3. Run for multiple timesteps (e.g., 256 timesteps per sample)
    for (int t = 0; t < 256; t++) {{
        SNN_Run_Timestep(input_spikes, output_spikes);
        
        // Process output_spikes as needed
        // output_spikes contains {self.layers[-1]['num_neurons']} values (0 or 1)
    }}
    
    // 4. Reset state between samples
    SNN_Reset_State(); // (Only if testing from dataset, or if you want to reset the network state)
}}

/*
 * Network Architecture:
 * Input: {self.num_inputs} neurons (binary spikes: 0 or 1)
"""
        for i, layer in enumerate(self.layers):
            rec_str = "with 1-to-1 recurrent" if layer['has_recurrent'] else "feedforward only"
            conn_str = "1-to-1 connection" if layer['is_one_to_one'] else "fully connected"
            param_str = "uniform params" if layer['uniform_params'] else "per-neuron params"
            example += f" * Layer {i+1}: {layer['num_neurons']} neurons ({conn_str}, {rec_str}, {param_str})\n"
        
        example += f""" * Output: {self.layers[-1]['num_neurons']} neurons (binary spikes: 0 or 1)
 * 
 * Total parameters: {sum(l['num_inputs'] * l['num_neurons'] for l in self.layers)} feedforward weights
 * Recurrent parameters: {sum(l['num_neurons'] if l['has_recurrent'] else 0 for l in self.layers)} recurrent weights
 * 
 * Input Format:
 * - Binary spikes only: q7_t values must be 0 or 1
 * - If you have analog/float sensor data, convert to binary before calling SNN_Run_Timestep()
 * - Conversion method depends on your application (threshold, rate coding, etc.)
 */
"""
        
        example_file = os.path.join(output_dir, "example_usage.c")
        with open(example_file, 'w') as f:
            f.write(example)
        print(f"✓ Generated: {example_file}")
    
    def _print_summary(self):
        """Print generation summary."""
        print(f"\nNetwork Summary:")
        print(f"  Input size: {self.num_inputs}")
        print(f"  Number of layers: {len(self.layers)}")
        print(f"  Architecture: {self.num_inputs}", end="")
        for layer in self.layers:
            print(f" → {layer['num_neurons']}", end="")
        print()
        
        print(f"\nLayer Details:")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i+1}:")
            print(f"    Neurons: {layer['num_neurons']}")
            print(f"    Connection: {'1-to-1' if layer['is_one_to_one'] else 'Fully connected'}")
            print(f"    Parameters: {'Uniform' if layer['uniform_params'] else 'Per-neuron'}")
            if layer['uniform_params']:
                print(f"    Beta (decay): {layer['beta'].flat[0]:.6f}")
                print(f"    Threshold: {layer['threshold'].flat[0]:.6f}")
                print(f"    Reset: {layer['v_reset'].flat[0]:.6f}")
            else:
                print(f"    Beta (decay): {layer['beta'].flat[0]:.6f} to {layer['beta'].flat[-1]:.6f}")
                print(f"    Threshold: {layer['threshold'].flat[0]:.6f} to {layer['threshold'].flat[-1]:.6f}")
                print(f"    Reset: {layer['v_reset'].flat[0]:.6f} to {layer['v_reset'].flat[-1]:.6f}")
            print(f"    Recurrent: {'Yes (1-to-1)' if layer['has_recurrent'] else 'No'}")
        
        total_params = sum(l['num_inputs'] * l['num_neurons'] for l in self.layers)
        total_rec_params = sum(l['num_neurons'] if l['has_recurrent'] else 0 for l in self.layers)
        print(f"\nTotal Parameters:")
        print(f"  Feedforward weights: {total_params}")
        print(f"  Recurrent weights: {total_rec_params}")
        print(f"  Total: {total_params + total_rec_params}")
        
        print(f"\nFeatures:")
        print(f"  ✓ Float input conversion supported")
        print(f"  ✓ Q15 fixed-point arithmetic (scale: {self.scale_factor})")
        print(f"  ✓ ARM CMSIS-DSP vectorized operations")
        print(f"  ✓ USART debug printing functions included")
        
        print(f"\nFiles generated successfully!")
        print(f"  - lif_neuron_gen.h")
        print(f"  - lif_neuron_gen.c")
        print(f"  - example_usage.c")


def main():
    """Main function to run the NIR to C generator."""
    import sys
    
    print("="*70)
    print("NIR TO C CODE GENERATOR")
    print("="*70)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        nir_file = sys.argv[1]
    else:
        nir_file = 'stmnist_with_reset.nir'
    
    if not os.path.exists(nir_file):
        print(f"Error: NIR file '{nir_file}' not found!")
        print(f"\nUsage: python nir_to_c_generator.py [nir_file.nir]")
        return
    
    print(f"\nInput NIR file: {nir_file}")
    
    try:
        # Create generator
        generator = NIRToCGenerator(nir_file)
        
        # Generate files
        generator.generate_files()
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
