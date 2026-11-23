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
"""

import nir
import numpy as np
import os
import argparse
from typing import Dict, List, Tuple

DEBUG = True

class NIRToCGenerator:
    def __init__(self, nir_file_path: str, output_prefix: str = "snn", opt_conv: bool = True, conv_mode: str = "tap"):
        """
        Initialize the NIR to C code generator.
        
        Args:
            nir_file_path: Path to the .nir file
            output_prefix: Prefix for output files (default: "snn")
            opt_conv: Whether to use optimized code generation (default: True)
            conv_mode: Convolution optimization mode: 'tap' or 'csr' (default: 'tap')
        """
        self.nir_graph = nir.read(nir_file_path)
        self.output_prefix = output_prefix
        self.scale_factor = 300.0  # Q15 scaling factor
        self.opt_conv = opt_conv
        self.conv_mode = conv_mode.lower()
        print(f"Is optimized: {self.opt_conv}")
        if self.opt_conv:
            print(f"Convolution mode: {self.conv_mode}")
        
        # Extract network architecture
        self.layers = []
        self.analyze_network()
    
    @staticmethod
    def _format_weight(value: float) -> str:
        """Format weight value in scientific notation with 4 decimal digits."""
        return f"{value:.4e}f"
        
    @staticmethod
    def _build_conv_csr(in_h, in_w, in_c,
                        out_h, out_w, out_c,
                        kh, kw,
                        stride_h, stride_w,
                        padding_h, padding_w):
        """Build the CSR (input-indexed) connectivity table for one conv layer."""
        num_inputs = in_c * in_h * in_w
        num_outputs = out_c * out_h * out_w
        num_weights = out_c * in_c * kh * kw
    
        if num_inputs > 0xFFFF or num_outputs > 0xFFFF:
            raise ValueError(
                f"CSR build: num_inputs={num_inputs} or num_outputs={num_outputs} "
                f"exceeds uint16_t range"
            )
        if num_weights > 0xFFFF:
            raise ValueError(
                f"CSR build: num_weights={num_weights} exceeds uint16_t weight_idx range"
            )
    
        row_ptr = [0] * (num_inputs + 1)
        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    for ic in range(in_c):
                        for fy in range(kh):
                            for fx in range(kw):
                                ih = oh * stride_h + fy - padding_h
                                iw = ow * stride_w + fx - padding_w
                                if ih < 0 or ih >= in_h or iw < 0 or iw >= in_w:
                                    continue
                                input_index = ic * in_h * in_w + ih * in_w + iw
                                row_ptr[input_index + 1] += 1
    
        for i in range(num_inputs):
            row_ptr[i + 1] += row_ptr[i]
    
        num_connections = row_ptr[num_inputs]
        out_idx = [0] * num_connections
        weight_idx = [0] * num_connections
        cursor = list(row_ptr)
    
        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    output_index = oc * out_h * out_w + oh * out_w + ow
                    for ic in range(in_c):
                        for fy in range(kh):
                            for fx in range(kw):
                                ih = oh * stride_h + fy - padding_h
                                iw = ow * stride_w + fx - padding_w
                                if ih < 0 or ih >= in_h or iw < 0 or iw >= in_w:
                                    continue
                                input_index = ic * in_h * in_w + ih * in_w + iw
                                weight_index = (oc * in_c * kh * kw +
                                                ic * kh * kw +
                                                fy * kw + fx)
                                c = cursor[input_index]
                                out_idx[c] = output_index
                                weight_idx[c] = weight_index
                                cursor[input_index] = c + 1
    
        return row_ptr, out_idx, weight_idx, num_connections

    @staticmethod
    def _build_conv_taps(in_h, in_w, out_h, out_w,
                         kh, kw,
                         stride_h, stride_w,
                         padding_h, padding_w,
                         dil_h=1, dil_w=1):
        """Build the per-input-pixel 'tap table' of one conv layer."""
        in_spatial = in_h * in_w
        out_spatial = out_h * out_w

        if in_spatial > 0xFFFF or out_spatial > 0xFFFF:
            raise ValueError(
                f"Tap table: in_h*in_w={in_spatial} or out_h*out_w={out_spatial} "
                f"exceeds uint16_t range")
        if kh * kw > 0x100:
            raise ValueError(f"Tap table: kh*kw={kh * kw} exceeds uint8_t range")

        rows = [[] for _ in range(in_spatial)]
        for oh in range(out_h):
            for ow in range(out_w):
                sp_out = oh * out_w + ow
                for fy in range(kh):
                    ih = oh * stride_h + fy * dil_h - padding_h
                    if ih < 0 or ih >= in_h:
                        continue
                    for fx in range(kw):
                        iw = ow * stride_w + fx * dil_w - padding_w
                        if iw < 0 or iw >= in_w:
                            continue
                        rows[ih * in_w + iw].append((sp_out, fy * kw + fx))

        tap_ptr, tap_out, tap_w = [0], [], []
        for row in rows:
            for sp_out, tap in row:
                tap_out.append(sp_out)
                tap_w.append(tap)
            tap_ptr.append(len(tap_out))

        num_taps = len(tap_out)
        if num_taps > 0xFFFF:
            raise ValueError(f"Tap table: num_taps={num_taps} exceeds uint16_t range")
        return tap_ptr, tap_out, tap_w, num_taps
    
    @staticmethod
    def _format_int_array(name: str, ctype: str, values, size_expr: str, per_line: int = 16) -> str:
        """Emit `static const <ctype> <name>[<size_expr>] = { ... };` with wrapped lines."""
        lines = [f"static const {ctype} {name}[{size_expr}] = {{"]
        for idx in range(0, len(values), per_line):
            chunk = values[idx: idx + per_line]
            row = "    " + ", ".join(str(v) for v in chunk)
            if idx + per_line < len(values):
                row += ","
            lines.append(row)
        lines.append("};")
        return "\n".join(lines)
    
    def analyze_network(self):
        """Analyze the NIR graph to extract layer information."""
        print("Analyzing NIR graph structure...")

        metadata = getattr(self.nir_graph, 'metadata', {}) or {}
        self.reset_mechanism = metadata.get('reset_mechanism', 'zero')
        print(f"Reset mechanism: {self.reset_mechanism}")
        
        adjacency = {}
        for src, dst in self.nir_graph.edges:
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(dst)
        
        input_node = self.nir_graph.nodes.get('input')
        if input_node is None:
            raise ValueError("No input node found in NIR graph")
        
        if isinstance(input_node.input_type, dict):
            input_type_value = list(input_node.input_type.values())[0]
            if isinstance(input_type_value, np.ndarray) and input_type_value.ndim > 0:
                if len(input_type_value) == 1:
                    self.num_inputs = int(input_type_value[0])
                    self.input_shape = (self.num_inputs,)
                else:
                    self.input_shape = tuple(int(x) for x in input_type_value)
                    self.num_inputs = int(np.prod(self.input_shape))
            else:
                self.num_inputs = int(input_type_value)
                self.input_shape = (self.num_inputs,)
        elif isinstance(input_node.input_type, np.ndarray):
            if len(input_node.input_type) == 1:
                self.num_inputs = int(input_node.input_type[0])
                self.input_shape = (self.num_inputs,)
            else:
                self.input_shape = tuple(int(x) for x in input_node.input_type)
                self.num_inputs = int(np.prod(self.input_shape))
        else:
            self.num_inputs = int(input_node.input_type)
        
        print(f"Input shape: {self.input_shape}, total elements: {self.num_inputs}")
        
        current = 'input'
        layer_idx = 0
        visited_lif_nodes = set()
        
        while current in adjacency:
            next_nodes = adjacency[current]
            affine_node = None
            for node_name in next_nodes:
                node = self.nir_graph.nodes[node_name]
                if (isinstance(node, (nir.Affine, nir.Linear, nir.Conv2d)) and 'rec' not in node_name):
                    affine_node = node_name
                    break
                if isinstance(node, nir.Flatten):
                    flatten_successors = adjacency.get(node_name, [])
                    for f_node_name in flatten_successors:
                        f_node = self.nir_graph.nodes[f_node_name]
                        if isinstance(f_node, (nir.Affine, nir.Linear, nir.Conv2d)):
                            affine_node = f_node_name
                            break
                    if affine_node: break
            
            if affine_node is None:
                break
            
            if affine_node not in adjacency:
                break
                
            lif_node = None
            for node_name in adjacency[affine_node]:
                if isinstance(self.nir_graph.nodes[node_name], nir.LIF):
                    lif_node = node_name
                    break
            
            if lif_node is None or lif_node in visited_lif_nodes:
                break
            
            visited_lif_nodes.add(lif_node)
            
            has_recurrent = False
            recurrent_weights = None
            if lif_node in adjacency:
                for next_node in adjacency[lif_node]:
                    node = self.nir_graph.nodes[next_node]
                    if 'rec' in next_node and isinstance(node, (nir.Affine, nir.Linear)):
                        has_recurrent = True
                        recurrent_weights = node.weight
                        if not np.allclose(recurrent_weights, np.diag(np.diag(recurrent_weights))):
                            raise ValueError(f"Layer {layer_idx}: Only diagonal (1-to-1) recurrent connections are supported")
                        break
            
            affine = self.nir_graph.nodes[affine_node]
            lif = self.nir_graph.nodes[lif_node]
            weights_to_store = affine.weight
            is_conv = (weights_to_store.ndim == 4)

            tau       = np.asarray(lif.tau).reshape(-1)
            threshold = np.asarray(lif.v_threshold).reshape(-1)
            v_leak    = np.asarray(lif.v_leak).reshape(-1)
            v_reset   = np.asarray(lif.v_reset).reshape(-1)

            if is_conv:
                if self.reset_mechanism != 'subtract':
                    raise NotImplementedError(
                        f"Layer {layer_idx} ('{affine_node}') is convolutional, but "
                        f"reset_mechanism='{self.reset_mechanism}'. Conv2d layers currently "
                        f"support only reset-by-subtract."
                    )

                out_c, in_c, kh, kw = weights_to_store.shape
                in_shape = self.nir_graph.nodes[affine_node].input_type['input']
                is_one_to_one = False
                stride_h, stride_w = getattr(affine, 'stride', (1, 1))
                padding_h, padding_w = getattr(affine, 'padding', (0, 0))
                dil_h, dil_w = getattr(affine, 'dilation', (1, 1))

                if stride_h != stride_w or padding_h != padding_w:
                    raise NotImplementedError(
                        f"Layer {layer_idx} ('{affine_node}'): asymmetric stride {(stride_h, stride_w)} / "
                        f"padding {(padding_h, padding_w)} is not supported."
                    )

                in_h = in_shape[1]
                in_w = in_shape[2]
                out_h = int((in_h + 2 * padding_h - dil_h * (kh - 1) - 1) / stride_h + 1)
                out_w = int((in_w + 2 * padding_w - dil_w * (kw - 1) - 1) / stride_w + 1)
            else:
                is_one_to_one = (affine.weight.shape[0] == affine.weight.shape[1] and 
                            np.allclose(affine.weight, np.diag(np.diag(affine.weight))))
                if is_one_to_one:
                    weights_to_store = np.diag(affine.weight)
                else:
                    weights_to_store = affine.weight
            
            has_bias = hasattr(affine, 'bias') and affine.bias is not None
            if has_bias and not np.allclose(affine.bias, 0.0):
                print(f"WARNING: Layer {layer_idx} has non-zero bias values and will be ignored!")
            
            dt = 1e-4
            beta = 1.0 - dt / tau
            
            if is_conv:
                in_shape = self.nir_graph.nodes[affine_node].input_type['input']
                total_inputs = int(np.prod(in_shape))
                n_inputs = total_inputs
                n_neurons = int(np.prod(lif.v_threshold.shape))
            else:
                n_inputs = weights_to_store.shape[1]
                n_neurons = weights_to_store.shape[0]

            layer_info = {
                'index': layer_idx,
                'affine_name': affine_node,
                'lif_name': lif_node,
                'type': 'conv2d' if is_conv else 'linear',
                'num_inputs': n_inputs,
                'num_neurons': n_neurons,
                'weights': weights_to_store,
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
                'is_conv': is_conv,

                # CSR fields
                'row_ptr': None,
                'out_idx': None,
                'weight_idx': None,
                'num_connections': None,

                # Tap fields
                'tap_ptr': None,
                'tap_out': None,
                'tap_w': None,
                'num_taps': None,

                'tau': tau,
                'threshold': threshold,
                'v_leak': v_leak,
                'v_reset': v_reset,
                'beta': beta,
                'has_recurrent': has_recurrent,
                'recurrent_weights': np.diag(recurrent_weights) if has_recurrent else None
            }

            if self.opt_conv and is_conv:
                if self.conv_mode == 'tap':
                    if (dil_h, dil_w) != (1, 1):
                        pass
                    tap_ptr, tap_out, tap_w, num_taps = self._build_conv_taps(
                        in_h, in_w, out_h, out_w,
                        kh, kw,
                        stride_h, stride_w,
                        padding_h, padding_w,
                        dil_h, dil_w,
                    )
                    layer_info['tap_ptr'] = tap_ptr
                    layer_info['tap_out'] = tap_out
                    layer_info['tap_w'] = tap_w
                    layer_info['num_taps'] = num_taps

                    if DEBUG:
                        print(f"Tap table: {num_taps} taps "
                              f"(= {num_taps * in_c * out_c} full connections the old CSR stored)")
                else:
                    if (dil_h, dil_w) != (1, 1):
                        raise NotImplementedError(
                            f"Layer {layer_idx} ('{affine_node}'): dilation {(dil_h, dil_w)} is only "
                            f"supported with the optimized tap table convolution.")
                    row_ptr, out_idx, weight_idx, num_connections = self._build_conv_csr(
                        in_h, in_w, in_c,
                        out_h, out_w, out_c,
                        kh, kw,
                        stride_h, stride_w,
                        padding_h, padding_w,
                    )
                    layer_info['row_ptr'] = row_ptr
                    layer_info['out_idx'] = out_idx
                    layer_info['weight_idx'] = weight_idx
                    layer_info['num_connections'] = num_connections
            
                    if DEBUG:
                        print(f"CSR: {num_connections} connections "
                            f"({num_connections / (out_c*out_h*out_w*in_c*kh*kw):.1%} density)")

            layer_info['uniform_params'] = (
                np.all(tau == tau[0]) and
                np.all(threshold == threshold[0]) and
                np.all(v_leak == v_leak[0]) and
                np.all(v_reset == v_reset[0])
            )
            
            self.layers.append(layer_info)
            layer_idx += 1
            current = lif_node
        
        print(f"Found {len(self.layers)} layers")

    def generate_header_file(self) -> str:
        """Generate the .h header file content."""
        conv_func_decl = ""
        if not self.opt_conv:
            conv_func_decl = """void LIFNeuron_Conv2d_Update_Subtract_Base(LIFNeuron* neurons,         
    const q7_t* input_spikes,  // Input feature map [In_CH * In_H * In_W]
    const q15_t* weights,       // Weights [Out_CH * In_CH * KH * KW]
    q7_t* output_spikes,        // Output spikes [Out_CH * Out_H * Out_W]
    uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w, uint16_t out_ch,
    uint16_t kh, uint16_t kw,
    uint16_t stride, uint16_t padding
);"""
        elif self.conv_mode == 'tap':
            conv_func_decl = """void LIFNeuron_Conv2d_Update_Subtract_Tap(
    LIFNeuron* neurons,
    const q7_t* input_spikes,     // [in_ch][in_h*in_w]
    const q15_t* weights,         // [out_ch][in_ch][kh*kw]
    const uint16_t* tap_ptr,      // [in_spatial + 1]
    const uint16_t* tap_out,      // [num_taps]
    const uint8_t*  tap_w,        // [num_taps]
    uint16_t in_ch,
    uint16_t in_spatial,
    uint16_t out_ch,
    uint16_t out_spatial,
    uint16_t kernel_size,
    q7_t* output_spikes
);"""
        else:
            conv_func_decl = """void LIFNeuron_Conv2d_Update_Subtract_CSR(
    LIFNeuron* neurons,
    const q7_t* input_spikes,
    const q15_t* weights,
    const uint16_t* row_ptr,
    const uint16_t* out_idx,
    const uint16_t* weight_idx,
    uint16_t num_inputs,
    uint16_t num_outputs,
    q7_t* output_spikes
);"""

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

void usart1_print(const char* str);
void print_float(const char* prefix, float_t value);
void LIFNeuron_Init(LIFNeuron* neuron, q15_t threshold, q15_t reset_value);

void LIFNeuron_Layer_Update_Vectorized(LIFNeuron* neurons, const q7_t* input_spikes, 
                                     const q15_t* weights, uint16_t num_inputs, 
                                     uint16_t num_neurons, q7_t* output_spikes,
                                     const q7_t* recurrent_spikes, const q15_t* recurrent_weights,
                                     uint8_t is_one_to_one);

void LIFNeuron_Layer_Update_Vectorized_NoRecurrent(LIFNeuron* neurons, const q7_t* input_spikes, 
                                                  const q15_t* weights, uint16_t num_inputs, 
                                                  uint16_t num_neurons, q7_t* output_spikes,
                                                  uint8_t is_one_to_one);

void LIFNeuron_Layer_Update_Subtract(LIFNeuron* neurons, const q7_t* input_spikes,
                                     const q15_t* weights, uint16_t num_inputs,
                                     uint16_t num_neurons, q7_t* output_spikes,
                                     const q7_t* recurrent_spikes, const q15_t* recurrent_weights,
                                     uint8_t is_one_to_one);

void LIFNeuron_Layer_Update_Subtract_NoRecurrent(LIFNeuron* neurons, const q7_t* input_spikes,
                                                  const q15_t* weights, uint16_t num_inputs,
                                                  uint16_t num_neurons, q7_t* output_spikes,
                                                  uint8_t is_one_to_one);

{conv_func_decl}

void Load_NIR_Weights(void);
void SNN_Init(void);
void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes);
void SNN_Reset_State(void);

#endif // LIF_NEURON_GEN_H
"""
        return h_content
    
    def generate_c_file(self) -> str:
        """Generate the .c implementation file content."""
        layer_defs = self._generate_layer_definitions()
        weight_defs = self._generate_weight_definitions()
        
        if self.opt_conv:
            if self.conv_mode == 'tap':
                table_defs = self._generate_tap_definitions()
                conv_lif_funcs = self._generate_sparse_conv_lif_functions()
            else:
                table_defs = self._generate_csr_definitions()
                conv_lif_funcs = self._generate_csr_conv_lif_functions()
        else:
            table_defs = ""
            conv_lif_funcs = self._generate_naive_conv_lif_functions()

        utility_funcs = self._generate_utility_functions()
        lif_funcs = self._generate_lif_functions()
        weight_load_func = self._generate_weight_loading_function()
        snn_init = self._generate_snn_init()
        snn_timestep = self._generate_snn_timestep()
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
{self._generate_architecture_comment()}

{layer_defs}

{weight_defs}

{table_defs}

{utility_funcs}

{lif_funcs}

{conv_lif_funcs}

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
            if layer['is_one_to_one']:
                conn_str = "1-to-1"
            elif layer['is_conv']:
                conn_str = "convolutional"
            else:
                conn_str = "fully connected"
            param_str = "uniform params" if layer['uniform_params'] else "per-neuron params"
            lines.append(f"// Layer {layer['index']}: {layer['num_inputs']} -> {layer['num_neurons']} ({conn_str}, {rec_str}, {param_str})")
        return '\n'.join(lines)
    
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
                
                if self.opt_conv:
                    if self.conv_mode == 'tap':
                        lines.append(f"#define L{i+1}_IN_SPATIAL  {layer['in_h'] * layer['in_w']}")
                        lines.append(f"#define L{i+1}_OUT_SPATIAL {layer['out_h'] * layer['out_w']}")
                        lines.append(f"#define L{i+1}_NUM_TAPS    {layer['num_taps']}")
                    else:
                        lines.append(f"#define L{i+1}_NUM_CONNECTIONS {layer['num_connections']}")

            lines.append(f"#define NUM_NEURONS_LAYER{i+1} {layer['num_neurons']}")
        
        lines.append("")
        layer_arrays = ", ".join([f"layer{i+1}[NUM_NEURONS_LAYER{i+1}]" for i in range(len(self.layers))])
        lines.append(f"static __attribute__((aligned(32))) LIFNeuron {layer_arrays};")
        
        for i in range(len(self.layers)):
            lines.append(f"static __attribute__((aligned(32))) q7_t l{i+1}_spikes[NUM_NEURONS_LAYER{i+1}];")
        
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"static __attribute__((aligned(32))) q7_t l{i+1}_spikes_prev[NUM_NEURONS_LAYER{i+1}];")
        
        return '\n'.join(lines)

    def _generate_csr_definitions(self) -> str:
        """Generate static CSR connectivity tables."""
        if not self.opt_conv or self.conv_mode != 'csr':
            return ""
        lines = []
        for i, layer in enumerate(self.layers):
            if not layer['is_conv']:
                continue
            input_size_str = "NUM_INPUTS" if i == 0 else f"NUM_NEURONS_LAYER{i}"
            lines.append(f"// Layer {i+1} CSR sparse connectivity table (precomputed in Python)")
            lines.append(self._format_int_array(
                f"row_ptr{i+1}", "uint16_t", layer['row_ptr'],
                size_expr=f"{input_size_str} + 1"))
            lines.append(self._format_int_array(
                f"out_idx{i+1}", "uint16_t", layer['out_idx'],
                size_expr=f"L{i+1}_NUM_CONNECTIONS"))
            lines.append(self._format_int_array(
                f"weight_idx{i+1}", "uint16_t", layer['weight_idx'],
                size_expr=f"L{i+1}_NUM_CONNECTIONS"))
            lines.append("")
        return '\n'.join(lines)

    def _generate_tap_definitions(self) -> str:
        """Generate static tap tables for every conv layer."""
        if not self.opt_conv or self.conv_mode != 'tap':
            return ""
        lines = []
        for i, layer in enumerate(self.layers):
            if not layer['is_conv']:
                continue
            n = i + 1
            lines.append(f"// Layer {n} conv tap table (precomputed in Python).")
            lines.append(self._format_int_array(
                f"tap_ptr{n}", "uint16_t", layer['tap_ptr'], size_expr=f"L{n}_IN_SPATIAL + 1"))
            lines.append(self._format_int_array(
                f"tap_out{n}", "uint16_t", layer['tap_out'], size_expr=f"L{n}_NUM_TAPS"))
            lines.append(self._format_int_array(
                f"tap_w{n}", "uint8_t", layer['tap_w'], size_expr=f"L{n}_NUM_TAPS"))
            lines.append("")
        return '\n'.join(lines)
    
    def _generate_weight_definitions(self) -> str:
        """Generate weight array definitions."""
        lines = []
        for i, layer in enumerate(self.layers):
            if layer['is_one_to_one']:
                if i == 0:
                    lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[NUM_INPUTS]; // 1-to-1 connection (vector)")
                else:
                    lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[NUM_NEURONS_LAYER{i}]; // 1-to-1 connection (vector)")
            elif layer['is_conv']:
                lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[L{i+1}_OUT_CH * L{i+1}_IN_CH * L{i+1}_KERNEL_H * L{i+1}_KERNEL_W]; // Conv connected")
            else:
                if i == 0:
                    lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[NUM_INPUTS*NUM_NEURONS_LAYER{i+1}]; // Fully connected")
                else:
                    lines.append(f"static __attribute__((aligned(32))) q15_t weights{i+1}[NUM_NEURONS_LAYER{i}*NUM_NEURONS_LAYER{i+1}]; // Fully connected")
        
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"static __attribute__((aligned(32))) q15_t recurrent_weights{i+1}[NUM_NEURONS_LAYER{i+1}]; // Recurrent 1-to-1 (vector)")
        
        return '\n'.join(lines)
    
    def _generate_utility_functions(self) -> str:
        """Generate utility functions for USART printing."""
        return """void usart1_print(const char* str) {
    HAL_UART_Transmit(&huart3, (uint8_t*)str, strlen(str), 1000);
}

void print_float(const char* prefix, float_t value) {
    char buf[100];
    int int_part = (int)value;
    int frac_part = (int)((fabs(value) - fabs((float)int_part)) * 10000);
    if (value < 0.0f && int_part == 0) {
        snprintf(buf, sizeof(buf), "%s-%d.%04d\\r\\n", prefix, int_part, frac_part);
    } else {
        snprintf(buf, sizeof(buf), "%s%d.%04d\\r\\n", prefix, int_part, frac_part);
    }
    usart1_print(buf);
}"""

    def _generate_lif_functions(self) -> str:
        """Generate standard LIF neuron update functions."""
        return """
void LIFNeuron_Init(LIFNeuron* neuron, q15_t threshold, q15_t reset_value) {
    neuron->threshold = threshold;
    neuron->reset_value = reset_value;
    neuron->membrane_potential = reset_value;
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

    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    arm_fill_q15(0, weighted_inputs, num_neurons);

    if (is_one_to_one) {
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }
    
    if (recurrent_spikes != NULL && recurrent_weights != NULL) {
        for (uint16_t i = 0; i < num_neurons; i++) {
            if (recurrent_spikes[i]) {
                arm_add_q15(&weighted_inputs[i], &recurrent_weights[i], &weighted_inputs[i], 1);
            }
        }
    }

    q15_t temp1[num_neurons], temp2[num_neurons], temp3[num_neurons];
    arm_sub_q15(membrane_potentials, reset_values, temp1, num_neurons);
    arm_mult_q15(temp1, decay_factors, temp2, num_neurons);
    arm_add_q15(reset_values, temp2, temp3, num_neurons);
    arm_add_q15(temp3, weighted_inputs, membrane_potentials, num_neurons);

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

    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    arm_fill_q15(0, weighted_inputs, num_neurons);

    if (is_one_to_one) {
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }

    q15_t temp1[num_neurons], temp2[num_neurons], temp3[num_neurons];
    arm_sub_q15(membrane_potentials, reset_values, temp1, num_neurons);
    arm_mult_q15(temp1, decay_factors, temp2, num_neurons);
    arm_add_q15(reset_values, temp2, temp3, num_neurons);
    arm_add_q15(temp3, weighted_inputs, membrane_potentials, num_neurons);

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

    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    arm_fill_q15(0, weighted_inputs, num_neurons);

    if (is_one_to_one) {
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }
    
    if (recurrent_spikes != NULL && recurrent_weights != NULL) {
        for (uint16_t i = 0; i < num_neurons; i++) {
            if (recurrent_spikes[i]) {
                arm_add_q15(&weighted_inputs[i], &recurrent_weights[i], &weighted_inputs[i], 1);
            }
        }
    }

    q15_t temp1[num_neurons], temp2[num_neurons];
    arm_mult_q15(membrane_potentials, decay_factors, temp1, num_neurons);
    arm_add_q15(temp1, weighted_inputs, temp2, num_neurons);
    arm_sub_q15(temp2, reset_values, membrane_potentials, num_neurons);

    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            neurons[i].reset_value = thresholds[i];
        } else {
            output_spikes[i] = 0;
            neurons[i].reset_value = 0;
        }
        neurons[i].membrane_potential = membrane_potentials[i];
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

    for (uint16_t i = 0; i < num_neurons; i++) {
        membrane_potentials[i] = neurons[i].membrane_potential;
        reset_values[i] = neurons[i].reset_value;
        decay_factors[i] = neurons[i].decay_factor;
        thresholds[i] = neurons[i].threshold;
    }

    arm_fill_q15(0, weighted_inputs, num_neurons);

    if (is_one_to_one) {
        for (uint16_t i = 0; i < num_inputs && i < num_neurons; i++) {
            if (input_spikes[i]) {
                arm_add_q15(&weighted_inputs[i], &weights[i], &weighted_inputs[i], 1);
            }
        }
    } else {
        for (uint16_t i = 0; i < num_inputs; i++) {
            if (input_spikes[i]) {
                arm_add_q15(weighted_inputs, &weights[i * num_neurons], weighted_inputs, num_neurons);
            }
        }
    }

    q15_t temp1[num_neurons], temp2[num_neurons];
    arm_mult_q15(membrane_potentials, decay_factors, temp1, num_neurons);
    arm_add_q15(temp1, weighted_inputs, temp2, num_neurons);
    arm_sub_q15(temp2, reset_values, membrane_potentials, num_neurons);

    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            neurons[i].reset_value = thresholds[i];
        } else {
            output_spikes[i] = 0;
            neurons[i].reset_value = 0;
        }
        neurons[i].membrane_potential = membrane_potentials[i];
    }
}
"""

    def _generate_naive_conv_lif_functions(self) -> str:
        """Generates the Naive Convolutional SNN layer function."""
        return """
void LIFNeuron_Conv2d_Update_Subtract_Base(LIFNeuron* neurons,
    const q7_t* input_spikes,
    const q15_t* weights,
    q7_t* output_spikes,
    uint16_t in_h, uint16_t in_w,
    uint16_t in_ch,
    uint16_t out_h, uint16_t out_w,
    uint16_t out_ch,
    uint16_t kh, uint16_t kw,
    uint16_t stride,
    uint16_t padding
) {
    for (uint16_t oc = 0; oc < out_ch; oc++) {
        for (uint16_t oh = 0; oh < out_h; oh++) {
            for (uint16_t ow = 0; ow < out_w; ow++) {
                q31_t acc = 0; 
                for (uint16_t ic = 0; ic < in_ch; ic++) {
                    for (uint16_t fy = 0; fy < kh; fy++) {
                        for (uint16_t fx = 0; fx < kw; fx++) {
                            int16_t ih = oh * stride + fy - padding;
                            int16_t iw = ow * stride + fx - padding;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                uint32_t input_idx = (ic * in_h * in_w) + (ih * in_w) + iw;
                                uint32_t weight_idx = (oc * in_ch * kh * kw) + (ic * kh * kw) + (fy * kw) + fx;
                                acc += (q31_t)input_spikes[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }

                uint32_t n_idx = (oc * out_h * out_w) + (oh * out_w) + ow;
                q31_t v_prev    = (q31_t)neurons[n_idx].membrane_potential;
                q31_t reset     = (q31_t)neurons[n_idx].reset_value;
                q31_t decay     = (q31_t)neurons[n_idx].decay_factor;
                q31_t threshold = (q31_t)neurons[n_idx].threshold;

                q31_t v_shifted = (v_prev * decay) >> 15;
                q31_t v_new     = v_shifted + acc - reset;

                neurons[n_idx].membrane_potential = (q15_t)__SSAT(v_new, 16);

                if (neurons[n_idx].membrane_potential > neurons[n_idx].threshold) {
                    output_spikes[n_idx] = 1;
                    neurons[n_idx].reset_value = (q15_t)threshold;
                } else {
                    output_spikes[n_idx] = 0;
                    neurons[n_idx].reset_value = 0;
                }
            }
        }
    }
}
"""

    def _generate_csr_conv_lif_functions(self) -> str:
        """Generates the CSR sparse convolutional LIF update function."""
        return """
void LIFNeuron_Conv2d_Update_Subtract_CSR(
    LIFNeuron* neurons,
    const q7_t* input_spikes,
    const q15_t* weights,
    const uint16_t* row_ptr,
    const uint16_t* out_idx,
    const uint16_t* weight_idx,
    uint16_t num_inputs,
    uint16_t num_outputs,
    q7_t* output_spikes
) {
    for (uint16_t o = 0; o < num_outputs; o++) {
        q31_t decayed = ((q31_t)neurons[o].membrane_potential *
                        (q31_t)neurons[o].decay_factor) >> 15;
        neurons[o].membrane_potential = (q15_t)__SSAT(decayed, 16);
    }

    for (uint16_t i = 0; i < num_inputs; i++) {
        if (input_spikes[i] == 0) {
            continue;
        }
        for (uint16_t c = row_ptr[i]; c < row_ptr[i + 1]; c++) {
            const uint16_t o = out_idx[c];
            const uint16_t w = weight_idx[c];
            q31_t v = (q31_t)neurons[o].membrane_potential + (q31_t)input_spikes[i] * (q31_t)weights[w];
            neurons[o].membrane_potential = (q15_t)__SSAT(v, 16);
        }
    }

    for (uint16_t o = 0; o < num_outputs; o++) {
        neurons[o].membrane_potential -= neurons[o].reset_value;
        if (neurons[o].membrane_potential > neurons[o].threshold) {
            output_spikes[o] = 1;
            neurons[o].reset_value = neurons[o].threshold;
        } else {
            output_spikes[o] = 0;
            neurons[o].reset_value = 0;
        }
    }
}
"""

    def _generate_sparse_conv_lif_functions(self) -> str:
        """Generates the event-driven, tap-table based convolutional LIF update function."""
        return """
void LIFNeuron_Conv2d_Update_Subtract_Tap(
    LIFNeuron* neurons,           
    const q7_t* input_spikes,     
    const q15_t* weights,         
    const uint16_t* tap_ptr,      
    const uint16_t* tap_out,      
    const uint8_t*  tap_w,        
    uint16_t in_ch,
    uint16_t in_spatial,          
    uint16_t out_ch,
    uint16_t out_spatial,         
    uint16_t kernel_size,         
    q7_t* output_spikes           
) {
    const uint32_t num_outputs = (uint32_t)out_ch * out_spatial;
    const uint32_t w_oc_stride = (uint32_t)in_ch * kernel_size;

    for (uint32_t o = 0; o < num_outputs; o++) {
        q31_t decayed = ((q31_t)neurons[o].membrane_potential *
                         (q31_t)neurons[o].decay_factor) >> 15;
        neurons[o].membrane_potential = (q15_t)__SSAT(decayed, 16);
    }

    for (uint16_t ic = 0; ic < in_ch; ic++) {
        const q7_t*  in_plane = input_spikes + (uint32_t)ic * in_spatial;
        const q15_t* w_ic     = weights      + (uint32_t)ic * kernel_size;

        for (uint16_t p = 0; p < in_spatial; p++) {
            const q31_t s = in_plane[p];
            if (s == 0) {
                continue;
            }

            for (uint16_t t = tap_ptr[p]; t < tap_ptr[p + 1]; t++) {
                LIFNeuron*   n0 = neurons + tap_out[t];
                const q15_t* w0 = w_ic + tap_w[t];

                for (uint16_t oc = 0; oc < out_ch; oc++) {
                    LIFNeuron* n = &n0[(uint32_t)oc * out_spatial];
                    q31_t v = (q31_t)n->membrane_potential +
                              s * (q31_t)w0[(uint32_t)oc * w_oc_stride];
                    n->membrane_potential = (q15_t)__SSAT(v, 16);
                }
            }
        }
    }

    for (uint32_t o = 0; o < num_outputs; o++) {
        q31_t v = (q31_t)neurons[o].membrane_potential - (q31_t)neurons[o].reset_value;
        v = __SSAT(v, 16);
        neurons[o].membrane_potential = (q15_t)v;

        if (v > neurons[o].threshold) {
            output_spikes[o] = 1;
            neurons[o].reset_value = neurons[o].threshold;
        } else {
            output_spikes[o] = 0;
            neurons[o].reset_value = 0;
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
        
        for i, layer in enumerate(self.layers):
            if layer['is_one_to_one']:
                lines.append(f"    // Layer {i+1} weights - 1-to-1 connection")
                lines.append(f"    static const float fc{i+1}_weights_vector[{layer['weights'].shape[0]}] = {{")
                weights_flat = layer['weights']
                for idx, w in enumerate(weights_flat):
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(weights_flat) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(weights_flat) - 1:
                        lines.append("")
                lines.append("    };")
            elif layer['is_conv']:
                out_ch, in_ch, kh, kw = layer['weights'].shape
                total_w = out_ch * in_ch * kh * kw
                lines.append(f"    // Layer {i+1} conv weights - Conv2d")
                lines.append(f"    static const float conv{i+1}_weights_vector[{total_w}] = {{")
                weights_flat = layer['weights'].reshape(-1)
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
                lines.append(f"    // Layer {i+1} feedforward weights - fully connected")
                lines.append(f"    static const float fc{i+1}_weights_vector[{layer['num_inputs'] * layer['num_neurons']}] = {{")
                weights_flat = layer['weights'].T.flatten()
                for idx, w in enumerate(weights_flat):
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(weights_flat) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(weights_flat) - 1:
                        lines.append("")
                lines.append("    };")
            lines.append("")
        
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    static const float recurrent_weights_layer{i+1}[{layer['num_neurons']}] = {{")
                rec_weights = layer['recurrent_weights']
                for idx, w in enumerate(rec_weights):
                    if idx % 8 == 0:
                        lines.append("        ")
                    lines[-1] += self._format_weight(w)
                    if idx < len(rec_weights) - 1:
                        lines[-1] += ", "
                    if (idx + 1) % 8 == 0 and idx < len(rec_weights) - 1:
                        lines.append("")
                lines.append("    };")
                lines.append("")
        
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

        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
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
        
        for i, layer in enumerate(self.layers):
            lines.append(f"    // Layer {i+1} initialization")
            if layer['uniform_params']:
                lines.append(f"    q15_t threshold_{i+1}, reset_value_{i+1}, decay_factor_{i+1};")
                lines.append(f"    float threshold_f_{i+1} = {self._format_weight(layer['threshold'].flat[0])[:-1]} / scale;")
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
        
        lines.append("    Load_NIR_Weights();")
        lines.append("")
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

        if self.reset_mechanism == 'subtract':
            update_func_rec    = "LIFNeuron_Layer_Update_Subtract"
            update_func_norec  = "LIFNeuron_Layer_Update_Subtract_NoRecurrent"
        else:
            update_func_rec    = "LIFNeuron_Layer_Update_Vectorized"
            update_func_norec  = "LIFNeuron_Layer_Update_Vectorized_NoRecurrent"
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                input_var = "input_spikes"
                input_size = "NUM_INPUTS"
            else:
                input_var = f"l{i}_spikes"
                input_size = f"NUM_NEURONS_LAYER{i}"
            
            is_one_to_one_flag = "1" if layer['is_one_to_one'] else "0"
            
            if layer['has_recurrent']:
                lines.append(f"    // Layer {i+1} with recurrent connections")
                lines.append(f"    {update_func_rec}(layer{i+1}, {input_var}, weights{i+1}, {input_size}, NUM_NEURONS_LAYER{i+1}, l{i+1}_spikes, l{i+1}_spikes_prev, recurrent_weights{i+1}, {is_one_to_one_flag});")
            elif layer['is_conv']:
                if not self.opt_conv:
                    lines.append(f"    // Layer {i+1} (convolutional naive)")
                    lines.append(f"    LIFNeuron_Conv2d_Update_Subtract_Base(layer{i+1}, {input_var}, weights{i+1}, l{i+1}_spikes, "
                                f"{layer['in_h']}, {layer['in_w']}, {layer['in_c']}, "
                                f"{layer['out_h']}, {layer['out_w']}, {layer['out_c']}, "
                                f"{layer['kh']}, {layer['kw']}, "
                                f"{layer['stride_h']}, {layer['padding_h']});")
                elif self.conv_mode == 'tap':
                    lines.append(f"    // Layer {i+1} (convolutional, tap-table event-driven)")
                    lines.append(
                            f"    LIFNeuron_Conv2d_Update_Subtract_Tap(layer{i+1}, {input_var}, weights{i+1}, "
                            f"tap_ptr{i+1}, tap_out{i+1}, tap_w{i+1}, "
                            f"L{i+1}_IN_CH, L{i+1}_IN_SPATIAL, L{i+1}_OUT_CH, L{i+1}_OUT_SPATIAL, "
                            f"L{i+1}_KERNEL_SIZE, l{i+1}_spikes);")
                else:
                    lines.append(f"    // Layer {i+1} (convolutional, CSR event-driven)")
                    lines.append(
                            f"    LIFNeuron_Conv2d_Update_Subtract_CSR(layer{i+1}, {input_var}, weights{i+1}, "
                            f"row_ptr{i+1}, out_idx{i+1}, weight_idx{i+1}, "
                            f"{input_size}, NUM_NEURONS_LAYER{i+1}, l{i+1}_spikes);")
            else:
                lines.append(f"    // Layer {i+1} (no recurrent)")
                lines.append(f"    {update_func_norec}(layer{i+1}, {input_var}, weights{i+1}, {input_size}, NUM_NEURONS_LAYER{i+1}, l{i+1}_spikes, {is_one_to_one_flag});")
            lines.append("")
        
        for i, layer in enumerate(self.layers):
            if layer['has_recurrent']:
                lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
                lines.append(f"        l{i+1}_spikes_prev[i] = l{i+1}_spikes[i];")
                lines.append("    }")
                lines.append("")
        
        last_layer_idx = len(self.layers)
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
            lines.append(f"    for (int i = 0; i < NUM_NEURONS_LAYER{i+1}; i++) {{")
            if self.reset_mechanism == 'subtract':
                lines.append(f"        layer{i+1}[i].membrane_potential = 0;")
                lines.append(f"        layer{i+1}[i].reset_value = 0;")
            else:
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
        
        h_content = self.generate_header_file()
        with open(h_file, 'w') as f:
            f.write(h_content)
        print(f"✓ Generated: {h_file}")
        
        c_content = self.generate_c_file()
        with open(c_file, 'w') as f:
            f.write(c_content)
        print(f"✓ Generated: {c_file}")
        
        print("\n" + "="*70)
        print("CODE GENERATION COMPLETE")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate C code from a NIR file."
    )

    parser.add_argument(
        "--nir_file",
        default="stmnist_with_reset.nir",
        help="Path to the NIR file"
    )

    parser.add_argument(
        "--optimized",
        type=lambda x: str(x).lower() in ['true', '1', 't', 'y', 'yes'],
        default=True,
        help="Enable or disable optimization (true/false, default: true)"
    )

    parser.add_argument(
        "--conv_mode",
        type=str,
        choices=['csr', 'tap'],
        default='tap',
        help="Convolution optimization mode when --optimized is true: 'tap' or 'csr' (default: tap)"
    )

    args = parser.parse_args()

    nir_file = args.nir_file
    use_optimization = args.optimized
    conv_mode = args.conv_mode

    if not os.path.exists(nir_file):
        print(f"Error: NIR file '{nir_file}' not found!")
        return

    print(f"\nInput NIR file: {nir_file}")
    print(f"Optimization: {'enabled' if use_optimization else 'disabled'}")
    if use_optimization:
        print(f"Convolution mode: {conv_mode}")

    try:
        generator = NIRToCGenerator(nir_file, opt_conv=use_optimization, conv_mode=conv_mode)
        generator.generate_files()
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()