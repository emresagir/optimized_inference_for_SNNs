#ifndef LIF_NEURON_GEN_H
#define LIF_NEURON_GEN_H

#include <stdint.h>
#include "arm_math.h"

typedef struct {
    q15_t threshold;     // Firing threshold in Q15
    q15_t reset_value;   // Reset potential in Q15
    q15_t membrane_potential; // Current membrane potential in Q15
    q15_t decay_factor;  // Precomputed beta (decay factor) in Q15
} LIFNeuron;

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
);


// Weight loading function
void Load_NIR_Weights(void);

// SNN main functions
void SNN_Init(void);
void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes);
void SNN_Reset_State(void);

#endif // LIF_NEURON_GEN_H
