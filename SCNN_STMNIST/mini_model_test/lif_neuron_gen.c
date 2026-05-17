#include "stm32h7xx_hal.h"
#include "../Inc/lif_neuron_gen.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "../Inc/usart.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Network architecture from NIR
// Input size: 9
// Layers: 1
// Layer 0: 9 -> 8 (convolutional, no recurrent, uniform params)

// Global variables for the SNN
#define NUM_INPUTS 9
#define NUM_INPUT_CHANNEL 1
#define L1_OUT_CH      2
#define L1_IN_CH       1
#define L1_KERNEL_H    2
#define L1_KERNEL_W    2
#define L1_KERNEL_SIZE 4
#define L1_STRIDE_H    1
#define L1_STRIDE_W    1
#define L1_PAD_H       0
#define L1_PAD_W       0
#define L1_OUT_H       2
#define L1_OUT_W       2
#define L1_COL_BUF_SIZE 8  // 2 * in_ch * kH * kW
#define NUM_NEURONS_LAYER1 8

static __attribute__((aligned(32))) LIFNeuron layer1[NUM_NEURONS_LAYER1];
static __attribute__((aligned(32))) q7_t l1_spikes[NUM_NEURONS_LAYER1];

static __attribute__((aligned(32))) q15_t weights1[L1_OUT_CH * L1_IN_CH * L1_KERNEL_H * L1_KERNEL_W]; // Conv connected

// Utility functions for USART printing
void usart1_print(const char* str) {
    HAL_UART_Transmit(&huart3, (uint8_t*)str, strlen(str), 1000);
}

void print_float(const char* prefix, float_t value) {
    char buf[100];
    int int_part = (int)value;
    int frac_part = (int)((fabs(value) - fabs((float)int_part)) * 10000); // 4 decimal places
    
    // Handle negative numbers between -1 and 0
    if (value < 0.0f && int_part == 0) {
        snprintf(buf, sizeof(buf), "%s-%d.%04d\r\n", prefix, int_part, frac_part);
    } else {
        snprintf(buf, sizeof(buf), "%s%d.%04d\r\n", prefix, int_part, frac_part);
    }
    usart1_print(buf);
}


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
    q15_t temp1[num_neurons], temp2[num_neurons], temp3[num_neurons];
    
    arm_sub_q15(membrane_potentials, reset_values, temp1, num_neurons);
    arm_mult_q15(temp1, decay_factors, temp2, num_neurons);
    arm_add_q15(reset_values, temp2, temp3, num_neurons);
    arm_add_q15(temp3, weighted_inputs, membrane_potentials, num_neurons);


    // Spike-reset: SUBTRACT threshold instead of resetting to reset_value
    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            membrane_potentials[i] -= thresholds[i];
        } else {
            output_spikes[i] = 0;
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

    // Spike-reset: SUBTRACT threshold instead of resetting to reset_value
    for (uint16_t i = 0; i < num_neurons; i++) {
        if (membrane_potentials[i] > thresholds[i]) {
            output_spikes[i] = 1;
            membrane_potentials[i] -= thresholds[i];
        } else {
            output_spikes[i] = 0;
        }
        neurons[i].membrane_potential = membrane_potentials[i];
    }
}



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

                // // TODO: DELETE THIS DEBUG PRINT FROM GENERATOR WHEN ITS FULLY WORKING
                // // WILL FOLLOW WITH DEBUG TO SEE THE MEMBRANE POTENTIAL FOR THAT SPECIFIC NEURON.
                // // oc == 10 oh == 0 ow == 0, makes index 490 for the first layer. 90 for the second layer.
                // // I will watch the membrane potential of the firts layer's this neuron.
                //if (n_idx == 0 && oc == 0 && oh == 0 && ow == 0 ) {
                //    char buf[200];
                //    // Use %ld for q31_t (long int) to avoid format warnings
                //    // We print the raw integer. 60 = 1.0 in float terms.
                //    snprintf(buf, sizeof(buf), "V:%ld = v_shifted:%ld + acc: %ld - Reset:%ld | threshold: %d S:%d | nindex = %ld | v_prev = %ld | decay = %ld \r\n", 
                //            (long)neurons[n_idx].membrane_potential, v_shifted, acc, reset, neurons[n_idx].threshold, 
                //             output_spikes[n_idx], n_idx, v_prev, decay);
                //    usart1_print(buf);
                //}


            }
        }
    }
}
        

void Load_NIR_Weights(void) {
    const float scale = 60.0f;

    // Layer 1 conv weights - Conv2d (2x1x2x2)
    // Stored in OUT_CH-MAJOR order: [oc][ic][kh][kw]
    static const float conv1_weights_vector[8] = {
        5.0000e-01f, 2.0000e-01f, -1.0000e-01f, 3.0000e-01f, 1.0000e-01f, -5.0000e-01f, 8.0000e-01f, 4.0000e-01f
    };

    // Convert and store feedforward weights
    for (int j = 0; j < 8; j++) {
        float scaled = conv1_weights_vector[j] / scale;
        arm_float_to_q15(&scaled, &weights1[j], 1);
    }

}

void SNN_Init(void) {
    const float scale = 60.0f;

    // Layer 1 initialization
    // Uniform parameters for all neurons
    q15_t threshold_1, reset_value_1, decay_factor_1;
    float threshold_f_1 = 1.0000e+00 / scale;
    float reset_value_f_1 = 0.0000e+00 / scale;
    float beta_1 = 8.0000e-01f;

    arm_float_to_q15(&threshold_f_1, &threshold_1, 1);
    arm_float_to_q15(&reset_value_f_1, &reset_value_1, 1);
    arm_float_to_q15(&beta_1, &decay_factor_1, 1);

    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        LIFNeuron_Init(&layer1[i], threshold_1, reset_value_1);
        layer1[i].decay_factor = decay_factor_1;
    }

    // Load weights from NIR
    Load_NIR_Weights();

}

void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes) {
    // Layer 1 (convolutional)
    LIFNeuron_Conv2d_Update_Subtract_Base(layer1, input_spikes, weights1, l1_spikes, 3, 3, 1, 2, 2, 2, 2, 2, 1, 0);

    // Copy output spikes
    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        output_spikes[i] = l1_spikes[i];
    }
}

void SNN_Reset_State(void) {
    // Reset layer 1
    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        layer1[i].membrane_potential = layer1[i].reset_value;
        l1_spikes[i] = 0;
    }

}
