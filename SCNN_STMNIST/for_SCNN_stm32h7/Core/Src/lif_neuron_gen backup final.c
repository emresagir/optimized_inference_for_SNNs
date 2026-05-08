#include "stm32h7xx_hal.h"
#include "../Inc/lif_neuron_gen.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "../Inc/usart.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Network architecture from NIR
// Input size: 200
// Layers: 3
// Layer 0: 200 -> 1568 (convolutional, no recurrent, uniform params)
// Layer 1: 1568 -> 576 (convolutional, no recurrent, uniform params)
// Layer 2: 576 -> 10 (fully connected, no recurrent, uniform params)

// Global variables for the SNN
#define NUM_INPUTS 200
#define NUM_INPUT_CHANNEL 2
#define L1_OUT_CH      32
#define L1_IN_CH       2
#define L1_KERNEL_H    4
#define L1_KERNEL_W    4
#define L1_KERNEL_SIZE 16
#define L1_STRIDE_H    1
#define L1_STRIDE_W    1
#define L1_PAD_H       0
#define L1_PAD_W       0
#define L1_OUT_H       7
#define L1_OUT_W       7
#define L1_COL_BUF_SIZE 64  // 2 * in_ch * kH * kW
#define NUM_NEURONS_LAYER1 1568
#define L2_OUT_CH      64
#define L2_IN_CH       32
#define L2_KERNEL_H    3
#define L2_KERNEL_W    3
#define L2_KERNEL_SIZE 9
#define L2_STRIDE_H    2
#define L2_STRIDE_W    2
#define L2_PAD_H       0
#define L2_PAD_W       0
#define L2_OUT_H       3
#define L2_OUT_W       3
#define L2_COL_BUF_SIZE 576  // 2 * in_ch * kH * kW
#define NUM_NEURONS_LAYER2 576
#define NUM_NEURONS_LAYER3 10

static __attribute__((aligned(32))) LIFNeuron layer1[NUM_NEURONS_LAYER1], layer2[NUM_NEURONS_LAYER2], layer3[NUM_NEURONS_LAYER3];
static __attribute__((aligned(32))) q7_t l1_spikes[NUM_NEURONS_LAYER1];
static __attribute__((aligned(32))) q7_t l2_spikes[NUM_NEURONS_LAYER2];
static __attribute__((aligned(32))) q7_t l3_spikes[NUM_NEURONS_LAYER3];

static __attribute__((aligned(32))) q15_t weights1[L1_OUT_CH * L1_IN_CH * L1_KERNEL_H * L1_KERNEL_W]; // Conv connected
static __attribute__((aligned(32))) q15_t weights2[L2_OUT_CH * L2_IN_CH * L2_KERNEL_H * L2_KERNEL_W]; // Conv connected
static __attribute__((aligned(32))) q15_t weights3[NUM_NEURONS_LAYER2*NUM_NEURONS_LAYER3]; // Fully connected

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
                q31_t reset     = (q31_t)neurons[n_idx].reset_value;
                q31_t decay     = (q31_t)neurons[n_idx].decay_factor;

                // All arithmetic stays in q31 — acc is already in Q15 scale (spike * Q15_weight)
                q31_t v_shifted = ((v_prev - reset) * decay) >> 15;
                q31_t v_new     = reset + v_shifted + acc;  // acc added here before any saturation

                // Only saturate when writing back to the q15_t struct field
                neurons[n_idx].membrane_potential = (q15_t)__SSAT(v_new, 16);
                

                // SOFT RESET
                if (neurons[n_idx].membrane_potential > neurons[n_idx].threshold) {
                    output_spikes[n_idx] = 1;
                    neurons[n_idx].membrane_potential -= neurons[n_idx].threshold;
                } else {
                    output_spikes[n_idx] = 0;
                }

                                                // WILL FOLLOW WITH DEBUG TO SEE THE MEMBRANE POTENTIAL FOR THAT SPECIFIC NEURON.
                // oc == 10 oh == 0 ow == 0, makes index 490 for the first layer. 90 for the second layer.
                // I will watch the membrane potential of the firts layer's this neuron.
                // if (n_idx == 490 && oc == 10 && oh == 0 && ow == 0 ) {
                //     char buf[200];
                //     // Use %ld for q31_t (long int) to avoid format warnings
                //     // We print the raw integer. 60 = 1.0 in float terms.
                //     snprintf(buf, sizeof(buf), "V:%ld = Reset:%ld + v_shifted:%ld + acc: %ld| threshold: %d S:%d | nindex = %ld | v_prev = %ld | decay = %ld \r\n", 
                //             (long)neurons[n_idx].membrane_potential, reset, v_shifted, acc, neurons[n_idx].threshold, output_spikes[n_idx], n_idx, v_prev, decay);
                //     usart1_print(buf);
                // }

            }
        }
    }
}
        

void Load_NIR_Weights(void) {
    const float scale = 60.0f;

    // Layer 1 conv weights - Conv2d (32x2x4x4)
    // Stored in OUT_CH-MAJOR order: [oc][ic][kh][kw]
    static const float conv1_weights_vector[1024] = {
        -2.6962e-01f, 1.9424e-01f, -7.3093e-01f, -3.8578e-01f, -6.6276e-01f, -6.3675e-01f, -4.6286e-01f, -5.4087e-01f, 

        4.0286e-01f, -2.9377e-01f, -1.4041e-01f, -3.0564e-01f, 2.5118e-01f, -5.2732e-01f, -4.5214e-01f, -3.4195e-01f, 

        -2.7573e-01f, -3.1989e-01f, -6.6703e-01f, -4.3685e-01f, -8.0832e-01f, -5.4812e-01f, -4.8675e-01f, -4.1844e-01f, 

        -8.6619e-01f, -2.2474e-01f, -4.7701e-01f, -3.4852e-01f, -3.1326e-01f, -8.7706e-01f, -2.6878e-01f, -5.0108e-01f, 

        -2.9543e-02f, -2.6496e-01f, -6.2344e-01f, -3.7725e-01f, -4.6746e-01f, -2.2147e-01f, -4.1842e-01f, -5.1823e-01f, 

        -2.9831e-01f, -5.4394e-01f, -3.2994e-01f, -2.3041e-01f, -3.2138e-01f, -5.0042e-01f, -3.4397e-01f, -3.1919e-01f, 

        -1.1990e-01f, -4.9500e-01f, -6.4339e-01f, -6.0159e-01f, -6.4245e-01f, -1.8838e-01f, -3.7077e-01f, -3.2994e-01f, 

        -2.9742e-01f, -4.0404e-01f, -2.4566e-01f, -6.1309e-01f, -4.3432e-01f, -5.3610e-01f, -3.2393e-01f, -4.0280e-01f, 

        6.5095e-02f, -1.2634e-01f, -7.0198e-02f, -6.6442e-02f, -8.9157e-01f, -3.6503e-01f, -1.0792e-01f, -2.0365e-01f, 

        1.0483e+00f, -4.3003e-01f, -7.0484e-01f, -7.3499e-01f, -4.4369e-01f, 1.0174e-01f, -2.4081e-01f, -1.7326e-01f, 

        -4.2664e-01f, -5.1793e-02f, -1.7370e-01f, -2.7392e-01f, -2.3899e-01f, -6.4072e-01f, -3.6527e-01f, -5.3160e-01f, 

        -1.1692e+00f, -3.8836e-01f, -5.0316e-01f, -5.2500e-01f, 4.8849e-02f, -2.3765e-01f, -8.1197e-01f, -3.9892e-01f, 

        4.5445e-01f, -3.6930e-01f, -1.8464e-01f, -5.5059e-01f, -8.3998e-02f, -9.8785e-02f, -2.4563e-01f, -4.9593e-01f, 

        -3.9314e-01f, -2.4391e-01f, -3.2315e-01f, -3.3639e-01f, -6.2029e-01f, -8.7202e-01f, -2.2907e-01f, -2.9189e-01f, 

        3.0867e-01f, -7.7164e-01f, -7.3386e-01f, -6.2223e-01f, -3.3729e-01f, -2.7994e-01f, -1.7392e-01f, -6.9800e-01f, 

        -3.2386e-01f, -7.3533e-02f, -2.2050e-01f, -3.2098e-01f, -1.1297e+00f, -7.3804e-01f, -4.0826e-01f, -5.5046e-01f, 

        9.2322e-02f, -4.0618e-01f, -2.7821e-01f, -5.7582e-01f, -6.1971e-01f, -3.1613e-01f, -5.9673e-01f, -8.5599e-01f, 

        -7.5097e-01f, -1.1155e+00f, -3.1361e-01f, -7.9917e-01f, -4.0246e-01f, -6.6547e-01f, -1.8515e-01f, -9.0195e-01f, 

        -4.4005e-01f, -2.3657e-01f, -3.1573e-01f, -4.3218e-01f, -1.8634e-01f, -7.3194e-01f, -4.6249e-01f, -8.5346e-01f, 

        -5.3631e-01f, -8.1684e-01f, -2.0579e-01f, -7.6926e-01f, -3.6022e-01f, -2.1207e-01f, -2.4863e-01f, -8.6577e-01f, 

        3.2664e-01f, -4.7564e-01f, -1.6691e+00f, -1.6701e+00f, 2.9248e-01f, 1.4950e-01f, -2.3127e+00f, -1.7918e-01f, 

        5.7771e-02f, -8.0961e-02f, -1.2043e+00f, -7.5725e-01f, 6.1477e-01f, -5.8415e-01f, -2.6308e+00f, 3.6084e-01f, 

        -3.3257e-01f, -5.4210e-01f, -9.6451e-01f, -9.9144e-01f, 1.9929e-01f, 3.4106e-01f, -1.2060e+00f, -5.8568e-01f, 

        9.7488e-02f, -1.7503e-01f, -9.0743e-01f, -1.2247e-01f, 3.8743e-01f, -4.7731e-01f, -1.8319e+00f, 2.1521e-01f, 

        4.3856e-01f, -6.6680e-02f, -3.6487e-01f, -8.0033e-01f, -2.4631e-01f, -5.9877e-01f, -7.0108e-01f, -5.0233e-01f, 

        -4.4623e-01f, -7.1950e-01f, -1.3028e-01f, 2.8795e-02f, -4.2944e-02f, -6.2922e-01f, -6.4448e-01f, -7.3107e-01f, 

        -8.7335e-01f, -1.0353e-01f, 7.7656e-02f, -5.9133e-01f, -8.5194e-01f, -4.8143e-01f, -8.2522e-01f, -2.5189e-01f, 

        -5.4215e-01f, -5.4634e-01f, -5.6353e-02f, -2.1793e-01f, -9.5538e-01f, -2.4005e-01f, -4.4774e-01f, -4.3499e-01f, 

        -6.8312e-01f, 2.5635e-01f, -4.4654e-01f, -2.3710e-01f, -5.5069e-01f, -1.9424e-01f, -3.1646e-01f, -1.3829e-01f, 

        2.8000e-01f, 1.7351e-01f, 3.2163e-02f, 1.8354e-01f, -1.1481e-01f, -4.0764e-01f, -1.0118e+00f, -2.7549e-01f, 

        -3.8859e-01f, -4.7208e-01f, -5.4893e-01f, -5.9561e-01f, -6.6760e-01f, -1.9539e-01f, -2.4344e-01f, -3.7292e-01f, 

        -2.8621e-01f, 3.4475e-03f, -1.3655e-01f, -3.4189e-01f, -1.8442e-01f, -3.3143e-01f, -1.2021e+00f, -3.1543e-01f, 

        7.3311e-02f, -2.3433e-01f, -2.0826e-01f, 1.9343e-01f, -4.5897e-01f, -6.1754e-01f, -3.2594e-01f, -6.6892e-01f, 

        -3.2688e-01f, -1.0714e+00f, -3.4949e-01f, -3.3906e-02f, -7.2511e-01f, -6.2668e-01f, -1.2325e+00f, 4.3940e-01f, 

        -2.6999e-01f, -1.5956e-01f, 8.6205e-02f, -3.0274e-01f, -5.0372e-01f, -4.5641e-01f, -3.5447e-01f, -4.8646e-01f, 

        -7.3556e-01f, -6.6230e-01f, -2.9141e-01f, 2.9313e-01f, -8.2405e-01f, -4.0751e-01f, -1.5494e+00f, 1.1614e+00f, 

        2.4805e-01f, -5.9053e-01f, -6.7190e-01f, -5.9474e-01f, -5.5999e-01f, -8.4715e-01f, -4.5213e-01f, -7.0798e-01f, 

        -4.0939e-01f, -6.5281e-01f, -4.0320e-01f, -4.7334e-01f, -1.3866e-01f, -5.0004e-01f, -4.7003e-01f, -6.3239e-01f, 

        -8.9938e-01f, -2.4866e-01f, -6.1534e-01f, -5.0253e-01f, -4.9006e-01f, -4.8315e-01f, -3.8385e-01f, -3.8156e-01f, 

        -8.7991e-01f, -4.3832e-01f, -2.0922e-01f, -2.9926e-01f, -2.6097e-01f, -1.4562e-01f, -4.7250e-01f, -4.3580e-01f, 

        -1.1728e-01f, 3.0446e-01f, 1.1781e+00f, 1.4779e+00f, 4.5772e-01f, 1.1073e+00f, 1.2246e+00f, 4.0484e+00f, 

        8.8613e-01f, 3.9994e-01f, 1.3139e+00f, 9.5509e-01f, 1.4357e+00f, 3.5033e+00f, 3.0078e+00f, 2.5745e+00f, 

        2.0370e-02f, -2.7864e-01f, 3.6751e-01f, 5.2910e-01f, 8.5265e-01f, 5.5555e-01f, 1.8329e-01f, 3.4050e+00f, 

        2.0074e-02f, -3.9781e-01f, 7.7844e-01f, 1.0999e-01f, 1.3701e+00f, 2.6066e+00f, 2.1153e+00f, 2.2566e+00f, 

        -1.4040e-02f, -4.6894e-01f, -6.7529e-01f, -4.8105e-01f, -3.8355e-01f, -6.2584e-01f, -9.3819e-01f, -6.1562e-01f, 

        -9.0515e-01f, -9.5265e-01f, -1.0091e+00f, -5.8551e-01f, -5.3155e-01f, -7.2958e-01f, -4.2293e-01f, -4.6593e-01f, 

        -6.4608e-01f, -5.1480e-01f, -1.7865e-01f, -3.6942e-01f, -7.6761e-01f, -5.2818e-01f, -9.0906e-01f, -7.8277e-01f, 

        -3.4180e-01f, -5.9393e-01f, -4.6244e-01f, -8.5435e-01f, -1.0312e-01f, -8.1757e-01f, -2.3772e-01f, -1.3338e-01f, 

        9.3220e-02f, 4.7273e-01f, -1.2265e-01f, -1.0568e-01f, 2.4723e-01f, 8.2990e-01f, -2.9918e-01f, -2.2893e-01f, 

        -3.4952e-02f, 1.0426e-02f, 9.3122e-02f, -2.4203e-01f, -2.6234e-01f, 1.9675e-02f, -3.1958e-01f, -1.4742e+00f, 

        -9.5233e-02f, -6.9497e-02f, 2.6316e-01f, -4.4820e-01f, -7.4084e-02f, 1.0708e+00f, 3.4936e-01f, -2.5916e-01f, 

        -1.1623e-01f, 3.1174e-01f, 9.9904e-02f, -4.4079e-01f, -2.7360e-01f, -1.4035e-01f, -7.5760e-02f, -1.4703e+00f, 

        -2.0556e+00f, -4.9947e-01f, 2.7196e-01f, -2.0918e+00f, -1.2623e+00f, -1.4435e+00f, -1.5441e+00f, -5.2366e-01f, 

        -4.9928e-01f, -7.3788e-01f, -8.6035e-01f, -7.4318e-01f, -1.2316e+00f, -7.9432e-01f, -4.6612e-01f, -1.0710e+00f, 

        -8.9316e-01f, -3.5271e-01f, 5.9316e-01f, -1.3114e+00f, -1.2177e+00f, -1.2579e+00f, -1.0192e+00f, -5.7955e-01f, 

        -1.0325e+00f, -5.2185e-01f, -4.6104e-01f, -4.0064e-01f, -8.4392e-01f, -8.8831e-01f, -4.9508e-01f, -3.1038e-01f, 

        -3.9021e-01f, -9.1513e-01f, -6.6455e-01f, -9.5401e-01f, 1.2824e-01f, -9.6816e-02f, -2.6824e-01f, 8.2100e-01f, 

        -8.7721e-01f, -3.2203e-01f, -1.4527e+00f, -1.1026e+00f, -3.4637e-01f, -1.0809e+00f, -1.0104e+00f, -2.2769e+00f, 

        -4.8747e-01f, -5.0672e-01f, -1.0054e+00f, -1.2534e+00f, -2.4578e-01f, -1.2562e-01f, -4.1980e-01f, 4.5585e-01f, 

        -3.6350e-01f, -7.4755e-01f, -1.3626e+00f, -5.6206e-01f, -1.1441e+00f, -1.4978e+00f, -3.4402e-01f, -9.8317e-01f, 

        -1.5793e-01f, -4.6438e-01f, 4.2353e-02f, -1.7325e-01f, -7.9575e-01f, 7.8436e-02f, -4.8571e-02f, -1.5655e-01f, 

        3.3542e-02f, 2.5292e-01f, 2.4447e-01f, -2.8799e-01f, 3.6993e-01f, -7.0170e-01f, -4.6095e-01f, -1.9687e-01f, 

        -1.5249e-01f, 8.5075e-02f, -3.1382e-01f, -4.0325e-01f, -6.5459e-01f, -6.4032e-01f, -4.5854e-01f, -2.8619e-01f, 

        -1.4382e-03f, -2.8587e-01f, -3.5379e-01f, -7.4398e-01f, -5.0314e-02f, -3.4434e-01f, -2.4172e-01f, -4.0170e-01f, 

        -2.5587e-01f, -1.6552e-01f, -1.8585e-01f, 7.1690e-01f, -4.0457e-01f, -4.8531e-01f, -7.9021e-01f, -6.3958e-01f, 

        -6.5207e-01f, -7.2497e-01f, -1.0908e-01f, -8.9301e-01f, -6.7085e-01f, -7.1741e-01f, -1.5774e+00f, -3.7126e-02f, 

        -1.0302e-01f, -4.7348e-02f, 1.3204e-01f, 6.5191e-01f, -4.5822e-02f, 6.6193e-02f, -5.6438e-01f, -4.0885e-01f, 

        -3.0436e-01f, -8.2755e-01f, -2.4172e-01f, -5.5423e-01f, -6.7384e-01f, -1.0434e+00f, -1.9904e+00f, -4.8006e-01f, 

        3.0112e-01f, 2.3686e-01f, 1.0695e-01f, -2.6776e-01f, 3.4794e-01f, -6.5644e-02f, 6.0222e-01f, -1.3638e-01f, 

        1.8748e-01f, -2.8935e-01f, -1.5645e-01f, -3.3251e-01f, 4.3718e-03f, -8.1635e-01f, -2.0967e-01f, 1.0530e+00f, 

        -1.5811e-01f, -2.3801e-01f, -9.6336e-02f, -1.5943e+00f, -7.7034e-01f, -1.0770e+00f, -1.5218e+00f, -1.1122e+00f, 

        -6.4673e-01f, -7.9311e-01f, -8.4727e-01f, -2.6271e-01f, -8.4777e-01f, -6.1848e-01f, 7.7267e-02f, -2.9641e-01f, 

        -4.8282e-01f, -4.8305e-01f, -1.2292e+00f, -6.1154e-01f, -7.3290e-01f, -1.9042e+00f, -5.8683e-01f, -3.0485e-01f, 

        -8.8464e-01f, -2.3371e-01f, -8.5545e-01f, -4.5755e-01f, -9.3422e-01f, 8.9153e-01f, -9.4604e-01f, -4.9390e-01f, 

        -9.9495e-01f, -6.3376e-01f, -1.1484e+00f, -8.8932e-01f, -1.1221e+00f, -2.3921e+00f, -3.2071e-01f, -5.8184e-01f, 

        -1.0896e+00f, 5.4472e-01f, -8.1077e-01f, -2.8728e-01f, -1.1448e+00f, -1.9113e-01f, -8.9327e-01f, -2.8037e-01f, 

        3.9469e-02f, 4.2475e-01f, 2.9489e-01f, -3.4449e-01f, -6.4043e-01f, -2.5704e-01f, -3.1492e-01f, -5.0680e-01f, 

        -4.4588e-01f, -8.0775e-01f, -3.0642e-01f, 5.1847e-03f, -3.8009e-01f, -3.0439e-01f, -4.7805e-01f, -5.4610e-01f, 

        -3.6664e-01f, -5.8723e-01f, -3.5792e-01f, 1.1354e-02f, -1.0312e+00f, -3.5714e-01f, -5.5040e-01f, -5.2895e-01f, 

        -6.5485e-01f, -1.7728e-01f, -2.3720e-01f, -4.6061e-01f, -7.8349e-01f, -5.6260e-01f, -4.3781e-01f, -3.2386e-01f, 

        -1.7147e+00f, -1.2768e+00f, -4.6138e-01f, -8.0696e-02f, -1.8544e+00f, -7.6332e-01f, -4.4071e-01f, 1.1415e-02f, 

        -7.5158e-01f, -7.6542e-01f, -6.6001e-01f, -7.0885e-01f, 1.0916e-01f, 5.5679e-01f, -8.1216e-01f, -3.3404e-01f, 

        -1.0865e+00f, -9.2733e-01f, -1.0002e+00f, 5.1869e-01f, -1.4678e+00f, -7.2813e-01f, -9.8211e-01f, 1.8105e-01f, 

        -7.3678e-01f, -4.5248e-01f, -7.7284e-01f, -9.4532e-01f, -2.6497e-01f, -3.8817e-01f, -6.3162e-01f, -2.0452e-01f, 

        1.7568e-01f, -1.2859e-01f, -4.2024e-01f, -1.6305e-01f, -3.2773e-01f, -3.0573e-01f, -4.1187e-01f, -3.6491e-01f, 

        -1.6635e-02f, -2.4600e-01f, -4.2737e-01f, -2.5035e-01f, -2.9928e-01f, -7.1664e-02f, -5.7299e-01f, -4.8403e-01f, 

        -6.6124e-01f, -2.1607e-01f, -1.8228e-01f, -1.5742e-01f, -2.5173e-01f, -7.8042e-02f, -2.5004e-01f, -2.0592e-01f, 

        -1.2135e-01f, -1.3756e-01f, -4.5498e-01f, -1.5198e-01f, -9.1557e-02f, -4.0060e-01f, -5.4297e-01f, -6.0404e-01f, 

        -3.4525e-01f, -1.4362e+00f, -9.3249e-01f, -6.0076e-01f, -1.5972e+00f, 1.3541e+00f, -1.4468e+00f, -3.6231e-01f, 

        -1.9654e+00f, -1.2163e+00f, 9.5418e-02f, -9.0553e-01f, -2.7529e-01f, -1.6271e+00f, 1.8125e-02f, -4.7143e-01f, 

        -1.0197e+00f, -1.6868e-01f, 7.6239e-03f, -8.8801e-01f, -1.2193e+00f, 1.2664e+00f, -7.3823e-01f, -3.2359e-01f, 

        -1.7279e+00f, -1.8555e+00f, -2.2420e-01f, -7.0768e-01f, -6.8557e-01f, -4.9297e-01f, -3.7088e-01f, 3.8574e-01f, 

        -1.7447e+00f, -1.0476e+00f, -1.0823e+00f, -4.7891e-02f, 1.5406e+00f, 3.4144e-01f, -2.1669e-01f, 3.8564e-02f, 

        -2.5613e-01f, -2.5174e-01f, 6.9503e-01f, -4.7325e-01f, -3.0066e+00f, -1.8337e-01f, -1.0575e+00f, -1.4546e+00f, 

        -1.1331e+00f, -1.8566e+00f, -5.4524e-01f, -3.3554e-01f, -2.0813e-01f, -3.3584e-01f, -6.7963e-01f, -1.4213e-01f, 

        -1.1010e+00f, -1.3537e-01f, -9.2513e-01f, -3.9135e-01f, -2.0974e+00f, -2.0712e-01f, -8.2809e-01f, -1.6777e+00f, 

        -2.0393e-01f, -4.1667e-01f, -2.6980e-01f, -1.6572e-01f, -1.1327e+00f, -6.8432e-01f, -1.3531e+00f, 1.1497e+00f, 

        -7.4745e-01f, -4.9137e-01f, 2.3134e-01f, 4.5582e-01f, -4.5748e-02f, -3.6350e-01f, -5.0238e-01f, 2.1281e-01f, 

        -1.1255e-01f, -2.7523e-01f, -4.1000e-01f, 1.7595e-02f, -2.4044e-01f, -4.9242e-01f, -8.2604e-01f, 6.5641e-01f, 

        -2.3270e-01f, 2.2170e-02f, 1.3960e-02f, 1.4271e-03f, 2.9722e-02f, 4.4744e-01f, -7.1518e-01f, 1.4856e-01f, 

        -1.8447e-01f, -1.0833e-01f, -5.8154e-01f, -2.5665e-01f, -4.7569e-01f, -5.5721e-01f, -4.3148e-01f, -3.3675e-01f, 

        -2.4945e-01f, -6.2196e-01f, -4.6343e-01f, -4.6564e-01f, -2.9988e-01f, -4.0438e-01f, -2.2871e-01f, -4.6712e-01f, 

        -5.0457e-02f, -1.1921e-01f, -4.8665e-01f, -1.9727e-01f, -2.4842e-01f, -2.4577e-01f, -5.7091e-01f, -5.8286e-01f, 

        -5.0192e-01f, -4.1067e-01f, -2.6335e-01f, -6.2162e-01f, -1.8420e-01f, -4.9322e-01f, -1.9154e-01f, -4.4455e-01f, 

        -1.2461e+00f, -8.5502e-01f, -7.1625e-01f, 7.3833e-01f, -7.1046e-01f, -3.6096e-01f, -5.5780e-01f, -1.9686e-01f, 

        3.8879e-01f, 2.7440e-01f, -2.8016e-01f, -2.4096e-01f, -1.3933e+00f, -3.7341e-01f, -3.0445e-01f, -8.8023e-01f, 

        -8.6850e-01f, -3.4780e-01f, -7.7802e-01f, -3.9956e-01f, 2.5050e-01f, -4.4366e-01f, 1.8556e-01f, -4.9818e-01f, 

        8.7285e-01f, 4.3540e-01f, -6.1335e-02f, 2.7310e-01f, -1.6041e+00f, -8.3035e-01f, 9.9761e-02f, 3.4733e-02f, 

        -7.4642e-01f, 1.0738e+00f, 3.0086e-01f, -3.1465e-01f, -9.8386e-02f, -5.4753e-01f, -5.4790e-01f, -4.6039e-01f, 

        -5.0342e-01f, -4.6888e-01f, -7.0095e-02f, -4.8028e-01f, 8.4159e-02f, -5.2050e-01f, 2.1881e-01f, 3.7153e-02f, 

        -5.0740e-01f, -5.1534e-01f, -2.5996e-01f, -2.2730e-01f, -1.6307e-01f, -6.0059e-02f, -3.9076e-01f, -6.2073e-01f, 

        -8.4828e-01f, -4.1588e-01f, -1.3700e-01f, -3.9951e-01f, -8.5847e-01f, -3.6527e-01f, -2.0619e-01f, -1.6784e-01f, 

        -5.6725e-01f, -5.0246e-01f, -4.5789e-01f, -4.7713e-01f, -2.8183e-01f, -4.1247e-01f, -1.5393e-01f, -3.9767e-01f, 

        -3.0571e-01f, -4.1303e-01f, -7.8092e-01f, -4.9101e-01f, -6.0876e-01f, -8.2298e-01f, -6.8597e-01f, -7.7323e-02f, 

        -2.7786e-01f, -2.2000e-01f, -1.8376e-01f, -4.2640e-01f, -4.9738e-01f, -4.0767e-01f, -4.1889e-01f, -4.3065e-01f, 

        -6.9457e-01f, -3.0081e-01f, -7.3358e-01f, -4.3512e-01f, -2.7352e-01f, -6.8950e-01f, -5.3245e-01f, -1.5894e-01f, 

        -5.7182e-01f, -2.0807e-01f, 6.8906e-01f, 2.2784e-01f, -3.0663e-01f, -4.7979e-01f, -9.1861e-02f, -4.5714e-01f, 

        -2.0121e-01f, -2.8036e-02f, -2.3033e-01f, 7.9015e-02f, 2.2714e-01f, -2.0311e-01f, -9.2081e-01f, -6.6375e-01f, 

        -1.0143e+00f, -1.3613e-01f, -6.4980e-01f, -4.8359e-01f, -5.5908e-01f, -6.0619e-02f, -2.4941e-01f, -2.7604e-01f, 

        -1.6220e-01f, -1.2968e-01f, -3.7500e-01f, -2.5425e-01f, -3.7351e-01f, -3.3241e-01f, -4.6703e-01f, -3.1874e-01f, 

        -1.5382e+00f, -9.5234e-01f, -1.9052e+00f, -7.7771e-01f, -3.0764e-01f, -1.8262e-01f, -1.2617e+00f, -1.4331e-01f, 

        -2.1227e-01f, -9.6925e-01f, -4.2235e-03f, 5.1400e-01f, -1.5937e-02f, -2.8149e-02f, 8.2247e-01f, 5.3481e-01f, 

        -1.0293e-01f, -6.1626e-01f, -7.0222e-01f, -4.6791e-01f, -1.0547e-01f, -7.0509e-01f, -1.0158e+00f, 3.2512e-01f, 

        -1.6031e-01f, -6.0989e-01f, -8.5701e-02f, 5.7744e-02f, -1.2885e-01f, 7.9243e-02f, 3.7411e-01f, 4.8159e-01f, 

        -3.7412e-01f, -4.4645e-01f, -3.5408e-01f, -3.9953e-01f, -4.1121e-01f, -4.3390e-01f, -4.2380e-01f, -6.7750e-01f, 

        -3.7411e-01f, -4.2173e-01f, -3.9482e-01f, -4.1101e-01f, -5.5382e-01f, -4.8684e-01f, -2.2415e-01f, -2.9871e-01f, 

        -6.3468e-01f, -3.4548e-01f, -5.5844e-01f, -5.6853e-01f, -5.2742e-01f, -6.2054e-01f, -4.8663e-01f, -4.3120e-01f, 

        -6.4383e-01f, -3.0314e-01f, -1.7973e-01f, -2.9384e-01f, -6.5730e-01f, -4.4929e-01f, -3.2321e-01f, -2.9063e-01f
    };

    // Layer 2 conv weights - Conv2d (64x32x3x3)
    // Stored in OUT_CH-MAJOR order: [oc][ic][kh][kw]
    __attribute__((section(".ext_weights")))
    static const float conv2_weights_vector[18432] = {
        4.7389e-01f, 2.7680e-01f, 1.5274e-01f, 1.6172e-02f, 1.6550e-01f, -2.7878e-01f, 1.2566e-01f, -1.9234e-01f, 

        -2.6982e-01f, -1.1794e-01f, -3.5462e-02f, -1.7714e-02f, -9.7435e-02f, -1.7030e-01f, -1.3922e-01f, -1.4143e-01f, 

        -1.7052e-01f, -8.6058e-02f, -3.3144e-01f, 9.2034e-01f, -1.0589e-01f, 6.7450e-01f, -2.1280e-01f, -4.8281e-01f, 

        3.2636e-02f, -9.8499e-02f, -2.4096e-01f, 2.0072e-01f, -5.2217e-02f, 6.5675e-01f, -9.7950e-01f, -6.4813e-01f, 

        -6.5437e-01f, -1.5573e+00f, -1.1190e+00f, 1.9811e-01f, 4.7608e-01f, 1.9196e-01f, 2.6215e-01f, -9.9263e-02f, 

        -1.7581e-01f, -9.1125e-02f, -2.4598e-01f, -2.1392e-02f, -9.2144e-03f, -1.8594e-01f, -3.6961e-02f, 9.6616e-01f, 

        8.5528e-02f, -2.4032e-01f, -1.3066e+00f, -5.5871e-02f, -3.6979e-01f, -8.5577e-01f, 1.1659e+00f, 6.2977e-01f, 

        2.3786e-01f, 1.8598e-01f, -3.3408e-01f, -1.8939e-01f, -5.0596e-01f, 8.5679e-02f, -2.5553e-01f, 1.0873e+00f, 

        2.3909e+00f, 1.0241e-01f, 5.7111e-01f, 1.4105e+00f, -7.7705e-01f, -2.3065e-01f, -5.3395e-01f, -4.3281e-01f, 

        -7.9110e-01f, -1.1342e+00f, -3.0443e-01f, 1.1915e-01f, 6.6350e-01f, 3.8789e-01f, 4.2895e-01f, 2.3038e-01f, 

        6.6148e-01f, 5.6686e-01f, 3.1318e-01f, -6.1604e-02f, 3.1448e-02f, 2.2182e-02f, -1.4440e-01f, -3.1012e-01f, 

        -1.0151e-01f, -3.1175e-02f, 1.1057e-01f, -4.3347e-01f, -8.9390e-01f, -5.2767e-02f, -4.5447e-02f, 1.0944e-01f, 

        1.6259e-01f, 3.3561e-01f, 2.3211e-01f, -1.0995e-01f, -1.5836e-01f, -1.8907e-01f, -9.2724e-02f, -5.0864e-02f, 

        -6.8527e-02f, -6.2752e-03f, -1.2768e-01f, -4.8393e-02f, 2.1225e-01f, 2.3482e-01f, -5.2609e-01f, -2.7739e-01f, 

        -1.0877e+00f, -8.1983e-02f, 2.5943e-01f, -1.4699e-01f, -3.7830e-01f, -5.1364e-01f, -3.2519e-01f, -3.3259e-01f, 

        -3.0603e-01f, -1.8819e-01f, -2.4873e-01f, 1.4991e-01f, -7.7439e-02f, -2.2061e-01f, 9.1867e-02f, 1.6960e-01f, 

        8.1542e-01f, 3.1150e-02f, -2.1180e-01f, -3.1687e-01f, -5.5361e-01f, -2.7061e-01f, -1.1015e+00f, -3.9238e-02f, 

        -8.3954e-01f, -9.3815e-01f, 5.8596e-01f, -7.0308e-01f, -7.1493e-01f, 1.4877e-01f, -1.7113e+00f, 5.2767e-01f, 

        -5.3288e-01f, -4.8909e-01f, 3.0974e-01f, -2.8129e-01f, -7.4435e-01f, -4.2426e-02f, -2.7839e-01f, -9.7644e-01f, 

        -2.2115e-01f, 7.3937e-01f, 5.1732e-02f, 4.6329e-01f, 2.4874e-01f, 6.6742e-01f, -3.8673e-01f, 1.0652e+00f, 

        9.2164e-01f, -5.4864e-01f, -7.5068e-01f, 3.2924e-01f, -1.3845e-02f, 1.0273e-01f, -1.0437e+00f, -2.8246e-01f, 

        -9.2226e-01f, -3.7219e-01f, -3.0727e-01f, 3.8898e-01f, 2.0382e-01f, 1.3669e+00f, 7.4894e-02f, -2.0423e-01f, 

        -4.3182e-01f, -4.3409e-01f, 5.2047e-01f, -1.8986e-01f, -1.4099e-02f, -5.3019e-01f, -2.8016e-01f, -5.0187e-01f, 

        -9.8330e-01f, -1.1445e-01f, 2.5080e-01f, -2.8072e-01f, 6.7131e-02f, 5.4094e-01f, -1.8144e-02f, -4.8375e-03f, 

        -1.5723e-02f, -3.7197e-02f, -5.0214e-02f, -3.8958e-01f, -2.2019e-01f, -8.7254e-02f, 2.2500e-01f, 5.5399e-01f, 

        1.9538e-01f, 6.9383e-01f, 9.1547e-02f, -4.2439e-02f, -2.3453e-01f, 7.3209e-01f, -4.2457e-01f, -2.2747e-01f, 

        -8.4231e-01f, 1.5556e+00f, 1.8222e-01f, -1.2790e+00f, 6.3545e-02f, -4.2702e-01f, 2.8729e-01f, 1.2075e-02f, 

        -2.5435e-01f, -8.3545e-02f, -1.4560e-01f, 3.8616e-01f, -7.3854e-01f, -3.9914e-01f, -4.7681e-01f, -4.5431e-01f, 

        -3.0743e-01f, -5.5302e-02f, -1.1936e-01f, -1.5581e-01f, -1.0509e-01f, -9.9776e-02f, -9.4014e-02f, -3.1057e-02f, 

        -9.7989e-02f, -6.4592e-03f, -2.9822e-01f, -3.2887e-01f, -1.2535e+00f, -2.0179e-01f, -3.5570e-01f, -2.9595e-01f, 

        -9.6286e-01f, -8.8175e-01f, -1.4509e+00f, 6.0552e-02f, -1.0225e+00f, 1.2762e+00f, -4.6801e-01f, 1.6551e-01f, 

        -4.9578e-01f, -1.0877e-01f, 9.6932e-02f, 1.8270e-01f, -5.0456e-03f, -4.7152e-02f, -1.0230e-01f, -7.8630e-02f, 

        -1.3191e-01f, -9.8490e-02f, -1.3517e-01f, -1.1936e-01f, -1.8221e-01f, 6.0601e-01f, 2.6117e+00f, 8.9273e-01f, 

        1.5072e+00f, -6.4593e-01f, -3.0581e-01f, -8.4056e-01f, 1.7592e-01f, -4.0628e-01f, -7.9627e-01f, -9.5937e-01f, 

        -1.3555e+00f, -1.9481e+00f, -5.0492e-01f, 3.0208e-02f, -6.7820e-01f, -7.9129e-01f, -1.7829e-01f, -1.0790e-01f, 

        -1.5440e-01f, -9.6949e-02f, -5.4151e-02f, -1.8162e-01f, -1.4258e-01f, -1.4021e-01f, -1.4526e-01f, -2.9425e-02f, 

        6.4366e-02f, -1.4686e-01f, -6.8905e-02f, 2.9765e-02f, -1.1835e-01f, -1.6258e-01f, -1.7332e-01f, -8.1870e-02f, 

        -1.3594e-01f, -2.1303e-02f, -3.9176e-02f, -1.5828e-01f, -7.9943e-02f, -1.3468e-01f, -1.2062e-01f, 4.3030e-02f, 

        -1.5092e-01f, -6.8533e-02f, -4.4246e-01f, -8.7598e-01f, 1.1188e-02f, -1.3193e-01f, -1.0771e-01f, -1.7333e-01f, 

        -4.8082e-01f, 1.0624e-01f, 1.1741e-01f, 3.6718e-01f, -2.1488e+00f, -1.0277e+00f, 5.1219e-01f, -1.1186e-01f, 

        -9.4060e-02f, -2.5554e-02f, -3.8672e-01f, 5.8532e-02f, 1.1189e+00f, -5.3958e-02f, -1.0427e-01f, -7.4424e-02f, 

        -5.9164e-02f, -1.1670e-01f, 3.4223e-02f, 2.2892e-02f, -1.1047e-01f, 2.0942e-02f, 5.4411e-02f, -3.2044e-01f, 

        3.1809e-01f, -7.4313e-02f, -2.4864e-01f, -3.6043e-01f, -2.2587e-01f, 1.6179e-01f, 1.0537e+00f, -6.4103e-01f, 

        -6.4990e-02f, 1.8226e+00f, -8.9756e-02f, -5.2192e-02f, -1.6068e-01f, -2.5654e-01f, -1.3710e-01f, 2.7928e-02f, 

        -1.1224e-01f, -2.7878e-01f, 4.8560e-02f, -5.0716e-02f, 2.1545e-01f, -2.7710e-01f, 9.9157e-02f, -1.6262e-01f, 

        6.1825e-01f, -7.0392e-02f, -2.0832e-01f, 4.6118e-02f, -6.5154e-02f, -3.6613e-01f, -1.3774e-01f, -2.1836e-01f, 

        -1.6376e-01f, 2.0144e+00f, -1.4178e-01f, -9.4401e-02f, 1.1210e+00f, -4.2564e-02f, -1.5218e-01f, -3.8808e-02f, 

        -3.8621e-02f, -5.9538e-02f, -1.4286e+00f, 9.8150e-02f, -1.4617e+00f, -3.2807e+00f, -8.0027e-02f, -2.2006e+00f, 

        -8.1702e-01f, 4.2677e-01f, -1.0930e+00f, -1.0094e-01f, 1.7877e-01f, -1.4125e-01f, 3.3559e-01f, -1.1281e-01f, 

        -2.0342e-01f, -8.9728e-02f, -7.4502e-02f, -1.0509e-01f, -6.3481e-01f, -3.7399e-01f, 9.8766e-02f, -1.1454e+00f, 

        -8.1638e-02f, -1.4498e-01f, -3.3484e-01f, 9.6700e-02f, -2.7896e-02f, -1.1751e+00f, -5.9689e-01f, -2.5788e-01f, 

        1.7483e-01f, -4.6203e-02f, -1.0127e-01f, 4.0763e-02f, -7.1883e-02f, -3.6009e-02f, -2.1449e-02f, -3.1426e-02f, 

        -1.6822e-01f, -1.1733e-01f, -1.7892e-01f, -2.2225e-01f, -1.6839e-01f, -1.4210e-01f, -2.1958e-01f, -1.0557e-01f, 

        -2.5003e-01f, -5.8061e-02f, 8.1132e-02f, -4.9428e-02f, -1.8497e-02f, -1.7954e-01f, -1.4135e-01f, -2.2057e-01f, 

        1.1524e-01f, 6.8861e-02f, 6.5009e-01f, -3.2024e-01f, -2.0110e-01f, -2.0041e-01f, -6.4474e-03f, -1.2149e-01f, 

        -1.9725e-01f, -2.9487e-01f, -8.5190e-02f, -5.4475e-02f, 7.1734e-02f, 6.7687e-02f, -5.2951e-02f, -3.7917e-02f, 

        -5.3445e-02f, -7.8717e-04f, -5.5370e-01f, 2.0411e-01f, -2.9642e-01f, 2.3842e-01f, -1.0952e-01f, 2.0898e-02f, 

        1.3122e-01f, 4.0252e-01f, -3.1432e-01f, 5.2424e-01f, -2.1018e-01f, 2.3791e-01f, 7.1437e-01f, 2.1023e-01f, 

        -1.3524e-01f, -9.8968e-02f, -2.0621e-01f, -2.1171e-01f, 5.1352e-03f, -4.2919e-02f, 3.4510e-01f, -1.7347e-01f, 

        -2.4455e-01f, -1.0135e-01f, 9.3053e-02f, -2.6693e-01f, -1.8826e-01f, 4.3926e-01f, -1.4989e-01f, -7.5829e-02f, 

        2.1101e-01f, -9.6506e-02f, -1.0021e-01f, -6.0134e-02f, -5.3000e-02f, -2.0145e-02f, -4.6796e-01f, 2.3439e-01f, 

        2.3263e-03f, -1.4946e+00f, 2.5803e-02f, -3.5273e-01f, -8.9116e-02f, -1.2855e-01f, -2.2251e-01f, -7.6470e-01f, 

        -1.1473e+00f, -1.6677e-01f, 5.6490e-02f, -8.7044e-01f, -2.3512e-01f, -1.2096e-01f, -7.2762e-02f, -2.4396e-01f, 

        -9.1884e-02f, 2.3509e-02f, 7.0976e-02f, -3.8087e-02f, -2.8258e-01f, -7.4934e-02f, -1.0863e-01f, -6.1365e-01f, 

        -2.5900e-01f, -1.2760e-01f, -1.2933e-01f, -8.1284e-02f, -1.1861e-01f, -6.1966e-02f, -1.0191e-01f, 5.5497e-03f, 

        -6.6388e-02f, -1.1917e-01f, -5.8805e-01f, -1.1519e+00f, -1.0397e+00f, -4.9034e-01f, -2.0837e-03f, -3.2157e-01f, 

        -4.2019e-01f, -4.3114e-02f, -2.4192e-01f, -2.5116e-01f, -1.9370e-01f, 7.3610e-02f, -2.2478e-01f, -1.3435e-01f, 

        -7.5167e-02f, -1.2188e-01f, -1.6315e-02f, -1.3142e-01f, -5.0971e-02f, -1.3720e-01f, -7.9622e-02f, -4.1097e-02f, 

        -7.2985e-02f, -1.5407e-02f, -3.4353e-02f, -6.5557e-02f, -1.2789e-01f, 3.3734e-01f, 9.9420e-01f, -1.6850e-01f, 

        1.7980e-01f, 9.8035e-02f, 7.9551e-02f, -9.3778e-02f, -1.7591e-01f, -1.2699e-01f, -7.3686e-02f, -1.3417e-02f, 

        -8.5086e-02f, -1.2455e-01f, -7.7136e-02f, -2.7413e-01f, -1.4134e-01f, -5.5782e-01f, -4.7498e-01f, -8.9526e-02f, 

        -3.0265e-02f, -7.9903e-02f, -2.0890e-01f, -1.6162e-01f, -2.3921e-01f, -3.2407e-01f, -2.3612e-01f, 4.2765e-02f, 

        -1.1992e-01f, 3.4369e-01f, -5.1493e-03f, -2.4392e-01f, -4.0628e-01f, 2.3130e-01f, -4.2590e-01f, -2.9515e-01f, 

        -9.3234e-02f, -2.3268e-02f, -1.9479e-01f, -5.0570e-02f, -6.7658e-02f, -3.5362e-02f, -5.7530e-02f, 1.2123e-01f, 

        -1.4974e-01f, -9.2572e-02f, 1.6048e-01f, 1.8271e+00f, 5.3286e-01f, -2.7058e-01f, 7.5192e-01f, -4.8634e-01f, 

        -5.9995e-01f, -1.5606e+00f, 1.1098e+00f, -6.5942e-02f, 2.7585e-01f, -9.0574e-02f, -1.7491e-01f, -3.9097e-01f, 

        5.8769e-01f, 4.3017e-01f, -3.3466e-01f, -5.2545e-01f, -6.2271e-01f, 4.8386e-02f, 1.3716e-01f, 1.0990e-01f, 

        -4.9404e-01f, -1.4125e+00f, 6.2691e-01f, -1.3981e-01f, -9.3945e-01f, -6.5306e-01f, 1.5309e+00f, 3.3279e-02f, 

        -2.6058e-01f, -6.5418e-01f, -1.2096e-01f, -8.7870e-01f, 3.1351e-01f, -3.4364e-01f, 6.8508e-01f, 2.9179e-01f, 

        -7.9783e-01f, 4.7593e-01f, -8.8241e-01f, 1.1237e-02f, 4.4757e-01f, -6.5300e-01f, -3.0855e-01f, -5.7198e-01f, 

        -2.9589e-02f, -3.1658e-01f, -4.5752e-01f, -9.3591e-02f, 1.3235e-01f, 2.3752e-01f, -3.8349e-01f, 1.2053e+00f, 

        3.5273e-01f, -5.4940e-01f, -6.3167e-01f, -6.4300e-01f, -1.2537e+00f, -1.4624e+00f, -8.4623e-01f, -8.8255e-01f, 

        -1.0912e+00f, -8.4668e-01f, 2.1093e-01f, -8.8785e-01f, -2.2309e-01f, -6.3562e-01f, -1.0199e+00f, 5.2249e-01f, 

        -3.0461e-01f, -6.9040e-01f, -2.6660e-01f, 6.5148e-01f, -7.0548e-01f, -2.2470e-01f, 3.1417e-01f, -5.1685e-03f, 

        -1.3886e+00f, -2.1427e-02f, 3.0501e-01f, -7.5726e-02f, -4.9654e-02f, -5.4225e-02f, 7.3338e-02f, -1.1256e-01f, 

        3.5967e-02f, 5.1187e-02f, -3.6155e-01f, -9.1931e-02f, -7.0937e-01f, 1.2241e-01f, -4.7455e-01f, -1.8450e-01f, 

        -9.2319e-01f, 4.2549e-01f, -4.3612e-01f, 4.6163e-01f, 2.4074e-01f, 1.1661e+00f, -7.5859e-01f, 3.1479e-01f, 

        1.0393e+00f, 6.0917e-01f, -7.3152e-01f, -8.0148e-01f, -1.1144e-01f, -1.4708e+00f, 3.7487e-02f, -4.4137e-02f, 

        -5.7741e-01f, -4.1938e-01f, 1.7848e+00f, -9.7502e-01f, -2.0453e-02f, -8.2790e-01f, -6.2452e-01f, 6.5279e-01f, 

        -5.9911e-01f, 9.2889e-02f, -2.9294e-01f, -1.4929e-01f, 4.8551e-01f, -8.1762e-02f, 2.2231e-01f, 1.1110e+00f, 

        3.3057e-01f, 9.7430e-01f, -7.7377e-01f, 1.1658e-01f, -7.2976e-01f, -2.1500e-01f, 4.7089e-01f, -4.7582e-01f, 

        -2.3063e+00f, -5.4330e-02f, -5.7683e-01f, -2.4652e-01f, 6.6418e-01f, -4.1185e-01f, -2.6521e-01f, 7.2308e-01f, 

        -1.4643e-01f, -4.7805e-01f, -1.2212e-01f, 4.7778e-01f, -7.0750e-01f, -1.0029e+00f, -3.1802e-01f, 1.1329e-01f, 

        3.9569e-01f, -1.1541e+00f, 9.1407e-01f, -1.0931e+00f, -4.2965e-01f, -1.3127e+00f, -1.2814e+00f, -1.2542e+00f, 

        -1.1074e+00f, -1.6756e-01f, 6.9168e-02f, -1.4246e+00f, -1.0766e+00f, -3.3582e-01f, -9.1636e-01f, -8.1996e-01f, 

        -1.3298e+00f, 5.5081e-01f, -7.0649e-01f, -7.6001e-01f, -8.8713e-02f, 3.7924e-01f, -8.6603e-02f, -1.9949e-01f, 

        -9.0528e-02f, -3.0748e-01f, -2.7223e-01f, 4.4543e-01f, 1.9815e-01f, -4.7407e-02f, -3.9890e-01f, 8.3618e-01f, 

        -9.3226e-01f, -4.8240e-02f, -5.7171e-01f, -7.9044e-01f, -1.1127e+00f, 4.5385e-01f, -1.3606e+00f, -3.0674e-01f, 

        3.0478e-01f, 1.6383e+00f, -6.5431e-01f, 5.2833e-01f, 1.0973e+00f, 2.9022e-02f, 1.9180e-01f, -5.0237e-01f, 

        3.0614e-01f, -2.0647e-01f, 1.1792e-01f, -1.5163e-01f, -1.3081e+00f, 1.0470e-01f, -1.0686e+00f, -1.7546e+00f, 

        -2.0842e+00f, -1.5684e-01f, 6.7196e-02f, -6.1346e-02f, -4.1308e-02f, -1.0305e-01f, 3.1897e-02f, -7.5852e-02f, 

        -1.1780e-01f, -6.1464e-02f, -4.8851e-01f, -5.5026e-01f, -2.4675e-01f, -8.1003e-01f, -5.6080e-01f, -6.5941e-01f, 

        -1.6883e+00f, 6.4428e-02f, 2.3881e+00f, -5.4426e-01f, 4.1300e-01f, -1.0920e+00f, -1.4130e+00f, 5.1156e-01f, 

        -3.5907e-01f, -4.1635e-01f, 9.2280e-02f, -6.2977e-01f, 1.7961e-03f, -1.9467e-03f, -9.7634e-03f, -4.2111e-02f, 

        -3.4984e-02f, -7.1193e-02f, -6.0887e-02f, -1.3841e-01f, -9.4610e-02f, -3.0949e-01f, -2.4567e-01f, -5.2016e-01f, 

        6.4925e-01f, 8.0445e-03f, -2.8470e-03f, -9.3871e-02f, -6.8799e-01f, -7.3703e-01f, -1.2487e+00f, -4.6523e-01f, 

        3.8550e-01f, -2.3437e+00f, 2.9606e-01f, -7.8543e-01f, -1.3388e+00f, -1.5793e+00f, -3.8685e-01f, 8.0473e-02f, 

        -3.7591e-01f, -1.3526e-01f, -5.4565e-02f, 8.3217e-02f, -1.2594e-01f, -6.6949e-03f, -1.1424e-01f, -2.9729e-01f, 

        -1.3173e-01f, -1.0441e-01f, 1.7668e-01f, -3.3329e-02f, 6.1411e-02f, -1.0078e-01f, -9.7274e-02f, -8.2870e-01f, 

        -6.8157e-02f, -1.0940e-01f, -5.6360e-02f, -1.1433e-01f, -5.5772e-02f, -7.9674e-02f, -2.5101e-01f, -4.9910e-03f, 

        -8.6517e-02f, -1.2741e-01f, -7.0204e-02f, 1.6191e+00f, -1.8868e-01f, -6.8960e-01f, 2.6087e-01f, -3.1183e-02f, 

        -9.5513e-01f, -1.2942e-01f, 5.2110e-01f, -2.0766e-01f, -4.0955e-01f, -4.0832e-01f, 1.3282e+00f, 5.7551e-01f, 

        -1.3260e-01f, 3.2346e-01f, -6.1442e-01f, -1.0667e-01f, -2.1660e-01f, -3.0066e-02f, -2.9586e-01f, -5.6936e-01f, 

        -4.0194e-01f, -5.6341e-02f, -3.8182e-01f, -9.6193e-02f, -1.0720e-01f, 5.5941e-01f, -6.5261e-01f, -4.2702e-01f, 

        -9.0301e-01f, -1.0596e+00f, -5.2203e-02f, -1.1605e-01f, 1.3512e+00f, -4.6003e-01f, -4.9278e-01f, 2.7777e-01f, 

        -3.7442e-01f, 6.5936e-01f, 1.3970e+00f, 1.7162e-01f, 2.1787e-02f, -3.1356e-01f, -2.2421e-01f, -8.4618e-01f, 

        -7.3216e-01f, 1.2406e-02f, -4.3469e-01f, 1.3321e-01f, -1.8017e-01f, -4.4579e-01f, -1.4822e-01f, 3.8637e-02f, 

        -4.2281e-01f, 1.0087e-01f, 4.0845e-01f, -1.0611e-01f, 2.9901e-02f, -8.7171e-01f, -3.3170e-01f, -4.7700e-01f, 

        -5.3240e-01f, -4.9590e-01f, -3.8912e-01f, -2.1584e-01f, -1.1314e+00f, -5.7338e-01f, -7.3906e-02f, -7.0818e-01f, 

        -3.8545e-01f, -1.5112e-01f, -7.3907e-01f, -7.3815e-01f, -1.2260e-02f, -7.5178e-01f, -4.9854e-01f, -9.2474e-01f, 

        4.3365e-01f, -5.7848e-01f, -7.6712e-01f, 7.9695e-02f, -1.7155e-01f, -1.1740e-01f, -1.7952e-01f, 4.0532e-02f, 

        -2.6013e-01f, -6.3553e-02f, -1.5274e-01f, -1.2363e-01f, 4.2315e-01f, -4.9262e-01f, -2.5345e-01f, 2.2520e-01f, 

        -4.2434e-01f, 3.6202e-01f, -7.8058e-01f, 8.2273e-02f, -1.6202e-01f, 1.0840e-01f, 9.1185e-01f, -5.4749e-01f, 

        -1.0115e-01f, -3.4984e-01f, -2.3867e-01f, -1.8492e-01f, -8.3843e-03f, -1.8222e-01f, 7.3072e-01f, -2.7907e-01f, 

        -1.4891e-01f, -3.3015e-01f, -2.3032e-01f, -5.3760e-01f, -7.6010e-02f, -1.0348e-01f, 1.0255e+00f, -4.0395e-01f, 

        -4.0535e-01f, 8.8583e-02f, -1.8323e-01f, 1.9687e-01f, 1.1160e-01f, -8.5159e-01f, -3.3762e-01f, -2.9309e-01f, 

        -1.8409e-01f, -5.9920e-01f, -1.6324e-01f, 1.8370e-01f, -1.3662e+00f, -8.4969e-01f, -4.5069e-02f, -4.4784e-01f, 

        -2.5270e-01f, 2.6342e-02f, 4.5759e-01f, 1.4253e-01f, 5.2164e-01f, 2.8290e-01f, -1.5470e-01f, -7.6781e-01f, 

        3.1875e-02f, 1.3790e+00f, 6.6898e-01f, -2.1893e-01f, 3.8149e-02f, -1.7599e-01f, 1.4802e-01f, -2.1545e-01f, 

        1.0455e-02f, -2.7306e-01f, -2.1930e-01f, -1.3214e+00f, -7.7995e-01f, 4.7041e-01f, -1.0006e+00f, 2.0874e-01f, 

        5.9307e-02f, -6.4991e-01f, -3.0708e-01f, -8.5172e-02f, 3.3947e-02f, 1.5755e-01f, -8.1301e-01f, -4.7119e-01f, 

        -2.2027e-01f, -1.6038e-02f, -3.7660e-01f, -8.0870e-01f, -3.4957e-01f, -6.2800e-01f, -1.3546e-01f, 1.4944e-01f, 

        -2.2529e-01f, -1.1483e-01f, 6.8209e-02f, -4.7181e-01f, -2.6111e-01f, -2.0016e-02f, -8.5085e-01f, -2.2562e-03f, 

        1.7805e+00f, 7.6589e-01f, 2.8301e-02f, -2.0668e-01f, -2.2135e-01f, 3.3207e-01f, -1.2235e+00f, 3.1314e-01f, 

        -1.9746e-02f, 1.8531e-02f, -1.8537e-01f, 4.7210e-01f, -2.4886e-01f, -1.0359e+00f, -1.5348e-01f, 2.4966e-01f, 

        2.9758e-03f, -9.8401e-01f, -1.6958e+00f, -5.9733e-01f, -3.7358e-01f, -1.2112e+00f, 5.5408e-01f, -2.0426e-01f, 

        1.0102e+00f, -1.1934e-01f, 3.5979e-02f, -7.9633e-02f, -3.7609e-02f, -9.3360e-02f, -5.9252e-02f, -1.2474e-01f, 

        -1.1911e-01f, -7.0205e-02f, 1.3943e-01f, 1.1033e-01f, 1.0117e+00f, -6.2027e-01f, -9.6943e-03f, 4.4315e-02f, 

        -9.7468e-01f, -2.2601e-02f, 2.5916e-01f, 1.2490e+00f, -1.1869e-01f, 4.5041e-01f, -5.2444e-01f, -7.2982e-02f, 

        9.4848e-02f, 4.9147e-01f, -3.2358e-02f, -1.1434e-02f, 1.5709e-03f, -1.5101e-01f, -4.0992e-03f, -1.7392e-01f, 

        -5.8540e-02f, 3.0186e-02f, -1.3300e-01f, -1.8563e-01f, 2.0692e-02f, 1.4076e-01f, -1.3200e+00f, -9.9039e-01f, 

        4.1401e-01f, 6.9908e-01f, -1.7032e-01f, -1.0352e-01f, 3.7081e-01f, -4.2226e-01f, -4.9487e-01f, -4.2269e-01f, 

        -3.9304e-01f, -4.9172e-01f, 6.0920e-01f, -6.5355e-01f, -1.2702e+00f, -8.2387e-01f, -3.1081e-01f, -8.6146e-02f, 

        7.3209e-02f, 1.5000e-01f, -1.0823e-01f, -1.4797e-01f, -2.7812e-01f, -1.2132e-01f, -8.1674e-02f, -2.4395e-01f, 

        -1.2019e-01f, -6.8444e-02f, -5.2907e-02f, -8.9249e-02f, -5.4481e-03f, -2.9678e-02f, -1.6808e-01f, -1.8480e-01f, 

        -1.4590e-01f, -1.0861e-01f, -1.8400e-01f, -2.6669e-02f, -1.7343e-01f, -1.6010e-01f, -1.6651e-01f, -1.8120e-01f, 

        -1.2220e-01f, -4.3992e-02f, -2.0415e-01f, -1.7537e-01f, -1.9241e-01f, -1.1524e-01f, -1.3551e-01f, -5.4088e-02f, 

        -1.1427e-01f, -2.7688e-01f, -1.8860e-01f, -4.1734e-01f, -5.2341e-01f, -1.6364e-01f, -2.7196e-01f, -1.8841e-01f, 

        -1.4456e-01f, -2.4961e-01f, -1.9522e-01f, -1.6367e-01f, -1.2933e-01f, -5.1917e-02f, -1.0990e-01f, -9.3133e-02f, 

        -1.3424e-01f, -5.4807e-02f, -1.3034e-01f, -6.8372e-02f, -6.1245e-02f, -3.8346e-01f, -2.9194e-01f, -2.0036e-01f, 

        -5.1027e-01f, -3.8692e-01f, -2.6261e-01f, -3.6980e-01f, -2.4099e-01f, -3.1242e-01f, -1.5145e-01f, -5.5313e-02f, 

        -2.3388e-02f, -2.0921e-01f, -6.9166e-02f, -1.5331e-01f, -4.2565e-02f, 7.4571e-02f, -1.9007e-01f, -4.5987e-01f, 

        2.6337e-02f, -1.3715e-01f, -3.6174e-01f, -9.2732e-02f, -2.6533e-01f, -6.1852e-02f, -1.2988e-01f, -4.6281e-01f, 

        -1.3666e-02f, -9.7979e-02f, -1.8217e-01f, -1.7213e-01f, -1.5810e-01f, -1.8127e-01f, -1.2836e-01f, -1.0423e-01f, 

        -7.2417e-02f, -1.6681e-01f, -1.3752e-01f, -1.9468e-02f, -1.5602e-01f, -7.0532e-02f, 5.5426e-02f, -6.0830e-02f, 

        4.1491e-02f, -1.2432e-01f, -1.4539e+00f, -3.2741e+00f, -1.2061e+00f, -3.9280e+00f, -2.7725e+00f, -2.9742e+00f, 

        -1.7579e+00f, -3.0200e+00f, -9.7004e-01f, -1.0198e-01f, -2.7006e-01f, 9.1192e-02f, -2.3059e-01f, 1.0986e-01f, 

        -1.2098e-01f, -2.3033e-01f, -1.3398e-01f, -1.8589e-01f, -6.0930e-01f, -2.1558e-01f, -3.8596e-02f, -5.0918e-01f, 

        -1.5743e-01f, -1.0960e-01f, -1.3812e-01f, -1.5003e-01f, -2.7130e-01f, -4.4164e-01f, -1.8979e-01f, 1.9127e-02f, 

        -1.2390e-01f, 6.0674e-02f, -1.8283e-01f, -9.6990e-02f, -7.7490e-02f, -1.8086e-01f, -1.3617e-01f, -2.2243e-01f, 

        -4.3298e-02f, -2.0655e-01f, -2.0646e-01f, -2.1171e-01f, -2.1663e-01f, -1.1409e-01f, -1.5217e-01f, -1.9389e-01f, 

        -1.6888e-01f, -1.0745e-02f, -2.2719e-01f, -2.3148e-01f, -2.1821e-01f, -2.0276e-01f, -3.4730e-02f, -2.1200e-01f, 

        -1.0602e+00f, -2.0644e-01f, -8.2120e-01f, -1.2364e-01f, -4.2431e-02f, -2.1475e-01f, -4.3472e-01f, -4.3262e-02f, 

        -5.2175e-01f, -2.2857e-01f, -8.9135e-02f, -1.5226e-01f, -1.9867e-01f, -1.4032e-01f, -2.9380e-02f, -1.8152e-01f, 

        -1.3607e-01f, -1.4280e-01f, -6.5202e-01f, -1.3169e-01f, 7.9257e-02f, -1.8295e-01f, -2.9742e-01f, -2.5510e-01f, 

        2.9872e-02f, -1.6808e-01f, -2.2193e-01f, -1.1358e+00f, -3.2304e-01f, -3.8004e-01f, -5.7290e-01f, -2.8952e-01f, 

        -3.3271e-01f, -2.9013e-01f, -1.9827e-01f, -2.9967e-01f, -1.1692e-01f, -8.9789e-02f, -1.9161e-01f, -2.4009e-01f, 

        4.0449e-02f, -6.3786e-02f, -3.1025e-01f, -1.5365e-01f, -1.1581e-01f, -7.2693e-02f, -1.2228e-01f, -7.8930e-02f, 

        -1.9461e-01f, -6.1580e-02f, 5.6706e-04f, -1.5313e-01f, -5.6557e-02f, -1.5609e-01f, -1.2259e-01f, -3.3840e-02f, 

        -1.3359e-01f, -3.5385e-01f, -3.3164e-02f, -6.9294e-02f, -2.0368e-01f, -8.9810e-02f, -2.0226e-02f, -3.2670e-01f, 

        -4.4640e-02f, 3.3574e-02f, -3.0343e-01f, -1.8973e-01f, -1.6401e-01f, -2.1011e-01f, -2.7527e-01f, -1.6720e-01f, 

        -1.9322e-01f, 3.2779e-02f, -2.8862e-01f, -4.5543e-01f, 1.0992e-02f, -4.2889e-01f, -2.0001e-01f, -2.4217e-01f, 

        -1.4766e-01f, -1.2992e-01f, -1.6271e-01f, -5.2835e-02f, -1.1168e-01f, -1.3742e-01f, 1.1649e-03f, -1.7951e-01f, 

        -1.8027e-01f, -1.2551e-01f, -9.8126e-01f, -4.6010e-01f, -1.7452e-01f, -4.3373e-01f, -7.6117e-01f, -1.1706e-01f, 

        -5.7319e-01f, -9.6469e-02f, -3.4477e-01f, -1.1764e-01f, -1.3578e-01f, -4.0076e-02f, -1.2134e-01f, -6.0641e-02f, 

        2.0535e-02f, -7.6977e-02f, -1.3064e-01f, -1.1515e-01f, -9.7366e-02f, -1.2133e-01f, -1.6845e-01f, -2.6826e-01f, 

        -1.5773e-01f, -1.2152e-01f, -1.7930e-01f, -1.0821e-01f, -7.5603e-02f, -6.3086e-01f, -1.1579e-01f, -3.1166e-01f, 

        -7.6941e-01f, -5.3361e-01f, -2.3951e-01f, -1.6044e-01f, -1.8124e-01f, -3.0734e-01f, -8.8412e-02f, -5.7692e-03f, 

        3.4932e-02f, -1.3443e-01f, 2.0252e-02f, -4.6754e-02f, -1.0182e-01f, -6.0838e-02f, -1.5106e-01f, 9.5835e-02f, 

        4.6184e-02f, -7.8968e-02f, 3.2373e-01f, -4.1291e-02f, -5.5468e-02f, -1.9482e-01f, 2.8740e-02f, -3.0505e-03f, 

        -1.4052e-01f, -1.9701e-01f, -1.2142e-01f, -1.3179e-01f, -7.3880e-02f, -1.9747e-01f, -1.6188e-01f, -1.6552e-01f, 

        -1.3992e-01f, -4.2348e-02f, -1.2877e-01f, -1.3094e-01f, -6.6133e-02f, -8.5903e-02f, -8.6377e-02f, -1.7108e-01f, 

        -1.3077e-01f, -1.4261e-01f, -1.2698e-01f, -1.8526e-01f, -1.6849e-01f, -1.8242e-01f, -1.1793e-01f, -1.2069e-01f, 

        -1.2451e-01f, -1.4332e-01f, -1.6773e-01f, -9.8190e-01f, -1.3859e+00f, -1.2907e+00f, -6.7085e-01f, -4.5439e-01f, 

        -5.7811e-01f, -7.9967e-01f, -3.1991e-01f, -5.1802e-01f, -7.2845e-02f, -1.6095e-01f, -1.6186e-01f, -1.3279e-01f, 

        -1.2918e-01f, -1.2954e-01f, -1.6480e-01f, -1.2376e-01f, -1.1922e-01f, -3.6276e-01f, -2.5463e-01f, -1.4361e-01f, 

        -3.2319e-01f, -2.1585e-01f, -1.6038e-01f, -3.3835e-01f, -1.7868e-01f, -1.7925e-01f, -7.3759e-02f, -1.1790e-01f, 

        -1.2964e-01f, -1.2566e-01f, -1.4300e-01f, -4.2829e-02f, -1.2772e-01f, 2.3734e-04f, -6.4042e-02f, -1.4484e-01f, 

        -9.6905e-02f, -8.6660e-02f, -2.3542e-01f, -2.6185e-01f, -2.0284e-01f, -1.3815e-01f, -9.3960e-02f, -1.3923e-01f, 

        -1.5790e-01f, -1.1947e-01f, -1.4496e-01f, -2.1102e-01f, -2.4251e-02f, -1.4593e-01f, -1.8052e-01f, -4.3438e-02f, 

        -1.1739e-01f, -1.3197e-01f, -1.4904e-01f, -1.4459e-01f, -1.6621e-01f, -6.5219e-02f, -7.9397e-02f, -1.4515e-01f, 

        -8.0502e-02f, -1.3841e-01f, -2.1226e+00f, -1.7970e+00f, -1.7292e+00f, -1.6968e+00f, -1.3812e+00f, -1.5138e+00f, 

        -1.6025e+00f, -1.3553e+00f, -1.4693e+00f, -1.0051e-01f, -1.1625e-01f, -6.2360e-02f, -1.0387e-02f, -3.0756e-02f, 

        -6.8468e-02f, -6.8311e-02f, -7.2169e-02f, -6.2231e-02f, -3.6428e-01f, -2.0908e-01f, -1.8567e-01f, -4.7308e-01f, 

        -1.7859e-01f, -2.4488e-01f, -2.3157e-01f, -1.7301e-01f, -1.4678e-01f, -1.4186e-01f, -7.5306e-02f, -1.5669e-01f, 

        -1.4894e-01f, -1.3583e-01f, -1.2392e-01f, -1.2469e-01f, -7.3051e-02f, -1.4388e-01f, -8.2985e-02f, -7.3515e-02f, 

        2.4121e-03f, -6.2877e-02f, 1.1349e-02f, -1.0795e-01f, -2.3344e-02f, -6.1866e-02f, -2.8513e-02f, -1.9526e-01f, 

        -2.1384e-01f, -2.6147e-01f, -2.2924e-01f, -1.8379e-01f, -2.3861e-01f, -1.6119e-01f, -1.7674e-01f, -1.7009e-01f, 

        -2.4539e-01f, -1.7541e-01f, -2.6420e-01f, -1.3272e-01f, -2.8305e-02f, -1.6312e-01f, -1.7767e-01f, -1.4684e-01f, 

        -9.1497e-02f, -2.3441e-01f, -2.8743e-01f, -3.2997e-01f, -1.9582e-01f, -2.5993e-01f, -2.6912e-01f, -2.1436e-01f, 

        -2.3324e-01f, -1.8656e-01f, -1.9762e-01f, -6.6665e-02f, -1.3117e-01f, -1.5990e-01f, -7.3020e-02f, -1.7346e-01f, 

        -1.6450e-01f, -8.0403e-02f, -1.5171e-01f, -3.2365e-01f, -1.5613e-01f, -2.7573e-01f, -1.8575e-01f, -1.3295e-01f, 

        -2.5455e-01f, -1.2594e-01f, -9.0108e-02f, -1.5871e-01f, -2.0626e-01f, -1.2438e-01f, -2.1443e-01f, -1.8436e-01f, 

        -1.0156e-01f, -1.4719e-01f, -1.8846e-01f, -1.3805e-01f, -1.3046e-01f, -2.0336e-01f, -1.9866e-01f, -2.2963e-01f, 

        -1.6178e-01f, -1.9959e-01f, -1.6242e-01f, -1.8125e-01f, -2.0844e-01f, -1.8668e-01f, -1.8971e-01f, -1.4079e-01f, 

        -1.3516e-01f, -2.4523e-01f, -1.7020e-01f, -7.8702e-02f, -1.0179e-01f, -1.6723e-01f, -1.3198e-01f, -1.4598e-01f, 

        -2.2208e-01f, -1.7154e-01f, -1.0609e-01f, -2.5576e-01f, -7.1258e-02f, -1.9996e-01f, -2.4906e-02f, -1.5266e-01f, 

        -1.0954e-01f, -1.4829e-01f, -1.6896e-01f, -1.0378e-01f, -1.6628e-01f, -8.5428e-02f, -1.6217e-01f, -5.8047e-02f, 

        -1.7371e-01f, -1.1085e-01f, -7.7445e-02f, -6.8206e-02f, -7.8948e-02f, -1.2897e-01f, -3.9498e-02f, -1.0948e-01f, 

        -8.0336e-02f, -1.5158e-01f, -2.2535e-01f, -2.4444e-01f, -1.2226e-01f, -1.7846e-01f, -1.8989e-01f, -1.3353e-01f, 

        -2.1296e-01f, -1.8689e-01f, -1.0136e-01f, -2.0679e-01f, -1.1170e-01f, -1.5487e-01f, -8.1368e-02f, -1.5220e-01f, 

        -9.6363e-02f, -6.9005e-02f, -7.1633e-02f, -1.0298e-01f, -7.2841e-02f, -5.4891e-02f, -9.4448e-02f, -7.0370e-02f, 

        -8.4966e-02f, -2.9061e-02f, -6.6491e-02f, -1.3623e-01f, -1.3099e-01f, -1.5128e-01f, -9.7583e-02f, -1.8818e-01f, 

        -1.5971e-01f, -1.9213e-01f, -1.7182e-01f, -8.6086e-02f, -1.5272e-01f, -2.5668e-02f, -9.5929e-02f, -1.6681e-01f, 

        -5.4915e-02f, -4.4578e-02f, -4.7185e-02f, -1.1947e-01f, -1.1001e-01f, -1.6339e-01f, -1.4384e-01f, 1.0164e-01f, 

        -4.7386e-02f, -8.1709e-02f, -1.7747e-01f, 4.4458e-02f, -1.0151e-01f, 4.4548e-02f, 9.1761e-02f, -6.0015e-02f, 

        -6.6750e-02f, -9.6217e-02f, -6.6610e-02f, -1.5466e-01f, -7.1937e-02f, -9.7144e-02f, -9.8252e-02f, -1.6687e-01f, 

        -1.0210e-01f, -6.8245e-02f, -1.7478e-01f, -7.1028e-02f, -1.7485e-01f, -1.7253e-01f, -1.6269e-01f, -1.0703e-01f, 

        -1.7205e-01f, -1.4959e-01f, -1.8041e-01f, -1.3856e-01f, -2.1072e-01f, -1.9187e-01f, -1.5589e-01f, -1.4486e-01f, 

        -1.8314e-01f, -1.2507e-01f, -1.8082e-01f, -1.3857e+00f, -1.4776e+00f, -1.0044e+00f, -5.0012e-01f, -4.6730e-01f, 

        -5.0022e-01f, -4.4737e-01f, -5.0923e-01f, -6.2022e-01f, -1.4205e-01f, -1.3660e-01f, -1.6071e-01f, -1.4390e-01f, 

        -1.5948e-01f, -1.7718e-01f, -1.1313e-01f, -1.3314e-01f, -7.0357e-02f, -1.8819e-01f, -1.5139e-01f, -1.5743e-01f, 

        -2.5304e-01f, -1.9319e-01f, -1.9572e-01f, -3.3178e-01f, -2.1690e-01f, -2.3847e-01f, -1.4718e-01f, -1.7772e-01f, 

        -7.1938e-02f, -1.3469e-01f, -1.2346e-01f, -1.7236e-01f, -1.8710e-01f, -1.3307e-01f, -1.6843e-01f, -2.4781e-01f, 

        -1.3671e-01f, -2.7575e-01f, -2.2203e-01f, -1.8495e-01f, -1.5930e-01f, -1.3989e-01f, -1.9388e-01f, -2.0837e-01f, 

        -1.9135e-01f, -9.9572e-02f, -1.4625e-01f, -1.6430e-01f, -8.0240e-02f, -2.9725e-01f, -1.2544e-01f, -1.8839e-01f, 

        -2.1332e-01f, -9.5111e-02f, -1.5942e-01f, -1.1914e-01f, -1.0445e-01f, -1.6998e-01f, -1.8169e-01f, -1.1544e-01f, 

        -7.8349e-02f, -1.4114e-01f, -2.4932e+00f, -1.6755e+00f, -1.3869e+00f, -2.1428e+00f, -9.5932e-01f, -1.3401e+00f, 

        -1.4644e+00f, -8.0829e-01f, -1.1448e+00f, -3.4518e-02f, -1.4193e-01f, -8.3889e-02f, -3.9639e-02f, -5.8133e-02f, 

        -8.4162e-02f, -1.6454e-01f, -1.3604e-01f, -6.7498e-02f, -3.6645e-01f, -2.9525e-01f, -1.3584e-01f, -3.4809e-01f, 

        -2.3723e-01f, -1.5778e-01f, -1.7840e-01f, -2.1960e-01f, -1.5431e-01f, -1.7225e-01f, -1.2697e-01f, -6.6148e-02f, 

        -8.7517e-02f, -1.0022e-01f, -1.4359e-01f, -1.2520e-01f, -1.0691e-01f, -1.2137e-01f, -8.1882e-02f, -8.6040e-02f, 

        -7.8960e-02f, -1.2648e-01f, -1.6293e-01f, -1.5318e-01f, -8.1172e-02f, -2.7332e-02f, -8.5235e-02f, -2.4421e-01f, 

        -2.3043e-01f, -1.8866e-01f, -2.2566e-01f, -2.0184e-01f, -2.0905e-01f, -2.1030e-01f, -2.0582e-01f, -2.0825e-01f, 

        -2.0280e-01f, -1.1000e-01f, -2.1500e-01f, -1.4758e-01f, -1.5378e-01f, -1.3975e-01f, -1.7737e-01f, -1.1494e-01f, 

        -1.6727e-01f, -2.5103e-01f, -2.5789e-01f, -1.8741e-01f, -1.6041e-01f, -1.3743e-01f, -2.3176e-01f, -1.9187e-01f, 

        -2.1209e-01f, -2.6763e-01f, -2.3727e-01f, -1.2152e-01f, -1.6995e-01f, -1.1197e-01f, -1.3061e-01f, -1.3520e-01f, 

        -1.1939e-01f, -1.1961e-01f, -1.8420e-01f, -3.6356e-01f, -2.6801e-01f, -3.6969e-01f, -2.3391e-01f, -1.3057e-01f, 

        -1.6310e-01f, -1.1011e-01f, -1.9447e-01f, -2.3797e-01f, -1.8712e-01f, -1.0594e-01f, -1.9447e-01f, -1.7308e-01f, 

        -8.6365e-02f, -1.2428e-01f, -1.7691e-01f, -1.8158e-01f, -1.7801e-01f, -1.1473e-01f, -2.2829e-01f, -1.1364e-01f, 

        -1.2758e-01f, -1.7125e-01f, -1.6618e-01f, -1.1749e-01f, -1.4051e-01f, -1.1217e-01f, -8.2695e-02f, -1.7107e-01f, 

        -1.0003e-01f, -1.8275e-01f, -1.6941e-01f, -1.2190e-01f, -1.7062e-01f, -1.7240e-01f, -1.2824e-01f, -2.4503e-01f, 

        -8.1103e-02f, -1.5112e-01f, -1.1718e-01f, -1.7480e-01f, -1.3487e-01f, -2.2600e-01f, -1.3068e-01f, -1.4555e-01f, 

        -1.9558e-01f, -1.5687e-01f, -1.1453e-01f, -2.1713e-01f, -1.4168e-01f, -1.9647e-01f, -2.0312e-01f, -1.5820e-01f, 

        -1.3996e-01f, -1.1638e-01f, -8.0779e-02f, -9.5829e-02f, -8.5203e-02f, -1.0258e-01f, -1.1744e-01f, -9.5338e-02f, 

        -1.6770e-01f, -9.8313e-02f, -1.4058e-01f, -2.3662e-01f, -1.6094e-01f, -2.3263e-01f, -1.9476e-01f, -1.3583e-01f, 

        -1.7433e-01f, -1.9885e-01f, -1.5248e-01f, -2.0874e-01f, -1.5996e-01f, -8.6455e-02f, -1.8585e-01f, -1.2113e-01f, 

        -8.2643e-02f, -9.2299e-02f, -1.1341e-01f, -1.8528e-01f, -1.0864e-01f, -1.1920e-01f, -1.4053e-01f, -7.4282e-02f, 

        -1.3879e-01f, -1.7600e-01f, -1.7547e-01f, -8.8182e-02f, -1.2686e-01f, -9.1967e-02f, -1.5697e-01f, -3.1017e-01f, 

        -1.9408e-01f, -1.1502e-01f, -1.3309e-01f, -6.2078e-02f, -2.1749e-01f, -2.3396e-01f, -8.1665e-02f, -1.6498e-01f, 

        -1.1184e-01f, -1.4002e-01f, -1.1353e-01f, -1.3867e-01f, -1.8611e-01f, -1.5544e-01f, -8.7049e-02f, -1.1261e-01f, 

        2.4014e-02f, -1.4349e-01f, -1.4111e-02f, -9.1280e-02f, -1.7108e-01f, -1.1235e-01f, -1.4411e-01f, -2.0733e-01f, 

        -1.2816e-01f, -1.2783e-01f, -1.6720e-01f, -6.2645e-02f, -1.6641e-01f, -6.2370e-02f, -5.3583e-02f, -8.5081e-02f, 

        -1.0692e-01f, -1.3839e-01f, -5.5073e-02f, -4.0084e-02f, 1.9771e-03f, -8.4409e-02f, -7.1842e-02f, -3.2705e-02f, 

        -9.5162e-02f, -1.3450e-01f, -1.7937e-01f, -1.0028e-01f, -1.0882e-01f, -7.9639e-02f, -1.3933e-01f, -1.1560e-01f, 

        -1.6002e-01f, -1.4946e-01f, -9.7761e-02f, -1.2889e+00f, -1.2241e+00f, -1.0785e+00f, -5.1351e-01f, -4.4303e-01f, 

        -4.1248e-01f, -5.6493e-01f, -3.2245e-01f, -4.5638e-01f, -1.2132e-01f, -7.9538e-02f, -1.3029e-01f, -1.3354e-01f, 

        -9.8480e-02f, -8.9287e-02f, -1.2808e-01f, -1.6494e-01f, -1.5023e-01f, -2.2780e-01f, -1.2868e-01f, -6.1589e-02f, 

        -2.2768e-01f, -1.3914e-01f, -7.7165e-02f, -2.3914e-01f, -1.3390e-01f, -8.6196e-02f, -8.1607e-02f, -1.7199e-01f, 

        -2.0791e-01f, -7.8084e-02f, -1.7714e-01f, -8.6486e-02f, -9.6761e-02f, -1.1172e-01f, -1.6340e-01f, -1.0687e-01f, 

        -8.3956e-02f, -2.3737e-01f, -2.0564e-01f, -1.2455e-01f, -1.7388e-01f, -1.0403e-01f, -1.8987e-01f, -2.5804e-01f, 

        -1.7549e-01f, -9.8243e-02f, -1.8617e-01f, -1.5550e-01f, -1.6258e-01f, -1.5722e-01f, -1.2545e-01f, -1.4210e-01f, 

        -1.2685e-01f, -1.0420e-01f, -8.4845e-02f, -1.2617e-01f, -6.7361e-02f, -1.3041e-01f, -1.4650e-01f, -1.2497e-01f, 

        -1.1553e-01f, -7.5570e-02f, -1.9463e+00f, -1.4029e+00f, -1.1367e+00f, -1.3650e+00f, -9.5478e-01f, -1.3217e+00f, 

        -1.2185e+00f, -9.6079e-01f, -1.1918e+00f, -6.6962e-02f, -1.0913e-01f, -1.1389e-01f, -1.0348e-01f, -1.1993e-01f, 

        -4.2695e-02f, -6.0353e-02f, -5.1860e-02f, -1.4864e-01f, -3.0690e-01f, -2.6440e-01f, -1.8002e-01f, -2.4522e-01f, 

        -2.1419e-01f, -9.9247e-02f, -1.7978e-01f, -9.4031e-02f, -1.5652e-01f, -7.4089e-02f, -5.7234e-02f, -1.3584e-01f, 

        -1.2315e-01f, -8.1996e-02f, -1.6978e-01f, -7.8238e-02f, -1.6088e-01f, -7.0308e-02f, -9.0854e-02f, -7.8509e-02f, 

        2.0892e-02f, -4.6663e-02f, 4.5302e-02f, -2.9185e-02f, 1.4145e-02f, -1.1694e-01f, -5.3851e-02f, -1.7678e-01f, 

        -9.9058e-02f, -2.0164e-01f, -1.3685e-01f, -1.3987e-01f, -1.7515e-01f, -1.9662e-01f, -1.1614e-01f, -1.7654e-01f, 

        -1.4540e-01f, -1.3960e-01f, -2.8738e-01f, -1.2037e-01f, -1.1490e-01f, -1.2887e-01f, -1.7987e-01f, -7.3220e-02f, 

        -1.8269e-01f, -1.6135e-01f, -2.1777e-01f, -1.6008e-01f, -1.2288e-01f, -2.1871e-01f, -1.3217e-01f, -1.7369e-01f, 

        -1.9523e-01f, -1.4903e-01f, -1.3322e-01f, -7.4127e-02f, -1.9314e-01f, -8.5214e-02f, -1.3998e-01f, -9.3002e-02f, 

        -1.2935e-01f, -1.6734e-01f, -1.0851e-01f, -2.2360e-01f, -2.6090e-01f, -2.4318e-01f, -1.1222e-01f, -7.0884e-02f, 

        -1.3597e-01f, -8.9042e-02f, -1.6052e-01f, -2.0603e-01f, -1.4817e-01f, -8.2726e-02f, -1.3233e-01f, -1.6905e-01f, 

        -1.2698e-01f, -1.7309e-01f, -1.8450e-01f, -8.3630e-02f, -1.6729e-01f, -1.0643e-01f, -1.4732e-01f, -1.5080e-01f, 

        -1.2735e-01f, -7.6285e-02f, -1.4006e-01f, -8.5943e-02f, -1.0657e-01f, -1.0836e-01f, -1.8423e-01f, -1.5453e-01f, 

        -1.8491e-01f, -1.7418e-01f, -1.3180e-01f, -1.3778e-01f, -4.7120e-02f, -1.0773e-01f, -1.6828e-01f, -2.1222e-01f, 

        -7.1246e-02f, -1.1694e-01f, -1.0324e-01f, -1.8863e-01f, -8.8775e-02f, -1.0521e-01f, -1.4747e-01f, -5.5182e-02f, 

        -1.1557e-01f, -1.7141e-01f, -1.7686e-01f, -1.5839e-01f, -1.2756e-01f, -1.7402e-01f, -6.6342e-02f, -1.9971e-01f, 

        -1.6098e-01f, -1.3381e-01f, -8.7065e-02f, -1.0347e-01f, -9.5921e-02f, -5.0105e-02f, -1.5357e-01f, -7.4085e-02f, 

        -1.3884e-01f, -1.1917e-01f, -2.1246e-01f, -2.1924e-01f, -2.0772e-01f, -1.1278e-01f, -1.5669e-01f, -9.2721e-02f, 

        -1.9971e-01f, -1.8364e-01f, -1.4649e-01f, -8.0201e-02f, -1.2340e-01f, -8.1110e-02f, -7.1221e-02f, -6.8491e-02f, 

        -7.1846e-02f, -7.4666e-02f, -1.2375e-01f, -1.3273e-01f, -7.8070e-02f, -5.9725e-02f, -1.0259e-01f, -7.1875e-02f, 

        -1.2687e-01f, -6.2343e-02f, -1.0597e-01f, -1.2371e-01f, -1.4358e-01f, -1.2703e-01f, -1.1331e-01f, -1.9770e-01f, 

        -1.5915e-01f, -1.6159e-01f, -1.9570e-01f, -9.4129e-02f, -1.2940e-01f, -2.5038e-01f, -6.4977e-02f, -1.5036e-01f, 

        -8.8506e-02f, -1.1087e-01f, -1.5012e-01f, -1.6035e-01f, -9.7058e-02f, -1.0900e-01f, -1.3358e-01f, -9.0163e-02f, 

        -8.3616e-02f, -8.9863e-02f, -2.3339e-01f, -1.0695e-01f, -7.2214e-02f, -1.8980e-01f, -1.4477e-01f, -2.1646e-01f, 

        1.4694e-01f, -1.5011e+00f, -7.2157e-02f, -1.1920e+00f, -1.0337e+00f, 3.4336e-01f, -4.1889e-01f, -3.2547e-01f, 

        8.2806e-01f, -2.7055e-02f, -1.8981e-01f, 1.3864e-01f, -2.1398e-01f, -2.4640e-01f, 1.5914e-01f, 1.0549e-02f, 

        2.3372e-01f, -1.2536e-02f, 4.1027e-01f, -4.6663e-01f, 1.3392e+00f, 2.3210e-01f, 6.3772e-01f, 6.7327e-01f, 

        -3.8185e-01f, -1.9168e+00f, -1.4172e+00f, -1.9839e-01f, -2.4326e-01f, 2.6557e-01f, 3.5336e-01f, 9.3044e-01f, 

        1.6185e+00f, 4.2864e-01f, 8.5469e-01f, 6.1344e-01f, 7.3222e-01f, -1.1629e+00f, -7.9458e-02f, 5.1335e-01f, 

        -4.1545e-01f, 4.4100e-01f, 7.8621e-01f, 7.8211e-01f, 5.7419e-01f, -1.5879e-03f, -5.8671e-01f, -2.4271e-01f, 

        1.2861e-02f, -6.6231e-01f, 1.4254e-01f, -1.0563e-01f, 8.0640e-01f, 3.6033e-03f, 5.9477e-01f, -4.2866e-01f, 

        5.0515e-01f, -9.6997e-02f, 4.3475e-02f, 2.4694e+00f, 6.2642e-01f, 2.6986e-01f, 1.4361e+00f, -3.5382e-01f, 

        -1.2393e+00f, 1.1537e-01f, 4.5153e-01f, -5.1960e-01f, 1.4550e+00f, -1.0256e+00f, -9.9138e-01f, 6.3797e-01f, 

        -7.0669e-01f, -3.9003e-01f, 4.9831e-01f, -2.0889e-01f, -5.4303e-01f, 6.7401e-01f, -1.4687e-01f, -1.1083e+00f, 

        -5.6803e-02f, 1.0768e+00f, -4.5909e-01f, -7.1415e-01f, -1.0959e-01f, -3.9846e-01f, 1.4237e+00f, 9.0724e-01f, 

        -5.4472e-02f, 1.3132e+00f, 1.5190e-02f, -2.7692e-01f, -8.9320e-01f, -1.2726e-03f, -2.6598e-01f, -7.8976e-01f, 

        3.5004e-01f, -2.0857e-01f, -2.4895e-01f, 2.7794e-01f, 4.9925e-01f, -3.9514e-01f, -1.9045e-01f, -4.9435e-01f, 

        -4.1031e-01f, 7.8613e-02f, 4.3985e-02f, -7.2852e-02f, 8.0764e-01f, 5.5452e-01f, 4.2087e-01f, 1.2644e+00f, 

        9.7251e-01f, 2.2294e-01f, 4.5132e-01f, 8.1733e-01f, 3.7264e-02f, -2.6996e-01f, -7.4077e-01f, 1.0304e+00f, 

        -1.5816e+00f, -1.5773e+00f, 2.3375e+00f, 9.4402e-01f, -5.2287e-01f, 9.9595e-01f, 4.8409e-01f, 8.4066e-01f, 

        1.5712e+00f, -1.5731e+00f, 3.5154e-02f, 6.1000e-01f, -1.0570e+00f, 3.2972e-03f, 2.0728e-01f, 7.9404e-01f, 

        -3.6567e-01f, -8.3859e-01f, -1.3539e-01f, -9.9847e-02f, -1.7631e+00f, -3.2853e+00f, -1.8805e+00f, 1.1253e-01f, 

        6.4462e-01f, 1.3626e-01f, 2.2854e-01f, 2.4435e-01f, -1.1446e+00f, -5.7296e-01f, 3.9988e-01f, -1.5837e+00f, 

        -4.4357e-01f, 1.5353e-01f, 2.7881e-01f, 6.7321e-01f, -1.2068e-01f, -1.6209e-01f, -6.8265e-03f, 2.2180e-01f, 

        -2.2854e-01f, 3.9346e-01f, -9.1462e-01f, -2.2088e-01f, -5.7873e-01f, -3.2466e-01f, -1.8940e-01f, 6.0210e-01f, 

        -9.1757e-01f, -2.2033e-01f, 2.6426e-01f, -6.6776e-01f, -4.9847e-01f, -7.0844e-01f, 3.3005e-01f, 1.1942e+00f, 

        1.8245e+00f, 7.4822e-01f, 7.2539e-01f, 1.0290e+00f, -1.2760e+00f, -3.0497e-01f, -4.3222e-01f, -1.8616e+00f, 

        -1.7444e+00f, -3.0023e-01f, -1.8899e+00f, -4.8821e-01f, -1.0254e+00f, 8.5671e-01f, -3.2932e-01f, -1.3591e-01f, 

        -2.2331e-01f, -1.4792e-01f, 1.2331e+00f, 2.0636e-01f, -2.9814e-01f, 5.6865e-01f, -1.0181e+00f, -8.1274e-01f, 

        -7.8867e-01f, -9.2563e-01f, -1.2457e+00f, -5.2719e-01f, -1.5792e+00f, -1.6294e+00f, -9.4103e-01f, 3.0580e-01f, 

        -3.4705e-01f, -1.8166e-01f, 4.6229e-01f, 1.6714e-01f, -2.9181e-01f, -7.8482e-02f, 2.1255e+00f, -1.3267e+00f, 

        2.6576e-01f, -9.0981e-01f, -1.1629e+00f, 1.8840e-02f, -9.0632e-01f, -5.4780e-01f, -5.1240e-01f, -1.3699e+00f, 

        1.4356e-01f, -3.0308e-02f, -1.7817e-01f, 5.8289e-02f, -4.3624e-01f, 6.4307e-01f, -1.9114e-01f, -9.5210e-03f, 

        -2.9188e-01f, -1.1381e-01f, 3.5675e-01f, 1.6153e-01f, -1.3069e+00f, -3.4851e-01f, 5.0682e-01f, -2.5073e-01f, 

        -3.5349e-01f, -1.3607e+00f, 5.2166e-01f, -1.8591e+00f, -2.4981e-02f, 7.7928e-01f, 8.2134e-01f, 9.3702e-01f, 

        7.9418e-01f, -4.0187e-01f, 3.0705e-01f, 2.2592e+00f, -1.1845e-01f, -6.7538e-01f, -2.4001e-01f, -1.8108e-01f, 

        -5.5762e-02f, 3.2163e-02f, 3.3647e-02f, -7.7443e-02f, -4.9036e-02f, 2.5261e-01f, 1.3449e-01f, -2.9697e-01f, 

        5.8541e-01f, 7.9542e-01f, -1.2322e+00f, 1.5864e+00f, 4.9516e-01f, -6.5679e-01f, -1.5607e+00f, -1.4216e+00f, 

        2.3389e-01f, -1.1868e+00f, -1.2374e+00f, -2.1558e-01f, -1.0288e+00f, -2.4493e-01f, 7.6748e-01f, 1.9254e-01f, 

        6.0968e-02f, 1.7322e-01f, 3.1207e-01f, 1.6954e-01f, 1.1967e-01f, 2.6310e-01f, 3.3814e-01f, -2.1858e-01f, 

        -6.9775e-01f, 1.9165e-01f, -4.9997e-01f, -1.2923e-01f, 6.9427e-01f, -2.5212e-01f, -3.8729e-01f, -1.6858e-01f, 

        1.1892e-01f, 4.3694e-02f, -2.3334e-01f, -1.5998e-01f, -2.2409e-01f, -5.1560e-02f, -1.2988e-01f, -1.1735e-01f, 

        -1.0454e-01f, -1.7516e-01f, 9.9077e-01f, 2.9553e-01f, 7.5564e-03f, 3.4749e-02f, 5.4575e-01f, 1.9055e+00f, 

        8.8185e-01f, 1.7376e+00f, -7.7367e-01f, 3.0284e-01f, 3.2207e-01f, -5.0159e-01f, 4.1659e-01f, -1.6233e-01f, 

        -2.5769e-02f, -1.6219e-01f, 5.9623e-01f, 1.2252e+00f, 6.1961e-02f, -7.9507e-01f, -1.5265e-02f, 2.1388e-01f, 

        6.2633e-01f, -9.8908e-03f, 2.4660e-02f, -4.0702e-03f, -1.9451e-01f, -1.6396e-01f, -2.6006e-01f, -5.0259e-01f, 

        1.4604e+00f, 1.1105e+00f, 7.5513e-01f, 4.0671e-02f, 1.1259e+00f, 4.6834e-01f, 3.0595e-01f, -6.6703e-01f, 

        8.2172e-02f, -2.7509e-01f, -2.4972e-01f, 3.7747e-01f, 1.2598e-01f, -2.6630e-01f, 8.2600e-01f, -1.4523e+00f, 

        5.7167e-01f, -6.4362e-01f, 3.1562e-01f, -2.5630e-02f, -3.6813e-01f, -9.0241e-01f, 2.0412e-01f, -9.1346e-01f, 

        1.2249e-01f, -1.5849e-01f, -1.0881e-01f, -1.1794e-01f, -8.5097e-01f, -2.3236e-01f, 7.2368e-01f, -7.6724e-01f, 

        -4.6562e-01f, 8.7866e-01f, 2.7025e-01f, 1.1817e-01f, -6.0145e-02f, -2.6859e-02f, 4.0177e-01f, -7.5816e-01f, 

        1.3194e+00f, -1.3248e-03f, 3.8908e-01f, -9.3486e-01f, -1.0344e+00f, -7.4006e-02f, -7.2594e-01f, -1.8506e-01f, 

        1.1048e-01f, 2.5173e-02f, -5.1118e-01f, -2.6392e-02f, -1.9369e-01f, -1.5728e-01f, 1.0827e-01f, -6.7247e-02f, 

        -5.5769e-02f, -1.0518e-01f, -1.8075e-02f, -1.7733e-01f, 4.8259e-01f, 7.2667e-02f, -9.9769e-01f, 5.9562e-01f, 

        -2.8155e-01f, -3.5972e-02f, 1.0868e+00f, 4.5912e-01f, -2.6586e+00f, -1.8250e+00f, 8.9155e-01f, 2.1801e-01f, 

        -1.3441e+00f, -2.4634e-02f, 1.9273e-01f, -1.6549e+00f, -1.9886e-01f, -4.0616e-01f, -4.1307e-01f, 1.1617e-01f, 

        -3.5415e+00f, -2.0101e-01f, -2.7338e-01f, -2.2176e-01f, -7.4613e-01f, 4.0298e-02f, -1.3890e+00f, -2.0733e-01f, 

        3.3948e-01f, -1.7621e+00f, 2.3315e-01f, 3.1634e-01f, -1.2620e+00f, -1.8683e-01f, 5.2821e-01f, -4.9043e-02f, 

        -1.0553e+00f, 2.8165e-01f, 3.1081e-01f, -2.4784e-01f, -1.2734e+00f, -4.9311e-01f, -7.2282e-01f, 4.1500e-02f, 

        2.4225e-01f, -2.9143e-01f, -6.4828e-01f, 5.7404e-01f, -1.0400e-01f, 3.0335e-01f, 9.4182e-02f, 9.6269e-01f, 

        6.7416e-01f, 1.0896e+00f, -1.0628e+00f, -2.3742e+00f, 4.7334e-01f, -1.4032e-01f, -3.6339e-01f, 5.8693e-01f, 

        -2.8533e+00f, -2.2944e-01f, 3.1790e-01f, -1.0014e+00f, -1.0736e-01f, -3.2847e-01f, 4.8295e-01f, 8.2278e-01f, 

        -1.6793e+00f, 8.8592e-01f, -1.0538e+00f, -4.2405e-01f, 3.5726e-01f, 1.2328e-02f, 2.5163e-01f, 1.2286e-01f, 

        1.7706e-02f, -6.6528e-01f, -4.6746e-01f, -4.6321e-01f, 1.6274e-01f, 8.3380e-01f, 9.6006e-02f, -5.9447e-02f, 

        -6.9759e-02f, 4.5971e-01f, 7.5496e-02f, -5.7978e-01f, -2.5865e-01f, 8.8219e-02f, -7.0591e-01f, -2.1208e+00f, 

        -2.1887e+00f, 7.1456e-01f, -2.8791e+00f, 1.5811e+00f, 1.1682e+00f, 1.6836e-01f, -9.5200e-01f, 6.9335e-01f, 

        -1.0671e-02f, -2.7180e+00f, -7.5346e-02f, -1.1328e+00f, -3.9131e-01f, 2.5810e-01f, 5.3305e-01f, 6.0230e-01f, 

        -4.1931e-01f, -7.5360e-01f, 1.7751e-01f, -4.9387e-01f, -5.5095e-01f, 2.0311e-01f, 4.4697e-01f, -6.7049e-01f, 

        6.1109e-01f, -1.8567e-01f, -1.1811e-01f, -1.1159e-01f, -2.4485e-02f, -6.8000e-02f, -1.6956e-01f, -2.8698e-01f, 

        -1.2122e-01f, -2.5265e-01f, -1.2950e+00f, -1.4923e+00f, -2.5819e-02f, -8.5946e-01f, -1.0791e+00f, -8.5752e-01f, 

        -3.9310e-01f, 5.0451e-01f, -4.0173e-01f, -9.8040e-01f, 1.2446e+00f, 1.5124e+00f, 2.9635e-01f, 3.7572e-02f, 

        -1.1398e+00f, 6.9835e-01f, -9.5751e-01f, -1.8649e-01f, -1.0250e-01f, -2.0564e-01f, -1.1535e-01f, -1.8057e-01f, 

        -1.0798e-01f, -2.2337e-02f, -1.5789e-01f, -1.0025e-01f, -1.1205e-01f, -1.4872e+00f, -3.5018e-01f, -3.1114e-01f, 

        -1.2829e-01f, -1.9604e-01f, -1.5767e-01f, -6.0148e-01f, -5.0650e-01f, -1.8326e-01f, -1.4925e+00f, -7.6067e-01f, 

        -6.7798e-02f, -1.6125e+00f, -2.6580e-01f, -1.8082e+00f, -8.1366e-01f, -1.1033e+00f, -9.9842e-01f, 2.3494e-03f, 

        -6.0876e-02f, -6.7870e-02f, -2.8455e-01f, -2.0241e-01f, -1.5217e-01f, -2.1781e-01f, -2.6326e-01f, -2.4393e-01f, 

        -1.5006e-01f, -1.5612e-01f, -1.1792e-01f, -1.3869e-01f, -2.6129e-02f, -1.7643e-02f, -2.7139e-02f, 1.6184e-02f, 

        -3.8899e-02f, -6.5658e-02f, -1.8932e-02f, -1.0341e-01f, -2.2566e-02f, -1.4019e-01f, -1.3818e-01f, -1.0360e-01f, 

        -1.2413e-01f, -9.0436e-02f, -5.4550e-02f, -8.5454e-02f, -7.7630e-02f, -1.3720e-01f, -1.3471e-01f, -1.4560e-01f, 

        -1.4310e-01f, -1.6237e-01f, -1.4183e-01f, -8.1890e-01f, -1.1663e+00f, -4.2565e-01f, -4.5202e-01f, -3.2860e-01f, 

        -1.2371e-01f, -5.1496e-01f, -1.1308e-01f, -1.5359e-01f, -1.3155e-01f, -5.6810e-02f, -6.4861e-02f, -1.0714e-01f, 

        -1.7330e-02f, -7.1999e-02f, -6.4759e-02f, -4.6836e-02f, -9.8496e-02f, -1.9885e-01f, 1.4096e-02f, -2.1340e-02f, 

        -1.8881e-01f, -9.1079e-02f, -6.4271e-02f, -8.3507e-02f, -1.3022e-01f, -1.4850e-01f, -7.9516e-02f, -1.6988e-01f, 

        -1.0107e-01f, -1.3705e-01f, -8.7741e-02f, -9.5287e-02f, -7.7613e-02f, -2.8780e-02f, -7.6913e-02f, 6.5976e-02f, 

        -1.1690e-01f, -2.4792e-01f, -1.6854e-01f, -1.1013e-01f, -2.1895e-01f, -9.1781e-02f, -1.8537e-01f, -9.0973e-02f, 

        -5.1652e-02f, -5.4823e-02f, -5.7415e-02f, -1.4339e-02f, -2.5604e-02f, -8.6362e-02f, -3.9390e-02f, -4.2436e-02f, 

        -9.6976e-02f, -7.0243e-02f, -9.5532e-02f, -1.7619e-01f, -1.2010e-01f, -1.6228e-01f, -1.6979e-01f, -5.9962e-02f, 

        -7.9675e-02f, -1.3133e-01f, -1.4659e+00f, -9.9620e-01f, -1.3127e+00f, -9.0169e-01f, -9.2805e-01f, -1.1166e+00f, 

        -1.0076e+00f, -8.1091e-01f, -1.1482e+00f, -1.0569e-01f, -1.6225e-01f, -9.3354e-02f, -3.0263e-02f, -7.2858e-02f, 

        -8.8563e-02f, -1.0483e-01f, -8.4416e-02f, -3.6258e-02f, -3.2111e-01f, -2.0858e-01f, -1.3856e-01f, -9.6359e-02f, 

        -1.1115e-01f, -2.6131e-01f, -6.8135e-02f, -1.5286e-01f, -1.3213e-01f, -4.5962e-02f, -1.7910e-01f, -9.7651e-02f, 

        -1.7243e-01f, -7.8747e-02f, -1.0845e-01f, -1.0140e-01f, -1.5336e-01f, -1.0467e-01f, 1.1859e-02f, 3.6401e-02f, 

        8.8092e-02f, -1.6066e-02f, -1.5581e-02f, -2.3355e-02f, -8.6092e-02f, -9.9268e-02f, -4.0092e-02f, -1.3005e-01f, 

        -2.4516e-02f, -7.7084e-02f, -1.7047e-01f, -8.5955e-02f, -9.1263e-02f, -7.5119e-02f, -1.2751e-01f, -9.7524e-02f, 

        -1.9635e-01f, -6.8854e-02f, -1.2771e-01f, -1.6694e-02f, -1.2459e-01f, -4.1412e-02f, -1.7123e-01f, -1.0725e-01f, 

        -9.8759e-02f, -2.2479e-01f, -1.8092e-01f, -1.4925e-01f, -1.2282e-01f, -9.6929e-02f, -1.0204e-01f, -8.5475e-02f, 

        -3.2692e-02f, -1.7782e-01f, -1.4608e-02f, -1.1221e-01f, -7.4717e-02f, -1.1198e-01f, -1.3568e-01f, -7.6492e-02f, 

        -2.1762e-01f, -1.2630e-01f, -1.0111e-01f, -1.1748e-01f, -3.5511e-02f, -1.7783e-01f, -9.6258e-02f, -6.9991e-02f, 

        -1.0520e-01f, -6.3751e-02f, -8.4070e-02f, -1.3898e-01f, -7.6291e-02f, -1.7743e-01f, -1.4216e-01f, -1.5962e-01f, 

        -1.6903e-01f, -1.3647e-01f, -1.8715e-01f, -9.0859e-02f, -1.4180e-01f, -7.6599e-02f, -1.0050e-01f, -1.1024e-01f, 

        -1.4948e-01f, -6.8835e-02f, -7.2714e-02f, -8.3467e-02f, -1.5465e-01f, -1.0430e-01f, -1.5392e-01f, -9.8443e-02f, 

        -1.3332e-01f, -1.3922e-01f, -1.7105e-01f, -1.8883e-01f, -9.4127e-02f, -1.4734e-01f, -1.5265e-01f, -8.4955e-02f, 

        -1.5343e-01f, -9.8646e-02f, -5.8119e-02f, 4.0809e-02f, -7.6688e-02f, -8.3223e-02f, -8.3469e-02f, -8.5673e-02f, 

        -8.1449e-02f, -1.2989e-01f, -1.3562e-01f, -8.1358e-02f, -1.2739e-01f, -1.7005e-01f, -9.9887e-02f, -1.3210e-01f, 

        -1.2968e-01f, -1.3200e-01f, -1.3005e-01f, -1.2573e-01f, -7.7827e-02f, -1.7838e-02f, -1.3523e-01f, -1.1459e-01f, 

        -1.2708e-01f, -1.0944e-01f, -1.5419e-01f, -1.1893e-01f, -9.6880e-02f, -2.0361e-01f, -7.4461e-02f, -1.1799e-01f, 

        -1.3463e-01f, -2.0525e-01f, -1.2439e-01f, -6.5686e-02f, -1.3588e-01f, -1.1743e-01f, -1.4312e-02f, -8.7253e-02f, 

        -1.4229e-01f, -9.8937e-02f, -9.7511e-02f, -1.2227e-01f, -1.1833e-01f, -8.2971e-02f, -7.3061e-02f, 6.3502e-03f, 

        -6.2245e-03f, -7.1706e-02f, -6.4547e-02f, -1.2803e-01f, -7.2050e-02f, 3.1862e-02f, -4.4043e-02f, -1.9085e-01f, 

        -7.3774e-02f, -2.3727e-01f, -2.1182e-01f, -8.8844e-02f, -2.0985e-01f, -2.7472e-01f, -1.4248e-01f, -9.2140e-02f, 

        -1.1728e-01f, -6.3080e-02f, -1.1520e-01f, -7.0718e-02f, -9.6634e-02f, -7.7511e-02f, -7.5759e-02f, -8.6370e-02f, 

        -7.6941e-02f, -8.1438e-02f, -9.7641e-02f, -1.2178e-01f, -4.4104e-02f, -1.0450e-01f, -6.0925e-02f, -2.6499e-02f, 

        3.8251e-01f, -1.6213e-01f, -5.3669e-02f, 1.1575e-02f, -3.0471e-01f, -1.2063e-02f, -4.1254e-01f, -2.7002e-01f, 

        -2.6182e-01f, 6.1412e-02f, -4.9340e-02f, 4.1523e-02f, -8.8540e-02f, -9.5169e-02f, -4.9069e-02f, -5.7107e-02f, 

        -1.6832e-01f, -9.9964e-02f, -2.4367e-01f, -2.0192e+00f, -1.0761e+00f, 1.4644e-01f, -1.8194e-01f, -4.2288e-03f, 

        -7.9836e-01f, -1.8011e+00f, -7.5552e-01f, 2.6792e-01f, -1.2782e-01f, -1.4409e+00f, -7.8533e-01f, -3.9064e-01f, 

        -1.5934e+00f, 1.0056e-01f, -5.2941e-01f, 1.2510e-01f, -3.2144e-03f, -4.0536e-02f, -8.6835e-02f, -2.3393e-01f, 

        -8.2537e-02f, -1.4581e-01f, -4.5516e-03f, -1.4208e-01f, -4.7210e-02f, 1.6337e-01f, 2.6947e-01f, 4.0775e-02f, 

        9.6632e-02f, 2.2965e-01f, -1.2904e-01f, 5.3742e-01f, -6.8870e-01f, -3.7463e-01f, 3.3567e-01f, -3.3950e-01f, 

        -7.6011e-01f, -5.8315e-01f, -3.7767e-01f, -4.2935e-01f, 1.6922e-01f, -3.0553e-01f, -2.7089e-01f, 4.4536e-01f, 

        -5.1197e-01f, -6.0131e-01f, -4.7966e-01f, 1.7303e-02f, -1.4462e+00f, -4.1761e-01f, 4.7137e-02f, -2.0912e-01f, 

        1.0833e+00f, 5.6530e-01f, 7.7297e-01f, -1.1852e-01f, -3.2573e-01f, -7.2077e-03f, 5.2427e-01f, -4.5016e-01f, 

        -2.8105e-01f, -3.4860e-01f, -1.6359e-01f, -7.8786e-01f, -3.1942e-01f, -5.0067e-01f, -2.0282e-01f, 5.2328e-01f, 

        -4.2181e-01f, -2.4202e-02f, -5.9711e-01f, 2.0700e-01f, 7.9485e-02f, -5.0421e-01f, 2.4490e-01f, 3.9437e-01f, 

        -4.7540e-01f, 3.3183e-01f, 4.8177e-01f, 3.4073e-01f, 5.3282e-01f, 2.6694e-01f, -7.1968e-02f, 3.2895e-01f, 

        -1.1206e-01f, -3.2727e-01f, -6.0981e-02f, 1.3566e-01f, -2.8494e-01f, -1.2109e+00f, 5.5619e-01f, 1.0571e-01f, 

        -2.9760e-01f, -1.0233e-01f, -7.3114e-02f, -4.6803e-01f, -5.8970e-02f, -1.0772e+00f, -1.6407e+00f, -1.1621e+00f, 

        -8.9300e-01f, 1.5410e+00f, 4.3199e-01f, -4.8903e-01f, -1.0152e-01f, -6.0361e-01f, -5.1635e-01f, 5.1861e-01f, 

        3.6493e-01f, -3.9784e-01f, -3.4870e-01f, -1.8244e+00f, -1.7122e-01f, -5.5534e-01f, -2.5471e-01f, -2.1682e-01f, 

        -5.9291e-01f, -4.8714e-01f, -2.2708e-01f, 1.1544e-01f, -6.1208e-01f, -5.8054e-01f, -3.2970e-01f, -1.1045e-01f, 

        8.5220e-01f, -1.0956e+00f, 1.3749e-01f, -5.2072e-01f, -3.9570e-01f, -3.8666e-01f, -1.5059e+00f, -1.2913e+00f, 

        -5.7485e-01f, 3.7169e-02f, -2.6852e-01f, 5.6024e-01f, 2.7455e-01f, -3.1694e-01f, -1.2978e-01f, -7.9844e-02f, 

        3.8460e-01f, -8.4074e-01f, -5.7065e-01f, -2.9132e-01f, -1.8411e-01f, 1.7957e-01f, -2.3816e-01f, -2.7714e-01f, 

        -4.1338e-01f, 7.2585e-01f, -3.8912e-01f, -1.1666e+00f, -1.6111e+00f, 4.6910e-03f, -1.5492e+00f, -8.7920e-01f, 

        -1.2763e+00f, -1.6099e+00f, -1.1689e+00f, -5.8896e-01f, 3.2304e-01f, -4.8816e-01f, -6.5772e-01f, -1.0642e+00f, 

        -5.4689e-01f, -5.1975e-01f, -1.4089e+00f, -1.5673e+00f, -9.6220e-03f, -6.8372e-02f, -9.8066e-02f, -1.5104e-01f, 

        -1.7062e-01f, -1.3657e-01f, -1.5012e-01f, 1.2674e-02f, -1.5663e-01f, -3.3370e-02f, -2.4320e-01f, -9.6203e-01f, 

        -1.2223e+00f, 2.4602e-01f, -2.6784e-01f, -1.8634e-01f, -1.2126e+00f, -4.0839e-02f, -1.4885e+00f, -9.5770e-01f, 

        9.5149e-01f, -7.6366e-01f, -1.7473e-01f, -8.0516e-01f, -2.4754e+00f, 5.1139e-01f, -7.5791e-01f, -1.2911e+00f, 

        -1.4664e-01f, -4.6296e-01f, -4.7704e-02f, -2.9168e-01f, -5.7881e-01f, -5.4096e-02f, 7.6715e-02f, -5.0341e-01f, 

        -1.5879e-01f, -1.1110e-01f, -1.7889e-01f, -2.2630e-02f, -1.6895e-01f, -2.2584e-01f, -1.9270e-01f, -9.5551e-02f, 

        -1.5412e-01f, -2.3810e-01f, 3.0491e-02f, -8.4315e-01f, -1.7284e+00f, -1.2948e-01f, -2.4894e-01f, -4.1315e-01f, 

        -4.8779e-01f, 7.3834e-01f, 1.4636e-01f, -2.9825e-01f, -5.2940e-01f, -1.8289e-01f, -1.0261e-01f, -1.7598e-01f, 

        -8.9929e-02f, -4.8773e-01f, -4.8091e-03f, -3.0140e-01f, -1.1068e-02f, 5.5096e-02f, 6.5677e-02f, -3.1751e-02f, 

        -1.2173e-01f, 7.8262e-02f, -2.0018e-01f, -1.5837e-01f, -4.1305e-01f, -2.0927e+00f, -3.4092e-02f, -1.0156e+00f, 

        1.1649e-02f, -9.8932e-01f, 5.8937e-02f, -4.5395e-01f, -1.6490e+00f, -1.3987e-01f, -7.1493e-01f, -3.2900e-01f, 

        -3.5581e-01f, -1.4739e+00f, -3.0618e-01f, -5.1920e-01f, -8.0080e-01f, -1.1396e-04f, -9.4774e-02f, -2.0438e-01f, 

        -2.3052e-01f, -2.1289e-01f, -1.1234e-01f, -2.6205e-01f, -5.2409e-02f, -1.6302e-01f, -2.7102e-01f, 2.2129e-01f, 

        -7.6385e-02f, -9.6738e-02f, 2.7058e-02f, -3.9519e-02f, -6.6811e-02f, -6.9202e-02f, -6.9664e-02f, -1.5461e-01f, 

        -1.5491e-01f, -4.7706e-02f, -6.6916e-02f, -1.3134e-02f, -1.4978e-01f, -5.6954e-02f, -8.4453e-02f, -1.2821e-01f, 

        -1.0832e-01f, -1.6444e-01f, -1.3266e-01f, -8.6452e-02f, -1.9174e-01f, -9.0701e-02f, -4.2745e-02f, -1.4672e-01f, 

        -1.5492e-01f, -1.5663e-01f, -1.4993e-01f, -8.3621e-01f, -1.1367e+00f, -6.0233e-01f, -4.3727e-01f, -3.7107e-01f, 

        -4.2661e-01f, -4.1395e-01f, -3.7861e-01f, -3.4230e-01f, -1.1887e-01f, -1.2505e-01f, -1.0290e-01f, -1.3679e-02f, 

        -1.0421e-01f, -1.3826e-01f, -1.6120e-01f, -1.5039e-01f, -1.0923e-01f, -2.1141e-01f, -1.3837e-01f, -7.5803e-02f, 

        -2.7282e-01f, -9.5751e-02f, -1.3685e-01f, -1.5552e-01f, -1.4568e-01f, -1.0984e-01f, -1.1705e-02f, -3.0284e-02f, 

        -4.0716e-02f, -1.2744e-01f, -8.6644e-02f, -5.0848e-02f, -8.2864e-02f, -1.7951e-02f, -8.4431e-02f, -1.0696e-01f, 

        -7.4767e-02f, -4.2067e-02f, -1.5065e-01f, -1.6129e-01f, -1.3079e-01f, 7.7744e-02f, -1.2672e-01f, -1.4912e-01f, 

        -6.9226e-02f, 1.2357e-02f, -4.7491e-02f, -4.7799e-02f, -6.8203e-02f, -1.3678e-01f, -1.6437e-01f, -1.2602e-01f, 

        -1.3926e-01f, -8.3727e-02f, -8.3211e-02f, -1.5695e-01f, -2.3630e-02f, -7.5650e-02f, -7.4957e-02f, -1.4756e-01f, 

        -7.7042e-02f, -2.0614e-01f, -1.8343e+00f, -1.6253e+00f, -1.3525e+00f, -1.3325e+00f, -9.5489e-01f, -1.2572e+00f, 

        -1.4915e+00f, -8.7831e-01f, -1.1067e+00f, -8.6258e-02f, -1.4383e-01f, -8.0049e-02f, 1.0324e-02f, 5.4131e-04f, 

        -9.3280e-03f, -1.3598e-01f, -1.3535e-01f, -1.0809e-01f, -2.9786e-01f, -1.7308e-01f, -2.0100e-01f, -3.2392e-01f, 

        -1.4990e-01f, -1.4183e-01f, -2.3431e-01f, -2.4410e-01f, -1.9682e-01f, -7.2028e-02f, -1.0821e-01f, -1.2263e-01f, 

        -1.2008e-01f, -1.0185e-01f, -1.2095e-01f, -4.1946e-02f, -1.4986e-01f, -1.5847e-01f, -6.4432e-02f, -5.2014e-02f, 

        -4.8481e-03f, -1.1836e-01f, -3.0456e-02f, -1.3047e-01f, -6.1320e-02f, -3.9783e-02f, 3.4096e-02f, -1.3432e-01f, 

        -1.6576e-01f, -2.0096e-01f, -2.0586e-01f, -1.8933e-01f, -7.3929e-02f, -1.4766e-01f, -5.4068e-02f, -1.8364e-01f, 

        -2.6331e-01f, 7.4723e-03f, -1.4765e-01f, -9.7410e-02f, -9.3397e-02f, -5.6584e-02f, -9.8285e-02f, -9.1549e-02f, 

        -1.2905e-01f, -1.8075e-01f, -1.7029e-01f, -1.1915e-01f, -7.9993e-02f, -1.0538e-01f, -1.7405e-01f, -1.3856e-01f, 

        -1.9737e-01f, -1.3201e-01f, -1.3115e-01f, -1.3213e-01f, -1.3825e-01f, -2.6997e-02f, -6.7670e-02f, -1.2222e-02f, 

        -2.3065e-03f, -2.9591e-02f, -1.2581e-01f, -2.8132e-01f, -1.8605e-01f, 5.8509e-02f, -1.7677e-01f, -3.3357e-03f, 

        -5.3602e-02f, 3.3312e-03f, -1.1100e-01f, 1.5284e-02f, -4.9296e-02f, -1.8981e-01f, -1.4669e-01f, -1.2467e-01f, 

        -6.4623e-02f, -1.6801e-01f, -1.4945e-01f, -7.7231e-02f, -1.3950e-01f, -1.4496e-01f, -1.6655e-01f, -7.7622e-02f, 

        -9.2222e-02f, -1.1424e-01f, -1.3139e-01f, -8.9354e-02f, -4.5591e-02f, -1.2306e-01f, -1.1008e-01f, -7.0843e-02f, 

        -1.1484e-01f, -1.5949e-01f, -1.2730e-01f, -1.1964e-01f, -1.7546e-01f, -1.4748e-01f, -5.7258e-02f, 1.8691e-02f, 

        -1.0832e-01f, 1.6519e-04f, -6.8189e-02f, -8.8481e-02f, -4.1190e-02f, -6.6569e-02f, -7.3647e-02f, -5.0564e-02f, 

        4.6657e-02f, -1.5404e-01f, -8.3382e-02f, -1.0894e-01f, -2.9794e-02f, -1.0400e-01f, -6.7270e-02f, -4.7508e-02f, 

        -2.9819e-02f, -9.2644e-02f, -6.6750e-02f, -1.6239e-02f, -1.0727e-01f, -1.0741e-01f, -4.6163e-02f, -1.5305e-01f, 

        -1.7247e-01f, -1.0597e-01f, -2.1325e-01f, -2.2464e-01f, -1.0988e-01f, -1.2765e-01f, -1.2633e-01f, -7.6798e-02f, 

        -1.0792e-01f, -2.8806e-02f, -1.9597e-01f, -1.2218e-01f, -6.0718e-02f, -3.0873e-02f, -9.6415e-02f, -7.9032e-02f, 

        -5.5858e-02f, -1.5009e-01f, -1.4633e-01f, -4.3605e-02f, -7.3411e-02f, -5.9261e-03f, -1.3479e-01f, -1.0588e-01f, 

        9.7164e-03f, -9.5035e-02f, -8.1841e-02f, -1.7425e-01f, -4.9160e-02f, -1.2856e-01f, -7.1041e-02f, -7.0080e-02f, 

        -2.1077e-01f, -1.6199e-01f, -2.1877e-01f, -4.0064e-03f, -7.8693e-02f, -4.3838e-03f, -6.8491e-02f, -2.7990e-02f, 

        -1.1356e-01f, -6.9028e-02f, -1.6449e-02f, -1.5284e-01f, -1.5042e-01f, -5.9441e-02f, -1.2812e-01f, 1.0750e-01f, 

        3.1196e-02f, -1.4702e-01f, -8.6694e-02f, 2.6510e-03f, -3.2103e-02f, 1.9468e-02f, -1.8131e-02f, -1.0148e-01f, 

        -5.5074e-01f, -7.1203e-01f, -5.2122e-01f, -5.1871e-01f, -9.3284e-01f, -5.8532e-01f, -9.0390e-01f, 4.5969e-01f, 

        4.4939e-01f, -2.3155e-02f, -2.0685e-01f, -1.0360e-01f, -1.1261e-01f, 3.4404e-03f, -1.3455e-01f, -1.4968e-01f, 

        -1.8284e-01f, -1.8638e-01f, -4.8246e-01f, -1.5420e+00f, -7.8084e-01f, -7.8453e-01f, -2.1124e+00f, -9.2238e-01f, 

        -2.1619e+00f, 6.4466e-01f, -2.2814e-02f, -2.2030e-01f, -2.4483e-01f, -1.3406e-01f, -6.7395e-02f, -1.1772e+00f, 

        -2.5918e-01f, -5.1447e-01f, -3.8506e-01f, -8.1106e-01f, 9.2039e-02f, -3.4702e-01f, 6.9167e-01f, -9.8681e-02f, 

        -4.1905e-01f, -4.4696e-01f, -7.2793e-02f, -2.4482e-01f, -5.7009e-01f, -5.6705e-01f, -6.8758e-01f, 6.1831e-01f, 

        -6.1612e-01f, 2.8675e-01f, -5.4778e-01f, 9.4114e-01f, 1.7294e+00f, 1.2881e+00f, -2.5211e-01f, 1.6584e+00f, 

        4.3570e+00f, 8.1712e-01f, 3.4824e-01f, 7.1721e-01f, -3.3585e-01f, 1.2646e-01f, -9.1122e-02f, 7.3131e-03f, 

        1.2453e-01f, -1.3817e-01f, -2.4418e-01f, -6.7495e-01f, -3.7181e-01f, -1.1341e+00f, -1.8397e-01f, -1.0627e+00f, 

        -4.9205e-01f, 3.4938e-01f, -5.7438e-01f, 2.6381e-01f, 5.1298e-01f, -3.0593e-01f, -5.6234e-01f, -5.2239e-01f, 

        -1.0954e+00f, -1.5512e-01f, 8.6776e-01f, 2.5110e+00f, 3.2522e-01f, -3.5452e-01f, 5.9774e-01f, -9.9679e-02f, 

        -6.7578e-01f, -2.4461e-01f, -3.8939e-01f, 3.0494e-01f, -2.0903e-01f, -3.7129e-01f, 5.7469e-01f, -7.0067e-02f, 

        -1.4127e-01f, 1.1694e-01f, -3.5890e-01f, -1.6295e-01f, -1.1381e-01f, -5.5802e-03f, -4.5148e-02f, 1.7140e-02f, 

        -1.3342e-01f, -2.3365e-01f, -1.3848e-02f, -1.9694e-01f, 1.3808e-01f, 4.1681e-01f, -8.1387e-01f, -5.3244e-01f, 

        3.6424e-02f, 2.9643e-01f, -5.5189e-01f, 4.3849e-01f, -3.1646e-01f, 1.4824e-01f, -5.1513e-01f, 1.3283e-01f, 

        5.2293e-02f, -1.2322e+00f, -4.9218e-01f, 7.8882e-01f, -7.2245e-02f, 3.4988e-01f, -4.1404e-01f, 3.9084e-01f, 

        -2.9598e-02f, -2.9927e-01f, 9.7028e-01f, 1.9458e-01f, -2.9373e-01f, 2.9839e-01f, -3.7326e-01f, -6.5823e-01f, 

        -3.7252e-01f, -1.1232e+00f, -4.8801e-01f, -1.0242e-02f, -3.4525e-01f, -4.3445e-01f, 2.4928e-01f, 7.3741e-01f, 

        -1.4469e-01f, -6.6480e-01f, -3.0624e-01f, -8.9402e-01f, -1.2024e+00f, -5.9410e-01f, -3.9648e-01f, -1.6016e+00f, 

        2.5403e-01f, -3.3453e-01f, 2.0572e-01f, -2.8428e-02f, -1.7883e-01f, 2.7865e-01f, 1.6662e-01f, 3.5868e-01f, 

        1.8593e-01f, 1.3563e-01f, 4.5632e-01f, -4.6668e-02f, 2.6643e-01f, -6.9655e-01f, 6.4459e-01f, -2.5463e-01f, 

        -5.8079e-01f, -1.8769e-01f, 2.9781e-02f, -1.3481e-01f, 1.7094e+00f, 1.7206e-01f, 3.5693e-01f, 2.9060e-01f, 

        -3.1206e-01f, 2.5208e-01f, 8.9199e-01f, -3.9578e-01f, -3.0090e-01f, -5.2072e-01f, -5.8606e-01f, 3.2746e-01f, 

        -9.7052e-02f, -2.5197e-01f, 1.8202e+00f, 1.4807e+00f, 6.2970e-01f, 1.5642e-01f, 5.0920e-01f, 8.6055e-01f, 

        -1.3450e-02f, 1.3747e-01f, 5.3946e-02f, 6.0584e-02f, 2.3489e-01f, 7.5848e-02f, 2.5054e-02f, -1.3781e+00f, 

        4.3201e-02f, -9.0015e-02f, 4.1952e-01f, 2.3117e-01f, -4.0926e-01f, 9.2260e-02f, -1.5830e+00f, 6.2535e-01f, 

        -4.9821e-01f, -5.1932e-01f, 1.8078e-02f, -1.4127e+00f, -1.0860e+00f, -5.4667e-01f, -1.1955e+00f, 5.0299e-01f, 

        -3.2983e-01f, -6.6513e-01f, -4.4628e-02f, -2.9278e-02f, 1.5036e-01f, 1.6746e-01f, -6.1240e-01f, -5.4378e-01f, 

        -6.8760e-01f, -1.1561e-01f, -1.2546e-01f, -2.8249e-02f, -1.7360e-01f, -7.0010e-02f, -1.1323e-01f, -8.4082e-02f, 

        -3.7024e-02f, -1.9190e-01f, 5.4871e-01f, -3.1055e-01f, -1.0922e+00f, -8.1533e-01f, -1.1113e-01f, 7.1766e-02f, 

        -1.1252e+00f, -8.4770e-01f, -3.2382e-01f, -4.6162e-01f, -1.8026e-01f, -5.4745e-01f, 8.1179e-01f, 9.9281e-01f, 

        4.3119e-02f, 1.2100e+00f, 5.5957e-01f, -6.0895e-01f, -2.1044e-01f, -2.2874e-01f, -2.6771e-01f, -2.2494e-01f, 

        -1.8217e-01f, -1.5827e-01f, -1.9215e-01f, -2.2111e-01f, -1.7956e-01f, 7.9376e-01f, 1.2555e+00f, -3.1965e-01f, 

        -4.0725e-02f, -2.7055e-01f, 4.7466e-01f, 3.1870e-01f, -9.8968e-01f, -4.3477e-01f, -1.7755e+00f, -1.0025e+00f, 

        -1.1682e+00f, -1.1346e+00f, 8.5007e-02f, -8.7236e-01f, 2.7520e-02f, -7.8289e-01f, -9.3581e-01f, -1.8079e-01f, 

        -2.0148e-01f, -2.5774e-01f, -2.2063e-01f, -1.4398e-01f, -2.1053e-01f, -3.2415e-01f, -3.6802e-01f, -6.8269e-02f, 

        -8.0799e-02f, 4.2242e-01f, 7.1751e-02f, -2.7553e-01f, 1.6449e-01f, -3.1079e-02f, -5.5593e-02f, -2.3748e-02f, 

        1.3653e-02f, 1.5927e-01f, 1.2080e-01f, 3.1612e-02f, 1.7180e-01f, 1.0201e-01f, 7.7083e-02f, 6.7886e-02f, 

        1.3214e-01f, 1.3504e-01f, 1.1306e-01f, 4.4427e-01f, -1.3827e-01f, -2.0831e-01f, 1.5037e-01f, 2.4726e-01f, 

        2.1840e-01f, 2.9946e-01f, -2.1707e-01f, -3.5741e-01f, -1.0377e+00f, -8.4802e-01f, -7.2882e-01f, -6.2278e-01f, 

        2.3589e-01f, -7.3228e-01f, 3.1879e-01f, 6.8998e-01f, 1.1501e-01f, 9.1173e-03f, -2.1789e-02f, 3.3157e-02f, 

        1.0025e-01f, -7.6716e-03f, 8.6911e-02f, 2.4857e-02f, -1.3007e-01f, -9.3689e-02f, 2.8027e-01f, -3.8900e-01f, 

        -7.5583e-01f, 2.5055e-01f, -2.1427e-01f, -3.1415e-01f, -9.6925e-01f, 5.5524e-01f, -2.4030e-03f, -4.1371e-02f, 

        -6.8381e-02f, -3.8579e-01f, -5.0859e-02f, -3.7582e-01f, -1.8020e-01f, 3.7730e-01f, 4.3978e-02f, 2.3036e-01f, 

        1.1793e-01f, -1.4496e-01f, 1.5136e-01f, -1.8485e-01f, -8.1379e-03f, -2.2364e-01f, -2.1684e-01f, 1.3288e-01f, 

        -8.2247e-01f, 1.0961e-01f, -3.7963e-02f, -6.1553e-01f, 3.2147e-01f, -2.2817e-01f, 2.3724e-01f, -3.6289e-01f, 

        -5.7480e-02f, 1.4989e-01f, 1.5185e-01f, 1.3366e-02f, 4.5905e-02f, -1.1665e-01f, -4.3703e-02f, 1.9317e-01f, 

        4.0849e-02f, -7.7661e-02f, 6.1956e-01f, -7.2783e-02f, -2.5618e-01f, 5.5060e-01f, -4.1535e-01f, -3.0346e-01f, 

        -1.5410e-01f, -3.5421e-01f, -1.2548e-01f, 1.7382e-01f, 5.0357e-02f, 4.0557e-02f, -6.8302e-02f, -8.4589e-02f, 

        2.3659e-02f, -7.6017e-02f, -6.9681e-03f, -5.3864e-02f, -6.2759e-01f, 5.1822e-01f, -3.2691e-01f, 8.5751e-02f, 

        3.1027e-01f, -7.5104e-01f, 2.9375e-01f, -5.4956e-01f, -2.2113e+00f, -1.0332e+00f, -6.7741e-01f, -2.5404e-01f, 

        -7.3702e-02f, -5.8761e-01f, -6.3989e-01f, -2.8413e-01f, 4.0008e-02f, -3.2594e-01f, -1.3559e-01f, 9.3034e-01f, 

        -5.4891e-01f, 1.5908e-01f, -4.1781e-01f, -2.8026e-01f, -2.0337e-01f, -4.2944e-02f, -3.6004e-01f, 1.2410e-01f, 

        1.9230e-01f, 7.4684e-02f, 4.5542e-01f, 1.1820e-01f, 6.5186e-02f, 2.3635e-01f, 1.4779e-01f, 2.9347e-01f, 

        -5.4501e-01f, -9.3357e-01f, -1.0638e+00f, -1.2166e+00f, 5.7481e-01f, -8.7814e-01f, -9.0147e-01f, 6.9549e-02f, 

        -8.0483e-01f, 2.9889e-01f, -2.3062e-01f, 1.3810e+00f, -1.1267e-01f, 2.0449e-01f, 6.7685e-01f, -4.9960e-01f, 

        4.2339e-01f, 6.3999e-01f, -5.5515e-01f, -7.0001e-01f, 7.9604e-02f, -1.0506e+00f, -4.6291e-01f, 4.9769e-01f, 

        -3.7893e-01f, 1.4424e-01f, -1.4031e-02f, -1.3025e-01f, -2.2756e-01f, -2.6971e-01f, -8.0983e-01f, -1.1335e+00f, 

        -6.9164e-01f, -4.5544e-01f, -5.9470e-02f, -4.0154e-01f, -3.0763e-01f, -2.6521e-02f, -3.9342e-01f, -2.8698e-01f, 

        2.4152e-02f, -2.2285e-01f, -1.3605e-01f, -8.5440e-02f, -9.1071e-01f, 6.4197e-02f, 2.3697e-01f, 5.3099e-02f, 

        1.4844e-01f, 3.8530e-02f, -5.9799e-02f, 7.4717e-02f, 7.2167e-02f, 3.4497e-03f, -2.6121e-01f, -8.0952e-01f, 

        -9.5347e-01f, -2.4602e-01f, 1.9289e-01f, -1.7360e-01f, -3.8987e-01f, -6.1765e-02f, -9.5044e-01f, -2.1744e-01f, 

        -1.5765e-01f, -7.2307e-01f, -1.6605e-01f, 4.2814e-01f, 6.6453e-01f, -2.0224e-01f, 3.2899e-01f, -4.9524e-01f, 

        9.8947e-01f, -5.9503e-01f, 1.0001e+00f, 3.8063e-02f, 5.3830e-01f, 8.7968e-02f, -8.9258e-01f, -1.2553e+00f, 

        7.7750e-02f, 5.0290e-02f, 8.7066e-02f, 5.6087e-03f, -1.5383e-03f, 9.1432e-02f, 1.1697e-01f, 1.5626e-01f, 

        3.5533e-02f, 8.4283e-02f, 4.9240e-01f, -4.1809e-01f, -9.7747e-01f, 1.6706e-01f, -7.5219e-01f, -1.1989e+00f, 

        -2.2949e-01f, -9.9241e-01f, -1.7676e+00f, 2.0120e-01f, 1.0058e-01f, 1.1240e-02f, -2.8387e-01f, 1.8036e-01f, 

        -1.3427e-01f, -2.5610e-01f, 2.0070e-01f, 1.7246e-02f, -1.7865e-02f, -5.8164e-02f, -4.5288e-02f, 1.5078e-02f, 

        4.0771e-02f, 5.1928e-03f, -2.0358e-02f, 3.0149e-02f, -7.8198e-02f, -1.0308e+00f, 2.7062e-02f, -2.3550e-01f, 

        -6.0299e-01f, -5.5795e-02f, -2.5123e-01f, -3.8659e-01f, -1.0340e-01f, 1.7072e-01f, 8.5706e-01f, 1.1679e+00f, 

        2.3610e-01f, 3.4777e-01f, 6.3422e-02f, -4.8827e-01f, -3.6116e-01f, 4.2985e-01f, 3.1156e-02f, -6.4341e-02f, 

        -2.2733e-01f, -8.6329e-02f, -6.5909e-02f, -1.2439e-01f, -1.5666e-01f, -4.8000e-02f, -1.9244e-01f, -1.6271e-01f, 

        2.4622e-01f, 1.6059e-01f, -1.8293e-02f, -4.9873e-02f, -4.3786e-04f, -4.5684e-02f, 2.4777e-01f, -3.7720e-02f, 

        -1.0341e-01f, 4.9721e-02f, 3.4939e-02f, -1.1904e-02f, 2.6104e-03f, -7.3664e-02f, -1.5137e-01f, 2.3811e-03f, 

        -5.3998e-02f, -7.8064e-02f, 5.4308e-01f, 1.2072e-01f, -3.1021e-01f, 3.7713e-02f, -2.0830e-01f, 1.0105e-01f, 

        -1.1764e-02f, 2.8941e-01f, 5.3439e-02f, -2.4421e+00f, -6.4292e-01f, 1.0040e+00f, -8.3385e-01f, 1.1895e+00f, 

        -5.4047e-01f, 1.3646e+00f, 5.1494e-01f, -6.2654e-01f, -6.2391e-02f, 1.1595e-01f, -5.1959e-02f, -2.2262e-01f, 

        -1.0612e-01f, 1.4350e-02f, 5.6440e-02f, -2.3749e-02f, -6.9024e-02f, 3.6150e-01f, 3.7692e-01f, 1.1272e-01f, 

        -8.0914e-02f, -3.8687e-02f, -5.1939e-01f, -1.7211e-01f, -6.9292e-01f, -2.8750e-01f, -6.5527e-01f, 1.8282e-01f, 

        -2.1651e-01f, -2.3321e-02f, -1.3758e-01f, -1.2096e-01f, -1.5109e-01f, -3.2837e-01f, -3.3594e-01f, 6.9687e-01f, 

        -2.3148e-01f, 2.1228e-01f, -3.0477e-01f, -9.8609e-02f, -1.3044e-01f, 8.3176e-01f, -2.9565e-01f, -5.2302e-04f, 

        -6.5970e-02f, -6.4061e-02f, 2.0600e-01f, 6.6785e-02f, 1.0019e-01f, -2.9772e-01f, -2.3358e-02f, 1.0687e-02f, 

        -1.0842e-01f, -1.5397e-01f, -1.5844e-01f, -1.7190e-01f, -3.0488e-01f, -1.0660e-01f, -8.7692e-02f, -1.3883e-01f, 

        -2.9043e-01f, -2.0340e-01f, -6.2982e-02f, -1.4642e-01f, -3.2005e-02f, -3.2896e-01f, -4.6370e-01f, -2.8742e-01f, 

        -7.0084e-01f, -7.6275e-01f, -2.5932e-02f, -1.1466e-01f, -6.3973e-03f, 7.6055e-02f, -8.5337e-02f, -1.0589e-01f, 

        1.0701e-01f, 5.7502e-02f, -6.1942e-02f, -1.6072e-02f, 5.1083e-01f, -6.1679e-01f, -5.4068e-01f, 8.8093e-01f, 

        -5.6935e-01f, -4.7862e-01f, -6.6968e-01f, 1.0754e-01f, -4.3638e-01f, -2.4231e-01f, -4.5108e-01f, -1.0536e-01f, 

        4.0358e-01f, -6.1411e-02f, -1.8618e-01f, -1.2637e-01f, -2.0553e-02f, -1.0224e-01f, -1.4810e-02f, -2.5786e-02f, 

        -2.8481e-01f, -2.2229e-01f, -4.6059e-02f, -4.3742e-01f, -1.1574e-01f, -1.1480e-01f, -4.7094e-01f, 1.1450e+00f, 

        -7.5140e-02f, -2.7601e-01f, 2.2510e-01f, 1.8315e-02f, -1.7711e-02f, 2.9854e-02f, -1.0772e-01f, -2.0097e-01f, 

        -1.1439e+00f, -3.4594e-01f, 3.3247e-02f, 3.7783e-02f, -6.2992e-01f, -1.7800e-01f, -1.5991e-02f, 2.5682e-02f, 

        -6.6087e-01f, -2.2932e-01f, 2.0466e-02f, -7.8394e-02f, -4.3097e-02f, 1.6109e-01f, -4.9676e-01f, 4.5952e-01f, 

        -1.8710e-02f, 1.5392e-01f, -1.8050e+00f, -1.4138e-01f, -1.9702e-02f, -1.4525e-01f, 1.4483e-01f, -4.7629e-01f, 

        5.3860e-03f, -7.4253e-02f, -1.4119e-02f, 8.8298e-01f, 4.3451e-01f, 9.1735e-01f, 7.1864e-01f, 2.6862e-01f, 

        8.2893e-02f, 9.6967e-01f, -1.3749e-01f, -2.0195e-01f, -1.6252e-01f, -1.3810e-01f, -1.3486e-02f, 1.5640e-01f, 

        -8.5844e-02f, -1.9847e-01f, -7.8373e-03f, -1.3535e-01f, -3.8404e-01f, 2.5248e-02f, 5.1610e-02f, -5.7392e-02f, 

        6.0850e-03f, -3.6086e-02f, -2.6128e-02f, -1.6425e-01f, -1.4410e-01f, -6.3222e-02f, 1.5351e-01f, -2.7018e-01f, 

        1.3046e-01f, -7.5439e-01f, -2.6585e-02f, -2.3062e-01f, -6.9928e-01f, 3.1853e-01f, -1.7314e-01f, -4.0499e-01f, 

        6.1820e-01f, 8.4935e-01f, 6.4028e-01f, 4.8524e-01f, -3.1614e-01f, 2.9472e-01f, -2.1793e-01f, 1.5843e-01f, 

        -1.6066e-02f, -7.3823e-01f, -5.7897e-01f, -1.3830e-01f, -4.5043e-02f, -5.6578e-02f, -1.7232e-01f, -4.6709e-01f, 

        1.2822e-01f, -1.6223e-02f, 1.5667e-02f, -5.1763e-02f, 5.0117e-02f, -2.1002e-02f, -3.9258e-02f, 5.9292e-02f, 

        -6.8276e-02f, -8.8301e-02f, -2.0934e-01f, 1.5125e-01f, -2.2652e-02f, 1.2579e-01f, -1.6093e-01f, 9.3326e-02f, 

        1.5582e-01f, 3.5353e-01f, 1.5357e-01f, 2.0056e-01f, 3.8445e-01f, 3.6300e-03f, 5.5063e-01f, -1.4054e-01f, 

        2.2156e-02f, 2.3426e-01f, -1.1466e-01f, -1.0540e-01f, -5.0785e-02f, 4.7179e-02f, 9.9322e-03f, -5.6059e-02f, 

        5.2525e-02f, 2.8934e-02f, -4.0440e-02f, 8.0563e-02f, -4.6378e-02f, 3.1602e-01f, 5.0880e-01f, -7.7898e-01f, 

        -2.7551e-01f, 1.3356e-01f, 4.4840e-01f, -5.5228e-01f, 4.4704e-01f, 3.9871e-01f, 9.0200e-02f, -2.4969e-02f, 

        3.1784e-02f, 1.6140e-01f, 1.5402e-01f, -3.3194e-01f, -2.5904e-01f, 1.3694e-01f, 1.5866e-02f, -1.7033e-01f, 

        -1.8368e-01f, -1.6675e-01f, -6.2956e-02f, -1.2078e-01f, -3.6018e-02f, -1.4751e-01f, -5.2721e-02f, 4.5343e-02f, 

        -1.1725e-01f, -1.6631e-01f, -1.7357e-01f, -7.3587e-02f, -1.2666e-01f, -4.6963e-02f, -8.7795e-02f, -9.7887e-02f, 

        -1.1059e-01f, -1.2570e-01f, -1.2727e-01f, -1.5760e-01f, -8.4422e-02f, -4.6652e-02f, -4.3643e-03f, -2.0324e-02f, 

        -9.1126e-02f, -5.3916e-02f, -1.5919e-01f, -6.9161e-02f, -8.0552e-02f, -3.0962e-02f, -9.0654e-02f, -4.6304e-02f, 

        -6.2935e-02f, -3.7088e-02f, -1.1332e-01f, -1.0192e+00f, -1.4730e+00f, -1.1466e+00f, -3.5407e-01f, -4.6518e-01f, 

        -4.4338e-01f, -4.3804e-01f, -2.9575e-01f, -4.0785e-01f, -6.9614e-02f, -1.1636e-01f, -1.3040e-01f, -1.0307e-01f, 

        -1.2347e-01f, -9.7681e-02f, -7.8533e-02f, -1.2957e-01f, -4.1563e-02f, -2.9793e-01f, -2.5061e-01f, -9.7189e-02f, 

        -2.5964e-01f, -1.1165e-01f, -1.7161e-01f, -8.4616e-02f, -1.0152e-01f, 3.3352e-03f, -2.3240e-01f, -1.8857e-01f, 

        -1.3011e-01f, -9.9638e-02f, -1.2660e-01f, -1.6814e-01f, -5.9067e-02f, -1.7975e-01f, -1.2401e-01f, -1.4386e-01f, 

        -2.6631e-01f, -2.2406e-01f, -1.5043e-01f, -1.3650e-01f, -1.7673e-01f, 8.2013e-02f, -1.3584e-01f, -1.3872e-01f, 

        -2.6528e-01f, -2.0399e-01f, -2.1522e-01f, -1.8190e-01f, -1.7882e-01f, -2.7856e-01f, -9.1230e-02f, -3.6872e-02f, 

        -9.6140e-02f, -1.1408e-01f, -3.2191e-02f, -7.9004e-02f, -7.9790e-02f, -3.6829e-03f, -6.4928e-02f, 1.1205e-02f, 

        -3.7424e-02f, -5.2702e-02f, -1.9977e+00f, -1.3754e+00f, -1.1853e+00f, -1.2090e+00f, -7.4030e-01f, -1.2475e+00f, 

        -9.2184e-01f, -6.2605e-01f, -5.0605e-01f, -7.5072e-02f, -7.1135e-02f, 4.1639e-02f, -6.7011e-02f, 3.1398e-02f, 

        -6.8000e-02f, 8.4518e-03f, -8.5123e-02f, -6.8668e-02f, -2.7981e-01f, -1.7667e-01f, -2.1036e-01f, -2.5162e-01f, 

        -8.7589e-02f, -1.1980e-01f, -1.0961e-01f, -1.4788e-01f, -8.1879e-02f, -8.0288e-02f, -4.2402e-02f, -5.8663e-02f, 

        -1.1778e-01f, -5.8521e-02f, -1.0378e-01f, -9.6518e-02f, -1.0500e-01f, -1.0821e-01f, -1.1710e-01f, -1.1847e-01f, 

        -1.4666e-01f, -1.4102e-01f, 4.6027e-03f, -2.1975e-02f, -8.2892e-02f, 4.6189e-02f, -6.7757e-02f, -1.8322e-01f, 

        -1.2874e-01f, -1.3133e-01f, -1.0450e-01f, -1.4560e-01f, -5.5027e-02f, -4.9756e-02f, 4.3150e-03f, -1.3764e-01f, 

        -1.9790e-01f, -1.2907e-01f, -9.6852e-02f, -1.2195e-01f, -1.6100e-01f, -1.2576e-01f, -8.4206e-02f, -1.3575e-01f, 

        -1.1374e-01f, -2.0349e-01f, -1.9806e-01f, -2.5872e-01f, -1.0420e-01f, -2.2187e-01f, -2.0855e-01f, -1.0945e-01f, 

        -1.9002e-01f, -9.6080e-02f, -2.4643e-01f, -2.1221e-01f, -8.8868e-02f, -1.4740e-01f, -9.6157e-02f, -1.5029e-01f, 

        -3.2932e-02f, -9.2820e-02f, -9.0328e-02f, -3.1996e-01f, -3.4935e-01f, -3.3088e-01f, -1.6280e-01f, -1.6602e-01f, 

        -1.5322e-01f, -2.9786e-02f, 3.1003e-02f, -2.4196e-01f, -1.3589e-01f, -1.1359e-01f, -2.0806e-02f, -1.3555e-01f, 

        -3.0677e-02f, -8.7096e-02f, -3.1681e-02f, -2.8921e-02f, -2.1527e-02f, -7.0715e-02f, -1.2481e-01f, -1.3227e-01f, 

        -9.0645e-02f, -4.2919e-02f, -8.5389e-02f, -1.0979e-01f, -7.5780e-02f, -1.2512e-01f, -8.4137e-02f, -1.7776e-01f, 

        -5.8298e-02f, -1.5365e-01f, -1.4753e-01f, -1.6275e-01f, -1.0759e-01f, -1.6031e-01f, -1.0398e-01f, -2.3542e-01f, 

        -1.0845e-01f, -1.3830e-01f, -1.8868e-02f, -1.3496e-01f, -1.6773e-01f, -7.6946e-02f, -1.0142e-01f, -5.1240e-02f, 

        -6.2476e-02f, -1.6232e-01f, -7.0538e-02f, -1.0097e-01f, -1.1621e-01f, -9.8823e-02f, -1.4614e-01f, -1.2724e-01f, 

        -1.2965e-01f, -1.4927e-01f, -9.3388e-02f, -4.9394e-02f, -1.3693e-01f, -1.0929e-01f, -1.2814e-01f, -7.5370e-02f, 

        -1.3463e-01f, -1.1706e-01f, -1.9278e-01f, -8.5500e-02f, -9.7428e-02f, -1.4447e-01f, -8.1283e-02f, -3.5260e-02f, 

        -5.2961e-02f, -4.9890e-02f, -1.0338e-01f, -1.0879e-01f, -8.1515e-02f, -6.9482e-02f, -4.5350e-02f, -1.0998e-01f, 

        -4.8644e-02f, -5.0036e-02f, -1.8531e-02f, -2.6631e-02f, -6.8696e-02f, -8.1835e-02f, -4.1040e-02f, -1.3794e-01f, 

        -5.4072e-02f, -9.8823e-02f, -1.0765e-01f, -7.4876e-03f, -5.4538e-02f, -1.2404e-01f, -1.7884e-01f, -2.3597e-01f, 

        -1.2555e-01f, -4.3239e-02f, -1.0185e-01f, 3.9893e-02f, 2.1595e-02f, -1.6648e-01f, -1.2390e-01f, -1.5748e-01f, 

        -6.2174e-02f, -1.3280e-01f, -1.2725e-01f, -8.8861e-02f, -4.6408e-02f, -1.2119e-01f, -5.6460e-02f, -2.1363e-01f, 

        -6.1058e-02f, -1.0703e-01f, -1.0631e-01f, -2.1345e-01f, -1.0374e-01f, -1.1633e-01f, -9.9257e-02f, -1.5756e-01f, 

        -2.7128e-01f, 2.7904e-01f, -9.4751e-01f, 4.6947e-02f, 7.5091e-02f, 1.0591e+00f, -1.0976e-01f, 1.0566e-01f, 

        -1.9357e-01f, -8.5540e-02f, -1.5361e-01f, -2.1469e-01f, -1.7171e-01f, -1.7025e-02f, -1.2509e-01f, 2.3875e-02f, 

        -7.2927e-03f, -4.1781e-02f, -8.6149e-02f, 1.0259e+00f, 2.0730e-01f, -6.8525e-01f, 7.6835e-01f, -2.0834e+00f, 

        -7.7189e-01f, 7.7937e-01f, -8.3245e-01f, 2.6577e-01f, 7.1146e-01f, -3.8466e-01f, 3.7361e-02f, 3.3101e-01f, 

        3.1864e-02f, 1.2974e+00f, 1.9566e-01f, -2.6993e-01f, 1.2165e+00f, -5.8170e-01f, -6.3040e-01f, -5.5127e-01f, 

        6.4277e-01f, 2.8832e-01f, -1.9881e-01f, -1.3215e-01f, -1.0155e+00f, -1.1868e+00f, 1.2761e+00f, -4.3690e-01f, 

        -6.8309e-01f, -1.6199e+00f, 9.3274e-02f, 7.2567e-01f, 3.3945e-01f, 5.2555e-01f, 2.0881e+00f, -2.9388e+00f, 

        -4.7816e-01f, -5.0963e-01f, 2.3519e+00f, 6.6490e-01f, -3.5946e-01f, 1.2246e-01f, -4.9098e-01f, 7.0953e-02f, 

        1.0151e+00f, -2.7436e-01f, 6.0682e-01f, -4.9068e-01f, 1.6201e+00f, 5.3099e-01f, 9.2632e-01f, 2.8567e-01f, 

        -9.8046e-02f, -1.0944e+00f, -8.0815e-01f, -4.9885e-02f, -5.4876e-01f, 9.3996e-01f, 1.7849e-01f, -1.2407e-01f, 

        9.0988e-02f, 7.7388e-01f, -1.8095e+00f, -6.7723e-01f, -8.9455e-01f, 8.9849e-01f, 1.4066e-01f, -6.1994e-01f, 

        -6.5601e-01f, -1.1966e+00f, 5.6606e-01f, -1.0217e-02f, 1.0472e-01f, -3.9996e-01f, 3.9848e-01f, -1.4636e-01f, 

        -1.6214e+00f, -6.5505e-01f, -4.6143e-01f, 2.5315e-01f, -1.1345e-01f, -2.4995e-01f, -2.1674e-01f, -5.0852e-01f, 

        -1.4005e-01f, 2.9121e-01f, -2.3823e-01f, -6.5234e-02f, 1.0559e-01f, -2.4039e-01f, -8.4263e-01f, 4.0477e-01f, 

        4.6351e-01f, -6.7702e-01f, -1.4057e+00f, -6.7559e-01f, -2.0551e-01f, 1.0703e-01f, -1.5900e+00f, 1.6669e+00f, 

        -8.1615e-01f, 3.9259e-01f, -2.2318e-01f, 8.0144e-01f, -3.9729e+00f, -4.6220e-01f, -7.1906e-01f, -1.2261e+00f, 

        -5.4233e-01f, 7.6141e-01f, 3.0643e-03f, -8.8040e-01f, -1.2693e+00f, 1.8549e+00f, -3.5788e-01f, -7.3782e-01f, 

        1.6201e-01f, -6.0019e-01f, 4.1710e-01f, -2.4861e-01f, -2.4073e-01f, -8.3620e-02f, -3.6993e-01f, -2.3879e-01f, 

        -1.6446e-01f, 7.5017e-01f, 3.7934e-01f, -7.0021e-01f, 4.3355e-01f, 2.7292e-01f, 1.1832e+00f, -7.4715e-02f, 

        2.9140e-01f, 1.1530e+00f, 4.8256e-01f, 5.5339e-01f, 6.9247e-01f, 2.6764e-01f, 5.8495e-01f, 4.0777e-01f, 

        5.1441e-01f, 9.3043e-01f, -1.2719e+00f, -1.0742e+00f, -6.8505e-01f, -9.6574e-02f, -1.4294e+00f, -3.8273e-02f, 

        2.2928e-02f, 9.5288e-02f, 6.7512e-02f, 5.2921e-01f, 6.0010e-01f, 8.6214e-01f, 1.1343e+00f, -3.9193e-01f, 

        1.8767e+00f, -1.5347e+00f, -9.9868e-01f, -1.8840e-01f, -1.1502e+00f, 1.2985e-02f, -1.1438e+00f, -6.4280e-01f, 

        1.7345e-01f, 4.9818e-01f, 1.4934e-01f, -8.1326e-02f, -1.5424e-01f, 8.1302e-01f, -7.7732e-01f, -5.6975e-01f, 

        -3.1293e-01f, 6.2181e-01f, -3.3559e-03f, -3.4582e-01f, -2.0643e-01f, -7.4764e-02f, 3.5558e-01f, -8.8165e-01f, 

        1.2523e+00f, -1.1492e+00f, 2.8546e-01f, -1.3462e+00f, -5.1737e-01f, 4.1162e-01f, -1.4434e+00f, -7.6276e-01f, 

        -9.9849e-01f, 1.2669e+00f, -1.2891e+00f, 7.2452e-01f, -6.8361e-01f, 1.3076e+00f, -7.1470e-01f, 9.9168e-02f, 

        -2.0844e-01f, -1.5809e-02f, 5.8042e-02f, -1.2868e+00f, -9.8690e-01f, -8.2974e-01f, -1.0134e+00f, -3.3229e-02f, 

        -5.0483e-01f, -2.9491e-01f, -3.4214e-01f, -7.9262e-03f, -3.2475e-01f, -3.5759e-01f, -1.9262e-01f, -7.5507e-02f, 

        -3.9780e-02f, -1.4627e-02f, -3.9471e-01f, 6.8083e-01f, 1.4904e-01f, 3.7848e-01f, -1.3887e+00f, -1.8340e+00f, 

        -4.2463e-01f, -2.5294e+00f, 9.0800e-01f, -7.2454e-01f, -8.8086e-01f, -3.6151e-01f, -1.8153e-01f, -8.0771e-01f, 

        8.2272e-01f, -9.9613e-01f, -1.6504e+00f, 3.1321e-01f, -8.3766e-02f, -8.1786e-02f, -4.4086e-02f, 1.1671e-02f, 

        3.2616e-04f, 1.3103e-02f, -4.9881e-02f, 1.1087e-01f, -6.2404e-02f, 1.1829e+00f, 3.9008e-01f, -5.2924e-01f, 

        -9.0381e-01f, -7.9048e-01f, 6.7950e-01f, -8.9165e-01f, -9.8783e-01f, 3.6546e-01f, -9.8691e-01f, -3.5293e-01f, 

        -6.6840e-01f, 2.8893e-01f, 1.0577e+00f, -7.3891e-01f, -1.1468e-01f, -7.2334e-02f, 8.0765e-01f, 6.6761e-03f, 

        -1.3867e-01f, -3.4716e-02f, 3.1888e-01f, 4.2496e-02f, 5.1195e-02f, 2.0567e-01f, 2.8400e-01f, 2.3340e-01f, 

        -1.8751e-01f, -2.4544e-01f, -1.8041e-01f, -1.8929e-01f, -1.8783e-01f, -1.7745e-01f, -1.8450e-01f, -1.9377e-01f, 

        -1.1444e-01f, -1.1584e-01f, -1.8584e-01f, -1.8748e-01f, -1.5473e-01f, -1.0315e-01f, -7.1204e-02f, -1.4508e-01f, 

        -9.1022e-02f, -2.0906e-01f, -1.6910e-01f, -1.3884e-01f, -2.3936e-01f, -2.1829e-01f, -1.1668e-01f, -1.0873e-01f, 

        -2.4133e-01f, -1.6325e-01f, -1.1981e-01f, -1.4362e+00f, -2.1111e+00f, -9.6069e-01f, -8.2912e-01f, -4.2484e-01f, 

        -5.7691e-01f, -7.3136e-01f, -3.6271e-01f, -6.0872e-01f, -1.9178e-01f, -8.2476e-02f, -1.4068e-01f, -1.0919e-01f, 

        -1.2286e-01f, -1.7477e-01f, -1.5271e-01f, -8.4068e-02f, -1.5216e-01f, -3.6648e-01f, -1.4040e-01f, -2.2900e-01f, 

        -2.7549e-01f, -2.7140e-01f, -2.4413e-01f, -3.4435e-01f, -2.4261e-01f, -2.3233e-01f, -1.4459e-01f, -2.1761e-01f, 

        -1.1168e-01f, -1.5028e-01f, -2.0972e-01f, -1.4511e-01f, -2.2045e-01f, -1.1855e-01f, -9.2016e-02f, -9.5746e-02f, 

        -3.5697e-01f, -1.8922e-01f, -1.5705e-01f, -1.4075e-01f, -2.2497e-01f, -2.2474e-01f, -1.2101e-01f, -2.0592e-01f, 

        -2.9773e-01f, -1.6514e-01f, -1.6621e-01f, -1.3546e-01f, -1.1656e-01f, -2.6168e-01f, -1.2781e-01f, -1.5184e-01f, 

        -1.3297e-01f, -1.3863e-01f, -1.4086e-01f, -1.9651e-01f, -1.7295e-01f, -2.1966e-01f, -1.1447e-01f, -7.0497e-02f, 

        -1.2848e-01f, -1.1347e-01f, -2.3309e+00f, -1.2635e+00f, -1.5311e+00f, -1.9277e+00f, -1.0056e+00f, -1.6391e+00f, 

        -1.9185e+00f, -8.0967e-01f, -1.7312e+00f, -3.9648e-02f, -1.2688e-01f, -3.4633e-02f, -1.4410e-01f, -1.2216e-01f, 

        -6.5213e-02f, -1.0031e-01f, -9.5398e-02f, -1.3126e-01f, -4.8450e-01f, -2.7988e-01f, -1.4959e-01f, -2.0483e-01f, 

        -1.7384e-01f, -1.7914e-01f, -1.9635e-01f, -1.9964e-01f, -2.0274e-01f, -1.8608e-01f, -1.6762e-01f, -1.4108e-01f, 

        -1.4282e-01f, -2.0443e-01f, -8.7044e-02f, -1.4800e-01f, -1.7714e-01f, -9.6040e-02f, -4.1929e-02f, -6.7212e-02f, 

        -1.0942e-01f, -9.6209e-02f, -4.0591e-02f, -1.1172e-01f, -6.9227e-02f, -9.6739e-02f, -9.6202e-02f, -2.2383e-01f, 

        -1.6005e-01f, -1.9545e-01f, -2.9353e-01f, -2.3785e-01f, -2.4735e-01f, -2.1761e-01f, -2.2110e-01f, -2.0659e-01f, 

        -3.2288e-01f, -2.2171e-01f, -2.5261e-01f, -1.8503e-01f, -1.3016e-01f, -1.8469e-01f, -1.6924e-01f, -1.6592e-01f, 

        -1.9215e-01f, -3.1613e-01f, -3.2912e-01f, -2.6563e-01f, -2.4169e-01f, -1.5686e-01f, -1.4992e-01f, -2.2107e-01f, 

        -1.6184e-01f, -1.9864e-01f, -1.6123e-01f, -1.6910e-01f, -1.7562e-01f, -1.9269e-01f, -1.3679e-01f, -1.9432e-01f, 

        -2.9869e-01f, -1.7038e-01f, -2.1424e-01f, -2.1658e-01f, -1.4957e-01f, -2.4165e-01f, -1.2919e-01f, -1.7843e-01f, 

        -1.4972e-01f, -3.0090e-01f, -1.6027e-01f, -2.5911e-01f, -1.3642e-01f, -1.1777e-01f, -1.5795e-01f, -1.5412e-01f, 

        -1.3186e-01f, -1.9676e-01f, -1.7842e-01f, -1.3908e-01f, -2.1132e-01f, -1.8941e-01f, -1.6416e-01f, -1.2955e-01f, 

        -1.5238e-01f, -2.5053e-01f, -1.2710e-01f, -1.9446e-01f, -1.6214e-01f, -1.7244e-01f, -1.0594e-01f, -1.3744e-01f, 

        -2.0161e-01f, -1.6772e-01f, -1.3812e-01f, -2.2449e-01f, -8.6299e-02f, -1.7441e-01f, -1.6213e-01f, -3.0234e-01f, 

        -1.6911e-01f, -1.4376e-01f, -2.2008e-01f, -9.5928e-02f, -2.2020e-01f, -2.6003e-01f, -1.4271e-01f, -1.9715e-01f, 

        -9.6885e-02f, -1.0112e-01f, -1.1745e-01f, -1.4505e-01f, -1.1012e-01f, -2.2461e-01f, -2.4757e-01f, -8.9355e-02f, 

        -2.0936e-01f, -1.2172e-01f, -1.2773e-01f, -8.3099e-02f, -6.5573e-02f, -1.7208e-01f, -1.2759e-01f, -1.6586e-01f, 

        -8.7491e-02f, -7.0144e-02f, -2.6933e-01f, -1.3440e-01f, -1.7141e-01f, -2.0397e-01f, -1.0859e-01f, -2.4266e-01f, 

        -2.1318e-01f, -2.2046e-01f, -2.1754e-01f, -1.6458e-01f, -9.9237e-02f, -1.8317e-01f, -1.0820e-01f, -1.3133e-01f, 

        -7.8220e-02f, -1.8577e-01f, -7.9179e-02f, -1.5136e-01f, -7.4391e-02f, -1.4807e-01f, -1.2152e-01f, -1.1432e-01f, 

        -9.1316e-02f, -1.4874e-01f, -9.5707e-02f, -1.2740e-01f, -2.2922e-01f, -1.4621e-01f, -1.1297e-01f, -1.6036e-01f, 

        -1.8085e-01f, -1.0601e-01f, -1.7506e-01f, -2.0907e-01f, -1.4608e-01f, -1.6722e-01f, -1.3610e-01f, -1.7257e-01f, 

        -1.8543e-01f, -8.3910e-02f, -1.0247e-01f, -2.3389e-01f, -1.4152e-01f, -9.3192e-02f, -2.0484e-01f, -1.0072e-01f, 

        -5.9552e-02f, -1.0849e-01f, -1.1228e-01f, -1.1205e-01f, -1.9450e-01f, -1.3674e-01f, -1.2219e-01f, -4.6767e-02f, 

        -3.4235e-02f, -1.6237e-02f, 3.1052e-03f, -9.1900e-02f, -6.8624e-02f, -4.1462e-02f, -1.1603e-01f, -1.3504e-01f, 

        -2.0140e-01f, -5.3000e-02f, -1.3880e-01f, -1.1664e-02f, -1.5362e-01f, -1.0773e-01f, -1.1592e-01f, -6.1193e-02f, 

        -1.5096e-01f, -1.4575e-01f, -7.1263e-02f, -1.7118e-01f, -1.4813e-01f, -1.1457e-01f, -1.5266e-01f, -7.5794e-02f, 

        -1.6931e-01f, -9.4117e-02f, -7.8641e-02f, -6.8637e-01f, -7.8935e-01f, -1.2567e-01f, -4.5271e-01f, -4.0247e-01f, 

        -2.4587e-01f, -4.7344e-01f, -4.2751e-01f, -4.4293e-01f, -7.1480e-02f, -7.9100e-02f, -7.5432e-02f, -6.9590e-02f, 

        -4.7319e-02f, -5.0799e-02f, -1.5127e-01f, -6.9290e-02f, -8.4276e-02f, -1.2047e-01f, -1.4910e-01f, -1.4894e-01f, 

        -2.9811e-01f, -2.7316e-01f, -2.0418e-01f, -2.5703e-01f, -2.3304e-01f, -1.8958e-01f, -6.4530e-02f, -4.7897e-03f, 

        3.1714e-02f, -1.6294e-01f, -1.0112e-01f, -8.1991e-02f, -1.1255e-01f, -1.5165e-01f, -1.2194e-01f, -1.8419e-01f, 

        -1.3615e-01f, -1.4851e-01f, -1.2321e-01f, -1.6066e-01f, -1.2551e-01f, -9.6335e-02f, -9.2623e-02f, -8.2049e-02f, 

        -1.6065e-02f, 2.9020e-02f, -3.0691e-02f, -5.1183e-02f, -1.2538e-01f, -7.9163e-02f, -1.1839e-01f, -7.7458e-02f, 

        -7.9392e-02f, -5.6007e-02f, -1.6332e-01f, -1.3686e-01f, -1.7753e-01f, -9.1370e-02f, -8.1572e-02f, -1.0485e-01f, 

        -9.1131e-02f, -1.7509e-01f, -1.4767e+00f, -1.1071e+00f, -1.2685e+00f, -1.1308e+00f, -1.2528e+00f, -1.3371e+00f, 

        -1.2112e+00f, -1.0604e+00f, -1.4322e+00f, -7.3172e-02f, -1.2035e-01f, -1.0931e-01f, -4.1641e-02f, -7.4488e-02f, 

        3.8097e-02f, -2.1408e-01f, -1.2984e-01f, -1.5564e-01f, -2.7931e-01f, -1.0180e-01f, -2.1728e-01f, -3.9087e-01f, 

        -1.6364e-01f, -5.3300e-02f, -2.3519e-01f, -1.6813e-01f, -1.8333e-01f, -8.7755e-02f, -1.3169e-01f, -8.8558e-02f, 

        -1.0763e-01f, -8.6551e-02f, -6.3269e-02f, -1.6730e-01f, -1.8293e-01f, -1.2210e-01f, -9.3715e-02f, -4.2432e-02f, 

        1.0907e-02f, -5.9552e-02f, -1.6084e-01f, -1.7798e-01f, -4.3569e-02f, -5.0312e-02f, -5.0298e-02f, -1.1332e-01f, 

        -9.0447e-02f, -1.3267e-01f, -2.0562e-01f, -1.2267e-01f, -9.0455e-02f, -2.0590e-01f, -1.5506e-01f, -1.6799e-01f, 

        -1.9383e-01f, -9.7934e-02f, -2.6498e-01f, -8.7253e-02f, -1.6713e-01f, -1.5982e-01f, -1.3071e-01f, -1.5767e-01f, 

        -1.4110e-01f, -1.5380e-01f, -9.6203e-02f, -9.2387e-02f, -1.5019e-01f, -1.5502e-01f, -1.3189e-01f, -2.1469e-01f, 

        -1.4418e-01f, -1.6639e-01f, -1.3843e-01f, -1.7966e-01f, -1.0285e-01f, -1.3962e-01f, -1.8895e-01f, -3.9229e-02f, 

        -1.7792e-01f, -6.6721e-02f, -8.7538e-02f, -1.7679e-01f, -6.9711e-02f, -1.7728e-01f, -1.1705e-01f, -9.6612e-02f, 

        -1.1571e-01f, -2.2144e-01f, -1.7604e-01f, 6.2749e-02f, -9.0376e-02f, -9.4384e-02f, -1.3879e-01f, -1.6774e-01f, 

        -1.0950e-01f, -1.3199e-01f, -2.0003e-01f, -1.0564e-01f, -1.0129e-01f, -1.8092e-01f, -1.8919e-01f, -1.3022e-01f, 

        -1.6683e-01f, -1.8227e-01f, -1.4807e-01f, -1.5698e-01f, -1.6208e-01f, -9.6273e-02f, -4.7010e-02f, -9.1154e-02f, 

        -1.2516e-01f, -1.6988e-01f, -6.2522e-02f, -1.0463e-01f, -1.9244e-01f, -1.0709e-01f, -9.8248e-02f, -7.1697e-02f, 

        -4.4985e-02f, 1.9815e-03f, -1.0726e-01f, 5.3140e-04f, -1.6395e-01f, -6.2401e-02f, -1.0810e-01f, -8.1156e-02f, 

        -8.0011e-02f, -1.3709e-01f, -4.7999e-02f, -1.0471e-01f, -9.9659e-02f, -1.0887e-01f, -1.0847e-01f, -1.2717e-01f, 

        -6.8991e-02f, -9.8271e-02f, -5.3779e-02f, -7.1338e-02f, -1.3253e-01f, -9.5751e-02f, -9.3619e-02f, -1.9402e-01f, 

        -1.3491e-01f, -1.2882e-01f, -1.0646e-01f, -1.9230e-01f, -1.9404e-01f, -1.6638e-01f, -5.6448e-02f, -1.5224e-01f, 

        -1.6472e-01f, -2.6616e-01f, -1.1275e-01f, -1.0904e-01f, -1.0729e-02f, -7.8614e-02f, -1.1475e-01f, -5.9089e-02f, 

        -7.1103e-02f, -1.7016e-01f, -1.2392e-01f, -8.2708e-02f, -3.3556e-02f, -3.1116e-02f, -1.3251e-01f, -1.3337e-01f, 

        -8.4041e-02f, -3.6968e-02f, -1.8983e-01f, -1.4594e-01f, -1.0924e-01f, -2.0843e-01f, -9.1066e-02f, -1.8414e-01f, 

        -1.8790e-01f, -1.0364e-01f, -1.0925e-01f, -1.4758e-01f, -1.0350e-01f, -3.3682e-02f, -8.1975e-02f, -6.9506e-02f, 

        -9.4637e-02f, -1.3267e-01f, -1.6729e-01f, -1.1806e-01f, -1.5379e-01f, -1.7945e-01f, -1.1208e-01f, -1.9544e-02f, 

        1.8854e-01f, -9.3344e-02f, -2.1263e-02f, 4.5251e-02f, -1.2518e-01f, 1.9784e-02f, -4.4290e-02f, 6.4988e-03f, 

        -1.6067e-01f, -1.3510e-01f, -1.4838e-01f, -1.3628e-01f, -1.2735e-01f, -9.2441e-02f, -2.0529e-01f, -1.1633e-01f, 

        -2.0173e-01f, -1.4609e-01f, -1.2996e-01f, -9.5776e-02f, -7.9892e-02f, -1.5907e-01f, -8.8900e-02f, -6.3611e-02f, 

        -1.3825e-01f, -1.3482e-01f, -1.6136e-01f, -1.2967e-01f, -1.3442e-01f, -2.3923e-01f, -1.5169e-01f, -6.5481e-02f, 

        -1.6741e-01f, -1.7107e-01f, -1.1445e-01f, -1.3022e+00f, -1.1622e+00f, -1.1591e+00f, -7.1229e-01f, -6.0280e-01f, 

        -5.2647e-01f, -7.6146e-01f, -5.5369e-01f, -5.5248e-01f, -1.3484e-01f, -1.2017e-01f, -1.3005e-01f, -1.5350e-01f, 

        -7.5838e-02f, -1.2324e-01f, -1.3277e-01f, -1.1312e-01f, -8.5262e-02f, -2.8898e-01f, -2.0374e-01f, -2.2210e-01f, 

        -4.1184e-01f, -2.8837e-01f, -2.6085e-01f, -4.0549e-01f, -3.0080e-01f, -2.6962e-01f, -1.2071e-01f, -1.6966e-01f, 

        -1.0776e-01f, -1.0459e-01f, -8.5986e-02f, -7.8439e-02f, -1.9543e-01f, -1.5695e-01f, -6.0973e-02f, -1.9754e-01f, 

        -2.6679e-01f, -2.5087e-01f, -2.3319e-01f, -2.2996e-01f, -2.1309e-01f, -2.8786e-01f, -1.2185e-01f, -2.6352e-01f, 

        -1.7890e-01f, -9.6862e-02f, -2.1419e-01f, -9.4611e-02f, -9.6999e-02f, -1.7396e-01f, -1.3446e-01f, -1.4266e-01f, 

        -1.1964e-01f, -1.2869e-01f, -1.3032e-01f, -1.5589e-01f, -1.9442e-01f, -1.4770e-01f, -8.4468e-02f, -1.8491e-01f, 

        -1.2997e-01f, -1.0627e-01f, -2.4031e+00f, -1.4130e+00f, -1.5285e+00f, -2.2699e+00f, -1.1851e+00f, -1.4462e+00f, 

        -2.4319e+00f, -1.3174e+00f, -1.6912e+00f, -1.5688e-01f, -1.8699e-01f, -1.4640e-01f, -3.3352e-02f, -1.0395e-01f, 

        -1.2959e-01f, -1.4628e-01f, -1.2187e-01f, -1.4832e-01f, -3.7486e-01f, -3.2919e-01f, -2.5487e-01f, -5.1913e-01f, 

        -1.5860e-01f, -2.3699e-01f, -2.7266e-01f, -1.7331e-01f, -2.2020e-01f, -1.9858e-01f, -1.3197e-01f, -7.1819e-02f, 

        -1.0165e-01f, -1.8834e-01f, -1.3455e-01f, -1.8063e-01f, -1.5293e-01f, -1.0576e-01f, -3.1406e-02f, -7.4397e-02f, 

        -1.4355e-01f, -6.9839e-02f, -5.9156e-02f, -9.3023e-02f, -9.3232e-02f, -1.5517e-01f, -2.0458e-01f, -1.7754e-01f, 

        -2.5056e-01f, -3.0851e-01f, -2.0621e-01f, -2.2751e-01f, -1.4593e-01f, -2.3509e-01f, -2.9751e-01f, -2.0079e-01f, 

        -2.0619e-01f, -1.6271e-01f, -2.5733e-01f, -8.6742e-02f, -1.0558e-01f, -1.8490e-01f, -1.0773e-01f, -7.4675e-02f, 

        -1.6517e-01f, -2.0390e-01f, -2.2561e-01f, -2.4123e-01f, -1.9224e-01f, -2.3949e-01f, -2.2159e-01f, -2.6925e-01f, 

        -3.0342e-01f, -2.0069e-01f, -2.6261e-01f, -1.5942e-01f, -1.2358e-01f, -1.5213e-01f, -1.2650e-01f, -1.3100e-01f, 

        -1.5721e-01f, -2.1202e-01f, -2.1760e-01f, -3.6437e-01f, -2.8072e-01f, -4.0293e-01f, -1.7066e-01f, -1.6269e-01f, 

        -2.2388e-01f, -2.4895e-01f, -1.8527e-01f, -2.6326e-01f, -1.0290e-01f, -1.2706e-01f, -9.4260e-02f, -1.7219e-01f, 

        -1.4036e-01f, -1.0826e-01f, -1.6908e-01f, -1.5782e-01f, -1.3026e-01f, -2.2397e-01f, -1.8335e-01f, -1.1006e-01f, 

        -2.1653e-01f, -2.5948e-01f, -1.6142e-01f, -2.6227e-01f, -1.4943e-01f, -1.7199e-01f, -1.4936e-01f, -7.8561e-02f, 

        -1.0786e-01f, -2.5589e-01f, -1.5959e-01f, -8.5614e-02f, -1.2719e-01f, -7.4021e-02f, -1.5519e-01f, -2.4859e-01f, 

        -1.2956e-01f, -5.2252e-02f, -1.9675e-01f, -1.9899e-01f, -1.5510e-01f, -2.8321e-01f, -8.3592e-02f, -7.7836e-02f, 

        -1.4820e-01f, -1.5641e-01f, -1.3523e-01f, -2.1876e-01f, -1.4144e-01f, -1.8522e-01f, -9.9488e-02f, -1.3481e-01f, 

        -1.5241e-01f, -5.4035e-02f, -1.0935e-01f, -1.1900e-01f, -1.3424e-01f, -1.0532e-01f, -1.4618e-01f, -1.0665e-01f, 

        -4.8799e-02f, -1.2592e-01f, -1.3461e-01f, -2.0907e-01f, -1.2023e-01f, -1.2614e-01f, -1.7283e-01f, -1.6528e-01f, 

        -2.2139e-01f, -2.0482e-01f, -2.3596e-01f, -1.6000e-01f, -1.3621e-01f, -1.2725e-01f, -1.4745e-01f, -1.1774e-01f, 

        -6.3757e-02f, -1.7942e-01f, -7.7309e-02f, -1.4967e-01f, -1.3599e-01f, -1.0297e-01f, -6.2628e-02f, -1.5576e-01f, 

        -8.0500e-02f, -9.4935e-02f, -1.5865e-01f, -1.8992e-01f, -1.2423e-01f, -2.5411e-01f, -1.8962e-01f, -3.1377e-01f, 

        -2.0067e-01f, -1.8713e-01f, -1.6189e-01f, -2.5320e-01f, -2.2698e-01f, -2.2973e-01f, -1.2345e-01f, -1.0694e-01f, 

        -7.9939e-02f, -1.2761e-01f, -9.3897e-02f, -1.4582e-01f, -1.7635e-01f, -6.3937e-02f, -1.7096e-01f, -1.7981e-01f, 

        -1.2913e-01f, -1.1196e-01f, -1.1918e-01f, -1.1492e-01f, -1.4685e-01f, -1.6435e-01f, -1.2574e-01f, -8.9747e-02f, 

        -1.5498e-01f, -1.3868e-01f, -1.2376e-01f, -1.3962e-01f, -1.9954e-01f, -1.7813e-01f, -1.7112e-01f, -1.1312e-01f, 

        -7.6205e-02f, -1.2390e-01f, -1.3208e-01f, -4.2364e-02f, -9.4837e-02f, -6.3135e-02f, -7.6852e-02f, -9.7629e-02f, 

        -1.0408e-01f, -7.9266e-02f, -1.4978e-01f, -1.2280e-01f, -1.5629e-01f, -1.9078e-01f, -1.4921e-01f, -1.5405e-01f, 

        -1.7392e-01f, -6.7723e-02f, -1.5226e-01f, -1.5908e+00f, -1.5034e+00f, -1.1345e+00f, -8.8699e-01f, -7.1728e-01f, 

        -6.6501e-01f, -8.1981e-01f, -4.4645e-01f, -5.3549e-01f, -7.6528e-02f, -4.4458e-02f, -1.3614e-01f, -3.6979e-02f, 

        -1.5196e-01f, -1.3763e-01f, -3.7419e-02f, -1.4391e-01f, -1.8270e-01f, -4.7018e-01f, -1.8699e-01f, -1.3739e-01f, 

        -4.6122e-01f, -3.1724e-01f, -2.6219e-01f, -4.5392e-01f, -2.0621e-01f, -2.3457e-01f, -1.8375e-01f, -1.1989e-01f, 

        -2.0281e-01f, -3.6738e-02f, -1.1957e-01f, -1.1773e-01f, -1.3594e-01f, -9.7397e-02f, -5.2139e-02f, -1.5787e-01f, 

        -2.0213e-01f, -1.9279e-01f, -1.5452e-01f, -2.9213e-01f, -2.1759e-01f, -1.5320e-01f, -2.1619e-01f, -2.3679e-01f, 

        -2.3032e-01f, -1.4892e-01f, -2.2782e-01f, -1.4974e-01f, -5.2207e-02f, -2.3870e-01f, -1.5340e-01f, -1.3364e-01f, 

        -1.1000e-01f, -8.4709e-02f, -1.3034e-01f, -8.1488e-02f, -4.7893e-02f, -3.3043e-02f, -1.7247e-01f, -1.3176e-01f, 

        -7.7659e-02f, -5.1680e-02f, -2.5864e+00f, -1.4508e+00f, -1.8468e+00f, -1.9445e+00f, -1.2200e+00f, -1.6428e+00f, 

        -1.9739e+00f, -9.9829e-01f, -1.5662e+00f, -3.6998e-02f, -1.1086e-01f, -6.3868e-02f, -1.1437e-01f, -8.0313e-02f, 

        -1.7885e-01f, -8.1699e-02f, -1.0486e-01f, -6.3757e-02f, -3.8574e-01f, -2.7965e-01f, -6.1934e-02f, -5.0374e-01f, 

        -1.8873e-01f, -1.6017e-01f, -2.1395e-01f, -2.1038e-01f, -1.2455e-01f, -9.3413e-02f, -1.9281e-01f, -1.2774e-01f, 

        -8.1894e-02f, -1.1666e-01f, -1.1705e-01f, -8.4621e-02f, -1.4746e-01f, -5.6193e-02f, -1.1921e-01f, -8.6364e-02f, 

        -1.1564e-01f, -6.3098e-02f, -8.3377e-02f, -9.8995e-02f, -9.8000e-02f, -8.7344e-02f, -8.3030e-02f, -2.1936e-01f, 

        -1.4351e-01f, -1.5183e-01f, -2.8455e-01f, -3.0650e-01f, -2.2319e-01f, -2.3912e-01f, -2.5324e-01f, -1.6961e-01f, 

        -3.2786e-01f, -1.5296e-01f, -3.5528e-01f, -1.7026e-01f, -1.0633e-01f, -1.5024e-01f, -1.2175e-01f, -1.6135e-01f, 

        -1.1101e-01f, -2.2558e-01f, -2.4485e-01f, -2.3343e-01f, -3.2490e-01f, -2.2846e-01f, -2.8308e-01f, -2.6760e-01f, 

        -1.8658e-01f, -2.1320e-01f, -2.0194e-01f, -9.9509e-02f, -4.1214e-02f, -9.7194e-02f, -5.8929e-02f, -2.2181e-01f, 

        -1.7712e-01f, -1.2903e-01f, -1.9432e-01f, -4.2060e-01f, -2.5892e-01f, -2.5305e-01f, -2.6468e-01f, -2.0751e-01f, 

        -1.2188e-01f, -1.3759e-01f, -1.3491e-01f, -3.3074e-01f, -2.1001e-01f, -1.3047e-01f, -1.2807e-01f, -1.2724e-01f, 

        -8.5593e-02f, -1.5937e-01f, -1.2906e-01f, -1.5966e-01f, -1.9284e-01f, -1.7135e-01f, -1.4964e-01f, -1.6957e-01f, 

        -2.2158e-01f, -1.4196e-01f, -1.2424e-01f, -2.3193e-01f, -2.3998e-01f, -2.1598e-01f, -9.4937e-02f, -9.0071e-02f, 

        -1.0993e-01f, -1.8398e-01f, -1.3612e-01f, -1.5158e-01f, -6.8892e-02f, -1.4680e-01f, -1.7885e-01f, -3.2293e-01f, 

        1.8065e-02f, -2.7308e-01f, -1.0758e-01f, -4.0017e-01f, -1.3821e-01f, -1.5649e-01f, -5.8906e-02f, -8.5069e-02f, 

        -1.5102e-01f, -8.1293e-03f, -1.5741e-01f, -1.5029e-01f, -1.0244e-01f, -1.9339e-01f, -1.6440e-01f, -9.0259e-02f, 

        -1.0171e-01f, -1.3522e-01f, -5.5050e-02f, -1.0691e-01f, -3.4619e-02f, -1.2798e-01f, -5.1139e-02f, -1.1871e-01f, 

        -7.2040e-02f, -8.3390e-02f, -1.4626e-01f, -1.5845e-01f, -1.5647e-01f, -2.7144e-01f, -1.5814e-01f, -1.2066e-01f, 

        -2.0142e-01f, -1.8010e-01f, -1.0360e-01f, -2.1960e-01f, -1.1314e-01f, -1.1366e-01f, -1.4921e-01f, -5.1529e-02f, 

        -1.0099e-01f, -1.6670e-01f, -1.3525e-01f, -1.0936e-01f, -1.3662e-01f, -8.9249e-02f, -9.3520e-03f, -1.1370e-01f, 

        -1.4158e-01f, -1.0732e-01f, -1.4647e-01f, -6.2483e-02f, -6.4189e-02f, -2.0823e-01f, -1.7291e-01f, 7.3856e-03f, 

        -2.9650e-01f, -9.9654e-02f, -8.1250e-02f, -1.6650e-01f, -8.7941e-02f, -2.6552e-01f, -6.6958e-02f, -1.5271e-01f, 

        -1.3141e-01f, -2.8615e-02f, -9.9279e-02f, -1.8209e-01f, -7.1574e-02f, -1.2711e-01f, -8.0470e-02f, 4.6265e-02f, 

        6.9585e-02f, -1.7677e-01f, -1.6129e-01f, -6.7904e-02f, -1.1477e-01f, -7.4526e-02f, -1.6553e-02f, -1.9793e-01f, 

        6.2535e-01f, 5.4920e-03f, -2.8745e-01f, 2.8522e-02f, -1.6082e-01f, -5.6190e-01f, -4.0114e-02f, -2.1387e-01f, 

        -1.4209e-01f, -1.6477e-01f, -1.0703e-01f, -1.7235e-01f, -1.4058e-01f, -2.4250e-01f, -3.3926e-02f, -1.6312e-01f, 

        -1.8691e-01f, -2.3198e-01f, -1.9153e-01f, 8.9455e-01f, 3.0742e-01f, -1.4995e+00f, -1.9983e-01f, 5.3406e-02f, 

        4.4035e-01f, -2.1177e-01f, -1.5850e-01f, 1.7576e-01f, 1.2561e-03f, 4.5821e-01f, -4.2310e-01f, -5.7344e-01f, 

        -9.6255e-01f, -2.3258e-01f, -9.6004e-01f, 1.8950e-01f, -9.2419e-01f, 2.2606e-01f, 1.5585e-01f, -6.8644e-01f, 

        -8.9067e-02f, 2.2538e-01f, -8.7541e-02f, -4.2702e-01f, -6.5016e-01f, 3.5146e-01f, 6.5925e-01f, -1.2488e+00f, 

        -3.9766e-01f, -5.1172e-01f, 1.6885e-01f, 1.8200e-02f, -4.6128e-01f, -5.1355e-01f, -1.0025e-01f, -3.8867e-01f, 

        5.5453e-01f, 5.9666e-01f, -4.9597e-01f, -6.9887e-01f, 4.7733e-01f, -3.7321e-01f, 3.8079e-01f, -7.3088e-01f, 

        1.0415e+00f, 4.4817e-01f, -9.4006e-01f, -7.7747e-01f, -9.3185e-02f, 2.9830e-01f, 1.3457e-01f, -3.0667e-01f, 

        4.4510e-01f, 2.4201e-01f, -7.2458e-01f, -6.5902e-01f, -2.3844e-01f, 3.5618e-02f, -6.4090e-01f, -5.9270e-02f, 

        -2.7092e-01f, -2.6739e-01f, -1.0081e-01f, 1.9495e-01f, -3.2887e-01f, 9.1928e-02f, -4.0887e-01f, -5.4587e-01f, 

        -3.6736e-01f, 4.8451e-01f, -5.3911e-01f, 1.4728e-01f, 5.4682e-01f, -3.0369e-01f, -1.7989e-02f, 5.0657e-01f, 

        -2.0768e-01f, -8.8057e-01f, -7.4855e-01f, -1.4390e-01f, -3.3254e-01f, 3.1419e-01f, -1.3212e-01f, -5.5657e-02f, 

        -5.5025e-02f, 1.5024e-02f, -3.1266e-01f, -1.5920e-01f, -2.9135e-02f, 6.8837e-02f, -3.1012e-01f, -3.9319e-01f, 

        2.5299e-01f, -3.8649e-02f, 4.2795e-01f, -1.1845e-01f, -5.5874e-01f, 5.5519e-02f, -1.2053e+00f, -6.8484e-01f, 

        -1.7392e+00f, 1.5054e+00f, -8.2789e-01f, -1.9638e+00f, 9.1571e-01f, -7.2042e-02f, -1.5796e+00f, -2.3258e-01f, 

        -1.2263e+00f, 4.3514e-01f, -8.2405e-01f, -3.0939e-01f, 1.4726e-01f, -1.1266e+00f, 5.4076e-01f, 1.2429e-01f, 

        4.0842e-01f, -2.1176e-01f, 1.1702e+00f, -2.6642e-01f, -6.4851e-01f, 1.1312e+00f, 3.4186e-01f, 5.6533e-02f, 

        -8.0872e-01f, -3.1593e-02f, 2.5421e-03f, -1.3188e+00f, -1.5512e+00f, 4.3145e-01f, 2.4921e-01f, 1.5160e+00f, 

        7.2022e-01f, 3.6665e-01f, 4.2503e-02f, -3.5089e-01f, -1.2874e-01f, -9.0254e-02f, -9.7574e-02f, 3.9572e-01f, 

        6.8093e-01f, 3.4782e-01f, 4.9308e-01f, 3.2775e-01f, 3.8622e-01f, -3.2143e-01f, 6.7285e-01f, 1.7632e-01f, 

        -3.0798e-01f, -6.6410e-01f, -8.3302e-02f, -7.3557e-01f, 4.5790e-01f, 9.8158e-01f, -1.9574e+00f, -3.8280e-01f, 

        -5.9605e-01f, -7.0455e-01f, -7.2233e-02f, 5.8382e-01f, -9.4266e-01f, -2.4229e-01f, -7.8427e-01f, 8.0860e-01f, 

        -2.4080e-01f, 1.4545e-01f, 1.3929e+00f, -3.4895e-01f, -3.2578e-01f, -1.7971e-01f, -2.2027e-01f, 2.7680e-01f, 

        1.4607e-02f, -1.7509e-01f, -8.0849e-02f, 3.8998e-01f, 9.3468e-02f, 5.9590e-02f, -8.0359e-01f, -2.0547e+00f, 

        -5.1799e-02f, -7.1661e-01f, -6.3633e-01f, 1.4969e-01f, -1.2942e+00f, 8.5197e-01f, -6.4348e-01f, -2.1028e-01f, 

        -1.4001e+00f, -6.9551e-01f, -5.3513e-03f, 1.8031e-01f, 4.8699e-01f, 4.5323e-02f, -7.3868e-01f, 7.2500e-01f, 

        2.5903e-01f, -2.9814e-01f, -3.2947e-01f, -9.2506e-01f, 8.0704e-01f, 5.3091e-02f, -1.4328e-01f, 3.4491e-01f, 

        8.4675e-02f, -3.3676e-01f, 2.0597e-01f, -2.8454e-01f, -1.1800e-01f, -1.3740e-01f, -6.3902e-02f, -4.5192e-02f, 

        -1.8525e-01f, -2.7561e-01f, 8.5416e-01f, -9.2687e-02f, -2.4323e-02f, 1.4938e-01f, 1.6363e-01f, 1.7040e-01f, 

        1.0843e+00f, -4.8388e-01f, 2.9602e-01f, -3.8776e-01f, 1.3229e+00f, 7.7481e-01f, -2.9254e-01f, -1.8402e+00f, 

        2.0526e-01f, -2.3775e+00f, -8.6536e-01f, -3.8815e-01f, 1.9536e-02f, -2.8693e-02f, 8.4639e-03f, -1.4920e-01f, 

        -5.5645e-02f, -1.6102e-01f, -1.0620e-01f, -4.6073e-03f, 1.9107e-02f, 3.4948e-01f, 6.2300e-02f, -1.0968e-01f, 

        -2.5150e+00f, -5.8161e-01f, 1.6475e+00f, -6.9608e-01f, -1.7749e+00f, 1.1393e+00f, -3.0111e-01f, -1.9874e+00f, 

        -2.8716e+00f, 2.8369e-01f, -3.2668e-01f, -8.2433e-01f, -9.6726e-01f, -7.8074e-01f, -3.1882e-01f, -2.2940e-01f, 

        1.4437e-01f, 6.1805e-03f, 1.5928e-01f, -3.2944e-01f, -1.1469e-01f, -4.6930e-02f, -1.4803e-01f, -7.1141e-02f, 

        -1.1103e-01f, -6.3020e-02f, -7.0164e-02f, -1.1491e-01f, -1.1319e-01f, -8.3241e-02f, -1.3067e-01f, -1.4470e-01f, 

        -1.9746e-01f, -5.9587e-02f, -7.7238e-02f, -6.3145e-02f, -9.6262e-02f, -7.8411e-02f, -9.0148e-02f, -8.5931e-02f, 

        -3.8490e-02f, -1.9695e-01f, -1.3601e-01f, -1.8322e-01f, -1.4853e-01f, -1.1150e-01f, -1.0045e-01f, -1.2623e-01f, 

        -1.4100e-01f, -1.8445e-01f, -1.2077e-01f, -9.1824e-01f, -1.3062e+00f, -1.0375e+00f, -7.6700e-01f, -5.7937e-01f, 

        -5.6019e-01f, -7.3531e-01f, -5.1469e-01f, -4.5545e-01f, -6.0352e-02f, -6.2614e-02f, -9.2045e-02f, -3.3853e-02f, 

        -1.1902e-01f, -1.0014e-01f, -2.0054e-01f, -1.4347e-01f, -1.2612e-01f, -3.6810e-01f, -1.7667e-01f, -2.3539e-01f, 

        -2.8678e-01f, -2.3139e-01f, -2.6327e-01f, -3.0389e-01f, -1.2335e-01f, -2.1213e-01f, -7.6258e-02f, -1.7893e-01f, 

        -6.7594e-02f, -1.6946e-01f, -1.2934e-01f, -1.2931e-01f, -1.1737e-01f, -1.0669e-01f, -1.3156e-01f, -1.1500e-01f, 

        -1.7609e-01f, -3.4675e-01f, -1.8884e-01f, -1.6845e-01f, -1.8447e-01f, -6.9786e-02f, -2.2312e-01f, -1.0636e-01f, 

        -1.1073e-01f, -8.3945e-02f, -1.2617e-01f, -1.8579e-01f, -1.5192e-01f, -1.9676e-01f, -1.4879e-01f, -1.8086e-01f, 

        -2.1879e-01f, -7.5663e-02f, -5.4089e-02f, -1.5262e-01f, -5.7854e-02f, -1.3596e-01f, -1.4845e-01f, -7.6014e-02f, 

        -9.3983e-02f, -1.7844e-01f, -2.1764e+00f, -1.3301e+00f, -1.4733e+00f, -1.6122e+00f, -1.1127e+00f, -1.4259e+00f, 

        -2.1008e+00f, -1.3117e+00f, -1.4133e+00f, -1.1352e-01f, -7.6710e-02f, -1.2744e-01f, -8.6445e-02f, -6.5557e-02f, 

        -1.3636e-01f, -7.2236e-02f, -1.7650e-01f, -1.0865e-01f, -3.2260e-01f, -2.2584e-01f, -2.6899e-01f, -4.8678e-01f, 

        -1.1054e-01f, -1.3671e-01f, -2.4837e-01f, -2.7171e-01f, -2.3721e-01f, -1.5891e-01f, -1.7164e-01f, -1.3717e-01f, 

        -5.6165e-02f, -3.8114e-02f, -1.2568e-01f, -3.6122e-02f, -8.4393e-02f, -1.6826e-01f, -5.7620e-02f, -1.4734e-01f, 

        -1.7300e-01f, -2.0550e-01f, -1.8482e-01f, -1.3046e-01f, -5.6386e-02f, -6.4833e-02f, -1.4613e-01f, -2.1867e-01f, 

        -1.7755e-01f, -1.8407e-01f, -2.5455e-01f, -1.3724e-01f, -1.7904e-01f, -1.4422e-01f, -1.9100e-01f, -1.5317e-01f, 

        -2.8175e-01f, -1.3956e-01f, -3.0819e-01f, -2.0902e-01f, -1.3252e-01f, -1.1934e-01f, -1.5018e-01f, -9.1621e-02f, 

        -1.1674e-01f, -1.6121e-01f, -2.2492e-01f, -1.9911e-01f, -2.2301e-01f, -2.6006e-01f, -2.2374e-01f, -3.0070e-01f, 

        -2.7806e-01f, -2.1128e-01f, -1.2073e-01f, -1.6058e-01f, -1.1439e-01f, -2.1335e-01f, -6.4876e-02f, -7.8441e-02f, 

        -6.9356e-02f, -1.9822e-01f, -1.3568e-01f, -3.5639e-01f, -2.3661e-01f, -3.8652e-01f, -2.3158e-01f, -2.7981e-01f, 

        -2.1671e-01f, -1.2514e-01f, -1.5814e-01f, -2.0472e-01f, -2.1312e-01f, -1.7551e-01f, -1.5102e-01f, -6.1815e-02f, 

        -1.0590e-01f, -4.4115e-02f, -1.6853e-01f, -1.1559e-01f, -1.1796e-01f, -2.1382e-01f, -1.7229e-01f, -1.0792e-01f, 

        -1.7432e-01f, -9.6329e-02f, -1.3648e-01f, -1.0320e-01f, -1.2074e-01f, -1.3063e-01f, -1.6383e-01f, -1.2369e-01f, 

        -8.4791e-02f, -1.6911e-01f, -9.0274e-02f, -1.2832e-01f, -1.4843e-01f, -5.1641e-02f, -1.0126e-01f, -1.9650e-01f, 

        -1.9715e-01f, -1.2575e-01f, -1.1372e-01f, -2.0939e-01f, -1.6579e-01f, -2.0797e-01f, -1.3928e-01f, -6.8574e-02f, 

        -1.3247e-01f, -2.2213e-01f, -1.6249e-01f, -1.4498e-01f, -1.0039e-01f, -1.5544e-01f, -1.2056e-01f, -6.3469e-02f, 

        -1.1834e-01f, -6.5015e-02f, -1.4698e-01f, -9.6552e-02f, -6.4177e-02f, -5.6945e-02f, -1.4672e-01f, -1.7699e-01f, 

        -5.5147e-02f, -1.9017e-01f, -1.4224e-01f, -2.4741e-01f, -1.8628e-01f, -1.7264e-01f, -1.0434e-01f, -1.4376e-01f, 

        -1.1816e-01f, -1.5240e-01f, -1.6280e-01f, -1.5083e-01f, -1.0910e-01f, -5.4071e-02f, -1.2911e-01f, -4.4204e-02f, 

        -7.1284e-02f, -1.3216e-01f, -1.0278e-01f, -1.3935e-01f, -1.5273e-01f, -9.3597e-02f, -6.9410e-02f, -6.2208e-02f, 

        -8.8055e-02f, -3.7711e-02f, -8.5275e-02f, -7.4726e-02f, -1.2719e-01f, -1.3403e-01f, -2.5971e-01f, -2.6849e-01f, 

        -2.2562e-01f, -2.0835e-01f, -1.7624e-01f, -1.1325e-01f, -6.0877e-02f, -1.3026e-01f, -1.4589e-01f, -9.7428e-02f, 

        -3.9626e-02f, -9.8818e-02f, -8.6122e-02f, -9.0642e-02f, -1.6556e-01f, -1.8055e-01f, -1.2559e-01f, -1.2717e-01f, 

        -8.9527e-02f, -1.9144e-01f, -7.9418e-02f, -1.5031e-01f, -9.4465e-02f, -4.5449e-02f, -9.3504e-02f, -4.8953e-02f, 

        -1.8386e-01f, -1.1720e-01f, -1.5432e-01f, -1.4337e-01f, -1.1362e-01f, -2.0027e-01f, -1.8935e-01f, -1.1081e-01f, 

        -1.7296e-01f, -2.0389e-01f, -1.1034e-01f, -1.0078e-01f, -1.4556e-01f, -7.1377e-02f, -1.4851e-01f, -6.9961e-02f, 

        -3.9995e-02f, -1.4065e-01f, -1.0993e-01f, -1.4701e-01f, -1.0174e-01f, -5.1089e-02f, -1.9626e-01f, -1.8935e-01f, 

        -2.0618e-01f, -1.1543e-01f, -1.7897e-01f, -1.4266e+00f, -1.4786e+00f, -8.8331e-01f, -7.1660e-01f, -6.4407e-01f, 

        -5.8898e-01f, -7.3029e-01f, -4.4870e-01f, -6.3451e-01f, -1.4592e-01f, -1.4352e-01f, -1.8145e-01f, -9.4588e-02f, 

        -1.0759e-01f, -1.4817e-01f, -9.6464e-02f, -1.2591e-01f, -1.9539e-01f, -3.2351e-01f, -3.1811e-01f, -2.1709e-01f, 

        -3.0574e-01f, -2.5890e-01f, -3.2087e-01f, -3.2102e-01f, -2.1943e-01f, -2.3565e-01f, -2.0795e-01f, -2.3616e-01f, 

        -2.2313e-01f, -2.2382e-01f, -2.6505e-01f, -1.7413e-01f, -1.2350e-01f, -1.4352e-01f, -2.2261e-01f, -1.5886e-01f, 

        -1.6431e-01f, -2.6427e-01f, -2.7805e-01f, -2.1599e-01f, -1.9249e-01f, -8.9320e-02f, -2.1549e-01f, -2.3980e-01f, 

        -2.2653e-01f, -1.3538e-01f, -1.9532e-01f, -1.6684e-01f, -1.7508e-01f, -2.5672e-01f, -1.6885e-01f, -2.0717e-01f, 

        -2.3066e-01f, -8.6091e-02f, -1.3820e-01f, -1.0823e-01f, -6.2827e-02f, -9.6669e-02f, -1.5572e-01f, -1.3230e-01f, 

        -1.3355e-01f, -1.3025e-01f, -2.6764e+00f, -9.8339e-01f, -1.5545e+00f, -1.8790e+00f, -1.3593e+00f, -1.7951e+00f, 

        -1.3856e+00f, -1.2521e+00f, -1.6624e+00f, -5.6226e-02f, -1.4392e-01f, -1.6540e-01f, -7.2933e-02f, -6.1812e-03f, 

        -1.2905e-01f, -5.9945e-02f, -8.3925e-02f, -1.7401e-01f, -3.9469e-01f, -2.3900e-01f, -2.1729e-01f, -5.1754e-01f, 

        -1.3873e-01f, -1.7869e-01f, -2.4267e-01f, -2.4431e-01f, -1.5667e-01f, -1.3950e-01f, -2.1520e-01f, -2.0051e-01f, 

        -8.6092e-02f, -3.3348e-02f, -1.5805e-01f, -4.7845e-02f, -8.4404e-02f, -6.4589e-02f, -1.6659e-01f, -1.4655e-01f, 

        -1.0664e-01f, -1.2003e-01f, -1.8578e-01f, -1.2180e-01f, 1.5189e-03f, -1.1423e-01f, -1.1609e-01f, -2.4529e-01f, 

        -2.2453e-01f, -2.4272e-01f, -3.0098e-01f, -2.6802e-01f, -2.3134e-01f, -1.1919e-01f, -1.0338e-01f, -1.5455e-01f, 

        -2.3594e-01f, -2.2436e-01f, -4.8728e-01f, -1.7739e-01f, -1.2411e-01f, -1.4045e-01f, -1.2038e-01f, -1.8385e-01f, 

        -1.6114e-01f, -2.5464e-01f, -2.3995e-01f, -2.8665e-01f, -2.3465e-01f, -2.7087e-01f, -2.5882e-01f, -2.5764e-01f, 

        -1.9009e-01f, -2.7968e-01f, -1.8752e-01f, -2.5936e-01f, -2.4607e-01f, -8.8065e-02f, -1.7032e-01f, -1.6667e-01f, 

        -9.0807e-02f, -1.9186e-01f, -1.2892e-01f, -3.6973e-01f, -2.4790e-01f, -3.4035e-01f, -3.4125e-01f, -2.8120e-01f, 

        -2.6640e-01f, -1.2625e-01f, -2.1643e-01f, -2.5730e-01f, -1.2900e-01f, -2.0261e-01f, -1.7354e-01f, -1.1447e-01f, 

        -1.2121e-01f, -9.6294e-02f, -1.2120e-01f, -8.7646e-02f, -9.6228e-02f, -1.8243e-01f, -1.5738e-01f, -2.2937e-01f, 

        -1.9193e-01f, -1.1644e-01f, -1.9572e-01f, -1.6060e-01f, -2.0059e-01f, -1.6168e-01f, -1.2569e-01f, -9.9793e-02f, 

        -1.9459e-01f, -1.7899e-01f, -1.0535e-01f, -2.0249e-01f, -1.9194e-01f, -9.9843e-02f, -2.2030e-01f, -2.6218e-01f, 

        -2.2574e-01f, -1.9722e-01f, -1.3386e-01f, -3.3998e-01f, -1.3088e-01f, -2.0428e-01f, -1.7709e-01f, -2.2054e-01f, 

        -1.3091e-01f, -1.1400e-01f, -2.0421e-01f, -1.2655e-01f, -1.8557e-01f, -1.6183e-01f, -1.1024e-01f, -9.9758e-02f, 

        -1.6554e-01f, -1.3926e-01f, -9.2381e-02f, -9.9593e-02f, -1.7898e-01f, -1.2010e-01f, -1.6928e-01f, -7.9375e-02f, 

        -7.0122e-02f, -1.6386e-01f, -2.2890e-01f, -2.2337e-01f, -2.4301e-01f, -2.1259e-01f, -1.8002e-01f, -2.4119e-01f, 

        -1.9470e-01f, -1.3205e-01f, -1.7192e-01f, -2.4519e-01f, -8.9100e-02f, -1.8448e-01f, -1.4862e-01f, -7.7658e-02f, 

        -1.4272e-01f, -1.8570e-01f, -1.0372e-01f, -1.6172e-01f, -7.9166e-02f, -1.4068e-01f, -1.8568e-01f, -1.5698e-01f, 

        -7.8744e-02f, -1.4185e-01f, -6.3730e-02f, -1.2925e-01f, -1.6462e-01f, -1.0449e-01f, -2.2229e-01f, -1.6794e-01f, 

        -1.9674e-01f, -1.8222e-01f, -2.5648e-01f, -7.3689e-02f, -1.3636e-01f, -1.8332e-01f, -1.1715e-01f, -1.8943e-01f, 

        -1.1114e-01f, -1.8197e-01f, -7.8358e-02f, -2.2463e-01f, -1.8429e-01f, -9.8380e-02f, -9.8759e-02f, -1.3148e-01f, 

        -1.0626e-01f, -1.6359e-01f, -9.6632e-02f, -1.2524e-01f, -1.2653e-01f, -7.4649e-02f, -1.6823e-01f, -5.0749e-02f, 

        -1.0112e-01f, -1.2531e-01f, -2.2935e-01f, -8.8210e-02f, -1.0788e-01f, -1.1410e-01f, -9.1320e-02f, -5.8732e-02f, 

        -5.4443e-02f, -3.1371e-02f, -1.8352e-01f, -1.2824e-01f, -1.1283e-01f, -1.4214e-01f, -4.3679e-02f, -4.7424e-02f, 

        -1.3605e-01f, -1.3709e-01f, -1.7732e-01f, -7.4907e-02f, -8.8514e-02f, -1.4368e-01f, -9.8076e-02f, -2.0714e-01f, 

        -7.4317e-02f, -7.2185e-02f, -1.5853e-01f, -8.5959e-01f, 3.1251e-01f, -1.9724e-01f, -7.4443e-01f, -3.4491e-01f, 

        -2.5980e-01f, -3.6904e-01f, -2.8985e-01f, -2.7614e-01f, -1.5117e-01f, -9.1890e-02f, -1.2256e-01f, -1.1775e-01f, 

        -9.8606e-02f, -1.5842e-01f, -1.2364e-01f, -7.2351e-02f, -6.1394e-02f, -3.2278e-01f, -2.6970e-01f, -2.0490e-01f, 

        -3.2528e-01f, -2.6323e-01f, -1.3721e-01f, -2.6528e-01f, -1.6684e-01f, -1.5092e-01f, -1.6958e-01f, -9.6562e-02f, 

        -1.3066e-01f, -1.7115e-01f, -8.0427e-02f, -1.3601e-01f, -1.1755e-01f, -7.3368e-02f, -5.2197e-02f, -7.6735e-02f, 

        -2.4700e-01f, -1.1817e-01f, -1.5431e-01f, -1.6209e-01f, -1.2174e-01f, -4.1789e-02f, -1.0811e-01f, -2.1524e-01f, 

        -1.6805e-01f, -1.4833e-01f, -3.0178e-01f, -1.8921e-01f, -1.8804e-01f, -2.4454e-01f, -2.0305e-01f, -8.6596e-02f, 

        -1.3589e-01f, -1.4526e-01f, -1.0995e-01f, -4.8051e-02f, -1.3638e-01f, -3.3167e-02f, -1.1627e-01f, -9.4065e-02f, 

        -1.1912e-01f, -9.3930e-02f, -8.9333e-01f, -6.3048e-01f, -1.4613e+00f, -1.2333e+00f, -6.9448e-01f, -1.1271e+00f, 

        -1.1009e+00f, -8.5629e-01f, -1.1852e+00f, -1.0126e-01f, -9.6841e-02f, -1.8745e-01f, -1.0087e-01f, -8.2296e-02f, 

        -1.1011e-01f, -1.4015e-01f, -7.5938e-02f, -3.2643e-02f, -8.3805e-01f, -1.6984e-01f, -1.3396e-01f, -2.6407e-01f, 

        -1.1374e-01f, -1.5813e-01f, -1.3910e-01f, -1.0294e-01f, -1.5371e-01f, -9.1432e-02f, -8.5232e-02f, -1.6094e-01f, 

        -7.6081e-02f, -9.3763e-02f, -1.4048e-01f, -1.1020e-01f, -8.7217e-02f, -6.9213e-02f, -1.2601e-01f, -2.1083e-01f, 

        -2.2537e-01f, -1.2768e-01f, -3.3875e-02f, -4.6012e-02f, -4.2417e-02f, -3.1729e-02f, -1.1550e-01f, -5.3788e-01f, 

        -2.2422e-01f, -1.9957e-01f, -2.2614e-01f, -2.2850e-01f, -8.0245e-02f, -7.1901e-02f, -8.6391e-02f, -6.4234e-02f, 

        -2.1004e-01f, -1.7577e-01f, -4.7497e-01f, -1.3765e-01f, -6.8305e-02f, -1.0309e-01f, -1.6039e-01f, -8.9224e-02f, 

        -1.2036e-01f, -1.6852e-01f, -9.7183e-02f, -1.5632e-01f, -1.2038e-01f, -1.2116e-01f, -1.9072e-01f, -1.7694e-01f, 

        -1.0891e-01f, -1.1467e-01f, -7.8588e-02f, -1.2842e-01f, -1.4030e-01f, -7.9201e-02f, -9.4718e-02f, -1.2513e-01f, 

        -1.1330e-01f, -8.8979e-02f, -1.2414e-01f, -4.7131e-03f, -1.6661e-01f, -1.6680e-01f, -1.1502e-01f, -1.0617e-01f, 

        -8.5898e-02f, -1.0105e-01f, -1.8007e-01f, -3.1162e-02f, -1.4910e-01f, -1.5933e-01f, -1.0638e-01f, -1.6074e-01f, 

        -8.5733e-02f, -1.4559e-01f, -1.1268e-01f, -1.2267e-01f, -7.0312e-02f, -1.9920e-01f, -1.2946e-01f, -1.0929e-01f, 

        -1.4799e-01f, -5.9524e-02f, -9.1746e-02f, -1.3489e-01f, -1.5689e-01f, -7.7774e-02f, -1.0071e-01f, -1.1900e-01f, 

        -1.6768e-01f, -2.3746e-01f, -6.1045e-02f, -1.4872e-01f, -1.3626e-01f, -8.1187e-02f, -6.1278e-02f, -2.1686e-01f, 

        -1.0222e-01f, -3.1520e-02f, -2.1600e-01f, -1.4934e-01f, -1.4935e-01f, -1.0248e-01f, -8.4038e-02f, -9.8988e-02f, 

        -1.2057e-01f, -2.4503e-01f, -2.8609e-01f, -2.1783e-01f, -7.9050e-02f, -2.4087e-01f, -1.2188e-01f, -1.2332e-01f, 

        -1.5938e-01f, -1.1955e-01f, -9.8227e-02f, -1.0870e-01f, -1.4878e-01f, -1.8556e-01f, -1.4870e-01f, -1.1319e-01f, 

        -1.1957e-01f, -1.5143e-01f, -2.4226e-01f, -1.4770e-01f, -3.6539e-01f, -1.9209e-01f, -1.2662e-01f, -1.5656e-01f, 

        -1.0163e-01f, -2.0203e-01f, -1.4307e-01f, -2.4710e-02f, -1.6249e-01f, -1.4343e-01f, -9.5528e-02f, -1.2607e-01f, 

        -9.0277e-02f, -8.5941e-02f, -1.2861e-01f, -1.2695e-01f, -1.2762e-01f, -1.5903e-01f, -1.3383e-01f, -1.5362e-01f, 

        -5.8314e-02f, -5.6546e-02f, -1.1698e-01f, -1.2232e-01f, -7.5837e-02f, -2.4867e-01f, -1.4232e-01f, -3.2909e-01f, 

        -1.2212e-01f, -7.3178e-02f, -1.6311e-01f, -4.2974e-02f, -1.2318e-01f, -2.0299e-01f, -1.1161e-01f, -1.4373e-01f, 

        -1.0149e-01f, -1.0467e-01f, -1.4418e-01f, -1.2729e-01f, -7.4974e-02f, -6.0903e-02f, -7.6946e-02f, -1.5916e-01f, 

        -6.5911e-02f, -1.0662e-01f, -1.5161e-01f, -9.5323e-02f, -1.5764e-01f, -1.2058e-01f, -1.3968e-01f, -6.5072e-02f, 

        -1.7311e-01f, -2.1695e-01f, -1.6199e-01f, -1.9869e-01f, -1.7277e-01f, -7.2345e-02f, -1.5359e-01f, -1.4172e-01f, 

        -1.1816e-01f, -9.7863e-02f, -1.6977e-01f, -1.2812e-01f, -1.6056e-01f, -1.6883e-01f, -1.7350e-01f, -1.6617e-01f, 

        -8.7847e-02f, -8.1782e-02f, -2.0192e-01f, -1.9883e-01f, -8.9394e-02f, -1.2679e-01f, -1.0778e-01f, -1.7793e-01f, 

        -1.7332e-01f, -1.8816e-01f, -1.1460e-01f, -1.4678e+00f, -1.5411e+00f, -1.1906e+00f, -7.9982e-01f, -5.3499e-01f, 

        -5.2421e-01f, -8.1922e-01f, -4.3723e-01f, -5.0169e-01f, -2.3684e-01f, -1.4893e-01f, -9.0618e-02f, -1.2936e-01f, 

        -9.6791e-02f, -1.4145e-01f, -1.2125e-01f, -1.8419e-01f, -1.2177e-01f, -3.4715e-01f, -2.7535e-01f, -1.5796e-01f, 

        -3.0549e-01f, -1.8630e-01f, -1.8898e-01f, -2.6315e-01f, -1.9188e-01f, -1.9768e-01f, -8.4228e-02f, -2.6440e-01f, 

        -1.3044e-01f, -1.1090e-01f, -1.6339e-01f, -1.7347e-01f, -1.5481e-01f, -1.8037e-01f, -1.0061e-01f, -1.0445e-01f, 

        -1.0727e-01f, -2.7768e-01f, -1.8320e-01f, -1.9692e-01f, -2.0663e-01f, -1.0036e-01f, -2.0612e-01f, -2.2720e-01f, 

        -2.0445e-01f, -1.3263e-01f, -2.3340e-01f, -1.5255e-01f, -1.5022e-01f, -2.8747e-01f, -1.0491e-01f, -9.4143e-02f, 

        -1.1294e-01f, -7.0280e-02f, -9.0232e-02f, -1.1355e-01f, -7.9996e-02f, -1.8166e-01f, -1.3638e-01f, -1.5065e-01f, 

        -1.6743e-01f, -1.8900e-01f, -2.5297e+00f, -1.0772e+00f, -1.6125e+00f, -2.1281e+00f, -1.5141e+00f, -1.3875e+00f, 

        -1.7719e+00f, -1.1873e+00f, -1.3468e+00f, -1.0203e-01f, -1.4358e-01f, -7.8481e-02f, -1.5282e-01f, -1.0401e-01f, 

        -1.4711e-01f, -1.2251e-01f, -1.2357e-01f, -1.1504e-01f, -3.2178e-01f, -2.2726e-01f, -1.7622e-01f, -3.5224e-01f, 

        -1.5553e-01f, -1.7020e-01f, -2.3709e-01f, -1.4251e-01f, -1.6230e-01f, -1.5480e-01f, -1.4775e-01f, -8.0801e-02f, 

        -9.6695e-02f, -1.2443e-01f, -1.5110e-01f, -1.5617e-01f, -8.9934e-02f, -1.7284e-01f, -1.5643e-01f, -1.3504e-01f, 

        -7.4425e-02f, -6.9706e-02f, -9.0561e-02f, -1.0582e-01f, -1.6937e-01f, -1.3968e-01f, -1.1627e-01f, -2.4732e-01f, 

        -1.9665e-01f, -2.6787e-01f, -2.5550e-01f, -2.4339e-01f, -1.7736e-01f, -1.8980e-01f, -1.8576e-01f, -1.7417e-01f, 

        -2.8149e-01f, -1.9959e-01f, -2.6547e-01f, -1.7981e-01f, -9.9896e-02f, -1.7312e-01f, -1.2133e-01f, -8.1536e-02f, 

        -1.4371e-01f, -2.6797e-01f, -2.4308e-01f, -3.2802e-01f, -2.3180e-01f, -2.5020e-01f, -1.6606e-01f, -2.2735e-01f, 

        -2.0258e-01f, -2.2411e-01f, -2.2510e-01f, -1.1540e-01f, -1.9613e-01f, -1.4851e-01f, -1.6909e-01f, -9.0603e-02f, 

        -1.7490e-01f, -8.4821e-02f, -2.1894e-01f, -3.9513e-01f, -2.8396e-01f, -2.3919e-01f, -2.0153e-01f, -2.7084e-01f, 

        -2.2045e-01f, -1.3990e-01f, -1.8226e-01f, -2.4733e-01f, -1.4006e-01f, -1.3759e-01f, -8.6319e-02f, -1.0147e-01f, 

        -1.3252e-01f, -1.4154e-01f, -1.3567e-01f, -1.7311e-01f, -1.0048e-01f, -2.4551e-01f, -1.8916e-01f, -1.4634e-01f, 

        -2.3513e-01f, -1.8056e-01f, -1.9261e-01f, -1.4562e-01f, -1.6681e-01f, -2.1275e-01f, -9.9103e-02f, -1.7056e-01f, 

        -1.3911e-01f, -1.7533e-01f, -1.7648e-01f, -1.5923e-01f, -1.4594e-01f, -1.5972e-01f, -1.5027e-01f, -1.7960e-01f, 

        -2.4207e-01f, -1.3183e-01f, -1.0971e-01f, -1.8174e-01f, -1.3714e-01f, -1.8853e-01f, -6.9465e-02f, -1.1128e-01f, 

        -1.9915e-01f, -2.1652e-01f, -1.0088e-01f, -1.3660e-01f, -1.1408e-01f, -2.1783e-01f, -1.7850e-01f, -1.9041e-01f, 

        -9.7048e-02f, -1.2599e-01f, -1.7016e-01f, -1.9180e-01f, -9.6678e-02f, -1.2189e-01f, -6.2843e-02f, -1.8234e-01f, 

        -1.7384e-01f, -1.1718e-01f, -1.6179e-01f, -1.6933e-01f, -9.2392e-02f, -1.5072e-01f, -1.8265e-01f, -1.7799e-01f, 

        -1.2449e-01f, -2.0336e-01f, -1.3196e-01f, -1.6641e-01f, -1.7249e-01f, -1.4268e-01f, -1.3292e-01f, -8.5112e-02f, 

        -7.7750e-02f, -1.2208e-01f, -1.6465e-01f, -1.1369e-01f, -1.4233e-01f, -1.5840e-01f, -8.1099e-02f, -1.1942e-01f, 

        -7.4324e-02f, -1.0285e-01f, -6.9641e-02f, -1.7242e-01f, -1.1488e-01f, -1.4748e-01f, -2.1390e-01f, -1.3767e-01f, 

        -2.2814e-01f, -2.7486e-01f, -2.1266e-01f, -1.0392e-01f, -1.4541e-01f, -3.0417e-01f, -1.3155e-01f, -1.4212e-01f, 

        -1.5222e-01f, -1.2787e-01f, -1.0328e-01f, -1.2264e-01f, -1.7587e-01f, -9.8429e-02f, -1.2934e-01f, -1.1458e-01f, 

        -6.6584e-02f, -1.0417e-01f, -1.0072e-01f, -3.2641e-02f, -7.3499e-02f, -1.0779e-01f, -1.6712e-01f, -2.2109e-02f, 

        5.3549e-01f, -5.2010e-02f, -1.9972e-01f, -5.6984e-01f, -9.6117e-02f, 3.7081e-01f, 2.4966e-01f, -3.2622e-01f, 

        1.3409e+00f, 7.4751e-02f, -3.6989e-03f, -1.2255e-01f, -1.5783e-01f, -1.2730e-01f, -1.9777e-01f, -6.0461e-02f, 

        -7.1889e-02f, -1.1735e-01f, -1.3230e+00f, 7.2146e-01f, 1.3723e-01f, -1.0324e+00f, -5.6905e-01f, -2.1818e-02f, 

        1.8928e-01f, 2.4730e-01f, 9.3303e-01f, 6.0652e-01f, 4.7131e-01f, 6.6371e-01f, -9.6138e-01f, 6.1945e-01f, 

        8.5606e-01f, -1.3468e+00f, -3.6144e-01f, -7.5599e-01f, -2.9368e-01f, 7.3292e-01f, -1.8020e-01f, -1.7111e-01f, 

        -4.2497e-01f, 8.9357e-02f, -5.2965e-02f, -6.9309e-02f, -3.2962e-01f, -3.9685e+00f, 8.4399e-01f, -5.9476e-01f, 

        -2.4825e+00f, -1.4077e+00f, -6.3918e-01f, -1.6008e+00f, -1.1083e+00f, -1.8215e+00f, -1.1589e+00f, 1.1339e+00f, 

        7.8819e-01f, -3.5367e-01f, 4.2963e-01f, 7.3123e-01f, -3.8652e-01f, -3.8744e-01f, -2.6268e-01f, 7.1650e-01f, 

        4.9704e-01f, 3.0264e-02f, -5.0034e-01f, -3.8919e-01f, -7.3060e-02f, 2.7461e-01f, 4.9682e-01f, 5.3566e-01f, 

        1.6042e-01f, 5.7506e-03f, -1.3013e-01f, 1.7860e-01f, -9.8009e-01f, -1.6821e+00f, -2.0613e+00f, -9.4880e-01f, 

        4.1596e-01f, -3.9964e-01f, 1.1214e+00f, -1.0355e-01f, -1.0649e-01f, -3.1995e-01f, -3.9356e-02f, -1.3396e-01f, 

        -5.6782e-02f, -3.8055e-01f, -2.3482e+00f, -1.2346e+00f, -2.2463e+00f, -6.8804e-01f, -2.7197e-01f, 5.6166e-02f, 

        4.9769e-01f, 1.5572e-01f, 8.1174e-01f, -1.1697e-01f, 2.0752e-01f, -3.1323e-01f, -2.3499e-01f, -8.5370e-02f, 

        -2.0516e-01f, -1.0621e-01f, -1.4198e-01f, -1.1409e-02f, 5.7707e-01f, 5.8656e-01f, 4.5375e-01f, 5.0053e-01f, 

        -1.8024e-01f, -1.0346e+00f, -6.9219e-01f, 7.3723e-01f, 5.8485e-01f, -4.0189e-02f, -3.6412e-01f, 6.7718e-01f, 

        -3.3820e-01f, -1.8430e-01f, -2.8830e-01f, 1.6162e-02f, 2.5081e-01f, -2.9142e-01f, -8.6331e-02f, 6.2891e-02f, 

        1.4933e+00f, 3.3732e-01f, -4.3491e-01f, -7.4722e-04f, -1.3340e-01f, 7.4054e-02f, 2.3763e-01f, -5.1336e-01f, 

        7.9028e-02f, -7.3572e-02f, -3.1502e-01f, 2.4750e-01f, 1.5951e-01f, 1.3341e+00f, 1.7811e-01f, 3.4010e-01f, 

        -4.9389e-02f, 6.5542e-01f, -8.6660e-01f, -3.3029e-01f, -8.3629e-01f, 8.0703e-02f, -1.4554e+00f, -6.4071e-01f, 

        -1.6734e+00f, -1.3070e-01f, 2.0404e-01f, 7.5051e-01f, 3.4133e-02f, 3.3748e-01f, -7.2224e-01f, -9.8442e-01f, 

        -1.0439e+00f, -2.7699e-01f, 1.6601e+00f, 2.3822e-01f, 4.1131e-01f, -3.3031e-01f, 2.5121e-01f, -2.4821e-01f, 

        -9.1647e-01f, -2.9558e-01f, -1.1098e+00f, 2.9890e-01f, 9.1776e-01f, 3.0481e-01f, 1.3039e+00f, 5.8618e-01f, 

        4.6149e-01f, -1.4275e-01f, 2.1747e-01f, -1.6229e-01f, -9.7907e-01f, -7.7873e-01f, 8.8000e-01f, -2.3762e-01f, 

        -9.4763e-01f, 6.0537e-01f, 3.0661e-01f, 2.6629e-01f, -1.4521e+00f, -1.7868e-01f, 7.8021e-01f, -1.2074e-02f, 

        -3.6628e-03f, -1.5216e-03f, 4.5258e-02f, -7.6129e-02f, 9.3260e-02f, 1.8327e-02f, -8.2773e-01f, -3.5908e-01f, 

        4.4207e-01f, -1.1874e-01f, -1.3840e-01f, 1.9709e-02f, -7.5719e-01f, -9.3406e-02f, 8.1018e-01f, 7.4914e-02f, 

        4.6722e-01f, 2.6374e+00f, -1.1594e+00f, -1.8769e-01f, -4.1911e-01f, -2.5255e+00f, 3.3351e-02f, 1.0779e+00f, 

        -6.2201e-03f, -1.6627e+00f, 1.1336e+00f, -6.9433e-01f, -1.6302e+00f, 3.1073e-01f, -1.7170e+00f, -1.9050e+00f, 

        -1.1315e+00f, 3.9014e-02f, -9.8441e-02f, -6.5704e-02f, -1.4112e-01f, -1.0636e-01f, -1.2654e-01f, -1.4152e-01f, 

        -1.3913e-01f, -3.6296e-02f, 5.1797e-01f, 1.3899e-02f, 3.0406e-01f, -9.1041e-01f, -3.7639e-01f, -7.2160e-01f, 

        5.7287e-01f, 1.0879e+00f, 1.1660e+00f, -3.7178e-01f, 1.0252e-01f, 3.5663e-01f, 1.4507e+00f, 4.4454e-02f, 

        -1.8513e-01f, 2.1617e-01f, -7.1146e-02f, -2.0798e-01f, 1.0624e-02f, -1.4160e-01f, -6.6388e-02f, -1.8649e-01f, 

        -1.1780e-01f, -8.1109e-02f, -1.8158e-01f, -1.9153e-01f, -2.0022e-02f, 5.9643e-02f, 5.8280e-01f, 1.0671e+00f, 

        -8.7149e-02f, -2.5755e-01f, -7.4342e-02f, 4.5164e-01f, -1.7813e-01f, 6.0638e-01f, 2.7923e-01f, -3.0832e-02f, 

        1.4902e+00f, -6.3843e-01f, 6.6382e-01f, 3.8390e-01f, -1.5134e+00f, 5.7113e-02f, -9.9841e-01f, -1.6681e-01f, 

        -1.4206e-01f, -7.6509e-02f, -1.7972e-01f, -1.8350e-01f, -1.3169e-01f, -1.7983e-01f, -1.7412e-01f, -9.4559e-02f, 

        -1.0934e-03f, -1.4118e-01f, -3.6340e-01f, -3.6083e-01f, -5.0707e-01f, -3.5439e-02f, -2.1587e-01f, -5.7349e-01f, 

        3.4073e-01f, -1.3788e-01f, -1.7786e-01f, -2.1291e-01f, -1.0432e-01f, -2.3502e-01f, -3.1885e-01f, -1.4522e-01f, 

        -1.3397e-01f, -1.6776e-01f, -2.4019e-01f, 7.5068e-01f, -2.7036e-01f, 6.7269e-01f, 5.4633e-01f, -1.5802e-01f, 

        8.9956e-01f, 1.1275e-01f, 3.4015e-01f, 1.4298e-01f, 2.6040e-02f, -2.4133e-01f, -4.4888e-01f, -1.3299e-01f, 

        -6.4777e-02f, -2.6411e-01f, -1.2365e-01f, -6.5880e-01f, -2.1424e-01f, -4.1791e-02f, -6.5509e-02f, -5.4744e-02f, 

        -6.8449e-01f, -3.1961e-02f, -2.5478e-01f, -6.7642e-01f, -3.3155e-01f, -4.1009e-01f, -1.1297e+00f, -5.2068e-01f, 

        -1.2553e-01f, 7.7117e-02f, 8.2195e-01f, -1.3635e+00f, 6.7251e-01f, -1.0578e-01f, -3.5337e-01f, -8.9612e-01f, 

        -1.9283e-01f, 2.4190e-01f, 1.4818e+00f, -1.3222e+00f, -5.0663e-01f, -3.2513e-01f, 3.5028e-02f, -1.4555e+00f, 

        -3.6044e-01f, -1.1294e+00f, -2.0613e-01f, -7.7264e-01f, -4.9501e-01f, -8.2071e-01f, -8.7190e-02f, 6.5520e-01f, 

        6.5560e-01f, -6.2149e-01f, -1.0438e+00f, -4.0476e-01f, 8.3975e-02f, 2.0084e-01f, -3.2293e-01f, -2.9201e-01f, 

        -1.1096e+00f, -1.3246e-01f, -6.1353e-01f, -2.8738e-01f, -3.0931e-01f, -8.9773e-01f, -1.2894e-01f, -4.3698e-01f, 

        -5.3593e-01f, 4.8783e-02f, 3.7071e-01f, -2.6001e-01f, 5.5218e-02f, -2.2628e-01f, -4.3637e-01f, 1.2084e-01f, 

        4.0277e-02f, -6.3406e-01f, 9.5695e-01f, -2.2831e-01f, -1.2021e-01f, 6.6046e-02f, -1.8868e-01f, -1.0308e-01f, 

        -1.4905e-01f, -1.2347e-01f, -5.7951e-02f, 2.8652e-01f, 8.9306e-02f, -8.0877e-01f, -4.6315e-01f, -8.2539e-01f, 

        -7.7646e-01f, -3.1980e-01f, -5.6011e-01f, -9.9726e-02f, -2.8230e-01f, -9.2015e-01f, 3.3841e-01f, 3.5770e-01f, 

        -9.5720e-01f, 1.0512e+00f, -6.2224e-01f, -1.0705e+00f, -1.3151e+00f, 1.7904e-01f, 1.2381e-01f, 5.4642e-01f, 

        8.9644e-02f, -1.3179e+00f, 3.3503e-01f, 4.5825e-01f, 2.5393e-01f, 3.8210e-01f, 4.8098e-02f, -2.7613e-01f, 

        -3.6020e-01f, 5.6748e-01f, 3.2831e-01f, -4.6163e-02f, -4.1921e-01f, -4.9353e-01f, -3.4758e-01f, -5.2037e-01f, 

        5.2457e-01f, -1.2640e+00f, -8.1864e-01f, -3.7970e-01f, 2.6819e-01f, -4.5897e-01f, -2.1232e-01f, 8.8340e-02f, 

        -1.8508e-01f, -5.4879e-01f, -5.5721e-01f, -6.5171e-01f, -3.9837e-01f, 9.4324e-01f, 1.2940e+00f, -6.3055e-01f, 

        -4.8230e-02f, -4.4566e-01f, -4.0605e-01f, 3.6535e-01f, -4.5443e-01f, 4.9583e-02f, -7.2304e-01f, 1.3356e+00f, 

        -2.3494e-01f, 5.9871e-01f, 4.9211e-02f, -9.9444e-01f, -3.8787e-01f, 1.1934e-01f, 3.3615e-01f, -6.6975e-01f, 

        -4.3161e-01f, 1.3213e-01f, 6.6321e-01f, -7.0932e-01f, -4.3143e-01f, 1.7937e-01f, -5.4813e-01f, -4.8615e-01f, 

        -7.3211e-01f, -3.4611e-01f, 4.1185e-01f, -8.1565e-01f, -5.3172e-01f, -3.7993e-01f, -5.0423e-01f, -2.6974e-01f, 

        -4.0665e-01f, -1.4114e-01f, -2.4558e-01f, -3.7198e-01f, -1.2993e-01f, 1.9440e-01f, -2.3099e-01f, -6.2761e-01f, 

        2.4751e-01f, -1.7821e-01f, 3.3186e-01f, -2.0611e+00f, 4.0606e-01f, 6.2258e-01f, 6.9620e-01f, -5.2152e-01f, 

        1.4799e-01f, -4.5107e-01f, 1.1323e-01f, 1.7240e-01f, 1.1406e+00f, -1.0525e-01f, 2.0299e+00f, -4.1140e-01f, 

        -7.6021e-01f, 2.6703e-01f, -4.9555e-02f, 1.9491e-01f, -3.1202e-01f, -1.1259e+00f, -2.3980e-01f, 2.7485e-01f, 

        -8.3935e-01f, -2.3201e-01f, -1.8056e-01f, -2.0885e-01f, -2.0926e-01f, -2.6007e-01f, 6.2983e-02f, -2.3756e-02f, 

        -2.4543e-01f, 1.4204e-02f, 3.0789e-01f, 2.4839e-01f, -6.6606e-01f, 1.8115e-01f, -5.5215e-01f, -1.7616e-03f, 

        1.2300e-01f, -3.4209e-01f, -2.4840e-01f, 5.2824e-01f, -1.7214e+00f, -7.0184e-01f, 1.0598e+00f, -1.8408e+00f, 

        4.2123e-01f, 1.1950e+00f, 7.4816e-01f, 4.4089e-01f, -1.6305e-01f, -2.7511e-01f, -3.1094e-01f, -8.8625e-02f, 

        -1.8168e-01f, -1.5164e-01f, -5.5438e-02f, -1.5957e-01f, -3.1142e-01f, -5.8628e-01f, 1.2942e+00f, -6.9631e-01f, 

        -1.3250e+00f, 1.7316e-01f, 2.4920e-01f, -7.6862e-01f, 2.5119e-02f, 5.4481e-01f, -1.0464e+00f, 9.2681e-02f, 

        -7.7871e-01f, -1.0140e+00f, -1.3242e+00f, -2.8249e-01f, 1.5841e-01f, -1.7804e-01f, -8.8770e-01f, -6.8510e-02f, 

        -2.3954e-01f, -2.4455e-01f, 6.2600e-02f, -2.9614e-01f, -2.3029e-01f, -2.0544e-01f, -2.9602e-01f, -1.7322e-01f, 

        -7.1177e-02f, -2.6280e-02f, 5.1558e-03f, -9.1362e-03f, -6.3913e-02f, -8.2500e-03f, -2.3411e-02f, -5.1873e-02f, 

        -1.2125e-01f, 6.6604e-03f, -5.0678e-02f, 3.4215e-03f, 2.7633e-02f, -1.0406e-01f, -6.2091e-02f, -4.9404e-02f, 

        -7.6391e-02f, -1.2928e-01f, -3.5864e-02f, -9.0657e-02f, -1.1474e-01f, -1.4860e-01f, -6.5772e-02f, -1.1298e-01f, 

        -5.3108e-02f, -1.4536e-01f, -8.3422e-02f, -2.7180e-01f, -5.3690e-01f, -4.3012e-01f, -6.8539e-02f, -1.7272e-01f, 

        -7.7350e-02f, -2.0309e-01f, -3.4410e-02f, -1.7793e-01f, -1.1747e-02f, -1.2036e-01f, -1.2921e-01f, -5.5942e-02f, 

        -1.4171e-02f, -6.0452e-02f, -1.2652e-01f, -1.1503e-01f, -1.2350e-01f, -8.2796e-02f, -5.8718e-02f, 4.3842e-02f, 

        -7.0091e-02f, 2.7775e-02f, -5.4134e-03f, -8.8567e-02f, -1.1873e-01f, -5.2384e-02f, -6.5165e-02f, -1.9297e-01f, 

        -1.0400e-01f, -1.9379e-02f, 1.5785e-03f, -6.9112e-02f, -4.6279e-02f, -1.0570e-01f, -3.9139e-02f, 4.2276e-02f, 

        -1.1424e-01f, -1.1478e-02f, -2.0581e-01f, -1.2736e-01f, -1.3784e-01f, -8.0141e-02f, -1.1542e-01f, -1.6549e-01f, 

        -3.9357e-02f, 3.4820e-02f, -1.1466e-01f, -1.8159e-02f, -8.8040e-02f, -9.0142e-02f, -8.6596e-02f, -9.2954e-02f, 

        -9.3226e-02f, -7.0976e-02f, -2.4771e-02f, -7.8094e-02f, -5.4197e-02f, -9.0368e-02f, -1.3086e-01f, -1.0269e-01f, 

        -1.4152e-01f, -1.7569e-01f, -1.5011e+00f, -1.2319e+00f, -7.7770e-01f, -8.1663e-01f, -5.5246e-01f, -5.2917e-01f, 

        -8.5118e-01f, -5.7662e-01f, -2.0547e-01f, -2.6717e-02f, -1.9535e-02f, -2.7500e-02f, -3.5572e-02f, -1.1995e-01f, 

        1.4185e-02f, -1.1092e-01f, -1.1327e-01f, -4.1184e-02f, -2.1237e-01f, -1.0599e-01f, -1.5325e-01f, -1.6289e-02f, 

        -1.1047e-01f, -5.3502e-02f, -3.4750e-02f, -1.7994e-01f, -1.3441e-01f, -9.9671e-02f, -5.4120e-02f, -5.0024e-02f, 

        -1.2028e-01f, -6.4485e-02f, -1.1121e-01f, -6.7040e-02f, -1.3173e-01f, -1.6420e-01f, 8.2212e-02f, 6.7149e-02f, 

        9.7680e-02f, 3.6937e-02f, -2.3404e-04f, 4.7157e-03f, -3.2867e-02f, 1.5109e-02f, -1.6088e-02f, 3.2271e-02f, 

        3.4950e-02f, -1.5940e-01f, -1.4373e-01f, -1.9784e-02f, -1.0186e-02f, -4.2865e-02f, -1.5155e-01f, -9.9058e-02f, 

        -1.1947e-01f, -1.4865e-01f, -1.2996e-01f, -1.8142e-03f, -1.0535e-01f, -6.7705e-02f, -6.9062e-02f, -8.9984e-02f, 

        -4.9591e-02f, -5.4357e-02f, -1.3904e-02f, -1.7571e-01f, -9.3852e-02f, -6.8806e-02f, -2.4636e-02f, -9.5640e-02f, 

        -2.4631e-03f, -3.3303e-02f, -3.5371e-02f, -3.4059e-02f, -3.1475e-02f, -1.2703e-01f, -1.0256e-01f, -1.5783e-01f, 

        -1.8656e-01f, -1.6484e-01f, -1.2549e-01f, -8.3433e-02f, -1.0498e-01f, -9.6010e-02f, -3.6094e-02f, -8.5930e-02f, 

        -1.3716e-01f, 2.1921e-02f, -4.6752e-02f, -1.0251e-01f, -7.0285e-02f, -1.4823e-01f, -6.0059e-02f, -6.2155e-02f, 

        -1.1943e-01f, -1.2034e-01f, -4.8113e-02f, -1.0732e-01f, -1.5055e-01f, -4.7615e-02f, -3.9507e-02f, -1.1467e-01f, 

        -3.4251e-02f, -2.6145e-02f, -8.7794e-02f, -1.3340e-01f, -1.3778e-01f, -1.4512e-01f, -6.4274e-02f, -6.6222e-02f, 

        -1.4409e-01f, -3.8867e-03f, -1.0329e-01f, -1.0919e-01f, -9.4165e-02f, -1.7943e-01f, -1.6214e-01f, -4.3827e-02f, 

        -9.6726e-02f, -1.0606e-02f, -4.8136e-02f, 6.6388e-02f, -1.1426e-01f, -9.4771e-02f, -1.2691e-01f, 1.0797e-02f, 

        -1.4500e-01f, -1.4525e-01f, -7.8776e-02f, -5.3581e-02f, -1.0048e-01f, -4.8749e-02f, -1.0217e-01f, -1.1144e-01f, 

        -4.8527e-02f, -5.5404e-02f, -1.2526e-01f, -9.3271e-02f, -9.5896e-04f, -4.5068e-02f, -4.3213e-02f, -1.2635e-02f, 

        -1.1621e-01f, -1.2292e-01f, -4.9380e-02f, -4.4429e-02f, -6.2135e-02f, -1.6016e-01f, -9.5366e-02f, -6.2704e-03f, 

        -1.2641e-01f, -1.0018e-01f, -1.7679e-01f, -5.8137e-02f, -1.0578e-01f, -5.6747e-02f, -2.3587e-02f, -5.2659e-02f, 

        -9.9890e-02f, -9.5522e-02f, -1.4474e-01f, -9.2724e-02f, -5.3221e-03f, -9.8086e-02f, -8.1214e-02f, -3.1619e-02f, 

        -3.0462e-02f, -5.2771e-02f, -1.7254e-02f, -4.1324e-02f, -1.2263e-01f, -6.2927e-02f, -1.0424e-01f, -1.5481e-01f, 

        -6.1810e-02f, -6.0321e-02f, -4.4698e-02f, -2.1408e-02f, -1.3782e-01f, -1.3233e-01f, -4.1222e-02f, -1.3246e-01f, 

        -1.0102e-01f, -1.2782e-01f, -4.3240e-02f, -8.5703e-02f, -3.2732e-02f, -1.1132e-01f, -5.5995e-02f, 1.5434e-01f, 

        -6.5126e-02f, -1.0138e-01f, -4.5382e-02f, -2.4147e-02f, -6.5279e-04f, -8.7050e-02f, -1.0238e-01f, -1.3078e-01f, 

        -1.6518e-01f, -7.6640e-02f, -1.0888e-01f, -1.4397e-01f, -1.9691e-01f, -1.0119e-01f, -1.2809e-01f, -1.8375e-01f, 

        -1.1366e-01f, -1.1245e-01f, -4.3528e-02f, -9.4367e-02f, -1.4686e-01f, -1.0943e-01f, -1.6491e-01f, -8.6276e-02f, 

        -1.1200e-01f, -7.8718e-02f, -1.8955e-01f, -1.7625e-01f, -1.6783e-01f, -2.1647e-01f, -1.7237e-01f, -1.8306e-01f, 

        -1.0683e-01f, -1.4322e-01f, -1.5480e-01f, -8.9922e-01f, -1.4297e+00f, -6.0395e-01f, -6.5412e-01f, -3.5258e-01f, 

        -4.3819e-01f, -7.6058e-01f, -3.8846e-01f, -4.6949e-01f, -1.6659e-01f, -8.1216e-02f, -1.0953e-01f, -1.4936e-01f, 

        -1.2593e-01f, -1.4225e-01f, -9.1185e-02f, -9.9432e-02f, -1.1187e-01f, -1.9233e-01f, -2.3462e-01f, -1.7937e-01f, 

        -2.6539e-01f, -2.1153e-01f, -1.9135e-01f, -2.0563e-01f, -9.3158e-02f, -8.3344e-02f, -1.8347e-01f, -1.6087e-01f, 

        -2.0606e-01f, -1.4331e-01f, -7.9070e-02f, -1.2731e-01f, -9.8021e-02f, -1.7809e-01f, -1.1390e-01f, -9.7514e-02f, 

        -1.7933e-01f, -1.5528e-01f, -1.7213e-01f, -2.1382e-01f, -2.2026e-01f, -1.9796e-01f, -2.2321e-01f, -2.4451e-01f, 

        -1.2929e-01f, -1.5596e-01f, -1.7196e-01f, -1.8635e-01f, -7.3020e-02f, -1.6384e-01f, -9.9708e-02f, -1.6877e-01f, 

        -1.6452e-01f, -9.3248e-02f, -8.0463e-02f, -1.4286e-01f, -1.0547e-01f, -1.3365e-01f, -1.2570e-01f, -8.9595e-02f, 

        -8.3453e-02f, -1.7460e-01f, -2.2149e+00f, -1.0895e+00f, -1.1267e+00f, -1.3836e+00f, -1.2006e+00f, -1.4666e+00f, 

        -1.2837e+00f, -9.5482e-01f, -1.4292e+00f, -7.7965e-02f, -1.3173e-01f, -6.5084e-02f, -1.3578e-01f, -1.2620e-01f, 

        -1.5341e-01f, -8.4549e-02f, -1.2387e-01f, -9.3523e-02f, -3.2373e-01f, -2.8070e-01f, -2.1728e-01f, -4.1700e-01f, 

        -2.1139e-01f, -1.3496e-01f, -1.8898e-01f, -1.9109e-01f, -9.9717e-02f, -1.2034e-01f, -1.7301e-01f, -1.1086e-01f, 

        -1.6236e-01f, -2.1384e-01f, -1.0379e-01f, -1.5942e-01f, -1.9209e-01f, -9.7673e-02f, -4.1410e-02f, -6.0049e-02f, 

        -8.0475e-03f, -1.4606e-01f, -1.0490e-01f, -1.2612e-01f, -3.2025e-02f, -1.4356e-01f, -1.0982e-01f, -2.0740e-01f, 

        -1.7420e-01f, -1.4170e-01f, -1.3925e-01f, -1.3313e-01f, -1.6717e-01f, -1.5862e-01f, -1.6493e-01f, -1.3438e-01f, 

        -2.1431e-01f, -1.4225e-01f, -2.9508e-01f, -1.7364e-01f, -1.4134e-01f, -8.3249e-02f, -1.2534e-01f, -6.9951e-02f, 

        -1.0409e-01f, -2.1048e-01f, -1.5953e-01f, -1.8618e-01f, -2.0123e-01f, -2.0418e-01f, -1.5284e-01f, -1.6663e-01f, 

        -1.4064e-01f, -1.9408e-01f, -2.2817e-01f, -1.1734e-01f, -1.2115e-01f, -1.6734e-01f, -1.2731e-01f, -1.5588e-01f, 

        -1.8393e-01f, -1.3094e-01f, -1.9748e-01f, -3.3064e-01f, -2.1181e-01f, -1.7766e-01f, -1.6548e-01f, -9.1803e-02f, 

        -1.5548e-01f, -2.0945e-01f, -1.2087e-01f, -1.5594e-01f, -8.2182e-02f, -1.6828e-01f, -1.6059e-01f, -7.3554e-02f, 

        -1.1563e-01f, -1.5755e-01f, -9.8560e-02f, -1.0379e-01f, -2.0327e-01f, -1.5295e-01f, -2.0603e-01f, -1.6253e-01f, 

        -1.5710e-01f, -1.9449e-01f, -1.7435e-01f, -1.2195e-01f, -1.6415e-01f, -2.1643e-01f, -1.2832e-01f, -1.7901e-01f, 

        -1.8136e-01f, -1.4969e-01f, -6.7094e-02f, -1.7105e-01f, -1.5476e-01f, -1.0080e-01f, -1.9904e-01f, -1.6074e-01f, 

        -1.1861e-01f, -6.8652e-02f, -1.3174e-01f, -1.2527e-01f, -1.6445e-01f, -1.1580e-01f, -4.8370e-02f, -1.4159e-01f, 

        -1.3789e-01f, -1.3099e-01f, -2.0008e-01f, -9.1747e-02f, -1.5918e-01f, -1.8652e-01f, -7.2891e-02f, -1.4973e-01f, 

        -2.1437e-01f, -1.1389e-01f, -6.9242e-02f, -7.4167e-02f, -1.1155e-01f, -8.9382e-02f, -7.5247e-02f, -1.2604e-01f, 

        -1.7057e-01f, -1.2498e-01f, -2.1534e-01f, -1.8218e-01f, -2.2676e-01f, -1.8796e-01f, -1.9874e-01f, -1.9854e-01f, 

        -1.6281e-01f, -2.5367e-01f, -1.5705e-01f, -1.8535e-01f, -1.4480e-01f, -1.4867e-01f, -9.2090e-02f, -7.8388e-02f, 

        -1.3472e-01f, -1.2279e-01f, -1.2245e-01f, -7.6699e-02f, -8.3969e-02f, -1.2879e-01f, -8.4614e-02f, -1.5477e-01f, 

        -9.3787e-02f, -4.8733e-02f, -1.5004e-01f, -7.8159e-02f, -8.5660e-02f, -2.2451e-01f, -1.3659e-01f, -8.8560e-02f, 

        -1.8325e-01f, -1.5396e-01f, -1.7955e-01f, -2.3580e-01f, -1.5874e-01f, -2.2543e-01f, -1.7016e-01f, -1.4653e-01f, 

        -1.9943e-01f, -6.2095e-02f, -1.5150e-01f, -1.7504e-01f, -1.1019e-01f, -1.0202e-01f, -9.8869e-02f, -9.3305e-02f, 

        -1.1360e-01f, -1.7678e-01f, -1.8378e-01f, -8.2655e-02f, -9.8343e-02f, -1.5456e-01f, -1.0912e-01f, -9.1519e-02f, 

        -7.0252e-01f, -3.1426e-01f, -2.2164e-01f, -4.0469e-01f, -4.7512e-01f, 6.7085e-01f, -3.8947e-01f, 6.0181e-01f, 

        2.5686e-01f, -9.1582e-02f, -2.2101e-01f, -2.1656e-01f, -1.3574e-01f, -1.5225e-01f, -1.6219e-01f, -2.2176e-02f, 

        -6.2108e-02f, -9.8620e-04f, 9.1724e-01f, -1.5789e+00f, 7.2156e-01f, -3.0371e-01f, -1.4326e+00f, 8.4128e-01f, 

        -3.9644e-01f, -6.2637e-01f, 2.2086e+00f, -2.1909e-01f, -8.6970e-01f, 3.9212e-01f, -1.7473e-01f, -1.0069e+00f, 

        -5.0718e-01f, 5.5838e-01f, -9.4921e-01f, -5.0984e-01f, -1.8773e-02f, 1.2365e+00f, -3.0626e-01f, 2.4313e-01f, 

        -1.5965e-01f, -5.4766e-01f, -3.9976e-02f, -1.3250e-01f, -1.5163e-01f, 1.4849e-01f, -8.7554e-01f, 1.2764e+00f, 

        -2.2470e+00f, 2.1207e-01f, 7.5715e-01f, -1.8582e+00f, 1.8089e-01f, 1.0576e+00f, -4.0764e-01f, 1.3901e+00f, 

        7.3632e-02f, -4.0329e-01f, -3.6671e-02f, -5.9677e-01f, 2.6214e-01f, -3.0092e-01f, -6.8839e-01f, -5.0570e-01f, 

        5.2995e-01f, -3.7608e-01f, -5.1048e-01f, -6.6527e-01f, 1.0317e+00f, -2.7895e-01f, 4.8944e-01f, -2.0739e-01f, 

        7.2143e-02f, -1.6884e-01f, -8.6501e-01f, 1.8119e-01f, -5.1012e-01f, -4.6661e-01f, -4.5502e-01f, -6.2700e-01f, 

        -1.9489e-01f, 7.1178e-01f, 2.5085e+00f, 8.4955e-01f, 7.9904e-01f, -1.9454e-01f, -3.4080e-01f, -3.8769e-01f, 

        -2.8569e-02f, -4.7783e-01f, -3.1764e-01f, -6.1725e-01f, -5.6413e-01f, -2.2167e-01f, -1.2458e-02f, -5.0096e-01f, 

        -7.5519e-01f, 5.8807e-02f, -5.9931e-01f, -2.0148e-02f, -2.5576e-01f, -2.8354e-01f, 2.5407e-02f, -1.2522e-01f, 

        -2.7806e-01f, -1.7490e-01f, -3.9845e-02f, 5.7735e-03f, -2.1105e-01f, 1.2175e-01f, -7.9167e-01f, -2.5962e-01f, 

        8.0563e-01f, 4.5834e-01f, 1.8984e-01f, 5.8416e-01f, -1.3040e-01f, -2.1723e-01f, 1.1480e+00f, -1.6247e+00f, 

        3.9218e-02f, 3.8955e-01f, -5.7418e-01f, 2.4839e-01f, -6.5711e-01f, -2.5913e-01f, -6.5438e-01f, -8.2842e-01f, 

        1.6552e-01f, -8.6617e-01f, -3.9884e-01f, -1.2843e-01f, -1.3843e+00f, -6.0548e-01f, -3.1088e-01f, -5.0088e-01f, 

        -1.4068e+00f, 1.1541e+00f, -2.1448e-02f, -7.7202e-01f, 1.4818e+00f, -1.3286e+00f, 4.5248e-02f, 1.9028e+00f, 

        -4.8516e-01f, -8.2509e-01f, -6.4737e-01f, -4.3259e-01f, -4.1554e-01f, 2.2020e-01f, -1.4859e-01f, -6.5149e-01f, 

        -4.3855e-02f, 4.9706e-01f, -3.2723e-01f, -4.4035e-01f, 1.7961e-01f, -5.0361e-01f, -4.8649e-01f, -1.6228e-01f, 

        -7.4419e-01f, -4.7886e-01f, -4.2050e-01f, 2.3663e-01f, 1.2186e+00f, 4.9876e-01f, -6.7644e-01f, 1.5655e-01f, 

        -4.6599e-01f, -5.9167e-01f, -1.4485e+00f, -3.0524e-01f, 1.2029e+00f, -1.6234e-01f, 8.9737e-01f, 8.6739e-01f, 

        -9.2981e-02f, 1.4050e-01f, -4.8278e-02f, -9.9846e-01f, -9.1612e-01f, 9.5269e-01f, 5.5707e-01f, -3.7190e-01f, 

        1.1500e+00f, 1.1765e+00f, -8.6131e-02f, 8.4439e-01f, 1.1469e-01f, -1.7612e-01f, 7.8188e-01f, 2.2337e-01f, 

        1.7621e-01f, 6.1257e-02f, -4.9096e-02f, -1.3147e-01f, 1.5561e-02f, 5.2550e-01f, 5.0899e-01f, -1.4134e+00f, 

        -1.3182e-01f, -4.3562e-01f, 3.7325e-02f, -4.9580e-01f, 2.4296e+00f, -7.2072e-01f, -6.3922e-01f, -7.5182e-01f, 

        -1.7227e+00f, -2.9193e-01f, 4.3160e-01f, -1.5156e+00f, -1.6250e+00f, 5.7901e-01f, -8.5650e-01f, -5.6460e-01f, 

        -2.6372e-01f, -4.1046e-02f, 7.0219e-01f, 7.0973e-01f, -4.1748e-01f, 9.2387e-01f, -1.7982e-01f, -4.7829e-01f, 

        -6.7997e-01f, -2.7375e-01f, -2.6366e-01f, -1.0727e-01f, -1.0378e-01f, -4.1241e-01f, 4.8125e-02f, -1.0526e-01f, 

        -1.8477e-01f, -5.0029e-02f, 2.6985e-01f, -1.3349e+00f, -1.0423e+00f, 5.6740e-01f, -5.6630e-01f, 4.0477e-01f, 

        -1.4153e+00f, -2.7772e-01f, 3.7452e-01f, -2.5049e-01f, -1.3734e-01f, 1.0606e+00f, 2.9929e-01f, 7.4674e-01f, 

        3.6036e-01f, -1.0760e-01f, -3.0977e-01f, 8.2284e-01f, -7.5757e-02f, -1.5677e-01f, -7.6956e-02f, -2.2439e-01f, 

        -1.2309e-03f, -1.5033e-01f, 6.4252e-02f, 2.1002e-01f, -2.5544e-01f, -7.4002e-01f, 3.7204e-01f, -3.1589e-01f, 

        6.7760e-01f, 2.3700e-03f, -1.1899e+00f, -5.7106e-01f, -2.7105e-01f, -3.3605e-01f, 6.0328e-01f, 1.1593e+00f, 

        -1.2043e-01f, 6.9840e-01f, -6.2524e-01f, -5.5462e-01f, 1.0957e+00f, -4.1265e-02f, -1.2073e+00f, -2.5161e-02f, 

        2.1000e-03f, -1.8296e-02f, 2.5894e-01f, -1.2662e-02f, 3.2311e-01f, 2.5346e-01f, -2.0539e-02f, 2.0394e-01f, 

        -2.5940e-01f, -2.4604e-01f, -1.7940e-01f, -2.7430e-01f, -2.3393e-01f, -2.5200e-01f, -2.4239e-01f, -2.5460e-01f, 

        -1.9697e-01f, -1.9758e-01f, -1.7760e-01f, -2.0596e-01f, -1.1464e-01f, -1.7596e-01f, -8.4653e-02f, -2.0145e-01f, 

        -1.0739e-01f, -1.7063e-01f, -2.4340e-01f, -1.4416e-01f, -2.2665e-01f, -2.5480e-01f, -1.3499e-01f, -1.7828e-01f, 

        -1.9165e-01f, -1.1291e-01f, -1.9948e-01f, -1.9208e+00f, -2.2297e+00f, -1.1285e+00f, -1.0637e+00f, -7.1770e-01f, 

        -7.9652e-01f, -9.9600e-01f, -5.5787e-01f, -6.4991e-01f, -1.2040e-01f, -1.1558e-01f, -1.2915e-01f, -1.0572e-01f, 

        -1.7231e-01f, -1.4388e-01f, -1.8781e-01f, -2.0095e-01f, -9.6393e-02f, -6.8657e-01f, -3.5955e-01f, -4.1777e-01f, 

        -5.8669e-01f, -4.7302e-01f, -3.9213e-01f, -5.2782e-01f, -3.7924e-01f, -3.6831e-01f, -2.6935e-01f, -2.5891e-01f, 

        -1.6134e-01f, -1.8861e-01f, -1.7428e-01f, -1.2134e-01f, -1.8601e-01f, -1.9822e-01f, -1.5520e-01f, -1.9142e-01f, 

        -2.4541e-01f, -4.8510e-01f, -3.1647e-01f, -3.9314e-01f, -2.9612e-01f, -3.6147e-01f, -3.2717e-01f, -3.6554e-01f, 

        -3.2315e-01f, -1.8306e-01f, -1.9637e-01f, -2.5808e-01f, -2.1530e-01f, -3.5494e-01f, -1.7363e-01f, -2.3596e-01f, 

        -1.9101e-01f, -1.2253e-01f, -1.5834e-01f, -1.0753e-01f, -1.4554e-01f, -1.6829e-01f, -9.9167e-02f, -1.2565e-01f, 

        -1.0135e-01f, -1.2881e-01f, -2.6464e+00f, -2.3637e+00f, -2.1301e+00f, -2.5566e+00f, -1.4255e+00f, -2.0187e+00f, 

        -2.3097e+00f, -1.1326e+00f, -1.7716e+00f, -1.5743e-01f, -9.0728e-02f, -1.1495e-01f, -1.1380e-01f, -3.2481e-02f, 

        -8.5127e-02f, -1.2445e-01f, -1.0235e-01f, -1.5689e-01f, -6.1415e-01f, -3.0692e-01f, -2.9717e-01f, -7.0785e-01f, 

        -2.4836e-01f, -2.3280e-01f, -3.0937e-01f, -2.6058e-01f, -1.6796e-01f, -2.1016e-01f, -1.4840e-01f, -1.5544e-01f, 

        -1.6284e-01f, -1.8883e-01f, -1.4784e-01f, -8.7536e-02f, -1.7157e-01f, -1.8167e-01f, -1.0316e-01f, -3.1913e-02f, 

        -1.8071e-01f, -9.7688e-02f, -8.1780e-02f, -1.0945e-01f, -1.7704e-01f, -1.6400e-01f, -2.0299e-01f, -3.2242e-01f, 

        -3.2265e-01f, -2.5659e-01f, -3.6341e-01f, -2.6672e-01f, -3.4572e-01f, -3.5987e-01f, -3.3596e-01f, -2.7579e-01f, 

        -3.9759e-01f, -2.5601e-01f, -4.4499e-01f, -1.9820e-01f, -1.4708e-01f, -2.1169e-01f, -2.1546e-01f, -1.7373e-01f, 

        -2.5347e-01f, -3.8174e-01f, -3.5569e-01f, -3.0498e-01f, -3.9526e-01f, -3.7891e-01f, -3.4167e-01f, -3.5849e-01f, 

        -2.3187e-01f, -3.5764e-01f, -2.0418e-01f, -2.2415e-01f, -2.0684e-01f, -2.4530e-01f, -1.5374e-01f, -1.5354e-01f, 

        -3.0097e-01f, -1.5741e-01f, -1.9050e-01f, -5.0903e-01f, -3.7333e-01f, -3.6117e-01f, -3.2029e-01f, -2.6925e-01f, 

        -3.5372e-01f, -2.9079e-01f, -3.4722e-01f, -3.9866e-01f, -1.9440e-01f, -1.3041e-01f, -2.8039e-01f, -2.0937e-01f, 

        -1.7136e-01f, -2.2207e-01f, -2.0957e-01f, -1.5320e-01f, -1.4047e-01f, -2.6670e-01f, -2.0173e-01f, -3.0160e-01f, 

        -2.2887e-01f, -2.9044e-01f, -2.8928e-01f, -2.3364e-01f, -2.5776e-01f, -2.2021e-01f, -1.1072e-01f, -1.0935e-01f, 

        -1.3376e-01f, -3.0677e-01f, -1.5852e-01f, -1.8923e-01f, -1.9369e-01f, -1.1020e-01f, -1.0767e-01f, -4.5393e-01f, 

        -1.9730e-01f, -2.9876e-01f, -2.6627e-01f, -3.6558e-01f, -1.9716e-01f, -3.2817e-01f, -1.9132e-01f, -1.6842e-01f, 

        -1.5695e-01f, -1.9972e-01f, -2.7243e-01f, -1.5189e-01f, -2.0360e-01f, -2.2145e-01f, -1.8964e-01f, -1.5172e-01f, 

        -1.9689e-01f, -1.8592e-01f, -1.9517e-01f, -1.1650e-01f, -1.8222e-01f, -1.2999e-01f, -8.7304e-02f, -1.8922e-01f, 

        -1.9001e-01f, -1.2986e-01f, -3.1033e-01f, -2.9575e-01f, -1.7275e-01f, -3.0502e-01f, -2.1353e-01f, -1.9798e-01f, 

        -2.2715e-01f, -2.4189e-01f, -1.8778e-01f, -2.2541e-01f, -1.6488e-01f, -1.8125e-01f, -1.0791e-01f, -1.4907e-01f, 

        -1.3750e-01f, -2.0730e-01f, -2.1015e-01f, -1.8020e-01f, -1.9840e-01f, -1.3791e-01f, -1.9499e-01f, -1.4392e-01f, 

        -6.9099e-02f, -1.9218e-01f, -1.3462e-01f, -1.4338e-01f, -1.2347e-01f, -1.8872e-01f, -2.4663e-01f, -2.6125e-01f, 

        -2.6500e-01f, -2.3156e-01f, -2.1446e-01f, -2.6658e-01f, -1.5932e-01f, -3.0097e-01f, -2.1243e-01f, -1.7858e-01f, 

        -1.4495e-01f, -1.6229e-01f, -1.2440e-01f, -1.4371e-01f, -1.1619e-01f, -1.9886e-01f, -1.4139e-01f, 1.0379e-02f, 

        -1.4450e-01f, -1.4027e-01f, -9.6899e-03f, -1.0297e-01f, -1.0297e-01f, -1.2885e-01f, -3.2732e-03f, -1.4704e-01f, 

        -7.3308e-02f, -2.8393e-02f, -1.7242e-01f, -3.9482e-02f, -8.9332e-02f, -9.6973e-02f, -9.0892e-02f, -3.5708e-03f, 

        -3.7335e-02f, -4.2481e-02f, -1.4815e-01f, -1.1980e-01f, -1.2009e-02f, -5.7328e-02f, -9.9649e-02f, 5.8165e-02f, 

        -1.2853e-01f, -7.4476e-02f, -3.5303e-01f, -4.3337e-01f, -1.3164e-01f, -9.6590e-02f, -2.7912e-01f, -4.7326e-01f, 

        -1.2659e-01f, -3.7764e-01f, -4.0961e-01f, -1.8256e-01f, -1.8321e-01f, -4.1844e-01f, -2.5869e-01f, -5.3373e-02f, 

        -1.0059e-01f, -2.4750e-01f, -1.4384e-01f, -4.3676e-02f, -7.0477e-02f, -2.3103e-02f, -1.3245e-01f, -1.0343e-01f, 

        -9.7950e-02f, -1.8677e-01f, -9.7803e-02f, -1.6633e-01f, -1.2227e-02f, -5.1389e-01f, -2.4410e-01f, -2.4997e-01f, 

        -5.9533e-01f, -4.6121e-01f, -3.1872e-01f, -2.9953e-01f, -4.6450e-01f, -2.9254e-01f, -2.0525e-01f, -1.5366e-01f, 

        -2.0596e-01f, -1.3588e-01f, -1.9017e-01f, -1.8558e-01f, -1.3614e-01f, -8.4029e-02f, -9.5177e-02f, -7.2595e-02f, 

        -6.3633e-02f, 2.4088e-01f, -2.3813e-01f, 6.6899e-02f, 1.2218e-01f, -3.5763e-01f, -2.3858e-02f, -7.7243e-02f, 

        -1.3769e-01f, -5.0991e-03f, -2.1169e-01f, -8.3455e-02f, -1.3536e-01f, -3.4323e-01f, -2.5969e-02f, -6.4623e-02f, 

        -4.3356e-02f, -6.8362e-02f, -6.7008e-02f, 1.3121e-01f, -1.8625e-01f, -8.4290e-02f, -5.7708e-02f, -7.3368e-02f, 

        -1.8816e-01f, 9.7488e-02f, -1.0284e+00f, -2.0727e+00f, -2.3903e+00f, -3.5269e+00f, -1.8224e+00f, -1.0606e+00f, 

        -2.7925e+00f, -2.1290e+00f, -2.7918e+00f, -1.3972e-01f, -1.6945e-01f, -8.4470e-02f, -7.8897e-02f, -1.8626e-01f, 

        -2.4360e-01f, 1.2785e-03f, -1.4473e-01f, -1.2211e-01f, -2.9154e-01f, -1.3784e-01f, -2.8271e-01f, -3.0666e-01f, 

        -5.0427e-01f, 1.4893e-01f, -3.0241e-02f, -4.7715e-02f, 1.2692e-01f, -3.7986e-02f, -1.4106e-01f, -7.6239e-02f, 

        -1.3071e-01f, -2.4424e-03f, -8.1896e-02f, -6.2087e-02f, -1.4041e-01f, -1.9658e-02f, -1.5451e-01f, -1.0326e-01f, 

        -2.4642e-01f, -1.0061e-01f, -1.6379e-01f, -4.7739e-01f, -1.7071e-01f, -1.6642e-01f, -1.0785e-01f, -1.2862e-01f, 

        -1.4703e-01f, -1.1675e-01f, -8.3754e-02f, -1.1624e-01f, 6.3270e-02f, -6.7298e-02f, -1.1243e-01f, -8.7556e-02f, 

        -8.1921e-01f, -4.6835e-01f, -7.0809e-01f, -8.8384e-02f, -3.0972e-01f, 6.1951e-02f, -1.9476e-01f, -2.1932e-01f, 

        -4.8459e-02f, -1.1201e-01f, -4.5684e-02f, -1.5085e-01f, -7.9509e-02f, 1.2836e-01f, -1.6309e-01f, -1.4524e-01f, 

        -2.2749e-02f, -7.7908e-02f, -1.6076e-01f, -1.8332e-01f, -2.0747e-02f, -2.8403e-01f, 6.2212e-02f, 5.9643e-02f, 

        -4.6926e-01f, -3.4603e-01f, -2.7818e-01f, -2.6125e-01f, -3.0119e-01f, -7.4948e-02f, -1.7064e-01f, -1.1877e-01f, 

        6.5150e-02f, 1.3136e-01f, 9.8206e-03f, -2.0607e-03f, -1.7632e-01f, -5.0965e-02f, -1.0549e-01f, 1.5428e-01f, 

        -6.6961e-02f, -9.0312e-02f, -3.3039e-01f, -1.0441e-01f, -1.9412e-01f, -1.1177e-01f, -8.1483e-02f, -7.0430e-02f, 

        -9.6038e-02f, -1.3150e-01f, -6.1284e-02f, -8.2874e-02f, -9.1231e-02f, -5.7375e-02f, -1.1367e-01f, -1.4514e-01f, 

        -2.4423e-01f, -4.2155e-01f, -2.5755e-01f, -4.4231e-01f, -3.7683e-02f, -2.8859e-01f, -1.1640e-01f, -6.3398e-01f, 

        -7.0802e-02f, -1.9421e-01f, -6.2549e-01f, -5.4609e-01f, -1.3484e-01f, -1.5194e-01f, -7.7990e-02f, -1.2141e-01f, 

        -1.7096e-01f, 1.3650e-01f, -3.6862e-01f, -3.0553e-01f, -1.1634e-01f, -1.7356e-02f, -1.6018e-01f, -3.7487e-01f, 

        -2.5689e-01f, -6.3143e-02f, -1.2598e-01f, -9.5704e-02f, -1.2895e-01f, -1.5199e-01f, -1.6210e-01f, -1.8549e-01f, 

        -1.0653e-01f, -1.3633e-01f, -5.1164e-01f, -3.1238e-01f, -1.8626e-01f, -4.2407e-01f, -4.1250e-01f, -5.6940e-02f, 

        -2.4842e-01f, -4.6193e-01f, -5.8860e-01f, -2.3661e-01f, -1.1555e-01f, -1.0619e-01f, -1.1269e-01f, -1.4485e-01f, 

        -1.4275e-01f, -1.2674e-02f, -1.1803e-01f, -1.1283e-01f, 1.3057e-02f, -1.1807e-01f, -8.1140e-02f, -9.2894e-03f, 

        -1.4743e-01f, -8.3663e-02f, -1.0339e-01f, 8.4364e-02f, -7.4473e-02f, -2.9858e-02f, -2.0139e-01f, -6.0236e-02f, 

        -1.5256e-01f, -1.5108e-01f, 4.1001e-01f, -3.7325e-01f, -1.2672e-01f, 7.2041e-02f, -7.5145e-02f, -2.1068e-02f, 

        -1.0152e-01f, -9.2680e-02f, -1.0509e-01f, -3.4566e-01f, -1.2624e-01f, -4.5804e-02f, -2.0271e-01f, -1.0587e-01f, 

        -3.3781e-02f, -1.8905e-01f, 1.0832e-01f, -1.1113e-01f, -1.4299e-01f, -8.6395e-02f, 1.0870e-01f, -2.6302e-02f, 

        -1.7993e-01f, -1.7962e-01f, -1.6347e-01f, -1.7536e-01f, -1.8182e-01f, -1.3583e-01f, -1.9469e-01f, -1.5631e-01f, 

        -1.1394e-01f, -5.5227e-02f, -1.4028e-01f, -1.7880e-01f, -1.0882e-01f, -1.2961e-01f, -5.7808e-02f, -1.1214e-01f, 

        -1.4747e-01f, -1.2555e-01f, -1.1200e-01f, -2.0103e-01f, -8.5700e-02f, -9.7781e-02f, -2.1195e-01f, -2.1434e-01f, 

        -1.6577e-01f, -1.0416e-01f, -1.4563e-01f, -1.2127e+00f, -1.2475e+00f, -1.0938e+00f, -6.2634e-01f, -6.4015e-01f, 

        -4.3604e-01f, -5.3543e-01f, -3.7952e-01f, -5.2237e-01f, -9.8393e-02f, -1.3756e-01f, -6.6510e-02f, -8.5359e-02f, 

        -1.0982e-01f, -7.0337e-02f, -1.4377e-01f, -1.4746e-01f, -1.1515e-01f, -3.1676e-01f, -2.5782e-01f, -1.1817e-01f, 

        -2.9540e-01f, -2.3547e-01f, -2.1405e-01f, -2.5292e-01f, -1.6462e-01f, -1.8260e-01f, -1.9050e-01f, -1.8240e-01f, 

        -1.7807e-01f, -1.5451e-01f, -1.0843e-01f, -5.8234e-02f, -7.9535e-02f, -1.4080e-01f, -1.6337e-01f, -1.6843e-01f, 

        -1.3112e-01f, -3.2881e-01f, -1.9382e-01f, -2.7067e-01f, -2.0372e-01f, -1.5330e-01f, -2.4370e-01f, -2.5759e-01f, 

        -1.7821e-01f, -1.5791e-01f, -1.0899e-01f, -1.3138e-01f, -1.8210e-01f, -1.7120e-01f, -9.2196e-02f, -1.8215e-01f, 

        -1.2487e-01f, -9.7974e-02f, -1.3932e-01f, -5.0946e-02f, -1.4329e-01f, -1.3937e-01f, -1.9399e-01f, -1.2606e-01f, 

        -9.3245e-02f, -1.3467e-01f, -2.0596e+00f, -1.5470e+00f, -1.8233e+00f, -1.7818e+00f, -1.1942e+00f, -1.2567e+00f, 

        -1.6241e+00f, -1.1157e+00f, -1.4656e+00f, -8.6586e-02f, -5.0270e-02f, -1.4054e-01f, -1.1753e-01f, -1.4381e-01f, 

        -3.6144e-02f, -7.4072e-02f, -4.9896e-02f, -9.3883e-02f, -3.7956e-01f, -2.4431e-01f, -2.6912e-01f, -3.3758e-01f, 

        -2.1741e-01f, -1.9755e-01f, -2.3601e-01f, -1.4000e-01f, -2.4868e-01f, -1.3899e-01f, -1.0831e-01f, -1.5206e-01f, 

        -1.8992e-01f, -1.3918e-01f, -1.4917e-01f, -9.6549e-02f, -5.8190e-02f, -1.1064e-01f, -1.4545e-01f, -9.8447e-02f, 

        -8.9278e-02f, -1.0918e-01f, -1.5310e-01f, -1.6125e-01f, -1.0901e-01f, -1.4964e-01f, -5.3310e-02f, -1.8516e-01f, 

        -2.3262e-01f, -2.7204e-01f, -2.0907e-01f, -1.6120e-01f, -1.9551e-01f, -1.9534e-01f, -1.4241e-01f, -2.0490e-01f, 

        -2.4932e-01f, -1.2704e-01f, -3.0517e-01f, -7.2023e-02f, -1.8684e-01f, -6.9377e-02f, -1.8366e-01f, -7.5248e-02f, 

        -1.3213e-01f, -2.8666e-01f, -1.9513e-01f, -2.5584e-01f, -1.8264e-01f, -2.2463e-01f, -1.7248e-01f, -1.6778e-01f, 

        -2.1660e-01f, -2.4341e-01f, -2.2577e-01f, -1.9311e-01f, -1.8606e-01f, -1.7699e-01f, -1.6654e-01f, -1.3832e-01f, 

        -1.5598e-01f, -8.4477e-02f, -2.0412e-01f, -3.8718e-01f, -2.4464e-01f, -3.9233e-01f, -2.6077e-01f, -1.4704e-01f, 

        -2.0135e-01f, -1.1377e-01f, -1.4524e-01f, -2.1732e-01f, -1.2449e-01f, -1.8220e-01f, -1.1778e-01f, -1.2322e-01f, 

        -8.0271e-02f, -7.5372e-02f, -1.8696e-01f, -1.1833e-01f, -1.1415e-01f, -2.2044e-01f, -9.1536e-02f, -1.0276e-01f, 

        -1.6885e-01f, -8.6235e-02f, -1.2904e-01f, -1.1202e-01f, -1.9682e-01f, -1.9030e-01f, -1.4092e-01f, -1.1280e-01f, 

        -1.6473e-01f, -2.3487e-01f, -6.5365e-02f, -8.1503e-02f, -1.4812e-01f, -1.7944e-01f, -1.0868e-01f, -2.4607e-01f, 

        -2.0240e-01f, -7.2031e-02f, -1.9223e-01f, -2.5320e-01f, -2.2203e-01f, -2.1289e-01f, -1.4822e-01f, -1.2991e-01f, 

        -1.4304e-01f, -1.3562e-01f, -1.2595e-01f, -1.5591e-01f, -1.4616e-01f, -1.5106e-01f, -6.7984e-02f, -1.0279e-01f, 

        -1.0471e-01f, -6.7865e-02f, -7.5960e-02f, -7.8417e-02f, -1.1379e-01f, -4.7334e-02f, -1.1927e-01f, -1.1883e-01f, 

        -1.2676e-01f, -1.1029e-01f, -1.3635e-01f, -1.7472e-01f, -1.5060e-01f, -1.3414e-01f, -9.6854e-02f, -8.4050e-02f, 

        -2.2877e-01f, -1.7669e-01f, -8.7310e-02f, -1.1136e-01f, -9.5903e-02f, -7.0099e-02f, -1.5082e-01f, -9.6841e-02f, 

        -1.5530e-01f, -8.8878e-02f, -1.0454e-01f, -1.1978e-01f, -9.1600e-02f, -1.1939e-01f, -8.1481e-02f, -1.2919e-01f, 

        -1.3599e-01f, -1.2900e-01f, -1.0726e-01f, -1.3911e-01f, -1.4300e-01f, -1.6242e-01f, -2.3148e-01f, -2.1133e-01f, 

        -2.4987e-01f, -2.2542e-01f, -1.8925e-01f, -1.6592e-01f, -2.2836e-01f, -2.8998e-01f, -1.6711e-01f, -1.6251e-01f, 

        -1.3449e-01f, -9.4685e-02f, -1.3401e-01f, -7.8175e-02f, -9.1228e-02f, -1.5938e-01f, -1.0541e-01f, -1.7062e-01f, 

        -9.6173e-02f, -1.2447e-01f, -1.0957e-01f, -1.6843e-01f, -8.4761e-02f, -1.7850e-01f, -1.4704e-01f, -2.1824e-02f, 

        -1.6508e-01f, -1.2006e-01f, -1.2359e-01f, -1.0405e-01f, -7.8818e-02f, -1.6294e-01f, -7.5188e-02f, -1.0872e-01f, 

        -3.1428e-02f, -1.1690e-01f, -1.2329e-01f, -1.0582e-01f, -1.1690e-01f, -3.3872e-02f, -5.0996e-02f, -1.6547e-02f, 

        -1.2240e-01f, -6.6450e-03f, -2.7508e-01f, -1.9365e-01f, -8.4809e-02f, -1.4847e-01f, -8.5840e-02f, -1.6730e-01f, 

        -1.6121e-01f, -1.1316e-01f, -1.2299e-01f, -2.5315e-01f, -1.5969e+00f, -4.9535e-01f, -1.1993e-01f, -5.2802e-01f, 

        -2.8927e-01f, -4.2484e-02f, -1.2107e-01f, -1.9371e-01f, -1.1690e-01f, -1.4067e-01f, -1.1146e-01f, -1.2180e-01f, 

        -8.7998e-02f, -1.4170e-01f, -8.0726e-02f, -1.8081e-01f, -1.1443e-01f, -8.8703e-01f, -3.7785e-01f, -5.0404e-01f, 

        -7.9712e-01f, -5.4918e-01f, -2.7339e-01f, -6.6332e-01f, -1.2682e-01f, -1.7281e-01f, -5.2304e-01f, -7.0863e-01f, 

        -2.6007e-01f, -1.4853e-01f, -1.7211e-01f, -1.6156e-01f, -6.2540e-02f, -1.5850e-01f, -1.2263e-01f, -2.2271e-01f, 

        -1.4417e-01f, -2.3107e-01f, -1.4199e-01f, -3.4200e-02f, -1.7801e-01f, -8.1580e-02f, -1.9198e-01f, -9.9975e-02f, 

        -1.8449e-01f, -1.9549e-01f, -3.7600e-01f, 6.8891e-03f, -9.5145e-02f, -4.4106e-01f, -5.4831e-02f, -1.0238e-01f, 

        -5.0429e-02f, -5.9675e-02f, -2.3946e-02f, -7.2297e-02f, -9.4371e-02f, -9.8685e-02f, -9.3519e-02f, -6.3798e-02f, 

        -2.6234e-02f, -1.3426e-01f, -2.2003e+00f, -2.4570e+00f, -1.4634e+00f, -2.0292e+00f, -1.6564e+00f, -2.4002e+00f, 

        -2.0054e+00f, -2.0504e+00f, -2.1034e+00f, -1.2431e-01f, -8.8310e-02f, -7.6388e-03f, -1.2127e-01f, -8.7033e-02f, 

        -1.0901e-01f, -9.5630e-02f, -1.2077e-01f, -4.2870e-02f, -1.1950e+00f, -2.4975e-01f, -3.8205e-01f, -8.5044e-01f, 

        -6.9349e-02f, -2.5956e-01f, -1.4047e-01f, -1.6261e-01f, -1.3969e-01f, -2.4408e-01f, -2.7023e-02f, -8.4642e-02f, 

        -1.6217e-01f, -8.3392e-02f, -5.6165e-02f, -4.7541e-02f, -3.0186e-02f, -1.2365e-01f, -1.0365e-01f, -2.2276e-01f, 

        -1.7495e-01f, -2.5218e-01f, -1.6381e-01f, -1.8310e-01f, -1.1223e-01f, -7.0583e-03f, -5.9689e-02f, -2.0844e-01f, 

        -3.5352e-01f, -2.1364e-01f, -2.3098e-01f, -1.9920e-02f, -1.4961e-01f, -4.6999e-02f, -1.0753e-01f, -1.2406e-01f, 

        -7.0242e-01f, -4.1151e-01f, -4.9659e-01f, -2.0933e-01f, -1.9478e-01f, -6.6310e-02f, -2.0338e-01f, -1.4336e-01f, 

        -4.5417e-02f, -2.4052e-01f, -1.3804e-01f, -3.0868e-01f, -8.4379e-02f, -1.4838e-01f, -2.6152e-01f, -1.3779e-01f, 

        -1.0643e-01f, -6.8618e-02f, -4.2511e-01f, -1.9384e-01f, -2.2681e-01f, -1.9595e-01f, -1.8728e-01f, -1.0120e-01f, 

        -1.4039e-01f, -1.8074e-01f, -1.9035e-01f, -1.1712e+00f, -3.9127e-01f, -6.2751e-01f, -5.6804e-01f, -4.3634e-01f, 

        -2.3901e-01f, -7.9052e-02f, -3.6246e-02f, -9.2425e-02f, -1.1352e-01f, -5.6551e-02f, -2.8492e-01f, -1.0996e-01f, 

        -4.8599e-02f, -6.5687e-02f, -1.2393e-01f, -9.3768e-02f, -5.8191e-02f, -5.3551e-02f, -1.7585e-01f, -1.1127e-01f, 

        -2.0982e-01f, -1.8668e-02f, -3.0522e-02f, -6.1983e-02f, -3.7872e-02f, -7.8759e-02f, -5.4393e-02f, -1.1614e-01f, 

        -9.8375e-02f, -2.2729e-01f, -1.9116e-01f, -2.0191e-01f, -9.0074e-02f, -1.4271e-01f, -1.7626e-01f, -6.4985e-01f, 

        -4.1556e-01f, -1.8415e-01f, -1.6574e-01f, -5.8700e-01f, -1.6975e-01f, -1.6434e-01f, -1.9975e-01f, -1.7619e-01f, 

        -1.8280e-01f, -1.5292e-01f, -2.5683e-01f, -4.3298e-01f, -2.4038e-01f, -1.4255e-01f, -1.3459e-01f, -8.6126e-02f, 

        -1.0668e-01f, -6.6026e-02f, -1.5198e-01f, -1.5326e-01f, -1.6135e-01f, -1.0803e-01f, -5.1990e-02f, -1.1418e-01f, 

        -8.3293e-02f, -1.0354e-01f, -5.1834e-01f, -4.6144e-01f, -5.1349e-01f, -2.3831e-01f, -1.4426e-01f, -1.4505e-01f, 

        -1.0786e-01f, -1.2173e-01f, -9.8292e-02f, -6.4145e-01f, -1.2549e-01f, -1.7285e-01f, -4.1881e-02f, -1.4049e-01f, 

        -1.2144e-01f, -3.7967e-02f, -4.1165e-02f, -5.3225e-02f, -1.6638e-01f, -1.4515e-01f, -1.6192e-01f, -1.1711e-01f, 

        -7.5375e-02f, -1.3272e-01f, -8.5716e-02f, -8.6441e-02f, -1.1824e-01f, -2.3800e-01f, -4.6975e-01f, -6.3677e-01f, 

        -2.5239e-01f, -2.4569e-01f, -2.7689e-01f, -5.0236e-02f, -1.6296e-01f, -7.9322e-02f, -1.5523e-01f, -1.6100e-01f, 

        -1.1607e-01f, -4.7707e-03f, -8.4231e-02f, -8.7964e-02f, -6.8994e-02f, -1.6747e-01f, -1.3235e-01f, 1.1985e-01f, 

        -7.1766e-02f, -1.0428e-01f, 2.0415e-02f, -1.5548e-01f, -1.3835e-01f, -4.6324e-02f, -1.7396e-01f, -1.5884e-01f, 

        3.1681e-01f, 2.2101e-01f, 1.3263e+00f, 5.0475e-01f, -7.2508e-01f, -8.0223e-01f, -4.4919e-02f, -3.5375e-01f, 

        -7.2143e-01f, -1.2079e-01f, -1.1406e-01f, -1.8761e-01f, -2.0026e-01f, -1.7486e-01f, -7.0787e-02f, -2.0748e-01f, 

        -2.5284e-01f, -1.9319e-01f, 1.4785e-01f, 1.3105e+00f, -7.3560e-01f, 7.5181e-01f, 3.8099e-01f, 7.3048e-01f, 

        9.2567e-01f, 2.1439e+00f, -1.6074e-01f, -8.3503e-01f, 3.3283e-01f, -1.9663e+00f, 8.5392e-02f, 2.7444e-02f, 

        -3.3809e-01f, -9.7445e-01f, -2.4202e-01f, 5.6028e-01f, -1.2293e-01f, -1.3395e-01f, -1.6978e+00f, -8.6803e-01f, 

        -1.9775e-01f, 6.5025e-01f, 7.3289e-02f, 8.0390e-01f, 1.5960e-03f, -5.9887e-01f, -1.7188e-01f, 6.1909e-01f, 

        -3.5534e-01f, -1.0901e+00f, -5.4009e-01f, -5.4491e-02f, -4.2484e-01f, -1.1252e-01f, 1.1058e+00f, -2.5352e-01f, 

        -2.0180e+00f, 6.9741e-01f, 3.8300e-01f, 2.0226e+00f, -7.3419e-01f, 1.9355e+00f, 5.3849e-01f, -9.6627e-02f, 

        5.9560e-01f, -2.7904e-01f, -8.8584e-01f, -1.0572e+00f, -1.1697e-01f, 2.2967e-02f, -4.2767e-01f, -9.2398e-01f, 

        -8.4066e-01f, 5.0146e-01f, 7.0543e-01f, -6.2987e-01f, -1.3199e+00f, -3.3641e-02f, 2.5346e-01f, 5.5258e-01f, 

        -7.2785e-01f, 1.7076e-01f, 5.5376e-01f, -1.3370e+00f, -9.9802e-01f, 1.2902e+00f, 1.5414e+00f, -3.3926e-01f, 

        6.7195e-01f, 5.5329e-01f, 2.6279e-01f, 5.7214e-02f, -2.3241e-01f, 3.9140e-03f, 6.6460e-02f, 1.8119e-01f, 

        -4.1195e-02f, 5.5001e-02f, 1.0716e-01f, -1.2952e-01f, -1.1824e-01f, -1.8007e-01f, -3.4058e-02f, -1.1722e-01f, 

        -9.0275e-02f, -1.5643e-02f, -1.3658e-01f, -1.5924e-02f, -3.1103e-01f, -2.5338e-01f, -4.5620e-01f, -2.9535e-01f, 

        -7.0827e-03f, -6.4038e-01f, -4.9814e-01f, -5.0838e-01f, 3.6200e-01f, -1.5021e+00f, -1.5159e+00f, -2.7782e-01f, 

        -4.2202e+00f, -1.8692e+00f, -1.6385e+00f, -4.4675e-02f, 2.1955e-01f, 3.1759e+00f, -7.6847e-01f, -1.0267e+00f, 

        1.2820e+00f, 1.1477e+00f, 1.5564e-03f, 1.2588e-01f, 3.4649e-01f, -8.3933e-01f, -8.1283e-01f, 3.1583e-01f, 

        2.9057e-01f, -8.2738e-01f, 7.7304e-01f, 6.7148e-01f, 2.3443e-01f, 6.6825e-01f, -6.4889e-02f, -4.4307e-01f, 

        -6.8013e-01f, 5.2577e-02f, -2.6806e-01f, -2.1700e-01f, -1.6061e-01f, -3.2282e-01f, -2.5333e-01f, -1.1696e+00f, 

        -6.2294e-01f, -1.9160e+00f, -1.0423e+00f, 2.3418e-01f, -9.1550e-01f, -1.3268e+00f, 6.1381e-02f, -8.7800e-01f, 

        7.8740e-01f, -3.6077e-03f, 5.0607e-01f, -2.7624e+00f, -6.9229e-01f, -1.8315e-01f, -3.1894e-01f, 3.1707e-01f, 

        1.7232e-01f, -7.8877e-01f, -1.3723e+00f, 3.7095e-02f, -9.4773e-01f, 4.8165e-01f, -1.3154e+00f, -2.3082e-01f, 

        5.7231e-01f, 8.4186e-01f, 6.1395e-01f, 1.4516e-02f, 9.3499e-01f, 8.9362e-01f, -1.0762e+00f, 1.3598e+00f, 

        -2.9191e-01f, -1.0274e+00f, -1.5122e-02f, -2.2701e-01f, 1.7609e-01f, 3.6811e-02f, 6.5950e-02f, -4.2788e-01f, 

        -8.2186e-03f, 7.9090e-02f, 2.3841e-01f, -3.3489e-01f, 1.4179e-01f, 7.4628e-02f, -6.0786e-01f, -1.5134e+00f, 

        3.1284e-01f, -2.5042e-01f, -7.2456e-01f, -3.0849e-01f, -1.1514e-01f, 6.5650e-02f, -1.2180e+00f, -2.3623e-01f, 

        2.0319e+00f, -1.2560e-01f, 1.8829e-01f, 1.8403e-01f, 9.5950e-01f, -4.0318e-02f, -1.6911e-01f, -2.2425e-01f, 

        -4.5475e-01f, -6.2790e-01f, -9.1120e-01f, -5.4593e-01f, -1.6943e+00f, -3.0031e-02f, 3.7797e-01f, -1.9257e-02f, 

        -2.9787e-01f, -1.5122e-01f, -6.7582e-02f, -5.4349e-02f, -2.3354e-01f, -2.4908e-01f, -1.3951e-01f, -8.4969e-02f, 

        9.7191e-02f, 1.1481e-01f, 1.6809e-01f, 6.1817e-02f, -8.8912e-02f, -2.2553e-01f, 8.0960e-01f, 2.2786e-01f, 

        1.4253e-01f, -6.5500e-01f, 7.8460e-01f, -7.9754e-01f, 1.3966e-01f, -8.3414e-01f, -3.3493e+00f, -1.4743e+00f, 

        -1.1259e+00f, 8.9174e-01f, 8.2506e-01f, -2.3474e+00f, -3.0790e-01f, -2.9195e-01f, -2.8464e-01f, -3.1782e-01f, 

        -2.4228e-01f, -2.1766e-01f, -3.2000e-01f, -2.7127e-01f, -3.2967e-01f, 7.3310e-01f, 8.9834e-01f, 2.4493e-01f, 

        -2.9656e-01f, 1.3264e+00f, -5.4959e-01f, -5.0504e-01f, -2.7059e-01f, 8.0406e-02f, -1.6476e-01f, -3.5071e-01f, 

        8.4974e-02f, 1.4361e-01f, -7.5762e-01f, -7.0512e-01f, 2.1621e-01f, -3.9488e-01f, -8.0949e-01f, -3.1224e-01f, 

        -1.8639e-01f, -1.7783e-01f, -2.4904e-01f, -1.3049e-01f, -1.8929e-01f, -2.0512e-01f, -3.4380e-01f, -2.1472e-01f, 

        -1.6200e-01f, -1.4971e-01f, -1.4946e-01f, -2.2093e-01f, -9.2108e-02f, -1.6729e-01f, -2.0222e-01f, -1.3859e-01f, 

        -1.0939e-01f, -9.4452e-02f, -1.0839e-01f, -1.6187e-01f, -1.8755e-01f, -9.7065e-02f, -1.5472e-01f, -1.3848e-01f, 

        -8.0272e-02f, -1.3284e-01f, -8.8815e-02f, -1.3755e-01f, -1.2181e-01f, -1.2706e-01f, -1.2930e-01f, -1.6373e-01f, 

        -2.2217e-01f, -9.3447e-02f, -1.8969e-01f, -1.1846e+00f, -1.5852e+00f, -1.1585e+00f, -9.0008e-01f, -6.2345e-01f, 

        -5.7354e-01f, -8.0763e-01f, -5.8558e-01f, -6.3985e-01f, -1.7438e-01f, -6.5285e-02f, -1.8489e-01f, -1.2028e-01f, 

        -6.3446e-02f, -9.6873e-02f, -1.1555e-01f, -1.3072e-01f, -7.3086e-02f, -4.2535e-01f, -3.3217e-01f, -2.8037e-01f, 

        -3.8798e-01f, -3.2851e-01f, -3.3075e-01f, -4.1333e-01f, -2.6709e-01f, -2.3209e-01f, -1.6510e-01f, -2.1093e-01f, 

        -1.9768e-01f, -2.1104e-01f, -1.0735e-01f, -1.0747e-01f, -7.2616e-02f, -1.6695e-01f, -1.1779e-01f, -2.0367e-01f, 

        -1.7202e-01f, -4.0099e-01f, -2.0710e-01f, -3.6102e-01f, -2.2429e-01f, -1.4355e-01f, -2.5235e-01f, -1.8075e-01f, 

        -1.5911e-01f, -1.8217e-01f, -1.9617e-01f, -1.5424e-01f, -1.0301e-01f, -2.1119e-01f, -1.3668e-01f, -1.4363e-01f, 

        -1.9288e-01f, -1.2565e-01f, -1.0550e-01f, -1.4598e-01f, -7.5319e-02f, -1.0633e-01f, -8.4090e-02f, -9.8156e-02f, 

        -6.2134e-02f, -1.6461e-01f, -2.6927e+00f, -1.6816e+00f, -1.8016e+00f, -2.4548e+00f, -1.2872e+00f, -1.5957e+00f, 

        -2.2930e+00f, -1.0133e+00f, -1.7664e+00f, -1.3923e-01f, 2.2969e-02f, -1.7851e-01f, -1.4030e-01f, -1.0368e-01f, 

        -1.6430e-01f, -1.2242e-01f, -1.4776e-01f, -4.1995e-02f, -4.0690e-01f, -2.0970e-01f, -1.9976e-01f, -5.9521e-01f, 

        -1.8967e-01f, -1.1717e-01f, -1.8687e-01f, -1.7653e-01f, -1.3070e-01f, -1.3109e-01f, -1.3417e-01f, -1.5973e-01f, 

        -6.9675e-02f, -1.5574e-01f, -6.5981e-02f, -2.1577e-01f, -1.5883e-01f, -4.3717e-02f, -1.3703e-01f, -1.3221e-01f, 

        -8.2233e-02f, -1.7959e-01f, -1.3995e-01f, -1.7268e-01f, -9.0909e-02f, -1.0981e-01f, -1.2513e-01f, -2.6833e-01f, 

        -2.5991e-01f, -3.2043e-01f, -3.4230e-01f, -2.3410e-01f, -2.9597e-01f, -2.5198e-01f, -2.7889e-01f, -1.6275e-01f, 

        -3.0239e-01f, -1.4724e-01f, -3.5851e-01f, -1.9170e-01f, -1.5428e-01f, -7.1585e-02f, -1.3836e-01f, -1.6008e-01f, 

        -1.9127e-01f, -2.6009e-01f, -2.6897e-01f, -2.5041e-01f, -3.0394e-01f, -2.3152e-01f, -1.3569e-01f, -3.2968e-01f, 

        -2.9125e-01f, -3.0435e-01f, -2.6218e-01f, -1.2267e-01f, -1.6645e-01f, -1.2806e-01f, -1.2912e-01f, -8.4134e-02f, 

        -1.2335e-01f, -1.6164e-01f, -9.5276e-02f, -4.3517e-01f, -2.1764e-01f, -5.2845e-01f, -2.7217e-01f, -2.6267e-01f, 

        -2.8325e-01f, -2.1391e-01f, -2.2222e-01f, -2.4776e-01f, -2.2116e-01f, -1.5874e-01f, -1.6774e-01f, -1.5471e-01f, 

        -1.1581e-01f, -1.8916e-01f, -1.3817e-01f, -1.5324e-01f, -1.0540e-01f, -2.0513e-01f, -1.8527e-01f, -1.5490e-01f, 

        -2.2180e-01f, -1.3953e-01f, -2.1301e-01f, -1.8813e-01f, -2.4246e-01f, -1.7279e-01f, -1.7809e-01f, -1.0806e-01f, 

        -1.7593e-01f, -1.7904e-01f, -3.9437e-02f, -1.1671e-01f, -1.8508e-01f, -8.3418e-02f, -1.6548e-01f, -2.1799e-01f, 

        -2.3471e-01f, -1.2192e-01f, -2.1966e-01f, -3.4592e-01f, -1.2485e-01f, -1.0726e-01f, -8.7856e-02f, -1.3439e-01f, 

        -2.1138e-01f, -1.4697e-01f, -1.6859e-01f, -1.6695e-01f, -1.2468e-01f, -1.2636e-01f, -8.8045e-02f, -1.0501e-01f, 

        -1.5822e-01f, -8.8269e-02f, -7.2774e-02f, -1.8755e-01f, -1.2311e-01f, -1.2389e-01f, -1.2510e-01f, -1.7845e-01f, 

        -1.6255e-01f, -1.4551e-01f, -2.6304e-01f, -2.3205e-01f, -2.5448e-01f, -2.4252e-01f, -6.7908e-02f, -1.3460e-01f, 

        -1.9374e-01f, -2.8406e-01f, -1.8151e-01f, -1.7796e-01f, -1.5687e-01f, -1.6869e-01f, -1.0500e-01f, -1.0558e-01f, 

        -7.0055e-02f, -1.1208e-01f, -8.8033e-02f, -1.5913e-01f, -1.5865e-01f, -9.0469e-02f, -1.2250e-01f, -7.1712e-02f, 

        -7.9270e-02f, -1.3483e-01f, -1.3904e-01f, -1.5552e-01f, -7.2203e-02f, -1.8333e-01f, -3.0232e-01f, -2.3548e-01f, 

        -2.9936e-01f, -5.0897e-02f, -1.9128e-01f, -9.1146e-02f, -2.1187e-01f, -6.7792e-02f, -7.9810e-02f, -5.8433e-02f, 

        -9.5438e-02f, -1.2868e-01f, -1.2503e-01f, -1.4131e-01f, -1.1515e-01f, -1.8347e-01f, -1.4456e-01f, -1.9762e-01f, 

        -1.6617e-01f, -1.5747e-01f, -1.2772e-01f, -7.9883e-02f, -4.8932e-02f, -4.1583e-02f, -1.3126e-01f, -9.5402e-02f, 

        9.4211e-02f, 9.2727e-02f, -1.0415e-01f, 2.9168e-01f, -1.1699e-01f, -1.2268e-01f, -1.2041e-01f, 2.8438e-01f, 

        -7.7801e-03f, -2.0181e-02f, -7.0365e-02f, -7.2125e-02f, -3.1219e-02f, 1.0107e-01f, -6.6794e-02f, -1.1716e-01f, 

        -2.5847e-02f, -1.9099e-01f, -1.2420e+00f, -1.4213e-01f, -1.0829e-01f, -7.8044e-02f, -2.3590e-02f, 5.1901e-01f, 

        3.5330e-01f, -3.8413e-01f, -4.3361e-01f, 1.4686e-01f, -5.9336e-01f, 1.2171e-01f, 3.1587e-01f, 1.2713e+00f, 

        7.1205e-01f, 1.0938e-01f, -1.4567e+00f, 3.6750e-01f, 3.0924e-01f, 1.0129e-01f, 9.7555e-02f, 1.6971e-01f, 

        1.0308e-01f, -8.4221e-03f, -1.4539e-01f, -3.6361e-01f, -2.4672e-01f, -8.3365e-01f, -1.4684e-02f, 6.7741e-01f, 

        -1.8558e+00f, -9.7000e-01f, -2.5138e-01f, -1.9984e+00f, -1.0876e+00f, -5.7700e-01f, 1.6514e-01f, 4.9050e-01f, 

        2.4909e-01f, 9.2507e-01f, 5.3770e-01f, 1.0749e+00f, -4.1865e-01f, -8.0453e-01f, 4.3339e-02f, 5.9415e-01f, 

        8.5361e-01f, 1.1165e-01f, 4.1146e-01f, -5.5472e-01f, 3.7650e-02f, -8.9082e-01f, -1.9689e-01f, 5.4758e-02f, 

        4.3281e-01f, 1.6647e-01f, 3.8616e-01f, 4.0224e-01f, 5.3225e-01f, -7.6574e-01f, -8.9285e-01f, -3.7234e-02f, 

        4.1850e-01f, 6.3512e-02f, 1.1645e-01f, 7.0162e-02f, 5.2698e-01f, -2.5934e-02f, 8.7971e-01f, 5.6333e-02f, 

        -3.6365e-01f, -1.1543e-01f, -4.5672e-01f, 1.3027e-01f, -4.2672e-01f, -3.8978e-01f, -3.3239e-01f, -5.4773e-01f, 

        -5.9615e-01f, 6.7343e-02f, -2.9717e-01f, -1.9214e-01f, 5.8146e-01f, -3.1385e-01f, 4.4343e-02f, -7.1429e-02f, 

        -2.5803e-01f, -2.7845e-01f, -3.4827e-01f, -3.0454e-01f, 9.3153e-01f, 1.4050e-01f, 7.3589e-01f, 2.6476e-01f, 

        -8.3978e-01f, 5.0666e-01f, -1.2617e+00f, -1.0044e+00f, 1.2523e-01f, 3.3997e-02f, -1.1283e+00f, 5.9840e-01f, 

        -9.9759e-01f, -1.2642e+00f, -9.9232e-01f, -5.1554e-02f, -7.9435e-01f, -4.5569e-01f, -1.5139e+00f, 6.9033e-01f, 

        -6.7302e-02f, 2.8958e-01f, 3.0261e-01f, -8.2341e-01f, -3.6379e-01f, -3.6559e-01f, -7.0320e-01f, -3.5616e-01f, 

        -3.6049e-01f, -4.5177e-02f, -2.0998e-02f, -2.9243e-01f, -3.8485e-01f, -1.4661e-01f, 1.3493e-01f, -4.4760e-01f, 

        3.9871e-01f, 2.1782e-01f, -2.4683e+00f, 1.2040e-01f, -8.3561e-01f, -5.3476e-02f, -3.3790e-01f, -6.1407e-01f, 

        -1.2124e+00f, 6.4214e-01f, 3.6449e-02f, 1.5976e-01f, 1.5577e-01f, 6.5385e-02f, -1.4319e-01f, -8.1578e-01f, 

        -3.5109e-01f, 4.7897e-02f, -1.6583e-01f, 2.4805e-01f, -8.6478e-01f, -2.2360e-01f, -7.8380e-01f, -9.3365e-01f, 

        3.4607e-01f, -3.5360e-02f, -6.5658e-01f, 1.9742e+00f, -3.5665e-01f, 5.0299e-01f, 9.9349e-01f, 4.5557e-02f, 

        1.3807e+00f, -6.1903e-02f, -4.0239e-01f, 1.3988e-01f, -9.1014e-01f, 1.0039e-01f, -3.7047e-01f, -6.0029e-01f, 

        -6.6057e-01f, -5.3912e-01f, 3.8234e-01f, -2.2576e-01f, -1.0973e-01f, 1.6678e-01f, -2.3060e-01f, 6.5053e-02f, 

        2.9212e-01f, 3.7261e-02f, 3.2753e-01f, -1.3116e-01f, -2.1025e-01f, -6.8822e-02f, -1.0457e+00f, -9.1980e-01f, 

        -9.8536e-02f, 4.9258e-01f, -3.3273e-01f, -1.1419e+00f, -5.3734e-01f, -2.3445e-01f, -1.0384e+00f, -8.9297e-01f, 

        -1.2339e+00f, -8.2826e-01f, -2.4365e+00f, -1.0204e+00f, -5.8188e-01f, 1.5781e-02f, -8.4318e-01f, 7.1111e-04f, 

        -2.2637e-01f, 1.5652e+00f, 3.0910e-01f, 1.6542e+00f, 1.3705e+00f, -1.8000e+00f, 3.8006e-01f, 7.6231e-01f, 

        5.4085e-01f, -5.1025e-02f, -2.1714e-02f, -7.2682e-02f, 3.0947e-02f, -7.8921e-02f, -9.4793e-02f, -5.5348e-02f, 

        -2.1949e-01f, -1.5673e-01f, 2.0640e-01f, 1.3380e+00f, -6.5401e-01f, 1.7249e-01f, 2.5695e-01f, 7.8156e-01f, 

        -9.9595e-01f, -1.2005e+00f, -1.4641e+00f, 8.1268e-01f, 5.5187e-02f, -4.5049e-02f, -4.2375e-01f, -4.7020e-01f, 

        -3.9549e-01f, -1.0886e-01f, -1.3036e-01f, -9.1667e-01f, -3.8891e-02f, -1.6629e-01f, -1.1798e-01f, -2.4495e-01f, 

        -6.0889e-03f, -1.1930e-01f, -1.1911e-01f, -2.5887e-01f, -1.3743e-01f, 1.3866e+00f, 5.4967e-01f, 2.2886e-01f, 

        7.4239e-01f, -2.1213e-01f, 1.7108e-01f, -6.7725e-01f, -7.5567e-01f, -5.9753e-02f, 1.5352e-01f, -2.6216e-01f, 

        -1.0166e+00f, 1.1618e+00f, 1.1398e-02f, -1.0452e+00f, 7.1120e-01f, 3.3420e-01f, -3.0966e+00f, -1.5988e-01f, 

        -1.9765e-01f, -2.5643e-01f, -1.8794e-01f, -2.4256e-01f, -2.8403e-01f, 2.3828e-03f, -1.3770e-01f, -2.5019e-01f, 

        -1.0851e+00f, -2.0932e-01f, -1.8875e-01f, -7.9332e-02f, -1.8708e-01f, -1.1771e-01f, -7.9124e-02f, -2.4601e-02f, 

        -9.7369e-02f, -9.3026e-02f, -7.3751e-02f, -1.3360e-01f, -3.9086e-02f, -6.9973e-02f, -1.3778e-01f, -6.4800e-02f, 

        -8.0201e-02f, -1.7304e-01f, 1.3869e-01f, 1.5417e-01f, -1.2548e-01f, -8.5974e-02f, -1.9163e-01f, -2.9273e-01f, 

        -1.2462e-01f, -1.4944e-01f, -7.0738e-02f, 5.2958e-02f, -5.2593e-01f, -1.7564e-01f, 1.4702e+00f, -3.1804e-01f, 

        -1.6998e-01f, 1.3060e+00f, -1.7099e-01f, -1.5214e-01f, -2.1456e-01f, 3.9311e-01f, -1.9344e-02f, -1.6485e-01f, 

        -2.3291e-01f, -1.4330e-01f, -7.0725e-02f, -1.5839e-02f, -8.7059e-02f, 9.0850e-02f, -9.0496e-02f, -3.8883e-01f, 

        4.1692e-01f, -4.9235e-01f, -6.1421e-01f, -8.7351e-01f, -5.3763e-01f, -8.2323e-01f, -4.1144e-01f, 1.3628e+00f, 

        2.5602e-01f, 1.0668e+00f, -4.3977e-01f, -5.6845e-02f, -9.8661e-02f, -1.8943e-01f, -1.2895e-01f, -1.7955e+00f, 

        -9.6788e-01f, -4.9083e-01f, -2.5817e-01f, -5.7907e-02f, -2.3042e-01f, -4.5830e-01f, -1.5911e-01f, 2.6395e-02f, 

        4.2760e-01f, -1.5251e-02f, -2.5693e-01f, -3.8774e-01f, -1.2334e-01f, -8.7032e-01f, -5.1697e-02f, -1.1766e-01f, 

        -3.6005e-01f, -1.6401e-01f, 7.3549e-01f, -1.2401e-01f, -3.2037e-01f, -3.6912e-01f, -1.5875e-01f, -5.4733e-02f, 

        -8.1470e-02f, -6.2536e-02f, -8.1494e-01f, -3.1328e+00f, -1.1230e+00f, -4.1652e+00f, -1.8320e+00f, -2.1704e+00f, 

        -4.1752e+00f, -1.7538e+00f, -2.1297e+00f, -3.8612e-01f, -1.5217e-01f, -2.9746e-02f, 1.2149e-01f, -2.1463e-01f, 

        -1.1019e-01f, 1.0118e-02f, -5.6669e-02f, -1.2836e-01f, 1.3650e-01f, -7.4959e-01f, -2.7910e-01f, -3.1558e-01f, 

        -4.4524e-01f, -2.9565e-01f, -2.0018e-01f, -2.3043e-01f, -1.5265e-01f, 1.1673e-01f, -8.8507e-01f, -3.1609e-02f, 

        -1.1914e-01f, -1.0921e-01f, -1.7270e-01f, -4.7005e-02f, -1.4164e-01f, -4.7856e-02f, 5.0319e-02f, -1.0029e-01f, 

        -7.7431e-02f, -5.7226e-02f, -7.2517e-02f, -5.6846e-01f, -1.2451e-01f, -7.7056e-02f, -7.5427e-02f, -2.1600e-01f, 

        -1.7917e-01f, -2.3479e-01f, -2.9338e-01f, -1.4668e-01f, -8.0190e-02f, -6.3479e-02f, -3.6725e-01f, -2.3306e-01f, 

        -2.0046e+00f, -9.7321e-01f, -1.0768e+00f, -3.5943e-01f, -1.3628e-01f, -1.2080e-01f, -2.4016e-01f, -1.2182e-01f, 

        -3.0320e-01f, 8.3165e-01f, 1.5763e-01f, 3.0383e-02f, 7.5913e-01f, -9.5347e-02f, -9.5172e-02f, 3.5564e-01f, 

        8.0006e-02f, -1.7470e-01f, -1.9064e-01f, -2.6680e-01f, -1.6951e-01f, -3.2137e-01f, -1.5322e-01f, -1.3751e-01f, 

        -4.1088e-01f, -2.0768e-01f, -9.7137e-02f, 1.3324e-02f, -6.1144e-02f, -3.3273e-01f, -4.3089e-01f, -3.3462e-01f, 

        -1.7718e-01f, 2.8605e-01f, -1.8047e-01f, -3.3839e-01f, -3.8910e-01f, -3.9891e-02f, -2.7784e-01f, -9.9402e-03f, 

        -1.9368e-01f, -1.6053e-01f, -2.1730e-01f, -2.2083e-01f, -1.0132e-01f, 5.3062e-02f, 1.0952e-01f, -8.7883e-02f, 

        -3.7614e-03f, -2.4341e-01f, -4.6377e-02f, 2.6162e-01f, -2.5353e-01f, -6.7708e-02f, -7.6913e-01f, -1.8973e-01f, 

        -1.2589e-02f, -4.6411e-01f, -1.6109e-01f, -1.7558e-01f, -1.4843e-01f, -1.4780e-01f, -1.2761e-01f, -3.9298e+00f, 

        7.9918e-01f, -2.7766e-01f, -7.6692e-01f, -5.7938e-01f, -8.0887e-02f, -3.9040e-01f, -3.1274e-02f, -2.6903e-02f, 

        -2.1819e-01f, -9.7112e-02f, -4.9212e-02f, -4.0361e-01f, -8.2484e-02f, -2.9843e-01f, -6.2225e-02f, -1.8731e-01f, 

        -1.2069e-01f, -1.3079e-01f, -1.2121e-01f, -7.7786e-02f, -1.1380e-01f, -1.0854e-01f, -5.8031e-02f, -1.0541e-01f, 

        -2.0342e-02f, -1.0048e-01f, -1.2747e+00f, -2.2299e-01f, -8.3291e-01f, -6.4385e-01f, -2.1588e-01f, -8.7442e-02f, 

        -3.7391e-01f, -3.8397e-01f, -3.5111e-01f, -1.2411e+00f, -4.3010e-01f, 5.5271e-02f, -7.4325e-02f, -1.0552e-01f, 

        -1.0627e-01f, -1.5765e-02f, -2.8130e-02f, -5.7985e-02f, -1.5834e-01f, -1.1710e-01f, -9.1953e-03f, -1.0153e-01f, 

        -1.0326e-01f, -6.4462e-02f, -2.5543e-03f, -1.4641e-01f, -2.1547e-01f, -3.6802e-01f, -2.5986e-01f, -3.2318e-02f, 

        -2.8020e-02f, -2.5738e-01f, -1.8086e-01f, -3.5064e-01f, -2.2223e-01f, -6.5633e-02f, -2.9028e-01f, -1.2947e-01f, 

        -1.0984e-01f, -2.2206e-01f, -2.0682e-01f, -2.0129e-01f, -1.6699e-01f, -1.8199e-01f, -6.4661e-02f, -1.7287e-01f, 

        -2.8056e-01f, -1.6365e-01f, -1.0018e-01f, -4.6818e-02f, -6.6567e-02f, -5.7236e-02f, -8.2953e-03f, -1.4421e-01f, 

        -9.2587e-02f, -7.4550e-02f, -1.3504e-01f, -9.9165e-02f, -1.8621e-01f, -1.8496e-01f, -1.8990e-01f, -1.3271e-01f, 

        -1.4514e-01f, -1.6846e-01f, -1.1789e-01f, -1.4553e-01f, -6.1705e-02f, -1.2909e-01f, -7.6016e-02f, -1.2209e-01f, 

        -1.0584e-01f, -8.7633e-02f, -2.0777e-01f, -1.7957e-01f, -1.9138e-01f, -1.0097e-01f, -1.0483e-01f, -1.7205e-01f, 

        -2.0638e-01f, -1.1368e-01f, -1.2530e-01f, -1.0732e+00f, -1.0150e+00f, -7.5091e-01f, -5.7329e-01f, -5.7901e-01f, 

        -5.7760e-01f, -6.9888e-01f, -4.0305e-01f, -4.6976e-01f, -1.2422e-01f, -1.0738e-01f, -1.1666e-01f, -9.0711e-02f, 

        -1.0817e-01f, -9.2767e-02f, -9.1791e-02f, -1.7573e-01f, -1.6927e-01f, -2.3827e-01f, -1.9108e-01f, -2.2629e-01f, 

        -3.1010e-01f, -2.9361e-01f, -2.3393e-01f, -2.9616e-01f, -1.9331e-01f, -2.1390e-01f, -1.3139e-01f, -1.3947e-01f, 

        -1.6745e-01f, -1.9644e-01f, -1.5879e-01f, -1.3732e-01f, -1.5217e-01f, -1.3013e-01f, -1.8423e-01f, -6.9053e-02f, 

        -1.1417e-01f, -1.6505e-01f, -1.9580e-01f, -2.4119e-01f, -2.1016e-01f, -1.4371e-01f, -2.1028e-01f, -1.6341e-01f, 

        -1.4654e-01f, -8.9939e-02f, -2.3448e-01f, -2.1781e-01f, -1.2779e-01f, -2.2830e-01f, -2.0332e-01f, -8.7393e-02f, 

        -1.4250e-01f, -1.2473e-01f, -7.0757e-02f, -1.9037e-01f, -1.9363e-01f, -1.1988e-01f, -7.8368e-02f, -9.9341e-02f, 

        -1.6733e-01f, -9.3363e-02f, -1.8911e+00f, -1.0428e+00f, -1.2422e+00f, -1.4306e+00f, -1.1724e+00f, -1.4692e+00f, 

        -1.4930e+00f, -1.0517e+00f, -1.4893e+00f, -1.4598e-01f, -5.4025e-02f, -7.5074e-02f, -1.7180e-01f, -1.6445e-01f, 

        -6.4557e-02f, -1.3202e-01f, -1.1617e-01f, -1.6691e-01f, -3.3910e-01f, -2.2089e-01f, -1.7267e-01f, -4.0930e-01f, 

        -1.9766e-01f, -2.1352e-01f, -2.1140e-01f, -1.4119e-01f, -1.5503e-01f, -1.1753e-01f, -1.3963e-01f, -8.9016e-02f, 

        -1.0082e-01f, -1.4499e-01f, -1.6018e-01f, -7.9522e-02f, -1.6103e-01f, -1.6956e-01f, -7.7141e-02f, -1.3067e-01f, 

        -1.4833e-01f, -1.2883e-01f, -8.1479e-02f, -1.6878e-01f, -7.0438e-02f, -9.0722e-02f, -1.0456e-01f, -2.2345e-01f, 

        -1.9639e-01f, -1.4296e-01f, -2.6903e-01f, -2.4955e-01f, -1.6765e-01f, -1.9007e-01f, -2.3549e-01f, -2.0820e-01f, 

        -1.5765e-01f, -1.9646e-01f, -2.6451e-01f, -1.6556e-01f, -1.8141e-01f, -1.8162e-01f, -1.1134e-01f, -1.0131e-01f, 

        -1.3986e-01f, -2.0616e-01f, -1.9446e-01f, -1.7190e-01f, -1.6968e-01f, -1.7357e-01f, -1.9386e-01f, -2.0927e-01f, 

        -1.7862e-01f, -2.6111e-01f, -1.3906e-01f, -1.2784e-01f, -1.2753e-01f, -1.6545e-01f, -9.2034e-02f, -1.5017e-01f, 

        -2.0859e-01f, -1.3541e-01f, -1.7939e-01f, -3.1498e-01f, -1.2765e-01f, -2.2488e-01f, -1.3934e-01f, -2.0401e-01f, 

        -2.0078e-01f, -2.3386e-01f, -2.2131e-01f, -1.5089e-01f, -1.5676e-01f, -1.2028e-01f, -1.9429e-01f, -1.9065e-01f, 

        -1.3310e-01f, -1.1421e-01f, -2.0764e-01f, -2.0561e-01f, -1.7032e-01f, -1.2706e-01f, -2.2351e-01f, -1.2626e-01f, 

        -1.1761e-01f, -1.7813e-01f, -1.5343e-01f, -1.7466e-01f, -1.1384e-01f, -1.5864e-01f, -1.2874e-01f, -1.4840e-01f, 

        -1.4326e-01f, -2.2657e-01f, -1.3236e-01f, -1.6288e-01f, -1.1774e-01f, -1.2964e-01f, -1.9205e-01f, -1.7535e-01f, 

        -1.2878e-01f, -7.8959e-02f, -1.0570e-01f, -1.5681e-01f, -1.3114e-01f, -1.4783e-01f, -8.2548e-02f, -1.6109e-01f, 

        -1.2084e-01f, -1.4783e-01f, -1.2181e-01f, -1.2378e-01f, -1.9176e-01f, -2.0181e-01f, -1.9079e-01f, -1.5958e-01f, 

        -1.3635e-01f, -1.1878e-01f, -7.6390e-02f, -1.7410e-01f, -1.0899e-01f, -7.9398e-02f, -1.7500e-01f, -1.7120e-01f, 

        -1.1513e-01f, -1.5801e-01f, -1.9521e-01f, -2.4157e-01f, -1.7070e-01f, -2.2389e-01f, -1.1003e-01f, -1.8320e-01f, 

        -2.4108e-01f, -2.4043e-01f, -1.5791e-01f, -9.9646e-02f, -1.1654e-01f, -1.4773e-01f, -1.6997e-01f, -1.0115e-01f, 

        -1.2959e-01f, -1.4942e-01f, -1.4720e-01f, -1.2904e-01f, -8.4063e-02f, -1.4535e-01f, -1.5205e-01f, -9.1845e-02f, 

        -1.7101e-01f, -1.2346e-01f, -7.8229e-02f, -7.8568e-02f, -1.8490e-01f, -1.6341e-01f, -1.1445e-01f, -1.1534e-01f, 

        -2.1667e-01f, -2.1468e-01f, -1.7251e-01f, -6.2165e-02f, -1.4951e-01f, -2.5397e-01f, -1.2371e-01f, -8.1733e-02f, 

        -1.8863e-01f, -1.1825e-01f, -1.0028e-01f, -1.4432e-01f, -1.6688e-01f, -1.6163e-01f, -1.8372e-01f, -9.0727e-02f, 

        -1.2521e-01f, -1.2747e-01f, -1.3433e-01f, -1.3091e-01f, -3.7367e-02f, -1.0757e-01f, -8.0798e-02f, -8.5841e-02f, 

        -1.7658e-01f, -5.1948e-01f, -1.1323e-01f, 4.5533e-02f, -1.5195e-01f, -1.3914e-01f, -3.4286e-01f, 1.0005e-01f, 

        -8.9233e-02f, 1.4293e-02f, -2.2900e-01f, -2.5157e-01f, -1.8800e-01f, -1.0625e-02f, -5.9408e-02f, -6.6264e-02f, 

        -9.1308e-02f, -1.6388e-01f, -1.3392e+00f, 1.2044e+00f, 8.8316e-01f, -5.3777e-01f, 1.9409e-01f, 2.7334e-01f, 

        -8.4893e-01f, -4.8841e-01f, 1.3040e-01f, 2.2668e-01f, -9.1610e-02f, 7.3771e-01f, -1.1856e-01f, 1.3653e+00f, 

        -1.0467e-01f, -3.7715e-01f, -4.2725e-01f, 3.1370e-01f, -1.1471e+00f, 1.5721e-01f, 1.0377e+00f, 1.0279e-01f, 

        7.8298e-01f, 8.8918e-03f, -2.2390e-01f, -3.9046e-01f, 1.6062e-01f, 4.2603e-01f, 9.8371e-01f, -3.4895e-02f, 

        -1.0293e+00f, -1.2257e+00f, -4.0393e-01f, -3.4171e-01f, -7.5262e-01f, 2.2484e-01f, -2.5704e-01f, -3.6646e-02f, 

        1.0302e+00f, 3.8417e-02f, -2.1509e-01f, -7.2633e-01f, -3.4141e-01f, 1.3298e+00f, 9.2311e-01f, -1.1381e+00f, 

        -2.0282e+00f, 3.2035e-01f, 1.3754e-01f, -1.7340e-01f, 4.2628e-02f, -9.8931e-01f, 3.4239e-01f, 4.9419e-02f, 

        -5.1421e-01f, -1.9834e-01f, -8.5910e-02f, -2.3440e-01f, 2.7011e-03f, -2.0170e-01f, -1.6881e-02f, -9.0677e-02f, 

        -2.0287e-01f, -1.6076e+00f, 3.4836e-01f, 1.3156e+00f, 6.8855e-01f, 8.1585e-01f, -3.0833e-01f, 3.4887e-01f, 

        -9.6752e-01f, 1.0204e+00f, -9.6630e-01f, -8.9963e-01f, -6.8208e-01f, -7.0043e-01f, -8.3693e-01f, -9.5735e-01f, 

        -7.9647e-01f, -4.4263e-01f, -1.4885e+00f, -2.0814e-01f, 5.9342e-02f, 1.1934e-01f, -2.2644e-01f, 2.6885e-01f, 

        -1.9878e-01f, -2.1564e-01f, -4.7478e-02f, 2.8682e-02f, -1.8638e-01f, 3.6666e-01f, -3.1163e-01f, 1.4027e+00f, 

        1.2856e-01f, -6.8955e-02f, -8.3481e-01f, 4.0043e-01f, 7.3348e-02f, -7.1858e-01f, -1.7142e+00f, -4.4252e-03f, 

        -2.1695e-01f, -1.0197e-01f, 4.1643e-02f, 1.7870e-01f, 1.0018e-01f, -4.9462e-02f, 4.2278e-01f, -2.7372e-01f, 

        -3.5021e-01f, -7.7109e-01f, 1.4336e-02f, 1.7498e-01f, 9.8491e-03f, -2.2465e-01f, 1.0613e+00f, -5.4166e-02f, 

        -1.3113e+00f, 1.4445e-01f, -1.4983e-01f, 2.9851e-02f, -2.8032e-02f, -2.0396e-01f, -8.6381e-02f, -6.0015e-02f, 

        -6.2322e-01f, -2.9587e-02f, -5.4771e-01f, 1.5494e-01f, -1.0658e+00f, -1.2419e-01f, -7.3724e-02f, -3.6029e-01f, 

        -1.6642e-01f, -4.4440e-01f, -2.5391e-01f, 5.5812e-01f, 6.5758e-02f, 8.4964e-01f, 1.1137e-02f, 2.7984e-01f, 

        -1.2499e-01f, 1.7563e-01f, 1.1327e+00f, -2.5109e-03f, -1.8257e-01f, -1.9160e-01f, -5.1638e-03f, -1.0153e-01f, 

        -7.2218e-01f, -2.5181e-01f, -2.1814e-01f, -6.7896e-01f, 4.7158e-01f, 1.1038e+00f, 1.1676e+00f, 5.0531e-01f, 

        2.9027e-03f, 8.0276e-01f, 1.1380e+00f, -1.3472e-01f, -4.7644e-01f, -2.3574e-01f, -4.2601e-01f, -6.3162e-01f, 

        -1.0151e-01f, 9.1066e-02f, -2.7532e-01f, 1.3773e-01f, 7.8052e-03f, -2.5235e-01f, 1.4762e-01f, 3.4200e-01f, 

        9.8177e-02f, 1.6826e-01f, -1.1783e-01f, 4.8529e-02f, -4.1421e-01f, 1.3532e-01f, 5.6260e-01f, -5.4699e-01f, 

        -4.3434e-02f, 7.9379e-01f, -5.7601e-02f, 2.1220e-02f, -7.4125e-01f, 5.3381e-01f, 9.2377e-01f, -1.6811e-01f, 

        -5.9024e-01f, -1.2010e+00f, -3.9797e-01f, 1.3449e-01f, 2.2653e-01f, -1.0728e+00f, -1.0277e-01f, 2.0018e-02f, 

        -3.5594e-01f, -1.6956e-01f, -1.3230e+00f, -4.3440e-01f, 1.2989e-01f, -4.7539e-01f, 9.7880e-01f, 6.8923e-01f, 

        4.2538e-01f, -3.1040e-01f, -2.0568e-01f, -6.8456e-02f, 1.7283e-02f, -1.5036e-01f, 2.9936e-02f, -6.9539e-02f, 

        -1.8259e-01f, 1.0245e-01f, -1.1500e+00f, 4.0739e-01f, 4.3101e-01f, -6.2259e-01f, 3.5266e-01f, 4.5685e-01f, 

        -9.4535e-01f, -1.0349e+00f, -5.3056e-01f, -9.1601e-01f, -3.8069e-01f, 6.7432e-02f, 1.0232e+00f, -3.4668e-02f, 

        1.6755e-01f, 1.3744e+00f, 3.0411e-01f, -4.6068e-01f, -1.2033e-01f, 1.0116e-01f, 7.1871e-02f, -1.6781e-01f, 

        2.5269e-02f, -9.2330e-02f, -1.3640e-01f, 5.2030e-03f, -6.2961e-02f, 1.8215e+00f, -8.1774e-01f, 9.7535e-02f, 

        5.9190e-01f, -2.8072e-01f, 1.6772e-01f, 5.7121e-01f, 2.3859e-02f, -2.2490e-01f, -5.6224e-01f, 4.4300e-01f, 

        2.2938e-03f, -7.5596e-01f, -4.1740e-01f, -3.8680e-01f, 2.2283e-01f, -4.9206e-02f, -1.7369e+00f, -3.0407e-01f, 

        -1.3378e-01f, -1.9429e-01f, 1.5728e-01f, -2.0076e-01f, -3.2299e-01f, -2.0676e-01f, 2.2926e-03f, -1.1866e-01f, 

        -1.3163e-02f, 1.2154e-01f, 2.1295e-01f, -2.3909e-01f, 3.2748e-01f, -2.0169e-01f, -2.2500e-01f, -8.1229e-02f, 

        -9.7870e-03f, 2.8219e-02f, -1.6545e-01f, -2.8868e-02f, -8.1014e-02f, 8.5583e-02f, 3.8373e-03f, 1.3089e-01f, 

        -5.4734e-02f, -1.7756e-01f, 3.2827e-01f, -4.8994e-02f, -1.7643e-01f, 2.8879e-01f, 7.3349e-01f, -2.1307e-01f, 

        1.5716e-01f, 2.2395e-01f, 2.6407e-01f, -1.5501e+00f, -3.6439e-01f, -5.7242e-01f, -3.2860e-02f, -2.5234e-01f, 

        1.4360e-01f, 2.6656e-01f, -8.1671e-01f, -5.6008e-01f, -3.5723e-01f, -1.5848e-01f, -4.1414e-01f, -6.3820e-01f, 

        -1.2363e-01f, -1.7575e-01f, 2.2081e-01f, -1.0970e-01f, -3.3510e-01f, 1.1339e-01f, 4.5493e-01f, -1.1327e+00f, 

        -7.2764e-01f, -4.5191e-01f, 2.7428e-01f, -4.4018e-01f, 9.9511e-02f, -3.2169e-01f, -4.1251e-01f, -6.3704e-01f, 

        -9.1215e-01f, -8.4767e-01f, -2.0625e-01f, -7.1234e-02f, -2.0556e-01f, -1.0205e+00f, -7.4539e-03f, -7.8254e-01f, 

        -3.6144e-01f, 4.1807e-01f, 4.4535e-01f, 4.0238e-01f, 3.3736e-01f, -6.9679e-01f, -5.6982e-01f, -2.5841e-01f, 

        -1.0334e+00f, -2.5275e-02f, -7.2949e-01f, -1.6048e-01f, -1.5905e-01f, -8.3989e-01f, -4.9634e-01f, -4.6403e-01f, 

        -1.4160e+00f, -1.1495e-01f, -1.6642e-01f, -2.4900e-01f, -8.1320e-01f, -2.9358e-01f, -3.8074e-01f, 5.2001e-02f, 

        -1.1910e-01f, 1.8349e-02f, -3.9516e-01f, -8.5674e-02f, -2.5474e-01f, -1.3082e-01f, -4.5051e-01f, -5.8412e-01f, 

        -5.9426e-01f, -9.0440e-01f, 5.3695e-01f, -1.8041e-02f, 1.5853e-01f, -7.4661e-02f, 1.7178e-01f, -7.5859e-02f, 

        -1.4841e-01f, 3.2187e-01f, -2.5047e-01f, -1.1788e-01f, -1.3895e+00f, 4.1924e-01f, -1.0059e+00f, -2.5121e-01f, 

        -1.1327e+00f, -1.2293e-01f, 8.0730e-01f, 2.9816e-01f, -6.2542e-01f, -4.9409e-01f, -6.5644e-01f, -8.8665e-02f, 

        3.6409e-01f, -6.6652e-02f, -2.3686e-01f, 2.3977e-01f, -6.9212e-01f, -6.0676e-01f, -1.7125e-01f, -4.6469e-01f, 

        -8.0793e-02f, -1.9728e-01f, 7.0959e-01f, -5.3831e-01f, 1.3962e+00f, 4.8545e-01f, 9.9474e-01f, 4.6766e-01f, 

        -1.2918e-01f, 5.2953e-01f, -1.8002e-01f, 1.1280e+00f, 1.3839e-01f, -7.3614e-01f, -3.9962e-01f, 1.1292e-02f, 

        -9.8385e-01f, 2.7527e-01f, -1.6345e-01f, -3.3381e-01f, -6.0393e-01f, 8.0354e-02f, 2.0040e-01f, 6.3854e-02f, 

        -3.8048e-01f, -3.8233e-01f, -2.4547e-01f, -1.0766e-01f, -2.7690e-01f, -7.2228e-02f, -2.7024e-02f, -4.1243e-01f, 

        1.5075e-01f, -7.3609e-02f, -5.7828e-01f, 2.0969e-02f, 1.4703e-01f, 1.1782e+00f, -5.2860e-01f, 1.0715e-01f, 

        -4.7877e-01f, -8.6596e-01f, -4.2590e-01f, -6.8825e-01f, -4.1672e-01f, 1.4994e-01f, -1.0153e+00f, -1.9714e-01f, 

        4.7326e-01f, 2.8799e-03f, -3.7139e-01f, 1.5177e-01f, -3.3643e-02f, 5.1773e-01f, -5.6717e-01f, -4.2029e-01f, 

        -4.8356e-01f, -2.6014e-01f, -6.9941e-01f, -2.6291e-01f, 7.3897e-03f, 2.8312e-01f, -7.8090e-02f, -2.4123e-01f, 

        -1.6776e-01f, 3.1632e-01f, -1.3833e-01f, 1.3211e-01f, -8.2810e-02f, -1.2999e-02f, -8.4328e-01f, 1.2571e+00f, 

        -6.5535e-02f, 5.8833e-01f, -3.5467e-02f, -2.3362e-01f, 1.2619e-01f, -3.6012e-01f, -1.8353e+00f, 2.6404e-01f, 

        1.0699e-01f, -4.2747e-01f, 9.6511e-01f, 6.2815e-01f, -2.3938e-01f, 8.2034e-01f, 2.0352e+00f, -8.1636e-01f, 

        3.4505e-01f, -1.4321e-01f, -4.4616e-01f, -1.0861e+00f, 2.6080e-01f, -1.8970e-01f, -2.4298e+00f, -9.8852e-01f, 

        5.9567e-01f, -1.6287e-02f, -1.8191e-01f, -1.0395e-02f, 1.3309e-01f, -1.4468e-01f, 1.3029e-01f, 5.0890e-02f, 

        -9.0630e-02f, -4.6421e-03f, 3.8688e-01f, -5.8019e-01f, -5.9195e-01f, 5.8624e-02f, 2.4183e+00f, 3.9225e-01f, 

        -1.0696e+00f, 2.8836e-01f, 5.1907e-01f, -2.8201e-01f, -1.9942e-01f, -4.2499e-01f, -9.7139e-01f, 2.1520e-02f, 

        1.2162e-01f, -4.4470e-01f, 3.3033e-02f, 2.5286e-02f, -3.6800e-02f, 9.4808e-02f, 1.3695e-01f, 1.4408e-01f, 

        5.7242e-03f, -1.7536e-01f, 1.5377e-02f, 1.7435e-01f, -1.0197e-01f, 2.4537e-01f, -5.0855e-01f, 3.1858e-01f, 

        -9.2012e-03f, -3.4018e-01f, 4.0363e-01f, 3.4076e-01f, -9.4803e-03f, 2.9187e-01f, 7.4098e-01f, 8.3993e-01f, 

        4.3480e-01f, -8.1722e-01f, -5.6476e-01f, -7.7035e-03f, -9.7917e-01f, -5.4372e-01f, -1.2490e+00f, 4.5168e-01f, 

        4.5784e-01f, 2.1503e-01f, 1.5503e-02f, 7.0911e-02f, 1.7809e-01f, -1.4532e-01f, 4.1209e-01f, 3.8645e-03f, 

        -3.7632e-02f, 4.9023e-01f, 2.1736e-01f, -5.6914e-01f, 7.8709e-01f, 2.7620e-01f, -4.8463e-01f, 3.4050e-01f, 

        7.5958e-01f, -8.6621e-02f, -1.4945e-01f, -6.1537e-02f, -2.1181e-01f, -1.9196e-02f, -1.8269e-01f, 1.1259e-01f, 

        -3.8100e-02f, 6.2190e-02f, -4.6472e-01f, 8.3828e-01f, 8.8766e-01f, 1.5584e-01f, -1.5999e-02f, 7.3765e-01f, 

        -4.2768e-02f, 7.0910e-01f, 1.6964e+00f, 2.7956e-01f, 3.3487e-02f, 1.1047e-01f, 7.2322e-01f, 1.0476e+00f, 

        -3.8025e-01f, 1.6489e+00f, 1.5223e+00f, 1.0011e+00f, -2.7994e-02f, 9.0571e-01f, 1.0810e+00f, -8.1012e-01f, 

        4.2293e-01f, -1.5285e-01f, 5.6850e-01f, 3.4452e-01f, -5.4735e-02f, 1.7049e-01f, 8.9613e-01f, 5.2390e-01f, 

        6.1427e-01f, 9.6879e-01f, 7.7113e-01f, 4.5672e-01f, 7.8074e-01f, 5.2329e-01f, -7.0791e-01f, -8.1099e-01f, 

        6.9360e-01f, 4.7362e-01f, 3.9453e-01f, -1.5658e-01f, 9.4738e-01f, 4.0019e-01f, 6.8984e-01f, -9.7067e-01f, 

        1.2963e-01f, 9.7946e-03f, -1.4432e+00f, -9.9296e-02f, -4.8050e-01f, -5.7600e-01f, -2.8724e-01f, 8.2177e-01f, 

        2.7781e-02f, -1.7630e-01f, -1.9978e-01f, -7.0298e-01f, -1.1132e-01f, -8.2611e-01f, -1.8010e-01f, -1.3643e-02f, 

        -1.1053e+00f, -5.1602e-01f, 1.1522e-01f, 1.7068e+00f, -4.4776e-01f, 3.9408e-01f, -1.9551e-01f, -1.4505e-01f, 

        -7.9764e-01f, 6.7092e-01f, -4.5177e-01f, -7.9222e-01f, -8.3827e-01f, -1.1313e-01f, -4.1545e-01f, 7.0113e-02f, 

        -2.6394e-01f, -7.6812e-01f, -6.0765e-01f, -6.6594e-02f, -1.6389e-02f, -1.0022e-01f, -4.1637e-01f, -1.3666e-02f, 

        -2.0340e-01f, 1.4654e-01f, -6.8699e-02f, -3.9402e-01f, -1.9854e-01f, -8.1277e-01f, 4.3780e-01f, 5.8137e-01f, 

        4.2353e-01f, 2.5922e-02f, 1.9126e-01f, 6.2031e-01f, 9.2072e-01f, 3.8661e-02f, 6.2214e-01f, -5.6883e-01f, 

        -4.0634e-01f, -6.9309e-01f, -9.9260e-01f, -1.3389e+00f, -6.6994e-01f, -4.2837e-01f, -9.6451e-01f, -4.5547e-01f, 

        2.6261e-01f, -4.7804e-01f, 1.6325e-01f, 3.2779e-01f, -3.2967e-01f, 4.5039e-02f, -2.4863e-01f, -4.6422e-01f, 

        8.2503e-01f, 1.1295e+00f, -3.2817e-01f, 1.2833e+00f, 9.6711e-01f, 2.5749e-01f, 5.0604e-01f, 2.8169e-01f, 

        -2.1039e-01f, -4.0173e-01f, -4.6834e-01f, -1.2920e+00f, -4.7015e-01f, -3.4304e-01f, -5.5018e-01f, 1.1789e-01f, 

        5.9545e-01f, -5.7923e-01f, -6.9590e-01f, 2.6379e-01f, -3.0507e-01f, 6.0638e-01f, 4.3940e-01f, 6.4688e-01f, 

        1.1714e+00f, 7.2831e-01f, -1.2463e-01f, -9.1201e-01f, 2.6967e-01f, -9.4937e-01f, 2.0136e-01f, 4.1098e-01f, 

        -1.3146e+00f, -5.4461e-01f, -4.3267e-02f, 7.7565e-01f, 4.1814e-01f, 3.0577e-01f, -9.9456e-01f, 1.9701e-01f, 

        -7.3013e-02f, -6.3497e-01f, 3.6126e-01f, 1.2867e-01f, -7.9858e-01f, 7.5131e-01f, -3.5926e-01f, 1.6341e+00f, 

        4.9542e-01f, -8.0343e-01f, -7.5650e-01f, 3.4358e-01f, 6.9282e-01f, -2.7577e-01f, -3.2332e-02f, 4.0888e-01f, 

        3.9263e-01f, 1.4274e-01f, 2.4296e-01f, 2.9465e-01f, 1.4321e-01f, 2.1060e-01f, -6.9429e-01f, -1.5240e+00f, 

        9.9580e-01f, -1.1546e+00f, -1.9372e+00f, -3.5909e-01f, -1.0560e+00f, -1.5875e+00f, 4.3737e-01f, -5.8766e-02f, 

        -3.4093e-01f, -2.8526e-01f, -1.8359e+00f, -1.1600e+00f, -1.4333e-01f, 3.6954e-01f, 4.7578e-01f, 5.9506e-01f, 

        8.0437e-01f, -9.1313e-02f, -8.1315e-01f, 9.7008e-01f, -4.3154e-02f, 1.4319e-01f, 1.3024e+00f, 5.0740e-02f, 

        -1.6398e+00f, -2.6595e-01f, 5.4422e-02f, -1.0047e-01f, -3.1276e-01f, -1.3875e-01f, -5.6518e-02f, -1.2336e-01f, 

        8.5880e-02f, -4.0362e-03f, -1.4523e+00f, -9.0542e-01f, -1.9529e-01f, -2.6480e-01f, -1.6908e-01f, -5.1018e-02f, 

        -4.8823e-01f, 2.5844e-01f, 3.2400e-01f, 9.0249e-01f, 2.4159e-01f, 1.1409e+00f, -1.3528e+00f, -7.2173e-01f, 

        9.7156e-01f, -8.8840e-01f, -2.1924e+00f, 9.2361e-01f, 5.7279e-03f, 3.6366e-02f, -6.9538e-02f, 1.6968e-02f, 

        1.7455e-01f, 8.8058e-02f, -1.7956e-01f, 3.7105e-02f, -7.4897e-03f, -4.1443e-01f, -1.6040e-01f, -1.1214e-01f, 

        -5.2643e-01f, 1.4840e-01f, 1.7980e-01f, -1.0400e+00f, 3.1983e-01f, -4.0198e-01f, -6.0105e-01f, 1.5435e-01f, 

        -1.7407e+00f, -1.7066e+00f, -3.2761e-01f, -5.0462e-02f, -9.7261e-01f, -6.7689e-01f, -1.3063e+00f, 3.6136e-01f, 

        7.1608e-02f, 1.3503e-01f, 1.3965e-01f, 1.1685e-01f, 1.3827e-01f, 1.5185e-01f, 2.4486e-01f, 1.8480e-01f, 

        2.9951e-01f, -1.0252e-01f, -9.1844e-04f, 2.1069e-01f, -1.0084e-01f, -3.1219e-01f, -3.6402e-01f, -8.0918e-02f, 

        -1.0793e-01f, 5.0122e-02f, -2.2709e-02f, -8.9724e-03f, 1.4178e-01f, 1.1590e-01f, 8.6799e-02f, 4.8137e-02f, 

        1.8315e-01f, 1.0088e-01f, 4.3061e-01f, -6.7451e-02f, 9.5248e-02f, 2.0196e-01f, 3.5064e-01f, -2.5852e-01f, 

        -5.5283e-01f, -4.0774e-01f, -4.6445e-01f, -7.5796e-02f, -5.8276e-01f, -6.4229e-01f, -4.9611e-01f, -3.9777e-01f, 

        -7.3919e-01f, -1.4382e+00f, -6.6522e-01f, -3.4717e-01f, 9.4874e-03f, -6.2736e-02f, 1.1531e-02f, -7.3467e-02f, 

        2.9166e-01f, -1.1588e-01f, 1.5933e-01f, -8.6542e-02f, 1.1165e-01f, -3.9325e-01f, -3.4067e-01f, -8.2486e-02f, 

        2.6573e-01f, -7.6675e-01f, -8.7965e-01f, -7.0065e-01f, -1.0059e+00f, -2.3180e+00f, -2.5049e-01f, -2.2947e-01f, 

        -4.4406e-01f, -1.3146e-01f, 3.7577e-02f, -1.8938e-01f, -1.8633e-01f, -7.2698e-01f, -1.5945e-01f, -4.5842e-01f, 

        -3.4424e-01f, 1.5751e-01f, 2.7050e-01f, -7.0100e-02f, -7.5260e-03f, -1.5071e-01f, -3.5482e-01f, 1.6787e-01f, 

        1.2076e+00f, 5.8466e-01f, 1.5389e-01f, 3.8476e-01f, 2.7885e-01f, 2.6160e-02f, -1.4866e-01f, -4.7714e-02f, 

        2.4754e-01f, 4.6799e-02f, -4.7210e-02f, 1.4918e-02f, -3.8208e-02f, 1.6517e-01f, 1.1698e-02f, 3.1934e-02f, 

        -3.8317e-01f, 1.5168e-01f, 2.0713e-01f, -1.2329e-01f, 4.4359e-02f, 5.9891e-01f, 4.8938e-02f, -1.6849e-01f, 

        -4.6764e-01f, -7.8894e-02f, 2.1613e-01f, 9.6705e-02f, 2.6631e-01f, 2.6396e-01f, 1.6494e-01f, 3.7762e-02f, 

        -7.8319e-02f, -1.0661e-01f, 1.0653e-02f, 1.9830e-01f, -3.7395e-02f, 4.2256e-01f, -2.0050e-01f, 5.8426e-01f, 

        -2.5488e-01f, -4.4142e-01f, 2.6820e-01f, -7.4695e-01f, -3.2670e-01f, -3.2905e-01f, -9.5143e-01f, -1.2143e+00f, 

        -1.5803e-01f, -3.4500e-01f, 3.8461e-01f, 6.5573e-02f, 3.8045e-01f, 4.9987e-01f, 5.7646e-01f, -2.0256e-01f, 

        -1.1033e+00f, -1.8355e-01f, -3.1332e-01f, -8.5583e-01f, -2.5711e-02f, 3.8678e-01f, -7.4756e-01f, 1.2653e-01f, 

        5.3878e-01f, 7.2067e-02f, 2.7763e-01f, -1.1644e-01f, -7.2033e-02f, -2.5947e-02f, -1.8852e-01f, -2.7001e-01f, 

        -2.3525e-01f, 7.9474e-01f, -1.0716e+00f, -4.9719e-01f, -1.2719e+00f, 3.5756e-01f, -1.2098e+00f, -8.7966e-01f, 

        3.3506e-01f, -1.5025e-01f, -9.9873e-02f, -3.8078e-01f, -3.6353e-01f, -9.0900e-02f, -1.9921e-01f, -7.9426e-02f, 

        2.1198e-01f, -4.4722e-01f, 5.1132e-01f, -2.5886e-01f, -5.9675e-01f, -1.9602e-01f, -6.4315e-01f, -7.5857e-01f, 

        -1.1587e-01f, 2.6833e-02f, -1.7863e-01f, -1.2932e-01f, -3.2356e-01f, -7.3293e-01f, 2.3658e-02f, -4.2589e-01f, 

        -2.7234e-01f, -4.2905e-01f, -2.4327e-01f, 1.5433e-01f, 2.5096e-01f, -3.0639e-01f, -3.7339e-03f, -1.2668e-01f, 

        -3.8615e-01f, -4.2596e-01f, 7.7911e-02f, -2.2829e-02f, -1.5989e-01f, 6.9936e-02f, 1.1705e-01f, -2.1210e-02f, 

        1.8189e-01f, 4.9280e-02f, -2.6718e-02f, 1.6959e-01f, -5.6660e-02f, 6.7022e-02f, -8.0384e-01f, -6.3486e-01f, 

        -2.2057e-01f, -2.1710e-01f, 4.6466e-01f, -2.3356e-01f, -3.7697e-01f, -1.1791e-01f, -4.4113e-01f, 1.4451e-01f, 

        8.3463e-02f, 2.7800e-01f, -8.2760e-01f, 7.6872e-02f, 2.0360e-01f, -4.3166e-01f, 4.3787e-01f, -6.6419e-01f, 

        -5.3546e-01f, -7.1629e-01f, -2.7004e-01f, -1.4967e-01f, -1.2100e+00f, 2.0901e-01f, -4.8655e-02f, 7.8654e-01f, 

        6.9709e-01f, 3.3322e-02f, 7.5341e-02f, 2.9844e-02f, 2.5409e-02f, 1.8090e-01f, 4.4633e-02f, 1.7390e-01f, 

        4.7355e-02f, 9.7093e-02f, 6.5829e-02f, 3.3132e-01f, -8.7539e-01f, -2.3139e-02f, 3.1926e-01f, 4.6369e-03f, 

        -1.2846e-01f, -9.1981e-02f, -4.6328e-01f, 3.5499e-01f, 4.2711e-02f, -1.6414e-01f, -2.9889e-01f, 2.4862e-01f, 

        1.8051e-01f, 5.8247e-02f, -2.2520e-01f, 2.6879e-01f, 4.6147e-02f, 3.1563e-03f, -8.4876e-02f, 1.2525e-01f, 

        3.4522e-02f, -7.8408e-03f, 9.3217e-02f, 7.2669e-02f, -8.2055e-02f, 6.2889e-02f, 4.0414e-01f, -3.9945e-01f, 

        -1.1720e-01f, -5.1394e-02f, 4.1409e-02f, 4.4521e-02f, 2.9849e-01f, 6.1090e-01f, -5.1357e-01f, -7.6545e-01f, 

        -1.3711e-01f, 1.0564e-01f, -1.7384e-01f, -3.5000e-01f, 2.4564e-01f, 7.4034e-01f, -2.1899e-01f, -8.1134e-02f, 

        -1.8744e-01f, -6.5393e-03f, -2.1436e-02f, -1.3047e-01f, -6.8742e-02f, -5.4969e-02f, -1.0092e-01f, -8.5147e-02f, 

        -1.7001e-01f, -1.3556e-01f, -1.8045e-01f, -1.2422e-01f, -1.0136e-01f, -1.5117e-01f, -1.4568e-01f, -1.3226e-01f, 

        -1.7496e-01f, -1.4717e-01f, -1.1840e-01f, -1.1887e-01f, -1.7324e-01f, -1.6596e-01f, -1.6609e-01f, -1.2983e-01f, 

        -7.3293e-02f, -1.5687e-01f, -1.2769e-01f, -1.8017e-01f, -1.1111e-01f, -1.7524e-01f, -1.5402e-01f, -1.2795e-01f, 

        -1.2304e-01f, -1.9323e-01f, -1.4371e-01f, -1.5597e+00f, -1.4069e+00f, -1.1186e+00f, -9.5316e-01f, -5.2807e-01f, 

        -5.2289e-01f, -9.5933e-01f, -4.8039e-01f, -6.2252e-01f, -1.7957e-01f, -6.9890e-02f, -1.6267e-01f, -1.3805e-01f, 

        -1.0563e-01f, -1.3514e-01f, -2.3021e-01f, -1.2035e-01f, -2.2515e-01f, -4.1354e-01f, -2.4655e-01f, -3.4637e-01f, 

        -3.1705e-01f, -3.7651e-01f, -3.7397e-01f, -3.5590e-01f, -2.7473e-01f, -1.8285e-01f, -1.7264e-01f, -2.5736e-01f, 

        -1.0927e-01f, -1.6087e-01f, -1.7275e-01f, -1.4694e-01f, -1.2293e-01f, -8.4953e-02f, -8.9865e-02f, -2.0663e-01f, 

        -3.0371e-01f, -3.1689e-01f, -2.5679e-01f, -2.3692e-01f, -2.2129e-01f, -1.1782e-01f, -1.0650e-01f, -1.2990e-01f, 

        -2.0766e-01f, -8.7577e-02f, -1.7660e-01f, -1.8661e-01f, -1.5451e-01f, -3.3527e-01f, -1.8299e-01f, -1.5383e-01f, 

        -1.2420e-01f, -8.7834e-02f, -1.7390e-01f, -1.5965e-01f, -2.2653e-01f, -1.8569e-01f, -1.8474e-01f, -1.0963e-01f, 

        -9.2009e-02f, -1.6312e-01f, -1.8737e+00f, -1.5726e+00f, -1.8255e+00f, -1.4258e+00f, -1.0968e+00f, -1.7530e+00f, 

        -1.6683e+00f, -1.1273e+00f, -1.6292e+00f, -1.2376e-01f, -1.3675e-01f, -1.5646e-01f, -9.2618e-02f, -7.0312e-02f, 

        -9.0240e-02f, -1.6706e-01f, -1.6135e-01f, -1.4662e-01f, -3.8924e-01f, -2.0180e-01f, -2.6260e-01f, -4.7122e-01f, 

        -1.9769e-01f, -1.8784e-01f, -1.8994e-01f, -1.7946e-01f, -1.3204e-01f, -1.6173e-01f, -1.1175e-01f, -1.3451e-01f, 

        -1.8012e-01f, -1.0609e-01f, -9.0189e-02f, -8.6693e-02f, -1.2748e-01f, -1.0211e-01f, -1.0077e-01f, -1.1717e-01f, 

        -1.1853e-01f, -1.2882e-01f, -1.4528e-01f, -1.7490e-01f, -1.5751e-01f, -1.3695e-01f, -1.6382e-01f, -1.7332e-01f, 

        -2.0488e-01f, -1.9393e-01f, -3.0452e-01f, -2.1026e-01f, -2.2659e-01f, -1.8059e-01f, -2.3877e-01f, -2.3605e-01f, 

        -3.3431e-01f, -2.1740e-01f, -2.5704e-01f, -1.4946e-01f, -8.7398e-02f, -1.5982e-01f, -1.7805e-01f, -1.9742e-01f, 

        -1.5038e-01f, -2.3346e-01f, -2.0190e-01f, -2.4227e-01f, -2.3313e-01f, -2.4295e-01f, -1.9575e-01f, -1.7375e-01f, 

        -2.0832e-01f, -2.9799e-01f, -2.4088e-01f, -1.1293e-01f, -9.2356e-02f, -1.7482e-01f, -1.3493e-01f, -1.1067e-01f, 

        -1.3654e-01f, -8.6169e-02f, -1.7408e-01f, -3.7382e-01f, -2.6736e-01f, -5.7751e-01f, -2.8740e-01f, -2.3998e-01f, 

        -1.5684e-01f, -1.5233e-01f, -2.1389e-01f, -2.8129e-01f, -1.4308e-01f, -1.1428e-01f, -1.2055e-01f, -1.9209e-01f, 

        -9.2934e-02f, -1.5257e-01f, -1.4046e-01f, -1.9716e-01f, -1.4883e-01f, -2.5573e-01f, -1.9971e-01f, -1.9923e-01f, 

        -2.0535e-01f, -1.9838e-01f, -1.2393e-01f, -1.6468e-01f, -2.0863e-01f, -1.8710e-01f, -1.6513e-01f, -1.2872e-01f, 

        -1.3544e-01f, -1.7753e-01f, -1.2707e-01f, -2.1574e-01f, -1.5946e-01f, -1.6363e-01f, -1.1152e-01f, -2.7817e-01f, 

        -1.4300e-01f, -1.1552e-01f, -2.3781e-01f, -2.6263e-01f, -1.4647e-01f, -2.1795e-01f, -1.0753e-01f, -7.0248e-02f, 

        -1.8735e-01f, -1.9676e-01f, -1.6013e-01f, -1.8281e-01f, -1.3239e-01f, -2.4700e-01f, -1.9778e-01f, -9.3568e-02f, 

        -7.5615e-02f, -1.8203e-01f, -1.6709e-01f, -1.3642e-01f, -1.8859e-01f, -1.7693e-01f, -1.0882e-01f, -1.6463e-01f, 

        -1.1392e-01f, -1.7498e-01f, -2.6154e-01f, -2.2286e-01f, -2.0173e-01f, -2.3075e-01f, -2.0381e-01f, -1.5493e-01f, 

        -1.8340e-01f, -2.3175e-01f, -1.1555e-01f, -1.8203e-01f, -1.4748e-01f, -1.1370e-01f, -1.3236e-01f, -1.1324e-01f, 

        -7.8333e-02f, -1.2105e-01f, -9.4235e-02f, -1.0577e-01f, -1.4091e-01f, -1.0894e-01f, -1.8000e-01f, -3.9560e-02f, 

        -1.8800e-01f, -8.7344e-02f, -1.4879e-01f, -8.4969e-02f, -1.0509e-01f, -1.3803e-01f, -2.5212e-01f, -2.4530e-01f, 

        -1.5880e-01f, -2.0912e-01f, -1.9010e-01f, -1.8591e-01f, -2.1103e-01f, -1.2020e-01f, -1.6552e-01f, -1.2141e-01f, 

        -1.6934e-01f, -1.7522e-01f, -7.0946e-02f, -1.3074e-01f, -1.5715e-01f, -1.5115e-01f, -8.0996e-02f, -6.6919e-02f, 

        -6.2141e-02f, -6.8874e-02f, -7.2918e-02f, -6.2239e-02f, -9.5120e-02f, -1.2946e-01f, -1.4573e-01f, -9.1586e-02f, 

        -4.0403e-01f, -2.1275e-01f, -2.6934e-01f, -2.9537e-02f, -5.0177e-01f, -7.4655e-01f, -5.1131e-01f, -4.1011e-01f, 

        -5.5250e-01f, 6.5805e-02f, -1.2431e-01f, -1.6675e-01f, -2.0948e-01f, -1.8567e-01f, -5.9379e-02f, -4.9575e-02f, 

        -1.2165e-01f, -2.1750e-01f, 2.2438e-02f, -1.3397e+00f, 1.1111e+00f, 3.5983e-01f, 7.2196e-04f, -2.7671e-01f, 

        2.1974e-01f, -1.5383e-01f, -3.7737e-01f, 2.2546e-01f, 2.6690e-01f, -9.7511e-01f, 1.5731e-02f, -6.9735e-01f, 

        -7.1619e-01f, -1.9471e-01f, 1.6796e-01f, -8.7862e-01f, -6.1442e-01f, -2.2204e-02f, 3.0370e-01f, 6.4701e-01f, 

        2.2009e-01f, 3.7088e-01f, 1.5801e-01f, -3.1476e-01f, -1.6687e-01f, -5.5807e-01f, -8.2396e-02f, -9.1985e-02f, 

        -4.8775e-01f, 5.9650e-02f, -9.3801e-01f, 6.8736e-02f, 7.7267e-01f, 5.8224e-01f, 2.2476e-01f, 2.2287e-01f, 

        4.2488e-01f, 2.5195e-01f, 1.1721e+00f, -8.3591e-01f, 2.7988e-01f, -1.2856e+00f, 7.5945e-01f, -3.4327e-01f, 

        -5.3792e-01f, 9.8496e-01f, 1.2370e-01f, -6.6319e-01f, -7.0160e-01f, -2.4032e-01f, -9.6232e-01f, -1.7647e-01f, 

        -1.1448e+00f, 7.2676e-02f, 1.8141e-01f, -9.6283e-01f, -5.9919e-02f, -4.1236e-01f, -1.4357e+00f, -8.9211e-01f, 

        -1.5105e-01f, -4.0611e-01f, -2.7634e-01f, 9.5734e-01f, 3.3829e-01f, 1.4696e+00f, -3.0441e-01f, -7.2116e-01f, 

        6.5535e-02f, 8.1708e-01f, -3.3426e-01f, -4.0344e-01f, -1.3905e-01f, 5.1883e-01f, 3.2192e-01f, 9.2123e-01f, 

        -3.1927e-01f, -2.5963e-01f, -9.9236e-01f, -1.5474e-01f, -1.6797e-01f, -2.6052e-01f, -2.8511e-01f, -2.6778e-01f, 

        -2.7204e-01f, -3.1516e-01f, -1.7854e-01f, -2.1611e-01f, -1.3575e-01f, 9.9031e-01f, -1.2412e-01f, -5.4803e-01f, 

        -2.7353e-01f, 4.2445e-03f, 7.9690e-01f, -8.1749e-01f, -1.1784e+00f, -1.5006e+00f, -1.2428e+00f, 7.5421e-01f, 

        -1.1889e+00f, 2.8794e-01f, -2.3064e+00f, -7.3765e-01f, -1.2690e+00f, -1.5629e+00f, -1.8455e+00f, 4.0022e-01f, 

        -7.1699e-01f, -8.0380e-01f, -1.1970e+00f, -4.6576e-01f, -7.2496e-01f, -4.5609e-01f, -1.5435e+00f, -6.4397e-01f, 

        -2.6521e-01f, 1.2347e-01f, -4.1104e-01f, -6.3552e-01f, -3.7252e-02f, 3.3876e-01f, -6.0970e-02f, 7.1686e-01f, 

        -2.5798e-01f, 8.8345e-01f, -3.0041e+00f, 7.9630e-01f, -2.1110e+00f, 6.5528e-01f, 6.0438e-01f, -4.0728e-01f, 

        6.6215e-01f, -9.7108e-02f, -1.4911e-01f, 9.0912e-01f, 1.3929e+00f, -1.5529e-01f, -7.5684e-02f, 5.2811e-01f, 

        -4.3479e-01f, -7.0338e-01f, 4.9882e-01f, -1.5597e-01f, -4.4999e-01f, -2.4266e-01f, -3.2605e-01f, 5.3903e-02f, 

        -1.0219e+00f, -1.9213e+00f, -1.2979e+00f, -1.1569e+00f, -3.2550e-01f, -6.9812e-01f, 3.8525e-01f, -4.9093e-01f, 

        -5.3650e-01f, -5.5521e-01f, -1.2182e+00f, -4.0052e-02f, -8.8684e-01f, -8.2435e-01f, -1.4595e+00f, -4.6566e-01f, 

        -1.9980e+00f, -4.5906e-01f, 1.1075e+00f, -6.4893e-01f, -6.7248e-01f, -2.7019e-02f, -2.1124e-01f, 2.2138e-01f, 

        1.0021e-01f, 4.5819e-01f, -2.0366e-01f, 1.3746e-01f, 9.2912e-03f, 1.9646e-01f, -2.5882e-01f, -6.1199e-01f, 

        8.6645e-01f, -4.0484e-01f, 5.2938e-01f, -2.5709e-03f, -5.3426e-01f, 2.1324e-01f, -7.3235e-01f, -3.7809e-02f, 

        -1.8470e-01f, 9.0672e-02f, 6.7506e-01f, -1.2238e+00f, 6.3518e-01f, -2.0260e-01f, -8.7131e-01f, 1.7706e+00f, 

        -6.2511e-01f, -6.0104e-01f, -4.1841e-01f, -3.7188e-01f, -1.0485e+00f, -5.7595e-01f, -1.4887e+00f, -4.1970e-01f, 

        -6.2843e-01f, -1.0500e-01f, -6.5279e-02f, -1.7557e-01f, -7.9760e-02f, -1.2895e-01f, -1.0946e-01f, -1.4528e-01f, 

        -4.2414e-02f, -1.4006e-01f, -2.3213e-01f, -7.5860e-01f, -9.2334e-01f, -2.9009e-01f, -9.5528e-01f, -2.0473e-01f, 

        -3.1574e-01f, -9.5962e-01f, -6.1361e-01f, -7.7445e-01f, -8.7747e-01f, -7.4428e-02f, 5.0384e-01f, -1.9443e+00f, 

        -8.6158e-01f, -1.3027e+00f, -2.1841e+00f, 1.8448e+00f, -2.1143e-01f, -2.0109e-01f, -1.7640e-01f, -1.8913e-01f, 

        -2.0855e-01f, -1.7015e-01f, -2.8276e-01f, -2.1638e-01f, -3.2251e-01f, -4.8833e-01f, -8.7976e-01f, -1.8559e-02f, 

        -7.7425e-03f, -2.2691e+00f, -8.6905e-01f, -1.4282e+00f, -1.9544e-01f, -2.0622e-01f, -3.8461e-01f, 1.6878e-01f, 

        4.1509e-01f, -1.1421e+00f, 1.2696e-01f, -2.6928e-01f, 3.1789e-02f, -3.6720e-01f, -1.1232e-01f, -1.0196e-01f, 

        -3.2588e-01f, -2.6740e-01f, -2.2359e-01f, -3.1260e-01f, -3.2895e-01f, -2.9003e-01f, -3.6559e-01f, -1.3907e-01f, 

        4.8658e-02f, -5.1812e-02f, 1.1589e-01f, -1.0372e-01f, 5.7975e-02f, -1.9993e-02f, 6.9478e-02f, -1.1822e-01f, 

        -2.4843e-01f, -1.3109e-01f, -1.8379e-01f, 3.1034e-03f, -8.0233e-02f, -1.2722e-01f, -1.1441e-01f, 8.5063e-02f, 

        -6.8096e-02f, -2.8023e-01f, -1.9320e-01f, -2.4777e-01f, -3.6912e-01f, 8.4608e-02f, -2.0130e-01f, -6.5867e-02f, 

        -9.7798e-02f, 5.3563e-01f, 1.7796e-02f, 2.1000e-01f, -1.0019e+00f, -2.8380e-01f, 1.5339e-02f, -1.2476e-01f, 

        -9.7411e-02f, 4.7121e-01f, -4.0623e-02f, 8.0672e-02f, -1.6360e-02f, -2.1354e-01f, -2.3080e-01f, -7.4514e-03f, 

        1.5103e-01f, -6.6701e-02f, -8.5771e-02f, 2.6068e-01f, -6.6350e-02f, 6.9625e-01f, -3.6112e-01f, -3.8802e-01f, 

        -4.5707e-01f, 6.9787e-03f, -2.7611e-01f, 1.1851e-01f, -7.1335e-01f, 7.1099e-02f, -8.1111e-01f, -2.8628e-01f, 

        -5.6805e-01f, 6.7303e-02f, 2.3419e-01f, -1.3640e-01f, 7.3241e-02f, -1.0120e-01f, 1.2815e-01f, -2.4669e-01f, 

        -9.3496e-02f, 1.6214e-01f, -1.4350e-01f, -1.5987e-01f, -2.5257e-02f, -6.9887e-01f, -1.8622e-01f, 1.5266e-01f, 

        -6.3543e-01f, -1.3907e-01f, -1.7559e-01f, -3.4707e-01f, -6.3490e-03f, -1.4506e-02f, 1.4111e-01f, -3.0774e-01f, 

        -2.2314e-01f, -3.6536e-01f, -2.7239e-01f, -1.4369e-01f, 5.1118e-03f, 1.4251e-02f, -1.2133e-01f, -1.1232e-01f, 

        2.9894e-02f, 1.4573e-01f, -3.3513e-01f, -2.8544e-01f, -2.8575e-01f, 7.1943e-02f, 4.5393e-01f, -3.0306e-01f, 

        -5.6696e-01f, -1.0369e+00f, -9.9300e-01f, -3.2206e-02f, -2.3010e-01f, -1.8120e-01f, -7.4545e-02f, -5.5832e-03f, 

        -2.9040e-01f, -2.0820e-03f, -1.0885e-01f, 9.4019e-02f, -3.5713e-01f, -1.2693e+00f, 2.1271e-01f, -2.3303e-01f, 

        6.4857e-01f, 1.7246e-02f, -6.4409e-01f, -2.4079e-01f, -1.6921e-02f, -6.1893e-01f, -9.7585e-01f, -7.0975e-01f, 

        1.7790e-01f, -4.2483e-01f, 1.0337e+00f, 7.2861e-01f, -3.6845e-01f, -3.8453e-02f, 7.1912e-01f, 2.4118e+00f, 

        1.3105e+00f, -5.4967e-01f, -2.0426e-01f, -3.2581e-03f, 5.4177e-02f, 2.7471e-01f, -5.6196e-01f, 1.8798e-01f, 

        5.5122e-02f, 3.4138e-01f, -1.9697e-01f, -2.5562e-02f, -2.6211e-02f, -2.7657e-01f, 3.4693e-02f, 2.2488e-01f, 

        2.8198e-01f, -7.8451e-02f, -1.2247e-01f, -1.2764e+00f, 1.2623e+00f, 2.9908e-01f, -4.7432e-01f, -6.2166e-01f, 

        -1.4139e-01f, -5.3106e-01f, -1.6794e-01f, 2.3600e-02f, 1.7855e-01f, -5.6902e-03f, 4.8046e-03f, 1.8217e-01f, 

        -1.8506e-02f, 6.1150e-02f, 7.8092e-03f, 4.1782e-02f, -9.6242e-02f, -7.7618e-02f, -6.3282e-01f, -4.1061e-01f, 

        -3.8525e-01f, 1.4502e-02f, -2.2126e-01f, -5.9086e-01f, -4.8410e-01f, -8.1170e-02f, -2.6951e-01f, -5.1448e-01f, 

        3.7933e-01f, -1.7663e-01f, -2.5349e-01f, -5.7036e-01f, -8.9033e-01f, 5.3000e-01f, -1.6652e-01f, -6.7758e-01f, 

        3.6446e-01f, -4.7669e-01f, -6.7122e-01f, -3.5232e-01f, -3.7755e-01f, 4.3891e-02f, -5.7119e-02f, 2.5906e-01f, 

        1.0974e-02f, 1.2936e-01f, -7.1076e-02f, -9.5706e-02f, 1.9046e-01f, 2.8383e-01f, -6.3438e-01f, -3.3222e-01f, 

        -4.0914e-01f, -1.1607e+00f, 1.7256e-01f, -1.6877e-02f, -1.1714e+00f, -7.6362e-01f, -4.8995e-01f, -4.6686e-01f, 

        -2.4170e-03f, 1.6300e-02f, -3.3952e-01f, -5.2325e-01f, -3.4912e-01f, -1.1388e-01f, -4.4042e-01f, -1.9992e-01f, 

        -1.1810e+00f, 8.0899e-01f, 3.1536e-01f, -3.3473e-01f, -2.6055e-01f, -3.7266e-01f, -3.6955e-01f, 8.7566e-01f, 

        1.1543e-01f, -2.0787e-01f, -1.8055e-01f, 2.5505e-02f, 3.2208e-03f, -2.0079e-01f, 2.0537e-01f, -1.7745e-01f, 

        -9.6964e-02f, 3.0136e-03f, -6.3654e-02f, -4.3034e-01f, -1.0820e-01f, -2.2972e-01f, 5.1390e-01f, 8.9343e-03f, 

        -1.1736e+00f, 5.0537e-02f, -1.1601e-01f, -1.0863e-01f, -5.0054e-01f, 6.7390e-02f, -1.3534e-01f, -2.6739e-02f, 

        1.1280e+00f, -2.7055e-01f, -7.7676e-02f, -3.0412e-01f, -1.7370e-01f, -4.1484e-02f, -1.8694e-01f, 8.2951e-02f, 

        -3.9572e-02f, 7.3385e-02f, -5.4260e-02f, 1.1692e-01f, -1.1723e-01f, -1.7027e-01f, 8.3245e-03f, 4.7959e-02f, 

        -2.3728e-01f, 3.4781e-01f, 5.3755e-02f, -3.8182e-01f, -5.2797e-01f, -3.0800e-01f, -7.3964e-01f, -3.3926e-01f, 

        -7.9797e-01f, 8.6451e-02f, -2.9794e-01f, -6.6410e-01f, 6.0174e-01f, 6.6460e-01f, -4.6611e-01f, 2.2790e-01f, 

        3.2696e-01f, 2.3812e-01f, 1.7972e-01f, 3.3761e-01f, 8.1148e-02f, -1.1280e-01f, 3.3502e-01f, 2.3903e-01f, 

        -2.0112e-01f, -1.5670e-01f, -1.8295e-01f, -9.9837e-02f, -1.0580e-01f, -9.3653e-02f, -1.6054e-01f, -1.7127e-01f, 

        -1.1002e-01f, -1.4254e-01f, -1.2804e-01f, -5.9063e-02f, -1.4612e-01f, -1.3304e-01f, -9.7297e-02f, -6.2307e-02f, 

        -9.8395e-02f, -1.1675e-01f, -1.0637e-01f, -1.7122e-01f, -7.6260e-02f, -1.5858e-01f, -1.3484e-01f, -1.5200e-01f, 

        -9.4941e-02f, -1.5783e-01f, -1.4352e-01f, -1.5650e+00f, -1.7331e+00f, -1.2135e+00f, -7.6438e-01f, -5.9553e-01f, 

        -5.1314e-01f, -7.2976e-01f, -4.4240e-01f, -5.2960e-01f, -1.3995e-01f, -8.0800e-02f, -1.5848e-01f, -8.5823e-02f, 

        -5.4995e-02f, -8.9628e-02f, -1.5559e-01f, -1.0101e-01f, -9.7482e-02f, -3.5042e-01f, -2.0531e-01f, -2.6744e-01f, 

        -3.0259e-01f, -2.5882e-01f, -2.2138e-01f, -3.0955e-01f, -2.3648e-01f, -1.8505e-01f, -1.0456e-01f, -2.0272e-01f, 

        -1.2383e-01f, -1.7514e-01f, -9.5625e-02f, -1.4821e-01f, -1.5915e-01f, -1.8669e-01f, -1.1599e-01f, -1.5643e-01f, 

        -2.4108e-01f, -1.3584e-01f, -1.9694e-01f, -1.4450e-01f, -1.2543e-01f, -1.7786e-01f, -9.9669e-02f, -2.6377e-01f, 

        -1.3100e-01f, -1.5655e-01f, -2.2855e-01f, -1.7088e-01f, -9.5010e-02f, -2.6338e-01f, -1.6353e-01f, -1.0608e-01f, 

        -1.3831e-01f, -1.4826e-01f, -1.6246e-01f, -1.3211e-01f, -2.0541e-01f, -1.0641e-01f, -9.4240e-02f, -1.4348e-01f, 

        -1.3775e-01f, -9.6156e-02f, -2.2226e+00f, -1.1012e+00f, -1.5611e+00f, -1.8804e+00f, -1.1395e+00f, -1.5351e+00f, 

        -1.7369e+00f, -1.1424e+00f, -1.3854e+00f, -1.3281e-01f, -5.4748e-02f, -4.1737e-02f, -1.1473e-01f, -9.1320e-02f, 

        -5.6995e-02f, -8.6756e-02f, -1.4927e-01f, -1.9215e-01f, -4.7666e-01f, -2.9884e-01f, -2.0975e-01f, -5.0836e-01f, 

        -1.4540e-01f, -1.2035e-01f, -2.4903e-01f, -1.8055e-01f, -1.7181e-01f, -1.0651e-01f, -2.2125e-01f, -1.7639e-01f, 

        -1.3980e-01f, -1.0858e-01f, -7.9735e-02f, -9.2287e-02f, -1.1664e-01f, -2.0019e-01f, -8.3504e-02f, -1.3461e-01f, 

        -9.2899e-02f, -1.4632e-01f, -9.8797e-02f, -1.0504e-01f, -4.0396e-02f, -5.3604e-02f, -6.4302e-02f, -1.9747e-01f, 

        -2.1859e-01f, -1.9602e-01f, -2.2922e-01f, -1.9193e-01f, -1.7416e-01f, -1.8418e-01f, -2.3073e-01f, -1.9179e-01f, 

        -2.5899e-01f, -1.1481e-01f, -2.6684e-01f, -7.2140e-02f, -1.2843e-01f, -1.4321e-01f, -1.8107e-01f, -1.0313e-01f, 

        -8.3553e-02f, -2.6314e-01f, -2.9522e-01f, -1.9448e-01f, -2.8809e-01f, -2.0996e-01f, -2.5424e-01f, -1.7296e-01f, 

        -1.9086e-01f, -1.9721e-01f, -1.7965e-01f, -1.3478e-01f, -1.9824e-01f, -1.1025e-01f, -1.3072e-01f, -1.5336e-01f, 

        -1.7539e-01f, -8.1228e-02f, -1.7297e-01f, -3.5383e-01f, -2.4695e-01f, -2.4056e-01f, -1.5615e-01f, -1.9219e-01f, 

        -1.7262e-01f, -2.4481e-01f, -1.2105e-01f, -2.3237e-01f, -2.0621e-01f, -2.0215e-01f, -2.3740e-01f, -1.4571e-01f, 

        -1.8394e-01f, -1.5462e-01f, -2.2599e-01f, -1.2086e-01f, -1.8020e-01f, -1.5300e-01f, -1.6330e-01f, -2.1411e-01f, 

        -1.6985e-01f, -1.4170e-01f, -1.6076e-01f, -1.8299e-01f, -1.5898e-01f, -1.4968e-01f, -1.5761e-01f, -1.4285e-01f, 

        -6.7757e-02f, -2.3988e-01f, -7.7540e-02f, -1.8695e-01f, -8.8769e-02f, -1.5837e-01f, -1.5675e-01f, -2.7279e-01f, 

        -7.6826e-02f, -1.0877e-01f, -1.8635e-01f, -1.8205e-01f, -1.4664e-01f, -2.1585e-01f, -1.5138e-01f, -1.0550e-01f, 

        -1.2393e-01f, -2.0924e-01f, -2.0253e-01f, -1.2033e-01f, -1.6007e-01f, -2.3771e-01f, -1.7334e-01f, -1.5728e-01f, 

        -1.1432e-01f, -1.7017e-01f, -7.7710e-02f, -8.7625e-02f, -9.1543e-02f, -8.3373e-02f, -8.4472e-02f, -8.3231e-02f, 

        -1.5342e-01f, -6.8090e-02f, -1.9278e-01f, -1.8203e-01f, -1.7926e-01f, -1.4452e-01f, -1.7214e-01f, -1.2949e-01f, 

        -1.9434e-01f, -2.2337e-01f, -1.1867e-01f, -1.9421e-01f, -1.1198e-01f, -1.7316e-01f, -8.9690e-02f, -1.6287e-01f, 

        -1.3071e-01f, -1.2272e-01f, -1.5001e-01f, -6.5633e-02f, -1.6560e-01f, -8.9693e-02f, -9.8080e-02f, -1.3515e-01f, 

        -1.1865e-01f, -1.7206e-01f, -1.6642e-01f, -7.4406e-02f, -1.3530e-01f, -1.7629e-01f, -2.1759e-01f, -1.2414e-01f, 

        -2.2779e-01f, -8.4999e-02f, -2.0822e-01f, -1.7555e-01f, -1.5068e-01f, -1.4449e-01f, -1.2213e-01f, -1.3426e-01f, 

        -1.3941e-01f, -1.4487e-01f, -6.8656e-02f, -1.7742e-01f, -1.8533e-01f, -1.2269e-01f, -1.7499e-01f, -5.0372e-02f, 

        -1.6744e-01f, -1.1200e-01f, -2.3133e-02f, -5.8864e-02f, -9.3630e-02f, -1.3909e-01f, -1.6330e-01f, -1.1441e-01f, 

        -4.9538e-01f, -3.4028e-01f, -5.9907e-01f, -6.6790e-01f, -1.3392e-01f, 1.2607e-01f, -6.2020e-01f, -4.8006e-01f, 

        -6.5571e-01f, -4.1596e-02f, -1.9477e-01f, -1.1481e-01f, -1.0505e-01f, -1.0688e-01f, -2.3497e-01f, 1.2343e-02f, 

        4.0266e-02f, -9.3517e-02f, -4.3674e-01f, -8.3383e-01f, -1.4101e-01f, -3.4469e-01f, 1.9728e-01f, -2.7483e+00f, 

        -1.1211e+00f, -3.8430e-01f, 2.9857e-01f, -8.8747e-02f, -1.3395e+00f, -1.6358e-01f, -5.8160e-01f, -7.1490e-01f, 

        -2.5794e-01f, -6.6668e-01f, -9.6818e-01f, 3.8638e-01f, 2.3959e-03f, -3.9926e-01f, 4.3376e-02f, 2.1544e-01f, 

        1.3364e-01f, -7.4067e-01f, -4.0217e-01f, -1.8105e-02f, -2.2981e-01f, -2.2582e-02f, -3.2762e-01f, 1.4600e+00f, 

        -6.4339e-01f, -3.6219e-01f, 9.1644e-03f, -9.2306e-01f, -1.5252e-01f, -8.1956e-01f, -2.8775e-01f, 2.5471e-01f, 

        6.6467e-01f, 1.8236e-01f, 5.4267e-01f, -6.4625e-01f, -1.3909e+00f, -9.7421e-01f, -4.1975e-02f, -1.3455e+00f, 

        -5.8212e-01f, -4.1094e-01f, -3.7048e-01f, 1.9051e-01f, 8.9143e-01f, -3.1849e-01f, -1.5700e-01f, -4.9738e-01f, 

        -5.7287e-01f, -6.8736e-01f, 1.3627e-01f, 3.9373e-01f, -7.5670e-01f, -7.3024e-01f, -4.6542e-01f, -2.9996e-02f, 

        1.9598e-01f, -7.3379e-01f, -1.0293e+00f, 8.5627e-02f, 2.6813e-01f, 8.0129e-01f, -5.2248e-01f, -8.3278e-01f, 

        -5.0420e-02f, 8.6695e-01f, 8.5666e-01f, 4.6113e-01f, -1.0175e-01f, 2.9284e-03f, 5.1165e-01f, -9.2127e-01f, 

        -2.9430e-01f, -1.4394e-01f, -3.8265e-01f, -2.0680e-01f, -1.5590e-01f, -1.8974e-01f, -1.3671e-01f, -3.9927e-02f, 

        -6.6312e-02f, -2.1306e-01f, 9.3431e-02f, -1.3719e-02f, -1.1969e+00f, -1.1334e+00f, -1.0151e+00f, -2.9675e-01f, 

        8.6029e-02f, -2.0592e-01f, -3.2292e-01f, -2.2613e-01f, 2.3269e-01f, -1.3842e+00f, -1.4285e+00f, 1.1342e+00f, 

        1.0265e+00f, 2.2119e-01f, 2.6721e-01f, 1.4766e+00f, -1.5909e-01f, -2.5060e+00f, -3.7836e-01f, -6.1297e-01f, 

        -8.3757e-02f, 1.1331e+00f, 8.1096e-01f, 2.3769e-01f, -1.1021e+00f, 1.0490e-01f, 9.0528e-01f, -7.7256e-01f, 

        -5.2656e-01f, -1.2892e+00f, -8.9809e-01f, -3.5529e-01f, -9.9614e-01f, -8.7030e-01f, -9.4138e-01f, -6.2108e-01f, 

        -2.3127e-01f, -4.9085e-01f, -1.9409e+00f, 4.3431e-02f, -2.1315e-01f, -9.6170e-01f, -5.5619e-01f, 2.5719e-01f, 

        6.2399e-01f, 1.2992e-01f, -1.5163e-01f, -3.7027e-01f, -3.6188e-01f, -1.0814e+00f, -2.1068e-01f, -6.1308e-01f, 

        -1.9077e-01f, -4.4887e-01f, -5.2412e-02f, -4.2046e-01f, -3.0911e-01f, -8.9517e-02f, -8.8315e-01f, -4.1434e-01f, 

        -3.8592e-01f, -3.3805e-01f, -5.5485e-01f, -8.0283e-01f, 3.0067e-02f, 8.6597e-01f, -6.2686e-01f, -8.6075e-02f, 

        1.3193e+00f, -7.8829e-02f, 2.1006e-01f, -1.1354e-01f, 4.1719e-01f, -5.9343e-01f, -2.0175e+00f, 2.7786e-01f, 

        3.4033e-01f, -7.0104e-01f, -1.1822e+00f, -6.0463e-01f, 3.5581e-01f, -5.0862e-01f, -6.2169e-01f, 1.7226e-03f, 

        -1.4668e-01f, 2.4456e-01f, -5.9845e-01f, -6.0971e-01f, -3.6777e-01f, 1.6895e-01f, -9.3304e-02f, 1.1599e-01f, 

        -4.5151e-01f, 6.8965e-01f, 3.8765e-01f, -7.4891e-01f, 3.5052e-01f, -2.0968e-01f, 3.0271e-01f, 2.8631e-02f, 

        -3.0751e-02f, -4.0210e-01f, -4.4926e-01f, -4.3627e-02f, -8.4528e-01f, -1.8462e+00f, 2.4081e-01f, -4.0961e-01f, 

        -1.6186e-01f, -7.8013e-02f, 2.8269e-01f, 1.7045e-01f, 6.2230e-01f, -5.1646e-01f, -4.2161e-02f, 4.5731e-01f, 

        -6.2883e-01f, -5.8123e-02f, -6.0881e-02f, -1.4325e-01f, -2.7863e-01f, -2.0469e-01f, -7.2185e-02f, -2.6973e-01f, 

        1.0895e-02f, -1.5098e-01f, 2.5638e-01f, -3.1023e-01f, 2.6169e-01f, -1.2836e+00f, -1.7456e-01f, -5.6146e-01f, 

        -2.8661e-01f, -9.4114e-01f, -7.4232e-01f, -4.8415e-01f, -4.1034e-01f, -1.6006e-01f, -7.2297e-01f, -3.9086e-02f, 

        7.2887e-01f, 1.5535e+00f, -4.1620e-01f, -6.1305e-01f, 1.0554e-01f, -1.5435e-02f, -7.7069e-02f, 1.9910e-02f, 

        -9.5627e-02f, -1.6620e-01f, -1.0470e-02f, -2.3052e-01f, -3.6890e-02f, -8.1413e-01f, 1.2333e+00f, -6.2647e-01f, 

        -2.3983e-01f, 1.0554e+00f, 1.0770e+00f, 5.2378e-01f, -6.7651e-01f, 3.8570e-01f, -4.2452e-01f, -3.9337e-01f, 

        -4.0664e-01f, 2.9105e-01f, -3.7887e-01f, -4.3320e-01f, 6.0127e-02f, 4.4831e-01f, 3.2839e-01f, -2.0379e-01f, 

        -4.0954e-04f, -2.9188e-01f, -7.1235e-02f, -1.1933e-01f, -2.2413e-01f, -2.9045e-01f, -3.0201e-01f, 1.8982e-02f, 

        -1.7329e-01f, -1.3091e-01f, -1.7171e-01f, -1.4062e-01f, -1.5688e-01f, -1.8738e-01f, -2.0028e-01f, -2.2266e-01f, 

        -1.6572e-01f, -1.0076e-01f, -1.1460e-01f, -1.5655e-01f, -2.0533e-01f, -8.0875e-02f, -1.1256e-01f, -8.6461e-02f, 

        -1.0158e-01f, -1.5713e-01f, -1.9357e-01f, -2.3230e-01f, -1.7606e-01f, -1.9967e-01f, -9.3291e-02f, -1.9155e-01f, 

        -2.0889e-01f, -1.0805e-01f, -1.1503e-01f, -1.5746e+00f, -1.1996e+00f, -1.2397e+00f, -6.7325e-01f, -6.3956e-01f, 

        -7.2047e-01f, -7.9139e-01f, -4.3078e-01f, -6.2609e-01f, -1.9680e-01f, -2.0717e-02f, -1.9077e-01f, -1.6125e-01f, 

        -1.5391e-01f, -9.4646e-02f, -1.4083e-01f, -1.2309e-01f, -1.2824e-01f, -3.3357e-01f, -3.1727e-01f, -3.4971e-01f, 

        -4.0588e-01f, -3.5152e-01f, -3.3143e-01f, -4.3520e-01f, -2.4324e-01f, -2.5584e-01f, -1.9738e-01f, -9.1915e-02f, 

        -1.3512e-01f, -1.5542e-01f, -8.0460e-02f, -1.6420e-01f, -1.5683e-01f, -1.3087e-01f, -1.2480e-01f, -1.9753e-01f, 

        -2.3916e-01f, -8.8319e-02f, -1.1807e-01f, -1.8461e-01f, -2.3366e-01f, -1.1199e-01f, -1.3903e-01f, -2.2884e-01f, 

        -2.0796e-01f, -6.7747e-02f, -2.1178e-01f, -1.5620e-01f, -1.3788e-01f, -2.8798e-01f, -2.2746e-01f, -8.2241e-02f, 

        -1.2840e-01f, -9.3545e-02f, -7.4161e-02f, -7.3338e-02f, -1.5581e-01f, -8.9997e-02f, -1.2013e-01f, -1.4908e-01f, 

        -1.8376e-01f, -1.5145e-01f, -2.2119e+00f, -1.6704e+00f, -1.9448e+00f, -1.7457e+00f, -1.5389e+00f, -1.8310e+00f, 

        -2.0484e+00f, -1.2496e+00f, -1.7086e+00f, -6.7616e-02f, -1.1049e-01f, -9.1274e-02f, -1.0762e-01f, -5.8369e-02f, 

        -6.4319e-02f, -9.4483e-02f, -6.4945e-02f, -1.2035e-01f, -3.5360e-01f, -2.6208e-01f, -1.7646e-01f, -5.1059e-01f, 

        -2.1153e-01f, -2.0967e-01f, -2.6623e-01f, -1.7298e-01f, -6.8653e-02f, -1.4592e-01f, -1.6425e-01f, -9.0085e-02f, 

        -9.5408e-02f, -1.2769e-01f, -1.8403e-01f, -1.1838e-01f, -1.5245e-01f, -5.5496e-02f, -1.2926e-01f, -4.9154e-02f, 

        -1.3676e-01f, -1.1891e-01f, -5.8153e-02f, -6.9319e-02f, -1.1527e-01f, -8.4316e-02f, -1.0497e-01f, -1.9631e-01f, 

        -1.9709e-01f, -1.4112e-01f, -2.6129e-01f, -2.2007e-01f, -2.7474e-01f, -2.2650e-01f, -2.3671e-01f, -2.0804e-01f, 

        -2.7568e-01f, -2.1491e-01f, -4.2221e-01f, -1.6847e-01f, -8.7752e-02f, -1.3351e-01f, -1.6220e-01f, -2.0373e-01f, 

        -2.0598e-01f, -2.2990e-01f, -2.0767e-01f, -2.9202e-01f, -2.6678e-01f, -2.7781e-01f, -3.2600e-01f, -2.6789e-01f, 

        -1.9821e-01f, -2.3459e-01f, -1.2340e-01f, -2.1238e-01f, -1.1318e-01f, -1.0867e-01f, -1.2793e-01f, -1.5727e-01f, 

        -1.4808e-01f, -1.8025e-01f, -1.9719e-01f, -3.8531e-01f, -1.8568e-01f, -2.7967e-01f, -2.6503e-01f, -1.8077e-01f, 

        -2.4014e-01f, -1.6096e-01f, -2.0236e-01f, -3.2104e-01f, -1.9080e-01f, -1.0278e-01f, -1.5800e-01f, -1.5726e-01f, 

        -1.2775e-01f, -1.3590e-01f, -1.7573e-01f, -1.2893e-01f, -1.7393e-01f, -2.1436e-01f, -1.6635e-01f, -1.8723e-01f, 

        -1.7787e-01f, -2.3288e-01f, -1.9401e-01f, -1.9981e-01f, -2.3249e-01f, -2.1323e-01f, -1.3876e-01f, -9.8524e-02f, 

        -1.4187e-01f, -2.5979e-01f, -1.1544e-01f, -1.3120e-01f, -7.0629e-02f, -1.0280e-01f, -1.5866e-01f, -3.6009e-01f, 

        -1.3711e-01f, -1.1098e-01f, -6.9706e-02f, -3.6413e-01f, -8.4189e-02f, -3.1388e-01f, -1.0995e-01f, -1.2067e-01f, 

        -1.5523e-01f, -1.1065e-01f, -1.4765e-01f, -1.5529e-01f, -1.4903e-01f, -2.2189e-01f, -1.6530e-01f, -8.9638e-02f, 

        -1.1260e-01f, -9.1524e-02f, -1.4139e-01f, -8.3455e-02f, -1.1100e-01f, -1.5937e-01f, -1.6952e-01f, -8.8226e-02f, 

        -1.3091e-01f, -1.2386e-01f, -2.1620e-01f, -1.7202e-01f, -1.7779e-01f, -2.6239e-01f, -1.1195e-01f, -1.8433e-01f, 

        -1.6055e-01f, -2.2875e-01f, -1.2506e-01f, -2.1053e-01f, -1.3314e-01f, -1.4068e-01f, -1.1517e-01f, -1.6684e-01f, 

        -1.6813e-01f, -9.5995e-02f, -1.7725e-01f, -1.0403e-01f, -1.8908e-01f, -8.5463e-02f, -1.2895e-01f, -1.4437e-01f, 

        -1.2504e-01f, -1.0146e-01f, -1.2868e-01f, -7.2190e-02f, -1.5894e-01f, -1.1991e-01f, -4.8676e-02f, -7.3698e-02f, 

        -2.6693e-01f, -1.5521e-01f, -2.7228e-01f, -1.4241e-01f, -1.9716e-01f, -1.1342e-01f, -1.2074e-01f, -1.1631e-01f, 

        -1.4234e-01f, -8.7817e-02f, -1.0197e-01f, -1.5951e-01f, -1.2072e-01f, -1.4944e-01f, -1.0522e-01f, 1.4067e-02f, 

        -1.4430e-04f, -1.4171e-01f, -9.6737e-02f, -1.4000e-01f, -4.4590e-02f, -7.3232e-02f, 3.1223e-02f, -1.3870e-01f, 

        -1.9965e-01f, -1.5011e-01f, -1.8784e-01f, -1.7828e-01f, -1.0331e-01f, -1.2205e-01f, -1.4087e-01f, -6.6054e-02f, 

        -1.5743e-01f, -1.5363e-01f, -7.9606e-02f, -1.0677e-01f, -9.8010e-02f, -1.4700e-01f, -1.0147e-01f, -5.5410e-02f, 

        -2.7887e-02f, -5.8361e-02f, -1.9913e-01f, -1.1295e-01f, -1.3285e-01f, -2.1698e-01f, -1.3223e-01f, -2.0435e-01f, 

        -1.3590e-01f, -1.8147e-01f, -1.8787e-01f, -1.6633e+00f, -1.3468e+00f, -1.0932e+00f, -8.2117e-01f, -6.3199e-01f, 

        -4.9968e-01f, -9.5605e-01f, -3.7282e-01f, -5.1692e-01f, -1.8970e-01f, -1.8616e-01f, -1.7863e-01f, -8.8766e-02f, 

        -1.4483e-01f, -1.2882e-01f, -1.7501e-01f, -1.3624e-01f, -1.0559e-01f, -3.3843e-01f, -2.0525e-01f, -2.1887e-01f, 

        -2.5771e-01f, -3.0586e-01f, -2.6053e-01f, -3.3928e-01f, -1.4305e-01f, -1.8738e-01f, -1.7946e-01f, -1.9968e-01f, 

        -1.4025e-01f, -1.4961e-01f, -1.5445e-01f, -1.6139e-01f, -7.1340e-02f, -1.5903e-01f, -7.7403e-02f, 7.9188e-03f, 

        -2.5457e-01f, -2.3313e-01f, -2.0107e-01f, -2.2227e-01f, -2.0092e-01f, -1.6608e-01f, -1.6509e-01f, -1.7818e-01f, 

        -2.2168e-01f, -1.6833e-01f, -2.4169e-01f, -2.2784e-01f, -1.5297e-01f, -2.6252e-01f, -1.7268e-01f, -1.1427e-01f, 

        -1.2609e-01f, -1.1283e-01f, -1.1957e-01f, -1.6050e-01f, -7.1193e-02f, -1.4621e-01f, -1.2889e-01f, -1.2367e-01f, 

        -1.0513e-01f, -1.7747e-01f, -1.7083e+00f, -1.0770e+00f, -1.8169e+00f, -1.3708e+00f, -1.1171e+00f, -1.7723e+00f, 

        -1.2831e+00f, -1.0293e+00f, -1.3199e+00f, -6.8307e-02f, -8.9369e-02f, -1.0415e-02f, -7.9125e-02f, -8.0259e-02f, 

        -1.4355e-01f, -9.2789e-02f, -1.7385e-01f, -1.2645e-01f, -3.4616e-01f, -2.5940e-01f, -1.6035e-01f, -3.9774e-01f, 

        -1.7665e-01f, -1.6061e-01f, -1.7295e-01f, -2.5504e-01f, -1.6110e-01f, -5.7730e-02f, -1.2599e-01f, -6.9759e-02f, 

        -1.5617e-01f, -1.3902e-01f, -1.8107e-01f, -1.6882e-01f, -1.7710e-01f, -9.3814e-02f, -1.2079e-01f, -4.1232e-02f, 

        -5.9292e-02f, 1.5036e-02f, -6.4743e-02f, -5.3747e-02f, -3.3467e-02f, -3.2563e-02f, -5.8594e-02f, -1.2367e-01f, 

        -1.2031e-01f, -1.9362e-01f, -2.0627e-01f, -2.4373e-01f, -1.9231e-01f, -2.2127e-01f, -2.4218e-01f, -1.2880e-01f, 

        -2.0605e-01f, -2.2297e-01f, -4.0270e-01f, -1.2996e-01f, -1.2436e-01f, -1.3726e-01f, -7.3459e-02f, -3.4256e-02f, 

        -1.1682e-01f, -2.6039e-01f, -2.3925e-01f, -2.2694e-01f, -2.7054e-01f, -1.9354e-01f, -2.7544e-01f, -1.6213e-01f, 

        -1.8623e-01f, -1.4724e-01f, -1.2021e-01f, -2.0213e-01f, -8.7944e-02f, -1.5579e-01f, -1.2456e-01f, -1.8035e-01f, 

        -2.3330e-01f, -1.8489e-01f, -1.8179e-01f, -3.2367e-01f, -2.2820e-01f, -3.4803e-01f, -1.5557e-01f, -2.7743e-01f, 

        -2.1486e-01f, -1.9267e-01f, -2.2970e-01f, -2.8879e-01f, -9.2833e-02f, -7.6015e-02f, -1.4252e-01f, -1.8194e-01f, 

        -1.2926e-01f, -1.0145e-01f, -1.6366e-01f, -1.4700e-01f, -1.6966e-01f, -1.3466e-01f, -2.0020e-01f, -1.9962e-01f, 

        -1.8574e-01f, -1.6988e-01f, -1.3319e-01f, -1.1101e-01f, -1.2854e-01f, -1.5830e-01f, -1.2315e-01f, -8.4314e-02f, 

        -1.6080e-01f, -1.9487e-01f, -9.1896e-02f, -1.1310e-01f, -1.2508e-01f, -1.3557e-01f, -8.0917e-02f, -3.0471e-01f, 

        -1.1788e-01f, -2.2211e-01f, -1.3820e-01f, -1.5872e-01f, -1.3851e-01f, -1.9555e-01f, -5.6298e-02f, -4.2778e-02f, 

        -2.0869e-01f, -9.3955e-02f, -1.0413e-01f, -1.2775e-01f, -1.7152e-01f, -2.2476e-01f, -1.5793e-01f, -1.1267e-01f, 

        -1.1886e-01f, -1.1094e-01f, -1.6286e-01f, -8.0543e-02f, -1.0983e-01f, -1.0552e-01f, -1.7288e-01f, -1.2718e-01f, 

        -1.3797e-01f, -7.9927e-02f, -1.2952e-01f, -1.9805e-01f, -2.1130e-01f, -2.0883e-01f, -1.6816e-01f, -1.8044e-01f, 

        -9.9395e-02f, -2.2748e-01f, -1.1430e-01f, -1.5149e-01f, -1.5630e-01f, -1.6621e-01f, -1.2101e-01f, -1.8156e-01f, 

        -1.5577e-01f, -1.1408e-01f, -1.8155e-01f, -6.8981e-02f, -1.6185e-01f, -1.2603e-01f, -1.5061e-01f, -1.2527e-01f, 

        -1.3491e-01f, -1.3162e-01f, -1.4376e-01f, -7.1321e-02f, -9.4041e-02f, -5.6216e-02f, -2.0992e-01f, -2.1990e-01f, 

        -1.4012e-01f, -9.7119e-02f, -1.4180e-01f, -1.7280e-01f, -1.5181e-01f, -1.4014e-01f, -1.3313e-01f, -1.0100e-01f, 

        -1.7218e-01f, -9.5409e-02f, -1.2107e-01f, -1.6593e-01f, -7.2822e-02f, -1.8091e-01f, -8.2380e-02f, -8.7998e-02f, 

        -9.6799e-02f, -1.2221e-01f, -1.2218e-01f, -1.1046e-01f, -6.3664e-02f, -9.5772e-02f, -7.9229e-02f, -1.7187e-01f, 

        -3.8268e-02f, 2.6518e-03f, -5.3751e-02f, -3.2340e-02f, 7.8179e-02f, -2.1091e-02f, -4.7652e-02f, -5.9877e-02f, 

        -2.5645e-02f, 9.9895e-02f, 6.5028e-03f, 8.1166e-02f, 2.0473e-02f, 9.2364e-02f, 1.2530e-01f, 9.2192e-02f, 

        5.7406e-02f, 5.1975e-02f, -1.6335e-01f, -1.1110e-01f, -3.4686e-02f, -7.1862e-02f, -7.3641e-03f, -1.5144e-01f, 

        -2.8806e-02f, -1.8525e-01f, -1.4036e-01f, -2.4860e-01f, -1.3224e-02f, -1.3931e-01f, -1.0388e-01f, 1.7795e-02f, 

        -1.0354e-01f, -1.4933e-01f, -2.8573e-01f, -9.8841e-02f, 2.3449e-02f, -4.9740e-02f, 4.2085e-02f, 7.8545e-02f, 

        5.6964e-02f, 5.0285e-02f, -4.1202e-03f, -4.0966e-02f, -8.5021e-02f, 3.0936e-02f, -1.8088e-02f, 5.6904e-02f, 

        8.6665e-02f, -6.3628e-03f, 1.1168e-01f, -7.4156e-02f, 2.1194e-02f, 1.0031e-01f, -1.4068e-01f, -4.1962e-02f, 

        -1.6123e-01f, 6.4061e-02f, -1.9628e-02f, -1.1782e-01f, -5.0282e-02f, -2.0152e-02f, -1.2959e-01f, -1.3533e-01f, 

        4.2382e-02f, -4.7299e-02f, 5.8719e-02f, -7.0668e-02f, -9.7009e-02f, 1.3335e-03f, -6.5894e-02f, 1.1344e-01f, 

        -3.1948e-02f, -4.6147e-02f, -6.0117e-02f, -5.5718e-02f, -3.7070e-02f, -1.8024e-02f, -1.1355e-01f, -3.6020e-02f, 

        2.8520e-02f, 5.7377e-02f, 3.7985e-02f, -9.2140e-02f, 1.1577e-01f, 2.4973e-02f, 1.4709e-02f, -4.1211e-02f, 

        -1.7883e-02f, -8.6502e-02f, 6.2535e-02f, 6.5433e-02f, -1.1539e-01f, -1.2610e-01f, 1.4545e-02f, -1.2501e-01f, 

        -1.3218e-01f, -2.7009e-02f, -1.3426e-01f, -9.4612e-03f, 4.5607e-02f, 5.6686e-02f, 9.2961e-02f, 1.3584e-01f, 

        8.6605e-02f, 4.5201e-03f, 3.7817e-03f, -6.0338e-02f, -1.1373e-01f, -1.6663e-01f, -2.2276e-01f, 4.6580e-02f, 

        -1.9216e-02f, -2.0014e-01f, -2.0492e-01f, -1.0984e-01f, -3.7204e-01f, 1.8640e-01f, 4.3046e-02f, -5.9341e-03f, 

        9.3163e-02f, 7.9703e-02f, 3.1016e-02f, 1.2719e-01f, 7.7544e-02f, 6.0625e-03f, -2.3996e-02f, 5.2603e-02f, 

        2.5211e-02f, -4.1127e-03f, 1.6086e-01f, 2.8839e-02f, -6.0816e-02f, 9.2596e-02f, -2.6461e-02f, 1.0004e-01f, 

        3.1763e-03f, -3.9186e-02f, -7.1450e-02f, -2.3394e-02f, 2.7771e-02f, 7.9847e-03f, 2.0298e-03f, -4.5839e-02f, 

        -1.9457e-01f, -5.3164e-02f, -5.9719e-02f, -8.4737e-02f, -5.9946e-02f, -3.8368e-02f, 2.3770e-02f, -1.0132e-01f, 

        -2.9682e-01f, 1.9571e-02f, -3.2732e-02f, -9.1943e-02f, 9.4419e-03f, 4.4709e-02f, 3.8378e-03f, 3.2246e-03f, 

        -6.2195e-02f, 5.5888e-02f, -2.7795e-01f, -2.1750e-01f, -1.9185e-01f, -9.7790e-02f, -1.8374e-01f, -1.1914e-01f, 

        -1.7576e-01f, -2.5703e-01f, -2.6990e-01f, 4.7247e-03f, -3.7847e-02f, -8.1789e-02f, 3.2419e-02f, -4.5009e-02f, 

        -5.6071e-02f, 6.2799e-02f, -4.6643e-02f, -1.1096e-01f, -7.9420e-02f, -8.6963e-02f, -1.5786e-01f, -1.8658e-02f, 

        -3.5072e-02f, -3.4040e-02f, 5.3762e-02f, 3.9756e-02f, -1.2794e-01f, 3.2925e-02f, -1.0047e-02f, -3.0035e-02f, 

        -3.8931e-02f, 5.6027e-02f, 1.2750e-02f, 3.9579e-02f, 6.2740e-02f, -1.0047e-01f, -6.3570e-03f, 1.3669e-02f, 

        -7.8273e-02f, 5.2254e-02f, 3.1620e-02f, -9.9068e-02f, -2.8934e-03f, 1.6790e-01f, -1.1577e-01f, 1.3621e-02f, 

        -6.6554e-02f, -4.3903e-03f, -6.3919e-02f, 9.9349e-02f, -9.5168e-02f, 1.9095e-01f, 1.4183e-01f, 3.7411e-02f, 

        -1.7218e-01f, -1.4090e-02f, -1.1936e-01f, -5.2410e-03f, 6.9205e-02f, -1.1423e-01f, -1.4587e-01f, -1.5080e-01f, 

        -1.6290e-01f, 4.2275e-03f, 4.4973e-02f, -1.8123e-02f, 3.5917e-02f, 9.7616e-02f, 4.2522e-02f, 2.4057e-02f, 

        -4.6120e-02f, -9.5617e-02f, 1.3510e-01f, -1.6750e-01f, -3.4122e-02f, -9.3248e-02f, -1.8291e-01f, -1.4056e-01f, 

        5.5497e-03f, -1.2551e-01f, 4.8670e-02f, 1.3725e-01f, 9.5808e-03f, -6.6436e-02f, 8.7445e-02f, 1.7960e-02f, 

        -1.5729e-02f, 3.8361e-02f, -5.3719e-02f, -1.9163e-02f, 7.7604e-03f, -3.5775e-02f, -8.8748e-02f, 1.8146e-02f, 

        4.9153e-02f, -7.9446e-03f, 5.1325e-02f, 4.0103e-03f, -4.1869e-02f, -1.4944e-01f, 2.6242e-01f, 1.4112e-02f, 

        -1.1581e-02f, -4.7241e-02f, 2.2919e-02f, -6.4005e-02f, 4.2652e-02f, 2.1243e-02f, 4.1786e-02f, 4.4709e-02f, 

        1.0624e-02f, 1.4827e-01f, 1.1539e-01f, 4.6481e-02f, 8.4292e-02f, -7.8780e-02f, -3.3205e-03f, -6.5240e-02f, 

        -2.3774e-01f, -2.4979e-01f, -2.2632e-01f, -3.2857e-01f, -3.3952e-01f, -8.7747e-02f, -1.8578e-01f, -3.1305e-01f, 

        -1.4544e-01f, -1.4354e-01f, -1.4515e-01f, -1.1530e-01f, -9.1648e-02f, -1.5925e-01f, 8.2301e-02f, -1.2460e-01f, 

        -1.3186e-01f, -5.1547e-02f, -1.2645e-01f, -4.2575e-02f, -4.9264e-02f, -1.9339e-01f, -1.4264e-01f, -1.2879e-01f, 

        -8.7787e-02f, -1.9492e-01f, -1.7027e-01f, -1.5992e-01f, -1.0532e-01f, -1.0661e-01f, -1.1385e-01f, -1.0570e-01f, 

        -7.5513e-02f, -1.1548e-01f, -8.5335e-02f, -2.7551e-01f, -8.3487e-01f, -3.2645e-01f, -3.9237e-01f, -1.9270e-01f, 

        -9.4688e-02f, -1.2929e-01f, -2.0212e-01f, 5.1751e-02f, -8.8057e-02f, -1.2877e-01f, -1.4349e-01f, -1.4604e-01f, 

        -1.0121e-01f, -9.9529e-02f, -2.1094e-02f, -2.7180e-02f, -1.5483e-01f, -4.2121e-01f, -1.6596e-01f, -1.3344e-01f, 

        -2.9059e-01f, -8.8833e-02f, -1.6639e-01f, 1.4231e-01f, -5.9620e-02f, -3.4719e-01f, -1.7740e-01f, -6.3489e-02f, 

        -1.7850e-01f, -1.3217e-01f, -1.1660e-01f, -3.2771e-02f, -1.0956e-01f, -2.1296e-02f, -1.0074e-01f, -9.2464e-02f, 

        -1.9945e-01f, -1.1496e-01f, -1.0987e-01f, -7.0795e-02f, -8.4387e-02f, -1.0414e-01f, -1.5904e-01f, -6.4244e-02f, 

        -1.3055e-01f, -1.5795e-01f, -2.1396e-01f, -1.5581e-01f, -7.3402e-02f, -3.5044e-01f, -7.3143e-03f, -1.1951e-01f, 

        -1.2480e-01f, -1.5091e-01f, -1.1679e-01f, -2.1192e-01f, -1.1017e-01f, -1.5542e-01f, -6.1176e-02f, -5.5342e-02f, 

        -1.1723e-01f, -1.3031e-01f, -4.0545e-01f, -2.6972e-01f, -1.9658e-01f, -3.1085e-01f, -2.0417e-01f, -2.0951e-01f, 

        -1.7742e-01f, -2.4148e-01f, -2.0066e-01f, -1.9658e-01f, -8.5935e-02f, -9.7972e-02f, -1.0396e-01f, -1.3666e-01f, 

        -7.6867e-02f, -1.7377e-01f, -1.0879e-01f, -1.9895e-01f, -5.0399e-01f, -4.7220e-01f, -3.1130e-01f, -5.3791e-01f, 

        -2.8716e-01f, -1.9544e-01f, -3.5241e-01f, -3.8355e-01f, -1.4028e-01f, -1.7531e-01f, -1.2861e-01f, -1.1674e-01f, 

        -2.0446e-01f, -1.6847e-01f, -1.6097e-01f, -1.4836e-01f, -9.2887e-02f, -9.7535e-02f, -2.1988e-02f, -8.6079e-02f, 

        -5.9211e-02f, 1.5010e-02f, -9.5613e-02f, -3.9927e-01f, -1.8943e-01f, -2.0004e-01f, -1.5015e-02f, -1.4998e-01f, 

        -9.8669e-02f, -1.8338e-01f, -5.9791e-02f, -1.9029e-01f, -1.3055e-01f, 3.4566e-02f, -3.8201e-02f, -1.1023e-01f, 

        -7.1949e-01f, -3.3245e-01f, -4.3979e-01f, -1.5212e-01f, -7.3063e-02f, -2.3458e-02f, -8.0520e-02f, -4.2114e-02f, 

        -2.9986e-01f, -1.4127e-01f, -9.7268e-02f, -8.5028e-02f, -3.7096e-02f, 1.8309e-02f, -4.2212e-02f, 1.7751e-02f, 

        9.0328e-03f, 3.8459e-03f, -2.4610e-01f, -8.2033e-02f, -3.5970e-01f, 2.0892e-03f, -2.4800e-01f, -2.9590e-01f, 

        -2.0794e-01f, -2.1887e-01f, -2.1226e-01f, -2.2798e-01f, -9.0244e-02f, -8.9060e-02f, -7.7982e-02f, -6.3529e-03f, 

        -1.4016e-01f, -1.7511e-01f, -9.4954e-02f, -1.4471e-01f, -1.5937e-01f, -2.0974e-01f, -1.5204e-01f, -6.4702e-02f, 

        -1.1330e-01f, -1.6077e-01f, -3.7234e-02f, -1.7294e-01f, -3.0471e-01f, -9.2884e-02f, -1.0801e-01f, -1.0993e-01f, 

        -2.8083e-02f, 1.0541e-02f, -1.2710e-01f, -1.7078e-01f, -1.3699e-01f, -3.4678e-02f, -3.0920e-01f, -2.4096e-01f, 

        -2.3737e-01f, -1.7477e-01f, -2.4028e-01f, -2.1094e-01f, -1.2304e-01f, -1.8999e-01f, -2.6331e-01f, -3.0980e-01f, 

        -2.4981e-01f, -2.3625e-01f, -2.6630e-01f, -1.7319e-01f, -2.2172e-01f, -3.3401e-01f, -1.6428e-01f, -3.2462e-01f, 

        -9.1718e-02f, -1.0331e-01f, 2.4653e-02f, -2.0925e-01f, -2.7799e-03f, -3.6139e-01f, 2.1901e-02f, 9.3219e-02f, 

        8.9469e-02f, -1.4249e-01f, -2.0043e-01f, -1.2239e-01f, -1.2757e-01f, -5.1472e-02f, -1.2629e-01f, -4.9932e-02f, 

        -1.7397e-02f, -1.4884e-01f, -1.3611e-01f, -1.4642e-01f, -2.6562e-01f, -1.8210e-01f, -2.2356e-01f, -1.4834e-01f, 

        -2.2366e-01f, -5.0865e-02f, -2.3794e-01f, -1.5724e-01f, -1.4009e-01f, -8.1782e-02f, -8.8283e-02f, -3.1344e-02f, 

        -8.0608e-02f, -3.7149e-02f, -1.0916e-01f, -1.8253e-01f, -6.7019e-02f, -7.9003e-02f, -5.4471e-02f, -7.2549e-02f, 

        -3.4612e-02f, -1.6489e-01f, -4.1977e-02f, -4.9226e-02f, -1.6692e-01f, -8.4825e-02f, -1.3026e-01f, -1.5738e-01f, 

        -6.5907e-02f, -6.1407e-02f, -7.9829e-02f, -1.2209e-01f, -2.8377e-02f, -5.5416e-02f, -3.2436e-02f, 3.0669e-02f, 

        -3.3740e-02f, 2.4602e-02f, 1.2089e-02f, -8.5443e-02f, -1.1199e-01f, -2.0346e-01f, -2.5679e-01f, -8.5919e-02f, 

        -1.3342e-01f, -9.9976e-02f, -1.3236e-01f, -6.6170e-02f, -9.3666e-02f, -2.0580e-01f, -1.6109e-01f, -1.7789e-01f, 

        -2.2056e-01f, -1.1914e-01f, -1.2079e-01f, -1.7888e-01f, -1.9081e-01f, -9.7227e-02f, -1.3343e-01f, -8.3764e-02f, 

        -1.7085e-01f, -8.3904e-02f, -1.2683e-01f, -1.3482e-01f, -1.2974e-01f, -1.4985e-01f, -1.1377e-01f, -1.0352e-01f, 

        -1.5541e-01f, -6.6253e-02f, -1.0374e-01f, -1.4587e-01f, -1.7170e-01f, -2.0764e-01f, -1.8858e-01f, -1.7846e-01f, 

        -1.0446e-01f, -1.1888e-01f, -1.5607e-01f, -1.4615e+00f, -1.4383e+00f, -1.4267e+00f, -8.4124e-01f, -6.4219e-01f, 

        -6.5802e-01f, -8.1450e-01f, -4.1849e-01f, -6.1200e-01f, -1.1738e-01f, -1.6703e-01f, -1.4135e-01f, -1.7905e-01f, 

        -1.7434e-01f, -1.5638e-01f, -1.6770e-01f, -8.7551e-02f, -1.7776e-01f, -3.9433e-01f, -2.7621e-01f, -2.1008e-01f, 

        -3.5928e-01f, -3.4610e-01f, -1.6990e-01f, -3.4729e-01f, -2.0377e-01f, -2.0567e-01f, -1.7829e-01f, -1.9749e-01f, 

        -5.6424e-02f, -1.9899e-01f, -1.8154e-01f, -7.4148e-02f, -8.8705e-02f, -1.7068e-01f, -8.3431e-02f, -6.5210e-02f, 

        -2.1035e-01f, -2.4761e-01f, -2.4527e-01f, -2.0476e-01f, -1.2709e-01f, -1.9343e-01f, -1.1242e-01f, -2.6234e-01f, 

        -1.6937e-01f, -5.6204e-02f, -2.1037e-01f, -2.4731e-01f, -1.9652e-01f, -2.7055e-01f, -1.9516e-01f, -2.0851e-01f, 

        -1.7981e-01f, -1.5301e-01f, -7.6483e-02f, -1.2747e-01f, -1.1614e-01f, -1.3195e-01f, -1.0924e-01f, -8.5099e-02f, 

        -1.0184e-01f, -1.2814e-01f, -2.0900e+00f, -1.1036e+00f, -1.9100e+00f, -1.3048e+00f, -1.2702e+00f, -1.7364e+00f, 

        -1.5522e+00f, -1.1743e+00f, -1.2842e+00f, -6.1137e-02f, -9.0437e-02f, -5.2002e-02f, -5.7482e-02f, -1.1153e-01f, 

        -8.6475e-02f, -1.3296e-02f, -1.3706e-02f, -1.4050e-02f, -4.1239e-01f, -2.0516e-01f, -1.9610e-01f, -5.3748e-01f, 

        -1.5865e-01f, -2.1163e-01f, -1.8066e-01f, -1.6706e-01f, -1.4639e-01f, -1.7352e-01f, -1.5662e-01f, -1.5441e-01f, 

        -9.6819e-02f, -1.6318e-01f, -1.0851e-01f, -1.0041e-01f, -1.1499e-01f, -1.4023e-01f, -1.2474e-01f, -1.0471e-01f, 

        -1.3507e-01f, -1.3394e-01f, -1.1695e-01f, -1.2478e-01f, -9.2555e-02f, -9.3807e-02f, -7.1576e-02f, -2.4078e-01f, 

        -2.2179e-01f, -2.3612e-01f, -1.9960e-01f, -2.1440e-01f, -2.0806e-01f, -2.4305e-01f, -1.9116e-01f, -1.3459e-01f, 

        -3.1904e-01f, -1.4890e-01f, -2.0106e-01f, -1.0105e-01f, -1.0182e-01f, -9.5489e-02f, -1.9493e-01f, -1.5233e-01f, 

        -1.3822e-01f, -2.6590e-01f, -2.2890e-01f, -2.5751e-01f, -2.4029e-01f, -2.2771e-01f, -2.8151e-01f, -1.7457e-01f, 

        -2.2858e-01f, -1.8125e-01f, -1.5282e-01f, -1.9770e-01f, -2.0586e-01f, -1.2731e-01f, -1.0530e-01f, -1.6016e-01f, 

        -1.8166e-01f, -1.2348e-01f, -1.7937e-01f, -3.0193e-01f, -2.0709e-01f, -3.8623e-01f, -2.7738e-01f, -2.8310e-01f, 

        -1.9697e-01f, -2.9402e-01f, -2.4340e-01f, -2.6539e-01f, -1.8186e-01f, -1.8164e-01f, -1.5841e-01f, -1.7820e-01f, 

        -9.5376e-02f, -7.0036e-02f, -1.0902e-01f, -1.5530e-01f, -1.1049e-01f, -2.3785e-01f, -2.1020e-01f, -1.9147e-01f, 

        -2.4320e-01f, -1.1500e-01f, -1.1204e-01f, -1.4050e-01f, -1.3397e-01f, -9.8996e-02f, -1.5745e-01f, -1.1966e-01f, 

        -1.4697e-01f, -2.2655e-01f, -1.8071e-01f, -1.1962e-01f, -1.5966e-01f, -1.5738e-01f, -7.3705e-02f, -3.5836e-01f, 

        -1.7001e-01f, -1.5727e-01f, -2.0169e-01f, -2.0572e-01f, -9.7169e-02f, -1.1771e-01f, -1.2988e-01f, -6.5195e-04f, 

        -1.0452e-01f, -1.4779e-01f, -1.7740e-01f, -1.6670e-01f, -1.1397e-01f, -1.4229e-01f, -7.5193e-02f, -7.5613e-02f, 

        -9.1746e-02f, -1.7360e-01f, -1.2995e-01f, -1.1640e-01f, -1.4218e-01f, -1.0096e-01f, -8.5775e-02f, -1.5817e-01f, 

        -1.3776e-01f, -1.6184e-01f, -1.9297e-01f, -1.7337e-01f, -1.0757e-01f, -1.7159e-01f, -9.8441e-02f, -1.8613e-01f, 

        -1.7472e-01f, -2.1036e-01f, -1.7160e-01f, -1.3173e-01f, -9.2576e-02f, -1.7794e-01f, -1.7852e-01f, -1.5542e-01f, 

        -1.0301e-01f, -1.0216e-01f, -8.7045e-02f, -7.4531e-02f, -1.7106e-01f, -1.1438e-01f, -1.6046e-01f, -1.5682e-01f, 

        -1.2150e-01f, -1.1137e-01f, -1.6855e-01f, -1.1189e-01f, -8.3493e-02f, -9.1626e-02f, -2.0803e-01f, -1.5797e-01f, 

        -2.3242e-01f, -1.8672e-01f, -1.1033e-01f, -1.3576e-01f, -1.4558e-01f, -2.1360e-01f, -8.2037e-02f, -1.5471e-01f, 

        -1.3691e-01f, -1.2287e-01f, -1.6526e-01f, -1.0981e-01f, -7.1955e-02f, -9.0960e-02f, -1.3585e-01f, -1.1984e-01f, 

        -1.7238e-01f, -1.6408e-01f, -1.2555e-01f, -2.4642e-02f, 1.3879e-03f, -1.0919e-01f, -1.2841e-01f, 2.4620e-02f, 

        -9.4835e-02f, -3.6966e-02f, -2.8535e-02f, -1.6623e-01f, -1.2364e-01f, -7.7478e-02f, -1.4379e-01f, -1.6288e-01f, 

        -1.3409e-01f, -1.3579e-01f, -1.0511e-02f, -1.0009e-01f, -1.7167e-01f, -8.4900e-02f, -3.3516e-03f, -6.5292e-02f, 

        -1.3985e-01f, -1.0290e-01f, -4.9139e-02f, -1.1410e-01f, -1.1838e-01f, -6.4476e-04f, -8.2456e-02f, -1.0888e-01f, 

        -4.8208e-02f, -1.2398e-01f, -1.1135e-01f, -6.4022e-01f, -5.1142e-01f, -5.0116e-01f, -4.2635e-01f, -5.1128e-01f, 

        -4.2896e-01f, -5.1063e-01f, -4.0934e-01f, -3.9297e-01f, -7.5201e-02f, -8.8764e-02f, -8.0341e-02f, -4.6006e-02f, 

        -5.6462e-02f, -1.4841e-01f, -1.6658e-01f, -1.7230e-01f, -4.6040e-02f, -2.2767e-01f, -2.3780e-01f, -2.0176e-01f, 

        -2.5074e-01f, -1.0629e-01f, -9.7995e-02f, -1.0788e-01f, -1.4895e-02f, -1.1893e-01f, -6.6904e-02f, -1.4023e-01f, 

        -1.7207e-01f, -1.1240e-01f, -8.9017e-02f, -1.4114e-01f, -1.5466e-01f, -1.2670e-01f, -7.9989e-02f, -1.1233e-01f, 

        -1.1532e-02f, -2.3685e-01f, -1.1880e-01f, -6.9300e-02f, -1.1623e-01f, -1.4468e-02f, -1.7939e-01f, -9.4560e-02f, 

        -4.9121e-02f, -5.7411e-02f, -4.2277e-02f, -1.5632e-01f, -1.4478e-01f, -1.4515e-01f, -1.6112e-01f, -1.5704e-01f, 

        -9.1623e-02f, -5.4759e-02f, -5.1744e-02f, -1.1650e-01f, -5.3933e-02f, -5.3839e-02f, -1.0803e-01f, -1.2626e-01f, 

        -8.4437e-02f, -7.3707e-02f, -1.7300e+00f, -1.3274e+00f, -1.4160e+00f, -9.0201e-01f, -1.1312e+00f, -1.1096e+00f, 

        -1.0371e+00f, -9.6255e-01f, -8.6223e-01f, -4.8954e-02f, -2.5473e-02f, -6.0141e-02f, -4.0538e-02f, -3.0685e-02f, 

        -3.4097e-02f, -1.1318e-01f, -1.5630e-01f, -1.2722e-01f, -1.7080e-01f, -1.4874e-01f, -1.2603e-01f, -3.6445e-01f, 

        -1.0953e-01f, -1.7883e-01f, -1.8664e-01f, -1.8257e-01f, -1.5551e-01f, -6.0938e-02f, -1.0664e-01f, -1.5531e-01f, 

        -4.4477e-02f, 2.5486e-02f, -1.1483e-01f, -1.2829e-01f, -8.8275e-02f, -8.3463e-02f, -1.3708e-01f, -7.6691e-02f, 

        -4.2630e-02f, -1.2406e-01f, -8.2399e-02f, -2.0978e-02f, 1.0126e-02f, -4.4934e-03f, 4.0355e-02f, -9.2846e-02f, 

        -6.8054e-02f, -1.0159e-01f, -2.0727e-01f, -7.0683e-02f, -7.7374e-02f, -9.4355e-02f, -1.6488e-02f, -1.7306e-01f, 

        -2.2136e-01f, -8.2093e-02f, -1.8482e-01f, -9.9759e-02f, -1.5393e-01f, -3.0814e-02f, -7.9454e-02f, -1.3357e-01f, 

        -5.8384e-02f, -5.9718e-02f, -5.7740e-02f, -1.8174e-01f, -1.7369e-01f, -2.1309e-01f, -1.3182e-01f, -1.4481e-01f, 

        -1.6994e-01f, -1.5371e-01f, -2.1788e-01f, -1.2796e-01f, -7.4610e-02f, -1.2015e-01f, -1.0019e-01f, -5.4062e-02f, 

        -7.0081e-02f, -5.5849e-02f, -1.2411e-01f, -2.6099e-01f, -1.9189e-01f, -2.2376e-01f, -1.7638e-01f, -1.1228e-01f, 

        -9.2781e-02f, -1.8248e-02f, -1.1779e-01f, -1.4343e-01f, -1.6962e-01f, -1.1515e-01f, -1.4515e-01f, -7.7806e-02f, 

        -4.0797e-02f, -3.7144e-02f, -6.3434e-02f, -1.0663e-01f, -5.9148e-02f, -1.4000e-01f, -1.9818e-01f, -1.5956e-01f, 

        -1.8660e-01f, -1.3146e-01f, -1.0119e-01f, -3.0071e-02f, -3.4423e-02f, -1.3470e-01f, -7.3120e-02f, -6.6835e-02f, 

        -9.8169e-02f, -2.1225e-01f, -1.0431e-01f, -1.0019e-01f, -8.9136e-02f, -6.4715e-02f, -3.4738e-02f, -1.7066e-01f, 

        -8.9026e-02f, -1.3267e-01f, -7.9938e-02f, -2.1448e-01f, -5.4204e-02f, -6.1718e-02f, -2.3898e-02f, -4.7153e-02f, 

        -6.8935e-02f, -1.6517e-01f, -9.5082e-02f, -1.1182e-01f, -2.5584e-02f, -1.4411e-02f, -6.3157e-02f, -1.3409e-01f, 

        -2.0968e-02f, -2.0173e-02f, -4.7212e-02f, -4.2415e-02f, -9.3698e-02f, -8.3636e-02f, -7.6462e-02f, -1.1215e-01f, 

        -1.2232e-01f, -9.5206e-02f, -1.1512e-01f, -2.3736e-01f, -1.8235e-01f, -1.2407e-01f, -1.1952e-01f, -9.6209e-02f, 

        -7.5202e-02f, -3.3827e-02f, -1.0618e-01f, -4.7688e-02f, -4.5479e-02f, -1.3735e-01f, -4.3664e-02f, -1.5165e-01f, 

        -1.4270e-01f, -1.2487e-01f, -3.8009e-02f, -6.4781e-02f, -6.8054e-02f, 1.9645e-03f, -1.4875e-01f, -1.3602e-01f, 

        -2.0570e-02f, -7.8680e-02f, -4.3504e-02f, -1.4169e-01f, -2.8526e-02f, -1.6627e-01f, -7.7811e-02f, -1.1729e-01f, 

        -1.8764e-01f, -4.4487e-02f, -1.1181e-01f, -1.1525e-02f, -1.1812e-01f, -1.0557e-01f, -8.9046e-02f, -1.1215e-01f, 

        -8.1283e-02f, -7.5916e-02f, -1.2407e-01f, -6.2641e-02f, -8.9931e-02f, -8.2580e-02f, -1.3085e-01f, 8.3247e-02f, 

        -9.3298e-02f, -2.4446e-02f, -3.9025e-02f, 2.3809e-02f, -2.9672e-02f, 3.9465e-02f, -5.5731e-02f, -4.2731e-03f, 

        -1.7262e-01f, -1.7610e-01f, -1.8064e-01f, -1.3781e-01f, -2.0207e-01f, -1.7832e-01f, -1.7842e-01f, -1.8081e-01f, 

        -2.0412e-01f, -1.3450e-01f, -1.0062e-01f, -1.8921e-01f, -1.2818e-01f, -1.7993e-01f, -1.0840e-01f, -9.7458e-02f, 

        -1.1372e-01f, -8.4337e-02f, -1.7187e-01f, -1.9038e-01f, -1.4147e-01f, -1.3573e-01f, -1.7016e-01f, -1.8688e-01f, 

        -1.7846e-01f, -1.3142e-01f, -1.1866e-01f, -1.3397e+00f, -1.3200e+00f, -1.4704e+00f, -7.6835e-01f, -6.7090e-01f, 

        -5.9070e-01f, -7.6050e-01f, -4.6433e-01f, -5.2564e-01f, -1.3544e-01f, -1.6802e-01f, -8.7573e-02f, -1.4487e-01f, 

        -1.1271e-01f, -6.2639e-02f, -2.3983e-01f, -8.9461e-02f, -1.6237e-01f, -4.1757e-01f, -2.9262e-01f, -2.6777e-01f, 

        -4.0157e-01f, -3.2724e-01f, -2.1956e-01f, -3.1916e-01f, -1.8705e-01f, -1.6719e-01f, -1.5452e-01f, -2.4807e-01f, 

        -1.4748e-01f, -1.2132e-01f, -2.1042e-01f, -1.3486e-01f, -1.3722e-01f, -1.6408e-01f, -8.5760e-02f, -1.7910e-01f, 

        -1.4417e-01f, -1.8762e-01f, -1.9953e-01f, -1.6483e-01f, -2.6410e-01f, -1.7594e-01f, -2.1299e-01f, -2.4522e-01f, 

        -2.5115e-01f, -1.5105e-01f, -2.7234e-01f, -2.8892e-01f, -1.4192e-01f, -2.4878e-01f, -1.9194e-01f, -1.9646e-01f, 

        -1.9079e-01f, -1.5822e-01f, -1.9433e-01f, -8.4715e-02f, -9.7015e-02f, -8.9081e-02f, -1.6703e-01f, -1.7976e-01f, 

        -8.3691e-02f, -1.6443e-01f, -2.1612e+00f, -1.3011e+00f, -1.4923e+00f, -1.2910e+00f, -1.4655e+00f, -1.5886e+00f, 

        -1.6702e+00f, -1.0633e+00f, -1.4714e+00f, -6.5469e-02f, -6.2841e-02f, -1.2252e-01f, -8.3069e-02f, -9.5957e-02f, 

        -8.6631e-02f, -1.5156e-01f, -8.9147e-02f, -9.3641e-02f, -3.4494e-01f, -2.7221e-01f, -1.8527e-01f, -5.5369e-01f, 

        -1.3753e-01f, -1.3139e-01f, -1.7214e-01f, -1.9443e-01f, -2.2483e-01f, -1.5694e-01f, -2.0496e-01f, -1.6255e-01f, 

        -8.3168e-02f, -1.3217e-01f, -7.3173e-02f, -1.5872e-01f, -1.5504e-01f, -1.6977e-01f, -1.6950e-01f, -1.5086e-01f, 

        -1.0986e-01f, -6.8681e-02f, -8.4941e-02f, -2.0710e-01f, -9.7479e-02f, -6.8074e-02f, -7.4495e-02f, -1.6515e-01f, 

        -2.1728e-01f, -2.5857e-01f, -2.0451e-01f, -2.2559e-01f, -1.7773e-01f, -2.0616e-01f, -2.1187e-01f, -1.7365e-01f, 

        -2.4650e-01f, -2.1679e-01f, -3.9316e-01f, -1.4440e-01f, -1.5029e-01f, -1.3805e-01f, -1.3648e-01f, -1.0852e-01f, 

        -1.8044e-01f, -2.0918e-01f, -2.4474e-01f, -2.2785e-01f, -3.2120e-01f, -2.0706e-01f, -2.2032e-01f, -1.9908e-01f, 

        -1.9524e-01f, -2.0358e-01f, -2.5806e-01f, -2.3899e-01f, -1.0368e-01f, -2.1023e-01f, -1.6078e-01f, -2.0127e-01f, 

        -2.3902e-01f, -1.7384e-01f, -1.5930e-01f, -4.5096e-01f, -2.6993e-01f, -4.1767e-01f, -2.9074e-01f, -2.1796e-01f, 

        -1.7080e-01f, -2.4762e-01f, -2.4016e-01f, -3.1415e-01f, -1.0584e-01f, -8.9845e-02f, -1.7589e-01f, -1.6698e-01f, 

        -1.2324e-01f, -1.3925e-01f, -1.6106e-01f, -1.7167e-01f, -1.7759e-01f, -1.8191e-01f, -1.5024e-01f, -2.1994e-01f, 

        -2.0055e-01f, -1.4789e-01f, -1.3037e-01f, -1.3911e-01f, -1.8406e-01f, -1.6347e-01f, -1.6531e-01f, -9.0442e-02f, 

        -1.1405e-01f, -2.6747e-01f, -1.1212e-01f, -9.6548e-02f, -1.2705e-01f, -1.9062e-01f, -9.1199e-02f, -2.8937e-01f, 

        -1.8265e-01f, -1.1604e-01f, -1.4354e-01f, -3.0407e-01f, -1.7998e-01f, -1.9798e-01f, -1.7689e-01f, -1.8215e-01f, 

        -1.9929e-01f, -1.2245e-01f, -1.1612e-01f, -1.4286e-01f, -1.2147e-01f, -2.3307e-01f, -1.8831e-01f, -1.8229e-01f, 

        -2.2900e-01f, -1.6775e-01f, -1.7658e-01f, -1.9011e-01f, -1.1471e-01f, -1.5873e-01f, -1.2854e-01f, -1.4083e-01f, 

        -1.3230e-01f, -1.8706e-01f, -2.5594e-01f, -2.8778e-01f, -1.4471e-01f, -1.9728e-01f, -2.0646e-01f, -1.8043e-01f, 

        -2.0174e-01f, -2.2172e-01f, -1.2724e-01f, -1.4309e-01f, -1.2924e-01f, -1.5230e-01f, -1.0321e-01f, -1.2155e-01f, 

        -1.3112e-01f, -1.3651e-01f, -1.8937e-01f, -1.2015e-01f, -1.1933e-01f, -8.9854e-02f, -1.1424e-01f, -8.7608e-02f, 

        -1.4821e-01f, -1.2920e-01f, -1.6952e-01f, -8.6059e-02f, -1.3768e-01f, -2.1835e-01f, -1.3640e-01f, -1.8721e-01f, 

        -2.1068e-01f, -9.6795e-02f, -2.0103e-01f, -1.5078e-01f, -1.8675e-01f, -1.9523e-01f, -1.8202e-01f, -1.3096e-01f, 

        -1.8689e-01f, -1.7448e-01f, -1.7020e-01f, -1.8307e-01f, -1.3248e-01f, -1.1563e-01f, -1.6131e-01f, -1.3499e-01f, 

        -1.1882e-01f, -1.4970e-01f, -1.8238e-01f, -1.4966e-01f, -1.9519e-01f, -1.1370e-01f, -5.7529e-02f, -9.2377e-02f, 

        -1.1170e-01f, -8.1029e-02f, -7.4300e-02f, -9.5803e-02f, -1.2466e-01f, -1.6594e-01f, 3.9808e-02f, -1.4996e-01f, 

        -6.5630e-02f, -3.3227e-02f, -1.9164e-02f, -1.0633e-01f, -1.1480e-01f, -9.5797e-02f, -8.8140e-02f, -2.3266e-02f, 

        -2.0013e-01f, -1.2923e-01f, -6.3999e-02f, -1.4047e-01f, -1.7215e-01f, -1.4160e-01f, -5.1669e-02f, -1.6128e-01f, 

        -1.5986e-01f, 1.1104e-01f, -1.7969e-01f, -1.5716e-01f, -2.3478e-01f, -2.9304e-01f, -1.0713e-01f, -8.5244e-02f, 

        -1.6308e-01f, -1.7607e-03f, -1.1259e-01f, -1.0263e-01f, -1.0177e-02f, -1.4885e-01f, -1.6975e-01f, 4.5395e-03f, 

        -1.3536e-01f, -1.8333e-01f, -4.6696e-02f, -7.1091e-02f, -2.8519e-02f, -4.4416e-02f, -1.6861e-01f, -1.7112e-01f, 

        1.3261e-02f, 1.3716e-02f, -5.2384e-01f, -2.5389e-01f, -2.2143e-01f, -2.0306e-01f, -1.3432e-01f, -2.9924e-01f, 

        -1.2549e-01f, -1.1397e-01f, 3.7551e-02f, -1.4411e-01f, -1.8985e-01f, -1.1502e-01f, -1.3386e-01f, -1.4714e-01f, 

        -2.4183e-01f, 6.1471e-02f, -1.0093e-01f, -7.0491e-02f, -8.9410e-03f, -1.4838e-01f, -2.1464e-01f, -9.5086e-02f, 

        -1.6266e-01f, -8.9889e-02f, -8.1121e-02f, -1.4574e-01f, 3.5829e-02f, -1.2615e-01f, -1.0965e-02f, -1.2529e-02f, 

        -1.7735e-01f, -1.3806e-01f, -7.8694e-02f, -1.4381e-01f, -1.5416e-02f, -4.6656e-02f, -1.8311e-01f, -4.0268e-02f, 

        -5.0871e-02f, -5.5653e-02f, -2.8968e+00f, -2.7311e+00f, -5.6766e-01f, -1.1800e+00f, -1.3513e+00f, -7.0723e-01f, 

        -1.5196e+00f, -7.8413e-01f, -2.1371e+00f, 9.3705e-02f, -1.4612e-01f, -9.7754e-02f, 1.0140e-01f, -1.0010e-01f, 

        -1.6517e-01f, -9.2621e-02f, -1.5892e-01f, 2.5006e-02f, -1.9697e-01f, -1.6497e-01f, -2.8883e-01f, -2.9374e-01f, 

        -2.1465e-01f, -1.8666e-01f, -1.2999e-01f, -1.6388e-01f, -1.3449e-01f, 5.4908e-02f, -7.8187e-02f, 8.2621e-02f, 

        1.8767e-01f, -1.9121e-02f, -1.4098e-02f, 4.6039e-02f, -1.0007e-01f, 9.0065e-02f, -1.6574e-02f, -8.1994e-03f, 

        -2.3927e-01f, -6.0069e-02f, 3.0505e-04f, -9.1631e-02f, -3.2097e-02f, -1.4529e-01f, -1.9424e-01f, -3.1727e-02f, 

        -1.3169e-01f, -2.3845e-01f, -3.5666e-02f, 2.1725e-02f, -5.4257e-02f, 3.0648e-02f, -1.2822e-01f, -1.6693e-01f, 

        -8.3818e-01f, 5.5912e-03f, -1.4123e-01f, -1.4258e-01f, -1.2216e-01f, 1.2541e-01f, -1.2875e-01f, -1.3306e-01f, 

        -2.9610e-01f, -1.7474e-01f, -1.0364e-01f, -1.2325e-01f, -1.2552e-01f, -3.7049e-02f, -1.9214e-01f, -2.7055e-02f, 

        -1.0943e-01f, -1.6509e-01f, -6.9604e-01f, 3.0313e-02f, -4.2305e-03f, -7.3661e-02f, -1.6255e-01f, -6.0696e-02f, 

        1.0381e-02f, 1.0055e-03f, -1.0512e-01f, -5.2712e-01f, -5.9935e-01f, -2.6006e-01f, -2.6217e-01f, -2.6227e-01f, 

        -3.4192e-01f, -1.0822e-01f, -3.5796e-01f, -2.7947e-01f, 8.2167e-02f, -7.1729e-03f, 1.9945e-02f, -2.2340e-02f, 

        -7.6381e-02f, -1.7355e-01f, -1.1171e-01f, -1.2800e-01f, 7.2260e-02f, -8.6868e-02f, -1.0296e-01f, -9.0756e-02f, 

        -3.0399e-02f, -1.1798e-01f, -1.0324e-01f, -1.0756e-01f, -9.7870e-02f, -8.0137e-02f, -1.2147e-01f, -1.0355e-01f, 

        -3.4081e-02f, -4.7676e-01f, 8.3655e-03f, -2.5800e-01f, -1.8564e-01f, 2.2345e-03f, -3.8356e-02f, -1.3922e-01f, 

        -2.0318e-01f, -4.9429e-02f, -1.4954e-01f, -1.4448e-01f, -1.3696e-01f, -1.7464e-01f, -1.6507e-01f, 2.7334e-02f, 

        -2.0218e-01f, 1.2836e-02f, -9.6645e-02f, -4.4974e-02f, 1.3763e-02f, 5.4894e-02f, -2.1050e-01f, -1.9574e-01f, 

        -1.2956e-02f, -1.5255e-01f, -6.7559e-02f, -1.0547e-01f, -9.1608e-02f, -1.3876e-01f, -1.0760e-01f, -6.5941e-02f, 

        -8.9806e-02f, -8.9410e-02f, -4.9239e-02f, -4.7047e-01f, -4.6680e-01f, -1.9552e-01f, -1.4821e-01f, -2.6578e-01f, 

        1.9573e-01f, -5.0765e-02f, -4.8002e-01f, -1.7053e-01f, -5.5425e-02f, -1.5659e-01f, -1.3260e-01f, -1.5842e-01f, 

        -5.8402e-02f, -1.2687e-01f, -1.0847e-01f, -1.0998e-01f, 5.3680e-02f, -1.4196e-01f, -8.0365e-02f, -1.1478e-01f, 

        -1.4362e-01f, 1.1490e-02f, -1.3842e-02f, 1.0927e-01f, 1.0324e-02f, -2.4144e-01f, -5.9933e-01f, -1.1983e-01f, 

        -3.1379e-01f, -7.2377e-02f, 3.3714e-01f, -1.5154e-01f, -1.8019e-01f, -6.1426e-02f, -4.9220e-02f, -1.0877e-01f, 

        -2.5936e-02f, -9.4963e-02f, -1.5958e-01f, -1.3370e-01f, -1.5871e-01f, -7.0327e-02f, -5.4137e-02f, 9.1992e-02f, 

        -5.2605e-02f, -3.6346e-02f, -7.8393e-02f, -9.6944e-02f, -1.5553e-02f, -2.0736e-01f, -2.5927e-01f, -5.8486e-02f, 

        -1.6531e-01f, -6.7432e-02f, 1.4010e-01f, -1.6494e-01f, -4.9782e-02f, 1.7913e-01f, -3.8517e-01f, -3.6494e-01f, 

        -1.8068e-01f, -1.5844e-01f, 3.7517e-02f, -1.2837e-01f, -8.7048e-02f, -4.3078e-02f, -1.0488e-01f, -9.0747e-02f, 

        -1.0681e-01f, -1.4352e-01f, -6.4532e-01f, -4.0664e-02f, -9.8343e-02f, -3.4107e-01f, -1.5719e-01f, -2.7296e-01f, 

        -2.4659e-01f, -1.4264e-01f, 3.5637e-01f, -3.8010e-01f, -2.4143e-02f, -2.2458e-01f, -8.3897e-02f, -5.7762e-01f, 

        -3.0857e-01f, -9.3823e-01f, 1.0102e-01f, 1.8340e-01f, -8.9852e-02f, -2.3170e-01f, 1.7245e-03f, -1.3897e-01f, 

        -1.4449e-01f, -2.4534e-01f, -1.0139e-01f, 6.1659e-02f, 2.6779e-01f, -7.7448e-02f, -9.3504e-01f, -7.3630e-01f, 

        -1.2700e-01f, -1.7466e-01f, -1.3613e+00f, 2.8756e-01f, -8.2338e-01f, 1.1145e+00f, -6.2984e-02f, -2.0800e-01f, 

        -2.8116e-01f, -2.1065e-01f, -4.9531e-01f, 9.4836e-02f, -9.5424e-02f, -1.5450e-01f, 4.9988e-01f, -9.0166e-02f, 

        -6.3429e-02f, 6.4902e-01f, 2.2532e-01f, -3.9883e-01f, 9.5829e-02f, -3.5825e-01f, 5.4941e-01f, 2.9455e-01f, 

        2.4845e-02f, 8.2428e-02f, -1.2095e-01f, -3.3760e-01f, -2.3491e-01f, -1.4328e+00f, 7.2364e-01f, -3.0857e-01f, 

        6.7246e-01f, -7.6816e-02f, -1.8512e-01f, -2.4190e-01f, -1.3257e-01f, -5.5710e-01f, -1.1864e-01f, 1.8840e-01f, 

        -1.1480e-01f, 7.0145e-01f, -1.2255e+00f, -2.9965e-01f, -9.0644e-01f, 7.9591e-02f, -2.2783e-01f, -1.7545e-01f, 

        -5.0559e-01f, -7.4420e-02f, -1.9706e-01f, 1.3194e-01f, 2.2575e-01f, -1.8567e-01f, -1.4462e-01f, -5.0079e-02f, 

        -1.6710e-01f, -2.2121e-01f, -3.0987e-01f, 4.0871e-01f, -2.5534e-01f, -4.3537e-01f, -1.1445e+00f, -8.7584e-01f, 

        -4.3258e-01f, -3.7888e-01f, -1.0591e+00f, 1.9171e-01f, -1.2982e+00f, 2.6195e-01f, -7.1170e-01f, -1.8148e+00f, 

        4.1566e-01f, -1.0098e+00f, 5.2940e-01f, 1.1234e+00f, -4.7804e-01f, -4.6050e-02f, 9.2160e-02f, -1.5973e-01f, 

        2.2839e-01f, 6.4170e-02f, -8.1493e-01f, 3.9257e-02f, 2.1591e-01f, 1.7089e-01f, -2.9610e-01f, -1.1886e-01f, 

        -3.4757e-02f, -2.6807e-02f, -3.3276e-01f, 1.8629e-01f, -3.3291e-01f, -3.2624e-01f, -3.4478e-01f, 7.8715e-01f, 

        8.2615e-01f, 1.3184e+00f, 1.1166e+00f, 1.4447e+00f, 7.4079e-01f, 1.3646e+00f, 1.4038e+00f, 4.7659e-01f, 

        -5.7526e-01f, -3.7417e-02f, 3.2144e-01f, 7.0795e-02f, -1.6686e-01f, 1.2985e-02f, -3.8809e-01f, 1.0176e+00f, 

        3.8609e-01f, 2.0683e-01f, 9.8281e-03f, -3.1096e-01f, -3.3047e-01f, -3.3245e-01f, 8.0383e-02f, -2.5795e-01f, 

        -2.5059e-01f, 2.8503e-01f, 1.5507e-01f, -4.8447e-01f, -3.2822e-01f, 6.7811e-01f, -1.1893e-01f, 2.7057e-01f, 

        2.1379e+00f, 3.0259e-01f, 9.1144e-01f, 2.1190e+00f, -6.4383e-02f, 1.6734e+00f, -5.1801e-01f, -7.5278e-01f, 

        -1.0066e-01f, -2.2136e-01f, -4.9680e-01f, -4.3837e-01f, 3.6816e-02f, -1.0027e-01f, -3.7357e-02f, -4.1574e-02f, 

        -1.9984e-02f, -1.7472e-01f, -2.3072e-02f, -4.9565e-02f, 8.7613e-03f, 3.8451e-02f, 8.0724e-02f, 1.7181e-01f, 

        -1.0081e+00f, -8.3149e-01f, -8.8950e-01f, -7.5310e-01f, -2.0462e+00f, -7.0375e-01f, 6.1955e-01f, -6.2826e-01f, 

        -1.1866e+00f, -3.5047e-02f, -2.0456e+00f, -1.2056e+00f, 9.1406e-02f, 1.6063e-01f, -6.7639e-01f, 1.2964e+00f, 

        2.8193e-01f, 3.8322e-01f, -3.0361e-01f, -4.2256e-01f, 1.2510e-01f, 2.6754e-01f, -9.2085e-02f, -3.6474e-01f, 

        -5.0578e-01f, -1.6698e-02f, 1.7259e-01f, -1.3216e-01f, -9.0167e-02f, -1.4760e-01f, -2.1335e-01f, -1.4164e-01f, 

        -1.7097e-01f, -3.1456e-01f, 5.5034e-01f, 1.1067e+00f, 7.1920e-01f, 5.2540e-01f, 5.5079e-01f, 1.4803e+00f, 

        -3.9259e-01f, 5.6115e-01f, -5.2376e-01f, 5.8227e-01f, -1.1323e-01f, -1.2732e-01f, -2.2927e-01f, -2.1954e-01f, 

        7.3452e-01f, 3.8526e-01f, 1.6529e-01f, 1.0494e+00f, -1.5287e-01f, -1.0706e-01f, -2.7286e-01f, -1.7263e-01f, 

        -1.7177e-01f, -2.5710e-01f, -2.2237e-01f, -2.2477e-01f, -5.5630e-02f, 5.8038e-01f, 2.0731e-01f, 2.6392e+00f, 

        1.6189e+00f, 3.8166e+00f, 3.8853e+00f, 1.5337e+00f, 4.0212e-01f, 2.7431e+00f, -1.6166e-01f, -3.3994e-01f, 

        -3.3531e-01f, -5.4934e-01f, -2.9168e-01f, -1.3010e+00f, -1.1088e-02f, 3.3049e-01f, 5.6849e-01f, -2.7999e-01f, 

        -1.0725e-01f, -1.8486e-02f, -3.2474e-01f, -3.1168e-01f, -2.3078e-01f, -6.3109e-02f, -3.0360e-01f, -2.3081e-01f, 

        1.4165e-01f, -2.2045e-02f, -5.7359e-03f, 1.9366e-02f, 4.3046e-02f, 5.8395e-03f, 1.1958e-01f, 4.4929e-02f, 

        5.9074e-02f, 1.6320e-01f, 4.6456e-02f, -1.4749e-02f, 1.1190e-01f, 8.0904e-02f, -1.7916e-02f, 7.8405e-02f, 

        6.0927e-02f, 1.1672e-01f, -1.7577e-01f, -2.9950e-02f, -5.9894e-02f, -1.0386e-01f, -2.5695e-02f, -2.5790e-01f, 

        -9.0269e-02f, -1.5957e-01f, -2.4688e-01f, -1.9328e-01f, 3.1452e-01f, 2.3789e-01f, 3.4387e-02f, -1.3870e-01f, 

        -1.1860e-01f, -2.1087e-01f, -1.3263e-01f, -3.0930e-02f, 3.8156e-03f, 1.3935e-01f, -4.5685e-03f, 2.2710e-02f, 

        3.9033e-04f, -1.1939e-01f, 7.1651e-02f, 3.7474e-02f, -7.9241e-02f, -2.9767e-02f, -1.1641e-01f, -3.6730e-02f, 

        1.9667e-02f, -1.0796e-01f, -1.2501e-01f, -1.7568e-01f, -1.7598e-01f, -3.0780e-01f, -1.2854e-01f, 5.7252e-02f, 

        -1.1847e-01f, 3.7353e-01f, -1.0931e-01f, -1.4370e-02f, -1.6120e-02f, -6.2106e-03f, -1.0037e-01f, 1.4202e-02f, 

        1.0715e-01f, 5.9572e-02f, 8.7342e-02f, -2.2775e-01f, 4.9145e-02f, -8.3740e-02f, -1.7242e-02f, 7.5298e-02f, 

        -1.6548e-01f, -3.9679e-04f, -1.3573e-01f, -9.1966e-02f, 6.8710e-03f, -1.4875e-02f, -1.7909e-01f, 5.5739e-02f, 

        -4.5594e-02f, 4.7670e-02f, 4.6485e-02f, -3.0559e-02f, 2.1036e-01f, 2.7150e-02f, -1.0794e-01f, -3.4685e-02f, 

        -9.0164e-03f, 3.5052e-03f, 7.6100e-02f, 5.6790e-03f, -1.3878e-01f, -2.4159e-01f, -2.0803e-01f, -1.9169e-01f, 

        -1.3380e-01f, 2.2568e-02f, -1.4413e-01f, -2.2601e-02f, 7.3650e-02f, 8.5236e-02f, 1.1128e-02f, 2.8801e-02f, 

        4.1298e-02f, 5.5617e-02f, -3.5461e-02f, 4.2074e-02f, -3.2268e-01f, 2.4812e-02f, -3.3055e-01f, 3.5715e-02f, 

        -3.0842e-01f, -2.3735e-01f, -2.5467e-01f, -1.6974e-01f, -2.8407e-01f, 1.1916e+00f, -3.5606e-01f, 4.7088e-02f, 

        -6.5007e-03f, -1.9081e-03f, 2.4384e-02f, -1.3101e-02f, 3.7028e-02f, -3.9462e-02f, 1.1431e-01f, 4.5234e-02f, 

        9.0595e-02f, 4.5597e-02f, 1.3656e-01f, 1.6660e-01f, 9.4369e-02f, -2.5502e-02f, 4.6301e-02f, 8.1868e-02f, 

        7.0155e-02f, 1.3102e-01f, 4.8697e-02f, -4.4848e-03f, 9.3982e-02f, 1.4722e-01f, -2.6755e-02f, 2.5960e-02f, 

        -8.5345e-02f, -5.7398e-01f, -1.4958e-01f, -1.0698e-01f, -1.2319e-01f, -2.6000e-02f, -1.8133e-01f, -5.2049e-02f, 

        -1.3054e-01f, 1.2688e-01f, 1.6762e-02f, -1.2177e-01f, 7.7776e-02f, 9.5274e-02f, 1.4176e-01f, 1.6910e-02f, 

        7.8713e-02f, 1.2039e-01f, -8.4310e-02f, -8.5937e-02f, -2.2302e-01f, -1.7532e-01f, -3.0168e-01f, -2.6526e-01f, 

        -1.7879e-01f, -2.2120e-01f, -1.6806e-01f, 4.0043e-01f, -6.4223e-03f, 1.0298e-01f, -8.7442e-02f, -1.1290e-01f, 

        -1.1686e-02f, -6.1199e-02f, -1.7662e-01f, -7.8554e-02f, -4.8221e-01f, -1.0792e-01f, -1.3290e-01f, -5.4604e-02f, 

        -1.4655e-01f, -7.2468e-02f, 3.8396e-04f, -8.7150e-02f, -2.4597e-01f, 1.1102e-01f, -2.0861e-02f, -2.5402e-02f, 

        6.8121e-02f, 4.6212e-02f, -3.6683e-02f, -4.3074e-03f, -1.0168e-01f, -7.5574e-03f, -6.3257e-02f, 5.9284e-02f, 

        -6.6372e-02f, 6.4558e-02f, -1.6869e-02f, -4.3559e-02f, -1.4351e-01f, 1.3459e-04f, -2.0469e-01f, 5.1650e-01f, 

        -3.2358e-01f, -5.4571e-02f, -1.4870e-01f, 9.6049e-02f, -8.9341e-02f, -4.9675e-02f, 3.9530e-02f, -1.4384e-01f, 

        -3.2128e-01f, -1.1679e-01f, -3.1466e-01f, -2.9263e-01f, -4.6284e-01f, -1.7413e-01f, -3.5756e-01f, -3.8064e-01f, 

        -3.7205e-01f, 8.3003e-02f, 6.3260e-02f, 4.9956e-02f, 4.9118e-02f, 2.1003e-02f, 7.1468e-03f, 1.2953e-01f, 

        2.1822e-02f, 1.5909e-02f, 3.3851e-01f, -5.1375e-02f, -3.5785e-02f, 1.7535e-02f, -2.1421e-01f, -1.7259e-01f, 

        7.5330e-02f, -6.7385e-02f, 5.0532e-02f, 3.1332e-01f, 1.7219e-01f, 1.7972e-01f, 1.9199e-02f, 7.7036e-02f, 

        -4.3251e-04f, 1.2334e-01f, 5.4683e-02f, -1.0346e-02f, 1.4999e-02f, -1.1571e-02f, -6.4020e-02f, -3.8318e-02f, 

        2.3352e-02f, 1.1041e-02f, 3.7411e-02f, 2.4472e-02f, -3.3085e-02f, 2.2001e-02f, 4.8890e-01f, -6.1695e-02f, 

        2.7594e-02f, -1.9989e-01f, 1.2098e-02f, -2.5966e-01f, -1.8077e-01f, -6.4701e-02f, 5.6455e-03f, -6.2482e-02f, 

        -1.6627e-02f, -1.0771e-01f, -1.2277e-01f, 1.4901e-02f, 4.6052e-02f, -1.8124e-01f, -1.0821e-01f, -9.1975e-02f, 

        -1.0229e-01f, -1.7457e-01f, -2.4578e-01f, -1.2949e-01f, -2.5821e-01f, -1.4215e-01f, -2.4073e-01f, -1.2647e-01f, 

        -8.6228e-02f, -7.5277e-02f, -8.5857e-02f, -1.6648e-01f, -1.2825e-01f, -1.2171e-01f, -5.7713e-02f, -7.2717e-02f, 

        -8.1462e-02f, -5.7178e-02f, 6.3535e-02f, -1.2081e-01f, -1.2677e-02f, -1.1057e-01f, -6.6147e-02f, -5.7226e-03f, 

        -7.0221e-02f, -7.7928e-02f, -2.3306e-01f, -3.0241e-01f, -2.2870e-01f, -1.2812e-01f, -2.1059e-01f, -1.1365e-01f, 

        -1.6608e-01f, -1.6411e-01f, -1.8414e-01f, -5.9841e-01f, -3.2805e-01f, -2.9650e-01f, -3.4436e-01f, -4.4688e-01f, 

        -6.3934e-02f, -1.8145e-01f, -2.3603e-01f, -2.0966e-01f, -1.3992e-01f, -5.3312e-02f, -5.9358e-02f, -5.2575e-02f, 

        -1.5191e-01f, -1.5758e-01f, -6.3607e-02f, -1.2942e-01f, -7.8774e-02f, -4.7155e-01f, -6.2803e-01f, -3.2948e-01f, 

        -5.3827e-01f, -6.4695e-01f, -4.3791e-01f, -6.8012e-01f, -4.0734e-01f, -3.2712e-01f, -6.0812e-02f, -4.3339e-01f, 

        -1.4043e-01f, -1.5859e-01f, -1.2464e-01f, -1.0367e-01f, -4.9887e-02f, -1.1243e-01f, -1.8145e-01f, -4.5783e-01f, 

        -1.6888e-01f, -4.0910e-01f, -1.7960e-01f, -2.3837e-01f, -3.0597e-01f, -1.3612e-01f, -3.6901e-01f, 1.8178e-01f, 

        -2.0169e-01f, -1.4150e-02f, -2.0116e-01f, -1.1379e-01f, -1.1131e-01f, -1.6953e-01f, -1.5183e-01f, -4.5501e-02f, 

        -6.5428e-02f, 1.2257e-01f, -1.3217e-01f, -1.9219e-01f, -1.0104e-01f, -2.3298e-02f, 6.0428e-02f, 2.0683e-02f, 

        -5.8166e-02f, -7.6384e-02f, -3.3951e+00f, -4.0663e+00f, -3.9745e+00f, -2.4830e+00f, -2.9294e+00f, -3.0693e+00f, 

        -3.1830e+00f, -2.0775e+00f, -7.7133e-01f, 7.7573e-02f, -6.9619e-02f, -1.8376e-01f, -7.5806e-02f, -8.1464e-02f, 

        -1.4805e-01f, 8.3746e-03f, -2.3908e-01f, 7.7178e-02f, -1.6131e-01f, -3.5557e-01f, -2.3571e-01f, -9.2585e-01f, 

        1.4670e-02f, -2.3557e-01f, -3.1899e-01f, -1.0253e-01f, -3.0634e-01f, 3.9212e-02f, -5.3666e-01f, -6.0500e-02f, 

        4.9998e-02f, 1.1331e-02f, -6.5205e-02f, 5.1893e-02f, -1.0739e-01f, 6.0210e-02f, -1.7681e-01f, 1.2323e-02f, 

        -1.9875e-01f, -1.3095e-01f, -2.1488e-01f, -3.4246e-02f, -6.1415e-02f, -1.4730e-01f, -1.6324e-01f, -1.4068e-01f, 

        -9.0965e-02f, -1.8693e-01f, -1.5714e-01f, -1.8388e-01f, -2.1063e-01f, -1.5948e-01f, -1.2833e-01f, -1.0855e-01f, 

        -1.5132e+00f, -5.7665e-01f, -4.6830e-01f, -3.0734e-01f, -4.9777e-01f, -4.9621e-01f, -9.4346e-02f, -5.7417e-02f, 

        -3.6645e-01f, -9.8753e-02f, -8.5611e-02f, -2.2138e-01f, -1.5205e-01f, -1.8916e-01f, -1.6998e-01f, -1.1278e-01f, 

        -1.3349e-01f, -1.9391e-01f, -6.6162e-01f, -1.3805e-01f, -4.0040e-01f, -4.7543e-01f, -1.5643e-02f, -8.3946e-02f, 

        -2.8704e-01f, -2.8189e-01f, 1.5887e-01f, -1.1892e+00f, -4.9890e-01f, -5.4302e-01f, -5.4947e-01f, -5.0185e-01f, 

        -6.4814e-01f, -3.1786e-01f, -4.7224e-01f, -1.4346e-01f, -1.8005e-01f, -2.0986e-01f, -3.0736e-01f, -1.3917e-01f, 

        -1.6131e-01f, -1.6108e-01f, -2.8470e-01f, -2.9150e-01f, -1.6186e-02f, -1.7144e-01f, -1.6995e-01f, -1.2198e-01f, 

        -1.8302e-01f, -9.3432e-02f, -1.1056e-01f, -1.6714e-01f, -8.2792e-02f, -8.5795e-02f, -1.8073e-01f, -2.4065e-01f, 

        -1.2090e-01f, -3.4937e-01f, -8.2883e-02f, -2.4380e-01f, -1.5636e-01f, -1.2909e-01f, 2.8556e-02f, -5.2247e-01f, 

        -5.0804e-01f, -2.2722e-01f, -4.2069e-01f, -5.2454e-01f, -1.1887e-01f, -4.4336e-01f, -2.8663e-01f, -2.2544e-01f, 

        -3.2551e-01f, -2.4369e-01f, -2.7712e-01f, 1.1359e-01f, -9.8584e-02f, -8.6270e-02f, -3.8358e-01f, -7.5011e-02f, 

        6.3324e-02f, -7.4130e-03f, -9.7775e-02f, -1.5012e-01f, -6.5924e-02f, -1.5990e-01f, -9.4994e-02f, -1.4671e-01f, 

        -1.1412e-01f, -7.4782e-02f, 1.1911e-02f, -9.7454e-01f, -9.4519e-01f, -4.9685e-01f, -5.8619e-03f, -2.2508e-01f, 

        -7.1230e-01f, -4.0118e-01f, -7.6383e-01f, -2.0993e-01f, -1.5580e-01f, -1.3065e-01f, -1.5040e-01f, -3.5987e-02f, 

        -1.2645e-01f, -1.2966e-01f, -3.1571e-02f, -9.2546e-02f, -1.6762e-01f, -9.1080e-02f, -1.6112e-01f, 9.6131e-02f, 

        -1.7448e-01f, -8.9022e-02f, -9.4643e-02f, -1.9316e-01f, 3.5190e-02f, -5.0095e-01f, -5.3082e-01f, -6.5791e-01f, 

        -2.9146e-01f, -3.6982e-01f, -5.8202e-01f, -2.2107e-01f, -3.5777e-01f, 3.0903e-01f, 9.4731e-02f, -1.4662e-01f, 

        -1.0310e-01f, -1.2831e-01f, -9.6532e-02f, -1.1630e-01f, -1.4375e-01f, -1.1780e-01f, -1.3433e-01f, 1.2766e-01f, 

        9.2822e-02f, 1.3692e-01f, 2.8543e-01f, 1.2529e-01f, 9.1438e-02f, 1.0709e-01f, -7.3245e-02f, -6.2006e-02f, 

        -1.3063e-02f, 8.6092e-02f, 3.6142e-01f, -8.4346e-02f, 1.1406e-01f, 3.3184e-01f, 3.6196e-01f, 3.6694e-01f, 

        -5.8694e-02f, 1.7239e-02f, 7.1653e-02f, -6.2043e-03f, -3.3369e-02f, 7.5939e-03f, -2.6951e-01f, -6.0928e-02f, 

        -9.9744e-02f, -1.0293e-01f, -6.9817e-02f, 1.2053e+00f, 5.3193e-01f, -3.8305e-01f, 3.8637e-04f, 2.3164e+00f, 

        -4.0788e-01f, -4.6542e-01f, 2.8425e-01f, 4.9420e-01f, 9.3666e-02f, -1.5608e-01f, 6.0764e-02f, -4.0437e-02f, 

        9.5925e-01f, -1.1786e+00f, 9.2086e-01f, 8.1794e-01f, 7.3745e-01f, -5.2601e-01f, 2.7990e-01f, -4.4182e-01f, 

        4.2978e-01f, -3.5598e-01f, -2.1861e-01f, 3.9878e-01f, -1.7874e-02f, -1.8100e+00f, 5.5899e-01f, -5.5760e-01f, 

        -2.3979e+00f, -7.3381e-01f, -9.1661e-02f, -2.6001e+00f, -2.0771e+00f, -2.0912e-01f, 1.1185e+00f, 1.0493e+00f, 

        1.1153e+00f, 8.8942e-01f, 4.7855e-01f, 5.7247e-02f, -4.3190e-01f, 9.6411e-01f, 1.4595e+00f, -2.9309e-01f, 

        5.4919e-01f, 7.9280e-01f, -2.8419e-01f, -2.0480e-01f, -4.4365e-01f, 1.1365e+00f, 6.2562e-01f, -7.6635e-01f, 

        2.7721e-02f, 6.2332e-02f, -2.1315e-02f, -1.2982e-01f, 1.7644e-01f, -5.1699e-01f, -5.4720e-01f, -2.0624e-01f, 

        -5.9508e-01f, 5.7371e-01f, -3.2932e-01f, 9.1217e-01f, -2.9452e-01f, 3.8093e-01f, -5.5140e-01f, -7.3439e-01f, 

        9.7081e-01f, 3.3972e-01f, -1.3056e-01f, -1.5804e-01f, -5.7191e-01f, -7.2508e-01f, 2.0936e-02f, -9.2919e-01f, 

        4.7326e-01f, -3.5335e-01f, -1.6335e+00f, -1.7569e-01f, 8.3334e-02f, -2.4670e-01f, -1.1275e-01f, 5.1243e-02f, 

        -1.2531e-01f, -1.5990e-01f, -6.4383e-02f, -2.4905e-01f, 7.6229e-02f, -2.1121e-01f, -7.1577e-03f, 1.3230e+00f, 

        -6.7465e-01f, -9.8045e-01f, 1.3867e+00f, 7.5382e-01f, -9.7937e-01f, -2.9337e-01f, 1.3570e-01f, -2.6544e-01f, 

        1.1874e+00f, -4.4435e-01f, -8.8642e-01f, 7.5273e-02f, 2.2532e-01f, -1.2394e-01f, 2.9961e-01f, 1.0426e-01f, 

        -1.9317e-01f, -4.0617e-01f, 6.1891e-01f, -3.7535e-01f, -2.2234e-01f, -4.2261e-02f, -4.1129e-01f, 2.4796e-01f, 

        -8.4284e-01f, 1.0437e-01f, -2.5675e-01f, 9.4468e-02f, -6.9735e-02f, -7.0776e-01f, -4.8150e-01f, -1.7887e-01f, 

        1.6808e-01f, -1.3702e-01f, 6.3998e-01f, -5.6465e-01f, -1.9203e+00f, 6.1608e-01f, -4.6025e-01f, 5.4788e-01f, 

        2.8360e-01f, 1.4609e-01f, 3.4167e-01f, -1.7815e-01f, 2.4203e-02f, 5.6134e-01f, 1.4733e-01f, 1.1482e+00f, 

        1.0273e+00f, 4.6561e-01f, 2.0852e-01f, 8.5315e-02f, -5.4187e-01f, -3.5716e-01f, -1.7268e-01f, 4.5308e-01f, 

        -1.0624e+00f, 5.0914e-02f, -6.4711e-01f, 8.9111e-01f, 3.8359e-01f, -6.3530e-02f, 6.0384e-01f, 1.0065e+00f, 

        8.4495e-02f, 1.4966e+00f, 1.0396e+00f, -3.6314e-01f, -4.2520e-01f, -5.0126e-01f, -3.2885e-01f, -2.4870e-01f, 

        -3.8965e-01f, -1.2745e-01f, -7.6420e-01f, 2.5883e-01f, -4.8816e-01f, 8.2459e-02f, 4.7280e-03f, 1.6629e-01f, 

        -1.8311e-01f, 3.0540e-02f, -2.6760e-01f, 1.5247e-01f, 2.6981e-01f, -8.5381e-02f, -1.1769e+00f, -9.1490e-01f, 

        -8.2194e-01f, -5.1868e-01f, -1.1434e+00f, -2.4297e-01f, -5.4537e-01f, 2.0847e+00f, -2.7721e-01f, -1.5732e+00f, 

        -1.1753e+00f, -2.1580e+00f, -9.1042e-01f, -8.7267e-01f, 5.7304e-01f, 1.2595e+00f, -6.5840e-01f, 1.1650e+00f, 

        1.1932e-01f, -5.0374e-01f, -1.4732e-01f, -3.2214e-01f, 6.6384e-01f, -3.2959e-01f, 8.9967e-02f, 7.0196e-01f, 

        -4.4083e-01f, -1.4323e-01f, -1.6966e-02f, -1.0159e-01f, -5.4413e-02f, -2.8707e-01f, -1.4297e-01f, -1.3553e-01f, 

        1.1541e-01f, -1.4634e-01f, -1.1988e+00f, -1.5104e+00f, 5.7717e-02f, -2.3126e+00f, -1.0036e+00f, 2.3409e-02f, 

        -2.0798e+00f, -7.1666e-01f, -1.2400e-01f, 9.2313e-02f, -2.3592e-01f, 7.2702e-01f, 7.0534e-01f, 7.3636e-01f, 

        -5.9770e-01f, 1.1466e+00f, 1.9327e+00f, 5.0563e-02f, -4.4856e-02f, -2.1224e-01f, -1.6169e-01f, -2.3176e-01f, 

        -1.1900e-01f, -1.8463e-02f, -1.4553e-02f, -2.7414e-01f, -2.4894e-01f, 9.9621e-01f, -1.3818e+00f, -3.3161e-01f, 

        1.7086e+00f, -3.2560e-01f, 7.9320e-02f, 1.9063e+00f, -8.2157e-01f, 6.8559e-01f, 6.8363e-01f, 2.2907e-01f, 

        -1.0677e+00f, -2.1042e+00f, -1.0217e+00f, -1.3049e+00f, 1.1839e+00f, -2.9100e+00f, -1.6147e+00f, -2.5413e-01f, 

        -6.6423e-02f, -1.7720e-01f, -1.7691e-01f, -2.6682e-01f, -2.9160e-01f, -4.2946e-01f, -4.4281e-01f, -7.5665e-04f, 

        -7.4393e-02f, -7.8109e-02f, -8.6554e-02f, -8.7575e-02f, -5.8600e-02f, -6.7303e-02f, -1.0068e-01f, -9.3048e-02f, 

        3.9400e-02f, -9.8796e-02f, -1.0147e-01f, -8.7749e-02f, -1.0367e-01f, -9.7648e-02f, -3.4610e-02f, 3.3164e-02f, 

        -8.5743e-02f, 3.1096e-02f, -7.5298e-02f, -3.5053e-02f, -3.9805e-02f, 4.1892e-02f, -1.2500e-01f, -1.1890e-02f, 

        -3.9965e-02f, -1.0862e-01f, -1.2359e-01f, -9.3670e-02f, -2.2819e-01f, 1.2949e-02f, -1.3363e-01f, -1.5684e-01f, 

        1.3632e-02f, -1.2582e-01f, -1.1834e-01f, -1.5645e-01f, -7.4437e-02f, -1.3659e-01f, -9.0255e-02f, -1.0772e-01f, 

        -1.0305e-01f, -1.1929e-01f, -1.3851e-01f, -6.6407e-02f, -1.3194e-01f, -2.8728e-01f, -1.7737e-01f, -2.0781e-01f, 

        -1.1885e-01f, -2.9449e-01f, -1.8817e-01f, -2.2542e-01f, -7.0333e-02f, 3.1430e-02f, -1.5487e-01f, -1.7781e-01f, 

        -1.7001e-01f, -8.6358e-02f, -1.5360e-01f, -2.2083e-01f, -6.4472e-02f, -1.0324e-01f, -4.2484e-02f, -1.0969e-01f, 

        -1.4919e-01f, -1.6551e-01f, -3.8922e-02f, 1.4053e-01f, -1.3283e-01f, -6.4990e-02f, -5.3431e-02f, 1.2196e-01f, 

        -9.2019e-02f, -1.4526e-01f, -1.3486e-01f, -7.5732e-02f, -1.2345e-01f, -8.0579e-02f, -1.4586e-01f, -1.4359e-01f, 

        -9.2810e-02f, 1.4670e-03f, -9.2786e-02f, -3.2901e-02f, 6.1245e-02f, 1.4913e-01f, -3.4913e-02f, 1.2963e-01f, 

        -1.0433e-01f, -1.1144e-01f, -4.4305e-01f, -8.9361e-01f, -1.7418e+00f, -1.9471e+00f, -1.9663e+00f, -1.3640e+00f, 

        -5.0148e-01f, -1.0439e+00f, -3.3242e+00f, -1.1049e-01f, -1.9526e-01f, 2.7406e-02f, -1.8555e-01f, 1.1846e-01f, 

        -1.9182e-01f, 1.3253e-01f, -3.3340e-02f, 8.2557e-02f, -4.0562e-01f, -2.1396e-02f, -8.4088e-02f, -8.5121e-02f, 

        1.8873e-01f, -1.9278e-01f, -5.2873e-02f, -4.4363e-02f, -1.6301e-01f, 2.7730e-02f, -3.8678e-02f, 1.2263e-01f, 

        -7.8560e-02f, 1.9774e-01f, -8.9537e-02f, 1.7280e-01f, 2.3449e-02f, 1.4565e-01f, -1.2654e-01f, -1.5863e-01f, 

        -1.3765e-01f, -2.1028e-01f, -1.1053e-01f, 3.7809e-02f, -1.4326e-01f, -2.1568e-02f, 2.9807e-02f, -1.5182e-01f, 

        -8.4576e-02f, -3.7818e-02f, -1.4687e-01f, -2.9271e-02f, -9.0211e-02f, 2.1399e-03f, -1.5259e-05f, 2.7248e-02f, 

        -3.0433e-01f, -2.4109e-01f, 7.4308e-02f, -1.8701e-02f, -1.0304e-01f, 4.7020e-02f, 2.7392e-02f, -9.1795e-02f, 

        -1.0568e-01f, -8.2018e-02f, -6.6396e-02f, -5.3728e-02f, -8.0990e-02f, -7.2629e-02f, -1.1685e-01f, -1.5340e-01f, 

        -9.9982e-02f, -9.7024e-02f, -2.0766e-01f, -1.7807e-01f, -7.6852e-02f, -1.8847e-01f, -5.0222e-02f, -1.3775e-01f, 

        -2.5154e-02f, -1.7250e-01f, 6.2506e-02f, -3.3105e-01f, -1.9282e-01f, -1.7195e-01f, -2.2690e-01f, -1.7720e-01f, 

        -3.5131e-02f, -5.3619e-02f, -7.2404e-02f, -1.0693e-01f, -1.5360e-01f, -2.8535e-02f, 3.0933e-02f, 6.1723e-04f, 

        7.8610e-02f, -1.3919e-02f, 5.2683e-02f, 1.0998e-01f, -3.0571e-02f, -1.5328e-01f, -7.2348e-02f, -1.0441e-01f, 

        -1.3526e-01f, -6.8354e-02f, -5.1567e-02f, -4.9891e-02f, 2.2603e-03f, -4.1421e-02f, 6.0319e-02f, -1.0521e-01f, 

        -7.6546e-02f, -1.9932e-01f, -1.0849e-01f, -1.2490e-01f, -6.5127e-02f, 1.9810e-02f, -1.9855e-01f, -2.1618e-01f, 

        -7.7985e-02f, -1.8840e-01f, 3.7030e-02f, -1.6444e-01f, -2.2584e-01f, 6.8783e-03f, -1.1966e-01f, -7.9587e-02f, 

        -1.6817e-01f, -2.2077e-02f, -9.0305e-02f, -1.3707e-01f, -8.9927e-02f, -1.6294e-01f, -1.5641e-01f, -4.4232e-03f, 

        5.6657e-03f, -1.1666e-01f, -1.3340e-01f, -1.0501e-01f, -1.7598e-01f, 2.9897e-03f, -1.0907e-01f, -5.3201e-02f, 

        -1.4488e-01f, -5.3850e-02f, -3.9914e-02f, -4.8223e-02f, -1.8927e-01f, 1.1724e-02f, -8.6814e-02f, 1.0474e-01f, 

        1.2067e-01f, -2.0100e-02f, 4.9671e-02f, -1.7560e-01f, -7.6521e-02f, -1.1293e-01f, -6.9409e-02f, -1.4282e-01f, 

        -1.7918e-01f, -1.4300e-02f, -1.5303e-01f, -9.0567e-02f, -4.1433e-02f, -1.2930e-01f, -1.1107e-01f, 1.8363e-02f, 

        8.3111e-03f, -1.0280e-01f, -1.1954e-01f, 6.3985e-02f, -2.0399e-02f, -9.2101e-02f, -1.8480e-01f, -1.2225e-01f, 

        -7.1490e-02f, 9.2311e-02f, -3.5753e-01f, -6.2210e-03f, -1.3007e-01f, 3.2295e-03f, -1.2432e-01f, -1.3228e-01f, 

        -5.8686e-02f, -1.4192e-01f, -1.7174e-01f, -8.1801e-02f, -9.4011e-02f, -1.7779e-01f, -4.8122e-02f, -1.0406e-01f, 

        -1.7309e-01f, -3.4827e-02f, 2.3785e-01f, -1.2922e-01f, 2.4450e-01f, -3.9231e-02f, -2.3453e-02f, -1.3640e-01f, 

        -1.8282e-01f, -6.2966e-01f, 2.6937e-01f, 1.5037e-01f, 7.0274e-03f, 5.1932e-01f, -1.1545e+00f, -6.6665e-01f, 

        -2.7232e-01f, 6.8555e-02f, -1.5567e-01f, 7.1378e-02f, -7.6124e-02f, 8.0332e-03f, -4.3803e-02f, -5.7348e-02f, 

        -1.4844e-01f, -2.3203e-01f, -1.7525e+00f, -2.3423e+00f, 8.6375e-02f, -5.1459e-01f, -5.2322e-01f, -8.5825e-02f, 

        -7.2091e-01f, 9.5402e-01f, -8.2488e-01f, 1.2007e-01f, -8.5355e-02f, -7.8112e-01f, -5.2700e-01f, -9.2709e-01f, 

        -5.7097e-01f, 3.8976e-02f, -1.0584e+00f, -1.7660e+00f, -1.4768e-01f, -5.5031e-01f, -3.0227e-02f, -5.9218e-02f, 

        4.2376e-01f, 1.6441e-01f, -1.9982e-01f, 1.6805e-02f, -2.1646e-01f, 2.2564e-01f, 7.0303e-01f, 4.3230e-02f, 

        7.5958e-01f, 6.8732e-01f, -3.5808e-01f, 1.3186e-01f, -7.6775e-01f, -1.0310e+00f, -1.3311e-01f, 2.6565e-01f, 

        8.0285e-01f, -5.3311e-01f, 7.9672e-01f, 2.9839e-01f, 1.0549e-01f, -1.3228e-01f, -9.1899e-03f, -1.2611e+00f, 

        -3.5317e-02f, 2.1403e-02f, 6.4475e-01f, -1.5530e-01f, 6.9214e-01f, -1.3768e+00f, 1.1134e+00f, -2.6993e-01f, 

        -9.4374e-01f, -7.5335e-01f, 3.9482e-01f, -1.1692e+00f, 1.5396e+00f, 3.5056e-01f, 7.6215e-01f, 2.0230e-01f, 

        -3.7985e-01f, 6.0938e-02f, -3.8013e-02f, 3.9773e-01f, -3.9138e-01f, 8.4465e-01f, 4.5822e-01f, 4.6293e-01f, 

        4.3301e-01f, 2.8597e-01f, 1.8514e-01f, -5.8103e-01f, -3.9359e-01f, 5.4871e-01f, -2.8027e-01f, 3.7606e-02f, 

        2.0074e-01f, -3.8652e-02f, -3.2738e-01f, -3.7295e-01f, -1.6999e-02f, -8.5083e-02f, -2.7790e-02f, -3.2966e-01f, 

        -2.5958e-03f, -4.5238e-02f, -2.4483e-01f, -9.8794e-02f, 2.7510e-01f, -2.3618e-01f, -9.1671e-02f, 4.9141e-01f, 

        -5.3700e-02f, -8.9233e-01f, 1.7673e-01f, -3.6359e-01f, -7.6582e-01f, -3.5237e-01f, -2.5191e-01f, 1.5445e+00f, 

        -4.2993e-01f, -5.1706e-01f, 1.8323e+00f, 1.1017e+00f, 1.5550e+00f, -1.6822e+00f, -1.7455e+00f, -1.5901e+00f, 

        -1.2972e+00f, -1.7661e+00f, -1.4082e+00f, 4.0885e-01f, 5.2884e-01f, -1.1786e-01f, -8.7934e-01f, -1.5815e+00f, 

        -2.8519e+00f, -6.9127e-01f, -1.5339e+00f, -1.3596e+00f, -1.0199e+00f, -7.1068e-01f, -1.4300e+00f, 8.7081e-02f, 

        2.6765e-01f, -2.7313e-01f, 1.0853e-01f, 8.5885e-01f, 7.9885e-01f, -5.6935e-01f, 8.4715e-01f, 1.3656e+00f, 

        9.7156e-01f, -5.0966e-01f, -4.4161e-02f, 1.1015e-01f, 1.6501e-01f, -4.8370e-01f, 3.5320e-01f, 8.0723e-01f, 

        -2.7994e-01f, -3.8057e-01f, -4.7757e-01f, -4.0773e-01f, -1.0116e+00f, -1.4343e+00f, -8.8951e-01f, -7.8781e-01f, 

        -8.5800e-01f, -1.1680e-01f, -9.4007e-01f, 3.1818e-01f, 5.2305e-01f, 1.4176e+00f, 3.2822e-01f, 2.5676e+00f, 

        1.9010e+00f, 2.2031e-03f, 3.0702e-01f, 7.4190e-01f, -6.3232e-01f, -1.2123e+00f, -8.9957e-02f, 1.4565e-01f, 

        -3.9467e-01f, -9.5436e-01f, 1.0811e+00f, 9.8386e-02f, -2.6833e+00f, 2.7239e-01f, -3.3660e-01f, 4.6451e-01f, 

        -5.0642e-01f, 4.9910e-01f, 1.6653e-01f, 1.7132e-01f, 1.7964e-01f, 3.4162e-01f, -6.6562e-01f, -4.1194e-01f, 

        -8.3397e-01f, 4.0929e-01f, 6.2743e-01f, -7.7201e-01f, 9.2822e-01f, 1.1740e+00f, -1.9642e-01f, 4.1556e-01f, 

        -8.3309e-01f, 1.2519e-01f, -1.5226e+00f, -1.5135e+00f, -3.1882e-01f, -6.9109e-01f, -8.9655e-01f, -1.2141e+00f, 

        4.1416e-02f, 5.9200e-02f, -3.4878e-01f, 7.5912e-01f, 3.2668e-01f, 2.8495e-01f, 1.9975e+00f, 9.9845e-01f, 

        1.4964e-01f, 3.5991e-02f, -1.1197e-01f, -9.6206e-03f, -1.4543e-02f, -8.3368e-02f, -2.0328e-01f, -2.1138e-01f, 

        -2.8008e-01f, -2.7942e-01f, -1.3347e+00f, -1.5940e+00f, -1.8988e+00f, -1.0247e+00f, -4.9877e-02f, -9.0052e-01f, 

        -8.6186e-01f, -4.7492e-01f, 1.1359e-03f, -6.4875e-01f, -2.9138e-01f, 2.2568e-01f, 2.9995e-01f, 2.2244e+00f, 

        5.3006e-01f, -1.0141e+00f, -3.2873e-01f, 9.3865e-01f, -1.7811e-01f, -7.1937e-02f, -1.3621e-01f, -1.4201e-01f, 

        3.3731e-02f, -1.8087e-01f, -1.2322e-01f, -2.7290e-01f, -1.7975e-01f, 8.9639e-01f, 1.1780e+00f, 7.6074e-01f, 

        1.2329e+00f, 1.5110e+00f, 4.7072e-01f, 4.5360e-01f, 1.7690e+00f, 2.8195e-01f, -1.5210e+00f, -1.9297e+00f, 

        -1.4070e-01f, -1.5025e+00f, 1.5779e-01f, -1.2742e+00f, -2.4507e-01f, -2.0173e+00f, -1.8103e+00f, -9.5179e-02f, 

        -1.0947e-01f, -8.1365e-02f, -2.0245e-01f, -1.7899e-01f, -3.4443e-01f, -1.0925e-01f, -1.4932e-01f, -2.5185e-01f
    };

    // Layer 3 feedforward weights - fully connected (576x10)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    static const float fc3_weights_vector[5760] = {
        4.5425e-01f, 9.3056e-02f, -3.7459e-01f, 4.6578e-01f, 5.9046e-01f, 2.0843e-01f, -1.6401e-01f, 3.5414e-01f, 

        1.6419e-01f, 1.5341e-01f, -2.3703e-01f, 2.8417e-01f, 1.3002e-01f, -3.4527e-01f, 1.8416e-01f, -2.2636e-01f, 

        3.3166e-01f, 5.3607e-02f, -1.2190e-01f, 5.3867e-01f, 2.1173e-01f, 3.9113e-02f, -1.1914e-01f, -7.4390e-01f, 

        3.7791e-01f, -6.3792e-01f, -3.9880e-01f, 9.3949e-01f, -6.4022e-01f, 3.7564e-01f, 5.0063e-01f, -1.0785e-01f, 

        2.7297e-01f, -1.8374e-01f, 1.5537e-01f, -3.1292e-01f, -4.2245e-01f, 5.8573e-01f, 5.3992e-01f, 2.9843e-01f, 

        -4.1526e-01f, 2.3415e-01f, 2.3983e-01f, 2.8453e-01f, 4.6121e-01f, -1.8470e-01f, 3.0208e-01f, 9.0101e-01f, 

        5.0641e-01f, 5.8624e-01f, 1.2844e-01f, 6.4961e-02f, 6.9339e-03f, -5.3090e-01f, 2.2783e-01f, 2.0152e-01f, 

        1.7372e-01f, 5.7677e-01f, -3.2527e-01f, 1.0815e+00f, -4.6385e-02f, 1.1353e-01f, 3.4489e-01f, 1.3935e-01f, 

        2.1289e-01f, -3.2262e-01f, 3.0993e-02f, 2.4832e-01f, -6.8725e-03f, 6.2355e-01f, -3.4336e-01f, 2.9600e-01f, 

        -3.5908e-02f, 3.4498e-02f, 3.1053e-01f, 5.8456e-02f, 2.2132e-01f, 9.5759e-01f, -4.4434e-01f, 2.1690e-01f, 

        8.6184e-02f, -1.8592e-01f, -6.2786e-01f, -3.2050e-01f, -1.4486e+00f, 4.1710e-01f, 2.2387e+00f, 2.5334e-01f, 

        2.6556e-01f, -7.4349e-01f, -3.9868e-01f, 1.7669e-02f, 1.7155e+00f, 4.3511e-01f, -8.0232e-01f, 2.9331e-01f, 

        -7.4780e-01f, 1.0541e+00f, 1.7615e+00f, 1.0773e+00f, 2.8149e-01f, -1.1304e+00f, -1.5905e-01f, 1.1935e+00f, 

        -1.0957e+00f, 9.2391e-01f, -1.9585e+00f, 5.2239e-01f, 1.3114e+00f, -1.3807e+00f, 7.9207e-01f, 7.4082e-02f, 

        5.7253e-01f, -2.5511e-01f, 2.6888e+00f, 3.7075e-01f, -6.9246e-01f, 4.3360e-01f, -1.4260e+00f, -6.4101e-01f, 

        1.0452e+00f, -1.5216e+00f, 7.1710e-02f, 1.9592e+00f, 2.1216e+00f, -7.5264e-01f, -4.0402e-01f, -1.4927e+00f, 

        -1.8947e+00f, 7.9618e-02f, -4.9470e-03f, -3.6604e-01f, -9.5463e-01f, 8.4406e-01f, 1.9201e+00f, 1.0590e+00f, 

        5.0332e-01f, -1.0292e+00f, -7.2806e-01f, -3.7551e-01f, 1.5865e+00f, -1.4361e+00f, -1.9040e+00f, 1.9223e+00f, 

        -7.4452e-01f, 2.3760e+00f, 2.3582e+00f, 1.9722e+00f, -2.1584e+00f, -5.6874e-01f, 2.2458e-01f, -1.5903e+00f, 

        -7.2404e-02f, 7.7239e-01f, 8.6838e-01f, 2.7937e-01f, 8.3152e-01f, -1.8876e+00f, -1.5020e+00f, 1.4488e+00f, 

        -8.4478e-02f, 2.6642e-01f, -4.1826e-01f, 3.6290e-02f, 2.9977e-01f, -1.0991e+00f, -5.6428e-01f, 3.2967e-01f, 

        6.6557e-01f, -1.2440e+00f, -3.8320e-01f, 1.2932e+00f, 2.1795e+00f, 1.6409e+00f, -1.2196e+00f, 7.0952e-01f, 

        -4.4856e-01f, -1.2985e+00f, -6.7764e-01f, -3.5464e-01f, -7.1587e-01f, 1.9988e-01f, 4.2888e-01f, 7.2357e-01f, 

        -5.2702e-02f, 4.6009e-01f, -6.4212e-01f, 9.7906e-01f, 9.2110e-01f, -1.2991e+00f, 5.1107e-01f, -1.2457e+00f, 

        -6.0281e-01f, 4.2467e-01f, 4.5791e-01f, 4.0323e-01f, -1.5887e-01f, -1.3226e+00f, -3.8086e-01f, 1.8054e-01f, 

        -3.2505e-01f, 7.1874e-01f, -2.4096e-01f, -8.1328e-02f, -1.5160e-01f, 2.3484e+00f, -1.3098e-01f, -1.1957e+00f, 

        4.5091e-02f, -1.7626e-01f, 7.6505e-01f, 8.0759e-01f, 1.1596e+00f, 2.2753e-01f, 9.1384e-02f, -1.1521e-01f, 

        -1.4022e-01f, 1.1154e-01f, -9.9061e-03f, -6.3088e-01f, 4.5322e-01f, -1.0771e-01f, 2.1547e-01f, -4.6887e-01f, 

        6.0704e-02f, 1.4290e+00f, -5.3931e-01f, 6.4558e-01f, -5.6201e-02f, 6.7706e-01f, 3.0355e-01f, 9.2237e-01f, 

        -2.0129e-01f, -8.4614e-02f, -1.3430e-01f, 4.6832e-01f, 1.4215e-01f, -3.9640e-01f, 1.3367e-01f, -7.9637e-01f, 

        -2.5634e-01f, -5.9498e-02f, 3.6489e-01f, -9.1061e-03f, 6.9396e-01f, -3.3215e-01f, 6.7194e-01f, 2.1558e-01f, 

        2.2210e-01f, -3.4997e-02f, -5.5001e-01f, 5.4178e-02f, -5.6852e-01f, -8.8849e-01f, -8.3038e-01f, 4.5165e-01f, 

        3.9637e-01f, -9.1858e-02f, 4.2849e-01f, 1.0383e+00f, 3.4965e-02f, 3.9135e-01f, -1.5436e-02f, -2.4806e-02f, 

        -4.8071e-01f, -6.7048e-01f, -1.0636e+00f, -6.5630e-01f, -6.3785e-02f, -1.0695e-01f, 7.4711e-01f, -3.0291e-01f, 

        8.0014e-01f, -3.5074e-01f, 9.0414e-01f, -5.9395e-02f, -6.4940e-02f, 7.9276e-01f, 7.0803e-01f, -1.5946e-01f, 

        6.2240e-01f, -5.6190e-01f, 6.0357e-01f, -4.2604e-01f, -6.2320e-01f, -7.2877e-01f, -6.0400e-01f, -3.7465e-01f, 

        5.8829e-01f, 1.4487e+00f, -2.9169e-01f, 5.4890e-01f, 3.5598e-01f, -4.2816e-01f, 2.9668e-01f, 4.5443e-01f, 

        2.0573e+00f, 1.2721e+00f, 1.7982e+00f, -1.9613e-01f, 6.1907e-01f, 4.5031e-01f, 4.6231e-01f, 6.5585e-01f, 

        8.1273e-01f, -1.1160e-01f, 1.8156e+00f, -6.2177e-01f, 5.4878e-01f, 8.2523e-01f, -2.3338e-01f, -5.8174e-01f, 

        1.0529e+00f, 1.1372e+00f, 9.9139e-01f, -4.6489e-01f, 7.2794e-01f, 1.4867e+00f, 5.4641e-01f, 1.2890e-01f, 

        -1.6342e+00f, 7.2259e-01f, 3.1630e-01f, 1.1626e-02f, -1.6790e+00f, 1.1723e+00f, 2.5159e+00f, -4.2842e-01f, 

        -7.2858e-01f, -3.2284e-01f, 2.5011e-01f, 6.7948e-01f, -2.8933e-01f, 4.0729e-01f, 9.6309e-01f, -4.6854e-01f, 

        3.5585e-01f, -3.5158e-02f, -1.2885e-01f, -2.2251e-01f, 1.4897e+00f, 4.8931e-01f, 1.1288e+00f, -4.6030e-01f, 

        -8.9276e-01f, 1.6896e-01f, -2.9869e-01f, -9.5411e-01f, -1.4186e+00f, -5.1383e-01f, -2.1966e-01f, 2.2279e-01f, 

        -1.0696e+00f, 1.7701e+00f, -1.7263e+00f, 5.7434e-01f, -6.7879e-01f, -1.5534e+00f, -5.1096e-01f, 4.3429e-01f, 

        -7.2438e-02f, -9.2983e-02f, 1.0883e-01f, -8.6487e-02f, -8.0075e-02f, 1.9762e-01f, 4.6403e-04f, -6.4187e-02f, 

        -1.3340e-01f, -6.8772e-02f, -6.1129e-02f, -6.9125e-02f, -4.1185e-02f, -1.0390e-01f, -6.7548e-03f, -5.2274e-02f, 

        -9.0587e-02f, -6.8176e-02f, -9.2924e-02f, -5.0556e-02f, -8.0623e-02f, -8.0336e-02f, 1.4774e-02f, -7.4422e-02f, 

        -4.6741e-02f, 2.1854e-01f, 1.0047e-02f, -5.2284e-02f, -1.1580e-01f, -6.7079e-02f, -1.5524e-01f, -1.6716e-01f, 

        5.9272e-02f, -1.1185e-01f, -5.3281e-02f, 1.9335e-01f, -1.2905e-01f, -9.5871e-02f, -5.0956e-02f, -6.8865e-02f, 

        -1.1201e-01f, -7.2524e-02f, -3.8538e-02f, -7.3244e-02f, -5.5705e-02f, 2.4861e-03f, -3.6378e-02f, -6.6098e-02f, 

        -1.1538e-01f, -6.2400e-03f, -1.1291e-01f, -8.5215e-02f, 4.3047e-02f, -1.4002e-01f, -9.6857e-02f, -5.0898e-03f, 

        -2.2006e-02f, -4.8551e-02f, -1.4279e-01f, -3.8792e-02f, -5.3600e-02f, -5.1057e-02f, -1.9411e-02f, -6.3787e-02f, 

        -3.7816e-02f, 1.7271e-01f, -5.9426e-02f, -7.0044e-02f, 1.2885e-01f, 1.2722e-02f, -1.9327e-01f, -1.0091e-01f, 

        -1.1221e-01f, -7.5563e-02f, -3.8183e-02f, -2.5963e-02f, -5.9899e-02f, -5.9047e-04f, -5.9732e-02f, -1.1520e-01f, 

        -9.4878e-02f, -1.1319e-01f, -6.7916e-02f, -6.5765e-02f, -8.6707e-02f, -3.2465e-02f, -5.0706e-02f, -2.9761e-02f, 

        -4.4457e-02f, -6.6480e-02f, -8.2092e-02f, -9.4316e-02f, -6.8729e-02f, -6.5494e-02f, -9.4392e-02f, -1.2081e-01f, 

        -7.5146e-02f, -9.4637e-02f, 3.6002e-02f, -1.5258e-01f, -1.2439e-01f, -8.8044e-02f, -1.0838e-01f, 5.1052e-02f, 

        -7.8641e-02f, -1.3716e-01f, 4.3970e-02f, -1.3434e-01f, 3.5415e-02f, -1.2371e-01f, -1.0380e-01f, 1.1980e-01f, 

        -7.8582e-02f, -1.8526e-02f, -1.1079e-01f, -1.6312e-01f, 6.1490e-02f, -1.2196e-01f, -2.1537e-02f, -1.3684e-01f, 

        -8.1854e-02f, -1.2100e-01f, -9.3217e-02f, -2.4374e-02f, -9.6920e-02f, -9.2733e-02f, -1.0488e-02f, -1.1056e-01f, 

        5.8962e-02f, -1.0991e-01f, -9.0031e-02f, -1.4695e-01f, -8.5546e-02f, -6.3298e-02f, -1.0587e-01f, -1.6175e-01f, 

        3.1204e-02f, -1.1222e-01f, -2.1281e-02f, -1.5374e-01f, -7.4259e-02f, -8.4350e-02f, -1.3977e-01f, -1.9521e-02f, 

        -1.1694e-01f, -1.6166e-01f, 2.4505e-02f, -1.1415e-01f, 1.6191e-02f, -9.0349e-02f, -1.0054e-01f, -4.3725e-02f, 

        -1.3503e-01f, -7.4415e-02f, -1.5696e-01f, -1.2652e-01f, 1.9827e-02f, -9.1796e-02f, 6.8164e-02f, -8.3782e-02f, 

        -7.4781e-02f, -5.1316e-02f, -1.3169e-01f, -3.6093e-02f, -1.3481e-01f, -9.6463e-02f, 5.6587e-02f, -1.4167e-01f, 

        4.4853e-02f, -5.6221e-02f, -1.1707e-01f, -5.0800e-02f, -1.5917e-01f, -2.6171e-02f, -8.5232e-02f, -1.3793e-01f, 

        -7.8295e-02f, -1.1397e-01f, -4.2365e-02f, -7.3751e-02f, -6.4898e-02f, -5.9169e-02f, -1.2232e-01f, -5.4417e-03f, 

        -3.8113e-02f, -1.0076e-01f, 4.4334e-02f, -1.5288e-01f, 3.6681e-03f, -1.5491e-01f, -1.2287e-01f, 5.5393e-03f, 

        -1.0274e-01f, 3.3575e-03f, -6.3868e-03f, -1.0316e-01f, 1.0155e-01f, -8.8255e-02f, 2.7632e-02f, -1.2076e-01f, 

        -1.6283e-01f, 8.7861e-02f, -1.0009e-01f, 2.7622e-02f, -6.7674e-02f, -9.9255e-02f, 6.6440e-02f, -1.4819e-01f, 

        8.7923e-03f, -1.3630e-01f, -1.2222e-01f, -1.0068e-01f, -1.2232e-01f, 8.0477e-03f, -1.3730e-01f, -1.3709e-01f, 

        7.4591e-02f, -1.0763e-01f, 4.3847e-02f, -1.3379e-01f, -6.2044e-02f, -6.0564e-02f, -9.3661e-02f, 7.1579e-03f, 

        -8.3111e-02f, -9.7068e-02f, 2.8813e-02f, -1.1092e-01f, -2.6687e-03f, -7.6844e-02f, -8.2314e-02f, 7.2495e-02f, 

        -1.0610e-01f, -6.8569e-02f, -1.6009e-01f, -1.1875e-01f, 7.7633e-02f, -1.1557e-01f, 1.2202e-03f, -1.1618e-01f, 

        -1.2579e-01f, -1.3858e-01f, -1.1613e-01f, -1.1175e-01f, -1.3427e-01f, -8.7884e-02f, 6.1609e-02f, -9.7766e-02f, 

        3.8244e-02f, -8.9365e-02f, -8.7892e-02f, -1.5476e-01f, -1.1003e-01f, -5.6723e-02f, -1.6648e-01f, -1.4266e-01f, 

        4.0299e-02f, -1.0906e-01f, 2.0604e-02f, -1.1021e-01f, -1.1256e-01f, -9.5540e-02f, -1.4205e-01f, -8.1502e-02f, 

        -8.6734e-02f, -8.1768e-02f, -1.6622e-02f, -1.2135e-01f, 1.0022e-03f, -8.3825e-02f, -6.5996e-02f, -1.0623e-01f, 

        -6.8131e-02f, -2.3778e-02f, -7.5749e-02f, -1.2949e-01f, 5.6654e-02f, -5.9814e-02f, 5.5559e-02f, -7.1947e-02f, 

        -6.4899e-02f, -8.5903e-02f, -1.4445e-01f, 4.1766e-02f, 4.7034e-02f, -1.3925e-01f, 1.2149e-01f, -1.0731e-01f, 

        3.5343e-02f, -8.3075e-02f, -9.8534e-02f, 9.5069e-02f, -1.0207e-01f, 5.3997e-02f, -3.0798e-02f, -6.7135e-02f, 

        8.4704e-02f, -9.3318e-02f, 6.6358e-03f, -1.4983e-01f, -7.1150e-02f, -4.6056e-02f, -7.9111e-02f, 4.0841e-03f, 

        -8.0005e-02f, -9.5311e-02f, 5.5197e-02f, -1.0315e-01f, 8.6518e-02f, -7.6950e-02f, -6.1273e-02f, -1.1551e-01f, 

        -8.0067e-02f, 4.3244e-02f, -1.5722e-01f, -9.1936e-02f, 2.7751e-02f, -7.3676e-02f, 7.9689e-02f, -1.0199e-01f, 

        -6.0674e-02f, -1.0495e-02f, -8.8367e-02f, -6.7314e-04f, -6.6910e-02f, -1.2943e-01f, 7.5238e-02f, -9.9515e-02f, 

        7.6704e-02f, -1.0121e-01f, -1.2805e-01f, -7.7260e-02f, -1.1793e-01f, -2.3002e-02f, -1.0459e-01f, -8.2717e-02f, 

        4.4248e-02f, -1.1650e-01f, 9.1476e-02f, -1.3453e-01f, -8.6632e-02f, -8.0543e-02f, -7.7427e-02f, -3.5591e-02f, 

        -1.0098e-01f, -1.2461e-01f, -4.6993e-03f, -8.5175e-02f, 3.7286e-02f, -1.0678e-01f, -1.0496e-01f, -1.3539e-01f, 

        -1.6714e-01f, -6.4226e-02f, -1.4644e-01f, -9.1915e-02f, 2.1943e-02f, -1.4189e-01f, 4.6006e-02f, -1.0249e-01f, 

        1.6932e-01f, -8.1569e-01f, 3.0502e-01f, -5.6472e-02f, -9.2453e-01f, 4.0070e-01f, -8.0164e-01f, 7.7540e-01f, 

        -3.9865e-01f, -3.2737e-01f, 9.0877e-02f, -1.6602e-01f, 1.3969e-01f, -1.0534e-01f, -1.4902e-01f, -2.4745e-01f, 

        -2.8042e-02f, 1.1119e+00f, -6.2009e-02f, -1.0306e+00f, -3.9554e-01f, -6.3837e-02f, 3.9406e-02f, -6.2954e-02f, 

        8.6903e-01f, 7.5072e-02f, -1.7734e-01f, -1.5686e+00f, -8.1184e-01f, -4.1250e-03f, 6.5599e-01f, -4.4780e-01f, 

        7.9665e-02f, 9.8143e-02f, -3.5882e-01f, -1.7507e-01f, -4.1162e-01f, -3.0333e-01f, -1.2911e-01f, 3.4701e-02f, 

        5.7382e-02f, -2.6971e-01f, -4.6616e-01f, -3.4359e-02f, 6.3926e-01f, 6.2809e-01f, 1.9945e-01f, -9.1919e-01f, 

        -2.1348e-01f, 2.9376e-01f, 6.3061e-01f, -1.3031e-01f, 2.3444e-01f, -1.7487e-01f, 1.0688e+00f, 3.0100e-02f, 

        -1.0860e-01f, -1.6194e+00f, -6.4726e-01f, -4.3322e-01f, 2.3662e-01f, -2.9890e-01f, 1.7671e-01f, 1.5468e-01f, 

        -1.1412e-01f, -3.0910e-01f, -2.2001e-01f, 2.6064e-01f, -6.5514e-01f, 2.5526e-01f, -5.0178e-01f, -2.8980e-01f, 

        1.7128e-01f, 8.9872e-02f, -1.6098e+00f, -2.1473e-01f, -9.4239e-02f, -1.7832e-01f, 7.3319e-01f, 6.2853e-01f, 

        1.6732e-01f, 3.2324e-02f, 3.3215e-01f, -1.9612e-01f, -4.4874e-01f, 4.4490e-02f, 1.5392e-02f, 1.0319e-01f, 

        -2.0486e-01f, -3.0574e-01f, -2.8064e-02f, -5.1928e-01f, -4.9448e-01f, 7.8476e-02f, -3.6186e-01f, -1.4803e-01f, 

        -5.6242e-01f, 1.7319e+00f, -2.5586e-02f, -2.3646e-01f, -5.0815e-01f, -1.9476e-01f, -7.7057e-02f, 4.1565e-02f, 

        -4.1586e-01f, 3.7799e-03f, -8.6596e-01f, 1.7041e-01f, 1.3064e+00f, 8.8810e-02f, -6.5466e-01f, -1.4136e-01f, 

        -3.8386e-01f, 4.4310e-01f, -1.0710e-01f, 5.4864e-02f, -8.6555e-01f, 2.9148e-01f, 4.2161e-01f, 3.6773e-01f, 

        -1.4589e-01f, -4.7107e-01f, 3.6952e-01f, 8.2295e-02f, -4.8646e-01f, -2.6483e-01f, 1.9651e-01f, 1.1537e-01f, 

        3.8913e-01f, 3.7789e-01f, 2.2456e-01f, -4.3479e-01f, -3.4736e-01f, 2.1501e-01f, -3.5614e-01f, 4.5349e-01f, 

        6.3526e-03f, 3.8904e-01f, -3.6881e-01f, 1.9196e-01f, -1.4330e-01f, -3.2882e-01f, 5.8256e-02f, 7.2694e-01f, 

        7.0091e-02f, 2.8725e-02f, -2.4306e+00f, -2.2228e+00f, 1.1074e+00f, 8.8352e-02f, -7.7192e-02f, -1.8384e-01f, 

        -3.6861e-01f, -4.6364e-01f, -4.4881e-01f, 1.0846e+00f, 4.9477e-01f, 2.3237e-01f, -1.6841e-02f, 3.5070e-01f, 

        -6.9521e-01f, -2.9697e-01f, -8.1979e-02f, 2.4080e-01f, -1.0401e-01f, 4.9328e-01f, 5.6884e-01f, 2.7930e-01f, 

        -8.2658e-01f, 1.8638e-01f, -4.6183e-01f, -3.2867e-01f, 7.0748e-02f, 3.7559e-01f, 1.3184e+00f, 1.1117e-01f, 

        -2.8470e-01f, -7.6219e-01f, -6.3680e-01f, -4.1578e-01f, -4.7698e-02f, -6.6762e-02f, -5.9029e-02f, 1.3025e-02f, 

        5.0461e-03f, -1.3550e-01f, 4.2508e-02f, -1.2876e-01f, 3.0173e-02f, -1.3760e-01f, -9.0544e-02f, -2.6826e-02f, 

        -1.3681e-01f, 4.2433e-02f, -3.2797e-02f, -8.9492e-02f, 1.2402e-01f, -1.2931e-01f, 8.5559e-02f, -5.2843e-02f, 

        -1.0921e-01f, 7.2283e-02f, -1.3344e-01f, 2.8320e-02f, 7.3941e-02f, -1.2293e-01f, 6.2974e-02f, -1.1770e-01f, 

        1.0865e-02f, -8.9145e-02f, 3.0764e-02f, 3.5597e-02f, -6.1488e-02f, 7.2117e-02f, -9.6816e-02f, -8.9589e-02f, 

        8.9141e-02f, -2.3377e-02f, 1.2380e-01f, -3.8517e-02f, -6.7620e-02f, -4.1272e-02f, -8.7248e-02f, 8.1103e-02f, 

        -7.4794e-02f, -7.2048e-02f, 5.8705e-02f, -1.0207e-01f, 1.0691e-01f, -7.0233e-03f, -6.2619e-02f, 7.5837e-02f, 

        -8.1446e-02f, 4.7201e-02f, -1.3940e-01f, -1.0697e-01f, 3.9282e-02f, -1.0212e-01f, 5.0378e-03f, -9.2777e-02f, 

        -7.5689e-02f, -7.3639e-02f, -1.1303e-01f, -6.7015e-02f, -8.4792e-02f, -9.4751e-02f, 2.1966e-02f, -4.5712e-02f, 

        9.3021e-02f, -1.1952e-01f, -4.9065e-02f, -4.2641e-02f, -7.0651e-02f, -4.1679e-02f, -7.7729e-02f, -1.1240e-01f, 

        3.8931e-02f, -7.1839e-02f, 3.3580e-02f, -6.2108e-02f, -4.9015e-02f, -5.2666e-03f, -1.3339e-01f, 6.3335e-02f, 

        -7.9582e-02f, -6.6748e-02f, 2.8301e-03f, -4.2684e-02f, 5.4061e-03f, -1.3037e-01f, -3.1180e-01f, 1.1099e-01f, 

        1.4475e-01f, 1.0888e-01f, 4.8906e-01f, -3.8708e-01f, 5.5728e-01f, 2.7672e-01f, 4.8961e-01f, -1.0084e-01f, 

        1.2756e-01f, 9.6257e-02f, 5.5902e-01f, 2.0095e-02f, 3.1953e-01f, 1.1733e-01f, -8.3113e-02f, -3.2985e-02f, 

        2.8059e-01f, 2.5700e-01f, 4.2522e-01f, 1.7040e-01f, 4.9707e-01f, 7.8049e-01f, 2.2738e-01f, 1.6860e-02f, 

        2.8595e-01f, 1.5356e-01f, -2.9734e-01f, -1.0906e-01f, -8.0361e-01f, 1.3684e-01f, 2.0041e-01f, 4.1656e-01f, 

        -6.6454e-02f, -3.2819e-01f, 1.5743e-01f, 4.8865e-02f, 9.3158e-02f, 6.7110e-01f, -3.8380e-01f, 5.6721e-02f, 

        4.3239e-01f, -4.8040e-01f, 2.5349e-01f, -1.4014e-02f, 1.9695e-01f, 2.9858e-01f, 3.6177e-01f, 4.9232e-02f, 

        5.1873e-01f, 2.2340e-01f, 1.1211e-01f, 4.3191e-01f, 1.7650e-01f, 4.6505e-01f, 5.5425e-02f, -3.5910e-01f, 

        -1.9601e-01f, -2.8574e-01f, -5.4448e-01f, 1.1004e-01f, 1.2757e-01f, 4.1342e-03f, -1.1916e-01f, 9.3319e-01f, 

        7.3204e-01f, 1.0788e-02f, -3.7015e-01f, -3.1521e-01f, -2.3105e-01f, 7.5839e-02f, 2.1813e-01f, 5.2022e-01f, 

        -6.6834e-02f, 2.6883e-01f, 4.1225e-01f, -3.0502e-02f, -6.2432e-01f, 5.6046e-02f, -6.0550e-01f, -1.9263e-01f, 

        4.0723e-01f, -4.4586e-01f, -6.3249e-02f, -5.5270e-01f, -1.9871e-01f, 2.9668e-01f, 5.2104e-01f, 1.5840e-01f, 

        -8.1448e-03f, -3.8830e-02f, -3.6152e-02f, 5.3720e-03f, 6.7109e-02f, -6.8199e-02f, 6.1945e-02f, -3.1489e-02f, 

        6.0768e-02f, -5.2526e-02f, -4.1449e-02f, 8.3268e-03f, -1.1874e-01f, 4.3774e-02f, 4.0459e-02f, -1.1419e-01f, 

        1.2997e-01f, -1.2207e-01f, 6.2947e-02f, -9.3204e-02f, -9.5331e-02f, 6.7686e-03f, -1.0966e-01f, -7.6715e-02f, 

        -6.6623e-02f, -1.2449e-01f, 3.8783e-02f, -1.4226e-01f, 2.5875e-02f, -1.1262e-01f, -5.1416e-02f, -1.3346e-01f, 

        -5.5597e-02f, -8.3617e-03f, -1.1003e-01f, -1.1326e-01f, 9.4602e-03f, -8.1128e-02f, 9.3840e-02f, -9.1999e-02f, 

        9.4860e-03f, -3.5369e-03f, -1.0202e-01f, -9.6156e-03f, -6.5111e-02f, -6.0441e-02f, 9.8979e-02f, -1.1718e-01f, 

        6.8883e-02f, -9.0518e-02f, -4.0574e-02f, -6.5915e-03f, -1.0143e-01f, 1.7409e-02f, -9.7306e-02f, -1.0828e-01f, 

        7.1141e-02f, -1.1551e-01f, 4.9082e-02f, -1.1479e-01f, -6.3778e-02f, -8.7814e-02f, -1.2763e-01f, -2.7675e-02f, 

        -9.8878e-02f, -1.1038e-01f, 4.4973e-02f, -1.1956e-01f, 8.3964e-02f, -1.4427e-02f, -3.0309e-05f, -1.2173e-01f, 

        -6.2466e-02f, -2.7975e-02f, -3.4658e-02f, -2.6256e-02f, 2.8050e-02f, -1.0813e-01f, 1.0610e-01f, -2.3773e-02f, 

        -6.1634e-02f, -7.2222e-02f, -1.3933e-01f, -1.0501e-02f, -8.6860e-02f, -7.2009e-02f, -1.0187e-02f, -7.8557e-02f, 

        2.0912e-02f, -4.1991e-02f, -1.1043e-01f, 7.0256e-01f, -2.7527e-01f, -1.1565e+00f, -1.0726e+00f, 1.1685e-01f, 

        7.2742e-01f, -1.4518e+00f, 9.4321e-01f, 3.8790e-01f, 4.9736e-02f, 1.6435e-01f, -6.3322e-01f, 7.7870e-01f, 

        2.1783e-01f, -8.8645e-02f, -3.0263e-01f, -2.6174e+00f, -4.5480e-01f, 1.3551e-01f, -7.9055e-01f, 7.7820e-02f, 

        -6.6908e-01f, 7.8369e-01f, -5.0522e-02f, -3.2210e-01f, 4.0192e-01f, 5.5045e-01f, 1.6801e-01f, -5.7177e-02f, 

        -1.3496e-02f, -7.2798e-02f, -1.3864e-01f, 1.0241e-02f, -3.6853e-01f, 8.7875e-03f, 2.4566e-01f, -3.3230e-01f, 

        1.6020e-01f, 7.9949e-02f, 1.4172e-01f, -5.1624e-01f, -1.9972e-01f, -2.2640e-01f, 1.9409e-01f, 2.2638e-01f, 

        2.3201e-01f, -1.6223e+00f, -3.2032e-01f, 2.4090e-01f, 3.2519e-01f, -3.8953e-01f, -3.3457e-01f, 4.5178e-01f, 

        5.8401e-01f, -4.6142e-01f, -4.1765e-02f, -4.2357e-01f, -1.7857e-01f, -2.3519e-03f, -3.5257e-01f, 2.6856e-01f, 

        -1.0417e-01f, -3.3254e-01f, -6.5860e-01f, 2.4368e-01f, -1.5747e-02f, 7.7731e-01f, 2.2286e-01f, -4.7680e-01f, 

        -4.9326e-01f, -3.0131e-01f, 5.8902e-02f, -7.5353e-01f, 2.0003e-01f, 2.7174e-01f, -8.9744e-01f, 7.7755e-01f, 

        -7.5897e-01f, -4.6495e-02f, -5.4257e-01f, -9.1146e-01f, -8.1975e-01f, -8.5296e-01f, 2.7572e+00f, -4.3822e-01f, 

        -9.9693e-01f, 1.6049e-01f, -3.8952e-01f, -7.0970e-01f, 3.5740e-01f, -4.2437e-01f, 6.7408e-02f, -2.4169e-01f, 

        1.7733e-01f, -4.4331e-01f, 4.2356e-01f, -3.0913e-01f, 8.2883e-02f, -1.9307e-03f, 2.4726e-02f, -1.4651e-01f, 

        1.3816e-01f, -1.1614e+00f, -3.3710e-01f, -4.7957e-01f, 1.4960e-01f, -2.1026e-01f, 6.4883e-01f, -6.6644e-01f, 

        -2.5144e-03f, 3.5450e-01f, 3.0118e-01f, -5.5226e-01f, -1.2470e-01f, -3.9904e-01f, -4.8923e-02f, 1.7506e-01f, 

        1.0292e+00f, -5.7593e-02f, 2.7568e-01f, -5.1180e-02f, -8.9842e-02f, 3.4876e-01f, 3.1608e-01f, 1.9179e-01f, 

        6.3041e-01f, -7.4309e-02f, -2.8682e-01f, 1.6942e-01f, 1.8012e-01f, 7.1786e-02f, 7.6420e-03f, -2.2229e-01f, 

        -1.2999e-01f, -6.7847e-02f, 5.4811e-02f, -1.1835e-01f, 5.4789e-01f, -1.5231e-01f, -3.2827e-01f, 7.6493e-03f, 

        -2.5016e-01f, 6.1385e-01f, -4.0395e-01f, 2.6721e-01f, 2.9941e-01f, 4.7258e-01f, 3.6990e-01f, 7.2643e-02f, 

        -4.9942e-01f, -5.5052e-01f, -3.4235e-01f, -6.8667e-02f, 6.3654e-03f, 4.4814e-01f, 6.0796e-01f, -2.5658e-01f, 

        -3.4234e-01f, 4.5032e-01f, 5.0773e-01f, 1.6386e-01f, 4.2968e-01f, -3.4328e-01f, 3.7796e-01f, 5.8996e-01f, 

        -8.3535e-02f, 4.3864e-01f, -7.9361e-02f, -2.9986e-01f, 9.7105e-02f, -1.4837e-01f, -1.4710e-01f, 4.4202e-01f, 

        1.0733e-01f, 3.6269e-01f, -2.2419e-01f, -4.0010e-03f, -7.4339e-02f, -1.1563e-01f, 3.5610e-01f, -2.8153e-01f, 

        4.0164e-01f, 9.1541e-02f, 2.2057e-01f, 1.0303e-01f, -2.2733e-01f, -8.6569e-02f, -2.7748e-01f, 4.9082e-02f, 

        2.9407e-02f, -4.4241e-01f, 4.5422e-01f, 4.8918e-01f, 5.5172e-01f, -2.4202e-01f, -3.4253e-01f, -5.1241e-01f, 

        -1.8611e-01f, 2.5951e-01f, -2.1474e-01f, 2.1506e-01f, 8.8506e-01f, -3.6799e-01f, -1.0726e+00f, 1.8208e-01f, 

        9.2222e-01f, 8.4871e-01f, 9.5083e-01f, -1.0312e+00f, -9.1082e-04f, -2.1776e-02f, 1.9303e-01f, 3.1149e-01f, 

        2.3044e-01f, 4.1024e-02f, 5.6362e-01f, -4.0230e-01f, 1.3979e-01f, 7.7594e-02f, 2.9790e-01f, -7.4548e-01f, 

        4.3518e-01f, 3.9579e-01f, 4.6776e-01f, -8.7048e-02f, 4.9432e-01f, -1.5754e-03f, 1.4752e-01f, -7.4473e-01f, 

        -4.6957e-01f, -2.9300e-01f, 4.8589e-01f, 1.1624e+00f, -1.2028e+00f, 1.8749e-01f, -4.8307e-01f, -9.6818e-01f, 

        6.2619e-01f, 4.8367e-01f, 2.0662e-01f, 1.5386e-01f, -1.3982e-01f, 2.9168e-01f, 7.0809e-02f, -2.7215e-01f, 

        7.9514e-01f, -2.2078e-01f, 3.7353e-01f, 2.7789e-01f, 4.0717e-01f, -3.5085e-01f, -4.0433e-02f, 4.2157e-01f, 

        -1.0934e+00f, -2.0640e-01f, 9.5376e-01f, -6.8885e-01f, 1.0610e-01f, -5.6101e-01f, 1.4940e-02f, -1.6262e-03f, 

        4.6963e-01f, 4.6589e-01f, -2.7516e-01f, -5.9124e-01f, -3.8261e-02f, -5.4224e-01f, -2.1332e-02f, 5.2490e-01f, 

        1.0529e-01f, 1.0790e-02f, -2.8554e-02f, 9.4028e-02f, -3.6146e-02f, -9.9284e-02f, 5.9123e-02f, -8.5406e-03f, 

        7.7879e-02f, -1.9711e-02f, -3.6000e-02f, 7.2186e-03f, -8.4276e-02f, 4.4508e-02f, 2.1910e-02f, -9.4288e-02f, 

        1.1860e-01f, -7.2717e-02f, 1.1564e-01f, -6.0251e-02f, -9.6594e-02f, 6.5612e-02f, -1.0260e-01f, 5.9381e-02f, 

        -6.5400e-02f, -7.3183e-02f, 1.0369e-01f, -8.6563e-02f, 9.4489e-02f, -1.4547e-01f, -3.5856e-02f, -6.5887e-02f, 

        -1.1102e-01f, 4.0111e-02f, -1.0167e-01f, -9.0501e-02f, 3.9213e-02f, -5.1301e-02f, 4.7498e-02f, -4.9805e-02f, 

        -2.9960e-02f, -5.1765e-02f, -9.3702e-02f, 7.1155e-02f, -4.5026e-02f, -5.7810e-02f, 9.7022e-02f, -7.7261e-02f, 

        6.9449e-02f, -4.2020e-02f, -9.6844e-02f, 1.4037e-02f, -9.3613e-02f, -2.6362e-02f, -1.4206e-01f, -7.3002e-02f, 

        3.9705e-02f, -1.4686e-01f, 5.5974e-02f, -8.1046e-02f, -1.1404e-01f, -1.3250e-01f, -7.2778e-02f, -3.9649e-02f, 

        -8.2313e-02f, -1.4294e-01f, 5.7102e-02f, -1.0774e-01f, 3.4025e-02f, -1.0885e-01f, -9.0037e-02f, -1.5291e-01f, 

        -8.8434e-02f, -8.6060e-03f, -7.3824e-02f, -1.2948e-01f, 3.3235e-02f, -7.0121e-02f, 4.3596e-03f, -6.0114e-02f, 

        -1.2667e-01f, -1.2198e-01f, -9.1265e-02f, -7.0501e-02f, -1.2369e-01f, -9.3878e-02f, -1.8926e-03f, -1.3961e-01f, 

        -1.9416e-02f, -9.9568e-02f, -1.2132e+00f, -1.7004e-01f, -4.7503e-01f, -2.0811e-01f, -1.4045e-01f, -2.2023e-01f, 

        2.3781e+00f, 1.9107e-02f, -1.1115e-01f, -7.5183e-01f, 2.3841e-01f, -2.1423e-01f, 2.8534e-01f, 2.5083e-02f, 

        -6.8409e-01f, -2.1520e-01f, 1.3613e+00f, -1.4226e+00f, 1.4581e-01f, 5.1853e-01f, 4.0137e-01f, 1.5703e+00f, 

        -4.2099e-01f, 5.3589e-02f, -1.9528e-01f, -2.3208e-01f, 1.4129e-01f, 4.8666e-01f, 4.8048e-01f, -2.2836e-01f, 

        -9.1935e-01f, 3.3720e-01f, -5.6713e-01f, -6.3949e-03f, 1.1139e+00f, -3.6876e-01f, -2.0377e-01f, -8.4685e-01f, 

        6.0676e-02f, -3.4089e-01f, -7.9344e-01f, 1.3778e-01f, 2.2716e-01f, 9.8819e-02f, -3.9279e-01f, -2.3474e-01f, 

        1.2294e+00f, 1.5391e-01f, -4.6238e-01f, -2.1968e-01f, -1.7333e-01f, 5.0680e-01f, 1.7957e-02f, -1.0573e-01f, 

        -3.1658e-01f, -1.5503e-01f, -8.0091e-01f, 2.3756e-01f, -2.1012e-01f, 2.6960e-01f, -5.5664e-01f, 1.7829e-01f, 

        -1.1368e-01f, 3.2528e-01f, 6.6435e-02f, -2.6355e-01f, 1.1162e-01f, -2.9607e-02f, 8.2819e-01f, -4.4217e-01f, 

        -2.8893e-01f, 2.3497e-01f, -1.2846e-01f, -2.1037e-01f, -2.0172e-01f, -5.3251e-01f, -2.9349e-01f, 3.0365e-02f, 

        -2.1812e-01f, 2.5383e-01f, -3.0275e-01f, 1.0706e-01f, -5.3486e-01f, -3.1268e-01f, -3.3055e-02f, -6.3407e-03f, 

        -6.1615e-01f, 2.7545e-01f, -5.4513e-01f, 7.7974e-01f, -1.0199e-01f, -1.1062e-01f, -5.5573e-02f, -9.1765e-02f, 

        -5.3030e-02f, -1.5195e-01f, -6.9127e-02f, -1.2697e-01f, -3.7470e-02f, -1.4110e-01f, -6.1798e-02f, -9.7794e-02f, 

        -1.2068e-01f, -9.5235e-02f, 7.7730e-02f, -1.0645e-01f, -2.3486e-02f, -1.1958e-01f, -4.9508e-02f, -1.0554e-01f, 

        -6.3229e-02f, 8.5099e-02f, -1.2558e-01f, -6.4095e-03f, 2.8086e-02f, -9.1949e-02f, 6.5428e-03f, -8.9274e-02f, 

        -4.9444e-02f, -1.3745e-01f, 2.5409e-02f, 4.7200e-03f, -1.0520e-02f, -1.8910e-02f, -4.3678e-02f, -6.7040e-02f, 

        -4.2387e-02f, -5.3225e-02f, -4.2800e-03f, 5.8517e-02f, -4.8780e-02f, -8.9100e-02f, -2.1140e-02f, 1.2210e-03f, 

        -1.6313e-02f, -4.8140e-02f, -4.1226e-02f, -8.4171e-02f, 5.7828e-02f, 5.8162e-02f, -1.0678e-01f, -3.1475e-02f, 

        -1.0335e-01f, -3.2214e-02f, -1.0199e-01f, -6.3602e-02f, 1.3796e-02f, -8.9774e-02f, -7.5119e-02f, -9.4832e-02f, 

        -6.4977e-02f, -8.8059e-02f, -1.2600e-01f, -1.5209e-01f, -1.4034e-01f, -1.0341e-01f, -1.3535e-01f, -1.0248e-01f, 

        -3.9343e-02f, -8.8460e-02f, -5.5185e-02f, -1.3526e-01f, -9.1906e-02f, -1.0842e-01f, -5.7169e-02f, -1.2146e-01f, 

        -1.0030e-01f, -9.1810e-02f, 6.8920e-02f, -6.3404e-02f, -2.2707e-03f, -3.8912e-02f, -8.8964e-02f, -4.6903e-02f, 

        -8.8230e-02f, -5.2585e-02f, -4.7381e-02f, 7.1245e-03f, -2.5077e-02f, -9.8668e-02f, -1.1933e-01f, -1.4584e-01f, 

        -5.8713e-02f, 9.9986e-03f, -5.1523e-02f, -1.3650e-01f, 2.0643e-02f, -7.4048e-02f, 4.8050e-02f, -7.8945e-02f, 

        -1.5667e-01f, -2.5644e-04f, -7.6649e-02f, 1.2982e-02f, 8.8679e-03f, -1.4922e-01f, 8.1104e-02f, -1.4334e-01f, 

        9.5861e-02f, -1.4538e-01f, -9.0176e-02f, 7.6300e-02f, -1.2377e-01f, 2.9206e-02f, -7.3920e-03f, -1.0788e-01f, 

        1.0894e-01f, -1.2295e-01f, 3.0740e-02f, -1.0238e-01f, -8.4793e-02f, -1.1555e-01f, -7.9181e-02f, -2.0909e-02f, 

        -9.7999e-02f, -8.5884e-02f, 3.6695e-02f, -1.3731e-01f, 8.2606e-02f, -8.9423e-02f, 4.8611e-03f, -1.0443e-01f, 

        -1.0794e-01f, 6.6817e-03f, -4.5319e-02f, -4.4744e-02f, 6.4147e-02f, -1.2630e-01f, 9.2781e-02f, -8.0565e-02f, 

        -9.4598e-02f, 7.5752e-02f, -1.2193e-01f, 3.2415e-02f, -1.1540e-01f, -1.3372e-01f, 5.5413e-02f, -9.6704e-02f, 

        2.2926e-02f, -1.0870e-01f, -3.0199e-02f, -4.6818e-02f, -9.7907e-02f, 3.0073e-02f, -7.2652e-02f, -6.8943e-02f, 

        4.6314e-02f, -2.6548e-02f, 1.0199e-01f, -1.7156e-02f, -3.6960e-02f, -1.8432e-03f, -9.0249e-02f, -6.4567e-03f, 

        -6.2117e-02f, -9.7992e-02f, 5.6120e-02f, -9.2477e-02f, 4.6711e-02f, -7.7933e-02f, -1.0166e-01f, -2.3639e-02f, 

        -8.3811e-02f, -8.8943e-03f, -5.4600e-02f, -1.4681e-01f, 6.3310e-02f, -7.7331e-02f, 3.3135e-02f, -1.2999e-01f, 

        -9.7393e-02f, -5.8575e-02f, -9.7071e-02f, -1.0266e-01f, -1.1637e-01f, -6.8176e-02f, -9.7955e-02f, -8.5100e-02f, 

        -9.5186e-02f, -2.2276e-02f, -4.2553e-02f, -7.8501e-02f, -3.0155e-02f, -6.6591e-02f, -5.9519e-02f, -5.7122e-02f, 

        -1.1309e-01f, -4.3632e-02f, -7.7493e-02f, -5.6740e-02f, -9.7466e-02f, -1.4632e-01f, -6.7522e-02f, -1.1532e-01f, 

        -1.2586e-01f, -1.2931e-01f, -5.7014e-02f, -1.0355e-01f, -1.1514e-01f, -9.6927e-02f, -1.2297e-01f, -5.8714e-02f, 

        -1.1702e-01f, -6.2818e-02f, -7.5139e-02f, -7.8517e-02f, -1.0845e-01f, -7.1883e-02f, -8.1745e-02f, -1.2657e-01f, 

        -5.0068e-02f, -5.0686e-03f, -1.1049e-01f, -2.8045e-02f, -1.3116e-01f, -6.8946e-02f, -4.9338e-02f, -5.3628e-02f, 

        -3.7774e-02f, -7.8889e-02f, -6.4310e-02f, -9.5900e-02f, -8.7632e-02f, -4.6786e-02f, -6.4606e-02f, -9.2553e-02f, 

        -4.8032e-02f, -9.1856e-02f, -6.6280e-02f, -6.5595e-02f, -5.5543e-02f, 6.5925e-02f, -9.6395e-03f, -7.8439e-02f, 

        -9.6274e-02f, -4.9521e-02f, -1.0223e-01f, -8.5812e-02f, -7.9467e-02f, -2.1455e-02f, -5.2308e-02f, -9.6498e-02f, 

        -1.0038e-01f, -1.0236e-01f, -1.1270e-01f, -7.4560e-02f, -4.4627e-02f, 3.0095e-03f, -3.9255e-03f, -7.4502e-02f, 

        -7.2375e-02f, -1.0772e-01f, -7.5142e-02f, -7.2919e-02f, -9.1525e-02f, -6.9332e-02f, -1.1977e-01f, -5.1602e-02f, 

        -4.8844e-02f, -8.4383e-02f, -4.9473e-02f, -1.1657e-01f, -1.4359e-01f, -1.4418e-03f, -9.5172e-02f, -1.1939e-01f, 

        -3.7779e-02f, -9.6110e-02f, -6.3707e-02f, -1.1018e-01f, -8.7611e-02f, -9.8131e-02f, -9.4890e-02f, -6.1370e-02f, 

        -1.3041e-02f, -1.5540e-01f, 1.6919e-02f, -1.5783e-01f, -1.1646e-01f, -9.7143e-02f, -1.2418e-01f, 1.3080e-01f, 

        -1.3484e-01f, -3.3076e-02f, -1.0583e-01f, -1.5464e-01f, 5.2109e-02f, -1.7310e-01f, -1.1319e-01f, -1.4943e-01f, 

        -5.4069e-02f, -1.2247e-01f, -4.9540e-02f, -5.4779e-02f, -1.2630e-01f, -1.4055e-01f, -1.6030e-02f, -1.3924e-01f, 

        2.6068e-02f, -3.9614e-02f, -3.4026e-02f, -9.8659e-02f, -1.3346e-01f, -4.6291e-02f, -1.0591e-01f, -1.5059e-01f, 

        -2.5406e-02f, -9.0480e-02f, 8.7562e-03f, -7.8738e-02f, -8.2637e-02f, -4.0985e-02f, -1.2162e-01f, -7.9767e-02f, 

        -8.5813e-02f, -1.4967e-01f, -9.1284e-03f, -6.9236e-02f, -4.7659e-02f, -9.0382e-02f, -4.7590e-02f, -1.7043e-01f, 

        -8.0333e-02f, -6.9244e-02f, -1.4477e-01f, -1.4104e-01f, 4.5261e-03f, -1.1651e-01f, -9.4735e-03f, -1.2070e-01f, 

        -4.9043e-02f, -1.3352e-01f, -8.4184e-02f, -3.9764e-02f, -1.2621e-01f, -8.4826e-02f, -4.1434e-02f, -1.4573e-01f, 

        -4.5492e-02f, -3.0333e-02f, -8.6471e-02f, -9.6858e-02f, -1.4069e-01f, -6.2883e-02f, -1.2499e-01f, -7.6189e-02f, 

        -8.6258e-02f, -1.4274e-01f, -7.2712e-02f, -5.1945e-02f, 3.8025e-01f, 3.5837e-02f, -2.1833e-01f, 5.6387e-02f, 

        4.8778e-01f, -1.0439e-01f, 4.4014e-01f, 8.6263e-02f, 6.7973e-01f, -1.0367e+00f, 1.0479e-01f, 1.7919e+00f, 

        4.4615e-01f, 1.6919e-02f, 1.4146e-02f, 1.3680e-01f, 4.3298e-01f, 4.0617e-01f, 7.4521e-01f, 7.5798e-01f, 

        1.7873e-01f, 5.9381e-01f, 2.2278e-01f, 1.0242e-01f, 3.2602e-01f, 3.0864e-01f, 1.3256e-01f, 2.2688e-01f, 

        2.0621e-01f, 1.5215e-01f, -3.0864e-01f, 6.2488e-02f, 3.1293e-01f, 2.1594e-01f, -3.3743e-01f, 2.2267e-01f, 

        1.5594e+00f, -2.6808e-02f, 2.3642e-01f, -6.6112e-01f, -2.6730e-01f, 1.3502e+00f, -5.1735e-01f, -3.8562e-01f, 

        -4.9422e-01f, 3.9983e-01f, -1.3592e-02f, -6.9157e-02f, 2.5940e-01f, -2.0774e-03f, 7.2992e-05f, 2.3854e+00f, 

        4.6553e-02f, -3.7741e-01f, 1.5532e-01f, 2.7888e-01f, -3.6628e-01f, 4.5515e-02f, 2.2939e-01f, -5.5247e-01f, 

        -7.8171e-01f, -1.2363e-02f, 1.3006e-02f, -1.0044e-01f, -1.5855e-02f, 7.2508e-01f, 6.5868e-01f, -5.1746e-01f, 

        -3.8789e-01f, -6.0408e-01f, -4.2940e-01f, 1.2711e-01f, -2.0359e-01f, 8.0517e-01f, 1.0501e-02f, 2.4445e-01f, 

        -2.5075e-02f, -4.1850e-01f, -7.8078e-01f, 3.2493e-01f, -8.9125e-02f, -2.1016e-01f, 2.6519e-01f, -8.4113e-02f, 

        -4.2347e-01f, 1.7515e-01f, 2.3735e-01f, -4.8575e-01f, -4.2679e-01f, 1.0531e-01f, -1.6642e-02f, 3.7218e-02f, 

        1.7883e-02f, -4.9548e-02f, -5.0707e-02f, -5.6623e-02f, 1.0653e-02f, 3.8376e-02f, -6.2046e-03f, -1.3294e-02f, 

        1.1370e-02f, -6.9047e-02f, -3.5472e-02f, 4.3047e-04f, -3.6349e-02f, -7.4728e-02f, -2.3488e-02f, -7.9599e-02f, 

        5.7302e-02f, 8.5372e-02f, -9.3317e-02f, -5.8554e-02f, -7.9612e-02f, -8.5260e-02f, -8.7163e-02f, -1.2224e-01f, 

        -6.0280e-02f, -7.5485e-02f, -1.0758e-01f, -1.3648e-01f, -6.3606e-02f, -8.2882e-02f, -1.1777e-01f, -1.0498e-01f, 

        -7.1961e-02f, -1.3941e-01f, -4.0953e-02f, -1.4685e-01f, -3.4266e-02f, -1.0335e-01f, -9.3550e-02f, 1.2788e-02f, 

        -1.1250e-01f, -6.9136e-02f, -7.0426e-02f, -7.4890e-02f, -5.5457e-02f, -1.1757e-01f, -1.9134e-03f, -1.2307e-01f, 

        -5.0367e-02f, -1.2093e-01f, -1.0110e-01f, -9.8285e-02f, -5.4163e-02f, -6.8198e-02f, -4.4747e-02f, -7.1508e-02f, 

        -8.4126e-02f, -7.8980e-02f, -7.6472e-02f, -7.4396e-02f, -1.4580e-01f, -1.1562e-01f, -1.1167e-01f, -6.9994e-02f, 

        8.1421e-03f, 9.2565e-02f, -6.3637e-02f, -1.2122e-01f, -4.7978e-02f, -3.5959e-02f, -1.2763e-01f, -2.9289e-02f, 

        -1.1993e-01f, -4.9383e-02f, -5.2773e-02f, 9.9130e-03f, -8.5854e-02f, 5.5420e-02f, -6.1178e-02f, -6.2330e-03f, 

        -1.2993e-01f, -1.0785e-01f, -1.0658e-01f, -8.9566e-02f, -2.8883e-02f, -1.3327e-01f, -5.9665e-02f, -1.0001e-01f, 

        -2.0110e-02f, -5.4237e-02f, -6.2235e-02f, -6.4749e-02f, -8.7377e-02f, -1.5047e-01f, 1.9809e-02f, -8.5916e-02f, 

        8.6559e-02f, -8.8437e-02f, -3.2095e-02f, -1.2074e-01f, -7.0945e-02f, 6.3777e-03f, -3.6633e-02f, -1.0773e-01f, 

        3.6480e-02f, -1.2230e-01f, 6.2543e-02f, -1.0303e-02f, -9.5793e-02f, -6.4326e-02f, -8.1763e-02f, -5.9456e-02f, 

        -9.2938e-02f, -9.2024e-02f, 3.3105e-02f, -8.9438e-02f, 4.0874e-02f, -1.1217e-01f, -4.9521e-02f, -1.3283e-01f, 

        -1.0900e-01f, -3.9609e-02f, -1.4812e-01f, -1.1195e-01f, 7.1837e-03f, -1.6692e-01f, 1.4499e-02f, -6.3540e-02f, 

        -4.4306e-02f, -1.1753e-01f, -1.4564e-01f, -5.4598e-02f, -1.4562e-01f, -1.5113e-01f, -1.3737e-02f, -1.3876e-01f, 

        3.7043e-02f, -1.2154e-01f, -1.2758e-01f, -8.7559e-02f, -1.2819e-01f, -6.7962e-02f, -8.6345e-02f, -9.3334e-02f, 

        -5.5233e-02f, -1.4901e-01f, -3.9404e-02f, -1.3829e-01f, -1.1728e-01f, -1.0689e-01f, -1.4693e-01f, -1.0022e-01f, 

        -1.4986e-01f, -9.9806e-02f, 1.3743e-02f, -3.8478e-02f, -3.5156e-02f, -1.2873e-01f, -1.4813e-01f, -6.6591e-02f, 

        -1.0029e-01f, -5.8454e-02f, -1.2942e-01f, -1.0413e-01f, -4.9423e-02f, -1.2061e-01f, 2.2229e-03f, -1.3541e-01f, 

        -7.9665e-02f, -4.4457e-02f, -1.1930e-01f, -4.1860e-02f, -1.1342e-01f, -1.5394e-01f, -2.6604e-02f, -1.4445e-01f, 

        -4.4569e-02f, -7.5735e-02f, 2.2465e-02f, -7.2111e-02f, 4.0122e-02f, -6.9863e-02f, 6.9991e-03f, 1.7878e-02f, 

        -4.7982e-02f, -5.6483e-02f, -1.0529e-01f, 6.1469e-02f, -6.4850e-02f, 8.7055e-02f, -1.3870e-01f, -1.7171e-02f, 

        -6.2213e-02f, -8.0583e-02f, -1.4627e-02f, -9.3842e-02f, -9.0564e-02f, -6.7011e-02f, -1.4064e-02f, -7.9821e-02f, 

        -3.1346e-02f, -6.6390e-02f, -1.3650e-02f, -1.7407e-02f, -3.4103e-02f, -1.8575e-02f, -2.2503e-02f, 2.1059e-03f, 

        -4.6481e-02f, -1.6145e-01f, 9.8672e-02f, -1.3759e-01f, 4.0473e-02f, 4.6613e-02f, 7.9310e-02f, 1.7484e-02f, 

        4.2699e-03f, 4.6143e-02f, 1.0899e-01f, -1.3623e-01f, 2.6310e-02f, -3.0744e-02f, 7.2299e-02f, 3.4855e-02f, 

        -4.8611e-02f, -2.8342e-02f, -4.9703e-02f, 7.1403e-02f, 7.4487e-02f, -4.9536e-02f, -1.3163e-02f, -3.3889e-02f, 

        -5.5792e-02f, -7.0592e-02f, -5.9889e-02f, 4.2148e-02f, -1.6387e-02f, 1.3075e-02f, 1.3704e-01f, -4.2656e-02f, 

        -4.6503e-01f, 6.0982e-01f, 1.3874e-01f, 2.8369e-02f, -2.5012e-01f, 1.2479e+00f, 3.4354e-02f, -9.2100e-02f, 

        1.5395e-01f, 4.9982e-02f, 1.2728e-03f, 2.2485e-01f, 9.5262e-02f, 5.6696e-02f, -1.0304e-01f, -7.7690e-02f, 

        -6.9947e-02f, 1.5239e-01f, -2.3011e-02f, -6.8482e-02f, -5.0360e-02f, -1.0231e-01f, -7.3937e-03f, 5.2797e-03f, 

        -6.8306e-02f, 2.2150e-01f, -3.8044e-02f, -2.6910e-02f, -8.4583e-02f, -7.3030e-02f, -7.0861e-02f, -1.1489e-02f, 

        -7.6564e-02f, -1.5867e-01f, -2.5622e-02f, -4.9558e-02f, -1.7550e-02f, -7.5346e-02f, -9.2583e-02f, -2.8175e-02f, 

        -9.5871e-02f, 3.5265e-02f, -3.3707e-02f, -1.1297e-01f, 3.6749e-02f, -1.6187e-01f, -3.6372e-02f, -1.0454e-01f, 

        -1.5509e-01f, 9.8644e-03f, -8.9913e-02f, -5.4312e-02f, -3.6544e-02f, -1.2264e-01f, 1.3193e-02f, -1.5898e-01f, 

        -2.0285e-03f, -1.3402e-01f, -4.2756e-02f, -8.3599e-02f, -6.1240e-02f, -3.5959e-02f, -1.5053e-01f, -1.1918e-01f, 

        2.3958e-02f, -6.0117e-02f, 6.9923e-03f, -4.8146e-02f, -1.2265e-01f, 1.0358e-02f, -9.2314e-02f, -6.0384e-02f, 

        -1.2210e-01f, -1.0432e-01f, 2.8976e-02f, -1.4111e-01f, 2.5594e-02f, -8.7289e-02f, -6.9208e-02f, -8.5928e-02f, 

        -7.8955e-02f, 1.8755e-02f, -8.1647e-02f, -1.2711e-01f, 4.0375e-02f, -6.6808e-02f, -2.5055e-02f, -7.3441e-02f, 

        -1.1418e-01f, -9.8547e-02f, -7.8460e-02f, -7.1560e-02f, -1.2783e-01f, -1.0157e-01f, 4.3884e-02f, -1.2380e-01f, 

        -3.4291e-02f, -1.2350e-01f, -6.6597e-02f, -9.7052e-02f, -8.1977e-02f, -5.0329e-02f, -9.0252e-02f, -9.6828e-02f, 

        -4.2217e-02f, -1.4440e-01f, 3.2890e-02f, -1.2697e-01f, -3.0903e-02f, -1.0809e-01f, -1.4855e-01f, -9.1135e-02f, 

        -1.0840e-01f, -1.4558e-01f, -1.3136e-02f, -9.8595e-02f, -3.4157e-02f, -1.0166e-01f, 3.1202e-01f, 3.6287e-01f, 

        1.4888e-01f, -5.2095e-01f, -3.4383e-01f, -1.1075e-01f, -2.1193e-01f, -6.1439e-02f, -1.4768e-01f, -3.2345e-01f, 

        1.9136e-01f, 1.8070e+00f, 1.9521e-01f, -4.9438e-02f, -3.2028e-01f, 4.3516e-01f, -1.6759e-01f, -1.1382e+00f, 

        -5.6198e-02f, -3.0738e-01f, -3.8797e-02f, 7.9335e-01f, 1.9014e-02f, 3.7633e-01f, 1.5763e-01f, 2.0048e-01f, 

        4.5208e-01f, -3.8667e-01f, 5.6189e-01f, -8.5147e-01f, 8.1095e-02f, 2.5686e+00f, -4.4277e-01f, -2.6714e-01f, 

        -8.4777e-03f, -2.7542e-02f, -1.6711e-01f, -5.2561e-01f, -1.6887e-01f, -8.6110e-01f, -5.7523e-01f, 8.9717e-01f, 

        1.4984e-01f, -1.3274e-01f, -2.3005e-01f, 3.4674e-01f, 1.2131e-01f, 4.3971e-02f, 5.3381e-01f, -4.9626e-01f, 

        2.6128e-01f, 7.5310e-01f, 2.4071e-01f, -8.3758e-01f, 1.2241e-02f, -4.0988e-01f, -3.2580e-01f, 2.7893e-01f, 

        -3.1930e-01f, -4.9590e-01f, 7.3580e-01f, 5.9343e-01f, -2.2503e-01f, -3.5787e-01f, -1.1913e-02f, -5.4002e-01f, 

        -2.6539e-01f, -3.6652e-01f, 3.1008e-01f, 2.8785e-02f, -3.6284e-01f, 6.2424e-01f, -5.0915e-01f, -2.3508e-02f, 

        -6.3578e-01f, -2.6750e-01f, -6.4697e-01f, 8.8159e-01f, -8.5697e-02f, -1.7766e-01f, 2.7321e-02f, -3.2200e-02f, 

        2.0819e-01f, 4.5175e-02f, -9.6471e-02f, 4.3548e-01f, -3.2255e-01f, -6.6374e-01f, -3.0178e-01f, -2.6337e-01f, 

        -1.0440e-01f, -3.4713e-01f, 2.4176e-02f, -7.8799e-02f, 2.6746e-01f, -2.3185e-01f, 1.7226e-01f, 5.1528e-01f, 

        6.4800e-02f, 4.9825e-01f, 9.1358e-01f, 2.6016e-01f, 1.7262e-01f, 3.6355e-01f, 5.2408e-01f, 6.8473e-01f, 

        -4.4160e-01f, -3.2777e-01f, -4.9194e-01f, 5.1428e-01f, -1.3363e-01f, -1.9415e-02f, -2.1367e-01f, 2.7194e-01f, 

        3.2239e-01f, 1.0705e-01f, 3.4824e-01f, -1.6772e-01f, -1.3127e-02f, 1.5103e+00f, -6.1160e-01f, 3.6777e-01f, 

        4.0601e-01f, 2.1626e-02f, -2.5226e-03f, 3.3255e-01f, 3.3236e-01f, -2.2526e-01f, 2.8661e-01f, 4.9965e-01f, 

        6.8967e-01f, 4.2683e-02f, 3.7599e-01f, 8.3408e-02f, 1.7447e-01f, -3.1535e-01f, -5.5882e-01f, 2.7950e-01f, 

        1.1408e-01f, -5.8060e-02f, 3.2171e-01f, 3.2447e-01f, 1.4578e-01f, 5.2644e-01f, 2.2398e-01f, 8.4869e-02f, 

        5.1556e-03f, -2.5961e-01f, 1.7177e-01f, 3.8769e-01f, -5.0629e-01f, 1.2630e-01f, 6.1275e-03f, 7.2740e-01f, 

        2.5622e-01f, -1.3503e-01f, 6.0893e-01f, 8.3069e-02f, 2.5862e-02f, -8.8487e-02f, 1.7553e-02f, 8.3415e-02f, 

        3.1571e-01f, 4.9110e-01f, 2.7978e-02f, 5.1246e-02f, 4.7735e-01f, 8.0383e-02f, 2.6322e-01f, 4.2356e-02f, 

        3.0848e-01f, -7.7733e-02f, -3.5569e-01f, 2.4166e-01f, -4.7560e-02f, 7.3949e-01f, -4.0838e-01f, 8.5162e-01f, 

        2.3515e-02f, -5.3311e-01f, 1.7489e-02f, -6.7434e-02f, -2.2920e-02f, 4.3691e-02f, 3.7007e-02f, -1.1989e-01f, 

        8.9559e-02f, -9.1105e-03f, 5.0839e-02f, -7.7174e-02f, -3.9486e-02f, -1.6572e-02f, -3.3369e-02f, 7.2905e-02f, 

        9.5595e-02f, -5.9284e-02f, 1.0200e-01f, -5.8102e-02f, 5.1015e-02f, -1.0847e-01f, -9.4113e-02f, 1.1552e-01f, 

        -7.1022e-02f, 6.2981e-02f, 6.4967e-02f, -1.4644e-01f, 1.3613e-01f, -1.0570e-01f, 6.0907e-02f, -8.4876e-02f, 

        3.9832e-02f, 1.6545e-02f, -5.7021e-02f, 7.1726e-02f, -3.8841e-02f, -3.4359e-02f, 9.5136e-02f, -2.1206e-02f, 

        5.9698e-02f, -3.3008e-02f, 9.5022e-03f, -8.5501e-02f, -6.4977e-02f, 9.1376e-02f, -1.0923e-01f, -5.5422e-02f, 

        9.4177e-02f, -3.6667e-02f, 1.1825e-01f, -9.0339e-02f, -7.7004e-02f, 1.2246e-02f, -1.3206e-01f, -1.1476e-02f, 

        -4.5055e-02f, -9.6031e-02f, 9.7231e-02f, -1.3379e-01f, 2.0494e-02f, -1.3490e-01f, -8.7870e-02f, -2.4259e-02f, 

        -5.8952e-02f, 4.4977e-03f, -1.1165e-01f, -4.7284e-02f, 6.6634e-02f, -6.4009e-02f, 9.7558e-02f, -7.4355e-02f, 

        9.8966e-03f, -5.1113e-02f, -5.4011e-02f, 3.3607e-02f, -2.3538e-02f, -8.0477e-02f, 3.6540e-02f, -3.2585e-02f, 

        9.7236e-02f, 4.9545e-04f, -7.7841e-02f, -1.7302e-02f, -6.7874e-02f, 1.3962e-02f, -9.3169e-02f, -9.8809e-02f, 

        1.0220e-02f, -9.1930e-02f, 3.8544e-02f, -1.0900e-01f, -5.5255e-02f, -9.1094e-02f, -1.0082e-01f, 3.8298e-02f, 

        -8.0217e-02f, -8.8005e-02f, 1.3759e-02f, -1.1982e-01f, 7.4799e-02f, -8.4140e-02f, -5.9190e-02f, -1.4277e-01f, 

        -7.8367e-02f, 2.5320e-02f, 4.1559e-02f, -1.3673e-01f, 4.9113e-02f, -9.4466e-02f, 3.7752e-02f, -3.7409e-02f, 

        -1.1484e-01f, 1.1630e-01f, -9.5003e-02f, 1.8912e-02f, -5.6822e-02f, -7.2475e-02f, 5.4325e-02f, -1.1141e-01f, 

        4.0545e-02f, -7.0463e-02f, -1.0973e-01f, -9.6742e-02f, -1.1772e-01f, -4.9664e-02f, -1.1802e-01f, -1.3489e-01f, 

        5.8864e-02f, -1.4993e-01f, 9.0749e-02f, -5.4874e-02f, -3.7227e-02f, -7.0241e-02f, -8.6741e-02f, -2.3655e-02f, 

        -8.8711e-02f, -1.1989e-01f, 3.4966e-02f, -1.0869e-01f, 9.9881e-02f, 2.3428e-03f, -4.7221e-02f, -5.8455e-02f, 

        -9.9289e-02f, -9.6359e-03f, -9.0292e-02f, -8.9677e-02f, 6.9096e-02f, -9.9169e-02f, 2.5277e-02f, -6.8224e-02f, 

        -8.7988e-02f, -1.0341e-01f, -1.2316e-01f, -1.0361e-01f, -8.9899e-02f, -1.3390e-01f, 4.3663e-02f, -8.9208e-02f, 

        4.2538e-02f, -7.1568e-02f, -8.6178e-02f, -8.3576e-02f, -7.4208e-02f, -6.0678e-02f, -8.1782e-02f, -8.5115e-02f, 

        -8.6972e-03f, -8.8599e-02f, 4.9042e-02f, -1.2611e-01f, -8.8243e-02f, -3.5898e-02f, -1.6180e-01f, -4.6293e-02f, 

        -6.2297e-02f, -1.2761e-01f, -2.2151e-02f, -1.5244e-01f, -3.7029e-02f, -6.4862e-02f, 2.2827e-01f, 2.6816e-02f, 

        -3.6635e-01f, 4.6458e-01f, -6.0152e-01f, 2.5410e-01f, -4.3518e-03f, 8.5812e-01f, -1.4278e-01f, -3.7393e-01f, 

        5.1125e-01f, -2.7290e-01f, -1.8249e-01f, 1.9303e+00f, -6.5168e-01f, 7.9105e-02f, 5.0676e-02f, 4.9935e-01f, 

        -7.9175e-01f, -2.4164e-01f, -2.4165e-01f, -1.0617e-01f, 9.6992e-01f, -4.3951e-01f, 5.8762e-02f, -3.6799e-03f, 

        5.8705e-02f, -9.5692e-01f, -2.4550e-01f, -1.4614e-01f, 3.0101e-01f, -9.1791e-02f, -7.3372e-01f, -1.3319e-02f, 

        5.0817e-03f, 4.4258e-01f, -4.2396e-01f, 7.6482e-01f, 8.0971e-01f, 2.9893e-01f, -5.3372e-02f, -8.3181e-01f, 

        2.6002e-01f, 1.1465e-01f, -6.8031e-02f, -2.4123e-01f, -2.0829e-01f, -1.0986e+00f, -1.6471e-01f, 1.9515e-01f, 

        -6.6775e-01f, -2.6428e-01f, -6.7490e-01f, 1.6397e+00f, -9.3895e-02f, 2.2915e-01f, -9.2429e-01f, -8.4700e-01f, 

        8.7844e-03f, 4.7266e-01f, -8.8494e-01f, -1.1697e+00f, -6.5158e-01f, -8.5268e-01f, -9.7423e-01f, -2.2212e-01f, 

        1.2272e+00f, 2.3206e-01f, 1.3226e-01f, 6.0056e-01f, -3.4746e-01f, 1.9142e-01f, -1.6356e+00f, -1.2449e+00f, 

        -6.8912e-01f, -4.4595e-02f, 1.3307e+00f, 4.4968e-01f, -3.0861e-02f, 6.6109e-01f, -4.8738e-01f, 1.3792e-01f, 

        2.2793e-01f, -1.0006e-01f, 7.9667e-01f, 7.4220e-02f, -2.0037e+00f, -6.5035e-01f, -4.1450e-01f, -1.1011e+00f, 

        -1.2578e-01f, -1.1564e-01f, -1.1587e-01f, -1.0567e-01f, -9.1497e-02f, -1.9431e-01f, -1.0285e-01f, -1.3967e-01f, 

        -1.0747e-01f, -1.1253e-01f, -1.4423e-01f, 1.2805e-02f, -1.4683e-01f, -1.4379e-02f, -3.1248e-03f, -9.5984e-02f, 

        2.9753e-02f, -9.5264e-02f, -7.9585e-02f, -1.2861e-01f, -1.1141e-01f, 1.1244e-01f, -1.3342e-01f, -8.4980e-02f, 

        -1.7410e-01f, -1.2246e-01f, -1.8305e-02f, -1.0399e-01f, -1.2743e-01f, -1.1840e-01f, -5.7753e-02f, -7.9503e-02f, 

        -1.5126e-01f, -1.0370e-01f, -1.0921e-01f, -1.3356e-01f, -6.3619e-02f, -1.2977e-01f, -3.5221e-02f, -1.4785e-01f, 

        -1.5003e-01f, -1.1003e-01f, -1.4394e-01f, -5.4612e-02f, -1.2108e-01f, -1.2086e-01f, -4.3916e-02f, -1.3425e-01f, 

        -1.3859e-01f, -1.2772e-01f, -7.6715e-02f, -4.2747e-02f, -1.2160e-01f, -5.0938e-02f, -1.4656e-01f, -1.2114e-01f, 

        -7.9324e-02f, -1.2673e-01f, -8.7209e-02f, -1.4643e-01f, -1.0387e-01f, -1.5176e-01f, -1.2732e-01f, -8.1969e-02f, 

        -1.8198e-01f, -1.4940e-01f, -1.2892e-01f, -1.0806e-01f, -1.4354e-01f, -8.1824e-02f, -1.7376e-01f, -9.0794e-02f, 

        -1.2862e-01f, -6.5651e-02f, -7.3039e-02f, -1.6694e-01f, -1.3281e-01f, -1.3627e-01f, -2.1200e-02f, -1.3538e-01f, 

        -7.9036e-02f, -8.0121e-02f, -1.0706e-01f, -9.1501e-02f, -8.3696e-02f, -1.5382e-01f, -3.6668e-02f, -1.1125e-01f, 

        -1.0819e-01f, -1.5156e-01f, -1.1021e-01f, -1.4184e-01f, 1.3789e-01f, -1.2480e-01f, -1.2443e-01f, -6.9059e-02f, 

        -3.8907e-01f, -1.5014e-01f, -1.8756e-01f, -6.0035e-02f, -3.7755e-01f, -7.4061e-02f, -3.8595e-01f, 5.4443e-02f, 

        -2.7428e-01f, -2.9991e-02f, -1.9202e-01f, -1.2905e-01f, -2.2523e-01f, -1.8741e-01f, 3.0120e-02f, 2.2094e-01f, 

        1.3530e-02f, -2.3659e-01f, -2.4203e-01f, 1.0577e-01f, -2.0815e-01f, 8.1634e-02f, -2.0097e-01f, -1.2626e-01f, 

        -3.4971e-02f, -1.3799e-01f, 2.6286e-02f, -6.0518e-02f, -1.9324e-03f, -1.8985e-04f, -6.1744e-02f, -7.2383e-02f, 

        2.1522e-01f, 8.8614e-03f, 1.7151e-01f, -7.1024e-02f, -1.9461e-02f, -2.2134e-01f, 1.0614e-01f, -3.4461e-02f, 

        -2.0942e-01f, -1.2861e-02f, 1.5973e-01f, 2.2325e-01f, -9.3067e-02f, -7.6371e-02f, -1.2765e-02f, 1.2930e-01f, 

        -7.3596e-03f, -4.0579e-02f, -8.4228e-02f, 1.9433e-02f, -2.1811e-01f, 1.0164e-02f, 2.3239e-02f, 3.3537e-02f, 

        3.1381e-03f, -1.8434e-01f, -1.6464e-01f, -1.5845e-01f, -2.3355e-01f, 1.0787e-01f, -1.1567e-02f, 1.1778e-01f, 

        -2.4739e-02f, 1.0322e-02f, -1.2817e-01f, -2.7129e-01f, 6.7990e-02f, -8.4580e-02f, -5.3057e-02f, -1.9511e-01f, 

        -2.3771e-02f, -1.1364e-01f, 1.2329e-01f, -1.6537e-01f, -2.3887e-02f, 5.3371e-02f, -5.9708e-02f, -4.3158e-02f, 

        -1.9554e-01f, -3.8625e-02f, 5.8683e-02f, 5.4354e-02f, -4.5039e-02f, -1.0722e-01f, -9.1000e-03f, -7.9051e-02f, 

        -3.5761e-02f, -1.0494e-01f, -5.3793e-02f, -1.0574e-01f, -4.3913e-02f, -5.8727e-02f, -5.3737e-02f, -9.5270e-02f, 

        1.1543e-02f, -2.1033e-03f, 3.6209e-02f, -5.8838e-02f, 2.0842e-02f, -2.6986e-02f, 1.8496e-02f, 1.9195e-03f, 

        -1.8649e-02f, -5.6503e-02f, -4.7988e-02f, -5.1347e-02f, 1.2044e-02f, -7.4803e-02f, -7.8679e-02f, -1.1028e-01f, 

        -1.7220e-02f, -4.1546e-03f, 3.5872e-02f, -6.1103e-02f, -3.5485e-02f, -4.0842e-02f, -7.2456e-02f, -9.0900e-02f, 

        -1.4237e-01f, -8.1648e-02f, 2.7840e-02f, 1.0037e-02f, 1.1867e-02f, -4.7968e-02f, -1.0270e-01f, -9.4469e-02f, 

        -4.9086e-02f, -5.7081e-02f, -7.2580e-02f, -1.2260e-01f, -1.2851e-01f, -2.7573e-02f, -4.1127e-02f, -1.2578e-01f, 

        -5.1639e-02f, -7.9919e-02f, -8.4943e-02f, -3.9982e-02f, -6.9942e-02f, -1.1980e-01f, -7.9050e-02f, -5.4219e-02f, 

        -5.3263e-02f, -8.3206e-02f, -4.2961e-02f, -8.2159e-02f, -1.8697e-02f, -5.6868e-03f, -5.0220e-02f, 1.6084e-02f, 

        -6.2629e-02f, -1.5856e-02f, 5.2362e-03f, -5.0649e-02f, -8.9974e-02f, -2.5249e-02f, -6.8076e-02f, -2.0207e-02f, 

        -1.7312e-02f, -7.0106e-03f, -8.9258e-02f, 2.1773e-03f, -4.5876e-02f, -1.4149e-01f, -8.4715e-02f, -5.1782e-02f, 

        -1.1803e-01f, -1.0467e-01f, -1.1420e-01f, -6.5033e-02f, -1.1681e-01f, -3.0646e-02f, 1.6288e-02f, -6.6445e-02f, 

        -8.3029e-02f, 1.9335e-02f, -4.6552e-02f, -9.4425e-02f, -7.0763e-02f, -3.8922e-02f, -2.3943e-02f, -2.6785e-02f, 

        3.8152e-02f, -7.0454e-02f, 5.4674e-02f, -4.0060e-02f, 5.5807e-02f, 1.1850e-02f, 1.3554e-01f, 1.4620e-02f, 

        1.7040e-01f, 5.1442e-02f, -1.1177e-01f, -1.6122e-01f, 1.7209e-02f, -1.1215e-01f, 1.1463e-01f, -2.7554e-01f, 

        -1.0402e-01f, 1.3052e-01f, 4.0855e-02f, -5.4497e-02f, 3.7763e-01f, -3.6686e-01f, -1.9943e-02f, 1.4499e-01f, 

        3.2669e-01f, -7.1180e-03f, -5.3443e-01f, -7.7787e-02f, 8.8254e-02f, -6.3646e-02f, 6.4592e-02f, -7.5608e-01f, 

        -5.0490e-01f, -3.8234e-01f, 8.5657e-02f, 1.0674e-01f, 2.4452e-02f, 5.7240e-01f, -1.6199e-01f, -7.8742e-01f, 

        -4.8962e-01f, -1.4030e-01f, 1.1124e-01f, 1.9458e-01f, -3.3081e-01f, -5.8818e-01f, 8.7185e-01f, -3.8252e-01f, 

        -6.1907e-01f, 2.7814e-01f, -1.0757e-02f, -5.3628e-01f, -6.9798e-01f, -7.8374e-01f, -5.2673e-01f, -4.5265e-01f, 

        -4.1794e-01f, 6.0903e-01f, 1.2686e-01f, 4.0339e-03f, 1.4205e-01f, 5.8161e-02f, -5.1546e-01f, 7.4130e-01f, 

        -5.1531e-01f, -2.7026e-01f, 5.3842e-01f, 1.7825e-01f, -5.3190e-01f, 5.6524e-01f, -1.0026e+00f, 3.1189e-01f, 

        3.1428e-01f, 4.6064e-01f, 3.3386e-01f, 1.9743e-01f, 8.4784e-02f, 3.0034e-01f, -4.6858e-01f, -1.1722e-01f, 

        7.1051e-02f, 1.5741e-01f, 2.0840e-01f, -5.2254e-01f, -3.5045e-01f, 1.3262e-01f, -8.4256e-01f, -3.0357e-01f, 

        2.8357e-01f, 8.5574e-01f, 5.4357e-01f, -1.2551e-02f, 8.4003e-03f, -1.3459e-01f, -2.8920e-01f, -1.9020e-01f, 

        2.3754e-01f, -1.4680e-01f, -3.5213e-01f, 6.5129e-01f, 9.2850e-01f, -9.1695e-02f, -3.8348e-01f, -1.1584e-01f, 

        -2.1778e-01f, -3.0262e-01f, 8.3577e-01f, -9.6249e-02f, 1.0275e-01f, -9.2395e-01f, 4.7967e-01f, 1.2464e-01f, 

        8.0549e-02f, -5.0716e-02f, 7.9189e-01f, -9.1717e-02f, 3.1497e-01f, 5.7981e-01f, -1.1649e-01f, 3.1089e-01f, 

        -3.3603e-01f, 3.4772e-02f, -1.0890e-02f, -6.5793e-02f, 3.1227e-01f, 1.2940e-01f, -6.1009e-01f, -1.8270e-01f, 

        4.6629e-01f, 2.9962e-01f, 4.8000e-01f, 1.2887e-02f, 3.4877e-01f, 9.4496e-02f, 3.1602e-02f, 3.2265e-01f, 

        2.3815e-01f, 3.6099e-01f, 4.7288e-01f, -1.4391e-01f, 5.1477e-01f, 4.8379e-02f, 3.6528e-01f, 2.4244e-01f, 

        -1.7526e-01f, 1.0620e-01f, -1.9383e-01f, -3.6697e-01f, -3.3979e-01f, 6.2624e-01f, 2.6360e-01f, 2.0130e-01f, 

        -5.9890e-01f, 8.1364e-01f, 6.0663e-01f, 1.4736e-02f, -4.9720e-01f, 2.8313e-01f, 2.9190e-01f, -2.9383e-01f, 

        -2.2085e-01f, 2.0706e-01f, 5.9849e-01f, -1.1647e-01f, -2.4087e-01f, 2.1642e-01f, 3.2735e-01f, -4.3923e-01f, 

        3.7882e-01f, -1.0518e-02f, -1.1726e-01f, -1.4087e-01f, -1.3436e-01f, -4.4409e-02f, -5.4247e-02f, -1.1156e-01f, 

        -4.9774e-02f, -1.3739e-01f, -3.7318e-02f, -1.0965e-01f, -1.4304e-01f, -7.5287e-02f, -1.6863e-01f, -3.0540e-02f, 

        3.5332e-02f, -1.2691e-01f, 3.6273e-02f, -1.6025e-01f, -1.1201e-02f, -1.6455e-01f, -1.5039e-01f, -4.1915e-02f, 

        -1.4001e-01f, -3.2652e-02f, 1.3186e-02f, -9.6872e-02f, -9.9629e-03f, -9.1171e-02f, -8.3243e-02f, -1.4808e-01f, 

        -9.4611e-02f, -9.4486e-02f, -1.5322e-01f, -9.7812e-02f, -1.6651e-01f, -1.7656e-01f, -5.8038e-02f, -1.7930e-01f, 

        -1.9330e-02f, -9.9072e-02f, -1.0045e-01f, -1.2412e-01f, -1.1038e-01f, -9.6133e-02f, -1.5167e-01f, -1.4131e-01f, 

        -1.8617e-02f, -1.4260e-01f, 4.4945e-03f, -9.5334e-02f, -1.5341e-01f, -7.8829e-02f, -1.0721e-01f, -7.7932e-02f, 

        -8.7202e-02f, -1.2234e-01f, -4.4659e-02f, -1.3501e-01f, -3.8485e-02f, -8.3434e-02f, -1.1564e-01f, -8.5141e-02f, 

        -1.4169e-01f, -1.2789e-01f, -1.3137e-01f, -1.6417e-01f, -2.1420e-02f, -1.1353e-01f, 5.4788e-02f, -1.0146e-01f, 

        -9.7758e-02f, -1.3475e-01f, -1.1220e-01f, -7.3655e-02f, -9.8413e-02f, -9.8906e-02f, -4.4014e-02f, -1.5087e-01f, 

        1.1859e-02f, -8.3677e-02f, -9.0898e-02f, -3.8446e-02f, -1.4954e-01f, -6.9917e-02f, -5.7957e-02f, -1.4130e-01f, 

        -8.6336e-03f, -1.0884e-01f, -4.5821e-02f, -1.4744e-01f, 1.5709e-01f, -2.1598e-03f, 7.5663e-01f, -2.9685e-02f, 

        4.2973e-01f, -6.1574e-02f, -1.0589e+00f, 1.3073e-01f, -7.7683e-01f, -2.2001e-01f, -3.6792e-01f, -3.8152e-01f, 

        3.0682e-01f, -2.4086e-01f, -4.7978e-01f, 1.4988e-01f, 1.8620e-01f, -1.8165e-01f, -6.5348e-01f, -3.6806e-01f, 

        -6.6916e-01f, -9.1512e-02f, 1.0516e+00f, -4.3905e-01f, -1.0077e-01f, -2.6029e-01f, 6.9103e-02f, -2.1023e+00f, 

        -1.8155e-01f, -1.7990e-01f, -3.6415e-01f, 2.5951e-01f, 5.0930e-01f, -1.1537e-03f, 4.9009e-02f, 6.1558e-01f, 

        8.8956e-01f, 1.2856e+00f, -5.7906e-01f, -4.2069e-01f, 3.7097e-01f, -2.5529e-01f, 1.3538e-01f, 3.2843e-01f, 

        1.7679e-03f, 1.3194e-01f, -5.2847e-01f, -7.6138e-01f, -5.0942e-01f, 1.0292e-01f, 5.0369e-01f, 2.3128e-02f, 

        4.7250e-01f, -1.1273e+00f, 7.3016e-03f, 9.2470e-02f, -4.2849e-01f, -1.5857e+00f, 1.1484e-02f, 1.7270e-01f, 

        3.1025e-01f, 1.1471e-01f, -2.8327e-01f, -7.2321e-01f, -4.2820e-02f, 4.3843e-01f, 2.6844e-01f, 3.1301e-01f, 

        -5.7816e-01f, 1.4856e-01f, -2.8769e-02f, -9.5702e-02f, 5.0658e-01f, -4.9975e-01f, 3.5030e-01f, -1.2289e+00f, 

        -9.8931e-01f, -1.2957e-01f, 7.5091e-01f, -3.2625e-01f, -2.9677e-01f, 1.9821e-01f, 1.5776e-01f, -5.1067e-01f, 

        -6.5120e-01f, 2.4061e-01f, -9.1047e-02f, -8.3005e-01f, 4.5810e-02f, 3.4209e-01f, 1.5882e+00f, -3.0489e+00f, 

        -1.0901e+00f, 1.3813e-01f, 8.2986e-01f, 1.3791e-02f, -3.4036e+00f, 3.8404e+00f, -5.3735e-01f, -1.3147e+00f, 

        -4.8576e-01f, -2.6372e+00f, 7.4298e-01f, 1.1005e+00f, -1.1853e+00f, -1.0109e-01f, -9.0921e-01f, 1.6590e-01f, 

        5.1093e-01f, 2.0023e+00f, -4.7708e-01f, -2.4974e+00f, 6.8484e-01f, -1.1722e+00f, 4.7826e-01f, 2.2915e+00f, 

        8.5395e-01f, 4.7790e-02f, -8.8646e-01f, -2.0334e+00f, -4.2508e-02f, -5.9316e-01f, 1.8327e+00f, 1.7604e+00f, 

        4.1269e-01f, -1.1605e+00f, 6.1613e-01f, -1.2155e+00f, -3.6724e-01f, 6.0641e-02f, -1.8366e+00f, -1.8089e-01f, 

        -1.6449e+00f, 6.0058e-01f, -2.9718e-01f, 6.1109e-01f, 1.3169e+00f, 1.2402e+00f, -1.6014e+00f, 1.2631e-01f, 

        -1.6260e-01f, 7.9133e-01f, -3.7724e-01f, -1.0936e+00f, 1.7292e-01f, 2.1618e+00f, -1.7735e+00f, -1.7819e+00f, 

        -8.5951e-01f, 4.7270e-01f, -7.9959e-03f, -7.0589e-01f, 2.4254e+00f, 1.4470e+00f, 4.5424e-02f, -5.5318e-01f, 

        -7.2780e-01f, -1.2244e-01f, 4.4172e-01f, -6.1854e-01f, -6.5892e-02f, 1.6837e+00f, 4.3140e-02f, 7.2322e-01f, 

        8.2217e-01f, -8.8429e-01f, -7.5454e-01f, 7.1793e-01f, 1.4496e+00f, -7.4785e-01f, -5.4290e-01f, -8.5515e-01f, 

        1.5766e-01f, -2.2049e+00f, 2.1085e-01f, 1.1836e+00f, 3.3878e+00f, -3.8356e+00f, -1.0326e+00f, 5.6273e-01f, 

        -1.3426e-01f, -7.5559e-02f, -9.1860e-02f, -2.4917e-02f, -6.5545e-02f, -1.3999e-01f, -7.5034e-03f, -5.5846e-02f, 

        4.9435e-02f, -1.1580e-01f, -1.2513e-01f, -1.1730e-01f, -1.3586e-01f, 4.6054e-02f, 2.7031e-02f, -1.0555e-01f, 

        1.0745e-01f, -1.0733e-01f, 4.8146e-02f, -1.1031e-01f, -9.6250e-02f, 7.4778e-02f, -1.3092e-01f, 5.3276e-02f, 

        -4.8718e-02f, -8.8356e-02f, 9.8269e-02f, -9.1143e-02f, 4.1404e-02f, -1.3274e-01f, -7.7642e-02f, -6.4606e-02f, 

        -9.6904e-02f, -2.2058e-02f, -1.0739e-01f, -1.2690e-01f, -1.9110e-02f, -1.3972e-01f, 5.8940e-02f, -1.2537e-01f, 

        -1.1447e-01f, -9.4405e-02f, -1.0158e-01f, 3.8594e-02f, -1.0688e-01f, -1.1319e-01f, 4.2969e-02f, -1.5627e-01f, 

        5.6149e-02f, -7.3323e-02f, -9.5285e-02f, 5.8806e-02f, -8.0297e-02f, -4.7310e-02f, -1.1379e-01f, -1.3144e-01f, 

        1.4121e-02f, -9.9647e-02f, 5.9114e-02f, -8.9463e-02f, -4.9081e-02f, -1.7179e-02f, -5.1416e-02f, -4.2628e-02f, 

        -1.2977e-01f, -1.3196e-01f, 4.3456e-02f, -7.0586e-02f, 6.0181e-02f, -2.6110e-02f, -8.4120e-02f, -1.0898e-01f, 

        -1.0605e-01f, 2.6896e-03f, -1.2484e-01f, -8.2061e-02f, -6.4616e-03f, -1.2058e-01f, 7.2704e-02f, -7.6105e-02f, 

        -8.2851e-02f, -7.2983e-02f, -1.0731e-01f, -7.5571e-02f, -1.0083e-01f, -1.5754e-01f, -4.2192e-03f, -1.3206e-01f, 

        2.5205e-02f, -1.3141e-01f, -9.4775e-02f, -8.0248e-01f, 1.4587e-01f, 4.3693e-01f, 1.2169e+00f, -3.3848e-01f, 

        -1.3346e+00f, -1.5600e+00f, 1.4180e+00f, -2.9090e-01f, 6.8824e-01f, 1.5701e+00f, -3.1234e-01f, 4.8792e-01f, 

        -4.3613e-01f, 1.0497e+00f, -4.5136e-01f, -1.7812e+00f, 8.1403e-01f, 6.5802e-01f, -1.2972e+00f, 1.0520e-01f, 

        1.3947e+00f, -6.4567e-01f, -1.0465e+00f, 1.0375e+00f, -2.5434e-01f, -1.6619e+00f, -1.5356e+00f, -5.6866e-01f, 

        8.4508e-01f, 3.2439e-01f, -9.1445e-02f, 8.3158e-01f, 5.4682e-01f, 5.1541e-01f, 1.1496e+00f, 1.1314e+00f, 

        5.6617e-01f, -2.6101e-01f, 1.3506e+00f, -3.6509e-01f, 1.2271e+00f, -3.9040e-01f, 8.3178e-01f, 1.7965e-01f, 

        5.1817e-02f, 5.3348e-01f, -1.5028e+00f, -2.4223e-01f, -5.0624e-01f, -1.1025e+00f, -3.4202e-01f, -5.9313e-01f, 

        1.1797e+00f, 1.1393e+00f, -1.1670e+00f, -1.8495e+00f, 1.4930e+00f, -1.5330e+00f, 1.0722e+00f, -3.2465e-01f, 

        6.0792e-01f, 1.1404e-01f, -7.5070e-01f, 2.9087e-01f, 4.1225e-01f, 9.7609e-01f, -7.8053e-01f, -2.5192e-01f, 

        -2.6481e-01f, -6.4236e-01f, 1.7509e+00f, -4.6144e-01f, -1.9146e+00f, 1.8056e+00f, -8.5606e-01f, -1.2903e-01f, 

        -6.4832e-01f, -1.5248e+00f, 1.1390e+00f, -1.7127e-01f, 3.5230e-01f, -1.2024e+00f, 2.1306e-01f, 1.7313e+00f, 

        4.6557e-01f, 4.5539e-01f, -3.8901e-01f, -8.0875e-01f, -3.1675e-01f, 1.5167e-01f, 3.4512e-01f, 2.3474e-02f, 

        3.2971e-01f, 9.8844e-02f, -1.5457e-01f, 1.1794e+00f, 2.8125e-01f, -1.4262e+00f, -7.9742e-01f, 4.2908e-01f, 

        -2.0876e-01f, -8.3626e-02f, -2.7471e-01f, 1.4215e-01f, -2.2215e-01f, 2.7462e-01f, -1.2413e-01f, 1.5902e-01f, 

        5.3244e-01f, -6.7295e-02f, -1.2836e-01f, -7.0019e-01f, -7.0650e-01f, 2.5921e-01f, 5.7311e-01f, 3.6922e-01f, 

        -5.2623e-01f, 3.0940e-01f, -1.0345e+00f, 7.5710e-01f, -1.6827e-01f, 4.6163e-01f, 7.1822e-01f, 5.2363e-01f, 

        -1.9618e-01f, -1.4520e-02f, -5.8024e-01f, -4.4812e-01f, -3.6497e-01f, 8.6040e-01f, -4.1221e-02f, -1.0020e+00f, 

        -1.9551e-01f, -1.4333e-02f, 3.9011e-01f, 8.1126e-01f, -5.5236e-02f, -7.5359e-01f, 5.5319e-01f, 3.7837e-01f, 

        -3.0844e-01f, -7.2998e-02f, -1.0742e-01f, 2.0077e-02f, -4.0974e-02f, -1.0191e-01f, 2.4663e-01f, -3.3398e-01f, 

        4.9797e-01f, 3.3546e-01f, 1.8468e-01f, -1.3593e-01f, -1.6040e+00f, -5.1340e-01f, 2.0712e-01f, 2.1065e-01f, 

        -4.2490e-02f, 8.1763e-01f, 6.8599e-01f, 3.9206e-01f, 7.9818e-03f, -7.7605e-01f, -1.3941e+00f, 3.6425e-01f, 

        2.0673e-01f, -2.1813e-01f, -1.0705e-01f, 1.7175e-01f, 6.0979e-01f, 3.4696e-01f, -5.8333e-02f, -5.4667e-01f, 

        -7.7345e-02f, 6.9678e-01f, -3.3261e-01f, -3.7153e-01f, -5.8107e-01f, -8.1894e-01f, -3.1210e-01f, -7.2176e-01f, 

        -7.0690e-01f, -3.3766e-01f, 3.1458e-01f, 2.9351e-01f, -1.9429e-01f, -3.1225e-01f, 1.2250e+00f, 7.2410e-01f, 

        3.1471e-01f, -1.8637e-02f, -4.4548e-01f, 8.2081e-01f, -2.2821e-02f, -4.1432e-01f, 1.1248e+00f, 7.9213e-01f, 

        6.0013e-01f, 1.2501e-01f, 6.8767e-01f, -6.0639e-02f, -1.5331e-01f, 1.2155e+00f, -2.6749e-01f, 6.5188e-01f, 

        -4.3389e-01f, -1.2602e+00f, -1.5780e+00f, -6.8382e-01f, 2.7388e-01f, 8.4135e-01f, -3.4423e-01f, 3.0719e-01f, 

        -1.5260e-01f, -5.5202e-01f, 4.3226e-02f, -4.1614e-02f, 2.1068e-01f, -9.4283e-01f, 5.4132e-01f, 7.1227e-01f, 

        -8.8902e-01f, 5.6423e-01f, 2.7313e-01f, 5.5084e-02f, -1.9469e-01f, -1.2111e-01f, -3.0631e-02f, -1.7892e-01f, 

        3.7894e-01f, 3.6892e-01f, -1.7024e-01f, 1.2032e+00f, -6.2129e-01f, 1.7549e-01f, -1.0467e+00f, -2.2899e+00f, 

        -1.1300e+00f, 7.2414e-02f, -4.4412e-01f, 3.8404e-01f, 6.6199e-01f, 8.0116e-01f, -5.7825e-01f, -6.1086e-01f, 

        4.5379e-01f, 7.7805e-02f, 3.7547e-01f, -1.0660e-01f, 2.1881e-01f, -5.0718e-01f, 7.6287e-01f, 4.0179e-01f, 

        -1.0909e+00f, -1.1203e+00f, 3.8332e-01f, 2.3350e+00f, -1.6893e+00f, -8.2053e-01f, 6.1467e-01f, 2.4407e-01f, 

        -4.6807e-01f, 3.6263e-01f, -5.0772e-01f, -3.7260e-01f, -1.9510e+00f, -1.7461e+00f, 7.6373e-01f, -2.2359e-01f, 

        6.5943e-01f, 3.5772e-01f, 3.7293e-02f, 3.9616e-01f, 1.8519e-01f, 1.3597e-01f, -1.5008e-01f, 3.3550e-01f, 

        -1.4949e-02f, 4.1568e-01f, 4.5957e-01f, 8.7217e-02f, 6.9638e-02f, 2.0568e-01f, -1.2073e-01f, -1.9443e-01f, 

        -1.4696e-01f, -9.1761e-02f, 2.5129e-01f, -3.3839e-01f, -3.0271e-01f, 3.3234e-02f, 3.8961e-02f, -3.6937e-01f, 

        1.6981e-02f, -2.4108e-01f, 3.7395e-01f, 4.8860e-01f, -5.1170e-01f, -2.8308e-01f, -1.4860e-02f, 1.3285e-01f, 

        3.4849e-02f, -5.4159e-03f, 1.7720e-01f, 9.6064e-02f, -7.6343e-02f, 3.5989e-02f, 3.4565e-01f, 2.9744e-01f, 

        8.3511e-01f, 2.2907e-01f, 5.9814e-02f, 1.3727e-01f, 3.5368e-01f, -2.5061e-02f, 1.8902e-01f, 4.2387e-01f, 

        2.0614e-01f, 5.8195e-02f, -1.8636e-01f, 5.3661e-02f, 4.0686e-02f, 3.4187e-02f, 2.0468e-03f, -2.3245e-02f, 

        6.8996e-01f, 1.4345e-01f, 3.2852e-01f, -1.2277e-01f, 6.1229e-02f, 9.3900e-02f, 2.7300e-01f, -3.4809e-01f, 

        2.2749e-01f, -2.9531e-02f, 2.8523e-01f, 2.5336e-01f, -1.4626e-01f, 2.2105e-01f, -2.3486e-01f, 5.8388e-02f, 

        4.4083e-01f, -3.6500e-01f, 2.2964e-01f, -3.4282e-01f, -1.7636e-02f, 3.3340e-01f, 4.7059e-01f, -2.7961e-01f, 

        5.5535e-02f, -9.0007e-02f, 1.7929e-01f, -1.4631e-01f, -7.5858e-02f, 8.5582e-02f, -5.7725e-02f, -2.3965e-01f, 

        2.0384e-01f, -1.4910e-01f, -6.9151e-02f, -7.1775e-02f, -8.3094e-02f, -5.3235e-02f, -7.7752e-02f, -7.2107e-02f, 

        -4.3278e-02f, -1.4019e-01f, -1.5680e-02f, -1.0490e-01f, -9.4021e-02f, -9.9476e-02f, -1.1848e-01f, -6.7575e-02f, 

        -1.2370e-01f, -7.2066e-02f, 3.9621e-02f, -9.5933e-02f, 4.0991e-02f, -7.9456e-02f, -1.1446e-01f, -1.1146e-01f, 

        -1.0419e-01f, -6.3686e-02f, -1.2863e-01f, -1.0634e-01f, -3.5693e-02f, -1.0516e-01f, 1.5821e-02f, -8.7420e-02f, 

        -3.9791e-02f, -9.4845e-02f, -9.1506e-02f, -9.7352e-02f, -6.7263e-02f, -1.1392e-01f, -6.2194e-02f, -1.0154e-01f, 

        6.3307e-03f, -8.3384e-02f, -5.0198e-02f, -1.1924e-01f, -1.2844e-01f, -6.6369e-02f, -8.7938e-02f, -1.4094e-01f, 

        1.4469e-02f, -7.5734e-02f, 2.6899e-02f, -7.7660e-02f, -7.1555e-02f, -8.9019e-02f, -1.7057e-01f, -9.9330e-02f, 

        -1.2282e-01f, -9.6517e-02f, -4.6200e-02f, -9.4808e-02f, -1.7149e-02f, -1.2068e-01f, -2.7606e-02f, -1.1762e-01f, 

        -8.9826e-02f, -2.1439e-02f, -1.2479e-01f, -9.6707e-02f, -3.6909e-02f, -1.4043e-01f, -1.5490e-02f, -1.1162e-01f, 

        -7.1345e-02f, -1.1675e-01f, -1.5730e-01f, -6.9014e-02f, -8.0601e-02f, -9.0147e-02f, -7.0091e-02f, -1.3513e-01f, 

        -7.4519e-02f, -1.3206e-01f, -6.0286e-02f, -8.7401e-02f, -8.1536e-02f, -9.2676e-03f, -3.9737e-02f, -8.4738e-02f, 

        -1.3894e-02f, -1.5921e-01f, -9.4155e-02f, -1.4644e-01f, 1.4426e-01f, 5.5736e-01f, -2.8440e-02f, 3.5158e-01f, 

        3.3287e-01f, 1.2961e-01f, 3.3301e-01f, -2.5859e-02f, -9.2291e-02f, 2.8618e-01f, 2.6222e-01f, 8.0032e-01f, 

        6.1901e-01f, 1.8297e-01f, 5.7472e-01f, -3.6774e-02f, 1.1211e-01f, -9.9303e-02f, 1.5905e-01f, 1.4326e-01f, 

        2.3749e-01f, 4.0532e-01f, -1.6806e-01f, -1.2522e-01f, 1.4916e-01f, 7.0391e-02f, -2.7447e-01f, 6.8380e-01f, 

        -6.7607e-01f, -2.5649e-01f, 5.3402e-01f, 8.2739e-01f, 2.8024e-01f, 9.2524e-02f, -1.9535e-01f, 1.5522e-02f, 

        -9.5650e-02f, -2.7969e-02f, 2.3502e-01f, -6.1908e-02f, -2.0651e-01f, 4.4134e-01f, -1.2401e-01f, 4.4421e-01f, 

        4.7786e-01f, -1.9627e-01f, 5.3264e-01f, 2.0173e-01f, 3.7831e-02f, 5.1548e-01f, 2.7966e-01f, 2.4035e-01f, 

        2.7435e-01f, 4.2230e-01f, 1.1826e-01f, 3.0530e-01f, 1.7825e-01f, 3.9582e-01f, -1.2181e+00f, 5.7252e-01f, 

        -4.4886e-01f, 3.1932e-01f, 1.1889e-01f, 2.3886e-01f, -1.2091e-02f, -7.9214e-02f, 2.8298e-01f, -5.1706e-02f, 

        -1.8250e-01f, 2.8916e-02f, 1.2647e-01f, 3.1629e-02f, -5.3449e-02f, -4.8924e-02f, 1.0621e-01f, -6.9898e-02f, 

        1.4382e-01f, 4.5038e-01f, 3.7800e-02f, 3.0152e-01f, -4.6893e-01f, 2.2938e-01f, -1.6958e-01f, -1.5274e-01f, 

        3.2058e-01f, 6.2384e-03f, 9.5268e-02f, 6.5502e-01f, 4.9060e-01f, 5.1697e-02f, 5.2169e-01f, 1.0452e-01f, 

        1.0772e+00f, 1.0229e-01f, -2.8101e-02f, 4.2542e-01f, 7.0978e-02f, 1.9661e-01f, -1.0748e-01f, 1.1586e+00f, 

        4.2592e-01f, -3.0927e-01f, -4.1335e-01f, 1.7944e-01f, -2.5784e-01f, -5.8486e-01f, -1.8095e-01f, -9.7912e-01f, 

        7.6902e-01f, 1.3807e-02f, -1.8762e-01f, 1.2569e-01f, -1.3905e-02f, 4.6426e-01f, 1.3434e-01f, -2.0664e-01f, 

        -2.2215e-03f, -1.2547e+00f, 7.0547e-01f, -3.0735e-01f, -2.4969e-01f, -9.1282e-01f, -3.6522e-01f, -3.7659e-01f, 

        -5.4739e-01f, -7.7578e-01f, -9.1418e-01f, 1.3943e+00f, -1.2735e-01f, -1.3228e+00f, 5.2077e-01f, 7.5446e-01f, 

        -4.0387e-01f, 5.7633e-01f, 1.5358e-01f, -5.5446e-01f, 9.4685e-01f, 9.9522e-01f, 9.0326e-02f, 3.5557e-02f, 

        -3.3684e-01f, -5.8153e-01f, -3.3589e-01f, -4.7968e-01f, -4.0498e-01f, 1.8170e-01f, -9.7414e-01f, 1.1212e-01f, 

        -2.9420e-01f, 1.1427e+00f, -4.0662e-01f, 1.7186e-01f, -5.5449e-01f, -1.6743e+00f, -3.7280e-01f, 4.5696e-01f, 

        9.5145e-01f, -3.6939e-01f, -3.2675e-01f, 3.8001e-02f, 7.8877e-01f, 2.0146e-01f, -5.8952e-01f, -2.5905e-01f, 

        1.4210e-01f, -6.4369e-01f, 4.8498e-01f, 1.7159e-01f, 5.5645e-01f, -3.4628e-01f, -6.2187e-01f, -3.4653e-02f, 

        -5.2317e-02f, 1.5305e-01f, -9.5089e-02f, -7.6980e-01f, -2.6034e-01f, 7.7196e-02f, 4.5768e-02f, 1.7197e-01f, 

        -5.5550e-02f, -1.0367e-01f, -3.4171e-04f, -4.7794e-02f, -3.6619e-02f, -8.6762e-02f, -5.9369e-02f, -3.7477e-02f, 

        -2.9371e-02f, -5.9942e-02f, -6.4974e-02f, -4.5435e-02f, -9.0372e-02f, -2.2326e-02f, 8.5856e-02f, -8.7488e-02f, 

        -7.1997e-03f, -9.9239e-02f, -5.3341e-02f, -5.9804e-02f, -6.1975e-02f, 7.8030e-02f, -8.6129e-02f, -1.7063e-02f, 

        -6.7633e-02f, -1.4296e-01f, 7.0414e-02f, -9.3730e-02f, -6.9510e-02f, -1.0701e-01f, -9.0084e-02f, -6.5741e-02f, 

        -5.8657e-02f, -1.9710e-02f, -7.7487e-02f, -1.0633e-01f, -5.9961e-02f, -1.3071e-01f, 1.8638e-03f, -9.6757e-02f, 

        4.1517e-03f, -7.0218e-02f, -7.1559e-02f, -8.1443e-03f, -1.1687e-01f, -6.6597e-02f, -5.5243e-02f, -1.2436e-01f, 

        1.3212e-02f, -7.6769e-03f, -5.0253e-02f, -2.9072e-02f, -2.4289e-02f, -8.8193e-02f, -9.4065e-02f, -9.2471e-02f, 

        -4.0873e-02f, -1.3416e-02f, -3.2275e-02f, -2.9600e-02f, -1.4425e-01f, -1.1551e-01f, -1.0557e-01f, -8.0915e-02f, 

        -1.4022e-01f, -1.4820e-01f, -1.4171e-01f, -1.3868e-01f, -5.0391e-02f, -7.9996e-02f, -5.6655e-02f, -8.7753e-02f, 

        -1.2289e-01f, -1.1368e-01f, -1.3815e-01f, -9.5298e-02f, -1.1200e-01f, -6.7793e-02f, -3.6179e-02f, -7.2062e-02f, 

        -5.1173e-02f, -4.9919e-02f, -8.7527e-02f, -1.0356e-01f, -5.7235e-02f, -9.5981e-02f, -8.4660e-02f, 2.1275e-02f, 

        -6.8203e-02f, -1.3058e-01f, -3.4529e-01f, 2.9642e-01f, 3.6857e-01f, -1.0661e+00f, 3.6914e-01f, -7.6451e-02f, 

        -5.2190e-02f, 3.0334e-01f, 1.7776e-01f, 6.0855e-01f, 4.0011e-01f, 3.7273e-01f, 4.1986e-02f, 1.8772e-01f, 

        -1.6402e-01f, 6.9914e-02f, -3.8246e-01f, 6.4337e-01f, 4.9042e-01f, 9.4064e-01f, 4.1146e-02f, -3.4808e-02f, 

        2.6341e-01f, -2.7049e-01f, 3.2331e-01f, -1.1183e-01f, 5.1293e-01f, 1.3043e-01f, -1.8167e-01f, -1.3317e-01f, 

        4.3898e-01f, 5.5021e-02f, 5.5619e-01f, -2.5548e-01f, 2.0647e-01f, 1.6620e-01f, -1.6135e-01f, 4.4500e-01f, 

        7.5362e-01f, 1.0313e-01f, 1.5895e-01f, 1.7391e-01f, 1.1161e-01f, 2.8831e-01f, 1.5437e-01f, 2.4182e-01f, 

        8.2649e-01f, 4.2624e-01f, 3.2886e-01f, -2.6773e-02f, 6.2925e-01f, 1.3852e-01f, 1.1587e-01f, 4.3856e-01f, 

        1.6553e-01f, 2.7287e-01f, 2.7977e-01f, 1.7145e-01f, 1.4316e-01f, -1.5631e-01f, -5.7203e-01f, 2.1847e-01f, 

        2.7630e-01f, -2.3579e-01f, 3.6512e-03f, 5.5988e-01f, 4.7900e-01f, -1.8007e-01f, 4.7197e-01f, 7.4463e-02f, 

        2.9076e-01f, 3.7734e-02f, 3.0636e-02f, 5.6270e-01f, 2.3768e-01f, -1.0077e-01f, 1.9090e-01f, 4.6805e-01f, 

        3.2095e-01f, -2.4039e-01f, 3.6041e-01f, 2.5268e-01f, 1.9772e-01f, 3.6175e-01f, 5.8198e-02f, 1.8157e-01f, 

        4.0436e-01f, 6.6715e-02f, 8.5824e-02f, 2.9472e-01f, -9.6242e-02f, -8.2669e-02f, -1.0434e-01f, -7.6610e-02f, 

        -1.1645e-01f, -1.0790e-01f, -5.8334e-02f, -1.1378e-01f, -2.7770e-02f, -1.0337e-01f, -1.4460e-01f, -9.6819e-02f, 

        -1.7350e-01f, 2.6643e-02f, -3.7516e-02f, -9.4049e-02f, 3.0277e-02f, -1.7069e-01f, -9.4391e-04f, -1.2606e-01f, 

        -1.2499e-01f, 9.0115e-02f, -1.2669e-01f, -5.6943e-02f, -1.6433e-01f, -1.1942e-01f, 5.6561e-02f, -1.3816e-01f, 

        -4.0431e-02f, -1.7389e-01f, -1.1482e-01f, -9.9447e-02f, -9.0391e-02f, -4.0040e-02f, -1.1911e-01f, -1.4449e-01f, 

        -3.8045e-02f, -9.6865e-02f, -3.5504e-02f, -1.2910e-01f, -8.5826e-02f, -1.2302e-01f, -1.3271e-01f, -1.3011e-01f, 

        -1.2575e-01f, -8.6699e-02f, -1.1619e-01f, -1.6019e-01f, 4.9640e-02f, -1.5388e-01f, -1.2311e-01f, 4.2706e-02f, 

        -1.3238e-01f, -3.7092e-02f, -1.1939e-01f, -1.3237e-01f, -3.9197e-02f, -9.7611e-02f, -6.4635e-02f, -1.3614e-01f, 

        -6.3915e-02f, -6.2269e-02f, -5.4503e-02f, -6.4384e-02f, -7.7884e-02f, -1.1270e-01f, 2.4189e-02f, -1.4495e-01f, 

        -1.1399e-02f, -8.5697e-02f, -1.3280e-01f, -1.1270e-01f, -8.9718e-02f, -9.0015e-02f, -1.0567e-01f, -1.2052e-01f, 

        -1.2647e-01f, -1.3352e-01f, -6.3207e-03f, -1.1493e-01f, -6.4475e-02f, -8.1051e-02f, -8.5130e-02f, -1.0991e-01f, 

        -6.5884e-02f, -1.4155e-01f, -3.1502e-02f, -1.3643e-01f, -9.5243e-02f, -9.8473e-02f, -1.0402e-01f, -1.2305e-01f, 

        -1.1099e-01f, 1.2305e-03f, -4.9905e-02f, -8.9525e-02f, 1.5386e-02f, -1.2872e-01f, 1.9896e-02f, -7.7996e-02f, 

        -6.2581e-02f, -4.1268e-02f, -7.4440e-02f, 2.0017e-02f, -5.0639e-02f, -1.0036e-01f, 4.9383e-02f, -1.1569e-01f, 

        -4.6152e-02f, -1.0819e-01f, -9.0873e-02f, 6.3170e-02f, -9.3613e-02f, -4.4162e-03f, -1.1785e-01f, -8.8314e-02f, 

        8.5860e-02f, -1.4077e-01f, 9.5363e-03f, -7.9409e-02f, -1.9318e-02f, -4.3061e-02f, -1.0494e-01f, 9.1299e-03f, 

        -1.1046e-01f, -1.0899e-01f, 5.4705e-02f, -1.3252e-01f, 6.2682e-02f, -1.1072e-01f, -1.4525e-01f, -1.0401e-01f, 

        -9.2283e-02f, -4.2076e-02f, -1.3889e-01f, -1.6893e-01f, 1.0061e-02f, -1.1341e-01f, 5.8967e-02f, -1.2654e-01f, 

        -1.3056e-01f, 5.9631e-02f, -1.1832e-01f, -3.1020e-02f, -1.0035e-01f, -1.1620e-01f, 3.0620e-03f, -1.2677e-01f, 

        2.9323e-02f, -1.6586e-01f, -1.4083e-01f, -1.0864e-01f, -9.9663e-02f, -6.7232e-02f, -9.2837e-02f, -1.3107e-01f, 

        4.0227e-02f, -1.2924e-01f, -2.8798e-03f, -1.3898e-01f, -1.5509e-01f, -1.3366e-01f, -1.5801e-01f, -4.9112e-02f, 

        -1.7381e-01f, -1.0168e-01f, -1.2448e-02f, -9.7266e-02f, 4.6912e-03f, -1.1789e-01f, -1.5295e-01f, -6.8120e-02f, 

        -1.2058e-01f, -2.2746e-02f, -1.2305e-01f, -1.1844e-01f, 2.4669e-02f, -8.5165e-02f, -1.2057e-02f, -9.6561e-02f, 

        7.6028e-03f, -1.2131e-02f, 3.4958e-02f, 5.4859e-02f, 2.4891e-03f, 8.2278e-02f, -4.5468e-02f, 1.9794e-01f, 

        8.6125e-02f, 6.8522e-03f, 2.7853e-02f, -1.3187e-01f, 6.2116e-02f, 4.3168e-02f, -8.4106e-02f, 1.5483e-02f, 

        2.7910e-02f, 2.5475e-02f, 2.4454e-02f, -7.3776e-02f, -2.1674e-02f, 7.8950e-02f, -5.0415e-03f, -3.1515e-02f, 

        4.8091e-02f, 1.9044e-02f, 7.8182e-02f, -1.0727e-01f, -5.7865e-02f, -1.1261e-01f, -2.5580e-02f, -4.8078e-03f, 

        8.3290e-03f, 3.7179e-02f, -5.4117e-02f, 3.9079e-02f, -3.4332e-02f, 1.9386e-01f, 1.2893e-01f, 2.8810e-02f, 

        7.6436e-02f, -2.5254e-02f, 6.8405e-02f, 5.2602e-02f, -8.6596e-02f, 6.9948e-02f, 1.1985e-02f, 1.1594e-02f, 

        1.0587e-01f, 2.0904e-02f, 2.0366e-02f, -1.3958e-02f, -2.5138e-02f, 8.3940e-02f, -9.5194e-02f, -9.9391e-04f, 

        2.4846e-02f, -6.8988e-02f, 1.1211e-01f, -1.1092e-01f, 6.5933e-02f, 5.9597e-02f, 1.2394e-01f, 7.6459e-02f, 

        -2.9877e-04f, -1.4638e-03f, 1.0161e-01f, 2.8491e-01f, 1.4769e-01f, 1.1335e-01f, 9.1073e-02f, -7.2213e-02f, 

        6.2319e-02f, 1.0168e-01f, -7.1382e-02f, -4.2047e-02f, -3.8354e-02f, 6.8635e-02f, 1.9432e-01f, 8.1893e-02f, 

        -1.5339e-02f, 3.8744e-02f, 4.1957e-02f, 1.0622e-01f, 1.9249e-02f, -2.8569e-02f, 4.0690e-02f, -6.6541e-02f, 

        1.6247e-01f, -4.1087e-02f, 7.4414e-03f, 1.8888e-02f, -6.7207e-02f, -2.8629e-02f, -9.0037e-02f, -2.1380e-01f, 

        8.3624e-02f, -1.7061e-01f, 2.0261e-01f, -3.2832e-02f, 2.3679e-01f, -1.7812e-01f, -7.5065e-02f, -1.1203e-01f, 

        -7.0785e-02f, -1.1654e-01f, 3.0020e-01f, -1.2727e-01f, 2.4366e-01f, -1.2216e-01f, 2.7046e-01f, -9.3848e-02f, 

        -3.6331e-02f, 4.5856e-02f, -1.2831e-01f, -1.4044e-01f, 3.6391e-02f, -1.6787e-01f, 1.9713e-01f, -1.5211e-01f, 

        1.2431e-01f, -6.8629e-02f, -1.2247e-01f, -5.8641e-02f, -5.8157e-02f, -9.8552e-02f, 1.8635e-01f, -2.0129e-01f, 

        1.1589e-01f, -1.0430e-01f, 2.3101e-01f, -1.2950e-01f, -1.6771e-01f, 7.4565e-02f, -7.7227e-02f, -1.2215e-01f, 

        1.9648e-01f, -1.2280e-01f, 3.4643e-01f, -7.4715e-02f, 1.3227e-01f, -7.8506e-02f, -2.3713e-02f, -6.4182e-03f, 

        -1.7587e-01f, -1.0034e-01f, 2.4780e-02f, -1.1428e-01f, 1.2781e-01f, -8.8004e-02f, 1.6319e-01f, -1.6514e-01f, 

        -1.3623e-01f, -1.0465e-02f, -1.2170e-01f, -2.0838e-01f, -9.5259e-02f, -1.2422e-01f, 1.4327e-01f, -1.2240e-01f, 

        9.2853e-02f, -1.6515e-01f, -9.6849e-02f, -6.2917e-03f, -2.8006e-02f, -1.4785e-01f, 7.1243e-02f, -2.9501e-02f, 

        1.9457e-01f, -1.2749e-01f, -1.9041e-02f, -8.2911e-02f, 4.2467e-02f, 3.7234e-02f, -8.2684e-02f, 4.1546e-02f, 

        -7.0943e-02f, 7.9637e-02f, 1.6687e-01f, -1.2544e-01f, -5.0149e-02f, -1.3065e-01f, -1.0623e-01f, -3.5705e-02f, 

        -1.3036e-01f, -1.0612e-01f, -4.2267e-02f, -9.1832e-02f, 6.6685e-03f, -7.5950e-02f, -8.3696e-02f, 2.8493e-02f, 

        -1.0831e-01f, -3.9518e-02f, -9.5602e-02f, -1.2471e-01f, 5.3231e-02f, -1.3308e-01f, 2.7520e-02f, -1.0403e-01f, 

        -6.9662e-02f, 7.0789e-02f, -7.7057e-02f, -4.5063e-02f, -1.1256e-01f, -1.0893e-01f, 9.5986e-02f, -1.4685e-01f, 

        3.8396e-02f, -1.0145e-01f, -1.1776e-01f, -1.0262e-01f, -1.5987e-01f, -1.2203e-01f, -1.2656e-01f, -1.1809e-01f, 

        1.3652e-02f, -1.0059e-01f, 7.5510e-02f, -1.1387e-01f, -1.1157e-01f, -1.6258e-01f, -9.4663e-02f, -1.1860e-02f, 

        -8.3688e-02f, -1.3977e-01f, 7.5279e-02f, -1.3229e-01f, 4.6118e-02f, -1.1575e-01f, -1.6661e-01f, -2.4717e-02f, 

        -1.0669e-01f, -9.9464e-03f, -1.4510e-01f, -9.9792e-02f, -1.6925e-02f, -1.1422e-01f, -6.4290e-02f, -1.4116e-01f, 

        -1.3625e-01f, -1.5009e-01f, -1.3209e-01f, -6.0370e-02f, -9.8903e-02f, -1.3190e-01f, 1.8159e-02f, -9.1279e-02f, 

        7.7708e-02f, -1.3069e-01f, -9.7440e-02f, -8.9992e-02f, -1.5794e-01f, -4.9616e-02f, -1.1282e-01f, -1.3593e-01f, 

        -6.6817e-03f, -1.3007e-01f, 3.8598e-02f, -1.7399e-01f, -1.0415e-01f, -1.1817e-01f, -1.0167e-01f, -6.6428e-02f, 

        -1.3911e-01f, -1.6382e-01f, -5.1335e-02f, -8.7193e-02f, -4.2962e-02f, -1.4851e-01f, 3.0698e-02f, -7.9549e-02f, 

        -2.7524e-03f, 1.0765e-01f, 2.7597e-02f, -1.2048e-02f, 4.8128e-02f, -1.9423e-02f, 9.9021e-02f, -4.8148e-02f, 

        -1.0791e-01f, 4.5274e-02f, -1.0719e-01f, 8.7710e-02f, -1.9319e-02f, -6.7438e-02f, 1.3666e-01f, -8.8788e-02f, 

        1.0765e-01f, -7.0122e-02f, -8.8055e-02f, 9.5672e-02f, -9.8429e-02f, 3.3764e-02f, -1.4489e-01f, -1.0525e-01f, 

        6.3378e-02f, -1.4712e-01f, 2.3905e-02f, -1.1438e-01f, -1.1187e-01f, -1.0440e-01f, -1.1720e-01f, 4.6243e-03f, 

        -8.2218e-02f, -1.4049e-01f, 7.1156e-02f, -9.4879e-02f, 6.5886e-02f, -1.2426e-01f, -2.3805e-02f, -1.1592e-01f, 

        -1.0647e-01f, 2.7672e-02f, -1.2318e-01f, -1.1497e-01f, 9.4098e-02f, -5.5506e-02f, 5.8229e-02f, -5.8599e-02f, 

        -8.5332e-02f, 4.0066e-03f, -7.0013e-02f, -1.2206e-02f, -7.6907e-02f, -1.5675e-01f, 2.5662e-02f, -1.2890e-01f, 

        4.9223e-02f, -1.0776e-01f, -8.7373e-03f, -3.0322e-02f, -7.3254e-02f, 2.2922e-02f, -7.8057e-02f, -7.8008e-02f, 

        5.8045e-02f, 5.1348e-03f, 7.7917e-02f, -8.9902e-02f, -6.5855e-02f, -5.2290e-02f, -9.9499e-02f, 1.3001e-03f, 

        -8.5431e-02f, -7.9903e-02f, 4.0584e-03f, -1.1872e-01f, 1.2047e-01f, -4.5085e-02f, -6.6840e-02f, -3.8558e-02f, 

        -9.1073e-02f, -5.1539e-04f, -1.2172e-01f, -1.3966e-01f, 3.7989e-02f, -8.1916e-02f, -2.8190e-02f, -9.3871e-02f, 

        -5.2535e-02f, -1.0866e-01f, -4.4631e-02f, 2.1988e-02f, -6.5270e-02f, -1.3178e-01f, -4.6973e-02f, -9.4957e-02f, 

        5.8837e-02f, -7.1406e-02f, -6.6711e-02f, -1.1673e-01f, -6.5791e-02f, 5.6856e-02f, 2.1598e-02f, -1.2935e-01f, 

        6.6662e-02f, -1.1660e-01f, 4.5512e-02f, -5.1555e-02f, -4.5573e-02f, 9.5734e-02f, -4.9520e-02f, 3.1036e-02f, 

        -4.6785e-02f, -7.7878e-02f, 4.4357e-02f, -7.9522e-02f, 2.1969e-02f, -9.6022e-02f, -8.0862e-02f, -1.2590e-01f, 

        -1.3310e-01f, -6.3511e-02f, -9.3792e-02f, -9.9773e-02f, -1.1179e-01f, -1.0824e-01f, -1.1107e-02f, -5.8037e-02f, 

        1.5440e-02f, -1.1680e-01f, -1.0555e-01f, -5.0272e-02f, -1.0165e-01f, -1.3221e-01f, -5.6590e-02f, -1.3213e-01f, 

        2.9538e-02f, -6.6563e-02f, -8.4677e-02f, -6.8352e-02f, -7.7300e-02f, -9.3335e-02f, -6.3106e-02f, -1.1945e-01f, 

        -7.4750e-02f, -1.4388e-01f, 1.0752e-02f, -7.2029e-02f, -4.6884e-02f, -7.9481e-02f, -9.6843e-02f, -7.8556e-02f, 

        -1.0379e-01f, -8.2795e-02f, -6.5965e-02f, -1.1976e-01f, -1.2617e-02f, -2.5831e-02f, -9.4201e-02f, -1.2393e-01f, 

        -1.0007e-01f, -5.2481e-02f, -1.2171e-01f, -8.3251e-02f, -7.8629e-02f, -9.9180e-02f, 1.9124e-02f, -7.1184e-02f, 

        -7.9403e-02f, -8.5173e-02f, -1.5859e-01f, -1.1955e-01f, -1.2788e-01f, -1.0167e-01f, -6.1150e-02f, -1.2531e-01f, 

        -2.5440e-02f, -7.5683e-02f, 8.3406e-02f, -6.2368e-02f, 4.4320e-02f, 1.1524e-01f, 5.8828e-02f, 1.9369e-01f, 

        -6.5370e-02f, -4.7692e-03f, -8.3944e-02f, 1.4694e-02f, -9.4437e-03f, -9.3841e-02f, 1.7296e-01f, -1.5844e-02f, 

        -1.7312e-02f, 1.8664e-01f, -9.2740e-02f, 7.1790e-02f, -4.7467e-02f, -9.7442e-03f, -1.3900e-02f, 2.5155e-02f, 

        -8.6162e-02f, 1.6030e-01f, -8.6016e-04f, 1.6274e-01f, -1.0848e-01f, 7.8152e-02f, -1.1431e-01f, 2.1025e-01f, 

        -9.3587e-02f, -1.2385e-01f, -2.2999e-02f, -5.4778e-02f, -1.1187e-01f, 2.1704e-01f, -9.5431e-02f, -1.0142e-01f, 

        -1.1270e-01f, -1.0831e-01f, 5.0598e-02f, -3.3272e-02f, 3.6810e-02f, -1.2571e-01f, -6.2777e-02f, 1.5221e-01f, 

        -4.5264e-02f, -6.0715e-02f, 9.8738e-02f, -1.1594e-02f, -1.2228e-01f, -5.9096e-02f, 4.7039e-02f, 1.2943e-01f, 

        -6.6040e-02f, 2.4522e-03f, -1.2147e-01f, -1.3887e-01f, -1.6434e-02f, -3.6566e-02f, -6.8439e-02f, -6.3360e-02f, 

        -5.8803e-02f, -1.0908e-01f, -6.5445e-02f, 1.5146e-01f, -7.0403e-02f, -6.6370e-02f, -9.0732e-04f, 1.1590e-01f, 

        -3.5656e-02f, -4.3737e-02f, 4.1931e-02f, -1.1053e-01f, -1.0366e-01f, -2.4952e-02f, -1.0114e-01f, -9.6460e-02f, 

        -1.1935e-01f, -1.9580e-02f, -6.7827e-02f, -7.8854e-02f, 4.5413e-05f, -3.7000e-03f, -1.0892e-01f, 4.4936e-02f, 

        -1.1275e-01f, -9.9834e-02f, -5.5506e-02f, 6.4559e-02f, -2.0028e-01f, 8.0142e-02f, 9.8609e-01f, -3.7189e-01f, 

        8.5257e-03f, 4.3755e-01f, -2.9077e-01f, -6.1962e-01f, 4.1246e-01f, -4.0663e-01f, -9.8067e-01f, -2.3442e-01f, 

        -8.1808e-01f, -4.5053e-01f, -1.3213e-01f, 1.3759e-01f, -9.2032e-01f, 2.5782e+00f, -6.8224e-01f, -1.5458e-01f, 

        5.3869e-01f, -5.6554e-01f, 1.6132e-01f, -2.1935e-02f, -6.3313e-01f, 1.2239e-01f, -7.9664e-01f, 1.4319e+00f, 

        -1.8900e-01f, 2.6678e-01f, 6.2146e-01f, -1.2796e+00f, 2.6421e+00f, -2.8406e-01f, -7.5287e-01f, -1.2447e+00f, 

        -2.1441e-01f, 6.4752e-01f, 4.5268e-01f, 1.6329e-01f, 6.3002e-01f, -8.3710e-01f, -2.2710e+00f, 1.7100e+00f, 

        -2.0077e-01f, -3.3432e-01f, -8.3526e-01f, -1.5544e-01f, 7.5984e-01f, -5.2911e-02f, 4.2021e-01f, 2.1873e-01f, 

        -1.2678e+00f, 1.1309e-01f, -9.5442e-01f, -1.4719e-01f, -2.1905e-01f, -4.6612e-01f, 2.2022e-01f, -3.4563e-01f, 

        6.0421e-01f, -3.5750e-01f, -4.4852e-01f, -1.8187e+00f, 2.2969e-01f, -5.4306e-01f, 5.6656e-01f, 9.7180e-01f, 

        1.1132e+00f, -5.9948e-01f, 1.4225e+00f, -4.2747e-01f, -2.7777e+00f, 2.4834e-01f, -2.0045e-01f, -1.9311e-01f, 

        -5.1133e-01f, -1.4019e+00f, -9.4201e-01f, -7.8695e-01f, 1.2596e+00f, -1.2800e+00f, -1.0570e+00f, -1.0756e+00f, 

        -1.0919e+00f, -1.6948e+00f, 2.8815e+00f, -1.7437e+00f, -7.9814e-01f, 5.1086e-01f, -3.2189e-02f, -2.1392e-02f, 

        1.1730e-01f, 1.6643e-01f, 9.2506e-02f, 7.2303e-02f, -4.4550e-02f, 2.5531e-01f, 1.6766e-01f, 3.2404e-02f, 

        -4.8532e-02f, 2.9077e-02f, 1.9356e-01f, -6.0888e-02f, 6.1078e-02f, -1.3018e-02f, 1.0420e-01f, 2.0452e-01f, 

        2.3457e-02f, -1.0848e-01f, -1.5311e-02f, 1.8663e-01f, 3.0855e-03f, 2.1850e-02f, 2.6073e-01f, -4.2880e-02f, 

        7.3155e-02f, -7.5494e-02f, -6.8332e-02f, -9.5893e-02f, 5.0569e-02f, 1.8685e-01f, 1.5861e-01f, -7.1401e-03f, 

        6.4374e-02f, 9.3422e-02f, 8.6555e-02f, 1.7037e-01f, 1.8266e-01f, 9.4414e-02f, 1.5630e-01f, 8.7134e-02f, 

        3.1260e-01f, 6.0470e-02f, 7.3920e-04f, 4.0230e-02f, 1.1250e-01f, 2.0557e-02f, -1.2451e-01f, 8.5138e-02f, 

        -1.0243e-01f, 2.7424e-01f, 1.5109e-01f, 3.7066e-01f, 1.0197e-01f, 2.6197e-02f, -3.2957e-01f, 4.1664e-03f, 

        1.5801e-01f, 2.1925e-01f, 4.1566e-01f, 3.2009e-01f, 1.1911e-01f, 8.2161e-02f, -2.4277e-01f, -2.2491e-02f, 

        -2.0994e-01f, 4.6599e-01f, 2.4393e-01f, 3.0365e-01f, 2.4381e-01f, -3.1256e-02f, 3.8083e-01f, 1.3468e-01f, 

        -6.0005e-02f, -4.6304e-02f, -3.9203e-02f, 5.3354e-02f, 2.6554e-01f, 3.2927e-01f, 3.7775e-01f, -3.9041e-02f, 

        9.0092e-02f, 1.8574e-01f, -2.4490e-01f, 2.0420e-01f, -2.5326e-02f, -3.1342e-01f, 3.4455e-01f, -1.1302e-01f, 

        -1.1619e-01f, -5.6715e-02f, 1.1979e-01f, -1.8176e-02f, 4.1014e-02f, 1.0630e-01f, -1.7302e-01f, -4.4310e-02f, 

        -7.8582e-02f, -1.1264e-01f, -9.2388e-02f, 9.8562e-02f, 1.4234e-01f, -2.5642e-02f, -6.8808e-02f, -3.7727e-02f, 

        -1.1837e-02f, 1.1027e-01f, -2.2196e-02f, -5.0616e-02f, -1.7905e-01f, -1.6035e-01f, -2.1615e-01f, -1.1475e-01f, 

        3.1205e-03f, -2.5431e-01f, -3.4447e-02f, 1.3187e-01f, 3.4101e-02f, -1.6980e-01f, -9.1870e-02f, -1.2936e-01f, 

        6.8339e-03f, 9.8902e-02f, -1.5170e-01f, -5.0331e-03f, -1.3324e-01f, -1.1520e-01f, -1.2028e-01f, -1.4986e-01f, 

        -3.3735e-01f, -3.3713e-01f, 1.3224e-01f, -1.4351e-01f, -8.4389e-02f, -1.4818e-02f, -1.0259e-01f, -2.5421e-01f, 

        -4.0027e-01f, -1.9297e-01f, -1.6663e-01f, -7.8992e-02f, -3.4057e-02f, -1.5975e-01f, -9.1175e-03f, -1.1906e-01f, 

        -1.0025e-01f, -8.9142e-02f, -1.3954e-01f, -9.7989e-02f, -2.3590e-01f, -2.4333e-01f, -1.8896e-01f, -4.3560e-02f, 

        -1.8901e-01f, 1.3014e-01f, -1.3931e-01f, -9.0818e-02f, -3.0163e-01f, -4.8315e-02f, -7.0599e-02f, -1.0731e-01f, 

        4.1782e-02f, -1.3701e-01f, -3.7998e-02f, 2.1303e-01f, -1.4295e-01f, -3.2111e-02f, -1.3337e-01f, 5.0270e-02f, 

        -7.9814e-02f, 5.7534e-02f, -2.8614e-01f, -3.3138e-01f, -7.2123e-02f, -8.9605e-03f, 3.4736e-02f, -3.2790e-02f, 

        -1.2925e-01f, -1.9628e-01f, -1.3978e-01f, -1.8641e+00f, 2.3448e-01f, 7.2463e-02f, -4.0950e-02f, -5.8214e-01f, 

        -1.5291e+00f, 3.5885e+00f, 4.7019e-02f, -5.9315e-02f, -1.1551e+00f, -1.2796e+00f, 2.3135e-01f, 6.8090e-01f, 

        3.5092e-01f, -7.9639e-01f, -1.5705e+00f, 1.4738e+00f, -9.2984e-01f, -8.3115e-01f, -5.2985e-01f, -1.0926e+00f, 

        -2.9040e-01f, -1.8356e-01f, 1.1966e+00f, -4.7896e-01f, 1.8182e-01f, -1.0978e+00f, -1.6356e-02f, 3.0715e-02f, 

        4.0481e-01f, 8.7127e-01f, 7.9842e-01f, 9.2989e-01f, 1.2543e+00f, -2.9011e-02f, -2.5477e-01f, -1.1994e+00f, 

        -5.5376e-01f, -1.6786e+00f, -2.2855e-01f, -1.2059e+00f, -7.2313e-02f, 3.3023e-01f, -3.1351e-01f, -3.2512e-01f, 

        -2.1136e+00f, 5.4542e-02f, 1.0075e+00f, 1.7794e+00f, -1.0121e+00f, -1.8543e+00f, 1.4420e-01f, 3.7462e-01f, 

        -2.0484e-01f, -6.2150e-02f, -1.2402e-01f, -4.6931e-01f, -2.0096e-01f, 1.3925e+00f, 5.8292e-02f, 2.4361e-01f, 

        8.3553e-01f, 5.8824e-01f, 3.9250e-01f, 8.8072e-01f, -1.7738e-01f, 4.0702e-01f, -9.9873e-02f, -4.2337e-01f, 

        1.3679e+00f, -1.7197e-01f, -2.6847e-01f, -2.8816e-01f, 7.3984e-01f, 8.3148e-01f, -1.8245e+00f, 6.0653e-01f, 

        -1.3879e+00f, 5.4164e-01f, 6.4944e-01f, -4.4577e-01f, -2.0003e-01f, -1.1710e+00f, -1.4961e+00f, 2.5833e-01f, 

        -5.2550e-01f, -1.3210e+00f, -7.4044e-01f, 2.6241e+00f, -1.5533e-01f, -6.0452e-02f, 1.0499e-01f, -1.1412e-01f, 

        5.7671e-02f, 9.8277e-02f, -7.7986e-03f, -1.4064e-01f, 1.7933e-02f, 1.6305e-01f, -1.2672e-01f, -1.0541e-01f, 

        1.2237e-01f, -3.8180e-02f, 1.6484e-02f, 2.3031e-01f, -1.4802e-01f, -3.2951e-02f, 5.5716e-02f, -8.4267e-02f, 

        -1.0256e-01f, -5.1505e-02f, 4.1518e-02f, -7.4180e-02f, -4.3104e-02f, -1.4901e-02f, -4.1730e-02f, -5.0774e-02f, 

        -5.7713e-02f, -6.3593e-02f, 7.3352e-02f, -1.3552e-03f, -2.1908e-02f, 2.5727e-02f, -4.0662e-02f, 1.2065e-01f, 

        -6.8965e-02f, -4.6331e-02f, -7.2995e-02f, 1.4810e-01f, -2.1504e-01f, -7.7821e-03f, -2.1985e-01f, 9.3988e-02f, 

        2.0186e-01f, 2.4400e-01f, 2.6458e-02f, -1.5222e-01f, 3.3784e-01f, -2.1331e-01f, -1.6793e-01f, -3.3069e-02f, 

        3.3576e-02f, 1.9885e-02f, 7.2262e-02f, -3.8157e-02f, -5.7067e-02f, 9.2742e-03f, -1.7566e-01f, -3.3132e-02f, 

        -6.1382e-03f, -5.6992e-02f, 7.9469e-02f, -1.8804e-01f, -4.0915e-02f, 1.0019e-01f, -8.1656e-02f, 2.9680e-01f, 

        -6.4361e-02f, 1.3747e-01f, -1.3734e-02f, -8.8633e-02f, -1.9917e-01f, -1.5237e-01f, -1.1870e-02f, 6.2622e-02f, 

        3.5295e-02f, -8.0524e-02f, 7.0753e-02f, 2.0527e-01f, 7.1976e-02f, -3.4012e-02f, -1.0594e-01f, -1.9983e-01f, 

        -4.6494e-03f, -1.3269e-01f, 1.9766e-02f, -1.8704e-02f, -1.2834e-01f, -6.5270e-02f, 1.0919e-01f, 2.5732e-01f, 

        -5.1963e-02f, 2.0048e-01f, 1.9096e-01f, 3.0701e-01f, -3.6385e-01f, 5.0913e-01f, -4.9879e-01f, -1.7643e-01f, 

        -4.1628e-01f, -4.8454e-01f, 3.3632e-01f, -5.9980e-01f, 9.0575e-01f, 3.4654e-01f, -1.3341e-01f, -1.8221e-03f, 

        -1.0800e+00f, -8.0761e-02f, 4.8956e-01f, -2.5087e-01f, -3.5166e-01f, -5.5544e-01f, -7.4853e-03f, -1.9522e-01f, 

        7.1448e-01f, -5.3678e-01f, -7.0833e-01f, 1.1455e-01f, -2.9053e-01f, -1.8474e-01f, -3.8412e-01f, -2.3383e-01f, 

        2.9133e-01f, 2.9859e-01f, -4.2294e-01f, 5.0661e-01f, -5.7226e-01f, -3.5378e-01f, 2.1600e-01f, 6.8345e-03f, 

        -7.6066e-01f, -1.0491e+00f, -6.7624e-02f, 5.7922e-01f, -8.3122e-01f, 1.3180e-01f, 4.1708e-02f, 1.4547e-01f, 

        3.1588e-01f, -4.0415e-01f, -2.7587e-01f, -1.0674e+00f, 5.7105e-03f, -6.4722e-01f, 8.7442e-01f, -3.5039e-01f, 

        -2.3729e-01f, -2.4739e-01f, 1.0636e-01f, -1.5275e-01f, 2.6681e-01f, -5.1167e-01f, 1.3667e-01f, 7.8066e-01f, 

        -1.2859e-01f, -8.0142e-01f, 1.8961e-01f, -4.3515e-01f, -6.0413e-01f, 1.2811e-01f, -2.2403e-01f, -8.7826e-01f, 

        -8.2447e-02f, 5.0583e-01f, -4.3688e-01f, -3.6518e-01f, 1.5489e+00f, -7.1051e-01f, 1.1042e-01f, 2.6384e-01f, 

        2.6137e-01f, -9.1761e-01f, 4.4637e-01f, -2.8385e-01f, 2.2707e-01f, 4.2908e-01f, 1.8408e-01f, 1.5607e-01f
    };

    // Convert and store feedforward weights
    for (int j = 0; j < 1024; j++) {
        float scaled = conv1_weights_vector[j] / scale;
        arm_float_to_q15(&scaled, &weights1[j], 1);
    }

    for (int j = 0; j < 18432; j++) {
        float scaled = conv2_weights_vector[j] / scale;
        arm_float_to_q15(&scaled, &weights2[j], 1);
    }

    for (int i = 0; i < 5760; i++) {
        float scaled = fc3_weights_vector[i] / scale;
        arm_float_to_q15(&scaled, &weights3[i], 1);
    }

}

void SNN_Init(void) {
    const float scale = 60.0f;

    // Layer 1 initialization
    // Uniform parameters for all neurons
    q15_t threshold_1, reset_value_1, decay_factor_1;
    float threshold_f_1 = 1.0000e+00 / scale;
    float reset_value_f_1 = 0.0000e+00 / scale;
    float beta_1 = 9.5000e-01f;

    arm_float_to_q15(&threshold_f_1, &threshold_1, 1);
    arm_float_to_q15(&reset_value_f_1, &reset_value_1, 1);
    arm_float_to_q15(&beta_1, &decay_factor_1, 1);

    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        LIFNeuron_Init(&layer1[i], threshold_1, reset_value_1);
        layer1[i].decay_factor = decay_factor_1;
    }

    // Layer 2 initialization
    // Uniform parameters for all neurons
    q15_t threshold_2, reset_value_2, decay_factor_2;
    float threshold_f_2 = 1.0000e+00 / scale;
    float reset_value_f_2 = 0.0000e+00 / scale;
    float beta_2 = 9.5000e-01f;

    arm_float_to_q15(&threshold_f_2, &threshold_2, 1);
    arm_float_to_q15(&reset_value_f_2, &reset_value_2, 1);
    arm_float_to_q15(&beta_2, &decay_factor_2, 1);

    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        LIFNeuron_Init(&layer2[i], threshold_2, reset_value_2);
        layer2[i].decay_factor = decay_factor_2;
    }

    // Layer 3 initialization
    // Uniform parameters for all neurons
    q15_t threshold_3, reset_value_3, decay_factor_3;
    float threshold_f_3 = 1.0000e+00 / scale;
    float reset_value_f_3 = 0.0000e+00 / scale;
    float beta_3 = 9.5000e-01f;

    arm_float_to_q15(&threshold_f_3, &threshold_3, 1);
    arm_float_to_q15(&reset_value_f_3, &reset_value_3, 1);
    arm_float_to_q15(&beta_3, &decay_factor_3, 1);

    for (int i = 0; i < NUM_NEURONS_LAYER3; i++) {
        LIFNeuron_Init(&layer3[i], threshold_3, reset_value_3);
        layer3[i].decay_factor = decay_factor_3;
    }

    // Load weights from NIR
    Load_NIR_Weights();

}

void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes) {
    // Layer 1 (convolutional)
    LIFNeuron_Conv2d_Update_Subtract_Base(layer1, input_spikes, weights1, l1_spikes, 10, 10, 2, 7, 7, 32, 4, 4, 1, 0);

    // Layer 2 (convolutional)
    LIFNeuron_Conv2d_Update_Subtract_Base(layer2, l1_spikes, weights2, l2_spikes, 7, 7, 32, 3, 3, 64, 3, 3, 2, 0);

    // Layer 3 (no recurrent, fully connected)
    LIFNeuron_Layer_Update_Vectorized_NoRecurrent(layer3, l2_spikes, weights3, NUM_NEURONS_LAYER2, NUM_NEURONS_LAYER3, l3_spikes, 0);

    // Copy output spikes
    for (int i = 0; i < NUM_NEURONS_LAYER3; i++) {
        output_spikes[i] = l3_spikes[i];
    }
}

void SNN_Reset_State(void) {
    // Reset layer 1
    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        layer1[i].membrane_potential = layer1[i].reset_value;
        l1_spikes[i] = 0;
    }

    // Reset layer 2
    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        layer2[i].membrane_potential = layer2[i].reset_value;
        l2_spikes[i] = 0;
    }

    // Reset layer 3
    for (int i = 0; i < NUM_NEURONS_LAYER3; i++) {
        layer3[i].membrane_potential = layer3[i].reset_value;
        l3_spikes[i] = 0;
    }

}
