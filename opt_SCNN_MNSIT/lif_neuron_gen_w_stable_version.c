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
// Layer 0: 200 -> 392 (convolutional, no recurrent, uniform params)
// Layer 1: 392 -> 144 (convolutional, no recurrent, uniform params)
// Layer 2: 144 -> 10 (fully connected, no recurrent, uniform params)

// Global variables for the SNN
#define NUM_INPUTS 200
#define NUM_INPUT_CHANNEL 2
#define L1_OUT_CH      8
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
#define NUM_NEURONS_LAYER1 392
#define L2_OUT_CH      16
#define L2_IN_CH       8
#define L2_KERNEL_H    3
#define L2_KERNEL_W    3
#define L2_KERNEL_SIZE 9
#define L2_STRIDE_H    2
#define L2_STRIDE_W    2
#define L2_PAD_H       0
#define L2_PAD_W       0
#define L2_OUT_H       3
#define L2_OUT_W       3
#define L2_COL_BUF_SIZE 144  // 2 * in_ch * kH * kW
#define NUM_NEURONS_LAYER2 144
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
        //     snprintf(buf, sizeof(buf), "V:%ld = Reset:%d + acc: %hu| threshold: %d S:%d | nindex = %d \r\n", 
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
        //     snprintf(buf, sizeof(buf), "V:%ld = Reset:%d + acc: %hu| threshold: %d S:%d | nindex = %d \r\n", 
        //             (long)neurons[i].membrane_potential, neurons[i].reset_value, weighted_inputs[i] , neurons[i].threshold, 
        //                 output_spikes[i], i);
        //     usart1_print(buf);
        // }
        
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
                //     snprintf(buf, sizeof(buf), "V:%ld = Reset:%ld + v_shifted:%ld + acc: %ld| threshold: %d S:%d | nindex = %ld | v_prev = %ld | decay = %ld \r\n", 
                //             (long)neurons[n_idx].membrane_potential, reset, v_shifted, acc, neurons[n_idx].threshold, 
                //              output_spikes[n_idx], n_idx, v_prev, decay);
                //     usart1_print(buf);
                // }


            }
        }
    }
}
        

void Load_NIR_Weights(void) {
    const float scale = 360.0f;

    // Layer 1 conv weights - Conv2d (8x2x4x4)
    // Stored in OUT_CH-MAJOR order: [oc][ic][kh][kw]
    static const float conv1_weights_vector[256] = {
        7.9196e-01f, -3.9047e-03f, -2.5340e-01f, -7.1724e-02f, -4.1208e-02f, -5.8685e-01f, -1.5212e+00f, -6.6925e-01f, 

        1.0829e-02f, -1.2151e-01f, -2.0535e+00f, 3.8118e-01f, -1.7819e-02f, -1.9845e-01f, -8.6854e-01f, -4.3855e-01f, 

        5.2480e-01f, -2.2710e-01f, -3.6062e-01f, -2.4615e-01f, -8.1053e-02f, -1.8101e-01f, -1.5128e+00f, -1.9975e-02f, 

        -4.6573e-02f, -9.8021e-02f, -1.4428e+00f, -2.8156e-01f, -7.2113e-02f, -2.4193e-01f, -9.4844e-02f, -1.1769e-03f, 

        -8.6418e-01f, -9.0540e-01f, -2.3864e+00f, -6.8120e-01f, 4.1442e-01f, 2.9814e-01f, 8.6930e-01f, -1.3653e-02f, 

        -2.7866e-01f, 1.1709e-01f, 1.1277e-01f, 4.3425e-02f, -1.2565e+00f, -3.4777e+00f, -2.7270e+00f, -3.6308e+00f, 

        -5.0768e-01f, -6.0403e-01f, -1.3124e+00f, -4.8003e-01f, -8.9230e-02f, -4.7552e-02f, 4.5249e-01f, 2.7173e-01f, 

        -3.5562e-01f, 1.6810e-01f, 1.0053e-01f, -7.2752e-01f, -1.5187e+00f, -2.9957e+00f, -2.2722e+00f, -2.7668e+00f, 

        -1.3420e+00f, -5.9447e-01f, -4.5827e-01f, -2.7412e+00f, -1.6270e+00f, -9.9096e-01f, 1.2749e+00f, -2.3335e+00f, 

        4.7536e-01f, -7.9123e-01f, -4.1439e-01f, -5.8635e-01f, -9.4562e-01f, -2.1780e+00f, -1.8371e+00f, -3.1710e+00f, 

        -1.3386e+00f, -8.9450e-01f, 2.0943e-01f, -2.0677e+00f, -4.1758e-01f, -1.8241e-01f, 5.3251e-01f, -1.7778e+00f, 

        -4.7060e-01f, -2.3842e-01f, -1.5190e+00f, -3.5973e-01f, -1.0704e+00f, -5.8375e-01f, -3.8336e-01f, -2.6264e+00f, 

        -1.3502e-01f, 4.8269e-01f, 6.1251e-01f, -1.0186e+00f, -4.2733e-01f, 8.9461e-01f, -6.0676e-01f, -1.6988e+00f, 

        -1.0931e+00f, -8.6140e-01f, -5.0924e-01f, -1.8471e-01f, -1.6872e+00f, -1.5964e+00f, -9.8358e-01f, -1.1088e+00f, 

        -6.2058e-01f, -4.8787e-01f, -9.3147e-01f, -7.8351e-01f, -2.1604e-01f, -3.4317e-01f, -8.2096e-01f, -2.1044e-01f, 

        -1.0251e+00f, -4.3211e-01f, 5.7026e-01f, -2.8746e-01f, -1.2947e+00f, -9.5251e-01f, -9.2251e-01f, -9.4506e-01f, 

        -1.2986e-01f, -3.4570e-01f, -1.1383e+00f, 9.5188e-02f, 1.6295e-01f, 3.2540e-01f, -2.1668e+00f, -1.1089e+00f, 

        5.1574e-02f, -1.7178e+00f, -1.5615e+00f, 1.2062e-01f, 4.2647e-01f, -4.9588e-01f, -2.0657e+00f, 5.7290e-02f, 

        -5.2750e-02f, -9.3233e-02f, -7.7699e-01f, 3.1540e-01f, 1.0254e-01f, -9.0103e-02f, -1.5895e+00f, 2.1126e-02f, 

        -1.9605e-02f, -5.7349e-01f, -6.7166e-01f, -3.6588e-01f, 6.2049e-01f, -7.6478e-02f, -1.4170e+00f, -4.7385e-01f, 

        6.1516e-01f, -8.8448e-01f, -5.5258e-01f, -6.4096e-01f, -5.8456e-01f, -3.4042e-02f, -4.4511e-01f, -6.6772e-01f, 

        -9.8508e-02f, -2.7669e-01f, 5.5923e-02f, -2.4752e-01f, -4.2395e-01f, -8.4695e-01f, -1.1800e+00f, -7.3250e-01f, 

        -6.8299e-01f, 1.7317e-01f, -2.8489e-01f, -5.7970e-01f, -2.7025e-02f, -1.1478e-01f, -4.7427e-01f, -4.5359e-01f, 

        -1.1570e-01f, -5.2822e-02f, -2.8675e-01f, -3.5555e-01f, -6.5962e-01f, -5.2246e-01f, -1.0350e+00f, -2.5401e-01f, 

        -5.6712e-01f, -1.6687e-01f, 1.6972e-01f, -4.2248e-01f, -2.0805e+00f, -1.4510e+00f, -1.1408e-01f, 8.0152e-01f, 

        -1.6822e+00f, -2.6480e+00f, 2.0610e-02f, 2.5401e-01f, -8.4954e-01f, -9.2870e-01f, -6.7463e-02f, -1.8372e+00f, 

        -7.9738e-01f, 7.2200e-02f, -1.1867e-01f, 2.0258e-01f, -1.8767e+00f, -5.9873e-02f, 2.0823e-01f, 4.9007e-01f, 

        -2.9589e-01f, -1.5984e+00f, -1.0509e+00f, -3.6709e-01f, -5.9392e-01f, -8.7057e-01f, -1.8593e-01f, -1.6302e+00f, 

        -1.8796e-01f, -1.7982e-01f, -9.2196e-02f, -1.2190e-01f, -1.2173e-01f, 7.4218e-01f, 3.8593e-01f, 5.6074e-02f, 

        4.9940e-01f, 3.8843e+00f, 1.0881e+00f, -1.9535e-02f, 1.1280e+00f, 4.4623e-01f, -7.3310e-03f, -2.5033e-01f, 

        -9.8640e-02f, -3.1111e-02f, 5.3066e-02f, -1.7783e-01f, -1.6021e-01f, 4.1813e-01f, 2.4321e-01f, 1.2267e-01f, 

        -1.1915e-02f, 3.5453e+00f, 4.7885e-01f, -1.3406e-02f, 4.7192e-01f, 4.0355e-02f, 2.1993e-01f, -4.5176e-02f
    };

    // Layer 2 conv weights - Conv2d (16x8x3x3)
    // Stored in OUT_CH-MAJOR order: [oc][ic][kh][kw]
    static const float conv2_weights_vector[1152] = {
        -5.1328e-01f, -5.7861e-01f, -3.0863e-01f, -3.1534e-01f, -2.6235e-01f, -3.8140e-01f, -2.4454e-01f, -2.0617e-01f, 

        -1.4903e-01f, -4.2351e-01f, -6.3199e-01f, -3.0922e-01f, -3.3094e-01f, -2.2299e-01f, -1.2203e-01f, -3.4599e-01f, 

        -6.0785e-02f, -3.0384e-02f, -1.3016e-01f, -5.3357e-01f, -4.7593e-01f, -2.0574e-01f, -1.4746e-01f, -3.6429e-01f, 

        -1.8834e-01f, -2.8769e-01f, -5.3100e-02f, -1.2906e+00f, -6.7289e-01f, -3.2649e-01f, -4.2684e-01f, -2.2517e-01f, 

        -3.1852e-01f, -3.5443e-01f, -3.1187e-01f, -1.1635e-01f, -3.7848e-01f, -3.1095e-01f, -2.6333e-01f, -2.4345e-01f, 

        -2.3322e-01f, -1.3984e-01f, -2.1136e-01f, -1.7912e-01f, -1.6323e-01f, -2.5977e-01f, -2.9837e-01f, -2.5992e-01f, 

        -3.4528e-01f, -6.3189e-02f, -2.2016e-01f, -1.0086e-01f, -1.5525e-01f, -1.0526e-01f, -1.4518e-01f, -2.0629e-01f, 

        -4.1818e-01f, -1.0455e-01f, -8.4332e-02f, -1.6235e-01f, -6.0322e-02f, -2.7622e-04f, -2.7685e-01f, -2.6401e+00f, 

        -1.2394e+00f, -1.4991e+00f, -1.9348e+00f, -1.8686e+00f, -2.4293e+00f, -2.2615e+00f, -6.4560e-01f, -3.0251e+00f, 

        -3.4061e-01f, -2.1784e-01f, -1.8665e-01f, -2.4930e+00f, -5.8124e-01f, 2.9088e-01f, 3.2926e-01f, -9.9695e-01f, 

        -3.8925e-01f, -3.6587e-01f, -1.2173e+00f, 4.4214e-01f, -1.1474e+00f, -9.0438e-01f, -1.6919e+00f, -1.7824e+00f, 

        -2.0881e+00f, -1.7310e+00f, 7.9748e-01f, 1.2148e+00f, -1.5717e+00f, -9.9855e-01f, -5.0442e-01f, 1.0097e-03f, 

        -1.5530e+00f, 8.9270e-02f, 2.0311e-01f, 2.6282e-01f, 8.7373e-01f, 2.8712e-01f, 1.7783e+00f, 1.5328e+00f, 

        -2.0789e-01f, 2.3296e-01f, 1.3960e+00f, 1.0564e+00f, -8.4596e-01f, -7.7597e-01f, -2.3404e-01f, -7.1612e-01f, 

        -1.4155e+00f, -1.2624e+00f, -6.6456e-01f, -1.0435e+00f, -2.9472e+00f, 2.6600e-01f, 6.0914e-01f, 9.0009e-01f, 

        4.2540e-02f, 4.8360e-01f, -2.2809e-01f, -1.4931e+00f, -5.8015e-01f, -5.1776e-01f, 1.1903e-01f, -1.7112e-01f, 

        -5.8342e-01f, 1.0300e+00f, 1.0229e+00f, -1.2303e+00f, 1.1419e+00f, 5.4367e-01f, -2.2368e-01f, 4.7778e-01f, 

        -5.6199e-02f, -5.1332e-01f, 3.8108e-01f, -2.0845e-01f, -1.5666e-01f, 2.0852e-01f, -7.5447e-01f, -4.3657e-02f, 

        3.4101e-01f, -1.6291e-01f, -6.6326e-02f, -1.0046e-01f, 7.3409e-01f, -1.8012e+00f, -1.1877e-01f, 9.8724e-01f, 

        -1.3312e+00f, -9.4862e-01f, -4.3423e-01f, -1.3145e+00f, 3.7490e-01f, -8.9112e-01f, -1.2408e+00f, 5.5072e-02f, 

        -1.0548e-01f, -5.3089e-01f, -9.8985e-01f, 1.2500e+00f, 2.1090e-01f, 4.9593e-02f, 1.3145e+00f, -3.8673e-01f, 

        8.9156e-01f, 7.4754e-01f, 1.6706e-01f, 3.6304e-01f, -5.3233e-01f, 2.5143e-01f, -2.4526e-01f, -1.1378e+00f, 

        4.4532e-01f, -3.2266e+00f, -5.3962e+00f, 5.0365e-01f, -1.4418e+00f, 4.7817e-01f, -6.5663e-01f, 4.5841e-01f, 

        -4.6424e-01f, -1.7213e+00f, -7.4636e-01f, -6.5156e-01f, -9.8199e-01f, -1.4784e+00f, -3.5823e-01f, -3.1447e-01f, 

        -9.5000e-01f, -9.7681e-01f, -3.8335e-01f, -1.3068e+00f, -9.4517e-01f, 2.6490e-01f, -7.2479e-01f, 1.0351e+00f, 

        4.3788e-01f, -1.5680e+00f, 4.9131e-01f, 7.9050e-01f, -5.0401e-01f, 1.2104e+00f, 1.1542e+00f, -9.0295e-01f, 

        -5.7041e-01f, 5.6311e-01f, -8.3301e-01f, -2.6567e-01f, 2.6155e-01f, 3.9455e-03f, -3.5107e-01f, 2.8558e-01f, 

        2.3810e-01f, -1.5227e+00f, -5.0867e-01f, -3.2297e-01f, -2.1199e+00f, -4.0068e-01f, 4.7418e-01f, -6.5019e-01f, 

        -7.3869e-01f, -1.2175e+00f, -1.8061e-01f, -5.8312e-01f, -5.5930e-01f, 1.9790e-01f, 1.5658e-01f, -9.2641e-02f, 

        -4.0664e-01f, -1.0740e+00f, -8.2075e-01f, -3.5324e-01f, -4.4036e-01f, -1.4880e-01f, 9.0570e-01f, -1.8139e-01f, 

        -2.2394e-01f, 6.3704e-02f, -1.5627e+00f, -4.6007e-03f, 1.7230e-01f, -9.9991e-02f, 4.5687e-01f, 1.0939e+00f, 

        -5.0669e-01f, 1.0728e+00f, -1.0513e+00f, -7.0820e-01f, 3.1038e-01f, -6.4561e-01f, -3.5218e-01f, -7.7216e-01f, 

        -1.2579e-01f, -1.0070e+00f, -8.9117e-01f, -6.7635e-01f, -5.8295e-01f, 5.8967e-02f, -3.1616e-01f, -5.9753e-04f, 

        7.2939e-01f, -1.0030e-01f, -4.2060e-01f, 7.2979e-01f, 2.0405e-01f, 9.2051e-02f, -1.3580e-01f, 8.4719e-01f, 

        9.5367e-01f, 4.8771e-01f, 3.4409e-01f, 8.1819e-01f, -1.5158e-01f, -4.6913e-01f, -1.9069e+00f, -3.4943e+00f, 

        -3.0574e-01f, 4.0817e-01f, -2.4471e-01f, -1.0107e+00f, -1.2928e+00f, 6.8976e-01f, 1.1587e-02f, 5.0153e-01f, 

        7.0607e-02f, 1.2846e-01f, -3.0644e-02f, -3.9237e-03f, 2.0612e-01f, -2.0401e-01f, -3.5886e-01f, 5.3074e-02f, 

        1.1358e-01f, -5.8483e-01f, -1.2127e+00f, -9.7518e-01f, -1.8321e+00f, -1.4644e+00f, 9.9609e-01f, -5.6248e-02f, 

        6.3715e-01f, 6.1496e-01f, -4.6345e-01f, -2.5039e-01f, 1.3942e+00f, -2.9266e-02f, 7.6384e-01f, -7.3641e-01f, 

        -2.6610e-01f, 5.4469e-01f, 1.1131e+00f, 4.6234e-01f, 1.3249e-01f, -8.6605e-01f, 6.3999e-01f, 1.0542e-01f, 

        -2.0327e+00f, -7.2551e-01f, -8.6345e-01f, -5.8183e-01f, 1.8014e-01f, 8.2356e-02f, -5.2298e-01f, 9.6011e-01f, 

        -7.0835e-02f, 6.2997e-01f, 3.5245e-01f, 2.5685e-01f, 1.0778e+00f, -2.2107e-01f, -2.7483e-01f, -4.6514e-01f, 

        -1.6978e-01f, -7.4244e-01f, 4.7874e-01f, -2.0854e-01f, 1.0524e+00f, 1.6517e+00f, -1.1520e+00f, 4.9363e-01f, 

        -1.1649e+00f, 1.5460e+00f, 1.7367e+00f, -1.3541e+00f, -1.1906e+00f, 4.5149e-01f, -1.1764e+00f, 7.8367e-02f, 

        -3.2519e-01f, -5.5901e-01f, 4.0908e-01f, -2.6670e-02f, -1.2028e+00f, 9.9257e-02f, 6.1490e-02f, -1.3919e+00f, 

        -1.0974e+00f, -2.1850e-01f, -4.5559e-02f, -2.5618e-01f, -1.4028e+00f, -6.5814e-01f, 5.2046e-01f, -9.3837e-01f, 

        -6.3664e-01f, -8.6017e-01f, -9.8202e-01f, -9.8020e-01f, -5.9579e-02f, -4.4029e-01f, -1.2823e+00f, -1.0798e+00f, 

        -1.0555e+00f, 2.2973e-01f, -6.1138e-01f, -2.6420e-01f, -9.1415e-01f, 9.0837e-01f, 8.2173e-02f, 1.0619e+00f, 

        -1.7913e+00f, 1.7674e-01f, -7.7449e-01f, -1.7487e+00f, -3.5589e-01f, 3.9796e-01f, 7.3558e-01f, 8.2629e-01f, 

        -1.8578e-01f, -6.4073e-01f, -2.6872e+00f, -4.7183e-01f, -3.0827e-01f, -1.8160e+00f, 3.5625e-01f, -4.2120e-01f, 

        -4.0624e-01f, 8.9972e-01f, -1.4382e-01f, 1.6996e-01f, 8.7092e-01f, -3.2799e-01f, -2.1887e-01f, -4.7352e-01f, 

        -5.3766e-02f, -2.8867e-01f, -4.2099e-01f, 2.0602e-01f, -4.5974e-03f, -3.2460e-02f, -6.9077e-01f, 5.9212e-01f, 

        -6.9175e-01f, 9.5381e-03f, -1.0631e+00f, -9.0591e-01f, -1.7665e-01f, -4.1601e-01f, -4.9746e-01f, -7.6466e-01f, 

        3.6467e-01f, 5.8721e-03f, -8.0888e-01f, 6.1863e-01f, -2.0317e-01f, -8.8427e-02f, 3.5776e-01f, -3.6291e-01f, 

        1.5194e-01f, -5.8659e-02f, 2.8018e-01f, 2.1624e-01f, 4.8658e-01f, 5.8478e-01f, 2.5516e-01f, 1.6613e-01f, 

        3.1203e-01f, 4.4517e-01f, -5.0895e-01f, -8.1877e-01f, 4.8536e-01f, 2.1374e-01f, -8.7761e-01f, 1.0753e-01f, 

        1.3469e+00f, -8.0619e-01f, 9.3741e-01f, -5.0002e-01f, -1.0372e+00f, -9.2060e-01f, 2.9302e-01f, -2.2336e-01f, 

        -5.1786e-01f, -2.3518e-01f, 9.4331e-01f, 4.0174e-01f, 1.5959e-01f, 2.3161e+00f, 7.5683e-01f, -2.3619e-02f, 

        1.1343e+00f, 1.2611e+00f, 4.0408e-01f, 6.3775e-01f, -1.2989e-01f, -3.0152e-01f, -5.9125e-01f, -3.0661e-01f, 

        -7.6726e-01f, 5.6024e-01f, 3.1177e-01f, -5.7724e-01f, 6.4802e-02f, -5.2073e-01f, -8.7863e-02f, 1.7728e-01f, 

        -1.7222e-01f, -1.1640e-01f, -1.0592e+00f, -7.3969e-01f, 5.0009e-01f, 1.7770e-01f, 1.6374e-01f, -5.9767e-01f, 

        -1.2756e+00f, -5.9897e-01f, -1.0274e+00f, -1.1933e+00f, -2.6526e-01f, 8.2638e-02f, -1.0413e+00f, -9.5321e-02f, 

        -1.7038e-01f, -2.6879e-01f, -1.0010e-01f, 1.8274e-01f, -1.3346e+00f, -3.4197e-02f, -1.1963e+00f, 1.0223e-01f, 

        -3.9526e-01f, -3.1202e-01f, -2.9911e-01f, -3.9479e-01f, -2.8445e-01f, -1.6092e-01f, -2.9306e-01f, -2.6050e-01f, 

        -2.3371e-01f, -5.4641e-01f, -3.6375e-01f, -2.4151e-01f, -4.5375e-01f, -2.2055e-01f, -2.1635e-01f, -2.7060e-01f, 

        -1.9629e-01f, -1.2765e-01f, -2.4143e-01f, -1.8637e-01f, -1.3456e-01f, -3.5559e-01f, -3.2856e-02f, 4.1225e-02f, 

        -5.2589e-01f, -3.1569e-01f, 3.0710e-02f, -2.2804e-01f, -5.9591e-01f, -8.4237e-02f, -2.8019e-01f, -1.7552e-01f, 

        -2.0228e-01f, -3.5880e-01f, -9.4296e-02f, -3.4177e-01f, -3.6665e-01f, -1.6409e-01f, -2.2069e-01f, -5.1161e-02f, 

        -1.1410e-01f, -8.1957e-02f, -2.9853e-01f, -9.9587e-02f, -2.2741e-01f, -2.7790e-01f, -9.9238e-02f, -1.0341e-01f, 

        -1.5103e-01f, -2.1963e-01f, -2.7472e-01f, -1.6265e-01f, -9.4515e-02f, -3.3196e-02f, -2.3886e-01f, -3.1020e-01f, 

        -2.6603e-01f, -5.0174e-02f, -2.1678e-01f, -3.9713e-01f, -6.6394e-02f, -3.0510e-03f, -3.2918e-01f, -1.6052e+00f, 

        -2.7362e+00f, -1.9747e+00f, -2.2042e+00f, -1.1931e+00f, -1.8518e+00f, -1.9972e+00f, -1.0996e+00f, -7.2732e-01f, 

        -8.2902e-01f, -1.5729e+00f, -3.1348e+00f, -2.9621e-01f, -9.0038e-01f, -1.8026e+00f, -4.8267e-01f, 5.7467e-02f, 

        -1.2804e-01f, -1.8026e+00f, -1.4535e+00f, -5.9142e-01f, -7.6402e-01f, 2.1379e-01f, 4.2874e-01f, 1.8336e-01f, 

        2.3830e-01f, -1.2738e-01f, -1.3508e+00f, -2.5812e+00f, 8.3924e-01f, -1.3691e+00f, 1.2478e-01f, -1.2703e+00f, 

        -4.1114e-03f, -5.1466e-01f, 4.7588e-01f, -3.4190e-01f, -2.9902e-01f, 4.8908e-01f, 1.4940e-03f, -1.5391e+00f, 

        -1.4880e+00f, -6.0068e-01f, -1.9854e-02f, -4.3154e-01f, -7.1756e-01f, -1.3880e+00f, -1.1585e-01f, -8.6063e-02f, 

        2.3714e-02f, -1.1766e-01f, -3.4132e-01f, -2.7840e-01f, -4.0481e-01f, -2.2771e+00f, 9.1882e-01f, -1.5287e+00f, 

        -1.1870e+00f, -4.9683e-01f, -3.1106e-01f, -7.3241e-01f, 1.2634e-01f, -7.3700e-01f, -2.2371e+00f, -6.9346e-01f, 

        -1.4743e-01f, 7.6495e-02f, -7.2728e-01f, 3.4031e-02f, -7.5145e-01f, -1.1871e+00f, 5.8415e-01f, -4.3651e-01f, 

        -4.2499e-01f, 3.1696e-02f, 1.5664e-01f, 2.0475e-01f, 2.7596e-02f, -1.2079e-01f, -4.1432e-03f, 1.9093e-01f, 

        5.6267e-01f, -9.9523e-01f, -1.9708e+00f, -1.5688e+00f, -2.3943e+00f, 7.8218e-02f, -7.5462e-01f, -1.5605e-01f, 

        -6.3485e-01f, -1.6668e+00f, -2.0826e+00f, 4.2637e-01f, -1.0288e-02f, -6.3569e-01f, -4.7389e-01f, 2.3851e-01f, 

        -4.5282e-01f, -1.9980e+00f, -6.5648e-01f, -4.6468e-01f, -5.2884e-01f, 5.4477e-01f, 4.2631e-01f, -8.8200e-01f, 

        -4.4154e-01f, -7.0852e-01f, -9.2977e-01f, -3.5187e+00f, -2.1624e+00f, -9.9361e-01f, 1.3579e+00f, -1.6143e-01f, 

        -2.3465e+00f, -3.6933e-01f, 4.3375e-02f, -2.3823e+00f, 1.9911e-01f, -7.7549e-01f, -1.5380e+00f, -1.4251e-01f, 

        -3.7769e-01f, -3.3156e-01f, -5.2807e-01f, -8.4724e-01f, 2.7565e-01f, -6.3445e-01f, -1.4221e+00f, -5.6748e+00f, 

        -3.1165e+00f, -3.0047e+00f, 5.2470e-01f, -5.9388e-01f, -1.9010e+00f, 1.4735e+00f, -1.9932e+00f, -3.6381e+00f, 

        -1.6269e-01f, -2.2871e+00f, -2.8953e+00f, -6.7754e-01f, -4.0286e+00f, -1.0726e+00f, -8.4842e-02f, 4.4150e-02f, 

        1.9628e-01f, 2.0501e-02f, 2.4416e-01f, -1.2117e-01f, 2.0499e-01f, 1.4484e-01f, -2.1719e-01f, -1.0353e-01f, 

        -1.9922e-01f, -1.3702e+00f, 6.0365e-02f, 2.4990e-01f, 2.6863e-02f, -6.0979e-01f, -2.1813e-01f, -5.2360e-01f, 

        -7.2920e-01f, -2.8976e-01f, -5.8346e-01f, -7.3917e-01f, -1.9598e-01f, -7.8686e-01f, -1.9445e+00f, -3.8545e-01f, 

        -2.8585e-01f, -2.2698e+00f, 3.4510e-01f, -1.1084e+00f, -2.1207e+00f, -9.8466e-02f, 1.0252e+00f, 6.4363e-01f, 

        -3.7246e-01f, 2.9078e-01f, 1.5490e+00f, 5.8276e-01f, -1.2038e+00f, 1.3888e+00f, -7.2912e-01f, -5.6376e-01f, 

        -2.4016e-01f, -6.7022e-02f, -2.8111e+00f, 1.7461e-02f, -5.6450e-01f, 1.0460e-01f, -2.0167e-01f, 1.7689e-01f, 

        3.7256e-01f, -6.3006e-01f, -2.6382e-01f, -3.6993e-01f, -4.3257e-01f, -3.1070e-01f, 5.1566e-01f, 5.5460e-01f, 

        7.6769e-01f, -4.1407e-01f, -5.7408e-01f, 9.4260e-02f, -2.4449e-01f, -2.4939e-01f, -2.2050e-01f, -3.0232e-01f, 

        -3.6262e-01f, -3.8361e-01f, 2.2135e-01f, -8.3952e-01f, -1.7233e-01f, -1.7065e+00f, -1.9579e+00f, 2.0113e-01f, 

        -2.8081e-01f, 2.3211e-01f, -3.1302e-01f, -2.1011e-01f, 3.2932e-01f, -7.3841e-01f, 2.7084e-01f, 2.6762e-01f, 

        -5.4223e-01f, 5.6367e-02f, 3.5377e-01f, -4.5070e-01f, -2.4516e-01f, 4.0388e-02f, -1.0894e+00f, -3.1314e+00f, 

        -3.8175e+00f, 5.4877e-01f, -3.2559e-01f, -4.8266e-01f, -4.5594e-01f, -2.3127e-01f, -1.1599e+00f, 1.3029e-01f, 

        2.7487e-01f, -9.6012e-01f, -1.0618e+00f, -3.9028e-02f, 1.8823e-01f, -1.0449e+00f, 3.6047e-01f, 3.6872e-01f, 

        -4.3398e-01f, -1.2911e+00f, 5.8327e-02f, 7.9683e-02f, 1.8942e-01f, 1.9621e+00f, -2.4016e-01f, -3.9649e-01f, 

        1.1902e+00f, 8.4100e-01f, -1.0210e+00f, 2.5658e+00f, -3.1550e-02f, 4.8548e-01f, 2.0138e-03f, 3.4191e-01f, 

        -1.8491e-01f, -1.2752e+00f, -1.0080e+00f, -1.1796e+00f, -1.3704e+00f, -1.2716e+00f, -6.1817e-01f, -5.3792e-01f, 

        -5.0869e-01f, -1.8920e+00f, 6.2135e-01f, -1.5368e+00f, -2.2654e-01f, 4.0652e-02f, 1.4863e-01f, 1.9201e-01f, 

        4.3400e-01f, 3.7212e-02f, 9.2244e-01f, 4.9638e-01f, -5.3292e-02f, 6.6075e-01f, 2.0396e-01f, 2.0372e-01f, 

        1.2999e-01f, -9.8565e-02f, 1.1697e-01f, 1.1157e-02f, -9.7021e-02f, -4.4699e-02f, 2.1976e-01f, 1.0251e-02f, 

        -3.8638e-01f, 2.4759e-01f, 2.0762e-01f, 3.8076e-02f, -1.2031e+00f, 2.0375e-01f, -2.0738e-01f, -1.1196e-01f, 

        2.3872e-01f, -2.8732e-01f, 4.0702e-01f, -3.4065e-01f, 2.1304e-01f, 1.2541e-01f, -5.1965e-01f, -3.2046e-01f, 

        1.0289e-01f, 3.3244e-01f, -2.0530e+00f, 8.5895e-01f, 9.1326e-01f, 1.7516e-01f, 7.5220e-01f, 8.1222e-01f, 

        -1.5960e-01f, 5.3942e-01f, -7.6265e-01f, -3.6346e-01f, -9.4031e-01f, -9.6145e-01f, -5.8760e-01f, 2.2887e-01f, 

        1.3162e-01f, -6.6985e-01f, -2.4749e-01f, -2.7954e-01f, -1.7369e-01f, 7.9969e-03f, -9.7470e-01f, -6.4923e-01f, 

        -8.8925e-01f, -1.4210e+00f, -3.0221e-01f, -6.6239e-01f, -7.9193e-01f, 3.2098e-01f, 1.5336e-01f, -2.9909e-01f, 

        1.6616e-01f, -4.0697e-01f, 1.3135e+00f, 1.2253e-01f, 1.8841e-01f, 5.3392e-01f, -2.6739e-01f, -2.0619e-01f, 

        -8.0908e-01f, -4.5019e-01f, 5.8610e-01f, -6.1577e-01f, -3.0815e-01f, -2.3282e-01f, -1.6882e+00f, 5.7198e-01f, 

        5.8415e-02f, -2.6979e-01f, -5.2290e-01f, -2.1222e-01f, -5.1588e-01f, 4.6384e-01f, 3.1202e-01f, 5.1274e-02f, 

        5.6688e-02f, 2.6717e-02f, 3.1863e-01f, -3.6440e-01f, 3.5022e-01f, -6.8170e-01f, -1.5884e+00f, 2.0109e-01f, 

        -1.1638e+00f, 9.3825e-02f, -5.4396e-01f, 2.2523e-01f, 2.2103e-01f, 2.0991e-01f, 2.0766e-01f, -7.7607e-01f, 

        8.6736e-01f, 7.0691e-01f, -1.3408e+00f, -1.4786e+00f, -1.5119e-01f, 3.3020e-01f, -1.1535e+00f, 4.2334e-01f, 

        -6.0444e-01f, -1.1142e+00f, 3.2268e-01f, 9.6151e-01f, 7.2364e-01f, 1.3767e+00f, 8.9301e-01f, 4.7279e-01f, 

        3.2136e-01f, 2.8698e-01f, -9.4824e-01f, -1.2685e+00f, -1.1065e-01f, 1.5528e-01f, 2.8133e-01f, -5.5647e-01f, 

        -9.1010e-01f, -3.3341e-01f, -1.3132e+00f, -9.4736e-01f, -9.5937e-01f, 1.0608e+00f, 6.7740e-01f, 2.8038e-01f, 

        -4.6301e-01f, 8.0236e-01f, 9.6687e-01f, 1.0415e+00f, -2.7928e-01f, -2.9370e-01f, 1.9263e-01f, 2.5766e-01f, 

        3.6419e-01f, 6.1377e-02f, 2.1554e-01f, 3.6832e-01f, -1.3031e+00f, 6.1236e-01f, -8.7912e-01f, -3.6619e-01f, 

        5.6473e-01f, 3.8445e-01f, -5.6995e-01f, -3.8091e-02f, -2.5647e-01f, -2.7036e+00f, -4.4660e-01f, -1.3544e+00f, 

        1.5205e-03f, 4.0842e-01f, -5.0423e-01f, -9.0070e-01f, 2.0799e-01f, -1.1438e-01f, -1.0141e+00f, -3.9151e-01f, 

        3.3314e-01f, -3.7931e-01f, 4.1729e-01f, -5.7818e-01f, -2.4537e-01f, -6.0037e-01f, -3.4976e-01f, 8.8369e-02f, 

        -3.6540e-01f, -1.4503e-02f, -1.8429e-01f, -1.9452e-01f, 1.7870e-01f, -1.0955e+00f, -8.8032e-01f, -3.2316e-01f, 

        -4.1778e-01f, -8.6351e-01f, -1.2308e+00f, -6.9795e-01f, -9.4536e-02f, -9.3142e-01f, 1.7105e-01f, 3.1939e-01f, 

        -2.5755e+00f, 6.6493e-01f, -2.5883e-01f, -5.7578e-01f, -7.1785e-01f, -6.9813e-02f, -4.6719e-02f, -2.0791e-01f, 

        -1.8821e-01f, 1.3140e-01f, -8.7888e-01f, -7.5571e-01f, -1.2022e-01f, 1.6683e-01f, -3.8302e-01f, 3.9449e-02f, 

        5.7049e-01f, 8.4418e-01f, 5.0215e-01f, 6.0950e-01f, 8.7617e-01f, 2.9264e-01f, -5.6735e-01f, 5.9739e-02f, 

        -4.3943e-01f, -6.3889e-01f, 2.2792e-01f, -9.1421e-01f, -3.6127e-01f, 9.6068e-02f, -2.2311e-01f, -1.6777e-01f, 

        1.8813e-01f, -3.7051e-01f, -1.6064e-01f, -1.2058e-01f, 9.1650e-01f, -4.6691e-01f, -4.5721e-02f, -3.9292e-01f, 

        -5.2014e-01f, 1.3629e-01f, -1.6849e-01f, 5.6745e-02f, -4.9780e-02f, -1.1233e+00f, -3.5294e-01f, -2.9735e-01f, 

        -3.1105e-01f, 1.4846e-02f, -4.9804e-01f, 1.0899e+00f, -2.6800e-01f, -1.2891e-01f, 2.9463e-01f, -9.4472e-02f, 

        -1.2850e+00f, -1.8896e-01f, 1.0334e+00f, 5.7661e-01f, -2.3572e-01f, -2.1271e-01f, -9.8815e-01f, 7.5456e-01f, 

        5.3979e-01f, -8.0751e-01f, 7.5953e-01f, -7.6326e-01f, -6.5052e-02f, -6.3328e-01f, 4.7939e-01f, 1.1310e+00f, 

        9.2556e-01f, -4.2608e-01f, 1.6172e-01f, 5.9600e-01f, -3.5121e-01f, -2.0775e-01f, -8.5750e-01f, -9.2607e-01f, 

        -1.5256e+00f, -8.5412e-01f, -6.6369e-01f, -3.1551e-01f, -3.4738e-01f, 2.2169e-01f, -1.0932e-01f, -1.6703e-01f, 

        -7.5589e-02f, 8.7424e-02f, -1.7004e-01f, 2.6332e-01f, 2.3319e-01f, -3.5993e-01f, 1.4618e-01f, 1.6285e-01f, 

        4.8590e-01f, -1.3835e-01f, -7.8703e-01f, 3.4594e-01f, -2.3931e-01f, 1.7514e-01f, -9.0379e-01f, -3.7526e-01f, 

        1.8646e-01f, -6.0803e-01f, 4.7553e-01f, 7.6259e-01f, -4.1945e-01f, -4.0990e-02f, 3.5405e-02f, -3.3385e-01f
    };

    // Layer 3 feedforward weights - fully connected (144x10)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    static const float fc3_weights_vector[1440] = {
        -2.0159e-01f, -1.0479e-01f, 6.9038e-02f, -1.1746e-01f, -1.7036e-01f, 2.0563e-02f, -2.1700e-02f, -2.0231e-01f, 

        -2.0318e-01f, -1.5651e-01f, -5.9035e-02f, -1.5905e-02f, 3.8363e-02f, 8.3864e-02f, -5.7366e-02f, 7.6145e-02f, 

        7.0640e-02f, -2.4581e-02f, 1.0461e-02f, -1.7634e-02f, -6.7334e-02f, 2.5021e-02f, 9.5375e-02f, -1.3475e-02f, 

        -1.7277e-01f, -3.3178e-02f, 1.1596e-01f, -1.7642e-01f, -4.1817e-02f, -1.3651e-02f, -4.2365e-02f, -9.1426e-02f, 

        -2.6061e-02f, -1.3629e-01f, -1.5325e-01f, 6.7879e-02f, -1.5138e-01f, -9.7942e-02f, -5.6679e-02f, 4.9567e-03f, 

        -8.6917e-02f, -1.4395e-01f, -9.3257e-02f, -7.8054e-02f, -1.5060e-01f, -1.4537e-01f, -1.1302e-01f, -1.9475e-01f, 

        -1.0659e-01f, -6.1577e-02f, -1.0426e-01f, -1.0283e-01f, 5.4641e-02f, -1.3441e-02f, -6.2038e-02f, 2.2236e-02f, 

        -5.8594e-02f, -1.3249e-01f, -5.4900e-02f, -9.3463e-02f, -5.5153e-02f, -1.4488e-01f, 8.7223e-02f, -5.5742e-02f, 

        -1.2904e-01f, -1.5486e-01f, -5.8895e-02f, -1.0546e-01f, -9.9241e-02f, -4.0185e-02f, -7.1207e-02f, -1.7158e-01f, 

        -1.1149e-01f, 5.4851e-02f, -1.0573e-01f, -7.4070e-02f, -1.9939e-01f, -8.7290e-02f, 7.1054e-02f, -1.9928e-01f, 

        -1.4148e-01f, -7.6155e-02f, 2.7712e-03f, 9.7207e-02f, -7.4752e-02f, -4.7109e-02f, -1.3040e-01f, -3.0355e-02f, 

        -9.4424e-02f, -4.9586e-02f, 6.7167e-01f, -2.4242e-01f, 1.2490e+00f, 6.3490e-02f, -2.0626e-01f, 3.0951e-01f, 

        3.0731e-01f, 7.7034e-01f, -1.2363e-01f, -5.7746e-01f, 4.0818e-01f, -6.1958e-02f, 5.7647e-01f, 9.0422e-02f, 

        -4.7302e-02f, -5.8082e-01f, 2.9646e-01f, 4.4973e-01f, 1.9351e-01f, -7.8321e-01f, 1.8703e-01f, 8.7858e-03f, 

        9.0179e-02f, -4.1803e-01f, -1.1155e-01f, 1.0138e+00f, 5.0504e-01f, 3.2620e-01f, -7.7084e-01f, -5.1179e-01f, 

        6.1918e-01f, -9.6768e-02f, 1.6743e-01f, 6.3593e-01f, 2.2032e-01f, 1.4954e-01f, -4.4550e-01f, 6.0313e-01f, 

        2.4096e-01f, 3.3357e-01f, -1.8674e-01f, -2.9227e+00f, -1.3928e-02f, -1.1530e+00f, -5.9506e-01f, 4.9222e-01f, 

        -3.7866e-01f, 1.3998e-01f, 6.2413e-01f, -3.8158e-03f, 4.4076e-01f, -6.0195e-01f, -4.7095e-01f, -2.9615e-01f, 

        -4.7810e-01f, -3.9030e-01f, 1.2440e+00f, 4.2886e-01f, -5.2358e-02f, -3.6877e-01f, 1.8508e-01f, -6.2058e-01f, 

        2.8167e-02f, 1.8927e-01f, -3.0291e-01f, 1.6245e-01f, -2.3301e-01f, 4.4450e-01f, -4.1438e-01f, -5.2494e-01f, 

        -2.2372e-01f, -5.2238e-01f, -9.6487e-03f, -2.6907e+00f, 1.5881e-01f, -4.3206e-02f, -7.3662e-01f, -1.5402e-01f, 

        2.0049e+00f, 1.1475e-01f, 8.2375e-01f, 8.2038e-01f, 8.0887e-02f, 1.1015e-01f, 7.4338e-02f, 1.2503e-01f, 

        -4.2724e-02f, -4.0366e-01f, 5.2783e-01f, -5.5966e-01f, 1.9068e-01f, -4.1910e-01f, 6.9099e-01f, -5.7640e-01f, 

        2.2551e-01f, -1.6919e-01f, -5.3886e-02f, 1.1668e+00f, -5.8027e-01f, -1.0762e+00f, 7.7753e-02f, -6.7799e-01f, 

        5.4291e-01f, -6.6062e-01f, 1.5109e-01f, -2.9981e-01f, 7.0515e-03f, 8.6329e-02f, -5.1182e-01f, -4.5750e-01f, 

        -1.0489e+00f, -4.8170e-01f, 1.7789e+00f, 3.0342e-02f, 1.9365e-01f, -4.8164e-01f, -3.4808e-01f, -9.4457e-01f, 

        -6.0034e-03f, 1.0793e-01f, -1.2110e-01f, 6.1913e-01f, -6.7985e-02f, 6.3926e-02f, -3.2352e-01f, 3.4842e-01f, 

        6.8714e-01f, 4.4758e-02f, -5.7223e-02f, 3.2254e-03f, 2.3840e-01f, 2.7837e-01f, -8.7750e-02f, -3.4330e-01f, 

        -2.3751e-01f, 1.2767e-01f, 3.5641e-01f, -5.7872e-01f, -3.7084e-02f, 2.7001e-01f, 5.5132e-01f, -7.3996e-02f, 

        -1.8163e-01f, -1.0570e-01f, -4.5639e-01f, -6.7334e-02f, 9.0143e-03f, -7.6313e-01f, 1.1708e-01f, -2.0206e-01f, 

        -8.3541e-01f, -1.3567e+00f, -4.5629e-01f, -1.1726e+00f, 3.7055e-01f, 6.9858e-01f, 4.6254e-01f, -9.6558e-01f, 

        -9.9629e-02f, -9.9159e-01f, -1.0887e+00f, -2.8923e+00f, 3.5610e-01f, -2.5744e-01f, 4.1542e-01f, 3.4137e-01f, 

        -1.0583e-01f, -5.1243e-01f, 3.4426e-01f, -1.7261e+00f, -1.5268e+00f, 3.6299e-01f, 4.1485e-01f, -3.0620e-01f, 

        1.2729e-01f, -8.6368e-01f, 6.2955e-01f, 2.7046e-01f, -3.0793e-01f, -1.0753e-02f, 4.9606e-01f, 2.4737e+00f, 

        3.0511e-01f, 3.9760e-01f, 6.4078e-02f, 3.6040e-01f, 5.9258e-01f, -1.0485e-01f, 2.1647e-01f, -3.8519e-01f, 

        1.7604e-01f, 1.0864e+00f, -1.1191e-01f, 4.9026e-01f, -7.2894e-01f, 2.2048e-01f, -2.1742e-01f, 2.1192e-02f, 

        1.3381e-01f, 4.0014e-02f, -1.9230e-01f, 1.1999e+00f, -5.4700e-01f, 6.4821e-01f, 5.9027e-02f, 2.4262e-01f, 

        2.3729e-01f, 4.4892e-01f, -1.0948e-01f, -1.6492e-01f, 6.8344e-01f, 1.5152e+00f, 6.2368e-01f, 1.1482e+00f, 

        9.0687e-01f, 1.2676e-01f, -1.1889e+00f, 2.3977e-01f, 1.7033e-01f, -5.5666e-01f, 9.1595e-02f, -2.6411e-01f, 

        -9.2765e-01f, -4.7284e-01f, -4.1465e-01f, 4.6246e-01f, -4.1212e-01f, 7.2170e-01f, 1.1883e-01f, -1.4220e+00f, 

        -1.1214e-01f, 1.0225e+00f, -2.0612e-01f, 1.1429e-01f, -2.6103e-02f, 6.7053e-01f, 1.6321e-01f, -7.0409e-02f, 

        1.7393e-01f, -5.4280e-01f, -1.7805e-01f, 2.7975e-01f, 3.1385e-02f, 1.2061e+00f, 6.5788e-02f, 4.9106e-01f, 

        -2.6867e-02f, -5.5922e-01f, -2.1849e-02f, -1.1514e+00f, 2.1420e-01f, 1.1490e-02f, -6.4135e-01f, -8.8798e-01f, 

        -1.9549e-01f, 3.9321e-01f, -3.3976e-01f, -9.3871e-01f, -8.8873e-01f, -1.3117e+00f, -1.5195e-01f, 1.1956e+00f, 

        4.6922e-02f, 3.1335e-01f, 2.1713e-01f, 3.7099e-01f, 1.0811e-01f, 6.4339e-01f, 8.9273e-01f, 2.2088e-02f, 

        4.5083e-01f, 6.9176e-01f, 1.1312e-01f, 5.6267e-01f, -6.4765e-03f, 1.2324e-01f, -5.8100e-01f, 1.4058e+00f, 

        3.1322e-01f, 9.8178e-02f, 1.9252e-01f, 9.2077e-01f, -2.0124e-01f, 3.9225e-01f, -3.0243e-01f, 5.1371e-01f, 

        -9.0195e-01f, 1.1448e+00f, 6.3692e-01f, -4.0480e-01f, -1.5196e-01f, -2.5425e-01f, 1.1563e-01f, 8.6110e-01f, 

        8.9019e-02f, -1.5070e-01f, -7.5182e-01f, -1.2657e-01f, 7.4474e-01f, 5.2583e-01f, 1.0535e-01f, 1.7734e-01f, 

        2.6290e-01f, 2.8576e-01f, -4.4782e-01f, 4.9212e-01f, 3.7955e-01f, 4.1943e-01f, 3.8324e-01f, 2.0328e-01f, 

        -8.6941e-02f, 5.3002e-01f, 1.6264e-01f, 3.6733e-01f, -4.5009e-01f, 5.6878e-01f, -4.0357e-02f, 1.3863e-02f, 

        4.7709e-01f, -3.5389e-01f, -4.8354e-01f, -2.1108e-01f, 7.6885e-02f, 4.2692e-01f, 5.7253e-01f, -2.2640e-01f, 

        -1.5643e+00f, -2.9439e+00f, 9.2301e-01f, 3.4743e-01f, -1.1772e-01f, -1.6566e+00f, -1.0534e-01f, -3.3771e-01f, 

        -6.8610e-01f, 1.8096e+00f, 1.0718e-01f, 2.3365e-01f, -3.8016e-01f, 6.8101e-02f, -5.8522e-01f, -2.6525e+00f, 

        -3.0456e-02f, 1.9470e-01f, 1.3002e-01f, 5.9483e-01f, -6.8399e-02f, 6.6991e-01f, -3.4660e-01f, -1.1940e-01f, 

        -9.6033e-01f, -1.8330e+00f, 7.8083e-02f, -3.1516e-02f, 6.0100e-01f, 3.4216e-02f, -1.6941e-01f, -2.8563e-02f, 

        -7.1746e-01f, 4.2989e-01f, 9.2465e-02f, -1.6731e-01f, 2.4484e-01f, 1.3584e-01f, -8.3797e-02f, 2.2075e-01f, 

        4.8496e-01f, 2.9709e-01f, -1.5770e-01f, -2.1764e-02f, 6.5111e-02f, -8.3522e-01f, -4.1045e-01f, 1.3628e+00f, 

        -2.1532e-01f, -4.1189e-01f, 2.3737e-01f, 4.4903e-02f, -6.4739e-01f, -9.1033e-01f, -8.0990e-01f, -1.8421e-01f, 

        7.7060e-01f, 3.5652e-01f, 2.6594e-02f, 2.4253e-02f, 7.9988e-02f, -4.5637e-02f, 2.9046e-01f, 5.6760e-02f, 

        1.9845e-01f, -2.4111e-02f, -3.2097e-01f, -1.1104e-01f, 9.7803e-02f, 1.2547e-01f, -1.9401e-01f, 5.4464e-01f, 

        4.6805e-01f, 2.3825e-01f, 9.0822e-02f, 8.9937e-02f, -1.7599e-01f, 6.2954e-01f, -2.0594e-01f, -1.5348e-04f, 

        7.8071e-03f, 1.5568e-01f, -4.3099e-01f, 4.6198e-02f, 1.7241e-02f, -8.3775e-01f, -6.9990e-02f, 5.7255e-01f, 

        2.4094e-01f, -2.1972e-01f, 6.4055e-02f, -1.1132e-01f, -2.7867e-01f, 2.9706e-01f, 1.6798e-01f, -5.2043e-01f, 

        -3.6551e-01f, -4.3235e-01f, -8.6038e-02f, 1.6011e-01f, 6.1383e-02f, 1.2549e+00f, -9.6952e-01f, 5.2508e-01f, 

        -1.7367e-02f, -1.0040e+00f, 5.6893e-01f, -1.0278e+00f, 8.6485e-01f, 9.1729e-02f, 6.8349e-01f, 8.7491e-01f, 

        2.2819e-01f, -1.5641e+00f, -6.5430e-01f, -2.9673e-02f, 5.8039e-01f, -2.6341e-01f, 9.9633e-01f, -1.8646e-01f, 

        1.3484e-01f, 5.0096e-01f, 1.0697e-01f, -7.4552e-01f, 5.2939e-01f, -1.1553e+00f, 5.1361e-01f, 5.3426e-01f, 

        1.1462e-02f, 6.2269e-01f, 7.4788e-02f, 1.3798e+00f, 4.3221e-01f, -2.5366e-01f, 1.7852e-01f, -1.6452e+00f, 

        1.2114e-01f, 1.3517e-01f, -6.2917e-01f, 6.0361e-02f, 1.9765e-01f, 1.0312e+00f, -5.5053e-01f, -4.8026e-01f, 

        -9.2319e-01f, -8.5389e-01f, 1.8464e-01f, 2.8646e-01f, 2.8475e-01f, 7.7484e-01f, 2.0330e-01f, -1.7942e+00f, 

        4.9621e-01f, 5.9929e-01f, 6.3263e-01f, 2.7543e-01f, 1.2996e-01f, 6.8813e-01f, 5.1621e-01f, 3.6345e-01f, 

        -3.6280e-01f, 6.4774e-01f, -2.7885e-01f, 7.6088e-02f, 4.0330e-01f, -1.0924e+00f, 6.3644e-01f, -3.6635e-01f, 

        -4.8337e-02f, -2.2371e-02f, 5.5076e-01f, 2.8619e-01f, 9.5547e-02f, 1.8930e-01f, 3.5465e-02f, 7.2393e-02f, 

        1.2898e-01f, -4.2443e-01f, 7.8514e-01f, 6.6212e-01f, 9.2961e-02f, -2.9762e+00f, -1.9379e-01f, -1.3411e-01f, 

        2.7607e-01f, 8.4944e-01f, 2.6891e-01f, 3.6941e-01f, 8.3771e-02f, -7.4369e-01f, 5.0089e-01f, -3.5236e-01f, 

        1.2327e-01f, 1.0416e+00f, -8.3345e-02f, -1.1745e+00f, -2.1531e-01f, -5.5592e-02f, -5.9963e-01f, -1.3867e-01f, 

        -4.2334e-02f, -2.6104e-01f, -2.7563e-01f, 1.7340e+00f, 4.4693e-01f, -1.6261e-01f, 7.8383e-01f, -4.5731e-01f, 

        -3.8861e-01f, 3.1243e-01f, -3.2675e-01f, -1.4821e+00f, -2.9476e-01f, 8.1517e-01f, 5.4692e-03f, -1.4595e-01f, 

        1.9553e-01f, -6.0160e-02f, -1.3226e-01f, 3.0296e-01f, -1.1015e-01f, -2.1294e-01f, -2.4255e-01f, -6.6629e-03f, 

        -6.2982e-02f, -1.5389e-01f, -4.0922e-02f, 2.8691e-01f, -9.5991e-02f, 2.9581e-02f, 2.0381e-02f, -1.7772e-01f, 

        -4.3152e-01f, -1.4090e-01f, -7.1294e-02f, -5.8664e-02f, 1.1899e-02f, -8.0737e-02f, -1.0342e-01f, -8.2162e-02f, 

        -2.2527e-01f, -1.2211e-01f, -9.3211e-02f, -8.0541e-02f, -1.0343e-01f, 3.2858e-01f, -1.0449e-01f, 2.5746e-02f, 

        -1.6156e-01f, 2.5368e-01f, -5.2738e-02f, 7.0124e-03f, -7.3288e-02f, 3.3675e-02f, -3.9944e-02f, -9.5475e-02f, 

        -2.8225e-02f, -1.7516e-02f, -5.7869e-02f, 1.7315e-02f, 3.0915e-02f, -1.0794e-01f, -9.1629e-02f, -3.6665e-02f, 

        -5.7560e-02f, -2.9534e-01f, -8.7146e-02f, -1.1930e-01f, -1.1926e-01f, -7.8939e-02f, 9.1645e-02f, 5.5197e-02f, 

        -9.4873e-02f, 1.2197e-02f, 1.7852e-02f, -1.0297e-01f, -5.3424e-02f, -5.3903e-02f, -7.0073e-02f, -6.7378e-02f, 

        -5.6445e-02f, -9.7062e-02f, -1.5158e-01f, 1.6407e-01f, -3.6404e-02f, -6.9875e-02f, -9.0304e-03f, -6.3625e-02f, 

        3.4333e-02f, -5.0068e-02f, -1.0495e-01f, 9.9629e-04f, -5.4242e-02f, -1.6594e-02f, -6.7268e-02f, 2.0742e-02f, 

        -5.6880e-02f, -5.8072e-02f, 4.2003e-02f, -1.2340e-01f, 2.0924e-01f, -6.5856e-02f, -4.1609e-03f, 1.6311e-01f, 

        3.1643e-01f, 3.6219e-01f, -2.4179e-01f, -6.3796e-01f, 1.4279e-01f, 3.8834e-01f, 3.4404e-01f, -1.0437e+00f, 

        4.9532e-02f, 4.3406e-01f, 3.1060e-01f, 6.5707e-02f, 2.1143e-01f, 2.5478e-01f, -5.9500e-01f, 2.4891e-02f, 

        4.8240e-01f, -6.4593e-01f, 1.2818e-01f, -9.1878e-01f, -2.1786e-01f, 2.5153e-01f, 6.3473e-01f, -1.1450e-01f, 

        5.3792e-02f, -5.1869e-01f, 2.8958e-01f, 7.7299e-01f, 8.7621e-01f, -8.1670e-01f, 1.5027e+00f, 1.7564e+00f, 

        -6.3245e-01f, -1.9970e-01f, 6.3494e-01f, 9.8330e-02f, 3.0804e-01f, -5.1953e-01f, 1.7467e-01f, 1.2323e+00f, 

        -4.7104e-01f, 1.5450e+00f, -7.8612e-01f, 2.1664e-01f, -2.1222e-01f, 7.6798e-01f, -7.6678e-02f, -4.1406e-01f, 

        3.1515e-01f, -9.9536e-02f, -5.2896e-01f, 5.1752e-01f, 1.5717e-01f, -1.3866e+00f, 2.8812e-01f, -1.7048e+00f, 

        2.7921e-02f, 1.4857e+00f, 1.2096e-01f, 2.6573e-01f, 5.0405e-01f, 1.8857e+00f, 5.2597e-01f, 1.8813e-01f, 

        7.8743e-03f, -1.7218e+00f, -5.9794e-01f, -4.2407e-01f, 2.5720e-01f, 1.6591e+00f, -6.3471e-02f, 5.1978e-01f, 

        2.2044e-01f, -3.4956e-01f, 7.1670e-01f, -1.9579e+00f, 1.3133e-01f, 1.0519e+00f, 3.3241e-01f, -3.8731e-01f, 

        2.4354e-01f, 1.2810e+00f, 3.7442e-01f, 4.5925e-01f, 3.2074e-01f, 2.6466e-01f, 9.5343e-01f, 5.5474e-01f, 

        4.5149e-01f, 1.1157e+00f, 6.4074e-01f, 4.6249e-01f, -2.2291e-01f, -4.5420e-01f, -1.8919e-01f, 6.7560e-01f, 

        -6.1931e-02f, -3.6436e-01f, 7.1074e-01f, 4.0770e-01f, 1.1984e+00f, -3.5194e-01f, -5.7465e-01f, 1.9815e-01f, 

        2.9260e-01f, -4.6623e-01f, 2.2621e-01f, -2.1589e-01f, -1.9917e+00f, -8.3193e-01f, -6.3127e-01f, -2.8050e-01f, 

        4.8591e-01f, -4.6902e-02f, -2.6812e-02f, -2.8386e-01f, 5.8594e-01f, 1.4316e-01f, 1.7244e+00f, -1.5494e+00f, 

        -2.4235e-01f, 3.3018e-01f, -4.4085e-01f, -4.5865e-01f, 3.2803e-01f, -2.0420e-01f, 3.0287e-01f, -8.1439e-01f, 

        5.9539e-01f, 2.2191e-01f, 1.4101e-01f, -2.5939e-01f, 1.3034e-01f, 2.5646e-01f, 4.0973e-01f, -3.3700e-02f, 

        1.3828e-02f, 6.5363e-02f, -3.0942e-01f, 1.5549e-01f, -1.2775e-01f, -1.9651e-01f, -1.4096e-01f, -4.0177e-01f, 

        -1.2728e-01f, 1.2956e-01f, -2.5637e-01f, 7.7883e-01f, 6.2416e-01f, -1.0117e-02f, 4.6702e-01f, -8.8243e-03f, 

        4.9819e-02f, 3.7357e-01f, 7.0786e-01f, -1.0338e+00f, 7.3969e-01f, -1.2188e-01f, -3.7235e-01f, 1.1197e+00f, 

        -8.2261e-01f, 1.5067e-02f, 4.0922e-01f, 1.6844e-01f, 5.4615e-01f, -7.2531e-01f, -6.3076e-03f, 2.0408e-01f, 

        2.7876e-01f, 1.0522e+00f, 4.7637e-01f, -2.7373e-01f, -5.0926e-02f, -4.8607e-01f, -2.3475e-01f, -6.4492e-01f, 

        -5.4603e-01f, 1.4857e+00f, 1.2117e-02f, 2.1878e-01f, -3.9100e-01f, 5.6805e-01f, 2.5746e-02f, 8.0767e-01f, 

        1.7103e-01f, 1.0979e-01f, 5.1676e-01f, -9.3402e-01f, 3.9893e-01f, 6.2045e-01f, -4.4756e-01f, -3.7142e-01f, 

        -2.4292e-01f, 8.5324e-01f, -1.2238e-01f, -2.3662e-01f, 1.8745e-01f, -1.8649e-01f, 9.9963e-02f, -4.2186e-02f, 

        -9.5275e-03f, -3.9213e-01f, 4.6625e-01f, -2.6646e-01f, -9.8373e-02f, 4.2573e-01f, 4.8495e-01f, -1.6627e-01f, 

        -4.3039e-01f, -1.9498e-01f, -4.7257e-02f, -4.3541e-01f, 2.0211e-01f, 6.8079e-01f, 4.3731e-02f, 5.4853e-02f, 

        2.9553e-01f, 1.6998e-02f, -1.8019e-01f, 1.7700e-01f, 2.0735e-02f, -7.5849e-01f, 4.1323e-01f, 2.8296e-01f, 

        4.2914e-01f, 4.7300e-01f, 1.1865e-01f, 3.0416e-01f, -4.9306e-01f, 2.4541e-01f, 4.4051e-01f, -4.7670e-01f, 

        3.3633e-01f, 3.9117e-01f, -2.5684e-01f, 3.9206e-01f, -2.8721e-01f, -8.0324e-02f, -2.1074e-01f, -1.4408e-01f, 

        3.8016e-01f, 2.9941e-01f, 3.9025e-03f, 5.8447e-02f, 4.1504e-01f, 2.6304e-01f, 5.9497e-01f, 3.8649e-01f, 

        -4.3382e-01f, -5.2646e-01f, -6.1004e-01f, -5.1004e-02f, 9.6063e-02f, -1.0742e-01f, 2.3493e+00f, -2.4926e-01f, 

        -3.4222e-01f, 2.9833e-02f, -1.2609e-01f, -1.2953e+00f, -5.3731e-01f, -6.1004e-02f, 9.3215e-01f, -3.7658e-02f, 

        1.0273e-01f, -3.4356e-01f, -5.4238e-01f, 1.1089e-01f, 2.1017e-01f, -2.0860e-02f, 4.4774e-01f, 4.7866e-01f, 

        -5.6500e-01f, 2.0168e-01f, 7.8398e-02f, -1.3658e-02f, -9.2491e-01f, -9.6825e-01f, 5.8400e-01f, -6.4174e-01f, 

        -3.4144e-02f, -6.5306e-01f, -3.1494e-02f, 4.8583e-01f, -3.5675e-01f, -1.8753e-01f, -3.6957e-01f, -9.7255e-01f, 

        7.3185e-01f, 1.4151e-01f, 4.2130e-01f, -2.6619e-01f, -2.4564e-01f, -8.5086e-01f, -1.1633e-03f, -5.6521e-02f, 

        5.3569e-01f, 3.7263e-01f, 6.8414e-02f, -4.3725e-01f, 2.7546e-01f, -4.7369e-01f, 2.6968e-01f, -1.5700e-01f, 

        -1.5542e-01f, 1.3278e-01f, -3.1167e-01f, -4.6462e-01f, 1.2073e-01f, 1.7505e-01f, 2.0074e-01f, -2.5939e-01f, 

        1.5616e-01f, 5.4350e-01f, -2.2640e-02f, -3.5715e-01f, -1.9371e-01f, 3.3734e-01f, 2.4521e-01f, -7.0108e-02f, 

        5.6448e-02f, -8.4231e-01f, -1.1630e-01f, -4.1379e-01f, -4.5971e-01f, -6.3786e-01f, 1.2629e+00f, 8.8004e-01f, 

        3.2887e-01f, -5.9051e-01f, 5.9033e-01f, -4.7137e-01f, -4.0669e-02f, 6.3796e-03f, -2.5336e-01f, 4.2573e-01f, 

        -1.1433e+00f, -8.6734e-01f, -1.7765e-01f, 2.4979e-01f, 6.1528e-01f, -5.4049e-01f, -5.7378e-01f, 1.2622e-02f, 

        2.7833e-01f, -2.9972e-02f, -2.4460e-01f, -1.5732e+00f, 2.1214e-01f, -3.9069e-01f, 7.0415e-01f, -8.9600e-01f, 

        -4.3719e-01f, -1.1645e-01f, -4.3367e-01f, -5.8121e-01f, 6.9750e-01f, -4.2625e-01f, 1.4656e-01f, 2.5259e-02f, 

        2.7606e-01f, 9.2412e-01f, 3.8889e-02f, 2.4756e-01f, 5.2028e-01f, 4.5147e-02f, -2.3569e-01f, -9.0941e-02f, 

        2.8772e-02f, 2.9537e-01f, -1.3691e-02f, -5.8300e-01f, -3.9810e-01f, -5.6979e-01f, 2.7574e-01f, 1.1758e-01f, 

        3.0073e-01f, 9.1206e-01f, -5.9147e-01f, -4.0437e-01f, -9.1024e-01f, -1.0783e-02f, -3.6116e-01f, -3.4580e-01f, 

        6.8987e-01f, -1.1438e+00f, 2.6310e-01f, 4.3628e-01f, 2.7715e-01f, 5.3490e-01f, 1.8102e-01f, 5.4245e-01f, 

        1.3630e-01f, 2.2549e-01f, 1.4717e-01f, 1.1046e-01f, 3.0270e-01f, 6.0035e-01f, 1.2243e-01f, 3.9034e-01f, 

        4.5515e-01f, 3.7325e-01f, 1.3775e-01f, 2.2488e-01f, 1.1654e-01f, -5.8458e-01f, 3.3934e-01f, 4.7933e-01f, 

        3.9555e-02f, 8.1777e-01f, -5.3198e-01f, 2.3436e-01f, -1.0843e-01f, 5.5337e-03f, 4.3563e-01f, 2.7807e-02f, 

        2.8701e-01f, 6.3326e-01f, -3.0264e-01f, 1.0604e+00f, 4.8994e-02f, -9.0167e-02f, 1.4401e-01f, 2.4997e-02f, 

        2.6474e-01f, -1.5607e-01f, 5.1076e-01f, 4.7832e-01f, 4.2402e-01f, 1.4557e+00f, 4.2910e-01f, 2.2024e-01f, 

        3.2296e-01f, -3.8341e-01f, 5.6008e-01f, -1.2944e-01f, 3.3138e-01f, 5.9222e-01f, -1.5592e-01f, 5.0354e-01f, 

        1.3482e-01f, -4.9619e-02f, -8.0461e-02f, 7.6482e-02f, 2.7665e-01f, 3.5159e-01f, 3.0059e-01f, 8.0251e-01f, 

        2.2892e-01f, 6.2180e-01f, 3.1906e-01f, 3.3912e-01f, 7.2198e-01f, 4.2356e-01f, 2.9618e-01f, 1.9481e-01f, 

        -4.4753e-01f, -8.1408e-02f, -3.2555e-01f, -3.8511e-01f, 4.3764e-01f, -2.2235e-01f, -3.8273e-01f, 6.7471e-02f, 

        4.8143e-01f, 1.0517e-01f, -5.9466e-02f, -1.0820e-01f, 7.7988e-02f, 2.5915e-01f, 5.1298e-01f, 2.5025e+00f, 

        6.0215e-01f, 5.7967e-01f, 4.9562e-01f, -9.9895e-03f, 5.8965e-01f, -7.4732e-01f, 5.8696e-01f, 6.6896e-01f, 

        -9.2499e-02f, 2.5294e+00f, -1.0266e-01f, 4.5183e-01f, 8.3660e-01f, -4.2164e-01f, -5.4621e-01f, 1.2564e+00f, 

        -5.8225e-02f, 2.5674e-01f, -8.4683e-02f, 8.1993e-01f, 5.5074e-01f, -5.2286e-01f, 8.0651e-02f, 5.9883e-01f, 

        -2.7290e-01f, -1.3576e-01f, 2.4216e-01f, 9.9291e-02f, 3.2351e-01f, 3.5909e+00f, -9.4812e-01f, 1.3310e-01f, 

        7.0819e-01f, 6.3556e-01f, 3.6520e-01f, 1.2884e-01f, -3.7819e-01f, 9.8213e-01f, 2.6805e-01f, 7.4512e-01f, 

        6.2370e-01f, -1.1857e-02f, 1.0641e-01f, -1.9843e-01f, 1.8729e-01f, 3.9120e-01f, -1.1652e-01f, -1.7983e-01f, 

        -1.4525e-01f, 6.5994e-01f, 5.4013e-02f, -4.8124e-02f, 1.1296e-01f, 6.9642e-01f, 2.6008e-01f, -3.4467e-01f, 

        1.8299e-01f, -4.4727e-02f, -4.1170e-02f, 9.3321e-01f, -2.5800e-01f, 2.4878e-01f, 1.7724e-01f, 3.7966e-01f, 

        6.5295e-01f, -5.1183e-01f, 8.5579e-03f, 8.6180e-01f, 8.9104e-01f, 3.4727e-01f, -5.3488e-02f, -2.1826e-01f, 

        -5.4079e-02f, 3.5797e-01f, 8.6730e-02f, 4.8170e-01f, -1.8601e+00f, 6.7147e-01f, -7.4397e-01f, -1.0814e-01f, 

        3.9267e-01f, 6.3407e-01f, -1.0228e-01f, -6.7007e-01f, 1.5046e-01f, 5.3300e-01f, -6.3064e-01f, 5.4951e-01f, 

        6.4573e-01f, 2.1246e+00f, 1.1405e-01f, 6.8212e-01f, 2.2086e-01f, -5.0929e-01f, -4.5815e-03f, -4.7065e-01f, 

        4.7169e-01f, 1.0832e+00f, 4.1884e-01f, 8.7513e-01f, 4.1217e-02f, 2.0475e-01f, 1.1727e-01f, -1.5081e-01f, 

        6.2812e-01f, 9.0972e-01f, 5.0164e-01f, 2.4829e-02f, 4.2423e-01f, 1.3116e+00f, 4.5588e-01f, 7.9509e-01f, 

        4.9318e-01f, -2.5245e-02f, 4.6238e-01f, -2.1932e-01f, 1.6925e-01f, 2.9413e-01f, 4.2688e-01f, 1.0533e+00f, 

        6.6719e-01f, 1.5259e-01f, 5.0864e-01f, 7.2512e-01f, 1.8370e-01f, 4.9188e-01f, -8.3164e-01f, 1.2972e+00f, 

        5.2020e-01f, 4.7589e-01f, 2.9588e-01f, 5.2344e-01f, 1.7355e-01f, -1.0854e-01f, -2.4071e-02f, 4.0458e-01f, 

        3.5949e-01f, -2.8197e-01f, -1.0632e-01f, 1.0133e+00f, -1.4320e-02f, 1.4315e+00f, 4.9742e-01f, 6.1581e-01f, 

        7.3110e-01f, -5.6038e-01f, 3.3218e-01f, -5.6600e-01f, -3.6828e-01f, 6.3061e-01f, 1.3510e-01f, -3.7310e-02f, 

        -1.4362e-01f, 1.0152e+00f, -2.1624e-01f, -1.5058e+00f, 4.8695e-01f, 1.2072e+00f, 3.0170e-01f, -1.2429e+00f, 

        5.2365e-01f, 1.2677e-01f, -1.0627e-01f, 1.5435e-01f, 2.9328e-01f, -1.7337e-01f, 1.2505e-01f, 4.6148e-02f, 

        9.2850e-02f, -3.2185e-01f, -1.8173e-01f, 2.8637e-01f, -1.4422e-01f, 7.7957e-01f, 2.5240e-02f, 5.9223e-01f, 

        -1.4971e+00f, 4.2410e-01f, 8.2259e-02f, 3.6299e-01f, 2.6816e-01f, -7.1709e-01f, 6.6392e-01f, 4.1473e-01f, 

        4.8705e-01f, 3.2751e-01f, 1.9134e-01f, -3.9477e-02f, 6.1980e-01f, 4.5708e-01f, 2.7284e-01f, 2.3963e-01f, 

        3.2321e-01f, 4.7349e-01f, 2.0222e-01f, 2.1195e-01f, -4.3241e-01f, 1.1272e-01f, -1.2758e-01f, 2.7723e-01f, 

        2.9303e-01f, 1.4501e-01f, 2.3382e-01f, -4.8221e-01f, 6.9916e-01f, 3.2466e-01f, 4.9928e-01f, -1.0067e-01f, 

        -1.1510e-01f, 2.4070e-01f, -9.2013e-01f, -3.6919e-02f, 2.8754e-01f, 1.1502e-01f, -2.2216e-01f, 1.3680e+00f, 

        -8.3392e-02f, 2.4829e-01f, 2.5528e-03f, 4.1173e-01f, 2.2686e-01f, 4.9588e-01f, 4.9234e-01f, 5.9353e-01f, 

        3.2281e-01f, 2.4342e-01f, 1.7852e-01f, -1.5060e-01f, 1.1776e-01f, -7.5987e-01f, 2.9893e-01f, -5.8531e-01f, 

        1.5553e-01f, 1.7486e-01f, 5.2140e-01f, 1.0601e+00f, 4.2134e-01f, 4.7368e-01f, 2.3708e-01f, -1.9378e-01f, 

        4.0330e-01f, -3.3088e-01f, 1.5181e-01f, -4.4583e-01f, -3.6284e-02f, 8.1219e-01f, 3.1880e-01f, -4.8538e-02f
    };

    // Convert and store feedforward weights
    for (int j = 0; j < 256; j++) {
        float scaled = conv1_weights_vector[j] / scale;
        arm_float_to_q15(&scaled, &weights1[j], 1);
    }

    for (int j = 0; j < 1152; j++) {
        float scaled = conv2_weights_vector[j] / scale;
        arm_float_to_q15(&scaled, &weights2[j], 1);
    }

    for (int i = 0; i < 1440; i++) {
        float scaled = fc3_weights_vector[i] / scale;
        arm_float_to_q15(&scaled, &weights3[i], 1);
    }

}

void SNN_Init(void) {
    const float scale = 360.0f;

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
    LIFNeuron_Conv2d_Update_Subtract_Base(layer1, input_spikes, weights1, l1_spikes, 10, 10, 2, 7, 7, 8, 4, 4, 1, 0);

    // Layer 2 (convolutional)
    LIFNeuron_Conv2d_Update_Subtract_Base(layer2, l1_spikes, weights2, l2_spikes, 7, 7, 8, 3, 3, 16, 3, 3, 2, 0);

    // Layer 3 (no recurrent, fully connected)
    LIFNeuron_Layer_Update_Subtract_NoRecurrent(layer3, l2_spikes, weights3, NUM_NEURONS_LAYER2, NUM_NEURONS_LAYER3, l3_spikes, 0);

    // Copy output spikes
    for (int i = 0; i < NUM_NEURONS_LAYER3; i++) {
        output_spikes[i] = l3_spikes[i];
    }
}

void SNN_Reset_State(void) {
    // Reset layer 1
    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        layer1[i].membrane_potential = 0;
        layer1[i].reset_value = 0;
        l1_spikes[i] = 0;
    }

    // Reset layer 2
    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        layer2[i].membrane_potential = 0;
        layer2[i].reset_value = 0;
        l2_spikes[i] = 0;
    }

    // Reset layer 3
    for (int i = 0; i < NUM_NEURONS_LAYER3; i++) {
        layer3[i].membrane_potential = 0;
        layer3[i].reset_value = 0;
        l3_spikes[i] = 0;
    }

}
