#include "stm32h7xx_hal.h"
#include "../Inc/lif_neuron_gen.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "../Inc/usart.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Network architecture from NIR
// Input size: 12
// Layers: 2
// Layer 0: 12 -> 38 (fully connected, with recurrent, uniform params)
// Layer 1: 38 -> 7 (fully connected, no recurrent, uniform params)

// Global variables for the SNN
#define NUM_INPUTS 12
#define NUM_NEURONS_LAYER1 38
#define NUM_NEURONS_LAYER2 7

static LIFNeuron layer1[NUM_NEURONS_LAYER1], layer2[NUM_NEURONS_LAYER2];
static q7_t l1_spikes[NUM_NEURONS_LAYER1], l2_spikes[NUM_NEURONS_LAYER2];
static q7_t l1_spikes_prev[NUM_NEURONS_LAYER1];

static q15_t weights1[NUM_INPUTS*NUM_NEURONS_LAYER1]; // Fully connected
static q15_t weights2[NUM_NEURONS_LAYER1*NUM_NEURONS_LAYER2]; // Fully connected
static q15_t recurrent_weights1[NUM_NEURONS_LAYER1]; // Recurrent 1-to-1 (vector)

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


void Load_NIR_Weights(void) {
    const float scale = 60.0f;

    // Layer 1 feedforward weights - fully connected (12x38)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    float fc1_weights_vector[456] = {
        -5.5690e-01f, -2.8890e-01f, -2.1405e-01f, -5.7090e-01f, -7.4180e-02f, -1.0214e+00f, -6.0636e-01f, -2.0222e-01f, 

        -1.1668e+00f, -2.0937e+00f, 1.9430e+00f, 4.6073e-01f, -1.1619e+00f, -2.6601e-01f, -1.1195e+00f, 2.9295e-01f, 

        -3.9575e-01f, -5.5226e-01f, -4.3371e-01f, -1.4543e+00f, -1.3989e+00f, -6.1552e+00f, -5.2681e-01f, -1.8968e+00f, 

        -9.2186e-01f, -6.0718e-01f, -2.9490e-01f, -2.3934e+00f, -1.2402e+00f, -4.5867e-01f, -1.1935e+00f, -6.7340e-01f, 

        -5.8978e-01f, -2.1939e+00f, -7.7409e-01f, -7.1873e-01f, 1.1458e-01f, 3.5302e-01f, -6.0994e-01f, -5.4818e-01f, 

        -1.7640e-01f, -5.6129e-01f, 6.3147e-02f, -1.6277e+00f, -2.2878e+00f, -9.0930e-01f, -2.6760e+00f, -2.2810e+00f, 

        -1.5674e+00f, 6.7959e-01f, -3.9391e-02f, -5.5608e-01f, -2.1866e+00f, -1.1152e-02f, 5.8115e-01f, -1.7566e+00f, 

        -1.2004e+00f, -1.7249e+00f, -3.1182e+00f, 1.5336e-01f, -4.9776e-01f, -2.4763e+00f, -2.1488e+00f, -1.2272e+00f, 

        4.2357e-01f, -2.1104e+00f, -7.8661e-01f, -5.0814e-01f, -1.8133e+00f, -9.0110e-01f, -3.8213e-01f, -1.0370e+00f, 

        -5.8494e-01f, -1.4968e+00f, -2.0628e-01f, -1.1185e+00f, -8.4994e-01f, -3.8449e-01f, -3.5380e-01f, -7.3638e-01f, 

        1.5910e-01f, 6.2544e-01f, -2.2740e+00f, -5.5904e-01f, -2.6173e+00f, -1.2985e+00f, -3.1023e-01f, -7.1062e-01f, 

        5.6876e-01f, -5.5385e-01f, -1.9422e+00f, -7.6066e-02f, -1.6190e-01f, -1.6631e+00f, -8.7874e-01f, -1.4633e+00f, 

        -2.8914e+00f, -1.4412e+00f, -7.5561e-01f, -1.8899e+00f, -2.2449e+00f, -1.2347e+00f, 9.5876e-02f, -2.0989e+00f, 

        -1.1869e+00f, -4.5700e-01f, -2.0036e+00f, -1.0641e+00f, -1.1197e+00f, -8.6228e-01f, -7.7306e-01f, -1.6318e+00f, 

        9.3500e-02f, -8.3452e-01f, -5.8630e-01f, -5.2844e-01f, -1.1904e+00f, -7.6404e-01f, 1.0498e-01f, -6.9509e-01f, 

        -2.1791e+00f, -9.8904e-01f, -2.7408e+00f, -1.9944e+00f, -1.6851e+00f, 3.2264e-01f, -3.1618e-01f, -7.4320e-01f, 

        -1.6304e+00f, 2.8710e-01f, -1.1888e+00f, -1.6178e+00f, -1.1243e+00f, -5.8592e-01f, -2.7841e+00f, -3.1400e-02f, 

        -7.0487e-01f, -2.7620e+00f, -2.0082e+00f, -1.1896e+00f, 1.6894e-01f, -2.3896e+00f, -1.1210e+00f, -3.3152e-01f, 

        -1.6861e+00f, -8.9291e-01f, -8.7399e-01f, -1.2707e+00f, -6.7456e-01f, -7.1089e-01f, 3.8579e-01f, -1.6404e-01f, 

        -7.1893e-01f, -2.0309e-01f, 3.7145e-01f, -6.4783e-01f, 3.1556e-01f, 1.4840e-01f, -1.7632e+00f, -7.8297e-01f, 

        -2.2657e+00f, -1.9330e+00f, 1.0572e+00f, -2.9692e-01f, -2.4937e-02f, -3.2443e-01f, -1.3157e+00f, 1.6927e-01f, 

        -1.3653e-01f, -1.2930e+00f, -1.1604e+00f, -6.0231e-01f, -2.7974e+00f, 3.7975e-01f, -7.0969e-01f, -2.4452e+00f, 

        -2.1656e+00f, -1.1601e+00f, -1.0805e+00f, -2.1047e+00f, -1.3929e+00f, -2.6581e-01f, -1.2367e+00f, -5.4657e-01f, 

        -3.0590e-01f, -1.1743e+00f, -1.8291e-01f, -3.6644e-01f, -5.9578e-02f, -2.9143e-01f, -7.0377e-01f, -4.5618e-01f, 

        3.0174e-01f, -7.1305e-01f, 4.3879e-01f, -5.5066e-01f, -1.3653e+00f, -5.1142e-01f, -2.2811e+00f, -2.5391e+00f, 

        -1.7925e-01f, -3.3531e+00f, -2.5062e-01f, -4.8802e-01f, -1.6500e+00f, 6.0043e-01f, -9.1117e-01f, -9.8157e-01f, 

        -5.7110e-01f, -5.4383e-01f, -2.8184e+00f, -2.2274e+00f, -8.8377e-01f, -3.1004e+00f, -2.6691e+00f, -6.1164e-01f, 

        6.8464e-02f, -2.1825e+00f, -1.2026e+00f, -3.4393e-01f, -1.2985e+00f, -5.3662e-01f, -8.6496e-01f, -5.9722e-01f, 

        -4.7542e-01f, -9.9414e-01f, 3.3459e-02f, -1.3204e+00f, -9.3269e-01f, -6.9260e-01f, 7.3450e-01f, -6.5787e-01f, 

        -5.9134e-01f, 2.0041e-01f, -1.6839e+00f, -7.2142e-01f, -2.6214e+00f, -2.2680e+00f, -2.5822e-01f, 2.3161e-01f, 

        -1.9483e-01f, -6.3830e-01f, -2.2823e+00f, -1.3476e-01f, -5.0158e-01f, -1.0768e+00f, -1.3242e+00f, -1.0456e+00f, 

        -3.1867e+00f, 7.5208e-01f, -6.4193e-01f, -2.3046e+00f, -1.9868e+00f, -1.3369e+00f, -8.7991e-03f, -9.5720e-01f, 

        -1.2664e+00f, -8.6846e-01f, -1.2259e+00f, -4.6917e-01f, -8.2900e-01f, -1.0060e+00f, -4.3469e-01f, -1.5140e+00f, 

        -7.8164e-01f, -1.1144e+00f, -5.4329e-01f, -3.3027e-01f, -1.8824e-01f, -4.8544e-01f, 1.9959e-01f, -2.0791e+00f, 

        -8.0572e-01f, -6.3257e-01f, -2.3889e+00f, -2.8583e+00f, -5.7592e-01f, -8.5910e-02f, -2.1531e+00f, -7.9434e-01f, 

        -1.7144e+00f, -1.4311e+00f, -4.4937e-01f, -7.4168e-01f, -9.6037e-01f, -6.6027e-01f, -2.4591e+00f, 3.2860e-02f, 

        -8.5244e-01f, -1.8662e+00f, -2.0170e+00f, -6.7967e-01f, -2.9960e-03f, -1.8330e+00f, -1.7744e+00f, -4.2606e-01f, 

        -2.1368e+00f, -7.0024e-01f, -5.3293e-01f, -8.8588e-01f, -4.8672e-01f, -1.2797e+00f, 4.2402e-01f, -6.4183e-01f, 

        -4.7713e-01f, -4.8135e-01f, -1.6739e+00f, -5.8510e-01f, 6.5730e-01f, -2.1063e+00f, -1.4863e+00f, -2.1828e-01f, 

        -1.7516e+00f, -2.1255e+00f, 6.4668e-01f, 7.1930e-01f, -4.4473e-01f, -4.1034e-01f, -2.1033e+00f, -2.3308e+00f, 

        -1.0964e+00f, -1.6969e+00f, -1.4107e+00f, -7.1978e-01f, -2.5304e+00f, -8.7836e-01f, -2.7749e-01f, -2.6337e+00f, 

        -1.8409e+00f, -9.0152e-01f, -5.8432e-01f, -2.0800e+00f, -2.3599e+00f, -2.3816e-01f, -2.5323e+00f, -4.0191e-01f, 

        -1.0216e-01f, -1.2156e+00f, -3.9979e-01f, -8.5081e-01f, 3.4155e-01f, -5.4823e-01f, -7.2443e-01f, -3.9475e-01f, 

        -8.2040e-02f, -6.5030e-01f, 2.5059e-01f, -1.4609e-01f, -1.9856e+00f, -5.3992e-02f, -1.5376e+00f, -1.3873e+00f, 

        4.3475e-01f, -7.0782e+00f, 7.8616e-01f, -4.3667e-01f, -1.3832e+00f, -7.8360e-01f, -4.8726e-01f, -8.5190e-01f, 

        -2.4058e-01f, -1.3583e+00f, -1.1063e+00f, -6.3772e+00f, 2.1350e-02f, -2.4262e+00f, -8.9117e-01f, -5.7565e-01f, 

        -4.4774e-01f, -1.9683e+00f, -2.2637e+00f, -2.6030e-01f, -1.2223e+00f, -7.7626e-01f, -2.2525e-01f, -2.1065e+00f, 

        -1.6715e-01f, -4.6290e-01f, -1.0661e+00f, 1.3231e+00f, -6.7298e-01f, -6.9549e-01f, -2.7164e-02f, -7.1832e-01f, 

        3.9887e-02f, 1.9298e-01f, -1.4070e+00f, -1.3487e+00f, -2.9643e+00f, -2.1794e+00f, -6.0816e-01f, -4.0180e-01f, 

        3.0973e-01f, -8.4792e-01f, -1.4632e+00f, -4.5638e-02f, 3.5254e-02f, -1.4217e+00f, -6.4302e-01f, -2.0599e+00f, 

        -2.9465e+00f, -7.3801e-02f, -6.1246e-01f, -3.0153e+00f, -2.3278e+00f, -5.9012e-01f, 8.6044e-02f, -1.0602e+00f, 

        -4.8646e-01f, -7.4671e-01f, -7.7037e-01f, -7.5531e-01f, -6.9232e-01f, -1.8108e+00f, -1.0484e+00f, -1.0036e+00f, 

        -2.1127e-01f, -5.5754e-01f, -5.6965e-01f, -2.1121e-01f, 5.6972e-02f, -6.8914e-01f, -1.4646e+00f, -3.7315e-01f, 

        -1.6373e+00f, -7.0686e-01f, -1.6550e+00f, -2.5806e+00f, -5.1593e-02f, -7.3552e-01f, -6.0587e-01f, -2.4756e-01f, 

        -1.6420e+00f, -1.5937e+00f, 3.1147e-01f, -1.2333e-01f, -3.6115e-01f, -1.4712e+00f, -1.9499e+00f, -1.7860e-01f, 

        -7.3909e-01f, -1.0203e+00f, -9.4322e-01f, -4.1742e-01f, -2.8041e-02f, -2.0901e+00f, -1.2600e+00f, -6.4736e-02f, 

        -9.6454e-01f, -1.5061e-01f, -6.5929e-01f, -2.0627e+00f, -8.6765e-01f, -6.7777e-01f, 9.9276e-01f, 7.4768e-01f
    };

    // Layer 2 feedforward weights - fully connected (38x7)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    float fc2_weights_vector[266] = {
        3.3325e-03f, -6.4183e-02f, -2.7025e-02f, 1.6197e-01f, 4.6680e-02f, -8.0338e-03f, 4.7617e-02f, 9.7057e-02f, 

        1.0866e-01f, -1.3965e-01f, 1.4518e-01f, -6.3540e-02f, 1.7445e-01f, 1.2116e-01f, -2.9155e+00f, 9.9319e-01f, 

        1.2164e+00f, 6.3470e-01f, 2.2856e-01f, 7.8904e-01f, -6.4608e-02f, 1.4363e-01f, -4.7870e-02f, 1.4466e-01f, 

        -1.3111e-02f, -8.1130e-02f, -5.5798e-02f, 4.6848e-02f, 1.5444e-01f, -2.6290e-03f, 7.7163e-01f, 8.3092e-01f, 

        8.7321e-01f, 1.2171e-01f, -3.3576e+00f, -1.6507e-01f, 1.0926e+00f, -9.6649e-01f, 1.4106e-01f, -1.3187e+00f, 

        -1.3672e-01f, -8.8751e-01f, -4.8085e-03f, -1.6225e-01f, 1.5191e-02f, 1.5590e-01f, 6.4600e-02f, 8.1951e-02f, 

        -2.1985e-01f, -3.3040e-02f, 9.3389e-02f, 1.2698e-01f, 5.8234e-02f, 4.5172e-02f, 1.2389e-01f, -1.1190e-01f, 

        2.7584e-02f, -9.8483e-02f, -8.0035e-02f, 2.1954e-02f, 3.2610e-02f, 2.7244e-02f, 1.1105e-01f, 1.4758e-01f, 

        -1.4276e-03f, -1.3053e-01f, -1.8645e-01f, 7.8604e-02f, 1.1627e-01f, -1.9078e-01f, 7.7509e-01f, -9.7783e-01f, 

        -6.0152e+00f, 5.1547e-01f, 7.0279e-01f, 6.2438e-01f, 3.3677e-01f, -1.2469e+00f, 1.1102e-01f, 4.2231e-01f, 

        6.0231e-01f, 2.6509e-01f, -6.9215e-02f, 2.2072e+00f, -7.3668e-01f, 1.1177e+00f, -2.0823e-01f, -1.6812e-01f, 

        2.5388e-01f, -5.1700e-01f, 1.1333e+00f, -2.5417e-02f, -2.6494e-02f, 1.9184e-01f, 3.0587e-02f, 1.0284e-01f, 

        1.4897e-01f, 1.1907e-01f, -3.5642e-02f, 9.3953e-02f, 2.7819e-02f, -1.3470e-01f, -1.2037e-01f, -9.0799e-02f, 

        -1.2247e-01f, -6.8487e-02f, 9.1290e-01f, 2.0708e-01f, -2.3181e-01f, -2.7004e-01f, -1.2585e+00f, 6.4970e-01f, 

        -4.8921e-01f, 1.6207e-01f, -7.0997e-02f, 1.4221e+00f, -6.5221e-01f, -4.0739e-01f, 5.9012e-01f, -9.8002e-02f, 

        -9.5972e-02f, 5.0366e-02f, -3.6821e-02f, 3.3964e-02f, -2.6348e-02f, 1.3000e-01f, 9.6464e-02f, -7.5199e-02f, 

        6.6303e-02f, -9.0809e-04f, 2.7529e-02f, -6.2622e-02f, 8.5396e-02f, -1.4226e-01f, 3.7553e-02f, 1.0894e-01f, 

        1.3418e-01f, 3.0646e-02f, -2.2599e-01f, -4.0026e-03f, -1.0264e-01f, -4.4265e-02f, 6.3030e-02f, -1.5868e-01f, 

        6.9624e-02f, 4.0740e-02f, 5.3464e-02f, -1.6388e+00f, -1.6121e+00f, -7.1951e-01f, 1.4529e+00f, -4.6905e-02f, 

        -3.0230e-01f, 8.9291e-01f, 1.1564e-01f, -7.2852e-02f, -6.4969e-02f, 2.9593e-02f, 7.4432e-02f, -1.5876e-02f, 

        -1.8152e-02f, -1.5354e-01f, -7.7694e-02f, -5.2931e-02f, -5.3520e-02f, -1.6145e-01f, 2.7507e-02f, -1.0421e-01f, 

        1.2237e-01f, 6.6031e-02f, -8.1052e-02f, 6.2278e-02f, -2.3823e-02f, -1.4094e-01f, 1.1347e-01f, 6.1712e-02f, 

        -1.3981e-02f, -1.0876e-02f, 1.4430e-01f, 3.6049e-02f, 2.1470e-02f, -1.2550e-01f, -1.2663e+00f, 8.0403e-01f, 

        7.8985e-01f, 6.9032e-01f, 7.7958e-01f, 9.9119e-01f, 8.0930e-01f, 3.7599e-02f, 1.1687e-01f, -1.2135e-01f, 

        3.1211e-02f, 1.4481e-02f, 1.9972e-02f, 4.7363e-02f, -9.1623e-02f, 9.9898e-02f, -1.5380e-03f, -2.2632e-01f, 

        -1.6254e-01f, 2.3990e-02f, 1.6530e-01f, 1.2432e-01f, 1.0927e-01f, 4.0035e-02f, -1.7422e-02f, 1.0851e-01f, 

        -5.0825e-02f, 3.7458e-02f, 1.6255e-01f, -2.3706e-02f, -9.7903e-03f, -6.9344e-02f, -6.7938e-02f, 1.3696e-01f, 

        -9.4192e-02f, -1.2671e-01f, -1.7776e-01f, 1.5099e-01f, 1.8828e-01f, -6.6891e-02f, 7.6432e-02f, -1.8261e-02f, 

        3.8846e-02f, -1.5527e-01f, -1.1099e-01f, 2.5219e-01f, -1.5191e-01f, -8.0440e-03f, 1.1705e-01f, -3.4883e-02f, 

        -1.0590e-01f, 1.6698e-03f, -8.9470e-02f, -5.8721e-02f, -2.1376e-01f, -1.6654e-01f, -2.8521e-02f, -1.4245e-01f, 

        1.0119e-01f, 2.5299e-01f, 8.1244e-02f, -1.6042e-01f, -4.8407e-02f, 8.0904e-02f, -1.2495e-01f, -9.4637e-02f, 

        1.5629e-01f, 3.3973e-03f, -9.0387e-03f, 4.8838e-02f, 8.4238e-01f, 4.0597e-02f, -3.4819e+00f, -3.1201e-01f, 

        -2.4516e-01f, 6.4291e-01f, 6.0381e-01f, 2.1017e-01f, -2.3653e-01f, -2.3032e+00f, -1.7288e-01f, 1.2417e+00f, 

        -1.9290e-01f, 1.0035e+00f
    };

    // Layer 1 recurrent weights - 1-to-1 (vector of 38 values)
    float recurrent_weights_layer1[38] = {
        4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 

        4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 

        4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 

        4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 

        4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f, 4.9912e-02f
    };

    // Convert and store feedforward weights
    for (int i = 0; i < 456; i++) {
        float scaled = fc1_weights_vector[i] / scale;
        arm_float_to_q15(&scaled, &weights1[i], 1);
    }

    for (int i = 0; i < 266; i++) {
        float scaled = fc2_weights_vector[i] / scale;
        arm_float_to_q15(&scaled, &weights2[i], 1);
    }

    // Convert recurrent weights (1-to-1)
    for (int i = 0; i < 38; i++) {
        float scaled = recurrent_weights_layer1[i] / scale;
        arm_float_to_q15(&scaled, &recurrent_weights1[i], 1);
    }

}

void SNN_Init(void) {
    const float scale = 60.0f;

    // Layer 1 initialization
    // Uniform parameters for all neurons
    q15_t threshold_1, reset_value_1, decay_factor_1;
    float threshold_f_1 = 1.0000e+00 / scale;
    float reset_value_f_1 = 0.0000e+00 / scale;
    float beta_1 = 9.0000e-01f;

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
    float beta_2 = 5.5000e-01f;

    arm_float_to_q15(&threshold_f_2, &threshold_2, 1);
    arm_float_to_q15(&reset_value_f_2, &reset_value_2, 1);
    arm_float_to_q15(&beta_2, &decay_factor_2, 1);

    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        LIFNeuron_Init(&layer2[i], threshold_2, reset_value_2);
        layer2[i].decay_factor = decay_factor_2;
    }

    // Load weights from NIR
    Load_NIR_Weights();

    arm_fill_q7(0, l1_spikes_prev, NUM_NEURONS_LAYER1);
}

// Test 
void SNN_Debug_Spike_Reset(void) { 

    char buf[64]; 

    uint8_t any_spike = 0; 

  

    // Layer 1 

    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) { 

        if (l1_spikes[i]) { 

            any_spike = 1; 

            snprintf(buf, sizeof(buf), "L1[%d] spiked, V_post=", i); 

            usart1_print(buf); 

            // Convert Q15 membrane back to float for readability 

            float mem_f; 

            arm_q15_to_float(&layer1[i].membrane_potential, &mem_f, 1); 

            mem_f *= 60.0f; // undo Q15 scale 

            print_float("", mem_f); 

        } 

    } 

  

    // Layer 2 

    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) { 

        if (l2_spikes[i]) { 

            any_spike = 1; 

            snprintf(buf, sizeof(buf), "L2[%d] spiked, V_post=", i); 

            usart1_print(buf); 

            float mem_f; 

            arm_q15_to_float(&layer2[i].membrane_potential, &mem_f, 1); 

            mem_f *= 60.0f; 

            print_float("", mem_f); 

        } 

    } 

  

    if (!any_spike) { 

        usart1_print("no spikes this step\r\n"); 

    } 

} 


void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes) {
    // Layer 1 with recurrent connections (fully connected)
    LIFNeuron_Layer_Update_Subtract(layer1, input_spikes, weights1, NUM_INPUTS, NUM_NEURONS_LAYER1, l1_spikes, l1_spikes_prev, recurrent_weights1, 0);

    // Layer 2 (no recurrent, fully connected)
    LIFNeuron_Layer_Update_Subtract_NoRecurrent(layer2, l1_spikes, weights2, NUM_NEURONS_LAYER1, NUM_NEURONS_LAYER2, l2_spikes, 0);
    
    // Store spikes for layer 1 recurrent connections
    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        l1_spikes_prev[i] = l1_spikes[i];
    }

    // Copy output spikes
    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        output_spikes[i] = l2_spikes[i];
    }
}

void SNN_Reset_State(void) {
    // Reset layer 1
    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        layer1[i].membrane_potential = layer1[i].reset_value;
        l1_spikes[i] = 0;
        l1_spikes_prev[i] = 0;
    }

    // Reset layer 2
    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        layer2[i].membrane_potential = layer2[i].reset_value;
        l2_spikes[i] = 0;
    }

}
