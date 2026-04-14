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
// Layer 0: 12 -> 42 (fully connected, no recurrent, uniform params)
// Layer 1: 42 -> 7 (fully connected, no recurrent, uniform params)

// Global variables for the SNN
#define NUM_INPUTS 12
#define NUM_NEURONS_LAYER1 42
#define NUM_NEURONS_LAYER2 7

static __attribute__((aligned(32))) LIFNeuron layer1[NUM_NEURONS_LAYER1], layer2[NUM_NEURONS_LAYER2];
static __attribute__((aligned(32))) q7_t l1_spikes[NUM_NEURONS_LAYER1];
static __attribute__((aligned(32))) q7_t l2_spikes[NUM_NEURONS_LAYER2];

static __attribute__((aligned(32))) q15_t weights1[NUM_INPUTS*NUM_NEURONS_LAYER1]; // Fully connected
static __attribute__((aligned(32))) q15_t weights2[NUM_NEURONS_LAYER1*NUM_NEURONS_LAYER2]; // Fully connected

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

    // Layer 1 feedforward weights - fully connected (12x42)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    static const float fc1_weights_vector[504] = {
        3.9857e-01f, -8.4425e-02f, -2.0660e-01f, 6.2159e-01f, -9.3243e-01f, -8.3697e-01f, -2.9711e-01f, 8.2442e-02f, 

        -6.8495e-01f, 6.0096e-01f, 7.2050e-03f, -5.6278e-03f, -9.4986e-01f, -1.0361e-01f, -4.6997e-01f, 9.6391e-02f, 

        -2.8693e-01f, -5.4542e-03f, -1.1504e+00f, -1.3793e-01f, -5.7794e-01f, -1.8319e+00f, 1.7275e-01f, 3.5667e-01f, 

        -4.4553e-02f, 2.2526e-01f, 4.4012e-01f, -3.3525e-01f, 4.4207e-01f, 3.7750e-02f, -1.4974e-01f, 4.7050e-01f, 

        -2.1071e-01f, -1.5344e-01f, -7.0040e-01f, 1.3756e-01f, 2.6320e-01f, -5.0193e-01f, -6.1505e-01f, -5.9089e-01f, 

        1.7665e-01f, -2.3865e-01f, -9.7903e-02f, 1.4833e-01f, 3.6497e-01f, 1.8636e-01f, 1.2130e-01f, -3.4244e-01f, 

        -5.5208e-02f, -1.0666e-01f, 3.0219e-01f, -2.3766e-01f, 1.5763e-01f, 1.5485e-01f, -4.1660e-01f, -2.8585e-01f, 

        -2.2136e-01f, -8.4098e-02f, -1.5894e-01f, -8.3504e-02f, 1.7133e-02f, -3.1186e-01f, -1.9425e-01f, 1.4962e-01f, 

        -2.8853e-01f, 2.4838e-01f, -4.9967e-01f, -1.4940e-01f, 4.8280e-02f, 4.7531e-01f, -9.7246e-02f, -1.0399e-01f, 

        -3.9485e-01f, 1.1315e-01f, 1.1952e-01f, -1.4766e-01f, -2.7381e-01f, 1.5422e-01f, -1.9359e-02f, 4.9110e-02f, 

        -7.0826e-02f, -2.5228e-01f, 2.6977e-01f, 2.1202e-01f, -3.1227e-01f, -1.5447e-01f, -2.1251e-01f, 3.6100e-02f, 

        -2.8000e-02f, 1.9007e-01f, -2.2366e-02f, -4.1416e-01f, 3.4460e-01f, -5.3589e-02f, -1.7267e-01f, 2.5951e-01f, 

        1.2498e-01f, 2.2340e-03f, 2.0843e-01f, 2.1249e-01f, 2.0499e-01f, -2.2318e-01f, -7.3335e-02f, 7.3512e-02f, 

        1.7850e-01f, 3.9043e-01f, -3.1277e-01f, -7.9422e-01f, -3.0518e-01f, -4.2485e-01f, 2.7652e-01f, 2.5652e-01f, 

        -1.3576e-01f, -5.6134e-02f, -1.6549e-02f, 2.0876e-01f, -4.0781e-02f, 1.9569e-01f, 8.6384e-02f, -1.1079e-01f, 

        4.0911e-02f, -1.0393e-01f, 4.1988e-01f, 4.3183e-01f, -3.4716e-01f, 2.9111e-01f, -2.8632e-03f, -1.6788e-01f, 

        4.6380e-02f, 1.9063e-01f, -1.9465e-01f, 5.0503e-02f, -1.5635e-01f, -9.3840e-01f, 1.0290e-02f, 6.2898e-02f, 

        -2.2024e-02f, -3.4019e-02f, 1.1604e-01f, -1.2829e-01f, -7.2144e-02f, 2.3149e-01f, 1.0616e-01f, -2.8644e-01f, 

        -9.4671e-02f, 5.5325e-03f, -1.7777e-01f, 5.8632e-02f, -9.8859e-02f, 1.9290e-02f, -2.7019e-01f, 2.0987e-01f, 

        -7.8027e-01f, -3.6565e-01f, -1.4319e-02f, -3.7434e-02f, -1.2815e-01f, 1.6562e-01f, -2.6707e-01f, 1.6589e-01f, 

        -2.1067e-01f, 2.8228e-01f, 3.5587e-01f, -1.8259e-01f, 2.5419e-01f, 2.4765e-01f, -1.0561e-01f, 2.9590e-02f, 

        -1.4381e-01f, -2.0387e-01f, -3.3770e-01f, -3.1183e-01f, 7.3597e-02f, 4.6098e-01f, 3.0521e-01f, 1.2532e-01f, 

        -4.8532e-02f, -4.3272e-01f, -5.0282e-01f, 4.6550e-01f, 2.3659e-01f, 2.1841e-03f, 3.4544e-01f, 3.1712e-01f, 

        -1.9574e-01f, -8.8092e-02f, 2.0428e-01f, 2.9622e-01f, -7.7924e-01f, -4.8903e-01f, 6.2092e-02f, 4.0343e-03f, 

        -3.4979e-01f, -1.4975e-01f, -2.6132e-01f, -4.1091e-02f, -9.7641e-02f, 2.1458e-01f, -2.3852e-01f, 1.4672e-01f, 

        -2.5266e-01f, -1.1721e+00f, 4.6130e-01f, -2.8815e-01f, 4.4603e-01f, -4.3146e-02f, -2.7149e-02f, 1.0393e-01f, 

        -1.5351e-01f, 3.4121e-02f, 3.0520e-01f, -2.0260e-01f, -3.1530e-01f, 1.0596e-01f, -6.0691e-01f, 2.3542e-02f, 

        8.8988e-02f, -1.0713e-01f, 6.4104e-02f, 1.3685e-01f, -1.0686e-01f, 2.5935e-01f, -2.6301e-01f, 2.6901e-01f, 

        4.8287e-02f, 3.8870e-01f, 4.2251e-01f, 1.2023e-01f, -3.6874e-01f, 1.6966e-01f, 1.9988e-01f, -4.6933e-03f, 

        1.0158e-01f, -1.7432e+00f, 1.2029e-01f, -4.1813e-01f, -1.0056e-01f, -3.5145e-01f, -1.5018e-01f, 2.3285e-01f, 

        -2.6990e-01f, -2.4859e-01f, 1.7199e-01f, 6.1006e-02f, 2.1831e-01f, -3.4519e-03f, -1.6889e-01f, -6.6260e-03f, 

        1.1588e-01f, 8.2391e-02f, -4.2870e-02f, 5.0451e-01f, -1.6647e-03f, -2.3368e-01f, 7.5896e-02f, -2.0770e-02f, 

        -3.0142e-01f, -3.5623e-01f, 1.0662e-01f, 1.6707e-01f, 1.3099e-02f, -2.7901e-01f, -2.2771e-01f, 1.1028e-01f, 

        -1.0731e-01f, -2.0586e-01f, -3.9339e-02f, -6.6601e-01f, 5.6130e-02f, 7.1562e-02f, -3.5019e-02f, 2.6333e-01f, 

        -7.3111e-02f, 3.2238e-01f, -3.0151e-01f, 3.9730e-01f, 6.7742e-02f, 3.5930e-01f, 3.7369e-01f, -4.5767e-02f, 

        1.0307e-01f, -5.4103e-01f, -3.0279e-01f, 3.6159e-01f, 1.6282e-01f, -8.0392e-02f, -8.9877e-02f, 3.7985e-01f, 

        -3.2838e-01f, -2.7560e-01f, 1.1812e-01f, 2.9764e-01f, 3.1806e-01f, -4.2314e-02f, 3.6377e-01f, 3.6377e-01f, 

        -1.2126e-01f, -4.7999e-02f, -7.6587e-01f, -7.2649e-02f, -3.7415e-02f, -2.8047e-01f, 2.5619e-01f, -2.7286e-01f, 

        1.3655e-01f, -1.7694e+00f, -1.6331e-01f, -1.0066e-01f, -1.2218e-01f, 1.8373e-01f, 6.1180e-02f, -9.1478e-01f, 

        2.9880e-01f, -2.1819e-01f, -1.2845e+00f, -3.4232e-01f, -3.6336e-01f, 1.3105e-01f, 7.1945e-01f, -1.4551e+00f, 

        2.9800e-01f, -5.9339e-02f, 1.2172e-01f, -3.3214e-01f, 1.5445e-02f, -3.7396e-01f, -1.7168e-01f, -1.0749e+00f, 

        1.2959e-01f, -4.1151e-01f, -3.0858e-01f, 1.5319e-01f, 1.0932e-01f, 1.1434e-01f, 2.3078e-01f, 2.0937e-02f, 

        1.5946e-01f, 1.0197e-01f, -5.9815e-01f, 1.4886e-01f, -1.8140e-01f, 3.8519e-01f, 2.4793e-01f, -1.0231e+00f, 

        4.6446e-01f, -1.0712e+00f, -1.1916e-01f, -1.8710e+00f, -9.9355e-03f, -1.8982e+00f, -2.3695e-01f, 3.4870e-01f, 

        -1.3216e-01f, -7.8291e-01f, 2.0800e-02f, -2.8590e-01f, -2.4696e-01f, -4.1165e-01f, 1.3609e-01f, 2.4767e-01f, 

        -4.8061e-02f, -8.9278e-01f, -1.0328e+00f, -8.9826e-02f, -4.5266e-01f, -3.1247e-02f, -2.9600e-01f, -5.7694e-01f, 

        -1.6660e+00f, 1.0242e-01f, -1.4263e+00f, 2.1991e-01f, 3.8147e-01f, -2.7012e-01f, -7.4719e-02f, 7.8719e-02f, 

        2.2469e-01f, 1.6280e-01f, 1.4837e-01f, 2.5272e-01f, -5.8941e-01f, -3.2461e-01f, 3.4370e-01f, 5.3715e-01f, 

        -1.9912e-01f, -1.8889e-01f, 5.8215e-01f, -7.6961e-01f, -3.7256e-01f, -1.0483e+00f, -7.2057e-01f, -8.4313e-01f, 

        -5.2176e-01f, -6.9279e-01f, -3.1760e-01f, -4.2694e-01f, -2.6931e-01f, -3.1778e-01f, -3.5363e-01f, -3.8373e-02f, 

        1.3010e-01f, -7.5491e-01f, -8.4494e-01f, -1.0936e+00f, -3.4917e-01f, 3.3614e-01f, 8.2943e-02f, 3.8829e-01f, 

        -3.2908e-01f, 4.2858e-02f, -7.2487e-01f, 1.3204e-01f, -5.9253e-01f, 2.3964e-01f, -6.9993e-01f, -5.1689e-01f, 

        -7.1411e-01f, 2.0153e-01f, -3.3414e-02f, 3.4881e-01f, -5.9367e-02f, -1.6146e-01f, 4.2748e-02f, -6.7788e-04f, 

        2.4523e-01f, 1.6815e-01f, 1.7675e-01f, -3.2574e-02f, -1.5656e-01f, 6.2672e-03f, 4.8451e-02f, -5.0478e-02f, 

        -4.0683e-02f, -1.8018e-01f, 2.4007e-01f, 7.5845e-02f, 2.1717e-01f, 1.8554e-01f, 7.4937e-02f, -2.0879e-01f, 

        4.3363e-01f, -2.3645e-02f, 1.8040e-01f, -2.8818e-02f, -4.7911e-01f, -3.3543e-01f, -1.7310e-01f, -3.6800e-01f, 

        -5.0726e-01f, 3.3891e-01f, 7.7511e-02f, -6.6478e-02f, 2.4510e-01f, 4.1633e-01f, 1.1944e-01f, -2.5208e-01f, 

        -2.2665e-01f, -1.1296e-01f, -5.1654e-02f, -1.3540e-01f, -4.5950e-02f, -1.7043e-02f, 1.3240e-01f, 7.0945e-01f, 

        -3.6807e-01f, 4.7676e-01f, -8.3395e-01f, 2.8856e-01f, -1.9019e-01f, 2.9436e-01f, -3.5196e-01f, 4.5212e-01f, 

        2.3472e-01f, -1.6456e-01f, -2.4291e-01f, 3.3629e-01f, -6.9364e-01f, -6.4806e-01f, -3.6462e-01f, -2.3768e-02f, 

        -3.6402e-01f, -2.2125e-01f, 2.4962e-01f, -2.0984e-01f, 1.3512e-01f, -1.1577e-01f, -4.0368e-01f, -9.4624e-01f, 

        3.3744e-01f, -5.5834e-01f, 6.8197e-01f, -9.2516e-03f, -2.8484e-01f, 1.0193e-02f, -1.2039e-01f, 1.3058e-01f, 

        -9.3170e-01f, -5.8332e-01f, 7.1385e-01f, 5.9473e-03f, 1.0169e+00f, -3.2180e-01f, -6.5392e-01f, -6.1164e-01f
    };

    // Layer 2 feedforward weights - fully connected (42x7)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    static const float fc2_weights_vector[294] = {
        2.1344e-01f, -1.6052e-01f, 1.4844e-01f, 5.6743e-02f, 1.0379e-01f, 1.7945e-01f, 9.5382e-02f, 1.1238e-01f, 

        -3.5936e-01f, -9.8929e-01f, 6.7433e-01f, 8.0082e-02f, 1.5412e-01f, 1.1299e-01f, -2.2717e-01f, 1.5914e-01f, 

        1.5655e-01f, 1.4934e-01f, 1.3045e-01f, 1.6508e-01f, 2.6855e-01f, 2.1938e-01f, -1.3980e-01f, 7.4052e-02f, 

        -5.5474e-02f, -1.2729e-02f, 2.3790e-01f, 1.7150e-01f, -4.9132e-01f, 5.8062e-01f, -2.0794e-01f, 3.0760e-01f, 

        -1.1262e-03f, -2.3519e-01f, 3.5280e-01f, -3.1690e-02f, -1.1893e+00f, -1.6948e+00f, 2.2249e-01f, 9.4537e-01f, 

        1.8787e-02f, 2.1622e-01f, 9.3776e-02f, 8.3657e-02f, 1.1210e-01f, 1.1176e-01f, 6.7976e-02f, -9.1792e-02f, 

        -3.0132e-02f, -3.3982e-02f, 2.5770e-01f, 2.5095e-01f, 5.0751e-01f, -8.2892e-01f, 4.2953e-01f, -1.9020e-01f, 

        -6.0609e-01f, -2.6426e-01f, 1.0263e-01f, 2.2052e-01f, 1.2544e-01f, -1.1975e-01f, 5.2885e-03f, 1.6906e-01f, 

        3.1161e-01f, 5.0305e-02f, -2.9688e-01f, -6.9557e-01f, 1.6440e-01f, 3.8484e-02f, -6.6591e-02f, -7.3358e-02f, 

        -4.2012e-02f, -1.6392e-02f, -7.9210e-02f, 1.3675e-01f, 1.9109e-01f, -6.5562e-01f, 1.6830e+00f, 1.4871e-02f, 

        -8.6958e-01f, -1.5048e-01f, -9.3807e-01f, 2.1247e-01f, -6.6987e-02f, 1.1982e-01f, 8.5313e-02f, 2.1297e-01f, 

        -9.1173e-02f, 9.6579e-02f, -4.0090e-02f, 2.3783e-01f, 4.5699e-01f, 3.8860e-01f, -4.5419e-01f, -7.1844e-01f, 

        -8.4953e-03f, -3.7249e-01f, -2.7607e-01f, 2.2659e-01f, 9.1246e-02f, 2.1176e-01f, 1.2305e-01f, 8.0945e-02f, 

        -3.6616e-01f, 4.0670e-01f, -1.2424e+00f, 2.6098e-02f, 1.9387e-01f, 3.4169e-01f, 2.0453e-01f, -1.0653e+00f, 

        2.9759e-02f, 2.1756e-01f, 1.5617e-01f, 1.7234e-01f, 1.5930e-01f, 9.8758e-02f, 2.0506e-02f, -1.3489e-03f, 

        5.7242e-01f, 2.0455e-01f, 6.4981e-02f, -9.4554e-02f, -2.6336e-02f, 1.6013e-01f, -1.8253e-01f, 1.0205e-02f, 

        7.9344e-02f, 2.3983e-01f, 1.5801e-01f, 9.7274e-02f, 1.4169e-01f, 1.3408e-01f, 2.0879e-01f, 1.5091e-01f, 

        1.1333e-01f, 9.2914e-02f, 1.0525e-01f, 2.6920e-02f, 1.2457e-01f, 8.3555e-01f, -1.3298e-01f, -3.3134e-01f, 

        1.6436e-01f, 1.2305e-01f, -1.5171e-01f, -1.1503e+00f, 2.7444e-01f, 7.3065e-02f, 1.8711e-01f, 1.6432e-01f, 

        1.5597e-01f, 2.1420e-01f, -3.0437e-02f, -5.6422e-02f, -2.6618e-02f, -2.0075e-01f, 6.2470e-02f, 1.8271e-01f, 

        -8.9441e-02f, -9.0923e-01f, -3.4075e+00f, -1.0638e+00f, 1.7872e+00f, -1.3178e-01f, -9.2528e-01f, 1.8773e+00f, 

        4.0419e-01f, -4.2858e-01f, -1.0840e+00f, 1.2838e+00f, -1.1854e+00f, -8.4320e-01f, 5.9287e-01f, -7.6733e-01f, 

        2.2147e-01f, 2.6164e-02f, 8.1901e-01f, -3.0842e-01f, -9.6242e-01f, 1.0838e+00f, 2.1459e-01f, 1.1169e+00f, 

        2.9155e-01f, -1.0402e+00f, -9.1792e-02f, 1.8412e-01f, 1.3401e-01f, -2.2060e-01f, 9.9240e-02f, 8.2363e-02f, 

        1.7100e-01f, 2.3544e-01f, 2.3147e-02f, 1.1669e-01f, 2.4529e-01f, -1.0974e-01f, -3.4485e-01f, 4.9894e-02f, 

        -4.2663e-01f, 1.2423e-01f, 1.5101e-01f, 4.7723e-01f, -8.2004e-02f, -9.6935e-01f, -1.0233e+00f, 1.2187e+00f, 

        -1.5201e-01f, -4.1137e-02f, -5.4567e-02f, 6.4238e-02f, -1.3859e-01f, -3.1121e-02f, -9.2563e-02f, 4.7914e-02f, 

        -1.4115e-01f, 1.9265e-01f, 2.2395e-01f, 1.4448e-01f, 1.1384e-01f, 8.7590e-02f, 1.7288e-01f, 1.9892e-01f, 

        -3.9705e-01f, 3.5912e-01f, 2.3937e-01f, 7.1418e-02f, -5.5651e-02f, 2.2413e-01f, 1.2365e-01f, -3.9292e-02f, 

        7.6343e-01f, -1.2859e-01f, -5.9722e-01f, 5.8794e-02f, -7.2825e-02f, 2.9162e-01f, -8.0038e-01f, 2.1090e-01f, 

        3.4476e-01f, 2.1626e-01f, -7.0408e-02f, -4.1065e-02f, -4.8348e-01f, -2.5323e-02f, 1.3701e-01f, 2.8164e-01f, 

        1.4827e-01f, 2.1917e-01f, 1.1227e-01f, 2.3996e-01f, 1.7814e-01f, -5.2162e-01f, -9.9590e-01f, -2.7888e-01f, 

        2.8057e-01f, 1.2051e-01f, 4.5180e-02f, -2.0337e-01f, -6.5643e-02f, -1.1087e-01f, -2.4176e-02f, 1.2805e-02f, 

        1.6598e-01f, -1.6937e-01f, 1.8754e-01f, -1.8498e-01f, -2.9654e-01f, 3.3071e-01f, -1.6684e-01f, 2.3817e-01f, 

        2.9617e-01f, -6.1961e-01f, 1.0656e-01f, 5.3153e-02f, 2.1523e-01f, 1.3650e-03f, -1.3882e-03f, 7.8970e-02f, 

        1.6425e-01f, -3.0277e-02f, 3.1121e-01f, 2.0815e-01f, 2.5739e-01f, 1.5402e-01f, 2.1713e-01f, -3.6396e-01f, 

        -6.4415e-02f, 2.2769e-01f, 1.9969e-01f, 1.3641e-01f, 3.5972e-02f, -2.6038e-01f
    };

    // Convert and store feedforward weights
    for (int i = 0; i < 504; i++) {
        float scaled = fc1_weights_vector[i] / scale;
        arm_float_to_q15(&scaled, &weights1[i], 1);
    }

    for (int i = 0; i < 294; i++) {
        float scaled = fc2_weights_vector[i] / scale;
        arm_float_to_q15(&scaled, &weights2[i], 1);
    }

}

void SNN_Init(void) {
    const float scale = 60.0f;

    // Layer 1 initialization
    // Uniform parameters for all neurons
    q15_t threshold_1, reset_value_1, decay_factor_1;
    float threshold_f_1 = 1.0000e+00 / scale;
    float reset_value_f_1 = 0.0000e+00 / scale;
    float beta_1 = 9.9999e-01f;

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
    float beta_2 = 9.9994e-01f;

    arm_float_to_q15(&threshold_f_2, &threshold_2, 1);
    arm_float_to_q15(&reset_value_f_2, &reset_value_2, 1);
    arm_float_to_q15(&beta_2, &decay_factor_2, 1);

    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        LIFNeuron_Init(&layer2[i], threshold_2, reset_value_2);
        layer2[i].decay_factor = decay_factor_2;
    }

    // Load weights from NIR
    Load_NIR_Weights();

}

void SNN_Run_Timestep(const q7_t* input_spikes, q7_t* output_spikes) {
    // Layer 1 (no recurrent, fully connected)
    LIFNeuron_Layer_Update_Subtract_NoRecurrent(layer1, input_spikes, weights1, NUM_INPUTS, NUM_NEURONS_LAYER1, l1_spikes, 0);

    // Layer 2 (no recurrent, fully connected)
    LIFNeuron_Layer_Update_Subtract_NoRecurrent(layer2, l1_spikes, weights2, NUM_NEURONS_LAYER1, NUM_NEURONS_LAYER2, l2_spikes, 0);


    SNN_Debug_Spike_Reset();
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
    }

    // Reset layer 2
    for (int i = 0; i < NUM_NEURONS_LAYER2; i++) {
        layer2[i].membrane_potential = layer2[i].reset_value;
        l2_spikes[i] = 0;
    }

}
