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
    }
}


void Load_NIR_Weights(void) {
    const float scale = 60.0f;

    // Layer 1 feedforward weights - fully connected (12x38)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    float fc1_weights_vector[456] = {
        1.0046e+00f, 9.0662e-02f, 1.4108e+00f, -3.7452e+00f, 5.2723e-01f, -4.2064e+00f, -1.8435e-01f, -1.8159e+00f, 

        -2.4500e+00f, -9.9136e-01f, -4.5348e+00f, 1.2742e-02f, 1.0404e+00f, -2.5342e+00f, -9.7087e-01f, -4.2855e-01f, 

        -1.7348e+00f, -3.3917e+00f, -2.1879e+00f, -2.9643e+00f, -2.7972e+00f, -3.4554e+00f, -2.0694e+00f, -8.3099e-01f, 

        -2.4591e+00f, -2.9869e+00f, 4.4645e-01f, -6.0913e-01f, -3.1907e-01f, -2.2960e+00f, -2.8958e+00f, -1.8707e+00f, 

        -3.3669e-01f, -2.5349e+00f, -8.8761e-01f, -1.0603e+00f, -9.5482e-01f, -1.1469e+00f, -2.4440e-01f, 5.7898e-02f, 

        -9.9466e-01f, -4.7203e+00f, 4.1105e-01f, -5.3782e+00f, 5.0526e-01f, -4.1248e+00f, -4.8012e+00f, -1.5613e+00f, 

        -2.0671e-01f, -1.0134e+00f, -6.1137e-01f, 9.7020e-03f, -3.8905e+00f, -2.0252e+00f, -1.7461e+00f, -4.7830e+00f, 

        -1.9872e+00f, 5.5488e-01f, -4.9124e+00f, -5.1493e+00f, -4.2310e+00f, -2.6050e+00f, -5.8082e+00f, -4.7706e+00f, 

        4.5116e-01f, -1.3408e+00f, -1.8632e+00f, -2.6401e+00f, -4.3249e+00f, -1.0586e+00f, -5.6600e-01f, -1.4612e+00f, 

        -2.5043e-02f, -2.8814e+00f, -2.3608e+00f, 4.1171e-02f, -1.1502e-01f, -7.8880e-02f, -8.1086e-01f, -4.5753e+00f, 

        -1.3025e+00f, -5.1206e+00f, -7.0493e-02f, -4.5368e+00f, -4.6780e+00f, -1.6028e+00f, 5.9728e-01f, -6.1524e-01f, 

        7.4005e-02f, 3.0542e-01f, -3.5350e+00f, -2.0281e+00f, -1.9065e+00f, -4.3884e+00f, -4.7501e-01f, -1.5755e+00f, 

        -4.5969e+00f, -5.3408e+00f, -4.2993e+00f, -2.2642e+00f, -5.1364e+00f, -4.5969e+00f, -3.5929e-01f, -1.1828e+00f, 

        -1.4348e+00f, -1.9023e+00f, -4.1883e+00f, -3.5606e-01f, -6.9248e-01f, -1.8999e+00f, 2.6721e-01f, -2.2343e+00f, 

        -2.1063e+00f, -1.3023e+00f, -1.0279e+00f, -3.1248e-01f, -1.9011e+00f, -3.0783e+00f, -2.2362e-01f, -4.3410e+00f, 

        4.5258e-02f, -3.6691e+00f, -4.7486e+00f, -2.0590e+00f, -5.4825e-01f, 1.9838e-01f, 9.7104e-02f, 2.0273e-01f, 

        -2.3646e+00f, -9.7062e-01f, -1.6503e+00f, -4.6849e+00f, -2.3641e+00f, 3.5767e-01f, -4.6596e+00f, -4.8598e+00f, 

        -3.1013e+00f, -1.6925e+00f, -4.8202e+00f, -4.6610e+00f, 5.1134e-01f, -8.9190e-01f, -9.3075e-01f, -2.7122e+00f, 

        -3.7223e+00f, -7.0340e-01f, -4.7789e-01f, -1.4538e+00f, 1.5444e-02f, -2.1419e+00f, -2.1237e+00f, 4.3430e-01f, 

        4.0989e-01f, 1.9131e-01f, 7.7501e-01f, -2.7574e+00f, -1.8147e-01f, -4.9689e+00f, -6.8891e-01f, -3.4273e+00f, 

        -3.7728e+00f, -1.9814e+00f, 1.5189e-02f, -9.1238e-01f, 5.6965e-01f, 4.3037e-01f, -2.2967e+00f, -1.0340e+00f, 

        -1.7687e+00f, -3.9537e+00f, -1.1675e+00f, 2.3470e-01f, -3.9492e+00f, -4.5161e+00f, -3.4417e+00f, -1.8900e+00f, 

        -4.3077e+00f, -5.4773e+00f, -2.4595e-02f, -6.2892e-01f, -9.5976e-01f, -2.7547e+00f, -4.0544e+00f, -1.7569e+00f, 

        -1.0227e+00f, -1.3350e+00f, -1.9587e+00f, -1.9047e+00f, -2.2632e+00f, -4.4247e-01f, 6.8367e-01f, 2.5946e-01f, 

        -2.4574e-02f, -2.8934e+00f, -1.7790e+00f, -4.6505e+00f, 1.1882e-01f, -4.2778e+00f, -4.7213e+00f, -2.1646e+00f, 

        -5.3308e-02f, -9.1042e-01f, 1.5980e-01f, -1.2764e-01f, -2.7782e+00f, -1.3917e+00f, -1.8676e+00f, -3.5922e+00f, 

        -1.1472e+00f, -2.1443e+00f, -4.1255e+00f, -4.9926e+00f, -3.4044e+00f, -1.9964e+00f, -4.6770e+00f, -5.5361e+00f, 

        -4.1063e+00f, -1.1839e+00f, -1.6998e+00f, -2.3826e+00f, -4.0258e+00f, -1.1462e+00f, -1.2583e+00f, -1.3099e+00f, 

        8.5758e-02f, -2.7975e+00f, -2.0350e+00f, -1.2560e+00f, 4.8626e-01f, -5.8989e-01f, -5.6970e-01f, -4.5202e+00f, 

        8.3245e-01f, -4.6157e+00f, 3.0975e-01f, -3.6280e+00f, -4.4285e+00f, -2.1757e+00f, -2.4343e-01f, -1.0026e+00f, 

        -9.9648e-01f, -3.8206e-01f, -4.0836e+00f, -2.0717e+00f, -1.9607e+00f, -4.4141e+00f, -7.4458e-01f, 4.5769e-01f, 

        -4.4326e+00f, -5.1381e+00f, -4.0411e+00f, -3.1936e+00f, -5.4423e+00f, -4.9833e+00f, -8.5785e-02f, -1.7017e+00f, 

        -1.5843e+00f, -1.9387e+00f, -4.2663e+00f, -1.0387e+00f, -9.4388e-01f, -1.7197e+00f, -7.8895e-02f, -2.2433e+00f, 

        -2.8636e+00f, 4.9480e-01f, -1.3573e-01f, 9.3235e-01f, -5.9502e-02f, -3.2288e+00f, -9.7771e-02f, -4.8070e+00f, 

        -4.1527e-01f, -3.1498e+00f, -4.1806e+00f, -2.4465e+00f, -2.3554e+00f, 1.2407e+00f, 3.7626e-01f, 3.2482e-01f, 

        -2.7079e+00f, -6.9714e-01f, -2.3679e+00f, -3.0879e+00f, -2.3232e+00f, -1.9562e-01f, -3.9700e+00f, -3.7140e+00f, 

        -2.9105e+00f, -1.9788e+00f, -4.3460e+00f, -5.1688e+00f, -4.3735e-01f, -5.8139e-01f, -1.1793e+00f, -2.2700e+00f, 

        -4.7909e+00f, -1.6279e+00f, -3.9064e-01f, -9.5807e-01f, 2.7537e-01f, -1.9821e+00f, -2.2437e+00f, 7.9627e-01f, 

        -5.9486e-01f, 1.4104e+00f, 6.3203e-01f, -2.0541e+00f, -5.9718e-01f, -4.7824e+00f, -3.3328e+00f, -1.8776e+00f, 

        -3.0584e+00f, -6.3233e-01f, -2.0498e+00f, 1.0255e-02f, 4.3282e-01f, 1.0373e-01f, -1.4386e+00f, -2.7859e-01f, 

        -1.6160e+00f, -3.0274e+00f, -2.4363e+00f, 7.1807e-01f, -3.6436e+00f, -3.8207e+00f, -1.5327e+00f, -1.0570e+00f, 

        -3.1757e+00f, -4.1412e+00f, 4.3065e-01f, -1.1639e-01f, -5.2398e-01f, -1.7054e+00f, -3.3513e+00f, -1.5964e+00f, 

        -3.7007e-01f, -7.9766e-01f, 2.9048e-02f, -2.7356e+00f, -1.7581e+00f, -3.3928e-01f, -1.0764e-02f, 1.8471e-01f, 

        1.3699e+00f, -3.0871e+00f, -2.4859e+00f, -4.1342e+00f, -1.4001e+00f, -9.5731e-01f, -2.8014e+00f, -7.7167e-01f, 

        -4.1010e-01f, -6.9347e-01f, -1.5746e+00f, 8.9519e-01f, -1.1759e-01f, -6.3268e-02f, -1.5828e+00f, -3.7313e+00f, 

        -2.5800e+00f, -3.7646e+00f, -1.8733e+00f, -4.2726e+00f, -1.2346e+00f, -3.7726e-01f, -3.2589e+00f, -3.9409e+00f, 

        -4.5699e+00f, -1.6064e-01f, 7.2461e-02f, -2.5743e+00f, -2.2459e+00f, -1.6132e+00f, -1.0995e-01f, -1.0028e+00f, 

        3.2811e-02f, -2.1715e+00f, -7.8920e-01f, -1.3063e+00f, -1.5462e-01f, 1.5927e-01f, 2.8874e-01f, -3.5658e+00f, 

        6.3510e-02f, -5.3093e+00f, 7.9692e-01f, -4.6892e+00f, -4.4971e+00f, -1.9611e+00f, 4.8411e-01f, 4.1297e-02f, 

        -3.7482e-01f, 3.9143e-01f, -3.0907e+00f, -2.9726e+00f, -1.9983e+00f, -4.6403e+00f, -3.9776e-01f, -2.4008e-01f, 

        -4.0294e+00f, -4.9216e+00f, -4.7159e+00f, -3.0996e+00f, -5.1893e+00f, -5.1062e+00f, -7.5288e-02f, -1.6866e+00f, 

        -1.7608e+00f, -3.2289e+00f, -4.5549e+00f, -8.3460e-01f, -1.1838e+00f, -1.4378e+00f, -2.9489e-01f, -1.7180e+00f, 

        -2.3750e+00f, -8.8453e-01f, -3.3922e+00f, -2.3001e+00f, -1.1866e+00f, -3.2563e+00f, -4.2821e-02f, -3.5144e+00f, 

        -1.9578e-01f, -1.5451e+00f, -2.1456e+00f, -7.0951e-01f, -4.8257e-01f, 8.2661e-01f, 9.7228e-01f, -3.2280e+00f, 

        -6.6046e-01f, -9.0151e-01f, -1.5731e+00f, -2.7183e+00f, -1.1255e-01f, -2.5051e-01f, -2.9120e+00f, -2.7159e+00f, 

        -1.5266e+00f, -1.2587e+00f, -1.7238e+00f, -2.3790e+00f, -4.6436e-01f, -3.9466e-01f, 2.0421e-01f, -1.0162e+00f, 

        -2.5248e+00f, -8.6266e-01f, 1.2510e-01f, -2.0106e+00f, 6.5742e-01f, -6.9131e-01f, -1.2036e+00f, -1.2367e+00f
    };

    // Layer 2 feedforward weights - fully connected (38x7)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    float fc2_weights_vector[266] = {
        -1.6296e+00f, 9.5331e-01f, 9.2688e-01f, 3.6863e-01f, -4.3394e-02f, -4.9141e-01f, -9.8172e-01f, 4.6668e-03f, 

        -3.9433e+00f, 6.3992e-01f, 6.2265e-01f, 9.9537e-01f, 8.4887e-02f, -3.8811e-01f, 1.4379e+00f, -3.6185e-01f, 

        -4.1244e+00f, -3.3292e-01f, 8.7192e-01f, 9.4533e-01f, -2.9457e-01f, -8.4552e-02f, 3.8775e-02f, -5.9071e-02f, 

        -6.1738e-02f, 7.3167e-03f, -3.2295e-02f, -1.7722e-01f, -5.6390e-02f, -3.3152e+00f, -8.3164e-04f, 9.9590e-01f, 

        -8.3205e-02f, 8.4115e-01f, 9.9119e-01f, 7.1098e-03f, -4.8255e-02f, 1.4157e-01f, -9.0681e-02f, -6.3888e-02f, 

        -1.3506e-01f, 9.1469e-02f, -5.5568e+00f, 1.0141e+00f, 9.9802e-01f, 3.4191e-02f, 7.5398e-01f, 1.1129e+00f, 

        1.0009e+00f, 1.8881e-01f, 8.1563e-03f, -2.8102e-02f, 8.5496e-02f, 7.2660e-02f, -1.6215e-01f, -2.3157e-01f, 

        -9.8483e-02f, -8.0035e-02f, 2.1954e-02f, 3.2610e-02f, 2.7244e-02f, 1.1105e-01f, -1.0441e-01f, 1.9874e-03f, 

        7.5800e-02f, -1.4859e-01f, 4.4637e-02f, 1.9042e-01f, -2.3172e-01f, 2.5267e-01f, -1.1451e+00f, 1.3793e+00f, 

        -6.0574e-01f, -1.7713e-02f, 6.5459e-02f, -5.3144e-01f, 1.1008e+00f, 2.7209e-01f, -3.8748e-01f, 2.0060e+00f, 

        -1.5196e+00f, -1.3877e+00f, -1.2420e+00f, 7.6284e-01f, 1.0022e+00f, -3.4848e+00f, -4.7366e+00f, 2.4783e-02f, 

        9.6033e-02f, 7.7441e-01f, 4.8574e-01f, -4.2820e+00f, -2.3446e-01f, 6.1429e-03f, 9.5886e-01f, 6.7446e-01f, 

        -4.8299e-01f, -6.0208e+00f, 2.2935e-01f, -1.4472e-01f, -3.0262e-01f, -1.2287e-01f, -8.0460e-02f, -3.6876e-01f, 

        4.9988e-03f, 1.8259e-01f, -2.2300e-01f, 6.7278e-04f, 2.6707e-01f, 1.2345e-01f, -1.3711e-01f, 2.5164e-02f, 

        -2.6177e-02f, 3.1674e-02f, 8.7829e-02f, -1.5155e-01f, 1.8337e-01f, -2.8930e-02f, -1.4055e-01f, -5.6023e-02f, 

        -2.2381e-03f, -1.0077e-01f, 8.5053e-02f, -7.5851e-02f, 1.4107e-01f, 1.3365e-02f, 4.8686e-02f, 6.6409e-01f, 

        -5.4005e-01f, 1.1036e-01f, -6.3127e-01f, 3.9589e-01f, -2.4892e-01f, -8.7493e-01f, 1.9905e-01f, -1.2528e-01f, 

        -7.0354e-02f, -2.7895e+00f, 1.6133e-01f, 1.8194e-01f, -7.2052e-02f, 9.0824e-02f, -1.8647e-01f, 4.1835e-02f, 

        1.2952e-02f, 2.5675e-02f, -5.2968e-03f, 7.5962e-02f, 1.4096e-01f, 4.8182e-02f, 1.8245e-04f, -1.3149e-01f, 

        1.5397e-01f, -6.4297e-02f, 4.1747e-02f, -1.9162e-01f, -1.4636e-01f, 1.4118e-01f, 2.8431e-02f, -2.1849e-01f, 

        -1.3338e-02f, -3.5597e-02f, -7.3820e-02f, -1.2710e-01f, -1.7232e-01f, 1.2718e-01f, -8.4613e-02f, 4.8016e-02f, 

        -1.4447e-02f, -1.4090e-01f, 1.9070e-03f, -8.4085e-02f, -1.9935e-01f, 5.9475e-02f, 1.1599e-02f, 2.4871e-02f, 

        -1.2833e-01f, 1.2899e-01f, 7.1964e-02f, 4.4205e-02f, -1.1100e-01f, -1.6086e-01f, -3.1492e+00f, 9.1268e-02f, 

        7.5377e-02f, 1.9566e-01f, 6.9900e-01f, -1.5572e+00f, 1.0004e+00f, 2.8546e-01f, -6.1783e-01f, -4.7477e-02f, 

        2.5237e-01f, 1.7833e-01f, 1.1397e-01f, 1.3909e-01f, 2.9199e-01f, -2.0964e-01f, -4.1391e-01f, -3.1087e-01f, 

        -1.2147e-01f, 1.8982e-01f, 3.0108e-01f, 9.9694e-02f, 2.0184e-01f, -1.5035e-01f, 3.7201e-02f, -1.3241e-01f, 

        -1.7944e-02f, 9.0751e-02f, 3.5417e-02f, -1.1416e-01f, -1.6877e-01f, -6.2902e-02f, 1.2204e-01f, -9.5485e-02f, 

        -1.0009e-01f, -8.4558e-01f, 5.5718e-01f, -7.8523e-03f, -1.4873e-01f, 3.3618e-01f, 2.5467e-01f, 1.6304e-01f, 

        2.4708e-01f, -3.6874e-01f, -7.1503e-02f, -2.2332e-01f, 2.8728e-01f, 2.4400e-01f, 5.6599e-01f, -3.8422e-01f, 

        -7.6442e-02f, 9.9530e-02f, 1.5025e-01f, 1.5776e-02f, -2.6576e-01f, -1.1823e-01f, 6.2723e-01f, 1.0008e+00f, 

        2.5089e-01f, 8.8378e-01f, 9.3648e-01f, 1.1017e+00f, 9.9998e-01f, -3.1743e-02f, -1.4911e-01f, 2.0988e-02f, 

        -3.1663e-02f, -1.3290e-01f, 2.2052e-01f, -8.7238e-02f, -2.0367e-01f, 4.9983e-02f, -1.2446e-01f, -1.8975e-01f, 

        -4.5847e-03f, -1.7398e-01f, -1.0978e-01f, -9.9827e-01f, -2.4436e-01f, -4.2127e-01f, 6.6722e-01f, -1.4748e+00f, 

        -1.3915e+00f, 8.1109e-01f
    };

    // Layer 1 recurrent weights - 1-to-1 (vector of 38 values)
    float recurrent_weights_layer1[38] = {
        9.5756e-01f, 8.7386e-01f, 8.8974e-01f, 1.9810e-01f, 9.6957e-01f, 5.8252e-02f, -7.2494e-01f, 2.1446e-01f, 

        2.5448e-01f, -2.8296e-01f, 9.9248e-01f, 1.3120e+00f, -2.9939e-01f, -2.6953e-01f, 2.6889e-01f, -5.1123e-02f, 

        1.1420e-01f, 4.2669e-02f, 1.6686e-01f, 2.4500e+00f, -1.0608e-01f, -3.3854e-02f, -1.2917e-01f, 1.2048e-01f, 

        -2.8869e-01f, -1.3308e-01f, 9.2589e-01f, -2.0193e-01f, 1.6705e-01f, -3.3076e-01f, 2.8720e-01f, -1.2302e-01f, 

        4.1303e-01f, -1.8404e-02f, 5.6042e-01f, 1.2703e-01f, -6.9307e-03f, 1.4362e+00f
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
