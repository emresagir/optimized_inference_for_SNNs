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
        -6.2166e-01f, -2.4635e-01f, -6.1061e+00f, -9.4104e-03f, -1.1667e-01f, -5.0454e-01f, -3.8929e-01f, -1.5444e-01f, 

        -8.4933e-01f, -1.5604e+00f, -2.7023e-01f, 5.8842e-01f, -1.9726e+00f, -3.4055e-01f, -8.6767e-01f, -2.7618e-01f, 

        -1.7665e-01f, -5.1809e-01f, -4.0022e-01f, -1.2932e+00f, -1.3175e+00f, -2.9601e-01f, -4.1553e-01f, -1.5392e+00f, 

        -7.4336e-01f, -5.2593e-01f, -8.5978e-01f, -1.9258e+00f, -9.8842e-01f, -4.8905e-01f, -2.4243e-01f, -9.1558e-02f, 

        4.5015e-01f, -1.7726e+00f, 1.1789e-01f, -5.4481e-01f, 1.1422e+00f, 8.7325e-01f, -7.5824e-01f, -5.7531e-01f, 

        -1.1441e-02f, -6.8115e-01f, -8.4831e-02f, -1.1175e+00f, -1.8397e+00f, -8.4134e-01f, -2.1955e+00f, -1.8268e+00f, 

        -1.7773e+00f, 4.1128e-01f, -1.9349e-01f, -6.0276e-01f, -1.8369e+00f, -5.1063e-01f, -4.4936e-01f, -1.5670e+00f, 

        -8.8836e-01f, -1.5987e+00f, -2.5514e+00f, 3.0977e-01f, -5.9471e-01f, -1.4995e+00f, -1.6787e+00f, -1.1161e+00f, 

        -7.0238e-01f, -1.7803e+00f, -7.0338e-01f, -5.6052e-01f, -1.4764e+00f, -1.4224e-01f, 5.5867e-01f, -9.0169e-01f, 

        -1.8380e-01f, -1.2561e+00f, -1.6085e-01f, -3.1736e-01f, -9.2098e-01f, -4.2357e-01f, -9.0203e-01f, -8.1971e-01f, 

        -5.7191e-03f, -7.5312e-01f, -1.9806e+00f, -5.0153e-01f, -2.1309e+00f, -1.1342e+00f, -1.8613e+00f, -4.9939e-01f, 

        5.0426e-01f, -6.1668e-01f, -1.6815e+00f, -5.7121e-01f, -6.8528e-01f, -1.5174e+00f, -6.6666e-01f, -1.3690e+00f, 

        -2.4578e+00f, 6.2807e-02f, -8.2708e-01f, -1.2819e+00f, -1.7911e+00f, -1.1488e+00f, -6.7078e-01f, -1.8729e+00f, 

        -1.0293e+00f, -5.0197e-01f, -1.6943e+00f, -3.4565e-01f, -4.5150e-01f, -7.6537e-01f, -2.0347e-01f, -1.3253e+00f, 

        -2.7752e-01f, -5.6704e-01f, -8.4218e-01f, -5.4368e-01f, 3.1751e-01f, -7.7953e-01f, -1.5131e-01f, -8.3131e-01f, 

        -1.7989e+00f, -8.8976e-01f, -2.3104e+00f, -1.6395e+00f, -2.0083e+00f, 2.6872e-01f, -6.3411e-01f, -7.8712e-01f, 

        -1.3342e+00f, -6.1508e-01f, -5.4346e-01f, -1.4593e+00f, -8.7674e-01f, -5.8430e-01f, -2.3379e+00f, 1.6900e-01f, 

        -8.1653e-01f, -2.1506e+00f, -1.6905e+00f, -1.0979e+00f, -6.8653e-01f, -2.0464e+00f, -9.9934e-01f, -4.2001e-01f, 

        -1.0915e+00f, -1.1746e+00f, -5.8400e-02f, -1.0810e+00f, 1.9342e-01f, -6.1528e-01f, 3.1353e-01f, 2.7582e-02f, 

        -8.4162e-01f, -3.1841e-01f, 2.7863e-01f, -5.5925e-01f, 2.5830e-01f, -5.5813e-01f, -1.5651e+00f, -6.9255e-01f, 

        -1.9647e+00f, -1.4512e+00f, -1.8628e+00f, -3.4190e-01f, 1.2752e-01f, -4.3115e-01f, -1.0670e+00f, -4.0462e-01f, 

        -5.4465e-01f, -1.0916e+00f, -8.5813e-01f, -5.1512e-01f, -2.1381e+00f, -1.2671e+00f, -7.8023e-01f, -2.0498e+00f, 

        -2.0973e+00f, -1.1078e+00f, -5.5616e-01f, -1.8704e+00f, -1.1397e+00f, -3.3935e-01f, -8.5311e-01f, 7.5424e-01f, 

        -5.5999e-01f, -9.6435e-01f, -1.3924e-01f, -3.6487e-01f, -2.2916e-02f, -3.8861e-01f, -8.0471e-01f, -4.9690e-01f, 

        -2.4046e+00f, -8.2255e-01f, 2.3182e-01f, -8.6613e-01f, -1.1948e+00f, -4.7570e-01f, -1.7869e+00f, -2.0755e+00f, 

        -2.0342e+00f, -2.7229e+00f, -2.2125e-01f, -5.6716e-01f, -1.4681e+00f, -4.5036e-01f, -5.5937e-01f, -8.8311e-01f, 

        -4.6319e-01f, -5.5407e-01f, -2.3678e+00f, 1.3302e-01f, -8.7810e-01f, -2.2640e+00f, -2.2873e+00f, -6.1017e-01f, 

        -4.2798e-01f, -1.7543e+00f, -1.0778e+00f, -4.2025e-01f, -9.6611e-01f, 2.0193e-01f, -1.2732e+00f, -5.5776e-01f, 

        3.2779e-01f, -8.5004e-01f, -1.6596e-01f, -1.1118e+00f, -1.0449e+00f, -7.4365e-01f, 6.2110e-01f, -7.5556e-01f, 

        -5.3585e-01f, -6.1684e-01f, -1.3159e+00f, -6.7801e-01f, -2.1799e+00f, -1.9235e+00f, -1.8261e+00f, -6.5912e-02f, 

        -1.8791e-01f, -6.9286e-01f, -1.8869e+00f, -5.9599e-01f, -6.4611e-01f, -9.7099e-01f, -9.9966e-01f, -9.5119e-01f, 

        -2.6473e+00f, 1.3727e-01f, -7.4387e-01f, -1.5684e+00f, -1.5528e+00f, -1.2168e+00f, -8.7479e-01f, -9.5492e-01f, 

        -1.0385e+00f, -8.9791e-01f, -1.1069e+00f, 5.6229e-01f, 1.8148e-01f, -8.9862e-01f, 3.2687e-01f, -1.2796e+00f, 

        -7.1370e-01f, -6.5775e-01f, -7.9244e-01f, -4.0277e-01f, 1.7068e-01f, -4.1444e-01f, 5.2451e-01f, -7.4821e-01f, 

        -5.5231e-01f, -5.7902e-01f, -2.0353e+00f, -2.1773e+00f, -1.8092e+00f, -4.3634e-02f, -2.2506e+00f, -8.1624e-01f, 

        -1.4514e+00f, -5.4134e-01f, -3.1358e-01f, -6.7979e-01f, -7.3660e-01f, -5.4541e-01f, -2.0047e+00f, 1.5256e-02f, 

        -8.8113e-01f, -1.6101e+00f, -1.8385e+00f, -6.0134e-01f, -5.7937e-01f, -1.5179e+00f, -1.4883e+00f, -5.1141e-01f, 

        -1.2968e+00f, -3.8025e-01f, -1.2477e+00f, -8.2453e-01f, -1.8150e-02f, -1.0905e+00f, 2.2643e-01f, 1.0313e-01f, 

        -8.0137e-01f, -5.2449e-01f, -1.6831e+00f, -3.2740e-01f, 9.5370e-01f, -9.5011e-01f, -1.1809e+00f, -1.7214e-01f, 

        -1.5071e+00f, -1.5884e+00f, -6.9723e-01f, 4.9711e-01f, -1.1473e+00f, -4.4081e-01f, -1.6764e+00f, -5.7265e-01f, 

        -7.9995e-01f, -1.2819e+00f, -9.9143e-01f, -3.5979e-01f, -1.9369e+00f, -1.9090e+00f, -3.9470e-01f, -1.9416e+00f, 

        -1.8558e+00f, -7.9273e-01f, -2.3415e-01f, -1.7669e+00f, -1.8147e+00f, -3.5244e-01f, -1.3682e+00f, -1.7676e+00f, 

        -1.1353e-01f, -1.0208e+00f, -2.1526e-03f, -8.1201e-01f, -2.4383e-01f, -9.3216e-01f, -7.4907e-01f, -3.8645e-01f, 

        -3.9301e+00f, -2.9816e-01f, 2.8452e-01f, -9.2001e-01f, -1.5131e+00f, 4.7696e-03f, -1.4842e+00f, -1.3977e+00f, 

        -4.7910e-01f, -2.3030e+00f, 1.9083e-01f, -4.1727e-01f, -9.6217e-01f, -3.7845e-01f, -4.8114e-01f, -7.0577e-01f, 

        -2.5860e-01f, -1.2329e+00f, -7.1032e-01f, -6.8216e-01f, -2.4179e-02f, -1.4260e+00f, -1.2794e+00f, -5.5081e-01f, 

        -2.7140e-01f, -1.7851e+00f, -1.7403e+00f, -3.8174e-01f, -1.2591e-01f, -5.3523e-01f, -1.9001e+00f, -1.7905e+00f, 

        8.1961e-02f, -5.3912e-01f, -8.1811e-01f, -2.7944e-01f, -9.0641e-01f, -7.4051e-01f, 3.2276e-02f, -7.2976e-01f, 

        1.6122e-01f, -6.1583e-01f, -1.2404e+00f, -1.1678e+00f, -2.4891e+00f, -1.8683e+00f, -1.1420e+00f, -5.5160e-02f, 

        4.6332e-01f, -8.5227e-01f, -1.1783e+00f, -6.2808e-01f, -5.4135e-01f, -1.2472e+00f, -5.8422e-01f, -1.8023e+00f, 

        -2.5199e+00f, 3.6299e-01f, -8.7555e-01f, -2.2653e+00f, -1.9000e+00f, -5.9774e-01f, -7.5087e-01f, -1.0825e+00f, 

        -4.0462e-01f, -8.4023e-01f, -6.2831e-01f, 1.2394e-01f, -3.4050e-01f, -1.4731e+00f, -2.9352e-01f, -8.6414e-01f, 

        -1.8024e-01f, -8.0118e-01f, -5.8613e-01f, -1.3077e-01f, -1.0836e+00f, -4.7923e-02f, -1.7814e+00f, -3.4811e-01f, 

        -1.0131e+00f, -5.5095e-01f, -1.3347e+00f, -1.6028e+00f, -3.2471e-01f, 1.0994e-01f, -6.8264e-01f, -3.1607e-01f, 

        -1.3215e+00f, -3.9246e-01f, 4.3983e-03f, -1.1410e-01f, -3.4668e-01f, -1.2836e+00f, -1.7543e+00f, 5.2595e-02f, 

        -5.3901e-01f, -7.1845e-01f, -1.0774e+00f, -3.5557e-01f, -7.5029e-01f, -1.7526e+00f, -1.2498e+00f, -1.5635e-01f, 

        -3.4639e-02f, 6.9167e-02f, -1.9475e-01f, -1.3796e+00f, -3.1041e+00f, -5.3753e-01f, 1.1609e+00f, 1.1094e+00f
    };

    // Layer 2 feedforward weights - fully connected (38x7)
    // Stored in INPUT-MAJOR order: [in0→all_neurons, in1→all_neurons, ...]
    float fc2_weights_vector[266] = {
        4.3294e-03f, -6.3588e-02f, -4.0074e-02f, 1.7673e-01f, 7.1278e-02f, -4.2052e-02f, 6.3421e-02f, 1.9530e-01f, 

        1.0890e-01f, -1.8171e-01f, 9.7261e-02f, -8.8324e-02f, 1.4802e-01f, 1.1967e-01f, -1.2668e+00f, -4.0294e+00f, 

        -1.0909e+00f, 1.6181e+00f, 4.9746e-01f, -7.0147e-01f, 8.2875e-01f, 1.5205e-01f, -4.7285e-02f, 1.2611e-01f, 

        -1.2166e-02f, -6.6316e-02f, -3.8422e-02f, 7.6980e-02f, 5.4077e-01f, -2.6071e+00f, 5.2935e-01f, 6.4869e-01f, 

        8.2831e-01f, 3.7463e-01f, -1.2018e+00f, -1.1778e-01f, 2.2657e-02f, 7.0984e-03f, 1.5507e-01f, -1.7385e-01f, 

        -1.2896e-01f, -1.5374e-01f, -1.5384e-02f, -1.6148e-01f, 2.4589e-02f, 1.8070e-01f, 6.3182e-02f, 6.6852e-02f, 

        -2.1782e-01f, -3.6630e-02f, 9.3111e-02f, 1.0007e-01f, 5.2909e-02f, 4.7567e-02f, 1.1575e-01f, -1.0214e-01f, 

        2.7584e-02f, -9.8483e-02f, -8.0035e-02f, 2.1954e-02f, 3.2610e-02f, 2.7244e-02f, 1.1105e-01f, 1.5327e-01f, 

        -7.3142e-04f, -1.3539e-01f, -1.8553e-01f, 7.4904e-02f, 1.1587e-01f, -2.0502e-01f, 2.0528e-01f, -6.8917e-02f, 

        3.0912e-02f, -2.0067e-01f, -2.2060e-02f, 1.5903e-01f, -1.7154e-01f, -3.4654e-01f, -3.0642e-01f, 8.2367e-02f, 

        2.9089e-01f, -2.7919e-03f, -2.6399e-02f, 1.0410e+00f, -2.9778e-01f, 1.0988e+00f, -1.1290e+00f, -2.0143e-01f, 

        5.0068e-02f, -8.7909e-01f, 3.5670e-01f, -2.3227e-02f, -2.6227e-02f, 1.8412e-01f, 2.0803e-02f, 1.0133e-01f, 

        1.5070e-01f, 1.3038e-01f, -3.5642e-02f, 9.3953e-02f, 2.7819e-02f, -1.3470e-01f, -1.2037e-01f, -9.0799e-02f, 

        -1.2247e-01f, -4.8222e-02f, 9.7134e-02f, 2.8809e-02f, -7.2615e-02f, 1.8378e-02f, 1.1513e-01f, -1.0828e-01f, 

        -6.1462e-02f, 8.7758e-02f, -2.5148e-02f, 1.0527e-01f, -1.8017e-01f, 2.1120e-01f, 1.4823e-01f, -9.8388e-02f, 

        -9.6568e-02f, 3.0299e-02f, -3.3810e-02f, 3.5493e-02f, -2.0603e-02f, 1.7525e-01f, 1.0582e-01f, -7.0581e-02f, 

        4.3427e-02f, -4.8811e-02f, 7.8768e-03f, -5.5647e-02f, 8.5697e-02f, -1.3083e-01f, 3.5726e-02f, 4.6642e-03f, 

        2.5734e-01f, 8.2796e-02f, -2.1737e-01f, -3.5080e-02f, -1.0264e-01f, -4.4265e-02f, 6.3030e-02f, -1.5868e-01f, 

        6.9624e-02f, 4.0740e-02f, 5.3464e-02f, -4.9168e+00f, 6.7115e-01f, 6.6686e-01f, 5.7133e-01f, 6.4798e-01f, 

        8.9858e-01f, 6.4441e-01f, 1.0613e-01f, -6.9400e-02f, -6.3374e-02f, 1.1298e-03f, 8.4993e-02f, 2.3410e-02f, 

        8.4385e-03f, -1.8557e-01f, -8.0540e-02f, -6.8727e-02f, -6.4221e-02f, -1.7348e-01f, 2.8163e-02f, -1.0308e-01f, 

        1.2237e-01f, 6.6031e-02f, -8.1052e-02f, 6.2278e-02f, -2.3823e-02f, -1.4094e-01f, 1.1347e-01f, 5.9500e-02f, 

        -1.1479e-02f, -4.6076e-02f, 1.3016e-01f, 2.8068e-02f, 8.4513e-02f, -1.1900e-01f, -1.0776e-01f, 7.7155e-02f, 

        1.5805e-01f, 1.3958e-01f, 2.3275e-01f, -3.1525e-02f, -1.6941e-01f, 2.5241e-02f, 1.1950e-01f, -9.5598e-02f, 

        3.1405e-02f, 3.2216e-02f, 1.4885e-02f, 4.1623e-02f, -9.7582e-02f, 1.2959e-01f, 3.9850e-02f, -1.0533e-01f, 

        -2.2424e-01f, 2.5606e-02f, 8.2999e-02f, 1.4818e-01f, 1.0961e-01f, 2.7736e-02f, -2.2160e-02f, 1.2342e-01f, 

        -1.1334e-01f, 5.1269e-02f, 1.6410e-01f, -1.0251e-02f, -3.2055e-02f, -1.1471e-01f, -1.1478e-01f, 1.2645e-01f, 

        1.1049e-02f, -3.9882e-02f, 9.9907e-01f, 9.9720e-01f, 4.3596e-02f, -5.2851e-01f, 6.0335e-01f, -4.6474e-02f, 

        -1.0954e-01f, -2.0444e-01f, 2.1656e-01f, 2.7856e-01f, -9.5272e-02f, -9.0387e-03f, 1.5014e+00f, -4.5352e-02f, 

        -8.3054e-02f, 1.0707e-02f, -9.2774e-02f, -5.0739e-02f, -2.1108e-01f, -1.8000e-01f, -3.2001e-01f, 4.5090e-01f, 

        4.9539e-01f, 4.9798e-01f, 3.4333e-01f, -1.1041e+00f, 1.8435e-01f, 8.1426e-02f, -1.2174e-01f, -9.9190e-02f, 

        1.6740e-01f, -7.0809e-03f, -1.7749e-02f, 4.7009e-02f, 9.9498e-01f, -5.1897e-02f, -3.8636e-01f, -1.7959e+00f, 

        -9.2722e-01f, 6.3915e-01f, 5.3989e-01f, 4.4296e-01f, -5.7580e-01f, -4.6359e-01f, -6.4560e-01f, -6.4766e-01f, 

        3.9866e-01f, 6.0380e-01f
    };

    // Layer 1 recurrent weights - 1-to-1 (vector of 38 values)
    float recurrent_weights_layer1[38] = {
        8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 

        8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 

        8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 

        8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 

        8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f, 8.0611e-01f
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
    LIFNeuron_Layer_Update_Vectorized(layer1, input_spikes, weights1, NUM_INPUTS, NUM_NEURONS_LAYER1, l1_spikes, l1_spikes_prev, recurrent_weights1, 0);

    // Layer 2 (no recurrent, fully connected)
    LIFNeuron_Layer_Update_Vectorized_NoRecurrent(layer2, l1_spikes, weights2, NUM_NEURONS_LAYER1, NUM_NEURONS_LAYER2, l2_spikes, 0);

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
