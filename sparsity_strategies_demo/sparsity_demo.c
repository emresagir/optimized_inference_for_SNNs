#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "sparsity_demo.h"


#define NUM_INPUTS 25
#define NUM_INPUT_CHANNEL 1
#define L1_OUT_CH      1
#define L1_IN_CH       1
#define L1_KERNEL_H    2
#define L1_KERNEL_W    2
#define L1_KERNEL_SIZE 4
#define L1_STRIDE_H    1
#define L1_STRIDE_W    1
#define L1_PAD_H       0
#define L1_PAD_W       0
#define L1_OUT_H       4
#define L1_OUT_W       4
#define NUM_NEURONS_LAYER1 16

#define NUM_SURVIVING_CONNECTIONS L1_KERNEL_SIZE * NUM_NEURONS_LAYER1 // Need to calculate how many connections the model will have.
/* For 5x5 - (2x2)Kernel - (4x4) 

2x2x4x4 = 64
L1_KERNEL_SIZE * NUM_NEURONS_LAYER1

*/


uint16_t weights1[L1_OUT_CH * L1_IN_CH * L1_KERNEL_H * L1_KERNEL_W] = {60, 120, 180, 240};

/* Example vector table in mind:  
in0,o0,w0   (1 entry for in0)
in1,o0,w1   (2 entry for in1,2)
in1,o1,w0
.. 
in4,o3,w1   (1)
..
in6,o0,w3   (4 entry for in6,7...19)
in6,o1,w2
in6,o4,w1
in6,o5,w0
..
in20,o12,w2 (1)
in21,o12,w3 (2)
in21,o13,w2 
..
in24,o15,w3 (1)

So at every timestep I need to use this table and refresh the input values?
*/
uint16_t row_ptr[NUM_INPUTS + 1];   // This will skip over the rows, basically stepping over the inputs.
uint16_t out_idx[NUM_SURVIVING_CONNECTIONS];   // which output is being fed
uint8_t  weight_idx[NUM_SURVIVING_CONNECTIONS]; // index into shared kernel-weight table


LIFNeuron layer1[NUM_NEURONS_LAYER1];
uint8_t l1_spikes[NUM_NEURONS_LAYER1];


void LIFNeuron_Init(LIFNeuron* neuron, uint16_t threshold, uint16_t reset_value) {
    neuron->threshold = threshold;
    neuron->reset_value = reset_value;
    neuron->membrane_potential = reset_value;
    // decay_factor (beta) will be set in SNN_Init
}

// Building the CSR table with the basic convolution geometry.
size_t Build_Conv_CSR(
    uint16_t in_h, uint16_t in_w,
    uint16_t in_ch,
    uint16_t out_h, uint16_t out_w,
    uint16_t out_ch,
    uint16_t kh, uint16_t kw,
    uint16_t stride,
    uint16_t padding,

    // CSR table elements.
    uint16_t* row_ptr,
    uint16_t* out_idx,
    uint8_t*  connection_weight_idx,

    size_t connection_limit

) {
    const uint32_t num_inputs =
        (uint32_t)in_ch * in_h * in_w;

    const uint32_t num_outputs =
        (uint32_t)out_ch * out_h * out_w;

    const uint32_t num_weights =
        (uint32_t)out_ch * in_ch * kh * kw;


    // NULL and size checks
    if (row_ptr == NULL ||
        out_idx == NULL ||
        connection_weight_idx == NULL ||
        stride == 0) {
        return SIZE_MAX;
    }

    if (num_inputs > UINT16_MAX ||
        num_outputs > UINT16_MAX ||
        num_weights > UINT8_MAX + 1U) {
        return SIZE_MAX;
    }

    // First pass to learn the connections.
    // At every input_index, increase the connection count of that specific row_ptr array.
    // When input_index 1 is found, increment the row_ptr[1] by 1. 
    // This way we will know how many connections each input neuron has.

    for (uint16_t oc = 0; oc < out_ch; ++oc) {
        for (uint16_t oh = 0; oh < out_h; ++oh) {
            for (uint16_t ow = 0; ow < out_w; ++ow) {
                for (uint16_t ic = 0; ic < in_ch; ++ic) {
                    for (uint16_t fy = 0; fy < kh; ++fy) {
                        for (uint16_t fx = 0; fx < kw; ++fx) {

                            // Finding the input coordinates
                            const int32_t ih = (int32_t)oh * stride + fy - padding;
                            const int32_t iw = (int32_t)ow * stride + fx - padding;

                            // Boundary check
                            if (ih < 0 || ih >= in_h ||
                                iw < 0 || iw >= in_w) {
                                continue;
                            }

                            // Finding the input index
                            const uint32_t input_index =
                                (uint32_t)ic * in_h * in_w +
                                (uint32_t)ih * in_w +
                                (uint32_t)iw;

                            /*
                             * Store the count for row input_index in
                             * row_ptr[input_index + 1].
                             */
                            if (row_ptr[input_index + 1U] == UINT16_MAX) {
                                return SIZE_MAX;
                            }

                            row_ptr[input_index + 1U]++;
                        }
                    }
                }
            }
        }
    }


    /* Converting row counts into CSR row offsets. 
    
    In previous pass, every row_ptr element has the value of the connection count. 

    1
    2
    2
    2
    1
    ...

    turns into 

    0
    1
    3
    5
    7
    8
    ...

    */

    for (uint32_t i = 0; i < num_inputs; ++i) {
        const uint32_t next = (uint32_t)row_ptr[i] + row_ptr[i + 1U];

        if (next > UINT16_MAX) {
            return SIZE_MAX;
        }

        row_ptr[i + 1U] = (uint16_t)next;
    }

    const size_t num_connections = row_ptr[num_inputs];

    if (num_connections > connection_limit) {
        return SIZE_MAX;
    }



    // Creating a cursor to be used for each row_ptr.
    // When a connection has found to a specific row(input) 
    // the out_index and weigh_index will be filled, 
    // and the cursor will be incremented for the next connection for that specific row(input).

    uint16_t cursor[num_inputs];

    for (uint32_t i = 0; i < num_inputs; ++i) {
        cursor[i] = row_ptr[i];
    }

    /*
     * Second pass:
     * Fill the connection arrays.
     */
    for (uint16_t oc = 0; oc < out_ch; ++oc) {
        for (uint16_t oh = 0; oh < out_h; ++oh) {
            for (uint16_t ow = 0; ow < out_w; ++ow) {

                const uint32_t output_index =
                    (uint32_t)oc * out_h * out_w +
                    (uint32_t)oh * out_w +
                    (uint32_t)ow;

                for (uint16_t ic = 0; ic < in_ch; ++ic) {
                    for (uint16_t fy = 0; fy < kh; ++fy) {
                        for (uint16_t fx = 0; fx < kw; ++fx) {

                            const int32_t ih = (int32_t)oh * stride + fy - padding;
                            const int32_t iw = (int32_t)ow * stride + fx - padding;

                            if (ih < 0 || ih >= in_h ||
                                iw < 0 || iw >= in_w) {
                                continue;
                            }

                            const uint32_t input_index =
                                (uint32_t)ic * in_h * in_w +
                                (uint32_t)ih * in_w +
                                (uint32_t)iw;

                            const uint32_t kernel_weight_index =
                                (uint32_t)oc * in_ch * kh * kw +
                                (uint32_t)ic * kh * kw +
                                (uint32_t)fy * kw +
                                (uint32_t)fx;

                            // After finding the input_index, using it to see which is the first connection of that input_index (row).
                            // And incrementing it for the next hit for that specific input_index(row). 
                            const uint16_t connection = cursor[input_index]++;

                            out_idx[connection] = (uint16_t)output_index;

                            connection_weight_idx[connection] = (uint8_t)kernel_weight_index;
                        }
                    }
                }
            }
        }
    }
    
    return num_connections;

}

void LIFNeuron_Conv2d_Update_Subtract_Demo(LIFNeuron* neurons,         // Array of neurons for this layer
    const uint8_t* input_spikes,  // Input feature map [In_CH * In_H * In_W]
    const uint16_t* weights,       // Weights [Out_CH * In_CH * KH * KW]
    uint8_t* output_spikes,        // Output spikes [Out_CH * Out_H * Out_W]
    uint16_t num_inputs,
    uint16_t num_outputs,
    uint16_t in_h, uint16_t in_w,
    uint16_t in_ch,
    uint16_t out_h, uint16_t out_w,
    uint16_t out_ch,
    uint16_t kh, uint16_t kw,
    uint16_t stride,
    uint16_t padding
) {

    // U(t+1) = (U(t)*Beta) + W*X(t+1) - Uth*S(t)
    // This is the equation from the snntorch tutorial 3. 


    // Decay (U(t)*Beta) (Can optimize with vectorized)
    for (uint16_t output = 0; output < num_outputs; ++output) {
        uint32_t decayed = (neurons[output].membrane_potential * neurons[output].decay_factor) >> 8; // Divide by 256
        neurons[output].membrane_potential = (uint16_t)decayed;
    }


    // Accumulate contributions from active inputs.
    //(U(t)*Beta) + W*X(t+1)
    for (uint16_t input = 0; input < num_inputs; ++input) {
        const uint8_t spike = input_spikes[input];

        if (spike == 0) {
            continue;
        }

        for (uint16_t connection = row_ptr[input]; connection < row_ptr[input + 1U]; ++connection) {
            const uint16_t output = out_idx[connection];
            const uint8_t weight_index = weight_idx[connection];
            
            neurons[output].membrane_potential += weights[weight_index];
            
            printf("[DEBUG] Input %d spiked -> Added Weight W[%d]=%u to Output %d (New Pot: %u)\n", 
                   input, weight_index, weights[weight_index], output, neurons[output].membrane_potential);
        }
    }

    
    
    for (uint16_t output = 0; output < num_outputs; ++output) {
        
        // Subtract if there was a spike in the previous timestep
        // U(t+1) = (U(t)*Beta) + W*X(t+1) - Uth*S(t) (Can optimize with vectorized)

        if (neurons[output].membrane_potential >= neurons[output].reset_value) {
            neurons[output].membrane_potential -= neurons[output].reset_value;
        } else {
            neurons[output].membrane_potential = 0; // Floor to 0 safely
        }
        
        // Spike check
        if (neurons[output].membrane_potential >= neurons[output].threshold) 
        {
            output_spikes[output] = 1;

            // Soft Reset (subtract at the next step)
            neurons[output].reset_value = neurons[output].threshold;

            printf("[DEBUG] Output %d SPIKED! (Pot: %u >= Thresh: %u)\n", 
                   output, neurons[output].membrane_potential, neurons[output].threshold);
        } else {
            output_spikes[output] = 0;
            neurons[output].reset_value = 0; 
        }
    }

}



void SNN_Init(void) {
    const float scale = 60.0f;

    float threshold_f_1 = 1.0000e+00;
    float reset_value_f_1 = 0.0000e+00;
    float beta_1 = 9.5000e-01f;

    uint16_t threshold_1 = (uint16_t)(threshold_f_1 * scale);        // 60
    uint16_t reset_value_1 = (uint16_t)(reset_value_f_1 * scale);    // 0
    
    // Fixed-point representation for decay (multiply by 256)
    // 0.95 * 256 = 243
    uint16_t decay_factor_1 = (uint16_t)(beta_1 * 256.0f); 

    printf("[DEBUG] SNN_Init: Threshold=%u, Reset=%u, Decay=%u/256\n", 
           threshold_1, reset_value_1, decay_factor_1);

    for (int i = 0; i < NUM_NEURONS_LAYER1; i++) {
        LIFNeuron_Init(&layer1[i], threshold_1, reset_value_1);
        layer1[i].decay_factor = decay_factor_1;
    }

    size_t connection_count = Build_Conv_CSR(
        5, 5, 1, 4, 4, 1, 2, 2, 1, 0,
        row_ptr, out_idx, weight_idx, NUM_SURVIVING_CONNECTIONS
    );

    if (connection_count == SIZE_MAX) {
        printf("[ERROR] Failed to build CSR tables.\n");
    } else {
        printf("[DEBUG] CSR Table built with %zu connections.\n", connection_count);
    }
}


void SNN_Run_Timestep(const uint8_t* input_spikes, uint8_t* output_spikes) {
    // Layer 1 (convolutional)
    LIFNeuron_Conv2d_Update_Subtract_Demo(layer1, input_spikes, weights1, l1_spikes, NUM_INPUTS, NUM_NEURONS_LAYER1, 5, 5, 1, 4, 4, 1, 2, 2, 1, 0);

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

