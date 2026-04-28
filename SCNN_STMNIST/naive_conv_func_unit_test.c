#include <stdio.h>
#include <stdint.h>

typedef int16_t q15_t;
typedef int32_t q31_t;
typedef int8_t  q7_t;

typedef struct {
    q15_t membrane_potential;
    q15_t threshold;
    q15_t reset_value;
    q15_t decay_factor;
} LIFNeuron;

// void LIFNeuron_Conv2d_Update_Standard(LIFNeuron* neurons,         // Array of neurons for this layer
//     const q15_t* input_spikes,  // Input feature map [In_CH * In_H * In_W]
//     const q15_t* weights,       // Weights [Out_CH * In_CH * KH * KW]
//     q7_t* output_spikes,        // Output spikes [Out_CH * Out_H * Out_W]
//     uint16_t in_h, uint16_t in_w,
//     uint16_t in_ch,
//     uint16_t out_h, uint16_t out_w,
//     uint16_t out_ch,
//     uint16_t kh, uint16_t kw,
//     uint16_t stride,
//     uint16_t padding
// ) {
//     // 1. Iterate over every output "pixel" (which is one LIF neuron)
//     for (uint16_t oc = 0; oc < out_ch; oc++) {
//         for (uint16_t oh = 0; oh < out_h; oh++) {
//             for (uint16_t ow = 0; ow < out_w; ow++) {
                
//                 // Accumulator for the current (this is the weighted input)
//                 q31_t acc = 0; 

//                 // 2. Perform the Convolution (Sliding Window)
//                 for (uint16_t ic = 0; ic < in_ch; ic++) {
//                     for (uint16_t fy = 0; fy < kh; fy++) {
//                         for (uint16_t fx = 0; fx < kw; fx++) {
                            
//                             // Calculate input coordinates
//                             int16_t ih = oh * stride + fy - padding;
//                             int16_t iw = ow * stride + fx - padding;

//                             // Check boundaries (Padding logic)
//                             if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
//                                 // Indexing for [CH][H][W] format
//                                 uint32_t input_idx = (ic * in_h * in_w) + (ih * in_w) + iw;
//                                 // Indexing for [OutCH][InCH][KH][KW] format
//                                 uint32_t weight_idx = (oc * in_ch * kh * kw) + (ic * kh * kw) + (fy * kw) + fx;

//                                 acc += (q31_t)input_spikes[input_idx] * weights[weight_idx];
//                             }
//                         }
//                     }
//                 }

//                 // 3. LIF Neuron Update Logic
//                 // Index of the specific neuron in the flat array
//                 uint32_t n_idx = (oc * out_h * out_w) + (oh * out_w) + ow;
                
//                 // Scale back the accumulated value (Q15 * Q15 results in Q30)
//                 q15_t weighted_input = (q15_t)(acc >> 15);

//                 // V = V_prev * beta + weighted_input
//                 // (Simplified for standard C, assuming reset to 0)
//                 neurons[n_idx].membrane_potential = 
//                     ((q31_t)neurons[n_idx].membrane_potential * neurons[n_idx].decay_factor >> 15) + weighted_input;

//                 // Thresholding
//                 if (neurons[n_idx].membrane_potential >= neurons[n_idx].threshold) {
//                     output_spikes[n_idx] = 1;
//                     neurons[n_idx].membrane_potential = neurons[n_idx].reset_value;
//                 } else {
//                     output_spikes[n_idx] = 0;
//                 }
//             }
//         }
//     }
// }


// w debug
void LIFNeuron_Conv2d_Update_Standard(
    LIFNeuron* neurons, const q15_t* input_spikes, const q15_t* weights, 
    q7_t* output_spikes, uint16_t in_h, uint16_t in_w, uint16_t in_ch,
    uint16_t out_h, uint16_t out_w, uint16_t out_ch,
    uint16_t kh, uint16_t kw, uint16_t stride, uint16_t padding
) {
    for (uint16_t oc = 0; oc < out_ch; oc++) {
        for (uint16_t oh = 0; oh < out_h; oh++) {
            for (uint16_t ow = 0; ow < out_w; ow++) {
                
                // Accumulate convolution result for this output neuron.
                q31_t acc = 0; 
                // Loop over every input channel and kernel element.
                for (uint16_t ic = 0; ic < in_ch; ic++) {
                    for (uint16_t fy = 0; fy < kh; fy++) {
                        for (uint16_t fx = 0; fx < kw; fx++) {
                            // Compute the corresponding input coordinates for this kernel position.
                            int16_t ih = oh * stride + fy - padding;
                            int16_t iw = ow * stride + fx - padding;

                            // Skip if the kernel window lies outside the input feature map.
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                // Flattened index for input spike tensor [in_ch][in_h][in_w].
                                uint32_t input_idx = (ic * in_h * in_w) + (ih * in_w) + iw;
                                // Flattened index for weight tensor [out_ch][in_ch][kh][kw].
                                uint32_t weight_idx = (oc * in_ch * kh * kw) + (ic * kh * kw) + (fy * kw) + fx;
                                // Multiply-accumulate the input spike and corresponding weight.
                                acc += (q31_t)input_spikes[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }

                uint32_t n_idx = (oc * out_h * out_w) + (oh * out_w) + ow;
                
                // --- DEBUG PRINTS ---
                // Convert acc (Q30) to Q15 for the update
                q15_t weighted_input = (q15_t)(acc >> 15);
                
                printf("Pixel [%d,%d]: Weighted Input (Q15)=%d, ", oh, ow, weighted_input);

                // Update V: V = V_prev * beta + weighted_input
                q31_t next_v = ((q31_t)neurons[n_idx].membrane_potential * neurons[n_idx].decay_factor >> 15) + weighted_input;
                neurons[n_idx].membrane_potential = (q15_t)next_v;

                printf("Membrane V=%d, Threshold=%d\n", neurons[n_idx].membrane_potential, neurons[n_idx].threshold);

                if (neurons[n_idx].membrane_potential >= neurons[n_idx].threshold) {
                    output_spikes[n_idx] = 1;
                    neurons[n_idx].membrane_potential = neurons[n_idx].reset_value;
                } else {
                    output_spikes[n_idx] = 0;
                }
            }
        }
    }
}


int main() {
    // 2 Input Channels, 7x7 Input
    uint16_t in_h = 7, in_w = 7, in_ch = 2;
    uint16_t kh = 3, kw = 3, stride = 2, padding = 0;
    uint16_t out_h = 3, out_w = 3, out_ch = 2; // Increased to 2

    // Total input size: 2 * 7 * 7 = 98 elements
    q15_t input[98]; 
    for(int i=0; i<98; i++) {
        input[i] = 3276 + i*3276; // All 0.1
    }
    // Total weights: out_ch(2) * in_ch(2) * kh(3) * kw(3) = 36   
    // Each filter has 2 channel and 3x3. So one filter has 18 weights. 
    // And the total filter size depends on how many output channel there is.
    /*
    So lets say I have 2x7x7 input. 
    I will name those in1 (first channel 7x7) and in2 (second channel 7x7). 
    I have 3x3 kernel. I have 2x5x5 output. 
    I will name them out1 (5x5) and out2 (5x5) So lets name the kernel and their dimensions. 
    First kernel which creates the result of out1, kernel1 I will name its dimensions f1 (3x3) and f2 (3x3). 
    And the second kernel which creates the result of out2, kernel2 I will name its dimensions f3 (3x3) and f4 (3x3). 
    
    So the convolution happens with: 
    in1 * f1 + in2 * f2 = out1 
    in1 * f3 + in2 * f4 = out2
    */
    q15_t weights[36]; 
    for(int i=0; i<36; i++) weights[i] = 3276; // All 0.1

    // Total neurons/spikes: out_ch(2) * out_h(3) * out_w(3) = 18    
    LIFNeuron neurons[18];
    q7_t output_spikes[18];
    for(int i=0; i<18; i++) {
        neurons[i].membrane_potential = 0;
        neurons[i].threshold = 5000; 
        neurons[i].reset_value = 0;
        neurons[i].decay_factor = 32767;
    }

    LIFNeuron_Conv2d_Update_Standard(neurons, input, weights, output_spikes,
                                    in_h, in_w, in_ch, out_h, out_w, out_ch,
                                    kh, kw, stride, padding);

    printf("Results for 7x7 Input -> 3x3 Output (Stride 2, No Padding):\n");
    for(int k=0; k<out_ch; k++){
        for(int i=0; i<out_h; i++) {
            for(int j=0; j<out_w; j++) {
                printf("%d ", output_spikes[k*out_h*out_w + i*out_w + j]);
            }
            printf("\n");
        }
    }
    return 0;
}