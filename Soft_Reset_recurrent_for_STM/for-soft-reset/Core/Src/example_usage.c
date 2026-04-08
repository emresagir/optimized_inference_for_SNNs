/*
 * Example usage of the generated SNN code
 * 
 * This shows how to use the NIR-generated SNN in your main.c
 * 
 * NOTE: Input must be binary spikes (q7_t: 0 or 1)
 */

#include "lif_neuron_gen.h"

void example_usage(void) {
    // 1. Initialize the SNN (call once at startup)
    SNN_Init();
    
    // 2. Prepare input spikes (size: 12)
    // Input must be binary values: 0 (no spike) or 1 (spike)
    q7_t input_spikes[12];
    q7_t output_spikes[7];
    
    // Example: Set some input spikes
    for (int i = 0; i < 12; i++) {
        input_spikes[i] = (i % 2 == 0) ? 1 : 0; // Example pattern
    }
    
    // 3. Run for multiple timesteps (e.g., 256 timesteps per sample)
    for (int t = 0; t < 256; t++) {
        SNN_Run_Timestep(input_spikes, output_spikes);
        
        // Process output_spikes as needed
        // output_spikes contains 7 values (0 or 1)
    }
    
    // 4. Reset state between samples
    SNN_Reset_State(); // (Only if testing from dataset, or if you want to reset the network state)
}

/*
 * Network Architecture:
 * Input: 12 neurons (binary spikes: 0 or 1)
 * Layer 1: 38 neurons (fully connected, with 1-to-1 recurrent, uniform params)
 * Layer 2: 7 neurons (fully connected, feedforward only, uniform params)
 * Output: 7 neurons (binary spikes: 0 or 1)
 * 
 * Total parameters: 722 feedforward weights
 * Recurrent parameters: 38 recurrent weights
 * 
 * Input Format:
 * - Binary spikes only: q7_t values must be 0 or 1
 * - If you have analog/float sensor data, convert to binary before calling SNN_Run_Timestep()
 * - Conversion method depends on your application (threshold, rate coding, etc.)
 */
