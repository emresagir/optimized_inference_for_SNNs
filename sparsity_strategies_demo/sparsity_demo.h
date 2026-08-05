#include <stdint.h>

typedef struct {
    uint16_t threshold;     // Firing threshold in Q15
    uint16_t reset_value;   // Reset potential in Q15
    uint16_t membrane_potential; // Current membrane potential in Q15
    uint16_t decay_factor;  // Precomputed beta (decay factor) in Q15
} LIFNeuron;


void SNN_Run_Timestep(const uint8_t* input_spikes, uint8_t* output_spikes);

void SNN_Reset_State(void);

void SNN_Init(void);

