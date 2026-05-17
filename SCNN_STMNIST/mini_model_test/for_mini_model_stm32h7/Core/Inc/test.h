/*
 * test_data.h — Mini Conv2d golden reference test vectors
 *
 * Input: C=1, H=3, W=3 → 9 inputs flattened
 * Timesteps: 5
 * Output neurons: 8 (conv output 2*2*2 flattened)
 */

#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <stdint.h>

/* ── Constants ──────────────────────────────────────── */
#define TD_NUM_INPUTS    9      /* C*H*W = 1*3*3 */
#define TD_NUM_CLASSES   8      /* out_c*out_H*out_W = 2*2*2 */
#define TD_NUM_SAMPLES   1

/* ── Per-sample descriptor ───────────────────────────── */
typedef struct {
    const int8_t  (*spikes)[TD_NUM_INPUTS]; /* [timestep][input_idx] */
    uint16_t       num_timesteps;
    uint8_t        label;
} TestSample;

/*
 * Input layout: flattened [C, H, W] row-major
 * T=0: [[1,0,1],[0,1,0],[1,0,1]]
 * T=1: [[0,1,0],[1,1,1],[0,1,0]]
 * T=2: [[1,1,0],[0,0,1],[1,0,0]]
 * T=3: all zeros (decay)
 * T=4: all zeros (decay)
 */
static const int8_t _spikes_0[5][TD_NUM_INPUTS] = {
    { 1, 0, 1,  0, 1, 0,  1, 0, 1 },  /* T=0 */
    { 0, 1, 0,  1, 1, 1,  0, 1, 0 },  /* T=1 */
    { 1, 1, 0,  0, 0, 1,  1, 0, 0 },  /* T=2 */
    { 0, 0, 0,  0, 0, 0,  0, 0, 0 },  /* T=3 decay */
    { 0, 0, 0,  0, 0, 0,  0, 0, 0 },  /* T=4 decay */
};

/* ── Sample table ───────────────────────────────────── */
static const TestSample test_samples[TD_NUM_SAMPLES] = {
    { _spikes_0, 5, 0 },  /* sample 0: mini conv2d golden reference */
};

#endif /* TEST_DATA_H */