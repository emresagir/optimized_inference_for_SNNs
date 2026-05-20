#ifndef TEST_H
#define TEST_H

#include <stdint.h>
#include "arm_math.h"

// (Sample, Timestep, Input size = 3)
const q7_t test_input[1][5][3] =
{
    {
        {1, 0, 0},   // t=0
        {0, 1, 0},   // t=1
        {0, 0, 1},   // t=2
        {0, 0, 0},   // t=3 (decay only)
        {0, 0, 0}    // t=4 (decay only)
    }
};

const uint32_t class_output[1] = {
    0
};

#endif
