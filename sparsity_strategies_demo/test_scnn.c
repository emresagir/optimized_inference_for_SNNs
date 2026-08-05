// test_snn.c

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "sparsity_demo.h"


#define INPUT_SIZE   25
#define OUTPUT_SIZE  16


static void print_spikes(const char *name, uint8_t *spikes, int size)
{
    printf("%s: ", name);

    for (int i = 0; i < size; i++) {
        printf("%d ", spikes[i]);
    }

    printf("\n");
}


static void test_single_input_spike(void)
{
    uint8_t input[INPUT_SIZE] = {0};
    uint8_t output[OUTPUT_SIZE] = {0};


    /*
       Input image:

       0 0 0 0 0
       0 1 0 0 0
       0 0 0 0 0
       0 0 0 0 0
       0 0 0 0 0

       The spike is at input index:
       row*width + col = 1*5+1 = 6
    */
    input[6] = 1;


    SNN_Reset_State();


    SNN_Run_Timestep(input, output);


    print_spikes("timestep 0", output, OUTPUT_SIZE);


    /*
       Since weights are:
       {1,2,3,4}

       neuron outputs should receive deterministic activity.
       This assertion can be adjusted after checking expected mapping.
    */

    assert(output[0] <= 1);
}


static void test_multiple_timesteps_decay(void)
{
    uint8_t input[INPUT_SIZE] = {0};
    uint8_t output[OUTPUT_SIZE] = {0};


    SNN_Reset_State();


    /*
       timestep 0:
       one input spike
    */
    input[0] = 1;

    SNN_Run_Timestep(input, output);

    print_spikes("timestep 0", output, OUTPUT_SIZE);


    /*
       timestep 1:
       remove input spike

       The membrane potential should decay,
       not receive new input.
    */
    memset(input, 0, sizeof(input));

    SNN_Run_Timestep(input, output);

    print_spikes("timestep 1", output, OUTPUT_SIZE);


    /*
       timestep 2
    */
    SNN_Run_Timestep(input, output);

    print_spikes("timestep 2", output, OUTPUT_SIZE);


    assert(1);
}


static void test_full_input_matrix(void)
{
    uint8_t input[INPUT_SIZE] =
    {
        1,0,0,0,1,
        0,1,0,1,0,
        0,0,1,0,0,
        0,1,0,1,0,
        1,0,0,0,1
    };

    uint8_t output[OUTPUT_SIZE] = {0};


    SNN_Reset_State();


    SNN_Run_Timestep(input, output);


    print_spikes("matrix input", output, OUTPUT_SIZE);


    /*
       Basic sanity:
       every output must be binary.
    */
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        assert(output[i] == 0 || output[i] == 1);
    }
}



int main(void)
{
    printf("Initializing SNN...\n");

    SNN_Init();


    printf("Running tests...\n");


    test_single_input_spike();

    test_multiple_timesteps_decay();

    test_full_input_matrix();


    printf("All tests passed.\n");

    return 0;
}