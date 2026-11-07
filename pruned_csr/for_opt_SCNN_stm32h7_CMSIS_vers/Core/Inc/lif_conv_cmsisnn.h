#ifndef LIF_CONV_CMSISNN_H
#define LIF_CONV_CMSISNN_H



#include <stdint.h>
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "lif_neuron_gen.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Define if your arm_convolve_s16 takes `const cmsis_nn_bias_data *` instead
 * of `const int64_t *` (newer CMSIS-NN releases). Must be defined before this
 * header is included (e.g. in compiler flags) so the .c and .h agree. */
#define CMSISNN_BIAS_AS_STRUCT 1 */

/* Size of the shared CMSIS-NN scratch buffer defined in lif_conv_cmsisnn.c.
 * Override with a compiler flag if a layer needs more. */
#ifndef CMSISNN_SCRATCH_BYTES
#define CMSISNN_SCRATCH_BYTES 16384
#endif

/* ------------------------------------------------------------------------ */
/* Cycle statistics (DWT->CYCCNT), measured around arm_convolve_s16() only    */
/* ------------------------------------------------------------------------ */
typedef struct {
    uint32_t last;
    uint32_t min;
    uint32_t max;
    uint64_t total;
    uint32_t calls;
} CMSISNN_CycleStats;

/* ------------------------------------------------------------------------ */
/* Layer descriptor                                                          */
/* ------------------------------------------------------------------------ */
typedef struct {
    /* geometry */
    uint16_t in_ch, in_h, in_w;
    uint16_t out_ch, out_h, out_w;

    /* CMSIS-NN descriptors */
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant;
    cmsis_nn_context ctx;

    /* buffers (provided by CMSISNN_CONV_DECLARE) */
    int8_t  *weights_s8;   /* OHWI, per-channel int8 */
    int32_t *mult;         /* per-channel requant multiplier */
    int32_t *shift;        /* per-channel requant shift */
    int64_t *bias;         /* all zero (generator does not support bias) */
    int16_t *in_buf;       /* NHWC input */
    int16_t *out_buf;      /* NHWC output, Q15 synaptic current */

    CMSISNN_CycleStats cycles;
} CMSISNN_ConvLayer;

/* Declares all static storage for one conv layer, e.g.
 *   CMSISNN_CONV_DECLARE(cnn1, L1_OUT_CH, L1_IN_CH, L1_KERNEL_H, L1_KERNEL_W,
 *                        IN_H, IN_W, L1_OUT_H, L1_OUT_W);
 * Creates: cnn1 (CMSISNN_ConvLayer), cnn1_w, cnn1_mult, cnn1_shift, cnn1_bias,
 *          cnn1_in, cnn1_out. */
#define CMSISNN_CONV_DECLARE(NAME, OC, IC, KH, KW, IH, IW, OH, OW)                  \
    static __attribute__((aligned(32))) int8_t  NAME##_w[(OC)*(IC)*(KH)*(KW)];     \
    static int32_t NAME##_mult[(OC)];                                               \
    static int32_t NAME##_shift[(OC)];                                              \
    static int64_t NAME##_bias[(OC)];                                               \
    static __attribute__((aligned(32))) int16_t NAME##_in[(IC)*(IH)*(IW)];          \
    static __attribute__((aligned(32))) int16_t NAME##_out[(OC)*(OH)*(OW)];         \
    static CMSISNN_ConvLayer NAME

/* ------------------------------------------------------------------------ */
/* API                                                                       */
/* ------------------------------------------------------------------------ */

/* Requantizes the generator's Q15 OIHW weights (weights{N}) to per-channel
 * int8 OHWI, builds multiplier/shift so the conv output is in Q15 current
 * units, fills the CMSIS-NN descriptors, checks the scratch size and enables
 * the DWT cycle counter. Call AFTER Load_NIR_Weights().
 * Returns 0 on success, negative on error (scratch buffer too small). */
int CMSISNN_Conv_Init(CMSISNN_ConvLayer *L,
                      const q15_t *w_q15,
                      int8_t *w_s8, int32_t *mult, int32_t *shift, int64_t *bias,
                      int16_t *in_buf, int16_t *out_buf,
                      uint16_t in_ch, uint16_t in_h, uint16_t in_w,
                      uint16_t out_ch, uint16_t out_h, uint16_t out_w,
                      uint16_t kh, uint16_t kw,
                      uint16_t stride_h, uint16_t stride_w,
                      uint16_t pad_h, uint16_t pad_w,
                      uint16_t dil_h, uint16_t dil_w);

/* One timestep of a conv + LIF (reset-by-subtract) layer.
 * input_spikes / output_spikes use the generator's CHW layout. */
void LIFNeuron_Conv2d_Update_Subtract_CMSISNN(CMSISNN_ConvLayer *L,
                                              LIFNeuron *neurons,
                                              const q7_t *input_spikes,
                                              q7_t *output_spikes);

/* Prints last/min/max/avg/total cycles of arm_convolve_s16 over USART. */
void CMSISNN_Cycles_Print(const CMSISNN_ConvLayer *L, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* LIF_CONV_CMSISNN_H */