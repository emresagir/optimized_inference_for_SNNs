/*
 * lif_conv_cmsisnn.c
 *
 * Replacement for LIFNeuron_Conv2d_Update_Subtract_CSR using
 * CMSIS-NN arm_convolve_s16 for the synaptic accumulation.
 *
 * Everything else matches the generator output:
 *   - same LIFNeuron struct, same Q15 neuron state, same soft reset (subtract
 *     previous threshold), same CHW neuron indexing / CHW output spikes
 *   - weights are taken from the generator's own q15 weights{N} array
 *     (OIHW order), so Load_NIR_Weights() is unchanged
 *
 * What is different (and why):
 *   - CMSIS-NN s16 = int16 activations + INT8 weights. The Q15 weights are
 *     requantized per output channel to int8 at init.
 *   - CMSIS-NN wants NHWC activations and OHWI weights. Layout conversion is
 *     done outside the timed region.
 *   - The conv output is requantized back to Q15 "current" units, so
 *     current = sum(spike * w_q15) approximately, as in the CSR kernel.
 *   - The conv is dense; it does not skip zero spikes.
 *
 * Cycle counting: DWT->CYCCNT is read immediately before and after
 * arm_convolve_s16() only. Layout conversion and the LIF update are NOT
 * included in the count.
 *
 * NOTE: written from memory of the CMSIS-NN API. Struct/field names and the
 * bias argument changed between CMSIS-NN releases. Check against the headers
 * in your project (arm_nnfunctions.h, arm_nn_types.h) and define
 * CMSISNN_BIAS_AS_STRUCT (see lif_conv_cmsisnn.h) if your version takes cmsis_nn_bias_data.
 */

#include "stm32h7xx_hal.h"
#include "../Inc/lif_neuron_gen.h"
#include "../Inc/lif_conv_cmsisnn.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "../Inc/usart.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Shared scratch buffer for CMSIS-NN. Size set by CMSISNN_SCRATCH_BYTES in the
 * header. Init checks it is large enough. */
static __attribute__((aligned(32))) int8_t cmsisnn_scratch[CMSISNN_SCRATCH_BYTES];

/* ------------------------------------------------------------------------ */
/* Cycle counter (DWT)                                                       */
/* ------------------------------------------------------------------------ */
static void CMSISNN_Cycles_Enable(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#if defined(__CORTEX_M) && (__CORTEX_M == 7U)
    DWT->LAR = 0xC5ACCE55;              /* unlock DWT on Cortex-M7 */
#endif
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline void CMSISNN_Cycles_Reset(CMSISNN_CycleStats *s) {
    s->last = 0;
    s->min = 0xFFFFFFFFu;
    s->max = 0;
    s->total = 0;
    s->calls = 0;
}

static inline void CMSISNN_Cycles_Record(CMSISNN_CycleStats *s, uint32_t cycles) {
    s->last = cycles;
    if (cycles < s->min) s->min = cycles;
    if (cycles > s->max) s->max = cycles;
    s->total += cycles;
    s->calls++;
}

/* ------------------------------------------------------------------------ */
/* Init: requantize generator's Q15 OIHW weights -> int8 OHWI + per-channel  */
/* multiplier/shift so that output is in Q15 current units.                  */
/* Call AFTER Load_NIR_Weights().                                            */
/* Returns 0 on success, negative on error.                                  */
/* ------------------------------------------------------------------------ */
int CMSISNN_Conv_Init(CMSISNN_ConvLayer *L,
                      const q15_t *w_q15,               /* weights{N}, OIHW */
                      int8_t *w_s8, int32_t *mult, int32_t *shift, int64_t *bias,
                      int16_t *in_buf, int16_t *out_buf,
                      uint16_t in_ch, uint16_t in_h, uint16_t in_w,
                      uint16_t out_ch, uint16_t out_h, uint16_t out_w,
                      uint16_t kh, uint16_t kw,
                      uint16_t stride_h, uint16_t stride_w,
                      uint16_t pad_h, uint16_t pad_w,
                      uint16_t dil_h, uint16_t dil_w)
{
    memset(L, 0, sizeof(*L));
    L->in_ch = in_ch;   L->in_h = in_h;   L->in_w = in_w;
    L->out_ch = out_ch; L->out_h = out_h; L->out_w = out_w;
    L->weights_s8 = w_s8; L->mult = mult; L->shift = shift; L->bias = bias;
    L->in_buf = in_buf;   L->out_buf = out_buf;

    /* Per-output-channel int8 quantization of the Q15 weights.
     *   w_q15 ~= w_s8 * (maxabs / 127)
     * so the real multiplier that maps int accumulator -> Q15 current is
     *   M = maxabs / 127
     * M is encoded as a REDUCED (Q15) multiplier and a shift, which is what
     * arm_convolve_s16 expects (arm_nn_requantize_s64):
     *   out = acc * multiplier / 2^(15 - shift)
     * i.e. M = multiplier * 2^(shift - 15). This differs from the s8 kernels,
     * which use a Q31 multiplier. */
    const uint32_t per_oc = (uint32_t)in_ch * kh * kw;
    for (uint16_t oc = 0; oc < out_ch; oc++) {
        int32_t maxabs = 0;
        for (uint32_t k = 0; k < per_oc; k++) {
            int32_t v = w_q15[oc * per_oc + k];
            if (v < 0) v = -v;
            if (v > maxabs) maxabs = v;
        }

        bias[oc] = 0;

        if (maxabs == 0) {
            mult[oc] = 0;
            shift[oc] = 0;
            for (uint16_t ic = 0; ic < in_ch; ic++)
                for (uint16_t fy = 0; fy < kh; fy++)
                    for (uint16_t fx = 0; fx < kw; fx++)
                        w_s8[(((uint32_t)oc * kh + fy) * kw + fx) * in_ch + ic] = 0;
            continue;
        }

        const double M = (double)maxabs / 127.0;
        int e;
        double mant = frexp(M, &e);                    /* M = mant * 2^e, mant in [0.5,1) */
        int64_t m31 = (int64_t)llround(mant * 2147483648.0);
        if (m31 == 2147483648LL) { m31 >>= 1; e++; }
        mult[oc]  = (int32_t)m31;
        shift[oc] = (int32_t)e;

        for (uint16_t ic = 0; ic < in_ch; ic++) {
            for (uint16_t fy = 0; fy < kh; fy++) {
                for (uint16_t fx = 0; fx < kw; fx++) {
                    /* source: OIHW (generator), dest: OHWI (CMSIS-NN) */
                    uint32_t src = (((uint32_t)oc * in_ch + ic) * kh + fy) * kw + fx;
                    uint32_t dst = (((uint32_t)oc * kh + fy) * kw + fx) * in_ch + ic;
                    long q = lround((double)w_q15[src] * 127.0 / (double)maxabs);
                    if (q > 127)  q = 127;
                    if (q < -127) q = -127;
                    w_s8[dst] = (int8_t)q;
                }
            }
        }
    }

    /* CMSIS-NN dims: n, h, w, c */
    L->input_dims  = (cmsis_nn_dims){ .n = 1,      .h = in_h,  .w = in_w,  .c = in_ch  };
    L->filter_dims = (cmsis_nn_dims){ .n = out_ch, .h = kh,    .w = kw,    .c = in_ch  };
    L->bias_dims   = (cmsis_nn_dims){ .n = 1,      .h = 1,     .w = 1,     .c = out_ch };
    L->output_dims = (cmsis_nn_dims){ .n = 1,      .h = out_h, .w = out_w, .c = out_ch };

    L->conv_params.input_offset  = 0;
    L->conv_params.output_offset = 0;
    L->conv_params.stride.h   = stride_h;
    L->conv_params.stride.w   = stride_w;
    L->conv_params.padding.h  = pad_h;
    L->conv_params.padding.w  = pad_w;
    L->conv_params.dilation.h = dil_h;
    L->conv_params.dilation.w = dil_w;
    L->conv_params.activation.min = INT16_MIN;
    L->conv_params.activation.max = INT16_MAX;

    L->quant.multiplier = mult;
    L->quant.shift      = shift;

    int32_t need = arm_convolve_s16_get_buffer_size(&L->input_dims, &L->filter_dims);
    if (need > (int32_t)CMSISNN_SCRATCH_BYTES) {
        char buf[96];
        snprintf(buf, sizeof(buf), "CMSIS-NN scratch too small: need %ld, have %d\r\n",
                 (long)need, (int)CMSISNN_SCRATCH_BYTES);
        usart1_print(buf);
        return -1;
    }
    L->ctx.buf  = cmsisnn_scratch;
    L->ctx.size = need;

    CMSISNN_Cycles_Enable();
    CMSISNN_Cycles_Reset(&L->cycles);
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Layer update: CMSIS-NN conv + LIF soft reset (same as the CSR kernel)      */
/* ------------------------------------------------------------------------ */
void LIFNeuron_Conv2d_Update_Subtract_CMSISNN(CMSISNN_ConvLayer *L,
                                              LIFNeuron *neurons,
                                              const q7_t *input_spikes,   /* CHW */
                                              q7_t *output_spikes)        /* CHW */
{
    const uint16_t IC = L->in_ch,  IH = L->in_h,  IW = L->in_w;
    const uint16_t OC = L->out_ch, OH = L->out_h, OW = L->out_w;

    /* 1. Layout conversion CHW q7 -> NHWC int16 (NOT timed). The spike value is
     *    kept as-is, so non-binary inputs (ST-MNIST) still work. */
    for (uint16_t c = 0; c < IC; c++) {
        for (uint16_t h = 0; h < IH; h++) {
            for (uint16_t w = 0; w < IW; w++) {
                L->in_buf[((uint32_t)h * IW + w) * IC + c] =
                    (int16_t)input_spikes[((uint32_t)c * IH + h) * IW + w];
            }
        }
    }

    /* 2. CMSIS-NN convolution: the ONLY region whose cycles are counted. */
    __DSB(); __ISB();
    const uint32_t t0 = DWT->CYCCNT;

#ifdef CMSISNN_BIAS_AS_STRUCT
    /* Built here (not stored in the layer) because the members are const. */
    const cmsis_nn_bias_data bias_struct = { .data = L->bias, .is_int32_bias = false };
    #define CMSISNN_BIAS_ARG (&bias_struct)
#else
    #define CMSISNN_BIAS_ARG (L->bias)
#endif
    int status = (int)arm_convolve_s16(&L->ctx, &L->conv_params, &L->quant,
                                       &L->input_dims,  L->in_buf,
                                       &L->filter_dims, L->weights_s8,
                                       &L->bias_dims,   CMSISNN_BIAS_ARG,
                                       &L->output_dims, L->out_buf);
    #undef CMSISNN_BIAS_ARG

    const uint32_t t1 = DWT->CYCCNT;
    __DSB(); __ISB();

    CMSISNN_Cycles_Record(&L->cycles, t1 - t0);   /* unsigned subtraction handles wrap */

    if (status != 0) {          /* ARM_CMSIS_NN_SUCCESS == 0 */
        usart1_print("arm_convolve_s16 failed\r\n");
        return;
    }

    /* 3. LIF update (same maths as the CSR kernel, CHW neuron indexing).
     *    v = v*beta + current - reset_prev ; spike if v > threshold ;
     *    next reset_value = threshold if spiked else 0. */
    for (uint16_t oc = 0; oc < OC; oc++) {
        for (uint16_t oh = 0; oh < OH; oh++) {
            for (uint16_t ow = 0; ow < OW; ow++) {
                const uint32_t n   = ((uint32_t)oc * OH + oh) * OW + ow;     /* CHW neuron idx */
                const uint32_t cur = ((uint32_t)oh * OW + ow) * OC + oc;     /* NHWC conv out  */

                q31_t v = ((q31_t)neurons[n].membrane_potential *
                           (q31_t)neurons[n].decay_factor) >> 15;
                v = __SSAT(v + (q31_t)L->out_buf[cur], 16);
                v = __SSAT(v - (q31_t)neurons[n].reset_value, 16);
                neurons[n].membrane_potential = (q15_t)v;

                if (neurons[n].membrane_potential > neurons[n].threshold) {
                    output_spikes[n] = 1;
                    neurons[n].reset_value = neurons[n].threshold;
                } else {
                    output_spikes[n] = 0;
                    neurons[n].reset_value = 0;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Reporting                                                                 */
/* ------------------------------------------------------------------------ */
void CMSISNN_Cycles_Print(const CMSISNN_ConvLayer *L, const char *name)
{
    char buf[160];
    const uint32_t avg = L->cycles.calls ? (uint32_t)(L->cycles.total / L->cycles.calls) : 0;
    snprintf(buf, sizeof(buf),
             "%s arm_convolve_s16 cycles: calls=%lu last=%lu min=%lu max=%lu avg=%lu total=%lu\r\n",
             name,
             (unsigned long)L->cycles.calls, (unsigned long)L->cycles.last,
             (unsigned long)L->cycles.min,   (unsigned long)L->cycles.max,
             (unsigned long)avg,             (unsigned long)L->cycles.total);
    usart1_print(buf);
}

/*
 * ---------------------------------------------------------------------------
 * Usage with the generator output (lif_neuron_gen.c)
 * ---------------------------------------------------------------------------
 *
 * 1. After the weights{N} declarations, declare the layer storage, e.g. for L1:
 *        CMSISNN_CONV_DECLARE(cnn1, L1_OUT_CH, L1_IN_CH, L1_KERNEL_H, L1_KERNEL_W,
 *                             <in_h>, <in_w>, L1_OUT_H, L1_OUT_W);
 *
 * 2. In SNN_Init(), after Load_NIR_Weights():
 *        CMSISNN_Conv_Init(&cnn1, weights1,
 *                          cnn1_w, cnn1_mult, cnn1_shift, cnn1_bias,
 *                          cnn1_in, cnn1_out,
 *                          L1_IN_CH, <in_h>, <in_w>,
 *                          L1_OUT_CH, L1_OUT_H, L1_OUT_W,
 *                          L1_KERNEL_H, L1_KERNEL_W,
 *                          L1_STRIDE_H, L1_STRIDE_W, L1_PAD_H, L1_PAD_W, 1, 1);
 *
 * 3. In SNN_Run_Timestep(), replace the CSR call for that layer with:
 *        LIFNeuron_Conv2d_Update_Subtract_CMSISNN(&cnn1, layer1, input_spikes, l1_spikes);
 *
 * 4. After a run:
 *        CMSISNN_Cycles_Print(&cnn1, "L1");
 *
 * The CSR tables (row_ptr/out_idx/weight_idx) are not needed for layers that
 * use this kernel, so build the generator with --optimized false or skip
 * emitting them to keep flash numbers fair.
 */