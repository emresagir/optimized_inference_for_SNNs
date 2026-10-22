#ifndef SNN_CYCLES_H
#define SNN_CYCLES_H

/*
 * Header-only cycle timing for the generated SNN code (STM32H7, DWT->CYCCNT).
 *
 * Include this ONLY from lif_neuron_gen.c (it defines static variables).
 * Time the same boundary in every build (naive / CSR / CMSIS-NN): the full
 * layer-update call, i.e. accumulation + LIF update. Nothing else is inside
 * the timed region.
 *
 * Usage:
 *   CYC_START(t_l1);
 *   <layer 1 update call>
 *   CYC_STOP(cyc_l1, t_l1);
 *   CYC_LOG(cyc_l1, "L1", cyc_count_nz(input_spikes, NUM_INPUTS));
 */

#include <stdint.h>
#include <stdio.h>
#include "stm32h7xx_hal.h"
#include "arm_math.h"
#include "lif_neuron_gen.h"     /* usart1_print */

/* 1 = also print one CSV line per timestep: CYC,<layer>,<step>,<nonzero_inputs>,<cycles>
 * (printed OUTSIDE the timed region, but it slows the run down a lot). */
#ifndef CYC_LOG_PER_STEP
#define CYC_LOG_PER_STEP 0
#endif

typedef struct {
    uint32_t last, min, max, calls;
    uint64_t total;
} CycStat;

/* One stat per conv layer. */
static CycStat cyc_l1 = {0, 0xFFFFFFFFu, 0, 0, 0};
static CycStat cyc_l2 = {0, 0xFFFFFFFFu, 0, 0, 0};

static void cyc_enable(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#if defined(__CORTEX_M) && (__CORTEX_M == 7U)
    DWT->LAR = 0xC5ACCE55;          /* unlock DWT on Cortex-M7 */
#endif
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void cyc_reset(CycStat *s)
{
    s->last = 0;
    s->min = 0xFFFFFFFFu;
    s->max = 0;
    s->calls = 0;
    s->total = 0;
}

static void cyc_record(CycStat *s, uint32_t c)
{
    s->last = c;
    if (c < s->min) s->min = c;
    if (c > s->max) s->max = c;
    s->total += c;
    s->calls++;
}

static uint32_t cyc_count_nz(const q7_t *a, uint32_t n)
{
    uint32_t k = 0;
    for (uint32_t i = 0; i < n; i++) k += (a[i] != 0);
    return k;
}

static void cyc_print(const CycStat *s, const char *name)
{
    char buf[160];
    const uint32_t avg = s->calls ? (uint32_t)(s->total / s->calls) : 0;
    snprintf(buf, sizeof(buf),
             "%s layer update cycles: calls=%lu last=%lu min=%lu max=%lu avg=%lu total=%lu\r\n",
             name,
             (unsigned long)s->calls, (unsigned long)s->last,
             (unsigned long)s->min,   (unsigned long)s->max,
             (unsigned long)avg,      (unsigned long)s->total);
    usart1_print(buf);
}

/* Barriers keep the counter reads from being reordered around the call. */
#define CYC_START(v)    uint32_t v; __DSB(); __ISB(); v = DWT->CYCCNT
#define CYC_STOP(s, v)  do { uint32_t _e = DWT->CYCCNT; __DSB(); __ISB(); \
                             cyc_record(&(s), _e - (v)); } while (0)

/* Optional per-timestep CSV line. The nz argument is only evaluated when
 * CYC_LOG_PER_STEP is 1. Call it AFTER CYC_STOP. */
#if CYC_LOG_PER_STEP
#define CYC_LOG(s, name, nz_expr) do {                                          \
        char _b[64];                                                            \
        snprintf(_b, sizeof(_b), "CYC,%s,%lu,%lu,%lu\r\n", (name),              \
                 (unsigned long)((s).calls - 1), (unsigned long)(nz_expr),      \
                 (unsigned long)(s).last);                                      \
        usart1_print(_b);                                                       \
    } while (0)
#else
#define CYC_LOG(s, name, nz_expr) do { } while (0)
#endif

#endif /* SNN_CYCLES_H */