/**
 * example_usage.c
 * ───────────────
 * ST-MNIST SNN inference test-harness for STM32H7.
 *
 * What it does
 * ────────────
 *  1. Iterates over all TD_NUM_SAMPLES in test_data.h.
 *  2. For each sample, resets neuron state then feeds every timestep
 *     through SNN_Run_Timestep().
 *  3. Accumulates output spikes per class over the full temporal window.
 *  4. Picks the class with the highest total spike count as the prediction.
 *  5. Streams a compact result string over USART1 for each sample, then
 *     prints a final accuracy summary.
 *
 * Build notes
 * ───────────
 *  • Include this file in your STM32CubeIDE project alongside snn.c.
 *  • Make sure test_data.h and snn.h (with SNN_Init / SNN_Reset_State /
 *    SNN_Run_Timestep declarations) are on the include path.
 *  • Call SNN_Run_Tests() once from main() after HAL_Init() and
 *    peripheral initialisation.
 */

#include "stm32h7xx_hal.h"
#include "usart.h"          /* huart1, usart1_print()                    */
#include "snn.h"            /* SNN_Init, SNN_Reset_State, SNN_Run_Timestep */
#include "test_data.h"      /* test_samples[], TD_NUM_SAMPLES, TD_NUM_CLASSES */
#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* ── Forward declarations of SNN internals (defined in snn.c) ─────────── */
extern void SNN_Init(void);
extern void SNN_Reset_State(void);
extern void SNN_Run_Timestep(const int8_t *input_spikes, int8_t *output_spikes);

/* ── Helpers ────────────────────────────────────────────────────────────── */

/**
 * tiny_itoa – write a non-negative integer into buf and return the
 * number of characters written (no null terminator).
 */
static int tiny_itoa(uint32_t v, char *buf) {
    if (v == 0) { buf[0] = '0'; return 1; }
    char tmp[12];
    int  len = 0;
    while (v) { tmp[len++] = '0' + (v % 10); v /= 10; }
    for (int i = 0; i < len; i++) buf[i] = tmp[len - 1 - i];
    return len;
}

/** Blocking USART1 print (re-exported here so example_usage.c is self-contained). */
static void print_str(const char *s) {
    HAL_UART_Transmit(&huart1, (uint8_t *)s, strlen(s), 1000);
}

/* ── Core classification logic ──────────────────────────────────────────── */

/**
 * classify_sample
 * ───────────────
 * Feeds one test sample (all its timesteps) through the SNN and returns
 * the predicted class (0-9) together with per-class spike totals.
 *
 * @param sample          Pointer to the TestSample descriptor.
 * @param spike_totals    Output array [TD_NUM_CLASSES] – caller provides storage.
 * @return                Predicted class index (argmax of spike_totals).
 */
static uint8_t classify_sample(const TestSample *sample,
                                uint32_t         spike_totals[TD_NUM_CLASSES]) {
    static int8_t output_spikes[TD_NUM_CLASSES];

    /* Zero the accumulator */
    memset(spike_totals, 0, TD_NUM_CLASSES * sizeof(uint32_t));

    /* Reset neuron membrane potentials before the new gesture */
    SNN_Reset_State();

    /* Feed every timestep */
    for (uint16_t t = 0; t < sample->num_timesteps; t++) {
        SNN_Run_Timestep(sample->spikes[t], output_spikes);

        /* Accumulate spikes – output_spikes[c] is 0 or 1 */
        for (uint8_t c = 0; c < TD_NUM_CLASSES; c++) {
            if (output_spikes[c] > 0) {
                spike_totals[c]++;
            }
        }
    }

    /* Argmax: find the class with the most spikes */
    uint8_t  best_class = 0;
    uint32_t best_count = spike_totals[0];
    for (uint8_t c = 1; c < TD_NUM_CLASSES; c++) {
        if (spike_totals[c] > best_count) {
            best_count = spike_totals[c];
            best_class = c;
        }
    }
    return best_class;
}

/* ── Public entry point ─────────────────────────────────────────────────── */

/**
 * SNN_Run_Tests
 * ─────────────
 * Call once from main() after all peripherals are initialised.
 *
 * Example USART1 output (115200 baud):
 *
 *   ╔══════════════════════════════════╗
 *   ║  SNN ST-MNIST Inference Test     ║
 *   ║  Samples: 50   Classes: 10       ║
 *   ╚══════════════════════════════════╝
 *   [00] T=62  GT=8  Pred=8  PASS  spikes: 0 0 0 0 0 0 0 0 14 0
 *   [01] T=47  GT=3  Pred=3  PASS  spikes: 0 0 0 7 0 0 0 0 0  0
 *   ...
 *   ─────────────────────────────────
 *   Accuracy: 47/50  (94.00%)
 *   ─────────────────────────────────
 */
void SNN_Run_Tests(void) {
    /* ── Banner ── */
    print_str("\r\n");
    print_str("╔══════════════════════════════════════╗\r\n");
    print_str("║  SNN ST-MNIST Inference Test          ║\r\n");
    print_str("╚══════════════════════════════════════╝\r\n");

    {
        char buf[64];
        int  n;
        n  = 0;
        n += snprintf(buf + n, sizeof(buf) - n, "  Samples : %u\r\n", (unsigned)TD_NUM_SAMPLES);
        print_str(buf);
        n  = 0;
        n += snprintf(buf + n, sizeof(buf) - n, "  Classes : %u\r\n", (unsigned)TD_NUM_CLASSES);
        print_str(buf);
    }
    print_str("\r\n");

    /* Initialise SNN weights + neuron parameters once */
    SNN_Init();

    uint32_t correct = 0;
    uint32_t spike_totals[TD_NUM_CLASSES];

    for (uint16_t s = 0; s < TD_NUM_SAMPLES; s++) {
        const TestSample *smp = &test_samples[s];

        uint8_t predicted = classify_sample(smp, spike_totals);
        uint8_t gt        = smp->label;
        uint8_t pass      = (predicted == gt);
        if (pass) correct++;

        /* ── Per-sample line ─────────────────────────────────────────── */
        char line[160];
        int  pos = 0;

        /* Index */
        pos += snprintf(line + pos, sizeof(line) - pos, "[%02u] ", (unsigned)s);

        /* Timesteps */
        pos += snprintf(line + pos, sizeof(line) - pos, "T=%3u  ", (unsigned)smp->num_timesteps);

        /* Ground-truth / prediction / pass-fail */
        pos += snprintf(line + pos, sizeof(line) - pos,
                        "GT=%u  Pred=%u  %s",
                        (unsigned)gt,
                        (unsigned)predicted,
                        pass ? "PASS" : "FAIL");

        /* Spike totals per class */
        pos += snprintf(line + pos, sizeof(line) - pos, "  spikes:");
        for (uint8_t c = 0; c < TD_NUM_CLASSES; c++) {
            pos += snprintf(line + pos, sizeof(line) - pos,
                            " %lu", (unsigned long)spike_totals[c]);
        }

        pos += snprintf(line + pos, sizeof(line) - pos, "\r\n");
        print_str(line);
    }

    /* ── Summary ── */
    print_str("─────────────────────────────────────────\r\n");
    {
        char buf[80];
        /* Compute percentage with 2 decimal places without floating-point */
        uint32_t pct_int  = (correct * 100u) / TD_NUM_SAMPLES;
        uint32_t pct_frac = ((correct * 10000u) / TD_NUM_SAMPLES) % 100u;
        snprintf(buf, sizeof(buf),
                 "Accuracy: %lu/%u  (%lu.%02lu%%)\r\n",
                 (unsigned long)correct,
                 (unsigned)TD_NUM_SAMPLES,
                 (unsigned long)pct_int,
                 (unsigned long)pct_frac);
        print_str(buf);
    }
    print_str("─────────────────────────────────────────\r\n");
}
