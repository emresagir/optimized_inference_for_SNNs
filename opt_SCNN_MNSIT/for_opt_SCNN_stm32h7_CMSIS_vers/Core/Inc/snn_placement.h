#ifndef SNN_PLACEMENT_H
#define SNN_PLACEMENT_H

/*
 * Header-only runtime memory-placement report (STM32H7 default memory map).
 * Include ONLY from lif_neuron_gen.c (it uses that file's static arrays).
 *
 *   PLC(array)                      address, size and region of a real array
 *   PLC_PTR("name", ptr, bytes)     same for a pointer (e.g. input_spikes)
 *   PLC_FUNC(function)              address and region of a function (code)
 *
 * Adjust plc_region() if your part or linker script uses other regions.
 */

#include <stdint.h>
#include <stdio.h>
#include "lif_neuron_gen.h"     /* usart1_print */

static const char *plc_region(const void *p)
{
    const uint32_t a = (uint32_t)p;
    if (a <  0x00010000u)                        return "ITCM";
    if (a >= 0x08000000u && a < 0x08100000u)     return "FLASH";
    if (a >= 0x20000000u && a < 0x20020000u)     return "DTCM";
    if (a >= 0x24000000u && a < 0x24080000u)     return "AXI_SRAM";
    if (a >= 0x30000000u && a < 0x30048000u)     return "SRAM_D2";
    if (a >= 0x38000000u && a < 0x38010000u)     return "SRAM_D3";
    if (a >= 0x90000000u && a < 0xA0000000u)     return "QSPI";
    if (a >= 0x60000000u && a < 0x90000000u)     return "EXT_FMC";
    if (a >= 0xC0000000u)                        return "EXT_SDRAM";
    return "OTHER";
}

static void plc_print(const char *name, const void *p, uint32_t bytes)
{
    char b[96];
    snprintf(b, sizeof(b), "PLC %-18s addr=0x%08lX size=%6lu  %s\r\n",
             name, (unsigned long)(uint32_t)p, (unsigned long)bytes, plc_region(p));
    usart1_print(b);
}

#define PLC(sym)               plc_print(#sym, (const void *)(sym), (uint32_t)sizeof(sym))
#define PLC_PTR(name, p, n)    plc_print((name), (const void *)(p), (uint32_t)(n))
#define PLC_FUNC(fn)           plc_print(#fn, (const void *)(fn), 0u)

#endif /* SNN_PLACEMENT_H */