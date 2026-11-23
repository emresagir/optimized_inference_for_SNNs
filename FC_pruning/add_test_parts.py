#!/usr/bin/env python3
"""FC pruning test: add the test parts to a generated lif_neuron_gen.c / .h.

The generator does not write any measurement code. This script adds the same parts to every
generated network, like the parts added by hand in the deployed projects:
  1. the cycle counter (snn_cycles.h) around the input accumulation of the FC layer update
  2. cycle counter on in SNN_Init
  3. CSR only: the CSR table (row_ptr, col) copied to RAM in SNN_Init, like the weights
  4. FCBench_PassEnd(): one result line after every pass (main.c calls it when FC_BENCH is 1)

Usage:  python3 add_test_parts.py <folder> <network.nir> <vec|csr> <pruning ratio>
"""
import sys
from pathlib import Path

import numpy as np
import nir

folder, nir_file, method, ratio = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]

weight = nir.read(nir_file).nodes["0"].weight
nnz = int(np.count_nonzero(weight))
total = weight.size

c_path = folder / "lif_neuron_gen.c"
h_path = folder / "lif_neuron_gen.h"
c = c_path.read_text()
h = h_path.read_text()


def insert_before(text, marker, new_text):
    """Put new_text in front of marker (marker must be in text exactly once)."""
    assert text.count(marker) == 1, f"marker found {text.count(marker)} times: {marker!r}"
    return text.replace(marker, new_text + marker)


def insert_after(text, marker, new_text):
    """Put new_text after marker (marker must be in text exactly once)."""
    assert text.count(marker) == 1, f"marker found {text.count(marker)} times: {marker!r}"
    return text.replace(marker, marker + new_text)


# ---- includes
c = insert_after(c, "#include <stdlib.h>\n",
                 '#include <string.h>\n'
                 '#include "../Inc/snn_cycles.h"     // test: DWT cycle counter\n')

# ---- cycle statistic, RAM copy of the CSR table, result line (before the utility functions)
test_block = (
    "// ---- Test additions (FC pruning test) ----\n"
    "// Cycles of the FC input accumulation (measured in the FC layer update)\n"
    "static CycStat fc_cyc_acc = {0, 0xFFFFFFFFu, 0, 0, 0};\n"
    "\n")
if method == "csr":
    test_block += (
        "// CSR table in RAM (copied in SNN_Init), like the weights\n"
        "static uint16_t fc1_row_ptr_ram[sizeof(fc1_row_ptr) / sizeof(fc1_row_ptr[0])];\n"
        "static uint8_t  fc1_col_ram[sizeof(fc1_col) / sizeof(fc1_col[0])];\n"
        "\n")
test_block += (
    "// Called by main after every pass: one result line\n"
    "void FCBench_PassEnd(uint32_t correct, uint32_t samples) {\n"
    "    uint32_t calls = fc_cyc_acc.calls;\n"
    "    if (calls == 0) calls = 1;\n"
    "    char buf[160];\n"
    f'    snprintf(buf, sizeof(buf), "FCB method={method.upper()} prune={ratio}%% nnz={nnz}/{total} '
    'calls=%lu cyc_acc=%lu acc=%lu/%lu\\r\\n",\n'
    "             (unsigned long)fc_cyc_acc.calls,\n"
    "             (unsigned long)(fc_cyc_acc.total / calls),   // average accumulation cycles per call\n"
    "             (unsigned long)correct, (unsigned long)samples);\n"
    "    usart1_print(buf);\n"
    "}\n"
    "\n")
c = insert_before(c, "// Utility functions for USART printing", test_block)

# ---- cycle counter around the input accumulation, in the FC function that is used
if method == "csr":
    start = c.index("void LIFNeuron_Layer_Update_Subtract_NoRecurrent_CSR(")
else:
    start = c.index("void LIFNeuron_Layer_Update_Subtract_NoRecurrent(")
end = c.index("\n}\n", start) + 3
function = c[start:end]
function = insert_before(function, "    // Calculate weighted input currents (no recurrent)",
                         "    CYC_START(t_acc);   // test: time the input accumulation only\n")
function = insert_before(function, "    // Vectorized membrane potential update",
                         "    CYC_STOP(fc_cyc_acc, t_acc);\n\n")
c = c[:start] + function + c[end:]

# ---- SNN_Init: cycle counter on (CSR: table to RAM)
init_part = ("\n"
             "    // Test: cycle counter on, statistic to zero\n"
             "    cyc_enable();\n"
             "    cyc_reset(&fc_cyc_acc);\n")
if method == "csr":
    init_part += ("\n"
                  "    // Test: CSR table to RAM\n"
                  "    memcpy(fc1_row_ptr_ram, fc1_row_ptr, sizeof(fc1_row_ptr));\n"
                  "    memcpy(fc1_col_ram, fc1_col, sizeof(fc1_col));\n")
c = insert_after(c, "    Load_NIR_Weights();\n", init_part)

# ---- CSR: the layer uses the RAM copy of the table
if method == "csr":
    c = c.replace("fc1_row_ptr, fc1_col, fc1_val,", "fc1_row_ptr_ram, fc1_col_ram, fc1_val,")

# ---- header: main.c prints the result line when FC_BENCH is 1
h = insert_before(h, "#endif // LIF_NEURON_GEN_H",
                  "// ---- Test additions (FC pruning test) ----\n"
                  "#define FC_BENCH 1   // main.c calls FCBench_PassEnd() after every pass\n"
                  "void FCBench_PassEnd(uint32_t correct, uint32_t samples);\n"
                  "\n")

c_path.write_text(c)
h_path.write_text(h)
print(f"  test parts added: {folder}  ({method}, {ratio}%, {nnz}/{total} nonzero)")
