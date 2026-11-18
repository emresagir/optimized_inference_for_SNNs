#!/usr/bin/env bash
# Copy the current builds and disassemble the CSR and tap kernels.
# Run from this folder after building both projects (Debug preset, -O3).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=..
OD=${OD:-$HOME/.local/share/stm32cube/bundles/gnu-tools-for-stm32/14.3.1+st.2/bin/arm-none-eabi-objdump}
mkdir -p elf objdump

cp "$ROOT/opt_SCNN_MNSIT/for_opt_SCNN_stm32h7_CMSIS_vers/build/Debug/for_opt_SCNN_stm32h7_CMSIS_vers_debug.elf" elf/csr_opt_SCNN_debug.elf
cp "$ROOT/tap_table_csr_improvement/for_tap_table_SCNN_stm32h7/build/Debug/for_tap_table_SCNN_stm32h7_debug.elf" elf/tap_debug.elf

# CSR: stand-alone function (called for Conv 1 and Conv 2 with pointer arguments)
"$OD" -d    --disassemble=LIFNeuron_Conv2d_Update_Subtract_CSR elf/csr_opt_SCNN_debug.elf > objdump/csr_kernel.txt
"$OD" -d -S --disassemble=LIFNeuron_Conv2d_Update_Subtract_CSR elf/csr_opt_SCNN_debug.elf > objdump/csr_kernel_src.txt
# Tap: inlined twice into SNN_Run_Timestep (Conv 1, then Conv 2), no symbol of its own
"$OD" -d    --disassemble=SNN_Run_Timestep elf/tap_debug.elf > objdump/tap_SNN_Run_Timestep.txt
"$OD" -d -S --disassemble=SNN_Run_Timestep elf/tap_debug.elf > objdump/tap_SNN_Run_Timestep_src.txt

python3 count_inner_loops.py  | tee results_inner_loops.txt
python3 fit_cycles_per_mac.py | tee results_cycles_per_mac.txt
