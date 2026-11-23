# Tap table vs CSR table: disassembly

Material for the thesis section "Why the Tap Table Is Faster than the CSR Table"
(`tap_vs_csr_section.tex`).

## Files
| File | What |
|---|---|
| `elf/csr_opt_SCNN_debug.elf` | opt_SCNN build (CSR for both layers, `USE_CMSIS_NN 0`), copied 2026-11-14 |
| `elf/tap_debug.elf` | tap project build, copied 2026-11-14 |
| `objdump/csr_kernel[_src].txt` | `LIFNeuron_Conv2d_Update_Subtract_CSR` (stand-alone function); `_src` = with C source |
| `objdump/tap_SNN_Run_Timestep[_src].txt` | `SNN_Run_Timestep`; the tap kernel is inlined here twice (Conv 1, then Conv 2) |
| `objdump/inner_loop_{csr,tap_conv1,tap_conv2}.txt` | the steady-state inner loops only |
| `count_inner_loops.py` → `results_inner_loops.txt` | instructions / loads / stores per MAC in those loops |
| `fit_cycles_per_mac.py` → `results_cycles_per_mac.txt` | fit `cycles = a*T + b*MACs` on the 694-sample logs |
| `make_objdump.sh` | copies the ELFs again and regenerates everything above |
| `tap_vs_csr_section.tex` | the LaTeX text with listings and the table |

## Results
```
loop                         MACs  insn insn/MAC loads/MAC stores/MAC
CSR (Conv 1 and Conv 2)         1    14    14.00      6.00       1.00
Tap, Conv 1 (out_ch = 8)        8    76     9.50      4.12       1.00
Tap, Conv 2 (out_ch = 16)      16   132     8.25      3.00       1.06

log    layer    n  a [cyc/step]  b [cyc/MAC]     R^2
csr    L1     693          9267        12.98  0.9999
csr    L2     693          5075        13.16  1.0000
tap    L1     693          9516         9.24  0.9999
tap    L2     693          5292         9.49  0.9999
```

## Notes
- Both builds are the Debug preset (GCC 14.3.1, `-O3`). The kernel sources (`lif_neuron_gen.c`) are
  unchanged since the 694-sample runs; only `main.c` changed (the `SNN_WINDOW` switch).
- Because of that, the CSR function is now at 0x0800be6c; the `csr_694.txt` log printed 0x0800be64.
  The instructions are the same, but addresses in the listings are from these ELFs, not from the
  measured ones.
- The tap ELF was built with `SNN_WINDOW 100` (main.c only, does not touch the kernel).
- The CSR log has no `snSP` lines, so the fit uses the MAC counts of the tap log for both kernels.
  This is valid because CSR and tap give bit-identical spikes (0/694 spike vectors differ).
- The address ranges in `count_inner_loops.py` are fixed. After a rebuild, check them against
  the new objdump before trusting the counts.
- Older presentation snippets (addresses 0x800be3c, 0x800bf00, `add r2, r0`) came from an earlier
  build where the tap kernel was `noinline` / `.constprop.0`. In the current build the per-channel
  offsets are immediates (`#392`, `#784`, ...) and the weight-block addresses are constants.
