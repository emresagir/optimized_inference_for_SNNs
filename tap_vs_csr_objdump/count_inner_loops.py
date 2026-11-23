#!/usr/bin/env python3
"""Count instructions, loads and stores in the steady-state inner loops of the CSR and tap kernels.

Address ranges are taken from objdump/*.txt (see make_objdump.sh). One CSR iteration = 1 MAC;
one tap iteration = out_ch MACs (8 for Conv 1, 16 for Conv 2), because the oc loop is fully unrolled.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSN = re.compile(r"^\s*([0-9a-f]{7,8}):\s+(?:[0-9a-f]{4}\s?){1,2}\s+(\S+)\s*(.*)$")

LOOPS = [  # name, file, first, last address (inclusive), MACs per iteration
    ("CSR (Conv 1 and Conv 2)", "objdump/csr_kernel.txt", 0x800bf0a, 0x800bf3a, 1),
    ("Tap, Conv 1 (out_ch = 8)", "objdump/tap_SNN_Run_Timestep.txt", 0x800ccf6, 0x800cdfe, 8),
    ("Tap, Conv 2 (out_ch = 16)", "objdump/tap_SNN_Run_Timestep.txt", 0x800cf7e, 0x800d146, 16),
]

def loop_insns(path, lo, hi):
    out = []
    for line in (HERE / path).read_text().splitlines():
        m = INSN.match(line)
        if m and lo <= int(m.group(1), 16) <= hi:
            out.append((int(m.group(1), 16), m.group(2), m.group(3)))
    return out

print(f"{'loop':28s} {'MACs':>4s} {'insn':>5s} {'insn/MAC':>8s} {'loads/MAC':>9s} {'stores/MAC':>10s}  load mnemonics")
for name, path, lo, hi, macs in LOOPS:
    ins = loop_insns(path, lo, hi)
    # save the loop on its own, as in the objdump file (one file per loop)
    tag = name.split(",")[0].split(" ")[0].lower() + ("" if "CSR" in name else "_conv" + name.split("Conv ")[1][0])
    src = [l for l in (HERE / path).read_text().splitlines()
           if (m := INSN.match(l)) and lo <= int(m.group(1), 16) <= hi]
    (HERE / "objdump" / f"inner_loop_{tag}.txt").write_text(f"# {name}: {macs} MAC(s) per iteration\n" + "\n".join(src) + "\n")
    loads = [i for i in ins if i[1].startswith("ldr")]
    stores = [i for i in ins if i[1].startswith("str")]
    kinds = {}
    for _, mn, ops in loads:
        kinds[mn] = kinds.get(mn, 0) + 1
    print(f"{name:28s} {macs:>4d} {len(ins):>5d} {len(ins)/macs:>8.2f} {len(loads)/macs:>9.2f} {len(stores)/macs:>10.2f}  {kinds}")
