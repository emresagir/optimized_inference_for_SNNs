#!/bin/bash
# FC pruning test: networks -> C files (generator) -> test parts.
#   generated/vec_p<ratio>/  current FC (generator without --fc_csr)
#   generated/csr_p<ratio>/  FC with CSR table (generator with --fc_csr true)
# Deploy: lif_neuron_gen.c -> for_FC_pruning_SCNN_stm32h7/Core/Src/
#         lif_neuron_gen.h -> for_FC_pruning_SCNN_stm32h7/Core/Inc/
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"

python3 make_pruned_networks.py

for ratio in 0 20 40 60 80 90 95 100; do
    nir_file="$HERE/networks/random_fc_144x10_p${ratio}.nir"
    for method in vec csr; do
        out="$HERE/generated/${method}_p${ratio}"
        mkdir -p "$out"
        if [ "$method" = "csr" ]; then flag="--fc_csr true"; else flag=""; fi
        (cd "$out" && python3 "$HERE/nir_to_c_generator_V2.py" --nir_file "$nir_file" $flag > generator_log.txt)
        python3 add_test_parts.py "$out" "$nir_file" "$method" "$ratio"
    done
done
