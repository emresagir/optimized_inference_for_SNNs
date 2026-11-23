#!/bin/bash
# FC pruning test: deploy every generated variant and log one pass of the test.
#
# For every variant in generated/ (vec_p0 ... csr_p100):
#   1. copy lif_neuron_gen.c / .h into the project, mark the FC layer functions noinline
#      (so the compiler gives them their own registers, the same way in every build)
#   2. build, make the _debug.elf (without .ext_weights), flash it
#      (the QSPI test data is flashed only when it changed, it is the same for all variants)
#   3. read /dev/ttyACM0, reset the board, wait for the end of the first pass ("Accuracy:")
#   4. save the pass in logs_isolated_noinline/<method>_p<ratio>.txt and the FCB line in summary.txt
#
# Usage:  ./run_all_tests.sh              all 16 variants
#         ./run_all_tests.sh csr_p20 ...  only these variants
# Close tio (or anything else using the port) before starting.
# The project's lif_neuron_gen.c / .h are saved before and put back at the end.

set -e
cd "$(dirname "$0")"
HERE="$(pwd)"

PROJECT="$HERE/for_FC_pruning_SCNN_stm32h7"
BUILD="$PROJECT/build/Debug"
ELF="$BUILD/for_FC_pruning_SCNN_stm32h7.elf"
DEBUG_ELF="$BUILD/for_FC_pruning_SCNN_stm32h7_debug.elf"
QSPI_BIN="$BUILD/weights_and_tests.bin"
PORT=/dev/ttyACM0
TIMEOUT=300                      # seconds to wait for one pass
LOGS="$HERE/logs_isolated_noinline"
# Test data header for main.c (default test.h = ST-MNIST), e.g.  TEST_H=test_dense.h ./run_all_tests.sh
TEST_H="${TEST_H:-test.h}"
if [ "$TEST_H" != "test.h" ]; then LOGS="${LOGS}_${TEST_H%.h}"; fi

CUBEPROG="$HOME/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin"
CLI="$CUBEPROG/STM32_Programmer_CLI"
LOADER="$CUBEPROG/ExternalLoader/MT25TL01G_STM32H750B-DISCO.stldr"

# Toolchain, cmake and ninja from the STM32 VS Code extension
BUNDLES="$HOME/.local/share/stm32cube/bundles"
export PATH="$(ls -d $BUNDLES/gnu-tools-for-stm32/*/bin | head -1):$(ls -d $BUNDLES/cmake/*/bin | head -1):$(ls -d $BUNDLES/ninja/*/bin | head -1):$PATH"

# ---------------------------------------------------------------- variants to run
if [ $# -gt 0 ]; then
    VARIANTS="$*"
else
    VARIANTS=""
    for method in vec csr; do
        for ratio in 0 20 40 60 80 90 95 100; do
            VARIANTS="$VARIANTS ${method}_p${ratio}"
        done
    done
fi

# ---------------------------------------------------------------- checks
if fuser -s "$PORT" 2>/dev/null; then
    echo "ERROR: $PORT is in use (tio open?). Close it and start again."
    exit 1
fi
mkdir -p "$LOGS"

# ---------------------------------------------------------------- save the project files, put them back at the end
# (a file that did not exist before the tests is removed again at the end)
BACKUP="$HERE/backup_before_tests"
rm -rf "$BACKUP"
mkdir -p "$BACKUP"
PROJECT_FILES="$PROJECT/Core/Src/lif_neuron_gen.c $PROJECT/Core/Inc/lif_neuron_gen.h $PROJECT/Core/Src/main.c"
for f in $PROJECT_FILES; do
    [ -f "$f" ] && cp "$f" "$BACKUP/"
done
restore() {
    for f in $PROJECT_FILES; do
        if [ -f "$BACKUP/$(basename "$f")" ]; then
            cp "$BACKUP/$(basename "$f")" "$f"
        else
            rm -f "$f"
        fi
    done
    echo "Project files put back as they were before the tests"
}
trap restore EXIT

# main.c: use the chosen test data header
if [ ! -f "$PROJECT/Core/Inc/$TEST_H" ]; then
    echo "ERROR: $PROJECT/Core/Inc/$TEST_H not found"
    exit 1
fi
sed -i "s/^#include \"test[^\"]*\.h\"/#include \"$TEST_H\"/" "$PROJECT/Core/Src/main.c"
echo "Test data: $(grep '^#include \"test' "$PROJECT/Core/Src/main.c")   logs: $LOGS"

# ---------------------------------------------------------------- one variant
run_variant() {
    local name="$1"
    local src="$HERE/generated/$name"
    local log="$LOGS/$name.txt"
    local raw="$LOGS/.$name.raw"

    echo "================ $name"
    if [ ! -f "$src/lif_neuron_gen.c" ]; then
        echo "ERROR: $src/lif_neuron_gen.c not found (run generate_all.sh first)"
        return 1
    fi

    # 1. copy the generated files
    cp "$src/lif_neuron_gen.c" "$PROJECT/Core/Src/"
    cp "$src/lif_neuron_gen.h" "$PROJECT/Core/Inc/"

    # FC layer functions noinline (only in the copy in the project)
    sed -i 's/^void LIFNeuron_Layer_Update_Subtract_NoRecurrent\(_CSR\)\?(/__attribute__((noinline)) &/' \
        "$PROJECT/Core/Src/lif_neuron_gen.c"
    echo "  noinline added to $(grep -c '^__attribute__((noinline)) void LIFNeuron_Layer_Update' "$PROJECT/Core/Src/lif_neuron_gen.c") FC function(s)"

    # 2. build
    echo "  building"
    if ! cmake --build "$BUILD" > "$LOGS/.build.txt" 2>&1; then
        grep -E "error" "$LOGS/.build.txt" | head -20
        echo "ERROR: build failed, see $LOGS/.build.txt"
        return 1
    fi

    # QSPI image and debug elf (what prepare_flashing.sh does)
    arm-none-eabi-objcopy -O binary --only-section=.ext_weights "$ELF" "$QSPI_BIN"
    arm-none-eabi-objcopy --remove-section=.ext_weights "$ELF" "$DEBUG_ELF"

    # QSPI only when the test data changed
    local md5_new md5_old=""
    md5_new=$(md5sum "$QSPI_BIN" | cut -d' ' -f1)
    [ -f "$LOGS/.qspi_md5" ] && md5_old=$(cat "$LOGS/.qspi_md5")
    if [ "$md5_new" != "$md5_old" ]; then
        echo "  flashing QSPI (test data)"
        "$CLI" -c port=SWD -el "$LOADER" -d "$QSPI_BIN" 0x90000000 -v > "$LOGS/.flash_qspi.txt" 2>&1 \
            || { echo "ERROR: QSPI flashing failed, see $LOGS/.flash_qspi.txt"; return 1; }
        echo "$md5_new" > "$LOGS/.qspi_md5"
    fi

    # flash the program
    echo "  flashing"
    "$CLI" -c port=SWD -w "$DEBUG_ELF" -v > "$LOGS/.flash.txt" 2>&1 \
        || { echo "ERROR: flashing failed, see $LOGS/.flash.txt"; return 1; }

    # 3. read the serial port, then reset so the log starts at boot
    stty -F "$PORT" 115200 cs8 -cstopb -parenb raw -echo
    cat "$PORT" > "$raw" &
    local reader=$!
    sleep 0.5
    "$CLI" -c port=SWD -rst > "$LOGS/.reset.txt" 2>&1 \
        || { kill $reader; echo "ERROR: reset failed, see $LOGS/.reset.txt"; return 1; }

    # 4. wait for the end of the first pass after the banner
    echo -n "  running"
    local waited=0
    while true; do
        if tr -d '\r' < "$raw" | awk '/Inference Test/ {b=1} b && /^Accuracy:/ {found=1} END {exit !found}'; then
            break
        fi
        if [ $waited -ge $TIMEOUT ]; then
            kill $reader
            echo ""
            echo "ERROR: no complete pass after ${TIMEOUT} s, raw output in $raw"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    sleep 0.5                    # let the last line arrive
    kill $reader
    wait $reader 2>/dev/null || true
    echo " done (${waited} s)"

    # keep only the pass: from the banner to the line after "Accuracy:"
    tr -d '\r' < "$raw" | awk '
        /Inference Test/ && !start {start = 1}
        start {print}
        start && /^Accuracy:/ {getline; print; exit}' > "$log"
    rm -f "$raw"

    local fcb
    fcb=$(grep "^FCB" "$log" || echo "FCB line missing")
    echo "  $fcb"
    echo "$fcb" >> "$LOGS/summary.txt"
}

# ---------------------------------------------------------------- all variants
echo "# $(date '+%Y-%m-%d %H:%M')  run_all_tests.sh $*" >> "$LOGS/summary.txt"
echo "# test data: $TEST_H" >> "$LOGS/summary.txt"
for v in $VARIANTS; do
    run_variant "$v"
done
echo ""
echo "Logs in $LOGS, FCB lines in $LOGS/summary.txt"
