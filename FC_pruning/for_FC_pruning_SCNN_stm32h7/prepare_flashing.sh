#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Run from the build output folder, wherever the script is called from
cd "$(dirname "$0")/build/Debug"

echo "=== Step 1: Extracting .ext_weights section to weights_and_tests.bin ==="
arm-none-eabi-objcopy -O binary --only-section=.ext_weights for_FC_pruning_SCNN_stm32h7.elf weights_and_tests.bin

echo "=== Step 2: Creating debug ELF by removing .ext_weights ==="
arm-none-eabi-objcopy --remove-section=.ext_weights for_FC_pruning_SCNN_stm32h7.elf for_FC_pruning_SCNN_stm32h7_debug.elf

echo "=== Step 3: Flashing weights to QSPI Flash using STM32_Programmer_CLI ==="
~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI \
    -c port=SWD \
    -el ~/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MT25TL01G_STM32H750B-DISCO.stldr \
    -d weights_and_tests.bin 0x90000000 \
    -v

echo "=== Success! QSPI data flashed and debug ELF prepared. ==="
