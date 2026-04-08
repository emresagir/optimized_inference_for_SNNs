cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME for-soft-reset)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Added compiler and link options for DSP
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    -mcpu=cortex-m7
    -mfpu=fpv5-sp-d16
    -mfloat-abi=hard
    -O3
)
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
    -mcpu=cortex-m7
    -mfpu=fpv5-sp-d16
    -mfloat-abi=hard
)

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Add CMSISCORE path, this is needed for the compiling process, bcs DSP depends on the core library.
set(CMSISCORE ${CMAKE_SOURCE_DIR}/Drivers/CMSIS)
# Add DSP Lib. as a subdirectory, the CMakeList.txt files inside will handle this call.
add_subdirectory(Drivers/CMSIS-DSP) 

# Add NN
add_subdirectory(Drivers/CMSIS-NN)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    Core/Src/lif_neuron_gen.c
    # Add user sources here
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
    Core/Inc
    # Adding the Includes for DSP to use while compiling.
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS-DSP/Include
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS/Core/Include
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS/Device/ST/STM32H7xx/Include
    # For NN
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS-NN/Include
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
    # For DSP, it needs to know which architecture its being compiled for, 
    # This way it will use the most optimized version for this particular cortex M7 arch.
    ARM_MATH_CM7
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx
    # Adding DSP as library to be linked.
    # CMSIS-DSP$ grep -nr "add_library"
    # Source/CMakeLists.txt:31:add_library(CMSISDSP STATIC)
    CMSISDSP 

    # Same for NN, to found what is the name of the library, inside the CMSIS-NN
    # grep -nr "add_library"
    # This shows the linkable objects name.
    # CMakeLists.txt:24:add_library(cmsis-nn STATIC)
    cmsis-nn

    # Add user defined libraries
)
