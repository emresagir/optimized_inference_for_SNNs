import serial
import numpy as np
import time
import re
import os

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
NUM_SAMPLES = 140
TIMESTEPS = 256
FEATURES = 12
HANDSHAKE_BYTE = b'\xa5'
HEADER_FILE = "test.h"

# --- EXPECTED CLASSES ---
EXPECTED_CLASSES = np.array([
    1, 3, 2, 2, 6, 1, 1, 3, 4, 5, 4, 0, 5, 5, 0, 2, 4, 3, 1, 2,
    5, 2, 4, 6, 2, 2, 4, 1, 4, 4, 1, 3, 2, 0, 4, 5, 1, 0, 3, 5,
    1, 2, 0, 4, 5, 4, 5, 6, 6, 1, 4, 5, 0, 2, 3, 4, 5, 0, 2, 5,
    5, 5, 6, 5, 6, 4, 1, 2, 6, 1, 0, 0, 6, 4, 0, 3, 3, 0, 1, 6,
    2, 0, 3, 1, 0, 1, 2, 0, 3, 0, 0, 0, 4, 6, 1, 3, 2, 5, 2, 6,
    0, 5, 5, 0, 3, 1, 6, 6, 3, 2, 4, 4, 6, 3, 6, 2, 2, 5, 3, 6,
    2, 1, 3, 6, 5, 4, 5, 4, 1, 6, 3, 0, 3, 6, 3, 1, 6, 4, 3, 1
], dtype=np.uint8)


def parse_test_h(filepath):
    """
    Parses the C header file containing:
        const q7_t test_input[140][256][12] = { ... };
    Returns a numpy array of shape (140, 256, 12) with dtype int8.
    """
    print(f"Parsing '{filepath}'...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Strip C comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Extract everything between the outermost { } of the array initializer
    match = re.search(r'test_input\s*\[.*?\]\s*=\s*(\{.*\})\s*;', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'test_input' array in the header file.")

    array_str = match.group(1)

    # Extract all integer tokens (handles negative values like -1 too)
    numbers = list(map(int, re.findall(r'-?\d+', array_str)))

    expected_count = NUM_SAMPLES * TIMESTEPS * FEATURES
    if len(numbers) != expected_count:
        raise ValueError(
            f"Parsed {len(numbers)} values, but expected {expected_count} "
            f"({NUM_SAMPLES} x {TIMESTEPS} x {FEATURES})."
        )

    arr = np.array(numbers, dtype=np.int8).reshape(NUM_SAMPLES, TIMESTEPS, FEATURES)
    print(f"Successfully parsed array with shape {arr.shape}.")
    return arr


def load_data():
    if not os.path.exists(HEADER_FILE):
        raise FileNotFoundError(f"Header file '{HEADER_FILE}' not found.")

    test_data = parse_test_h(HEADER_FILE)

    if len(EXPECTED_CLASSES) != NUM_SAMPLES:
        raise ValueError(
            f"EXPECTED_CLASSES has {len(EXPECTED_CLASSES)} entries, "
            f"but NUM_SAMPLES is {NUM_SAMPLES}."
        )

    return test_data, EXPECTED_CLASSES


def run_simulation(test_data, expected_classes):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    num_matches = 0
    print("Starting Simulation Feed...")
    start_time = time.time()

    for i in range(NUM_SAMPLES):
        # 1. Wait for MCU "Ready" handshake
        while True:
            byte = ser.read(1)
            if byte == HANDSHAKE_BYTE:
                print("Handshake done \n")
                break

        # 2. Flatten sample from (256, 12) → 3072 bytes (q7_t = int8)
        sample = test_data[i].tobytes()  # 256 * 12 = 3072 bytes
        assert len(sample) == TIMESTEPS * FEATURES, "Unexpected sample size!"

        # 3. Send the full sample
        ser.write(sample)

        # 4. Read 1-byte prediction from MCU
        prediction = ser.read(1)
        if prediction:
            pred_val = int.from_bytes(prediction, "big")
            expected_val = int(expected_classes[i])

            match = pred_val == expected_val
            if match:
                num_matches += 1

            status = "✓ Match" if match else "✗ Miss"
            print(f"Sample {i+1:>3}/{NUM_SAMPLES}: Pred={pred_val}, Exp={expected_val}  [{status}]")
        else:
            print(f"Sample {i+1:>3}/{NUM_SAMPLES}: Timeout — no response from MCU.")

    end_time = time.time()
    accuracy = (num_matches / NUM_SAMPLES) * 100
    print("-" * 40)
    print(f"Accuracy : {accuracy:.2f}%  ({num_matches}/{NUM_SAMPLES})")
    print(f"Duration : {end_time - start_time:.2f} seconds")
    ser.close()


if __name__ == "__main__":
    test_data, expected_classes = load_data()
    run_simulation(test_data, expected_classes)