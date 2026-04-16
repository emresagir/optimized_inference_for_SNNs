import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0'  # Double check this in Device Manager
BAUD_RATE = 115200    # Start slow for testing

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error: {e}")
    exit()

print("Sending test bytes. Watch your board's LED!")

try:
    while True:
        # Send a single 'A' (0x41)
        ser.write(b'A')
        print("Sent: A", end='\r')
        
        # Wait for a response if you added a transmit back in C
        response = ser.read(1)
        if response:
            print(f"Received from MCU: {response.decode('utf-8', errors='ignore')}")
            
        time.sleep(0.5)  # Half-second delay to prevent overrunning the MCU
except KeyboardInterrupt:
    print("\nTest stopped.")
    ser.close()