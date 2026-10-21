"""Talk to the ESP32 stepper controller (firmware/stepper_control) over USB serial.

The firmware takes single-key commands: 1/2/3 select a motor, + / - move it 18 degrees,
0 sets or returns to home, p prints positions, x disables all motors, ? shows help.

    python3 -m camsync.motors 1 + + p      # send keys, print the replies
    python3 -m camsync.motors              # interactive terminal (Ctrl+] quits)
    python3 -m camsync.motors --flash      # compile + upload the firmware

Opening the port does not reset the ESP32 (DTR and RTS are left asserted together, which
the auto-reset circuit ignores; clearing DTR first pulses EN), so its software positions
and home survive reconnecting. Positions are lost on a reset or power cycle.
"""
import argparse
import os
import subprocess
import sys
import time

import serial
from serial.tools import list_ports

BAUD = 115200
FQBN = "esp32:esp32:esp32"  # ESP32 DevKit V1 = generic ESP32 Dev Module
SKETCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "firmware", "stepper_control")
USB_IDS = {(0x10C4, 0xEA60), (0x1A86, 0x7523)}  # CP210x, CH340


def find_port():
    ports = [p.device for p in list_ports.comports() if (p.vid, p.pid) in USB_IDS]
    if not ports:
        sys.exit("No ESP32 found (no CP210x/CH340 USB serial device). Is it plugged in?")
    return ports[0]


class Motors:
    def __init__(self, port=None, baud=BAUD):
        # Leave DTR/RTS at their defaults: pyserial clears DTR before RTS, and RTS alone
        # pulls EN low, so "dtr = rts = False" resets the ESP32 on about half the opens.
        self.ser = serial.Serial(port or find_port(), baud, timeout=0.1)

    def send(self, keys, quiet=0.3):
        """Send command keys and return the reply lines, collected until the ESP32 has
        been quiet for `quiet` seconds (a move prints 'completed' when it ends)."""
        self.ser.reset_input_buffer()
        self.ser.write(keys.encode())
        out, last = b"", time.time()
        while time.time() - last < quiet:
            chunk = self.ser.read(256)
            if chunk:
                out, last = out + chunk, time.time()
        return out.decode(errors="replace").splitlines()

    def close(self):
        self.ser.close()


def flash(port):
    cli = "arduino-cli"
    subprocess.run([cli, "compile", "--fqbn", FQBN, SKETCH], check=True)
    subprocess.run([cli, "upload", "--fqbn", FQBN, "-p", port, SKETCH], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help="command keys to send, e.g. 1 + p")
    ap.add_argument("--port", help="serial port (default: the first CP210x/CH340)")
    ap.add_argument("--flash", action="store_true", help="compile and upload the firmware")
    args = ap.parse_args()
    port = args.port or find_port()
    if args.flash:
        flash(port)
    elif args.keys:
        m = Motors(port)
        for line in m.send("".join(args.keys)):
            print(line)
        m.close()
    else:
        from serial.tools import miniterm
        sys.argv = ["miniterm", port, str(BAUD), "--raw", "--eol", "LF"]
        print("Type command keys (? for help). Ctrl+] quits.")
        miniterm.main()


if __name__ == "__main__":
    main()
