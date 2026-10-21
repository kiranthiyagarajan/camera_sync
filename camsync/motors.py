"""Talk to the ESP32 stepper controller (firmware/stepper_control) over USB serial.

Firmware v2 keeps all motors enabled and saves positions (relative to home) in flash.
Line commands: "G a1 a2 a3" moves all motors together to angles from home, H sets home,
P prints "POS a1 a2 a3 home=.. enabled=..". Single keys as before: 1/2/3 select a motor,
+ / - move it 18 degrees, 0 sets or returns to home, p prints positions, x disables all
motors (e enables), ? shows help.

The three tendons are coupled (pulling two forces the third to pay out), so prefer
bend(direction, amount): motor i turns amount*cos(direction - 120*i), which always sums
to zero.

    python3 -m camsync.motors 1 + + p      # send keys, print the replies
    python3 -m camsync.motors --bend 0 30  # bend toward motor 1's side by 30 (motor deg)
    python3 -m camsync.motors --goto 0 0 0 # back home
    python3 -m camsync.motors              # interactive terminal (Ctrl+] quits)
    python3 -m camsync.motors --flash      # compile + upload the firmware

Opening the port does not reset the ESP32 (DTR and RTS are left asserted together, which
the auto-reset circuit ignores; clearing DTR first pulses EN), so its software positions
and home survive reconnecting.
"""
import argparse
import math
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
LIMIT_DEG = 90.0         # refuse motor angles beyond +-this from home
SECONDS_PER_DEG = 0.0178  # 1600 pulses/rev at 4 ms per pulse


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

    def _reply(self, timeout):
        """Read lines until an OK/STOPPED/ERR line; returns (status, angles or None)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            for status in ("OK", "STOPPED", "ERR"):
                if line.startswith(status):
                    if status == "ERR":
                        raise RuntimeError(line)
                    return status, _parse_pos(line[line.index("POS"):]) if "POS" in line else None
        raise TimeoutError("no reply from the ESP32")

    def position(self):
        """Motor angles from home (degrees), and whether home is set."""
        self.ser.reset_input_buffer()
        self.ser.write(b"P\n")
        deadline = time.time() + 2
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if line.startswith("POS"):
                return _parse_pos(line), "home=1" in line
        raise TimeoutError("no reply from the ESP32")

    def goto(self, a1, a2, a3, limit=LIMIT_DEG):
        """Move all motors together to these angles from home; blocks until done.
        Returns the reached angles."""
        target = (a1, a2, a3)
        if any(abs(a) > limit for a in target):
            raise ValueError(f"motor angles {target} beyond +-{limit} deg")
        current, _ = self.position()
        move = max(abs(t - c) for t, c in zip(target, current))
        self.ser.write(f"G {a1:.3f} {a2:.3f} {a3:.3f}\n".encode())
        status, reached = self._reply(move * SECONDS_PER_DEG + 3)
        if status != "OK":
            raise RuntimeError(f"move {status.lower()} at {reached}")
        return reached

    def bend(self, direction, amount, limit=LIMIT_DEG):
        """Bend toward `direction` (degrees, 0 = motor 1, 120 = motor 2, 240 = motor 3)
        by `amount` motor degrees; the three angles sum to zero."""
        return self.goto(*bend_angles(direction, amount), limit=limit)

    def close(self):
        self.ser.close()


def bend_angles(direction, amount):
    return tuple(amount * math.cos(math.radians(direction - 120 * i)) for i in range(3))


def _parse_pos(line):
    return tuple(float(v) for v in line.split()[1:4])


def flash(port):
    cli = "arduino-cli"
    subprocess.run([cli, "compile", "--fqbn", FQBN, SKETCH], check=True)
    subprocess.run([cli, "upload", "--fqbn", FQBN, "-p", port, SKETCH], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help="command keys to send, e.g. 1 + p")
    ap.add_argument("--port", help="serial port (default: the first CP210x/CH340)")
    ap.add_argument("--flash", action="store_true", help="compile and upload the firmware")
    ap.add_argument("--goto", nargs=3, type=float, metavar="A", help="move to motor angles from home")
    ap.add_argument("--bend", nargs=2, type=float, metavar=("DIR", "AMOUNT"),
                    help="bend toward DIR degrees (0 = motor 1) by AMOUNT motor degrees")
    args = ap.parse_args()
    port = args.port or find_port()
    if args.flash:
        flash(port)
    elif args.goto or args.bend:
        m = Motors(port)
        print("reached", m.goto(*args.goto) if args.goto else m.bend(*args.bend))
        m.close()
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
