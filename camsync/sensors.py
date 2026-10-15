"""Laptop temperature sensors (hwmon), for correlating frame loss with heat."""
import glob
import os


def read_temperatures():
    """{label: degrees C} for every hwmon temperature sensor, e.g. 'dell_ddv Ambient 2'."""
    out = {}
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(os.path.join(hw, "name")) as f:
                chip = f.read().strip()
        except OSError:
            continue
        if chip in ("dell_smm",):  # duplicates dell_ddv without labels
            continue
        seen = {}
        for inp in sorted(glob.glob(os.path.join(hw, "temp*_input"))):
            try:
                with open(inp) as f:
                    value = int(f.read()) / 1000
                label_file = inp[:-len("_input")] + "_label"
                label = open(label_file).read().strip() if os.path.exists(label_file) \
                    else os.path.basename(inp)[:-len("_input")]
            except (OSError, ValueError):
                continue
            if (chip == "coretemp" and label.startswith("Core")) or \
                    (chip == "nvme" and label.startswith("Sensor")):
                continue  # duplicates of the package / composite temperature
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1 or label in ("Ambient",):
                label = f"{label} {seen[label]}"
            out[f"{chip} {label}"] = value
    return out
