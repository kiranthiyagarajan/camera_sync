"""Record raw synchronized MJPEG from all cameras while the motors run a script:
python3 -m camsync.record [--still 10] [--duration 30]

Writes recordings/<date-time>/: <cam>.mjpeg (frames back to back), <cam>.csv (n, t,
t_raw, offset, size per frame; t is CLOCK_MONOTONIC like the motor log), motors.csv
(time, key sent, replies) and info.json (format, controls, camera ports). The robot is
held still for --still seconds, then each motor in turn goes +54 deg and back in 18 deg
steps until --duration, and all motors return home after recording.
"""
import argparse
import csv
import json
import os
import threading
import time

from . import config
from .config import PROJECT_DIR
from .motors import Motors
from .rig import Rig

REC_DIR = os.path.join(PROJECT_DIR, "recordings")
STEPS = 3  # 3 x 18 = 54 degrees


def motor_script(m, t_start, still, duration, log, stop):
    def send(key):
        replies = m.send(key)
        log.writerow([f"{time.monotonic():.6f}", key, " | ".join(r for r in replies if r)])
    while time.monotonic() < t_start + still and not stop.is_set():
        time.sleep(0.01)
    while not stop.is_set():
        for motor in "123":
            send(motor)
            for key in "+" * STEPS + "-" * STEPS:
                if time.monotonic() > t_start + duration or stop.is_set():
                    return
                send(key)
                time.sleep(0.3)  # short hold at each pose


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--still", type=float, default=10, help="seconds held still first")
    p.add_argument("--duration", type=float, default=30, help="total seconds")
    p.add_argument("--no-motors", action="store_true")
    args = p.parse_args()

    out = os.path.join(REC_DIR, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out)
    m = None if args.no_motors else Motors()
    if m:
        time.sleep(1.5)  # opening the port may reset the ESP32 once
        m.ser.reset_input_buffer()
    with Rig(queue=1024) as rig:
        json.dump({"format": {"width": rig.width, "height": rig.height, "fps": rig.fps},
                   "ports": rig.ports, "controls": rig.current},
                  open(os.path.join(out, "info.json"), "w"), indent=2)
        files = {c: open(os.path.join(out, f"{c}.mjpeg"), "wb") for c in rig.ids}
        idx = {c: open(os.path.join(out, f"{c}.csv"), "w", newline="") for c in rig.ids}
        writers = {c: csv.writer(f) for c, f in idx.items()}
        for w in writers.values():
            w.writerow(["n", "t", "t_raw", "offset", "size"])
        last = {c: None for c in rig.ids}
        mlog_f = open(os.path.join(out, "motors.csv"), "w", newline="")
        mlog = csv.writer(mlog_f)
        mlog.writerow(["t", "key", "reply"])
        stop = threading.Event()
        t_start = None
        thread = None
        for s in rig.sets():
            if t_start is None:
                t_start = time.monotonic()
                print(f"recording to {out}")
                if m:
                    thread = threading.Thread(target=motor_script, daemon=True,
                                              args=(m, t_start, args.still, args.duration, mlog, stop))
                    thread.start()
            for c, f in s.frames.items():
                if f is None or f.n == last[c]:
                    continue
                last[c] = f.n
                writers[c].writerow([f.n, f"{f.t:.6f}", f"{f.t_raw:.6f}", files[c].tell(), len(f.data)])
                files[c].write(f.data)
            if time.monotonic() > t_start + args.duration:
                break
        stop.set()
        if thread:
            thread.join()
        for f in list(files.values()) + list(idx.values()):
            f.close()
        for c in rig.ids:
            st = rig.stats[c]
            print(f"{c}: {st.total_frames} frames, {st.total_lost} lost, {rig.corrupt[c]} corrupt")
    if m:
        for motor in "123":
            mlog.writerow([f"{time.monotonic():.6f}", motor + "0", " | ".join(m.send(motor + "0", quiet=1.0))])
        mlog_f.close()
        print("\n".join(m.send("p")))
        m.close()


if __name__ == "__main__":
    main()
