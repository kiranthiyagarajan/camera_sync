"""Record raw synchronized MJPEG from all cameras while the motors run a script:
python3 -m camsync.record [--still 10] [--duration 30]

Writes recordings/<date-time>/: <cam>.mjpeg (frames back to back), <cam>.csv (n, t,
t_raw, offset, size per frame; t is CLOCK_MONOTONIC like the motor log), motors.csv
(t_start, t_end of each move and the motor angles reached) and info.json (format,
controls, camera ports). The robot is held still for --still seconds, then bends by
--amount motor degrees toward 0, 120 and 240 deg (home in between), then sweeps the bend
direction round a circle in 30 deg steps, holding --hold s at each pose, repeating until
--duration. All motors return home after recording.
"""
import argparse
import csv
import json
import os
import threading
import time

from . import config
from .config import PROJECT_DIR
from .motors import Motors, bend_angles
from .rig import Rig

REC_DIR = os.path.join(PROJECT_DIR, "recordings")


def poses(amount):
    for d in (0, 120, 240):
        yield bend_angles(d, amount)
        yield (0.0, 0.0, 0.0)
    for d in range(0, 361, 30):
        yield bend_angles(d, amount)
    yield (0.0, 0.0, 0.0)


def goto(m, angles, log):
    t0 = time.monotonic()
    reached = m.goto(*angles)
    log.writerow([f"{t0:.6f}", f"{time.monotonic():.6f}"] + [f"{a:.3f}" for a in reached])


def motor_script(m, t_start, args, log, stop):
    while time.monotonic() < t_start + args.still and not stop.is_set():
        time.sleep(0.01)
    while not stop.is_set():
        for angles in poses(args.amount):
            if time.monotonic() > t_start + args.duration or stop.is_set():
                return
            goto(m, angles, log)
            time.sleep(args.hold)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--still", type=float, default=10, help="seconds held still first")
    p.add_argument("--duration", type=float, default=30, help="total seconds")
    p.add_argument("--amount", type=float, default=20, help="bend amount, motor degrees")
    p.add_argument("--hold", type=float, default=1.0, help="seconds held at each pose")
    p.add_argument("--exposure", type=float, help="exposure in ms for all cameras (default: rig.json)")
    p.add_argument("--gain", type=int, help="gain for all cameras (default: rig.json)")
    p.add_argument("--no-motors", action="store_true")
    args = p.parse_args()

    out = os.path.join(REC_DIR, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out)
    m = None if args.no_motors else Motors()
    if m:
        if not m.position()[1]:
            raise SystemExit("motor home not set")
    with Rig(queue=1024) as rig:
        for c in rig.ids:
            if args.exposure is not None:
                rig.set_control(c, "exposure_time_absolute", int(round(args.exposure * 10)))
            if args.gain is not None:
                rig.set_control(c, "gain", args.gain)
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
        mlog.writerow(["t_start", "t_end", "a1", "a2", "a3"])
        stop = threading.Event()
        t_start = None
        thread = None
        for s in rig.sets():
            if t_start is None:
                t_start = time.monotonic()
                print(f"recording to {out}")
                if m:
                    thread = threading.Thread(target=motor_script, daemon=True,
                                              args=(m, t_start, args, mlog, stop))
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
        goto(m, (0.0, 0.0, 0.0), mlog)
        mlog_f.close()
        print("motors at", m.position()[0])
        m.close()


if __name__ == "__main__":
    main()
