"""Benchmark: python3 -m camsync [--seconds 10] [--exposure 50] [--decode]

Uses rig.json for camera names and settings; --exposure/--gain override them for this run.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config
from .rig import Rig


def main():
    p = argparse.ArgumentParser(description="Capture all cameras and report sync quality.")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--size", help="1280x720, 800x600 or 640x480 (default: rig.json)")
    p.add_argument("--exposure", type=int, help="100 µs units (max 83 for 120 fps)")
    p.add_argument("--gain", type=int)
    p.add_argument("--decode", type=int, nargs="?", const=1, metavar="REDUCE",
                   help="also decode every frame to grayscale (optionally at 1/2, 1/4, 1/8 size)")
    args = p.parse_args()

    cfg = config.load()
    if args.size:
        cfg["format"]["width"], cfg["format"]["height"] = map(int, args.size.split("x"))

    offsets, complete, total, decoded = [], 0, 0, 0
    pool = ThreadPoolExecutor(8) if args.decode else None
    rig = Rig(cfg)
    for cid in rig.ids:
        print(f"camera {cid}: usb-port {rig.ports[cid]}")
        if args.exposure is not None:
            rig.set_control(cid, "exposure_time_absolute", args.exposure)
        if args.gain is not None:
            rig.set_control(cid, "gain", args.gain)
    with rig:
        end = time.monotonic() + args.seconds
        for s in rig.sets():
            total += 1
            if s.complete:
                complete += 1
                offsets.extend(abs(s.offset(c)) for c in rig.ids[1:])
            if pool:
                frames = [f for f in s.frames.values() if f is not None]
                decoded += len(list(pool.map(lambda f: f.decode_gray(args.decode), frames)))
            if time.monotonic() >= end:
                break

    print()
    for cid in rig.ids:
        st, clock = rig.stats[cid], rig.clocks[cid]
        print(f"camera {cid}: {st.total_frames} frames, {1 / clock.period:.2f} fps, "
              f"lost {st.total_lost} ({st.total_loss:.2%}), corrupt {rig.corrupt[cid]}, "
              f"timestamp jitter {clock.jitter * 1e3:.2f} ms rms (removed by clock model)")
    print(f"sets: {total} emitted, {complete} complete, {rig.sets_dropped} discarded by slow consumer")
    if decoded:
        print(f"decoded {decoded} frames ({decoded / args.seconds:.0f}/s)")
    if offsets:
        o = np.array(offsets) * 1e3
        print(f"inter-camera skew of nearest frames: mean {o.mean():.2f} ms, "
              f"p95 {np.percentile(o, 95):.2f} ms, max {o.max():.2f} ms")


if __name__ == "__main__":
    main()
