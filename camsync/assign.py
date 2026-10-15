"""Name the cameras: python3 -m camsync.assign [--names cam1,cam2,cam3,cam4]

All cameras look identical to the computer (same serial number), so each name is
tied to the USB port the camera is plugged into. For each name in turn, cover
that camera (or shine a light into it); the feed that changes gets the name.
Saved to rig.json; camera settings and calibration are then keyed by name.
"""
import argparse
import copy
import sys
import time

import cv2
import numpy as np

from . import config
from .imaging import CoverDetector, decode
from .rig import Rig
from .rigview import COLORS, text

WINDOW = "assign cameras"


def mean_brightness(rig):
    out = {}
    for cid in rig.ids:
        f = rig.latest.get(cid)
        if f is not None:
            small = decode(f.data, 8)
            if small is not None:
                out[cid] = float(small.mean())
    return out


def main():
    p = argparse.ArgumentParser(description="Assign names to cameras by covering them one at a time.")
    p.add_argument("--config", default=config.DEFAULT_PATH)
    p.add_argument("--names", default="cam1,cam2,cam3,cam4")
    args = p.parse_args()
    names = [n.strip() for n in args.names.split(",") if n.strip()]

    cfg = config.load(args.config)
    unnamed = copy.deepcopy(cfg)
    unnamed["cameras"] = {}  # identify every camera by port while assigning
    cams, _, _ = config.resolve(unnamed)
    if not cams:
        sys.exit("no cameras found")
    if len(names) > len(cams):
        print(f"only {len(cams)} cameras connected; naming the first {len(cams)} names")
        names = names[:len(cams)]

    rig = Rig(unnamed, cams)
    detector = CoverDetector()
    assigned = {}        # port -> name
    waiting_release = None
    tw, th = rig.width // 2, rig.height // 2
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with rig:
        # Baseline brightness with every camera uncovered.
        samples = {c: [] for c in rig.ids}
        end = time.monotonic() + 2.5
        while time.monotonic() < end:
            for c, m in mean_brightness(rig).items():
                samples[c].append(m)
            time.sleep(0.05)
        missing = [c for c in rig.ids if not samples[c]]
        if missing:
            sys.exit(f"no frames from: {', '.join(missing)}")
        detector.set_baseline({c: float(np.median(v)) for c, v in samples.items()})
        save = False
        while True:
            means = mean_brightness(rig)
            now = time.monotonic()
            idx = len(assigned)
            if waiting_release:
                if detector.released(means, waiting_release):
                    waiting_release = None
            elif idx < len(names):
                cid = detector.update(means, now, [c for c in rig.ids if c not in assigned])
                if cid:
                    assigned[cid] = names[idx]
                    waiting_release = cid
                    print(f"{names[idx]} = port {cid}")

            if waiting_release:
                prompt = f"{assigned[waiting_release]} = port {waiting_release}. Uncover it."
            elif idx < len(names):
                prompt = f"Cover the camera to name '{names[idx]}' (or shine a light into it)"
            else:
                prompt = "All cameras named. s = save, u = undo last, q = quit without saving"

            grid = np.zeros((2 * th, 2 * tw, 3), np.uint8)
            for i, cid in enumerate(rig.ids[:4]):
                f = rig.latest.get(cid)
                tile = np.zeros((th, tw, 3), np.uint8)
                if f is not None:
                    g = decode(f.data, 2)
                    if g is not None:
                        tile[:] = cv2.cvtColor(g[:th, :tw], cv2.COLOR_GRAY2BGR)
                name = assigned.get(cid, "?")
                text(tile, f"port {cid}  ->  {name}", (10, 34), 0.9,
                     COLORS[i % len(COLORS)] if cid in assigned else (255, 255, 255), 2)
                if cid in means:
                    text(tile, f"brightness {means[cid]:.0f} (baseline {detector.baseline[cid]:.0f})",
                         (10, 62))
                r, c = divmod(i, 2)
                grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
            panel = np.full((60, grid.shape[1], 3), 15, np.uint8)
            text(panel, prompt, (10, 26), 0.7, (0, 255, 255), 2)
            text(panel, "u undo last   s save   q quit without saving", (10, 50), 0.45, (190, 190, 190))
            cv2.imshow(WINDOW, np.vstack([grid, panel]))

            k = cv2.waitKey(60) & 0xFF
            if k in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                break
            if k == ord("u") and assigned:
                last = list(assigned)[-1]
                print(f"undo: {assigned.pop(last)} (port {last})")
                waiting_release = None
            if k == ord("s") and assigned:
                save = True
                break
    cv2.destroyAllWindows()

    if not save:
        print("not saved")
        return
    cfg = config.load(args.config)
    cfg["cameras"] = {name: port for port, name in assigned.items()}
    # Settings saved earlier under a port name move to the camera's new name.
    for port, name in assigned.items():
        if port in cfg["controls"] and name not in cfg["controls"]:
            cfg["controls"][name] = cfg["controls"].pop(port)
    config.save(cfg, args.config)
    print(f"saved to {args.config}:")
    for name, port in sorted(cfg["cameras"].items()):
        print(f"  {name}: port {port}")
    if len(assigned) < len(names):
        print(f"note: only {len(assigned)} of {len(names)} names assigned")


if __name__ == "__main__":
    main()
