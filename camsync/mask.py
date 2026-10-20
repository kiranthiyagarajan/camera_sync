"""Static reflection mask: python3 -m camsync.mask [--seconds 3]

Run with the tracking exposure, the IR rings on and no markers in view. Every pixel
that gets bright during the recording (cube rails, other cameras' IR rings, glints)
is masked, plus a margin around it. Saved per camera as calib/masks/<camera>.png
(white = masked) and used by marker auto exposure, the dashboard overlay and later by
marker detection. Redo it whenever a camera moves or the exposure changes a lot.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from . import config
from .config import CALIB_DIR
from .imaging import decode
from .rig import Rig

MASK_DIR = os.path.join(CALIB_DIR, "masks")
THRESHOLD = 100   # brightest value a pixel reached; markers are exposed to ~235
GROW_PX = 4       # margin added around every bright area
MAX_EXPOSURE = 20  # 2 ms: above this the mask is built at the wrong (calibration) exposure


def mask_path(cam):
    return os.path.join(MASK_DIR, f"{cam}.png")


def load(cam, shape=None):
    """Boolean mask (True = ignore) for a camera, or None if there is none or it was made
    at another resolution."""
    img = cv2.imread(mask_path(cam), cv2.IMREAD_GRAYSCALE)
    if img is None or (shape is not None and img.shape != tuple(shape)):
        return None
    return img > 0


def build(peak, threshold=THRESHOLD, grow=GROW_PX):
    """Mask from the per-pixel maximum image: bright pixels plus a `grow`-pixel margin."""
    mask = (peak >= threshold).astype(np.uint8)
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        mask = cv2.dilate(mask, k)
    return mask > 0


def preview(peak, mask, label):
    img = cv2.cvtColor(peak, cv2.COLOR_GRAY2BGR)
    img[mask] = (0.5 * img[mask] + (0, 0, 127)).astype(np.uint8)
    cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return img


def main():
    p = argparse.ArgumentParser(description="Record a static reflection mask per camera "
                                            "(no markers in view, tracking exposure).")
    p.add_argument("--config", default=config.DEFAULT_PATH)
    p.add_argument("--seconds", type=float, default=3)
    p.add_argument("--threshold", type=int, default=THRESHOLD,
                   help="mask pixels that reach this brightness (0-255)")
    p.add_argument("--grow", type=int, default=GROW_PX, help="margin around bright areas (px)")
    args = p.parse_args()

    cfg = config.load(args.config)
    cams, missing, _ = config.resolve(cfg)
    for name in missing:
        print(f"WARNING: {name} (port {cfg['cameras'][name]}) is not connected")
    if not cams:
        sys.exit("no cameras found")
    print(f"recording {args.seconds:.0f} s: remove or cover all markers, IR rings on")
    with Rig(cfg, cams) as rig:
        peak, last = {}, {}
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            for c in rig.ids:
                f = rig.latest.get(c)
                if f is None or last.get(c) is f:
                    continue
                last[c] = f
                gray = decode(f.data)
                if gray is not None:
                    peak[c] = gray if c not in peak else np.maximum(peak[c], gray)
            time.sleep(0.005)
        controls = {c: dict(rig.current[c]) for c in rig.ids}

    os.makedirs(MASK_DIR, exist_ok=True)
    tiles = []
    for c in sorted(peak):
        mask = build(peak[c], args.threshold, args.grow)
        cv2.imwrite(mask_path(c), mask.astype(np.uint8) * 255)
        regions = cv2.connectedComponents(mask.astype(np.uint8))[0] - 1
        exposure = controls[c].get("exposure_time_absolute", 0)
        note = ""
        if exposure > MAX_EXPOSURE:
            note += f"  [exposure {exposure / 10:.1f} ms is a calibration setting: set the " \
                    "tracking exposure first (m in the dashboard)]"
        if mask.mean() > 0.05:
            note += "  [large: markers in view, or exposure too high?]"
        print(f"{c}: {mask.mean():.2%} of the image masked in {regions} areas "
              f"(exposure {exposure / 10:.1f} ms, gain {controls[c].get('gain')}) "
              f"-> {mask_path(c)}{note}")
        tiles.append(preview(peak[c], mask, f"{c}: {mask.mean():.2%} masked"))
    if tiles:
        h, w = tiles[0].shape[:2]
        tiles = [cv2.resize(t, (w // 2, h // 2)) for t in tiles]
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        out = os.path.join(MASK_DIR, "preview.png")
        cv2.imwrite(out, grid)
        print(f"preview (masked areas tinted red): {out}")


if __name__ == "__main__":
    main()
