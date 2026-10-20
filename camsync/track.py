"""Live proof-of-concept tracking: python3 -m camsync.track

Detects markers in all cameras, triangulates them, groups them into disks and shows
the pose in a 3D view (left-drag rotates, wheel or right-drag zooms, v resets) and
over each camera image (o toggles), with the bend angle of each segment. Press r to
record disk poses to logs/track-*.csv, q to quit. Needs calib/intrinsics, calib/extrinsics.json and
(recommended) calib/masks from the reflection-mask tool.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import calib, config, geometry as geo, mask, markers, pose, view3d
from .config import LOG_DIR
from .imaging import decode, downscale2
from .plots import COLORS, text
from .recon import Reconstructor
from .rig import Rig

WINDOW = "track"
VIEW_H = 440          # px, height of the 3D view and info panel
INFO_W = 400


class Tracker:
    def __init__(self, rig, cams, intr):
        self.rig = rig
        self.intr = intr
        self.cams = cams
        self.recon = Reconstructor(cams, intr)
        self.centres = {c: geo.camera_center(T) for c, T in cams.items()}
        self.masks = {c: mask.load(c, (rig.height, rig.width)) for c in rig.ids}
        self.pool = ThreadPoolExecutor(max(1, len(rig.ids)))
        self.tw, self.th = rig.width // 2, rig.height // 2
        self.writer = self.file = None
        self.rate = 0.0
        self.centre = None  # 3D view centre, follows the robot
        self.orbit = view3d.Orbit(2 * self.tw - INFO_W, VIEW_H)
        self.overlay = True

    def process(self, s):
        def one(c):
            f = s.frames.get(c)
            gray = decode(f.data) if f is not None else None
            dets = markers.detect(gray, self.masks[c])[0] if gray is not None else np.zeros((0, 2))
            return c, gray, dets
        res = list(self.pool.map(one, self.rig.ids))
        dets = {c: d for c, _, d in res}
        points = self.recon.reconstruct(dets)
        xyz = np.array([p[0] for p in points]).reshape(-1, 3)
        seen_from = [[self.centres[c] for c in views] for _, views, _ in points]
        return res, points, xyz, pose.disks(xyz, seen_from)

    def toggle_record(self):
        if self.file:
            self.file.close()
            print(f"recording stopped: {self.path}")
            self.file = self.writer = None
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = os.path.join(LOG_DIR, time.strftime("track-%Y%m%d-%H%M%S.csv"))
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["frame_time", "disk", "x_mm", "y_mm", "z_mm", "dir_x", "dir_y",
                              "dir_z", "markers"])
        print(f"recording to {self.path}")

    def record(self, t, ds):
        for i, d in enumerate(ds):
            v = d["direction"] if d["direction"] is not None else [np.nan] * 3
            self.writer.writerow([f"{t:.6f}", i, *np.round(d["centre"], 2), *np.round(v, 4), d["n"]])

    # ---- drawing ---------------------------------------------------------------
    def tile(self, i, c, gray, dets, points, ds):
        tw, th = self.tw, self.th
        if gray is None:
            img = np.zeros((th, tw, 3), np.uint8)
        else:
            img = cv2.cvtColor(downscale2(gray)[:th, :tw], cv2.COLOR_GRAY2BGR)
        m = self.masks[c]
        if m is not None:
            m2 = m[::2, ::2][:th, :tw]
            img[m2] = (img[m2] * 0.5 + (0, 0, 90)).astype(np.uint8)
        for x, y in dets:
            cv2.circle(img, (int(x / 2), int(y / 2)), 9, (0, 255, 0), 1)
        if self.overlay and c in self.cams:
            view3d.draw_overlay(img, self.cams[c], *self.intr[c], ds, 0.5)
        for X, views, _ in points:
            if c in self.cams:
                px = geo.project(X[None], self.cams[c], *self.intr[c])[0] / 2
                col = (0, 255, 255) if c in views else (0, 0, 255)
                cv2.drawMarker(img, (int(px[0]), int(px[1])), col, cv2.MARKER_CROSS, 12, 1)
        text(img, f"{c}: {len(dets)} blobs", (8, 22), 0.6, COLORS[i % len(COLORS)], 2)
        return img

    def views(self, xyz, ds):
        if len(xyz):
            target = xyz.mean(axis=0)
            self.centre = target if self.centre is None else 0.8 * self.centre + 0.2 * target
        ctr = self.centre if self.centre is not None else np.zeros(3)
        p = np.full((VIEW_H, self.orbit.w, 3), 25, np.uint8)
        view3d.draw_scene(p, self.orbit, ctr, self.cams, self.intr, (self.rig.width, self.rig.height),
                          xyz, ds)
        text(p, "drag rotate, wheel/right-drag zoom, v reset (grid 50 mm)", (8, 20), 0.5,
             (200, 200, 200))
        info = np.full((VIEW_H, INFO_W, 3), 15, np.uint8)
        lines = [f"{len(xyz)} markers, {len(ds)} disks   {self.rate:.0f} Hz",
                 "REC " + os.path.basename(self.path) if self.file else "r record  o overlay  q quit"]
        for i, d in enumerate(ds):
            c = d["centre"]
            tag = "tip  " if i == len(ds) - 1 else f"disk{i}"
            lines.append(f"{tag} ({c[0]:6.0f},{c[1]:6.0f},{c[2]:6.0f}) n={d['n']}")
        lines.append("")
        segs, total = pose.bend_angles(ds)
        for i, a in enumerate(segs):
            lines.append(f"bend disk{i}-{'tip' if i == len(segs) - 1 else f'disk{i + 1}'}: " + _deg(a))
        if len(ds) > 1:
            lines.append("bend base-tip: " + _deg(total))
            lines.append("~ = from disk centres (disk normal unknown)")
        for i, ln in enumerate(lines):
            text(info, ln, (8, 22 + 22 * i), 0.5, (0, 255, 255) if i < 2 else (220, 220, 220))
        return np.hstack([p, info])

    def run(self, duration=None, snapshot=None):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, lambda e, x, y, f, _: self.orbit.mouse(e, x, y - 2 * self.th, f))
        start = last = time.monotonic()
        sized = False
        try:
            while True:
                sets = [s for s in self.rig.poll_sets() if s.frames]
                if not sets:
                    if cv2.waitKey(2) & 0xFF == ord("q"):
                        break
                    continue
                s = sets[-1]
                res, points, xyz, ds = self.process(s)
                now = time.monotonic()
                self.rate = 0.9 * self.rate + 0.1 / max(now - last, 1e-3)
                last = now
                if self.writer:
                    self.record(s.t, ds)
                tiles = [self.tile(i, c, g, d, points, ds) for i, (c, g, d) in enumerate(res)]
                while len(tiles) < 4:
                    tiles.append(np.zeros_like(tiles[0]))
                grid = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])])
                bottom = self.views(xyz, ds)
                w = max(grid.shape[1], bottom.shape[1])
                canvas = np.vstack([_pad(grid, w), _pad(bottom, w)])
                cv2.imshow(WINDOW, canvas)
                if not sized:
                    cv2.resizeWindow(WINDOW, canvas.shape[1] // 2 * 2, canvas.shape[0])
                    sized = True
                if duration and now - start >= duration:
                    if snapshot:
                        cv2.imwrite(snapshot, canvas)
                    break
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    self.toggle_record()
                if k == ord("o"):
                    self.overlay = not self.overlay
                if k == ord("v"):
                    self.orbit.reset()
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                    break
        finally:
            if self.file:
                self.toggle_record()
            cv2.destroyAllWindows()
            self.pool.shutdown()


def _deg(a):
    deg, fitted = a
    return "-" if deg is None else f"{'' if fitted else '~'}{deg:5.1f} deg"


def _pad(img, w):
    if img.shape[1] == w:
        return img
    return np.hstack([img, np.zeros((img.shape[0], w - img.shape[1], 3), np.uint8)])


def main():
    p = argparse.ArgumentParser(description="Live proof-of-concept marker tracking and robot pose.")
    p.add_argument("--config", default=config.DEFAULT_PATH)
    p.add_argument("--exposure", type=float, help="exposure in ms for all cameras (default: rig.json)")
    p.add_argument("--gain", type=int, help="gain for all cameras (default: rig.json)")
    p.add_argument("--duration", type=float, help="quit after this many seconds")
    p.add_argument("--snapshot", help="with --duration: save the final window image")
    args = p.parse_args()

    cfg = config.load(args.config)
    ext = calib.load_extrinsics()
    if ext is None:
        sys.exit("no calib/extrinsics.json: run the camera position calibration first")
    cams = ext[0]
    intr = {}
    for c in cams:
        got = calib.load_intrinsics(c)
        if got is not None:
            intr[c] = got[:2]
    infos, missing, _ = config.resolve(cfg)
    for name in missing:
        print(f"WARNING: {name} is not connected")
    rig = Rig(cfg, infos, queue=2)
    with rig:
        for c in rig.ids:
            if args.exposure is not None:
                rig.set_control(c, "exposure_time_absolute", int(round(args.exposure * 10)))
            if args.gain is not None:
                rig.set_control(c, "gain", args.gain)
        nomask = [c for c in rig.ids if mask.load(c) is None]
        if nomask:
            print(f"no reflection mask for {', '.join(nomask)}: IR rings and rails may show as markers")
        Tracker(rig, cams, intr).run(args.duration, args.snapshot)


if __name__ == "__main__":
    main()
