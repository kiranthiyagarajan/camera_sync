"""Stable rough robot pose and tip position from triangulated markers.

Points that persist over the last few frames are kept (ghost matches come and go),
grouped into disk levels by their distance from a fixed base point below disk 1, and
each level's mean is taken as a rough disk centre. The tip is the top disk (disk 4); if
its markers are not triangulated it is extrapolated from the two highest disks found.
Disk centres and tip are smoothed over time.

Offline on a recording (10 Hz): python3 -m camsync.tip DIR  -> DIR/tip.csv
The base point is set from the robot upright at home (below the highest marker) and
saved to calib/robot_base.json (key b in the tracker sets it again).
"""
import argparse
import csv
import json
import os
from collections import deque

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from . import calib
from .config import CALIB_DIR

N_DISKS = 4
FIRST_MM = 70.0     # base point -> disk 1
SPACING_MM = 68.0   # disk to disk (centre to centre)
LEVEL_GAP_MM = 30.0  # points further apart than this (in distance from base) are separate levels
SAME_POINT_MM = 4.0  # one marker seen in different frames
BASE_PATH = os.path.join(CALIB_DIR, "robot_base.json")


def load_base():
    if not os.path.exists(BASE_PATH):
        return None
    with open(BASE_PATH) as f:
        return np.array(json.load(f)["base_mm"])


def save_base(base):
    with open(BASE_PATH, "w") as f:
        json.dump({"base_mm": np.round(base, 2).tolist()}, f)


class TipEstimator:
    def __init__(self, base=None, window=6, min_frac=0.4, smooth=0.5):
        self.base = base
        self.frames = deque(maxlen=window)
        self.min_frac = min_frac
        self.smooth = smooth
        self.disks = [None] * N_DISKS  # smoothed centres
        self.tip = None
        self.found = []                # disk indices seen in the last update

    def persistent(self):
        pts = [p for f in self.frames for p in f]
        if len(pts) < 2:
            return np.zeros((0, 3))
        pts = np.array(pts)
        lab = fcluster(linkage(pts, "single"), SAME_POINT_MM, "distance")
        need = self.min_frac * len(self.frames)
        return np.array([pts[lab == k].mean(0) for k in np.unique(lab) if (lab == k).sum() >= need])

    def levels(self, P):
        """{disk index: rough centre} from persistent points."""
        dist = np.linalg.norm(P - self.base, axis=1)
        order = np.argsort(dist)
        groups = [[order[0]]]
        for i, j in zip(order[:-1], order[1:]):
            if dist[j] - dist[i] < LEVEL_GAP_MM:
                groups[-1].append(j)
            else:
                groups.append([j])
        out = {}
        for g in groups:
            c = P[g].mean(0)
            k = int(round((np.linalg.norm(c - self.base) - FIRST_MM) / SPACING_MM))
            if 0 <= k < N_DISKS:
                out.setdefault(k, []).append(c)
        return {k: np.mean(v, axis=0) for k, v in out.items()}

    def update(self, points):
        """Add one frame's 3D points; returns the smoothed tip (3,) or None."""
        self.frames.append(np.asarray(points, float).reshape(-1, 3))
        P = self.persistent()
        if not len(P):
            self.found = []
            return self.tip
        if self.base is None:  # first sight of the robot, upright at home: the highest
            # point is on the top disk (stray points turn up lower down, not higher)
            self.base = np.r_[np.median(P[:, :2], axis=0),
                              P[:, 2].max() - FIRST_MM - (N_DISKS - 1) * SPACING_MM]
        lv = self.levels(P)
        self.found = sorted(lv)
        if not lv:
            return self.tip
        top = max(lv)
        tip = lv[top]
        if top < N_DISKS - 1:
            below = [k for k in lv if k < top]
            ref = lv[max(below)] if below else self.base
            v = tip - ref
            n = np.linalg.norm(v)
            if n > 1e-6:
                tip = tip + v / n * SPACING_MM * (N_DISKS - 1 - top)
        a = self.smooth
        for k, c in lv.items():
            self.disks[k] = c if self.disks[k] is None else a * self.disks[k] + (1 - a) * c
        self.tip = tip if self.tip is None else a * self.tip + (1 - a) * tip
        return self.tip

    def pose(self):
        """Disk dicts (centre, normal, direction, n) for the 3D view, base to tip; the tip
        disk uses the (possibly extrapolated) tip position."""
        cs = [c for c in self.disks[:-1] if c is not None]
        if self.tip is not None:
            cs.append(self.tip)
        out = []
        for i, c in enumerate(cs):
            v = cs[min(len(cs) - 1, i + 1)] - cs[max(0, i - 1)] if len(cs) > 1 else np.array([0, 0, 1.0])
            n = np.linalg.norm(v)
            out.append({"centre": c, "normal": None, "direction": v / n if n else None, "n": 0,
                        "markers": np.zeros((0, 3))})
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir")
    p.add_argument("--rate", type=float, default=10.0, help="output rate, Hz")
    p.add_argument("--per-update", type=int, default=2, help="frames added per output")
    args = p.parse_args()
    from .replay import Recording
    from .recon import Reconstructor

    cams, _ = calib.load_extrinsics()
    intr = {c: calib.load_intrinsics(c)[:2] for c in cams}
    rec = Recording(args.dir, cams)
    recon = Reconstructor(cams, intr)
    est = TipEstimator(load_base())
    t_end = max(ts[-1] for ts, _ in rec.idx.values())
    out = os.path.join(args.dir, "tip.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "tip_x", "tip_y", "tip_z", "disks_found"])
        dt = 1.0 / args.rate
        for t in np.arange(rec.t0 + 0.5, t_end, dt):
            for k in range(args.per_update):
                tip = est.update([q[0] for q in recon.reconstruct(rec.detections(t - k * dt / args.per_update))])
            if tip is not None:
                w.writerow([f"{t:.3f}", *np.round(tip, 1), "".join(str(k + 1) for k in est.found)])
    if load_base() is None and est.base is not None:
        save_base(est.base)
        print(f"base point saved to {BASE_PATH}: {np.round(est.base, 1)}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
