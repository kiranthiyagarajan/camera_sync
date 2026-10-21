"""Fit the robot model to every held pose of a recording: python3 -m camsync.holds DIR

For each motor move in DIR/motors.csv, frames from the last --window seconds before the
next move (the robot has settled) are pooled and the model is fitted to their 2D
detections. Writes DIR/holds.csv: motor angles, tip position (mm) and direction in the
robot's base frame, fit rms (px), detections matched per frame. Home poses are listed
with their spread at the end (repeatability).

The fixed robot parameters (base pose, disk rotations) are fitted from the first home
poses and saved to calib/robot_model.json; --refit redoes that (after the robot or the
cameras moved).
"""
import argparse
import csv
import os

import numpy as np

from . import calib, model as rm, pose
from .replay import Recording
from .recon import Reconstructor


def holds(path, window):
    """(angles, t_from, t_to) for the initial still pose and every move's settled hold."""
    moves = list(csv.DictReader(open(os.path.join(path, "motors.csv"))))
    out = [((0.0, 0.0, 0.0), float(moves[0]["t_start"]) - window - 0.1,
            float(moves[0]["t_start"]) - 0.1)] if moves else []
    for i, m in enumerate(moves):
        end = float(moves[i + 1]["t_start"]) if i + 1 < len(moves) else float(m["t_end"]) + 1.0
        out.append((tuple(float(m[k]) for k in ("a1", "a2", "a3")), max(float(m["t_end"]), end - window - 0.1), end - 0.1))
    return out


def build_model(rec, cams, intr, home_frames):
    recon = Reconstructor(cams, intr)
    pts = [p[0] for f in home_frames[0] for p in recon.reconstruct(f)]
    if pts:
        keep = pose.on_backbone(pts)
        pts = [p for p, k in zip(pts, keep) if k]
    if len(pts) < 4:
        raise SystemExit("too few triangulated markers at home to place the model")
    m = rm.search_start(rm.initial_model(pts), home_frames[0], cams, intr)
    m, _ = rm.fit_model(home_frames, cams, intr, m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir")
    p.add_argument("--window", type=float, default=1.5, help="settled seconds pooled per hold")
    p.add_argument("--frames", type=int, default=15, help="frames pooled per hold")
    p.add_argument("--refit", action="store_true", help="refit the fixed robot parameters")
    args = p.parse_args()

    cams, _ = calib.load_extrinsics()
    intr = {c: calib.load_intrinsics(c)[:2] for c in cams}
    rec = Recording(args.dir, cams)
    hs = holds(args.dir, args.window)
    frames = [rec.frames(a, b, args.frames) for _, a, b in hs]
    home = [f for (ang, _, _), f in zip(hs, frames) if not any(ang)]

    model = None if args.refit else rm.Model.load()
    if model is None:
        model = build_model(rec, cams, intr, home[:3])
        model.save()
        print(f"robot model saved to {rm.PATH}: psi {np.round(np.degrees(model.psi), 1)} deg")

    out = os.path.join(args.dir, "holds.csv")
    tips_home = []
    prev = None
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "a1", "a2", "a3", "tip_x", "tip_y", "tip_z", "dir_x", "dir_y", "dir_z",
                    "rms_px", "matched"] + [f"k{i}{a}" for i in range(rm.N_DISKS) for a in "xy"])
        print("    a1     a2     a3 |  tip x    y      z  (mm, base) | rms px matched")
        for (ang, a, b), fr in zip(hs, frames):
            curv, rms, matched = rm.fit_best(model, fr, cams, intr, prev)
            prev = curv
            pos, d = rm.tip(model, curv)
            if not any(ang):
                tips_home.append(pos)
            w.writerow([f"{b:.3f}", *ang, *np.round(pos, 2), *np.round(d, 4), round(rms, 2),
                        round(matched, 1), *np.round(curv, 6)])
            print(f"{ang[0]:6.1f} {ang[1]:6.1f} {ang[2]:6.1f} | {pos[0]:6.1f} {pos[1]:6.1f} "
                  f"{pos[2]:6.1f}             | {rms:5.2f} {matched:5.1f}")
    th = np.array(tips_home)
    if len(th) > 1:
        print(f"home tip over {len(th)} returns: std {np.round(th.std(axis=0), 1)} mm, "
              f"max distance from mean {np.linalg.norm(th - th.mean(0), axis=1).max():.1f} mm")
    print(f"written {out}")


if __name__ == "__main__":
    main()
