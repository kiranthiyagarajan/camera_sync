"""Guided calibration with the ChArUco board.

  python3 -m camsync.calibrate intrinsics   # each camera's lens (do this first)
  python3 -m camsync.calibrate extrinsics   # where the cameras are (world frame)

Set exposure for the board first (rigview: b, then s). Focus must not change after
calibrating. Results go to calib/; captured images to calib/views/ (not in git).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import calib, config
from .board import Board
from .config import CALIB_DIR
from .imaging import decode
from .rig import Rig
from .plots import COLORS, text

WINDOW = "calibrate"
TICK_S = 0.12          # detection rate per camera
STILL_PX = 0.8         # mean corner movement between ticks that counts as "still"
STILL_TICKS = 3
MIN_CORNERS = 12
MIN_VIEWS, TARGET_VIEWS = 15, 25
NOVEL_PX = 50          # a new view must differ this much (mean corner shift) from all others
GRID = (16, 9)         # coverage cells across x, y


class CamState:
    def __init__(self):
        self.det = None         # (corners, ids) from the latest tick
        self.prev = None
        self.still = 0
        self.views = []
        self.coverage = np.zeros(GRID[::-1], int)
        self.flash_until = 0.0
        self.result = None      # text after calibration
        self.pose_err = None    # extrinsics: board reprojection error in px


def shift(a, b):
    """Mean movement (px) of corners common to two detections, and the fraction in common."""
    (ca, ia), (cb, ib) = a, b
    common, xa, xb = np.intersect1d(ia, ib, return_indices=True)
    if not len(common):
        return np.inf, 0.0
    d = np.linalg.norm(ca[xa] - cb[xb], axis=1).mean()
    return float(d), len(common) / max(len(ia), 1)


class Calibrator:
    def __init__(self, rig, board, mode):
        self.rig, self.board, self.mode = rig, board, mode
        self.state = {c: CamState() for c in rig.ids}
        self.pool = ThreadPoolExecutor(max(1, len(rig.ids)))
        self.tw, self.th = rig.width // 2, rig.height // 2
        self.message, self.message_until = "", 0.0
        self.intr = {}
        self.captures = []      # extrinsics: list of {cam: (corners, ids)}
        self.summary = []
        self.last_dict_check = time.monotonic()
        if mode == "extrinsics":
            missing = []
            for c in rig.ids:
                got = calib.load_intrinsics(c)
                if got is None:
                    missing.append(c)
                else:
                    self.intr[c] = got[:2]
            if missing:
                sys.exit(f"no intrinsics for {', '.join(missing)}: run "
                         "python3 -m camsync.calibrate intrinsics first")

    def say(self, msg, secs=6):
        self.message, self.message_until = msg, time.monotonic() + secs
        print(msg)

    def detect(self, cid):
        f = self.rig.latest.get(cid)
        if f is None:
            return cid, None, None
        gray = decode(f.data)
        if gray is None:
            return cid, None, None
        K, dist = self.intr.get(cid, (None, None))
        return cid, gray, self.board.detect(gray, K, dist)

    # ---- per tick --------------------------------------------------------------
    def update(self, results):
        now = time.monotonic()
        for cid, gray, det in results:
            st = self.state[cid]
            st.prev, st.det = st.det, det
            if det is None or len(det[1]) < MIN_CORNERS:
                st.still = 0
                st.pose_err = None
                continue
            moved = shift(det, st.prev)[0] if st.prev is not None else np.inf
            st.still = st.still + 1 if moved < STILL_PX else 0
            if self.mode == "intrinsics":
                self.maybe_capture_view(cid, gray, now)
            else:
                st.pose_err = calib.board_pose(self.board, *det, *self.intr[cid])[1]
        if all(r[2] is None for r in results) and now - self.last_dict_check > 5:
            self.last_dict_check = now
            gray = next((r[1] for r in results if r[1] is not None), None)
            if gray is not None:
                guess = self.board.guess_dictionary(gray)
                if guess and guess != self.board.spec["dictionary"]:
                    self.say(f"board not found with {self.board.spec['dictionary']} but markers "
                             f"from {guess} are visible - check the board settings in rig.json")

    def maybe_capture_view(self, cid, gray, now):
        st = self.state[cid]
        if st.still < STILL_TICKS:
            return
        for v in st.views:
            d, frac = shift(st.det, v)
            if d < NOVEL_PX and frac > 0.6:
                return  # too similar to a view we already have
        st.views.append(st.det)
        corners = st.det[0]
        gx = np.clip((corners[:, 0] / self.rig.width * GRID[0]).astype(int), 0, GRID[0] - 1)
        gy = np.clip((corners[:, 1] / self.rig.height * GRID[1]).astype(int), 0, GRID[1] - 1)
        np.add.at(st.coverage, (gy, gx), 1)
        st.flash_until = now + 0.4
        out = os.path.join(CALIB_DIR, "views", cid)
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, f"{len(st.views):03d}.png"), gray)

    # ---- actions ---------------------------------------------------------------
    def compute_intrinsics(self):
        size = (self.rig.width, self.rig.height)
        for cid, st in self.state.items():
            if len(st.views) < MIN_VIEWS:
                self.say(f"{cid}: only {len(st.views)} views, need {MIN_VIEWS}")
                continue
            res = calib.calibrate_intrinsics(self.board, st.views, size)
            path = calib.save_intrinsics(cid, self.rig.ports[cid], res, size, self.board.spec,
                                         self.rig.fps, len(st.views))
            quality = "good" if res["rms"] < 0.5 else "OK" if res["rms"] < 1.0 else "POOR - recapture"
            st.result = f"rms {res['rms']:.2f} px ({quality}), {len(res['used'])} views"
            self.say(f"{cid}: {st.result} -> {path}", 10)

    def capture_extrinsics(self):
        seen = {c: st.det for c, st in self.state.items()
                if st.det is not None and len(st.det[1]) >= MIN_CORNERS}
        moving = [c for c in seen if self.state[c].still < STILL_TICKS]
        if moving:
            self.say(f"hold the board still ({', '.join(moving)} see it moving)")
            return
        if not self.captures and len(seen) < len(self.rig.ids):
            missing = [c for c in self.rig.ids if c not in seen]
            self.say(f"first capture defines the world frame and needs every camera "
                     f"(not seeing it: {', '.join(missing)})")
            return
        if len(seen) < 2:
            self.say("at least 2 cameras must see the board")
            return
        self.captures.append(seen)
        self.say(f"capture {len(self.captures)}: board seen by {', '.join(sorted(seen))}", 3)

    def compute_extrinsics(self):
        if not self.captures:
            self.say("no captures yet")
            return
        cams, boards = calib.initial_extrinsics(self.board, self.captures, self.intr)
        refined = False
        try:
            cams, boards = calib.bundle_adjust(self.board, self.captures, self.intr, cams, boards)
            refined = True
        except ImportError:
            self.say("scipy missing: saved the unrefined solution "
                     "(install: sudo apt install python3-scipy)", 10)
        cams, boards = calib.z_up(cams, boards)
        res = calib.residuals_by_camera(self.board, self.captures, self.intr, cams, boards)
        check = calib.board_check(self.board, self.captures, self.intr, cams)
        meta = {c: calib.load_intrinsics(c)[2] for c in cams}
        path = calib.save_extrinsics(cams, meta, res, check, self.rig.ports,
                                     len(self.captures), refined)
        self.summary = []
        for c in sorted(cams):
            r = res[c]
            rms = np.sqrt((r ** 2).sum(axis=1).mean()) if len(r) else float("nan")
            ctr = calib.geo.camera_center(cams[c])
            self.state[c].result = (f"rms {rms:.2f} px  at ({ctr[0]:.0f}, {ctr[1]:.0f}, "
                                    f"{ctr[2]:.0f}) mm")
        if check:
            self.summary.append(f"board check: neighbouring corners {check[0]:.3f} mm mean, "
                                f"{check[1]:.3f} mm max error ({check[2]} pairs)")
        not_solved = [c for c in self.rig.ids if c not in cams]
        if not_solved:
            self.summary.append(f"not solved (never seen with another camera): {', '.join(not_solved)}")
        self.say(f"saved {path}" + ("" if refined else " (unrefined)"), 10)
        for s in self.summary:
            print(s)

    def on_key(self, k):
        if k in (ord("q"), 27):
            return False
        if self.mode == "intrinsics":
            if k == ord("k"):
                self.compute_intrinsics()
            elif k == ord("u"):
                for st in self.state.values():
                    if st.views:
                        st.views.pop()
                self.say("removed the last view of each camera", 3)
        else:
            if k in (ord("c"), ord(" ")):
                self.capture_extrinsics()
            elif k == ord("k"):
                self.compute_extrinsics()
            elif k == ord("u") and self.captures:
                self.captures.pop()
                self.say(f"removed capture {len(self.captures) + 1}", 3)
        return True

    # ---- drawing ---------------------------------------------------------------
    def tile(self, i, cid, gray):
        tw, th = self.tw, self.th
        st = self.state[cid]
        img = np.zeros((th, tw, 3), np.uint8)
        if gray is not None:
            img[:] = cv2.cvtColor(cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_GRAY2BGR)
        if self.mode == "intrinsics" and st.coverage.any():
            cells = cv2.resize((st.coverage > 0).astype(np.uint8) * 255, (tw, th),
                               interpolation=cv2.INTER_NEAREST)
            green = img.copy()
            green[cells > 0] = (0, 160, 0)
            img = cv2.addWeighted(img, 0.7, green, 0.3, 0)
        if st.det is not None:
            color = (0, 255, 0) if st.still >= STILL_TICKS else (0, 220, 255)
            for x, y in st.det[0] / 2:
                cv2.circle(img, (int(x), int(y)), 3, color, -1)
        if time.monotonic() < st.flash_until:
            cv2.rectangle(img, (0, 0), (tw - 1, th - 1), (0, 255, 0), 8)
        text(img, f"{cid}  (port {self.rig.ports[cid]})", (10, 28), 0.8, COLORS[i % len(COLORS)], 2)
        n = 0 if st.det is None else len(st.det[1])
        state = "no board" if st.det is None else ("still" if st.still >= STILL_TICKS else "moving")
        if self.mode == "intrinsics":
            cov = (st.coverage > 0).mean()
            text(img, f"views {len(st.views)}/{TARGET_VIEWS} (min {MIN_VIEWS})   coverage {cov:.0%}",
                 (10, 54))
        else:
            err = "" if st.pose_err is None else f"   fit {st.pose_err:.2f} px"
            text(img, f"captures with this camera: "
                      f"{sum(cid in cap for cap in self.captures)}{err}", (10, 54))
        text(img, f"corners {n}/{self.board.n_corners}   {state}", (10, 78))
        if st.result:
            text(img, st.result, (10, 102), 0.55, (0, 255, 255))
        return img

    def help_panel(self, w):
        if self.mode == "intrinsics":
            lines = ["Move the board slowly through each camera's view: near/far, tilted, into the "
                     "corners. Views are captured automatically when the board is still.",
                     "Aim for the whole image green. k compute & save | u undo last view | q quit"]
        else:
            first = ("FIRST: lay the board flat in the centre of the cube (defines the world frame), "
                     "then press c." if not self.captures else
                     "Now place the board in other spots and tilts (2+ cameras must see it), c each time.")
            lines = [first, f"captures: {len(self.captures)}   c capture | k compute & save | "
                            "u undo last capture | q quit"]
        h = 70 + 20 * len(self.summary)
        panel = np.full((h, w, 3), 15, np.uint8)
        if time.monotonic() < self.message_until:
            text(panel, self.message, (10, 18), 0.5, (0, 255, 255))
        for j, line in enumerate(lines + self.summary):
            text(panel, line, (10, 40 + 20 * j), 0.45, (190, 190, 190))
        return panel

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        sized = False
        try:
            while True:
                t0 = time.monotonic()
                results = list(self.pool.map(self.detect, self.rig.ids))
                self.update(results)
                grid = np.zeros((2 * self.th, 2 * self.tw, 3), np.uint8)
                for i, (cid, gray, _) in enumerate(results[:4]):
                    r, c = divmod(i, 2)
                    grid[r * self.th:(r + 1) * self.th, c * self.tw:(c + 1) * self.tw] = \
                        self.tile(i, cid, gray)
                canvas = np.vstack([grid, self.help_panel(grid.shape[1])])
                cv2.imshow(WINDOW, canvas)
                if not sized:
                    cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
                    sized = True
                k = cv2.waitKey(max(1, int((TICK_S - (time.monotonic() - t0)) * 1000)))
                if k != -1 and not self.on_key(k & 0xFF):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                    break
        finally:
            cv2.destroyAllWindows()
            self.pool.shutdown()


def main():
    p = argparse.ArgumentParser(description="Guided ChArUco calibration.")
    p.add_argument("mode", choices=["intrinsics", "extrinsics"])
    p.add_argument("--config", default=config.DEFAULT_PATH)
    args = p.parse_args()
    cfg = config.load(args.config)
    cams, missing, _ = config.resolve(cfg)
    for name in missing:
        print(f"WARNING: {name} (port {cfg['cameras'][name]}) is not connected")
    if not cams:
        sys.exit("no cameras found")
    board = Board(cfg["board"])
    rig = Rig(cfg, cams)
    cal = Calibrator(rig, board, args.mode)
    with rig:
        cal.run()


if __name__ == "__main__":
    main()
