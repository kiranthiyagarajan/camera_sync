"""Live rig dashboard: python3 -m camsync.rigview

All cameras in a grid with live frame rate, frame loss, exposure/gain and sync
offset; frame-rate and loss graphs over the last 5 minutes; exposure assistant,
clipped-pixel overlay, histograms, focus aid and synchronized calibration capture.
Statistics are logged to logs/rigview-*.csv every second.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config
from .config import CAPTURE_DIR, max_exposure
from .imaging import (AutoExposure, decode, downscale2, focus_score, histogram_image,
                      saturation_mask2)
from .plots import COLORS, stats_graphs, text
from .rig import Rig
from .stats import CsvLogger

WINDOW = "rigview"
GRAPH_H, HELP_H = 210, 66
HISTORY_S = 300
SETTLE_S = 0.15  # wait after changing exposure before judging a frame
HELP = [
    "1-4 select camera, 0 all | e/E exposure -/+ | g/G gain -/+ | "
    "m auto exposure (markers) | b auto exposure (calibration board)",
    "o clipped pixels | h histogram | f focus aid | c capture synced set | "
    "s save settings | r reset stats | q quit",
]


class RigView:
    def __init__(self, rig, cfg_path):
        self.rig = rig
        self.cfg_path = cfg_path
        self.tw, self.th = rig.width // 2, rig.height // 2
        self.sel = None  # selected camera, None = all
        self.show_clip = self.show_hist = self.show_focus = False
        self.auto = {}  # cam -> [AutoExposure, settle_until]
        self.focus_peak = {}
        self.last_set = None
        self.message, self.message_until = "", 0.0
        self.pool = ThreadPoolExecutor(max(1, len(rig.ids)))
        self.capture_dir = None
        self.capture_count = 0
        self.logger = CsvLogger("rigview")

    # ---- helpers -------------------------------------------------------------
    def selected(self):
        return list(self.rig.ids) if self.sel is None else [self.sel]

    def say(self, msg, secs=5):
        self.message, self.message_until = msg, time.monotonic() + secs
        print(msg)

    def offsets(self):
        s = self.last_set
        if s is None:
            return {}
        return {c: s.offset(c) for c in self.rig.ids if s.frames.get(c) is not None}

    def decode_latest(self, cid):
        f = self.rig.latest.get(cid)
        return cid, f, (decode(f.data) if f is not None else None)

    # ---- actions -------------------------------------------------------------
    def nudge(self, name, up):
        for cid in self.selected():
            self.auto.pop(cid, None)
            cur = self.rig.current[cid].get(name, 0)
            if name == "gain":
                new = min(63, cur + 4) if up else max(0, cur - 4)
            else:
                new = max(cur + 1, round(cur * 1.25)) if up else min(cur - 1, round(cur * 0.8))
                cap = max_exposure(self.rig.fps)
                if new > cap:
                    self.say(f"exposure is capped at {cap / 10} ms to keep {self.rig.fps} fps")
                new = max(1, min(cap, new))
            self.rig.set_control(cid, name, new)
        cid = self.selected()[0]
        ctl = self.rig.current[cid]
        who = "all cameras" if self.sel is None else cid
        self.say(f"{who}: exposure {ctl['exposure_time_absolute'] / 10:.1f} ms, gain {ctl['gain']}", 3)

    def start_auto(self, mode):
        for cid in self.selected():
            ctl = self.rig.current[cid]
            self.auto[cid] = [AutoExposure(mode, ctl["exposure_time_absolute"], ctl["gain"],
                                           fps=self.rig.fps), 0.0]
        hint = ("keep markers in view of every camera" if mode == "marker"
                else "hold the calibration board in view")
        self.say(f"auto exposure ({mode}) started: {hint}", 6)

    def run_auto(self, cid, frame, gray):
        ae, settle_until = self.auto[cid]
        if frame.t_raw < settle_until:  # frame was exposed before the last change settled
            return
        res = ae.step(gray)
        if res is None:
            del self.auto[cid]
            self.say(f"{cid}: auto exposure ({ae.mode}) {ae.message} -> exposure "
                     f"{ae.exposure / 10:.1f} ms, gain {ae.gain}" + ("" if ae.ok else "  [CHECK]"), 10)
            return
        exposure, gain = res
        self.rig.set_control(cid, "exposure_time_absolute", exposure)
        self.rig.set_control(cid, "gain", gain)
        self.auto[cid][1] = time.monotonic() + SETTLE_S

    def capture(self):
        s = self.last_set
        if s is None or not s.complete:
            self.say("no complete synchronized set yet")
            return
        if self.capture_dir is None:
            self.capture_dir = os.path.join(CAPTURE_DIR, time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(self.capture_dir)
        self.capture_count += 1
        idx = self.capture_count
        cams = list(self.rig.ids)
        grays = list(self.pool.map(lambda c: decode(s.frames[c].data), cams))
        for cid, gray in zip(cams, grays):
            cv2.imwrite(os.path.join(self.capture_dir, f"{idx:04d}_{cid}.png"), gray)
        meta = os.path.join(self.capture_dir, "sets.csv")
        new = not os.path.exists(meta)
        with open(meta, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["set", "camera", "port", "frame_time", "offset_ms", "exposure_ms", "gain"])
            for cid in cams:
                ctl = self.rig.current[cid]
                w.writerow([idx, cid, self.rig.ports[cid], f"{s.frames[cid].t:.6f}",
                            f"{s.offset(cid) * 1e3:.3f}",
                            ctl["exposure_time_absolute"] / 10, ctl["gain"]])
        spread = max(abs(s.offset(c)) for c in cams) * 1e3
        self.say(f"captured set {idx} -> {self.capture_dir} (frames within {spread:.1f} ms; "
                 "hold the board still)")

    def save(self):
        cfg = config.load(self.cfg_path)  # re-read so names assigned elsewhere are kept
        for cid in self.rig.ids:
            ctl = self.rig.current[cid]
            cfg["controls"].setdefault(cid, {}).update(
                exposure_time_absolute=ctl["exposure_time_absolute"], gain=ctl["gain"])
        config.save(cfg, self.cfg_path)
        unnamed = [c for c in self.rig.ids if c not in cfg["cameras"]]
        note = f" (unnamed cameras saved by port: {', '.join(unnamed)})" if unnamed else ""
        self.say(f"settings saved to {self.cfg_path}{note}")

    def on_key(self, k):
        ids = self.rig.ids
        if k in (ord("q"), 27):
            return False
        if ord("1") <= k <= ord("9") and k - ord("1") < len(ids):
            self.sel = ids[k - ord("1")]
            self.say(f"selected {self.sel}", 2)
        elif k in (ord("0"), ord("a")):
            self.sel = None
            self.say("selected all cameras", 2)
        elif k in (ord("e"), ord("E")):
            self.nudge("exposure_time_absolute", k == ord("E"))
        elif k in (ord("g"), ord("G")):
            self.nudge("gain", k == ord("G"))
        elif k == ord("m"):
            self.start_auto("marker")
        elif k == ord("b"):
            self.start_auto("board")
        elif k == ord("o"):
            self.show_clip = not self.show_clip
        elif k == ord("h"):
            self.show_hist = not self.show_hist
        elif k == ord("f"):
            self.show_focus = not self.show_focus
            self.focus_peak.clear()
            if self.show_focus:
                self.say("focus aid: turn the focus ring until the score peaks", 6)
        elif k in (ord("c"), ord(" ")):
            self.capture()
        elif k == ord("s"):
            self.save()
        elif k == ord("r"):
            for st in self.rig.stats.values():
                st.reset()
            self.say("statistics reset", 2)
        return True

    # ---- drawing -------------------------------------------------------------
    def render_tile(self, i, cid, frame, gray):
        tw, th = self.tw, self.th
        tile = np.zeros((th, tw, 3), np.uint8)
        color = COLORS[i % len(COLORS)]
        now = time.monotonic()
        stale = True
        if gray is not None:
            if cid in self.auto:
                self.run_auto(cid, frame, gray)
            tile[:] = cv2.cvtColor(downscale2(gray)[:th, :tw], cv2.COLOR_GRAY2BGR)
            if self.show_clip:
                tile[saturation_mask2(gray)[:th, :tw]] = (0, 0, 255)
            if self.show_hist:
                hist = histogram_image(gray)
                hh, hw = hist.shape[:2]
                tile[th - hh - 6:th - 6, tw - hw - 6:tw - 6] = hist
            stale = now - frame.t_raw > 0.5

        st = self.rig.stats[cid]
        fps, _ = st.window(3)
        _, loss10 = st.window(10)
        ctl = self.rig.current[cid]
        text(tile, f"{cid}  (port {self.rig.ports[cid]})", (10, 28), 0.8, color, 2)
        text(tile, f"{fps:5.1f} fps   loss 10s {loss10:6.2%}   total {st.total_loss:6.2%}", (10, 54))
        auto = f"   AUTO {self.auto[cid][0].mode}" if cid in self.auto else ""
        text(tile, f"exposure {ctl['exposure_time_absolute'] / 10:.1f} ms   gain {ctl['gain']}{auto}",
             (10, 78))
        offs = self.offsets()
        if not self.rig.synced:
            sync = "sync: locking clocks..."
        elif cid == self.rig.ids[0]:
            sync = "sync: reference"
        elif cid in offs:
            sync = f"sync offset {offs[cid] * 1e3:+.2f} ms"
        else:
            sync = "sync: missing from last set"
        text(tile, sync, (10, 102))
        if self.show_focus and gray is not None:
            score = focus_score(gray)
            peak = max(self.focus_peak.get(cid, 0.0), score)
            self.focus_peak[cid] = peak
            text(tile, f"focus {score:7.1f}   peak {peak:7.1f}", (10, 126), 0.55, (0, 255, 255))
            bar = int((tw - 20) * score / peak) if peak > 0 else 0
            cv2.rectangle(tile, (10, 134), (tw - 10, 144), (60, 60, 60), -1)
            cv2.rectangle(tile, (10, 134), (10 + bar, 144),
                          (0, 255, 0) if score >= 0.95 * peak else (0, 200, 255), -1)
        if stale:
            text(tile, "NO SIGNAL", (tw // 2 - 110, th // 2), 1.4, (0, 0, 255), 3)
        if loss10 > 0.01 or stale:
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (0, 0, 255), 4)
        elif loss10 > 0:
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (0, 165, 255), 3)
        if self.sel == cid:
            cv2.rectangle(tile, (5, 5), (tw - 6, th - 6), (0, 255, 255), 2)
        return tile

    def draw_graphs(self, w):
        return stats_graphs(self.rig, w, GRAPH_H, HISTORY_S)

    def draw_help(self, w):
        panel = np.full((HELP_H, w, 3), 15, np.uint8)
        if time.monotonic() < self.message_until:
            text(panel, self.message, (10, 18), 0.5, (0, 255, 255))
        for i, line in enumerate(HELP):
            text(panel, line, (10, 40 + 20 * i), 0.45, (190, 190, 190))
        return panel

    def render(self):
        results = list(self.pool.map(self.decode_latest, self.rig.ids))
        cols = 2
        rows = max(2, (len(results) + 1) // 2)
        grid = np.zeros((rows * self.th, cols * self.tw, 3), np.uint8)
        for i, (cid, frame, gray) in enumerate(results):
            r, c = divmod(i, cols)
            grid[r * self.th:(r + 1) * self.th, c * self.tw:(c + 1) * self.tw] = \
                self.render_tile(i, cid, frame, gray)
        w = grid.shape[1]
        return np.vstack([grid, self.draw_graphs(w), self.draw_help(w)])

    def run(self, duration=None, snapshot=None, keys=""):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        start = time.monotonic()
        next_log = start + 1
        sized = False
        try:
            while True:
                t0 = time.monotonic()
                for st in self.rig.stats.values():
                    st.tick(t0)
                for s in self.rig.poll_sets():
                    if s.complete:
                        self.last_set = s
                canvas = self.render()
                cv2.imshow(WINDOW, canvas)
                if not sized:
                    cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
                    sized = True
                while t0 >= next_log:
                    self.logger.log(self.rig, self.offsets())
                    next_log += 1
                if keys and self.rig.synced:  # scripted key presses (testing)
                    if not self.on_key(ord(keys[0])):
                        break
                    keys = keys[1:]
                if duration and t0 - start >= duration:
                    if snapshot:
                        cv2.imwrite(snapshot, canvas)
                    break
                k = cv2.waitKey(max(1, int((1 / 30 - (time.monotonic() - t0)) * 1000)))
                if k != -1 and not self.on_key(k & 0xFF):
                    break
                # WND_PROP_VISIBLE is unsupported on this GTK build; any property reads
                # -1 once the window has been closed.
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                    break
        finally:
            self.logger.close()
            cv2.destroyAllWindows()
            self.pool.shutdown()
            print(f"statistics log: {self.logger.path}")


def main():
    p = argparse.ArgumentParser(description="Live dashboard for the camera rig.")
    p.add_argument("--config", default=config.DEFAULT_PATH)
    p.add_argument("--fps", type=int, choices=config.FPS_CHOICES, help="override rig.json frame rate")
    p.add_argument("--duration", type=float, help="quit after this many seconds")
    p.add_argument("--snapshot", help="with --duration: save the final dashboard image to this file")
    p.add_argument("--keys", default="", help=argparse.SUPPRESS)  # scripted key presses for testing
    args = p.parse_args()

    cfg = config.load(args.config)
    if args.fps:
        config.set_fps(cfg, args.fps)
    cams, missing, _ = config.resolve(cfg)
    for name in missing:
        print(f"WARNING: {name} (port {cfg['cameras'][name]}) is not connected")
    if not cams:
        sys.exit("no cameras found")
    rig = Rig(cfg, cams)
    for cid, bad in rig.mismatched.items():
        print(f"WARNING: {cid} did not accept: {bad}")
    with rig:
        RigView(rig, args.config).run(args.duration, args.snapshot, args.keys)


if __name__ == "__main__":
    main()
