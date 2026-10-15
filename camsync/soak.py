"""Long stability test without video: python3 -m camsync.soak [--minutes 10] [--graph]

Prints frame rate and loss per camera at a fixed interval, logs every second to
logs/soak-*.csv, and ends with frames lost per minute (heat problems show up as
loss that grows over time). --graph opens a live window with frame rate and loss
graphs for the whole run (q stops early); the final graph is saved next to the log.
Ctrl+C stops early and still prints the summary.
"""
import argparse
import time

import numpy as np

from . import config
from .plots import COLORS, stats_graphs, text
from .rig import Rig
from .stats import CsvLogger

WINDOW = "soak test"


def graph_window(rig, start, minutes, w=1280):
    """Header with live numbers per camera, then fps and loss graphs over the whole run."""
    now = time.monotonic()
    el, total = int(now - start), int(minutes * 60)
    header = np.full((40 + 26 * len(rig.ids), w, 3), 15, np.uint8)
    text(header, f"soak test {el // 60:02d}:{el % 60:02d} / {total // 60:02d}:{total % 60:02d}"
                 f"   {rig.fps} fps target   q = stop early", (10, 26), 0.6, (220, 220, 220))
    for i, cid in enumerate(rig.ids):
        st = rig.stats[cid]
        fps, _ = st.window(3)
        _, loss10 = st.window(10)
        status = "OK" if loss10 == 0 else "LOSING FRAMES"
        color = (0, 220, 0) if loss10 == 0 else (0, 0, 255) if loss10 > 0.01 else (0, 165, 255)
        y = 56 + 26 * i
        text(header, f"{cid:>8}  port {rig.ports[cid]:<6}", (10, y), 0.55, COLORS[i % len(COLORS)])
        text(header, f"{fps:6.1f} fps   loss 10s {loss10:7.2%}   total {st.total_loss:7.2%}"
                     f" ({st.total_lost} frames)", (260, y), 0.55)
        text(header, status, (900, y), 0.55, color)
    span = max(60, min(total, 3600))
    return np.vstack([header, stats_graphs(rig, w, 420, span)])


def main():
    p = argparse.ArgumentParser(description="Long-running stability test for the camera rig.")
    p.add_argument("--config", default=config.DEFAULT_PATH)
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--every", type=float, default=10, help="seconds between status lines")
    p.add_argument("--fps", type=int, choices=config.FPS_CHOICES, help="override rig.json frame rate")
    p.add_argument("--graph", action="store_true", help="show live frame rate and loss graphs")
    args = p.parse_args()
    if args.graph:
        import cv2
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    cfg = config.load(args.config)
    if args.fps:
        config.set_fps(cfg, args.fps)
    rig = Rig(cfg, queue=1024)  # keep every set; it is counted, not displayed
    logger = CsvLogger("soak")
    last_set, max_offset = None, 0.0
    print(f"soak test: {len(rig.ids)} cameras ({', '.join(f'{c}=port {rig.ports[c]}' for c in rig.ids)}), "
          f"{rig.fps} fps, {args.minutes:g} min, log {logger.path}")
    with rig:
        # Count from the moment every clock has locked, so startup doesn't skew the numbers.
        deadline = time.monotonic() + 5
        while not rig.synced and time.monotonic() < deadline:
            rig.poll_sets()
            time.sleep(0.1)
        for st in rig.stats.values():
            st.reset()
        start = time.monotonic()
        next_print, next_log = start + args.every, start + 1
        canvas = None
        try:
            while time.monotonic() - start < args.minutes * 60:
                if args.graph:
                    canvas = graph_window(rig, start, args.minutes)
                    cv2.imshow(WINDOW, canvas)
                    k = cv2.waitKey(200) & 0xFF
                    if k in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                        print("stopped early")
                        break
                else:
                    time.sleep(0.2)
                now = time.monotonic()
                for st in rig.stats.values():
                    st.tick(now)
                for s in rig.poll_sets():
                    if s.complete:
                        last_set = s
                        max_offset = max(max_offset, *(abs(s.offset(c)) for c in rig.ids))
                while now >= next_log:
                    offs = {c: last_set.offset(c) for c in rig.ids} if last_set else {}
                    logger.log(rig, offs)
                    next_log += 1
                if now >= next_print:
                    k = int(args.every)
                    parts = []
                    for cid in rig.ids:
                        fps, loss = rig.stats[cid].window(k)
                        parts.append(f"{cid} {fps:5.1f} fps {loss:6.2%}")
                    el = int(now - start)
                    print(f"[{el // 60:02d}:{el % 60:02d}] " + " | ".join(parts), flush=True)
                    next_print += args.every
        except KeyboardInterrupt:
            print("stopped early")
        elapsed = time.monotonic() - start
        if args.graph:
            canvas = graph_window(rig, start, args.minutes)
    logger.close()
    if args.graph:
        png = logger.path[:-4] + ".png"
        cv2.imwrite(png, canvas)
        cv2.destroyAllWindows()
        print(f"graph saved: {png}")

    print(f"\n== summary ({elapsed / 60:.1f} min)")
    for cid in rig.ids:
        st = rig.stats[cid]
        secs = st.seconds(st.history)
        per_min = {}
        if secs:
            first = secs[0][0]
            for sec, _, lost in secs:
                per_min[(sec - first) // 60] = per_min.get((sec - first) // 60, 0) + lost
        verdict = "OK" if st.total_lost == 0 else "LOSING FRAMES"
        print(f"{cid} (port {rig.ports[cid]}): {st.total_frames} frames, lost {st.total_lost} "
              f"({st.total_loss:.2%}) {verdict}")
        print(f"    lost per minute: {[per_min[m] for m in sorted(per_min)]}")
    if last_set:
        print(f"largest nearest-frame offset between cameras: {max_offset * 1e3:.2f} ms")
    print(f"log: {logger.path}")


if __name__ == "__main__":
    main()
