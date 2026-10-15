"""Drawing helpers shared by the rig tools (OpenCV, no extra dependencies)."""
import time

import cv2
import numpy as np

COLORS = [(80, 200, 255), (255, 170, 60), (110, 220, 110), (210, 120, 255)]  # BGR per camera


def text(img, s, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def fmt_span(seconds):
    return f"-{seconds // 60} min" if seconds >= 120 else f"-{seconds} s"


def plot(area, title, series, lo, hi, ticks, span_s, scale=lambda v: v, now=None):
    """Draw time series into `area` (a BGR image slice).

    series: list of (color, [(second, value)]) with seconds on the time.monotonic() scale;
    the x axis covers the last span_s seconds.
    """
    h, w = area.shape[:2]
    x0, x1, y0, y1 = 48, w - 12, 26, h - 20
    cv2.rectangle(area, (x0, y0), (x1, y1), (70, 70, 70), 1)
    text(area, title, (x0, 18), 0.5, (200, 200, 200))
    s_lo, s_hi = scale(lo), scale(hi)

    def ypos(v):
        v = min(max(v, lo), hi)
        return int(y1 - (scale(v) - s_lo) / (s_hi - s_lo) * (y1 - y0))

    for t in ticks:
        y = ypos(t)
        cv2.line(area, (x0, y), (x1, y), (50, 50, 50), 1)
        text(area, f"{t:g}", (4, y + 5), 0.4, (170, 170, 170))
    text(area, fmt_span(span_s), (x0, h - 4), 0.4, (170, 170, 170))
    text(area, "now", (x1 - 30, h - 4), 0.4, (170, 170, 170))
    now_sec = int(time.monotonic() if now is None else now)
    for color, pts in series:
        if len(pts) < 2:
            continue
        xy = np.array([(x1 - (now_sec - 1 - sec) * (x1 - x0) / span_s, ypos(v))
                       for sec, v in pts if now_sec - 1 - sec <= span_s], np.int32)
        if len(xy) >= 2:
            cv2.polylines(area, [xy], False, color, 1 if span_s <= 600 else 1, cv2.LINE_AA)


def stats_graphs(rig, w, h, span_s, ids=None):
    """Frame rate and per-second frame loss for every camera, side by side."""
    ids = rig.ids if ids is None else ids
    g = np.full((h, w, 3), 25, np.uint8)
    half = w // 2
    fps_series, loss_series = [], []
    for i, cid in enumerate(ids):
        secs = rig.stats[cid].seconds(span_s + 1)
        color = COLORS[i % len(COLORS)]
        fps_series.append((color, [(s, f) for s, f, l in secs]))
        loss_series.append((color, [(s, 100.0 * l / (f + l) if f + l else 100.0) for s, f, l in secs]))
    top = max(125, rig.fps * 1.05)
    ticks = [0, 30, 60, 90, 120] if rig.fps > 60 else [0, 15, 30, 45, 60]
    plot(g[:, :half], "frame rate (fps)", fps_series, 0, top if rig.fps > 60 else 65, ticks, span_s)
    plot(g[:, half:], "frame loss % per second (log scale)", loss_series, 0, 100, [0, 1, 10, 100],
         span_s, scale=lambda v: np.log10(1 + v))
    x = half - 12
    for i, cid in reversed(list(enumerate(ids))):
        x -= 10 * len(cid) + 18
        text(g, cid, (x, 18), 0.5, COLORS[i % len(COLORS)], 1)
    return g
