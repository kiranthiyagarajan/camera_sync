"""2D detection of reflective markers: compact blobs brighter than their surroundings.

The markers are only a few times brighter than the white PLA disks they sit on, so
blobs are found by local contrast (image minus its local mean), not by one fixed level.
Centroids are intensity weighted (sub-pixel). Pixels in the static reflection mask
(mask.py) are ignored.
"""
import cv2
import numpy as np

CONTRAST = 35       # blob pixels are this much brighter than the local mean
FLOOR = 60          # and at least this bright
LOCAL_PX = 41       # size of the local-mean window (about 2 marker diameters)
MIN_AREA = 150      # px; markers are ~200-800 px at 0.4-0.6 m; smaller = edges, noise
MAX_AREA = 3000     # px; larger is a reflection, not a marker
MIN_FILL = 0.5      # blob area / bounding-box area: rejects streaks and edges
MAX_ASPECT = 2.0


def detect(gray, mask=None, contrast=CONTRAST, floor=FLOOR, min_area=MIN_AREA,
           max_area=MAX_AREA):
    """Returns (centroids (N,2) in pixels, areas (N,))."""
    smooth = cv2.GaussianBlur(gray, (7, 7), 0)
    local = cv2.blur(smooth, (LOCAL_PX, LOCAL_PX))
    excess = cv2.subtract(smooth, local)
    bw = ((excess >= contrast) & (smooth >= floor)).astype(np.uint8)
    if mask is not None:
        bw[mask] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    pts, areas = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_area <= area <= max_area):
            continue
        if area / float(w * h) < MIN_FILL or max(w, h) / float(min(w, h)) > MAX_ASPECT:
            continue
        win = excess[y:y + h, x:x + w].astype(np.float64)
        win[labels[y:y + h, x:x + w] != i] = 0
        total = win.sum()
        ys, xs = np.mgrid[y:y + h, x:x + w]
        pts.append(((win * xs).sum() / total, (win * ys).sum() / total))
        areas.append(area)
    return np.array(pts, float).reshape(-1, 2), np.array(areas, int)
