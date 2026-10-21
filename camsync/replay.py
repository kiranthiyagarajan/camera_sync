"""Read recordings made by camsync.record: frames by time, with marker detections."""
import csv
import os

import numpy as np

from . import mask, markers
from .imaging import decode


class Recording:
    def __init__(self, path, cams):
        self.path = path
        self.idx, self.files = {}, {}
        for c in cams:
            rows = list(csv.DictReader(open(os.path.join(path, f"{c}.csv"))))
            self.idx[c] = (np.array([float(r["t"]) for r in rows]),
                           np.array([[int(r["offset"]), int(r["size"])] for r in rows]))
            self.files[c] = open(os.path.join(path, f"{c}.mjpeg"), "rb")
        self.masks = {c: mask.load(c, (720, 1280)) for c in cams}
        self.t0 = min(ts[0] for ts, _ in self.idx.values())

    def detections(self, t):
        out = {}
        for c, (ts, off) in self.idx.items():
            i = int(np.argmin(abs(ts - t)))
            if abs(ts[i] - t) > 0.02:
                continue
            f = self.files[c]
            f.seek(off[i, 0])
            out[c] = markers.detect(decode(f.read(off[i, 1])), self.masks[c])[0]
        return out

    def frames(self, t_from, t_to, n):
        return [self.detections(t) for t in np.linspace(t_from, t_to, n)]
