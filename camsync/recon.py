"""3D markers from 2D detections in calibrated cameras.

Proof-of-concept matching: every pair of detections from two cameras that lies on
corresponding epipolar lines is a candidate; each candidate is triangulated and
checked against the other cameras. Candidates seen by the most cameras (then lowest
reprojection error) are accepted first, each detection used once.

Cameras that face each other (horizontal view directions more than OPPOSITE_DEG apart,
world z up) only confirm
points: the robot's vertical line lies in their epipolar plane, so a marker matches every
other marker on that line equally well.
"""
from itertools import combinations

import numpy as np

from . import geometry as geo

EPIPOLAR_PX = 12.0  # max distance from the epipolar line, in pixels (calibration and sticker
REPROJ_PX = 8.0     # centroids differing between views give 2-10 px) and max reprojection error
OPPOSITE_DEG = 150.0


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Reconstructor:
    def __init__(self, cams, intr, epipolar_px=EPIPOLAR_PX, reproj_px=REPROJ_PX):
        """cams: {cam: T_cam_world}; intr: {cam: (K, dist)}."""
        self.ids = sorted(c for c in cams if c in intr)
        self.T = {c: cams[c] for c in self.ids}
        self.intr = {c: intr[c] for c in self.ids}
        self.epipolar_px, self.reproj_px = epipolar_px, reproj_px
        self.E = {}
        self.pairs = []  # camera pairs that propose candidates
        axis = {}  # horizontal part of each optical axis, in world coordinates
        for c in self.ids:
            v = self.T[c][2, :2]
            axis[c] = v / np.linalg.norm(v)
        for a, b in combinations(self.ids, 2):
            T_ba = self.T[b] @ geo.inv(self.T[a])  # camera a -> camera b
            self.E[a, b] = skew(T_ba[:3, 3]) @ T_ba[:3, :3]
            if axis[a] @ axis[b] > np.cos(np.radians(OPPOSITE_DEG)):
                self.pairs.append((a, b))

    def reconstruct(self, dets):
        """dets: {cam: (N,2) pixel centroids}. Returns a list of
        (point (3,), {cam: detection index}, mean reprojection error px)."""
        cams = [c for c in self.ids if c in dets and len(dets[c])]
        norm = {c: geo.undistort(dets[c], *self.intr[c]) for c in cams}
        candidates = []
        for a, b in self.pairs:
            if a not in norm or b not in norm:
                continue
            ha = np.c_[norm[a], np.ones(len(norm[a]))]
            hb = np.c_[norm[b], np.ones(len(norm[b]))]
            lines = ha @ self.E[a, b].T                  # epipolar lines in b, one per point in a
            d = np.abs(lines @ hb.T) / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
            d *= self.intr[b][0][0, 0]                     # normalized units -> pixels
            for i, j in zip(*np.nonzero(d < self.epipolar_px)):
                candidates.append({a: i, b: j})
        scored = []
        for views in candidates:
            X = self._triangulate(views, norm)
            if X is None:
                continue
            views = self._support(X, dets, cams, views)
            X = self._triangulate(views, norm)
            if X is None:
                continue
            err = self._error(X, views, dets)
            if err <= self.reproj_px:
                scored.append((-len(views), err, X, views))
        scored.sort(key=lambda s: (s[0], s[1]))
        used, points = set(), []
        for _, err, X, views in scored:
            if any((c, i) in used for c, i in views.items()):
                continue
            used.update(views.items())
            points.append((X, views, err))
        return points

    def _triangulate(self, views, norm):
        X = geo.triangulate([(self.T[c], norm[c][i]) for c, i in views.items()])
        for c in views:  # must be in front of every camera that sees it
            if (self.T[c] @ np.r_[X, 1])[2] <= 0:
                return None
        return X

    def _support(self, X, dets, cams, views):
        views = dict(views)
        for c in cams:
            if c in views:
                continue
            px = geo.project(X[None], self.T[c], *self.intr[c])[0]
            d = np.linalg.norm(dets[c] - px, axis=1)
            i = int(d.argmin())
            if d[i] < self.reproj_px:
                views[c] = i
        return views

    def _error(self, X, views, dets):
        errs = [np.linalg.norm(geo.project(X[None], self.T[c], *self.intr[c])[0] - dets[c][i])
                for c, i in views.items()]
        return float(np.mean(errs))
