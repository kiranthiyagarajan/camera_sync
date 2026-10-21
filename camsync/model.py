"""Robot model fitted directly to 2D marker detections in all cameras.

The robot: a base, then 4 segments of constant curvature, each ending in a disk with 4
markers on its rim, 90 deg apart. Shape = curvature vector (kx, ky) per segment, in the
segment's own frame (8 numbers). Fixed per robot (fitted once from home poses): base pose,
each disk's rotation about the backbone (psi), segment lengths, rim radius.

Fitting needs no 3D matching: the model's markers that face a camera are projected into
it, each detection is assigned to the nearest projected marker (Hungarian, gated), and
the shape is adjusted to minimise the pixel distances; repeat until assignments settle.
A marker seen by one camera still counts, and pooling frames of a held pose fills in
markers that flicker.
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from . import geometry as geo
from .config import CALIB_DIR

N_DISKS = 4
PER_DISK = 4
RIM_MM = 25.0  # disk diameter 50 mm
SPACING_MM = 70.0
LENGTHS_MM = (60.0, 56.6, 56.8, 62.0)  # base -> disk 1 (fitted), then disk to disk (measured)
VISIBLE_DEG = 75.0  # a flat sticker is detected up to about this far off its normal
GATE_PX = 25.0      # max distance between a detection and the marker assigned to it
MISS_WEIGHT = 0.3   # score of a predicted marker with no detection, x GATE_PX (dim markers
                    # often go undetected; a detection with no marker scores a full GATE_PX)
LOSS_PX = 3.0       # robust loss scale
PATH = os.path.join(CALIB_DIR, "robot_model.json")


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def segment(k, length):
    """4x4 transform along a constant-curvature segment: tangent z, curvature (kx, ky)
    (1/mm) as the body angular rate about local x and y."""
    w = np.array([k[0], k[1], 0.0]) * length
    u = np.array([0.0, 0.0, length])
    th = np.linalg.norm(w)
    W = _skew(w)
    if th < 1e-9:
        R, V = np.eye(3) + W, np.eye(3) + W / 2
    else:
        R = np.eye(3) + np.sin(th) / th * W + (1 - np.cos(th)) / th ** 2 * W @ W
        V = np.eye(3) + (1 - np.cos(th)) / th ** 2 * W + (th - np.sin(th)) / th ** 3 * W @ W
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, V @ u
    return T


class Model:
    def __init__(self, base=None, psi=None, lengths=None, rim=RIM_MM):
        self.base = np.eye(4) if base is None else np.asarray(base, float)  # world <- base
        self.psi = np.zeros(N_DISKS) if psi is None else np.asarray(psi, float)
        self.lengths = np.array(LENGTHS_MM) if lengths is None else np.asarray(lengths, float)
        self.rim = rim

    def disks(self, curv, base=None):
        """World 4x4 frame of each disk (z = backbone tangent), base to tip."""
        T = self.base if base is None else base
        out = []
        for k, L in zip(np.reshape(curv, (N_DISKS, 2)), self.lengths):
            T = T @ segment(k, L)
            out.append(T)
        return out

    def markers(self, curv, base=None, psi=None):
        """(16, 3) marker positions and outward normals in world coordinates."""
        psi = self.psi if psi is None else psi
        P, N = [], []
        for T, p in zip(self.disks(curv, base), psi):
            for j in range(PER_DISK):
                a = p + j * np.pi / 2
                n = T[:3, :3] @ np.array([np.cos(a), np.sin(a), 0.0])
                P.append(T[:3, 3] + self.rim * n)
                N.append(n)
        return np.array(P), np.array(N)

    def save(self, path=PATH):
        with open(path, "w") as f:
            json.dump({"base": self.base.tolist(), "psi_deg": np.degrees(self.psi).tolist(),
                       "lengths_mm": self.lengths.tolist(), "rim_mm": self.rim}, f, indent=2)

    @classmethod
    def load(cls, path=PATH):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        return cls(d["base"], np.radians(d["psi_deg"]), d["lengths_mm"], d["rim_mm"])


def project_visible(P, N, cams, intr):
    """{cam: (marker indices, (n,2) pixels)} for markers facing each camera."""
    out = {}
    cos_min = np.cos(np.radians(VISIBLE_DEG))
    for c, T in cams.items():
        C = geo.camera_center(T)
        view = C - P
        vis = np.einsum("ij,ij->i", N, view) / np.linalg.norm(view, axis=1) > cos_min
        vis &= (T[:3, :3] @ P.T + T[:3, 3:])[2] > 0
        idx = np.nonzero(vis)[0]
        px = geo.project(P[idx], T, *intr[c]) if len(idx) else np.zeros((0, 2))
        out[c] = (idx, px)
    return out


def associate(pred, frames, gate=GATE_PX):
    """pred: {cam: (idx, px)}; frames: list of {cam: (n,2) detections}. Returns a list of
    (cam, marker index, detection xy) and the matching score: distance per match, gate per
    unmatched detection, MISS_WEIGHT * gate per unmatched predicted marker."""
    pairs, score = [], 0.0
    for dets in frames:
        for c, (idx, px) in pred.items():
            d = dets.get(c)
            if d is None or not len(d) or not len(idx):
                score += gate * ((0 if d is None else len(d)) + MISS_WEIGHT * len(idx))
                continue
            D = np.linalg.norm(px[:, None] - d[None], axis=2)
            r, k = linear_sum_assignment(np.minimum(D, gate))
            ok = D[r, k] < gate
            n = ok.sum()
            score += D[r, k][ok].sum() + gate * (len(d) - n + MISS_WEIGHT * (len(idx) - n))
            pairs += [(c, idx[i], d[j]) for i, j in zip(r[ok], k[ok])]
    return pairs, score


def _residuals(model, curv, pairs, cams, intr, base=None, psi=None):
    P, _ = model.markers(curv, base, psi)
    res = []
    for c in cams:
        sel = [(m, xy) for cc, m, xy in pairs if cc == c]
        if sel:
            px = geo.project(P[[m for m, _ in sel]], cams[c], *intr[c])
            res.append((px - np.array([xy for _, xy in sel])).ravel())
    return np.concatenate(res) if res else np.zeros(0)


def score(model, curv, frames, cams, intr):
    P, N = model.markers(curv)
    return associate(project_visible(P, N, cams, intr), frames)[1] / max(1, len(frames))


def fit_shape(model, frames, cams, intr, init, rounds=5):
    """Curvatures (8,) for one held pose from frames = [{cam: detections}]. Returns
    (curv, rms px, matched detections per frame)."""
    curv = np.asarray(init, float).copy()
    pairs = []
    for _ in range(rounds):
        P, N = model.markers(curv)
        new, _ = associate(project_visible(P, N, cams, intr), frames)
        if len(new) < 6:
            break
        same = len(new) == len(pairs) and all(a[:2] == b[:2] for a, b in zip(new, pairs))
        pairs = new
        if same:
            break
        r = least_squares(lambda x: _residuals(model, x, pairs, cams, intr), curv,
                          loss="soft_l1", f_scale=LOSS_PX, x_scale=1e-3)
        curv = r.x
    res = _residuals(model, curv, pairs, cams, intr)
    rms = float(np.sqrt(np.mean(res ** 2))) if len(res) else np.inf
    return curv, rms, len(pairs) / max(1, len(frames))


def candidates(prev=None, mags=(0.0, 0.002, 0.004, 0.007), dirs=12):
    """Starting shapes: the previous fit and uniform bends in several directions."""
    out = [] if prev is None else [np.asarray(prev, float)]
    for m in mags:
        for a in (np.arange(dirs) * 2 * np.pi / dirs if m else [0.0]):
            out.append(np.tile([m * np.cos(a), m * np.sin(a)], N_DISKS))
    return out


def fit_best(model, frames, cams, intr, prev=None, keep=3):
    """Try all starting shapes, refine the best few, return the best fit."""
    starts = sorted(candidates(prev), key=lambda c: score(model, c, frames, cams, intr))[:keep]
    fits = [fit_shape(model, frames, cams, intr, s) for s in starts]
    return min(fits, key=lambda f: score(model, f[0], frames, cams, intr))


def fit_model(frames_per_hold, cams, intr, model, rounds=4):
    """Fit the fixed parameters (base pose, disk rotations) together with each hold's
    shape, from a few held poses (home poses are best). Returns the model and shapes."""
    shapes = [np.zeros(8) for _ in frames_per_hold]
    for r in range(rounds):
        free_shape = r > 0  # first place the base with the robot straight
        pairs = []
        for f, s in zip(frames_per_hold, shapes):
            P, N = model.markers(s)
            pairs.append(associate(project_visible(P, N, cams, intr), f)[0])
        rv, tv = geo.from_T(model.base)
        x0 = np.concatenate([rv, tv, model.psi] + list(shapes))

        def fun(x):
            base = geo.to_T(x[:3], x[3:6])
            psi = x[6:6 + N_DISKS]
            out = []
            for h, p in enumerate(pairs):
                s = x[6 + N_DISKS + 8 * h:6 + N_DISKS + 8 * (h + 1)] if free_shape else shapes[h]
                out.append(_residuals(model, s, p, cams, intr, base, psi))
            return np.concatenate(out)
        scale = np.r_[[0.01] * 3, [1.0] * 3, [0.01] * N_DISKS, [1e-3] * 8 * len(shapes)]
        x = least_squares(fun, x0, loss="soft_l1", f_scale=LOSS_PX, x_scale=scale).x
        model.base, model.psi = geo.to_T(x[:3], x[3:6]), x[6:6 + N_DISKS]
        shapes = [x[6 + N_DISKS + 8 * h:6 + N_DISKS + 8 * (h + 1)] for h in range(len(shapes))]
    return model, shapes


def initial_model(points):
    """Rough model from triangulated marker points (N,3) of the robot at home: upright,
    base one spacing below the lowest marker; disk rotation chosen by grid search later."""
    points = np.asarray(points, float)
    base = np.eye(4)
    base[:3, 3] = [np.median(points[:, 0]), np.median(points[:, 1]),
                   points[:, 2].min() - SPACING_MM]
    return Model(base)


def search_start(model, frames, cams, intr, steps=9, dz=np.arange(-100, 101, 10),
                 scales=np.arange(0.9, 1.45, 0.05)):
    """Base height (+-100 mm), disk spacing (x scales of the measured lengths) and common
    disk rotation (0-90 deg, the markers repeat every 90) by grid search, robot straight."""
    best, best_score = None, np.inf
    for z, sc, p in ((z, sc, p) for z in dz for sc in scales
                     for p in np.arange(steps) * (np.pi / 2) / steps):
        base = model.base.copy()
        base[2, 3] += z
        if True:
            m = Model(base, np.full(N_DISKS, p), model.lengths * sc, model.rim)
            sc = score(m, np.zeros(8), frames, cams, intr)
            if sc < best_score:
                best, best_score = m, sc
    return best


def tip(model, curv):
    """Tip disk position (3,) and backbone direction (3,) in the base frame."""
    T = geo.inv(model.base) @ model.disks(curv)[-1]
    return T[:3, 3], T[:3, 2]
