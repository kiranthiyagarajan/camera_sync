"""Proof-of-concept robot pose from unlabeled 3D markers.

The robot has 4 disks with 4 markers each around the rim. A camera sees the markers
facing it, so each marker's disk centre lies about one rim radius behind it (away from
the cameras that see it, in the horizontal plane: the robot stands upright). These
per-marker centre estimates of one disk land close together, so they are grouped by
distance; disks are ordered base -> tip by height. With 3+ markers a plane fit gives
the disk normal; the backbone direction comes from neighbouring disk centres.
"""
import numpy as np

RIM_RADIUS_MM = 30.0  # disk centre to marker centre
LINK_MM = 30.0        # centre estimates closer than this (chained) belong to one disk
N_DISKS = 4


def cluster(points, link=LINK_MM):
    """Single-linkage clusters of (N,3) points. Returns a list of index arrays."""
    n = len(points)
    label = -np.ones(n, int)
    k = 0
    for s in range(n):
        if label[s] >= 0:
            continue
        label[s] = k
        stack = [s]
        while stack:
            i = stack.pop()
            near = np.nonzero((label < 0) & (np.linalg.norm(points - points[i], axis=1) < link))[0]
            label[near] = k
            stack.extend(near.tolist())
        k += 1
    return [np.nonzero(label == j)[0] for j in range(k)]


def centre_estimates(points, seen_from, radius=RIM_RADIUS_MM):
    """Move each marker `radius` away from the cameras that see it, horizontally.
    seen_from: list of (M,3) camera centres per point."""
    out = []
    for X, centres in zip(points, seen_from):
        d = (X - np.asarray(centres)).mean(axis=0)
        d[2] = 0
        n = np.linalg.norm(d)
        out.append(X + radius * d / n if n > 0 else X)
    return np.array(out).reshape(-1, 3)


def disks(points, seen_from=None, radius=RIM_RADIUS_MM, link=LINK_MM, n_disks=N_DISKS):
    """Returns a list, base to tip, of dicts: centre (3,), normal (3,) or None,
    direction (3,) or None, n markers, markers (n,3)."""
    points = np.asarray(points, float).reshape(-1, 3)
    est = centre_estimates(points, seen_from, radius) if seen_from is not None else points
    out = []
    for idx in cluster(est, link):
        p = points[idx]
        normal = None
        if len(p) >= 3:
            normal = np.linalg.svd(p - p.mean(axis=0))[2][2]
            if normal[2] < 0:
                normal = -normal
        out.append({"centre": est[idx].mean(axis=0), "normal": normal, "n": len(p), "markers": p})
    # Keep the best-supported disks (stray points form small extra groups).
    out.sort(key=lambda d: -d["n"])
    out = out[:n_disks]
    out.sort(key=lambda d: d["centre"][2])
    centres = [d["centre"] for d in out]
    for i, d in enumerate(out):  # direction along the backbone at this disk
        v = centres[min(len(out) - 1, i + 1)] - centres[max(0, i - 1)]
        d["direction"] = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else d["normal"]
    return out
