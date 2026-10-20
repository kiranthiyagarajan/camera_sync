"""Proof-of-concept robot pose from unlabeled 3D markers.

Markers are grouped into disks by distance (markers on one disk are closer to each
other than to the next disk), disks are ordered base -> tip by height, and each disk
gives a centre and, with 3+ markers, a normal from a plane fit. Segment direction
falls back to the line between neighbouring disk centres.
"""
import numpy as np

LINK_MM = 40.0  # markers closer than this (chained) belong to the same disk


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


def disks(points, link=LINK_MM):
    """Returns a list, base to tip, of dicts: centre (3,), normal (3,) or None, n markers."""
    points = np.asarray(points, float).reshape(-1, 3)
    out = []
    for idx in cluster(points, link):
        p = points[idx]
        centre = p.mean(axis=0)
        normal = None
        if len(p) >= 3:
            normal = np.linalg.svd(p - centre)[2][2]
            if normal[2] < 0:
                normal = -normal
        out.append({"centre": centre, "normal": normal, "n": len(p)})
    out.sort(key=lambda d: d["centre"][2])
    centres = [d["centre"] for d in out]
    for i, d in enumerate(out):  # direction along the backbone at this disk
        a = centres[max(0, i - 1)]
        b = centres[min(len(out) - 1, i + 1)]
        v = b - a
        d["direction"] = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else d["normal"]
    return out
