"""Rotatable 3D view of the tracked robot, and the same pose drawn over camera images.

Orbit is a perspective camera circling a target point (z up): left-drag rotates,
the scroll wheel or right-drag zooms. Units are millimetres, colours BGR.
"""
import cv2
import numpy as np

from . import geometry as geo
from .plots import COLORS, text
from .pose import RIM_RADIUS_MM, tangents

DISK = (0, 255, 0)
TIP = (0, 0, 255)
BACKBONE = (0, 200, 255)
AXES = ((0, 0, 255), (0, 255, 0), (255, 120, 0))  # x red, y green, z blue
TRIAD_MM = 40.0


def rim_points(centre, normal, radius=RIM_RADIUS_MM, n=32):
    """(n,3) points on a circle around `centre` in the plane with this normal."""
    a = np.cross(normal, [1.0, 0, 0])
    if np.linalg.norm(a) < 0.1:
        a = np.cross(normal, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return centre + radius * (np.outer(np.cos(ang), a) + np.outer(np.sin(ang), b))


def disk_frame(tangent):
    """Rows x, y, z of a frame with z along the backbone and x as close to world x as
    possible (the roll about the backbone is unknown until markers are labelled)."""
    z = tangent / np.linalg.norm(tangent)
    x = np.array([1.0, 0, 0]) - z[0] * z
    if np.linalg.norm(x) < 0.1:
        x = np.array([0, 1.0, 0]) - z[1] * z
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z])


class Orbit:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.f = 0.9 * w
        self.drag = None
        self.reset()

    def reset(self):
        self.yaw, self.pitch, self.dist = -60.0, 25.0, 1300.0

    def zoom(self, steps):
        self.dist = float(np.clip(self.dist * 0.9 ** steps, 200, 5000))

    def mouse(self, event, x, y, flags):
        """Feed OpenCV mouse events with x, y relative to the view panel."""
        inside = 0 <= x < self.w and 0 <= y < self.h
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and inside:
            self.drag = (event, x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            kind, x0, y0 = self.drag
            if kind == cv2.EVENT_LBUTTONDOWN:
                self.yaw -= 0.4 * (x - x0)
                self.pitch = float(np.clip(self.pitch + 0.4 * (y - y0), -89, 89))
            else:
                self.zoom((y0 - y) / 20)
            self.drag = (kind, x, y)
        elif event == cv2.EVENT_MOUSEWHEEL and inside:
            self.zoom(1 if flags > 0 else -1)

    def look(self, target):
        y, p = np.radians(self.yaw), np.radians(self.pitch)
        back = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
        self.eye = target + self.dist * back
        fwd = -back
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.array([right, down, fwd])

    def project(self, X):
        """World (N,3) -> pixels (N,2) and in-front mask (N,)."""
        Xc = (np.asarray(X, float).reshape(-1, 3) - self.eye) @ self.R.T
        z = Xc[:, 2]
        zs = np.where(z > 1, z, 1)
        px = self.f * Xc[:, :2] / zs[:, None] + (self.w / 2, self.h / 2)
        return px, z > 1


def _poly(img, px, closed, color, thick=1):
    cv2.polylines(img, [np.round(px).astype(np.int32)], closed, color, thick, cv2.LINE_AA)


def draw_scene(img, orbit, target, cams, intr, size, xyz, ds):
    """Floor grid, world axes, camera frustums, markers, disk rims and backbone."""
    orbit.look(target)

    def line(a, b, color, thick=1):
        px, ok = orbit.project([a, b])
        if ok.all():
            _poly(img, px, False, color, thick)

    centres = [geo.camera_center(T) for T in cams.values()]
    if centres:
        c = np.array(centres)
        lo = np.floor(c[:, :2].min(axis=0) / 50) * 50
        hi = np.ceil(c[:, :2].max(axis=0) / 50) * 50
    else:
        lo, hi = np.array([-300.0, -300.0]), np.array([300.0, 300.0])
    for gx in np.arange(lo[0], hi[0] + 1, 50):  # 50 mm floor grid
        line((gx, lo[1], 0), (gx, hi[1], 0), (50, 50, 50))
    for gy in np.arange(lo[1], hi[1] + 1, 50):
        line((lo[0], gy, 0), (hi[0], gy, 0), (50, 50, 50))
    for k in range(3):
        line(np.zeros(3), 100 * np.eye(3)[k], AXES[k], 2)

    w, h = size
    for i, (name, T) in enumerate(cams.items()):
        col = COLORS[i % len(COLORS)]
        Ti = geo.inv(T)
        K = intr[name][0] if name in intr else np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
        rays = np.linalg.inv(K) @ np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
        corners = (Ti[:3, :3] @ (60 * rays) + Ti[:3, 3:4]).T
        o = Ti[:3, 3]
        for a in corners:
            line(o, a, col)
        px, ok = orbit.project(corners)
        if ok.all():
            _poly(img, px, True, col)
        px, ok = orbit.project(o)
        if ok[0]:
            text(img, name, (int(px[0, 0]) + 5, int(px[0, 1]) - 5), 0.45, col)

    px, ok = orbit.project(xyz)
    for p in px[ok]:
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (200, 200, 200), -1, cv2.LINE_AA)
    for d in ds:
        for X in d["markers"]:
            line(X, d["centre"], (90, 90, 90))
    for i in range(len(ds) - 1):
        line(ds[i]["centre"], ds[i + 1]["centre"], BACKBONE, 2)
    for i, (d, (t, fitted)) in enumerate(zip(ds, tangents(ds))):
        col = TIP if i == len(ds) - 1 else DISK
        if fitted:
            px, ok = orbit.project(rim_points(d["centre"], t))
            if ok.all():
                _poly(img, px, True, col, 2)
            line(d["centre"], d["centre"] + 25 * t, col, 1)
        else:
            px, ok = orbit.project(d["centre"])
            if ok[0]:
                cv2.circle(img, (int(px[0, 0]), int(px[0, 1])), 5, col, -1, cv2.LINE_AA)


def draw_overlay(img, T, K, dist, ds, scale=1.0):
    """The pose over one camera image (pixels multiplied by `scale`)."""
    if not ds:
        return

    def proj(X):
        return geo.project(np.asarray(X).reshape(-1, 3), T, K, dist) * scale

    ts = tangents(ds)
    if len(ds) > 1:
        _poly(img, proj([d["centre"] for d in ds]), False, BACKBONE, 2)
    for i, (d, (t, fitted)) in enumerate(zip(ds, ts)):
        col = TIP if i == len(ds) - 1 else DISK
        if fitted:
            _poly(img, proj(rim_points(d["centre"], t)), True, col, 1)
        else:
            c = proj(d["centre"])[0]
            cv2.circle(img, (int(c[0]), int(c[1])), 4, col, -1, cv2.LINE_AA)
    t0 = ts[0][0]
    if t0 is not None:  # axes at the base disk
        o = ds[0]["centre"]
        for k, a in enumerate(disk_frame(t0)):
            _poly(img, proj([o, o + TRIAD_MM * a]), False, AXES[k], 2)
