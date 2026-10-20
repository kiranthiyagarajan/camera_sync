"""Camera calibration: intrinsics (per camera) and extrinsics (all cameras in one world frame).

Files (calib/):
  intrinsics/<camera>.json  K, distortion, image size, reprojection error
  extrinsics.json           per camera R, t (x_cam = R x_world + t, mm), centre in world
The world frame is the calibration board lying flat in the first extrinsics capture:
origin at its first inner corner, x along the long side, z up.
"""
import json
import os
import time

import cv2
import numpy as np

from . import geometry as geo
from .config import CALIB_DIR

INTRINSICS_DIR = os.path.join(CALIB_DIR, "intrinsics")
EXTRINSICS_PATH = os.path.join(CALIB_DIR, "extrinsics.json")


# ---- intrinsics ---------------------------------------------------------------
def calibrate_intrinsics(board, views, image_size, reject=2.5):
    """views: list of (corners (N,2), ids (N,)). Views whose error is more than `reject`
    times the median are dropped once and the calibration is repeated.
    Returns dict(K, dist, rms, per_view, used)."""
    def run(vs):
        corners = [c.reshape(-1, 1, 2).astype(np.float32) for c, _ in vs]
        ids = [i.reshape(-1, 1).astype(np.int32) for _, i in vs]
        rms, K, dist, _, _, _, _, per_view = cv2.aruco.calibrateCameraCharucoExtended(
            corners, ids, board.board, tuple(image_size), None, None)
        return rms, K, dist, per_view.ravel()

    rms, K, dist, per_view = run(views)
    keep = per_view <= reject * np.median(per_view)
    used = list(np.flatnonzero(keep))
    if keep.sum() < len(views) and keep.sum() >= 10:
        rms, K, dist, per_view = run([views[i] for i in used])
    else:
        used = list(range(len(views)))
    return dict(K=K, dist=dist.ravel(), rms=float(rms), per_view=per_view, used=used)


def save_intrinsics(cam, port, result, image_size, board_spec, fps, n_views):
    os.makedirs(INTRINSICS_DIR, exist_ok=True)
    data = {
        "camera": cam, "port": port, "image_size": list(image_size),
        "K": result["K"].tolist(), "dist": list(map(float, result["dist"])),
        "rms_px": result["rms"], "views_used": len(result["used"]), "views_captured": n_views,
        "board": board_spec, "fps": fps, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(INTRINSICS_DIR, f"{cam}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_intrinsics(cam):
    path = os.path.join(INTRINSICS_DIR, f"{cam}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return np.array(d["K"]), np.array(d["dist"]), d


# ---- extrinsics ---------------------------------------------------------------
def board_pose(board, corners, ids, K, dist):
    """Board -> camera transform from one detection (planar PnP, refined). Returns (T, rms_px)."""
    obj = board.corners3d[ids]
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, None
    rvec, tvec = cv2.solvePnPRefineLM(obj, corners, K, dist, rvec, tvec)
    T = geo.to_T(rvec, tvec)
    err = geo.project(obj, T, K, dist) - corners
    return T, float(np.sqrt((err ** 2).sum(axis=1).mean()))


def initial_extrinsics(board, captures, intr):
    """captures: list of {cam: (corners, ids)}; the first defines the world frame.
    intr: {cam: (K, dist)}. Returns (cams {cam: T_cam_world}, boards [T_world_board])
    by chaining board poses through cameras that see several captures."""
    poses = [{c: board_pose(board, co, i, *intr[c])[0] for c, (co, i) in cap.items()}
             for cap in captures]
    cams = {c: T for c, T in poses[0].items() if T is not None}
    boards = [np.eye(4)] + [None] * (len(captures) - 1)
    changed = True
    while changed:
        changed = False
        for m, p in enumerate(poses):
            if boards[m] is None:
                known = [c for c in p if c in cams and p[c] is not None]
                if known:
                    c = known[0]
                    boards[m] = geo.inv(cams[c]) @ p[c]
                    changed = True
            if boards[m] is not None:
                for c, T in p.items():
                    if c not in cams and T is not None:
                        cams[c] = T @ geo.inv(boards[m])
                        changed = True
    return cams, boards


def residuals_by_camera(board, captures, intr, cams, boards):
    out = {c: [] for c in cams}
    for m, cap in enumerate(captures):
        if boards[m] is None:
            continue
        for c, (corners, ids) in cap.items():
            if c in cams:
                pts = (boards[m] @ np.c_[board.corners3d[ids], np.ones(len(ids))].T).T[:, :3]
                out[c].append(geo.project(pts, cams[c], *intr[c]) - corners)
    return {c: np.vstack(v) if v else np.zeros((0, 2)) for c, v in out.items()}


def bundle_adjust(board, captures, intr, cams, boards, f_scale=1.0, max_nfev=100):
    """Refine all camera poses and board poses jointly (board 0 fixed = world frame),
    minimising reprojection error with a robust (Huber) loss. Needs scipy."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    cam_ids = sorted(cams)
    board_idx = [m for m in range(1, len(captures)) if boards[m] is not None]
    n_c = len(cam_ids)
    obs = []  # (capture, cam index, corners, ids)
    for m, cap in enumerate(captures):
        if m == 0 or m in board_idx:
            for c, (corners, ids) in cap.items():
                if c in cams:
                    obs.append((m, cam_ids.index(c), corners, ids))

    def pack():
        x = [np.r_[geo.from_T(cams[c])] for c in cam_ids]
        x += [np.r_[geo.from_T(boards[m])] for m in board_idx]
        return np.concatenate(x)

    def unpack(x):
        Tc = [geo.to_T(x[6 * i:6 * i + 3], x[6 * i + 3:6 * i + 6]) for i in range(n_c)]
        Tb = {0: np.eye(4)}
        for j, m in enumerate(board_idx):
            k = 6 * (n_c + j)
            Tb[m] = geo.to_T(x[k:k + 3], x[k + 3:k + 6])
        return Tc, Tb

    def fun(x):
        Tc, Tb = unpack(x)
        res = []
        for m, ci, corners, ids in obs:
            pts = (Tb[m] @ np.c_[board.corners3d[ids], np.ones(len(ids))].T).T[:, :3]
            res.append((geo.project(pts, Tc[ci], *intr[cam_ids[ci]]) - corners).ravel())
        return np.concatenate(res)

    n_res = sum(2 * len(o[3]) for o in obs)
    sparsity = lil_matrix((n_res, 6 * (n_c + len(board_idx))), dtype=int)
    row = 0
    for m, ci, _, ids in obs:
        r = slice(row, row + 2 * len(ids))
        sparsity[r, 6 * ci:6 * ci + 6] = 1
        if m != 0:
            j = n_c + board_idx.index(m)
            sparsity[r, 6 * j:6 * j + 6] = 1
        row += 2 * len(ids)
    # Capped: the last ~2000 iterations change camera centres by < 0.1 mm but take minutes
    # (the window froze and was force-quit).
    sol = least_squares(fun, pack(), jac_sparsity=sparsity, loss="huber", f_scale=f_scale,
                        x_scale="jac", method="trf", max_nfev=max_nfev)
    Tc, Tb = unpack(sol.x)
    new_boards = [Tb.get(m) for m in range(len(captures))]
    return {c: Tc[i] for i, c in enumerate(cam_ids)}, new_boards


def z_up(cams, boards):
    """Make the world z axis point from the board towards the cameras (they are above it)."""
    centres = np.array([geo.camera_center(T) for T in cams.values()])
    if centres[:, 2].mean() >= 0:
        return cams, boards
    F = np.diag([1.0, -1.0, -1.0, 1.0])  # 180 deg about x: keeps x, flips y and z
    return ({c: T @ F for c, T in cams.items()},
            [None if B is None else F @ B for B in boards])


def board_check(board, captures, intr, cams):
    """Triangulate board corners seen by 2+ cameras and compare neighbouring-corner distances
    with the true square size. Returns (mean abs error mm, max abs error mm, n)."""
    sq = board.spec["square_mm"]
    nx = board.spec["squares_x"] - 1
    errs = []
    for cap in captures:
        seen = {}
        for c, (corners, ids) in cap.items():
            if c not in cams:
                continue
            und = geo.undistort(corners, *intr[c])
            for i, xy in zip(ids, und):
                seen.setdefault(int(i), []).append((cams[c], xy))
        pts = {i: geo.triangulate(o) for i, o in seen.items() if len(o) >= 2}
        for i, p in pts.items():
            for j in (i + 1, i + nx):  # right and next-row neighbours
                if j in pts and (j != i + 1 or (i + 1) % nx):
                    errs.append(abs(np.linalg.norm(pts[j] - p) - sq))
    if not errs:
        return None
    errs = np.array(errs)
    return float(errs.mean()), float(errs.max()), len(errs)


def save_extrinsics(cams, intr_meta, residuals, check, ports, n_captures, refined):
    os.makedirs(CALIB_DIR, exist_ok=True)
    data = {
        "world_frame": "calibration board lying flat in the first capture; origin at its first "
                       "inner corner, x along the long side, z up; millimetres",
        "convention": "x_cam = R @ x_world + t",
        "refined_by_bundle_adjustment": refined,
        "captures": n_captures,
        "board_check_mm": None if check is None else {"mean": check[0], "max": check[1], "n": check[2]},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cameras": {},
    }
    for c, T in cams.items():
        r = residuals.get(c, np.zeros((0, 2)))
        data["cameras"][c] = {
            "port": ports.get(c), "R": T[:3, :3].tolist(), "t": T[:3, 3].tolist(),
            "center_mm": geo.camera_center(T).tolist(),
            "rms_px": float(np.sqrt((r ** 2).sum(axis=1).mean())) if len(r) else None,
            "intrinsics_created": intr_meta.get(c, {}).get("created"),
        }
    with open(EXTRINSICS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return EXTRINSICS_PATH


def load_extrinsics():
    if not os.path.exists(EXTRINSICS_PATH):
        return None
    with open(EXTRINSICS_PATH) as f:
        d = json.load(f)
    cams = {}
    for c, v in d["cameras"].items():
        T = np.eye(4)
        T[:3, :3] = np.array(v["R"])
        T[:3, 3] = np.array(v["t"])
        cams[c] = T
    return cams, d
