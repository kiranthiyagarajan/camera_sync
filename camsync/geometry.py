"""Camera geometry: poses, projection, triangulation. Units are millimetres."""
import cv2
import numpy as np


def to_T(rvec, tvec):
    """Rodrigues rvec + tvec -> 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    T[:3, 3] = np.asarray(tvec, float).reshape(3)
    return T


def from_T(T):
    """4x4 transform -> (rvec, tvec)."""
    return cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3].copy()


def inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def project(points, T_cam_world, K, dist):
    """World points (N,3) -> pixel coordinates (N,2) in a camera."""
    rvec, tvec = from_T(T_cam_world)
    px, _ = cv2.projectPoints(np.asarray(points, float).reshape(-1, 1, 3), rvec, tvec, K, dist)
    return px.reshape(-1, 2)


def undistort(px, K, dist):
    """Pixel coordinates (N,2) -> normalized, undistorted image coordinates (N,2)."""
    return cv2.undistortPoints(np.asarray(px, float).reshape(-1, 1, 2), K, dist).reshape(-1, 2)


def triangulate(observations):
    """Least-squares (DLT) triangulation of one point.

    observations: list of (T_cam_world, xy) with xy normalized & undistorted.
    Returns the world point (3,).
    """
    rows = []
    for T, (x, y) in observations:
        P = T[:3, :]
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    return X[:3] / X[3]


def camera_center(T_cam_world):
    return inv(T_cam_world)[:3, 3]
