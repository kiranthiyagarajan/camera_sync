"""Calibration maths on synthetic data: 4 cameras at the corners of a 500 mm cube."""
import os
import unittest

import cv2
import numpy as np

from camsync import calib, config
from camsync import geometry as geo
from camsync.board import Board

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
DIST = np.array([-0.12, 0.05, 0.0005, -0.0003, 0.0])
SIZE = (1280, 720)


def look_at(eye, target, up=(0, 0, 1)):
    """T_cam_world for a camera at `eye` looking at `target` (OpenCV: z forward, y down)."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T


def observe(board, T_cam_world, T_world_board, rng, noise=0.15):
    pts = (T_world_board @ np.c_[board.corners3d, np.ones(board.n_corners)].T).T[:, :3]
    px = geo.project(pts, T_cam_world, K, DIST)
    cam_pts = (T_cam_world @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    ok = (px[:, 0] > 5) & (px[:, 0] < SIZE[0] - 5) & (px[:, 1] > 5) & (px[:, 1] < SIZE[1] - 5) \
        & (cam_pts[:, 2] > 0)
    # board must face the camera
    normal = T_world_board[:3, 2]
    centre = T_world_board[:3, 3]
    facing = np.dot(geo.camera_center(T_cam_world) - centre, normal) > 0
    ids = np.flatnonzero(ok)
    if not facing or len(ids) < 8:
        return None
    return px[ids] + rng.normal(0, noise, (len(ids), 2)), ids


def board_pose_world(rng, centre, tilt_deg):
    rx, ry = np.radians(rng.uniform(-tilt_deg, tilt_deg, 2))
    rz = rng.uniform(0, 2 * np.pi)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([rx, ry, 0.0]))[0] @ cv2.Rodrigues(np.array([0, 0, rz]))[0]
    T[:3, 3] = centre
    return T


class IntrinsicsTest(unittest.TestCase):
    def test_recovers_lens(self):
        board = Board(config.default_config()["board"])
        rng = np.random.default_rng(1)
        views = []
        flip = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]  # camera on the board's front (+z) side
        for _ in range(2000):
            if len(views) == 25:
                break
            T = np.eye(4)
            T[:3, :3] = cv2.Rodrigues(rng.uniform(-0.5, 0.5, 3))[0] @ flip
            T[:3, 3] = [rng.uniform(-200, 50), rng.uniform(-20, 150), rng.uniform(350, 600)]
            o = observe(board, T, np.eye(4), rng)
            if o is not None and len(o[1]) >= 20:
                views.append(o)
        self.assertEqual(len(views), 25, "could not generate test views")
        views[3] = (views[3][0] + rng.normal(0, 6, views[3][0].shape), views[3][1])  # a bad view
        res = calib.calibrate_intrinsics(board, views, SIZE)
        self.assertNotIn(3, res["used"])
        self.assertLess(res["rms"], 0.3)
        self.assertLess(abs(res["K"][0, 0] - 900) / 900, 0.01)
        self.assertLess(abs(res["K"][0, 2] - 640), 5)


class ExtrinsicsTest(unittest.TestCase):
    def setUp(self):
        self.board = Board(config.default_config()["board"])
        self.rng = np.random.default_rng(2)
        heights = [380, 420, 400, 440]
        corners = [(-250, -250), (250, -250), (250, 250), (-250, 250)]
        self.truth = {f"cam{i + 1}": look_at(np.array([x, y, h], float), (0, 0, 60))
                      for i, ((x, y), h) in enumerate(zip(corners, heights))}
        self.intr = {c: (K, DIST) for c in self.truth}
        # World = first board: lying flat, centred near the middle of the cube floor.
        flat = np.eye(4)
        flat[:3, 3] = [-157.5, -105, 0]
        self.boards = [flat] + [board_pose_world(self.rng, [self.rng.uniform(-60, 60),
                                                            self.rng.uniform(-60, 60),
                                                            self.rng.uniform(20, 150)], 35)
                                for _ in range(15)]
        self.captures = []
        for B in self.boards:
            cap = {}
            for c, T in self.truth.items():
                o = observe(self.board, T, B, self.rng)
                if o is not None:
                    cap[c] = o
            self.captures.append(cap)
        # Express truth in the world frame defined by the first board.
        self.truth_w = {c: T @ flat for c, T in self.truth.items()}

    def centre_errors(self, cams):
        return {c: np.linalg.norm(geo.camera_center(cams[c]) - geo.camera_center(self.truth_w[c]))
                for c in self.truth}

    def test_initial_solution(self):
        self.assertEqual(len(self.captures[0]), 4)  # flat board seen by all cameras
        cams, boards = calib.initial_extrinsics(self.board, self.captures, self.intr)
        cams, boards = calib.z_up(cams, boards)
        errs = self.centre_errors(cams)
        self.assertLess(max(errs.values()), 15, errs)  # single-view PnP: rough
        for c in cams:  # all cameras above the board
            self.assertGreater(geo.camera_center(cams[c])[2], 300)

    def test_bundle_adjustment(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        cams, boards = calib.initial_extrinsics(self.board, self.captures, self.intr)
        cams, boards = calib.bundle_adjust(self.board, self.captures, self.intr, cams, boards)
        cams, boards = calib.z_up(cams, boards)
        errs = self.centre_errors(cams)
        self.assertLess(max(errs.values()), 1.0, errs)
        check = calib.board_check(self.board, self.captures, self.intr, cams)
        self.assertLess(check[0], 0.1)

    def test_triangulation_check(self):
        check = calib.board_check(self.board, self.captures, self.intr, self.truth_w)
        self.assertIsNotNone(check)
        self.assertLess(check[0], 0.15)  # true cameras: only pixel noise remains


if __name__ == "__main__":
    unittest.main()


class AutoCaptureTest(unittest.TestCase):
    """Intrinsics view capture: only still boards, only new views."""

    def test_capture_rules(self):
        import tempfile
        from types import SimpleNamespace
        import camsync.calibrate as C
        rig = SimpleNamespace(ids=["cam1"], width=1280, height=720, ports={"cam1": "3-1"},
                              fps=120, latest={})
        board = Board(config.default_config()["board"])
        with tempfile.TemporaryDirectory() as tmp:
            C.CALIB_DIR = tmp
            cal = C.Calibrator(rig, board, "intrinsics")
            gray = np.zeros((720, 1280), np.uint8)
            ids = np.arange(30)
            base = np.c_[np.linspace(200, 900, 30), np.linspace(150, 500, 30)]

            def tick(corners):
                cal.update([("cam1", gray, (corners, ids))])

            tick(base + 30)  # moving
            for _ in range(C.STILL_TICKS):
                self.assertEqual(len(cal.state["cam1"].views), 0)
                tick(base)
            tick(base)
            self.assertEqual(len(cal.state["cam1"].views), 1)       # still -> captured
            for _ in range(5):
                tick(base + 0.2)                                     # same view, still
            self.assertEqual(len(cal.state["cam1"].views), 1)       # not novel -> skipped
            moved = base + [200, 100]
            for _ in range(C.STILL_TICKS + 1):
                tick(moved)
            self.assertEqual(len(cal.state["cam1"].views), 2)       # new position -> captured
            self.assertTrue(cal.state["cam1"].coverage.any())
            self.assertEqual(len(os.listdir(os.path.join(tmp, "views", "cam1"))), 2)
