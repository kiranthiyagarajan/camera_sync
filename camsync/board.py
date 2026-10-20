"""ChArUco calibration board: definition (from rig.json) and detection."""
import cv2
import numpy as np

DICTIONARIES = ("DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250",
                "DICT_6X6_50", "DICT_6X6_250", "DICT_7X7_50", "DICT_ARUCO_ORIGINAL")


class Board:
    """A ChArUco board. Coordinates are in millimetres in the board's own frame."""

    def __init__(self, spec):
        self.spec = dict(spec)
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
        self.board = cv2.aruco.CharucoBoard_create(
            spec["squares_x"], spec["squares_y"], float(spec["square_mm"]),
            float(spec["marker_mm"]), self.dictionary)
        self.corners3d = np.asarray(self.board.chessboardCorners, float).reshape(-1, 3)
        # Relaxed for small, oblique, partly clipped markers (~2 px per bit at cube distances).
        p = self.params = cv2.aruco.DetectorParameters_create()
        p.perspectiveRemovePixelPerCell = 8
        p.perspectiveRemoveIgnoredMarginPerCell = 0.25
        p.errorCorrectionRate = 0.8
        p.minMarkerPerimeterRate = 0.01
        p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 33, 6

    @property
    def n_corners(self):
        return len(self.corners3d)

    def detect(self, gray, K=None, dist=None):
        """Find the board's chessboard corners. Returns (corners (N,2), ids (N,)) or None.
        With camera intrinsics, corners hidden between detected markers are recovered too."""
        # Markers are found in a 2x enlarged image (the detector needs a few pixels per bit),
        # then mapped back; chessboard corners are refined in the original image.
        big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        mk, ids, rejected = cv2.aruco.detectMarkers(big, self.dictionary, parameters=self.params)
        if ids is None or len(ids) < 2:
            return None
        mk = [(c + 0.5) / 2 - 0.5 for c in mk]
        rejected = [(c + 0.5) / 2 - 0.5 for c in rejected]
        mk, ids, _, _ = cv2.aruco.refineDetectedMarkers(gray, self.board, mk, ids, rejected,
                                                        cameraMatrix=K, distCoeffs=dist)
        if ids is None or len(ids) < 2:
            return None
        n, corners, cids = cv2.aruco.interpolateCornersCharuco(mk, ids, gray, self.board,
                                                               cameraMatrix=K, distCoeffs=dist)
        if not n or n < 4:
            return None
        return corners.reshape(-1, 2).astype(float), cids.ravel().astype(int)

    def guess_dictionary(self, gray):
        """Which predefined dictionary has the most markers in this image (to catch a wrong spec)."""
        best_n, best_name = 0, None
        for name in DICTIONARIES:
            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
            _, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=self.params)
            n = 0 if ids is None else len(ids)
            if n > best_n:
                best_n, best_name = n, name
        return best_name if best_n >= 4 else None

    def image(self, px_per_mm=4, margin_mm=10):
        w = int((self.spec["squares_x"] * self.spec["square_mm"] + 2 * margin_mm) * px_per_mm)
        h = int((self.spec["squares_y"] * self.spec["square_mm"] + 2 * margin_mm) * px_per_mm)
        return self.board.draw((w, h), marginSize=int(margin_mm * px_per_mm), borderBits=1)
