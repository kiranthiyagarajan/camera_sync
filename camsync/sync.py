from collections import deque
from dataclasses import dataclass, field


@dataclass
class Frame:
    cam: str
    n: int             # frame index from the camera's clock model (gaps = dropped frames)
    t: float           # de-jittered capture time, CLOCK_MONOTONIC seconds
    t_raw: float       # kernel timestamp
    data: bytes        # MJPEG

    def decode_gray(self, reduce=1):
        """Decode to a grayscale numpy array; reduce=2/4/8 decodes at lower resolution (faster)."""
        import cv2
        import numpy as np
        flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
                4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[reduce]
        return cv2.imdecode(np.frombuffer(self.data, np.uint8), flag)


@dataclass
class FrameSet:
    """One synchronized moment: every camera's frames around the reference time t."""
    t: float
    frames: dict = field(default_factory=dict)    # cam -> nearest frame, or None if missing
    brackets: dict = field(default_factory=dict)  # cam -> (before, after) frames around t, or None

    @property
    def complete(self):
        return all(f is not None for f in self.frames.values())

    def offset(self, cam):
        """How far (s) this camera's nearest frame is from the set time."""
        f = self.frames[cam]
        return None if f is None else f.t - self.t

    def weight(self, cam):
        """Interpolation weight w so that value(t) = (1-w)*before + w*after, for per-camera
        results such as 2D marker positions. None if the camera has no bracketing pair."""
        b = self.brackets[cam]
        if b is None:
            return None
        before, after = b
        return 1.0 if after.t == before.t else (self.t - before.t) / (after.t - before.t)


class Synchronizer:
    """Groups frames from free-running cameras into FrameSets.

    Each frame of the reference camera (the first one) defines a set time. A set is
    emitted once every camera has a frame at or after that time, or once any camera
    is max_wait periods past it (then late cameras are marked missing). Driven
    purely by frame timestamps, so it is deterministic and testable.
    """

    def __init__(self, cams, period, max_wait=1.5, tolerance=0.6):
        self.cams = list(cams)
        self.ref = self.cams[0]
        self.period = period
        self.max_wait = max_wait    # in periods
        self.tolerance = tolerance  # nearest frame further than this (periods) counts as missing
        self.buffers = {c: deque(maxlen=32) for c in self.cams}
        self.pending = deque()
        self.latest = float("-inf")

    def push(self, frame):
        """Add a frame (frames of one camera must arrive in time order). Returns ready sets."""
        self.buffers[frame.cam].append(frame)
        self.latest = max(self.latest, frame.t)
        if frame.cam == self.ref:
            self.pending.append(frame)
        out = []
        while self.pending:
            s = self._try_emit(self.pending[0].t)
            if s is None:
                break
            self.pending.popleft()
            out.append(s)
        return out

    def _try_emit(self, t):
        timed_out = self.latest >= t + self.max_wait * self.period
        if not timed_out and any(not buf or buf[-1].t < t for buf in self.buffers.values()):
            return None
        s = FrameSet(t)
        for cam, buf in self.buffers.items():
            before = next((f for f in reversed(buf) if f.t < t), None)
            after = next((f for f in buf if f.t >= t), None)
            near = min((f for f in (before, after) if f is not None),
                       key=lambda f: abs(f.t - t), default=None)
            if near is not None and abs(near.t - t) > self.tolerance * self.period:
                near = None
            s.frames[cam] = near
            s.brackets[cam] = ((before, after) if before and after
                               and after.n - before.n == 1 else None)
            while buf and buf[0].t < t - self.period:
                buf.popleft()
        return s
