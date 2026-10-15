import select
import threading
import time
from collections import deque

from . import config
from .clock import LinearClock
from .stats import CamStats
from .sync import Frame, Synchronizer
from .v4l2 import Camera


class Rig:
    """Captures all cameras on one thread and yields synchronized FrameSets.

        with Rig() as rig:
            for s in rig.sets():
                ...  # s.frames[cam].decode_gray(), s.offset(cam), s.weight(cam)

    Cameras are named and configured from rig.json (see config.py). The consumer
    always gets the newest sets: if it falls behind, older sets are discarded
    (counted in sets_dropped) instead of building up latency. Display tools can
    instead read `latest` (newest frame per camera) and `poll_sets()`.
    """

    def __init__(self, cfg=None, cameras=None, queue=4, nbuf=8):
        self.cfg = config.load() if cfg is None else cfg
        infos = config.resolve(self.cfg)[0] if cameras is None else cameras
        if not infos:
            raise RuntimeError("no cameras found")
        fmt = self.cfg["format"]
        self.ids = [c.id for c in infos]
        self.ports = {c.id: c.port for c in infos}
        self.cams, self.clocks, self.stats = {}, {}, {}
        self.current = {}     # cam -> {control: value read back from the camera}
        self.mismatched = {}  # cam -> {control: (wanted, actual)} the camera didn't accept
        try:
            for info in infos:
                cam = Camera(info.path, fmt["width"], fmt["height"], fmt["fps"], nbuf)
                self.cams[info.id] = cam
                self.current[info.id] = {}
                self.apply_controls(info.id, config.controls_for(self.cfg, info.id))
                self.clocks[info.id] = LinearClock(cam.nominal_period)
                self.stats[info.id] = CamStats()
        except Exception:
            self.close()
            raise
        self.width, self.height = fmt["width"], fmt["height"]
        self.sync = Synchronizer(self.ids, self.clocks[self.ids[0]].period)
        self.latest = {}  # cam -> newest Frame
        self.corrupt = {c: 0 for c in self.ids}
        self.sets_dropped = 0
        self._all_warm = False
        self._out = deque()
        self._queue = queue
        self._cond = threading.Condition()
        self._running = False
        self._error = None
        self._thread = None

    def set_control(self, cam, name, value, tries=3):
        """Set a camera control and read it back; returns the value the camera reports.

        The camera occasionally rejects a request (UVC error -71) and accepts it on retry.
        """
        dev = self.cams[cam]
        for attempt in range(tries):
            try:
                dev.set_control(name, value)
                actual = dev.get_control(name)
            except OSError:
                if attempt == tries - 1:
                    raise
                time.sleep(0.1)
                continue
            if actual == value or attempt == tries - 1:
                break
            time.sleep(0.1)
        self.current[cam][name] = actual
        if actual != value:
            self.mismatched.setdefault(cam, {})[name] = (value, actual)
        else:
            self.mismatched.get(cam, {}).pop(name, None)
        return actual

    def apply_controls(self, cam, controls):
        for name, value in controls.items():
            self.set_control(cam, name, value)

    def start(self):
        for cam in self.cams.values():
            cam.start()
        for st in self.stats.values():
            st.reset()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="rig-capture", daemon=True)
        self._thread.start()

    def _loop(self):
        poller = select.poll()
        by_fd = {}
        for cid, cam in self.cams.items():
            poller.register(cam.fileno(), select.POLLIN)
            by_fd[cam.fileno()] = cid
        try:
            while self._running:
                for fd, _ in poller.poll(100):
                    cid = by_fd[fd]
                    got = self.cams[cid].read()
                    if got is None:
                        continue
                    data, _, t_raw = got
                    clock = self.clocks[cid]
                    dropped_before = clock.dropped
                    n, t = clock.update(t_raw)
                    lost = clock.dropped - dropped_before
                    if data is None:
                        self.corrupt[cid] += 1
                        self.stats[cid].add(frames=0, lost=lost + 1)
                        continue
                    self.stats[cid].add(frames=1, lost=lost)
                    frame = Frame(cid, n, t, t_raw, data)
                    self.latest[cid] = frame
                    # Start grouping only once every camera has a trustworthy clock,
                    # otherwise the first sets are missing the slower-locking cameras.
                    if not self._all_warm:
                        self._all_warm = all(c.warm for c in self.clocks.values())
                        if not self._all_warm:
                            continue
                    if not clock.warm:
                        continue  # re-locking after a stall; the synchronizer marks it missing
                    ready = self.sync.push(frame)
                    if ready:
                        with self._cond:
                            self._out.extend(ready)
                            while len(self._out) > self._queue:
                                self._out.popleft()
                                self.sets_dropped += 1
                            self._cond.notify_all()
                self.sync.period = self.clocks[self.ids[0]].period
        except Exception as e:
            self._error = e
            with self._cond:
                self._cond.notify_all()

    def sets(self, timeout=2.0):
        """Yield FrameSets as they become ready; raises if capture stalls or fails."""
        while True:
            with self._cond:
                deadline = time.monotonic() + timeout
                while not self._out and self._error is None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError(f"no synchronized frames for {timeout}s")
                    self._cond.wait(left)
                self._check_error()
                s = self._out.popleft()
            yield s

    def poll_sets(self):
        """All FrameSets ready right now (non-blocking)."""
        with self._cond:
            self._check_error()
            out = list(self._out)
            self._out.clear()
        return out

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError("capture thread failed") from self._error

    @property
    def synced(self):
        """True once every camera's clock has locked and sets are being produced."""
        return self._all_warm

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join()
            self._thread = None
        for cam in self.cams.values():
            try:
                cam.stop()
            except OSError:
                pass
            cam.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
