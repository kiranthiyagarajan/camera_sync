"""Per-camera frame rate and frame-loss statistics, bucketed per second."""
import csv
import os
import threading
import time
from collections import deque


class CamStats:
    """Counts received and lost frames per wall-clock second.

    The capture thread calls add(); readers call tick() so that a stalled camera
    still produces (empty) seconds and its frame rate visibly drops to zero.
    """

    def __init__(self, history=3600):
        self._lock = threading.Lock()
        self.history = history
        self.reset()

    def reset(self):
        with self._lock:
            self.hist = deque(maxlen=self.history)  # (second, frames, lost)
            self._sec = None
            self._partial = True  # the first second after a reset is incomplete; not recorded
            self._frames = self._lost = 0
            self.total_frames = self.total_lost = 0
            self.started = time.monotonic()

    def add(self, frames=1, lost=0, now=None):
        with self._lock:
            self._roll(int(time.monotonic() if now is None else now))
            self._frames += frames
            self._lost += lost
            self.total_frames += frames
            self.total_lost += lost

    def tick(self, now=None):
        with self._lock:
            self._roll(int(time.monotonic() if now is None else now))

    def _roll(self, sec):
        if self._sec is None:
            self._sec = sec
            return
        if sec - self._sec > self.history:  # long pause: skip ahead
            self._sec = sec - self.history
        while self._sec < sec:
            if self._partial:
                self._partial = False
            else:
                self.hist.append((self._sec, self._frames, self._lost))
            self._frames = self._lost = 0
            self._sec += 1

    def seconds(self, k):
        """The last k complete seconds as [(second, frames, lost)], oldest first."""
        with self._lock:
            return list(self.hist)[-k:]

    def window(self, k):
        """(fps, loss_fraction) over the last k complete seconds."""
        secs = self.seconds(k)
        if not secs:
            return float("nan"), 0.0
        frames = sum(s[1] for s in secs)
        lost = sum(s[2] for s in secs)
        return frames / len(secs), (lost / (frames + lost) if frames + lost else 1.0)

    @property
    def total_loss(self):
        n = self.total_frames + self.total_lost
        return self.total_lost / n if n else 0.0


class CsvLogger:
    """Writes one row per camera per second: frame rate, loss and camera settings."""

    FIELDS = ["time", "elapsed_s", "camera", "port", "fps", "lost", "loss_pct_10s",
              "loss_pct_total", "exposure_ms", "gain", "sync_offset_ms"]

    def __init__(self, prefix):
        from .config import LOG_DIR
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = os.path.join(LOG_DIR, f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.csv")
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.FIELDS)
        self._start = time.monotonic()
        self._last_sec = {}

    def log(self, rig, offsets=None):
        offsets = offsets or {}
        now = time.time()
        for cid in rig.ids:
            st = rig.stats[cid]
            secs = st.seconds(1)
            if not secs or secs[-1][0] == self._last_sec.get(cid):
                continue
            self._last_sec[cid] = secs[-1][0]
            _, frames, lost = secs[-1]
            _, loss10 = st.window(10)
            ctl = rig.current.get(cid, {})
            off = offsets.get(cid)
            self._w.writerow([
                time.strftime("%H:%M:%S", time.localtime(now)),
                f"{time.monotonic() - self._start:.0f}", cid, rig.ports[cid], frames, lost,
                f"{loss10 * 100:.3f}", f"{st.total_loss * 100:.3f}",
                f"{ctl.get('exposure_time_absolute', 0) / 10:.1f}", ctl.get("gain", ""),
                "" if off is None else f"{off * 1e3:.3f}"])
        self._f.flush()

    def close(self):
        self._f.close()
