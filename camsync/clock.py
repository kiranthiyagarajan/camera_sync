from collections import deque

import numpy as np


class LinearClock:
    """De-jitters one camera's frame timestamps.

    The sensor fires on its own crystal at a near-constant period, but the kernel
    timestamps are quantised to USB transfer timing (about ±2 ms here). Fitting
    t = a + b*n over the recent frames recovers the true frame times to ~0.1 ms,
    tracks slow clock drift, and exposes dropped frames as gaps in n.

    Jitter is close to half a frame period, so drops can't be told apart from
    jitter by comparing consecutive timestamps. Instead frames are numbered
    sequentially until a first fit is validated (`lock_frames`), and from then on
    numbered by where they fall on the fitted line.
    """

    def __init__(self, nominal_period, window=256, lock_frames=32, warmup=128, reject=0.003,
                 max_period_error=0.03, reset_gap=0.5):
        self.nominal_period = nominal_period
        self.period = nominal_period
        self.lock_frames = lock_frames
        self.warmup = warmup              # fitted points before smoothed times are reported
        self.reject = reject              # ignore points this far (s) off the fit, e.g. USB stalls
        self.max_period_error = max_period_error  # vs the driver's nominal frame interval
        self.reset_gap = reset_gap        # a stall this long (s) restarts the model
        self.ns = deque(maxlen=window)
        self.ts = deque(maxlen=window)
        self.n = -1
        self.last_raw = None
        self.a = None
        self.locked = False
        self.dropped = 0
        self._resid_sq = 0.0
        self._resid_count = 0

    @property
    def warm(self):
        return self.locked and len(self.ns) >= self.warmup

    @property
    def jitter(self):
        """RMS of raw timestamps around the fitted clock (s)."""
        return (self._resid_sq / self._resid_count) ** 0.5 if self._resid_count else float("nan")

    def update(self, t_raw):
        """Add a raw timestamp. Returns (frame_index, smoothed_time); the time is raw until warm."""
        if self.last_raw is not None and t_raw - self.last_raw > self.reset_gap:
            missed = max(0, round((t_raw - self.last_raw) / self.period) - 1)
            if self.locked:
                self.dropped += missed  # frames lost during the stall
            self.n += missed  # keep frame numbering continuous across the stall
            self._reset()
        if self.locked:
            n = max(self.n + 1, round((t_raw - self.a) / self.period))
            if self.n >= 0:
                self.dropped += n - self.n - 1
        else:
            n = self.n + 1
        self.n, self.last_raw = n, t_raw

        if self.locked:
            resid = t_raw - (self.a + self.period * n)
            if self.warm:
                self._resid_sq += resid * resid
                self._resid_count += 1
            if abs(resid) <= self.reject:
                self.ns.append(n)
                self.ts.append(t_raw)
                self._fit()
        else:
            self.ns.append(n)
            self.ts.append(t_raw)
            if len(self.ns) >= self.lock_frames:
                self._try_lock()
        return n, (self.a + self.period * n) if self.warm else t_raw

    def _reset(self):
        self.ns.clear()
        self.ts.clear()
        self.a = None
        self.locked = False
        self.period = self.nominal_period

    def _try_lock(self):
        # A frame dropped during warm-up shows up as a large residual or a skewed period.
        resid = self._fit()
        period_ok = abs(self.period / self.nominal_period - 1) < self.max_period_error
        if period_ok and np.abs(resid).max() <= self.reject:
            self.locked = True
            return
        for _ in range(len(self.ns) // 2):
            self.ns.popleft()
            self.ts.popleft()
        self.a = None
        self.period = self.nominal_period

    def _fit(self):
        n = np.fromiter(self.ns, float, len(self.ns))
        t = np.fromiter(self.ts, float, len(self.ts))
        nm, tm = n.mean(), t.mean()
        dn = n - nm
        self.period = float(np.dot(dn, t - tm) / np.dot(dn, dn))
        self.a = float(tm - self.period * nm)
        return t - (self.a + self.period * n)
