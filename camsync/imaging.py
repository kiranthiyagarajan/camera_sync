"""Image helpers for the rig tools: decoding, exposure analysis, focus, camera identification."""
import cv2
import numpy as np

from .config import max_exposure

SATURATED = 250  # pixel value treated as clipped


def decode(data, reduce=1):
    """MJPEG bytes -> grayscale array. The camera sends YUV JPEGs with empty colour
    channels (mono sensor), so grayscale decoding loses nothing and skips the chroma."""
    flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
            4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[reduce]
    return cv2.imdecode(np.frombuffer(data, np.uint8), flag)


def downscale2(gray):
    return cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)


def saturation_mask2(gray):
    """Clipped-pixel mask at half resolution; max-pooled so a single clipped pixel still shows."""
    h, w = gray.shape[0] // 2 * 2, gray.shape[1] // 2 * 2
    m = gray[:h, :w] >= SATURATED
    return m.reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))


def histogram_image(gray, w=170, h=64):
    """Log-scaled brightness histogram as a BGR image; the clipped range is drawn red."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = np.log1p(hist)
    hist = hist / hist.max() if hist.max() > 0 else hist
    img = np.full((h, w, 3), 20, np.uint8)
    for v in range(256):
        x = v * w // 256
        y = int(round(hist[v] * (h - 2)))
        if y:
            color = (60, 60, 255) if v >= SATURATED else (220, 220, 220)
            cv2.line(img, (x, h - 1), (x, h - 1 - y), color, 1)
    return img


def focus_score(gray):
    """Sharpness of the centre half of the image (variance of the Laplacian); higher is sharper.
    Only meaningful relative to itself: turn the focus ring to maximise it."""
    h, w = gray.shape
    crop = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def exposure_metrics(gray, mask=None):
    """Brightness statistics; pixels where `mask` is True (static reflections) are ignored."""
    # Median filter removes isolated hot pixels but keeps any marker of 5+ pixels.
    filtered = cv2.medianBlur(gray, 3)
    if mask is not None:
        keep = ~mask
        gray, filtered = gray[keep], filtered[keep]
    return {
        "peak": int(filtered.max()),
        "p995": float(np.percentile(gray, 99.5)),
        "median": float(np.median(gray)),
        "clipped": float((filtered >= SATURATED).mean()),
        "bright": float((gray >= 128).mean()),
    }


class AutoExposure:
    """Finds exposure/gain for one camera, one adjustment per call to step().

    marker: reflective markers bright but not clipped (median-filtered peak in
            215-250), exposure capped at 2 ms to limit motion blur, gain only if needed.
    board:  calibration board well exposed (99.5th percentile in 180-220, <0.1% clipped),
            up to the full frame time.
    A reflection mask (see mask.py) keeps rails and IR rings from being taken for markers.
    Brightness is roughly linear in exposure because gamma is fixed at 1.0. Gain is
    reset to 0 and only added in coarse steps once exposure is at its cap; exposure
    then does the fine adjustment.
    """

    MODES = {
        "marker": dict(metric="peak", target=235, low=215, high=250, max_exposure=20),
        "board": dict(metric="p995", target=200, low=180, high=220, max_exposure=None),
    }
    GAIN_STEP = 8

    def __init__(self, mode, exposure, gain, fps=120, max_gain=63, max_iter=15, mask=None):
        self.mode = mode
        self.mask = mask  # True = static reflection, ignored when judging brightness
        self.p = dict(self.MODES[mode])
        frame_cap = max_exposure(fps)  # exposure can't exceed one frame
        self.p["max_exposure"] = min(self.p["max_exposure"] or frame_cap, frame_cap)
        self.exposure = int(exposure)
        self.gain = int(gain)
        self.max_gain = max_gain
        self.max_iter = max_iter
        self.iter = 0
        self.done = False
        self.ok = False
        self.message = "adjusting"
        self.metrics = None

    def step(self, gray):
        """Analyse a frame taken with the current settings. Returns the new (exposure, gain)
        to apply, or None when finished (see .ok and .message)."""
        if self.iter == 0 and self.gain > 0:
            # Start from no gain (less noise); this frame used the old gain, so skip it.
            self.iter, self.gain = 1, 0
            return self.exposure, self.gain
        m = self.metrics = exposure_metrics(gray, self.mask)
        p = self.p
        value = m[p["metric"]]
        too_bright = (m["clipped"] > 0) if self.mode == "marker" else (m["clipped"] > 0.001)
        if p["low"] <= value <= p["high"] and not too_bright:
            return self._finish(True, f"OK ({p['metric']} {value:.0f})")
        self.iter += 1
        if self.iter > self.max_iter:
            return self._finish(False, f"did not converge ({p['metric']} {value:.0f})")

        if too_bright or value > p["high"]:
            # Exposure gives fine control; gain only comes off once exposure is at minimum.
            if self.exposure > 1:
                factor = 0.6 if too_bright else p["target"] / value
                self.exposure = max(1, min(self.exposure - 1, int(self.exposure * factor)))
            elif self.gain > 0:
                self.gain = max(0, self.gain - self.GAIN_STEP)
            else:
                return self._finish(False, "too bright even at the shortest exposure")
        else:
            want = self.exposure * min(4.0, p["target"] / max(value, 1.0))
            if self.exposure < p["max_exposure"]:
                self.exposure = min(p["max_exposure"], max(self.exposure + 1, int(want)))
            elif self.gain < self.max_gain:
                self.gain = min(self.max_gain, self.gain + self.GAIN_STEP)
            else:
                return self._finish(False, "too dark at maximum exposure and gain")
        return self.exposure, self.gain

    def _finish(self, ok, message):
        self.done, self.ok = True, ok
        warn = ""
        if self.mode == "marker" and self.metrics and self.metrics["bright"] > 0.01:
            warn = f"; {self.metrics['bright']:.1%} of the image is bright - reflections?"
        self.message = message + warn
        return None


class CoverDetector:
    """Identifies which camera the user covers (or shines a light into).

    A camera is picked when its mean brightness stays far from its baseline
    (relative change >= 40% or >= 20 grey levels) for `hold` seconds, and no other
    unassigned camera changed as much.
    """

    def __init__(self, hold=1.0, rel=0.4, abs_=20.0):
        self.hold, self.rel, self.abs = hold, rel, abs_
        self.baseline = {}
        self._since = {}

    def set_baseline(self, means):
        self.baseline = dict(means)
        self._since.clear()

    def changed(self, cam, mean):
        b = self.baseline[cam]
        return abs(mean - b) >= max(self.abs, self.rel * b) if b > 0 else mean >= self.abs

    def update(self, means, now, candidates):
        """means: cam -> current mean brightness. Returns the identified camera or None."""
        hits = [c for c in candidates if self.changed(c, means[c])]
        for c in list(self._since):
            if c not in hits:
                del self._since[c]
        if len(hits) != 1:
            if len(hits) > 1:
                self._since.clear()  # ambiguous (lights changed?) - wait
            return None
        cam = hits[0]
        self._since.setdefault(cam, now)
        return cam if now - self._since[cam] >= self.hold else None

    def released(self, means, cam):
        return not self.changed(cam, means[cam])
