"""Tests for the rig tools' logic: stats, config, auto exposure, camera identification."""
import unittest

import numpy as np

from camsync import config
from camsync.imaging import AutoExposure, CoverDetector
from camsync.stats import CamStats
from camsync.v4l2 import CameraInfo


class StatsTest(unittest.TestCase):
    def test_window_and_loss(self):
        st = CamStats()
        for sec in range(10):
            for _ in range(118):
                st.add(now=100 + sec + 0.5)
            st.add(frames=0, lost=2, now=100 + sec + 0.9)
        st.tick(now=110.1)
        fps, loss = st.window(5)
        self.assertEqual(fps, 118)
        self.assertAlmostEqual(loss, 2 / 120)
        self.assertEqual(st.total_lost, 20)

    def test_stalled_camera_shows_empty_seconds(self):
        st = CamStats()
        st.add(now=10.5)  # first (partial) second is not recorded
        st.add(now=11.5)
        st.tick(now=15.2)
        self.assertEqual([s[1] for s in st.seconds(10)], [1, 0, 0, 0])
        self.assertEqual(st.window(3)[0], 0)


class ConfigTest(unittest.TestCase):
    def test_resolve_by_port(self):
        cfg = config.default_config()
        cfg["cameras"] = {"cam1": "3-1", "cam2": "3-3", "cam9": "3-9"}
        infos = [CameraInfo("/dev/video4", "s", "3-3"), CameraInfo("/dev/video2", "s", "3-1"),
                 CameraInfo("/dev/video8", "s", "3-6.2")]
        cams, missing, unassigned = config.resolve(cfg, infos)
        self.assertEqual([c.id for c in cams], ["cam1", "cam2", "3-6.2"])
        self.assertEqual(missing, ["cam9"])
        self.assertEqual([c.port for c in unassigned], ["3-6.2"])

    def test_controls_precedence(self):
        cfg = config.default_config()
        cfg["controls"] = {"all": {"gain": 5}, "cam2": {"gain": 9, "exposure_time_absolute": 12}}
        self.assertEqual(config.controls_for(cfg, "cam1")["gain"], 5)
        c2 = config.controls_for(cfg, "cam2")
        self.assertEqual((c2["gain"], c2["exposure_time_absolute"]), (9, 12))
        self.assertEqual(list(c2)[0], "auto_exposure")  # manual mode is set first


def fake_camera(exposure, gain, marker_rate, background_rate, rng, board=False):
    """Brightness proportional to exposure x gain factor, clipped at 255, with noise."""
    k = exposure * (1 + gain / 16)
    img = np.full((360, 640), background_rate * k)
    if board:
        img[(np.indices(img.shape) // 40).sum(axis=0) % 2 == 1] = marker_rate * k
    else:
        for y, x in ((100, 100), (200, 300), (250, 500)):
            img[y:y + 7, x:x + 7] = marker_rate * k
    img += rng.normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def run_auto(mode, marker_rate, background_rate, exposure=50, gain=0, board=False):
    rng = np.random.default_rng(0)
    ae = AutoExposure(mode, exposure, gain)
    for _ in range(40):
        res = ae.step(fake_camera(ae.exposure, ae.gain, marker_rate, background_rate, rng, board))
        if res is None:
            return ae
    raise AssertionError("auto exposure never finished")


class AutoExposureTest(unittest.TestCase):
    def test_marker_mode_unclips_bright_markers(self):
        ae = run_auto("marker", marker_rate=40, background_rate=0.5)  # clipped at 5 ms
        self.assertTrue(ae.ok, ae.message)
        self.assertTrue(215 <= ae.metrics["peak"] <= 250)
        self.assertEqual(ae.metrics["clipped"], 0)

    def test_marker_mode_caps_exposure_then_uses_gain(self):
        ae = run_auto("marker", marker_rate=6, background_rate=0.2, exposure=5)
        self.assertTrue(ae.ok, ae.message)
        self.assertEqual(ae.exposure, 20)  # 2 ms motion-blur cap
        self.assertGreater(ae.gain, 0)

    def test_marker_mode_reports_too_dark(self):
        ae = run_auto("marker", marker_rate=0.5, background_rate=0.1, exposure=5)
        self.assertFalse(ae.ok)
        self.assertIn("too dark", ae.message)

    def test_exposure_cap_follows_frame_rate(self):
        self.assertEqual(AutoExposure("board", 50, 0, fps=120).p["max_exposure"], 83)
        self.assertEqual(AutoExposure("board", 50, 0, fps=60).p["max_exposure"], 165)
        self.assertEqual(AutoExposure("marker", 50, 0, fps=60).p["max_exposure"], 20)  # blur cap

    def test_board_mode(self):
        ae = run_auto("board", marker_rate=2.0, background_rate=0.6, exposure=10, board=True)
        self.assertTrue(ae.ok, ae.message)
        self.assertTrue(180 <= ae.metrics["p995"] <= 220)


class CoverDetectorTest(unittest.TestCase):
    def test_picks_covered_camera_after_hold(self):
        d = CoverDetector(hold=1.0)
        d.set_baseline({"a": 80, "b": 90, "c": 70})
        means = {"a": 80, "b": 12, "c": 71}
        self.assertIsNone(d.update(means, 0.0, ["a", "b", "c"]))
        self.assertIsNone(d.update(means, 0.6, ["a", "b", "c"]))
        self.assertEqual(d.update(means, 1.1, ["a", "b", "c"]), "b")
        self.assertFalse(d.released(means, "b"))
        self.assertTrue(d.released({"a": 80, "b": 88, "c": 70}, "b"))

    def test_ambiguous_change_is_ignored(self):
        d = CoverDetector(hold=0.5)
        d.set_baseline({"a": 80, "b": 90})
        dark = {"a": 10, "b": 10}  # room lights off: every camera changes
        for t in (0, 0.6, 1.2):
            self.assertIsNone(d.update(dark, t, ["a", "b"]))

    def test_light_into_dark_camera(self):
        d = CoverDetector(hold=0.2)
        d.set_baseline({"a": 5, "b": 6})
        bright = {"a": 5, "b": 60}
        d.update(bright, 0, ["a", "b"])
        self.assertEqual(d.update(bright, 0.3, ["a", "b"]), "b")


if __name__ == "__main__":
    unittest.main()
