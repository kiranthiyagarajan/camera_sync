"""Simulated 4-camera rig: independent clocks, USB timestamp jitter, dropped frames."""
import random
import unittest

from camsync.clock import LinearClock
from camsync.sync import Frame, Synchronizer

PERIOD = 0.0084363  # measured on the real camera (118.54 fps)


def simulate(seconds=20, cams=4, drop_rate=0.005, jitter=0.002, seed=14):
    rng = random.Random(seed)
    events = []  # (arrival_time, cam, true_t, raw_t)
    for c in range(cams):
        period = PERIOD * (1 + rng.uniform(-50e-6, 50e-6))  # crystal tolerance
        t = rng.uniform(0, PERIOD)
        while t < seconds:
            if rng.random() > drop_rate:
                raw = t + rng.uniform(-jitter, jitter)
                events.append((raw + 0.001, f"cam{c}", t, raw))
            t += period
    events.sort()
    return events


class SyncTest(unittest.TestCase):
    def run_rig(self, **kw):
        events = simulate(**kw)
        ids = sorted({e[1] for e in events})
        clocks = {c: LinearClock(PERIOD * 1.012) for c in ids}  # driver claims 120 fps
        sync = Synchronizer(ids, PERIOD)
        sets, clock_err, last_raw, all_warm = [], [], {}, False
        for _, cam, true_t, raw in events:
            assert raw > last_raw.get(cam, -1)  # per-camera order holds after jitter
            last_raw[cam] = raw
            n, t = clocks[cam].update(raw)
            if clocks[cam].warm:
                clock_err.append(abs(t - true_t))
            all_warm = all_warm or all(c.warm for c in clocks.values())  # same gate as Rig
            if all_warm:
                f = Frame(cam, n, t, raw, b"")
                f.true_t = true_t
                sets.extend(sync.push(f))
        return ids, clocks, sets, clock_err

    def test_clock_removes_jitter(self):
        _, clocks, _, err = self.run_rig()
        err.sort()
        self.assertLess(err[len(err) // 2], 0.0002)   # median well under 1 ms jitter
        self.assertLess(err[int(len(err) * 0.99)], 0.0006)
        for c in clocks.values():
            self.assertAlmostEqual(c.period, PERIOD, delta=PERIOD * 1e-3)

    def test_drops_detected(self):
        events = simulate()
        ids = sorted({e[1] for e in events})
        _, clocks, _, _ = self.run_rig()
        for c in ids:
            ts = [e[2] for e in events if e[1] == c]
            gaps = [(b, round((b - a) / PERIOD) - 1) for a, b in zip(ts, ts[1:])]
            total = sum(g for _, g in gaps)
            # Drops before the clock locks (first ~0.3 s, longer if warm-up restarts) aren't counted.
            after_warmup = sum(g for t, g in gaps if t > 2.0)
            self.assertLessEqual(clocks[c].dropped, total, c)
            self.assertGreaterEqual(clocks[c].dropped, after_warmup, c)

    def test_sets_are_nearest_and_bounded(self):
        ids, _, sets, _ = self.run_rig()
        sync_tolerance = Synchronizer(ids, PERIOD).tolerance
        self.assertGreater(len(sets), 2000)
        complete = sum(s.complete for s in sets)
        self.assertGreater(complete / len(sets), 0.97)
        for s in sets:
            for c in ids:
                f = s.frames[c]
                if f is None:
                    continue
                # within the tolerance always; within half a period when nothing was dropped
                self.assertLessEqual(abs(s.offset(c)), PERIOD * sync_tolerance)
                if s.brackets[c] is not None:
                    self.assertLessEqual(abs(s.offset(c)), PERIOD / 2 + 0.0006)
                w = s.weight(c)
                if w is not None:
                    self.assertTrue(-0.1 <= w <= 1.1)
                    before, after = s.brackets[c]
                    true_interp = before.true_t + w * (after.true_t - before.true_t)
                    ref_true = s.frames[ids[0]].true_t
                    # interpolating to the set time lands on the true moment, up to the
                    # combined clock-model error of this camera and the reference
                    self.assertLess(abs(true_interp - ref_true), 0.001)

    def test_clock_recovers_after_stall(self):
        # A 1 s stall (e.g. USB hiccup) resets the model; it must re-lock and be accurate again.
        events = [e for e in simulate(seconds=6, cams=1) if not (2.0 < e[2] < 3.0)]
        clock = LinearClock(PERIOD * 1.012)
        errs_after = []
        for _, _, true_t, raw in events:
            _, t = clock.update(raw)
            if clock.warm and true_t > 4.5:
                errs_after.append(abs(t - true_t))
        self.assertGreater(len(errs_after), 100)
        self.assertLess(max(errs_after), 0.0008)

    def test_crawling_camera_counts_its_losses(self):
        # A camera delivering one frame per second never re-locks; every gap is still loss.
        clock = LinearClock(PERIOD)
        t = 0.0
        for _ in range(40):  # healthy start, then a frame every second
            clock.update(t)
            t += PERIOD
        for _ in range(10):
            t += 1.0
            clock.update(t)
        self.assertGreater(clock.dropped, 10 * 110)

    def test_dead_camera_times_out(self):
        events = [e for e in simulate(seconds=3) if not (e[1] == "cam2" and e[2] > 1.5)]
        ids = sorted({e[1] for e in events})
        clocks = {c: LinearClock(PERIOD) for c in ids}
        sync = Synchronizer(ids, PERIOD)
        sets = []
        for _, cam, _, raw in events:
            n, t = clocks[cam].update(raw)
            if clocks[cam].warm:
                sets.extend(sync.push(Frame(cam, n, t, raw, b"")))
        late = [s for s in sets if s.t > 1.6]
        self.assertGreater(len(late), 100)
        self.assertTrue(all(s.frames["cam2"] is None for s in late))
        self.assertTrue(all(s.frames["cam0"] is not None for s in late))


if __name__ == "__main__":
    unittest.main()
