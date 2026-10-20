import unittest

import numpy as np

from camsync import pose


def disk(normal, direction):
    return {"normal": None if normal is None else np.asarray(normal, float),
            "direction": np.asarray(direction, float)}


class BendTest(unittest.TestCase):
    def test_segment_and_total_angles(self):
        c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
        ds = [disk([0, 0, 1], [0, 0, 1]),
              disk([-s, 0, -c], [s, 0, c]),  # flipped normal is turned to the backbone
              disk(None, [1, 0, 0])]
        segs, total = pose.bend_angles(ds)
        self.assertAlmostEqual(segs[0][0], 30)
        self.assertTrue(segs[0][1])
        self.assertAlmostEqual(segs[1][0], 60)
        self.assertFalse(segs[1][1])
        self.assertAlmostEqual(total[0], 90)
        self.assertFalse(total[1])

    def test_single_disk(self):
        self.assertEqual(pose.bend_angles([disk([0, 0, 1], [0, 0, 1])]), ([], (None, False)))


if __name__ == "__main__":
    unittest.main()
