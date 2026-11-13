import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svr_depth.datasets.kitti_sceneflow import _select_protocol_frames


class TestKITTIFrameSelection(unittest.TestCase):

    def make_rows(self, n):
        return [(i, Path(f"{i:02d}.png")) for i in range(n)]

    def test_21_frames_select_first_20(self):
        rows = self.make_rows(21)
        selected = _select_protocol_frames(rows)

        self.assertIsNotNone(selected)
        self.assertEqual([x[0] for x in selected], list(range(20)))

    def test_keep_both_labelled_frames(self):
        selected = _select_protocol_frames(self.make_rows(21))
        ids = [x[0] for x in selected]

        self.assertIn(10, ids)
        self.assertIn(11, ids)

    def test_exact_20_frames(self):
        selected = _select_protocol_frames(self.make_rows(20))
        self.assertEqual(len(selected), 20)

    def test_reject_missing_frame(self):
        rows = self.make_rows(21)
        rows.pop(5)
        self.assertIsNone(_select_protocol_frames(rows))

    def test_reject_abnormal_frame_ids(self):
        rows = self.make_rows(21)
        rows[-1] = (25, Path("25.png"))
        self.assertIsNone(_select_protocol_frames(rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
