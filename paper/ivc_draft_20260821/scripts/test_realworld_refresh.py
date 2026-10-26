"""Checks for deterministic, geometry-preserving qualitative rendering."""
import unittest
import numpy as np
from generate_realworld_refresh import select_midpoint, display_body, display_limits


class RefreshTest(unittest.TestCase):
    def test_selection_uses_shared_frame_index_not_scores(self):
        self.assertEqual(select_midpoint([4, 8, 12, 16], [0, 4, 8, 12]), 8)
        with self.assertRaises(ValueError):
            select_midpoint([1], [2])

    def test_display_drops_only_eyes_and_permutates_axes(self):
        pose = np.arange(45).reshape(15, 3)
        body = display_body(pose)
        self.assertEqual(body.shape, (13, 3))
        np.testing.assert_array_equal(body[0], [6, 8, 7])
        np.testing.assert_array_equal(body[-1], [42, 44, 43])

    def test_limits_enclose_all_outputs_without_per_method_scaling(self):
        poses = np.zeros((2, 4, 15, 3))
        poses[1, 3, 8] = [10, -20, 30]
        center, radius = display_limits(poses)
        points = poses[..., 2:, :][..., [0, 2, 1]]
        self.assertTrue(np.all(np.abs(points - center) <= radius))


if __name__ == "__main__":
    unittest.main()
