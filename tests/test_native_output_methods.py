"""Behavioral tests for pre-Canon controls and view-level pooling."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dual2pose.eval.native_output_methods import (
    align_right_to_left_sequence,
    aligned_average,
    aligned_quality_weighted,
    pooled_view_metrics,
    smoothnet_native_eligibility,
    unaligned_average,
)


class NativeOutputMethodsTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        self.left = rng.normal(size=(2, 30, 13, 3)).astype(np.float64)
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.right = 1.7 * (self.left @ rotation.T) + np.array([2.0, -4.0, 1.0])

    def test_unaligned_average_is_literal_average(self) -> None:
        np.testing.assert_allclose(
            unaligned_average(self.left, self.right), 0.5 * (self.left + self.right)
        )

    def test_sequence_alignment_uses_predictions_only_and_recovers_left(self) -> None:
        self.assertNotIn(
            "target", inspect.signature(align_right_to_left_sequence).parameters
        )
        aligned, transform = align_right_to_left_sequence(self.left, self.right)
        np.testing.assert_allclose(aligned, self.left, atol=1e-7, rtol=1e-7)
        self.assertEqual(transform["scale"].shape, (2,))
        self.assertTrue(np.all(transform["det_rotation"] > 0.0))

    def test_aligned_average_is_left_for_exactly_related_views(self) -> None:
        np.testing.assert_allclose(
            aligned_average(self.left, self.right), self.left, atol=1e-7, rtol=1e-7
        )

    def test_quality_weights_depend_only_on_inputs_and_sum_to_one(self) -> None:
        self.assertNotIn(
            "target", inspect.signature(aligned_quality_weighted).parameters
        )
        first_pose, first_weights = aligned_quality_weighted(self.left, self.right)
        second_pose, second_weights = aligned_quality_weighted(
            self.left.copy(), self.right.copy()
        )
        np.testing.assert_array_equal(first_pose, second_pose)
        np.testing.assert_array_equal(first_weights, second_weights)
        np.testing.assert_allclose(first_weights.sum(axis=1), 1.0)
        self.assertEqual(first_weights.shape, (2, 2))

    def test_view_pooling_uses_all_points_not_mean_of_view_metrics(self) -> None:
        left_gt = np.zeros((1, 30, 13, 3), dtype=np.float64)
        right_gt = np.zeros((3, 30, 13, 3), dtype=np.float64)
        left_pred = left_gt.copy()
        right_pred = right_gt.copy()
        left_pred[..., 0, 0] = 1.0
        right_pred[..., 0, 0] = 3.0
        result = pooled_view_metrics(left_pred, right_pred, left_gt, right_gt)
        expected = ((1.0 / 13.0) + 3.0 * (3.0 / 13.0)) / 4.0
        unweighted_view_mean = 0.5 * ((1.0 / 13.0) + (3.0 / 13.0))
        self.assertAlmostEqual(result["mpjpe"], expected)
        self.assertNotAlmostEqual(result["mpjpe"], unweighted_view_mean)
        self.assertEqual(result["view_count"], 2)
        self.assertEqual(result["sample_count"], 4)

    def test_common13_and_thirty_frame_shapes_are_enforced(self) -> None:
        invalid = np.zeros((1, 29, 13, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "30 frames and 13 joints"):
            unaligned_average(invalid, invalid)

    def test_canonical_trained_smoothnet_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "smoothnet.json"
            result.write_text(
                json.dumps(
                    {
                        "method": "smoothnet",
                        "provenance": {
                            "canonicalization": "independent left/right/target"
                        },
                    }
                ),
                encoding="utf-8",
            )
            decision = smoothnet_native_eligibility(result)
        self.assertFalse(decision["eligible"])
        self.assertIn("canonical", decision["reason"].lower())


if __name__ == "__main__":
    unittest.main()
