"""Metric-contract tests for the native-output comparison."""
from __future__ import annotations

import unittest

import numpy as np

from dual2pose.eval.native_output_protocol import (
    NativeMetricAccumulator,
    metric_result,
    root_center,
)


class NativeOutputProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260911)
        self.target = rng.normal(size=(2, 30, 13, 3)).astype(np.float64)
        self.rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def test_root_center_uses_mean_of_common13_hips_per_frame(self) -> None:
        centred = root_center(self.target)
        expected_root = 0.5 * (self.target[..., 4, :] + self.target[..., 5, :])
        np.testing.assert_allclose(
            centred, self.target - expected_root[..., None, :], atol=0.0, rtol=0.0
        )
        np.testing.assert_allclose(
            0.5 * (centred[..., 4, :] + centred[..., 5, :]), 0.0, atol=1e-15
        )

    def test_translation_is_removed_but_rotation_and_scale_are_not(self) -> None:
        translated = self.target + np.array([8.0, -3.0, 2.0])
        self.assertAlmostEqual(metric_result(translated, self.target)["mpjpe"], 0.0)
        self.assertGreater(metric_result(self.target @ self.rotation, self.target)["mpjpe"], 0.1)
        self.assertGreater(metric_result(2.0 * self.target, self.target)["mpjpe"], 0.1)

    def test_pa_does_not_change_mpjpe_or_acceleration(self) -> None:
        prediction = 1.4 * (self.target @ self.rotation) + np.array([2.0, 1.0, -4.0])
        with_pa = metric_result(prediction, self.target)
        without_pa = metric_result(prediction, self.target, compute_pa=False)
        self.assertEqual(with_pa["mpjpe"], without_pa["mpjpe"])
        self.assertEqual(with_pa["acceleration_error"], without_pa["acceleration_error"])
        self.assertLess(with_pa["pa_mpjpe"], with_pa["mpjpe"])
        self.assertIsNone(without_pa["pa_mpjpe"])

    def test_acceleration_never_crosses_sequence_boundaries(self) -> None:
        target = np.zeros((2, 30, 13, 3), dtype=np.float64)
        prediction = target.copy()
        prediction[1, :, 0, 0] = 100.0
        accumulator = NativeMetricAccumulator()
        accumulator.update(prediction[:1], target[:1])
        accumulator.update(prediction[1:], target[1:])
        result = accumulator.result()
        self.assertEqual(result["acceleration_error"], 0.0)
        self.assertEqual(result["acceleration_point_count"], 2 * 28 * 13)

    def test_known_quadratic_acceleration_is_point_weighted(self) -> None:
        target = np.zeros((1, 30, 13, 3), dtype=np.float64)
        prediction = target.copy()
        prediction[0, :, 0, 0] = np.arange(30, dtype=np.float64) ** 2
        result = metric_result(prediction, target)
        self.assertAlmostEqual(result["acceleration_error"], 2.0 / 13.0)

    def test_reflection_is_not_removed_by_pa(self) -> None:
        reflected = self.target.copy()
        reflected[..., 0] *= -1.0
        result = metric_result(reflected, self.target)
        self.assertGreater(result["pa_mpjpe"], 0.05)

    def test_partial_batches_are_pooled_by_points(self) -> None:
        target_one = np.zeros((1, 30, 13, 3), dtype=np.float64)
        target_three = np.zeros((3, 30, 13, 3), dtype=np.float64)
        prediction_one = target_one.copy()
        prediction_three = target_three.copy()
        prediction_one[..., 0, 0] = 1.0
        prediction_three[..., 0, 0] = 3.0
        accumulator = NativeMetricAccumulator()
        accumulator.update(prediction_one, target_one)
        accumulator.update(prediction_three, target_three)
        expected = ((1.0 / 13.0) * 1 + (3.0 / 13.0) * 3) / 4
        self.assertAlmostEqual(accumulator.result()["mpjpe"], expected)

    def test_invalid_shape_and_nonfinite_input_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "30 frames and 13 joints"):
            metric_result(np.zeros((1, 29, 13, 3)), np.zeros((1, 29, 13, 3)))
        invalid = self.target.copy()
        invalid[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            metric_result(invalid, self.target)


if __name__ == "__main__":
    unittest.main()
