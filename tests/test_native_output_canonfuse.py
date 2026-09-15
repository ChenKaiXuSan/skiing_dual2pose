"""Tests for the proposed method's isolated internal coordinate transform."""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from dual2pose.eval.native_output_canonfuse import (
    LeftInputTransform,
    canonicalize_left_with_transform,
    canonfuse_native_admission,
    inverse_left_transform,
    predict_native_canonfuse,
    prepare_canonfuse_inputs,
)


class NativeOutputCanonFuseTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        self.left = rng.normal(size=(3, 30, 13, 3)).astype(np.float32)
        self.right = rng.normal(size=(3, 30, 13, 3)).astype(np.float32)
        # Make every first-frame body basis nondegenerate and sample-specific.
        for index in range(3):
            self.left[index, 0, 4] = [-1.0 - index, float(index), 0.2 * index]
            self.left[index, 0, 5] = [1.0 + index, float(index), 0.2 * index]
            self.left[index, 0, 12] = [0.2 * index, 2.0, 0.5]
            self.right[index, 0, 4] = [-0.5, -index, 0.0]
            self.right[index, 0, 5] = [0.5, index + 0.5, 0.0]
            self.right[index, 0, 12] = [0.0, 1.0, 1.0 + index]

    def test_inverse_left_transform_round_trips_every_sequence(self) -> None:
        transformed, transform = canonicalize_left_with_transform(
            self.left, left_hip=4, right_hip=5, neck=12
        )
        recovered = inverse_left_transform(transformed, transform)
        np.testing.assert_allclose(recovered, self.left, atol=2e-6, rtol=2e-6)
        self.assertEqual(transform.pelvis.shape, (3, 3))
        self.assertEqual(transform.linear.shape, (3, 3, 3))

    def test_independent_sequences_do_not_share_first_sample_transform(self) -> None:
        _, transform = canonicalize_left_with_transform(
            self.left, left_hip=4, right_hip=5, neck=12
        )
        self.assertFalse(np.array_equal(transform.pelvis[0], transform.pelvis[1]))
        self.assertFalse(np.array_equal(transform.linear[0], transform.linear[1]))

    def test_inverse_uses_matrix_solve_not_transpose_assumption(self) -> None:
        linear = np.array(
            [[[1.0, 0.4, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]],
            dtype=np.float64,
        )
        pelvis = np.array([[3.0, -2.0, 1.0]], dtype=np.float64)
        native = self.left[:1].astype(np.float64)
        transformed = (native - pelvis[:, None, None, :]) @ linear[:, None]
        recovered = inverse_left_transform(
            transformed, LeftInputTransform(pelvis=pelvis, linear=linear)
        )
        np.testing.assert_allclose(recovered, native, atol=1e-12, rtol=1e-12)

    def test_prepare_inputs_uses_only_left_and_right(self) -> None:
        self.assertNotIn("target", inspect.signature(prepare_canonfuse_inputs).parameters)
        (left_transformed, right_transformed), left_transform = prepare_canonfuse_inputs(
            self.left, self.right, left_hip=4, right_hip=5, neck=12
        )
        self.assertEqual(left_transformed.shape, self.left.shape)
        self.assertEqual(right_transformed.shape, self.right.shape)
        np.testing.assert_allclose(
            inverse_left_transform(left_transformed, left_transform),
            self.left,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_public_prediction_api_has_no_target_argument(self) -> None:
        self.assertNotIn("target", inspect.signature(predict_native_canonfuse).parameters)

    def test_current_model_contract_does_not_claim_left_native_gauge(self) -> None:
        decision = canonfuse_native_admission()
        self.assertFalse(decision["gauge_supported_by_model_contract"])
        self.assertFalse(decision["native_mpjpe_admitted"])
        self.assertFalse(decision["native_pa_mpjpe_admitted"])
        self.assertFalse(decision["native_acceleration_admitted"])
        self.assertIn("target canonical", decision["reason"].lower())


if __name__ == "__main__":
    unittest.main()
