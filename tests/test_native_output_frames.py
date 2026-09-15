"""Coordinate-frame tests for native-output scoring."""
from __future__ import annotations

import hashlib
import inspect
import unittest

import numpy as np

from dual2pose.eval.native_output_frames import (
    CoordinateTransform,
    apply_target_to_prediction_frame,
    decide_common_frame,
    transform_pair,
    transform_points,
)


class NativeOutputFramesTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(41)
        self.prediction = rng.normal(size=(2, 30, 13, 3))
        self.target = rng.normal(size=(2, 30, 13, 3))
        self.rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.translation = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        self.source_sha = hashlib.sha256(b"calibration").hexdigest()

    def test_rigid_reporting_transform_is_applied_to_both_arrays(self) -> None:
        pred, target = transform_pair(
            self.prediction,
            self.target,
            rotation=self.rotation,
            translation=self.translation,
        )
        np.testing.assert_allclose(pred, self.prediction @ self.rotation + self.translation)
        np.testing.assert_allclose(target, self.target @ self.rotation + self.translation)

    def test_unit_conversion_changes_prediction_and_target_together(self) -> None:
        pred, target = transform_pair(
            self.prediction,
            self.target,
            rotation=np.eye(3),
            translation=np.zeros(3),
            scale=0.001,
        )
        np.testing.assert_allclose(pred, self.prediction * 0.001)
        np.testing.assert_allclose(target, self.target * 0.001)

    def test_target_to_prediction_transform_does_not_modify_prediction(self) -> None:
        transform = CoordinateTransform(
            source_frame="world_m",
            destination_frame="left_camera_m",
            rotation=self.rotation,
            translation=self.translation,
            scale=1.0,
            source_sha256=self.source_sha,
            transform_kind="camera_extrinsic",
        )
        pred, target = apply_target_to_prediction_frame(
            self.prediction, self.target, transform
        )
        np.testing.assert_array_equal(pred, self.prediction)
        np.testing.assert_allclose(target, self.target @ self.rotation + self.translation)

    def test_public_api_has_no_target_fit_mode(self) -> None:
        forbidden = {
            "fit",
            "procrustes",
            "landmarks",
            "target_rotation",
            "target_scale",
        }
        for function in (transform_pair, apply_target_to_prediction_frame):
            self.assertTrue(
                forbidden.isdisjoint(inspect.signature(function).parameters)
            )

    def test_improper_rotation_and_nonpositive_scale_are_rejected(self) -> None:
        reflection = np.diag([-1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            transform_points(self.prediction, reflection, np.zeros(3))
        with self.assertRaisesRegex(ValueError, "positive scale"):
            transform_points(self.prediction, np.eye(3), np.zeros(3), scale=0.0)

    def test_same_frame_is_comparable_without_conversion(self) -> None:
        decision = decide_common_frame(
            method_frame="left_camera_m",
            target_frame="left_camera_m",
            supplied_transform=None,
        )
        self.assertEqual(decision.status, "comparable")
        self.assertEqual(decision.reporting_frame, "left_camera_m")
        self.assertIsNone(decision.reason)

    def test_ski_dlt_is_unavailable_without_reference_world_mapping(self) -> None:
        decision = decide_common_frame(
            method_frame="calibrated_world_m",
            target_frame="constructed_camera_local_m",
            supplied_transform=None,
        )
        self.assertEqual(decision.status, "unavailable")
        self.assertIn("no supplied frame mapping", decision.reason)

    def test_supplied_mapping_requires_independent_provenance(self) -> None:
        transform = CoordinateTransform(
            source_frame="world_m",
            destination_frame="left_camera_m",
            rotation=self.rotation,
            translation=self.translation,
            scale=1.0,
            source_sha256="",
            transform_kind="camera_extrinsic",
        )
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            decide_common_frame(
                method_frame="left_camera_m",
                target_frame="world_m",
                supplied_transform=transform,
            )

    def test_body_or_target_fitted_mapping_is_rejected(self) -> None:
        for kind in ("body_canonical", "target_procrustes"):
            with self.subTest(kind=kind):
                transform = CoordinateTransform(
                    source_frame="world_m",
                    destination_frame="left_camera_m",
                    rotation=self.rotation,
                    translation=self.translation,
                    scale=1.0,
                    source_sha256=self.source_sha,
                    transform_kind=kind,
                )
                with self.assertRaisesRegex(ValueError, "camera|observation"):
                    decide_common_frame(
                        method_frame="left_camera_m",
                        target_frame="world_m",
                        supplied_transform=transform,
                    )


if __name__ == "__main__":
    unittest.main()
