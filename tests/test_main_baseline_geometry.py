import unittest
from pathlib import Path
import json
import tempfile

import numpy as np

from dual2pose.eval.main_baseline_geometry import (
    _sha256,
    _validate_existing_export,
    build_dlt_residual_features,
    project_points,
    reprojection_gated_dlt,
    triangulate_sequence,
    unity_projection_matrix,
)


class MainBaselineGeometryTest(unittest.TestCase):
    @staticmethod
    def _cameras(num_frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
        intrinsic = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        left = intrinsic @ np.hstack([np.eye(3), np.zeros((3, 1))])
        right = intrinsic @ np.hstack(
            [np.eye(3), np.array([[-0.6], [0.0], [0.0]])]
        )
        return (
            np.broadcast_to(left, (num_frames, 3, 4)).copy(),
            np.broadcast_to(right, (num_frames, 3, 4)).copy(),
        )

    def test_dlt_recovers_synthetic_calibrated_sequence(self) -> None:
        p_left, p_right = self._cameras()
        xyz = np.array(
            [
                [[0.0, 0.1, 5.0], [0.4, -0.2, 6.0]],
                [[0.1, 0.1, 5.1], [0.5, -0.2, 6.1]],
                [[0.2, 0.1, 5.2], [0.6, -0.2, 6.2]],
            ],
            dtype=np.float64,
        )
        uv_left = project_points(p_left, xyz)
        uv_right = project_points(p_right, xyz)

        result = triangulate_sequence(p_left, p_right, uv_left, uv_right)

        np.testing.assert_allclose(result.prediction, xyz, atol=1e-8)
        self.assertTrue(result.valid_mask.all())
        np.testing.assert_allclose(result.reprojection_error_px, 0.0, atol=1e-8)

    def test_unity_camera_json_accepts_dataset_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            camera_dir = Path(tmp) / "data_pole_ski/male/cameras/L0_A000"
            camera_dir.mkdir(parents=True)
            (camera_dir / "intrinsics.json").write_text(
                json.dumps({"fx": 10, "fy": 11, "cx": 2, "cy": 3}),
                encoding="utf-8-sig",
            )
            (camera_dir / "extrinsics.json").write_text(
                json.dumps({"t_world_cam_4x4": np.eye(4).reshape(-1).tolist()}),
                encoding="utf-8-sig",
            )

            projection = unity_projection_matrix(Path(tmp), "male", "capture_L0_A000")

        self.assertEqual(projection.shape, (3, 4))
        self.assertTrue(np.isfinite(projection).all())

    def test_existing_export_reuse_rejects_tampered_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            array_path = output / "dlt.npy"
            np.save(array_path, np.zeros((1, 1, 1, 3), dtype=np.float32))
            expected = {"dataset": "unity", "split": "test"}
            manifest = {
                **expected,
                "output_sha256": {"dlt.npy": _sha256(array_path)},
            }
            (output / "manifest.json").write_text(json.dumps(manifest))
            _validate_existing_export(output, {"dlt": "dlt.npy"}, expected)

            array_path.write_bytes(array_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _validate_existing_export(output, {"dlt": "dlt.npy"}, expected)


    def test_invalid_observations_are_retained_as_nan_with_false_validity(self) -> None:
        p_left, p_right = self._cameras(num_frames=1)
        xyz = np.array([[[0.0, 0.0, 4.0], [0.2, 0.1, 5.0]]])
        uv_left = project_points(p_left, xyz)
        uv_right = project_points(p_right, xyz)
        uv_left[0, 1] = np.nan

        result = triangulate_sequence(p_left, p_right, uv_left, uv_right)

        self.assertTrue(result.valid_mask[0, 0])
        self.assertFalse(result.valid_mask[0, 1])
        self.assertTrue(np.isnan(result.prediction[0, 1]).all())
        self.assertTrue(np.isnan(result.reprojection_error_px[0, 1]))

    def test_reprojection_gate_interpolates_and_falls_back_without_dropping_tracks(self) -> None:
        p_left, p_right = self._cameras(num_frames=3)
        xyz = np.array(
            [
                [[0.0, 0.0, 5.0], [0.2, 0.0, 5.0]],
                [[0.1, 0.0, 5.0], [0.3, 0.0, 5.0]],
                [[0.2, 0.0, 5.0], [0.4, 0.0, 5.0]],
            ]
        )
        uv_left = project_points(p_left, xyz)
        uv_right = project_points(p_right, xyz)
        # One interior rejection on joint 0, and every frame rejected on joint 1.
        uv_right[1, 0, 1] += 50.0
        uv_right[:, 1, 1] += 50.0

        result = reprojection_gated_dlt(
            p_left, p_right, uv_left, uv_right, threshold_px=1.0
        )

        self.assertFalse(result.gate_accepted[1, 0])
        self.assertTrue(result.interpolated_mask[1, 0])
        np.testing.assert_allclose(
            result.prediction[1, 0],
            0.5 * (result.ungated_prediction[0, 0] + result.ungated_prediction[2, 0]),
        )
        self.assertFalse(result.gate_accepted[:, 1].any())
        self.assertTrue(result.fallback_mask[:, 1].all())
        np.testing.assert_allclose(
            result.prediction[:, 1], result.ungated_prediction[:, 1]
        )
        self.assertTrue(result.valid_mask.all())

    def test_residual_features_use_full_dlt_pose_and_have_no_target_input(self) -> None:
        p_left, p_right = self._cameras(num_frames=2)
        xyz = np.array(
            [
                [[0.0, 0.0, 5.0], [0.2, 0.1, 5.0]],
                [[0.1, 0.0, 5.0], [0.3, 0.1, 5.0]],
            ]
        )
        uv_left = project_points(p_left, xyz)
        uv_right = project_points(p_right, xyz)
        dlt = triangulate_sequence(p_left, p_right, uv_left, uv_right)

        features = build_dlt_residual_features(
            dlt, uv_left, uv_right, image_size_px=(640.0, 480.0)
        )

        # full pose (J*3), two 2D observations, residual, validity, joint one-hot
        self.assertEqual(features.values.shape, (2, 2, 6 + 4 + 1 + 1 + 2))
        self.assertTrue(features.valid_mask.all())
        np.testing.assert_allclose(
            features.values[0, 0, :6], xyz[0].reshape(-1), atol=1e-7
        )
        np.testing.assert_allclose(
            features.values[0, 1, :6], xyz[0].reshape(-1), atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
