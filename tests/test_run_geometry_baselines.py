from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dual2pose.experiments.run_main_baselines import MetricAccumulator
from dual2pose.experiments.run_geometry_baselines import (
    _require_new_paths,
    _validate_array_file,
)
from dual2pose.eval.main_baseline_data import sha256
from dual2pose.models.dlt_residual_baseline import (
    DLTResidualMLP,
    dlt_residual_features,
    dlt_residual_loss,
    ski_predicted_pelvis_relative,
)


class DLTResidualModelTests(unittest.TestCase):
    """Catch target leakage, wrong feature layout, and a non-residual head."""

    def test_zero_output_head_is_exact_dlt_identity(self) -> None:
        torch.manual_seed(3)
        dlt = torch.randn(2, 5, 13, 3)
        reprojection = torch.rand(2, 5, 13) * 50.0
        valid = torch.ones(2, 5, 13, dtype=torch.bool)
        model = DLTResidualMLP(num_joints=13).eval()

        actual = model(dlt, reprojection, valid)

        self.assertTrue(torch.equal(actual, dlt))

    def test_features_are_full_frame_dlt_log_reprojection_and_validity(self) -> None:
        dlt = torch.tensor(
            [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=torch.float32
        )
        reprojection = torch.tensor([[[0.0, 25.0]]], dtype=torch.float32)
        valid = torch.tensor([[[True, False]]])

        actual = dlt_residual_features(dlt, reprojection, valid)

        expected = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.69314718056, 1.0, 0.0]]]
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=0.0))

    def test_small_synthetic_residual_mapping_learns_without_target_features(self) -> None:
        torch.manual_seed(17)
        dlt = torch.randn(24, 4, 2, 3)
        reprojection = torch.zeros(24, 4, 2)
        valid = torch.ones(24, 4, 2, dtype=torch.bool)
        target = dlt + torch.tensor([0.2, -0.1, 0.3])
        model = DLTResidualMLP(num_joints=2, hidden_size=32, dropout=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        initial = torch.nn.functional.l1_loss(model(dlt, reprojection, valid), target).item()

        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(dlt, reprojection, valid)
            loss = dlt_residual_loss(prediction, target)
            loss.backward()
            optimizer.step()

        final = torch.nn.functional.l1_loss(model(dlt, reprojection, valid), target).item()
        self.assertLess(final, initial * 0.35)

    def test_nonfinite_or_negative_geometry_features_are_rejected(self) -> None:
        model = DLTResidualMLP(num_joints=2)
        dlt = torch.zeros(1, 3, 2, 3)
        reprojection = torch.zeros(1, 3, 2)
        valid = torch.ones(1, 3, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "finite"):
            model(dlt + float("nan"), reprojection, valid)
        reprojection[0, 0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            model(dlt, reprojection, valid)

    def test_ski_predicted_pelvis_centering_removes_translation_trajectory(self) -> None:
        torch.manual_seed(29)
        body = torch.randn(2, 5, 13, 3)
        trajectory = torch.randn(2, 5, 1, 3) * 20.0
        translated = body + trajectory
        body_pelvis = (body[:, :, 4:5] + body[:, :, 5:6]) * 0.5

        actual = ski_predicted_pelvis_relative(translated)

        self.assertTrue(
            torch.allclose(actual, body - body_pelvis, atol=6e-6, rtol=0.0)
        )
        actual_pelvis = (actual[:, :, 4] + actual[:, :, 5]) * 0.5
        self.assertTrue(
            torch.allclose(
                actual_pelvis, torch.zeros_like(actual_pelvis), atol=2e-6, rtol=0.0
            )
        )


class SharedMetricAccumulatorTests(unittest.TestCase):
    """Catch per-batch averaging and acceleration across sample boundaries."""

    def test_partial_batches_are_weighted_by_points(self) -> None:
        metric = MetricAccumulator()
        for sample_count, error in ((4, 1.0), (1, 3.0)):
            target = torch.zeros(sample_count, 5, 13, 3)
            prediction = target.clone()
            prediction[..., 0] = error
            metric.update(prediction, target)

        result = metric.result()
        self.assertAlmostEqual(result["mpjpe"], 1.4, places=6)
        self.assertEqual(result["sample_count"], 5)
        self.assertEqual(result["acceleration_error"], 0.0)

    def test_acceleration_is_computed_inside_each_sequence(self) -> None:
        target = torch.zeros(2, 5, 13, 3)
        prediction = target.clone()
        prediction[0, :, :, 0] = torch.arange(5).square()[:, None]
        prediction[1, :, :, 0] = 100.0

        metric = MetricAccumulator()
        metric.update(prediction, target)

        self.assertAlmostEqual(metric.result()["acceleration_error"], 1.0)


class GeometryRunnerSafetyTests(unittest.TestCase):
    """Catch provenance bypasses and accidental overwrites of completed results."""

    def test_array_validation_rejects_hash_mismatch_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.npy"
            array = np.zeros((2, 3, 4, 3), dtype=np.float32)
            np.save(path, array)
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                _validate_array_file(
                    path,
                    expected_sha256="0" * 64,
                    expected_shape=array.shape,
                    expected_dtype=np.dtype("float32"),
                )

            array[0, 0, 0, 0] = np.nan
            np.save(path, array)
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                _validate_array_file(
                    path,
                    expected_sha256=sha256(path),
                    expected_shape=array.shape,
                    expected_dtype=np.dtype("float32"),
                )

    def test_completed_result_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "dlt.json"
            existing.write_text("{}")
            with self.assertRaisesRegex(FileExistsError, "Preserving existing result"):
                _require_new_paths([existing, Path(directory) / "robust_dlt.json"])


if __name__ == "__main__":
    unittest.main()
