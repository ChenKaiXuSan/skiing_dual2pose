from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pytest
import torch

from dual2pose.experiments.run_main_baselines import MetricAccumulator
from dual2pose.experiments.run_geometry_baselines import (
    EXPECTED,
    _load_verified_split,
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


@pytest.fixture
def geometry_inputs(tmp_path, monkeypatch):
    """A complete on-disk export with small arrays and independent source indices."""
    monkeypatch.setitem(EXPECTED["unity"], "test", (2, 3, 15, 3))
    root = tmp_path / "export"
    cache = root / "cache/unity/test"
    geometry = root / "geometry/unity/test"
    cache.mkdir(parents=True)
    geometry.mkdir(parents=True)
    indices = tmp_path / "current"
    indices.mkdir()
    (indices / "cache.json").write_text('{"kind": "raw"}')
    (indices / "geometry.json").write_text('{"kind": "calibrated"}')
    for name in ("left", "right", "target"):
        np.save(cache / f"{name}.npy", np.ones((2, 3, 15, 3), dtype=np.float32))
    np.save(cache / "frames.npy", np.arange(6, dtype=np.int64).reshape(2, 3))
    (cache / "samples.jsonl").write_text('{}\n{}\n')
    cache_hashes = {path.name: sha256(path) for path in cache.iterdir()}
    cache_manifest = {
        "dataset": "unity", "split": "test", "sample_count": 2,
        "time_window": 3, "num_joints": 15,
        "representation": "raw_pre_batch_canonicalization",
        "files": cache_hashes, "index": str(indices / "cache.json"),
        "index_sha256": sha256(indices / "cache.json"),
    }
    files = {}
    for name in ("dlt", "robust", "dlt_valid", "robust_valid", "gate",
                 "reprojection", "interpolated", "fallback"):
        pose = name in {"dlt", "robust"}
        dtype = np.float32 if pose or name == "reprojection" else np.bool_
        np.save(geometry / f"{name}.npy", np.ones((2, 3, 15, 3) if pose else (2, 3, 15), dtype=dtype))
        files[name] = f"{name}.npy"
    geometry_manifest = {
        "schema_version": 1, "dataset": "unity", "split": "test",
        "shape": [2, 3, 15, 3], "samples": 2, "frames_per_sample": 3,
        "joints": 15, "coordinate_space": "calibrated_world_m",
        "estimated_2d_only": True, "gt_2d_used": False,
        "gt_fitted_transform_used": False, "gate_threshold_px": 25.0,
        "cache_frames_sha256": cache_hashes["frames.npy"],
        "cache_samples_sha256": cache_hashes["samples.jsonl"],
        "source_index": str(indices / "geometry.json"),
        "source_index_sha256": sha256(indices / "geometry.json"),
        "files": files,
        "output_sha256": {name: sha256(geometry / name) for name in files.values()},
        "counts": {"points": 90, "dlt_valid": 90, "robust_valid": 90},
    }
    manifests = [(cache / "manifest.json", cache_manifest, "index", "cache.json"),
                 (geometry / "manifest.json", geometry_manifest, "source_index", "geometry.json")]
    monkeypatch.delenv("DUAL2POSE_INPUT_PATH_MAP", raising=False)
    return root, indices, manifests


@pytest.mark.parametrize("relocated", [(), (0,), (1,), (0, 1)])
def test_geometry_loader_reads_relocated_indices_without_rewriting_manifests(
    geometry_inputs, monkeypatch, relocated,
):
    root, indices, manifests = geometry_inputs
    archived = indices.parent / "archived"
    if relocated:
        monkeypatch.setenv("DUAL2POSE_INPUT_PATH_MAP", json.dumps({str(archived): str(indices)}))
    before = {}
    for number, (path, manifest, key, filename) in enumerate(manifests):
        if number in relocated:
            manifest[key] = str(archived / filename)
        path.write_text(json.dumps(manifest))
        before[path] = path.read_bytes()

    result = _load_verified_split(root, "unity", "test")

    np.testing.assert_array_equal(result.target, np.ones((2, 3, 15, 3)))
    np.testing.assert_array_equal(result.dlt, np.ones((2, 3, 15, 3)))
    assert result.cache_manifest == manifests[0][1]
    assert result.geometry_manifest == manifests[1][1]
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("corrupted, message", [(0, "cache index differs"), (1, "geometry index differs")])
def test_geometry_loader_rejects_tampered_relocated_index(geometry_inputs, monkeypatch, corrupted, message):
    root, indices, manifests = geometry_inputs
    archived = indices.parent / "archived"
    monkeypatch.setenv("DUAL2POSE_INPUT_PATH_MAP", json.dumps({str(archived): str(indices)}))
    for path, manifest, key, filename in manifests:
        manifest[key] = str(archived / filename)
        path.write_text(json.dumps(manifest))
    (indices / manifests[corrupted][3]).write_text("changed after export")

    with pytest.raises(RuntimeError, match=message):
        _load_verified_split(root, "unity", "test")


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
