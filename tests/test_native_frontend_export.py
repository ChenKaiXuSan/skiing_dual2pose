"""Target-free export tests for SAM3D and pre-Canon controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dual2pose.experiments.export_native_frontend_predictions import (
    export_cached_sam3d_and_controls,
    export_unity_lifter_windows,
    validate_export_request,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeFrontendExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        rng = np.random.default_rng(88)
        common = rng.normal(size=(4, 30, 13, 3)).astype(np.float32)
        eyes = rng.normal(size=(4, 30, 2, 3)).astype(np.float32)
        self.left = np.concatenate([eyes, common], axis=2)
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.right = 1.4 * (self.left @ rotation.T) + np.array(
            [2.0, -1.0, 4.0], dtype=np.float32
        )
        np.save(self.cache / "left.npy", self.left)
        np.save(self.cache / "right.npy", self.right)
        np.save(self.cache / "frames.npy", np.tile(np.arange(30), (4, 1)))
        # Deliberately not a NumPy file: target-free export must never open it.
        (self.cache / "target.npy").write_bytes(b"forbidden target")
        samples = self.cache / "samples.jsonl"
        samples.write_text(
            "".join(
                json.dumps(
                    {
                        "index": index,
                        "person_id": "female",
                        "action_id": f"action_{index}",
                        "cam1_id": "capture_L0_A000",
                        "cam2_id": "capture_L0_A010",
                    }
                )
                + "\n"
                for index in range(4)
            ),
            encoding="utf-8",
        )
        manifest = {
            "dataset": "unity",
            "split": "val",
            "sample_count": 4,
            "time_window": 30,
            "num_joints": 15,
            "representation": "raw_pre_batch_canonicalization",
            "config_sha256": "2" * 64,
            "files": {
                name: _sha256(self.cache / name)
                for name in ("left.npy", "right.npy", "target.npy", "frames.npy", "samples.jsonl")
            },
        }
        (self.cache / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        self.output = self.root / "output"

    def make_lifter_archive(self, frontend: str = "motionbert") -> Path:
        archive = self.root / frontend
        entries = []
        for sample_index in range(2):
            for camera_index, camera_id in enumerate(("capture_L0_A000", "capture_L0_A010")):
                pose = np.full((30, 15, 3), 10 * sample_index + camera_index, dtype=np.float32)
                relative = Path("poses") / f"sample_{sample_index}_{camera_index}.npz"
                path = archive / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, pose=pose, frame_indices=np.arange(30))
                entries.append({"person_id": "female", "action_id": f"action_{sample_index}",
                    "camera_id": camera_id, "pose_path": str(relative), "split": "val"})
        manifest = {"frontend_name": frontend, "metadata": {
            "input_2d_source": "unity_gt_h36m17",
            "output_joint_convention": "canonfuse15",
            "estimator_checkpoint_sha256": "7" * 64}, "entries": entries}
        path = archive / f"{frontend}_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_export_reads_no_target_and_selects_unity_common13(self) -> None:
        report = export_cached_sam3d_and_controls(self.cache, self.output, limit=2)
        self.assertFalse(report["target_loaded"])
        left = np.load(self.output / "left_native" / "predictions.npy")
        np.testing.assert_array_equal(left, self.left[:2, :, 2:15])
        self.assertEqual(left.shape, (2, 30, 13, 3))

    def test_all_precanon_controls_are_exported_or_explicitly_unavailable(self) -> None:
        report = export_cached_sam3d_and_controls(self.cache, self.output, limit=2)
        expected_measured = {
            "left_native",
            "right_native",
            "unaligned_native_average",
            "native_sequence_aligned_average",
            "native_aligned_quality_weighted",
        }
        self.assertEqual(
            {key for key, value in report["methods"].items() if value["status"] == "frozen"},
            expected_measured,
        )
        smoothnet = report["methods"]["native_aligned_average_smoothnet"]
        self.assertEqual(smoothnet["status"], "unavailable")
        self.assertIn("canonical", smoothnet["reason"].lower())

    def test_exactly_related_views_align_to_left_without_target(self) -> None:
        export_cached_sam3d_and_controls(self.cache, self.output, limit=2)
        left = np.load(self.output / "left_native" / "predictions.npy")
        aligned = np.load(
            self.output / "native_sequence_aligned_average" / "predictions.npy"
        )
        np.testing.assert_allclose(aligned, left, atol=1e-5, rtol=1e-5)

    def test_every_prediction_manifest_is_frozen_and_target_free(self) -> None:
        report = export_cached_sam3d_and_controls(self.cache, self.output, limit=2)
        for method, record in report["methods"].items():
            if record["status"] != "frozen":
                continue
            payload = json.loads(Path(record["manifest_path"]).read_text(encoding="utf-8"))
            self.assertTrue(payload["frozen"], method)
            self.assertFalse(any(key.startswith("target") for key in payload), method)
            self.assertEqual(payload["sample_count"], 2)

    def test_existing_output_root_is_preserved(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            export_cached_sam3d_and_controls(self.cache, self.output, limit=2)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_partial_test_export_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden for test"):
            validate_export_request(split="test", limit=2)


    def test_unity_lifter_archive_is_assembled_per_physical_view(self) -> None:
        manifest = self.make_lifter_archive()
        output = self.root / "motionbert_output"
        report = export_unity_lifter_windows(self.cache, manifest, output, limit=2)
        left = np.load(output / "motionbert_native_left" / "predictions.npy")
        right = np.load(output / "motionbert_native_right" / "predictions.npy")
        self.assertEqual(left.shape, (2, 30, 13, 3))
        np.testing.assert_array_equal(left[0], 0.0)
        np.testing.assert_array_equal(right[0], 1.0)
        np.testing.assert_array_equal(left[1], 10.0)
        self.assertTrue(report["ground_truth_2d_used"])
        self.assertFalse(report["target_loaded"])
        self.assertEqual(report["view_groups"]["motionbert_native_view_pool"],
                         ["motionbert_native_left", "motionbert_native_right"])

    def test_combined_lifter_manifest_uses_matching_split_metadata(self) -> None:
        manifest = self.make_lifter_archive("motionbert")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["metadata"] = {"source_metadata": {"val": payload["metadata"]}}
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        report = export_unity_lifter_windows(self.cache, manifest, self.root / "combined", limit=2)
        self.assertTrue(report["ground_truth_2d_used"])
        self.assertEqual(report["sample_count"], 2)

    def test_lifter_archive_frame_mismatch_is_rejected(self) -> None:
        manifest = self.make_lifter_archive("poseformer")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        first = manifest.parent / payload["entries"][0]["pose_path"]
        np.savez_compressed(first, pose=np.zeros((30, 15, 3), dtype=np.float32),
                            frame_indices=np.arange(1, 31))
        with self.assertRaisesRegex(ValueError, "frame"):
            export_unity_lifter_windows(self.cache, manifest, self.root / "bad_lifter", limit=2)


if __name__ == "__main__":
    unittest.main()
