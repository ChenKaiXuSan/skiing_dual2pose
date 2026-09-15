"""Controller tests for the independent native-output comparison."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dual2pose.eval.native_output_manifest import (
    PredictionManifest,
    freeze_prediction_manifest,
    verify_prediction_manifest,
)
from dual2pose.eval.evaluate_native_output_comparison import (
    METHOD_IDS,
    _descriptor_loader,
    build_empty_report,
    score_frozen_prediction,
    unavailable_result,
    validate_coverage,
    write_result_exclusive,
)


class EvaluateNativeOutputComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.samples = self.root / "samples.jsonl"
        self.samples.write_text(json.dumps({"index": 0}) + "\n", encoding="utf-8")
        self.prediction = self.root / "prediction.npy"
        prediction = np.zeros((1, 30, 13, 3), dtype=np.float32)
        prediction[:, :, 0, 0] = 1.0
        np.save(self.prediction, prediction)
        self.manifest = self.root / "manifest.json"
        freeze_prediction_manifest(
            PredictionManifest(
                method="left_native",
                dataset="unity",
                coordinate_frame="left_camera_m",
                prediction_path=str(self.prediction),
                sample_count=1,
                frame_count=30,
                joint_count=13,
                unit="m",
                source_checkpoint_sha256="1" * 64,
                config_sha256="2" * 64,
                code_sha256="3" * 64,
                sample_index_path=str(self.samples),
            ),
            self.manifest,
        )
        self.output = self.root / "result.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_target_is_opened_only_after_prediction_manifest_verifies(self) -> None:
        np.save(self.prediction, np.zeros((1, 30, 13, 3), dtype=np.float32))
        target_loader = mock.Mock(side_effect=AssertionError("target opened early"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            score_frozen_prediction(
                self.manifest, target_loader=target_loader, require_full_coverage=False
            )
        target_loader.assert_not_called()

    def test_nonzero_fixture_is_scored_after_prediction_freezes(self) -> None:
        target = np.zeros((1, 30, 13, 3), dtype=np.float32)
        target_loader = mock.Mock(return_value=target)
        row = score_frozen_prediction(
            self.manifest, target_loader=target_loader, require_full_coverage=False
        )
        self.assertEqual(row["status"], "measured")
        self.assertGreater(row["mpjpe"], 0.0)
        target_loader.assert_called_once()

    def test_target_descriptor_must_match_prediction_sample_order(self) -> None:
        target = self.root / "target.npy"
        np.save(target, np.zeros((1, 30, 13, 3), dtype=np.float32))
        descriptor = self.root / "target_descriptor.json"
        descriptor.write_text(
            json.dumps(
                {
                    "dataset": "unity",
                    "sample_index_sha256": "0" * 64,
                    "targets": {
                        "left_camera_m": {"path": str(target), "sha256": "0" * 64}
                    },
                }
            ),
            encoding="utf-8",
        )
        payload = verify_prediction_manifest(self.manifest)
        loader = _descriptor_loader(descriptor, "unity")
        with self.assertRaisesRegex(ValueError, "sample index"):
            loader(payload)

    def test_unity_requires_64440_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "64440"):
            validate_coverage(dataset="unity", sample_count=64439, frame_count=30, joint_count=13)

    def test_ski_requires_30_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "30"):
            validate_coverage(dataset="ski", sample_count=29, frame_count=30, joint_count=13)

    def test_unavailable_row_has_reason_and_no_numeric_metric(self) -> None:
        row = unavailable_result("no supplied frame mapping")
        self.assertEqual(row["status"], "unavailable")
        self.assertTrue(row["reason"])
        self.assertIsNone(row["mpjpe"])
        self.assertIsNone(row["pa_mpjpe"])
        self.assertIsNone(row["acceleration_error"])

    def test_existing_result_is_never_overwritten(self) -> None:
        report = build_empty_report()
        write_result_exclusive(self.output, report)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            write_result_exclusive(self.output, report)
        self.assertEqual(self.output.read_bytes(), before)

    def test_all_seventeen_method_ids_are_emitted(self) -> None:
        self.assertEqual(tuple(build_empty_report()["methods"]), METHOD_IDS)


if __name__ == "__main__":
    unittest.main()
