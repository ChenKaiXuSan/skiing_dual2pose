"""Tests for immutable, target-free native prediction manifests."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dual2pose.eval.native_output_manifest import (
    PredictionManifest,
    freeze_prediction_manifest,
    open_scoring_manifest,
    verify_prediction_manifest,
)


class NativeOutputManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prediction = self.root / "predictions.npy"
        np.save(self.prediction, np.zeros((2, 30, 13, 3), dtype=np.float32))
        self.sample_index = self.root / "samples.jsonl"
        self.sample_index.write_text(
            "".join(json.dumps({"index": index}) + "\n" for index in range(2)),
            encoding="utf-8",
        )
        self.manifest = PredictionManifest(
            method="sam3d_left_native",
            dataset="unity",
            coordinate_frame="left_camera_m",
            prediction_path=str(self.prediction),
            sample_count=2,
            frame_count=30,
            joint_count=13,
            unit="m",
            source_checkpoint_sha256="1" * 64,
            config_sha256="2" * 64,
            code_sha256="3" * 64,
            sample_index_path=str(self.sample_index),
            sample_index_sha256=None,
        )
        self.output = self.root / "manifest.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_freeze_rejects_target_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "target"):
            freeze_prediction_manifest(
                replace(self.manifest, target_path=str(self.root / "target.npy")),
                self.output,
            )

    def test_freeze_hashes_prediction_and_sample_index(self) -> None:
        payload = freeze_prediction_manifest(self.manifest, self.output)
        self.assertTrue(payload["frozen"])
        self.assertEqual(len(payload["prediction_sha256"]), 64)
        self.assertEqual(len(payload["sample_index_sha256"]), 64)
        self.assertEqual(verify_prediction_manifest(self.output), payload)

    def test_existing_frozen_manifest_is_not_overwritten(self) -> None:
        freeze_prediction_manifest(self.manifest, self.output)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            freeze_prediction_manifest(self.manifest, self.output)
        self.assertEqual(self.output.read_bytes(), before)

    def test_scoring_requires_frozen_prediction_checksum_before_target_open(self) -> None:
        unfrozen = self.root / "unfrozen.json"
        unfrozen.write_text(json.dumps({"frozen": False}), encoding="utf-8")
        target = self.root / "target.npy"
        with self.assertRaisesRegex(ValueError, "not frozen"):
            open_scoring_manifest(unfrozen, target)
        self.assertFalse(target.exists())

    def test_scoring_rejects_prediction_changed_after_freeze(self) -> None:
        freeze_prediction_manifest(self.manifest, self.output)
        np.save(self.prediction, np.ones((2, 30, 13, 3), dtype=np.float32))
        target = self.root / "target.npy"
        np.save(target, np.zeros((2, 30, 13, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "prediction checksum"):
            open_scoring_manifest(self.output, target)


if __name__ == "__main__":
    unittest.main()
