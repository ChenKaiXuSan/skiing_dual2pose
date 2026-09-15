"""Contracts for target-free external native-output exports."""
from __future__ import annotations

import json
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import numpy as np

from dual2pose.experiments.run_native_external_predictions import (
    _run_stride,
    export_external,
    export_two_view_refiner,
    collect_unique_frame_jobs,
    select_stride_views,
    validate_export_request,
    write_run_status,
)


class NativeExternalPredictionsTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(61)
        self.left = rng.normal(size=(2, 30, 13, 3)).astype(np.float32)
        self.right = rng.normal(size=(2, 30, 13, 3)).astype(np.float32)
        self.left17 = np.pad(self.left, ((0, 0), (0, 0), (0, 4), (0, 0)))
        self.right17 = np.pad(self.right, ((0, 0), (0, 0), (0, 4), (0, 0)))

    def test_stride_receives_one_native_view_at_a_time(self) -> None:
        runner = mock.Mock(side_effect=lambda pose: pose)
        prediction = export_two_view_refiner("stride", self.left, self.right, runner)
        self.assertEqual(runner.call_count, 2)
        np.testing.assert_array_equal(runner.call_args_list[0].args[0], self.left)
        np.testing.assert_array_equal(runner.call_args_list[1].args[0], self.right)
        np.testing.assert_array_equal(prediction[0], self.left)
        np.testing.assert_array_equal(prediction[1], self.right)

    def test_stride_frame_jobs_are_deduplicated_before_disk_reads(self) -> None:
        directories = [["left_a"], ["left_a"]]
        frames = np.array([[1, 2, 2], [2, 3, 3]])
        self.assertEqual(
            collect_unique_frame_jobs(directories, frames),
            [("left_a", 1), ("left_a", 2), ("left_a", 3)],
        )

    def test_stride_view_shards_preserve_the_requested_native_stream(self) -> None:
        left = select_stride_views(self.left17, self.right17, "left")
        right = select_stride_views(self.left17, self.right17, "right")
        both = select_stride_views(self.left17, self.right17, "both")
        self.assertEqual([name for name, _ in left], ["left"])
        self.assertEqual([name for name, _ in right], ["right"])
        self.assertEqual([name for name, _ in both], ["left", "right"])
        np.testing.assert_array_equal(left[0][1], self.left17)
        np.testing.assert_array_equal(right[0][1], self.right17)

    def test_invalid_stride_view_shard_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stride view"):
            select_stride_views(self.left17, self.right17, "canonical")

    def test_deciwatch_receives_independent_native_views(self) -> None:
        runner = mock.Mock(side_effect=lambda pose: pose)
        export_two_view_refiner("deciwatch", self.left, self.right, runner)
        self.assertEqual(runner.call_count, 2)
        self.assertIsNot(runner.call_args_list[0].args[0], runner.call_args_list[1].args[0])

    def test_metapose_export_is_raw_restored_left_camera_output(self) -> None:
        prediction, manifest = export_external(
            "metapose", self.left, {"dataset": "unity", "input_label": "raw_restored_left"}
        )
        np.testing.assert_array_equal(prediction, self.left)
        self.assertEqual(manifest["coordinate_frame"], "left_camera_m")

    def test_dlt_export_preserves_calibrated_world_frame(self) -> None:
        _, manifest = export_external(
            "dlt", self.left, {"dataset": "unity", "input_label": "calibrated_world"}
        )
        self.assertEqual(manifest["coordinate_frame"], "calibrated_world_m")

    def test_canonical_input_label_is_rejected(self) -> None:
        for label in ("canonical_average", "left_canonical", "gt_canonical"):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "canonical"):
                export_external("dlt", self.left, {"dataset": "unity", "input_label": label})

    def test_export_stage_has_no_target_parameter(self) -> None:
        self.assertNotIn("target", inspect.signature(export_external).parameters)
        self.assertNotIn("target", inspect.signature(export_two_view_refiner).parameters)

    def test_test_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit is forbidden for test"):
            validate_export_request(split="test", limit=1)

    def test_invalid_shape_is_rejected_before_runner(self) -> None:
        runner = mock.Mock()
        with self.assertRaisesRegex(ValueError, "shape"):
            export_two_view_refiner("stride", self.left[:, :-1], self.right, runner)
        runner.assert_not_called()

    def test_stride_emits_periodic_progress_and_a_final_window_count(self) -> None:
        class IdentityAdapter:
            def refine(self, value):
                return torch.from_numpy(value), None

        events = []
        native = self.left17[:1]
        with (
            mock.patch(
                "dual2pose.eval.stride_external.load_official_stride",
                return_value=(IdentityAdapter(), {"checkpoint_sha256": "checkpoint"}),
            ),
            mock.patch(
                "dual2pose.experiments.run_native_external_predictions.subprocess.check_output",
                return_value="commit\n",
            ),
        ):
            _run_stride(
                [("left", native)],
                repo=Path("stride"),
                checkpoint=Path("checkpoint.bin"),
                device="cpu",
                on_progress=events.append,
                progress_every=2,
            )
        self.assertEqual(
            events,
            [
                {
                    "view": "left",
                    "completed_windows": 1,
                    "total_windows": 1,
                }
            ],
        )

    def test_run_status_is_persisted_as_machine_readable_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_run_status(
                Path(directory),
                status="running",
                phase="stride_inference",
                dataset="unity",
                view="right",
                completed_windows=20,
                total_windows=100,
            )
            self.assertEqual(path, Path(directory) / "run_status.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["phase"], "stride_inference")
        self.assertEqual(payload["dataset"], "unity")
        self.assertEqual(payload["view"], "right")
        self.assertEqual(payload["completed_windows"], 20)
        self.assertEqual(payload["total_windows"], 100)
        self.assertIn("updated_at_utc", payload)


if __name__ == "__main__":
    unittest.main()
