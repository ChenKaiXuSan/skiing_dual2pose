"""Static and runtime guards against accidental competitor canonicalization."""
from __future__ import annotations

import importlib
import inspect
import unittest
from unittest import mock

import numpy as np


COMPETITOR_MODULES = (
    "dual2pose.eval.native_output_methods",
    "dual2pose.experiments.export_native_frontend_predictions",
    "dual2pose.experiments.run_native_external_predictions",
    "dual2pose.experiments.export_metapose_native_split",
)


class NativeOutputNoCanonGuardTest(unittest.TestCase):
    def test_competitor_modules_do_not_reference_canonfuse_canonicalizer(self) -> None:
        for module_name in COMPETITOR_MODULES:
            with self.subTest(module=module_name):
                source = inspect.getsource(importlib.import_module(module_name))
                self.assertNotIn("canonicalize_pose_torch", source)
                self.assertNotIn("canonical_batch", source)

    def test_precanon_controls_run_when_canonicalizer_is_disabled(self) -> None:
        methods = importlib.import_module("dual2pose.eval.native_output_methods")
        rng = np.random.default_rng(9)
        left = rng.normal(size=(1, 30, 13, 3))
        right = rng.normal(size=(1, 30, 13, 3))
        with mock.patch(
            "dual2pose.trainer.canonicalize.canonicalize_pose_torch",
            side_effect=AssertionError("competitor called Canon"),
        ):
            methods.unaligned_average(left, right)
            methods.aligned_average(left, right)
            methods.aligned_quality_weighted(left, right)


if __name__ == "__main__":
    unittest.main()
