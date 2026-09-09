import unittest

import torch

from dual2pose.eval.main_baseline_data import batch_slices, canonical_batch
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


class MainBaselineDataTests(unittest.TestCase):
    def test_final_partial_batch_is_retained(self):
        self.assertEqual(list(batch_slices(30, 4)), [(0, 4), (4, 8), (8, 12),
                         (12, 16), (16, 20), (20, 24), (24, 28), (28, 30)])
        self.assertEqual(list(batch_slices(64440, 256))[-1], (64256, 64440))

    def test_training_drop_last_is_explicit(self):
        self.assertEqual(list(batch_slices(9, 4, drop_last=True)), [(0, 4), (4, 8)])

    def test_each_stream_preserves_archived_batch_anchor(self):
        gen = torch.Generator().manual_seed(19)
        inputs = [torch.randn(3, 30, 15, 3, generator=gen) for _ in range(3)]
        actual = canonical_batch(*inputs, dataset="unity")
        for source, output in zip(inputs, actual):
            expected = canonicalize_pose_torch(source)[0]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        independently = torch.cat([canonicalize_pose_torch(inputs[0][i:i+1])[0]
                                   for i in range(3)])
        self.assertGreater((actual[0] - independently).abs().max().item(), .1)

    def test_ski_uses_common13_hip_and_neck_indices(self):
        inputs = [torch.randn(2, 30, 13, 3) for _ in range(3)]
        actual = canonical_batch(*inputs, dataset="ski")
        for source, output in zip(inputs, actual):
            expected = canonicalize_pose_torch(source, left_hip=4, right_hip=5, neck=12)[0]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
