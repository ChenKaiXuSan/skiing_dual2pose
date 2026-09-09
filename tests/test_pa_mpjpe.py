import importlib
import importlib.util
import unittest
import numpy as np
import torch


class PAMetricTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('dual2pose.eval.pa_mpjpe'),
                             'PA-MPJPE metric module must be implemented')
        return importlib.import_module('dual2pose.eval.pa_mpjpe')

    def test_independent_frame_similarity_is_removed_without_mutation(self):
        api = self.module()
        rng = np.random.default_rng(8)
        target = rng.normal(size=(2, 5, 13, 3))
        pred = np.empty_like(target)
        for b in range(2):
            for t in range(5):
                angle = .37 * (t + 1) + b
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                pred[b, t] = (1.2 + t) * target[b, t] @ rotation + [b, t, 3]
        saved = pred.copy()
        distances, degenerate = api.pa_joint_errors(pred, target)
        np.testing.assert_allclose(distances, 0, atol=1e-12)
        np.testing.assert_array_equal(pred, saved)
        self.assertEqual(degenerate.sum(), 0)

    def test_reflection_is_not_removed(self):
        api = self.module()
        target = np.random.default_rng(9).normal(size=(2, 4, 15, 3))
        pred = target * np.array([-1, 1, 1])
        distances, _ = api.pa_joint_errors(pred, target)
        self.assertGreater(distances.mean(), .1)

    def test_collapsed_predictions_are_retained_at_target_centroid(self):
        api = self.module()
        target = np.random.default_rng(1).normal(size=(2, 4, 13, 3))
        pred = np.full_like(target, 17.)
        distances, degenerate = api.pa_joint_errors(pred, target)
        expected = np.linalg.norm(target - target.mean(axis=-2, keepdims=True), axis=-1)
        np.testing.assert_allclose(distances, expected, atol=1e-12)
        self.assertEqual(degenerate.sum(), 8)
        flat, _ = api.pa_joint_errors(target, pred)
        np.testing.assert_allclose(flat, 0, atol=1e-12)

    def test_partial_batches_preserve_every_pa_point(self):
        api = self.module()
        target = torch.from_numpy(np.random.default_rng(7).normal(size=(5, 4, 13, 3))).float()
        pred = target.clone(); pred[-1, :, 0, 0] += 5
        total = api.PAMetricAccumulator(); total.update(pred, target)
        split = api.PAMetricAccumulator()
        split.update(pred[:4], target[:4]); split.update(pred[4:], target[4:])
        result = split.result()
        self.assertAlmostEqual(result['pa_mpjpe'], total.result()['pa_mpjpe'], places=14)
        self.assertEqual(result['pa_point_count'], 5 * 4 * 13)
        self.assertEqual(result['pa_frame_count'], 20)
        self.assertEqual(result['sample_count'], 5)
        self.assertEqual(result['point_count'], result['pa_point_count'])

    def test_nonfinite_input_fails_without_skipping_frames(self):
        api = self.module()
        target = np.zeros((1, 4, 13, 3)); pred = target.copy()
        pred[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError): api.pa_joint_errors(pred, target)


if __name__ == '__main__': unittest.main()
