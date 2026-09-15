"""Scientific-contract tests for the native-input STRIDE adaptation."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from dual2pose.eval.stride_external import (
    assemble_h36m17, canonical_average17, IndependentStride,
)


class StrideMappingTests(unittest.TestCase):
    def test_mapping_keeps_all_scored_points_and_uses_real_extra_bones(self):
        common = np.arange(39, dtype=np.float32).reshape(13, 3)
        bones = np.arange(381, dtype=np.float32).reshape(127, 3) + 1000
        result = assemble_h36m17(common, bones)
        # Literal externally defined H36M -> common-13 order, not adapter helper.
        np.testing.assert_array_equal(result[[11,14,12,15,4,1,5,2,6,3,16,13,9]], common)
        np.testing.assert_array_equal(result[[0,7,8,10]],
                                      [[1003,1004,1005],[1105,1106,1107],
                                       [1111,1112,1113],[1378,1379,1380]])

    def test_missing_or_nonfinite_bones_are_not_zero_filled(self):
        common = np.zeros((13,3), np.float32)
        for bones in (np.zeros((126,3)), np.full((127,3), np.nan)):
            with self.assertRaises(ValueError):
                assemble_h36m17(common, bones)

    def test_extra_points_use_original_batch_anchor_not_their_own_pelvis(self):
        # Native unity-15: hips +-x, neck +y, eyes towards +z => identity R.
        native = np.zeros((2,30,15,3), np.float32)
        native[:,:,6] = [-1,0,0]; native[:,:,7] = [1,0,0]
        native[:,:,14] = [0,2,0]; native[:,:,0:2] = [0,2,1]
        native[1] += [10,20,30]
        bones = np.zeros((2,30,127,3), np.float32)
        bones[:,:,[1,35,37,126]] = [3,4,5]
        bones[1] += [10,20,30]
        result = canonical_average17(native, native, bones, bones, 'unity')
        np.testing.assert_allclose(result[1,0,0], [13,24,35], atol=1e-6)
        np.testing.assert_allclose(result[1,0,4], [9,20,30], atol=1e-6)


class StrideResetTests(unittest.TestCase):
    def make_adapter(self):
        model = torch.nn.Linear(3, 3)
        with torch.no_grad():
            model.weight.copy_(torch.eye(3)*.4);model.bias.fill_(.1)
        residual = lambda a,b: ((a-b)**2).mean()
        losses = SimpleNamespace(loss_mpjpe=residual, n_mpjpe=residual,
                                 loss_velocity=residual,
                                 loss_limb_var=lambda x: (x[:,1:]-x[:,:-1]).square().mean())
        config = SimpleNamespace(epochs=3, learning_rate=.01, weight_decay=.01,
                                 lr_decay=.99, lambda_3d_pos=1.,lambda_scale=.5,
                                 lambda_3d_velocity=20.,lambda_lv=200., flip=False)
        return IndependentStride(model, losses, lambda x:x.clone(), config, seed=42)

    def test_unrelated_window_cannot_change_repeated_window_output(self):
        adapter = self.make_adapter()
        data = torch.randn(1,30,17,3, generator=torch.Generator().manual_seed(7))
        original = data.clone()
        first, report1 = adapter.refine(data)
        adapter.refine(data*2+1)
        again, report2 = adapter.refine(data)
        torch.testing.assert_close(first, again, rtol=0, atol=0)
        torch.testing.assert_close(data, original, rtol=0, atol=0)
        self.assertEqual(report1['steps'], 3)
        self.assertEqual(report1['losses'], report2['losses'])

    def test_batch_of_independent_windows_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_adapter().refine(torch.ones(2,30,17,3))

    def test_nonfinite_prior_or_input_is_rejected(self):
        adapter = self.make_adapter()
        with self.assertRaises(ValueError):
            adapter.refine(torch.full((1,30,17,3),float('nan')))
        with torch.no_grad():
            adapter.model.weight.fill_(float('nan'))
        with self.assertRaises(ValueError):
            IndependentStride(adapter.model,adapter.loss,adapter.flip_fn,adapter.config,42)


class StridePreparationTests(unittest.TestCase):
    def test_preparation_needs_no_target_and_retains_sample_and_frame_order(self):
        from tests.test_external_baseline_inputs import ExternalBaselineInputTests
        from dual2pose.experiments.run_stride_external import prepare_inputs
        fixture = ExternalBaselineInputTests()
        fixture.setUp()
        try:
            fixture.make_cache('unity','val')
            mapping = [1,2,5,6,7,8,9,10,11,12,13,14,41,62,69]
            for i in range(4):
                for view, poses in [('left',fixture.left),('right',fixture.right)]:
                    directory = fixture.root/'sam3d_body_results/inference/female'/f'action{i}'/'frames'/view
                    directory.mkdir(parents=True)
                    for t,f in enumerate(fixture.frames[i]):
                        points=np.zeros((70,3),np.float32);points[mapping]=poses[i,t]
                        np.savez(directory/f'{f:06d}_sam3d_body.npz',output={
                            'pred_keypoints_3d':points,'pred_joint_coords':np.ones((127,3),np.float32)})
            average, provenance = prepare_inputs(fixture.cache,fixture.root,limit=2,workers=1)
            self.assertFalse((fixture.cache/'target.npy').exists())
            self.assertEqual(average.shape,(2,30,17,3))
            self.assertEqual(provenance['sample_count'],2)
            from dual2pose.eval.external_baseline_inputs import load_pose_probe
            expected=load_pose_probe(fixture.cache,2).average
            np.testing.assert_allclose(average[:,:, [11,14,12,15,4,1,5,2,6,3,16,13,9]],expected,atol=1e-6)
            first = fixture.root/'sam3d_body_results/inference/female/action0/frames/left/000010_sam3d_body.npz'
            with np.load(first,allow_pickle=True) as a:
                output=a['output'].item()
            output['pred_keypoints_3d'][5,0] += .1
            np.savez(first,output=output)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                prepare_inputs(fixture.cache,fixture.root,limit=1,workers=1)
        finally:
            fixture.tearDown()

    def test_partial_test_run_is_not_allowed(self):
        from tests.test_external_baseline_inputs import ExternalBaselineInputTests
        from dual2pose.experiments.run_stride_external import prepare_inputs
        fixture=ExternalBaselineInputTests();fixture.setUp()
        try:
            fixture.make_cache('unity','test')
            with self.assertRaisesRegex(ValueError,'test'):
                prepare_inputs(fixture.cache,fixture.root,limit=1,workers=1)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
