import inspect
import unittest
import numpy as np
from dual2pose.eval.metapose_inputs import normalize_observations, restore_left_gauge, fit_point_uncertainty, point_mixtures, NativeInputs
from dual2pose.eval.metapose_supervision import require_supervised_split, Supervision2D


class MetaPoseInputsTest(unittest.TestCase):
    def setUp(self):
        self.xyz = np.random.default_rng(4).normal(size=(5, 2, 13, 3))
        self.uv = 150 * self.xyz[..., :2] + [256, 128]

    def test_roundtrip_and_projection(self):
        x, u, q, t, origin, side = normalize_observations(self.xyz, self.uv)
        np.testing.assert_allclose(x[..., :2], u, atol=1e-6)
        np.testing.assert_allclose((x - t[..., None, :]) / q[..., None, None], self.xyz, atol=1e-6)
        np.testing.assert_allclose((self.uv-origin[..., None, :])/side[..., None, None], u, atol=1e-6)

    def test_left_root_is_input_derived(self):
        x, _, q, t, _, _ = normalize_observations(self.xyz, self.uv)
        actual = restore_left_gauge(x[:, 0] + [0, 0, 18], q[:, 0], t[:, 0], self.xyz[:, 0])
        np.testing.assert_allclose(actual, self.xyz[:, 0], atol=1e-5)
        self.assertNotIn('target', inspect.signature(restore_left_gauge).parameters)

    def test_invalid_input_rejected(self):
        for uv in (np.zeros_like(self.uv), -self.uv, self.uv*np.nan):
            with self.assertRaises(ValueError):
                normalize_observations(self.xyz, uv)

    def test_uncertainty_centers_never_use_labels(self):
        fitted = fit_point_uncertainty(np.random.default_rng(42).normal(0, .03, (100, 2, 13, 2)))
        _, uv, *_ = normalize_observations(self.xyz, self.uv)
        mixtures = point_mixtures(uv, fitted)
        self.assertEqual(mixtures.shape, (5, 2, 13, 4, 4))
        np.testing.assert_allclose(mixtures[..., 0].sum(-1), 1)
        self.assertTrue((mixtures[..., 3] >= 1e-6).all())
        for k in range(4):
            np.testing.assert_allclose(mixtures[..., k, 1:3], uv)
        self.assertEqual(set(inspect.signature(point_mixtures).parameters), {'uv', 'uncertainty'})

    def test_no_test_supervision(self):
        for split in ('test', 'testing', ''):
            with self.assertRaises(ValueError):
                require_supervised_split(split)
        require_supervised_split('train')
        require_supervised_split('val')

    def test_full_reader_pairs_test_inputs_without_any_gt_files(self):
        from tests.test_external_baseline_inputs import ExternalBaselineInputTests
        fixture = ExternalBaselineInputTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.make_cache('unity','test')
        fixture.write_2d('unity')
        inp = NativeInputs(fixture.cache,fixture.root)
        raw,uv = inp.batch(2,4)
        self.assertFalse((fixture.cache/'target.npy').exists())
        np.testing.assert_array_equal(raw[:,:,0],fixture.left[2:,:,2:])
        np.testing.assert_array_equal(uv[0,0,0,[0,1,12]],[[15,5],[16,6],[79,69]])
        with self.assertRaisesRegex(ValueError,'never test'):
            Supervision2D(inp)

    def test_full_reader_rejects_corrupt_cache(self):
        from tests.test_external_baseline_inputs import ExternalBaselineInputTests
        fixture = ExternalBaselineInputTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.make_cache('ski','test')
        np.save(fixture.cache/'left.npy',fixture.left+1)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            NativeInputs(fixture.cache,fixture.root)

    def test_ski_supervision_reads_only_2d_and_identity_fields(self):
        import h5py
        from types import SimpleNamespace
        from pathlib import Path
        import tempfile
        from dual2pose.map_config import H36M_17_TO_COMMON_13_INDICES
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels.h5'
            with h5py.File(path,'w') as f:
                for k,v in [('subj',0),('seq',103),('cam',1),('frame',95)]:
                    f.create_dataset(k,data=[v])
                f.create_dataset('2D',data=np.arange(34).reshape(1,34)/256)
                # Deliberately no 3D or camera fields in this fixture.
            reader=Supervision2D(SimpleNamespace(split='train'))
            labels=reader.ski_labels(path)
            expected=np.arange(34).reshape(17,2)[H36M_17_TO_COMMON_13_INDICES]
            np.testing.assert_array_equal(labels[(0,103,1,95)],expected)


if __name__ == '__main__':
    unittest.main()
