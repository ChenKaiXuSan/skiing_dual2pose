"""Run in the isolated TF environment with METAPOSE_REPO supplied."""
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from dual2pose.experiments.run_metapose import OfficialMetaPose
from dual2pose.eval.metapose_inputs import normalize_observations, point_mixtures


@unittest.skipUnless(os.environ.get('METAPOSE_REPO'), 'isolated official MetaPose environment required')
class OfficialMetaPoseTest(unittest.TestCase):
    def test_identity_stage_preserves_solution_then_learns(self):
        model = OfficialMetaPose(Path(os.environ['METAPOSE_REPO']), seed=42,
                                 learning_rate=1e-5, gradient_clip_norm=1., identity_stages=True)
        p = np.random.default_rng(8).normal(0,.2,(4,13,3)).astype(np.float32)
        xyz = np.stack([p,p+[.1,.05,0]],axis=1)
        norm,uv,*_ = normalize_observations(xyz,xyz[...,:2]*150+[256,128])
        x=model.initialize(norm)
        task=point_mixtures(uv,dict(weights=np.full((13,4),.25),variances=np.full((13,4),.02))).reshape(4,-1)
        model.add_stage()
        np.testing.assert_array_equal(model.predict(x,task),x)
        before=model.eval_loss(x,task,uv+.02)
        for _ in range(10):model.train_batch(x,task,uv+.02)
        self.assertLess(model.eval_loss(x,task,uv+.02),before)
        pred=model.predict(x,task)
        model.add_stage()
        np.testing.assert_array_equal(model.predict(x,task),pred)

    def test_configured_gradient_clip_and_frozen_previous_stage(self):
        model = OfficialMetaPose(Path(os.environ['METAPOSE_REPO']), seed=42,
                                 learning_rate=1e-5, gradient_clip_norm=0.001)
        p = np.random.default_rng(5).normal(0, .2, (4, 13, 3)).astype(np.float32)
        xyz = np.stack([p, p+[.1,.05,0]], axis=1)
        norm, uv, *_ = normalize_observations(xyz, xyz[..., :2]*150+[256,128])
        x = model.initialize(norm)
        task = point_mixtures(uv, dict(weights=np.full((13,4),.25),
                                      variances=np.full((13,4),.02))).reshape(4,-1)
        model.add_stage()
        model.train_batch(x,task,uv)
        self.assertGreater(model.last_gradient_norm, .001)
        self.assertLessEqual(model.last_clipped_gradient_norm, .001000001)
        frozen = [v.numpy().copy() for v in model.stages[0].weights]
        model.add_stage(); model.train_batch(x,task,uv)
        for value, original in zip(model.stages[0].weights, frozen):
            np.testing.assert_array_equal(value.numpy(), original)

    def test_initialize_train_freeze_and_reload(self):
        model = OfficialMetaPose(Path(os.environ['METAPOSE_REPO']), seed=42)
        p = np.random.default_rng(42).normal(0, .2, (8, 13, 3)).astype(np.float32)
        xyz = np.stack([p, 1.2*p+[.1,.05,0]], axis=1)
        norm, uv, *_ = normalize_observations(xyz, xyz[..., :2]*150 + [256,128])
        x = model.initialize(norm)
        self.assertEqual(x.shape, (8, 59))
        for i in range(8):
            pose, (r,s,t) = model.mp.inf_opt.initial_epi_estimate(model.tf.constant(norm[i],model.tf.float32))
            rr,ss = model.mp.inf_opt.reparam(r,s)
            reference = model.mp.pack_tensors_unbatched([pose,rr,ss,t])[0].numpy()
            np.testing.assert_allclose(x[i],reference,atol=2e-5)
        fwd = model.forward(x).reshape(8, 2, 13, 2)
        np.testing.assert_allclose(fwd, uv, atol=2e-5)
        stats = dict(weights=np.full((13,4), .25), variances=np.full((13,4), .02))
        task = point_mixtures(uv, stats).reshape(8,-1)
        model.add_stage()
        before = model.eval_loss(x, task, uv)
        for _ in range(25):
            model.train_batch(x, task, uv)
        after = model.eval_loss(x, task, uv)
        self.assertLess(after, before)
        prediction = model.predict(x, task)
        self.assertTrue(np.isfinite(prediction).all())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'stage.weights.h5'
            model.save_stage(path)
            reloaded = OfficialMetaPose(Path(os.environ['METAPOSE_REPO']), seed=0)
            reloaded.add_stage(); reloaded.load_stage(path)
            np.testing.assert_array_equal(prediction, reloaded.predict(x, task))
        frozen = [v.numpy().copy() for v in model.stages[0].weights]
        model.add_stage(); model.train_batch(x, task, uv)
        for value, original in zip(model.stages[0].weights, frozen):
            np.testing.assert_array_equal(value.numpy(), original)


if __name__ == '__main__':
    unittest.main()
