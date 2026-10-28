import unittest
import torch
import tempfile
from pathlib import Path
from dual2pose.experiments.run_main_baselines import MetricAccumulator

class MainBaselineMetricTests(unittest.TestCase):
    def test_manual_checkpoint_has_protocol_and_rejects_legacy(self):
        from dual2pose.experiments.run_main_baselines import _save_checkpoint
        from dual2pose.eval.main_baseline_data import require_current_protocol
        from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
        model = torch.nn.Linear(3, 3)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'weights.pt'
            _save_checkpoint(path, model, optimizer, 0, 1., {})
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['canonical_mode'], 'per_frame')
            self.assertEqual(saved['evaluation_protocol'], EVALUATION_PROTOCOL)
            require_current_protocol(saved)
        for saved in ({}, {'canonical_mode':'first_frame'}, {'canonical_mode':'per_frame'}):
            with self.assertRaisesRegex(ValueError, 'protocol'):
                require_current_protocol(saved)

    def test_reference_model_rejects_legacy_before_weight_loading(self):
        from dual2pose.experiments.run_main_baselines import ReferenceModel
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'legacy.pt'
            torch.save({'state_dict': {}}, path)
            with self.assertRaisesRegex(ValueError, 'protocol|canonical_mode'):
                ReferenceModel(path)

    def test_reference_model_preserves_and_validates_skeleton_contract(self):
        from dual2pose.experiments.run_main_baselines import ReferenceModel
        from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer
        from omegaconf import OmegaConf
        config = OmegaConf.create({'data': {'left_hip_idx':4, 'right_hip_idx':5,
            'neck_idx':12, 'face_sign':False}, 'model': {'arch': {'hidden_dim':32}}})
        trainer = CrossViewFusionTrainer(config)
        payload = {'state_dict':trainer.state_dict(), 'hyper_parameters':{'hparams':config}}
        trainer.on_save_checkpoint(payload)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'ski.pt'
            torch.save(payload, path)
            model = ReferenceModel(path)
            model.validate_dataset('ski')
            with self.assertRaisesRegex(ValueError, 'skeleton'):
                model.validate_dataset('unity')
            inputs = torch.randn(2, 5, 13, 3)
            self.assertEqual(model(inputs, inputs).shape, inputs.shape)

    def test_partial_batches_are_sample_weighted(self):
        metric=MetricAccumulator()
        for n, error in [(4,1.),(1,3.)]:
            target=torch.zeros(n,5,13,3)
            pred=target.clone(); pred[...,0]=error
            metric.update(pred,target)
        result=metric.result()
        self.assertAlmostEqual(result['mpjpe'],1.4,places=6)
        self.assertEqual(result['sample_count'],5)
        self.assertEqual(result['acceleration_error'],0.)

    def test_acceleration_does_not_cross_sequence_boundaries(self):
        target=torch.zeros(2,4,13,3)
        pred=target.clone(); pred[1,...,0]=100
        metric=MetricAccumulator(); metric.update(pred,target)
        self.assertEqual(metric.result()['acceleration_error'],0.)

    def test_known_quadratic_acceleration(self):
        target=torch.zeros(1,5,13,3)
        pred=target.clone(); pred[0,:,:,0]=torch.arange(5).square()[:,None]
        metric=MetricAccumulator(); metric.update(pred,target)
        self.assertAlmostEqual(metric.result()['acceleration_error'],2.)

    def test_nonfinite_predictions_fail_instead_of_changing_denominator(self):
        pred=torch.zeros(2,5,13,3); pred[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):
            MetricAccumulator().update(pred,torch.zeros_like(pred))

if __name__=='__main__': unittest.main()
