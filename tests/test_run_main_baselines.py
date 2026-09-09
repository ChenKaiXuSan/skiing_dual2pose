import unittest
import torch
from dual2pose.experiments.run_main_baselines import MetricAccumulator

class MainBaselineMetricTests(unittest.TestCase):
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
