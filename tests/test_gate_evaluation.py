"""Tests for frozen, raw-input gate interventions and metrics."""
import importlib.util
import unittest
import torch


class GateEvaluationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("dual2pose.eval.evaluate_gate_interventions"),
                             "gate evaluator is not implemented")
        from dual2pose.eval import evaluate_gate_interventions
        return evaluate_gate_interventions

    def test_switch_corruption_preserves_inputs_and_other_half(self):
        module = self.module()
        left = torch.ones(2, 6, 15, 3)
        right = 2 * left
        l, r = module.corrupt_pair(left, right, "switch_left_to_right")
        self.assertTrue(torch.all(l[:, :3] == 0))
        self.assertTrue(torch.all(l[:, 3:] == 1))
        self.assertTrue(torch.all(r[:, :3] == 2))
        self.assertTrue(torch.all(r[:, 3:] == 0))
        self.assertTrue(torch.all(left == 1))
        self.assertTrue(torch.all(right == 2))
        with self.assertRaises(ValueError):
            module.corrupt_pair(left, right, "unknown")

    def test_candidate_blending_uses_candidates_and_optional_output_residual(self):
        module = self.module()
        base = torch.ones(1, 3, 3, 3)
        aux = dict(alpha=base[..., :1] * .25, base_l=base * 2,
                   base_r=base * 6, output_residual=base * 10)
        predictions = module.intervention_predictions(aux)
        expected = {"dynamic_residual": 15., "mean_residual": 14.,
                    "left_residual": 12., "right_residual": 16.,
                    "dynamic_no_residual": 5., "mean_no_residual": 4.,
                    "left_no_residual": 2., "right_no_residual": 6.}
        self.assertEqual(set(predictions), set(expected))
        for key, value in expected.items():
            self.assertTrue(torch.all(predictions[key] == value), key)

    def test_complete_metrics_count_final_partial_batch(self):
        module = self.module()
        from dual2pose.models.crossview_fusion import CrossViewCanonicalFusion
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = CrossViewCanonicalFusion(hidden_dim=16, num_heads=4, dropout=0).eval()
        arrays = tuple(torch.randn(5, 6, 15, 3) for _ in range(3))
        result = module.evaluate_condition(model, arrays, "unity", "clean", batch_size=4)
        for metrics in result["metrics"].values():
            self.assertEqual(metrics["common13"]["sample_count"], 5)
            self.assertEqual(metrics["common13"]["point_count"], 390)
        self.assertEqual(result["gate_diagnostics"]["joint_frame_count"], 450)


if __name__ == "__main__":
    unittest.main()
