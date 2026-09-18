import importlib.util
import unittest
import warnings
import torch
from omegaconf import OmegaConf


class GateAblationTrainingTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("dual2pose.experiments.run_gate_ablation"),
                             "controlled gate training is not implemented")
        from dual2pose.experiments import run_gate_ablation
        return run_gate_ablation

    def test_loss_matches_original_trainer(self):
        module = self.module()
        from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer
        from dual2pose.eval.main_baseline_data import canonical_batch
        torch.manual_seed(42)
        torch.set_num_threads(1)
        trainer = CrossViewFusionTrainer(OmegaConf.create({"loss": {"lr": .001}})).eval()
        raw = tuple(torch.randn(2, 6, 15, 3) for _ in range(3))
        batch = {"kpt3d_sam": {"cam1": raw[0], "cam2": raw[1]}, "kpt3d_gt": raw[2]}
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            original = trainer._shared_step(batch, "val")
            left, right, truth = canonical_batch(*raw, dataset="unity")
            prediction, aux = trainer.models(left, right)
            actual = module.fusion_objective(prediction, aux["alpha"], left, right, truth)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_only_three_variants_have_trainable_output(self):
        module = self.module()
        self.assertEqual(set(module.TRAIN_VARIANTS), {"full", "fixed_gate", "no_output_residual"})
        for variant in module.TRAIN_VARIANTS:
            model = module.build_model(variant)
            self.assertTrue(any(p.requires_grad for p in model.parameters()))
        with self.assertRaises(ValueError):
            module.build_model("fixed_gate_no_residual")


if __name__ == "__main__":
    unittest.main()
