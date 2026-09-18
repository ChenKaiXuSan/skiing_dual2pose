"""Gate interventions must preserve the archived model's default computation."""
import unittest
import torch
from dual2pose.models.crossview_fusion import CrossViewCanonicalFusion


class GateInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.left = torch.randn(2, 6, 15, 3)
        self.right = torch.randn(2, 6, 15, 3)

    def model(self, **kwargs):
        return CrossViewCanonicalFusion(hidden_dim=16, num_heads=4, dropout=0, **kwargs).eval()

    def test_default_diagnostics_reconstruct_prediction_without_changing_output(self):
        model = self.model()
        with torch.no_grad():
            output, original_aux = model(self.left, self.right)
            detailed, aux = model(self.left, self.right, return_components=True)
        self.assertEqual(set(original_aux), {"alpha", "attn_l", "attn_r"})
        torch.testing.assert_close(output, detailed, rtol=0, atol=0)
        torch.testing.assert_close(
            detailed, aux["alpha"] * aux["base_l"] +
            (1 - aux["alpha"]) * aux["base_r"] + aux["output_residual"],
            rtol=0, atol=0,
        )

    def test_fixed_gate_selects_candidates_not_raw_views(self):
        for mode, value in (("mean", .5), ("left", 1.), ("right", 0.)):
            with self.subTest(mode=mode):
                model = self.model(gate_mode=mode, disable_output_residual=True)
                with torch.no_grad():
                    output, aux = model(self.left, self.right, return_components=True)
                self.assertTrue(torch.all(aux["alpha"] == value))
                torch.testing.assert_close(
                    output, value * aux["base_l"] + (1 - value) * aux["base_r"],
                    rtol=0, atol=0,
                )
                if mode == "left":
                    self.assertFalse(torch.allclose(output, self.left))
                self.assertFalse(any(p.requires_grad for p in model.gate_head.parameters()))

    def test_output_residual_switch_is_distinct_from_input_difference_switch(self):
        model = self.model(disable_residual=True)
        with torch.no_grad():
            for p in model.residual_head.parameters():
                p.zero_()
            model.residual_head[-1].bias.fill_(2.)
            output, aux = model(self.left, self.right, return_components=True)
            model.disable_output_residual = True
            without, no_aux = model(self.left, self.right, return_components=True)
        torch.testing.assert_close(output - without, torch.full_like(output, 2.))
        self.assertTrue(torch.all(no_aux["output_residual"] == 0))

    def test_old_checkpoint_strictly_loads_in_all_four_configurations(self):
        state = self.model().state_dict()
        for mode in ("dynamic", "mean"):
            for disabled in (False, True):
                model = self.model(gate_mode=mode, disable_output_residual=disabled)
                model.load_state_dict(state, strict=True)
                output, _ = model(self.left, self.right)
                self.assertTrue(torch.isfinite(output).all())
                if mode == "mean" and disabled:
                    self.assertFalse(output.requires_grad)
                    self.assertFalse(any(p.requires_grad for p in model.parameters()))
                else:
                    output.square().mean().backward()
                    self.assertIsNotNone(next(model.encoder.parameters()).grad)
                if disabled:
                    self.assertFalse(any(p.requires_grad for p in model.residual_head.parameters()))

    def test_invalid_gate_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model(gate_mode="typo")


if __name__ == "__main__":
    unittest.main()
