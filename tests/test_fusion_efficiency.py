import importlib.util
import unittest
import torch


class FusionEfficiencyTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("dual2pose.eval.profile_fusion_efficiency"),
                             "efficiency profiler is not implemented")
        from dual2pose.eval import profile_fusion_efficiency
        return profile_fusion_efficiency

    def test_parameter_size_excludes_optimizer_and_tracks_freezing(self):
        module = self.module()
        model = torch.nn.Linear(3, 2)
        model.bias.requires_grad_(False)
        self.assertEqual(module.parameter_summary(model),
                         dict(total_parameters=8, trainable_parameters=6, weight_bytes=32))

    def test_convolution_count_is_not_silently_dropped(self):
        module = self.module()
        class PairConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Conv1d(3, 2, 3, bias=False)
            def forward(self, left, right):
                return self.layer(left)
        result = module.measure(PairConv(), torch.ones(1, 3, 5), torch.ones(1, 3, 5),
                                warmup=1, repeats=3)
        self.assertEqual(result["counted_flops"], 108)

    def test_linear_flops_are_partial_not_claimed_as_complete_pipeline(self):
        module = self.module()
        class PairLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3, 2, bias=False)
            def forward(self, left, right):
                return self.layer(left)
        result = module.measure(PairLinear(), torch.ones(1, 3), torch.ones(1, 3),
                                warmup=1, repeats=3)
        self.assertEqual(result["counted_flops"], 12)
        self.assertFalse(result["flops_complete"])
        self.assertEqual(len(result["latency_ms_samples"]), 3)
        self.assertGreater(result["latency_ms_median"], 0)
        self.assertEqual(result["peak_allocated_bytes"], None)


if __name__ == "__main__":
    unittest.main()
