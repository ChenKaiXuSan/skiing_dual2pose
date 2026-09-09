import unittest

import torch
from torch import nn

from dual2pose.models.main_baselines import (
    aligned_average,
    build_baseline,
    quality_weighted_fusion,
)


def _rotation_z(angle: float, dtype: torch.dtype) -> torch.Tensor:
    angle_tensor = torch.tensor(angle, dtype=dtype)
    cosine = torch.cos(angle_tensor)
    sine = torch.sin(angle_tensor)
    return torch.stack(
        [
            torch.stack([cosine, -sine, cosine.new_zeros(())]),
            torch.stack([sine, cosine, cosine.new_zeros(())]),
            torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
        ]
    )


def _static_pose(num_joints: int, frames: int = 30) -> torch.Tensor:
    coordinates = torch.arange(num_joints * 3, dtype=torch.float32).reshape(
        num_joints, 3
    )
    coordinates[:, 0] *= 0.07
    coordinates[:, 1] = torch.sin(coordinates[:, 1])
    coordinates[:, 2] = torch.cos(coordinates[:, 2])
    return coordinates.view(1, 1, num_joints, 3).repeat(1, frames, 1, 1)


class AlignedAverageTest(unittest.TestCase):
    """Breaks caught: fitting each frame or mapping left into the right frame."""

    def test_sequence_sim3_recovers_left_pose_before_averaging(self) -> None:
        generator = torch.Generator().manual_seed(7)
        left = torch.randn((2, 8, 13, 3), generator=generator, dtype=torch.float64)
        rotation = _rotation_z(0.63, left.dtype)
        right = 1.7 * (left @ rotation.T) + torch.tensor(
            [2.0, -0.5, 1.25], dtype=left.dtype
        )

        actual = aligned_average(left, right)

        self.assertTrue(torch.allclose(actual, left, atol=1e-7, rtol=1e-7))

    def test_zero_variance_sequences_are_finite(self) -> None:
        pose = torch.zeros((2, 30, 15, 3), dtype=torch.float32)
        actual = aligned_average(pose, pose)
        self.assertTrue(torch.equal(actual, pose))
        self.assertTrue(torch.isfinite(actual).all())


class QualityWeightedFusionTest(unittest.TestCase):
    """Breaks caught: quality weights reward jitter or use the wrong skeleton."""

    def test_temporal_jitter_and_bone_changes_reduce_a_views_weight(self) -> None:
        left = _static_pose(13)
        right = left.clone()
        alternating = torch.where(
            torch.arange(30) % 2 == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        )
        right[0, :, 2, 0] += 2.0 * alternating

        actual = quality_weighted_fusion(left, right)
        equal_average = (left + right) * 0.5

        self.assertLess(
            torch.mean((actual - left) ** 2).item(),
            torch.mean((equal_average - left) ** 2).item(),
        )
        self.assertTrue(torch.isfinite(actual).all())

    def test_identical_static_and_degenerate_poses_remain_unchanged(self) -> None:
        for num_joints in (13, 15):
            with self.subTest(num_joints=num_joints):
                static = _static_pose(num_joints)
                self.assertTrue(
                    torch.allclose(
                        quality_weighted_fusion(static, static), static, atol=1e-7
                    )
                )

                zeros = torch.zeros_like(static)
                actual = quality_weighted_fusion(zeros, zeros)
                self.assertTrue(torch.equal(actual, zeros))
                self.assertTrue(torch.isfinite(actual).all())


class _OfficialSmoothNetResBlock(nn.Module):
    """Literal reference topology from cure-lab/SmoothNet, for parity testing."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=0.25, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        output = self.linear1(inputs)
        output = self.dropout(output)
        output = self.lrelu(output)
        output = self.linear2(output)
        output = self.dropout(output)
        output = self.lrelu(output)
        return output + identity


class _OfficialSmoothNet(nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 512), nn.LeakyReLU(0.1, inplace=True)
        )
        self.res_blocks = nn.Sequential(
            *[_OfficialSmoothNetResBlock(512, 128) for _ in range(5)]
        )
        self.decoder = nn.Linear(512, window_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.res_blocks(self.encoder(inputs.float())))


class LearnedBaselineTest(unittest.TestCase):
    """Breaks caught: per-joint processing, wrong layout, or detached predictions."""

    def test_all_models_preserve_pose_shape_and_backpropagate_to_both_views(self) -> None:
        for name in ("mlp", "tcn", "smoothnet"):
            with self.subTest(name=name):
                torch.manual_seed(11)
                model = build_baseline(name, num_joints=13, time_window=30)
                left = torch.randn((2, 30, 13, 3), requires_grad=True)
                right = torch.randn((2, 30, 13, 3), requires_grad=True)

                output = model(left, right)
                self.assertEqual(output.shape, left.shape)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()

                self.assertIsNotNone(left.grad)
                self.assertIsNotNone(right.grad)
                self.assertGreater(left.grad.abs().sum().item(), 0.0)
                self.assertGreater(right.grad.abs().sum().item(), 0.0)
                parameter_grads = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(all(grad is not None for grad in parameter_grads))
                self.assertTrue(all(torch.isfinite(grad).all() for grad in parameter_grads))
                self.assertTrue(any(grad.abs().sum() > 0 for grad in parameter_grads))

    def test_dropout_is_disabled_in_evaluation_mode(self) -> None:
        left = torch.randn((2, 30, 13, 3))
        right = torch.randn((2, 30, 13, 3))
        for name in ("mlp", "tcn", "smoothnet"):
            with self.subTest(name=name):
                model = build_baseline(name, num_joints=13, time_window=30).eval()
                first = model(left, right)
                second = model(left, right)
                self.assertTrue(torch.equal(first, second))

    def test_smoothnet_matches_official_layer_order_and_widths(self) -> None:
        torch.manual_seed(23)
        model = build_baseline("smoothnet", num_joints=15, time_window=30).eval()
        reference = _OfficialSmoothNet(window_size=30).eval()
        reference.load_state_dict(model.state_dict())
        left = torch.randn((2, 30, 15, 3))
        right = torch.randn((2, 30, 15, 3))
        average_bct = ((left + right) * 0.5).flatten(2).transpose(1, 2)

        expected = reference(average_bct).transpose(1, 2).reshape_as(left)
        actual = model(left, right)

        self.assertTrue(torch.equal(actual, expected))

    def test_models_reject_targets_and_wrong_sequence_shapes(self) -> None:
        model = build_baseline("mlp", num_joints=13, time_window=30)
        valid = torch.zeros((1, 30, 13, 3))
        with self.assertRaises(TypeError):
            model(valid, valid, valid)
        with self.assertRaisesRegex(ValueError, "Expected left/right shape"):
            model(valid[:, :-1], valid[:, :-1])
        with self.assertRaisesRegex(ValueError, "Unsupported baseline"):
            build_baseline("oracle", num_joints=13)


if __name__ == "__main__":
    unittest.main()
