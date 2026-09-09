"""Pose-only reference baselines for the main IVC comparison.

Every public method consumes independently canonicalized left/right poses with
shape ``(B, T, J, 3)``.  These baselines intentionally accept no target pose,
external confidence, image, or calibration input.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from .sim3 import align_points_sim3


# Joint orders are documented in dual2pose.map_config.  The 13-joint skeleton
# is the 15-joint skeleton after removing the two eye joints and reindexing.
SKELETON_EDGES: Final[dict[int, tuple[tuple[int, int], ...]]] = {
    15: (
        (14, 2),
        (2, 4),
        (4, 13),
        (14, 3),
        (3, 5),
        (5, 12),
        (14, 6),
        (14, 7),
        (6, 8),
        (8, 10),
        (7, 9),
        (9, 11),
    ),
    13: (
        (12, 0),
        (0, 2),
        (2, 11),
        (12, 1),
        (1, 3),
        (3, 10),
        (12, 4),
        (12, 5),
        (4, 6),
        (6, 8),
        (5, 7),
        (7, 9),
    ),
}


def _validate_pose_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    num_joints: int | None = None,
    time_window: int | None = None,
) -> None:
    expected = "Expected left/right shape (B,T,J,3)"
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError(f"{expected}, got non-tensor input")
    if left.shape != right.shape or left.ndim != 4 or left.shape[-1] != 3:
        raise ValueError(f"{expected}, got {tuple(left.shape)} and {tuple(right.shape)}")
    if not left.is_floating_point() or not right.is_floating_point():
        raise TypeError(f"{expected} with floating-point values")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("left and right must have the same device and dtype")
    if num_joints is not None and left.shape[2] != num_joints:
        raise ValueError(
            f"{expected} with J={num_joints}, got {tuple(left.shape)}"
        )
    if time_window is not None and left.shape[1] != time_window:
        raise ValueError(
            f"{expected} with T={time_window}, got {tuple(left.shape)}"
        )


def aligned_average(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Align each complete right sequence to the left with Sim(3), then average.

    A single transform is estimated from all ``T * J`` correspondences in each
    batch item.  Constant/zero-variance inputs remain finite through the
    regularized repository Sim(3) estimator.
    """

    _validate_pose_pair(left, right)
    aligned_right, _ = align_points_sim3(right, left)
    return (left + aligned_right) * 0.5


def _view_quality(pose: torch.Tensor, edges: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Return a dimensionless sequence quality cost from 3-D motion alone."""

    first = torch.tensor([edge[0] for edge in edges], device=pose.device)
    second = torch.tensor([edge[1] for edge in edges], device=pose.device)
    lengths = (pose[:, :, first] - pose[:, :, second]).norm(dim=-1)

    # Normalize both terms by a robust body scale so coordinate units do not
    # affect view selection. Degenerate zero skeletons have zero cost.
    body_scale = lengths.median(dim=1).values.median(dim=1).values
    body_scale = body_scale.clamp_min(torch.finfo(pose.dtype).eps)

    reference_lengths = lengths.median(dim=1, keepdim=True).values
    bone_cost = (lengths - reference_lengths).abs().mean(dim=(1, 2)) / body_scale

    if pose.shape[1] >= 3:
        acceleration = pose[:, 2:] - 2.0 * pose[:, 1:-1] + pose[:, :-2]
        acceleration_cost = acceleration.norm(dim=-1).mean(dim=(1, 2)) / body_scale
    else:
        acceleration_cost = body_scale.new_zeros(body_scale.shape)
    return acceleration_cost + bone_cost


def quality_weighted_fusion(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Fuse views using only acceleration and bone-length consistency.

    One scalar weight per sequence and view is obtained by applying a softmax
    to the negative dimensionless quality costs.  No ground truth, learned
    confidence, cross-view error, image evidence, or calibration is used.
    """

    _validate_pose_pair(left, right)
    num_joints = int(left.shape[2])
    if num_joints not in SKELETON_EDGES:
        raise ValueError(
            "quality_weighted_fusion supports the documented 13- and "
            f"15-joint skeletons, got J={num_joints}"
        )
    edges = SKELETON_EDGES[num_joints]
    costs = torch.stack([_view_quality(left, edges), _view_quality(right, edges)], dim=1)
    # A completely collapsed skeleton is missing information, not a stable
    # observation. Reject it when the other stream has spatial extent; retain
    # equal weights if both streams are collapsed so the output stays finite.
    valid = torch.stack([
        (pose.amax(dim=2) - pose.amin(dim=2)).abs().amax(dim=(1, 2))
        > torch.finfo(pose.dtype).eps for pose in (left, right)
    ], dim=1)
    allowed = valid | ~valid.any(dim=1, keepdim=True)
    costs = costs.masked_fill(~allowed, float("inf"))
    weights = torch.softmax(-costs, dim=1)
    left_weight = weights[:, 0].view(-1, 1, 1, 1)
    right_weight = weights[:, 1].view(-1, 1, 1, 1)
    return left_weight * left + right_weight * right


class _PosePairBaseline(nn.Module):
    def __init__(self, num_joints: int, time_window: int) -> None:
        super().__init__()
        if num_joints <= 0 or time_window <= 0:
            raise ValueError("num_joints and time_window must be positive")
        self.num_joints = int(num_joints)
        self.time_window = int(time_window)

    def _validate(self, left: torch.Tensor, right: torch.Tensor) -> None:
        _validate_pose_pair(
            left,
            right,
            num_joints=self.num_joints,
            time_window=self.time_window,
        )


class ResidualCrossViewMLP(_PosePairBaseline):
    """Framewise full-pose MLP predicting a residual over the view average."""

    def __init__(
        self,
        num_joints: int,
        time_window: int = 30,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_joints, time_window)
        input_size = 2 * num_joints * 3
        output_size = num_joints * 3
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        self._validate(left, right)
        batch_size, frames = left.shape[:2]
        features = torch.cat([left.flatten(2), right.flatten(2)], dim=-1)
        residual = self.network(features.reshape(batch_size * frames, -1))
        residual = residual.reshape(batch_size, frames, self.num_joints, 3)
        return (left + right) * 0.5 + residual


class TemporalConvolutionFusion(_PosePairBaseline):
    """Full-pose temporal convolution predicting a residual over the average."""

    def __init__(
        self,
        num_joints: int,
        time_window: int = 30,
        hidden_size: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_joints, time_window)
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        input_channels = 2 * num_joints * 3
        output_channels = num_joints * 3
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, output_channels, kernel_size, padding=padding),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        self._validate(left, right)
        batch_size, frames = left.shape[:2]
        features = torch.cat([left.flatten(2), right.flatten(2)], dim=-1)
        residual = self.network(features.transpose(1, 2))
        residual = residual.transpose(1, 2).reshape(
            batch_size, frames, self.num_joints, 3
        )
        return (left + right) * 0.5 + residual


class SmoothNetResBlock(nn.Module):
    """Residual block faithfully ported from the official SmoothNet source."""

    def __init__(
        self, in_channels: int, hidden_channels: int, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        output = self.linear1(inputs)
        output = self.dropout(output)
        output = self.lrelu(output)
        output = self.linear2(output)
        output = self.dropout(output)
        output = self.lrelu(output)
        return output + identity


class SmoothNetBaseline(_PosePairBaseline):
    """Official SmoothNet topology applied to the canonical view average.

    Architecture source: https://github.com/cure-lab/SmoothNet/blob/main/
    lib/models/smoothnet.py.  The official H36M-FCN 3-D widths are retained
    (512 encoder width, 128 residual width, five blocks, dropout 0.25); only
    the input/output temporal window is explicitly adapted to 30 frames.
    """

    def __init__(
        self,
        num_joints: int,
        time_window: int = 30,
        hidden_size: int = 512,
        res_hidden_size: int = 128,
        num_blocks: int = 5,
        dropout: float = 0.25,
    ) -> None:
        super().__init__(num_joints, time_window)
        self.encoder = nn.Sequential(
            nn.Linear(time_window, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[
                SmoothNetResBlock(hidden_size, res_hidden_size, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.decoder = nn.Linear(hidden_size, time_window)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        self._validate(left, right)
        batch_size, frames = left.shape[:2]
        average = (left + right) * 0.5
        features = average.flatten(2).transpose(1, 2).float()
        output = self.encoder(features)
        output = self.res_blocks(output)
        output = self.decoder(output)
        return output.transpose(1, 2).reshape(
            batch_size, frames, self.num_joints, 3
        )


def build_baseline(
    name: str, num_joints: int, time_window: int = 30
) -> nn.Module:
    """Build a learned pose-only baseline by its experiment-table name."""

    normalized_name = name.strip().lower()
    builders: dict[str, type[_PosePairBaseline]] = {
        "mlp": ResidualCrossViewMLP,
        "tcn": TemporalConvolutionFusion,
        "smoothnet": SmoothNetBaseline,
    }
    if normalized_name not in builders:
        supported = ", ".join(sorted(builders))
        raise ValueError(f"Unsupported baseline '{name}'. Choose one of: {supported}")
    return builders[normalized_name](num_joints=num_joints, time_window=time_window)


__all__ = [
    "SKELETON_EDGES",
    "ResidualCrossViewMLP",
    "TemporalConvolutionFusion",
    "SmoothNetResBlock",
    "SmoothNetBaseline",
    "aligned_average",
    "quality_weighted_fusion",
    "build_baseline",
]
