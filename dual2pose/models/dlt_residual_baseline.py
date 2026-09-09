"""Learned residual baseline over independently canonicalized DLT poses."""

from __future__ import annotations

import torch
from torch import nn


def ski_predicted_pelvis_relative(dlt: torch.Tensor) -> torch.Tensor:
    """Remove calibrated-world translation using only the predicted hip midpoint."""
    if (
        not isinstance(dlt, torch.Tensor)
        or dlt.ndim != 4
        or dlt.shape[2:] != (13, 3)
    ):
        raise ValueError(
            f"Expected Ski DLT shape (B,T,13,3), got {getattr(dlt, 'shape', None)}"
        )
    if not dlt.is_floating_point() or not torch.isfinite(dlt).all():
        raise ValueError("Ski DLT must be finite floating-point coordinates")
    pelvis = (dlt[:, :, 4:5] + dlt[:, :, 5:6]) * 0.5
    return dlt - pelvis


def dlt_residual_features(
    dlt: torch.Tensor,
    reprojection_error_px: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    expected = "Expected DLT shape (B,T,J,3) with matching reprojection/validity"
    if not isinstance(dlt, torch.Tensor) or dlt.ndim != 4 or dlt.shape[-1] != 3:
        raise ValueError(f"{expected}, got {getattr(dlt, 'shape', None)}")
    point_shape = dlt.shape[:-1]
    if reprojection_error_px.shape != point_shape or valid.shape != point_shape:
        raise ValueError(
            f"{expected}, got {tuple(dlt.shape)}, "
            f"{tuple(reprojection_error_px.shape)}, {tuple(valid.shape)}"
        )
    if not dlt.is_floating_point() or not reprojection_error_px.is_floating_point():
        raise TypeError("DLT and reprojection inputs must be floating-point tensors")
    if valid.dtype != torch.bool:
        raise TypeError("DLT validity input must be boolean")
    if dlt.device != reprojection_error_px.device or dlt.device != valid.device:
        raise ValueError("DLT, reprojection, and validity inputs must share a device")
    if not torch.isfinite(dlt).all() or not torch.isfinite(reprojection_error_px).all():
        raise ValueError("DLT and reprojection inputs must be finite")
    if torch.any(reprojection_error_px < 0):
        raise ValueError("Reprojection errors must be nonnegative")
    normalized_reprojection = torch.log1p(reprojection_error_px / 25.0)
    return torch.cat(
        [dlt.flatten(2), normalized_reprojection, valid.to(dtype=dlt.dtype)], dim=-1
    )


def dlt_residual_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Prediction and target must share shape (B,T,J,3)")
    if prediction.shape[1] < 3:
        raise ValueError("At least three frames are required for acceleration loss")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Prediction and target must be finite")
    position = torch.nn.functional.l1_loss(prediction, target)
    acceleration = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
    return position + 0.01 * torch.linalg.vector_norm(acceleration, dim=-1).mean()


class DLTResidualMLP(nn.Module):
    def __init__(
        self,
        num_joints: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_joints <= 0 or hidden_size <= 0:
            raise ValueError("num_joints and hidden_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.num_joints = int(num_joints)
        input_size = 5 * self.num_joints
        output_size = 3 * self.num_joints
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        output_head = self.network[-1]
        nn.init.zeros_(output_head.weight)
        nn.init.zeros_(output_head.bias)

    def forward(
        self,
        dlt: torch.Tensor,
        reprojection_error_px: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if dlt.ndim != 4 or dlt.shape[2] != self.num_joints:
            raise ValueError(
                f"Expected DLT shape (B,T,{self.num_joints},3), got {tuple(dlt.shape)}"
            )
        features = dlt_residual_features(dlt, reprojection_error_px, valid)
        batch_size, frames = dlt.shape[:2]
        residual = self.network(features.reshape(batch_size * frames, -1))
        return dlt + residual.reshape(batch_size, frames, self.num_joints, 3)
