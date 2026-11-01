#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/models/train_crossview_fusion.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/models
Created Date: Tuesday May 12th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday May 12th 2026 12:14:47 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

# cross_view_canonical_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sim3 import align_points_sim3

# =========================================================
# Utilities
# =========================================================


def rotation_matrix_to_rotvec(R):
    """Principal rotation vectors for proper rotations of shape (B,3,3).

    Recover the best-conditioned quaternion component instead of dividing
    the skew part by sin(theta), which vanishes at both zero and pi. At pi
    the axis sign is inherently ambiguous; the dominant component fixes it.
    """
    r00, r11, r22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    terms = torch.stack([
        1 + r00 + r11 + r22,
        1 + r00 - r11 - r22,
        1 - r00 + r11 - r22,
        1 - r00 - r11 + r22,
    ], dim=-1)  # four times the squared quaternion components
    x = R[:, 2, 1] - R[:, 1, 2]
    y = R[:, 0, 2] - R[:, 2, 0]
    z = R[:, 1, 0] - R[:, 0, 1]
    xy = R[:, 0, 1] + R[:, 1, 0]
    xz = R[:, 0, 2] + R[:, 2, 0]
    yz = R[:, 1, 2] + R[:, 2, 1]
    candidates = torch.stack([
        torch.stack([terms[:, 0], x, y, z], dim=-1),
        torch.stack([x, terms[:, 1], xy, xz], dim=-1),
        torch.stack([y, xy, terms[:, 2], yz], dim=-1),
        torch.stack([z, xz, yz, terms[:, 3]], dim=-1),
    ], dim=1)
    choice = terms.argmax(dim=-1)
    # For a proper rotation the chosen term is >= 1. Gather before sqrt
    # so unused zero components cannot introduce infinite gradients.
    selected = terms.gather(1, choice[:, None])
    quaternion = candidates.gather(1, choice[:, None, None].expand(-1, 1, 4))[:, 0]
    quaternion = F.normalize(quaternion / (2 * selected.sqrt()), dim=-1)
    quaternion = torch.where(quaternion[:, :1] < 0, -quaternion, quaternion)
    vector = quaternion[:, 1:]
    angle = 2 * torch.atan2(torch.linalg.vector_norm(vector, dim=-1), quaternion[:, 0])
    # sin(angle/2)/(angle/2) has a finite, differentiable limit at zero.
    return vector * (2 / torch.sinc(angle / (2 * torch.pi)))[:, None]


def compute_velocity(x):
    """
    x: (B,T,J,C)
    """
    vel = x[:, 1:] - x[:, :-1]
    vel = F.pad(vel, (0, 0, 0, 0, 1, 0))
    return vel


# =========================================================
# Joint Encoder
# =========================================================


class JointEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        x: (B,T,J,C)
        """
        return self.net(x)


# =========================================================
# Cross View Attention
# =========================================================


class CrossViewAttention(nn.Module):

    def __init__(
        self,
        dim=128,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.left_to_right = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.right_to_left = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_l = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)

    def forward(self, feat_l, feat_r):
        """
        feat_l: (B,T,J,C)
        feat_r: (B,T,J,C)
        """

        B, T, J, C = feat_l.shape

        feat_l = feat_l.reshape(B * T, J, C)
        feat_r = feat_r.reshape(B * T, J, C)

        # left query -> right
        left_ctx, attn_l = self.left_to_right(
            query=feat_l,
            key=feat_r,
            value=feat_r,
        )

        # right query -> left
        right_ctx, attn_r = self.right_to_left(
            query=feat_r,
            key=feat_l,
            value=feat_l,
        )

        feat_l = self.norm_l(feat_l + left_ctx)
        feat_r = self.norm_r(feat_r + right_ctx)

        feat_l = feat_l.reshape(B, T, J, C)
        feat_r = feat_r.reshape(B, T, J, C)

        return feat_l, feat_r, attn_l, attn_r


# =========================================================
# Temporal Refiner
# =========================================================


class TemporalRefiner(nn.Module):

    def __init__(
        self,
        dim=128,
        kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.norm = nn.LayerNorm(dim)

        self.conv = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size,
                padding=padding,
                groups=dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        """
        x: (B,T,J,C)
        """

        B, T, J, C = x.shape

        residual = x

        x = self.norm(x)

        # (B,T,J,C)
        # -> (B,J,C,T)
        x = x.permute(0, 2, 3, 1)

        # -> (B*J,C,T)
        x = x.reshape(B * J, C, T)

        x = self.conv(x)

        # -> (B,J,C,T)
        x = x.reshape(B, J, C, T)

        # -> (B,T,J,C)
        x = x.permute(0, 3, 1, 2)

        return residual + x


# =========================================================
# Main Fusion Network
# =========================================================


class CrossViewCanonicalFusion(nn.Module):
    """
    INPUT:
        left_canon              (B,T,J,3)
        right_canon             (B,T,J,3)

        left_to_right_canon     (B,T,J,3)
        right_to_left_canon     (B,T,J,3)

        rotvec_lr               (B,T,3)
        rotvec_rl               (B,T,3)

    OUTPUT:
        fused pose              (B,T,J,3)
    """

    def __init__(
        self,
        hidden_dim=128,
        num_heads=4,
        dropout=0.1,
        temporal_kernel_size=5,
        disable_aligned: bool = False,
        disable_residual: bool = False,
        disable_velocity: bool = False,
        disable_rotvec: bool = False,
        gate_mode: str = "dynamic",
        disable_output_residual: bool = False,
    ):
        super().__init__()

        # feature:
        #
        # left pose
        # right pose
        # left_to_right
        # right_to_left
        # diff_lr
        # diff_rl
        # velocity_l
        # velocity_r
        # rotvec_lr
        # rotvec_rl
        # 共 10 个特征，每个特征 3 维，总共 30 维输入特征。

        self.encoder = JointEncoder(
            in_dim=15,
            hidden_dim=hidden_dim,
        )

        self.cross_attn = CrossViewAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.temporal = TemporalRefiner(
            dim=hidden_dim,
            kernel_size=temporal_kernel_size,
            dropout=dropout,
        )

        # joint-wise fusion gate
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # residual correction
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        # ablation flags
        self.disable_aligned = bool(disable_aligned)
        self.disable_residual = bool(disable_residual)
        self.disable_velocity = bool(disable_velocity)
        self.disable_rotvec = bool(disable_rotvec)
        if gate_mode not in {"dynamic", "mean", "left", "right"}:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")
        self.gate_mode = gate_mode
        self.disable_output_residual = bool(disable_output_residual)
        # Keep state-dict keys compatible with archived checkpoints.
        if gate_mode != "dynamic":
            self.gate_head.requires_grad_(False)
        if self.disable_output_residual:
            self.residual_head.requires_grad_(False)
            if gate_mode != "dynamic":
                # Fixed candidate blending has no learned output path.
                self.requires_grad_(False)

    def forward(
        self,
        left_canon,
        right_canon,
        *,
        return_components: bool = False,
    ):

        # =====================================================
        # motion feature
        # =====================================================

        vel_l = compute_velocity(left_canon)
        vel_r = compute_velocity(right_canon)
        # optionally disable velocity feature
        if getattr(self, "disable_velocity", False):
            vel_l = torch.zeros_like(vel_l)
            vel_r = torch.zeros_like(vel_r)

        # =====================================================
        # sequence-level Sim3 alignment
        # =====================================================

        B, T, J, _ = left_canon.shape

        left_flat = left_canon.reshape(B, T * J, 3)
        right_flat = right_canon.reshape(B, T * J, 3)

        left_to_right_flat, info_lr = align_points_sim3(
            left_flat,
            right_flat,
        )

        right_to_left_flat, info_rl = align_points_sim3(
            right_flat,
            left_flat,
        )

        left_to_right_canon = left_to_right_flat.reshape(B, T, J, 3)
        right_to_left_canon = right_to_left_flat.reshape(B, T, J, 3)
        # optionally disable aligned opposite-view poses
        if getattr(self, "disable_aligned", False):
            left_to_right_canon = torch.zeros_like(left_to_right_canon)
            right_to_left_canon = torch.zeros_like(right_to_left_canon)

        # =====================================================
        # cross-view residual
        # =====================================================

        diff_lr = left_to_right_canon - right_canon
        diff_rl = right_to_left_canon - left_canon
        # optionally disable explicit residuals
        if getattr(self, "disable_residual", False):
            diff_lr = torch.zeros_like(diff_lr)
            diff_rl = torch.zeros_like(diff_rl)

        # =====================================================
        # rotation feature
        # =====================================================

        rot_lr = info_lr["rotation"]  # (B,3,3)
        rot_rl = info_rl["rotation"]

        rotvec_lr = rotation_matrix_to_rotvec(rot_lr)
        rotvec_rl = rotation_matrix_to_rotvec(rot_rl)

        # (B,3)
        # -> (B,T,3)
        rotvec_lr = rotvec_lr[:, None, :].repeat(1, T, 1)
        rotvec_rl = rotvec_rl[:, None, :].repeat(1, T, 1)

        # -> (B,T,J,3)
        rotvec_lr = rotvec_lr[:, :, None, :].repeat(1, 1, J, 1)
        rotvec_rl = rotvec_rl[:, :, None, :].repeat(1, 1, J, 1)

        # apply ablation switches: optionally zero out specific features
        if self.disable_rotvec:
            rotvec_lr = torch.zeros_like(rotvec_lr)
            rotvec_rl = torch.zeros_like(rotvec_rl)

        # =====================================================
        # feature concat
        # =====================================================

        feat_l = torch.cat(
            [
                left_canon,
                right_to_left_canon,
                diff_rl,
                vel_l,
                rotvec_rl,
            ],
            dim=-1,
        )

        feat_r = torch.cat(
            [
                right_canon,
                left_to_right_canon,
                diff_lr,
                vel_r,
                rotvec_lr,
            ],
            dim=-1,
        )

        # =====================================================
        # encoder
        # =====================================================

        feat_l = self.encoder(feat_l)
        feat_r = self.encoder(feat_r)

        # =====================================================
        # cross-view attention
        # =====================================================

        feat_l, feat_r, attn_l, attn_r = self.cross_attn(
            feat_l,
            feat_r,
        )

        # =====================================================
        # temporal refinement
        # =====================================================

        feat_l = self.temporal(feat_l)
        feat_r = self.temporal(feat_r)

        # =====================================================
        # fusion
        # =====================================================

        fusion_feat = torch.cat(
            [
                feat_l,
                feat_r,
            ],
            dim=-1,
        )

        if self.gate_mode == "dynamic":
            alpha = torch.sigmoid(self.gate_head(fusion_feat))
        else:
            value = {"mean": 0.5, "left": 1.0, "right": 0.0}[self.gate_mode]
            alpha = fusion_feat.new_full((*fusion_feat.shape[:-1], 1), value)

        base_l = 0.5 * (left_canon + right_to_left_canon)

        base_r = 0.5 * (right_canon + left_to_right_canon)

        fused = alpha * base_l + (1.0 - alpha) * base_r

        residual = (
            torch.zeros_like(fused)
            if self.disable_output_residual
            else self.residual_head(fusion_feat)
        )

        fused = fused + residual

        aux = {
            "alpha": alpha,
            "attn_l": attn_l,
            "attn_r": attn_r,
        }

        if return_components:
            aux.update(base_l=base_l, base_r=base_r, output_residual=residual)

        return fused, aux


# =========================================================
# Loss
# =========================================================


def mpjpe_loss(pred, target):

    return torch.norm(
        pred - target,
        dim=-1,
    ).mean()


def temporal_smooth_loss(pred):

    if pred.shape[1] < 3:
        return pred.new_tensor(0.0)

    acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]

    return torch.norm(acc, dim=-1).mean()


# =========================================================
# Example
# =========================================================

if __name__ == "__main__":

    B, T, J = 2, 32, 15

    left_canon = torch.randn(B, T, J, 3)
    right_canon = torch.randn(B, T, J, 3)

    left_to_right_canon = torch.randn(B, T, J, 3)
    right_to_left_canon = torch.randn(B, T, J, 3)

    rotvec_lr = torch.randn(B, T, 3)
    rotvec_rl = torch.randn(B, T, 3)

    gt = torch.randn(B, T, J, 3)

    model = CrossViewCanonicalFusion()

    fused, aux = model(
        left_canon,
        right_canon,
    )

    loss_pose = mpjpe_loss(fused, gt)
    loss_temp = temporal_smooth_loss(fused)

    loss = loss_pose + 0.02 * loss_temp

    print("fused:", fused.shape)
    print("alpha:", aux["alpha"].shape)
    print("loss:", loss.item())
