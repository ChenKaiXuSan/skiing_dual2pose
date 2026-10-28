#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/dataloader/canonicalize.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/dataloader
Created Date: Sunday May 10th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday May 11th 2026 6:59:18 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

"""Body-frame canonicalization of 3-D poses.

The CanonFuse3D training protocol is fixed to ``mode="per_frame"``: every frame
of both input streams and of the supervision target is centred at its own
pelvis and rotated into its own hip/neck body frame, so the representation is
independent of batch composition and carries no global motion.

``mode="first_frame"`` (the default of this library function) is kept only for
the archived manuscript analyses, which share one reference frame per batch:
the non-learned baselines and the native-output audit.  It is not a training
option; the trainer always passes ``mode="per_frame"``.
"""

import torch

CANONICAL_MODES = ("first_frame", "per_frame")


def validate_canonical_mode(mode) -> str:
    """Return ``mode`` as a plain string or raise if it is not a known mode."""
    mode_str = str(mode)
    if mode_str not in CANONICAL_MODES:
        raise ValueError(
            f"canonical_mode must be one of {CANONICAL_MODES}, got {mode!r}"
        )
    return mode_str


def _normalize_torch(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def canonicalize_pose_torch(
    x: torch.Tensor,
    left_hip: int = 6,
    right_hip: int = 7,
    neck: int = 14,
    mode: str = "first_frame",
    eps: float = 1e-8,
    enforce_face_z_positive: bool = True,
    left_eye: int = 0,
    right_eye: int = 1,
):
    """Express poses in a body frame built from the hips, the neck and the eyes.

    x: (J, 3) or (T, J, 3) or (B, T, J, 3)
    mode:
        "first_frame": one reference, the pelvis and body axes of the first
            frame of the flattened input, shared by every frame.  With a
            (B, T, J, 3) batch that is the first frame of the FIRST sequence,
            so the result depends on batch composition and order.  Archived
            manuscript protocol only.
        "per_frame": every frame is centred at its own pelvis and rotated into
            its own body frame (per-frame person centering); independent of
            batching.  The training protocol.
    enforce_face_z_positive:
        force +Z to point along (eye midpoint - neck) when eye indices are given.
    Returns:
        x_canon: (B, T, J, 3) for 3-D/4-D input, (J, 3) for 2-D input
        transform: dict with "pelvis" and "R".  For "per_frame" they carry one
            entry per frame, shaped like the input's leading dimensions.
    """
    mode = validate_canonical_mode(mode)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    else:
        x = x.float()

    original_shape = x.shape
    squeeze_time = False

    if x.ndim == 2:  # (J, 3) -> (1, J, 3)
        x = x.unsqueeze(0)
        squeeze_time = True
        b, t = 1, 1
        j, c = original_shape
    elif x.ndim == 3:  # (T, J, 3)
        b, t = 1, x.shape[0]
        j, c = x.shape[1], x.shape[2]
    elif x.ndim == 4:  # (B, T, J, 3)
        b, t, j, c = x.shape
        x = x.reshape(b * t, j, c)
    else:
        raise ValueError("x must have shape (J,3), (T,J,3) or (B,T,J,3)")

    # x is now (N, J, 3) where N = B*T or T or 1
    left_hip_pos = x[:, left_hip]  # (N, 3)
    right_hip_pos = x[:, right_hip]  # (N, 3)
    neck_pos = x[:, neck]  # (N, 3)

    pelvis = (left_hip_pos + right_hip_pos) / 2.0  # (N, 3)
    x_centered = x - pelvis[:, None, :]  # (N, J, 3)

    x_axis = _normalize_torch(right_hip_pos - left_hip_pos, eps)  # body right
    y_axis = _normalize_torch(neck_pos - pelvis, eps)  # body up
    z_axis = _normalize_torch(torch.linalg.cross(x_axis, y_axis), eps)  # forward/back

    if enforce_face_z_positive and left_eye is not None and right_eye is not None:
        eye_mid = (x[:, left_eye] + x[:, right_eye]) / 2.0  # (N, 3)
        face_dir = _normalize_torch(eye_mid - neck_pos, eps)  # (N, 3)
        sign = torch.sign((z_axis * face_dir).sum(dim=-1, keepdim=True))  # (N,1)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        z_axis = z_axis * sign

    # R: (N, 3, 3), columns are [x_axis, y_axis, z_axis]
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3)

    if mode == "first_frame":
        ref_pelvis = pelvis[0]  # (3,)
        ref_R = R[0]  # (3, 3)
        x_canon = (x - ref_pelvis[None, None, :]) @ ref_R  # (N, J, 3)
        transform = {"pelvis": ref_pelvis, "R": ref_R}
    else:  # per_frame
        x_canon = torch.einsum("njc,nck->njk", x_centered, R)  # (N, J, 3)
        if squeeze_time:
            transform = {"pelvis": pelvis[0], "R": R[0]}
        elif len(original_shape) == 4:
            transform = {"pelvis": pelvis.reshape(b, t, 3), "R": R.reshape(b, t, 3, 3)}
        else:
            transform = {"pelvis": pelvis, "R": R}

    if squeeze_time:
        x_canon = x_canon[0]  # (J, 3)
    else:
        x_canon = x_canon.reshape(b, t, j, c)

    return x_canon, transform
