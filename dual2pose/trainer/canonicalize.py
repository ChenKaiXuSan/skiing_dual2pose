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

Both constructions share the landmark rule (pelvis = hip midpoint, x = left hip
-> right hip, y = pelvis -> neck, z = x cross y with its sign fixed by the eye
direction) but differ in what they do with it.

``mode="per_frame"`` is the training protocol.  Every frame is centred at its
own pelvis and expressed in an orthonormal body frame: the hip line is the
first axis and the spine is projected onto its orthogonal complement
(Gram-Schmidt), so the transform is rigid, bone lengths and joint distances are
preserved, and the representation carries no global motion and does not depend
on batching.  Frames whose landmarks cannot define a frame (coincident hips or
a spine parallel to the hip line, e.g. the zero pose of a failed detection)
reuse the rotation of the previous valid frame of the same sequence (the next
valid one for leading frames, the identity if the whole sequence is
degenerate) and are flagged in ``transform["degenerate"]``.

``mode="first_frame"`` (the default of this library function) is the archived
manuscript construction: one reference frame per batch, axes normalised but
not orthogonalised.  It is kept unchanged for the archived baseline and
native-output analyses and is not a training option.
"""

import torch

CANONICAL_MODES = ("first_frame", "per_frame")
# Hip width or orthogonalised spine length below which a frame cannot define a
# body frame (coordinates are metres or Unity units; real hips are ~0.2 apart).
DEGENERATE_EPS = 1e-4


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


def _face_sign(
    z_axis: torch.Tensor,
    x: torch.Tensor,
    neck_pos: torch.Tensor,
    left_eye: int,
    right_eye: int,
    eps: float,
) -> torch.Tensor:
    """Flip ``z_axis`` where needed so that it points along (eye midpoint - neck)."""
    eye_mid = (x[:, left_eye] + x[:, right_eye]) / 2.0  # (N, 3)
    face_dir = _normalize_torch(eye_mid - neck_pos, eps)  # (N, 3)
    sign = torch.sign((z_axis * face_dir).sum(dim=-1, keepdim=True))  # (N, 1)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return z_axis * sign


def _fill_degenerate_frames(
    R: torch.Tensor, degenerate: torch.Tensor, batch: int, frames: int
) -> torch.Tensor:
    """Replace degenerate rotations by the previous valid one of the same sequence.

    Leading degenerate frames take the first valid rotation; a sequence without
    any valid frame gets the identity.
    """
    R = R.reshape(batch, frames, 3, 3)
    valid = ~degenerate.reshape(batch, frames)
    index = torch.arange(frames, device=R.device).expand(batch, frames)
    last_valid = torch.where(valid, index, torch.full_like(index, -1)).cummax(dim=1).values
    first_valid = (
        torch.where(valid, index, torch.full_like(index, frames)).flip(1).cummin(dim=1).values.flip(1)
    )
    source = torch.where(last_valid >= 0, last_valid, first_valid).clamp(max=frames - 1)
    filled = torch.gather(R, 1, source[..., None, None].expand(batch, frames, 3, 3))
    none_valid = ~valid.any(dim=1)
    if bool(none_valid.any()):
        filled[none_valid] = torch.eye(3, dtype=R.dtype, device=R.device)
    return filled.reshape(batch * frames, 3, 3)


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
    degenerate_eps: float = DEGENERATE_EPS,
):
    """Express poses in a body frame built from the hips, the neck and the eyes.

    x: (J, 3) or (T, J, 3) or (B, T, J, 3)
    mode:
        "first_frame": archived manuscript construction.  One reference, the
            pelvis and (non-orthogonalised) body axes of the first frame of
            the flattened input, shared by every frame.  With a (B, T, J, 3)
            batch that is the first frame of the FIRST sequence, so the result
            depends on batch composition and order.
        "per_frame": the training protocol.  Every frame is centred at its own
            pelvis and rotated into its own orthonormal body frame (hip line
            first, spine orthogonalised against it); independent of batching.
    enforce_face_z_positive:
        force +Z to point along (eye midpoint - neck) when eye indices are given.
    degenerate_eps:
        "per_frame" only: hip width or orthogonalised spine length below which
        a frame is degenerate and reuses a neighbouring rotation.
    Returns:
        x_canon: (B, T, J, 3) for 3-D/4-D input, (J, 3) for 2-D input
        transform: dict with "pelvis" and "R"; for "per_frame" they carry one
            entry per frame, shaped like the input's leading dimensions, plus
            the boolean "degenerate" mask of frames whose rotation was filled in.
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
    hip = right_hip_pos - left_hip_pos  # body right, unnormalised
    spine = neck_pos - pelvis  # body up, unnormalised
    use_face = enforce_face_z_positive and left_eye is not None and right_eye is not None

    if mode == "first_frame":
        # Archived manuscript construction, unchanged: normalised, not orthogonalised.
        x_axis = _normalize_torch(hip, eps)
        y_axis = _normalize_torch(spine, eps)
        z_axis = _normalize_torch(torch.linalg.cross(x_axis, y_axis), eps)
        if use_face:
            z_axis = _face_sign(z_axis, x, neck_pos, left_eye, right_eye, eps)
        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3)
        ref_pelvis = pelvis[0]  # (3,)
        ref_R = R[0]  # (3, 3)
        x_canon = (x - ref_pelvis[None, None, :]) @ ref_R  # (N, J, 3)
        transform = {"pelvis": ref_pelvis, "R": ref_R}
    else:
        # Training protocol: orthonormal frame, hip line first (Gram-Schmidt).
        x_axis = _normalize_torch(hip, eps)
        spine_perp = spine - (spine * x_axis).sum(dim=-1, keepdim=True) * x_axis
        y_axis = _normalize_torch(spine_perp, eps)
        z_axis = _normalize_torch(torch.linalg.cross(x_axis, y_axis), eps)
        if use_face:
            z_axis = _face_sign(z_axis, x, neck_pos, left_eye, right_eye, eps)
        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3), orthogonal columns
        degenerate = (
            (hip.norm(dim=-1) < degenerate_eps)
            | (spine_perp.norm(dim=-1) < degenerate_eps)
            | ~torch.isfinite(R).flatten(1).all(dim=1)
        )
        R = _fill_degenerate_frames(R, degenerate, b, t)
        x_canon = torch.einsum("njc,nck->njk", x_centered, R)  # (N, J, 3)
        if squeeze_time:
            transform = {"pelvis": pelvis[0], "R": R[0], "degenerate": degenerate[0]}
        elif len(original_shape) == 4:
            transform = {
                "pelvis": pelvis.reshape(b, t, 3),
                "R": R.reshape(b, t, 3, 3),
                "degenerate": degenerate.reshape(b, t),
            }
        else:
            transform = {"pelvis": pelvis, "R": R, "degenerate": degenerate}

    if squeeze_time:
        x_canon = x_canon[0]  # (J, 3)
    else:
        x_canon = x_canon.reshape(b, t, j, c)

    return x_canon, transform
