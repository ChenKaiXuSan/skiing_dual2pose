"""Prediction-only controls for the native-output comparison."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dual2pose.eval.native_output_protocol import NativeMetricAccumulator
from dual2pose.models.main_baselines import SKELETON_EDGES, _view_quality
from dual2pose.models.sim3 import align_points_sim3


def _validate_native_pose(value: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(value)
    if pose.ndim != 4 or pose.shape[1:] != (30, 13, 3):
        raise ValueError(f"{name} requires 30 frames and 13 joints: (N,30,13,3)")
    if pose.shape[0] == 0 or not np.issubdtype(pose.dtype, np.floating):
        raise ValueError(f"{name} must contain floating pose sequences")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} contains nonfinite values")
    return pose


def _validate_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_pose = _validate_native_pose(left, "left")
    right_pose = _validate_native_pose(right, "right")
    if left_pose.shape != right_pose.shape:
        raise ValueError(f"left/right shapes differ: {left_pose.shape}/{right_pose.shape}")
    dtype = np.result_type(left_pose.dtype, right_pose.dtype, np.float64)
    return left_pose.astype(dtype, copy=False), right_pose.astype(dtype, copy=False)


def unaligned_average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Literal arithmetic average retained only as a diagnostic."""
    left_pose, right_pose = _validate_pair(left, right)
    return 0.5 * (left_pose + right_pose)


def align_right_to_left_sequence(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Estimate one proper Sim(3) per sequence from the two predictions only."""
    left_pose, right_pose = _validate_pair(left, right)
    left_tensor = torch.from_numpy(left_pose)
    right_tensor = torch.from_numpy(right_pose)
    with torch.inference_mode():
        aligned, transform = align_points_sim3(
            right_tensor, left_tensor, allow_reflection=False
        )
    result = aligned.numpy()
    if not np.isfinite(result).all():
        raise ValueError("prediction-only sequence alignment produced nonfinite values")
    record = {
        "scale": transform["scale"].numpy(),
        "rotation": transform["rotation"].numpy(),
        "translation": transform["translation"].numpy(),
        "rmse": transform["rmse"].numpy(),
        "det_rotation": transform["det_r"].numpy(),
    }
    if np.any(record["scale"] < 0.0) or np.any(record["det_rotation"] <= 0.0):
        raise ValueError("alignment must use nonnegative scale and proper rotation")
    return result, record


def aligned_average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_pose, _ = _validate_pair(left, right)
    aligned_right, _ = align_right_to_left_sequence(left, right)
    return 0.5 * (left_pose + aligned_right)


def aligned_quality_weighted(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align right to left, then weight views using pose-only quality costs."""
    left_pose, _ = _validate_pair(left, right)
    aligned_right, _ = align_right_to_left_sequence(left, right)
    left_tensor = torch.from_numpy(left_pose)
    right_tensor = torch.from_numpy(aligned_right)
    edges = SKELETON_EDGES[13]
    with torch.inference_mode():
        costs = torch.stack(
            [_view_quality(left_tensor, edges), _view_quality(right_tensor, edges)],
            dim=1,
        )
        valid = torch.stack(
            [
                (pose.amax(dim=2) - pose.amin(dim=2)).abs().amax(dim=(1, 2))
                > torch.finfo(pose.dtype).eps
                for pose in (left_tensor, right_tensor)
            ],
            dim=1,
        )
        allowed = valid | ~valid.any(dim=1, keepdim=True)
        weights = torch.softmax(costs.masked_fill(~allowed, float("inf")).neg(), dim=1)
        fused = (
            weights[:, 0].view(-1, 1, 1, 1) * left_tensor
            + weights[:, 1].view(-1, 1, 1, 1) * right_tensor
        )
    return fused.numpy(), weights.numpy()


def pooled_view_metrics(
    left_prediction: np.ndarray,
    right_prediction: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
) -> dict[str, Any]:
    """Pool point sums from physical views; never average two poses or means."""
    accumulator = NativeMetricAccumulator()
    accumulator.update(left_prediction, left_target)
    accumulator.update(right_prediction, right_target)
    result = accumulator.result()
    result["view_count"] = 2
    result["aggregation"] = "pooled_joint_frame_errors"
    return result


def smoothnet_native_eligibility(result_or_metadata: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Admit only a checkpoint explicitly trained for the native-coordinate target."""
    if isinstance(result_or_metadata, (str, Path)):
        metadata = json.loads(Path(result_or_metadata).read_text(encoding="utf-8"))
    else:
        metadata = result_or_metadata
    serialized = json.dumps(metadata, sort_keys=True).lower()
    if "canonical" in serialized:
        return {
            "eligible": False,
            "reason": "available SmoothNet artifact was trained on canonical-coordinate averages",
        }
    target_space = metadata.get("target_coordinate_space")
    if target_space != "native_left_camera_m":
        return {
            "eligible": False,
            "reason": "checkpoint has no frozen native-coordinate training contract",
        }
    return {"eligible": True, "reason": None}


__all__ = [
    "align_right_to_left_sequence",
    "aligned_average",
    "aligned_quality_weighted",
    "pooled_view_metrics",
    "smoothnet_native_eligibility",
    "unaligned_average",
]
