"""Metrics for native-output and pre-canonicalization comparisons.

MPJPE removes only per-frame pelvis translation.  PA-MPJPE is computed as a
separate, reflection-free diagnostic and never mutates the arrays used for
MPJPE or acceleration error.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from dual2pose.eval.pa_mpjpe import pa_joint_errors


COMMON13_JOINTS = 13
WINDOW_FRAMES = 30
LEFT_HIP = 4
RIGHT_HIP = 5


def _as_valid_pose(value: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    expected_suffix = (WINDOW_FRAMES, COMMON13_JOINTS, 3)
    if pose.ndim != 4 or pose.shape[1:] != expected_suffix:
        raise ValueError(
            f"{name} must have shape (N,30,13,3): 30 frames and 13 joints required"
        )
    if pose.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sequence")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} must contain only finite values")
    return pose


def root_center(
    pose: np.ndarray, *, left_hip: int = LEFT_HIP, right_hip: int = RIGHT_HIP
) -> np.ndarray:
    """Subtract the mean hip position independently in every frame."""
    value = np.asarray(pose, dtype=np.float64)
    if value.ndim < 3 or value.shape[-1] != 3:
        raise ValueError("pose must end in (J,3)")
    if not (0 <= left_hip < value.shape[-2] and 0 <= right_hip < value.shape[-2]):
        raise ValueError("hip index is outside the pose joint dimension")
    pelvis = 0.5 * (value[..., left_hip, :] + value[..., right_hip, :])
    return value - pelvis[..., None, :]


class NativeMetricAccumulator:
    """Float64, point-weighted accumulator that preserves sequence boundaries."""

    def __init__(self, *, compute_pa: bool = True) -> None:
        self.compute_pa = compute_pa
        self.distance_sum = np.float64(0.0)
        self.point_count = 0
        self.acceleration_distance_sum = np.float64(0.0)
        self.acceleration_point_count = 0
        self.pa_distance_sum = np.float64(0.0)
        self.pa_point_count = 0
        self.pa_frame_count = 0
        self.pa_degenerate_prediction_frames = 0
        self.sample_count = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = _as_valid_pose(prediction, "prediction")
        truth = _as_valid_pose(target, "target")
        if pred.shape != truth.shape:
            raise ValueError(f"prediction and target shapes differ: {pred.shape}/{truth.shape}")

        pred_root = root_center(pred)
        truth_root = root_center(truth)
        distances = np.linalg.norm(pred_root - truth_root, axis=-1)
        self.distance_sum += distances.sum(dtype=np.float64)
        self.point_count += int(distances.size)

        pred_acceleration = pred_root[:, 2:] - 2.0 * pred_root[:, 1:-1] + pred_root[:, :-2]
        truth_acceleration = (
            truth_root[:, 2:] - 2.0 * truth_root[:, 1:-1] + truth_root[:, :-2]
        )
        acceleration_distances = np.linalg.norm(
            pred_acceleration - truth_acceleration, axis=-1
        )
        self.acceleration_distance_sum += acceleration_distances.sum(dtype=np.float64)
        self.acceleration_point_count += int(acceleration_distances.size)

        if self.compute_pa:
            pa_distances, degenerate = pa_joint_errors(pred_root, truth_root)
            self.pa_distance_sum += pa_distances.sum(dtype=np.float64)
            self.pa_point_count += int(pa_distances.size)
            self.pa_frame_count += int(degenerate.size)
            self.pa_degenerate_prediction_frames += int(degenerate.sum())

        self.sample_count += int(pred.shape[0])

    def result(self) -> dict[str, Any]:
        if self.point_count == 0:
            raise ValueError("no native-output points were accumulated")
        expected_points = self.sample_count * WINDOW_FRAMES * COMMON13_JOINTS
        expected_acceleration_points = (
            self.sample_count * (WINDOW_FRAMES - 2) * COMMON13_JOINTS
        )
        if self.point_count != expected_points:
            raise ValueError("MPJPE point denominator does not match sequence coverage")
        if self.acceleration_point_count != expected_acceleration_points:
            raise ValueError("acceleration denominator does not match sequence coverage")
        if self.compute_pa and self.pa_point_count != self.point_count:
            raise ValueError("PA-MPJPE coverage differs from MPJPE coverage")
        return {
            "mpjpe": float(self.distance_sum / self.point_count),
            "pa_mpjpe": (
                float(self.pa_distance_sum / self.pa_point_count)
                if self.compute_pa
                else None
            ),
            "acceleration_error": float(
                self.acceleration_distance_sum / self.acceleration_point_count
            ),
            "sample_count": self.sample_count,
            "point_count": self.point_count,
            "pa_point_count": self.pa_point_count if self.compute_pa else 0,
            "pa_frame_count": self.pa_frame_count if self.compute_pa else 0,
            "pa_degenerate_prediction_frames": (
                self.pa_degenerate_prediction_frames if self.compute_pa else 0
            ),
            "acceleration_point_count": self.acceleration_point_count,
            "distance_sum": float(self.distance_sum),
            "pa_distance_sum": float(self.pa_distance_sum) if self.compute_pa else None,
            "acceleration_distance_sum": float(self.acceleration_distance_sum),
        }


def metric_result(
    prediction: np.ndarray, target: np.ndarray, *, compute_pa: bool = True
) -> dict[str, Any]:
    accumulator = NativeMetricAccumulator(compute_pa=compute_pa)
    accumulator.update(prediction, target)
    return accumulator.result()


__all__ = ["NativeMetricAccumulator", "metric_result", "root_center"]
