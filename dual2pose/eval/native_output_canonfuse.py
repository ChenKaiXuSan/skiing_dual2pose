"""Isolated CanonFuse3D adapter for native-output protocol admission.

This is the only native-output module allowed to use the method's internal
body-coordinate transform.  Its inverse is input-derived and is never fitted
to a scoring target.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dual2pose.models.crossview_fusion import CrossViewCanonicalFusion
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


@dataclass(frozen=True)
class LeftInputTransform:
    pelvis: np.ndarray
    linear: np.ndarray


def _validate_pose(pose: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float32)
    if value.ndim != 4 or value.shape[1] != 30 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (N,30,J,3)")
    if value.shape[0] == 0 or value.shape[2] not in {13, 15}:
        raise ValueError(f"{name} must contain common13 or CanonFuse15 sequences")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def canonicalize_left_with_transform(
    left: np.ndarray,
    *,
    left_hip: int,
    right_hip: int,
    neck: int,
) -> tuple[np.ndarray, LeftInputTransform]:
    """Transform each sequence separately and retain one inverse per sample."""
    value = _validate_pose(left, "left")
    transformed: list[np.ndarray] = []
    pelvis: list[np.ndarray] = []
    linear: list[np.ndarray] = []
    for sequence in value:
        output, record = canonicalize_pose_torch(
            torch.from_numpy(sequence[None]),
            left_hip=left_hip,
            right_hip=right_hip,
            neck=neck,
            mode="first_frame",
        )
        transformed.append(output[0].cpu().numpy())
        pelvis.append(record["pelvis"].cpu().numpy())
        linear.append(record["R"].cpu().numpy())
    result = np.stack(transformed)
    transform = LeftInputTransform(pelvis=np.stack(pelvis), linear=np.stack(linear))
    recovered = inverse_left_transform(result, transform)
    maximum = float(np.max(np.abs(recovered - value)))
    if maximum > 2e-5:
        raise ValueError(f"input-only coordinate round trip failed: {maximum:.9g} m")
    return result, transform


def inverse_left_transform(
    transformed: np.ndarray, transform: LeftInputTransform
) -> np.ndarray:
    """Invert `(native - pelvis) @ linear` with a true matrix inverse."""
    value = np.asarray(transformed, dtype=np.float64)
    pelvis = np.asarray(transform.pelvis, dtype=np.float64)
    linear = np.asarray(transform.linear, dtype=np.float64)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError("transformed pose must have shape (N,T,J,3)")
    if pelvis.shape != (value.shape[0], 3) or linear.shape != (value.shape[0], 3, 3):
        raise ValueError("one pelvis and linear transform are required per sequence")
    determinant = np.linalg.det(linear)
    if not np.isfinite(linear).all() or np.any(np.abs(determinant) <= 1e-10):
        raise ValueError("left input transform is singular or nonfinite")
    inverse = np.linalg.inv(linear)
    recovered = value @ inverse[:, None, :, :]
    return recovered + pelvis[:, None, None, :]


def prepare_canonfuse_inputs(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_hip: int,
    right_hip: int,
    neck: int,
) -> tuple[tuple[np.ndarray, np.ndarray], LeftInputTransform]:
    """Prepare both inputs independently; expose only the left input inverse."""
    left_value = _validate_pose(left, "left")
    right_value = _validate_pose(right, "right")
    if left_value.shape != right_value.shape:
        raise ValueError("left and right inputs must have matching shapes")
    left_transformed, left_transform = canonicalize_left_with_transform(
        left_value, left_hip=left_hip, right_hip=right_hip, neck=neck
    )
    right_transformed, _ = canonicalize_left_with_transform(
        right_value, left_hip=left_hip, right_hip=right_hip, neck=neck
    )
    return (left_transformed, right_transformed), left_transform


def canonfuse_native_admission() -> dict[str, Any]:
    """Declare the current model's output-gauge limitation without test data."""
    return {
        "round_trip_required": True,
        "output_gauge_declared": "supervised_target_canonical",
        "gauge_supported_by_model_contract": False,
        "native_mpjpe_admitted": False,
        "native_pa_mpjpe_admitted": False,
        "native_acceleration_admitted": False,
        "reason": (
            "the current network is supervised in independently constructed target "
            "canonical coordinates and mixes left- and right-gauge bases; its output "
            "is not guaranteed to equal the left-input canonical gauge"
        ),
    }


def _load_model(checkpoint: str | Path, device: torch.device) -> CrossViewCanonicalFusion:
    model = CrossViewCanonicalFusion(num_heads=4).to(device)
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    state = {
        key[len("models.") :]: value
        for key, value in payload["state_dict"].items()
        if key.startswith("models.")
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_native_canonfuse(
    left: np.ndarray,
    right: np.ndarray,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 32,
    allow_unadmitted_diagnostic: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run a diagnostic left-inverse export; never label it comparable by default."""
    admission = canonfuse_native_admission()
    if not allow_unadmitted_diagnostic:
        raise RuntimeError(admission["reason"])
    left_value = _validate_pose(left, "left")
    right_value = _validate_pose(right, "right")
    if left_value.shape != right_value.shape or batch_size <= 0:
        raise ValueError("matching inputs and a positive batch size are required")
    if left_value.shape[2] == 15:
        indices = {"left_hip": 6, "right_hip": 7, "neck": 14}
    else:
        indices = {"left_hip": 4, "right_hip": 5, "neck": 12}
    torch_device = torch.device(device)
    model = _load_model(checkpoint, torch_device)
    recovered_batches: list[np.ndarray] = []
    maximum_round_trip = 0.0
    with torch.inference_mode():
        for start in range(0, len(left_value), batch_size):
            stop = min(start + batch_size, len(left_value))
            (left_input, right_input), transform = prepare_canonfuse_inputs(
                left_value[start:stop], right_value[start:stop], **indices
            )
            output, _ = model(
                torch.from_numpy(left_input).to(torch_device),
                torch.from_numpy(right_input).to(torch_device),
            )
            recovered = inverse_left_transform(output.cpu().numpy(), transform)
            recovered_batches.append(recovered.astype(np.float32))
            left_round_trip = inverse_left_transform(left_input, transform)
            maximum_round_trip = max(
                maximum_round_trip,
                float(np.max(np.abs(left_round_trip - left_value[start:stop]))),
            )
    admission = {**admission, "round_trip_max_abs_m": maximum_round_trip}
    return np.concatenate(recovered_batches), admission


__all__ = [
    "LeftInputTransform",
    "canonicalize_left_with_transform",
    "canonfuse_native_admission",
    "inverse_left_transform",
    "predict_native_canonfuse",
    "prepare_canonfuse_inputs",
]
