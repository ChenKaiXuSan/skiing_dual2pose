"""Explicit coordinate-frame conversion for native-output evaluation.

This module accepts only independently supplied camera/observation transforms.
It intentionally contains no prediction-to-target fitting API.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np


ALLOWED_TRANSFORM_KINDS = {
    "camera_extrinsic",
    "documented_observation_inverse",
    "unit_conversion",
}


@dataclass(frozen=True)
class CoordinateTransform:
    source_frame: str
    destination_frame: str
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    source_sha256: str
    transform_kind: str


@dataclass(frozen=True)
class FrameDecision:
    status: Literal["comparable", "unavailable"]
    reporting_frame: str
    reason: str | None
    transform_source_sha256: str | None = None


def _validate_rotation(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in (3,3)")
    identity = np.eye(3, dtype=np.float64)
    gram = np.swapaxes(value, -1, -2) @ value
    determinant = np.linalg.det(value)
    if not np.allclose(gram, identity, atol=1e-7, rtol=1e-7) or not np.allclose(
        determinant, 1.0, atol=1e-7, rtol=1e-7
    ):
        raise ValueError("coordinate conversion requires a proper rotation")
    return value


def _validate_translation(translation: np.ndarray) -> np.ndarray:
    value = np.asarray(translation, dtype=np.float64)
    if value.shape[-1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("translation must be finite and end in (3,)")
    return value


def _validate_scale(scale: float) -> float:
    value = float(scale)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("coordinate conversion requires a positive scale")
    return value


def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply a documented row-vector similarity transform."""
    value = np.asarray(points, dtype=np.float64)
    if value.shape[-1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("points must be finite and end in xyz")
    proper_rotation = _validate_rotation(rotation)
    offset = _validate_translation(translation)
    positive_scale = _validate_scale(scale)
    transformed = positive_scale * (value @ proper_rotation)
    while offset.ndim < transformed.ndim:
        offset = np.expand_dims(offset, axis=-2)
    try:
        return transformed + offset
    except ValueError as error:
        raise ValueError("translation is not broadcastable to points") from error


def transform_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one reporting-frame transform to both already comparable arrays."""
    return (
        transform_points(prediction, rotation, translation, scale),
        transform_points(target, rotation, translation, scale),
    )


def _validate_transform_provenance(transform: CoordinateTransform) -> None:
    if transform.transform_kind not in ALLOWED_TRANSFORM_KINDS:
        raise ValueError(
            "only independently supplied camera or observation transforms are allowed"
        )
    if len(transform.source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in transform.source_sha256
    ):
        raise ValueError("transform source_sha256 must be a lowercase SHA-256 digest")
    _validate_rotation(transform.rotation)
    _validate_translation(transform.translation)
    _validate_scale(transform.scale)


def decide_common_frame(
    *,
    method_frame: str,
    target_frame: str,
    supplied_transform: CoordinateTransform | None,
) -> FrameDecision:
    """Decide comparability without inspecting prediction or target values."""
    if method_frame == target_frame:
        return FrameDecision("comparable", method_frame, None, None)
    if supplied_transform is None:
        return FrameDecision(
            "unavailable",
            method_frame,
            f"no supplied frame mapping from {target_frame} to {method_frame}",
            None,
        )
    _validate_transform_provenance(supplied_transform)
    if (
        supplied_transform.source_frame != target_frame
        or supplied_transform.destination_frame != method_frame
    ):
        raise ValueError(
            "supplied transform must map the target frame into the prediction frame"
        )
    return FrameDecision(
        "comparable",
        method_frame,
        None,
        supplied_transform.source_sha256,
    )


def apply_target_to_prediction_frame(
    prediction: np.ndarray,
    target: np.ndarray,
    transform: CoordinateTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep prediction fixed and map the reference via supplied metadata."""
    _validate_transform_provenance(transform)
    prediction_value = np.asarray(prediction, dtype=np.float64)
    target_value = np.asarray(target, dtype=np.float64)
    if prediction_value.shape != target_value.shape:
        raise ValueError("prediction and target shapes must match before scoring")
    if not np.isfinite(prediction_value).all():
        raise ValueError("prediction must contain only finite values")
    return prediction_value.copy(), transform_points(
        target_value,
        transform.rotation,
        transform.translation,
        transform.scale,
    )


def conservative_frame_audit() -> dict[str, object]:
    """Emit protocol decisions that do not assume unavailable dataset metadata."""
    rows = [
        {
            "dataset": "unity",
            "method_group": "native_per_view",
            "status": "requires_supplied_transform",
            "required_mapping": "unity_world_m_to_matching_camera_m",
        },
        {
            "dataset": "unity",
            "method_group": "calibrated_dlt",
            "status": "requires_target_frame_verification",
            "method_frame": "unity_world_m",
        },
        {
            "dataset": "ski",
            "method_group": "native_per_view",
            "status": "requires_supplied_transform",
            "required_mapping": "constructed_reference_to_matching_camera_m",
        },
        {
            "dataset": "ski",
            "method_group": "calibrated_dlt",
            "status": "unavailable",
            "reason": (
                "no independently supplied mapping from calibrated_world_m to "
                "constructed_camera_local_m"
            ),
        },
    ]
    return {
        "protocol": "native_output_frame_audit_v1",
        "target_fitting_allowed": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-datasets", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.audit_datasets:
        parser.error("--audit-datasets is required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frame audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = conservative_frame_audit()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()


__all__ = [
    "CoordinateTransform",
    "FrameDecision",
    "apply_target_to_prediction_frame",
    "conservative_frame_audit",
    "decide_common_frame",
    "transform_pair",
    "transform_points",
]
