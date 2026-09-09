#!/usr/bin/env python3
"""Calibrated 2D geometry references for the IVC main-baseline extension.

The DLT implementation is derived from ``camera_projection_matrix``,
``triangulate_point``, ``project_points`` and
``reprojection_gated_triangulate_sequence`` in the Array submission script
``scripts/unity/evaluate_large_unity_benchmark.py``.  This module extends that
code to batched/per-frame projection matrices and preserves invalid entries.

Only estimated SAM3D 2D detections and supplied camera calibration are read.
No GT 2D, GT 3D, or fitted test transform is accepted by this API.  Unity DLT
is in the Unity metric world frame.  Ski-PosePTZ DLT is in its calibrated
metric world frame; its archived pseudo-GT is a separately constructed
camera-local reference, so the controller must label the comparison as
independently body-canonicalized rather than claim a raw-frame correspondence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from numpy.lib.format import open_memmap

from dual2pose.map_config import (
    FILTERED_15_TO_COMMON_13_INDICES,
    filter_sam3d_body_kpts,
)


EPS = 1e-10
DEFAULT_GATE_PX = 25.0
DEFAULT_UNITY_ROOT = Path("/home/kaixu_chen/skiing/data/skiing_unity_dataset")
DEFAULT_SKI_ROOT = Path("/home/kaixu_chen/skiing/data/Ski-PosePTZ-CameraDataset-png")


@dataclass(frozen=True)
class GeometryResult:
    prediction: np.ndarray
    valid_mask: np.ndarray
    reprojection_error_px: np.ndarray
    method: str
    coordinate_space: str = "calibrated_world_m"
    ungated_prediction: np.ndarray | None = None
    gate_accepted: np.ndarray | None = None
    interpolated_mask: np.ndarray | None = None
    fallback_mask: np.ndarray | None = None


@dataclass(frozen=True)
class ResidualFeatures:
    values: np.ndarray
    valid_mask: np.ndarray
    feature_names: tuple[str, ...]


def _broadcast_projection(projection: np.ndarray, points: np.ndarray) -> np.ndarray:
    projection = np.asarray(projection, dtype=np.float64)
    if projection.shape[-2:] != (3, 4):
        raise ValueError(f"projection must end in (3,4), got {projection.shape}")
    target_prefix = points.shape[:-2]
    return np.broadcast_to(projection, (*target_prefix, 3, 4))


def project_points(projection: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Project ``(...,J,3)`` points with static or matching ``(...,3,4)`` P."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim < 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must end in (J,3), got {xyz.shape}")
    projection = _broadcast_projection(projection, xyz)
    homogeneous = np.concatenate(
        [xyz, np.ones((*xyz.shape[:-1], 1), dtype=np.float64)], axis=-1
    )
    projected = np.einsum("...ik,...jk->...ji", projection, homogeneous)
    uv = np.full((*xyz.shape[:-1], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(projected).all(axis=-1) & (np.abs(projected[..., 2]) > EPS)
    uv[valid] = projected[valid, :2] / projected[valid, 2:3]
    return uv


def triangulate_sequence(
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    keypoints_left: np.ndarray,
    keypoints_right: np.ndarray,
) -> GeometryResult:
    """Batched linear DLT with explicit per-point validity and reprojection."""
    left = np.asarray(keypoints_left, dtype=np.float64)
    right = np.asarray(keypoints_right, dtype=np.float64)
    if left.shape != right.shape or left.ndim < 3 or left.shape[-1] != 2:
        raise ValueError(
            f"2D inputs must have matching (...,T,J,2) shapes, got {left.shape}/{right.shape}"
        )
    p_left = _broadcast_projection(projection_left, left)
    p_right = _broadcast_projection(projection_right, right)
    pl = np.broadcast_to(p_left[..., None, :, :], (*left.shape[:-1], 3, 4))
    pr = np.broadcast_to(p_right[..., None, :, :], (*right.shape[:-1], 3, 4))
    a = np.stack(
        [
            left[..., 0, None] * pl[..., 2, :] - pl[..., 0, :],
            left[..., 1, None] * pl[..., 2, :] - pl[..., 1, :],
            right[..., 0, None] * pr[..., 2, :] - pr[..., 0, :],
            right[..., 1, None] * pr[..., 2, :] - pr[..., 1, :],
        ],
        axis=-2,
    )
    observed = np.isfinite(left).all(axis=-1) & np.isfinite(right).all(axis=-1)
    observed &= np.isfinite(a).all(axis=(-2, -1))
    prediction = np.full((*left.shape[:-1], 3), np.nan, dtype=np.float64)
    flat_a = a.reshape(-1, 4, 4)
    flat_observed = observed.reshape(-1)
    flat_prediction = prediction.reshape(-1, 3)
    if flat_observed.any():
        observed_indices = np.flatnonzero(flat_observed)
        try:
            _, _, vh = np.linalg.svd(flat_a[observed_indices], full_matrices=False)
        except np.linalg.LinAlgError:
            # Fall back point-by-point so one bad SVD cannot erase a chunk.
            vh = np.full((observed_indices.size, 4, 4), np.nan, dtype=np.float64)
            for out_idx, matrix in enumerate(flat_a[observed_indices]):
                try:
                    _, _, vh[out_idx] = np.linalg.svd(matrix, full_matrices=False)
                except np.linalg.LinAlgError:
                    continue
        homogeneous = vh[:, -1, :]
        denom = homogeneous[:, 3]
        solved = np.isfinite(homogeneous).all(axis=-1) & (np.abs(denom) > EPS)
        flat_prediction[observed_indices[solved]] = (
            homogeneous[solved, :3] / denom[solved, None]
        )
    valid = np.isfinite(prediction).all(axis=-1)
    reprojection_left = project_points(p_left, prediction)
    reprojection_right = project_points(p_right, prediction)
    error_left = np.linalg.norm(reprojection_left - left, axis=-1)
    error_right = np.linalg.norm(reprojection_right - right, axis=-1)
    reprojection = np.maximum(error_left, error_right)
    reprojection[~valid] = np.nan
    return GeometryResult(
        prediction=prediction,
        valid_mask=valid,
        reprojection_error_px=reprojection,
        method="dlt",
    )


def _temporal_gate_fill(
    prediction: np.ndarray, gate_accepted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate gate rejects; use ungated DLT for all-rejected joint tracks."""
    if prediction.ndim < 3:
        raise ValueError("prediction must end in (T,J,3)")
    time_count, joint_count = prediction.shape[-3:-1]
    prefix = prediction.shape[:-3]
    sequences = int(np.prod(prefix)) if prefix else 1
    raw = prediction.reshape(sequences, time_count, joint_count, 3)
    accepted = gate_accepted.reshape(sequences, time_count, joint_count)
    filled = np.full_like(raw, np.nan)
    interpolated = np.zeros(accepted.shape, dtype=bool)
    fallback = np.zeros(accepted.shape, dtype=bool)
    timeline = np.arange(time_count, dtype=np.float64)
    for sequence_idx in range(sequences):
        for joint_idx in range(joint_count):
            finite = np.isfinite(raw[sequence_idx, :, joint_idx]).all(axis=-1)
            anchors = accepted[sequence_idx, :, joint_idx] & finite
            if anchors.any():
                anchor_t = timeline[anchors]
                for coordinate in range(3):
                    values = raw[sequence_idx, anchors, joint_idx, coordinate]
                    filled[sequence_idx, finite, joint_idx, coordinate] = np.interp(
                        timeline[finite], anchor_t, values
                    )
                interpolated[sequence_idx, finite & ~anchors, joint_idx] = True
            elif finite.any():
                filled[sequence_idx, finite, joint_idx] = raw[
                    sequence_idx, finite, joint_idx
                ]
                fallback[sequence_idx, finite, joint_idx] = True
    return (
        filled.reshape(prediction.shape),
        interpolated.reshape(gate_accepted.shape),
        fallback.reshape(gate_accepted.shape),
    )


def gate_dlt_result(
    dlt: GeometryResult, threshold_px: float = DEFAULT_GATE_PX
) -> GeometryResult:
    if threshold_px <= 0:
        raise ValueError("threshold_px must be positive")
    accepted = dlt.valid_mask & np.isfinite(dlt.reprojection_error_px)
    accepted &= dlt.reprojection_error_px <= threshold_px
    filled, interpolated, fallback = _temporal_gate_fill(dlt.prediction, accepted)
    valid = np.isfinite(filled).all(axis=-1)
    return GeometryResult(
        prediction=filled,
        valid_mask=valid,
        reprojection_error_px=dlt.reprojection_error_px.copy(),
        method=f"reprojection_gated_dlt_{threshold_px:g}px",
        coordinate_space=dlt.coordinate_space,
        ungated_prediction=dlt.prediction.copy(),
        gate_accepted=accepted,
        interpolated_mask=interpolated,
        fallback_mask=fallback,
    )


def reprojection_gated_dlt(
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    keypoints_left: np.ndarray,
    keypoints_right: np.ndarray,
    threshold_px: float = DEFAULT_GATE_PX,
) -> GeometryResult:
    return gate_dlt_result(
        triangulate_sequence(
            projection_left, projection_right, keypoints_left, keypoints_right
        ),
        threshold_px=threshold_px,
    )


def build_dlt_residual_features(
    dlt: GeometryResult,
    keypoints_left: np.ndarray,
    keypoints_right: np.ndarray,
    *,
    image_size_px: tuple[float, float],
) -> ResidualFeatures:
    """Build geometry-only per-joint MLP inputs; there is no target argument.

    Each joint receives the full DLT pose for its frame, normalized left/right
    estimated 2D, reprojection residual, DLT validity, and joint identity.
    Nonfinite values are zero-filled and accompanied by ``valid_mask``.
    """
    left = np.asarray(keypoints_left, dtype=np.float64)
    right = np.asarray(keypoints_right, dtype=np.float64)
    if left.shape != right.shape or left.shape[:-1] != dlt.prediction.shape[:-1]:
        raise ValueError("2D inputs and DLT prediction must share leading dimensions")
    width, height = (float(image_size_px[0]), float(image_size_px[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image_size_px must be positive")
    joint_count = left.shape[-2]
    full_pose = dlt.prediction.reshape(*dlt.prediction.shape[:-2], joint_count * 3)
    full_pose = np.broadcast_to(
        full_pose[..., None, :], (*left.shape[:-1], joint_count * 3)
    )
    scale = np.array([width, height], dtype=np.float64)
    joint_identity = np.eye(joint_count, dtype=np.float64)
    joint_identity = np.broadcast_to(
        joint_identity, (*left.shape[:-2], joint_count, joint_count)
    )
    values = np.concatenate(
        [
            full_pose,
            left / scale,
            right / scale,
            dlt.reprojection_error_px[..., None] / max(width, height),
            dlt.valid_mask[..., None].astype(np.float64),
            joint_identity,
        ],
        axis=-1,
    )
    valid = dlt.valid_mask & np.isfinite(left).all(axis=-1)
    valid &= np.isfinite(right).all(axis=-1)
    values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    names = tuple(
        [f"dlt_pose_{index}" for index in range(joint_count * 3)]
        + ["left_u", "left_v", "right_u", "right_v", "reprojection", "valid"]
        + [f"joint_{index}" for index in range(joint_count)]
    )
    return ResidualFeatures(values=values, valid_mask=valid, feature_names=names)


def unity_projection_matrix(root: Path, person_id: str, camera_id: str) -> np.ndarray:
    """Array-submission Unity projection matrix, including Unity axis flips."""
    camera = camera_id.removeprefix("capture_")
    camera_dir = root / "data_pole_ski" / person_id / "cameras" / camera
    intrinsic = json.loads((camera_dir / "intrinsics.json").read_text(encoding="utf-8-sig"))
    extrinsic = json.loads((camera_dir / "extrinsics.json").read_text(encoding="utf-8-sig"))
    k = np.array(
        [
            [float(intrinsic["fx"]), 0.0, float(intrinsic["cx"])],
            [0.0, float(intrinsic["fy"]), float(intrinsic["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    world_to_camera = np.asarray(
        extrinsic["t_world_cam_4x4"], dtype=np.float64
    ).reshape(4, 4)
    unity_to_pinhole = np.diag([1.0, -1.0, -1.0, 1.0])[:3, :4]
    return k @ unity_to_pinhole @ world_to_camera


def ski_projection_matrix(
    intrinsic_normalized: np.ndarray,
    camera_position: np.ndarray,
    rotation_camera_to_world: np.ndarray,
    *,
    image_size_px: tuple[float, float] = (256.0, 256.0),
) -> np.ndarray:
    """Build Ski-PosePTZ P from documented per-frame camera fields."""
    width, height = image_size_px
    k = np.asarray(intrinsic_normalized, dtype=np.float64).copy()
    pixel_scale = np.diag([float(width), float(height), 1.0])
    k_px = pixel_scale @ k
    rotation_world_to_camera = np.asarray(
        rotation_camera_to_world, dtype=np.float64
    ).T
    center = np.asarray(camera_position, dtype=np.float64).reshape(3)
    translation = -rotation_world_to_camera @ center
    return k_px @ np.column_stack([rotation_world_to_camera, translation])


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resolve_dataset_path(path: str, dataset_root: Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    parts = candidate.parts
    try:
        dataset_index = parts.index(dataset_root.name)
    except ValueError as exc:
        raise FileNotFoundError(f"Cannot rebase archived dataset path: {path}") from exc
    rebased = dataset_root.joinpath(*parts[dataset_index + 1 :])
    if not rebased.exists():
        raise FileNotFoundError(f"Dataset path missing after rebase: {rebased}")
    return rebased


@lru_cache(maxsize=None)
def _load_unity_estimated_sequence(directory: str) -> dict[int, np.ndarray]:
    root = Path(directory)
    loaded: dict[int, np.ndarray] = {}
    for path in root.glob("kpt2d_*.npy"):
        frame = int(path.stem.rsplit("_", 1)[-1])
        loaded[frame] = filter_sam3d_body_kpts(np.load(path)).astype(np.float64)
    return loaded


@lru_cache(maxsize=None)
def _load_ski_estimated_sequence(directory: str) -> dict[int, np.ndarray]:
    root = Path(directory)
    loaded: dict[int, np.ndarray] = {}
    for path in root.glob("*_sam3d_body.npz"):
        frame = int(path.name.split("_", 1)[0])
        archive = np.load(path, allow_pickle=True)
        output = archive["output"].item()
        pose = filter_sam3d_body_kpts(output["pred_keypoints_2d"])
        loaded[frame] = pose[FILTERED_15_TO_COMMON_13_INDICES].astype(np.float64)
    return loaded


@lru_cache(maxsize=4)
def _load_ski_calibration(labels_h5: str) -> dict[tuple[int, int, int, int], np.ndarray]:
    output: dict[tuple[int, int, int, int], np.ndarray] = {}
    with h5py.File(labels_h5, "r") as handle:
        subjects = np.asarray(handle["subj"], dtype=np.int32)
        sequences = np.asarray(handle["seq"], dtype=np.int32)
        cameras = np.asarray(handle["cam"], dtype=np.int32)
        frames = np.asarray(handle["frame"], dtype=np.int32)
        intrinsics = np.asarray(handle["cam_intrinsic"], dtype=np.float64)
        positions = np.asarray(handle["cam_position"], dtype=np.float64)
        rotations = np.asarray(handle["R_cam_2_world"], dtype=np.float64)
    for idx in range(frames.size):
        key = (
            int(subjects[idx]),
            int(sequences[idx]),
            int(cameras[idx]),
            int(frames[idx]),
        )
        if key in output:
            raise ValueError(f"Duplicate Ski calibration key: {key}")
        output[key] = ski_projection_matrix(
            intrinsics[idx], positions[idx], rotations[idx]
        )
    return output


def _load_index(index_path: Path, split: str) -> list[dict[str, Any]]:
    root = json.loads(index_path.read_text())
    if split not in root:
        raise KeyError(f"split {split!r} missing from {index_path}")
    return list(root[split])


def _sample_key(dataset: str, sample: Mapping[str, Any]) -> tuple[Any, ...]:
    if dataset == "unity":
        return (
            str(sample["person_id"]),
            str(sample["action_id"]),
            str(sample["cam1_id"]),
            str(sample["cam2_id"]),
        )
    return (
        int(sample.get("subj", sample.get("subject_id"))),
        int(sample.get("seq", sample.get("sequence_id"))),
        int(sample.get("cam1", sample.get("cam1_id"))),
        int(sample.get("cam2", sample.get("cam2_id"))),
    )


def _match_source_samples(
    dataset: str,
    cached_samples: Sequence[Mapping[str, Any]],
    source_samples: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    source_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for sample in source_samples:
        key = _sample_key(dataset, sample)
        if key in source_by_key:
            raise ValueError(f"Duplicate source sample identity: {key}")
        source_by_key[key] = sample
    matched = []
    for cached in cached_samples:
        key = _sample_key(dataset, cached)
        if key not in source_by_key:
            raise KeyError(f"Cached sample is absent from source index: {key}")
        matched.append(source_by_key[key])
    return matched


def _load_sample_inputs(
    dataset: str,
    source: Mapping[str, Any],
    frame_ids: np.ndarray,
    dataset_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = [int(value) for value in frame_ids]
    if dataset == "unity":
        left_dir = _resolve_dataset_path(str(source["sam3d_cam1_kpt2d_dir"]), dataset_root)
        right_dir = _resolve_dataset_path(str(source["sam3d_cam2_kpt2d_dir"]), dataset_root)
        left_map = _load_unity_estimated_sequence(str(left_dir))
        right_map = _load_unity_estimated_sequence(str(right_dir))
        try:
            left = np.stack([left_map[frame] for frame in frames])
            right = np.stack([right_map[frame] for frame in frames])
        except KeyError as exc:
            raise FileNotFoundError(
                f"Cached Unity frame lacks estimated 2D in {left_dir}/{right_dir}: {exc}"
            ) from exc
        person = str(source["person_id"])
        p_left_static = unity_projection_matrix(
            dataset_root, person, str(source["cam1_id"])
        )
        p_right_static = unity_projection_matrix(
            dataset_root, person, str(source["cam2_id"])
        )
        p_left = np.broadcast_to(p_left_static, (len(frames), 3, 4)).copy()
        p_right = np.broadcast_to(p_right_static, (len(frames), 3, 4)).copy()
        return left, right, p_left, p_right

    left_dir = _resolve_dataset_path(str(source["cam1_sam3d_kpt2d_dir"]), dataset_root)
    right_dir = _resolve_dataset_path(str(source["cam2_sam3d_kpt2d_dir"]), dataset_root)
    left_map = _load_ski_estimated_sequence(str(left_dir))
    right_map = _load_ski_estimated_sequence(str(right_dir))
    try:
        left = np.stack([left_map[frame] for frame in frames])
        right = np.stack([right_map[frame] for frame in frames])
    except KeyError as exc:
        raise FileNotFoundError(
            f"Cached Ski frame lacks estimated 2D in {left_dir}/{right_dir}: {exc}"
        ) from exc
    labels_path = _resolve_dataset_path(str(source["labels_h5"]), dataset_root)
    calibration = _load_ski_calibration(str(labels_path))
    subject = int(source["subject_id"])
    sequence = int(source["sequence_id"])
    cam_left = int(source["cam1_id"])
    cam_right = int(source["cam2_id"])
    try:
        p_left = np.stack(
            [calibration[(subject, sequence, cam_left, frame)] for frame in frames]
        )
        p_right = np.stack(
            [calibration[(subject, sequence, cam_right, frame)] for frame in frames]
        )
    except KeyError as exc:
        raise FileNotFoundError(f"Cached Ski frame lacks supplied calibration: {exc}") from exc
    return left, right, p_left, p_right


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_existing_export(
    output_dir: Path,
    final_names: Mapping[str, str],
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Existing geometry outputs lack manifest: {output_dir}")
    manifest = json.loads(manifest_path.read_text())
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Existing geometry manifest mismatch for {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    recorded_hashes = manifest.get("output_sha256", {})
    for filename in final_names.values():
        path = output_dir / filename
        if not path.exists():
            raise ValueError(f"Existing geometry output is incomplete: {path}")
        actual = _sha256(path)
        if recorded_hashes.get(filename) != actual:
            raise ValueError(f"Existing geometry output hash mismatch: {path}")
    return manifest


def export_split(
    *,
    dataset: str,
    split: str,
    cache_dir: Path,
    output_dir: Path,
    index_path: Path,
    dataset_root: Path,
    threshold_px: float = DEFAULT_GATE_PX,
    chunk_size: int = 64,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export geometry arrays in exact cache sample/frame order."""
    if dataset not in {"unity", "ski"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    frames_path = cache_dir / "frames.npy"
    samples_path = cache_dir / "samples.jsonl"
    target_path = cache_dir / "target.npy"
    cached_samples = _jsonl(samples_path)
    frames = np.load(frames_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")
    expected_shape = tuple(int(value) for value in target.shape)
    if frames.shape != expected_shape[:2]:
        raise ValueError(f"frames/target mismatch: {frames.shape} vs {target.shape}")
    if len(cached_samples) != expected_shape[0]:
        raise ValueError("samples.jsonl count does not match target.npy")
    source_samples = _load_index(index_path, split)
    matched_sources = _match_source_samples(dataset, cached_samples, source_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_names = {
        "dlt": "dlt.npy",
        "robust": "robust_dlt.npy",
        "dlt_valid": "dlt_valid.npy",
        "robust_valid": "robust_valid.npy",
        "gate": "gate_accepted.npy",
        "reprojection": "reprojection_error_px.npy",
        "interpolated": "robust_interpolated.npy",
        "fallback": "robust_fallback.npy",
    }
    partials = {key: output_dir / f".{name}.partial" for key, name in final_names.items()}
    expected_manifest = {
        "dataset": dataset,
        "split": split,
        "shape": list(expected_shape),
        "gate_threshold_px": float(threshold_px),
        "source_index_sha256": _sha256(index_path),
        "cache_frames_sha256": _sha256(frames_path),
        "cache_samples_sha256": _sha256(samples_path),
        "exporter_source_sha256": _sha256(Path(__file__)),
    }
    final_exists = [(output_dir / value).exists() for value in final_names.values()]
    partial_exists = [path.exists() for path in partials.values()]
    if not overwrite and any(partial_exists):
        raise ValueError(
            f"Stale partial geometry outputs exist in {output_dir}; "
            "inspect them or rerun explicitly with --overwrite"
        )
    if not overwrite and all(final_exists):
        return _validate_existing_export(output_dir, final_names, expected_manifest)
    if not overwrite and any(final_exists):
        raise ValueError(
            f"Incomplete geometry outputs exist in {output_dir}; "
            "inspect them or rerun explicitly with --overwrite"
        )
    arrays = {
        "dlt": open_memmap(partials["dlt"], mode="w+", dtype=np.float32, shape=expected_shape),
        "robust": open_memmap(partials["robust"], mode="w+", dtype=np.float32, shape=expected_shape),
        "dlt_valid": open_memmap(partials["dlt_valid"], mode="w+", dtype=np.bool_, shape=expected_shape[:-1]),
        "robust_valid": open_memmap(partials["robust_valid"], mode="w+", dtype=np.bool_, shape=expected_shape[:-1]),
        "gate": open_memmap(partials["gate"], mode="w+", dtype=np.bool_, shape=expected_shape[:-1]),
        "reprojection": open_memmap(partials["reprojection"], mode="w+", dtype=np.float32, shape=expected_shape[:-1]),
        "interpolated": open_memmap(partials["interpolated"], mode="w+", dtype=np.bool_, shape=expected_shape[:-1]),
        "fallback": open_memmap(partials["fallback"], mode="w+", dtype=np.bool_, shape=expected_shape[:-1]),
    }
    counts = {
        "points": int(np.prod(expected_shape[:-1])),
        "dlt_valid": 0,
        "robust_valid": 0,
        "gate_accepted": 0,
        "interpolated": 0,
        "all_rejected_track_fallback_points": 0,
    }
    for start in range(0, expected_shape[0], chunk_size):
        stop = min(start + chunk_size, expected_shape[0])
        loaded = [
            _load_sample_inputs(
                dataset,
                matched_sources[index],
                np.asarray(frames[index]),
                dataset_root,
            )
            for index in range(start, stop)
        ]
        left = np.stack([item[0] for item in loaded])
        right = np.stack([item[1] for item in loaded])
        p_left = np.stack([item[2] for item in loaded])
        p_right = np.stack([item[3] for item in loaded])
        expected_chunk = (stop - start, *expected_shape[1:-1])
        if left.shape[:-1] != expected_chunk:
            raise ValueError(
                f"Estimated 2D shape {left.shape} does not match cache {expected_chunk}"
            )
        dlt = triangulate_sequence(p_left, p_right, left, right)
        robust = gate_dlt_result(dlt, threshold_px=threshold_px)
        arrays["dlt"][start:stop] = dlt.prediction.astype(np.float32)
        arrays["robust"][start:stop] = robust.prediction.astype(np.float32)
        arrays["dlt_valid"][start:stop] = dlt.valid_mask
        arrays["robust_valid"][start:stop] = robust.valid_mask
        arrays["gate"][start:stop] = robust.gate_accepted
        arrays["reprojection"][start:stop] = dlt.reprojection_error_px.astype(np.float32)
        arrays["interpolated"][start:stop] = robust.interpolated_mask
        arrays["fallback"][start:stop] = robust.fallback_mask
        counts["dlt_valid"] += int(dlt.valid_mask.sum())
        counts["robust_valid"] += int(robust.valid_mask.sum())
        counts["gate_accepted"] += int(robust.gate_accepted.sum())
        counts["interpolated"] += int(robust.interpolated_mask.sum())
        counts["all_rejected_track_fallback_points"] += int(robust.fallback_mask.sum())
        if start == 0 or stop == expected_shape[0] or stop % 1024 == 0:
            progress = {"completed_samples": stop, "total_samples": expected_shape[0]}
            (output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    for array in arrays.values():
        array.flush()
    del arrays
    for key, name in final_names.items():
        os.replace(partials[key], output_dir / name)
    output_sha256 = {
        filename: _sha256(output_dir / filename) for filename in final_names.values()
    }
    points = counts["points"]
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "shape": list(expected_shape),
        "samples": expected_shape[0],
        "frames_per_sample": expected_shape[1],
        "joints": expected_shape[2],
        "coordinate_space": "calibrated_world_m",
        "estimated_2d_only": True,
        "gt_2d_used": False,
        "gt_fitted_transform_used": False,
        "gate_threshold_px": float(threshold_px),
        "robust_policy": "interpolate accepted finite DLT over time; nearest endpoint; all-rejected finite track falls back to ungated DLT; nonfinite DLT remains invalid",
        "counts": counts,
        "coverage": {
            key: float(counts[key]) / points
            for key in ("dlt_valid", "robust_valid", "gate_accepted", "interpolated")
        },
        "source_index": str(index_path.resolve()),
        "source_index_sha256": expected_manifest["source_index_sha256"],
        "cache_frames_sha256": expected_manifest["cache_frames_sha256"],
        "cache_samples_sha256": expected_manifest["cache_samples_sha256"],
        "exporter_source_sha256": expected_manifest["exporter_source_sha256"],
        "output_sha256": output_sha256,
        "array_geometry_provenance": "Array submission evaluate_large_unity_benchmark.py camera_projection_matrix/triangulate_point/project_points/reprojection_gated_triangulate_sequence; locally batched with explicit coverage and temporal fill",
        "ski_limitation": (
            "Ski pseudo-GT is a GT-3D-fitted camera-local constructed reference; raw calibrated-world DLT has no GT-free exact pre-canonical map. Controller comparison is independently body-canonicalized."
            if dataset == "ski"
            else None
        ),
        "files": final_names,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _default_index(dataset: str, root: Path) -> Path:
    if dataset == "unity":
        return root / "index_mapping/use_layer_camera_filter_disabled/camera_pairs_by_action_folds/fold_00.json"
    return root / "index_mapping/ski_pose_ptz_index_mapping.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("unity", "ski"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--gate-threshold-px", type=float, default=DEFAULT_GATE_PX)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root or (
        DEFAULT_UNITY_ROOT if args.dataset == "unity" else DEFAULT_SKI_ROOT
    )
    index_path = args.index_path or _default_index(args.dataset, root)
    for split in args.splits:
        cache_dir = args.cache_root / args.dataset / split
        if not cache_dir.exists():
            print(f"[{args.dataset}:{split}] cache missing, skipped: {cache_dir}", flush=True)
            continue
        manifest = export_split(
            dataset=args.dataset,
            split=split,
            cache_dir=cache_dir,
            output_dir=args.output / args.dataset / split,
            index_path=index_path,
            dataset_root=root,
            threshold_px=args.gate_threshold_px,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )
        print(
            f"[{args.dataset}:{split}] samples={manifest['samples']} coverage={manifest['coverage']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
