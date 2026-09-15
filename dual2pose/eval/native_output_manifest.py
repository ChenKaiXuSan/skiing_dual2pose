"""Immutable manifests for target-free native-output predictions.

Prediction artifacts are frozen and verified before a scoring target path is
examined.  This keeps model execution/configuration separate from test-label
access and makes every scored array traceable by checksum.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_VERSION = 1


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictionManifest:
    method: str
    dataset: str
    coordinate_frame: str
    prediction_path: str
    sample_count: int
    frame_count: int
    joint_count: int
    unit: str
    source_checkpoint_sha256: str
    config_sha256: str
    code_sha256: str
    sample_index_path: str
    sample_index_sha256: str | None = None
    prediction_sha256: str | None = None
    target_path: str | None = None


@dataclass(frozen=True)
class ScoringManifest:
    prediction_manifest_path: str
    prediction_manifest_sha256: str
    target_path: str
    target_sha256: str
    expected_point_count: int
    expected_acceleration_point_count: int


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _validate_prediction_array(manifest: PredictionManifest) -> None:
    prediction = np.load(manifest.prediction_path, mmap_mode="r", allow_pickle=False)
    expected = (manifest.sample_count, manifest.frame_count, manifest.joint_count, 3)
    if prediction.shape != expected:
        raise ValueError(f"prediction shape {prediction.shape} does not match {expected}")
    if not np.issubdtype(prediction.dtype, np.floating):
        raise ValueError("prediction array must use a floating dtype")
    for start in range(0, manifest.sample_count, 1024):
        if not np.isfinite(prediction[start : start + 1024]).all():
            raise ValueError("prediction array contains nonfinite values")


def _validate_sample_index(manifest: PredictionManifest) -> None:
    with Path(manifest.sample_index_path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != manifest.sample_count:
        raise ValueError("sample index count does not match prediction sample count")
    indices = [record.get("index") for record in records]
    if indices != list(range(manifest.sample_count)):
        raise ValueError("sample index must contain unique contiguous indices in prediction order")


def _validate_manifest_fields(manifest: PredictionManifest) -> None:
    if manifest.target_path is not None:
        raise ValueError("target fields are forbidden in a prediction manifest")
    if manifest.dataset not in {"unity", "ski"}:
        raise ValueError("dataset must be unity or ski")
    if manifest.unit != "m":
        raise ValueError("native-output predictions must declare metres")
    if min(manifest.sample_count, manifest.frame_count, manifest.joint_count) <= 0:
        raise ValueError("sample, frame, and joint counts must be positive")
    for field in ("source_checkpoint_sha256", "config_sha256", "code_sha256"):
        _require_sha256(getattr(manifest, field), field)
    _validate_prediction_array(manifest)
    _validate_sample_index(manifest)


def freeze_prediction_manifest(
    manifest: PredictionManifest, output: str | Path
) -> dict[str, Any]:
    """Validate and exclusively write a target-free frozen manifest."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {output}")
    _validate_manifest_fields(manifest)
    prediction_digest = sha256(manifest.prediction_path)
    sample_index_digest = sha256(manifest.sample_index_path)
    if manifest.prediction_sha256 not in (None, prediction_digest):
        raise ValueError("declared prediction checksum does not match prediction file")
    if manifest.sample_index_sha256 not in (None, sample_index_digest):
        raise ValueError("declared sample index checksum does not match sample index file")
    payload = asdict(manifest)
    payload.pop("target_path")
    payload.update(
        manifest_version=MANIFEST_VERSION,
        frozen=True,
        prediction_sha256=prediction_digest,
        sample_index_sha256=sample_index_digest,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def verify_prediction_manifest(path: str | Path) -> dict[str, Any]:
    """Verify the manifest state and every file hash without reading a target."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("frozen") is not True:
        raise ValueError("prediction manifest is not frozen")
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unsupported prediction manifest version")
    if any(key.startswith("target") for key in payload):
        raise ValueError("frozen prediction manifest contains a target field")
    prediction_path = Path(payload["prediction_path"])
    sample_index_path = Path(payload["sample_index_path"])
    if sha256(prediction_path) != payload.get("prediction_sha256"):
        raise ValueError("prediction checksum mismatch after freeze")
    if sha256(sample_index_path) != payload.get("sample_index_sha256"):
        raise ValueError("sample index checksum mismatch after freeze")
    return payload


def open_scoring_manifest(
    prediction_manifest: str | Path, target_path: str | Path
) -> ScoringManifest:
    """Verify predictions first, then inspect and hash the scoring target."""
    prediction_manifest = Path(prediction_manifest)
    payload = verify_prediction_manifest(prediction_manifest)
    target_path = Path(target_path)
    target = np.load(target_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        int(payload["sample_count"]),
        int(payload["frame_count"]),
        int(payload["joint_count"]),
        3,
    )
    if target.shape != expected_shape:
        raise ValueError(f"target shape {target.shape} does not match {expected_shape}")
    return ScoringManifest(
        prediction_manifest_path=str(prediction_manifest.resolve()),
        prediction_manifest_sha256=sha256(prediction_manifest),
        target_path=str(target_path.resolve()),
        target_sha256=sha256(target_path),
        expected_point_count=int(np.prod(expected_shape[:-1])),
        expected_acceleration_point_count=(
            expected_shape[0] * max(expected_shape[1] - 2, 0) * expected_shape[2]
        ),
    )


__all__ = [
    "PredictionManifest",
    "ScoringManifest",
    "freeze_prediction_manifest",
    "open_scoring_manifest",
    "sha256",
    "verify_prediction_manifest",
]
