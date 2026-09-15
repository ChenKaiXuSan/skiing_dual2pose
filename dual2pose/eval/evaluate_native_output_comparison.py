"""Score frozen pre-Canon predictions with complete-coverage gates.

Prediction manifests are verified before the target descriptor or target
array is opened.  A method without a target-independent common frame remains
in the report with explicit unavailable metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dual2pose.eval.native_output_manifest import sha256, verify_prediction_manifest
from dual2pose.eval.native_output_protocol import metric_result


METHOD_IDS = (
    "sam3d_native_view_pool",
    "motionbert_native_view_pool",
    "poseformer_native_view_pool",
    "videopose3d_native_view_pool",
    "left_native",
    "right_native",
    "unaligned_native_average",
    "native_sequence_aligned_average",
    "native_aligned_quality_weighted",
    "native_aligned_average_smoothnet",
    "stride_native_view_pool",
    "deciwatch_native_view_pool",
    "metapose_native",
    "calibrated_dlt_native",
    "reprojection_gated_dlt_native",
    "dlt_residual_mlp_native",
    "canonfuse3d_internal_canon",
)
EXPECTED_SAMPLES = {"unity": 64440, "ski": 30}


def validate_coverage(
    *, dataset: str, sample_count: int, frame_count: int, joint_count: int
) -> None:
    if dataset not in EXPECTED_SAMPLES:
        raise ValueError(f"unsupported dataset: {dataset}")
    expected = EXPECTED_SAMPLES[dataset]
    if sample_count != expected:
        raise ValueError(f"{dataset} requires exactly {expected} sequences")
    if frame_count != 30:
        raise ValueError("native-output comparison requires exactly 30 frames")
    if joint_count != 13:
        raise ValueError("native-output comparison requires exactly 13 joints")


def unavailable_result(reason: str, *, coordinate_frame: str | None = None) -> dict[str, Any]:
    if not str(reason).strip():
        raise ValueError("an unavailable result requires a reason")
    return {
        "status": "unavailable",
        "coordinate_frame": coordinate_frame,
        "reason": str(reason),
        "mpjpe": None,
        "pa_mpjpe": None,
        "acceleration_error": None,
    }


def build_empty_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "native_output_before_body_normalization",
        "methods": {
            method: unavailable_result("not evaluated") for method in METHOD_IDS
        },
    }


def write_result_exclusive(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def score_frozen_prediction(
    manifest_path: str | Path,
    *,
    target_loader: Callable[[dict[str, Any]], np.ndarray],
    require_full_coverage: bool = True,
) -> dict[str, Any]:
    """Verify prediction integrity and only then invoke the target loader."""
    payload = verify_prediction_manifest(manifest_path)
    if require_full_coverage:
        validate_coverage(
            dataset=str(payload["dataset"]),
            sample_count=int(payload["sample_count"]),
            frame_count=int(payload["frame_count"]),
            joint_count=int(payload["joint_count"]),
        )
    prediction = np.load(payload["prediction_path"], mmap_mode="r", allow_pickle=False)
    target = np.asarray(target_loader(payload))
    if target.shape != prediction.shape:
        raise ValueError(f"target/prediction coverage mismatch: {target.shape}/{prediction.shape}")
    metrics = metric_result(prediction, target)
    return {
        "status": "measured",
        "coordinate_frame": payload["coordinate_frame"],
        "reason": None,
        **metrics,
        "prediction_manifest": str(Path(manifest_path).resolve()),
        "prediction_manifest_sha256": sha256(manifest_path),
    }


def pool_measured_views(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or any(row.get("status") != "measured" for row in rows):
        raise ValueError("only measured native views can be pooled")
    sample_counts = {int(row["sample_count"]) for row in rows}
    if len(sample_counts) != 1:
        raise ValueError("pooled views must have identical sequence coverage")
    point_count = sum(int(row["point_count"]) for row in rows)
    pa_point_count = sum(int(row["pa_point_count"]) for row in rows)
    acceleration_count = sum(int(row["acceleration_point_count"]) for row in rows)
    return {
        "status": "measured",
        "coordinate_frame": "pooled_matched_native_views",
        "reason": None,
        "mpjpe": sum(float(row["distance_sum"]) for row in rows) / point_count,
        "pa_mpjpe": sum(float(row["pa_distance_sum"]) for row in rows) / pa_point_count,
        "acceleration_error": (
            sum(float(row["acceleration_distance_sum"]) for row in rows)
            / acceleration_count
        ),
        "sample_count": sample_counts.pop(),
        "view_count": len(rows),
        "point_count": point_count,
        "pa_point_count": pa_point_count,
        "pa_frame_count": sum(int(row["pa_frame_count"]) for row in rows),
        "pa_degenerate_prediction_frames": sum(
            int(row["pa_degenerate_prediction_frames"]) for row in rows
        ),
        "acceleration_point_count": acceleration_count,
        "distance_sum": sum(float(row["distance_sum"]) for row in rows),
        "pa_distance_sum": sum(float(row["pa_distance_sum"]) for row in rows),
        "acceleration_distance_sum": sum(
            float(row["acceleration_distance_sum"]) for row in rows
        ),
        "constituent_manifests": [row["prediction_manifest"] for row in rows],
    }


def _discover_manifests(root: Path) -> dict[str, Path]:
    manifests: dict[str, Path] = {}
    for path in sorted(root.rglob("manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("frozen") is not True or "method" not in payload:
            continue
        method = str(payload["method"])
        if method in manifests:
            raise ValueError(f"duplicate frozen method manifest: {method}")
        manifests[method] = path
    return manifests


def _descriptor_loader(descriptor_path: Path, expected_dataset: str):
    """Delay descriptor and target access until after prediction verification."""
    cache: dict[str, Any] = {}

    def load(payload: dict[str, Any]) -> np.ndarray:
        if "descriptor" not in cache:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            if descriptor.get("dataset") != expected_dataset:
                raise ValueError("target descriptor dataset mismatch")
            cache["descriptor"] = descriptor
        descriptor = cache["descriptor"]
        if descriptor.get("sample_index_sha256") != payload.get("sample_index_sha256"):
            raise ValueError("target and prediction sample index checksums differ")
        frame = str(payload["coordinate_frame"])
        entry = descriptor.get("targets", {}).get(frame)
        if not entry:
            raise KeyError(f"no supplied target in prediction frame {frame}")
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"target checksum mismatch: {path}")
        target = np.load(path, mmap_mode="r", allow_pickle=False)
        return target

    return load


def evaluate_prediction_root(
    *, dataset: str, prediction_root: Path, target_descriptor: Path
) -> dict[str, Any]:
    manifests = _discover_manifests(prediction_root)
    report = build_empty_report()
    report.update(
        dataset=dataset,
        expected_sample_count=EXPECTED_SAMPLES[dataset],
        prediction_root=str(prediction_root.resolve()),
        target_descriptor=str(target_descriptor.resolve()),
    )
    loader = _descriptor_loader(target_descriptor, dataset)

    def score(method: str) -> dict[str, Any]:
        manifest = manifests.get(method)
        if manifest is None:
            return unavailable_result("no frozen prediction manifest")
        try:
            return score_frozen_prediction(manifest, target_loader=loader)
        except KeyError as error:
            payload = verify_prediction_manifest(manifest)
            return unavailable_result(str(error).strip("'"), coordinate_frame=payload["coordinate_frame"])

    singles = (
        "left_native",
        "right_native",
        "unaligned_native_average",
        "native_sequence_aligned_average",
        "native_aligned_quality_weighted",
        "metapose_native",
        "calibrated_dlt_native",
        "reprojection_gated_dlt_native",
    )
    for method in singles:
        report["methods"][method] = score(method)

    groups = {
        "sam3d_native_view_pool": ("left_native", "right_native"),
        "motionbert_native_view_pool": ("motionbert_native_left", "motionbert_native_right"),
        "poseformer_native_view_pool": ("poseformer_native_left", "poseformer_native_right"),
        "videopose3d_native_view_pool": ("videopose3d_native_left", "videopose3d_native_right"),
        "stride_native_view_pool": ("stride_native_left", "stride_native_right"),
        "deciwatch_native_view_pool": ("deciwatch_native_left", "deciwatch_native_right"),
    }
    for group, members in groups.items():
        rows = [score(member) for member in members]
        missing = [row["reason"] for row in rows if row["status"] != "measured"]
        report["methods"][group] = (
            unavailable_result("; ".join(str(reason) for reason in missing))
            if missing
            else pool_measured_views(rows)
        )

    report["methods"]["native_aligned_average_smoothnet"] = unavailable_result(
        "the archived SmoothNet checkpoint was trained on normalized inputs, not native aligned inputs"
    )
    report["methods"]["dlt_residual_mlp_native"] = unavailable_result(
        "the existing residual checkpoint has no calibrated-world output gauge"
    )
    report["methods"]["canonfuse3d_internal_canon"] = unavailable_result(
        "the current learned output is not guaranteed to equal the invertible left-input gauge"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("unity", "ski"), required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--target-descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_prediction_root(
        dataset=args.dataset,
        prediction_root=args.prediction_root,
        target_descriptor=args.target_descriptor,
    )
    write_result_exclusive(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "METHOD_IDS",
    "build_empty_report",
    "evaluate_prediction_root",
    "pool_measured_views",
    "score_frozen_prediction",
    "unavailable_result",
    "validate_coverage",
    "write_result_exclusive",
]
