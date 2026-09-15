"""Target-free native front-end prediction export controller.

The full exporter is assembled incrementally by the native-output experiment
plan.  Its public request validator already forbids partial test publication
runs and its module has no dependency on body-coordinate preprocessing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dual2pose.eval.native_output_manifest import PredictionManifest, freeze_prediction_manifest, sha256
from dual2pose.eval.native_output_methods import align_right_to_left_sequence, aligned_average, aligned_quality_weighted, smoothnet_native_eligibility, unaligned_average


def validate_export_request(*, split: str, limit: int | None) -> None:
    if split not in {"train", "val", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if split == "test" and limit is not None:
        raise ValueError("limit is forbidden for test exports")


def _verify(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"input checksum mismatch: {path}")


def _combined_digest(*values: str) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _load_target_free_views(cache: Path, limit: int | None):
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("representation") != "raw_pre_batch_canonicalization":
        raise ValueError("native export requires a raw pre-processing cache")
    validate_export_request(split=str(manifest["split"]), limit=limit)
    dataset = str(manifest["dataset"])
    joints = 15 if dataset == "unity" else 13
    expected = (int(manifest["sample_count"]), 30, joints, 3)
    if int(manifest["time_window"]) != 30 or int(manifest["num_joints"]) != joints:
        raise ValueError("cache must contain the fixed 30-frame native skeleton")
    for name in ("left.npy", "right.npy", "samples.jsonl"):
        _verify(cache / name, manifest["files"][name])
    left_source = np.load(cache / "left.npy", mmap_mode="r", allow_pickle=False)
    right_source = np.load(cache / "right.npy", mmap_mode="r", allow_pickle=False)
    if left_source.shape != expected or right_source.shape != expected:
        raise ValueError("cached native-view shape does not match its manifest")
    count = expected[0] if limit is None else min(int(limit), expected[0])
    left = np.array(left_source[:count], dtype=np.float32, copy=True)
    right = np.array(right_source[:count], dtype=np.float32, copy=True)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("native views contain nonfinite values")
    indices = slice(2, 15) if dataset == "unity" else slice(None)
    left = np.ascontiguousarray(left[:, :, indices])
    right = np.ascontiguousarray(right[:, :, indices])
    with (cache / "samples.jsonl").open(encoding="utf-8") as handle:
        samples = [json.loads(next(handle)) for _ in range(count)]
    if [sample.get("index") for sample in samples] != list(range(count)):
        raise ValueError("sample index/order does not match cached prediction order")
    return left, right, samples, manifest


def _write_prediction(*, output_root: Path, dataset: str, method: str, prediction: np.ndarray, coordinate_frame: str, samples_path: Path, source_digest: str, config_digest: str, code_digest: str):
    method_root = output_root / method
    method_root.mkdir()
    prediction_path = method_root / "predictions.npy"
    np.save(prediction_path, np.asarray(prediction, dtype=np.float32))
    manifest_path = method_root / "manifest.json"
    payload = freeze_prediction_manifest(PredictionManifest(
        method=method, dataset=dataset, coordinate_frame=coordinate_frame,
        prediction_path=str(prediction_path.resolve()), sample_count=int(prediction.shape[0]),
        frame_count=30, joint_count=13, unit="m",
        source_checkpoint_sha256=source_digest, config_sha256=config_digest,
        code_sha256=code_digest, sample_index_path=str(samples_path.resolve()),
    ), manifest_path)
    return {"status": "frozen", "manifest_path": str(manifest_path.resolve()),
            "prediction_sha256": payload["prediction_sha256"],
            "coordinate_frame": coordinate_frame}


def export_cached_sam3d_and_controls(cache: str | Path, output_root: str | Path, *, limit: int | None = None, smoothnet_result: str | Path | None = None):
    """Freeze raw SAM3D controls without touching the cache target file."""
    cache = Path(cache).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite native export root: {output_root}")
    left, right, samples, source = _load_target_free_views(cache, limit)
    output_root.mkdir(parents=True)
    source_record = {"dataset": source["dataset"], "split": source["split"],
        "cache_manifest_path": str((cache / "manifest.json").resolve()),
        "cache_manifest_sha256": sha256(cache / "manifest.json"),
        "target_loaded": False, "source_artifact_kind": "frozen_sam3d_native_cache"}
    (output_root / "source.json").write_text(json.dumps(source_record, indent=2) + "\n", encoding="utf-8")
    samples_path = output_root / "samples.jsonl"
    samples_path.write_text("".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples), encoding="utf-8")
    aligned = aligned_average(left, right)
    weighted, weights = aligned_quality_weighted(left, right)
    predictions = {
        "left_native": (left, "left_camera_m"),
        "right_native": (right, "right_camera_m"),
        "unaligned_native_average": (unaligned_average(left, right), "mixed_camera_diagnostic_m"),
        "native_sequence_aligned_average": (aligned, "left_camera_m"),
        "native_aligned_quality_weighted": (weighted, "left_camera_m"),
    }
    left_digest = source["files"]["left.npy"]
    right_digest = source["files"]["right.npy"]
    combined_source = _combined_digest(left_digest, right_digest)
    config_digest = source.get("config_sha256") or sha256(cache / "manifest.json")
    code_digest = sha256(Path(__file__))
    methods = {}
    for method, (prediction, frame) in predictions.items():
        source_digest = left_digest if method == "left_native" else right_digest if method == "right_native" else combined_source
        methods[method] = _write_prediction(output_root=output_root, dataset=str(source["dataset"]),
            method=method, prediction=prediction, coordinate_frame=frame,
            samples_path=samples_path, source_digest=source_digest,
            config_digest=config_digest, code_digest=code_digest)
    transforms_path = output_root / "prediction_only_alignment.npz"
    _, transform = align_right_to_left_sequence(left, right)
    np.savez_compressed(transforms_path, weights=weights, **transform)
    smoothnet = smoothnet_native_eligibility(smoothnet_result if smoothnet_result is not None else {"provenance": {"canonicalization": "existing archived training protocol"}})
    methods["native_aligned_average_smoothnet"] = {"status": "unavailable", "reason": smoothnet["reason"]}
    report = {**source_record, "sample_count": len(samples), "joint_subset": "common13",
        "methods": methods, "view_groups": {"sam3d_native_view_pool": ["left_native", "right_native"]},
        "alignment_artifact": str(transforms_path.resolve()),
        "alignment_artifact_sha256": sha256(transforms_path)}
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def export_unity_lifter_windows(cache: str | Path, archive_manifest: str | Path, output_root: str | Path, *, limit: int | None = None):
    """Assemble archived Unity lifter streams into ordered camera-pair windows."""
    cache = Path(cache).resolve()
    archive_manifest = Path(archive_manifest).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite native export root: {output_root}")
    source = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if source.get("dataset") != "unity" or source.get("representation") != "raw_pre_batch_canonicalization":
        raise ValueError("lifter assembly requires the raw Unity cache")
    validate_export_request(split=str(source["split"]), limit=limit)
    for name in ("frames.npy", "samples.jsonl"):
        _verify(cache / name, source["files"][name])
    total = int(source["sample_count"])
    count = total if limit is None else min(int(limit), total)
    frames_source = np.load(cache / "frames.npy", mmap_mode="r", allow_pickle=False)
    if frames_source.shape != (total, 30):
        raise ValueError("cached frame index must have shape (N,30)")
    frames = np.array(frames_source[:count], dtype=np.int64, copy=True)
    with (cache / "samples.jsonl").open(encoding="utf-8") as handle:
        samples = [json.loads(next(handle)) for _ in range(count)]
    if [sample.get("index") for sample in samples] != list(range(count)):
        raise ValueError("sample index/order does not match cached prediction order")
    archive = json.loads(archive_manifest.read_text(encoding="utf-8"))
    frontend = str(archive.get("frontend_name", "")).lower()
    if frontend not in {"motionbert", "poseformer", "videopose3d"}:
        raise ValueError(f"unsupported archived lifter: {frontend}")
    metadata = archive.get("metadata", {})
    source_metadata = metadata.get("source_metadata")
    if isinstance(source_metadata, dict):
        split_key = "val" if source["split"] == "validation" else str(source["split"])
        metadata = source_metadata.get(split_key, {})
    if metadata.get("input_2d_source") != "unity_gt_h36m17":
        raise ValueError("lifter archive must disclose Unity ground-truth 2D input")
    if metadata.get("output_joint_convention") != "canonfuse15":
        raise ValueError("lifter archive must use the documented 15-joint mapping")
    checkpoint_digest = str(metadata.get("estimator_checkpoint_sha256", ""))
    if len(checkpoint_digest) != 64:
        raise ValueError("lifter archive has no valid checkpoint checksum")
    entries = {}
    for entry in archive.get("entries", []):
        key = (str(entry["person_id"]), str(entry["action_id"]), str(entry["camera_id"]))
        if key in entries:
            raise ValueError(f"duplicate lifter stream: {key}")
        entries[key] = archive_manifest.parent / entry["pose_path"]
    output_root.mkdir(parents=True)
    samples_path = output_root / "samples.jsonl"
    samples_path.write_text("".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples), encoding="utf-8")
    method_names = [f"{frontend}_native_left", f"{frontend}_native_right"]
    arrays = []
    for method in method_names:
        method_root = output_root / method
        method_root.mkdir()
        arrays.append(np.lib.format.open_memmap(method_root / "predictions.npy", mode="w+", dtype=np.float32, shape=(count, 30, 13, 3)))
    loaded = {}
    source_hashes = {}
    for sample_index, (sample, frame_ids) in enumerate(zip(samples, frames)):
        for view_index, camera_key in enumerate(("cam1_id", "cam2_id")):
            key = (str(sample["person_id"]), str(sample["action_id"]), str(sample[camera_key]))
            if key not in entries:
                raise KeyError(f"missing lifter stream: {key}")
            path = entries[key].resolve()
            if path not in loaded:
                with np.load(path, allow_pickle=False) as bundle:
                    pose = np.asarray(bundle["pose"], dtype=np.float32)
                    stream_frames = np.asarray(bundle["frame_indices"], dtype=np.int64)
                if pose.ndim != 3 or pose.shape[1:] != (15, 3) or stream_frames.shape != (pose.shape[0],):
                    raise ValueError(f"invalid lifter archive shape: {path}")
                if not np.isfinite(pose).all() or len(np.unique(stream_frames)) != len(stream_frames):
                    raise ValueError(f"invalid lifter archive values or frames: {path}")
                loaded[path] = (pose[:, 2:15], {int(frame): index for index, frame in enumerate(stream_frames)})
                source_hashes[str(path)] = sha256(path)
            pose, frame_map = loaded[path]
            if any(int(frame) not in frame_map for frame in frame_ids):
                raise ValueError(f"lifter frame mismatch for {path}")
            arrays[view_index][sample_index] = pose[[frame_map[int(frame)] for frame in frame_ids]]
    for array in arrays:
        array.flush()
    del arrays
    config_digest = source.get("config_sha256") or sha256(cache / "manifest.json")
    code_digest = sha256(Path(__file__))
    methods = {}
    for method, frame in zip(method_names, ("left_camera_m", "right_camera_m")):
        prediction_path = output_root / method / "predictions.npy"
        manifest_path = output_root / method / "manifest.json"
        payload = freeze_prediction_manifest(PredictionManifest(method=method, dataset="unity",
            coordinate_frame=frame, prediction_path=str(prediction_path), sample_count=count,
            frame_count=30, joint_count=13, unit="m", source_checkpoint_sha256=checkpoint_digest,
            config_sha256=config_digest, code_sha256=code_digest, sample_index_path=str(samples_path)),
            manifest_path)
        methods[method] = {"status": "frozen", "manifest_path": str(manifest_path),
                           "prediction_sha256": payload["prediction_sha256"],
                           "coordinate_frame": frame}
    report = {"dataset": "unity", "split": source["split"], "frontend": frontend,
        "sample_count": count, "ground_truth_2d_used": True, "target_loaded": False,
        "archive_manifest": str(archive_manifest), "archive_manifest_sha256": sha256(archive_manifest),
        "source_prediction_sha256": source_hashes, "methods": methods,
        "view_groups": {f"{frontend}_native_view_pool": method_names}}
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("unity", "ski"), required=True)
    parser.add_argument("--split", choices=("train", "val", "validation", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("sam3d", "motionbert", "poseformer", "videopose3d", "canonfuse3d"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    validate_export_request(split=args.split, limit=args.limit)
    return args


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite native export root: {output_root}")
    output_root.mkdir(parents=True)
    baseline_root = Path("logs/ivc_mmsports_extension/main_baselines/20260906")
    split = "val" if args.split == "validation" else args.split
    cache = baseline_root / "cache" / args.dataset / split
    reports: dict[str, Any] = {}
    if "sam3d" in args.methods:
        reports["sam3d"] = export_cached_sam3d_and_controls(
            cache, output_root / "sam3d_controls", limit=args.limit
        )
    archives = {
        frontend: Path("logs/ivc_p1/frontend_adaptation/predictions")
        / frontend
        / f"{frontend}_all_manifest.json"
        for frontend in ("motionbert", "poseformer", "videopose3d")
    }
    for frontend in ("motionbert", "poseformer", "videopose3d"):
        if frontend not in args.methods:
            continue
        if args.dataset != "unity":
            reports[frontend] = {
                "status": "unavailable",
                "reason": "no frozen Ski native prediction archive exists for this lifting front end",
            }
        else:
            reports[frontend] = export_unity_lifter_windows(
                cache, archives[frontend], output_root / frontend, limit=args.limit
            )
    if "canonfuse3d" in args.methods:
        reports["canonfuse3d"] = {
            "status": "unavailable",
            "reason": (
                "the learned output gauge is not guaranteed to equal the invertible "
                "left-input gauge; see canonfuse_admission.json"
            ),
        }
    report = {
        "dataset": args.dataset,
        "split": split,
        "target_loaded": False,
        "requested_methods": args.methods,
        "reports": reports,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
