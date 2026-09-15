"""Prepare fixed references in independently documented native frames.

This is a scoring-stage utility: it is run only after prediction artifacts are
frozen.  Unity camera references use supplied extrinsics; Ski retains only its
constructed camera-local reference because no independent mapping to either
monocular estimator frame has been established.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dual2pose.eval.native_output_manifest import sha256


def _world_to_pinhole_camera(dataset_root: Path, person: str, camera: str) -> np.ndarray:
    camera_name = camera.removeprefix("capture_")
    path = dataset_root / "data_pole_ski" / person / "cameras" / camera_name / "extrinsics.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    world_to_unity_camera = np.asarray(payload["t_world_cam_4x4"], dtype=np.float64).reshape(4, 4)
    unity_to_pinhole = np.diag([1.0, -1.0, -1.0, 1.0])
    transform = unity_to_pinhole @ world_to_unity_camera
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        abs(determinant), 1.0, atol=1e-6
    ):
        raise ValueError(f"camera extrinsic is not a rigid axis transform: {path}")
    return transform


def prepare_targets(
    *,
    dataset: str,
    split: str,
    cache: Path,
    output_root: Path,
    dataset_root: Path | None = None,
    limit: int | None = None,
) -> dict:
    cache, output_root = cache.resolve(), output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite target descriptor root: {output_root}")
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset or manifest["split"] != split:
        raise ValueError("cache dataset/split mismatch")
    if limit is not None and (limit <= 0 or split == "test"):
        raise ValueError("a limit is allowed only for positive non-test preflights")
    for name in ("target.npy", "samples.jsonl"):
        if sha256(cache / name) != manifest["files"][name]:
            raise ValueError(f"cache checksum mismatch: {name}")
    total = int(manifest["sample_count"])
    count = total if limit is None else min(total, int(limit))
    joint_slice = slice(2, 15) if dataset == "unity" else slice(None)
    source = np.load(cache / "target.npy", mmap_mode="r", allow_pickle=False)
    target = np.ascontiguousarray(source[:count, :, joint_slice], dtype=np.float32)
    if target.shape != (count, 30, 13, 3) or not np.isfinite(target).all():
        raise ValueError("target cache has invalid native-output coverage")
    with (cache / "samples.jsonl").open(encoding="utf-8") as handle:
        samples = [json.loads(next(handle)) for _ in range(count)]
    if [sample.get("index") for sample in samples] != list(range(count)):
        raise ValueError("target sample order is not contiguous")
    output_root.mkdir(parents=True)
    samples_path = output_root / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples), encoding="utf-8"
    )
    targets = {}

    def save(frame: str, values: np.ndarray, provenance: dict):
        path = output_root / f"{frame}.npy"
        np.save(path, np.asarray(values, dtype=np.float32))
        targets[frame] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "shape": list(values.shape),
            "unit": "m",
            **provenance,
        }

    if dataset == "unity":
        if dataset_root is None:
            raise ValueError("Unity camera-frame targets require a dataset root")
        save(
            "calibrated_world_m",
            target,
            {"mapping": "identity from archived Unity world-coordinate target"},
        )
        for view, frame in ((1, "left_camera_m"), (2, "right_camera_m")):
            transformed = np.empty_like(target)
            source_hashes = {}
            for index, sample in enumerate(samples):
                camera = sample[f"cam{view}_id"]
                matrix = _world_to_pinhole_camera(dataset_root, sample["person_id"], camera)
                transformed[index] = target[index] @ matrix[:3, :3].T + matrix[:3, 3]
                path = (
                    dataset_root
                    / "data_pole_ski"
                    / sample["person_id"]
                    / "cameras"
                    / camera.removeprefix("capture_")
                    / "extrinsics.json"
                )
                source_hashes[str(path)] = sha256(path)
            save(
                frame,
                transformed,
                {
                    "mapping": "supplied Unity world-to-camera extrinsic followed by Unity-to-pinhole axis conversion",
                    "handedness_conversion": True,
                    "fit_to_prediction_or_target": False,
                    "extrinsic_sha256": source_hashes,
                },
            )
    elif dataset == "ski":
        save(
            "constructed_camera_local_m",
            target,
            {
                "mapping": "identity from archived constructed pseudo-GT",
                "limitation": "no independently supplied mapping to estimator left/right camera frames or calibrated DLT world",
            },
        )
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    descriptor = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "sample_count": count,
        "frame_count": 30,
        "joint_count": 13,
        "sample_index_path": str(samples_path.resolve()),
        "sample_index_sha256": sha256(samples_path),
        "source_cache_manifest": str(manifest_path),
        "source_cache_manifest_sha256": sha256(manifest_path),
        "targets": targets,
    }
    descriptor_path = output_root / "target_descriptor.json"
    descriptor_path.open("x", encoding="utf-8").write(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
    )
    return descriptor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("unity", "ski"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare_targets(**vars(args)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
