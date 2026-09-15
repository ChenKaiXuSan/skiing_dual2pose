"""Export a frozen MetaPose split in its restored left-camera coordinates.

This helper is intended for non-test preflight splits.  It consumes only the
prepared estimated-2D task arrays and already selected stage checkpoints; no
3D label array is accepted or opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dual2pose.eval.metapose_inputs import finite, point_mixtures, restore_left_gauge, sha256
from dual2pose.experiments.run_metapose import OfficialMetaPose


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def export_split(run_dir: Path, data_dir: Path, split: str, limit: int, batch_size: int) -> dict:
    if split == "test":
        raise ValueError("use the already frozen full test artifact")
    if limit <= 0 or batch_size <= 0:
        raise ValueError("limit and batch size must be positive")
    run_dir, data_dir = run_dir.resolve(), data_dir.resolve()
    output = run_dir / f"{split}_metapose_raw_left.npy"
    sidecar = run_dir / f"{split}_native_export.json"
    if output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing MetaPose native export")
    config = _json(run_dir / "config.json")
    report = _json(run_dir / "report.json")
    manifest = _json(data_dir / "manifest.json")
    if manifest["split"] != split or manifest["ground_truth_3d_loaded"]:
        raise ValueError("prepared split is not a target-free matching role")
    arrays = {}
    for name in ("uv", "q", "t", "native_left"):
        path = data_dir / f"{name}.npy"
        if sha256(path) != manifest["files"][path.name]:
            raise ValueError(f"prepared checksum mismatch: {name}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    x_path = run_dir / f"{split}_x_init.npy"
    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    frame_count = min(int(limit) * 30, len(x))
    if frame_count % 30 or any(len(value) != len(x) for value in arrays.values()):
        raise ValueError("prepared MetaPose frame coverage mismatch")
    uncertainty = _json(run_dir / "uncertainty.json")
    model = OfficialMetaPose(config["metapose_repo"], int(config["seed"]))
    stage_hashes = []
    for index, selection in enumerate(report["selected_stages"], 1):
        checkpoint = run_dir / f"stage_{index}.weights.h5"
        if sha256(checkpoint) != selection["checkpoint_sha256"]:
            raise ValueError(f"selected stage checksum mismatch: {checkpoint}")
        model.add_stage()
        model.load_stage(checkpoint)
        stage_hashes.append(selection["checkpoint_sha256"])
    prediction = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(frame_count // 30, 30, 13, 3)
    )
    flat = prediction.reshape(-1, 13, 3)
    for start in range(0, frame_count, batch_size):
        stop = min(start + batch_size, frame_count)
        task = point_mixtures(arrays["uv"][start:stop], uncertainty).reshape(stop - start, -1)
        state = model.predict(x[start:stop], task)
        projected = model.project(state)[:, 0]
        flat[start:stop] = restore_left_gauge(
            projected,
            arrays["q"][start:stop, 0],
            arrays["t"][start:stop, 0],
            arrays["native_left"][start:stop],
        )
    prediction.flush()
    finite(prediction, "complete restored MetaPose prediction")
    record = {
        "status": "frozen_non_test_preflight",
        "dataset": manifest["dataset"],
        "split": split,
        "sample_count": frame_count // 30,
        "frame_count": frame_count,
        "coordinate_frame": "left_camera_m",
        "prediction_path": str(output),
        "prediction_sha256": sha256(output),
        "prepared_manifest_sha256": sha256(data_dir / "manifest.json"),
        "x_init_sha256": sha256(x_path),
        "selected_stage_sha256": stage_hashes,
        "ground_truth_3d_loaded": False,
    }
    sidecar.open("x", encoding="utf-8").write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(export_split(**vars(args)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
