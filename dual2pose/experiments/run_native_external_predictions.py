"""Freeze external-method predictions before body-coordinate normalization.

The export stage is target-free.  Temporal refiners are run independently on
the two camera-native streams; geometry arrays keep their calibrated world
frame; MetaPose keeps its restored left-camera output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

import numpy as np
import torch

from dual2pose.eval.native_output_manifest import (
    PredictionManifest,
    freeze_prediction_manifest,
    sha256,
)


FORBIDDEN_INPUT_LABELS = {"canonical_average", "left_canonical", "gt_canonical"}
DEFAULT_BASELINE_ROOT = Path("logs/ivc_mmsports_extension/main_baselines/20260906")
DEFAULT_EXTERNAL_ROOT = Path("logs/ivc_mmsports_extension/external_baselines/20260911")
DEFAULT_STRIDE_REPO = Path("/tmp/ivc-table-citations-20260910-hc8JTg/stride_source")
DEFAULT_STRIDE_CHECKPOINT = Path("/tmp/ivc-full-preflight-20260910-UDRYsD/stride_latest_epoch.bin")


def validate_export_request(*, split: str, limit: int | None) -> None:
    if split not in {"train", "val", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if split == "test" and limit is not None:
        raise ValueError("limit is forbidden for test exports")


def _pose(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 4 or array.shape[1:] != (30, 13, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape (N,30,13,3)")
    return array


def export_two_view_refiner(
    method: str,
    left: np.ndarray,
    right: np.ndarray,
    runner: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Call a refiner once per independent native-view collection."""
    if method not in {"stride", "deciwatch"}:
        raise ValueError(f"unsupported two-view refiner: {method}")
    left_value = _pose(left, "left")
    right_value = _pose(right, "right")
    if left_value.shape != right_value.shape:
        raise ValueError("left/right shape mismatch")
    outputs = []
    for value in (left_value, right_value):
        prediction = _pose(runner(np.array(value, copy=True)), f"{method} prediction")
        if prediction.shape != value.shape:
            raise ValueError(f"{method} changed sequence coverage")
        outputs.append(prediction)
    return outputs[0], outputs[1]


def select_stride_views(
    left: np.ndarray, right: np.ndarray, selection: str
) -> list[tuple[str, np.ndarray]]:
    """Select native camera streams for independently runnable STRIDE shards."""
    if selection not in {"left", "right", "both"}:
        raise ValueError(f"unsupported stride view: {selection}")
    left_value = np.asarray(left, dtype=np.float32)
    right_value = np.asarray(right, dtype=np.float32)
    if (
        left_value.ndim != 4
        or left_value.shape[1:] != (30, 17, 3)
        or right_value.shape != left_value.shape
        or not np.isfinite(left_value).all()
        or not np.isfinite(right_value).all()
    ):
        raise ValueError("stride views must have matching finite shape (N,30,17,3)")
    views = [("left", left_value), ("right", right_value)]
    return views if selection == "both" else [views[0 if selection == "left" else 1]]


def export_external(
    method: str, prediction: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, str]]:
    """Validate an already computed raw output; there is deliberately no label input."""
    label = str(config.get("input_label", ""))
    if label in FORBIDDEN_INPUT_LABELS or "canonical" in label:
        raise ValueError("canonical inputs are forbidden for external native export")
    value = _pose(prediction, f"{method} prediction")
    frames = {
        "metapose": "left_camera_m",
        "dlt": "calibrated_world_m",
        "gated_dlt": "calibrated_world_m",
        "dlt_residual": "unadmitted_model_space",
    }
    if method not in frames:
        raise ValueError(f"unsupported external method: {method}")
    return value, {"coordinate_frame": frames[method], "input_label": label}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_run_status(output_root: Path, **fields: Any) -> Path:
    """Atomically persist a heartbeat that survives interrupted CLI sessions."""
    path = output_root / "run_status.json"
    temporary = output_root / "run_status.json.tmp"
    payload = dict(fields)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _combined_digest(*values: str) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _load_cache(cache: Path, limit: int | None):
    manifest_path = cache / "manifest.json"
    manifest = _read_json(manifest_path)
    validate_export_request(split=str(manifest["split"]), limit=limit)
    if manifest.get("representation") != "raw_pre_batch_canonicalization":
        raise ValueError("external native export requires the raw cache")
    for name in ("left.npy", "right.npy", "samples.jsonl"):
        if sha256(cache / name) != manifest["files"][name]:
            raise ValueError(f"cache checksum mismatch: {name}")
    total = int(manifest["sample_count"])
    count = total if limit is None else min(total, int(limit))
    joint_slice = slice(2, 15) if manifest["dataset"] == "unity" else slice(None)
    left_source = np.load(cache / "left.npy", mmap_mode="r", allow_pickle=False)
    right_source = np.load(cache / "right.npy", mmap_mode="r", allow_pickle=False)
    left = np.ascontiguousarray(left_source[:count, :, joint_slice], dtype=np.float32)
    right = np.ascontiguousarray(right_source[:count, :, joint_slice], dtype=np.float32)
    _pose(left, "left cache")
    _pose(right, "right cache")
    with (cache / "samples.jsonl").open(encoding="utf-8") as handle:
        samples = [json.loads(next(handle)) for _ in range(count)]
    if [sample.get("index") for sample in samples] != list(range(count)):
        raise ValueError("cache sample ordering is not contiguous")
    return left, right, samples, manifest, sha256(manifest_path)


def _freeze(
    output_root: Path,
    *,
    method: str,
    dataset: str,
    prediction: np.ndarray,
    coordinate_frame: str,
    samples_path: Path,
    checkpoint_digest: str,
    config_digest: str,
) -> dict[str, Any]:
    method_root = output_root / method
    method_root.mkdir()
    prediction_path = method_root / "predictions.npy"
    np.save(prediction_path, np.asarray(prediction, dtype=np.float32))
    manifest_path = method_root / "manifest.json"
    payload = freeze_prediction_manifest(
        PredictionManifest(
            method=method,
            dataset=dataset,
            coordinate_frame=coordinate_frame,
            prediction_path=str(prediction_path.resolve()),
            sample_count=int(prediction.shape[0]),
            frame_count=30,
            joint_count=13,
            unit="m",
            source_checkpoint_sha256=checkpoint_digest,
            config_sha256=config_digest,
            code_sha256=sha256(Path(__file__)),
            sample_index_path=str(samples_path.resolve()),
        ),
        manifest_path,
    )
    return {
        "status": "frozen",
        "manifest_path": str(manifest_path.resolve()),
        "prediction_sha256": payload["prediction_sha256"],
        "coordinate_frame": coordinate_frame,
    }


def collect_unique_frame_jobs(
    directories: list[list[str]], frames: np.ndarray
) -> list[tuple[str, int]]:
    frame_ids = np.asarray(frames)
    if frame_ids.ndim != 2 or len(directories) != len(frame_ids):
        raise ValueError("stride directory/frame coverage mismatch")
    return sorted(
        {
            (str(directory), int(frame))
            for row, sequence in zip(directories, frame_ids)
            for directory in row
            for frame in sequence
        }
    )


def _native_stride_inputs(
    cache: Path,
    dataset_root: Path,
    left: np.ndarray,
    right: np.ndarray,
    samples: list[dict[str, Any]],
    *,
    selection: str,
    workers: int,
) -> list[tuple[str, np.ndarray]]:
    """Recover native H36M17 streams with each raw frame read exactly once."""
    from dual2pose.eval.main_baseline_geometry import (
        _load_index,
        _match_source_samples,
        _resolve_dataset_path,
    )
    from dual2pose.eval.stride_external import assemble_h36m17
    from dual2pose.experiments.run_stride_external import _read_native_frame

    if selection not in {"left", "right", "both"} or workers < 1:
        raise ValueError("invalid STRIDE view selection or worker count")
    selected = [("left", 0), ("right", 1)]
    if selection != "both":
        selected = [selected[0 if selection == "left" else 1]]

    manifest = _read_json(cache / "manifest.json")
    frames = np.load(cache / "frames.npy", mmap_mode="r", allow_pickle=False)[: len(samples)]
    index = Path(manifest["index"])
    sources = _match_source_samples(manifest["dataset"], samples, _load_index(index, manifest["split"]))
    outputs = {
        name: np.empty((len(samples), 30, 17, 3), np.float32)
        for name, _ in selected
    }
    directories: list[list[str]] = []
    for sample, source in zip(samples, sources):
        row = []
        for _, view in selected:
            if manifest["dataset"] == "unity":
                directory = (
                    dataset_root
                    / "sam3d_body_results/inference"
                    / sample["person_id"]
                    / sample["action_id"]
                    / "frames"
                    / sample[f"cam{view + 1}_id"]
                )
            else:
                directory = _resolve_dataset_path(
                    source[f"cam{view + 1}_sam3d_kpt3d_dir"], dataset_root
                )
            row.append(str(directory))
        directories.append(row)

    raw: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    jobs = collect_unique_frame_jobs(directories, frames)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for directory, frame, points, bones, _ in pool.map(_read_native_frame, jobs):
            raw[(directory, frame)] = (points, bones)

    native_common = {"left": left, "right": right}
    for sample_index, (row, frame_ids) in enumerate(zip(directories, frames)):
        for column, (name, _) in enumerate(selected):
            sequence = [raw[(row[column], int(frame))] for frame in frame_ids]
            points = np.stack([item[0] for item in sequence])[:, 2:15]
            common = native_common[name][sample_index]
            if not np.allclose(points, common, rtol=0, atol=1e-6):
                raise ValueError(f"raw/cache mismatch at sample {sample_index}, {name}")
            bones = np.stack([item[1] for item in sequence])
            outputs[name][sample_index] = assemble_h36m17(common, bones)
    return [(name, outputs[name]) for name, _ in selected]


def _run_stride(
    native_views: list[tuple[str, np.ndarray]],
    *,
    repo: Path,
    checkpoint: Path,
    device: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 100,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from dual2pose.eval.stride_external import COMMON, load_official_stride

    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    adapter, model_info = load_official_stride(repo, checkpoint, device)
    outputs: dict[str, np.ndarray] = {}
    for view, native in native_views:
        prediction = np.empty((len(native), 30, 13, 3), np.float32)
        for index in range(len(native)):
            refined, _ = adapter.refine(native[index : index + 1])
            prediction[index] = refined.numpy()[0][:, COMMON]
            completed = index + 1
            if on_progress is not None and (
                completed % progress_every == 0 or completed == len(native)
            ):
                on_progress(
                    {
                        "view": view,
                        "completed_windows": completed,
                        "total_windows": len(native),
                    }
                )
        outputs[view] = prediction
    model_info["source_commit"] = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    model_info["native_view_selection"] = [view for view, _ in native_views]
    return outputs, model_info


def _run_deciwatch(
    left: np.ndarray,
    right: np.ndarray,
    *,
    run_dir: Path,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from dual2pose.experiments.run_deciwatch import DeciWatchWindow

    config = _read_json(run_dir / "config.json")
    model = DeciWatchWindow(
        config["deciwatch_repo"],
        39,
        interval=int(config["interval"]),
        hidden=int(config["hidden"]),
        layers=int(config["layers"]),
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    def refine(collection: np.ndarray) -> np.ndarray:
        output = np.empty_like(collection)
        with torch.inference_mode():
            for start in range(0, len(collection), batch_size):
                stop = min(start + batch_size, len(collection))
                inputs = torch.from_numpy(collection[start:stop].reshape(stop - start, 30, 39)).to(device)
                recovered, _ = model(inputs, device)
                output[start:stop] = recovered.cpu().numpy().reshape(stop - start, 30, 13, 3)
        return output

    left_prediction, right_prediction = export_two_view_refiner("deciwatch", left, right, refine)
    info = {
        "checkpoint_sha256": sha256(run_dir / "best.pth"),
        "source_commit": config["source_commit"],
        "official_reference_config": config["official_reference_config"],
        "observation_indices": [0, 10, 20, 29],
        "boundary_policy": config["boundary_policy"],
        "input_policy": "each native camera view independently",
    }
    return left_prediction, right_prediction, info


def _load_metapose(run_dir: Path, split: str, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    report = _read_json(run_dir / "report.json")
    if split == "test":
        path = run_dir / "metapose_raw_left.npy"
        expected = report.get("predictions", {}).get(path.name)
    else:
        path = run_dir / f"{split}_metapose_raw_left.npy"
        sidecar = run_dir / f"{split}_native_export.json"
        expected = _read_json(sidecar).get("prediction_sha256") if sidecar.exists() else None
    if not path.exists() or expected is None or sha256(path) != expected:
        raise RuntimeError(f"no verified restored MetaPose output for split {split}")
    prediction = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False)[:count], dtype=np.float32)
    stages = [entry["checkpoint_sha256"] for entry in report["selected_stages"]]
    return prediction, {"checkpoint_sha256": _combined_digest(*stages), "source": str(path)}


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    split = "val" if args.split == "validation" else args.split
    cache = args.baseline_root / "cache" / args.dataset / split
    left, right, samples, cache_manifest, cache_digest = _load_cache(cache, args.limit)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite native export root: {output}")
    output.mkdir(parents=True)
    started_at = time.monotonic()
    write_run_status(
        output,
        status="running",
        phase="initializing",
        dataset=args.dataset,
        split=split,
        process_id=os.getpid(),
        completed_windows=0,
        total_windows=len(samples),
    )
    samples_path = output / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "dataset": args.dataset,
        "split": split,
        "sample_count": len(samples),
        "target_loaded": False,
        "cache_manifest_sha256": cache_digest,
        "methods": {},
        "view_groups": {},
    }

    def freeze_views(prefix: str, values, checkpoint_digest: str, config_digest: str):
        names = [f"{prefix}_native_left", f"{prefix}_native_right"]
        for name, prediction, frame in zip(names, values, ("left_camera_m", "right_camera_m")):
            report["methods"][name] = _freeze(
                output,
                method=name,
                dataset=args.dataset,
                prediction=prediction,
                coordinate_frame=frame,
                samples_path=samples_path,
                checkpoint_digest=checkpoint_digest,
                config_digest=config_digest,
            )
        report["view_groups"][f"{prefix}_native_view_pool"] = names

    if "stride" in args.methods:
        write_run_status(
            output,
            status="running",
            phase="stride_input_assembly",
            dataset=args.dataset,
            split=split,
            process_id=os.getpid(),
            view=args.stride_view,
            completed_windows=0,
            total_windows=len(samples),
        )
        native_views = _native_stride_inputs(
            cache,
            args.dataset_root,
            left,
            right,
            samples,
            selection=args.stride_view,
            workers=args.workers,
        )

        def record_stride_progress(event: dict[str, Any]) -> None:
            write_run_status(
                output,
                status="running",
                phase="stride_inference",
                dataset=args.dataset,
                split=split,
                process_id=os.getpid(),
                elapsed_seconds=round(time.monotonic() - started_at, 3),
                **event,
            )

        stride_predictions, info = _run_stride(
            native_views,
            repo=args.stride_repo,
            checkpoint=args.stride_checkpoint,
            device=args.device,
            on_progress=record_stride_progress,
            progress_every=args.progress_every,
        )
        stride_methods = []
        for view, prediction in stride_predictions.items():
            method = f"stride_native_{view}"
            report["methods"][method] = _freeze(
                output,
                method=method,
                dataset=args.dataset,
                prediction=prediction,
                coordinate_frame=f"{view}_camera_m",
                samples_path=samples_path,
                checkpoint_digest=info["checkpoint_sha256"],
                config_digest=cache_digest,
            )
            stride_methods.append(method)
        report["view_groups"]["stride_native_view_pool"] = stride_methods
        report["stride"] = info

    if "deciwatch" in args.methods:
        run_dir = args.external_root / f"deciwatch_{args.dataset}_official_protocol_full_seed4321"
        deci_left, deci_right, info = _run_deciwatch(
            left, right, run_dir=run_dir, device=args.device, batch_size=args.batch_size
        )
        freeze_views("deciwatch", (deci_left, deci_right), info["checkpoint_sha256"], cache_digest)
        report["deciwatch"] = info

    if "metapose" in args.methods:
        run_dir = args.external_root / f"metapose_{args.dataset}_full_seed42"
        prediction, info = _load_metapose(run_dir, split, len(samples))
        prediction, descriptor = export_external(
            "metapose", prediction, {"dataset": args.dataset, "input_label": "raw_restored_left"}
        )
        report["methods"]["metapose_native"] = _freeze(
            output,
            method="metapose_native",
            dataset=args.dataset,
            prediction=prediction,
            coordinate_frame=descriptor["coordinate_frame"],
            samples_path=samples_path,
            checkpoint_digest=info["checkpoint_sha256"],
            config_digest=sha256(run_dir / "config.json"),
        )
        report["metapose"] = info

    geometry = args.baseline_root / "geometry" / args.dataset / split
    geometry_manifest_path = geometry / "manifest.json"
    geometry_manifest = _read_json(geometry_manifest_path)
    geometry_digest = sha256(geometry_manifest_path)
    geometry_slice = slice(2, 15) if args.dataset == "unity" else slice(None)
    for request, filename, exported in (
        ("dlt", "dlt.npy", "calibrated_dlt_native"),
        ("gated_dlt", "robust_dlt.npy", "reprojection_gated_dlt_native"),
    ):
        if request not in args.methods:
            continue
        path = geometry / filename
        expected = geometry_manifest["output_sha256"][filename]
        if sha256(path) != expected:
            raise ValueError(f"geometry checksum mismatch: {filename}")
        prediction = np.ascontiguousarray(
            np.load(path, mmap_mode="r", allow_pickle=False)[: len(samples), :, geometry_slice],
            dtype=np.float32,
        )
        prediction, descriptor = export_external(
            request, prediction, {"dataset": args.dataset, "input_label": "calibrated_world"}
        )
        report["methods"][exported] = _freeze(
            output,
            method=exported,
            dataset=args.dataset,
            prediction=prediction,
            coordinate_frame=descriptor["coordinate_frame"],
            samples_path=samples_path,
            checkpoint_digest=geometry_digest,
            config_digest=geometry_digest,
        )

    if "dlt_residual" in args.methods:
        report["methods"]["dlt_residual_mlp_native"] = {
            "status": "unavailable",
            "reason": (
                "the existing DLT-residual checkpoint consumes body-normalized DLT and is "
                "supervised in target-normalized coordinates; no calibrated-world output "
                "gauge is established by its model contract"
            ),
        }

    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_run_status(
        output,
        status="completed",
        phase="frozen",
        dataset=args.dataset,
        split=split,
        process_id=os.getpid(),
        elapsed_seconds=round(time.monotonic() - started_at, 3),
        completed_windows=len(samples),
        total_windows=len(samples),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("unity", "ski"), required=True)
    parser.add_argument("--split", choices=("train", "val", "validation", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("stride", "deciwatch", "metapose", "dlt", "gated_dlt", "dlt_residual"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--stride-repo", type=Path, default=DEFAULT_STRIDE_REPO)
    parser.add_argument("--stride-checkpoint", type=Path, default=DEFAULT_STRIDE_CHECKPOINT)
    parser.add_argument("--stride-view", choices=("left", "right", "both"), default="both")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    validate_export_request(split=args.split, limit=args.limit)
    if args.batch_size <= 0 or args.workers <= 0 or args.progress_every <= 0:
        parser.error("batch size, worker count, and progress interval must be positive")
    if args.dataset_root is None:
        args.dataset_root = Path(
            "/home/kaixu_chen/skiing/data/skiing_unity_dataset"
            if args.dataset == "unity"
            else "/home/kaixu_chen/skiing/data/Ski-PosePTZ-CameraDataset-png"
        )
    return args


def main() -> None:
    args = parse_args()
    try:
        report = run_export(args)
    except BaseException as error:
        output = args.output_root.resolve()
        if output.exists():
            try:
                write_run_status(
                    output,
                    status="failed",
                    phase="failed",
                    dataset=args.dataset,
                    split=args.split,
                    process_id=os.getpid(),
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            except Exception:
                pass
        raise
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
